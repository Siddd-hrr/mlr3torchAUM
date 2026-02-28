library(testthat)

## Used to verify the O(N log N) cumsum result is correct by taking Naive O(N^2) algorithm reference.
all_pairs_sq_hinge_naive = function(pred, label, margin=1){
  label = as.integer(label)
  pos_idx = which(label == 1L)
  neg_idx = which(label == -1L)
  if(length(pos_idx)==0L || length(neg_idx)==0L){
    return(torch::torch_sum(pred*0))
  }
  pred_d = pred$to(dtype=torch::torch_double())
  total = torch::torch_sum(pred_d*0)
  for(j in pos_idx){
    for(k in neg_idx){
      diff = pred_d[j] - pred_d[k]
      hinge = torch::torch_clamp(margin - diff, min=0)
      total = total + hinge*hinge
    }
  }
  total$to(dtype=pred$dtype)
}

if(torch::torch_is_installed()){

  test_that("loss is zero when positive is greater than negative by more than margin", {
    pred = torch::torch_tensor(c(2.0, 0.0))
    label = c(1L, -1L)
    expect_equal(
      torch::as_array(all_pairs_squared_hinge_loss(pred, label)),
      0.0)
  })

  test_that("When both scores are equal (zero gap)", {
    pred = torch::torch_tensor(c(0.0, 0.0))
    label = c(1L, -1L)
    expect_equal(
      torch::as_array(all_pairs_squared_hinge_loss(pred, label)),
      torch::as_array(all_pairs_sq_hinge_naive(pred, label)))
  })

  test_that("loss should be equal to naive O(N^2) algorithm on two positives & two negatives", {
    pred = torch::torch_tensor(c(0.3, 0.8, 0.6, 0.1))
    label = c(1L, 1L, -1L, -1L)
    expect_equal(
      torch::as_array(all_pairs_squared_hinge_loss(pred, label)),
      torch::as_array(all_pairs_sq_hinge_naive(pred, label)))
  })

  test_that("loss should be equal to naive O(N^2) algorithm on random predictions", {
    set.seed(1)
    for(trial in seq_len(5)){
      n = sample(4:10, 1)
      pred = torch::torch_tensor(rnorm(n))
      label = c(1L, -1L, sample(c(-1L, 1L), n-2, replace=TRUE))
      expect_equal(
        torch::as_array(all_pairs_squared_hinge_loss(pred, label)),
        torch::as_array(all_pairs_sq_hinge_naive(pred, label)))
    }
  })

  test_that("loss should be equal to naive O(N^2) algorithm with margin zero", {
    pred = torch::torch_tensor(c(0.5, -0.5, 0.2, -0.2))
    label = c(1L, 1L, -1L, -1L)
    expect_equal(
      torch::as_array(all_pairs_squared_hinge_loss(pred, label, margin=0)),
      torch::as_array(all_pairs_sq_hinge_naive(pred, label, margin=0)))
  })

  ## "If there are NO negatives or NO positives, there are ZERO pairs to compare,so loss must be zero."
  test_that("loss is zero when only one class present", {
    pred = torch::torch_tensor(c(1.0, 2.0, 3.0))
    expect_equal(
      torch::as_array(all_pairs_squared_hinge_loss(pred, c(1L, 1L, 1L))),
      0.0)
    expect_equal(
      torch::as_array(all_pairs_squared_hinge_loss(pred, c(-1L, -1L, -1L))),
      0.0)
  })

  test_that("gradient exists and is finite", {
    pred = torch::torch_tensor(c(0.0, 0.0), requires_grad=TRUE)
    label = c(1L, -1L)
    loss = all_pairs_squared_hinge_loss(pred, label)
    loss$backward()
    expect_true(!is.null(pred$grad))
    expect_true(all(is.finite(torch::as_array(pred$grad))))
  })

  test_that("gradient equal to naive gradient", {
    set.seed(2)
    pred_vals = rnorm(6)
    label = c(1L, -1L, 1L, -1L, 1L, -1L)
    pred_fast = torch::torch_tensor(pred_vals, requires_grad=TRUE)
    loss_fast = all_pairs_squared_hinge_loss(pred_fast, label)
    loss_fast$backward()
    pred_naive = torch::torch_tensor(pred_vals, requires_grad=TRUE)
    loss_naive = all_pairs_sq_hinge_naive(pred_naive, label)
    loss_naive$backward()
    expect_equal(
      torch::as_array(pred_fast$grad),
      torch::as_array(pred_naive$grad))
  })

}