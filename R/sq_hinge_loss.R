##' All-Pairs Squared Hinge Loss (O(N log N) via cumsum)
##'
##' Implements the all-pairs squared hinge loss from Rust & Hocking (2023)
##' https://arxiv.org/abs/2302.11062 using cumulative sums for O(N log N) time.
##' @export
all_pairs_squared_hinge_loss = function(pred, label, margin=1){
  label = as.integer(label)
  stopifnot(
    inherits(pred, "torch_tensor"),
    length(label) == pred$shape[1],
    all(label %in% c(-1L, 1L)))
  n = pred$shape[1]
  is_pos_r = (label == 1L)
  is_neg_r = (label == -1L)
  if(!any(is_pos_r) || !any(is_neg_r)){
    return(torch::torch_sum(pred*0))
  }
  ## Sort ascending by score.
  sorted_idx = torch::torch_argsort(pred$detach())
  pred_s = pred[sorted_idx]
  is_pos_s = torch::torch_tensor(
    as.numeric(is_pos_r[as.integer(sorted_idx)]),
    dtype=pred$dtype, device=pred$device)
  is_neg_s = 1 - is_pos_s
  ## z = (margin - score) for positives (0 for negatives).
  ## Quadratic coefficients: a=1, b=2*z, c=z^2 per positive.
  z = (margin - pred_s)*is_pos_s
  cum_a = torch::torch_cumsum(is_pos_s, dim=1)
  cum_b = torch::torch_cumsum(2*z, dim=1)
  cum_c = torch::torch_cumsum(z*z, dim=1)
  ## Shift left by one: each negative uses only positives ranked below it.
  zero = torch::torch_zeros(1L, dtype=pred$dtype, device=pred$device)
  a_prev = torch::torch_cat(c(zero, cum_a[1:(n-1)]))
  b_prev = torch::torch_cat(c(zero, cum_b[1:(n-1)]))
  c_prev = torch::torch_cat(c(zero, cum_c[1:(n-1)]))
  ## Sum of contributions from negatives only (positives are masked to zero).
  contrib = is_neg_s*(a_prev*pred_s*pred_s + b_prev*pred_s + c_prev)
  torch::torch_sum(contrib)
}