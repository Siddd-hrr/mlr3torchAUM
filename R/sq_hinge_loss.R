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
  ## Sort positives by score to use cumsum for the sums above.
  pos_scores = pred[torch::torch_tensor(which(is_pos_r), dtype=torch::torch_long())]
  neg_scores = pred[torch::torch_tensor(which(is_neg_r), dtype=torch::torch_long())]
  ## Sort ascending by score.
  pos_sorted = pos_scores[torch::torch_argsort(pos_scores$detach())]
  n_pos = pos_sorted$shape[1]
  ## z = (margin - score) for positives (0 for negatives).
  ## Quadratic coefficients: a=1, b=2*z, c=z^2 per positive.
  pos_sorted_d = pos_sorted$to(dtype=torch::torch_double())
  neg_scores_d = neg_scores$to(dtype=torch::torch_double())
  pos_cumsum1 = torch::torch_cumsum(pos_sorted_d, dim=1)                  #sum of pos scores
  pos_cumsum2 = torch::torch_cumsum(pos_sorted_d*pos_sorted_d, dim=1)     #sum of pos scores^2
  ## For each negative, find how many positives have score < neg + margin.
  ## threshold = neg + margin; count = searchsorted result.
  thresholds = neg_scores_d$detach() + margin
  k_idx = torch::torch_searchsorted(
    pos_sorted_d$detach()$contiguous(),
    thresholds$contiguous(),
    right=FALSE)
  zero_scalar = torch::torch_zeros(1L, dtype=torch::torch_double(), device=pred$device)
  k_float = k_idx$to(dtype=torch::torch_double())
  ## Shift left by one: each negative uses only positives ranked below it.
  k_clamped = torch::torch_clamp(k_idx, min=1L)
  sum_pos = torch::torch_where(
    k_idx == 0L,
    zero_scalar$expand_as(neg_scores_d),
    pos_cumsum1[k_clamped])
  sum_sq = torch::torch_where(
    k_idx == 0L,
    zero_scalar$expand_as(neg_scores_d),
    pos_cumsum2[k_clamped])
  neg2 = neg_scores_d*neg_scores_d
  ## Sum of contributions from negatives only (positives are masked to zero).
  contrib = neg2*k_float +
    neg_scores_d*2*(k_float*margin - sum_pos) +
    (k_float*margin*margin - 2*margin*sum_pos + sum_sq)
  torch::torch_sum(contrib)
}