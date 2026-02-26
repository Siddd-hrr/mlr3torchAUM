BatchSamplerBase <- torch::sampler(
  "BatchSamplerBase",
  .iter = function() {
    batch.i <- 0
    function() {
      if (batch.i < length(self$batch_list)) {
        batch.i <<- batch.i + 1L
        indices <- self$batch_list[[batch.i]]
        if (batch.i == length(self$batch_list)) {
          self$set_batch_list()
        }
        return(indices)
      }
      coro::exhausted()
    }
  },
  .length = function() {
    length(self$batch_list)
  }
)