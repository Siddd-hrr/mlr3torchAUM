batch_sampler_random <- function(batch_size, shuffle=TRUE){
  .N <- `:=` <- i.in.stratum <- . <- max.i <- n.samp <- batch.i <- self <- NULL
  ## Above for CRAN check.
  torch::sampler(
    "RandomSampler",
    inherit = BatchSamplerBase,
    initialize = function(data_source) {
      self$N <- data_source$task$nrow
      self$batch_vec <- seq_len(self$N) %/% batch_size
      self$set_batch_list()
    },
    set_batch_list = function() {
      index_vec <- if(shuffle){
        if(torch::torch_is_installed()){
          torch::as_array(torch::torch_randperm(self$N))+1L
        }else{
          sample(self$N)
        }
      }else{
        seq_len(self$N)
      }
      self$batch_list <- split(index_vec, self$batch_vec)
    }
  )
}