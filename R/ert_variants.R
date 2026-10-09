# Re-run ERT::get_eps for (a) the real-time check and (b) threshold variants.
#
#   Rscript R/ert_variants.R <job.json>
#
# The job file is written by flypredictor.data.ert_jobs (Python). Fields:
#   mode:      "truncated" or "thresholds"
#   params:    default ERT parameters {start_incl, cum_incl, year_turn, cum_turn, tolerance}
#   years:     (truncated) truncation years T; V-Dem is cut at year <= T and get_eps is re-run,
#              which gives each country's episode status as it would have been coded at T
#   variants:  (thresholds) named list of parameter sets
#   output:    Parquet file to write
#
# Uses ERT::vdem, the exact V-Dem v16 data ERT was built from (same file as data/raw/ert/vdem.RData).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: Rscript R/ert_variants.R <job.json>")
job <- jsonlite::fromJSON(args[[1]], simplifyVector = TRUE)

suppressPackageStartupMessages({
  library(ERT)
  library(dplyr)
})

keep_cols <- c("country_id", "country_text_id", "year", "v2x_regime", "v2x_polyarchy", "reg_type",
               "aut_ep", "aut_ep_id", "aut_ep_start_year", "aut_ep_end_year", "aut_ep_prch",
               "aut_ep_pbr", "aut_ep_outcome", "aut_ep_censored")

run_eps <- function(data, p) {
  ERT::get_eps(data = data,
               start_incl = p$start_incl, cum_incl = p$cum_incl,
               year_turn = p$year_turn, cum_turn = p$cum_turn,
               tolerance = as.integer(p$tolerance))
}

vdem <- ERT::vdem
out <- NULL

if (job$mode == "truncated") {
  for (T in job$years) {
    t0 <- Sys.time()
    d <- vdem[vdem$year <= T, ]
    # Pretend coding ends at T, as it would have for a forecaster standing in year T.
    d$codingend <- pmin(d$codingend, T)
    eps <- run_eps(d, job$params)
    snap <- eps[eps$year == T, intersect(keep_cols, names(eps))]
    snap$trunc_year <- as.integer(T)
    out <- bind_rows(out, snap)
    cat(sprintf("truncated at %d: %d country-years, %d in an autocratization episode (%.0fs)\n",
                T, nrow(snap), sum(snap$aut_ep == 1, na.rm = TRUE),
                as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  }
} else if (job$mode == "thresholds") {
  for (nm in names(job$variants)) {
    t0 <- Sys.time()
    eps <- run_eps(vdem, job$variants[[nm]])
    eps <- eps[, intersect(keep_cols, names(eps))]
    eps$variant <- nm
    out <- bind_rows(out, eps)
    n_ep <- length(unique(na.omit(eps$aut_ep_id[eps$aut_ep == 1 & eps$aut_ep_prch == 1])))
    cat(sprintf("variant %s: %d democratic episodes (%.0fs)\n", nm, n_ep,
                as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  }
} else {
  stop("unknown mode: ", job$mode)
}

dir.create(dirname(job$output), recursive = TRUE, showWarnings = FALSE)
nanoparquet::write_parquet(as.data.frame(out), job$output)
cat("wrote", job$output, "\n")
