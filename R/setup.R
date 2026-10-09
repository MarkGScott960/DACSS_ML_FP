# FlyPredictor: one-time R setup (run once from the repository root).
#
#   Rscript R/setup.R
#
# Creates a project-local R library with renv, installs the pinned packages,
# and writes renv.lock (commit it). After this, any Rscript started in the
# repository root uses the pinned library automatically (via .Rprofile).
#
# Windows note: the ERT package contains C++ code (Rcpp), so installing it from
# GitHub needs Rtools matching your R version:
#   https://cran.r-project.org/bin/windows/Rtools/
# Check with:  pkgbuild::has_build_tools(debug = TRUE)

ERT_COMMIT <- "7fd51ce856a054582820bb0ac0ccae85371ff9be"   # must match config/project.yaml

if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv", repos = "https://cloud.r-project.org")

if (!file.exists("renv.lock")) {
  renv::init(bare = TRUE, restart = FALSE)
} else {
  renv::activate()
  renv::restore(prompt = FALSE)
}

pkgs <- c("remotes", "jsonlite", "nanoparquet", "dplyr", "tidyr", "stringr", "hablar", "plm", "Rcpp", "pkgbuild")
renv::install(pkgs, prompt = FALSE)

if (!pkgbuild::has_build_tools(debug = FALSE)) {
  stop("Build tools (Rtools on Windows) not found; ERT cannot be compiled. Install Rtools, then rerun.")
}
renv::install(paste0("vdeminstitute/ERT@", ERT_COMMIT), prompt = FALSE)

renv::snapshot(prompt = FALSE)

cat("\nR setup complete.\n")
cat("ERT version:   ", as.character(packageVersion("ERT")), "\n")
cat("nanoparquet:   ", as.character(packageVersion("nanoparquet")), "\n")
