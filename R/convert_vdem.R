# Convert the V-Dem .RData file (as bundled in ERT v16) to Parquet.
# Fallback for when Python's pyreadr is unavailable. Called by
# flypredictor.fetch.vdem_to_parquet via flypredictor.rbridge.run_r.
#
#   Rscript R/convert_vdem.R <input.RData> <output.parquet>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: Rscript R/convert_vdem.R <input.RData> <output.parquet>")
input  <- args[[1]]
output <- args[[2]]


env <- new.env()
objs <- load(input, envir = env)
if (length(objs) != 1) stop("expected exactly one object in ", input, ", found: ", paste(objs, collapse = ", "))
df <- get(objs[[1]], envir = env)

df$country_id <- as.integer(df$country_id)
df$year <- as.integer(df$year)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
nanoparquet::write_parquet(as.data.frame(df), output)

cat(sprintf("wrote %s: %d rows x %d columns, years %d-%d\n",
            output, nrow(df), ncol(df), min(df$year), max(df$year)))
