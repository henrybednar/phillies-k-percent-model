# Knits methodology.Rmd to methodology.pdf. Run after data_merge.R and
# k_percentage_model.R, from this folder:  Rscript render_methodology.R
#
# The knit runs in a short temporary directory because pdflatex on Windows
# can't open files whose full path exceeds ~260 characters. Needs pandoc:
# RStudio's bundled copy is used if pandoc isn't already on PATH.
# Developed with assistance from Claude (Anthropic), https://claude.ai

if (!rmarkdown::pandoc_available()) {
  rstudio_pandoc <- "C:/Program Files/RStudio/resources/app/bin/quarto/bin/tools"
  if (dir.exists(rstudio_pandoc)) Sys.setenv(RSTUDIO_PANDOC = rstudio_pandoc)
}
stopifnot(rmarkdown::pandoc_available())

src <- "methodology.Rmd"
inputs <- c(src, "k_2026_model_summary.rds", "k_2026_2025_predictions.csv", "k_2026_features.csv")
stopifnot(all(file.exists(inputs)), file.exists("experiments/design_decisions.csv"))

here <- getwd()
tmp <- file.path(tempdir(), "methodology")
dir.create(file.path(tmp, "experiments"), recursive = TRUE, showWarnings = FALSE)
file.copy(inputs, tmp, overwrite = TRUE)
file.copy("experiments/design_decisions.csv", file.path(tmp, "experiments"), overwrite = TRUE)

setwd(tmp)
out <- file.path(tmp, rmarkdown::render(src, quiet = TRUE) |> basename())
setwd(here)

dest <- "methodology.pdf"
stopifnot(file.copy(out, dest, overwrite = TRUE))
cat("Wrote", dest, "\n")
