#!/usr/bin/env Rscript
# TraMineR reference for the OM dissimilarity of `om`.
#
#   Rscript validation/tramineR_reference.R
#
# Reads cases.json + sequences.csv (written by make_sequences.py), writes reference.json.
# Nothing is computed here that TraMineR does not compute itself.
#
# What is reproduced, against what:
#
#   substitution  seqcost(method="TRATE", cval=2)     == compute_trate_subst_matrix
#                   i.e. 2 - P - t(P) with zero diagonal, P = seqtrate() = transition
#                   counts row-normalised, a row of zeros where the state is never
#                   the source of a transition
#   indel         0.5 * max(sm), or the value given  == delta, constant over states
#   distance      seqdist(method="OM", norm="none")   == om_distance
#
# `norm="none"` is the raw d_OM. The paper's normalisation is d_OM / n, applied by
# `gamma_hat_pairs` / `om_matrices` after the fact and identical on both sides at a
# fixed horizon, so there is nothing for TraMineR to say about it.

suppressPackageStartupMessages({library(TraMineR); library(jsonlite)})

HERE <- (function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f)) else normalizePath(".")
})()

cases <- fromJSON(file.path(HERE, "cases.json"), simplifyVector = FALSE)
cells <- read.csv(file.path(HERE, "sequences.csv"),
                  colClasses = c("character", "integer", "integer", "character"))

out <- list()
for (name in names(cases)) {
  meta <- cases[[name]]
  N <- meta$n_sequences; n <- meta$seq_len
  alphabet <- unlist(meta$alphabet)
  cat(sprintf("\n=== %s (N=%d, n=%d, d=%d) ===\n", name, N, n, meta$n_states))

  rows <- cells[cells$case == name, ]
  m <- matrix(NA_character_, nrow = N, ncol = n)
  m[cbind(rows$seq + 1L, rows$t + 1L)] <- rows$state
  stopifnot(!any(is.na(m)))
  s <- suppressMessages(seqdef(m, alphabet = alphabet))

  # Estimated once per case: the two TRATE schemes differ only by their indel.
  trate <- suppressMessages(seqcost(s, method = "TRATE", cval = 2))$sm
  trate <- matrix(as.numeric(trate), nrow = length(alphabet))

  schemes <- list()
  for (scheme in names(meta$schemes)) {
    spec <- meta$schemes[[scheme]]
    sm <- if (spec$source == "TRATE") trate else
      matrix(unlist(lapply(spec$sub, unlist)), nrow = length(alphabet), byrow = TRUE)
    indel <- if (identical(spec$indel, "half_max")) 0.5 * max(sm) else as.numeric(spec$indel)

    d <- suppressMessages(seqdist(s, method = "OM", sm = sm, indel = indel,
                                  norm = "none", full.matrix = TRUE))
    cat(sprintf("  %-14s indel = %.17g   max dist = %.17g\n", scheme, indel, max(d)))
    schemes[[scheme]] <- list(sub = sm, indel = indel, dist = d)
  }
  out[[name]] <- schemes
}

# digits = 17 and not the `digits = NA` that looks like "everything": NA writes 15
# significant digits and truncates the last bits, which alone caps agreement at 5e-15.
writeLines(toJSON(out, digits = 17, matrix = "rowmajor", na = "null"),
           file.path(HERE, "reference.json"))
cat(sprintf("\n-> %s\n", file.path(HERE, "reference.json")))
