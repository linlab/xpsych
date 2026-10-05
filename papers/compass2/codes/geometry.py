"""COMPASS 2.0 numerical compatibility adapter to the xpsych package.

These functions preserve the audited paper implementation. New applications
should use the validated interfaces in xpsych.geometry and xpsych.scoring.
"""
from xpsych._geometry import (  # noqa: F401
    upper, cov_to_corr, sym_power, whiten, deconvolve, deconvolve_wording,
    center_by_group, split_half_reliability, partial_corr, mantel,
    keyed_scores, mantel_partial_fl,
)
