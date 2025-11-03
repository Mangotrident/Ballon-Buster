"""Cohort-level metric utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from ..engine.qle import SimulationResult


@dataclass
class SummaryMetrics:
    ete_peak: float
    gamma_star: float
    tau_c_mean: float
    qls_mean: float
    resilience_index: float


def compute_summary_metrics(results: Sequence[SimulationResult]) -> SummaryMetrics:
    if not results:
        raise ValueError("results cannot be empty")
    etes = np.array([r.ete for r in results])
    gammas = np.array([r.parameters.get("gamma", np.nan) for r in results])
    qls_values = np.array([r.qls for r in results])
    tau_c = np.array([r.coherence_lifetime for r in results])
    resilience = np.array([r.parameters.get("resilience", np.nan) for r in results])

    idx = int(np.nanargmax(etes))
    ete_peak = float(etes[idx])
    gamma_star = float(gammas[idx])
    tau_c_mean = float(np.mean(tau_c))
    qls_mean = float(np.mean(qls_values))
    resilience_index = float(np.nanmean(resilience))

    return SummaryMetrics(
        ete_peak=ete_peak,
        gamma_star=gamma_star,
        tau_c_mean=tau_c_mean,
        qls_mean=qls_mean,
        resilience_index=resilience_index,
    )
