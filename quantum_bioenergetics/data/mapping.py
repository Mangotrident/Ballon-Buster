"""Utilities to map omics measurements to quantum simulation parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd


@dataclass
class MappingConfig:
    epsilon_0: float = 0.2
    alpha: float = 0.015
    j_0: float = 0.02
    j_max: float = 0.05
    gamma_min: float = 0.0
    gamma_max: float = 0.05


class ParameterMapper:
    def __init__(self, config: MappingConfig, node_genes: Mapping[str, Iterable[str]]):
        self.config = config
        self.node_genes = node_genes

    def _aggregate_expression(self, expression: pd.Series, genes: Iterable[str]) -> float:
        values = [expression.get(gene, np.nan) for gene in genes]
        values = [v for v in values if not np.isnan(v)]
        if not values:
            return 0.0
        return float(np.mean(values))

    def map_expression(self, expression: Mapping[str, float]) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        series = pd.Series(expression, dtype=float)
        z_scores = (series - series.mean()) / (series.std(ddof=0) + 1e-8)
        site_energies: Dict[str, float] = {}
        couplings: Dict[Tuple[str, str], float] = {}
        for node, genes in self.node_genes.items():
            z_value = self._aggregate_expression(z_scores, genes)
            site_energies[node] = self.config.epsilon_0 - self.config.alpha * z_value

        nodes = list(self.node_genes)
        for idx, src in enumerate(nodes):
            for dst in nodes[idx + 1 :]:
                value = self.config.j_0 * np.sqrt(
                    max(site_energies[src], 0.0) * max(site_energies[dst], 0.0)
                )
                couplings[(src, dst)] = float(min(value, self.config.j_max))
        return site_energies, couplings

    def gamma_schedule(self, steps: int) -> np.ndarray:
        return np.linspace(self.config.gamma_min, self.config.gamma_max, steps)
