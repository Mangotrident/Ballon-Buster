"""Command line interface for Quantum Bioenergetics Mapping."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from quantum_bioenergetics import QuantumLifeEngine, compute_summary_metrics
from quantum_bioenergetics.data.mapping import MappingConfig, ParameterMapper


def load_topology(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_expression(path: Path) -> Dict[str, float]:
    data = pd.read_csv(path, index_col=0)
    if data.shape[1] != 1:
        raise ValueError("expression CSV must have exactly one column of expression values")
    return data.iloc[:, 0].to_dict()


def run_simulation(args: argparse.Namespace) -> None:
    topology = load_topology(Path(args.graph))
    with Path(args.params).open("r", encoding="utf-8") as fp:
        params = yaml.safe_load(fp)

    mapper = ParameterMapper(MappingConfig(**params["mapping"]), params["node_genes"])
    expression = load_expression(Path(args.expression))
    site_energies, couplings = mapper.map_expression(expression)
    adjacency = {node: set(neighbors) for node, neighbors in topology.items()}
    filtered_couplings = {}
    for (a, b), value in couplings.items():
        if b in adjacency.get(a, set()) or a in adjacency.get(b, set()):
            filtered_couplings[(a, b)] = value
    site_energies = {k: float(v) for k, v in site_energies.items()}
    couplings = filtered_couplings

    engine = QuantumLifeEngine(topology)
    hamiltonian = engine.build_hamiltonian(site_energies, couplings)

    gamma_schedule = mapper.gamma_schedule(args.steps)
    initial_state = np.zeros(engine.dim)
    initial_state[0] = 1.0
    sink_site = params.get("sink_site", engine.site_order[-2])
    sink_rates = {sink_site: params["sink_rate"]}
    loss_rates = {site: params["loss_rate"] for site in engine.site_order if site != sink_site}

    results = engine.parameter_sweep(
        hamiltonian,
        gamma_schedule,
        sink_rates=sink_rates,
        loss_rates=loss_rates,
        initial_state=initial_state,
        t_span=(0.0, args.t_final),
        t_eval=np.linspace(0.0, args.t_final, args.time_points),
    )

    summary = compute_summary_metrics(results)

    output = {
        "results": [
            {
                "gamma": r.parameters["gamma"],
                "ete": r.ete,
                "coherence_lifetime": r.coherence_lifetime,
                "qls": r.qls,
            }
            for r in results
        ],
        "summary": {
            "ete_peak": summary.ete_peak,
            "gamma_star": summary.gamma_star,
            "tau_c_mean": summary.tau_c_mean,
            "qls_mean": summary.qls_mean,
            "resilience_index": summary.resilience_index,
        },
    }

    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulate", help="run the quantum simulation", nargs="?")
    parser.add_argument("--graph", required=True, help="Path to network graph JSON")
    parser.add_argument("--expression", required=True, help="Path to expression CSV")
    parser.add_argument("--params", required=True, help="Path to parameter YAML")
    parser.add_argument("--output", required=True, help="Path to write metrics JSON")
    parser.add_argument("--steps", type=int, default=21, help="Gamma sweep steps")
    parser.add_argument("--t-final", type=float, default=50.0, dest="t_final")
    parser.add_argument("--time-points", type=int, default=200)
    args = parser.parse_args()
    run_simulation(args)


if __name__ == "__main__":
    main()
