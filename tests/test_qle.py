import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from quantum_bioenergetics.engine.qle import QuantumLifeEngine
from quantum_bioenergetics.data.mapping import MappingConfig, ParameterMapper


def load_inputs():
    graph = json.loads(Path("data/graph.json").read_text())
    params = yaml.safe_load(Path("data/params.yaml").read_text())
    mapper = ParameterMapper(MappingConfig(**params["mapping"]), params["node_genes"])
    expr_df = pd.read_csv("data/sample_expression.csv")
    expression = dict(zip(expr_df["gene"], expr_df["expr"]))
    return graph, params, mapper, expression


def test_trace_preservation():
    graph, params, mapper, expression = load_inputs()
    engine = QuantumLifeEngine(graph)
    site_energies, couplings = mapper.map_expression(expression)
    hamiltonian = engine.build_hamiltonian(site_energies, couplings)
    initial_state = np.zeros(engine.dim)
    initial_state[0] = 1.0
    sink_site = params.get("sink_site", engine.site_order[-2])
    result = engine.simulate(
        hamiltonian,
        gamma={site: 0.02 for site in engine.site_order},
        sink_rates={sink_site: params["sink_rate"]},
        loss_rates={site: params["loss_rate"] for site in engine.site_order if site != sink_site},
        initial_state=initial_state,
        t_span=(0.0, 25.0),
        t_eval=np.linspace(0.0, 25.0, 100),
    )
    traces = [np.trace(rho) for rho in result.density_matrices]
    assert np.allclose(traces, 1.0, atol=1e-3)


def test_enaqt_curve():
    graph, params, mapper, expression = load_inputs()
    engine = QuantumLifeEngine(graph)
    site_energies, couplings = mapper.map_expression(expression)
    hamiltonian = engine.build_hamiltonian(site_energies, couplings)
    gammas = np.linspace(0.0, 0.05, 11)
    initial_state = np.zeros(engine.dim)
    initial_state[0] = 1.0
    sink_site = params.get("sink_site", engine.site_order[-2])
    results = engine.parameter_sweep(
        hamiltonian,
        gamma_schedule=gammas,
        sink_rates={sink_site: params["sink_rate"]},
        loss_rates={site: params["loss_rate"] for site in engine.site_order if site != sink_site},
        initial_state=initial_state,
        t_span=(0.0, 40.0),
        t_eval=np.linspace(0.0, 40.0, 150),
    )
    etes = [r.ete for r in results]
    peak_idx = int(np.argmax(etes))
    assert 0 < peak_idx < len(etes) - 1
    assert etes[peak_idx] > etes[0] and etes[peak_idx] > etes[-1]
