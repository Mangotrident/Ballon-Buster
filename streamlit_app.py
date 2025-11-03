"""Streamlit dashboard for Quantum Bioenergetics Mapping."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from quantum_bioenergetics import QuantumLifeEngine, compute_summary_metrics
from quantum_bioenergetics.data.mapping import MappingConfig, ParameterMapper

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_GRAPH_PATH = DATA_DIR / "graph.json"
DEFAULT_PARAMS_PATH = DATA_DIR / "params.yaml"
DEFAULT_EXPRESSION_PATH = DATA_DIR / "sample_expression.csv"


def _load_json(path: Path) -> Dict[str, Sequence[str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Mapping[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_expression(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        series = df.iloc[:, 0]
        series.index = df.index
        return series.astype(float)
    if df.shape[1] >= 2:
        series = df.iloc[:, 1]
        series.index = df.iloc[:, 0]
        return series.astype(float)
    raise ValueError("Expression file must contain at least one column of numeric values")


def _parse_expression_upload(uploaded) -> Optional[pd.Series]:
    if uploaded is None:
        return None
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:  # pragma: no cover - user input guard
        st.error(f"Failed to read expression CSV: {exc}")
        return None
    if df.empty:
        st.error("Uploaded expression file is empty")
        return None
    if df.shape[1] == 1:
        series = df.iloc[:, 0]
        series.index = df.index
    else:
        series = df.iloc[:, -1]
        series.index = df.iloc[:, 0]
    try:
        return series.astype(float)
    except ValueError:
        st.error("Expression values must be numeric")
        return None


def _parse_graph_upload(uploaded) -> Optional[Dict[str, Sequence[str]]]:
    if uploaded is None:
        return None
    try:
        return json.load(uploaded)
    except Exception as exc:  # pragma: no cover - user input guard
        st.error(f"Failed to parse graph JSON: {exc}")
        return None


def _parse_params_upload(uploaded) -> Optional[Mapping[str, object]]:
    if uploaded is None:
        return None
    try:
        return yaml.safe_load(uploaded)
    except Exception as exc:  # pragma: no cover - user input guard
        st.error(f"Failed to parse parameter YAML: {exc}")
        return None


def _filter_couplings(
    couplings: Mapping[Tuple[str, str], float],
    topology: Mapping[str, Iterable[str]],
) -> Dict[Tuple[str, str], float]:
    adjacency = {node: set(neigh) for node, neigh in topology.items()}
    filtered: Dict[Tuple[str, str], float] = {}
    for (a, b), value in couplings.items():
        if b in adjacency.get(a, set()) or a in adjacency.get(b, set()):
            filtered[(a, b)] = value
    return filtered


def _run_simulation(
    topology: Mapping[str, Sequence[str]],
    params: Mapping[str, object],
    expression: pd.Series,
    steps: int,
    t_final: float,
    time_points: int,
):
    mapping_cfg = MappingConfig(**params["mapping"])
    mapper = ParameterMapper(mapping_cfg, params["node_genes"])
    site_energies, couplings = mapper.map_expression(expression.to_dict())
    couplings = _filter_couplings(couplings, topology)

    engine = QuantumLifeEngine(topology)
    hamiltonian = engine.build_hamiltonian(site_energies, couplings)
    gamma_schedule = mapper.gamma_schedule(steps)

    initial_state = np.zeros(engine.dim)
    initial_state[0] = 1.0

    sink_site = params.get("sink_site", engine.site_order[-2])
    sink_rate = params.get("sink_rate", 0.1)
    loss_rate = params.get("loss_rate", 0.01)
    sink_rates = {sink_site: float(sink_rate)}
    loss_rates = {site: float(loss_rate) for site in engine.site_order if site != sink_site}

    results = engine.parameter_sweep(
        hamiltonian,
        gamma_schedule,
        sink_rates=sink_rates,
        loss_rates=loss_rates,
        initial_state=initial_state,
        t_span=(0.0, t_final),
        t_eval=np.linspace(0.0, t_final, time_points),
    )
    summary = compute_summary_metrics(results)
    return results, summary


def _build_payload(results, summary) -> Dict[str, object]:
    return {
        "results": [
            {
                "gamma": float(r.parameters.get("gamma", np.nan)),
                "ete": float(r.ete),
                "coherence_lifetime": float(r.coherence_lifetime),
                "qls": float(r.qls),
            }
            for r in results
        ],
        "summary": {
            "ete_peak": float(summary.ete_peak),
            "gamma_star": float(summary.gamma_star),
            "tau_c_mean": float(summary.tau_c_mean),
            "qls_mean": float(summary.qls_mean),
            "resilience_index": float(summary.resilience_index),
        },
    }


st.set_page_config(
    page_title="Quantum Energy Mapping Explorer",
    page_icon="🧬",
    layout="wide",
)

st.title("Quantum Energy Mapping Explorer")
st.write(
    "Interactively explore quantum-coherent energy transport simulations. "
    "Upload your own omics-derived expression profiles or use the bundled "
    "reference dataset to generate ENAQT curves, biomarker metrics, and "
    "trajectory diagnostics."
)

with st.sidebar:
    st.header("Inputs")
    use_sample_graph = st.toggle("Use bundled graph", value=True)
    graph_upload = None if use_sample_graph else st.file_uploader("Graph JSON", type="json")
    use_sample_params = st.toggle("Use bundled parameters", value=True)
    params_upload = None if use_sample_params else st.file_uploader("Parameter YAML", type=["yml", "yaml"])
    st.markdown("---")
    use_sample_expression = st.toggle("Use bundled expression", value=True)
    expression_upload = None if use_sample_expression else st.file_uploader("Expression CSV", type="csv")
    steps = st.slider("Gamma steps", min_value=5, max_value=101, value=41, step=2)
    t_final = st.slider("Simulation horizon", min_value=10.0, max_value=120.0, value=50.0, step=5.0)
    time_points = st.slider("Time samples", min_value=50, max_value=600, value=200, step=10)
    run_request = st.button("Run simulation", type="primary")

if use_sample_graph:
    topology = _load_json(DEFAULT_GRAPH_PATH)
else:
    topology = _parse_graph_upload(graph_upload)

if use_sample_params:
    params = _load_yaml(DEFAULT_PARAMS_PATH)
else:
    params = _parse_params_upload(params_upload)

if use_sample_expression:
    expression = _load_expression(DEFAULT_EXPRESSION_PATH)
else:
    expression = _parse_expression_upload(expression_upload)

ready = topology is not None and params is not None and expression is not None

if ready and (run_request or "results" not in st.session_state):
    with st.spinner("Running quantum transport simulations..."):
        results, summary = _run_simulation(topology, params, expression, steps, t_final, time_points)
    st.session_state["results"] = results
    st.session_state["summary"] = summary
    st.session_state["payload"] = _build_payload(results, summary)
elif not ready:
    st.info("Awaiting all required inputs before running the simulation.")

results = st.session_state.get("results")
summary = st.session_state.get("summary")
payload = st.session_state.get("payload")

if results and summary:
    st.subheader("Summary metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ETE peak", f"{summary.ete_peak:.4f}", help="Maximum energy-transfer efficiency across the gamma sweep")
    c2.metric("Gamma*", f"{summary.gamma_star:.4f}", help="Noise rate that maximises ETE")
    c3.metric("Mean τc", f"{summary.tau_c_mean:.4f}", help="Average coherence lifetime across the sweep")
    c4.metric("Mean QLS", f"{summary.qls_mean:.4f}", help="Mean Quantum Life Score across simulations")
    c5.metric("Resilience", f"{summary.resilience_index:.4f}", help="Stability of coherence under perturbations")

    df = pd.DataFrame(
        [
            {
                "gamma": float(r.parameters.get("gamma", np.nan)),
                "ETE": float(r.ete),
                "Coherence lifetime": float(r.coherence_lifetime),
                "QLS": float(r.qls),
            }
            for r in results
        ]
    ).sort_values("gamma")
    st.subheader("Gamma sweep metrics")
    st.dataframe(df.style.format({"ETE": "{:.4f}", "Coherence lifetime": "{:.4f}", "QLS": "{:.4f}"}))
    chart_df = df.set_index("gamma")
    st.line_chart(chart_df)

    gamma_options = df["gamma"].to_list()
    st.subheader("Sink population dynamics")
    selected_gamma = st.select_slider("Select gamma value", options=gamma_options, value=gamma_options[len(gamma_options) // 2])
    selected_result = next(r for r in results if abs(r.parameters.get("gamma", 0.0) - selected_gamma) < 1e-9)
    trajectory_df = pd.DataFrame(
        {
            "time": selected_result.times,
            "sink_population": selected_result.sink_population,
        }
    ).set_index("time")
    st.area_chart(trajectory_df)

    if payload:
        st.download_button(
            "Download metrics JSON",
            data=json.dumps(payload, indent=2),
            file_name="simulation_metrics.json",
            mime="application/json",
        )
else:
    st.stop()
