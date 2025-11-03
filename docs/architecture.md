# Quantum Bioenergetics Architecture

## Layer A — Quantum Life Engine
- Hamiltonian construction from mitochondrial graph.
- Lindblad integration with configurable dephasing, sink, and loss channels.
- Metrics: ETE, coherence lifetime, Quantum Life Score, resilience.

## Layer B — Quantum-Metabolic Network
- ParameterMapper converts expression (omics) to site energies and couplings.
- Gamma schedule sweep generates ENAQT curves for cohort analyses.

## Layer C — Experience Layer (Streamlit Dashboard)
- `streamlit_app.py` orchestrates inputs, sweeps, and visualisations.
- Supports bundled data for validation and custom uploads for real cohorts.
- Provides download-ready metrics payloads for downstream APIs and analytics.

## Pipelines
1. Ingest omics → ParameterMapper → Hamiltonian.
2. Run QuantumLifeEngine parameter sweep to produce metrics per gamma.
3. Aggregate via `compute_summary_metrics` for cohort-level comparison.
4. Visualise via Streamlit dashboard; export JSON for APIs or Render deployments.

## Artefacts
- `metrics.json`: run-level metrics.
- `cohort_metrics.parquet`: (future) aggregated outputs for cohorts.
- `edge_ranking.csv`: (future) interpretability ranking placeholder.

## Testing Strategy
- Trace preservation regression tests.
- ENAQT bell-curve shape validation.
