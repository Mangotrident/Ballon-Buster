# Quantum Bioenergetics Mapping

Quantum Bioenergetics Mapping is an applied-physics software stack that
simulates energy transport in mitochondrial networks and converts patient
omics into first-principles biomarkers. The project implements the Quantum
Life Engine (QLE) and Quantum-Metabolic Network (Q-MNet) foundations
described in the product vision.

## Features

- Lindblad master equation solver tailored for excitation transport.
- Automatic mapping from gene expression to Hamiltonian parameters.
- Gamma noise sweeps that reproduce ENAQT-like energy-transfer profiles.
- Cohort-ready summary metrics (ETE peak, coherence lifetime, QLS).
- CLI for end-to-end simulation on curated mitochondrial graphs.

## Quickstart

1. Install dependencies (preferably in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

2. Run the sample simulation:

   ```bash
   python cli.py simulate \
     --graph data/graph.json \
     --expression data/sample_expression.csv \
     --params data/params.yaml \
     --output examples/sample_metrics.json
   ```

3. Inspect the resulting metrics in `examples/sample_metrics.json`.

## Tests

Execute the unit tests to validate trace preservation and ENAQT behaviour:

```bash
pytest
```

## Repository layout

- `quantum_bioenergetics/engine` – Quantum Life Engine implementation.
- `quantum_bioenergetics/data` – Omics-to-physics mapping utilities.
- `quantum_bioenergetics/analysis` – Cohort-level metric aggregation.
- `data/` – Input graphs, parameters, and expression templates.
- `examples/` – Generated artefacts and notebook-ready outputs.
- `tests/` – Physics validation tests.

## License

MIT License
