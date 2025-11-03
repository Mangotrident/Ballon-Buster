"""Quantum Life Engine implementation.

This module implements a Lindblad master equation integrator tailored to
mitochondrial-like excitation transport networks.  It exposes a high level
``QuantumLifeEngine`` API capable of constructing Hamiltonians from graph
specifications, evolving the system under environmental noise, and computing
transport metrics such as Energy Transfer Efficiency (ETE), coherence
lifetime, and a composite Quantum Life Score (QLS).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp


ComplexMatrix = np.ndarray


@dataclass
class SimulationResult:
    """Container with per-run metrics and trajectories."""

    times: np.ndarray
    density_matrices: np.ndarray
    ete: float
    coherence_lifetime: float
    qls: float
    sink_population: np.ndarray
    parameters: Dict[str, float]


def _commutator(a: ComplexMatrix, b: ComplexMatrix) -> ComplexMatrix:
    return a @ b - b @ a


def _vectorize_density(rho: ComplexMatrix) -> np.ndarray:
    return rho.reshape(-1)


def _devectorize_density(vec: np.ndarray, dim: int) -> ComplexMatrix:
    return vec.reshape((dim, dim))


def _lindblad_superoperator(l_op: ComplexMatrix, rho: ComplexMatrix) -> ComplexMatrix:
    term1 = l_op @ rho @ l_op.conj().T
    term2 = l_op.conj().T @ l_op @ rho
    term3 = rho @ l_op.conj().T @ l_op
    return term1 - 0.5 * (term2 + term3)


class QuantumLifeEngine:
    """Core solver for quantum-coherent energy transport.

    Parameters
    ----------
    topology : Mapping[str, Sequence[str]]
        Graph adjacency list describing the network topology.
    site_order : Optional[Sequence[str]]
        Optional order of sites. Defaults to sorted topology keys.
    """

    def __init__(
        self,
        topology: Mapping[str, Sequence[str]],
        site_order: Optional[Sequence[str]] = None,
    ) -> None:
        self.topology = topology
        self.site_order = list(site_order) if site_order is not None else sorted(topology)
        self.index_map = {site: idx for idx, site in enumerate(self.site_order)}
        self.dim = len(self.site_order)

    # ------------------------------------------------------------------
    # Hamiltonian construction
    # ------------------------------------------------------------------
    def build_hamiltonian(
        self,
        site_energies: Mapping[str, float],
        couplings: Mapping[Tuple[str, str], float],
    ) -> ComplexMatrix:
        hamiltonian = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for site, idx in self.index_map.items():
            hamiltonian[idx, idx] = site_energies.get(site, 0.0)
        for (a, b), value in couplings.items():
            ia, ib = self.index_map[a], self.index_map[b]
            hamiltonian[ia, ib] = value
            hamiltonian[ib, ia] = np.conj(value)
        return hamiltonian

    # ------------------------------------------------------------------
    def _build_dephasing_ops(self, gamma: Mapping[str, float]) -> List[ComplexMatrix]:
        operators: List[ComplexMatrix] = []
        for site, rate in gamma.items():
            if rate == 0.0:
                continue
            idx = self.index_map[site]
            op = np.zeros((self.dim, self.dim), dtype=np.complex128)
            op[idx, idx] = np.sqrt(rate)
            operators.append(op)
        return operators

    def _build_sink_ops(
        self,
        sink_rates: Mapping[str, float],
        sink_index: int,
    ) -> Tuple[List[ComplexMatrix], List[float]]:
        operators: List[ComplexMatrix] = []
        rates: List[float] = []
        for site, rate in sink_rates.items():
            if rate == 0.0:
                continue
            idx = self.index_map[site]
            op = np.zeros((self.dim, self.dim), dtype=np.complex128)
            op[sink_index, idx] = np.sqrt(rate)
            operators.append(op)
            rates.append(rate)
        return operators, rates

    # ------------------------------------------------------------------
    def simulate(
        self,
        hamiltonian: ComplexMatrix,
        gamma: Mapping[str, float],
        sink_rates: Mapping[str, float],
        loss_rates: Mapping[str, float],
        initial_state: ArrayLike,
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        atol: float = 1e-8,
        rtol: float = 1e-7,
    ) -> SimulationResult:
        dim = self.dim
        rho0 = np.array(initial_state, dtype=np.complex128)
        if rho0.shape == (dim,):
            rho0 = np.outer(rho0, rho0.conj())
        if rho0.shape != (dim, dim):
            raise ValueError("initial_state must be a state vector or density matrix")
        rho0 = rho0 / np.trace(rho0)

        gamma_ops = self._build_dephasing_ops(gamma)
        sink_index = dim - 1
        sink_ops, sink_rates_array = self._build_sink_ops(sink_rates, sink_index)
        loss_ops, _ = self._build_sink_ops(loss_rates, sink_index)
        lindblad_ops = gamma_ops + sink_ops + loss_ops

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            rho = _devectorize_density(y, dim)
            comm = -1j * _commutator(hamiltonian, rho)
            dissipative = np.zeros_like(comm)
            for op in lindblad_ops:
                dissipative += _lindblad_superoperator(op, rho)
            drho_dt = comm + dissipative
            return _vectorize_density(drho_dt)

        sol = solve_ivp(
            rhs,
            t_span,
            _vectorize_density(rho0),
            t_eval=np.array(t_eval) if t_eval is not None else None,
            atol=atol,
            rtol=rtol,
            method="RK45",
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        times = sol.t
        density_matrices = np.array([
            _devectorize_density(y, dim) for y in sol.y.T
        ])

        sink_pop_rate = np.zeros_like(times)
        sink_population = np.zeros_like(times)
        sink_basis = np.zeros(dim)
        sink_basis[sink_index] = 1.0
        projector_sink = np.outer(sink_basis, sink_basis)
        for idx, rho in enumerate(density_matrices):
            population = np.real(np.trace(projector_sink @ rho))
            sink_population[idx] = population
            inflow = 0.0
            for rate, site in zip(sink_rates_array, sink_rates.keys()):
                op = np.zeros((dim, dim), dtype=np.complex128)
                op[sink_index, self.index_map[site]] = np.sqrt(rate)
                inflow += np.real(np.trace(op @ rho @ op.conj().T))
            sink_pop_rate[idx] = inflow

        ete = np.trapz(sink_pop_rate, times)

        coherence_terms: List[float] = []
        for rho in density_matrices:
            off_diag = rho.copy()
            np.fill_diagonal(off_diag, 0.0)
            coherence_terms.append(np.linalg.norm(off_diag))
        coherence_terms = np.array(coherence_terms)
        coherence_initial = coherence_terms[0]
        threshold = coherence_initial * np.exp(-1)
        above = coherence_terms >= threshold
        if not np.any(above):
            coherence_lifetime = 0.0
        else:
            last_idx = np.argmax(~above)
            if last_idx == 0 and not above[0]:
                coherence_lifetime = 0.0
            elif last_idx == 0 and above[0]:
                coherence_lifetime = times[-1] - times[0]
            else:
                coherence_lifetime = times[last_idx] - times[0]

        resilience = float(np.exp(-np.var(coherence_terms)))
        qls = float(0.6 * ete + 0.3 * coherence_lifetime + 0.1 * resilience)

        return SimulationResult(
            times=times,
            density_matrices=density_matrices,
            ete=float(np.real(ete)),
            coherence_lifetime=float(coherence_lifetime),
            qls=qls,
            sink_population=sink_population,
            parameters={
                "ete": float(np.real(ete)),
                "coherence_lifetime": float(coherence_lifetime),
                "resilience": resilience,
            },
        )

    # ------------------------------------------------------------------
    def parameter_sweep(
        self,
        base_hamiltonian: ComplexMatrix,
        gamma_schedule: Sequence[float],
        sink_rates: Mapping[str, float],
        loss_rates: Mapping[str, float],
        initial_state: ArrayLike,
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
    ) -> List[SimulationResult]:
        results: List[SimulationResult] = []
        for gamma_value in gamma_schedule:
            gamma_map = {site: gamma_value for site in self.site_order}
            result = self.simulate(
                base_hamiltonian,
                gamma=gamma_map,
                sink_rates=sink_rates,
                loss_rates=loss_rates,
                initial_state=initial_state,
                t_span=t_span,
                t_eval=t_eval,
            )
            result.parameters["gamma"] = gamma_value
            results.append(result)
        return results
