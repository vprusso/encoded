"""FH-realistic benchmark harness for random_walk_extend (slide 33).

Each scenario: take an initial stabilizer group + target errors + ancilla
template, run the random walk extender, and report:
  - whether the extension succeeded (all targets correctable)
  - resulting (n, k, # stabilizers added, # uncorrectables remaining)
  - wall time
  - # walks succeeded / total walks

Run with: uv run python experiments/bench_extend.py
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Optional

import stim

from encoded.add_stabilizers import random_walk_extend, beam_search_extend


PAULI_INDEX = {"X": 1, "Y": 2, "Z": 3}


def _all_single_qubit_paulis(n: int, paulis: str = "XYZ") -> List[stim.PauliString]:
    """Identity + all single-qubit P errors for P in `paulis` over n qubits."""
    out = [stim.PauliString("_" * n)]
    for i in range(n):
        for p_char in paulis:
            mask = [0] * n
            mask[i] = PAULI_INDEX[p_char]
            out.append(stim.PauliString(mask))
    return out


def _fh_symmetries(n: int) -> List[stim.PauliString]:
    """Particle-number-conservation generators for n-site Fermi-Hubbard with the
    staggered spin mapping (even qubits = spin up, odd = spin down) used in
    Ben's slide deck:
        G_up   = Z _ Z _ Z _ ... Z _
        G_down = _ Z _ Z _ Z ... _ Z
    """
    g_up = stim.PauliString("".join("Z" if i % 2 == 0 else "_" for i in range(n)))
    g_down = stim.PauliString("".join("_" if i % 2 == 0 else "Z" for i in range(n)))
    return [g_up, g_down]


@dataclass
class Scenario:
    name: str
    initial_stabilizers: List[stim.PauliString]
    errors: List[stim.PauliString]
    extra_support: Optional[stim.PauliString] = None
    ancilla_budget: int = 0
    method: str = "random_walk"  # "random_walk" or "beam_search"
    # random_walk params
    max_stabilizers_per_walk: int = 3
    max_walks: int = 20
    n_solution_samples: int = 1
    # beam_search params
    max_stabilizers: int = 4
    beam_width: int = 8
    n_expansions_per_slot: int = 4
    seed_val: int = 137


def _scenarios() -> List[Scenario]:
    return [
        Scenario(
            name="rep ZZ + {X1, X2}",
            initial_stabilizers=[stim.PauliString("ZZ")],
            errors=[stim.PauliString("__"), stim.PauliString("X_"), stim.PauliString("_X")],
            extra_support=stim.PauliString("Z"),
            max_stabilizers_per_walk=1, max_walks=10,
        ),
        Scenario(
            name="rep [[3,1]] + all weight-1",
            initial_stabilizers=[stim.PauliString("ZZ_"), stim.PauliString("_ZZ")],
            errors=_all_single_qubit_paulis(3),
            extra_support=stim.PauliString("X"),
            max_stabilizers_per_walk=2, max_walks=15,
        ),
        Scenario(
            name="FH [[4,2]] + weight-1 X (slide 13)",
            initial_stabilizers=_fh_symmetries(4),
            errors=_all_single_qubit_paulis(4, "X"),
            ancilla_budget=1,
            max_stabilizers_per_walk=2, max_walks=20,
        ),
        Scenario(
            name="FH [[4,2]] + all weight-1 (slide 34)",
            initial_stabilizers=_fh_symmetries(4),
            errors=_all_single_qubit_paulis(4),
            ancilla_budget=3,
            max_stabilizers_per_walk=4, max_walks=30,
        ),
        Scenario(
            name="FH [[8,6]] + weight-1 X (slide 14, beam)",
            initial_stabilizers=_fh_symmetries(8),
            errors=_all_single_qubit_paulis(8, "X"),
            ancilla_budget=2,
            method="beam_search",
            max_stabilizers=3, beam_width=8, n_expansions_per_slot=4,
            n_solution_samples=8,
        ),
        Scenario(
            name="FH [[16,14]] + weight-1 X (stress, beam)",
            initial_stabilizers=_fh_symmetries(16),
            errors=_all_single_qubit_paulis(16, "X"),
            ancilla_budget=4,
            method="beam_search",
            max_stabilizers=4, beam_width=8, n_expansions_per_slot=4,
            n_solution_samples=8,
        ),
    ]


def _run(sc: Scenario):
    t0 = time.perf_counter()
    if sc.method == "beam_search":
        result = beam_search_extend(
            sc.initial_stabilizers, sc.errors,
            extra_support=sc.extra_support,
            ancilla_budget=sc.ancilla_budget,
            max_stabilizers=sc.max_stabilizers,
            beam_width=sc.beam_width,
            n_expansions_per_slot=sc.n_expansions_per_slot,
            n_solution_samples=sc.n_solution_samples if sc.n_solution_samples > 1 else 8,
            seed_val=sc.seed_val,
        )
    else:
        result = random_walk_extend(
            sc.initial_stabilizers, sc.errors,
            extra_support=sc.extra_support,
            ancilla_budget=sc.ancilla_budget,
            max_stabilizers_per_walk=sc.max_stabilizers_per_walk,
            max_walks=sc.max_walks,
            n_solution_samples=sc.n_solution_samples,
            seed_val=sc.seed_val,
        )
    elapsed = time.perf_counter() - t0
    n = max(len(g) for g in result.code) if result.code else 0
    k = n - len(result.code)
    return n, k, result, elapsed


def main():
    print(
        f"{'scenario':<46} {'n':>3} {'k':>3} {'added':>5} {'rem':>4} "
        f"{'ok?':>4} {'time(s)':>8} {'succ/walks':>11}",
        flush=True,
    )
    print("-" * 100, flush=True)
    for sc in _scenarios():
        n, k, result, elapsed = _run(sc)
        ok = "yes" if result.succeeded else "NO"
        print(
            f"{sc.name:<46} {n:>3d} {k:>3d} {result.n_stabilizers_added:>5d} "
            f"{result.uncorrectables_remaining:>4d} {ok:>4} {elapsed:>8.2f} "
            f"{result.successful_walks:>3d}/{result.n_walks:<7d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
