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

from encoded.add_stabilizers import random_walk_extend


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
    max_stabilizers_per_walk: int = 3
    max_walks: int = 20
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
            name="FH [[8,6]] + weight-1 X (slide 14)",
            initial_stabilizers=_fh_symmetries(8),
            errors=_all_single_qubit_paulis(8, "X"),
            ancilla_budget=2,
            max_stabilizers_per_walk=3, max_walks=50,
        ),
        Scenario(
            name="FH [[16,14]] + weight-1 X (stress)",
            initial_stabilizers=_fh_symmetries(16),
            errors=_all_single_qubit_paulis(16, "X"),
            ancilla_budget=4,
            max_stabilizers_per_walk=4, max_walks=50,
        ),
    ]


def _run(sc: Scenario):
    t0 = time.perf_counter()
    result = random_walk_extend(
        sc.initial_stabilizers, sc.errors,
        extra_support=sc.extra_support,
        ancilla_budget=sc.ancilla_budget,
        max_stabilizers_per_walk=sc.max_stabilizers_per_walk,
        max_walks=sc.max_walks,
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
