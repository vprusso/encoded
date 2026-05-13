"""Micro-bench: is_in_stabilizer_group (new, O(k*n^2)) vs. _legacy_group_membership_check
(old, O(2^k)). Run with PYTHONPATH=. .venv/bin/python experiments/bench_membership.py

Scenarios are sized to match realistic call sites of get_uncorrectable_errors during a
build_code_randomly run on (a) the rep code, (b) the [[5,2]] Fermi-Hubbard code, (c) Shor."""

import time
from dataclasses import dataclass

import stim

from encoded.add_stabilizers import is_in_stabilizer_group, _legacy_group_membership_check


@dataclass
class Scenario:
    name: str
    generators: list[stim.PauliString]
    queries: list[stim.PauliString]


def _all_single_qubit_paulis(n: int) -> list[stim.PauliString]:
    out = [stim.PauliString("_" * n)]
    for i in range(n):
        for p in (1, 2, 3):
            mask = [0] * n
            mask[i] = p
            out.append(stim.PauliString(mask))
    return out


def _scenarios() -> list[Scenario]:
    # Shor-shaped: 8 ZZ-pair generators on 9 qubits, query against products of single-Pauli errors.
    shor_gens = [
        stim.PauliString("ZZ_______"),
        stim.PauliString("_ZZ______"),
        stim.PauliString("___ZZ____"),
        stim.PauliString("____ZZ___"),
        stim.PauliString("______ZZ_"),
        stim.PauliString("_______ZZ"),
        stim.PauliString("XXXXXX___"),
        stim.PauliString("___XXXXXX"),
    ]
    shor_errs = _all_single_qubit_paulis(9)
    shor_queries = [e1 * e2 for e1 in shor_errs for e2 in shor_errs]

    # 5-qubit-perfect-shaped: 4 generators on 5 qubits, query against single-qubit error products.
    five_gens = [
        stim.PauliString("XZZX_"),
        stim.PauliString("_XZZX"),
        stim.PauliString("X_XZZ"),
        stim.PauliString("ZX_XZ"),
    ]
    five_errs = _all_single_qubit_paulis(5)
    five_queries = [e1 * e2 for e1 in five_errs for e2 in five_errs]

    # [[4,2]] Fermi-Hubbard-shaped: 2 generators on 4 qubits.
    fh4_gens = [
        stim.PauliString("ZIZI"),
        stim.PauliString("IZIZ"),
    ]
    fh4_errs = _all_single_qubit_paulis(4)
    fh4_queries = [e1 * e2 for e1 in fh4_errs for e2 in fh4_errs]

    return [
        Scenario("[[4,2]] FH (k=2 gens, n=4)", fh4_gens, fh4_queries),
        Scenario("5-qubit perfect (k=4 gens, n=5)", five_gens, five_queries),
        Scenario("Shor (k=8 gens, n=9)", shor_gens, shor_queries),
    ]


def _time(fn, gens, queries):
    t0 = time.perf_counter()
    truthy = 0
    for q in queries:
        truthy += int(fn(q, gens))
    elapsed = time.perf_counter() - t0
    return elapsed, truthy


def main():
    print(f"{'scenario':<40} {'queries':>8} {'legacy(s)':>12} {'fast(s)':>10} {'speedup':>10}  agree?")
    print("-" * 100)
    for sc in _scenarios():
        legacy_s, legacy_true = _time(_legacy_group_membership_check, sc.generators, sc.queries)
        fast_s, fast_true = _time(is_in_stabilizer_group, sc.generators, sc.queries)
        speedup = legacy_s / fast_s if fast_s > 0 else float("inf")
        agree = legacy_true == fast_true
        print(
            f"{sc.name:<40} {len(sc.queries):>8d} {legacy_s:>12.4f} {fast_s:>10.4f} "
            f"{speedup:>9.1f}x  {'yes' if agree else 'NO!'}"
        )


if __name__ == "__main__":
    main()
