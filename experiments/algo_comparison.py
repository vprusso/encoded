"""Head-to-head algorithm comparison on the d=5 wall: which classical search
algorithm pushes past where beam search plateaus?

Compares random walk + max-coverage selector (existing), beam search (existing),
DFS + branch-and-bound (new), and simulated annealing (new) on three scenarios
of increasing difficulty.

Run: uv run python experiments/algo_comparison.py
"""

import itertools
import time

import stim

from encoded.add_stabilizers import (
    random_walk_extend,
    beam_search_extend,
    dfs_extend,
    simulated_annealing_extend,
)


PAULI_INDEX = {"X": 1, "Y": 2, "Z": 3}


def _fh_symmetries(n):
    g_up = stim.PauliString("".join("Z" if i % 2 == 0 else "_" for i in range(n)))
    g_down = stim.PauliString("".join("_" if i % 2 == 0 else "Z" for i in range(n)))
    return [g_up, g_down]


def _all_paulis_up_to_weight(n, weights, paulis="XYZ"):
    out = [stim.PauliString("_" * n)]
    codes = [PAULI_INDEX[c] for c in paulis]
    for w in sorted(weights):
        if w == 0:
            continue
        for positions in itertools.combinations(range(n), w):
            for picks in itertools.product(codes, repeat=w):
                mask = [0] * n
                for pos, code in zip(positions, picks):
                    mask[pos] = code
                out.append(stim.PauliString(mask))
    return out


def make_inputs(initial, total_n, weights=(1,)):
    n_data = max(len(g) for g in initial)
    pad_len = total_n - n_data
    pad = stim.PauliString("_" * pad_len) if pad_len > 0 else None
    padded = [g + pad for g in initial] if pad else list(initial)
    errors = _all_paulis_up_to_weight(total_n, weights)
    return padded, errors


def _fmt(result, elapsed, extra_info=""):
    n = max(len(g) for g in result.code) if result.code else 0
    k = n - len(result.code)
    d = ("-" if result.distance is None
         else f"{result.distance}" if result.distance_is_exact
         else f">={result.distance}")
    ok = "yes" if result.succeeded else "NO"
    return (
        f"  n={n} k={k} d={d} added={result.n_stabilizers_added} "
        f"rem={result.uncorrectables_remaining} ok={ok} "
        f"time={elapsed:.1f}s {extra_info}"
    )


def run_all_on(label, stabs, errs, max_stabilizers, target_stabilizers):
    print(f"\n=== {label} ===", flush=True)
    print(f"  budget: max_stabilizers={max_stabilizers} (target_stabilizers={target_stabilizers} for SA)", flush=True)

    # Beam search
    t0 = time.perf_counter()
    r = beam_search_extend(
        stabs, errs, extra_qubits=0,
        max_stabilizers=max_stabilizers,
        beam_width=8, n_expansions_per_slot=4, n_solution_samples=8,
        distance_max_weight=4, seed_val=137,
    )
    print(f"  beam_search:                {_fmt(r, time.perf_counter() - t0, f'walks={r.successful_walks}/{r.n_walks}')}", flush=True)

    # DFS + B&B
    t0 = time.perf_counter()
    r = dfs_extend(
        stabs, errs, extra_qubits=0,
        max_stabilizers=max_stabilizers,
        n_candidates_per_node=4,
        distance_max_weight=4, seed_val=137,
        max_seconds=120.0,
    )
    print(f"  dfs (n_cands=4):            {_fmt(r, time.perf_counter() - t0, f'nodes={r.n_walks}')}", flush=True)

    # DFS + B&B with bigger branching
    t0 = time.perf_counter()
    r = dfs_extend(
        stabs, errs, extra_qubits=0,
        max_stabilizers=max_stabilizers,
        n_candidates_per_node=8,
        distance_max_weight=4, seed_val=137,
        max_seconds=120.0,
    )
    print(f"  dfs (n_cands=8):            {_fmt(r, time.perf_counter() - t0, f'nodes={r.n_walks}')}", flush=True)

    # Simulated annealing
    t0 = time.perf_counter()
    r = simulated_annealing_extend(
        stabs, errs, extra_qubits=0,
        target_stabilizers=target_stabilizers,
        n_iterations=300, initial_temperature=5.0, cooling=0.99,
        distance_max_weight=4, seed_val=137,
    )
    print(f"  simulated_annealing:        {_fmt(r, time.perf_counter() - t0)}", flush=True)


def main():
    # Sanity: FH [[4,2]] -> d=3 (all algos should crush this)
    stabs, errs = make_inputs(_fh_symmetries(4), 6, weights=(1,))
    run_all_on("FH [[4,2]] -> d=3 (6 qubits) — sanity", stabs, errs,
               max_stabilizers=6, target_stabilizers=4)

    # The real test: FH [[4,2]] -> d=5 at 10 qubits with k>=1 forced.
    # Beam search already failed here with 0/8 walks at max_stabilizers=6 (forcing k=2).
    stabs, errs = make_inputs(_fh_symmetries(4), 10, weights=(1, 2))
    run_all_on("FH [[4,2]] -> d=5 (10 qubits, force k>=2, max_stabs=6)", stabs, errs,
               max_stabilizers=6, target_stabilizers=6)

    # Loosen: allow up to 7 added stabilizers (force k>=1).
    run_all_on("FH [[4,2]] -> d=5 (10 qubits, force k>=1, max_stabs=7)", stabs, errs,
               max_stabilizers=7, target_stabilizers=7)


if __name__ == "__main__":
    main()
