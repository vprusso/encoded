"""Config B: FH [[4,2]] -> d=5 at 10 qubits but with `max_stabilizers=6` so the
algorithm is forced to either find a [[10, 2, 5]] code (2 initial + 6 added = 8
stabilizers; k = 10 - 8 = 2) or fail. This is the diagnostic of whether the
random walk *can't* find shorter walks (vs. the lex tie-breaker just never seeing
them in config 1 within budget).

Run while explore_d5.py is still working on config 2 in the background.
"""

import itertools
import time

import stim

from encoded.add_stabilizers import beam_search_extend


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


def make_inputs(initial_stabilizers, total_n_qubits, error_weights=(1,)):
    n_data = max(len(g) for g in initial_stabilizers)
    pad_len = total_n_qubits - n_data
    pad = stim.PauliString("_" * pad_len) if pad_len > 0 else None
    padded = [g + pad for g in initial_stabilizers] if pad else list(initial_stabilizers)
    errors = _all_paulis_up_to_weight(total_n_qubits, error_weights)
    return padded, errors


def main():
    stabs, errs = make_inputs(_fh_symmetries(4), total_n_qubits=10, error_weights=(1, 2))
    print("=== FH [[4,2]] -> d=5 (10 qubits, capped at 6 added stabilizers) ===", flush=True)
    print("  Target: [[10, 2, 5]] or fail-fast", flush=True)
    t0 = time.perf_counter()
    result = beam_search_extend(
        stabs, errs,
        extra_qubits=0,
        max_stabilizers=6, beam_width=8, n_expansions_per_slot=4,
        n_solution_samples=8,
        distance_max_weight=4,
        seed_val=137,
    )
    elapsed = time.perf_counter() - t0
    n = max(len(g) for g in result.code) if result.code else 0
    k = n - len(result.code)
    if result.distance is None:
        d_str = "-"
    elif result.distance_is_exact:
        d_str = f"{result.distance}"
    else:
        d_str = f">={result.distance}"
    print(
        f"  result: n={n}, k={k}, d={d_str}, added={result.n_stabilizers_added}, "
        f"remaining={result.uncorrectables_remaining}, succeeded={result.succeeded}, "
        f"successful_walks={result.successful_walks}/{result.n_walks}, "
        f"time={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
