"""Focused d=5 retries with wider beams / more extra qubits.

Two attempts:
  1. FH [[4,2]] -> d=5 with extra_qubits=6 (10 total). The default-bench config
     produced [[8, 0, >=5]] (k=0 collapse). With 2 more extra qubits there's
     room for the algorithm to land at [[10, k>=1, >=5]] -- our first d=5
     code that actually encodes a logical qubit.
  2. FH [[8,6]] -> d=5 with a wider beam and more extra qubits. The default
     bench failed (0/8 walks, 60 uncorrectables remaining at beam=8,
     n_exp=4, max_stabilizers=10, extra_qubits=6). Bumping all four knobs.

Run: uv run python experiments/explore_d5.py
"""

import itertools
import time

import stim

from encoded.add_stabilizers import beam_search_extend


PAULI_INDEX = {"X": 1, "Y": 2, "Z": 3}


def _fh_symmetries(n: int) -> list[stim.PauliString]:
    g_up = stim.PauliString("".join("Z" if i % 2 == 0 else "_" for i in range(n)))
    g_down = stim.PauliString("".join("_" if i % 2 == 0 else "Z" for i in range(n)))
    return [g_up, g_down]


def _all_paulis_up_to_weight(n, weights, paulis="XYZ"):
    out = [stim.PauliString("_" * n)]
    pauli_codes = [PAULI_INDEX[c] for c in paulis]
    for w in sorted(weights):
        if w == 0:
            continue
        for positions in itertools.combinations(range(n), w):
            for codes in itertools.product(pauli_codes, repeat=w):
                mask = [0] * n
                for pos, code in zip(positions, codes):
                    mask[pos] = code
                out.append(stim.PauliString(mask))
    return out


def make_extension_inputs(initial_stabilizers, total_n_qubits, error_weights=(1,)):
    n_data = max(len(g) for g in initial_stabilizers)
    pad_len = total_n_qubits - n_data
    pad = stim.PauliString("_" * pad_len) if pad_len > 0 else None
    padded = [g + pad for g in initial_stabilizers] if pad is not None else list(initial_stabilizers)
    errors = _all_paulis_up_to_weight(total_n_qubits, error_weights)
    return padded, errors


def _run_config(label, initial_stabs, errors, **kwargs):
    print(f"\n=== {label} ===", flush=True)
    for k, v in kwargs.items():
        print(f"  {k} = {v}", flush=True)
    t0 = time.perf_counter()
    result = beam_search_extend(initial_stabs, errors, **kwargs)
    elapsed = time.perf_counter() - t0
    n = max(len(g) for g in result.code) if result.code else 0
    k = n - len(result.code)
    d_str = (
        f"{result.distance}" if result.distance_is_exact and result.distance is not None
        else f">={result.distance}" if result.distance is not None
        else "-"
    )
    print(
        f"  result: n={n}, k={k}, d={d_str}, added={result.n_stabilizers_added}, "
        f"remaining={result.uncorrectables_remaining}, succeeded={result.succeeded}, "
        f"successful_walks={result.successful_walks}/{result.n_walks}, "
        f"time={elapsed:.1f}s",
        flush=True,
    )
    return result, elapsed


def main():
    # 1. FH [[4,2]] -> d=5 with 6 extra qubits (10 total). Cheap.
    stabs, errs = make_extension_inputs(
        _fh_symmetries(4), total_n_qubits=10, error_weights=(1, 2),
    )
    _run_config(
        "FH [[4,2]] -> d=5 (10 qubits, 6 extra, wider room)",
        stabs, errs,
        extra_qubits=0,
        max_stabilizers=8, beam_width=8, n_expansions_per_slot=4,
        n_solution_samples=8,
        distance_max_weight=4,
    )

    # 2. FH [[8,6]] -> d=5 retry with bigger budget on all axes.
    stabs, errs = make_extension_inputs(
        _fh_symmetries(8), total_n_qubits=16, error_weights=(1, 2),
    )
    _run_config(
        "FH [[8,6]] -> d=5 (16 qubits, 8 extra, wider beam)",
        stabs, errs,
        extra_qubits=0,
        max_stabilizers=12, beam_width=16, n_expansions_per_slot=4,
        n_solution_samples=16,
        distance_max_weight=4,
    )


if __name__ == "__main__":
    main()
