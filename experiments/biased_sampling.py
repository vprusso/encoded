"""Biased low-weight sampling sweep on the d=5 wall.

Restart-based beam at 16 seeds confirmed the d=5 wall is structural under
uniform sampling: 15/16 hit rem=768, 1/16 hit rem=784, 0/16 succeed.

Hypothesis: the wall might be due to uniform sampling missing low-weight
stabilizers that good codes typically use. Bias each free-variable
assignment toward 0 (sparser solutions -> lower Pauli weight) and see if
any bias value escapes rem=768.

For each bias in {0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1}, run
beam_search_extend on FH [[4,2]] -> d=5 at n=10, max_stabilizers=6,
3 seeds each. Parallel via multiprocessing.

bias=0.5 is the uniform baseline (matches restart_beam.py result).
bias<0.5 is the new experiment.
"""

import itertools
import multiprocessing as mp
import time

import stim

from encoded.add_stabilizers import beam_search_extend


PAULI_INDEX = {"X": 1, "Y": 2, "Z": 3}
BIASES = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
SEEDS_PER_BIAS = 3
N_WORKERS = 6


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


def run_one(args):
    bias, seed = args
    stabs, errs = make_inputs(_fh_symmetries(4), 10, weights=(1, 2))
    t0 = time.perf_counter()
    r = beam_search_extend(
        stabs, errs,
        max_stabilizers=6, beam_width=8, n_expansions_per_slot=4,
        n_solution_samples=8,
        distance_max_weight=4,
        seed_val=seed,
        sampling_bias=bias,
    )
    return {
        "bias": bias,
        "seed": seed,
        "remaining": r.uncorrectables_remaining,
        "added": r.n_stabilizers_added,
        "succeeded": r.succeeded,
        "time": time.perf_counter() - t0,
    }


def main():
    jobs = [(b, s) for b in BIASES for s in range(1, SEEDS_PER_BIAS + 1)]
    print(f"Biased sampling sweep: {len(BIASES)} biases x {SEEDS_PER_BIAS} seeds = {len(jobs)} runs", flush=True)
    print(f"Workers: {N_WORKERS}", flush=True)
    print(f"Scenario: FH [[4,2]] -> d=5 (n=10, max_stabilizers=6, force k>=2)", flush=True)
    print(f"Reference: uniform sampling (bias=0.5) consistently hits rem=768.", flush=True)
    print(flush=True)
    print(f"{'bias':>5} {'seed':>5} {'rem':>5} {'added':>6} {'ok':>3} {'time(s)':>8}", flush=True)
    print("-" * 45, flush=True)

    t_start = time.perf_counter()
    by_bias: dict[float, list[int]] = {b: [] for b in BIASES}
    with mp.Pool(processes=N_WORKERS) as pool:
        for r in pool.imap_unordered(run_one, jobs):
            ok = "yes" if r["succeeded"] else "NO"
            print(
                f"{r['bias']:>5.2f} {r['seed']:>5d} {r['remaining']:>5d} "
                f"{r['added']:>6d} {ok:>3} {r['time']:>8.1f}",
                flush=True,
            )
            by_bias[r["bias"]].append(r["remaining"])

    total = time.perf_counter() - t_start
    print(f"\nTotal wall time: {total:.1f}s", flush=True)
    print("\nBest (min) remaining by bias:", flush=True)
    for b in BIASES:
        if by_bias[b]:
            print(f"  bias={b:.2f}: rems={sorted(by_bias[b])} (min={min(by_bias[b])})", flush=True)

    # Verdict
    any_success = any(any(rem == 0 for rem in by_bias[b]) for b in BIASES)
    biased_better_than_uniform = any(
        b != 0.5 and by_bias[b] and min(by_bias[b]) < 768
        for b in BIASES
    )
    if any_success:
        print("\n>>> Bias produced a SUCCESSFUL run. Sampler distribution matters.", flush=True)
    elif biased_better_than_uniform:
        print("\n>>> Biased sampling improved the plateau (got below 768). Worth more tuning.", flush=True)
    else:
        print("\n>>> No bias setting beat uniform sampling's rem=768 plateau.", flush=True)
        print("    Combined with restart_beam result, the d=5 wall is robust against", flush=True)
        print("    sampler-distribution variation. Reformulation likely required.", flush=True)


if __name__ == "__main__":
    main()
