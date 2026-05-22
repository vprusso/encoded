"""Restart-based beam search across many random seeds — the cheapest possible
test of whether the d=5 wall is structural or just exploration luck.

Hypothesis: if all seeds converge to the same exact local minimum
(remaining=768 on FH [[4,2]] -> d=5 at n=10 with k>=2 forced), the wall is
structural — no amount of random restart will break it, and we can
confidently move to reformulation (MAX-XORSAT, CSS, etc).

If any seed escapes the 768 plateau, the wall is seed-dependent and worth
more search-side work (more seeds, smarter heuristics, etc).

Runs `N_SEEDS` seeds in parallel via multiprocessing.Pool. Each beam search
uses the same config that produced rem=768 in the original failing run.
"""

import itertools
import multiprocessing as mp
import time

import stim

from encoded.add_stabilizers import beam_search_extend


PAULI_INDEX = {"X": 1, "Y": 2, "Z": 3}
N_SEEDS = 16
N_WORKERS = 6


def _fh_symmetries(n: int) -> list[stim.PauliString]:
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


def run_one_seed(seed: int) -> dict:
    """One beam search run on FH [[4,2]] -> d=5 at n=10 with k>=2 forced."""
    stabs, errs = make_inputs(_fh_symmetries(4), 10, weights=(1, 2))
    t0 = time.perf_counter()
    r = beam_search_extend(
        stabs, errs,
        max_stabilizers=6, beam_width=8, n_expansions_per_slot=4,
        n_solution_samples=8,
        distance_max_weight=4,
        seed_val=seed,
    )
    return {
        "seed": seed,
        "remaining": r.uncorrectables_remaining,
        "added": r.n_stabilizers_added,
        "succeeded": r.succeeded,
        "time": time.perf_counter() - t0,
    }


def main():
    print(f"Restart-based beam: {N_SEEDS} seeds, {N_WORKERS} parallel workers", flush=True)
    print(f"Scenario: FH [[4,2]] -> d=5 (n=10, max_stabilizers=6, force k>=2)", flush=True)
    print(f"Reference: original failing run (seed=137) hit rem=768.", flush=True)
    print(f"Decision criterion: if ALL seeds converge to rem=768, wall is structural.", flush=True)
    print(flush=True)
    print(f"{'seed':>5} {'rem':>5} {'added':>6} {'ok':>3} {'time(s)':>8}", flush=True)
    print("-" * 40, flush=True)

    seeds = list(range(1, N_SEEDS + 1))
    t_start = time.perf_counter()
    results = []
    with mp.Pool(processes=N_WORKERS) as pool:
        for r in pool.imap_unordered(run_one_seed, seeds):
            ok = "yes" if r["succeeded"] else "NO"
            print(
                f"{r['seed']:>5d} {r['remaining']:>5d} {r['added']:>6d} {ok:>3} "
                f"{r['time']:>8.1f}",
                flush=True,
            )
            results.append(r)

    total = time.perf_counter() - t_start
    print(f"\nTotal wall time: {total:.1f}s", flush=True)

    # Summary
    remainings = sorted({r["remaining"] for r in results})
    n_succeeded = sum(1 for r in results if r["succeeded"])
    print(f"\nUnique remaining values: {remainings}", flush=True)
    print(f"Successful runs: {n_succeeded} / {len(results)}", flush=True)
    if remainings == [768]:
        print("\n>>> ALL SEEDS CONVERGED TO rem=768. Wall is structural.", flush=True)
    elif n_succeeded > 0:
        print(f"\n>>> {n_succeeded} seeds SUCCEEDED. Wall is seed-dependent.", flush=True)
    else:
        print(f"\n>>> Multiple plateaus found: {remainings}. Wall has texture; worth more search.", flush=True)


if __name__ == "__main__":
    main()
