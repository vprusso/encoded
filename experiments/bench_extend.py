"""FH-realistic benchmark harness for random_walk_extend (slide 33).

Each scenario: take an initial stabilizer group + target errors + extra-qubit
template, run the random walk extender, and report:
  - whether the extension succeeded (all targets correctable)
  - resulting (n, k, # stabilizers added, # uncorrectables remaining)
  - wall time
  - # walks succeeded / total walks

Run with: uv run python experiments/bench_extend.py
"""

import itertools
import time
from dataclasses import dataclass

import stim

from encoded.add_stabilizers import random_walk_extend, beam_search_extend
from encoded import connectivity as connect


PAULI_INDEX = {"X": 1, "Y": 2, "Z": 3}


def _all_single_qubit_paulis(n: int, paulis: str = "XYZ") -> list[stim.PauliString]:
    """Identity + all single-qubit P errors for P in `paulis` over n qubits."""
    out = [stim.PauliString("_" * n)]
    for i in range(n):
        for p_char in paulis:
            mask = [0] * n
            mask[i] = PAULI_INDEX[p_char]
            out.append(stim.PauliString(mask))
    return out


def _all_paulis_up_to_weight(
    n: int, weights: tuple[int, ...], paulis: str = "XYZ",
) -> list[stim.PauliString]:
    """Identity + all Paulis on n qubits whose weight is in `weights`. Spans
    every qubit position (not just a subset). Use with stabilizers padded to
    length n so the algorithm sees a true distance constraint over the full
    code space (data + extra qubit)."""
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


def make_extension_inputs(
    initial_stabilizers: list[stim.PauliString],
    total_n_qubits: int,
    error_weights: tuple[int, ...] = (1,),
    paulis: str = "XYZ",
) -> tuple[list[stim.PauliString], list[stim.PauliString]]:
    """Pre-pad `initial_stabilizers` to length `total_n_qubits` and construct
    an error set covering all Paulis of weight in `error_weights` over the full
    qubit register. The returned (stabilizers, errors) should be passed to
    random_walk_extend / beam_search_extend with `extra_qubits=0` (the
    padding is already done here).

    For a code with target distance d, pass `error_weights=tuple(range(1, (d-1)//2 + 1))`
    so that Knill-Laflamme on this error set forces distance >= d."""

    n_data = max(len(g) for g in initial_stabilizers)
    if total_n_qubits < n_data:
        raise ValueError(
            f"total_n_qubits={total_n_qubits} is smaller than the existing "
            f"stabilizer length {n_data}."
        )
    pad_len = total_n_qubits - n_data
    if pad_len > 0:
        pad = stim.PauliString("_" * pad_len)
        padded = [g + pad for g in initial_stabilizers]
    else:
        padded = list(initial_stabilizers)
    errors = _all_paulis_up_to_weight(total_n_qubits, error_weights, paulis=paulis)
    return padded, errors


def _fh_symmetries(n: int) -> list[stim.PauliString]:
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
    initial_stabilizers: list[stim.PauliString]
    errors: list[stim.PauliString]
    extra_support: stim.PauliString | None = None
    extra_qubits: int = 0
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
    # distance verification (max Pauli weight to search; for KL-implied d=2t+1
    # set this to 2t so a >=2t+1 lower bound is reported when the search finds no
    # logical of weight <= 2t)
    distance_max_weight: int = 3
    # hardware connectivity (dict[int, set[int]]) — added stabilizers must have
    # support that's a connected subgraph in this graph. None = all-to-all.
    connectivity: dict | None = None


def _scenarios() -> list[Scenario]:
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
            extra_qubits=1,
            max_stabilizers_per_walk=2, max_walks=20,
        ),
        Scenario(
            name="FH [[4,2]] + all weight-1 (slide 34)",
            initial_stabilizers=_fh_symmetries(4),
            errors=_all_single_qubit_paulis(4),
            extra_qubits=3,
            max_stabilizers_per_walk=4, max_walks=30,
        ),
        Scenario(
            name="FH [[8,6]] + weight-1 X (slide 14, beam)",
            initial_stabilizers=_fh_symmetries(8),
            errors=_all_single_qubit_paulis(8, "X"),
            extra_qubits=2,
            method="beam_search",
            max_stabilizers=3, beam_width=8, n_expansions_per_slot=4,
            n_solution_samples=8,
        ),
        Scenario(
            name="FH [[16,14]] + weight-1 X (stress, beam)",
            initial_stabilizers=_fh_symmetries(16),
            errors=_all_single_qubit_paulis(16, "X"),
            extra_qubits=4,
            method="beam_search",
            max_stabilizers=4, beam_width=8, n_expansions_per_slot=4,
            n_solution_samples=8,
        ),
        # ---- d=3 target scenarios: errors on ALL qubits, no internal padding ----
        # These force the code's actual distance >= 3 by including weight-1 Paulis
        # on every qubit (data + extra qubit) in the input error set.
    ] + _d3_target_scenarios()


def _d3_target_scenarios() -> list["Scenario"]:
    """d=3 target scenarios: weight-1 Paulis (X/Y/Z) on all qubits, stabilizers
    pre-padded so extra_qubits=0 is correct. Ancilla count is chosen large
    enough that the algorithm can find a valid extension; can be tuned smaller
    once we know what's necessary."""
    scenarios: list[Scenario] = []

    # FH [[4,2]] -> aim for d=3 with 2 extra qubits (6 total qubits).
    stabs, errs = make_extension_inputs(_fh_symmetries(4), total_n_qubits=6, error_weights=(1,))
    scenarios.append(Scenario(
        name="FH [[4,2]] -> d=3 (6 qubits, beam)",
        initial_stabilizers=stabs, errors=errs,
        extra_qubits=0,
        method="beam_search",
        max_stabilizers=6, beam_width=8, n_expansions_per_slot=4,
        n_solution_samples=8,
    ))

    # FH [[8,6]] -> aim for d=3 with 4 extra qubits (12 total qubits).
    stabs, errs = make_extension_inputs(_fh_symmetries(8), total_n_qubits=12, error_weights=(1,))
    scenarios.append(Scenario(
        name="FH [[8,6]] -> d=3 (12 qubits, beam)",
        initial_stabilizers=stabs, errors=errs,
        extra_qubits=0,
        method="beam_search",
        max_stabilizers=8, beam_width=8, n_expansions_per_slot=4,
        n_solution_samples=8,
    ))

    # FH [[16,14]] -> aim for d=3 with 6 extra qubits (22 total qubits).
    stabs, errs = make_extension_inputs(_fh_symmetries(16), total_n_qubits=22, error_weights=(1,))
    scenarios.append(Scenario(
        name="FH [[16,14]] -> d=3 (22 qubits, beam)",
        initial_stabilizers=stabs, errors=errs,
        extra_qubits=0,
        method="beam_search",
        max_stabilizers=10, beam_width=8, n_expansions_per_slot=4,
        n_solution_samples=8,
    ))

    return scenarios + _d5_target_scenarios() + _connectivity_scenarios()


def _connectivity_scenarios() -> list[Scenario]:
    """Hardware-tailored scenarios: identical to the d=3 targets above but with
    linear nearest-neighbor connectivity, so added stabilizers must have support
    that's a connected subgraph in the 1D line. Note: the initial FH symmetry
    generators (ZIZI...) have non-NN-connected support and so already violate
    the constraint -- the added stabilizers are constrained, but the starting
    Hamiltonian-derived ones aren't (they're a property of the problem, not the
    algorithm's choice). Real-device deployment of the resulting code would
    still need SWAPs to measure the FH symmetry stabilizers."""
    scenarios: list[Scenario] = []

    # FH [[4,2]] -> d=3 under linear NN on 6 qubits.
    stabs, errs = make_extension_inputs(_fh_symmetries(4), total_n_qubits=6, error_weights=(1,))
    scenarios.append(Scenario(
        name="FH [[4,2]] -> d=3, linear NN (6 qubits)",
        initial_stabilizers=stabs, errors=errs,
        extra_qubits=0,
        method="beam_search",
        max_stabilizers=6, beam_width=8, n_expansions_per_slot=4,
        n_solution_samples=8,
        connectivity=connect.linear(6),
    ))

    return scenarios


def _d5_target_scenarios() -> list["Scenario"]:
    """d=5 target scenarios: weight-<=2 Paulis on all qubits.

    By the Knill-Laflamme theorem, if the walk succeeds (all input error
    products satisfy KL), the resulting code has distance >= 5 by construction.
    distance_max_weight=4 lets compute_distance verify this explicitly (search
    weights 1..4, find no logical, report >=5).

    Only FH [[4,2]] -> d=5 is in the default bench. The bigger cases
    (FH [[8,6]], FH [[16,14]]) did NOT converge within beam_width=8,
    n_expansions=4, max_stabilizers=10..12 in our exploration runs --
    [[8,6]] -> d=5 ran for 24 minutes with 60 uncorrectables remaining and
    0/8 successful walks. They're left commented below; uncomment to retry
    with a wider beam, more expansions, or more extra qubits."""
    scenarios: list[Scenario] = []

    # FH [[4,2]] -> aim for d=5 with 4 extra qubits (8 total).
    # Succeeds but collapses to k=0 (all dimensions consumed by stabilizers).
    # For a k>=1 d=5 code, allocate more extra qubits.
    stabs, errs = make_extension_inputs(_fh_symmetries(4), total_n_qubits=8, error_weights=(1, 2))
    scenarios.append(Scenario(
        name="FH [[4,2]] -> d=5 (8 qubits, beam)",
        initial_stabilizers=stabs, errors=errs,
        extra_qubits=0,
        method="beam_search",
        max_stabilizers=8, beam_width=8, n_expansions_per_slot=4,
        n_solution_samples=8,
        distance_max_weight=4,
    ))

    # ---- Currently-failing d=5 stretch cases (uncomment to retry tuned) ----
    #
    # stabs, errs = make_extension_inputs(_fh_symmetries(8), total_n_qubits=14, error_weights=(1, 2))
    # scenarios.append(Scenario(
    #     name="FH [[8,6]] -> d=5 (14 qubits, beam)",
    #     initial_stabilizers=stabs, errors=errs,
    #     extra_qubits=0,
    #     method="beam_search",
    #     max_stabilizers=10, beam_width=8, n_expansions_per_slot=4,
    #     n_solution_samples=8,
    #     distance_max_weight=4,
    # ))
    #
    # stabs, errs = make_extension_inputs(_fh_symmetries(16), total_n_qubits=24, error_weights=(1, 2))
    # scenarios.append(Scenario(
    #     name="FH [[16,14]] -> d=5 (24 qubits, beam)",
    #     initial_stabilizers=stabs, errors=errs,
    #     extra_qubits=0,
    #     method="beam_search",
    #     max_stabilizers=12, beam_width=8, n_expansions_per_slot=4,
    #     n_solution_samples=8,
    #     distance_max_weight=4,
    # ))

    return scenarios


def _run(sc: Scenario):
    t0 = time.perf_counter()
    if sc.method == "beam_search":
        result = beam_search_extend(
            sc.initial_stabilizers, sc.errors,
            extra_support=sc.extra_support,
            extra_qubits=sc.extra_qubits,
            max_stabilizers=sc.max_stabilizers,
            beam_width=sc.beam_width,
            n_expansions_per_slot=sc.n_expansions_per_slot,
            n_solution_samples=sc.n_solution_samples if sc.n_solution_samples > 1 else 8,
            seed_val=sc.seed_val,
            distance_max_weight=sc.distance_max_weight,
            connectivity=sc.connectivity,
        )
    else:
        result = random_walk_extend(
            sc.initial_stabilizers, sc.errors,
            extra_support=sc.extra_support,
            extra_qubits=sc.extra_qubits,
            max_stabilizers_per_walk=sc.max_stabilizers_per_walk,
            max_walks=sc.max_walks,
            n_solution_samples=sc.n_solution_samples,
            seed_val=sc.seed_val,
            distance_max_weight=sc.distance_max_weight,
            connectivity=sc.connectivity,
        )
    elapsed = time.perf_counter() - t0
    n = max(len(g) for g in result.code) if result.code else 0
    k = n - len(result.code)
    return n, k, result, elapsed


def _format_distance(result):
    if result.distance is None:
        return "  -"
    if result.distance_is_exact:
        return f"{result.distance:>4d}"
    # Not exact: compute_distance returns max_weight + 1, meaning the true
    # distance is >= max_weight + 1.
    return f">={result.distance}"


def main():
    print(
        f"{'scenario':<46} {'n':>3} {'k':>3} {'d':>4} {'added':>5} "
        f"{'rem':>4} {'ok?':>4} {'time(s)':>8} {'succ/walks':>11}",
        flush=True,
    )
    print("-" * 110, flush=True)
    for sc in _scenarios():
        n, k, result, elapsed = _run(sc)
        ok = "yes" if result.succeeded else "NO"
        d_str = _format_distance(result)
        print(
            f"{sc.name:<46} {n:>3d} {k:>3d} {d_str:>4} {result.n_stabilizers_added:>5d} "
            f"{result.uncorrectables_remaining:>4d} {ok:>4} {elapsed:>8.2f} "
            f"{result.successful_walks:>3d}/{result.n_walks:<7d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
