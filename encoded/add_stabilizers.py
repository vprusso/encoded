from typing import Collection
from warnings import warn
from dataclasses import dataclass, field
import itertools
import functools
import time
from random import randrange, seed, sample
from copy import deepcopy
import numpy as np
import stim
from encoded.decompose_operators import generators_to_matrix
from encoded.binary_linalg import solve_boolean_system, _boolean_rref, enumerate_all_solutions, system_has_solutions, sample_random_solution, sample_random_solution_biased
from encoded.connectivity import has_connected_support


class NoConnectivityValidSample(Exception):
    """Raised by the max-coverage selector when no candidate solution within the
    sample budget satisfies the connectivity constraint. Caught by the walk loops
    to skip that expansion."""

def metric_tensor(nq: int) -> np.ndarray:
    id_nq = np.eye(nq).astype(bool)
    zeros_nq = np.zeros((nq, nq)).astype(bool)
    return np.vstack((
        np.hstack((zeros_nq, id_nq)),
        np.hstack((id_nq, zeros_nq))
    ))


def _form_linear_system(generators: list[stim.PauliString], errors: list[stim.PauliString]) -> tuple[np.ndarray, np.ndarray]:
    """Form the linear system that requires the new generator to commute with the existing
    generators and anticommute with the given errors."""

    max_nq_gen = max([len(gen) for gen in generators])
    max_nq_err = max([len(err) for err in errors])
    max_nq = max(max_nq_gen, max_nq_err)

    generator_matrix = generators_to_matrix(generators, max_nq=max_nq)
    error_matrix = generators_to_matrix(errors, max_nq=max_nq)
    try:
        pstring_matrix = np.hstack((generator_matrix, error_matrix))
    except ValueError as e:
        print("Failed on the hstack.")
        print("generators=")
        for gen in generators:
            print(gen)
        print("errors=")
        for err in errors:
            print(err)
        print("generator_matrix=")
        print(generator_matrix)
        print(f"with shape {generator_matrix.shape}")
        print("error_matrix=")
        print(error_matrix)
        print(f"with shape {error_matrix.shape}")
        raise e
    nq = max([len(ps) for ps in generators + errors])
    lamb = metric_tensor(nq)
    A = pstring_matrix.T @ lamb
    b = np.array([False] * len(generators) + [True] * len(errors))
    return A, b


def add_stabilizer(
    generators: list[stim.PauliString], errors: list[stim.PauliString], extra_support: stim.PauliString | None=None,
    verbose: bool=False, choose_solution_randomly: bool=False,
    _solution_callback=None,
) -> list[stim.PauliString]:
    """Add a stabilizer to the code given a set of errors the new stabilizer should anticommute with.

    Arguments:
    generators - Generators of the current code.
    errors - The errors that the new stabilizer should anticommute with.
    extra_suport - If None, no support is added on extra qubit. Otherwise, we add the Pauli to the new stabilizer.
    verbose - Whether to print information while solving the system.
    choose_solution_randomly - Whether to choose one of the solutions randomly (True) or choose the
    solution with the lowest Pauli weight (False). Defaults to False.
    _solution_callback - Internal hook: a callable (A_rref, b_rref) -> symplectic_vector
    that picks a solution from the affine subspace. When set, overrides both
    choose_solution_randomly and the min-weight path. Used by smart walk heuristics."""

    A, b = _form_linear_system(generators, errors)
    if verbose:
        print("A=\n", A)
        print("b=\n", b)
    A_rref, b_rref = _boolean_rref(A, b)
    if verbose:
        print("A_rref=\n", A_rref)
        print("b_rref=\n", b_rref)
    if _solution_callback is not None:
        x = _solution_callback(A_rref, b_rref)
        new_generator = stim.PauliString.from_numpy(
            xs=x[:x.size // 2], zs=x[x.size // 2:],
        )
    elif choose_solution_randomly:
        # Sample one solution from the affine subspace without materializing
        # 2^free_vars candidates. Bit-identical to the old enumerate+randrange
        # pattern under the same seed.
        x = sample_random_solution(A_rref, b_rref)
        new_generator = stim.PauliString.from_numpy(
            xs=x[:x.size // 2], zs=x[x.size // 2:],
        )
    else:
        # Lowest-weight selection still needs all candidates (finding the
        # min-weight solution to a boolean system is NP-hard — slide 17).
        solutions = enumerate_all_solutions(A_rref, b_rref)
        candidate_strings = []
        for x in solutions:
            xs = x[:x.size // 2]
            zs = x[(x.size // 2):]
            pstring = stim.PauliString.from_numpy(xs=xs, zs=zs)
            candidate_strings.append(pstring)
        new_generator = min(candidate_strings, key=lambda ps: ps.weight)
    if extra_support is not None:
        new_generator += extra_support
        # Add identity to all the original generators.
        id_extra = stim.PauliString("I" * len(extra_support))
        old_generators = [gen + id_extra for gen in generators]
    else:
        old_generators = deepcopy(generators)
    for err in errors:
        assert not new_generator.commutes(err), f"[{new_generator}, {err}] = 0"
    new_generators = old_generators + [new_generator]
    return new_generators


def generate_stabilizer_elements(generators: list[stim.PauliString]) -> list[stim.PauliString]:
    nq = max([len(ps) for ps in generators])
    elements = []
    for string in itertools.chain.from_iterable(itertools.combinations(generators, r) for r in range(len(generators) + 1)):
        elements.append(
            functools.reduce(lambda a, b: a * b, string, stim.PauliString('_' * nq))
        )
    return elements


def stim_strings_equal_up_to_phase(ps_a: stim.PauliString, ps_b: stim.PauliString) -> bool:
    return list(ps_a) == list(ps_b)


def is_in_stabilizer_group(operator: stim.PauliString, generators: list[stim.PauliString]) -> bool:
    """Test if `operator` (ignoring phase) lies in the stabilizer group generated by
    `generators`. O(k * n^2) via Gaussian elimination over GF(2) — much faster than
    enumerating all 2^k group elements once k is more than a handful.

    An operator P is in <g_1, ..., g_k> iff its symplectic vector v lies in the column
    span of the symplectic generator matrix M, i.e. iff the linear system M @ c = v has
    a solution over GF(2)."""

    nq = max(len(gen) for gen in generators)
    if len(operator) > nq:
        # The group lives in nq qubits; an operator with support beyond that can't be in it.
        return False
    if len(operator) < nq:
        operator = operator + stim.PauliString("_" * (nq - len(operator)))

    M = generators_to_matrix(generators, max_nq=nq)
    x, z = operator.to_numpy()
    v = np.concatenate([x, z])
    M_rref, v_rref = _boolean_rref(M, v)
    return system_has_solutions(M_rref, v_rref)


def _legacy_group_membership_check(operator: stim.PauliString, generators: list[stim.PauliString]) -> bool:
    """Original O(2^k) implementation kept for cross-validating `is_in_stabilizer_group`
    and for benchmarking. Do not use in production paths."""

    nq = max([len(gen) for gen in generators])
    if len(operator) < nq:
        new_operator = operator + stim.PauliString("_" * (nq - len(operator)))
    else:
        new_operator = operator

    equality_tests = []
    for ps in generate_stabilizer_elements(generators):
        test = stim_strings_equal_up_to_phase(new_operator, ps)
        equality_tests.append(test)
    return any(equality_tests)


def knill_laflamme_cost_function(generators: list[stim.PauliString], errors: list[float], weights: list[float]) -> float:
    """Cost function from the RL paper."""

    total_loss = 0.
    for weight, err in zip(weights, errors):
        anticommutation_tests = [not gen.commutes(err) for gen in generators]
        in_stabilizer_group = is_in_stabilizer_group(err, generators)
        if any(anticommutation_tests) or in_stabilizer_group:
            # The error is correctable, so K_mu = 1.
            total_loss -= weight
    return total_loss


def knill_laflamme_correctable_cost_function(generators: list[stim.PauliString], errors: list[float], weights: list[float]) -> float:
    """Check pairs of errors."""

    total_loss = 0.
    for weight1, err1 in zip(weights, errors):
        for weight2, err2 in zip(weights, errors):
            weight = weight1 * weight2
            err = err1 * err2
            anticommutation_tests = [not gen.commutes(err) for gen in generators]
            in_stabilizer_group = is_in_stabilizer_group(err, generators)
            if any(anticommutation_tests) or in_stabilizer_group:
                # The error is correctable, so K_mu = 1.
                total_loss -= weight
    return total_loss


def get_uncorrectable_errors(
    generators: list[stim.PauliString], errors: list[stim.PauliString],
) -> list[stim.PauliString]:
    """Get the products `e_i * e_j` that the code cannot correct (Knill-Laflamme
    violators).

    Vectorized: build the (N_errors x n_gens) anti-commutation matrix in one
    numpy boolean matrix product via the symplectic formula
        AC[i, m] = (e_i.x . g_m.z) XOR (e_i.z . g_m.x)   (mod 2)
    Each error has a "syndrome" given by its row of AC. The product e_i * e_j
    anticommutes with at least one generator iff AC[i] != AC[j] (since
    anticommutation is linear in the symplectic representation). Pairs with
    different syndromes are therefore correctable and can be skipped; only
    same-syndrome pairs need the (relatively expensive) is_in_stabilizer_group
    check, dropping the membership-check count from O(N^2) to roughly
    O(N^2 / 2^k) where k is the number of stabilizers.

    The outer (i, j) iteration order matches the legacy implementation kept as
    `_legacy_get_uncorrectable_errors`, so seeded random walks behave
    bit-identically."""

    if not generators or not errors:
        return []

    n = max(
        max(len(g) for g in generators),
        max(len(e) for e in errors),
    )
    n_gens = len(generators)
    n_errs = len(errors)

    g_x = np.zeros((n_gens, n), dtype=np.uint8)
    g_z = np.zeros((n_gens, n), dtype=np.uint8)
    for m, g in enumerate(generators):
        gp = g if len(g) == n else g + stim.PauliString("_" * (n - len(g)))
        x, z = gp.to_numpy()
        g_x[m] = x
        g_z[m] = z

    e_x = np.zeros((n_errs, n), dtype=np.uint8)
    e_z = np.zeros((n_errs, n), dtype=np.uint8)
    padded_errors: list[stim.PauliString] = []
    for i, e in enumerate(errors):
        ep = e if len(e) == n else e + stim.PauliString("_" * (n - len(e)))
        padded_errors.append(ep)
        x, z = ep.to_numpy()
        e_x[i] = x
        e_z[i] = z

    # Symplectic anticommutation: integer-sum of element-wise AND, taken mod 2.
    ac = ((e_x @ g_z.T) ^ (e_z @ g_x.T)) & 1   # shape (N, K)

    # Pack each row of ac into an integer for fast equality.
    if n_gens <= 64:
        powers = (1 << np.arange(n_gens, dtype=np.uint64))
        row_ids = (ac.astype(np.uint64) @ powers).tolist()
    else:
        row_ids = [hash(tuple(row.tolist())) for row in ac]

    # Bucket error indices by their syndrome. Buckets preserve insertion order
    # (Python dicts since 3.7), so iterating syndrome_groups[row_ids[i]] yields
    # j's in ascending order, matching the legacy nested-loop pattern.
    syndrome_groups: dict[int, list[int]] = {}
    for i, rid in enumerate(row_ids):
        syndrome_groups.setdefault(rid, []).append(i)

    uncorrectable: list[stim.PauliString] = []
    for i in range(n_errs):
        for j in syndrome_groups[row_ids[i]]:
            e_prod = padded_errors[i] * padded_errors[j]
            if not is_in_stabilizer_group(e_prod, generators):
                uncorrectable.append(e_prod)
    return uncorrectable


def _legacy_get_uncorrectable_errors(
    generators: list[stim.PauliString], errors: list[stim.PauliString],
) -> list[stim.PauliString]:
    """Original O(N^2 * (K + membership)) implementation. Kept for cross-validating
    the vectorized `get_uncorrectable_errors` and for benchmarking."""

    uncorrectable_errs = []
    for i, e_i in enumerate(errors):
        for j, e_j in enumerate(errors):
            e = e_i * e_j
            anti_commute_tests = [not e.commutes(g) for g in generators]
            in_group = is_in_stabilizer_group(e, generators)
            if not (any(anti_commute_tests) or in_group):
                uncorrectable_errs.append(e)
    return uncorrectable_errs


def build_code_randomly(
    generators: list[stim.PauliString], errors: list[stim.PauliString], extra_support: stim.PauliString | None=None,
    max_iter: int = 1_000, seed_val: int=137, errors_per_round=1
) -> list[stim.PauliString]:
    """Build a code by randomly picking the error(s) that the new stabilizer will anticommute with."""

    seed(seed_val)
    new_generators = deepcopy(generators)
    uncorrectables = get_uncorrectable_errors(new_generators, errors)
    iters = 0
    while len(uncorrectables) != 0:
        # i = randrange(len(uncorrectables))
        inds = sample(range(len(uncorrectables)), errors_per_round)
        errs = [uncorrectables[i] for i in inds]
        new_generators = add_stabilizer(new_generators, errs, extra_support=extra_support, verbose=False)
        # print("New stabilizer:", new_generators[-1])
        uncorrectables = get_uncorrectable_errors(new_generators, errors)
        # If there are now more qubits in the stabilizers than the erros, add qubits to the errors.
        nq = max(len(gen) for gen in new_generators)
        for i in range(len(uncorrectables)):
            if len(uncorrectables[i]) < nq:
                uncorrectables[i] += stim.PauliString("_" * (nq - len(uncorrectables[i])))
        iters += 1
        if iters > max_iter:
            warn(f"Exceeded max iterations ({max_iter}).")
            break
    return new_generators


def prune_duplicate_pauli_strings(strings: list[stim.PauliString]) -> list[stim.PauliString]:
    """stim.PauliString objects are not hashable, so you can't make a set of them. This function
    removed duplicated from the list."""

    new_strings = []
    for pstring in strings:
        if not any([pstring == ps for ps in new_strings]):
            new_strings.append(pstring)
    return new_strings


def all_new_codes_for_errors(
    generators: Collection[stim.PauliString], errors: list[stim.PauliString],
    extra_support: stim.PauliString | None = None
) -> list[stim.PauliString]:
    """Given a code and a set of errors to correct, enumerate all new stabilizers that we could add."""

    A, b = _form_linear_system(generators, errors)
    A_rref, b_rref = _boolean_rref(A, b)
    solutions = enumerate_all_solutions(A_rref, b_rref)
    candidate_strings = []
    for x in solutions:
        xs = x[:x.size // 2]
        zs = x[(x.size // 2):]
        pstring = stim.PauliString.from_numpy(xs=xs, zs=zs)
        candidate_strings.append(pstring)
    new_codes = []
    for new_generator in candidate_strings:
        if extra_support is not None:
            new_generator += extra_support
            # Add identity to all the original generators.
            id_extra = stim.PauliString("I" * len(extra_support))
            old_generators = [gen + id_extra for gen in generators]
        else:
            old_generators = deepcopy(generators)
        for err in errors:
            assert not new_generator.commutes(err), f"[{new_generator}, {err}] = 0"
        new_generators = old_generators + [new_generator]
        new_codes.append(new_generators)
    return new_codes


def random_depth_first_search(
    stabilizers: list[stim.PauliString], errors: list[stim.PauliString],
    extra_support: stim.PauliString | None=None,
    steps: int = 1, max_tries: int = 10, seed_val: int=137
) -> list[stim.PauliString]:
    """Do a depth-first search, at each step picking a random error to anticommute with
    and a random stabilizer from the list of systems of equations.
    
    Arguments:
    stabilizer - The initial stabilizers of the code.
    errors - The errors the new code should correct.
    steps - The number of new stabilizers to add.
    max_tries - The number of times to randomly descend down the tree.
    seed_val - Value for seeding RNG."""

    seed(seed_val)

    best_code = deepcopy(stabilizers)
    best_num_uncorredtable = len(get_uncorrectable_errors(stabilizers, errors))

    for i in range(max_tries):
        # print(f"On try {i}.")
        temp_stabilizers = deepcopy(stabilizers)
        for j in range(steps):
            # print(f"On step {j}.")
            # for stab in temp_stabilizers:
            #     print(stab)
            uncorrectables = get_uncorrectable_errors(temp_stabilizers, errors)
            # print(f"Code has {len(uncorrectables)} uncorrectable errors.")
            if len(uncorrectables) == 0:
                break
            new_err = uncorrectables[randrange(0, len(uncorrectables))]
            temp_stabilizers = add_stabilizer(
                temp_stabilizers, [new_err], extra_support=extra_support, choose_solution_randomly=True
            )
        uncorrectables = get_uncorrectable_errors(temp_stabilizers, errors)
        if len(uncorrectables) < best_num_uncorredtable:
            best_code = deepcopy(temp_stabilizers)
            best_num_uncorredtable = len(uncorrectables)
    return best_code


def compute_distance(
    stabilizers: list[stim.PauliString], max_weight: int = 3,
) -> tuple[int, bool]:
    """Distance of the stabilizer code generated by `stabilizers`.

    Distance d = minimum weight of a Pauli operator that (a) commutes with every
    stabilizer and (b) is not itself in the stabilizer group — i.e., the lightest
    non-trivial logical operator.

    Searches Paulis of weight 1, 2, ..., max_weight in order. Returns (d, exact):
      - If a logical of weight w <= max_weight is found: (w, True).
      - If none is found by weight max_weight: (max_weight + 1, False), meaning
        "distance is at least max_weight + 1; we did not search further."

    Computing distance is NP-hard in general. This brute-forces enumeration of
    weight-w Paulis, which is O(3^w * C(n,w)) per weight — tractable up to n ~ 25
    and max_weight ~ 5. Pick max_weight to match what your application needs."""

    n = max(len(g) for g in stabilizers)

    for w in range(1, max_weight + 1):
        for positions in itertools.combinations(range(n), w):
            for paulis in itertools.product((1, 2, 3), repeat=w):  # stim: 1=X, 2=Y, 3=Z
                mask = [0] * n
                for pos, p in zip(positions, paulis):
                    mask[pos] = p
                candidate = stim.PauliString(mask)
                if not all(candidate.commutes(g) for g in stabilizers):
                    continue
                if is_in_stabilizer_group(candidate, stabilizers):
                    continue
                return w, True

    return max_weight + 1, False


@dataclass
class PerWalkResult:
    """Outcome of a single walk down the (which-error × which-solution) tree."""
    stabilizers_added: int
    uncorrectables_remaining: int

    @property
    def succeeded(self) -> bool:
        return self.uncorrectables_remaining == 0


@dataclass
class WalkResult:
    """Aggregate result of `random_walk_extend` / `beam_search_extend`.

    `code` is the best stabilizer set found, picked by the lexicographic min of
    (uncorrectables_remaining, stabilizers_added) so that successful walks beat
    failed ones, and among successful walks, shorter extensions beat longer ones
    (preserving more logical qubits).

    `distance` is the code's [[n,k,d]] distance — computed by `compute_distance`
    only when `succeeded` is True. None means the search did not compute it (e.g.
    the extension failed). `distance_is_exact=False` means the search ran up to
    its max_weight without finding a logical, so the reported value is a lower
    bound (>=) on the true distance."""
    code: list[stim.PauliString]
    succeeded: bool
    uncorrectables_remaining: int
    n_stabilizers_added: int
    n_walks: int
    successful_walks: int
    walks: list[PerWalkResult] = field(default_factory=list)
    distance: int | None = None
    distance_is_exact: bool = True


def _build_max_coverage_selector(
    uncorrectables: list[stim.PauliString],
    extra_support: stim.PauliString | None,
    n_samples: int,
    connectivity: dict[int, set[int]] | None = None,
    sampling_bias: float = 0.5,
):
    """Build a solution-selection callback for add_stabilizer that, given the
    affine solution space (A_rref, b_rref), samples `n_samples` candidate
    solutions and returns the one that anticommutes with the most
    `uncorrectables`. Score includes any `extra_support` qubits appended after
    the linear system is solved.

    When `connectivity` is given, samples whose non-identity support violates the
    connectivity graph are rejected before scoring. If no connectivity-valid
    sample is found within `n_samples` tries, raises NoConnectivityValidSample
    so the caller can skip this expansion.

    `sampling_bias` controls the symplectic-vector sampling distribution:
        0.5  = uniform (default; bit-identical to old behavior under same seed)
        <0.5 = biased toward sparser solutions (lower Pauli weight)
        >0.5 = biased toward denser solutions"""

    def selector(A_rref: np.ndarray, b_rref: np.ndarray) -> np.ndarray:
        best_x = None
        best_score = -1
        for _ in range(n_samples):
            if sampling_bias == 0.5:
                x = sample_random_solution(A_rref, b_rref)
            else:
                x = sample_random_solution_biased(A_rref, b_rref, bias=sampling_bias)
            candidate = stim.PauliString.from_numpy(
                xs=x[:x.size // 2], zs=x[x.size // 2:],
            )
            full = candidate + extra_support if extra_support is not None else candidate
            if connectivity is not None and not has_connected_support(full, connectivity):
                continue
            score = sum(1 for e in uncorrectables if not full.commutes(e))
            if score > best_score:
                best_score = score
                best_x = x
        if best_x is None:
            raise NoConnectivityValidSample(
                f"No connectivity-valid sample found in {n_samples} draws."
            )
        return best_x

    return selector


def random_walk_extend(
    stabilizers: list[stim.PauliString],
    errors: list[stim.PauliString],
    extra_support: stim.PauliString | None = None,
    extra_qubits: int = 0,
    max_stabilizers_per_walk: int = 1,
    max_walks: int = 10,
    n_solution_samples: int = 1,
    seed_val: int = 137,
    distance_max_weight: int = 3,
    verbose: bool = False,
    connectivity: dict[int, set[int]] | None = None,
    sampling_bias: float = 0.5,
) -> WalkResult:
    """Slide-33 random walk: at each step pick a random uncorrectable error product
    and a random solution from the resulting affine system, add it to the group,
    repeat up to `max_stabilizers_per_walk` times per walk and `max_walks` walks
    total. Returns the best code found together with per-walk telemetry.

    Differs from `random_depth_first_search` in three ways: (1) returns a structured
    result with success signal + telemetry instead of just the code, (2) tie-breaks
    "best" by (uncorrectables_remaining, stabilizers_added) so a successful walk with
    fewer added stabilizers wins over a longer successful walk, (3) the original
    function is left untouched so existing experiments keep working.

    Arguments:
    stabilizers - The initial stabilizer generators.
    errors - The single-Pauli errors the extended code should make correctable.
    extra_support - If given, every new generator gets this Pauli appended on a new
        extra qubit; all prior generators are padded with identity. So a walk of
        depth `d` with single-qubit `extra_support` adds `d` extra qubits. Mutually
        exclusive with `extra_qubits`.
    extra_qubits - Number of shared extra qubits to pre-allocate up front. With
        `extra_qubits=m`, every initial generator is padded with `m` identities and
        every error gets `m` identities appended (errors act only on the original data
        qubits). Each added stabilizer can then place arbitrary support on any of the
        n+m qubits, so multiple stabilizers can SHARE the same extra qubit — matching the
        slide 14 hand-design which uses 2 extra qubits for 2 added stabilizers on FH [[8,6]]
        rather than 2 extra qubits per stabilizer.
    max_stabilizers_per_walk - Cap on stabilizers added per walk. Walks can finish
        early if uncorrectables hit zero before this cap.
    max_walks - Number of independent walks (the tree exploration budget).
    n_solution_samples - Smart-walk knob. At each step, sample this many candidate
        solutions from the affine subspace and pick the one that anticommutes with
        the most currently-uncorrectable error pairs (greedy coverage). Default 1 =
        pure random walk per slide 33 (bit-identical to the original under the same
        seed). Higher values trade per-step CPU for a much better hit rate at the
        optimal extension — on FH [[8,6]] the slide-14 [[10,6]] target is hit ~4%
        of the time at n_solution_samples=1 vs. nearly every walk at 64+.
    seed_val - RNG seed.
    distance_max_weight - Upper weight to search when computing the code's
        distance on a successful walk. Default 3 (cheap, sufficient to confirm a
        d<=3 detection / d=3 correction code). Set higher (4-5) to verify d=5
        codes; cost grows as O(3^w * C(n,w)) per weight."""

    if extra_qubits > 0 and extra_support is not None:
        raise ValueError(
            "Specify either `extra_qubits` (shared extra-qubit register, set once at the "
            "start of each walk) or `extra_support` (one fresh extra qubit per step), not both."
        )

    if extra_qubits > 0:
        pad = stim.PauliString("_" * extra_qubits)
        stabilizers = [g + pad for g in stabilizers]
        errors = [e + pad for e in errors]

    seed(seed_val)
    initial_count = len(stabilizers)
    initial_uncorrectables = len(get_uncorrectable_errors(stabilizers, errors))

    best_code = deepcopy(stabilizers)
    best_key: tuple[int, int] = (initial_uncorrectables, 0)  # (remaining, added)
    walks: list[PerWalkResult] = []

    # If connectivity is set, always go through the max-coverage selector so we
    # can filter by connectivity; bump the sample budget to give the filter room
    # to find a valid candidate.
    effective_samples = max(n_solution_samples, 16) if connectivity is not None else n_solution_samples

    t_start = time.perf_counter()
    for walk_idx in range(max_walks):
        temp = deepcopy(stabilizers)
        for _ in range(max_stabilizers_per_walk):
            uncorrectables = get_uncorrectable_errors(temp, errors)
            if not uncorrectables:
                break
            new_err = uncorrectables[randrange(0, len(uncorrectables))]
            if effective_samples > 1 or connectivity is not None or sampling_bias != 0.5:
                selector = _build_max_coverage_selector(
                    uncorrectables, extra_support, effective_samples,
                    connectivity=connectivity,
                    sampling_bias=sampling_bias,
                )
                try:
                    temp = add_stabilizer(
                        temp, [new_err], extra_support=extra_support, _solution_callback=selector,
                    )
                except NoConnectivityValidSample:
                    # No connectivity-respecting solution found; this walk stops here.
                    break
            else:
                temp = add_stabilizer(
                    temp, [new_err], extra_support=extra_support, choose_solution_randomly=True,
                )

        added = len(temp) - initial_count
        remaining = len(get_uncorrectable_errors(temp, errors))
        walks.append(PerWalkResult(stabilizers_added=added, uncorrectables_remaining=remaining))

        key = (remaining, added)
        if key < best_key:
            best_code = deepcopy(temp)
            best_key = key

        if verbose:
            n_succ = sum(1 for w in walks if w.succeeded)
            print(
                f"[random_walk_extend walk={walk_idx + 1}/{max_walks} "
                f"elapsed={time.perf_counter() - t_start:.1f}s] "
                f"best: remaining={best_key[0]} added={best_key[1]} | "
                f"this walk: remaining={remaining} added={added} | "
                f"succeeded {n_succ}/{walk_idx + 1}",
                flush=True,
            )

    succeeded = best_key[0] == 0
    distance, distance_exact = (None, True)
    if succeeded:
        distance, distance_exact = compute_distance(best_code, max_weight=distance_max_weight)
    return WalkResult(
        code=best_code,
        succeeded=succeeded,
        uncorrectables_remaining=best_key[0],
        n_stabilizers_added=best_key[1],
        n_walks=max_walks,
        successful_walks=sum(1 for w in walks if w.succeeded),
        walks=walks,
        distance=distance,
        distance_is_exact=distance_exact,
    )


def beam_search_extend(
    stabilizers: list[stim.PauliString],
    errors: list[stim.PauliString],
    extra_support: stim.PauliString | None = None,
    extra_qubits: int = 0,
    max_stabilizers: int = 4,
    beam_width: int = 8,
    n_expansions_per_slot: int = 4,
    n_solution_samples: int = 8,
    seed_val: int = 137,
    distance_max_weight: int = 3,
    verbose: bool = False,
    connectivity: dict[int, set[int]] | None = None,
    sampling_bias: float = 0.5,
) -> WalkResult:
    """Beam-search variant of `random_walk_extend`.

    Maintains `beam_width` partial codes at each depth. At every step, each
    beam slot is expanded by `n_expansions_per_slot` candidate next-stabilizers
    (each chosen via the max-coverage scorer over the slot's currently
    uncorrectable error pairs, with `n_solution_samples` candidate solutions
    considered per expansion). The combined `beam_width * n_expansions_per_slot`
    candidates are then pruned to the top `beam_width` by
    `(uncorrectables_remaining, stabilizers_added)` lex order.

    The key win vs. `random_walk_extend`: random walks make one stabilizer
    commitment per step and can't backtrack — once a walk picks a bad first
    stabilizer the whole walk is doomed. Beam search keeps multiple alternative
    paths alive and prunes by quality across all of them simultaneously, so a
    bad early commitment in one slot doesn't waste budget that another slot
    could spend more productively.

    Returns a WalkResult identical in shape to `random_walk_extend`'s. The
    `n_walks` / `successful_walks` fields report the final beam state's totals;
    `walks` is left empty (per-slot history could be added later)."""

    if extra_qubits > 0 and extra_support is not None:
        raise ValueError(
            "Specify either `extra_qubits` (shared extra-qubit register, set once at the "
            "start of each walk) or `extra_support` (one fresh extra qubit per step), not both."
        )

    if extra_qubits > 0:
        pad = stim.PauliString("_" * extra_qubits)
        stabilizers = [g + pad for g in stabilizers]
        errors = [e + pad for e in errors]

    seed(seed_val)
    initial_count = len(stabilizers)
    initial_uncorr = get_uncorrectable_errors(stabilizers, errors)

    # Each beam slot: (stabilizers_list, current_uncorrectables_list).
    beam: list[tuple[list[stim.PauliString], list[stim.PauliString]]] = [
        (deepcopy(stabilizers), initial_uncorr),
    ]
    best_code = deepcopy(stabilizers)
    best_key: tuple[int, int] = (len(initial_uncorr), 0)

    t_start = time.perf_counter()
    if verbose:
        print(
            f"[beam_search_extend depth=0/{max_stabilizers} elapsed=0.0s] "
            f"initial: remaining={best_key[0]} | beam: 1 slot",
            flush=True,
        )
    for _depth in range(max_stabilizers):
        next_beam: list[tuple[list[stim.PauliString], list[stim.PauliString]]] = []

        for slot_stabs, slot_uncorr in beam:
            if not slot_uncorr:
                # Already-resolved slots stay in the beam unchanged so they can
                # still win on the (remaining, added) tie-break.
                next_beam.append((slot_stabs, slot_uncorr))
                continue

            effective_samples = (
                max(n_solution_samples, 16) if connectivity is not None else n_solution_samples
            )
            for _ in range(n_expansions_per_slot):
                err = slot_uncorr[randrange(0, len(slot_uncorr))]
                if effective_samples > 1 or connectivity is not None or sampling_bias != 0.5:
                    selector = _build_max_coverage_selector(
                        slot_uncorr, extra_support, effective_samples,
                        connectivity=connectivity,
                        sampling_bias=sampling_bias,
                    )
                    try:
                        new_stabs = add_stabilizer(
                            slot_stabs, [err], extra_support=extra_support,
                            _solution_callback=selector,
                        )
                    except NoConnectivityValidSample:
                        # No connectivity-valid extension for this expansion; skip.
                        continue
                else:
                    new_stabs = add_stabilizer(
                        slot_stabs, [err], extra_support=extra_support,
                        choose_solution_randomly=True,
                    )
                new_uncorr = get_uncorrectable_errors(new_stabs, errors)
                next_beam.append((new_stabs, new_uncorr))

        # Prune: keep top beam_width by (remaining, added) lex order.
        next_beam.sort(key=lambda slot: (len(slot[1]), len(slot[0])))
        beam = next_beam[:beam_width]

        # Update global best across the new beam.
        for slot_stabs, slot_uncorr in beam:
            added = len(slot_stabs) - initial_count
            key = (len(slot_uncorr), added)
            if key < best_key:
                best_key = key
                best_code = deepcopy(slot_stabs)

        if verbose:
            n_succ_in_beam = sum(1 for _, u in beam if not u)
            print(
                f"[beam_search_extend depth={_depth + 1}/{max_stabilizers} "
                f"elapsed={time.perf_counter() - t_start:.1f}s] "
                f"best: remaining={best_key[0]} added={best_key[1]} | "
                f"beam: {len(beam)} slots, {n_succ_in_beam} succeeded",
                flush=True,
            )

        # Early termination: if every slot in the beam is fully resolved, we're done.
        if all(not u for _, u in beam):
            break

    successful_slots = sum(1 for _, u in beam if not u)
    succeeded = best_key[0] == 0
    distance, distance_exact = (None, True)
    if succeeded:
        distance, distance_exact = compute_distance(best_code, max_weight=distance_max_weight)
    return WalkResult(
        code=best_code,
        succeeded=succeeded,
        uncorrectables_remaining=best_key[0],
        n_stabilizers_added=best_key[1],
        n_walks=len(beam),
        successful_walks=successful_slots,
        walks=[],
        distance=distance,
        distance_is_exact=distance_exact,
    )


def dfs_extend(
    stabilizers: list[stim.PauliString],
    errors: list[stim.PauliString],
    extra_support: stim.PauliString | None = None,
    extra_qubits: int = 0,
    max_stabilizers: int = 6,
    n_candidates_per_node: int = 4,
    seed_val: int = 137,
    distance_max_weight: int = 3,
    connectivity: dict[int, set[int]] | None = None,
    max_seconds: float = 120.0,
    verbose: bool = False,
) -> WalkResult:
    """Depth-first search with branch-and-bound on the (which-error x which-solution)
    tree, with iterative-deepening-style pruning on the stabilizer-count budget.

    Differs from random_walk_extend / beam_search_extend in one fundamental way:
    DFS *backtracks*. Random walks and beam search explore forward only — once a
    walk picks a stabilizer, that commitment is permanent until the walk ends
    (random walk) or the slot is pruned (beam). DFS retreats from dead ends and
    tries alternatives at earlier depths.

    Combined with branch-and-bound (prune any subtree that can't possibly produce
    a shorter solution than the best already found) and a hard `max_stabilizers`
    cap, this gives a clean "exists a k>=initial_count + 1 - max_stabilizers
    extension?" answer rather than just "I tried hard and failed."

    At each node:
      1. Compute uncorrectables for the current partial code; update best-so-far.
      2. Prune if depth >= max_stabilizers, or if the best-known is already a
         valid solution with <= depth+1 stabilizers (branch-and-bound).
      3. Sample multiple candidate next-stabilizers via the existing
         max-coverage scorer; recurse on the top n_candidates_per_node in
         descending score order (greedy-first).
      4. Cut off via wall-clock budget `max_seconds`; return best-so-far on
         timeout.

    Returns a WalkResult; n_walks is repurposed to "tree nodes visited."""

    if extra_qubits > 0 and extra_support is not None:
        raise ValueError(
            "Specify either `extra_qubits` or `extra_support`, not both."
        )

    if extra_qubits > 0:
        pad = stim.PauliString("_" * extra_qubits)
        stabilizers = [g + pad for g in stabilizers]
        errors = [e + pad for e in errors]

    seed(seed_val)
    initial_count = len(stabilizers)
    initial_uncorr = get_uncorrectable_errors(stabilizers, errors)

    state = {
        "best_code": deepcopy(stabilizers),
        "best_remaining": len(initial_uncorr),
        "best_added": 0,
        "nodes_explored": 0,
        "timed_out": False,
    }
    t_start = time.perf_counter()

    def dfs(stabs: list[stim.PauliString], uncorr: list[stim.PauliString], depth: int) -> None:
        if time.perf_counter() - t_start > max_seconds:
            state["timed_out"] = True
            return
        state["nodes_explored"] += 1

        remaining = len(uncorr)
        key = (remaining, depth)
        if key < (state["best_remaining"], state["best_added"]):
            state["best_code"] = deepcopy(stabs)
            state["best_remaining"] = remaining
            state["best_added"] = depth
            if verbose:
                print(
                    f"[dfs_extend new best depth={depth} "
                    f"elapsed={time.perf_counter() - t_start:.1f}s nodes={state['nodes_explored']}] "
                    f"remaining={remaining} added={depth}",
                    flush=True,
                )

        if remaining == 0:
            return
        if depth >= max_stabilizers:
            return
        # Branch-and-bound: if best is already a valid solution no longer than
        # depth+1, this subtree can't help.
        if state["best_remaining"] == 0 and depth + 1 >= state["best_added"]:
            return

        # Pick a target error and sample candidate next-stabilizers.
        err = uncorr[randrange(0, len(uncorr))]
        A, b = _form_linear_system(stabs, [err])
        A_rref, b_rref = _boolean_rref(A, b)

        attempts = max(n_candidates_per_node * 4, 16)
        seen: set[tuple] = set()
        candidates: list[tuple[int, stim.PauliString, stim.PauliString]] = []
        for _ in range(attempts):
            x = sample_random_solution(A_rref, b_rref)
            xs = x[: x.size // 2]
            zs = x[x.size // 2 :]
            cache_key = (tuple(xs.tolist()), tuple(zs.tolist()))
            if cache_key in seen:
                continue
            seen.add(cache_key)
            ps = stim.PauliString.from_numpy(xs=xs, zs=zs)
            full = ps + extra_support if extra_support is not None else ps
            if connectivity is not None and not has_connected_support(full, connectivity):
                continue
            score = sum(1 for e in uncorr if not full.commutes(e))
            candidates.append((score, ps, full))

        # Highest score first — greedy ordering speeds finding any solution.
        candidates.sort(key=lambda t: -t[0])

        for _, ps, full in candidates[:n_candidates_per_node]:
            if extra_support is not None:
                id_extra = stim.PauliString("I" * len(extra_support))
                new_stabs = [g + id_extra for g in stabs] + [full]
            else:
                new_stabs = stabs + [full]
            new_uncorr = get_uncorrectable_errors(new_stabs, errors)
            dfs(new_stabs, new_uncorr, depth + 1)
            if state["timed_out"]:
                return

    dfs(deepcopy(stabilizers), initial_uncorr, 0)

    succeeded = state["best_remaining"] == 0
    distance, distance_exact = (None, True)
    if succeeded:
        distance, distance_exact = compute_distance(
            state["best_code"], max_weight=distance_max_weight,
        )
    return WalkResult(
        code=state["best_code"],
        succeeded=succeeded,
        uncorrectables_remaining=state["best_remaining"],
        n_stabilizers_added=state["best_added"],
        n_walks=state["nodes_explored"],
        successful_walks=1 if succeeded else 0,
        walks=[],
        distance=distance,
        distance_is_exact=distance_exact,
    )


def simulated_annealing_extend(
    stabilizers: list[stim.PauliString],
    errors: list[stim.PauliString],
    extra_support: stim.PauliString | None = None,
    extra_qubits: int = 0,
    target_stabilizers: int = 6,
    n_iterations: int = 200,
    initial_temperature: float = 5.0,
    cooling: float = 0.99,
    seed_val: int = 137,
    distance_max_weight: int = 3,
    connectivity: dict[int, set[int]] | None = None,
    verbose: bool = False,
) -> WalkResult:
    """Simulated annealing over fixed-size extensions.

    Maintains a single "candidate" extension of exactly `target_stabilizers`
    added stabilizers, scored by uncorrectable_count. At each iteration, picks a
    random one of the added stabilizers and swaps it for a freshly-sampled
    replacement (drawn from the constraint subspace for some uncorrectable
    error). Accepts unconditionally if the new uncorrectable count is lower;
    otherwise accepts with probability exp(-delta/T). Cools T geometrically.

    Differs from random walk + beam + DFS: those build the extension incrementally,
    each commitment shrinks the search space. SA works in a *fixed-size* space and
    explores neighborhoods via swaps -- naturally escapes the local minima beam
    search gets stuck in, at the cost of needing the target size up front.

    `target_stabilizers` = number of stabilizers to add (so the resulting code
    has `initial_count + target_stabilizers` stabilizers and k = n -
    (initial_count + target_stabilizers) logical qubits). Set explicitly to
    target a specific [[n,k,d]].
    """

    if extra_qubits > 0 and extra_support is not None:
        raise ValueError(
            "Specify either `extra_qubits` or `extra_support`, not both."
        )
    if extra_qubits > 0:
        pad = stim.PauliString("_" * extra_qubits)
        stabilizers = [g + pad for g in stabilizers]
        errors = [e + pad for e in errors]

    seed(seed_val)
    initial_count = len(stabilizers)

    def sample_one_stabilizer(stabs, target_err):
        """Sample a single candidate stabilizer that anticommutes with target_err
        and commutes with all current stabilizers, respecting connectivity if set."""
        A, b = _form_linear_system(stabs, [target_err])
        A_rref, b_rref = _boolean_rref(A, b)
        for _ in range(64):
            x = sample_random_solution(A_rref, b_rref)
            ps = stim.PauliString.from_numpy(xs=x[: x.size // 2], zs=x[x.size // 2 :])
            full = ps + extra_support if extra_support is not None else ps
            if connectivity is None or has_connected_support(full, connectivity):
                return full
        return None

    def make_extended(stabs, full_new):
        if extra_support is not None:
            id_extra = stim.PauliString("I" * len(extra_support))
            return [g + id_extra for g in stabs] + [full_new]
        return stabs + [full_new]

    # Initial state: target_stabilizers random additions on top of `stabilizers`.
    current = deepcopy(stabilizers)
    failed_init = False
    for _ in range(target_stabilizers):
        uncorr = get_uncorrectable_errors(current, errors)
        if not uncorr:
            # Already correctable before we hit the target size; pad with identity?
            # Just stop early; SA on a smaller code is fine.
            break
        err = uncorr[randrange(0, len(uncorr))]
        full_new = sample_one_stabilizer(current, err)
        if full_new is None:
            failed_init = True
            break
        current = make_extended(current, full_new)

    current_score = len(get_uncorrectable_errors(current, errors)) if not failed_init else 10 ** 9
    best_code = deepcopy(current)
    best_score = current_score
    T = initial_temperature
    t_start = time.perf_counter()

    for it in range(n_iterations):
        # Pick a random added stabilizer to swap out
        n_added = len(current) - initial_count
        if n_added == 0:
            break
        swap_idx = initial_count + randrange(0, n_added)

        # Drop it from the partial code
        partial = current[:swap_idx] + current[swap_idx + 1 :]
        uncorr = get_uncorrectable_errors(partial, errors)
        if not uncorr:
            # Partial is already correctable -- great, accept it
            new_state = partial
            new_score = 0
        else:
            err = uncorr[randrange(0, len(uncorr))]
            full_new = sample_one_stabilizer(partial, err)
            if full_new is None:
                continue
            new_state = make_extended(partial, full_new)
            new_score = len(get_uncorrectable_errors(new_state, errors))

        delta = new_score - current_score
        accept = delta <= 0 or (np.random.random() < np.exp(-delta / max(T, 1e-9)))
        if accept:
            current = new_state
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_code = deepcopy(current)
                if verbose:
                    print(
                        f"[sa_extend iter={it} T={T:.3f} "
                        f"elapsed={time.perf_counter() - t_start:.1f}s] "
                        f"new best: remaining={best_score}",
                        flush=True,
                    )
                if best_score == 0:
                    break  # solution found
        T *= cooling

    succeeded = best_score == 0
    distance, distance_exact = (None, True)
    if succeeded:
        distance, distance_exact = compute_distance(
            best_code, max_weight=distance_max_weight,
        )
    return WalkResult(
        code=best_code,
        succeeded=succeeded,
        uncorrectables_remaining=best_score,
        n_stabilizers_added=len(best_code) - initial_count,
        n_walks=n_iterations,
        successful_walks=1 if succeeded else 0,
        walks=[],
        distance=distance,
        distance_is_exact=distance_exact,
    )


if __name__ == "__main__":
    stabilizers = [stim.PauliString("ZZ")]
    errors = [stim.PauliString("X_"), stim.PauliString("_X")]
    new_stabilizers = random_depth_first_search(
        stabilizers, errors, extra_support=stim.PauliString("Z"),
        steps=1
    )
    for stab in new_stabilizers:
        print(stab)