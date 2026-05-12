from typing import List, Tuple, Optional, Set, Collection
from warnings import warn
from dataclasses import dataclass, field
import itertools
import functools
from random import randrange, seed, sample
from copy import deepcopy
import numpy as np
import stim
from encoded.decompose_operators import generators_to_matrix
from encoded.binary_linalg import solve_boolean_system, _boolean_rref, enumerate_all_solutions, system_has_solutions, sample_random_solution

def metric_tensor(nq: int) -> np.ndarray:
    id_nq = np.eye(nq).astype(bool)
    zeros_nq = np.zeros((nq, nq)).astype(bool)
    return np.vstack((
        np.hstack((zeros_nq, id_nq)),
        np.hstack((id_nq, zeros_nq))
    ))


def _form_linear_system(generators: List[stim.PauliString], errors: List[stim.PauliString]) -> Tuple[np.ndarray, np.ndarray]:
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
    generators: List[stim.PauliString], errors: List[stim.PauliString], extra_support: Optional[stim.PauliString]=None,
    verbose: bool=False, choose_solution_randomly: bool=False,
    _solution_callback=None,
) -> List[stim.PauliString]:
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


def generate_stabilizer_elements(generators: List[stim.PauliString]) -> List[stim.PauliString]:
    nq = max([len(ps) for ps in generators])
    elements = []
    for string in itertools.chain.from_iterable(itertools.combinations(generators, r) for r in range(len(generators) + 1)):
        elements.append(
            functools.reduce(lambda a, b: a * b, string, stim.PauliString('_' * nq))
        )
    return elements


def stim_strings_equal_up_to_phase(ps_a: stim.PauliString, ps_b: stim.PauliString) -> bool:
    return list(ps_a) == list(ps_b)


def is_in_stabilizer_group(operator: stim.PauliString, generators: List[stim.PauliString]) -> bool:
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


def _legacy_group_membership_check(operator: stim.PauliString, generators: List[stim.PauliString]) -> bool:
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


def knill_laflamme_cost_function(generators: List[stim.PauliString], errors: List[float], weights: List[float]) -> float:
    """Cost function from the RL paper."""

    total_loss = 0.
    for weight, err in zip(weights, errors):
        anticommutation_tests = [not gen.commutes(err) for gen in generators]
        in_stabilizer_group = is_in_stabilizer_group(err, generators)
        if any(anticommutation_tests) or in_stabilizer_group:
            # The error is correctable, so K_mu = 1.
            total_loss -= weight
    return total_loss


def knill_laflamme_correctable_cost_function(generators: List[stim.PauliString], errors: List[float], weights: List[float]) -> float:
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


def get_uncorrectable_errors(generators: List[stim.PauliString], errors: List[stim.PauliString]) -> List[stim.PauliString]:
    """Get the products of errors that the code cannot correct."""

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
    generators: List[stim.PauliString], errors: List[stim.PauliString], extra_support: Optional[stim.PauliString]=None,
    max_iter: int = 1_000, seed_val: int=137, errors_per_round=1
) -> List[stim.PauliString]:
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


def prune_duplicate_pauli_strings(strings: List[stim.PauliString]) -> List[stim.PauliString]:
    """stim.PauliString objects are not hashable, so you can't make a set of them. This function
    removed duplicated from the list."""

    new_strings = []
    for pstring in strings:
        if not any([pstring == ps for ps in new_strings]):
            new_strings.append(pstring)
    return new_strings


def all_new_codes_for_errors(
    generators: Collection[stim.PauliString], errors: List[stim.PauliString],
    extra_support: Optional[stim.PauliString] = None
) -> List[stim.PauliString]:
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
    stabilizers: List[stim.PauliString], errors: List[stim.PauliString],
    extra_support: Optional[stim.PauliString]=None,
    steps: int = 1, max_tries: int = 10, seed_val: int=137
) -> List[stim.PauliString]:
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
    """Aggregate result of `random_walk_extend`.

    `code` is the best stabilizer set found, picked by the lexicographic min of
    (uncorrectables_remaining, stabilizers_added) so that successful walks beat
    failed ones, and among successful walks, shorter extensions beat longer ones
    (preserving more logical qubits)."""
    code: List[stim.PauliString]
    succeeded: bool
    uncorrectables_remaining: int
    n_stabilizers_added: int
    n_walks: int
    successful_walks: int
    walks: List[PerWalkResult] = field(default_factory=list)


def _build_max_coverage_selector(
    uncorrectables: List[stim.PauliString],
    extra_support: Optional[stim.PauliString],
    n_samples: int,
):
    """Build a solution-selection callback for add_stabilizer that, given the
    affine solution space (A_rref, b_rref), samples `n_samples` candidate
    solutions and returns the one that anticommutes with the most
    `uncorrectables`. Score includes any `extra_support` qubits appended after
    the linear system is solved."""

    def selector(A_rref: np.ndarray, b_rref: np.ndarray) -> np.ndarray:
        best_x = None
        best_score = -1
        for _ in range(n_samples):
            x = sample_random_solution(A_rref, b_rref)
            candidate = stim.PauliString.from_numpy(
                xs=x[:x.size // 2], zs=x[x.size // 2:],
            )
            full = candidate + extra_support if extra_support is not None else candidate
            score = sum(1 for e in uncorrectables if not full.commutes(e))
            if score > best_score:
                best_score = score
                best_x = x
        return best_x

    return selector


def random_walk_extend(
    stabilizers: List[stim.PauliString],
    errors: List[stim.PauliString],
    extra_support: Optional[stim.PauliString] = None,
    ancilla_budget: int = 0,
    max_stabilizers_per_walk: int = 1,
    max_walks: int = 10,
    n_solution_samples: int = 1,
    seed_val: int = 137,
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
        ancilla qubit; all prior generators are padded with identity. So a walk of
        depth `d` with single-qubit `extra_support` adds `d` ancilla qubits. Mutually
        exclusive with `ancilla_budget`.
    ancilla_budget - Number of shared ancilla qubits to pre-allocate up front. With
        `ancilla_budget=m`, every initial generator is padded with `m` identities and
        every error gets `m` identities appended (errors act only on the original data
        qubits). Each added stabilizer can then place arbitrary support on any of the
        n+m qubits, so multiple stabilizers can SHARE the same ancilla — matching the
        slide 14 hand-design which uses 2 ancillae for 2 added stabilizers on FH [[8,6]]
        rather than 2 ancillae per stabilizer.
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
    seed_val - RNG seed."""

    if ancilla_budget > 0 and extra_support is not None:
        raise ValueError(
            "Specify either `ancilla_budget` (shared ancilla register, set once at the "
            "start of each walk) or `extra_support` (one fresh ancilla per step), not both."
        )

    if ancilla_budget > 0:
        pad = stim.PauliString("_" * ancilla_budget)
        stabilizers = [g + pad for g in stabilizers]
        errors = [e + pad for e in errors]

    seed(seed_val)
    initial_count = len(stabilizers)
    initial_uncorrectables = len(get_uncorrectable_errors(stabilizers, errors))

    best_code = deepcopy(stabilizers)
    best_key: Tuple[int, int] = (initial_uncorrectables, 0)  # (remaining, added)
    walks: List[PerWalkResult] = []

    for _ in range(max_walks):
        temp = deepcopy(stabilizers)
        for _ in range(max_stabilizers_per_walk):
            uncorrectables = get_uncorrectable_errors(temp, errors)
            if not uncorrectables:
                break
            new_err = uncorrectables[randrange(0, len(uncorrectables))]
            if n_solution_samples > 1:
                selector = _build_max_coverage_selector(
                    uncorrectables, extra_support, n_solution_samples,
                )
                temp = add_stabilizer(
                    temp, [new_err], extra_support=extra_support, _solution_callback=selector,
                )
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

    return WalkResult(
        code=best_code,
        succeeded=(best_key[0] == 0),
        uncorrectables_remaining=best_key[0],
        n_stabilizers_added=best_key[1],
        n_walks=max_walks,
        successful_walks=sum(1 for w in walks if w.succeeded),
        walks=walks,
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