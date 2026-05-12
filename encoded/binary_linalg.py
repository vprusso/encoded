from typing import List, Dict, Tuple
import itertools as it
from copy import deepcopy
from random import randrange
import numpy as np

def _swap_row(arr: np.ndarray, i: int, j: int):
    temp = arr[i, :].copy()
    arr_copy = arr.copy()
    arr_copy[i, :] = arr_copy[j, :]
    arr_copy[j, :] = temp
    return arr_copy


def _swap_elems(b: np.ndarray, i, j):
    b_copy = b.copy()
    temp = b[i]
    b_copy[i] = b[j]
    b_copy[j] = temp
    return b_copy


# def _boolean_rref(A: np.ndarray, b: np.ndarray) -> np.ndarray:
#     # assert A.shape[1] <= A.shape[0]

#     A_copy = A.copy()
#     b_copy = b.copy()

#     max_j = min(A_copy.shape[0], A_copy.shape[1])
#     for j in range(max_j):
#         # Find the first index i s.t. A[i, j] = 1.
#         found = False
#         for idx_first_one in range(j, A_copy.shape[0]):
#             if A_copy[idx_first_one, j]:
#                 found = True
#                 break
#         # Swap that row with the j^th row.
#         A_copy = _swap_row(A_copy, idx_first_one, j)
#         b_copy = _swap_elems(b_copy, idx_first_one, j)
#         # Eliminate all other rows i where A[i, j] = 1.
#         for i in range(j+1, A.shape[0]):
#             if A_copy[i, j]:
#                 A_copy[i, :] = A_copy[i, :] ^ A_copy[j, :]
#                 b_copy[i] = b_copy[i] ^ b_copy[j]
#     return A_copy, b_copy


def _boolean_rref(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    # assert A.shape[1] <= A.shape[0]

    A_copy = A.copy()
    b_copy = b.copy()

    i = 0 # Row at which to make a pivot.
    for j in range(A.shape[1]):
        # print(f"i={i} j={j}")
        # print("A=")
        # print(A_copy)
        # print("b=")
        # print(b_copy)
        if np.all(np.invert(A_copy[i:, j])):
            # print("Premature continue")
            continue
        # Find the first index i s.t. A[i, j] = 1.
        found = False
        for idx_first_one in range(i, A_copy.shape[0]):
            # print(f"idx_first_one={idx_first_one}")
            if A_copy[idx_first_one, j]:
                found = True
                # print("Found!")
                break
        if not found:
            continue
        # Swap that row with the j^th row.
        # print(f"Swapping {idx_first_one} <-> {i}")
        A_copy = _swap_row(A_copy, idx_first_one, i)
        b_copy = _swap_elems(b_copy, idx_first_one, i)
        # Eliminate all other rows i where A[i, j] = 1.
        for k in range(i+1, A.shape[0]):
            # print(f"k={k}")
            # if A_copy[k, j] and k != j:
            #     A_copy[k, :] = A_copy[k, :] ^ A_copy[j, :]
            #     b_copy[k] = b_copy[k] ^ b_copy[j]
            if A_copy[k, j]:
                # print(f"XOR {k} {j}")
                A_copy[k, :] = A_copy[k, :] ^ A_copy[i, :]
                b_copy[k] = b_copy[k] ^ b_copy[i]
        i += 1
        if i >= A.shape[0]:
            break
    # print("Final")
    # print("A=")
    # print(A_copy)
    # print("b=")
    # print(b_copy)
    return A_copy, b_copy

def _boolean_backsub_solve(A_rref: np.ndarray, b_rref: np.ndarray) -> np.ndarray:
    x = np.zeros(A_rref.shape[1]).astype(bool)
    # Find the first value of i s.t. A[i, i] != 1.
    found = False
    for first_i in range(A_rref.shape[1]):
        if first_i >= min(A_rref.shape):
            first_i -= 1
            break
        if not A_rref[first_i, first_i]:
            found = True
            break
    if not found:
        first_i += 1
    for i in range(first_i - 1, -1, -1):
        x_i = False
        for j in range(i+1, first_i):
            x_i ^= A_rref[i, j] and x[j]
        x_i ^= b_rref[i]
        x[i] = x_i
    return x


def _pivot_columns(A: np.ndarray) -> List[int]:
    """Find the pivot columns of the binary matrix A in RREF."""

    i = 0 # Index of row where the pivot is.
    j = 0 # Index of column we are currently searching.
    pivot_columns = []
    while i < A.shape[0]:
        if A[i, j]:
            i += 1
            pivot_columns.append(j)
        j += 1
        if j >= A.shape[1]:
            break
    return pivot_columns


def _pivot_locations(A: np.ndarray) -> List[Tuple[int, int]]:
    """Find the pivot columns of the binary matrix A in RREF."""

    i = 0 # Index of row where the pivot is.
    j = 0 # Index of column we are currently searching.
    pivots = []
    while i < A.shape[0]:
        if A[i, j]:
            pivots.append((i, j))
            i += 1
        j += 1
        if j >= A.shape[1]:
            break
    return pivots


def solve_boolean_system(A, b, verbose: bool=False):
    A_rref, b_rref = _boolean_rref(A, b)
    if verbose:
        print("A_rref=\n", A_rref)
        print("b_rref=\n", b_rref)
    x = _boolean_backsub_solve(A_rref, b_rref)
    return x


# TODO This should be a generator.
def _enumerate_bitstrings(n: int) -> List[np.ndarray]:
    """Enumerate all bitstrings with n bits in the form of numpy arrays"""

    binary_lists = it.product([False, True], repeat=n)
    bstrings = []
    for bs in binary_lists:
        bstrings.append(np.array(list(bs)))
    return bstrings


def _single_row_backsub(row: np.ndarray, i: int, known_values: Dict[int, bool], rhs: bool) -> bool:
    """Solve for the value in column i for this row during backsubstitution.
    The values we have already solved for are encoded in known_values.
    
    Arguments:
    row - The row of the matrix we are currenty solving.
    i - The index of the pivot column. This should be True in the row that is passed.
    known_values - Values we have previously solved for, or free variables.
    rhs - The value of the right hand side for the current row."""

    assert row[i], f"Column i has value: {row[i]}, should be True."
    
    known_true_sum = False # sum of values that are known and have True for their column in the row.
    for j in range(i+1, row.size):
        if row[j] and j not in known_values.keys():
            raise ValueError(f"Column {j} > pivot column {i} is True, but a known value is not given.")
        if row[j]:
            known_true_sum ^= known_values[j]
    return rhs ^ known_true_sum


def solve_with_known_values(A: np.ndarray, b: np.ndarray, known_values: Dict[int, bool]) -> np.ndarray:
    """Given A and b in RREF and values for the free variables, solve the solution vector x."""

    # Check that all of the free variables are known.
    pivots = _pivot_locations(A)
    pivot_columns = [t[1] for t in pivots]
    free_columns = set(range(A.shape[1])) - set(pivot_columns)
    assert free_columns.issubset(set(known_values.keys()))

    # Sort the pivots from greatest to least by their column index.
    sorted_pivots = reversed(sorted(pivots, key=lambda t: t[1]))
    # Do the backsubstitution starting from the pivot that is furthest down.
    known_copy = deepcopy(known_values)
    for i, j in sorted_pivots:
        x_j = _single_row_backsub(A[i, :], j, known_copy, b[i])
        known_copy[j] = x_j
    return np.array([known_copy[i] for i in range(A.shape[1])])


def enumerate_all_solutions(A: np.ndarray, b: np.ndarray) -> List[np.ndarray]:
    """Enumerate all solutions to a system of binary equations. A must be in
    reduced row echelon form."""

    # Check that all of the free variables are known.
    pivots = _pivot_locations(A)
    pivot_columns = [t[1] for t in pivots]
    free_columns = sorted(list(set(range(A.shape[1])) - set(pivot_columns)))

    solutions = []
    if len(free_columns) == 0:
        solutions.append(solve_with_known_values(A, b, {}))
    else:
        for bitstr in _enumerate_bitstrings(len(free_columns)):
            known_values = dict(zip(free_columns, bitstr))
            x = solve_with_known_values(A, b, known_values)
            solutions.append(x)
    return solutions


def sample_random_solution(A_rref: np.ndarray, b_rref: np.ndarray) -> np.ndarray:
    """Sample one solution uniformly from the affine subspace of solutions to
    A_rref @ x = b_rref over GF(2). A_rref must be in RREF; behavior is undefined
    if the system is inconsistent (caller should check `system_has_solutions`).

    O(n^2) — never materializes 2^(free vars) candidates the way
    `enumerate_all_solutions` does, so it scales to ~30+ free variables.

    Consumes exactly one `random.randrange(0, 2^n_free)` draw, matching the
    `randrange(0, len(enumerate_all_solutions(...)))` pattern callers used
    previously — so existing seeded tests get bit-identical results."""

    pivots = _pivot_locations(A_rref)
    pivot_columns = {t[1] for t in pivots}
    free_columns = sorted(j for j in range(A_rref.shape[1]) if j not in pivot_columns)
    n_free = len(free_columns)

    # Match enumerate_all_solutions's behavior: when there are no free variables
    # it still appends one solution, so callers' randrange(0, 1) consumed one draw.
    n_solutions = 1 << n_free if n_free > 0 else 1
    k = randrange(0, n_solutions)

    if n_free == 0:
        return solve_with_known_values(A_rref, b_rref, {})

    # itertools.product([False, True], repeat=n_free) walks tuples in lex order,
    # so the i-th element of the k-th tuple is bit (n_free - 1 - i) of k.
    known = {
        free_columns[i]: bool((k >> (n_free - 1 - i)) & 1)
        for i in range(n_free)
    }
    return solve_with_known_values(A_rref, b_rref, known)


def system_has_solutions(A_rref: np.ndarray, b_rref: np.ndarray) -> bool:
    """Test is the system in RREF has a solution. The criterion is that
    there should be no row in the reduced augmented matrix of the form
    [0 0 ... 0 | 1] (i.e. the row of A is all zero and the element of b is 1)."""

    assert b_rref.size == A_rref.shape[0]

    for i in range(A_rref.shape[0]):
        if np.all(np.invert(A_rref[i, :])) and b_rref[i]:
            return False
    return True