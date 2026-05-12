import unittest
import numpy as np
import stim
import random
from encoded.add_stabilizers import (
    _form_linear_system, add_stabilizer, is_in_stabilizer_group, knill_laflamme_cost_function,
    build_code_randomly, prune_duplicate_pauli_strings, random_depth_first_search,
    _legacy_group_membership_check, random_walk_extend, get_uncorrectable_errors,
    beam_search_extend,
)

class TestLinearSystem(unittest.TestCase):

    def test_zzi_izz_e_z(self):
        """Test the case where the existing generators are ZZI and IZZ,
        and we want the new stabilizer to anticommte with ZII."""

        stabilizers = [
            stim.PauliString("ZZI"),
            stim.PauliString("IZZ")
        ]
        errs = [
            stim.PauliString("ZII")
        ]
        A, b = _form_linear_system(stabilizers, errs)
        A_target = np.array([
            [True, True, False, False, False, False],
            [False, True, True, False, False, False],
            [True, False, False, False, False, False]
        ])
        b_target = np.array([False, False, True])
        self.assertTrue(np.allclose(A, A_target) and np.allclose(b, b_target))
    
    def test_errors_too_short(self):
        """If we pass errors with support on less qubits, they should be lengthened."""

        stabilizers = [
            stim.PauliString("ZZI"),
            stim.PauliString("IZZ")
        ]
        errs = [
            stim.PauliString("ZI") # N.b. length 3 vs. length 3 for generators.
        ]
        A, b = _form_linear_system(stabilizers, errs)
        A_target = np.array([
            [True, True, False, False, False, False],
            [False, True, True, False, False, False],
            [True, False, False, False, False, False]
        ])
        b_target = np.array([False, False, True])
        self.assertTrue(np.allclose(A, A_target) and np.allclose(b, b_target))


class TestAddStabilizer(unittest.TestCase):

    def test_zzi_izz_e_z(self):
        """Test the case where the existing generators are ZZI and IZZ,
        and we want the new stabilizer to anticommte with ZII."""

        stabilizers = [
            stim.PauliString("ZZI"),
            stim.PauliString("IZZ")
        ]
        errs = [
            stim.PauliString("ZII")
        ]
        new_stabilizers = add_stabilizer(stabilizers, errs, extra_support=stim.PauliString("X"))
        target_stabilizers = [
            stim.PauliString("ZZII"),
            stim.PauliString("IZZI"),
            stim.PauliString("XXXX")
        ]
        self.assertEqual(new_stabilizers, target_stabilizers)

    def test_zzi_e_iix(self):
        stabilizers = [
            stim.PauliString("ZZ"),
        ]
        errs = [
            stim.PauliString("XX")
        ]
        new_stabilizers = add_stabilizer(stabilizers, errs, verbose=False, extra_support=stim.PauliString("Z"))
        target_stabilizers = [
            stim.PauliString("ZZI"),
            stim.PauliString("ZIZ"),
        ]
        self.assertEqual(new_stabilizers, target_stabilizers)


class TestGroupMembership(unittest.TestCase):

    def test_id(self):
        operator = stim.PauliString("__")
        generators = [
            stim.PauliString("X_"),
            stim.PauliString("_X")
        ]
        self.assertTrue(is_in_stabilizer_group(operator, generators))

    def test_repetition(self):
        operator = stim.PauliString("Z_Z")
        generators = [
            stim.PauliString("ZZ_"),
            stim.PauliString("_ZZ")
        ]
        self.assertTrue(is_in_stabilizer_group(operator, generators))
    
    def test_id_in_repetition(self):
        operator = stim.PauliString("__")
        generators = [
            stim.PauliString("ZZ_"),
            stim.PauliString("_ZZ")
        ]


class TestKLCost(unittest.TestCase):

    def test_repetition(self):
        generators = [
            stim.PauliString("ZZ_"),
            stim.PauliString("_ZZ")
        ]
        errors = [
            stim.PauliString("X__"),
            stim.PauliString("_X_"),
            stim.PauliString("__X")
        ]
        weights = [1.] * len(errors)
        loss = knill_laflamme_cost_function(generators, errors, weights)
        self.assertTrue(abs(loss + 3.) <= 1e-12)

    def test_phase_flip(self):
        generators = [
            stim.PauliString("XX_"),
            stim.PauliString("_XX")
        ]
        errors = [
            stim.PauliString("Z__"),
            stim.PauliString("_Z_"),
            stim.PauliString("__Z")
        ]
        weights = [1.] * len(errors)
        loss = knill_laflamme_cost_function(generators, errors, weights)
        self.assertTrue(abs(loss + 3.) <= 1e-12)

    def test_five_qubit(self):
        generators = [
            stim.PauliString("XZZXI"),
            stim.PauliString("IXZZX"),
            stim.PauliString("XIXZZ"),
            stim.PauliString("ZXIXZ"),
        ]
        errors = [
            stim.PauliString("Z____"),
            stim.PauliString("_Z___"),
            stim.PauliString("__Z__"),
            stim.PauliString("___Z_"),
            stim.PauliString("____Z"),
            stim.PauliString("X____"),
            stim.PauliString("_X___"),
            stim.PauliString("__X__"),
            stim.PauliString("___X_"),
            stim.PauliString("____X"),
            stim.PauliString("Y____"),
            stim.PauliString("_Y___"),
            stim.PauliString("__Y__"),
            stim.PauliString("___Y_"),
            stim.PauliString("____Y")
        ]
        weights = [1.] * len(errors)
        loss = knill_laflamme_cost_function(generators, errors, weights)
        self.assertTrue(abs(loss + len(errors)) <= 1e-12)


class TestBuildCodes(unittest.TestCase):

    def test_full_repetition(self):
        stabilizers = [stim.PauliString("ZZ_"), stim.PauliString("_ZZ")]
        errors = [stim.PauliString("X__"), stim.PauliString("_X_"), stim.PauliString("__X")]
        new_stabilizers = build_code_randomly(stabilizers, errors, extra_support=None)
        self.assertTrue(new_stabilizers == stabilizers)

    def test_partial_repetition(self):
        stabilizers = [stim.PauliString("ZZ")]
        errors = [stim.PauliString("X_"), stim.PauliString("_X")]
        new_stabilizers = build_code_randomly(stabilizers, errors, extra_support=stim.PauliString("Z"))
        target_stabilizers = [stim.PauliString("ZZ_"), stim.PauliString("Z_Z")]
        self.assertTrue(new_stabilizers == target_stabilizers)

    def test_expanded_repetition(self):
        stabilizers = [stim.PauliString("ZZ_"), stim.PauliString("Z_Z")]
        errors = [
            stim.PauliString("___"),
            stim.PauliString("X__"), stim.PauliString("_X_"), stim.PauliString("__X"),
            stim.PauliString("Z__"), stim.PauliString("_Z_"), stim.PauliString("__Z")
        ]
        new_stabilizers = build_code_randomly(stabilizers, errors, extra_support=stim.PauliString("X"))
        target_stabilizers = [stim.PauliString("ZZ__"), stim.PauliString("Z_Z_"), stim.PauliString("XXXX")]
        # print(new_stabilizers)
        self.assertTrue(new_stabilizers == target_stabilizers)
    
    def test_z1z2_arbitrary(self):
        """Second example from the overleaf."""

        stabilizers = [stim.PauliString("ZZ")]
        errors = [stim.PauliString("__"), stim.PauliString("X_"), stim.PauliString("_X"), stim.PauliString("Z_"), stim.PauliString("_Z")]
        new_stabilizers = build_code_randomly(stabilizers, errors, extra_support=stim.PauliString("Z"))
        target_stabilizers = [stim.PauliString("ZZ__"), stim.PauliString("XXZ_"), stim.PauliString("Z_XZ")]
        self.assertTrue(new_stabilizers == target_stabilizers)


class TestPrune(unittest.TestCase):

    def test_all_unequal(self):
        pstrings = [
            stim.PauliString("___"),
            stim.PauliString("_XZ")
        ]
        pruned = prune_duplicate_pauli_strings(pstrings)
        self.assertTrue(pruned == pstrings)

    def test_one_duplicated(self):
        pstrings = [
            stim.PauliString("___"),
            stim.PauliString("_XZ"),
            stim.PauliString("Y"),
            stim.PauliString("_XZ"),
            stim.PauliString("Z_Z__")
        ]
        target = [
            stim.PauliString("___"),
            stim.PauliString("_XZ"),
            stim.PauliString("Y"),
            stim.PauliString("Z_Z__")
        ]
        pruned = prune_duplicate_pauli_strings(pstrings)
        self.assertTrue(pruned == target)


class TestRandomDescent(unittest.TestCase):

    def test_reptition(self):
        stabilizers = [stim.PauliString("ZZ")]
        errors = [stim.PauliString("X_"), stim.PauliString("_X")]
        new_stabilizers = random_depth_first_search(
            stabilizers, errors, extra_support=stim.PauliString("Z"),
            steps=1
        )
        target_stabilizers = [stim.PauliString("ZZ_"), stim.PauliString("_ZZ")]
        self.assertEqual(new_stabilizers, target_stabilizers)

    def test_expanded_repetition(self):
        stabilizers = [stim.PauliString("ZZ_"), stim.PauliString("_ZZ")]
        errors = [stim.PauliString("___"), stim.PauliString("Z__"), stim.PauliString("_Z_"), stim.PauliString("__Z")]
        new_stabilizers = random_depth_first_search(
            stabilizers, errors, extra_support=stim.PauliString("X"),
            steps=1, seed_val=12
        )
        for stab in new_stabilizers:
            print(stab)
        target_stabilizers = [stim.PauliString("ZZ__"), stim.PauliString("_ZZ_"), stim.PauliString("YXXX")]
        self.assertEqual(new_stabilizers, target_stabilizers)

class TestGroupMembershipAgreesWithLegacy(unittest.TestCase):
    """Cross-validate is_in_stabilizer_group against the O(2^k) legacy implementation
    over a stress sample of (generator set, query operator) pairs. The two must agree
    on every input — if they ever disagree, the fast path has a bug."""

    @staticmethod
    def _random_pauli(rng: random.Random, nq: int) -> stim.PauliString:
        return stim.PauliString("".join(rng.choice("_XYZ") for _ in range(nq)))

    def _assert_agreement_on_random_inputs(self, nq: int, n_gens: int, n_queries: int, seed_val: int):
        rng = random.Random(seed_val)
        generators = [self._random_pauli(rng, nq) for _ in range(n_gens)]
        for _ in range(n_queries):
            op = self._random_pauli(rng, nq)
            legacy = _legacy_group_membership_check(op, generators)
            fast = is_in_stabilizer_group(op, generators)
            self.assertEqual(
                legacy, fast,
                msg=f"Disagreement on operator {op} with generators {generators}",
            )

    def test_random_3qubit(self):
        self._assert_agreement_on_random_inputs(nq=3, n_gens=2, n_queries=80, seed_val=1)

    def test_random_4qubit(self):
        self._assert_agreement_on_random_inputs(nq=4, n_gens=3, n_queries=80, seed_val=2)

    def test_random_5qubit(self):
        self._assert_agreement_on_random_inputs(nq=5, n_gens=4, n_queries=80, seed_val=3)

    def test_products_of_generators_are_members(self):
        """Any product of generators is in the group by definition — both impls must agree on True."""
        generators = [
            stim.PauliString("XZZX_"),
            stim.PauliString("_XZZX"),
            stim.PauliString("ZX___"),
            stim.PauliString("__XZZ"),
        ]
        nq = 5
        for mask in range(1, 1 << len(generators)):
            prod = stim.PauliString("_" * nq)
            for i in range(len(generators)):
                if mask & (1 << i):
                    prod = prod * generators[i]
            self.assertTrue(is_in_stabilizer_group(prod, generators))
            self.assertTrue(_legacy_group_membership_check(prod, generators))


class TestSampleRandomSolution(unittest.TestCase):
    """sample_random_solution should always return a valid solution to the system,
    and over many draws should cover the affine subspace."""

    def test_returns_valid_solution(self):
        from encoded.binary_linalg import _boolean_rref, sample_random_solution
        # A non-trivial 3x4 system that has multiple solutions.
        A = np.array([
            [True,  True,  False, False],
            [False, True,  True,  False],
            [True,  False, False, True],
        ])
        b = np.array([True, False, True])
        A_rref, b_rref = _boolean_rref(A, b)

        random.seed(0)
        for _ in range(20):
            x = sample_random_solution(A_rref, b_rref)
            # Check x is a solution: A @ x = b (mod 2)
            product = np.zeros(A.shape[0], dtype=bool)
            for j in range(A.shape[1]):
                if x[j]:
                    product ^= A[:, j]
            self.assertTrue(np.array_equal(product, b))

    def test_covers_solution_space(self):
        """Across many samples, sample_random_solution should hit every solution
        in the affine subspace at least once."""
        from encoded.binary_linalg import _boolean_rref, enumerate_all_solutions, sample_random_solution
        # Pick a system with a small known solution space so we can enumerate.
        A = np.array([
            [True,  False, True, False],
            [False, True,  True, True],
        ])
        b = np.array([True, False])
        A_rref, b_rref = _boolean_rref(A, b)
        all_sols = enumerate_all_solutions(A_rref, b_rref)
        all_sols_as_tuples = {tuple(s.tolist()) for s in all_sols}

        random.seed(42)
        seen = set()
        for _ in range(200):
            x = sample_random_solution(A_rref, b_rref)
            seen.add(tuple(x.tolist()))

        # Every drawn solution should be in the enumerated set.
        self.assertTrue(seen.issubset(all_sols_as_tuples))
        # 200 draws over (typically) 4 solutions: should hit them all.
        self.assertEqual(seen, all_sols_as_tuples)


class TestRandomWalkExtend(unittest.TestCase):
    """random_walk_extend: structured WalkResult + tie-breaking + telemetry."""

    def test_repetition_extension_succeeds(self):
        """ZZ + {X1,X2} should be correctable by adding one ancilla + one stabilizer."""
        stabilizers = [stim.PauliString("ZZ")]
        errors = [stim.PauliString("X_"), stim.PauliString("_X")]
        result = random_walk_extend(
            stabilizers, errors, extra_support=stim.PauliString("Z"),
            max_stabilizers_per_walk=1, max_walks=10, seed_val=12,
        )
        self.assertTrue(result.succeeded)
        self.assertEqual(result.uncorrectables_remaining, 0)
        self.assertEqual(result.n_stabilizers_added, 1)
        self.assertEqual(len(result.walks), 10)
        # Confirm the returned code actually has no uncorrectable errors.
        self.assertEqual(len(get_uncorrectable_errors(result.code, errors)), 0)

    def test_telemetry_consistency(self):
        """successful_walks count must equal the # of walks with uncorrectables=0."""
        stabilizers = [stim.PauliString("ZZ_"), stim.PauliString("_ZZ")]
        errors = [stim.PauliString("___"), stim.PauliString("Z__"),
                  stim.PauliString("_Z_"), stim.PauliString("__Z")]
        result = random_walk_extend(
            stabilizers, errors, extra_support=stim.PauliString("X"),
            max_stabilizers_per_walk=1, max_walks=15, seed_val=12,
        )
        explicit = sum(1 for w in result.walks if w.uncorrectables_remaining == 0)
        self.assertEqual(result.successful_walks, explicit)
        # If any walk succeeded, the returned best code should be successful too.
        if result.successful_walks > 0:
            self.assertTrue(result.succeeded)

    def test_ancilla_budget_equivalent_to_extra_support_on_single_step(self):
        """For a single-step walk, ancilla_budget=1 should be equivalent in shape to
        extra_support=single-qubit (both produce a 3-qubit result for ZZ + {X1,X2})."""
        stabilizers = [stim.PauliString("ZZ")]
        errors = [stim.PauliString("X_"), stim.PauliString("_X")]
        result = random_walk_extend(
            stabilizers, errors, ancilla_budget=1,
            max_stabilizers_per_walk=1, max_walks=10, seed_val=12,
        )
        self.assertTrue(result.succeeded)
        # Result code lives on 3 qubits, 2 stabilizers (= [[3,1]]).
        self.assertEqual(max(len(g) for g in result.code), 3)
        self.assertEqual(len(result.code), 2)

    def test_smart_walk_beats_random_walk_on_fh_8_6(self):
        """n_solution_samples > 1 should hit the slide-14 [[10,6]] target far more
        often than n_solution_samples=1 (pure random walk). At n_samples=1 the hit
        rate is ~4% for the optimal 2-stabilizer extension; at n_samples=16 it's
        ~100%."""
        g_up = stim.PauliString("Z_Z_Z_Z_")
        g_down = stim.PauliString("_Z_Z_Z_Z")
        stabilizers = [g_up, g_down]
        errors = [stim.PauliString("_" * 8)]
        for i in range(8):
            mask = [0] * 8
            mask[i] = 1
            errors.append(stim.PauliString(mask))

        common_kwargs = dict(
            stabilizers=stabilizers, errors=errors, ancilla_budget=2,
            max_stabilizers_per_walk=3, max_walks=20, seed_val=137,
        )
        random_result = random_walk_extend(**common_kwargs, n_solution_samples=1)
        smart_result = random_walk_extend(**common_kwargs, n_solution_samples=16)
        # Smart walk must succeed more often than random walk (strict improvement,
        # not just equality — under this seed random gets ~half, smart gets all).
        self.assertGreater(smart_result.successful_walks, random_result.successful_walks)
        # And the best code it returns should be the slide-14 [[10,6]] (2 stabs added).
        self.assertTrue(smart_result.succeeded)
        self.assertEqual(smart_result.n_stabilizers_added, 2)

    def test_ancilla_budget_and_extra_support_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            random_walk_extend(
                [stim.PauliString("ZZ")], [stim.PauliString("X_")],
                extra_support=stim.PauliString("Z"),
                ancilla_budget=1,
            )

    def test_failure_signal_when_budget_too_small(self):
        """If max_stabilizers_per_walk is smaller than the actual extension needs,
        every walk should fail and succeeded must be False."""
        # Shor seed needs to add at least 2 X-stabilizers to correct phase flips —
        # cap walks at 1 stabilizer so they can't possibly succeed.
        stabilizers = [
            stim.PauliString("ZZ_______"),
            stim.PauliString("_ZZ______"),
            stim.PauliString("___ZZ____"),
            stim.PauliString("____ZZ___"),
            stim.PauliString("______ZZ_"),
            stim.PauliString("_______ZZ"),
        ]
        # All single-qubit errors.
        errs = [stim.PauliString("_________")]
        for i in range(9):
            for p in (1, 2, 3):
                mask = [0] * 9
                mask[i] = p
                errs.append(stim.PauliString(mask))
        result = random_walk_extend(
            stabilizers, errs,
            max_stabilizers_per_walk=1, max_walks=5, seed_val=12,
        )
        self.assertFalse(result.succeeded)
        self.assertEqual(result.successful_walks, 0)
        self.assertGreater(result.uncorrectables_remaining, 0)


class TestBeamSearchExtend(unittest.TestCase):
    """beam_search_extend should find slide-target extensions and at least match
    random_walk_extend with similar budget."""

    def test_finds_slide_14_optimum_for_fh_8_6(self):
        g_up = stim.PauliString("Z_Z_Z_Z_")
        g_down = stim.PauliString("_Z_Z_Z_Z")
        stabilizers = [g_up, g_down]
        errors = [stim.PauliString("_" * 8)]
        for i in range(8):
            mask = [0] * 8
            mask[i] = 1
            errors.append(stim.PauliString(mask))

        result = beam_search_extend(
            stabilizers, errors, ancilla_budget=2,
            max_stabilizers=3, beam_width=8, n_expansions_per_slot=4,
            n_solution_samples=8, seed_val=137,
        )
        self.assertTrue(result.succeeded)
        # Slide 14 hand-design is [[10,6]] with 2 added stabilizers.
        self.assertEqual(result.n_stabilizers_added, 2)
        n = max(len(g) for g in result.code)
        self.assertEqual(n, 10)
        self.assertEqual(n - len(result.code), 6)

    def test_returns_failure_signal_when_max_stabilizers_too_low(self):
        """With max_stabilizers=1 the FH [[8,6]] target can't be reached — beam
        must report succeeded=False."""
        g_up = stim.PauliString("Z_Z_Z_Z_")
        g_down = stim.PauliString("_Z_Z_Z_Z")
        errors = [stim.PauliString("_" * 8)]
        for i in range(8):
            mask = [0] * 8
            mask[i] = 1
            errors.append(stim.PauliString(mask))
        result = beam_search_extend(
            [g_up, g_down], errors, ancilla_budget=2,
            max_stabilizers=1, beam_width=4, n_expansions_per_slot=2,
            seed_val=137,
        )
        self.assertFalse(result.succeeded)
        self.assertGreater(result.uncorrectables_remaining, 0)


if __name__ == "__main__":
    unittest.main()