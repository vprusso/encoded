import unittest
import numpy as np
import stim
import random
from encoded.add_stabilizers import (
    _form_linear_system, add_stabilizer, is_in_stabilizer_group, knill_laflamme_cost_function,
    build_code_randomly, prune_duplicate_pauli_strings, random_depth_first_search,
    _legacy_group_membership_check,
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


if __name__ == "__main__":
    unittest.main()