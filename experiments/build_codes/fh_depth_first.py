import stim
from encoded.add_stabilizers import random_depth_first_search, is_in_stabilizer_group

def all_single_qubit_errors(n: int):
    all_errors = [stim.PauliString('_' * n)]
    for i in range(n):
        for p in [1, 2, 3]:
            mask = [0] * n
            mask[i] = p
            all_errors.append(stim.PauliString(mask))
    return all_errors

stabilizers = [
    stim.PauliString("ZIZII"),
    stim.PauliString("IZIZI"),
    stim.PauliString("IZZIZ")
]
errors = all_single_qubit_errors(4)
new_stabilizers = random_depth_first_search(stabilizers, errors, steps=3, max_tries=20, extra_support=stim.PauliString("Z"))
print(f"New code has {len(new_stabilizers)} stabilizers.")
for stab in new_stabilizers:
    print(stab)

for s1 in new_stabilizers:
    for s2 in new_stabilizers:
        assert s1.commutes(s2)

for e1 in errors:
    for e2 in errors:
        if e1 != e2:
            e = e1 * e2
            anticommute_tests = [not e.commutes(gen) for gen in new_stabilizers]
            in_group = is_in_stabilizer_group(e, new_stabilizers)
            assert any(anticommute_tests) or in_group, f"e1 = {e1}, e2 = {e2}"