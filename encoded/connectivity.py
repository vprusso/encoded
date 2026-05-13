"""Hardware connectivity helpers.

A connectivity graph here is an `adjacency: dict[int, set[int]]` mapping each
qubit index to the set of qubits it shares a native two-qubit gate with. A
Pauli operator's *support* is its set of non-identity qubits; the operator is
"connectivity-respecting" iff its support induces a connected subgraph (i.e.
the corresponding stabilizer can be measured using nearest-neighbor gates on
the device without SWAP overhead).

When a code's stabilizers are constrained to respect a hardware connectivity
graph, the algorithm produces codes that map directly to that device.
"""

from collections.abc import Iterable

import stim


def all_to_all(n: int) -> dict[int, set[int]]:
    """Complete graph: every pair of qubits shares an edge."""
    return {i: set(range(n)) - {i} for i in range(n)}


def linear(n: int) -> dict[int, set[int]]:
    """Linear nearest-neighbor: edges (0,1), (1,2), ..., (n-2, n-1)."""
    g: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n - 1):
        g[i].add(i + 1)
        g[i + 1].add(i)
    return g


def ring(n: int) -> dict[int, set[int]]:
    """Linear NN plus the wrap-around edge (n-1, 0)."""
    g = linear(n)
    if n >= 2:
        g[0].add(n - 1)
        g[n - 1].add(0)
    return g


def from_edges(n: int, edges: Iterable[tuple[int, int]]) -> dict[int, set[int]]:
    """Build an adjacency dict from an edge list. Each edge (a, b) is added
    in both directions; duplicates are deduplicated by set semantics."""
    g: dict[int, set[int]] = {i: set() for i in range(n)}
    for a, b in edges:
        if a == b:
            continue
        g[a].add(b)
        g[b].add(a)
    return g


def has_connected_support(ps: stim.PauliString, graph: dict[int, set[int]]) -> bool:
    """True iff the non-identity support of `ps` induces a connected subgraph
    in `graph`. An empty or weight-1 support is trivially connected."""
    support = [i for i, p in enumerate(ps) if p != 0]
    if len(support) <= 1:
        return True
    support_set = set(support)
    seen = {support[0]}
    stack = [support[0]]
    while stack:
        node = stack.pop()
        for neighbor in graph.get(node, ()):
            if neighbor in support_set and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == support_set
