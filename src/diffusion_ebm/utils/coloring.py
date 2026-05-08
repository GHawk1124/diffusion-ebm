"""Chromatic coloring of factor-graph nodes for THRML block-Gibbs.

THRML's block-Gibbs sampler needs a partition of the free nodes into
color classes such that no edge of the factor graph connects two nodes
in the same class.  We compute that partition with NetworkX's DSATUR
heuristic — exact for bipartite graphs (giving the M1 even/odd split on
chains) and small enough graphs (≤ a few dozen holes) to be cheap.

This module is the single place that converts an integer-indexed edge
list into a `list[thrml.Block]`.  The joint sampler in
``src/diffusion_ebm/sampler/thrml_joint.py`` consumes the result.
"""

from __future__ import annotations

from typing import Sequence

import networkx as nx

from thrml import Block


def chromatic_blocks(
    nodes: Sequence,
    edges: Sequence[tuple[int, int]],
) -> list[Block]:
    """Partition `nodes` into THRML `Block`s by graph coloring.

    `nodes` is an arbitrary list of THRML node instances (e.g.
    `CategoricalNode()`); `edges` is a list of ``(i, j)`` integer index
    pairs into `nodes`.  Self-loops and duplicate edges are tolerated and
    have no effect on the coloring.

    Returns a list of `Block`s, one per color class, sorted by color id.
    A node with no edges goes into its own (or shared) color class as
    the heuristic sees fit; either way every node appears in exactly one
    block.

    Example (4-node chain):

        >>> from thrml import CategoricalNode
        >>> ns = [CategoricalNode() for _ in range(4)]
        >>> blocks = chromatic_blocks(ns, [(0, 1), (1, 2), (2, 3)])
        >>> len(blocks)
        2
        >>> sorted(len(b.nodes) for b in blocks)
        [2, 2]
    """
    g = nx.Graph()
    g.add_nodes_from(range(len(nodes)))
    g.add_edges_from((i, j) for i, j in edges if i != j)

    coloring = nx.coloring.greedy_color(g, strategy="DSATUR")
    by_color: dict[int, list] = {}
    for idx, color in coloring.items():
        by_color.setdefault(color, []).append(nodes[idx])

    return [Block(by_color[c]) for c in sorted(by_color)]
