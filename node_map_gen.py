"""
node_map_gen.py
================
Darkest-Dungeon-style branching dungeon graph generator.

Guarantees:
  * every node is reachable from `start`
  * `boss` is reachable from every leaf (there is always a path to the boss)
  * the graph is NOT a fully-connected mesh (it's a spanning tree over a
    grid, plus a small number of extra "loop" edges between grid-adjacent
    nodes already in the tree)
  * exactly 4 room kinds: "enemy", "boss", "loot", "nothing"

Unchanged from the Phase 2a redesign — no finding in the review targets
this file directly.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random

OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
DELTA = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


@dataclass
class MapNode:
    idx: int
    col: int
    row: int
    kind: str = "nothing"          # enemy | boss | loot | nothing
    cleared: bool = False
    looted: bool = False


class NodeMap:
    def __init__(self):
        self.nodes: Dict[int, MapNode] = {}
        self.edges: Dict[int, Dict[str, int]] = {}   # idx -> {dir: idx}
        self.start: Optional[int] = None
        self.boss: Optional[int] = None

    def neighbors(self, idx: int) -> Dict[str, int]:
        return self.edges.get(idx, {})

    def direction(self, a: int, b: int) -> Optional[str]:
        for d, nb in self.edges.get(a, {}).items():
            if nb == b:
                return d
        return None

    def doors(self, idx: int) -> List[str]:
        return list(self.edges.get(idx, {}).keys())

    def _add_edge(self, a: int, b: int, d: str):
        self.edges.setdefault(a, {})
        self.edges.setdefault(b, {})
        self.edges[a][d] = b
        self.edges[b][OPPOSITE[d]] = a


def _bfs_depths(node_map: NodeMap, start: int) -> Dict[int, int]:
    depths = {start: 0}
    queue = [start]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        for _, nb in node_map.edges.get(cur, {}).items():
            if nb not in depths:
                depths[nb] = depths[cur] + 1
                queue.append(nb)
    return depths


def compute_node_depth(node_map: NodeMap, start: Optional[int] = None) -> Dict[int, int]:
    s = start if start is not None else node_map.start
    return _bfs_depths(node_map, s)


def generate_node_map_graph(
    node_count: int = 16,
    cols: int = 6,
    rows: int = 6,
    extra_loops: int = 2,
    enemy_weight: float = 0.55,
    loot_weight: float = 0.22,
    seed: Optional[int] = None,
) -> NodeMap:
    """
    Randomized-Prim's-style spanning-tree growth on a `cols` x `rows` grid.
    Picks `node_count` distinct grid cells, grows a tree from a random start
    cell by repeatedly adding a random frontier cell (grid-adjacent to an
    already-placed cell), then adds a small number of extra edges between
    grid-adjacent placed cells that aren't already connected (creates
    alternate routes without ever becoming a full mesh). The boss is the
    node with maximum BFS depth from start.
    """
    rng = random.Random(seed)
    node_count = min(node_count, cols * rows)

    all_cells = [(c, r) for c in range(cols) for r in range(rows)]
    rng.shuffle(all_cells)

    start_cell = all_cells[0]
    placed_cells = {start_cell}
    frontier: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []

    def push_frontier(cell):
        c, r = cell
        for d, (dc, dr) in DELTA.items():
            nc, nr = c + dc, r + dr
            if 0 <= nc < cols and 0 <= nr < rows and (nc, nr) not in placed_cells:
                frontier.append((cell, (nc, nr), d))

    push_frontier(start_cell)

    cell_to_idx: Dict[Tuple[int, int], int] = {start_cell: 0}
    node_map = NodeMap()
    node_map.nodes[0] = MapNode(idx=0, col=start_cell[0], row=start_cell[1])
    next_idx = 1

    tree_edges: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []

    while len(placed_cells) < node_count and frontier:
        rng.shuffle(frontier)
        a, b, d = frontier.pop()
        if b in placed_cells:
            continue
        placed_cells.add(b)
        cell_to_idx[b] = next_idx
        node_map.nodes[next_idx] = MapNode(idx=next_idx, col=b[0], row=b[1])
        next_idx += 1
        tree_edges.append((a, b, d))
        push_frontier(b)

    for a, b, d in tree_edges:
        node_map._add_edge(cell_to_idx[a], cell_to_idx[b], d)

    # A few extra loop edges between grid-adjacent already-placed cells that
    # aren't yet connected. Kept small (`extra_loops`) so this never becomes
    # a full mesh -- most rooms should still have exactly 1-2 exits.
    placed_list = list(placed_cells)
    candidate_pairs = []
    for cell in placed_list:
        c, r = cell
        for d, (dc, dr) in DELTA.items():
            nb = (c + dc, r + dr)
            if nb in placed_cells:
                a_idx, b_idx = cell_to_idx[cell], cell_to_idx[nb]
                if node_map.direction(a_idx, b_idx) is None:
                    candidate_pairs.append((cell, nb, d))
    rng.shuffle(candidate_pairs)
    added = 0
    seen_pairs = set()
    for a, b, d in candidate_pairs:
        if added >= extra_loops:
            break
        a_idx, b_idx = cell_to_idx[a], cell_to_idx[b]
        key = tuple(sorted((a_idx, b_idx)))
        if key in seen_pairs:
            continue
        if node_map.direction(a_idx, b_idx) is None:
            node_map._add_edge(a_idx, b_idx, d)
            seen_pairs.add(key)
            added += 1

    node_map.start = cell_to_idx[start_cell]
    depths = _bfs_depths(node_map, node_map.start)
    boss_idx = max(depths.items(), key=lambda kv: kv[1])[0]
    node_map.boss = boss_idx

    for idx, node in node_map.nodes.items():
        if idx == node_map.start:
            node.kind = "nothing"
        elif idx == boss_idx:
            node.kind = "boss"
        else:
            roll = rng.random()
            if roll < enemy_weight:
                node.kind = "enemy"
            elif roll < enemy_weight + loot_weight:
                node.kind = "loot"
            else:
                node.kind = "nothing"

    return node_map
