"""
node_map_gen.py
================
Generates the dungeon layout as a sparse, tree-shaped graph of rooms placed
on an actual grid, so that:

  * every room's neighbours are physically N/S/E/W of it (a real floor plan,
    not an abstract node graph) — this is what lets rogue.py let the player
    walk between rooms instead of clicking a map.
  * the layout is a spanning tree (grown with a randomized Prim's-style walk),
    so there is never more than one route between any two rooms — this is
    deliberately NOT a fully-connected mesh. Every room you can reach, you
    reach by a single, specific path of doors.
  * the boss room is placed at the room with the greatest tree-distance from
    the start, so a path from start to boss is guaranteed to exist by
    construction (it's the literal path used to compute that distance).
  * only four room kinds exist: "nothing" (incl. the start room), "enemy",
    "loot", "boss". No elites, no shops — kept deliberately simple.

Rooms are still allowed a *few* extra doors beyond the spanning tree (see
`extra_loops`) purely to avoid every dungeon being a single strict corridor
with no interesting branching-and-rejoining, but the default is small and
this never turns the layout into a mesh.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# Cardinal directions and their opposites / (dcol, drow) deltas.
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
DELTA = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}

ROOM_KINDS = ("nothing", "enemy", "loot", "boss")


@dataclass
class MapNode:
    idx: int
    col: int
    row: int
    kind: str = "nothing"
    cleared: bool = False
    looted: bool = False   # has the loot pedestal in this room already been used?


class NodeMap:
    def __init__(self, nodes: List[MapNode], edges: Dict[int, Set[int]], start: int, boss: int):
        self.nodes = nodes
        self.edges = edges
        self.start = start
        self.boss = boss

    def neighbors(self, i: int) -> List[int]:
        return list(self.edges.get(i, set()))

    def direction(self, a: int, b: int) -> Optional[str]:
        """Cardinal direction to walk from room `a` to reach room `b`.
        Returns None if the rooms aren't grid-adjacent (shouldn't happen for
        connected edges, since we only ever add edges between adjacent cells)."""
        na, nb = self.nodes[a], self.nodes[b]
        dc, dr = nb.col - na.col, nb.row - na.row
        for d, (ddc, ddr) in DELTA.items():
            if (ddc, ddr) == (dc, dr):
                return d
        return None

    def doors(self, i: int) -> Dict[str, int]:
        """{direction: neighbor_idx} for every connected neighbor of room i."""
        out = {}
        for j in self.neighbors(i):
            d = self.direction(i, j)
            if d is not None:
                out[d] = j
        return out


def _bfs_depths(node_count: int, edges: Dict[int, Set[int]], start: int) -> List[int]:
    from collections import deque
    depth = [-1] * node_count
    depth[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in edges.get(u, set()):
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                q.append(v)
    return depth


def _add_edge(edges: Dict[int, Set[int]], a: int, b: int):
    edges.setdefault(a, set()).add(b)
    edges.setdefault(b, set()).add(a)


def compute_node_depth(node_map: NodeMap, start: int) -> List[int]:
    """BFS from start; returns depth[i] for every node i.

    Single shared implementation — previously duplicated near-identically in
    both rogue.py and bot_runner.py.
    """
    return _bfs_depths(len(node_map.nodes), node_map.edges, start)


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
    Grow a branching, tree-shaped dungeon on a `cols` x `rows` grid.

    Algorithm (randomized Prim's-style growth — produces a spanning tree by
    construction, i.e. never a fully-connected mesh):
      1. Start from a random cell; mark it visited.
      2. Maintain a frontier of (visited_cell -> unvisited_adjacent_cell) pairs.
      3. Repeatedly pick a random frontier pair, carve a door between them,
         mark the new cell visited, and add its own unvisited neighbours to
         the frontier. Stop once `node_count` cells are visited (or the grid
         is exhausted).
      4. Optionally add a handful of `extra_loops` extra doors between
         already-adjacent visited cells that aren't yet connected, purely for
         minor branch variety. Kept small on purpose.
      5. The start room is the first cell visited. The boss room is whichever
         visited cell has the greatest tree-distance (BFS depth) from start —
         guaranteeing a path from start to boss exists.
      6. Remaining rooms are randomly assigned "enemy" / "loot" / "nothing"
         by weight.
    """
    rng = random.Random(seed)

    if node_count > cols * rows:
        # Grow the grid just enough to fit the requested room count.
        side = int(math.ceil(math.sqrt(node_count)))
        cols = rows = side

    def cell_id(c, r):
        return r * cols + c

    def in_bounds(c, r):
        return 0 <= c < cols and 0 <= r < rows

    start_c, start_r = rng.randrange(cols), rng.randrange(rows)
    start_cell = cell_id(start_c, start_r)
    visited: Set[int] = {start_cell}
    coord_of: Dict[int, Tuple[int, int]] = {start_cell: (start_c, start_r)}
    edges: Dict[int, Set[int]] = {}

    # Frontier: list of (visited_cell_id, unvisited_cell_id) candidate doors.
    frontier: List[Tuple[int, int]] = []

    def push_frontier(cid, c, r):
        for d, (dc, dr) in DELTA.items():
            nc, nr = c + dc, r + dr
            if in_bounds(nc, nr):
                nid = cell_id(nc, nr)
                if nid not in visited:
                    frontier.append((cid, nid))

    push_frontier(start_cell, start_c, start_r)

    while len(visited) < node_count and frontier:
        pick = rng.randrange(len(frontier))
        a_id, b_id = frontier.pop(pick)
        if b_id in visited:
            continue  # became visited via another frontier edge meanwhile
        bc, br = b_id % cols, b_id // cols
        visited.add(b_id)
        coord_of[b_id] = (bc, br)
        _add_edge(edges, a_id, b_id)
        push_frontier(b_id, bc, br)

    # Re-index visited cells 0..N-1 in a stable order (grid scan order) so
    # MapNode indices are compact regardless of grid size.
    ordered_cell_ids = sorted(visited)
    remap = {cell_id_: i for i, cell_id_ in enumerate(ordered_cell_ids)}

    nodes: List[MapNode] = []
    for cell_id_ in ordered_cell_ids:
        c, r = coord_of[cell_id_]
        nodes.append(MapNode(idx=remap[cell_id_], col=c, row=r))

    remapped_edges: Dict[int, Set[int]] = {}
    for a_id, nbrs in edges.items():
        for b_id in nbrs:
            _add_edge(remapped_edges, remap[a_id], remap[b_id])

    start_idx = remap[start_cell]

    # A handful of extra doors between grid-adjacent visited rooms that
    # aren't already connected — kept small so it never becomes a mesh.
    n = len(nodes)
    coord_to_idx = {(nd.col, nd.row): nd.idx for nd in nodes}
    added = 0
    tries = 0
    while added < extra_loops and tries < 200:
        tries += 1
        i = rng.randrange(n)
        ni = nodes[i]
        d = rng.choice(list(DELTA.keys()))
        dc, dr = DELTA[d]
        j = coord_to_idx.get((ni.col + dc, ni.row + dr))
        if j is None or j == i:
            continue
        if j in remapped_edges.get(i, set()):
            continue
        _add_edge(remapped_edges, i, j)
        added += 1

    # Boss = farthest room from start by tree distance -> guarantees a path.
    depths = _bfs_depths(n, remapped_edges, start_idx)
    boss_idx = max(range(n), key=lambda i: depths[i])

    nodes[start_idx].kind = "nothing"
    nodes[boss_idx].kind = "boss"

    # Assign remaining kinds by weight.
    nothing_weight = max(0.0, 1.0 - enemy_weight - loot_weight)
    kinds = ["enemy", "loot", "nothing"]
    weights = [enemy_weight, loot_weight, nothing_weight]
    for nd in nodes:
        if nd.idx in (start_idx, boss_idx):
            continue
        nd.kind = rng.choices(kinds, weights=weights, k=1)[0]

    return NodeMap(nodes, remapped_edges, start_idx, boss_idx)
