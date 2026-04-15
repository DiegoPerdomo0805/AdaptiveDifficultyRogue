import math
import random
from typing import Dict, List, Optional, Set

from node_map_gen import NodeMap, MapNode, add_edge, bfs_components, dist, segments_intersect, shortest_path


def generate_node_map_graph(
    node_count: int = 18,
    width: int = 820,
    height: int = 420,
    margin: int = 40,
    min_sep: float = 55.0,
    k_nearest: int = 3,
    max_degree: int = 4,
    allow_crossings: bool = False,
    extra_edges: int = 3,
    seed: Optional[int] = None,
) -> NodeMap:

    rng = random.Random(seed)

    # Place nodes on a (rough) grid so edges are strictly N/S/E/W.
    # This makes the generated graph behave like a 4-way dungeon map.
    cols = int(math.sqrt(node_count))
    if cols * cols < node_count:
        cols += 1
    rows = (node_count + cols - 1) // cols

    x_coords = [
        margin + (width - 2 * margin) * (i / (cols - 1 if cols > 1 else 1))
        for i in range(cols)
    ]
    y_coords = [
        margin + (height - 2 * margin) * (i / (rows - 1 if rows > 1 else 1))
        for i in range(rows)
    ]

    nodes: List[MapNode] = []
    for r in range(rows):
        for c in range(cols):
            if len(nodes) >= node_count:
                break
            nodes.append(MapNode(idx=len(nodes), x=x_coords[c], y=y_coords[r]))
        if len(nodes) >= node_count:
            break

    far = (0, 1, -1.0)
    for i in range(node_count):
        for j in range(i + 1, node_count):
            d = dist((nodes[i].x, nodes[i].y), (nodes[j].x, nodes[j].y))
            if d > far[2]:
                far = (i, j, d)
    start, boss = far[0], far[1]
    nodes[start].kind = "start"
    nodes[boss].kind = "boss"

    def is_cardinal(a: MapNode, b: MapNode, tol: float = 1e-6) -> bool:
        return abs(a.x - b.x) < tol or abs(a.y - b.y) < tol

    edges: Dict[int, Set[int]] = {}

    # Ensure connectivity along the grid in NESW directions.
    for i in range(node_count):
        r = i // cols
        c = i % cols

        # connect to east neighbor
        if c + 1 < cols:
            j = i + 1
            if j < node_count:
                add_edge(edges, i, j)

        # connect to south neighbor
        j = i + cols
        if j < node_count:
            add_edge(edges, i, j)

    # Add some additional cardinal edges to meet k_nearest.
    def cardinal_neighbors(i: int) -> List[int]:
        base = nodes[i]
        candidates = []
        for j in range(node_count):
            if i == j:
                continue
            if not is_cardinal(base, nodes[j]):
                continue
            candidates.append((dist((base.x, base.y), (nodes[j].x, nodes[j].y)), j))
        candidates.sort(key=lambda t: t[0])
        return [j for _, j in candidates]

    for i in range(node_count):
        neighbors = cardinal_neighbors(i)[:k_nearest]
        for j in neighbors:
            add_edge(edges, i, j)


    def edge_list():
        seen = set()
        out = []
        for a, nbrs in edges.items():
            for b in nbrs:
                if (b, a) in seen:
                    continue
                seen.add((a, b))
                out.append((a, b))
        return out

    def degree(i): 
        return len(edges.get(i, set()))


    for i in range(node_count):
        while degree(i) > max_degree:
            nbrs = list(edges[i])
            nbrs.sort(key=lambda j: dist((nodes[i].x, nodes[i].y), (nodes[j].x, nodes[j].y)), reverse=True)
            remove_edge(edges, i, nbrs[0])


    if not allow_crossings:
        changed = True
        while changed:
            changed = False
            el = edge_list()
            for a, b in el:
                p1 = (nodes[a].x, nodes[a].y)
                p2 = (nodes[b].x, nodes[b].y)
                for c, d in el:
                    if len({a, b, c, d}) < 4:
                        continue  
                    q1 = (nodes[c].x, nodes[c].y)
                    q2 = (nodes[d].x, nodes[d].y)
                    if segments_intersect(p1, p2, q1, q2):
                        dab = dist(p1, p2)
                        dcd = dist(q1, q2)
                        if dab >= dcd:
                            remove_edge(edges, a, b)
                        else:
                            remove_edge(edges, c, d)
                        changed = True
                        break
                if changed:
                    break


    comps = bfs_components(node_count, edges)
    while len(comps) > 1:
        comp_a = comps[0]
        comp_b = comps[1]
        best = (None, None, 1e18)
        for i in comp_a:
            for j in comp_b:
                d = dist((nodes[i].x, nodes[i].y), (nodes[j].x, nodes[j].y))
                if d < best[2]:
                    best = (i, j, d)
        add_edge(edges, best[0], best[1])
        comps = bfs_components(node_count, edges)

    path = shortest_path(node_count, edges, start, boss)
    if not path:
        add_edge(edges, start, boss)
        path = shortest_path(node_count, edges, start, boss)
        if not path:
            raise RuntimeError("Failed to guarantee start->boss path.")


    def can_add(a, b) -> bool:
        # Only allow NESW connections (no diagonals)
        if not is_cardinal(nodes[a], nodes[b]):
            return False
        if b in edges.get(a, set()):
            return False
        if degree(a) >= max_degree or degree(b) >= max_degree:
            return False
        if not allow_crossings:
            p1 = (nodes[a].x, nodes[a].y)
            p2 = (nodes[b].x, nodes[b].y)
            for c, d in edge_list():
                if len({a, b, c, d}) < 4:
                    continue
                q1 = (nodes[c].x, nodes[c].y)
                q2 = (nodes[d].x, nodes[d].y)
                if segments_intersect(p1, p2, q1, q2):
                    return False
        return True

    tries = 0
    added = 0
    while added < extra_edges and tries < 2000:
        tries += 1
        a = rng.randrange(node_count)
        b = rng.randrange(node_count)
        if a == b:
            continue
        # Only add additional edges along NESW directions.
        if not is_cardinal(nodes[a], nodes[b]):
            continue
        d = dist((nodes[a].x, nodes[a].y), (nodes[b].x, nodes[b].y))
        if d < min_sep * 1.2 or d > min_sep * 4.0:
            continue
        if can_add(a, b):
            add_edge(edges, a, b)
            added += 1

    path_set = set(path)
    sd = [(dist((nodes[i].x, nodes[i].y), (nodes[start].x, nodes[start].y)), i) for i in range(node_count)]
    sd.sort()
    ordered = [i for _, i in sd]

    for i in ordered:
        if i in (start, boss):
            continue
        nodes[i].kind = "combat"

    mid = ordered[node_count//3: 2*node_count//3]
    far_nodes = ordered[2*node_count//3:]

    for i in rng.sample(mid, k=min(2, len(mid))):
        if nodes[i].kind not in ("start", "boss"):
            nodes[i].kind = "shop"

    for i in rng.sample(far_nodes, k=min(2, len(far_nodes))):
        if nodes[i].kind not in ("start", "boss"):
            nodes[i].kind = "elite"

    candidates = [i for i in range(node_count) if nodes[i].kind == "combat"]
    loot_n = max(2, int(0.25 * len(candidates)))
    for i in rng.sample(candidates, k=min(loot_n, len(candidates))):
        nodes[i].kind = "loot"

    return NodeMap(nodes, edges, start, boss)
