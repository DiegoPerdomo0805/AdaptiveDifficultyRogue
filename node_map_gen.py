import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional



@dataclass
class MapNode:
    idx: int
    x: float
    y: float
    kind: str = "combat"   
    cleared: bool = False

class NodeMap:
    def __init__(self, nodes: List[MapNode], edges: Dict[int, Set[int]], start: int, boss: int):
        self.nodes = nodes
        self.edges = edges
        self.start = start
        self.boss = boss

    def neighbors(self, i: int) -> List[int]:
        return list(self.edges.get(i, set()))




def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def segments_intersect(p1, p2, q1, q2) -> bool:

    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    def on_segment(a, b, c):
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)


    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True


    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, p2, q2): return True
    if o3 == 0 and on_segment(q1, q2, p1): return True
    if o4 == 0 and on_segment(q1, q2, p2): return True
    return False




def add_edge(edges: Dict[int, Set[int]], a: int, b: int):
    if a == b: 
        return
    edges.setdefault(a, set()).add(b)
    edges.setdefault(b, set()).add(a)

def remove_edge(edges: Dict[int, Set[int]], a: int, b: int):
    edges.get(a, set()).discard(b)
    edges.get(b, set()).discard(a)

def bfs_components(n: int, edges: Dict[int, Set[int]]) -> List[List[int]]:
    seen = [False]*n
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        q = [i]
        seen[i] = True
        comp = []
        while q:
            u = q.pop()
            comp.append(u)
            for v in edges.get(u, set()):
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        comps.append(comp)
    return comps

def shortest_path(n: int, edges: Dict[int, Set[int]], start: int, goal: int) -> List[int]:
    from collections import deque
    prev = [-1]*n
    dq = deque([start])
    prev[start] = start
    while dq:
        u = dq.popleft()
        if u == goal:
            break
        for v in edges.get(u, set()):
            if prev[v] == -1:
                prev[v] = u
                dq.append(v)
    if prev[goal] == -1:
        return []
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path




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

    nodes: List[MapNode] = []
    attempts = 0
    max_attempts = node_count * 500

    while len(nodes) < node_count and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(margin, width - margin)
        y = rng.uniform(margin, height - margin)
        ok = True
        for n in nodes:
            if dist((x, y), (n.x, n.y)) < min_sep:
                ok = False
                break
        if ok:
            nodes.append(MapNode(idx=len(nodes), x=x, y=y))

    if len(nodes) < node_count:
        raise RuntimeError(f"Could not place {node_count} nodes with min_sep={min_sep} in {width}x{height}.")


    far = (0, 1, -1.0)
    for i in range(node_count):
        for j in range(i+1, node_count):
            d = dist((nodes[i].x, nodes[i].y), (nodes[j].x, nodes[j].y))
            if d > far[2]:
                far = (i, j, d)
    start, boss = far[0], far[1]
    nodes[start].kind = "start"
    nodes[boss].kind = "boss"


    edges: Dict[int, Set[int]] = {}
    for i in range(node_count):
        if i == boss:
            pass
        dists = []
        for j in range(node_count):
            if i == j:
                continue
            dists.append((dist((nodes[i].x, nodes[i].y), (nodes[j].x, nodes[j].y)), j))
        dists.sort(key=lambda t: t[0])
        for _, j in dists[:k_nearest]:
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