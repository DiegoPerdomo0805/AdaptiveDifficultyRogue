import node_map_gen as nmg
import bot_runner
import controllers as ctl


def test_maps_connected_and_boss_reachable():
    for seed in range(200):
        m = nmg.generate_node_map_graph(seed=seed)
        depths = nmg.compute_node_depth(m)
        assert set(depths) == set(m.nodes)          # every node reachable
        assert m.boss in depths and m.nodes[m.boss].kind == "boss"
        assert depths[m.boss] == max(depths.values())


def test_dfs_backtracking_reaches_boss_when_invincible():
    c = ctl.make_controller("heuristic")
    for seed in range(100000, 100060):
        _, samples, reason, boss, _ = bot_runner.simulate_run(seed, c, invincible=True)
        assert reason == "victory" and boss
        seq = [s.room_seq for s in samples]
        assert seq == list(range(1, len(seq) + 1))
        assert len({s.room_idx for s in samples}) == len(samples)   # each room sampled once


def test_equivalent_seeds_share_map_and_archetype():
    a = bot_runner.simulate_run(100123, ctl.make_controller("fixed"), invincible=True)
    b = bot_runner.simulate_run(100123, ctl.make_controller("heuristic"), invincible=True)
    assert a[4] == b[4]
    assert [s.room_idx for s in a[1]] == [s.room_idx for s in b[1]]
    assert [s.room_kind for s in a[1]] == [s.room_kind for s in b[1]]
