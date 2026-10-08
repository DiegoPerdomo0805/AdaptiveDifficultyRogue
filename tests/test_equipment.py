"""Equipment parity: the bot applies the same functional item effects as the
human game (shared dda_core functions), and representative items change the
corresponding bot combat variables / outcomes."""
import random

import pytest

import bot_runner
import dda_core as core

W = {w.item_id: w for w in core.WEAPONS}
S = {s.item_id: s for s in core.SPELLS}
A = {a.item_id: a for a in core.ARMORS}
B = {b.item_id: b for b in core.BOOTS}

CFG = {"enemy_hp_mult": 1.5, "enemy_dmg_mult": 1.3, "enemy_speed_mult": 1.15, "spawn_count": 4}
MELEE = bot_runner.Archetype("t_melee", "melee", 1.0, 0.0, 0.0, 0.0, [0.25] * 4)
MAGIC = bot_runner.Archetype("t_magic", "magic", 0.0, 1.0, 0.0, 0.0, [0.25] * 4)
TANK = bot_runner.Archetype("t_defend", "melee", 0.0, 0.0, 1.0, 0.0, [0.25] * 4)


def eq(weapon="weapon.sword", spell="spell.magic_bolt", armor="armor.cloth_wrap", boots="boots.leather"):
    return core.Equipment(weapon=W[weapon], spell=S[spell], armor=A[armor], boots=B[boots])


def room(equipment, arch=MELEE, seed=0, cfg=CFG, hp=100.0):
    p = bot_runner.BotPlayer(equipment=equipment, hp=hp)
    survived, delta = bot_runner._simulate_combat_room(p, arch, dict(cfg), random.Random(seed))
    return p, survived, delta


def many(equipment, arch=MELEE, n=200, cfg=CFG):
    tot = {"dealt": 0.0, "mh": 0, "gh": 0, "taken": 0.0, "time": 0.0, "kills": 0, "hits_in": 0}
    for s in range(n):
        p, _, d = room(equipment, arch, s, cfg)
        tot["dealt"] += d["damage_dealt"]
        tot["mh"] += d["melee_hits"]
        tot["gh"] += d["magic_hits"]
        tot["taken"] += d["damage_taken"]
        tot["time"] += d["time_in_room"]
        tot["kills"] += d["melee_kills"] + d["magic_kills"]
    return tot


def test_item_ids_are_stable_and_unique():
    assert len(core.ALL_ITEM_IDS) == len(set(core.ALL_ITEM_IDS)) == 12
    for slot, items in core.ITEMS_BY_SLOT.items():
        assert all(i.item_id.startswith(slot + ".") for i in items)


def test_weapon_damage_changes_damage_per_hit():
    hatchet = many(eq(weapon="weapon.hatchet"))
    rapier = many(eq(weapon="weapon.rapier"))
    assert hatchet["dealt"] / hatchet["mh"] == pytest.approx(W["weapon.hatchet"].dmg, rel=0.03)
    assert rapier["dealt"] / rapier["mh"] == pytest.approx(W["weapon.rapier"].dmg, rel=0.03)


def test_weapon_cooldown_scales_exchange_time():
    # Same rng stream -> same number of exchanges per room when outcomes coincide;
    # compare time per attack instead, which must equal BOT_EXCHANGE_SEC * cd_mult.
    for wid in ("weapon.rapier", "weapon.hatchet", "weapon.sword"):
        p, _, d = room(eq(weapon=wid), MELEE, seed=3)
        exchanges = round(d["time_in_room"] / (bot_runner.BOT_EXCHANGE_SEC * W[wid].cd_mult))
        assert d["time_in_room"] == pytest.approx(exchanges * bot_runner.BOT_EXCHANGE_SEC * W[wid].cd_mult)
    assert core.melee_cooldown(W["weapon.rapier"]) < core.melee_cooldown(W["weapon.hatchet"])


def test_spell_damage_and_ammo_limit():
    bolt = many(eq(spell="spell.magic_bolt"), MAGIC)
    assert bolt["dealt"] / (bolt["gh"] + bolt["mh"]) > 0      # sanity
    # Ammo is refilled per room and caps magic casts: hits <= ammo_max per room.
    for sid in ("spell.hex_bomb", "spell.magic_spike"):
        cfg = dict(CFG, enemy_hp_mult=3.0, spawn_count=10)
        for s in range(30):
            p, _, d = room(eq(spell=sid, armor="armor.chain_shirt"), MAGIC, s, cfg, hp=1e9)
            assert d["magic_hits"] <= S[sid].ammo_max
            if d["magic_hits"] + d["melee_hits"] > S[sid].ammo_max:
                assert d["melee_hits"] > 0     # out of ammo -> melee fallback, as in the game
    hex_ = many(eq(spell="spell.hex_bomb"), MAGIC, n=100)
    spike = many(eq(spell="spell.magic_spike"), MAGIC, n=100)
    assert hex_["dealt"] / max(hex_["gh"], 1) > spike["dealt"] / max(spike["gh"], 1)


def test_armor_absorb_reduces_damage_taken_and_matches_game_formula():
    assert core.incoming_damage(10.0, A["armor.chain_shirt"], False) == pytest.approx(8.0)
    assert core.incoming_damage(10.0, A["armor.cloth_wrap"], True) == pytest.approx(9.5 * core.DEFEND_DMG_MULT)
    assert core.incoming_damage(0.3, A["armor.chain_shirt"], False) == 0.0
    chain = many(eq(armor="armor.chain_shirt"))
    cloth = many(eq(armor="armor.cloth_wrap"))
    assert chain["taken"] < cloth["taken"]


def test_heal_on_kill_uses_weapon_and_armor():
    assert core.heal_on_kill_amount(W["weapon.sword"], A["armor.cloth_wrap"]) == 8
    assert core.heal_on_kill_amount(W["weapon.hatchet"], A["armor.blood_harness"]) == int(8 * 1.05) + 6
    p = bot_runner.BotPlayer(equipment=eq(armor="armor.blood_harness"), hp=50.0)
    p.heal_on_kill()
    assert p.hp == 50.0 + 8 + 6


def test_boots_are_inert_in_game_and_bot():
    """dash_dist_mult is not read by any game mechanic, so boots must not
    change bot combat either (no artificial bot-only meaning)."""
    import inspect
    import rogue
    assert "dash_dist_mult" not in inspect.getsource(rogue.Player)
    for s in range(20):
        a = room(eq(boots="boots.leather"), MELEE, s)[2]
        b = room(eq(boots="boots.voidstep_treads"), MELEE, s)[2]
        assert a == b


def test_game_player_uses_the_same_item_functions(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pytest.importorskip("pygame")
    import rogue
    pl = rogue.Player(0, 0, equipment=eq(weapon="weapon.hatchet", armor="armor.chain_shirt"))
    pl.defending = True
    pl.receive_damage(12.0)
    assert pl.hp == pytest.approx(100 - core.incoming_damage(12.0, A["armor.chain_shirt"], True))
    pl.hp = 50
    pl.heal_on_kill()
    assert pl.hp == 50 + core.heal_on_kill_amount(W["weapon.hatchet"], A["armor.chain_shirt"])
    pl.try_melee([])
    assert pl.melee_cd == pytest.approx(core.melee_cooldown(W["weapon.hatchet"]))
    replaced = rogue.apply_loot_choice(pl, ("spell", S["spell.hex_bomb"]))
    assert pl.spell.item_id == "spell.hex_bomb" and pl.magic_ammo == S["spell.hex_bomb"].ammo_max
    assert replaced == (eq().spell.item_id != "spell.hex_bomb")


def test_starting_loadout_is_seeded_and_shared_rule():
    import rogue
    for seed in (1, 2, 3):
        assert bot_runner.starting_equipment(seed).ids() == rogue.run_loadout(seed).ids()
    seen = {bot_runner.starting_equipment(s).weapon.item_id for s in range(200)}
    assert seen == {w.item_id for w in core.WEAPONS}


def test_equip_rejects_wrong_slot_and_reports_replacement():
    e = eq()
    assert e.equip("weapon", W["weapon.rapier"]) is True
    assert e.equip("weapon", W["weapon.rapier"]) is False
    with pytest.raises(ValueError):
        e.equip("armor", W["weapon.sword"])
