"""the night over frames and the live, dark night (docs/SIM_DESIGN.md 3.6, 3.7, 5.4, 7.3, 7.4 item 2, 8's R8 row, 9, 10; A46; the core
refactor's step R8; body/core/sleep.py). Run: python3 -m body.tests.test_night (the organ tests run these too).

What must hold: it is inert for language (night 1); each awake tick is taped as the cortex received it, beside its record, the offset's
end kept on it (night 2); at nightfall the day is cut into episodes at the frames' event ends, each with its entry and T_e by R7d's law
over the day's record, its window ending at its peak tag (or its end), each row's replayed dopamine's credit; the episodes are kept
across nights, fading as the store fades, the weakest giving way past the cap, the reels compacted to the rows the kept windows read
(night 3). Each is a switch the language body does not hold: it has none of it (the eight pinned digests are the guard's,
tools/pins/digests.txt). The stub worlds here are instruments of these tests, not the G1's world."""
import math
import os
import sys
import tempfile
import time

import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)   # this tree's body, not a fixed one
from body.life import Life  # noqa: E402
from body.core.amygdala import episode_entries, tag_star  # noqa: E402
from body.core.sleep import TAPE_BLOCK, credit_after, peak_of, tag_star_fast  # noqa: E402
from body.core.world import WorldLoop  # noqa: E402

TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
G = 0.9375


def _g1(cfg, world, d=32, seed=0, window=8):
    from body.sim.anatomy import SimAnatomy, born_table
    torch.manual_seed(seed)
    return Life.birth(SimAnatomy(born_table(), cfg), device="cpu", d=d, layers=1, heads=2, window=window, cfg=cfg, seed=seed, world=world)


def _cfg(**k):
    from body.sim.anatomy import SIM_CFG
    c = dict(SIM_CFG, wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.3, night_starts=16, night_starts_max=16, night_rounds=1,
             night_batch=8, rem_dreams=2, rem_steps=3, night_dev="", wake_ticks=10 ** 6)
    c.update(k)
    return c


def _events_world(seed=0):
    from body.tests.test_frames import _g1_events_world
    return _g1_events_world(seed=seed, burst=True)


# ---------------- night 1: inert for language ----------------

def test_night_inert_for_language():
    """night 1 (step R8): INERT FOR LANGUAGE. The language body's cfg holds no key of SLEEP; a language life born, living 80 ticks with a line
    typed and a night by hand holds no tape, reel, episode or twitch attribute, no window position holds a row of a tape, and its save
    holds no key of them; the switch without the frames it cuts is refused (the eight digests are the guard's)"""
    from body.core.physiology import SLEEP
    torch.manual_seed(0); L = Life.birth(TOK, device="cpu", d=32, layers=1, heads=2, window=32, cfg={}, seed=0)
    assert not any(k in L.cfg for k in SLEEP)
    L.type_text("the ball is red", who="parent")
    for _ in range(80):
        L.tick()
    rep = L.night()
    assert not rep.get("error"), rep
    for _ in range(10):
        L.tick()
    own = [k for k in vars(L) if k.startswith(("_tape", "_reel", "_episode", "_ep_", "_twitch", "_night_due"))]
    assert own == [] and not any("tape" in w for w in L.win) and "episodes" not in rep, (own, rep.keys())
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    try:
        L.save(path); blob = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        os.remove(path)
    assert "episodes" not in blob["life"] and not any(k in blob["cfg"] for k in SLEEP)
    try:
        torch.manual_seed(0); Life.birth(TOK, device="cpu", d=32, layers=1, heads=2, window=8, cfg={"night_frames": 1}, seed=0)
        raise AssertionError("the night over frames without the frames was not refused")
    except ValueError as e:
        assert "night_frames" in str(e) and "frames" in str(e)
    print("night 1: inert for language: no key of SLEEP in its cfg, no tape, reel, episode or twitch attribute after 90 ticks and a night,",
          "no window position with a row of a tape, no save key; night_frames without frames refused")


# ---------------- night 2: the day's tape ----------------

def test_the_tape():
    """night 2 (step R8a; SIM_DESIGN.md 9's tape, 8's R8 row): THE DAY'S TAPE. The G1 (SimAnatomy, SIM_CFG, d 32) lives 300 ticks in a stub
    of its world with bursts (so the frames' events end) and a word every 29 ticks (so the offset ends utterances): each tick's row is its
    window position as the cortex received it, every vector channel's 3,833 numbers at fp16 (the position's float32 rounded), the words'
    symbol, the voice's own symbol and every motor effector's act; the tape's rows are the record's rows (one a tick); the end flags are
    the offset's marks on the window, set when the offset fell (after the row was taped); the rows sit in blocks of TAPE_BLOCK. About 7.8
    KB a row at the G1's sizes"""
    from body.sim.anatomy import SIZES
    L = _g1(_cfg(), _events_world())
    seen = []; run = WorldLoop(L)
    for _ in range(300):
        run.step(); seen.append(L.win[-1])
    n = int(L._tape_n)
    assert n == 300 == int(L._rec_n) and len(L._tape) == math.ceil(300 / TAPE_BLOCK)
    sym = [c for c in L.anatomy.channels if c.kind == "symbol"]; vec = [c for c in L.anatomy.channels if c.kind == "vector"]
    V = sum(int(c.size) for c in vec)
    assert [c.name for c in sym] == ["words"] and V == sum(SIZES.values()) == 3833
    rows = L._tape_rows(n); ends = 0; words = 0
    for t, w in enumerate(seen):
        assert w["tape"] == t
        want = torch.cat([w[c.field].reshape(-1).float() for c in vec]).half()
        assert torch.equal(rows["v"][t], want), t
        assert int(rows["s"][t, 0]) == int(w["x"]) and int(rows["xo"][t]) == int(w["xo"])
        assert [int(x) for x in rows["a"][t]] == [int(w[e.field]) for e in L.anatomy.motors]
        assert bool(rows["end"][t]) == bool(w.get("end", False)), t
        ends += int(bool(w.get("end", False))); words += int(int(w["x"]) != L.sil)
    assert words >= 9 and ends >= 8, (words, ends)
    per = rows["v"].element_size() * V + 8 * (1 + 1 + len(L.anatomy.motors)) + 1 + 16
    print(f"night 2: the tape: {n} rows in {len(L._tape)} block(s) of {TAPE_BLOCK}, each its window position (the 3,833 numbers at fp16,",
          f"the word, the voice's symbol, the {len(L.anatomy.motors)} motor acts) and the record's row; the offset's {ends} end marks of {words}",
          f"words on it; {per:,} bytes a tick with the record (the design's 7.7 KB of fp16)")


# ---------------- night 3: the episodes ----------------

def test_the_episodes():
    """night 3 (step R8a; SIM_DESIGN.md 7.4 item 2, 9, 10's episode cap): THE EPISODES. At nightfall the G1's day of 300 ticks is cut at the
    frames' event ends (the night ending the last); over the day's record the tag reaches back (the fast tag* equals R7d's tag_star to the
    bit), each episode's entry and T_e are R7d's episode_entries (the bias-corrected running mean of 64 entries, moved in place and
    kept), its window of the cortex's length ends at the tick whose discounted tag is T_e (its value T_e to the bit) or at its last tick
    where T_e is under 0.1, reaching back before its start within the day, and each row's replayed dopamine's credit is the sum over the
    next 64 ticks at dopamine's discount (by hand); a window of one tick is not kept. The kept windows' rows equal the day's tape. After
    the night every entry has faded by the store's fade (0.9), and under a cap of 30 ticks the weakest give way (the lowest entries, of
    equals the oldest: the greedy rule by hand) until the kept windows read at most 30 distinct ticks, each reel compacted to them, each
    kept window's rows still the day's. A second day adds its episodes beside the first's (theirs faded twice)"""
    L = _g1(_cfg(), _events_world())
    run = WorldLoop(L)
    for _ in range(300):
        run.step()
    n = int(L._tape_n); tape = L._tape_rows(n)
    rec = L._rec[:n].clone().double(); ends = list(L._rec_ends); mean0 = list(getattr(L, "_ep_mean", None) or [0.0, 0])
    tags = rec[:, 2]
    ts_slow = tag_star(tags.tolist(), G, 64); ts_fast = tag_star_fast(tags, G, 64)
    assert ts_fast.tolist() == ts_slow
    G_ = credit_after(rec[:, 1], G, 64)
    for t in (0, 17, 150, n - 30, n - 2, n - 1):
        want = 0.0
        for k in range(1, 65):
            if t + k < n:
                want += G ** (k - 1) * float(rec[t + k, 1])
        assert abs(float(G_[t]) - want) < 1e-12, (t, float(G_[t]), want)
    L.cfg["episode_cap"] = 10 ** 9                                          # the first night keeps all (the cap is tried below)
    rep = L.night(); assert not rep.get("error"), rep.get("error")
    E = rep["episodes"]
    cuts = sorted({int(e) for e in ends if 0 < int(e) < n} | {n})
    spans, a = [], 0
    for e in cuts:
        if e > a:
            spans.append((a, e)); a = e
    ms = list(mean0); ent = episode_entries(spans, rec[:, 0].tolist(), rec[:, 1].tolist(), ts_slow, ms)
    assert E["spans"] == len(spans) >= 5 and [float(x) for x in L._ep_mean] == [float(x) for x in ms], (E, spans)
    W = int(L.m.window); kept = [(sp, en) for sp, en in zip(spans, ent)]
    eps = L._episodes; j = 0; at_peak = 0
    for (a_, b_), (entry, T_e) in kept:
        if T_e >= 0.1:
            p, pv = peak_of(tags, a_, b_, G, 64); assert pv == T_e, (pv, T_e); at_peak += 1
        else:
            p = b_ - 1
        w0 = max(0, p - W + 1)
        if p - w0 + 1 < 2:
            continue
        ep = eps[j]; j += 1
        assert (ep["a"], ep["b"], ep["peak"]) == (a_, b_, p) and abs(ep["entry"] - entry * 0.9) < 1e-15 and ep["T_e"] == T_e, (ep, entry, T_e)
        rows = L._episode_rows(ep)
        assert torch.equal(rows["v"], tape["v"][w0:p + 1]) and torch.equal(rows["a"], tape["a"][w0:p + 1]) and torch.equal(rows["G"], G_[w0:p + 1])
        assert torch.equal(rows["end"], tape["end"][w0:p + 1]) and torch.equal(rows["s"], tape["s"][w0:p + 1])
    assert j == len(eps) == E["kept"] and at_peak >= 1 and E["tagged"] >= 1, (j, len(eps), E)
    assert not hasattr(L, "_tape") or int(L._tape_n) == 0
    # the cap: a second day, then a cap of 30 ticks
    first = [dict(ep) for ep in eps]; old = {ep["serial"]: (ep["entry"], ep["w0"], ep["w1"]) for ep in eps}
    orig = {ep["serial"]: {k_: v_.clone() for k_, v_ in L._episode_rows(ep).items()} for ep in eps}
    for _ in range(200):
        run.step()
    n2 = int(L._tape_n); tape2 = L._tape_rows(n2)
    cap = 30; L.cfg["episode_cap"] = cap
    caught = {}; fade = L._episodes_fade

    def spy(rep_, f=fade, caught=caught, L=L):
        caught["eps"] = [dict(ep) for ep in L._episodes]; caught["rows"] = [int(r_["v"].shape[0]) for r_ in L._reels]
        return f(rep_)
    L._episodes_fade = spy
    rep2 = L.night(); assert not rep2.get("error"), rep2.get("error")
    del L._episodes_fade
    E2 = rep2["episodes"]
    assert E2["kept_before"] == len(first) and E2["rows"] <= cap, E2
    # the greedy rule by hand, on the entries after the fade (day 1's a second time), over the windows as they stood before the compaction
    pre = caught["eps"]
    assert all(abs(ep["entry"] - old[ep["serial"]][0]) < 1e-15 for ep in pre if ep["serial"] in old)
    refs = [[0] * r_ for r_ in caught["rows"]]
    for ep in pre:
        for r_ in range(ep["w0"], ep["w1"] + 1):
            refs[ep["reel"]][r_] += 1
    held = sum(1 for c_ in refs for x_ in c_ if x_ > 0); gone = set()
    for i in sorted(range(len(pre)), key=lambda i: (pre[i]["entry"] * 0.9, i)):
        if held <= cap:
            break
        ep = pre[i]; gone.add(pre[i]["serial"])
        for r_ in range(ep["w0"], ep["w1"] + 1):
            refs[ep["reel"]][r_] -= 1; held -= int(refs[ep["reel"]][r_] == 0)
    kept_serials = {ep["serial"] for ep in L._episodes}
    assert kept_serials == {ep["serial"] for ep in pre} - gone and held == E2["rows"], (sorted(kept_serials), sorted(gone), held, E2)
    assert all(ep["day"] in (0, 1) for ep in L._episodes) and len(L._episodes) == E2["kept"]
    for ep in L._episodes:
        if ep["serial"] in orig:
            now = L._episode_rows(ep)
            assert all(torch.equal(now[k_], orig[ep["serial"]][k_]) for k_ in now), ep["serial"]
        else:
            w0, p = ep["peak"] - (ep["w1"] - ep["w0"]), ep["peak"]
            assert torch.equal(L._episode_rows(ep)["v"], tape2["v"][w0:p + 1])
    olds = [ep for ep in L._episodes if ep["serial"] in old]
    assert all(abs(ep["entry"] - old[ep["serial"]][0] * 0.9) < 1e-15 for ep in olds)
    weakest_kept = min(ep["entry"] for ep in L._episodes)
    print(f"night 3: the episodes: day 1 of {n} ticks cut at {len(cuts) - 1} ends into {len(spans)} spans, {E['kept']} kept ({at_peak} at",
          f"their peak tag, {E['tagged']} tagged T_e >= 1); tag* fast equal to tag_star; entries, T_e, windows and the credit by hand; faded",
          f"0.9; day 2 of {n2} ticks: {E2['day']} more; under a cap of {cap} ticks {E2['gave_way']} gave way, {E2['kept']} kept on",
          f"{E2['rows']} rows in {E2['reels']} reel(s) (the weakest kept {weakest_kept:.3f}), each kept window's rows its day's; "
          f"{len(olds)} of day 1 kept at 0.81 of their entries")


NIGHT_TESTS = [test_night_inert_for_language, test_the_tape, test_the_episodes]


if __name__ == "__main__":
    t0 = time.time(); failed = 0
    for t in NIGHT_TESTS:
        try:
            t()
        except AssertionError as e:
            failed += 1; print("FAIL", t.__name__, ":", e)
        except Exception as e:
            failed += 1; print("ERROR", t.__name__, ":", type(e).__name__, str(e)[:300])
    print(f"{len(NIGHT_TESTS) - failed}/{len(NIGHT_TESTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if failed else 0)
