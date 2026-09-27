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
    symbol, the voice's own symbol, every motor effector's act and the ladder's bands the cortex received (fp16; the lead's item 1); the
    tape's rows are the record's rows (one a tick); the end flags are
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
    assert [c.name for c in sym] == ["words"] and V == sum(SIZES.values()) == 3831        # A88: no charge (3833 with it)
    rows = L._tape_rows(n); ends = 0; words = 0
    for t, w in enumerate(seen):
        assert w["tape"] == t
        want = torch.cat([w[c.field].reshape(-1).float() for c in vec]).half()
        assert torch.equal(rows["v"][t], want), t
        assert int(rows["s"][t, 0]) == int(w["x"]) and int(rows["xo"][t]) == int(w["xo"])
        assert [int(x) for x in rows["a"][t]] == [int(w[e.field]) for e in L.anatomy.motors]
        assert bool(rows["end"][t]) == bool(w.get("end", False)), t
        assert torch.equal(L._tape_bands(t), w["bundle"].half())         # the bands the cortex received there (the lead's item 1)
        ends += int(bool(w.get("end", False))); words += int(int(w["x"]) != L.sil)
    assert words >= 9 and ends >= 8, (words, ends)
    per = rows["v"].element_size() * V + 8 * (1 + 1 + len(L.anatomy.motors)) + 1 + 16
    bb = 2 * len(L.m.clocks) * int(L.m.d)
    print(f"night 2: the tape: {n} rows in {len(L._tape)} block(s) of {TAPE_BLOCK}, each its window position (the 3,833 numbers at fp16,",
          f"the word, the voice's symbol, the {len(L.anatomy.motors)} motor acts, the bands it received) and the record's row; the offset's",
          f"{ends} end marks of {words} words on it; {per:,} bytes a tick with the record (the design's 7.7 KB of fp16) and the bands' {bb:,}",
          f"here ({2 * 8 * 512:,} at d 512)")


# ---------------- night 3: the episodes ----------------

def test_the_episodes():
    """night 3 (step R8a; SIM_DESIGN.md 7.4 item 2, 9, 10's episode cap): THE EPISODES. At nightfall the G1's day of 300 ticks is cut at the
    frames' event ends (the night ending the last), each keeping the day's bands at its window's first tick; over the day's record the tag
    reaches back (the fast tag* equals R7d's tag_star to the
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
    n = int(L._tape_n); tape = L._tape_rows(n); bands = torch.stack([L._tape_bands(r_) for r_ in range(n)])
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
        assert torch.equal(ep["bands0"], bands[w0])                     # its dreams begin in the day's bands at the window's start
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


# ---------------- night 4: the night over frames ----------------

def test_the_night_over_frames():
    """night 4 (step R8b; SIM_DESIGN.md 7.4 item 2, 8's R8 row; the night's note since R6 fix 7): THE NIGHT OVER FRAMES, on the G1's day of
    300 ticks. THE DRAW: the day's episodes with T_e >= 1 first, highest first, at most half the night, the rest by entry; no words'
    dream (neither the store's nor its words-only copy). THE BATCH: each channel's observations from the reels (the fp16 numbers as
    float32), the words, the voice's symbol and every motor effector's act as the day received them, right-padded, and each window's
    bands from the day's at its first tick (the lead's decision, item 1 of the PFC study); the words' targets
    the waking lesson's (the next symbol, the offset's end) by hand. ACT_PRED'S NIGHT WEIGHT: a window whose credit is -2 at every
    position (weight clip(1 + G, 0, 1) = 0) gives act_pred, its correction and the maps exactly no gradient, while the forward half's is
    not zero; at credit +1 (weight 1) act_pred's is not zero; and every other parameter's gradient (the stream's, every head's) is the
    same to the bit at either weight: act_pred's weighted targets never reach the stream. THE TWO OPTIMIZERS: the night's Adam holds
    every trainable parameter but act_pred's, its correction's and the maps'; act_pred steps by its plain step once a batch, by hand
    (clamp(lr_motor x grad, +-bound)). ACT_INV REPLAYED over the windows' own acts, its reliability left as it was. REM ON FRAMES: from a
    window's first positions the cortex runs free (the words drawn, every vector channel's code its head's forecast) and only the
    prefrontal heads take a gradient. The night reports its dreams, steps, weights and gauge"""
    import body.core.sleep as S
    L = _g1(_cfg(night_starts=12, night_starts_max=12, night_rounds=2, night_batch=4), _events_world())
    run = WorldLoop(L)
    for _ in range(300):
        run.step()
    # what the night draws, and that no words' dream is drawn
    calls = {"dreams": 0, "ws": 0}; dr = L.dreams; wsf = L._words_store
    L.dreams = lambda *a, **k: (calls.__setitem__("dreams", calls["dreams"] + 1), dr(*a, **k))[1]
    L._words_store = lambda *a, **k: (calls.__setitem__("ws", calls["ws"] + 1), wsf(*a, **k))[1]
    seen = {}; nd = S.night_draw

    def spy_draw(entries, tags, n, gen, nd=nd, seen=seen):
        out = nd(entries, tags, n, gen); seen.update(entries=list(entries), tags=list(tags), n=n, out=list(out)); return out
    S.night_draw = spy_draw
    # the optimizers: the night's Adam's parameters, act_pred's plain step by hand
    adam_ids = []; Adam = torch.optim.Adam

    class SpyAdam(Adam):
        def __init__(self, params, *a, **k):
            params = list(params); adam_ids.append({id(p_) for p_ in params}); super().__init__(params, *a, **k)
    S.torch.optim.Adam = SpyAdam
    op = L.opt_pred; ost = op.step; steps = []

    def spy_step(ost=ost, op=op, steps=steps):
        g0 = op.param_groups[0]; p0 = g0["params"][0]
        before = p0.detach().clone(); gr = p0.grad.detach().clone() if p0.grad is not None else None
        out = ost(); steps.append((before, gr, p0.detach().clone(), float(g0["lr"]) * float(g0["rate"]), float(g0["lr"]) * float(g0["bound"])))
        return out
    op.step = spy_step
    pend = sum(len(st_.get("inv_batch") or []) for st_ in L.motor)
    for i_, st_ in enumerate(L.motor, 1):                                 # the day's pending pairs (learned at nightfall, before the night's
        if st_.get("inv_batch"):                                          # lessons: R8c) learned here first, so the replay's own is seen
            L._inverse_batch(i_)
    rel0 = [(dict((k_, st_[k_]) for k_ in ("inv_kappa", "inv_gain", "inv_n")), [[list(r) for r in c] for c in st_["inv_conf"]] if st_["inv_conf"] else None,
             [list(r) for r in st_["inv_ch"]] if st_.get("inv_ch") else None) for st_ in L.motor]
    inv0 = {e.name: [p_.detach().clone() for p_ in L.m.timing[e.name].inv.parameters()] for e in L.anatomy.motors if e.inverse}
    nf = L._night_frames; rel_after = []

    def spy_nf(rep_, nf=nf, L=L, rel_after=rel_after):
        out = nf(rep_)
        rel_after.extend((dict((k_, st_[k_]) for k_ in ("inv_kappa", "inv_gain", "inv_n")), [[list(r) for r in c] for c in st_["inv_conf"]] if st_["inv_conf"] else None,
                          [list(r) for r in st_["inv_ch"]] if st_.get("inv_ch") else None) for st_ in L.motor)
        rel_after.append({e.name: [p_.detach().clone() for p_ in L.m.timing[e.name].inv.parameters()] for e in L.anatomy.motors if e.inverse})
        return out
    L._night_frames = spy_nf
    try:
        rep = L.night()
    finally:
        S.night_draw = nd; S.torch.optim.Adam = Adam; del L.opt_pred.step; del L._night_frames; del L.dreams; del L._words_store
    assert not rep.get("error"), rep.get("error")
    assert calls == {"dreams": 0, "ws": 0}, calls
    tags, out, n = seen["tags"], seen["out"], seen["n"]
    firsts = sorted([i for i, t_ in enumerate(tags) if t_ >= 1.0], key=lambda i: (-tags[i], i))[:n // 2]
    assert out[:len(firsts)] == firsts and len(out) == n == rep["dreams"] == 12 and rep["tagged_first"] == len(firsts) >= 1, (out, firsts, rep)
    gated = {id(p_) for e_ in L.anatomy.motors for p_ in L._gated_params(e_)}
    trainable = {id(p_) for p_ in L.m.parameters() if p_.requires_grad}
    assert len(adam_ids) == 1 and not (adam_ids[0] & gated) and adam_ids[0] | gated >= trainable, len(adam_ids)
    assert len(steps) == rep["nrem_steps"] == 2 * 3, (len(steps), rep["nrem_steps"])
    for before, gr, after, s_, b_ in steps:
        assert gr is not None and torch.equal(after, before - (gr * s_).clamp(-b_, b_))
    # act_inv replayed: its weights moved, its reliability as it was
    assert pend > 0 and all(r_ == a_ for r_, a_ in zip(rel0, rel_after[:len(rel0)])), "the replay moved act_inv's reliability"
    moved = [n_ for n_, ps_ in inv0.items() if any(not torch.equal(p_, q_) for p_, q_ in zip(ps_, rel_after[-1][n_]))]
    assert moved and rep["act_inv_pairs"] > 0, (moved, rep["act_inv_pairs"])
    # the batch, the words' targets, act_pred's weight
    eps = L._episodes; batch = eps[:3]
    obs, xos, bundles, ends, G_, mask, lens = L._frames_batch(batch)
    off = 0
    for c in [c for c in L.anatomy.channels if c.kind == "vector"]:
        for b, ep in enumerate(batch):
            assert torch.equal(obs[c.name][b, :lens[b]], L._episode_rows(ep)["v"][:, off:off + c.size].float())
            assert float(obs[c.name][b, lens[b]:].abs().sum()) == 0.0
        off += c.size
    for b, ep in enumerate(batch):
        r_ = L._episode_rows(ep)
        assert torch.equal(bundles[b, 0], ep["bands0"].float())           # each window from the day's bands at its start (the lead's item 1)
        assert torch.equal(obs["words"][b, :lens[b]], r_["s"][:, 0]) and torch.equal(xos[b, :lens[b]], r_["xo"])
        assert all(torch.equal(obs[e.name][b, :lens[b]], r_["a"][:, j]) for j, e in enumerate(L.anatomy.motors))
    y, w = L._word_targets(obs["words"], ends, lens)
    for b in range(len(batch)):
        xs = obs["words"][b].tolist(); en = ends[b].tolist()
        for t in range(lens[b]):
            nx = next((u for u in range(t + 1, lens[b]) if xs[u] != L.sil), None)
            want = (L.end_id, 1.0) if en[t] else ((xs[nx], 1.0) if nx is not None else (None, 0.0))
            assert float(w[b, t]) == want[1] and (want[0] is None or int(y[b, t]) == want[0]), (b, t)
    ep = max(eps, key=lambda e_: e_["w1"] - e_["w0"])
    grads = {}
    for tag_, gval in (("harm", -2.0), ("full", 1.0)):
        rows = L._reels[ep["reel"]]; saved = rows["G"].clone(); rows["G"][ep["w0"]:ep["w1"] + 1] = gval
        try:
            L.m.zero_grad(set_to_none=True)
            loss, parts = L._frames_lesson([ep]); loss.backward()
        finally:
            rows["G"].copy_(saved)
        grads[tag_] = {n_: (p_.grad.clone() if p_.grad is not None else None) for n_, p_ in L.m.named_parameters()}
    L.m.zero_grad(set_to_none=True)
    gnames = {n_ for n_, p_ in L.m.named_parameters() if id(p_) in gated}
    for n_ in gnames:
        g_ = grads["harm"][n_]
        assert g_ is None or float(g_.abs().max()) == 0.0, n_
    assert any(grads["full"][n_] is not None and float(grads["full"][n_].abs().max()) > 0.0 for n_ in gnames if ".pred." in n_)
    fwd = [n_ for n_ in grads["harm"] if ".fwd." in n_]
    assert fwd and all(grads["harm"][n_] is not None and float(grads["harm"][n_].abs().max()) > 0.0 for n_ in fwd if n_.endswith("weight"))
    other = [n_ for n_ in grads["harm"] if n_ not in gnames]
    same = [n_ for n_ in other if (grads["harm"][n_] is None) == (grads["full"][n_] is None) and (grads["harm"][n_] is None or torch.equal(grads["harm"][n_], grads["full"][n_]))]
    assert same == other, [n_ for n_ in other if n_ not in same][:5]
    # REM on frames: only the prefrontal heads learn
    L.m.zero_grad(set_to_none=True)
    fl, fc = L._rem_frames(eps[0]); fl.backward()
    with_grad = sorted({n_.split(".")[0] for n_, p_ in L.m.named_parameters() if p_.grad is not None and float(p_.grad.abs().sum()) > 0})
    L.m.zero_grad(set_to_none=True)
    assert with_grad == ["pfc_pred"], with_grad
    assert rep["rem_steps"] == int(L.cfg["rem_rounds"]) and rep["gauge"]["before"] and rep["act_pred_weight"]["positions"] > 0
    print(f"night 4: the night over frames: {rep['dreams']} dreams ({rep['tagged_first']} tagged first, highest first; no words' dream),",
          f"{rep['nrem_steps']} NREM steps in batches of 4, act_pred by its plain step by hand at each, the night's Adam over every other",
          f"parameter; act_inv replayed on {rep['act_inv_pairs']} own acts, its reliability as it was; a window of net harm gives act_pred no",
          f"gradient and the forward half its own, the stream's gradients the same to the bit at weight 0 and 1 ({len(other)} parameters);",
          f"act_pred's weights below 1 on {rep['act_pred_weight']['below_1']:.3f} of the positions, 0 on {rep['act_pred_weight']['zero']:.3f};",
          f"REM on frames {rep['rem_steps']} steps (only the prefrontal heads learn); the gauge's words {rep['gauge']['before']['words']} ->",
          f"{rep['gauge']['after']['words']}, its channels {rep['gauge']['before']['channels']} -> {rep['gauge']['after']['channels']}")


# ---------------- night 5: the live, dark night ----------------

def _live_world(seed=0):
    """a stub of the G1's world that runs through the night (never the sim's): motor 12's random senses, loud 6 ticks in every 40 (so the
    frames' events end), a word every 17 ticks, a smile every 40, pain every 97, the torso's unit turning, the cerebellum's hook called 15
    times a tick as SimWorld's contract says; at dusk its frames carry the body's own senses alone (the body, touch, the inertial units,
    the charge, the pain flags, the torso's unit), the eyes and the ears off and no word; it records every call (each apply by day or by
    night with its acts, dusk, dawn, pause, resume) and its whole state goes with its save"""
    import pickle
    import random as _r
    from body.core.world import Frame, SimWorld, SubFrame
    from body.sim.anatomy import SIZES
    from body.tests.test_motor import _g1_mossy

    class LiveG1(SimWorld):
        live_night = True

        def __init__(self):
            self.t = 0; self.rng = _r.Random(seed); self.rng_cb = _r.Random(seed + 1); self.dark = False; self.log = []

        def frame(self):
            t = self.t; R = self.rng; a_ = 3.0 if t % 40 < 6 else 0.3
            obs = {n: [a_ * R.uniform(-1, 1) for _ in range(k)] for n, k in SIZES.items() if n != "face"}
            obs["face"] = [2.0 if t % 40 == 20 else 0.0, 0.0]
            obs["body"][241] = R.uniform(0.2, 1.0)
            obs["pain"] = [1.0 if (t % 97 == 50 and k == 5) else 0.0 for k in range(44)]
            if t % 17 == 0:
                obs["words"] = R.randrange(3, 79)
            obs["imu_torso"] = [0.3 * math.sin(t / 11.0), 0.2 * math.cos(t / 13.0), 9.81, 0.05 * math.cos(t / 5.0), 0.04 * math.sin(t / 9.0),
                                0.1 * math.sin(t / 7.0) + 0.02]
            if self.dark:
                for k in ("ears", "eye_p", "eye_f", "face", "words"):
                    obs.pop(k, None)
            return Frame(t, obs, 0.0, {"who": "parent", "yaw": 0.01 * t})

        def apply(self, acts):
            self.log.append(("night" if self.dark else "day", self.t, dict(acts)))
            if self.below is not None:
                life = self.below._life(); J = len(self.below.joints); R = self.rng_cb
                for s_ in range(15):
                    self.sub_tick(SubFrame(self.t, s_, _g1_mossy(life.anatomy, acts, R), [R.uniform(-2.0, 2.0) for _ in range(J)],
                                           None if self.dark else ([R.uniform(-0.01, 0.01) for _ in range(2)] if s_ == 0 else None),
                                           None if self.dark else ([R.uniform(-0.1, 0.1) for _ in range(2)] if s_ == 0 else None),
                                           limit=[25.0] * J))
            self.t += 1

        def dusk(self):
            self.dark = True; self.log.append(("dusk", self.t, None))

        def dawn(self):
            self.dark = False; self.log.append(("dawn", self.t, None))

        def pause(self):
            self.log.append(("pause", self.t, None))

        def resume(self):
            self.log.append(("resume", self.t, None))

        def save_state(self):
            return pickle.dumps((self.t, self.rng.getstate(), self.rng_cb.getstate(), self.dark))

        def load_state(self, blob):
            self.t, a_, b_, self.dark = pickle.loads(blob); self.rng.setstate(a_); self.rng_cb.setstate(b_)
    return LiveG1()


def test_value_sweep():
    """night 8 (A93): the reverse value sweep. Two G1s (d 32) born alike live the same 150 ticks in the stub world (its smiles at ticks
    20, 60, 100 and 140), one sleeping with night_reverse on, one off. After the night the critic's value of the states in the 8 ticks
    before each smile (band 3, the 64-tick clock) rose more with the sweep than without it, and the night's report counted its chunks
    and ticks; the sweep runs only under night_frames (the language body has none)"""
    vals = {}
    for rev in (0, 1):
        cfg = _cfg(wake_ticks=150, night_ticks=600, sleep_cycle=400.0, twitch_rate=0.1, night_starts=8, night_starts_max=8, night_rounds=1,
                   night_batch=4, night_reverse=rev)
        L = _g1(cfg, _live_world()); run = WorldLoop(L)
        states, rewards = [], []
        for t in range(150):
            states.append(L.bands.detach().clone()); run.step()
            rewards.append(float(L._rec[L._rec_n - 1, 3]) if t < 149 else 0.0)
        assert L.nights == 1 and not L.last_night.get("error"), L.last_night.get("error")
        smiles = [t for t, r in enumerate(rewards) if r >= 1.5]
        assert smiles, rewards[:60]
        pre = [t_ for s in smiles for t_ in range(max(0, s - 8), s)]
        with torch.no_grad():
            v_after = float(torch.stack([L.m.value_of(3, states[t_][3]) for t_ in pre]).mean())
        vals[rev] = dict(v=v_after, sweep=L.last_night.get("sweep"), smiles=smiles)
    assert vals[0]["sweep"] is None and vals[1]["sweep"] and vals[1]["sweep"]["chunks"] >= 3 and vals[1]["sweep"]["ticks"] >= 140, vals
    assert vals[1]["v"] > vals[0]["v"], vals
    print(f"night 8: the reverse value sweep (A93): the smiles at {vals[1]['smiles']}; the value of the states 8 ticks before them on band",
          f"3 after the night {vals[0]['v']:.4f} without the sweep, {vals[1]['v']:.4f} with it (its report {vals[1]['sweep']})")


def test_the_live_dark_night():
    """night 5 (step R8c; SIM_DESIGN.md 3.6, 3.7, 5.4, A46, C74): THE LIVE, DARK NIGHT. The G1 (d 32) lives 150 ticks in a stub of its world
    that runs through the night (a sleep cycle shortened to 400 ticks and a twitch rate of 0.1 a tick of active sleep, the test's, so the
    phases and the rate are seen in a night of 600 ticks). THE NIGHT AT THE TICK'S END (C74): the sleep switch fires in tick 149 and the
    night runs after the world has applied that tick's acts and run its cerebellum's sub-steps; dusk, 600 night ticks, dawn; no pause.
    EVERY NIGHT ACT A REST BUT THE TWITCHES: each twitch one joint's small step (the setting beside its hold) on the waist, an arm, a hand
    or a leg, never the tract or the gaze; only in active sleep (the first half of each cycle); their count over the active ticks within
    3 sigma of the rate; the born generator's schedule (a function of the body's seed and the night); logged as reflex (the cord's counts);
    the night's frames with the eyes and the ears off. THE CEREBELLUM called 15 times a night tick (it learns wherever the world runs).
    THE PAIRS TEACH act_inv (its reliability updated on each, its weights moved) and the twitching limb's forward half, the stream held.
    THE NIGHT'S OWN SAVE is written after dawn, between ticks: a life loaded from it goes on 75 ticks as the life that went on, bit for
    bit. At the rate as read (0.025) and the 47-minute cycle a night of 24,000 ticks has about 365 twitches (the law's expectation)"""
    from body.tests.test_anatomy import _whole
    cfg = _cfg(wake_ticks=150, night_ticks=600, sleep_cycle=400.0, twitch_rate=0.1, night_starts=8, night_starts_max=8, night_rounds=1,
               night_batch=4)
    L = _g1(cfg, _live_world()); w = L.world; run = WorldLoop(L)
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd); L.save_path = path
    sv = L.save
    L.save = lambda p_=None, sv=sv, w=w: (w.log.append(("save", w.t, None)), sv(p_))[1]
    cortex = [n_ for n_, _ in L.m.named_parameters() if n_.split(".")[0] in ("blocks", "lnf", "in_ln", "bundle_in", "latent_pred", "chan_pred")]
    tl = L._twitch_lessons; seen = {}

    def spy_tl(pairs, opt, params, tl=tl, L=L, seen=seen):
        sd0 = {n_: p_.detach().clone() for n_, p_ in L.m.named_parameters()}
        inv0 = [int(st_["inv_n"]) for st_ in L.motor]
        out = tl(pairs, opt, params)
        seen.update(pairs=list(pairs), sd0=sd0, sd1={n_: p_.detach().clone() for n_, p_ in L.m.named_parameters()}, inv0=inv0,
                    inv1=[int(st_["inv_n"]) for st_ in L.motor])
        return out
    L._twitch_lessons = spy_tl
    cb0 = int(L.m.cereb.n_sub)
    try:
        for _ in range(150):
            run.step()
        assert L.nights == 1 and not L.last_night.get("error"), L.last_night.get("error")
        wB = w.save_state()
        Lc = None
        g_ = torch.get_rng_state(); torch.manual_seed(0)
        from body.sim.anatomy import SimAnatomy, born_table
        w2 = _live_world(); w2.load_state(wB)
        Lc = Life.load(path, SimAnatomy(born_table(), L.cfg), save_path=None, world=w2); torch.set_rng_state(g_)
    finally:
        os.remove(path)
    del L.save; del L._twitch_lessons; L.save_path = None
    log = w.log; kinds = [k_ for k_, _, _ in log]
    i_d = kinds.index("dusk"); i_w = kinds.index("dawn")
    assert log[i_d - 1][:2] == ("day", 149) and kinds[i_d + 1:i_w] == ["night"] * 600 and kinds[i_w + 1] == "save", kinds[i_d - 2:i_d + 3]
    assert "pause" not in kinds and "resume" not in kinds and log[i_w][1] == 750
    names = {e.name: e for e in L.anatomy.effectors}; rest = {n_: int(e.rest_id) for n_, e in names.items()}
    tw = []
    for k, (_, t, acts) in enumerate(log[i_d + 1:i_w]):
        moved = [(n_, a) for n_, a in acts.items() if a != rest[n_]]
        assert len(moved) <= 1, (k, moved)
        if moved:
            n_, a = moved[0]; e = names[n_]; tab = L.m.acts[n_]
            dig = [int(x) for x in tab.digits(torch.tensor(a)).tolist()]
            off = [(j, x - 2) for j, x in enumerate(dig) if x != 2]
            assert e.twitch and len(off) == 1 and off[0][1] in (-1, 1), (n_, dig)
            tw.append((k, n_, off[0][0], off[0][1]))
    assert all(names[n_].name not in ("voice", "gaze", "words") for _, n_, _, _ in tw)
    act_ticks = [k for k in range(600) if (k % 400) < 200]
    assert all((k % 400) < 200 for k, _, _, _ in tw) and tw, tw[:3]
    n_act = len(act_ticks); mu = 0.1 * n_act; sd = math.sqrt(n_act * 0.1 * 0.9)
    assert abs(len(tw) - mu) <= 3 * sd, (len(tw), mu, sd)
    L.nights -= 1
    try:
        sched = L._twitch_schedule(600)
    finally:
        L.nights += 1
    mot = list(L.anatomy.motors)
    assert [(k, mot[i - 1].name, j, 1 if sg else -1) for k, ((i, j), sg) in sorted(sched.items())] == tw
    assert sum(int(st_["cord_n"].get("twitch", 0)) for st_ in L.motor) == len(tw) == L.last_night["live"]["twitches"]
    assert L.last_night["live"]["cereb_sub"] == 15 * 600 and int(L.m.cereb.n_sub) - cb0 >= 15 * 750
    # the pairs' lessons
    pairs = seen["pairs"]; assert len(pairs) == len(tw)
    d_inv = [b_ - a_ for a_, b_ in zip(seen["inv0"], seen["inv1"])]
    want = [sum(1 for p_ in pairs if p_[0] == i) if e.inverse else 0 for i, e in enumerate(mot, 1)]
    assert d_inv == want, (d_inv, want)
    assert all(torch.equal(seen["sd0"][n_], seen["sd1"][n_]) for n_ in cortex), "the twitch lessons moved the stream"
    fwd_moved = sorted({n_.split(".")[1] for n_ in seen["sd0"] if ".fwd." in n_ and not torch.equal(seen["sd0"][n_], seen["sd1"][n_])})
    assert fwd_moved == sorted({mot[p_[0] - 1].name for p_ in pairs}), fwd_moved
    inv_moved = sorted({n_.split(".")[1] for n_ in seen["sd0"] if ".inv." in n_ and not torch.equal(seen["sd0"][n_], seen["sd1"][n_])})
    assert inv_moved == fwd_moved, (inv_moved, fwd_moved)
    # the night's own save: loaded, the life goes on as the life that went on
    for _ in range(75):
        run.step()
    run2 = WorldLoop(Lc)
    for _ in range(75):
        run2.step()
    hA, hC = _whole(L), _whole(Lc)
    assert hA == hC, [k_ for k_ in hA if hA[k_] != hC[k_]]
    live = L.last_night["live"]
    print(f"night 5: the live, dark night at the tick's end (tick 149's acts applied, then dusk, 600 night ticks, dawn, the save; no pause):",
          f"{len(tw)} twitches on {n_act} active ticks (the rate 0.1: {mu:.0f} +- {sd:.1f}), none in quiet sleep, each one joint's small step",
          f"on {sorted({n_ for _, n_, _, _ in tw})}, the born generator's schedule, logged as reflex; the cerebellum {live['cereb_sub']}",
          f"sub-steps in the night; act_inv learned {sum(d_inv)} pairs (its reliability on each) and the forward halves of {fwd_moved},",
          f"the stream held; the night's events {live['events']}; the night's own save, loaded, went on 75 ticks as the life that went on",
          f"({' '.join(f'{k_} {v_[:10]}' for k_, v_ in hA.items())}); at the design's law a night of 24,000 ticks has about",
          f"{0.025 * (9400 + 5200):.0f} twitches")


# ---------------- night 6: C51's instrument ----------------

def test_the_heading_drift():
    """night 6 (step R8d; SIM_DESIGN.md 7.6, A45, C51): THE HEADING'S DRIFT AGAINST THE WORLD'S TRUE YAW, an instrument: the G1's heading
    (R7f's path integration of the torso gyro about the accelerometer's up) against the true yaw the world keeps in its frame's truth
    (the stub's: 0.01 rad a tick), the drift and its wrap by hand; reported in `insides` and at dusk in the night's report; never
    corrected and never read by the body: two lives, one whose frames carry the true yaw and one whose do not, are the same life in every
    section of the whole state over 120 ticks. The language body and a body without recall report none"""
    from body.tests.test_anatomy import _whole
    cfg = _cfg(wake_ticks=150, night_ticks=40, night_starts=8, night_starts_max=8, night_rounds=1, night_batch=4)
    L = _g1(cfg, _live_world()); run = WorldLoop(L)
    for _ in range(120):
        run.step()
    hd = L.heading_drift(); y = float(L.world.now.truth["yaw"]); h = float(L._heading)
    assert hd == {"heading": h, "true_yaw": y, "drift": h - y, "drift_wrapped": math.atan2(math.sin(h - y), math.cos(h - y))}, hd
    assert L.insides()["heading"] == hd and y == 0.01 * 119
    for _ in range(30):
        run.step()
    rep = L.last_night; assert L.nights == 1 and "heading" in rep, rep.keys()
    assert abs(rep["heading"]["drift"] - (rep["heading"]["heading"] - rep["heading"]["true_yaw"])) < 2e-6
    # never read by the body: the same life without the truth
    class NoTruth(type(_live_world())):
        def frame(self):
            f = super().frame(); f.truth.pop("yaw", None); return f
    A = _g1(dict(cfg, wake_ticks=10 ** 6), _live_world()); B = _g1(dict(cfg, wake_ticks=10 ** 6), NoTruth())
    ra, rb = WorldLoop(A), WorldLoop(B)
    for _ in range(120):
        ra.step(); rb.step()
    assert _whole(A) == _whole(B) and B.heading_drift() is None and A.heading_drift() is not None
    torch.manual_seed(0); Lg = Life.birth(TOK, device="cpu", d=32, layers=1, heads=2, window=8, cfg={}, seed=0)
    Lg.tick(); assert Lg.heading_drift() is None and "heading" not in Lg.insides()
    print(f"night 6: C51's instrument: after 120 ticks the heading {h:+.4f} rad against the world's true yaw {y:+.4f}, the drift",
          f"{h - y:+.4f} (wrapped {hd['drift_wrapped']:+.4f}), in insides and at dusk in the night's report ({rep['heading']['drift']:+.4f});",
          f"the same life with and without the truth in the frames; none for the language body")


# ---------------- night 7: A71, the born config at the served values ----------------

def test_the_born_config():
    """night 7 (A71, the lead's decision of 2026-09-25 on the PFC-maturation study): THE SIM IS BORN AT THE SERVED VALUES. SIM_CFG's slow
    bands kept across the night, striatal fast critic, working memory's slot and long critic with its earned voice each at the served
    language body's value (tools/pins/served_cfg.pkl), but the three the design sets for the sim (the critics' solves every 256 ticks,
    SIM_DESIGN.md 10; the long critic on the fast bands 0-2, 7.2); amyg_pav born on with its earned weight. A G1 born under it has the
    striatal expansion (the language block, every motor effector's per-joint rows and R7a's event lines' block), the working-memory slot
    beside it (the fast critic's evidence 2 x 2,048 wide and its level), the long critic's evidence over bands 0-2 with the tonic traces and the clock
    (3 d + 8 + 1, and its level), its voice 0 at birth; working memory latches at the frames' event ends; and after a night its bands are
    the evening's, not zeroed"""
    import pickle
    from body.sim.anatomy import SIM_CFG
    with open(os.path.join(ROOT, "tools", "pins", "served_cfg.pkl"), "rb") as f:
        served = pickle.load(f)
    taken = ("night_keep_bands", "fast_input", "fast_rls", "stri_k", "stri_m", "stri_quiet", "fast_rls_prior", "wm", "wm_burst", "wm_max",
             "vcrit_rls", "vcrit_auto", "vcrit_ceiling", "vcrit_forget", "vcrit_traces", "vcrit_clock", "vcrit_norm_tau", "vcrit_rls_prior",
             "vcrit_lambda", "vcrit_center")
    assert all(SIM_CFG[k] == served[k] for k in taken), [(k, SIM_CFG[k], served[k]) for k in taken if SIM_CFG[k] != served[k]]
    assert (SIM_CFG["fast_rls_every"], SIM_CFG["vcrit_rls_every"], SIM_CFG["vcrit_bands"]) == (256, 256, "0,1,2")
    assert (served["fast_rls_every"], served["vcrit_rls_every"], served["vcrit_bands"]) == (64, 64, "-")
    L = _g1(_cfg(wake_ticks=120, night_ticks=20, night_starts=4, night_starts_max=4, night_rounds=1, night_batch=4), _events_world())
    m = L.m; d = int(m.d); k = int(L.cfg["stri_k"])
    rows = k * (2 * m.vocab + 3) + sum(k * sum(int(f_) for f_ in e.factors) for e in L.anatomy.motors) + k * 13
    assert tuple(m.stri_W.shape) == (rows, 2048) and m.stri_wm == 1 and m.vf_A.shape == (2 * 2048 + 1, 2 * 2048 + 1)
    assert L._vc_idx.numel() == 3 * d + 8 + 1 and m.vc_A.shape == (3 * d + 8 + 1 + 1, 3 * d + 8 + 1 + 1)
    assert L._vrel_gain == 0.0 and L.cfg["amyg_pav"] == 1
    latched = []; wl = m.wm_latch
    m.wm_latch = lambda z, wl=wl, L=L: (latched.append(L.ticks), wl(z))[1]
    run = WorldLoop(L)
    for _ in range(119):
        run.step()
    del m.wm_latch
    ends = list(L._rec_ends)
    seen = {}; nf = L.night
    L.night = lambda nf=nf, L=L, seen=seen: (seen.__setitem__("evening", L.bands.clone()), nf())[1]
    run.step()                                                           # the 120th tick: its night at the tick's end
    del L.night
    assert L.nights == 1 and not L.last_night.get("error")
    evening = seen["evening"]
    assert torch.equal(L.bands, evening) and float(L.bands.abs().sum()) > 0.0   # the evening's bands, kept (night_keep_bands 1)
    assert ends and set(ends) <= set(latched), (ends[:5], latched[:5])
    print(f"night 7: A71's born config: {len(taken)} values the served body's, the solves every 256 ticks and the long critic on bands 0-2",
          f"the sim's; born: the striatal expansion {rows} x 2048 (the event lines' block in it), the slot beside it (the fast critic's",
          f"evidence {2 * 2048 + 1} wide with its level), the long critic's {3 * d + 10}, its voice 0; working memory latched at all {len(ends)} frame event ends",
          f"(and {len(latched) - len(ends)} bursts); after the night the evening's bands kept")



def test_the_night_rests_the_slow_states():
    """night 9 (A106, 2026-09-27): the live night's ticks pass for the body's slow states as the day's do: stress, mood and fatigue (the
    body's and each motor effector's) fall over the night by their half-lives, in closed form; a body that slept at its stress ceiling
    wakes rested; the report says the states before and after"""
    cfg = _cfg(wake_ticks=150, night_ticks=600, sleep_cycle=400.0, twitch_rate=0.1, night_starts=8, night_starts_max=8, night_rounds=1)
    L = _g1(cfg, _live_world()); run = WorldLoop(L)
    for _ in range(148):
        run.step()
    L.stress, L.mood, L.fatigue = 20.0, -4.0, 5.0
    for st_ in L.motor:
        st_["fatigue"] = 3.0
    hl = {k: float(L.cfg[k]) for k in ("stress_half_life", "mood_half_life", "fatigue_half_life")}
    for _ in range(4):
        run.step()
        if L.nights >= 1:
            break
    assert L.nights == 1 and not L.last_night.get("error"), L.last_night.get("error")
    N = 600
    slow = (L.last_night.get("live") or {}).get("slow_states")
    assert slow is not None and slow["before"]["stress"] > 15.0, slow
    want = dict(stress=slow["before"]["stress"] * 0.5 ** (N / hl["stress_half_life"]), mood=slow["before"]["mood"] * 0.5 ** (N / hl["mood_half_life"]),
                fatigue=slow["before"]["fatigue"] * 0.5 ** (N / hl["fatigue_half_life"]))
    for k in want:
        assert abs(slow["after"][k] - want[k]) < 1e-3 + abs(want[k]) * 1e-3, (k, slow["after"][k], want[k])
    assert all(float(st_["fatigue"]) < 3.0 * 0.5 ** (N / hl["fatigue_half_life"]) + 1.0 for st_ in L.motor)
    print(f"night 9: over a night of {N} ticks the report's slow states: stress {slow['before']['stress']} -> {slow['after']['stress']}",
          f"(the law {want['stress']:.3f}), mood {slow['before']['mood']} -> {slow['after']['mood']} ({want['mood']:.3f}), fatigue",
          f"{slow['before']['fatigue']} -> {slow['after']['fatigue']} ({want['fatigue']:.3f}); each motor effector's fatigue likewise (A106)")

NIGHT_TESTS = [test_night_inert_for_language, test_the_tape, test_the_episodes, test_the_night_over_frames, test_the_live_dark_night,
               test_the_heading_drift, test_the_born_config, test_value_sweep, test_the_night_rests_the_slow_states]


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
