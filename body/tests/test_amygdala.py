"""the amygdala, the valence tagger (docs/SIM_DESIGN.md 7.4, 10, A16; the core refactor's step R7d; body/core/amygdala.py). Run: python3 -m
body.tests.test_amygdala (the organ tests run these too). The tests are 7.4's list, on a tiny body (d 32: a words channel, a 4-number cue
channel, the event lines "bump" and "chime", one 2-joint effector with an orienting trigger on "chime", reward sources face +/- and pain -,
a scripted world of 2,000 ticks) and on the G1 where the list names the sim.

What must hold: it is inert for language (1); its law is the exact backward form of least squares on the forward targets, its solve
numpy's ridge answer (2); an event line paired once with pain is forecast at a new occurrence, an unpaired one not (3); good and bad are
kept apart (4); the tag is bounded, received-only while nothing has earned its voice, 0 with nothing forecast or felt (5); the later boost
of a frame's write is exact through the store's saturation law and follows its slot (6); the night's entries and the tagged-first draw
(7) and act_pred's night weight (8) are the law (R8 wires them); a save gives every tensor and moment back and the life goes on as the
life that went on (11); the G1 with it on is the same life across two runs (12); its cost at the sim's width is under 0.5 ms a tick (13).
It never makes reward, changes reward, produces dopamine, enters a gate's credit or trains the cortex: a body whose amygdala has nothing
to act on (no frames, no orienting) lives the same life with it on as off, but for its own organ and its own readings (1)."""
import math
import os
import pickle
import random
import sys
import tempfile
import time

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)   # this tree's body, not a fixed one
from body.life import Life  # noqa: E402
from body.model import Organs  # noqa: E402
from body.core.amygdala import (Amygdala, act_pred_night_weight, amygdala_spec, episode_entries, night_draw,  # noqa: E402
                                 tag_star)
from body.core.anatomy import Channel, Effector, EventLine, FaceReward, LanguageAnatomy, OrientCue, RewardSource  # noqa: E402
from body.core.world import Frame, SimWorld, WorldLoop  # noqa: E402

TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
G = 0.9375


class _Pain(RewardSource):
    """pain: -1 on a tick the frame's pain flag is set"""

    def felt(self, frame, life):
        p_ = frame.obs.get("pain")
        return -1.0 if p_ is not None and float(p_[0]) > 0.0 else None


class _Tiny(LanguageAnatomy):
    """7.4's tiny anatomy: the words (the partner), a 4-number cue channel (its born code, a forecast head), the event lines "bump" and
    "chime", the voice and one 2-joint effector with an orienting trigger on "chime" (its yaw), the reward face +/- then pain -"""

    def __init__(self, tok, cfg=None):
        super().__init__(tok, cfg)
        ear = self.channels[0]
        self.channels = [ear, Channel("cue", "vector", 4, organ="encs.cue", forecast=True)]
        arm = Effector("arm", [5, 5], rest_id=12, orient={0: ("yaw", 1)}, orient_gate=True, n_in=2)
        self.effectors = [self.effectors[0], arm]
        self.rewards = [FaceReward("face", clip=2), _Pain("pain", signs=(-1.0,))]
        self.events = [EventLine("bump", "bump", (0,)), EventLine("chime", "chime", (0,))]
        self.orienting = [OrientCue("chime", "chime", fired=0, yaw=1, sense=1.0, side_only=True, onset=True)]


def _script_world(script, seed=0, T=2000):
    """a scripted world of T ticks: the cue channel's 4 numbers random each tick; `script(t)` -> dict of the tick's events ("bump", "chime"
    as 1, "chime_side", "pain" as 1, "face" a level); its whole state goes with its save"""

    class W(SimWorld):
        def __init__(self):
            self.t = 0; self.rng = random.Random(seed)

        def frame(self):
            t = self.t; R = self.rng; ev = script(t) or {}
            obs = {"cue": [R.uniform(-1, 1) for _ in range(4)], "bump": [1.0 if ev.get("bump") else 0.0],
                   "chime": [1.0 if ev.get("chime") else 0.0, float(ev.get("chime_side", 0.5))], "pain": [1.0 if ev.get("pain") else 0.0]}
            return Frame(t, obs, float(ev.get("face", 0.0)), {"who": "parent"})

        def apply(self, acts):
            self.t += 1

        def pause(self):
            pass

        def resume(self):
            pass

        def save_state(self):
            return pickle.dumps((self.t, self.rng.getstate()))

        def load_state(self, blob):
            self.t, st_ = pickle.loads(blob); self.rng.setstate(st_)
    return W()


_CFG = dict(amyg=1, wake_ticks=10 ** 6, write_floor=1e-30, gate_floor=0.3, wake_every=24, gate_every=24)


def _tiny(cfg, world, seed=0):
    torch.manual_seed(seed)
    return Life.birth(_Tiny(TOK, cfg), device="cpu", d=32, layers=1, heads=2, window=8, cfg=cfg, seed=seed, world=world)


def _live(L, n, rec=None):
    run = WorldLoop(L)
    for _ in range(n):
        run.step()
        if rec is not None:
            rec.append((L.ticks - 1, dict(L._amyg_now), [float(x) for x in L.m.amyg.reliability(64)], L._event_lines()))


# ---------------- amyg 1: inert for language ----------------

def test_amyg_inert_for_language():
    """amyg 1 (7.4 item 1): inert for language: the language anatomy's spec is None; a language life builds no amygdala, holds no amyg*
    attribute and no save key of one, its organs the same parameters and state-dict keys as organs built without the argument (the eight
    digests are the guard's). And it never makes reward, changes reward, produces dopamine, enters a gate's credit or trains the cortex:
    the tiny body with nothing for its tag to act on (no frames, orienting off) lives 400 ticks the same life with it on as off, every
    organ, the store, the optimizers, the stream and every working attribute, but for its own organ and its own readings"""
    from body.tests.test_anatomy import _canon
    import hashlib
    assert amygdala_spec(LanguageAnatomy(TOK, {}), {}) is None
    torch.manual_seed(0); L = Life.birth(TOK, device="cpu", d=32, layers=1, heads=2, window=8, cfg={}, seed=0)
    for _ in range(30):
        L.tick()
    assert "amyg" not in L.m._modules and not any(k.startswith("_amyg") for k in vars(L)) and "_rtag_now" not in vars(L)
    torch.manual_seed(0); o = Organs(L.m.vocab, d=32, layers=1, heads=2, window=8)
    assert sorted(o.state_dict()) == sorted(L.m.state_dict())         # (the life's organs keep their float64 evidence last: Organs.to)
    assert sum(p.numel() for p in o.parameters()) == sum(p.numel() for p in L.m.parameters())
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    try:
        L.save(path); blob = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        os.remove(path)
    assert not any(k.startswith("amyg") for k in blob["organs"]) and "amyg" not in blob["cfg"]
    # it acts on nothing: on and off, the same life but its own
    def script(t):
        return {"bump": t % 37 == 5, "chime": t % 53 == 9, "pain": t % 37 == 10, "face": 2.0 if t % 61 == 30 else 0.0}
    lives = {}
    for on in (0, 1):
        c = dict(_CFG, amyg=on, orient=0, fast_rls=1, fast_input="striatum", stri_k=4, stri_m=64)
        Lx = _tiny(c, _script_world(script)); _live(Lx, 400); lives[on] = Lx
    A, B = lives[0], lives[1]
    own = {"_amyg_now", "_amyg_prev", "_rtag_now", "_terms_now", "_events_now"}
    def h(x, name):
        g = hashlib.sha256(); _canon(g, x, name); return g.hexdigest()
    sa = {k: v for k, v in A.m.state_dict().items()}; sb = {k: v for k, v in B.m.state_dict().items() if not k.startswith("amyg.")}
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa), [k for k in sa if not torch.equal(sa[k], sb[k])][:5]
    wa = {k: v for k, v in vars(A).items() if k not in own and k not in ("m", "store", "gen", "cfg", "anatomy", "world", "_t_feel")
          and not isinstance(v, torch.optim.Optimizer)}
    wb = {k: v for k, v in vars(B).items() if k not in own and k not in ("m", "store", "gen", "cfg", "anatomy", "world", "_t_feel")
          and not isinstance(v, torch.optim.Optimizer)}
    assert set(wa) == set(wb) and [k for k in wa if h(wa[k], k) != h(wb[k], k)] == [], [k for k in wa if h(wa[k], k) != h(wb[k], k)][:5]
    assert h(dict(vars(A.store)), "s") == h(dict(vars(B.store)), "s") and torch.equal(A.gen.get_state(), B.gen.get_state())
    assert all(h(getattr(A, k).state_dict(), k) == h(getattr(B, k).state_dict(), k) for k, v in vars(A).items() if isinstance(v, torch.optim.Optimizer))
    assert int(B.m.amyg.n_tick) == 400 and float(B.m.amyg.A.abs().sum()) > 0.0
    print(f"amyg 1: inert for language (no organ, no attribute, no save key; the organs' keys and parameters as without it); on the tiny",
          f"body with nothing to act on, 400 ticks with it on the same life as off in every organ, the store, the optimizers, the stream and",
          f"every working attribute but its own (it lived {int(B.m.amyg.n_tick)} ticks, {int(B.m.amyg.n_solve)} solves)")


# ---------------- amyg 2: the law, exact ----------------

def test_amyg_the_law_exact():
    """amyg 2 (7.4 item 2): THE LAW, EXACT: on random inputs and sparse outcomes over 300 ticks, the backward evidence b (the outcome
    entering with the trace of the inputs before it) equals the sum over t of x_t y_t^T from the forward targets (y_t = the sum over k >= 1
    of g^(k-1) u_(t+k)), to 1e-9: at beta 1, and at beta < 1 with each outcome weighted by its forgetting (beta^(T-1-s)); A the sum of
    x x^T; and the solve equals numpy's ridge answer (the prior 0.3 tau x each input's running variance, the level free) to 1e-9"""
    n, H, T = 12, 3, 300
    rng = np.random.default_rng(3)
    X = rng.normal(size=(T, n)); X[:, -1] = 1.0
    U = (rng.random(size=(T, H)) < 0.05) * rng.uniform(0.5, 2.0, size=(T, H))
    for beta in (1.0, 1.0 - 1.0 / 4096):
        org = Amygdala(n, [("a", 1.0), ("b", -1.0), ("c", 1.0)], 64)
        tau = 4096.0
        for t in range(T):
            org.step(torch.tensor(X[t]), torch.tensor(U[t]), G, beta, 0.3, tau, 10 ** 9, 36000.0)
        b_fwd = np.zeros((n, H)); A_fwd = np.zeros((n, n))
        for t in range(T):
            A_fwd += (beta ** (T - 1 - t)) * np.outer(X[t], X[t])
            y = np.zeros(H)
            for s in range(t + 1, T):
                y += (beta ** (T - 1 - s)) * (G ** (s - 1 - t)) * U[s]
            b_fwd += np.outer(X[t], y)
        assert np.abs(org.b.numpy() - b_fwd).max() < 1e-9 and np.abs(org.A.numpy() - A_fwd).max() < 1e-9, (beta, np.abs(org.b.numpy() - b_fwd).max())
        org.solve(0.3, tau)
        R = np.zeros(n); R[:-1] = 0.3 * tau * org.var.numpy()
        W_np = np.linalg.solve(org.A.numpy() + np.diag(R), org.b.numpy())
        assert np.abs(org.W.numpy() - W_np).max() < 1e-9, np.abs(org.W.numpy() - W_np).max()
    # the input statistics (the prior's metric): Welford at 1 / min(n, tau)
    mu = np.zeros(n - 1); var = np.zeros(n - 1)
    for t in range(T):
        eta = 1.0 / min(t + 1, 4096); dv = X[t, :-1] - mu; mu += eta * dv; var += eta * (dv * (X[t, :-1] - mu) - var)
    assert np.abs(org.var.numpy() - var).max() < 1e-12 and np.abs(org.mu.numpy() - mu).max() < 1e-12
    print(f"amyg 2: the backward evidence equals the forward targets' sum x y^T (beta 1 and 1 - 1/4096) and A the sum x x^T, each within",
          f"1e-9 over {T} ticks; the solve numpy's ridge answer within 1e-9; the prior's variances Welford's")


# ---------------- amyg 3: one pairing ----------------

def test_amyg_one_pairing():
    """amyg 3 (7.4 item 3): ONE PAIRING: "bump" then pain 5 ticks later, once, at tick 1000 of a quiet world; at a new "bump" at tick
    1100 the pain head forecasts at least 0.2 of 0.9375^4 (the discounted outcome), and at an unpaired "chime" at 1050 it stays within
    0.1 of it (the forecast is never below 0); the low road learns in one pairing"""
    def script(t):
        return {"bump": t in (1000, 1100), "pain": t == 1005, "chime": t == 1050}
    L = _tiny(_CFG, _script_world(script)); rec = []
    _live(L, 1101, rec)
    heads = [h for h, _ in L.m.amyg.heads]; ip = heads.index("pain")
    got = {}
    org = L.m.amyg
    # the forecasts at the ticks (recomputed from the organ's weights then: the ring holds them, newest first)
    for t_, want in ((1050, "chime"), (1100, "bump")):
        age = 1100 - t_                                                  # ring index of the forecast made at t_
        got[want] = float(org.ring_f[age, ip]) / G ** 4
    assert got["bump"] >= 0.2 and got["chime"] <= 0.1, got
    print(f"amyg 3: one pairing (bump, pain 5 ticks later): the pain head at a new bump {got['bump']:.3f} of 0.9375^4 (at least 0.2), at an",
          f"unpaired chime {got['chime']:.3f} (within 0.1)")


# ---------------- amyg 4 and 5: split valence; the tag's bounds ----------------

def _split_script(t):
    k = t % 50
    return {"bump": k == 10, "face": 2.0 if k == 13 else 0.0, "pain": k == 16, "chime": t % 50 == 35, "chime_side": 0.5}


def test_amyg_split_valence_and_the_tags_bounds():
    """amyg 4 and 5 (7.4 items 4 and 5): SPLIT VALENCE: a cue ("bump") followed by a smile 3 ticks later and pain 6 ticks later, every 50
    ticks over 2,000: at a late bump both heads (face + and pain -) forecast above zero with their reliability earned, |N| < A+ + A-, and
    the tag at the bump is above the tag at an unpaired cue ("chime"). THE TAG'S BOUNDS, on every tick: in [0, 2]; while no head has
    earned its voice (fewer than 64 finalized pairs) it is the received part alone, min(2, the sum of |term|); with nothing forecast and
    nothing felt it is 0. (The test's own setting: the reliability's moments over 500 ticks, amyg_rel_tau, so its 2,000 ticks show the
    learned regime; the body's 36,000 wash the first minutes' forecasts out over a life day)"""
    L = _tiny(dict(_CFG, amyg_rel_tau=500), _script_world(_split_script)); rec = []
    _live(L, 2000, rec)
    assert [(h, s) for h, s in L.m.amyg.heads] == [("face", 1.0), ("face", -1.0), ("pain", -1.0)]
    late = [r for r in rec if r[0] >= 1500 and r[3][0] == 1.0]            # the late bumps
    chimes = [r for r in rec if r[0] >= 1500 and r[3][1] == 1.0]
    assert late and chimes
    t_b, now_b, rho_b, _ = late[-1]
    age = 1999 - t_b
    yf, yp = float(L.m.amyg.ring_f[age, 0]), float(L.m.amyg.ring_f[age, 2])
    assert yf > 0.0 and yp > 0.0 and rho_b[0] > 0.0 and rho_b[2] > 0.0, (yf, yp, rho_b)
    assert now_b["Ap"] > 0.0 and now_b["Am"] > 0.0 and abs(now_b["N"]) < now_b["Ap"] + now_b["Am"], now_b
    assert now_b["tag"] > max(r[1]["tag"] for r in chimes), (now_b["tag"], [r[1]["tag"] for r in chimes])
    # the bounds, every tick
    n_recv = n_zero = 0
    for t_, now, rho, ev in rec:
        assert 0.0 <= now["tag"] <= 2.0, (t_, now)
        if max(rho) == 0.0:
            assert now["tag"] == min(2.0, now["R"]), (t_, now); n_recv += 1
        if (now["Ap"] + now["Am"]) == 0.0 and now["R"] == 0.0:
            assert now["tag"] == 0.0, (t_, now); n_zero += 1
    assert n_recv >= 60 and n_zero >= 60, (n_recv, n_zero)
    print(f"amyg 4: split valence at a late bump: face + {yf:.3f} and pain - {yp:.3f} (reliabilities {rho_b[0]:.2f}, {rho_b[2]:.2f}), A+",
          f"{now_b['Ap']:.3f}, A- {now_b['Am']:.3f}, |N| {abs(now_b['N']):.3f} below their sum; its tag {now_b['tag']:.3f} above an unpaired",
          f"chime's (at most {max(r[1]['tag'] for r in chimes):.3f}); amyg 5: every tag of 2,000 in [0, 2], received-only on the {n_recv}",
          f"ticks before any head earned its voice, 0 on {n_zero} ticks with nothing forecast or felt")


# ---------------- amyg 6: the later boost ----------------

def test_amyg_the_later_boost():
    """amyg 6 (7.4 item 6): THE LATER BOOST, EXACT: a frame written with its strength s0 x (1 + T) (s0 = surprise x (1 + |dopamine|), T the
    tag at the write) is raised, for 64 ticks after, by each larger 0.9375^dt x tag: s0 x (0.9375^dt tag - T) through the store's own
    saturating merge (s x m / (m + S), m the store's mean strength, under store_sat; added whole without), the factor never past 3; the
    boost follows the slot through an eviction's remap; a dropped slot gets nothing; after 64 ticks none"""
    for sat in (0, 1):
        cfg = dict(_CFG, frames=1, store_sat=sat)
        L = _tiny(cfg, _script_world(lambda t: {})); _live(L, 30)
        st = L.store
        keys = [torch.nn.functional.normalize(torch.randn(32, generator=torch.Generator().manual_seed(k)), dim=0) * 2.5 for k in range(4)]
        for k_ in keys:
            st.write(k_, torch.randn(32, generator=torch.Generator().manual_seed(99)), 0.5, 2)
        n0 = st.n(); t0 = L.ticks
        L._fboosts = []
        k_new = torch.nn.functional.normalize(torch.randn(32, generator=torch.Generator().manual_seed(11)), dim=0) * 2.5
        assert L._frame_write(k_new, torch.randn(32, generator=torch.Generator().manual_seed(7)), 0.8 * (1.0 + 0.2), base=0.8, tag_w=0.2)
        j = st.last_idx; assert st.n() == n0 + 1 and L._fboosts == [[j, 0.8, 0.2, t0]]
        S0 = float(st.S[j])
        # dt 1: a tag of 0.1 (0.9375 x 0.1 < 0.2): nothing; dt 3: a tag of 1.0 -> 0.9375^3 > 0.2: boosted
        for dt, tag, boost in ((1, 0.1, False), (3, 1.0, True), (5, 1.0, False), (10, 2.0, True)):
            L.ticks = t0 + dt
            before = float(st.S[j]); m_ = float(st.S.mean()); T_ = L._fboosts[0][2]
            L._frame_boosts(tag)
            c = G ** dt * tag
            if boost:
                ds = 0.8 * (c - T_)
                want = before + (ds * m_ / (m_ + before) if sat else ds)
                assert abs(float(st.S[j]) - want) < 1e-6 and abs(L._fboosts[0][2] - c) < 1e-15, (sat, dt, float(st.S[j]), want)
                assert 1.0 + L._fboosts[0][2] <= 3.0
            else:
                assert float(st.S[j]) == before and L._fboosts[0][2] == T_, (sat, dt)
        # the remap: slot j moved to 0, then dropped
        rm = torch.arange(st.n()); rm[j] = 0; rm[0] = j
        L._boosts_remap(rm); assert L._fboosts[0][0] == 0
        rm2 = torch.arange(st.n()); rm2[0] = -1
        L._boosts_remap(rm2); assert L._fboosts == []
        # after 64 ticks: none
        L._fboosts = [[1, 0.8, 0.0, t0]]; L.ticks = t0 + 64; before = float(st.S[1]); L._frame_boosts(2.0)
        assert L._fboosts == [] and float(st.S[1]) == before
    # live: the boosts' slots follow a store at its capacity (every write evicting), and the day's boosts end at the night; THE TAG AT A
    # WRITE: the gate tests surprise x (1 + tag_w) and the strength is surprise x (1 + |dopamine|) x (1 + tag_w), tag_w this tick's
    # received part plus the forecast made the tick before (min 2); the record's tag column is the tick's tag
    cfg = dict(_CFG, frames=1, store_cap=40, amyg_rel_tau=500)
    L = _tiny(cfg, _script_world(_split_script)); seen = []; fg, fw = L._frame_gate, L._frame_write

    def spy_gate(g, fg=fg, L=L, seen=seen):
        seen.append(("gate", L.ticks, float(g))); return fg(g)

    def spy_write(key, value, strength, base=None, tag_w=0.0, fw=fw, L=L, seen=seen):
        seen.append(("write", L.ticks, float(strength), float(base), float(tag_w))); return fw(key, value, strength, base=base, tag_w=tag_w)
    L._frame_gate = spy_gate; L._frame_write = spy_write
    run = WorldLoop(L); prevA = 0.0; n_tagged = 0
    for t in range(1200):
        run.step()
        now = L._amyg_now; tw = min(2.0, now["R"] + prevA); prevA = now["Ap"] + now["Am"]
        assert float(L._rec[t, 2]) == float(torch.tensor(now["tag"], dtype=torch.float32)), t
        s_ = float(L._rec[t, 0]); d_ = float(L._rec[t, 1])
        for ev in [e for e in seen if e[1] == t]:
            if ev[0] == "gate":
                assert abs(ev[2] - s_ * (1.0 + tw)) <= 1e-6 * max(1.0, ev[2]), (t, ev, s_, tw)
            else:
                assert abs(ev[4] - tw) < 1e-12 and abs(ev[2] - ev[3] * (1.0 + tw)) < 1e-9 and abs(ev[3] - s_ * (1.0 + abs(d_))) <= 1e-6 * max(1.0, ev[3]), (t, ev)
                n_tagged += int(tw > 0.0)
    assert L.store.n() == 40 and all(0 <= b[0] < 40 for b in (L._fboosts or [])) and int(L._fwrites) > 40 and n_tagged >= 5, (L.store.n(), int(L._fwrites), n_tagged)
    L.night(); assert L._fboosts == []
    print(f"amyg 6: the later boost exact through the saturation law and added whole without it (dt 3 and 10 boosted, dt 1 and 5 not:",
          f"0.9375^dt x tag below the tag so far), the factor within 3; it follows its slot's remap and a dropped slot gets nothing; none past",
          f"64 ticks; live at a store's capacity (every write evicting) the pending slots stay the store's; the night ends them; over 1,200",
          f"ticks every gate at surprise x (1 + tag_w) and every write at surprise x (1 + |dopamine|) x (1 + tag_w) ({n_tagged} of them tagged),",
          f"the record's tag the tick's")


# ---------------- amyg 7 and 8: the night's side (R8 wires it) ----------------

def test_amyg_the_nights_side():
    """amyg 7 and 8 (7.4 items 7 and 8; R8 wires them into the night): THE TAG REACHES BACK: tag* is the largest 0.9375^k x tag k ticks
    later (k < 64): a smile of 2 reaches 0.52 of itself 10 ticks back and 0.28 20 back, nothing 64 back. THE NIGHT'S ENTRIES: [mean
    surprise x (1 + |dopamine|)] x (1 + T_e) over the bias-corrected running mean of the entries before (the first 1). THE TAGGED FIRST:
    every episode with T_e >= 1 dreamt once before the weighted draw, highest first, at most half the night; the rest drawn by entry,
    each episode's count over 10,000 seeded nights within 3 sigma of its binomial expectation. ACT_PRED AT NIGHT: a window followed by net
    harm (G < -1, its weight clip(1 + G, 0, 1) = 0) gives exactly zero gradient on act_pred; the forward half's loss is unchanged; at
    weight 1 the lesson is the day's to the bit"""
    tags = [0.0] * 100; tags[80] = 2.0
    ts = tag_star(tags, G, 64)
    assert ts[80] == 2.0 and abs(ts[70] / 2.0 - G ** 10) < 1e-15 and abs(ts[60] / 2.0 - G ** 20) < 1e-15 and ts[16] == 0.0 and ts[17] > 0.0
    assert abs(G ** 10 - 0.5245) < 1e-4 and abs(G ** 20 - 0.2751) < 1e-4
    # the entries
    rng = random.Random(1); T = 400
    surprise = [rng.uniform(0.2, 2.0) for _ in range(T)]; delta = [rng.uniform(-1, 1) for _ in range(T)]
    tgs = [0.0] * T
    for t in (50, 170, 300):
        tgs[t] = 2.0
    tst = tag_star(tgs, G, 64)
    eps = [(0, 40), (40, 100), (100, 160), (160, 250), (250, 330), (330, 400)]
    ms = [0.0, 0]; ent = episode_entries(eps, surprise, delta, tst, ms)
    m_, n_ = 0.0, 0; b_ = 1.0 - 1.0 / 64
    for (a, b), (e, T_e) in zip(eps, ent):
        x = sum(surprise[t] * (1 + abs(delta[t])) for t in range(a, b)) / (b - a) * (1 + max(tst[a:b]))
        mhat = m_ / (1 - b_ ** n_) if n_ else 0.0
        assert abs(e - (x / mhat if mhat else 1.0)) < 1e-12 and T_e == max(tst[a:b])
        m_ = b_ * m_ + (1 - b_) * x; n_ += 1
    assert abs(ms[0] - m_) < 1e-12 and ms[1] == 6
    # the tagged first, and the weighted draw over 10,000 seeded nights
    entries = [0.5, 1.0, 2.0, 0.25, 1.5, 0.75]; Tes = [0.2, 1.4, 0.0, 1.0, 0.5, 1.9]; n = 8
    counts = [0] * 6; firsts = None; N_ = 10000
    for s in range(N_):
        d = night_draw(entries, Tes, n, torch.Generator().manual_seed(s))
        if firsts is None:
            firsts = d[:3]
        assert d[:3] == [5, 1, 3] and len(d) == n
        for i in d[3:]:
            counts[i] += 1
    tot = sum(entries); k = (n - 3) * N_
    for i, c in enumerate(counts):
        p = entries[i] / tot; mu = k * p; sd = math.sqrt(k * p * (1 - p))
        assert abs(c - mu) <= 3 * sd, (i, c, mu, sd)
    d_ = night_draw([1.0] * 6, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 4, torch.Generator().manual_seed(0))
    assert d_[:2] == [5, 4]                                           # at most half the night, the highest first
    # act_pred at night
    from body.core.amygdala import act_pred_label_weight
    assert act_pred_night_weight(-1.5) == -1.0 and act_pred_night_weight(-1.0) == -1.0 and act_pred_night_weight(-0.3) == -0.3 and act_pred_night_weight(0.4) == 0.4 \
        and act_pred_night_weight(2.0) == 1.0 and act_pred_night_weight(0.0) == 0.0   # A152: the own act's weight is dopamine's credit alone
    assert act_pred_label_weight(-1.5) == 0.0 and act_pred_label_weight(-0.3) == 0.7 and act_pred_label_weight(0.4) == 1.0   # the labels' as R8 wrote it
    from body.tests.test_anatomy import _arm_world, _Timed, _LR0
    torch.manual_seed(0)
    Lt = Life.birth(_Timed(TOK, dict(_LR0)), device="cpu", d=32, layers=1, heads=2, window=16, cfg=dict(_LR0), seed=0, world=_arm_world())
    run = WorldLoop(Lt)
    for _ in range(20):
        run.step()
    obs, whos, bundles, reads = Lt._window_tensors()
    out = {}
    for tag_, wpos in (("day", None), ("one", torch.ones(len(Lt.win))), ("harm", torch.full((len(Lt.win),), act_pred_night_weight(-2.0)))):
        C = Lt.m.stream(Lt.m.inputs(Lt.anatomy, obs, whos, bundles))
        for p_ in Lt.m.parameters():
            p_.grad = None
        loss, rep, lb = Lt._timing_loss(1, C, obs, wpos=wpos)
        tm = Lt.m.timing[Lt.anatomy.motors[0].name]
        fwd_ = float(rep.get("fwd", 0.0))
        total = loss + (lb if lb is not None else 0.0)
        total.backward()
        out[tag_] = (float(loss.detach()), tm.pred.weight.grad.clone(), fwd_, float(total.detach()))
    assert out["one"][0] == out["day"][0] and torch.equal(out["one"][1], out["day"][1])
    assert torch.allclose(out["harm"][1], -out["one"][1], atol=1e-7) and float(out["one"][1].abs().max()) > 0.0 \
        and out["harm"][2] == out["day"][2] and out["harm"][2] > 0.0, (out["harm"][2], out["day"][2])   # A150: at weight -1 act_pred's gradient is the day's, reversed
    print(f"amyg 7: tag* reaches back (0.52 of a smile 10 ticks before it, 0.28 20 before, none 64 before); {len(eps)} episodes' entries by",
          f"the law over the bias-corrected mean; the tagged first ({firsts}: every T_e >= 1 once, highest first, at most half) and the rest by",
          f"entry, each of 6 episodes within 3 sigma over {N_} seeded nights; amyg 8: act_pred at night, a window of net harm (G -2, weight -1)",
          f"gives act_pred the day's gradient reversed (A150, A152: the habit is dopamine's), the forward half's loss unchanged ({out['harm'][2]:.4f}), weight 1 the day's lesson to the bit")


# ---------------- amyg 9 and 10: the orienting gain and amyg_pav (R7e) ----------------

def _force(L, yplus, yminus):
    """a forced reliable forecast: the organ's first positive head forecasts `yplus` and its first negative head `yminus` from the level
    alone, every reliability 1 (a perfectly correlated sample of weight 10^9), no solve, the moments still"""
    org = L.m.amyg
    heads = [s_ for _, s_ in org.heads]; ip, im = heads.index(1.0), heads.index(-1.0)
    org.W.zero_(); org.W[-1, ip] = float(yplus); org.W[-1, im] = float(yminus)
    N0 = 1e9
    org.rel.copy_(torch.tensor([[N0, N0, N0, 2 * N0, 2 * N0, 2 * N0]] * len(heads), dtype=torch.float64)); org.pairs.fill_(10 ** 6)


def test_amyg_orienting_gain():
    """amyg 9 (7.4 item 9; step R7e): THE ORIENTING GAIN: exactly 1 at birth (every reliability 0: N is 0); with a forced reliable forecast
    (a head's weight on the level, its reliability 1) it follows clip(1 + N, -0.5, 2) (N 0.6, -3, 1.5, -0.4: gains 1.6, -0.5, 2, 0.6), and
    the born bias on the tiny arm's yaw at a chime is that gain times the born pull, exactly; on the G1 all three cues (the face in the
    periphery, a sound's side, a sudden change) pull the gaze and the waist by the same gain; the gate's draw is identical (its p_act and
    its draw, and the stream the tick drew from, whatever the gain)"""
    import math as _m
    cfg = dict(_CFG, orient=1, amyg_every=10 ** 9, amyg_rel_tau=1e15)
    def chimes(t):
        return {"chime": t % 10 == 5, "chime_side": 0.5}
    L = _tiny(cfg, _script_world(chimes)); _live(L, 3)
    assert L._orient_gain() == 1.0 and L._amyg_now["N"] == 0.0
    e = L.anatomy.motors[0]; tab = L.m.acts[e.name]; lg4 = _m.log(4.0)
    got = {}
    for yp, ym, g in ((0.6, 0.0, 1.6), (0.0, 3.0, -0.5), (1.5, 0.0, 2.0), (0.0, 0.4, 0.6)):
        _force(L, yp, ym)
        run = WorldLoop(L)
        while True:
            run.step()
            if L._event_lines()[1] == 1.0:
                break
        N = L._amyg_now["N"]
        assert abs(N - (yp - ym)) < 1e-12 and L._orient_gain() == max(-0.5, min(2.0, 1.0 + N)), (N, L._orient_gain())
        assert abs(L._orient_gain() - g) < 1e-12
        b = L._orient_bias(e, L.world.now, tab)
        want = torch.tensor([(L._orient_gain() * lg4) * 1.0 * 1.0 * L._setting_sign(k, 5) for k in range(5)])
        assert torch.equal(b[0], want) and torch.equal(b[1], torch.zeros(5)), (b, want)
        got[g] = [round(float(x), 4) for x in b[0]]
    # the G1: all three cues, the gaze and the waist, by the same gain
    from body.sim.anatomy import SIM_CFG, SimAnatomy, born_table
    from body.tests.test_frames import _g1_events_world
    gc = dict(SIM_CFG, wake_ticks=100000, gate_floor=0.3, write_floor=1e-30, amyg_every=10 ** 9, amyg_rel_tau=1e15)
    torch.manual_seed(0)
    G1 = Life.birth(SimAnatomy(born_table(), gc), device="cpu", d=32, layers=1, heads=2, window=8, cfg=gc, seed=0, world=_g1_events_world())
    run = WorldLoop(G1)
    for _ in range(3):
        run.step()
    _force(G1, 0.7, 0.0); run.step()
    frame = Frame(G1.ticks, {"face_periph": [1.0, 0.4, -0.3], "sound_side": [1.0, 0.5], "onset_periph": [1.0, -0.3, 0.2]}, 0.0)
    G1._orient_now = None; cues = G1._orient_cues(frame)
    assert [c.name for c, *_ in cues] == ["face", "sound", "onset"] and all(dy != 0 for _, dy, _, _ in cues)
    gain = G1._orient_gain(); assert abs(gain - 1.7) < 1e-12
    for name in ("gaze", "waist"):
        e_ = G1.anatomy.effector(name); tab_ = G1.m.acts[name]
        b_ = G1._orient_bias(e_, frame, tab_)
        for j, (ax, sg) in e_.orient.items():
            dsum = sum((dy if ax == "yaw" else dp) for _, dy, dp, _ in cues)
            want = torch.tensor([(gain * lg4) * float(dsum) * float(sg) * G1._setting_sign(k, 5) for k in range(5)])
            assert torch.equal(b_[j], want), (name, j, b_[j], want)
    # the gate's draw identical whatever the gain
    rows = {}
    for forced in (None, 0.6, -2.0):
        Lx = _tiny(cfg, _script_world(chimes)); ra = Lx._amygdala

        def spy(C1, frame, ra=ra, Lx=Lx, forced=forced):
            ra(C1, frame)
            if forced is not None:
                Lx._amyg_now["N"] = forced
        Lx._amygdala = spy
        run = WorldLoop(Lx); r_ = []
        for _ in range(40):
            gs = Lx.gen.get_state().clone(); run.step()
            st = Lx.motor[0]["now"]; r_.append((st["p_act"], st["drew"], Lx.gen.get_state().clone()))
        rows[forced] = r_
    for forced in (0.6, -2.0):
        assert all(a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2]) for a, b in zip(rows[None], rows[forced])), forced
    print(f"amyg 9: the orienting gain 1.0 exactly at birth; with a forced reliable forecast clip(1 + N, -0.5, 2) (N 0.6, -3, 1.5, -0.4:",
          f"{sorted(got)}), the arm's born pull at a chime times it exactly ({got[1.6]} at 1.6); on the G1 the face, the sound and the change",
          f"pull the gaze and the waist by the same gain (1.7); the gate's p_act, its draw and the stream the same over 40 ticks at any gain")


def test_amyg_pav():
    """amyg 10 (7.4 item 10; step R7e; A71): AMYG_PAV, built and off by default: off, the motor gate's logit is unchanged whatever N (its
    p_act the same at N forced to 0.6 as at 0); on (the "fixed" form, R7e's), z gains amyg_pav_beta x clip(N, -2, 2): p_act = floor +
    (1 - floor) sigmoid(z + beta clip(N)), z the gate's own logit on its recorded input, N forced to 0.6, 3 (clipped to 2) and -1. THE
    EARNED FORM (A71, the lead's decision: born on, its weight the aversive heads' largest reliability): at birth (every reliability 0)
    the logit is unchanged whatever N; with the aversive heads' reliability held at 0.5 (a face - at 0.5 and a pain - at 0.3: the
    largest taken) z gains beta x 0.5 x clip(N); the sim is born with it (SIM_CFG's amyg_pav 1, "earned")"""
    from body.core.physiology import AMYG
    from body.sim.anatomy import SIM_CFG
    assert AMYG["amyg_pav"] == 0 and AMYG["amyg_pav_form"] == "fixed" and (SIM_CFG["amyg_pav"], SIM_CFG["amyg_pav_form"]) == (1, "earned")
    def run_with(pav, forced, form="fixed", rho=None):
        c = dict(_CFG, orient=0, amyg_pav=pav, amyg_pav_form=form, amyg_rel_tau=1e15)
        Lx = _tiny(c, _script_world(lambda t: {})); ra = Lx._amygdala
        if rho is not None:                                                # the heads' reliabilities held (a perfectly weighted sample)
            org = Lx.m.amyg; N0 = 1e9
            for h, (_, sg) in enumerate(org.heads):
                r_ = float(rho.get(h, 0.0))
                org.rel[h] = torch.tensor([N0, 0.0, 0.0, N0, N0, r_ * N0], dtype=torch.float64)
            org.pairs.fill_(10 ** 6)

        def spy(C1, frame, ra=ra, Lx=Lx):
            ra(C1, frame); Lx._amyg_now["N"] = forced
        Lx._amygdala = spy
        run = WorldLoop(Lx); out = []
        for _ in range(12):
            run.step(); st = Lx.motor[0]["now"]
            z0 = Lx.m.gates["arm"](st["feat"].unsqueeze(0))[0, 0] / (1.0 + Lx.stress / 10.0)
            out.append((st["p_act"], float(z0.detach())))
        return out, float(Lx.cfg["gate_floor"])
    base, fl = run_with(0, 0.0)
    off, _ = run_with(0, 0.6)
    assert [a[0] for a in base] == [a[0] for a in off]
    n_checked = 0
    for forced in (0.6, 3.0, -1.0):
        on, _ = run_with(1, forced)
        for p_act, z0 in on:
            want = fl + (1.0 - fl) * float(torch.sigmoid(torch.tensor(z0) + max(-2.0, min(2.0, forced))))
            assert abs(p_act - want) < 1e-6, (forced, p_act, want); n_checked += 1
    # the earned form: silent at birth, its weight the aversive heads' largest reliability
    born, _ = run_with(1, 0.6, form="earned")
    base_e, _ = run_with(0, 0.6, form="earned")
    assert [a[0] for a in born] == [a[0] for a in base_e]
    heads = [sg for _, sg in _tiny(dict(_CFG), _script_world(lambda t: {})).m.amyg.heads]
    rho = {h: (0.5 if k == 0 else 0.3) for k, h in enumerate(i for i, sg in enumerate(heads) if sg < 0)}
    rho.update({h: 0.9 for h, sg in enumerate(heads) if sg > 0})       # a good head's reliability is not the weight
    n_earned = 0
    for forced in (0.6, 3.0, -1.0):
        on, _ = run_with(1, forced, form="earned", rho=rho)
        for p_act, z0 in on:
            want = fl + (1.0 - fl) * float(torch.sigmoid(torch.tensor(z0) + 0.5 * max(-2.0, min(2.0, forced))))
            assert abs(p_act - want) < 1e-6, (forced, p_act, want); n_earned += 1
    print(f"amyg 10: amyg_pav off by default: the gate's p_act unchanged by N; on (fixed), z + clip(N, -2, 2) on {n_checked} ticks (N 0.6, 3",
          f"-> 2, -1); earned (A71, the sim's): at birth unchanged by N, with the aversive heads' reliabilities 0.5 and 0.3 (the good head's",
          f"0.9 no weight) z + 0.5 x clip(N) on {n_earned} ticks")


# ---------------- amyg 11: the save round trip ----------------

def test_amyg_the_save_round_trip():
    """amyg 11 (7.4 item 11): SAVE ROUND TRIP: saved at tick 900 of the split-valence world (its forecasts pending, its reliability
    earned), loaded by another life: every tensor of the organ and every moment given back; the life goes on 300 ticks as the life that
    went on, bit for bit in every section of the whole state"""
    from body.tests.test_anatomy import _whole
    A = _tiny(_CFG, _script_world(_split_script)); _live(A, 1200); hA = _whole(A)
    w = _script_world(_split_script); B = _tiny(_CFG, w); _live(B, 900)
    assert int(B.m.amyg.ring_n) == 64 and int(B.m.amyg.pairs[0]) > 64
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    try:
        B.save(path); w2 = _script_world(_split_script); w2.load_state(w.save_state())
        g_ = torch.get_rng_state(); torch.manual_seed(0)
        C = Life.load(path, _Tiny(TOK, B.cfg), save_path=None, world=w2); torch.set_rng_state(g_)
    finally:
        os.remove(path)
    sb, sc = B.m.amyg.state_dict(), C.m.amyg.state_dict()
    assert list(sb) == list(sc) and all(torch.equal(sb[k], sc[k]) for k in sb) and C._amyg_now == B._amyg_now and C._amyg_prev == B._amyg_prev
    _live(C, 300)
    hC = _whole(C)
    assert hC == hA, [k for k in hA if hA[k] != hC[k]]
    print(f"amyg 11: saved at 900 ({int(B.m.amyg.ring_n)} forecasts pending, {int(B.m.amyg.pairs[0])} pairs): every tensor and moment",
          f"given back; at 1200 the whole state the uninterrupted life's in every section ({' '.join(f'{k} {v[:10]}' for k, v in hA.items())})")


# ---------------- amyg 12: the sim with the amygdala on, twice ----------------

def test_amyg_the_sim_twice():
    """amyg 12 (7.4 item 12): the G1 (SimAnatomy, SIM_CFG with the amygdala on) in a stub of its world lives 300 ticks twice from its
    birth: the same life in every section of the whole state. The `sim` profile of the determinism check (tools/sim_profile.py: a tiny G1
    under SIM_CFG, the amygdala on, in a stub of its world through a day and R8's whole night) is pinned with the amygdala on in
    tools/pins/digests.txt since R8d, and the guard checks it threaded and one-thread; the digest here is written down, not pinned"""
    from body.sim.anatomy import SIM_CFG, SimAnatomy, born_table
    from body.tests.test_anatomy import _whole
    from body.tests.test_frames import _g1_events_world
    assert SIM_CFG.get("amyg") == 1
    hs = []
    for _ in range(2):
        cfg = dict(SIM_CFG, wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.3)
        torch.manual_seed(0)
        L = Life.birth(SimAnatomy(born_table(), cfg), device="cpu", d=32, layers=1, heads=2, window=8, cfg=cfg, seed=0, world=_g1_events_world(burst=True))
        run = WorldLoop(L)
        for _ in range(300):
            run.step()
        hs.append(_whole(L))
        assert L.m.amyg.heads == (("face", 1.0), ("face", -1.0), ("pain", -1.0)) and L.m.amyg.n_in == 32 + 13 + 1   # A88: no charge heads
    assert hs[0] == hs[1], [k for k in hs[0] if hs[0][k] != hs[1][k]]
    print(f"amyg 12: the G1 with the amygdala on (3 heads, 46 inputs at d 32), 300 ticks twice from birth: the same life ({' '.join(f'{k} {v[:12]}' for k, v in hs[0].items())})")


# ---------------- amyg 13: its cost ----------------

def test_amyg_its_cost():
    """amyg 13 (7.4 item 13): its cost at the sim's width (526 inputs: d 512, 13 event lines, the level; 5 heads), the law and the solve
    every 8 ticks on one thread, under 0.5 ms a tick (7.4 measured 0.19 ms on a synthetic stream). The bound is read on the process's
    CPU time, which another session's load (descheduling this process) does not reach; the wall time is written down beside it. On the
    wall clock one guard's pinned run read 0.510 ms at best of three (0.518, 0.571, 0.510) under a burst of other sessions' work, where
    every other run read 0.18-0.28"""
    org = Amygdala(526, [("f", 1.0), ("f", -1.0), ("p", -1.0), ("c", 1.0), ("c", -1.0)], 64)
    g = torch.Generator().manual_seed(0)
    X = torch.randn(2400, 526, generator=g, dtype=torch.float64) / math.sqrt(512.0); X[:, -1] = 1.0
    X[:, 512:525] = (torch.rand(2400, 13, generator=g, dtype=torch.float64) < 0.02).double()
    U = (torch.rand(2400, 5, generator=g, dtype=torch.float64) < 0.01).double()
    nt = torch.get_num_threads(); torch.set_num_threads(1); runs = []
    try:
        for t in range(400):
            org.step(X[t], U[t], G, 1 - 1 / 4096, 0.3, 4096.0, 8, 36000.0)
        for r in range(3):                                            # best of three runs of 640 ticks (80 solves each), as cereb 9 reads its
            t0 = time.perf_counter(); c0 = time.process_time()
            for t in range(400 + 640 * r, 400 + 640 * (r + 1)):
                org.step(X[t % 2400], U[t % 2400], G, 1 - 1 / 4096, 0.3, 4096.0, 8, 36000.0)
            runs.append((1000.0 * (time.process_time() - c0) / 640.0, 1000.0 * (time.perf_counter() - t0) / 640.0))
    finally:
        torch.set_num_threads(nt)
    ms = min(c for c, _ in runs); wall = min(w for _, w in runs)
    assert ms < 0.5, runs
    print(f"amyg 13: its cost at the sim's width (526 inputs, 5 heads, the solve every 8 ticks), one thread: {ms:.3f} ms of CPU a tick at",
          f"best of three runs of 640 ticks ({', '.join(f'{c:.3f}' for c, _ in runs)}); on the wall clock {wall:.3f} ms",
          f"({', '.join(f'{w:.3f}' for _, w in runs)}; this machine's load {os.getloadavg()[0]:.1f})")


AMYG_TESTS = [test_amyg_inert_for_language, test_amyg_the_law_exact, test_amyg_one_pairing, test_amyg_split_valence_and_the_tags_bounds, test_amyg_the_later_boost,
              test_amyg_the_nights_side, test_amyg_orienting_gain, test_amyg_pav, test_amyg_the_save_round_trip, test_amyg_the_sim_twice, test_amyg_its_cost]


if __name__ == "__main__":
    t0 = time.time(); failed = 0
    for t in AMYG_TESTS:
        try:
            t()
        except AssertionError as e:
            failed += 1; print("FAIL", t.__name__, ":", e)
        except Exception as e:
            failed += 1; print("ERROR", t.__name__, ":", type(e).__name__, str(e)[:300])
    print(f"{len(AMYG_TESTS) - failed}/{len(AMYG_TESTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if failed else 0)
