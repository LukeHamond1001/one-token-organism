"""the body in frames (docs/SIM_DESIGN.md 7.2, 7.4, 7.6, 8's R7 row, 10; the core refactor's step R7). Run: python3 -m body.tests.test_frames
(the organ tests run these too).

What must hold: the anatomy's born event lines are read from each frame by their rule (any number above 0, on its side, isolated along
its limb's chain), the G1's 13 in the design's order, and enter the striatal expansion through a block of their own appended after the
effectors' (every row before born as it was), kept through the night's emptying, REM's imagination and a save (frames 1, R7a). Each is a
switch or a declaration the language body does not hold: it has none of it (the eight pinned digests are the guard's,
tools/pins/digests.txt). The stub worlds here are instruments of these tests, not the G1's world."""
import math
import os
import pickle
import random
import sys
import tempfile
import time

import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)   # this tree's body, not a fixed one
from body.life import Life  # noqa: E402
from body.core.anatomy import EventLine, LanguageAnatomy  # noqa: E402
from body.core.frames import read_event_lines  # noqa: E402
from body.core.world import Frame, SimWorld, WorldLoop  # noqa: E402

TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")


def _g1_events_world(seed=0, burst=False):
    """a stub of the G1's world for the event lines (never the sim's): motor 11's random senses, but the touch channel's onsets 0 except
    where a contact is scripted (the left hand's palm every 11 ticks with its arm's and the waist's joints loaded too, the right leg's
    joints every 13 with the base, the base alone every 17), pain every 19, a face in the fovea every 23, a sound's onset every 7 (its
    side turning left, right, none), a sudden change every 9 (left, right, centre); `burst`: the senses loud for 6 ticks in every 40 (x 3)
    and soft between (x 0.3), so a newborn's surprise rises and settles; its whole state goes with its save"""
    from body.sim.anatomy import BODY_JOINTS, LIMBS, SIZES, ZONES

    class EvWorld(SimWorld):
        def __init__(self):
            self.t = 0; self.rng = random.Random(seed)

        def frame(self):
            t = self.t; R = self.rng
            a_ = (3.0 if t % 40 < 6 else 0.3) if burst else 1.0
            obs = {n: [a_ * R.uniform(-1, 1) for _ in range(k)] for n, k in SIZES.items() if n not in ("face", "charge")}
            obs["face"] = [2.0 if t % 40 == 20 else 0.0, 0.0]; obs["charge"] = [0.9, -0.0001]
            tch = obs["touch"]
            for i in range(1, len(tch), 2):
                tch[i] = 0.0                                                # every onset quiet unless a contact is scripted
            Z, J = len(ZONES), len(BODY_JOINTS)
            js = dict(LIMBS)
            if t % 11 == 3:                                                 # the left palm: the hand's, the arm's and the waist's joints load
                tch[1] = 0.4
                for n_ in ("hand_l", "arm_l", "waist"):
                    for j in js[n_]:
                        tch[2 * Z + 2 * BODY_JOINTS.index(j) + 1] = 0.05
            if t % 13 == 5:                                                 # the right leg: its joints and the base
                for j in js["leg_r"]:
                    tch[2 * Z + 2 * BODY_JOINTS.index(j) + 1] = 0.1
                tch[2 * Z + 2 * J + 1] = 0.02
            if t % 17 == 8:                                                 # the base alone (a touch on the pelvis)
                tch[2 * Z + 2 * J + 5] = 0.03
            obs["pain"] = [1.0 if (t % 19 == 4 and k == 20) else 0.0 for k in range(44)]
            obs["face_fovea"] = [1.0 if t % 23 == 6 else 0.0]
            if t % 7 == 2:
                obs["sound_side"] = [1.0, (0.5, -0.5, 0.0)[(t // 7) % 3]]
            if t % 9 == 1:
                obs["onset_periph"] = [1.0, (-0.3, 0.3, 0.0)[(t // 9) % 3], 0.1]
            if t % 29 == 0:
                obs["words"] = R.randrange(3, 79)
            return Frame(t, obs, 0.0, {"who": "parent"})

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
    return EvWorld()


def _g1(cfg, world, d=32, seed=0):
    from body.sim.anatomy import SimAnatomy, born_table
    torch.manual_seed(seed)
    return Life.birth(SimAnatomy(born_table(), cfg), device="cpu", d=d, layers=1, heads=2, window=8, cfg=cfg, seed=seed, world=world)


def _stri_ref(L):
    """the striatal expansion by hand from the organs' lines and rows: the language line's rows, each motor effector's per joint, then the
    event lines' (every fired line's row at its position), over the thresholds, relu"""
    m = L.m; width = 2 * m.vocab + 3; z = m.stri_b.clone()
    for p_, e in enumerate(m.stri_line.tolist()):
        if e >= 0:
            z += m.stri_W[p_ * width + e]
    for j, (base, fac) in enumerate(m.stri_blocks):
        S = sum(fac)
        for p_, a in enumerate(m.stri_mline[j].tolist()):
            if a >= 0:
                off = base + p_ * S
                for K, dgt in zip(fac, [(a // math.prod(fac[k + 1:])) % fac[k] for k in range(len(fac))]):
                    z += m.stri_W[off + dgt]; off += K
    base, E = m.stri_eblock
    for p_, mask in enumerate(m.stri_eline.tolist()):
        if mask > 0:
            for i in range(E):
                if (mask >> i) & 1:
                    z += m.stri_W[base + p_ * E + i]
    return torch.relu(z)


# ---------------- frames 1: the event lines (R7a) ----------------

def test_the_event_lines():
    """frames 1 (step R7a; SIM_DESIGN.md 7.2, 7.4's low road, A37, A43): THE G1'S 13 EVENT LINES, declared in the design's order (touch onset
    in 7 groups from the observer and the hands' arrays, pain, a face in the fovea, a sound onset on the left and the right, a visual
    onset on the left and the right), each read by the born rule: any of its numbers above 0; its side where it has one, both lines of a
    pair where the direction is exactly 0 (no side: C42); isolation along its limb's chain (a contact on the left palm loads the hand's,
    the arm's and the waist's joints and fires the hand's line alone; the base alone fires the trunk's; two contacts fire both limbs and
    not the trunk). The check refuses a line twice named, a distal line undeclared, a bad side, a line reading nothing. THE STRIATAL
    EXPANSION: a body that declares them has a block of born rows after the effectors' blocks (every row before it born as without it,
    each new row drawn at 1 / sqrt(k) from the same generator after them), and each tick the fired lines enter their own delay line as
    one event (their bits; under stri_quiet an empty one when none fired); the expansion is the rows of every line's events over the
    thresholds, relu, checked by hand on every tick; the night empties the line, REM's imagination gives it back as it was, and a save
    keeps it. The language anatomy declares none: its organs hold no event line and its striatum is sized as before"""
    from body.sim.anatomy import BODY_JOINTS, LIMBS, SIM_CFG, SimAnatomy, ZONES, born_table
    a = SimAnatomy(born_table(), SIM_CFG).check()
    names = [x.name for x in a.events]
    assert names == ["touch_trunk", "touch_arm_l", "touch_arm_r", "touch_hand_l", "touch_hand_r", "touch_leg_l", "touch_leg_r", "pain",
                     "face_fovea", "sound_l", "sound_r", "visual_l", "visual_r"], names
    Z, J = len(ZONES), len(BODY_JOINTS); js = dict(LIMBS)
    jo = lambda n_: {2 * Z + 2 * BODY_JOINTS.index(j) + 1 for j in js[n_]}  # noqa: E731
    ev = {x.name: x for x in a.events}
    assert set(ev["touch_trunk"].fired) == jo("waist") | {2 * Z + 2 * J + 2 * w + 1 for w in range(6)}
    assert set(ev["touch_hand_l"].fired) == jo("hand_l") | {2 * k + 1 for k in range(8)} and set(ev["touch_hand_r"].fired) == jo("hand_r") | {2 * k + 1 for k in range(8, 16)}
    assert all(set(ev[f"touch_{n_}"].fired) == jo(n_) for n_ in ("arm_l", "arm_r", "leg_l", "leg_r"))
    assert ev["touch_arm_l"].distal == ("touch_hand_l",) and len(ev["touch_trunk"].distal) == 6 and ev["touch_leg_l"].distal == ()
    assert (ev["pain"].obs, ev["pain"].fired, ev["face_fovea"].obs) == ("pain", tuple(range(44)), "face_fovea")
    assert [(ev[n_].obs, ev[n_].side) for n_ in ("sound_l", "sound_r", "visual_l", "visual_r")] == \
        [("sound_side", (1, 1.0)), ("sound_side", (1, -1.0)), ("onset_periph", (1, -1.0)), ("onset_periph", (1, 1.0))]
    assert LanguageAnatomy(TOK, {}).events is None
    # THE RULE on crafted frames
    def fire(**obs):
        t_ = [0.0] * 130
        for k_, v_ in obs.pop("touch", {}).items():
            t_[k_] = v_
        return {n_ for n_, v_ in zip(names, read_event_lines(a.events, dict(obs, touch=t_))) if v_}
    palm = {1: 0.4, **{k_: 0.05 for k_ in jo("hand_l") | jo("arm_l") | jo("waist")}}
    cases = [(dict(touch=palm), {"touch_hand_l"}),
             (dict(touch={k_: 0.05 for k_ in jo("arm_l") | jo("waist")}), {"touch_arm_l"}),
             (dict(touch={2 * Z + 2 * J + 1: 0.02}), {"touch_trunk"}),
             (dict(touch={**{k_: 0.1 for k_ in jo("leg_r")}, 2 * Z + 2 * J + 1: 0.02}), {"touch_leg_r"}),
             (dict(touch={**palm, **{k_: 0.1 for k_ in jo("leg_r")}}), {"touch_hand_l", "touch_leg_r"}),
             (dict(touch={k_: 0.05 for k_ in jo("hand_r")}), {"touch_hand_r"}),
             (dict(touch={k_: -0.3 for k_ in jo("arm_l")}), set()),
             (dict(touch={k_: 0.0 for k_ in jo("arm_l")}), set()),
             (dict(pain=[0.0] * 43 + [1.0]), {"pain"}), (dict(pain=[0.0] * 5 + [1.0] + [0.0] * 38), {"pain"}), (dict(pain=[0.0] * 44), set()),
             (dict(face_fovea=[1.0]), {"face_fovea"}), (dict(face_fovea=[0.0]), set()),
             (dict(sound_side=[1.0, 0.4]), {"sound_l"}), (dict(sound_side=[1.0, -0.4]), {"sound_r"}),
             (dict(sound_side=[1.0, 0.0]), {"sound_l", "sound_r"}), (dict(sound_side=[0.0, 0.4]), set()),
             (dict(onset_periph=[1.0, -0.3, 0.1]), {"visual_l"}), (dict(onset_periph=[1.0, 0.3, 0.0]), {"visual_r"}),
             (dict(onset_periph=[1.0, 0.0, 0.2]), {"visual_l", "visual_r"}), (dict(onset_periph=[0.0, 0.3, 0.0]), set()),
             (dict(), set())]
    for obs_, want in cases:
        got = fire(**obs_)
        assert got == want, (obs_, got, want)
    # THE CHECK refuses bad declarations
    for bad in ([EventLine("x", "touch", (1,)), EventLine("x", "pain", (0,))], [EventLine("x", "touch", (1,), distal=("y",))],
                [EventLine("x", "sound_side", (0,), side=(1, 0.5))], [EventLine("x", "touch", ())], [EventLine("x", "", (0,))]):
        b_ = SimAnatomy(born_table(), SIM_CFG); b_.events = bad
        try:
            b_.check()
        except ValueError:
            pass
        else:
            raise AssertionError(f"the check took {bad}")
    # THE STRIATAL EXPANSION, live, by hand on every tick
    cfg = dict(SIM_CFG, wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.3, fast_rls=1, fast_input="striatum",
               stri_k=4, stri_m=64, actor=1, face_form="foresee", face_input="striatum", night_starts=16, night_rounds=1, night_batch=4,
               rem_dreams=2, rem_steps=2, night_dev="")
    counts = {}
    for quiet in (0, 1):
        c_ = dict(cfg, stri_quiet=quiet); w = _g1_events_world(); L = _g1(c_, w)
        m = L.m; k, V = 4, m.vocab; E = len(a.events)
        n_lang = k * (2 * V + 3); n_mot = k * sum(sum(e.factors) for e in L.anatomy.motors)
        assert m.stri_W.shape == (n_lang + n_mot + k * E, 64) and m.stri_eblock == (n_lang + n_mot, E) and m.stri_eline.tolist() == [-1] * k
        g = torch.Generator().manual_seed(0 + 7919)                          # the rows as the striatum's generator drew them
        ref = [torch.randn(n_lang, 64, generator=g) / math.sqrt(k)]; torch.rand(64, generator=g)
        for e in L.anatomy.motors:
            ref.append(torch.randn(k * sum(e.factors), 64, generator=g) / math.sqrt(k * len(e.factors)))
        ref.append(torch.randn(k * E, 64, generator=g) / math.sqrt(k))
        assert torch.equal(m.stri_W, torch.cat(ref)), "the striatum's rows are not born in order"
        run = WorldLoop(L); fired = pushed = empty = 0; seen = set()
        for t in range(120):
            prev = m.stri_eline.clone()
            run.step()
            evs = L._event_lines(); mask = sum(1 << i for i, v in enumerate(evs) if v)
            seen |= {names[i] for i, v in enumerate(evs) if v}
            if mask:
                fired += 1; assert int(m.stri_eline[0]) == mask and torch.equal(m.stri_eline[1:], prev[:-1]), t
            elif quiet:
                empty += 1; assert int(m.stri_eline[0]) == 0 and torch.equal(m.stri_eline[1:], prev[:-1]), t
            else:
                assert torch.equal(m.stri_eline, prev), t
            pushed += int(bool(mask) or bool(quiet))
            assert torch.equal(m.striatum_read(), _stri_ref(L)), t
        assert seen == {"touch_trunk", "touch_hand_l", "touch_leg_r", "pain", "face_fovea", "sound_l", "sound_r", "visual_l", "visual_r"}, seen   # the scripted
        counts[quiet] = (fired, empty)
        # REM's imagination gives the line back (it runs the dreams' events through the striatum, then restores every line)
        before = m.stri_eline.clone(); L._frel_corr = 1.0
        n_imag, _ = L._rem_imagine([5, 6, 7, 8, 9])
        assert n_imag > 0 and torch.equal(m.stri_eline, before), n_imag
        # a save keeps it
        fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
        try:
            L.save(path); w2 = _g1_events_world(); w2.load_state(w.save_state())
            g_ = torch.get_rng_state(); torch.manual_seed(0)
            B = Life.load(path, SimAnatomy(born_table(), L.cfg), save_path=None, world=w2); torch.set_rng_state(g_)
        finally:
            os.remove(path)
        assert torch.equal(B.m.stri_eline, before) and torch.equal(B.m.stri_W, m.stri_W) and B.m.stri_eblock == m.stri_eblock
        # the night empties it
        rep = L.night(); assert not rep.get("error"), rep.get("error")
        assert m.stri_eline.tolist() == [-1] * k
    # the language body: no event line in its organs, its striatum sized as the language block alone
    L = Life.birth(TOK, device="cpu", d=32, layers=1, heads=2, window=8, cfg=dict(fast_rls=1, fast_input="striatum", stri_k=4, stri_m=64), seed=0)
    assert "stri_eline" not in L.m._buffers and not hasattr(L.m, "stri_eblock") and L.m.stri_W.shape == (4 * (2 * L.m.vocab + 3), 64)
    assert not hasattr(L, "_events_now")
    print(f"frames 1: the G1's 13 event lines in the design's order, each by the born rule on {len(cases)} crafted frames (isolation: the",
          f"palm fires the hand alone, the base the trunk; the sides, both at no side); the check refused 5 bad declarations; the striatal",
          f"block after the effectors' ({E} lines x {k} positions at 1/sqrt(k), every row before born as it was); live 120 ticks, the fired",
          f"lines one event a tick ({counts[0][0]} ticks; under stri_quiet also {counts[1][1]} empty events), the expansion by hand on every",
          f"tick, every scripted line seen; the night empties it, REM gives it back, a save keeps it; the language body holds none")


# ---------------- frames 2: the frames' surprise, the event's end, the gated writes, the tick's record (R7b) ----------------

def _codes_by_hand(L, u):
    """each channel's code of the tick's frame by hand (the words' lexicon row; a vector channel's born code of its observation)"""
    m = L.m; out = {}
    with torch.no_grad():
        for c in L.anatomy.channels:
            if c.kind == "symbol":
                out[c.name] = m.E.weight[int(u)].clone()
            else:
                o = torch.as_tensor(L.world.now.obs.get(c.name, [0.0] * int(c.size)), dtype=torch.float32)
                out[c.name] = (o @ m.encs[c.name].rows) / math.sqrt(float(c.size))
    return out


def test_the_frames():
    """frames 2 (step R7b; SIM_DESIGN.md 7.4's "the fast memory's writes", 9's store, 10's "event end"; the switch `frames`, on in SIM_CFG):
    THE FRAME'S SURPRISE, by hand on every tick: each forecasting channel's error on the frame that came (the words' the tick's surprise,
    each later channel's head's squared error to the code that came, on the forecast made at the last tick's end), each over its own
    running mean (1 / min(n, 36,000), this tick's error in it), their mean; none after birth and after the night. THE EVENT'S END: today's
    settle law on it (4 and 64 ticks, settled at half), an end on the first settled tick after one that was not, replayed by hand over
    the day's record to the same ticks. THE WRITE: the gate's running 0.9 quantile in log units replayed by hand (its first 20 samples
    settling it, nothing written meanwhile) decides every write; each write's key is the stream's key of the tick before, its value the
    frame's codes summed, its strength surprise x (1 + |dopamine|), who 2; the frames written about a tenth of the ticks. THE MARKS: the
    last frame written before an end is an event's end (its boundary), the first after it an event's start. THE RECORD: one row a tick,
    (surprise, dopamine, 0, the reward felt), float32. THE NIGHT: an open event ends at nightfall; the record and its ends let go; the
    morning's first frame is a start; no frame is ever a words' dream's onset. The language body gains none of it"""
    from body.sim.anatomy import SIM_CFG
    assert SIM_CFG["frames"] == 1
    cfg = dict(SIM_CFG, wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.3, night_starts=16, night_rounds=1,
               night_batch=4, rem_dreams=2, rem_steps=2, night_dev="")
    w = _g1_events_world(burst=True); L = _g1(cfg, w); m = L.m
    run = WorldLoop(L)
    caught = []; fw = L._frame_write; ft = L._frame_tick

    def spy_write(key, value, strength, fw=fw, caught=caught, L=L):
        out = fw(key, value, strength); caught.append((L.ticks, key.clone(), value.clone(), float(strength), out)); return out

    def spy_tick(u, delta, r, ft=ft, L=L):
        L._probe = (int(u), float(delta), float(r)); return ft(u, delta, r)
    L._frame_write = spy_write; L._frame_tick = spy_tick
    T = 400; mus = {}; hand_s = []; keys_prev = []; probes = []; tots = []
    for t in range(T):
        ffc = {k: v.clone() for k, v in (L._ffc or {}).items()} if getattr(L, "_ffc", None) else None
        key_before = L._fkey_prev.clone() if getattr(L, "_fkey_prev", None) is not None else None
        had_pred = L.pred_prev is not None
        run.step()
        u, delta, r = L._probe; probes.append(L._probe); keys_prev.append(key_before)
        codes = _codes_by_hand(L, u); tot = None
        for c in L.anatomy.channels:
            tot = codes[c.name] if tot is None else tot + codes[c.name]
        tots.append(tot)
        errs = {}
        if had_pred:
            errs["words"] = float(L._surp_tick)
        if ffc:
            for name, pr in ffc.items():
                errs[name] = float(0.5 * ((pr.float() - codes[name].float()) ** 2).sum())
        vals = []
        for c in L.anatomy.channels:
            if c.name in errs:
                n_, mu = mus.get(c.name, [0, 0.0]); n_ += 1; mu = mu + (errs[c.name] - mu) / min(float(n_), 36000.0); mus[c.name] = [n_, mu]
                vals.append(errs[c.name] / mu)
        s_ = sum(vals) / len(vals) if vals else None
        hand_s.append(s_)
        rec = L._rec[t]
        assert (float(rec[0]) == float(torch.tensor(s_ if s_ is not None else 0.0, dtype=torch.float32))), (t, float(rec[0]), s_)
        assert float(rec[1]) == float(torch.tensor(delta, dtype=torch.float32)) and float(rec[2]) == 0.0 and float(rec[3]) == float(torch.tensor(r, dtype=torch.float32)), t
    assert L._rec_n == T and hand_s[0] is None and all(x is not None for x in hand_s[1:]), hand_s[:3]
    assert set(mus) == {c.name for c in L.anatomy.channels} and all(abs(L._ferr[k][1] - v[1]) < 1e-12 and L._ferr[k][0] == v[0] for k, v in mus.items())
    # the settle law replayed over the day
    fast = slow = None; prev = True; ends = []
    for t, s_ in enumerate(hand_s):
        if s_ is None:
            continue
        fast = s_ if fast is None else 0.75 * fast + 0.25 * s_; slow = s_ if slow is None else (1 - 1 / 64) * slow + (1 / 64) * s_
        st_ = fast <= 0.5 * max(1e-6, slow)
        if st_ and not prev:
            ends.append(t)
        prev = st_
    assert ends == L._rec_ends and len(ends) >= 3, (ends, L._rec_ends)
    # the write gate replayed
    q = None; warm = []; writes = []
    for t, s_ in enumerate(hand_s):
        if s_ is None or keys_prev[t] is None:
            continue
        lx = math.log(s_)
        if q is None:
            warm.append(lx)
            if len(warm) >= 20:
                q = L._pace_q(warm, 0.9)
            continue
        if lx > q:
            writes.append(t)
        q = q + 0.05 * (0.9 - (1.0 if lx <= q else 0.0))
    assert [c_[0] for c_ in caught] == writes and 0.05 * T <= len(writes) <= 0.2 * T, (len(writes), [c_[0] for c_ in caught][:5], writes[:5])
    for (t, key, value, strength, kept) in caught:
        u, delta, r = probes[t]
        assert torch.equal(key, keys_prev[t]) and torch.equal(value, tots[t]) and kept, t
        assert abs(strength - hand_s[t] * (1.0 + abs(delta))) < 1e-9 * max(1.0, strength), (t, strength)
    fr = (L.store.W == 2); n_fr = int(fr.sum())
    assert n_fr >= 1 and n_fr <= len(writes) and int((L.store.W != 2).sum()) == L.store.n() - n_fr
    # the marks: the last write before each end is an event's end, the first after each end an event's start
    starts_ok = ends_ok = 0
    wt = [c_[0] for c_ in caught]
    for e_ in ends:
        before = [i for i, t in enumerate(wt) if t < e_]; after = [i for i, t in enumerate(wt) if t >= e_]
        if before:
            j = L.store._find(caught[before[-1]][1], caught[before[-1]][2]); assert j >= 0 and bool(L.store.B[j]); ends_ok += 1
        if after:
            j = L.store._find(caught[after[0]][1], caught[after[0]][2]); assert j >= 0 and bool(L.store.Bs[j]); starts_ok += 1
    assert ends_ok >= 2 and starts_ok >= 2, (ends_ok, starts_ok)
    # the night: an open event ends at nightfall, the record and its ends let go, the morning's first frame a start, no frame an onset
    L._fs_settled = False; n_end = len(L._rec_ends); last = L._flast_write
    views = []; wsf = L._words_store; n_words = int((L.store.W != 2).sum())

    def spy_ws(wsf=wsf, views=views):
        v_ = wsf(); views.append(v_); return v_
    L._words_store = spy_ws
    rep = L.night(); assert not rep.get("error"), rep.get("error")
    assert last is None or L.store._find(*last) < 0 or bool(L.store.B[L.store._find(*last)])
    assert (L._rec, L._rec_n, L._rec_ends, L._ffc, L._fkey_prev, L._fstart_armed) == (None, 0, [], None, None, True)
    assert len(views) == 1 and views[0].n() == n_words and not bool((views[0].W == 2).any()), (len(views), views[0].n(), n_words)
    del L._words_store
    n_caught = len(caught)
    for _ in range(80):
        run.step()
    morning = caught[n_caught:]
    assert morning and bool(L.store.Bs[L.store._find(morning[0][1], morning[0][2])]) and L._rec_n == 80
    # the language body: nothing of it
    Lg = Life.birth(TOK, device="cpu", d=32, layers=1, heads=2, window=8, cfg={}, seed=0)
    for _ in range(40):
        Lg.tick()
    assert not any(k in vars(Lg) for k in ("_ferr", "_rec", "_rec_n", "_fkey_prev", "_ffc", "_fw_err", "_fs_fast", "_fq")) and "frames" not in Lg.cfg
    print(f"frames 2: {T} ticks of the G1 in frames: the frame's surprise by hand on every tick (9 channels, each over its running mean),",
          f"the record's rows (surprise, dopamine, 0, reward) float32; the settle law replayed to the same {len(ends)} ends; the gate's",
          f"running 0.9 quantile replayed to the same {len(writes)} writes ({len(writes) / T:.3f} of the ticks), each under the key of the",
          f"stream before it with the frame's codes at surprise x (1 + |dopamine|), {n_fr} frame slots (who 2); {ends_ok} ends and",
          f"{starts_ok} starts marked; the night ended the open event, let the record go, dreamt the words from the store's {n_words} words",
          f"alone (no frame), and the morning's first frame was a start; the language body holds none of it")


FRAME_TESTS = [test_the_event_lines, test_the_frames]


if __name__ == "__main__":
    t0 = time.time(); failed = 0
    for t in FRAME_TESTS:
        try:
            t()
        except AssertionError as e:
            failed += 1; print("FAIL", t.__name__, ":", e)
        except Exception as e:
            failed += 1; print("ERROR", t.__name__, ":", type(e).__name__, str(e)[:300])
    print(f"{len(FRAME_TESTS) - failed}/{len(FRAME_TESTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if failed else 0)
