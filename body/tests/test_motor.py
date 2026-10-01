"""the motor effectors of the G1's design (docs/SIM_DESIGN.md 3.5-3.7, 7.3, 8's R6h row, A41, A43, A47, A48, C38, C53, C54, C61; the core
refactor's step R6h). Run: python3 -m body.tests.test_motor (the organ tests run these too).

What must hold: the voice (the lexicon's effector) may stand at any place among the effectors, so a body numbers its effectors as its
design does, and the place changes nothing of its life but the numbering (motor 1); the gate's intrinsic term, the performance error,
reaches only the gate of the effector that declares it, per joint for a motor effector, and never a voice that declares none (motor 1);
a movement unit holds its act unless the choice passes the persistence margin, and the born units' lengths are the gate's continuation
draw's law (motor 2, C38 at the core); act_inv's reliability takes each label's chance from the act's own choice (motor 3); act_inv's
lessons are batched (motor 4); each motor effector's fatigue is its own, and its forward error reaches its gate (motor 5); the spinal
pattern generator (C54: a movement of 2 + 3 ticks, then a pause drawn from the seed; the legs one rhythm) and the born cry are summed
below the gate, each by its rule (motors 6 and 7); a vector channel's born code is fixed from the body's seed (motor 8); the born
orienting bias, its gate input and the VOR's constants (motor 9); the G1's anatomy numbers its effectors as the design does and its
performance error lands on the tract's gate alone (motor 11); a motor body saved anywhere (inside a day, at a night's boundary) goes on
as the life that went on, bit for bit, its day saved whole (motor 12, A70). Each is a switch or a declaration the language body does
not hold: it has none of it (the eight pinned digests are the guard's, tools/pins/digests.txt). The stub worlds here are instruments
of these tests, not the G1's world."""
import collections
import math
import os
import pickle
import statistics
import sys
import tempfile
import time

import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)   # this tree's body, not a fixed one
from body.life import Life  # noqa: E402
from body.core.anatomy import Channel, Effector, LanguageAnatomy, VoiceEffector  # noqa: E402
from body.core.world import Frame, SimWorld, WorldLoop  # noqa: E402

TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")


def _born(anatomy, cfg, world, seed=0, d=32):
    torch.manual_seed(seed)
    return Life.birth(anatomy, device="cpu", d=d, layers=2, heads=2, window=16, cfg=cfg, seed=seed, world=world)


# ---------------- motor 1: the voice's place and the performance error (C61) ----------------

class _TractBody(LanguageAnatomy):
    """the diary's words and face; two senses of the world's frames (the tract's "ears", the arm's "body": two numbers each, through the
    face's map); a vocal tract of three articulators of five settings (its rest 62, all held) sensing its ears, with act_inv and the
    intrinsic term; the words' silent output, the voice of the code (named "words", declaring no intrinsic term unless `words_int`);
    an arm of two joints (its rest 12). `tract_first`: the tract is effector 0 and the words effector 1, as the G1's design numbers
    them (SIM_DESIGN.md 3.5); else the words first, as a language body's voice is"""

    def __init__(self, tok, cfg=None, tract_first=True, words_int=False):
        super().__init__(tok, cfg)
        self.channels += [Channel("ears", "vector", 2, organ="face_in"), Channel("body", "vector", 2, organ="face_in")]
        v = self.effectors[0]; v.name = "words"; v.intrinsic = bool(words_int)
        tract = Effector("tract", [5, 5, 5], rest_id=62, effort=0.12, sense="ears", inverse=True, inv_hidden=16, intrinsic=True)
        arm = Effector("arm", [5, 5], rest_id=12, effort=0.05, sense="body")
        self.effectors = [tract, v, arm] if tract_first else [v, tract, arm]


def _tract_world():
    """a stub of the simulated world for motor 1: the tract's act heard on its ears (the mean size of its steps, the step of its first
    articulator), the arm's joints' steps on its body, the parent's short line on the words now and then, a smile now and then"""

    class TractWorld(SimWorld):
        LINE = "ma ma "

        def __init__(self):
            self.t = 0; self.ears = [0.0, 0.0]; self.body = [0.0, 0.0]; self.applied = []

        def frame(self):
            obs = {"ears": list(self.ears), "body": list(self.body)}
            k = self.t % 30 - 4
            if 0 <= k < len(self.LINE):
                obs["ear"] = TOK.token_to_id(self.LINE[k])
            return Frame(self.t, obs, 2.0 if self.t % 25 == 12 else 0.0, {"who": "parent"})

        def apply(self, acts):
            self.applied.append((list(acts), dict(acts)))
            a = int(acts.get("tract", 62)); dg = [a // 25, (a // 5) % 5, a % 5]
            self.ears = [sum(abs(x - 2) for x in dg) / 6.0, 0.3 * (dg[0] - 2)]
            b = int(acts.get("arm", 12)); self.body = [0.1 * (b // 5 - 2), 0.1 * (b % 5 - 2)]
            self.t += 1

        def pause(self):
            pass

        def resume(self):
            pass

        def save_state(self):
            return pickle.dumps((self.t, self.ears, self.body))

        def load_state(self, blob):
            self.t, self.ears, self.body = pickle.loads(blob)
    return TractWorld()


_CFG1 = dict(wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.4, gate_int=0.5, gate_int_form="error",
             fast_rls=1, fast_input="striatum", actor=1, stri_k=4, stri_m=64, chunk_gate=1)


def test_the_voice_at_any_place():
    """motor 1 (step R6h; SIM_DESIGN.md 3.5, A41, C61): THE NUMBERING. A body may declare its vocal tract effector 0 and the words'
    silent output (the voice of the code, the lexicon's effector) effector 1, as the G1's design numbers them: the check takes it, the
    voice is found at its place and every other effector is a motor effector in the declared order. The place changes nothing but the
    numbering: the same body declared with the words first lives the same 150 ticks bit for bit (every organ, the store, the optimizers,
    the random streams, every working attribute), each tick's acts equal and listed in the declared order. THE PERFORMANCE ERROR (A41):
    on every tick the tract acted its row in its gate's buffer carries the mean over its articulators of the belief its choice gave the
    setting made less that setting's running mean (checked against the law by hand, its 15 means), and it enters the tract's
    gate's credit alone: the arm's rows carry 0, and the words' output, which declares no intrinsic term, carries 0 on every tick and
    keeps no table of its own. Declared on the words (as the diary's voice declares it), the words' rows carry the voice's own term and
    the tract's are unchanged. A motor effector's term under another form than the error is refused; its means go through a save"""
    from body.tests.test_anatomy import _whole
    a1 = _TractBody(TOK, _CFG1); a1.check()
    assert a1.voice_at == 1 and a1.voice.name == "words" and [e.name for e in a1.motors] == ["tract", "arm"]
    assert [e.name for e in a1.effectors] == ["tract", "words", "arm"] and a1.effectors[0].intrinsic and not a1.voice.intrinsic
    lives, rec = {}, {}
    for tf in (True, False):
        w = _tract_world(); L = _born(_TractBody(TOK, _CFG1, tract_first=tf), _CFG1, w)
        assert [e.name for e in L.anatomy.motors] == ["tract", "arm"] and list(L.m.acts) == ["tract", "arm"]
        rows = []; f = L._act_effectors

        def spy(u, stri, gam, tick_tr, f=f, rows=rows, L=L):
            out = f(u, stri, gam, tick_tr)
            now = L.motor[0]["now"]
            rows.append((bool(now["acted"]), [p_.clone() for p_ in now["probs"]], list(now["digits"]), float(now["int"]),
                         float(L.motor[1]["now"]["int"])))
            return out
        L._act_effectors = spy
        run = WorldLoop(L); vrows = []
        for _ in range(150):
            run.step()
            vrows.append((L.gate_buf[-1][3], L.motor[0]["buf"][-1][3], L.motor[1]["buf"][-1][3]))
        lives[tf] = (L, w); rec[tf] = (rows, vrows)
    (A, wa), (B, wb) = lives[True], lives[False]
    del A._act_effectors, B._act_effectors
    assert _whole(A) == _whole(B), [k for k in _whole(A) if _whole(A)[k] != _whole(B)[k]]
    assert [x[1] for x in wa.applied] == [x[1] for x in wb.applied]
    assert all(x[0] == ["tract", "words", "arm"] for x in wa.applied) and all(x[0] == ["words", "tract", "arm"] for x in wb.applied)
    # the performance error by hand
    rows, vrows = rec[True]
    hab = float(A.cfg["gate_habit"]); ref = [[0.0] * 5 for _ in range(3)]; n_act = 0; n_nz = 0
    for (acted, probs, dig, it, iarm), (iv, itr, ia) in zip(rows, vrows):
        want = 0.0
        if acted:
            errs = []
            for j, (p_, a_) in enumerate(zip(probs, dig)):
                b_ = float(p_[a_]); errs.append(b_ - ref[j][a_]); ref[j][a_] += (1.0 - hab) * (b_ - ref[j][a_])
            want = sum(errs) / len(errs); n_act += 1; n_nz += int(want != 0.0)
        assert abs(it - want) < 1e-12 and itr == it, (it, want, itr)
        assert iarm == 0.0 and ia == 0.0 and iv == 0.0, (iarm, ia, iv)
    assert n_act >= 20 and n_nz >= 20, (n_act, n_nz)
    assert all(abs(A.motor[0]["perf"][j][k] - ref[j][k]) < 1e-12 for j in range(3) for k in range(5)) and A.motor[1]["perf"] is None
    assert A.perf == {} and A.sym_freq == {}, (A.perf, A.sym_freq)
    n_said = sum(1 for p_ in A.page if p_[1] == 1 and p_[0])
    # the words declaring the term (as the diary's voice does): the voice's rows carry its own, the tract's unchanged
    w = _tract_world(); C = _born(_TractBody(TOK, _CFG1, words_int=True), _CFG1, w); run = WorldLoop(C); cv = []
    for _ in range(150):
        run.step(); cv.append((C.gate_buf[-1][3], C.motor[0]["buf"][-1][3]))
    assert sum(1 for iv, _ in cv if iv != 0.0) >= 5 and C.perf, (sum(1 for iv, _ in cv if iv != 0.0), n_said)
    # refused: a motor effector's term under the value form
    try:
        _born(_TractBody(TOK, dict(_CFG1, gate_int_form="value")), dict(_CFG1, gate_int_form="value"), _tract_world())
    except ValueError as e_:
        assert "performance error" in str(e_)
    else:
        raise AssertionError("a motor effector's intrinsic term was taken under the value form")
    # its means go through a save
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    try:
        A.save(path)
        D = Life.load(path, _TractBody(TOK, A.cfg), save_path=None, world=_tract_world())
    finally:
        os.remove(path)
    assert D.motor[0]["perf"] == A.motor[0]["perf"] and D.motor[1]["perf"] is None
    print(f"motor 1: the tract effector 0 and the words 1 (the voice at 1, the motors tract and arm): the same 150 ticks bit for bit as the",
          f"words first, each tick's acts equal, in the declared order; the tract acted {n_act} ticks, its performance error the law's on",
          f"each ({n_nz} not zero) and in its gate's rows alone; the words' rows 0 on all 150; declared on the words, their rows carry it",
          f"({sum(1 for iv, _ in cv if iv != 0.0)} ticks); a motor term under the value form refused; its means saved")


# ---------------- motor 2-5: movement units, the kappa correction, act_inv batched, fatigue per effector, the forward error ----------------

class _Limbs(LanguageAnatomy):
    """the diary's words and face; two senses of the world's frames through the face's map (the "body": a limb's two joints'
    velocities; the "touch": a grip's velocity and a 0); a limb of two joints of five settings (its rest 12) with act_inv and (fwd) its
    forward error among its gate's inputs; a grip of one joint of three settings (its rest 1) sensing its touch"""

    def __init__(self, tok, cfg=None, fwd=False):
        super().__init__(tok, cfg)
        self.channels += [Channel("body", "vector", 2, organ="face_in"), Channel("touch", "vector", 2, organ="face_in")]
        self.effectors += [Effector("limb", [5, 5], rest_id=12, effort=0.05, sense="body", inverse=True, inv_hidden=16,
                                    fwd_gate=bool(fwd), n_in=2 if fwd else 1),
                           Effector("grip", [3], rest_id=1, effort=0.03, sense="touch", sense_idx=[0])]


_STEP5 = (-0.27, -0.09, 0.0, 0.09, 0.27)


def _limb_world(guide_every=0):
    """a stub world for motor 2-5: each joint's velocity is the step its setting in the tick's act makes (rad a tick over 0.15 s); the
    parent's hand moves the resting limb now and then (every `guide_every` ticks, one act of a fixed cycle: a demonstration); a short
    line on the words now and then"""

    class LimbWorld(SimWorld):
        LINE = "up we go "

        def __init__(self):
            self.t = 0; self.v = [0.0] * 3; self.applied = []; self.k = 0

        def frame(self):
            obs = {"body": list(self.v[:2]), "touch": [self.v[2], 0.0]}
            k = self.t % 40 - 5
            if 0 <= k < len(self.LINE):
                obs["ear"] = TOK.token_to_id(self.LINE[k])
            return Frame(self.t, obs, 2.0 if self.t % 33 == 20 else 0.0, {"who": "parent"})

        def apply(self, acts):
            self.applied.append(dict(acts))
            a = int(acts.get("limb", 12))
            if a == 12 and guide_every and self.t % guide_every == 0:
                a = (7, 18, 3, 22)[self.k % 4]; self.k += 1
            dg = [a // 5, a % 5]
            g = int(acts.get("grip", 1))
            self.v = [_STEP5[x] / 0.15 for x in dg] + [0.5 * (g - 1)]
            self.t += 1

        def pause(self):
            pass

        def resume(self):
            pass

        def save_state(self):
            return pickle.dumps((self.t, self.v, self.k))

        def load_state(self, blob):
            self.t, self.v, self.k = pickle.loads(blob)
    return LimbWorld()


def _born_limbs(cfg, world, fwd=False, seed=0):
    return _born(_Limbs(TOK, cfg, fwd=fwd), cfg, world, seed=seed)


_LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0,
            act_inv_lr=0.0)


def test_movement_units():
    """motor 2 (step R6h; SIM_DESIGN.md 3.6, 10, C38): THE MOVEMENT UNIT. Under chunk_gate and unit_margin a unit under way holds its act
    joint by joint, a joint taking another setting only where the choice's logits prefer it to the held setting by more than the margin
    (log 4): checked on every continuation of a body whose act_pred is made strong (its weights x 300), where holds and switches both
    occur, each by the rule. Absent (R6), the continuation is act_pred's best guess, the choice's argmax. THE BORN UNITS (C38's
    measurement at the core, every learning rate 0): at birth the gate's p_act is the floor plus (1 - floor) x birth_act, 0.2875, and
    the born proposal never passes the margin, so a unit holds its first act whole until its gate's own draw closes it or chunk_max
    (8) ends it: the lengths are geometric, P(length 1) = 1 - 0.2875, mean (1 - 0.2875^8) / (1 - 0.2875) = 1.40 ticks (0.21 s), the
    units measured over 6000 ticks inside their binomial error; every continuation held its unit's act"""
    import math as _m
    base = dict(_LR0, wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, fast_rls=0, chunk_gate=1, chunk_max=8,
                unit_margin=_m.log(4.0))
    # the rule, where holds and switches both occur
    L = _born_limbs(dict(base, gate_floor=0.8), _limb_world())
    with torch.no_grad():
        L.m.timing["limb"].pred.weight.mul_(300.0)
    rec = []; f = L._choose_effector

    def spy(i, frame, C1, level, stri, f=f, L=L, rec=rec):
        st_ = L.motor[i - 1]; held = list(st_["unit"]) if st_["unit"] is not None else None
        out = f(i, frame, C1, level, stri)
        now = st_["now"]
        if i == 1 and now["cont"] and now["drew"]:
            with torch.no_grad():
                lg = L.m.acts["limb"].logits(L.m.timing["limb"].pred(C1) + (L.m.timing["limb"].cor(st_["err"]) if st_["err"] is not None else 0.0),
                                             float(st_["now"]["sharp"]))          # A97: read at the limb's earned sharpness that tick
            rec.append((held, [x_.clone() for x_ in lg], list(now["digits"])))
        return out
    L._choose_effector = spy
    run = WorldLoop(L)
    for _ in range(400):
        run.step()
    holds = switches = 0
    for held, lg, dig in rec:
        for j, (l_, h_, a_) in enumerate(zip(lg, held, dig)):
            b_ = int(l_.argmax()); want = b_ if float(l_[b_]) - float(l_[h_]) > _m.log(4.0) else h_
            assert a_ == want, (j, h_, a_, want, l_)
            holds += int(a_ == h_); switches += int(a_ != h_)
    assert len(rec) >= 40 and holds >= 20 and switches >= 20, (len(rec), holds, switches)
    # absent: R6's best guess
    B = _born_limbs(dict(base, gate_floor=0.8, unit_margin=None), _limb_world())
    with torch.no_grad():
        B.m.timing["limb"].pred.weight.mul_(300.0)
    rb = []; fb = B._choose_effector

    def spyb(i, frame, C1, level, stri, f=fb, L=B, rec=rb):
        out = f(i, frame, C1, level, stri); now = L.motor[i - 1]["now"]
        if i == 1 and now["cont"] and now["drew"]:
            rec.append((list(now["digits"]), L._best_guess(L.anatomy.motors[0], L.anatomy.motors[0].propose(L, C1))))
        return out
    B._choose_effector = spyb
    run = WorldLoop(B)
    for _ in range(200):
        run.step()
    assert rb and all(B.m.acts["limb"].flat(d_) == g_ for d_, g_ in rb), rb[:3]
    # the born units (C38 at the core): every learning rate 0, the gate at birth
    C = _born_limbs(dict(base), _limb_world()); run = WorldLoop(C)
    p_act = float(C.cfg["gate_floor"]) + (1.0 - float(C.cfg["gate_floor"])) * float(C.cfg["birth_act"])
    lens = []; cur = 0; held_all = True; prev_dig = None
    for _ in range(6000):
        run.step(); now = C.motor[0]["now"]
        assert abs(now["p_act"] - p_act) < 1e-6, now["p_act"]
        if now["acted"]:
            if now["cont"]:
                held_all &= now["digits"] == prev_dig; cur += 1
            else:
                if cur:
                    lens.append(cur)
                cur = 1
            prev_dig = list(now["digits"])
        else:
            if cur:
                lens.append(cur)
            cur = 0
    n = len(lens); mean = sum(lens) / n; p1 = sum(1 for x in lens if x == 1) / n
    want_mean = (1.0 - p_act ** 8) / (1.0 - p_act); want_p1 = 1.0 - p_act
    sd_p1 = _m.sqrt(want_p1 * (1 - want_p1) / n)
    assert held_all and n >= 400 and abs(p1 - want_p1) < 4 * sd_p1 and abs(mean - want_mean) < 0.08, (n, mean, want_mean, p1, want_p1)
    assert max(lens) <= 8
    print(f"motor 2: the movement unit's rule on {len(rec)} continuations of a strong proposal ({holds} joints held, {switches} switched past",
          f"log 4); absent, R6's best guess on {len(rb)}; the born units (p_act {p_act:.4f}): {n} units over 6000 ticks, mean",
          f"{mean:.3f} ticks (the law {want_mean:.3f}), {p1:.3f} of length 1 (the law {want_p1:.3f}), the longest {max(lens)}, every",
          "continuation holding its unit's act")


def test_the_kappa_correction():
    """motor 3 (step R6h; R6's verifiers, 6d6d246's proposal, 8's R6h row): act_inv's reliability with each label's chance taken from the
    act's own choice. A probe of one joint of five, the acts drawn at known rates whose mode flips (hold 80% <-> +big 80%, every 500
    acts, the draw's own probabilities known before each act): a blind label that names the regime's mode reads 0.26-0.65 under R6's
    pooled kappa (the verifier's finding) and near 0 under the correction; a label right 90% of the time reads high under both; an act
    chosen without a draw (a held unit's) adds its chance 1 where the label is right and 0 where it is wrong, so it shows no skill when
    right. In a living body under act_inv_chance the reliability is the corrected kappa, the pooled one kept beside it"""
    import random as _r
    from body.core.timing import TimingMixin

    class _Probe(TimingMixin):
        def __init__(self, chance):
            self.cfg = dict(act_inv_tau=8192, act_inv_chance=int(chance))
    e = Effector("j", [5], rest_id=2)
    out = {}
    for kind in ("blind", "skilled"):
        for chance in (0, 1):
            P = _Probe(chance); st = {"inv_conf": None}; rng = _r.Random(3)
            for t in range(6000):
                mode = 2 if (t // 500) % 2 == 0 else 4
                probs = [0.0] * 5; probs[mode] = 0.8; other = 4 if mode == 2 else 2; probs[other] = 0.2
                a_ = mode if rng.random() < 0.8 else other
                if kind == "blind":
                    lab = mode
                else:
                    lab = a_ if rng.random() < 0.9 else rng.choice([k for k in range(5) if k != a_])
                P._inv_rel_update(e, st, [lab], [a_], chance=[torch.tensor(probs)], held=False)   # the draw's own probabilities
            out[(kind, chance)] = st["inv_gain"]
    assert 0.2 < out[("blind", 0)] and abs(out[("blind", 1)]) < 0.03 and out[("skilled", 0)] > 0.6 and out[("skilled", 1)] > 0.6, out
    # the chance of a drawn act, given that the draw was not the rest (`_act_chance`), against the joint law enumerated
    L0 = _born_limbs(dict(_LR0, wake_ticks=100000, fast_rls=0), _limb_world())
    p0, p1 = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1]), torch.tensor([0.05, 0.15, 0.6, 0.1, 0.1])
    q = L0._act_chance(L0.anatomy.motors[0], {"cont": False, "probs": [p0, p1]})
    for j_, qj in enumerate(q):
        for s_ in range(5):
            num = sum(float(p0[a0] * p1[a1]) for a0 in range(5) for a1 in range(5) if (a0, a1) != (2, 2) and (a0, a1)[j_] == s_)
            assert abs(float(qj[s_]) - num / (1.0 - 0.24)) < 1e-6, (j_, s_, float(qj[s_]), num / 0.76)
    assert L0._act_chance(L0.anatomy.motors[0], {"cont": True, "probs": [p0, p1]}) is None
    # held acts: chance 1 where the label is right, 0 where wrong
    P = _Probe(1); st = {"inv_conf": None}
    for t in range(200):
        P._inv_rel_update(e, st, [3], [3], chance=None, held=True)
    assert st["inv_ch"][0][1] == st["inv_ch"][0][2] and st["inv_gain"] == 0.0, st["inv_ch"]
    # in a living body: the corrected kappa is the reliability, the pooled kept beside it
    L = _born_limbs(dict(wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, fast_rls=0, gate_floor=0.6, act_inv_chance=1,
                         act_inv_lr=3e-3), _limb_world()); run = WorldLoop(L)
    for _ in range(400):
        run.step()
    st = L.motor[0]
    assert st["inv_ch"] is not None and st["inv_ch"][0][0] > 64 and "inv_kappa_pooled" in st and st["inv_gain"] > 0.1, (st["inv_gain"], st.get("inv_kappa_pooled"))
    # (a sanity floor: 0.2 until A104, when the readout stayed soft below "fair" agreement, the limb's acts varied more and its kappa
    # after 400 ticks read 0.198; the floor is not a claim about the limb's learning rate)
    print(f"motor 3: the kappa correction: a blind label on the regime's mode reads {out[('blind', 0)]:.3f} pooled and",
          f"{out[('blind', 1)]:.3f} corrected; a label right 90% reads {out[('skilled', 0)]:.3f} and {out[('skilled', 1)]:.3f}; held acts",
          f"show no skill when right; a living limb's reliability {st['inv_gain']:.3f} corrected (pooled kappas {[round(k, 3) for k in st['inv_kappa_pooled']]})")


def test_act_inv_batched():
    """motor 4 (step R6h; SIM_DESIGN.md 3.6: act_inv's lessons batched every 8 ticks; unbatched they cost 4-6 ms a tick at the humanoid's
    size): with act_inv_every 8 act_inv steps only on ticks divisible by 8, once, on the pairs gathered since (each the sense before and
    after an own act and the act), labelled by act_inv as it stood before the step, the step the mean of the joints' summed
    cross-entropy (checked against a copy stepped by hand); a batch of one is R6's lesson to the bit; the pairs gathered at dusk are
    learned at nightfall; pairs pending at a save come back with the load"""
    import copy as _c
    cfg = dict(wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, fast_rls=0, gate_floor=0.6, act_inv_every=8)
    L = _born_limbs(cfg, _limb_world()); run = WorldLoop(L)
    steps = []; fb = L._inverse_batch

    def spy(i, f=fb, L=L):
        st_ = L.motor[i - 1]; pairs = list(st_["inv_batch"])
        inv0 = _c.deepcopy(L.m.timing["limb"].inv); opt0 = _c.deepcopy(L.opt_inv.state_dict())
        f(i); steps.append((L.ticks, len(pairs), pairs, inv0, opt0, _c.deepcopy(L.m.timing["limb"].inv)))
    L._inverse_batch = spy
    for _ in range(120):
        run.step()
    assert steps and all(t_ % 8 == 0 for t_, *_ in steps) and all(1 <= n_ <= 8 for _, n_, *_ in steps), [(t_, n_) for t_, n_, *_ in steps]
    t_, n_, pairs, inv0, opt0, inv1 = steps[3]
    with torch.no_grad():
        want = _c.deepcopy(inv0)
    o = torch.optim.Adam(want.parameters(), lr=float(L.cfg.get("act_inv_lr", 1e-3))); o.load_state_dict(opt0)
    S0 = torch.stack([p_[0] for p_ in pairs]); S1 = torch.stack([p_[1] for p_ in pairs])
    tab = L.m.acts["limb"]; tr = torch.stack([tab.digits(torch.tensor(p_[2])) for p_ in pairs])
    z = want(torch.cat([S0, S1 - S0], -1)); lg = torch.split(z, [5, 5], -1)
    loss = sum(torch.nn.functional.cross_entropy(l_, tr[:, j_]) for j_, l_ in enumerate(lg)); o.zero_grad(); loss.backward(); o.step()
    assert all(torch.equal(a_, b_) for a_, b_ in zip(want.parameters(), inv1.parameters())), "the batch's step is not the mean lesson"
    # a batch of one is R6's lesson to the bit
    A1 = _born_limbs(dict(cfg, act_inv_every=1), _limb_world()); B1 = _born_limbs(dict(cfg, act_inv_every=1), _limb_world())
    for L_ in (A1, B1):
        r_ = WorldLoop(L_)
        for _ in range(30):
            r_.step()
    s0, s1 = torch.randn(2), torch.randn(2)
    A1._inverse_lesson(1, s0, s1, 18); B1.motor[0]["inv_batch"] = [(s0, s1, 18, None)]; B1._inverse_batch(1)
    assert all(torch.equal(a_, b_) for a_, b_ in zip(A1.m.timing["limb"].inv.parameters(), B1.m.timing["limb"].inv.parameters()))
    assert A1.motor[0]["inv_kappa"] == B1.motor[0]["inv_kappa"] and A1.motor[0]["inv_n"] == B1.motor[0]["inv_n"]
    # nightfall learns the pairs gathered; a save keeps pending pairs
    del L._inverse_batch
    L.motor[0]["inv_batch"] = [(torch.randn(2), torch.randn(2), 7, None), (torch.randn(2), torch.randn(2), 18, None)]
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    try:
        L.save(path); D = Life.load(path, _Limbs(TOK, L.cfg), save_path=None, world=_limb_world())
    finally:
        os.remove(path)
    assert [(torch.equal(a_[0], b_[0]), torch.equal(a_[1], b_[1]), a_[2] == b_[2]) for a_, b_ in zip(L.motor[0]["inv_batch"], D.motor[0]["inv_batch"])] == [(True, True, True)] * 2
    n0 = L.motor[0]["inv_n"]; L.night()
    assert L.motor[0]["inv_n"] == n0 + 2 and L.motor[0]["inv_batch"] == []
    print(f"motor 4: act_inv batched every 8 ticks: {len(steps)} steps in 120 ticks, each on ticks divisible by 8 over 2-8 pairs, the step",
          "the mean lesson by hand; a batch of one R6's lesson to the bit; pairs pending at dusk learned at nightfall, kept by a save")


def test_fatigue_per_effector_and_the_forward_error():
    """motor 5 (step R6h; SIM_DESIGN.md 3.5, 3.6, 10): FATIGUE PER EFFECTOR. Under own_fatigue each motor effector's acts' cost is its own
    fatigue: it rises by its cost on each act, recovers at the body's half-life, is what its gate reads (the feature fatigue / 10) and
    what its lesson's rows carry, rests at nightfall and goes through a save; the body's fatigue takes the voice's cost alone. Absent
    (R5), every cost joins the body's one fatigue. THE FORWARD ERROR INTO THE GATE: a limb that declares it reads, beside its own act last
    tick, the root mean square of its body sense less what its forward half foresaw (0 before the first foresight); a declaration whose
    inputs do not count it is refused as its gate is read"""
    cfg = dict(wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, fast_rls=0, gate_floor=0.6, own_fatigue=1)
    L = _born_limbs(cfg, _limb_world(), fwd=True); run = WorldLoop(L)
    hl = 0.5 ** (1.0 / float(L.cfg["fatigue_half_life"])); f_prev = [0.0, 0.0]; body_prev = 0.0; checked = 0
    for t in range(200):
        run.step()
        for j_, st_ in enumerate(L.motor):
            now = st_["now"]; want = f_prev[j_] * hl + (now["cost"] if (now["acted"] or now["reflex"]) else 0.0)
            assert abs(st_["fatigue"] - want) < 1e-12, (t, j_, st_["fatigue"], want)
            assert abs(float(now["feat"][32]) - f_prev[j_] * hl / 10.0) < 1e-6, (float(now["feat"][32]), f_prev[j_] * hl / 10.0)
            assert st_["buf"][-1][4] == st_["fatigue"]
            f_prev[j_] = st_["fatigue"]
        said = L.page[-1][0] != ""
        want_b = body_prev * hl + (float(L.cfg["symbol_cost"]) if L._acted_last else 0.0)
        assert abs(L.fatigue - want_b) < 1e-9, (t, L.fatigue, want_b)
        body_prev = L.fatigue
        # the forward error: the limb's gate input is the RMS of its error now
        st0 = L.motor[0]; err = st0["err"]
        want_e = 0.0 if err is None else float(err.float().pow(2).mean().sqrt())
        assert abs(float(st0["now"]["feat"][-1]) - want_e) < 1e-6 and st0["now"]["feat"].numel() == 32 + 5 + 2
        checked += int(err is not None and want_e > 0)
    assert min(f_prev) > 0 and checked >= 150, (f_prev, checked)
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    try:
        L.save(path); D = Life.load(path, _Limbs(TOK, L.cfg, fwd=True), save_path=None, world=_limb_world())
    finally:
        os.remove(path)
    assert [st_["fatigue"] for st_ in D.motor] == [st_["fatigue"] for st_ in L.motor]
    L.night(); assert all(st_["fatigue"] == 0.0 for st_ in L.motor)
    # absent: every cost joins the body's fatigue (R5)
    B = _born_limbs(dict(cfg, own_fatigue=0), _limb_world()); run = WorldLoop(B)
    for _ in range(60):
        run.step()
    assert all(st_["fatigue"] == 0.0 for st_ in B.motor) and B.fatigue > 0
    # a declaration that does not count the forward error is refused
    a = _Limbs(TOK, cfg, fwd=True); a.effectors[1].n_in = 1
    try:
        X = _born(a, cfg, _limb_world()); r_ = WorldLoop(X); r_.step(); r_.step()
    except ValueError as e_:
        assert "gate inputs" in str(e_)
    else:
        raise AssertionError("a forward error not counted in the gate's inputs was taken")
    print(f"motor 5: fatigue per effector over 200 ticks (limb {f_prev[0]:.3f}, grip {f_prev[1]:.3f}; the body's the voice's alone,",
          f"{L.fatigue if L.nights == 0 else body_prev:.3f}): its cost, its half-life, its gate's feature, its rows, its night and its save;",
          f"absent, the body's one fatigue; the forward error's RMS in the limb's gate on {checked} ticks; uncounted, refused")


# ---------------- motor 6-7: the born patterns summed at the cord (the spinal pattern generator, the born cry) ----------------

class _Kicker(LanguageAnatomy):
    """the diary's words and face; two legs of three joints (hip pitch, knee, ankle pitch: flexion senses -1, +1, -1, as the G1's
    withdrawal measured them), each with a spinal pattern generator, keeping one rhythm ("legs", the left leading it), born half a cycle
    apart; an arm of two joints (shoulder pitch, elbow: -1, -1) with one of its own, whose phase is drawn at birth from the body's seed"""

    def __init__(self, tok, cfg=None):
        super().__init__(tok, cfg)
        self.effectors += [Effector("leg_l", [5, 5, 5], rest_id=62, effort=0.05, spg={0: -1, 1: 1, 2: -1}, spg_phase=0.0, spg_rhythm="legs"),
                           Effector("leg_r", [5, 5, 5], rest_id=62, effort=0.05, spg={0: -1, 1: 1, 2: -1}, spg_phase=0.5, spg_rhythm="legs"),
                           Effector("arm", [5, 5], rest_id=12, effort=0.05, spg={0: -1, 1: -1})]


def _spg_law(seed, n_motor, r, n):
    """C54's cycle written again for the tests (body/core/cord.py `_spg_cycle`): cycle n of the rhythm led by motor effector r, in ticks
    of 0.15 s: the standard normal of the generator seeded seed + n_motor x n + r, through the log-normal of mean 3.56 s and SD 1.93 s,
    held to 1.0-8.5 s; with the unheld draw beside it"""
    z = float(torch.randn((), generator=torch.Generator().manual_seed(int(seed) + int(n_motor) * int(n) + int(r)), dtype=torch.float64))
    mean, sd = 3.56 / 0.15, 1.93 / 0.15
    s2 = math.log(1.0 + (sd / mean) ** 2)
    x = math.exp(math.log(mean) - s2 / 2.0 + math.sqrt(s2) * z)
    return min(max(x, 1.0 / 0.15), 8.5 / 0.15), x


def _spg_starts(seed, n_motor, r, lead, lag, T):
    """the ticks a limb's movements begin, from the rhythm's first cycle to past tick T: cycle n runs from s_n (s_0 = -lead x its length)
    and the limb's movement in it begins at the first tick at or after s_n + lag x its length; with each cycle (s_n, its length)"""
    starts, cycles, s, n = [], [], None, 0
    while True:
        Ln = _spg_law(seed, n_motor, r, n)[0]
        s = -lead * Ln if s is None else s
        starts.append(math.ceil(s + lag * Ln)); cycles.append((s, Ln))
        if starts[-1] > T:
            return starts, cycles
        s += Ln; n += 1


def _spg_place(starts, t):
    """+1 flexion (the movement's first 2 ticks), -1 extension (its next 3), 0 the pause: the rule at tick t, written again"""
    m = max([x for x in starts if x <= t], default=None)
    if m is None:
        return 0
    d = t - m
    return 1 if d < 2 else (-1 if d < 5 else 0)


def _cord_world(extra=None):
    """a stub world that records each tick's acts and what the body added below its gates (acts.cord); `extra` (a function of the tick)
    gives the frame's other observations"""

    class CordWorld(SimWorld):
        def __init__(self):
            self.t = 0; self.applied = []; self.cords = []; self.last = None

        def frame(self):
            obs = dict(extra(self) if extra else {})
            return Frame(self.t, obs, 0.0, {"who": "parent"})

        def apply(self, acts):
            self.applied.append(dict(acts)); self.cords.append(dict(getattr(acts, "cord", {}))); self.last = acts; self.t += 1

        def pause(self):
            pass

        def resume(self):
            pass

        def save_state(self):
            return pickle.dumps(self.t)

        def load_state(self, blob):
            self.t = pickle.loads(blob)
    return CordWorld()


def test_the_spinal_pattern_generator():
    """motor 6 (step R6h, C54 closed; SIM_DESIGN.md 3.6, 3.7, 10, A35, A48, C54): each limb's half-centre generator, summed at the cord
    with its own act, its cycle a short movement and a pause. On every tick of 400, for each limb, its place is the rule written again
    here (_spg_law, _spg_starts, _spg_place): its rhythm's cycles drawn from the organs' spg_seed (the generator seeded spg_seed + 3n + r
    for cycle n of the rhythm led by motor effector r), each a log-normal of mean 3.56 s and SD 1.93 s held to 1.0-8.5 s, in ticks of
    0.15 s; the limb's movement begins at the first tick at or after its cycle's start plus its lag; flexion its first 2 ticks,
    extension its next 3, then the pause. The step the world receives on each declared flexion joint is +A (its gate's p_act x 0.09 rad)
    in its flexion sense in the flexion, -2A/3 in the extension, nothing in the pause, summed with the own act whatever its sense (A97,
    2026-09-26: until A97 an own act against it cancelled it there, which let a constant act cancel every return and ratchet the joint
    to its limit), 0 on every other joint: a function of the tick, the rhythm and p_act alone (no posture, balance or gravity term: it
    reads no sense). THE CYCLE RETURNS WHERE IT BEGAN (the lead's decision of 2026-09-25: a kick is a flexion and a return): on every
    joint of every complete movement, the steps the world received sum to 0 (2A out, 2A back; the equal steps of R6h and C54 left -A, a
    drift toward extension every cycle). THE LEGS keep one rhythm (the left leading): in every cycle the right
    leg's movement begins half the cycle after the left's and before the left's next; THE ARM keeps its own, its phase drawn at birth
    from the body's seed. THE LAW OF THE DRAWS over
    20,000 cycles of one rhythm: every cycle inside 6.67-56.67 ticks, the held draws' mean and SD the clipped law's (23.425 and 11.613
    ticks: 3.514 and 1.742 s), the unheld the log-normal's own (3.56 and 1.93 s), about 1.2% and 2.5% held at the bounds. The rhythm is
    the tick's: a body saved and loaded stands where the one that lived on stands, tick for tick (its acts drawn afresh). The gate draws
    every tick and its rows' acts are its own; its ticks are counted as reflex. Refused: a rhythm named on a limb with no generator.
    Switched off, the world receives no cord"""
    cfg = dict(_LR0, wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, fast_rls=0, gate_floor=0.3, spg=1)
    w = _cord_world(); L = _born(_Kicker(TOK, cfg), cfg, w); run = WorldLoop(L)
    seed = int(L.m.spg_seed); ph = [0.0, 0.5, float(L.m.spg_phase[2])]
    assert 0 <= seed < 2 ** 31 and 0.0 <= ph[2] < 1.0 and L.m.spg_phase.dtype == torch.float64
    T = 400
    rhythm = {"leg_l": (0, 0.0, 0.0), "leg_r": (0, 0.0, 0.5), "arm": (2, ph[2], 0.0)}    # (its leader's place, the leader's phase, its lag)
    ref = {n_: _spg_starts(seed, 3, *rhythm[n_], T) for n_ in rhythm}
    n_cancel = n_sum = 0; places = collections.Counter(); steps = collections.defaultdict(list)
    for t in range(T):
        run.step()
        for e_, st_ in zip(L.anatomy.motors, L.motor):
            now = st_["now"]; want_place = _spg_place(ref[e_.name][0], L.ticks - 1)
            assert now["spg"] == want_place, (t, e_.name, now["spg"], want_place)
            places[(e_.name, want_place)] += 1
            A = float(now["p_act"]) * 0.09; want = [0.0] * len(e_.factors)
            for j_, sg_ in e_.spg.items():
                if not want_place:
                    continue
                own = (now["digits"][j_] > 2) - (now["digits"][j_] < 2); step = sg_ * (A if want_place > 0 else -A * 2.0 / 3.0)
                if own != 0 and (own > 0) != (step > 0):
                    n_cancel += 1                                       # A97: an own act against it no longer cancels it: it sums
                want[j_] = step; n_sum += int(own != 0)
            got = w.cords[-1].get(e_.name)
            assert (got is None and not any(want)) or (got is not None and all(abs(a_ - b_) < 1e-12 for a_, b_ in zip(got, want))), (t, e_.name, got, want)
            assert st_["buf"][-1][1] == now["acted"] and st_["buf"][-1][9] is False
            steps[e_.name].append((want_place, float(now["p_act"]), list(got) if got is not None else [0.0] * len(e_.factors)))
    # the cycle returns where it began: each complete movement (its 2 + 3 ticks inside the run), each declared joint (A97: whatever
    # the own acts did): the steps the world received sum to 0 (the born gate's p_act is the same on every tick here, every rate 0)
    n_ret = 0; worst = 0.0
    for e_ in L.anatomy.motors:
        rec = steps[e_.name]
        for m_ in [x_ for x_ in ref[e_.name][0] if 0 <= x_ and x_ + 5 <= T]:
            mv = rec[m_:m_ + 5]
            assert [p_ for p_, _, _ in mv] == [1, 1, -1, -1, -1], (e_.name, m_, mv)
            assert len({pa_ for _, pa_, _ in mv}) == 1, (e_.name, m_, [pa_ for _, pa_, _ in mv])
            for j_, sg_ in e_.spg.items():
                if all(g_[j_] != 0.0 for _, _, g_ in mv):
                    ex = sum(g_[j_] for _, _, g_ in mv); worst = max(worst, abs(ex)); n_ret += 1
                    assert abs(ex) < 1e-12 and abs(mv[0][2][j_] + mv[1][2][j_] - 2.0 * sg_ * mv[0][1] * 0.09) < 1e-12, (e_.name, m_, j_, ex)
    assert n_ret >= 30, n_ret
    # the legs: one rhythm, the right leg half of each cycle behind the left and before its next movement
    (sl, cyc), (sr, _) = ref["leg_l"], ref["leg_r"]
    n_alt = 0
    for n_ in range(min(len(sl) - 1, len(sr))):
        s_, Ln = cyc[n_]
        assert sl[n_] == math.ceil(s_) and sr[n_] == math.ceil(s_ + 0.5 * Ln) and sl[n_] < sr[n_] < sl[n_ + 1], (n_, sl[n_], sr[n_], sl[n_ + 1])
        assert abs((sr[n_] - sl[n_]) - 0.5 * Ln) < 1.0; n_alt += 1
    assert n_alt >= 12 and ref["arm"][0] != sl, n_alt
    for e_, st_ in zip(L.anatomy.motors, L.motor):                     # the life's own rhythm is the rule's: its cycle under way
        r_, lead_, lag_ = rhythm[e_.name]; cy = st_["spg_cyc"]
        assert (cy["r"], cy["lag"]) == (r_, lag_) and abs(cy["L"] - _spg_law(seed, 3, r_, cy["n"])[0]) < 1e-9 * cy["L"]
    assert all(places[(n_, p_)] > 0 for n_ in rhythm for p_ in (1, -1, 0)), places
    assert all(st_["cord_n"].get("spg", 0) >= 40 for st_ in L.motor), [st_["cord_n"] for st_ in L.motor]
    # the law of the draws, over 20,000 cycles of the legs' rhythm (the life's own `_spg_cycle` against the law written again)
    held, free = [], []
    for n_ in range(20000):
        h_, f_ = _spg_law(seed, 3, 0, n_)
        assert abs(L._spg_cycle(0, n_) - h_) < 1e-9 * h_
        held.append(h_); free.append(f_)
    mh, sh = statistics.mean(held), statistics.pstdev(held); mf, sf = statistics.mean(free), statistics.pstdev(free)
    lo_, hi_ = sum(1 for x_ in held if x_ == 1.0 / 0.15) / len(held), sum(1 for x_ in held if x_ == 8.5 / 0.15) / len(held)
    assert min(held) >= 1.0 / 0.15 and max(held) <= 8.5 / 0.15 and abs(mh - 23.425) < 0.35 and abs(sh - 11.613) < 0.45, (mh, sh)
    assert abs(mf - 3.56 / 0.15) < 0.4 and abs(sf - 1.93 / 0.15) < 0.8 and 0.008 < lo_ < 0.017 and 0.019 < hi_ < 0.031, (mf, sf, lo_, hi_)
    # the phases and the rhythms' seed: the body's seed's
    B = _born(_Kicker(TOK, cfg), cfg, _cord_world(), seed=0); C = _born(_Kicker(TOK, cfg), cfg, _cord_world(), seed=1)
    assert torch.equal(B.m.spg_phase, L.m.spg_phase) and not torch.equal(C.m.spg_phase, L.m.spg_phase)
    assert int(B.m.spg_seed) == seed and int(C.m.spg_seed) != seed
    # the rhythm is the tick's: saved at tick 400 and loaded, the body stands where the one that lives on stands, tick for tick
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    try:
        L.save(path); D = Life.load(path, _Kicker(TOK, L.cfg), save_path=None, world=_cord_world())
    finally:
        os.remove(path)
    assert torch.equal(D.m.spg_phase, L.m.spg_phase) and int(D.m.spg_seed) == seed and all(st_["spg_cyc"] is None for st_ in D.motor)
    rD = WorldLoop(D)
    for _ in range(60):
        run.step(); rD.step()
        assert [st_["now"]["spg"] for st_ in D.motor] == [st_["now"]["spg"] for st_ in L.motor] == \
            [_spg_place(_spg_starts(seed, 3, *rhythm[e_.name], L.ticks)[0], L.ticks - 1) for e_ in L.anatomy.motors]
    # refused: a rhythm named on a limb with no generator
    a = _Kicker(TOK, cfg); a.effectors[-1].spg = None; a.effectors[-1].spg_rhythm = "legs"
    try:
        a.check()
    except ValueError:
        pass
    else:
        raise AssertionError("a rhythm named on a limb with no pattern generator was taken")
    # switched off: no cord
    w0 = _cord_world(); E = _born(_Kicker(TOK, dict(cfg, spg=0)), dict(cfg, spg=0), w0); r_ = WorldLoop(E)
    for _ in range(30):
        r_.step()
    assert all(c_ == {} for c_ in w0.cords) and all(not st_["cord_n"] for st_ in E.motor) and all(st_["spg_cyc"] is None for st_ in E.motor)
    fr = {n_: [places[(n_, p_)] / T for p_ in (1, -1, 0)] for n_ in rhythm}
    print(f"motor 6: the pattern generator on {T} ticks: each limb's place and step the rule's (a movement of 2 ticks' flexion at +p_act x",
          f"0.09 rad and 3 ticks' extension at 2/3 of it back, then the drawn pause; {n_cancel} joint-ticks summed against an own act (A97),",
          f"{n_sum} summed with any own act); the cycle returns where it began: {n_ret} complete joint-movements, each summing to 0",
          f"(largest {worst:.1e} rad); flexion, extension, pause {', '.join(f'{n_} ' + '/'.join(f'{x_:.3f}' for x_ in v_) for n_, v_ in fr.items())};",
          f"the legs one rhythm, the right half a cycle behind in all {n_alt} cycles; the arm its own (phase {ph[2]:.3f}, the seed's); 20,000",
          f"draws held {0.15 * mh:.3f} +- {0.15 * sh:.3f} s (the clipped law's 3.514 +- 1.742), unheld {0.15 * mf:.3f} +- {0.15 * sf:.3f} s",
          f"(3.56 +- 1.93), {100 * lo_:.1f}% and {100 * hi_:.1f}% at the bounds; a load stands where the life stands; a stray rhythm refused;",
          "off, no cord")


class _Pain(__import__("body.core.anatomy", fromlist=["RewardSource"]).RewardSource):
    """pain as a reward source for the tests: -1 on a tick the frame's `pain` is above 0 (the G1's is the joints' observer's, A37)"""

    def felt(self, frame, life):
        p_ = frame.obs.get("pain")
        return -1.0 if p_ is not None and float(p_) > 0 else None


class _Crier(LanguageAnatomy):
    """the diary's words and face; the tract's breath left and its charge, senses of the world's frames through the face's map; pain as
    the fourth reward source; a tract of four articulators (lungs, glottis, pitch, jaw; its rest 312, all held) with the born cry: its
    posture every articulator's big step up, its lungs articulator 0"""

    def __init__(self, tok, cfg=None):
        super().__init__(tok, cfg)
        self.channels += [Channel("body", "vector", 2, organ="face_in"), Channel("charge", "vector", 2, organ="face_in")]
        self.rewards.append(_Pain("pain"))
        self.effectors.append(Effector("tract", [5, 5, 5, 5], rest_id=312, effort=0.12,
                                       cry={"posture": {0: 0.6, 1: 0.6, 2: 0.6, 3: 0.6}, "lungs": 0, "breath": ("body", 0),
                                            "charge": ("charge", 0), "pain": "pain"}))


def test_the_born_cry():
    """motor 7 (step R6h; SIM_DESIGN.md 3.7, 10, A47, C53; Jurgens 2002; Robb, Sinton-White and Kaipa 2011): the born cry, summed below
    the tract's gate. It cries on the ticks pain is felt and on every tick the charge is below 0.2, and never else; its breath groups
    are 5 ticks of the posture (each articulator's step up) then 2 of the lungs drawn back, the expiration cut short where the breath
    left is 0; where the tract's own act steps an articulator the cry's step there is dropped (the cortex can hush it), and a tick wholly
    overridden adds nothing; its ticks are counted as reflex and recorded; the gate draws on every tick and its rows are its own. Each
    tick checked against the rule written again here. Switched off it never cries"""
    def extra(w):
        br = getattr(w, "breath", 1.0); last = w.last
        if last is not None:
            c_ = getattr(last, "cord", {}).get("tract"); own = int(last.get("tract", 312)); lungs_own = own // 125 - 2
            push = (c_ is not None and c_[0] > 0) or lungs_own > 0
            br = max(0.0, br - 0.3) if push else min(1.0, br + 0.2)
        w.breath = br
        h = 0.5 - 0.004 * w.t
        return {"body": [br, 0.0], "charge": [h, -0.004], **({"pain": 1.0} if w.t in (7, 8, 30) else {})}
    cfg = dict(_LR0, wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, fast_rls=0, gate_floor=0.2, cry=1)
    w = _cord_world(extra); L = _born(_Crier(TOK, cfg), cfg, w); run = WorldLoop(L)
    t_ = 0; cries = hushed = early = 0
    for t in range(160):
        run.step(); st_ = L.motor[0]; now = st_["now"]; f_ = L.world.now
        pain = f_.obs.get("pain") is not None; low = f_.obs["charge"][0] < 0.2
        want = None
        if pain or low:
            k = t_ % 7
            if k < 5 and f_.obs["body"][0] <= 0.0:
                k = 5; t_ = (t_ // 7) * 7 + 5; early += 1
            want = [0.6, 0.6, 0.6, 0.6] if k < 5 else [-0.6, 0.0, 0.0, 0.0]
            t_ += 1
            dg = now["digits"]
            want = [0.0 if dg[j] != 2 else v for j, v in enumerate(want)]
            if not any(want):
                want = None; hushed += 1
        else:
            t_ = 0
        got = w.cords[-1].get("tract")
        assert (want is None and got is None) or (got is not None and list(got) == want), (t, pain, low, got, want, now["digits"])
        assert now["cord"] == (None if want is None else want) and st_["buf"][-1][9] is False and st_["buf"][-1][1] == now["acted"]
        cries += int(want is not None)
    assert cries >= 60 and early >= 3 and st_["cord_n"].get("cry", 0) == cries, (cries, early, hushed, st_["cord_n"])
    assert L.last["acts"]["tract"]["cord"] == now["cord"]
    w0 = _cord_world(extra); E = _born(_Crier(TOK, dict(cfg, cry=0)), dict(cfg, cry=0), w0); r_ = WorldLoop(E)
    for _ in range(80):
        r_.step()
    assert all(c_ == {} for c_ in w0.cords)
    print(f"motor 7: the born cry on 160 ticks: {cries} crying (on pain and below the 0.2 charge line, never else), in groups of 5 out and",
          f"2 in, {early} expirations cut short by an empty reservoir, {hushed} ticks wholly hushed by its own acts; counted as reflex;",
          "the gate's rows its own; off, never")


# ---------------- motor 8-10: the born codes, orienting, the VOR ----------------

class _Seer(LanguageAnatomy):
    """the diary's words and face; three senses of the world's frames through born codes (encs.<name>): "eyes" (40 numbers), "ears" (12)
    and "gazes" (the gaze's state, 3); the orienting cues (a face in the periphery, a sound's side, a sudden change); a gaze of three
    joints (yaw, pitch, vergence) that orients on its yaw and pitch, reads the cues' appearance at its gate and carries the VOR on its
    yaw and pitch; a waist of three joints that orients on its yaw, its positive step turning left"""

    def __init__(self, tok, cfg=None):
        from body.core.anatomy import OrientCue
        super().__init__(tok, cfg)
        self.channels += [Channel("eyes", "vector", 40, organ="encs.eyes", forecast=True), Channel("ears", "vector", 12, organ="encs.ears"),
                          Channel("gazes", "vector", 3, organ="encs.gazes")]
        self.effectors += [Effector("gaze", [5, 5, 5], rest_id=62, effort=0.03, sense="gazes", orient={0: ("yaw", 1), 1: ("pitch", 1)},
                                    orient_gate=True, vor=[0, 1], n_in=2),
                           Effector("waist", [5, 5, 5], rest_id=62, effort=0.05, orient={0: ("yaw", -1)}, orient_gate=True, n_in=2)]
        self.orienting = [OrientCue("face", "face_periph", fired=0, yaw=1, pitch=2, zone=0.18),
                          OrientCue("sound", "sound_side", fired=0, yaw=1, sense=-1.0, side_only=True, onset=True),
                          OrientCue("onset", "onset_periph", fired=0, yaw=1, pitch=2, zone=0.18, onset=True)]


def _seer_world():
    """a stub world for motor 8-10: random eyes and ears, the gaze's state from its acts; a face in the periphery on ticks 10-29 (right
    and above, then foveated from 20), a sound on the left at ticks 40 and 41, a sudden change right and below at 50, both the face (left)
    and the sound (right) at 60"""
    import random as _r

    class SeerWorld(SimWorld):
        def __init__(self):
            self.t = 0; self.g = [0.0, 0.0, 0.0]; self.rng = _r.Random(4); self.applied = []

        def frame(self):
            t = self.t; obs = {"eyes": [self.rng.uniform(-1, 1) for _ in range(40)], "ears": [self.rng.uniform(0, 1) for _ in range(12)],
                               "gazes": list(self.g)}
            if 10 <= t < 30 or t == 60:
                obs["face_periph"] = [1.0, 0.4 if t < 20 else (0.05 if t < 30 else -0.5), 0.3 if t < 20 else 0.02]
            if t in (40, 41, 60):
                obs["sound_side"] = [1.0, 0.6 if t < 60 else -0.6]
            if t == 50:
                obs["onset_periph"] = [1.0, 0.5, -0.4]
            return Frame(t, obs, 0.0, {"who": "parent"})

        def apply(self, acts):
            self.applied.append(acts); a = int(acts.get("gaze", 62))
            self.g = [0.07 * (a // 25 - 2), 0.07 * ((a // 5) % 5 - 2), 0.03 * (a % 5 - 2)]; self.t += 1

        def pause(self):
            pass

        def resume(self):
            pass

        def save_state(self):
            return pickle.dumps((self.t, self.g))

        def load_state(self, blob):
            self.t, self.g = pickle.loads(blob)
    return SeerWorld()


def test_the_born_codes():
    """motor 8 (step R6h; SIM_DESIGN.md 3.4, 10: each channel's code born fixed from the body's seed, unit-scaled and projected to d): a
    vector channel whose organ is encs.<its name> is encoded by its born code, built by the organs last from a generator of their own
    (the global random stream untouched), one fixed unit row per number, the code the rows' sum weighted by the numbers over sqrt(size)
    (its size the observation's root mean square: measured at d 512 over a hundred random observations of 1,536 numbers); a buffer that
    no lesson moves (a day of waking lessons leaves it as born), the same for the same seed and another for another, kept by a save;
    the window keeps the raw numbers and the input sum encodes them each time. At the G1's channels (3.4) the born codes' sizes are
    2, 1,725, 172, 1,536, 242, 130, 24 and 2. A channel naming another's code is refused"""
    from body.model import BornCode, Organs
    g0 = torch.get_rng_state().clone()
    bc = BornCode(1536, 512, torch.Generator().manual_seed(3))
    assert torch.equal(g0, torch.get_rng_state()) and not list(bc.parameters()) and torch.allclose(bc.rows.norm(dim=-1), torch.ones(1536), atol=1e-5)
    xs = torch.randn(100, 1536, generator=torch.Generator().manual_seed(5)) * 0.7
    ratio = (bc(xs).norm(dim=-1) / xs.pow(2).mean(-1).sqrt())
    assert abs(float(ratio.mean()) - 1.0) < 0.05 and float(ratio.std()) < 0.1, (float(ratio.mean()), float(ratio.std()))
    assert torch.allclose(bc(xs[0]), (xs[0] @ bc.rows) / math.sqrt(1536.0))
    cfg = dict(wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, fast_rls=0, gate_floor=0.5, live_lr=1e-3)
    L = _born(_Seer(TOK, cfg), cfg, _seer_world()); rows0 = {k_: v_.rows.clone() for k_, v_ in L.m.encs.items()}
    assert list(L.m.encs) == ["eyes", "ears", "gazes"] and not any(id(p_) in {id(q_) for g_ in L.opt_day.param_groups for q_ in g_["params"]}
                                                                  for p_ in L.m.encs.parameters())
    run = WorldLoop(L)
    for _ in range(80):
        run.step()
    assert L._wake_last and "latent_cos" in L._wake_last, L._wake_last
    assert all(torch.equal(v_.rows, rows0[k_]) for k_, v_ in L.m.encs.items()), "a lesson moved a born code"
    assert L.win[-1]["eyes"].shape == (40,) and torch.equal(L.win[-1]["eyes"], torch.tensor(L.world.now.obs["eyes"], dtype=torch.float32))
    obs, whos, bundles, reads = L._window_tensors()
    ch = L.anatomy.channel("eyes")
    assert torch.allclose(ch.encode(L.m, obs["eyes"]), (obs["eyes"] @ L.m.encs["eyes"].rows) / math.sqrt(40.0))
    B = _born(_Seer(TOK, cfg), cfg, _seer_world()); C = _born(_Seer(TOK, cfg), cfg, _seer_world(), seed=1)
    assert torch.equal(B.m.encs["eyes"].rows, rows0["eyes"]) and not torch.equal(C.m.encs["eyes"].rows, rows0["eyes"])
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    try:
        L.save(path); D = Life.load(path, _Seer(TOK, L.cfg), save_path=None, world=_seer_world())
    finally:
        os.remove(path)
    assert all(torch.equal(D.m.encs[k_].rows, rows0[k_]) for k_ in rows0)
    from body.sim.anatomy import SIM_CFG, SimAnatomy, born_table
    g = SimAnatomy(born_table(), SIM_CFG).check()
    o = Organs(g.vocab, d=32, layers=1, heads=2, window=8, channels=g.channels, effectors=g.effectors)
    assert {k_: v_.size for k_, v_ in o.encs.items()} == dict(face=2, ears=1725, eye_p=172, eye_f=1536, body=242, touch=130, vestibular=24)   # A88
    a = _Seer(TOK, cfg); a.channels[2].organ = "encs.ears"
    try:
        a.check()
    except ValueError:
        pass
    else:
        raise AssertionError("a channel naming another's born code was taken")
    print(f"motor 8: the born codes: unit rows from the seed, the global stream untouched, no parameter; the code's size the",
          f"observation's RMS x {float(ratio.mean()):.3f} (sd {float(ratio.std()):.3f}) at d 512 over 1,536 numbers; fixed through 80 ticks",
          "of waking lessons; the window raw, encoded in the sum; the seed's (another seed another; saved); the G1's eight at 3.4's sizes")


def test_orienting_and_the_vor():
    """motor 9 (step R6h; SIM_DESIGN.md 3.7, 10, A23, A43): THE BORN ORIENTING BIAS. On each tick of 70 the gaze's and the waist's choice
    probabilities are the softmax of their proposal's logits plus the born bias by the rule written again here: each cue that fires with
    a direction outside the fovea's zone (or a sound's side) adds log 4 x the gain (exactly 1 at birth) to each setting stepping toward
    it on each joint that orients on its axis and takes as much from each stepping away, the hold untouched (the face right and up pulls
    the gaze right and up and the waist, whose positive step turns left, the other way; foveated from tick 20, it pulls nothing; a sound
    on the left, read + left by the ears, pulls left; a face left and a sound right at once cancel on the yaw); the born gate input is 1
    exactly on the ticks a cue appeared (the face's first tick, the sound's onsets, the change's). Off, nothing pulls and the input is 0.
    THE VOR (motor 10's part): the tick's acts carry the gaze's VOR, its axes, born gain 1 and quick phase 0.5; off, none; the diary's
    tick returns a plain dict"""
    import math as _m
    from body.core.world import Acts
    cfg = dict(_LR0, wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, fast_rls=0, gate_floor=0.5, orient=1, vor=1)
    w = _seer_world(); L = _born(_Seer(TOK, cfg), cfg, w)
    rec = []; f = L._choose_effector

    def spy(i, frame, C1, level, stri, f=f, L=L, rec=rec):
        out = f(i, frame, C1, level, stri)
        e_ = L.anatomy.motors[i - 1]; st_ = L.motor[i - 1]
        with torch.no_grad():
            lg = L.m.acts[e_.name].logits(e_.propose(L, C1), float(L.m.read_sharp))
        rec.append((L.ticks, e_.name, frame, lg, [p_.clone() for p_ in st_["now"]["probs"]], float(st_["now"]["feat"][-1])))
        return out
    L._choose_effector = spy
    run = WorldLoop(L)
    for _ in range(70):
        run.step()
    last_face = False; pulls = 0; appeared_ticks = []
    for t, name, fr, lg, probs, gin in rec:
        cues = []
        fp, ss, op = fr.obs.get("face_periph"), fr.obs.get("sound_side"), fr.obs.get("onset_periph")
        face_on = fp is not None and fp[0] > 0
        dy = dp = 0
        if face_on:
            dy += (1 if fp[1] > 0 else -1) if abs(fp[1]) > 0.18 else 0; dp += (1 if fp[2] > 0 else -1) if abs(fp[2]) > 0.18 else 0
        if ss is not None and ss[0] > 0:
            dy += -1 if ss[1] > 0 else 1
        if op is not None and op[0] > 0:
            dy += (1 if op[1] > 0 else -1) if abs(op[1]) > 0.18 else 0; dp += (1 if op[2] > 0 else -1) if abs(op[2]) > 0.18 else 0
        appeared = (face_on and not last_face) or (ss is not None and ss[0] > 0) or (op is not None and op[0] > 0)
        if name == "waist":
            last_face = face_on
        axes = {"gaze": {0: ("yaw", 1), 1: ("pitch", 1)}, "waist": {0: ("yaw", -1)}}[name]
        want = []
        for j_, l_ in enumerate(lg):
            b_ = torch.zeros(5)
            if j_ in axes:
                ax, sg = axes[j_]; d_ = dy if ax == "yaw" else dp
                b_ = torch.tensor([_m.log(4.0) * d_ * sg * ((k > 2) - (k < 2)) for k in range(5)], dtype=torch.float32)
                pulls += int(d_ != 0)
            want.append(torch.softmax(l_ + b_, -1))
        assert all(torch.allclose(p_, w_, atol=1e-6) for p_, w_ in zip(probs, want)), (t, name, probs, want)
        assert gin == (1.0 if appeared else 0.0), (t, name, gin, appeared)
        if appeared and name == "gaze":
            appeared_ticks.append(t)
    assert appeared_ticks == [10, 40, 41, 50, 60] and pulls >= 30, (appeared_ticks, pulls)
    # the VOR in the tick's acts
    assert all(isinstance(a_, Acts) and a_.vor == {"gaze": {"axes": [0, 1], "gain": 1.0, "quick": 0.5}} for a_ in w.applied)
    # off: nothing pulls, the input 0, no VOR
    w0 = _seer_world(); E = _born(_Seer(TOK, dict(cfg, orient=0, vor=0)), dict(cfg, orient=0, vor=0), w0); r_ = WorldLoop(E); fe = E._choose_effector
    zero = []

    def spy0(i, frame, C1, level, stri, f=fe, L=E):
        out = f(i, frame, C1, level, stri); e_ = L.anatomy.motors[i - 1]; st_ = L.motor[i - 1]
        with torch.no_grad():
            lg = L.m.acts[e_.name].logits(e_.propose(L, C1), float(L.m.read_sharp))
        zero.append(all(torch.allclose(p_, torch.softmax(l_, -1), atol=1e-6) for p_, l_ in zip(st_["now"]["probs"], lg)) and float(st_["now"]["feat"][-1]) == 0.0)
        return out
    E._choose_effector = spy0
    for _ in range(70):
        r_.step()
    assert all(zero) and all(a_.vor == {} for a_ in w0.applied)
    D = _born(LanguageAnatomy(TOK, {}), {}, None); assert type(D.tick()) is dict
    print(f"motor 9: the born orienting bias by the rule on the gaze's and the waist's choices over 70 ticks ({pulls} joint-ticks pulled:",
          "a face right and up, foveated from tick 20, a sound on the left, a change right and below, a face left against a sound right);",
          f"the born gate input on the ticks a cue appeared {appeared_ticks}; the VOR's gain 1 and quick phase 0.5 in every tick's acts;",
          "off, nothing pulls, no input, no VOR; the diary's acts a plain dict")


# ---------------- motor 11: the G1's anatomy (the numbering, the drives) ----------------

def _g1_mossy(anatomy, acts, R):
    """the G1's mossy numbers for one sub-step as a world hands them, in the anatomy's declared order (SimAnatomy.mossy; an instrument of
    motor 11): the efference copy from the tick's acts (each motor effector's joint's setting, its flat act's base-5 digit, joint 0 the
    most significant), every other number drawn about its declared middle within 1.1 of its half-range"""
    cb = anatomy.cerebellar; digits = {}
    for e in anatomy.motors:
        a = int(acts.get(e.name, e.rest_id)); J = len(e.factors)
        digits[e.name] = [(a // 5 ** (J - 1 - k)) % 5 for k in range(J)]
    out, seen = [], collections.Counter()
    for name, mid, hr in zip(anatomy.mossy, cb.mossy_offset, cb.mossy_scale):
        kind, what = name.split(" ", 1)
        if kind == "act":
            eff = what.split(".", 1)[0]
            out.append(float(digits[eff][seen[eff]])); seen[eff] += 1
        else:
            out.append(mid + hr * R.uniform(-1.1, 1.1))
    return out


def _g1_world(seed=0):
    """a stub of the G1's world for motor 11 (never the sim's: random unit-scaled senses at every key of SimAnatomy's frame, the charge
    falling, now and then pain and the orienting cues), recording each tick's acts; when a life hooks its cerebellum below the tick, it
    calls it 15 times a tick as SimWorld's contract says, the mossy numbers from the tick's acts and a stream of its own (_g1_mossy), a
    teacher at each of the readouts' joints, a limit of 25 N m, and at sub-step 0 a slip and a turn, recording each answer"""
    import random as _r
    from body.core.world import SubFrame
    from body.sim.anatomy import SIZES

    class G1Stub(SimWorld):
        def __init__(self):
            self.t = 0; self.rng = _r.Random(seed); self.applied = []; self.rng_cb = _r.Random(seed + 1); self.answers = []

        def frame(self):
            t = self.t; R = self.rng
            obs = {n: [R.uniform(-1, 1) for _ in range(k)] for n, k in SIZES.items() if n != "face"}
            obs["face"] = [2.0 if t % 40 == 20 else 0.0, 0.0]
            obs["body"][241] = R.uniform(0, 1)
            obs["pain"] = [1.0 if (t % 97 == 50 and k == 5) else 0.0 for k in range(44)]
            if t % 30 == 3:
                obs["face_periph"] = [1.0, 0.4, -0.2]
            if t % 45 == 7:
                obs["sound_side"] = [1.0, 0.5]
            if t % 17 == 0:
                obs["words"] = R.randrange(3, 79)
            # the torso's unit, turning and a little tilted (R7f's heading integrates it; a function of the tick, drawing nothing), and
            # a face in the fovea every 23 ticks (R7a's event line)
            obs["imu_torso"] = [0.3 * math.sin(t / 11.0), 0.2 * math.cos(t / 13.0), 9.81, 0.05 * math.cos(t / 5.0), 0.04 * math.sin(t / 9.0),
                                0.1 * math.sin(t / 7.0) + 0.02]
            if t % 23 == 5:
                obs["face_fovea"] = [1.0]
            return Frame(t, obs, 0.0, {"who": "parent"})

        def apply(self, acts):
            self.applied.append(acts)
            if self.below is not None:                                   # the loop below the tick (the cerebellum on: SIM_CFG)
                life = self.below._life(); J = len(self.below.joints); R = self.rng_cb
                for s_ in range(15):
                    sf = SubFrame(self.t, s_, _g1_mossy(life.anatomy, acts, R), [R.uniform(-2.0, 2.0) for _ in range(J)],
                                  [R.uniform(-0.01, 0.01) for _ in range(2)] if s_ == 0 else None,
                                  [R.uniform(-0.1, 0.1) for _ in range(2)] if s_ == 0 else None, limit=[25.0] * J)
                    self.answers.append((s_, self.sub_tick(sf)))
            self.t += 1

        def pause(self):
            pass

        def resume(self):
            pass

        def save_state(self):
            return pickle.dumps(self.t)

        def load_state(self, blob):
            self.t = pickle.loads(blob)
    return G1Stub()


def test_the_g1_anatomy():
    """motor 11 (step R6h; SIM_DESIGN.md 3.4, 3.5, 6, 10, A41, C61): THE G1 AS THE CORE MEETS IT (body/sim/anatomy.py). Its check passes;
    its nine channels are 3.4's (the words' 79 symbols, then 2, 1,725, 172, 1,536, 242, 130, 24 and 2 numbers, each through its born code
    and with a forecast head of its own); its ten effectors are numbered as 3.5 numbers them: effector 0 the vocal tract (10 articulators
    of five settings, its consequence sense the ears, act_inv over them, the performance error, the born cry), effector 1 the words'
    silent output (the voice of the code: the 79 rows, mouth_gate, no intrinsic term), 2 the gaze (orienting, the VOR), 3 the waist
    (orienting), 4-5 the arms and 8-9 the legs (the pattern generators: the legs one rhythm, half a cycle apart; each arm its own), 6-7
    the Dex3 hands: 56 joint readouts of five; its rewards her face (pays) and pain (cortisol, A88). SIM_CFG sets the gates' drives as disclosed (0.25 + 4.656 x
    the reward's mean at the 256-tick clock, the error at 0.5, gate_vigor 0). Born at a small width under SIM_CFG in a stub of its world, it lives 120 ticks: every effector
    acts, each tick's acts in the design's order; THE PERFORMANCE ERROR LANDS ON THE TRACT'S GATE: its rows carry it on the ticks it acted
    and every other gate's rows (the words', the voice of the code, included) carry 0 on every tick; the legs' and arms' patterns and the
    gaze's VOR reach the world; each of the tract's lessons takes the credit 3.5 discloses (the dopamine that followed, the tonic drive
    0.25 + 4.656 x the reward's mean, 0.5 x its performance error, its cost at its own fatigue), its mean checked by hand.
    THE CEREBELLUM ON AT BIRTH (7.5, A44; the lead's mossy list): the G1 declares its cerebellar interface, 219 mossy numbers named in
    order (every one checked against the order written again here), the body's own signals only (no vision, no hearing): (1) the
    efference copy of every motor effector's acts (each joint's setting of the tract, the gaze, the waist, the arms, the hands and the
    legs in the declared order: 56; the words' token output none, the lead's decision of 2026-09-25: it moves no joint and has no body
    sense to predict), (2) the 43 joints' angles and velocities (86), (3) both inertial units' accelerometer and gyro (12), (4) touch and
    contact (the Dex3's 16 zones, the observer's torque on the 43 joints, the base's wrench: 65); each number's declared middle and
    half-range by its rule (a setting 2 and 2, an angle its joint's range from the model's file, a velocity and a gyro 0 and 1.8 rad/s,
    an accelerometer 0 and 9.81 m/s^2, touch and contact 0 and 1); the readouts on the 29 joints of the waist, the arms and the legs,
    none on the hands'; the flocculus on the gaze's yaw and pitch. SIM_CFG switches it on; the organs build it at birth for that
    declaration and the world's hook is the life's; the stub calls it 15 times a tick and every sub-step runs the law, its answers inside
    the limit and the flocculus's at sub-step 0 alone"""
    import itertools
    from body.core.anatomy import Cerebellar
    from body.core.cerebellum import Below
    from body.model import Organs
    from body.sim.anatomy import BODY_JOINTS, CEREB_JOINTS, LIMBS, MOSSY_G, MOSSY_SPEED, RANGES, SIM_CFG, SimAnatomy, born_table, TRACT, ZONES
    a = SimAnatomy(born_table(), SIM_CFG).check()
    assert [c.name for c in a.channels] == ["words", "face", "ears", "eye_p", "eye_f", "body", "touch", "vestibular"]   # A88: no charge
    assert [c.size for c in a.channels] == [79, 2, 1725, 172, 1536, 242, 130, 24] and sum(c.size for c in a.channels[1:]) == 3831   # A88
    assert all(c.forecast for c in a.channels) and all(c.organ == f"encs.{c.name}" for c in a.channels[1:])
    names = [e.name for e in a.effectors]
    assert names == ["voice", "words", "gaze", "waist", "arm_l", "arm_r", "hand_l", "hand_r", "leg_l", "leg_r"], names
    t_, w_ = a.effectors[0], a.effectors[1]
    assert a.voice is w_ and a.voice_at == 1 and len(t_.factors) == len(TRACT) == 10 and t_.sense == "ears" and t_.inverse and t_.intrinsic and t_.cry
    assert w_.factors == [79] and w_.gate == "mouth_gate" and not w_.intrinsic
    assert [len(e.factors) for e in a.motors] == [10, 3, 3, 7, 7, 7, 7, 6, 6] and sum(len(e.factors) for e in a.motors) == 56
    assert [e.name for e in a.motors if e.intrinsic] == ["voice"] and [e.name for e in a.motors if e.spg] == ["arm_l", "arm_r", "leg_l", "leg_r"]
    assert (a.effector("leg_l").spg_phase, a.effector("leg_r").spg_phase) == (0.0, 0.5) and a.effector("gaze").vor == [0, 1]
    assert [(e.name, e.spg_rhythm) for e in a.motors if e.spg] == [("arm_l", None), ("arm_r", None), ("leg_l", "legs"), ("leg_r", "legs")]
    assert [e.name for e in a.motors if e.orient] == ["gaze", "waist"] and [r.name for r in a.rewards] == ["face", "pain"]
    assert abs(SIM_CFG["gate_tonic_rate"] - 4.656402) < 1e-6 and (SIM_CFG["gate_tonic"], SIM_CFG["gate_tonic_clock"], SIM_CFG["gate_int"],
                                                                  SIM_CFG["gate_int_form"], SIM_CFG["gate_vigor"]) == (0.25, 4, 0.5, "error", 0.0)
    cfg = dict(SIM_CFG, wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.3)
    w = _g1_world(); torch.manual_seed(0)
    L = Life.birth(SimAnatomy(born_table(), cfg), device="cpu", d=32, layers=1, heads=2, window=8, cfg=cfg, seed=0, world=w)
    assert [L.m.get_submodule(e.gate).in_features - 32 - 5 for e in L.anatomy.motors] == [3, 3, 5, 4, 4, 4, 4, 4, 4]
    caught = []; fg = L._gate_lesson

    def spyg(i=0, f=fg, L=L, caught=caught):
        before = list(L.motor[0]["buf"]) if i == 1 else None
        out = f(i)
        if i == 1 and L.motor[0]["last"] and L.motor[0]["last"].get("tick") == L.ticks:
            caught.append((before, dict(L.motor[0]["last"])))
        return out
    L._gate_lesson = spyg
    run = WorldLoop(L); acted = collections.Counter(); tract_int = 0; others = 0; voice_rows = 0
    for _ in range(120):
        run.step()
        for e_, st_ in zip(L.anatomy.motors, L.motor):
            acted[e_.name] += int(st_["now"]["acted"]); row = st_["buf"][-1]
            if e_.name == "voice":
                assert row[3] == st_["now"]["int"] and (row[3] != 0.0) <= st_["now"]["acted"]
                tract_int += int(row[3] != 0.0)
            else:
                assert row[3] == 0.0, (e_.name, row[3]); others += 1
        assert L.gate_buf[-1][3] == 0.0; voice_rows += 1
        acted["words"] += int(L._acted_last)
    assert list(w.applied[-1]) == names and all(acted[n] > 0 for n in names), acted
    assert tract_int >= 10 and L.perf == {} and L.motor[0]["perf"] is not None, (tract_int, L.perf)
    assert all(any(n_ in a_.cord for a_ in w.applied) for n_ in ("arm_l", "arm_r", "leg_l", "leg_r")) and all("gaze" in a_.vor for a_ in w.applied)
    # THE DRIVES AS DISCLOSED, in the tract's own lessons (A41; 3.5's credit): on each tick it acted, G_t = the dopamine that followed
    # (from the tick after, elig_from: sum over k < 12 of 0.8^k) + 0.25 + 4.656 x the felt reward's mean at the 256-tick clock + 0.5 x its
    # performance error - its act's cost x (1 + (its own fatigue / 10)^2); the lesson's mean credit by hand
    rate = sum(0.8 ** k for k in range(12)); n_les = 0
    for buf, last in caught:
        K = 12; n = len(buf) - K - 1; G = []
        for t in range(n):
            g_ = sum((0.8 ** k) * float(buf[t + 1 + k][2]) for k in range(K))
            if buf[t][1]:
                g_ += 0.25 + rate * float(buf[t][5]) + 0.5 * float(buf[t][3]) - float(buf[t][8]) * (1.0 + (float(buf[t][4]) / 10.0) ** 2)
            G.append(g_)
        assert last["n"] == n and abs(round(sum(G) / n, 4) - last["credit_mean"]) < 1.5e-4, (last, sum(G) / n)
        n_les += 1
    assert n_les >= 3, n_les
    # THE CEREBELLUM ON AT BIRTH: the declaration (the lead's list, the body's own signals only), each number's rule, the organ and its hook
    cb, nm = a.cerebellar, list(a.mossy)
    kinds = collections.Counter(n_.split(" ", 1)[0] for n_ in nm)
    assert SIM_CFG["cereb"] == 1 and isinstance(cb, Cerebellar) and cb.n_mossy == len(nm) == 219, (SIM_CFG.get("cereb"), len(nm))
    assert kinds == {"act": 56, "angle": 43, "velocity": 43, "acc": 6, "gyro": 6, "touch": 16, "contact": 49}, kinds
    eff = [n_.split(" ", 1)[1].split(".", 1)[0] for n_ in nm if n_.startswith("act ")]
    assert [k_ for k_, _ in itertools.groupby(eff)] == [e_.name for e_ in a.motors] == [n_ for n_ in names if n_ != "words"], eff
    jn = {"voice": list(TRACT), "gaze": ["yaw", "pitch", "vergence"], "words": []}
    for e_ in a.effectors:
        want_ = jn[e_.name] if e_.name in jn else [BODY_JOINTS[j_] for j_ in e_.joints]
        assert [n_.split(".", 1)[1] for n_ in nm if n_.startswith(f"act {e_.name}.")] == want_, e_.name
    order = ([f"act {e_.name}.{j_}" for e_ in a.motors for j_ in (jn[e_.name] if e_.name in jn else [BODY_JOINTS[k_] for k_ in e_.joints])]
             + [f"angle {j_}" for j_ in BODY_JOINTS] + [f"velocity {j_}" for j_ in BODY_JOINTS]
             + [f"{k_} {u_}.{x_}" for u_ in ("torso", "pelvis") for k_ in ("acc", "gyro") for x_ in "xyz"]
             + [f"touch {z_}" for z_ in ZONES] + [f"contact {j_}" for j_ in BODY_JOINTS]
             + [f"contact base.{w_}" for w_ in ("force.x", "force.y", "force.z", "torque.x", "torque.y", "torque.z")])
    assert nm == order and not any("words" in n_ for n_ in nm), [(i_, x_, y_) for i_, (x_, y_) in enumerate(zip(nm, order)) if x_ != y_][:3]
    assert [n_ for n_ in nm if not n_.startswith("act ")][:86] == [f"angle {j_}" for j_ in BODY_JOINTS] + [f"velocity {j_}" for j_ in BODY_JOINTS]
    assert [n_ for n_ in nm if n_.split(" ", 1)[0] in ("acc", "gyro")] == [f"{k_} {u_}.{x_}" for u_ in ("torso", "pelvis") for k_ in ("acc", "gyro") for x_ in "xyz"]
    for n_, o_, h_ in zip(nm, cb.mossy_offset, cb.mossy_scale):
        kind, what = n_.split(" ", 1)
        if kind == "act":
            assert (o_, h_) == (2.0, 2.0), n_
        elif kind == "angle":
            lo_, hi_ = RANGES[BODY_JOINTS.index(what)]
            assert (o_, h_) == ((lo_ + hi_) / 2.0, (hi_ - lo_) / 2.0) and h_ > 0, n_
        elif kind in ("velocity", "gyro"):
            assert (o_, h_) == (0.0, MOSSY_SPEED) and abs(MOSSY_SPEED - 1.8) < 1e-12, n_
        elif kind == "acc":
            assert (o_, h_) == (0.0, MOSSY_G) and MOSSY_G == 9.81, n_
        else:
            assert (o_, h_) == (0.0, 1.0), n_
    assert list(cb.joints) == list(CEREB_JOINTS) == [j_ for n_, js_ in LIMBS if not n_.startswith("hand") for j_ in js_] and len(cb.joints) == 29
    assert not set(cb.joints) & {j_ for n_, js_ in LIMBS if n_.startswith("hand") for j_ in js_} and list(cb.vor) == ["yaw", "pitch"]
    org = L.m.cereb
    assert isinstance(org, torch.nn.Module) and org.declares(L.anatomy.cerebellar) and (org.n_mossy, len(org.joints), len(org.vor)) == (219, 29, 2)
    assert isinstance(w.below, Below) and w.below._life() is L and w.below.calls == 15 * 120 == len(w.answers)
    assert (int(org.n_sub), int(org.n_limb), int(org.n_vor)) == (15 * 120, 15 * 120, 119), (int(org.n_sub), int(org.n_limb), int(org.n_vor))
    assert all(ans_.torque is not None and len(ans_.torque) == 29 and max(abs(float(x_)) for x_ in ans_.torque) <= 25.0 for _, ans_ in w.answers)
    assert all((ans_.vor_gain is not None and ans_.vor_offset is not None) == (s_ == 0) for s_, ans_ in w.answers)
    assert float(org.pc_w.abs().sum()) > 0.0 and float(org.fl_w.abs().sum()) > 0.0
    print(f"motor 11: the G1's anatomy: 9 channels at 3.4's sizes (3,833 numbers and a symbol), 10 effectors numbered as 3.5 (the tract",
          f"0, the words 1), 56 joint readouts; the drives as disclosed; born at d 32 it lived 120 ticks, every effector acting ({dict(acted)});",
          f"the performance error on the tract's gate on {tract_int} ticks and 0 on every other gate's {others} rows and the words' {voice_rows};",
          f"the drives by hand in the tract's {n_les} lessons; THE CEREBELLUM on at birth: {cb.n_mossy} mossy numbers ({dict(kinds)}), each by",
          f"its rule, 29 readouts (none on the hands), the flocculus on yaw and pitch; hooked, {w.below.calls} calls in 120 ticks, the law on",
          f"every sub-step ({int(org.n_limb)} limb lessons, {int(org.n_vor)} of the flocculus)")


# ---------------- motor 12: the day saved (A70: exact replay across a save and a night) ----------------

def _g1_replay_world(seed=0):
    """a stub of the G1's world for motor 12 whose whole state goes with its save (its tick and both its streams, as SimWorld's contract
    asks, so a world saved and loaded goes on exactly): motor 11's random senses; the charge falling below the born cry's line (0.2) from
    tick 134, so the cry runs from there on; the breath left empty on ticks 170 and 260 (expirations cut short after the save); pain on
    ticks 60 and 147; a face held in the periphery on ticks 140-159 (across the save at 150) and for one tick in every 30; a sound's
    onset every 45 ticks, a sudden change every 37, a word every 17; the torso's unit turning (R7f's heading) and a face in the fovea
    every 23 ticks (R7a's line), functions of the tick; the cerebellum's hook called 15 times a tick as motor 11's stub calls it, from
    the world's second stream. It RUNS THROUGH THE NIGHT (step R8c, the live, dark night): at dusk its frames carry the body's own senses
    alone (the eyes, the ears, the face and the words off), the hook is called with no slip, and its dark goes with its save"""
    import random as _r
    from body.core.world import SubFrame
    from body.sim.anatomy import SIZES

    class G1Replay(SimWorld):
        live_night = True

        def __init__(self):
            self.t = 0; self.rng = _r.Random(seed); self.rng_cb = _r.Random(seed + 1); self.dark = False

        def frame(self):
            t = self.t; R = self.rng
            obs = {n: [R.uniform(-1, 1) for _ in range(k)] for n, k in SIZES.items() if n != "face"}
            obs["face"] = [2.0 if t % 40 == 20 else 0.0, 0.0]
            obs["body"][241] = 0.0 if t in (170, 260) else R.uniform(0.2, 1.0)
            obs["pain"] = [1.0 if ((t == 60 or 147 <= t <= 150) and k == 5) else 0.0 for k in range(44)]   # pain through the save at 150,
            # so the born cry is under way there (its breath clock 4: A88 took the charge, which had kept it crying from tick 134)
            if 140 <= t < 160 or t % 30 == 3:
                obs["face_periph"] = [1.0, 0.4, -0.2]
            if t % 45 == 7:
                obs["sound_side"] = [1.0, 0.5]
            if t % 37 == 11:
                obs["onset_periph"] = [1.0, -0.3, 0.1]
            if t % 17 == 0:
                obs["words"] = R.randrange(3, 79)
            # the torso's unit, turning and a little tilted (R7f's heading integrates it; a function of the tick, drawing nothing), and
            # a face in the fovea every 23 ticks (R7a's event line)
            obs["imu_torso"] = [0.3 * math.sin(t / 11.0), 0.2 * math.cos(t / 13.0), 9.81, 0.05 * math.cos(t / 5.0), 0.04 * math.sin(t / 9.0),
                                0.1 * math.sin(t / 7.0) + 0.02]
            if t % 23 == 5:
                obs["face_fovea"] = [1.0]
            if self.dark:                                                # the live, dark night: the eyes and the ears off (R8c)
                for k in ("ears", "eye_p", "eye_f", "face", "words", "face_periph", "sound_side", "onset_periph", "face_fovea"):
                    obs.pop(k, None)
            return Frame(t, obs, 0.0, {"who": "parent"})

        def apply(self, acts):
            if self.below is not None:
                life = self.below._life(); J = len(self.below.joints); R = self.rng_cb
                for s_ in range(15):
                    self.sub_tick(SubFrame(self.t, s_, _g1_mossy(life.anatomy, acts, R), [R.uniform(-2.0, 2.0) for _ in range(J)],
                                           [R.uniform(-0.01, 0.01) for _ in range(2)] if (s_ == 0 and not self.dark) else None,
                                           [R.uniform(-0.1, 0.1) for _ in range(2)] if (s_ == 0 and not self.dark) else None, limit=[25.0] * J))
            self.t += 1

        def pause(self):
            pass

        def resume(self):
            pass

        def dusk(self):
            self.dark = True

        def dawn(self):
            self.dark = False

        def save_state(self):
            return pickle.dumps((self.t, self.rng.getstate(), self.rng_cb.getstate(), self.dark))

        def load_state(self, blob):
            self.t, a_, b_, self.dark = pickle.loads(blob); self.rng.setstate(a_); self.rng_cb.setstate(b_)
    return G1Replay()


def test_the_day_saved():
    """motor 12 (A70, the lead's decision of 2026-09-25; SIM_DESIGN.md 6.5: exact replay across a save and a night is the law): A LIFE
    SAVED ANYWHERE GOES ON AS THE LIFE THAT WENT ON, BIT FOR BIT. The G1 (SimAnatomy, SIM_CFG with its own learning rates, at d 32) lives
    in a stub of its world whose state goes with its save. Against the life that went on, in every section of the whole state (the organs
    with their gradients, the store, the optimizers, the random streams, every working attribute, the constants) at tick 300:
    (i) INSIDE A DAY: saved at tick 150 mid-cry (its breath clock in an expiration), mid-chunk (units under way), a face held across the
        save, the cord's counts running, act_inv's pairs pending. At the load every organ, the store, the random stream, the optimizers'
        moments, every working attribute (among them the orienting's memory of the cues' last fire) and every motor effector's state
        (the breath clock, the unit, the chunk, the act last tick, the foresight, the counts) are the saved life's, but for the pattern
        generator's cache of its cycle (found again from birth at the next tick) and the gradients the last lesson left (each lesson
        makes its own again before it steps);
    (ii) AT A NIGHT'S BOUNDARY: the sleep switch fires in tick 225 and the night runs at that tick's end (C74, R8c), the save after it;
    (iii) ACROSS A NIGHT: saved at tick 150 and living on through that night.
    THE NIGHT IS R8's (SIM_CFG's night over frames and twitches, in a world that runs through the night, 120 night ticks here): the day's
    tape, the episodes and their reels, the entries' running mean, the frames' dreams, REM on frames, the live night's twitches and their
    lessons, the cerebellum learning at night; the tape inside the day (i) and the episodes after the night (ii, iii) go with the save.
    The process's global stream is not the body's: a load draws from it for the organs it builds before reading the save (anatomy 5), so
    the test seeds it as the birth did and gives it back after. AN OLDER SAVE (a motor body's save from before A70: its day taken out)
    loads with one line saying so and begins its day afresh (R6h's rule for older saves: no cry, unit or chunk under way, the stream from
    the load's seed), so it parts from the life that went on. The language body saves no day (its digests are the guard's)."""
    import contextlib
    import hashlib
    import io
    from body.sim.anatomy import SIM_CFG, SimAnatomy, born_table
    from body.tests.test_anatomy import _canon, _whole
    base = dict(SIM_CFG, wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.3, night_starts=64, night_starts_max=64, night_rounds=2,
                night_batch=8, rem_dreams=4, rem_steps=4, night_dev="", night_ticks=120)   # a night of 120 ticks, R8's whole night in it
    plain, night = dict(base, wake_ticks=100000), dict(base, wake_ticks=225)

    def born(c):
        w_ = _g1_replay_world(); torch.manual_seed(0)
        return Life.birth(SimAnatomy(born_table(), c), device="cpu", d=32, layers=1, heads=2, window=8, cfg=c, seed=0, world=w_), w_

    def live(L_, n):
        run_ = WorldLoop(L_)
        for _ in range(n):
            run_.step()

    def reborn(L_, w_, strip=False):
        fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
        try:
            L_.save(path); size = os.path.getsize(path)
            if strip:                                              # a motor body's save from before A70: no day
                blob = torch.load(path, map_location="cpu", weights_only=False); blob.pop("day"); torch.save(blob, path)
            w2 = _g1_replay_world(); w2.load_state(w_.save_state())
            g_ = torch.get_rng_state(); torch.manual_seed(0); said = io.StringIO()
            with contextlib.redirect_stdout(said):
                C_ = Life.load(path, SimAnatomy(born_table(), L_.cfg), save_path=None, world=w2)
            C_.save_path = None                                    # a load saves back to its file at its nights; these lives never save,
            torch.set_rng_state(g_)                                # as the lives that went on never do
            if not strip:
                blob = torch.load(path, map_location="cpu", weights_only=False); blob.pop("day"); torch.save(blob, path)
                size = (size, os.path.getsize(path))                # the save, and the save without its day
        finally:
            os.remove(path)
        return C_, [l_ for l_ in said.getvalue().splitlines() if l_.startswith("load:")], size

    def hsh(x, name):
        g = hashlib.sha256(); _canon(g, x, name); return g.hexdigest()

    def lost(B_, C_):
        """what of the saved life the loaded one does not hold: working attributes, motor keys, optimizers' states, organs, the stream,
        the parameters whose gradient differs"""
        skip = {"m", "store", "gen", "cfg", "save_path", "_t_feel", "anatomy", "world"}
        life_ = [k_ for k_ in sorted(set(vars(B_)) | set(vars(C_))) if k_ not in skip and not isinstance(getattr(B_, k_, None), torch.optim.Optimizer)
                 and k_ != "motor" and (k_ not in vars(B_) or k_ not in vars(C_) or hsh(getattr(B_, k_), k_) != hsh(getattr(C_, k_), k_))]
        mot_ = sorted({k_ for sb_, sc_ in zip(B_.motor, C_.motor) for k_ in set(sb_) | set(sc_) if hsh(sb_.get(k_), k_) != hsh(sc_.get(k_), k_)})
        opt_ = [k_ for k_, v_ in vars(B_).items() if isinstance(v_, torch.optim.Optimizer) and hsh(v_.state_dict(), k_) != hsh(getattr(C_, k_).state_dict(), k_)]
        sB, sC = B_.m.state_dict(), C_.m.state_dict()
        org_ = [k_ for k_ in sB if not torch.equal(sB[k_], sC[k_])]
        grads = sum(1 for (n_, p_), (_, q_) in zip(B_.m.named_parameters(), C_.m.named_parameters()) if hsh(p_.grad, n_) != hsh(q_.grad, n_))
        return life_, mot_, opt_, org_, torch.equal(B_.gen.get_state(), C_.gen.get_state()), grads
    # the lives that went on
    A1, _ = born(plain); live(A1, 300); h1 = _whole(A1)
    A2, _ = born(night); live(A2, 300); h2 = _whole(A2)
    assert (A1.nights, A2.nights) == (0, 1) and h1 != h2
    # (i) inside a day: saved at 150, mid-cry and mid-chunk, a face held across the save
    B, wB = born(plain); live(B, 150)
    cry = B.motor[0]["cry_t"]; under = [e_.name for e_, s_ in zip(B.anatomy.motors, B.motor) if s_["chunk"] > 0 and s_["unit"] is not None]
    assert B.anatomy.motors[0].cry and cry > 0 and 0 < cry % 7 < 5, cry                          # in an expiration (5 out, 2 in)
    assert len(under) >= 2 and B._orient_last["face"] and all(B.motor[i_]["cord_n"] for i_ in (0, 3, 4, 7, 8)), (under, getattr(B, "_orient_last", None))
    assert any(s_["inv_batch"] for s_ in B.motor) and any(float(s_["fatigue"]) > 0.0 for s_ in B.motor)
    # R7's state under way at the save (SIM_CFG's frames, amyg, recall): the heading turned, frames written, the record, the amygdala
    r7 = (float(B._heading), int(B._fwrites), int(B._rec_n), int(B.m.amyg.n_solve), int(B.m.amyg.ring_n), len(B._fboosts or ()))
    assert r7[0] != 0.0 and r7[1] > 0 and r7[2] == 150 and r7[3] > 0 and r7[4] > 0 and B._frec_now is not None, r7
    assert int(B._tape_n) == 150 and len(B._tape) == 1                  # R8's day's tape under way at the save
    C1, said1, size1 = reborn(B, wB)
    life_, mot_, opt_, org_, gen_, grads1 = lost(B, C1)
    assert not said1 and (life_, mot_, opt_, org_, gen_) == ([], ["spg_cyc"], [], [], True), (said1, life_, mot_, opt_, org_, gen_)
    assert C1.motor[0]["cry_t"] == cry and C1._orient_last == B._orient_last and all(s_["spg_cyc"] is None for s_ in C1.motor)
    live(C1, 150)
    assert _whole(C1) == h1, [k_ for k_ in h1 if _whole(C1)[k_] != h1[k_]]
    # (ii) at the night's boundary and (iii) across the night
    B2, wB2 = born(night); live(B2, 150)
    C3, said3, _ = reborn(B2, wB2)
    live(B2, 75); assert (B2.ticks, B2.nights) == (225, 1), (B2.ticks, B2.nights)
    r8 = B2.last_night
    assert not r8.get("error") and r8["episodes"]["kept"] > 0 and r8["dreams"] == 64 and r8["live"]["ticks"] == 120 and r8["rem_steps"] > 0, r8
    assert len(B2._episodes) == r8["episodes"]["kept"] and int(B2._tape_n) == 0
    C2, said2, _ = reborn(B2, wB2)
    life2_, mot2_, opt2_, org2_, gen2_, grads2 = lost(B2, C2)
    assert not said2 and not said3 and (life2_, mot2_, opt2_, org2_, gen2_) == ([], ["spg_cyc"], [], [], True), (life2_, mot2_, opt2_, org2_)
    live(C2, 75)
    assert _whole(C2) == h2, [k_ for k_ in h2 if _whole(C2)[k_] != h2[k_]]
    live(C3, 150); assert C3.nights == 1
    assert _whole(C3) == h2, [k_ for k_ in h2 if _whole(C3)[k_] != h2[k_]]
    # an older save: the day taken out; said once, begun afresh, and it parts from the life that went on
    C4, said4, _ = reborn(B, wB, strip=True)
    assert len(said4) == 1 and "holds no day" in said4[0] and "R6h's rule for older saves" in said4[0], said4
    assert C4.motor[0]["cry_t"] == 0 and all(s_["chunk"] == 0 and s_["unit"] is None for s_ in C4.motor) and not hasattr(C4, "_orient_last")
    live(C4, 150); h4 = _whole(C4)
    parted = [k_ for k_ in h1 if h4[k_] != h1[k_]]
    assert "work" in parted and "rng" in parted, parted
    print(f"motor 12: a life saved anywhere goes on as the life that went on (R8's night: {r8['episodes']['kept']} episodes kept,",
          f"{r8['dreams']} dreams, {r8['live']['twitches']} twitches in {r8['live']['ticks']} live night ticks):",
          f"(i) saved at tick 150 mid-cry (its breath clock {cry}, in an",
          f"expiration), {len(under)} units under way ({', '.join(under)}), a face held across the save, R7's state under way (the heading",
          f"{r7[0]:+.3f} rad, {r7[1]} frames written, the record's {r7[2]} rows, the amygdala's {r7[3]} solves and {r7[4]} forecasts pending,",
          f"{r7[5]} boosts pending); at the load all of it but the",
          f"rhythm's cache and {grads1} gradients of the last lesson; at tick 300 the whole state the uninterrupted life's in every section",
          f"({', '.join(h1)}); (ii) saved at the night's boundary (tick 225, {grads2} gradients left) and (iii) saved at 150 and living",
          f"through the night: the same; the save {size1[0]:,} bytes, {size1[0] - size1[1]:,} of them the day; an older save (no day) said",
          f"once, begun afresh, parted in {parted}; the whole state at 300, straight and continued: (i) {' '.join(f'{k_} {v_[:12]}' for k_, v_ in h1.items())};",
          f"(ii) and (iii) {' '.join(f'{k_} {v_[:12]}' for k_, v_ in h2.items())}; the older save's (i) {' '.join(f'{k_} {v_[:12]}' for k_, v_ in h4.items())}")



def test_earned_decisiveness():
    """motor 12 (A97, A104): A LATER EFFECTOR'S DECISIVENESS IS EARNED. An effector with an inverse model reads its proposal at the
    mouth's sharpness to the power of its reliability's place on kappa's scale (nothing below "fair" 0.2, complete at "almost perfect"
    0.8: Landis and Koch 1977), so at birth (inv_gain 0 until 64 acts) and up to kappa 0.2 it reads at 1, at kappa 0.5 at the mouth's
    square root, at 0.8 and above at the mouth's; an effector without an inverse model (the grip) reads at the mouth's sharpness. Born soft, the limb's joints' top probabilities stay well below 1 (the fixed point of life 1's
    second day, every joint's at 1.000, cannot form at sharpness 1)"""
    base = dict(_LR0, wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, fast_rls=0, gate_floor=0.8)
    L = _born_limbs(base, _limb_world()); run = WorldLoop(L)
    names = [e_.name for e_ in L.anatomy.motors]
    li, gi = names.index("limb"), names.index("grip")
    for _ in range(12):
        run.step()
    S = float(L.m.read_sharp)
    now_l, now_g = L.motor[li]["now"], L.motor[gi]["now"]
    assert L.motor[li]["inv_gain"] == 0.0 and abs(now_l["sharp"] - 1.0) < 1e-9, (L.motor[li]["inv_gain"], now_l["sharp"])
    assert abs(now_g["sharp"] - S) < 1e-9, (now_g["sharp"], S)
    top0 = max(float(p_.max()) for p_ in now_l["probs"])
    e_l, e_g = L.anatomy.motors[li], L.anatomy.motors[gi]                  # the law itself, at reliabilities set by hand (a living
    st_l = dict(L.motor[li])                                               # body's own kappa is recomputed at every act, so a step
    for rel, want in ((0.0, 1.0), (0.1, 1.0), (0.2, 1.0), (0.5, S ** 0.5), (0.8, S), (1.0, S), (-0.3, 1.0)):   # would reset it); A104:
        st_l["inv_gain"] = rel
        assert abs(L._motor_sharp(e_l, st_l) - want) < 1e-9, (rel, L._motor_sharp(e_l, st_l), want)
        assert abs(L._motor_sharp(e_g, dict(L.motor[gi], inv_gain=rel)) - S) < 1e-9
    assert top0 < 0.9, top0
    print(f"motor 12: a limb with an inverse model reads its proposal at sharpness 1 at birth and up to kappa 0.2 (its top probability",
          f"{top0:.2f}, the mouth's sharpness {S:.0f}), at {S ** 0.5:.1f} at kappa 0.5 and at the mouth's from 0.8 (Landis and Koch's",
          f"scale, A104); the grip (no inverse model) at the mouth's {S:.0f} throughout (A97)")


def test_moments_aligned():
    """motor 13 (A98, 2026-09-26): an optimizer's saved moments given back at a load whose recipe has a parameter the saved life had not
    (an organ born fresh): each saved state placed, in order, on the next parameter of its shape; the same shapes in the same order the
    identity; a parameter born fresh in the middle takes no moments and shifts none onto a wrong shape; a state that fits nothing is
    dropped"""
    from body.core.persistence import PersistenceMixin as P
    a, b, c = torch.zeros(3), torch.zeros(4, 2), torch.zeros(5)
    st = lambda shape: {"step": torch.tensor(1.0), "exp_avg": torch.ones(*shape), "exp_avg_sq": torch.ones(*shape)}
    opt = torch.optim.Adam([a, b, c])
    saved = {0: st((3,)), 1: st((4, 2)), 2: st((5,))}
    out = P._moments_aligned("t", opt, saved)
    assert sorted(out) == [0, 1, 2] and all(out[k] is saved[k] for k in out), out.keys()          # the identity
    saved2 = {0: st((3,)), 1: st((5,))}                                                            # b born fresh in the middle
    out = P._moments_aligned("t", opt, saved2)
    assert sorted(out) == [0, 2] and out[0] is saved2[0] and out[2] is saved2[1], out.keys()
    saved3 = {0: st((3,)), 1: st((7,)), 2: st((5,))}                                               # a state that fits nothing: dropped
    out = P._moments_aligned("t", opt, saved3)
    assert sorted(out) == [0, 2] and out[2] is saved3[2], out.keys()
    d, e_ = torch.zeros(3), torch.zeros(3)                                                        # Adam's lazy gaps on the same recipe:
    opt2 = torch.optim.Adam([a, d, e_])                                                            # a state keeps its own index, never an
    saved4 = {0: st((3,)), 2: st((3,))}                                                            # earlier same-shaped parameter's
    out = P._moments_aligned("t", opt2, saved4)
    assert sorted(out) == [0, 2] and out[2] is saved4[2] and out[0] is saved4[0], out.keys()
    print("motor 13: an optimizer's saved moments follow their parameters' shapes at a load with a parameter born fresh (A98): the",
          "identity on the same recipe, no moment on the fresh parameter, a state fitting nothing dropped")

MOTOR_TESTS = [test_the_voice_at_any_place, test_movement_units, test_the_kappa_correction, test_act_inv_batched,
               test_fatigue_per_effector_and_the_forward_error, test_the_spinal_pattern_generator, test_the_born_cry,
               test_the_born_codes, test_orienting_and_the_vor, test_the_g1_anatomy, test_the_day_saved, test_earned_decisiveness, test_moments_aligned]



def test_the_earned_certainty():
    """motor (C148, 2026-09-30): a joint that has shown nothing draws from the unit forecast's cosines, however sure the forecast of its
    own habit. An ActTable of three joints of five; the proposal 8 x a setting's own row (a habit's certain forecast): read with every
    exponent earned (1) at sharpness 1 the habit's setting takes over 0.99; with nothing earned (0) under 0.5 and each logit within
    -1..1; with no `earned` given the readout is the old one exactly; half earned lies between"""
    import torch as _t, math as _m
    from body.model import ActTable
    tab = ActTable([5, 5, 5], 32, _t.Generator().manual_seed(3))
    pred = 8.0 * tab.rows[4].clone()                        # joint 0's setting 4, the forecast sure of it
    full = tab.logits(pred, 1.0, earned=[1.0, 1.0, 1.0]); old = tab.logits(pred, 1.0)
    assert all(_t.allclose(a, b, atol=1e-5) for a, b in zip(full, old)), "earned 1 must read as before"
    p_full = _t.softmax(full[0], -1)[4].item()
    none = tab.logits(pred, 1.0, earned=[0.0, 0.0, 0.0])
    p_none = _t.softmax(none[0], -1)[4].item()
    assert p_full > 0.99 and p_none < 0.5, (p_full, p_none)
    from body.model import UNEARNED_SMALL as _US
    assert all(float(lg[1:-1].abs().max()) <= 1.0 + 1e-6 and float(lg.abs().max()) <= 1.0 + _US + 1e-6 for lg in none), [float(lg.abs().max()) for lg in none]   # (A144: the ends carry the unearned penalty)
    half = _t.softmax(tab.logits(pred, 1.0, earned=[0.5, 0.5, 0.5])[0], -1)[4].item()
    assert p_none < half < p_full, (p_none, half, p_full)
    pb0 = [float(_t.softmax(lg, -1)[[0, -1]].sum()) for lg in none]                     # A144: at nothing earned the big steps (the ends) are rare
    assert all(p_ < 0.12 for p_ in pb0), pb0
    flat = tab.logits(_t.zeros_like(pred) + 1e-9, 1.0, earned=[0.0, 0.0, 0.0])          # (a flat forecast: p(big) = 2 e^-2 / (2 e^-2 + 3))
    assert abs(float(_t.softmax(flat[1], -1)[[0, -1]].sum()) - 2 * _m.exp(-2.0) / (2 * _m.exp(-2.0) + 3)) < 1e-3
    sc1 = tab.logits(pred, 1.0, earned=1.0); sc0 = tab.logits(pred, 1.0, earned=0.0)     # C154: one exponent for every joint (an effector
    assert all(_t.allclose(a, b, atol=1e-6) for a, b in zip(sc1, full)) and all(_t.allclose(a, b, atol=1e-6) for a, b in zip(sc0, none)), \
        "a scalar earned must read as the same exponent at every joint"                    # without the per-joint law) must not raise
    print(f"motor C148: a sure habit's forecast (norm 8) at sharpness 1: setting 4 drawn at {p_full:.3f} with its certainty earned, "
          f"{p_none:.3f} with nothing earned (the cosines alone), {half:.3f} at half; the old readout unchanged")

if __name__ == "__main__":
    t0 = time.time(); failed = 0
    for t in MOTOR_TESTS:
        try:
            t()
        except AssertionError as e:
            failed += 1; print("FAIL", t.__name__, ":", e)
        except Exception as e:
            failed += 1; print("ERROR", t.__name__, ":", type(e).__name__, str(e)[:300])
    print(f"{len(MOTOR_TESTS) - failed}/{len(MOTOR_TESTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if failed else 0)
