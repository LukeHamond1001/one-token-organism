"""the motor effectors of the G1's design (docs/SIM_DESIGN.md 3.5-3.7, 7.3, 8's R6h row, A41, A43, A47, A48, C38, C53, C54, C61; the core
refactor's step R6h). Run: python3 -m body.tests.test_motor (the organ tests run these too).

What must hold: the voice (the lexicon's effector) may stand at any place among the effectors, so a body numbers its effectors as its
design does, and the place changes nothing of its life but the numbering (motor 1); the gate's intrinsic term, the performance error,
reaches only the gate of the effector that declares it, per joint for a motor effector, and never a voice that declares none (motor 1);
the language body has none of it (the eight pinned digests are the guard's, tools/pins/digests.txt). The stub worlds here are
instruments of these tests, not the G1's world."""
import collections
import math
import os
import pickle
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


MOTOR_TESTS = [test_the_voice_at_any_place]


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
