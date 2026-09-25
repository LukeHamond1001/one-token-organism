"""the parent's lane (body/sim/lane.py; S5a; docs/SIM_DESIGN.md 4.3-4.10, A1-A3, A29, A40, A49): her conduct, voice, feelings and
face joined to the G1's world. A stand-in voice (each word 2 ticks of a tone, its marks as the playback reads them) keeps these
tests off the speech engine; the real voice cache is P1's (test_sim_voice). Run: python3 -m body.tests.test_sim_lane (at nice -n 19;
about a minute)."""
import hashlib
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from body.sim import anatomy as AN  # noqa: E402
from body.sim import lane as L  # noqa: E402
from body.sim import world as W  # noqa: E402
from body.sim.lang import lexicon as LX  # noqa: E402
from body.sim.voice import synth as V  # noqa: E402

INV = {int(L.LX_TO_AN[i]): s for i, s in enumerate(LX.TABLE)}


class FakeVoice:
    """each word 2 ticks of a 200 Hz tone; the clip's word marks as the playback reads them"""

    def clip(self, text, register="plain", emphasis=None, heard=True, shape=None, **kw):
        ws = [w.strip(".,!?") for w in text.split() if w.strip(".,!?")]
        n = 2 * len(ws) * 2400
        pcm = (6000 * np.sin(2 * np.pi * 200 * np.arange(n) / 16000)).astype(np.int16)
        words = [(w, 2 * i * 2400, (2 * i + 2) * 2400) for i, w in enumerate(ws)]
        key = hashlib.sha256((text + "|" + register).encode()).hexdigest()
        return V.Clip(key, text, register, pcm, words, key)


def _world(seed=1):
    w = W.G1World(seed=seed)
    return w, L.ParentLane(w, seed=seed, voice=FakeVoice())


def _run(w, n, greet_at=None, acts=None):
    out = []
    for k in range(n):
        if greet_at is not None and w.tick == greet_at:
            w.lane.conduct.request("greet")
        fr = w.frame()
        w.apply(acts or {})
        out.append((fr.tick, int(fr.obs.get("words", L.AN_REST)), np.asarray(fr.obs.get("face", np.zeros(2))).copy(),
                    np.asarray(fr.obs.get("ears", np.zeros(1))).copy(), w.lane.last.get("line")))
    return out


def test_the_tables():
    """lane 1: her lexicon's 79 rows map one to one onto the anatomy's born table (rest 0, end 1, space 2, the letters 3-28, her 50
    birth words 29-78, in her order), read from the table itself; the article "a" keeps its own row apart from the letter's
    (AN.word_key), and the child's token output maps back"""
    assert sorted(L.LX_TO_AN.tolist()) == list(range(79))
    assert L.AN_REST == 0 and L.LX_TO_AN[LX.ID[LX.END]] == 1 and L.LX_TO_AN[LX.ID[LX.SPACE]] == 2
    assert [int(L.LX_TO_AN[LX.LETTER_ID[c]]) for c in "az"] == [3, 28]
    assert [int(L.LX_TO_AN[LX.WORD_ID[w]]) for w in LX.BIRTH_WORDS] == list(range(29, 79))
    a_word, a_letter = int(L.LX_TO_AN[LX.WORD_ID["a"]]), int(L.LX_TO_AN[LX.LETTER_ID["a"]])
    assert a_word != a_letter and a_letter == 3
    tok = AN.born_table(LX.BIRTH_WORDS)
    assert tok.token_to_id("a_") == a_word and tok.token_to_id("a") == a_letter
    assert all(int(L.AN_TO_LX[int(L.LX_TO_AN[i])]) == i for i in range(79))
    print("lane 1: her lexicon's 79 rows map one to one onto the anatomy's born table (rest 0, end 1, space 2, letters 3-28, words",
          f"29-78 in her order); the word 'a' keeps its row ({a_word}) apart from the letter's ({a_letter})")


def test_a_line_heard():
    """lane 2: a greeting asked of her conduct is said on that tick: her voice at the child's ears from the line's first tick, each
    birth word's token on the frame after the tick its sound ends (the tick its last samples reach the ears), then END, and rest
    otherwise; her percept at birth: awake, the child in her view, the room's fixtures (her face in its camera's field reported)"""
    w, lane = _world()
    got = _run(w, 30, greet_at=5)
    said = [(t, ln) for t, _s, _f, _e, ln in got if ln]
    assert len(said) == 1 and said[0][0] == 5, said
    text = said[0][1]
    ws = [x.strip(".,!?") for x in text.split()]
    syms = [(t, INV[s]) for t, s, _f, _e, _ln in got if s != L.AN_REST]
    want = [(5 + 2 * i + 2, x) for i, x in enumerate(ws)] + [(5 + 2 * len(ws) + 1, "<end>")]
    assert syms == want, (syms, want)
    ears = [float(np.abs(e).sum()) for _t, _s, _f, e, _ln in got]
    assert all(e == 0.0 for e in ears[:6]) and all(e > 0 for e in ears[6:6 + 2 * len(ws)]), ears
    p = lane._p
    assert p.present and p.child_in_view and {"mat", "sofa", "floor"} <= set(p.fixtures), p
    assert lane.conduct.world["objects"].keys() == set(lane.toys), lane.conduct.world["objects"]
    print(f"lane 2: '{text}' said on tick 5, heard at its ears from its first tick; the words channel {syms}; her percept at birth:",
          f"awake, the child in view, her face {'within' if p.seen_by_child else 'outside'} its camera's field at tick {p.tick},",
          f"fixtures {sorted(p.fixtures)}, {len(p.seen)} toys seen")


def test_exact_replay_mid_line():
    """lane 3: the world with her lane saved in the middle of her line and restored, in the same world and in a new one: the next
    frames (words, face, ears) and her lane's whole state bit for bit, the clip under way included"""
    w, lane = _world()
    _run(w, 8, greet_at=5)
    assert lane.utt is not None and not lane.utt.done
    blob = w.save_state()
    a = _run(w, 16)
    sa = pickle.dumps(W._canon(lane.state()))
    for fresh in (False, True):
        if fresh:
            w2, lane2 = _world()
        else:
            w2, lane2 = w, lane
        w2.load_state(blob)
        b = _run(w2, 16)
        for x, y in zip(a, b):
            assert x[0] == y[0] and x[1] == y[1] and np.array_equal(x[2], y[2]) and np.array_equal(x[3], y[3]) and x[4] == y[4], (x[0], y[0])
        assert pickle.dumps(W._canon(lane2.state())) == sa
    print("lane 3: saved at tick 8 mid-line and restored (same world, new world): 16 frames' words, face and ears and her lane's",
          "state bit for bit")


def test_the_night():
    """lane 4: dusk in the middle of her line: it stops where it is (its sound over, no symbol after its words already handed),
    her conduct's night kept (the day's new words join hers), the words channel at rest through the night; at dawn she speaks
    again when asked"""
    w, lane = _world()
    _run(w, 8, greet_at=5)
    n0 = lane.conduct.fast.n_lines if hasattr(lane.conduct.fast, "n_lines") else None
    w.dusk()
    assert lane.utt is None and not lane.words.queue
    night = _run(w, 12)
    assert all(s == L.AN_REST for _t, s, _f, _e, _ln in night), [s for _t, s, *_ in night]
    assert all(not ln for *_x, ln in night)
    w.dawn()
    day = _run(w, 20, greet_at=w.tick + 25)
    _run(w, 30, greet_at=w.tick + 2)
    assert lane.n_lines >= 2, lane.n_lines
    print(f"lane 4: dusk mid-line stops her line and empties the channel; 12 night ticks at rest; at dawn she greets again",
          f"({lane.n_lines} lines in all; conduct lines before the night {n0})")


def test_the_born_reading():
    """lane 5: the born reading (A1, A2, A49): 2 x (smile - frown) of the face she shows while seen; out of view it holds 30 ticks
    (READING_HOLD), then reads 0; the frame's pair is [the reading, its change]"""
    w, lane = _world()
    lane.fp = dict(L.kin.FACE_NEUTRAL, smile=0.5)
    lane._read_face(100, True)
    r = lane.reading
    assert r > 0 and np.allclose(lane.face_seen, [r, r])
    lane.fp = dict(L.kin.FACE_NEUTRAL)
    for t in range(101, 131):
        lane._read_face(t, False)
        assert lane.reading == r
    lane._read_face(131, False)
    assert lane.reading == 0.0 and np.allclose(lane.face_seen, [0.0, -r])
    lane._read_face(132, True)
    assert lane.reading == 0.0
    print(f"lane 5: the born reading {r:.2f} while seen, held {L.READING_HOLD} ticks out of view, then 0 (its change -{r:.2f})")


def test_a_toy_falls():
    """lane 6: her events from what she sees: a toy lifted 0.4 m over the mat and let go falls, "fell" once for the drop, and not
    again while it lies still"""
    w, lane = _world()
    _run(w, 3)
    m, d = w.m, w.d
    j = m.body("toy_ball").jntadr[0]
    a = m.jnt_qposadr[j]
    d.qpos[a + 2] += 0.4
    import mujoco
    mujoco.mj_forward(m, d)
    evs = []
    for _ in range(12):
        w.frame(); w.apply({})
        evs += [tuple(e) for e in lane.last["events"]]
    fell = [e for e in evs if e[0] == "fell"]
    assert fell == [("fell", "ball")], evs
    print(f"lane 6: a ball dropped from 0.4 m: {fell} once; all events {evs}")


LANE_TESTS = [test_the_tables, test_a_line_heard, test_exact_replay_mid_line, test_the_night, test_the_born_reading, test_a_toy_falls]

if __name__ == "__main__":
    t0 = time.time(); failed = 0
    for t in LANE_TESTS:
        try:
            t()
        except AssertionError as e:
            failed += 1; print("FAIL", t.__name__, ":", str(e)[:600])
        except Exception as e:
            failed += 1; print("ERROR", t.__name__, ":", type(e).__name__, str(e)[:600])
    print(f"{len(LANE_TESTS) - failed}/{len(LANE_TESTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if failed else 0)
