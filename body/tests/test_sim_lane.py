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
from body.sim.lang import dayplan as DP  # noqa: E402
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


def _world(seed=1, plan=False, day_ticks=24000):
    w = W.G1World(seed=seed)
    return w, L.ParentLane(w, seed=seed, voice=FakeVoice(), plan=plan, day_ticks=day_ticks)


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
    assert min(ears[6:6 + 2 * len(ws)]) > max(ears[:5]), ears            # her voice over the room's own sounds (W5) before it
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
    """lane 5: the born reading (A1, A2, A49; A158): 2 x (smile - frown) of the face she shows while she is with the child, looked at
    or not; away it holds 30 ticks (READING_HOLD), then reads 0; the frame's pair is [the reading, its change]"""
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
    lane.fp = dict(L.kin.FACE_NEUTRAL, smile=0.5)                     # A158: with her, unlooked at, her smile is read the tick she shows it
    lane._read_face(133, True)
    assert lane.reading == r and np.allclose(lane.face_seen, [r, r])
    print(f"lane 5: the born reading {r:.2f} while she is with the child (A158: looked at or not), held {L.READING_HOLD} ticks once away,",
          f"then 0 (its change -{r:.2f}); shown again, read again")


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


def test_the_days_layout():
    """lane 7: her day (4.7) over 200 seeds: floor play opens it, the absences are never first, last or two running, every block of
    4.7's table is there in its count (3 floor, 2 motor, 1 show, 2-4 away, 1 tasks), and the blocks fill the time from the wake to
    the winding down exactly; a short day scales every length"""
    counts = set()
    for seed in range(1, 201):
        d = DP.DayPlan(seed); d.lay_out(0)
        ks = [k for _s, _e, k, _i in d.blocks]
        assert ks[0] == "floor" and ks[-1] != "away" and not any(a == b == "away" for a, b in zip(ks, ks[1:])), ks
        assert d.blocks[0][0] == DP.WAKE and d.blocks[-1][1] == DP.DAY_TICKS - DP.WIND
        assert all(a[1] == b[0] for a, b in zip(d.blocks, d.blocks[1:]))
        n = {k: ks.count(k) for k in set(ks)}
        assert n["floor"] == 3 and n["motor"] == 2 and n["show"] == 1 and n["tasks"] == 1 and n["away"] == 1, n   # (the training day, 2026-09-29: away once)
        counts.add(n["away"])
        assert 2 <= len(d.focus) <= 3 and set(d.focus) <= set(DP.BIRTH_TOYS)
    d = DP.DayPlan(1, day_ticks=2400); d.lay_out(0)
    assert d.blocks[0][0] == 30 and d.blocks[-1][1] == 2300 and d.episode(2299) != "wind"
    # A161: a motor block entered with the child on its back owes one pull-to-sit, its first offer; later offers are the lesson
    class _Ch: posture = "back"
    class _Mo: child = _Ch()
    class _Con:
        def __init__(self): self.motion = _Mo(); self.asked = []; self.routine = None
        def request(self, intent, **kw): self.asked.append(intent)
    class _Feel:
        def set_engagement(self, x): pass
        def set_wind_down(self, x): pass
    class _Lane: pass
    ln = _Lane(); ln.conduct = _Con(); ln.feel = _Feel(); ln._p = None; ln.last = {}
    d2 = DP.DayPlan(1); d2.lay_out(0); d2._lesson = lambda t, lane: lane.conduct.asked.append("lesson")
    d2._enter("motor", 1000, 1000, ln, None)
    assert d2.sit_due
    d2._sit_or_lesson(1000, ln); d2._sit_or_lesson(1300, ln)
    assert ln.conduct.asked == ["motor_sit", "lesson"], ln.conduct.asked
    ln.conduct.motion.child.posture = "back"; d2._enter("motor", 4000, 4000, ln, None)
    d3 = DP.DayPlan(2); d3.load_state(d2.state())                        # C207: the owed pull-to-sit survives a save and a load
    assert d3.sit_due and d2.state()["sit_due"] is True
    d2.block_i = 3; d4 = DP.DayPlan(3); d4.load_state(d2.state())          # C220: the block index travels
    assert d4.block_i == 3 and d2.state()["block_i"] == 3, (d4.block_i, d2.state().get("block_i"))
    # C220: two motor blocks laid end to end: the second owes its own sit (the kind never changes, so _enter never comes)
    import types as _ty
    d2.blocks = [[1000, 4000, "motor", {}], [4000, 8000, "motor", {}]]; d2.kind = "motor"; d2.block_i = 0; d2.sit_due = False; d2.day = 0
    ln.day = 0; ln._p = _ty.SimpleNamespace(events=[], child_sounding=False); ln.conduct.motion.child.posture = "back"
    assert d2.block_index(2000) == 0 and d2.block_index(4500) == 1 and d2.block_index(9000) == -1
    assert d2.episode(4500) == "motor", d2.episode(4500)
    try:
        d2.tick(4500, 4500, ln, _ty.SimpleNamespace(parent=ln.conduct.motion))
    except AttributeError as e:                                            # (the stubs reach only the block's entry; what follows needs her)
        pass
    assert d2.sit_due and d2.block_i == 1 and any("C220" in str(x) for x in d2.log), (d2.sit_due, d2.block_i, d2.log[-2:])
    # C224: a pull-to-sit refused at her reach or her hold is owed again in the block, SIT_TRIES_PER_BLOCK tries at most
    d2.sit_due = False; d2.sit_tries = 1; d2.kind = "motor"; d2.block_i = 1
    ln.conduct.ended = {7: "refused"}; ln.conduct.motion._act = lambda mid: {"kind": "pull_to_sit", "why": "her left hand cannot reach it from here (64 cm short)"}
    for _ in range(3):
        try:
            d2.tick(5000, 5000, ln, _ty.SimpleNamespace(parent=ln.conduct.motion))
        except AttributeError:
            pass
    assert d2.sit_due and d2.sit_tries == 2 and d2.next_play >= 5000 + DP.SIT_RETRY_GAP, (d2.sit_due, d2.sit_tries, d2.next_play)
    d2.sit_due = False; d2.sit_tries = DP.SIT_TRIES_PER_BLOCK
    try:
        d2.tick(5600, 5600, ln, _ty.SimpleNamespace(parent=ln.conduct.motion))
    except AttributeError:
        pass
    assert not d2.sit_due, "the block's tries are spent"
    ln.conduct.motion._act = lambda mid: {"kind": "pull_to_sit", "why": "stopped: the pull sat at her cap for 2 ticks; it rises only with its own flexion (A9), laid back gently"}; d2.sit_tries = 1
    try:
        d2.tick(5700, 5700, ln, _ty.SimpleNamespace(parent=ln.conduct.motion))
    except AttributeError:
        pass
    assert d2.sit_due and d2.sit_tries == 2, "C257: a pull that sat at her cap for want of its flexion is tried again (the rung is learned by repetition)"
    d2.sit_due = False; d2.sit_tries = 1
    ln.conduct.motion._act = lambda mid: {"kind": "pull_to_sit", "why": "stopped: the trunk fell beyond 40 deg and she laid it back (A9, A25)"}
    try:
        d2.tick(5750, 5750, ln, _ty.SimpleNamespace(parent=ln.conduct.motion))
    except AttributeError:
        pass
    assert not d2.sit_due, "a refusal of the child, not of the moment, is not tried again"
    d5 = DP.DayPlan(5); d2.sit_tries = 2; d5.load_state(d2.state()); assert d5.sit_tries == 2
    d2.sit_due = False
    # C254: a motor block entered with the child on its front owes the sit still: at the offer it is turned onto its back first (once a
    # block), the lesson goes on while it lies so, and the sit is offered when it lies on its back; the turn count travels with a save
    ln.conduct.asked = []; ln.conduct.motion.child.posture = "front"
    d2._enter("motor", 9000, 9000, ln, None)
    assert d2.sit_due and d2.sit_turns == 0 and any("C254" in str(x) for x in d2.log[-2:]), (d2.sit_due, d2.log[-2:])
    d2._sit_or_lesson(9000, ln); d2._sit_or_lesson(9300, ln)
    assert ln.conduct.asked == ["turn_over", "lesson"] and d2.sit_due and d2.sit_turns == 1, (ln.conduct.asked, d2.sit_due, d2.sit_turns)
    ln.conduct.motion.child.posture = "back"; d2._sit_or_lesson(9600, ln)
    assert ln.conduct.asked[-1] == "motor_sit" and not d2.sit_due, ln.conduct.asked
    d6 = DP.DayPlan(6); d6.load_state(d2.state()); assert d6.sit_turns == 1
    ln.conduct.asked = []; ln.conduct.motion.child.posture = "side"; d2.sit_due = True; d2.sit_turns = DP.SIT_TURNS_PER_BLOCK
    d2._sit_or_lesson(9900, ln); assert ln.conduct.asked == ["lesson"] and d2.sit_due, "the block's turn spent: the lesson, the sit still owed"
    ln.conduct.asked = []; ln.conduct.motion.child.posture = "sitting"; d2._enter("motor", 5000, 5000, ln, None)
    assert d2.sit_due                                                      # (C254: owed whatever its posture; before: none on its front)
    d2._sit_or_lesson(5000, ln)
    assert ln.conduct.asked == ["lesson"] and d2.sit_due, ln.conduct.asked   # a sitting child is neither pulled nor turned: the lesson
    assert d.episode(2300) == "wind" and d.episode(2370) == "goodnight" and d.episode(10) == "wake"
    print(f"lane 7: 200 days laid out as 4.7 says (away {sorted(counts)} times a day, never first, last or running), filling 300 to",
          "23,000 exactly; a day of 2,400 ticks scales every length")


def test_a_short_day():
    """lane 8: a day of 2,400 ticks with the child lying still: she greets it at the wake, leaves with a goodbye and comes back with
    a greeting, shows it a toy, and says goodnight; saved in the middle of the day and restored in a new world, the rest of the day
    is the same, line for line and frame for frame"""
    w, lane = _world(plan=True, day_ticks=2400)
    lines = []
    for k in range(1200):
        w.frame(); w.apply({})
        if lane.last.get("line"):
            lines.append((k, lane.plan.kind, lane.last["line"]))
    blob = w.save_state()
    a = _run(w, 1200)
    w2, lane2 = _world(plan=True, day_ticks=2400)
    w2.load_state(blob)
    b = _run(w2, 1200)
    for x, y in zip(a, b):
        assert x[0] == y[0] and x[1] == y[1] and np.array_equal(x[3], y[3]) and x[4] == y[4], (x[0], x[4], y[4])
    lines += [(t, None, ln) for t, _s, _f, _e, ln in a if ln]
    texts = [ln for _t, _k, ln in lines]
    assert lines[0][0] == 0 and lines[0][2].startswith("hi"), lines[:2]
    assert any("bye" in x for x in texts) and any(x.startswith("night night") for x in texts), texts
    kinds = [k for _t, k, _ln in lines if k]
    assert "away" in kinds and "floor" in kinds
    print(f"lane 8: a short day: {len(lines)} lines, from '{texts[0]}' to '{texts[-1]}', a goodbye and a return; saved at tick 1200",
          "and restored in a new world, the rest of the day the same")


def test_no_meal():
    """lane 9 (A88, the owner's decision 2026-09-26): there is no meal. Through the wake and the first play block her plan asks no
    feed, no bottle is offered, her routine is never "feed", no event of hers is a low charge, and nothing she says names a bottle
    (the word stays among her 50, with nothing in the room to show it)"""
    w, lane = _world(plan=True, day_ticks=24000)
    said, kinds = [], set()
    for k in range(500):
        w.frame(); w.apply({})
        if lane.last.get("line"):
            said.append(lane.last["line"])
        kinds |= {a[1] for a in lane.conduct.acts_open}
        assert lane.conduct.routine != "feed" and not [e for e in lane.last.get("events", []) if e[0] == "charge_low"]
    assert "offer_bottle" not in kinds and not [s for s in said if "bottle" in s], (kinds, said)
    assert not hasattr(lane.plan, "feeding") and not hasattr(DP, "CHARGE_LOW") and not hasattr(w, "h")
    assert "bottle" not in DP.BIRTH_TOYS and "bottle" in LX.BIRTH_WORDS and "bottle" not in lane.toys   # the word stays; no bottle in the room
    assert [x for x in lane.plan.log if x[1] in ("feed begins", "bottle offered", "feed done")] == []
    print(f"lane 9: no meal (A88): {len(said)} lines in 500 ticks, none a bottle's; her acts {sorted(kinds)}; no feed routine, no low",
          "charge; no bottle in the room")


def test_her_eyes():
    """lane 10 (A89, the teacher's build 2a): her eyes on its acts. A ball set against its still left palm is not "got" (its own
    reach and hold needs that hand to have moved, or reached toward it, as the touch began); in another world, after its left arm
    moves for two ticks, the ball set into that palm and kept there GOT_HOLD ticks is "got ball", judged worth 2 within the tick,
    and she says "yes!" on the next free tick; never at the first tick, never a toy that lay against its hand at birth; her
    notebook and her eyes' state survive a save; a fall is still "fell", never "threw" """
    import mujoco

    def placer(w):
        m, d = w.m, w.d
        j = m.body("toy_ball").jntadr[0]
        a = m.jnt_qposadr[j]

        def place(gap):
            ch = L.PM_child(w)
            d.qpos[a:a + 3] = ch.grasp["L"] + ch.palm_n["L"] * (0.03 + gap)
            d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j] + 6] = 0.0
            mujoco.mj_forward(m, d)
        return place
    # a still palm: the ball set against it is not its own reach
    w0, lane0 = _world()
    _run(w0, 4)
    assert not [e for e in lane0.last["events"] if e[0] == "got"], lane0.last["events"]      # what lay against its hand at birth
    place0 = placer(w0)
    still = []
    for _ in range(5):
        place0(0.0); w0.frame(); w0.apply({})
        still += [tuple(e) for e in lane0.last["events"]]
    assert not [e for e in still if e[0] == "got"], still
    # its own reach: the arm moves, then the ball comes into the palm and stays
    w, lane = _world()
    _run(w, 4)
    m, d = w.m, w.d
    place = placer(w)
    evs, judged, said = [], [], []
    from body.sim import reflexes as R
    from body.sim import g1scene as G
    cl = R.CLOSING["hand_l"]
    opening = W.act_flat([2 if j not in cl else (0 if cl[j] > 0 else 4) for j in dict(G.EFFECTORS)["hand_l"]])
    for _ in range(6):                                                              # its fist (the grasp on the mat) opened by its own act
        w.frame(); w.apply({"hand_l": opening})
    for _ in range(2):                                                              # its left arm moves (the elbow's big step)
        w.frame(); w.apply({"arm_l": W.act_flat([2, 2, 2, 0, 2, 2, 2]), "hand_l": opening})
    for k in range(5):                                                              # the ball into that open palm within MOVED_TICKS, kept
        place(0.02); w.frame(); w.apply({"hand_l": opening} if k < 2 else {})       # there GOT_HOLD ticks (a hold, not a graze); then
        evs += [tuple(e) for e in lane.last["events"]]                             # the grasp closes on it
        judged += [tuple(x) for x in (lane.last.get("judged") or ())]
        if lane.last.get("line"):
            said.append((w.tick, lane.last["line"]))
    for _ in range(6):
        w.frame(); w.apply({})
        evs += [tuple(e) for e in lane.last["events"]]
        judged += [tuple(x) for x in (lane.last.get("judged") or ())]
        if lane.last.get("line"):
            said.append((w.tick, lane.last["line"]))
    got = [e for e in evs if e[0] == "got"]
    assert got == [("got", "ball")], evs
    assert [j for j in judged if j[1] == "got"] == [(2, "got", "ball")], judged
    assert said and any(ln.startswith(("yes", "good")) for _t, ln in said), said     # her "yes!" marks the smile (after her label of
                                                                                     # the ball coming into its hand, if that came first)
    seen_at = []                                                                     # A94: she smiles en face: her mouth in its fovea
    for k in range(40):                                                              # (the face test) within the smile's hold, and
        w.frame(); w.apply({})                                                       # the born reading rises: the smile is felt
        if lane.last.get("face_test"):
            seen_at.append(w.tick)
        if seen_at and lane.reading > 0:                                             # (A158: the reading rises the tick she smiles, looked
            break                                                                    # at or not; her en-face conduct is read on the test)
    assert seen_at and lane.reading > 0, (seen_at, lane.reading, lane.feel.log[-6:])
    import body.sim.lane as LN_
    from body.sim.lang import consts as LC, percept as LP
    assert "held" in LP.EVENT_KINDS and LC.MOTOR_WORTH["held"] == (1, "shook")   # C225: her smile for the hand that holds on (shaping toward the shake)
    lane.lifted.add("ball"); lane.held_run = {}
    ev2 = []
    for _ in range(LN_.HELD_TICKS + 10):                                             # the ball, lifted, kept in its hand HELD_TICKS running:
        lane._held(("ball", "rattle"), ev2)                                          # "held" once; the rattle its right hand lies on, never
    assert ev2 == [("held", "ball")], (ev2, lane.held_run)                           # lifted, is not held
    assert lane.held_run["ball"] == LN_.HELD_TICKS + 10 and lane.held_run["rattle"] == 0
    lane._held(("rattle",), ev2); assert lane.held_run["ball"] == 0                  # the ball out of its hand: the count falls
    for _ in range(LN_.HELD_TICKS):                                                  # back in its hand: a new hold, held again
        lane._held(("ball",), ev2)
    assert ev2 == [("held", "ball"), ("held", "ball")], ev2
    # C229: a toy in a hand whose palmar grasp is engaged counts without a lift (the cord's 'grasp' on that hand); a toy a resting
    # hand merely lies on (no grasp, no lift) does not
    lane.lifted.discard("ball"); lane.held_run = {}; ev3 = []
    touch_ = {"ball": {"child": {"left"}}, "rattle": {"child": {"right"}}}
    for _ in range(LN_.HELD_TICKS + 2):
        lane._held(("ball", "rattle"), ev3, touch=touch_, grasping={"left"})
    assert ev3 == [("held", "ball")] and lane.held_run["rattle"] == 0, (ev3, lane.held_run)
    lane.held_run = {}; ev4 = []
    for _ in range(LN_.HELD_TICKS + 2):
        lane._held(("ball",), ev4, touch=touch_, grasping=set())                    # the hand open on it: not held
    assert ev4 == [], ev4
    assert lane.conduct.book["got"] == {"ball": 1} and set(lane.conduct.book) <= {"got", "lifted", "hit", "held"}, lane.conduct.book   # in a
    st = lane.state()                                                                # raised hand: "lifted" too, as a person would see it
    assert st["eyes"]["hand_prev"]["left"] and st["conduct"]["book"]["got"] == {"ball": 1}
    w2, lane2 = _world()
    w2.load_state(w.save_state())
    assert lane2.conduct.book == lane.conduct.book and lane2.toy_rest_z == lane.toy_rest_z and lane2.hand_moved_t == lane.hand_moved_t
    # a fall is a fall, not a throw (the duck dropped from 0.4 m)
    j2 = m.body("toy_duck").jntadr[0]
    d.qpos[m.jnt_qposadr[j2] + 2] += 0.4
    mujoco.mj_forward(m, d)
    evs2 = []
    for _ in range(10):
        w.frame(); w.apply({})
        evs2 += [tuple(e) for e in lane.last["events"]]
    assert ("fell", "duck") in evs2 and not [e for e in evs2 if e[0] == "threw"], evs2
    print(f"lane 10: her face in its fovea at ticks {seen_at[:3]} after the judgment, its reading {lane.reading:.1f} (A94);",
          "a ball against a still palm: no 'got'; after its arm moved, the ball kept in its palm: events",
          f"{[e for e in evs if e[0] in ('got', 'reach_nearer', 'lifted', 'hit')]}, judged {judged}, her line {said[0]}; her book",
          f"{lane.conduct.book}; saved and restored; a dropped duck fell, not thrown")


def test_her_lessons():
    """lane 11 (A90, the teacher's build 2c): her lessons in a short day with a still child. In motor time her plan sets a focus toy
    within its reach ("set_near": her line "here. the X." and her motion's bring_back at lesson_dist 0.10, level 0); the toy comes
    to rest within reach of a hand as she sees it; a "got" smile on that toy (written into her book here) raises the next lesson
    on it a level (0.15); her plan's levels survive a save; floor play gives lessons among its offers"""
    _blocks = DP.BLOCKS                                                  # (these scenarios were drawn under 4.7's own table: the training
    DP.BLOCKS = (("floor", 3, 4000, 5000), ("motor", 2, 1000, 1500), ("show", 1, 1500, 1500), ("away", (2, 4), 400, 1200), ("tasks", 1, 3000, 3000))   # day (2026-09-29) draws another day; the lesson's law is what is tested)
    try:
        return _test_her_lessons_body()
    finally:
        DP.BLOCKS = _blocks


def _test_her_lessons_body():
    w, lane = _world(plan=True, day_ticks=2400)
    lessons, lines = [], []
    for k in range(1300):
        w.frame(); w.apply({})
        if lane.last.get("line"):
            lines.append((k, lane.plan.kind, lane.last["line"]))
        for x in lane.plan.log:
            if x[1] == "lesson" and x not in lessons:
                lessons.append(x)
    reach = [x for x in lessons if x[2] == "reach"]
    assert reach and all(x[4] == 0 and x[5] == 0.1 for x in reach), reach[:4]
    toy = reach[0][3]
    assert [ln for _k, _kind, ln in lines if ln.startswith(("here.", "look. here.")) and toy in ln], lines[:20]
    p = lane._p
    seen = {s.id: s for s in p.seen}
    assert toy in seen and (seen[toy].child_can_reach or seen[toy].on in ("hand", "mama")), (toy, seen.get(toy))
    lane.conduct.book.setdefault("got", {})[toy] = 1                                # it got the toy once (its own reach): a level up
    lane.plan.focus = [toy]
    lane.plan._lesson(w.tick, lane)
    later = [x for x in lane.plan.log if x[1] == "lesson" and x[2] == "reach" and x[3] == toy and x[0] == w.tick]
    assert later and later[0][4] == 1 and later[0][5] == 0.15, later[:2]
    assert abs(lane.conduct.motion.lesson_dist - 0.15) < 1e-9
    st = lane.plan.state()
    assert st["level"] == {toy: 1} and toy in st["got_seen"]
    w2, lane2 = _world(plan=True, day_ticks=2400)
    w2.load_state(w.save_state())
    assert lane2.plan.level == lane.plan.level and lane2.conduct.motion.lesson_dist == lane.conduct.motion.lesson_dist
    lane.plan.level_t[toy] = w.tick - DP.NO_PROGRESS - 1                            # a block with no new "got": a level back
    lane.plan._lesson(w.tick + 1, lane)
    back = [x for x in lane.plan.log if x[1] == "lesson" and x[2] == "reach" and x[3] == toy and x[0] == w.tick + 1]
    assert back and back[0][4] == 0 and back[0][5] == 0.1, back
    print(f"lane 11: her lessons: {len(reach)} reach lessons at level 0 (0.10 m) on {sorted({x[3] for x in reach})}, her line",
          f"'{[ln for _k, _kind, ln in lines if toy in ln][0]}'; the {toy} within reach as she sees it; after one 'got' the next lesson",
          f"on it at level 1 (0.15 m); the plan's levels saved and restored")



def test_smile_brought():
    """lane 12 (A96): a judged smile she is bringing to the child's line of sight (her lean_in child_line under way) is held until
    she is there and 20 ticks more, never past queue_expiry (40) ticks from its start; one not brought eases after 20 as before;
    one seen while brought is held 10 ticks after the look, as every seen smile is"""
    from body.sim import parent_feel as PF
    C = PF.FEEL

    def joy(bringing_ticks, seen_at=None):
        F = PF.Feelings(1)
        F.step(False)
        F.judge(2.0, "got")
        out = []
        for k in range(60):
            fp = F.step(seen_at is not None and k >= seen_at, bringing=k < bringing_ticks)
            out.append(round(float(F.state["joy"]), 3))
        return out
    plain = joy(0)
    assert plain[0] > 0 and plain[C["smile_wait"]] > 0 and plain[C["smile_wait"] + C["offset"] + 1] == 0, plain[:30]
    brought = joy(30)                                                                # under way 30 ticks: held through, then 10 more
    assert brought[30] > 0 and brought[min(30 + C["smile_wait"], C["queue_expiry"])] > 0, brought[:45]
    assert brought[C["queue_expiry"] + C["offset"] + 1] == 0, brought[35:50]         # never past the cap
    forever = joy(60)                                                                # never there: the cap ends it
    assert forever[C["queue_expiry"]] > 0 and forever[C["queue_expiry"] + C["offset"] + 1] == 0, forever[35:50]
    seen = joy(30, seen_at=12)                                                       # seen on the way: 10 ticks after the look
    assert seen[13 + C["smile_after_seen"]] > 0 and seen[13 + C["smile_after_seen"] + C["offset"] + 1] == 0, seen[:35]
    print("12 a smile brought to its line of sight is held while she brings it (at most 40 ticks), 20 more once there, 10 after a look;",
          "one not brought eases after 20 as before")



def test_a_face_down_morning():
    """lane 13 (A107, C89): a child that slept face down wakes to a new spell: the lane's `distressed` flag (one distress event a
    spell) and its face-down count start again at dawn, so the distress fires again after DISTRESS_TICKS face down and her turn is
    owed (life day 6 opened prone with no turn asked: the flag had been saved True across the night)"""
    w, lane = _world()
    _run(w, 3)
    lane.distressed = True; lane.face_down = 500; lane.cry_down = 3        # as day 5's dusk left it
    lane.off_back = 900; lane.tummy_over = True                            # (C217: tummy time's clock and its flag too)
    lane.dusk(w); lane.dawn(w)
    assert lane.distressed is False and lane.face_down == 0 and lane.cry_down == 0, (lane.distressed, lane.face_down, lane.cry_down)
    assert lane.off_back == 0 and lane.tummy_over is False, (lane.off_back, lane.tummy_over)
    print("13 a face-down spell starts anew at dawn: the distress flag and the face-down count reset with the day (A107)")

def test_the_roll_rung():
    """lane 14 (A109, C91): once the child rolls (her book's 'rolled' at MASTERED_N) and lies on its back or front, every other
    reach-rung lesson is the roll rung: "set_far" (her line "look. the X." or "look. here. the X.") with her motion's bring_far, the
    toy fetched and, from a kneel on the child's far side, set down beside its far shoulder ROLL_BEYOND_M past the reach of the arm
    on that side: out of its reach as she sees it (child_can_reach False), on the side away from where she knelt. The next lesson is
    a reach lesson again (they alternate), and the turn is saved. Life day 7: 89 rolls, 0 reaches, and no motor act judged"""
    _blocks = DP.BLOCKS                                                  # (these scenarios were drawn under 4.7's own table: the training
    DP.BLOCKS = (("floor", 3, 4000, 5000), ("motor", 2, 1000, 1500), ("show", 1, 1500, 1500), ("away", (2, 4), 400, 1200), ("tasks", 1, 3000, 3000))   # day (2026-09-29) draws another day; the lesson's law is what is tested)
    try:
        return _test_the_roll_rung_body()
    finally:
        DP.BLOCKS = _blocks


def _test_the_roll_rung_body():
    from body.sim import parent_motion as PM
    w, lane = _world(plan=True, day_ticks=2400)
    for _k in range(200):
        w.frame(); w.apply({})
    p = lane._p
    seen = {s_.id: s_ for s_ in p.seen}
    free = [o for o in sorted(seen) if seen[o].on != "hand" and w.parent._toy_clearance(o, np.zeros(3)) >= 0.02]
    free = [o for o in free if o not in ("ball", "ring", "car")] or free   # (one that stays where she sets it: a ball rolls off the mat)
    toy = ([o for o in lane.plan.focus if o in free] or free)[0]      # a toy the child does not hold (her fetch never takes one from it)
    lane.plan.focus = [toy]
    lane.conduct.book.setdefault("rolled", {})[""] = DP.K.MASTERED_N
    lane.plan.roll_turn = True
    assert lane.posture == "back", lane.posture
    n0 = len(w.parent.acts)
    lane.plan._lesson(w.tick, lane)
    roll = [x for x in lane.plan.log if x[1] == "lesson" and x[2] == "roll"]
    assert roll and roll[-1][3] == toy and roll[-1][4] == DP.K.MASTERED_N, lane.plan.log[-3:]
    lines, act = [], None
    for _k in range(1500):
        w.frame(); w.apply({})
        if lane.last.get("line"):
            lines.append(lane.last["line"])
        fars = [a for a in w.parent.acts if a["kind"] == "bring_far"]
        if fars and fars[-1]["status"] in PM.DONE_STATES:
            act = fars[-1]
            for _j in range(40):                                        # the toy comes to rest, she looks back to its eyes
                w.frame(); w.apply({})
            break
    assert act is not None and act["status"] == "done", (act, lines[-5:])
    said = [ln for ln in lines if toy in ln and ln.startswith("look.")]
    assert said, lines[-12:]
    pm = w.parent
    ch = PM.Child(w.m, w.d, w.scene.g1_set)
    far = "R" if ch.face_side() == "L" else "L"
    pos = w.d.xpos[lane.toy_body[toy]]
    d_sh = {s_: float(np.linalg.norm(pos - w.d.xpos[lane.shoulder[s_]])) for s_ in lane.shoulder}
    far_name = "left" if far == "L" else "right"
    assert all(d_sh[s_] > lane.arm_reach[s_] for s_ in d_sh), (d_sh, lane.arm_reach)         # out of its reach from either shoulder
    assert d_sh[far_name] < lane.arm_reach[far_name] + 0.30, (d_sh, lane.arm_reach)         # and not far past it
    side = float((pos[:2] - ch.torso[:2]) @ (ch.lat[:2] * (1 if far == "L" else -1)))
    assert side > 0.20, (side, far)                                                          # on its far side
    p = lane._p
    seen = {s_.id: s_ for s_ in p.seen}
    assert toy in seen and not seen[toy].child_can_reach and seen[toy].on != "mama", seen.get(toy)
    lane.plan._lesson(w.tick, lane)                                    # the next lesson: the reach rung again (they alternate)
    last = [x for x in lane.plan.log if x[1] == "lesson"][-1]
    assert last[2] == "reach" and lane.plan.roll_turn is True, last
    st = lane.plan.state()
    assert st["roll_turn"] is True
    lane.plan.roll_turn = False
    st2 = lane.plan.state(); lane.plan.load_state(st2)
    assert lane.plan.roll_turn is False
    print(f"lane 14: the roll rung (A109): once it rolls, her lesson set the {toy} beside its far ({far_name}) shoulder, out of reach",
          f"from either shoulder ({', '.join(f'{k} {v:.2f} m of {lane.arm_reach[k]:.2f}' for k, v in d_sh.items())}) and {side:.2f} m to",
          f"its far side, her line '{said[0]}', the act done in {act['end'] - act['start']} ticks; the next lesson a reach lesson again;",
          f"the turn saved")



def test_a_toy_she_could_not_get_to():
    """lane 15 (A117, C102): a show of the cup in the room's corner behind the plant is refused by her motion ("nowhere to kneel by
    the cup"), and her conduct, reading the refusal, leaves the cup where it lies (`conduct.left`): her lessons pass it over (the
    lesson's toy is another), until it has moved LEFT_MOVED_M or the dawn (lane.dawn clears it). Saved with her conduct. Life days
    13-14: the cup under the table asked for 17 times, refused each"""
    import mujoco
    from types import SimpleNamespace
    from body.sim import parent_motion as PM
    w, lane = _world(plan=True, day_ticks=2400)
    m, d = w.m, w.d
    b = m.body("toy_cup").id; j = m.body_jntadr[b]; adr = m.jnt_qposadr[j]; dof = m.jnt_dofadr[j]
    d.qpos[adr:adr + 2] = (-2.5, -2.2); d.qvel[dof:dof + 6] = 0        # the room's corner behind the plant (the stacker's place on
    for _k in range(60):                                                 # life days 13-14): no spot to kneel at
        w.frame(); w.apply({})
    assert w.parent._toy_spot(d.xpos[b][:2]) is None
    c = lane.conduct
    assert c.left == {}, c.left
    i = w.parent.request(SimpleNamespace(kind="show", target="cup", during=None, thing=None))   # the act as her conduct's line
    c.acts_open.append([i, "show", "cup", None, int(w.tick), "running"])                       # would ask it, opened in its book
    for _k in range(900):
        w.frame(); w.apply({})
        if w.parent.status(i) in PM.DONE_STATES:
            for _j in range(3):
                w.frame(); w.apply({})
            break
    why = w.parent.why(i)
    assert w.parent.status(i) == "refused" and "nowhere to kneel" in why, (w.parent.status(i), why)
    assert "cup" in c.left, (why, c.left)
    at = list(c.left["cup"])
    assert c.left_where_it_lies("cup")
    p = lane._p
    seen = {s_.id: s_ for s_ in p.seen}
    other = [o for o in ("duck", "block", "car", "bear", "ball") if o in seen and seen[o].on != "hand"]
    lane.plan.focus = ["cup"] + other[:1]
    lane.plan.roll_turn = False
    n0 = len(lane.plan.log)
    for _k in range(6):
        lane.plan._lesson(w.tick, lane)
    lessons = [x for x in lane.plan.log[n0:] if x[1] == "lesson"]
    assert all(x[3] != "cup" for x in lessons), lessons             # never the cup she left
    assert (lessons and all(x[3] == other[0] for x in lessons)) or not other, (lessons, other)
    blob = c.state()
    assert blob["left"] == {"cup": at}, blob["left"]
    d.qpos[adr:adr + 2] = (-2.5 + 0.2, -2.2)                             # the cup moved (the child, or she): a toy again
    mujoco.mj_forward(m, d)
    assert not c.left_where_it_lies("cup") and "cup" not in c.left
    c.left["duck"] = [0.0, 0.0]
    lane.dawn(w)
    assert c.left == {}, "the dawn clears what she left"
    print(f"lane 15: the show of the cup behind the plant refused ({why[:40]}), the cup left where it lay {np.round(at, 2).tolist()};",
          f"{len(lessons)} lessons since went to {sorted(set(x[3] for x in lessons))} (the cup passed over); moved 0.2 m it is a toy again;",
          "the dawn clears the rest")



def test_the_new_toy_in_her_focus():
    """lane 16 (A119): the toys she has named join the day's focus pool, and the newest of them is in every day's focus: in the room
    with the book, before she has said "book" her focus is drawn from the birth toys; once "book" is among her new words (the day's)
    or her words (after a night) the book is in the focus of every day laid out, with one or two birth toys beside it; a growth word
    she has said for a toy that is not in the room adds nothing. Life day 15: the book's acts all fell in the day's first quarter,
    then it lay out of the child's reach and no lesson brought it back"""
    from body.sim import extras as X
    w = W.G1World(seed=1, extra=X.add_book())
    lane = L.ParentLane(w, seed=1, voice=FakeVoice(), plan=True, day_ticks=2400)
    assert "book" in lane.toys
    f = lane.conduct.fast
    assert "book" not in f.vocab and "book" not in f.new_words
    for d in range(1, 4):
        lane.plan.lay_out(d, lane)
        assert "book" not in lane.plan.focus and set(lane.plan.focus) <= set(DP.BIRTH_TOYS) and 2 <= len(lane.plan.focus) <= 3, lane.plan.focus
    f.new_words = ("book",)
    focs = []
    for d in range(4, 10):
        lane.plan.lay_out(d, lane)
        focs.append(list(lane.plan.focus))
        assert "book" in lane.plan.focus and 2 <= len(lane.plan.focus) <= 3 and all(o in DP.BIRTH_TOYS or o == "book" for o in lane.plan.focus), lane.plan.focus
    f.vocab = f.vocab + ("book",); f.new_words = ()
    lane.plan.lay_out(10, lane)
    assert "book" in lane.plan.focus
    f.new_words = ("kiwi",)                                              # a word for nothing in the room adds nothing
    lane.plan.lay_out(11, lane)
    assert "kiwi" not in lane.plan.focus and "book" in lane.plan.focus
    w2, lane2 = _world(plan=True, day_ticks=2400)                       # the room without the book: "book" said, no book to focus on
    lane2.conduct.fast.new_words = ("book",)
    lane2.plan.lay_out(1, lane2)
    assert "book" not in lane2.plan.focus and set(lane2.plan.focus) <= set(DP.BIRTH_TOYS)
    print(f"lane 16: before 'book' is said the focus is the birth toys'; after, the book is in every day's focus ({focs[:3]} ...); a word",
          "for nothing in the room adds nothing; without the book in the room, nothing")


def test_the_find():
    """lane 18 (A129, the hide game on the bucket A126): a toy that lies in the bucket from HER hand (her hand let it go within HANDOVER_TICKS)
    is hidden (`hidden`, saved with the lane); a toy seen there is `on` "bucket"; its own hand on the hidden toy, held GOT_HOLD ticks, is
    the event "found", worth 2 in her book (consts.MOTOR_WORTH), and the toy is hidden no more; a toy that came into the bucket by no hand
    of hers is not hidden, and its grasp is no find; a hidden toy out of the bucket with no hand on it for HIDDEN_OUT_TICKS is hidden
    no more"""
    import mujoco
    from body.sim import extras as X
    from body.sim import reflexes as R
    from body.sim import g1scene as G
    from body.sim.lang import consts as CK, percept as PC
    assert CK.MOTOR_WORTH["found"][0] == 2 and "found" in PC.EVENT_KINDS
    w = W.G1World(seed=1, extra=X.add_bucket(xy=(-0.3, -0.45)))
    lane = L.ParentLane(w, seed=1, voice=FakeVoice(), plan=False, day_ticks=2400)
    _run(w, 4)
    m, d = w.m, w.d
    bucket = m.body("toy_bucket").id; jd = m.body("toy_duck").jntadr[0]; a = m.jnt_qposadr[jd]; va = m.jnt_dofadr[jd]

    def into_bucket():
        d.qpos[a:a + 3] = d.xpos[bucket] + np.array([0.0, 0.0, X.BUCKET_WALL + 0.04]); d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        d.qvel[va:va + 6] = 0.0; mujoco.mj_forward(m, d)

    def to_hand(gap):
        ch = L.PM_child(w)
        d.qpos[a:a + 3] = ch.grasp["L"] + ch.palm_n["L"] * (0.03 + gap); d.qvel[va:va + 6] = 0.0; mujoco.mj_forward(m, d)

    cl = R.CLOSING["hand_l"]
    opening = W.act_flat([2 if j not in cl else (0 if cl[j] > 0 else 4) for j in dict(G.EFFECTORS)["hand_l"]])

    def grasp(evs, judged):
        """its fist opened, its arm moved, the duck into the open palm (lane 10's road): the duck leaves the bucket for its hand within
        HIDDEN_OUT_TICKS, as a hand reaching in and lifting takes it out"""
        jr = m.body("toy_rattle").jntadr[0]; ar = m.jnt_qposadr[jr]; vr = m.jnt_dofadr[jr]
        d.qpos[ar:ar + 2] = [1.5, 1.5]; d.qvel[vr:vr + 6] = 0.0; mujoco.mj_forward(m, d)   # (C180: the rattle out from under its left hand:
        for _ in range(6):                                                  # the resting pose lays the hand on it with some bucket masses, and a
            w.frame(); w.apply({"hand_l": opening})                         # duck let down to the palm meets the rattle, not the hand)
        for k in range(5):                                                  # the duck pressed to the open palm each tick: at rest the palm faces
            to_hand(-0.06); w.frame(); w.apply({"hand_l": opening} if k < 2 else {})   # sideways and its thin plate lies 3 cm inside the grasp
            evs += [tuple(e) for e in lane.last["events"]]; judged += [tuple(x) for x in (lane.last.get("judged") or ())]   # point (a duck set
        for _ in range(6):                                                  # at the point, or 5 cm out as lane 10's ball is, falls to the floor
            w.frame(); w.apply({})                                          # beside it, no touch); the find needs no arm move, "got" does
            evs += [tuple(e) for e in lane.last["events"]]; judged += [tuple(x) for x in (lane.last.get("judged") or ())]

    w.parent.holding["R"] = "duck"                                          # in her hand, as the lane sees it
    for _ in range(3):
        w.frame(); w.apply({})
    into_bucket(); w.parent.holding["R"] = None                                 # let go into the bucket
    evs, judged = [], []
    for _ in range(8):
        w.frame(); w.apply({}); evs += [tuple(e) for e in lane.last["events"]]
    assert lane.last["in_bucket"] == ["duck"] and "duck" in lane.hidden, (lane.last["in_bucket"], lane.hidden)
    assert "hidden" in lane.state() and lane.state()["hidden"] == lane.hidden
    seen_on = {s_.id: s_.on for s_ in lane._p.seen}
    import body.sim.eyes as EY
    _face = EY.mouth_point(m, d)[2]; _at = d.xpos[bucket]; _rim = _at + d.xmat[bucket].reshape(3, 3) @ np.array([0.0, 0.0, X.BUCKET_WALL + X.BUCKET_H])
    _hit = lane._ray_first(m, d, _face, _rim)
    if _hit is None or int(m.body_rootid[_hit]) != lane.g1_root:            # C179: with the rim at 8 cm her line from the door to it may pass through
        assert "bucket" in seen_on, ("C129: the bucket seen while a toy lies on its floor plate", sorted(seen_on))   # the child lying between (a real
    else:                                                                   # occlusion); C129's check holds whenever the line is clear
        print("  (the child's body takes her line to the rim from the door: C129's check skipped)")
    grasp(evs, judged)
    assert ("found", "duck") in evs, evs
    assert [j for j in judged if j[1] == "found"] == [(2, "found", "duck")], judged
    assert "duck" not in lane.hidden, lane.hidden
    for _ in range(L.HANDOVER_TICKS + 2):                                    # her hand-over window from the first release passes
        w.frame(); w.apply({})
    into_bucket()                                                               # by no hand of hers: in the bucket, not hidden
    for _ in range(L.HIDDEN_OUT_TICKS + 2):
        w.frame(); w.apply({})
    assert lane.last["in_bucket"] == ["duck"] and "duck" not in lane.hidden, (lane.last["in_bucket"], lane.hidden)
    evs2, judged2 = [], []
    grasp(evs2, judged2)
    assert ("found", "duck") not in evs2 and "duck" not in lane.hidden, (evs2, lane.hidden)
    print(f"lane 18: the duck let go into the bucket from her hand is hidden (seen on {seen_on.get('duck')!r}); its own hand on it: 'found', worth",
          f"{[j for j in judged if j[1] == 'found'][0][0]}, hidden no more; a duck in the bucket by no hand of hers is not hidden and its grasp is no find")


def test_the_find_of_a_rattling_toy():
    """lane 19 (C116): a toy in the bucket rattles against the hand that reaches in: on life day 29's first hide its hand was on the hidden
    car at ticks +152, +154 and +156, shaking it, never three ticks running, and no find was judged. The find is its hand on the hidden
    toy on FOUND_TOUCHES of the last FOUND_WINDOW ticks: a duck hidden by her hand, its hand's touch on alternate ticks: "found", worth
    2, the duck hidden no more; one touch alone: no find"""
    import mujoco
    from body.sim import extras as X
    w = W.G1World(seed=1, extra=X.add_bucket(xy=(-0.3, -0.45)))
    lane = L.ParentLane(w, seed=1, voice=FakeVoice(), plan=False, day_ticks=2400)
    _run(w, 4)
    m, d = w.m, w.d
    bucket = m.body("toy_bucket").id; jd = m.body("toy_duck").jntadr[0]; a = m.jnt_qposadr[jd]; va = m.jnt_dofadr[jd]
    w.parent.holding["R"] = "duck"
    for _ in range(3):
        w.frame(); w.apply({})
    d.qpos[a:a + 3] = d.xpos[bucket] + np.array([0.0, 0.0, X.BUCKET_WALL + 0.04]); d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
    d.qvel[va:va + 6] = 0.0; mujoco.mj_forward(m, d)
    w.parent.holding["R"] = None
    for _ in range(8):
        w.frame(); w.apply({})
    assert "duck" in lane.hidden, lane.hidden
    real = lane._contacts
    touch_now = {"on": False}

    def contacts(m_, d_):
        t_ = real(m_, d_)
        if touch_now["on"]:
            t_["duck"]["child"].add("left")
        return t_
    lane._contacts = contacts
    evs, judged = [], []
    touch_now["on"] = True; w.frame(); w.apply({}); evs += [tuple(e) for e in lane.last["events"]]      # one touch alone
    touch_now["on"] = False
    for _ in range(L.FOUND_WINDOW + 1):
        w.frame(); w.apply({}); evs += [tuple(e) for e in lane.last["events"]]
    assert ("found", "duck") not in evs and "duck" in lane.hidden, evs
    for k in range(6):                                                                                  # its touch on alternate ticks
        touch_now["on"] = k % 2 == 0
        w.frame(); w.apply({})
        evs += [tuple(e) for e in lane.last["events"]]; judged += [tuple(x) for x in (lane.last.get("judged") or ())]
    assert ("found", "duck") in evs and "duck" not in lane.hidden, (evs, lane.hidden)
    assert [j for j in judged if j[1] == "found"] == [(2, "found", "duck")], judged
    print("lane 19: a hidden duck touched once: no find; touched on alternate ticks (3 of 6): 'found', worth 2, hidden no more")


def test_the_crawl():
    """lane 17 (A125, the crawl rung): on its front, its pelvis carried CRAWL_M along the floor from where the prone spell began (or the
    last crawl counted) is the event "crawled", once in CRAWL_GAP, worth 2 in her book (consts.MOTOR_WORTH, the design's motor acts);
    a shorter move, or a move on its back, is none; the mark resets when it leaves its front"""
    import mujoco
    from body.sim import parent_motion as PM
    from body.sim.lang import consts as CK, percept as PC
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
    import sim_parent_motion as T
    assert CK.MOTOR_WORTH["crawled"][0] == 2 and "crawled" in PC.EVENT_KINDS
    w, lane = _world(plan=False, day_ticks=2400)
    m, d = w.m, w.d
    T.place_g1(w, "front")
    evs = []
    for k in range(30):
        w.frame(); w.apply({}); evs += [e[0] for e in lane.last.get("events", ())]
    assert lane.posture == "front", lane.posture
    ax = -PM.Child(m, d, w.scene.g1_set).len_axis[:2]
    d.qpos[0:2] += ax * 0.12; mujoco.mj_forward(m, d)                        # a short move: no crawl
    for k in range(10):
        w.frame(); w.apply({}); evs += [e[0] for e in lane.last.get("events", ())]
    assert "crawled" not in evs, evs
    d.qpos[0:2] += ax * 0.12; mujoco.mj_forward(m, d)                        # 0.24 m from the spell's start: a crawl
    for k in range(10):
        w.frame(); w.apply({}); evs += [e[0] for e in lane.last.get("events", ())]
    assert evs.count("crawled") == 1, evs
    d.qpos[0:2] += ax * 0.30; mujoco.mj_forward(m, d)                        # within CRAWL_GAP: not counted again yet
    for k in range(5):
        w.frame(); w.apply({}); evs += [e[0] for e in lane.last.get("events", ())]
    assert evs.count("crawled") == 1, evs
    for k in range(L.CRAWL_GAP):
        w.frame(); w.apply({}); evs += [e[0] for e in lane.last.get("events", ())]
    n_after_gap = evs.count("crawled")
    print(f"lane 17: a prone child carried 0.12 m: no crawl; 0.24 m: crawled (worth {CK.MOTOR_WORTH['crawled'][0]}); a third move within the gap not counted; after the gap {n_after_gap} crawls")

def test_floor_play_reaches_the_lesson():
    """lane 20 (C125): floor play asks a lesson with no focus toy before her eyes when she knows the place of one (C121's fallback,
    in motor time only before) or holds a toy (C122); with a toy in her hand she asks no peekaboo (it needs both her hands: refused
    "her hands are busy" 26 times on life day 28) and never asks the child to give her the toy she holds; her plan's hide turn and
    her conduct's comfort gap, cry window and look at the bucket are saved with them"""
    from types import SimpleNamespace
    from body.sim.lang import templates as TP
    w, lane = _world(plan=True, day_ticks=2400)
    for _k in range(5):
        w.frame(); w.apply({})
    c, plan = lane.conduct, lane.plan
    known = [o for o in (lane.toys or ()) if o not in TP.OPEN_CONTAINERS]
    assert len(known) >= 2, known
    plan.focus = known[:2]
    asked = []
    real_request, real_p, real_holding = c.request, lane._p, c.motion.holding
    try:
        c.request = lambda k, **kw: asked.append((k, kw))
        lane._p = SimpleNamespace(seen=[], events=[])                    # her eyes on the child: no toy before them
        n0 = len(plan.log)
        for k in range(200):
            plan._play(1000 + k, lane, "floor")
        lessons = [x for x in plan.log[n0:] if x[1] == "lesson"]
        assert 40 <= len(lessons) <= 120, len(lessons)                   # LESSON_SHARE of her offers (0.4 of 200)
        assert all(x[3] in plan.focus for x in lessons), lessons[:3]
        lane._p = SimpleNamespace(seen=[], events=[], child_holds=(plan.focus[0],))   # C132: the toy in ITS hand, out of her view
        n0 = len(plan.log)
        for k in range(120):
            plan._lesson(1500 + k, lane)
        held_by_it = [x for x in plan.log[n0:] if x[1] == "lesson" and x[3] == plan.focus[0]]
        assert not held_by_it and any(x[1] == "lesson" for x in plan.log[n0:]), held_by_it[:2]
        lane._p = SimpleNamespace(seen=[], events=[])
        peek = sum(1 for k, _kw in asked if k == "peekaboo_hide")
        assert peek > 0, "her peekaboo with her hands free, as before"
        held = plan.focus[0]
        c.motion.holding = {"L": held, "R": None}
        c.book.setdefault("got", {})[held] = 2 * DP.K.MASTERED_N         # the toy mastered through every rung: the give rung next
        for key in ("lifted", "shook", "hit"):
            c.book.setdefault(key, {})[held] = DP.K.MASTERED_N
        asked.clear(); n0 = len(plan.log)
        for k in range(200):
            plan._play(2000 + k, lane, "floor")
        lessons2 = [x for x in plan.log[n0:] if x[1] == "lesson"]
        assert not any(k == "peekaboo_hide" for k, _kw in asked), "no peekaboo with a toy in her hand"
        assert lessons2 and all(x[3] == held for x in lessons2), lessons2[:3]
        assert not any(x[2] == "give" for x in lessons2), [x for x in lessons2 if x[2] == "give"][:2]
        assert not any(k == "ask_give" and kw.get("o") == held for k, kw in asked)
        c.motion.holding = {"L": None, "R": None}
        worn, fresh_toy = plan.focus[0], plan.focus[1]                   # C141: a toy her smiles have worn out is offered only when no
        for key in ("got", "lifted", "shook", "hit"):                    # fresh one is at hand
            c.book.setdefault(key, {})[worn] = 60
        assert DP._worn(c.book, worn) and not DP._worn(c.book, fresh_toy)
        lane._p = SimpleNamespace(seen=[], events=[])
        n0 = len(plan.log)
        for k in range(60):
            plan._lesson(1700 + k, lane)
        on_worn = [x for x in plan.log[n0:] if x[1] == "lesson" and x[3] == worn]
        assert not on_worn and any(x[1] == "lesson" for x in plan.log[n0:]), on_worn[:2]
        c.book["got"][worn] = 2 * DP.K.MASTERED_N                        # the counts the give section below relies on, restored
        for key in ("lifted", "shook", "hit"):
            c.book[key][worn] = DP.K.MASTERED_N
        c.motion.holding = {"L": None, "R": None}                        # C126: the give asked only of a toy in the child's view and
        plan.focus = [held]                                              # reach (life day 29: 36 gives refused); else the toy into
        gives = {}                                                       # its hand (the handle act)
        for sees, reach in ((False, False), (True, False), (True, True)):
            lane._p = SimpleNamespace(seen=[SimpleNamespace(id=held, name=held, on="floor", child_sees=sees, child_can_reach=reach)],
                                      events=[])
            asked.clear(); n0 = len(plan.log)
            for k in range(60):
                plan._play(3000 + k, lane, "floor"); plan._lesson(3000 + k, lane)
            gives[(sees, reach)] = (sum(1 for k, _kw in asked if k == "ask_give"),
                                    sum(1 for x in plan.log[n0:] if x[1] == "lesson" and x[2] == "handle"))
        assert gives[(False, False)][0] == 0 and gives[(True, False)][0] == 0, gives
        assert gives[(False, False)][1] > 0 and gives[(True, True)][0] > 0 and gives[(True, True)][1] == 0, gives
    finally:
        c.request, lane._p, c.motion.holding = real_request, real_p, real_holding
    plan.hide_turn = True
    c.told_looked, c.cry_ticks, c.comfort_t, c.comfort_held = True, [5, 9], 7, 3
    pb, cb = pickle.loads(pickle.dumps(plan.state())), pickle.loads(pickle.dumps(c.state()))
    plan.hide_turn = False
    c.told_looked, c.cry_ticks, c.comfort_t, c.comfort_held = False, [], None, 0
    plan.load_state(pb); c.load_state(cb)
    assert plan.hide_turn is True
    assert (c.told_looked, c.cry_ticks, c.comfort_t, c.comfort_held) == (True, [5, 9], 7, 3)
    print(f"lane 20: floor play with no toy before her eyes: {len(lessons)} lessons of 200 offers on the toys she knows the place of",
          f"({peek} peekaboos, her hands free); the {held} in her hand: {len(lessons2)} lessons on it, no peekaboo, no give asked of it;",
          f"a give asked {gives[(True, True)][0]} times of a toy in its view and reach, never of one out of them",
          f"({gives[(False, False)][1]} handle lessons instead);",
          "her hide turn, comfort gap, cry window and look at the bucket saved and restored")


def test_the_leave_with_the_child_at_the_door():
    """lane 21 (C131): her away block is not entered while the child lies within DOOR_NEAR_M of the room's door (life day 32: it crawled
    into the doorway while she was out and she stood in the hall 5,000 ticks with no path back); with the child on the mat she leaves"""
    import mujoco
    w, lane = _world(plan=True, day_ticks=2400)
    for _ in range(5):
        w.frame(); w.apply({})
    plan = lane.plan
    c = lane.conduct
    asked = []
    real = c.request
    c.request = lambda k, **kw: asked.append(k)
    try:
        lane.last["child_xy"] = [2.2, -1.7]                                 # by the door
        plan.last_pain = -10 ** 6
        plan._enter("away", 1000, 1000, lane, w)
        assert "leave" not in asked and not plan.away and any("child at the door" in str(x) for x in plan.log[-2:]), (asked, plan.log[-2:])
        lane.last["child_xy"] = [0.3, -0.6]                                 # on the mat
        plan._enter("away", 1100, 1100, lane, w)
        assert "leave" in asked and plan.away, (asked, plan.away)
    finally:
        c.request = real
        plan.away = False
    print("lane 21: the away block skipped with the child 0.45 m from the door; entered with it on the mat")



def test_the_show_block_with_nothing_new():
    """lane 22 (C150, 2026-09-30): a show block whose growth words are all known and whose child attends nothing is play with a lesson:
    of 100 offers, lessons and shows of her focus toys (and calls) are asked, none a new word. Life day 40's last show block gave no act
    in 1,888 ticks"""
    from types import SimpleNamespace
    from body.sim.lang import templates as TP
    w, lane = _world(plan=True, day_ticks=2400)
    for _k in range(5):
        w.frame(); w.apply({})
    c, plan = lane.conduct, lane.plan
    known = [o for o in (lane.toys or ()) if o not in TP.OPEN_CONTAINERS]
    plan.focus = known[:2]
    asked = []
    real_request, real_p, real_vocab = c.request, lane._p, c.fast.vocab
    try:
        c.request = lambda k, **kw: asked.append((k, kw))
        c.fast.vocab = tuple(sorted(set(c.fast.vocab) | set(TP.GROWTH_WORDS)))    # every growth word known: nothing new to show
        lane._p = SimpleNamespace(seen=[], events=[], attended=lambda: [])       # the child attends nothing she reads
        n0 = len(plan.log)
        for k in range(100):
            plan._show(1000 + k, lane)
        kinds = [k for k, _kw in asked]
        lessons = [x for x in plan.log[n0:] if x[1] == "lesson"]
        assert not any(k == "new_word" for k in kinds) and not any(k == "ask_what" for k in kinds), kinds[:5]
        assert lessons and any(k in ("show", "set_near", "hand_over", "call") for k in kinds), (len(lessons), kinds[:6])
        assert all(x[3] in plan.focus for x in lessons), lessons[:3]
    finally:
        c.request, lane._p, c.fast.vocab = real_request, real_p, real_vocab
    print(f"lane 22 (C150): a show block with every growth word known and nothing attended: {len(lessons)} lessons and "
          f"{sum(1 for k in kinds if k == 'show')} shows of her focus toys in 100 offers, no new word asked")



def test_the_hide_with_the_bucket_out_of_reach():
    """lane 23 (C152, 2026-09-30): the hide rung asked with the bucket seen but beyond the child's reach (C138 brings it), and with the
    bucket unseen but its place known; not with no bucket at all; the hide turn set after a handle lesson with the bucket at hand.
    Life days 38 to 41: 0 to 1 hide a day under C130's gate, no find ever"""
    from types import SimpleNamespace
    from body.sim.lang import templates as TP
    w, lane = _world(plan=True, day_ticks=2400)
    for _k in range(5):
        w.frame(); w.apply({})
    c, plan = lane.conduct, lane.plan
    known = [o for o in (lane.toys or ()) if o not in TP.OPEN_CONTAINERS]
    toy = known[0]; plan.focus = [toy]
    asked = []
    real_request, real_p, real_toys = c.request, lane._p, lane.toys
    try:
        c.request = lambda k, **kw: asked.append((k, kw))
        c.book.setdefault("got", {})[toy] = 2 * DP.K.MASTERED_N            # the reach rung mastered: the hide's turn
        def S(id_, reach): return SimpleNamespace(id=id_, name=id_, on="floor", child_sees=True, child_can_reach=reach)
        plan.hide_turn = True
        lane._p = SimpleNamespace(seen=[S("bucket", False), S(toy, True)], events=[], child_holds=())
        plan._lesson(100, lane)
        assert asked and asked[-1][0] == "hide" and asked[-1][1].get("o") == toy, asked[-2:]
        plan.hide_turn = True
        lane._p = SimpleNamespace(seen=[S(toy, True)], events=[], child_holds=())    # the bucket unseen, its place known
        lane.toys = tuple(real_toys) + (("bucket",) if "bucket" not in set(real_toys) else ())   # (the test's room has no bucket body)
        plan._lesson(200, lane)
        assert asked[-1][0] == "hide", asked[-2:]
        plan.hide_turn = True
        lane.toys = tuple(o for o in lane.toys if o != "bucket")                    # no bucket she knows of: no hide
        plan._lesson(300, lane)
        assert asked[-1][0] != "hide", asked[-2:]
        lane.toys = tuple(real_toys) + (("bucket",) if "bucket" not in set(real_toys) else ())   # the bucket's place known again
        plan.hide_turn = False
        c.book["got"][toy] = 2 * DP.K.MASTERED_N                                   # mastered, the hide's turn spent: the handle rung
        lane._p = SimpleNamespace(seen=[S(toy, True)], events=[], child_holds=())    # (its lesson sets the turn with the bucket at hand)
        plan._lesson(400, lane)
        assert asked[-1][0] == "hand_over" and plan.hide_turn is True, (asked[-1], plan.hide_turn)
    finally:
        c.request, lane._p, lane.toys = real_request, real_p, real_toys
    print(f"lane 23 (C152): the hide asked with the bucket seen beyond its reach and with the bucket's place known, none with no bucket; "
          f"the hide turn set by a handle lesson with the bucket at hand")


LANE_TESTS = [test_the_tables, test_a_line_heard, test_exact_replay_mid_line, test_the_night, test_the_born_reading, test_a_toy_falls,
              test_the_days_layout, test_a_short_day, test_no_meal, test_her_eyes, test_her_lessons, test_smile_brought, test_a_face_down_morning,
              test_the_roll_rung, test_a_toy_she_could_not_get_to, test_the_new_toy_in_her_focus, test_the_crawl,
              test_the_find, test_the_find_of_a_rattling_toy, test_floor_play_reaches_the_lesson, test_the_leave_with_the_child_at_the_door, test_the_show_block_with_nothing_new, test_the_hide_with_the_bucket_out_of_reach]

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


def test_the_talk_over_frown_retired():
    """lane 11 (C231, 2026-10-02): the talk-over frown is nothing for the babbling child: her feelings keep the smile she had judged, show no
    frown and lose no mood when talked over; a hit still frowns (stage 2's harm as it was)"""
    from body.sim import parent_feel as PF
    f = PF.Feelings(1)
    for _ in range(3):
        f.step(False)
    f.judge(1.0, "act")
    had = (f.pulse is not None) or (f.queued is not None); M0 = f.M
    f.talk_over()
    assert had and ((f.pulse is not None) or (f.queued is not None)), "the talk-over cancelled her smile"
    assert not f.frown_active() and f.M == M0, (f.frown_active(), f.M, M0)
    assert not any(e[1] == "frown" for e in f.log), f.log[-4:]
    f.harm()
    assert f.frown_active() and f.frown_amp == PF.FEEL["frown_hit"], (f.frown_active(), f.frown_amp)
    print("LANE 11 GREEN")
