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
        assert n["floor"] == 3 and n["motor"] == 2 and n["show"] == 1 and n["tasks"] == 1 and 2 <= n["away"] <= 4, n
        counts.add(n["away"])
        assert 2 <= len(d.focus) <= 3 and set(d.focus) <= set(DP.BIRTH_TOYS)
    d = DP.DayPlan(1, day_ticks=2400); d.lay_out(0)
    assert d.blocks[0][0] == 30 and d.blocks[-1][1] == 2300 and d.episode(2299) != "wind"
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
        if lane.reading > 0:
            break
    assert seen_at and lane.reading > 0, (seen_at, lane.reading, lane.feel.log[-6:])
    assert lane.conduct.book["got"] == {"ball": 1} and set(lane.conduct.book) <= {"got", "lifted", "hit"}, lane.conduct.book   # in a
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


LANE_TESTS = [test_the_tables, test_a_line_heard, test_exact_replay_mid_line, test_the_night, test_the_born_reading, test_a_toy_falls,
              test_the_days_layout, test_a_short_day, test_no_meal, test_her_eyes, test_her_lessons, test_smile_brought]

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
