"""the parent's motion (docs/SIM_DESIGN.md 4.1, 4.2, 4.10, A3-A10, A22, A25; the build plan's W2; body/sim/parent_motion.py and
parent_consts.py). Run: python3 -m body.tests.test_sim_parent (at nice -n 19; a few minutes on this Mac).
  THE SCENE: no weld on the G1 (a world with one refuses to be born), no contact exclusion between her and it, her collision shapes
  soft (solref 0.05) at contact priority 2 so her softness is every contact's with the G1; her face's spline table read bit for
  bit as before its cache; her caps as checked against their sources (NIOSH's 35 lb: 156 N; 200 N brief; one hand 100 / 150 N).
  THE CAPPED SPRING: a hold pulled far past its cap applies exactly its cap on every step, as an outside force at the held point;
  her own caps bound her holds together (one hand 100 N, both 156 N, or 150 / 200 N for at most 2 s, then the sustained caps until
  she has rested 2 s); its force is felt on the held link's touch zone; a hold at its cap for 2 ticks stops (the guide).
  THE INTERFACE (P3's StubMotion's: request, status, cancel, state): an unknown act refused with its reason; the gaze and the body
  on their own channels, each in the order asked; a look during a focus word held FOCUS_TICKS; a cancel mid-act lets a transition
  finish; pruned acts keep their final status.
  HER ACTS ON THE G1: attend (she comes to it, kneels beside it and rests a hand on its trunk at TOUCH_N), lean_in (A1's face test
  passes with the child's fovea on her mouth, 25 cm or more from its eyes, 15 deg or more off its fovea's born line: C34), the guide
  (its cap the arm's own push x 1.5, at most 100 N; its peak within it), the pull-to-sit's pull placed within reach (stopped at her
  cap, laid back, the G1 never sat up, lifted or slid), the prop (the trunk held within 30 deg at her sustained caps; the catch 2
  ticks after the trunk passes 35 deg from hovering), the brief turn (at most 2 s, within 200 N, no slide); in every one her body
  stops and backs off within 2 steps of a contact over a resting hand's weight, and no act lifts, slides or sits the G1 up.
  TOYS: the block handed into its hand ends in its hand; a toy the child touches is never taken (A4).
  HER PACE: no segment of hers moves faster than a person does (walking, kneeling down: parent_consts.KNEEL_SEG_MPS; KNEEL_PATH as
  measured on parent_poses.kneel_down); her idle cost a tick.
  THE EXACT REPLAY TEST WITH HER ACTING: a world saved while she comes to the child and restored in the same world and in a new one
  lives the same ticks bit for bit (every frame, her motion's state, the final save); and a save taken before one of her trunk
  solves, restored in a new world (her solve's wall time is never her state).
  THE W2 VERIFIER'S FINDINGS, each a test that would have caught it: under babble her force on the child stays under F_pain and
  her body backs off along the contact, never into it (14); from beside a still child she gets up and goes on without pressing on
  it, and a give-up never fails the next act (15); her hands on it never pass through its hulls (16); the save across a solve (17);
  the catch pushes at once at her brief caps and every fall is written down (18, C6); DOES and 'copy' for P3 (19); her caps bound
  her holds and her body's contacts together, friction counted (20); her lean-in within LEAN_DIST_M (5); an equality of any kind on
  the G1 refused (1); the brief caps given back only after BRIEF_REST_S (2)."""
import math
import os
import sys
import time

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "body", "sim"))
from body.sim import world as W  # noqa: E402
from body.sim import parent_motion as PM  # noqa: E402
K = PM.K                        # her constants: the very module her motion reads (a second import would be a copy)
import sim_parent_motion as T  # noqa: E402

G = W.G
kin = G.kin
Act = T.Act


def _live(w, n, until=None):
    for k in range(n):
        w.apply({})
        if until is not None and until(w):
            return k + 1
    return n


def _done(w, i):
    return lambda w_: w_.parent.status(i) in PM.DONE_STATES and not w_.parent.phases


def test_the_scene():
    """parent 1: no weld on the G1, no exclusion with her, her shapes soft at priority 2; the face table's cache bit for bit; her caps"""
    w = W.G1World(seed=1)
    m = w.m
    g1 = w.scene.g1_set
    assert not [e for e in range(m.neq) if m.eq_type[e] == mujoco.mjtEq.mjEQ_WELD and (m.eq_obj1id[e] in g1 or m.eq_obj2id[e] in g1)]
    assert m.nexclude == 0
    her = [g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.startswith("parent_") and m.geom_contype[g]]
    assert len(her) > 100 and all(m.geom_priority[g] == K.SOFT_PRIORITY and np.allclose(m.geom_solref[g], K.SOFT_SOLREF) for g in her)
    assert max(int(m.geom_priority[g]) for g in range(m.ngeom) if m.geom_bodyid[g] in g1) < K.SOFT_PRIORITY
    # a weld on the G1 refuses the world (4.1: every hold on it is a capped spring)
    def rig(spec):
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name="bad", name1="parent_hand_L", name2="torso_link", objtype=mujoco.mjtObj.mjOBJ_BODY)
    def rig_connect(spec):
        spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, name="bad", name1="parent_hand_L", name2="torso_link",
                          objtype=mujoco.mjtObj.mjOBJ_BODY)

    def rig_site(spec):
        spec.body("torso_link").add_site(name="bad_site", pos=[0.05, 0, 0.1])
        spec.body("parent_hand_L").add_site(name="bad_hand", pos=[0, 0, -0.1])
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name="bad", name1="bad_hand", name2="bad_site", objtype=mujoco.mjtObj.mjOBJ_SITE)

    def rig_world(spec):
        spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, name="bad", name1="torso_link", objtype=mujoco.mjtObj.mjOBJ_BODY)
    for r in (rig, rig_connect, rig_site, rig_world):                   # a weld, a connect, a weld by sites, a pin to the world
        try:
            W.G1World(seed=1, extra=r)
        except ValueError as e:
            assert "capped spring" in str(e), e
        else:
            raise AssertionError(f"an equality on the G1 was taken ({r.__name__})")
    # her face's table through the cached spline coefficients: the same numbers as map_coordinates' own prefilter, bit for bit
    from scipy.ndimage import map_coordinates
    face = kin.face
    rng = np.random.default_rng(3)
    y = rng.uniform(-0.09, 0.09, 400); z = rng.uniform(0.03, 0.26, 400)
    old = map_coordinates(face.front_table(), [(z - face.FRONT_Z[0]) / .001, (y - face.FRONT_Y[0]) / .001], order=3, mode="nearest")
    assert np.array_equal(old, face.head_surface_x(y, z))
    assert (K.CAP_ONE, K.CAP_ONE_BRIEF, K.CAP_TWO, K.CAP_TWO_BRIEF, K.BRIEF_S) == (100.0, 150.0, 156.0, 200.0, 2.0)
    assert abs(35 * 0.45359237 * 9.80665 - K.CAP_TWO) < 0.5             # NIOSH's 35 lb (Waters 2007), the source it came from
    print("parent 1: no equality on the G1 (a world with a weld, a connect, a weld by sites or a pin to the world refused), no",
          "contact exclusion, her", len(her), "collision shapes soft",
          "(solref 0.05) at priority 2; her face's table read bit for bit through the cached spline; her caps 100/150 N one hand,",
          "156/200 N both (NIOSH's 35 lb), 2 s brief")


def _spring_world():
    """the child as born, her placed kneeling beside it (instrument)"""
    w = W.G1World(seed=1)
    pm = w.parent
    ch = pm.child
    mid = (ch.torso[:2] + ch.pelvis[:2]) / 2
    H = mid + ch.lat[:2] * 0.84 + ch.len_axis[:2] * 0.10
    pm.place("heels", H, math.atan2(-ch.lat[1], -ch.lat[0]))
    return w


def _add_hold(w, name, body, side, cap, target, brief=False):
    pm = w.parent
    b = w.m.body(body).id
    h = PM.Hold(name, b, [0.0, 0.0, 0.0], side, cap, brief, "fixed", [0, 0, 1.0])
    h.ctl = dict(target=list(target), state="run", t=0)
    pm.holds.append(h)
    return h


def test_the_capped_spring():
    """parent 2: a hold pulled past its cap applies exactly its cap each step at the held point; her caps bound her holds together
    (one hand, both hands, the brief caps for 2 s then the sustained until rested); its force is felt on the held link's zone"""
    w = _spring_world()
    m, d = w.m, w.d
    pm = w.parent
    fa = d.xpos[m.body("left_elbow_link").id].copy()
    h = _add_hold(w, "t1", "left_elbow_link", "L", 40.0, fa + [0, 0, 1.0])
    z = w.zones.index("left_elbow")
    seen = []
    orig = pm._apply_holds

    def spy(s):
        orig(s)
        seen.append((float(np.linalg.norm(h.force)), float(np.linalg.norm(d.xfrc_applied[h.body, :3])), float(pm.hold_zone[z])))
    pm._apply_holds = spy
    _live(w, 3)
    assert all(abs(f - 40.0) < 1e-9 and abs(x - 40.0) < 1e-9 and abs(zf - 40.0) < 1e-9 for f, x, zf in seen), seen[:3]
    assert w.frame().truth["touch_N"][z] >= 40.0 - 1e-6                 # being held is felt (4.2)
    # her caps: two holds on one hand of 80 N each: 100 N together; a hold on the other hand too: 156 N all together
    pm._apply_holds = orig
    pm.holds = []
    h1 = _add_hold(w, "a", "left_elbow_link", "L", 80.0, fa + [0, 0, 1.0])
    h2 = _add_hold(w, "b", "torso_link", "L", 80.0, d.xpos[m.body("torso_link").id] + [0, 0, 1.0])
    _live(w, 2)
    assert abs(np.linalg.norm(h1.force) + np.linalg.norm(h2.force) - K.CAP_ONE) < 1e-6
    h3 = _add_hold(w, "c", "right_elbow_link", "R", 90.0, d.xpos[m.body("right_elbow_link").id] + [0, 0, 1.0])
    _live(w, 2)
    tot = sum(np.linalg.norm(x.force) for x in (h1, h2, h3))
    assert abs(tot - K.CAP_TWO) < 1e-6, tot
    # the brief caps: 100 + 100 N asked, brief: 200 N for 2 s, then 156 N until she has rested 2 s under it
    pm.holds = []
    pm.brief_s = 0.0
    b1 = _add_hold(w, "d", "left_elbow_link", "L", 100.0, fa + [0, 0, 1.0], brief=True)
    b2 = _add_hold(w, "e", "right_elbow_link", "R", 100.0, d.xpos[m.body("right_elbow_link").id] + [0, 0, 1.0], brief=True)
    tots = []
    for _ in range(18):
        _live(w, 1)
        tots.append(float(np.linalg.norm(b1.force) + np.linalg.norm(b2.force)))
    over = sum(1 for x in tots if x > K.CAP_TWO + 1e-6)
    assert abs(max(tots) - K.CAP_TWO_BRIEF) < 1e-6 and over * W.TICK_S <= K.BRIEF_S + W.TICK_S and abs(tots[-1] - K.CAP_TWO) < 1e-6, tots
    # rested 2 s (the burst's own length, the old rule) her brief caps are not hers again: the source's initial forces are for one
    # exertion every 5 min at most (BRIEF_REST_S)
    for h in (b1, b2):
        h.ctl["target"] = list(h.point(d))
    _live(w, int(round(2.5 / W.TICK_S)))
    for h in (b1, b2):
        h.ctl["target"] = list(h.point(d) + np.array([0, 0, 1.0]))
    _live(w, 3)
    assert abs(float(np.linalg.norm(b1.force) + np.linalg.norm(b2.force)) - K.CAP_TWO) < 1e-6 and K.BRIEF_REST_S >= 300.0
    pm.holds = []
    _live(w, 2)
    assert not np.any(d.xfrc_applied[list(w.scene.g1_set)])            # no force left behind when her holds end
    print(f"parent 2: a hold asked past its cap gave exactly 40 N on each step at the held point, felt on the link's zone; two",
          f"on one hand {K.CAP_ONE:g} N together, three on both hands {K.CAP_TWO:g} N; brief holds {K.CAP_TWO_BRIEF:g} N for {over} ticks",
          f"(at most {K.BRIEF_S:g} s), then {K.CAP_TWO:g} N; nothing left on the G1 once she let go")


def test_the_interface():
    """parent 3: P3's interface: unknown acts refused with a reason; two channels in order; a focus look held; cancel; pruning"""
    w = W.G1World(seed=1)
    pm = w.parent
    i = pm.request(Act("fly"))
    assert pm.status(i) == "refused" and "no such act" in pm.why(i)
    a = pm.request(Act("wave")); b = pm.request(Act("look", "child_eyes", "focus")); c = pm.request(Act("look", "child"))
    d_ = pm.request(Act("point", "drum"))
    assert [pm.status(x) for x in (a, b, c, d_)] == ["queued"] * 4
    _live(w, 1)
    assert pm.status(a) == "running" and pm.status(b) == "running" and pm.status(c) == "queued" and pm.status(d_) == "queued"
    _live(w, K.FOCUS_TICKS)
    assert pm.status(b) == "done" and pm.status(c) in ("running", "done")
    pm.cancel(d_)
    assert pm.status(d_) == "cancelled"
    _live(w, 60, until=_done(w, a))
    assert pm.status(a) == "done"
    e = pm.request(Act("walk", "door"))
    _live(w, 6)
    pm.cancel(e)
    assert pm.status(e) == "cancelled"
    n = _live(w, 20, until=lambda w_: not w_.parent.phases)
    assert not pm.phases and pm.base["mode"] in ("stand", "turn"), pm.base
    for _ in range(PM.KEEP_ACTS + 10):
        pm.request(Act("fly"))
    assert pm.status(i) == "refused" and len(pm.acts) <= PM.KEEP_ACTS + 1
    print("parent 3: an unknown act refused with its reason; the gaze and the body on their own channels in order; a focus look",
          f"held {K.FOCUS_TICKS} ticks; a queued act cancelled; a walk cancelled finishes its step ({n} ticks) and stands; old acts",
          "pruned keep their final status")


def test_attend():
    """parent 4: attend from the door: she comes to the child (A6), kneels beside it and rests a hand on its trunk at TOUCH_N; her
    body never pushes it (every contact of hers over a resting hand's weight stopped within 2 steps), nothing lifts or slides it"""
    w = W.G1World(seed=1)
    out = T.run(w, [("attend", None)], 400)
    a = out["acts"][0]
    assert a["status"] == "done", a
    hs = out["holds"]
    assert hs and all(h["peak_N"] <= K.TOUCH_N + 1e-6 for h in hs.values()), hs
    assert w.parent.holds and w.parent.holds[0].kind == "touch"
    assert out["g1"]["com_rise_cm"] < 1.0 and out["g1"]["pelvis_travel_cm"] < 2.0, out["g1"]
    assert out["jumps_refused"] == 0 and out["pen_max_mm"] < 15, out
    print(f"parent 4: attend: she came to the child and knelt beside it in {a['ticks']} ticks ({a['ticks'] * W.TICK_S:.0f} s; toys",
          f"cleared {a.get('cleared', [])}), her hand resting on its trunk at {max(h['peak_N'] for h in hs.values()):.1f} N (cap",
          f"{K.TOUCH_N:g}); her body's contacts {out['contact_peak_N']} N, deepest {out['pen_max_mm']} mm; the G1's centre of mass",
          f"rose {out['g1']['com_rise_cm']} cm, its pelvis moved {out['g1']['pelvis_travel_cm']} cm")


def test_lean_in():
    """parent 5: lean_in: her face where the child's eyes can reach (A22, C34): A1's face test passes with its fovea on her mouth,
    25 cm or more from its eyes, 15 deg or more off its fovea's line as it lies"""
    out = T.sc_lean_in()
    a = out["acts"][0]
    f = out["face"]
    assert a["status"] == "done", a
    assert all(v[0] for v in f["face_test_on_her_mouth"].values()), f
    assert min(f["mouth_to_eyes_m"].values()) >= K.FACE_MIN_M and f["off_born_line_deg"] >= K.FACE_OFF_LINE_DEG, f
    assert K.LEAN_DIST_M[0] <= min(f["mouth_to_eyes_m"].values()) <= K.LEAN_DIST_M[1], f   # the lean-in distance (the verifier's ninth)
    print(f"parent 5: lean_in in {a['ticks']} ticks: her face ({f['plan']['mode']}, lean {f['plan']['lean']:g}, spine",
          f"{f['plan']['spine']:g}, twist {f['plan']['twist']:g}) {f['mouth_to_eyes_m']} m from its eyes, {f['off_born_line_deg']} deg off",
          f"its fovea's born line; A1's face test passes in both eyes with the fovea put on her mouth")


def test_the_guide():
    """parent 6: the guide (A10): its cap min(1.5 x the arm's own push at that pose, 100 N); its peak within it; the wrist raised.
    The far knee bent over (A8) from beside its hips: within 76 N, stopped when it sat there 2 ticks; nothing lifted or slid"""
    out = T.sc_guide(None)()
    a = out["acts"][0]
    assert a["status"] == "done", a
    cap = min(K.GUIDE_CAP_FACTOR * a["limb_push"], K.CAP_ONE)
    assert abs(a["guide_cap"] - cap) < 1e-3
    pk = max(h["peak_N"] for n, h in out["holds"].items() if "guide" in n)
    assert pk <= cap + 1e-6 and out["held_point_moved_cm"][2] > 3.0, (pk, cap, out["held_point_moved_cm"])
    assert out["g1"]["held"]["pelvis_travel_cm"] < 1.0
    ko = T.SCENARIOS["knee_over"]()
    k = ko["acts"][0]
    kp = max((h["peak_N"] for n, h in ko["holds"].items() if "knee" in n), default=0.0)
    assert kp > 0.0 and kp <= K.ROLL_KNEE_N + 1e-6 and (k["status"] == "done" or "its cap" in k["why"]), (k, ko["holds"])
    assert ko["g1"]["held"]["com_rise_cm"] < 1.0 and ko["g1"]["held"]["pelvis_travel_cm"] < 1.0, ko["g1"]
    print(f"parent 6: the guide: the arm's own push {a['limb_push']:.0f} N at that pose, its cap {cap:.0f} N, its peak {pk:.0f} N;",
          f"the forearm raised {out['held_point_moved_cm'][2]} cm at {K.GUIDE_SPEED:g} m/s; the G1 not moved. The far knee from",
          f"beside its hips: {k['status']} ({k['why'][:40]}), peak {kp:.0f} N of {K.ROLL_KNEE_N:.0f}; its pelvis moved",
          f"{ko['g1']['held']['pelvis_travel_cm']} cm")


def test_the_pull_never_sits_it_up():
    """parent 7: the pull-to-sit's pull (placed within reach: in the world no spot reaches both forearms, C7) grows to her brief cap,
    sits there 2 ticks, stops and lays the G1 back: it never rose, and was never lifted or slid"""
    out = T.sc_pull_placed()
    a = out["acts"][0]
    assert a["status"] == "refused" and "her cap" in a["why"], a
    assert out["effort_peak_N"] <= K.CAP_TWO_BRIEF + 1e-6 and out["over_sustained_s"] <= K.BRIEF_S + W.TICK_S, out
    g = out["g1"]["held"]
    assert g["trunk_min_deg"] > 60.0 and g["com_rise_cm"] < 2.0 and g["pelvis_travel_cm"] < 2.0, g
    w = W.G1World(seed=1)
    out2 = T.run(w, [("pull_to_sit", None)], 500)
    assert out2["acts"][0]["status"] == "refused", out2["acts"]
    print(f"parent 7: the pull (placed): her effort peaked at {out['effort_peak_N']} N ({out['over_sustained_s']} s over the sustained",
          f"{K.CAP_TWO:g} N), stopped at her cap and laid it back: its trunk stayed {g['trunk_min_deg']}-{g['trunk_max_deg']} deg from",
          f"vertical, its centre of mass rose {g['com_rise_cm']} cm, its pelvis moved {g['pelvis_travel_cm']} cm; asked in the world:",
          f"'{out2['acts'][0]['why'][:70]}'")


def test_the_prop_and_the_catch():
    """parent 8: the prop (A9) holds a trunk placed within 30 deg at her sustained caps; from hovering, the catch engages 2 ticks
    (her reaction) after the trunk passes 35 deg"""
    w = W.G1World(seed=1)
    T.place_g1(w, "sit")
    pm = w.parent
    ch = pm.child
    fr = PM.unit(ch.torso_R[:, 0] * [1, 1, 0])[:2]
    lat = np.array([-fr[1], fr[0]])
    pm.place("heels", ch.pelvis[:2] + lat * 0.75 - fr * 0.10, math.atan2(-lat[1], -lat[0]))
    i = pm.request(Act("prop"))
    n = _live(w, 80, until=lambda w_: len([h for h in w_.parent.holds if h.kind == "prop"]) == 2 and
              all(h.ctl.get("go") for h in w_.parent.holds))
    hs = [h for h in pm.holds if h.kind == "prop"]
    assert len(hs) == 2 and pm.status(i) == "running", (pm.status(i), pm.why(i))
    ths = []
    for _ in range(40):
        _live(w, 1)
        ths.append(pm.child.trunk_deg)
        assert sum(np.linalg.norm(h.force) for h in hs) <= K.CAP_TWO + 1e-6
    assert max(ths) <= K.CATCH_DEG, ths
    # hovering (the easing's last step, set by the instrument): the trunk left to fall; the catch 2 ticks after it passes 35 deg
    for h in hs:
        h.ctl.update(mode="hover", hover=K.HOVER_M); h.cap = 0.0
    crossed = caught = None
    for k in range(200):                                                # the resting law lets the trunk sink about 1 deg a second
        _live(w, 1)
        th = pm.child.trunk_deg
        if crossed is None and th > K.CATCH_DEG:
            crossed = k
        if caught is None and all(h.ctl.get("mode") in ("catch", "hold", "lay") for h in hs if pm._hold(h.name)):
            caught = k
        if caught is not None and k > caught + 6:
            break
    assert crossed is not None and caught is not None and caught - crossed == K.REACTION_TICKS, (crossed, caught)
    print(f"parent 8: the prop engaged with both hands after {n} ticks and held the trunk at {min(ths):.0f}-{max(ths):.0f} deg from",
          f"vertical within her sustained {K.CAP_TWO:g} N; hovering, the trunk passed {K.CATCH_DEG:g} deg at tick {crossed} and her hands",
          f"engaged {caught - crossed} ticks later ({K.REACTION_TICKS} ticks: her 300 ms reaction)")


def test_the_turn():
    """parent 9: the brief turn from its front (A7): at most 2 s of pushing, within 200 N, no slide or lift"""
    out = T.sc_turn()
    a = out["acts"][0]
    assert a["status"] in ("done", "refused"), a
    assert out["effort_peak_N"] <= K.CAP_TWO_BRIEF + 1e-6 and out["over_sustained_s"] <= K.BRIEF_S + W.TICK_S
    g = out["g1"]["held"]
    assert g["pelvis_travel_cm"] < 3.0 and g["com_rise_cm"] < 8.0, g
    print(f"parent 9: the brief turn: {a['status']} ('{a['why'][:60]}'), her effort peak {out['effort_peak_N']} N, over the sustained",
          f"cap {out['over_sustained_s']} s; while held its pelvis moved {g['pelvis_travel_cm']} cm, its centre of mass rose",
          f"{g['com_rise_cm']} cm; after: {out['posture_after']}")


def test_toys():
    """parent 10: the block handed into its open left hand (her placed beside it at birth, the block in her near hand): brought to its
    palm, released by A4's rule, never pressed on it past a resting hand's weight for more than 2 steps; a toy the child touches is
    never taken (A4). (From the door her hand-over finds a fist: at rest the born hands close on their own within 3 s, C41.)"""
    out = T.sc_hand_over(True)
    a = out["acts"][0]
    assert a["status"] == "done" and a.get("palm_N", 0.0) > K.HANDOVER_PALM_N and min(out["block_to_palm_m"].values()) < 0.10, \
        (a, out["block_to_palm_m"])
    w = W.G1World(seed=1)
    _live(w, 40)                                                       # the rattle by its right hand: its arm sinks onto it
    pm = w.parent
    assert pm._toy_clearance("rattle", np.zeros(3)) < 0.01
    i = pm.request(Act("show", "rattle"))
    _live(w, 3)
    assert pm.status(i) == "refused" and "never takes" in pm.why(i), pm.why(i)
    print(f"parent 10: the block handed over in {a['ticks']} ticks ({a['why']}; its hand on the block {a.get('palm_N', 0):.1f} N,",
          f"its fingers closed {a.get('closure_deg', 0):.0f} deg), {min(out['block_to_palm_m'].values()):.3f} m from its grasp point",
          f"30 ticks after (held: {out['block_held']}); the rattle its arm lies on refused (A4)")


def test_her_pace():
    """parent 11: no segment of hers moves faster than a person: walking to the door and kneeling down by the child, every
    segment's travel a tick within 0.6 m and her pelvis's within 0.25 m; KNEEL_PATH as measured on parent_poses.kneel_down"""
    w = W.G1World(seed=1)
    pm = w.parent
    worst = [0.0, 0.0]
    ids = [pm.request(Act("walk", "door")), pm.request(Act("walk", "child"))]
    prev = w.d.mocap_pos[pm.mocap_ids].copy()
    for _ in range(500):
        w.apply({})
        cur = w.d.mocap_pos[pm.mocap_ids].copy()
        mv = np.linalg.norm(cur - prev, axis=1)
        worst = [max(worst[0], float(mv[0])), max(worst[1], float(mv.max()))]
        prev = cur
        if all(pm.status(i) in PM.DONE_STATES for i in ids) and not pm.phases:
            break
    assert pm.stats.get("jumps_refused", 0) == 0 and worst[0] <= PM.MAX_JUMP_M and worst[1] <= PM.MAX_LIMB_JUMP_M, worst
    us = np.linspace(0.0, 0.25, 26)
    pos = np.array([[kin.fk(PM.P.kneel_down((0, 0), 0.0, u))[s][0] for s in kin.SEGS] for u in us])
    L = float(np.linalg.norm(np.diff(pos, axis=0), axis=2).max(axis=1).sum())
    assert abs(L - PM.KNEEL_PATH[1]) < 0.02, L
    print(f"parent 11: to the door and back to the child ({[pm.status(i) for i in ids]}): her pelvis moved at most {worst[0]:.3f} m",
          f"a tick, any segment at most {worst[1]:.3f} m; KNEEL_PATH's first quarter {PM.KNEEL_PATH[1]} m, measured again {L:.3f} m")


def test_exact_replay_with_her_acting():
    """parent 12: THE EXACT REPLAY TEST WITH HER ACTING: attend asked; 60 ticks; a save; 40 more; restored in the same world and in
    a new one, the 40 again: every frame, her motion's state and the final save bit for bit"""
    N, M = 60, 40
    w = W.G1World(seed=1)
    w.parent.request(Act("attend"))
    _live(w, N)
    blob = w.save_state()
    frames1 = []
    for _ in range(M):
        w.apply({}); frames1.append(w.frame())
    end1 = w.save_state()
    st1 = w.parent.state()
    for where in ("the same world", "a new world"):
        w2 = w if where == "the same world" else W.G1World(seed=1)
        w2.load_state(blob)
        frames2 = []
        for _ in range(M):
            w2.apply({}); frames2.append(w2.frame())
        for x, y in zip(frames1, frames2):
            assert x.tick == y.tick and all(np.array_equal(x.obs[k], y.obs[k]) for k in x.obs), where
            assert x.truth["parent"] == y.truth["parent"], where
        assert w2.save_state() == end1, where
        assert w2.parent.state() == st1, where
    print(f"parent 12: the exact replay with her acting: attend asked, {N} ticks (she was {w.parent.base['mode']} by then), a save",
          f"({len(blob) / 1e3:.0f} KB); {M} more restored in the same world and in a new one: every frame, her motion and the final",
          "save bit for bit")


def test_her_cost():
    """parent 13: idle, her motion draws nothing (no pose is computed, no mocap written); its cost a tick written down (this Mac,
    under its load)"""
    w = W.G1World(seed=1)
    pm = w.parent
    calls = [0]
    orig = pm._pose

    def counting(*a, **k):
        calls[0] += 1
        return orig(*a, **k)
    pm._pose = counting
    m0 = w.d.mocap_pos.copy()
    t = []
    for _ in range(20):
        p0 = w.timing["parent_s"]; w.apply({}); t.append((w.timing["parent_s"] - p0) * 1e3)
    assert calls[0] == 0 and np.array_equal(m0, w.d.mocap_pos), calls
    idle = float(np.median(t))
    print(f"parent 13: idle at the door her motion computed no pose and moved nothing; it cost {idle:.2f} ms a tick (median; load",
          f"average {os.getloadavg()[0]:.1f})")


def test_her_yield_under_babble():
    """parent 14 (the W2 verifier's first and third findings): the babbling G1 (the verifier's case: babbler seed 4, p_rest 0.3),
    attend asked over and over for 300 ticks: her force on any link of the child as a 10 ms mean never reaches F_pain (C8: "under
    F_pain always"); her body backs off along each contact, away from the child, never into it (every tick her body yields, its
    way out points away from the child's contacts on her); no planning fault; she gives up at most half of what is asked"""
    w = W.G1World(seed=1)
    pm = w.parent
    probe = T.Probe(w)
    b = T.babbler(4, 0.3)
    orig = getattr(pm, "_yield_dir", None)
    bad = []

    def spy(c, away):
        y = orig(c, away)
        if y is not None and c == "core" and float(np.linalg.norm(away[:2])) >= K.YIELD_FLAT_SHARE * float(np.linalg.norm(away)) > 0:
            if float(y @ PM.unit(away * [1, 1, 0])) < 0.999:
                bad.append(float(y @ PM.unit(away * [1, 1, 0])))
        return y
    if orig is not None:
        pm._yield_dir = spy
    asked = []
    for k in range(300):
        if not asked or (pm.status(asked[-1]) in PM.DONE_STATES and not pm.phases):
            asked.append(pm.request(Act("attend")))
        w.apply(b.acts())
    r = probe.result()
    given_up = sum(1 for i in asked if pm.why(i).startswith("given up"))
    assert r["ticks_over_f_pain"] == 0, r
    assert not bad, bad[:5]
    assert pm.stats.get("jumps_refused", 0) == 0 and given_up * 2 <= len(asked), (given_up, len(asked), [pm.why(i) for i in asked])
    print(f"parent 14: under babble (seed 4, p_rest 0.3), attend asked {len(asked)} times in 300 ticks, {given_up} given up; her force",
          f"on the child at most {r['her_10ms_on_child_N']} N as a 10 ms mean (F_pain {w.f_pain:.0f} N; over her own 150 N on",
          f"{r['ticks_over_her_150N']} of 300 ticks); her body backed off along every contact; no planning fault")


def test_getting_up_beside_it():
    """parent 15 (the W2 verifier's second finding): a still child: she gets up from beside it without pressing on it (a shuffle
    back, or a turn on her knees, planned against it) and goes on: attend then the door, lean_in then bring_back, and in two other
    placements attend, lean_in and show; her body never pressed it over a resting hand's weight. A stale give-up never fails the
    next act"""
    for where, seq in ((None, [("attend", None), ("walk", "door")]), (None, [("lean_in", None), ("bring_back", "car")]),
                       ("rot90", [("attend", None), ("lean_in", None), ("show", "ball")]),
                       ("wall", [("attend", None), ("lean_in", None), ("show", "block")])):
        w = W.G1World(seed=1)
        if where is not None:
            T.place_child(w, where)
        out = T.run(w, seq, 900 * len(seq))
        assert all(a["status"] == "done" for a in out["acts"]), (where, out["acts"])
        assert out["probe"]["body_peak_N"] <= K.TOUCH_N and out["jumps_refused"] == 0, (where, out["probe"])
    # a push that gave up an act is that act's alone: the next is not failed by it
    w = W.G1World(seed=1)
    pm = w.parent
    i = pm.request(Act("attend"))
    _live(w, 30)
    pm.blocked = pm.cur["body"]                                         # as a push she backed off 12 cm from sets it
    _live(w, 2)
    assert pm.status(i) == "refused" and "given up" in pm.why(i), pm.why(i)
    j = pm.request(Act("walk", "door"))
    _live(w, 400, until=_done(w, j))
    assert pm.status(j) == "done", pm.why(j)
    print("parent 15: from beside the still child she got up without pressing on it and went on (attend, the door; lean_in,",
          "bring_back; rotated 90 deg and by the wall: attend, lean_in, show), her body's force on it at most a resting hand's; an",
          "act after one given up ran")


def test_her_hands_stay_out():
    """parent 16 (the W2 verifier's fourth finding): her hands on the child (resting, guiding, propping, pulling) never pass
    through it: her palm, fingers and thumb stay outside the G1's convex hulls (which enclose its drawn meshes) at four steps a tick,
    within 1 mm (MuJoCo's own distance)"""
    rows = {}
    T.HANDS = True
    try:
        for n in ("attend", "guide", "prop", "pull_placed"):
            r = T.SCENARIOS[n]()
            rows[n] = (r["probe"]["hand_to_hull_mm"], r["probe"]["hand_worst"])
            assert rows[n][0] is not None and rows[n][0] >= -1.0, (n, rows[n])
    finally:
        T.HANDS = False
    print("parent 16: her hands on the child, least distance to its hulls:", {k: v[0] for k, v in rows.items()}, "mm")


def test_exact_replay_across_a_solve():
    """parent 17 (the W2 verifier's fifth finding): a save taken while she shows a toy, before her trunk is solved again: restored
    in a new world, every frame, her state and the final save bit for bit (her solve's wall time is an instrument's, never her
    state)"""
    w = W.G1World(seed=1)
    pm = w.parent
    pm.request(Act("show", "block"))
    solves = []
    orig = pm._solve_trunk

    def counting(*a, **k):
        solves.append(w.tick)
        return orig(*a, **k)
    pm._solve_trunk = counting
    _live(w, 200, until=lambda w_: bool(solves))                        # up to her first trunk solve
    blob = w.save_state()
    t0 = w.tick
    frames1 = [None] * 30
    for k in range(30):
        w.apply({}); frames1[k] = w.frame()
    assert any(t > t0 for t in solves), solves                          # a solve after the save
    end1 = w.save_state(); st1 = pm.state()
    w2 = W.G1World(seed=1)
    w2.load_state(blob)
    for k in range(30):
        w2.apply({})
        f = w2.frame()
        assert all(np.array_equal(frames1[k].obs[x], f.obs[x]) for x in f.obs) and frames1[k].truth["parent"] == f.truth["parent"]
    assert w2.save_state() == end1 and w2.parent.state() == st1
    print(f"parent 17: a save at tick {t0} while she showed the block, her trunk solved again at ticks {[t for t in solves if t > t0][:3]}:",
          "restored in a new world, 30 ticks bit for bit")


def test_the_catch_pushes_at_once():
    """parent 18 (the W2 verifier's sixth finding; C6): from hovering, 2 ticks after the trunk passes 35 deg her hands' springs pull
    at once toward where the trunk sat most upright (never merely holding where her hands met it), up to her brief caps, and
    the fall is written down with its depth and its end (C6's count: tools/sim_parent_motion.py catch)"""
    w = W.G1World(seed=1)
    T.place_g1(w, "sit")
    pm = w.parent
    ch = pm.child
    fr = PM.unit(ch.torso_R[:, 0] * [1, 1, 0])[:2]
    lat = np.array([-fr[1], fr[0]])
    pm.place("heels", ch.pelvis[:2] + lat * 0.75 - fr * 0.10, math.atan2(-lat[1], -lat[0]))
    pm.request(Act("prop"))
    _live(w, 80, until=lambda w_: len([h for h in w_.parent.holds if h.kind == "prop"]) == 2 and
          all(h.ctl.get("go") for h in w_.parent.holds))
    hs = [h for h in pm.holds if h.kind == "prop"]
    for h in hs:
        h.ctl.update(mode="hover", hover=K.HOVER_M); h.cap = 0.0
    first = None
    for k in range(250):
        met = {h.name: h.point(w.d).copy() for h in hs}                 # where her hands meet it as the tick begins
        _live(w, 1)
        live = [h for h in hs if pm._hold(h.name)]
        if first is None and live and all(h.ctl.get("mode") == "catch" for h in live):
            on_ref = all(np.allclose(h.t1, h.ctl["ref"]) for h in live)
            gap = min(float(np.linalg.norm(np.asarray(h.ctl["ref"]) - met[h.name])) for h in live)
            first = (sum(float(np.linalg.norm(h.force)) for h in live), on_ref, gap)
        if pm.stats.get("falls"):
            break
    assert first is not None and first[1] and first[2] > 0.01, first
    f = pm.stats["falls"][0]
    assert f["how"] and f["peak_deg"] >= K.CATCH_DEG, f
    print(f"parent 18: the catch's springs pulled toward where the trunk sat most upright, {100 * first[2]:.1f} cm from where",
          f"her hands met it ({first[0]:.0f} N by the tick's end; her brief cap {K.CAP_TWO_BRIEF:g} N); the fall written down:", f)


def test_the_interface_does_and_copies():
    """parent 19 (the W2 verifier's seventh finding): P3's conduct reads her motion's DOES; every kind in it is done through 'do'
    (never refused standing at the door), and 'copy' makes each movement P3 copies (A52) with the hand named"""
    does = getattr(PM.ParentMotion, "DOES", ())                         # what P3's conduct reads (getattr(motion, "DOES", ()))
    assert {"wave", "clap", "stand", "walk", "open_hand"} <= set(does), does
    for kind, tgt in [("do", k) for k in does] + [("copy", f"{k}:{s_}") for k in ("arm_raise", "wave", "shake", "open_hand")
                                                      for s_ in ("left", "right")]:
        w = W.G1World(seed=1)
        out = T.run(w, [(kind, tgt)], 700)
        a = out["acts"][0]
        assert a["status"] == "done", (kind, tgt, a)
    w = W.G1World(seed=1)
    i = w.parent.request(Act("do", "hug")); j = w.parent.request(Act("copy", "kick:left"))
    _live(w, 2)
    assert w.parent.status(i) == "refused" and w.parent.status(j) == "refused"
    print(f"parent 19: DOES {does} each done through 'do'; copy's arm_raise, wave, shake and open_hand done with either hand;",
          "a kind she cannot show refused with its reason")


def test_her_caps_count_her_body():
    """parent 20 (the W2 verifier's eighth finding): her caps bound her holds and her body's contacts together, friction
    included: in the placed pull and the brief turn her whole force on the G1 never passes the brief cap, and passes the
    sustained cap for at most BRIEF_S"""
    for name in ("turn", "pull_placed"):
        out = T.SCENARIOS[name]()
        p_ = out["probe"]
        assert p_["total_steps_over_brief_cap"] == 0 and p_["total_steps_over_sustained_cap"] * 0.002 <= K.BRIEF_S + 1e-9, (name, p_)
    print("parent 20: in the pull and the turn her holds and her body together stayed within her caps (friction counted)")


PARENT_TESTS = [test_the_scene, test_the_capped_spring, test_the_interface, test_attend, test_lean_in, test_the_guide,
                test_the_pull_never_sits_it_up, test_the_prop_and_the_catch, test_the_turn, test_toys, test_her_pace,
                test_exact_replay_with_her_acting, test_her_cost, test_her_yield_under_babble, test_getting_up_beside_it,
                test_her_hands_stay_out, test_exact_replay_across_a_solve, test_the_catch_pushes_at_once,
                test_the_interface_does_and_copies, test_her_caps_count_her_body]

if __name__ == "__main__":
    t0 = time.time(); failed = 0
    only = sys.argv[1:]
    tests = [t for t in PARENT_TESTS if not only or any(o in t.__name__ for o in only)]
    for t in tests:
        t1 = time.time()
        try:
            t()
        except AssertionError as e:
            failed += 1; print("FAIL", t.__name__, ":", str(e)[:500])
        except Exception as e:
            failed += 1; print("ERROR", t.__name__, ":", type(e).__name__, str(e)[:500])
        print(f"   ({time.time() - t1:.0f} s)")
    print(f"{len(tests) - failed}/{len(tests)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if failed else 0)
