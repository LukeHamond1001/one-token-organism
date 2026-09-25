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
  the G1 refused (1); the brief caps given back only after BRIEF_REST_S (2).
  ITS SECOND ROUND: C8 on the verifier's babble seeds 1, 5, 6, 8 with its act mix, her body's work on the G1 never enough to slide
  or lift it, her way out never toward the child, no planning fault (14); a tendon rope or tendon equality on the G1 refused and her
  palm, fingers and thumb touching the G1 (1); every hand, whatever it does, against the G1's hulls on the still child and under
  babble (16); the pull-to-sit from its feet as A9 has it (7); a save in the middle of a push under babble, her report read every
  tick, replayed bit for bit (12); P3's contract (21: statuses, report(), cancel(id, tick), Act.thing, eyes_on_child)."""
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

    def _rope(spec, name, a, b):                                        # a spatial tendon between two sites
        spec.body(a[0]).add_site(name=f"{name}_a", pos=a[1]); spec.body(b[0]).add_site(name=f"{name}_b", pos=b[1])
        t = spec.add_tendon(name=name, limited=True, range=[0.0, 0.5])
        t.wrap_site(f"{name}_a"); t.wrap_site(f"{name}_b")

    def rig_rope(spec):                                                 # a rope from the room to the G1 (the W2 verifier's finding)
        _rope(spec, "bad", ("world", [0, 0, 1.0]), ("torso_link", [0, 0, 0.1]))

    def rig_tendon_eq(spec):                                            # a tendon in the G1 tied by an equality to one outside it
        _rope(spec, "in_g1", ("torso_link", [0, 0, 0.1]), ("pelvis", [0, 0, 0]))
        _rope(spec, "out", ("world", [0, 0, 1.0]), ("world", [0.1, 0, 1.0]))
        spec.add_equality(type=mujoco.mjtEq.mjEQ_TENDON, name="bad", name1="in_g1", name2="out", objtype=mujoco.mjtObj.mjOBJ_TENDON)
    for r in (rig, rig_connect, rig_site, rig_world, rig_rope, rig_tendon_eq):   # a weld, a connect, a weld by sites, a pin to
        try:                                                            # the world, a rope, a tendon equality
            W.G1World(seed=1, extra=r)
        except ValueError as e:
            assert "capped spring" in str(e), e
        else:
            raise AssertionError(f"an equality or tendon tying the G1 was taken ({r.__name__})")
    # her palm, fingers and thumb touch the G1 (and only it), soft as the rest of her (the W2 verifier's finding)
    for sd in "LR":
        fg = w.parent.finger_geoms[sd]
        assert len(fg) >= 11 and all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 1 and m.geom_priority[g] == K.SOFT_PRIORITY
                                     and np.allclose(m.geom_solref[g], K.SOFT_SOLREF) for g in fg), sd
    # her face's table through the cached spline coefficients: the same numbers as map_coordinates' own prefilter, bit for bit
    from scipy.ndimage import map_coordinates
    face = kin.face
    rng = np.random.default_rng(3)
    y = rng.uniform(-0.09, 0.09, 400); z = rng.uniform(0.03, 0.26, 400)
    old = map_coordinates(face.front_table(), [(z - face.FRONT_Z[0]) / .001, (y - face.FRONT_Y[0]) / .001], order=3, mode="nearest")
    assert np.array_equal(old, face.head_surface_x(y, z))
    assert (K.CAP_ONE, K.CAP_ONE_BRIEF, K.CAP_TWO, K.CAP_TWO_BRIEF, K.BRIEF_S) == (100.0, 150.0, 156.0, 200.0, 2.0)
    assert abs(35 * 0.45359237 * 9.80665 - K.CAP_TWO) < 0.5             # NIOSH's 35 lb (Waters 2007), the source it came from
    print("parent 1: no equality on the G1 (a world with a weld, a connect, a weld by sites, a pin to the world, a rope or a tendon",
          "equality refused), her palm, fingers and thumb touching the G1 alone, soft; no",
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
    assert [pm.status(x) for x in (a, b, c, d_)] == ["running"] * 4    # P3's contract: waiting its turn is 'running'
    _live(w, 1)
    assert pm.cur["body"] == a and pm.cur["gaze"] == b and c in pm.queue["gaze"] and d_ in pm.queue["body"]
    _live(w, K.FOCUS_TICKS)
    assert pm.status(b) == "done" and pm.status(c) in ("running", "done")
    pm.cancel(d_, w.tick + 1)
    assert pm.status(d_) == "cancelled"
    _live(w, 60, until=_done(w, a))
    assert pm.status(a) == "done"
    e = pm.request(Act("walk", "door"))
    _live(w, 6)
    pm.cancel(e, w.tick + 3)                                            # a cancel for a later tick waits for it: the act runs through
    _live(w, 2)                                                         # the next two ticks and is stopped in the third (the tick
    assert pm.status(e) == "running"                                    # whose report is that tick's)
    _live(w, 1)
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
    """parent 7 (the W2 verifier's sixth finding: A9 has her kneel at its feet, W2 knelt beside its hips and refused): asked in the
    world, she comes to its feet (clearing a toy there), kneels tall, takes both forearms, and the pull grows to her brief cap, sits
    there 2 ticks, stops and lays the G1 back: it never rose, and was never lifted or slid"""
    w = W.G1World(seed=1)
    out = T.run(w, [("pull_to_sit", None)], 700)
    a = out["acts"][0]
    assert a["status"] == "refused" and "her cap" in a["why"], a
    spot = w.parent.info(0)["info"].get("spot") or {}
    assert spot.get("where") == "feet", spot
    assert out["effort_peak_N"] <= K.CAP_TWO_BRIEF + 1e-6 and out["over_sustained_s"] <= K.BRIEF_S + W.TICK_S, out
    assert {h.split(" ")[0] for h in out["holds"]} == {"pull_L", "pull_R"}, out["holds"]
    g = out["g1"]["held"]
    assert g["trunk_min_deg"] > 60.0 and g["com_rise_cm"] < 2.0 and g["pelvis_travel_cm"] < 2.0, g
    print(f"parent 7: the pull-to-sit from its feet ({a['ticks']} ticks, toys cleared {a.get('cleared', [])}): her effort peaked at",
          f"{out['effort_peak_N']} N ({out['over_sustained_s']} s over the sustained {K.CAP_TWO:g} N), stopped at her cap and laid it",
          f"back: its trunk stayed {g['trunk_min_deg']}-{g['trunk_max_deg']} deg from vertical, its centre of mass rose",
          f"{g['com_rise_cm']} cm, its pelvis moved {g['pelvis_travel_cm']} cm")


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
    """parent 9: the brief turn from its front (A7), by its near shoulder and its near hip (the W2 verifier's finding: W2 lifted its
    pelvis at its middle, which rolls nothing): at most 2 s of pushing, within 200 N; it rolls toward its side (its chest turned
    from face down), and her whole force on it never carries its weight nor slides it (upward under its 337 N, along the floor
    under the 312 N that slides it on the mat, as a 10 ms mean): its pelvis moves only as the roll carries it"""
    out = T.sc_turn()
    a = out["acts"][0]
    assert a["status"] in ("done", "refused"), a
    assert out["effort_peak_N"] <= K.CAP_TWO_BRIEF + 1e-6 and out["over_sustained_s"] <= K.BRIEF_S + W.TICK_S
    assert a["status"] == "done" or "chest turned" in a["why"], a
    p_ = out["probe"]
    assert p_["net_up_N"]["ms10"] < 337.0 and p_["net_side_N"]["ms10"] < 312.0, p_
    g = out["g1"]["held"]
    print(f"parent 9: the brief turn: {a['status']} ('{a['why'][:130]}'), her effort peak {out['effort_peak_N']} N, over the",
          f"sustained cap {out['over_sustained_s']} s; her whole force on it at most {p_['net_up_N']['ms10']} N up and",
          f"{p_['net_side_N']['ms10']} N along the floor (10 ms); while held its pelvis moved {g['pelvis_travel_cm']} cm as it rolled,",
          f"its centre of mass rose {g['com_rise_cm']} cm; after: {out['posture_after']}")


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
    # under babble, her report read every tick, a save between two ticks of a push (her yield within the tick, her blows and her
    # report carried by the save: the W2 fix found the yield made within a tick dropped by load_state)
    w = W.G1World(seed=1)
    pm = w.parent
    b = T.babbler(4, 0.3)
    asked = []

    def tick(w_, b_, asked_):
        pm_ = w_.parent
        if not asked_ or (pm_.status(asked_[-1]) in PM.DONE_STATES and not pm_.phases):
            asked_.append(pm_.request(Act(*T.BABBLE_ACTS[len(asked_) % len(T.BABBLE_ACTS)])))
        w_.apply(b_.acts())
        return w_.frame(), pm_.report(w_.tick)
    for _ in range(400):
        tick(w, b, asked)
        if any(np.any(pm.doff[c]) for c in PM.CHAINS) and pm.stats.get("blows", 0) > 0:
            break
    assert any(np.any(pm.doff[c]) for c in PM.CHAINS), "no yield within a tick before the save"
    blob = w.save_state(); bst = b.state(); ask0 = list(asked); blows0 = pm.stats.get("blows", 0)
    rec1 = [tick(w, b, asked) for _ in range(M)]
    end1 = w.save_state(); st1 = pm.state(); blows1 = pm.stats.get("blows", 0)
    w2 = W.G1World(seed=1); w2.load_state(blob)
    b2 = T.babbler(4, 0.3); b2.load(bst); asked2 = list(ask0)
    for (f1, r1), (f2, r2) in zip(rec1, [tick(w2, b2, asked2) for _ in range(M)]):
        assert all(np.array_equal(f1.obs[k], f2.obs[k]) for k in f1.obs) and f1.truth["parent"] == f2.truth["parent"] and r1 == r2
    assert w2.save_state() == end1 and w2.parent.state() == st1
    # the world's save the same bytes for the same acts, whoever made the names (the W2 verifier's finding: pickle memoizes a
    # string by its identity, so two callers' equal names made two saves of one content differ)
    saves = []
    for fresh in (False, True):
        w3 = W.G1World(seed=1, parent=False)
        acts = T.babbler(4, 0.3).acts()
        if fresh:
            acts = {"".join(list(k)): v for k, v in reversed(list(acts.items()))}
        w3.apply(acts)
        saves.append(w3.save_state())
    assert saves[0] == saves[1]
    print(f"parent 12: the exact replay with her acting: attend asked, {N} ticks, a save ({len(blob) / 1e3:.0f} KB); {M} more",
          "restored in the same world and in a new one: every frame, her motion and the final save bit for bit; under babble (seed",
          f"4, p_rest 0.3) a save in the middle of a push, {blows1 - blows0} blows after it: restored in a new world, {M} ticks of",
          "frames, her reports, her state and the final save bit for bit")


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
    """parent 14 (the W2 verifier's findings 1, 2 and 5 of its second round): the babbling G1 under the verifier's act mix (attend,
    show the block, lean_in, touch its tummy, asked in turn; babbler seeds 1, 5, 6 and 8 at p_rest 0.3, 600 ticks each; on seed 6
    W2 had pressed its knee at 1,746 N): her force on any link of the child as a 10 ms mean never reaches F_pain (C8: "under F_pain
    always"); her body never lifts or slides it: the work her body's contacts do on it (her force at each contact against the
    velocity of the child's point there) is never enough, in any tick or over the whole run, to slide it a centimetre on the mat
    (312 N, the least force measured to slide it, x 1 cm) or lift it a centimetre (its 337 N weight x 1 cm): she resists it, never
    pushes it along (the forces themselves, which a child throwing itself against a person meets whatever she does, are written
    down: her whole force on it upward and along the floor, as a tick's mean and a 10 ms mean); her body's way out never points
    toward the child's centre of mass; no planning fault"""
    rows = []
    for seed in (1, 5, 6, 8):
        w = W.G1World(seed=1)
        pm = w.parent
        weight = w.body_mass * float(np.linalg.norm(w.m.opt.gravity))
        orig = pm._yield_dir
        bad = []

        def spy(c, away, fsum=0.0, _o=orig, _pm=pm, _bad=bad):
            try:
                y = _o(c, away, fsum)
            except TypeError:                                           # a motion whose way out reads no sum of forces
                y = _o(c, away)
            if c == "core" and y is not None:
                v = _pm.d.mocap_pos[_pm.mocap_ids[0]][:2] - _pm.d.subtree_com[_pm.m.body("pelvis").id][:2]
                if float(np.linalg.norm(v)) > 1e-6 and float(y[:2] @ v) < 0:
                    _bad.append((w.tick, [round(float(x), 3) for x in y[:2]]))
            return y
        pm._yield_dir = spy
        r = T.ask_repeatedly(w, T.BABBLE_ACTS, 600, T.babbler(seed, 0.3))
        p_ = r["probe"]
        rows.append((seed, p_["her_10ms_on_child_N"], p_["ticks_over_her_150N"], p_["net_up_N"], p_["net_side_N"], p_["body_work_J"]))
        assert p_["ticks_over_f_pain"] == 0, (seed, p_)
        wmin = min(312.0, weight) * 0.01
        assert p_["body_work_J"]["tick_max"] < wmin and p_["body_work_J"]["positive_sum"] < wmin, (seed, p_)
        assert not bad, (seed, bad[:5])
        assert r["jumps_refused"] == 0, (seed, r["outcome"])
    print("parent 14: under babble (the verifier's mix, seeds 1, 5, 6, 8 at p_rest 0.3, 600 ticks) her 10 ms force on the child at",
          "most", {x[0]: x[1] for x in rows}, f"N (F_pain {w.f_pain:.0f} N; ticks over her own 150 N:", {x[0]: x[2] for x in rows},
          "); her body's work on it, largest in a tick / all positive work over the run,", {x[0]: (x[5]["tick_max"], x[5]["positive_sum"])
                                                                                                 for x in rows},
          "J (a centimetre's slide takes 3.1 J, a centimetre's lift 3.4 J); her whole force on it, upward / along the floor, largest",
          "tick mean", {x[0]: (x[3]["tick"], x[4]["tick"]) for x in rows}, "N and 10 ms mean", {x[0]: (x[3]["ms10"], x[4]["ms10"])
                                                                                                 for x in rows},
          f"N; her body's way out never toward it; no planning fault")


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


HANDS_BABBLE_MM = 10.0          # under babble a limb may be a physics step's travel into her hand before she moves it out (a hand
                                # swung at 2.5-5 m/s covers 5-10 mm in 2 ms): the probe samples the step before her hand gives


def test_her_hands_stay_out():
    """parent 16 (the W2 verifier's third finding in both its rounds): her hands never pass through the child. Every hand (its
    palm, fingers, thumb and proxy), whatever it does (reaching onto it, holding it, letting go, resting), at four steps a tick,
    against the G1's convex hulls (which enclose its drawn meshes): on the still child, in every act that puts a hand on it
    (attend, the guide, the prop with its reach onto the torso, the pull-to-sit, the turn), within 1 mm (MuJoCo's own distance);
    under babble, the verifier's seeds 1, 6 and 8 at p_rest 0.3 (a hip 58 mm and a knee 39 mm into her fingers), within
    HANDS_BABBLE_MM"""
    rows = {}
    T.HANDS = True
    try:
        for n in ("attend", "guide", "prop", "pull_to_sit", "turn"):
            r = T.SCENARIOS[n]()
            rows[n] = (r["probe"]["hand_to_hull_mm"], r["probe"]["hand_worst"])
            assert rows[n][0] is not None and rows[n][0] >= -1.0, (n, rows[n])
    finally:
        T.HANDS = False
    bab = {}
    for seed in (1, 6, 8):
        w = W.G1World(seed=1)
        r = T.ask_repeatedly(w, T.BABBLE_ACTS, 600, T.babbler(seed, 0.3), hands=True)
        bab[seed] = (r["probe"]["hand_to_hull_mm"], r["probe"]["hand_worst"])
        assert bab[seed][0] is None or bab[seed][0] >= -HANDS_BABBLE_MM, (seed, bab[seed])
    print("parent 16: her hands against the child's hulls, least distance (mm), every hand at four steps a tick: still child",
          {k: v[0] for k, v in rows.items()}, "; under babble", {k: v[0] for k, v in bab.items()}, "(worst:",
          {k: v[1] for k, v in bab.items()}, ")")


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
    included: in the pull-to-sit and the brief turn her holds never take more than her caps leave after what her body pressed
    (the 5 steps before), so her whole force on the G1 passes the brief cap only on a step where the child pressed newly into her
    (a contact is read after its step: her holds give way on the next), and passes the sustained cap for at most BRIEF_S"""
    rows = {}
    for name in ("turn", "pull_to_sit"):
        out = T.SCENARIOS[name]()
        p_ = out["probe"]
        rows[name] = (p_["total_peak_N"], p_["total_steps_over_brief_cap"], p_["total_steps_over_brief_cap_by_a_new_contact"],
                      round(p_["total_steps_over_sustained_cap"] * 0.002, 2))
        assert p_["total_steps_over_brief_cap"] == p_["total_steps_over_brief_cap_by_a_new_contact"], (name, p_)
        assert p_["total_steps_over_sustained_cap"] * 0.002 <= K.BRIEF_S + 1e-9, (name, p_)
    print("parent 20: in the pull and the turn her holds and her body together (friction counted): peak N, steps over the brief",
          "cap, of them steps where the child pressed newly into her, s over the sustained cap:", rows)


class _Conduct:
    """a stand-in for P3's Conduct: only what her motion reads of it (eyes_on_child)"""
    eyes_on_child = False


def test_the_contract():
    """parent 21 (the W2 verifier's fourth finding): P3's contract, as StubMotion writes it (body/sim/lang/conduct.py on
    sim-parent, 883ec52 and b301745): statuses exactly running / done / refused / cancelled (an act waiting its turn is running);
    report(tick) -> eyes, head, left, right and acts, every act from its request until it is reported ended and never after, the
    same report for the same tick, carried by the save; cancel(id, tick); Act.thing names the toy a 'do' handles; while her
    conduct's eyes_on_child holds, her eyes stay on the child's eyes, no glance, and no act starts but PENDING_OK's"""
    STAT = {"running", "done", "refused", "cancelled"}
    w = W.G1World(seed=1)
    pm = w.parent
    c = _Conduct()
    pm.bind_conduct(c)
    ids = [pm.request(Act("look", "block", "focus")), pm.request(Act("walk", "door")), pm.request(Act("fly"))]
    seen = {i: [] for i in ids}
    ended = set()
    for k in range(12):
        w.apply({})
        r = pm.report(w.tick)
        assert set(r) == {"eyes", "head", "left", "right", "acts"}, r
        assert r == pm.report(w.tick)                                   # asked again: the same report
        for i, st in r["acts"].items():
            assert st in STAT and i not in ended, (i, st)
            seen[i].append(st)
            if st in PM.DONE_STATES:
                ended.add(i)
        for i in ids:
            assert pm.status(i) in STAT
    assert seen[ids[2]] == ["refused"] and seen[ids[0]][0] == "running" and seen[ids[0]][-1] == "done", seen
    assert ids[1] not in ended and all(st == "running" for st in seen[ids[1]])
    # the save carries the report: restored, the same tick's report is the same
    blob = w.save_state(); rep = pm.report(w.tick)
    w2 = W.G1World(seed=1); w2.load_state(blob)
    assert w2.parent.report(w2.tick) == rep
    pm.cancel(ids[1], w.tick + 1)
    w.apply({}); r = pm.report(w.tick)
    assert r["acts"].get(ids[1]) == "cancelled", r
    w.apply({}); assert ids[1] not in pm.report(w.tick)["acts"]
    # eyes_on_child: a show asked waits (running, not begun), her eyes on the child's eyes, a glance ignored; PENDING_OK's open
    # hand starts; the ask over, the show begins
    _live(w, 30, until=lambda w_: not w_.parent.phases)
    c.eyes_on_child = True
    sh = pm.request(Act("show", "ball")); pm.glance("drum")
    for k in range(3):
        w.apply({}); r = pm.report(w.tick)
        assert r["acts"][sh] == "running" and pm.cur["body"] is None and r["eyes"] == "child_eyes" and r["head"] == "child_eyes", r
    oh = pm.request(Act("open_hand", "child"))
    _live(w, 1)
    assert pm.cur["body"] == oh and pm.queue["body"] == [sh], (pm.cur, pm.queue)     # an ask allows her open hand, the show waits
    c.eyes_on_child = False
    pm.cancel(oh, w.tick + 1)
    _live(w, 12, until=lambda w_: w_.parent.cur["body"] == sh)
    assert pm.cur["body"] == sh, (pm.cur, pm.queue)
    # Act.thing: 'do show' of the ball fetches the ball, never the nearest toy
    w = W.G1World(seed=1)
    class _A:
        kind, target, during, thing = "do", "show", None, "ball"
    i = w.parent.request(_A())
    held = []
    _live(w, 900, until=lambda w_: (held.append(dict(w_.parent.holding)) or w_.parent.status(i) in PM.DONE_STATES) and not w_.parent.phases)
    assert w.parent.status(i) == "done", w.parent.why(i)
    got = {t for h in held for t in h.values() if t is not None}
    assert got == {"ball"}, got
    print("parent 21: P3's contract: statuses running/done/refused/cancelled only (waiting is running); report() every act until",
          "reported ended and never after, the same for the same tick and across a save; cancel(id, tick) for this tick or a later",
          "one; eyes_on_child held her eyes on the child's eyes, ignored a glance and started only PENDING_OK's acts; 'do show' with",
          "thing=ball handled the ball alone")


PARENT_TESTS = [test_the_scene, test_the_capped_spring, test_the_interface, test_attend, test_lean_in, test_the_guide,
                test_the_pull_never_sits_it_up, test_the_prop_and_the_catch, test_the_turn, test_toys, test_her_pace,
                test_exact_replay_with_her_acting, test_her_cost, test_her_yield_under_babble, test_getting_up_beside_it,
                test_her_hands_stay_out, test_exact_replay_across_a_solve, test_the_catch_pushes_at_once,
                test_the_interface_does_and_copies, test_her_caps_count_her_body, test_the_contract]

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
