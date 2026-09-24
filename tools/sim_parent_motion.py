"""THE PARENT'S MOTION MEASURED (docs/SIM_DESIGN.md 4.1, 4.2, 4.10, A3-A10, A22, A25, C6, C7, C34; the build plan's W2). An instrument,
never the body: each scenario is a fresh world of seed 1 (the G1 as born on its back, or placed by the instrument in another
posture), the parent asked for one or a few acts through her motion's interface (body/sim/parent_motion.py), the world living
until they end. Per scenario it writes down:
  - each act's status, its reason, its ticks from the ask to its end (her timing), her largest reach error (her reach);
  - every hold's peak force against its own cap, her effort's peak (the sum of her hands' force magnitudes) against her caps, and
    how long she spent over a sustained cap (her brief caps' clock);
  - her body's contacts with the child (the peak force per chain, the deepest penetration, the ticks she yielded) and the hits she
    felt (her own pain);
  - what her acts did to the G1: its centre of mass's largest rise above its start (a lift), its pelvis's largest travel on the
    floor (a slide), its trunk's least angle from vertical (a sit-up), each against the posture it began in;
  - her cost: the wall ms of her motion's work a tick (mean, median, and the largest tick, which is an act's planning), on this Mac
    under whatever load it carries (the load average is written beside it).
Scenarios: approach, attend, lean_in (with A1's face test run with the fovea put on her mouth), show, hand_over (the toy in the
child's hand 30 ticks after), bring_back, point, wave, cover_face and reveal_face, clap, walk to the door and back, the sofa, the
guide (the near arm raised; the far arm across), the knee over, the pull-to-sit (asked in the world; and its pull with her placed
by the instrument within reach of both forearms, the G1's arms as born), the prop (the G1 placed in the leaning sit, her placed
kneeling beside it: the catch and the easing), the brief turn (the G1 placed face down), the feed (a bottle added by the
instrument as a rig: the charge while it is held in the palm), guide_pace (the guide's peak force against the arm's own push at
each candidate pace: parent_consts.GUIDE_SPEED is the fastest within it) and idle and steady costs.
Run: nice -n 19 python3 tools/sim_parent_motion.py [scenario ...] [--out=FILE]   (JSON on stdout; FILE if given)"""
import json
import math
import os
import statistics
import sys
import time

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "body", "sim"))
from body.sim import world as W  # noqa: E402
from body.sim import parent_motion as PM  # noqa: E402
K = PM.K                        # her constants: the very module her motion reads (a second import would be a copy)

G = W.G
kin = G.kin


class Act:
    """an act as her conduct asks for it (P3's conduct.Act: kind, target, during)"""

    def __init__(self, kind, target=None, during=None):
        self.kind, self.target, self.during = kind, target, during


def trunk_deg(w):
    return math.degrees(math.acos(float(np.clip(w.d.xmat[w.m.body("torso_link").id].reshape(3, 3)[2, 2], -1, 1))))


# ---------------------------------------------------------------------------------------------------- placements (instrument)
POSTURES = {
    "sit": (np.eye(3), dict(hip_pitch=-1.7, hip_roll=.12, knee=.4, waist_pitch=.35, shoulder_pitch=-.6, shoulder_roll=.2, elbow=.6)),
    "front": (kin.ry(math.pi / 2), dict(G.BIRTH)),
}


def place_g1(w, posture):
    """the G1 placed on the mat in a posture (section 3.8's), everything else as born (an instrument's placement; the whole state is
    reset as the scene's own birth does it); her motion born again beside it"""
    R, joints = POSTURES[posture]
    w.scene.place_on_mat(joints, R, G.BIRTH_XY, settle_s=0.0)
    w.scene.set_parent(G.born_parent())
    w.d.qvel[:] = 0
    mujoco.mj_forward(w.m, w.d)
    w._sense_birth()
    w.parent = PM.ParentMotion(w)


def bottle_rig(spec):
    """a feeding bottle for the feed (the bottle is W3's to build: a rig, never the room): a 6 cm cylinder named bottle (the world's
    charger through a palm), free, on the mat by the child's left side, with a weld from each of her hands"""
    b = spec.worldbody.add_body(name="toy_bottle", pos=[-0.40, 0.00, 0.012 + 0.07])
    b.add_freejoint(name="toy_bottle")
    b.add_geom(name="bottle", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, 0.07, 0], mass=0.15, contype=4, conaffinity=13,
               priority=2, rgba=[0.95, 0.95, 0.9, 1])
    for sd in "LR":
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name=f"hold_{sd}_bottle", name1=f"parent_hand_{sd}", name2="toy_bottle",
                          objtype=mujoco.mjtObj.mjOBJ_BODY, active=False, solref=[0.006, 1])


# ---------------------------------------------------------------------------------------------------- the runner
def run(w, seq, N=600, after=0, on_tick=None):
    """ask for the acts in `seq` [(kind, target)], live until they end (and `after` ticks more), measuring"""
    pm = w.parent
    m, d = w.m, w.d
    ids = [pm.request(Act(k, t)) for k, t in seq]
    pel = m.body("pelvis").id
    com0 = d.subtree_com[pel].copy(); xy0 = d.qpos[:2].copy(); th0 = trunk_deg(w)
    rise = slide = 0.0; th_min = th0; th_max = th0
    hold0 = None; hold_rise = hold_slide = 0.0; hold_th = None                # the G1 from the moment her first hold engaged
    over_s = 0.0
    hold_pk = {}
    costs = []
    t_end = {}
    extra = 0
    load = os.getloadavg()[0]
    for k in range(N):
        tp = w.timing["parent_s"]
        w.apply({})
        costs.append((w.timing["parent_s"] - tp) * 1e3)
        rise = max(rise, float(d.subtree_com[pel][2] - com0[2]))
        slide = max(slide, float(np.linalg.norm(d.qpos[:2] - xy0)))
        th = trunk_deg(w); th_min = min(th_min, th); th_max = max(th_max, th)
        if pm.holds and hold0 is None:
            hold0 = (d.subtree_com[pel].copy(), d.qpos[:2].copy(), th); hold_th = [th, th]
        if hold0 is not None:
            hold_rise = max(hold_rise, float(d.subtree_com[pel][2] - hold0[0][2]))
            hold_slide = max(hold_slide, float(np.linalg.norm(d.qpos[:2] - hold0[1])))
            hold_th = [min(hold_th[0], th), max(hold_th[1], th)]
        for h in pm.holds:
            key = (h.name, h.kind)
            hold_pk[key] = (max(hold_pk.get(key, (0.0, 0.0))[0], h.peak), max(hold_pk.get(key, (0.0, 0.0))[1], h.cap))
        eff = sum(float(np.linalg.norm(h.force)) for h in pm.holds)
        if eff > K.CAP_TWO:
            over_s += W.TICK_S
        for i in ids:
            if i not in t_end and pm.status(i) in PM.DONE_STATES:
                t_end[i] = k + 1
        if on_tick is not None:
            on_tick(w, k)
        if len(t_end) == len(ids) and not pm.phases:
            extra += 1
            if extra > after:
                break
    acts = []
    for (kind, tgt), i in zip(seq, ids):
        a = pm.info(i)
        inf = a.get("info", {})
        acts.append(dict(kind=kind, target=tgt, status=pm.status(i), why=pm.why(i), ticks=t_end.get(i), reach_err_m=round(inf.get("err", 0.0), 3),
                         paused=inf.get("paused", 0), solve_ms=round(inf.get("solve_ms", 0.0)),
                         **{k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in inf.items()
                            if k2 in ("limb_push", "guide_cap", "palm_N", "closure_deg", "cleared", "shuffle_short_m", "replans")}))
    return dict(acts=acts, ticks=k + 1,
                holds={f"{n} ({kd})": dict(peak_N=round(pk, 1), cap_N=round(cap, 1)) for (n, kd), (pk, cap) in hold_pk.items()},
                effort_peak_N=round(pm.effort_peak, 1), over_sustained_s=round(over_s, 2),
                contact_peak_N={c: round(v, 1) for c, v in pm.stats["contact_peak"].items()}, pen_max_mm=round(pm.stats["pen_max"] * 1e3, 1),
                yield_ticks=pm.stats["yield_ticks"], jumps_refused=pm.stats.get("jumps_refused", 0), slips=pm.stats.get("slips", 0),
                her_hits=len(pm.hits),
                g1=dict(com_rise_cm=round(rise * 100, 2), pelvis_travel_cm=round(slide * 100, 2), trunk_start_deg=round(th0, 1),
                        trunk_min_deg=round(th_min, 1), trunk_max_deg=round(th_max, 1),
                        held=None if hold0 is None else dict(com_rise_cm=round(hold_rise * 100, 2), pelvis_travel_cm=round(hold_slide * 100, 2),
                                                            trunk_start_deg=round(hold0[2], 1), trunk_min_deg=round(hold_th[0], 1),
                                                            trunk_max_deg=round(hold_th[1], 1))),
                cost_ms=dict(mean=round(statistics.fmean(costs), 1), median=round(statistics.median(costs), 1), max=round(max(costs), 1),
                             load_avg=round(load, 1)))


# ---------------------------------------------------------------------------------------------------- scenarios
def sc_simple(seq, N=600, after=0):
    def f():
        w = W.G1World(seed=1)
        return run(w, seq, N, after)
    return f


def sc_lean_in():
    from body.sim import eyes as E
    w = W.G1World(seed=1)
    out = run(w, [("lean_in", None)], 600)
    m, d = w.m, w.d
    mouth = E.mouth_point(m, d)[0]
    g = W.clamp_gaze(E.gaze_at(m, d, mouth))
    ft = E.face_test(m, d, g)
    out["face"] = dict(plan=w.parent.info(0)["info"].get("face"), face_test_on_her_mouth={s: list(v) for s, v in ft.items()},
                       gaze_to_mouth_deg=[round(math.degrees(x), 1) for x in g],
                       off_born_line_deg=round(math.degrees(math.hypot(g[0], g[1])), 1),
                       mouth_to_eyes_m={s: round(float(np.linalg.norm(mouth - d.cam_xpos[m.camera(f"eye_{s}").id])), 3) for s in "LR"})
    return out


def sc_hand_over(placed=False):
    """the block handed over from the door (placed=False: the child's hands have closed into fists on their own by the time she
    arrives, C41), or with her placed kneeling on its left with the block in her hand at birth (its left hand still open)"""
    w = W.G1World(seed=1)
    if placed:
        pm = w.parent
        ch = pm.child
        mid = (ch.torso[:2] + ch.pelvis[:2]) / 2
        pm.place("heels", mid + ch.lat[:2] * 0.84 + ch.len_axis[:2] * 0.15, math.atan2(-ch.lat[1], -ch.lat[0]))
        pm.give_toy(pm._near_hand(ch.grasp["L"]), "block")
    out = run(w, [("hand_over", "block")], 700, after=30)
    m, d = w.m, w.d
    ch = PM.Child(m, d, w.scene.g1_set)
    b = d.xpos[m.body("toy_block").id]
    out["block_to_palm_m"] = {s: round(float(np.linalg.norm(b - ch.grasp[s])), 3) for s in "LR"}
    out["block_held"] = bool(min(out["block_to_palm_m"].values()) < 0.07 and b[2] > 0.05)
    return out


def sc_bring_back():
    w = W.G1World(seed=1)
    out = run(w, [("bring_back", "car")], 900)
    m, d = w.m, w.d
    ch = PM.Child(m, d, w.scene.g1_set)
    c = d.xpos[m.body("toy_car").id]
    out["car_to_hand_m"] = round(min(float(np.linalg.norm(c[:2] - ch.grasp[s][:2])) for s in "LR"), 3)
    return out


def sc_guide(which=None):
    def f():
        w = W.G1World(seed=1)
        m, d = w.m, w.d
        rec = {}

        def tick(w_, k):
            for h in w_.parent.holds:
                if h.kind == "guide":
                    p = h.point(d)
                    rec.setdefault("p0", p.copy()); rec["p"] = p.copy()
        out = run(w, [("guide", which)], 700, on_tick=tick)
        if "p0" in rec:
            out["held_point_moved_cm"] = [round(x * 100, 1) for x in (rec["p"] - rec["p0"])]
        return out
    return f


def sc_pull_placed():
    """the pull's physics with her placed (instrument) kneeling tall beside its hips at the nearest spot 3 cm clear of it from which
    both forearms are within her reach (none is, among the spots her own planning may use: C7), the G1 as born"""
    w = W.G1World(seed=1)
    pm = w.parent
    ch = pm.child
    m, d = w.m, w.d
    mid = (ch.torso[:2] + ch.pelvis[:2]) / 2
    lat = ch.lat[:2]
    yaw = math.atan2(-lat[1], -lat[0])
    fore = [d.xpos[m.body(f"{s}_elbow_link").id] + d.xmat[m.body(f"{s}_elbow_link").id].reshape(3, 3) @ np.array([0.07, 0, -0.042])
            for s in ("left", "right")]                                         # her grip point on each forearm (her palm on its top)
    found = None
    tried = []
    for along in (0.20, 0.30, 0.10):
        for off in np.arange(0.40, 0.91, 0.05):
            at = mid + ch.len_axis[:2] * along + lat * off
            cl = pm._clearance(PM.frame_segs("kneel", "tall", at, yaw))
            if cl < K.CLEAR_M:
                continue
            up = [d.xmat[m.body(f"{s}_elbow_link").id].reshape(3, 3)[:, 2] for s in ("left", "right")]
            both = []
            for pair in ((("L", 0), ("R", 1)), (("R", 0), ("L", 1))):          # both hands at once, one trunk (as she would hold them)
                pm.base = dict(mode="tall", at=list(at), yaw=yaw, lean=0.0, spine=0.0, twist=0.0)
                tg = {sd: (fore[i], PM.NatR(-up[i]), dict(curl=.9, thumb=.8, index=None)) for sd, i in pair}
                both.append(pm._solve_trunk(tg, None)[3])
            tried.append((round(float(along), 2), round(float(off), 2), both))
            if any(both):
                found = at
                break
        if found is not None:
            break
    if found is None:
        return dict(acts=[dict(kind="pull_to_sit (placed)", status="no placement", why="no tall kneel 3 cm clear of the child reaches both forearms")],
                    tried=tried)
    pm.place("tall", found, yaw)
    a = pm.request(Act("pull_to_sit"))
    pm.queue["body"].remove(a)
    act = pm._act(a)
    act["status"] = "running"; act["start"] = w.tick; pm.cur["body"] = a
    act["info"] = dict(ticks=0, paused=0, err=0.0, viol=0, hold_peak=0.0, hold_cap=0.0, solve_ms=0.0)
    pm.phases = pm._plan_pull(act)
    out = run(w, [], 400)
    out["acts"] = [dict(kind="pull_to_sit (placed)", status=pm.status(a), why=pm.why(a), reach_err_m=round(pm.info(a)["info"].get("err", 0.0), 3),
                        placed_off_m=round(float((found - mid) @ lat), 2))]
    return out


def sc_prop():
    """the prop: the G1 placed in the leaning sit (it sinks under its resting law), her placed kneeling beside it"""
    w = W.G1World(seed=1)
    place_g1(w, "sit")
    pm = w.parent
    ch = pm.child
    fr = PM.unit(ch.torso_R[:, 0] * [1, 1, 0])[:2]
    lat = np.array([-fr[1], fr[0]])
    H = ch.pelvis[:2] + lat * 0.75 - fr * 0.10
    pm.place("heels", H, math.atan2(-lat[1], -lat[0]))
    modes = []

    def tick(w_, k):
        for h in w_.parent.holds:
            if h.kind == "prop":
                modes.append((k, h.ctl.get("mode"), h.ctl.get("k"), round(h.ctl.get("theta", 0.0), 1), round(h.cap, 1)))
                break
    out = run(w, [("prop", None)], 120, on_tick=tick)
    seen = []
    for k, mo, kk, th, cap in modes:
        if not seen or seen[-1][1:3] != (mo, kk):
            seen.append((k, mo, kk, th, cap))
    out["prop_steps"] = seen[:30]
    return out


def sc_turn():
    w = W.G1World(seed=1)
    place_g1(w, "front")
    out = run(w, [("turn", None)], 700)
    out["posture_after"] = w.parent.child.posture
    return out


def sc_feed():
    w = W.G1World(seed=1, extra=bottle_rig)
    w.h = 0.5
    out = run(w, [("offer_bottle", "bottle")], 900)
    out["charge"] = round(w.h, 3)
    return out


def sc_guide_pace():
    """the guide's pace (A25): the near forearm raised 12 cm at each candidate pace, the G1 resting under its servo law; the peak
    spring force against the arm's own push at that pose. GUIDE_SPEED is the fastest pace within the push."""
    rows = []
    for v in (0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08):
        K.GUIDE_SPEED = v
        w = W.G1World(seed=1)
        out = run(w, [("guide", None)], 700)
        a = out["acts"][0]
        pk = max((h["peak_N"] for n, h in out["holds"].items() if "guide" in n), default=0.0)
        rows.append(dict(speed=v, status=a["status"], peak_N=pk, push_N=a.get("limb_push"), cap_N=a.get("guide_cap"),
                         within_push=bool(a.get("limb_push") and pk <= a["limb_push"])))
    return dict(rows=rows, fastest_within_push=max((r["speed"] for r in rows if r["within_push"] and r["status"] == "done"), default=None))


def sc_costs():
    """her cost a tick: idle at the door; walking; kneeling beside the child with a hand resting on it"""
    w = W.G1World(seed=1)
    out = {}
    t = []
    for _ in range(40):
        p0 = w.timing["parent_s"]; w.apply({}); t.append((w.timing["parent_s"] - p0) * 1e3)
    out["idle_ms"] = dict(mean=round(statistics.fmean(t), 2), median=round(statistics.median(t), 2))
    w.parent.request(Act("attend"))
    t = []
    for _ in range(400):
        p0 = w.timing["parent_s"]; w.apply({}); t.append((w.timing["parent_s"] - p0) * 1e3)
        if w.parent.status(0) in PM.DONE_STATES and not w.parent.phases:
            break
    t = []
    for _ in range(40):
        p0 = w.timing["parent_s"]; w.apply({}); t.append((w.timing["parent_s"] - p0) * 1e3)
    out["attending_ms"] = dict(mean=round(statistics.fmean(t), 2), median=round(statistics.median(t), 2))
    out["load_avg"] = round(os.getloadavg()[0], 1)
    return out


SCENARIOS = {
    "approach": sc_simple([("approach", None)]),
    "attend": sc_simple([("attend", None)]),
    "lean_in": sc_lean_in,
    "show": sc_simple([("show", "block")]),
    "hand_over": sc_hand_over,
    "hand_over_placed": lambda: sc_hand_over(True),
    "bring_back": sc_bring_back,
    "point": sc_simple([("point", "drum")]),
    "gestures": sc_simple([("wave", None), ("cover_face", None), ("reveal_face", None), ("do", "clap")]),
    "leave_return": sc_simple([("walk", "door"), ("walk", "child")], 900),
    "sofa": sc_simple([("walk", "sofa")], 600),
    "guide": sc_guide(None),
    "guide_far": sc_guide("far_arm"),
    "knee_over": sc_simple([("knee_over", None)]),
    "pull_to_sit": sc_simple([("pull_to_sit", None)]),
    "pull_placed": sc_pull_placed,
    "prop": sc_prop,
    "turn": sc_turn,
    "feed": sc_feed,
    "guide_pace": sc_guide_pace,
    "costs": sc_costs,
}


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or [k for k in SCENARIOS if k != "guide_pace"]
    out_path = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), None)
    res = {}
    for n in names:
        t0 = time.time()
        try:
            res[n] = SCENARIOS[n]()
        except Exception as e:                                          # a scenario's fault is written down, the others go on
            res[n] = dict(error=f"{type(e).__name__}: {e}")
        res[n]["wall_s"] = round(time.time() - t0, 1)
        print(n, json.dumps(res[n], default=str), flush=True)
    if out_path:
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
