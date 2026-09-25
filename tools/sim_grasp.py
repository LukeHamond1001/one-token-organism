"""CAN A STOCK G1 DEX3 HAND HOLD EACH TOY? (docs/SIM_DESIGN.md 3.8, 5.2, A11, B1 and C28; an instrument, never the body.) The
G1 study's trials (the all-out G1 pass's m_grasp_g1.py, ported to the built room: body/sim/g1scene.py's Scene on body/sim/
g1room.xml, the toys at the room's own sizes, or every toy at one scale with --scale), run at a given impratio so the friction
choice can be seen (the owner's decision of 2026-09-24, made for him: impratio is chosen by physics and MuJoCo's guidance, never by
these counts; B1's premise follows from whatever the physics gives). The G1 stands held by a test rig (a weld from its pelvis to a
mocap body, added for this instrument only) with the stock servos of the file (kp 500, the study's; the body's servo law and the
grasp reflex's small steps press less, W4), its left forearm forward. Per toy, 3 placements (the toy 1.2 cm toward the wrist,
centred, 1.2 cm toward the fingertips) x 3 trials:
  HAND-OVER x 2 (across and along the fingers)  palm up: the parent's right hand (its mocap hand and hold weld) brings the toy down
             onto the palm until they touch; the thumb and both fingers close (targets at the closing end of their ranges over
             0.4 s: each servo stops on contact and presses with at most its limit, 1.4-2.45 N m); 0.3 s later she lets go; 0.5 s
             later the wrist rolls 180 deg to palm down over 1.0 s; then 1.5 s hanging. HELD: the toy within 7 cm of the grasp point.
  TOP GRASP  palm down over the toy resting on a test table, 1 cm above its top; the hand closes over 0.4 s, holds 0.3 s, then the
             rig lifts the whole G1 12 cm over 1.0 s and holds 1.0 s. HELD: the toy rose at least 8 cm with the hand.
Each trial is flagged unstable if MuJoCo's warning counters moved. Run: nice -n 19 python3 tools/sim_grasp.py [--impratio 10]
[--scale 1.0] [--toys ball,cup]  (JSON lines on stdout; a --scale scene is written to the system's temporary folder and removed)"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "body" / "sim"
sys.path.insert(0, str(SIM))
import g1acts as GA  # noqa: E402
import g1scene as G  # noqa: E402
import parent_kin as kin  # noqa: E402

TOYS = ["ball", "block", "duck", "cup", "rattle", "car", "bear", "stacker", "drum", "ring"]
RIG_XY = (-1.9, .8)                   # clear floor between the window and the sofa
CLOSED = {"left_hand_thumb_0_joint": 0.0, "left_hand_thumb_1_joint": 1.0, "left_hand_thumb_2_joint": 1.745,
          "left_hand_index_0_joint": -1.5708, "left_hand_index_1_joint": -1.745,
          "left_hand_middle_0_joint": -1.5708, "left_hand_middle_1_joint": -1.745}
PRESHAPE = {"left_hand_thumb_1_joint": -0.5, "left_hand_thumb_2_joint": 0.0,
            "left_hand_index_0_joint": -.2, "left_hand_middle_0_joint": -.2}


def rig(spec):
    spec.worldbody.add_body(name="rig", mocap=True, pos=[0, 0, .95])
    e = spec.add_equality(name="rig_weld", type=mujoco.mjtEq.mjEQ_WELD, name1="rig", name2="pelvis", objtype=mujoco.mjtObj.mjOBJ_BODY)
    e.solref = [.005, 1]
    t = spec.worldbody.add_body(name="test_table", pos=[RIG_XY[0] + .4, RIG_XY[1], -1.0])
    t.add_geom(name="test_table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[.12, .12, .01], contype=1, conaffinity=0, rgba=[.8, .8, .8, 1])


def toy_hand_contact(m, d, tb, hand_bodies):
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 == tb and b2 in hand_bodies) or (b2 == tb and b1 in hand_bodies):
            return True
    return False


def jq(w, name):
    return w.m.jnt_qposadr[w.m.joint(name).id]


def setup(w, arm_roll):
    m, d = w.m, w.d
    mujoco.mj_resetData(m, d)
    d.qpos[:7] = [RIG_XY[0], RIG_XY[1], .793, 1, 0, 0, 0]
    d.mocap_pos[m.body_mocapid[m.body("rig").id]] = [RIG_XY[0], RIG_XY[1], .95]
    pose = {"left_shoulder_roll_joint": .25, "left_elbow_joint": 0.1, "left_wrist_roll_joint": arm_roll, "right_shoulder_roll_joint": -.2}
    pose.update(PRESHAPE)
    for k, v in pose.items():
        d.qpos[jq(w, k)] = v
    for k in TOYS:
        a = m.jnt_qposadr[m.body_jntadr[m.body(f"toy_{k}").id]]
        d.qpos[a:a + 3] = [-2.2 + .35 * TOYS.index(k), -2.0, .15]
    w.set_parent(G.born_parent())
    mujoco.mj_kinematics(m, d)
    w.hold_ctrl()
    wy = m.body("left_wrist_yaw_link").id
    Rw = d.xmat[wy].reshape(3, 3)
    return wy, Rw, d.xpos[wy] + Rw @ GA.GRASP_LOCAL["L"]


def place_toy(w, tb, pt, direction, R, hand_bodies, start=.16):
    m, d = w.m, w.d
    a = m.jnt_qposadr[m.body_jntadr[tb]]
    d.qpos[a + 3:a + 7] = kin.mjquat(R)
    t = start
    while t > -.05:
        d.qpos[a:a + 3] = pt + direction * t
        mujoco.mj_forward(m, d)
        if toy_hand_contact(m, d, tb, hand_bodies):
            t += .002
            d.qpos[a:a + 3] = pt + direction * t
            mujoco.mj_forward(m, d)
            return t
        t -= .002
    return None


def run_phases(w, phases):
    m, d = w.m, w.d
    for dur, fn in phases:
        n = int(round(dur / m.opt.timestep))
        for i in range(n):
            if fn:
                fn((i + 1) / n)
            mujoco.mj_step(m, d)


def close_fn(w, open_):
    def f(a):
        for k, v in CLOSED.items():
            w.d.ctrl[w.act[k]] = open_[k] + (v - open_[k]) * a
    return f


def handover(w, toy, along=False, off=0.0):
    m, d = w.m, w.d
    wy, Rw, g = setup(w, -1.5708)                        # palm up
    hand = {b for b in w.g1_set if m.body(b).name.startswith(("left_hand", "left_wrist"))}
    up = -Rw[:, 1]
    tb = m.body(f"toy_{toy}").id
    ax = Rw[:, 2] if not along else Rw[:, 0]
    R = np.column_stack([kin.perp(ax, up), np.cross(up, kin.perp(ax, up)), up])
    if place_toy(w, tb, g - up * .03 + Rw[:, 0] * off, up, R, hand) is None:
        return dict(test="hand-over", toy=toy, held=False, error="no contact found")
    hm = m.body_mocapid[m.body("parent_hand_R").id]
    d.mocap_pos[hm] = d.xpos[tb] + up * .12
    w.hand_proxy("R", False, forearm=True)
    mujoco.mj_forward(m, d)
    w.weld(f"hold_R_{toy}", True)
    open_ = {k: d.ctrl[w.act[k]] for k in CLOSED}
    roll0 = d.ctrl[w.act["left_wrist_roll_joint"]]
    run_phases(w, [(.4, close_fn(w, open_)), (.3, None)])
    w.weld(f"hold_R_{toy}", False)
    run_phases(w, [(.5, None), (1.0, lambda a: d.ctrl.__setitem__(w.act["left_wrist_roll_joint"], roll0 + math.pi * a)), (1.5, None)])
    w.hand_proxy("R", True, forearm=True)
    gi = d.xpos[wy] + d.xmat[wy].reshape(3, 3) @ GA.GRASP_LOCAL["L"]
    dist = float(np.linalg.norm(d.xpos[tb] - gi))
    return dict(test="hand-over", toy=toy, orientation="along the fingers" if along else "across the fingers",
                held=dist < .07, toy_to_grasp_point_m=round(dist, 3))


def _half_z(m, d, g):
    R = d.geom_xmat[g].reshape(3, 3)
    return float(np.abs(R[2]) @ m.geom_aabb[g][3:])


def _toy_below(w, tb, g, R, hand):
    m, d = w.m, w.d
    a = m.jnt_qposadr[m.body_jntadr[tb]]
    d.qpos[a + 3:a + 7] = kin.mjquat(R)
    z = -.25
    while z < .1:
        d.qpos[a:a + 3] = g + [0, 0, z]
        mujoco.mj_forward(m, d)
        if toy_hand_contact(m, d, tb, hand):
            return g + [0, 0, z - .012]
        z += .002
    return g + [0, 0, -.25]


def top_grasp(w, toy, off=0.0):
    m, d = w.m, w.d
    wy, Rw, g = setup(w, 1.5708)                          # palm down
    hand = {b for b in w.g1_set if m.body(b).name.startswith(("left_hand", "left_wrist"))}
    tb = m.body(f"toy_{toy}").id
    zt = np.array([0, 0, 1.0])
    R = np.column_stack([kin.perp(Rw[:, 2], zt), np.cross(zt, kin.perp(Rw[:, 2], zt)), zt])
    a = m.jnt_qposadr[m.body_jntadr[tb]]
    d.qpos[a:a + 3] = _toy_below(w, tb, g + Rw[:, 0] * off, R, hand)
    mujoco.mj_forward(m, d)
    low = min(d.geom_xpos[gg][2] - _half_z(m, d, gg) for gg in range(m.ngeom) if m.geom_bodyid[gg] == tb and m.geom_contype[gg])
    tid = m.body("test_table").id
    m.body_pos[tid] = [d.xpos[tb][0], d.xpos[tb][1], low - .01]
    mujoco.mj_forward(m, d)
    run_phases(w, [(.3, None)])
    z0 = float(d.xpos[tb][2])
    open_ = {k: d.ctrl[w.act[k]] for k in CLOSED}
    rigm = m.body_mocapid[m.body("rig").id]
    r0 = d.mocap_pos[rigm].copy()
    run_phases(w, [(.4, close_fn(w, open_)), (.3, None), (1.0, lambda s: d.mocap_pos.__setitem__(rigm, r0 + [0, 0, .12 * s])), (1.0, None)])
    rise = float(d.xpos[tb][2] - z0)
    m.body_pos[tid] = [RIG_XY[0] + .4, RIG_XY[1], -1.0]
    return dict(test="top grasp", toy=toy, held=rise > .08, toy_rise_m=round(rise, 3))


def warn_count(d):
    return sum(int(x.number) for x in d.warning)


def main(impratio=None, scale=None, toys=TOYS):
    xml, tmp = G.XML, None
    if scale is not None:                                               # every toy at one scale (the maker's --toy-scale)
        tmp = Path(tempfile.mkdtemp()) / f"g1room_toys{scale:g}.xml"
        subprocess.run([sys.executable, str(SIM / "make_g1room.py"), f"--toy-scale={scale}", f"--out={tmp}"], check=True, capture_output=True)
        xml = tmp
    try:
        w = G.Scene(xml=xml, extra=rig)
    finally:
        if tmp is not None:
            tmp.unlink(); tmp.parent.rmdir()
    if impratio is not None:
        w.m.opt.impratio = float(impratio)
    for toy in toys:
        for off in (-.012, 0.0, .012):
            for fn, kw in ((handover, dict(along=False)), (handover, dict(along=True)), (top_grasp, {})):
                try:
                    r = fn(w, toy, off=off, **kw)
                except mujoco.FatalError as e:
                    r = dict(test=fn.__name__, toy=toy, held=False, error=str(e))
                r.update(impratio=float(w.m.opt.impratio), scale=scale if scale is not None else "the room's", offset_m=off,
                         unstable=warn_count(w.d) > 0)                  # each trial starts with mj_resetData (counters 0)
                print(json.dumps(r), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--impratio", type=float, default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--toys", default=",".join(TOYS))
    a = ap.parse_args()
    main(a.impratio, a.scale, [t for t in TOYS if t in a.toys.split(",")])
