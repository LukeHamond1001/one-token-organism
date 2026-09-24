"""THE FRICTION MODEL AND impratio (docs/SIM_DESIGN.md 4.2, 5.1, A11, C5, C22 and C26; the build plan's W1; the owner's decision
of 2026-09-24, made for him in the W1 verifier's third round: impratio is chosen by the physical realism of these contacts and by
MuJoCo 3.9's documented guidance, never by pain rates or by which toys can be held). An instrument, never the body. Each
measurement at impratio 1 (MuJoCo's default) and 10 (the builder's), everything else the room's (the elliptic cone, multi-point
CCD off, implicitfast at 2 ms, the Newton solver):
  STATIC FRICTION (is there creep below the sliding force? real rubber, plastic and foam hold without a macroscopic slide there):
    a rigid 10 kg box (friction 1.0, like the G1's shapes) on a flat surface with the mat's contact (solref, solimp, friction 1,
    priority 2), pushed sideways at 0.3-0.97 mu M g for 2 s: its slide (physics: 0);
    the G1 at rest (as born, the resting servo law) on the mat pushed sideways at the pelvis with 100-200 N for 2 s (its friction
    holds up to about mu x 337 N): the pelvis's slide (the body's own give aside);
  KINETIC FRICTION (does it slide as Coulomb says?): the box pushed at 1.2 and 1.5 mu M g for 1 s from rest: the distance slid
    against (F - mu M g) t^2 / 2M;
  THE NORMAL FORCE WHILE SLIDING (does friction push the normal force up? In Coulomb's law it cannot: the normal force is what
    the load and the surfaces' stiffness make it): the box pressed down with 0 or 400 N more (a pressed motor housing's load) and
    pushed sideways from rest at 1.5 mu N0 (N0 = M g + the press): the contact's normal force (the elliptic cone's first row, as
    touch and pain read it) as the 10 ms means of the slide's first 30 ms and of its steady run, over N0;
  THE G1'S HOUSINGS (world 16's case, SIM_DESIGN.md C22): the left leg drawn in until the hip's roll link presses the pelvis past
    F_pain, then the newborn's flexion (body/sim/reflexes.py) for two ticks: the roll link's pain force (its largest 10 ms mean) in
    each tick against the onset's, with the pair's friction live and with the pair frictionless (condim 1: the same move with no
    friction coupling, the reference for what friction adds).
Run: nice -n 19 python3 tools/sim_friction.py  (JSON on stdout; nothing written)"""
import json
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from body.sim import reflexes as R  # noqa: E402
from body.sim import world as W  # noqa: E402

IMPRATIOS = (1.0, 10.0)
BOX_KG, MU, G_ = 10.0, 1.0, 9.81


def plane_model(impratio):
    """the room's solver and the mat's contact on a clean plane, with the 10 kg box (an instrument's rig)"""
    return mujoco.MjModel.from_xml_string(f"""
<mujoco>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="{impratio}">
    <flag multiccd="disable" autoreset="disable"/>
  </option>
  <worldbody>
    <geom name="mat" type="box" size="5 5 .05" pos="0 0 -.05" friction="1 .01 .001" solref=".03 1" solimp=".85 .95 .004" priority="2"/>
    <body name="box" pos="0 0 .05">
      <freejoint/>
      <geom name="box" type="box" size=".1 .1 .05" mass="{BOX_KG}" friction="1 .005 .0001"/>
    </body>
  </worldbody>
</mujoco>""")


def box_contact_normal(m, d):
    return float(sum(d.efc_force[c.efc_address] for c in d.contact[:d.ncon] if c.efc_address >= 0))


def plane_rows(impratio):
    m = plane_model(impratio)
    d = mujoco.MjData(m)
    b = m.body("box").id
    for _ in range(500):                                               # 1 s to settle
        mujoco.mj_step(m, d)
    rest = mujoco.MjData(m); mujoco.mj_copyData(rest, m, d)
    W_ = BOX_KG * G_
    out = {}
    for k in (0.3, 0.5, 0.8, 0.9, 0.97):                               # static: below the sliding force
        mujoco.mj_copyData(d, m, rest)
        x0 = d.xpos[b].copy(); d.xfrc_applied[b, 1] = k * MU * W_
        for _ in range(1000):
            mujoco.mj_step(m, d)
        out[f"static {k:g} mu M g, 2 s: slide mm"] = round(1e3 * float(np.linalg.norm((d.xpos[b] - x0)[:2])), 3)
    for k in (1.2, 1.5):                                               # kinetic: above it
        mujoco.mj_copyData(d, m, rest)
        x0 = d.xpos[b].copy(); d.xfrc_applied[b, 1] = k * MU * W_
        for _ in range(500):
            mujoco.mj_step(m, d)
        coulomb = (k - 1) * MU * W_ / BOX_KG * 1.0 ** 2 / 2
        out[f"kinetic {k:g} mu M g, 1 s: slide / Coulomb's"] = round(float(np.linalg.norm((d.xpos[b] - x0)[:2])) / coulomb, 4)
    for press in (0.0, 400.0):                                         # the normal force while sliding
        mujoco.mj_copyData(d, m, rest)
        n0 = W_ + press
        d.xfrc_applied[b, 2] = -press
        for _ in range(250):                                           # 0.5 s pressed, at rest
            mujoco.mj_step(m, d)
        pre = []
        for _ in range(5):
            mujoco.mj_step(m, d); pre.append(box_contact_normal(m, d))
        d.xfrc_applied[b, 1] = 1.5 * MU * n0
        fn = []
        for _ in range(250):
            mujoco.mj_step(m, d); fn.append(box_contact_normal(m, d))
        fn = np.array(fn)
        win = np.convolve(fn, np.ones(5) / 5, mode="valid")           # the pain filter's 10 ms means
        out[f"pressed {press:g} N: normal at rest / N0"] = round(float(np.mean(pre)) / n0, 4)
        out[f"pressed {press:g} N: sliding, the first 30 ms' largest 10 ms mean / N0"] = round(float(win[:11].max()) / n0, 4)
        out[f"pressed {press:g} N: sliding, steady (0.1-0.5 s) mean / N0"] = round(float(fn[50:].mean()) / n0, 4)
    return out


def g1_push_rows(impratio):
    w = W.G1World(seed=1)
    w.m.opt.impratio = impratio
    base = w.save_state()
    b = w.m.body("pelvis").id
    out = {}
    for f in (100, 150, 200):
        w.load_state(base)
        x0 = w.d.xpos[b].copy(); w.d.xfrc_applied[b, 1] = f
        for _ in range(13):
            w.apply({})
        w.d.xfrc_applied[b, :] = 0
        out[f"G1 at rest pushed {f} N at the pelvis, 2 s: slide cm"] = round(100 * float(np.linalg.norm((w.d.xpos[b] - x0)[:2])), 3)
    return out


def housing_rows(impratio):
    """world 16's case: the hip's roll link pressed into the pelvis, then the newborn's flexion, two ticks"""
    w = W.G1World(seed=1)
    m = w.m
    m.opt.impratio = impratio
    z = w.zones.index("left_hip_roll")
    push = W.act_flat([0, 0, 2, 2, 2, 2])
    f = w.frame()
    for _ in range(12):
        f = w.frame()
        if f.obs["pain"][z]:
            break
        w.apply({"leg_l": push})
    if not f.obs["pain"][z]:
        return {"housings": "no onset within 12 ticks"}
    blob, at = w.save_state(), float(f.truth["pain_N"][z])

    def two(frictionless=False):
        w.load_state(blob)
        geoms = [g for g in range(m.ngeom) if w.zone_of_geom[g] in (z, w.zones.index("pelvis"))]
        cd = m.geom_condim[geoms].copy()
        if frictionless:
            m.geom_condim[geoms] = 1
        try:
            out = []
            for _ in range(2):
                w.apply({"leg_l": R.flexion_act("leg_l")}); out.append(round(float(w.frame().truth["pain_N"][z])))
        finally:
            m.geom_condim[geoms] = cd
        return out
    return {"housings: onset N": round(at), "housings: the flexion's two ticks N (friction live)": two(),
            "housings: the same, the pair frictionless N": two(True)}


def main():
    out = []
    for imp in IMPRATIOS:
        row = {"impratio": imp}
        row.update(plane_rows(imp))
        row.update(g1_push_rows(imp))
        row.update(housing_rows(imp))
        out.append(row)
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
