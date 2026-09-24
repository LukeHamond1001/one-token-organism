"""THE FRICTION MODEL (docs/SIM_DESIGN.md 4.2, A11 and C26; the build plan's W1): does anything creep below its sliding force, and
does it slide at mu x its weight? An instrument, never the body. Two measurements, each at impratio 1 (MuJoCo's default, the
prototype's) and at the room's own (g1room.xml: 10):
  the G1 at rest (as born, the resting servo law) pushed sideways at the pelvis with 100-400 N for 2 s (13 ticks): the pelvis's
  slide (its friction holds up to about mu x 337 N = 337 N, the body's own give aside);
  a rigid 10 kg box on the mat (an instrument's rig, friction 1.0 like the G1's shapes) pushed at its centre with 0.3-1.2 x mu M g
  for 2 s: its slide. The model is right when the box stays below mu M g and slides above it.
Run: nice -n 19 python3 tools/sim_friction.py  (JSON on stdout)"""
import json
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from body.sim.world import G1World  # noqa: E402


def box_rig(spec):
    b = spec.worldbody.add_body(name="rig_box", pos=[0.9, 0.2, 0.012 + 0.05])
    b.add_freejoint()
    b.add_geom(name="rig_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[.1, .1, .05], mass=10.0, contype=1, conaffinity=1,
               friction=[1, .005, .0001])


def slide(w, body, force, ticks=13):
    m, d = w.m, w.d
    b = m.body(body).id
    x0 = d.xpos[b].copy()
    d.xfrc_applied[b, 1] = force
    for _ in range(ticks):
        w.apply({})
    d.xfrc_applied[b, :] = 0
    return round(float(np.linalg.norm((d.xpos[b] - x0)[:2])) * 100, 3)


def main():
    out = []
    for imp in (1.0, None):
        w = G1World(seed=1, extra=box_rig)
        if imp is not None:
            w.m.opt.impratio = imp
        base = w.save_state()
        row = {"impratio": float(w.m.opt.impratio)}
        for f in (100, 150, 200, 300, 400):
            w.load_state(base)
            row[f"G1 pelvis {f} N: slide cm"] = slide(w, "pelvis", f)
        for k in (0.3, 0.5, 0.8, 0.9, 0.97, 1.03, 1.2):
            w.load_state(base)
            row[f"box {k:g} mu M g: slide cm"] = slide(w, "rig_box", k * 1.0 * 10.0 * 9.81)
        out.append(row)
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
