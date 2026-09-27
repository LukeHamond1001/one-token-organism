"""THE SINK RATES under the resting law (docs/SIM_DESIGN.md 3.3 and 3.8; the build plan's W1): how fast each posture the G1 study
held with the stock servos locked gives way when the body rests under its own servo law (body/sim/world.py: kp = limit / 0.25 rad,
the Dex3's 0.1 rad, damping 0.04 s x kp; at rest each target relaxing to the measured angle at 3 ticks; no gravity compensation).
An instrument, never the body: each posture is set on the mat (g1scene's place_on_mat, not settled), its targets at the posture,
and the world lives 3 s at rest. It reports, at 0.15, 0.45, 1.5 and 3 s, the trunk's tilt from where it began, the eyes' height,
the largest joint sag among the legs, waist, shoulders and elbows (and which), and the first tick the trunk has tilted 20 deg.
Run: nice -n 19 python3 tools/sim_sink.py  (JSON on stdout)"""
import json
import math
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from body.sim import world as W  # noqa: E402

G = W.G
kin = G.kin
PRONE = kin.ry(math.pi / 2)            # face down, head toward +x
SUPINE = kin.ry(-math.pi / 2)          # face up, head toward -x
UP = np.eye(3)
POSTURES = {                           # the G1 study's postures (m_postures_g1.py), the wrists left at 0
    "lying on its back (birth)": (SUPINE, dict(G.BIRTH)),
    "lying, both arms raised to vertical": (SUPINE, dict(G.BIRTH, shoulder_pitch=-1.5, elbow=1.28)),
    "lying, both arms raised 45 deg": (SUPINE, dict(G.BIRTH, shoulder_pitch=-0.8, elbow=1.28)),
    "prone on its elbows, chest up": (PRONE, dict(shoulder_pitch=-1.45, shoulder_roll=.25, elbow=-.25, waist_pitch=-.35,
                                                   hip_pitch=.05, knee=.1, ankle_pitch=.45)),
    "sitting leaning forward (hips 97 deg, knees 23), hands on its legs": (UP, dict(
        hip_pitch=-1.7, hip_roll=.12, knee=.4, waist_pitch=.35, shoulder_pitch=-.6, shoulder_roll=.2, elbow=.6)),
    "standing (the stand keyframe's legs, arms a little out)": (UP, dict(shoulder_pitch=.2, shoulder_roll=.3, elbow=1.28)),
}
MAJOR = [j for j in W.JOINTS if any(p in j for p in ("hip", "knee", "ankle", "waist", "shoulder", "elbow"))]
MARKS = (1, 3, 10, 20)                 # ticks: 0.15, 0.45, 1.5, 3 s


def sink(w, R, joints, ticks=20):
    m, d = w.m, w.d
    w.scene.place_on_mat(joints, R, G.BIRTH_XY, settle_s=0.0)
    mujoco.mj_forward(m, d)
    w.h = 1.0; w._sense_birth()
    torso = m.body("torso_link").id
    up0 = d.xmat[torso].reshape(3, 3)[:, 2].copy()
    q0 = {j: float(d.qpos[w.qadr[W.JOINTS.index(j)]]) for j in MAJOR}
    out, fell = {}, None
    for t in range(1, ticks + 1):
        w.apply({})
        up = d.xmat[torso].reshape(3, 3)[:, 2]
        tilt = math.degrees(math.acos(float(np.clip(up @ up0, -1, 1))))
        if fell is None and tilt > 20:
            fell = t
        if t in MARKS:
            sag = {j: math.degrees(float(d.qpos[w.qadr[W.JOINTS.index(j)]]) - q0[j]) for j in MAJOR}
            worst = max(sag, key=lambda k: abs(sag[k]))
            out[f"{t * 0.15:.2f}s"] = dict(trunk_tilt_deg=round(tilt, 1), eye_height_m=round(float(d.cam_xpos[m.camera("eye_L").id][2]), 3),
                                           largest_sag_deg=round(sag[worst], 1), joint=worst.replace("_joint", ""))
    return dict(at=out, trunk_past_20deg_after_s=None if fell is None else round(fell * 0.15, 2))


if __name__ == "__main__":
    w = W.G1World(seed=1)
    res = {name: sink(w, R, joints) for name, (R, joints) in POSTURES.items()}
    print(json.dumps(res, indent=1))
