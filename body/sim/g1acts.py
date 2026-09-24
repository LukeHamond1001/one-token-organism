"""The parent's acts around the G1 (from the 2026-09-24 prototype): the parent's IK (parent_kin.py, parent_poses.py,
parent_acts.solve) aimed at the G1's parts. Unlike parent_acts, these still take an expr for the picture (stills and
physics tests); in the world the face comes from the parent's feelings (parent_feel.py). Everything here is the world's side; nothing enters the
G1 except through its senses."""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parent_acts as A  # noqa: E402
import parent_kin as kin  # noqa: E402
import parent_poses as P  # noqa: E402
from parent_kin import unit  # noqa: E402

# the hand's grasp point in each wrist_yaw_link frame: in front of the palm, between the thumb and the two fingers
GRASP_LOCAL = {"L": np.array([.13, -.06, 0.0]), "R": np.array([.13, .06, 0.0])}   # measured: the half-closed hand encloses it


class G1View:
    """The G1's geometry as the parent sees it (world truth, for the parent's IK only)."""

    def __init__(self, m, d):
        b = lambda n: m.body(n).id
        c = lambda n: m.camera(n).id
        self.eyes = (d.cam_xpos[c("eye_L")] + d.cam_xpos[c("eye_R")]) / 2
        self.eye_R = d.cam_xmat[c("eye_L")].reshape(3, 3).copy()
        self.face_dir = -self.eye_R[:, 2]                         # the optical axis
        self.torso = d.xpos[b("torso_link")].copy()
        self.torso_R = d.xmat[b("torso_link")].reshape(3, 3).copy()
        self.pelvis = d.xpos[b("pelvis")].copy()
        self.pelvis_R = d.xmat[b("pelvis")].reshape(3, 3).copy()
        self.chest = self.torso + self.torso_R @ np.array([.085, 0, .20])     # the chest's front surface
        self.belly = self.torso + self.torso_R @ np.array([.080, 0, .07])
        self.front = self.torso_R[:, 0].copy()                               # the torso's forward normal
        self.shoulder = {sd: self.torso + self.torso_R @ np.array([0, sg * .15, .24]) for sd, sg in (("L", 1), ("R", -1))}
        self.side = {sd: self.torso + self.torso_R @ np.array([0, sg * .11, .17]) for sd, sg in (("L", 1), ("R", -1))}
        self.hip = {sd: self.pelvis + self.pelvis_R @ np.array([0, sg * .13, -.08]) for sd, sg in (("L", 1), ("R", -1))}
        self.grasp = {}
        self.forearm = {}
        for sd, nm in (("L", "left"), ("R", "right")):
            wy = b(f"{nm}_wrist_yaw_link")
            self.grasp[sd] = d.xpos[wy] + d.xmat[wy].reshape(3, 3) @ GRASP_LOCAL[sd]
            el = b(f"{nm}_elbow_link")
            self.forearm[sd] = (d.xpos[el] + d.xmat[el].reshape(3, 3) @ np.array([.08, 0, -.01]), d.xmat[el].reshape(3, 3)[:, 0].copy())


def kneel_spot(v, side="L", along=0.0, off=.72):
    """A kneeling spot on the G1's left (+y) or right side, `off` from its torso's centre line, `along` metres from
    the torso toward the feet; facing the G1."""
    sg = 1 if side == "L" else -1
    lat = unit(v.torso_R[:, 1] * [1, 1, 0]) * sg                           # the G1's left (or right) on the floor plan
    axis = unit(-v.torso_R[:, 2] * [1, 1, 0])                              # toward the feet on the floor plan
    mid = (v.torso[:2] + v.pelvis[:2]) / 2
    at = mid + axis[:2] * along + lat[:2] * off
    yaw = math.atan2(-lat[1], -lat[0])
    return at, yaw


def attend(v, at, yaw, expr=1.0, touch=None, look_at=None):
    """Kneel on the heels beside the G1, look at its eyes, smile; one hand resting on its belly if touch."""
    hands = {}
    if touch:
        n = v.front
        hands = [{sd: (lambda sd=sd: lambda p: P.reach(p, sd, v.belly + n * .03, -n, None, curl=.2, thumb=.35))()} for sd in ("R", "L")]
    p, info = A.solve(at, yaw, hands, look_at=v.eyes if look_at is None else look_at, expr=expr, max_lean=70, max_spine=45)
    if not touch:
        P.arms_relaxed(p, bend=40, abd=14)
    return p, info


def show(v, at, yaw, side=None, dist=.40, lateral=0.0, expr=1.0):
    """Hold a toy dist in front of the G1's eyes along their optical axis (shifted `lateral` across it), the palm on
    the far side of the toy; the hand (either, unless side) and the fingers' direction about the palm's normal are
    searched for a posture inside the human ranges."""
    tgt = v.eyes + v.face_dir * dist + v.eye_R[:, 0] * lateral
    palm_n = -v.face_dir
    f0 = kin.perp(np.array([math.cos(yaw), math.sin(yaw), 0.0]), palm_n)
    alts = []
    for sd in ([side] if side else ["R", "L"]):
        for a in range(0, 360, 30):
            fing = kin.axang(palm_n, math.radians(a)) @ f0
            alts.append({sd: (lambda sd=sd, fing=fing: lambda p: P.reach(p, sd, tgt, palm_n, fing, curl=.9, thumb=.8))()})
    p, info = A.solve(at, yaw, alts, look_at=v.eyes, expr=expr, max_lean=70, max_spine=45)
    info["toy_at"] = tgt
    return p, info


def guide(v, at, yaw, child_side="L", expr=.5):
    """Hold the G1's forearm near the wrist, palm on it from above."""
    tgt, axis = v.forearm[child_side]
    alts = []
    for sd in ("R", "L"):
        def fn(p, sd=sd):
            to = unit((tgt - p.pos) * [1, 1, 0])
            n = kin.perp(unit(np.array([0, 0, -1.0]) * .75 + to * .25), axis)
            return P.reach(p, sd, tgt - n * .045, n, None, curl=1.0, thumb=.9)
        alts.append({sd: fn})
    return A.solve(at, yaw, alts, look_at=v.eyes, expr=expr, max_lean=70, max_spine=45)


def two_hands(v, at, yaw, pts, normals, expr=.6, look_at=None):
    """Both hands on two points of the G1 (palms facing along normals): prop, roll, lift attempts."""
    hands = {}
    for sd in ("L", "R"):
        hands[sd] = (lambda sd=sd: lambda p: P.reach(p, sd, pts[sd], normals[sd], None, curl=.5, thumb=.2))()
    return A.solve(at, yaw, hands, look_at=v.eyes if look_at is None else look_at, expr=expr, max_lean=70, max_spine=45)
