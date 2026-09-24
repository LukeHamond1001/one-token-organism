"""The parent's acts around the child (from the 2026-09-24 prototype): each finds a kneeling posture whose arms reach by
two-bone IK inside human ranges, searching the trunk's lean (hip flexion) and spine flexion from upright.

An act never chooses the parent's face: expr defaults to None, which leaves the pose's face as it is (neutral in a new
pose). The face comes from the parent's feelings (parent_feel.py), set on the pose each tick; a caller that passes an
expr (a still, a physics test) sets it for that picture only."""
import math

import numpy as np

import parent_kin as kin
import parent_poses as P
from parent_kin import unit


def violations(p):
    return {k: r["violations"] for k, r in p.report.items() if isinstance(r, dict) and r.get("violations")}


def reach_errors(p):
    return {k: r["reach_err_m"] for k, r in p.report.items() if isinstance(r, dict) and "reach_err_m" in r}


def solve(at, yaw, hands, mode="heels", knee_w=.115, look_at=None, expr=None, twist=0.0, max_lean=60, max_spine=45,
          shifts=(0.0,)):
    """hands: {side: fn(pose) -> reach error} or a list of such dicts (alternatives, e.g. either hand); each fn places
    that hand (it may read the pose's chest). Tries the smallest trunk bend first; returns (pose, info); info['ok'] is
    False when nothing in the search reached inside the human ranges."""
    if isinstance(hands, list):
        found = [solve(at, yaw, h, mode, knee_w, look_at, expr, twist, max_lean, max_spine, shifts) for h in hands]
        ok = [f for f in found if f[1]["ok"]]
        pick = min(ok, key=lambda f: f[1]["lean"] + f[1]["spine"]) if ok else min(found, key=lambda f: f[1]["reach_err"])
        pick[1]["hand"] = "/".join(hands[found.index(pick)].keys())
        return pick
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    best = None
    for shift in shifts:
        for total in range(0, max_lean + max_spine + 1, 5):
            for lean in range(0, min(total, max_lean) + 1, 5):
                sp = total - lean
                if sp > max_spine:
                    continue
                spot = np.asarray(at) + fwd * shift
                if mode == "sit":
                    p = P.sit_floor(spot, yaw, lean=lean, spine_flex=sp, twist=twist)
                else:
                    p = P.kneel(spot, yaw, mode, lean=lean, spine_flex=sp, twist=twist, knee_w=knee_w)
                errs = [fn(p) for fn in hands.values()]
                if look_at is not None:
                    kin.look(p, look_at)
                if expr is not None:
                    p.expr = expr
                bad = violations(p)
                score = sum(errs) + .05 * len(bad)
                if best is None or score < best[0]:
                    best = (score, p, dict(lean=lean, spine=sp, shift=shift))
                if max(errs, default=0) < .005 and not bad:
                    return p, dict(ok=True, lean=lean, spine=sp, shift=shift, reach_err=max(errs, default=0))
    p = best[1]
    return p, dict(ok=False, **best[2], reach_err=max(reach_errors(p).values()), violations=violations(p))


# ---------------------------------------------------------------- the child's geometry, read from the sim
class ChildView:
    def __init__(self, m, d):
        import mujoco
        self.m, self.d = m, d
        b = lambda n: m.body(n).id
        s = lambda n: m.site(n).id
        self.head = d.xpos[b("child_head")].copy()
        hR = d.xmat[b("child_head")].reshape(3, 3)
        self.head_R = hR.copy()
        eL, eR = d.xpos[b("child_L_eye")], d.xpos[b("child_R_eye")]
        self.eyes = (eL + eR) / 2
        self.face_dir = hR[:, 0].copy()                 # the head's forward (+x) axis
        self.chest = d.xpos[b("child_chest")].copy()
        self.chest_R = d.xmat[b("child_chest")].reshape(3, 3).copy()
        self.prop = {sd: d.site_xpos[s(f"child_prop_{sd}")].copy() for sd in "LR"}
        self.grasp = {sd: d.site_xpos[s(f"child_{sd}_grasp")].copy() for sd in "LR"}
        self.hold = {sd: d.site_xpos[s(f"child_{sd}_forearm_hold")].copy() for sd in "LR"}
        self.forearm_R = {sd: d.xmat[b(f"child_{sd}_forearm")].reshape(3, 3).copy() for sd in "LR"}
        self.pelvis = d.xpos[b("child_pelvis")].copy()


def toy_pos(m, d, k):
    return d.xpos[m.body(f"toy_{k}").id].copy()


# ---------------------------------------------------------------- acts
def attend(cv, at, yaw, expr=None, touch=None):
    """Kneel on the heels beside the child, look at its face, smile; one hand resting on its tummy if touch."""
    hands = {}
    if touch:
        belly = cv.chest + cv.chest_R @ np.array([.07, 0, -.075])     # the tummy's front, below the chest
        n = cv.chest_R[:, 0]                                         # its outward normal (up when on its back)
        hands = [{sd: (lambda sd=sd: lambda p: P.reach(p, sd, belly + n * .03, -n, None, curl=.2, thumb=.35))()} for sd in ("R", "L")]
    p, info = solve(at, yaw, hands, look_at=cv.eyes, expr=expr)
    if not touch:
        P.arms_relaxed(p, bend=40, abd=14)
    return p, info


def point(cv, at, yaw, target, side=None, look_at=None, expr=None):
    """Point at a world target with either hand (the one that can, inside the ranges); the trunk may turn up to 30 deg."""
    sides = [side] if side else ["R", "L"]
    best = None
    for tw in (0, 15, -15, 30, -30):
        for sd in sides:
            p, info = solve(at, yaw, {sd: (lambda sd=sd: lambda p: P.point_at(p, sd, target))()}, twist=tw,
                            look_at=cv.eyes if look_at is None else look_at, expr=expr, max_lean=20, max_spine=20)
            info["hand"], info["twist"] = sd, tw
            if info["ok"]:
                return p, info
            best = best or (p, info)
    return best


def show(cv, at, yaw, side="R", dist=.30, expr=None):
    """Hold a toy (at the grip point) dist in front of the child's eyes, the palm on the far side of the toy."""
    tgt = cv.eyes + cv.face_dir * dist
    palm_n = -cv.face_dir                                  # the palm faces the child: the toy between hand and child
    fing = unit(np.cross(palm_n, [0, 0, 1])) if abs(palm_n[2]) < .9 else unit(np.r_[-math.cos(yaw), -math.sin(yaw), 0] * 0 + [math.cos(yaw), math.sin(yaw), 0])
    hands = {side: lambda p: P.reach(p, side, tgt, palm_n, fing, curl=.9, thumb=.8)}
    p, info = solve(at, yaw, hands, look_at=cv.eyes, expr=expr)
    info["toy_at"] = tgt
    return p, info


def hand_over(cv, at, yaw, child_side="L", side="R", expr=None):
    """Bring a held toy to the child's hand: the parent's grip point at the child's grasp point."""
    tgt = cv.grasp[child_side]
    hands = {side: lambda p: P.reach(p, side, tgt + np.array([0, 0, .035]), [0, 0, -1],
                                     unit(np.r_[math.cos(yaw), math.sin(yaw), 0]), curl=.9, thumb=.8)}
    p, info = solve(at, yaw, hands, look_at=cv.eyes, expr=expr)
    info["toy_at"] = tgt + np.array([0, 0, .035])
    return p, info


def pick_up(cv, at, yaw, toy_xyz, expr=None):
    """Reach a toy on the floor or mat from kneeling, either hand: the grip point just above its centre, palm down,
    the fingers pointing away from the parent."""
    alts = []
    for side in ("R", "L"):
        g = np.asarray(toy_xyz) + [0, 0, .01]
        alts.append({side: (lambda side=side, g=g: lambda p: P.reach(
            p, side, g, [0, 0, -1], None, curl=.95, thumb=.85))()})
    return solve(at, yaw, alts, look_at=toy_xyz, expr=expr, max_lean=70, max_spine=45)


def approach_spot(toy_xy, child_xy, dist=.62, avoid_r=.45):
    """Where to kneel to pick a toy up: on a circle of radius dist around it, the point farthest from the child,
    facing the toy."""
    best = None
    for a in np.radians(np.arange(0, 360, 15)):
        spot = np.asarray(toy_xy) + dist * np.array([math.cos(a), math.sin(a)])
        dmin = min(np.linalg.norm(spot - np.asarray(c)) for c in child_xy)
        if best is None or dmin > best[0]:
            best = (dmin, spot)
    spot = best[1]
    yaw = math.atan2(toy_xy[1] - spot[1], toy_xy[0] - spot[0])
    return spot, yaw


def guide(cv, at, yaw, child_side="L", side="R", expr=None):
    """Hold the child's forearm at its hold point (the soft weld's anchor), palm on the forearm."""
    tgt = cv.hold[child_side]
    axis = cv.forearm_R[child_side][:, 2]
    alts = []
    for sd in ("R", "L"):
        for sgn in (1, -1):
            def fn(p, sd=sd, sgn=sgn):
                to = unit((tgt - p.pos) * [1, 1, 0])
                n = unit(np.array([0, 0, -1.0]) * .75 + to * .25)
                n = kin.perp(n, axis)                        # the palm faces the forearm, from above and the parent's side
                return P.reach(p, sd, tgt - n * .03, n, None if sgn > 0 else sgn * unit(np.cross(n, axis)), curl=1.0, thumb=.9)
            alts.append({sd: fn})
    return solve(at, yaw, alts, look_at=cv.eyes, expr=expr)


def prop(cv, at, yaw, expr=None, knee_w=.21, chest=None, chest_R=None, mode="heels"):
    """Both hands on the child's chest sides (the prop sites), palms facing each other, fingers up the back."""
    chest = cv.chest if chest is None else chest
    cR = cv.chest_R if chest_R is None else chest_R
    tg = {sd: chest + cR @ np.array([0, sg * .075, .045]) for sd, sg in (("L", 1), ("R", -1))}
    hands = {}
    for psd, csd in (("L", "R"), ("R", "L")):     # facing the child from its feet: the parent's left meets the child's right
        sgc = 1 if csd == "L" else -1
        normal = cR @ np.array([0, -sgc, 0])        # the palm faces the child's midline
        fing = cR @ np.array([0, 0, 1.0])           # fingers up along the child's back and sides
        grip = tg[csd] - cR @ np.array([0, sgc * .03, 0]) + cR @ np.array([0, 0, -.02])
        hands[psd] = (lambda psd=psd, grip=grip, normal=normal, fing=fing:
                      (lambda p: P.reach(p, psd, grip, normal, None, curl=.5, thumb=.2,
                                         pole=None)))()
    return solve(at, yaw, hands, mode=mode, knee_w=knee_w, look_at=cv.eyes, expr=expr, max_lean=70, max_spine=45)
