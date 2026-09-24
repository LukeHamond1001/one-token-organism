"""The parent's scripted poses and acts (from the 2026-09-24 prototype): stand, walk, kneel down, sit on the heels, point, show,
hand over, pick up, guide, prop to sit. Every limb is placed by kin's two-bone IK; each pose carries a report of its
joint angles and any human-range violations (pose.report), which the IK check collects.

Placement: `at` is the parent's root (pelvis) on the floor plan (x, y) and `yaw` its facing (0: +x)."""
import math

import numpy as np

import parent_kin as kin
from parent_kin import rz, ry, rx, unit, frame

ANKLE_H = .07          # the ankle joint's height when the sole is flat on the floor
KNEE_FLOOR = .062      # the knee joint's height when the knee rests on the floor (shin radius + a little)
FOOT_TOE = .20         # the toe's distance ahead of the ankle


def base(at, yaw, z):
    return kin.Pose((at[0], at[1], z), rz(yaw))


def to_world(pose, v):
    return pose.pos + pose.R @ np.asarray(v, float)


def dir_world(pose, v):
    return pose.R @ np.asarray(v, float)


def arms_relaxed(pose, bend=12, abd=8):
    """Arms hanging a little away from the body, elbows a little bent, hands soft."""
    segs = kin.fk(pose)
    for sd, sg in (("L", 1), ("R", -1)):
        cp, cR = segs["chest"]
        s = cp + cR @ kin.OFFSET[f"upper_arm_{sd}"]
        down = cR @ unit([.06, sg * math.sin(math.radians(abd)), -1])
        d = kin.L_UA + kin.L_FA * math.cos(math.radians(bend)) - .01
        w = s + down * d + cR @ np.array([.05, 0, 0])
        kin.arm_ik(pose, sd, w, cR @ np.array([-1, sg * .3, 0]), segs=segs)
        pose.hand[sd] = dict(curl=.35, thumb=.25, index=None)


def legs_to(pose, feet, knees=None, foot_R=None):
    """Legs by IK to ankle targets (world) with the knees pointing along knees (world directions; default: forward)."""
    segs = kin.fk(pose)
    fwd = pose.R[:, 0]
    for sd in ("L", "R"):
        kd = fwd if knees is None else knees[sd]
        fr = None if foot_R is None else foot_R[sd]
        kin.leg_ik(pose, sd, feet[sd], kd, fr, segs=segs)


def flat_foot_R(yaw, pitch=0.0):
    return rz(yaw) @ ry(pitch)


# ---------------------------------------------------------------- standing and walking
def stand(at, yaw, knee_bend=.015):
    p = base(at, yaw, kin.HIP_Z - knee_bend)
    feet = {sd: to_world(p, (0.0, sg * .095, 0)) * [1, 1, 0] + [0, 0, ANKLE_H] for sd, sg in (("L", 1), ("R", -1))}
    legs_to(p, feet, foot_R={sd: flat_foot_R(yaw) for sd in "LR"})
    kin.spine(p)
    arms_relaxed(p)
    return p


STEP_L, SPEED = .60, .9          # step length (m) and walking speed (m/s): a relaxed adult gait (cadence 90 steps/min)


def walk(start, end, t):
    """The pose t seconds into a walk from start to end (floor-plan points), steady gait from the first step (the
    start and the stop are blended to standing over half a step). Returns (pose, done)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    Dv = end - start
    D = float(np.linalg.norm(Dv))
    hd = Dv / D
    yaw = math.atan2(hd[1], hd[0])
    s = min(SPEED * t, D)
    phi = s / STEP_L + .5
    bob = .012 * math.cos(2 * math.pi * phi)
    sway = .012 * math.sin(math.pi * phi)
    lat = np.array([-hd[1], hd[0]])
    ramp = min(1.0, s / (.5 * STEP_L), (D - s) / (.5 * STEP_L))
    p = base(start + hd * s + lat * sway * ramp, yaw, kin.HIP_Z - .03 + bob * ramp)
    p.R = rz(yaw) @ rx(math.radians(2.5) * math.sin(math.pi * phi) * ramp)
    kin.spine(p, lumbar=(3, 0, 4 * math.sin(math.pi * phi) * ramp), chest=(2, 0, -6 * math.sin(math.pi * phi) * ramp))
    feet, pitch = {}, {}
    for sd, sg, off in (("L", 1, 0.0), ("R", -1, 1.0)):
        ph = phi - off
        n = math.floor(ph / 2)
        u = ph - 2 * n
        if u < 1:                                  # stance
            a = (2 * n + off + .5) * STEP_L - .5 * STEP_L
            pos, lift, pt = a, 0.0, 0.0
        else:                                      # swing, smoothstep
            v = u - 1
            e = v * v * (3 - 2 * v)
            a0 = (2 * n + off + .5) * STEP_L - .5 * STEP_L
            pos, lift = a0 + e * 2 * STEP_L, .085 * math.sin(math.pi * min(1.0, v * 1.25)) ** .8
            pt = None if v < 1 else 0.0
            pt = math.radians(-10) * (1 - v) ** 2 + math.radians(4) * v ** 2      # ankle angle rel. the shin
        pos = float(np.clip(pos, 0, D))
        w = start + hd * pos + lat * sg * .095
        stand_w = start + hd * s + lat * sg * .095
        w = stand_w + (w - stand_w) * ramp
        feet[sd] = np.array([w[0], w[1], ANKLE_H + lift * ramp])
        pitch[sd] = (pt * ramp) if u >= 1 else None
    # the pelvis rides as high as both legs allow (it dips at double support, as in a real gait)
    zmax = p.pos[2]
    for sd in "LR":
        hip = to_world(p, kin.OFFSET[f"thigh_{sd}"])
        dh = np.linalg.norm((feet[sd] - hip)[:2])
        zmax = min(zmax, feet[sd][2] + math.sqrt(max((kin.L_TH + kin.L_SH - .006) ** 2 - dh * dh, 0)) + (p.pos[2] - hip[2]))
    p.pos[2] = zmax
    legs_to(p, feet, foot_R={sd: (flat_foot_R(yaw) if pitch[sd] is None else pitch[sd]) for sd in "LR"})
    clear_toes(p, hd)
    for sd in "LR":                                # heel-off: a stance foot whose ankle runs out of range pivots on its toe
        if pitch[sd] is None and p.report[f"leg_{sd}"]["ankle_dorsi"] > 20:
            toe = feet[sd] + flat_foot_R(yaw) @ SHOE_TOE
            for th in np.radians(np.arange(2, 42, 2)):
                fR = flat_foot_R(yaw, th)
                feet[sd] = toe - fR @ SHOE_TOE
                segs0 = kin.fk(p)
                kin.leg_ik(p, sd, feet[sd], np.r_[hd, 0], fR, segs=segs0)
                if p.report[f"leg_{sd}"]["ankle_dorsi"] <= 20:
                    break
    segs = kin.fk(p)
    cp, cR = segs["chest"]
    for sd, sg, off in (("L", 1, 1.0), ("R", -1, 0.0)):   # arms swing opposite the legs
        sw = .28 * math.sin(math.pi * (phi - off)) * ramp
        s_ = cp + cR @ kin.OFFSET[f"upper_arm_{sd}"]
        dirv = cR @ unit([math.sin(sw) + .05, sg * .12, -math.cos(sw)])
        w = s_ + dirv * (kin.L_UA + kin.L_FA * .97)
        kin.arm_ik(p, sd, w, cR @ np.array([-1, sg * .3, 0]), segs=segs)
        p.hand[sd] = dict(curl=.4, thumb=.3, index=None)
    return p, s >= D


# ---------------------------------------------------------------- kneeling and sitting on the floor
def kneel(at, yaw, mode="tall", lean=0.0, spine_flex=0.0, twist=0.0, knee_w=.115):
    """Knees on the floor under (tall) or ahead of (heels) the hips; toes tucked under. lean: extra hip flexion of the
    pelvis (deg); spine_flex: flexion spread over the lumbar and chest joints (deg); twist: turn of the chest (deg)."""
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    if mode == "tall":
        hip_z, knee_ahead = KNEE_FLOOR + kin.L_TH * .995, .03
    elif mode == "heels":
        hip_z, knee_ahead = KNEE_FLOOR + .23, .338
    else:
        raise ValueError(mode)
    p = base(at, yaw, hip_z)
    p.R = rz(yaw) @ ry(math.radians(lean))
    kin.spine(p, lumbar=(spine_flex * .55, 0, twist * .3), chest=(spine_flex * .45, 0, twist * .7))
    feet, knees = {}, {}
    left = np.array([-fwd[1], fwd[0]])
    rise = math.asin((.15 - KNEE_FLOOR) / kin.L_SH)
    for sd, sg in (("L", 1), ("R", -1)):
        knee = np.r_[np.array(at) + fwd * knee_ahead + left * sg * knee_w, KNEE_FLOOR]
        feet[sd] = knee + kin.L_SH * np.r_[-fwd * math.cos(rise), math.sin(rise)]
        knees[sd] = np.r_[fwd * .5, -1.0]
    legs_to(p, feet, knees)
    tuck_toes(p, fwd, "LR")
    arms_relaxed(p, bend=25, abd=12)
    return p


SHOE_TOE = np.array([.186, 0, -.052])     # the shoe's toe tip in the foot frame
SHOE_HEEL = np.array([-.068, 0, -.066])   # the heel's bottom


def _recheck_ankle(p, sd, sR, R_):
    b, a_, g = kin.Rot.from_matrix(sR.T @ R_).as_euler("YXZ")
    rep = p.report[f"leg_{sd}"]
    rep["violations"] = [v for v in rep.get("violations", []) if not v.startswith("ankle")]
    kin._check(rep, "ankle_dorsi", -math.degrees(b))
    kin._check(rep, "ankle_inv", math.degrees(a_) * kin.side_sign(sd))
    kin._check(rep, "ankle_twist", math.degrees(g))
    if not rep["violations"]:
        rep.pop("violations")


_DORS = np.radians(np.linspace(kin.LIM_DEG["ankle_dorsi"][0], kin.LIM_DEG["ankle_dorsi"][1], 161))
_RD = np.stack([kin.axang((0, -1, 0), a) for a in _DORS])          # (n, 3, 3): the ankle's own flexion axis


def plant_foot(p, sd, fwd, mode="toe"):
    """Set one ankle's flexion (inside its range; no twist or inversion) so that nothing of the shoe goes below the
    floor and, for mode 'toe', the toe rests on the floor (kneeling on tucked toes, or rising onto the toes); among
    those, the angle nearest neutral."""
    segs = kin.fk(p)
    sp, sR = segs[f"shin_{sd}"]
    ank = sp + sR @ kin.OFFSET[f"foot_{sd}"]
    Rs = np.einsum("ij,njk->nik", sR, _RD)
    toe_z = ank[2] + Rs[:, 2, :] @ SHOE_TOE
    heel_z = ank[2] + Rs[:, 2, :] @ SHOE_HEEL
    pen = np.maximum(0, .008 - toe_z) + np.maximum(0, -.002 - heel_z)
    err = pen * 50 + (np.abs(toe_z - .012) * 20 if mode == "toe" else 0) + np.abs(_DORS) * (.05 if mode == "toe" else .5)
    i = int(np.argmin(err))
    R_ = Rs[i]
    p.world_override[f"foot_{sd}"] = R_
    _recheck_ankle(p, sd, sR, R_)
    rep = p.report[f"leg_{sd}"]
    rep["knee_z"] = round(float(sp[2]), 3)
    rep["toe_z"] = round(float(toe_z[i]), 3)


def tuck_toes(p, fwd, sides):
    for sd in sides:
        plant_foot(p, sd, fwd, "toe")


def clear_toes(p, fwd):
    """Walking: a foot whose toe or heel would go below the floor is pitched just enough to clear it."""
    segs = kin.fk(p)
    for sd in "LR":
        fp, fR = segs[f"foot_{sd}"]
        if min((fp + fR @ SHOE_TOE)[2], (fp + fR @ SHOE_HEEL)[2] + .01) < .008:
            plant_foot(p, sd, fwd, "clear")


def half_kneel(at, yaw, up_side="L", t=1.0):
    """One knee on the floor (the other side's foot planted ahead): the way down to kneeling. t in [0, 1] lowers
    the hips from standing (0) to the half kneel (1)."""
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    latv = np.array([-fwd[1], fwd[0]])
    z = (1 - t) * (kin.HIP_Z - .015) + t * (KNEE_FLOOR + kin.L_TH * .97)
    p = base(at, yaw, z)
    kin.spine(p, lumbar=(6 * t, 0, 0), chest=(4 * t, 0, 0))
    feet, knees, fR = {}, {}, {}
    for sd, sg in (("L", 1), ("R", -1)):
        if sd == up_side:
            a = np.r_[np.array(at) + fwd * (.42 * t) + latv * sg * .12, ANKLE_H]
            feet[sd], knees[sd], fR[sd] = a, np.r_[fwd, 0], flat_foot_R(yaw)
        else:
            k_end = np.r_[np.array(at) + fwd * .0 + latv * sg * .11, KNEE_FLOOR]
            rise = math.asin((.15 - KNEE_FLOOR) / kin.L_SH)
            a_end = k_end + kin.L_SH * np.r_[-fwd * math.cos(rise), math.sin(rise)]
            a_start = np.r_[np.array(at) + latv * sg * .095, ANKLE_H]
            feet[sd] = (1 - t) * a_start + t * a_end
            knees[sd] = np.r_[fwd * .6, -.8 * t]
            fR[sd] = None
    legs_to(p, feet, knees, foot_R=fR)
    tuck_toes(p, fwd, "R" if up_side == "L" else "L")
    arms_relaxed(p, bend=20, abd=10)
    return p


# ---------------------------------------------------------------- the arms' acts
def hand_R_palm(side, palm_normal, fingers):
    """A hand rotation whose palm faces palm_normal (world) with the fingers pointing along fingers (world)."""
    sg = kin.side_sign(side)
    z = -unit(fingers)                     # the fingers run along the hand's -z
    y = -sg * unit(palm_normal)            # the left palm faces the hand's -y, the right its +y
    y = kin.perp(y, z)
    return np.column_stack([np.cross(y, z), y, z])


def _excess(rep):
    tot = 0.0
    for k, v in rep.items():
        if k in kin.LIM_DEG and isinstance(v, (int, float)):
            lo, hi = kin.LIM_DEG[k]
            tot += max(0, lo - v, v - hi)
    return tot


def reach_swivel(pose, side, grip_point, hand_R, curl=.5, thumb=.3, swivels=range(-120, 121, 15)):
    """Place the hand at an exact frame hand_R with its grip point at grip_point, searching the elbow's swivel about
    the shoulder-wrist line for the arm posture that stays inside the human ranges (least excess, then least swivel)."""
    sg = kin.side_sign(side)
    grip_local = np.array([0, -sg * .045, -.10])
    wrist = np.asarray(grip_point, float) - np.asarray(hand_R) @ grip_local
    segs = kin.fk(pose)
    cp, cR = segs["chest"]
    s = cp + cR @ kin.OFFSET[f"upper_arm_{side}"]
    axis = unit(wrist - s)
    base_pole = cR @ np.array([-.5, sg * 1.0, -1.0])
    best = None
    for sw in swivels:
        pole = kin.axang(axis, math.radians(sw)) @ base_pole
        err = kin.arm_ik(pose, side, wrist, pole, hand_R, segs=segs)
        ex = _excess(pose.report[f"arm_{side}"])
        key = (round(ex, 1), abs(sw))
        if best is None or key < best[0]:
            best = (key, pole)
    err = kin.arm_ik(pose, side, wrist, best[1], hand_R, segs=segs)
    pose.hand[side] = dict(curl=curl, thumb=thumb, index=None)
    return err


def reach(pose, side, grip_point, palm_normal, fingers=None, pole=None, curl=.4, thumb=.35, index=None):
    """Put the hand so its grip point (the centre in front of the palm, where a held thing sits) is at grip_point,
    the palm facing palm_normal. fingers None: the fingers continue the forearm's line (a straight wrist as far as the
    palm's facing allows), found by two IK passes."""
    sg = kin.side_sign(side)
    grip_local = np.array([0, -sg * .045, -.10])
    segs = kin.fk(pose)
    cp, cR = segs["chest"]
    if pole is None:
        pole = cR @ np.array([-.5, sg * 1.0, -1.0])      # elbow out, back and down
    s = cp + cR @ kin.OFFSET[f"upper_arm_{side}"]
    fing = fingers if fingers is not None else unit(np.asarray(grip_point, float) - s)
    for it in range(1 if fingers is not None else 3):
        Rh = hand_R_palm(side, palm_normal, fing)
        wrist = np.asarray(grip_point, float) - Rh @ grip_local
        err = kin.arm_ik(pose, side, wrist, pole, Rh, segs=segs)
        if fingers is None:
            fa = pose.world_override[f"forearm_{side}"]
            fing = -fa[:, 2]
    pose.hand[side] = dict(curl=curl, thumb=thumb, index=index)
    return err


def point_at(pose, side, target, dist=.52):
    """Point the index finger at a world target: the arm extended toward it, palm down."""
    segs = kin.fk(pose)
    cp, cR = segs["chest"]
    s = cp + cR @ kin.OFFSET[f"upper_arm_{side}"]
    dv = unit(np.asarray(target, float) - s)
    Rh = hand_R_palm(side, [0, 0, -1] if abs(dv[2]) < .9 else cR[:, 0], dv)
    wrist = s + dv * dist
    sg = kin.side_sign(side)
    err = kin.arm_ik(pose, side, wrist, cR @ np.array([-.3, sg * 1.0, -1.0]), Rh, segs=segs)
    pose.hand[side] = dict(curl=1.45, thumb=.9, index=0.0)
    return err


def grip_point_world(pose, side):
    segs = kin.fk(pose)
    hp, hR = segs[f"hand_{side}"]
    sg = kin.side_sign(side)
    return hp + hR @ np.array([0, -sg * .045, -.10])


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _sag_R(fwd, th):
    """A foot rotation pitched in the walking plane: th = 0 flat and forward, pi/2 pointing down, pi pointing back."""
    f3 = np.r_[fwd, 0]
    left = np.r_[-fwd[1], fwd[0], 0]
    fx = f3 * math.cos(th) + np.array([0, 0, -1.0]) * math.sin(th)
    return np.column_stack([fx, left, np.cross(fx, left)])


def kneel_down(at, yaw, u, step=(.40, .12)):
    """The way from standing to sitting on the heels, u in [0, 3]: [0, 1] a step forward with the left foot while the
    right knee lowers to the floor, the right foot pivoting onto its toes (the toe slides back as it would on a
    carpet); [1, 2] the left knee comes down beside it; [2, 3] the hips sink back onto the heels. `at` is the tall
    kneel's pelvis spot."""
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-fwd[1], fwd[0]])
    W = lambda x, y, z: np.r_[np.asarray(at) + fwd * x + left * y, z]
    tall = kneel(at, yaw, "tall")
    heels = kneel(np.asarray(at) - fwd * (.338 - .03), yaw, "heels")          # the knees stay where they knelt
    tk = kin.fk(tall)
    fin = {}
    for sd in "LR":
        fp, fR = tk[f"foot_{sd}"]
        th = math.atan2(-(fR[:, 0] @ np.array([0, 0, 1.0])), fR[:, 0] @ np.r_[fwd, 0])
        fin[sd] = dict(ank=fp, toe=fp + fR @ SHOE_TOE, th=th)
    stand_x = -.30
    if u <= 1:
        a = _smooth(u / .35)                      # the step forward
        b = _smooth((u - .35) / .65)              # the descent
        px = stand_x + (-.02 - stand_x) * a + (0 - -.02) * b
        pz = (kin.HIP_Z - .015) + (.74 - kin.HIP_Z + .015) * a + (tall.pos[2] - .74) * b
        p = base(W(px, 0, 0)[:2], yaw, pz)
        kin.spine(p, lumbar=(8 * b, 0, 0), chest=(6 * b, 0, 0))
        lL = W(stand_x, .095, kin_ank := ANKLE_H) * (1 - a) + W(step[0], step[1], ANKLE_H) * a
        lL[2] += .08 * math.sin(math.pi * a) if u < .35 else 0
        toe0 = W(stand_x, -.095, 0) + np.r_[fwd * SHOE_TOE[0], 0] + [0, 0, ANKLE_H + SHOE_TOE[2]]
        toe = toe0 * (1 - b) + fin["R"]["toe"] * b
        th = fin["R"]["th"] * b
        # the rear heel rises as the shin tips forward over it: the foot pitches up until the ankle is back in range
        for _ in range(6):
            RR = _sag_R(fwd, th)
            aR = toe - RR @ SHOE_TOE
            legs_to(p, {"L": lL, "R": aR}, {"L": np.r_[fwd, 0], "R": np.r_[fwd * .6, -.8]},
                    foot_R={"L": flat_foot_R(yaw), "R": RR})
            dors = p.report["leg_R"]["ankle_dorsi"]
            if dors <= 20:
                break
            th += math.radians(dors - 18)
        arms_relaxed(p, bend=20 + 10 * b, abd=10)
        return p
    if u <= 2:
        c = _smooth(u - 1)
        p = base(W(0, 0, 0)[:2], yaw, tall.pos[2])
        kin.spine(p, lumbar=(8 * (1 - c), 0, 0), chest=(6 * (1 - c), 0, 0))
        # the left knee swings down on its arc about the hip while the shin swings back ahead of it
        start = kneel_down(at, yaw, 1.0, step)
        s0, t0 = kin.fk(start), kin.fk(tall)
        f3, up = np.r_[fwd, 0], np.array([0, 0, 1.0])
        l3 = np.r_[left, 0]
        ang = lambda v: math.atan2(v @ up, v @ f3)
        hip = kin.fk(p)["thigh_L"][0]
        k0, a0 = s0["shin_L"][0], s0["foot_L"][0]
        k1, a1 = t0["shin_L"][0], t0["foot_L"][0]
        ph0, ph1 = ang(k0 - hip), ang(k1 - hip)
        ps0, ps1 = ang(a0 - k0), ang(a1 - k1)
        if ps1 > ps0:
            ps1 -= 2 * math.pi
        lead = _smooth(min(1.0, c * 2.5))
        ph, ps = ph0 + (ph1 - ph0) * c ** 1.5, ps0 + (ps1 - ps0) * lead
        ps = max(ps, ph - math.radians(148))                    # the knee folds no further than 148 deg on the way
        latk = ((k0 - hip) @ l3) * (1 - c) + ((k1 - hip) @ l3) * c
        rad = math.sqrt(max(kin.L_TH ** 2 - latk ** 2, 1e-6))
        knee = hip + l3 * latk + rad * (f3 * math.cos(ph) + up * math.sin(ph))
        latf = ((a0 - k0) @ l3) * (1 - c) + ((a1 - k1) @ l3) * c
        rads = math.sqrt(max(kin.L_SH ** 2 - latf ** 2, 1e-6))
        aL = knee + l3 * latf + rads * (f3 * math.cos(ps) + up * math.sin(ps))
        # the moving foot turns with its shin: its ankle angle eases from where it was to the tucked angle
        d0, d1 = start.report["leg_L"]["ankle_dorsi"], tall.report["leg_L"]["ankle_dorsi"]
        dors = math.radians(d0 + (d1 - d0) * lead)
        legs_to(p, {"L": aL, "R": fin["R"]["ank"]}, {"L": knee - hip, "R": np.r_[fwd * .6, -.8]},
                foot_R={"L": dors, "R": _sag_R(fwd, fin["R"]["th"])})
        clear_toes(p, fwd)
        arms_relaxed(p, bend=25, abd=12)
        return p
    e = _smooth(u - 2)
    p = base((1 - e) * np.asarray(tall.pos[:2]) + e * np.asarray(heels.pos[:2]), yaw, (1 - e) * tall.pos[2] + e * heels.pos[2])
    hk = kin.fk(heels)
    feet, fR = {}, {}
    for sd in "LR":
        feet[sd] = (1 - e) * fin[sd]["ank"] + e * hk[f"foot_{sd}"][0]
    legs_to(p, feet, {sd: np.r_[fwd * .5, -1.0] for sd in "LR"})
    tuck_toes(p, fwd, "LR")
    arms_relaxed(p, bend=25, abd=12)
    return p


def sit_floor(at, yaw, lean=0.0, spine_flex=0.0, twist=0.0, spread=.34, reach=.76, knee_w=None):
    """Sitting on the floor, legs out in a V (the child can sit between them): the pelvis on the floor, the trunk
    leaning forward by lean (hip flexion beyond upright, deg) and spine_flex, the knees a little bent, heels down."""
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-fwd[1], fwd[0]])
    p = base(at, yaw, .095)
    p.R = rz(yaw) @ ry(math.radians(lean))
    kin.spine(p, lumbar=(spine_flex * .55, 0, twist * .3), chest=(spine_flex * .45, 0, twist * .7))
    feet, knees = {}, {}
    for sd, sg in (("L", 1), ("R", -1)):
        feet[sd] = np.r_[np.asarray(at) + fwd * reach + left * sg * spread, ANKLE_H]
        knees[sd] = np.r_[left * sg * .2, 1.0]                  # the knees point up and a little out
    segs = kin.fk(p)
    for sd in "LR":
        kin.leg_ik(p, sd, feet[sd], knees[sd], math.radians(-10), segs=segs)   # the ankle eased a little (toes up)
        plant_foot(p, sd, fwd, "clear")
    arms_relaxed(p, bend=30, abd=14)
    return p


def kneel_shuffle(at0, at1, yaw, u, mode="tall"):
    """Walking forward on the knees (a knee shuffle) from the kneel spot at0 to at1, u in [0, 1]: three small steps,
    each knee in turn lifted and set down ahead while the hips glide."""
    at0, at1 = np.asarray(at0, float), np.asarray(at1, float)
    n = 4
    k = min(int(u * n), n - 1)
    v = u * n - k
    p0 = at0 + (at1 - at0) * (k / n)
    p1 = at0 + (at1 - at0) * ((k + 1) / n)
    hips = p0 + (p1 - p0) * _smooth(v)
    pose = kneel(hips, yaw, mode)
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-fwd[1], fwd[0]])
    moving = "L" if k % 2 == 0 else "R"
    # the moving knee lifts off and lands one step ahead; the other stays down
    segs = kin.fk(pose)
    for sd, sg in (("L", 1), ("R", -1)):
        base_k = (p1 if sd == moving else p0) if v > .5 else p0
        kx = (p0 + (p1 - p0) * _smooth(v)) if sd == moving else (p0 if k == 0 or True else p0)
        kn = np.r_[kx + fwd * .03 + left * sg * .115, KNEE_FLOOR + (.05 * math.sin(math.pi * v) if sd == moving else 0)]
        if sd != moving:
            kn[:2] = (at0 + (at1 - at0) * ((k + (1 if k % 2 == 1 and sd == "L" else 0)) / n))[:2] * 0 + kn[:2]
        rise = math.asin((.15 - KNEE_FLOOR) / kin.L_SH)
        ank = kn + kin.L_SH * np.r_[-fwd * math.cos(rise), math.sin(rise)]
        kin.leg_ik(pose, sd, ank, np.r_[fwd * .5, -1.0], None, segs=segs)
    tuck_toes(pose, fwd, "LR")
    arms_relaxed(pose, bend=25, abd=12)
    return pose
