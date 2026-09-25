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
guide (the near arm raised; the far arm across), the knee over, the pull-to-sit (asked in the world: from its feet, A9), the prop (the G1 placed in the leaning sit, her placed
kneeling beside it: the catch and the easing), the brief turn (the G1 placed face down), the feed (a bottle added by the
instrument as a rig: the charge while it is held in the palm), guide_pace (the guide's peak force against the arm's own push at
each candidate pace, on the limp arm A25 names and under the resting law: parent_consts.GUIDE_SPEED is the fastest within it on the
limp arm) and idle and steady costs. Every run also carries the per-step probe (Probe): her body's contact force on the G1 with its
friction, her holds and body together against her caps, the runs of steps she pressed it over a resting hand's weight, her 10 ms
mean force on each of its links (C8), and in `hands` her hands' least distance to its convex hulls.
The W2 verifier's cases: babble_attend (seeds 2, 3, 4, 7 at p_rest 0.6 and 0.3, attend asked over and over for 600 ticks),
babble_acts (attend, show, lean_in, touch in turn), still_door / still_sofa / still_lean_bring (a still child: getting up from
beside it and going on), still_<place> (the child placed rotated, in a corner, by a wall, in the hall, by the sofa, by the table:
attend, lean_in, show), hands, catch (C6) and copy_do.
c8 (the lead's decision of 2026-09-25, she is a body): C8 over many seeds at p_rest 0.3 and 0.6, attend alone and the mix. The exact
replay across processes: --replay=save:PATH, then --replay=load:PATH and --replay=birth:PATH in new processes (replay_proc).
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
    """an act as her conduct asks for it (P3's conduct.Act: kind, target, during, thing)"""

    def __init__(self, kind, target=None, during=None, thing=None):
        self.kind, self.target, self.during, self.thing = kind, target, during, thing


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
    b.add_geom(name="bottle", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, 0.07, 0], mass=0.15, contype=4, conaffinity=15,
               priority=2, rgba=[0.95, 0.95, 0.9, 1])
    for sd in "LR":
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name=f"hold_{sd}_bottle", name1=f"parent_hand_{sd}", name2="toy_bottle",
                          objtype=mujoco.mjtObj.mjOBJ_BODY, active=False, solref=[0.006, 1])


# ---------------------------------------------------------------------------------------------------- the per-step probe
class Probe:
    """an instrument on every physics step (never her motion's own reading): her body's contact force on the G1 (each contact's
    normal and friction together, MuJoCo's mj_contactForce), her holds' and her body's force together (against her caps), the
    runs of steps her body pressed the child over a resting hand's weight, her whole force on it as a vector (her holds and her
    body's contacts: its upward and its horizontal part, as a 10 ms mean and a tick's mean, against its weight and the force that
    slides it on the mat), the work her body's contacts do on it each tick (her force at each contact against the velocity of the
    child's point there: positive only where she pushes it along its own motion, never where she only resists it) and its net over
    each contact between them (from the step they touch to 10 ms after they part: the energy her body put into the child, what
    it gave back of the child's own blow netted out), the 10 ms mean
    (5 steps, A12's pain measure) of her
    normal force on each of the child's links (C8: under F_pain always; each tick over it told as hers, where her contacts did
    positive work on that link in those 10 ms, or the child's, where it did: its blow or its press, as on the floor), and with
    hands=True the least signed distance of each
    of her hands (its palm, capsule, fingers, thumb; holding the child or not) to the G1's convex hulls, at four steps a tick"""
    HAND_STEPS = (0, 25, 50, PM.STEPS - 1)

    def __init__(self, w, hands=False):
        self.w = w; pm = w.parent; m = w.m
        self.hands = hands
        self.f6 = np.zeros(6); self.ft = np.zeros(6)
        self.win = {}
        self.body_peak = 0.0; self.total_peak = 0.0; self.total_over_brief = 0; self.total_over_two_steps = 0
        self.over_brief_new = 0; self.body_hist = []
        self.run = 0; self.runs = []
        self.link10 = 0.0; self.link10_ticks = []; self._tick10 = 0.0
        self.hand_min = None; self.hand_worst = None
        self.depth = {"hand": 0.0, "forearm": 0.0, "body": 0.0}; self.depth_worst = {}   # her shapes' deepest contact into the G1 (m)
        self.depth_reg = {k: 0.0 for k in ("hand", "forearm", "upper_arm", "trunk", "head", "legs")}   # ... by her body's region
        self.net_tick = np.zeros(3); self.net_win = []                  # her whole force on the G1 as a vector (holds and body)
        self.up10 = self.side10 = self.upT = self.sideT = 0.0
        self.work_tick = 0.0; self.work_max = 0.0; self.work_pos = 0.0  # her body's work on the G1 (J): positive only if she pushes it
        self.ep = 0.0; self.ep_gap = 0; self.ep_max = 0.0; self.ep_pos = 0.0   # ... and its net over each contact between them (the
                                                                        # energy that flowed from her into it while they touched)
        self.jac = np.zeros((3, m.nv))                                  # along its own motion, never while she only resists it
        self.wwin = {}                                                  # per link, her contacts' work on it over the last 5 steps: a
        self.fp_by = {"hers": 0, "child": 0}; self._tick_fp = None     # 10 ms over F_pain is hers when she did positive work on that
        self.fp_events = []                                             # link in those 10 ms (she pushed it), the child's when it did
                                                                        # (it struck or pressed her: its own blow, as on the floor, C5)
        self.hand_geoms = {sd: [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == m.body(f"parent_hand_{sd}").id
                                and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH] + [m.geom(f"parent_forearm_{sd}").id] for sd in "LR"}
        self.seg_kind = {}                                              # her collision shapes' kind: hand, forearm or body
        for g in range(m.ngeom):
            sg = int(pm.geom_seg[g])
            if sg >= 0:
                nm = PM.kin.SEGS[sg]
                self.seg_kind[g] = "hand" if nm.startswith("hand") else "forearm" if nm.startswith("forearm") else "body"
        self.g1 = [g for g in range(m.ngeom) if m.geom_bodyid[g] in w.scene.g1_set and m.geom_contype[g]]
        # THE JOINTS' LAW FROM HER (A37; C8 as the lead's decision of 2026-09-25 words it: the observer's torque over a joint's limit
        # from her contacts): the outside torque her contacts and her holds put on each of the G1's 43 joints (J^T f at each point,
        # the world's truth, which the born observer estimates) and the outside force on its free base, as 10 ms means (A12's 5
        # steps), against each joint's own torque limit (the model's, world.tau_max) and F_pain
        self.jdof = np.asarray(w.dof, dtype=np.int64); self.jlim = np.asarray(w.tau_max, float).copy()
        self.jnames = [m.joint(int(j)).name for j in w.jid]
        self.root_dof = int(m.jnt_dofadr[m.body("pelvis").jntadr[0]])
        self.jwin = []; self.bwin = []
        self.jt_ratio = 0.0; self.jt_worst = None; self.jt_over_ticks = 0; self._jt_tick_over = False
        self.jt_events = []; self._segs_hist = []                     # (tick, joint, N m, limit, her segments touching it)
        self.jt_by = {}; self._jt_tick_over_who = False               # the ticks over a joint's limit by who did the work
        self.base10 = 0.0; self.base_over_ticks = 0; self._base_tick_over = False
        # THE LEAD'S BABBLE CRITERIA (2026-09-25, A25b): her contacts with the child by her region (hand, forearm, upper arm, trunk,
        # head, legs), as 10 ms means and tick means (ISO/TS 15066's body-region bounds for her hands and forearms: K.HAND_N,
        # K.FOREARM_N), the largest single step's contact force (a spike), her trunk's least distance to the child's body (its trunk's
        # links) at four steps a tick, the longest run of ticks her trunk, head or legs touched the child, and her muscles against
        # their limits every step (the actuators' force over its range; any other torque of hers on her joints)
        self.region = {}
        for g, kd in self.seg_kind.items():
            nm = PM.kin.SEGS[int(pm.geom_seg[g])]
            self.region[g] = ("hand" if nm.startswith("hand") else "forearm" if nm.startswith("forearm") else
                              "upper_arm" if nm.startswith("upper_arm") else "head" if nm == "head" else
                              "legs" if nm.startswith(("thigh", "shin", "foot")) else "trunk")
        self.reg_names = ("hand_L", "hand_R", "forearm_L", "forearm_R", "upper_arm", "trunk", "head", "legs")
        self.reg_win = []; self.reg10 = {k: 0.0 for k in self.reg_names}; self.reg_tick = {k: 0.0 for k in self.reg_names}
        self._reg_acc = {k: 0.0 for k in self.reg_names}
        self.step_peak = 0.0; self.step_peak_at = None
        self.core_run = 0; self.core_run_max = 0; self._core_touch_tick = False; self.core_touch_ticks = 0
        self.core_down = 0.0                                            # her trunk, head or legs pressing down on the child (N, a tick)
        self._core_down_acc = 0.0
        body_ids = {m.body(n).id for n in PM.CHILD_BODY}
        self.child_body = [g for g in self.g1 if int(m.geom_bodyid[g]) in body_ids]
        self.core_geoms = [g for g, kd in self.region.items() if kd == "trunk" or   # her trunk and her head's solid (c: the lead's
                           (kd == "head" and (m.geom(g).name or "") == "parent_head")]   # "her trunk"), against its body
        self.standoff_min = None; self.standoff_worst = None
        self.act_ratio = 0.0; self.act_worst = None; self.applied_max = 0.0
        self.sup_f = 0.0; self.sup_fz_min = None; self.sup_t = 0.0      # her support's force (largest, least upward) and torque (A25b)
        self._orig = pm.after_step
        pm.after_step = self.after

    def after(self, s):
        self._orig(s)
        w = self.w; m, d = w.m, w.d; pm = w.parent
        body = 0.0; per = {}; net = np.zeros(3); touched = False; wk = {}
        tau = np.zeros(m.nv)                                            # her contacts' and holds' generalized force on the G1
        segs_now = set(); reg = {}
        for i in range(d.ncon):
            c = d.contact[i]
            g0, g1 = int(c.geom[0]), int(c.geom[1])
            if pm.geom_seg[g0] >= 0 and pm.g1_geom[g1]:
                ch = g1
            elif pm.geom_seg[g1] >= 0 and pm.g1_geom[g0]:
                ch = g0
            else:
                continue
            if c.efc_address < 0:
                continue
            mine = g0 if ch == g1 else g1
            kd = self.seg_kind.get(mine, "body")
            segs_now.add((PM.kin.SEGS[int(pm.geom_seg[mine])] if pm.geom_seg[mine] >= 0 else "toy", m.body(int(m.geom_bodyid[ch])).name))
            if -float(c.dist) > self.depth[kd]:
                self.depth[kd] = -float(c.dist)
                self.depth_worst[kd] = (w.tick, s, m.geom(mine).name, m.body(int(m.geom_bodyid[ch])).name)
            rk = self.region.get(mine, "trunk")
            self.depth_reg[rk] = max(self.depth_reg[rk], -float(c.dist))
            mujoco.mj_contactForce(m, d, i, self.f6)
            fmag = float(np.linalg.norm(self.f6[:3]))
            body += fmag
            lk = int(m.geom_bodyid[ch]); per[lk] = per.get(lk, 0.0) + float(self.f6[0])
            fw = np.asarray(c.frame).reshape(3, 3).T @ self.f6[:3]      # on the second geom, from the first (world frame)
            fc = fw if ch == g1 else -fw
            rg = self.region.get(mine, "trunk")
            if rg in ("hand", "forearm"):
                rg = rg + "_" + PM.kin.SEGS[int(pm.geom_seg[mine])][-1]
            reg[rg] = reg.get(rg, 0.0) + fmag
            if fmag > self.step_peak:
                self.step_peak = fmag; self.step_peak_at = (w.tick, s, m.geom(mine).name or rg, m.body(lk).name)
            if rg in ("trunk", "head"):
                self._core_touch_tick = True
                self._core_down_acc += max(0.0, -float(fc[2]))            # her pushing the child down (its force from her, downward)
            net += fc
            mujoco.mj_jac(m, d, self.jac, None, np.asarray(c.pos), lk)   # the child's point there, moving
            tau += self.jac.T @ fc
            dw = float(fc @ (self.jac @ d.qvel)) * m.opt.timestep
            self.work_tick += dw; self.ep += dw; touched = True
            wk[lk] = wk.get(lk, 0.0) + dw
        if touched:
            self.ep_gap = 0
        elif self.ep != 0.0:                                            # a contact ends after 5 steps (10 ms) with none
            self.ep_gap += 1
            if self.ep_gap >= 5:
                self.ep_max = max(self.ep_max, self.ep); self.ep_pos += max(0.0, self.ep); self.ep = 0.0; self.ep_gap = 0
        for h in pm.holds:                                              # her holds' force on the link each holds (C8: her force on
            per[h.body] = per.get(h.body, 0.0) + float(np.linalg.norm(h.force))   # the child, holds and body together)
            mujoco.mj_jac(m, d, self.jac, None, h.point(d), h.body)
            tau += self.jac.T @ h.force
        for h in pm.holds:                                              # a hold's force is her hand's on the child
            k_ = "hand_" + h.side
            reg[k_] = reg.get(k_, 0.0) + float(np.linalg.norm(h.force))
        self.reg_win = (self.reg_win + [reg])[-5:]
        for k_ in self.reg_names:
            self._reg_acc[k_] += reg.get(k_, 0.0)
            if len(self.reg_win) == 5:
                self.reg10[k_] = max(self.reg10[k_], sum(r.get(k_, 0.0) for r in self.reg_win) / 5.0)
        af = d.actuator_force[pm.bm.act]                                # HER MUSCLES against her strength, every step
        r_ = float(np.max(np.maximum(af / pm.bm.hi, af / pm.bm.lo)))
        if r_ > self.act_ratio:
            j_ = int(np.argmax(np.maximum(af / pm.bm.hi, af / pm.bm.lo)))
            self.act_ratio = r_; self.act_worst = (w.tick, s, j_, round(float(af[j_]), 2), float(pm.bm.lo[j_]), float(pm.bm.hi[j_]))
        self.applied_max = max(self.applied_max, float(np.abs(d.qfrc_applied[pm.bm.dofs]).max()),
                               float(np.abs(m.dof_damping[pm.bm.dofs]).max()))
        sf = pm.drive.sup
        self.sup_f = max(self.sup_f, float(np.linalg.norm(sf[:3])))
        self.sup_fz_min = float(sf[2]) if self.sup_fz_min is None else min(self.sup_fz_min, float(sf[2]))
        self.sup_t = max(self.sup_t, float(sum(np.linalg.norm(t_) for t_ in pm.drive.sup_t)))
        if s in self.HAND_STEPS and self.child_body:                    # her trunk's standoff from the child's body
            best = 0.2
            gp = d.geom_xpos; rb = m.geom_rbound
            for g in self.core_geoms:
                for h in self.child_body:
                    if float(np.linalg.norm(gp[h] - gp[g])) - rb[h] - rb[g] > best:
                        continue
                    best = min(best, float(mujoco.mj_geomDistance(m, d, g, h, best, self.ft)))
            if self.standoff_min is None or best < self.standoff_min:
                self.standoff_min = best; self.standoff_worst = (w.tick, s, pm.base["mode"], pm.phases[0]["type"] if pm.phases else None)
        self._segs_hist = (self._segs_hist + [segs_now | {("hold:" + h.kind, m.body(h.body).name) for h in pm.holds}])[-5:]
        self.jwin = (self.jwin + [tau[self.jdof]])[-5:]                 # the joints' law from her: 10 ms means
        self.bwin = (self.bwin + [tau[self.root_dof:self.root_dof + 3]])[-5:]
        if len(self.jwin) == 5:
            jm = np.abs(sum(self.jwin) / 5.0) / self.jlim
            k = int(np.argmax(jm))
            if jm[k] > self.jt_ratio:
                self.jt_ratio = float(jm[k]); self.jt_worst = (w.tick, s, self.jnames[k], round(float(jm[k] * self.jlim[k]), 1),
                                                              float(self.jlim[k]))
            if jm[k] > 1.0:
                self._jt_tick_over = True
                wsum = sum(sum(q) for q in self.wwin.values())          # who did the work at her contacts over those 10 ms: she
                who = "hers" if wsum > 0.0 else "child"                 # (positive: she pushed it) or the child (it struck or
                self.jt_by[who] = self.jt_by.get(who, 0) + (0 if self._jt_tick_over_who else 1)   # pressed her)
                self._jt_tick_over_who = True
                if len(self.jt_events) < 30 and (not self.jt_events or self.jt_events[-1][0] != w.tick):
                    self.jt_events.append((w.tick, self.jnames[k], round(float(jm[k] * self.jlim[k]), 1), float(self.jlim[k]),
                                           sorted(set().union(*self._segs_hist))[:5], sorted(h.kind for h in pm.holds),
                                           pm.phases[0]["type"] if pm.phases else None, who, round(wsum, 4)))
            bm_ = float(np.linalg.norm(sum(self.bwin) / 5.0))
            self.base10 = max(self.base10, bm_)
            if bm_ > w.f_pain:
                self._base_tick_over = True
        for lk in set(self.win) | set(per):
            q = self.win.setdefault(lk, [])
            q.append(per.get(lk, 0.0)); del q[:-5]
        mean10 = max((sum(q) / 5.0 for q in self.win.values()), default=0.0)
        for lk in set(self.wwin) | set(wk):
            q = self.wwin.setdefault(lk, [])
            q.append(wk.get(lk, 0.0)); del q[:-5]
        f_pain = w.f_pain
        for lk, q in self.win.items():
            if sum(q) / 5.0 > f_pain:
                who = "hers" if sum(self.wwin.get(lk, [0.0])) > 0.0 else "child"
                if self._tick_fp != "hers":
                    self._tick_fp = who
                if len(self.fp_events) < 12 and (not self.fp_events or self.fp_events[-1][0] != w.tick or self.fp_events[-1][2] != who):
                    self.fp_events.append((w.tick, m.body(lk).name, who, round(sum(q) / 5.0), round(sum(self.wwin.get(lk, [0.0])), 3)))
        self.link10 = max(self.link10, mean10); self._tick10 = max(self._tick10, mean10)
        hold = sum(float(np.linalg.norm(h.force)) for h in pm.holds)
        for h in pm.holds:
            net += h.force
        self.net_tick += net; self.net_win = (self.net_win + [net])[-5:]
        n10 = sum(self.net_win) / 5.0
        self.up10 = max(self.up10, float(n10[2])); self.side10 = max(self.side10, float(np.hypot(n10[0], n10[1])))
        if s == PM.STEPS - 1:
            self.work_max = max(self.work_max, self.work_tick); self.work_pos += max(0.0, self.work_tick); self.work_tick = 0.0
            nt = self.net_tick / PM.STEPS
            self.upT = max(self.upT, float(nt[2])); self.sideT = max(self.sideT, float(np.hypot(nt[0], nt[1])))
            self.net_tick = np.zeros(3)
        tot = hold + body
        self.body_peak = max(self.body_peak, body); self.total_peak = max(self.total_peak, tot)
        self.total_over_brief += int(tot > K.CAP_TWO_BRIEF + 1e-6); self.total_over_two_steps += int(tot > K.CAP_TWO + 1e-6)
        prev5 = max(self.body_hist, default=0.0)                        # what her body pressed before this step (her holds' budget)
        if tot > K.CAP_TWO_BRIEF + 1e-6 and body > prev5 and hold <= K.CAP_TWO_BRIEF - prev5 + 1e-6:
            self.over_brief_new += 1                                    # over only by a contact new within this step (read after it)
        self.body_hist = (self.body_hist + [body])[-5:]
        if body > K.TOUCH_N:
            self.run += 1
        elif self.run:
            self.runs.append(self.run); self.run = 0
        if s == PM.STEPS - 1:
            for k_ in self.reg_names:
                self.reg_tick[k_] = max(self.reg_tick[k_], self._reg_acc[k_] / PM.STEPS); self._reg_acc[k_] = 0.0
            self.core_down = max(self.core_down, self._core_down_acc / PM.STEPS); self._core_down_acc = 0.0
            if self._core_touch_tick:
                self.core_run += 1; self.core_touch_ticks += 1
            else:
                self.core_run = 0
            self.core_run_max = max(self.core_run_max, self.core_run); self._core_touch_tick = False
            self.jt_over_ticks += int(self._jt_tick_over); self._jt_tick_over = False; self._jt_tick_over_who = False
            self.base_over_ticks += int(self._base_tick_over); self._base_tick_over = False
            self.link10_ticks.append(self._tick10); self._tick10 = 0.0
            if self._tick_fp is not None:
                self.fp_by[self._tick_fp] += 1; self._tick_fp = None
        if self.hands and s in self.HAND_STEPS:
            for sd in "LR":                                             # every hand, whatever it does (holding the child, reaching,
                for g in self.hand_geoms[sd]:                           # letting go, at rest: the W2 verifier's third finding)
                    for h in self.g1:
                        if np.linalg.norm(d.geom_xpos[h] - d.geom_xpos[g]) - m.geom_rbound[h] - m.geom_rbound[g] > 0.05:
                            continue
                        dist = float(mujoco.mj_geomDistance(m, d, g, h, 0.05, self.ft))
                        if self.hand_min is None or dist < self.hand_min:
                            self.hand_min = dist
                            self.hand_worst = (w.tick, s, m.geom(g).name, m.body(int(m.geom_bodyid[h])).name,
                                               pm.arms[sd].get("mode"), (pm.phases[0]["type"] if pm.phases else None))

    def result(self):
        runs = self.runs + ([self.run] if self.run else [])
        f_pain = self.w.f_pain
        t10 = self.link10_ticks
        out = dict(body_peak_N=round(self.body_peak, 1), total_peak_N=round(self.total_peak, 1),
                   total_steps_over_brief_cap=self.total_over_brief, total_steps_over_brief_cap_by_a_new_contact=self.over_brief_new,
                   total_steps_over_sustained_cap=self.total_over_two_steps,
                   press_runs_steps=dict(n=len(runs), max=max(runs, default=0), median=float(statistics.median(runs)) if runs else 0.0),
                   her_10ms_on_child_N=round(self.link10, 1), ticks_over_her_150N=int(sum(1 for x in t10 if x > K.HER_PAIN_N)),
                   ticks_over_f_pain=int(sum(1 for x in t10 if x > f_pain)), ticks_over_f_pain_by=dict(self.fp_by),
                   f_pain_events=list(self.fp_events), ticks=len(t10),
                   net_up_N=dict(ms10=round(self.up10, 1), tick=round(self.upT, 1)),
                   net_side_N=dict(ms10=round(self.side10, 1), tick=round(self.sideT, 1)),
                   body_work_J=dict(tick_max=round(self.work_max, 3), positive_sum=round(self.work_pos, 3),
                                    contact_max=round(max(self.ep_max, self.ep), 3), contact_positive_sum=round(self.ep_pos + max(0.0, self.ep), 3)))
        out["joint_law"] = dict(ticks_over_a_joints_limit=self.jt_over_ticks, largest_share_of_a_limit=round(self.jt_ratio, 3),
                                worst=self.jt_worst, base_10ms_N=round(self.base10, 1), ticks_base_over_f_pain=self.base_over_ticks,
                                by=dict(self.jt_by),
                                events=list(self.jt_events))
        out["region_N"] = {k: dict(ms10=round(self.reg10[k], 1), tick=round(self.reg_tick[k], 1)) for k in self.reg_names}
        out["step_peak_N"] = round(self.step_peak, 1); out["step_peak_at"] = self.step_peak_at
        out["trunk"] = dict(standoff_mm=None if self.standoff_min is None else round(self.standoff_min * 1e3, 1), standoff_at=self.standoff_worst,
                            touch_ticks=self.core_touch_ticks, longest_touch_run_ticks=self.core_run_max, press_down_N=round(self.core_down, 1))
        out["muscles"] = dict(largest_share_of_strength=round(self.act_ratio, 6), worst=self.act_worst, other_torque_max=self.applied_max)
        out["support"] = dict(force_max_N=round(self.sup_f, 2), force_up_min_N=None if self.sup_fz_min is None else round(self.sup_fz_min, 2),
                              torque_max_Nm=round(self.sup_t, 2))
        out["contact_depth_mm"] = {k: round(v * 1e3, 1) for k, v in self.depth.items()}
        out["depth_by_region_mm"] = {k: round(v * 1e3, 1) for k, v in self.depth_reg.items()}
        out["contact_depth_worst"] = self.depth_worst
        if self.hands:
            out["hand_to_hull_mm"] = None if self.hand_min is None else round(self.hand_min * 1e3, 1)
            out["hand_worst"] = self.hand_worst
        return out


# ---------------------------------------------------------------------------------------------------- the runner
HANDS = False                   # every run measures her hands against the G1 (the 'hands' scenario sets it)


def run(w, seq, N=600, after=0, on_tick=None, hands=None, babble=None):
    """ask for the acts in `seq` [(kind, target)], live until they end (and `after` ticks more), measuring (babble: a babbler whose
    acts the G1 makes every tick)"""
    pm = w.parent
    m, d = w.m, w.d
    probe = Probe(w, HANDS if hands is None else hands)
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
        w.apply(babble.acts() if babble is not None else {})
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
                         paused=inf.get("paused", 0), solve_ms=round(getattr(pm, "solve_ms", {}).get(i, 0.0)),
                         **{k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in inf.items()
                            if k2 in ("limb_push", "guide_cap", "palm_N", "closure_deg", "cleared", "shuffle_short_m", "replans")}))
    return dict(acts=acts, ticks=k + 1,
                holds={f"{n} ({kd})": dict(peak_N=round(pk, 1), cap_N=round(cap, 1)) for (n, kd), (pk, cap) in hold_pk.items()},
                effort_peak_N=round(pm.effort_peak, 1), over_sustained_s=round(over_s, 2),
                contact_peak_N={c: round(v, 1) for c, v in pm.stats["contact_peak"].items()}, pen_max_mm=round(pm.stats["pen_max"] * 1e3, 1),
                yield_ticks=pm.stats["yield_ticks"], jumps_refused=pm.stats.get("jumps_refused", 0), slips=pm.stats.get("slips", 0),
                her_hits=len(pm.hits), probe=probe.result(),
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
                    rec.setdefault("p0", p.copy())
                    if "p" not in rec or p[2] > rec["p"][2]:
                        rec["p"] = p.copy()                             # the held point at its highest while she guided it
        out = run(w, [("guide", which)], 700, on_tick=tick)
        if "p0" in rec:
            out["held_point_moved_cm"] = [round(x * 100, 1) for x in (rec["p"] - rec["p0"])]
        return out
    return f


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


def limp_arm(w, limb):
    """an instrument's switch: the G1's arm (its effector, e.g. 'arm_l') made limp, its actuators' gains and biases zeroed (A25's
    limp arm: the servos push nothing; its joints keep their passive damping and its weight)"""
    for j in range(w.eff_slices[limb].start, w.eff_slices[limb].stop):
        a = w.aid[j]
        w.m.actuator_gainprm[a, :] = 0.0; w.m.actuator_biasprm[a, :] = 0.0


def sc_guide_pace():
    """the guide's pace (A25: set on the limp arm, so a guide needs at most the arm's own push): the near forearm raised
    GUIDE_RAISE_M at each candidate pace, its arm limp (the instrument's switch), and again under its resting servo law; the peak
    spring force against the arm's own push at that pose. GUIDE_SPEED is the fastest pace within the push on the limp arm"""
    keep = K.GUIDE_SPEED
    rows = []
    try:
        for law in ("limp", "resting"):
            for v in (0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08):
                K.GUIDE_SPEED = v
                w = W.G1World(seed=1)
                if law == "limp":
                    ch = w.parent.child                                         # the near arm: on the side she kneels (its face side)
                    limp_arm(w, "arm_l" if ch.face_side() == "L" else "arm_r")
                out = run(w, [("guide", None)], 700)
                a = out["acts"][0]
                pk = max((h["peak_N"] for n, h in out["holds"].items() if "guide" in n), default=0.0)
                rows.append(dict(law=law, speed=v, status=a["status"], why=a["why"][:60], peak_N=pk, push_N=a.get("limb_push"),
                                 cap_N=a.get("guide_cap"), within_push=bool(a.get("limb_push") and pk <= a["limb_push"])))
    finally:
        K.GUIDE_SPEED = keep
    lim = [r for r in rows if r["law"] == "limp" and r["within_push"] and r["status"] == "done"]
    return dict(rows=rows, fastest_within_push_limp=max((r["speed"] for r in lim), default=None))


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


def kneel_time(w=None, step=0.01):
    """parent_motion.KNEEL_TIME measured (A25b: her carried body kneels down at a person's pace): parent_poses.kneel_down at 301 values of
    u (0 standing, 2 the tall kneel, 3 on her heels), each raised off the floor as her plan raises it (ParentMotion._floor_lift), and
    for each step of u the time its slowest-allowed segment needs: her pelvis at parent_consts.KNEEL_PELVIS_MPS, every other segment at
    KNEEL_SEG_MPS; the cumulative time at u = 0, 0.05, ... 3"""
    w = W.G1World(seed=1) if w is None else w
    pm = w.parent
    n = int(round(3.0 / step))
    us = np.linspace(0.0, 3.0, n + 1)
    pos = []
    for u in us:
        p = PM.P.kneel_down((0.0, 0.0), 0.0, float(u))
        segs = kin.fk(p)
        lift = pm._floor_lift(p)
        pos.append([segs[s_][0] + np.array([0.0, 0.0, lift]) for s_ in kin.SEGS])
    d = np.linalg.norm(np.diff(np.array(pos), axis=0), axis=2)
    lim = np.full(len(kin.SEGS), K.KNEEL_SEG_MPS)
    lim[kin.SEGS.index("pelvis")] = K.KNEEL_PELVIS_MPS
    T = np.concatenate([[0.0], np.cumsum((d / lim).max(axis=1))])
    k = int(round(0.05 / step))
    return tuple(round(float(T[i]), 4) for i in range(0, n + 1, k))


# ---------------------------------------------------------------------------------------------------- the W2 verifier's cases
PLACES = {"rot90": (G.BIRTH_XY, 90.0), "corner": ((-2.05, -1.75), 45.0), "wall": ((-2.2, -0.3), 90.0), "hall": ((3.9, -1.45), 90.0),
          "by_sofa": ((0.3, 1.15), 0.0), "by_table": ((0.25, 0.25), 0.0)}


def place_child(w, where):
    """the G1 as born (its birth joints) placed elsewhere on the floor (an instrument's placement, as place_g1): PLACES[where] =
    (its pelvis's floor point, its turn about the vertical from its birth heading)"""
    xy, yaw = PLACES[where]
    w.scene.place_on_mat(G.BIRTH, kin.rz(math.radians(yaw)) @ kin.ry(-math.pi / 2), xy)
    w.scene.set_parent(G.born_parent())
    w.d.qvel[:] = 0
    mujoco.mj_forward(w.m, w.d)
    w._sense_birth()
    w.parent = PM.ParentMotion(w)


def babbler(seed, p_rest):
    from sim_babble import Babbler
    return Babbler(seed=seed, p_rest=p_rest)


def ask_repeatedly(w, kinds, N, babble=None, hands=False):
    """her acts asked over and over (the next when the last has ended), the G1 babbling, N ticks: each act's outcome, and the
    probe's measures (with hands, her hands against its hulls)"""
    pm = w.parent
    probe = Probe(w, hands)
    asked = []
    for k in range(N):
        if not asked or (pm.status(asked[-1]) in PM.DONE_STATES and not pm.phases):
            asked.append(pm.request(Act(*kinds[len(asked) % len(kinds)])))
        w.apply(babble.acts() if babble is not None else {})
    outcome = {}
    for i in asked:
        st = pm.status(i)
        key = st if st != "refused" else "refused: " + pm.why(i).split(" (")[0][:70]
        outcome[key] = outcome.get(key, 0) + 1
    given_up = sum(v for k2, v in outcome.items() if k2.startswith("refused: given up"))
    return dict(asked=len(asked), outcome=outcome, given_up=given_up, jumps_refused=pm.stats.get("jumps_refused", 0), probe=probe.result())


def sc_babble(kinds=(("attend",),), seeds=(2, 3, 4, 7), N=600, p_rests=(0.6, 0.3), hands=False):
    """the babbling G1 (tools/sim_babble.py, p_rest 0.6 the design's sparse babble and 0.3 a busy one), her acts asked over and over:
    how many she gives up, and her force on the child (C8: under her own 150 N on most ticks, under F_pain always)"""
    def f():
        rows = []
        for pr in p_rests:
            for sd in seeds:
                w = W.G1World(seed=1)
                r = ask_repeatedly(w, [tuple(k) for k in kinds], N, babbler(sd, pr), hands)
                rows.append(dict(seed=sd, p_rest=pr, **r))
        return dict(rows=rows)
    return f


def sc_still(where=None, seq=(("attend", None), ("walk", "door")), N=900):
    """a still child (as born, placed elsewhere, or placed sitting: 'sit'), acts in turn: each one's outcome and ticks, her force
    on it"""
    def f():
        w = W.G1World(seed=1)
        if where == "sit":
            place_g1(w, "sit")
        elif where is not None:
            place_child(w, where)
        out = run(w, list(seq), N * len(seq))
        out["posture_after"] = w.parent.child.posture
        return out
    return f


class Strike:
    """C6's strike (an instrument, every physics step): the force on the G1's trunk, head and pelvis from what a fall strikes (the
    mat, the floor, the furniture, a toy), never from the G1's own links (its pelvis against its own hip housings is no fall: the
    W2 verifier's finding, 2.5-3.2 kN of self-contact had marked falls "not caught") nor from her hands catching it; as A12's
    pain measure, the largest 10 ms mean (5 steps) in the tick, per link"""

    def __init__(self, w):
        self.w = w; pm = w.parent; m = w.m
        self.links = {m.body(n).id for n in ("torso_link", "pelvis")}
        self.g1 = np.zeros(m.ngeom, dtype=bool)
        for g in range(m.ngeom):
            self.g1[g] = m.geom_bodyid[g] in w.scene.g1_set
        self.hers = pm.geom_seg >= 0
        self.win = {b: [0.0] * 5 for b in self.links}
        self.tick_max = 0.0
        self._orig = pm.after_step
        pm.after_step = self.after

    def after(self, s):
        self._orig(s)
        m, d = self.w.m, self.w.d
        per = {b: 0.0 for b in self.links}
        for i in range(d.ncon):
            c = d.contact[i]
            if c.efc_address < 0:
                continue
            g0, g1 = int(c.geom[0]), int(c.geom[1])
            for a_, b_ in ((g0, g1), (g1, g0)):
                lk = int(m.geom_bodyid[a_])
                if lk in self.links and not self.g1[b_] and not self.hers[b_]:
                    per[lk] += float(d.efc_force[c.efc_address])
        for b in self.links:
            q = self.win[b]; q.append(per[b]); del q[0]
            self.tick_max = max(self.tick_max, sum(q) / 5.0)

    def take(self):
        v = self.tick_max; self.tick_max = 0.0
        return v


def sc_catch(seeds=(1, 2, 3, 4, 5, 6, 7, 8), N=600, p_rest=0.6):
    """C6: the G1 placed in the leaning sit, her prop engaged beside it and eased to hovering (the instrument sets the easing's last
    step at once, and again whenever she holds the trunk back within 30 deg after a catch), then the G1 babbling. Every fall
    start is counted from the trunk itself (past 35 deg from vertical after sitting within 30): stopped when the trunk comes back
    within 30 deg, or is held short of CATCH_STOP_DEG until she lays it back; not stopped when it passes CATCH_STOP_DEG first or
    slips from her hands; and whether its trunk or head (the torso link carries both) or its pelvis struck the mat, the floor,
    the furniture or a toy at F_pain while it fell (Strike: the strike a catch prevents, never its own links pressing each other;
    its hands' and feet's own blows under babble are C5's). Her
    controller's mode when it started (hovering, holding) and how she ended it are written beside each"""
    falls = []
    ends = []
    for sd in seeds:
        w = W.G1World(seed=1)
        place_g1(w, "sit")
        pm = w.parent
        ch = pm.child
        fr = PM.unit(ch.torso_R[:, 0] * [1, 1, 0])[:2]
        lat = np.array([-fr[1], fr[0]])
        pm.place("heels", ch.pelvis[:2] + lat * 0.75 - fr * 0.10, math.atan2(-lat[1], -lat[0]))
        i = pm.request(Act("prop"))
        for _ in range(80):
            w.apply({})
            hs = [h for h in pm.holds if h.kind == "prop"]
            if len(hs) == 2 and all(h.ctl.get("go") for h in hs):
                break
        if len([h for h in pm.holds if h.kind == "prop"]) != 2:
            ends.append(dict(seed=sd, error=f"the prop did not engage: {pm.status(i)} {pm.why(i)}"))
            continue
        strike = Strike(w)
        b = babbler(sd, p_rest)
        cur = None; ready = True
        for k in range(N):
            live = [h for h in pm.holds if h.kind == "prop"]
            if cur is None and len(live) == 2 and all(h.ctl.get("mode") == "hold" for h in live) and \
                    pm.child.trunk_deg <= K.PROP_MAX_DEG:
                for h in live:
                    h.ctl.update(mode="hover", hover=K.HOVER_M); h.cap = 0.0
            strike.take()
            w.apply(b.acts())
            th = pm.child.trunk_deg
            hurt = strike.take()
            if cur is None:
                if th <= K.PROP_MAX_DEG:
                    ready = True
                elif ready and th > K.CATCH_DEG:
                    modes = [h.ctl.get("mode") for h in pm.holds if h.kind == "prop"]
                    cur = dict(seed=sd, t=w.tick, start_deg=round(th, 1), peak_deg=round(th, 1), her_mode=modes, trunk_pain_N=hurt)
                    ready = False
                continue
            cur["peak_deg"] = round(max(cur["peak_deg"], th), 1); cur["trunk_pain_N"] = round(max(cur["trunk_pain_N"], hurt), 1)
            modes = [h.ctl.get("mode") for h in pm.holds if h.kind == "prop"]
            laying = bool(modes) and all(m_ == "lay" for m_ in modes) and "lay" not in cur["her_mode"]   # held, now laid back by her
            if th <= K.PROP_MAX_DEG or th > K.CATCH_STOP_DEG or laying or not modes or k == N - 1:
                cur.update(end=w.tick, stopped=bool(th <= K.CATCH_STOP_DEG and modes), back_within_30=bool(th <= K.PROP_MAX_DEG),
                           laid_back=laying, her_end=(pm.stats.get("falls") or [{}])[-1].get("how"), prop=pm.status(i))
                cur["caught"] = bool(cur["stopped"] and cur["trunk_pain_N"] < w.f_pain)
                falls.append(cur); cur = None
            if pm.status(i) in PM.DONE_STATES and cur is None:
                ends.append(dict(seed=sd, prop_ended=pm.why(i)[:80] or "her holds on it gone", at_tick=w.tick))
                break
    return dict(falls=falls, n=len(falls), caught=sum(1 for f_ in falls if f_["caught"]),
                share=round(sum(1 for f_ in falls if f_["caught"]) / len(falls), 3) if falls else None, ends=ends,
                brief_caps_note=f"her brief caps return only after {K.BRIEF_REST_S:g} s under her sustained ones: a second catch "
                                "within that pushes at her sustained caps")


def sc_hands():
    """her hands against the G1 in every act that puts a hand on it (C21's holds; the W2 verifier's fourth finding)"""
    global HANDS
    HANDS = True
    try:
        out = {}
        for n in ("attend", "guide", "guide_far", "knee_over", "prop", "feed", "pull_to_sit", "turn", "hand_over_placed"):
            r = SCENARIOS[n]()
            out[n] = dict(acts=[(a["kind"], a["status"], a["why"][:60]) for a in r["acts"]],
                          hand_to_hull_mm=r.get("probe", {}).get("hand_to_hull_mm"), hand_worst=r.get("probe", {}).get("hand_worst"))
        return out
    finally:
        HANDS = False


def sc_copy_do():
    """P3's interface: every kind in DOES asked through 'do', and every movement 'copy' carries, from the door and kneeling beside
    the child"""
    rows = []
    for where in ("door", "kneeling"):
        for kind, tgt in [("do", k) for k in PM.DOES] + [("copy", f"{k}:{s_}") for k in ("arm_raise", "wave", "shake", "open_hand")
                                                            for s_ in ("left", "right")]:
            w = W.G1World(seed=1)
            if where == "kneeling":
                ch = w.parent.child
                mid = (ch.torso[:2] + ch.pelvis[:2]) / 2
                w.parent.place("heels", mid + ch.lat[:2] * 0.84 + ch.len_axis[:2] * 0.10, math.atan2(-ch.lat[1], -ch.lat[0]))
            r = run(w, [(kind, tgt)], 700)
            a = r["acts"][0]
            rows.append(dict(where=where, kind=kind, target=tgt, status=a["status"], why=a["why"][:60], ticks=a["ticks"],
                             body_peak_N=r["probe"]["body_peak_N"]))
    return dict(DOES=list(PM.DOES), rows=rows)


BABBLE_ACTS = (("attend",), ("show", "block"), ("lean_in",), ("touch", "tummy"))   # the W2 verifier's mix


def sc_c8(seeds=tuple(range(1, 13)), N=600, p_rests=(0.3, 0.6), mixes=("attend", "acts")):
    """C8 over many seeds (the lead's decision of 2026-09-25: she is a body): the babbling G1 at p_rest 0.3 and 0.6, attend alone
    asked over and over and the verifier's mix: per run her 10 ms force on any link (her holds and her body together) against
    F_pain (each tick over it told as hers or the child's: Probe), the ticks over her own 150 N, her hands', forearms' and body's deepest contact into it, her body's work on it, her
    whole force on it upward and along the floor, and how many acts she did and refused"""
    def f():
        rows = []
        for mix in mixes:
            kinds = [("attend",)] if mix == "attend" else list(BABBLE_ACTS)
            for pr in p_rests:
                for sd in seeds:
                    w = W.G1World(seed=1)
                    t0 = time.time()
                    r = ask_repeatedly(w, kinds, N, babbler(sd, pr))
                    p_ = r["probe"]
                    done = sum(v for k2, v in r["outcome"].items() if k2 == "done")
                    rows.append(dict(mix=mix, seed=sd, p_rest=pr, her_10ms_N=p_["her_10ms_on_child_N"], ticks_over_f_pain=p_["ticks_over_f_pain"],
                                     ticks_over_f_pain_by=p_["ticks_over_f_pain_by"], f_pain_events=p_["f_pain_events"],
                                     ticks_over_her_150N=p_["ticks_over_her_150N"], depth_mm=p_["contact_depth_mm"], depth_worst=p_["contact_depth_worst"],
                                     work_J=p_["body_work_J"], net_up_N=p_["net_up_N"], net_side_N=p_["net_side_N"], asked=r["asked"], done=done,
                                     outcome=r["outcome"], jumps_refused=r["jumps_refused"], tick_ms=round((time.time() - t0) / N * 1e3, 1)))
                    print(json.dumps(rows[-1]), flush=True)
        return dict(rows=rows, f_pain=round(W.G1World(seed=1).f_pain, 1))
    return f


def replay_proc(mode, path, seed=3, p_rest=0.3, t0=150, n=60):
    """EXACT REPLAY ACROSS PROCESSES (an instrument): mode 'save' lives from birth to tick t0 under babble with her act mix asked
    in turn and her report read every tick, writes the world's save and the babbler's and the asks' state to path + '.blob', then
    lives n ticks more; 'load' restores that save in a new process and lives the same n ticks; 'birth' lives from birth to t0 + n.
    Each writes, for each of the last n ticks, a digest of the frame's channels, her report, her motion's state, the physics'
    qpos, qvel, qfrc_applied and xfrc_applied (her support's among them), the controls and the actuators' forces (her muscles'
    among them), and the final save's, to path + '.' + mode"""
    import hashlib
    import pickle
    w = W.G1World(seed=1)
    pm = w.parent
    b = babbler(seed, p_rest)
    asked = []
    if mode == "load":
        st = pickle.loads(open(path + ".blob", "rb").read())
        w.load_state(st["world"]); b.load(st["babbler"]); asked = list(st["asked"])
        start = w.tick
    else:
        start = 0
    rows = []

    def h(x):
        return hashlib.sha256(x).hexdigest()[:16]
    while w.tick < t0 + n:
        if mode == "save" and w.tick == t0:
            open(path + ".blob", "wb").write(pickle.dumps(dict(world=w.save_state(), babbler=b.state(), asked=list(asked))))
        if not asked or (pm.status(asked[-1]) in PM.DONE_STATES and not pm.phases):
            asked.append(pm.request(Act(*BABBLE_ACTS[len(asked) % len(BABBLE_ACTS)])))
        w.apply(b.acts())
        f = w.frame()
        rep = pm.report(w.tick)
        if w.tick > t0:
            d = w.d
            rows.append([w.tick, h(b"".join(np.ascontiguousarray(f.obs[k]).tobytes() for k in sorted(f.obs))),
                         h(repr(sorted((k, str(v)) for k, v in rep.items())).encode()), h(pickle.dumps(pm.state())),
                         h(d.qpos.tobytes()), h(d.qvel.tobytes()), h(d.qfrc_applied.tobytes()), h(d.xfrc_applied.tobytes()),
                         h(d.ctrl.tobytes()), h(d.actuator_force.tobytes())])
    rows.append(["save", h(w.save_state())])
    open(path + "." + mode, "w").write(json.dumps(dict(start=start, rows=rows, stops=pm.stats.get("over_run", 0),
                                                        yield_ticks=pm.stats.get("yield_ticks", 0))))


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
    "prop": sc_prop,
    "turn": sc_turn,
    "feed": sc_feed,
    "guide_pace": sc_guide_pace,
    "costs": sc_costs,
    "babble_attend": sc_babble(),
    "babble_acts": sc_babble(kinds=BABBLE_ACTS, seeds=(1, 2, 3, 4, 5, 6, 7, 8)),
    "babble_hands": sc_babble(kinds=BABBLE_ACTS, seeds=(1, 6, 8), p_rests=(0.3,), hands=True),
    "still_door": sc_still(None, (("attend", None), ("walk", "door"))),
    "still_sofa": sc_still(None, (("attend", None), ("walk", "sofa"))),
    "still_lean_bring": sc_still(None, (("lean_in", None), ("bring_back", "car"))),
    **{f"still_{p_}": sc_still(p_, (("attend", None), ("lean_in", None), ("show", "block")))
       for p_ in ("rot90", "corner", "wall", "hall", "by_sofa", "by_table")},
    "still_sit_lean": sc_still("sit", (("lean_in", None),)),
    "catch": sc_catch,
    "catch_rest": lambda: sc_catch(seeds=(1,), N=600, p_rest=1.0),
    "hands": sc_hands,
    "copy_do": sc_copy_do,
    "c8": sc_c8(),
}


LONG = ("guide_pace", "babble_attend", "babble_acts", "babble_hands", "catch", "catch_rest", "hands", "copy_do", "c8")   # run only when named (each takes minutes)


def main():
    rp = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--replay=")), None)
    if rp is not None:                                                  # --replay=MODE:PATH[:seed:p_rest:t0:n] (replay_proc)
        parts = rp.split(":")
        extra = [int(parts[2]), float(parts[3]), int(parts[4]), int(parts[5])] if len(parts) > 2 else []
        replay_proc(parts[0], parts[1], *extra)
        return
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or [k for k in SCENARIOS if k not in LONG]
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
