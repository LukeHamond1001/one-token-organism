"""THE GHOST (W7, 2026-10-09, the owner's word at 15:00: 'we kinda cheat and force it to stand up and walk ... a ghost teacher that
constrains the child to walk, and gives dopamine while it does'): THE ROOM'S GUIDANCE FIELD, FADING.

What it is. A field of the room, not of the child: a hold on the TORSO toward its standing height (a spring up, slack above: it
never pushes down) with a gentle lead forward at a walking pace, as her hands held and led the chest (C268), and a torque on the
PELVIS that rights it and turns it to a heading, all
scaled by one number, the STRENGTH s in [0, 1], recorded every tick. At 1 the ghost does everything her hands did in a held walk
(the chest led up and forward, the stepping reflex A193 and the postural tone A195 doing the legs); at 0 it is gone and the body
stands and walks on its own or falls. The strength FALLS on every tick the body bears its own weight (the hold carrying less than
BEARS of it: a parent's hands lighten as the baby takes its weight), RISES while the ghost carries more than HANG of it (it hangs),
and returns to 1 when it falls (the catch: the harness). (The first law, 2026-10-09 16:30, took the child's own hip acts' agreement
with its hips' motion: its leg policy acts every tick and the sign agreed by chance, a coin's fade; withdrawn the same hour.) The steps are the lane's 'stepped' events as before
(the pelvis carried STEP_M, standing), paid by her smile as every step is (C284): the reward channel is not touched. Nothing is
written into the child. The walls: the lead turns before anything standing (her C345 rule). The constants: ours, of the order of
her hands' (HOLD_K 2000 N/m, her lead 33 to 116 N) and of the body's weight (337 N).

Why. 134 days of her hands: stands 60 to 200 a day, held steps 30 to 90, unaided steps 0 to 1 a day; her scripts the bottleneck
(the owner, 2026-10-09). The ghost is disclosed as the environment's act, the strength is the ruler, and the demo's claim is only
what the body does at strength 0. On the real body the same field is the tether's slack."""
import math

import numpy as np
import mujoco

K_Z, C_Z, F_Z_MAX = 5000.0, 300.0, 500.0     # N/m, N s/m, N: the hold up on the torso's centre of mass (ours: her HOLD_K 2000; the cap
                                             # its weight 337 N and a little, so it can be lifted from lying)
K_R, C_R, T_R_MAX = 150.0, 30.0, 120.0       # N m/rad, N m s/rad, N m: the righting torque toward upright (ours)
K_YAW, C_YAW, T_YAW_MAX = 30.0, 8.0, 30.0    # the turn toward the heading (ours)
K_XY, C_XY, F_XY_MAX = 300.0, 40.0, 60.0     # the lead's spring and its cap (ours: within her lead's 33 to 116 N)
SPEED = float(__import__('os').environ.get('GHOST_SPEED', 0.05))   # m/s: the lead's pace (ours: her held walks went 0.2 to 0.5 m in a 100-tick stand)
LEAD_M = 0.30                                # m: the lead never farther ahead of the torso than this (a step)
LIFT_TICKS, LOWER_TICKS = 20, 20             # ticks: the rise from lying to standing height, the lowering at the end (3 s each; ours)
HOLD_DROP = 0.0                              # m: the hold's height under the torso's standing height (ours: none; the cap below decides what it bears)
CARRY_SHARE = 0.5                            # the most of its weight the hold carries once it stands (ours: her two hands' sustained cap, 156 N, is 0.46
                                             # of it; 17:50: with the whole weight on offer, the copy hung, the soles unloaded and the standing reflex off)
# 17:10-17:40, the copies: held and righted at the torso it walked (900 ticks: 16 steps, 9.1 m, 2 falls) but the righting torque went
# through the waist pitch joint at its load line (54 of 62 pain ticks); held and righted at the pelvis, a walker's seat, no pain and
# no walking: the legs hung bent, the soles unloaded, the standing reflex off, the seat carrying 0.99 of its weight. So the hold and
# the lead at the torso, where her hands were, and the righting at the pelvis, which the floor and the strong hips react
LOWER_Z = 0.30                               # m: the pelvis at the lowering's end (a crouch; the day plan lays it on its back from there)
FADE = 1.0 / 2000.0                          # the strength's fall a tick on which the body bears its own weight (ours: a block of bearing takes it to 0)
RISE = 0.002                                 # the strength's rise a tick on which the ghost carries more than HANG of its weight (ours: 500 ticks hanging bring it back)
HANG, BEARS = 0.6, 0.3                       # the share of its weight carried that counts as hanging; under which it bears its own (ours)
FALL_Z, FALL_UP = 0.45, 0.5                  # fallen: the pelvis below this (m), or the trunk's long axis past 60 deg from vertical (ours)
CLEAR_M = 0.5                                # m: the lead turns before anything standing nearer than this (C345's WALL)
TICK_S = 0.150
G = 9.81


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Ghost:
    def __init__(self, w):
        m = w.m
        self.b = int(m.body("torso_link").id); self.pelvis = int(w.pelvis_id)   # the held link (the torso) and the righted one (the pelvis)
        d0 = mujoco.MjData(m); d0.qpos[:] = m.qpos0
        mujoco.mj_kinematics(m, d0); mujoco.mj_comPos(m, d0)
        self.z_stand = float(d0.xipos[self.b][2]) - HOLD_DROP          # the torso's centre of mass with the G1 standing at its qpos0, less the drop
        self.pelvis_stand = float(d0.xpos[self.pelvis][2])              # (the G1's own standing height: sourced, not ours)
        del d0
        self.weight = float(w.body_mass) * G
        self.on = False; self.s = 0.0; self.heading = 0.0; self.lead = np.zeros(2)
        self.ticks = 0; self.lower = 0; self.falls = 0; self.turns = 0; self.z_from = 0.0; self.z_to = self.z_stand
        self.last = np.zeros(6); self.last_p = np.zeros(6); self.fz_sum = 0.0; self.n = 0; self.metres = 0.0; self.pxy = None
        self.rec = None

    # ------------------------------------------------------------------ the day plan's switch
    def on_(self, w, heading=None):
        d = w.d
        R = d.xmat[self.b].reshape(3, 3)
        fwd = R[:, 0]
        self.heading = float(math.atan2(fwd[1], fwd[0])) if heading is None else float(heading)
        self.on = True; self.s = 1.0; self.ticks = 0; self.lower = 0
        self.z_from = float(d.xipos[self.b][2]); self.z_to = self.z_stand; self.lead = np.asarray(d.xipos[self.b][:2], float).copy()
        self.pxy = np.asarray(d.xpos[self.pelvis][:2], float).copy(); self.metres = 0.0
        self.fz_sum = 0.0; self.n = 0

    def off_(self, w=None, at_once=False):
        if not self.on:
            return
        if at_once or w is None:
            if w is not None:
                w.d.xfrc_applied[self.b] -= self.last; w.d.xfrc_applied[self.pelvis] -= self.last_p
            self.last[:] = 0.0; self.last_p[:] = 0.0; self.on = False; self.lower = 0; self.s = 0.0; self.rec = None
            return
        self.lower = LOWER_TICKS; self.z_from = float(w.d.xipos[self.b][2]); self.z_to = LOWER_Z

    def clear(self):
        """the applied wrenches forgotten (the world's fault roll-back cleared xfrc_applied, W6)"""
        self.last[:] = 0.0; self.last_p[:] = 0.0

    # ------------------------------------------------------------------ the physics step
    def before_step(self, w):
        m, d, b = w.m, w.d, self.b
        bp = self.pelvis
        if not d.xfrc_applied[b].any():                                 # (the row cleared by the world's laying, carry_to_mat, or a
            self.last[:] = 0.0                                          # fault's roll-back: nothing of ours is in it to take back)
        if not d.xfrc_applied[bp].any():
            self.last_p[:] = 0.0
        d.xfrc_applied[b] -= self.last; d.xfrc_applied[bp] -= self.last_p
        s = self.s
        p = d.xipos[b]
        vel = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
        w_ang, v_lin = vel[:3], vel[3:]
        if self.lower > 0:
            prog = 1.0 - self.lower / LOWER_TICKS; frac = self.lower / LOWER_TICKS
        else:
            prog = frac = min(1.0, self.ticks / LIFT_TICKS)
        z_t = self.z_from + (self.z_to - self.z_from) * prog
        fz = s * (K_Z * (z_t - float(p[2])) - C_Z * float(v_lin[2]))
        cap = F_Z_MAX if (self.ticks <= LIFT_TICKS or self.lower > 0) else CARRY_SHARE * self.weight   # the lift may carry it; standing, the legs must
        fz = min(max(fz, 0.0), cap)                                       # up only: slack above
        velp = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bp, velp, 0)
        w_ang = velp[:3]
        R = d.xmat[bp].reshape(3, 3)                                    # the pelvis righted and turned
        up = R[:, 2]
        tau = K_R * np.cross(up, np.array([0.0, 0.0, 1.0])) - C_R * w_ang
        tau[2] = 0.0
        n_ = float(np.linalg.norm(tau))
        if n_ > T_R_MAX:
            tau *= T_R_MAX / n_
        fwd = R[:, 0]
        yaw = math.atan2(float(fwd[1]), float(fwd[0]))
        tz = K_YAW * _wrap(self.heading - yaw) - C_YAW * float(w_ang[2])
        tau[2] = min(max(tz, -T_YAW_MAX), T_YAW_MAX)
        tau *= s * frac
        fxy = np.zeros(2)
        if frac >= 1.0 and self.lower == 0:
            fxy = s * (K_XY * (self.lead - np.asarray(p[:2], float)) - C_XY * np.asarray(v_lin[:2], float))
            n_ = float(np.linalg.norm(fxy))
            if n_ > F_XY_MAX:
                fxy *= F_XY_MAX / n_
        wrench = np.array([fxy[0], fxy[1], fz, 0.0, 0.0, 0.0]); wrench_p = np.array([0.0, 0.0, 0.0, tau[0], tau[1], tau[2]])
        d.xfrc_applied[b] += wrench; d.xfrc_applied[bp] += wrench_p
        self.last = wrench; self.last_p = wrench_p
        self.fz_sum += fz; self.n += 1

    # ------------------------------------------------------------------ the tick's end
    def tick_end(self, w, acts, digits, settings):
        m, d = w.m, w.d
        self.ticks += 1
        share = (self.fz_sum / self.n) / self.weight if self.n else 0.0
        self.fz_sum = 0.0; self.n = 0
        pz = float(d.xpos[self.pelvis][2]); pxy = np.asarray(d.xpos[self.pelvis][:2], float)
        if self.lower > 0:
            self.lower -= 1
            if self.lower == 0:
                self.off_(w, at_once=True)
                return
        elif self.ticks > LIFT_TICKS:
            up = d.xmat[self.b].reshape(3, 3)[:, 2]
            fallen = pz < FALL_Z or float(up[2]) < FALL_UP
            if fallen:                                                      # the catch: the harness takes it again. The first catch
                self.s = 1.0; self.ticks = 0; self.falls += 1               # (16:40) restarted the lift from the fallen pose and hauled a
                lay = getattr(w, "carry_to_mat", None)                   # child on its front up by the chest, head down (the copy: pain 13
                if lay is not None:                                         # ticks in 30); the environment lays it on its back where it fell
                    try:                                                    # first (its laying, C338 amended), and the lift is from lying
                        lay(to=np.asarray(d.qpos[:2], float))
                    except Exception:
                        pass
                    self.last[:] = 0.0; self.last_p[:] = 0.0                # (the laying clears the applied forces)
                self.z_from = float(d.xipos[self.b][2]); self.z_to = self.z_stand; self.lead = np.asarray(d.xipos[self.b][:2], float).copy()
            else:
                fwd = np.array([math.cos(self.heading), math.sin(self.heading)])
                txy = np.asarray(d.xipos[self.b][:2], float)
                self.lead = self.lead + SPEED * TICK_S * fwd
                ahead = float((self.lead - txy) @ fwd)
                self.lead = txy + fwd * min(max(ahead, 0.0), LEAD_M)          # along the heading, a step ahead at most
                probe = txy + fwd * (LEAD_M + CLEAR_M)
                par = getattr(w, "parent", None)
                clear = getattr(par, "_standing_clear_m", None)
                if clear is not None:
                    for _ in range(3):
                        if float(clear(probe)) >= CLEAR_M:
                            break
                        self.heading = _wrap(self.heading + math.pi / 2); self.turns += 1
                        fwd = np.array([math.cos(self.heading), math.sin(self.heading)])
                        probe = txy + fwd * (LEAD_M + CLEAR_M); self.lead = txy.copy()
                if share > HANG:
                    self.s = min(1.0, self.s + RISE)
                elif share < BEARS:
                    self.s = max(0.0, self.s - FADE)
        if self.pxy is not None and pz > FALL_Z:
            self.metres += float(np.linalg.norm(pxy - self.pxy))
        self.pxy = pxy.copy()
        self.rec = [round(self.s, 4), round(share, 3), round(self.heading, 2), int(self.falls), round(self.metres, 2)]

    # ------------------------------------------------------------------ the save
    def state(self):
        return dict(on=bool(self.on), s=float(self.s), heading=float(self.heading), lead=[float(x) for x in self.lead],
                    ticks=int(self.ticks), lower=int(self.lower), falls=int(self.falls), turns=int(self.turns), z_from=float(self.z_from), z_to=float(self.z_to),
                    last=[float(x) for x in self.last], metres=float(self.metres),
                    pxy=None if self.pxy is None else [float(x) for x in self.pxy])

    def load_state(self, st):
        if not st:
            self.on = False; self.s = 0.0; self.last[:] = 0.0; self.rec = None; self.lower = 0
            return
        self.on = bool(st["on"]); self.s = float(st["s"]); self.heading = float(st["heading"]); self.lead = np.asarray(st["lead"], float).copy()
        self.ticks = int(st["ticks"]); self.lower = int(st["lower"]); self.falls = int(st["falls"]); self.turns = int(st.get("turns", 0))
        self.z_from = float(st["z_from"]); self.z_to = float(st.get("z_to", self.z_stand)); self.last[:] = 0.0; self.last_p[:] = 0.0; self.metres = float(st["metres"])   # (xfrc_applied is not in the
                                                                                        # physics state: a loaded world applies afresh)
        self.pxy = None if st.get("pxy") is None else np.asarray(st["pxy"], float).copy()
        self.rec = None
