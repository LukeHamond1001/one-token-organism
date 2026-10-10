"""THE GHOST (W7, 2026-10-09, the owner's word at 15:00: 'we kinda cheat and force it to stand up and walk ... a ghost teacher that
constrains the child to walk, and gives dopamine while it does'): THE ROOM'S GUIDANCE FIELD, FADING.

What it is. A field of the room, not of the child: a hold on the TORSO toward its standing height (a spring up, slack above: it
never pushes down) with a gentle lead forward at a walking pace, as her hands held and led the chest (C268), and a torque on the
PELVIS that rights it and turns it to a heading, all
scaled by one number, the STRENGTH s in [0, 1], recorded every tick. At 1 the ghost does everything her hands did in a held walk
(the chest led up and forward, the stepping reflex A193 and the postural tone A195 doing the legs); at 0 it is gone and the body
stands and walks on its own or falls. The strength FALLS on every tick the body bears its own weight (the hold carrying less than
BEARS of it: a parent's hands lighten as the baby takes its weight), RISES while the ghost carries more than HANG of it (it hangs),
and returns to 1 when it falls (the catch: the harness). (The first law, 2026-10-09 16:00, took the child's own hip acts' agreement
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
SPEED = float(__import__('os').environ.get('GHOST_SPEED', 0.10))   # m/s: the lead's pace (ours: her held walks went 0.2 to 0.5 m in a 100-tick stand). C350 amended
                                             # (17:50): 0.05 in the first live block (day 135: 2,432 ticks, 64 steps, 10 falls caught, pain 1.1%, the
                                             # strength 1.0 to 0.95); the copy at 0.15 doubled the steps in 600 ticks (14 against 7) with 2 falls against
                                             # none and less weight borne (the strength 0.996 against 0.948); 0.10 from the day's second block
METRES_WIN = 10                              # ticks: the metres are the pelvis's displacement over this window, not its sway tick by tick (C350 amended:
                                             # the first block's 47 m were mostly sway, 2 cm a tick; ours)
LEAD_M = 0.30                                # m: the lead never farther ahead of the torso than this (a step)
LIFT_TICKS, LOWER_TICKS = 20, 20             # ticks: the rise from lying to standing height, the lowering at the end (3 s each; ours)
HOLD_DROP = 0.0                              # m: the hold's height under the torso's standing height (ours: none; the cap below decides what it bears)
CARRY_SHARE = 0.5                            # the most of its weight the hold carries once it stands (ours: her two hands' sustained cap, 156 N, is 0.46
                                             # of it; 16:20: with the whole weight on offer, the copy hung, the soles unloaded and the standing reflex off)
# 16:00-16:20, the copies (the clock read right at 16:25): held and righted at the torso it walked (900 ticks: 16 steps, 9.1 m, 2 falls) but the righting torque went
# through the waist pitch joint at its load line (54 of 62 pain ticks); held and righted at the pelvis, a walker's seat, no pain and
# no walking: the legs hung bent, the soles unloaded, the standing reflex off, the seat carrying 0.99 of its weight. So the hold and
# the lead at the torso, where her hands were, and the righting at the pelvis, which the floor and the strong hips react
LOWER_Z = 0.30                               # m: the pelvis at the lowering's end (a crouch; the day plan lays it on its back from there)
FADE = 1.0 / 2000.0                          # the strength's fall a tick on which the body bears its own weight (ours: a block of bearing takes it to 0)
RISE = 0.002                                 # the strength's rise a tick on which the ghost carries more than HANG of its weight (ours: 500 ticks hanging bring it back)
HANG, BEARS = 0.6, 0.3                       # the share of its weight carried that counts as hanging; under which it bears its own (ours)
EDGE_UP, EDGE_START = 0.10, 0.20             # W7c (2026-10-09, 20:50, the owner's word: as fast as possible): PRACTICE AT THE EDGE. A fall raises the
                                             # strength by EDGE_UP over where it fell, not back to 1, and a block begins EDGE_START over the last fall's
                                             # strength: day 136's blocks cycled 1 to 0.2 and fell there (0.23, 0.26, 0.21), most of each block spent
                                             # at help levels where nothing is at stake; now nearly every tick is walked near the edge, the fade
                                             # (FADE a tick) bringing it back to the edge in 200 ticks and a little past it each time. Ours
FALL_Z, FALL_UP = 0.45, 0.5                  # fallen: the pelvis below this (m), or the trunk's long axis past 60 deg from vertical (ours)
CLEAR_M = 0.5                                # m: the lead turns before anything standing nearer than this (C345's WALL)
TICK_S = 0.150
G = 9.81
# THE EXPERT GAIT (W7b, 2026-10-09 18:40, the owner's word: 'find a Unitree G1 walking script to help our ghost, I want this fast'):
# Unitree's own pretrained G1 locomotion policy (unitree_rl_gym, deploy/pre_train/g1/motion.pt, TorchScript; its MuJoCo deploy
# config g1.yaml: 50 Hz, PD gains kps/kds, default angles, scales, a 0.8 s gait phase, 47 observations, 12 leg actions; downloaded
# 2026-10-09 18:35 to data/expert/unitree_g1, never in git) runs inside the room's loop and its PD torques for the twelve leg
# joints are the ghost's hands at the legs, scaled by the strength and capped at each joint's own torque limit (the room's hands
# no stronger than the robot's motors), added to the joint as an outside torque (qfrc_applied, as the passive tissue is): the
# child's servos, acts and reflexes run underneath untouched, nothing of the policy enters the brain. With the expert on, the lead
# is the policy's own command (EXPERT_V forward, a turn toward the heading) and the torso hold hangs slack EXPERT_DROP under the
# standing height (the policy stands with its knees at 0.3 rad), catching only a sink. The policy's numbers are Unitree's; ours: the
# command, the drop, the cap
POLICY_PATH = __import__('os').environ.get('GHOST_POLICY', '/Users/lukehamond/Projects/project/data/expert/unitree_g1/motion.pt')
EXPERT_KP = (100., 100., 100., 150., 40., 40., 100., 100., 100., 150., 40., 40.)   # Unitree's g1.yaml
EXPERT_KD = (2., 2., 2., 4., 2., 2., 2., 2., 2., 4., 2., 2.)
EXPERT_DEFAULT = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0)
EXPERT_ANG_VEL_SCALE, EXPERT_DOF_VEL_SCALE, EXPERT_ACTION_SCALE = 0.25, 0.05, 0.25
EXPERT_CMD_SCALE = (2.0, 2.0, 0.25)
EXPERT_DECIMATION, EXPERT_DT, EXPERT_PERIOD = 10, 0.002, 0.8   # the policy every 10 physics steps of 2 ms (50 Hz); the gait's phase
EXPERT_V = float(__import__('os').environ.get('GHOST_EXPERT_V', 0.4))   # m/s: the walk commanded (ours; Unitree's deploy walks at 0.5)
EXPERT_TURN, EXPERT_WZ_MAX = 1.0, 0.5    # the turn command per rad of heading error, and its cap (ours)
EXPERT_DROP = 0.08                       # m: the torso hold's slack under the standing height with the expert on (ours: the policy's stance)
EXPERT_LOOK_M = 1.5                      # m: with the expert on, the walls are looked for this far ahead (ours: 4 s of its walk; at 0.8 m it
                                         # turned too late at 0.4 m/s and walked through the second room's wall into the void, day 135, 6,499,0xx)
TURN_DONE = 0.3                          # rad: a turn in place is over when the pelvis faces within this of the heading (ours)
LAY_CLEAR_M = 1.0                        # m: the catch lays it where it fell only this clear of anything standing, else on its mat (ours: C281's rule)
ROOMS = ((-2.6, 2.6, -2.3, 2.3), (2.7, 5.9, -3.3, 0.3))   # the rooms' floor plans (make_g1room ROOM_X/ROOM_Y; g1scene ROOM2), x0, x1, y0, y1
ROOM_SLACK = 0.3                         # m: beyond the plans by this the child has left the rooms: carried to its mat (ours)


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
        self.last = np.zeros(6); self.last_p = np.zeros(6); self.fz_sum = 0.0; self.n = 0; self.metres = 0.0; self.pxy = None; self.win = []
        self.rec = None
        self.expert = None; self.e_action = np.zeros(12); self.e_target = np.array(EXPERT_DEFAULT); self.e_count = 0; self.e_steps = 0
        self.e_idx = list(w.eff_joint_idx["leg_l"]) + list(w.eff_joint_idx["leg_r"])   # the twelve leg joints in the policy's order
        self.e_qadr = np.asarray(w.qadr)[self.e_idx]; self.e_dof = np.asarray(w.dof)[self.e_idx]; self.e_lim = np.asarray(w.tau_max, float)[self.e_idx]
        self.e_kp = np.array(EXPERT_KP); self.e_kd = np.array(EXPERT_KD); self.e_default = np.array(EXPERT_DEFAULT)
        self.e_aid = np.asarray(w.aid)[self.e_idx]; self.e_lo = np.asarray(w.lo, float)[self.e_idx]; self.e_hi = np.asarray(w.hi, float)[self.e_idx]
        self.use_expert = bool(int(__import__('os').environ.get('GHOST_EXPERT', '1')))
        self.turning = False; self.left_room = 0; self.edge = None   # W7c: the strength at the last fall (None: no fall yet)
        self.base = int(w.base_dof)

    def _policy(self):
        if self.expert is None and self.use_expert:
            import os, torch
            if os.path.exists(POLICY_PATH):
                self.expert = torch.jit.load(POLICY_PATH).eval()
            else:
                self.use_expert = False
        return self.expert

    # ------------------------------------------------------------------ the day plan's switch
    def on_(self, w, heading=None):
        d = w.d
        R = d.xmat[self.b].reshape(3, 3)
        fwd = R[:, 0]
        self.heading = float(math.atan2(fwd[1], fwd[0])) if heading is None else float(heading)
        self.on = True; self.s = 1.0 if self.edge is None else min(1.0, self.edge + EDGE_START); self.ticks = 0; self.lower = 0   # W7c
        self.z_from = float(d.xipos[self.b][2]); self.z_to = self.z_stand; self.lead = np.asarray(d.xipos[self.b][:2], float).copy()
        self.pxy = np.asarray(d.xpos[self.pelvis][:2], float).copy(); self.metres = 0.0; self.win = []
        self.fz_sum = 0.0; self.n = 0
        self.e_action[:] = 0.0; self.e_target = self.e_default.copy(); self.e_count = 0; self.e_steps = 0

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
        pol = self._policy() if self.ticks > LIFT_TICKS or self.lower == 0 else None
        expert_on = pol is not None and self.use_expert and self.lower == 0 and self.ticks > LIFT_TICKS
        z_t = self.z_from + (self.z_to - self.z_from) * prog - (EXPERT_DROP if expert_on else 0.0)
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
        if frac >= 1.0 and self.lower == 0 and not expert_on:
            fxy = s * (K_XY * (self.lead - np.asarray(p[:2], float)) - C_XY * np.asarray(v_lin[:2], float))
            n_ = float(np.linalg.norm(fxy))
            if n_ > F_XY_MAX:
                fxy *= F_XY_MAX / n_
        wrench = np.array([fxy[0], fxy[1], fz, 0.0, 0.0, 0.0]); wrench_p = np.array([0.0, 0.0, 0.0, tau[0], tau[1], tau[2]])
        d.xfrc_applied[b] += wrench; d.xfrc_applied[bp] += wrench_p
        self.last = wrench; self.last_p = wrench_p
        if expert_on:
            self._expert_step(w, d, s)
        self.fz_sum += fz; self.n += 1

    def _expert_step(self, w, d, s):
        """W7b: one physics step of the expert gait: every EXPERT_DECIMATION steps the policy reads the pelvis's angular velocity and
        gravity direction (the free joint's own frame, as Unitree's deploy reads them), the command, the legs' angles and speeds, its
        last action and the gait's phase, and sets the twelve leg targets; every step its PD torque toward them, scaled by s and
        capped at each joint's torque limit, is the room's torque on the joint"""
        import torch
        self.e_count += 1
        q = d.qpos[self.e_qadr]; dq = d.qvel[self.e_dof]
        if self.e_count % EXPERT_DECIMATION == 0:
            quat = d.qpos[3:7]; qw, qx, qy, qz = (float(x) for x in quat)
            grav = np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])
            omega = np.asarray(d.qvel[self.base + 3:self.base + 6], float) * EXPERT_ANG_VEL_SCALE
            R = d.xmat[self.pelvis].reshape(3, 3); yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
            wz = min(max(EXPERT_TURN * _wrap(self.heading - yaw), -EXPERT_WZ_MAX), EXPERT_WZ_MAX)
            if self.turning and abs(_wrap(self.heading - yaw)) < TURN_DONE:
                self.turning = False
            cmd = np.array([0.0 if self.turning else EXPERT_V, 0.0, wz]) * np.array(EXPERT_CMD_SCALE)
            phase = (self.e_count * EXPERT_DT) % EXPERT_PERIOD / EXPERT_PERIOD
            obs = np.concatenate([omega, grav, cmd, (q - self.e_default), dq * EXPERT_DOF_VEL_SCALE, self.e_action,
                                  [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)]]).astype(np.float32)
            with torch.no_grad():
                self.e_action = self.expert(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze().astype(float)
            self.e_target = self.e_action * EXPERT_ACTION_SCALE + self.e_default
            self.e_steps += 1
        tau = self.e_kp * (self.e_target - q) - self.e_kd * dq
        tau = np.clip(s * tau, -self.e_lim, self.e_lim)
        d.qfrc_applied[self.e_dof] += tau
        # the leg servos' targets drawn toward the expert's by s (18:50: with the torque alone the body marched in place, its own
        # servos and stance tone straining against the room's hands, the strain its gear load, its pain): moved along, its motors
        # go with the movement, as a stiff toddler's legs give to the parent walking it; at s 0 the targets are its own again
        a = self.e_aid
        d.ctrl[a] = (1.0 - s) * d.ctrl[a] + s * np.clip(self.e_target, self.e_lo, self.e_hi)

    @staticmethod
    def _room_mid(xy):
        """the middle of the room the point is in (ROOMS; the first room when in neither)"""
        for x0, x1, y0, y1 in ROOMS:
            if x0 - ROOM_SLACK <= xy[0] <= x1 + ROOM_SLACK and y0 - ROOM_SLACK <= xy[1] <= y1 + ROOM_SLACK:
                return np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        x0, x1, y0, y1 = ROOMS[0]
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])

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
            x_, y_ = float(pxy[0]), float(pxy[1])
            inside = any(x0 - ROOM_SLACK <= x_ <= x1 + ROOM_SLACK and y0 - ROOM_SLACK <= y_ <= y1 + ROOM_SLACK for x0, x1, y0, y1 in ROOMS)
            if not inside and getattr(w, "carry_to_mat", None) is not None:   # W7b: out of the rooms (day 135: 80 m into the void): back to its mat
                try:
                    w.carry_to_mat()
                except Exception:
                    pass
                self.last[:] = 0.0; self.last_p[:] = 0.0; self.left_room += 1; self.ticks = 0; self.s = min(1.0, self.s + EDGE_UP)
                self.z_from = float(d.xipos[self.b][2]); self.z_to = self.z_stand; self.lead = np.asarray(d.xipos[self.b][:2], float).copy(); self.turning = False
                print(f"the ghost's walk left the rooms at ({x_:.1f}, {y_:.1f}) at tick {w.tick}: carried to its mat (W7b)", flush=True)
                self.win = []; self.pxy = np.asarray(d.xpos[self.pelvis][:2], float).copy()
                self.rec = [round(self.s, 4), round(share, 3), round(self.heading, 2), int(self.falls), round(self.metres, 2)]
                return
            up = d.xmat[self.b].reshape(3, 3)[:, 2]
            fallen = pz < FALL_Z or float(up[2]) < FALL_UP
            if fallen:                                                      # the catch: the harness takes it again. The first catch
                self.edge = float(self.s); self.s = min(1.0, self.s + EDGE_UP); self.ticks = 0; self.falls += 1   # W7c: the edge kept, a tenth more help. (16:05) the catch restarted the lift from the fallen pose and hauled a
                lay = getattr(w, "carry_to_mat", None)                   # child on its front up by the chest, head down (the copy: pain 13
                if lay is not None:                                         # ticks in 30); the environment lays it on its back where it fell
                    try:                                                    # first (its laying, C338 amended), and the lift is from lying;
                        clear_ = getattr(getattr(w, "parent", None), "_standing_clear_m", None)   # near a wall, on its mat (a lying body
                        here_ = np.asarray(d.qpos[:2], float)                                       # is 1.3 m long: laid across a wall it
                        lay(to=here_ if (clear_ is None or float(clear_(here_)) >= LAY_CLEAR_M) else None)   # came up on the far side)
                    except Exception:
                        pass
                    self.turning = False
                    self.last[:] = 0.0; self.last_p[:] = 0.0                # (the laying clears the applied forces)
                self.z_from = float(d.xipos[self.b][2]); self.z_to = self.z_stand; self.lead = np.asarray(d.xipos[self.b][:2], float).copy()
            else:
                fwd = np.array([math.cos(self.heading), math.sin(self.heading)])
                txy = np.asarray(d.xipos[self.b][:2], float)
                self.lead = self.lead + SPEED * TICK_S * fwd
                ahead = float((self.lead - txy) @ fwd)
                self.lead = txy + fwd * min(max(ahead, 0.0), LEAD_M)          # along the heading, a step ahead at most
                expert_ = self.use_expert and self.expert is not None
                look = (EXPERT_LOOK_M if expert_ else LEAD_M + CLEAR_M)
                probe = txy + fwd * look
                par = getattr(w, "parent", None)
                clear = getattr(par, "_standing_clear_m", None)
                if clear is not None and not self.turning and float(clear(probe)) < CLEAR_M:
                    # blocked ahead: toward the room's middle when far from it (20:25: a quarter turn left it turning in place against
                    # the west wall for a block, 17 m and 9% of ticks with anything in view), else a quarter turn; the expert turns in place
                    mid = self._room_mid(txy)
                    to_mid = mid - txy
                    if float(np.linalg.norm(to_mid)) > 1.0:
                        self.heading = float(math.atan2(to_mid[1], to_mid[0]))
                    else:
                        self.heading = _wrap(self.heading + math.pi / 2)
                    self.turns += 1; self.turning = expert_; self.lead = txy.copy()
                if share > HANG:
                    self.s = min(1.0, self.s + RISE)
                elif share < BEARS:
                    self.s = max(0.0, self.s - FADE)
        if pz > FALL_Z:
            self.win.append(pxy.copy())
            if len(self.win) > METRES_WIN:
                self.metres += float(np.linalg.norm(self.win[-1] - self.win[0])); self.win = [self.win[-1]]
        else:
            self.win = []
        self.pxy = pxy.copy()
        self.rec = [round(self.s, 4), round(share, 3), round(self.heading, 2), int(self.falls), round(self.metres, 2)]

    # ------------------------------------------------------------------ the save
    def state(self):
        return dict(on=bool(self.on), s=float(self.s), heading=float(self.heading), lead=[float(x) for x in self.lead],
                    ticks=int(self.ticks), lower=int(self.lower), falls=int(self.falls), turns=int(self.turns), z_from=float(self.z_from), z_to=float(self.z_to),
                    last=[float(x) for x in self.last], metres=float(self.metres),
                    e_action=[float(x) for x in self.e_action], e_target=[float(x) for x in self.e_target], e_count=int(self.e_count), e_steps=int(self.e_steps), turning=bool(self.turning), left_room=int(self.left_room), edge=self.edge,
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
        self.e_action = np.asarray(st.get("e_action", np.zeros(12)), float).copy(); self.e_target = np.asarray(st.get("e_target", EXPERT_DEFAULT), float).copy()
        self.e_count = int(st.get("e_count", 0)); self.e_steps = int(st.get("e_steps", 0)); self.turning = bool(st.get("turning", False)); self.left_room = int(st.get("left_room", 0)); self.edge = st.get("edge", None)
        self.rec = None
