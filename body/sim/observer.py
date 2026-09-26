"""THE BORN MOMENTUM OBSERVER: CONTACT FELT THROUGH THE JOINTS, FROM THE ROBOT'S OWN SENSORS (docs/SIM_DESIGN.md 3.4, A37, A73;
built at S5a, replacing the world's exact contact torques that the first S5a build had handed the body).

A real G1 has no skin outside its Dex3 hands. What it can know of a push, a lean or a blow is what its own sensors show: each motor's
angle and velocity (its encoder), the torque its current makes, and the pelvis's inertial unit. A robot's collision detector turns
these into each joint's outside torque with a generalized-momentum observer (De Luca and Mattone 2005; Haddadin, De Luca and
Albu-Schaeffer 2017), and so does this one, in software the real G1 can run on its own sensors. It never reads a contact, a force or
an acceleration from the physics: only the sensed signals below and the robot's own model file (its masses, inertias and rotor
inertias, as the real robot's software carries them).

THE SENSED SIGNALS, every physics step (2 ms, the G1's 500 Hz low-level loop):
  q, qd   the 43 joints' angles and velocities as the motors report them (ideal until the encoders' steps are read: C46)
  tau     each motor's torque as its current gives it (the actuator's force; ideal until the estimate's noise is read: C46)
  imu     the pelvis unit's accelerometer (specific force) and gyro, with the model's declared noise (the world's stream)
  R       the pelvis's attitude, as the unit's own fusion gives it (ideal until its error is read: C46)

THE JOINTS' OUTSIDE TORQUE (the observer proper), every OBS_STEPS = 5 steps (10 ms, the pain filter's window). With the floating base,
the joint rows of the body's dynamics are M_jj q'' + M_jb a_b + c_j = tau + tau_ext; their momentum p = M_jj qd changes as
  dp/dt = tau + tau_ext - c_j - M_jb a_b + (dM_jj/dt) qd,
so the residual r = K (p - p_0 - integral(tau - c_j - M_jb a_b + (dM_jj/dt) qd + r) dt) follows tau_ext through a first-order lag
of time constant 1 / K (r' = K (tau_ext - r)). a_b, the base's acceleration, comes from the unit: its angular part from the gyro's
change over the window, its linear part from the specific force turned into the world by the attitude, gravity added back and the
unit's offset from the pelvis's origin removed. c_j (gravity, Coriolis and the base's turning) is MuJoCo's recursive Newton-Euler on
the sensed state; the base's linear velocity drops out (a uniform translation changes no force: Galilean invariance), so the robot
never needs it. What the model does not carry reads as outside torque, as it does on a real robot: the joints' friction loss
(0.3 N m in the file) while a joint moves, and a joint's stop when it reaches its range's end (phantom touch, C43).
THE BASE'S OUTSIDE WRENCH: the base rows' Newton-Euler on the same sensed state, M_b. a + c_b, with the joints' accelerations from
their velocities' change over the window: the whole body's net outside force (in the pelvis's frame) and torque, as a legged robot
estimates its contact wrench. It is noisier than the joints' (a differentiated signal): the real condition.

THE GEAR'S LOAD (A73, pain): each step, tau - armature x (qd_k - qd_(k-1)) / dt, the rotor's inertia times the joint's acceleration as
the encoder's velocity shows it; its 10 ms mean past the joint's declared limit is the joint's pain.

K = K_OBS = 50 per s (time constant 20 ms, two of its 10 ms samples: the observer's bandwidth well below its sample rate, as the
method asks; ours, from the method, never from a pain rate: C43). At birth it starts from the static estimate of the born state at
rest (the gravity its motors do not hold), as a robot's observer can when it is switched on still.
state() / load_state() carry it, so a replay is exact.
"""
import numpy as np
import mujoco

OBS_STEPS = 5            # the observer's sample: every 5 physics steps, 10 ms (A37)
K_OBS = 50.0             # its gain, 1/s (the module's doc)


class Observer:
    def __init__(self, m, d, dof, base_dof, pelvis_id, imu_site, armature):
        self.m = m
        self.dof = np.asarray(dof, np.int64)                      # the 43 joints' dofs, in the body's order
        self.b = int(base_dof)                                    # the free joint's first dof
        self.rows = np.concatenate([np.arange(self.b, self.b + 6), self.dof])
        self.pelvis = int(pelvis_id)
        sid = m.site(imu_site).id
        self.r_imu = m.site_pos[sid].copy()                        # the unit in the pelvis's frame
        Rs = np.zeros(9); mujoco.mju_quat2Mat(Rs, m.site_quat[sid]); self.R_site = Rs.reshape(3, 3)
        self.armature = np.asarray(armature, float).copy()
        self.g = m.opt.gravity.copy()
        self.dt = float(m.opt.timestep)
        self.T = self.dt * OBS_STEPS
        self.sc = mujoco.MjData(m)                                 # the observer's own scratch state (never the world's)
        self.Mfull = np.zeros((m.nv, m.nv))
        self.J = len(self.dof)
        self.started = False
        self.p0 = np.zeros(self.J); self.S = np.zeros(self.J); self.r = np.zeros(self.J)
        self.Mjj = None
        self.w = np.zeros(6)                                       # the base's last estimate: force (pelvis frame), torque
        self.qd_win = None                                         # the joints' velocities at the last sample
        self.om_win = None                                         # the pelvis's angular velocity (its frame) at the last sample
        self.qd_last = None                                        # the last step's joint velocities (the gear's load)
        self.tau_sum = np.zeros(self.J); self.n_sum = 0

    # ------------------------------------------------------------------ the robot's model on the sensed state
    def _dynamics(self, q, qd, quat, om, a_lin, alpha, qdd):
        """M (49 x 49 over the G1's rows) and the bias c (gravity, Coriolis) at the sensed state, and M x the sensed accelerations"""
        m, sc = self.m, self.sc
        sc.qpos[self.b:self.b + 3] = 0.0
        sc.qpos[self.b + 3:self.b + 7] = quat
        qa = self.m.jnt_qposadr[self.m.dof_jntid[self.dof]]
        sc.qpos[qa] = q
        sc.qvel[:] = 0.0
        sc.qvel[self.b + 3:self.b + 6] = om
        sc.qvel[self.dof] = qd
        mujoco.mj_kinematics(m, sc)
        mujoco.mj_comPos(m, sc)
        mujoco.mj_crb(m, sc)
        mujoco.mj_comVel(m, sc)
        mujoco.mj_fullM(m, self.Mfull, sc.qM)
        M = self.Mfull[np.ix_(self.rows, self.rows)]
        c = np.zeros(m.nv)
        mujoco.mj_rne(m, sc, 0, c)
        acc = np.concatenate([a_lin, alpha, qdd])
        return M, c[self.rows], acc

    def _base_motion(self, R, f_imu, om_imu, om_prev):
        """the pelvis's angular velocity (its frame), angular acceleration (its frame) and its origin's linear acceleration (world)"""
        om = self.R_site @ om_imu
        alpha = np.zeros(3) if om_prev is None else (om - om_prev) / self.T
        a_site = R @ (self.R_site @ f_imu) + self.g                    # the unit's acceleration in the world
        w_w, al_w, r_w = R @ om, R @ alpha, R @ self.r_imu
        a_o = a_site - np.cross(al_w, r_w) - np.cross(w_w, np.cross(w_w, r_w))
        return om, alpha, a_o

    # ------------------------------------------------------------------ every step and every sample
    def gear_load(self, tau, qd):
        """THE GEAR'S LOAD this step (A73) from the sensed torque and the encoder's velocity change"""
        qdd = np.zeros(self.J) if self.qd_last is None else (qd - self.qd_last) / self.dt
        self.qd_last = np.asarray(qd, float).copy()
        self.tau_sum += tau; self.n_sum += 1
        return tau - self.armature * qdd

    def sample(self, q, qd, quat, f_imu, om_imu, tau=None):
        """one 10 ms sample: the joints' outside torque r (43) and the base's outside wrench (force in the pelvis's frame, torque),
        from the sensed signals alone. The first sample (the born state, at rest: `tau` the motors' sensed torques then) starts the
        observer at the static estimate, r = c_j - tau with the joints still (the gravity its motors do not hold), and the base's
        wrench with the joints' accelerations 0"""
        R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(quat, float)); R = R.reshape(3, 3)
        om, alpha, a_o = self._base_motion(R, np.asarray(f_imu, float), np.asarray(om_imu, float), self.om_win)
        qdd = np.zeros(self.J) if self.qd_win is None else (qd - self.qd_win) / self.T   # (at birth, at rest: 0)
        M, c, acc = self._dynamics(q, qd, quat, om, a_o, alpha, qdd)
        Mjj, Mjb = M[6:, 6:], M[6:, :6]
        p = Mjj @ qd
        tau = self.tau_sum / self.n_sum if self.n_sum else (np.zeros(self.J) if tau is None else np.asarray(tau, float))
        self.tau_sum = np.zeros(self.J); self.n_sum = 0
        if not self.started:                                       # at birth: the static estimate of the born state at rest
            self.r = c[6:] + Mjb @ acc[:6] - tau
            self.p0, self.S, self.started = p.copy(), -self.r / K_OBS, True
        else:
            dM = (Mjj - self.Mjj) / self.T
            self.S = self.S + self.T * (tau - c[6:] - Mjb @ acc[:6] + dM @ qd + self.r)
            self.r = K_OBS * (p - self.p0 - self.S)
        self.Mjj = Mjj.copy()
        wb = M[:6] @ acc + c[:6]                                   # the base rows: the body's net outside wrench
        self.w = np.concatenate([R.T @ wb[:3], wb[3:]])
        self.qd_win, self.om_win = np.asarray(qd, float).copy(), om.copy()
        return self.r.copy(), self.w.copy()

    # ------------------------------------------------------------------ the save
    def state(self):
        arr = lambda x: None if x is None else np.asarray(x, float).copy()
        return dict(started=self.started, p0=arr(self.p0), S=arr(self.S), r=arr(self.r), Mjj=arr(self.Mjj), w=arr(self.w),
                    qd_win=arr(self.qd_win), om_win=arr(self.om_win), qd_last=arr(self.qd_last), tau_sum=arr(self.tau_sum),
                    n_sum=int(self.n_sum))

    def load_state(self, s):
        arr = lambda x: None if x is None else np.array(x, dtype=np.float64)
        self.started = bool(s["started"])
        self.p0, self.S, self.r, self.Mjj, self.w = arr(s["p0"]), arr(s["S"]), arr(s["r"]), arr(s["Mjj"]), arr(s["w"])
        self.qd_win, self.om_win, self.qd_last = arr(s["qd_win"]), arr(s["om_win"]), arr(s["qd_last"])
        self.tau_sum, self.n_sum = arr(s["tau_sum"]), int(s["n_sum"])
