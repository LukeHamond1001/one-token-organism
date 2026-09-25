"""THE CHILD'S VOCAL TRACT (the owner's decision 5, 2026-09-24; docs/SIM_DESIGN.md 4.9 and the decision log's B6): the voice
effector's physics. A source-filter articulatory synthesizer in numpy, brought in from the G1 amendment's voice study
($S/g1/voice/tract.py, 2026-09-24) with its calibration unchanged; what is new here is the seeded stream, save and restore, the
body sense, and the level in pascals.

One call per 150 ms tick: Tract.tick(act) -> 2,400 samples at 16 kHz, the sound the tract radiates in that tick, in the parent
voice's engine units at 1 m in front of the mouth (x PA_PER_UNIT for pascals; the ears place it at the G1's head, body/sim/ears.py,
from its speaker's place on the head's front).
The samples run from 10 ms before the tick's start to 10 ms before its end (half a control frame: the sound lags the articulators).

Structure (a hybrid time/frequency-domain synthesizer in the manner of Sondhi & Schroeter 1987):
  articulators (10, each 0..1)  --second-order muscle dynamics, 10 ms control frames-->  positions
  positions --geometry--> an area function (24 tube sections, glottis to lips) + the velopharyngeal port + lip length
  the area function --lossy chain matrices (Flanagan's per-unit-length losses and yielding walls)--> the tract's
      transfer functions at each control frame: glottis->mouth, glottis->nose, and constriction->mouth
  quasi-static aerodynamics (Bernoulli orifices in series: glottis, the narrowest supraglottal place) --> the DC airflow,
      the pressure across the glottis (which decides whether the folds vibrate), and each orifice's Reynolds number
  sources: glottal pulses (Rosenberg) with pitch, jitter and shimmer; turbulence noise at the glottis and at the
      narrowest supraglottal place, each with the same law (amplitude ~ Re^2 - Re_c^2 above Re_c; Flanagan)
  filtering: each 20 ms Hann frame of each source times its transfer function (zero-padded FFT), overlap-added;
      the radiation (d/dt) applied as j*omega.

No sound category is written anywhere: a vowel, a nasal murmur, a hiss, a burst, a stop's silence all come out of the same
geometry and the same aerodynamic law. A closure stops voicing by itself: with the lips (or the tongue) shut and the velum raised
the pressure across the glottis falls to nothing and the folds stop; with the velum lowered the air leaves by the nose and the
voice goes on as a nasal murmur (test_sim_voice.py measures both).

THE EFFECTOR (the voice's joints): 10 articulators (NAMES), each taking one of STEPS = {-0.6, -0.2, 0, +0.2, +0.6} of its range a
tick, re-anchored on where it is (the servo law); at rest the targets relax to the silent rest posture (NEUTRAL: nose breathing,
lips nearly closed) with a time constant of 1 tick. Its body sense (proprio()): the 10 positions, their 10 velocities and the
breath left: 21 numbers for the body channel. Its random numbers (jitter, shimmer, both noises) come from its own stream of the
body's seed, a PCG64 from SeedSequence(seed, spawn_key=(STREAM,)), the world's convention (body/sim/world.py: its own stream is
spawn key 1; the tract's is 2, ours). The tract owns its generator (none is handed in, so restoring it can never rewind a stream
another part draws from); state() and load_state() carry it, so a replay is exact.

THE CALIBRATION (the study's; all ours, anatomy; disclosed):
  - the child's tract: 12 cm glottis to lips (a young child's; an adult woman's is about 14.5-15), 24 sections, a nasal branch at
    its middle; resting pitch about 265 Hz (180-546 Hz over the pitch articulator's range)
  - the tongue's four corners (place, least area) fitted once to children's vowel means (Peterson & Barney 1952), each
    constriction held at its anatomical place (Wood 1979): /i/ 344/2719 Hz, /a/ 938/1625 Hz, rounded /u/ about 375/1250 Hz.
    The honest gap: the vowel space is about an adult woman's size (front vowels short in F2, 2700 against 3200; low vowels
    short in F1, about 900 against 1030)
  - levels: the steady /a/ at ANSI S3.5-1997's "normal" vocal effort at 1 m, 62 dB SPL (GAIN), the same reference level the
    parent's speech takes (synth.SPEECH_PA), not a fit to her voice (measured 62.2 dB SPL at 1 m on the study's stream, 62.5 on
    the tract's own stream of seed 1, the jitter's and shimmer's draws); the
    turbulence constants set so a steady alveolar hiss is -14.1 dB and a steady /h/ -13.4 dB re /a/ (measured; Fletcher's
    relative phonetic powers put /s/ about -16 dB and /sh/ -9 dB re /a/)
  - the breath reservoir: 400 cm^3 usable (about 2.5 s of speech), a full breath in 0.8 s at rest
  - the muscles: critically damped, natural periods TAU_MS (a step 95% done in 57-120 ms)
Its cost (this Mac, shared with other jobs; tools/sim_voice_check.py, 2026-09-24): a sounding tick 1.7-3.2 ms (a held vowel, a
glide, a hiss; best of repeats at load averages 3.4-5.5), up to 7.4 ms under heavier load; at rest 0.1-0.2 ms.
"""
import numpy as np
from scipy.signal import lfilter

try:
    from .voice.synth import PA_PER_UNIT  # noqa: F401  (the engine units -> pascals at 1 m, shared with the parent's voice)
except ImportError:                       # imported as a top-level module from body/sim/ (the study's scripts)
    from voice.synth import PA_PER_UNIT  # noqa: F401

STREAM = 2                   # the tract's own random stream of the body's seed: SeedSequence(seed, spawn_key=(2,)) (the world: 1)

SR = 16000
TICK = 2400                  # 150 ms
HOP = 160                    # 10 ms control frames (15 per tick)
WIN = 2 * HOP                # 20 ms periodic Hann, 50% overlap (sums to 1)
NFFT = 1024
NF = TICK // HOP             # 15 frames per tick
NH = 512                     # the tract's transfer is computed on NH/2+1 bins (31 Hz), then zero-padded in time to NFFT
FREQ = np.fft.rfftfreq(NH, 1 / SR)
W = 2 * np.pi * FREQ
W[0] = 2 * np.pi * 1.0       # avoid 0 in the loss terms (DC bin is zeroed by the radiation anyway)

# ---- air and tissue (CGS) ----
RHO, C, MU = 1.14e-3, 35000.0, 1.86e-4
NU = MU / RHO                # kinematic viscosity, cm^2/s
KAPPA = 0.228                # thermal diffusivity of air, cm^2/s
ETA = 1.4
WALL_M, WALL_R, WALL_K = 1.5, 1600.0, 3.0e5     # yielding walls per unit area (Ishizaka, French & Flanagan 1975)
CMH2O = 980.0

# ---- the child's tract (anatomy, ours) ----
L0 = 12.0                    # glottis-to-lips length, cm, lips spread (an adult woman ~14.5-15; a young child)
N = 24                       # tube sections
XS = (np.arange(N) + 0.5) / N
KB = 12                      # the velopharyngeal port sits at the boundary before section 12 (half way)
A_NEUTRAL = np.where(XS < 0.08, 0.6, np.where(XS < 0.45, 1.6, 2.0))   # larynx tube, pharynx, mouth (cm^2)
NASAL = np.array([1.0, 1.4, 1.8, 1.8, 1.5, 1.1, 0.7, 0.6]) * 0.5       # nasal cavity (child), 1 cm sections, to the nostrils
NASAL_L = 1.0
NASAL_LOSS = 4.0             # the nasal mucosa's extra viscous loss (x)
A_PORT_MAX = 0.5             # velopharyngeal port fully open, cm^2
A_LIP_MAX = 3.0
LIP_PROTRUDE = 0.8           # cm added to the tract when fully rounded
A_FLOOR = 1e-4               # a closure (cm^2)
P_MAX = 12.0 * CMH2O         # lungs at full drive
P_TH = 3.0 * CMH2O           # phonation threshold pressure across the glottis
A_OSC = 0.05                 # the folds' oscillating opening at full voicing, cm^2
A_LEAK_MAX = 0.2             # the glottis wide open (breathing), cm^2
F0_LO, F0_OCT = 180.0, 1.6   # pitch articulator 0..1 -> 180 Hz x 2^(1.6 p): 180..546 Hz
RE_C = 1800.0                # critical Reynolds number (Flanagan)
K_NOISE = 7.5e-6             # turbulence source strength (calibrated: t_calib2.py)
K_ASP = 8.5e-9               # glottal turbulence as a flow, per unit of the same law (calibrated: t_calib2.py)
V_CAP = 400.0                # usable breath, cm^3 (about 2.5 s of speech)
REFILL_S = 0.8               # a full breath in at rest, seconds
# the sources' peak amplitudes in speech (a steady /a/, /h/, hiss) at 4 sd of the noise: the silence gate's reference
SRC_REF = (160.0, 8.5e-9 * (8000.0 ** 2 - RE_C ** 2) * 4, 7.5e-6 * (6000.0 ** 2 - RE_C ** 2) * 4)
TILT_LO, TILT_HI = 700.0, 2500.0   # the return phase's corner, breathy .. pressed (Hz); Klatt's spectral tilt
GAIN = 3.67e-7                   # output scale (the tract's /a/ at ANSI's normal effort, 62 dB SPL at 1 m: t_calib2.py)

# ---- the articulators (the voice effector's joints) ----
NAMES = ['lungs', 'glottis', 'pitch', 'jaw', 'tongue_front', 'tongue_height', 'tongue_tip', 'lips', 'rounding', 'velum']
NEUTRAL = np.array([0.0, 0.1, 0.35, 0.15, 0.5, 0.45, 0.0, 0.1, 0.3, 1.0])   # the passive rest posture: silent, nose breathing
# natural periods of the critically damped muscle model, 1/omega in ms (95% of a step in 4.74/omega)
TAU_MS = np.array([25, 12, 25, 20, 20, 20, 12, 12, 20, 20], float)
STEPS = np.array([-0.6, -0.2, 0.0, 0.2, 0.6])      # {-big, -small, 0, +small, +big} per articulator per tick
REST_RELAX = 1.0             # at rest the targets relax to NEUTRAL with this time constant, in ticks
REST_SNAP = 0.02             # ... and settle on it once this near
N_ART = len(NAMES)

# place (fraction of the length from the glottis) and minimum area (cm^2) of the tongue body at the four corners of
# (front, height): back-low /a/-like pharyngeal, front-low, back-high velar, front-high palatal (Wood 1979's places)
P_CORNER = np.array([[0.24, 0.57], [0.28, 0.72]])      # [front][height]
A_CORNER = np.array([[0.25, 0.45], [1.20, 0.06]])
H_VOWEL = 0.85               # height above this closes the tongue body toward a velar/palatal closure at 1
TONGUE_SIG = 0.14            # the constriction's half-width (fraction of length, Gaussian)
TONGUE_BULK = 0.8            # the widening elsewhere per unit of constriction depth (incompressible tongue)
RIDGE_X, RIDGE_W = (21 + 0.5) / N, 0.035    # the alveolar ridge, 1.25 cm behind the lips


def _bilin(M, b, h):
    return (M[0, 0] * (1 - b) * (1 - h) + M[1, 0] * b * (1 - h) + M[0, 1] * (1 - b) * h + M[1, 1] * b * h)


def geometry(x):
    """articulator positions [..., 10] -> areas [..., N], port area [...], tract length [...] (cm)."""
    x = np.asarray(x, float)
    jaw, fr, ht, tip, lip, rnd, vel = (x[..., 3], x[..., 4], x[..., 5], x[..., 6], x[..., 7], x[..., 8], x[..., 9])
    h_eff = np.clip(ht - 0.4 * (jaw - 0.25), 0, 1)          # the tongue rides on the jaw
    hv = np.minimum(h_eff, H_VOWEL) / H_VOWEL
    place = _bilin(P_CORNER, fr, hv)
    amin = _bilin(A_CORNER, fr, hv) * np.clip((1 - h_eff) / (1 - H_VOWEL), 0, 1) ** 2   # above H_VOWEL: toward closure
    xs = XS
    a0 = A_NEUTRAL * np.where(xs > 0.6, 0.5 + 1.2 * jaw[..., None], 1.0)            # the jaw opens the front mouth
    region = (xs > 0.1) & (xs < 0.92)                                                 # where the tongue can reach
    g = np.exp(-((xs - place[..., None]) / TONGUE_SIG) ** 2) * region
    g = g / g.max(-1, keepdims=True)                                                  # a full closure is possible
    gw = np.exp(-((xs - place[..., None]) / (2.2 * TONGUE_SIG)) ** 2)
    a0p = np.take_along_axis(a0, np.clip(np.round(place * N - 0.5).astype(int), 0, N - 1)[..., None], -1)
    depth = np.maximum(a0p[..., 0] - amin, 0)
    # the tongue is incompressible: what it takes from the constriction widens the rest of its region
    a = a0 - depth[..., None] * g + TONGUE_BULK * depth[..., None] * (1 - gw) * region
    a = a * (1 - (1 - (1 - tip[..., None]) ** 2) * np.exp(-((xs - RIDGE_X) / RIDGE_W) ** 2))   # the tip at the ridge:
    # the opening left there goes as (1 - tip)^2, so the upper fifth of its range is the fricative and closure zone
    alip = A_LIP_MAX * lip ** 1.5 * (0.4 + 0.8 * jaw) * (1 - 0.8 * rnd)              # an elliptic opening narrows as it closes
    a = np.concatenate([a[..., :N - 2], np.repeat(alip[..., None], 2, -1)], -1)
    a = np.maximum(a, A_FLOOR)
    return a, A_PORT_MAX * vel, L0 + LIP_PROTRUDE * rnd


SQW = np.sqrt(W)
JW = 1j * W
GY = (ETA - 1) / (RHO * C ** 2) * np.sqrt(KAPPA / 2) * SQW + 1 / (WALL_R + 1j * W * WALL_M + WALL_K / (1j * W))


def _zy(a, l, w=None, loss=1.0):
    """per-section series impedance Z and shunt admittance Y (lumped, length l) with Flanagan's losses: viscous and
    thermal boundary layers and yielding walls. a, l broadcast against the frequency axis (last)."""
    s = 2 * np.sqrt(np.pi * a)                                    # perimeter
    Z = l * ((loss * s / a ** 2 * np.sqrt(RHO * MU / 2)) * SQW + (RHO / a) * JW)
    Y = l * (s * GY + (a / (RHO * C ** 2)) * JW)
    return Z, Y


SQW32, JW32, GY32 = SQW.astype(np.float32), JW.astype(np.complex64), GY.astype(np.complex64)
_KV = np.float32(np.sqrt(RHO * MU / 2)); _RHO = np.float32(RHO); _RC2 = np.float32(RHO * C ** 2)


def _zy32(a, l):
    s = 2 * np.sqrt(np.float32(np.pi) * a)
    Z = l * ((s / a ** 2 * _KV) * SQW32 + (_RHO / a) * JW32)
    Y = l * (s * GY32 + (a / _RC2) * JW32)
    return Z, Y


def _zrad(a, w):
    rr = 128 * RHO * C / (9 * np.pi ** 2 * a)
    lr = 8 * RHO / (3 * np.pi * np.sqrt(np.pi * a))
    return 1j * w * lr * rr / (rr + 1j * w * lr)


# the nasal cavity beyond the port is fixed: its (P, U) at the port's downstream end for a unit nostril flow
def _nasal_rest():
    P, U = _zrad(NASAL[-1], W), np.ones_like(W, complex)
    for k in range(len(NASAL) - 1, -1, -1):
        Z, Y = _zy(NASAL[k], NASAL_L, W, NASAL_LOSS)
        a = 1 + Z * Y / 2
        P, U = a * P + Z * (1 + Z * Y / 4) * U, Y * P + a * U
    return P, U
NAS_P, NAS_U = _nasal_rest()


def _port(port):
    ap = np.maximum(port, A_FLOOR)[:, None]
    Zp, Yp = _zy(ap, 1.0, None, NASAL_LOSS)
    app = 1 + Zp * Yp / 2
    Pn = app * NAS_P[None] + Zp * (1 + Zp * Yp / 4) * NAS_U[None]
    Un = Yp * NAS_P[None] + app * NAS_U[None]
    return Pn, Un


def transfer(areas, port, length, u_dc, a_glot, noise_at=None):
    """areas [F, N], port [F], length [F] -> H_g [F, NB] (glottal flow -> radiated pressure, mouth + nose) and, if
    noise_at [F] (section index of the narrowest place) is given, H_n [F, NB] (a series pressure source just downstream
    of that section -> radiated pressure). Radiation d/dt included."""
    F = areas.shape[0]
    l = (length / N)[:, None, None]
    Z, Y = _zy32(areas[:, :, None].astype(np.float32), l.astype(np.float32))     # [F, N, NB], single precision
    ZY = Z * Y
    A = 1 + 0.5 * ZY
    B = Z * (1 + 0.25 * ZY)
    P = (_zrad(areas[:, -1:], W[None, :]) * np.ones((F, 1))).astype(np.complex64)
    U = np.ones_like(P)
    keepP, keepU = [None] * (N + 1), [None] * (N + 1)
    keepP[N], keepU[N] = P, U
    Pn, Un = _port(port)
    Pn, Un = Pn.astype(np.complex64), Un.astype(np.complex64)
    for i in range(N - 1, -1, -1):
        a, b, y = A[:, i], B[:, i], Y[:, i]
        P, U = a * P + b * U, y * P + a * U
        if i == KB:
            U = U + (Un / Pn) * P                         # the nasal branch as a shunt at the port
            Pb = P
        keepP[i], keepU[i] = P, U
    Hg = JW * (1.0 + Pb / Pn) / U                         # (mouth flow 1 + nose flow Pb/Pn) per glottal flow, d/dt
    if noise_at is None:
        return Hg, None
    Zu = (RHO * u_dc / np.maximum(a_glot, 1e-3) ** 2)[:, None] + JW * (RHO * 0.3 / np.maximum(a_glot, 1e-3))[:, None]
    Hn = np.zeros((F, W.size), complex)
    tgt = noise_at + 1
    last = int(tgt.max())
    for i in range(0, last):
        if i == KB:
            Zu = 1 / (1 / Zu + Un / Pn)
        a, b, y = A[:, i], B[:, i], Y[:, i]
        Zu = (a * Zu + b) / (y * Zu + a)
        sel = tgt == i + 1
        if sel.any():
            Ud = keepU[i + 1][sel]
            Hn[sel] = JW / ((Zu[sel] + keepP[i + 1][sel] / Ud) * Ud)
    return Hg, Hn


def aero(x, areas, port, vol):
    """quasi-static airflow for positions x [F, 10]: returns u_dc, dp_glottis, voicing amount (0..1), glottal mean
    area, leak area, supraglottal outlet area, and the narrowest section index."""
    drive, g = x[:, 0], x[:, 1]
    psub = P_MAX * drive * np.clip(vol / 0.15, 0, 1)
    a_leak = A_LEAK_MAX * (1 - g) ** 2
    add = np.clip((g - 0.45) / 0.15, 0, 1)
    up = areas[:, 3:KB].min(1)
    idx_dn = np.argmin(areas[:, KB:], 1) + KB
    dn = areas[np.arange(len(areas)), idx_dn]
    a_out = np.minimum(up, dn + port)
    narrow = np.where(dn < up, idx_dn, np.argmin(areas[:, 3:KB], 1) + 3)
    v = np.zeros_like(drive)
    for _ in range(3):                                   # the voicing and the mean glottal area settle together
        ag = a_leak + 0.5 * A_OSC * v
        ag = np.maximum(ag, 1e-3)
        k = 1 / ag ** 2 + 1 / a_out ** 2
        dpg = psub * (1 / ag ** 2) / k
        v = np.clip((dpg - P_TH) / P_TH, 0, 1) * add
    u = np.sqrt(2 * psub / RHO / k)
    return dict(u=u, dpg=dpg, v=v, ag=ag, a_leak=a_leak, a_out=a_out, dn=dn, up=up, port=port, narrow=narrow, psub=psub)


def reynolds(u, a):
    return u * np.sqrt(4 / np.pi) / (NU * np.sqrt(np.maximum(a, 1e-6)))


def _rest_last(vol):
    X = np.tile(NEUTRAL, (NF, 1))
    a, p, L = geometry(X)
    return dict(X=X, areas=a, port=p, aero=aero(X, a, p, np.full(NF, vol)), f0=np.full(TICK, F0_LO), vol=vol)


class Tract:
    """one child's tract; its state carries across ticks (positions, velocities, targets, breath, phase, overlap tails)."""

    def __init__(self, seed=1):
        """seed: the body's seed (the tract draws from its own stream of it, spawn key STREAM)."""
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(STREAM,))))
        self.x = NEUTRAL.copy()
        self.v = np.zeros(N_ART)
        self.target = NEUTRAL.copy()
        self.vol = 1.0
        self.phase = 0.0
        self.tail = np.zeros(NFFT)
        self.src_prev = np.zeros((3, HOP))               # the last half-window of each source
        self.hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)
        om = 1000.0 / TAU_MS
        dt = HOP / SR
        # exact discrete step of the critically damped system e'' = -2 om e' - om^2 e (e = x - target)
        e = np.exp(-om * dt)
        self.A11, self.A12 = e * (1 + om * dt), e * dt
        self.A21, self.A22 = -e * om ** 2 * dt, e * (1 - om * dt)
        self.last = {}
        self.tilt_y = 0.0
        self.quiet = True

    STATE = ("x", "v", "target", "vol", "phase", "tail", "src_prev", "tilt_y", "quiet")

    def state(self):
        """everything the next ticks depend on (the positions, velocities, targets, breath, the glottal phase, the overlap
        tails, the tilt filter's memory and the random stream): load_state() continues exactly."""
        out = {k: (np.array(getattr(self, k)) if isinstance(getattr(self, k), np.ndarray) else getattr(self, k))
               for k in self.STATE}
        out["rng"] = self.rng.bit_generator.state
        return out

    def load_state(self, s):
        for k in self.STATE:
            v = s[k]
            setattr(self, k, np.array(v) if isinstance(v, np.ndarray) else v)
        self.rng.bit_generator.state = s["rng"]
        self.last = {}

    def proprio(self):
        """the voice's body sense: 10 positions (0..1), 10 velocities (range per second) and the breath left (0..1)."""
        return np.concatenate([self.x, self.v, [self.vol]])

    def set_act(self, act):
        """act: None (the voice rests) or int[10] in 0..4 (per-articulator steps). Returns the targets."""
        if act is None:
            self.target = NEUTRAL + (self.target - NEUTRAL) * np.exp(-1.0 / REST_RELAX)
            near = np.abs(self.target - NEUTRAL) < REST_SNAP
            self.target[near] = NEUTRAL[near]                             # the rest posture reached
        else:
            self.target = np.clip(self.x + STEPS[np.asarray(act)], 0, 1)    # re-anchored on the measured position
        return self.target

    def tick(self, act, cord=None):
        """one tick: the act (None at rest, else int[10] in 0..4) and the cord's steps below the gate (S5a: body/core/world.py Acts.cord,
        the born cry's per-articulator steps as a fraction of the range, added to the targets the act re-anchors, clipped to the
        range) -> the tick's radiated samples"""
        self.set_act(act)
        if cord is not None:
            self.target = np.clip(self.target + np.asarray(cord, float), 0, 1)
        if act is None and self.quiet and not np.any(self.target != NEUTRAL) and np.allclose(self.x, NEUTRAL, atol=1e-3) \
                and not self.tail.any():
            self.x, self.v = NEUTRAL.copy(), np.zeros(N_ART)                 # at rest and silent: nothing to compute
            self.vol = min(1.0, self.vol + TICK / SR / REFILL_S)
            self.last = _rest_last(self.vol)
            return np.zeros(TICK)
        # positions at the 15 control frames' centres
        X = np.empty((NF, N_ART))
        for k in range(NF):
            e = self.x - self.target
            e, self.v = self.A11 * e + self.A12 * self.v, self.A21 * e + self.A22 * self.v
            self.x = np.clip(self.target + e, 0, 1)
            X[k] = self.x
        areas, port, length = geometry(X)
        # breath: drained by the flow, refilled at rest
        ad = aero(X, areas, port, np.full(NF, self.vol))
        dt = HOP / SR
        for k in range(NF):
            if X[k, 0] < 0.05:
                self.vol = min(1.0, self.vol + dt / REFILL_S)
            else:
                self.vol = max(0.0, self.vol - ad['u'][k] * dt / V_CAP)
        # sources, sample-rate, over the tick
        n = np.arange(TICK)
        fk = np.clip((n + 0.5) / HOP - 0.5, 0, NF - 1)
        interp = lambda a: np.interp(fk, np.arange(NF), a)
        f0 = F0_LO * 2 ** (F0_OCT * interp(X[:, 2])) * (1 + 0.01 * (interp(ad['psub']) / CMH2O - 6))
        f0 = f0 * (1 + 0.005 * self.rng.standard_normal())                               # jitter (per tick)
        ph = self.phase + np.cumsum(f0) / SR
        self.phase = ph[-1] % 1.0
        phi = ph % 1.0
        oq = 0.75 - 0.35 * np.clip((interp(X[:, 1]) - 0.6) / 0.4, 0, 1)
        tp, tn = 0.66 * oq, 0.34 * oq
        pulse = np.where(phi < tp, 0.5 * (1 - np.cos(np.pi * phi / tp)),
                         np.where(phi < tp + tn, np.cos(0.5 * np.pi * (phi - tp) / tn), 0.0))
        vamp = interp(A_OSC * np.sqrt(2 * np.maximum(ad['dpg'], 0) / RHO) * ad['v'])
        cyc = np.floor(ph)
        shimmer = 1 + 0.03 * self.rng.standard_normal(int(cyc[-1] - cyc[0]) + 2)[(cyc - cyc[0]).astype(int)]
        ug = vamp * shimmer * pulse
        # the folds' closing is not instantaneous (the LF model's return phase): a spectral tilt, steeper when breathy
        fa = TILT_LO + (TILT_HI - TILT_LO) * np.clip((X[:, 1].mean() - 0.5) / 0.5, 0, 1)
        a = np.exp(-2 * np.pi * fa / SR)
        ug, _ = lfilter([1 - a], [1, -a], ug, zi=[a * self.tilt_y])
        self.tilt_y = ug[-1]
        # turbulence: at the glottal leak and at the narrowest supraglottal place, one law
        u_leak = ad['u'] * ad['a_leak'] / ad['ag']
        re_g = reynolds(u_leak, ad['a_leak'])
        oral = np.where(ad['dn'] < ad['up'], ad['dn'] / (ad['dn'] + ad['port']), 1.0)   # the nose shares the flow
        re_c = reynolds(ad['u'] * oral, np.minimum(ad['dn'], ad['a_out'] + 1e-9))
        pn_g = np.maximum(re_g ** 2 - RE_C ** 2, 0)                                   # one law at both places
        pn_c = K_NOISE * np.maximum(re_c ** 2 - RE_C ** 2, 0)
        mod = np.where(interp(ad['v']) > 0.05, 0.5 + 0.5 * pulse, 1.0)                # aspiration rides the cycle
        asp = K_ASP * interp(pn_g) * mod * self.rng.standard_normal(TICK)               # as a flow at the glottis
        fric = interp(pn_c) * self.rng.standard_normal(TICK)                          # a pressure at the constriction
        src = np.stack([ug, asp, fric])
        self.last = dict(X=X, areas=areas, port=port, aero=ad, f0=f0, vol=self.vol)
        # is there anything audible to filter? (every source below -60 dB of its speech level: skip the tract)
        loud = max(np.abs(ug).max() / SRC_REF[0], np.abs(asp).max() / SRC_REF[1], np.abs(fric).max() / SRC_REF[2])
        if loud < 1e-3 and self.quiet:
            out = np.zeros(TICK)
            out[:NFFT] = self.tail
            self.tail = np.zeros(NFFT)
            self.src_prev = np.zeros((3, HOP))
            return out * GAIN
        self.quiet = loud < 1e-3
        want_noise = pn_c.max() > 0
        Hg, Hn = transfer(areas, port, length, ad['u'], ad['ag'], ad['narrow'] if want_noise else None)
        Hg = np.fft.rfft(np.fft.irfft(Hg, NH, axis=1), NFFT, axis=1)                    # to the synthesis FFT's grid
        if Hn is not None:
            Hn = np.fft.rfft(np.fft.irfft(Hn, NH, axis=1), NFFT, axis=1)
        # frames centred at 0, 160, ..., 2240 of this tick: window k spans [160k-160, 160k+160)
        full = np.concatenate([self.src_prev, src], 1)                                   # starts at -160
        idx = np.arange(WIN)[None, :] + HOP * np.arange(NF)[:, None]
        seg = full[:, idx] * self.hann                                                   # [3, NF, WIN]
        S = np.fft.rfft(seg, NFFT, axis=2)
        Y = (S[0] + S[1]) * Hg
        if Hn is not None:
            Y = Y + S[2] * Hn
        y = np.fft.irfft(Y, NFFT, axis=1)                                                # [NF, NFFT]
        buf = np.zeros(NFFT + TICK)
        buf[:NFFT] += self.tail[:NFFT]
        for k in range(NF):
            buf[HOP * k:HOP * k + NFFT] += y[k]
        out = buf[:TICK].copy()                                                          # samples -160..2240
        self.tail = np.concatenate([buf[TICK:], np.zeros(TICK)])[:NFFT]
        self.src_prev = src[:, -HOP:]
        return out * GAIN


def formants(x, nmax=5):
    """the peaks of the glottis->radiated transfer (radiation removed) for one posture: an instrument."""
    a, p, L = geometry(np.asarray(x, float)[None])
    Hg, _ = transfer(a, p, L, np.array([100.0]), np.array([0.05]))
    m = np.abs(Hg[0] / (1j * W))
    pk = [FREQ[i] for i in range(2, len(m) - 1) if m[i] > m[i - 1] and m[i] >= m[i + 1] and FREQ[i] > 150]
    return pk[:nmax], m
