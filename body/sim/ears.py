"""THE EARS (docs/SIM_DESIGN.md 3.4 channel 2, 3.7's orienting, 5.2; package P2): sounds placed in the room and heard at the G1's
two ear sites, two cochleas, the brainstem's delay lines and the born lateral read. Pure numpy; the world hands it, each tick, the
two ear sites' positions and every sounding source (its 2,400 samples as pressure at 1 m, and where it is).

  THE SPATIALIZER: the exact rigid sphere. The head is a rigid sphere through the two ear sites (centre: their midpoint; radius:
  half their distance, 0.079 m on the G1, whose ear sites are 15.8 cm apart). A point source's pressure at an ear on the sphere
  is the free field's at the centre, x(t - r/c) / r, times the sphere's exact transfer H(rho, ka, theta) (the classical series
  in spherical Hankel functions and Legendre polynomials, in Duda and Martens's 1998 range-dependent form): rho the source's
  distance over the radius, theta the angle between the ear and the source seen from the centre. It carries everything a
  sphere does to a sound, near or far: the head's shadow and the near ear's boost (+6 dB), the bright spot behind, the larger
  interaural delay at low frequency (3 a / c far away, Kuhn 1977, falling toward the high-frequency value above ka = 1), the
  near field's level differences; there is no fitted constant in it. It is applied in two parts, so a moving source or a
  turning head is heard without a click:
    - the bulk path, per ear: the ray model's path (the straight line when the ear is in view of the source, else the tangent
      line and the arc around the head: Woodworth's, exact for a near source as well as a far one), as a fractional delay line
      (a 16-tap Kaiser-windowed sinc: flat within 0.1 dB to 6 kHz and 2.2 dB down at 7 kHz at the worst fraction; a cubic
      Lagrange was 5 dB down at 6 kHz) with its 1 / path spreading, both swept linearly across the tick from the last tick's
      path to this one's (and with them the Doppler shift)
    - the rest of the sphere, per ear: the residual R = H (path / r) e^{ik (path - r)}, the sphere's response less the bulk
      path's delay and spreading, as a 97-tap FIR (a Tukey-tapered window of +-48 samples around its own zero lag), from a
      table built once, when the first Ears is made (0.2 s), over 1 / rho (25 steps from 0 to 1 / RHO_MIN) and theta (1 degree
      steps), bilinear between them;
      when the source moves against the head, last tick's filter is faded into this tick's across the tick
  Against the exact series (measured when the table was chosen, 2026-09-24): the FIR within 0.55 dB (99th percentile 0.10 dB)
  and 0.2 us of phase delay below 1.5 kHz; the table's interpolation within 0.03 dB in 1 / rho and 0.17 dB in theta. Every path
  carries a common latency of 3.5 ms (the delay line's 8 samples and the FIR's 48), which changes no difference between the ears.
  A source at or inside the sphere (the child's own speaker, on the head's front) is taken just off its surface, at RHO_MIN =
  1.02 (1.6 mm), where the series converges; its level at the ears changes by under 0.05 dB between rho 1.005 and 1.1. There is
  no floor on the level: pressure falls as the sphere's law says all the way in (the prototype's 1 / max(r, 0.3 m) is gone).
  No room echo here: the decision log's B9 (a first-order echo from the room's six surfaces, if it costs under 1 ms a tick) is
  W5's to measure and would enter as image sources through this same spatializer (at about 0.4 ms a source a tick, measured,
  six images would not fit B9's 1 ms: a cheaper far-field path for images is W5's to build). Sources beyond 20 m are heard as
  at 20 m (the delay line's length; never in this room).

  THE COCHLEAS. Per ear, 40 bands ERB-spaced 80-7,600 Hz, each a 4th-order gammatone's power response (Patterson; Glasberg and
  Moore's ERB) applied to a short-time spectrum (25 ms Hann, 10 ms hop, 512-point FFT): 15 frames a tick. Each band's power is in
  Pa^2 (calibrated: a tone at the band's centre of mean square p^2 reads p^2), and the code is its cube root re 60 dB SPL (the
  loudness power law's exponent on intensity, Stevens). The lowest bands are wider than their ERB: a 25 ms window resolves
  about 80 Hz (the fast form's price, disclosed). The prototype (allout/lang/ear2.py) weighted power by the gammatone's
  magnitude response, filters about 1.6 ERB wide; here the power response, 1.00 ERB (so the parent's-ear figures measured
  through ear2 in the voice study are to be measured again on this cochlea: P3v).

  THE DELAY LINES (the medial superior olive, Jeffress). For each band below 1,500 Hz (21 bands; fine-structure timing fades
  above it), the band-limited interaural cross-correlation over the tick at lags -L..+L samples (+ = the left ear leads),
  normalized as a correlation coefficient (divided by the geometric mean of the two ears' zero-lag band power in the same
  estimator, so its size is at most 1 by Cauchy-Schwarz, plus the band's threshold, so a sound near threshold drives the
  coincidences weakly), times the band's loudness on the cochlea's own scale (the cube root of the geometric mean of the two
  ears' band power over the tick, re 60 dB SPL): a coincidence cell fires with its inputs' rates, and the two halves of the
  code share one scale (a 60 dB band perfectly correlated reads 1 at its best lag, as its cochlea frames read 1). The
  prototype's delay lines had lost a factor NFFT / 2 = 256 in their normalization (the verifier's finding), so they were
  about 100 times smaller than the cochlea's numbers in the same vector. L is the head's own largest interaural delay in those
  bands (the exact sphere's, far field: 11.3 samples at 300 Hz, 0.71 ms), so L = 12 samples at 16 kHz; a source as near as
  0.12 m adds 0.2 samples (measured), inside it. (The design's +-8 was a 6.5 cm head's high-frequency delay.)

  THE BORN LATERAL READ (the orienting reflex's side, 3.7): for each band whose period is longer than twice the head's largest
  delay in that band (no lag in range is ambiguous: CF 80-757 Hz, 15 bands), the best lag by parabolic interpolation
  around the peak, turned into an angle by that band's own law: the exact sphere's far-field interaural phase delay at the band's
  centre over azimuth 0..90 degrees, inverted (the head's own acoustics, as anatomy; a lag past the law's 90 degrees reads 90);
  the angles averaged weighted by each band's power above threshold; + is left. It is 0 with no weight when no such band is
  above threshold. No learning.

  ONSETS (the cochlear nucleus's onset cells; the orienting trigger and the amygdala's two event lines): per ear and frame, the
  mean over bands of each band's rise in dB above its own running level, which adapts with a 50 ms time constant (the auditory
  nerve's short-term adaptation, about 40-60 ms: Westerman and Smith 1984, from memory); the tick's onset is its largest frame.
  An onset is heard when it passes ONSET_DB. The event lines are (an onset heard, and the lateral read not right of the midline)
  for the left and (an onset heard, and the read not left of it) for the right; the midline is the angle of half a sample of
  interaural delay by the lowest read band's law (2.58 degrees; a sound there fires both).

The channel code (the born projection reads it): the two cochleas' 15 x 40 frames and the 21 x 25 delay lines, 1,725 numbers a
tick (the design's 1,557 was 17 lags). On speech at 1.5 m the two parts are of one scale (RMS 0.23-0.25 each: the test and
tools/sim_voice_check.py measure it). World truth never enters it: Heard.levels (each source's level at each ear this tick,
dB SPL) and Heard.gains (each source's ray-path spreading to each ear, dB re 1 m, whatever it plays; the head's diffraction is
in the levels) are for the world's side only (the words channel's "audible", the instruments).

THE CONSTANTS (disclosed; ours unless marked): SR 16 kHz, the tick 2,400 samples; the speed of sound 343 m/s (world: air at 20 C);
the head a rigid sphere through the ear sites (anatomy: the real head is not a sphere, and the pinnae's cues are absent, as on the
G1's microphones); RHO_MIN 1.02; the residual's table (25 x 181, 256-point spectra) and its 97-tap Tukey window; the bands as
above; the delay lines' bands below 1,500 Hz and L from the head's own delays (anatomy); THRESH_DB 10 dB SPL per band (a young
ear's threshold is about 0-10 dB SPL at 1-4 kHz and higher at the extremes: from memory, one flat value here); the code's
reference 60 dB SPL; ONSET_TAU 50 ms and ONSET_DB 10 dB. Nothing here draws a random number; state() and load_state() continue
exactly. The ear sites (EAR_SITES) are the world's (body/sim/g1scene.py EAR_Y, EAR_XZ); the world hands their positions each tick,
and the test checks the two agree whenever g1scene.py is in the tree.
"""
import functools
import math

import numpy as np

SR = 16000
TICK = 2400
HOP = 160
WIN = 400
NFFT = 512
NF = TICK // HOP                         # 15 frames a tick
NB = 40
F_LO, F_HI = 80.0, 7600.0
C_SOUND = 343.0
HIST = 1024                              # samples of each source kept for the delay lines (64 ms, 20.6 m)
LOW_CF = 1500.0
P_REF = 20e-6
THRESH_DB = 10.0
CODE_REF_DB = 60.0
ONSET_TAU = 5                            # frames (50 ms)
ONSET_DB = 10.0
RHO_MIN = 1.02                           # a source at or inside the head is taken this far out (radii): the series converges
N_U, TH_STEP_DEG = 25, 1.0               # the residual's table: 1 / rho in 25 steps, theta in 1 degree steps
R_NFFT, FIR_K = 256, 48                  # its spectra (62.5 Hz apart) and the FIR's half-length (+-48 samples, 3 ms)
TOL = 1e-7                               # the series is summed until rho^-m falls below this

# the G1's head, in torso_link's frame (metres): the ear sites the world adds (g1scene: on the head's sides, at its widest point,
# 0.078 m from the midline; the real G1's 4-microphone array positions are not in the model: flagged) and the voice's source
# (the tract drives the G1's own loudspeaker; placed on the head's front, on the midline, below the camera window: assumed)
EAR_SITES = {"L": np.array([0.005, 0.079, 0.395]), "R": np.array([0.005, -0.079, 0.395])}
MOUTH_SITE = np.array([0.06, 0.0, 0.38])
HEAD_RADIUS = 0.079


def erb_space(lo=F_LO, hi=F_HI, n=NB):
    q, bw = 9.26449, 24.7
    return (np.exp(np.linspace(q * np.log(1 + lo / (q * bw)), q * np.log(1 + hi / (q * bw)), n) / q) - 1) * q * bw


CF = erb_space()
ERB = 24.7 * (4.37 * CF / 1000 + 1)
FREQS = np.fft.rfftfreq(NFFT, 1 / SR)
GT = (1 + ((FREQS[None, :] - CF[:, None]) / (1.019 * ERB[:, None])) ** 2) ** (-4.0)   # 4th-order gammatone, power response
LOW = np.where(CF < LOW_CF)[0]
HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)
TWO = np.r_[1.0, np.full(NFFT // 2 - 1, 2.0), 1.0]   # irfft's weights at lag 0 (the spectrum's two halves)


def _band_cal():
    """per band: the factor that makes a tone at the band's centre of mean square 1 read 1 (a frame in the middle of the tone)."""
    n = np.arange(WIN)
    cal = np.empty(NB)
    for b, f in enumerate(CF):
        x = math.sqrt(2) * np.cos(2 * np.pi * f * n / SR + 0.3)
        p = np.abs(np.fft.rfft(x * HANN, NFFT)) ** 2
        cal[b] = 1.0 / float(GT[b] @ p)
    return cal


CAL = _band_cal()
GTC = GT * CAL[:, None]                               # calibrated weights: band power in Pa^2 from |X|^2
GTC0 = GTC[LOW] * TWO / NFFT                          # the same bands' zero-lag power in the delay lines' own units
E_THR = (P_REF * 10 ** (THRESH_DB / 20)) ** 2         # a band's threshold power (Pa^2)
E_CODE = (P_REF * 10 ** (CODE_REF_DB / 20)) ** 2
THR0 = NF * E_THR * 2.0 / NFFT                        # a tick's band threshold in the delay lines' units


# ----------------------------------------------------------------------------------------------------------- the sphere
def _hankel_ratios(rho, mu, M):
    """T_m = h_m(mu rho) / h_m'(mu) for m = 0..M-1 (h the outgoing spherical Hankel function, e^{i w t}: h_0(x) = i e^{-ix} / x),
    by the ratios q_m(x) = h_m(x) / h_{m-1}(x), q_1 = 1 / x + i, q_{m+1} = (2m + 1) / x - 1 / q_m, which never overflow.
    mu: an array (ka); -> [len(mu), M] complex."""
    mu = np.asarray(mu, float)
    x1, x0 = mu * rho, mu
    out = np.empty(mu.shape + (M,), complex)
    qa, qb = 1 / x1 + 1j, 1 / x0 + 1j
    r = (np.exp(-1j * x1) / x1) / (np.exp(-1j * x0) / x0)          # h_0(x1) / h_0(x0)
    out[..., 0] = r / (-qb)                                          # h_0' = -h_1
    for m in range(1, M):
        r = r * qa / qb                                              # h_m(x1) / h_m(x0)
        out[..., m] = r / (1 / qb - (m + 1) / x0)                    # h_m' = h_{m-1} - (m + 1) / x h_m
        qa, qb = (2 * m + 1) / x1 - 1 / qa, (2 * m + 1) / x0 - 1 / qb
    return out


def _legendre(M, ct):
    P = np.empty((M,) + np.shape(ct))
    P[0] = 1.0
    if M > 1:
        P[1] = ct
    for m in range(1, M - 1):
        P[m + 1] = ((2 * m + 1) * ct * P[m] - m * P[m - 1]) / (m + 1)
    return P


def _terms(rho, mu_max):
    return int(math.ceil(mu_max) + 40 + (math.ceil(math.log(TOL) / -math.log(rho)) if rho < 1e3 else 0))


def sphere_H(rho, mu, theta):
    """the rigid sphere's exact transfer to a point on its surface (Duda and Martens 1998's form, in e^{i w t}): the pressure there
    over the free field's at the centre, for a point source at rho radii, ka = mu (array), at angle theta (array, rad) from the
    point. The series is summed in order (no BLAS), so it is the same number in every process. -> [len(mu), len(theta)]."""
    mu = np.atleast_1d(np.asarray(mu, float))
    th = np.atleast_1d(np.asarray(theta, float))
    M = _terms(rho, mu.max())
    T = _hankel_ratios(rho, mu, M) * (2 * np.arange(M) + 1)
    P = _legendre(M, np.cos(th))
    s = np.zeros((len(mu), len(th)), complex)
    for m in range(M):
        s += T[:, m, None] * P[m][None, :]
    return -(rho / mu)[:, None] * np.exp(1j * mu * rho)[:, None] * s


def ray_path(rho, theta):
    """the ray model's path (in radii) to a point on the sphere at angle theta from a source at rho radii (rho >= 1): the straight
    line while the point is in view, else the tangent and the arc (Woodworth's, exact near as well as far)."""
    beta = np.arccos(np.minimum(1.0, 1.0 / rho))
    straight = np.sqrt(np.maximum(rho * rho + 1 - 2 * rho * np.cos(theta), 0.0))
    arc = np.sqrt(np.maximum(rho * rho - 1, 0.0)) + (theta - beta)
    return np.where(theta <= beta, straight, arc)


def _tukey(K):
    t = np.abs(np.arange(-K, K + 1)) / K
    return np.where(t < 0.5, 1.0, 0.5 + 0.5 * np.cos(np.pi * (t - 0.5) / 0.5))


@functools.lru_cache(maxsize=4)
def residual_table(radius=HEAD_RADIUS):
    """the sphere less the bulk path, as FIRs: [N_U, n_theta, 2K + 1], tap K at zero lag; U the 1 / rho grid, TH theta (rad)."""
    f = np.fft.rfftfreq(R_NFFT, 1 / SR)
    f[0] = 1.0                                                      # the low-frequency limit (1 Hz)
    mu = 2 * np.pi * f * radius / C_SOUND
    th = np.radians(np.arange(0.0, 180.0 + TH_STEP_DEG / 2, TH_STEP_DEG))
    us = np.linspace(0.0, 1.0 / RHO_MIN, N_U)
    win = _tukey(FIR_K)
    fir = np.empty((N_U, len(th), 2 * FIR_K + 1))
    for i, u in enumerate(us):
        rho = 1e4 if u == 0 else 1.0 / u                            # u = 0: a plane wave (a source 790 m away)
        H = sphere_H(rho, mu, th)                                   # [F, th]
        p = ray_path(rho, th)
        R = H * (p / rho)[None, :] * np.exp(1j * mu[:, None] * (p - rho)[None, :])
        ir = np.fft.irfft(R.T, R_NFFT, axis=-1)                     # [th, R_NFFT], zero lag at 0
        ir = np.concatenate([ir[:, -FIR_K:], ir[:, :FIR_K + 1]], 1)
        fir[i] = ir * win
    return us, th, fir


@functools.lru_cache(maxsize=4)
def lateral_laws(radius=HEAD_RADIUS):
    """per delay-line band: the exact sphere's far-field interaural phase delay (s; + = the left ear leads) at its centre, over
    azimuth 0..90 degrees in 0.25 degree steps (the ears' axis in the plane) -> (azimuths in rad, [n_low, n_az] delays)."""
    az = np.radians(np.arange(0.0, 90.0 + 0.125, 0.25))
    mu = 2 * np.pi * CF[LOW] * radius / C_SOUND
    rho = 1e4
    hl = sphere_H(rho, mu, np.pi / 2 - az)                            # the left ear: theta = 90 - azimuth
    hr = sphere_H(rho, mu, np.pi / 2 + az)
    dphi = np.unwrap(np.angle(hl / hr), axis=1)
    return az, dphi / (2 * np.pi * CF[LOW][:, None])


_AZ, _ITD = lateral_laws()
ITD_MAX = _ITD[:, -1]                                               # per low band: the head's largest delay (far field, 90 deg)
LAGS = int(math.ceil(ITD_MAX.max() * SR))                            # 12: the head's own largest delay in the delay lines' bands
READ = LOW[CF[LOW] < 1.0 / (2.0 * ITD_MAX)]                           # the lateral read's bands: a period over twice that delay
_READ_ROWS = np.isin(LOW, READ)
N_CODE = 2 * NF * NB + len(LOW) * (2 * LAGS + 1)
MIDLINE = float(np.interp(0.5 / SR, _ITD[0], _AZ))                    # half a sample of interaural delay, the lowest band's law


# ------------------------------------------------------------------------------------------------------------------ geometry
def head_from_torso(pos, R, sites=None, mouth=MOUTH_SITE):
    """the G1's ear sites and mouth in the world, from torso_link's position and rotation (columns: its axes in the world)."""
    sites = EAR_SITES if sites is None else sites
    pos, R = np.asarray(pos, float), np.asarray(R, float)
    return pos + R @ np.asarray(sites["L"], float), pos + R @ np.asarray(sites["R"], float), pos + R @ np.asarray(mouth, float)


def source_geometry(src, ear_l, ear_r):
    """a point source against the sphere through the two ear sites -> (rho in radii, taken at least RHO_MIN; per ear the angle
    theta (rad) between the ear and the source seen from the centre; per ear the ray path in metres; the radius)."""
    ear_l, ear_r, src = np.asarray(ear_l, float), np.asarray(ear_r, float), np.asarray(src, float)
    c = 0.5 * (ear_l + ear_r)
    a = 0.5 * float(np.linalg.norm(ear_l - ear_r))
    axis = (ear_l - ear_r) / (2 * a)
    d = src - c
    D = float(np.linalg.norm(d))
    if D > 1e-9:
        u = d / D
    else:                                                     # a source at the centre: any direction off the ears' axis
        e = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        u = e - (e @ axis) * axis
        u = u / np.linalg.norm(u)
    rho = max(D / a, RHO_MIN)
    th = np.array([math.acos(float(np.clip(u @ axis, -1.0, 1.0))), math.acos(float(np.clip(-(u @ axis), -1.0, 1.0)))])
    return rho, th, a * ray_path(rho, th), a


def paths(src, ear_l, ear_r):
    """the ray path length (m) from a point source to each ear site, the angle (rad) between each ear and the source seen from
    the head's centre, and the radius."""
    rho, th, p, a = source_geometry(src, ear_l, ear_r)
    return p, th, a


def residual_fir(rho, theta, radius=HEAD_RADIUS):
    """the sphere's residual FIR for one ear (bilinear in 1 / rho and theta over the table)."""
    us, th, fir = residual_table(radius)
    x = min(1.0 / rho, us[-1]) / us[-1] * (len(us) - 1)
    i = min(int(x), len(us) - 2)
    fx = x - i
    y = min(max(theta, 0.0), math.pi) / math.pi * (len(th) - 1)
    j = min(int(y), len(th) - 2)
    fy = y - j
    return ((1 - fx) * ((1 - fy) * fir[i, j] + fy * fir[i, j + 1]) + fx * ((1 - fy) * fir[i + 1, j] + fy * fir[i + 1, j + 1]))


def _kernel_table(taps=16, phases=512, beta=6.0):
    """the fractional delay's kernel: a Kaiser-windowed sinc of `taps` taps (k = -taps/2+1 .. taps/2) at phases+1 fractions."""
    k = np.arange(-taps // 2 + 1, taps // 2 + 1)
    f = np.arange(phases + 1)[:, None] / phases
    t = k[None, :] - f
    w = np.i0(beta * np.sqrt(np.clip(1 - (t / (taps / 2)) ** 2, 0, None))) / np.i0(beta)
    h = np.sinc(t) * w
    return h / h.sum(1, keepdims=True)


TAPS, PHASES = 16, 512
KERNEL = _kernel_table(TAPS, PHASES)
LOOKAHEAD = TAPS // 2                    # the delay line's latency (0.5 ms); the FIR adds FIR_K (3 ms): 3.5 ms on every path
LATENCY = LOOKAHEAD + FIR_K
N_CONV = 2592                            # the FIR by FFT: TICK + 2 FIR_K + 2 FIR_K samples (2^5 3^4)


def frac_read(buf, pos):
    """buf read at fractional positions pos (a 16-tap Kaiser-windowed sinc, its phase interpolated from a 512-phase table)."""
    i = np.floor(pos).astype(np.int64)
    p = (pos - i) * PHASES
    j = np.minimum(np.floor(p).astype(np.int64), PHASES - 1)
    t = (p - j)[:, None]
    W = KERNEL[j] + (KERNEL[j + 1] - KERNEL[j]) * t
    win = np.lib.stride_tricks.sliding_window_view(buf, TAPS)[i - (TAPS // 2 - 1)]
    return (win * W).sum(1)


def fir_apply(y_ext, firs):
    """y_ext: the delayed signal with its last 2 FIR_K samples before the tick in front -> each FIR's TICK outputs (by FFT:
    pocketfft, the same numbers in every process)."""
    Y = np.fft.rfft(y_ext, N_CONV)
    return [np.fft.irfft(Y * np.fft.rfft(h, N_CONV), N_CONV)[2 * FIR_K:2 * FIR_K + TICK] for h in firs]


class _Source:
    __slots__ = ("hist", "delay", "gain", "fir", "ytail", "pos", "quiet")

    def __init__(self):
        self.hist = np.zeros(HIST)
        self.delay = None                 # samples, per ear
        self.gain = None
        self.fir = None                   # per ear: the sphere's residual FIR last tick
        self.ytail = np.zeros((2, 2 * FIR_K))   # per ear: the bulk path's last 2 FIR_K samples
        self.pos = None
        self.quiet = 0                    # ticks since it last sounded


class Heard:
    """one tick of hearing."""
    __slots__ = ("left", "right", "delay", "lateral", "lateral_weight", "onset", "onset_heard", "events", "levels", "gains",
                 "ear_l", "ear_r")

    def code(self):
        """the channel's numbers: the two cochleas' frames and the delay lines (float32)."""
        return np.concatenate([self.left.ravel(), self.right.ravel(), self.delay.ravel()]).astype(np.float32)


class Ears:
    def __init__(self, radius=HEAD_RADIUS):
        self.radius = float(radius)
        residual_table(self.radius)
        self.src = {}
        self.tail = np.zeros((2, WIN - HOP))
        self.run = None                   # per ear and band: the running level (dB) for the onsets
        self.ticks = 0
        self.gains = {}

    # ---------------------------------------------------------------------------------------------------------- the room
    def spatialize(self, ear_l, ear_r, sources):
        """sources: {name: (samples at 1 m in pascals, position)} -> (left, right) pressure at the two ear sites, and each
        source's level at each ear this tick ({name: (dB SPL left, dB SPL right)})."""
        L = np.zeros(TICK)
        R = np.zeros(TICK)
        levels, self.gains = {}, {}
        names = sorted(set(self.src) | set(sources))
        n = np.arange(TICK)
        ramp = (n + 1) / TICK
        for name in names:
            s = self.src.get(name)
            if s is None:
                s = self.src[name] = _Source()
            if name in sources:
                x, p = sources[name]
                x = np.asarray(x, np.float64)
                assert x.shape == (TICK,), f"source {name!r}: {x.shape} samples, not {TICK}"
                s.pos = np.asarray(p, float)
                s.quiet = 0 if np.any(x) else s.quiet + 1
            else:
                x = np.zeros(TICK)
                s.quiet += 1
            rho, th, path, a = source_geometry(s.pos, ear_l, ear_r)
            assert abs(a - self.radius) < 1e-3, f"the ear sites are {2 * a:.4f} m apart, not {2 * self.radius:.4f}"
            path = np.minimum(path, C_SOUND * (HIST - 2 * TAPS) / SR)
            delay = path / C_SOUND * SR + LOOKAHEAD
            gain = 1.0 / path
            fir = [residual_fir(rho, th[e], self.radius) for e in (0, 1)]
            if s.delay is None:
                s.delay, s.gain, s.fir = delay.copy(), gain.copy(), fir
            buf = np.concatenate([s.hist, x])
            lv = []
            for e, out in ((0, L), (1, R)):
                d = s.delay[e] + (delay[e] - s.delay[e]) * ramp
                g = s.gain[e] + (gain[e] - s.gain[e]) * ramp
                y = frac_read(buf, HIST + n - d) * g
                y_ext = np.concatenate([s.ytail[e], y])
                s.ytail[e] = y[-2 * FIR_K:]
                if np.array_equal(s.fir[e], fir[e]):
                    o, = fir_apply(y_ext, [fir[e]])
                else:                                            # the source moved against the head: last tick's faded
                    o0, o1 = fir_apply(y_ext, [s.fir[e], fir[e]])  # into this one's
                    o = o0 + (o1 - o0) * ramp
                out += o
                lv.append(10 * math.log10(max(float(np.mean(o * o)), 1e-30) / P_REF ** 2))
            levels[name] = tuple(lv)
            self.gains[name] = tuple(20 * math.log10(float(g)) for g in gain)
            s.hist = buf[-HIST:]
            s.delay, s.gain, s.fir = delay, gain, fir
            if name not in sources and s.quiet * TICK > HIST + 4 * FIR_K and not np.any(s.ytail):
                del self.src[name]
        return L, R, levels

    # ------------------------------------------------------------------------------------------------------ the cochleas
    def _spectra(self, L, R):
        idx = np.arange(WIN)[None, :] + HOP * np.arange(NF)[:, None]
        xl = np.concatenate([self.tail[0], L])[idx] * HANN
        xr = np.concatenate([self.tail[1], R])[idx] * HANN
        self.tail = np.stack([L[-(WIN - HOP):], R[-(WIN - HOP):]])
        return np.fft.rfft(xl, NFFT, axis=1), np.fft.rfft(xr, NFFT, axis=1)

    def tick(self, ear_l, ear_r, sources):
        """one tick: the sources heard at the two ear sites -> Heard."""
        L, R, levels = self.spatialize(ear_l, ear_r, sources)
        XL, XR = self._spectra(L, R)
        PL, PR = np.abs(XL) ** 2, np.abs(XR) ** 2
        EL, ER = PL @ GTC.T, PR @ GTC.T                              # [15, 40] band power, Pa^2
        h = Heard()
        h.left = np.cbrt(EL / E_CODE)
        h.right = np.cbrt(ER / E_CODE)
        # the delay lines: per low band, the band-limited cross-correlation over the tick, as a correlation coefficient
        S = (np.conj(XL) * XR).sum(0)                                # the right ear's signal against the left's
        cc = np.fft.irfft(GTC[LOW] * S[None, :], NFFT, axis=1)
        cc = np.concatenate([cc[:, -LAGS:], cc[:, :LAGS + 1]], 1)   # lags -L..+L: + means the left ear leads
        a0l, a0r = (GTC0 * PL.sum(0)[None, :]).sum(1), (GTC0 * PR.sum(0)[None, :]).sum(1)   # the same estimator at lag 0
        corr = cc / (np.sqrt(a0l * a0r)[:, None] + THR0)             # |corr| <= 1 (Cauchy-Schwarz)
        el, er = EL[:, LOW].sum(0), ER[:, LOW].sum(0)
        h.delay = corr * np.cbrt(np.sqrt(el * er) / (NF * E_CODE))[:, None]   # x the band's loudness, the cochlea's scale
        # the born lateral read: the unambiguous low bands' best lags, each by its own law, weighted by power above threshold
        w = np.maximum(el + er - 2 * NF * E_THR, 0.0)
        k = corr.argmax(1)
        rows = np.arange(len(k))
        y1 = corr[rows, k]
        inner = (k > 0) & (k < 2 * LAGS)
        y0 = corr[rows, np.clip(k - 1, 0, 2 * LAGS)]
        y2 = corr[rows, np.clip(k + 1, 0, 2 * LAGS)]
        den = np.where(inner, y0 - 2 * y1 + y2, -1.0)
        frac = np.where(inner & (den < -1e-12), 0.5 * (y0 - y2) / np.where(den < -1e-12, den, -1.0), 0.0)
        lag = (k - LAGS) + np.clip(frac, -0.5, 0.5)
        w = w * (_READ_ROWS & (y1 > 0))
        h.lateral_weight = float(w.sum())
        if h.lateral_weight > 0:
            ang = np.zeros(len(LOW))
            for b in np.nonzero(w)[0]:
                ang[b] = math.copysign(float(np.interp(abs(lag[b]) / SR, _ITD[b], _AZ)), lag[b])
            h.lateral = float((w * ang).sum() / h.lateral_weight)
        else:
            h.lateral = 0.0
        # onsets
        lev = 10 * np.log10(np.maximum(np.stack([EL, ER]), E_THR) / E_THR)     # [2, 15, 40] dB over threshold
        if self.run is None:
            self.run = np.zeros((2, NB))
        on = np.zeros((2, NF))
        for f in range(NF):
            on[:, f] = np.maximum(lev[:, f] - self.run, 0.0).mean(1)
            self.run += (lev[:, f] - self.run) / ONSET_TAU
        h.onset = on.max(1)
        h.onset_heard = bool(h.onset.max() >= ONSET_DB)
        h.events = (h.onset_heard and h.lateral > -MIDLINE, h.onset_heard and h.lateral < MIDLINE)
        h.levels, h.gains = levels, dict(self.gains)
        h.ear_l, h.ear_r = L, R
        self.ticks += 1
        return h

    # ---------------------------------------------------------------------------------------------------- save, restore
    def state(self):
        return dict(radius=self.radius, tail=self.tail.copy(), run=None if self.run is None else self.run.copy(), ticks=self.ticks,
                    src={k: dict(hist=s.hist.copy(), delay=None if s.delay is None else s.delay.copy(),
                                 gain=None if s.gain is None else s.gain.copy(), ytail=s.ytail.copy(),
                                 fir=None if s.fir is None else [f.copy() for f in s.fir],
                                 pos=None if s.pos is None else s.pos.copy(), quiet=s.quiet) for k, s in self.src.items()})

    def load_state(self, st):
        self.radius, self.tail, self.ticks = st["radius"], st["tail"].copy(), st["ticks"]
        residual_table(self.radius)
        self.run = None if st["run"] is None else st["run"].copy()
        self.src = {}
        for k, v in st["src"].items():
            s = _Source()
            s.hist = v["hist"].copy()
            s.delay = None if v["delay"] is None else v["delay"].copy()
            s.gain = None if v["gain"] is None else v["gain"].copy()
            s.ytail = v["ytail"].copy()
            s.fir = None if v["fir"] is None else [f.copy() for f in v["fir"]]
            s.pos = None if v["pos"] is None else v["pos"].copy()
            s.quiet = v["quiet"]
            self.src[k] = s


def lateral_angle(itd, band=0):
    """an interaural delay (s; + = the left ear leads) -> the angle (rad, + = left) by one delay-line band's law (the exact
    sphere's far-field phase delay at its centre; the lowest band's by default), saturating at +-90 degrees."""
    return math.copysign(float(np.interp(abs(itd), _ITD[band], _AZ)), itd)


def cochlea(x):
    """one ear's cochlea over a whole mono signal (pascals): band power (Pa^2) per 10 ms frame, [frames, 40], each frame the 25 ms
    ending at its hop (the same filterbank as the child's ears; the parent's ear for the child's voice reads it, P3)."""
    x = np.concatenate([np.zeros(WIN - HOP), np.asarray(x, np.float64)])
    nf = max(0, (len(x) - WIN) // HOP + 1)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(nf)[:, None]
    return (np.abs(np.fft.rfft(x[idx] * HANN, NFFT, axis=1)) ** 2) @ GTC.T
