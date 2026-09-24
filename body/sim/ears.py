"""THE EARS (docs/SIM_DESIGN.md 3.4 channel 2, 3.7's orienting, 5.2; package P2): sounds placed in the room and heard at the G1's
two ear sites, two cochleas, the brainstem's delay lines and the born lateral read. Pure numpy; the world hands it, each tick, the
two ear sites' positions and every sounding source (its 2,400 samples as pressure at 1 m, and where it is).

  THE SPATIALIZER. Each source reaches each ear along its own path, as sound does around a head:
    - the head is a rigid sphere through the two ear sites (centre: their midpoint; radius: half their distance, 0.079 m on the
      G1, whose ear sites are 15.8 cm apart); a source inside the sphere (the child's own mouth) is taken on its surface
    - the path: the straight line when the ear is in view of the source, else the tangent line and the arc around the head (the
      ray model of diffraction, as in Woodworth's formula, but exact for a near source as well as a far one)
    - the delay: the path over the speed of sound, as a fractional delay line (a 16-tap Kaiser-windowed sinc: flat within 0.1 dB
      to 6 kHz and 2.2 dB down at 7 kHz at the worst fraction; a cubic Lagrange was 5 dB down at 6 kHz, which made a source's
      treble depend on where it stood to a fraction of a sample), swept linearly across the tick from the last tick's path to
      this one's, so a moving source or a turning head is heard without a click (and with its Doppler shift); every path
      carries the interpolator's common 0.5 ms (8 samples), which changes no difference between the ears
    - the level: 1 / path (pressure falls as 1/r), held at 1 / NEAR within NEAR of the ear (a source's own size: the child's own
      voice then arrives at +14 dB over the parent's at 1.5 m, the design's 4.9 figure)
    - the head's shadow: Brown and Duda's (1998) one-pole one-zero filter per ear, its high-frequency gain set by the angle
      between the ear's axis and the source (+6 dB facing it, -20 dB at 150 degrees, the bright spot behind); when the angle
      changes, last tick's filter fades into this tick's across the tick. The filter's own phase adds to the path's delay at
      low frequency, which gives a sphere's larger low-frequency interaural delay (Kuhn's 3 a / c against the ray's 2.57)
  No room echo here: the decision log's B9 (a first-order echo from the room's six surfaces, if it costs under 1 ms a tick) is
  W5's to measure and would enter as image sources through this same spatializer. Sources beyond 20 m are heard as at 20 m (the
  delay line's length; never in this room).

  THE COCHLEAS. Per ear, 40 bands ERB-spaced 80-7,600 Hz, each a 4th-order gammatone's power response (Patterson; Glasberg and
  Moore's ERB) applied to a short-time spectrum (25 ms Hann, 10 ms hop, 512-point FFT): 15 frames a tick. Each band's power is in
  Pa^2 (calibrated: a tone at the band's centre of mean square p^2 reads p^2), and the code is its cube root re 60 dB SPL (the
  loudness power law's exponent on intensity, Stevens). The lowest bands are wider than their ERB: a 25 ms window resolves
  about 80 Hz (the fast form's price, disclosed). The prototype (allout/lang/ear2.py) weighted power by the gammatone's
  magnitude response, filters about 1.5 times too wide; here the power response.

  THE DELAY LINES (the medial superior olive, Jeffress). For each band below 1,500 Hz (21 bands; fine-structure timing fades
  above it), the band-limited interaural cross-correlation over the tick at lags -L..+L samples (+ = the left ear leads). L is
  the head's own largest delay at low frequency: a sphere's is 3 a / c (Kuhn 1977; the head shadow's own phase adds it to the
  path's, as the spatializer's filter does), 0.69 ms for the G1's 0.079 m, so L = 12 samples at 16 kHz (the design's +-8 was a
  6.5 cm head's high-frequency delay). Each is normalized by the two ears' band powers plus the band's threshold, so a sound
  near threshold drives the coincidences weakly.

  THE BORN LATERAL READ (the orienting reflex's side, 3.7): the best lag of each band whose period is longer than twice the
  head's largest delay (CF below c / (6 a) = 724 Hz: 14 bands, where no lag in range is ambiguous), by parabolic interpolation
  around the peak, averaged weighted by the band's power above threshold, is an interaural delay; the head's own radius turns
  it into an angle by the same low-frequency law inverted, asin(itd c / (3 a)): + is left. It is 0 with no weight when no such
  band is above threshold. No learning.

  ONSETS (the cochlear nucleus's onset cells; the orienting trigger and the amygdala's two event lines): per ear and frame, the
  mean over bands of each band's rise in dB above its own running level, which adapts with a 50 ms time constant (the auditory
  nerve's short-term adaptation, about 40-60 ms: Westerman and Smith 1984, from memory); the tick's onset is its largest frame.
  An onset is heard when it passes ONSET_DB. Measured on the parent's lines with pauses: every line's start is an onset, and 29%
  of its sounding ticks are (syllables rising). The event lines are (an onset heard, and the lateral read not right of the
  midline) for the left and (an onset heard, and the read not left of it) for the right; the midline is +-2.6 degrees, half a
  sample of interaural delay (a sound there fires both).

The channel code (the born projection reads it): the two cochleas' 15 x 40 frames and the 21 x 25 delay lines, 1,725 numbers a
tick (the design's 1,557 was 17 lags). World truth never enters it: Heard.levels (each source's level at each ear this tick, dB
SPL) and Heard.gains (each source's path gain to each ear, dB re 1 m, whatever it plays) are for the world's side only (the words
channel's "audible", the instruments).

THE CONSTANTS (disclosed; ours unless marked): SR 16 kHz, the tick 2,400 samples; the speed of sound 343 m/s (world: air at 20 C);
NEAR 0.3 m; the sphere and the shadow's alpha_min 0.1 and theta_min 150 degrees (Brown and Duda); the bands as above; the delay
lines' bands below 1,500 Hz and L from the head's radius (anatomy); THRESH_DB 10 dB SPL per band (a young ear's threshold is
about 0-10 dB SPL at 1-4 kHz and higher at the extremes: from memory, one flat value here); the code's reference 60 dB SPL;
ONSET_TAU 50 ms and ONSET_DB 10 dB. Nothing here draws a random number; state() and load_state() continue exactly.
"""
import math

import numpy as np
from scipy.signal import lfilter

SR = 16000
TICK = 2400
HOP = 160
WIN = 400
NFFT = 512
NF = TICK // HOP                         # 15 frames a tick
NB = 40
F_LO, F_HI = 80.0, 7600.0
C_SOUND = 343.0
NEAR = 0.3
HIST = 1024                              # samples of each source kept for the delay lines (64 ms, 20.6 m)
ALPHA_MIN, THETA_MIN = 0.1, math.radians(150.0)
LOW_CF = 1500.0
P_REF = 20e-6
THRESH_DB = 10.0
CODE_REF_DB = 60.0
ONSET_TAU = 5                            # frames (50 ms)
ONSET_DB = 10.0

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
E_THR = (P_REF * 10 ** (THRESH_DB / 20)) ** 2         # a band's threshold power (Pa^2)
E_CODE = (P_REF * 10 ** (CODE_REF_DB / 20)) ** 2
KUHN = 3.0                                            # a sphere's low-frequency interaural delay: KUHN (a / c) sin(azimuth)
LAGS = int(math.ceil(KUHN * HEAD_RADIUS / C_SOUND * SR))   # 12: the head's own largest delay at low frequency, 0.69 ms
READ = np.where(CF < C_SOUND / (2 * KUHN * HEAD_RADIUS))[0]  # the lateral read's bands: a period longer than twice that delay
N_CODE = 2 * NF * NB + len(LOW) * (2 * LAGS + 1)
MIDLINE = math.asin(0.5 * C_SOUND / (KUHN * HEAD_RADIUS * SR))   # 2.6 degrees: half a sample of interaural delay


# ------------------------------------------------------------------------------------------------------------------ geometry
def head_from_torso(pos, R):
    """the G1's ear sites and mouth in the world, from torso_link's position and rotation (columns: its axes in the world)."""
    pos, R = np.asarray(pos, float), np.asarray(R, float)
    return pos + R @ EAR_SITES["L"], pos + R @ EAR_SITES["R"], pos + R @ MOUTH_SITE


def paths(src, ear_l, ear_r):
    """the path length (m) from a point source to each ear site around a rigid sphere through them, and the angle (rad) between
    each ear's axis and the source seen from the head's centre."""
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
    De = max(D, a)
    beta = math.acos(min(1.0, a / De))
    out, ang = [], []
    for ax in (axis, -axis):
        phi = math.acos(float(np.clip(u @ ax, -1.0, 1.0)))
        if phi <= beta:
            p = math.sqrt(max(De * De + a * a - 2 * a * De * math.cos(phi), 0.0))
        else:
            p = math.sqrt(max(De * De - a * a, 0.0)) + a * (phi - beta)
        out.append(p)
        ang.append(phi)
    return np.array(out), np.array(ang), a


def shadow_coeffs(phi, a):
    """Brown and Duda's head shadow for one ear (angle phi from its axis), bilinear transform at SR: (b, a) of a first-order IIR."""
    alpha = (1 + ALPHA_MIN / 2) + (1 - ALPHA_MIN / 2) * math.cos(phi / THETA_MIN * math.pi)
    w0 = C_SOUND / a
    K = 2.0 * SR
    b0, b1 = 2 * w0 + alpha * K, 2 * w0 - alpha * K
    a0, a1 = 2 * w0 + K, 2 * w0 - K
    return np.array([b0 / a0, b1 / a0]), np.array([1.0, a1 / a0])


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
LOOKAHEAD = TAPS // 2                    # a common 0.5 ms latency on every path: the interpolator's right half


def frac_read(buf, pos):
    """buf read at fractional positions pos (a 16-tap Kaiser-windowed sinc, its phase interpolated from a 512-phase table)."""
    i = np.floor(pos).astype(np.int64)
    p = (pos - i) * PHASES
    j = np.minimum(np.floor(p).astype(np.int64), PHASES - 1)
    t = (p - j)[:, None]
    W = KERNEL[j] + (KERNEL[j + 1] - KERNEL[j]) * t
    win = np.lib.stride_tricks.sliding_window_view(buf, TAPS)[i - (TAPS // 2 - 1)]
    return (win * W).sum(1)


class _Source:
    __slots__ = ("hist", "delay", "gain", "coef", "zi", "pos", "quiet")

    def __init__(self):
        self.hist = np.zeros(HIST)
        self.delay = None                 # samples, per ear
        self.gain = None
        self.coef = None                  # per ear: the head shadow's (b, a) last tick
        self.zi = np.zeros((2, 1))
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
            path, phi, a = paths(s.pos, ear_l, ear_r)
            assert abs(a - self.radius) < 1e-3, f"the ear sites are {2 * a:.4f} m apart, not {2 * self.radius:.4f}"
            path = np.minimum(path, C_SOUND * (HIST - 2 * TAPS) / SR)
            delay = path / C_SOUND * SR + LOOKAHEAD
            gain = 1.0 / np.maximum(path, NEAR)
            coef = [shadow_coeffs(phi[e], a) for e in (0, 1)]
            if s.delay is None:
                s.delay, s.gain, s.coef = delay.copy(), gain.copy(), coef
            buf = np.concatenate([s.hist, x])
            lv = []
            for e, out in ((0, L), (1, R)):
                d = s.delay[e] + (delay[e] - s.delay[e]) * ramp
                g = s.gain[e] + (gain[e] - s.gain[e]) * ramp
                y = frac_read(buf, HIST + n - d) * g
                (b0, a0), (b1, a1) = s.coef[e], coef[e]
                y1, zi = lfilter(b1, a1, y, zi=s.zi[e])
                if not (np.array_equal(b0, b1) and np.array_equal(a0, a1)):
                    y0, _ = lfilter(b0, a0, y, zi=s.zi[e])        # the shadow moved: last tick's filter faded into this one's
                    y1 = y0 + (y1 - y0) * ramp
                s.zi[e] = zi
                y = y1
                out += y
                lv.append(10 * math.log10(max(float(np.mean(y * y)), 1e-30) / P_REF ** 2))
            levels[name] = tuple(lv)
            self.gains[name] = tuple(20 * math.log10(float(g)) for g in gain)
            s.hist = buf[-HIST:]
            s.delay, s.gain, s.coef = delay, gain, coef
            if name not in sources and s.quiet * TICK > HIST and not np.any(np.abs(s.zi) > 1e-12):
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
        # the delay lines: per low band, the band-limited cross-correlation over the tick
        S = (np.conj(XL) * XR).sum(0)                                # the right ear's signal against the left's
        cc = np.fft.irfft(GTC[LOW] * S[None, :], NFFT, axis=1)
        cc = np.concatenate([cc[:, -LAGS:], cc[:, :LAGS + 1]], 1)   # lags -L..+L: + means the left ear leads
        el, er = EL[:, LOW].sum(0), ER[:, LOW].sum(0)
        h.delay = cc / (np.sqrt(el * er)[:, None] + NF * E_THR)
        # the born lateral read: the unambiguous low bands' best lags, weighted by their power above threshold
        w = np.maximum(el + er - 2 * NF * E_THR, 0.0)
        k = h.delay.argmax(1)
        rows = np.arange(len(k))
        y1 = h.delay[rows, k]
        inner = (k > 0) & (k < 2 * LAGS)
        y0 = h.delay[rows, np.clip(k - 1, 0, 2 * LAGS)]
        y2 = h.delay[rows, np.clip(k + 1, 0, 2 * LAGS)]
        den = np.where(inner, y0 - 2 * y1 + y2, -1.0)
        frac = np.where(inner & (den < -1e-12), 0.5 * (y0 - y2) / np.where(den < -1e-12, den, -1.0), 0.0)
        lag = (k - LAGS) + np.clip(frac, -0.5, 0.5)
        read = np.isin(LOW, READ) & (y1 > 0)
        w = w * read
        h.lateral_weight = float(w.sum())
        h.lateral = lateral_angle(float((w * lag).sum() / w.sum()) / SR, self.radius) if h.lateral_weight > 0 else 0.0
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
                                 gain=None if s.gain is None else s.gain.copy(), zi=s.zi.copy(),
                                 coef=None if s.coef is None else [(b.copy(), a.copy()) for b, a in s.coef],
                                 pos=None if s.pos is None else s.pos.copy(), quiet=s.quiet) for k, s in self.src.items()})

    def load_state(self, st):
        self.radius, self.tail, self.ticks = st["radius"], st["tail"].copy(), st["ticks"]
        self.run = None if st["run"] is None else st["run"].copy()
        self.src = {}
        for k, v in st["src"].items():
            s = _Source()
            s.hist = v["hist"].copy()
            s.delay = None if v["delay"] is None else v["delay"].copy()
            s.gain = None if v["gain"] is None else v["gain"].copy()
            s.zi = v["zi"].copy()
            s.coef = None if v["coef"] is None else [(b.copy(), a.copy()) for b, a in v["coef"]]
            s.pos = None if v["pos"] is None else v["pos"].copy()
            s.quiet = v["quiet"]
            self.src[k] = s


def lateral_angle(itd, radius=HEAD_RADIUS):
    """an interaural delay (s; + = the left ear leads) -> the angle (rad, + = left): the inverse of a sphere's low-frequency law
    itd = KUHN (a / c) sin(theta) (Kuhn 1977), saturating at +-90 degrees."""
    return math.asin(max(-1.0, min(1.0, itd * C_SOUND / (KUHN * radius))))


def cochlea(x):
    """one ear's cochlea over a whole mono signal (pascals): band power (Pa^2) per 10 ms frame, [frames, 40], each frame the 25 ms
    ending at its hop (the same filterbank as the child's ears; the parent's ear for the child's voice reads it, P3)."""
    x = np.concatenate([np.zeros(WIN - HOP), np.asarray(x, np.float64)])
    nf = max(0, (len(x) - WIN) // HOP + 1)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(nf)[:, None]
    return (np.abs(np.fft.rfft(x[idx] * HANN, NFFT, axis=1)) ** 2) @ GTC.T
