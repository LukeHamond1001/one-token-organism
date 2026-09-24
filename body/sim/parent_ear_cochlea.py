"""THE PARENT'S EAR'S PROTOTYPE COCHLEA, kept only for body/sim/parent_ear.py until P3v (docs/SIM_DESIGN.md 4.9, 11 day 5).
This is main's 1eb268b body/sim/ears.py, unchanged below this paragraph: the voice study's cochlea, through which every figure
of the parent's ear was measured (its filters about 1.6 ERB wide). The child's ears are body/sim/ears.py (the built P2: the
exact sphere, the calibrated 1.00 ERB cochlea, the delay lines); at the merge that file took the name, so this one moved here
and parent_ear.py imports it, so the prototype runs as it was measured. P3v moves the parent's ear onto ears.cochlea (the same
cochlea as the child's, as the design says), measures its figures again, and deletes this file. Not part of the body.

The two ears (docs/SIM_DESIGN.md section 3.4; from the 2026-09-24 prototype, "ear2", the ear chosen), fast form: the same born cochlea as 40 gammatone magnitude responses (ERB-spaced 80-7600 Hz, 4th order)
applied to a short-time spectrum (25 ms Hann, 10 ms hop), cube-root compressed band energy per frame; the brainstem:
per band the level difference, and per low band (CF < 1500 Hz) a Jeffress delay line computed as the band-limited
interaural cross-correlation over lags -8..+8 samples (from the cross-spectrum; a normalized coincidence count)."""
import json, numpy as np, scipy.signal as sg
SR = 16000; TICK = int(0.15 * SR); C = 343.0; HEAD_R = 0.065  # the G1's head is +-0.078 m wide (W5 sets it)
NFFT = 512; WIN = 400; HOP = 160; LAGS = 8
def erb_space(lo=80, hi=7600, n=40):
    q, bw = 9.26449, 24.7
    return (np.exp(np.linspace(q*np.log(1+lo/(q*bw)), q*np.log(1+hi/(q*bw)), n)/q)-1)*q*bw
CF = erb_space(); FREQS = np.fft.rfftfreq(NFFT, 1/SR)
ERB = 24.7 * (4.37 * CF / 1000 + 1)
GT = (1 + ((FREQS[None, :] - CF[:, None]) / (1.019 * ERB[:, None])) ** 2) ** (-2)   # 4th-order gammatone magnitude^2... (power)
GT /= GT.sum(1, keepdims=True)
LOW = np.where(CF < 1500)[0]
HANN = np.hanning(WIN).astype(np.float32)
def load(path_json):
    j = json.load(open(path_json)); x = np.fromfile(path_json[:-5] + '.f32', np.float32)
    y = sg.resample_poly(x, 320, 441) if int(j['sr']) == 22050 else x
    return y.astype(np.float32), [(m[0].strip('.?!,').lower(), m[1] / 4 / j['sr']) for m in j['marks']], len(y)/SR
def _shadow_delay(x, th, extra_delay, dist):
    w0 = C / HEAD_R; amin, thmin = 0.1, np.deg2rad(150)
    alpha = (1 + amin/2) + (1 - amin/2) * np.cos(th / thmin * np.pi)
    n = 1 << int(np.ceil(np.log2(len(x) + 400)))
    w = 2*np.pi*np.fft.rfftfreq(n, 1/SR)
    H = (1 + 1j*alpha*w/(2*w0)) / (1 + 1j*w/(2*w0)) * np.exp(-1j*w*extra_delay) / max(dist, 0.3)
    return np.fft.irfft(np.fft.rfft(x, n) * H, n)[:len(x)]
def spatialize(x, az, dist):
    """az: azimuth of the source in the head's frame (rad, + = left, 0 = straight ahead). Returns (left, right)."""
    out = []
    for ear_dir in (np.pi/2, -np.pi/2):
        th = np.arccos(np.clip(np.cos(az - ear_dir), -1, 1))
        d = -HEAD_R/C*np.cos(th) if th <= np.pi/2 else HEAD_R/C*(th - np.pi/2)
        out.append(_shadow_delay(x, th, d + HEAD_R/C, dist))     # + a/c keeps every delay causal; dist/c common to both ears omitted
    return out
def stft(x):
    nf = 1 + (len(x) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(nf)[:, None]
    return np.fft.rfft(x[idx] * HANN, NFFT, axis=1)
def ear_tick(xl, xr):
    """150 ms (+240 samples of the last tick) of each ear -> left frames [15,40], right frames [15,40], brainstem [21,17]."""
    XL, XR = stft(xl), stft(xr)
    fl = (np.abs(XL)**2 @ GT.T) ** (1/3); fr = (np.abs(XR)**2 @ GT.T) ** (1/3)
    S = (XL * np.conj(XR)).sum(0)                                   # cross-spectrum over the tick
    band = GT[LOW] * S[None, :]
    cc = np.fft.irfft(band, NFFT, axis=1)
    cc = np.concatenate([cc[:, -LAGS:], cc[:, :LAGS+1]], 1)          # lags -8..+8
    pl = (GT[LOW] * (np.abs(XL)**2).sum(0)).sum(1); pr = (GT[LOW] * (np.abs(XR)**2).sum(0)).sum(1)
    return fl, fr, cc / (np.sqrt(pl*pr)[:, None] / NFFT + 1e-12)
