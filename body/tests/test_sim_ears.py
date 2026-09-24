"""the ears (docs/SIM_DESIGN.md 3.4 channel 2, 3.7's orienting; package P2): body/sim/ears.py. Run: python3 -m body.tests.test_sim_ears
Each test fails when the ears stop doing their job: the code's size and silence; the sphere's series against scipy's spherical
Bessel functions; the ear signals' interaural delay, level difference and level against that exact rigid sphere, near and far,
from 250 Hz to the top band's centre (7.6 kHz), and the delay lines' lags covering the head's own delays; the delay lines' scale (a correlation at most 1, a coherent sound at
its band's loudness, the two halves of the code of one scale); the cochlea's calibration; the delay lines at the head's own
delay and the born lateral read over -80..+80 degrees; the level difference growing with frequency; the level following the
sphere all the way in, with no floor; the child's own voice at the sphere's level, read at the midline; streaming by ticks equal
to one block; a moving source without a click; the onsets and the two event lines; save and restore (a source leaving and coming
back); a scene with the child's tract heard the same in a single-thread process and restored there from a pickle; the ear sites
against the world's; the cost; a white source moved by eighths of a sample, every band's level and level difference holding still
(the P1-P2 verifier's blocker: the 16 kHz fractional delay line swung the top band 6 dB and its level difference 8-10 dB); the
ears' converter (the top bands' centre tones at their level, the fixed roll-off above 7.6 kHz). Numpy and scipy only, seconds."""
import ast
import math
import os
import sys
import time

import numpy as np
from scipy.special import spherical_jn, spherical_yn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # this tree's body
from body.sim import ears as E  # noqa: E402

T = E.TICK
EAR_L, EAR_R, MOUTH = E.head_from_torso(np.array([0.0, 0.0, 0.3]), np.eye(3))   # the G1's head, facing +x, its left at +y
CENTRE = 0.5 * (EAR_L + EAR_R)
A = E.HEAD_RADIUS


def at(az_deg, dist=1.5, el=0.0):
    a, e = math.radians(az_deg), math.radians(el)
    return CENTRE + dist * np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)])


def noise(n_ticks, seed=0, pa=0.02, band=6000.0):
    """white noise low-passed at `band` (a real source's treble; the delay line is flat to 6 kHz), RMS pa."""
    x = np.random.default_rng(seed).standard_normal(n_ticks * T + 64)
    b = np.sinc(2 * band / E.SR * (np.arange(65) - 32)) * np.hamming(65)
    x = np.convolve(x, b / b.sum(), "valid")[:n_ticks * T]
    return x / np.sqrt(np.mean(x ** 2)) * pa


def speechlike(n_ticks, seed=0, pa=0.02):
    """a harmonic voice at 200 Hz with a 4 Hz syllable envelope (an instrument's stand-in for speech: no synthesizer needed)."""
    t = np.arange(n_ticks * T) / E.SR
    f0 = 200 * (1 + 0.05 * np.sin(2 * np.pi * 1.3 * t))
    ph = 2 * np.pi * np.cumsum(f0) / E.SR
    x = sum(np.sin(k * ph) / k for k in range(1, 20))
    env = 0.5 - 0.5 * np.cos(2 * np.pi * 4 * t)
    x = x * env + 0.05 * np.random.default_rng(seed).standard_normal(len(t))
    return x / np.sqrt(np.mean(x ** 2)) * pa


def tone(f, n_ticks, pa=0.02):
    return math.sqrt(2) * pa * np.sin(2 * np.pi * f * np.arange(n_ticks * T) / E.SR)


def run(sources, n_ticks, ears=None, ear_l=EAR_L, ear_r=EAR_R):
    """sources: {name: (signal, position or a function of the tick)} -> the list of Heard."""
    ears = ears or E.Ears()
    out = []
    for k in range(n_ticks):
        src = {}
        for name, (x, p) in sources.items():
            src[name] = (x[k * T:(k + 1) * T], p(k) if callable(p) else p)
        out.append(ears.tick(ear_l, ear_r, src))
    return out


def exact(rho, f, theta, M=160):
    """an independent instrument: the rigid sphere's transfer summed directly from scipy's spherical Bessel functions, term by
    term until the terms fall below 1e-15 of the sum (for rho >= 1.3 that comes before the Neumann functions overflow; nearer the
    surface they overflow first, where ears.sphere_H's ratios, checked against this, are used instead)."""
    mu = 2 * math.pi * f * A / E.C_SOUND
    s = 0j
    for m in range(M):
        P = np.polynomial.legendre.legval(math.cos(theta), [0] * m + [1])
        h = spherical_jn(m, mu * rho) - 1j * spherical_yn(m, mu * rho)
        hp = spherical_jn(m, mu, True) - 1j * spherical_yn(m, mu, True)
        term = (2 * m + 1) * P * h / hp
        if m > mu * rho + 5 and (not np.isfinite(term) or abs(term) < 1e-15 * abs(s)):
            break
        assert np.isfinite(term), (rho, f, m)
        s += term
    return -(rho / mu) * np.exp(1j * mu * rho) * s


def at_f(y, f):
    """the complex amplitude of a tone at f in y (a whole number of its cycles is not needed: many cycles, one bin)."""
    return np.sum(y * np.exp(-2j * np.pi * f * np.arange(len(y)) / E.SR)) * 2 / len(y)


def test_code_and_silence():
    ears = E.Ears()
    h = ears.tick(EAR_L, EAR_R, {})
    c = h.code()
    assert c.shape == (E.N_CODE,) == (2 * 15 * 40 + 21 * 25,), c.shape
    assert c.dtype == np.float32 and not np.any(c), "silence is not all zeros"
    assert h.lateral == 0.0 and h.lateral_weight == 0.0 and not h.onset_heard and h.events == (False, False)
    h = ears.tick(EAR_L, EAR_R, {"toy": (np.zeros(T), at(30))})
    assert not np.any(h.code()) and h.levels["toy"][0] < -200
    assert abs(E.HEAD_RADIUS - 0.5 * np.linalg.norm(EAR_L - EAR_R)) < 1e-12 and E.LAGS == 12 and len(E.LOW) == 21
    print(f"1 the code: {E.N_CODE} numbers a tick (2 x 15 x 40 cochlea, 21 x 25 delay lines, lags +-{E.LAGS}); silence all zeros")


def test_sphere_series():
    worst = 0.0
    for rho, f, th in ((12.7, 250, 0.4), (3.8, 1000, 1.9), (1.52, 4000, 2.8), (19.0, 7000, 3.1), (2.5, 80, 1.57)):
        a = E.sphere_H(rho, [2 * math.pi * f * A / E.C_SOUND], [th])[0, 0]
        b = exact(rho, f, th)
        worst = max(worst, abs(a - b) / abs(b))
        assert abs(a - b) < 1e-6 * abs(b), (rho, f, th, a, b)
    far = E.sphere_H(1e4, 2 * np.pi * np.array([20.0, 16000.0]) * A / E.C_SOUND, [0.0])[:, 0]
    assert abs(abs(far[0]) - 1) < 0.01 and abs(abs(far[1]) - 2) < 0.05, far  # no shadow at 20 Hz; +6 dB facing it, high
    on = [E.sphere_H(r, [2 * math.pi * 500 * A / E.C_SOUND], [math.pi / 2])[0, 0] for r in (1.005, E.RHO_MIN, 1.1)]
    spread = max(abs(20 * math.log10(abs(h) / r / (abs(on[1]) / E.RHO_MIN))) for h, r in zip(on, (1.005, E.RHO_MIN, 1.1)))
    assert spread < 0.1, spread
    print(f"2 the sphere's series (ratios of Hankel functions) equals scipy's direct sum to {worst:.1e}; 20 Hz passes it "
          f"unshadowed, 16 kHz facing it +6 dB; a source on its surface: the level at 90 degrees within {spread:.2f} dB from "
          f"rho 1.005 to 1.1")


def test_against_the_exact_sphere():
    rows, worst = [], [0.0, 0.0, 0.0]
    for d in (1.5, 0.3, 0.12):
        for f in (250.0, 500.0, 4000.0, 7200.0, E.CF[-1]):              # up to the top band's centre (7.6 kHz)
            for az in (30, 60, 90, 150):
                hs = run({"t": (tone(f, 8), at(az, d))}, 8)
                yl = np.concatenate([h.ear_l for h in hs[3:]])
                yr = np.concatenate([h.ear_r for h in hs[3:]])
                pl, pr = at_f(yl, f), at_f(yr, f)
                hl, hr = exact(d / A, f, math.radians(abs(90 - az))), exact(d / A, f, math.radians(min(90 + az, 270 - az)))
                itd, itd_x = np.angle(pl / pr) / (2 * np.pi * f), np.angle(hl / hr) / (2 * np.pi * f)
                ild, ild_x = 20 * math.log10(abs(pl / pr)), 20 * math.log10(abs(hl / hr))
                lvl, lvl_x = 20 * math.log10(abs(pl) / (0.02 * math.sqrt(2))), 20 * math.log10(abs(hl) / d)
                lvr, lvr_x = 20 * math.log10(abs(pr) / (0.02 * math.sqrt(2))), 20 * math.log10(abs(hr) / d)
                e = (abs(np.angle((pl / pr) / (hl / hr))) / (2 * np.pi * f) * 1e6, abs(ild - ild_x),
                     max(abs(lvl - lvl_x), abs(lvr - lvr_x)))
                worst = [max(w, v) for w, v in zip(worst, e)]
                assert e[0] < 2 and e[1] < 0.1 and e[2] < 0.1, (d, f, az, itd * 1e6, itd_x * 1e6, ild, ild_x, lvl, lvl_x, lvr, lvr_x)
                if az == 90 and f < 1000:
                    rows.append((d, f, itd * 1e6))
                    assert abs(itd) * E.SR < E.LAGS, f"{d} m {f} Hz: the head's own delay {itd * E.SR:.2f} samples, lags +-{E.LAGS}"
    print(f"3 the ears against the exact sphere (scipy's series) at 0.12, 0.3 and 1.5 m, 250 Hz to 7.6 kHz (the top band's centre), "
          f"30-150 degrees: the interaural delay within {worst[0]:.2f} us, level difference within {worst[1]:.3f} dB, the level at "
          f"each ear within {worst[2]:.3f} dB; "
          f"at 90 degrees " + ", ".join(f"{d} m {f:.0f} Hz {t:.0f} us" for d, f, t in rows)
          + f" (inside the lags' {E.LAGS / E.SR * 1e6:.0f} us)")


def test_sub_sample_distance():
    """the P1-P2 verifier's blocker: with a fractional delay line at 16 kHz the top bands' level and level difference swung by
    several dB with where between two samples a source's delay fell (a 16-tap kernel is 0 to -8.3 dB at 7.6 kHz by the fraction).
    A source moved by eighths of a sample: every band's level, less the sphere's own change over the step, and its level difference
    must hold still."""
    x = np.random.default_rng(41).standard_normal(8 * T) * 0.02         # white to 8 kHz: the top bands at their fullest
    step = E.C_SOUND / E.SR / 8                                         # an eighth of a sample (2.7 mm)
    worst_l, worst_i, rows = np.zeros(E.NB), np.zeros(E.NB), []
    for az, d0 in ((40, 1.5), (-65, 0.5), (90, 0.25)):
        lv, il = [], []
        for k in range(9):
            d = d0 + k * step
            hs = run({"s": (x, at(az, d))}, 8)
            L = np.mean([h.left ** 3 for h in hs[3:]], axis=(0, 1))
            R = np.mean([h.right ** 3 for h in hs[3:]], axis=(0, 1))
            mu = 2 * np.pi * E.CF * A / E.C_SOUND
            hl = np.abs(E.sphere_H(d / A, mu, [math.radians(abs(90 - az))])[:, 0]) / d
            hr = np.abs(E.sphere_H(d / A, mu, [math.radians(min(90 + az, 270 - az))])[:, 0]) / d
            lv.append(10 * np.log10(L) - 20 * np.log10(hl))                 # less the sphere's own change
            il.append(10 * np.log10(L / R) - 20 * np.log10(hl / hr))
        sw_l, sw_i = np.ptp(lv, 0), np.ptp(il, 0)
        worst_l, worst_i = np.maximum(worst_l, sw_l), np.maximum(worst_i, sw_i)
        rows.append(f"{az:+d} deg {d0} m: top band {sw_l[-1]:.3f} / {sw_i[-1]:.3f} dB")
    assert worst_l.max() < 0.1 and worst_i.max() < 0.1, \
        f"moved by eighths of a sample, a band's level swings {worst_l.max():.2f} dB (band {E.CF[worst_l.argmax()]:.0f} Hz) " \
        f"and its level difference {worst_i.max():.2f} dB (band {E.CF[worst_i.argmax()]:.0f} Hz): {rows}"
    print(f"17 a white sound moved by eighths of a sample (1.5, 0.5, 0.25 m): every band's level, less the sphere's own change, "
          f"holds within {worst_l.max():.3f} dB and its level difference within {worst_i.max():.3f} dB (the top two bands "
          f"{worst_l[-2:].max():.3f} / {worst_i[-2:].max():.3f}; the 16 kHz delay line swung them by up to 6 / 10 dB):", "; ".join(rows))


def test_converter():
    """the ears' converter: flat to the top band's centre, the same fixed roll-off above it for every source and fraction, and nothing
    of the 32 kHz images left: a tone at each band's centre reads its level through the whole chain, a tone above 8 kHz's image
    region is gone, and the band levels of white noise from straight ahead match between the ears to rounding."""
    got = []
    for f in (E.CF[-3], E.CF[-2], E.CF[-1]):
        hs = run({"tone": (tone(f, 8), at(0, 1.0))}, 8)
        b = (hs[-1].left ** 3 * E.E_CODE).mean(0)
        tot = 10 * math.log10(b[-4:].max() / E.P_REF ** 2)
        want = 60 + 20 * math.log10(abs(exact(1.0 / A, f, math.pi / 2)))
        assert abs(tot - want) < 0.5, f"{f:.0f} Hz: the band reads {tot:.1f} dB SPL, the ear receives {want:.1f}"
        got.append(f"{f:.0f} Hz {tot:.1f} ({want:.1f})")
    resp = {}
    for f in (7700.0, 7800.0, 7900.0, 7990.0):
        hs = run({"t": (tone(f, 8), at(0, 1.0))}, 8)
        pl = at_f(np.concatenate([h.ear_l for h in hs[3:]]), f)
        resp[f] = 20 * math.log10(abs(pl) / (0.02 * math.sqrt(2)) / abs(exact(1.0 / A, f, math.pi / 2)))
    assert resp[7700.0] > -3 and resp[7990.0] < -50 and all(b < a for a, b in zip(list(resp.values()), list(resp.values())[1:])), resp
    print("18 the ears' converter: a tone at each of the top three bands' centres reads its level (dB SPL, the ear's in brackets):",
          "; ".join(got), "; above 7.6 kHz the fixed roll-off:", ", ".join(f"{f:.0f} Hz {v:+.1f} dB" for f, v in resp.items()))


def loudness(h):
    el = (h.left[:, E.LOW] ** 3).sum(0) * E.E_CODE
    er = (h.right[:, E.LOW] ** 3).sum(0) * E.E_CODE
    return np.cbrt(np.sqrt(el * er) / (E.NF * E.E_CODE))


def test_delay_line_scale():
    ratio = 0.0
    for sig, pos in ((speechlike(6, seed=21), at(40, 1.2)), (noise(6, seed=22), at(-70, 0.4)),
                     (noise(6, seed=23, pa=2e-4), at(10))):
        for h in run({"s": (sig, pos)}, 6)[1:]:
            ratio = max(ratio, float((np.abs(h.delay) / loudness(h)[:, None]).max()))
    assert ratio <= 1 + 1e-9, f"a delay line exceeds its band's loudness x 1: {ratio}"
    loud = run({"n": (noise(4, seed=24, pa=0.2), at(0, 1.0))}, 4)[-1]          # 80 dB SPL from straight ahead
    peak = loud.delay[:, E.LAGS] / loudness(loud)
    assert peak.min() > 0.999 and np.all(loud.delay.argmax(1) == E.LAGS), peak
    x = speechlike(10, seed=25)
    c = np.array([h.code() for h in run({"parent": (x, at(30, 1.5))}, 10)[2:]])
    n_coch = 2 * E.NF * E.NB
    r_coch, r_dl = float(np.sqrt((c[:, :n_coch] ** 2).mean())), float(np.sqrt((c[:, n_coch:] ** 2).mean()))
    assert 0.5 < r_dl / r_coch < 2, f"the delay lines' RMS {r_dl:.4f} against the cochlea's {r_coch:.4f} in one code"
    print(f"4 the delay lines' scale: a correlation at most 1 times the band's loudness (worst {ratio:.4f}); a coherent sound "
          f"from ahead peaks at lag 0 at {peak.min():.4f} of its loudness in every band; on speech at 1.5 m the code's delay "
          f"lines RMS {r_dl:.3f} against its cochlea's {r_coch:.3f}")


def test_cochlea_calibrated():
    got = []
    for f in (E.CF[np.argmin(np.abs(E.CF - f))] for f in (250.0, 1000.0, 4000.0)):   # a tone at a band's centre
        hs = run({"tone": (tone(f, 8), at(0, 1.0))}, 8)                            # 60 dB SPL at 1 m
        B = hs[-1].left ** 3 * E.E_CODE
        b = B.mean(0)
        k = int(b.argmax())
        near = int(np.argmin(np.abs(E.CF - f)))
        assert abs(k - near) <= 1, f"a {f:.0f} Hz tone peaks in the band at {E.CF[k]:.0f} Hz"
        tot = 10 * math.log10(b[max(0, k - 2):k + 3].max() / E.P_REF ** 2)
        want = 60 + 20 * math.log10(abs(exact(1.0 / A, f, math.pi / 2)))
        assert abs(tot - want) < 0.5, f"{f:.0f} Hz: the band reads {tot:.1f} dB SPL, the ear receives {want:.1f}"
        got.append(f"{f:.0f} Hz {tot:.1f} (at the ear {want:.1f})")
    print("5 the cochlea: a tone peaks in its own band and reads its level in dB SPL:", "; ".join(got))


def test_delay_lines_and_lateral_read():
    n, worst = noise(10, seed=31), 0.0
    for az in (-80, -40, 20, 60, 80):                                 # each read band's peak at the sphere's own delay there
        hs = run({"v": (n, at(az))}, 10)
        for i, b in enumerate(E.LOW):
            if b not in E.READ:
                continue
            pk = np.mean([h.delay[i] for h in hs[2:]], 0)
            k = int(np.argmax(pk))
            y0, y1, y2 = pk[k - 1], pk[k], pk[k + 1]
            lag = k - E.LAGS + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            hl, hr = exact(1.5 / A, E.CF[b], math.radians(90 - az)), exact(1.5 / A, E.CF[b], math.radians(90 + az))
            want = np.angle(hl / hr) / (2 * np.pi * E.CF[b]) * E.SR
            worst = max(worst, abs(lag - want))
            assert abs(lag - want) < 0.25, f"{az}: the {E.CF[b]:.0f} Hz delay line peaks at {lag:.2f} samples, the head: {want:.2f}"
    x = speechlike(10, seed=1)
    rows, errs = [], []
    for az in range(-90, 91, 10):
        hs = run({"v": (x, at(az))}, 10)
        reads = [math.degrees(h.lateral) for h in hs[1:] if h.lateral_weight > 0]
        assert len(reads) >= 8, f"{az}: the read had no weight on {9 - len(reads)} sounding ticks"
        m = float(np.mean(reads))
        if abs(az) <= 80:
            errs += [abs(r - az) for r in reads]
            assert abs(m - az) < 3, f"{az}: read at {m:.1f}"
        rows.append((az, m))
        assert (m > 0) == (az > 0) or az == 0, f"{az}: read at {m:.1f} (+ must be left)"
    ms = [m for _, m in rows]
    assert all(b > a for a, b in zip(ms, ms[1:])), f"the read is not monotone in azimuth: {rows}"
    assert abs(ms[0]) > 75 and abs(ms[1]) < 88, f"at 90 and 80 degrees the read is {ms[0]:.1f}, {ms[1]:.1f}"
    assert np.mean(errs) < 2, np.mean(errs)
    print(f"6 the delay lines peak at the sphere's own delay in each read band (noise, -80..+80 degrees: within {worst:.2f} "
          f"samples); the born lateral read on a voice at 1.5 m:", ", ".join(f"{az:+d}->{m:+.1f}" for az, m in rows),
          f"(mean abs error within 80 degrees {np.mean(errs):.2f} deg)")


def test_level_difference_grows_with_frequency():
    x = noise(8, seed=2)
    hs = run({"n": (x, at(90))}, 8)
    L = np.mean([h.left ** 3 for h in hs[2:]], axis=(0, 1))
    R = np.mean([h.right ** 3 for h in hs[2:]], axis=(0, 1))
    ild = 10 * np.log10(L / R)
    lo, hi = ild[E.CF < 300].mean(), ild[E.CF > 4000].mean()
    assert lo < 3 and hi > 5 and hi > lo + 4, f"the level difference {lo:.1f} dB below 300 Hz, {hi:.1f} dB above 4 kHz"
    front = run({"n": (x, at(0))}, 6)
    f_ild = 10 * np.log10(np.mean([h.left ** 3 for h in front[2:]]) / np.mean([h.right ** 3 for h in front[2:]]))
    assert abs(f_ild) < 1e-9, f_ild
    print(f"7 a source at the left ear: the left louder by {lo:.1f} dB below 300 Hz and {hi:.1f} dB above 4 kHz (the head's "
          f"shadow; the sphere's bright spot keeps the far ear's treble); straight ahead {f_ild:.1e} dB")


def test_distance_no_floor():
    lv, rows = {}, []
    for d in (4.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.15, 0.12):
        hs = run({"t": (tone(500.0, 6), at(0, d))}, 6)
        pl = at_f(np.concatenate([h.ear_l for h in hs[2:]]), 500.0)
        lv[d] = 20 * math.log10(abs(pl) / (0.02 * math.sqrt(2)))
        want = 20 * math.log10(abs(exact(d / A, 500.0, math.pi / 2)) / d)
        assert abs(lv[d] - want) < 0.1, f"{d} m: {lv[d]:.2f} dB re 1 m, the sphere gives {want:.2f}"
        path = E.paths(at(0, d), EAR_L, EAR_R)[0]
        assert np.allclose(hs[-1].gains["t"], -20 * np.log10(path)), (d, hs[-1].gains["t"])
    assert abs((lv[1.0] - lv[2.0]) - 6.02) < 0.1 and abs((lv[2.0] - lv[4.0]) - 6.02) < 0.1, lv
    assert lv[0.15] - lv[0.3] > 4.0, f"from 0.3 to 0.15 m the level rose {lv[0.15] - lv[0.3]:.1f} dB: a floor"
    print("8 the level follows the sphere all the way in (500 Hz, straight ahead, within 0.1 dB): " +
          ", ".join(f"{d} m {v:+.1f} dB" for d, v in lv.items()) + " re 1 m; no floor")


def test_own_voice():
    rho = E.RHO_MIN
    mu = 2 * math.pi * 500.0 * A / E.C_SOUND
    _, th, _, _ = E.source_geometry(MOUTH, EAR_L, EAR_R)
    assert np.allclose(th, math.pi / 2), th
    want = 20 * math.log10(abs(E.sphere_H(rho, [mu], [th[0]])[0, 0]) / (rho * A) / (abs(exact(1.5 / A, 500.0, math.pi / 2)) / 1.5))
    own = run({"self": (tone(500.0, 6), MOUTH)}, 6)
    far = run({"p": (tone(500.0, 6), at(0, 1.5))}, 6)
    got = 20 * math.log10(abs(at_f(np.concatenate([h.ear_l for h in own[2:]]), 500.0)) /
                          abs(at_f(np.concatenate([h.ear_l for h in far[2:]]), 500.0)))
    assert abs(got - want) < 0.1, f"its own voice at 500 Hz: {got:+.2f} dB over 1.5 m, the sphere gives {want:+.2f}"
    x = speechlike(8, seed=4)
    own = run({"self": (x, MOUTH)}, 8)
    far = run({"parent": (x, at(0, 1.5))}, 8)
    lo = np.mean([np.mean(h.levels["self"]) for h in own[2:]])
    lf = np.mean([np.mean(h.levels["parent"]) for h in far[2:]])
    reads = [math.degrees(h.lateral) for h in own[1:] if h.lateral_weight > 0]
    assert max(abs(r) for r in reads) < 1, reads
    L = np.mean([h.left for h in own[2:]])
    R = np.mean([h.right for h in own[2:]])
    assert abs(L - R) < 1e-9 * L, "its own voice is not heard equally at both ears"
    print(f"9 its own voice from its speaker on the head's front, the sphere's level (no floor): a 500 Hz tone {got:+.2f} dB "
          f"over the same from 1.5 m in front (the series {want:+.2f}); speech {lo - lf:+.1f} dB; read at {np.mean(reads):+.1f} "
          f"deg, equal at both ears")


def test_streaming_equals_one_block():
    x = noise(12, seed=5)
    p = at(35, 1.2, el=10)
    hs = run({"n": (x, p)}, 12)
    got = np.concatenate([h.ear_l for h in hs]), np.concatenate([h.ear_r for h in hs])
    rho, th, path, a = E.source_geometry(p, EAR_L, EAR_R)
    up = E.convert_block(np.concatenate([np.zeros(E.HIST), x]))          # the converter over the whole signal at once
    for e in (0, 1):
        d = path[e] / E.C_SOUND * E.SR + E.LOOKAHEAD
        y = E.frac_read(up, E.OS * (E.HIST + np.arange(len(x)) - d)) / path[e]
        y = np.convolve(y, E.residual_fir(rho, th[e]))[:len(x)]
        err = np.abs(got[e] - y).max() / np.abs(y).max()
        assert err < 1e-9, f"ear {e}: the ticks differ from one block by {err:.1e}"
    print("10 twelve ticks streamed equal one block computed at once (the converter, the delay line and the sphere over the whole "
          "signal; both ears, to 1e-9 of the peak: rounding only)")


def test_moving_source():
    x = tone(440.0, 21)
    hs = run({"t": (x, lambda k: at(-60 + 6 * k))}, 21)
    y = np.concatenate([h.ear_l for h in hs])
    d2 = np.abs(np.diff(y, 2))
    edges = d2[[k * T - 2 for k in range(2, 20)]].max()
    inner = np.percentile(d2[2 * T:20 * T], 99.9)
    assert edges <= 1.05 * inner, f"a click at a tick's edge: {edges:.3g} against {inner:.3g} inside the ticks"
    reads = [math.degrees(h.lateral) for h in hs[2:]]
    assert reads[0] < -30 and reads[-1] > 30 and np.all(np.diff(reads) > -3), reads
    print(f"11 a tone moving from -60 to +60 degrees in 20 ticks: no click at the ticks' edges; the read goes "
          f"{reads[0]:+.0f} -> {reads[-1]:+.0f}")


def test_onsets_and_events():
    x = np.concatenate([np.zeros(3 * T), noise(6, seed=6)])
    hs = run({"n": (x, at(60))}, 9)
    heard = [h.onset_heard for h in hs]
    assert heard[3] and not any(heard[:3]) and not any(heard[4:]), heard
    assert hs[3].events == (True, False) and hs[3].onset.min() >= E.ONSET_DB, (hs[3].events, hs[3].onset)
    hs = run({"n": (x, at(-60))}, 9)
    assert hs[3].events == (False, True)
    hs = run({"n": (x, at(0))}, 9)
    assert hs[3].events == (True, True)
    x2 = np.concatenate([noise(4, seed=7), noise(4, seed=8, pa=0.2)])
    hs = run({"n": (x2, at(20))}, 8)
    assert hs[4].onset_heard and not hs[6].onset_heard, [h.onset_heard for h in hs]
    print(f"12 onsets: a sound after silence is heard on its first tick, not while it holds; +20 dB mid-sound is an onset; the "
          f"event lines fire left, right, or both at the midline (+-{math.degrees(E.MIDLINE):.2f} deg)")


def test_save_and_restore():
    x, y = speechlike(16, seed=9), noise(16, seed=10)

    def feed(k0, n, ears):
        out = []
        for k in range(k0, k0 + n):
            s = {"parent": (x[k * T:(k + 1) * T], at(40 - 5 * k, 1.3))}
            if not 3 <= k < 9:                                      # the toy leaves (its source dropped) and comes back
                s["toy"] = (y[k * T:(k + 1) * T], at(-50, 0.7, el=-20))
            out.append(ears.tick(EAR_L, EAR_R, s))
        return out
    ears = E.Ears()
    feed(0, 7, ears)
    assert "toy" not in ears.src, "a source silent and gone for 4 ticks was kept"
    st = ears.state()
    a = feed(7, 9, ears)
    ears2 = E.Ears()
    ears2.load_state(st)
    b = feed(7, 9, ears2)
    for ha, hb in zip(a, b):
        assert np.array_equal(ha.code(), hb.code()) and ha.lateral == hb.lateral and ha.events == hb.events
    whole = feed(0, 16, E.Ears())
    assert all(np.array_equal(p.code(), q.code()) for p, q in zip(whole[7:], a))
    print("13 save and restore: the ears continue bit for bit (a moving voice; a toy leaving, dropped and coming back)")


def scene_digest(n=40, save_at=None, state_in=None, state_out=None, seed=4242):
    """a scene for exactness: the child's tract babbling at the mouth of a head that turns and moves, a parent walking past, a
    toy that leaves and comes back, a far source. -> sha256 of every tick's code and ear samples; state_out: pickle the ears'
    and the tract's state after `save_at` ticks; state_in: start from such a pickle at tick save_at."""
    import hashlib
    import pickle
    from body.sim import tract as TR
    from body.sim.voice.synth import PA_PER_UNIT
    ears, tr, k0 = E.Ears(), TR.Tract(seed), 0
    acts = np.random.default_rng(seed + 1)
    plan = [None if acts.random() < 0.5 else acts.integers(0, 5, 10) for _ in range(n)]
    if state_in:
        with open(state_in, "rb") as f:
            st = pickle.load(f)
        ears.load_state(st["ears"])
        tr.load_state(st["tract"])
        k0 = save_at
    x, y = speechlike(n, seed=seed), noise(n, seed=seed + 2, pa=0.01)
    h = hashlib.sha256()
    for k in range(k0, n):
        c, s_ = math.cos(0.07 * k), math.sin(0.07 * k)
        el, er, mo = E.head_from_torso(np.array([0.01 * k, 0.0, 0.3]), np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]]))
        src = {"self": (tr.tick(plan[k]) * PA_PER_UNIT, mo), "parent": (x[k * T:(k + 1) * T], at(60 - 4 * k, 1.4)),
               "far": (y[k * T:(k + 1) * T], at(170, 6.0))}
        if not 10 <= k < 16:
            src["toy"] = (y[k * T:(k + 1) * T] * 3, at(-30, 0.35, -40))
        hd = ears.tick(el, er, src)
        h.update(hd.code().tobytes() + hd.ear_l.tobytes() + hd.ear_r.tobytes() + np.float64(hd.lateral).tobytes())
        if state_out and k + 1 == save_at:
            with open(state_out, "wb") as f:
                pickle.dump(dict(ears=ears.state(), tract=tr.state()), f)
            h = hashlib.sha256()                                    # the digest of the ticks after the save
    return h.hexdigest()


def test_exact_across_processes():
    import subprocess
    import tempfile
    tmp = tempfile.mkdtemp(prefix="ears_exact_")
    try:
        st = os.path.join(tmp, "state.pkl")
        whole_tail = scene_digest(save_at=17, state_out=st)
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        code = ("import sys; sys.path.insert(0, %r); from body.tests import test_sim_ears as t; "
                "print(t.scene_digest(save_at=17, state_out=%r)); print(t.scene_digest(save_at=17, state_in=%r))"
                % (root, os.path.join(tmp, "state2.pkl"), st))
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr[-2000:]
        single, restored = out.stdout.split()
        assert single == whole_tail, "a single-thread process heard the scene differently"
        assert restored == whole_tail, "the ears and the tract restored in another process did not continue exactly"
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"14 exact across processes: a babbling tract at the mouth of a turning, moving head, a walking parent, a toy leaving "
          f"and coming back and a far source, 40 ticks: the same digest threaded and single-threaded, and restored from a pickle "
          f"at tick 17 in the other process ({whole_tail[:12]})")


def test_ear_sites_are_the_worlds():
    f = os.path.join(os.path.dirname(E.__file__), "g1scene.py")
    if not os.path.exists(f):
        print("15 the world's g1scene.py is not in this tree: the ear sites' check waits for the merge")
        return
    vals = {}
    for node in ast.parse(open(f).read()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") in ("EAR_Y", "EAR_XZ"):
            vals[node.targets[0].id] = ast.literal_eval(node.value)
    assert set(vals) == {"EAR_Y", "EAR_XZ"}, vals
    want = {"L": [vals["EAR_XZ"][0], vals["EAR_Y"], vals["EAR_XZ"][1]], "R": [vals["EAR_XZ"][0], -vals["EAR_Y"], vals["EAR_XZ"][1]]}
    for k in "LR":
        assert np.allclose(E.EAR_SITES[k], want[k]), (k, E.EAR_SITES[k], want[k])
    print(f"15 the ear sites equal the world's (g1scene EAR_Y {vals['EAR_Y']}, EAR_XZ {vals['EAR_XZ']})")


def test_cost():
    x, y, z = speechlike(40, seed=11), noise(40, seed=12, pa=0.01), speechlike(40, seed=13, pa=0.05)
    ears = E.Ears()
    tt = []
    for k in range(40):
        s = {"parent": (x[k * T:(k + 1) * T], at(30, 1.2)), "toy": (y[k * T:(k + 1) * T], at(-40, 0.6, -30)),
             "self": (z[k * T:(k + 1) * T], MOUTH)}
        t0 = time.perf_counter()
        ears.tick(EAR_L, EAR_R, s)
        tt.append(time.perf_counter() - t0)
    med = 1e3 * float(np.median(tt[5:]))
    assert med < 10, f"the ears take {med:.2f} ms a tick with three sources"
    ears, tm = E.Ears(), []
    for k in range(40):
        c, s_ = math.cos(0.05 * k), math.sin(0.05 * k)                 # the head turning 19 degrees a second
        el, er, mo = E.head_from_torso(np.array([0.0, 0.0, 0.3]), np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]]))
        s = {"parent": (x[k * T:(k + 1) * T], at(30 + 2 * k, 1.2)), "toy": (y[k * T:(k + 1) * T], at(-40, 0.6, -30)),
             "self": (z[k * T:(k + 1) * T], mo), "far": (y[k * T:(k + 1) * T], at(120, 4.0)),
             "rattle": (z[k * T:(k + 1) * T], at(-100 + k, 0.4))}
        t0 = time.perf_counter()
        ears.tick(el, er, s)
        tm.append(time.perf_counter() - t0)
    print(f"16 cost: three still sources, both cochleas, the delay lines and the read: {med:.2f} ms a tick (median); a turning "
          f"head with five sources {1e3 * float(np.median(tm[5:])):.2f} ms (this Mac, shared)")


TESTS = [test_code_and_silence, test_sphere_series, test_against_the_exact_sphere, test_delay_line_scale, test_cochlea_calibrated,
         test_delay_lines_and_lateral_read, test_level_difference_grows_with_frequency, test_distance_no_floor, test_own_voice,
         test_streaming_equals_one_block, test_moving_source, test_onsets_and_events, test_save_and_restore,
         test_exact_across_processes, test_ear_sites_are_the_worlds, test_cost, test_sub_sample_distance, test_converter]

if __name__ == "__main__":
    t0 = time.time()
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print("FAIL", t.__name__, ":", e)
        except Exception as e:
            failed += 1
            print("ERROR", t.__name__, ":", type(e).__name__, str(e)[:300])
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if failed else 0)
