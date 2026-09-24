"""the ears (docs/SIM_DESIGN.md 3.4 channel 2, 3.7's orienting; package P2): body/sim/ears.py. Run: python3 -m body.tests.test_sim_ears
Each test fails when the ears stop doing their job: the code's size and silence; the cochlea's calibration and its bands; the delay
lines' interaural delays against the head's geometry and the born lateral read over -80..+80 degrees; the level differences
between the ears growing with frequency; pressure falling as 1/r and held within NEAR; the child's own voice, from its head's
front, at +14 dB over the same sound from 1.5 m and read at the midline; streaming by ticks equal to one block; a moving source
without a click and its read following it; the onsets and the two event lines; save and restore; the cost. Numpy only, seconds."""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # this tree's body
from body.sim import ears as E  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

T = E.TICK
EAR_L, EAR_R, MOUTH = E.head_from_torso(np.array([0.0, 0.0, 0.3]), np.eye(3))   # the G1's head, facing +x, its left at +y
CENTRE = 0.5 * (EAR_L + EAR_R)


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


def test_cochlea_calibrated():
    got = []
    for f in (E.CF[np.argmin(np.abs(E.CF - f))] for f in (250.0, 1000.0, 4000.0)):   # a tone at a band's centre
        x = math.sqrt(2) * 0.02 * np.sin(2 * np.pi * f * np.arange(8 * T) / E.SR)   # 60 dB SPL at 1 m
        hs = run({"tone": (x, at(0, 1.0))}, 8)
        B = hs[-1].left ** 3 * E.E_CODE
        b = B.mean(0)
        k = int(b.argmax())
        near = int(np.argmin(np.abs(E.CF - f)))
        assert abs(k - near) <= 1, f"a {f:.0f} Hz tone peaks in the band at {E.CF[k]:.0f} Hz"
        tot = 10 * math.log10(b[max(0, k - 2):k + 3].max() / E.P_REF ** 2)
        path, phi, a = E.paths(at(0, 1.0), EAR_L, EAR_R)
        bb, aa = E.shadow_coeffs(phi[0], a)
        w = 2 * math.pi * f / E.SR
        sh = abs((bb[0] + bb[1] * np.exp(-1j * w)) / (aa[0] + aa[1] * np.exp(-1j * w)))
        want = 60 - 20 * math.log10(path[0]) + 20 * math.log10(sh)
        assert abs(tot - want) < 0.5, f"{f:.0f} Hz: the band reads {tot:.1f} dB SPL, the ear receives {want:.1f}"
        got.append(f"{f:.0f} Hz {tot:.1f} (at the ear {want:.1f})")
    print("2 the cochlea: a tone peaks in its own band and reads its level in dB SPL:", "; ".join(got))


def test_delay_lines_and_lateral_read():
    x = speechlike(10, seed=1)
    rows, errs = [], []
    for az in range(-80, 81, 20):
        hs = run({"v": (x, at(az))}, 10)
        reads = [math.degrees(h.lateral) for h in hs[1:] if h.lateral_weight > 0]
        assert len(reads) >= 8, f"{az}: the read had no weight on {9 - len(reads)} sounding ticks"
        m = float(np.mean(reads))
        want = 3 * E.HEAD_RADIUS / E.C_SOUND * math.sin(math.radians(az)) * E.SR   # a sphere's low-frequency delay (Kuhn)
        pk = hs[5].delay[np.isin(E.LOW, E.READ)].mean(0)
        lag = int(np.argmax(pk)) - E.LAGS
        assert abs(lag - want) <= 1.5, f"{az}: the delay lines peak at {lag} samples, the head gives {want:.1f}"
        errs += [abs(r - az) for r in reads]
        rows.append((az, m))
        assert (m > 0) == (az > 0) or az == 0, f"{az}: read at {m:.1f} (+ must be left)"
    ms = [m for _, m in rows]
    assert all(b > a for a, b in zip(ms, ms[1:])), f"the read is not monotone in azimuth: {rows}"
    assert np.mean(errs) < 8 and max(abs(m - az) for az, m in rows if abs(az) <= 60) < 8, rows
    print("3 the delay lines peak at the head's own delay; the born lateral read at 1.5 m:",
          ", ".join(f"{az:+d}->{m:+.1f}" for az, m in rows), f"(mean abs error {np.mean(errs):.1f} deg)")


def test_level_difference_grows_with_frequency():
    x = noise(8, seed=2)
    hs = run({"n": (x, at(90))}, 8)
    L = np.mean([h.left ** 3 for h in hs[2:]], axis=(0, 1))
    R = np.mean([h.right ** 3 for h in hs[2:]], axis=(0, 1))
    ild = 10 * np.log10(L / R)
    lo, hi = ild[E.CF < 300].mean(), ild[E.CF > 4000].mean()
    assert lo < 3 and hi > 10 and hi > lo + 8, f"the level difference {lo:.1f} dB below 300 Hz, {hi:.1f} dB above 4 kHz"
    front = run({"n": (x, at(0))}, 6)
    f_ild = 10 * np.log10(np.mean([h.left ** 3 for h in front[2:]]) / np.mean([h.right ** 3 for h in front[2:]]))
    assert abs(f_ild) < 1e-9, f_ild
    print(f"4 a source at the left ear: the left louder by {lo:.1f} dB below 300 Hz and {hi:.1f} dB above 4 kHz (the head's "
          f"shadow); straight ahead {f_ild:.1e} dB")


def test_distance():
    x = noise(6, seed=3)
    lv = {}
    for d in (0.1, 0.3, 1.0, 2.0, 4.0):
        hs = run({"n": (x, at(0, d))}, 6)
        lv[d] = np.mean([np.mean(h.levels["n"]) for h in hs[2:]])
        path = E.paths(at(0, d), EAR_L, EAR_R)[0]
        assert np.allclose(hs[-1].gains["n"], -20 * np.log10(np.maximum(path, E.NEAR))), (d, hs[-1].gains["n"])
    assert abs((lv[1.0] - lv[2.0]) - 6.02) < 0.2 and abs((lv[2.0] - lv[4.0]) - 6.02) < 0.2, lv
    assert abs(lv[0.1] - lv[0.3]) < 1.5 and lv[0.3] - lv[1.0] < 10.8, lv
    print("5 pressure falls as 1/r: " + ", ".join(f"{d} m {v:.1f} dB SPL" for d, v in lv.items()) + " (held within 0.3 m)")


def test_own_voice():
    x = speechlike(8, seed=4)
    own = run({"self": (x, MOUTH)}, 8)
    far = run({"parent": (x, at(0, 1.5))}, 8)
    lo = np.mean([np.mean(h.levels["self"]) for h in own[2:]])
    lf = np.mean([np.mean(h.levels["parent"]) for h in far[2:]])
    reads = [math.degrees(h.lateral) for h in own[1:] if h.lateral_weight > 0]
    assert 12.5 < lo - lf < 15.5, f"its own voice is {lo - lf:+.1f} dB over the same sound from 1.5 m"
    assert max(abs(r) for r in reads) < 3, reads
    L = np.mean([h.left for h in own[2:]])
    R = np.mean([h.right for h in own[2:]])
    assert abs(L - R) < 1e-9 * L, "its own voice is not heard equally at both ears"
    print(f"6 its own voice from the head's front: {lo - lf:+.1f} dB over the same sound from 1.5 m in front, read at "
          f"{np.mean(reads):+.1f} deg, equal at both ears")


def test_streaming_equals_one_block():
    x = noise(12, seed=5)
    p = at(35, 1.2, el=10)
    hs = run({"n": (x, p)}, 12)
    got = np.concatenate([h.ear_l for h in hs]), np.concatenate([h.ear_r for h in hs])
    path, phi, a = E.paths(p, EAR_L, EAR_R)
    buf = np.concatenate([np.zeros(E.HIST), x])
    for e in (0, 1):
        d = path[e] / E.C_SOUND * E.SR + E.LOOKAHEAD
        y = E.frac_read(buf, E.HIST + np.arange(len(x)) - d) / max(path[e], E.NEAR)
        b, aa = E.shadow_coeffs(phi[e], a)
        y = lfilter(b, aa, y)
        err = np.abs(got[e] - y).max() / np.abs(y).max()
        assert err < 1e-9, f"ear {e}: the ticks differ from one block by {err:.1e}"
    print("7 twelve ticks streamed equal one block computed at once (both ears, to 1e-9 of the peak: rounding only)")


def test_moving_source():
    x = math.sqrt(2) * 0.02 * np.sin(2 * np.pi * 440 * np.arange(21 * T) / E.SR)
    hs = run({"t": (x, lambda k: at(-60 + 6 * k))}, 21)
    y = np.concatenate([h.ear_l for h in hs])
    d2 = np.abs(np.diff(y, 2))
    edges = d2[[k * T - 2 for k in range(2, 20)]].max()
    inner = np.percentile(d2[2 * T:20 * T], 99.9)
    assert edges <= 1.05 * inner, f"a click at a tick's edge: {edges:.3g} against {inner:.3g} inside the ticks"
    reads = [math.degrees(h.lateral) for h in hs[2:]]
    assert reads[0] < -30 and reads[-1] > 30 and np.all(np.diff(reads) > -3), reads
    print(f"8 a tone moving from -60 to +60 degrees in 20 ticks: no click at the ticks' edges; the read goes "
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
    print(f"9 onsets: a sound after silence is heard on its first tick ({hs[0].onset.max():.0f} dB rise), not while it "
          f"holds; +20 dB mid-sound is an onset; the event lines fire left, right, or both at the midline")


def test_save_and_restore():
    x, y = speechlike(12, seed=9), noise(12, seed=10)
    srcs = {"parent": (x, lambda k: at(40 - 5 * k, 1.3)), "toy": (y, at(-50, 0.7, el=-20))}
    ears = E.Ears()
    run({k: (v[0][:6 * T], v[1]) for k, v in srcs.items()}, 6, ears)
    st = ears.state()
    rest = {k: (v[0][6 * T:], (lambda f: (lambda k: f(k + 6)))(v[1]) if callable(v[1]) else v[1]) for k, v in srcs.items()}
    a = run(rest, 6, ears)
    ears2 = E.Ears()
    ears2.load_state(st)
    b = run(rest, 6, ears2)
    for ha, hb in zip(a, b):
        assert np.array_equal(ha.code(), hb.code()) and ha.lateral == hb.lateral and ha.events == hb.events
    whole = run(srcs, 12)
    assert all(np.array_equal(p.code(), q.code()) for p, q in zip(whole[6:], a))
    print("10 save and restore: the ears continue bit for bit (two sources, one moving)")


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
    print(f"11 cost: three sources, both cochleas, the delay lines and the read: {med:.2f} ms a tick (median; this Mac, shared)")


TESTS = [test_code_and_silence, test_cochlea_calibrated, test_delay_lines_and_lateral_read, test_level_difference_grows_with_frequency,
         test_distance, test_own_voice, test_streaming_equals_one_block, test_moving_source, test_onsets_and_events,
         test_save_and_restore, test_cost]

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
