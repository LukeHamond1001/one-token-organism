"""THE VOICES AND THE EARS, MEASURED (docs/SIM_DESIGN.md 4.4, 4.9, 3.4; packages P1 and P2): an instrument, never part of a body.
Prints what the commit reports; nothing it measures sets a constant of the body (the one calibration it gives, the engine's speech
RMS behind SYNTH_RMS, is a level, written into body/sim/voice/synth.py by hand and re-checked by the tests).

  voice      the engine bit for bit across two server processes; its wall time a line (warm); the plain register's speech level;
             words a second, seconds a line, F0 and level per register over every line; the words channel at one symbol a tick
             (tokens on their word's end tick or late); the cache's size for the lines; the focus word (each line's last)
             emphasized, by the line's ending (". ? !"), in the new-word register and in the plain one: its length and F0 against
             the same line unemphasized and against the plain line, and the lines' words a second (pooled, per line, and over the
             design's variation set for a new word: C25); the talk-over stop cut at every tick of every line, plain and as
             new-word lines (ticks to the stop, words broken off by the 3-tick cap, and the cuts the uncapped finish would have
             taken past 3 ticks)
  tract      its cost a tick (a held vowel, a glide, a hiss, rest; best and median of repeats on this shared Mac); its /a/ against
             the parent's loud frames
  ears       the born lateral read on the parent's voice from -90 to +90 degrees at 1.5 m; the code's two halves' scale on it; the
             child's own voice (level over the same sound from 1.5 m, its lateral read); the cost a tick with three sources;
             the parent's speech moved by eighths of a sample (the top bands' level and level difference must hold still)

Run: nice -n 19 python3 tools/sim_voice_check.py [--lines FILE] [--cache DIR] [--n N] [--growth]
  --lines  one line a line (default: 16 lines written here; the commits used the all-out study's 331 birth template lines,
           $S/allout/lang/all_lines.txt: 275 end in ".", 37 in "?", 19 in "!"; 314 distinct)
  --cache  the voice cache to use (default: a temporary folder, removed after)
  --growth only the growth words' introductions (P3's INTRO frames), each set's words a second in the new-word register (C25)
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np
from scipy.signal import medfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from body.sim import ears as E  # noqa: E402
from body.sim.lang import lexicon as LX  # noqa: E402
from body.sim.voice import synth as V  # noqa: E402
from body.sim.voice.playback import CUT_MAX, Utterance  # noqa: E402
from body.sim import tract as T  # noqa: E402

FALLBACK = ["look at the duck.", "where is the ball?", "here is your bottle.", "hi pip. mama is here.", "yes. the duck!",
            "the ball is on the mat.", "see? a drum.", "give me the cup.", "your hand.", "uh oh. the car is down.",
            "night night pip.", "what is this?", "roll the ball.", "sit. you sit.", "peekaboo!", "bye bye pip."]


def f0_track(x, sr=V.SR, lo=100.0, hi=700.0, frame=640, hop=160):
    """an instrument: each 40 ms frame's F0 by normalized autocorrelation (0 where the peak is under 0.5)."""
    out = []
    for s in range(0, len(x) - frame, hop):
        f = x[s:s + frame].astype(np.float64)
        f = f - f.mean()
        if np.dot(f, f) < 1e-9:
            out.append(0.0)
            continue
        ac = np.correlate(f, f, "full")[frame - 1:]
        ac = ac / ac[0]
        a, b = int(sr / hi), int(sr / lo)
        k = a + int(np.argmax(ac[a:b]))
        out.append(sr / k if ac[k] > 0.5 else 0.0)
    return np.array(out)


def f0_word(pcm):
    """an instrument: a word's F0 over its voiced frames, without octave errors (frames more than a factor 1.6 from the word's
    median are dropped: a stop's burst can read as a periodic run near 520 Hz), the median and the peak (the largest of the
    frames median-filtered over 3); nan where none is voiced."""
    v = f0_track(pcm)
    v = v[v > 0]
    if not len(v):
        return float("nan"), float("nan")
    m = np.median(v)
    v = v[(v < 1.6 * m) & (v > m / 1.6)]
    return float(np.median(v)), float(np.max(medfilt(v, 3) if len(v) >= 3 else v))


def focus_of(line):
    """the line's focus word: its last (the design's 4.5: the new word placed last; Fernald and Mazzie 1991)."""
    return "".join(ch if ch.isalpha() else " " for ch in line.lower()).split()[-1]


ENDINGS = (".", "?", "!")


def emphasis_by_ending(lines, cache, server, reg="new_word"):
    """the focus word's emphasis on every line, by the line's final punctuation: the focus word (the line's last) emphasized in
    the register (the parent's new-word line: the new_word register) against the same line in the same register unemphasized
    (the emphasis alone; in the new_word register the voice never says it, request() refuses it, so it is made here from
    line_ssml) and against the plain line (the plain register); the words a second of the emphasized lines over their spoken
    span (the design's measure: C25); and, for the new word, the design's variation set (4.5: "a duck." / "the duck!" / "you see
    the duck?") for each toy."""
    p, r, _ = V.REGISTERS[reg]
    rows = {e: [] for e in ENDINGS}
    for ln in lines:
        fw = focus_of(ln)
        e = cache.clip(ln, reg, emphasis=fw)
        pl = cache.clip(ln, "plain")
        q = V.line_ssml(ln, p, r)
        x, sr, marks, _ = server.synth(q, rate=V.ENGINE_RATE, pitch=1.0, ssml=True)
        upcm = V.to_16k(x, sr)
        uw = V.words_of(upcm, sr, marks, q)
        m = {}
        for name, pcm, ws in (("e", e.pcm, e.words), ("u", upcm, uw), ("p", pl.pcm, pl.words)):
            _, on, end = [w for w in ws if w[0] == fw][-1]
            m[name] = ((end - on) / V.SR,) + f0_word(pcm[on:end])
        rows[ln.rstrip()[-1]].append((ln, m, len(e.words), (e.words[-1][2] - e.words[0][1]) / V.SR))
    print(f"the focus word emphasized in the {reg} register (pitch {p}, rate {r}; SSML pitch {V.EMPHASIS[0]} on the line's, rate "
          f"{V.EMPHASIS[1]} x the line's), by the line's ending; the focus word is each line's last:")
    for end in ENDINGS:
        rs = rows[end]
        if not rs:
            continue
        a = lambda k, i: np.array([row[1][k][i] for row in rs])          # noqa: E731
        lu, lp = a("e", 0) / a("u", 0), a("e", 0) / a("p", 0)
        fu, fp, kp = a("e", 1) / a("u", 1), a("e", 1) / a("p", 1), a("e", 2) / a("p", 2)
        wps = np.array([row[2] / row[3] for row in rs])
        pooled = sum(row[2] for row in rs) / sum(row[3] for row in rs)
        print(f"  '{end}' {len(rs)} lines ({len(set(row[0] for row in rs))} distinct): the focus word's length x{np.median(lu):.2f} "
              f"(min {lu.min():.2f}, max {lu.max():.2f}) over the same line unemphasized, x{np.median(lp):.2f} (min {lp.min():.2f}) "
              f"over the plain line ({1e3 * np.median(a('e', 0)):.0f} ms against {1e3 * np.median(a('u', 0)):.0f} and "
              f"{1e3 * np.median(a('p', 0)):.0f}); its F0 x{np.nanmedian(fu):.2f} (min {np.nanmin(fu):.2f}) and x{np.nanmedian(fp):.2f} "
              f"(min {np.nanmin(fp):.2f}) ({np.nanmedian(a('e', 1)):.0f} Hz against {np.nanmedian(a('u', 1)):.0f} and "
              f"{np.nanmedian(a('p', 1)):.0f}), its peak x{np.nanmedian(kp):.2f} (min {np.nanmin(kp):.2f}) over the plain line's; "
              f"{pooled:.2f} words a second pooled, {wps.mean():.2f} a line on average (max {wps.max():.2f}; {np.sum(wps > 3)} of "
              f"{len(rs)} lines over 3)")
    if reg != "new_word":
        return
    sets = []
    for toy in ("ball", "duck", "block", "cup", "bear", "car", "drum", "bottle"):
        cs = [cache.clip(ln, "new_word", emphasis=toy) for ln in (f"a {toy}.", f"the {toy}!", f"you see the {toy}?")]
        sets.append(sum(len(c.words) for c in cs) / sum((c.words[-1][2] - c.words[0][1]) / V.SR for c in cs))
    print(f"  the variation set for a new word (\"a X.\" / \"the X!\" / \"you see the X?\", the 8 toys): {np.mean(sets):.2f} words a "
          f"second on average, {min(sets):.2f}-{max(sets):.2f}")


def sounding_rms(pcm):
    e = V.frame_rms(pcm)
    act = e > e.max() * 10 ** (V.WORD_END_DB / 20)
    return float(np.sqrt((e[act] ** 2).mean()))


def voice(lines, cache, n_det):
    print("== the parent's voice")
    s1, s2 = V.SynthServer(), V.SynthServer()
    info = s1.info()
    print(f"engine: {info['name']} ({info['identifier']}), quality {info['quality']}, {info['os']}")
    same, walls = 0, []
    for ln in lines[:n_det]:
        q = V.request(ln)[1]["ssml"]
        a = s1.synth(q, rate=V.ENGINE_RATE, pitch=1.0, ssml=True)
        b = s2.synth(q, rate=V.ENGINE_RATE, pitch=1.0, ssml=True)
        same += int(np.array_equal(a[0], b[0]) and a[2] == b[2])
        walls += [a[3], b[3]]
    s2.close()
    print(f"bit for bit across two server processes: {same}/{min(n_det, len(lines))} lines; "
          f"wall a line {1e3 * np.median(walls[4:]):.0f} ms median, {1e3 * np.percentile(walls[4:], 95):.0f} ms 95th pct "
          f"(first, cold: {1e3 * walls[0]:.0f} ms)")
    t0 = time.perf_counter()
    clips = [cache.clip(ln, "plain") for ln in lines]
    wall = time.perf_counter() - t0
    size = cache.kept_size + cache.size
    rms = np.array([sounding_rms(c.pcm) for c in clips])
    print(f"{len(lines)} lines synthesized and cached in {wall:.1f} s; the cache {size / 2 ** 20:.1f} MB (16 kHz int16 + meta)")
    print(f"the plain register's sounding-frame RMS: mean {rms.mean():.4f}, median {np.median(rms):.4f}, sd {rms.std():.4f} "
          f"engine units (SYNTH_RMS in synth.py: {V.SYNTH_RMS}); as written, plain speech at 1 m is "
          f"{20 * math.log10(np.sqrt((rms ** 2).mean()) * V.PA_PER_UNIT / E.P_REF):.1f} dB SPL")
    print(f"the registers over the {len(lines)} lines (every ending; a focus word, where emphasized, the line's last):")
    for reg in ("plain", "question", "approval", "comfort", "calling", "no", "new_word"):
        for emph in ((False, True) if reg == "plain" else (True,) if reg == "new_word" else (False,)):   # new words: always
            cs = [cache.clip(ln, reg, emphasis=(focus_of(ln) if emph else None)) for ln in lines]
            nw = sum(len(c.words) for c in cs)
            span = np.array([(c.words[-1][2] - c.words[0][1]) / V.SR for c in cs])
            f0 = np.concatenate([f0_track(c.pcm) for c in cs])
            f0 = f0[f0 > 0]
            lvl = np.array([sounding_rms(c.pcm) for c in cs]) * V.PA_PER_UNIT * 10 ** (V.REGISTERS[reg][2] / 20)
            print(f"  {reg + (' +emph' if emph else ''):14s} pitch {V.REGISTERS[reg][0]:.2f} rate {V.REGISTERS[reg][1]:.2f}: "
                  f"{nw / span.sum():.2f} words a second over the spoken span, {span.mean():.2f} s spoken a line "
                  f"({span.mean() / 0.15:.1f} ticks); F0 median {np.median(f0):.0f} Hz; "
                  f"level {20 * math.log10(np.sqrt((lvl ** 2).mean()) / E.P_REF):.1f} dB SPL at 1 m")
    # the words channel at one symbol a tick
    w = LX.Words()
    t = 0
    for c in clips:
        w.push(c.words, t, len(c.pcm))
        end = t + int(math.ceil(len(c.pcm) / 2400)) + 6
        while t < end or w.queue:
            w.tick(t)
            t += 1
    tok = np.array([late for sym, late in w.late if sym < LX.N_BIRTH])
    let = np.array([late for sym, late in w.late if LX.N_BIRTH <= sym < LX.N_BIRTH + 26])
    print(f"the words channel: {len(tok)} tokens, {100 * np.mean(tok == 0):.1f}% on their word's end tick, "
          f"{100 * np.mean(tok == 1):.1f}% one tick late, {100 * np.mean(tok >= 2):.1f}% two or more"
          + (f"; {len(let)} letters, {100 * np.mean(let == 0):.0f}% on time" if len(let) else ""))
    emphasis_by_ending(lines, cache, s1, "new_word")
    emphasis_by_ending(lines, cache, s1, "plain")
    # the talk-over stop at every tick of every line
    left, broken, over, n_words = [], 0, 0, LX.Words()
    for c in clips:
        for at in range(1, int(math.ceil(len(c.pcm) / 2400))):
            u = Utterance(c, 0)
            for t in range(at):
                u.tick(t, n_words)
            cur = [end for _, on, end in c.words if on < u.pos < end]
            over += bool(cur) and int(math.ceil((cur[0] - u.pos) / 2400)) > CUT_MAX
            left.append(u.cut(n_words))
            broken += u.broken is not None
    left = np.array(left)
    print(f"the talk-over stop cut at every tick of the {len(clips)} lines ({len(left)} cuts): ticks to the stop "
          + ", ".join(f"{k}: {100 * np.mean(left == k):.1f}%" for k in range(0, int(left.max()) + 1))
          + f"; {broken} ({100 * broken / len(left):.2f}%) broke a word off at {CUT_MAX} ticks (the uncapped finish: {over} cuts "
          f"past {CUT_MAX} ticks)")
    # the same on the new-word lines (the new word emphasized): the uncapped finish, and the cap
    over_nw, broken_nw, n_nw, need_nw = 0, 0, 0, 0
    for ln in lines:
        c = cache.clip(ln, "new_word", emphasis=focus_of(ln))
        for at in range(1, int(math.ceil(len(c.pcm) / 2400))):
            u = Utterance(c, 0)
            for t in range(at):
                u.tick(t, n_words)
            cur = [end for _, on, end in c.words if on < u.pos < end]
            need = int(math.ceil((cur[0] - u.pos) / 2400)) if cur else 0
            over_nw += need > CUT_MAX
            need_nw = max(need_nw, need)
            u.cut(n_words)
            broken_nw += u.broken is not None
            n_nw += 1
    print(f"  on the {len(lines)} new-word lines ({n_nw} cuts): {broken_nw} ({100 * broken_nw / n_nw:.2f}%) broke the word off at "
          f"{CUT_MAX} ticks (the uncapped finish: {over_nw} cuts past {CUT_MAX} ticks, at most {need_nw})")
    s1.close()
    return clips, rms


def _bench(fn, reps=7, n=20):
    runs = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        runs.append((time.perf_counter() - t) / n * 1e3)
    return min(runs), float(np.median(runs))


def tract(parent_rms):
    print("== the child's vocal tract")

    def state(post):
        tr = T.Tract(1)
        x = T.NEUTRAL.copy()
        for k, v in post.items():
            x[T.NAMES.index(k)] = v
        tr.x, tr.target = x.copy(), x.copy()
        for _ in range(3):
            tr.tick(np.full(10, 2))
        return tr
    A = dict(lungs=0.6, glottis=0.7, velum=0.0, jaw=0.75, tongue_front=0.0, tongue_height=0.0, lips=0.6)
    S = dict(lungs=0.6, glottis=0.1, velum=0.0, jaw=0.15, tongue_front=0.7, tongue_height=0.5, tongue_tip=0.8, lips=0.6)
    glide = np.array([2, 2, 3, 0, 4, 4, 2, 2, 2, 2])
    for name, post, act in (("a vowel, held", A, np.full(10, 2)), ("a vowel, gliding", A, glide),
                            ("a hiss (the noise path)", S, np.full(10, 2)), ("rest", None, None)):
        tr0 = state(post) if post else T.Tract(1)
        st = tr0.state()

        def one():
            tr0.load_state(st)
            tr0.tick(act)
        b, m = _bench(one)
        print(f"  {name:24s} best {b:5.2f} ms, median {m:5.2f} ms a tick")
    tr = state(dict(A, jaw=0.8, lips=0.9, rounding=0.2, pitch=0.35))
    y = np.concatenate([tr.tick(np.full(10, 2)) for _ in range(4)])[2 * 2400:]
    a_rms = float(np.sqrt((y ** 2).mean()))
    print(f"  its steady /a/: {20 * math.log10(a_rms * V.PA_PER_UNIT / E.P_REF):.1f} dB SPL at 1 m "
          f"({20 * math.log10(a_rms / np.sqrt((parent_rms ** 2).mean())):+.1f} dB re the parent's plain sounding frames)")


def ears(clips):
    print("== the ears")
    print(f"channel: 2 x {E.NF} x {E.NB} cochlea + {len(E.LOW)} x {2 * E.LAGS + 1} delay lines = {E.N_CODE} numbers a tick; "
          f"the lateral read's bands: CF {E.CF[E.READ].min():.0f}-{E.CF[E.READ].max():.0f} Hz ({len(E.READ)})")
    eL, eR, mouth = E.head_from_torso(np.array([0.0, 0.0, 0.6]), np.eye(3))
    centre = 0.5 * (eL + eR)
    res, tt = {}, []
    for az in range(-90, 91, 10):
        a = math.radians(az)
        src = centre + 1.5 * np.array([math.cos(a), math.sin(a), 0.0])
        got = []
        for c in clips[::max(1, len(clips) // 8)][:8]:
            ea = E.Ears()
            x = c.pa()
            for k in range(len(x) // 2400):
                t0 = time.perf_counter()
                h = ea.tick(eL, eR, {"parent": (x[k * 2400:(k + 1) * 2400], src)})
                tt.append(time.perf_counter() - t0)
                if h.lateral_weight > 0:
                    got.append(math.degrees(h.lateral))
        res[az] = np.array(got)
    err = np.concatenate([np.abs(v - az) for az, v in res.items() if abs(az) <= 80])
    print("the born lateral read on the parent's voice at 1.5 m (mean over sounding ticks): " +
          ", ".join(f"{az:+d}: {v.mean():+.1f}" for az, v in res.items()))
    print(f"  absolute error within 80 degrees: mean {err.mean():.1f}, median {np.median(err):.1f}, 90th pct "
          f"{np.percentile(err, 90):.1f} degrees over {len(err)} ticks; one source {1e3 * np.median(tt):.2f} ms a tick median")
    # the code's two halves on the parent's speech at 1.5 m, 30 degrees
    codes = []
    src = centre + 1.5 * np.array([math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0])
    for c in clips[:20]:
        ea, x = E.Ears(), c.pa()
        for k in range(len(x) // 2400):
            h = ea.tick(eL, eR, {"parent": (x[k * 2400:(k + 1) * 2400], src)})
            if k:
                codes.append(h.code())
    codes = np.array(codes)
    nco = 2 * E.NF * E.NB
    print(f"the code on the parent's speech (20 lines, 1.5 m, 30 degrees): cochlea RMS "
          f"{np.sqrt((codes[:, :nco] ** 2).mean()):.3f}, delay lines RMS {np.sqrt((codes[:, nco:] ** 2).mean()):.3f} "
          f"(largest {np.abs(codes[:, nco:]).max():.3f})")
    # the P1-P2 verifier's blocker, on the parent's real speech: moved by eighths of a sample at 1.5 m (40 degrees), the top two
    # bands' level and level difference (the 16 kHz delay line swung them 6.4 and 10.4 dB; physics, about 0.1 and 0)
    x = np.concatenate([c.pa() for c in clips[:6]])
    n = len(x) // 2400
    lvl, ild = [], []
    for k in range(9):
        a, d = math.radians(40), 1.5 + k * E.C_SOUND / E.SR / 8
        src = centre + d * np.array([math.cos(a), math.sin(a), 0.0])
        ea, L, R = E.Ears(), 0.0, 0.0
        for j in range(n):
            h = ea.tick(eL, eR, {"parent": (x[j * 2400:(j + 1) * 2400], src)})
            L, R = L + (h.left[:, -2:] ** 3).sum(0), R + (h.right[:, -2:] ** 3).sum(0)
        lvl.append(10 * np.log10(L) + 20 * math.log10(d))
        ild.append(10 * np.log10(L / R))
    print(f"the parent's speech (6 lines) moved by eighths of a sample at 1.5 m, 40 degrees: the top two bands' level (less the "
          f"spreading) swings {np.ptp(lvl, 0)[0]:.3f} / {np.ptp(lvl, 0)[1]:.3f} dB, their level difference "
          f"{np.ptp(ild, 0)[0]:.3f} / {np.ptp(ild, 0)[1]:.3f} dB")
    # the child's own voice against the parent at 1.5 m, the same sound
    tr = T.Tract(1)
    x = T.NEUTRAL.copy()
    for k, v in dict(lungs=0.6, glottis=0.7, velum=0.0, jaw=0.8, lips=0.9, rounding=0.2, tongue_front=0.0,
                     tongue_height=0.0).items():
        x[T.NAMES.index(k)] = v
    tr.x, tr.target = x.copy(), x.copy()
    y = np.concatenate([tr.tick(np.full(10, 2)) for _ in range(8)]) * V.PA_PER_UNIT
    lv = {}
    for name, pos in (("self", mouth), ("parent", centre + np.array([1.5, 0.0, 0.0]))):
        ea = E.Ears()
        for k in range(8):
            h = ea.tick(eL, eR, {name: (y[k * 2400:(k + 1) * 2400], pos)})
        lv[name] = (np.mean(h.levels[name]), math.degrees(h.lateral))
    print(f"its own voice (a held /a/ from the head's front) at its ears: {lv['self'][0]:.1f} dB SPL, "
          f"{lv['self'][0] - lv['parent'][0]:+.1f} dB over the same sound from 1.5 m in front; its lateral read {lv['self'][1]:+.1f} deg")
    # three sources at once: the parent, a toy's noise, its own voice
    rng = np.random.default_rng(0)
    toy = rng.standard_normal(2400 * 20) * 0.02
    par = np.concatenate([c.pa() for c in clips[:6]])[:2400 * 20]
    ea = E.Ears()
    tt = []
    for k in range(20):
        srcs = {"parent": (par[k * 2400:(k + 1) * 2400], centre + np.array([1.2, 0.5, 0.3])),
                "toy": (toy[k * 2400:(k + 1) * 2400], centre + np.array([0.5, -0.4, -0.3])),
                "self": (y[(k % 8) * 2400:(k % 8 + 1) * 2400], mouth)}
        t0 = time.perf_counter()
        ea.tick(eL, eR, srcs)
        tt.append(time.perf_counter() - t0)
    print(f"the ears with three sources: {1e3 * np.median(tt):.2f} ms a tick median, {1e3 * np.percentile(tt, 95):.2f} ms 95th pct")


def growth_sets(cache):
    """C25 over every growth word's introduction (P3: body/sim/lang/templates.py INTRO, INTRO_WORD): its 3 lines in the new-word
    register, the word emphasized, the words a second pooled over the set; a frame's object slot filled with "block"."""
    from body.sim.lang import templates as TP                             # noqa: PLC0415
    from body.sim.lang.percept import Seen                                  # noqa: PLC0415
    rows = []
    for w in TP.GROWTH_WORDS:
        frames = TP.intro_frames(w)
        if not frames:
            continue
        lines = [TP.fill(fr, o=Seen("block", "block", "red", "mat"), w=w)[0] for fr in frames]
        cs = [cache.clip(ln, "new_word", emphasis=w) for ln in lines]
        rows.append((w, sum(len(c.words) for c in cs) / sum((c.words[-1][2] - c.words[0][1]) / V.SR for c in cs), lines))
    r = np.array([x[1] for x in rows])
    over = [(w, round(v, 2)) for w, v, _ in rows if v > 3]
    print(f"the growth words' introductions (C25), {len(rows)} sets of 3 lines in the new-word register: {r.mean():.2f} words a "
          f"second on average, {r.min():.2f}-{r.max():.2f}; {len(over)} over 3: {over}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--growth", action="store_true", help="only the growth words' introductions (C25)")
    a = ap.parse_args()
    lines = [ln.strip() for ln in open(a.lines) if ln.strip()] if a.lines else FALLBACK
    tmp = None
    root = a.cache or (tmp := tempfile.mkdtemp(prefix="voicecheck_"))
    cache = V.VoiceCache(root)
    try:
        if a.growth:
            growth_sets(cache)
            return
        clips, rms = voice(lines, cache, a.n)
        tract(rms)
        ears(clips)
    finally:
        cache.close()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
