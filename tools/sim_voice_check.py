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

  peak       (--peak) the new word on the line's pitch peak, by frame (A34): every line she can say with a growth word as the
             new word (templates.new_word_lines(): its introduction frames and her intents' frames the word can fill), in the
             new-word register with the word emphasized, each word's peak F0 by f0_word; on its peak when it peaks above every
             other word of the line (a tie at the tracker's resolution is no peak). Writes the table the line check reads
             (body/sim/lang/peak_lines.json), and reports the lines off their peak and, per growth word (and object, for a frame
             with an object slot), how many of its introduction lines of distinct words are on it (a set needs 3: 4.5's frames
             differing by at least one word)

  trial      (--trial) a formal trial's stimuli time-matched by construction (P3's twelfth round, the lead's decision on C67):
             the engine's per-word rate measured in its steps; for each form (its carrier fixed: "where is the X?", the name
             and its foils, "where is the C X?" of each held pair's noun) each test word said at every step, its timeline taken
             (lang/stimuli.timeline); the form's timeline the one the most words reach with every tick loud or silent, each
             word's rate the step nearest its natural one and its pitch matching its contour to the form's (TRIAL_F0); the
             words unmatched, with why; each word's words-channel symbols; each form's one carrier recording (its first
             matched word's sentence, P3's thirteenth round), every matched sentence spliced onto it and checked one
             timeline, sample for sample the same before its test word's onset tick (lang/stimuli.splice). Writes the table
             the conduct reads (body/sim/lang/trial_lines.json)

Run: nice -n 19 python3 tools/sim_voice_check.py [--lines FILE] [--cache DIR] [--n N] [--growth] [--peak [--write]] [--trial
     [--write]]
  --lines  one line a line (default: 16 lines written here; the commits used the all-out study's 331 birth template lines,
           $S/allout/lang/all_lines.txt: 275 end in ".", 37 in "?", 19 in "!"; 314 distinct)
  --cache  the voice cache to use (default: a temporary folder, removed after)
  --growth only the growth words' introductions (P3's INTRO frames on their pitch peak, A34), each word's lines' words a second
           in the new-word register, pooled (C25)
  --peak   only the new word's pitch peak by frame (A34); --write writes the table into body/sim/lang/peak_lines.json
  --trial  only the formal trials' stimuli; --write writes the table into body/sim/lang/trial_lines.json
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
from body.sim.voice.playback import CUT_MAX, TICK, Utterance  # noqa: E402
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


def intro_lines(TP, w):
    """a growth word's introduction lines as she can say them (A34: on the pitch peak, as measured), a frame's object slot filled
    with each object word: [(frame, line)]."""
    from body.sim.lang.percept import Seen                                  # noqa: PLC0415
    out = []
    for fr in TP.intro_frames(w):
        for n in (sorted(TP.OBJECT_NOUNS) if "{o}" in fr[0] else [None]):
            got = TP.fill(fr, o=None if n is None else Seen(n, n, "", "mat"), w=w)
            if got is not None:
                out.append((fr[0], got[0]))
    return out


def growth_sets(cache):
    """C25 over every growth word's introduction (P3: body/sim/lang/templates.py INTRO, INTRO_WORD): its lines on the pitch peak
    (A34, the table --peak writes), in the new-word register, the word emphasized; for a word whose frames have an object slot,
    once for each object word it may be shown on (a colour: the room's toys of it, never its held-out twin, A55). The words a
    second pooled over them, and the most over any 3 of them of distinct words (a set she may draw, 4.5)."""
    import itertools                                                         # noqa: PLC0415
    from body.sim.lang import consts as K                                   # noqa: PLC0415
    from body.sim.lang import templates as TP                               # noqa: PLC0415
    rows = []
    for w in TP.GROWTH_WORDS:
        got = [(fr, ln) for fr, ln in intro_lines(TP, w) if TP.on_peak(ln)[0]]
        if not got:
            continue
        objs = sorted({ln.split()[1] for fr, ln in got if "{o}" in fr}) or [None]
        if TP.GROWTH_CLASS[w] == "colour":
            objs = [n for n in objs if w in TP.ROOM_AT_BIRTH["objects"].get(n, ()) and (w, n) not in K.HELD_PAIRS]
        for o in objs:
            lines = [ln for fr, ln in got if "{o}" not in fr or ln.split()[1] == o]
            cs = [cache.clip(ln, "new_word", emphasis=w) for ln in lines]
            per = [(len(c.words), (c.words[-1][2] - c.words[0][1]) / V.SR) for c in cs]
            sets = [k for k in itertools.combinations(range(len(per)), 3) if len({TP.key(lines[i]) for i in k}) == 3]
            worst = max(((sum(per[i][0] for i in k) / sum(per[i][1] for i in k)), tuple(lines[i] for i in k))
                        for k in sets) if sets else (float("nan"), ())
            rows.append((w if o is None else f"{w} ({o})", sum(a for a, _ in per) / sum(b for _, b in per), worst, lines))
    r = np.array([x[1] for x in rows])
    wr = np.array([x[2][0] for x in rows if np.isfinite(x[2][0])])
    over = [(w, round(v, 2)) for w, v, _, _ in rows if v > 3]
    over3 = [(w, round(v[0], 2), v[1]) for w, _, v, _ in rows if np.isfinite(v[0]) and v[0] > 3]
    n_lines = len({ln for x in rows for ln in x[3]})
    print(f"the growth words' introductions (C25), {len(rows)} words (or word and object), their lines on the pitch peak (A34; "
          f"{n_lines} lines) in the new-word register: {r.mean():.2f} words a second pooled on average, {r.min():.2f}-"
          f"{r.max():.2f}, {len(over)} over 3: {over}; the fastest 3 of a word's lines {wr.mean():.2f} on average, at most "
          f"{wr.max():.2f}, {len(over3)} over 3: {over3}")
    return rows


def peak_table(cache, write=False):
    """A34: every line she can say with a growth word as the new word, its word emphasized in the new-word register; each word's
    peak F0 (f0_word); the new word on the line's pitch peak when it peaks above every other word (a tie is no peak). Reports
    the lines off their peak, the introduction lines among them, and each growth word's count of introduction lines of
    distinct words on it (per object, for a frame with an object slot); writes the table."""
    from body.sim.lang import consts as K                                   # noqa: PLC0415
    from body.sim.lang import templates as TP                               # noqa: PLC0415
    lines = TP.new_word_lines()
    table = {}
    t0 = time.perf_counter()
    for text, w in lines:
        c = cache.clip(text, "new_word", emphasis=w, heard=False)
        pk = []
        for word, on, end in c.words:
            _, p = f0_word(c.pcm[on:end])
            pk.append((word, None if not np.isfinite(p) else round(float(p), 1)))
        assert pk[-1][0] == w, (text, pk)
        others = [(p, word) for word, p in pk[:-1] if p is not None]
        top = max(others) if others else (None, None)
        table[text] = [w, pk[-1][1], top[0], top[1], len(c.words), round((c.words[-1][2] - c.words[0][1]) / V.SR, 4)]
    wall = time.perf_counter() - t0
    off = {t: r for t, r in table.items() if r[1] is None or (r[2] is not None and r[2] >= r[1])}
    n_off_measured = len(off)
    intro = {}
    for w in TP.GROWTH_WORDS:
        for fr, ln in intro_lines(TP, w):
            intro.setdefault(w, []).append((fr, ln))
    n_intro = sum(len(v) for v in intro.values())
    unmeasured = {ln for v in intro.values() for _, ln in v if ln not in table}    # a line the check would refuse anyway
    off.update({ln: [None, None, None, "not measured"] for ln in unmeasured})
    off_intro = [ln for v in intro.values() for _, ln in v if ln in off]
    print(f"the new word's pitch peak (A34): {len(table)} lines measured in {wall:.0f} s, {n_off_measured} off their peak; "
          f"{n_intro} introduction lines (every frame, an object slot filled with each object word), {len(off_intro)} off")
    by_frame = {}
    for w, v in intro.items():
        for fr, ln in v:
            by_frame.setdefault((TP.GROWTH_CLASS[w] if w not in TP.INTRO_WORD else w, fr), []).append(ln not in off)
    for (cls, fr), oks in sorted(by_frame.items()):
        if not all(oks):
            print(f"  {cls:8s} {fr!r}: {sum(oks)} of {len(oks)} on the peak")
    short = {}
    for w, v in intro.items():
        objs = sorted({ln.split()[1] for fr, ln in v if "{o}" in fr}) or [None]
        if TP.GROWTH_CLASS[w] == "colour":                              # a colour on the room's toys of it, never its twin (A55)
            objs = [n for n in objs if w in TP.ROOM_AT_BIRTH["objects"].get(n, ()) and (w, n) not in K.HELD_PAIRS]
        for o in objs:                                                  # a set is said of one object: count per object
            keys = {TP.key(ln) for fr, ln in v if ln not in off and ("{o}" not in fr or ln.split()[1] == o)}
            if len(keys) < 3:
                short[w if o is None else f"{w} ({o})"] = sorted(keys)
    print(f"  growth words (and objects) with fewer than 3 introduction lines of distinct words on the peak (they wait there, "
          f"A34, 4.5): {len(short)}: " + ", ".join(f"{w} ({len(f)})" for w, f in sorted(short.items())))
    print(f"  introduction lines no line check passes (never said, not measured): {sorted(unmeasured)}")
    print("  the lines off the peak: " + "; ".join(f"{t!r} ({r[3]} {r[2]} over {r[0]} {r[1]})" for t, r in sorted(off.items())
                                                  if t in table))
    if write:
        info = cache._server().info()
        meta = dict(tool="tools/sim_voice_check.py --peak", measure="f0_word's peak: the largest of the word's voiced 40 ms "
                    "frames (hop 10 ms, autocorrelation over 0.5), octave errors dropped (1.6 x its median), median-filtered "
                    "over 3", register="new_word", emphasis=list(V.EMPHASIS), rule="on the peak: the new word's peak above "
                    "every other word's (a tie is no peak)", engine=f"{info['name']} ({info['identifier']}), {info['os']}",
                    lines=len(table),
                    fields="the new word; its peak F0, Hz; the highest other word's peak F0, Hz; that word; the line's words; "
                    "its spoken span, s (the first word's onset to the last word's end: C25's words a second)",
                    off=n_off_measured, date=time.strftime("%Y-%m-%d"))
        path = os.path.join(ROOT, "body", "sim", "lang", K.PEAK_FILE)
        with open(path, "w") as fh:                                     # one line a measured line, sorted, for review
            fh.write('{"meta": ' + json.dumps(meta, sort_keys=True) + ',\n "lines": {\n')
            fh.write(",\n".join(f"  {json.dumps(t)}: {json.dumps(r)}" for t, r in sorted(table.items())))
            fh.write("\n}}\n")
        print(f"  written: {path} ({os.path.getsize(path) / 1024:.0f} KB)")
    return table, off


def f0_contour(pcm):
    """an instrument: a word's pitch contour, its F0 at the 10th, 50th and 90th percentile of its voiced frames (octave errors
    dropped as f0_word drops them), or None where none is voiced."""
    v = f0_track(pcm)
    v = v[v > 0]
    if not len(v):
        return None
    m = np.median(v)
    v = v[(v < 1.6 * m) & (v > m / 1.6)]
    return [float(np.percentile(v, q)) for q in (10, 50, 90)]


def _dev(c, ref):
    """the largest relative difference of a contour from the form's, over its three percentiles."""
    return max(abs(a / b - 1.0) for a, b in zip(c, ref))


def rate_steps(cache, text="where is the stacker?", word="stacker", register="question", top=200):
    """the engine's per-word rate in its steps: a word's own prosody rate from 1% to `top`% of the engine's default, its clip
    changing only at a step (measured on one line) -> the first rate of each step."""
    steps, prev = [], None
    for r in range(1, top + 1):
        c = cache.clip(text, register, emphasis=word, heard=False, shape=(word, r, 30.0))
        sig = (len(c.pcm), tuple(c.words[-1][1:]))
        if sig != prev:
            steps.append(r)
            prev = sig
    return steps


def trial_table(cache, write=False):
    """P3's twelfth round (the lead's decision on C67): a formal trial's stimuli time-matched by construction. For each form (its
    carrier fixed: "where is the X?" of every object word, "X." of the name and each foil, "where is the C X?" of each colour for
    each held pair's noun), each test word is said at every step of the engine's per-word rate at its natural pitch
    (the emphasis's +30% on the emphasized word, the line's on a colour), and its timeline taken (lang/stimuli.timeline, the
    words channel aside: the conduct adds it while the scaffold labels her lines). The form's timeline is the one the most words
    reach with every tick loud or silent (ties: the least change of rate from the natural); each word on it takes the step
    nearest its natural rate, then the pitch that best matches its contour to the form's (the median of its words' 10th, 50th
    and 90th percentile F0; the name's own for its foils, the name said as she always says it), within TRIAL_F0; a rate is
    tried only within TRIAL_RATE_SPAN of the natural one (the span her own registers' rates run). A word that reaches no step on the timeline, or whose contour cannot be matched,
    is unmatched, with why. The form's carrier (P3's thirteenth round): its first matched word's sentence, whose samples before
    the test word's onset tick every sentence of the form takes (lang/stimuli.splice), each matched sentence so spliced checked
    one timeline with the others (stimuli.same, their samples before the tick among it). Reports each form's words, their
    channel (a birth word's token, or its letters) and which pairs share it; writes the table the conduct reads
    (body/sim/lang/trial_lines.json)."""
    from body.sim.lang import consts as K                                   # noqa: PLC0415
    from body.sim.lang import stimuli as ST                                 # noqa: PLC0415
    from body.sim.lang import templates as TP                               # noqa: PLC0415
    t0 = time.perf_counter()
    steps = rate_steps(cache)
    line_rate = V.REGISTERS["question"][1] / V.ENGINE_RATE * 100           # the question and calling registers: 50%
    emph_rate = V.EMPHASIS[1] * line_rate                                   # the emphasized word: 35%
    forms = [("where", "where is the {w}?", "question", True, sorted(TP.OBJECT_NOUNS), None),
             ("name", "{w}.", "calling", True, [LX.NAME] + list(K.NAME_FOILS), LX.NAME)]
    for noun in sorted({n for _c, n in K.HELD_PAIRS}):
        forms.append((f"combination:{noun}", "where is the {w} " + noun + "?", "question", False, sorted(TP.COLOURS), None))
    out = {}
    for key, text, reg, emph, words, anchor in forms:
        nat_r, nat_p = (emph_rate, 30.0) if emph else (line_rate, 0.0)
        slot = TP.words(text.replace("{w}", "x")).index("x")
        reps = sorted({nat_r if a <= nat_r < b else a for a, b in zip(steps, steps[1:] + [10 ** 6])})
        reps = [r for r in reps if abs(math.log(r / nat_r)) <= math.log(K.TRIAL_RATE_SPAN) + 1e-9]

        def render(w, r, p, text=text, reg=reg, emph=emph):
            t = text.replace("{w}", w)
            e = w if emph else TP.words(t)[-1]
            return cache.clip(t, reg, emphasis=e, heard=False, shape=(w, r, p))

        seen = {}                                                       # timeline -> {word: [rates]}
        clips = {}
        for w in words:
            for r in reps:
                c = render(w, r, nat_p)
                tl = ST.timeline(c.words, len(c.pcm), c.pcm, slot=slot, pre=False)
                clips[(w, r)] = (c, tl)
                if "?" in tl["sound"]:
                    continue
                k = json.dumps(tl, sort_keys=True)
                seen.setdefault(k, {}).setdefault(w, []).append(r)

        def cost(ws):
            return sum(min(abs(math.log(r / nat_r)) for r in rs) for rs in ws.values()) / len(ws)
        if anchor is not None:                                          # the name: said as she always says it, the foils to it
            seen = {k: ws for k, ws in seen.items() if nat_r in ws.get(anchor, ())}
        if not seen:                                                    # no word on any timeline with every tick loud or silent
            why = {w: f"no step of the engine's per-word rate within a factor of {K.TRIAL_RATE_SPAN:g} of its natural "
                      f"{nat_r:g}% says it with every tick loud or silent" for w in words}
            out[key] = dict(text=text, register=reg, emphasis="{w}" if emph else TP.words(text.replace("{w}", "x"))[-1],
                            slot=slot, natural=dict(rate=nat_r, pitch=nat_p), timeline=None, f0=None, words={}, unmatched=why)
            print(f"{key}: none matched ({len(words)} unmatched: no timeline with every tick loud or silent)")
            continue
        best = min(seen.items(), key=lambda kv: (-len(kv[1]), cost(kv[1])))
        tl_ref, reach = json.loads(best[0]), best[1]
        chosen = {w: min(rs, key=lambda r: (abs(math.log(r / nat_r)), r)) for w, rs in reach.items()}
        cont = {w: f0_contour(clips[(w, r)][0].pcm[slice(*clips[(w, r)][0].words[slot][1:])]) for w, r in chosen.items()}
        ref = cont[anchor] if anchor is not None else [float(np.median([c[i] for c in cont.values() if c])) for i in range(3)]
        rec, unmatched = {}, {}
        for w, r in chosen.items():
            best_p, best_d, best_c = nat_p, _dev(cont[w], ref) if cont[w] else 9.9, cont[w]
            if cont[w] and w != anchor:
                k_ = math.exp(np.mean([math.log(a / b) for a, b in zip(ref, cont[w])]))
                p0 = round(((1 + nat_p / 100.0) * k_ - 1) * 100)
                for p in (p0 - 1, p0, p0 + 1):
                    c = render(w, r, float(p))
                    tl = ST.timeline(c.words, len(c.pcm), c.pcm, slot=slot, pre=False)
                    ct = f0_contour(c.pcm[c.words[slot][1]:c.words[slot][2]])
                    if json.loads(json.dumps(tl, sort_keys=True)) == tl_ref and ct and _dev(ct, ref) < best_d:
                        best_p, best_d, best_c = float(p), _dev(ct, ref), ct
            c = render(w, r, best_p)
            x = c.pcm[c.words[slot][1]:c.words[slot][2]].astype(np.float64)
            if best_d > K.TRIAL_F0:
                unmatched[w] = (f"its pitch contour {[round(v) for v in best_c or []]} Hz against the form's "
                                f"{[round(v) for v in ref]}: {best_d:.0%} off, beyond {K.TRIAL_F0:.0%}")
                continue
            rec[w] = dict(rate=r, pitch=best_p, natural=bool(r == nat_r and best_p == nat_p), f0=[round(v, 1) for v in best_c],
                          f0_dev=round(best_d, 4), rms=round(float(np.sqrt(np.mean(x * x))), 1),
                          ms=round((c.words[slot][2] - c.words[slot][1]) / 16.0), channel=len(LX.symbols_for(w)))
        for w in words:
            if w in reach or w in unmatched:
                continue
            r = min(reps, key=lambda r: abs(math.log(r / nat_r)))
            c, tl = clips[(w, r)]
            why = [f"{f} {_brief(tl[f])} against {_brief(tl_ref[f])}" for f in ("ticks", "words", "sound")
                   if json.loads(json.dumps(tl[f])) != tl_ref[f]]
            near = [(r2, clips[(w, r2)][1]) for r2 in reps if clips[(w, r2)][1]["ticks"] == tl_ref["ticks"] and
                    json.loads(json.dumps(clips[(w, r2)][1]["words"])) == tl_ref["words"]]
            if near:
                r2, tl2 = min(near, key=lambda x: abs(math.log(x[0] / nat_r)))
                c2 = clips[(w, r2)][0]
                db = _tick_db(c2.pcm)
                bad = [(k, round(db[k][0], 1), round(db[k][1], 1)) for k, s in enumerate(tl2["sound"])
                       if s == "?" or s != tl_ref["sound"][k]]
                why = [f"at rate {r2:g}% on its ticks, but its tick{'s' if len(bad) > 1 else ''} " +
                       ", ".join(f"{k} at {a:g} dB RMS, {b:g} dB its loudest 10 ms" for k, a, b in bad) +
                       f" (loud: its RMS above {K.TRIAL_LOUD_DB:g}; silent: its loudest 10 ms below {K.TRIAL_SILENT_DB:g})"]
            unmatched[w] = (f"no step of the engine's per-word rate within a factor of {K.TRIAL_RATE_SPAN:g} of its natural "
                            f"{nat_r:g}% puts it on the form's timeline: " + "; ".join(why))
        rms = np.median([v["rms"] for v in rec.values()]) if rec else 1.0
        for v in rec.values():
            v["rms"] = round(v["rms"] / rms, 3)
        # the form's one carrier recording (P3's thirteenth round): its first matched word's sentence (none before a name)
        car = sorted(rec)[0] if rec and tl_ref["words"][slot][0] > 0 else None
        out[key] = dict(text=text, register=reg, emphasis="{w}" if emph else TP.words(text.replace("{w}", "x"))[-1], slot=slot,
                        natural=dict(rate=nat_r, pitch=nat_p), timeline=tl_ref, f0=[round(v, 1) for v in ref],
                        words=dict(sorted(rec.items())), unmatched=dict(sorted(unmatched.items())), carrier=car)
        chans = {}
        for w in rec:
            chans.setdefault("a token" if rec[w]["channel"] == 1 else f"{rec[w]['channel'] - 1} letters", []).append(w)
        print(f"{key}: {tl_ref['ticks']} ticks, its test word on ticks {tl_ref['words'][slot]}; matched {len(rec)} of "
              f"{len(words)}: " + ", ".join(f"{w} (rate {v['rate']:g}%, pitch {v['pitch']:+g}%, F0 {v['f0_dev']:.0%} off, "
                                              f"{v['ms']} ms, rms {v['rms']:.2f})" for w, v in rec.items()))
        print("   the words channel: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in sorted(chans.items())))
        for w, why in unmatched.items():
            print(f"   unmatched {w}: {why}")
        if car is not None:                                             # every sentence on it: one timeline, sample for
            at = tl_ref["words"][slot][0] * TICK                        # sample the same before its test word's onset tick
            cc = render(car, rec[car]["rate"], rec[car]["pitch"])
            tls = {}
            for w, v in rec.items():
                sc = ST.splice(cc, render(w, v["rate"], v["pitch"]), at)
                tls[w] = ST.timeline(sc.words, len(sc.pcm), sc.pcm, slot=slot)
            why = ST.same(list(tls.values()), list(tls))
            print(f"   its one carrier: {car!r}'s sentence, every sample before tick {tl_ref['words'][slot][0]} "
                  f"({at} samples) taken by each: " + ("one timeline" if why is None else f"NOT one timeline: {why}"))
    print(f"  the engine's per-word rate steps (first rate of each, %): {steps}; {time.perf_counter() - t0:.0f} s")
    if write:
        info = cache._server().info()
        meta = dict(tool="tools/sim_voice_check.py --trial", date=time.strftime("%Y-%m-%d"),
                    engine=f"{info['name']} ({info['identifier']}), {info['os']}", steps=steps,
                    band=[K.TRIAL_LOUD_DB, K.TRIAL_SILENT_DB], f0_tol=K.TRIAL_F0,
                    band_rule="a tick loud when its RMS is above the first (dB of full scale), silent when its loudest 10 ms is "
                              "below the second, else neither (a timeline no level makes one)",
                    rate_span=K.TRIAL_RATE_SPAN,
                    carrier="each form's one carrier recording (P3's thirteenth round): its first matched word's sentence, "
                            "whose samples before the test word's onset tick every sentence of the form takes "
                            "(lang/stimuli.splice: a 10 ms raised cosine into the sentence's own at that tick)",
                    rule="a form's timeline: the one the most test words reach (every tick loud or silent) at a rate within "
                         "the span of her registers' rates of its natural one (the name: said at its own; ties: the least "
                         "change of rate); each word's rate the step nearest its natural rate, its pitch the one matching its "
                         "contour (10th, 50th, 90th percentile F0) to the form's median; the conduct holds every trial to the "
                         "rendered timelines (stimuli.same), the words channel added while the scaffold labels her lines",
                    fields="words: rate (% of the engine's default), pitch (% over the line's), natural (the word said as "
                           "before), f0 (Hz at 10/50/90%), f0_dev (the largest off the form's), rms (relative to the form's "
                           "median), ms (the word's length), channel (its symbols on the words channel)")
        path = os.path.join(ROOT, "body", "sim", "lang", K.TRIAL_FILE)
        with open(path, "w") as fh:
            fh.write('{"meta": ' + json.dumps(meta, sort_keys=True) + ',\n "forms": {\n')
            fh.write(",\n".join(f"  {json.dumps(k)}: {json.dumps(v, sort_keys=True)}" for k, v in sorted(out.items())))
            fh.write("\n}}\n")
        print(f"  written: {path} ({os.path.getsize(path) / 1024:.0f} KB)")
    return out


def _brief(v):
    s = json.dumps(v)
    return s if len(s) < 80 else s[:77] + "..."


def _tick_db(pcm):
    """each tick's (RMS, loudest 10 ms frame), dB of full scale (lang/stimuli.levels)."""
    from body.sim.lang.stimuli import levels                                # noqa: PLC0415
    return list(zip(*levels(pcm)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--growth", action="store_true", help="only the growth words' introductions (C25)")
    ap.add_argument("--peak", action="store_true", help="only the new word's pitch peak by frame (A34)")
    ap.add_argument("--trial", action="store_true", help="only the formal trials' time-matched stimuli (P3's twelfth round)")
    ap.add_argument("--write", action="store_true", help="with --peak or --trial: write body/sim/lang/peak_lines.json or "
                                                         "trial_lines.json")
    a = ap.parse_args()
    lines = [ln.strip() for ln in open(a.lines) if ln.strip()] if a.lines else FALLBACK
    tmp = None
    root = a.cache or (tmp := tempfile.mkdtemp(prefix="voicecheck_"))
    cache = V.VoiceCache(root)
    try:
        if a.peak:
            peak_table(cache, a.write)
            return
        if a.trial:
            trial_table(cache, a.write)
            return
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
