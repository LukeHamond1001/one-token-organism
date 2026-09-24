"""the voices (docs/SIM_DESIGN.md 4.4, 4.9, 3.4 channel 0; the owner's decision 5; package P1): the child's vocal tract
(body/sim/tract.py), the parent's voice (body/sim/voice/synth.py and playback.py) and the words channel
(body/sim/lang/lexicon.py). Run: python3 -m body.tests.test_sim_voice

The tract: its vowels' formants where the study left them; the same acts from the same seed give the same samples, and save and
restore continue bit for bit; a closure stops voicing by itself (velum raised) and a lowered velum keeps a nasal murmur; rest is
silent; the breath runs out and refills; its body sense; its cost. The parent's voice (macOS only; skipped without swiftc): the
engine bit for bit across two server processes; the cache's hits, its ledger, a damaged file made again, a changed engine
refused, the size limit; the words' onsets and ends; the registers' pitch and level, and plain speech at 62 dB SPL; a line played
tick by tick into the words channel, a talk-over cut at a word's end, and its save and restore. The table: 79 rows, "a" the word
apart from "a" the letter, a later word spelled, and the channel's audibility."""
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # this tree's body
from body.sim.lang import lexicon as LX  # noqa: E402
from body.sim.voice import synth as V  # noqa: E402
from body.sim import tract as T  # noqa: E402
from body.sim.voice.playback import Utterance  # noqa: E402

TICK = 2400


def posture(**kw):
    x = T.NEUTRAL.copy()
    for k, v in kw.items():
        x[T.NAMES.index(k)] = v
    return x


def held(tr, x, n=3):
    """hold the tract at posture x (as if it had got there): n ticks of 'no step' from it."""
    tr.x, tr.target = x.copy(), x.copy()
    return np.concatenate([tr.tick(np.full(10, 2)) for _ in range(n)])


VOICED = dict(lungs=0.6, glottis=0.7, velum=0.0)


def f0_track(x, sr=16000, lo=100.0, hi=700.0, frame=640, hop=160):
    """an instrument: each 40 ms frame's F0 by normalized autocorrelation (0 where its peak is under 0.5)."""
    out = []
    for s in range(0, len(x) - frame, hop):
        f = x[s:s + frame].astype(np.float64)
        f = f - f.mean()
        if np.dot(f, f) < 1e-9:
            out.append(0.0)
            continue
        ac = np.correlate(f, f, "full")[frame - 1:]
        ac = ac / ac[0]
        k = int(sr / hi) + int(np.argmax(ac[int(sr / hi):int(sr / lo)]))
        out.append(sr / k if ac[k] > 0.5 else 0.0)
    return np.array(out)


def sounding_rms(pcm):
    e = V.frame_rms(pcm)
    act = e > e.max() * 10 ** (V.WORD_END_DB / 20)
    return float(np.sqrt((e[act] ** 2).mean()))


# ------------------------------------------------------------------------------------------------------------------ the tract
def test_tract_vowels():
    want = {"i": ((344, 2719), posture(**VOICED, jaw=0.15, tongue_front=1.0, tongue_height=0.85, lips=0.8, rounding=0.0)),
            "a": ((938, 1625), posture(**VOICED, jaw=0.8, tongue_front=0.0, tongue_height=0.0, lips=0.9, rounding=0.2)),
            "u": ((375, 1250), posture(**VOICED, jaw=0.15, tongue_front=0.2, tongue_height=0.85, lips=0.35, rounding=1.0))}
    got = []
    for v, ((f1, f2), x) in want.items():
        f, _ = T.formants(x)
        assert abs(f[0] - f1) <= 0.05 * f1 and abs(f[1] - f2) <= 0.05 * f2, f"/{v}/: F1 {f[0]:.0f}, F2 {f[1]:.0f}; the study's {f1}/{f2}"
        got.append(f"/{v}/ {f[0]:.0f}/{f[1]:.0f}")
    fa, _ = T.formants(want["a"][1])
    fi, _ = T.formants(want["i"][1])
    assert fa[0] > 2 * fi[0] and fi[1] > 1.5 * fa[1], "the corner vowels do not span the space"
    print("1 the tract's vowels (F1/F2 Hz):", ", ".join(got), "(the study's 344/2719, 938/1625, 375/1250)")


def babble(seed, n, rng_acts=0):
    r = np.random.default_rng(rng_acts)
    acts = [None if r.random() < 0.4 else r.integers(0, 5, 10) for _ in range(n)]
    tr = T.Tract(seed)
    return tr, acts


def test_tract_deterministic():
    tr, acts = babble(1, 40)
    y1 = np.concatenate([tr.tick(a) for a in acts])
    tr2, _ = babble(1, 40)
    y2 = np.concatenate([tr2.tick(a) for a in acts])
    assert np.array_equal(y1, y2) and np.any(y1), "the same acts from the same seed gave different samples"
    tr3, _ = babble(2, 40)
    y3 = np.concatenate([tr3.tick(a) for a in acts])
    assert not np.array_equal(y1, y3), "another seed gave the same jitter and noise"
    tr4, _ = babble(1, 40)
    for a in acts[:17]:
        tr4.tick(a)
    st = tr4.state()
    rest = np.concatenate([tr4.tick(a) for a in acts[17:]])
    tr5 = T.Tract(99)
    tr5.load_state(st)
    rest2 = np.concatenate([tr5.tick(a) for a in acts[17:]])
    assert np.array_equal(rest, rest2) and np.array_equal(rest, y1[17 * TICK:]), "save and restore did not continue exactly"
    print(f"2 the tract is exact: the same seed and acts give the same {len(y1)} samples, another seed differs, and a restored "
          f"tract continues bit for bit")


def test_closure_stops_voicing():
    a = posture(**VOICED, jaw=0.8, tongue_front=0.0, tongue_height=0.0, lips=0.9, rounding=0.2)
    out = {}
    for name, vel in (("oral", 2), ("nasal", 4)):                     # the velum held raised, or lowered by two big steps
        tr = T.Tract(1)
        y0 = held(tr, a, 3)[-TICK:]
        v_open = float(tr.last["aero"]["v"].mean())
        shut = np.full(10, 2)
        shut[T.NAMES.index("lips")] = 0                              # the lips shut by big steps (0.9 -> 0.3 -> 0)
        shut[T.NAMES.index("jaw")] = 0
        shut[T.NAMES.index("velum")] = vel
        tr.tick(shut)
        tr.tick(shut)
        y1 = tr.tick(np.full(10, 2))                                 # held shut
        v_shut = float(tr.last["aero"]["v"].mean())
        out[name] = (v_open, v_shut, 20 * math.log10(max(np.sqrt(np.mean(y1 ** 2)), 1e-12) / np.sqrt(np.mean(y0 ** 2))))
    vo, vs, db = out["oral"]
    assert vo > 0.5 and vs < 0.02 and db < -30, f"lips shut, velum raised: voicing {vo:.2f} -> {vs:.2f}, level {db:.1f} dB"
    _, vn, dbn = out["nasal"]
    assert vn > 0.5 and -30 < dbn < 0, f"lips shut, velum lowered: voicing {vn:.2f}, level {dbn:.1f} dB (a nasal murmur)"
    print(f"3 a closure stops voicing by itself: lips shut with the velum raised, voicing {vo:.2f} -> {vs:.3f} and the sound "
          f"{'silent' if db < -200 else f'{db:.0f} dB'}; with the velum lowered voicing holds at {vn:.2f}, a murmur at {dbn:.0f} dB")


def test_rest_is_silent_and_breath():
    tr = T.Tract(1)
    y = np.concatenate([tr.tick(None) for _ in range(5)])
    assert not np.any(y) and tr.vol == 1.0, "the rest posture made a sound"
    a = posture(lungs=1.0, glottis=0.7, velum=0.0, jaw=0.8, tongue_front=0.0, tongue_height=0.0, lips=0.9, rounding=0.2)
    tr.x, tr.target = a.copy(), a.copy()
    vols, rms = [], []
    for _ in range(40):
        rms.append(np.sqrt(np.mean(tr.tick(np.full(10, 2)) ** 2)))
        vols.append(tr.vol)
    k_out = next(i for i, v in enumerate(vols) if v <= 0.0)
    assert 8 <= k_out <= 30, f"a full breath lasted {k_out} ticks of a loud vowel"
    assert rms[-1] < 0.01 * max(rms), "the voice went on with no breath left"
    back, relaxed = 0, None
    while tr.vol < 1.0:
        tr.tick(None)
        back += 1
        if relaxed is None and tr.vol > 0:
            relaxed = back                                           # the lungs' drive let go: breathing in starts
    assert relaxed <= 5 and back - relaxed + 1 <= 7, f"the breath refilled in {back} ticks at rest ({relaxed} to let go)"
    p = tr.proprio()
    assert p.shape == (21,) and p[-1] == tr.vol
    print(f"4 rest is silent; a loud held vowel empties the breath in {k_out} ticks and the voice stops by itself; at rest the "
          f"lungs let go in {relaxed} ticks and it refills in {back} (0.8 s a breath); the body sense is 21 numbers")


def test_tract_cost():
    a = posture(**VOICED, jaw=0.75, tongue_front=0.0, tongue_height=0.0, lips=0.6)
    tr = T.Tract(1)
    held(tr, a, 3)
    st = tr.state()
    tt = []
    for _ in range(30):
        tr.load_state(st)
        t0 = time.perf_counter()
        tr.tick(np.array([2, 2, 3, 0, 4, 4, 2, 2, 2, 2]))
        tt.append(time.perf_counter() - t0)
    rest = T.Tract(1)
    tr_ = []
    for _ in range(30):
        t0 = time.perf_counter()
        rest.tick(None)
        tr_.append(time.perf_counter() - t0)
    m, r = 1e3 * float(np.median(tt)), 1e3 * float(np.median(tr_))
    assert m < 25 and r < 2, (m, r)
    print(f"5 the tract's cost: {m:.2f} ms a sounding tick (a glide), {r:.2f} ms at rest (median; this Mac, shared)")


# ----------------------------------------------------------------------------------------------------------- the words table
def test_table_and_channel():
    assert LX.N_TABLE == 79 and LX.TABLE[:2] == ("pip", "mama") and LX.TABLE[-3:] == ("<space>", "<rest>", "<end>")
    a_word, a_letter = LX.symbols_for("a"), LX.LETTER_ID["a"]
    assert a_word == [LX.BIRTH_WORDS.index("a")] and a_word[0] != a_letter, "the word 'a' took the letter's row"
    assert LX.spell(LX.symbols_for("rattle")) == "r a t t l e <space>" and LX.symbols_for("ball") == [LX.WORD_ID["ball"]]
    w = LX.Words()
    w.push([("look", 0, 3000), ("rattle", 3000, 9000), ("ball", 9100, 12000)], start_tick=10, n_samples=13000)
    got = [LX.TABLE[w.tick(t)] for t in range(10, 24)]
    assert got[:11] == ["<rest>", "look", "r", "a", "t", "t", "l", "e", "<space>", "ball", "<end>"], got
    assert got[11:] == ["<rest>"] * 3 and dict(w.late)[LX.WORD_ID["ball"]] == 5, (got, w.late)
    w2 = LX.Words()
    w2.push([("look", 0, 3000), ("ball", 3000, 6000)], 0, 6000)
    assert LX.TABLE[w2.tick(0, audible=False)] == "<rest>" and LX.TABLE[w2.tick(1, audible=False)] == "<rest>"
    assert LX.TABLE[w2.tick(2)] == "ball" and w2.dropped == 1, "a word said out of earshot was delivered"
    assert LX.audible(62.0, (-6.0, -7.0)) and not LX.audible(62.0, (-43.0, -45.0)), "the gate: 20 dB SPL at the nearer ear"
    print("6 the table: 79 rows; 'a' the word apart from 'a' the letter; 'rattle' spelled from its onset, one a tick; a token on "
          "its word's end tick, a tick late behind a busy channel; nothing delivered while the parent is not audible:", got)


# ---------------------------------------------------------------------------------------------------------- the parent's voice
def _have_engine():
    return sys.platform == "darwin" and shutil.which("swiftc") is not None


def _cache(tmp):
    return V.VoiceCache(tmp, server=V.SynthServer(V.DEFAULT_CACHE / "bin"))


LINES = ["look at the duck.", "where is the ball?", "hi pip. mama is here.", "here is your bottle.", "yes. the drum!",
         "you see the cup."]


def test_engine_bit_for_bit():
    if not _have_engine():
        print("7 skipped: no swiftc / AVSpeech here")
        return
    s1, s2 = V.SynthServer(V.DEFAULT_CACHE / "bin"), V.SynthServer(V.DEFAULT_CACHE / "bin")
    try:
        for ln in LINES[:3]:
            q = V.request(ln)[1]["ssml"]
            a, b = s1.synth(q, rate=V.ENGINE_RATE, pitch=1.0, ssml=True), s2.synth(q, rate=V.ENGINE_RATE, pitch=1.0, ssml=True)
            assert np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2], f"{ln!r} differs between two server processes"
        plain = s1.synth(LINES[0], rate=0.25, pitch=1.15)
        ssml = s1.synth(V.request(LINES[0])[1]["ssml"], rate=V.ENGINE_RATE, pitch=1.0, ssml=True)
        assert np.array_equal(plain[0], ssml[0]), "the line in SSML is not the utterance at the register's pitch and rate"
    finally:
        s1.close()
        s2.close()
    print("7 the engine is exact: three lines bit for bit across two server processes; a line in SSML is, bit for bit, the "
          "utterance at its register's pitch and rate")


def test_cache_and_ledger():
    if not _have_engine():
        print("8 skipped")
        return
    tmp = tempfile.mkdtemp(prefix="voice_test_")
    try:
        c = _cache(tmp)
        a = c.clip(LINES[0])
        assert c.misses == 1 and c.hits == 0
        b = c.clip(LINES[0])
        assert c.hits == 1 and np.array_equal(a.pcm, b.pcm) and a.words == b.words and a.digest == b.digest
        led = [ln for ln in open(os.path.join(tmp, "ledger.jsonl")) if ln.strip()]
        assert len(led) == 1 and a.digest in led[0]
        pp, pj = c._paths(a.key)
        raw = bytearray(pp.read_bytes())
        raw[1000] ^= 0xFF
        pp.write_bytes(bytes(raw))                                 # a damaged file is made again, and equals the ledger
        d = c.clip(LINES[0])
        assert c.misses == 2 and np.array_equal(d.pcm, a.pcm)
        os.unlink(pp)
        os.unlink(pj)
        c.ledger[a.key]["digest"] = "0" * 64                       # an engine that changed under the life is refused
        try:
            c.clip(LINES[0])
            raise AssertionError("a clip that differs from the ledger was accepted")
        except V.VoiceChanged:
            pass
        c.close()
        c2 = V.VoiceCache(tmp, limit=150_000, server=V.SynthServer(V.DEFAULT_CACHE / "bin"))
        assert c2.ledger[a.key]["digest"] == a.digest, "the ledger was not read back"
        for ln in LINES:
            c2.clip(ln)
            time.sleep(0.01)
        assert c2.size <= c2.limit and len(list(c2.clips.glob("*/*.pcm"))) < len(LINES), "the size limit dropped nothing"
        kept = {p.stem for p in c2.clips.glob("*/*.pcm")}
        assert V.request(LINES[-1])[0] in kept and V.request(LINES[1])[0] not in kept, "not the least recently used dropped"
        assert len(c2.ledger) == len(LINES), "the ledger lost a clip's digest"
        c2.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("8 the cache: a hit returns the same clip; the ledger keeps its digest; a damaged file is made again equal; a changed "
          "engine is refused; the size limit drops the least recently used and the ledger keeps them all")


def test_words_registers_and_level():
    if not _have_engine():
        print("9 skipped")
        return
    tmp = tempfile.mkdtemp(prefix="voice_test_")
    try:
        c = _cache(tmp)
        k = c.clip("look at the duck.")
        assert [w for w, _, _ in k.words] == ["look", "at", "the", "duck"], k.words
        for (w0, on0, e0), (w1, on1, e1) in zip(k.words, k.words[1:]):
            assert on0 < e0 <= on1 < e1, k.words
        assert k.words[-1][2] <= len(k.pcm) and k.words[0][1] == 0
        f0 = {}
        for reg in ("comfort", "plain", "approval"):
            f = np.concatenate([f0_track(c.clip(ln, reg).pcm) for ln in LINES[:3]])
            f0[reg] = float(np.median(f[f > 0]))
        assert f0["comfort"] < f0["plain"] < f0["approval"], f0
        assert abs(f0["approval"] / f0["plain"] - 1.35 / 1.15) < 0.04, f0
        call, plain = c.clip(LINES[2], "calling"), c.clip(LINES[2], "plain")
        assert abs(20 * math.log10(np.sqrt(np.mean(call.pa() ** 2)) / np.sqrt(np.mean(plain.pa() ** 2))) - 6.0) < 0.1
        rms = np.array([sounding_rms(c.clip(ln).pcm) for ln in LINES])
        spl = 20 * math.log10(np.sqrt(np.mean(rms ** 2)) * V.PA_PER_UNIT / 20e-6)
        assert abs(spl - 62.0) < 1.5, f"plain speech at 1 m is {spl:.1f} dB SPL"
        e = c.clip("look. a duck.", emphasis="duck")
        p = c.clip("look. a duck.")
        de, dp = [x for x in e.words if x[0] == "duck"][0], [x for x in p.words if x[0] == "duck"][0]
        assert de[2] - de[1] > 1.3 * (dp[2] - dp[1]), "the emphasized word is not longer"
        c.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"9 the words' onsets and ends in order; F0 comfort {f0['comfort']:.0f} < plain {f0['plain']:.0f} < approval "
          f"{f0['approval']:.0f} Hz; calling +6 dB; plain speech {spl:.1f} dB SPL at 1 m; an emphasized word longer")


def test_playback_into_the_words_channel():
    if not _have_engine():
        print("10 skipped")
        return
    tmp = tempfile.mkdtemp(prefix="voice_test_")
    try:
        c = _cache(tmp)
        k = c.clip("hi pip. mama is here.")
        u, w = Utterance(k, 100), LX.Words()
        got, out = [], []
        t = 100
        while not u.done or w.queue:
            out.append(u.tick(t, w))
            got.append((t, LX.TABLE[w.tick(t)]))
            t += 1
        y = np.concatenate(out)[:len(k.pcm)]
        assert np.array_equal(y, k.pa()), "the line played is not its clip"
        toks = [(t, s) for t, s in got if s in LX.BIRTH_WORDS]
        assert [s for _, s in toks] == ["hi", "pip", "mama", "is", "here"], toks
        for (t, s), (word, on, end) in zip(toks, k.words):
            assert 0 <= t - (100 + (end - 1) // TICK) <= 1, (s, t, end)
        assert got[-1][1] == "<end>" or any(s == "<end>" for _, s in got)
        # talk-over: cut two ticks in; the word sounding is finished, the rest is never said nor labelled
        u2, w2 = Utterance(k, 0), LX.Words()
        u2.tick(0, w2)
        u2.tick(1, w2)
        st = u2.state()
        left = u2.cut()
        assert 0 <= left <= 3, f"the cut takes {left} ticks more"
        seq = [u2.tick(t, w2) for t in range(2, 2 + left + 2)]
        said = [x for x in k.words if x[1] < u2.stop_at]
        assert u2.done and all(e <= u2.stop_at for _, _, e in said), (said, u2.stop_at)
        labels = []
        for t in range(0, 12):
            s = LX.TABLE[w2.tick(t)]
            if s not in ("<rest>",):
                labels.append(s)
        assert labels == [x[0] for x in said] + ["<end>"], (labels, said)
        u3 = Utterance.restore(k, st)
        u3.cut()
        seq3 = [u3.tick(t) for t in range(2, 2 + left + 2)]
        assert all(np.array_equal(a, b) for a, b in zip(seq, seq3)), "a restored line did not continue exactly"
        c.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"10 a line played tick by tick is its clip; each token arrives on its word's end tick (or one late); a talk-over "
          f"cut finishes the word ({left} ticks) and labels only what was said: {labels}; a restored line continues exactly")


TESTS = [test_tract_vowels, test_tract_deterministic, test_closure_stops_voicing, test_rest_is_silent_and_breath, test_tract_cost,
         test_table_and_channel, test_engine_bit_for_bit, test_cache_and_ledger, test_words_registers_and_level,
         test_playback_into_the_words_channel]

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
