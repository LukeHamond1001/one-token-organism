"""A FORMAL TRIAL'S STIMULI, TIME-MATCHED BY CONSTRUCTION (docs/SIM_DESIGN.md 4.8, 12; P3's twelfth round: the lead's decision on
C67, as infant labs match their stimuli; its thirteenth: one carrier recording a form, spliced).

Every sentence a trial's draw could give (each thing's "where is the X?", a combination's "where is the C X?" of each colour, the
name and each foil) is said on one timeline: ONE CARRIER RECORDING a form, its samples before the test word's onset tick taken
by every sentence of the form (splice: the table's carrier word's sentence, as labs splice one recorded carrier phrase onto each
test word), and its test word said at its own rate (the engine's own per-word rate, whose steps tools/sim_voice_check.py --trial
measures) and pitch (its contour matched), so the test word's onset and end, the sentence's end, the ticks her voice sounds,
and what the words channel carries fall on the same ticks whichever is named, and nothing she sounds before the test word's
onset tick differs by a sample. The recipe is measured before birth and kept in trial_lines.json (tools/sim_voice_check.py
--trial --write); the conduct holds every trial to it on the rendered audio (timeline(), same()), not to the table's word: a set
whose timelines differ in any way is not used for a trial (the probe is dropped and logged).

What this does not match, and cannot: from the test word's onset tick to her sound's end (and one tick more, the child's ear:
TRIAL_EAR_TICKS) two sentences differ as any two words do, in what the child hears 10 ms by 10 ms, at a level other than the
band's two, and through its own ears (P3's twelfth verifier: her sound's last frame above -36 dB is frame 113 of "ball" and 107
of "block", both in tick 7; at the ear, 0.7 m away, "car" and "drum" stop a tick later than "ball" at -48 and -60 dB; tick 7 of
"ball" is at -27.6 dB, of "block" -33.5). A child that times an act from such a moment tells the words apart by their sound.
The trial's scoring, not the stimuli, makes that count for nothing when the child does not know which thing a word names
(ledger: every trial is scored once its window opens, its none or void against it; the name test's foil held to no turn over
its window and this span): see docs/SIM_DESIGN.md 4.8, C67.

  timeline(words, n_samples, pcm, scaffold, slot)  what a child can time from on the tick grid (150 ms), which every rule of the
      trial reads:
      ticks     the clip's ticks: her voice sounds, and the world's playback runs, for exactly these
      words     each word's first tick and last tick (the engine's marks, a word's end at its last 10 ms frame within 40 dB of
                the clip's loudest: synth.ends_of; the test word's onset opens the window), and the words themselves but the
                test word's (the carrier)
      pre       the digest of every sample before the test word's onset tick (the carrier recording's, spliced): nothing a
                child hears or sees of her before its window opens differs, at any level or precision
      sound     each tick "loud" (its RMS above TRIAL_LOUD_DB, within 20 dB of her speech's level) or "silent" (its loudest
                10 ms below TRIAL_SILENT_DB, the edge of hearing at a metre), else "?": a tick's RMS is at most its loudest 10
                ms, so her voice (10 ms by 10 ms) and her mouth (the tick's loudness the face shows, playback.Utterance.mouth)
                start, stop and pause on the same ticks at the band's two levels; a "?" tick is a timeline no level makes one,
                and matches nothing
      channel   while the word scaffold labels her lines (A29: from birth until its removal test), the ticks the words
                channel carries a symbol and the tick it carries END (lexicon.Words, as the world's playback hands them: a birth
                word's token on its word's last tick, a later word's letters one a tick from its onset, then a space; END on the
                line's last tick, after them): a token and a word spelled in letters, or two spelled words of different
                lengths, never share one
  same(timelines) -> None, or why not
  STIMULI, shape(form, word) -> ((word, rate, pitch), None) or (None, why): the table's recipe for a test word
  form_of(line), carrier(form) -> the form's one carrier recording (its sentence, register, emphasis, shape, and the sample
      its test word's onset tick begins at), splice(carrier_clip, clip, at) -> the sentence on it
"""
import copy
import hashlib
import json
import math
import os
import re

import numpy as np

from . import consts as K
from . import lexicon as LX
from ..voice.playback import TICK
from ..voice.synth import clip_digest, ends_of, frame_rms

FRAMES_A_TICK = TICK // 160                   # the voice's 10 ms frames (synth.HOP) in a tick: 15
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), K.TRIAL_FILE)
STIMULI, STIMULI_META = {}, {}
if os.path.exists(PATH):
    with open(PATH) as _fh:
        _d = json.load(_fh)
    STIMULI.update(_d["forms"])
    STIMULI_META.update(_d["meta"])


def form_key(form, noun=None):
    """a trial form -> its stimuli's key in the table: "where" (place, exemplar), "name", "combination:<noun>"."""
    if form in ("place", "exemplar"):
        return "where"
    if form == "combination":
        return f"combination:{noun}"
    return form


def shape(key, word):
    """the table's recipe for a test word of a form -> ((word, rate, pitch), None), or (None, why): its rate (percent of the
    engine's default rate) and pitch (percent over the line's), the voice's shape (synth.line_ssml)."""
    f = STIMULI.get(key)
    if f is None:
        return None, f"no stimuli built for the form {key!r} ({K.TRIAL_FILE}: tools/sim_voice_check.py --trial)"
    w = f["words"].get(word)
    if w is None:
        why = f.get("unmatched", {}).get(word, "not measured")
        return None, f"{word!r} has no time-matched stimulus in her voice ({K.TRIAL_FILE}: {why})"
    return (word, float(w["rate"]), float(w["pitch"])), None


def levels(pcm):
    """each tick of a clip -> (its RMS, its loudest 10 ms frame's RMS), dB of the engine's full scale."""
    e = frame_rms(np.asarray(pcm))
    n = int(math.ceil(len(pcm) / TICK))
    e = np.pad(e, (0, max(0, n * FRAMES_A_TICK - len(e))))[:n * FRAMES_A_TICK].reshape(n, FRAMES_A_TICK)
    rms = np.sqrt((e ** 2).mean(1))                          # (the frames are 10 ms each: the tick's RMS)
    return 20.0 * np.log10(np.maximum(rms, 1e-12)), 20.0 * np.log10(np.maximum(e.max(1), 1e-12))


def profile(pcm):
    """each tick of a clip "loud" (its RMS above TRIAL_LOUD_DB), "silent" (its loudest 10 ms below TRIAL_SILENT_DB) or "?"."""
    if pcm is None or not len(pcm):
        return ()
    rms, top = levels(pcm)
    return tuple("loud" if a > K.TRIAL_LOUD_DB else "silent" if b < K.TRIAL_SILENT_DB else "?" for a, b in zip(rms, top))


def channel(words, n_samples):
    """the words channel's timeline for a line said whole (lexicon.Words, as the world's playback hands its words): the ticks it
    carries a symbol ("sym") and END ("end"), from the line's first tick."""
    ch = LX.Words()
    ch.push([(w, int(a), int(e)) for w, a, e in words], 0, int(n_samples))
    out, t = [], 0
    while ch.queue and t < 10 ** 4:
        s = ch.tick(t)
        if s != LX.ID[LX.REST]:
            out.append((t, "end" if s == LX.ID[LX.END] else "sym"))
        t += 1
    return tuple(out)


def timeline(words, n_samples, pcm=None, scaffold=False, slot=None, pre=True):
    """a sentence as a child can time from it (module doc) -> dict(ticks, words, carrier, pre, sound, channel). words: [(word,
    first sample, end sample)]; slot: the test word's index (its word left out of the carrier); pre=False leaves the carrier's
    samples out (the tool's search for a word's rate, before its sentence is spliced)."""
    n = int(math.ceil(int(n_samples) / TICK))
    out = dict(ticks=n, words=tuple((int(a) // TICK, (max(int(e), 1) - 1) // TICK) for _w, a, e in words),
               carrier=tuple("_" if i == slot else w for i, (w, _a, _e) in enumerate(words)),
               sound=profile(pcm), channel=channel(words, n_samples) if scaffold else None)
    if pre:
        at = (int(words[slot][1]) // TICK) * TICK if slot is not None and slot < len(words) else 0
        out["pre"] = None if pcm is None else \
            hashlib.sha256(np.asarray(pcm, "<i2")[:at].tobytes()).hexdigest()[:24] + f":{at}"
    return out


def form_of(line):
    """a trial sentence's form in the table, from its intent and words: "where", "name", "combination:<noun>"."""
    if line.intent == "trial_name":
        return "name"
    if line.intent == "trial_combo":
        return "combination:" + re.findall(r"[a-z]+", line.text.lower())[-1]
    return "where"


def carrier(key):
    """the form's one carrier recording -> (its sentence, register, emphasis, shape, at) or None (no carrier before the test
    word, as the name's; or none in the table: then no set of the form is one timeline, fail-closed): the table's carrier
    word's sentence, whose samples before `at`, the first sample of its test word's onset tick, every sentence of the form
    takes (splice)."""
    f = STIMULI.get(key)
    if f is None or not f.get("carrier") or not f.get("timeline"):
        return None
    w = f["carrier"]
    rec = f["words"].get(w)
    at = int(f["timeline"]["words"][f["slot"]][0]) * TICK
    if rec is None or at <= 0:
        return None
    return (f["text"].replace("{w}", w), f["register"], w if f["emphasis"] == "{w}" else f["emphasis"],
            (w, float(rec["rate"]), float(rec["pitch"])), at)


def splice(carrier_clip, clip, at, fade=160):
    """a trial sentence on its form's one carrier recording: the carrier's samples before `at` (its test word's onset tick's
    first sample), the sentence's own from there, a 10 ms raised cosine from the one into the other at the start of that tick
    (inside the window: nothing before the tick differs, whichever is named). Its words: those begun before `at` by the
    carrier's marks, the rest by its own, each word's end found again on the spliced samples as the voice finds it
    (synth.ends_of) -> a clip of the sentence's own kind, its key and digest its own. A sentence and a carrier whose words
    before `at` differ are refused (ValueError): they are not one carrier."""
    a, b = np.asarray(carrier_clip.pcm), np.asarray(clip.pcm)
    before = [(w, int(on)) for w, on, _e in carrier_clip.words if int(on) < at]
    after = [(w, int(on)) for w, on, _e in clip.words if int(on) >= at]
    own = [w for w, on, _e in clip.words if int(on) < at]
    if [w for w, _on in before] != own or len(a) < at + fade or len(b) < at + fade:
        raise ValueError(f"{getattr(clip, 'text', '?')!r} is not on the carrier {getattr(carrier_clip, 'text', '?')!r}: "
                         f"its words before sample {at} {own} against {[w for w, _on in before]}")
    r = 0.5 - 0.5 * np.cos(np.pi * (np.arange(fade) + 0.5) / fade)
    mid = np.round(a[at:at + fade].astype(np.float64) * (1.0 - r) + b[at:at + fade].astype(np.float64) * r)
    pcm = np.concatenate([a[:at], np.clip(mid, -32768, 32767).astype(a.dtype), b[at + fade:]])
    words = ends_of(pcm, before + after)
    out = copy.copy(clip)
    out.pcm, out.words = pcm, [(w, int(on), int(e)) for w, on, e in words]
    out.digest = clip_digest(pcm, [list(x) for x in out.words])
    out.key = hashlib.sha256(f"splice:{carrier_clip.key}:{clip.key}:{int(at)}:{int(fade)}".encode()).hexdigest()
    if isinstance(getattr(clip, "meta", None), dict):
        out.meta = dict(clip.meta, spliced=dict(carrier=carrier_clip.key, clip=clip.key, at=int(at), fade=int(fade)))
    return out


def same(tls, names=None):
    """the timelines of every sentence a trial's draw could give -> None when they are one (and no tick of any is between loud
    and silent), else why not."""
    names = names or [str(i) for i in range(len(tls))]
    for nm, tl in zip(names, tls):
        if "?" in (tl["sound"] or ()):
            k = tl["sound"].index("?")
            return (f"{nm!r}: tick {k} neither loud nor silent (its RMS at most {K.TRIAL_LOUD_DB:g} and its loudest 10 ms at "
                    f"least {K.TRIAL_SILENT_DB:g} dB of full scale)")
    ref = tls[0]
    for nm, tl in zip(names[1:], tls[1:]):
        for f in ("ticks", "words", "carrier", "pre", "sound", "channel"):
            if tl.get(f) != ref.get(f):
                return f"{names[0]!r} and {nm!r} differ in their {f}: {_short(ref[f])} against {_short(tl[f])}"
    return None


def _short(v):
    s = repr(v)
    return s if len(s) <= 160 else s[:157] + "..."
