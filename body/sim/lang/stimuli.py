"""A FORMAL TRIAL'S STIMULI, TIME-MATCHED BY CONSTRUCTION (docs/SIM_DESIGN.md 4.8, 12; P3's twelfth round: the lead's decision on
C67, as infant labs match their stimuli).

Every sentence a trial's draw could give (each thing's "where is the X?", a combination's "where is the C X?" of each colour, the
name and each foil) is said on one IDENTICAL timeline: one carrier phrase a form, and its test word said at its own rate (the
engine's own per-word rate, whose steps tools/sim_voice_check.py --trial measures) and pitch (its contour matched), so the test
word's onset and end, the sentence's end, the ticks her voice sounds, and what the words channel carries fall on the same ticks
whichever is named. The recipe is measured before birth and kept in trial_lines.json (tools/sim_voice_check.py --trial --write);
the conduct holds every trial to it on the rendered audio (timeline(), same()), not to the table's word: a set whose timelines
differ in any way is not used for a trial (the probe is dropped and logged).

  timeline(words, n_samples, pcm, scaffold, slot)  what a child can time a look, a sound or an act from, in ticks (150 ms):
      ticks     the clip's ticks: her voice sounds, and the world's playback runs, for exactly these
      words     each word's first tick and last tick (the engine's marks, a word's end at its last 10 ms frame within 40 dB of
                the clip's loudest: synth.words_of; the test word's onset opens the window), and the words themselves but the
                test word's (the carrier)
      sound     each tick "loud" (its RMS above TRIAL_LOUD_DB, within 20 dB of her speech's level) or "silent" (its loudest
                10 ms below TRIAL_SILENT_DB, the edge of hearing at a metre), else "?": a tick's RMS is at most its loudest 10
                ms, so her voice to the ear (10 ms by 10 ms) and her mouth to the eye (the tick's loudness the face shows,
                playback.Utterance.mouth) start, stop and pause on the same ticks for a child that takes any level in that
                band for silence or a closed mouth; a "?" tick is a timeline no level makes one, and matches nothing
      channel   while the word scaffold labels her lines (A29: from birth until its removal test), the ticks the words
                channel carries a symbol and the tick it carries END (lexicon.Words, as the world's playback hands them: a birth
                word's token on its word's last tick, a later word's letters one a tick from its onset, then a space; END on the
                line's last tick, after them): a token and a word spelled in letters, or two spelled words of different
                lengths, never share one
  same(timelines) -> None, or why not
  STIMULI, shape(form, word) -> ((word, rate, pitch), None) or (None, why): the table's recipe for a test word
"""
import json
import math
import os

import numpy as np

from . import consts as K
from . import lexicon as LX
from ..voice.playback import TICK
from ..voice.synth import frame_rms

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


def timeline(words, n_samples, pcm=None, scaffold=False, slot=None):
    """a sentence as a child can time from it (module doc) -> dict(ticks, words, carrier, sound, channel). words: [(word, first
    sample, end sample)]; slot: the test word's index (its word left out of the carrier)."""
    n = int(math.ceil(int(n_samples) / TICK))
    return dict(ticks=n, words=tuple((int(a) // TICK, (max(int(e), 1) - 1) // TICK) for _w, a, e in words),
                carrier=tuple("_" if i == slot else w for i, (w, _a, _e) in enumerate(words)),
                sound=profile(pcm), channel=channel(words, n_samples) if scaffold else None)


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
        for f in ("ticks", "words", "carrier", "sound", "channel"):
            if tl[f] != ref[f]:
                return f"{names[0]!r} and {nm!r} differ in their {f}: {_short(ref[f])} against {_short(tl[f])}"
    return None


def _short(v):
    s = repr(v)
    return s if len(s) <= 160 else s[:157] + "..."
