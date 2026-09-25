"""A FORMAL TRIAL'S STIMULI, TIME-MATCHED BY CONSTRUCTION (docs/SIM_DESIGN.md 4.8, 12; P3's twelfth round: the lead's decision on
C67, as infant labs match their stimuli; its thirteenth: one carrier recording a form; its fourteenth: one carrier phrase around
the test word, its tag after it one recording too, and the test word never louder than either).

Every sentence a trial's draw could give (each thing's "where is the X? see?", a combination's "where is the C X? see?" of each
colour, "hi. pip. hi." and each foil's) is one sentence built as labs build theirs, by splicing the test word into ONE CARRIER
PHRASE a form (splice):
  pre   the carrier before it: one recording (the form's carrier word's sentence, or for the name "hi."), its samples before the
        test word's onset tick
  own   the test word: its own sentence's samples from its onset tick through its last loud tick and one silent tick after it,
        the word said at its own rate (the engine's own per-word rate, whose steps tools/sim_voice_check.py --trial measures)
        and pitch (its contour matched), and at its own level (gain): every test word of the form at one loudest 10 ms, the
        form's TARGET, set under the carrier's and the tag's loudest moments by TRIAL_CEILING_DB at the clip and
        TRIAL_CEILING_EAR_DB at the child's own ears (the level ceiling)
  tag   the carrier after it: one recording ("see?", or for the name "hi."), whole, from the tick after that silent tick
So everything she sounds before the test word's onset tick and from the tag on is the same, sample for sample, whichever is
named, and the test word's slot between (its ticks, loud or silent alike) is never as loud as the loudest moment before it or
after it. Her voice's start and her sound's stop at every level of her sound as a whole (each window of her sound from 2.5 ms
to a tick, her mouth as the tick's loudness, and the child's own ears summed over their 40 bands, 25 ms frames, near or far,
either ear) therefore fall in the carrier or the tag: the same moment whichever is named (to the sample at the clip; at the
ears, to their own float rounding, under 1e-6 dB). Band by band at its ears they do not: the tag has little energy in many of
the cochlea's bands, so in most of them her start and stop fall inside the slot and differ as the words' spectra do (P3's
fourteenth verifier; docs/SIM_DESIGN.md C69b), which the trial's measure since the lead's decision A60b, the proportion of
looking over a fixed window, never times anything from. The recipe is measured before birth and kept in trial_lines.json
(tools/sim_voice_check.py --trial --write, the ears' ceiling measured there); the conduct holds every trial to it on the
rendered audio (timeline(), same()): a set whose timelines differ in any way, or a sentence whose slot is not under its
ceiling at the clip, is not used for a trial (the probe is dropped and logged).

What this does not match, and cannot: the test word's own slot. Inside it two sentences differ as their words do (their
segments, the moments within the slot at which each word's own sound rises and falls, 10 ms by 10 ms, its spectrum), and a
process whose memory reaches back into the slot (a loudness integrator over seconds) carries that on. A child that times an act
from inside the slot hears the word said: docs/SIM_DESIGN.md 4.8, C68, C69.

  timeline(words, n_samples, pcm, scaffold, slot, pre, tag_at)  what a child can time from on the tick grid (150 ms), which
      every rule of the trial reads:
      ticks     the clip's ticks: her voice sounds, and the world's playback runs, for exactly these
      words     each word's first tick and last tick (the engine's marks, a word's end at its last 10 ms frame within 40 dB of
                the clip's loudest: synth.ends_of; the test word's onset opens the window), and the words themselves but the
                test word's (the carrier)
      pre       the digest of every sample before the test word's onset tick (the carrier's)
      post      the digest of every sample from the tag's first tick (the tag's), and that tick (tag)
      sound     each tick of the test word's slot "loud" (its RMS above TRIAL_LOUD_DB, within 20 dB of her speech's level) or
                "silent" (its loudest 10 ms below TRIAL_SILENT_DB, the edge of hearing at a metre), else "?": a "?" tick is a
                timeline no level makes one, and matches nothing (the ticks outside the slot are the same by their digests)
      ceiling   the slot under its ceiling at the clip (headroom() > 0: every window of her sound from 2.5 ms to a tick that
                overlaps the slot quieter than the loudest wholly before it and the loudest wholly after it)
      channel   while the word scaffold labels her lines (A29: from birth until its removal test), the ticks the words
                channel carries a symbol and the tick it carries END (lexicon.Words, as the world's playback hands them: a birth
                word's token on its word's last tick, a later word's letters one a tick from its onset, then a space; END on the
                line's last tick, after them): a token and a word spelled in letters, or two spelled words of different
                lengths, never share one
  same(timelines) -> None, or why not
  STIMULI, shape(form, word) -> ((word, rate, pitch), None) or (None, why): the table's recipe for a test word
  parts(form, word) -> the recordings a sentence is built of and where (pre, own, tag, at, own_from, n_own, gain), or None
  headroom(pcm, at, tag_at) -> the level ceiling's least margin at the clip, dB
  splice(pre_clip, own_clip, tag_clip, at, own_from, n_own, gain_db, text) -> the sentence on its carrier phrase
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
WINDOWS = (40, 80, 160, 320, 400, 800, TICK)  # the level ceiling's windows at the clip, in samples: 2.5, 5, 10, 20, 25 (the ears'
                                              # frame) and 50 ms, and a tick (her mouth, playback.Utterance.mouth)
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
    if not f.get("pre") or not f.get("tag") or not f.get("own") or "gain" not in w:
        return None, f"the form {key!r} has no carrier phrase in {K.TRIAL_FILE} (built before P3's fourteenth round: fail-closed)"
    return (word, float(w["rate"]), float(w["pitch"])), None


def parts(key, word):
    """the recordings a test sentence of a form is built of (splice) -> dict(pre, own, tag: (text, register, emphasis, shape),
    at: the test word's onset tick's first sample, own_from: the same tick's first sample in its own sentence, n_own: its slot's
    samples, gain: its level, dB), or None (no such word or form in the table: fail-closed)."""
    f = STIMULI.get(key)
    shp, _why = shape(key, word)
    if f is None or shp is None:
        return None
    p, g = f["pre"], f["tag"]
    own = (f["text"].replace("{w}", word), f["register"], word if f["emphasis"] == "{w}" else f["emphasis"], shp)
    pre = (p["text"], p["register"], p.get("emphasis"), None if p.get("shape") is None else tuple(p["shape"]))
    tag = (g["text"], g["register"], g.get("emphasis"), None)
    return dict(pre=pre, own=own, tag=tag, at=int(p["at"]) * TICK, own_from=int(f["own"]["from"]) * TICK,
                n_own=int(f["own"]["ticks"]) * TICK, gain=float(f["words"][word]["gain"]))


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


def _peak(c, w, lo, hi, overlap):
    """the loudest window of w samples (its RMS, dB of full scale) wholly within [lo, hi), or (overlap) overlapping it, from the
    cumulative energy c; -inf when there is none."""
    n = len(c) - 1
    a, b = (max(0, lo - w + 1), min(n - w, hi - 1)) if overlap else (max(0, lo), min(n, hi) - w)
    if b < a:
        return -math.inf
    s = (c[a + w:b + w + 1] - c[a:b + 1]) / w
    return 10.0 * math.log10(max(float(s.max()), 1e-30))


def headroom(pcm, at, tag_at, windows=WINDOWS):
    """the level ceiling at the clip (P3's fourteenth round): over windows of her sound from 2.5 ms to a tick, the least margin
    (dB) by which every window that overlaps the test word's slot [at, tag_at) is quieter than the loudest wholly before it (the
    carrier) and the loudest wholly after it (the tag) -> dB (positive: the ceiling holds, so at every level her sound's first
    and last moments above it, her sound as a whole, lie in the carrier and the tag, the same whichever is named; band by band
    at the child's ears they need not: C69b); per window, -inf where a part is
    empty. A single sample, or a window under 2.5 ms, is no measure the child has (its ears read 25 ms windows 10 ms apart)."""
    x = np.asarray(pcm, np.float64) / 32767.0
    c = np.concatenate([[0.0], np.cumsum(x * x)])
    out = math.inf
    for w in windows:
        slot = _peak(c, w, at, tag_at, True)
        out = min(out, min(_peak(c, w, 0, at, False), _peak(c, w, tag_at, len(x), False)) - slot)
    return out


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


def _digest(pcm, a, b):
    return hashlib.sha256(np.asarray(pcm, "<i2")[a:b].tobytes()).hexdigest()[:24]


def timeline(words, n_samples, pcm=None, scaffold=False, slot=None, pre=True, tag_at=None):
    """a sentence as a child can time from it (module doc) -> dict(ticks, words, carrier, sound, channel[, pre][, post, tag,
    ceiling]). words: [(word, first sample, end sample)]; slot: the test word's index (its word left out of the carrier);
    pre=False leaves the carrier's samples out (the tool's search for a word's rate, before its sentence is spliced); tag_at:
    the tag's first sample (a sentence on its carrier phrase: its slot's ticks alone held to the band, the samples from the tag
    on to their digest, the slot to its ceiling)."""
    n = int(math.ceil(int(n_samples) / TICK))
    out = dict(ticks=n, words=tuple((int(a) // TICK, (max(int(e), 1) - 1) // TICK) for _w, a, e in words),
               carrier=tuple("_" if i == slot else w for i, (w, _a, _e) in enumerate(words)),
               sound=profile(pcm), channel=channel(words, n_samples) if scaffold else None)
    at = (int(words[slot][1]) // TICK) * TICK if slot is not None and slot < len(words) else 0
    if pre:
        out["pre"] = None if pcm is None else _digest(pcm, 0, at) + f":{at}"
    if tag_at is not None:
        tag_at = int(tag_at)
        out["tag"] = tag_at // TICK
        out["sound"] = out["sound"][at // TICK:tag_at // TICK]
        out["post"] = None if pcm is None else _digest(pcm, tag_at, len(pcm)) + f":{tag_at}"
        out["ceiling"] = None if pcm is None else bool(headroom(pcm, at, tag_at) > 0.0)
    return out


def form_of(line):
    """a trial sentence's form in the table, from its intent and focus: "where", "name", "combination:<noun>" (its frame's focus
    is its noun)."""
    if line.intent == "trial_name":
        return "name"
    if line.intent == "trial_combo":
        return "combination:" + str(line.focus)
    return "where"


def splice(pre_clip, own_clip, tag_clip, at, own_from, n_own, gain_db=0.0, text=None, fade=160):
    """a trial sentence on its form's one carrier phrase (P3's fourteenth round), as labs splice a test word into one recorded
    carrier: the pre recording's samples before `at` (the test word's onset tick's first sample), then its own sentence's
    n_own samples from own_from (the same tick in it: the word, then silence) at its gain, a 10 ms raised cosine from the one
    into the other at the start of that tick, then the tag recording whole. Its words: those the pre recording began before
    `at`, those its own sentence began within its slot, and the tag's, each word's end found again on the spliced samples as the
    voice finds it (synth.ends_of) -> a clip of the sentence's own kind, its text `text`, its key and digest its own, and
    `spliced` (where each part lies). Refused (ValueError), fail-closed: an own sentence whose words before its slot are not the
    pre's (not one carrier), one with a word begun after its slot or any sound after it (it would be cut), parts at different
    levels of her voice (their registers' gains), and words that are not `text`'s."""
    a, b, c = (np.asarray(x.pcm) for x in (pre_clip, own_clip, tag_clip))
    at, own_from, n_own = int(at), int(own_from), int(n_own)
    before = [(w, int(on)) for w, on, _e in pre_clip.words if int(on) < at]
    own_before = [w for w, on, _e in own_clip.words if int(on) < own_from]
    inside = [(w, int(on) - own_from + at) for w, on, _e in own_clip.words if own_from <= int(on) < own_from + n_own]
    after = [w for w, on, _e in own_clip.words if int(on) >= own_from + n_own]
    tag_at = at + n_own
    tags = [(w, int(on) + tag_at) for w, on, _e in tag_clip.words]
    why = None
    if own_from > 0 and own_before != [w for w, _on in before]:
        why = f"its words before its slot {own_before} against the carrier's {[w for w, _on in before]}"
    elif not inside or after:
        why = f"its words in its slot {[w for w, _on in inside]}, after it {after}"
    elif len(b) > own_from + n_own and 20.0 * math.log10(max(float(frame_rms(b[own_from + n_own:]).max()), 1e-12)) >= \
            K.TRIAL_SILENT_DB:
        why = f"its sound runs past its slot (its loudest 10 ms after it at or above {K.TRIAL_SILENT_DB:g} dB): it would be cut"
    elif len({float(getattr(x, 'gain_db', 0.0)) for x in (pre_clip, own_clip, tag_clip)}) > 1:
        why = "its parts at different levels of her voice (their registers)"
    if why is not None:
        raise ValueError(f"{getattr(own_clip, 'text', '?')!r} is not on the carrier {getattr(pre_clip, 'text', '?')!r} ... "
                         f"{getattr(tag_clip, 'text', '?')!r}: {why}")
    x_pre = np.zeros(at + fade)
    seg = a[:at + fade].astype(np.float64)
    x_pre[:len(seg)] = seg
    x_own = np.zeros(n_own)
    seg = b[own_from:own_from + n_own].astype(np.float64)
    x_own[:len(seg)] = seg
    x_own *= 10.0 ** (float(gain_db) / 20.0)
    r = 0.5 - 0.5 * np.cos(np.pi * (np.arange(fade) + 0.5) / fade)
    x_own[:fade] = x_pre[at:at + fade] * (1.0 - r) + x_own[:fade] * r
    x = np.concatenate([x_pre[:at], x_own, c.astype(np.float64)])
    pcm = np.clip(np.round(x), -32768, 32767).astype(np.int16)
    words = [(w, int(on), int(e)) for w, on, e in ends_of(pcm, before + inside + tags)]
    if text is not None and [w for w, _on, _e in words] != re.findall(r"[a-z]+", text.lower()):
        raise ValueError(f"the spliced sentence's words {[w for w, _on, _e in words]} are not {text!r}'s")
    out = copy.copy(own_clip)
    out.pcm, out.words = pcm, words
    if text is not None:
        out.text = text
    out.digest = clip_digest(pcm, [list(x) for x in words])
    out.key = hashlib.sha256(f"splice2:{pre_clip.key}:{own_clip.key}:{tag_clip.key}:{at}:{own_from}:{n_own}:"
                             f"{float(gain_db)!r}:{int(fade)}".encode()).hexdigest()
    out.spliced = dict(pre=pre_clip.key, own=own_clip.key, tag=tag_clip.key, at=at, own_from=own_from, n_own=n_own,
                       tag_at=tag_at, gain_db=float(gain_db), fade=int(fade))
    if isinstance(getattr(own_clip, "meta", None), dict):
        out.meta = dict(own_clip.meta, spliced=out.spliced)
    return out


def same(tls, names=None):
    """the timelines of every sentence a trial's draw could give -> None when they are one (no tick of any test word's slot is
    between loud and silent, and every slot is under its ceiling), else why not."""
    names = names or [str(i) for i in range(len(tls))]
    for nm, tl in zip(names, tls):
        if "tag" in tl and tl["sound"] and tl["sound"][-1] != "silent":
            return f"{nm!r}: its test word's slot does not end silent (its word runs to its tag)"
        if "?" in (tl["sound"] or ()):
            k = tl["sound"].index("?")
            return (f"{nm!r}: tick {k} of its {'slot' if 'tag' in tl else 'sentence'} neither loud nor silent (its RMS at most "
                    f"{K.TRIAL_LOUD_DB:g} and its loudest 10 ms at least {K.TRIAL_SILENT_DB:g} dB of full scale)")
        if tl.get("ceiling") is False:
            return f"{nm!r}: its test word's slot not under its ceiling (as loud as her carrier or her tag somewhere)"
    ref = tls[0]
    for nm, tl in zip(names[1:], tls[1:]):
        for f in ("ticks", "words", "carrier", "pre", "tag", "post", "sound", "ceiling", "channel"):
            if tl.get(f) != ref.get(f):
                return f"{names[0]!r} and {nm!r} differ in their {f}: {_short(ref.get(f))} against {_short(tl.get(f))}"
    return None


def _short(v):
    s = repr(v)
    return s if len(s) <= 160 else s[:157] + "..."
