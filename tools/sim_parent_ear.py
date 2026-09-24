"""THE PARENT'S EAR, BUILT AND MEASURED (docs/SIM_DESIGN.md 4.9, A27, 11's P3v; package P3): an instrument and the ear's builder,
never part of a body. Nothing it measures reaches the child; the margins it measures are the teacher's method, set before birth.

  build     her templates (consts.EAR_VOICES: each word said alone by 11 voices, through the life's voice cache, so each clip is
            in its ledger) and the babble bank (the babbler's utterances, as she hears them through the transcriber at 1 m), into
            <root>/parent_ear.npz with its digest
  margins   m for each expected-set size (A27: held-out babble passes as the context word at most 2% of the time), from the
            held-out babbler's utterances against random context sets of her words; written into the ear with --write-margins
  deltas    delta for each size (the ear's P3 test): the held-out voices' words she does not expect pass as the context
            word at most 2% of the time, as babble does for m
  voices    each held-out voice's words (its own templates removed) accepted in context at those margins, and exact; the same
            words when she does not expect them (accepted as another word, the rule's own false acceptance); with and without
            delta
  cost      her ear per utterance (the expected words and the bank; her other words only when a word is accepted), and the
            transcriber per tick over the babble stream (listening on every tick, hearing at each turn's end)

THE BABBLER is the voice study's ($S/g1/voice/t_reject.py, babble_utts): each tick a movement unit starts with probability 1 / 14.5,
lasting 1-8 ticks (uniform), each articulator's step drawn uniformly from the five and held through the unit; the tract otherwise
rests. It stands in for W4's babbler (the born body's own motor timing on the G1, section 11 day 5), which is to record the bank
before birth (P3v): the bank, and the margins measured on it, are this stand-in's until then. Its seeds: the bank 8 (12,000
ticks), held-out 7 (4,000 ticks), as in the study; the tract's own stream is the babbler's seed (a pre-birth instrument, not the
life's seed 1).

Run: nice -n 19 python3 tools/sim_parent_ear.py --root DIR [--words birth|all] [--write-margins] [--out results.json]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from body.sim import parent_ear as PE  # noqa: E402
from body.sim import tract as T  # noqa: E402
from body.sim.lang import consts as K  # noqa: E402
from body.sim.lang import lexicon as LX  # noqa: E402
from body.sim.lang import templates as TP  # noqa: E402
from body.sim.lang.percept import Percept  # noqa: E402
from body.sim.lang.transcriber import Transcriber  # noqa: E402

BANK_SEED, BANK_TICKS = 8, 12000
HELD_SEED, HELD_TICKS = 7, 4000
UNIT_P, UNIT_MAX = 1 / 14.5, 8
SIZES = tuple(range(1, 9))


def babble(seed, ticks):
    """the study's babbler: -> [samples per tick] (engine units at 1 m)."""
    rng = np.random.default_rng(seed)
    tr = T.Tract(seed)
    out, k = [], 0
    while k < ticks:
        if rng.random() < UNIT_P:
            L = int(rng.integers(1, UNIT_MAX + 1))
            a = rng.integers(0, 5, T.N_ART)
            for _ in range(min(L, ticks - k)):
                out.append(tr.tick(a))
                k += 1
        else:
            out.append(tr.tick(None))
            k += 1
    return out


def utterances(stream, distance=K.EAR_DISTANCE_M):
    """the stream as she hears it: [(start, end, x_pa)] by the transcriber's own turns (no ear)."""
    tx = Transcriber(None)
    p = Percept(0)
    out = []
    for t, x in enumerate(stream):
        got = tx.tick(t, x, distance, None, False, (), p)
        for cw in got:
            if cw.channel == "tract":
                out.append((cw.start, cw.end, tx.last_x))
    return out


def build(root, words, server_nice=0):          # the server inherits this process's nice (run it at nice 19)
    from body.sim.voice import synth as V                                   # noqa: PLC0415
    cache = V.VoiceCache(os.path.join(root, "voice"), server=V.SynthServer(nice=server_nice))
    t0 = time.perf_counter()
    tmpl = {w: [] for w in words}
    digests = []
    for lab, voice, pitch, rate in K.EAR_VOICES:
        for w in words:
            c = cache.clip(w, "ear", voice=voice, prosody=(pitch, rate))
            tmpl[w].append((lab, PE.template_of(c.pa())))
            digests.append([w, lab, c.key, c.digest])
    cache.close()
    t_tmpl = time.perf_counter() - t0
    t0 = time.perf_counter()
    bank_utts = utterances(babble(BANK_SEED, BANK_TICKS))
    bank = [PE.template_of(x) for _, _, x in bank_utts]
    t_bank = time.perf_counter() - t0
    meta = dict(words=list(words), voices=[list(v) for v in K.EAR_VOICES], clips=digests,
                bank=dict(babbler="the voice study's (tools/sim_parent_ear.py babble)", seed=BANK_SEED, ticks=BANK_TICKS,
                          utterances=len(bank), distance_m=K.EAR_DISTANCE_M),
                margins_source=K.EAR_M_SOURCE)
    ear = PE.ParentEar(tmpl, bank, K.EAR_M, K.EAR_DELTA, meta)
    ear.save(os.path.join(root, "parent_ear.npz"))
    return ear, dict(templates=sum(len(v) for v in tmpl.values()), t_templates_s=round(t_tmpl, 1), bank=len(bank),
                     t_bank_s=round(t_bank, 1), digest=ear.digest)


def per_template(ear, feats):
    """{word: distances to each of its templates, min over the shifts}."""
    out = {}
    for w in ear.words:
        st = ear._stacks[w]
        out[w] = np.min([PE.dtw(q, st) for q in feats], axis=0)
    return out


def margin_for(rel_nearest, n_trials, fa=K.EAR_FALSE_PASS, grid=0.05):
    """the most lenient m on a 0.05 grid with at most fa of the trials passing (rel < -m among the nearest)."""
    r = np.sort(np.asarray(rel_nearest))
    for m in np.round(np.arange(-10.0, 10.0 + grid / 2, grid), 2):
        if np.sum(r < -m) / n_trials <= fa:
            return float(m)
    return 10.0


def measure_margins(ear, words, rng, sets=200):
    held = utterances(babble(HELD_SEED, HELD_TICKS))
    rows = []
    for _, _, x in held:
        feats, n = ear.features(x)
        pt = per_template(ear, feats)
        d = {w: float(pt[w].min()) for w in words}
        rows.append((d, ear.bank_distance(feats)))
    out = {}
    order = {w: i for i, w in enumerate(words)}
    for k in SIZES:
        rel, n = [], 0
        for d, db in rows:
            for _ in range(sets):
                c = list(rng.choice(words, k, replace=False))
                n += 1
                if min(c, key=lambda w: (d[w], order[w])) == c[0]:
                    rel.append(d[c[0]] - db)
        m = margin_for(rel, n)
        out[k] = dict(m=m, pass_rate=float(np.sum(np.asarray(rel) < -m) / n), trials=n)
    return out, len(held), rows


def babble_with_delta(rows, words, margins, deltas, rng, sets=200):
    """the held-out babble passing as the context word with both tests (m and delta)."""
    order = {w: i for i, w in enumerate(words)}
    out = {}
    for k in SIZES:
        n = p = 0
        for d, db in rows:
            best = min(words, key=lambda v: (d[v], order[v]))
            for _ in range(sets):
                c = list(rng.choice(words, k, replace=False))
                n += 1
                if min(c, key=lambda w: (d[w], order[w])) == c[0] and d[c[0]] - db < -margins[k] and \
                        (c[0] == best or d[c[0]] - d[best] < deltas[k]):
                    p += 1
        out[k] = p / n
    return out


def voice_rows(ear, words, root):
    """each held-out voice's words: (voice, word, {word: distance with that voice's templates held out}, bank distance)."""
    from body.sim.voice import synth as V                                   # noqa: PLC0415
    cache = V.VoiceCache(os.path.join(root, "voice"), server=V.SynthServer(nice=0))
    rows = []
    for lab, voice, pitch, rate in K.EAR_VOICES[3:]:
        for w in words:
            x = cache.clip(w, "ear", voice=voice, prosody=(pitch, rate)).pa()
            feats, _ = ear.features(x)
            pt = per_template(ear, feats)
            d = {v: float(min((dv for dv, lv in zip(pt[v], ear.labels[v]) if lv != lab), default=math.inf)) for v in words}
            rows.append((lab, w, d, ear.bank_distance(feats)))
    cache.close()
    return rows


def measure_deltas(rows, words, margins, rng, sets=80, fa=K.EAR_FALSE_PASS, grid=0.05):
    """delta by size: the held-out voices' words she does not expect pass as the context word at most fa of the time (among
    the trials where the context word is the set's nearest and passes the bank test, an approximation needs d - d_best <
    delta; an exact one always passes, and counts against fa)."""
    order = {w: i for i, w in enumerate(words)}
    out = {}
    for k in SIZES:
        m, n, exact, rel = margins[k], 0, 0, []
        for _, w, d, db in rows:
            best = min(d[v] for v in words)
            others = [v for v in words if v != w]
            for _ in range(sets):
                c = list(rng.choice(others, k, replace=False))
                n += 1
                if min(c, key=lambda v: (d[v], order[v])) == c[0] and d[c[0]] - db < -m:
                    if d[c[0]] <= best:
                        exact += 1
                    else:
                        rel.append(d[c[0]] - best)
        r = np.asarray(rel)
        delta = 0.0
        for dl in np.round(np.arange(0.0, 10.0 + grid / 2, grid), 2):
            if (exact + np.sum(r < dl)) / n <= fa:
                delta = float(dl)
            else:
                break
        out[k] = dict(delta=delta, pass_rate=float((exact + np.sum(r < delta)) / n), exact_rate=exact / n, trials=n)
    return out


def measure_voices(rows, words, margins, deltas, rng, sets=40):
    """at the margins: each held-out voice's word in a random expected set of each size (accepted, exact), and not in it
    (accepted as another word: the false acceptance); nearest of all words (top-1 among all her words)."""
    order = {w: i for i, w in enumerate(words)}
    res = {k: dict(accepted=0, exact=0, n=0, unexpected_accepted=0, unexpected_exact=0, n_unexpected=0) for k in SIZES}
    top1 = dict(right=0, n=0)
    for _, w, d, db in rows:
        best_all = min(words, key=lambda v: (d[v], order[v]))
        top1["right"] += best_all == w
        top1["n"] += 1
        others = [v for v in words if v != w]

        def accept(c, k):
            near = min(c, key=lambda v: (d[v], order[v]))
            ok = d[near] - db < -margins[k] and (near == best_all or d[near] - d[best_all] < deltas[k])
            return (near if ok else None), near == best_all
        for k in SIZES:
            r = res[k]
            for _ in range(sets):
                got, ex = accept([w] + list(rng.choice(others, k - 1, replace=False)), k)
                r["n"] += 1
                r["accepted"] += got == w
                r["exact"] += got == w and ex
                got2, ex2 = accept(list(rng.choice(others, k, replace=False)), k)
                r["n_unexpected"] += 1
                r["unexpected_accepted"] += got2 is not None
                r["unexpected_exact"] += got2 is not None and ex2
    return res, top1


def measure_cost(ear, words, rng):
    stream = babble(HELD_SEED, HELD_TICKS)
    tx = Transcriber(ear)
    toys = [w for w in words if w in TP.TOYS]
    p = Percept(0)
    t0 = time.perf_counter()
    n_utt = 0
    for t, x in enumerate(stream):
        exp = tuple(sorted(set(rng.choice(toys, 2, replace=False)) | {"mama"}, key=words.index))   # 3 words: two names, mama
        got = tx.tick(t, x, 1.0, None, False, exp, p, vocab=words)
        n_utt += sum(cw.channel == "tract" for cw in got)
    wall = (time.perf_counter() - t0) * 1e3
    hear = np.array([ms for _, ms in tx.wall])
    return dict(ticks=len(stream), utterances=n_utt, per_minute=round(n_utt / (len(stream) * 0.15 / 60), 2),
                ms_per_tick=round(wall / len(stream), 3), hear_ms_mean=round(float(hear.mean()), 1),
                hear_ms_median=round(float(np.median(hear)), 1), hear_ms_p95=round(float(np.percentile(hear, 95)), 1),
                hear_ms_max=round(float(hear.max()), 1), listen_ms_per_tick=round((wall - hear.sum()) / len(stream), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--words", default="birth", choices=("birth", "all"))
    ap.add_argument("--write-margins", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip", default="", help="comma list of: margins, voices, cost")
    a = ap.parse_args()
    words = list(LX.BIRTH_WORDS) + (list(TP.GROWTH_WORDS) if a.words == "all" else [])
    os.makedirs(a.root, exist_ok=True)
    rng = np.random.default_rng(2)
    ear, info = build(a.root, words)
    print(f"built: {info}")
    res = dict(build=info)
    skip = set(a.skip.split(","))
    margins = dict(K.EAR_M)
    if "margins" not in skip:
        mm, n_held, babble_rows = measure_margins(ear, words, rng)
        res["margins"] = mm
        print(f"held-out babble: {n_held} utterances; m by expected-set size (2% of held-out babble passing as the context word):")
        for k, r in mm.items():
            print(f"  size {k}: m = {r['m']:+.2f} (passes {100 * r['pass_rate']:.2f}% of {r['trials']} trials)")
        margins = {k: r["m"] for k, r in mm.items()}
    if "voices" not in skip:
        rows = voice_rows(ear, words, a.root)
        before, _ = measure_voices(rows, words, margins, {k: 1e9 for k in SIZES}, np.random.default_rng(3))
        dd = measure_deltas(rows, words, margins, np.random.default_rng(4))
        deltas = {k: r["delta"] for k, r in dd.items()}
        res["deltas"] = dd
        print("delta by size (the held-out voices' words she does not expect pass as the context word at most 2%):")
        for k, r in dd.items():
            print(f"  size {k}: delta = {r['delta']:.2f} (passes {100 * r['pass_rate']:.2f}%, of which exact "
                  f"{100 * r['exact_rate']:.2f}%, of {r['trials']} trials)")
        vr, top1 = measure_voices(rows, words, margins, deltas, np.random.default_rng(5))
        if "margins" not in skip:
            bb = babble_with_delta(babble_rows, words, margins, deltas, np.random.default_rng(6))
            res["babble_with_delta"] = bb
            print("held-out babble passing as the context word with both tests: " +
                  ", ".join(f"{k}: {100 * v:.2f}%" for k, v in bb.items()))
        res["voices_without_delta"], res["voices"], res["top1"] = before, vr, top1
        print(f"the 8 other voices, each held out: nearest of all {len(words)} words right {100 * top1['right'] / top1['n']:.1f}%")
        for k in SIZES:
            r0, r = before[k], vr[k]
            print(f"  size {k}: expected, accepted {100 * r['accepted'] / r['n']:.1f}% (exact {100 * r['exact'] / r['n']:.1f}%; "
                  f"without delta {100 * r0['accepted'] / r0['n']:.1f}%); not expected, accepted as another word "
                  f"{100 * r['unexpected_accepted'] / r['n_unexpected']:.1f}% (without delta "
                  f"{100 * r0['unexpected_accepted'] / r0['n_unexpected']:.1f}%)")
        if a.write_margins:
            ear = PE.ParentEar({w: list(zip(ear.labels[w], ear._w[w])) for w in ear.words}, list(ear.bank), margins, deltas,
                               dict(ear.meta, margins_source="tools/sim_parent_ear.py on the stand-in babbler"))
            ear.save(os.path.join(a.root, "parent_ear.npz"))
            print(f"  m and delta written into the ear: digest {ear.digest[:16]}")
    if "cost" not in skip:
        c = measure_cost(ear, words, rng)
        res["cost"] = c
        print(f"cost: {c}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
