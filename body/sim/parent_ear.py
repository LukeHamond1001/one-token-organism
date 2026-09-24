"""THE PARENT'S EAR FOR THE CHILD'S VOICE (docs/SIM_DESIGN.md 4.9, A27; package P3): a disclosed recognizer in the world, never in
the body. The child's utterance, as sound pressure at her head, is heard through the same cochlea as the child's own ears
(body/sim/ears.cochlea: 40 gammatone bands, the power response, 1.00 ERB, calibrated in Pa^2), and she listens only for the words
she expects in the moment:

  the utterance -> the cochlea's band power per 10 ms frame -> log(power + her threshold) (a band under her hearing threshold,
  ears.THRESH_DB = 10 dB SPL, reads as the threshold) -> its frames within 25 dB of the loudest -> the bands shifted down by
  0-4 (she adapts to a shorter tract) -> cepstra c1-c12 (the spectral envelope, blind to pitch and level), mean-normalized
  -> Itakura's slope-constrained DTW against each template of a word (steps of 0, 1 or 2 template frames a frame; a template
  1/2 to 2 times the utterance's length) -> a word's distance: its nearest template over the shifts

THE DECISION (A27, and P3's delta). With E the words she expects (listed per situation before birth,
body/sim/lang/transcriber.expected_words), and m and delta her margins for a set of that size (consts.EAR_M, EAR_DELTA):
  accepted    c, the nearest of E (ties by her vocabulary's order), when
                (1) d(c) - d(the babble bank) < -m: nearer than the child's own babble by the margin (A27; the bank is the
                    babbler's utterances recorded before birth), and
                (2) c is the nearest of all her words, or d(c) - d(that nearest) < delta: heard as c or as a word close to
                    it, not as another word she knows (P3: without it, a clear word she did not expect passed as an expected one on 36-94% of trials,
                    by the set's size, measured on the 8 held-out voices; with it, as often as babble does, 2%)
  exact       accepted, and c also the nearest of all her words (none strictly nearer): a right name (worth 2)
  approximate accepted and not exact: a recast and a smile of 1, until the exact word has been said 3 times (4.6)
Nothing else is accepted: a word she does not expect is never heard as itself, however clear, and is heard as an expected word
only when it sounds close to it. Both margins are set by one principle before birth: a sound that is not the context word (the
held-out babble for m, the held-out voices' other words for delta) passes as the context word at most 2% of the time.

WHAT NEVER CHANGES. The templates, the bank and the margins are fixed before birth, and nothing in life adds to them: there is no
add_template (the prototype had one, "the child's own accepted productions"; A27 refuses it, since each accepted production added
would widen what passes, so the child's own chance acceptances would loosen the criterion over time). Every array she scores
with (the templates, the bank, the stacks the DTW reads) is a view of an immutable bytes buffer, so no flag can make it writable;
the margins, the words and the tables are read-only mappings; no attribute can be set after the build. verify() rebuilds the
digest from the very arrays she scores with (load() calls it, and the conduct at every night boundary). hear() reads them and
writes nothing. So the same utterance in the same situation is heard the same way on the first day and the hundredth
(body/tests/test_sim_lang.py holds it, and that no write reaches her ear).

HER TEMPLATES (A27; consts.EAR_VOICES): each word she knows said alone by her own voice at the plain and approval pitch, by the
same synthesizer at the old child pitch, and by 8 other macOS voices (Flo, Sandy, Shelley, Eddy, Reed, Junior, Kathy, Fred:
other synthesizers, not recordings of people), each through the life's voice cache (body/sim/voice/synth.py), so every template
clip is in its ledger. They are made for the birth words and the growth queue's words before birth; she listens only for the
words she has (the vocabulary grows; the templates were all there from the start: A15, "her ear knows only the vocabulary").

COST. The expected words and the bank are scored at the utterance's end; the rest of her words only when a word is accepted (the
exact test), so a rejection costs the expected words and the bank alone. An utterance longer than twice her longest template can
match no template (the slope constraint), so it is refused without scoring (the same answer). tools/sim_parent_ear.py measures the
cost per utterance and per tick at the babble's rate.

The first prototype (the voice study's recog.py, then this file until P3) heard through the study's cochlea (filters about 1.6 ERB
wide, body/sim/parent_ear_cochlea.py, now removed), scored every word of every template, and could add the child's productions;
the design's figures for it (94-100% within the synthesizer, 72-96% across held-out voices, m = -0.35) were measured there.
"""
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np

try:
    from . import ears as E
    from .lang import consts as K
except ImportError:                                     # imported as a top-level module from body/sim/
    import ears as E                                    # noqa: I001
    from lang import consts as K

NCEP = K.EAR_NCEP
SHIFTS = K.EAR_SHIFTS
_k = np.arange(E.NB)
DCT = np.cos(np.pi / E.NB * (_k[None, :] + 0.5) * np.arange(1, NCEP + 1)[:, None])      # c1..c12
FLOOR = E.E_THR                                          # her hearing threshold per band (Pa^2): the log's floor


def log_bands(x_pa):
    """sound pressure at her head (Pa, 16 kHz) -> log band power per 10 ms frame [frames, 40]."""
    return np.log(E.cochlea(np.asarray(x_pa, np.float64)) + FLOOR)


def trim(B, db=K.EAR_TRIM_DB):
    """the frames from the first to the last within db of the loudest frame's loudest band."""
    e = B.max(1)
    k = np.where(e > e.max() - db / 10 * np.log(10))[0]
    return B[k[0]:k[-1] + 1] if len(k) else B[:1]


def ceps(B, shift=0):
    if shift:
        B = np.concatenate([B[:, shift:], np.repeat(B[:, -1:], shift, 1)], 1)
    c = B @ DCT.T
    return c - c.mean(0, keepdims=True)


def template_of(x_pa):
    """a template (or a bank item): its cepstra at no shift."""
    return ceps(trim(log_bands(x_pa)), 0)


def frozen(a, dtype=np.float64):
    """a copy of a as a view of an immutable bytes buffer: read-only, and no flag can make it writable."""
    a = np.ascontiguousarray(a, dtype)
    return np.frombuffer(a.tobytes(), dtype).reshape(a.shape)


class _Stack:
    """templates stacked for the DTW: frames [K, M, d] padded, lengths, squared norms; frozen once built."""

    def __init__(self, items):
        self.K = len(items)
        self.L = frozen([len(t) for t in items], np.int64)
        self.M = int(self.L.max()) if self.K else 0
        T = np.zeros((self.K, max(self.M, 1), NCEP))
        for i, t in enumerate(items):
            T[i, :len(t)] = t
        self.valid = frozen(np.arange(max(self.M, 1))[None, :] < self.L[:, None], np.bool_)
        self.flat = frozen(T.reshape(self.K * max(self.M, 1), NCEP))
        self.t2 = frozen((self.flat ** 2).sum(1))
        self._built = True

    def __setattr__(self, k, v):
        if getattr(self, "_built", False):
            raise AttributeError("the parent's ear is fixed before birth (A27): nothing in life changes it")
        object.__setattr__(self, k, v)

    def items(self):
        """the templates as scored (for verify)."""
        T = self.flat.reshape(self.K, max(self.M, 1), NCEP)
        return [T[i, :int(n)] for i, n in enumerate(self.L)]


def dtw(q, st):
    """q [n, d] against a _Stack -> the normalized DTW distance per template (inf outside the slope constraint): Itakura's
    steps (the utterance advances a frame each step; the template 0, 1 or 2), the path from both starts to both ends, summed
    frame distances over n. Only the templates inside the slope constraint (1/2 to 2 times the utterance) are computed, each
    over its own length: a template's path never reads a column past its end, so the answer is the whole stack's."""
    n = len(q)
    d = np.full(st.K, np.inf)
    ok = np.where((st.L * K.EAR_SLOPE >= n) & (st.L <= K.EAR_SLOPE * n))[0]
    if len(ok) == 0:
        return d
    M = int(st.L[ok].max())
    Mf = max(st.M, 1)
    rows = (ok[:, None] * Mf + np.arange(M)[None, :]).ravel()
    flat, t2 = st.flat[rows], st.t2[rows]
    d2 = (q ** 2).sum(1)[:, None] + t2[None, :] - 2.0 * (q @ flat.T)
    C = np.sqrt(np.maximum(d2, 0.0)).reshape(n, len(ok), M)
    C = np.where(st.valid[ok, :M][None], C, np.inf)
    D = np.full((len(ok), M), np.inf)
    D[:, 0] = C[0, :, 0]
    best = np.empty_like(D)
    for i in range(1, n):
        best[:] = D                                            # the template stays
        np.minimum(best[:, 1:], D[:, :-1], out=best[:, 1:])    # or advances 1
        np.minimum(best[:, 2:], D[:, :-2], out=best[:, 2:])    # or 2
        np.add(C[i], best, out=D)
    d[ok] = D[np.arange(len(ok)), st.L[ok] - 1] / n
    return d


class Hearing:
    """what she heard in one utterance."""
    __slots__ = ("word", "exact", "nearest", "d", "d_bank", "m", "nearest_all", "d_best", "delta", "expected", "frames",
                 "scored", "why")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def as_dict(self):
        return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in
                ((k, getattr(self, k)) for k in self.__slots__)}


class ParentEar:
    """her ear, fixed: templates {word: [cepstra, ...]} in her words' order (with a voice label each), the babble bank [cepstra],
    the margins m and delta by expected-set size. Read-only after build."""

    def __init__(self, templates, bank, margins=None, deltas=None, meta=None):
        self.words = tuple(templates)
        self.order = MappingProxyType({w: i for i, w in enumerate(self.words)})
        self._w = MappingProxyType({w: tuple(frozen(it[1] if isinstance(it, tuple) else it) for it in items)
                                    for w, items in templates.items()})
        self.labels = MappingProxyType({w: tuple(it[0] if isinstance(it, tuple) else "" for it in items)
                                        for w, items in templates.items()})
        self.bank = tuple(frozen(b) for b in bank)
        self._bank = _Stack(self.bank)
        self._stacks = MappingProxyType({w: _Stack(a) for w, a in self._w.items()})
        self.max_len = max([len(a) for arrs in self._w.values() for a in arrs] + [0])
        self.margins = MappingProxyType({int(k): float(v) for k, v in (margins if margins is not None else K.EAR_M).items()})
        self.deltas = MappingProxyType({int(k): float(v) for k, v in (deltas if deltas is not None else K.EAR_DELTA).items()})
        self.meta = MappingProxyType(json.loads(json.dumps(dict(meta or {}))))
        self.digest = self._digest()

    def __setattr__(self, k, v):
        if getattr(self, "digest", None) is not None:
            raise AttributeError("the parent's ear is fixed before birth (A27): nothing in life changes it")
        object.__setattr__(self, k, v)

    def _digest(self, scored=False):
        """sha256 of her words, templates, bank and margins; scored=True: of the arrays the DTW actually reads."""
        h = hashlib.sha256()
        for w in self.words:
            h.update(w.encode() + b"\0")
            for a in (self._stacks[w].items() if scored else self._w[w]):
                h.update(np.ascontiguousarray(a).tobytes())
        for b in (self._bank.items() if scored else self.bank):
            h.update(np.ascontiguousarray(b).tobytes())
        h.update(json.dumps(sorted((int(k), float(v)) for k, v in self.margins.items())).encode())
        h.update(json.dumps(sorted((int(k), float(v)) for k, v in self.deltas.items())).encode())
        return h.hexdigest()

    def verify(self):
        """her ear as built: the digest rebuilt from the arrays she keeps and from the stacks she scores with (their norms
        and masks too) must be the one fixed at build; else ValueError (the life pauses)."""
        ok = self._digest() == self.digest and self._digest(scored=True) == self.digest
        for st in list(self._stacks.values()) + [self._bank]:
            ok = ok and np.array_equal(st.t2, (st.flat ** 2).sum(1)) and \
                np.array_equal(st.valid, np.arange(max(st.M, 1))[None, :] < st.L[:, None])
        if not ok:
            raise ValueError("the parent's ear no longer matches its digest: something changed it after its build (A27)")
        return True

    def missing(self, words):
        """the words she has no templates for (her ear is built with every word before birth, 4.9)."""
        return [w for w in words if w not in self._w]

    @staticmethod
    def _by_size(table, size):
        ks = sorted(table)
        return table[size] if size in table else table[ks[-1] if size > ks[-1] else ks[0]]

    def margin(self, size):
        """m for an expected set of this size (A27); a size beyond the table takes the largest size's."""
        return self._by_size(self.margins, size)

    def delta(self, size):
        """delta for an expected set of this size; a size beyond the table takes the largest size's."""
        return self._by_size(self.deltas, size)

    # ------------------------------------------------------------------ hearing
    def features(self, x_pa):
        B = trim(log_bands(x_pa))
        return [ceps(B, s) for s in SHIFTS], len(B)

    def distances(self, feats, words):
        """-> {word: its nearest template's distance over the shifts} for the given words."""
        out = {}
        for w in words:
            st = self._stacks[w]
            out[w] = float(min(dtw(q, st).min() for q in feats)) if st.K else math.inf
        return out

    def bank_distance(self, feats):
        return float(min(dtw(q, self._bank).min() for q in feats)) if self._bank.K else math.inf

    def hear(self, x_pa, expected, vocab=None):
        """the utterance (Pa at her head) when she expects `expected` -> Hearing. vocab: the words she has (the exact test is
        against these; default all her templates' words). Words she has no template for are not heard."""
        vocab = [w for w in (vocab if vocab is not None else self.words) if w in self._w]
        exp = sorted({w for w in expected if w in self._w and w in vocab}, key=self.order.get)
        feats, n = self.features(x_pa)
        h = Hearing(word=None, exact=False, nearest=None, d=math.inf, d_bank=math.inf, m=self.margin(len(exp)),
                    nearest_all=None, d_best=math.inf, delta=self.delta(len(exp)), expected=tuple(exp), frames=n, scored=0,
                    why="nothing expected")
        if not exp or n > K.EAR_SLOPE * self.max_len:        # nothing expected, or longer than any template can match
            h.why = h.why if not exp else "longer than any template"
            return h
        d = self.distances(feats, exp)
        c = min(exp, key=lambda w: (d[w], self.order[w]))
        h.nearest, h.d = c, d[c]
        h.scored = len(exp)
        if not math.isfinite(h.d):
            h.why = "no template within the slope"
            return h
        h.d_bank = self.bank_distance(feats)
        if not (h.d - h.d_bank < -h.m):
            h.why = "not nearer than the babble"
            return h
        rest = [w for w in vocab if w not in d]
        d.update(self.distances(feats, rest))
        h.scored += len(rest)
        h.nearest_all = min(vocab, key=lambda w: (d[w], self.order[w]))
        h.d_best = d[h.nearest_all]
        exact = not any(d[w] < d[c] for w in vocab if w != c)
        if not (exact or h.d - h.d_best < h.delta):
            h.why = f"heard as {h.nearest_all!r}, not close to {c!r}"
            return h
        h.word, h.exact, h.why = c, exact, ""
        return h

    # ------------------------------------------------------------------ files
    def save(self, path):
        """npz of the arrays and a JSON manifest (the words, each template's voice label, the margins, the meta) and the
        digest; written as a temporary file, then renamed."""
        import os                                                          # noqa: PLC0415
        tl, tw, tv, frames = [], [], [], []
        for w in self.words:
            for lab, a in zip(self.labels[w], self._w[w]):
                tl.append(len(a))
                tw.append(self.order[w])
                tv.append(lab)
                frames.append(a)
        manifest = dict(words=list(self.words), voices=tv, margins={str(k): v for k, v in self.margins.items()},
                        deltas={str(k): v for k, v in self.deltas.items()}, meta=dict(self.meta), digest=self.digest)
        tmp = str(path) + ".tmp.npz"
        np.savez(tmp, tmpl=np.concatenate(frames) if frames else np.zeros((0, NCEP)), tmpl_len=np.array(tl, np.int64),
                 tmpl_word=np.array(tw, np.int64), bank=np.concatenate(self.bank) if self.bank else np.zeros((0, NCEP)),
                 bank_len=np.array([len(b) for b in self.bank], np.int64),
                 manifest=np.frombuffer(json.dumps(manifest, sort_keys=True).encode(), np.uint8))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        man = json.loads(bytes(z["manifest"]).decode())
        words = man["words"]
        tmpl = {w: [] for w in words}
        off = 0
        for L, wi, lab in zip(z["tmpl_len"], z["tmpl_word"], man["voices"]):
            tmpl[words[int(wi)]].append((lab, z["tmpl"][off:off + L]))
            off += int(L)
        bank, off = [], 0
        for L in z["bank_len"]:
            bank.append(z["bank"][off:off + L])
            off += int(L)
        ear = cls(tmpl, bank, {int(k): v for k, v in man["margins"].items()}, {int(k): v for k, v in man["deltas"].items()},
                  man["meta"])
        if ear.digest != man["digest"]:
            raise ValueError(f"the parent's ear at {path} does not match its digest: it changed after it was built")
        ear.verify()
        return ear
