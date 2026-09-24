"""The parent's ear for the child's voice: a disclosed recogniser in the environment, never in the body
(docs/SIM_DESIGN.md section 4.9; from the 2026-09-24 prototype).

  the child's utterance (its tract's waveform; the parent attends to the child's voice)
  -> the same born cochlea as the child's ears (40 ERB bands, 10 ms frames, ears.py) -> log band energies
  -> a speaker-length warp: the bands shifted down by 0..4 (the parent adapts to a shorter tract; best shift kept)
  -> cepstra c1..c12 (DCT of the log bands: the spectral envelope, blind to pitch and level), mean-normalized
  -> DTW (Itakura's slope-constrained steps) against every template of every word the parent knows
  -> the nearest word and its distance; accepted by the thresholds (set before birth, never loosened)

Templates: each vocabulary word in the parent's own voice at two registers (plain 1.15, approval 1.35), plus the
same synthesizer at the child's old pitch (1.45); later, the child's own accepted productions (the parent learning
its child's words, only when accepted in context).

Accepting a word in context (accept()): the word c is accepted when c is the nearest of the words the parent expects
in this situation AND d(c) - d(the child's own babble bank) < -m. The bank is the babbler's utterances recorded
before birth and kept fixed; m = -0.35 was set on a 193-utterance bank so held-out babble passes as the context word 2%
of the time, and is re-set for the real context set size (P3/P6). Measured: real other speakers' words pass 100%.

Templates: each clip is <voice_dir>/<i>.json (+ .f32), i the word's index in the lexicon file (the synthesizer's
output, P1). The prototype's voices were the parent at pitch 1.15 and 1.35, the old child pitch 1.45, and 8 other
system voices (BANK).
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parent_ear_cochlea import stft, GT, WIN, load           # the prototype cochlea its figures were measured through
#                                                               (16 kHz, 25 ms / 10 ms; P3v moves it to ears.cochlea)

NB = GT.shape[0]
NCEP = 12
SHIFTS = (0, 1, 2, 3, 4)
_k = np.arange(NB)
DCT = np.cos(np.pi / NB * (_k[None, :] + 0.5) * np.arange(1, NCEP + 1)[:, None])   # c1..c12


def bands(x):
    X = stft(np.pad(np.asarray(x, np.float32), (0, WIN)))
    return np.log((np.abs(X) ** 2 @ GT.T) + 1e-7)


def trim(B, rel_db=25):
    e = B.max(1)
    k = np.where(e > e.max() - rel_db / 10 * np.log(10))[0]
    return B[k[0]:k[-1] + 1] if len(k) else B[:1]


def ceps(B, shift=0):
    if shift:
        B = np.concatenate([B[:, shift:], np.repeat(B[:, -1:], shift, 1)], 1)
    c = B @ DCT.T
    return c - c.mean(0, keepdims=True)


def dtw_batch(q, temps):
    """q [n, d]; temps list of [m_k, d] -> normalized Itakura DTW distance per template."""
    n = len(q)
    M = max(len(t) for t in temps)
    K = len(temps)
    T = np.full((K, M, q.shape[1]), np.nan)
    L = np.array([len(t) for t in temps])
    for k, t in enumerate(temps):
        T[k, :len(t)] = t
    C = np.sqrt(((q[:, None, None, :] - T[None]) ** 2).sum(-1))       # [n, K, M]
    C = np.where(np.isnan(C), np.inf, C)
    D = np.full((K, M), np.inf)
    D[:, 0] = C[0, :, 0]
    for i in range(1, n):
        prev = D
        s1 = np.concatenate([np.full((K, 1), np.inf), prev[:, :-1]], 1)
        s2 = np.concatenate([np.full((K, 2), np.inf), prev[:, :-2]], 1)
        D = C[i] + np.minimum(np.minimum(prev, s1), s2)
    d = D[np.arange(K), L - 1] / n
    d[(L < n / 2) | (L > 2 * n)] = np.inf                              # outside the slope constraint
    return d


BANK = ('L_p115', 'L_p135', 'L_k1.45') + tuple(f'xvoices/{v}' for v in ('Flo', 'Sandy', 'Shelley', 'Eddy', 'Reed', 'Junior', 'Kathy', 'Fred'))


class Ear:
    def __init__(self, words, voice_dirs, lex_path):
        self.words = list(words)
        self.temps = []                                               # (word, cepstra)
        lex = open(lex_path).read().split()
        for d in voice_dirs:
            for w in self.words:
                i = lex.index(w)
                x, _, _ = load(os.path.join(d, f'{i}.json'))
                self.temps.append((w, ceps(trim(bands(x)))))

    def hear_among(self, x, cands):
        """the context-gated hearing: only the candidate words (what the parent expects in this situation)."""
        sc = self.scores(x)
        return sorted(((w, sc[w]) for w in cands), key=lambda kv: kv[1][0])

    def scores(self, x):
        """-> (per-word best distance over templates and shifts, the shift used per word)."""
        B = trim(bands(x))
        best = {w: (np.inf, 0) for w in self.words}
        T = [t for _, t in self.temps]
        for s in SHIFTS:
            d = dtw_batch(ceps(B, s), T)
            for (w, _), dv in zip(self.temps, d):
                if dv < best[w][0]:
                    best[w] = (dv, s)
        return best

    def hear(self, x):
        sc = self.scores(x)
        ranked = sorted(sc.items(), key=lambda kv: kv[1][0])
        return ranked                                                 # [(word, (dist, shift)), ...]

    def add_template(self, word, x):
        self.temps.append((word, ceps(trim(bands(x)))))


def bank_distance(x, bank):
    """The nearest distance from an utterance to the babble bank (a list of cepstra: ceps(trim(bands(u))))."""
    B = trim(bands(x))
    return min(dtw_batch(ceps(B, s), bank).min() for s in SHIFTS)


def accept(ear, x, word, expected, bank, m=-0.35):
    """The parent's decision for an utterance x when it expects `expected` (which includes `word`): accepted as `word`
    only if word is the nearest of the expected words and nearer than the babble bank by the margin (see above)."""
    sc = ear.scores(x)
    d = {k: v[0] for k, v in sc.items()}
    nearest = min(expected, key=d.get)
    return nearest == word and d[word] - bank_distance(x, bank) < -m
