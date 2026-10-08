"""A214: the grounding of words in the situation (body/core/situation.py). Run:
python3 -c "import sys; sys.path.insert(0,'.'); from body.tests.test_sim_situation import test_the_word_for_what_is_going_on as t; t()"
"""
import types

import numpy as np
import torch

from body.core import grounding as GR
from body.core.anatomy import Situation
from body.core.situation import SituationMixin


class _Body(SituationMixin):
    def __init__(self, d=16, vocab=40):
        self.anatomy = types.SimpleNamespace(situation=Situation(band=1, skip=(1,), name_skip=(5,)))
        self.m = types.SimpleNamespace(vocab=vocab, band_mu=torch.zeros(3, d))
        self.bands = torch.zeros(3, d); self.sil = 0; self.bans = (); self.end_id = 1


def test_the_word_for_what_is_going_on():
    """'up' heard as the line's last word while the body stands (state S), 'down' while it lies (state L), 'the' over both: standing
    primes 'up' and nothing else, lying 'down'; 'the' (inconsistent) and a word that never ends a line are never primed; the skipped
    symbol binds nothing"""
    b = _Body(); d = 16
    S = torch.zeros(d); S[0] = 1.0; S[3] = 0.5
    Lz = torch.zeros(d); Lz[1] = 1.0; Lz[4] = -0.5
    up, down, the, look, xl = 10, 11, 12, 13, 5
    rng = np.random.default_rng(0)
    for k in range(GR.GROUND_SAY_MIN_N + 2):
        for st, w in ((S, up), (Lz, down)):
            b.bands[1] = st + 0.05 * torch.tensor(rng.standard_normal(d), dtype=torch.float32)
            b._situ_learn(look); b._situ_learn(the); b._situ_learn(w); b._situ_learn(b.end_id)   # 'look the up.' / 'look the down.'
            b._situ_learn(xl); b._situ_learn(xl); b._situ_learn(b.end_id)                        # a spelled word's letters
    A, n = b._situ_state()
    assert n[up] > n[look] and n[the] == n[look] and n[xl] > 0 and n[1] == 0, (n[up], n[look], n[the], n[xl], n[1])
    b.bands[1] = S
    say = b._situ_say()
    assert say is not None and say[0] == up and say[1] >= GR.GROUND_MARGIN, say
    b.bands[1] = Lz
    say2 = b._situ_say()
    assert say2 is not None and say2[0] == down, say2
    b.bands[1] = S
    cons = b._situ_consist()
    assert cons[the] < GR.GROUND_CONSIST < cons[up], (cons[the], cons[up])
    f_, h_ = b._situ_f.numpy(), b._situ_h.numpy()
    assert f_[look] == 0 and f_[up] > 0 and f_[xl] > 0, (f_[look], f_[up], f_[xl])            # 'look' never ends a line; the letter does, and is name_skip
    b.bands[1] = S * 0.0 + 0.3 * torch.tensor(rng.standard_normal(d), dtype=torch.float32)       # a situation like neither
    assert b._situ_say() is None or b._situ_say()[1] < 1.0
    r = b.situ_report()
    assert r["bind"] > 0 and r["say"] >= 2 and r["words"] >= 2, r
    print(f"A214 ok: standing primes 'up' (margin {say[1]:.2f}), lying 'down' ({say2[1]:.2f}); 'the' inconsistent ({cons[the]:.2f}), 'look' never final, the letter never primed; {r}")
