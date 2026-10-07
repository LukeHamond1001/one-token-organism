"""A202: the grounding of words in joint attention (body/core/grounding.py; the G1's looks, body/sim/eyes.py). Run:
python3 -c "import sys; sys.path.insert(0,'.'); from body.tests.test_sim_grounding import test_the_word_heard_on_a_thing_draws_the_eyes_to_its_look as t; t()"
"""
import math
import types

import numpy as np

from body.core import grounding as GR
from body.core.anatomy import Grounding
from body.sim import eyes as E
from body.sim import world as W
from body.sim import g1scene as G


def _frame(eye_p, eye_f, gaze=(0.0, 0.0, 0.0), body_n=250):
    body = np.zeros(body_n)
    from body.sim.anatomy import GAZE_AT
    body[GAZE_AT:GAZE_AT + 3] = gaze
    return types.SimpleNamespace(obs={"eye_p": eye_p, "eye_f": eye_f, "body": body}, truth={})


def _eye_f(centre_rgby, outer_rgby):
    """a colour window of 8 x 8 cells: the central 4 x 4 one opponent code, the rest another"""
    n = W.FOVEA_PX // E.CELL_F
    c = np.zeros((n, n, 4)); c[:] = outer_rgby; c[2:6, 2:6] = centre_rgby
    return np.concatenate([np.zeros(E.EYE_F_SIZE - n * n * 4), c.reshape(-1)])


def _eye_p(cell, rgby, rest_rgby):
    rows, cols = E.COL_CELLS
    c = np.zeros((rows, cols, 4)); c[:] = rest_rgby; c[cell] = rgby
    return np.concatenate([np.zeros(E.EYE_P_SIZE - rows * cols * 4), c.reshape(-1)])


class _Body(GR.GroundingMixin):
    def __init__(self):
        from body.sim.anatomy import _ground_appearance, _ground_periphery
        self.anatomy = types.SimpleNamespace(grounding=Grounding("named_periph", 4, _ground_appearance, _ground_periphery, skip=(1,)))
        self.m = types.SimpleNamespace(vocab=80)
        self.sil = 0
        self.bans = ()


RED = (0.6, 0.0, 0.0, 0.3)      # red-green ON, no OFF, no blue-yellow ON, blue-yellow OFF (a red ball on the opponent axes)
BLUE = (0.0, 0.2, 0.7, 0.0)
FLOOR = (0.02, 0.0, 0.0, 0.05)  # the beige floor: a faint warm cast


def test_the_word_heard_on_a_thing_draws_the_eyes_to_its_look():
    b = _Body()
    ball, block, the = 10, 11, 12
    # the word "ball" heard three times with the ball in the fovea, "block" with the block, "the" with both things in view (their mixed look)
    MIX = tuple((r_ + b_) / 2 for r_, b_ in zip(RED, BLUE))
    for w, look in ((ball, RED), (block, BLUE), (the, MIX), (ball, RED), (block, BLUE), (the, MIX), (ball, RED), (block, BLUE), (the, MIX), (the, MIX)):
        b._ground_learn(w, _frame(_eye_p((1, 2), look, FLOOR), _eye_f(look, FLOOR)))
    A, n, tr = b._ground_state()
    assert n[ball] == 3 and n[block] == 3 and n[the] == 4, (n[ball], n[block], n[the])
    assert tr[0] == the and tr[1] == GR.GROUND_TRACE
    # a word heard on the empty floor binds nothing
    b._ground_learn(ball, _frame(_eye_p((1, 2), FLOOR, FLOOR), _eye_f(FLOOR, FLOOR)))
    assert n[ball] == 3
    # "ball" heard: the ball in the periphery's top-left cell, the block in the bottom-right; the cue points at the ball
    b._ground_learn(ball, _frame(_eye_p((1, 2), FLOOR, FLOOR), _eye_f(FLOOR, FLOOR)))
    rows, cols = E.COL_CELLS
    ep = _eye_p((0, 0), RED, FLOOR)
    ep[-rows * cols * 4:].reshape(rows, cols, 4)[rows - 1, cols - 1] = BLUE
    f = _frame(ep, _eye_f(FLOOR, FLOOR))
    b._ground_sense(f)
    cue = f.obs["named_periph"]
    assert cue[0] == 1.0, cue
    assert cue[1] < 0 and cue[2] > 0, cue                                # left and up of the fovea's centre
    x, y = 0.5 * G.COL_W / cols, 0.5 * G.COL_H / rows
    assert abs(cue[1] - math.atan((x - G.COL_W / 2) / E.COL_F_PX)) < 1e-9 and abs(cue[2] - math.atan((G.COL_H / 2 - y) / E.COL_F_PX)) < 1e-9
    # "block" heard: the cue points at the block, bottom-right
    b._ground_learn(block, _frame(_eye_p((1, 2), FLOOR, FLOOR), _eye_f(FLOOR, FLOOR)))
    f2 = _frame(ep, _eye_f(FLOOR, FLOOR)); b._ground_sense(f2)
    assert f2.obs["named_periph"][0] == 1.0 and f2.obs["named_periph"][1] > 0 and f2.obs["named_periph"][2] < 0, f2.obs["named_periph"]
    # "the" heard: its look is the mean of red and blue, no cell matches it clearly enough -> no cue
    b._ground_learn(the, _frame(_eye_p((1, 2), FLOOR, FLOOR), _eye_f(FLOOR, FLOOR)))
    f3 = _frame(ep, _eye_f(FLOOR, FLOOR)); b._ground_sense(f3)
    assert f3.obs["named_periph"][0] == 0.0, f3.obs["named_periph"]
    # the trace runs out: after GROUND_TRACE ticks the cue is quiet
    b._ground_learn(ball, _frame(_eye_p((1, 2), FLOOR, FLOOR), _eye_f(FLOOR, FLOOR)))
    for _ in range(GR.GROUND_TRACE):
        f4 = _frame(ep, _eye_f(FLOOR, FLOOR)); b._ground_sense(f4)
    assert f4.obs["named_periph"][0] == 1.0
    f5 = _frame(ep, _eye_f(FLOOR, FLOOR)); b._ground_sense(f5)
    assert f5.obs["named_periph"][0] == 0.0
    # the name: the ball in the fovea primes "ball"; the empty floor primes nothing; the skipped symbol never
    say = b._ground_say(_frame(ep, _eye_f(RED, FLOOR)))
    assert say is not None and say[0] == ball and say[1] > GR.GROUND_MARGIN, say
    import torch                                                           # A202i: the prior in the readout's units: GROUND_SAY standard
    lg = torch.tensor([0.0, 1.0, 2.0, 3.0])                                 # deviations of the logits per GROUND_MARGIN of margin
    assert abs(GR.ground_prior(lg, GR.GROUND_MARGIN) - GR.GROUND_SAY * float(lg.std())) < 1e-6
    assert abs(GR.ground_prior(lg, 2 * GR.GROUND_MARGIN) - 2 * GR.GROUND_SAY * float(lg.std())) < 1e-6
    assert b._ground_say(_frame(ep, _eye_f(FLOOR, FLOOR))) is None
    assert b._ground_say(_frame(ep, _eye_f(BLUE, FLOOR)))[0] == block
    # A202b: a word heard over everything ("is": red, blue, red, blue ...) averages to no look (the running mean), its consistency falls,
    # and it neither draws the eyes nor is primed, while "ball" (red every time) keeps a consistent look
    is_ = 13
    for k in range(8):
        b._ground_learn(is_, _frame(_eye_p((1, 2), FLOOR, FLOOR), _eye_f(RED if k % 2 == 0 else BLUE, FLOOR)))
    assert n[is_] == 8 and b._ground_consist(is_) < GR.GROUND_CONSIST < b._ground_consist(ball), (b._ground_consist(is_), b._ground_consist(ball))
    f6 = _frame(ep, _eye_f(FLOOR, FLOOR)); b._ground_sense(f6)
    assert f6.obs["named_periph"][0] == 0.0, f6.obs["named_periph"]
    assert b._ground_say(_frame(ep, _eye_f(RED, FLOOR)))[0] == ball
    # A202e: the line's last word binds GROUND_FINAL times as hard: "look the duck." three times over yellow, "look" and "the" bound at 1
    # each, "duck" (before the end symbol) at 3: the yellow fovea primes "duck", not "look"
    look, the2, duck, END = 20, 21, 22, 1
    b.end_id = END
    YEL = (0.16, 0.0, 0.0, 0.84)   # the room's yellow (1, .84, .08) on the opponent axes
    for _ in range(3):
        for w in (look, the2, duck):
            b._ground_learn(w, _frame(_eye_p((1, 2), YEL, FLOOR), _eye_f(YEL, FLOOR)))
        b._ground_learn(END, _frame(_eye_p((1, 2), YEL, FLOOR), _eye_f(YEL, FLOOR)))
    assert n[duck] == 9 and n[look] == 3 and n[the2] == 3, (n[duck], n[look], n[the2])
    # "look" and "the" heard over the ball too: their looks mix, the duck's stays; the yellow fovea primes "duck"
    for _ in range(3):
        for w in (look, the2):
            b._ground_learn(w, _frame(_eye_p((1, 2), RED, FLOOR), _eye_f(RED, FLOOR)))
    say = b._ground_say(_frame(ep, _eye_f(YEL, FLOOR)))
    assert say is not None and say[0] == duck, say
    # A202f: a line of one word ("oh.") gets no final weight: three "oh." over yellow bind 3, not 9
    oh = 23
    b._ground_line_n = 0                                                   # (the hearings above came without a line's end)
    for _ in range(3):
        b._ground_learn(oh, _frame(_eye_p((1, 2), YEL, FLOOR), _eye_f(YEL, FLOOR)))
        b._ground_learn(END, _frame(_eye_p((1, 2), YEL, FLOOR), _eye_f(YEL, FLOOR)))
    assert n[oh] == 3, n[oh]
    # the night: no eyes, no look, nothing bound, the cue quiet
    nf = types.SimpleNamespace(obs={"body": np.zeros(250)}, truth={})
    b._ground_learn(ball, nf); b._ground_sense(nf)
    assert n[ball] == 3 and nf.obs["named_periph"][0] == 0.0
    r = b.ground_report()
    assert r["words"] >= 4 and r["bind"] >= 30 and r["cue"] >= 3, r
    print("A202 ok", r)


if __name__ == "__main__":
    test_the_word_heard_on_a_thing_draws_the_eyes_to_its_look()
