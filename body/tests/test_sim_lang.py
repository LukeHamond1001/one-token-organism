"""the parent's language (docs/SIM_DESIGN.md 4.4-4.10, A13-A15, A26-A28; package P3): the fast layer's templates and the line check
(body/sim/lang/templates.py), her intents and acts, her voice's manners and the speech side of L2 (conduct.py), her ear
(body/sim/parent_ear.py), the transcriber of the child's two outputs (transcriber.py) and the ledger (ledger.py).
Run: python3 -m body.tests.test_sim_lang

Each rule is tested both ways: what must hold, and what must NOT hold (a line naming what she cannot see, a held-out pair by its
words or by its referent, a new word not last, Claude's praise; her ear accepting a word she did not expect, a word heard in place
of another called exact, her ear changing with what she hears or being written to, a word counted as said when it was an echo or
within one day, "understood" at the child's own base rate, a replay that differs from the ledger). The ear's rules are tested on an
ear built from the tract's own sounds (no engine needed), in two processes threaded and on one thread; the engine's templates are
tested for exact rebuilds where swiftc and macOS speech are present. The full ear (11 voices, the babble bank, its margins and
cost) is built and measured by tools/sim_parent_ear.py.
Tests 17-23 are the P3 verifier's findings, each failing on 5ea4c4e at its own assertion: an approximation smiled at out of context
(17), an ask met unanswered (18), a word the talk-over stopped counted as said (19), a second new word erasing the first (20), the
acts losing their object (21), the line check blind to what a line claims (22), the small items (23); and a write reaching her
ear (in 10). Test 24 replays the whole speech side across processes from snapshots taken mid-line and mid-turn.
Tests 25-30 are the P3 verifier's second findings (d053dff), each failing there at its own first assertion: Claude's lines asking
past the check (25, with the claims it could not hold), a name ask met by a word made before its question was heard (26), calls
the child could not answer by a look counted as missed and the call's window (27), words introduced that the world cannot
show (28, with the pace of 1 a minute), a give before its word counted as missed (29), deliberate writes to her ear uncaught
(30).
Tests 31-37 are the P3 verifier's third findings (7447f73), after the merge of main's design (A30-A57), each failing there at its
own first assertion: the gaze leak in her replies during an ask (31, A51, the verifier's probe both ways), the new word off its
pitch peak (32, A34), her reading of the child's fovea rather than its trunk, head line and hands (33, A40, with "reaching
toward"), the old held pairs (34, A55), echoes and "more?" (35), Claude's shows past the redirect rule and its loose forms (36),
and her imperfection and stage 2's "no." to a talk-over (37, A52).
Tests 38-42 are the P3 verifier's fourth findings (1ae6942), each failing there at its own first assertion: a variation set with
two lines of the same words, and the colour words short of a set where A55 puts them (38, 4.5); an ask made right after her cue
at its X, met by a child that only follows her cues (39, A51); asks about a toy in her hands or beyond the child's reach (40);
her reading of its head's line held 3 ticks only for follow-in naming (41, 4.10, A40); the low items (42).
Tests 43-45 are the P3 verifier's fifth findings (c816b5a), each failing there at its own first assertion: her cue's end taken
as her line's end, not her act's, so a child that only follows an offer outlasting its line met the ask (43, A51, the verifier's
probe both ways); cued() dropping a cue at a twin's id silently (44); the low items (45).
Tests 46-50 are the P3 verifier's sixth findings (6449921) under the lead's decision (stop patching routes, close the class: one
attention log, fail-closed, A51), each failing there at its own first assertion: L1's gaze while under way (46), a verb's
demonstration with a toy it never named (47), a queued act failing open (48), a copy starting as an ask opened (49), and the
property over random lives with a W2-like motion (50: no ask about X opens with X in her log within 44 ticks, none sees X enter
it, her body still while one is pending; the same lives with the ask's gate switched off break each). Tests 21, 26, 39, 43 and
44 were moved to the log (the name ask shows nothing, a cue's end is the log's, a target she cannot name stands for every
thing).
Tests 51-53 are the P3 verifier's seventh findings (883ec52), each failing there at its own first assertion, as does test 50,
now with places, touches on the child, acts she did not ask for and ticks skipped in its lives: a cue at a place that holds X
(the shelf's point and look, a walk to the sofa, L1's gaze at the table, her touch on the hand that holds it) no cue at X, met
by a child that only follows her (51); acts in her motion's report she did not ask for unread (52); ticks never read counted
clean, a glance after the tick's report lost, a look cancelled within a tick on no tick of the log (53). Test 44 now has a
place and a part of the child stand for every thing; 43 reads cue_clear at its own tick and has a cancelled act on its way back
the tick it was stopped; 35 ticks the tick of its direct ask (a tick not read voids a pending ask). Test 54 is the same class
for the call, found in this round's review (it fails on 883ec52 too): every movement of her body is at her face or beside it,
so a call said as a show, a glance, a greeting's lean-in or a copy ended was met by a child that only follows her.
Tests 55-58 are the P3 verifier's eighth findings (b301745), each failing there at its own first assertion, as does test 50, now
with toys near each other and beside her face, her face and voice, and the given toy in her hand on the tick it is given: a met
give voided on its own tick (55: each tick's percept now judges an ask before that tick's log is held to it; a give of another toy
missed); a cue at the box the ball rests in, or at a toy a few degrees from it, no cue at the ball (56: each field now every thing
a look that follows it could be read as, from Seen.near and Percept.face_near, which the world fills from Reader.near, NEAR_DEG
measured with the Reader); the call's own lean-in, and her smile and voice while it was pending (57: the call is its name alone
with no lean-in, her face and voice are in the log, and no line is said while it is pending); a save without her birth tick
failing open (58, with a copy's hand where it points). Tests 6, 37 and 54 moved off the call's longer frames (6, 37) and count her
voice among her movements (54); 44 has a thing she no longer sees stand for every thing (what lies near it is unknown)."""
import json
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # this tree's body
from body.sim import parent_ear as PE  # noqa: E402
from body.sim import tract as T  # noqa: E402
from body.sim.lang import conduct as C  # noqa: E402
from body.sim.lang import consts as K  # noqa: E402
from body.sim.lang import lexicon as LX  # noqa: E402
from body.sim.lang import templates as TP  # noqa: E402
from body.sim.lang.ledger import Ledger, LedgerDiverged, binom_tail  # noqa: E402
from body.sim.lang.percept import Percept, Seen  # noqa: E402
from body.sim.lang.transcriber import ChildWord, Transcriber, expected_words, read_letters  # noqa: E402
from body.sim.voice import synth as V  # noqa: E402

VOCAB = LX.BIRTH_WORDS
ROOMS = frozenset({"mat", "sofa", "window"})


def seen1(id_, name, colour="", on="", sees=False, reach=None, near=()):
    """a Seen (id, name, colour, on, child_sees, child_can_reach, near); within its reach, unless given, a toy on the mat or in
    its hand (the test's room: the child lies on the mat; a toy on the sofa, the shelf or in her hands is beyond it); near: the
    ids a look at it could be read as (the test's room: its toys lie more than NEAR_DEG apart as the child sees them, unless
    given; None: the world does not say). (Where Seen has no near, as on b301745, or no child_can_reach, as on 1ae6942, they are
    left out, so the new tests fail there at their own assertions.)"""
    reach = on in ("mat", "hand") if reach is None else reach
    for args in ((id_, name, colour, on, sees, reach, near), (id_, name, colour, on, sees, reach), (id_, name, colour, on, sees)):
        try:
            return Seen(*args)
        except TypeError:
            continue
    raise TypeError("Seen")


def seen(*objs):
    """Seen objects: (id, name, colour, on, child_sees[, child_can_reach]) (seen1)."""
    return tuple(seen1(*o) for o in objs)


TOYS = seen(("duck", "duck", "yellow", "mat", True), ("ball", "ball", "red", "mat", True), ("cup", "cup", "green", "mat", True),
            ("block", "block", "blue", "sofa", False), ("block_red", "block", "red", "mat", True))


def P(t, **kw):
    """a Percept; face_near () unless given (the test's room: no toy within NEAR_DEG of her face as the child sees it; None: the
    world does not say), left out where Percept has none (b301745 and before)."""
    kw.setdefault("seen", TOYS)
    kw.setdefault("fixtures", ROOMS)
    kw.setdefault("face_near", ())
    try:
        return Percept(t, **kw)
    except TypeError:
        kw.pop("face_near")
        return Percept(t, **kw)


def _log_has(e, w):
    """a tick of her attention log shows w in a field (since P3's eighth round each field is the list of words a look that
    follows it could be read as; before, one word)."""
    return any(e.get(f) == w or (isinstance(e.get(f), list) and w in e.get(f))
               for f in getattr(C, "LOG_FIELDS", getattr(C, "FIELDS", ())))


def script(n, seed=0):
    """a scripted moment stream: the child's target moves among toys, her face and nothing; it holds the cup at times; a toy
    falls now and then."""
    rng = np.random.default_rng(seed)
    out, tg = [], None
    for t in range(n):
        if t % 37 == 0:
            tg = [None, "duck", "ball", "mama", "cup", None][int(rng.integers(6))]
        holds = ("cup",) if (t // 150) % 3 == 1 else ()
        ev = (("fell", "ball"),) if t % 211 == 50 else ()
        out.append(P(t, child_target=tg, child_holds=holds, events=ev))
    return out


# ---------------------------------------------------------------------------------------------------- templates, line check
def test_frames_and_birth_lines():
    for k, frames in list(TP.FRAMES.items()) + list(TP.INTRO.items()) + list(TP.INTRO_WORD.items()):
        for text, focus in frames:
            ws = TP.words(text.replace("{", "").replace("}", ""))
            assert len(ws) <= K.MAX_WORDS, (k, text)
            if focus is not None:
                assert ws[-1] == focus.strip("{}"), f"{k}: {text!r}'s focus {focus!r} is not its last word"
    lines = TP.birth_lines()
    regs = {r for _, r, _ in lines}
    assert regs <= set(V.REGISTERS), regs - set(V.REGISTERS)
    for text, reg, focus in lines:
        ok, why = TP.check(text, VOCAB)
        assert ok, (text, why)
        assert focus is None or TP.words(text)[-1] == focus, (text, focus)
        assert not any(w in TP.FUNCTION for w in ([focus] if focus else [])), (text, focus)
    for intent, it in C.INTENTS.items():
        assert it.register in V.REGISTERS, (intent, it.register)
    n_growth = len(TP.GROWTH)
    no_frames = [w for w, c in TP.GROWTH if not TP.intro_frames(w)]
    print(f"1 every frame's focus is its last word and every line at most 6 words; {len(lines)} birth lines "
          f"({len({t for t, _, _ in lines})} distinct), each passing the check, in P1's registers "
          f"({', '.join(sorted(regs))}); the growth queue {n_growth} words, {len(no_frames)} with no sentence-final frame "
          f"({', '.join(no_frames)})")


def test_line_check_refuses():
    p = P(0, child_target="duck")
    ok = lambda *a, **k: TP.check(*a, **k)[0]                                  # noqa: E731
    must = [("look at the duck.", {}), ("where is the ball?", {}), ("hi pip. mama is here.", {}),
            ("the ball is on the mat.", {}),
            ("a rattle.", dict(new_word="rattle", percept=P(0, seen=TOYS + (Seen("rattle", "rattle"),)))),
            ("the ball is red!", dict(vocab=VOCAB + ("red",), refs=("ball",)))]
    for text, kw in must:
        kw = dict(kw)
        vocab = kw.pop("vocab", VOCAB)
        kw.setdefault("percept", p)
        assert ok(text, vocab, **kw), (text, TP.check(text, vocab, **kw))
    must_not = [
        ("look at the duck and the ball.", {}, "too long"), ("look, the duck.", {}, "form"), ("Look at the duck.", {}, "form"),
        ("look at  the duck.", {}, "form"), ("look at the duck", {}, "form"), ("? the duck.", {}, "form"),
        ("the rattle.", {}, "not her word"), ("a rattle is here.", dict(new_word="rattle"), "not last"),
        ("the red block.", dict(vocab=VOCAB + ("red",)), "held-out pair"),
        ("it is red.", dict(vocab=VOCAB + ("red",), refs=("block_red",)), "held-out pair (red, block) by its referent"),
        ("it is yellow.", dict(vocab=VOCAB + ("yellow",), refs=("ball",)), "unseen: a yellow"),
        ("look at the drum.", {}, "unseen: 'drum'"), ("look at the window.", dict(percept=P(0, fixtures=frozenset())), "unseen"),
        ("your foot.", dict(percept=P(0, child_in_view=False)), "unseen: the child's"),
        ("the ball fell.", dict(vocab=VOCAB + ("fell",)), "not seen happen"),
        ("yes. the duck!", dict(source="claude"), "praise"), ("look at the duck.", dict(source="claude", register="approval"),
                                                                "praise"),
        ("the duck.", dict(percept=P(0, present=False, seen=())), "unseen"),
    ]
    for text, kw, why in must_not:
        kw = dict(kw)
        vocab = kw.pop("vocab", VOCAB)
        kw.setdefault("percept", p)
        got, reason = TP.check(text, vocab, **kw)
        assert not got and why.split(":")[0] in reason, (text, reason, why)
    assert ok("the ball fell.", VOCAB + ("fell",), percept=p, recent_events=[(0, "fell", "ball")])
    assert ok("yes. the duck!", VOCAB, percept=p) and ok("the red block.", VOCAB + ("red",), percept=p, held=())
    print(f"2 the line check passes {len(must)} lines and refuses {len(must_not)}: too long, form (a comma, a capital, two "
          f"spaces, no ending, a mark before a word), a word not hers, the new word not last, a held-out pair by its words and by "
          f"its referent, a colour or object or fixture or body she cannot see, a past event she did not see, Claude's praise and "
          f"approval register, anything named while she is away")


def test_compose_from_percept():
    f = C.FastLayer(1)
    p = P(10, child_target="duck")
    ln = f.compose("label", 10, p, o=p.obj("duck"))
    assert ln is not None and ln.refs == ("duck",) and TP.words(ln.text)[-1] == "duck" and ln.emphasis == "duck"
    assert f.compose("label", 10, P(10, seen=()), o=Seen("drum", "drum")) is None, "a line named an unseen drum"
    got = f.compose("narrate_on", 10, p, o=p.obj("block"))
    assert got is not None and got.text == "the block is on the sofa." and got.focus == "sofa"
    assert f.compose("narrate_on", 10, P(10, fixtures=frozenset({"mat"})), o=p.obj("block")) is None, \
        "she named the sofa she cannot see"
    q = f.compose("ask_where", 10, p, o=p.obj("ball"))
    assert q.register == "question" or q.text.endswith("."), q
    acts = C.acts_for("ask_where", q.refs)
    assert all(a.kind not in ("point", "show") and a.target != "ball" for a in acts), \
        f"an ask pointed at or showed its own answer: {acts}"
    assert {a.kind for a in C.acts_for("show", ("duck",))} == {"show", "look"}
    assert C.acts_for("label", ("duck",))[0] == C.Act("look", "duck", "focus")
    assert all(k in C.ACT_KINDS for it in C.INTENTS.values() for k in (a.kind for a in it.acts))
    m = C.StubMotion()
    i = m.request(C.Act("show", "duck"), 5)
    assert m.status(i, 6) == "running" and m.status(i, 5 + C.STUB_TICKS["show"]) == "done"
    print("3 a line is filled only from what she perceives (an unseen drum, a sofa out of her view: no line); the label looks "
          "at its object through the focus word; an ask never points at or shows its answer; the stub motion runs each act "
          "for its nominal time")


# --------------------------------------------------------------------------------------------------- the fast layer, conduct
def run(con, stream, tract=None, tokens=None):
    said, judg = [], []
    for p in stream:
        t = p.tick
        s = con.tick(t, p, tract=None if tract is None else tract.get(t), token=None if tokens is None else tokens.get(t))
        if s.line is not None:
            said.append((t, s.line))
        judg += [(t,) + j for j in s.judgments]
    return said, judg


def test_variation_sets_and_repeats():
    con = C.Conduct(seed=1)
    stream = [P(t, child_target="duck" if t >= 5 else None) for t in range(400)]
    said, _ = run(con, stream)
    first = [(t, ln) for t, ln in said if ln.focus == "duck"]
    # her reading of its gaze is held 3 ticks running by her Reader, which fills the Percept (test 41): the set begins on the
    # tick the held reading is on the duck
    assert first and first[0][0] == 5, f"the follow-in set began at {first[0][0] if first else None}"
    s0 = [x for x in first if x[0] < 5 + 60]
    assert 2 <= len(s0) <= 3 and len({TP.key(ln.text) for _, ln in s0}) == len(s0), s0
    for (ta, la), (tb, _) in zip(s0, s0[1:]):
        assert tb - (ta + 3 * len(TP.words(la.text))) == K.PAUSE_RELATED, (ta, tb, la.text)
    set_starts = [first[0][0]] + [t for (ta, _), (t, _) in zip(first, first[1:]) if t - ta > 40]
    assert all(b - a >= K.SET_PER_OBJECT for a, b in zip(set_starts, set_starts[1:])), set_starts
    texts = [(t, ln.text) for t, ln in said]
    for i, (t, x) in enumerate(texts):
        for t2, x2 in texts[i + 1:]:
            assert TP.key(x2) != TP.key(x) or t2 - t >= K.SAME_LINE, (x, t, t2)
    # the call at most once per 240 ticks; the expectant pause after it
    con2 = C.Conduct(seed=1)
    out = []
    for t in range(600):
        if t in (0, 100, 250):
            con2.request("call")
        s = con2.tick(t, P(t, child_target=None))
        if s.line is not None:
            out.append((t, s.line.text, s.line.register))
    calls = [x for x in out if x[2] == "calling"]
    assert [c[0] for c in calls] == [0, 250], calls
    print(f"4 a follow-in set of {len(s0)} lines on the duck the tick her held reading settled on it, frames distinct in their "
          f"words, each 6 ticks after the last ended; sets on one object {K.SET_PER_OBJECT}+ ticks apart ({len(set_starts)} in "
          f"400 ticks); no line (its words) again within 60 ticks; the call refused at tick 100 (under 240 after the last) and "
          f"said at 250")


def test_replies_and_judgments():
    """the token output (the scaffold) drives the child's words here: exact and approximate words, stage 1 and 2, the echo."""
    def one(stage, sym_seq, target="duck", said_duck_at=None):
        con = C.Conduct(seed=3, transcriber=Transcriber(None), stage=stage, imperfect=False)    # her latency: test 37
        if said_duck_at is not None:
            con.fast.said_word["duck"] = said_duck_at
        toks = {40 + i: s for i, s in enumerate(sym_seq)}
        stream = [P(t, child_target=target if t >= 30 else None) for t in range(90)]
        con.fast.last_set["duck"] = 0                              # no follow-in set in the way
        con.fast.last_named["duck"] = 0
        said, judg = run(con, stream, tokens=toks)
        return con, said, judg
    duck = LX.WORD_ID["duck"]
    letters = [LX.LETTER_ID[c] for c in "duk"] + [LX.ID[LX.SPACE]]
    con, said, judg = one(1, [duck])
    assert [j[1:] for j in judg] == [(2, "right_name", "duck")], judg
    reply = [x for x in said if x[0] >= 40]
    assert reply and reply[0][0] == 40 + K.REPLY_AFTER and reply[0][1].intent == "confirm" and \
        reply[0][1].register == "approval", reply
    con, said, judg = one(1, letters)
    assert judg == [], f"stage 1 smiled at an approximation: {judg}"
    assert [x for x in said if x[0] >= 40][0][1].intent == "echo"
    con, said, judg = one(2, letters)
    assert [j[1:] for j in judg] == [(1, "approximation", "duck")], judg
    r = [x for x in said if x[0] >= 40][0][1]
    assert r.intent == "recast" and r.register == "approval" and TP.words(r.text)[-1] == "duck", r
    con, said, judg = one(2, letters)
    con.ledger.words["duck"]["exact"]["token"] = K.EXACT_UNTIL
    s = con.tick(200, P(200, child_target="duck"), token=None)
    for i, sym in enumerate(letters):
        s = con.tick(201 + i, P(201 + i, child_target="duck"), token=sym)
        assert not [j for j in s.judgments if j[1] == "approximation"], "an approximation smiled after 3 exact"
    con, said, judg = one(1, [duck], said_duck_at=38)
    st = con.ledger.words["duck"]
    assert st["echoes"]["token"] == 1 and st["says"]["token"] == [], f"an echo was counted as said: {st}"
    con, said, judg = one(1, [LX.WORD_ID["ball"]], target="duck")
    assert not judg, f"a wrong name smiled: {judg}"
    print("5 a right name by token: a smile of 2 and 'yes. the duck!' in approval 3 ticks later; 'duk' in stage 1: an echo, no "
          "smile; in stage 2: a recast in approval and a smile of 1, until the word is said exactly 3 times; said within 10 "
          "ticks of hers: an echo, not 'says'; a wrong name: no smile")


def test_talk_over_and_turns():
    tract = T.Tract(1)
    snd = {}
    act = np.array([4, 4, 2, 4, 2, 2, 2, 4, 2, 0])                  # lungs and voicing up, the jaw open: a sound
    for t in range(0, 200):
        snd[t] = tract.tick(act if 6 <= t < 10 or 70 <= t < 73 else None)
    con = C.Conduct(seed=2, transcriber=Transcriber(None), imperfect=False)       # her latency and misses: test 37
    con.request("answer_bid")                         # "mama is here." in the calling register (the call itself is its name
    cut_at, lines, heard = None, [], []               # alone since P3's eighth round, over before the child sounds)
    for t in range(120):
        s = con.tick(t, P(t, child_target=None), tract=snd[t])
        if s.cut and cut_at is None:
            cut_at = t
        if s.line is not None:
            lines.append((t, s.line.text))
        heard += [(t, cw) for cw in s.heard]
    assert lines[0][0] == 0 and cut_at is not None and 6 <= cut_at <= 7, (lines, cut_at)
    turns = [(t, cw) for t, cw in heard if cw.channel == "tract"]
    assert len(turns) == 2, turns
    t_end = turns[0][0]
    assert t_end == turns[0][1].end and turns[0][1].in_pause is False, turns[0]
    during = [ln for ln in lines if turns[0][1].start <= ln[0] <= t_end]
    assert not during, f"she spoke during the child's turn: {during}"
    after = [ln for ln in lines if ln[0] > t_end]
    assert after and after[0][0] == t_end + K.REPLY_AFTER, (t_end, after)
    assert turns[1][1].in_pause is True
    print(f"6 the child sounding during her line ('mama is here.', calling): the stop at tick {cut_at} (the world's playback "
          f"finishes her word); no line "
          f"while its turn runs; the turn ends after 2 quiet ticks (tick {t_end}) and her reply comes 3 ticks later; a turn in "
          f"her silence is marked in a pause")


def test_new_word_and_night():
    con = C.Conduct(seed=5)
    rattle = Seen("rattle", "rattle", "purple", "mat", True)
    p = lambda t, **k: P(t, seen=TOYS + (rattle,), **k)                 # noqa: E731
    con.request("new_word", word="rattle", o="rattle")
    said, _ = run(con, [p(t) for t in range(60)])
    nw = [ln for _, ln in said if ln.intent == "new_word"]
    assert len(nw) == 3 and all(ln.register == "new_word" and ln.emphasis == "rattle" and TP.words(ln.text)[-1] == "rattle"
                                for ln in nw), nw
    assert len({ln.frame for ln in nw}) == 3
    assert not TP.check("the rattle is here.", con.fast.vocab, "rattle", p(0))[0]
    assert "rattle" not in con.fast.vocab
    con.fast.night()
    assert "rattle" in con.fast.vocab and con.fast.new_words == ()
    assert TP.check("the rattle is on the mat.", con.fast.vocab, None, p(0))[0]
    con2 = C.Conduct(seed=5)
    con2.request("new_word", word="rattle", o="rattle")
    said2, _ = run(con2, [P(t) for t in range(60)])
    assert not [ln for _, ln in said2 if ln.intent == "new_word"], "a new word introduced with its referent unseen"
    ok, why = TP.showable("toes", {})
    assert not ok and "toes" in why
    assert TP.showable("rattle", dict(objects={"rattle": ["purple"]}))[0]
    assert not TP.showable("red", dict(objects={"ball": ["red"]}))[0]
    assert TP.showable("red", dict(objects={"ball": ["red"], "block": ["blue", "red"]}))[0]
    assert not TP.showable("and", {})[0]
    print("7 a new word's set: 3 lines from 3 frames, each ending on the word, in the new-word register and emphasized; not "
          "said elsewhere than last; it joins her words at the night; refused with its referent unseen; the queue waits for "
          "what the world can show (no toes on a G1, colour twins for 'red', no sentence-final frame for 'and')")


def test_steer():
    f = C.FastLayer(1)
    assert not f.add_steer("yes. the duck!", "any", 0)[0], "Claude's praise accepted"
    assert not f.add_steer("the rattle is here.", "any", 0)[0]
    assert f.add_steer("you see your cup.", "holds:cup", 0)[0]
    con = C.Conduct(seed=1)
    con.fast = f
    said, _ = run(con, [P(t, child_holds=("cup",) if t >= 100 else ()) for t in range(900)])
    st = [(t, ln) for t, ln in said if ln.source == "claude"]
    assert st and st[0][0] >= 100 and len(st) == K.STEER_USES, st
    assert all(ln.register == "plain" for _, ln in st)
    print(f"8 Claude's lines: praise and a word not hers refused; a line tied to 'holds:cup' said only once the child held the "
          f"cup, at most {K.STEER_USES} times, in the plain register")


# ------------------------------------------------------------------------- the P3 verifier's findings, each tested both ways
def _letters(word):
    return [LX.LETTER_ID[c] for c in word] + [LX.ID[LX.SPACE]]


def _token_turn(stage, syms, target=None, holds=(), seen_=TOYS, pre=None, t0=40, n=70, routine=None):
    """the child's token output at t0.. in a quiet moment; -> (conduct, lines said from t0, judgments, heard words)."""
    con = C.Conduct(seed=3, transcriber=Transcriber(None), stage=stage)
    con.routine = routine
    for o in ("duck", "ball", "cup", "block", "block_red", "ball_blue"):
        con.fast.last_set[o] = con.fast.last_named[o] = 0                   # no follow-in set in the way
    if pre is not None:
        pre(con)
    toks = {t0 + i: sym for i, sym in enumerate(syms)}
    said, judg, heard = [], [], []
    for t in range(n + t0):
        s = con.tick(t, P(t, child_target=target if t >= 20 else None, child_holds=holds, seen=seen_), token=toks.get(t))
        if s.line is not None and t >= t0:
            said.append((t, s.line))
        judg += [j for j in s.judgments]
        heard += s.heard
    return con, said, judg, heard


def test_approximation_in_context():
    """finding 1: stage 2's approximation smiled at a letter string with nothing in view ("z" read as "a", "qp" as "up", "bal"
    as "ball"). An approximation now earns its smile only where the exact word would be a right name."""
    for letters, word in (("z", "a"), ("x", "a"), ("qp", "up"), ("bal", "ball")):
        con, said, judg, heard = _token_turn(2, _letters(letters))
        assert [cw.word for cw in heard] == [word], (letters, heard)
        assert judg == [], f"{letters!r} read as {word!r} with nothing in view earned {judg}"
        assert not [ln for _, ln in said if ln.intent in ("recast", "recast_word")], f"{letters!r} was recast: {said}"
        assert not con.ledger.words[word]["says"]["token"], f"{letters!r} counted toward 'says' of {word!r}"
    con, said, judg, _ = _token_turn(2, [LX.WORD_ID["ball"]])
    assert judg == [], "the control: an exact 'ball' with no ball in view smiled"
    con, said, judg, _ = _token_turn(2, _letters("bal"), target="ball")
    assert judg == [(1, "approximation", "ball")] and said[0][1].intent == "recast", (judg, said[:1])
    con, said, judg, _ = _token_turn(2, _letters("bal"), holds=("ball",))
    assert judg == [(1, "approximation", "ball")], judg
    # the ear's wider set (her last focus word) lets her hear it, never smile at it
    con, said, judg, _ = _token_turn(2, _letters("duk"), pre=lambda c: setattr(c.fast, "last_focus", "duck"))
    assert judg == [] and said[0][1].intent in ("echo", "echo_word"), (judg, said[:1])
    con, said, judg, _ = _token_turn(2, [LX.WORD_ID["duck"]], pre=lambda c: setattr(c.fast, "last_focus", "duck"))
    assert judg == [], "an exact word she only expected as an echo smiled"
    # the tract's approximation: the ear accepted it among her expected words; the smile still needs its referent
    con = C.Conduct(seed=3, stage=2)
    out = C.Say()
    con._owe_reply(50, ChildWord(50, "tract", "duck", False, 45, 50, ("duck", "mama")), P(50, child_target=None), out)
    assert out.judgments == [] and con.reply_due["kind"] == "echo", (out.judgments, con.reply_due)
    out = C.Say()
    con._owe_reply(50, ChildWord(50, "tract", "duck", False, 45, 50, ("duck",)), P(50, child_target="duck"), out)
    assert out.judgments == [(1, "approximation", "duck")], out.judgments
    # "mama" at her face: a right name, as the ledger's "says" has it
    con, said, judg, _ = _token_turn(2, [LX.WORD_ID["mama"]], target="mama")
    assert judg == [(2, "right_name", "mama")], judg
    print("17 stage 2: 'z' and 'x' (read as 'a'), 'qp' ('up'), 'bal' ('ball') with nothing in view: no smile, no recast, not "
          "'says'; 'bal' with the ball where she reads it looking or in its hand: a recast and a smile of 1; 'duk' or 'duck' "
          "when she only expects it as an echo of her last line: echoed, no smile (the approximation never looser than the "
          "exact word); the tract's approximation likewise; 'mama' at her face: a right name")


def test_asks_answered():
    """finding 2: an ask met without an answer (a twin already where she reads it looking; the call while it already looks at
    her; a look before the word was heard)."""
    twins = TOYS + seen(("ball_blue", "ball", "blue", "mat", True))

    def ask_run(target_at, intent="ask_where", o="ball", n=60):
        con = C.Conduct(seed=4, transcriber=Transcriber(None))
        for x in ("duck", "ball", "cup", "block", "block_red", "ball_blue"):
            con.fast.last_set[x] = con.fast.last_named[x] = -1000
        con.request(intent, o=o) if o else con.request(intent)
        said, judg = [], []
        for t in range(n):
            tg = target_at(t, con)
            s = con.tick(t, P(t, child_target=tg, seen=twins))
            if s.line is not None:
                said.append((t, s.line))
            judg += s.judgments
        return con, said, judg
    # asked of the red ball while she reads it looking at the blue one: not asked (it would be met unasked)
    con, said, judg = ask_run(lambda t, c: "ball_blue")
    assert not [ln for _, ln in said if ln.intent == "ask_where"] and judg == [], (said, judg)
    assert any("already where she reads the child looking" in r[2] for r in con.fast.refused), con.fast.refused[-3:]
    # its gaze lands on a ball while she says the line, before "ball" is heard: void, no smile, not counted either way
    con, said, judg = ask_run(lambda t, c: "ball_blue" if 1 <= t else None)
    ask = [ln for _, ln in said if ln.intent == "ask_where"]
    assert ask and judg == [], (said, judg)
    st = con.ledger.standing("ball")
    assert st["asks"] == [], f"a look before the word was heard counted: {st}"
    # the look lands after the word was heard: met, a smile of 2, counted
    con, said, judg = ask_run(lambda t, c: "ball_blue" if (c.pending is None and t > 15) or
                              (c.pending is not None and t >= c.pending["open"] + 3) else None)
    assert [j[1] for j in judg] == ["met_ask"] and con.ledger.standing("ball")["asks"] == [1], (judg, said)
    # the call while it already looks at her face: not called
    con, said, judg = ask_run(lambda t, c: "mama", intent="call", o=None)
    assert not [ln for _, ln in said if ln.intent == "call"] and judg == [], (said, judg)
    # it turns to her while she says its name (before "pip" has ended): void; after: met
    con, said, judg = ask_run(lambda t, c: "mama" if 1 <= t else None, intent="call", o=None)
    assert [ln.intent for _, ln in said][:1] == ["call"] and judg == [], (said, judg)
    con, said, judg = ask_run(lambda t, c: "mama" if (c.pending is not None and t >= c.pending["open"] + 2) else
                              (None if c.pending is not None else "mama" if t > 3 else None), intent="call", o=None)
    assert [j[1] for j in judg] == ["met_ask"], (judg, said)
    # a spoken word is no answer to a gaze ask: saying "ball" after "where is the ball?" is not a met ask
    con = C.Conduct(seed=4, transcriber=Transcriber(None))
    con.request("ask_where", o="ball")
    judg = []
    for t in range(40):
        s = con.tick(t, P(t, child_target=None, seen=twins), token=LX.WORD_ID["ball"] if t == 30 else None)
        judg += s.judgments
    assert not [j for j in judg if j[1] == "met_ask"], f"a word met a gaze ask: {judg}"
    print("18 an ask of the red ball while she reads it looking at the blue one is not made (logged); a look landing on a "
          "ball before its word is heard voids the ask (no smile, not counted); a look after it: met, a smile of 2, counted; "
          "the call is not made while it looks at her face, a turn during its name is void, a turn after it is met; a spoken "
          "word never meets a gaze ask")


class _FakeClip:
    def __init__(self, words, n):
        self.words, self.pcm, self.key, self.digest = words, np.zeros(n * 2400, np.int16), "fake", "fake"


class _FakeVoice:
    """a voice whose clips have given word spans (samples), for the talk-over's arithmetic."""

    def __init__(self, spans):
        self.spans = spans

    def clip(self, text, register, emphasis=None):
        ws = TP.words(text)
        sp = self.spans.get(text) or [(w, 3 * i * 2400, (3 * i + 3) * 2400) for i, w in enumerate(ws)]
        return _FakeClip(sp, -(-sp[-1][2] // 2400))


def test_cut_words_not_said():
    """finding 3: a word the talk-over stopped was still recorded as said (her echo window, the ledger's heard and said
    counts, base-rate voiding). Now a word is said only as its sound ends."""
    act = np.array([4, 4, 2, 4, 2, 2, 2, 4, 2, 0])

    def go(voice, cut_at, n=12, tok=None, target=lambda t: "duck"):
        """her line "look. the duck." from tick 0, the child sounding at cut_at and cut_at + 1; -> (conduct, cuts, what she
        had said by tick 9, the line's own end: her echo window, her last focus, the ledger's said counts)."""
        tr = T.Tract(1)
        snd = {t: tr.tick(act if cut_at <= t < cut_at + 2 else None) for t in range(n)}
        con = C.Conduct(seed=2, transcriber=Transcriber(None), voice=voice)
        con.fast.last_set["duck"] = con.fast.last_named["duck"] = -1000
        line = TP.Line("look. the duck.", "label", "plain", "duck", "duck", ("duck",))
        con._say(line, 0, P(0, child_target="duck"), C.Say())
        cuts, at9 = [], None
        for t in range(1, n):
            s = con.tick(t, P(t, child_target=target(t)), tract=snd[t], token=(tok or {}).get(t))
            if s.cut:
                cuts.append(t)
            if t == 9:
                at9 = (dict(con.fast.said_word), con.fast.last_focus, {w: dict(said=v["said"], heard=v["heard"])
                                                                      for w, v in con.ledger.words.items()},
                       dict(con.ledger.said_ticks))
        return con, cuts, at9
    con, cuts, (said, focus, counts, ticks) = go(None, 3, n=40, tok={16: LX.WORD_ID["duck"]},
                                                 target=lambda t: "duck" if t < 3 or t >= 14 else None)
    assert cuts == [3], cuts
    assert "duck" not in said and said.get("the") == 5, said
    assert "duck" not in counts and "duck" not in ticks and counts["the"]["said"] == 1, counts
    assert focus is None, focus
    st = con.ledger.words["duck"]
    assert st["echoes"]["token"] == 0 and len(st["says"]["token"]) == 1, \
        f"the child's own 'duck' after a cut 'duck' was taken as an echo: {st}"
    # with a voice: "duck" lengthened to 6 ticks is broken off at 3 ticks and never said; "the" finishing within 3 is said
    spans = {"look. the duck.": [("look", 0, 2400), ("the", 2400, 4800), ("duck", 4800, 4800 + 6 * 2400)]}
    con, cuts, (said, focus, counts, _) = go(_FakeVoice(spans), 3)
    assert cuts == [3] and "duck" not in said and "duck" not in counts and said.get("the") == 1, (said, counts)
    con, cuts, (said, focus, counts, _) = go(_FakeVoice(spans), 7)         # cut 2 ticks from "duck"'s end: finished
    assert said.get("duck") == 7 and focus == "duck" and counts["duck"]["heard"] == 1, (said, focus, counts)
    con, cuts, (said, focus, counts, _) = go(None, 30)                     # not cut: every word said as it ends
    assert said == {"look": 2, "the": 5, "duck": 8} and counts["duck"]["heard"] == 1 and focus == "duck", (said, counts)
    print("19 'look. the duck.' cut at tick 3: 'the' finished and said, 'duck' withdrawn: not in her echo window, not said or "
          "heard in the ledger, not her last focus, so the child's own 'duck' at tick 16 counts toward 'says'; a lengthened "
          "word 6 ticks long broken off at 3 ticks is not said, one that ends within 3 ticks is")


def test_new_words_in_a_day():
    """finding 4: a second new word in one day erased the first ("rattle" then "ring": only "ring" joined at the night). The
    requests come a minute apart (NEW_EVERY, A14; test 28 holds the pace)."""
    toys = TOYS + seen(("rattle", "rattle", "purple", "mat", True), ("ring", "ring", "green", "mat", True),
                       ("stacker", "stacker", "orange", "mat", True), ("book", "book", "white", "mat", True))
    con = C.Conduct(seed=5)
    said = []
    for t in range(4 * K.NEW_EVERY):
        if t % K.NEW_EVERY == 0:
            k = t // K.NEW_EVERY
            con.request("new_word", word=("rattle", "ring", "stacker", "book")[k], o=("rattle", "ring", "stacker", "book")[k])
        s = con.tick(t, P(t, seen=toys))
        if s.line is not None:
            said.append((t, s.line))
    by = {}
    for _, ln in said:
        if ln.intent == "new_word":
            by.setdefault(ln.focus, []).append(ln)
    assert sorted(by) == ["rattle", "ring", "stacker"] and all(len(v) == 3 for v in by.values()), by
    assert con.fast.new_words == ("rattle", "ring", "stacker")
    assert any("at most 3 new words" in r[2] for r in con.fast.refused), "the fourth new word was not refused and logged"
    assert TP.check("the rattle.", con.fast.vocab, con.fast.new_words, P(0, seen=toys), ())[0], "the first new word lost"
    ok, why = TP.check("the ring rattle.", con.fast.vocab, con.fast.new_words, P(0, seen=toys), ())
    assert not ok and "two new words" in why, why
    con.fast.night()
    assert all(w in con.fast.vocab for w in ("rattle", "ring", "stacker")) and con.fast.new_words == ()
    assert "book" not in con.fast.vocab
    # her ear must know a word before she introduces it (built with every word before birth), and is checked at the night
    ear = tiny_ear()
    con2 = C.Conduct(seed=5, transcriber=Transcriber(ear), vocab=tuple(SOUNDS))
    con2.request("new_word", word="rattle", o="rattle")
    for t in range(5):
        con2.tick(t, P(t, seen=toys))
    assert con2.fast.new_words == () and any("no templates" in r[2] for r in con2.fast.refused), con2.fast.refused[-2:]
    con2.night()
    con3 = C.Conduct(seed=5, transcriber=Transcriber(ear), vocab=tuple(SOUNDS) + ("cup",))
    try:
        con3.night()
        raise AssertionError("a night passed with a word her ear cannot hear")
    except ValueError:
        pass
    print("20 three new words in a day (rattle, ring, stacker): each its set of 3, each sayable (last) until the night, all "
          "three hers after it; a fourth refused and logged; two new words in one line refused; a word her ear has no "
          "templates for is never introduced, and a night with such a word in her vocabulary stops")


def test_acts_carry_the_object():
    """finding 5: the motion interface dropped the object of a line with no object slot ("what is this?", "a rattle."). (Since
    P3's sixth round a name ask shows nothing, A51's attention log: "what is this?" is asked of what the child attends.)"""
    toys = TOYS + seen(("rattle", "rattle", "purple", "mat", True))
    held = tuple(s_ for s_ in toys if s_.id != "cup") + seen(("cup", "cup", "green", "hand", True))
    con = C.Conduct(seed=6)
    _no_sets(con)
    con.request("ask_what", o="cup")
    s = None
    for t in range(3):
        s = con.tick(t, P(t, seen=held, child_holds=("cup",)))
        if s.line is not None:
            break
    assert s.line.intent == "ask_what" and s.line.refs == ("cup",) and s.acts == (C.Act("look", "child_eyes"),), \
        (s.line, s.acts)
    con1 = C.Conduct(seed=6)
    _no_sets(con1)
    con1.request("ask_what", o="cup")
    assert all(con1.tick(t, P(t, seen=toys)).line is None for t in range(3))          # a cup it does not attend: not asked
    assert any(r[1] == "ask_what" and "what the child attends" in r[2] for r in con1.fast.refused), con1.fast.refused[-1:]
    con2 = C.Conduct(seed=6)
    con2.request("new_word", word="rattle", o="rattle")
    got = []
    for t in range(60):
        s = con2.tick(t, P(t, seen=toys))
        if s.line is not None and s.line.intent == "new_word":
            got.append(s.acts)
    assert len(got) == 3 and all(C.Act("show", "rattle") in a for a in got), got
    assert C.acts_for("new_word", (), w="table", word_class="fixture")[0] == C.Act("point", "table")
    assert C.acts_for("new_word", (), w="clap", word_class="verb")[0] == C.Act("do", "clap")
    assert C.acts_for("new_word", ("ball",), w="red", word_class="colour")[0] == C.Act("show", "ball")
    assert not [a for a in C.acts_for("ask_where", ("ball",)) if a.kind in ("show", "point")]
    kinds = {a.kind for m in (con.motion, con2.motion) for a in [C.Act(x[2], x[3], x[4]) for x in m.acts]}
    assert "show" in kinds, kinds
    print("21 the motion interface is asked to show: 'what is this?' is asked of the cup the child holds, her eyes on it and "
          "nothing shown (A51's log; not asked of a cup it does not attend); each of a new toy's 3 lines shows the toy (A15); a "
          "new fixture is pointed at, a verb done, a colour shown on its toy; an ask for a gaze never shows or points at its "
          "answer")


def test_line_check_truth():
    """finding 8: the line check tested only that things exist ("it is a duck." while the child holds the ball passed)."""
    held_ball = P(0, child_holds=("ball",))
    for text, p, why in (("it is a duck.", held_ball, "not true"), ("this is the duck.", held_ball, "not true"),
                         ("the ball is on the sofa.", P(0), "not true"), ("the block is on the mat.", P(0, seen=seen(
                             ("block", "block", "blue", "sofa", True))), "not true"),
                         ("the ball is down.", P(0), "not seen happen"), ("uh oh. the ball is down.", P(0), "not seen happen"),
                         ("the ball is in the sofa.", P(0), "containment"),
                         ("you see the block.", P(0, seen=seen(("block", "block", "blue", "sofa", False))), "not true")):
        ok, reason = TP.check(text, VOCAB, None, p, (), source="claude")
        assert not ok and why in reason, (text, reason)
    for text, p, ev in (("it is a ball.", held_ball, ()), ("the ball is on the mat.", P(0), ()),
                        ("the ball is down.", P(0), [(0, "fell", "ball")]), ("you see the duck.", P(0), ())):
        ok, reason = TP.check(text, VOCAB, None, p, (), source="claude", recent_events=ev)
        assert ok, (text, reason)
    for text in ("where is the duck?", "the duck? where is the duck?", "what is this?", "give me the cup.",
                 "look at the duck.", "pip. look at mama."):
        ok, reason = TP.check(text, VOCAB, None, P(0), (), source="claude")
        assert not ok and "ask" in reason, (text, reason)
        assert TP.check(text, VOCAB, None, P(0, child_target="duck"), ("duck",) if "duck" in text else
                        (("cup",) if "cup" in text else ()))[0] or text.startswith("what"), text
    f = C.FastLayer(1)
    assert not f.add_steer("where is the duck?", "any", 0)[0]
    # the fast layer's own lines are true by construction and still pass
    p = P(10, child_target="duck", child_holds=("cup",))
    for intent, o in (("label", "duck"), ("label_held", "cup"), ("narrate_on", "block"), ("confirm", "duck"),
                      ("reply", "duck"), ("show", "cup")):
        assert f.compose(intent, 10, p, o=p.obj(o)) is not None, intent
    assert f.compose("narrate_fell", 10, p, o=p.obj("ball")) is None, "narrated a fall she did not see"
    f.observe(P(11, events=(("fell", "ball"),)))
    assert f.compose("narrate_fell", 12, p, o=p.obj("ball")) is not None
    print("22 the line check holds what a line says, not only what it names: 'it is a duck.' while the child holds the ball, "
          "'the ball is on the sofa.' of the ball on the mat, 'the ball is down.' with no fall seen, a thing 'in' what she "
          "cannot see into, 'you see the block.' of a block out of its view: refused; the same lines true: passed; Claude's "
          "asks refused (an ask is judged: the fast layer's); the fast layer's own lines pass, a fall narrated only once seen")


def test_small_items():
    """finding 9: 'what is this?' with no object stored the word None (saved as "null"); dropped requests unlogged; Claude's
    line with the day's new word neither emphasized nor in its register; the redirect's 40 ticks unenforced."""
    con = C.Conduct(seed=7)
    con.request("ask_what")
    for t in range(5):
        con.tick(t, P(t))
    assert not con.ledger.trials and None not in con.ledger.words and "null" not in con.ledger.words
    assert any(r[1] == "ask_what" and "needs its object" in r[2] for r in con.fast.refused), con.fast.refused
    try:
        con.ledger.ask(9, None, "name", None, 20)
        raise AssertionError("the ledger took an ask with no word")
    except ValueError:
        pass
    con.request("redirect", o="drum")
    con.request("redirect", o="ball")
    con.tick(6, P(6))
    why = [r[2] for r in con.fast.refused[-2:]]
    assert "unseen: 'drum'" in why[0] and "redirect only after" in why[1], why
    s = None
    con.fast.follow_in = K.FOLLOW_PER_REDIRECT                   # two follow-in namings today (4.10's 2 to 1: test 36)
    for t in range(40, 60):
        con.request("redirect", o="ball") if t == 45 else None
        s = con.tick(t, P(t))
        if s.line is not None:
            break
    assert s.line is not None and s.line.intent == "redirect", "the redirect after 40 ticks with no target was not said"
    # the ledger's save round-trips exactly through JSON, with an ask open and a line sounding
    con2 = C.Conduct(seed=8, transcriber=Transcriber(None))
    con2.request("ask_where", o="ball")
    for t in range(3):
        con2.tick(t, P(t))
    st = con2.state()
    st2 = json.loads(json.dumps(st, default=_np))
    con3 = C.Conduct(seed=8, transcriber=Transcriber(None))
    con3.load_state(st2)
    assert json.dumps(con3.state(), default=_np, sort_keys=True) == json.dumps(st, default=_np, sort_keys=True)
    # Claude's line ending on the day's new word: the new-word register, the word emphasized
    toys = TOYS + seen(("rattle", "rattle", "purple", "mat", True))
    f = C.FastLayer(1)
    f.new_words = ("rattle",)
    assert f.add_steer("you see the rattle?", "any", 0)[0]                  # on its peak ("you see the rattle." ties: A34)
    ln, _kind = f.steer_line(10, P(10, seen=toys, child_target="rattle"))    # a follow-in naming (else a redirect: test 36)
    assert ln.register == "new_word" and ln.emphasis == "rattle" and ln.focus == "rattle", ln
    ln2 = C.FastLayer(1)
    ln2.add_steer("you see your cup.", "any", 0)
    got, _kind = ln2.steer_line(10, P(10, child_holds=("cup",)))
    assert got.register == "plain" and got.emphasis == "cup", got
    assert "'cancelled'" in C.StubMotion.__doc__
    assert not hasattr(TP, "_last_is_focus")
    print("23 'what is this?' with no object is not asked (logged), and the ledger refuses a word of None; every dropped "
          "request is logged (an unseen object, the redirect before 40 ticks with no target), and the redirect is said after; "
          "a saved conduct with an ask open and a line sounding round-trips JSON exactly; Claude's line ending on the day's new "
          "word in the new-word register, emphasized; Claude's other lines emphasize their last content word")


# ---------------------------------------------------------------------------------------------------------- the transcriber
def test_transcriber():
    assert read_letters("duck", VOCAB) == ("duck", True, 0)
    assert read_letters("duk", VOCAB)[:2] == ("duck", False)
    assert read_letters("bal", VOCAB)[0] == "ball"
    assert read_letters("peekabo", VOCAB)[0] == "peekaboo" and read_letters("pekabo", VOCAB)[0] == "peekaboo"
    assert read_letters("bo", VOCAB, near={"bottle"})[:2] == ("bottle", False)
    assert read_letters("bo", VOCAB)[0] == "no" and read_letters("zqx", VOCAB)[0] is None
    assert read_letters("rattle", VOCAB)[0] != "rattle", "a queue word read before it entered"
    assert read_letters("rattle", VOCAB + ("rattle",))[0] == "rattle"
    tx = Transcriber(None)
    got = []
    seq = {0: LX.LETTER_ID["b"], 1: LX.LETTER_ID["a"], 2: LX.LETTER_ID["l"], 3: LX.LETTER_ID["l"], 4: LX.ID[LX.SPACE],
           10: LX.WORD_ID["duck"], 20: LX.LETTER_ID["c"], 21: LX.LETTER_ID["u"], 22: LX.LETTER_ID["p"]}
    for t in range(40):
        speaking = 8 <= t < 15
        got += tx.tick(t, None, 1.0, seq.get(t), speaking, (), P(t))
    words = [(cw.tick, cw.word, cw.exact) for cw in got]
    assert words == [(4, "ball", True), (15, "duck", True), (24, "cup", True)], words
    p = P(0, child_target="duck", child_holds=("cup",))
    exp = expected_words(p, dict(word="ball"), "mama", "feed", VOCAB)
    assert exp == ("mama", "ball", "duck", "cup", "bottle", "more"), exp
    assert expected_words(P(0, present=False, seen=()), None, None, None, VOCAB) == ("mama",)
    assert "rattle" not in expected_words(P(0, child_holds=("r",), seen=(Seen("r", "rattle"),)), None, None, None, VOCAB)
    print("9 the token output read as a transcript: a token exactly, letters exactly, within edit distance 1 (2 at 6 letters), "
          "a 2-letter prefix only of what it sees or holds, and what it sees or holds first ('bo' holding the bottle is "
          "'bottle', else 'no'), a queue word not before it entered; a token made during her line "
          "read when it ends; her expected set is A27's (names where she reads it looking and in its hand, the ask, her "
          "last focus, the routine; 'mama' when away), kept to her words")


# --------------------------------------------------------------------------------------------------------------- the ear
SOUNDS = {                                                             # an ear for the rules, built from the tract's own sounds
    "ball": dict(jaw=0.8, tongue_front=0.0, tongue_height=0.0, lips=0.9, rounding=0.2),         # an open /a/
    "see": dict(jaw=0.15, tongue_front=1.0, tongue_height=0.85, lips=0.8, rounding=0.0),        # a front /i/
    "duck": dict(jaw=0.3, tongue_front=0.2, tongue_height=0.7, lips=0.3, rounding=1.0),         # a rounded /u/
    "mama": dict(jaw=0.4, tongue_front=0.5, tongue_height=0.4, lips=0.0, velum=1.0),            # a nasal murmur
    "hi": dict(jaw=0.5, tongue_front=0.7, tongue_height=0.5, lips=0.6, rounding=0.0),           # a mid vowel
}


REPLAY_SOUNDS = dict(SOUNDS, rattle=dict(jaw=0.6, tongue_front=0.3, tongue_height=0.3, lips=0.7, rounding=0.6))


def say(word, pitch, n=4, seed=11):
    tr = T.Tract(seed)
    x = T.NEUTRAL.copy()
    for k, v in {**dict(lungs=0.6, glottis=0.7, velum=0.0, pitch=pitch), **REPLAY_SOUNDS[word]}.items():
        x[T.NAMES.index(k)] = v
    tr.x, tr.target = x.copy(), x.copy()
    y = np.concatenate([tr.tick(np.full(10, 2)) for _ in range(n)] + [tr.tick(None) for _ in range(2)])
    return y * V.PA_PER_UNIT


def babble(seed, ticks):
    """the voice study's babbler (as tools/sim_parent_ear.py): a unit of 1-8 ticks with probability 1 / 14.5 a tick, each
    articulator's step drawn and held through it."""
    rng = np.random.default_rng(seed)
    tr = T.Tract(seed)
    out = []
    while len(out) < ticks:
        if rng.random() < 1 / 14.5:
            a = rng.integers(0, 5, T.N_ART)
            out += [tr.tick(a) for _ in range(int(rng.integers(1, 9)))]
        else:
            out.append(tr.tick(None))
    return out[:ticks]


_BANK = []


def tiny_ear(margins=None, deltas=None, sounds=SOUNDS):
    tmpl = {w: [(f"p{p}", PE.template_of(say(w, p, seed=3 + i))) for i, p in enumerate((0.25, 0.45, 0.65))] for w in sounds}
    if not _BANK:
        tx = Transcriber(None)
        for t, y in enumerate(babble(4, 1500)):
            if tx.tick(t, y, 1.0, None, False, (), P(t)):
                _BANK.append(PE.template_of(tx.last_x))
    return PE.ParentEar(tmpl, list(_BANK), margins or {k: -2.0 for k in range(1, 9)},
                        deltas if deltas is not None else {k: 1e9 for k in range(1, 9)})


def test_ear_rules():
    ear = tiny_ear()
    words = tuple(SOUNDS)
    x = say("ball", 0.35, seed=21)
    h = ear.hear(x, ("ball", "see"), words)
    assert h.word == "ball" and h.exact and h.nearest_all == "ball", h.as_dict()
    for exp in (("see", "duck"), ("duck",), ("mama", "hi", "see")):
        h2 = ear.hear(x, exp, words)
        assert h2.word != "ball", f"her ear accepted 'ball' when she expected {exp}: {h2.as_dict()}"
        if h2.word is not None:
            assert not h2.exact, f"a word she heard in place of the one said was called exact: {h2.as_dict()}"
    assert ear.hear(x, (), words).word is None, "a word accepted with nothing expected"
    assert ear.hear(x, ("ball",), ("see", "duck")).word is None, "a word not hers accepted"
    loose = tiny_ear({k: -50.0 for k in range(1, 9)})
    hx = loose.hear(x, ("see", "duck"), words)
    assert hx.word in ("see", "duck") and not hx.exact and hx.nearest_all == "ball", hx.as_dict()
    strict = tiny_ear({k: -50.0 for k in range(1, 9)}, {k: 0.0 for k in range(1, 9)})     # delta at its tightest
    hs = strict.hear(x, ("see", "duck"), words)
    assert hs.word is None and hs.why == "heard as 'ball', not close to %r" % hs.nearest, hs.as_dict()
    assert strict.hear(x, ("ball", "see"), words).exact, "an exact word refused by delta"
    tight = tiny_ear({k: 50.0 for k in range(1, 9)})
    assert tight.hear(x, ("ball",), words).word is None, "accepted past a margin no sound can meet"
    # the bar never moves: hearing leaves the ear as it was, and the same sound is heard the same way before and after
    d0, probe = ear.digest, ear.hear(x, ("ball", "see"), words).as_dict()
    for i in range(60):
        w = words[i % len(words)]
        ear.hear(say(w, 0.3 + 0.005 * i, seed=100 + i), (w, words[(i + 1) % len(words)]), words)
    assert ear.digest == d0 and ear.hear(x, ("ball", "see"), words).as_dict() == probe, "her ear changed with what it heard"
    assert not hasattr(ear, "add_template"), "the ear can learn the child's words"
    for bad in (lambda: setattr(ear, "bank", ()), lambda: setattr(ear, "margins", {1: 9.0})):
        try:
            bad()
            raise AssertionError("the ear was changed after its build")
        except AttributeError:
            pass
    try:
        ear._w["ball"][0][0, 0] = 1.0
        raise AssertionError("a template was written in place")
    except ValueError:
        pass
    # finding 7: what she scores with was writable (the stacks' arrays, the margins in place): no write reaches her ear now
    st = ear._stacks["ball"]
    writes = [lambda: ear.margins.__setitem__(2, 9.0), lambda: ear.deltas.__setitem__(1, 0.0),
              lambda: ear._w.__setitem__("ball", ()), lambda: ear._stacks.__setitem__("ball", None),
              lambda: st.flat.__setitem__((0, 0), 1.0), lambda: st.t2.__setitem__(0, 0.0),
              lambda: st.L.__setitem__(0, 1), lambda: st.valid.__setitem__((0, 0), False),
              lambda: ear._bank.flat.__setitem__((0, 0), 1.0), lambda: ear.bank[0].__setitem__((0, 0), 1.0),
              lambda: st.flat.setflags(write=True), lambda: ear.bank[0].setflags(write=True),
              lambda: setattr(st, "flat", np.zeros((1, PE.NCEP))), lambda: ear.labels.__setitem__("ball", ()),
              lambda: ear.order.__setitem__("ball", 9)]
    for i, bad in enumerate(writes):
        try:
            bad()
            raise AssertionError(f"write {i} reached her ear")
        except (ValueError, TypeError, AttributeError):
            pass
    assert ear.verify() and ear.hear(x, ("ball", "see"), words).as_dict() == probe
    assert ear.margin(3) == -2.0 and ear.margin(40) == ear.margins[8]
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "ear.npz")
        ear.save(path)
        e2 = PE.ParentEar.load(path)
        assert e2.digest == ear.digest and e2.hear(x, ("ball", "see"), words).as_dict() == probe
        z = dict(np.load(path))
        z["bank"] = z["bank"] + 1e-9
        np.savez(path.replace(".npz", "_x.npz"), **z)
        try:
            PE.ParentEar.load(path.replace(".npz", "_x.npz"))
            raise AssertionError("a changed ear file was loaded")
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    long = np.concatenate([say("ball", 0.4, n=40)])
    h3 = ear.hear(long, ("ball",), words)
    assert h3.word is None and h3.scored == 0, "an utterance longer than any template can match was scored"
    print(f"10 her ear (built from the tract's sounds): 'ball' accepted and exact when expected; never heard as 'ball' when she "
          f"did not expect it, and a word heard in its place never exact; nothing expected, nothing heard; a word not hers not "
          f"heard; a word she expected but not the nearest of all hers accepted as an approximation only, and refused when it is not within delta of the nearest (an exact word always passes delta); 60 utterances later "
          f"the same sound heard the same way and the digest unchanged; no add_template, no writes (attributes, arrays, the "
          f"stacks she scores with, the margins in place: {len(writes)} tried, none reached her ear, verify() holds); saved "
          f"and loaded to the same digest, a changed file refused; an utterance too long for any template refused unscored")


def test_transcriber_with_ear():
    ear = tiny_ear()
    words = tuple(SOUNDS)
    tx = Transcriber(ear)
    tr = T.Tract(9)
    x = T.NEUTRAL.copy()
    for k, v in {**dict(lungs=0.6, glottis=0.7, velum=0.0, pitch=0.4), **SOUNDS["duck"]}.items():
        x[T.NAMES.index(k)] = v
    got = []
    for t in range(30):
        if t == 5:
            tr.x, tr.target = x.copy(), x.copy()
        y = tr.tick(np.full(10, 2) if 5 <= t < 9 else None)
        got += tx.tick(t, y, 0.8, None, False, ("duck", "see"), P(t), {"duck": -20}, vocab=words)
    tw = [cw for cw in got if cw.channel == "tract"]
    assert len(tw) == 1 and tw[0].word == "duck" and tw[0].start == 5 and tw[0].tick == tw[0].end, tw
    assert not tw[0].echo, "an echo marked 25 ticks after she said the word"
    tx2 = Transcriber(ear)
    tr = T.Tract(9)
    got2 = []
    for t in range(30):
        if t == 5:
            tr.x, tr.target = x.copy(), x.copy()
        y = tr.tick(np.full(10, 2) if 5 <= t < 9 else None)
        got2 += tx2.tick(t, y, 0.8, None, False, ("duck",), P(t), {"duck": 1}, vocab=words)
    assert [cw.echo for cw in got2 if cw.channel == "tract"] == [True]
    print(f"11 the transcriber hears a tract utterance through her ear at its turn's end (tick {tw[0].tick}, started "
          f"{tw[0].start}): 'duck', expected, accepted; the same said within 10 ticks of hers is an echo")


# ------------------------------------------------------------------------------------------------------------- the ledger
def test_ledger_standing():
    led = Ledger()
    p_sees = P(0, child_target=None)
    ln = TP.Line("look at the duck.", "show", "plain", "duck", "duck", ("duck",))
    led.said(0, ln, p_sees, word_ends=[("look", 0), ("at", 0), ("the", 0), ("duck", 0)])
    assert led.standing("duck")["heard"] == 0, "a word counted before its sound ended"
    led.voiced(0, p_sees)
    led.said(1, TP.Line("look at the block.", "show", "plain", "block", "block", ("block",)), p_sees,
             word_ends=[("look", 1), ("at", 1), ("the", 1), ("block", 1)])
    led.voiced(1, p_sees)
    assert led.standing("duck")["heard"] == 1 and led.standing("block")["heard"] == 0, "heard with its referent out of view"
    for i in range(10):                                   # base rate: 1 of 10 random moments land on the duck
        led.base_trial(100 + 30 * i, "duck")
        for t in range(100 + 30 * i, 100 + 30 * i + 21):
            led.observe(t, P(t, child_target="duck" if (i == 0 and t >= 101) else None))
    for i in range(10):                                   # asks: 6 of 10 met
        t0 = 1000 + 40 * i
        led.ask(t0, "duck", "gaze", "duck", K.JUDGE_GAZE)
        for t in range(t0, t0 + 22):
            led.observe(t, P(t, child_target="duck" if (i < 6 and t >= t0 + 3) else None))
    assert led.standing("duck")["asks"] == [1] * 6 + [0] * 4 and led.understood("duck"), led.standing("duck")
    led2 = Ledger()
    for i in range(10):
        led2.base_trial(100 + 30 * i, "duck")
        for t in range(100 + 30 * i, 100 + 30 * i + 21):
            led2.observe(t, P(t, child_target="duck" if i < 6 and t >= 101 + 30 * i else None))
    hits = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]                 # 6 of 10, no run of them beating 6 in 10 by chance
    for i in range(10):
        t0 = 1000 + 40 * i
        led2.ask(t0, "duck", "gaze", "duck", K.JUDGE_GAZE)
        for t in range(t0, t0 + 22):
            led2.observe(t, P(t, child_target="duck" if (hits[i] and t >= t0 + 3) else None))
    assert led2.standing("duck")["asks"] == hits and not led2.understood("duck"), "understood at the child's own base rate"
    led3 = Ledger()
    for i in range(10):
        t0 = 1000 + 40 * i
        led3.ask(t0, "duck", "gaze", "duck", K.JUDGE_GAZE)
        for t in range(t0, t0 + 22):
            led3.observe(t, P(t, child_target="duck" if t >= t0 + 3 else None))
    assert not led3.understood("duck"), "understood with no base rate"
    assert abs(binom_tail(6, 10, 0.1) - 1.46902600e-4) < 1e-11 and binom_tail(5, 5, 0.6) > K.UNDERSTOOD_P
    # says: 3 times over 2 days, per channel, never an echo, the referent where she reads it looking or in its hand
    led4 = Ledger()
    for t, ch, tg, echo in ((10, "tract", "duck", False), (20, "tract", None, False), (30, "tract", "duck", True),
                            (24010, "tract", "duck", False), (24050, "token", "duck", False), (24090, "tract", "duck", False)):
        led4.accepted(ChildWord(t, ch, "duck", True, t - 3, t, ("duck",), echo), P(t, child_target=tg))
    assert led4.says("duck", "tract") and not led4.says("duck", "token"), led4.standing("duck")
    assert led4.words["duck"]["says"]["tract"] == [10, 24010, 24090] and led4.words["duck"]["echoes"]["tract"] == 1
    led5 = Ledger()
    for t in (10, 20, 30):
        led5.accepted(ChildWord(t, "tract", "duck", True, t - 3, t), P(t, child_target="duck"))
    assert not led5.says("duck", "tract"), "says reached within one day"
    print("12 the ledger: 'heard' only with the referent in the child's view; understood at 6 of 10 against a base rate of 1 in "
          "10 (p = 0.00015), not against its own base rate of 6 in 10, not with no base rate; 'says' 3 times over 2 days per "
          "channel, the echo and a referent not where she reads it looking not counted, never within one day")


def test_replay_exact():
    tmp = tempfile.mkdtemp()
    try:
        stream = script(700, seed=1)
        toks = {t: LX.WORD_ID["duck"] for t in range(60, 700, 97)}

        def fresh(path):
            return C.Conduct(seed=7, transcriber=Transcriber(None), ledger=Ledger(path))
        a = fresh(os.path.join(tmp, "a.jsonl"))
        said_a, j_a = [], []
        snap = None
        for p in stream:
            if p.tick == 350:
                snap = json.loads(json.dumps(a.state(), default=_np))
            s = a.tick(p.tick, p, token=toks.get(p.tick))
            said_a.append(None if s.line is None else s.line.text)
            j_a += s.judgments
        full = a.ledger.digest
        n_lines = sum(x is not None for x in said_a)
        b = fresh(os.path.join(tmp, "b.jsonl"))
        for p in stream:
            b.tick(p.tick, p, token=toks.get(p.tick))
        assert b.ledger.digest == full, "the same seed and moments gave another ledger"
        assert open(os.path.join(tmp, "a.jsonl")).read() == open(os.path.join(tmp, "b.jsonl")).read()
        c = fresh(os.path.join(tmp, "a.jsonl"))                       # the crash: restored from tick 350, replays the rest
        c.load_state(snap)
        assert c.ledger.replay_left() > 0
        said_c = []
        for p in stream[350:]:
            s = c.tick(p.tick, p, token=toks.get(p.tick))
            said_c.append(None if s.line is None else s.line.text)
        assert said_c == said_a[350:] and c.ledger.digest == full and c.ledger.replay_left() == 0
        d = fresh(os.path.join(tmp, "a.jsonl"))                       # a replay that goes otherwise is caught
        d.load_state(snap)
        d.fast.rng = np.random.Generator(np.random.PCG64(99))
        try:
            for p in stream[350:]:
                d.tick(p.tick, p, token=toks.get(p.tick))
            raise AssertionError("a different replay went unnoticed")
        except LedgerDiverged:
            pass
        e = fresh(None)
        for p in stream:
            e.tick(p.tick, p, token=toks.get(p.tick))
        other = C.Conduct(seed=8, transcriber=Transcriber(None))
        said_o = []
        for p in stream:
            s = other.tick(p.tick, p, token=toks.get(p.tick))
            said_o.append(None if s.line is None else s.line.text)
        assert e.ledger.digest == full and said_o != said_a
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"13 700 scripted ticks ({n_lines} lines, {len(j_a)} judgments): the same seed gives the same lines and the same "
          f"ledger, file and digest; restored at tick 350 it replays the rest exactly, each event checked against the file; a "
          f"replay that chooses otherwise raises LedgerDiverged; another seed chooses other lines")


def _np(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def test_cost():
    ear = tiny_ear()
    words = tuple(SOUNDS)
    tx = Transcriber(ear)
    rng = np.random.default_rng(7)
    tr = T.Tract(7)
    stream = []
    k = 0
    while k < 1500:
        if rng.random() < 1 / 14.5:
            L = int(rng.integers(1, 9))
            a = rng.integers(0, 5, T.N_ART)
            for _ in range(L):
                stream.append(tr.tick(a))
            k += L
        else:
            stream.append(tr.tick(None))
            k += 1
    rest = [i for i, y in enumerate(stream) if not np.any(y)]
    t0 = time.perf_counter()
    for t, y in enumerate(stream):
        tx.tick(t, y, 1.0, None, False, ("ball", "mama", "duck"), P(t), vocab=words)
    wall = (time.perf_counter() - t0) * 1e3
    t1 = time.perf_counter()
    for t in rest[:500]:
        Transcriber(None).tick(t, stream[t], 1.0, None, False, (), P(t))
    rest_ms = (time.perf_counter() - t1) * 1e3 / max(1, min(500, len(rest)))
    assert rest_ms < 0.5, f"a silent tick costs {rest_ms:.3f} ms"
    print(f"14 the transcriber over {len(stream)} babble ticks ({tx.n_utts} turns): {wall / len(stream):.3f} ms a tick with this "
          f"small ear (tools/sim_parent_ear.py measures the full one); a silent tick {rest_ms * 1e3:.0f} us")


def test_ear_exact_across_threads():
    """her ear's hearing, bit for bit, in a process with the linear algebra on one thread (vecLib, OpenBLAS, MKL, OMP)."""
    import subprocess                                                     # noqa: PLC0415
    code = ("import json, sys; sys.path.insert(0, %r); import body.tests.test_sim_lang as M; ear = M.tiny_ear(); "
            "w = tuple(M.SOUNDS); out = [ear.hear(M.say(x, 0.3 + 0.05 * i, seed=50 + i), (x, w[(i + 2) %% 5]), w).as_dict() "
            "for i, x in enumerate(w * 2)]; print(json.dumps([ear.digest, out], default=str))") % \
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    runs = []
    for env1 in (False, True):
        env = dict(os.environ)
        if env1:
            env.update(VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
        assert r.returncode == 0, r.stderr[-1500:]
        runs.append(r.stdout.strip().splitlines()[-1])
    assert runs[0] == runs[1], "her ear heard differently with one thread"
    got = json.loads(runs[0])[1]
    print(f"16 her ear in two processes, threaded and on one thread: the same digest and {len(got)} hearings bit for bit "
          f"({sum(g['word'] is not None for g in got)} accepted)")


def scenario(n, seed, stage):
    """a scripted stretch of life for the exact-replay test: the child's gaze among toys, her face and nothing, a toy held,
    falls, the stand-in babbler's tract (with talk-overs), word tokens and letters, and asks, calls, a show and a new word
    requested. -> dict(percepts, tract, tokens, requests, stage)."""
    rng = np.random.default_rng(seed)
    toys = TOYS + seen(("rattle", "rattle", "purple", "mat", True))
    tr = T.Tract(seed)
    tract, k = [], 0
    while len(tract) < n:
        if rng.random() < 1 / 14.5:
            a = rng.integers(0, 5, T.N_ART)
            tract += [tr.tick(a) for _ in range(int(rng.integers(1, 9)))]
        else:
            tract.append(tr.tick(None))
    tract = tract[:n]
    percepts, tg = [], None
    for t in range(n):
        if t % 29 == 0:
            tg = [None, "duck", "ball", "mama", "cup", None, "block_red"][int(rng.integers(7))]
        holds = ("cup",) if (t // 120) % 3 == 1 else ()
        ev = (("fell", "ball"),) if t % 173 == 60 else ()
        ev += (("wave", "left"),) if t % 89 == 20 else ()                    # its hand waves: she may copy it (A52)
        reach = ("duck",) if t % 97 in (10, 11, 12, 13) else ()               # a hand closing on the duck, as she reads it
        percepts.append(P(t, child_target=tg, child_holds=holds, events=ev, seen=toys, child_reaches=reach))
    tokens = {}
    for t in range(40, n, 83):
        tokens[t] = int(rng.choice([LX.WORD_ID["duck"], LX.WORD_ID["ball"], LX.WORD_ID["mama"]]))
    for t0 in range(70, n, 131):
        for i, c in enumerate(["bal", "duk", "z", "qp"][(t0 // 131) % 4]):
            tokens[t0 + i] = LX.LETTER_ID[c]
        tokens[t0 + len(["bal", "duk", "z", "qp"][(t0 // 131) % 4])] = LX.ID[LX.SPACE]
    requests = {100: [("ask_where", dict(o="ball"))], 230: [("call", {})], 360: [("ask_what", dict(o="cup"))],
                470: [("new_word", dict(word="rattle", o="rattle"))], 600: [("ask_give", dict(o="cup"))],
                720: [("show", dict(o="duck"))], 800: [("call", {})], 850: [("show", dict(o="rattle"))]}
    return dict(percepts=percepts, tract=tract, tokens=tokens, requests=requests, stage=stage)


def run_scenario(con, sc, t_from, t_to):
    log = []
    for t in range(t_from, t_to):
        for intent, kw in sc["requests"].get(t, ()):
            con.request(intent, **kw)
        s = con.tick(t, sc["percepts"][t], tract=sc["tract"][t], token=sc["tokens"].get(t))
        log.append(json.dumps([t, None if s.line is None else s.line.text, s.judgments, s.cut,
                               [(cw.channel, cw.word, cw.exact, cw.echo) for cw in s.heard],
                               [(a.kind, a.target) for a in s.acts], [(a.kind, a.target) for a in s.copy], s.frown]))
    return log


def _fresh_conduct(stage, seed=11):
    return C.Conduct(seed=seed, transcriber=Transcriber(tiny_ear(sounds=REPLAY_SOUNDS)), stage=stage, vocab=VOCAB)


def test_replay_across_processes():
    """the whole speech side (the fast layer, the transcriber through a small ear, the tract's babble with its talk-overs,
    tokens and letters, asks, calls, a new word) run 900 ticks in stages 1 and 2; snapshots taken mid-line and mid-turn are
    restored in a fresh process with the linear algebra on one thread and replay the rest exactly (every tick's lines,
    judgments, readings and acts, and the ledger's digest)."""
    import subprocess                                                     # noqa: PLC0415
    n = 900
    tmp = tempfile.mkdtemp()
    try:
        jobs, summary = [], []
        for stage in (1, 2):
            sc = scenario(n, 5 + stage, stage)
            con = _fresh_conduct(stage)
            snaps, log = [], []
            for t in range(n):
                cur = con.fast.current is not None and con.fast.speaking(t)
                utt = con.transcriber.utt is not None
                if (cur and len([x for x in snaps if x[1] == "line"]) < 2 and t > 90 + 200 * len(snaps)) or \
                        (utt and not [x for x in snaps if x[1] == "turn"] and t > 300):
                    snaps.append((t, "line" if cur else "turn", json.loads(json.dumps(con.state(), default=_np))))
                log += run_scenario(con, sc, t, t + 1)
            digest = con.ledger.digest
            summary.append((stage, len([x for x in log if json.loads(x)[1]]), sum(len(json.loads(x)[2]) for x in log),
                            sum(json.loads(x)[3] for x in log), len(snaps)))
            assert len(snaps) >= 2, snaps
            for t0, kind, st in snaps:
                path = os.path.join(tmp, f"s{stage}_{t0}.json")
                with open(path, "w") as fh:
                    json.dump(dict(stage=stage, t0=t0, state=st), fh)
                jobs.append((path, log[t0:], digest))
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        code = ("import json, sys; sys.path.insert(0, %r); import body.tests.test_sim_lang as M\n"
                "for path in sys.argv[1:]:\n"
                "    d = json.load(open(path)); sc = M.scenario(%d, 5 + d['stage'], d['stage'])\n"
                "    con = M._fresh_conduct(d['stage']); con.load_state(d['state'])\n"
                "    log = M.run_scenario(con, sc, d['t0'], %d)\n"
                "    print(json.dumps([path, con.ledger.digest, log]))\n") % (root, n, n)
        env = dict(os.environ, VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OMP_NUM_THREADS="1")
        r = subprocess.run([sys.executable, "-c", code] + [j[0] for j in jobs], capture_output=True, text=True, env=env,
                           timeout=900)
        assert r.returncode == 0, r.stderr[-2000:]
        got = {x[0]: x for x in (json.loads(ln) for ln in r.stdout.strip().splitlines())}
        for path, want_log, want_digest in jobs:
            _, dg, lg = got[path]
            assert lg == want_log, f"{os.path.basename(path)}: the replay differs at " \
                                   f"{next(i for i, (a, b) in enumerate(zip(lg, want_log)) if a != b)}"
            assert dg == want_digest, f"{os.path.basename(path)}: the ledger's digest differs"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"24 {n} ticks of the whole speech side, stage 1 and 2 (lines, judgments, cuts, snapshots): "
          f"{'; '.join(f'stage {a}: {b} lines, {c} judgments, {d} cuts, {e} snapshots' for a, b, c, d, e in summary)}; each "
          f"snapshot (mid-line and mid-turn) restored in a fresh one-thread process replays the rest exactly, ledger digest "
          f"and every tick")


# ------------------------------------------------------------ the P3 verifier's second findings (d053dff), each tested both ways
def _fill_all(intent):
    """every line an intent's frames make, filled with each toy she sees and the child's name (no line check)."""
    out = set()
    for fr in TP.FRAMES[intent]:
        for o in TOYS:
            got = TP.fill(fr, o=o, w="duck", b="foot")
            if got is not None:
                out.add(got[0])
    return out


def test_claude_never_asks():
    """finding 1: Claude's lines could still ask, and nothing judged the ask: after the child's name ("pip where is the ball?",
    a call with none of the call's rules), ending on a full stop ("where is the ball."), after a greeting; test 22 tried asks only
    at a sentence's start. And finding 8: claims the check could not hold passed ("the ball is here.", "you roll.", "it is a
    mat.", "no.")."""
    p = P(0, child_target="ball", child_holds=("cup",))
    found = ["pip where is the ball?", "pip give me the cup.", "pip look at the ball.", "pip look at mama.",
             "where is the ball.", "what is this.", "hi pip where is the ball?"]
    for text in found:
        ok, why = TP.check(text, VOCAB, None, p, (), source="claude")
        assert not ok and "ask" in why, f"Claude's {text!r} passed the check ({why!r})"
        assert not C.FastLayer(1).add_steer(text, "any", 0)[0], text
    # every line of every intent that asks, calls or questions (the expectant pause's), filled with each toy she sees, at every
    # place in a line and with every ending: whatever the fast layer's own check would let it say, Claude may not
    asks, n_said_fast = set(), 0
    for intent, it in C.INTENTS.items():
        if not it.expect and it.ask is None:
            continue
        for base in _fill_all(intent):
            body = base[:-1]
            for end in ".?!":
                for pre in ("", "pip ", "hi pip ", "oh! ", "the duck. ", "mama is here. ", "look. "):
                    asks.add(pre + body + end)
                    asks.add(pre + body.replace(". ", " ").replace("? ", " ") + end)
                asks.add(base + " the duck" + end)
    refused_as_ask = 0
    for text in sorted(asks):
        fast_ok = TP.check(text, VOCAB, None, p, ())[0]
        ok, why = TP.check(text, VOCAB, None, p, (), source="claude")
        assert not ok, f"Claude may say the ask {text!r}"
        n_said_fast += fast_ok
        refused_as_ask += fast_ok and ("ask" in why or "judgment" in why)
    assert n_said_fast > 500 and refused_as_ask == n_said_fast, (refused_as_ask, n_said_fast)
    # said through a conduct: each placed in Claude's rows directly (past add_steer's check), never said
    con = C.Conduct(seed=1)
    for text in found + ["more?", "sit.", "no.", "peekaboo!", "bye bye.", "night night."]:
        con.fast.steer.append(dict(text=text, situation="any", uses=0, tick_from=0, ttl=2000))
    said, _ = run(con, [P(t, child_target=None) for t in range(400)])
    assert not [ln for _, ln in said if ln.source == "claude"], [ln.text for _, ln in said if ln.source == "claude"]
    # finding 8: what a Claude line claims is held true, or the line is refused
    fell = [(0, "fell", "ball")]
    blocks = TOYS[:3] + seen(("block", "block", "blue", "sofa", False))
    for text, ev, pp in (("the ball is here.", (), p), ("it is a mat.", (), p), ("no.", (), p), ("you roll.", (), p),
                         ("sit.", (), p), ("hi.", (), p), ("bye bye.", (), p), ("peekaboo!", (), p), ("more?", (), p),
                         ("look here.", (), p), ("your duck.", (), p), ("it is a duck.", (), p), ("the duck fell.", fell, p),
                         ("you see the block.", (), P(0, seen=blocks)), ("mama is here.", (), P(0, present=False)),
                         ("the cup is red.", (), p), ("the ball is down.", (), p), ("it is yellow.", (), p)):
        ok, why = TP.check(text, VOCAB + ("fell", "red", "yellow", "green"), None, pp, (), source="claude",
                           recent_events=ev)
        assert not ok and "her word" not in why, f"Claude's {text!r} passed ({why!r})"
    good = [("a duck.", ()), ("the duck!", ()), ("you see the duck?", ()), ("look. the ball.", ()), ("it is a ball.", ()),
            ("your cup.", ()), ("it is your cup.", ()), ("the ball is on the mat.", ()), ("mama is here.", ()),
            ("oh! you roll.", [(0, "rolled", None)]), ("the ball fell.", fell), ("uh oh. the ball is down.", fell),
            ("see? the duck.", ()), ("the cup is green.", ()), ("see the duck?", ()), ("oh.", ()), ("your foot.", ())]
    for text, ev in good:
        ok, why = TP.check(text, VOCAB + ("fell", "red", "yellow", "green"), None, p, (), source="claude",
                           recent_events=ev)
        assert ok, (text, why)
    print(f"25 Claude's lines never ask: the verifier's 7 ({', '.join(repr(x) for x in found[:2])}, ...) refused as asks at "
          f"add_steer and at the check; {len(asks)} variants of the asking intents' lines (every frame, toy and ending; a name, "
          f"a greeting or a sentence before; the sentences run together; a sentence after), {n_said_fast} of them lines the "
          f"fast layer itself may say: every one refused for Claude ({refused_as_ask} of those {n_said_fast} as an ask or a "
          f"judgment); placed in Claude's rows past the check, none said in 400 ticks; 18 claims she cannot hold refused, "
          f"{len(good)} true lines passed")


def test_name_ask_answered_after_its_question():
    """finding 2: "what is it?" about the cup ends at tick 8; the child's "cup" token made at tick 1, held until her line ended
    and read at tick 9, was scored a met ask (worth 2): the answer was timed by when she read it, not when it was made."""
    held = tuple(s_ for s_ in TOYS if s_.id != "cup") + seen(("cup", "cup", "green", "hand", True))

    def go(*tok_at):
        """(since P3's sixth round a name ask is of what the child attends, A51: here the cup in its hand, so its word there is a
        right name whatever the ask; the ask is what the timing decides)"""
        con = C.Conduct(seed=6, transcriber=Transcriber(None))
        for x in ("duck", "ball", "cup", "block", "block_red"):
            con.fast.last_set[x], con.fast.last_named[x] = 0, -1000
        con.request("ask_what", o="cup")
        judg, heard, pend = [], [], None
        for t in range(40):
            s = con.tick(t, P(t, child_target=None, child_holds=("cup",), seen=held),
                         token=LX.WORD_ID["cup"] if t in tok_at else None)
            if pend is None and con.pending is not None:
                pend = dict(con.pending)
            judg += s.judgments
            heard += s.heard
        return con, pend, judg, heard
    con, pend, judg, heard = go(1)
    cw = [c for c in heard if c.word == "cup"]
    assert pend is not None and pend["kind"] == "name" and pend["open"] > 1, pend
    assert cw and cw[0].start == 1 and cw[0].tick >= pend["open"], cw
    assert not [j for j in judg if j[1] == "met_ask"], f"a word made before the question was heard met it: {judg}"
    assert [j[1] for j in judg] == ["right_name"] and con.pending is None and con.ledger.trials == [], (judg, con.pending)
    con, _p, judg, heard = go(1, cw[0].tick + 12)                  # said again after her echo of it: the ask was void
    assert not [j for j in judg if j[1] == "met_ask"], f"the ask, answered before it was heard and echoed by her, was met " \
                                                        f"after: {judg}"
    con, pend2, judg, heard = go(pend["open"] + 2)
    assert [j[:2] for j in judg] == [(2, "met_ask")], judg
    con, pend3, judg, heard = go(pend["open"])
    assert [j[:2] for j in judg] == [(2, "met_ask")], f"a word begun on the tick the question was heard: {judg}"
    for start, want in ((pend["open"] - 2, []), (pend["open"] + 1, [(2, "met_ask")])):   # the tract's turn likewise
        con = C.Conduct(seed=6, stage=2)
        con._say(TP.Line("what is it?", "ask_what", "question", None, None, ("cup",)), 0, P(0), C.Say())
        out = C.Say()
        con._owe_reply(12, ChildWord(12, "tract", "cup", True, start, 12, ("cup",)), P(12, child_target=None), out)
        assert [j[:2] for j in out.judgments] == want and con.pending is None and con.ledger.trials == [], \
            (start, out.judgments, con.pending)
    print(f"26 'what is it?' of the cup in its hand, heard at tick {pend['open']}: the child's 'cup' made at tick 1, held through "
          f"her line and read at tick {cw[0].tick}, is no answer (a right name of what it holds, never a met ask; the ask void, "
          f"as a gaze ask with its X already where she reads it looking, so its word said again after her echo meets no ask); "
          f"made at tick {pend['open']} or later: met, a smile of 2; the tract's turn the same by the tick its sound began")


def test_calls_out_of_sight():
    """finding 3: three calls from the hall, where the child cannot see her, left the name's record at [0, 0, 0] (a call no look
    can answer, counted a missed ask of 'pip'). And finding 7: the call's window ran from its name's end to 20 ticks after the
    line's end (21-30 ticks by frame), not a gaze ask's 20."""
    away = dict(present=False, seen_by_child=False, seen=(), child_target=None)
    con = C.Conduct(seed=9, transcriber=Transcriber(None))
    said = []
    for t in range(3 * 300):
        if t % 300 == 0:
            con.request("hall_call")
        s = con.tick(t, P(t, **away))
        if s.line is not None:
            said.append(s.line)
    assert [ln.intent for ln in said] == ["hall_call"] * 3, [ln.text for ln in said]
    assert con.ledger.standing(LX.NAME)["asks"] == [], f"calls from the hall counted: {con.ledger.standing(LX.NAME)}"
    # a call while she is where its eyes cannot reach her (behind its head): void, counted neither way
    con = C.Conduct(seed=9, transcriber=Transcriber(None))
    con.request("call")
    for t in range(60):
        con.tick(t, P(t, seen_by_child=False, child_target=None))
    assert con.ledger.standing(LX.NAME)["asks"] == [] and con.pending is None, con.ledger.standing(LX.NAME)
    con = C.Conduct(seed=9, transcriber=Transcriber(None))                   # the control: seen, unanswered: missed
    con.request("call")
    for t in range(60):
        con.tick(t, P(t, child_target=None))
    assert con.ledger.standing(LX.NAME)["asks"] == [0], con.ledger.standing(LX.NAME)
    # every call frame: judged 20 ticks from the end of the name, as a gaze ask
    wins = []
    for fr in TP.FRAMES["call"]:
        con = C.Conduct(seed=9)
        text, focus, _r = TP.fill(fr)
        con._say(TP.Line(text, "call", "calling", focus, focus, ()), 0, P(0), C.Say())
        name_end = 3 - 1                                              # no voice: 3 ticks a word, the name first
        wins.append((text, con.pending["open"], con.pending["until"] - con.pending["open"]))
        assert con.pending["open"] == name_end and con.pending["until"] - con.pending["open"] == K.JUDGE_GAZE, wins[-1]
    print(f"27 three calls from the hall: said, not judged (the name's record empty); a call where the child cannot see her: "
          f"void; seen and unanswered: missed; each call frame judged {K.JUDGE_GAZE} ticks from the end of its name "
          f"({', '.join(f'{a!r} {c}' for a, _b, c in wins)})")


def test_introduce_only_what_she_can_show():
    """finding 4: nothing in the conduct asked showable(): 'toes' was introduced with a touch on the G1's toes it does not have
    ('here is your toes!'), and 'big', 'hot', 'want', 'sing', 'happy' and 'clap' likewise ('it is hot.', 'it is happy.')."""
    room = TOYS + seen(("rattle", "rattle", "purple", "mat", True), ("ring", "ring", "pink", "mat", True),
                       ("stacker", "stacker", "white", "mat", True), ("bear", "bear", "brown", "mat", True),
                       ("drum", "drum", "cyan", "mat", True), ("car", "car", "orange", "mat", True))
    fix = frozenset({"mat", "sofa", "window", "table", "shelf", "door", "light", "floor"})
    events = (("fell", "ball"), ("rolled", None), ("sat", None), ("got", "cup"), ("gave", "duck"))

    def intro(word, con=None, t0=0):
        con = con or C.Conduct(seed=5)
        con.request("new_word", word=word)
        got = []
        for t in range(t0, t0 + 60):
            s = con.tick(t, P(t, seen=room, fixtures=fix, events=events if (t - t0) % 10 == 0 else ()))
            if s.line is not None and s.line.intent == "new_word":
                got.append((s.line.text, s.acts))
        return con, got
    waiting = ("toes", "big", "hot", "want", "sing", "happy", "clap")
    for w in waiting:
        con, got = intro(w)
        assert not got, f"{w!r} introduced though the world cannot show it: {got}"
    words = [w for w, c in TP.GROWTH if c != "frame"]
    introduced, refused = set(), {}
    for w in words:
        con, got = intro(w)
        if got:
            assert len(got) == 3 and all(TP.words(x[0])[-1] == w for x in got), (w, got)
            introduced.add(w)
        else:
            refused[w] = [r[2] for r in con.fast.refused if r[1] == "new_word"][-1:]
    p0 = P(0, seen=room, fixtures=fix, events=events)
    ev0 = [(0, k, o) for k, o in events]
    want = {w for w in words if TP.showable(w, con.world)[0] and TP.show_now(w, p0, ev0)[0] and
            any(len(TP.intro_on_peak(w, o)) >= 3 for o in TP.show_now(w, p0, ev0)[0])}          # 3 lines on its peak (A34)
    assert introduced == want, (sorted(introduced - want), sorted(want - introduced))
    assert all(r and "waits" in r[0] for w, r in refused.items() if w in waiting), {w: refused[w] for w in waiting}

    class Clapper(C.StubMotion):                          # a verb waits for her motion: one that can clap (W2's to declare)
        DOES = C.StubMotion.DOES + ("clap",)
    con, got = intro("clap", C.Conduct(seed=5, motion=Clapper()))
    assert len(got) == 3 and C.Act("do", "clap") in got[0][1], got
    # moments: a toy she does not see, a fixture out of her view, a fall she did not see
    for w, pp in (("rattle", dict(seen=TOYS)), ("table", dict(fixtures=frozenset({"mat"}))), ("drop", {})):
        con = C.Conduct(seed=5)
        con.request("new_word", word=w)
        for t in range(12):
            s = con.tick(t, P(t, **pp))
            assert s.line is None or s.line.intent != "new_word", (w, s.line)
        assert "cannot be shown now" in con.fast.refused[-1][2], (w, con.fast.refused[-1])
    # the colour twins (B2): 'red' waits until two red toys are in the world, then is shown on one that is no held-out pair
    twins = room + seen(("block_red2", "block", "red", "mat", True))
    con = C.Conduct(seed=5)
    con.request("new_word", word="red")
    con.tick(0, P(0, seen=twins))
    assert "fewer than two red" in con.fast.refused[-1][2], con.fast.refused[-1]
    con.set_world(dict(TP.ROOM_AT_BIRTH, objects=dict(TP.ROOM_AT_BIRTH["objects"], block=["blue", "red"])))
    con.request("new_word", word="red")
    got = []
    for t in range(1, 40):
        s = con.tick(t, P(t, seen=twins))
        if s.line is not None and s.line.intent == "new_word":
            got.append(s.line)
    assert len(got) == 3 and all(ln.refs == ("ball",) for ln in got), got          # A55: "red" on the red ball, never a block
    # the pace (A14): a second new word within a minute is refused and logged; after the minute it is said
    con, got = intro("rattle")
    con.request("new_word", word="ring")
    con.tick(100, P(100, seen=room))
    assert "at most 1 new word a minute" in con.fast.refused[-1][2], con.fast.refused[-1]
    con, got2 = intro("ring", con, t0=K.NEW_EVERY)
    assert len(got2) == 3, got2
    print(f"28 a growth word enters only when the world can show it (showable, against the room's inventory and her motion's "
          f"acts) and she can show it now (show_now): of the {len(words)} with frames, {len(introduced)} introduced here "
          f"(exactly those), {len(refused)} wait, each logged ({', '.join(waiting)} among them: no 'here is your toes!'); a "
          f"motion that can clap introduces 'clap' with the act; a toy unseen, a fixture out of view, a fall not seen: not "
          f"now; 'red' only once the world has two red toys, and on the red ball, never a red block (A55); at most 1 new word a "
          f"minute")


def test_give_before_its_word():
    """finding 5: the child giving the cup during "give me the cup.", before "cup" was heard, counted a missed ask of 'cup' (a
    look before its word voids a gaze ask; a give before its word was no answer, and no failure either)."""
    def go(give_at):
        con = C.Conduct(seed=4, transcriber=Transcriber(None))
        for x in ("duck", "ball", "cup", "block", "block_red"):
            con.fast.last_set[x] = con.fast.last_named[x] = -1000
        con.request("ask_give", o="cup")
        judg, opened = [], None
        for t in range(60):
            s = con.tick(t, P(t, child_target=None, events=(("gave", "cup"),) if t == give_at else ()))
            if opened is None and con.pending is not None:
                opened = con.pending["open"]
            judg += s.judgments
        return con, opened, judg
    con, opened, judg = go(1)
    assert opened > 1 and judg == [], (opened, judg)
    assert con.ledger.standing("cup")["asks"] == [], f"a give before its word counted: {con.ledger.standing('cup')}"
    con, opened, judg = go(opened + 3)
    assert [j[1] for j in judg] == ["met_ask"] and con.ledger.standing("cup")["asks"] == [1], judg
    print(f"29 'give me the cup.' (heard at tick {opened}): the cup given at tick 1: void, counted neither way; given after "
          f"it was heard: met, counted")


def test_ear_deliberate_writes():
    """finding 6: her ear could still be written on purpose (through ear.__dict__ or object.__setattr__): her longest template,
    her words' order and the module's DCT changed what she heard, and verify() still passed; it checked the digest kept inside
    the ear, not an outside pin."""
    ear = tiny_ear()
    words = tuple(SOUNDS)
    x = say("ball", 0.35, seed=21)
    probe = ear.hear(x, ("ball", "see"), words).as_dict()
    pin = tuple(ear.pin) if hasattr(ear, "pin") else None

    def caught(name, write, undo):
        """a deliberate write -> verify() must raise; undone, it holds again."""
        write()
        try:
            try:
                ear.verify()
            except ValueError:
                return True
            raise AssertionError(f"{name}: a deliberate write went uncaught by verify()")
        finally:
            undo()
    if hasattr(ear, "__dict__"):                                     # the verifier's own write
        d, ml0 = ear.__dict__, ear.max_len
        caught("ear.__dict__['max_len']", lambda: d.__setitem__("max_len", 1), lambda: d.__setitem__("max_len", ml0))
    ml, order, dct, slope, coch = ear.max_len, ear.order, PE.DCT, K.EAR_SLOPE, PE.E.cochlea
    swapped = dict(order)
    swapped["ball"], swapped["see"] = swapped["see"], swapped["ball"]
    writes = [("max_len", lambda: object.__setattr__(ear, "max_len", 1), lambda: object.__setattr__(ear, "max_len", ml)),
              ("order", lambda: object.__setattr__(ear, "order", swapped), lambda: object.__setattr__(ear, "order", order)),
              ("the module's DCT", lambda: setattr(PE, "DCT", dct * 1.001), lambda: setattr(PE, "DCT", dct)),
              ("the slope", lambda: setattr(K, "EAR_SLOPE", 3.0), lambda: setattr(K, "EAR_SLOPE", slope)),
              ("the cochlea", lambda: setattr(PE.E, "cochlea", lambda a: coch(a) * 1.01), lambda: setattr(PE.E, "cochlea", coch))]
    heard_other = 0
    for name, w, u in writes:
        w()
        try:
            heard_other += ear.hear(x, ("see", "ball"), words).as_dict() != probe
        finally:
            u()
        assert caught(name, w, u)
    assert ear.verify() and ear.hear(x, ("ball", "see"), words).as_dict() == probe
    assert not hasattr(ear, "__dict__") and not hasattr(ear._stacks["ball"], "__dict__"), "the ear has a __dict__"
    # the digest kept inside the ear can itself be written: the life's pin (taken at load, saved with the world) catches it
    other = tiny_ear(margins={k: -2.5 for k in range(1, 9)})
    margins0 = ear.margins
    object.__setattr__(ear, "margins", other.margins)
    object.__setattr__(ear, "digest", other.digest)                  # the margins and the digest written together
    try:
        assert ear.verify(), "the ear's own digest agrees with its arrays"
        try:
            ear.verify(pin)
            raise AssertionError("the margins and the digest written together went uncaught by the pin")
        except ValueError:
            pass
        con = C.Conduct(seed=5, transcriber=Transcriber(other), vocab=tuple(SOUNDS))
        con.transcriber.ear, con.transcriber.pin = ear, pin            # the life loaded `ear` (its pin); `ear` was written since
        try:
            con.night()
            raise AssertionError("a night passed with her ear written")
        except ValueError:
            pass
    finally:
        object.__setattr__(ear, "margins", margins0)
        object.__setattr__(ear, "digest", pin[0])
    assert ear.verify(pin)
    tx = Transcriber(ear)                                            # a world saved with one ear is not restored with another
    st = json.loads(json.dumps(tx.state(), default=_np))
    try:
        Transcriber(other).load_state(st)
        raise AssertionError("a world saved with one ear restored with another")
    except ValueError:
        pass
    Transcriber(tiny_ear()).load_state(st)
    print(f"30 deliberate writes to her ear ({', '.join(n for n, _w, _u in writes)}): {heard_other} of {len(writes)} changed "
          f"what she heard, and verify() caught every one; no __dict__ on the ear or its stacks; its margins and digest "
          f"written together agree with each other but not with the life's pin (taken at load, saved with the world): "
          f"verify(pin) and the night refuse it; a world saved with one ear is not restored with another")


# ------------------------------------------------------------- the P3 verifier's third findings (7447f73), each tested both ways
def _perfect(**kw):
    """a conduct with her imperfection off (A52; test 37 holds it), so a test isolates another rule. (Before A52 was built, as on
    7447f73, she was always so: the new tests then fail at their own assertions, not at this call.)"""
    try:
        return C.Conduct(imperfect=False, **kw)
    except TypeError:
        return C.Conduct(**kw)


def _no_sets(con, ids=("duck", "ball", "cup", "block", "block_red", "ball_blue")):
    """named at tick 0: no follow-in set on these in the way for 120 ticks."""
    for x in ids:
        con.fast.last_set[x] = con.fast.last_named[x] = 0


def test_gaze_leak_closed():
    """finding 1 (A51): during an ask her replies still looked at the answer. The verifier's probe: "where is the ball?" (or "the
    ball? where is the ball?"), the child's "ball" token made during it while its head is on the duck; her echo "ball! the
    ball!" came with look -> ball while the ask was pending, and a child that follows her gaze met the ask (worth 2, counted).
    The same with "give mama the cup.", whose open hand also targeted the toy."""
    twins = TOYS + seen(("ball_blue", "ball", "blue", "mat", True), ("bottle", "bottle", "", "mat", True))

    def probe(intent, o, word, tok_at=(6,), n=90, extra=None):
        """a child that follows her eyes and hands: from the tick after she looks, points or shows at a thing, it looks there."""
        con = _perfect(seed=4, transcriber=Transcriber(None))
        _no_sets(con)
        follow, rows, judg = None, [], []
        for t in range(n):
            if t == 5:
                con.request(intent, o=o)
            ev = extra(t, con) if extra else ()
            s = con.tick(t, P(t, child_target=follow or "duck", seen=twins, events=ev),
                         token=LX.WORD_ID[word] if t in tok_at else None)
            if s.line is not None:
                pend = con.pending is not None and con.pending["kind"] in ("gaze", "act", "call")
                rows.append((t, s.line, s.acts, pend or s.line.intent.startswith("ask")))
            for a in s.acts:
                if a.kind in ("look", "point", "show") and a.target not in ("child_eyes", "child", None):
                    follow = a.target
            judg += s.judgments
        return con, rows, judg
    con, rows, judg = probe("ask_where", "ball", "ball")
    during = [(t, ln.text, acts) for t, ln, acts, pend in rows if pend]
    looked = [(t, x, a) for t, x, acts in during for a in acts
              if a.kind in ("look", "point", "show") and a.target != "child_eyes"]
    assert not looked, f"while the ask was pending her acts cued its answer: {looked}"
    echo = [(t, ln.text) for t, ln, _a, pend in rows if pend and ln.intent in ("echo", "echo_word")]
    assert echo and all(C.Act("look", "child_eyes") in acts for t, x, acts in during), (echo, during)
    assert not [j for j in judg if j[1] == "met_ask"] and con.ledger.standing("ball")["asks"] == [0], \
        f"a child that follows her gaze met the ask: {judg}, {con.ledger.standing('ball')}"
    # both ways: the same echo once the ask is judged looks at the ball on its naming word (the joint-attention cue, 4.3)
    con, rows, judg = probe("ask_where", "ball", "ball", tok_at=(6, 60))
    after = [(t, ln.text, acts) for t, ln, acts, pend in rows if not pend and t > 60 and ln.intent in ("echo", "echo_word")]
    assert after and C.Act("look", "ball", "focus") in after[0][2], after
    # the give: her open hand held out to the child, never toward the cup, and her echo of "cup" on the child's eyes
    con, rows, judg = probe("ask_give", "cup", "cup")
    ask = [(ln.text, acts) for t, ln, acts, pend in rows if ln.intent == "ask_give"]
    assert ask and C.Act("open_hand", "child") in ask[0][1] and not [a for a in ask[0][1] if a.target == "cup"], ask
    during = [(t, ln.text, acts) for t, ln, acts, pend in rows if pend]
    assert [x for _t, x, _a in during if "cup" in x and not x.startswith("give")] and \
        not [a for _t, _x, acts in during for a in acts if a.target == "cup"], during
    assert not [j for j in judg if j[1] == "met_ask"], judg
    # the meal's first line waits while the ask is judged (its bottle in her hand would cue), and is said once it is judged
    con, rows, judg = probe("ask_where", "ball", "ball", tok_at=(),
                            extra=lambda t, c: (("charge_low", None),) if t >= 12 else ())
    feeds = [(t, pend) for t, ln, acts, pend in rows if ln.intent == "feed"]
    assert feeds and not [f for f in feeds if f[1]], feeds
    con = _perfect(seed=4, transcriber=Transcriber(None))
    _no_sets(con)
    con.request("ask_where", o="ball")
    con.tick(0, P(0, child_target="duck", seen=twins))
    assert con.eyes_on_child and con.pending["kind"] == "gaze"                 # L1 (W2) keeps her eyes on the child
    # blind(): looks to the child's eyes; no point, show or hand-over; the open hand to the child; acts on the child kept
    got = C.blind((C.Act("look", "ball", "focus"), C.Act("point", "ball"), C.Act("show", "cup"), C.Act("open_hand", "cup"),
                   C.Act("hand_over", "duck"), C.Act("lean_in", "child_periphery"), C.Act("look", "child_eyes")))
    assert got == (C.Act("look", "child_eyes"), C.Act("open_hand", "child"), C.Act("lean_in", "child_periphery")), got
    print(f"31 the gaze leak closed (A51), the verifier's probe: during 'where is the ball?' the child's 'ball' is echoed "
          f"({echo[0][1]!r} at tick {echo[0][0]}) with her eyes on the child's eyes, so a child that follows her gaze never "
          f"meets it (the ask missed, no smile); the same echo after the ask is judged looks at the ball on its naming word; "
          f"'give ... the cup.' holds her open hand out to the child, never toward the cup, and her echo of 'cup' stays on "
          f"its eyes; the meal's line waits for the judgment; blind() keeps only looks at its eyes, the open hand to it and "
          f"acts on it")


def test_new_word_on_its_peak():
    """finding 2 (A34): she introduced new words in frames that put an earlier word on the line's pitch peak (17 of the 210
    introduction lines measured by the verifier with the tool's own f0_word). The line check now refuses a new word's line
    unless it was measured on its peak (body/sim/lang/peak_lines.json, tools/sim_voice_check.py --peak)."""
    found = [("look. eyes.", "eyes"), ("see? eyes!", "eyes"), ("look. open.", "open"), ("it is red.", "red"),
             ("the block is red!", "red"), ("it is yellow.", "yellow"), ("the block is yellow!", "yellow"),
             ("it is little.", "little"), ("it is all.", "all"), ("it is out.", "out"), ("it is off.", "off"),
             ("it is under.", "under"), ("out. out.", "out"), ("off. off.", "off"), ("look at the eyes.", "eyes")]
    for text, w in found:
        vocab = VOCAB + tuple(g for g in TP.GROWTH_WORDS if g != w)
        ok, why = TP.check(text, vocab, w, held=())
        assert not ok and "pitch peak" in why, f"{text!r} (new word {w!r}) passed the check: {why!r}"
    for text, w in (("a rattle.", "rattle"), ("the rattle!", "rattle"), ("you see the rattle?", "rattle"), ("see? red.", "red"),
                    ("red!", "red")):
        assert TP.check(text, VOCAB, w, held=())[0], (text, TP.check(text, VOCAB, w, held=()))
    assert not TP.check("look. the rattle!", VOCAB, "rattle", held=())[0], "an unmeasured new-word line passed"
    assert TP.check("look. eyes.", VOCAB + ("eyes",), None, held=())[0], "the peak rule held a word no longer new"
    lines = TP.new_word_lines()
    assert set(TP.PEAK) == {t for t, _w in lines}, "the table and the lines she can say with a new word differ"
    assert all(TP.PEAK[t][0] == w == TP.words(t)[-1] for t, w in lines)
    off = [t for t in TP.PEAK if not TP.on_peak(t)[0]]
    assert TP.PEAK_META["lines"] == len(lines) and TP.PEAK_META["off"] == len(off), TP.PEAK_META
    # every introduction said is on its peak: each growth word the room can show, introduced as in test 28
    room = TOYS + seen(("rattle", "rattle", "purple", "mat", True), ("ring", "ring", "pink", "mat", True),
                       ("stacker", "stacker", "white", "mat", True), ("bear", "bear", "brown", "mat", True),
                       ("drum", "drum", "cyan", "mat", True), ("car", "car", "orange", "mat", True),
                       ("ball_blue", "ball", "blue", "mat", True))
    fix = frozenset({"mat", "sofa", "window", "table", "shelf", "door", "light", "floor"})
    events = (("fell", "ball"), ("rolled", None), ("sat", None), ("got", "cup"), ("gave", "duck"))
    world = dict(TP.ROOM_AT_BIRTH, objects=dict(TP.ROOM_AT_BIRTH["objects"], ball=["red", "blue"]))
    said, n_words = [], 0
    for w in [w for w, c in TP.GROWTH if c != "frame"]:
        con = C.Conduct(seed=5, world=world)
        con.request("new_word", word=w)
        got = []
        for t in range(60):
            s = con.tick(t, P(t, seen=room, fixtures=fix, events=events if t % 10 == 0 else ()))
            if s.line is not None and s.line.intent == "new_word":
                got.append(s.line.text)
        n_words += bool(got)
        said += got
    assert said and all(TP.on_peak(x)[0] for x in said), [x for x in said if not TP.on_peak(x)[0]]
    assert not {x for x, _w in found} & set(said)
    # C25 over what she may say as a set: every 3 of a word's introduction lines on its peak (for each object a colour or "fell"
    # is shown on) at most 3 words a second, pooled over their measured spans
    import itertools                                                         # noqa: PLC0415
    worst = (0.0, ())
    for w in [w for w, c in TP.GROWTH if c != "frame"]:
        for n in sorted(TP.OBJECT_NOUNS):
            ok_lines = TP.intro_on_peak(w, Seen(n, n, "", "mat"))
            for k in itertools.combinations(ok_lines, 3):
                if len({TP.key(x) for x in k}) == 3:                         # a set she may draw: 3 lines of distinct words
                    worst = max(worst, (TP.set_rate(k), k))
    assert worst[0] <= 3.0, worst
    engine = ""
    if _have_engine():                                                       # the measure made again: bit for bit
        import importlib.util                                                # noqa: PLC0415
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        spec = importlib.util.spec_from_file_location("svc", os.path.join(root, "tools", "sim_voice_check.py"))
        svc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(svc)
        tmp = tempfile.mkdtemp()
        try:
            cache = V.VoiceCache(tmp)
            rng = np.random.default_rng(0)
            sample = [x for x, _w in found if x in TP.PEAK] + [str(x) for x in rng.choice(sorted(TP.PEAK), 12, replace=False)]
            for text in sample:
                w = TP.PEAK[text][0]
                c = cache.clip(text, "new_word", emphasis=w, heard=False)
                pk = [(word, None if not np.isfinite(svc.f0_word(c.pcm[a:e])[1]) else round(svc.f0_word(c.pcm[a:e])[1], 1))
                      for word, a, e in c.words]
                others = [(p_, word) for word, p_ in pk[:-1] if p_ is not None]
                top = max(others) if others else (None, None)
                span = round((c.words[-1][2] - c.words[0][1]) / V.SR, 4)
                assert [w, pk[-1][1], top[0], top[1], len(c.words), span] == TP.PEAK[text], (text, pk, TP.PEAK[text])
            cache.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        engine = f"; {len(sample)} of the table's lines synthesized and measured again, each entry the same"
    print(f"32 a new word only on its pitch peak (A34): the verifier's lines ({', '.join(repr(x) for x, _w in found[:3])}, ...) "
          f"refused by the check as measured off the peak (or never measured); {len(lines)} lines she can say with a growth "
          f"word as the new word, all measured ({len(off)} off the peak, never said); every introduction the conduct makes "
          f"({len(said)} lines over {n_words} words) on its peak; any 3 of a word's lines on its peak at most "
          f"{worst[0]:.2f} words a second (C25: {', '.join(worst[1])}){engine}")


def test_reads_trunk_and_hands():
    """finding 3 (A40): her judgments read the child's software fovea's window, which a real G1 does not show. Now every read is
    of its head's line and its hands, as a person sees a robot with no eyes: the Percept's child_target (Reader.look, her
    error from her own stream), child_holds and child_reaches (Reader.reaches, the new field)."""
    import dataclasses                                                       # noqa: PLC0415
    import inspect                                                           # noqa: PLC0415
    from body.sim.lang import percept as PC                                  # noqa: PLC0415
    fields = {f.name for f in dataclasses.fields(Percept)}
    assert "child_reaches" in fields and not [f for f in fields if "fovea" in f], fields
    assert list(inspect.signature(PC.Reader.look).parameters) == ["self", "head_pos", "head_axes", "things"]
    # her reading of its head's line: the duck straight ahead, the ball 8 degrees off it, the drum at 60 (beyond the reach)
    axes = np.eye(3)
    things = [("duck", (1.0, 0.0, 0.0)), ("ball", (1.0, math.tan(math.radians(8)), 0.0)),
              ("drum", (1.0, math.tan(math.radians(60)), 0.0)), ("mama", (-1.0, 0.0, 0.3))]
    r1, r2, r3 = PC.Reader(1), PC.Reader(1), PC.Reader(2)
    a = [r1.read((0, 0, 0), axes, things) for _ in range(400)]                # each reading (look() holds it: test 41)
    assert a == [r2.read((0, 0, 0), axes, things) for _ in range(400)], "her reading is not exact from her seed"
    assert a != [r3.read((0, 0, 0), axes, things) for _ in range(400)]
    tg = [x[0] for x in a]
    share_ball = tg.count("ball") / len(tg)
    assert set(tg) <= {"duck", "ball"} and 0.05 < share_ball < 0.4, (set(tg), share_ball)   # her error, not the fovea's
    assert all("drum" not in x[1] and "mama" not in x[1] and "duck" in x[1] for x in a)      # before its camera
    assert abs(K.READ_ERR_DEG * math.sqrt(math.pi / 2) - 5.0) < 0.02                           # Poppe et al. 2007's 5 degrees
    # a hand reaching toward the cup: its path closing on it over 3 ticks; still, or moving away: no reach
    rd = PC.Reader(1)
    cup = [("cup", (0.5, 0.0, 0.0)), ("duck", (0.0, 0.5, 0.0))]
    got = [rd.reaches({"left": (0.1 * k, 0.0, 0.0)}, cup) for k in range(4)]
    assert got[:3] == [(), (), ()] and got[3] == ("cup",), got
    rd2 = PC.Reader(1)
    assert [rd2.reaches({"left": (0.0, 0.0, 0.0)}, cup) for _ in range(5)][-1] == ()
    assert rd.reaches({"left": (0.4, 0.0, 0.0)}, cup, holds=("cup",)) == ()
    st = json.loads(json.dumps(rd.state()))
    rd3 = PC.Reader(9)
    rd3.load_state(st)
    assert rd3.look((0, 0, 0), axes, things) == rd.look((0, 0, 0), axes, things)
    # a gaze ask met by a reach: "where is the ball?", its head on nothing she can name, a hand closing on the ball
    twins = TOYS + seen(("ball_blue", "ball", "blue", "mat", True))

    def ask(reach_at, n=60):
        con = C.Conduct(seed=4, transcriber=Transcriber(None), imperfect=False)
        _no_sets(con)
        con.request("ask_where", o="ball")
        judg = []
        for t in range(n):
            s = con.tick(t, P(t, child_target=None, seen=twins, child_reaches=("ball_blue",) if reach_at(t, con) else ()))
            judg += s.judgments
        return con, judg
    con, judg = ask(lambda t, c: c.pending is not None and t >= c.pending["open"] + 2)
    assert [j[1] for j in judg] == ["met_ask"] and con.ledger.standing("ball")["asks"] == [1], judg
    con, judg = ask(lambda t, c: False)
    assert judg == [] and con.ledger.standing("ball")["asks"] == [0]
    con, judg = ask(lambda t, c: True)                                      # already reaching toward a ball: not asked
    assert judg == [] and any("already where she reads" in r[2] for r in con.fast.refused), con.fast.refused[-2:]
    # a right name by what its hand reaches toward; its word counts toward "says"; her ear expects it; Claude's situations
    con = C.Conduct(seed=3, transcriber=Transcriber(None), imperfect=False)
    _no_sets(con)
    judg = []
    for t in range(60):
        s = con.tick(t, P(t, child_target=None, child_reaches=("duck",) if t >= 30 else ()),
                     token=LX.WORD_ID["duck"] if t == 40 else None)
        judg += s.judgments
    assert (2, "right_name", "duck") in judg and con.ledger.words["duck"]["says"]["token"] == [40], judg
    p = P(0, child_target=None, child_reaches=("ball",))
    assert "ball" in expected_words(p, None, None, None, VOCAB)
    assert C.FastLayer.situation("target:ball", p) and C.FastLayer.situation("reaches:ball", p)
    assert not C.FastLayer.situation("target:ball", P(0, child_target=None, extra={"fovea": "ball"}))
    # its follow-in naming: a reach is a target (4.10); what an instrument puts in extra (its fovea's window) is never read
    con = C.Conduct(seed=1, imperfect=False)
    said, _ = run(con, [P(t, child_target=None, child_reaches=("cup",) if t >= 5 else ()) for t in range(40)])
    assert said and said[0][1].focus == "cup", said[:1]
    con = C.Conduct(seed=1, imperfect=False)
    said, _ = run(con, [P(t, child_target=None, extra={"fovea_target": "cup"}) for t in range(40)])
    assert not said, said
    print(f"33 she reads its head's line and its hands, never its fovea (A40): the Percept has child_reaches and no fovea "
          f"field; Reader.look, exact from her seed, reads the duck straight ahead and, by her 4-degree error per axis "
          f"(Poppe et al. 2007's 5 degrees mean), the ball 8 degrees off it {100 * share_ball:.0f}% of the time, never the "
          f"drum beyond the reach; Reader.reaches finds a hand closing on the cup over 3 ticks, not a still hand or a held "
          f"toy; 'where is the ball?' met by a reach toward the blue ball (a smile of 2, counted), missed without, not asked "
          f"while it already reaches; 'duck' while reaching toward the duck is a right name and counts toward 'says'; her "
          f"ear expects the reached toy; Claude's 'target:' and 'reaches:' read it; a reach starts her follow-in naming; an "
          f"instrument's fovea in extra is never read")


def test_held_pairs_a55():
    """finding 4 (A55): consts.HELD_PAIRS was the old list ((red, ball), (blue, block), ...), so the check let "the blue ball"
    through and refused "red" on the red ball, which A55 requires her to say."""
    ball_blue = seen(("ball_blue", "ball", "blue", "mat", True))
    toys = TOYS + ball_blue + seen(("cup_yellow", "cup", "yellow", "mat", True), ("car_green", "car", "green", "mat", True),
                                   ("car", "car", "orange", "mat", True))
    p = P(0, seen=toys)
    vocab = VOCAB + ("red", "blue", "yellow", "green")
    for text, refs in (("the blue ball.", ()), ("it is blue.", ("ball_blue",)), ("the red block.", ()),
                       ("it is red.", ("block_red",)), ("the yellow cup.", ()), ("it is yellow.", ("cup_yellow",)),
                       ("the green car.", ()), ("it is green.", ("car_green",))):
        ok, why = TP.check(text, vocab, None, p, refs)
        assert not ok and "held-out pair" in why, f"{text!r} said of {refs} passed ({why!r})"
    for text, refs in (("the red ball.", ()), ("it is red.", ("ball",)), ("the ball is red.", ("ball",)),
                       ("the blue block.", ()), ("it is blue.", ("block",)), ("the yellow duck.", ()),
                       ("the green cup.", ()), ("it is green.", ("cup",))):
        ok, why = TP.check(text, vocab, None, p, refs)
        assert ok, (text, why)
    assert K.HELD_PAIRS == (("blue", "ball"), ("red", "block"), ("yellow", "cup"), ("green", "car"))
    # "blue" introduced once the twins are in the world: on the blue block, never the blue ball, the held-out twin
    world = dict(TP.ROOM_AT_BIRTH, objects=dict(TP.ROOM_AT_BIRTH["objects"], ball=["blue", "red"]))
    blocks = TOYS[:3] + seen(("block", "block", "blue", "mat", True)) + ball_blue
    con = C.Conduct(seed=5, world=world)
    con.request("new_word", word="blue")
    got = []
    for t in range(60):
        s = con.tick(t, P(t, seen=blocks))
        if s.line is not None and s.line.intent == "new_word":
            got.append(s.line)
    assert len(got) == 3 and all(ln.refs == ("block",) for ln in got), got
    f = C.FastLayer(1)
    f.open_pair(("blue", "ball"))
    assert TP.check("the blue ball.", vocab, None, p, (), f.held)[0], "its test opened, the pair was still held"
    print("34 A55's held pairs: 'the blue ball.', 'the red block.', 'the yellow cup.', 'the green car.' and each colour said "
          "of its twin refused; 'the red ball.', 'it is red.' of the red ball, 'the blue block.', 'the yellow duck.', 'the "
          "green cup.' said; 'blue' introduced on the blue block, never the blue ball; a pair's test opens it")


def test_echoes():
    """finding 5: an echo of her word earned a smile, and "more?" at the feed was a name ask, so echoing "bottle" one tick after
    "more bottle?" was a met ask worth 2. Now "more?" is the feed's line, never an ask; an echo may earn her smile (a parent
    answers imitation, Goldstein and Schwade 2008: her method, disclosed), but counts toward nothing in the ledger."""
    assert C.INTENTS["feed_more"].ask is None and C.INTENTS["feed_more"].expect, "'more?' is an ask"
    bottle_seen = TOYS + seen(("bottle", "bottle", "", "mat", True))
    frames = set()
    for seed in range(12):                                                    # every frame of "more?" ("more?", "more bottle?")
        c2 = C.Conduct(seed=seed, imperfect=False)
        c2.routine = "feed"
        c2.request("feed_more")
        s = c2.tick(0, P(0, seen=bottle_seen))
        frames.add(s.line.text)
        assert c2.pending is None and not c2.ledger.trials, (s.line.text, c2.pending)
    assert frames == {"more?", "more bottle?"}, frames

    def feed(holds):
        """the verifier's probe: "more bottle?" (ending at tick 5), the child's "bottle" 2 ticks after it."""
        con = C.Conduct(seed=3, transcriber=Transcriber(None), imperfect=False)
        _no_sets(con, ("duck", "ball", "cup", "block", "block_red", "bottle"))
        con.routine = "feed"
        things = TOYS + seen(("bottle", "bottle", "", "hand" if holds else "mat", True))
        pp = lambda t: P(t, seen=things, child_holds=("bottle",) if holds else ())      # noqa: E731
        con._say(TP.Line("more bottle?", "feed_more", "comfort", "bottle", "bottle", ()), 0, pp(0), C.Say())
        judg = []
        for t in range(1, 40):
            s = con.tick(t, pp(t), token=LX.WORD_ID["bottle"] if t == 7 else None)
            judg += s.judgments
        return con, judg
    con, judg = feed(False)
    assert judg == [] and not con.ledger.trials and con.ledger.standing("bottle")["asks"] == [], judg
    con, judg = feed(True)                                                    # in its hand: an echo of a right name
    assert judg == [(2, "echo", "bottle")] and con.ledger.words["bottle"]["says"]["token"] == [], judg
    # the verifier's probe: "a duck." ending at tick 5, the child's "duck" at tick 9 while its head is on the duck: an echo
    con = C.Conduct(seed=3, transcriber=Transcriber(None), imperfect=False)
    _no_sets(con)
    con._say(TP.Line("a duck.", "label", "plain", "duck", "duck", ("duck",)), 0, P(0, child_target="duck"), C.Say())
    judg, heard = [], []
    for t in range(1, 30):
        s = con.tick(t, P(t, child_target="duck"), token=LX.WORD_ID["duck"] if t == 9 else None)
        judg += s.judgments
        heard += s.heard
    st = con.ledger.words["duck"]
    assert heard and heard[0].echo and judg == [(2, "echo", "duck")], (heard, judg)
    assert st["says"]["token"] == [] and st["echoes"]["token"] == 1 and st["asks"] == [], st
    # a name ask answered by an echo: smiled at, but void in the ledger, never met
    con = C.Conduct(seed=6, transcriber=Transcriber(None), imperfect=False)
    _no_sets(con)
    con._say(TP.Line("a cup.", "label", "plain", "cup", "cup", ("cup",)), 0, P(0), C.Say())
    for t in range(1, 8):                     # tick 7 too: the world ticks her every tick (a tick she never read voids an ask)
        con.tick(t, P(t))
    con._say(TP.Line("what is it?", "ask_what", "question", None, None, ("cup",)), 7, P(7), C.Say())
    opened = con.pending["open"]
    judg = []
    for t in range(8, 40):
        s = con.tick(t, P(t), token=LX.WORD_ID["cup"] if t == opened else None)     # begun as it was heard, 10 after her "cup"
        judg += s.judgments
    assert judg == [(2, "echo", "cup")] and con.pending is None and con.ledger.trials == [], (judg, con.pending)
    # never "understood" by echoes: ten echoes of "duck" with the duck in view and attended
    led = Ledger()
    for i in range(10):
        led.accepted(ChildWord(100 + 30 * i, "token", "duck", True, 99 + 30 * i, 100 + 30 * i, ("duck",), True),
                     P(100 + 30 * i, child_target="duck"))
    assert not led.understood("duck") and led.standing("duck")["asks"] == [] and not led.says("duck", "token")
    print("35 echoes: 'more?' at the feed is the feed's line, never an ask (no trial; its echo of 'bottle' no met ask); 'a "
          "duck.' then the child's 'duck' 4 ticks after it: an echo, smiled at as imitation (2, 'echo'), never toward 'says' "
          "or 'understood'; 'what is it?' answered by an echo of her own 'cup': smiled at, the ask void, never met; ten "
          "echoes understand nothing")


def test_claude_shows_and_forms():
    """finding 6 (low): Claude's "look! the X." and "see the X!" passed as shows, past the 40-tick redirect rule and the 2 to 1
    ratio of follow-in naming to redirects; and CLAUDE_FORMS let "a mama?", "the mama.", "see see see!" and "uh see look."
    through."""
    drum = TOYS + seen(("drum", "drum", "cyan", "mat", True))

    def steer(texts, stream, follow_in=0):
        con = _perfect(seed=1)
        _no_sets(con)
        con.fast.follow_in = follow_in
        for x in texts:
            con.fast.add_steer(x, "any", 0)
        said, _ = run(con, stream)
        return con, [(t, ln.text) for t, ln in said if ln.source == "claude"]
    duck = [P(t, child_target="duck", seen=drum) for t in range(60)]
    con, said = steer(["look! the drum.", "see the drum!"], duck)
    assert not said, f"Claude's shows of the drum, the child on the duck, were said: {said}"
    assert any("does not attend" in r[2] and "40 ticks" in r[2] for r in con.fast.refused), con.fast.refused[-2:]
    nothing = [P(t, child_target=None, seen=drum) for t in range(120)]
    con, said = steer(["look! the drum."], nothing)
    assert not said and any("2 to 1" in r[2] for r in con.fast.refused), (said, con.fast.refused[-1:])
    con, said = steer(["look! the drum.", "see the drum!"], nothing, follow_in=2)
    assert [x for _t, x in said] == ["look! the drum."] and said[0][0] >= K.REDIRECT_AFTER and con.fast.redirects == 1, said
    con, said = steer(["look! the drum.", "see the drum!"], nothing, follow_in=4)
    assert len(said) == 2 and con.fast.redirects == 2, said
    con, said = steer(["look! the duck.", "a duck."], duck)                  # the duck it attends: follow-in naming
    assert len(said) == 2 and con.fast.follow_in == 2 and con.fast.redirects == 0, said
    # the fast layer's own redirect keeps the ratio too
    con = _perfect(seed=1)
    for t in range(60):
        if t == 45:
            con.request("redirect", o="drum")
        con.tick(t, P(t, child_target=None, seen=drum))
    assert any("2 to 1" in r[2] for r in con.fast.refused if r[1] == "redirect"), con.fast.refused[-2:]
    # CLAUDE_FORMS as stated
    p = P(0, child_target="duck", child_holds=("cup",), seen=drum)
    for text in ("a mama?", "the mama.", "see see see!", "uh see look.", "the duck?", "a duck?", "here is red.",
                 "the duck is on the mama.", "the duck is on the hand.", "a foot.", "the foot.", "your mat.", "oh oh.",
                 "a mat.", "uh."):
        ok, why = TP.check(text, VOCAB + ("red",), None, p, (), source="claude")
        assert not ok and "her word" not in why, f"Claude's {text!r} passed ({why!r})"
    for text in ("mama.", "uh oh.", "oh!", "see?", "see? the duck.", "the duck!", "your foot.", "the mat.", "your cup.",
                 "you see the drum?", "it is a duck.", "the cup is on the mat."):
        ok, why = TP.check(text, VOCAB, None, P(0, child_target="duck", child_holds=("cup",), seen=drum[:2] + seen(
            ("cup", "cup", "green", "mat", True), ("drum", "drum", "cyan", "mat", True))), (), source="claude")
        assert ok, (text, why)
    print("36 Claude's shows: 'look! the drum.' and 'see the drum!' with the child on the duck are redirects: refused before 40 "
          "ticks with no target, and after them until the day's follow-in namings are 2 to 1 (2 allow one, 4 two); naming "
          "the duck it attends is follow-in naming, counted; the fast layer's own redirect keeps the ratio; CLAUDE_FORMS as "
          "stated: 'a mama?', 'the mama.', 'see see see!', 'uh see look.', a question, 'here is red.', 'on the mama' refused, "
          "'mama.', 'uh oh.', 'see? the duck.' said")


def test_imperfect_parent_and_talk_over():
    """finding 6 (low): A52 was not built (REPLY_AFTER a fixed 3), and stage 2 had no "no." for talking over her (4.4's register
    table). Now her latency is drawn from human switching pauses, she misses a share of the turns she makes no judgment of,
    she copies its arm and hand movements, and stage 2 answers a talk-over with the frown and "no."."""
    act = np.array([4, 4, 2, 4, 2, 2, 2, 4, 2, 0])

    def talk(stage, nonstop=False):
        tract = T.Tract(1)
        on = (lambda t: t % 10 < 8) if nonstop else (lambda t: 6 <= t < 9)
        snd = {t: tract.tick(act if on(t) else None) for t in range(80)}
        con = _perfect(seed=2, transcriber=Transcriber(None), stage=stage)
        if nonstop:
            con.nonstop_since, con.sound_hist = 0, [True] * K.NONSTOP[1]
        con.request("answer_bid")                     # "mama is here." (the call is its name alone since P3's eighth round)
        out = []
        for t in range(80):
            s = con.tick(t, P(t, child_target=None), tract=snd[t])
            out.append(s)
        return con, out
    con, out = talk(2)
    cut = [t for t, s in enumerate(out) if s.cut]
    assert cut and getattr(out[cut[0]], "frown", None) == "talk_over", "stage 2's talk-over: no frown"
    after = [(t, s.line) for t, s in enumerate(out) if s.line is not None and t > cut[0]]
    assert after and after[0][1].text == "no." and after[0][1].register == "no" and \
        not [j for s in out for j in s.judgments], after[:1]
    con, out = talk(1)
    assert [s.frown for s in out if s.cut] == [None] and not [s for s in out if s.line is not None and s.line.text == "no."]
    con, out = talk(2, nonstop=True)
    assert [s for s in out if s.cut] and not [s.frown for s in out if s.frown], "a frown for babble that never stops"
    for stage, want in ((1, (None, "oh!")), (2, ("hit", "no."))):              # being hit by its own act (4.10)
        con = _perfect(seed=2, stage=stage)
        s = con.tick(0, P(0, events=(("hit_her", None),)))
        assert (s.frown, s.line.text) == want, (stage, s.frown, s.line)
    # her latency: Gratier et al. 2015's switching pauses less the turn's 2 quiet ticks, at least 1; fixed only when made perfect
    con = C.Conduct(seed=7)
    lat = np.array([con._latency() for _ in range(4000)])
    assert lat.min() == 0 and 2.8 < lat.mean() < 3.05 and np.median(lat) == 2, (lat.min(), lat.mean(), np.median(lat))
    real = (lat + K.TURN_END_REST) * 150.0                                    # her realized switching pause, ms
    assert abs(real.mean() - K.REPLY_PAUSE_MS[0]) < 25 and (lat == 0).mean() < 0.3, (real.mean(), (lat == 0).mean())
    assert C.Conduct(seed=7, imperfect=False)._latency() == K.REPLY_AFTER
    lat2 = np.array([C.Conduct(seed=7)._latency() for _ in range(3)])
    assert lat2.tolist() == [lat[0]] * 3
    # misses: of 200 unjudged turns (a word it says of nothing it attends) about 30% get no reply; a judged turn always one
    con = C.Conduct(seed=11, transcriber=Transcriber(None))
    _no_sets(con)
    replies, n = 0, 200
    for i in range(n):
        t0 = 100 * i
        said = [t0] if con.tick(t0, P(t0, child_target=None), token=LX.WORD_ID["ball"]).line is not None else []
        for t in range(t0 + 1, t0 + 30):                                     # her reply (0 ticks after: said at t0 itself)
            said += [t] if con.tick(t, P(t, child_target=None)).line is not None else []
        replies += bool(said)
        assert len(said) <= 1 and all(0 <= x - t0 <= 20 for x in said), said
    assert con.turns == [replies, n - replies] and abs((n - replies) / n - K.MISS_TURN) < 0.1, (con.turns, replies)
    con = C.Conduct(seed=11, transcriber=Transcriber(None))
    _no_sets(con)
    for i in range(60):
        t0 = 100 * i
        s = con.tick(t0, P(t0, child_target="mama"), token=LX.WORD_ID["mama"])        # a right name each time
        assert s.judgments and (con.reply_due is not None or s.line is not None), (i, s.judgments)   # answered (at t0 itself
                                                                                                      # at a latency of 0)
        for t in range(t0 + 1, t0 + 30):
            con.tick(t, P(t, child_target="mama"))
    assert con.turns == [0, 0]
    # a miss never touches a judgment: the reply owed to a met ask this tick stands when the child's turn is missed
    for seed in range(40):
        con = C.Conduct(seed=seed)
        st = con.imp.bit_generator.state
        con._latency()
        missed = con.imp.random() < K.MISS_TURN
        con.imp.bit_generator.state = st
        if missed:
            break
    due = dict(tick=5, kind="confirm", word="ball", obj="ball")
    con.reply_due = dict(due)
    con._owe_reply(5, ChildWord(5, "tract", None, False, 1, 5), P(5), C.Say())
    assert missed and con.reply_due == due and con.turns == [0, 1], (con.reply_due, con.turns)
    # copying: its left arm raised, her right arm raised within 1-2 s; another movement within her gap not copied; none while
    # an ask is pending
    con = C.Conduct(seed=3)
    copies, gap, before150 = [], None, None
    for t in range(200):
        ev = (("arm_raise", "left"),) if t in (10, 14) else ((("wave", "right"),) if t == 150 else ())
        s = con.tick(t, P(t, child_target=None, events=ev))
        copies += [(t, a) for a in s.copy]
        if t == 10:
            gap = con.copy_next
        if t == 149:
            before150 = con.copy_next
    assert copies and copies[0][1] == C.Act("copy", "arm_raise:right") and 7 <= copies[0][0] - 10 <= 13, copies
    assert len([c for c in copies if c[1].target.startswith("arm_raise")]) == 1 + (14 >= gap), (copies, gap)
    assert len([c for c in copies if c[1] == C.Act("copy", "wave:left")]) == (150 >= before150), (copies, before150)
    con = C.Conduct(seed=3, transcriber=Transcriber(None))
    _no_sets(con)
    con.request("ask_where", o="ball")
    copies = []
    for t in range(40):
        s = con.tick(t, P(t, child_target=None, events=(("wave", "left"),) if t == 3 else ()))
        copies += s.copy
    assert not copies, copies
    gaps = []
    con = C.Conduct(seed=5)
    last = None
    for t in range(20000):
        s = con.tick(t, P(t, child_target=None, events=(("wave", "left"),) if t % 5 == 0 else ()))
        for _a in s.copy:
            if last is not None:
                gaps.append(t - last)
            last = t
    rate = len(gaps) / (20000 * 0.15 / 60)
    assert 4.0 < rate < 7.0, rate
    print(f"37 stage 2's talk-over: stopped, the frown and 'no.' (the 'no.' register), the turn not judged; none in stage 1 "
          f"or for babble that never stops; her latency {lat.mean():.2f} ticks after the turn's end on average (median "
          f"{np.median(lat):.0f}, {lat.min()} to {lat.max()}; a realized pause of {real.mean():.0f} ms, sd {real.std():.0f}, "
          f"{100 * (lat == 0).mean():.0f}% on the turn-end tick; Gratier et al. 2015's 730), exact from her seed; "
          f"{n - replies} of {n} unjudged "
          f"turns missed ({K.MISS_TURN:.0%} declared; Gros-Louis et al. 2006), no judged turn missed; its raised left arm "
          f"copied with her right within 1-2 s, another only once her gap had passed, none during an ask; {rate:.1f} copies a "
          f"minute "
          f"when it waves every 5 ticks (Pawlby 1977's about 6)")


# ------------------------------------------------------------ the P3 verifier's fourth findings (1ae6942), each tested both ways
def _wkey(text):
    """a line's words, its punctuation dropped (templates.key, written here so the test runs where it is missing)."""
    return " ".join(TP.words(text))


# the room with the colour twins (B2): each colour on its original toy and on its twin, the twin the held-out target (A55)
TWIN_ROOM = TOYS + seen(("rattle", "rattle", "purple", "mat", True), ("ring", "ring", "pink", "mat", True),
                        ("stacker", "stacker", "white", "mat", True), ("bear", "bear", "brown", "mat", True),
                        ("drum", "drum", "cyan", "mat", True), ("car", "car", "orange", "mat", True),
                        ("ball_blue", "ball", "blue", "mat", True), ("cup_yellow", "cup", "yellow", "mat", True),
                        ("car_green", "car", "green", "mat", True))
TWIN_WORLD = dict(TP.ROOM_AT_BIRTH, objects=dict(TP.ROOM_AT_BIRTH["objects"], ball=["red", "blue"], block=["blue", "red"],
                                                  cup=["green", "yellow"], car=["orange", "green"]))
ALL_FIXTURES = frozenset({"mat", "sofa", "window", "table", "shelf", "door", "light", "floor"})


def test_sets_distinct_in_words():
    """finding 1 (4.5): the third round widened the introduction frames with punctuation-only twins ("a {w}." / "a {w}!", "see?
    {w}." / "see? {w}!"), so a new word's variation set came out with two lines of the same words ("red!" / "see? red!" / "see?
    red.", "a rattle." / "a rattle!" / "the rattle!"), against 4.5's frames differing by at least one word; and "red" on the red
    ball and "yellow" on the duck, where A55 has the colour words said, had only 2 lines of distinct words on their peak."""
    events = (("fell", "ball"), ("rolled", None), ("sat", None), ("got", "cup"), ("gave", "duck"))
    words_ = ("red", "yellow", "blue", "green", "eyes", "table", "door", "shake", "go", "get", "rattle", "nose")
    sets, repeats = {}, []
    for w in words_:
        for seed in range(40):
            con = C.Conduct(seed=seed, world=TWIN_WORLD)
            con.request("new_word", word=w)
            got = []
            for t in range(60):
                s = con.tick(t, P(t, seen=TWIN_ROOM, fixtures=ALL_FIXTURES, events=events if t % 10 == 0 else ()))
                if s.line is not None and s.line.intent == "new_word":
                    got.append(s.line)
            sets.setdefault(w, []).append(got)
            if len({_wkey(ln.text) for ln in got}) < len(got):
                repeats.append((w, seed, [ln.text for ln in got]))
    assert not repeats, f"{len(repeats)} sets of 3 had two lines of the same words: {repeats[:4]}"
    short = [(w, i, [ln.text for ln in g]) for w in words_ for i, g in enumerate(sets[w]) if len(g) != 3]
    assert not short, f"sets not said in full: {short[:4]}"
    # A55: each colour on its original toy, never its twin, in 3 lines of distinct words ("red" on the red ball, "yellow" on the
    # duck: the colour test's words, where A55 puts them)
    where = {w: {ln.refs for g in sets[w] for ln in g} for w in ("red", "yellow", "blue", "green")}
    assert where == {"red": {("ball",)}, "yellow": {("duck",)}, "blue": {("block",)}, "green": {("cup",)}}, where
    kinds = {w: sorted({_wkey(ln.text) for g in sets[w] for ln in g}) for w in ("red", "yellow")}
    assert all(len(v) >= 4 for v in kinds.values()), kinds
    # the same line is the same words: "ball! ball." then "ball. ball!" not within 60 ticks (4.6)
    f = C.FastLayer(1)
    ln = f.compose("echo_word", 0, P(0), w="hi")
    f.commit(ln, 0, 6, [("hi", 0, 1), ("hi", 1, 2)])
    assert f.compose("echo_word", 10, P(10), w="hi") is None and f.compose("echo_word", 61, P(61), w="hi") is not None
    # every word the world can show keeps 3 introduction lines of distinct words on its peak, on each toy it is shown on; the
    # six that do not are words she cannot show in any room (their moment is P4's, or no two sizes)
    few = {}
    for w in [w for w, c in TP.GROWTH if c != "frame"]:
        objs = [None]
        if any("{o}" in fr[0] for fr in TP.intro_frames(w)):
            objs = sorted(TP.OBJECT_NOUNS)
            if TP.GROWTH_CLASS[w] == "colour":
                objs = [n for n, cs in TWIN_WORLD["objects"].items() if w in cs and (w, n) not in K.HELD_PAIRS]
        for n in objs:
            k = {_wkey(x) for x in TP.intro_on_peak(w, None if n is None else Seen(n, n, "", "mat"))}
            if len(k) < 3:
                few.setdefault(w, []).append((n, len(k)))
    assert set(few) == {"all", "little", "now", "off", "out", "under"}, few
    assert not [w for w in few if TP.showable(w, dict(TWIN_WORLD, acts=list(C.StubMotion.DOES)))[0]]
    print(f"38 a new word's set is 3 lines of distinct words (4.5): {sum(len(v) for v in sets.values())} sets over "
          f"{len(words_)} words and 40 seeds, none with two lines of the same words (the third round's red: 40 of 40); 'red' on "
          f"the red ball and 'yellow' on the duck (A55) in full sets from {len(kinds['red'])} and {len(kinds['yellow'])} lines "
          f"of distinct words on their peak, each colour never on its twin; the same line is its words ('hi! hi.' then 'hi. "
          f"hi!' not within 60 ticks); every word the world can show keeps 3 lines of distinct words on its peak on each of "
          f"its toys; the six that do not ({', '.join(sorted(few))}) cannot be shown in any room")


def test_ask_after_her_cue():
    """finding 2 (A51's purpose): the blind rule held only while an ask was pending, so "where is the ball?" could be said the tick
    after a label, a show or a redirect of the ball ended, and a child that only follows her looks, points and shows (17-25
    ticks later, 5-tick glances) met it (the verifier: 2 of 8 latencies after a label, 1 after a show, 2 after a redirect; in a
    seeded life, 10 ticks after a follow-in set). Now, her method: a gaze or act ask about an X only CUE_CLEAR ticks after her
    last cue at an X ended (Brooks and Meltzoff 2005 scored infants' following looks over 6.5 s from an adult's head turn)."""
    twins = TOYS + seen(("ball_blue", "ball", "blue", "mat", True))

    def probe(cue, lat, answer=False, n=170):
        """she cues the ball (a label set, a show, a redirect), then L3 asks "where is the ball?" every tick from the cue's end
        until it is said; the child follows each look, point or show of hers at a thing lat ticks later for 5 ticks, and looks at
        the duck otherwise (at nothing, for the redirect); answer: it also turns to the ball 2 ticks after the ask's word. (Since
        P3's sixth round cue_end is the last tick after which her attention log showed no ball: the gap is the log's.)"""
        con = _perfect(seed=4, transcriber=Transcriber(None))
        _no_sets(con)
        con.fast.last_set.pop("ball")                                        # her cue is a set on the ball
        con.fast.last_named.pop("ball")
        con.fast.follow_in = 4                                               # the redirect's 2 to 1 (4.10)
        rest = None if cue == "redirect" else "duck"
        at = 45 if cue == "redirect" else 2                                  # a redirect after 40 ticks with no target
        follows, judg, asked, cue_end, lastx = [], [], None, None, None
        for t in range(n):
            if t == at:
                con.request(cue, o="ball")
            if cue_end is not None and asked is None and t >= cue_end:
                con.request("ask_where", o="ball")
            tg = next((x for a, x in follows if a <= t < a + 5), rest)
            if answer and con.pending is not None and t >= con.pending["open"] + 2:
                tg = "ball_blue"
            s = con.tick(t, P(t, child_target=tg, seen=twins))
            judg += s.judgments
            if s.line is not None and s.line.intent == "ask_where":
                asked = t
            if asked is None and _log_has(con.attn[-1], "ball"):
                lastx = t                                                    # her log shows a ball (her eyes, head, hand)
            for a in s.acts:
                if a.kind in ("look", "point", "show") and a.target not in ("child_eyes", "child", None):
                    follows.append((t + lat, a.target))
            if cue_end is None and s.line is not None and s.line.intent == cue and not con.fast.queue:
                cue_end = con.fast.busy_until                                # the cue's last line ends
        runs_x.append(lastx)
        return con, judg, asked, cue_end
    met, runs, runs_x = [], [], []
    for cue in ("label", "show", "redirect"):
        for lat in range(17, 26):
            con, judg, asked, cue_end = probe(cue, lat)
            runs.append((cue, lat, asked, cue_end, con.ledger.standing("ball")["asks"]))
            if [j for j in judg if j[1] == "met_ask"] or 1 in con.ledger.standing("ball")["asks"]:
                met.append((cue, lat, asked - cue_end if asked is not None else None))
    assert not met, f"a child that only follows her cues met the ask: {met}"
    assert all(a is not None and x is not None and a >= x + 1 + K.CUE_CLEAR and a >= e
               for (_c, _l, a, e, _s), x in zip(runs, runs_x)), list(zip(runs, runs_x))     # asked after the log's gap
    con, judg, asked, cue_end = probe("label", 20)
    assert any(r[1] == "ask_where" and "her own cue at a ball" in r[2] for r in con.fast.refused), con.fast.refused[-2:]
    # both ways: after the gap, a child that turns to the ball once the word is heard meets it (a smile of 2, counted)
    con, judg, asked, cue_end = probe("label", 20, answer=True)
    assert [j[1] for j in judg if j[1] == "met_ask"] == ["met_ask"] and con.ledger.standing("ball")["asks"] == [1], judg
    # L1's own gaze at a thing (in her attention log as it happens: W2's report, the stub's glance) holds an ask about it back
    # likewise; an ask about another toy is not
    con = _perfect(seed=4, transcriber=Transcriber(None))
    _no_sets(con)
    con.motion.glance("ball_blue", 0, 1)                                    # her eyes on the blue ball on tick 0
    con.tick(0, P(0, child_target="duck", seen=twins))
    con.request("ask_where", o="ball")
    con.request("ask_where", o="cup")
    said = [(t, s.line.text) for t in range(1, 60) for s in [con.tick(t, P(t, child_target="duck", seen=twins))]
            if s.line is not None]
    assert said and "cup" in said[0][1] and not [x for _t, x in said if "ball" in x], said
    assert any("her own cue at a ball (her eyes and head at ball, tick 0) ended 0 ticks ago" in r[2]
               for r in con.fast.refused), con.fast.refused[-3:]
    con = _perfect(seed=4, transcriber=Transcriber(None))
    _no_sets(con)
    con.motion.glance("ball_blue", 0, 1)
    for t in range(0, 60):
        if t in (K.CUE_CLEAR, K.CUE_CLEAR + 1):
            con.request("ask_where", o="ball")
        s = con.tick(t, P(t, child_target="duck", seen=twins))
        if s.line is not None:
            break
    assert s.line is not None and s.line.intent == "ask_where" and t == K.CUE_CLEAR + 1, (t, s.line)
    print(f"39 no ask right after her cue (A51): a child that only follows her looks, points and shows ({len(runs)} runs: a "
          f"label set, a show and a redirect of the ball, following 17-25 ticks later for 5 ticks) never meets 'where is the "
          f"ball?', which she says only {K.CUE_CLEAR} ticks (6.6 s; Brooks and Meltzoff 2005's 6.5-s response period) after "
          f"her attention log last showed a ball (the request dropped before, logged); after the gap a child that turns to "
          f"the ball once the word is heard meets it; L1's gaze at the blue ball on tick 0 (in her log) holds back an ask about "
          f"a ball to tick {K.CUE_CLEAR + 1}, not one about the cup")


def test_asks_she_can_judge():
    """finding 3: a gaze ask about a toy in her own hand was made and met by a look at her hand ("look at the ball." with the ball
    resting on "mama": worth 2, recorded as met), and after a show the toy is beside her face; "give me the bottle" with the
    bottle on the sofa counted as a miss against "understood", a trial no act could answer."""
    def ask(intent, o, things, look=None, give=None, n=70):
        con = _perfect(seed=4, transcriber=Transcriber(None))
        _no_sets(con, ("duck", "ball", "cup", "block", "block_red", "bottle"))
        con.request(intent, o=o)
        judg, said = [], []
        for t in range(n):
            pend = con.pending
            tg = look if (look and pend is not None and t >= pend["open"] + 1) else "duck"
            ev = (("gave", give),) if (give and pend is not None and t == pend["open"] + 3) else ()
            s = con.tick(t, P(t, child_target=tg, seen=things, events=ev))
            judg += s.judgments
            if s.line is not None:
                said.append(s.line.text)
        return con, judg, said
    in_hand = tuple(x if x.id != "ball" else seen1("ball", "ball", "red", "mama", True, False) for x in TOYS)
    con, judg, said = ask("ask_where", "ball", in_hand, look="ball")
    assert not [x for x in said if "ball" in x] and not judg and con.ledger.standing("ball")["asks"] == [], (said, judg)
    assert any("in her own hands" in r[2] for r in con.fast.refused if r[1] == "ask_where"), con.fast.refused[-2:]
    con, judg, said = ask("ask_give", "ball", in_hand, give="ball")                 # nor "give me the ball" of the ball she holds
    assert not [x for x in said if "ball" in x] and con.ledger.standing("ball")["asks"] == [], said
    # both ways: the ball on the mat is asked for, and a look at it meets the ask
    con, judg, said = ask("ask_where", "ball", TOYS, look="ball")
    assert [j[1] for j in judg] == ["met_ask"] and con.ledger.standing("ball")["asks"] == [1], (said, judg)
    # "give me the bottle": on the sofa, beyond its reach, never asked (no trial, no miss); on the mat, asked and met by the give
    sofa = TOYS + seen(("bottle", "bottle", "", "sofa", True))
    con, judg, said = ask("ask_give", "bottle", sofa)
    assert not [x for x in said if "bottle" in x] and con.ledger.standing("bottle")["asks"] == [], said
    assert any("within the child's reach" in r[2] for r in con.fast.refused if r[1] == "ask_give"), con.fast.refused[-2:]
    mat = TOYS + seen(("bottle", "bottle", "", "mat", True))
    con, judg, said = ask("ask_give", "bottle", mat, give="bottle")
    assert [x for x in said if "give" in x] and con.ledger.standing("bottle")["asks"] == [1], (said, judg)
    # the ledger's own test when the word is heard: an X come into her hands voids a gaze ask; an X gone beyond its reach voids
    # an act ask (a base trial alike); on the mat, neither is void
    for kind, things, void in (("gaze", in_hand, True), ("gaze", TOYS, False), ("act", sofa, True), ("act", mat, False)):
        led = Ledger()
        word = "ball" if kind == "gaze" else "bottle"
        led.ask(0, word, kind, word, 20, open_at=2)
        led.base_trial(0, word, kind, window=20)
        got = [led.observe(t, P(t, child_target="duck", seen=things)) for t in range(2, 30)]
        res = [r[3] for g in got for r in g]
        assert (res == ["void"]) == void and (res[:1] == ["missed"]) != void, (kind, void, res)
        assert (led.words[word]["base"] == []) == void, (kind, void, led.words[word]["base"])
    print("40 asks she can judge: 'where is the ball?' and 'give me the ball' are never asked of the ball in her own hands "
          "(dropped, logged; the third round's was met by a look at her hand), and a ball on the mat still is (met by a "
          "look); 'give me the bottle' is never asked of the bottle on the sofa, beyond its reach (no miss counted), and is of "
          "one on the mat (met by the give); the ledger voids a gaze ask whose X is in her hands, and an act ask with no X "
          "within its reach, when the word is heard (base trials alike)")


def _read_lines(seed, head, axes, things, n):
    """her single readings of the head's line, tick by tick, made in the test from her stream and the Reader's geometry (so it
    runs where Reader.read is missing): the thing nearest the line turned by her error, within the fovea's reach, or None."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(K.READ_STREAM,))))
    out = []
    for _ in range(n):
        ea, ee = (float(e) for e in rng.normal(0.0, K.READ_ERR_DEG, 2))
        line = np.array([math.cos(math.radians(ee)) * math.cos(math.radians(ea)),
                         math.cos(math.radians(ee)) * math.sin(math.radians(ea)), math.sin(math.radians(ee))])
        best = None
        for tid, pos in things:
            v = np.asarray(pos, np.float64) - np.asarray(head, np.float64)
            u = np.array([float(np.dot(a, v)) for a in axes]) / float(np.linalg.norm(v))
            if u[0] <= 0:
                continue
            az, el = math.degrees(math.atan2(u[1], u[0])), math.degrees(math.atan2(u[2], math.hypot(u[0], u[1])))
            if abs(az - ea) <= K.FOVEA_REACH_DEG[0] and abs(el - ee) <= K.FOVEA_REACH_DEG[1]:
                off = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u, line))))))
                if best is None or off < best[0]:
                    best = (off, tid)
        out.append(None if best is None else best[1])
    return out


def test_reading_held_three_ticks():
    """finding 4 (4.10, A40: "held 3 ticks running"): only follow-in naming held her reading of its head's line 3 ticks; right
    names, "says", the expected set, Claude's target: situations and the redirect's split read it on a single tick, so a reading
    that flickered onto the ball for one tick (her 4-degree error, the toys close), the tick its "ball" was read, earned (2,
    right_name) and a "says" entry. Now her Reader holds the reading 3 ticks running, and the Percept carries only that."""
    axes = np.eye(3)
    things = [("duck", (1.0, 0.0, 0.0)), ("ball", (1.0, math.tan(math.radians(6)), 0.0))]
    room = seen(("duck", "duck", "yellow", "mat", True), ("ball", "ball", "red", "mat", True), ("cup", "cup", "green", "mat", True))
    seed = 3
    raw = _read_lines(seed, (0, 0, 0), axes, things, 400)
    star = next(t for t in range(6, 399) if raw[t] == "ball" and raw[t - 3:t] == ["duck"] * 3 and raw[t + 1] == "duck")

    def life(turn_at=None, n=None):
        """the Conduct's own Reader reads the head facing the duck (or, from turn_at, the ball) each tick; the child's "ball"
        token at the tick `star` (or 6 ticks after the turn)."""
        con = C.Conduct(seed=seed, transcriber=Transcriber(None), imperfect=False)
        for x in ("duck", "ball", "cup"):                                    # no follow-in set in the way
            con.fast.last_set[x] = con.fast.last_named[x] = 10 ** 6
        say_at = star if turn_at is None else turn_at + 6
        judg, ps = [], {}
        for t in range(n or say_at + 12):
            face = things if turn_at is None or t < turn_at else [("ball", (1.0, 0.0, 0.0)),
                                                                   ("duck", (1.0, -math.tan(math.radians(30)), 0.0))]
            tg, before = con.reader.look((0, 0, 0), axes, face)
            ps[t] = P(t, child_target=tg, seen=tuple(seen1(s.id, s.name, s.colour, s.on, s.id in before, True) for s in room))
            s = con.tick(t, ps[t], token=LX.WORD_ID["ball"] if t == say_at else None)
            judg += [(t,) + j for j in s.judgments]
        return con, judg, ps
    con, judg, ps = life()
    assert not [j for j in judg if j[3] == "ball"] and con.ledger.words["ball"]["says"]["token"] == [], \
        f"'ball' said as her reading flickered onto the ball for one tick ({star}) was judged a right name: {judg}"
    p = ps[star]
    assert p.child_target == "duck" and "ball" not in expected_words(p, None, None, None, VOCAB), p.child_target
    assert not C.FastLayer.situation("target:ball", p) and C.FastLayer.situation("target:duck", p)
    f = C.FastLayer(1)
    f.add_steer("the ball.", "any", 0)
    f.follow_in = 4
    ln, kind = f.steer_line(star, p, redirect_ok=True)
    assert ln is not None and kind == "redirect", (ln, kind)                 # the ball not attended: a redirect (4.10's split)
    # both ways: its head turned to the ball, her reading held on it from the third reading; "ball" then is a right name
    con, judg, ps = life(turn_at=40)
    tgs = [ps[t].child_target for t in range(38, 46)]
    assert tgs[:4] == ["duck", "duck", "duck", "duck"] and tgs[4] == "ball", tgs
    assert (46, 2, "right_name", "ball") in judg and con.ledger.words["ball"]["says"]["token"] == [46], judg
    # Reader.look: a flicker of one or two readings never moves the held reading; three running do, onto a thing or onto
    # nothing; its state (the run and the held reading) is saved
    from body.sim.lang import percept as PC                                  # noqa: PLC0415
    far = [("duck", (1.0, 0.0, 0.0)), ("ball", (1.0, math.tan(math.radians(30)), 0.0))]
    turned = [("duck", (1.0, -math.tan(math.radians(30)), 0.0)), ("ball", (1.0, 0.0, 0.0))]
    away = [("duck", (-1.0, 0.0, 0.0)), ("ball", (-1.0, 0.2, 0.0))]
    rd = PC.Reader(5)
    seq = [far] * 4 + [turned] * 2 + [far] * 2 + [turned] * 3 + [away] * 2 + [turned] + [away] * 3
    got = [rd.look((0, 0, 0), axes, x)[0] for x in seq]
    assert got == [None, None] + ["duck"] * 8 + ["ball"] * 6 + [None], got
    st = json.loads(json.dumps(rd.state()))
    rd2 = PC.Reader(9)
    rd2.load_state(st)
    assert [rd2.look((0, 0, 0), axes, x)[0] for x in (turned, turned, turned)] == \
        [rd.look((0, 0, 0), axes, x)[0] for x in (turned, turned, turned)] == [None, None, "ball"]
    print(f"41 her reading held 3 ticks everywhere (4.10, A40): its head on the duck, the ball 6 degrees off, her reading flickered "
          f"onto the ball for one tick ({star}) as it said 'ball': no smile, no 'says', not expected, not Claude's 'target:ball', "
          f"a Claude line naming the ball a redirect (the third round: (2, 'right_name')); its head turned to the ball, her reading "
          f"moved on the third tick and 'ball' was a right name (2, counted); Reader.look holds through flickers of 1-2 "
          f"readings, moves after 3 (onto a thing or onto nothing), and saves its run")


def test_low_items_fourth():
    """finding 5 (low): _introduce logged "3 of its 3 lines pass (the check ...)" when the pause stopped it; ties counted as on
    the pitch peak; consts' comments still said "fovea or hand"; the conduct said her copies "wait" during an ask, where they are
    dropped (a late copy is no copy); her realized pause averaged 774 ms against Gratier's 730 (test 37 measures it now)."""
    con = C.Conduct(seed=5, world=TWIN_WORLD)
    con.fast.last_expect = 0                                               # the expectant pause after a question at tick 0
    con.request("new_word", word="rattle")
    con.tick(5, P(5, seen=TWIN_ROOM, fixtures=ALL_FIXTURES))
    why = [r[2] for r in con.fast.refused if r[1] == "new_word"]
    assert why and "not now" in why[-1] and "expectant pause" in why[-1] and "of the 3" not in why[-1], why
    # ties: the new word level with another word's peak is off it ("you see the rattle.": 'see' and 'rattle' both 262.3 Hz)
    assert TP.PEAK["you see the rattle."][1] == TP.PEAK["you see the rattle."][2]
    ok, why = TP.check("you see the rattle.", VOCAB, "rattle", held=())
    assert not ok and "level with" in why, why
    assert not [t for t, r in TP.PEAK.items() if TP.on_peak(t)[0] and r[2] is not None and r[2] >= r[1]]
    assert "tie" in TP.PEAK_META["rule"], TP.PEAK_META["rule"]
    import inspect                                                           # noqa: PLC0415
    assert "fovea or hand" not in inspect.getsource(K) and "child's fovea" not in inspect.getsource(K)
    assert "copies wait" not in C.__doc__ and "not made" in C.__doc__
    # a copy due while an ask is pending is dropped, and not made after the ask is judged
    con = C.Conduct(seed=3, transcriber=Transcriber(None))
    _no_sets(con)
    con.request("ask_where", o="ball")
    copies = []
    for t in range(120):
        s = con.tick(t, P(t, child_target=None, events=(("wave", "left"),) if t == 3 else ()))
        copies += s.copy
    assert not copies, copies
    print("42 the low items: a new word held back by the expectant pause is logged as that, not as its lines failing; a new word "
          "level with another word's peak is off it (the rule and the table), none such said; consts' comments read its head's "
          "line and hands; a copy due during an ask is dropped, never made late; her realized pause 735 ms (test 37)")


# ------------------------------------------------------------- the P3 verifier's fifth findings (c816b5a), each tested both ways
CUE_KINDS = ("look", "point", "show", "hand_over", "offer_bottle", "open_hand")   # her acts that may be at a thing
NOT_A_THING = (None, "child", "child_eyes", "child_periphery")


class _SlowMotion(C.StubMotion):
    """a motion whose acts at a thing run `extra` ticks beyond the stub's nominal times (W2's act times are unknown: a show held
    while she shakes it, an offer held out until the bottle is taken)."""

    def __init__(self, extra=0):
        super().__init__()
        self.extra = extra

    def request(self, act, tick):
        i = super().request(act, tick)
        if act.kind in CUE_KINDS and act.target not in NOT_A_THING:
            self.acts[i][5] += self.extra
        return i


BOTTLE_ROOM = TOYS + seen(("bottle", "bottle", "", "mat", True))
FEED_ONLY = ("here is your bottle.", "bottle. your bottle.")                 # the feed's other frames ("bottle?" left)
ASK_ONLY = ("where is the bottle?", "look at the bottle.")                  # the ask's ("the bottle? where is the bottle?")


def _cue_probe(cue, x, extra=0, lat=None, answer=False, block=(), n=220, seed=0, stage=2, snap_at=None):
    """she cues x (the feed's line with its offer of the bottle at tick 2, a show, or a redirect with its point after 40 ticks
    with no target), then L3 asks "where is the x?" every tick from the end of the cue's last line until it is said; the child
    follows each act of hers at a thing lat ticks after her motion ends it, for 5 ticks, and looks at the duck otherwise (at
    nothing, for the redirect); answer: it turns to x 2 ticks after the ask's word is heard. block: her lines said at tick 40
    (not again within 60 ticks: the frames the verifier drew are left). snap_at: at that tick her state is saved, restored into a fresh conduct and the run continued
    there. -> (con, judgments, the tick the ask was said, the last end of her cue acts at x before it, the lines said)."""
    def fresh():
        con = _perfect(seed=seed, stage=stage, transcriber=Transcriber(None), motion=_SlowMotion(extra))
        _no_sets(con, ("duck", "ball", "cup", "block", "block_red", "bottle"))
        for y in ("duck", "cup", "block", "block_red", "bottle", "ball"):
            con.fast.last_set[y] = 10 ** 6                                      # no follow-in set in the way
        if cue != "feed":
            con.fast.last_set.pop(x)
            con.fast.last_named.pop(x)
        con.fast.follow_in = 4                                                  # the redirect's 2 to 1 (4.10)
        for k in block:
            con.fast.last_said[TP.key(k) if hasattr(TP, "key") else k] = 40
        return con
    con = fresh()
    rest = None if cue == "redirect" else "duck"
    at = 45 if cue == "redirect" else 2
    judg, asked, line_end, said = [], None, None, []
    for t in range(n):
        if snap_at is not None and t == snap_at:
            st = json.loads(json.dumps(con.state(), default=_np))
            con = fresh()
            con.load_state(st)
        if cue != "feed" and t == at:
            con.request(cue, o=x)
        if line_end is not None and asked is None and t >= line_end:
            con.request("ask_where", o=x)
        ends = [a[5] for a in con.motion.acts if a[2] in CUE_KINDS and a[3] == x]
        tg = next((x for e in ends if lat is not None and e + lat <= t < e + lat + 5), rest)
        if answer and con.pending is not None and t >= con.pending["open"] + 2:
            tg = x
        ev = (("charge_low", None),) if cue == "feed" and t == at else ()
        s = con.tick(t, P(t, child_target=tg, seen=BOTTLE_ROOM, events=ev))
        judg += s.judgments
        if s.line is not None:
            said.append((t, s.line.text))
            if s.line.intent == "ask_where" and asked is None:
                asked = t
            if line_end is None and s.line.intent == cue and not con.fast.queue:
                line_end = con.fast.busy_until                                  # the cue's last line ends
    act_end = max([a[5] for a in con.motion.acts if a[2] in CUE_KINDS and a[3] == x and (asked is None or a[1] < asked)] +
                  [line_end or 0])
    return con, judg, asked, act_end, said


def test_cue_ends_with_her_act():
    """finding 1 (A51's purpose): her cue's end was taken as her line's end, not her act's. An offer, a show or a point can
    outlast its line: the verifier's probe, stage 2, seed 0: "bottle?" ended at tick 5 but its offer of the bottle at 14; "the
    bottle? where is the bottle?" was said at 49 (its word heard at 54), and a child that only follows the offer, 41-43 ticks
    after it ended, met it, counted toward "understood". Now a cue in her lines' acts ends when her motion reports the act ended
    (its status read each tick), or with the line if that is later, and no ask about its X is made while it runs."""
    # the verifier's probe, its frames as it drew them: the stub's offer (12 ticks) outlasting "bottle?" (3 ticks)
    met, setup = [], []
    for lat in (41, 42, 43):
        con, judg, asked, act_end, said = _cue_probe("feed", "bottle", lat=lat, block=FEED_ONLY + ASK_ONLY)
        setup.append((said[0], act_end))
        if [j for j in judg if j[1] == "met_ask"] or 1 in con.ledger.standing("bottle")["asks"]:
            met.append((lat, asked, said[1]))
    assert not met, f"a child that only follows her offer, 41-43 ticks after it ended, met the ask: {met}"
    assert setup == [((2, "bottle?"), 14)] * 3, setup                     # the probe as the verifier ran it
    assert asked >= 14 + K.CUE_CLEAR and said[1] == (asked, "the bottle? where is the bottle?"), (asked, said[:2])
    con, judg, asked, act_end, said = _cue_probe("feed", "bottle", block=FEED_ONLY + ASK_ONLY)        # the child on the duck
    assert said[:2] == [(2, "bottle?"), (14 + K.CUE_CLEAR, "the bottle? where is the bottle?")], said[:2]
    why = [(r[0], r[2]) for r in con.fast.refused if r[1] == "ask_where"]
    assert [w for t, w in why if t < 14 and "still under way" in w] and \
        [w for t, w in why if t >= 14 and "ended" in w], why[:3]
    # a W2 whose acts at a thing outlast their lines by 30 ticks: the feed's offer, a show set, a redirect's point; a child
    # that follows each act 1-43 ticks after it ends never meets the ask, said only 44 ticks after the last act's end
    runs, met = [], []
    for cue, x in (("feed", "bottle"), ("show", "ball"), ("redirect", "ball")):
        for lat in range(1, K.CUE_CLEAR, 3):
            con, judg, asked, act_end, said = _cue_probe(cue, x, extra=30, lat=lat)
            runs.append((cue, lat, asked, act_end))
            if [j for j in judg if j[1] == "met_ask"] or 1 in con.ledger.standing(x)["asks"]:
                met.append((cue, lat, asked, act_end))
    assert not met, f"a child that only follows her acts met the ask: {met}"
    assert all(a is not None and a >= e + K.CUE_CLEAR and (a == e + K.CUE_CLEAR or lat + 5 > K.CUE_CLEAR)
               for _c, lat, a, e in runs), runs                               # later only while it looks at x (4.8)
    # both ways: after the gap, a child that turns to the bottle once the word is heard meets the ask (a smile of 2, counted)
    con, judg, asked, act_end, said = _cue_probe("feed", "bottle", extra=30, answer=True)
    assert asked == act_end + K.CUE_CLEAR and [j[1] for j in judg] == ["met_ask"] and \
        con.ledger.standing("bottle")["asks"] == [1], (asked, act_end, judg)
    # cue_clear (P4's base-rate trials use it) is false while the act runs, even past 44 ticks after its line, and holds from
    # the tick 44 after the motion reports it ended; a cancelled act ends its cue on the tick it is cancelled
    con = _perfect(seed=4, transcriber=Transcriber(None), motion=_SlowMotion(60))
    _no_sets(con)
    for y in ("duck", "cup", "block", "block_red", "bottle"):
        con.fast.last_set[y] = 10 ** 6                                      # no follow-in set in the way (its acts would run)
    con.fast.last_set.pop("ball")
    con.fast.last_named.pop("ball")
    con.request("show", o="ball", n=2)
    clear = {}
    for t in range(200):
        con.tick(t, P(t, child_target="duck", seen=BOTTLE_ROOM))
        clear[t] = con.cue_clear("ball", t)
    shows = [a for a in con.motion.acts if a[2] == "show"]
    end = max(a[5] for a in shows)
    assert len(shows) == 2 and not any(clear[t] for t in range(end + K.CUE_CLEAR)) and \
        all(clear[t] for t in range(end + K.CUE_CLEAR, 200)), (end, [t for t in clear if clear[t]][:2])
    con = _perfect(seed=4, transcriber=Transcriber(None), motion=_SlowMotion(60))
    _no_sets(con)
    con.fast.last_set.pop("ball")
    con.fast.last_named.pop("ball")
    con.request("show", o="ball", n=1)
    lastx, clear = None, {}
    for t in range(80):
        con.tick(t, P(t, child_target="duck", seen=BOTTLE_ROOM))
        if t == 20:                                                          # after its line ended (at most 12 ticks)
            con.motion.cancel(0)                                             # stopped from the next report (P3's seventh
        if _log_has(con.attn[-1], "ball"):                                   # round: her hand on its way back that tick)
            lastx = t
        clear[t] = con.cue_clear("ball", t)                                  # read at its own tick (a tick not yet read
    assert lastx == 21 and not clear[21 + K.CUE_CLEAR] and clear[22 + K.CUE_CLEAR], (lastx, con.attn[-4:])  # is no clear one)
    # her cue acts under way are saved: a snapshot taken while the offer runs restores to the same lines and the same ask
    _c, _j, asked0, _e, said0 = _cue_probe("feed", "bottle", extra=30, answer=True)
    _c, _j, asked1, _e, said1 = _cue_probe("feed", "bottle", extra=30, answer=True, snap_at=20)
    assert asked1 == asked0 and said1 == said0, (asked0, asked1)
    print(f"43 her cue ends when her act ends (A51): the verifier's probe ('bottle?' ended at tick 5, its offer at 14): 'the "
          f"bottle? where is the bottle?' said at {14 + K.CUE_CLEAR}, not 49, and a child that only follows the offer 41-43 "
          f"ticks after it ended never meets it (the fourth round's met it, counted); with a W2 whose acts outlast their lines "
          f"by 30 ticks (the feed's offer, a show, a redirect's point), {len(runs)} runs of a child following 1-43 ticks after "
          f"each act's end: none met, every ask said {K.CUE_CLEAR} ticks after the last act's end (later only while it looks "
          f"at the toy); after the gap a child that turns once the word is heard meets it; cue_clear false while a show runs "
          f"past its line, true from 44 ticks after it ends or is cancelled; her cue acts under way saved and restored exactly")


def test_cued_refuses_what_she_cannot_name():
    """finding 2: Conduct.cued(t, target, p=None) silently dropped a cue at an id with no percept, or one she no longer saw (a
    twin's id, "ball_blue", is no word), and "look at the ball." was said on the next tick; the fifth round made it raise. Since
    P3's sixth round L1's gaze and every act of hers are in her attention log (A51), each field by its word: an id she has seen
    is named by its word (the tick's percept, or her memory of the things she has seen), a nameable word stands for itself, and
    a target she cannot name is UNNAMED, which stands for every thing: no ask of any kind opens on its tick or within 44 ticks
    after it (fail-closed, where raising would stop the life)."""
    twins = TOYS + seen(("ball_blue", "ball", "blue", "mat", True))

    def after(target, first, then=TOYS, n=70):
        """she sees `first` on tick 0, `then` after; L1's gaze on `target` on tick 1 (W2's report); L3 asks where the ball and
        where the cup are every tick until each is said -> (the conduct, her log's eyes on tick 1, {toy: the tick its ask was
        said})."""
        con = _perfect(seed=4, transcriber=Transcriber(None))
        _no_sets(con)
        con.tick(0, P(0, child_target="duck", seen=first))
        con.motion.glance(target, 1, 1)
        said, eyes = {}, None
        for t in range(1, n):
            for o in ("ball", "cup"):
                if o not in said:
                    con.request("ask_where", o=o)
            s = con.tick(t, P(t, child_target="duck", seen=then))
            eyes = con.attn[-1]["eyes"] if t == 1 else eyes
            if s.line is not None and s.line.intent == "ask_where":
                said[s.line.refs[0]] = t
        return con, eyes, said

    for bad in ("zebra", "ball_red_3", "", 7):                                # never seen, and no word: UNNAMED
        con, eyes, said = after(bad, TOYS)
        assert eyes == [C.UNNAMED], (bad, eyes)
        assert said.get("ball", 99) >= 1 + 1 + K.CUE_CLEAR and said.get("cup", 99) >= 1 + 1 + K.CUE_CLEAR, (bad, said)
        assert any("a thing she cannot name" in r[2] for r in con.fast.refused), (bad, con.fast.refused[-1:])
    con, _e, said = after("zebra", TOYS, n=80)
    assert sorted(said.values())[0] == 1 + 1 + K.CUE_CLEAR, said                # and at once after the gap
    # the verifier's probe: she has seen the blue ball; L1's gaze at it is reported by its id: where she sees it, her log names
    # it a ball, and "where is the ball?" waits 44 ticks, while an ask about the cup does not; where she no longer sees it,
    # what a look that follows her gaze could be read as is unknown (what lies near it: P3's eighth round), so it stands for
    # every thing and holds back both
    con, eyes, said = after("ball_blue", twins, twins)
    assert eyes == ["ball"] and said["ball"] == 1 + 1 + K.CUE_CLEAR and said["cup"] == 2, (eyes, said)   # (tick 1: her eyes
                                                                                  # not on the child, so no ask then)
    con, eyes, said = after("ball_blue", twins, TOYS)
    assert eyes == [C.UNNAMED] and min(said.values()) == 1 + 1 + K.CUE_CLEAR, (eyes, said)
    # a place, or a part of the child, stands for every thing (P3's seventh round: a look that follows it is read as whatever
    # thing is nearest its line, the ball on the shelf, the cup in its hand): it holds back every ask, as an id she cannot name
    for place in ("window", "shelf", "hand"):
        con, eyes, said = after(place, TOYS)
        assert eyes == [C.UNNAMED] and min(said.values()) == 1 + 1 + K.CUE_CLEAR, (place, eyes, said)
    lg = _log_word_probe()
    assert lg == {"zebra": [C.UNNAMED], "ball_blue": ["ball"], "window": [C.UNNAMED], "shelf": [C.UNNAMED],
                  "child_eyes": ["child"], "child_periphery": ["child"], "foot": [C.UNNAMED], "trunk": [C.UNNAMED],
                  "mama": ["mama"], "eyes": [C.UNNAMED], "duck": ["duck"], None: []}, lg
    print(f"44 a target she cannot name is no dropped cue: an id she has never seen in her attention log (L1's gaze on it) is "
          f"UNNAMED and holds back every ask, the ball's and the cup's, to tick {1 + 1 + K.CUE_CLEAR} (fail-closed; the "
          f"fourth round dropped it, the fifth raised); the blue ball's id is a ball where she sees it: 'where is the ball?' "
          f"waits {K.CUE_CLEAR} ticks and the cup's ask goes on the next, and where she no longer sees it (what lies near it "
          f"unknown) it holds back both; a place (the window, the shelf) or a part of the child (its hand, its foot) stands "
          f"for every thing and holds back every ask, as the unnamed id does")


def _log_word_probe():
    """her log's words for each kind of target (Conduct._words), the blue ball in view."""
    con = _perfect(seed=4)
    p = P(1, seen=TOYS + seen(("ball_blue", "ball", "blue", "mat", True)))
    return {x: con._words(x, p) for x in ("zebra", "ball_blue", "window", "shelf", "child_eyes", "child_periphery", "foot",
                                           "trunk", "mama", "eyes", "duck", None)}


def _latency_mean():
    """the exact mean of her reply's latency in ticks (A52): round(pause / 150 ms) - TURN_END_REST at least 0, the pause lognormal
    with REPLY_PAUSE_MS's mean and sd, clipped to its range (conduct._latency), summed over the ticks from the lognormal's CDF."""
    m, sd, lo, hi = K.REPLY_PAUSE_MS
    s2 = math.log(1.0 + (sd / m) ** 2)
    mu, sg = math.log(m) - s2 / 2, math.sqrt(s2)

    def cdf(x):
        return 0.5 * (1.0 + math.erf((math.log(x) - mu) / (sg * math.sqrt(2.0))))
    tm = 1000.0 * C.TICK / C.SR

    def ticks(j):
        return max(0, j - K.TURN_END_REST)
    e = cdf(lo) * ticks(round(lo / tm)) + (1.0 - cdf(hi)) * ticks(round(hi / tm))
    for j in range(int(hi / tm) + 2):
        a, b = max(lo, (j - 0.5) * tm), min(hi, (j + 0.5) * tm)
        if b > a:
            e += (cdf(b) - cdf(a)) * ticks(j)
    return e


def test_low_items_fifth():
    """finding 3 (low): the conduct's docstring said her latency averages 3.16 ticks, where it is 2.91; the print strings of tests
    18 and 26 (and 9, 12 and 17 alike) still said "in its fovea" of what she reads, where she reads its head's line and hands."""
    import inspect                                                           # noqa: PLC0415
    import re                                                                # noqa: PLC0415
    got = re.search(r"(\d+\.\d+) ticks after the turn's end on average", C.__doc__)
    exact = _latency_mean()
    assert got is not None and float(got.group(1)) == round(exact, 2), (got and got.group(1), exact)
    assert f"({round(exact, 2):.2f})" in inspect.getsource(K), "consts' REPLY_AFTER comment"
    stale = ("in its fovea", "in the fovea", "fovea or hand", "fovea and hand", "out of its fovea")
    for fn in (test_transcriber, test_ledger_standing, test_approximation_in_context, test_asks_answered,
               test_name_ask_answered_after_its_question):
        src = inspect.getsource(fn)
        assert not [x for x in stale if x in src], (fn.__name__, [x for x in stale if x in src])
    print(f"45 the low items: the conduct's docstring gives her latency as {got.group(1)} ticks after the turn's end on average, "
          f"the exact mean of her draws ({exact:.3f}; it said 3.16), as consts does; tests 9, 12, 17, 18 and 26 say what she "
          f"reads (where she reads it looking, its hand), not its fovea")

# ------------------------------------------ the P3 verifier's sixth findings (6449921): one attention log, fail-closed (A51)
AT_CHILD_T = ("child", "child_eyes", "child_periphery")
STILL_T = (("look", "child_eyes"), ("open_hand", "child"), ("lean_in", "child_periphery"), ("withdraw", "child"))


ANY = "?"                                    # the checker's own mark: a target a following look could read as any thing
PLACES = ("shelf", "door", "window", "floor", "table")


def _rest(tg):
    """at rest or at the child itself (its face, its periphery, a hand held open to it); a part of it touched is a place."""
    return tg is None or tg in AT_CHILD_T


class _W2:
    """a W2-like motion, written here (the test's own, so these tests run on the conduct before P3's sixth round too, which read
    status() and took L1's gaze through cued() as it ended). Each act runs `lengths` ticks (drawn 0-40 from its own stream; None:
    the stub's nominal times; `fixed`: kind -> (wait, length)); some first wait their turn, reported as `wait_status` ("queued":
    not one of the four statuses); a demonstration handles the toy its act names or, naming none, one it picks itself; a hand
    touching the child reports the part it touches, and a cancelled act shows through the tick it was stopped (its eyes or hand
    on the way back: StubMotion's contract since P3's seventh round). L1's gaze goes to a toy that falls or is taken, to where
    the child looks, and (place_p, act_p) to a place (a sound at the door, the shelf) or to the child's hand that moved, from its
    own stream, never while eyes_on_child (W2's contract), and to what `glances` says ([target, from, until]). outside_p: acts it
    starts that the conduct never asked for (P4's point or touch, its own walk), never while eyes_on_child, reported with the
    rest. truth(t) is where her eyes, head and hands are directed on tick t and which acts run (an outside act as
    "outside:kind"): the checker's log, independent of the conduct's; report(t) gives it the conduct (StubMotion's contract)."""
    DOES = ("show", "walk", "wave", "pick_up", "open_hand")
    OUTSIDE_ACTS = (("point", "ball"), ("point", "car"), ("touch", "hand"), ("walk", "door"), ("look", "shelf"),
                    ("show", "cup"), ("guide", "far_arm"))

    def __init__(self, seed=0, lengths=(0, 40), queue_p=0.3, queue_max=15, l1=True, wait_status="queued", fixed=None,
                 glances=(), omit=(), place_p=0.0, act_p=0.0, outside_p=0.0):
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(77,))))
        self.lengths, self.queue_p, self.queue_max, self.l1 = lengths, queue_p, queue_max, l1
        self.wait_status, self.fixed, self.omit = wait_status, dict(fixed or {}), tuple(omit)
        self.place_p, self.act_p, self.outside_p = place_p, act_p, outside_p
        self.n_place = self.n_hand = 0        # L1's gazes at places and at its hand (the property's counts)
        self.acts = []                        # [id, tick, kind, target, during, thing, start, end, cancelled at, outside]
        self.active = []                      # ids that may still run
        self.live = []                        # ids not yet reported ended
        self.glances = [list(g) for g in glances]
        self.last_target = None
        self.p = None                         # the tick's percept (step())

    def request(self, act, tick):
        i = len(self.acts)
        thing = getattr(act, "thing", None)
        if act.kind == "do" and act.target in ("show", "pick_up") and thing is None:   # it names no toy: W2 takes one it sees
            thing = next((x.id for x in sorted(self.p.seen, key=lambda x: x.id) if x.child_sees and x.on != "hand"), None)
        if act.kind in self.fixed:
            wait, n = self.fixed[act.kind]
        elif self.lengths is None:
            wait, n = 0, (60 if act.during == "focus" else C.STUB_TICKS[act.kind])
        else:
            wait = int(self.rng.integers(1, self.queue_max + 1)) if self.rng.random() < self.queue_p else 0
            n = int(self.rng.integers(self.lengths[0], self.lengths[1] + 1))
        self.acts.append([i, int(tick), act.kind, act.target, act.during, thing, int(tick) + wait, int(tick) + wait + n, None,
                          False])
        self.active.append(i)
        self.live.append(i)
        return i

    def outside(self, kind, target, tick):
        """an act it starts that the conduct never asked for (P4's, its own), under way from this tick's report (queued first at
        queue_p, like the rest), reported with the others -> its id."""
        i = self.request(C.Act(kind, target), tick - 1)
        self.acts[i][9] = True
        return i

    def status(self, i, tick):
        a = self.acts[i]
        if a[8] is not None and tick > a[8]:
            return "cancelled"
        return self.wait_status if tick < a[6] else ("running" if tick < a[7] else "done")

    def cancel(self, i, tick=None):
        a = self.acts[i]
        a[8] = -10 ** 9 if tick is None else min(int(tick), a[8] if a[8] is not None else 10 ** 9)

    def _runs(self, a, tick):
        return a[1] < tick and a[6] <= tick < a[7] and not (a[8] is not None and tick > a[8])

    @staticmethod
    def _directs(a):
        k, tg, thing = a[2], a[3], a[5]
        if k in ("look", "walk"):
            return [("eyes", tg), ("head", tg)]
        if k == "lean_in":
            return [("head", "child")]
        if k in ("show", "point", "open_hand", "hand_over", "offer_bottle"):
            return [("hand", tg)]
        if k == "attend":
            return [("head", "child"), ("hand", "trunk")]
        if k in ("touch", "guide"):
            return [("hand", tg)]
        if k == "wave":
            return [("hand", "child")]
        if k == "pull_to_sit":
            return [("both", "arm")]
        if k in ("cover_face", "reveal_face"):
            return [("both", "mama")]
        if k == "withdraw":
            return [("hand", None)]
        if k == "copy":                                                  # a raised arm or a shaken hand points at a
            kind, _, side = (tg or ":").partition(":")                   # place (the air), a wave or an open hand at the
            return [(side or "hand", "child" if kind in ("wave", "open_hand") else "air")]    # child
        if k == "do":
            if tg in ("show", "pick_up"):
                return [("hand", thing)]
            if tg == "walk":
                return [("eyes", "door"), ("head", "door")]
            return [("hand", "child")]
        return []

    def truth(self, tick):
        """-> ({eyes, head, left, right, face}: where each is directed on this tick, the raw target, her face still (it has
        none: the conduct logs her face from its own judgments); [(kind, target)] running)."""
        self.active = [i for i in self.active if self.acts[i][7] > tick and not (self.acts[i][8] is not None and
                                                                                    self.acts[i][8] < tick)]
        f = dict(eyes="child", head="child", left=None, right=None)
        claimed, run = set(), []
        for i in self.active:
            a = self.acts[i]
            if not self._runs(a, tick):
                continue
            run.append(("outside:" + a[2] if a[9] else a[2], a[3]))
            for fld, tg in self._directs(a):
                if fld == "hand":
                    fld = "right" if "right" not in claimed else ("left" if "left" not in claimed else "right")
                for x in (("left", "right") if fld == "both" else (fld,)):
                    if x not in claimed or _rest(f[x]) or not _rest(tg):
                        f[x] = tg
                    claimed.add(x)
        for g in self.glances:
            if g[1] <= tick < g[2]:
                f["eyes"] = f["head"] = g[0]
        word = {x.id: x.name for x in self.p.seen} if self.p is not None else {}
        for x in (self.p.seen if self.p is not None else ()):                # a toy in her hands: a hand at it (the contract)
            if x.on == LX.PARENT_NAME and x.id not in (f["left"], f["right"]) and \
                    x.name not in [word.get(f[k]) for k in ("left", "right")]:
                f[next((k for k in ("left", "right") if _rest(f[k])), "left")] = x.id
        f["face"] = None
        return f, run

    def step(self, t, p, con):
        """L1 on tick t, before the conduct's: a gaze broken off while an ask is pending (eyes_on_child), a gaze ended reported
        through cued() where the conduct has it (its contract before P3's sixth round: as it ends), and new gazes."""
        self.p = p
        if con.eyes_on_child:
            for g in self.glances:
                g[2] = max(g[1], min(g[2], t))
        for g in self.glances:
            if g[2] == t and g[2] > g[1] and hasattr(con, "cued"):
                con.cued(t, g[0], p)
        self.glances = [g for g in self.glances if g[2] > t]
        if self.l1 and not con.eyes_on_child:
            for k, o in p.events:
                if o is not None and k in ("fell", "got", "lost_toy", "gave") and self.rng.random() < 0.9:
                    d = int(self.rng.integers(0, 3))
                    self.glances.append([o, t + d, t + d + int(self.rng.integers(2, 13))])
                if o in ("left", "right") and self.act_p and self.rng.random() < self.act_p:   # its hand moved: her eyes on it
                    self.glances.append(["hand", t, t + int(self.rng.integers(2, 9))])
                    self.n_hand += 1
            if p.child_target not in (None, "mama", self.last_target) and self.rng.random() < 0.5:
                d = int(self.rng.integers(1, 3))
                self.glances.append([p.child_target, t + d, t + d + int(self.rng.integers(2, 9))])
            if self.place_p and self.rng.random() < self.place_p:                             # a sound at the door, the shelf
                self.glances.append([PLACES[int(self.rng.integers(len(PLACES)))], t, t + int(self.rng.integers(2, 13))])
                self.n_place += 1
        if self.outside_p and not con.eyes_on_child and self.rng.random() < self.outside_p:
            k, tg = self.OUTSIDE_ACTS[int(self.rng.integers(len(self.OUTSIDE_ACTS)))]
            self.outside(k, tg, t)
        self.last_target = p.child_target

    def report(self, tick):
        f, _run = self.truth(tick)
        acts = {i: self.status(i, tick) for i in self.live if i not in self.omit}
        self.live = [i for i in self.live if acts.get(i) not in ("done", "refused", "cancelled")]
        return dict(f, acts=acts)

    def state(self):
        return dict(rng=self.rng.bit_generator.state, acts=[list(a) for a in self.acts], active=list(self.active),
                    live=list(self.live), glances=[list(g) for g in self.glances], last_target=self.last_target,
                    n=[self.n_place, self.n_hand])

    def load_state(self, s):
        self.rng.bit_generator.state = s["rng"]
        self.acts = [list(a) for a in s["acts"]]
        self.active, self.live = list(s["active"]), list(s["live"])
        self.glances, self.last_target = [list(g) for g in s["glances"]], s["last_target"]
        self.n_place, self.n_hand = s.get("n", (0, 0))


def _names(f, room, face_near=()):
    """the truth's fields by every word a look that follows each could be read as (the checker's own naming, P3's eighth round):
    [] at rest; the child itself "child", and her face "mama", each with the things beside her face (face_near) and in her
    hands; a thing its word, with the things near it (its near) and resting on or in it or it on or in; ANY for a place, a part
    of the child, a thing out of view, or a thing (or her face) whose near is not given (None): a look that follows it is read
    as whatever thing is nearest its line."""
    ids = {x.id: x for x in room}

    def with_near(base, near, rests):
        if near is None:
            return sorted(base | {ANY})
        return sorted(base | rests | {ids[z].name if z in ids else (LX.PARENT_NAME if z == LX.PARENT_NAME else ANY)
                                      for z in near})

    def name(v):
        if v is None:
            return []
        if v in AT_CHILD_T or v == LX.PARENT_NAME:
            return with_near({"child" if v in AT_CHILD_T else LX.PARENT_NAME}, face_near,
                             {x.name for x in room if x.on == LX.PARENT_NAME})
        x = ids.get(v)
        if x is None:
            return [ANY]
        return with_near({x.name}, getattr(x, "near", ()), {y.name for y in room if y.on == x.id} |
                         ({ids[x.on].name} if x.on in ids else set()))
    return {k: name(v) for k, v in f.items()}


def _pending_bad(names, run, x):
    """-> (X, or a target that could be read as any thing, in her log's fields; what is not still) for a tick of a pending ask
    about x (her face moving and her voice count for the call: _prop_life gives them)."""
    hit = [k for k, v in names.items() if x in v or ANY in v]
    unstill = [k for k in ("eyes", "head") if "child" not in names[k]] + \
        [k for k in ("left", "right") if names[k] and "child" not in names[k]] + [r for r in run if r not in STILL_T]
    return hit, unstill


def test_log_l1_gaze_under_way():
    """finding 1 (blocking, A51's purpose): cued() was documented "as it ends", so while L1's gaze was on X an ask about X could
    be made: the ball fell at tick 3 and her gaze went to it, "look at the ball." was said at tick 4, and a child that follows
    her head turn 13-31 ticks later met it in 19 of 43 latencies. Now L1's gaze is in her attention log as it happens (W2's
    report, before the conduct's tick), and no ask about X opens while X is in it or within 44 ticks after; and a W2 that turns
    her eyes to X while the ask is pending voids it (fail-closed)."""
    twins = TOYS + seen(("ball_blue", "ball", "blue", "mat", True))

    def probe(lat, hold=20, answer=False, rogue=None, n=150, path=None):
        """the ball falls at tick 3 and L1's gaze holds it `hold` ticks; L3 asks where the ball is every tick from 4 until it is
        said; the child looks at the ball lat ticks after her head turned (for 5 ticks), at the duck otherwise; answer: it
        turns to the ball 2 ticks after the word is heard; rogue: a tick after the ask's word when W2 turns her eyes to the
        blue ball anyway."""
        mo = _W2(lengths=None, queue_p=0, l1=False, glances=[["ball", 3, 3 + hold]])
        con = _perfect(seed=4, stage=2, transcriber=Transcriber(None), motion=mo, ledger=Ledger(path))
        _no_sets(con)
        judg, asked, head = [], None, []
        for t in range(n):
            tg = "ball" if lat is not None and 3 + lat <= t < 3 + lat + 5 else "duck"
            if answer and con.pending is not None and t >= con.pending["open"] + 2:
                tg = "ball_blue"
            p = P(t, child_target=tg, seen=twins, events=(("fell", "ball"),) if t == 3 else ())
            mo.step(t, p, con)
            if rogue is not None and con.pending is not None and t == con.pending["open"] + rogue:
                mo.glances.append(["ball_blue", t, t + 3])                    # a W2 that breaks eyes_on_child
            head.append(_names(mo.truth(t)[0], twins)["head"])
            if t >= 4 and asked is None:
                con.request("ask_where", o="ball")
            s = con.tick(t, p)
            judg += s.judgments
            if s.line is not None and s.line.intent == "ask_where" and asked is None:
                asked = t
        return con, judg, asked, head
    runs = []
    for lat in range(13, 32):
        con, judg, asked, head = probe(lat)
        last = max(t for t, h in enumerate(head[:asked + 1 if asked is not None else None]) if "ball" in h)
        runs.append((lat, asked, last, [j[1] for j in judg], con.ledger.standing("ball")["asks"]))
    assert all(a is not None and a >= last + 1 + K.CUE_CLEAR for _l, a, last, _j, _s in runs), \
        f"an ask about the ball while L1's gaze held it, or within {K.CUE_CLEAR} ticks after: {runs[:3]}"
    assert not [r for r in runs if "met_ask" in r[3] or 1 in r[4]], f"a child following her head turn met the ask: {runs}"
    assert {a for _l, a, _x, _j, _s in runs} == {3 + 20 + K.CUE_CLEAR}, runs                     # at once after the gap
    con, judg, asked, head = probe(20)
    why = [r[2] for r in con.fast.refused if r[1] == "ask_where"]
    assert "her own cue at a ball is under way (her eyes and head at ball on this tick)" in why[0], why[:2]
    # both ways: after the gap, a child that turns to the ball once its word is heard meets the ask
    con, judg, asked, head = probe(None, answer=True)
    assert [j[1] for j in judg] == ["met_ask"] and con.ledger.standing("ball")["asks"] == [1], judg
    # fail-closed: a W2 that turns her eyes to a ball while the ask is pending voids it (counted neither way), whatever the
    # child does then
    tmp = tempfile.mkdtemp()
    try:
        con, judg, asked, head = probe(None, answer=True, rogue=0, path=os.path.join(tmp, "ledger.jsonl"))
        rows = [json.loads(x) for x in open(os.path.join(tmp, "ledger.jsonl"))]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    void = [r for r in rows if r["ev"] == "outcome"]
    assert judg == [] and con.ledger.standing("ball")["asks"] == [] and con.pending is None, (judg, con.pending)
    assert len(void) == 1 and void[0]["result"] == "void" and "her attention log while it was pending: her eyes at ball" in \
        void[0]["why"], void
    print(f"46 L1's gaze under way (A51): the ball falls at tick 3 and her gaze holds it 20 ticks; 'where is the ball?' is said "
          f"only at tick {3 + 20 + K.CUE_CLEAR}, {K.CUE_CLEAR} ticks after her log last showed it (the fifth round said it at "
          f"tick 4), and a child following her head turn 13-31 ticks later never meets it ({len(runs)} latencies); after the "
          f"gap a child turning on the word meets it; a W2 that turns her eyes to a ball while the ask is pending voids it")


def test_log_demonstration_names_its_toy():
    """finding 2 (medium): a verb's demonstration handled a toy the conduct never named: "shake" and "hold" introduce with
    Act("do", "show") and "get" with Act("do", "pick_up"), no toy, and "do" was no cue kind, so nothing was recorded: after
    "look. shake!", "see? shake.", "you shake?", "where is the ball?" was said at tick 30 while the third show ran to 31. Now the
    act names its toy (the one its set is said of: templates.show_now, a toy in the child's view and not in its hand), her log
    shows her hand at it, and no ask opens while any demonstration runs, nor about its toy within 44 ticks after."""
    room = seen(("ball", "ball", "red", "mat", True), ("cup", "cup", "green", "mat", True),
                ("duck", "duck", "yellow", "hand", True), ("block", "block", "blue", "sofa", False))

    def probe(x, n=160):
        mo = _W2(lengths=None, queue_p=0, l1=False)
        con = _perfect(seed=4, stage=2, transcriber=Transcriber(None), motion=mo)
        for y in ("ball", "cup", "duck", "block"):
            con.fast.last_set[y], con.fast.last_named[y] = 0, -1000              # no follow-in set in the way
        con.request("new_word", word="shake")
        asked, hands, demos, lines = None, [], [], []
        for t in range(n):
            p = P(t, child_target="duck", child_holds=("duck",), seen=room)
            mo.step(t, p, con)
            f, run = mo.truth(t)
            hands.append(_names(f, room))
            demos.append(any(k == "do" for k, _tg in run))
            if t >= 1 and asked is None:
                con.request("ask_where", o=x)
            s = con.tick(t, p)
            if s.line is not None:
                lines.append((t, s.line.text))
                if s.line.intent == "ask_where" and asked is None:
                    asked = t
        return con, asked, hands, demos, lines
    con, asked, hands, demos, lines = probe("ball")
    last = max([t for t, h in enumerate(hands[:asked + 1 if asked is not None else None])
                if any("ball" in v for v in h.values())] or [-1])
    assert asked is not None and not demos[asked] and last >= 0 and asked >= last + 1 + K.CUE_CLEAR, \
        f"an ask about the ball while her demonstration ran, or within {K.CUE_CLEAR} ticks of it: {(asked, last, lines[:4])}"
    dos = [a for a in con.motion.acts if a[2] == "do"]
    assert len(dos) == 3 and all(a[3] == "show" and a[5] == "ball" for a in dos), dos        # the act names its toy
    assert [ln for ln in lines if "shake" in ln[1]] and asked == max(a[7] for a in dos) - 1 + 1 + K.CUE_CLEAR, (asked, dos)
    # an ask about another toy waits only while a demonstration runs (her act at the ball could cue the child anywhere), and is
    # said on the tick her motion reports the last one ended
    con, asked_c, hands, demos, lines = probe("cup")
    dos = [a for a in con.motion.acts if a[2] == "do"]
    assert asked_c == max(a[7] for a in dos), (asked_c, dos, lines)
    why = [r[2] for r in con.fast.refused if r[1] == "ask_where"]
    assert any("her do at ball is still under way" in w for w in why), why[:3]
    print(f"47 a verb's demonstration names its toy (A51): 'shake' shown by 3 shows of the ball (the act's thing, the toy its "
          f"set is said of), her log with her hand at it; 'where is the ball?' said only {K.CUE_CLEAR} ticks after the last show "
          f"ended (tick {asked}; the fifth round said it at 30, the third show running to 31), 'where is the cup?' on the tick "
          f"the last show ended (tick {asked_c}), held while any ran")


def test_log_queued_status_fails_closed():
    """finding 3 (low): _cue_poll failed open: any status but "running" ended a cue, so a show waiting its turn, reported
    "queued" (it then ran 20-27), closed at its line's end, and the ask came 29 ticks after the show ended. Now only done,
    refused and cancelled end an act; anything else, or an act left out of the report, is running (fail-closed)."""
    def probe(wait_status="queued", omit=(), n=160):
        mo = _W2(lengths=None, queue_p=0, l1=False, fixed={"show": (20, 7)}, wait_status=wait_status, omit=omit)
        con = _perfect(seed=4, stage=2, transcriber=Transcriber(None), motion=mo)
        _no_sets(con)
        for y in ("duck", "cup", "block", "block_red"):
            con.fast.last_set[y] = 10 ** 6
        con.fast.last_set.pop("ball")
        con.fast.last_named.pop("ball")
        asked, line_end, hands = None, None, []
        for t in range(n):
            p = P(t, child_target="duck")
            mo.step(t, p, con)
            hands.append(_names(mo.truth(t)[0], TOYS))
            if t == 2:
                con.request("show", o="ball", n=1)
            if line_end is not None and asked is None and t >= line_end:
                con.request("ask_where", o="ball")
            s = con.tick(t, p)
            if s.line is not None and s.line.intent == "show":
                line_end = con.fast.busy_until
            if s.line is not None and s.line.intent == "ask_where" and asked is None:
                asked = t
        return con, asked, hands, line_end
    con, asked, hands, line_end = probe()
    last = max(t for t, h in enumerate(hands) if any("ball" in v for v in h.values()))
    assert asked is not None and asked >= last + 1 + K.CUE_CLEAR, \
        f"the ask came {asked - last - 1} ticks after the show, queued until tick 22, ended: {(asked, last, line_end)}"
    assert (last, asked) == (28, 28 + 1 + K.CUE_CLEAR), (last, asked)
    why = [r[2] for r in con.fast.refused if r[1] == "ask_where"]
    assert any("still under way (her motion reports it 'queued'" in w for w in why), why[:2]
    for st in ("waiting", None, "Running", "", 3, "done "):                 # any status but the four is running
        _c, a2, _h, _e = probe(wait_status=st)
        assert a2 == asked, (st, a2, asked)
    con, a3, _h, _e = probe(omit=(0,))                                     # an act never reported: running for good
    assert a3 is None and any("reports it 'unreported'" in r[2] for r in con.fast.refused), con.fast.refused[-1:]
    print(f"48 a queued act fails closed (A51): a show reported 'queued' until tick 22 (then running to 28) holds 'where is the "
          f"ball?' to tick {asked}, {K.CUE_CLEAR} ticks after it ended (the fifth round asked 29 ticks after, at its line's "
          f"end); 'waiting', None, 'Running', '', 3 and 'done ' likewise; an act left out of her motion's report is running for "
          f"good: no ask")


def test_log_copy_before_an_ask():
    """finding 4 (low): _copying ran before _choose, so a copy began on the tick an ask opened (seed 6, stage 1, tick 1235: "where
    is the ball?" with a copy, open_hand:right, running 7 ticks into it): her hands were not still. Now a copy is an act like any
    other in her log: no ask opens while one runs, is queued or is unreported, and none is made while an ask is pending."""
    runs = []
    for seed in range(8):
        mo = _W2(lengths=None, queue_p=0, l1=False)
        con = C.Conduct(seed=seed, stage=1, transcriber=Transcriber(None), motion=mo)     # imperfect: she copies (A52)
        _no_sets(con)
        due, asked, bad = None, None, []
        for t in range(120):
            p = P(t, child_target=None, events=(("open_hand", "left"),) if t == 5 else ())
            mo.step(t, p, con)
            f, run = mo.truth(t)
            if con.pending is not None:
                hit, unstill = _pending_bad(_names(f, TOYS), run, "ball")
                if hit or unstill:
                    bad.append((t, hit, unstill))
            if due is not None and t >= due and asked is None:
                con.request("ask_where", o="ball")                          # asked for from the tick its copy is due
            s = con.tick(t, p)
            if t == 5 and con.copies:
                due = con.copies[0][0]
            if s.line is not None and s.line.intent == "ask_where" and asked is None:
                asked = t
        runs.append((seed, due, asked, bad[:2]))
    assert all(a is not None and not b for _s, _d, a, b in runs), f"her hands not still while the ask was pending: {runs}"
    assert all(a == d + C.STUB_TICKS["copy"] for _s, d, a, _b in runs), runs          # said as her copy ended
    print(f"49 a copy is in her log (A51): its copy of an open hand due on the tick L3 asks where the ball is ({len(runs)} "
          f"seeds): the ask waits until her motion reports the copy ended ({C.STUB_TICKS['copy']} ticks), and while it is "
          f"pending her eyes and head stay on the child and her hands at rest (the fifth round opened it with the copy running "
          f"7 ticks into it)")


PROP_ROOM = (("duck", "duck", "yellow"), ("ball", "ball", "red"), ("cup", "cup", "green"), ("block_red", "block", "red"),
             ("ball_blue", "ball", "blue"), ("bottle", "bottle", ""))
PROP_SHELF = (("car", "car", "orange"),)                                # on the shelf: in its view, beyond its reach
PROP_NEAR = {"ball": ("cup",), "cup": ("ball",), "duck": ("block_red",), "block_red": ("duck",)}   # within NEAR_DEG of each
                                                                        # other as the child sees them (P3's eighth round)
PROP_FACE = ((), (), (), (), ("ball_blue",), ("bottle",), None)        # beside her face: none, a toy, unknown (the world silent)
PROP_REQS = (("ask_where", 3), ("ask_give", 2), ("ask_what", 2), ("call", 2), ("show", 2), ("label", 1), ("redirect", 1),
             ("new_word", 1), ("peekaboo_hide", 1), ("peekaboo", 1), ("body", 1), ("motor_sit", 1))
PROP_WORDS = ("shake", "hold", "get", "go", "shelf", "door", "table")   # verbs shown on a toy, and fixtures shown by a point
PROP_FIXTURES = ROOMS | {"shelf", "door", "table"}
PROP_W2 = dict(place_p=0.004, act_p=0.1, outside_p=0.003)              # L1 at places and its hand; acts she did not ask for
PROP_SKIP = 0.003                                                       # ticks the world does not tick her (the log's gaps)


def _prop_world(seed, stage, n):
    """a random life's script, from its own stream: where the child looks and what it holds, a ball resting in her hands at
    times, a toy beside her face at times (or the world silent about it), the events (falls and takes she may glance at,
    movements she may copy, a low charge, hits, pain), L3's requests (asks of every kind, shows, labels, redirects, verbs to
    demonstrate, peekaboo), and the draws its answers use (it answers a pending ask now and then: the world reads con.pending,
    so a replay from a snapshot is exact)."""
    rng = np.random.default_rng([seed, stage, 5150])
    ids = [r[0] for r in PROP_ROOM + PROP_SHELF]
    names, wts = zip(*PROP_REQS)
    tgt, holds, hers, faces, ev, req = [], [], [], [], [], [[] for _ in range(n)]
    cur_t = cur_h = cur_m = None
    cur_f = ()
    for t in range(n):
        if rng.random() < 1 / 30:
            cur_t = ([None, "mama"] + ids)[int(rng.integers(len(ids) + 2))]
        if rng.random() < 1 / 150:
            cur_h = (None, None, "cup", "duck")[int(rng.integers(4))]
        if rng.random() < 1 / 200:
            cur_m = (None, None, None, "ball", "ball_blue")[int(rng.integers(5))]
        if rng.random() < 1 / 300:
            cur_f = PROP_FACE[int(rng.integers(len(PROP_FACE)))]
        tgt.append(cur_t)
        holds.append(cur_h)
        hers.append(cur_m if cur_m != cur_h else None)
        faces.append(cur_f)
        e = []
        if rng.random() < 0.012:
            e.append(("fell", ids[int(rng.integers(len(ids)))]))
        if rng.random() < 0.004:
            e.append(("got", ids[int(rng.integers(len(ids)))]))
        if rng.random() < 0.03:
            e.append((("wave", "arm_raise", "shake", "open_hand")[int(rng.integers(4))], ("left", "right")[int(rng.integers(2))]))
        if rng.random() < 0.002:
            e.append(("charge_low", None))
        if rng.random() < 0.002:
            e.append(("hit_her", None))
        if rng.random() < 0.001:
            e.append(("pain", None))
        ev.append(tuple(e))
        if rng.random() < 0.08:
            k = names[int(rng.choice(len(names), p=np.array(wts) / sum(wts)))]
            o = ids[int(rng.integers(len(ids)))]
            kw = {"call": {}, "peekaboo_hide": {}, "peekaboo": {}, "motor_sit": {},
                  "body": dict(b=sorted(TP.BODY)[int(rng.integers(len(TP.BODY)))]),
                  "new_word": dict(word=PROP_WORDS[int(rng.integers(len(PROP_WORDS)))])}.get(k, dict(o=o))
            for u in ((t,) if not k.startswith("ask") and k != "call" else range(t, min(n, t + 241), 5)):
                req[u].append((k, kw))                                 # an ask the episode keeps wanting for 240 ticks
    return dict(tgt=tgt, holds=holds, hers=hers, faces=faces, ev=ev, req=req, u=rng.random(n), v=rng.random(n),
                skip=rng.random(n) < PROP_SKIP)


def _prop_percept(w, t, con, her):
    """the tick's percept and token: the script (her: the toy in her hands, kept by _prop_life while an ask is pending: she
    takes or puts down nothing during one; the toys near each other, and beside her face), and the child's answer to a pending
    ask when its draw says so (a look at the ask's X, a look at her face for the call, the name's word, or the give: the toy in
    her hand on the tick it is given, as the world shows it, P3's eighth round) -> (percept, token, room, the toy given)."""
    tg, hold, ev = w["tgt"][t], w["holds"][t], list(w["ev"][t])
    word = {i: nm for i, nm, _c in PROP_ROOM + PROP_SHELF}
    pd, tok, gave = con.pending, None, None
    if pd is not None and w["u"][t] < 0.2:
        pick = next((i for i, nm, _c in PROP_ROOM + PROP_SHELF if nm == pd["word"] and i not in (hold, her)), None)
        if pd["kind"] == "gaze" and pick is not None:
            tg = pick
        elif pd["kind"] == "call":
            tg = "mama"
        elif pd["kind"] == "act" and pick is not None and w["u"][t] < 0.05:
            ev.append(("gave", pick))
            gave = her = pick
        elif pd["kind"] == "name" and w["u"][t] < 0.1:
            tok = LX.WORD_ID.get(pd["word"])
    if tok is None and w["v"][t] < 0.01 and tg in word:
        tok = LX.WORD_ID.get(word[tg])
    room = tuple(seen1(i, nm, c, "hand" if i == hold else (LX.PARENT_NAME if i == her else "mat"), True, None,
                       PROP_NEAR.get(i, ())) for i, nm, c in PROP_ROOM) + \
        tuple(seen1(i, nm, c, "shelf", True, False) for i, nm, c in PROP_SHELF) + seen(("block", "block", "blue", "sofa", False))
    return P(t, child_target=tg, child_holds=(hold,) if hold else (), seen=room, events=tuple(ev), fixtures=PROP_FIXTURES,
             face_near=w["faces"][t]), tok, room, gave


def _prop_life(seed, stage, n, snaps=(), start=None, rule=True, w2=None):
    """one random life with the W2-like motion (acts 0-40 ticks, 30% of them queued first; L1's gazes at toys, places and its
    hand; acts she never asked for; w2: its options, PROP_W2 by default), toys near each other and beside her face, the given
    toy in her hand on the tick it is given, and ticks the world does not tick her (PROP_SKIP) -> (the conduct, the violations,
    the asks opened, a record per tick, the snapshots taken, the checker's counts). The checker reads the motion's truth and
    keeps its own account of her voice (3 ticks a word, no voice given) and her face (FACE_COURSE ticks from a judgment or a
    frown), never the conduct's log: an ask about X opened with X, a thing a look at X could be read as, or a target that
    stands for every thing (a place, a part of the child, a thing whose near is unknown) in a field on its tick or the 44
    before, or for the call any movement of her body, her face or her voice ("opened"); either in a field on a tick it was
    still pending once that tick's percept had judged it, and the ask not void on that tick ("pending"; her voice exempt for
    the call until its name is heard); her eyes or head off the child, a hand at anything but it, or an act running but those
    she may make while an ask is pending, on such a tick ("still"); the conduct's log not the truth on a tick it read ("log").
    rule=False: the ask's gate and the pending hold switched off (both ways: the checker must find what they guard)."""
    w = _prop_world(seed, stage, n)
    opts = dict(PROP_W2 if w2 is None else w2)

    def fresh():
        return C.Conduct(seed=seed, stage=stage, transcriber=Transcriber(None), motion=_W2(seed, **opts))
    con, t0, world = fresh(), 0, dict(her=None, voice=(-1, -1), face=-10 ** 9)
    if start is not None:
        t0 = start[0]
        con.load_state(json.loads(json.dumps(start[1]["con"])))
        world = dict(start[1]["world"])
    if not rule:
        con._ask_block = lambda x, t: None
        con._hold = lambda t: None
    outs = []
    judge = con.ledger._outcome

    def outcome(t_, tr, ok):                                               # the checker's record of every outcome: judged
        r = judge(t_, tr, ok)                                              # by the tick's percept (met, missed, or void by
        by_log = r == "void" and str(tr.get("void")).startswith("her attention log")    # the ledger's rules: a give before
        outs.append((t_, tr["id"], "log" if by_log else "percept"))       # its word), or void by her log
        return r
    con.ledger._outcome = outcome
    viol = dict(opened=[], pending=[], still=[], log=[])
    cnt = dict(entered={}, met_give=0, near=0, face_near=0, face=0, voice=0)
    hist, moved, asks, recs, snapped = {}, {}, [], [], {}
    for t in range(t0, n):
        if t in snaps:
            snapped[t] = dict(con=json.loads(json.dumps(con.state(), default=_np)), world=dict(world))
        if con.pending is None:                                            # she takes or puts down a toy only when no ask
            world["her"] = w["hers"][t]                                    # is pending (the world's state, snapshotted)
        p, tok, room, gave = _prop_percept(w, t, con, world["her"])
        if gave is not None:
            world["her"] = gave
        con.motion.step(t, p, con)
        f, run = con.motion.truth(t)
        names = _names(f, room, w["faces"][t])                            # (the world's own face_near, whatever the
        mama = _names({"m": LX.PARENT_NAME}, room, w["faces"][t])["m"]     # Percept of the code under test carries)
        names["face"] = mama if t <= world["face"] else []                 # her face, from judgments before this tick
        names["voice"] = mama if world["voice"][0] <= t < world["voice"][1] else []   # a line begun before this tick
        hist[t] = names
        cnt["near"] += any(len([x for x in v if x not in ("child", ANY)]) > 1 for v in names.values())
        cnt["face_near"] += w["faces"][t] != ()
        pd = con.pending
        for intent, kw in w["req"][t]:
            con.request(intent, **kw)
        before = None if pd is None else pd["trial"]
        if w["skip"][t]:                                                   # the world does not tick her (her log's gap)
            recs.append((t, "(not ticked)"))
            moved[t] = []
            continue
        s = con.tick(t, p, token=tok)
        if s.judgments or s.frown:                                         # her face moves from this tick (A3)
            world["face"] = t + getattr(K, "FACE_COURSE", 77)
            names["face"] = mama
        moved[t] = [k for k in ("eyes", "head") if "child" not in names[k]] + \
            [k for k in ("left", "right", "face", "voice") if names[k]] + \
            [r for r in run if r != ("look", "child_eyes")]                # her body not at rest: by her face, for the call
        truth = dict(names)
        if s.line is not None:                                             # her voice: 3 ticks a word from this tick
            world["voice"] = (t, t + 3 * len(TP.words(s.line.text)))
            truth["voice"] = mama
        cnt["face"] += bool(truth["face"])
        cnt["voice"] += bool(truth["voice"])
        if getattr(con, "attn", None):
            got = {k: con.attn[-1].get(k) for k in getattr(C, "LOG_FIELDS", C.FIELDS)}
            if got != {k: truth[k] for k in got}:
                viol["log"].append((t, got, truth))
        if pd is not None:                                                 # the ask pending into this tick
            res = [r for tt, tid, r in outs if tid == pd["trial"] and tt == t]
            if res != ["percept"]:                                         # judged by this tick's percept: not pending on it
                x = LX.PARENT_NAME if pd["kind"] == "call" else pd["word"]
                chk = dict(names, voice=[]) if pd["kind"] == "call" and t <= pd["open"] else names
                hit, unstill = _pending_bad(chk, run, x)
                if x == LX.PARENT_NAME:
                    hit += [m for m in moved[t] if m not in hit and not (m == "voice" and t <= pd["open"])]
                voided = res == ["log"]
                if hit and not voided:
                    viol["pending"].append((pd["tick"], x, t, hit))
                if hit and voided:
                    cnt["entered"][pd["kind"]] = cnt["entered"].get(pd["kind"], 0) + 1
                if unstill:
                    viol["still"].append((pd["tick"], x, t, unstill))
            if pd["kind"] == "act" and res == ["percept"] and con.ledger.standing(pd["word"])["asks"][-1:] == [1] and \
                    any(k == "gave" for k, _o in p.events) and any(x.on == LX.PARENT_NAME and x.name == pd["word"] for x in room):
                cnt["met_give"] += 1                                       # met with the given toy in her hand
        pa = con.pending
        if pa is not None and pa["trial"] != before:
            x = LX.PARENT_NAME if pa["kind"] == "call" else pa["word"]
            asks.append((t, pa["kind"], x))
            lo = max(t0, t - K.CUE_CLEAR)
            bad = [(u, k) for u in range(lo, t + 1) for k, v in hist[u].items() if x in v or ANY in v]
            if pa["kind"] == "call":
                bad += [(u, m) for u in range(lo, t + 1) for m in moved.get(u, ())]
            if bad:
                viol["opened"].append((t, x, bad[-1]))
        recs.append((t, None if s.line is None else s.line.text,
                     tuple((a.kind, a.target, a.during, getattr(a, "thing", None)) for a in s.acts),
                     tuple(s.judgments), tuple(a.target for a in s.copy), None if pa is None else pa["trial"],
                     None if not getattr(con, "attn", None) else json.dumps(con.attn[-1], sort_keys=True)))
    return con, viol, asks, recs, snapped, cnt


def _prop_size():
    """the property test's size: SIM_LANG_PROP="seeds,ticks" (default 20 seeds x both stages x 2,400 ticks; the sixth round's
    7 seeds, raised in the seventh as its places, touches and acts she did not ask for hold more asks back)."""
    a, b = os.environ.get("SIM_LANG_PROP", "20,2400").split(",")
    return int(a), int(b)


def test_log_property():
    """the lead's decision in P3's sixth round (five rounds each closed one route and the verifier found another): close the
    class. Random lives, both stages, with a W2-like motion whose acts run 0-40 ticks (30% queued first, reported "queued"),
    L1's gazes at falls and takes, at where the child looks, at places (the shelf, the door, the floor) and at the child's hand
    that moved, acts she never asked for (P4's point or touch, W2's walk), demonstrations, copies, a toy on the shelf, asks of
    every kind, shows, labels, redirects, fixtures introduced by a point, her touch on its body, the pull-to-sit, peekaboo, a
    ball in her hands at times, toys near each other and beside her face (P3's eighth round), hits, pain, a low charge, the
    child's answers (the given toy in her hand on the tick it is given), and ticks the world does not tick her: no ask about X
    ever opens with X, a thing a look at X could be read as, or a target that stands for every thing in her log on its tick or
    the 44 before (for the call, no movement of her body, her face or her voice either); no ask still pending once a tick's
    percept has judged it sees either in the log on that tick unless it is void there; while one is pending her eyes and head
    are on the child, her hands at rest or open to it, and nothing else runs; the conduct's log is the checker's truth every
    tick it read; and gives are met with the toy in her hand. Both ways: the same lives with the ask's gate and the pending
    hold switched off break each property, and on b301745 (no near things, no face or voice, the give voided) they break it.
    Snapshots replay exactly."""
    seeds, n = _prop_size()
    tot = dict(lives=0, ticks=0, lines=0, asks={}, held=0, demos=0, copies=0, glances=0, queued=0, void_log=0, places=0,
               hands=0, outside=0, touches=0, fixture_points=0, skipped=0, held_place=0, held_outside=0, held_gap=0, entered={},
               met_give=0, near=0, face_near=0, face=0, voice=0)
    for seed in range(seeds):
        for stage in (1, 2):
            snaps = (n // 3, (2 * n) // 3) if seed < 2 else ()
            con, viol, asks, recs, snapped, cnt = _prop_life(seed, stage, n, snaps=snaps)
            assert not any(viol.values()), (seed, stage, {k: v[:3] for k, v in viol.items() if v})
            tot["lives"] += 1
            tot["ticks"] += n
            tot["lines"] += sum(1 for r in recs if r[1] is not None)
            for _t, k, _x in asks:
                tot["asks"][k] = tot["asks"].get(k, 0) + 1
            for k, v in cnt["entered"].items():
                tot["entered"][k] = tot["entered"].get(k, 0) + v
            for k in ("met_give", "near", "face_near", "face", "voice"):
                tot[k] += cnt[k]
            tot["held"] += sum(1 for r in con.fast.refused if r[1] in ("ask_where", "ask_give", "ask_what", "call") and
                               "(A51)" in r[2])
            tot["demos"] += sum(1 for a in con.motion.acts if a[2] == "do")
            tot["copies"] += sum(1 for a in con.motion.acts if a[2] == "copy")
            tot["queued"] += sum(1 for a in con.motion.acts if a[6] > a[1])
            tot["places"] += con.motion.n_place
            tot["hands"] += con.motion.n_hand
            tot["outside"] += sum(1 for a in con.motion.acts if a[9])
            tot["touches"] += sum(1 for a in con.motion.acts if not a[9] and a[2] in ("touch", "pull_to_sit", "attend", "guide"))
            tot["fixture_points"] += sum(1 for a in con.motion.acts if not a[9] and a[2] == "point" and a[3] in PLACES)
            tot["skipped"] += sum(1 for r in recs if r[1:] == ("(not ticked)",))
            held = [r[2] for r in con.fast.refused if "(A51)" in r[2]]
            tot["held_place"] += sum(1 for w in held if "a place or a thing she cannot name" in w)
            tot["held_outside"] += sum(1 for w in held if "an act she did not ask for" in w)
            tot["held_gap"] += sum(1 for w in held if "never read" in w)
            for k, st in snapped.items():                                    # a snapshot restored into a fresh conduct
                c2, _v, _a, recs2, _s, _n = _prop_life(seed, stage, n, start=(k, st))
                assert recs2 == [r for r in recs if r[0] >= k] and c2.ledger.digest == con.ledger.digest, (seed, stage, k)
    assert tot["asks"].get("gaze", 0) >= 7 and tot["asks"].get("act", 0) >= 3 and tot["asks"].get("name", 0) >= 1 and \
        tot["asks"].get("call", 0) >= 3 and tot["demos"] >= 3 and tot["copies"] >= 20 and tot["met_give"] >= 1, tot
    assert min(tot[k] for k in ("places", "hands", "outside", "touches", "fixture_points", "skipped", "held_place",
                                "held_outside", "held_gap", "near", "face_near", "face", "voice")) >= 3, tot
    # both ways: the ask's gate and the pending hold switched off, the checker finds an ask opened with its X in her log, one
    # that saw its X enter it and was not voided (an act queued or running at the opening, then at X), and her body not still
    # while one was pending
    found = {}
    for seed in range(seeds):
        for stage in (1, 2):
            _c, viol, _a, _r, _s, _n = _prop_life(seed, stage, n, rule=False)
            for k in ("opened", "pending", "still"):
                found[k] = found.get(k, 0) + len(viol[k])
    assert all(found[k] for k in ("opened", "pending", "still")), found
    print(f"50 one attention log, fail-closed (A51; the property): {tot['lives']} random lives ({seeds} seeds x both stages x "
          f"{n} ticks) with a W2-like motion (acts 0-40 ticks, {tot['queued']} queued first; L1's gazes, {tot['places']} at "
          f"places and {tot['hands']} at its hand; {tot['outside']} acts she did not ask for; {tot['demos']} demonstrations; "
          f"{tot['copies']} copies; {tot['touches']} touches on its body; {tot['fixture_points']} points at a fixture; "
          f"{tot['skipped']} ticks not ticked; {tot['near']} ticks with a field at a thing near another, {tot['face_near']} "
          f"with a toy beside her face or the world silent on it; her face moving {tot['face']} ticks, her voice "
          f"{tot['voice']}): "
          f"{tot['lines']} lines, asks opened {tot['asks']}, {tot['held']} held back by the log's rule ({tot['held_place']} "
          f"for a place or a thing she cannot name, {tot['held_outside']} for an act she did not ask for, {tot['held_gap']} for "
          f"a tick not read); no ask opened with its X, a thing near it or a place in her log within {K.CUE_CLEAR} ticks, "
          f"none still pending saw either enter it unvoided (voided as it entered: {tot['entered'] or 'none'}), her body "
          f"still through every one, her log the truth every tick; {tot['met_give']} gives met with the toy in her hand; with "
          f"the gate and the hold off the same lives break it ({found}); {2 * min(seeds, 2) * 2} snapshots replayed exactly")


# ---------------------------------- the P3 verifier's seventh findings (883ec52): the log names what a following look reads
PLACE_FIX = ROOMS | {"shelf", "table", "door"}


def _place_probe(cue, lat=None, answer=False, n=200):
    """the verifier's probe and its kin (A51, finding 1): the ball rests where her cue points, not at the ball itself: "shelf"
    introduced with its point and look (the ball on the shelf, in the child's view, beyond its reach); her walk to the sofa for
    the night (the ball on the sofa); L1's gaze at the table (the world's glance, 8 ticks; the ball on the table); her touch on
    the child's hand that holds the ball ("your hand.", then "give me the ball"). L3 asks every tick from tick 2 until the ask is
    said. The child follows each of her cues lat ticks after it ends, for 5 ticks: its look there is read as the ball (her Reader
    reads a look as the nameable thing nearest its line, 4.10), or it gives the ball it holds; the duck otherwise. answer: it
    turns to the ball (or gives it) 2 ticks after the ask's word is heard. Her cue ticks are read from her motion itself (the
    raw place in its report, or an act of hers at the place running), whatever the conduct names it. -> (con, judgments, the
    tick the ask was said, her cue ticks, the lines said)."""
    place = {"shelf": "shelf", "walk": "sofa", "glance": "table", "touch": "hand"}[cue]
    give = cue == "touch"
    room = seen(("duck", "duck", "yellow", "mat", True), ("ball", "ball", "red", "hand" if give else place, True, give),
                ("cup", "cup", "green", "mat", True))
    con = _perfect(seed=4, stage=2, transcriber=Transcriber(None))
    _no_sets(con, ("duck", "ball", "cup"))
    intent = "ask_give" if give else "ask_where"
    cues, judg, asked, said = [], [], None, []
    for t in range(n):
        if t == 1:
            if cue == "shelf":
                con.request("new_word", word="shelf")
            elif cue == "walk":
                con.request("night")
            elif cue == "touch":
                con.request("body", b="hand")
        if cue == "glance" and t == 2:
            con.motion.glance("table", 2, 8)                                    # L1: a sound at the table (the world's call)
        ends = [u for u in cues if u + 1 not in cues and u + 1 < t]              # each cue's last tick
        follow = lat is not None and any(e + lat <= t < e + lat + 5 for e in ends)
        ans = answer and con.pending is not None and t >= con.pending["open"] + 2
        tg = "ball" if (follow or ans) and not give else "duck"
        ev = (("gave", "ball"),) if give and (follow or ans) else ()
        p = P(t, child_target=tg, child_holds=("ball",) if give else (), seen=room, fixtures=PLACE_FIX, events=ev)
        if t >= 2 and asked is None:
            con.request(intent, o="ball")
        s = con.tick(t, p)
        judg += s.judgments
        raw = con.motion.report(t)                                              # the tick's report, as the conduct read it
        if place in [raw.get(f) for f in C.FIELDS] or any(a[3] == place and a[1] < t < a[5] for a in con.motion.acts):
            cues.append(t)
        if s.line is not None:
            said.append((t, s.line.text))
            if s.line.intent == intent and asked is None:
                asked = t
    return con, judg, asked, cues, said


def test_log_place_holding_x():
    """finding 1 (blocking, A51's purpose): the log named a place by its word, so a point and a look at the shelf where the ball
    rested were no cue at the ball: "shelf" introduced with "the shelf!", "you see the shelf?", "see the shelf?", its point and
    look in her log through tick 38, and "where is the ball?" said at tick 39; a child that turns to the shelf 13-31 ticks after
    each of her cues (read as the ball, the nameable thing nearest its line) met it in 10 of 19 latencies (worth 2, counted).
    Here the child follows 13-31 ticks after each cue ends.
    Walks, a look at the floor or the table, and her touch on the child's hand holding the ball before "give me the ball" (a
    body word logged as the child, at rest) were the same route. Now each field is logged by what a look that follows it would
    be read as attending: a place, or the part of the child a hand touches, is UNNAMED, which stands for every thing (fail-
    closed): no ask of any kind opens within 44 ticks of it."""
    runs, met = [], []
    for cue in ("shelf", "walk", "glance", "touch"):
        for lat in range(13, 32):
            con, judg, asked, cues, said = _place_probe(cue, lat)
            before = [u for u in cues if asked is None or u <= asked]
            runs.append((cue, lat, asked, before[-1] if before else None))
            if [j for j in judg if j[1] == "met_ask"] or 1 in con.ledger.standing("ball")["asks"]:
                met.append((cue, lat, asked, before[-1] if before else None, said[:4]))
    assert not met, f"a child that only follows her cue at the place where the ball rests met the ask: {met[:4]}"
    assert all(a is not None and c is not None and a >= c + 1 + K.CUE_CLEAR for _q, _l, a, c in runs), \
        [r for r in runs if r[2] is None or r[3] is None or r[2] < r[3] + 1 + K.CUE_CLEAR][:4]
    at = {}
    for cue in ("shelf", "walk", "glance", "touch"):                            # the child on the duck: at once after the gap
        con, judg, asked, cues, said = _place_probe(cue)
        at[cue] = (asked, cues[-1])
        assert asked == cues[-1] + 1 + K.CUE_CLEAR, (cue, asked, cues[-1], said[:5])
        why = [r[2] for r in con.fast.refused if r[1] in ("ask_where", "ask_give")]
        assert any("a place or a thing she cannot name" in w for w in why), (cue, why[:2])
    # both ways: after the gap, a child that turns to the ball (or gives it) once the word is heard meets the ask (2, counted)
    for cue in ("shelf", "walk", "glance", "touch"):
        con, judg, asked, cues, said = _place_probe(cue, answer=True)
        assert [j[1] for j in judg] == ["met_ask"] and con.ledger.standing("ball")["asks"] == [1], (cue, judg)
    print(f"51 a cue at a place holds back every ask (A51): the verifier's probe ('shelf' introduced by its point and look, the "
          f"ball on the shelf) and its kin (her walk to the sofa the ball rests on, L1's gaze at the table under it, her touch "
          f"on the child's hand that holds it before 'give me the ball'): each logged as a place or a thing she cannot name, "
          f"at every thing, so a child following each cue 13-31 ticks later never meets the ask ({len(runs)} runs; the sixth "
          f"round said it the tick after the shelf's look and a follower met it); each ask said {K.CUE_CLEAR} ticks after her "
          f"log last showed the place ({', '.join(f'{k} {v[0]}' for k, v in at.items())}); after the gap a child answering on "
          f"the word meets it")


def _face_probe(cue, lat=None, answer=False, n=160):
    """her face is the call's X, and every movement of her body is at it or beside it (A40: a look drawn to her turning head or
    to her hand is read as her face, the nameable thing nearest its line): a show of the ball at tick 2 (her hand), L1's gaze at
    the ball as it falls at tick 2 (her head, 8 ticks), the greeting's lean-in at tick 2 (her face into its periphery), or her
    copy of its raised arm (tick 1; she copies within 1-2 s). L3 calls every tick from tick 3 (the copy: from the tick it is
    due) until the call is said. The child looks at her face lat ticks after each movement before the call ends, for 5 ticks,
    at the duck otherwise (the call's own lean-in, A3, is its ask: not followed here); answer: at her face 2 ticks after its
    name is heard. Her movement ticks are read from her motion itself (a field off the child or a hand not at rest in its
    report, or an act of hers running but her eyes on its eyes). -> (con, judgments, the tick the call was said, the movement
    ticks, the lines said)."""
    con = C.Conduct(seed=4, stage=2, transcriber=Transcriber(None), imperfect=(cue == "copy"))
    _no_sets(con)
    moved, judg, called, said, start, voice = [], [], None, [], (None if cue == "copy" else 3), -1
    for t in range(n):
        if t == 2 and cue == "show":
            con.fast.last_set.pop("ball")
            con.fast.last_named.pop("ball")
            con.request("show", o="ball", n=1)
        if t == 2 and cue == "glance":
            con.motion.glance("ball", 2, 8)
        if t == 2 and cue == "greet":
            con.request("greet")
        ends = [u for u in moved if u + 1 not in moved and u + 1 < t and (called is None or u < called)]
        follow = lat is not None and any(e + lat <= t < e + lat + 5 for e in ends)
        ans = answer and con.pending is not None and t >= con.pending["open"] + 2
        ev = (("arm_raise", "left"),) if cue == "copy" and t == 1 else ((("fell", "ball"),) if cue == "glance" and t == 2 else ())
        p = P(t, child_target="mama" if follow or ans else "duck", events=ev)
        if start is not None and t >= start and called is None:
            con.request("call")
        s = con.tick(t, p)
        if cue == "copy" and t == 1:
            start = con.copies[0][0]                                            # its copy's tick
        judg += s.judgments
        raw = con.motion.report(t)
        if s.line is not None:                                                  # her voice: 3 ticks a word (P3's eighth
            voice = t + 3 * len(TP.words(s.line.text))                          # round: at her face)
        if [f for f in ("eyes", "head") if raw.get(f) not in ("child", "child_eyes", "child_periphery")] or \
                [f for f in ("left", "right") if raw.get(f) is not None] or \
                [a for a in con.motion.acts if a[1] < t < a[5] and (a[2], a[3]) != ("look", "child_eyes")] or \
                (t < voice and not (s.line is not None and s.line.intent == "call")):
            moved.append(t)
        if s.line is not None:
            said.append((t, s.line.text))
            if s.line.intent == "call" and called is None:
                called = t
    return con, judg, called, moved, said


def test_log_her_face():
    """the class, for the call (not a finding of the verifier's; the same route by direction): the call's X is her face, and her
    log held it only where a field was at her face (peekaboo's hands). But every movement of her body is at her face or beside
    it: a child that looks at her show, her turning head, her lean-in or her copy of its arm is read as looking at her face, the
    nameable thing nearest its line (A40). On 883ec52 the call was said as soon as the movement ended, and a child that only
    follows her movements, 13-31 ticks later, met it (worth 2). Now, for her face, a tick with her eyes or head off the child, a
    hand not at rest or an act of hers under way but her eyes on its eyes could cue it: the call waits 44 ticks after her body
    was last still; since P3's eighth round her voice is in the log too (at her face: the show's line), and the call has no
    lean-in of its own."""
    runs, met = [], []
    for cue in ("show", "glance", "greet", "copy"):
        for lat in range(13, 32, 2):
            con, judg, called, moved, said = _face_probe(cue, lat)
            before = [u for u in moved if called is None or u < called]
            runs.append((cue, lat, called, before[-1] if before else None))
            if [j for j in judg if j[1] == "met_ask"] or 1 in con.ledger.standing(LX.NAME)["asks"]:
                met.append((cue, lat, called, before[-1] if before else None))
    assert not met, f"a child that only follows her movements met the call: {met[:4]}"
    assert all(c is not None and m is not None and c >= m + 1 + K.CUE_CLEAR for _q, _l, c, m in runs), \
        [r for r in runs if r[2] is None or r[3] is None or r[2] < r[3] + 1 + K.CUE_CLEAR][:4]
    at = {}
    for cue in ("show", "glance", "greet", "copy"):
        con, judg, called, moved, said = _face_probe(cue)
        last = [u for u in moved if u < called][-1]
        at[cue] = called
        assert called == last + 1 + K.CUE_CLEAR, (cue, called, last, said[:4])
        why = [r[2] for r in con.fast.refused if r[1] == "call"]
        assert any("her body, by her face" in w or "at mama" in w for w in why), (cue, why[:2])    # (her voice: at her face)
        con, judg, called, moved, said = _face_probe(cue, answer=True)            # both ways: its name heard, then a look
        assert [j[1] for j in judg] == ["met_ask"] and con.ledger.standing(LX.NAME)["asks"] == [1], (cue, judg)
    print(f"54 her face and her body (A51, A40): the call waits {K.CUE_CLEAR} ticks after her body and voice were last still, "
          f"after a show "
          f"of the ball, L1's gaze at its fall, the greeting's lean-in and her copy of its raised arm "
          f"({', '.join(f'{k} {v}' for k, v in at.items())}), so a child that only follows her movements 13-31 ticks later "
          f"never meets it ({len(runs)} runs; on 883ec52 it was said as each ended and met); a child that looks at her once its "
          f"name is heard meets it")


class _Outside(C.StubMotion):
    """the stub, reporting acts the conduct never asked for (P4's, W2's own): script [(id, from, to, status, fields)], each
    reported with its status (and its fields over the stub's) on the ticks from <= t < to."""

    def __init__(self, script):
        super().__init__()
        self.script = [tuple(x) for x in script]

    def report(self, tick):
        rep = super().report(tick)
        for mid, a, b, st, flds in self.script:
            if a <= tick < b:
                rep["acts"][mid] = st
                rep.update(flds)
        return rep


def _outside_probe(script, intent="ask_where", give_at=None, n=160):
    """L3 asks about the ball every tick from tick 1 until said (the child holds it for the give); give_at: ticks after the ask
    opens when the child gives it. -> (con, judgments, the tick the ask was said, the ledger's outcome rows)."""
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "ledger.jsonl")
        con = _perfect(seed=4, stage=2, transcriber=Transcriber(None), motion=_Outside(script), ledger=Ledger(path))
        _no_sets(con)
        give = intent == "ask_give"
        room = TOYS if not give else seen(("duck", "duck", "yellow", "mat", True), ("ball", "ball", "red", "hand", True),
                                          ("cup", "cup", "green", "mat", True))
        judg, asked = [], None
        for t in range(n):
            if t >= 1 and asked is None:
                con.request(intent, o="ball")
            ev = (("gave", "ball"),) if give and asked is not None and give_at is not None and t == asked + give_at else ()
            s = con.tick(t, P(t, child_target="duck", child_holds=("ball",) if give else (), seen=room, events=ev))
            judg += s.judgments
            if s.line is not None and s.line.intent == intent and asked is None:
                asked = t
        rows = [json.loads(x) for x in open(path)]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return con, judg, asked, [r for r in rows if r["ev"] == "outcome"]


def test_log_acts_she_did_not_ask_for():
    """finding 2 (medium): the ask's rule read only the acts the conduct asked for itself, and a hand at the child counted as
    still: an act in her motion's report the conduct never asked for (P4's, W2's own: the contract has W2 report every act asked
    of it), reported "queued" with clean fields, did not hold back "where is the ball?" (opened at tick 2); and while "give me
    the ball" was pending, an act guiding the child's hand (her left hand at "hand", act 901 running) voided nothing, and the
    child's give was a met ask (worth 2, counted). Now any act in the report that has not ended blocks every ask and voids a
    pending one unless it is a PENDING_OK act she asked for herself; it stands for every thing in her log while it runs; and a
    hand at a part of the child is a place (finding 1)."""
    # A1: an act she did not ask for, queued with clean fields, from tick 1 to 29, then reported done
    con, judg, asked, _o = _outside_probe([(900, 1, 30, "queued", {}), (900, 30, 31, "done", {})])
    assert asked == 29 + 1 + K.CUE_CLEAR, f"'where is the ball?' opened at {asked} with an act she did not ask for queued"
    a1 = asked
    why = [r[2] for r in con.fast.refused if r[1] == "ask_where"]
    assert "an act she did not ask for (her motion's id 900) is still under way (her motion reports it 'queued'" in why[0], \
        why[:1]
    for st in ("running", "waiting", None, "Running"):                        # any status but the ended three: under way
        _c, _j, a2, _o = _outside_probe([(900, 1, 30, st, {}), (900, 30, 31, "cancelled", {})])
        assert a2 == asked, (st, a2)
    _c, _j, a3, _o = _outside_probe([(900, 1, 30, "queued", {})])             # reported, then left out: under way for good
    assert a3 is None, a3
    # A2: while "give me the ball" is pending, an act she did not ask for guides the child's hand: the ask is void on that tick,
    # the child's give counted neither way; clean fields with the act alone void it too; a hand reported at the child's hand
    # (a place) voids it with no act at all
    base = _outside_probe([], intent="ask_give", give_at=6)
    op = base[2]
    assert [j[1] for j in base[1]] == ["met_ask"] and base[0].ledger.standing("ball")["asks"] == [1], base[1]   # (both ways)
    for script, want in (([(901, op + 2, op + 12, "running", {"left": "hand"})], "her left hand at ?"),
                         ([(901, op + 2, op + 12, "running", {})], "an act she did not ask for (her motion's id 901"),
                         ([(None, op + 2, op + 12, None, {"left": "hand"})], "her left hand at ?")):
        script = [(m, a, b, st, f) for m, a, b, st, f in script] if script[0][0] is not None else \
            [(902, a, b, "done", f) for _m, a, b, _s, f in script]
        con, judg, asked, outc = _outside_probe(script, intent="ask_give", give_at=6)
        assert asked == op and judg == [] and con.ledger.standing("ball")["asks"] == [], (script, judg)
        assert len(outc) == 1 and outc[0]["result"] == "void" and want in outc[0]["why"], (script, outc)
    print(f"52 acts she did not ask for (A51): one reported 'queued' with clean fields holds 'where is the ball?' until it is "
          f"reported ended and {K.CUE_CLEAR} ticks more (tick {a1}; the sixth round opened it at once, tick 1); 'running', "
          f"'waiting', None and 'Running' likewise; one reported and then left out holds every ask for good; while 'give me "
          f"the ball' is pending, one guiding the child's hand voids it (the sixth round's give was met, counted), as does one "
          f"with clean fields, or her hand reported at the child's hand")


class _OneTickVoice:
    """a voice whose every word sounds one tick."""

    def clip(self, text, register, emphasis=None):
        ws = TP.words(text)
        return _FakeClip([(w, i * 2400, (i + 1) * 2400) for i, w in enumerate(ws)], len(ws))


def test_log_low_items_seventh():
    """findings 3-5 (low; W2's contract): ticks the conduct never read counted as clean (not ticked on 50-60 while L1's gaze held
    the ball, the ask opened at 61); StubMotion.glance for a tick already reported was lost (a 1-tick glance at tick 50 made
    after the conduct's tick: the ask opened at 51); a look asked for and cancelled within one tick (a naming word one tick
    long) appeared on no tick of the log. Now a tick of her life with no log is a cue at every thing (and voids a pending ask);
    a glance for a tick already reported starts at the next report; an act is in her log from the tick she asks for it, and the
    stub shows an act cancelled from a tick on that tick (W2 reports where her body physically points until it is back)."""
    # 3: not ticked on 50-60 while L1's gaze holds the ball (the world's glance); asks about the ball and the cup from 50
    con = _perfect(seed=4, transcriber=Transcriber(None))
    _no_sets(con)
    said = {}
    for t in range(220):
        if t == 50:
            con.motion.glance("ball", 50, 11)
        if 50 <= t <= 60:
            continue                                                          # the world does not tick her
        for o in ("ball", "cup"):
            if t >= 50 and o not in said:
                con.request("ask_where", o=o)
        s = con.tick(t, P(t, child_target="duck"))
        if s.line is not None and s.line.intent == "ask_where":
            said[s.line.refs[0]] = t
    assert said.get("ball") == 60 + 1 + K.CUE_CLEAR and said.get("cup", 0) > 60 + 1 + K.CUE_CLEAR, said
    assert any("no report of tick 60 (never read)" in r[2] for r in con.fast.refused), con.fast.refused[-1:]
    # a pending ask across a tick not read is void (her body then unknown), whatever the child does after
    con = _perfect(seed=4, transcriber=Transcriber(None))
    _no_sets(con)
    judg, op = [], None
    for t in range(80):
        if t == 1:
            con.request("ask_where", o="ball")
        if op is not None and t == op + 1:
            continue                                                          # the world does not tick her once
        s = con.tick(t, P(t, child_target="ball" if op is not None and t >= op + 3 else "duck"))
        judg += s.judgments
        if op is None and con.pending is not None:
            op = con.pending["open"]
    assert op is not None and judg == [] and con.ledger.standing("ball")["asks"] == [], (op, judg)
    # 4: L1's 1-tick glance at the ball for tick 50, made after the conduct's tick 50: it shows at 51, the ask waits 44 ticks
    con = _perfect(seed=4, transcriber=Transcriber(None))
    _no_sets(con)
    asked = None
    for t in range(130):
        if t >= 51 and asked is None:
            con.request("ask_where", o="ball")
        s = con.tick(t, P(t, child_target="duck"))
        if t == 50:
            con.motion.glance("ball", 50, 1)                                  # after the tick's report: too late for it
        if s.line is not None and s.line.intent == "ask_where" and asked is None:
            asked = t
    assert asked == 51 + 1 + K.CUE_CLEAR, asked
    # 5: "a ball." with every word one tick long: her look on "ball" is asked for and cancelled within its tick; it is in her
    # log (on the tick she asked for it, and on the next, her eyes on their way back), and the ask waits 44 ticks after it
    con = _perfect(seed=4, transcriber=Transcriber(None), voice=_OneTickVoice())
    _no_sets(con)
    con.fast.last_set.pop("ball")
    con.fast.last_named.pop("ball")
    con.request("label", o="ball", n=1)
    asked, line, xs = None, None, []
    for t in range(120):
        if line is not None and asked is None:
            con.request("ask_where", o="ball")
        s = con.tick(t, P(t, child_target="duck"))
        e = con.attn[-1]
        if _log_has(e, "ball") or any("ball" in a[4] for a in e["acts"]):
            xs.append(t)
        if s.line is not None and s.line.intent == "label":
            line = (t, s.line.text)
        if s.line is not None and s.line.intent == "ask_where" and asked is None:
            asked = t
    looks = [a for a in con.motion.acts if a[2] == "look" and a[3] == "ball"]
    assert line is not None and len(looks) == 1 and looks[0][6] == looks[0][1] + 1, (line, looks)
    wt = looks[0][1]                                                          # the word's tick: asked for and cancelled there
    assert xs == [wt, wt + 1] and asked == wt + 1 + 1 + K.CUE_CLEAR, (xs, asked, line)
    print(f"53 the low items (A51, W2's contract): not ticked on 50-60 while L1's gaze held the ball, 'where is the ball?' "
          f"waits to tick {said['ball']} and the cup's ask as long (a tick not read is a cue at every thing; the sixth round "
          f"asked at 61), and a pending ask across a tick not read is void; a 1-tick glance for a tick already reported shows "
          f"on the next and holds the ask to {51 + 1 + K.CUE_CLEAR} (it was lost: 51); a look on a one-tick naming word is in "
          f"her log on its tick and the next (it was on none), and the ask waits {K.CUE_CLEAR} ticks after")


# ------------------------------------------ the P3 verifier's eighth findings (b301745): the log closed at the tick, near, face
def _ledger_rows(fn):
    """fn(path) run with a ledger file in a temporary directory -> (fn's result, the ledger's outcome rows)."""
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "ledger.jsonl")
        got = fn(path)
        rows = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return got, [r for r in rows if r["ev"] == "outcome"]


def _give_probe(shown="hers", give_at=6, gift="ball", rogue=None, n=80):
    """"give me the ball" asked from tick 1 (the child holds the ball and the cup); the child gives `gift` give_at ticks after the
    ask's word is heard; shown: where the world shows the given toy from the tick it is given ("hers": in her hand, as the
    percept's contract (on "mama") and W2's ("a hand holding a thing is at it") have it; "child": still in its hand, as the tests
    before P3's eighth round showed it); rogue: ticks after the opening when W2 reports her left hand at the ball with no give
    (a W2 breaking its contract). -> ((con, judgments, the opening tick), outcome rows)."""
    def run(path):
        mo = _Outside([] if rogue is None else [])
        con = _perfect(seed=4, stage=2, transcriber=Transcriber(None), motion=mo, ledger=Ledger(path))
        _no_sets(con)
        judg, op, given = [], None, None
        for t in range(n):
            if op is None and con.pending is None:
                con.request("ask_give", o="ball")
            if op is not None and rogue is not None and t == op + rogue:
                mo.script.append((903, t, t + 1, "done", {"left": "ball"}))
            ev = ()
            if op is not None and given is None and t == op + give_at:
                ev, given = (("gave", gift),), t
            hers = given is not None and shown == "hers"
            room = seen(("duck", "duck", "yellow", "mat", True),
                        ("ball", "ball", "red", "mama" if hers and gift == "ball" else "hand", True),
                        ("cup", "cup", "green", "mama" if hers and gift == "cup" else "hand", True))
            holds = tuple(x.id for x in room if x.on == "hand")
            s = con.tick(t, P(t, child_target="duck", child_holds=holds, seen=room, events=ev))
            judg += s.judgments
            if op is None and con.pending is not None:
                op = con.pending["open"]
        return con, judg, op
    return _ledger_rows(run)


def _gaze_rogue_probe(rogue_at=None, n=60):
    """"where is the ball?" from tick 1; the child turns to the ball 2 ticks after its word is heard; rogue_at: a tick on which
    W2 reports her eyes at the ball (breaking its contract). -> ((con, judgments, the tick it was met or None), rows)."""
    def run(path):
        mo = _Outside([] if rogue_at is None else [(904, rogue_at, rogue_at + 1, "done", {"eyes": "ball"})])
        con = _perfect(seed=4, stage=2, transcriber=Transcriber(None), motion=mo, ledger=Ledger(path))
        _no_sets(con)
        con.request("ask_where", o="ball")
        judg, met = [], None
        for t in range(n):
            tg = "ball" if con.pending is not None and t >= con.pending["open"] + 2 else "duck"
            s = con.tick(t, P(t, child_target=tg))
            if [j for j in s.judgments if j[1] == "met_ask"] and met is None:
                met = t
            judg += s.judgments
        return con, judg, met
    return _ledger_rows(run)


def test_log_met_give():
    """finding 1 (blocking): a met give was voided on the tick it succeeded. Conduct.tick held the tick's attention log to the
    pending ask (_attend, first) before the ledger judged it, and on the "gave" tick the given toy is in her hand, by the
    percept's contract (on "mama") and W2's ("a hand holding a thing is at it"): "give me the ball" given at open+6 with the
    ball shown in her hand was void (no smile, asks []), met (worth 2, asks [1]) only with the ball still shown in the child's
    hand. Now the conduct judges an ask by the tick's percept before it holds the tick's log to it: an answer the percept of a
    tick shows was made before that tick (a give at the end of its hand's path, a look held 3 readings and then 2 ticks), so her
    body on the tick cannot have cued it, and every tick before was held on its own tick (A51). A give of another toy (the cup,
    then in her hand) is its answer, missed, not void (it had gone uncounted while a right give was met); a W2 that reports her
    hand at the ball with no give still voids the ask (fail-closed)."""
    (con, judg, op), rows = _give_probe("hers")
    assert judg == [(2, "met_ask", "ball")] and con.ledger.standing("ball")["asks"] == [1], \
        f"a correct give with the ball in her hand on the tick it was given: {judg}, {rows}"
    assert [r["result"] for r in rows] == ["met"], rows
    (con2, judg2, _o), rows2 = _give_probe("child")                             # the old world's showing: the same
    assert judg2 == [(2, "met_ask", "ball")] and con2.ledger.standing("ball")["asks"] == [1], judg2
    (con3, judg3, _o), rows3 = _give_probe("hers", gift="cup")                  # the cup given: missed, counted
    assert judg3 == [] and con3.ledger.standing("ball")["asks"] == [0] and [r["result"] for r in rows3] == ["missed"], rows3
    (con4, judg4, _o), rows4 = _give_probe("hers", rogue=3)                     # her hand at the ball, no give: void
    assert judg4 == [] and con4.ledger.standing("ball")["asks"] == [] and [r["result"] for r in rows4] == ["void"] and \
        "her left hand at ball" in rows4[0]["why"], rows4
    # a base trial of the give (the ledger's own): the cup given is missed, the ball met
    for gift, want in (("cup", [0]), ("ball", [1])):
        led = Ledger()
        led.base_trial(1, "ball", "act", window=K.JUDGE_ACT)
        room = seen(("ball", "ball", "red", "hand", True), ("cup", "cup", "green", "hand", True))
        for t in range(1, 20):
            led.observe(t, P(t, seen=room, events=(("gave", gift),) if t == 6 else ()))
        assert led.standing("ball")["base"] == want, (gift, led.standing("ball"))
    # the rule is the tick's: a W2 that reports her eyes at the ball on the very tick a gaze ask is met leaves it met (the look
    # was held from 4 ticks before); on the tick before, it voids it
    (_c, judg5, met), _r = _gaze_rogue_probe()
    assert met is not None and [j[1] for j in judg5] == ["met_ask"], judg5
    (c6, judg6, met6), _r6 = _gaze_rogue_probe(met)
    assert met6 == met and c6.ledger.standing("ball")["asks"] == [1], (met, met6, judg6)
    (c7, judg7, met7), r7 = _gaze_rogue_probe(met - 1)
    assert met7 is None and c7.ledger.standing("ball")["asks"] == [] and "her eyes at ball" in r7[-1]["why"], (judg7, r7)
    print(f"55 a met give is met (A51): 'give me the ball' given {6} ticks after its word, the ball in her hand on that tick as "
          f"the world shows it: met, a smile of 2, counted (b301745 voided it); with the ball still in the child's hand, the "
          f"same; the cup given instead: missed, counted (base trials alike); her hand reported at the ball with no give: void; "
          f"a W2 reporting her eyes at the ball on the tick a gaze ask is met leaves it met (tick {met}), on the tick before "
          f"voids it: each tick's percept judges the ask before that tick's log is held to it")


def _near_probe(lat=None, answer=False, near=True, link=True, n=200):
    """the verifier's probe (A51, finding 2): the ball rests in the box (both nameable; near: each in the other's Seen.near, as
    the world's Reader.near gives it; link: the ball's on is the box), in the child's view; L1 glances at the box on ticks 2-9
    (the world's glance); L3 asks where the ball is every tick from 10 until said. The child follows her glance lat ticks after
    it ends, for 5 ticks: its look at the box is read as the ball (her Reader reads a look as the thing nearest its line, 4.10);
    the duck otherwise; answer: it turns to the ball 2 ticks after the word is heard. -> (con, judgments, the tick said)."""
    room = seen(("duck", "duck", "yellow", "mat", True), ("cup", "cup", "green", "mat", True))
    room += (seen1("box", "box", "blue", "mat", True, None, ("ball",) if near else ()),
             seen1("ball", "ball", "red", "box" if link else "mat", True, True, ("box",) if near else ()))
    con = _perfect(seed=4, stage=2, transcriber=Transcriber(None))
    _no_sets(con, ("duck", "cup", "box", "ball"))
    judg, asked = [], None
    for t in range(n):
        if t == 2:
            con.motion.glance("box", 2, 8)
        follow = lat is not None and 9 + lat <= t < 9 + lat + 5
        ans = answer and con.pending is not None and t >= con.pending["open"] + 2
        if t >= 10 and asked is None:
            con.request("ask_where", o="ball")
        s = con.tick(t, P(t, child_target="ball" if follow or ans else "duck", seen=room))
        judg += s.judgments
        if s.line is not None and s.line.intent == "ask_where" and asked is None:
            asked = t
    return con, judg, asked


def _reader_met(sep, window, trials, seed0=0):
    """her Reader (A40) on a head resting exactly on one thing, another `sep` degrees off: how many of `trials` readings are read
    as the other for a met gaze ask (the ledger's rule: her held reading on it HOLD ticks running) within `window` ticks."""
    from body.sim.lang.percept import Reader
    yx = (1.0, 0.0, 0.0)
    xx = (math.cos(math.radians(sep)), math.sin(math.radians(sep)), 0.0)
    axes = np.eye(3)
    got = 0
    for s_ in range(trials):
        r = Reader(seed0 + s_)
        r.held, r.run = "Y", ["Y", K.TARGET_TICKS]                              # its look has landed on Y
        run = 0
        for _ in range(window):
            held, _b = r.look(np.zeros(3), axes, [("Y", yx), ("X", xx)])
            run = run + 1 if held == "X" else 0
            if run >= K.HOLD:
                got += 1
                break
    return got


def test_log_near_things():
    """finding 2 (blocking-class): a cue at a named thing that holds X, or sits a few degrees from it, left no X in her log: the
    log named a thing by its own word, but her Reader reads a look as the thing nearest its line, with her 4-degree error. The
    verifier's probe: the ball in the box, L1's glance at the box on ticks 2-9, "where is the ball?" said at tick 10, and a child
    that follows her glance and is read as the ball met it (worth 2); measured with the Reader, a head resting on one thing is
    read as another 2 degrees off for 2 ticks running 29-54% of the time, 8 degrees off 3-6%. Now each field is logged by every
    thing a look that follows it could be read as: the thing, those its Seen.near names (the world's Reader.near: within
    NEAR_DEG = 20 degrees of it as seen from the child's head, and resting on or in it, or it on or in them), and those its
    Seen.on links; her face, and the child itself (the mutual gaze comes back to her face), with the things beside her face
    (Percept.face_near) and in her hands; a near the world does not give (None) stands for every thing."""
    runs, met = [], []
    for lat in range(13, 32):
        con, judg, asked = _near_probe(lat)
        runs.append((lat, asked))
        if [j for j in judg if j[1] == "met_ask"] or 1 in con.ledger.standing("ball")["asks"]:
            met.append((lat, asked))
    assert not met, f"a child that only follows her glance at the box the ball rests in met the ask: {met[:4]}"
    assert {a for _l, a in runs} == {9 + 1 + K.CUE_CLEAR}, runs
    con, judg, asked = _near_probe()
    why = [r[2] for r in con.fast.refused if r[1] == "ask_where"]
    assert asked == 9 + 1 + K.CUE_CLEAR and any("her eyes and head at ball" in w for w in why), (asked, why[:2])
    for near, link in ((True, False), (False, True)):                           # either the world's near or the ball's on
        assert _near_probe(near=near, link=link)[2] == 9 + 1 + K.CUE_CLEAR, (near, link)
    assert _near_probe(near=False, link=False)[2] == 10                          # nothing near: the box is no cue at the ball
    con, judg, asked = _near_probe(answer=True)                                 # both ways: after the gap, on the word, met
    assert [j[1] for j in judg] == ["met_ask"] and con.ledger.standing("ball")["asks"] == [1], judg
    # the Reader's geometry (A40): the ball 6 degrees from the duck, the cup 30 degrees, her face 53 degrees, as the child's head
    # sees them: Reader.near puts the ball and the duck in each other's near, nothing else; L1's glance at the duck holds back
    # "where is the ball?" 44 ticks and "where is the cup?" not at all
    from body.sim.lang.percept import Reader
    at = {"duck": (1.0, 0.0, 0.0), "ball": (math.cos(math.radians(6)), math.sin(math.radians(6)), 0.0),
          "cup": (math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0), "mama": (0.6, -0.8, 0.0)}
    nr = Reader.near(np.zeros(3), sorted(at.items()))
    assert nr == {"ball": ("duck",), "cup": (), "duck": ("ball",), "mama": ()}, nr
    assert Reader.near(np.zeros(3), sorted(at.items()), on=[("cup", "duck")])["cup"] == ("duck",)
    room = tuple(seen1(i, i, "", "mat", True, None, nr[i]) for i in ("duck", "ball", "cup"))
    con = _perfect(seed=4, stage=2, transcriber=Transcriber(None))
    _no_sets(con, ("duck", "ball", "cup"))
    said = {}
    for t in range(80):
        if t == 2:
            con.motion.glance("duck", 2, 8)
        for o in ("ball", "cup"):
            if t >= 10 and o not in said:
                con.request("ask_where", o=o)
        s = con.tick(t, P(t, child_target=None, seen=room, face_near=nr["mama"]))
        if s.line is not None and s.line.intent == "ask_where":
            said[s.line.refs[0]] = t
    assert said == {"cup": 10, "ball": 9 + 1 + K.CUE_CLEAR}, said
    # the measure behind NEAR_DEG, with the Reader itself: a head resting on the duck is read as a toy 6 degrees off for a met
    # within 20 ticks often, as one at NEAR_DEG + 1 in none of 1,500 looks over 40 ticks
    six, beyond = _reader_met(6.0, K.JUDGE_GAZE, 400), _reader_met(K.NEAR_DEG + 1, K.JUDGE_ACT, 1500, seed0=1000)
    assert six >= 40 and beyond == 0, (six, beyond)
    # fail-closed: a thing whose near the world does not give stands for every thing (a glance at it holds back the cup's ask
    # too); her face's likewise (face_near None: the mutual gaze every ask is made in could be read as any thing, so no ask
    # opens); a toy beside her face (the bottle) holds back asks about it while it is there and 44 ticks after, not others
    room = seen(("ball", "ball", "red", "mat", True), ("cup", "cup", "green", "mat", True)) + \
        (seen1("duck", "duck", "yellow", "mat", True, None, None), seen1("bottle", "bottle", "", "mat", True))

    def asks(glance=None, face=lambda t: (), n=120):
        con = _perfect(seed=4, stage=2, transcriber=Transcriber(None))
        _no_sets(con, ("duck", "ball", "cup", "bottle"))
        said = {}
        for t in range(n):
            if glance is not None and t == 2:
                con.motion.glance(glance, 2, 8)
            for o in ("cup", "bottle"):
                if t >= 10 and o not in said:
                    con.request("ask_where", o=o)
            s = con.tick(t, P(t, child_target=None, seen=room, face_near=face(t)))
            if s.line is not None and s.line.intent == "ask_where":
                said[s.line.refs[0]] = t
        return said, con
    said, _c = asks("duck")
    assert min(said.values()) == 9 + 1 + K.CUE_CLEAR, said
    said, con = asks(face=lambda t: None)
    assert said == {} and any("a place or a thing she cannot name" in r[2] for r in con.fast.refused), said
    said, _c = asks(face=lambda t: ("bottle",) if t < 30 else ())
    assert said.get("cup", 0) < 30 and said.get("bottle") == 29 + 1 + K.CUE_CLEAR, said
    # and a toy coming beside her face while an ask about it is pending voids it (her eyes on the child could be read as it)
    def run(path):
        con = _perfect(seed=4, stage=2, transcriber=Transcriber(None), ledger=Ledger(path))
        _no_sets(con, ("duck", "ball", "cup", "bottle"))
        con.request("ask_where", o="bottle")
        op = None
        for t in range(60):
            s = con.tick(t, P(t, child_target=None, seen=room,
                              face_near=("bottle",) if op is not None and t >= op + 3 else ()))
            if op is None and con.pending is not None:
                op = con.pending["open"]
        return op
    op, rows = _ledger_rows(run)
    assert op is not None and [r["result"] for r in rows] == ["void"] and "bottle" in rows[0]["why"], rows
    print(f"56 a cue at a thing near X is a cue at X (A51): the verifier's probe (the ball in the box, L1's glance at the box on "
          f"ticks 2-9): 'where is the ball?' said only at tick {9 + 1 + K.CUE_CLEAR} (b301745: at 10), and a child following "
          f"her glance 13-31 ticks after it, read as the ball, never meets it; the world's near or the ball's on alone holds it "
          f"back; with the Reader's geometry (the ball 6 degrees from the duck, the cup 30) her glance at the duck holds back "
          f"the "
          f"ball's ask, not the cup's; her Reader reads a head resting on the duck as a toy 6 degrees off for a met in {six} of "
          f"400 looks, as one {K.NEAR_DEG + 1:.0f} degrees off in {beyond} of 1500 over 40 ticks; a near the world does not "
          f"give holds back every ask, her face's too; a toy beside her face holds back asks about it, and voids one pending")


def _call_probe(stage=2, events=None, token=None, target=None, n=60, greet=False):
    """the call ("pip.", its name alone) at tick 0 (or, greet: after the greeting's lean-in at tick 0, when its log allows);
    events(t, con) and token(t, con) and target(t, con): the world's. -> ((con, lines, judgments, frowns, the call's tick),
    rows)."""
    def run(path):
        con = _perfect(seed=4, stage=stage, transcriber=Transcriber(None), ledger=Ledger(path))
        _no_sets(con)
        if greet:
            con.request("greet")
        lines, judg, frowns, called = [], [], [], None
        for t in range(n):
            if called is None and (not greet or t >= 1):
                con.request("call")
            ev = events(t, con) if events else ()
            tg = target(t, con) if target else "duck"
            s = con.tick(t, P(t, child_target=tg, events=ev), token=token(t, con) if token else None)
            if s.line is not None:
                lines.append((t, s.line.text, s.line.intent))
                if s.line.intent == "call" and called is None:
                    called = t
            judg += s.judgments
            frowns += [(t, s.frown)] if s.frown else []
        return con, lines, judg, frowns, called
    return _ledger_rows(run)


def test_log_call_and_her_face():
    """findings 3 and 4 (for the lead; the class, by the lead's rule that no pending ask ever sees X enter the log): while the
    call was pending the conduct allowed movements it counted as cues at her face (its own lean-in, a withdraw, an open hand):
    a child that turns to her face 1-15 ticks after the call's lean-in ends met the call at 15 of 15 latencies; and her face's
    changes were not in the log (a right name 3 ticks after the call's name earned her smile at tick 10 while the call was
    pending to 23; stage 2's frown at a hit the same). Now the call is said from where her face already rests, still (A3), with
    no lean-in of its own, and ends on its name; while it is pending blind() keeps her eyes on its eyes alone; her face (a
    judgment's smile, a frown, for FACE_COURSE ticks) and her voice (her words as they sound) are in her log, at her face: a
    judgment or a frown voids the call, she says no line while it is pending (the reply she owes waits for its judgment), and
    the child's pain or its hit voids it before her answer."""
    # the verifier's probe: a child that looks at her face 1-15 ticks after any movement of her body ends (her lean-in among
    # them) never meets the call; the call's line is its name alone, and her body is still through it
    met, calls = [], []
    for lat in range(1, 16):
        def follow(t, con, lat=lat):
            mv = [u for u, e in ((e["t"], e) for e in con.attn) if [f for f in ("eyes", "head") if "child" not in e[f]] or
                  [f for f in ("left", "right") if e[f]] or [a for a in e["acts"] if (a[1], a[2]) != ("look", "child_eyes")]]
            return "mama" if mv and max(mv) + lat <= t < max(mv) + lat + 5 else "duck"
        (con, lines, judg, _f, called), _r = _call_probe(target=follow)
        calls.append([ln for ln in lines if ln[2] == "call"])
        if [j for j in judg if j[1] == "met_ask"] or 1 in con.ledger.standing(LX.NAME)["asks"]:
            met.append((lat, lines[:3]))
    assert not met, f"a child that only follows her movements met the call: {met[:3]}"
    assert all(c == [(0, "pip.", "call")] for c in calls), calls[:2]
    assert C.INTENTS["call"].acts == (C.EYES,), C.INTENTS["call"]
    acts = (C.Act("lean_in", "child_periphery"), C.Act("open_hand", "child"), C.Act("withdraw", "child"), C.EYES)
    assert C.blind(acts, LX.PARENT_NAME) == (C.EYES,) and C.blind(acts, "ball") == (C.Act("lean_in", "child_periphery"),
                                                                                    C.Act("open_hand", "child"),
                                                                                    C.Act("withdraw", "child"), C.EYES)
    # a right name by token 3 ticks after the call's name is heard: her smile voids the call (worth 2 for the name, no met)
    (con, lines, judg, _f, called), rows = _call_probe(token=lambda t, c: LX.WORD_ID["duck"] if t == 5 else None,
                                                      target=lambda t, c: "mama" if 10 <= t < 16 else "duck")
    assert judg == [(2, "right_name", "duck")] and con.ledger.standing(LX.NAME)["asks"] == [], (judg, rows)
    assert [r["result"] for r in rows] == ["void"] and "her face" in rows[0]["why"] and "at mama" in rows[0]["why"], rows
    assert con.face_until == 5 + K.FACE_COURSE, con.face_until
    # stage 2's hit while the call is pending: the frown voids it, then "no."
    (con, lines, judg, frowns, called), rows = _call_probe(events=lambda t, c: (("hit_her", None),) if t == 5 else (),
                                                          target=lambda t, c: "mama" if 10 <= t < 16 else "duck")
    assert frowns == [(5, "hit")] and [r["result"] for r in rows] == ["void"] and not judg, (frowns, rows, judg)
    assert [ln[1] for ln in lines if ln[0] >= 5][:1] == ["no."], lines
    # a word that earns nothing (the ball, where it looks at the duck) at tick 5: her reply waits for the call's judgment (her
    # voice is at her face): no line between the call and its window's end
    (con, lines, judg, _f, called), rows = _call_probe(stage=1, token=lambda t, c: LX.WORD_ID["ball"] if t == 5 else None)
    until = 2 + K.JUDGE_GAZE
    assert [r["result"] for r in rows] == ["missed"] and [ln for ln in lines if 0 < ln[0] < until] == [] and \
        [ln[0] for ln in lines if ln[0] >= until][:1] == [until], (lines, rows)
    # its pain while the call is pending: void, then comfort
    (con, lines, judg, _f, called), rows = _call_probe(events=lambda t, c: (("pain", None),) if t == 6 else ())
    assert [r["result"] for r in rows] == ["void"] and [ln[2] for ln in lines if ln[0] == 6] == ["comfort"], (lines, rows)
    # both ways: a child that looks at her once its name is heard meets the call
    (con, lines, judg, _f, called), rows = _call_probe(target=lambda t, c: "mama" if 4 <= t < 10 else "duck")
    assert [j[1] for j in judg] == ["met_ask"] and con.ledger.standing(LX.NAME)["asks"] == [1], (judg, rows)
    print(f"57 the call and her face (A51, A3): the call is its name alone, said from where her face rests, no lean-in; while it "
          f"is pending only her eyes on its eyes; a child that looks at her 1-15 ticks after any movement of hers never meets "
          f"it (b301745: 15 of 15 after its lean-in); a right name while it is pending earns her smile and voids it (her face "
          f"in her log for {K.FACE_COURSE} ticks; b301745 kept it pending), a hit's frown likewise; the reply she owes waits for "
          f"its judgment (her voice at her face); its pain voids it before her comfort; a look at her once its name is heard "
          f"meets it")


def test_log_low_items_eighth():
    """finding 5 (low): an older save without `born` failed open: load_state left it None and the next tick set it to that
    tick, so no tick before the restore was checked (the ball in her log through tick 29, the save restored at 30: "where is
    the ball?" said at 31, where a current save says it at 74). Now a save without it counts every tick before its log as
    unread (fail-closed), and a log from before P3's eighth round (one word a field, no face or voice) restores at every thing.
    And the notes: a copy's hand points where it physically points (a raised arm or a shaken hand at a place, at every thing; a
    wave or an open hand toward the child); StubMotion reports her face."""
    def life(strip=None, objs=("ball",), n=120):
        con = _perfect(seed=4, transcriber=Transcriber(None))
        _no_sets(con)
        con.motion.glance("ball", 20, 10)                                       # L1's gaze at the ball, ticks 20-29
        for t in range(30):
            con.tick(t, P(t, child_target="duck"))
        st = json.loads(json.dumps(con.state()))
        if strip:
            strip(st)
        con2 = _perfect(seed=4, transcriber=Transcriber(None))
        con2.load_state(st)
        said = {}
        for t in range(30, n):
            for o in objs:
                if t >= 31 and o not in said:
                    con2.request("ask_where", o=o)
            s = con2.tick(t, P(t, child_target="duck"))
            if s.line is not None and s.line.intent == "ask_where":
                said[s.line.refs[0]] = t
        return said
    said = life(lambda st: st.pop("born"))
    assert said["ball"] == 29 + 1 + K.CUE_CLEAR, f"an older save without 'born' failed open: {said}"
    assert life()["ball"] == 29 + 1 + K.CUE_CLEAR, life()                     # as a current save
    assert life(objs=("cup",))["cup"] == 31 and life(lambda st: st.pop("born"), objs=("cup",))["cup"] == K.CUE_CLEAR, \
        "(its birth unknown, the ticks before its first logged tick are unread: an ask about the cup waits for 44 of them)"

    def old(st):                                                              # the seventh round's log: one word a field
        for e in st["attn"]:
            for f in C.LOG_FIELDS:
                v = e.pop(f, None)
                if f in C.FIELDS[:4]:
                    e[f] = None if not v else ("child" if "child" in v else v[0])
    said = life(old, objs=("cup",))
    assert said["cup"] == 29 + 1 + K.CUE_CLEAR, said
    assert C.directs(C.Act("copy", "arm_raise:right")) == [("right", C.UNNAMED)] and \
        C.directs(C.Act("copy", "shake:left")) == [("left", C.UNNAMED)] and \
        C.directs(C.Act("copy", "wave:left")) == [("left", "child")] and \
        C.directs(C.Act("copy", "open_hand:right")) == [("right", "child")]
    mo = C.StubMotion()
    assert mo.report(0)["face"] is None
    print(f"58 the low items (A51): an older save without 'born' counts every tick before its log unread: 'where is the ball?' "
          f"at {29 + 1 + K.CUE_CLEAR}, as with a current save (b301745: 31); a log of one word a field restores at every thing "
          f"(its near things and her face unknown); a copy's raised arm or shaken hand at a place, a wave or an open hand at the "
          f"child; the stub reports her face")


def _have_engine():
    return sys.platform == "darwin" and shutil.which("swiftc") is not None


def test_ear_templates_exact():
    if not _have_engine():
        print("15 skipped: no swiftc / AVSpeech here")
        return
    tmp = tempfile.mkdtemp()
    try:
        words = ("ball", "duck", "mama", "hi", "see", "up")
        digs = []
        for run_ in range(2):
            cache = V.VoiceCache(os.path.join(tmp, f"v{run_}"), server=V.SynthServer(nice=0))
            tm = {w: [] for w in words}
            for lab, voice, pitch, rate in K.EAR_VOICES:
                for w in words:
                    c = cache.clip(w, "ear", voice=voice, prosody=(pitch, rate))
                    tm[w].append((lab, PE.template_of(c.pa())))
            cache.close()
            ear = PE.ParentEar(tm, [PE.template_of(say("hi", 0.3))], K.EAR_M)
            digs.append(ear.digest)
            led = [json.loads(x) for x in open(os.path.join(tmp, f"v{run_}", "ledger.jsonl"))]
            assert len(led) == len(words) * len(K.EAR_VOICES) and {r["register"] for r in led} == {"ear"}
        assert digs[0] == digs[1], "the ear's templates differ between two builds"
        own = {w: [] for w in words}
        for w in words:                                                  # her own voice's templates, the rest held out
            own[w] = [x for x in tm[w] if x[0].startswith("parent")]
        hits = 0
        cache = V.VoiceCache(os.path.join(tmp, "v0"), server=V.SynthServer(nice=0))
        e_own = PE.ParentEar(own, [PE.template_of(say("hi", 0.3))], {k: 50.0 for k in range(1, 9)})
        for w in words:
            c = cache.clip(w, "ear", voice=K.EAR_VOICES[2][1], prosody=K.EAR_VOICES[2][2:])
            h = e_own.hear(c.pa(), words, words)
            hits += h.nearest == w
        cache.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"15 her templates ({len(words)} words x {len(K.EAR_VOICES)} voices) built twice through two voice caches: the same "
          f"digest, every clip in its ledger; her own voice's templates find the old child pitch's words {hits} of {len(words)}")


TESTS = [test_frames_and_birth_lines, test_line_check_refuses, test_compose_from_percept, test_variation_sets_and_repeats,
         test_replies_and_judgments, test_talk_over_and_turns, test_new_word_and_night, test_steer, test_transcriber,
         test_ear_rules, test_transcriber_with_ear, test_ledger_standing, test_replay_exact, test_cost, test_ear_templates_exact,
         test_ear_exact_across_threads, test_approximation_in_context, test_asks_answered, test_cut_words_not_said,
         test_new_words_in_a_day, test_acts_carry_the_object, test_line_check_truth, test_small_items,
         test_replay_across_processes, test_claude_never_asks, test_name_ask_answered_after_its_question,
         test_calls_out_of_sight, test_introduce_only_what_she_can_show, test_give_before_its_word, test_ear_deliberate_writes,
         test_gaze_leak_closed, test_new_word_on_its_peak, test_reads_trunk_and_hands, test_held_pairs_a55, test_echoes,
         test_claude_shows_and_forms, test_imperfect_parent_and_talk_over, test_sets_distinct_in_words, test_ask_after_her_cue,
         test_asks_she_can_judge, test_reading_held_three_ticks, test_low_items_fourth, test_cue_ends_with_her_act,
         test_cued_refuses_what_she_cannot_name, test_low_items_fifth, test_log_l1_gaze_under_way,
         test_log_demonstration_names_its_toy, test_log_queued_status_fails_closed, test_log_copy_before_an_ask,
         test_log_property, test_log_place_holding_x, test_log_acts_she_did_not_ask_for, test_log_low_items_seventh,
         test_log_her_face, test_log_met_give, test_log_near_things, test_log_call_and_her_face, test_log_low_items_eighth]

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
            import traceback
            traceback.print_exc()
            print("ERROR", t.__name__, ":", type(e).__name__, str(e)[:300])
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if failed else 0)
