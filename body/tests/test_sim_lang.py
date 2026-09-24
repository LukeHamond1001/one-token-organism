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
ear (in 10). Test 24 replays the whole speech side across processes from snapshots taken mid-line and mid-turn."""
import json
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


def seen(*objs):
    """Seen objects: (id, name, colour, on, child_sees)."""
    return tuple(Seen(*o) for o in objs)


TOYS = seen(("duck", "duck", "yellow", "mat", True), ("ball", "ball", "red", "mat", True), ("cup", "cup", "green", "mat", True),
            ("block", "block", "blue", "sofa", False), ("block_red", "block", "red", "mat", True))


def P(t, **kw):
    kw.setdefault("seen", TOYS)
    kw.setdefault("fixtures", ROOMS)
    return Percept(t, **kw)


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
            ("the block is red!", dict(vocab=VOCAB + ("red",), refs=("block_red",)))]
    for text, kw in must:
        kw = dict(kw)
        vocab = kw.pop("vocab", VOCAB)
        kw.setdefault("percept", p)
        assert ok(text, vocab, **kw), (text, TP.check(text, vocab, **kw))
    must_not = [
        ("look at the duck and the ball.", {}, "too long"), ("look, the duck.", {}, "form"), ("Look at the duck.", {}, "form"),
        ("look at  the duck.", {}, "form"), ("look at the duck", {}, "form"), ("? the duck.", {}, "form"),
        ("the rattle.", {}, "not her word"), ("a rattle is here.", dict(new_word="rattle"), "not last"),
        ("the red ball.", dict(vocab=VOCAB + ("red",)), "held-out pair"),
        ("it is red.", dict(vocab=VOCAB + ("red",), refs=("ball",)), "held-out pair (red, ball) by its referent"),
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
    assert ok("yes. the duck!", VOCAB, percept=p) and ok("the red ball.", VOCAB + ("red",), percept=p, held=())
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
    assert first and first[0][0] == 5 + K.TARGET_TICKS - 1, f"the follow-in set began at {first[0][0] if first else None}"
    s0 = [x for x in first if x[0] < 5 + 60]
    assert 2 <= len(s0) <= 3 and len({ln.text for _, ln in s0}) == len(s0), s0
    for (ta, la), (tb, _) in zip(s0, s0[1:]):
        assert tb - (ta + 3 * len(TP.words(la.text))) == K.PAUSE_RELATED, (ta, tb, la.text)
    set_starts = [first[0][0]] + [t for (ta, _), (t, _) in zip(first, first[1:]) if t - ta > 40]
    assert all(b - a >= K.SET_PER_OBJECT for a, b in zip(set_starts, set_starts[1:])), set_starts
    texts = [(t, ln.text) for t, ln in said]
    for i, (t, x) in enumerate(texts):
        for t2, x2 in texts[i + 1:]:
            assert x2 != x or t2 - t >= K.SAME_LINE, (x, t, t2)
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
    print(f"4 a follow-in set of {len(s0)} lines on the duck 3 ticks after the child's gaze settled, frames distinct, each 6 ticks "
          f"after the last ended; sets on one object {K.SET_PER_OBJECT}+ ticks apart ({len(set_starts)} in 400 ticks); no line "
          f"again within 60 ticks; the call refused at tick 100 (under 240 after the last) and said at 250")


def test_replies_and_judgments():
    """the token output (the scaffold) drives the child's words here: exact and approximate words, stage 1 and 2, the echo."""
    def one(stage, sym_seq, target="duck", said_duck_at=None):
        con = C.Conduct(seed=3, transcriber=Transcriber(None), stage=stage)
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
    con = C.Conduct(seed=2, transcriber=Transcriber(None))
    con.request("call")
    cut_at, lines, heard = None, [], []
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
    print(f"6 the child sounding during her call: the stop at tick {cut_at} (the world's playback finishes her word); no line "
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
          "'says'; 'bal' with the ball in its fovea or hand: a recast and a smile of 1; 'duk' or 'duck' when she only expects "
          "it as an echo of her last line: echoed, no smile (the approximation never looser than the exact word); the tract's "
          "approximation likewise; 'mama' at her face: a right name")


def test_asks_answered():
    """finding 2: an ask met without an answer (a twin already in the fovea; the call while it already looks at her; a look
    before the word was heard)."""
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
    # asked of the red ball while the blue one is in its fovea: not asked (it would be met unasked)
    con, said, judg = ask_run(lambda t, c: "ball_blue")
    assert not [ln for _, ln in said if ln.intent == "ask_where"] and judg == [], (said, judg)
    assert any("already in the child's fovea" in r[2] for r in con.fast.refused), con.fast.refused[-3:]
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
    print("18 an ask of the red ball with the blue one in its fovea is not made (logged); a look landing on a ball before its "
          "word is heard voids the ask (no smile, not counted); a look after it: met, a smile of 2, counted; the call is "
          "not made while it looks at her face, a turn during its name is void, a turn after it is met; a spoken word never "
          "meets a gaze ask")


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
    """finding 4: a second new word in one day erased the first ("rattle" then "ring": only "ring" joined at the night)."""
    toys = TOYS + seen(("rattle", "rattle", "purple", "mat", True), ("ring", "ring", "green", "mat", True),
                       ("stacker", "stacker", "orange", "mat", True), ("book", "book", "white", "mat", True))
    con = C.Conduct(seed=5)
    said = []
    for t in range(400):
        if t in (0, 100, 200, 300):
            con.request("new_word", word=("rattle", "ring", "stacker", "book")[t // 100], o=("rattle", "ring", "stacker",
                                                                                             "book")[t // 100])
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
    """finding 5: the motion interface dropped the object of a line with no object slot ("what is this?", "a rattle.")."""
    toys = TOYS + seen(("rattle", "rattle", "purple", "mat", True))
    con = C.Conduct(seed=6)
    con.request("ask_what", o="cup")
    s = None
    for t in range(3):
        s = con.tick(t, P(t, seen=toys))
        if s.line is not None:
            break
    assert s.line.intent == "ask_what" and C.Act("show", "cup") in s.acts, (s.line, s.acts)
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
    print("21 the motion interface is asked to show: 'what is this?' shows the cup it asks about; each of a new toy's 3 "
          "lines shows the toy (A15); a new fixture is pointed at, a verb done, a colour shown on its toy; an ask for a gaze "
          "never shows or points at its answer")


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
    assert f.add_steer("you see the rattle.", "any", 0)[0]
    ln = f.steer_line(10, P(10, seen=toys))
    assert ln.register == "new_word" and ln.emphasis == "rattle" and ln.focus == "rattle", ln
    ln2 = C.FastLayer(1)
    ln2.add_steer("you see your cup.", "any", 0)
    got = ln2.steer_line(10, P(10, child_holds=("cup",)))
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
          "read when it ends; her expected set is A27's (names in fovea and hand, the ask, her last focus, the routine; 'mama' "
          "when away), kept to her words")


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
    # says: 3 times over 2 days, per channel, never an echo, the referent in fovea or hand
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
          "channel, the echo and the referent out of its fovea not counted, never within one day")


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
        percepts.append(P(t, child_target=tg, child_holds=holds, events=ev, seen=toys))
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
                               [(a.kind, a.target) for a in s.acts]]))
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
         test_replay_across_processes]

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
