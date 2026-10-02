"""the parent's language (docs/SIM_DESIGN.md 4.3-4.10, 12, A13-A15, A26-A28, A51; package P3): the fast layer's templates and the
line check (body/sim/lang/templates.py), her intents and acts, her voice's manners, the speech side of L2 and her formal trials
(conduct.py), her ear (body/sim/parent_ear.py), the transcriber of the child's two outputs (transcriber.py) and the ledger
(ledger.py).
Run: python3 -m body.tests.test_sim_lang

Each rule is tested both ways: what must hold, and what must NOT hold (a line naming what she cannot see, a held-out pair by its
words or by its referent, a new word not last, Claude's praise; her ear accepting a word she did not expect, a word heard in place
of another called exact, her ear changing with what she hears or being written to, a word counted as said when it was an echo or
within one day, "understood" by her everyday asks or at a word's own chance, a trial scored with her body moving, a replay that
differs from the ledger). The ear's rules are tested on an
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
two lines of the same words, and the colour words short of a set where A55 puts them (38, 4.5); asks about a toy in her hands or
beyond the child's reach (40); her reading of its head's line held 3 ticks only for follow-in naming (41, 4.10, A40); the low
items (42). Test 45 is the fifth findings' low items (c816b5a).
Tests 39, 43, 44 and 46-58 are gone with the machinery they tested: the lead's decision after P3's eighth round ended the rounds
that tried to make every everyday ask leak-free (one attention log gating each ask by the 44 ticks before it, what lay near its
X, her face and voice, and voiding it by her body while pending). Understanding is scored only in formal trials, as infant labs
score it (Golinkoff et al. 1987; Fernald et al. 2008; Bergelson and Swingley 2012; Mandel, Jusczyk and Pisoni 1995), and her
everyday asks are her teaching, counted toward nothing (4.8, 12). Tests 59-64 test it, each failing on e2a1df4 (and on its merge
with main, 62cfb68) at its own first assertion: the protocol and its stillness enforced (59: she brings the two things, settles,
holds still but her mouth, says the single test sentence with no act, the window from the target word's onset, the first look;
each breach of her motion's contract after the sentence voids it, in the settle it holds the sentence back); chance by
counterbalancing (60: the named thing and the sides drawn by her trial stream, so followers of her gaze and hands, a side's and a
toy's favourite and a wanderer score 50%, a knower far above; a favourite's word never understood, its yoked rate its chance; a
motion that knew the target and glanced at it after the sentence voided every time, met by its follower with the rule off);
the name test against a foil name in the same voice (61); everyday asks teaching only (62: never "understood", an answer never
"says"); the property over 49 random lives (63: no trial scored with anything but her mouth moving by the motion's own account,
understood exactly as its trials give, never by asks; determinism whole and from snapshots in each phase of a trial); the
protocol's edges (64). Tests 12, 19 and 40 moved off the base-rate trials (a word's chance is now its trials' own), test 24's
replay carries a place trial and a name trial with snapshots in their settle and window, and test 21 reads her method, not the
log.
Tests 65-67 are P3's ninth verifier's findings on the formal trials (ae8280c), each failing there at its own first assertion: the
rule that turns trial outcomes into "understood" (65: a yoked or foil rate counted over all time and treated as known let a
voice-turner whose turning rose have its name understood in 94% of lives; now 4.8's (i) and (ii) on one stretch of trials, and
every one of 16 kinds of child whose look does not depend on the word reaches it at most as often as a fair coin reaches (ii),
at one fixed look (ii) rejects at its size, section 12's pooled test at its 0.01, the M6 exemplar test testable, knowers pass,
through her conduct too), the name's foils matched to it in her voice (66), and the low items (67: any word said while an ask
or a trial of hers is open never "says", her trial's line with no refs, a dropped display a presentation, her trunk in the
log). Tests 12 and 59-64 follow the rule: 12's cases in turn as her stream draws them, 60, 61 and 63 hold "understood" to the
rule written again with scipy's binomial tail (_rule_scipy, never the ledger's code), 59 and 63 add her trunk to the breaches.
Test 68 is P3's tenth verifier's blocking finding (e03c662), failing there at its own first assertion: two void rules followed the
named sentence's own timing in her voice (the talk-over's cut before its word, stage 2's talk-over frown), so a child that knew no
word, sounding at a fixed time from her voice, had every "block" trial voided and "ball" understood. Now each trial runs on one
timeline for every sentence its draw could have given (its window from the latest onset, her voice held to the latest end), and
nothing the child's voice does voids it: with her draw turned over every trial ends the same with the other thing named (430
children and trials; there, 87 did not), in her real voice too where the engine is present. Test 63 now says her trial sentences
as long as her voice makes them (TIMED, one onset a tick later, a foil a tick longer) to children that babble and take turns
after her sound stops, holds every trial to its one timeline, and runs each non-knower's trial again from its settle with the
other thing named (on e03c662, 78 of 613 ended otherwise); test 64's child talking over the sentence no longer voids it, a
combination's first 3 trials are its tests whichever is named, and the ninth round's save keeps no "understood".
Test 69 is P3's eleventh verifier's blocking finding (1387a44) and the lead's decision on it (C67): the two test sentences still
ended on different ticks in her voice, so a child that knew no word, hitting her 2 ticks after her sound stopped, had one word's
trials voided and the other's met ("block" understood in every life), and one looking 14 ticks after her sound stopped had one
word's trials met and the other's none. Now every sentence a trial's draw could give is made on one identical timeline, as infant
labs match their stimuli (body/sim/lang/stimuli.py, trial_lines.json from tools/sim_voice_check.py --trial: one carrier a form,
its test word at the engine's own per-word rate and a matched pitch contour), checked on the rendered audio, the words channel
with it while the scaffold labels her lines; a set that is not one timeline is never used. 69 turns her draw over for children
timed from 12 anchors of what they hear and see (her voice's start, her sound's stop and her mouth's closing at three levels, her
clip's end, the test word's onset and end, the channel's END, the window's close) at every offset from -4 to +30, with looks, reaches, sounds, tokens, hits, pain
and distress, in the timed voice and her real voice, and over 54 lives; it fails on 1387a44 at its own first assertion. Tests 63,
64 and 68 follow: the timed voice (TIMED) says her sentences as the engine now makes them (the eleventh round's, TIMED_OLD, is
refused pair by pair), 63's lives run the words channel in two of three (the name then never tried) and pairs with "duck" (no
time-matched stimulus: never tried), 64's save from before the stimuli is void in its window and dropped in its settle.
Test 70 is P3's twelfth verifier's finding (f7cde4c): every trial scored by intention to treat. Test 71 is its thirteenth
verifier's (13fb5dc): with one recording a word, where a sentence stopped as the child perceives it was the same against every
other word, so a child that knew no word, with a favourite toy, one look at a fixed tick and one timed from her sound's stop, had
the order of its looks follow the word ("drum" understood in 12 of 12 lives, "pip" by a gated turn). Now every sentence of a form
is one carrier phrase, its test word between one recording before it and one tag after it at one level under both (stimuli.parts,
splice, headroom), so her voice's start and her sound's stop are one moment at every level, however perceived: 71 checks it on
the rendered audio, turns her draw over for the verifier's kinds of child, runs its children over lives, and fails on 13fb5dc at
its own first assertion; 1, 61, 63, 64 and 68-70 follow (the trial frames' tag, the name's "hi. X. hi.", the timed voice 2 dB
softer after its first word, 63 over 49 lives, the onset's reading the tick before the onset's).
Tests 70 and 72 are the lead's decision A60b on P3's fourteenth verifier's finding (f5ed8cd: band by band at the child's own ears
her voice's start and her sound's stop fall inside the test word's slot and differ by word, so children timed from them passed):
understanding is scored by the proportion of looking over a fixed window, the test word's onset tick + 2 to + 23, the same
whichever is named (a pair's trial with under 4 of its ticks on either thing void, counted, never scored), and "understood" by a
permutation test over the draw labels, its tests at a word's 12th, 24th, 48th, ... registered trial at 0.005, 0.0025, ..., 0.01 a
life (ledger.perm_test; section 12's form by the flip test, ledger.flip_test). 70 (in place of intention to treat's) tests the
window's share: only how many of its ticks, never when or in what order; the void rule; every child timed from any function of
the sentence it hears alike with the draw turned over; a knower not. 72 is the lead's acceptance: the zoo of children that know
no word over 12 lives each, the knowers detected, the flip, and the determinism across processes and threads; disclosed, the
fourteenth verifier's children keyed on one band of the child's own ear (C69b). Each fails on f5ed8cd at its own first assertion.
Tests 12, 59-61, 63-65, 68, 69 and 71 follow: 12's ledger records the window's ticks and holds the permutation test to brute
force; 59 the window and its judgment at its end; 60, 61 and 63 count the trials whose looking went more to the named thing and
hold "understood" to the rule as stated (_rule_share); 64's child talking over the sentence scored; 65 holds the permutation test
to its level at one fixed test and a life's tests to 0.01 over 15 kinds of child; 68, 69 and 71 compare the window's ticks on
each thing with the draw turned over (_same_but_named, _inv_same); 71's lives at 0.01 a life, the onset's own tick outside the
window.
Test 73 is the lead's two further decisions on A60b (after 535e32b): no feedback in a trial (her smile and confirm after it
removed: a rewarded trial is trainable by construction) and two levels, "maps" (the word's record, its trained thing) and
"understood" (an object noun's new-exemplar block passed too: "exemplar" trials of new exemplars of its kind never named before
the probe, registered apart): a knower of the kind passes both, a knower of its trained things alone, a favourite and the child
keyed by one band of its ear to the drum fail level 2; no judgment follows a trial. Since the lead's three decisions after
200e57a it also holds level 2's control to a never-told foil (each new exemplar set down beside two things of two other words,
its word or a foil of its written syllables said by her stream's fair coin; consts.WORD_FOILS, measured in her voice by
tools/sim_voice_check.py --trial-foils): a child that knows only the other words, excluding at any word it does not know, maps
and is never understood; a knower of the kind is understood (at 12 trials when its looks at a foil spread over the three, at 24
when its look goes to one of them); disclosed, what the foil leaves open (exclusion gated on how often a word was heard, a novelty
cue beside familiar things, closed by things as new as the exemplar); the foils never her words; a probe of two words, with a
foil of hers or under the scaffold dropped; A53's exemplars at least 4 a noun. It fails on 200e57a at its own first
assertion. 12, 59, 60, 63-65 and 70-72 follow: level 1 read as "maps", "understood" at both levels, the score entries' level,
72's band learners with no smile to learn from, and its band-gated children at level 2; 12 and 65 hold level 2 to the foil (its
new exemplar the target whichever is said; a child that knows only the other words never understood there; the M6 exemplar
test by the permutation test, its words against their foils). Since the lead's three decisions after 67741fd, 73 holds level 2
to a block of 24 registered trials (its first test at its 24th; A53's exemplars at least 8 a noun), each noun's own foil
matched in exposure (her chatter carries it at its noun's running rate, only while the child attends and holds nothing, each
trial logging both counts; consts.NOUN_FOILS, measured in her voice) and its three things matched in newness (drawn by the
conduct from the room's new exemplars, each noun's pool serving both roles): the frequency-gated excluder and the knower of its
trained things that looks at the newest thing, understood in 10 of 12 lives on 67741fd, now in 0; a knower of the kind still
understood at 24 trials, including one whose look at a foil goes to one thing; disclosed, an excluder gated on the trial
sentence's familiarity. It fails on 67741fd at its own first assertion; 12, 65 and 72 follow (the block at 24, a save of
67741fd's ledger starting its blocks again, the M6 exemplar test with 8 exemplars a noun, the band-gated children's block)."""
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
from body.sim.lang.ledger import Ledger, LedgerDiverged  # noqa: E402
try:                                                     # the tests over the draw labels (the lead's decision A60b): absent
    from body.sim.lang.ledger import flip_test, perm_test  # noqa: E402   # before, where the tests that read them fail at
except ImportError:                                      # their own assertions
    flip_test = perm_test = None
try:                                                     # a trial's time-matched stimuli (P3's twelfth round): absent before,
    from body.sim.lang import stimuli as ST  # noqa: E402  # where the tests that read it fail at their own assertions
except ImportError:
    ST = None
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
            if focus is not None:                             # (a trial's test word: before its one tag, P3's fourteenth round)
                assert ws[-2 if k in TP.TRIAL_FRAMES else -1] == focus.strip("{}"), \
                    f"{k}: {text!r}'s focus {focus!r} is not its last word"
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
    print(f"1 every frame's focus is its last word (a trial's before its tag) and every line at most 6 words; {len(lines)} "
          f"birth lines "
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
    assert judg == [(K.WORTH_WORD, "word", "ball")], f"the control: an exact 'ball' with no ball in view: {judg} (C145: a word smile of {K.WORTH_WORD}, never a right name or an approximation)"
    con, said, judg, _ = _token_turn(2, _letters("bal"), target="ball")
    assert judg == [(1, "approximation", "ball")] and said[0][1].intent == "recast", (judg, said[:1])
    con, said, judg, _ = _token_turn(2, _letters("bal"), holds=("ball",))
    assert judg == [(1, "approximation", "ball")], judg
    # the ear's wider set (her last focus word) lets her hear it, never smile at it
    con, said, judg, _ = _token_turn(2, _letters("duk"), pre=lambda c: setattr(c.fast, "last_focus", "duck"))
    assert judg == [] and said[0][1].intent in ("echo", "echo_word"), (judg, said[:1])
    con, said, judg, _ = _token_turn(2, [LX.WORD_ID["duck"]], pre=lambda c: setattr(c.fast, "last_focus", "duck"))
    assert judg == [(K.WORTH_WORD, "word", "duck")], f"an exact word she only expected as an echo: {judg} (C145: a word smile, not a right name)"
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
          "'says'; 'bal' with the ball where she reads it looking or in its hand: a recast and a smile of 1; 'duk' when she only "
          "expects it as an echo of her last line: echoed, no smile (the approximation never looser than the exact word); the "
          "tract's approximation likewise; an exact 'ball' or 'duck' with nothing in view: echoed with a word smile of 1 (C145); "
          "'mama' at her face: a right name")


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
    counts). Now a word is said only as its sound ends."""
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
                                                                      for w, v in con.ledger.words.items()})
        return con, cuts, at9
    con, cuts, (said, focus, counts) = go(None, 3, n=40, tok={16: LX.WORD_ID["duck"]},
                                          target=lambda t: "duck" if t < 3 or t >= 14 else None)
    assert cuts == [3], cuts
    assert "duck" not in said and said.get("the") == 5, said
    assert "duck" not in counts and counts["the"]["said"] == 1, counts
    assert focus is None, focus
    st = con.ledger.words["duck"]
    assert st["echoes"]["token"] == 0 and len(st["says"]["token"]) == 1, \
        f"the child's own 'duck' after a cut 'duck' was taken as an echo: {st}"
    # with a voice: "duck" lengthened to 6 ticks is broken off at 3 ticks and never said; "the" finishing within 3 is said
    spans = {"look. the duck.": [("look", 0, 2400), ("the", 2400, 4800), ("duck", 4800, 4800 + 6 * 2400)]}
    con, cuts, (said, focus, counts) = go(_FakeVoice(spans), 3)
    assert cuts == [3] and "duck" not in said and "duck" not in counts and said.get("the") == 1, (said, counts)
    con, cuts, (said, focus, counts) = go(_FakeVoice(spans), 7)            # cut 2 ticks from "duck"'s end: finished
    assert said.get("duck") == 7 and focus == "duck" and counts["duck"]["heard"] == 1, (said, focus, counts)
    con, cuts, (said, focus, counts) = go(None, 30)                        # not cut: every word said as it ends
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
    P3's sixth round a name ask shows nothing, A51, her method: "what is this?" is asked of what the child attends.)"""
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
          "nothing shown (A51; not asked of a cup it does not attend); each of a new toy's 3 lines shows the toy (A15); a "
          "new fixture is pointed at, a verb done, a colour shown on its toy; an ask for a gaze never shows or points at its "
          "answer")


def test_the_hide_line():
    """lang 62 (A129, the hide game): "the duck is in the bucket." passes the check when the duck is seen lying in the bucket (the percept's
    `on` "bucket": the bucket is open, templates.OPEN_CONTAINERS, she sees into it), is "not true" of a duck on the mat, and "in the box"
    stays containment she cannot see; the hide intent's two frames end on their focus and carry the toy's word"""
    V2 = set(VOCAB) | {"bucket", "box"}                                     # the bucket's and the box's words in (the growth queue admits them)
    bucket = ("bucket", "bucket", "olive", "mat", True); box = ("box", "box", "grey", "mat", True)
    ok, reason = TP.check("the duck is in the bucket.", V2, None, P(0, seen=seen(("duck", "duck", "yellow", "bucket", True), bucket)), (), source="fast")
    assert ok, reason
    ok, reason = TP.check("the duck is in the bucket.", V2, None, P(0, seen=seen(("duck", "duck", "yellow", "mat", True), bucket)), (), source="fast")
    assert not ok and "not true" in reason, reason
    ok, reason = TP.check("the duck is in the box.", V2, None, P(0, seen=seen(("duck", "duck", "yellow", "box", True), box)), (), source="fast")
    assert not ok and "containment" in reason, reason
    fr = TP.FRAMES["hide_told"]
    assert [f[0] for f in fr] == ["the {o} is in the bucket.", "look. in the bucket. the {o}."] and "bucket" in TP.OPEN_CONTAINERS, fr
    assert [f[0] for f in TP.FRAMES["hide"]] == ["look. the {o}."], TP.FRAMES["hide"]   # C115: the bucket lines belong to the drop
    print("lang 62: 'the duck is in the bucket.' passes of a duck seen in the bucket, 'not true' of one on the mat; 'in the box' stays containment",
          "she cannot see; the drop's two frames (hide_told) end on their focus; the hide's own line is true as she takes the toy")


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
    exp = expected_words(p, dict(word="ball"), "mama", "peekaboo", VOCAB)
    assert exp == ("mama", "peekaboo", "ball", "duck", "cup"), exp          # the routine's words before the ask's and the things
    assert expected_words(P(0, present=False, seen=()), None, None, None, VOCAB) == ("mama",)
    assert "rattle" not in expected_words(P(0, child_holds=("r",), seen=(Seen("r", "rattle"),)), None, None, None, VOCAB)
    print("9 the token output read as a transcript: a token exactly, letters exactly, within edit distance 1 (2 at 6 letters), "
          "a 2-letter prefix only of what it sees or holds, and what it sees or holds first ('bo' holding a bottle is "
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
def _ledger_rows(led, key, rows, name=False, t0=5000, form="place", other="cup", others=None):
    """formal trials recorded straight into a ledger (the conduct's calls) in their order (the lead's decision A60b): rows
    [(named, a, b)], named 1 when key's word was said (the name test: its name; 0: a foil, or the other thing's word), a and b
    the window's ticks on key's thing (her face) and on the other thing (the window's other ticks); (named, None, why): a void.
    form "exemplar": its word's new-exemplar block (level 2; since the lead's decision after 200e57a against a never-told
    foil): its new exemplar the target whichever is said, a its ticks, b those on the two things of `others` beside it (a
    ledger from before that decision takes them as it did, the other thing's word in turn its control)."""
    t = t0
    level = 2 if form == "exemplar" else 1
    import inspect                                                        # noqa: PLC0415
    if form == "exemplar" and "others" in inspect.signature(led.trial).parameters:
        oth = list(others or ("cup", "duck"))
        for i, (named, a, b) in enumerate(rows):
            said = key if named else getattr(K, "NOUN_FOILS", {}).get(key, "zeb")
            tid = led.trial(t, form, dict(id=f"{key}_{t}", word=key), dict(id=oth[0], word=oth[0]), t + 5, (t + 7, t + 28),
                            said, score=[[key, "trials" if named else "yoked", 1, 0, 2]], items=[f"{key}@{t}"],
                            others=[dict(id=x, word=x) for x in oth[1:]])
            if a is None:
                led.trial_outcome(t + 28, tid, "void", b)
            else:
                led.trial_outcome(t + 28, tid, "scored", "a test", on=a, off=b)
            t += 40
        return
    for i, (named, a, b) in enumerate(rows):
        oth = others[i % len(others)] if others else other
        said = key if named else (K.NAME_FOILS[0] if name else oth)
        tgt = dict(id=None, word=said) if name else (dict(id=key, word=key) if named else dict(id=oth, word=oth))
        dis = None if name else (dict(id=oth, word=oth) if named else dict(id=key, word=key))
        tid = led.trial(t, "name" if name else form, tgt, dis, t + 5, (t + 7, t + 28), said,
                        score=[[key, "trials" if named else "yoked"] + ([1, 0] if named or name else [0, 1]) + [level]],
                        items=[] if name else [f"{key}@{t}"])
        if a is None:
            led.trial_outcome(t + 28, tid, "void", b)
        else:
            on, off = (a, b) if (name or named) else (b, a)
            led.trial_outcome(t + 28, tid, "scored", "a test", on=on, off=off)
        t += 40


def _ledger_knows(led, key, n=12, name=False, t0=5000, kind=False):
    """a knower's n trials straight into a ledger, its word said and not in turn: its window's looks all on its thing when named
    and on the other when not (its name: 12 ticks on her face after it, none after a foil); kind: as many more of its kind's
    new exemplars, its word or its never-told foil said, beside new exemplars of two other kinds (its second level, A60b: a
    block of 24)."""
    rows = [(1, 12, 10) if name else (1, 8, 0), (0, 0, 22) if name else (0, 0, 8)] * (n // 2)
    _ledger_rows(led, key, rows, name=name, t0=t0)
    if kind:                                              # (its block of 24: KIND_FIRST, since the lead's decision after
        _ledger_rows(led, key, [(1, 8, 0), (0, 0, 8)] * 12, t0=t0 + 100000, form="exemplar", others=("cup", "duck"))


def _brute_p(seq):
    """the permutation test's p written again by brute force (every choice of which k trials named it), never the ledger's
    code: seq [(named, a, b)] -> p."""
    import itertools                                                      # noqa: PLC0415
    from fractions import Fraction                                        # noqa: PLC0415
    rows = [(nm, a, b) for nm, a, b in seq if a + b > 0]
    n, k = len(rows), sum(r[0] for r in rows)
    q = [Fraction(a, a + b) for _nm, a, b in rows]
    s = sum(x for x, r in zip(q, rows) if r[0])
    return sum(sum(q[i] for i in c) >= s for c in itertools.combinations(range(n), k)) / math.comb(n, k)


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
    # understood: from formal trials only (the lead's decision), by the proportion of looking and a permutation test (A60b)
    assert led.understood("duck") is False
    for i in range(10):                                   # her everyday asks, all met: teaching, counted toward nothing
        t0 = 1000 + 40 * i
        led.ask(t0, "duck", "gaze", "duck", K.JUDGE_GAZE)
        for t in range(t0, t0 + 22):
            led.observe(t, P(t, child_target="duck" if t >= t0 + 3 else None))
    assert led.standing("duck")["asks"] == [1] * 10 and not led.understood("duck"), "understood by her everyday asks"
    assert hasattr(led, "trial") and perm_test is not None, "no permutation test over the draw labels in her ledger (A60b)"
    assert hasattr(led, "maps"), "one level of 'understood' only (A60b's two levels: maps and understood)"
    assert hasattr(K, "NOUN_FOILS") and hasattr(K, "KIND_FIRST"), \
        "level 2 as 67741fd built it: a block of 12, its foil not heard as often as its noun, beside familiar things (the " \
        "lead's decisions after 67741fd)"
    # a knower at its 12th trial, 6 naming it: the one relabeling as far of C(12, 6) = 924, p = 0.0011 under the first test's
    # 0.005; at its 11th, no test yet. Level 1, "maps": the word to its trained thing; an object noun is "understood" (level 2)
    # only once its new-exemplar block passes too (A60b)
    _ledger_rows(led, "duck", [(1, 8, 0), (0, 0, 8)] * 5 + [(1, 7, 1)])
    assert not led.maps("duck") and led.standing("duck")["tests"] == [], led.standing("duck")
    _ledger_rows(led, "duck", [(0, 1, 7)], t0=9000)
    st = led.standing("duck")
    assert led.maps("duck") and not led.understood("duck") and st["trials"][:2] == [[8, 0], [8, 0]] and \
        st["yoked"][0] == [0, 8] and len(st["tests"]) == 1 and abs(st["tests"][0][2] - 1 / 924) < 1e-12 and \
        st["tests"][0][3] == 0.005, st
    # its second level: 23 trials of its kind's new exemplars (its new exemplar's ticks after its word against after its
    # never-told foil, beside new exemplars of two other kinds: the lead's decisions after 200e57a and 67741fd), no test yet;
    # at its 24th (its block: KIND_FIRST), understood (p = 1/C(24, 12) at 0.005)
    _ledger_rows(led, "duck", ([(1, 8, 0), (0, 0, 8)] * 12)[:23], t0=20000, form="exemplar", others=("cup", "ball"))
    st2 = led.standing("duck")
    assert not led.understood("duck") and st2["tests2"] == [] and len(st2["seq2"]) == 23, st2
    _ledger_rows(led, "duck", [(0, 0, 8)], t0=30000, form="exemplar", others=("cup", "ball"))
    st2 = led.standing("duck")
    assert led.understood("duck") and st2["kind"] and [x[3] for x in st2["tests2"]] == [0.005] and \
        abs(st2["tests2"][0][2] - 1 / math.comb(24, 12)) < 1e-15 and {x[3] for x in st2["seq2"] if not x[0]} == {"zeb"}, st2
    for old_key in ("look-share-2", "look-share-3"):      # a save of 200e57a's ledger (level 2 against a known word) or of
        old = led.state()                                 # 67741fd's (a foil beside familiar things, not heard as often):
        old["scoring"] = old_key                          # its record (level 1) kept, its new-exemplar block started again
        lo = Ledger()
        lo.load_state(old)
        so = lo.standing("duck")
        assert lo.maps("duck") and not lo.understood("duck") and so["seq2"] == [] and so["tests2"] == [] and \
            len(so["seq"]) == 12 and "exemplar" not in lo.forms and "place" in lo.forms, (old_key, so)
    lk = Ledger()                                          # a child keyed to its trained thing: maps, and at chance on its
    _ledger_rows(lk, "duck", [(1, 8, 0), (0, 0, 8)] * 6)   # kind's new exemplars: never understood
    _ledger_rows(lk, "duck", [(1, 4, 4), (0, 4, 4)] * 12, t0=20000, form="exemplar", others=("cup", "ball"))
    assert lk.maps("duck") and not lk.understood("duck") and lk.test("duck", 2)["p"] == 1.0, lk.test("duck", 2)
    lx = Ledger()                                          # it knows only the other words: credited at level 1 (the yoked
    _ledger_rows(lx, "duck", [(1, 4, 4), (0, 0, 8)] * 6)   # comparison's own limit, disclosed); at level 2, turning from the
    _ledger_rows(lx, "duck", [(1, 8, 0), (0, 8, 0)] * 12, t0=20000, form="exemplar", others=("cup", "ball"))  # things it can
    assert lx.maps("duck") and not lx.understood("duck") and lx.test("duck", 2)["p"] == 1.0, lx.test("duck", 2)   # name at
    # its word and at a foil alike, never understood
    led2 = Ledger()                                       # a favourite: all its looking on the duck, whichever is said
    _ledger_rows(led2, "duck", [(1, 8, 0), (0, 8, 0)] * 6)
    t2 = led2.test("duck")
    assert not led2.maps("duck") and t2["diff"] == 0.0 and t2["p"] == 1.0, t2
    led2b = Ledger()                                      # its share on the duck grows, whichever is said
    _ledger_rows(led2b, "duck", [(j % 2, 2 + j // 2, 10 - j // 2) for j in range(12)])
    assert not led2b.maps("duck"), led2b.test("duck")
    led3 = Ledger()                                       # it knows only the other word: the yoked comparison's own limit,
    _ledger_rows(led3, "duck", [(1, 4, 4), (0, 0, 8)] * 6)    # disclosed (A60b): its share on the duck 1/2 when "duck" is
    assert led3.maps("duck"), led3.test("duck")          # said, 0 when "cup" is, so "duck" is credited too
    led3b = Ledger()                                      # voids never scored, counted: a scored pair with under 4 ticks on
    _ledger_rows(led3b, "duck", [(1, None, "no look"), (0, None, "its pain")] * 3 + [(1, 8, 0), (0, 0, 8)] * 6)
    sb = led3b.standing("duck")
    assert led3b.maps("duck") and len(sb["seq"]) == 12 and sb["voids"] == 6 and led3b.voided == {"place": 6}, sb
    try:                                                  # either thing is refused
        _ledger_rows(Ledger(), "duck", [(1, 2, 1)])
        refused = False
    except ValueError:
        refused = True
    assert refused, "a pair's trial scored with 3 of its window's ticks on either thing"
    led3e = Ledger()                                      # 2 of its 12 naming it: C(12, 2) = 66, its first test not testable at
    _ledger_rows(led3e, "duck", [(1, 8, 0), (0, 0, 8), (0, 0, 8), (0, 0, 8), (0, 0, 8), (0, 0, 8)] * 2)   # 0.005: its level
    assert not led3e.maps("duck") and led3e.standing("duck")["tests"][0][4] is False, led3e.standing("duck")   # unspent
    _ledger_rows(led3e, "duck", [(1, 8, 0), (0, 0, 8)] * 6, t0=20000)                     # at its 24th, 0.0025: maps
    assert led3e.maps("duck") and [x[3] for x in led3e.standing("duck")["tests"]] == [0.005, 0.0025]
    led3c = Ledger()                                      # the name: its face's share after its name against after a foil
    _ledger_knows(led3c, LX.NAME, 12, name=True)
    led3d = Ledger()                                      # a voice-turner: 12 of 22 ticks on her face after either
    _ledger_rows(led3d, LX.NAME, [(1, 12, 10), (0, 12, 10)] * 6, name=True)
    assert led3c.understood(LX.NAME) and led3c.maps(LX.NAME) and not led3d.understood(LX.NAME), \
        (led3c.test(LX.NAME), led3d.test(LX.NAME))                 # (the name has one level: understood as it maps)
    led3f = Ledger()                                      # a rate from other times is never its chance: a child that turned
    _ledger_rows(led3f, LX.NAME, [(1, 0, 22), (0, 0, 22)] * 20 + [(1, 14, 8), (0, 14, 8)] * 6, name=True)   # to no voice,
    assert not led3f.understood(LX.NAME) and led3f.test(LX.NAME)["diff"] == 0.0, led3f.test(LX.NAME)   # then to every one
    # the permutation test exact: against brute force, ties and all; the flip test likewise; the tests' levels spent
    r_ = np.random.default_rng(12)
    for _ in range(60):
        seq = [(int(r_.random() < 0.5), int(r_.integers(0, 9)), int(r_.integers(1, 9))) for _ in range(int(r_.integers(3, 13)))]
        if 0 < sum(s_[0] for s_ in seq) < len(seq):
            assert abs(perm_test(seq, 0.01)["p"] - _brute_p(seq)) < 1e-12, seq
    ft = flip_test([(8, 0)] * 7)
    assert ft["testable"] and abs(ft["p"] - 0.5 ** 7) < 1e-15 and not flip_test([(8, 0)] * 6)["testable"], ft
    from body.sim.lang.ledger import test_level                           # noqa: PLC0415
    lv = [test_level(n_) for n_ in range(1, 400)]
    assert [n_ for n_, x in enumerate(lv, 1) if x] == [12, 24, 48, 96, 192, 384] and sum(x for x in lv if x) < K.UNDERSTOOD_P
    lv2 = [test_level(n_, K.KIND_FIRST) for n_ in range(1, 400)]         # a new-exemplar block's: from its 24th
    assert [n_ for n_, x in enumerate(lv2, 1) if x] == [24, 48, 96, 192, 384] and \
        [x for x in lv2 if x][:2] == [0.005, 0.0025], lv2
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
    print("12 the ledger: 'heard' only with the referent in the child's view; understood only from formal trials, by the "
          "proportion of looking and a permutation test over the draw labels (A60b): never by 10 met everyday asks; a knower "
          "maps at its 12th trial, 6 naming it (p = 1/924 under the first test's 0.005), not at its 11th (no test due), and is "
          "understood (level 2) only once its kind's new exemplars pass too, its word against its never-told foil, at "
          "their 24th trial, not their 23rd (a save of 200e57a's or 67741fd's ledger keeps its level 1 and starts that block "
          "again); a child keyed to its trained thing, at chance on new exemplars, maps and is never understood, nor one "
          "that knows only the other words (on its new exemplar after its word and a foil alike); not for a "
          "favourite (the same share either way: p = 1), nor one whose share grows either way; a child that knows only the "
          "other word credited at level 1 (the yoked comparison's own limit, disclosed); 6 voids never scored and counted, a pair "
          "scored with 3 of its ticks on either thing refused; 2 of 12 naming it: its first test not testable, its level "
          "unspent, understood at its 24th at 0.0025; its name at 12 of 22 ticks on her face after it against none after a "
          "foil, not a voice-turner's 12 after either, nor a child that turned to no voice for 40 trials and then to every "
          "one; the permutation test exact against brute force over 60 records, the flip test's 2^-7; the tests at 12, 24, 48, "
          "..., a new-exemplar block's at 24, 48, ..., their levels summing under 0.01; 'says' 3 times over 2 days per "
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
                720: [("show", dict(o="duck"))], 800: [("call", {})], 850: [("show", dict(o="rattle"))],
                140: [("probe", dict(form="place", a="ball", b="duck", new=dict(ball="ball@x", duck="duck@y")))],
                640: [("probe", dict(form="name"))]}                      # her formal trials (4.8), where the conduct has them
    return dict(percepts=percepts, tract=tract, tokens=tokens, requests=requests, stage=stage)


def run_scenario(con, sc, t_from, t_to):
    log = []
    for t in range(t_from, t_to):
        for intent, kw in sc["requests"].get(t, ()):
            if intent == "probe":
                if hasattr(con, "probe"):
                    con.probe(**kw)
                continue
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
                ph = (getattr(con, "trial", None) or {}).get("phase")
                if ph in ("settle", "said") and not [x for x in snaps if x[1] == ph]:
                    snaps.append((t, ph, json.loads(json.dumps(con.state(), default=_np))))      # mid-trial
                elif (cur and len([x for x in snaps if x[1] == "line"]) < 2 and t > 90 + 200 * len(snaps)) or \
                        (utt and not [x for x in snaps if x[1] == "turn"] and t > 300):
                    snaps.append((t, "line" if cur else "turn", json.loads(json.dumps(con.state(), default=_np))))
                log += run_scenario(con, sc, t, t + 1)
            digest = con.ledger.digest
            summary.append((stage, len([x for x in log if json.loads(x)[1]]), sum(len(json.loads(x)[2]) for x in log),
                            sum(json.loads(x)[3] for x in log), len(snaps), [x[1] for x in snaps]))
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
    parts = [f"stage {a}: {b} lines, {c} judgments, {d} cuts, {e} snapshots ({', '.join(f_)})" for a, b, c, d, e, f_ in summary]
    print(f"24 {n} ticks of the whole speech side, stage 1 and 2 (lines, judgments, cuts, a place trial and a name trial, "
          f"snapshots): {'; '.join(parts)}; each snapshot (mid-line, mid-turn, in a trial's settle and window) restored in a "
          f"fresh one-thread process replays the rest exactly, ledger digest and every tick")


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
def test_motor_judgments():
    """A89 (the teacher's build 2b): her smiles for its acts. "got ball" earns 2 and her "yes!" at once; the n-th smile for the same
    act and object falls as 2 e^(-n/10) and stops under 0.05 (no floor: nothing is farmed); another toy starts afresh; a reach
    that ends nearer earns 1 only until "got" on that toy has been smiled at 3 times; a half roll until the whole roll has; a lift,
    a shake, a hit and its head up earn 1; nothing is judged while she is away; stage 2 frowns at a thrown toy ("no."), stage 1
    does not; the book and its log survive a save."""
    con = _perfect(seed=4, transcriber=Transcriber(None))
    _no_sets(con)
    s = con.tick(0, P(0, seen=TOYS, events=(("got", "ball"),)))
    assert s.judgments == [(2, "got", "ball")], s.judgments
    assert s.line is not None and s.line.intent in ("confirm", "confirm_act") and s.line.text.startswith(("yes", "good")), s.line
    yes = s.line.text
    ws = [2.0]
    for t in range(1, 80):
        s = con.tick(t, P(t, seen=TOYS, events=(("got", "ball"),)))
        ws += [j[0] for j in s.judgments if j[1] == "got"]
    assert abs(ws[1] - 2 * math.exp(-0.1)) < 1e-3 and all(b < a for a, b in zip(ws, ws[1:])) and 30 < len(ws) < 45, (len(ws), ws[:4])
    assert con.book["got"]["ball"] == len(ws) and ws[-1] >= K.HABIT_FLOOR
    assert [x for x in con.book_log if x[3] == "habituated"], con.book_log[-3:]
    s = con.tick(80, P(80, seen=TOYS, events=(("got", "duck"),)))
    assert s.judgments == [(2, "got", "duck")], s.judgments                      # another toy: afresh
    # shaping: a reach nearer the cup earns 1 until "got cup" is mastered (3 smiles)
    s = con.tick(81, P(81, seen=TOYS, events=(("reach_nearer", "cup"),)))
    assert s.judgments == [(1, "reach_nearer", "cup")], s.judgments
    for t in range(82, 85):
        con.tick(t, P(t, seen=TOYS, events=(("got", "cup"),)))
    s = con.tick(85, P(85, seen=TOYS, events=(("reach_nearer", "cup"),)))
    assert s.judgments == [] and con.book_log[-1][3] == "past mastery", (s.judgments, con.book_log[-1])
    s = con.tick(86, P(86, seen=TOYS, events=(("half_roll", None),)))
    assert s.judgments == [(1, "half_roll", None)], s.judgments
    s = con.tick(87, P(87, seen=TOYS, events=(("lifted", "duck"), ("shook", "duck"), ("hit", "duck"), ("head_up", None))))
    assert [(w, k) for w, k, _o in s.judgments] == [(1, "lifted"), (1, "shook"), (1, "hit"), (1, "head_up")], s.judgments
    s = con.tick(88, P(88, seen=TOYS, present=False, events=(("got", "block"),)))
    assert s.judgments == [] and "got" in con.book and "block" not in con.book["got"], s.judgments   # away: nothing judged
    s = con.tick(89, P(89, seen=TOYS, events=(("threw", "ball"),)))
    assert s.frown is None and s.judgments == [], (s.frown, s.judgments)                # stage 1: no frown
    con2 = C.Conduct(seed=4, transcriber=Transcriber(None), imperfect=False, stage=2)
    _no_sets(con2)
    s = con2.tick(0, P(0, seen=TOYS, events=(("threw", "ball"),)))
    assert s.frown == "threw" and s.line is not None and s.line.text == "no.", (s.frown, s.line)
    con3 = _perfect(seed=4, transcriber=Transcriber(None))
    _no_sets(con3)
    con3.load_state(con.state())
    assert con3.book == con.book and con3.book_log == con.book_log
    print(f"A89 motor judgments: 'got ball' 2 and her line '{yes}' at once; the same act {len(ws)} smiles falling",
          f"2 -> {ws[-1]:.3f} then none; another toy afresh; a reach nearer 1 until 'got' is mastered, then 'past mastery'; a half",
          "roll 1; a lift, a shake, a hit, its head up 1 each; away: nothing; stage 2 frowns 'no.' at a thrown toy, stage 1 not;",
          "the book saved")


def test_her_lessons_and_hands():
    """A90 (the teacher's build 2c, 2d): her lesson's setup intents compose as one line with their acts: "set_near" on the ball says
    "here. the ball." (or "look. here. the ball.") with the bring_back of the ball, her look at it and back to its eyes; "hand_over"
    with the hand-over; her book's smiles count as before while a set is queued; and while her hands move its body (a guide, a
    turn, a pull running) nothing it does is judged, and the log says so; the guide and the knee over are no longer closed at
    birth, the pull-to-sit and the prop are"""
    from body.sim import parent_motion as PM
    con = _perfect(seed=4, transcriber=Transcriber(None))
    _no_sets(con)
    con.request("set_near", o="ball")
    s = con.tick(0, P(0, seen=TOYS))
    assert s.line is not None and s.line.intent == "set_near" and s.line.text in ("here. the ball.", "look. here. the ball."), s.line
    assert C.Act("bring_back", "ball") in s.acts and C.Act("look", "child_eyes") in s.acts, s.acts
    con.request("hand_over", o="duck")
    s2 = None
    for t in range(1, 40):
        s2 = con.tick(t, P(t, seen=TOYS))
        if s2.line is not None and s2.line.intent == "hand_over":
            break
    assert s2.line is not None and s2.line.intent == "hand_over" and C.Act("hand_over", "duck") in s2.acts, (s2.line, s2.acts)
    mid = con.motion.request(C.Act("guide", "far_arm"), 40)                      # her guide of its far arm, running (the stub)
    con.acts_open.append([mid, "guide", "far_arm", None, 40, "unreported"])
    s3a = con.tick(40, P(40, seen=TOYS, events=(("head_up", None),)))                   # her approach (no hold yet): its own (A111)
    assert s3a.judgments == [(1, "head_up", None)], s3a.judgments
    s3 = con.tick(41, P(41, seen=TOYS, events=(("rolled", None), ("got", "cup")), her_hold=True))   # her hold engaged: hers
    assert s3.judgments == [] and "her hands on it" in con.book_log[-1][3], (s3.judgments, con.book_log[-1])
    s3b = con.tick(42, P(42, seen=TOYS, events=(("rolled", None),)))                    # and to the act's end, hold or not
    assert s3b.judgments == [] and "her hands on it" in con.book_log[-1][3], s3b.judgments
    t_end = 40 + C.STUB_TICKS["guide"] + 1
    for t in range(41, t_end):
        con.tick(t, P(t, seen=TOYS))
    s4 = con.tick(t_end, P(t_end, seen=TOYS, events=(("rolled", None),)))
    assert s4.judgments == [(2, "rolled", None)], s4.judgments                      # its own repeat after the guide: the worth
    assert PM.NOT_AT_BIRTH == ("pull_to_sit", "prop") and "guide" in C.HANDS_ON and "turn" in C.HANDS_ON
    con5 = _perfect(seed=4, transcriber=Transcriber(None))
    _no_sets(con5)
    s5 = con5.tick(0, P(0, seen=TOYS, events=(("distress", None),)))                # face down in distress: she turns it over first
    assert s5.line is not None and s5.line.intent == "turn_over" and C.Act("turn", "child") in s5.acts, (s5.line, s5.acts)
    s6 = con5.tick(1, P(1, seen=TOYS, events=(("pain", None),)))
    assert s6.line is None or s6.line.intent == "comfort", s6.line                  # its pain alone: comfort as before
    print(f"A90 her lessons and hands: 'set_near' -> '{s.line.text}' with {[a.kind for a in s.acts]}; 'hand_over' -> '{s2.line.text}';",
          "nothing judged while a guide runs (the log says so), the roll after it worth 2; guide and knee_over open at birth, the",
          "pull-to-sit and the prop closed")


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
    twins = TOYS + seen(("ball_blue", "ball", "blue", "mat", True))

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
    """finding 5: an echo of her word earned a smile as a met ask ("more?" at the meal was a name ask). An echo may earn her smile
    (a parent answers imitation, Goldstein and Schwade 2008: her method, disclosed), but counts toward nothing in the ledger. The
    meal and its "more bottle?" line are gone (A88); the probe stands on her label "a bottle." of a bottle in this test's own scene
    ("bottle" is still a first word; the room holds none)."""
    assert "feed_more" not in C.INTENTS and "feed" not in C.INTENTS, "the meal's intents are gone (A88)"

    def bottle(holds):
        """the verifier's probe: "a bottle." (ending at tick 5), the child's "bottle" 2 ticks after it."""
        con = C.Conduct(seed=3, transcriber=Transcriber(None), imperfect=False)
        _no_sets(con, ("duck", "ball", "cup", "block", "block_red", "bottle"))
        things = TOYS + seen(("bottle", "bottle", "", "hand" if holds else "mat", True))
        pp = lambda t: P(t, seen=things, child_holds=("bottle",) if holds else ())      # noqa: E731
        con._say(TP.Line("a bottle.", "label", "plain", "bottle", "bottle", ()), 0, pp(0), C.Say())
        judg = []
        for t in range(1, 40):
            s = con.tick(t, pp(t), token=LX.WORD_ID["bottle"] if t == 7 else None)
            judg += s.judgments
        return con, judg
    con, judg = bottle(False)
    assert judg == [] and not con.ledger.trials and con.ledger.standing("bottle")["asks"] == [], judg
    con, judg = bottle(True)                                                  # in its hand: an echo of a right name
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
    import math as _m
    n_pay = sum(1 for n in range(60) if K.WORTH_RIGHT_NAME * _m.exp(-n / K.HABIT_TAU) >= K.HABIT_FLOOR)   # A99: 37 of 60
    for i in range(60):
        t0 = 100 * i
        s = con.tick(t0, P(t0, child_target="mama"), token=LX.WORD_ID["mama"])        # a right name each time
        if i < n_pay:                                                                   # judged while her smile at the word lasts
            assert s.judgments and (con.reply_due is not None or s.line is not None), (i, s.judgments)   # answered (at t0 itself
        else:                                                                           # at a latency of 0); then (A99) habituated:
            assert not s.judgments, (i, s.judgments)                                    # unjudged, answered or missed as any such turn
        for t in range(t0 + 1, t0 + 30):
            con.tick(t, P(t, child_target="mama"))
    assert sum(con.turns) == 60 - n_pay and con.turns[0] > 0, (con.turns, n_pay)      # no judged turn ever counted as unjudged
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
    # an act ask; on the mat, neither is void
    for kind, things, void in (("gaze", in_hand, True), ("gaze", TOYS, False), ("act", sofa, True), ("act", mat, False)):
        led = Ledger()
        word = "ball" if kind == "gaze" else "bottle"
        led.ask(0, word, kind, word, 20, open_at=2)
        got = [led.observe(t, P(t, child_target="duck", seen=things)) for t in range(2, 30)]
        res = [r[3] for g in got for r in g]
        assert (res == ["void"]) == void and (res[:1] == ["missed"]) != void, (kind, void, res)
    print("40 asks she can judge: 'where is the ball?' and 'give me the ball' are never asked of the ball in her own hands "
          "(dropped, logged; the third round's was met by a look at her hand), and a ball on the mat still is (met by a "
          "look); 'give me the bottle' is never asked of the bottle on the sofa, beyond its reach (no miss counted), and is of "
          "one on the mat (met by the give); the ledger voids a gaze ask whose X is in her hands, and an act ask with no X "
          "within its reach, when the word is heard")


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

# ------------------------------------ formal trials (the lead's decision after P3's eighth round: 4.8, 12, A28, A51), both ways
class _TapLedger(Ledger):
    """a ledger that also keeps its records in memory: the tests read its trials, asks and outcomes."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rows = []

    def _rec(self, d):
        self.rows.append(json.loads(json.dumps(d, sort_keys=True)))
        super()._rec(d)


def _trials(led):
    """the formal trials a tap ledger recorded, each with its outcome: [dict(t, form, target, distractor, said, sides, open,
    window, score, items, id, result, why, end, scored, on, off)] (result None while open; since the lead's decision A60b
    "scored" or "void", on and off its window's ticks on the target and on the distractor, the name test's on her face and
    off it)."""
    out, by = [], {}
    for i, r in enumerate(led.rows):
        if r["ev"] == "trial":
            d = dict(r, id=None, result=None, why=None, end=None, n=i, on=None, off=None)
            out.append(d)
            by[len(out) - 1] = d
        elif r["ev"] == "trial_outcome":
            d = next(x for x in out if x["id"] is None and x["result"] is None)
            d.update(id=r["trial"], result=r["result"], why=r["why"], end=r["t"], scored=r.get("scored"), on=r.get("on"),
                     off=r.get("off"))
    return out


def _oc(x):
    """a trial's end as the flip tests compare it: (result, why, its end's tick, form, the word said, on, off)."""
    return (x["result"], x.get("why"), x.get("end"), x.get("form"), x.get("said"), x.get("on"), x.get("off"))


def _share(x):
    """a scored pair trial's share of the window's looking at either thing that went to the named thing (the name test: its
    face's share of the window), or None."""
    if x.get("result") != "scored":
        return None
    return x["on"] / (x["on"] + x["off"])


def _has_trials():
    return hasattr(C.Conduct, "probe")


class _Kid:
    """a child for the trials, written here: where she reads its head line (the Percept's child_target) tick by tick, from what it
    can know: the room, her body's movements as her motion's fields show them, her words as they sound (3 ticks a word, no
    voice), and the display's sides as placed; never which of the two she will name. Kinds: "gaze" follows her eyes and head to a
    thing, "hands" her hands; "side" looks at the left thing after any line of hers, "favourite" at the ball; "voice" turns to her
    face after any line; "knower" hears its words (knows) and looks at a thing so named (a colour before it chooses among
    twins), and turns to her face after its name alone; "random" only wanders. Each wanders now and then among the things, her
    face and nothing; its latency is drawn in `lat`. rate(t): a voice-turner's chance of turning after a line, a favourite's of
    looking at its favourite rather than the other thing shown (None: always)."""
    KINDS = ("gaze", "hands", "side", "favourite", "voice", "knower", "random")

    def __init__(self, kind, seed=0, knows=("ball", "duck", "cup", "block", "car", "blue", "red"), lat=(4, 10), fav="ball",
                 wander=0.08, rate=None, babble=0.0, turns=0.0):
        assert kind in self.KINDS, kind
        self.kind, self.knows, self.lat, self.fav, self.wander = kind, tuple(knows), lat, fav, wander
        self.rate = rate                  # "voice": its chance of turning after a line, by tick; "favourite": of looking at its
                                          # favourite rather than the other thing shown (None: always)
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(91,))))
        self.cur, self.until, self.plan, self.prev = None, 0, [], {}
        self.babble, self.turns = babble, turns   # its voice (P3's eleventh round): a babble begun at this chance a tick, and
        self.srng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(93,))))   # a turn after
        self.sounds = []                  # her line at this chance, 0-3 ticks after her sound stops (a turn-taker's timing: it
                                          # follows her line's own length); [first tick, ticks], from its own stream

    def _lat(self):
        return int(self.rng.integers(self.lat[0], self.lat[1] + 1))

    def _at(self, tick, spec, hold=None):
        self.plan.append([int(tick), spec, int(self.rng.integers(6, 13)) if hold is None else hold])

    def sounding(self, t):
        """its voice this tick (its own stream, drawn the same whatever it hears)."""
        if self.babble or self.turns:
            if self.srng.random() < self.babble:
                self.sounds.append([t, int(self.srng.integers(2, 5))])
            self.sounds = [x for x in self.sounds if x[0] + x[1] > t]
        return any(a <= t < a + n for a, n in self.sounds)

    def hear(self, t, text, n=None):
        if self.babble or self.turns:     # a turn after her sound stops (n: her line's ticks as it sounds), drawn every line
            go, k, m = self.srng.random() < self.turns, int(self.srng.integers(0, 4)), int(self.srng.integers(2, 5))
            if go:
                self.sounds.append([t + (3 * len(TP.words(text)) if n is None else n) + k, m])
        ws = TP.words(text)
        if self.kind == "voice":
            if self.rate is None or self.rng.random() < self.rate(t):
                self._at(t + self._lat(), "mama")
        elif self.kind == "side":
            self._at(t + 3 * (len(ws) - 1) + self._lat(), "left")
        elif self.kind == "favourite":
            fav = self.rate is None or self.rng.random() < self.rate(t)
            self._at(t + 3 * (len(ws) - 1) + self._lat(), "fav" if fav else "unfav")
        elif self.kind == "knower":
            if TP.key(text) == TP.key(TP.FRAMES["trial_name"][0][0].replace("{n}", LX.NAME)):   # "hi. pip. hi." (P3's
                self._at(t + 3 * ws.index(LX.NAME) + self._lat(), "mama")                     # fourteenth round)
            for i, w in enumerate(ws):
                if w in self.knows and w in TP.OBJECT_NOUNS:
                    col = ws[i - 1] if i and ws[i - 1] in TP.COLOURS and ws[i - 1] in self.knows else None
                    self._at(t + 3 * i + self._lat(), ("word", w, col))

    def see(self, t, fields, things):
        if self.kind not in ("gaze", "hands"):
            return
        for f in (("eyes", "head") if self.kind == "gaze" else ("left", "right")):
            v = fields.get(f)
            if v in things and v != self.prev.get(f):
                self._at(t + self._lat(), v)
        self.prev = dict(fields)

    def look(self, t, seen_, layout):
        ids = [x.id for x in seen_ if x.child_sees]
        due = [x for x in self.plan if x[0] <= t]
        self.plan = [x for x in self.plan if x[0] > t]
        for tick, spec, hold in due[-1:]:
            tg = spec
            if spec == "left":
                tg = layout[0] if layout else None
            elif spec == "fav":
                tg = self.fav if self.fav in ids else None
            elif spec == "unfav":
                tg = next((x for x in (layout or ()) if x != self.fav), None)
            elif isinstance(spec, (tuple, list)):
                _, w, col = spec
                c = [x.id for x in seen_ if x.child_sees and x.name == w and (col is None or x.colour == col)]
                tg = c[0] if c else None
            if tg is not None:
                self.cur, self.until = tg, t + hold
        if not due and t >= self.until and self.rng.random() < self.wander:
            opts = ids + ["mama", None]
            self.cur, self.until = opts[int(self.rng.integers(len(opts)))], t + int(self.rng.integers(4, 13))
        if self.cur is not None and self.cur != "mama" and self.cur not in ids:
            self.cur = None
        return self.cur


class _TW2:
    """a W2-like motion for the trials, written here (its own stream): each act runs its own length (1-30 ticks), some first
    reported 'queued' (not one of the four statuses: running, fail-closed); a trial's two things placed one hand after the other,
    in an order it draws itself (it is never told which will be named); L1's glances at the child's new target and at places,
    never while eyes_on_child (its contract). Rogue, each at its rate per tick while the conduct holds still (con.still): a glance
    at a thing (at the trial's target itself when `cheat`: a motion that knew it), a hand to a thing, its face moving, its trunk
    leaning or shifting, an act it starts, an act reported 'waiting', a field left out: each breaks the contract. moved[t] is
    its own account of tick t, whether anything but her mouth moved (its fields off the child or at rest, its face, its trunk,
    any act not reported ended, a field left out): the checker's truth, independent of the conduct's log."""
    DOES = ("show", "walk", "wave", "pick_up", "open_hand")
    PLACES = ("shelf", "door", "window", "floor")

    def __init__(self, seed=0, lengths=(1, 30), queue_p=0.3, l1_p=0.5, place_p=0.01, rogue=None, cheat=False, when="still"):
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(78,))))
        self.lengths, self.queue_p, self.l1_p, self.place_p, self.cheat = lengths, queue_p, l1_p, place_p, cheat
        self.when = when                      # "still": its breaches through a trial's settle and window; "said": after its
                                              # sentence begins only
        self.rogue = dict(rogue or {})
        self.acts, self.live, self.glances, self.hands, self.faces, self.odd, self.trunks = [], [], [], [], [], [], []
        self.last_target, self.moved, self.fields = None, {}, {}

    def request(self, act, tick, outside=False):
        i = len(self.acts)
        wait = int(self.rng.integers(1, 10)) if self.rng.random() < self.queue_p else 0
        n = 60 if act.during == "focus" else int(self.rng.integers(self.lengths[0], self.lengths[1] + 1))
        order = int(self.rng.integers(2)) if act.kind == "present" else 0
        self.acts.append([i, int(tick), act.kind, act.target, int(tick) + wait, int(tick) + wait + max(n, 2), None, outside,
                          order])
        self.live.append(i)
        return i

    def status(self, i, tick):
        a = self.acts[i]
        if a[6] is not None and tick > a[6]:
            return "cancelled"
        return "queued" if tick < a[4] else ("running" if tick < a[5] else "done")

    def cancel(self, i, tick=None):
        a = self.acts[i]
        a[6] = a[1] if tick is None else int(tick)

    def step(self, t, p, con):
        """L1 and the rogue on tick t, before its report: legal gazes only while no ask or trial holds her eyes; the contract's
        breaches only while con.still (so every breach is where a trial's rule must catch it)."""
        things = [x.id for x in p.seen]
        if not con.eyes_on_child:
            if p.child_target not in (None, "mama", self.last_target) and self.rng.random() < self.l1_p:
                d = int(self.rng.integers(1, 3))
                self.glances.append([p.child_target, t + d, t + d + int(self.rng.integers(2, 10))])
            if self.rng.random() < self.place_p:
                self.glances.append([self.PLACES[int(self.rng.integers(len(self.PLACES)))], t, t + int(self.rng.integers(2, 12))])
        else:                                                   # its contract: a gaze under way is broken off
            for g in self.glances:
                g[2] = min(g[2], t)
        if con.still and (self.when == "still" or con.trial["phase"] == "said"):
            r = self.rogue
            n = int(self.rng.integers(1, 5))
            if self.rng.random() < r.get("glance", 0):
                tr = con.trial
                tg = tr.get("target") if (self.cheat and tr and tr.get("target")) else things[int(self.rng.integers(len(things)))]
                self.glances.append([tg, t, t + n])
            if self.rng.random() < r.get("hand", 0):
                self.hands.append([things[int(self.rng.integers(len(things)))], t, t + n])
            if self.rng.random() < r.get("face", 0):
                self.faces.append([t, t + n])
            if "trunk" in r and self.rng.random() < r["trunk"]:
                self.trunks.append([t, t + n])
            if self.rng.random() < r.get("outside", 0):
                self.request(C.Act("touch", "hand"), t - 1, outside=True)
            if self.rng.random() < r.get("status", 0) or self.rng.random() < r.get("omit", 0):
                self.odd.append(["status" if self.rng.random() < 0.5 else "omit", t, t + n])
        self.last_target = p.child_target

    def report(self, tick):
        f = dict(eyes="child_eyes", head="child_eyes", left=None, right=None, trunk="child", face=None)
        for i in list(self.live):
            a = self.acts[i]
            if self.status(i, tick) != "running" or a[1] >= tick:
                continue
            k, tg = a[2], a[3]
            if k in ("look", "walk"):
                f["eyes"] = f["head"] = tg
            elif k == "lean_in":
                f["head"], f["trunk"] = "child_periphery", "?"
            elif k == "present":
                left, _, right = tg.partition("|")
                half = (a[4] + a[5]) // 2
                first, second = (("left", left), ("right", right)) if a[8] == 0 else (("right", right), ("left", left))
                f[first[0] if tick < half else second[0]] = first[1] if tick < half else second[1]
            elif k in ("copy", "wave", "open_hand"):
                f["right"] = "child"
            elif k != "withdraw":
                f["right" if f["right"] is None else "left"] = tg if isinstance(tg, str) else "air"
        for g in self.glances:
            if g[1] <= tick < g[2]:
                f["eyes"] = f["head"] = g[0]
        for h in self.hands:
            if h[1] <= tick < h[2]:
                f["left"] = h[0]
        if any(a_ <= tick < b_ for a_, b_ in self.faces):
            f["face"] = LX.PARENT_NAME
        if any(a_ <= tick < b_ for a_, b_ in self.trunks):
            f["trunk"] = "?"                                   # her trunk leans or shifts
        acts = {i: self.status(i, tick) for i in self.live}
        why = [k for k in ("eyes", "head") if f[k] not in AT_CHILD_T] + [k for k in ("left", "right", "face") if f[k] is not None]
        why += ["trunk"] if f["trunk"] != "child" else []
        why += [f"act {i} {s_}" for i, s_ in acts.items() if s_ not in ("done", "refused", "cancelled")]
        for kind, a_, b_ in self.odd:
            if a_ <= tick < b_:
                if kind == "status" and acts:
                    acts[next(iter(acts))] = "waiting"
                    why.append("a status 'waiting'")
                elif kind == "omit":
                    f.pop("head", None)
                    why.append("its head left out")
        self.live = [i for i in self.live if acts.get(i) not in ("done", "refused", "cancelled")]
        self.glances = [g for g in self.glances if g[2] > tick]
        self.hands = [h for h in self.hands if h[2] > tick]
        self.faces = [x for x in self.faces if x[1] > tick]
        self.trunks = [x for x in self.trunks if x[1] > tick]
        self.odd = [x for x in self.odd if x[2] > tick]
        self.moved[tick] = "; ".join(why) or None
        self.fields = dict(f)
        return dict(f, acts=acts)

    def state(self):
        return json.loads(json.dumps(dict(rng=self.rng.bit_generator.state, acts=self.acts, live=self.live,
                                          glances=self.glances, hands=self.hands, faces=self.faces, odd=self.odd,
                                          trunks=self.trunks, last_target=self.last_target)))

    def load_state(self, s):
        self.rng.bit_generator.state = s["rng"]
        self.acts, self.live, self.glances = [list(a) for a in s["acts"]], list(s["live"]), [list(g) for g in s["glances"]]
        self.hands, self.faces = [list(h) for h in s["hands"]], [list(x) for x in s["faces"]]
        self.odd = [list(x) for x in s["odd"]]
        self.trunks = [list(x) for x in s.get("trunks", ())]
        self.last_target = s["last_target"]


AT_CHILD_T = ("child", "child_eyes", "child_periphery")


class _Life:
    """a stretch of life with formal trials, stepped tick by tick so it can be snapshotted: each tick her motion's L1 (a _TW2's
    step), the child's reading from what it can know (_Kid), her conduct's tick (skipped at skip_p: a tick the world does not
    tick her), her lines heard by the child. probes: {tick: (form, a, b, new)}; asks: {tick: (intent, kw)}. It keeps her lines,
    her judgments, the ticks skipped and the ticks her face moved with her own judgments (FACE_COURSE from each: the checker's,
    from Say)."""

    def __init__(self, con, kid, probes=None, asks=None, seen_=TOYS, motion=None, skip_p=0.0, seed=0, events=None):
        self.con, self.kid, self.probes, self.asks, self.seen, self.events = con, kid, dict(probes or {}), dict(asks or {}), \
            seen_, dict(events or {})
        self.motion = motion if motion is not None else con.motion
        self.skip_p = skip_p
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(79,))))
        self.said, self.judg, self.skipped, self.face = [], [], set(), set()
        self.face_at = []                                         # (tick, its judgments' kinds, a frown) her face moved from
        self.ids = {x.id for x in seen_}
        self.voiced = set()                                       # the ticks the child's voice sounded

    def step(self, t):
        con, kid, motion = self.con, self.kid, self.motion
        if t in self.probes:
            form, a, b, new = self.probes[t]
            con.probe(form, a, b, new=new)
        if t in self.asks:
            intent, kw = self.asks[t]
            con.request(intent, **kw)
        tr = con.trial
        layout = (tr["left"], tr["right"]) if tr and tr.get("left") and tr["phase"] != "bring" else None
        p = P(t, child_target=kid.look(t, self.seen, layout), seen=self.seen, events=tuple(self.events.get(t, ())),
              child_sounding=kid.sounding(t) if hasattr(kid, "sounding") else False)
        if p.child_sounding:
            self.voiced.add(t)
        if hasattr(motion, "step"):
            motion.step(t, p, con)
        if self.skip_p and self.rng.random() < self.skip_p:
            self.skipped.add(t)
            if hasattr(motion, "moved"):
                motion.report(t)                                  # the world moved on; the conduct never read it
            return None
        s = con.tick(t, p)
        kid.see(t, getattr(motion, "fields", {}) or {}, self.ids)
        if s.line is not None:
            self.said.append((t, s.line))
            if s.clip is not None:
                kid.hear(t, s.line.text, -(-len(s.clip.pcm) // 2400))
            else:
                kid.hear(t, s.line.text)
        self.judg += [(t,) + j for j in s.judgments]
        if s.judgments or s.frown:
            self.face.update(range(t, t + K.FACE_COURSE + 1))
            self.face_at.append((t, tuple(j[1] for j in s.judgments), bool(s.frown)))
        return s

    def run(self, t0, t1):
        for t in range(t0, t1):
            self.step(t)
        return self

    def snapshot(self):
        import copy                                                        # noqa: PLC0415
        return dict(con=json.loads(json.dumps(self.con.state(), default=_np)), motion=self.motion.state(),
                    kid=copy.deepcopy(self.kid), rng=self.rng.bit_generator.state, said=len(self.said),
                    judg=len(self.judg), skipped=sorted(self.skipped), face=sorted(self.face),
                    face_at=[list(x) for x in self.face_at],
                    rows=len(self.con.ledger.rows))


def _trial_life(con, kid, n, probes=None, asks=None, seen_=TOYS, motion=None, skip_p=0.0, seed=0, events=None, t0=0):
    """_Life(...).run(t0, t0 + n) -> dict(said, judg, skipped, face)."""
    lf = _Life(con, kid, probes, asks, seen_, motion, skip_p, seed, events).run(t0, t0 + n)
    return dict(said=lf.said, judg=lf.judg, skipped=lf.skipped, face=lf.face)


def _pair_con(seed=1, stage=2, motion=None, led=None):
    con = C.Conduct(seed=seed, stage=stage, imperfect=False, motion=motion, ledger=led if led is not None else _TapLedger())
    _no_sets(con, ("duck", "ball", "cup", "block", "block_red", "ball_blue", "car"))
    return con


def test_trial_protocol():
    """the lead's decision: understanding is scored only in formal trials (4.8, 12), by the proportion of looking over a fixed
    window (A60b). The protocol with her own stub motion: at a probe she brings the two things into its view side by side (one
    'present' act, W2 never told which will be named), settles SETTLE ticks with only her mouth free, says the single test
    sentence with no act; its window runs from the test word's onset tick + 2 to its onset tick + 23, the same ticks whichever is
    named, and each tick her reading of its head line on the target, on the distractor or on neither is counted: only how many,
    never when; a pair's trial with fewer than 4 of them on either thing is void. No feedback: she gives nothing for a trial
    (A60b). Stillness enforced: a motion that moves in the settle holds the sentence back; one that moves after it (a glance, a
    hand, its face,
    its trunk, an act, an odd status, a field left out, a tick not read) voids the trial, logged with why."""
    assert _has_trials(), "no formal trial in her conduct: understanding is still scored in her everyday asks"
    assert hasattr(K, "TRIAL_LOOK"), "no fixed window of looking: a trial is still decided by a look timed from its word (A60b)"
    assert hasattr(Ledger, "maps"), "one level of 'understood' only (A60b's two levels; a trial gives no feedback)"
    con = _pair_con()
    con.probe("place", "ball", "duck", new={"ball": "ball@table", "duck": "duck@floor"})
    kid = _Kid("knower", seed=3, wander=0.0)
    run = _trial_life(con, kid, 140)
    trs = _trials(con.ledger)
    assert len(trs) == 1 and trs[0]["result"] == "scored" and trs[0]["on"] > 0 == trs[0]["off"], trs
    tr = trs[0]
    acts = [a for a in con.motion.acts if a[2] == "present"]
    assert len(acts) == 1 and set(acts[0][3].split("|")) == {"ball", "duck"}, acts
    t_line = next(t for t, ln in run["said"] if ln.intent == "trial_where")
    line = next(ln for t, ln in run["said"] if ln.intent == "trial_where")
    assert line.text == f"where is the {tr['target']['word']}? see?" and line.register == "question", line
    assert t_line - acts[0][5] + 1 >= K.SETTLE, (t_line, acts[0][5])        # the sentence's own tick the settle's last
    assert tr["open"] == t_line + 9 and tr["window"] == [tr["open"] + 2, tr["open"] + 23] and tr["end"] == tr["open"] + 23 \
        and tr["score"] == [[tr["target"]["word"], "trials", 1, 0, 1], [tr["distractor"]["word"], "yoked", 0, 1, 1]], tr
    after = [ln.text for t, ln in run["said"] if t >= tr["t"] and ln.intent != "trial_where"]
    assert run["judg"] == [] and not [x for x in after if x.startswith("yes")], (run["judg"], after)   # no feedback (A60b)
    # no act with the sentence, and her hold on the child's eyes flagged for W2 through the settle and the window
    con2 = _pair_con()
    con2.probe("place", "cup", "duck", new={"cup": "c1", "duck": "d1"})
    flags, acts2 = [], []
    for t in range(120):
        s = con2.tick(t, P(t, child_target=None))
        flags.append((t, con2.still, con2.eyes_on_child, con2.trial["phase"] if con2.trial else None))
        if s.line is not None:
            acts2.append((t, s.line.intent, s.acts))
    said_at = [t for t, it, a in acts2 if it == "trial_where"]
    assert said_at and [a for t, it, a in acts2 if it == "trial_where"] == [()], acts2
    assert all(st and eo for t, st, eo, ph in flags if ph in ("settle", "said")) and \
        not any(st for t, st, eo, ph in flags if ph in ("bring", None)), flags[:20]
    none = _trials(con2.ledger)[0]
    assert none["result"] == "void" and "fewer than 4" in none["why"] and none["end"] == none["open"] + 23 and \
        not none["scored"] and con2.ledger.voided == {"place": 1}, none
    # only the window's ticks count, never when in it: on the distractor through it, scored, no smile; on the target from 2
    # ticks before its onset, 2 after, 20 or 21 after it, 8 ticks each
    shares, judg3 = {}, {}
    for name, first in (("distractor", None), ("onset - 2", -2), ("onset + 2", 2), ("onset + 20", 20), ("onset + 21", 21)):
        con3 = _pair_con()
        con3.probe("place", "cup", "duck", new={"cup": "c1", "duck": "d1"})
        judg3[name] = []
        for t in range(140):
            tr3 = con3.trial
            tg = None
            if tr3 and tr3["phase"] == "said":
                if first is None:
                    tg = tr3["distractor"]
                elif tr3["onset"] + first <= t < tr3["onset"] + first + 8:
                    tg = tr3["target"]
            judg3[name] += con3.tick(t, P(t, child_target=tg)).judgments
        v = _trials(con3.ledger)[0]
        shares[name] = (v["result"], v["on"], v["off"])
    assert shares == {"distractor": ("scored", 0, 22), "onset - 2": ("scored", 4, 0), "onset + 2": ("scored", 8, 0),
                      "onset + 20": ("scored", 4, 0), "onset + 21": ("void", None, None)} and \
        not any(judg3.values()), (shares, judg3)
    # stillness: each breach after the sentence voids it; in the settle it holds the sentence back (the settle restarts)
    breaches = dict(glance=dict(glance=1.0), hand=dict(hand=1.0), face=dict(face=1.0), trunk=dict(trunk=1.0),
                    outside=dict(outside=1.0), status=dict(status=1.0), omit=dict(omit=1.0))
    got = {}
    for name, rogue in breaches.items():
        m = _TW2(seed=5, rogue=rogue, lengths=(3, 3), queue_p=0.0, l1_p=0.0, place_p=0.0)
        con4 = _pair_con(motion=m)
        con4.probe("place", "cup", "duck", new={"cup": "c1", "duck": "d1"})
        said4 = []
        for t in range(500):
            tr4 = con4.trial
            if tr4 is not None and tr4["phase"] == "said" and t == tr4["line_start"] + 1:
                m.rogue = rogue                                   # a breach from the tick after the sentence begins
            elif tr4 is not None and tr4["phase"] != "said":
                m.rogue = {}
            p = P(t, child_target=None)
            m.step(t, p, con4)
            s = con4.tick(t, p)
            if s.line is not None:
                said4.append((t, s.line.text))
        got[name] = _trials(con4.ledger)
        assert got[name] and got[name][0]["result"] == "void" and "her attention log" in got[name][0]["why"], (name, got[name])
    m = _TW2(seed=5, rogue=dict(glance=1.0), lengths=(3, 3), queue_p=0.0, l1_p=0.0, place_p=0.0)
    con5 = _pair_con(motion=m)
    con5.probe("place", "cup", "duck", new={"cup": "c1", "duck": "d1"})
    for t in range(K.TRIAL_WAIT + 20):
        p = P(t, child_target=None)
        m.step(t, p, con5)
        con5.tick(t, p)
    assert not _trials(con5.ledger) and any("not said within" in r[2] for r in con5.fast.refused), con5.fast.refused[-2:]
    # a tick not read after the sentence voids it; one in the settle restarts it
    con6 = _pair_con()
    con6.probe("place", "cup", "duck", new={"cup": "c1", "duck": "d1"})
    for t in range(140):
        tr6 = con6.trial
        if tr6 is not None and tr6["phase"] == "said" and t == tr6["line_start"] + 2:
            continue
        con6.tick(t, P(t, child_target=None))
    v6 = _trials(con6.ledger)[0]
    assert v6["result"] == "void" and "not read" in v6["why"], v6
    print(f"59 a formal trial (4.8, 12): 'present' asked once for the two things (sides drawn, W2 never told which is named), "
          f"{t_line - acts[0][5] + 1} ticks of stillness from its end (the sentence's tick the last), then '{line.text}' "
          f"with no act (only her mouth), the test word's onset at tick {tr['open']} and its window ticks {tr['window'][0]}-"
          f"{tr['window'][1]} (A60b); a knower {tr['on']} of them on the target, none on the other: scored as the named "
          f"word's test and the distractor's yoked trial, and no smile, no confirm, nothing for it (no feedback in a trial, "
          f"A60b); no look: void (fewer than 4 ticks on either), counted, never scored; on the distractor through it: scored "
          f"0 to 22; the same "
          f"8-tick look on the target counted 4, 8, 4 and 3 ticks begun 2 before its onset, 2, 20 and 21 after it (the "
          f"last void): only the window's ticks, never when; W2 holds her still through the settle and window (still, "
          f"eyes_on_child); each breach after the sentence ({', '.join(got)}) voids it, logged; in the settle it holds the "
          f"sentence back (dropped after {K.TRIAL_WAIT} ticks); a tick not read voids it")


def _many_trials(kind, n_trials, seed=0, motion=None, rule=True, knows=None, forms=("place",), pairs=None, wander=0.3):
    """one long stretch of probes, back to back, with a child of one kind: -> (trials, conduct). rule False switches the trial's
    void rule off (moving() read as still), to show it is what holds a follower of a cheating motion at chance."""
    led = _TapLedger()
    con = _pair_con(seed=seed, motion=motion, led=led)
    kid = _Kid(kind, seed=seed + 100, wander=wander, **({} if knows is None else dict(knows=knows)))
    pairs = pairs or (("ball", "duck"), ("cup", "block_red"), ("ball", "cup"), ("duck", "block_red"), ("duck", "cup"),
                      ("ball", "block_red"))
    saved = C.moving
    if not rule:
        C.moving = lambda e: None
    try:
        t, k = 0, 0
        while k < n_trials:
            if con.trial is None and not con.probes:
                a, b = pairs[k % len(pairs)]
                con.probe(forms[k % len(forms)], a, b, new={a: f"{a}@{k}", b: f"{b}@{k}"})
                k += 1
            tr = con.trial
            layout = (tr["left"], tr["right"]) if tr and tr.get("left") and tr["phase"] != "bring" else None
            p = P(t, child_target=kid.look(t, TOYS, layout))
            if hasattr(con.motion, "step"):
                con.motion.step(t, p, con)
            s = con.tick(t, p)
            kid.see(t, getattr(con.motion, "fields", {}) or {}, {x.id for x in TOYS})
            if s.line is not None:
                kid.hear(t, s.line.text)
            t += 1
        for u in range(t, t + 200):
            s = con.tick(u, P(u, child_target=kid.look(u, TOYS, None)))
            if s.line is not None:
                kid.hear(u, s.line.text)
    finally:
        C.moving = saved
    return [x for x in _trials(led) if x["result"] is not None], con


def _rate(trs):
    """a child's scored pair trials whose looking was not even (A60b): how many went more to the named thing than to the other
    -> (k, n). For a child whose looking does not depend on the word said, each is a fair coin (the named thing is)."""
    c = [x for x in trs if x["result"] == "scored" and x["form"] != "name" and x["on"] != x["off"]]
    return sum(x["on"] > x["off"] for x in c), len(c)


def _btail(k, n, p0=0.5):
    """P(X >= k), X ~ Binomial(n, p0), exact in rationals."""
    from fractions import Fraction                                        # noqa: PLC0415
    q = Fraction(p0)
    return float(sum(math.comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(max(0, k), n + 1)))


def _two_sided(k, n, p0=0.5):
    return min(1.0, 2 * min(_btail(k, n, p0), 1 - _btail(k + 1, n, p0)))


def _crit(m, alpha):
    """the fewest of m that a one-sided binomial test against 1/2 finds above it at alpha (scipy's), or m + 1."""
    from scipy.stats import binom                                         # noqa: PLC0415
    return next((k for k in range(m + 1) if binom.sf(k - 1, m, 0.5) < alpha), m + 1)


def _rule_share(seq):
    """the ledger's rule (4.8, the lead's decision A60b) as it states it, from a word's record [(named, a, b)] in its order:
    tested at its 12th registered trial and at each doubling (24, 48, ...), the j-th over all its trials so far at 0.01 / 2^j,
    only where 1 / C(n, k) is under that level, by the permutation test's p (exact: test 12 holds it to brute force) -> the index
    of the trial it was first reached at, or None."""
    n, j = K.UNDERSTOOD_FIRST, 1
    while n <= len(seq):
        w = [tuple(x) for x in seq[:n]]
        r = perm_test(w, K.UNDERSTOOD_P / 2 ** j)
        if 1.0 / math.comb(n, r["named"]) < K.UNDERSTOOD_P / 2 ** j and r["p"] < K.UNDERSTOOD_P / 2 ** j:
            return n - 1
        n, j = 2 * n, j + 1
    return None


def test_trial_chance_and_counterbalance():
    """chance is 50% by counterbalancing (the lead's decision A60b): her trial stream draws which of the two is named and the
    sides, each a fair coin, so a child that follows her gaze or her hands (a W2 placing the two one hand after the other, L1
    glancing at its targets before the settle), a side's or a toy's favourite and a wanderer each look more at the named thing
    in half their trials (two-sided binomial), and a child that knows the words far more; a favourite's share of the looking on
    its toy is high whichever is named, and its difference null; each word's level 1 ("maps") exactly as the rule gives (its
    tests at 12, 24, 48, ... registered trials), a knower's words, and the non-knowers' no more often than its levels allow;
    level 2 ("understood") waits for its kind's new exemplars (A60b); section 12's flip test
    of the form, each trial once: the knower's far below 0.01, the others' above. Against a motion that knew the target and
    glanced at it after the sentence, every such trial is void; with the void rule switched off the gaze follower's looking
    goes to the target: the rule is what holds it at chance."""
    assert _has_trials() and perm_test is not None, "no formal trial scored by the proportion of looking (A60b)"
    assert hasattr(Ledger, "maps"), "one level of 'understood' only (A60b's two levels)"
    res, rows = {}, []
    for i, kind in enumerate(("gaze", "hands", "side", "favourite", "random", "knower")):
        m = _TW2(seed=11, lengths=(4, 16), queue_p=0.2, l1_p=0.9, place_p=0.02) if kind in ("gaze", "hands") else None
        trs, con = _many_trials(kind, 300, seed=7 + i, motion=m)
        k, n = _rate(trs)
        res[kind] = (k, n, _two_sided(k, n), con)
        rows += trs
    by_chance, n_words = [], 0
    for kind in ("gaze", "hands", "side", "favourite", "random"):
        k, n, p2, con = res[kind]
        assert n >= 100 and p2 > 0.01, f"a {kind} child looked more at the named thing in {k} of {n} (two-sided p = {p2:.4f})"
        for w in ("ball", "duck", "cup", "block"):
            st = con.ledger.words[w]
            got = _rule_share(st["seq"])
            assert (st["maps_at"] is not None) == (got is not None), (kind, w, got, st["maps_at"], st["tests"])
            n_words += 1
            if got is not None:
                by_chance.append(f"{kind}:{w}")
    assert len(by_chance) <= 1, by_chance                # (at most 1% of words by construction: 20 words here)
    k, n, p2, con = res["knower"]
    assert n >= 100 and k / n > 0.8 and _btail(k, n) < 1e-12, (k, n)
    assert all(con.ledger.maps(w) and not con.ledger.understood(w) for w in ("ball", "duck", "cup", "block")), \
        {w: con.ledger.test(w) for w in ("ball", "duck", "cup", "block")}      # (level 2 waits for new exemplars: A60b)
    # the favourite's ball: its share of the looking on the ball high when named and when not ((the old (i) alone would have
    # credited it), its difference null: the permutation test's comparison is the same thing's share when the other is named
    fav = res["favourite"][3].ledger.test("ball")
    seqf = res["favourite"][3].ledger.words["ball"]["seq"]
    sh = [a / (a + b) for _nm, a, b in seqf]
    assert sum(sh) / len(sh) > 0.8 and fav["p"] > 0.01 and not res["favourite"][3].ledger.maps("ball"), fav
    # section 12's test of the form (A19, A28): the flip test over its trials, each once, one-sided p < 0.01
    pooled = {kd: res[kd][3].ledger.pooled("place") for kd in res}
    assert pooled["knower"]["p"] <= 1.0 / (K.PERM_DRAWS + 1) and all(pooled[kd]["p"] > 0.01 for kd in res if kd != "knower"), \
        {kd: (v["n"], v["x"], v["p"]) for kd, v in pooled.items()}      # (its 300 trials' p drawn: at its floor, 1 / 20,001)
    least = min(v["p"] for kd, v in pooled.items() if kd != "knower")
    # the counterbalance: over every trial, which of the two is named and which is on the left, each near half, independent
    named_left = [x["target"]["id"] == x["sides"][0] for x in rows if x["sides"]]
    k_left, n_all = sum(named_left), len(named_left)
    both = {}
    for x in rows:
        tr_pair = tuple(sorted((x["target"]["id"], x["distractor"]["id"])))
        both.setdefault(tr_pair, [0, 0])[0 if x["target"]["id"] == tr_pair[0] else 1] += 1
    k_first = sum(a for a, b in both.values())
    lr = [(x["target"]["id"] == x["sides"][0], x["target"]["id"] == min(x["target"]["id"], x["distractor"]["id"]))
          for x in rows if x["sides"]]
    same = sum(a == b for a, b in lr)
    assert _two_sided(k_left, n_all) > 0.01 and _two_sided(k_first, n_all) > 0.01 and _two_sided(same, len(lr)) > 0.01, \
        (k_left, k_first, same, n_all)
    assert all(_two_sided(a, a + b) > 0.001 for a, b in both.values()), both
    # a motion that knew the target and glanced at it after the sentence: void every time; without the rule, its follower's
    # looking goes to the target
    cheat = _TW2(seed=12, lengths=(4, 8), queue_p=0.0, l1_p=0.0, place_p=0.0, rogue=dict(glance=0.3), cheat=True, when="said")
    trs_c, _ = _many_trials("gaze", 150, seed=8, motion=cheat)
    void = [x for x in trs_c if x["result"] == "void" and "her attention log" in (x["why"] or "")]
    kc, nc = _rate(trs_c)
    cheat2 = _TW2(seed=12, lengths=(4, 8), queue_p=0.0, l1_p=0.0, place_p=0.0, rogue=dict(glance=0.3), cheat=True,
                  when="said")
    trs_o, _ = _many_trials("gaze", 150, seed=8, motion=cheat2, rule=False)
    ko, no = _rate(trs_o)
    assert len(void) >= 40 and (nc == 0 or _two_sided(kc, nc) > 0.01), (len(void), kc, nc)
    assert no >= 40 and ko / no > 0.8, (ko, no)
    print(f"60 chance 50% by counterbalancing (A60b): over 300 probes each, the trials whose looking went more to the named "
          f"thing: a follower of her gaze {res['gaze'][0]}/{res['gaze'][1]}, of her hands {res['hands'][0]}/"
          f"{res['hands'][1]} (a W2 placing one hand after the other, L1 at its targets before the settle), a side's "
          f"favourite {res['side'][0]}/{res['side'][1]}, the ball's {res['favourite'][0]}/{res['favourite'][1]}, a wanderer "
          f"{res['random'][0]}/{res['random'][1]} (each two-sided p > 0.01; the ball's favourite's mean share on the ball "
          f"{sum(sh) / len(sh):.2f} over {len(sh)} trials whichever was named, its difference {fav['diff']:+.3f}, p = "
          f"{fav['p']:.2f}); {len(by_chance)} of their {n_words} words mapped (level 1), exactly as the rule gives (its tests "
          f"at 12, 24, 48 and 96 registered trials, at 0.005, 0.0025, ...); a knower {k}/{n}, every word mapped (understood, "
          f"level 2, waits for its kind's new exemplars); section 12's "
          f"flip test of the form, each trial once: the knower's p = {pooled['knower']['p']:.1e} (the floor of its 20,000 "
          f"draws), "
          f"the others' each above 0.01 "
          f"(least {least:.2f}); over its {n_all} trials the target the first of its pair {k_first}, on the left {k_left}, "
          f"the two alike {same} (each two-sided p > 0.01); a motion that knew the target and glanced at it after the "
          f"sentence: {len(void)} of {len(trs_c)} void (the rest {kc}/{nc}), and with the void rule off its follower's "
          f"looking went more to the target in {ko}/{no}")


def test_name_trial_foil():
    """the name test (Mandel, Jusczyk and Pisoni 1995): its name or a foil name in the same voice (the calling register) and
    stillness, drawn by her trial stream, no things; the measure its face's share of the window (the lead's decision A60b)
    after its name against after a foil. A child that turns to any voice turns after both: its share the same, not understood;
    a child that knows its name turns after it alone: understood. The foil is no word of hers, never Claude's, and never said
    outside the name test; her everyday calls answered count toward nothing."""
    assert _has_trials() and perm_test is not None, "no name test scored by its face's share of the window (A60b)"
    out = {}
    for kind in ("voice", "knower", "random"):
        led = _TapLedger()
        con = C.Conduct(seed=21, stage=2, imperfect=False, ledger=led)
        _no_sets(con, ("duck", "ball", "cup", "block", "block_red"))
        kid = _Kid(kind, seed=5)
        t = 0
        lines = []
        while len([x for x in _trials(led) if x["result"] is not None]) < 80:
            if con.trial is None and not con.probes:
                con.probe("name")
            p = P(t, child_target=kid.look(t, TOYS, None))
            s = con.tick(t, p)
            if s.line is not None:
                kid.hear(t, s.line.text)
                lines.append(s.line)
            t += 1
        trs = [x for x in _trials(led) if x["result"] is not None]
        nm = [_share(x) for x in trs if x["said"] == LX.NAME and x["result"] == "scored"]
        fo = [_share(x) for x in trs if x["said"] != LX.NAME and x["result"] == "scored"]
        out[kind] = (sum(nm) / len(nm), len(nm), sum(fo) / len(fo), len(fo), con.ledger.understood(LX.NAME), lines, trs,
                     con.ledger.words.get(LX.NAME, {}).get("seq", []), con.ledger.test(LX.NAME))
    fn, nn, ff, nf, und, lines, trs, seq, tv = out["voice"]
    assert nn >= 25 and nf >= 25 and fn > 0.2 and ff > 0.2 and tv["p"] > 0.01 and not und, (out["voice"][:5], tv)
    fn2, nn2, ff2, nf2, und2, lines2, _, seq2, tk = out["knower"]
    assert fn2 > 0.3 and ff2 < 0.1 and und2, out["knower"][:5]
    for kind in out:                                      # understood exactly as the rule gives
        assert out[kind][4] == (_rule_share(out[kind][7]) is not None), (kind, out[kind][4], out[kind][8])
    foils = {ln.text for ln in lines if ln.intent == "trial_name" and LX.NAME not in TP.words(ln.text)}
    assert foils and foils <= {TP.foil_line(f_) for f_ in K.NAME_FOILS} and {ln.register for ln in lines if ln.intent ==
                                                                              "trial_name"} == {"calling"}, (foils, lines[:4])
    assert all(ln.intent == "trial_name" for ln in lines if set(TP.words(ln.text)) & set(K.NAME_FOILS))
    for f_ in K.NAME_FOILS:
        assert not TP.check(f"{f_}.", VOCAB)[0] and not TP.check(f"{f_}.", VOCAB, source="claude")[0]
        assert f_ not in VOCAB and f_ not in TP.GROWTH_WORDS
    draws = [x["said"] == LX.NAME for x in trs]
    assert _two_sided(sum(draws), len(draws)) > 0.01, (sum(draws), len(draws))
    # her everyday calls met, over and over: counted toward nothing
    led = _TapLedger()
    con = C.Conduct(seed=22, stage=2, imperfect=False, ledger=led)
    for t in range(3000):
        if t % 250 == 0:
            con.request("call")
        pd = con.pending
        tg = "mama" if pd is not None and pd["kind"] == "call" and t >= pd["open"] + 2 else "duck"
        con.tick(t, P(t, child_target=tg))
    st = con.ledger.standing(LX.NAME)
    assert st["asks"] and all(st["asks"]) and not con.ledger.understood(LX.NAME) and st["trials"] == [], st
    print(f"61 the name test (Mandel et al. 1995): its name or a foil ({', '.join(K.NAME_FOILS)}) drawn by her stream "
          f"({sum(draws)} names of {len(draws)}), each in 'hi. X. hi.' in the calling register, her body still; its face's "
          f"share of the window (A60b): a child that turns to any voice {fn:.2f} after its name ({nn} trials) and {ff:.2f} "
          f"after a foil ({nf}), p = {tv['p']:.2f}, not understood; one that knows its name {fn2:.2f} against {ff2:.2f}: "
          f"understood; each as the rule gives; the foils no words of hers and refused to Claude; "
          f"{len(st['asks'])} everyday calls all met: the name not understood (teaching only)")


def test_everyday_asks_teaching_only():
    """her everyday asks stay her teaching (4.3's worth table: a met ask earns her smile) and count toward nothing: a child that
    meets every gaze ask, give and call never has a word understood by them (on the code before the lead's decision it had, at
    6 of 10 against a base rate); a word the child says in answer to her name ask is never 'says', and a word it names in a
    trial's window neither, while the same word said of what it attends, unasked, counts; a give of another toy is a missed ask;
    each ask and outcome is recorded as teaching."""
    led = _TapLedger()
    con = C.Conduct(seed=31, stage=2, imperfect=False, ledger=led, transcriber=Transcriber(None))
    _no_sets(con, ("duck", "ball", "cup", "block", "block_red"))
    if hasattr(led, "base_trial"):                                # the code before: its base rate, 0 in 10 random moments
        for i in range(10):
            led.base_trial(-1000 + 30 * i, "ball")
            for t in range(-1000 + 30 * i, -1000 + 30 * i + 21):
                led.observe(t, P(t, child_target=None))
    judg = []
    for t in range(4000):
        if t % 200 == 0:
            con.request(("ask_where", "ask_give")[(t // 200) % 2], o="ball")
        pd = con.pending
        tg, ev = "duck", ()
        if pd is not None and t >= pd["open"] + 1:
            if pd["kind"] == "gaze":
                tg = "ball"
            elif pd["kind"] == "act" and t == pd["open"] + 3:
                ev = (("gave", "ball"),)
        judg += con.tick(t, P(t, child_target=tg, events=ev)).judgments
    st = con.ledger.standing("ball")
    assert not con.ledger.understood("ball"), f"understood by her everyday asks: {st}"
    assert len(st["asks"]) == K.ASKS_KEEP and all(st["asks"]) and [j for j in judg if j[1] == "met_ask"], (st, judg[:3])
    asks = [r for r in led.rows if r["ev"] in ("ask", "outcome")]
    assert asks and all(r.get("teaching") for r in asks), asks[:2]
    n_met = sum(r["ev"] == "outcome" and r["result"] == "met" for r in asks)
    # a give of another toy: a missed ask
    con = C.Conduct(seed=32, stage=2, imperfect=False, ledger=_TapLedger())
    _no_sets(con)
    con.request("ask_give", o="ball")
    for t in range(60):
        pd = con.pending
        ev = (("gave", "cup"),) if pd is not None and t == pd["open"] + 3 else ()
        con.tick(t, P(t, child_target="duck", events=ev))
    assert con.ledger.standing("ball")["asks"] == [0], con.ledger.standing("ball")
    # "says": an answer to her name ask, and a word named in a trial's window, never; the same word said unasked, of what it
    # attends, over two days, counts
    led = _TapLedger()
    con = C.Conduct(seed=33, stage=2, imperfect=False, ledger=led, transcriber=Transcriber(None))
    _no_sets(con, ("duck", "ball", "cup", "block", "block_red"))
    cup = LX.WORD_ID["cup"]
    toks = {}
    for day in (0, 1):
        for k in range(3):
            base = day * K.TICKS_PER_DAY + 300 * k
            con.request("ask_what", o="cup")
            for t in range(base, base + 60):
                pd = con.pending
                tok = cup if pd is not None and pd["kind"] == "name" and t == pd["open"] + 12 else None
                con.tick(t, P(t, child_target=None, child_holds=("cup",)), token=tok)
    rows = [r for r in led.rows if r["ev"] == "child" and r["word"] == "cup"]
    assert rows and not any(r["counted"] for r in rows) and all("teaching" in r["why"] for r in rows), rows[:2]
    assert not con.ledger.says("cup", "token") and [r for r in led.rows if r["ev"] == "outcome" and r["result"] == "met"]
    con.probe("place", "cup", "duck", new={"cup": "cup@x", "duck": "duck@y"})
    t = 3 * K.TICKS_PER_DAY
    named = None
    for t in range(t, t + 200):
        tr = con.trial
        tok = LX.WORD_ID[tr["word"]] if tr is not None and tr["phase"] == "said" and t == tr["onset"] + 14 else None
        named = named or (tok and tr["word"])
        con.tick(t, P(t, child_target=None, child_holds=("cup",) if named == "cup" else ()), token=tok)
    rows2 = [r for r in led.rows if r["ev"] == "child" and r["word"] == named]
    assert named and rows2 and not rows2[-1]["counted"] and "teaching" in rows2[-1]["why"], (named, rows2[-1:])
    for day in (4, 5):
        for k in range(2):
            t0 = day * K.TICKS_PER_DAY + 400 * k
            for t in range(t0, t0 + 120):                         # said well after her own naming of the cup it holds
                con.tick(t, P(t, child_target=None, child_holds=("cup",)), token=cup if t == t0 + 90 else None)
    assert con.ledger.says("cup", "token"), con.ledger.standing("cup")
    print(f"62 her everyday asks are teaching only: {n_met} gaze asks and gives met, the last {len(st['asks'])} in a row (a "
          f"smile each, each recorded as teaching): 'ball' not understood (the code before: understood at them against its base rate); a give "
          f"of the cup to 'give me the ball': missed; 'cup' in answer to 'what is this?' ({len(rows)} times over 2 days) and "
          f"named in a trial's window: never 'says'; the same word said unasked of the cup it holds on 2 more days: 'says'")


PROP_KIDS = ("gaze", "hands", "side", "favourite", "voice", "random", "knower")
PROP_PAIRS = (("ball", "cup"), ("cup", "block_red"), ("ball", "block_red"), ("cup", "ball"), ("block_red", "ball"),
              ("ball", "duck"))                          # ("duck" has no time-matched stimulus in her voice: never run)


def _prop_life(seed, n=5000):
    """one random life for the property, set up and not yet run: a child of a kind by seed, a W2-like motion with its own rates
    of queued acts, L1's glances and breaches of its contract while she holds still (a third by a motion that knew the target),
    ticks the world skips, her day plan's probes (place and name, every 200-350 ticks; a fifth of the place items already shown
    3 times) and everyday asks between them -> (_Life, kind, n)."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(80,))))
    kind = PROP_KIDS[seed % len(PROP_KIDS)]
    rogue = {k: float(rng.choice([0.0, 0.0, 0.0, 0.003, 0.01])) for k in ("glance", "hand", "face", "outside", "status",
                                                                           "omit", "trunk")}
    m = _TW2(seed=seed, lengths=(1, int(rng.integers(8, 31))), queue_p=float(rng.uniform(0, 0.5)),
             l1_p=float(rng.uniform(0, 1)), place_p=float(rng.uniform(0, 0.03)), rogue=rogue, cheat=bool(rng.random() < 0.3))
    vr = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(81,))))   # its voice (P3's
    con = _conduct(seed=seed, stage=1 + seed % 2, imperfect=True, motion=m, ledger=_TapLedger(),   # eleventh round): her
                   voice=_TimedVoice(), scaffold=seed % 3 != 0)    # sentences as her voice makes them, its babble and turns;
                                                  # the words channel labelling her lines in two lives of three (twelfth)
    kid = _Kid(kind, seed=seed + 1000, wander=float(rng.uniform(0.15, 0.35)), babble=float(vr.choice([0.0, 0.005, 0.02])),
               turns=float(vr.choice([0.0, 0.3, 0.7])))
    probes, asks, t, k = {}, {}, 100, 0
    while t < n - 300:
        if rng.random() < 0.3:
            probes[t] = ("name", None, None, None)
        else:
            a, b = PROP_PAIRS[int(rng.integers(len(PROP_PAIRS)))]
            stale = rng.random() < 0.2
            probes[t] = ("place", a, b, {a: f"{a}@{'old' if stale else k}", b: f"{b}@{'old' if stale else k}"})
        k += 1
        asks[t + int(rng.integers(90, 160))] = [("ask_where", dict(o="ball")), ("ask_give", dict(o="cup")), ("call", {}),
                                                ("ask_where", dict(o="duck"))][int(rng.integers(4))]
        t += int(rng.integers(200, 351))
    lf = _Life(con, kid, probes, asks, TOYS, m, float(rng.choice([0.0, 0.002, 0.01])), seed)
    return lf, kind, n


def _trial_of(rows, i0, since=None):
    """the trial among ledger rows from i0: -> (result, why's first clause, its end's tick, form, on, off), or ("dropped",)."""
    tr = next((r for r in rows[i0:] if r["ev"] == "trial"), None)
    if tr is None:
        return ("dropped",)
    oc = next((r for r in rows[i0:] if r["ev"] == "trial_outcome"), None)
    return (oc["result"], (oc["why"] or "").split(":")[0], oc["t"], tr["form"], oc.get("on"), oc.get("off"))


def _flipped(x, y):
    """the base trial x and its run with the other thing named y: the same by every rule (A60b: a pair's scored trial with its
    window's ticks on each thing the same, on and off turned over; the name's, its face's ticks the same)."""
    if x[0] == "scored" and x[3] != "name":
        return y[:4] == x[:4] and (y[4], y[5]) == (x[5], x[4])
    return y == x


def _flip_life(seed, n=5000):
    """a property life run whole (-> the _Life, its kind) with, for each trial, a snapshot on the first tick of its settle (its
    draw made, its sentence not yet said), restored into a fresh conduct, motion and child with the draw turned over (the other
    thing named; the name test: its name for its foil, its foil for its name) and run until that trial ends; each trial's end
    in the life and with the other named kept in lf.flips [(tick, base, flipped)]."""
    lf, kind, n = _prop_life(seed, n)
    open_, pending = None, []
    for t in range(n):
        tr = lf.con.trial
        if open_ is not None and (tr is None or tr["since"] != open_[0]):
            pending.append((open_[1], open_[2], _trial_of(lf.con.ledger.rows, open_[2]["rows"])))
            open_ = None
        if open_ is None and tr is not None and tr["phase"] == "settle":
            open_ = (tr["since"], t, lf.snapshot())
        lf.step(t)
    lf.flips = []
    for t0, sn, base in pending:
        fresh, _k2, _n2 = _prop_life(seed, n)
        fresh.motion.load_state(sn["motion"])
        st = json.loads(json.dumps(sn["con"]))
        tr = st["trial"]
        if tr["form"] == "name":
            tr["name"] = not tr["name"]                  # (before P3's eleventh round her foil was kept only when drawn)
            tr["said"] = LX.NAME if tr["name"] else (tr.get("foil") or K.NAME_FOILS[0])
        else:
            tr["target"], tr["distractor"] = tr["distractor"], tr["target"]
        fresh.con.load_state(st)
        fresh.con.ledger.rows = [dict(r) for r in lf.con.ledger.rows[:sn["rows"]]]
        fresh.kid = sn["kid"]
        fresh.rng.bit_generator.state = sn["rng"]
        fresh.skipped, fresh.face = set(sn["skipped"]), set(sn["face"])
        for u in range(t0, n):
            ft = fresh.con.trial
            if ft is None or ft["since"] != tr["since"]:
                break
            fresh.step(u)
        lf.flips.append((t0, base, _trial_of(fresh.con.ledger.rows, sn["rows"])))
    return lf, kind


def _prop_check(lf):
    """the property on one life -> (breaks, counted): no trial scored with anything but her mouth moving on a tick from its
    sentence to its window's end (the motion's own account, the ticks the world skipped, her face after her own judgments);
    every trial her log voided with a movement on its void tick; the ledger's understood exactly what its trials give by the
    rule as 4.8 states it (_rule_share), never its asks."""
    m, breaks, counted = lf.motion, [], []
    trs = [x for x in _trials(lf.con.ledger) if x["result"] is not None]
    def moved(u, end):                                    # her face from her judgments and frowns (a trial earns none: A60b)
        return m.moved.get(u) or u in lf.skipped or any(f0 <= u <= f0 + K.FACE_COURSE for f0, _kinds, _fr in lf.face_at)
    for x in trs:
        last = x["end"]
        mv = [u for u in range(x["t"], last + 1) if moved(u, last)]
        if x["result"] != "void" and mv:
            breaks.append(("scored with her body moving", x["t"], x["result"], mv[:3], [m.moved.get(u) for u in mv[:3]]))
        if x["result"] == "void" and "her attention log" in (x["why"] or "") and \
                not (moved(x["end"], x["end"]) if "not read" not in x["why"] else x["end"] - 1 in lf.skipped):
            breaks.append(("voided with her body still", x["t"], x["end"], x["why"]))
        if x["result"] == "scored":
            counted.append(x)
    rows = lf.con.ledger.rows                             # its stimuli time-matched (P3's twelfth round): every sentence its
    scaffold = getattr(lf.con, "scaffold", None)          # draw could have given one timeline, as the timed voice says them
    for i, r in enumerate(rows):                          # (10 ticks, the noun from tick 4; the name's 5, from tick 0)
        if r["ev"] != "trial":
            continue
        if ST is None or "stimulus" not in r:
            breaks.append(("no time-matched stimuli", r["t"], r["said"]))
            continue
        words = (r["target"]["word"], r["distractor"]["word"]) if r["form"] != "name" else (LX.NAME,) + tuple(K.NAME_FOILS)
        got = [_timed_tl("name" if r["form"] == "name" else "where", w, scaffold) for w in words]
        tls = {json.dumps(g[0], sort_keys=True) for g in got if g is not None}
        tl, slot = (json.loads(next(iter(tls))), got[0][1]) if tls else (dict(ticks=None, words=[[None]]), 0)
        if len(tls) != 1 or None in got or rows[i - 1]["ev"] != "said" or rows[i - 1]["ticks"] != tl["ticks"] or \
                r["open"] - r["t"] != tl["words"][slot][0] or r["stimulus"]["ticks"] != tl["ticks"] or \
                (r["form"] == "name" and scaffold) or "duck" in (r["target"]["id"], (r["distractor"] or {}).get("id")):
            breaks.append(("not one timeline", r["t"], r["said"], rows[i - 1].get("ticks"), r["open"] - r["t"], len(tls)))
    led = lf.con.ledger
    seqs = {}
    for x in trs:                                         # each word's record from the trials' own rows, in their order
        if x["result"] != "scored":                       # (A60b: a void never scored)
            continue
        for key, lst, _a, _b, *lv in x["score"]:
            if lv and lv[0] == 2:                         # (its new-exemplar block: none in these lives)
                continue
            nm = int(lst == "trials")
            seqs.setdefault(key, []).append((nm,) + ((x["on"], x["off"]) if nm or x["form"] == "name" else (x["off"], x["on"])))
    for w, st in led.words.items():
        got = _rule_share(seqs.get(w, []))
        if (st["maps_at"] is not None) != (got is not None):
            breaks.append(("maps not as its trials give", w, st["maps_at"], got, seqs.get(w)))
        if st["understood_at"] is not None and (st["maps_at"] is None or (w in TP.OBJECT_NOUNS and st["kind_at"] is None)):
            breaks.append(("understood without both its levels", w, st["maps_at"], st["kind_at"]))
    return breaks, counted


def _timed_tl(key, word, scaffold):
    """a trial sentence in the suite's timed voice on its form's carrier phrase (P3's fourteenth round), built from the table's
    parts as labs splice theirs (stimuli.parts, splice) -> (its timeline, its test word's index), or None (no stimulus)."""
    pt = ST.parts(key, word) if hasattr(ST, "parts") else None      # (none before P3's fourteenth round)
    if pt is None:
        return None
    v = _TimedVoice()
    pre, own, tag = (v.clip(x[0], x[1], emphasis=x[2], shape=x[3]) for x in (pt["pre"], pt["own"], pt["tag"]))
    c = ST.splice(pre, own, tag, pt["at"], pt["own_from"], pt["n_own"], pt["gain"])
    ws = [w for w, _a, _e in c.words]
    return ST.timeline(c.words, len(c.pcm), c.pcm, scaffold, ws.index(word), tag_at=c.spliced["tag_at"]), ws.index(word)


def test_trial_property():
    """the property over 49 random lives (7 of each kind of child: a follower of her gaze, of her hands, a side's and a toy's
    favourite, a voice-turner, a wanderer, a knower), each 5,000 ticks with a W2-like motion (queued acts, L1's glances, breaches
    of its contract while she holds still, a third of them by a motion that knew the target), ticks the world skips, her day
    plan's probes (place and name) and everyday asks between them, her sentences as her voice says them (the timed voice; the
    words channel labelling her lines in two lives of three: P3's twelfth round, every trial's sentences one timeline, a pair
    with no time-matched stimulus or the name while the channel runs never tried): no trial is scored with anything but her
    mouth moving from its sentence to its window's end (the motion's own account, not her log), none voided by her log with
    her body still; each word's level 1 ("maps") is exactly what its trials give (the rule as stated), never its asks, and no
    word understood without both its levels; the
    non-knowers' trials whose looking went more to the named thing sit at chance, the knowers' far above (A60b); each
    non-knower's trial run again with the other thing named ends the same, its window's ticks on each thing the same; the
    target's side and its name against a foil drawn near half and half; a life replays exactly, whole and from snapshots
    taken in each phase of a trial."""
    assert _has_trials() and perm_test is not None, "no formal trial scored by the proportion of looking (A60b)"
    assert hasattr(Ledger, "maps"), "one level of 'understood' only (A60b's two levels)"
    lives, flips = [], []
    for seed in range(49):                           # (7 of each since P3's fourteenth round: 6 left a voice-turner 14 scored pair
        if PROP_KIDS[seed % len(PROP_KIDS)] == "knower":          # trials once her sentences took their tag)
            # (its looks depend on the word: nothing to turn over)
            lf, kind, n = _prop_life(seed)
            lives.append((lf.run(0, n), kind))
        else:
            lf, kind = _flip_life(seed)
            lives.append((lf, kind))
            flips += [(seed, kind) + f for f in lf.flips]
    bad = [f for f in flips if not _flipped(f[3], f[4])]
    assert not bad, (len(bad), len(flips), bad[:4])
    all_breaks, by_kind = [], {}
    counts = dict(trials=0, void=0, void_log=0, void_trunk=0, few=0, asks=0, met_asks=0, stale=0, sounded=0, no_stim=0,
                  channel=0)
    for lf, kind in lives:
        br, counted = _prop_check(lf)
        all_breaks += [(kind,) + b for b in br]
        by_kind.setdefault(kind, []).extend(counted)
        trs = [x for x in _trials(lf.con.ledger) if x["result"] is not None]
        counts["trials"] += len(trs)
        counts["void"] += sum(x["result"] == "void" for x in trs)
        counts["void_log"] += sum(x["result"] == "void" and "her attention log" in (x["why"] or "") for x in trs)
        counts["void_trunk"] += sum(x["result"] == "void" and "her trunk" in (x["why"] or "") for x in trs)
        counts["few"] += sum(x["result"] == "void" and "fewer than" in (x["why"] or "") for x in trs)
        counts["sounded"] += sum(any(u in lf.voiced for u in range(x["t"], x["end"] + 1)) for x in trs)
        counts["stale"] += sum(x["form"] == "place" and not [s_ for s_ in x["score"] if s_[1] == "trials"] for x in trs)
        counts["asks"] += sum(r["ev"] == "ask" for r in lf.con.ledger.rows)
        counts["met_asks"] += sum(r["ev"] == "outcome" and r["result"] == "met" for r in lf.con.ledger.rows)
        counts["no_stim"] += sum("no time-matched stimulus" in r[2] for r in lf.con.fast.refused if r[1].startswith("trial:"))
        counts["channel"] += sum("in their channel" in r[2] for r in lf.con.fast.refused if r[1].startswith("trial:"))
    assert not all_breaks, all_breaks[:5]
    stats = {}
    for kind, xs in sorted(by_kind.items()):
        k, n = _rate(xs)
        stats[kind] = (k, n)
        if kind != "knower":                              # (a voice-turner's pair trials mostly void: its looking on her face)
            assert n >= 10 and _two_sided(k, n) > 0.001, (kind, k, n)
    kc, nc = sum(v[0] for kd, v in stats.items() if kd != "knower"), sum(v[1] for kd, v in stats.items() if kd != "knower")
    assert nc >= 120 and _two_sided(kc, nc) > 0.01, (kc, nc)
    k, n = stats["knower"]
    assert k / n > 0.7 and _btail(k, n) < 0.001, stats["knower"]     # (a wanderer too: its looks between its known ones)
    rows = [x for lf, _ in lives for x in _trials(lf.con.ledger) if x["form"] != "name" and x["sides"]]
    left = sum(x["target"]["id"] == x["sides"][0] for x in rows)
    names = [x for lf, _ in lives for x in _trials(lf.con.ledger) if x["form"] == "name"]
    nm = sum(x["said"] == LX.NAME for x in names)
    assert _two_sided(left, len(rows)) > 0.001 and _two_sided(nm, len(names)) > 0.001, (left, len(rows), nm, len(names))
    assert counts["void_log"] > 0 and counts["void_trunk"] > 0 and counts["met_asks"] > 0 and counts["stale"] > 0 and \
        counts["sounded"] > 0 and counts["no_stim"] > 0 and counts["channel"] > 0, counts
    # determinism: a life twice from its seed; snapshots in each phase of a trial restored into a fresh conduct, motion and child
    lf2, _k, n2 = _prop_life(5)
    lf2.run(0, n2)
    assert lf2.con.ledger.digest == lives[5][0].con.ledger.digest and lf2.con.ledger.rows == lives[5][0].con.ledger.rows
    snaps = _prop_snapshots(23) + _prop_snapshots(16)       # (a side's favourite in stage 2, a wanderer in stage 1)
    print(f"63 the property over {len(lives)} lives (7 of each child, 5000 ticks, W2-like motions breaking their contract "
          f"while she holds still, her trunk among them, ticks skipped; her trial sentences as her voice says them since the "
          f"twelfth round, on one carrier phrase since the fourteenth, 12 ticks with the noun from tick 4, the name's 12 from "
          f"tick 5; the words channel labelling her lines in two "
          f"lives of three; the child babbling and taking turns after her sound stops): {counts['trials']} trials "
          f"({counts['sounded']} with its voice in the sentence or window; "
          f"{counts['void']} void, never scored: "
          f"{counts['void_log']} by her log, {counts['void_trunk']} of them by her trunk, {counts['few']} with fewer than 4 "
          f"of its window's ticks on either thing; {counts['stale']} of a place already shown 3 times, scored as no "
          f"test), none scored with her body moving (the motion's own account) and none voided by her log with it still; "
          f"each word's level 1 (maps) exactly as the trials give in every life, never by her {counts['asks']} everyday asks "
          f"({counts['met_asks']} met); its looking more on the named thing (A60b): "
          f"{', '.join(f'{kd} {a}/{b}' for kd, (a, b) in stats.items())}"
          f" (each non-knower two-sided p > 0.001; together {kc}/{nc}, p > 0.01); the target on the left {left}/{len(rows)}, "
          f"its name {nm}/{len(names)}; every trial's sentences one timeline in the timed voice, the words channel's too where "
          f"it ran; never a pair with 'duck' ({counts['no_stim']} dropped: no time-matched stimulus) nor the name while the "
          f"channel ran ({counts['channel']} dropped); each of {len(flips)} trials of the non-knowers run again from its "
          f"settle with the other thing "
          f"named ends the same by every rule (its window's ticks on each thing the same, on and off turned over); a "
          f"life replays exactly, and {snaps} snapshots (bring, settle, window) restored continue exactly")


def _prop_snapshots(seed, n=1800):
    """a property life run whole with a snapshot taken in each phase of a trial (bring, settle, said: the third tick in it); each
    restored into a fresh conduct, motion and child, with the life's own streams, continues exactly: the ledger's rows and
    digest, her lines and judgments -> the number restored."""
    lf, _kind, _n = _prop_life(seed, n)
    snaps, run_in = {}, 0
    for t in range(n):
        tr = lf.con.trial
        ph = tr["phase"] if tr is not None else None
        run_in = run_in + 1 if ph is not None and ph == getattr(lf, "_ph", None) else 1
        lf._ph = ph
        if ph is not None and ph not in snaps and run_in == 3:
            snaps[ph] = (t, lf.snapshot())
        lf.step(t)
    base_rows, base_said, base_judg = lf.con.ledger.rows, [(t, ln.text) for t, ln in lf.said], lf.judg
    for ph, (t0, sn) in sorted(snaps.items()):
        fresh, _k2, _n2 = _prop_life(seed, n)
        fresh.motion.load_state(sn["motion"])
        fresh.con.load_state(sn["con"])
        fresh.con.ledger.rows = [dict(r) for r in base_rows[:sn["rows"]]]
        fresh.kid = sn["kid"]
        fresh.rng.bit_generator.state = sn["rng"]
        fresh.said, fresh.judg = list(lf.said[:sn["said"]]), list(base_judg[:sn["judg"]])
        fresh.skipped, fresh.face = set(sn["skipped"]), set(sn["face"])
        fresh.run(t0, n)
        assert fresh.con.ledger.rows == base_rows and fresh.con.ledger.digest == lf.con.ledger.digest, \
            (ph, t0, next((i for i, (a, b) in enumerate(zip(fresh.con.ledger.rows, base_rows)) if a != b), None))
        assert [(t, ln.text) for t, ln in fresh.said] == base_said and fresh.judg == base_judg, (ph, t0)
    assert set(snaps) == {"bring", "settle", "said"}, sorted(snaps)
    return len(snaps)

def _run_to(con, n, target=None, t0=0, events=None, seen_=TOYS, face_near=(), near_of=None, token=None, tract=None):
    """ticks t0..t0+n-1 with a still stub motion and a scripted child: target(t, con) its reading; -> [(t, Say)]."""
    out = []
    for t in range(t0, t0 + n):
        things = seen_ if near_of is None else tuple(seen1(x.id, x.name, x.colour, x.on, x.child_sees, x.child_can_reach,
                                                           near_of.get(x.id, x.near)) for x in seen_)
        p = P(t, child_target=None if target is None else target(t, con), seen=things, face_near=face_near,
              events=tuple((events or {}).get(t, ())))
        out.append((t, con.tick(t, p, token=None if token is None else token.get(t),
                                tract=None if tract is None else tract.get(t))))
    return out


def _blank_word():
    """a word's record as the ledger starts it (ledger._blank), for an older save written here."""
    from body.sim.lang import ledger as LG                                 # noqa: PLC0415
    return LG._blank()


def test_trial_low_items():
    """the protocol's edges: her display (the two in its view, each beyond NEAR_DEG of the other and of her face as the world says;
    not said: fail-closed, dropped after TRIAL_WAIT); the present act refused: dropped; its pain after the sentence: void, before:
    dropped; the child's voice over its sentence: said whole, no stop, no frown, not void (P3's eleventh round); a word not hers:
    the trial dropped by the line check; the combination only once both its words are understood (A28), its held-out pair
    allowed in its own line alone, the twin's key its colour and word, its yoked trial the original named, its first 3 trials
    scored whichever is named; a place shown 3 times in trials scores no test on the 4th (A28); an ended act whose
    one report the conduct never read no longer holds her unstill for good (status asked of the motion); her log's reader for
    P4's probes (still_over); an older save's log and ledger restored fail-closed (asks never understood)."""
    assert _has_trials(), "no formal trial in her conduct"
    # the display: near not given, the two near each other, one beside her face: never said, dropped after TRIAL_WAIT
    whys = {}
    for name, kw in (("unknown", dict(near_of={"cup": None})), ("near", dict(near_of={"cup": ("duck",), "duck": ("cup",)})),
                     ("face", dict(face_near=("cup",))), ("face unknown", dict(face_near=None))):
        con = _pair_con()
        con.probe("place", "cup", "duck", new={"cup": "c", "duck": "d"})
        said = [s.line.text for t, s in _run_to(con, K.TRIAL_WAIT + 20, **kw) if s.line is not None]
        whys[name] = [r[2] for r in con.fast.refused if r[1] == "trial:place"]
        assert not [x for x in said if x.startswith("where")] and whys[name] and "not said within" in whys[name][-1], \
            (name, said, whys[name])
    # the present act refused: dropped, never said
    class _Refuse(C.StubMotion):
        def status(self, i, tick):
            return "refused" if self.acts[i][2] == "present" and tick > self.acts[i][1] else super().status(i, tick)

        def report(self, tick):
            r = super().report(tick)
            return dict(r, acts={i: self.status(i, tick) for i in r["acts"]})
    con = _pair_con(motion=_Refuse())
    con.probe("place", "cup", "duck", new={"cup": "c", "duck": "d"})
    _run_to(con, 60)
    assert con.trial is None and not _trials(con.ledger) and "refused" in con.fast.refused[-1][2], con.fast.refused[-1:]
    # its pain: after the sentence, void; before it, dropped
    con = _pair_con()
    con.probe("place", "cup", "duck", new={"cup": "c", "duck": "d"})
    got = _run_to(con, 60, events={58: (("pain", None),)})
    assert _trials(con.ledger)[0]["result"] == "void" and "pain" in _trials(con.ledger)[0]["why"], _trials(con.ledger)
    con = _pair_con()
    con.probe("place", "cup", "duck", new={"cup": "c", "duck": "d"})
    _run_to(con, 60, events={30: (("pain", None),)})
    assert not _trials(con.ledger) and "pain" in con.fast.refused[-1][2], con.fast.refused[-1:]
    # the child's voice over its sentence before its word (P3's eleventh round): she says it whole, no stop and no frown, and
    # the trial is not void (nothing the child's voice does voids one; the tenth round voided it as cut before its word)
    act = np.array([4, 4, 2, 4, 2, 2, 2, 4, 2, 0])
    tr_ = T.Tract(1)
    snd = {t: tr_.tick(act if 56 <= t < 58 else None) for t in range(90)}
    con = C.Conduct(seed=1, stage=2, imperfect=False, ledger=_TapLedger(), transcriber=Transcriber(None))
    _no_sets(con, ("duck", "ball", "cup", "block", "block_red"))
    con.probe("place", "cup", "duck", new={"cup": "c", "duck": "d"})
    got = _run_to(con, 90, tract=snd, target=lambda t, c: "cup" if c.trial is not None and c.trial["phase"] == "said" else None)
    over = _trials(con.ledger)
    voiced = [r for r in con.ledger.rows if r["ev"] == "voiced" and r["text"].startswith("where")]
    assert over and over[0]["t"] < 56 < over[0]["open"] and over[0]["result"] == "scored" and \
        not [r for r in con.ledger.rows if r["ev"] == "cut"] and not [s for t, s in got if s.cut or s.frown] and \
        voiced and voiced[0]["said"] == TP.words(voiced[0]["text"]), (over, voiced)
    # a word not hers: dropped by the line check (the rattle before it is hers)
    rattle = TOYS + seen(("rattle", "rattle", "purple", "mat", True))
    con = _pair_con()
    for k in range(6):
        con.probe("place", "rattle", "duck", new={"rattle": f"r{k}", "duck": f"d{k}"})
        _run_to(con, 200, t0=200 * k, seen_=rattle)
    tr_rows = _trials(con.ledger)
    drops = [r for r in con.fast.refused if r[1] == "trial:place" and "line check" in r[2]]
    assert not tr_rows and len(drops) == 6, (tr_rows, drops)     # whichever is named: never only the sayable one
    # the combination: only once both its words are understood alone (A28); its pair allowed in its line alone
    twins = TOYS + seen(("ball_blue", "ball", "blue", "mat", True))
    con = C.Conduct(seed=3, stage=2, imperfect=False, ledger=_TapLedger(), vocab=VOCAB + ("blue", "red"))   # (its draws name
    # each twin within the first 3 trials)
    _no_sets(con)
    con.probe("combination", "ball_blue", "ball", new={})
    _run_to(con, 10, seen_=twins)
    assert "understood alone" in con.fast.refused[-1][2], con.fast.refused[-1:]
    for w in ("blue", "ball"):                             # (the noun understood at both levels, A60b; the colour its one)
        _ledger_knows(con.ledger, w, 12, kind=w in TP.OBJECT_NOUNS)
    lines, keys = [], []
    for k in range(8):
        con.probe("combination", "ball_blue", "ball")
        t0 = 1000 + 200 * k
        for t, s in _run_to(con, 200, t0=t0, seen_=twins,
                            target=lambda t, c: (c.trial["target"] if c.trial and c.trial["phase"] == "said" and
                                                 t >= c.trial["onset"] + 5 else None)):
            if s.line is not None and s.line.intent == "trial_combo":
                lines.append(s.line.text)
    trs = [x for x in _trials(con.ledger) if x["form"] == "combination"]
    blue = [x for x in trs if x["target"]["id"] == "ball_blue"]
    red = [x for x in trs if x["target"]["id"] == "ball"]
    assert blue and red and {x["target"]["word"] for x in blue} == {"blue ball"} and \
        {x["target"]["word"] for x in red} == {"red ball"}, [x["target"] for x in trs]
    assert blue[0]["score"] == [["blue ball", "trials", 1, 0, 1]] and red[0]["score"] == [["blue ball", "yoked", 0, 1, 1]], \
        ([x["score"] for x in blue], [x["score"] for x in red])
    assert "where is the blue ball? see?" in lines and "where is the red ball? see?" in lines, lines
    assert not TP.check("the blue ball.", con.fast.vocab, None, P(0, seen=twins), ("ball_blue",), con.fast.held)[0]
    assert ("blue", "ball") in [tuple(h) for h in con.fast.held], con.fast.held
    assert all(x["score"] for x in trs[:K.NOVEL_PRESENTATIONS]) and not [x for x in trs[K.NOVEL_PRESENTATIONS:] if x["score"]] \
        and all(x["items"] == ["blue ball"] for x in trs), [(x["target"]["word"], x["score"], x["items"]) for x in trs]
    # a place shown 3 times in trials: the 4th scores no test (A28)
    con = _pair_con()
    for k in range(5):
        con.probe("place", "cup", "duck", new={"cup": "cup@table", "duck": f"duck@{k}"})
        _run_to(con, 200, t0=200 * k)
    sc = [(x["target"]["id"], x["score"]) for x in _trials(con.ledger)]
    assert len(sc) == 5 and con.ledger.presented("cup@table") == 5 and \
        all(not [x for x in s_ if x[0] == "cup"] for tid, s_ in sc[3:]), sc
    # an ended act whose one report she never read no longer holds her unstill (the motion's own status asked)
    con = _pair_con()
    con.request("show", o="duck")
    _run_to(con, 3)
    for t in range(3, 3 + C.STUB_TICKS["show"] + 2):
        con.motion.report(t)                                  # the world moved on; the conduct was not ticked
    _run_to(con, 3, t0=3 + C.STUB_TICKS["show"] + 2)
    assert not [a for a in con.acts_open if a[1] == "show"], con.acts_open
    # a trial's sentence is never said on request (Claude's rows, L3): only in its trial
    con = _pair_con()
    con.request("trial_where", o="ball")
    con.request("trial_name")
    said = [s.line for t, s in _run_to(con, 10) if s.line is not None]
    assert not said and all("only in its trial" in r[2] for r in con.fast.refused[-2:]), (said, con.fast.refused[-2:])
    # her log's reader for P4's probes
    con = _pair_con()
    _run_to(con, 30)
    assert con.still_over(5, 29) is None and "not in her attention log" in con.still_over(-1, 5), con.still_over(-1, 5)
    con.motion.glance("ball", 30, 3)
    _run_to(con, 5, t0=30)
    assert "eyes at ball" in (con.still_over(25, 34) or ""), con.still_over(25, 34)
    # an older save: its log fail-closed, its asks never understood
    con = _pair_con()
    _run_to(con, 5)
    st = json.loads(json.dumps(con.state(), default=_np))
    st["attn"][-1].update(eyes=["child"], head=["ball", "cup"], left=[], right=["?"], face=[], voice=["mama"])
    st["ledger"]["words"]["ball"] = dict(heard=3, said=3, asks=[1] * 10, base=[0] * 10, says={"tract": [], "token": []},
                                         echoes={"tract": 0, "token": 0}, exact={"tract": 0, "token": 0},
                                         approx={"tract": 0, "token": 0}, understood_at=100, says_at={})
    st["ledger"]["words"]["duck"] = dict(_blank_word(), trials=[1] * 10, yoked=[0] * 10, understood_at=200)
    st["ledger"]["words"]["duck"].pop("seq")               # the ninth round's save: trials, no order, "understood" by its rule
    con2 = _pair_con()
    con2.load_state(st)
    e = con2.attn[-1]
    assert e["eyes"] == "child" and e["head"] == C.UNNAMED and e["left"] is None and e["face"] is None, e
    assert not con2.ledger.understood("ball") and con2.ledger.standing("ball")["asks"] == [1] * 10, con2.ledger.standing("ball")
    assert not con2.ledger.understood("duck") and con2.ledger.standing("duck")["seq"] == [], con2.ledger.standing("duck")
    # a save from before P3's twelfth round (its trial's sentences never made one timeline): in its window, void on its next
    # tick; in its settle, dropped, never said
    oc = {}
    for phase in ("said", "settle"):
        con = _pair_con()
        con.probe("place", "cup", "duck", new={"cup": "c", "duck": "d"})
        t = 0
        while con.trial is None or con.trial["phase"] != phase:
            con.tick(t, P(t, child_target=None))
            t += 1
        st = json.loads(json.dumps(con.state(), default=_np))
        for k_ in ("timeline", "lines"):
            st["trial"].pop(k_, None)
        con2 = _pair_con()
        con2.load_state(st)
        con2.tick(t, P(t, child_target=None))
        oc[phase] = [r for r in con2.ledger.rows if r["ev"] == "trial_outcome"] or \
            [r for r in con2.fast.refused if r[1] == "trial:place"]
        assert con2.trial is None and len(oc[phase]) == 1 and "time-matched" in str(oc[phase][0]), (phase, oc[phase])
    assert oc["said"][0]["result"] == "void", oc
    print(f"64 the protocol's edges: never said while the world does not say what lies near the two, or they lie within "
          f"{K.NEAR_DEG:g} degrees of each other or of her face (dropped after {K.TRIAL_WAIT} ticks, logged); the present act "
          f"refused: dropped; its pain after the sentence: void, before: dropped; the child's voice over its sentence before its "
          f"word: said whole, no stop, no frown, scored; a word "
          f"not hers as either thing: dropped by the line check whichever is named ({len(drops)} of 6; a drop after the draw "
          f"would have named the sayable thing every time); the combination refused until 'blue' and 'ball' are "
          f"understood alone (A28), "
          f"then 'where is the blue ball?' (its test) and 'where is the red ball?' (the blue ball's yoked trial) by her "
          f"stream, the pair held out of every other line, "
          f"its first {K.NOVEL_PRESENTATIONS} trials its tests whichever is named (the pair displayed in each); a place shown "
          f"in trials counts for 3; a trial's sentence never "
          f"said on request; an ended act whose report "
          f"she never read no longer holds her unstill; still_over for P4's probes; an older save's log and asks restored "
          f"fail-closed, the ninth round's save its 'understood' not kept (its trials' order unknown), and one from before the "
          f"trial's stimuli were time-matched void in its window and dropped in its settle")


def _led_life(n, rng, look, name=False, form="place", led=None, key="ball", other="duck", fresh_other=False, knower=None,
              level=1):
    """one child's n scored trials straight through a ledger as the conduct records them (its thing a fresh never-taught item,
    which of the two is named a fair coin from rng; the lead's decision A60b): look(j, last) its share of looking at key's thing
    (the name: at her face) on trial j, which never sees the word said now: its window's ticks on either thing drawn (4 to 22,
    as it happens to look), key's thing's among them Binomial with that share (the name: of the window's 22); last: the trial
    before, (named, looked more at the named thing): what a child could have read from her after it (since A60b she gives
    nothing for a trial: a child that reads it anyway is no worse). fresh_other: the other thing fresh too (its word scored as
    well). knower: (share when named, share when not) instead. level 2: its kind's new exemplars (form "exemplar"), its word or
    a never-told foil said (the lead's decision after 200e57a), the share its new exemplar's of the looking at it and the two
    things beside it ("duck" and "cup"). -> the ledger (a new one unless given)."""
    led = Ledger() if led is None else led
    t, last = 1000, None
    for j in range(n):
        named = int(rng.random() < 0.5)
        sh = (knower[0] if named else knower[1]) if knower is not None else look(j, last)
        if name:
            a = int(rng.binomial(22, sh))
            said = LX.NAME if named else K.NAME_FOILS[0]
            tid = led.trial(t, "name", dict(id=None, word=said), None, t + 5, (t + 7, t + 28), said,
                            score=[[LX.NAME, "trials" if named else "yoked", 1, 0]])
            led.trial_outcome(t + 28, tid, "scored", "a test", on=a, off=22 - a)
            last = (named, int(named and a >= K.HOLD))
        elif level == 2:                                  # its new exemplar the target, its word or a never-told foil said
            m = int(rng.integers(4, 23))
            a = int(rng.binomial(m, sh))                  # its ticks on its new exemplar, the rest on the two beside it
            said = key if named else getattr(K, "NOUN_FOILS", {}).get(key, "zeb")
            tid = led.trial(t, "exemplar", dict(id=f"{key}_{j}", word=key), dict(id="duck", word="duck"), t + 9,
                            (t + 11, t + 32), said, score=[[key, "trials" if named else "yoked", 1, 0, 2]],
                            others=[dict(id="cup", word="cup")])
            led.trial_outcome(t + 32, tid, "scored", "a test", on=a, off=m - a)
            last = (named, int(a > m - a))
        else:
            m = int(rng.integers(4, 23))
            a = int(rng.binomial(m, sh))                  # its ticks on key's thing, the rest on the other
            tg, ds = (key, other) if named else (other, key)
            sc = [[key, "trials", 1, 0] if named else [key, "yoked", 0, 1]]
            if fresh_other:
                sc.append([other, "yoked", 0, 1] if named else [other, "trials", 1, 0])
            on, off = (a, m - a) if named else (m - a, a)
            tid = led.trial(t, form, dict(id=tg, word=tg), dict(id=ds, word=ds), t + 9, (t + 11, t + 32), tg, score=sc)
            led.trial_outcome(t + 32, tid, "scored", "a test", on=on, off=off)
            last = (named, int(on > off))
        t += 100
    return led


H0_NAME = {                          # children whose looking at her face never depends on the word said now (the name test)
    "rising 0.1-0.9": lambda j, last: 0.1 + 0.8 * j / 47,
    "turns 0.3": lambda j, last: 0.3, "turns 0.5": lambda j, last: 0.5, "turns 0.7": lambda j, last: 0.7,
    "step 0.3-0.6": lambda j, last: 0.3 if j < 24 else 0.6, "falling 0.9-0.1": lambda j, last: 0.9 - 0.8 * j / 47,
    "turns again once smiled at": lambda j, last: 0.9 if last == (1, 1) else 0.3,
}
H0_WORD = {                          # children whose looking never depends on the word said now (a pair's trials)
    "favourite 0.6": lambda j, last: 0.6, "favourite 0.7": lambda j, last: 0.7, "favourite 0.95": lambda j, last: 0.95,
    "shift 0.5-0.8": lambda j, last: 0.5 if j < 24 else 0.8, "shift 0.5-0.85": lambda j, last: 0.5 if j < 24 else 0.85,
    "rising 0.2-0.95": lambda j, last: 0.2 + 0.75 * j / 47,
    "repeats what was smiled at": lambda j, last: 0.5 if last is None else (0.9 if last == (1, 1) or last == (0, 0) else 0.1),
    "looks where the last word sent it": lambda j, last: 0.5 if last is None else (0.9 if last[0] else 0.1),
}


def test_understood_controlled():
    """the rule that turns trials into "understood" (4.8, the lead's decision A60b): the permutation test over the draw labels is
    exact for any child whose looking does not depend on the word said now, whatever its favourites, drift or learning from her
    smiles (the named thing is a fresh fair coin each trial, so its labels are exchangeable): at one fixed test it rejects at
    most at its level; a life's tests (at 12, 24, 48, ... registered trials, the j-th at 0.01 / 2^j) together at most at 0.01,
    where the rule before re-tested its window of 20 after every trial at 0.05 and was reached by a fair coin's trials 23-63% of
    the time (C64); section 12's flip test at its size; the M6 exemplar test testable with A53's 8 exemplars a noun, its word
    against its never-told foil (the lead's decisions after 200e57a and 67741fd); knowers pass; a child that knows only the
    other word is credited at level 1 (the yoked comparison's own limit, disclosed) and not at level 2, against the foil.
    Through her conduct too: a voice-turner whose turning rises, and a favourite that shifts, at most at the rule's rate. The two levels (A60b): a
    child keyed to its trained thing maps and is not understood; a knower of the kind is. (Level 1 here is maps.)"""
    assert perm_test is not None and hasattr(K, "UNDERSTOOD_FIRST"), "no permutation test over the draw labels (A60b)"
    assert hasattr(Ledger, "maps"), "one level of 'understood' only (A60b's two levels)"
    assert hasattr(K, "NOUN_FOILS") and hasattr(K, "KIND_FIRST"), \
        "level 2 as 67741fd built it: a block of 12, its foil not heard as often as its noun, beside familiar things (the " \
        "lead's decisions after 67741fd)"
    kids = [(label, look, True) for label, look in H0_NAME.items()] + [(label, look, False) for label, look in H0_WORD.items()]
    # at one fixed test (its 24th trial, at 0.01): the permutation test rejects at most at its level, each kind of child
    fixed, reps = {}, 1500
    for i, (label, look, name) in enumerate(kids):
        rng = np.random.default_rng(40 + i)
        rej = 0
        for _ in range(reps):
            led = _led_life(24, rng, lambda j, last, lk=look: lk(2 * j, last), name=name)
            rej += perm_test(led.words[LX.NAME if name else "ball"]["seq"], 0.01)["p"] < 0.01
        fixed[label] = rej / reps
        assert fixed[label] <= 0.01 + 3 * math.sqrt(0.01 * 0.99 / reps), (label, fixed[label])
    # a life of 48 trials, its tests at 12, 24 and 48 (0.005, 0.0025, 0.00125): at most 0.01 of lives, each kind of child
    rates, lives = {}, 300
    bound = sum(0.01 / 2 ** j for j in (1, 2, 3))
    for i, (label, look, name) in enumerate(kids):
        rng = np.random.default_rng(10 + i)
        rates[label] = sum(_led_life(48, rng, look, name=name).maps(LX.NAME if name else "ball")
                           for _ in range(lives)) / lives
        assert rates[label] <= bound + 3 * math.sqrt(bound * (1 - bound) / lives), (label, rates[label], bound)
    # a child that knows only the other word: on the duck whenever "duck" is said, its share 1/2 when "ball" is: credited to
    # "ball" at level 1 (disclosed: the yoked comparison's limit); at level 2, on its new exemplar by exclusion after "ball" and
    # after a never-told foil alike (0.8 each), understood no more often than level 2's levels allow
    rng = np.random.default_rng(5)
    other = sum(_led_life(24, rng, None, knower=(0.5, 0.0)).maps("ball") for _ in range(100)) / 100
    assert other > 0.9, other
    other2 = 0
    for _ in range(300):
        led = _led_life(24, rng, None, knower=(0.5, 0.0))
        _led_life(48, rng, None, knower=(0.8, 0.8), led=led, level=2)
        other2 += led.understood("ball")
    # section 12's flip test at 0.01, each trial once, over 40 trials of the form: at most at its size
    claims = {}
    for i, (label, name) in enumerate((("turns 0.5", True), ("rising 0.1-0.9", True), ("favourite 0.7", False),
                                       ("shift 0.5-0.85", False), ("repeats what was smiled at", False))):
        look = (H0_NAME if name else H0_WORD)[label]
        rng = np.random.default_rng(70 + i)
        rej = sum(_led_life(40, rng, look, name=name, fresh_other=not name).pooled("name" if name else "place")["p"] <
                  K.CLAIM_P for _ in range(400))
        claims[label] = rej / 400
        assert claims[label] <= K.CLAIM_P + 3 * math.sqrt(K.CLAIM_P * (1 - K.CLAIM_P) / 400), (label, claims[label])
    # the M6 exemplar test (A53: 8 exemplars a noun, each fresh on its first 3 presentations, beside new exemplars of two other
    # kinds, its word or its never-told foil said: the lead's decisions after 200e57a and 67741fd): testable, by the
    # permutation test of its new exemplar's share after its word against after its foil
    led_k, led_n, led_x = Ledger(), Ledger(), Ledger()
    rng = np.random.default_rng(8)
    t = 1000
    for noun in ("ball", "duck", "cup", "block", "car", "bear", "drum", "bottle"):
        for led, sh_named, sh_foil in ((led_k, 0.9, 1 / 3), (led_n, 0.7, 0.7), (led_x, 0.8, 0.8)):   # a knower; a lover of
            for ex in range(8):                                        # the new exemplar; a child that knows only the other
                for _rep in range(3):                                  # words (on it by exclusion after its word or a foil)
                    named = int(rng.random() < 0.5)
                    m_ = int(rng.integers(4, 23))
                    a = int(rng.binomial(m_, sh_named if named else sh_foil))       # its ticks on the new exemplar
                    said = noun if named else getattr(K, "NOUN_FOILS", {}).get(noun, "zeb")
                    tid = led.trial(t, "exemplar", dict(id=f"{noun}{ex}", word=noun), dict(id="mat_toy", word="cup"),
                                    t + 9, (t + 11, t + 32), said, score=[[noun, "trials" if named else "yoked", 1, 0, 2]],
                                    items=[f"{noun}{ex}"], others=[dict(id="mat_toy2", word="car")])
                    led.trial_outcome(t + 32, tid, "scored", "a test", on=a, off=m_ - a)
                    t += 100
    ek, en, ex_ = led_k.pooled("exemplar"), led_n.pooled("exemplar"), led_x.pooled("exemplar")
    assert ek["testable"] and ek["p"] < K.CLAIM_P and en["testable"] and en["p"] > K.CLAIM_P and ex_["p"] > K.CLAIM_P, \
        (ek, en, ex_)
    # knowers pass: its name (0.6 of the window on her face after it, 0.05 after a foil), a word (0.9 of its looking on the
    # named thing), within 24 trials
    rng = np.random.default_rng(9)
    got_n = sum(_led_life(24, rng, None, name=True, knower=(0.6, 0.05)).understood(LX.NAME) for _ in range(100))
    got_w = sum(_led_life(24, rng, None, knower=(0.9, 0.1)).maps("ball") for _ in range(100))
    assert got_n >= 95 and got_w >= 95, (got_n, got_w)
    # the two levels (A60b): a child keyed to its trained thing (a knower of it in place trials, at chance on its kind's new
    # exemplars) maps in every life and is understood no more often than level 2's levels allow (its tests at 24 and 48); a
    # knower of the kind is understood (both levels) within 24 trials of each
    rng = np.random.default_rng(15)
    one, both, kind_k = 0, 0, 0
    for _ in range(300):
        led = _led_life(24, rng, None, knower=(0.9, 0.1))
        _led_life(48, rng, lambda j, last: 0.5, led=led, level=2)
        one += led.maps("ball")
        both += led.understood("ball")
    for _ in range(100):
        led = _led_life(24, rng, None, knower=(0.9, 0.1))
        _led_life(24, rng, None, knower=(0.9, 0.1), led=led, level=2)
        kind_k += led.understood("ball")
    assert one >= 295 and both <= 300 * bound + 3 * math.sqrt(300 * bound * (1 - bound)) and kind_k >= 95, (one, both, kind_k)
    assert other2 <= 300 * bound + 3 * math.sqrt(300 * bound * (1 - bound)), other2
    # through her conduct: a voice-turner whose turning rises (the name test), a favourite that shifts (place trials)
    con_rates = {}
    for label, n_con, lives_con in (("voice rising 0.1-0.9", 48, 20), ("favourite shift 0.5-0.85", 48, 20)):
        hits = sum(_con_life(label, seed, n_con, level1=True) for seed in range(lives_con))
        con_rates[label] = (hits, lives_con)
        assert hits <= 1, (label, hits, lives_con)       # (at most 0.00875 a life: Bin(20, 0.00875) >= 2 has p = 0.013)
    worst = max(rates.values())
    print(f"65 'understood' controlled (A60b): the permutation test over the draw labels at one fixed test (the 24th trial, "
          f"0.01) rejected at most {max(fixed.values()):.3f} over {reps} lives of each of {len(kids)} children whose looking "
          f"does not depend on the word said now ({', '.join(f'{k} {v:.3f}' for k, v in fixed.items())}); a life's tests at "
          f"12, 24 and 48 trials (0.005, 0.0025, 0.00125: {bound:.5f} in all) reached at most {worst:.3f} of {lives} lives "
          f"({', '.join(f'{k} {v:.3f}' for k, v in rates.items())}), where the rule before re-tested its last 20 after every "
          f"trial and a fair coin's trials reached it 23-63% of the time (C64); a child that knows only the other word "
          f"credited at level 1 in {other:.0%} (the yoked comparison's limit, disclosed) and understood (level 2, against a "
          f"never-told foil) in {other2} of 300 lives; section 12's flip test at 0.01 at most "
          f"{max(claims.values()):.3f} over 400 lives of 40 trials; the M6 exemplar test testable with A53's 8 exemplars a "
          f"noun ({ek['n']} trials, its word against its never-told foil: a knower p = {ek['p']:.1e}, a lover of the new "
          f"exemplar p = {en['p']:.2f}, one that knows only the other words p = {ex_['p']:.2f}); knowers "
          f"understood in {got_n}/100 (its name) and mapped in {got_w}/100 (a word) within 24 trials; the two levels: a child "
          f"keyed to its trained thing, at chance on its kind's new exemplars, mapped in {one} of 300 lives and understood in "
          f"{both} (level 2's tests at 24 and 48 of its new-exemplar trials), a knower of the kind understood in {kind_k} "
          f"of 100; through her conduct, "
          f"{'; '.join(f'{k}: {h} of {m} lives' for k, (h, m) in con_rates.items())}")


def _con_life(label, seed, n, keep=None, level1=False):
    """one life through her conduct (the stub motion, still): a voice-turner whose chance of turning after any line rises from 0.1
    to 0.9 over n name tests, or the ball's favourite whose chance of looking at the ball rather than the other thing shown steps
    from 0.5 to 0.85 halfway through n place trials (pairs with the ball, each thing fresh) -> understood (its name, or 'ball');
    the conduct appended to keep when given."""
    con = C.Conduct(seed=seed, stage=2, imperfect=False, ledger=Ledger())
    _no_sets(con, ("duck", "ball", "cup", "block", "block_red"))
    if label.startswith("voice"):
        kid = _Kid("voice", seed=seed + 500, rate=lambda t: 0.1 + 0.8 * min(1.0, t / (n * 80.0)))
        key, form = LX.NAME, "name"
    else:
        kid = _Kid("favourite", seed=seed + 500, wander=0.05, rate=lambda t: 0.5 if t < n * 55 else 0.85)
        key, form = "ball", "place"
    t, k = 0, 0
    while len(con.ledger.words.get(key, {}).get("seq", [])) < n and t < n * 400:
        if con.trial is None and not con.probes:
            if form == "name":
                con.probe("name")
            else:
                b = ("duck", "cup", "block_red")[k % 3]
                con.probe("place", "ball", b, new={"ball": f"ball@{k}", b: f"{b}@{k}"})
            k += 1
        tr = con.trial
        layout = (tr["left"], tr["right"]) if tr and tr.get("left") and tr["phase"] != "bring" else None
        s = con.tick(t, P(t, child_target=kid.look(t, TOYS, layout)))
        if s.line is not None:
            kid.hear(t, s.line.text)
        t += 1
    assert len(con.ledger.words.get(key, {}).get("seq", [])) >= n, (label, seed, t)
    if keep is not None:
        keep.append(con)
    return con.ledger.maps(key) if level1 else con.ledger.understood(key)


def test_name_foils_matched():
    """P3's ninth verifier (finding 6): the name's foils ran longer and louder than its name in her voice ("pip." 5 ticks and
    360 ms; "tib.", "vek.", "jem." 6 ticks, 490-570 ms, 109-176% of its energy), so a child drawn to short or quiet sounds
    would have turned after its name more. Now each foil's "X." in the calling register, emphasized as the name is, matches
    "pip."'s: its ticks equal, its energy within 5%, its loudest 10 ms within 12%, its rise within 10 ms, its F0 within 12%
    (consts.FOIL_MATCH; measured where the engine is present). Always: none within edit distance 1 of any of her 128 words."""
    tol = getattr(K, "FOIL_MATCH", dict(energy=0.05, peak=0.12, rise_ms=10, f0=0.12))
    rows = {}
    if _have_engine():
        tmp = tempfile.mkdtemp()
        try:
            cache = V.VoiceCache(os.path.join(tmp, "v"), server=V.SynthServer(nice=19))
            for w in (LX.NAME,) + tuple(K.NAME_FOILS):
                c = cache.clip(f"{w}.", "calling", emphasis=w)
                rows[w] = _clip_features(c)
            cache.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        ref = rows[LX.NAME]
        for w in K.NAME_FOILS:
            r = rows[w]
            assert r["ticks"] == ref["ticks"] and abs(r["energy"] / ref["energy"] - 1) <= tol["energy"] and \
                abs(r["peak"] / ref["peak"] - 1) <= tol["peak"] and abs(r["rise_ms"] - ref["rise_ms"]) <= tol["rise_ms"] and \
                abs(r["f0"] / ref["f0"] - 1) <= tol["f0"], (w, r, ref)
    words = set(LX.BIRTH_WORDS) | set(TP.GROWTH_WORDS)
    # 129 words: the 128 of the design and the bucket (A126)
    assert len(words) == 129 and all(min(_edit(f, x) for x in words) >= 2 for f in K.NAME_FOILS), \
        [(f, min(words, key=lambda x: _edit(f, x))) for f in K.NAME_FOILS]
    assert len(set(K.NAME_FOILS)) == len(K.NAME_FOILS) and LX.NAME not in K.NAME_FOILS
    for f in K.NAME_FOILS:
        assert not TP.check(f"{f}.", VOCAB)[0] and not TP.check(f"{f}.", VOCAB, source="claude")[0]
    ref = rows.get(LX.NAME)
    print(f"66 the name's foils ({', '.join(K.NAME_FOILS)}) matched to 'pip.' in her voice: " +
          ("; ".join(f"{w} {r['ticks']} ticks, {r['clip_ms']} ms, energy {r['energy'] / ref['energy']:.0%}, loudest "
                     f"{r['peak'] / ref['peak']:.0%}, rise {r['rise_ms']} ms, F0 {r['f0']:.0f} Hz" for w, r in rows.items())
           if rows else "the engine absent here: not measured") +
          "; none within edit distance 1 of her 128 words, none sayable by her or Claude")


def _edit(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def _clip_features(c):
    """a clip's ticks, length, energy, loudest 10 ms frame, the rise to half of it from the clip's start, and its median F0 over
    its loud frames (autocorrelation, 120-500 Hz): the tenth round's measure of a foil against the name."""
    from body.sim.voice.playback import TICK                              # noqa: PLC0415
    x = c.pcm.astype(np.float64)
    fr = np.array([np.sum(x[i:i + 160] ** 2) for i in range(0, len(x) - 160, 160)])
    pk = float(fr.max())
    f0s, n = [], 640
    en = [float(np.sum(x[i:i + n] ** 2)) for i in range(0, len(x) - n, 160)]
    for k, i in enumerate(range(0, len(x) - n, 160)):
        if en[k] < 0.3 * max(en):
            continue
        seg = x[i:i + n] - x[i:i + n].mean()
        ac = np.correlate(seg, seg, "full")[n - 1:]
        lo, hi = V.SR // 500, V.SR // 120
        j = lo + int(np.argmax(ac[lo:hi]))
        if ac[j] > 0.3 * ac[0]:
            f0s.append(V.SR / j)
    return dict(ticks=int(math.ceil(len(x) / TICK)), clip_ms=int(round(len(x) / 16.0)), energy=float(x @ x), peak=pk,
                rise_ms=10 * int(np.argmax(fr > 0.5 * pk)), f0=float(np.median(f0s)) if f0s else float("nan"))


def test_low_items_ninth():
    """P3's ninth verifier's low items 7 and 8: "says" excluded only the ask's own word ("mama" said on answering her call, another
    word to "what is this?", the distractor's word in a trial's window all counted); a trial's line carried its target's id in
    Say.line.refs, out to the world; a probe dropped after its display was no presentation of its items, though her follow-in
    naming may label them there; her attention log had no field for her trunk, so a torso turn or lean never voided a trial.
    Now: a word of the child's begun or ended while an ask or a trial of hers is open (its line to its window's end) is teaching
    only, whatever the word; the trial's line has no refs; a dropped display counts as a presentation (a refused one does not);
    her trunk is in the log ("child" while it faces the child and holds still), a turn, a lean or a trunk left out is movement."""
    mama = LX.WORD_ID["mama"]
    led = _TapLedger()
    con = C.Conduct(seed=41, stage=2, imperfect=False, ledger=led, transcriber=Transcriber(None))
    _no_sets(con, ("duck", "ball", "cup", "block", "block_red"))
    answered = []
    for day in (0, 1):
        k, base = 0, day * K.TICKS_PER_DAY
        while len([a for a in answered if a[0] == day]) < 3 and k < 12:
            t0 = base + 300 * k
            con.request("call")
            call, text, said_at = None, "", None
            for t in range(t0, t0 + 120):
                pd = con.pending
                if pd is not None and pd["kind"] == "call":
                    call = dict(pd)
                look = call is not None and call["open"] + 2 <= t <= call["until"]
                tok = mama if call is not None and said_at is None and t == call["open"] + 16 and "mama" not in text else None
                s_ = con.tick(t, P(t, child_target="mama" if look else "duck"), token=tok)
                if s_.line is not None and s_.line.intent == "call":
                    text = s_.line.text                       # (a call that says "mama" itself: its answer would be an echo)
                if tok is not None:
                    said_at = t
                    answered.append((day, t, call["trial"]))
            k += 1
    rows = [r for r in led.rows if r["ev"] == "child" and r["word"] == "mama"]
    assert rows and not any(r["counted"] for r in rows), \
        f"'mama' said on answering her call counted toward 'says': {[(r['tick'], r['counted'], r['why']) for r in rows]}"
    met = {r["trial"] for r in led.rows if r["ev"] == "outcome" and r["kind"] == "call" and r["result"] == "met"}
    assert len(answered) == 6 and all(tid in met for _d, _t, tid in answered), (answered, met)
    assert len(rows) >= 6 and not con.ledger.says("mama", "token") and all("teaching" in r["why"] for r in rows), rows[:2]
    # another word to "what is this?" (the duck, where she reads it looking, while she asks of the cup it holds): teaching only
    led2 = _TapLedger()
    con2 = C.Conduct(seed=42, stage=2, imperfect=False, ledger=led2, transcriber=Transcriber(None))
    _no_sets(con2, ("duck", "ball", "cup", "block", "block_red"))
    con2.request("ask_what", o="cup")
    for t in range(60):
        pd = con2.pending
        tok = LX.WORD_ID["duck"] if pd is not None and pd["kind"] == "name" and t == pd["open"] + 6 else None
        con2.tick(t, P(t, child_target="duck", child_holds=("cup",)), token=tok)
    r2 = [r for r in led2.rows if r["ev"] == "child" and r["word"] == "duck"]
    assert r2 and not r2[0]["counted"] and "teaching" in r2[0]["why"], r2
    # the distractor's word in a trial's window, the child looking at it: teaching only; the trial's line has no refs
    led3 = _TapLedger()
    con3 = C.Conduct(seed=43, stage=2, imperfect=False, ledger=led3, transcriber=Transcriber(None))
    _no_sets(con3, ("duck", "ball", "cup", "block", "block_red"))
    con3.probe("place", "cup", "duck", new={"cup": "cup@1", "duck": "duck@1"})
    trial_says, word = [], None
    for t in range(200):
        tr = con3.trial
        tok = None
        if tr is not None and tr["phase"] == "said" and t == tr["onset"] + 12:
            word = tr["distractor"]
            tok = LX.WORD_ID[word]
        s_ = con3.tick(t, P(t, child_target=word if word is not None else None), token=tok)
        if s_.line is not None and s_.line.intent in C.TRIAL_INTENTS:
            trial_says.append(s_)
    r3 = [r for r in led3.rows if r["ev"] == "child" and r["word"] == word]
    assert word and r3 and not r3[0]["counted"] and "teaching" in r3[0]["why"], (word, r3)
    assert trial_says and all(s_.line.refs == () and s_.acts == () for s_ in trial_says), [s_.line for s_ in trial_says]
    said_rows = [r for r in led3.rows if r["ev"] == "said" and r["intent"] == "trial_where"]
    assert said_rows and all(r["refs"] == [] for r in said_rows), said_rows
    # a probe dropped after its display: its items presented all the same; a refused display: not
    con4 = _pair_con()
    con4.probe("place", "cup", "duck", new={"cup": "cup@shelf", "duck": "duck@floor"})
    _run_to(con4, K.TRIAL_WAIT + 30, face_near=None)                       # the world does not say what lies near her face
    disp = [r for r in con4.ledger.rows if r["ev"] == "displayed"]
    assert not _trials(con4.ledger) and disp and con4.ledger.presented("cup@shelf") == 1 and \
        con4.ledger.presented("duck@floor") == 1, (disp, con4.ledger.items)

    class _Refuse(C.StubMotion):
        def status(self, i, tick):
            return "refused" if self.acts[i][2] == "present" and tick > self.acts[i][1] else super().status(i, tick)

        def report(self, tick):
            r = super().report(tick)
            return dict(r, acts={i: self.status(i, tick) for i in r["acts"]})
    con5 = _pair_con(motion=_Refuse())
    con5.probe("place", "cup", "duck", new={"cup": "cup@shelf", "duck": "duck@floor"})
    _run_to(con5, 60)
    assert con5.ledger.presented("cup@shelf") == 0 and not [r for r in con5.ledger.rows if r["ev"] == "displayed"]
    # her trunk in the log: at rest "child"; a walk turns it; a turn or lean after the sentence voids a trial; left out: moving
    rest = C.StubMotion().report(0)
    assert rest["trunk"] == "child" and C.moving(dict(rest, acts=[])) is None, rest
    assert "trunk" in (C.moving(dict(rest, trunk="door", acts=[])) or "") and \
        "trunk" in (C.moving(dict({k: v for k, v in rest.items() if k != "trunk"}, acts=[])) or ""), rest
    assert ("trunk", "door") in C.directs(C.Act("walk", "door")) and ("trunk", C.UNNAMED) in C.directs(C.Act("lean_in", "child"))

    class _Lean(C.StubMotion):
        def __init__(self):
            super().__init__()
            self.lean = None

        def report(self, tick):
            r = super().report(tick)
            return dict(r, trunk="?") if self.lean is not None and tick >= self.lean else r
    m = _Lean()
    con6 = _pair_con(motion=m)
    con6.probe("place", "cup", "duck", new={"cup": "c9", "duck": "d9"})
    for t in range(150):
        tr = con6.trial
        if tr is not None and tr["phase"] == "said" and m.lean is None:
            m.lean = t + 1
        con6.tick(t, P(t, child_target=None))
    v6 = _trials(con6.ledger)
    assert v6 and v6[0]["result"] == "void" and "trunk" in v6[0]["why"], v6

    class _NoTrunk(C.StubMotion):
        def report(self, tick):
            return {k: v for k, v in super().report(tick).items() if k != "trunk"}
    con7 = _pair_con(motion=_NoTrunk())
    con7.probe("place", "cup", "duck", new={"cup": "c8", "duck": "d8"})
    _run_to(con7, K.TRIAL_WAIT + 30)
    assert not _trials(con7.ledger) and any("not said within" in r[2] for r in con7.fast.refused), con7.fast.refused[-1:]
    old = C._old_entry(dict(t=5, eyes="child", head="child", left=None, right=None, face=None, acts=[]))
    assert old["trunk"] == C.UNNAMED and "trunk" in C.moving(old), old
    print(f"67 the ninth verifier's low items: 'mama' said on answering her call ({len(rows)} times over 2 days, the call met "
          f"first), 'duck' to 'what is this?' of the cup, and the distractor's word '{word}' in a trial's window: each teaching "
          f"only, never 'says'; her trial's line leaves her with no refs (the world hears its words, never which thing); a probe "
          f"dropped after its display: its items presented once each (a refused display: none); her trunk in her log ('child' "
          f"while it faces the child and holds still): a walk turns it, a lean after the sentence voids the trial, a report "
          f"without it never lets the sentence be said, an older save's log has it moving")


# ------------------------------------------------------------------------------ one timeline for the trial (P3's eleventh round)
def _sp(on, end, name):
    """a trial sentence's words as her voice marks them: "where", "is", "the", then its noun from `on` to `end` (samples)."""
    return [("where", 0, 3200), ("is", 3200, 5600), ("the", 5600, on), (name, on, end)]


# her real voice's trial sentences as the engine makes them since P3's twelfth round (body/sim/lang/trial_lines.json: "where is
# the X?" in the question register, X at its own rate and pitch; the name and its foils in the calling register): every matched
# noun's onset on tick 4 and its end on tick 7, the line 10 ticks; the name and each foil 5 ticks, the word's end on tick 2. The
# suite's timed voice says them so (their samples within those ticks differ, as the engine's do: "ball" ends at 18560,
# "block" at 18900, "cup" at 17500), and every other line 3 ticks a word. TIMED_OLD is the eleventh round's voice, each sentence
# as long as the engine made it before the stimuli were matched ("ball" 10 ticks, "block" 11, "cup" 12 as "stacker" and "box"
# run, "duck" from tick 5, "pew." 6 ticks against "pip."'s 5): the conduct refuses every pair of its that is not one timeline.
TIMED = {"where is the ball?": (_sp(10017, 18560, "ball"), 23310), "where is the block?": (_sp(10017, 18900, "block"), 23867),
         "where is the cup?": (_sp(10017, 17500, "cup"), 22567), "where is the duck?": (_sp(10017, 18880, "duck"), 23867),
         "pip.": ([("pip", 0, 5760)], 11808), "viv.": ([("viv", 0, 6080)], 11470), "vib.": ([("vib", 0, 6560)], 11660),
         "pew.": ([("pew", 0, 6880)], 11900),
         "where is the zeb?": (_sp(10017, 18200, "zeb"), 22900), "where is the fep?": (_sp(10017, 17800, "fep"), 22567),
         "where is the gub?": (_sp(10017, 18420, "gub"), 23100), "where is the tam?": (_sp(10017, 18100, "tam"), 22900),
         "where is the koob?": (_sp(10017, 18600, "koob"), 23310), "where is the kem?": (_sp(10017, 17900, "kem"), 22567),
         "where is the tuv?": (_sp(10017, 18300, "tuv"), 23100), "where is the tuma?": (_sp(10017, 18700, "tuma"), 23310),
         "where is the modi?": (_sp(10017, 18300, "modi"), 23100), "where is the zibo?": (_sp(10017, 18650, "zibo"), 23310)}
# (A60b's level 2, since the lead's decisions after 200e57a and 67741fd: each noun's never-told foil as her voice makes it on
# the "where" form's timeline, trial_lines.json's "foils", each word's end on tick 7 at a sample of its own)
TIMED_OLD = {"where is the ball?": (_sp(10017, 18560, "ball"), 23310), "where is the block?": (_sp(10017, 20480, "block"), 25539),
             "where is the cup?": (_sp(10017, 23200, "cup"), 27954), "where is the duck?": (_sp(12500, 19500, "duck"), 23867),
             "pip.": ([("pip", 0, 5760)], 11808), "viv.": ([("viv", 0, 5500)], 11470), "vib.": ([("vib", 0, 5600)], 11660),
             "pew.": ([("pew", 0, 7000)], 13900)}


def _timed_pcm(words, n):
    """a timed voice's samples: loud (a 200 Hz tone near -15 dB of full scale, 2 dB softer after its first word, as her voice
    says a sentence) from its first word's onset to its last word's end, silent elsewhere, so her sound's start and stop are
    where its words put them."""
    x = np.zeros(int(n), np.int16)
    if words:
        a, e = int(words[0][1]), int(words[-1][2])
        g = np.where(np.arange(a, e) < int(words[0][2]), 0.25, 0.2)
        x[a:e] = (g * 32767 * np.sin(2 * np.pi * 200 * np.arange(a, e) / 16000.0)).astype(np.int16)
    return x


class _TClip:
    def __init__(self, words, n):
        self.words, self.pcm, self.key, self.digest = words, _timed_pcm(words, n), "timed", "timed"


class _TimedVoice:
    """a voice whose trial sentences run as TIMED gives (every other line 3 ticks a word); `made` keeps each clip asked of it
    and whether heard; it takes a trial word's shape and says the sentence as its table has it (the engine's recipe is in
    trial_lines.json; the table here is the engine's result)."""

    def __init__(self, table=None):
        self.table = dict(TIMED if table is None else table)
        self.made = []

    def clip(self, text, register, emphasis=None, heard=True, shape=None):
        self.made.append((text, bool(heard)))
        if text in self.table:
            sp, n = self.table[text]
            return _TClip(sp, n)
        ws = TP.words(text)
        return _TClip([(w, 3 * i * 2400, (3 * i + 3) * 2400) for i, w in enumerate(ws)], 3 * len(ws) * 2400)


class _Flip:
    """her trial stream with its first draw turned over (u to 1 - u): the same trial with the other thing named (the name test:
    its name for the foil, the foil for its name, the same foil), every other draw as drawn."""

    def __init__(self, g):
        self.g = g

    @property
    def bit_generator(self):
        return self.g.bit_generator

    def random(self, n):
        u = self.g.random(n)
        u[0] = 1.0 - u[0]
        return u


_BURST = []


def _burst():
    """two ticks of the tract's sound (the child's voice, as test 64 makes it)."""
    if not _BURST:
        tr_ = T.Tract(1)
        act = np.array([4, 4, 2, 4, 2, 2, 2, 4, 2, 0])
        _BURST.extend([tr_.tick(act), tr_.tick(act)])
    return _BURST


class _Timed:
    """a child that knows no word (written here), each of its acts timed from her trial's sentence as it hears it: its head on
    `look` (a thing's id; "mama", her face) from `look_at` ticks after her voice starts, held 8 ticks, the same tick whichever is
    named; its voice, sound = (ref, k): 2 ticks of the tract from k ticks after her voice starts ("start") or after her sound
    stops ("end": a turn-taker's timing, which follows the sentence's own length, so its voice depends on the word said while its
    looks do not); with face, its head on her face from its voice's first tick for 4 ticks and a "mama" token on that tick (a
    right "mama", and in stage 1 a vocal turn in a pause while looking at her: each a smile, where one is made)."""

    def __init__(self, look="ball", look_at=12, sound=None, face=False):
        self.look, self.look_at, self.sound, self.face = look, look_at, sound, face
        self.start = self.stop = None

    def heard(self, t, s):
        if s.line is not None and s.line.intent in C.TRIAL_INTENTS:
            self.start = t
            self.stop = t + (-(-len(s.clip.pcm) // 2400) if s.clip is not None else 3 * len(TP.words(s.line.text)))

    def at(self, t):
        """-> (its head's target as she reads it, its tract's samples or None, its token or None) at tick t."""
        tg, snd, tok = None, None, None
        if self.start is None:
            return tg, snd, tok
        if self.look is not None and self.start + self.look_at <= t < self.start + self.look_at + 8:
            tg = self.look
        if self.sound is not None:
            s0 = (self.start if self.sound[0] == "start" else self.stop) + self.sound[1]
            if s0 <= t < s0 + 2:
                snd = _burst()[t - s0]
            if self.face and s0 <= t < s0 + 4 and tg is None:
                tg = "mama"
            if self.face and t == s0:
                tok = LX.WORD_ID["mama"]
        return tg, snd, tok


def _timed_trial(form, a, b, kid, stage, seed, flip=False, voice=None, n=300, scaffold=True, seen_=TOYS, vocab=VOCAB,
                 keep=None, noun=None, pool=None, level2=None):
    """one formal trial through her conduct (its stub motion, still; her ear on the tract and the token output) with a timed
    child (_Timed, _Clock: kid.at(t) -> its head's target, its tract, its token[, its reaches, its events]) -> (its trial's row
    with its outcome, or the probe's drop, her voice's busy end after its sentence, the ledger's said row of it). scaffold: the
    words channel labels her lines (A29; the conduct holds a trial's stimuli to it while it does); keep: a list the conduct is
    put in (its ledger read after); noun, pool: an exemplar trial's (A60b's level 2: its things drawn by the conduct from the
    pool, the child shown them as set down; its noun registered unless level2 says otherwise)."""
    con = _conduct(seed=seed, stage=stage, imperfect=False, voice=voice if voice is not None else _TimedVoice(),
                   ledger=_TapLedger(), transcriber=Transcriber(None), scaffold=scaffold, vocab=vocab,
                   level2=(level2 if level2 is not None else ((noun,) if noun else ())))
    if keep is not None:
        keep.append(con)
    _no_sets(con, ("duck", "ball", "cup", "block", "block_red", "ball_blue"))
    if flip:
        con.trial_rng = _Flip(con.trial_rng)
    if form == "name":
        con.probe("name")
    elif form == "exemplar":
        con.probe("exemplar", noun=noun, new=pool)
    else:
        con.probe(form, a, b, new={a: f"{a}@{seed}", b: f"{b}@{seed}"})
    busy, begun, shown = None, False, False
    for t in range(n):
        if con.trial is not None and con.trial.get("sides") and not shown and hasattr(kid, "show"):
            kid.show(*con.trial["sides"])
            shown = True
        got = kid.at(t)
        tg, snd, tok = got[:3]
        reach, ev = (tuple(got[3]), tuple(got[4])) if len(got) > 3 else ((), ())
        seen_t = got[5] if len(got) > 5 and got[5] is not None else seen_     # (what it sees this tick, when it says)
        s = con.tick(t, P(t, child_target=tg, child_reaches=reach, events=ev, seen=seen_t), tract=snd, token=tok)
        kid.heard(t, s)
        if s.line is not None and s.line.intent in C.TRIAL_INTENTS:
            busy, begun = con.fast.busy_until, True
        if con.trial is None and (begun or not con.probes):
            break
    trs = _trials(con.ledger)
    said = next((r for r in con.ledger.rows if r["ev"] == "said" and r["intent"] in C.TRIAL_INTENTS), None)
    return (trs[0] if trs else dict(result="dropped", why=con.fast.refused[-1][2] if con.fast.refused else None)), busy, said


def _same_but_named(x, y):
    """two runs of one trial, the other thing named in the second: the same end by every rule (the same void and why on the same
    tick), its window's ticks on each thing the same (A60b: a pair's on and off turned over; the name's, its face's ticks the
    same, and an exemplar trial's, its new exemplar's and the two things' beside it the same, its word said or a foil: the
    ledger scores them against which was said)."""
    if x["result"] == "scored" and x.get("form") not in ("name", "exemplar"):
        return y["result"] == "scored" and y["end"] == x["end"] and (y["on"], y["off"]) == (x["off"], x["on"])
    return y["result"] == x["result"] and y.get("why") == x.get("why") and y.get("end") == x.get("end") and \
        (y.get("on"), y.get("off")) == (x.get("on"), x.get("off"))


def test_trial_one_timeline():
    """P3's tenth verifier (finding 1): two void rules followed the named sentence's own timing, which her voice made differ by
    word ("ball" 10 ticks, "block" 11-12): the talk-over's cut before its word and stage 2's frown at it. A child that knows no
    word, looking at the ball 12 ticks after her voice starts and sounding 2 ticks at a fixed time from it, had every "block"
    trial voided (stage 2 at +10 by her frown, stage 1 at +4 by the cut) and every "ball" trial met: "ball" understood in each
    of its lives. Nothing the child's voice does voids a trial (no stop, no frown, no judgment of its turns until the trial is
    decided), and since P3's twelfth round every sentence a trial's draw could give is made on one identical timeline (the
    stimuli, stimuli.py), or the trial is not run. Tested by turning her trial stream's draw over: for children whose looks are
    timed from her voice's start, whose voice comes at any time from it or as a turn-taker's after her sound stops, with its
    head on her face and a right 'mama', in stage 1 and 2, pairs of matched things and the name test (the words channel silent),
    every trial ends the same with the other thing named (the same void and why on the same tick, its window's ticks on each
    thing the same: A60b); her real voice too, where the engine is present; the verifier's lives score both things; the
    timeline itself; the
    eleventh round's voice (TIMED_OLD: 10, 11 and 12 ticks) never runs a pair its sentences differ in; the distractor's
    sentence made, never kept as heard."""
    assert _has_trials(), "no formal trial in her conduct"
    # the eleventh round's voice: no pair whose sentences differ is run (dropped before anything is brought, logged)
    old = {}
    for a, b in (("ball", "block_red"), ("cup", "ball"), ("cup", "block_red")):
        x = _timed_trial("place", a, b, _Timed(None), 1, 0, voice=_TimedVoice(TIMED_OLD))[0]
        old[(a, b)] = x
        assert x["result"] == "dropped" and "not one timeline" in x["why"], x
    x = _timed_trial("name", None, None, _Timed(None), 1, 0, voice=_TimedVoice(TIMED_OLD), scaffold=False)[0]
    assert x["result"] != "dropped", x                   # (its foils ran a tick longer after the word, which the carrier
                                                         # phrase's tag now takes the place of: P3's fourteenth round)
    # the verifier's child: stage 2, its voice at +10 from hers, its look on the ball at +12 (ball against the red block)
    base = [_timed_trial("place", "ball", "block_red", _Timed("ball", 12, ("start", 10)), 2, s)[0] for s in range(4)]
    flip = [_timed_trial("place", "ball", "block_red", _Timed("ball", 12, ("start", 10)), 2, s, flip=True)[0]
            for s in range(4)]
    assert all(_same_but_named(x, y) for x, y in zip(base, flip)), \
        [(x["said"], x["result"], x["why"], y["said"], y["result"], y["why"]) for x, y in zip(base, flip)]
    # the sweep: every trial the same with the other thing named
    kids = []
    for look_at in (0, 8, 12, 20, 30):
        for sound in (None, ("start", 0), ("start", 4), ("start", 8), ("start", 10), ("start", 14), ("end", 0), ("end", 2)):
            for face in ((False, True) if sound else (False,)):
                kids.append((look_at, sound, face))
    n_cfg, res = 0, {}
    for stage in (1, 2):
        for form, a, b in (("place", "ball", "block_red"), ("place", "cup", "ball"), ("name", None, None)):
            for k, (look_at, sound, face) in enumerate(kids):
                if form == "name" and face and sound[0] == "end":
                    continue                     # its turn to her face is the name's measure: timed from her sound's stop,
                look = "mama" if form == "name" else a       # it would be a look that depends on the word said
                seed = 7 * k + stage
                sc = form != "name"              # (the name and its foils are one timeline with the words channel silent)
                x = _timed_trial(form, a, b, _Timed(look, look_at, sound, face), stage, seed, scaffold=sc)[0]
                y = _timed_trial(form, a, b, _Timed(look, look_at, sound, face), stage, seed, flip=True, scaffold=sc)[0]
                assert _same_but_named(x, y), (stage, form, look_at, sound, face, x, y)
                n_cfg += 1
                res[x["result"]] = res.get(x["result"], 0) + 1
    assert res.get("scored", 0) > n_cfg // 2 and "dropped" not in res, res
    # the timeline: the window at the test word's onset, her voice held to the sentence's own end, whichever is named
    tl = {}
    for seed in range(6):
        row, busy, said = _timed_trial("place", "cup", "ball", _Timed(None), 1, seed)
        tl[row["said"]] = (row["open"] - row["t"], busy - row["t"], said["ticks"], (row.get("stimulus") or {}).get("ticks"))
    assert set(tl) == {"cup", "ball"} and set(tl.values()) == {(4, 12, 12, 12)}, tl        # (its slot to tick 8, its tag
    row, busy, said = _timed_trial("name", None, None, _Timed(None), 1, 3, scaffold=False)   # 3 ticks: P3's fourteenth round)
    assert row["open"] - row["t"] == 5 and busy - row["t"] == 12 and said["ticks"] == 12, (row, busy, said)
    # the world's playback: its report of her sentence's own end keeps her voice held to it; a stop before it (the world's,
    # never hers in a trial) voids the trial
    wd = {}
    for name, stop in (("end", None), ("stop", 2)):
        con = C.Conduct(seed=3, stage=2, imperfect=False, voice=_TimedVoice(), ledger=_TapLedger())
        _no_sets(con, ("duck", "ball", "cup", "block", "block_red"))
        con.probe("place", "ball", "block_red", new={"ball": "ball@w", "block_red": "block@w"})
        t0 = n_own = busy = after = None
        for t in range(200):
            vd = None if t0 is None else (t if t == t0 + (n_own if stop is None else stop) else None)
            s = con.tick(t, P(t, child_target=None), voice_done=vd)
            if vd is not None:
                after = con.fast.busy_until
            if s.line is not None and s.line.intent in C.TRIAL_INTENTS:
                t0, n_own, busy = t, -(-len(s.clip.pcm) // 2400), con.fast.busy_until
        wd[name] = (_trials(con.ledger)[0], t0, n_own, busy, after)
    x, t0, n_own, busy, after = wd["end"]
    assert x["result"] == "void" and "fewer than" in x["why"] and busy == after == t0 + 12 and n_own == 12, wd["end"]
    x, t0, n_own, busy, after = wd["stop"]
    assert x["result"] == "void" and "world stopped" in x["why"] and after == t0 + 2, wd["stop"]
    v = _TimedVoice()
    _timed_trial("place", "ball", "block_red", _Timed(None), 1, 0, voice=v)
    heard = sorted({x for x in v.made if x[0].startswith("where")})
    assert len([x for x in heard if x[1]]) == 1 and len(heard) == 3, heard     # the one said heard, the other made ahead
    # the verifier's lives through her conduct: both things' trials scored, about half each
    lives = []
    for stage, k in ((2, 10), (1, 4)):
        for seed in range(2):
            con = C.Conduct(seed=seed, stage=stage, imperfect=False, voice=_TimedVoice(), ledger=_TapLedger(),
                            transcriber=Transcriber(None))
            kid, t, j = _Timed("ball", 12, ("start", k)), 0, 0
            while j < 40 or con.trial is not None:
                if con.trial is None and not con.probes and j < 40:
                    _no_sets(con, ("duck", "ball", "cup", "block", "block_red"))
                    con.probe("place", "ball", "block_red", new={"ball": f"ball@{j}", "block_red": f"block@{j}"})
                    j += 1
                tg, snd, tok = kid.at(t)
                s = con.tick(t, P(t, child_target=tg), tract=snd, token=tok)
                kid.heard(t, s)
                t += 1
            sc = [x for x in _trials(con.ledger) if x["result"] == "scored"]
            nb = sum(x["said"] == "ball" for x in sc)
            lives.append((stage, seed, nb, len(sc), (con.ledger.maps if hasattr(con.ledger, "maps") else    # (level 1: A60b)
                                                     con.ledger.understood)("ball")))
            assert len(sc) >= 30 and 0.25 < nb / len(sc) < 0.75, lives[-1]
    # her real voice, where the engine is present
    real = "the engine absent here: not run"
    if _have_engine():
        tmp = tempfile.mkdtemp()
        try:
            cache = V.VoiceCache(os.path.join(tmp, "v"), server=V.SynthServer(nice=19))
            n_real = 0
            for stage in (1, 2):
                for form, a, b in (("place", "ball", "block_red"), ("name", None, None)):
                    for sound in (("start", 4), ("start", 10), ("end", 0)):
                        for face in (False, True):
                            if form == "name" and face and sound[0] == "end":
                                continue
                            look, sc = ("mama", False) if form == "name" else (a, True)
                            x = _timed_trial(form, a, b, _Timed(look, 12, sound, face), stage, 5, voice=cache, scaffold=sc)[0]
                            y = _timed_trial(form, a, b, _Timed(look, 12, sound, face), stage, 5, flip=True, voice=cache,
                                             scaffold=sc)[0]
                            assert x["result"] != "dropped" and _same_but_named(x, y), ("her voice", stage, form, sound, face,
                                                                                        x, y)
                            n_real += 1
            cache.close()
            real = f"{n_real} of them in her real voice, the same"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"68 one timeline for the trial (P3's tenth verifier, finding 1; its stimuli time-matched since the twelfth round): "
          f"the verifier's child (its voice at +10, its look on the ball at +12, stage 2) ends each of 4 trials the same with "
          f"the other thing named; {n_cfg} children and trials (looks at +0, +8, +12, +20, +30; its voice from her voice's "
          f"start or, a turn-taker's, from her sound's stop; its head on her face with a right 'mama'; stage 1 and 2; ball and "
          f"block, cup and ball, each 12 ticks with its noun on ticks 4-7; the name and its foils, 12 ticks, the name on 5-7, "
          f"the words channel silent) each the same whichever is named ({', '.join(f'{k} {v}' for k, v in sorted(res.items()))}); {real}; "
          f"cup or ball named: the test word's onset at +4 (its window +6 to +27, A60b) and her voice held 12 ticks (its tag "
          f"since P3's fourteenth round), the name's onset at +5 and 12; the eleventh round's voice (10, 11 and 12 ticks, an "
          f"onset a tick late) runs none of its "
          f"{len(old)} pairs (dropped before anything is brought, logged), its name and foils now one timeline on the carrier "
          f"phrase; the world's report of her sentence's own end keeps her voice held (a child that never looks: void, fewer "
          f"than 4 ticks on either), a stop before it voids the trial; the distractor's sentence made ahead, never kept as "
          f"heard; the verifier's lives score "
          f"both things: "
          + "; ".join(f"stage {st} seed {sd} ball named {nb} of {n}{', mapped' if u else ''}" for st, sd, nb, n, u in lives))


# ---------------------------------------------------------------------------- P3's twelfth round: time-matched stimuli
INV_ANCHORS = ("start", "stop36", "stop48", "stop60", "shut36", "shut48", "shut60", "free", "onset", "wend", "end", "close")
INV_OFFSETS = tuple(range(-4, 31))


def _tick_levels(pcm):
    """each tick's loudest 10 ms frame and its RMS (her mouth's opening, as the face shows it: playback.Utterance.mouth), in dB
    of full scale, on the clip's tick grid (its own measure, not her rule's; through its ears and 10 ms by 10 ms: _perceived)."""
    x = np.asarray(pcm, np.float64) / 32767.0
    n = -(-len(x) // 2400)
    x = np.pad(x, (0, n * 2400 - len(x)))
    fr = np.sqrt((x.reshape(n * 15, 160) ** 2).mean(1)).reshape(n, 15).max(1)
    tick = np.sqrt((x.reshape(n, 2400) ** 2).mean(1))
    return 20 * np.log10(np.maximum(fr, 1e-12)), 20 * np.log10(np.maximum(tick, 1e-12))


TRIAL_SLOT = {"trial_where": 3, "trial_combo": 3, "trial_name": 1}   # the test word's place in its frame (P3's fourteenth round)


def _heard_anchors(t, clip, text, channel, intent="trial_where"):
    """what a child can time from in her sentence as it hears it, begun on tick t (its clip's samples and the voice's word marks;
    the words channel's END while the channel runs) -> {anchor: tick} (_Clock's anchors)."""
    if clip is None:                                     # no voice: 3 ticks a word
        ws = TP.words(text)
        words, n, lv = [(w, 3 * i * 2400, (3 * i + 3) * 2400) for i, w in enumerate(ws)], 3 * len(ws) * 2400, None
    else:
        words, n, lv = clip.words, len(clip.pcm), _tick_levels(clip.pcm)
    k = -(-n // 2400)
    out = dict(start=t, free=t + k)
    for L in (36, 48, 60):
        for name, j in (("stop", 0), ("shut", 1)):       # its ears: the loudest 10 ms; its eyes: her mouth, the tick's loudness
            loud = [i for i in range(k) if lv is not None and lv[j][i] > -L]
            out[f"{name}{L}"] = t + (loud[-1] + 1 if loud else k)
    _w, a, e = words[min(TRIAL_SLOT[intent], len(words) - 1)]   # the test word (before its tag since P3's fourteenth round;
                                                         # its line's last word before it)
    out["onset"], out["wend"] = t + a // 2400, t + (max(e, 1) - 1) // 2400
    if channel:
        ends = [i for i, kind in ST.channel(words, n) if kind == "end"]
        out["end"] = t + ends[0] if ends else None
    out["close"] = out["onset"] + (K.TRIAL_LOOK[1] if hasattr(K, "TRIAL_LOOK") else 20)   # the tick after its window's last
                                                         # (A60b; before it, the 20 ticks from the onset)
    return out


def _rng(*key):
    """a stream keyed by integers (a tick before the first shifted past every tick: never negative)."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(k) if k >= 0 else 10 ** 9 - int(k) for k in key])))


def _draw_acts(seed, t0, things):
    """1-4 acts of a _Clock for the sentence heard at t0, drawn from its own stream keyed by that tick (so a life turned over
    at a trial draws the same): an anchor, an offset of -4 to +30 ticks from it, a kind (looks and reaches most, then its voice,
    its token, a hit, pain, distress), what it acts on (either thing or her face) and how long."""
    r = _rng(seed, t0, 7)
    kinds = ("look", "look", "look", "reach", "reach", "sound", "mama", "hit", "pain", "distress")
    out = []
    for _ in range(int(r.integers(1, 5))):
        out.append((INV_ANCHORS[int(r.integers(len(INV_ANCHORS)))], int(r.integers(-4, 31)), kinds[int(r.integers(len(kinds)))],
                    (tuple(things) + ("mama",))[int(r.integers(len(things) + 1))], int(r.integers(1, 11))))
    return out


class _Clock:
    """a child that knows no word (written here): each of its acts timed from what it hears of her trial's sentence
    (_heard_anchors), the same whatever is named. acts [(anchor, offset, kind, arg, hold)]; anchors "start" (her voice's first
    tick), "stop36", "stop48", "stop60" (the tick after the last in which her sound's loudest 10 ms is above -36, -48 or -60 dB
    of full scale, on the clip's tick grid: not through its ears, nor 10 ms by 10 ms, which are test 70's _Hears), "shut36",
    "shut48", "shut60" (the tick after the last in which her mouth is open at those levels, the tick's loudness the face
    shows), "free" (the tick after her clip's last), "onset" and "wend" (its last word's first and last tick by the
    voice's marks), "end" (the words channel's END, while the channel runs), "close" (the onset and 20 ticks: the window's
    close); kinds "look" (its head on arg: a thing's id or "mama", her face), "reach" (a hand toward arg), "sound" (2 ticks of its
    tract), "mama" (its token), "hit" (it hits her), "pain", "distress". draw: its acts drawn afresh for each sentence it hears
    (_draw_acts, from seed); wander (seed, p): its head on a thing, her face or nothing for 3-10 ticks begun at a tick's chance p;
    babble (seed, p): 2 ticks of its tract begun at a tick's chance p (each a function of the tick: nothing it hears moves them).
    knower: it looks at the thing its word names 6 ticks after the word's onset (a child the property must tell apart)."""

    def __init__(self, acts=(), channel=True, things=("ball", "cup", "block_red"), draw=None, wander=None, babble=None,
                 knower=False):
        self.acts, self.channel, self.things = list(acts), channel, tuple(things)
        self.draw, self.wander, self.babble, self.knower = draw, wander, babble, knower
        self.anch, self.named = None, None
        self._w, self._b = {}, {}                             # its wandering and babbling by tick (functions of the tick)

    def _wander_at(self, u):
        if u not in self._w:
            r = _rng(self.wander[0], u, 3)
            go = r.random() < self.wander[1]
            opts = self.things + ("mama", None)
            self._w[u] = (u + 3 + int(r.integers(8)), opts[int(r.integers(len(opts)))]) if go else None
        return self._w[u]

    def _babble_at(self, u):
        if u not in self._b:
            self._b[u] = _rng(self.babble[0], u, 5).random() < self.babble[1]
        return self._b[u]

    def heard(self, t, s):
        if s.line is not None and s.line.intent in C.TRIAL_INTENTS:
            self.anch = _heard_anchors(t, s.clip, s.line.text, self.channel, s.line.intent)
            if self.draw is not None:
                self.acts = _draw_acts(self.draw, t, self.things)
            if self.knower:
                w = s.line.focus
                self.named = next((x.id for x in TOYS if x.name == w and x.id in self.things), None)

    def at(self, t):
        tg, snd, tok, reach, ev = None, None, None, [], []
        if self.wander is not None:
            for u in range(t - 10, t + 1):
                w = self._wander_at(u)
                if w is not None and t < w[0]:
                    tg = w[1]
        if self.babble is not None:
            for u0 in (t, t - 1):
                if self._babble_at(u0):
                    snd = _burst()[t - u0]
                    break
            for u in [u for u in self._b if u < t - 20]:
                del self._b[u]
        for u in [u for u in self._w if u < t - 20]:
            del self._w[u]
        if self.anch is not None:
            for anchor, off, kind, arg, hold in self.acts:
                a = self.anch.get(anchor)
                if a is None:
                    continue
                t0 = a + off
                if kind == "look" and t0 <= t < t0 + hold:
                    tg = arg
                elif kind == "reach" and t0 <= t < t0 + hold and arg != "mama":
                    reach.append(arg)
                elif kind == "sound" and t0 <= t < t0 + 2:
                    snd = _burst()[t - t0]
                elif kind == "mama" and t == t0:
                    tok = LX.WORD_ID["mama"]
                elif kind in ("hit", "pain", "distress") and t == t0:
                    ev.append(({"hit": "hit_her"}.get(kind, kind), None))
            if self.knower and self.named and self.anch["onset"] + 6 <= t < self.anch["onset"] + 14:
                tg = self.named
        return tg, snd, tok, tuple(reach), tuple(ev)


def _inv_outcome(rows, i0, i1, refused, t0, t1):
    """a trial's end in the ledger's rows i0..i1 (and her refusals of ticks t0..t1): (result, why, its end's tick, form, the word
    said, on, off) or ("dropped", why)."""
    tr = next((r for r in rows[i0:i1] if r["ev"] == "trial"), None)
    if tr is None:
        dr = [r for r in refused if t0 <= r[0] <= t1 and str(r[1]).startswith("trial:")]
        return ("dropped", dr[0][2] if dr else None)
    oc = next((r for r in rows[i0:i1] if r["ev"] == "trial_outcome"), None)
    return (oc["result"] if oc else None, oc["why"] if oc else None, oc["t"] if oc else None, tr["form"], tr["said"],
            oc.get("on") if oc else None, oc.get("off") if oc else None)


def _inv_same(x, y):
    """two runs of one trial, the other thing named in the second: the same end by every rule (the same void and why on the same
    tick, the same drop), its window's ticks on each thing the same (A60b: a pair's scored trial's on and off turned over, the
    other thing said; the name's, its face's ticks the same)."""
    if x[0] == "scored" and x[3] != "name":
        return y[:4] == x[:4] and y[4] != x[4] and (y[5], y[6]) == (x[6], x[5])
    return y[:4] == x[:4] and y[5:] == x[5:]


INV_PAIRS = (("ball", "block_red"), ("cup", "ball"), ("block_red", "cup"), ("ball", "duck"), ("cup", "block_red"))


def _inv_life(seed, voice, n_trials=8, n_max=6000):
    """one life of formal trials for the invariance property: her conduct (stage 1 or 2, her imperfection on, the words channel
    labelling her lines in two lives of three), a W2-like motion breaking its contract now and then while she holds still (never
    knowing the target), a child that knows no word (_Clock: its head wandering, its voice babbling, and 1-4 acts timed from what
    it hears of each trial sentence), her day plan's probes one after another (the pairs of INV_PAIRS, "duck" among them with no
    time-matched stimulus, and the name). Run whole; each trial snapshotted on the tick it began (its draw made), restored into
    a fresh conduct, motion and child with the draw turned over and run to its end -> [(its end, its end turned over)], the
    conduct, stats."""
    import copy                                                          # noqa: PLC0415
    r = _rng(seed, 11)
    scaffold = seed % 3 != 0
    rogue = {k: float(r.choice([0.0, 0.0, 0.004, 0.01])) for k in ("glance", "hand", "face", "outside", "status", "omit", "trunk")}
    longest = int(r.integers(8, 31))

    def make():
        m = _TW2(seed=seed, lengths=(1, longest), queue_p=0.2, l1_p=0.5, place_p=0.01, rogue=rogue, cheat=False)
        con = _conduct(seed=seed, stage=1 + seed % 2, imperfect=True, motion=m, ledger=_TapLedger(), voice=voice,
                       transcriber=Transcriber(None), scaffold=scaffold)
        return con, m
    con, m = make()
    kid = _Clock(channel=scaffold, draw=seed + 100, wander=(seed + 200, 0.03), babble=(seed + 300, float(r.choice([0, 0.01, 0.03]))))
    probes = [("name", None, None)] + [("place", a, b) for a, b in INV_PAIRS]
    snaps, ends, k, t, cur = [], [], 0, 0, None
    while (k < n_trials or con.trial is not None or con.probes) and t < n_max:
        if con.trial is None and not con.probes and k < n_trials and (not ends or t > ends[-1] + 30):
            form, a, b = probes[int(r.integers(len(probes)))]
            con.probe(form, a, b, new=None if form == "name" else {a: f"{a}@{seed}.{k}", b: f"{b}@{seed}.{k}"})
            k += 1
            cur = dict(rows=len(con.ledger.rows), ref=len(con.fast.refused), t=t)
        got = kid.at(t)
        p = P(t, child_target=got[0], child_reaches=got[3], events=got[4], child_sounding=got[1] is not None)
        m.step(t, p, con)
        s = con.tick(t, p, tract=got[1], token=got[2])
        kid.heard(t, s)
        tr = con.trial
        if cur is not None and tr is not None and tr["since"] == t:
            snaps.append(dict(cur, since=t, con=json.loads(json.dumps(con.state(), default=_np)), motion=m.state(),
                              kid=copy.deepcopy(kid)))
        if cur is not None and tr is None and not con.probes:
            ends.append(t)
            if not snaps or snaps[-1]["t"] != cur["t"]:                   # dropped when it began: no draw to turn over
                snaps.append(dict(cur, since=None))
            snaps[-1].update(end=t, rows_end=len(con.ledger.rows))
            cur = None
        t += 1
    out = []
    for sn in snaps:
        if "end" not in sn:
            continue
        base = _inv_outcome(con.ledger.rows, sn["rows"], sn["rows_end"], con.fast.refused, sn["t"], sn["end"])
        if sn["since"] is None:
            out.append((base, base))
            continue
        fresh, fm = make()
        fm.load_state(sn["motion"])
        st = json.loads(json.dumps(sn["con"]))
        tr = st["trial"]
        if tr["form"] == "name":
            tr["name"] = not tr["name"]
            tr["said"] = LX.NAME if tr["name"] else tr["foil"]
        else:
            tr["target"], tr["distractor"] = tr["distractor"], tr["target"]
        fresh.load_state(st)
        fk = copy.deepcopy(sn["kid"])
        for u in range(sn["since"] + 1, sn["since"] + 1000):
            ft = fresh.trial
            if ft is None or ft["since"] != sn["since"]:
                break
            got = fk.at(u)
            p = P(u, child_target=got[0], child_reaches=got[3], events=got[4], child_sounding=got[1] is not None)
            fm.step(u, p, fresh)
            s = fresh.tick(u, p, tract=got[1], token=got[2])
            fk.heard(u, s)
        out.append((base, _inv_outcome(fresh.ledger.rows, 0, len(fresh.ledger.rows), fresh.fast.refused, sn["since"], u)))
    return out, con


def test_trial_invariance():
    """P3's twelfth round (the lead's decision on C67, the eleventh verifier's finding 1): the two test sentences ended on
    different ticks in her voice ("where is the ball?" 10, "where is the block?" 11), so a child that knew no word, hitting her
    2 ticks after her sound stopped, had every ball trial voided before its look was judged and every block trial met ("block"
    understood in every life), and one looking 14 ticks after her sound stopped had ball trials met and block trials none. Now
    every sentence a trial's draw could give is made on one identical timeline, as infant labs match their stimuli: one carrier
    a form, its test word at the engine's own per-word rate and a matched pitch contour (body/sim/lang/trial_lines.json), checked
    on the rendered audio (stimuli.timeline: the clip's ticks, each word's first and last tick, every tick loud or silent alike
    by its RMS and its loudest 10 ms, the words channel while the scaffold labels her lines); a set that is not one timeline is
    never used; since P3's thirteenth round every sentence of a form on one carrier recording before its test word, since its
    fourteenth on one carrier phrase, a tag after it too (stimuli.parts, splice). The acceptance test on the tick grid:
    for any child that does not know the words, every trial ends the same with the draw turned over (the same void and why on
    the same tick, its window's ticks on each thing the same: A60b), for children timed from her voice's start, her sound's stop
    (on the clip's
    tick grid, at -36, -48 or -60 dB), her mouth's closing (the same), her clip's end, the test word's onset and end, the words
    channel's END and the window's close, with looks, reaches, sounds, its token, hits, pain and distress at every offset from -4
    to +30 ticks of each; in the suite's timed voice and her real voice (where the engine is present), stage 1 and 2, a pair and
    the name; and over many lives (random children, W2-like motions breaking their contract, her imperfection, the channel on and
    off). A knower's trials differ (the check can tell). The matched sets measured again on the rendered audio (the engine
    present): each form's words one timeline, their contours within TRIAL_F0; with the words channel labelling her lines, only a
    birth word's token pairs with another (the name and its spelled foils, spelled words of different lengths, and every
    combination never share one: not run while it does); the unmatched words ("duck", "book": a tick between loud and silent;
    "box", "stacker": too long at any rate within her registers' span) never tested; each form's carrier one recording (as the
    engine says them, its carriers differ before the test word). Her voice's start and her sound's stop as the child perceives
    them (10 ms by 10 ms, at every level, its own ears) are test 71's; what it perceives within the test word's slot, where no
    two words are alike, test 70's (C69)."""
    assert ST is not None and hasattr(C.Conduct, "set_scaffold"), "no time-matched stimuli in her conduct (P3's twelfth round)"
    # the eleventh verifier's children: a hit 2 ticks after her sound stops, a look 14 ticks after it (ball against the block)
    for acts in ([("free", 2, "hit", None, 1), ("start", 12, "look", "block_red", 8)], [("free", 14, "look", "ball", 8)],
                 [("stop60", 2, "hit", None, 1), ("start", 12, "look", "block_red", 8)], [("shut36", 14, "look", "ball", 8)]):
        for stage in (1, 2):
            for seed in range(4):
                x = _timed_trial("place", "ball", "block_red", _Clock(acts), stage, seed)[0]
                y = _timed_trial("place", "ball", "block_red", _Clock(acts), stage, seed, flip=True)[0]
                xo = _oc(x)
                yo = _oc(y)
                assert x["result"] != "dropped" and _inv_same(xo, yo), (acts, stage, seed, xo, yo)
    # the sweep: every anchor, every offset, every kind of act, the suite's timed voice and her real voice
    voices = [("timed", _TimedVoice())]
    tmp = None
    if _have_engine():
        tmp = tempfile.mkdtemp()
        voices.append(("real", V.VoiceCache(os.path.join(tmp, "v"), server=V.SynthServer(nice=19))))
    sweep = {}
    try:
        for vname, voice in voices:
            for form, a, b, sc in (("place", "ball", "block_red", True), ("name", None, None, False)):
                kinds = ([("look", a), ("look", b), ("look", "mama"), ("reach", a), ("reach", b)] if form != "name" else
                         [("look", "mama")]) + [("sound", None), ("mama", None), ("hit", None), ("pain", None),
                                                ("distress", None)]
                for stage in (1, 2):
                    n_cfg, res = 0, {}
                    for anchor in INV_ANCHORS:
                        if anchor == "end" and not sc:
                            continue                         # (no words channel: nothing to time from)
                        for off in INV_OFFSETS:
                            for kind, arg in kinds:
                                act = [(anchor, off, kind, arg, 8)]
                                seed = 3 + stage
                                x = _timed_trial(form, a, b, _Clock(act, channel=sc), stage, seed, voice=voice, scaffold=sc)[0]
                                y = _timed_trial(form, a, b, _Clock(act, channel=sc), stage, seed, flip=True, voice=voice,
                                                 scaffold=sc)[0]
                                xo = _oc(x)
                                yo = _oc(y)
                                assert x["result"] != "dropped" and _inv_same(xo, yo), (vname, form, stage, act, xo, yo)
                                n_cfg += 1
                                res[x["result"]] = res.get(x["result"], 0) + 1
                    sweep[(vname, form, stage)] = (n_cfg, res)
        # a knower: its trials differ with the other thing named (the property tells it apart)
        kn = []
        for seed in range(6):
            x = _timed_trial("place", "ball", "block_red", _Clock(knower=True), 2, seed)[0]
            y = _timed_trial("place", "ball", "block_red", _Clock(knower=True), 2, seed, flip=True)[0]
            kn.append(_inv_same(_oc(x),
                                _oc(y)))
        assert not any(kn), kn
        # many lives: every trial turned over at its draw ends the same
        lives, flips, res_l, drops = 0, 0, {}, {}
        for vname, voice in voices:
            for seed in range(48 if vname == "timed" else 6):
                pairs, _con = _inv_life(seed, voice, n_trials=8 if vname == "timed" else 6)
                lives += 1
                for base, turned in pairs:
                    assert _inv_same(base, turned), (vname, seed, base, turned)
                    flips += base[0] != "dropped"
                    res_l[base[0]] = res_l.get(base[0], 0) + 1
                    if base[0] == "dropped":
                        why = str(base[1])
                        k_ = "no stimulus" if "no time-matched stimulus" in why else "channel" if "channel" in why else "other"
                        drops[k_] = drops.get(k_, 0) + 1
        assert flips >= 200 and drops.get("no stimulus") and drops.get("channel"), (flips, drops)
        # the matched sets on the rendered audio, where the engine is present: each form's sentences built on its carrier
        # phrase (P3's fourteenth round: one recording before the test word, one tag after it, the word at the form's level)
        measured = "the engine absent here: the table's recipe held to its own timelines only"
        carriers = {}
        for key, f in ST.STIMULI.items():
            assert set(f["words"]) | set(f["unmatched"]) and not set(f["words"]) & set(f["unmatched"]), key
        if len(voices) > 1:
            import importlib.util                                            # noqa: PLC0415
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            spec = importlib.util.spec_from_file_location("svc", os.path.join(root, "tools", "sim_voice_check.py"))
            svc = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(svc)
            cache = voices[1][1]
            rows_m = []
            for key, f in sorted(ST.STIMULI.items()):
                if not f["words"]:
                    continue
                tls, tls_ch, conts, own = {}, {}, {}, {}
                for w in sorted(f["words"]):
                    pt = ST.parts(key, w) if hasattr(ST, "parts") else None
                    assert pt is not None, ("no carrier phrase for the test word (P3's fourteenth round)", key, w)
                    pre, c0, tag = (cache.clip(x[0], x[1], emphasis=x[2], heard=False, shape=x[3]) for x in
                                    (pt["pre"], pt["own"], pt["tag"]))
                    own[w] = ST.timeline(c0.words, len(c0.pcm), c0.pcm, False, f["slot"])["pre"]
                    said = f["said"].replace("{o}" if "{o}" in f["said"] else "{n}", w) if key in ("where", "name") else \
                        f["said"].replace("{c}", w).replace("{o}", key.split(":")[1])
                    c = ST.splice(pre, c0, tag, pt["at"], pt["own_from"], pt["n_own"], pt["gain"], text=said)
                    sl = [x for x, _a, _e in c.words].index(w)
                    tls[w] = ST.timeline(c.words, len(c.pcm), c.pcm, False, sl, tag_at=c.spliced["tag_at"])
                    tls_ch[w] = json.dumps(ST.timeline(c.words, len(c.pcm), c.pcm, True, sl, tag_at=c.spliced["tag_at"]),
                                           sort_keys=True)
                    conts[w] = svc.f0_contour(c.pcm[c.words[sl][1]:c.words[sl][2]])
                assert ST.same(list(tls.values()), list(tls)) is None, (key, ST.same(list(tls.values()), list(tls)))
                carriers[key] = (f["pre"]["text"], len(set(own.values())), len(own))
                ref = f["f0"]
                assert all(max(abs(x / y - 1) for x, y in zip(conts[w], ref)) <= K.TRIAL_F0 for w in conts), (key, conts, ref)
                ws = sorted(tls)
                pairs_ch = sum(tls_ch[x] == tls_ch[y] for i, x in enumerate(ws) for y in ws[i + 1:])
                for w in f["unmatched"]:                                     # said as she says it everyday: not one timeline
                    text = f["text"].replace("{w}", w)
                    emph = w if f["emphasis"] == "{w}" else f["emphasis"]
                    c = cache.clip(text, f["register"], emphasis=emph, heard=False)
                    tl = ST.timeline(c.words, len(c.pcm), c.pcm, False, f["slot"])
                    assert ST.same([tls[ws[0]], tl], [ws[0], w]) is not None, (key, w)
                rows_m.append(f"{key}: {len(ws)} words one timeline ({', '.join(ws)}), {pairs_ch} of "
                              f"{len(ws) * (len(ws) - 1) // 2} pairs one timeline with the words channel too; unmatched "
                              f"{', '.join(sorted(f['unmatched'])) or 'none'}")
            measured = "; ".join(rows_m)
            # as the engine says them, their own sentences differ before the test word (its prosody plans the sentence
            # whole; the name's are the name alone): the carrier phrase is what makes them one
            assert carriers and all(k_ > 1 for k, (_t, k_, n_) in carriers.items() if n_ > 1 and k != "name"), carriers
            measured += "; each form's carrier phrase one recording before its word and one after it (" + ", ".join(
                f"{k}: {t_!r} before it, where the engine's own {n_} sentences had {k_} different carriers" for k, (t_, k_, n_)
                in sorted(carriers.items()) if k != "name") + ")"
        # the conduct holds her to it: no stimulus, a channel that differs, the channel silent
        rattle = TOYS + seen(("rattle", "rattle", "purple", "mat", True))
        voice = voices[-1][1]
        con_rows = {}
        for name, a, b, sc in (("duck", "ball", "duck", True), ("rattle on", "ball", "rattle", True),
                               ("rattle off", "ball", "rattle", False)):
            x = _timed_trial("place", a, b, _Clock(), 1, 2, voice=voice, scaffold=sc, seen_=rattle, vocab=VOCAB + ("rattle",))[0]
            con_rows[name] = x
        assert con_rows["duck"]["result"] == "dropped" and "no time-matched stimulus" in con_rows["duck"]["why"], con_rows
        if voices[-1][0] == "real":
            assert con_rows["rattle on"]["result"] == "dropped" and "channel" in con_rows["rattle on"]["why"] and \
                con_rows["rattle off"]["result"] != "dropped", con_rows
    finally:
        for _v, voice in voices[1:]:
            voice.close()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    total = sum(n for n, _r in sweep.values())
    print(f"69 time-matched stimuli on the tick grid (P3's twelfth round; its thirteenth: one carrier recording): the eleventh "
          f"verifier's children (a hit "
          f"2 ticks after her sound stops, a look 14 ticks after it) end each trial the same with the other thing named; "
          f"{total} children and trials, each one act timed from one of {len(INV_ANCHORS)} anchors (her voice's start, her "
          f"sound's stop at -36, -48 and -60 dB on the clip's tick grid, her mouth's closing at those levels, "
          f"her clip's end, the test word's onset and end, the words channel's END, the window's close) at every offset from {INV_OFFSETS[0]} to +{INV_OFFSETS[-1]}, a look or reach on "
          f"either thing or her face, a sound, its token, a hit, pain or distress, in stage 1 and 2: every trial the same "
          f"whichever is named (" + "; ".join(f"{v} voice {f} stage {s}: {n} ({', '.join(f'{k} {c}' for k, c in sorted(r.items()))})"
                                              for (v, f, s), (n, r) in sorted(sweep.items())) +
          f"); a knower's differ ({len(kn)} of {len(kn)}); {lives} lives ({flips} trials turned over at their draw, each the "
          f"same: {', '.join(f'{k} {c}' for k, c in sorted(res_l.items()))}; dropped before anything was brought: "
          f"{', '.join(f'{k} {c}' for k, c in sorted(drops.items()))}); {measured}; the conduct drops a pair with 'duck' (no "
          f"stimulus), and in her real voice ball and rattle while the words channel labels her lines (a token against 6 "
          f"letters), running them once it is silent")


# ---------------------------------------------------------------------------- P3's thirteenth round: scored by intention to treat
ITT_TOYS = seen(("ball", "ball", "", "mat", True), ("bear", "bear", "", "mat", True), ("block", "block", "", "mat", True),
                ("bottle", "bottle", "", "mat", True), ("car", "car", "", "mat", True), ("cup", "cup", "", "mat", True),
                ("drum", "drum", "", "mat", True))
_PERC = {}


def _perceived(clip, dists=(0.5, 0.7, 1.0)):
    """what a child perceives of a sentence of hers 10 ms by 10 ms (P3's twelfth verifier's three routes; kept by its clip) ->
    dict(frames: each 10 ms frame's RMS, dB of full scale (its ears at the clip); ticks: each tick's RMS (her mouth as the face
    shows it); ear: {distance: the louder of its two ears' level each 10 ms frame, dB SPL, through body/sim/ears.Ears with her
    mouth that far in front of its head: their latency and 25 ms frames and all}; bands: {"L" | "R": each 10 ms frame's level
    in each of the cochlea's 40 bands at that ear, dB SPL, her mouth 0.7 m in front of it: the fourteenth verifier's route};
    tag: the tick her tag begins on, since P3's fourteenth round (its clip's end without one)), each frame ending at its
    hop."""
    key = (clip.key, clip.digest, len(clip.pcm))
    if key in _PERC:
        return _PERC[key]
    from body.sim import ears as E                                       # noqa: PLC0415
    x = np.asarray(clip.pcm, np.float64)
    n = -(-len(x) // 2400)
    fr = 20 * np.log10(np.maximum(V.frame_rms(np.asarray(clip.pcm)), 1e-12))
    tk = 20 * np.log10(np.maximum(np.sqrt((np.pad(x / 32767.0, (0, n * 2400 - len(x))).reshape(n, 2400) ** 2).mean(1)), 1e-12))
    pa = clip.pa() if hasattr(clip, "pa") else x * (V.PA_PER_UNIT / 32767.0)
    pa = np.pad(np.asarray(pa, np.float64), (0, (n + 2) * 2400 - len(pa)))
    mid = (E.EAR_SITES["L"] + E.EAR_SITES["R"]) / 2
    ear, bands = {}, {"L": [], "R": []}
    for d in dists:
        ears, lv = E.Ears(), []
        for u in range(n + 2):
            h = ears.tick(E.EAR_SITES["L"], E.EAR_SITES["R"],
                          {"mama": (pa[u * 2400:(u + 1) * 2400], mid + np.array([d, 0.0, 0.0]))})
            pw = (np.maximum(h.left, h.right).astype(np.float64) ** 3 * E.E_CODE).sum(1)
            lv.extend(10 * np.log10(np.maximum(pw, 1e-30) / E.P_REF ** 2))
            if d == 0.7:
                for side, code in (("L", h.left), ("R", h.right)):
                    bands[side].extend(10 * np.log10(np.maximum(code.astype(np.float64) ** 3 * E.E_CODE, 1e-30) /
                                                     E.P_REF ** 2))
        ear[d] = np.array(lv)
    sp = getattr(clip, "spliced", None)
    _PERC[key] = dict(frames=fr, ticks=tk, ear=ear, bands={k_: np.array(v) for k_, v in bands.items()},
                      tag=sp["tag_at"] // 2400 if isinstance(sp, dict) else n)
    return _PERC[key]


def _moment(perc, anchor):
    """when a child takes her sound to stop, in ticks from her sentence's start to the 10 ms: ("frame", L) the end of her last
    10 ms frame above L dB of full scale; ("mouth", L) the tick after her last tick above L (her mouth's last opening, as its
    eyes see her face); ("ear", d, L) the end of the last 10 ms frame its own ears hear above L dB SPL, her mouth d m in front
    of it; ("band", ear, b, L, "stop" | "start") the end of the last 10 ms frame, or the start of the first, above L dB SPL in
    band b (of body/sim/ears.Ears' 40) at its "L" or "R" ear, her mouth 0.7 m in front of it (the fourteenth verifier's
    route); ("start",) her voice's start; ("in", ...) the same within the test word's slot, before her tag's first tick (P3's
    fourteenth round: a moment of the test word's own sound)."""
    if anchor[0] == "start":
        return 0.0
    if anchor[0] == "band":
        _b, side, b, lv_, which = anchor
        up = np.nonzero(perc["bands"][side][:, b] > lv_)[0]
        if not len(up):
            return 0.0
        return float(up[0]) / 15.0 if which == "start" else float(up[-1] + 1) / 15.0
    lim = None
    if anchor[0] == "in":
        anchor, lim = anchor[1:], perc["tag"]
    if anchor[0] == "mouth":
        up = np.nonzero(perc["ticks"] > anchor[1])[0]
        up = up if lim is None else up[up < lim]
        return float(up[-1] + 1) if len(up) else 0.0
    lv = perc["frames"] if anchor[0] == "frame" else perc["ear"][anchor[1]]
    up = np.nonzero(lv > anchor[-1])[0]
    up = up if lim is None else up[up < lim * 15]
    return float(up[-1] + 1) / 15.0 if len(up) else 0.0


class _Hears:
    """a child that knows no word (written here), each act timed from a moment of her trial sentence as it perceives it (_moment:
    its ears 10 ms by 10 ms, her mouth, its own ears through body/sim/ears.Ears), `delay` ticks after it, fractions kept (the
    act on the tick floor(moment + delay) from her sentence's start), on a thing the word said never chooses: acts [(anchor,
    delay, kind, arg, hold)], kinds as _Clock's: "look" (its head on arg), "reach", "hit", "pain", "distress"."""

    def __init__(self, acts):
        self.acts, self.at_ = list(acts), []

    def heard(self, t, s):
        if s.line is not None and s.line.intent in C.TRIAL_INTENTS:
            perc = _perceived(s.clip)
            self.at_ = [(t + int(math.floor(_moment(perc, a) + d + 1e-9)), kind, arg, hold)
                        for a, d, kind, arg, hold in self.acts]

    def at(self, t):
        tg, reach, ev = None, [], []
        for t0, kind, arg, hold in self.at_:
            if kind == "look" and t0 <= t < t0 + hold:
                tg = arg
            elif kind == "reach" and t0 <= t < t0 + hold:
                reach.append(arg)
            elif kind in ("hit", "pain", "distress") and t == t0:
                ev.append(({"hit": "hit_her"}.get(kind, kind), None))
        return tg, None, None, tuple(reach), tuple(ev)


class _KnowsName:
    """a child that knows its name (written here): its head on her face 6 ticks after "pip" begins, never after a foil."""

    def __init__(self):
        self.t0 = None

    def show(self, a, b):
        pass

    def ended(self):
        pass

    def heard(self, t, s):
        if s.line is not None and s.line.intent in C.TRIAL_INTENTS:
            on = [a for w, a, _e in s.clip.words if w == LX.NAME][:1] if s.clip is not None else \
                ([3 * 2400 * TP.words(s.line.text).index(LX.NAME)] if LX.NAME in TP.words(s.line.text) else [])
            self.t0 = t + on[0] // 2400 + 6 if on else None

    def at(self, t):
        return ("mama" if self.t0 is not None and self.t0 <= t < self.t0 + 8 else None), None, None, (), ()


def _share_run(form, a, b, acts, flip, stage, voice, seed=5):
    """one trial of a child acting at ticks from her sentence's start (_Clock's anchors: "start" its first tick) -> its trial's
    row (_trials) with seq (the registered trials it added to its words' records), rel_end and onset (from the sentence's
    start)."""
    keep = []
    x = _timed_trial(form, a, b, _Clock(acts, channel=form != "name"), stage, seed, flip=flip, voice=voice,
                     scaffold=form != "name", keep=keep)[0]
    assert x["result"] not in ("dropped", None), (form, acts, flip, x)
    return dict(x, seq=sum(len(st["seq"]) for st in keep[0].ledger.words.values()), rel_end=x["end"] - x["t"],
                onset=x["open"] - x["t"])


def _share_sweep(form, a, b, stage, voice):
    """every child whose looks, reaches, turns, hits, pain and distress land on any tick after the test word's onset, the tick a
    function of the sentence it hears (so of any moment of its sound, however perceived), its looks on a thing the word never
    chooses; one act, or a look and an event, one at a fixed tick and the other on the sentence's: each trial run with each draw
    at each tick -> ([(config, {tick: (the draw's trial, the other draw's)})], its onset from the sentence's start)."""
    first = _share_run(form, a, b, [("start", 1, "sound", None, 1)], False, stage, voice)
    on = first["onset"]
    offs = list(range(on + 1, on + 33))
    looks = [("look", a), ("look", b), ("look", "mama"), ("reach", a), ("reach", b)] if form != "name" else [("look", "mama")]
    evs = ["hit", "pain", "distress"]
    cfgs = [(f"{k} {g}", lambda o, k=k, g=g: [("start", o, k, g, 8)]) for k, g in looks] + \
        [(e, lambda o, e=e: [("start", o, e, None, 1)]) for e in evs] + \
        ([("sound", lambda o: [("start", o, "sound", None, 2)])] if form == "name" else [])
    look = ("look", a) if form != "name" else ("look", "mama")
    for fixed in (on + 2, on + 8, on + 14, on + 20):
        for e in evs:
            cfgs.append((f"{look[1]} at +{fixed - on}, {e} after the word", lambda o, e=e, f_=fixed:
                         [("start", f_, look[0], look[1], 8), ("start", o, e, None, 1)]))
            cfgs.append((f"{look[1]} after the word, {e} at +{fixed - on}", lambda o, e=e, f_=fixed:
                         [("start", o, look[0], look[1], 8), ("start", f_, e, None, 1)]))
    out = []
    for name, mk in cfgs:
        out.append((name, {o: tuple(_share_run(form, a, b, mk(o), fl, stage, voice) for fl in (False, True)) for o in offs}))
    return out, on


def _share_life(kid, form, a, b, voice, seed, n_probes=24, toys=ITT_TOYS):
    """one life of formal trials through her conduct and ledger (its stub motion, still; stage 1 and 2 by seed; the words
    channel labelling her lines for a pair, silent for the name, as 4.8 runs them): n_probes probes of a form one after another,
    each pair's items fresh, the child in each -> the ledger."""
    con = _conduct(seed=seed, stage=1 + seed % 2, imperfect=False, voice=voice, ledger=_TapLedger(),
                   transcriber=Transcriber(None), scaffold=form != "name")
    _no_sets(con, tuple(o.id for o in toys))
    k, t, last, busy = 0, 0, -100, False
    while (k < n_probes or con.trial is not None or con.probes) and t < 400 * n_probes:
        if con.trial is None and not con.probes and k < n_probes and t > last + 30:
            if form == "name":
                con.probe("name")
            else:
                con.probe(form, a, b, new={a: f"{a}@{seed}.{k}", b: f"{b}@{seed}.{k}"})
            k += 1
        got = kid.at(t)
        s = con.tick(t, P(t, child_target=got[0], child_reaches=got[3], events=got[4], seen=toys), tract=got[1], token=got[2])
        kid.heard(t, s)
        now = con.trial is not None or bool(con.probes)
        if busy and not now:
            last = t
        busy = now
        t += 1
    return con.ledger


def _exploit(m_good, m_bad, target):
    """the delay that puts an act timed from a moment on tick `target` for the sentence whose moment is m_good and on the tick
    after it for m_bad (m_bad later): the eleventh and twelfth verifiers' construction -> the delay, or None if one tick cannot
    part them."""
    lo, hi = target + 1 - m_bad, target + 1 - m_good                  # floor(m_good + d) = target, floor(m_bad + d) = target + 1
    if not m_bad > m_good or hi <= lo:
        return None
    d = (lo + hi) / 2
    return d if math.floor(m_good + d + 1e-9) == target and math.floor(m_bad + d + 1e-9) == target + 1 else None


def test_trial_window_share():
    """the lead's decision A60b (P3's fifteenth round; it retires intention to treat, P3's thirteenth, with the first look it was
    built for): a trial is scored by the proportion of looking over a fixed window, the test word's onset tick + 2 to its onset
    tick + 23, the same ticks whichever is named: each tick her reading of its head line on the target, on the distractor or on
    neither (the name: on her face or not), and nothing else about the ticks, not when they fall nor their order; a pair's trial
    with fewer than 4 of them on either thing void, counted and never scored, as are the voids of her body, its pain, distress or
    hit. Tested: looks of the same length anywhere in the window, or split, or in another order, score alike, and ticks before
    and after it count for nothing; the void rule at 3 and 4 ticks; over every child whose acts land on any tick after the
    onset, the tick any function of the sentence it hears, its looks on a thing the word never chooses (the twelfth
    verifier's construction), in the suite's timed voice and her real voice, stage 1 and 2, a pair and the name: each trial and
    the same trial with the other thing named end alike, the same void and why on the same tick, the window's ticks on each
    thing the same; a knower's do not (the check tells it apart); the twelfth verifier's children, their one look timed from a
    moment within the test word's slot to the old window's close, over lives: every trial of the word that moment puts later
    void (fewer than 4 ticks), so none of its trials naming the other thing is scored and nothing is testable; an older save's
    record starts again, and a trial saved mid-window before this rule is void."""
    assert hasattr(K, "TRIAL_LOOK") and perm_test is not None, "a trial still decided by a look timed from its word (A60b)"
    assert hasattr(Ledger, "maps"), "one level of 'understood' only (A60b's two levels)"
    # (a) only how many of the window's ticks (+6 to +27 from her sentence's start in the timed voice, its onset at +4), never
    # when or in what order; before and after it nothing
    on0 = _share_run("place", "ball", "block_red", [], False, 1, None)["onset"]
    w0, w1 = on0 + 2, on0 + 23
    orders = {"8 on the ball at the window's start": [("start", w0, "look", "ball", 8)],
              "8 at its end": [("start", w1 - 7, "look", "ball", 8)],
              "4 and 4, apart": [("start", w0, "look", "ball", 4), ("start", w1 - 3, "look", "ball", 4)],
              "four of 2, among nothing": [("start", w0 + 3 * i, "look", "ball", 2) for i in range(4)],
              "8, and 6 before it and 6 after it": [("start", w0 - 6, "look", "ball", 6), ("start", w0 + 5, "look", "ball", 8),
                                                   ("start", w1 + 1, "look", "ball", 6)]}
    got_o = {k: _share_run("place", "ball", "block_red", acts, fl, 1, None) for k, acts in orders.items() for fl in (False,)}
    shares = {k: (x["result"], x["on"] + x["off"], x["on"] if x["target"]["id"] == "ball" else x["off"])
              for k, x in got_o.items()}
    assert set(shares.values()) == {("scored", 8, 8)}, shares
    mixed = {}
    for k, acts in (("ball then block", [("start", w0, "look", "ball", 6), ("start", w0 + 6, "look", "block_red", 6)]),
                    ("block then ball", [("start", w0, "look", "block_red", 6), ("start", w0 + 6, "look", "ball", 6)]),
                    ("interleaved", [("start", w0 + 2 * i, "look", ("ball", "block_red")[i % 2], 2) for i in range(6)])):
        x = _share_run("place", "ball", "block_red", acts, False, 1, None)
        mixed[k] = (x["result"], x["on"], x["off"])
    assert len(set(mixed.values())) == 1 and next(iter(mixed.values()))[1:] == (6, 6), mixed
    # (b) the void rule: 3 ticks on either thing void, counted, never scored; 4 scored
    v3 = _share_run("place", "ball", "block_red", [("start", w0, "look", "ball", 3)], False, 1, None)
    v4 = _share_run("place", "ball", "block_red", [("start", w0, "look", "ball", 4)], False, 1, None)
    assert v3["result"] == "void" and not v3["scored"] and v3["seq"] == 0 and "fewer than 4" in v3["why"] and \
        v4["result"] == "scored" and v4["seq"] == 2, (v3, v4)
    voices = [("timed", _TimedVoice())]
    tmp = None
    if _have_engine():
        tmp = tempfile.mkdtemp()
        voices.append(("real", V.VoiceCache(os.path.join(tmp, "v"), server=V.SynthServer(nice=19))))
    rows, lives = {}, {}
    try:
        # (c) the property: every child whose acts' ticks follow the sentence, its looks on a thing the word never chooses
        for vname, voice in voices:
            for form, a, b in (("place", "ball", "block_red"), ("name", None, None)):
                for stage in (1, 2):
                    sw, on = _share_sweep(form, a, b, stage, voice)
                    n_runs, res = 0, {}
                    for name, by_off in sw:
                        for o, (x, y) in by_off.items():
                            assert x["said"] != y["said"] and _same_but_named(x, y), (vname, form, stage, name, o, _oc(x),
                                                                                    _oc(y))
                            n_runs += 1
                            res[x["result"]] = res.get(x["result"], 0) + 1
                    rows[(vname, form, stage)] = (len(sw), n_runs, res)
        # (d) a knower's trials differ with the other thing named: the check tells it apart
        kn = [_timed_trial("place", "ball", "block_red", _Clock(knower=True), 2, s_, flip=fl)[0] for s_ in range(3)
              for fl in (False, True)]
        kn_name = [_timed_trial("name", None, None, _KnowsName(), 1, s_, flip=fl, scaffold=False)[0] for s_ in range(3)
                   for fl in (False, True)]
        assert not any(_same_but_named(kn[i], kn[i + 1]) for i in range(0, len(kn), 2)) and \
            not any(_same_but_named(kn_name[i], kn_name[i + 1]) for i in range(0, len(kn_name), 2)), \
            [_oc(x_) for x_ in kn + kn_name]
        # (e) the twelfth verifier's children: one look (or a turn) timed from a moment within the test word's slot to the old
        # window's close, over lives of 24 probes (the moments within the slot differ as the words do: C69)
        for vname, voice in voices:
            con = _conduct(seed=1, voice=voice, ledger=_TapLedger(), scaffold=True)
            clips = {}
            for w in (("ball", "bear", "block", "bottle", "car", "cup", "drum") if vname == "real" else ("ball", "block", "cup")):
                ln = TP.Line(f"where is the {w}? see?", "trial_where", "question", w, w, (), "fast", "where is the {o}? see?",
                             ST.shape("where", w)[0])
                clips[w] = con._clip(ln, heard=False)
            perc = {w: _perceived(c) for w, c in clips.items()}
            nouns = sorted(clips)
            pairs = [(u, v) for i, u in enumerate(nouns) for v in nouns[i + 1:]]
            onset = clips[nouns[0]].words[3][1] // 2400
            kids = []
            for label, anchor in (("a look 10 ms", ("in", "frame", -36)), ("a look at the ear", ("in", "ear", 0.7, 33.0)),
                                  ("a look at the mouth", ("in", "mouth", -30))):
                cand = [(u, v) for u, v in pairs if _moment(perc[u], anchor) != _moment(perc[v], anchor)]
                for u, v in cand[:1]:
                    mu, mv = _moment(perc[u], anchor), _moment(perc[v], anchor)
                    good, bad = (u, v) if mu < mv else (v, u)
                    d = _exploit(min(mu, mv), max(mu, mv), onset + 20)
                    if d is not None:
                        kids.append((f"{label}: on the {good} {d:.2f} ticks after {anchor}", good, bad,
                                     [(anchor, d, "look", good, 8)]))
            ids = {o.name: o.id for o in ITT_TOYS} if vname == "real" else dict(ball="ball", block="block_red", cup="cup")
            toys = ITT_TOYS if vname == "real" else TOYS
            for label, good, bad, acts in kids:
                und, n_sc, n_other = 0, 0, 0
                for s_ in range(6):
                    led = _share_life(_Hears(acts), "place", ids.get(good), ids.get(bad), voice, s_, toys=toys)
                    seq = led.words[good]["seq"]
                    und += led.maps(good)
                    n_sc += len(seq)
                    n_other += sum(1 for nm_, _a, _b in seq if not nm_)
                lives[(vname, label)] = (und, n_sc, n_other, 6)
        assert all(u == 0 for (u, _n, _o, _l) in lives.values()), lives
        # (f) an older save: its record starts again; a trial saved mid-window before this rule is void
        led = _TapLedger()
        _ledger_knows(led, "duck", 12, kind=True)
        assert led.understood("duck")
        st_ = led.state()
        st_["scoring"] = "itt-carrier"                            # P3's fourteenth round's ledger
        led2 = Ledger()
        led2.load_state(st_)
        assert not led2.understood("duck") and led2.words["duck"]["seq"] == [] and led2.forms == {}, led2.standing("duck")
        con = _conduct(seed=3, stage=1, imperfect=False, voice=_TimedVoice(), ledger=_TapLedger(),
                       transcriber=Transcriber(None), scaffold=True)
        _no_sets(con)
        con.probe("place", "ball", "block_red", new={"ball": "b@1", "block_red": "r@1"})
        t = 0
        while con.trial is None or con.trial["phase"] != "said":
            con.tick(t, P(t))
            t += 1
        st_ = json.loads(json.dumps(con.state(), default=_np))
        for k_ in ("look_from", "on", "off"):
            st_["trial"].pop(k_)
        con2 = _conduct(seed=3, stage=1, imperfect=False, voice=_TimedVoice(), ledger=_TapLedger(),
                        transcriber=Transcriber(None), scaffold=True)
        con2.load_state(st_)
        con2.tick(t, P(t))
        oc = [r for r in con2.ledger.rows if r["ev"] == "trial_outcome"]
        assert con2.trial is None and oc and oc[0]["result"] == "void" and "proportion of looking" in oc[0]["why"], oc
    finally:
        for _v, voice in voices[1:]:
            voice.close()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"70 the window's share of looking (A60b, P3's fifteenth round): its window +{w0 - on0} to +{w1 - on0} from the "
          f"test word's onset (ticks {w0}-{w1} of its sentence); 8 ticks on the ball at its start, at its end, 4 and 4 apart, "
          f"four of 2, "
          f"or 8 with 6 before it and 6 after it: each scored 8 of 8; 6 on each thing in either order or interleaved: each 6 to "
          f"6; 3 ticks on either thing void, counted, never scored, 4 scored; every child whose looks, reaches, turns, hits, "
          f"pain and distress land on any tick after the onset, a function of the sentence it hears, its looks on a thing the "
          f"word never chooses: each trial and the same trial with the other thing named alike (" + "; ".join(
              f"{v} voice {f} stage {s}: {c} children, {n} trials each way (" +
              ", ".join(f"{k} {m}" for k, m in sorted(r.items())) + ")"
              for (v, f, s), (c, n, r) in sorted(rows.items())) +
          f"); a knower's and a name-knower's differ; the twelfth verifier's children, one look timed from a moment within "
          f"the test word's slot, over 6 lives of 24 probes: " + "; ".join(
              f"{v} voice, {k}: mapped in {u} of {nl} ({n} registered trials, {o} of them naming the other thing)"
              for (v, k), (u, n, o, nl) in lives.items()) +
          "; an older save's record started again, a trial saved mid-window before A60b void")


# ------------------------------------------------------------ P3's fourteenth round: one carrier phrase, the test word under it
FAV_STOPS = (("frame", -44), ("frame", -36), ("frame", -30), ("frame", -22), ("frame", -18), ("mouth", -30), ("mouth", -22),
             ("mouth", -18), ("ear", 0.7, 21.0), ("ear", 0.7, 33.0), ("ear", 0.7, 45.0))
# the thirteenth verifier's children, as they stood on 13fb5dc (their moments measured there, in each voice): a favourite, a look
# at a thing 14 ticks after her voice starts and one timed from her sound's stop; (voice, favourite, moment, delay, which look is
# on the favourite). On 13fb5dc each one's two looks fell in the order the word said gave them, every pair it was shown in
FAV_KIDS = (("real", "drum", ("mouth", -22), 7.0, "other"), ("real", "bear", ("ear", 0.7, 21.0), 7.15, "fav"),
            ("real", "cup", ("frame", -44), 7.55, "fav"), ("timed", "cup", ("frame", -36), 7.3, "fav"))


class _Fav:
    """the thirteenth verifier's child (written here), knowing no word: a favourite thing and two looks, one `fixed` ticks after
    her voice starts, the other `delay` ticks after a moment of her sound as it perceives it (_moment: her sound's stop 10 ms by
    10 ms, her mouth, its own ears), each held 8 ticks, the later one only once the first has ended (a tie to the timed look,
    or to `tie`); on: which look is on its favourite ("other": the fixed one, the timed one on the other thing shown; "fav":
    the timed one); a look on a thing not shown is on nothing. learn (a seed): its delay drawn afresh from its own stream,
    0-20 ticks, whenever a trial with its favourite shown brings no smile (the verifier's learned child, steered by her smiles
    after met trials). Its thing never depends on the word said, unless the moment does."""

    def __init__(self, fav, anchor, delay, fixed=14, on="other", tie="timed", learn=None, shown=()):
        self.fav, self.anchor, self.delay, self.fixed, self.on, self.tie = fav, anchor, float(delay), int(fixed), on, tie
        self.shown, self.plan, self.smiled = tuple(shown), [], False
        self.rng = None if learn is None else \
            np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(learn), spawn_key=(97,))))
        if self.rng is not None:
            self.delay = float(self.rng.uniform(0, 20))

    def show(self, *things):
        self.shown = tuple(things)

    def heard(self, t, s):
        if any(j[1] == "met_trial" for j in (s.judgments or ())):
            self.smiled = True
        if s.line is None or s.line.intent not in C.TRIAL_INTENTS:
            return
        m = _moment(_perceived(s.clip), self.anchor)
        other = next((x for x in self.shown if x != self.fav), None)
        fav = self.fav if self.fav in self.shown else None
        tf, tt = t + self.fixed, t + int(math.floor(m + self.delay + 1e-9))
        fixed_on, timed_on = (fav, other) if self.on == "other" else (other, fav)
        looks = sorted([(tf, 1 if self.tie == "timed" else 0, fixed_on), (tt, 0 if self.tie == "timed" else 1, timed_on)])
        self.plan = [(looks[0][0], looks[0][2]), (max(looks[1][0], looks[0][0] + 8), looks[1][2])]

    def ended(self):
        """its trial ended: with its favourite shown and no smile, a learner draws its delay afresh."""
        if self.rng is not None and self.fav in self.shown and not self.smiled:
            self.delay = float(self.rng.uniform(0, 20))
        self.smiled = False

    def at(self, t):
        tg = next((x for t0, x in self.plan if t0 <= t < t0 + 8), None)
        return tg, None, None, (), ()


class _Acts(_Hears):
    """a child that knows no word (written here): _Hears's acts timed from a moment of her sound as it perceives it, and its
    voice (kind "sound": 2 ticks of its tract) and its "mama" token among them."""

    def at(self, t):
        tg, _snd, _tok, reach, ev = super().at(t)
        snd = next((_burst()[t - t0] for t0, kind, _a, _h in self.at_ if kind == "sound" and t0 <= t < t0 + 2), None)
        tok = LX.WORD_ID["mama"] if any(kind == "mama" and t == t0 for t0, kind, _a, _h in self.at_) else None
        return tg, snd, tok, reach, ev


class _Gate:
    """the thirteenth verifier's name child (written here), knowing no name: its head on her face `delay` ticks after a moment of
    her sound as it perceives it when that comes before `gate` ticks after her voice starts; otherwise, from then, on a toy for
    40 ticks (no turn at all)."""

    def __init__(self, anchor, delay, gate=15, toy="ball"):
        self.anchor, self.delay, self.gate, self.toy, self.plan = anchor, float(delay), int(gate), toy, None

    def heard(self, t, s):
        if s.line is not None and s.line.intent in C.TRIAL_INTENTS:
            tt = t + int(math.floor(_moment(_perceived(s.clip), self.anchor) + self.delay + 1e-9))
            self.plan = (tt, "mama", 8) if tt < t + self.gate else (t + self.gate, self.toy, 40)

    def ended(self):
        pass

    def at(self, t):
        tg = self.plan[1] if self.plan is not None and self.plan[0] <= t < self.plan[0] + self.plan[2] else None
        return tg, None, None, (), ()


class _Onset:
    """a child (written here) whose act lands on the test word's onset tick itself, whatever its word: its view of the two
    things lost on that tick alone (what it sees: _timed_trial's sixth; it looks at neither after), or a look she first reads
    on that tick (the harness's child_target is her reading), or on the tick before it."""

    def __init__(self, kind, onset=4, thing="ball"):
        self.kind, self.onset, self.thing, self.t0 = kind, onset, thing, None

    def heard(self, t, s):
        if s.line is not None and s.line.intent in C.TRIAL_INTENTS:
            self.t0 = t

    def at(self, t):
        if self.t0 is None:
            return None, None, None, (), ()
        u = t - self.t0
        if self.kind == "view":
            blind = tuple(seen1(o.id, o.name, o.colour, o.on, False) for o in TOYS) if u == self.onset else None
            return None, None, None, (), (), blind
        start = self.onset - (0 if self.kind == "look_on" else 1)
        return (self.thing if start <= u < start + 8 else None), None, None, (), ()


def _fav_life(kid, pairs, voice, seed, toys=ITT_TOYS, form="place", flip=False):
    """one life of formal trials through her conduct and ledger (its stub motion, still; stage 1 and 2 by seed; the words
    channel labelling her lines for a pair, silent for the name): a probe for each pair in turn (each pair's items fresh), the
    child shown the two things; flip: every draw of which is named turned over (_Flip) -> the ledger."""
    con = _conduct(seed=seed, stage=1 + seed % 2, imperfect=False, voice=voice, ledger=_TapLedger(),
                   transcriber=Transcriber(None), scaffold=form != "name")
    if flip:
        con.trial_rng = _Flip(con.trial_rng)
    _no_sets(con, tuple(o.id for o in toys))
    k, t, last, busy = 0, 0, -100, False
    while (k < len(pairs) or con.trial is not None or con.probes) and t < 400 * len(pairs):
        if con.trial is None and not con.probes and k < len(pairs) and t > last + 30:
            a, b = pairs[k]
            if form == "name":
                con.probe("name")
            else:
                con.probe(form, a, b, new={a: f"{a}@{seed}.{k}", b: f"{b}@{seed}.{k}"})
                kid.show(a, b)
            k += 1
        got = kid.at(t)
        s = con.tick(t, P(t, child_target=got[0], child_reaches=got[3], events=got[4], seen=toys), tract=got[1], token=got[2])
        kid.heard(t, s)
        now = con.trial is not None or bool(con.probes)
        if busy and not now:
            last = t
            kid.ended()
        busy = now
        t += 1
    return con.ledger


def _levels_ear(pcm, gain_db, place):
    """what the child's own ears hear of a clip, her mouth at place (metres, degrees round its head in the plane of its ears):
    each ear's level each 10 ms frame (dB SPL, its cochleas' band powers summed), and whether each tick's onset was heard."""
    from body.sim import ears as E                                       # noqa: PLC0415
    d, adeg = place
    x = np.asarray(pcm, np.float64) * (V.PA_PER_UNIT / 32767.0 * 10 ** (gain_db / 20))
    n = -(-len(x) // 2400) + 2
    x = np.pad(x, (0, n * 2400 - len(x)))
    pos = (E.EAR_SITES["L"] + E.EAR_SITES["R"]) / 2 + np.array([d * math.cos(math.radians(adeg)),
                                                                d * math.sin(math.radians(adeg)), 0.0])
    ears, lv, ev = E.Ears(), ([], []), []
    for u in range(n):
        h = ears.tick(E.EAR_SITES["L"], E.EAR_SITES["R"], {"mama": (x[u * 2400:(u + 1) * 2400], pos)})
        for side, acc in ((h.left, lv[0]), (h.right, lv[1])):
            acc.extend(10 * np.log10(np.maximum((side.astype(np.float64) ** 3 * E.E_CODE).sum(1), 1e-30) / E.P_REF ** 2))
        ev.append(h.events)
    return np.array(lv[0]), np.array(lv[1]), ev


def _first_last(lv, grid):
    """for each level of grid, the first and the last frame above it (None: none)."""
    out = []
    for L in grid:
        up = np.nonzero(lv > L)[0]
        out.append((int(up[0]), int(up[-1])) if len(up) else None)
    return out


def test_trial_carrier_phrase():
    """P3's fourteenth round (the thirteenth verifier's finding 1, the lead's acceptance test): each test word had one recording,
    so where its sound stopped as the child perceives it (her mouth's last opening above -22 dB, its ears at 0.7 m above 21 dB
    SPL, her last 10 ms above -44 dB) was the same against every other word: "drum" stopped last, "bear" and "cup" first. A child
    that knew no word, with a favourite toy, one look at a fixed tick and one timed from her sound's stop, had the order of its
    two looks follow the word said: "drum" understood in 12 of 12 lives whatever its distractor (P4's schedule and all 21 pairs),
    "bear" and "cup" alike, a child that learned its delay from her smiles in 9 of 12, and "pip" in 12 of 12 by a turn 12.7
    ticks after her sound stopped, gated at tick 15, where her draws alone gave 1 or 2. Now every sentence of a form is one
    carrier phrase, as looking-while-listening's "where's the X? can you find it?" (stimuli.parts, splice): one recording before
    the test word, its own sentence through its slot at the form's one loudest 10 ms, one tag after it ("see?"; the name's "hi.
    X. hi."), the slot under the loudest before it and after it at every window from 2.5 ms to a tick and at the child's own
    ears (the level ceiling), so her voice's start and her sound's stop at every level lie in the carrier and the tag: one
    moment, to the sample, whichever is named. Tested: on the rendered audio (her real voice where the engine is present; the
    suite's timed voice always), each form's sentences the same outside the slot sample for sample, and the first and last
    moment above every level from -90 to 0 dB one moment for all its words, 10 ms by 10 ms, tick by tick (her mouth) and at
    the child's own ears (either ear, near and far, before and behind it), their onsets the same; the name and its foils at one
    loudest 10 ms (C66); the verifier's children, and every child with a favourite and two looks timed from her voice's start
    or her sound's stop at 11 levels and measures, delays, fixed ticks, ties and roles, every gated name child, and every child
    with one act of any kind (a look, a reach, its voice, its token, a hit, pain, distress) timed from her sound's stop so
    perceived at offsets from -4 to +30, each trial the same with the other thing named (the lead's invariance property); over
    lives, since the lead's decision A60b, the verifier's children, the learned child and the gated name child "understood" in
    at most 1 of 12 (the rule's tests' levels sum under 0.01 a life); the window from the onset + 2; an older save's stimuli
    void."""
    assert ST is not None and hasattr(ST, "parts"), "no carrier phrase for the test word (P3's fourteenth round)"
    assert hasattr(Ledger, "maps"), "one level of 'understood' only (A60b's two levels)"
    voices = [("timed", _TimedVoice())]
    tmp = None
    if _have_engine():
        tmp = tempfile.mkdtemp()
        voices.append(("real", V.VoiceCache(os.path.join(tmp, "v"), server=V.SynthServer(nice=19))))
    grid = np.arange(-90.0, 0.01, 0.5)
    places = ((0.3, 0), (0.7, 0), (0.7, 90), (0.7, -90), (1.5, 180))
    audio, sweep, lives, gates, onset_rows = {}, {}, {}, {}, {}
    try:
        # (A) the rendered audio: one moment at every level, in every measure the child has
        for vname, voice in voices:
            con = _conduct(seed=1, voice=voice, ledger=_TapLedger(), scaffold=False)
            for key in ("where", "name"):
                f = ST.STIMULI[key]
                ws = sorted(f["words"]) if vname == "real" else (["ball", "block", "cup"] if key == "where" else
                                                                  [LX.NAME] + list(K.NAME_FOILS))
                clips = {}
                for w in ws:
                    fr = TP.FRAMES["trial_where" if key == "where" else "trial_name"][0][0]
                    ln = TP.Line(fr.replace("{o}", w).replace("{n}", w), "trial_where" if key == "where" else "trial_name",
                                 f["register"], w, w, (), "fast", fr, ST.shape(key, w)[0])
                    clips[w] = con._clip(ln, heard=False)
                c0 = clips[ws[0]]
                at, tag_at = c0.spliced["at"], c0.spliced["tag_at"]
                for w, c in clips.items():                        # outside the slot, sample for sample
                    assert np.array_equal(c.pcm[:at], c0.pcm[:at]) and np.array_equal(c.pcm[tag_at:], c0.pcm[tag_at:]) and \
                        len(c.pcm) == len(c0.pcm) and ST.headroom(c.pcm, at, tag_at) > 0, (vname, key, w)
                n_lv, n_meas = 0, 0
                fr10 = {w: 20 * np.log10(np.maximum(V.frame_rms(np.asarray(c.pcm)), 1e-12)) for w, c in clips.items()}
                tks = {w: ST.levels(c.pcm)[0] for w, c in clips.items()}
                for name, lvs in (("10 ms", fr10), ("tick", tks)):
                    ref = _first_last(lvs[ws[0]], grid)
                    for w in ws[1:]:
                        assert _first_last(lvs[w], grid) == ref, (vname, key, name, w)
                    n_lv += len(grid)
                    n_meas += 1
                for pl in places:
                    got = {w: _levels_ear(c.pcm, getattr(c, "gain_db", 0.0), pl) for w, c in clips.items()}
                    for e in (0, 1):
                        ref = _first_last(got[ws[0]][e], grid)
                        for w in ws[1:]:
                            assert _first_last(got[w][e], grid) == ref, (vname, key, pl, e, w)
                        n_lv += len(grid)
                        n_meas += 1
                    slot = range(at // 2400, tag_at // 2400 + 1)             # (its ear's tick: the slot's reach)
                    for w in ws[1:]:
                        assert [x for u, x in enumerate(got[w][2]) if u not in slot] == \
                            [x for u, x in enumerate(got[ws[0]][2]) if u not in slot], (vname, key, pl, w)
                peaks = {}                                        # its loudest 10 ms, wherever it falls (the form's level)
                for w, c in clips.items():
                    x = np.asarray(c.pcm, np.float64) / 32767.0
                    cs = np.concatenate([[0.0], np.cumsum(x * x)])
                    e = (cs[at - 159 + 160:tag_at + 160] - cs[at - 159:tag_at]) / 160.0
                    peaks[w] = 10 * math.log10(max(float(e.max()), 1e-30))
                if key == "name" and vname == "real":            # C66: the name and its foils at one loudest 10 ms
                    assert max(peaks.values()) - min(peaks.values()) < 0.05, peaks
                audio[(vname, key)] = (len(ws), n_meas, n_lv, round(min(ST.headroom(c.pcm, at, tag_at) for c in
                                                                        clips.values()), 2),
                                       round(max(peaks.values()) - min(peaks.values()), 3))
        # (B) the lead's invariance property for the verifier's kinds of child: each trial the same with the other thing named
        for vname, voice in voices:
            toys = ITT_TOYS if vname == "real" else TOYS
            pairs = (("drum", "ball"), ("bear", "block"), ("cup", "bottle")) if vname == "real" else \
                (("cup", "ball"), ("ball", "block_red"))
            n_cfg, res = 0, {}
            for fav, other in pairs:
                for anchor in FAV_STOPS + (("start",),):
                    for d in (-2.0, 0.55, 3.3, 6.7, 7.15, 9.25, 12.5):
                        for fixed in (6, 14, 19):
                            for on in ("other", "fav"):
                                for tie in (("timed", "fixed") if fixed == 14 else ("timed",)):
                                    runs = [_timed_trial("place", fav, other, _Fav(fav, anchor, d, fixed, on, tie,
                                                                                    shown=(fav, other)),
                                                         1 + n_cfg % 2, n_cfg % 5, flip=fl, voice=voice, seen_=toys)[0]
                                            for fl in (False, True)]
                                    xo, yo = (_oc(r) for r in runs)
                                    assert xo[0] != "dropped" and _inv_same(xo, yo), (vname, fav, anchor, d, fixed, on, tie,
                                                                                      xo, yo)
                                    n_cfg += 1
                                    res[xo[0]] = res.get(xo[0], 0) + 1
            n_gate = 0
            for anchor in FAV_STOPS + (("start",),):
                for d in (0.0, 2.3, 5.0, 8.1, 12.7, 16.4):
                    for gate in (8, 12, 15, 19, 25):
                        runs = [_timed_trial("name", None, None, _Gate(anchor, d, gate), 1 + n_gate % 2, n_gate % 5,
                                             flip=fl, voice=voice, scaffold=False)[0] for fl in (False, True)]
                        xo, yo = (_oc(r) for r in runs)
                        assert xo[0] != "dropped" and _inv_same(xo, yo), (vname, "name", anchor, d, gate, xo, yo)
                        n_gate += 1
                        res["name " + str(xo[0])] = res.get("name " + str(xo[0]), 0) + 1
            n_acts = 0                                           # every kind of act, timed from her sound's stop as it
            for form, a, b in (("place", pairs[0][0], pairs[0][1]), ("name", None, None)):     # perceives it, at every other
                kinds = ([("look", a), ("look", b), ("look", "mama"), ("reach", a), ("reach", b)] if form != "name" else
                         [("look", "mama")]) + [("sound", None), ("mama", None), ("hit", None), ("pain", None),
                                                ("distress", None)]
                for anchor in FAV_STOPS:                         # offset from -4 to +30
                    for off in range(-4, 31, 2):
                        for kind, arg in kinds:
                            acts = [(anchor, float(off), kind, arg, 8)]
                            runs = [_timed_trial(form, a, b, _Acts(acts), 1 + n_acts % 2, n_acts % 5, flip=fl, voice=voice,
                                                 seen_=toys, scaffold=form != "name")[0] for fl in (False, True)]
                            xo, yo = (_oc(r) for r in runs)
                            assert xo[0] != "dropped" and _inv_same(xo, yo), (vname, form, acts, xo, yo)
                            n_acts += 1
            sweep[vname] = (n_cfg, n_gate, res, n_acts)
        # (C) the verifier's children over lives: P4's schedule (the favourite against each other toy in turn, 24 probes) and
        # all 21 pairs at random (72 probes), 12 lives each; the learned child; the gated name child: each a child whose
        # looking does not depend on the word said (her sound's stop one moment, however it hears it), so it reaches
        # "understood" in at most 1 life in 100 by construction (A60b: its tests' levels sum under 0.01)
        vmap = dict(voices)
        for vname, fav, anchor, d, on in FAV_KIDS:
            if vname not in vmap:
                continue
            voice = vmap[vname]
            toys = ITT_TOYS if vname == "real" else TOYS
            ids = [o.id for o in toys if o.child_sees and (vname == "real" or o.id in ("ball", "cup", "block_red"))]
            others = [x for x in ids if x != fav]
            sched = {"P4": [(fav, others[k % len(others)]) for k in range(24)]}
            if vname == "real" and fav == "drum":
                r_ = np.random.default_rng(71)
                allp = [(u, v) for i, u in enumerate(ids) for v in ids[i + 1:]]
                sched["21 pairs"] = [allp[int(r_.integers(len(allp)))] for _ in range(72)]
            for sname, pairs in sched.items():
                for label, mk in (("timed", lambda s_: _Fav(fav, anchor, d, 14, on)),
                                  ("control, no timing", lambda s_: _Fav(fav, ("start",), 99.0, 14, on)),
                                  ("learned", lambda s_: _Fav(fav, anchor, d, 14, on, learn=500 + s_))):
                    if label != "timed" and (sname != "P4" or fav != FAV_KIDS[0][1] and vname == "real"):
                        continue
                    und, words_und, n_sc = 0, {}, []
                    for s_ in range(12):
                        led = _fav_life(mk(s_), pairs, voice, s_, toys=toys)
                        st = led.words.get(fav) or {"seq": []}
                        und += led.maps(fav)
                        n_sc.append(len(st["seq"]))
                        for w_, st_ in led.words.items():
                            if st_.get("seq") and w_ != fav:
                                words_und[w_] = words_und.get(w_, 0) + int(led.maps(w_))
                    assert und <= 1 and all(v_ <= 1 for v_ in words_und.values()), (vname, fav, sname, label, und, words_und)
                    lives[(vname, fav, sname, label)] = (und, sum(n_sc), words_und)
        for vname, voice in voices:                               # the gated name child, its turn 12.7 ticks after her
            und = 0                                               # last 10 ms above -36 dB, gated at tick 15
            for s_ in range(12):
                led = _fav_life(_Gate(("frame", -36), 12.7, 15), [(None, None)] * 24, voice, s_,
                                toys=ITT_TOYS if vname == "real" else TOYS, form="name")
                und += led.understood(LX.NAME)
            assert und <= 1, (vname, "gate", und)
            gates[vname] = und
        # (D) the window from the onset + 2 (A60b): a look first read on the onset's own tick or the one before counts only
        # for its ticks in the window; its view lost on the onset's tick alone, looking at nothing: void, fewer than 4
        for kind in ("view", "look_on", "look_before"):
            x = _timed_trial("place", "ball", "block_red", _Onset(kind), 1, 3)[0]
            y = _timed_trial("place", "ball", "block_red", _Onset(kind), 1, 3, flip=True)[0]
            onset_rows[kind] = (x["result"], x["on"] + x["off"] if x["result"] == "scored" else None, _same_but_named(x, y))
        assert onset_rows == {"view": ("void", None, True), "look_on": ("scored", 6, True),
                              "look_before": ("scored", 5, True)}, onset_rows
        # (E) an older save: its record starts again; a trial saved before its stimuli were one carrier phrase is void
        led = _TapLedger()
        _ledger_knows(led, "duck", 12, kind=True)
        st_ = led.state()
        st_["scoring"] = "itt"                                    # P3's thirteenth round's ledger
        led2 = Ledger()
        led2.load_state(st_)
        assert led.understood("duck") and not led2.understood("duck") and led2.words["duck"]["seq"] == [], led2.standing("duck")
        con = _conduct(seed=3, stage=1, imperfect=False, voice=_TimedVoice(), ledger=_TapLedger(),
                       transcriber=Transcriber(None), scaffold=True)
        _no_sets(con)
        con.probe("place", "ball", "block_red", new={"ball": "b@1", "block_red": "r@1"})
        for t in range(70):
            con.tick(t, P(t))
        st_ = json.loads(json.dumps(con.state(), default=_np))
        st_["trial"].pop("slot")
        con2 = _conduct(seed=3, stage=1, imperfect=False, voice=_TimedVoice(), ledger=_TapLedger(),
                        transcriber=Transcriber(None), scaffold=True)
        con2.load_state(st_)
        con2.tick(70, P(70))
        oc = [r for r in con2.ledger.rows if r["ev"] == "trial_outcome"]
        assert con2.trial is None and oc and oc[0]["result"] == "void" and "one carrier phrase" in oc[0]["why"], oc
    finally:
        for _v, voice in voices[1:]:
            voice.close()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"71 one carrier phrase, the test word under it (P3's fourteenth round, the thirteenth verifier's finding 1): each "
          f"form's sentences the same outside the test word's slot, sample for sample, the slot under its ceiling, and the "
          f"first and last moment above every level from -90 to 0 dB in 0.5 dB steps one moment for all its words (" + "; ".join(
              f"{v} voice {k}: {n} words, {m} measures (10 ms, her mouth's tick, either ear at {len(places)} places), {nl} "
              f"level crossings; the ceiling's least margin at the clip {h} dB; the loudest 10 ms in the slot within {pk} dB"
              for (v, k), (n, m, nl, h, pk) in sorted(audio.items())) +
          f"); each trial the same with the other thing named (" + "; ".join(
              f"{v} voice: {nc} children with a favourite and two looks (her voice's start or her sound's stop at "
              f"{len(FAV_STOPS)} levels and measures, 7 delays, 3 fixed ticks, either look on the favourite, ties either way) "
              f"and {ng} gated name children (" + ", ".join(f"{k} {c}" for k, c in sorted(r.items())) + f"), and {na} "
              f"children each with one act, a look or a reach on either thing or her face, its voice, its token, a hit, pain "
              f"or distress, timed from her sound's stop at those levels and measures at every other offset from -4 to +30, "
              f"a pair and the name"
              for v, (nc, ng, r, na) in sweep.items()) +
          f"); over 12 lives each, by the proportion of looking and its tests (A60b), its word's level 1: " + "; ".join(
              f"{v} voice, the {fv} child ({sn}{'' if lb == 'timed' else ', ' + lb}): '{fv}' mapped in {u} of 12 ({n} "
              f"registered trials), other words {dict(sorted(wu.items()))}"
              for (v, fv, sn, lb), (u, n, wu) in lives.items()) +
          "; the gated name child: " + "; ".join(f"{v} voice 'pip' understood in {u} of 12" for v, u in gates.items()) +
          f"; the window from the onset + 2: its view lost on the onset's own tick alone, no look: void; a look read from it "
          f"and from the tick before it, 8 ticks: {onset_rows['look_on'][1]} and {onset_rows['look_before'][1]} of them "
          f"counted, the same with the other thing named; an older save's record started again, its trial saved before one "
          f"carrier phrase void")


# ------------------------------------------------ the lead's decision A60b: the proportion of looking, tested by permutation
BAND_KIDS = (("drum", ("band", "L", 1, 48.0, "stop")), ("drum", ("band", "L", 3, 62.5, "stop")),
             ("bear", ("band", "L", 22, 51.5, "stop")), ("cup", ("band", "L", 2, 56.0, "stop")),
             ("car", ("band", "L", 7, 61.0, "start")))
# the fourteenth verifier's children (f5ed8cd): a favourite, and its timed look's moment one band of its own left ear, her mouth
# 0.7 m ahead (107 Hz, 167 Hz, 1684 Hz, 136 Hz: body/sim/ears.Ears' bands 1, 3, 22, 2; the car's the 323 Hz band's start,
# band 7); the name's, its turn keyed on the 2667 Hz band's stop (band 27) above 48.5 dB SPL
BAND_NAME = ("band", "L", 27, 48.5, "stop")


class _GateLook:
    """a child that knows no word (written here), the fourteenth verifier's construction turned to the window (A60b): `gate`
    ticks after her voice starts it looks for `hold` ticks at its favourite when a moment of her sound as it perceives it
    (_moment, `delay` ticks after it) has not come before the gate (late; or, not late, when it has), else at the other thing
    shown: a gate on the moment, not an order of looks. Timed from a moment within the test word's slot, its looking follows
    the word's own sound."""

    def __init__(self, fav, anchor, delay, gate, hold=16, shown=(), late=True):
        self.fav, self.anchor, self.delay, self.gate, self.hold = fav, anchor, float(delay), int(gate), int(hold)
        self.shown, self.plan, self.late = tuple(shown), None, bool(late)

    def show(self, *things):
        self.shown = tuple(things)

    def heard(self, t, s):
        if s.line is None or s.line.intent not in C.TRIAL_INTENTS:
            return
        tt = int(math.floor(_moment(_perceived(s.clip), self.anchor) + self.delay + 1e-9))
        other = next((x for x in self.shown if x != self.fav), None)
        self.plan = (t + self.gate, self.fav if (tt >= self.gate) == self.late else other)

    def ended(self):
        pass

    def at(self, t):
        tg = self.plan[1] if self.plan is not None and self.plan[0] <= t < self.plan[0] + self.hold else None
        return tg, None, None, (), ()


class _Knows:
    """a child that knows the words (written here): from its test word's onset + `lat` ticks it looks for `hold` ticks at the
    thing the word names, with chance p, else at another thing shown; before its trial `from_` it knows none (a coin between the
    two: a child that learns the word then). trained None: the word names every thing of its kind (a knower of the kind:
    "cup_2" is a cup); else only the trained things among `trained` (a knower of those things alone: a new exemplar is none of
    them). A word naming nothing shown (its thing absent, a foil): its look on one of the things shown, each as likely, or
    (spread) its looks on each in turn, `hold` ticks among them, in an order of its own. Its own stream."""

    def __init__(self, p=1.0, lat=4, hold=12, from_=0, seed=0, shown=(), trained=None, spread=False):
        self.p, self.lat, self.hold, self.from_, self.shown, self.plan, self.k = p, lat, hold, from_, tuple(shown), None, 0
        self.trained = None if trained is None else frozenset(trained)
        self.spread = bool(spread)
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(98,))))

    def show(self, *things):
        self.shown = tuple(things)

    def _coin(self, xs):
        """one of xs, each as likely (its own stream: for two, the first under 0.5)."""
        xs = list(xs)
        return xs[min(len(xs) - 1, int(self.rng.random() * len(xs)))]

    @staticmethod
    def _onset(s):
        return ([a for w, a, _e in s.clip.words if w == s.line.focus][:1] if s.clip is not None else [9 * 2400])[0] // 2400

    def heard(self, t, s):
        if s.line is None or s.line.intent not in C.TRIAL_INTENTS:
            return
        self.k += 1
        named = next((x for x in self.shown if x is not None and x.split("_")[0] == s.line.focus and
                      (self.trained is None or x in self.trained)), None)
        others = [x for x in self.shown if x != named]
        right = self.rng.random() < (self.p if self.k > self.from_ else 0.5)
        if named is None and self.spread:                 # its thing absent, or a foil: its looks on each thing in turn
            order = [self.shown[i] for i in self.rng.permutation(len(self.shown))]
            d, t0 = max(1, self.hold // len(order)), t + self._onset(s) + self.lat
            self.plan = [(t0 + i * d, x, d) for i, x in enumerate(order)]
            return
        if named is None:                                 # or its look on any thing shown, each as likely
            self.plan = (t + self._onset(s) + self.lat, self._coin(self.shown))
            return
        self.plan = (t + self._onset(s) + self.lat, named if right else others[0] if len(others) == 1 else self._coin(others))

    def ended(self):
        pass

    def at(self, t):
        if isinstance(self.plan, list):
            return next((x for t0, x, n in self.plan if t0 <= t < t0 + n), None), None, None, (), ()
        tg = self.plan[1] if self.plan is not None and self.plan[0] <= t < self.plan[0] + self.hold else None
        return tg, None, None, (), ()


class _Excludes(_Knows):
    """a child that knows every object word but `unknown` (written here), at its words as _Knows is; at a word it does not know
    (its unknown word, a foil), by EXCLUSION (mutual exclusivity, Markman and Wachtel 1988; Halberda 2003): its look on a thing
    shown whose word it does not know, each as likely (none: any thing shown). often: it turns from what it can name only at a
    word it has heard at least `often` times before (in any of her lines; carrier: in her trial sentences alone), else its look
    on any thing shown (a contrivance: infants exclude at a novel word most of all). Its own stream."""

    def __init__(self, unknown, seed=0, often=None, lat=4, hold=12, carrier=False):
        super().__init__(seed=seed, lat=lat, hold=hold)
        self.unknown, self.often, self.count, self.carrier = unknown, often, {}, bool(carrier)
        self.words = frozenset(TP.OBJECT_NOUNS) - {unknown}

    def heard(self, t, s):
        if s.line is not None and s.line.intent in C.TRIAL_INTENTS:
            w = s.line.focus
            mine = [x for x in self.shown if x.split("_")[0] == w and w in self.words]
            if mine:
                tg = mine[0]
            elif w not in self.words and (self.often is None or self.count.get(w, 0) >= self.often):
                tg = self._coin([x for x in self.shown if x.split("_")[0] not in self.words] or self.shown)
            else:
                tg = self._coin(self.shown)
            self.plan = (t + self._onset(s) + self.lat, tg)
        if s.line is not None and (not self.carrier or s.line.intent in C.TRIAL_INTENTS):
            for w in TP.words(s.line.text):
                self.count[w] = self.count.get(w, 0) + 1


class _Newest(_Knows):
    """a child (written here) that counts how often it has been shown each thing. knows None: its look on the newest thing shown
    (the least shown; ties, each as likely) whatever is said. "kind": a knower of the kind, its look on the thing a word names
    when shown, and at a word naming nothing shown (a novel name) on the newest thing shown (novel name, nameless category:
    Mervis and Bertrand 1994). A list: a knower of those trained things alone (_Knows' trained), its look on a trained thing's
    word's thing when shown and on the newest thing shown when it is not, and at any other word (a foil) on any thing shown.
    Its own stream."""

    def __init__(self, seed=0, knows=None, lat=4, hold=12):
        super().__init__(seed=seed, trained=None if knows in (None, "kind") else knows, lat=lat, hold=hold)
        self.knows, self.n_shown = knows, {}

    def show(self, *things):
        super().show(*things)
        for x in things:
            self.n_shown[x] = self.n_shown.get(x, 0) + 1

    def _newest(self):
        m = min(self.n_shown.get(x, 0) for x in self.shown)
        return self._coin([x for x in self.shown if self.n_shown.get(x, 0) == m])

    def heard(self, t, s):
        if s.line is None or s.line.intent not in C.TRIAL_INTENTS:
            return
        w = s.line.focus
        if self.knows is None:
            tg = self._newest()
        elif self.knows == "kind":
            tg = next((x for x in self.shown if x.split("_")[0] == w), None) or self._newest()
        elif w in {x.split("_")[0] for x in self.trained}:
            tg = next((x for x in self.shown if x in self.trained and x.split("_")[0] == w), None) or self._newest()
        else:
            tg = self._coin(self.shown)
        self.plan = (t + self._onset(s) + self.lat, tg)


def _zoo_pairs(vname, fav, n=24):
    """P4's schedule: the favourite against each other toy shown in turn."""
    ids = [o.id for o in (ITT_TOYS if vname == "real" else TOYS) if o.child_sees and
           (vname == "real" or o.id in ("ball", "cup", "block_red"))]
    others = [x for x in ids if x != fav]
    return [(fav, others[k % len(others)]) for k in range(n)]


def _zoo_life(label, vname, voice, seed, mk, fav, form="place", n=24, flip=False):
    """one life of the zoo (_fav_life: her conduct, its stub motion still, stage 1 and 2 by seed; flip: every draw turned over)
    -> the ledger."""
    toys = ITT_TOYS if vname == "real" else TOYS
    pairs = [(None, None)] * n if form == "name" else _zoo_pairs(vname, fav, n)
    return _fav_life(mk(seed), pairs, voice, seed, toys=toys, form=form, flip=flip)


def _zoo_digests(seeds=(0, 1)):
    """lives of the zoo in the timed voice for the determinism check (a learner, the gated name child, a knower) -> [[the ledger's
    chain, the sha256 of its rows]]."""
    import hashlib                                                        # noqa: PLC0415
    out = []
    for s_ in seeds:
        for mk, fav, form in ((lambda x: _Fav("cup", ("frame", -36), 7.3, 14, "fav", learn=500 + x), "cup", "place"),
                              (lambda x: _Gate(("frame", -36), 12.7, 15), None, "name"),
                              (lambda x: _Knows(p=0.8, seed=x), "ball", "place")):
            led = _zoo_life("", "timed", _TimedVoice(), s_, mk, fav, form)
            out.append([led.digest, hashlib.sha256(json.dumps(led.rows, sort_keys=True).encode()).hexdigest()])
    return out


def test_trial_acceptance_a60b():
    """the lead's decision A60b and its acceptance (P3's fifteenth round): understanding scored by the proportion of looking
    over a fixed window (the test word's onset tick + 2 to + 23, the same whichever is named), a trial with fewer than 4 of its
    ticks on either thing void, and "understood" by a permutation test over the draw labels, its tests' levels summing to 0.01 a
    life. Measured and reported: the zoo of children that know no word, 12 lives each through her conduct and ledger (P4's
    schedule, the favourite against each other toy in turn, 24 probes; the name 24): a favourite that looks only at it, the
    thirteenth verifier's children timed from her sound's stop at 11 levels and measures (her voice's start, 10 ms frames, her
    mouth, its own ears), children timed from 12 anchors of what they hear and see (_Clock's), the gated name child, the child
    that would learn its delay from her smiles (none since A60b: no feedback in a trial), in the suite's timed voice and her
    real voice (where the engine is present): each at
    most 1 of its 12 lives, their total within Binomial(lives, 0.01)'s 0.999 quantile; every trial of theirs the same with the
    draw turned over (its window's ticks on each thing), so their labels are exchangeable and the statistic's distribution the
    same whichever was drawn. Knowers detected: a knower, one right 80% of the time, the name's, 12 of 12; one that learns the
    word at its 12th trial, by its 48th. Disclosed (C69b), in her real voice: the fourteenth verifier's children timed from one
    band of its own ear, as built and turned to the window (a gate on the band's moment), and a learner and the gated name
    child on a band: their looking follows the test word's own sound (their trials differ with the draw turned over), and no
    measure of looking tells them from a knower of that word; at level 2 (its kind's new exemplars, A60b) the gated ones fail,
    and with no feedback the learners learn nothing. Determinism: zoo lives in two processes, threaded and on one thread, the
    same ledger chains and rows. Level 1 ("maps") is read throughout, the zoo's schedule having no new exemplars."""
    assert hasattr(K, "TRIAL_LOOK") and perm_test is not None, "understanding still scored by a first look (A60b)"
    assert hasattr(Ledger, "maps"), "one level of 'understood' only (A60b's two levels)"
    assert hasattr(K, "NOUN_FOILS") and hasattr(K, "KIND_FIRST"), \
        "level 2 as 67741fd built it: a block of 12, its foil not heard as often as its noun, beside familiar things (the " \
        "lead's decisions after 67741fd)"
    import subprocess                                                     # noqa: PLC0415
    from scipy.stats import binom                                         # noqa: PLC0415
    voices = [("timed", _TimedVoice())]
    tmp = None
    if _have_engine():
        tmp = tempfile.mkdtemp()
        voices.append(("real", V.VoiceCache(os.path.join(tmp, "v"), server=V.SynthServer(nice=19))))
    zoo, flips, knowers, band, whole, band2 = {}, {}, {}, {}, {}, {}
    try:
        for vname, voice in voices:
            favs = (("cup", "ball") if vname == "timed" else ("drum", "bear", "cup"))
            kinds = []                                    # (label, maker(seed), favourite, form)
            for fav in favs:
                kinds.append((f"favourite {fav}, looks only at it", lambda s_, f_=fav: _Fav(f_, ("start",), 99.0, 14, "other"),
                              fav, "place"))
            fav = favs[0]
            for anchor in FAV_STOPS + (("start",),):
                kinds.append((f"{fav}, its other look 7.3 ticks after {anchor}",
                              lambda s_, a_=anchor, f_=fav: _Fav(f_, a_, 7.3, 14, "fav"), fav, "place"))
            for vn, fv, anchor, d, on in FAV_KIDS:
                if vn == vname:
                    kinds.append((f"{fv}, the thirteenth verifier's ({anchor}, {d})",
                                  lambda s_, f_=fv, a_=anchor, d_=d, o_=on: _Fav(f_, a_, d_, 14, o_), fv, "place"))
            kinds.append((f"{fav}, learns its delay from her smiles ({FAV_STOPS[1]})",
                          lambda s_, f_=fav: _Fav(f_, FAV_STOPS[1], 7.3, 14, "fav", learn=500 + s_), fav, "place"))
            kinds.append(("the gated name child (her last 10 ms above -36 dB, 12.7 ticks, gate 15)",
                          lambda s_: _Gate(("frame", -36), 12.7, 15), None, "name"))
            kinds.append(("the gated name child (her mouth above -22 dB, 8.1 ticks, gate 12)",
                          lambda s_: _Gate(("mouth", -22), 8.1, 12), None, "name"))
            for label, mk, fv, form in kinds:
                und, n_sc, ps = 0, 0, []
                for s_ in range(12):
                    led = _zoo_life(label, vname, voice, s_, mk, fv, form)
                    key = LX.NAME if form == "name" else fv
                    und += led.maps(key)
                    st = led.words.get(key) or {"seq": []}
                    n_sc += len(st["seq"])
                    if st["seq"]:
                        ps.append(led.test(key)["p"])
                zoo[(vname, label)] = (und, n_sc, min(ps) if ps else None)
                assert und <= 1, (vname, label, und)
                # its trials with the draw turned over: the same window's ticks on each thing, 4 of them each
                same = 0
                pairs = [(None, None)] * 4 if form == "name" else _zoo_pairs(vname, fv, 4)
                for j, (a, b) in enumerate(pairs):
                    xs = []
                    for fl in (False, True):
                        kid = mk(j)
                        if hasattr(kid, "show") and form != "name":
                            kid.show(a, b)
                        xs.append(_timed_trial(form, a, b, kid, 1 + j % 2, j, flip=fl, voice=voice,
                                               scaffold=form != "name", seen_=ITT_TOYS if vname == "real" else TOYS)[0])
                    same += _same_but_named(*xs)
                flips[(vname, label)] = (same, len(pairs))
                assert same == len(pairs), (vname, label, same)
                if vname == "timed" and "learns" not in label:        # a whole life with every draw turned over: the same
                    key = LX.NAME if form == "name" else fv         # looking in every trial, each label turned over, so
                    s0 = (_zoo_life(label, vname, voice, 0, mk, fv, form).words.get(key) or {"seq": []})["seq"]
                    s1 = (_zoo_life(label, vname, voice, 0, mk, fv, form, flip=True).words.get(key) or {"seq": []})["seq"]
                    assert [x[1:] for x in s0] == [x[1:] for x in s1] and [1 - x[0] for x in s0] == [x[0] for x in s1], \
                        (label, s0, s1)                             # its statistic's permutation distribution the same
                    whole[label] = len(s0)
            # knowers: detected
            fk = favs[0]
            for label, mk, n_pr, form, least in (("a knower", lambda s_: _Knows(seed=s_), 24, "place", 11),
                                                 ("a knower right 80% of the time", lambda s_: _Knows(p=0.8, seed=s_), 48,
                                                  "place", 9),
                                                 ("one that learns the word at its 12th trial",
                                                  lambda s_: _Knows(from_=12, seed=s_), 48, "place", 9),
                                                 ("a knower of its name", lambda s_: _KnowsName(), 24, "name", 11)):
                und = sum(_zoo_life(label, vname, voice, s_, mk, fk, form, n_pr).maps(LX.NAME if form == "name" else fk)
                          for s_ in range(12))
                knowers[(vname, label)] = (und, 12, n_pr)
                assert und >= least, (vname, label, und)
        total = sum(u for u, _n, _p in zoo.values())
        n_lives = 12 * len(zoo)
        assert total <= binom.ppf(0.999, n_lives, 0.01), (total, n_lives)
        # disclosed (C69b): children timed from one band of its own ear, her real voice
        if len(voices) > 1:
            vname, voice = voices[1]
            con = _conduct(seed=1, voice=voice, ledger=_TapLedger(), scaffold=False)
            ids = [o.id for o in ITT_TOYS]
            perc = {}
            for w in ids:
                ln = TP.Line(f"where is the {w}? see?", "trial_where", "question", w, w, (), "fast", "where is the {o}? see?",
                             ST.shape("where", w)[0])
                perc[w] = _perceived(con._clip(ln, heard=False))
            for fav, anchor in BAND_KIDS:
                m = {w: _moment(perc[w], anchor) for w in ids}
                others = [w for w in ids if w != fav]
                built = None                                  # as the verifier built it: the order of its two looks follows
                for d in np.arange(0.0, 20.0, 0.05):          # the word (a look at tick 14, one timed from the band's moment)
                    tf, to = math.floor(m[fav] + d + 1e-9), [math.floor(m[o] + d + 1e-9) for o in others]
                    if tf >= 14 > max(to):
                        built = (float(d), "other")
                    elif tf < 14 <= min(to):
                        built = (float(d), "fav")
                    if built:
                        break
                gate = None                                   # turned to the window: a gate between its moment and all
                for d in np.arange(0.0, 1.0, 0.01):           # the others', its moment the last of them or the first
                    for g in range(6, 20):
                        tf, to = math.floor(m[fav] + d + 1e-9), [math.floor(m[o] + d + 1e-9) for o in others]
                        if tf >= g > max(to):
                            gate = (float(d), g, True)
                        elif tf < g <= min(to):
                            gate = (float(d), g, False)
                        if gate:
                            break
                    if gate:
                        break
                rows_b = {}
                for label, mk in ((f"as built ({built})", None if built is None else
                                   (lambda s_, d_=built[0], o_=built[1]: _Fav(fav, anchor, d_, 14, o_))),
                                  (f"a gate ({gate})", None if gate is None else
                                   (lambda s_, g_=gate: _GateLook(fav, anchor, g_[0], g_[1], late=g_[2]))),
                                  ("would learn its delay from her smiles (none since A60b)",
                                   lambda s_: _Fav(fav, anchor, 7.0, 14, "other", learn=500 + s_))):
                    if mk is None:
                        continue
                    und = sum(_zoo_life(label, vname, voice, s_, mk, fav).maps(fav) for s_ in range(12))
                    if label.startswith("a gate"):                # its word's second level: its kind's new exemplars
                        lv = [_levels_life(mk(s_), _level_probes("real", fav, s_), voice, s_, KIND_REAL, fav=fav,
                                           pool=_level_pool("real", fav))[0] for s_ in range(12)]
                        band2[(fav, anchor)] = (sum(x.maps(fav) for x in lv), sum(x.understood(fav) for x in lv))
                        assert band2[(fav, anchor)][1] <= 1, (fav, anchor, band2[(fav, anchor)])
                    kid0, kid1 = mk(0), mk(0)
                    for k_ in (kid0, kid1):
                        k_.show(fav, others[0])
                    xs = [_timed_trial("place", fav, others[0], k_, 1, 0, flip=fl, voice=voice, seen_=ITT_TOYS)[0]
                          for k_, fl in ((kid0, False), (kid1, True))]
                    rows_b[label] = (und, _same_but_named(*xs))
                band[(fav, anchor)] = (round(m[fav], 3), round(min(m[o] for o in others), 3),
                                       round(max(m[o] for o in others), 3), rows_b)
            mn = {}
            for w in (LX.NAME,) + tuple(K.NAME_FOILS):
                ln = TP.Line(TP.FRAMES["trial_name"][0][0].replace("{n}", w), "trial_name", "calling", w, w, (), "fast",
                             TP.FRAMES["trial_name"][0][0], ST.shape("name", w)[0])
                mn[w] = _moment(_perceived(con._clip(ln, heard=False)), BAND_NAME)
            g_n = int(math.floor(mn[LX.NAME])) + 1             # its turn before the gate after "pip", never after a foil
            assert all(math.floor(mn[f_]) >= g_n for f_ in K.NAME_FOILS), mn
            und = sum(_zoo_life("band name", vname, voice, s_, lambda x: _Gate(BAND_NAME, 0.0, g_n), None, "name")
                      .understood(LX.NAME) for s_ in range(12))
            xs = [_timed_trial("name", None, None, _Gate(BAND_NAME, 0.0, g_n), 1, 0, flip=fl, voice=voice, scaffold=False)[0]
                  for fl in (False, True)]
            band[("pip", BAND_NAME)] = (round(mn[LX.NAME], 3), round(min(mn[f_] for f_ in K.NAME_FOILS), 3),
                                        round(max(mn[f_] for f_ in K.NAME_FOILS), 3),
                                        {f"a turn gated at tick {g_n}": (und, _same_but_named(*xs))})
        # determinism: zoo lives in two processes, threaded (hash seed 1) and on one thread (hash seed 4242)
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        code = ("import json, sys; sys.path.insert(0, %r); import body.tests.test_sim_lang as M; "
                "print(json.dumps(M._zoo_digests()))") % root
        runs = []
        for seed_, one in (("1", False), ("4242", True)):
            env = dict(os.environ, PYTHONHASHSEED=seed_)
            if one:
                env.update(VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=1200)
            assert r.returncode == 0, r.stderr[-1500:]
            runs.append(json.loads(r.stdout.strip().splitlines()[-1]))
        assert runs[0] == runs[1] == _zoo_digests(), "a zoo life differs across processes or threads"
    finally:
        for _v, voice in voices[1:]:
            voice.close()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"72 the lead's decision A60b, its acceptance: the proportion of looking over the window (the test word's onset + 2 "
          f"to + 23), void under 4 ticks on either thing, 'understood' by a permutation test over the draw labels at 0.005, "
          f"0.0025, ... (0.01 a life). The zoo, 12 lives each, 'understood' / its registered trials / its least p: " + "; ".join(
              f"{v} voice, {k}: {u} of 12 ({n} trials, least p {p:.3f})" if p is not None else f"{v} voice, {k}: {u} of 12"
              for (v, k), (u, n, p) in zoo.items()) +
          f"; in all {total} of {n_lives} lives (at 0.01 a life, Binomial's 0.999 quantile "
          f"{binom.ppf(0.999, n_lives, 0.01):.0f}); "
          f"their trials with the draw turned over: " + "; ".join(f"{v} {k}: {a} of {b} the same" for (v, k), (a, b) in
                                                                 flips.items()) +
          f"; a whole life with every draw turned over, each non-learner of the timed voice: its looking the same in every one "
          f"of its registered trials, each label turned over (so its statistic's permutation distribution the same: " +
          ", ".join(f"{n} trials" for n in whole.values()) + "); a learner's lives differ, her smiles following the draw" +
          f"; knowers: " + "; ".join(f"{v} voice, {k}: {u} of {n} ({m} probes)" for (v, k), (u, n, m) in knowers.items()) +
          ("; disclosed (C69b), children keyed on one band of its own left ear, her mouth 0.7 m ahead, 12 lives each: " +
           "; ".join(f"the {w} child on {a} (its moment {mw} ticks, the others' {lo}-{hi}): " + ", ".join(
               f"{lb}: '{w}' mapped in {u} of 12" + ("" if same is None else ", its trial " +
                                                          ("the same" if same else "not the same") +
                                                          " with the draw turned over")
               for lb, (u, same) in rows.items()) for (w, a), (mw, lo, hi, rows) in band.items())
           if band else "; the engine absent here: the band children not run") +
          ("; the gated ones at level 2, 12 lives of 24 place probes and a block of 24 of their kind's new exemplars (beside "
           "new exemplars of two other kinds as new as them, their foil heard as often): " + "; ".join(
              f"the {w} child on {a}: maps {m} of 12, understood {u} of 12" for (w, a), (m, u) in band2.items())
           if band2 else "") +
          "; zoo lives in two processes, threaded (hash seed 1) and on one thread (hash seed 4242): the same ledger chains "
          "and rows")


KIND_TIMED = TOYS + seen(*[(f"{w}_{k}", w, c, "mat", True) for w, c in (("cup", "green"), ("ball", "red"), ("block", "blue"))
                           for k in range(2, 12)])
KIND_REAL = ITT_TOYS + seen(*[(f"{w}_{k}", w, "", "mat", True) for w in ("drum", "bear", "cup", "car") for k in range(2, 12)])


def _level_probes(vname, fav, seed, n1=24):
    """a life's place probes for level 1 (A60b): n1 place probes of the trained thing against each other toy in turn ->
    [(form, a, b, new)]."""
    return [("place", a, b, {a: f"{a}@{seed}.{k}", b: f"{b}@{seed}.{k}"}) for k, (a, b) in enumerate(_zoo_pairs(vname, fav, n1))]


def _level_pool(vname, fav, n=10, kinds=None):
    """the room's new exemplars for level 2 (A53: at least 8 a tested noun; here 10, so a void's spent presentation can be
    made up): n of the trained thing's kind and n of each of two other kinds, whose pools serve as its others -> {id: item}."""
    kinds = kinds or (("ball", "block") if vname == "timed" else tuple(x for x in ("bear", "cup", "car") if x != fav)[:2])
    return {f"{w}_{k}": f"{w}_{k}" for w in (fav,) + tuple(kinds) for k in range(2, 2 + n)}


def _levels_life(kid, probes, voice, seed, toys, fav=None, pool=None, n2=24, flip=False, label=None):
    """one life of the two levels through her conduct and ledger (its stub motion, still; stage 1 and 2 by seed; `fav`'s level 2
    registered, its foil carried in her chatter from birth): the place probes in turn (level 1; the words channel labelling her
    lines), then the scaffold silenced (A29, as a life's order has it: a word's token and a foil's letters are never one
    timeline on it) and her exemplar probes of `fav` (level 2), each drawing its three things from `pool` (the room's new
    exemplars), asked again while its foil is not yet heard as often as it (her chatter catches up) until its block holds n2
    registered trials or the pool has no set left; label: an exemplar she names first (her follow-in label of it, asked for);
    the child shown each trial's things (a place probe's in its order; an exemplar trial's as they are set down, left to
    right) -> (the ledger, her judgments, her lines after each trial, dict(chatter: [(tick, line, the child's target, its
    holds)], dropped: [why], her talk's density (4.6): ticks, words (hers said, their sound ended), chatter_words, sounding
    (ticks her voice sounded), play (ticks with no formal trial under way), play_sounding)). flip: every draw of her stream
    turned over (_Flip)."""
    con = _conduct(seed=seed, stage=1 + seed % 2, imperfect=False, voice=voice, ledger=_TapLedger(),
                   transcriber=Transcriber(None), scaffold=True, level2=(fav,) if fav and pool else ())
    if flip:
        con.trial_rng = _Flip(con.trial_rng)
    _no_sets(con, tuple(o.id for o in toys))
    k, t, last, busy, judg, after, shown_at = 0, 0, -100, False, [], [], None
    info, n_ref, block, labelled = dict(chatter=[], dropped=[], sounding=0, play=0, play_sounding=0), 0, False, label is None
    cap = 400 * (len(probes) + (3 * n2 if pool else 0))
    while t < cap:
        idle = con.trial is None and not con.probes and t > last + 30
        if idle and k < len(probes):
            form, a, b, new = probes[k]
            con.probe(form, a, b, new=new)
            kid.show(a, b)
            k += 1
        elif idle and pool and not block:
            block = True
            con.set_scaffold(False)                       # the scaffold silenced before level 2's block (A29)
        elif idle and block and not labelled:
            con.request("label", o=label)                 # she names that exemplar first: no new exemplar then
            labelled, last = True, t + 30
        elif idle and block:
            st = con.ledger.words.get(fav) or {"seq2": []}
            why = (con.fast.refused[-1][2] if con.fast.refused and con.fast.refused[-1][0] >= last else "")
            if len(st["seq2"]) >= n2 or "no new exemplar" in why:
                break
            con.probe("exemplar", noun=fav, new=pool)
            last = t
        elif idle and not pool:
            break
        got = kid.at(t)
        tg, holds = got[0], ()
        s = con.tick(t, P(t, child_target=tg, child_reaches=got[3], events=got[4], seen=toys), tract=got[1], token=got[2])
        info["sounding"] += int(con.fast.speaking(t))    # her voice sounding this tick (her talk's density, 4.6), and on a
        if con.trial is None:                               # play tick (no formal trial under way)
            info["play"] += 1
            info["play_sounding"] += int(con.fast.speaking(t))
        if con.trial is not None and con.trial.get("sides") and shown_at != con.trial["since"]:
            shown_at = con.trial["since"]
            kid.show(*con.trial["sides"])
        kid.heard(t, s)
        judg += list(s.judgments or ())
        if s.line is not None and s.line.intent == "foil_chatter":
            info["chatter"].append((t, s.line.text, tg, holds, tuple(s.line.refs), tuple(a_.kind for a_ in (s.acts or ()))))
        if s.line is not None and last >= 0 and con.trial is None and s.line.intent not in C.TRIAL_INTENTS:
            after.append((t, s.line.text))
        now = con.trial is not None or bool(con.probes)
        if busy and not now:
            last = t
            kid.ended()
        busy = now
        t += 1
    info["dropped"] = [w for _t, x, w in con.fast.refused if x == "trial:exemplar"]
    info["ticks"] = t
    info["words"] = sum(int(st_.get("said", 0)) for st_ in con.ledger.words.values())
    info["chatter_words"] = sum(len(TP.words(x)) for _t, x, *_r in info["chatter"])
    return con.ledger, judg, after, info


LEVEL_KIDS = (   # (label, the child (seed, fav, trained things), maps within, understood within): 12 lives each, 24 place probes
    # and exemplar probes to a block of 24 registered trials (A60b 9, 12-14)
    ("a knower of the kind, its look at a foil on one of the three", lambda s_, f_, t_: _Knows(seed=s_), (11, 12), (8, 12)),
    ("a knower of the kind, its looks at a foil spread over the three", lambda s_, f_, t_: _Knows(seed=s_, spread=True),
     (11, 12), (11, 12)),
    ("a knower of its trained things alone", lambda s_, f_, t_: _Knows(seed=s_, trained=t_), (11, 12), (0, 1)),
    ("its favourite, the trained thing", lambda s_, f_, t_: _Fav(f_, ("start",), 99.0, 14, "other"), (0, 1), (0, 1)),
    ("its favourite, the newest thing shown", lambda s_, f_, t_: _Newest(seed=s_), (0, 1), (0, 1)),
    ("it knows only the other words, excluding at any word it does not know", lambda s_, f_, t_: _Excludes(f_, seed=s_),
     (11, 12), (0, 1)),
    ("it knows only the other words, excluding only at a word it has heard 6 times",
     lambda s_, f_, t_: _Excludes(f_, seed=s_, often=6), (10, 12), (0, 1)),
    ("a knower of its trained things, on the newest thing shown when its thing is not",
     lambda s_, f_, t_: _Newest(seed=s_, knows=t_), (11, 12), (0, 1)),
    ("a knower of the kind, on the newest thing shown at a novel name", lambda s_, f_, t_: _Newest(seed=s_, knows="kind"),
     (11, 12), (6, 12)),
    ("disclosed: it knows only the other words, excluding only at a word heard 6 times in her trial sentences",
     lambda s_, f_, t_: _Excludes(f_, seed=s_, often=6, carrier=True), (10, 12), (0, 2)))
LEVEL_BAND = ("keyed by its ear's 167 Hz band to the drum", lambda s_, f_, t_: _GateLook(
    "drum", ("band", "L", 3, 62.5, "stop"), 0.0, 6), (11, 12), (0, 1))


def test_trial_levels_no_feedback():
    """the lead's further decisions on A60b (P3's fifteenth round). NO FEEDBACK IN TEST TRIALS (after 535e32b): she gives nothing
    for a trial, no smile and no confirm, her face neutral through the window and after it, as labs give none (Fernald et al.
    2008; Golinkoff et al. 1987). TWO LEVELS (after 535e32b): level 1, "maps", the word's record over place trials of its
    trained thing; level 2, "understood", for an object noun, the same test passed also in its new-exemplar block, registered
    apart. LEVEL 2's CONTROL A NEVER-TOLD FOIL (after 200e57a, the skeptic's design). And after 67741fd: its BLOCK 24
    REGISTERED TRIALS (its first test at its 24th, 0.005; A53: at least 8 exemplars a tested noun, a void's presentation spent,
    so the conduct draws a further one), its foil MATCHED IN EXPOSURE (her chatter carries each registered noun's foil at its
    noun's running rate, never a label, no act, no object named, only while the child attends nothing and holds nothing; a
    probe runs only while the foil has been heard about as often as its noun, each trial logging both), and MATCHED NEWNESS
    (the conduct draws each trial's three things from the room's new exemplars: its new exemplar and two of two other kinds
    presented exactly as often, fresh, never named, each noun's pool serving both roles). Tested through her conduct, 12 lives
    each of 24 place probes and exemplar probes to a block of 24, in the timed voice and her real voice: a knower of the kind
    is understood, whether its look at a foil goes to one of the three things or spreads over them, and one that looks at the
    newest thing at a novel name; a knower of its trained things alone, its favourites (the trained thing, the newest thing),
    the child keyed by one band of its ear to the drum, a child that knows only the other words (excluding at any word it does
    not know), the frequency-gated excluder (only at a word heard 6 times: now it meets the foil as familiar) and the knower of
    its trained things that looks at the newest thing when its thing is absent (now nothing shown is newer than another) are
    not; disclosed, an excluder gated on the trial sentence's own familiarity (its noun heard there in level 1, its foil only
    in the block). The foils: none of her words, none near them, the name, its foils or one another, none sayable, each noun's
    measured in her voice; her chatter's lines where the child attends nothing; the named exemplar never drawn; a probe with no
    matched set, a foil of hers, a foil not yet heard as often as its noun, a noun not registered, or under the scaffold, dropped
    before it is said; a non-knower's exemplar trials alike with the draw turned over; no judgment or line of hers follows a
    trial. After 176da4e (the lead's two calls): level 2 only for a REGISTERED SHORT LIST of at most 6 nouns (a seventh refused
    at her construction and in a save she loads, a noun off the list refused its probe), the room at A53's numbers (24 new
    exemplars of each registered noun, 4 of each other tested noun) carrying every registered block to 24 with its things
    matched in newness, its blocks one after another or a trial of each in turn, the draw keeping each unfinished block's own
    exemplars back; her foil lines in her IDLE SLOT (at most one line a 40 ticks, shared with P4's idle lines: they replace her
    filler rather than add to it), her talk over these lives within 4.6's upper bound and at least 40% of its play ticks (no
    formal trial under way), and of all its ticks, free of her voice."""
    import inspect                                                        # noqa: PLC0415
    assert hasattr(K, "NOUN_FOILS") and hasattr(K, "KIND_FIRST") and hasattr(K, "LEVEL2_MAX") and \
        hasattr(C.Conduct, "idle_free") and "noun" in inspect.signature(C.Conduct.probe).parameters, \
        "level 2 with no registered short list, its foil lines outside her idle slot (the lead's decisions after 67741fd " \
        "and 176da4e)"
    assert hasattr(Ledger, "maps") and hasattr(K, "KIND_DISTRACTORS"), "one level of 'understood' only (A60b's two levels)"
    # the foils: words she never tells, none near her words, the name, its foils or one another, none sayable; each tested
    # noun's measured in her voice, of its written syllables; each noun its own, first and last sounds unlike its noun's
    words = set(LX.BIRTH_WORDS) | set(TP.GROWTH_WORDS)
    foils = sorted(K.NOUN_FOILS.values())
    near = words | {LX.NAME} | set(K.NAME_FOILS)
    # 129 words: the 128 of the design and the bucket (A126)
    assert len(words) == 129 and len(set(foils)) == len(foils) and \
        all(min(_edit(f_, x) for x in near | (set(foils) - {f_})) >= 2 for f_ in foils), \
        [(f_, min(near, key=lambda x, g=f_: _edit(g, x))) for f_ in foils]
    for f_ in foils:
        assert not TP.check(TP.foil_line(f_), VOCAB)[0] and not TP.check(TP.foil_line(f_), VOCAB, source="claude")[0], f_
        assert not TP.check(TP.foil_chatter(f_), VOCAB)[0], f_
    snd = lambda ch: "k" if ch in "ckq" else ch                            # noqa: E731
    tab = ST.STIMULI["where"]
    assert set(K.NOUN_FOILS) == set(tab["words"]), (sorted(K.NOUN_FOILS), sorted(tab["words"]))
    for w, f_ in K.NOUN_FOILS.items():
        assert tab.get("foils", {}).get(f_, {}).get("syllables") == K.NOUN_SYLLABLES[w] and snd(f_[0]) != snd(w[0]) and \
            snd(f_[-1]) != snd(w[-1]), (w, f_)
    voices = [("timed", _TimedVoice())]
    tmp = None
    if _have_engine():
        tmp = tempfile.mkdtemp()
        voices.append(("real", V.VoiceCache(os.path.join(tmp, "v"), server=V.SynthServer(nice=19))))
    rows, flips, fb, drops = {}, {}, dict(judgments=0, confirms=0, lives=0), {}
    chat = dict(lines=0, bad=0, first=[])
    dens = dict(words=0, ticks=0, sounding=0, chatter=0, play=0, play_sounding=0)
    try:
        for vname, voice in voices:
            fav, toys = ("cup", KIND_TIMED) if vname == "timed" else ("drum", KIND_REAL)
            trained = [o.id for o in toys if not o.id.split("_")[-1].isdigit()]
            pool = _level_pool(vname, fav)
            kinds = list(LEVEL_KIDS) + ([LEVEL_BAND] if vname == "real" else [])
            for label, mk, b1, b2 in kinds:
                m1, u2, n2, v2 = 0, 0, 0, 0
                for s_ in range(12):
                    led, judg, after, info = _levels_life(mk(s_, fav, trained), _level_probes(vname, fav, s_), voice, s_,
                                                          toys, fav=fav, pool=pool)
                    m1 += led.maps(fav)
                    u2 += led.understood(fav)
                    n2 += len((led.words.get(fav) or {"seq2": []})["seq2"])
                    ex = [r for r in _trials(led) if r["form"] == "exemplar"]
                    v2 += sum(r["result"] == "void" for r in ex)
                    fb["lives"] += 1
                    fb["judgments"] += sum(j[1] == "met_trial" for j in judg)
                    fb["confirms"] += sum(x.startswith("yes") for _t, x in after)
                    chat["lines"] += len(info["chatter"])
                    for k_ in ("words", "ticks", "sounding", "play", "play_sounding"):
                        dens[k_] += info[k_]
                    dens["chatter"] += info["chatter_words"]
                    chat["bad"] += sum(bool(tg is not None or hd or rf or ac) for _t, _x, tg, hd, rf, ac in info["chatter"])
                    if ex:
                        e0 = ex[0]["exposure"]
                        chat["first"].append((e0[fav], e0[K.NOUN_FOILS[fav]]))
                        assert all(set(r["exposure"]) == {fav, K.NOUN_FOILS[fav]} for r in ex), ex[0]
                        for r in ex:                              # matched newness: its three things presented as often
                            items_ = [r["target"]["id"], r["distractor"]["id"], r["others"][0]["id"]]
                            assert len({x.split("_")[0] for x in items_}) == 3 and all(x in pool for x in items_), items_
                key = (vname, label)
                rows[key] = (m1, u2, n2, v2)
                assert b1[0] <= m1 <= b1[1], (key, m1)
                assert b2[0] <= u2 <= b2[1], (key, u2)
            # a non-knower's exemplar trials, the draw turned over (its word said for its foil, its foil for its word): the
            # same looking at each thing
            same, n_ex = 0, 0
            for j in range(6):
                xs = []
                for fl in (False, True):
                    kid = (_Newest(seed=j) if j % 2 else _Knows(seed=j, trained=trained))
                    xs.append(_timed_trial("exemplar", None, None, kid, 1 + j % 2, j, flip=fl, voice=voice, seen_=toys,
                                           scaffold=False, noun=fav, pool=pool)[0])
                same += xs[0]["result"] == "scored" and _same_but_named(*xs)
                n_ex += 1
            flips[vname] = (same, n_ex)
            assert same == n_ex, (vname, same, n_ex)
        # the room at A53's numbers (the lead's decision after 176da4e): a registered short list of at most 6 nouns, 24 new
        # exemplars of each and 4 of each other tested noun; the conduct's draw, each unfinished block's own kept back, runs
        # every registered block to 24 with its things matched in newness (its blocks one after another, or a trial of each
        # in turn; no voids), a seventh noun refused, at her construction and in a save she loads, a noun off the list
        # refused its probe (below)
        regs, rest = ("ball", "bear", "block", "car", "cup", "drum"), ("ring", "bottle", "rattle", "tower")
        room = seen(*[(f"{w}_{k}", w, "", "mat", True) for w in regs for k in range(24)] +
                    [(f"{w}_{k}", w, "", "mat", True) for w in rest for k in range(4)])
        blocks, spare = {}, {}
        for order in ("in turn", "round robin"):
            con_r = _conduct(seed=1, stage=1, imperfect=False, ledger=Ledger(), level2=regs)
            p_r, new_r = P(0, seen=room), {o.id: o.id for o in room}

            def _draw(w, c_=con_r, p_=p_r, n_=new_r):        # one registered trial of w's block, its three drawn by the conduct
                got_r, _why = c_._exemplar_set(p_, w, n_, {})
                if got_r is None:
                    return False
                assert len({x.split("_")[0] for x in got_r}) == 3 and len({c_.ledger.presented(x) for x in got_r}) == 1, got_r
                for x in got_r:
                    c_.ledger.items[x] = c_.ledger.presented(x) + 1
                c_.ledger._w(w)["seq2"].append([1, 8, 0, w])
                return True
            if order == "in turn":
                for w in regs:
                    while len(con_r.ledger._w(w)["seq2"]) < K.KIND_FIRST and _draw(w):
                        pass
            else:
                live = list(regs)
                while live:
                    live = [w for w in live if len(con_r.ledger._w(w)["seq2"]) < K.KIND_FIRST and _draw(w)]
            blocks[order] = {w: len(con_r.ledger._w(w)["seq2"]) for w in regs}
            spare[order] = sum(1 for o in room if con_r.ledger.presented(o.id) == 0)
        try:
            _conduct(seed=1, level2=regs + ("ring",))
            seventh = False
        except ValueError:
            seventh = True
        st_r = _conduct(seed=1, level2=regs).state()
        kept = _conduct(seed=1)
        kept.load_state(st_r)
        try:
            _conduct(seed=1).load_state(dict(st_r, level2=list(regs) + ["ring"]))
            seventh_saved = False
        except ValueError:
            seventh_saved = True
        assert all(n == K.KIND_FIRST for b_ in blocks.values() for n in b_.values()) and seventh and seventh_saved and \
            kept.level2 == regs, (blocks, seventh, seventh_saved, kept.level2)
        # her talk's density with her chatter on, over these lives (the lead's decision after 176da4e): 4.6's bounds, at most
        # 2,500 words a life day, at least 40% of play ticks free of her voice (a play tick: no formal trial under way), and
        # of all its ticks
        assert dens["words"] / dens["ticks"] * 24000 <= 2500 and 1 - dens["sounding"] / dens["ticks"] >= 0.4 and \
            1 - dens["play_sounding"] / dens["play"] >= 0.4, dens
        # an exemplar she named before the block is never drawn (as its noun's or as another's)
        led_n, _j, _a, info_n = _levels_life(_Knows(seed=3), _level_probes("timed", "cup", 3), voices[0][1], 3, KIND_TIMED,
                                             fav="cup", pool=_level_pool("timed", "cup"), label="cup_2")
        drawn = {x["id"] for r in _trials(led_n) if r["form"] == "exemplar"
                 for x in [r["target"], r["distractor"]] + r["others"]}
        assert "cup_2" not in drawn and len(led_n.standing("cup")["seq2"]) == 24, sorted(drawn)
        # dropped, never said: no matched set in the room; its foil a word of hers; its foil not yet heard as often as its
        # noun; its noun not registered; the scaffold labelling her lines
        few = {k: v for k, v in _level_pool("timed", "cup").items() if not k.startswith("block")}
        for why, pool_, scaf, vocab, lv2, prep in (("set", few, False, VOCAB, None, None),
                                                   ("hers", None, False, VOCAB + ("tuv",), None, None),
                                                   ("exposure", None, False, VOCAB, None, 30),
                                                   ("registered", None, False, VOCAB, (), None),
                                                   ("channel", None, True, VOCAB, None, None)):
            keep = []
            if prep:                                       # its noun heard 30 times, its foil never
                con_ = _conduct(seed=1, stage=1, imperfect=False, voice=_TimedVoice(), ledger=_TapLedger(),
                                transcriber=Transcriber(None), scaffold=False, level2=("cup",))
                con_.ledger._w("cup")["said"] = prep
                con_.probe("exemplar", noun="cup", new=_level_pool("timed", "cup"))
                for t_ in range(3):
                    con_.tick(t_, P(t_, seen=KIND_TIMED))
                r_, said = dict(result="dropped" if con_.trial is None else "run", why=con_.fast.refused[-1][2]), None
            else:
                r_, _b, said = _timed_trial("exemplar", None, None, _Knows(seed=1), 1, 1, seen_=KIND_TIMED, scaffold=scaf,
                                            vocab=vocab, noun="cup", pool=pool_ or _level_pool("timed", "cup"), level2=lv2,
                                            keep=keep)
            drops[why] = r_.get("why")
            assert r_["result"] == "dropped" and said is None, (why, r_)
        assert "no new exemplar" in drops["set"] and "tuv" in drops["hers"] and "not yet matched" in drops["exposure"] and \
            "not registered" in drops["registered"] and "channel" in drops["channel"], drops
        assert fb["judgments"] == 0 and fb["confirms"] == 0 and chat["bad"] == 0 and chat["lines"] > 0, (fb, chat)
        tol, slack = K.FOIL_EXPOSURE
        assert all(f2 >= n1 - max(slack, tol * n1) for n1, f2 in chat["first"]), chat["first"]
    finally:
        for _v, voice in voices[1:]:
            voice.close()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    fe = chat["first"]
    print(f"73 no feedback in a trial, two levels, level 2's control a never-told foil, its block 24 trials, its foil matched in "
          f"exposure and its things in newness (A60b): over {fb['lives']} lives of 24 place probes and a block of 24 "
          f"registered exemplar trials, no smile and no confirm after any trial ({fb['judgments']} judgments, "
          f"{fb['confirms']} confirms); the foils {', '.join(f'{w} {f_}' for w, f_ in sorted(K.NOUN_FOILS.items()))}: none "
          f"within edit distance 1 of her 128 words, the name, its foils or one another, none sayable, each measured for its "
          f"noun; her chatter's foil lines {chat['lines']} (her idle slot's, one a 40 ticks at most), every one while the child "
          f"attended and held nothing, no act, no object named; her talk over these lives "
          f"{dens['words'] / dens['ticks'] * 24000:.0f} words a life day ({dens['chatter'] / dens['words']:.0%} of them her "
          f"chatter's), {1 - dens['play_sounding'] / dens['play']:.0%} of the play ticks free of her voice and "
          f"{1 - dens['sounding'] / dens['ticks']:.0%} of all ({dens['play'] / dens['ticks']:.0%} of them play ticks; 4.6: at "
          f"most 2,500 words, at least 40% free); the room at A53's numbers (6 registered nouns at 24 new exemplars, 4 others "
          f"at 4): every registered block drawn to {min(n for b_ in blocks.values() for n in b_.values())} trials (" +
          "; ".join(f"{o}, {spare[o]} new exemplars left unpresented" for o in blocks) +
          "), a seventh noun refused at her construction and in a save; at each block's first trial its noun heard "
          f"{min(x for x, _y in fe)}-{max(x for x, _y in fe)} "
          f"times, its foil {min(y for _x, y in fe)}-{max(y for _x, y in fe)}; 'maps' (level 1) and 'understood' (level 2: "
          f"its new exemplar's share after its word against after its foil, beside new exemplars of two other kinds as new "
          f"as it, from its 24th trial) in 12 lives each: " + "; ".join(
              f"{v} voice, {k}: maps {m1} of 12, understood {u} of 12 ({n2} registered new-exemplar trials, {v2} void)"
              for (v, k), (m1, u, n2, v2) in rows.items()) +
          "; an exemplar she named before the block never drawn; dropped, never said: " +
          "; ".join(f"{k}: {v}" for k, v in drops.items()) + "; non-knowers' exemplar trials turned over at the draw: " +
          ", ".join(f"{v} voice {a} of {b} the same" for v, (a, b) in flips.items()))


def _conduct(**kw):
    """a Conduct with the keywords this tree's takes (scaffold since P3's twelfth round: left out on the code before it, so a
    test fails there at its own assertion)."""
    import inspect                                                        # noqa: PLC0415
    ok = inspect.signature(C.Conduct.__init__).parameters
    return C.Conduct(**{k: v for k, v in kw.items() if k in ok})


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



def test_vocal_habituation():
    """lang 60 (C85, 2026-09-26): her smile at a word said right again habituates as at a motor act: the n-th right name or echo of
    the same word is worth 2 e^(-n/HABIT_TAU), none under HABIT_FLOOR (then the turn is answered as an echo: heard, no smile); a new
    word starts at 0 and pays 2; a met ask pays in full; the count survives a save"""
    import math as _m
    con = C.Conduct(seed=3, transcriber=Transcriber(None), stage=1, imperfect=False)
    con.fast.last_set["duck"] = 0; con.fast.last_named["duck"] = 0
    duck, ball = LX.WORD_ID["duck"], LX.WORD_ID["ball"]
    N = 45
    toks = {40 + 20 * i: duck for i in range(N)}
    stream = [P(t, child_target="duck" if t >= 30 else None) for t in range(40 + 20 * N + 10)]
    con.fast.last_set["ball"] = 0; con.fast.last_named["ball"] = 0
    said, judg = run(con, stream, tokens=toks)
    worths = [j[1] for j in judg if j[2] in ("right_name", "echo")]
    want = [K.WORTH_RIGHT_NAME * _m.exp(-n / K.HABIT_TAU) for n in range(N)]
    want = [w for w in want if w >= K.HABIT_FLOOR]
    assert len(worths) == len(want) and all(abs(a - b) < 1e-9 for a, b in zip(worths, want)), (len(worths), len(want), worths[:5], worths[-3:])
    assert len(want) < N and con.vocal_book["duck"] == len(want), (len(want), con.vocal_book)
    late = [x for x in said if x[0] >= 40 + 20 * (N - 1)]
    assert late and late[0][1].intent != "confirm" and late[0][1].register != "approval", late[:2]   # answered (an echo, a reply, a
    assert not [j for j in judg if j[0] >= 40 + 20 * len(want)], judg[-2:]                       # label), never a "yes!": no smile
    st = con.state(); con2 = C.Conduct(seed=3, transcriber=Transcriber(None), stage=1, imperfect=False); con2.load_state(st)
    assert con2.vocal_book == con.vocal_book
    con3 = C.Conduct(seed=3, transcriber=Transcriber(None), stage=1, imperfect=False)
    con3.fast.last_set["ball"] = 0; con3.fast.last_named["ball"] = 0
    _s, judg3 = run(con3, [P(t, child_target="ball" if t >= 30 else None) for t in range(90)], tokens={40: ball})
    assert [j[1:] for j in judg3] == [(2, "right_name", "ball")], judg3
    print(f"60 the n-th right name or echo of the same word is worth 2 e^(-n/{K.HABIT_TAU:.0f}) (C85): {len(want)} smiles for 'duck' of",
          f"{N} right names, the last {worths[-1]:.3f}, then answered as an echo with no smile; a new word pays 2; the count saved")


def test_one_comfort_at_a_time():
    """lang 63 (C111, A136's dawn): a pain event every second tick for 90 ticks: her comfort's words are said again and again, but its
    hands (a lean-in, a hand on its trunk) are asked once and not again while the last ones are still on the way: at no tick are two
    lean-ins or two attends open, and the skipped hands are counted. Life day 25 had two of each open at every tick and its lessons
    cancelled before they began (74 in a morning)"""
    con = C.Conduct(seed=3, transcriber=Transcriber(None), stage=1, imperfect=False)
    _no_sets(con)
    comforts = 0; worst = {"lean_in": 0, "attend": 0}
    for t in range(0, 90):
        s = con.tick(t, P(t, events=((("pain", None),) if t % 2 == 0 else ())))
        if s.line is not None and s.line.intent == "comfort":
            comforts += 1
        for k in worst:
            n = sum(1 for a in con.acts_open if a[1] == k and a[5] not in C.ENDED)
            worst[k] = max(worst[k], n)
    assert comforts >= 2, comforts                                                  # the words keep coming (C120: once the cry has lasted, again at its length)
    assert worst["lean_in"] <= 1 and worst["attend"] <= 1, worst                    # the hands one at a time
    con.comfort_skipped = int(getattr(con, "comfort_skipped", 0))                   # (C120 spaces the comforts, so the hands are rarely asked while on the way)
    print(f"lang 63: {comforts} comfort lines in 90 ticks of pain every second tick; at most {worst['lean_in']} lean-in and {worst['attend']} attend open",
          f"at any tick; {con.comfort_skipped} hands skipped while the last were on the way (one comfort at a time)")


def test_the_hide_told():
    """lang 64 (C115): the hide told at its drop, before the reply she owes, and a lesson's set kept across a reply. Life day 26: the
    child streamed her own words back and every one was answered (896 lines by midday, four in five replies), the hide's bucket lines
    composed before the drop were untrue and refused, and every reply emptied the queue, so the one hide done was never told. Here a
    conduct owes a reply (reply_due now) and has a hide just done on the duck, the duck seen in the bucket: her next line is the
    drop's ("the duck is in the bucket."), its second line queued; a queued show line survives the reply, a queued label line does not.
    C124: with the bucket out of her view at the drop the telling waits (up to TOLD_WAIT ticks) and she looks at the bucket"""
    V2 = set(VOCAB) | {"bucket"}
    con = C.Conduct(seed=3, transcriber=Transcriber(None), stage=1, imperfect=False)
    _no_sets(con)
    if "bucket" not in con.fast.vocab:
        con.fast.vocab = tuple(con.fast.vocab) + ("bucket",)                # the bucket's word hers (the growth queue admits it)
    bucket = ("bucket", "bucket", "olive", "mat", True)
    p = P(0, seen=seen(("duck", "duck", "yellow", "bucket", True), bucket))
    con.tick(0, p)
    blind = P(1, seen=seen(("duck", "duck", "yellow", "bucket", True)))                  # C124: the bucket out of her view at the drop
    con.told_due = (1, "duck")
    con.reply_due = dict(tick=1, kind="echo", word="oh", obj=None)      # C128: a reply owed at the drop waits for the telling
    s0 = con.tick(1, blind)
    assert s0.line is None and con.told_due is not None and con.reply_due is not None, (s0.line and s0.line.text, con.told_due)
    assert any(a[1] == "look" and a[2] == "bucket" for a in con.acts_open), con.acts_open   # she looks at the bucket first
    s1 = con.tick(2, P(2, seen=seen(("duck", "duck", "yellow", "bucket", True), bucket)))   # the bucket seen: told
    assert s1.line is not None and s1.line.intent == "hide_told" and "bucket" in s1.line.text, (s1.line and (s1.line.intent, s1.line.text))
    assert any(ln.intent == "hide_told" for ln in con.fast.queue), [ln.intent for ln in con.fast.queue]   # the second line waits
    # C135: the bucket out of her sight from the drop on (the child between): told from her memory of it at TOLD_WAIT
    con3 = C.Conduct(seed=3, transcriber=Transcriber(None), stage=1, imperfect=False)
    _no_sets(con3)
    if "bucket" not in con3.fast.vocab:
        con3.fast.vocab = tuple(con3.fast.vocab) + ("bucket",)
    con3.tick(0, P(0, seen=seen(("duck", "duck", "yellow", "mat", True), bucket)))       # she sees the bucket and the duck
    con3.told_due = (1, "duck")
    s3 = None
    for t in range(1, C.TOLD_WAIT + 3):
        s3 = con3.tick(t, P(t, seen=()))                                                 # nothing before her eyes after
        if s3.line is not None:
            break
    assert s3.line is not None and s3.line.intent == "hide_told" and "bucket" in s3.line.text, (t, s3.line and (s3.line.intent, s3.line.text), con3.fast.refused[-2:])
    # a lesson's set across a reply
    con2 = C.Conduct(seed=3, transcriber=Transcriber(None), stage=1, imperfect=False)
    _no_sets(con2)
    p2 = P(0)
    con2.tick(0, p2)
    show = TP.Line("look. the duck.", "show", "plain", "duck", "duck", (), "fast")
    label = TP.Line("a duck.", "label", "plain", "duck", "duck", (), "fast")
    con2.fast.queue = [show, label]
    con2.reply_due = dict(tick=1, kind="social", word=None, obj=None)
    s2 = con2.tick(1, p2)
    kept = [ln.intent for ln in con2.fast.queue]
    assert "show" in kept and "label" not in kept, (kept, s2.line and s2.line.text)
    print(f"lang 64: a hide just done, a reply owed: her next line {s1.line.text!r} (hide_told), its second line queued; the bucket out of sight after the drop: told from memory at tick {t} ({s3.line.text!r}); across a reply the",
          f"queue keeps the show line and drops the label line ({kept})")


def test_comfort_for_a_cry_that_lasts():
    """lang 65 (C120): a wince is not comforted, a cry is, and not again at once. A pain event on 3 ticks: no comfort line. Pain on 8
    ticks running: a comfort (the cry heard on 5 of the last 10). Pain on 6 ticks again 60 ticks later: no comfort (within COMFORT_GAP of
    the last, the cry short; counted as held). Pain on every tick for 70 ticks: a second comfort (the cry on 30 of the last 60). A
    distress event: comfort at once whatever the gap. Day 28's morning: 172 pain ticks, about 50 comforts of 100 ticks each, no lesson"""
    con = C.Conduct(seed=3, transcriber=Transcriber(None), stage=1, imperfect=False)
    _no_sets(con)
    got = []
    def live(t0, n, pain_ticks, ev=("pain", None)):
        out = []
        for k in range(n):
            t = t0 + k
            s = con.tick(t, P(t, events=((ev,) if k in pain_ticks else ())))
            if s.line is not None and s.line.intent in ("comfort", "turn_over"):   # (distress face down is answered by her turn, A90)
                out.append(t)
        return out
    got += live(0, 30, set(range(0, 3)))
    assert not got, got                                                 # a wince of 3 ticks: nothing
    got += live(30, 40, set(range(0, 8)))
    assert len(got) == 1, got                                           # a cry of 8 ticks: one comfort
    got += live(70, 40, set(range(0, 6)))
    assert len(got) == 1 and int(getattr(con, "comfort_held", 0)) >= 1, (got, getattr(con, "comfort_held", None))   # 6 ticks again soon after: held
    got += live(110, 90, set(range(0, 70)))
    assert len(got) >= 2, got                                           # a long cry: comforted again
    n0 = len(got)
    got += live(200, 20, {0}, ev=("distress", None))
    assert len(got) == n0 + 1, got                                      # distress: at once
    print(f"lang 65: a 3-tick wince: no comfort; an 8-tick cry: one; a 6-tick cry 40 ticks later: held ({con.comfort_held}); a 70-tick cry: again;",
          f"distress: at once (comforts at ticks {got})")


def test_distress_owed():
    """lang 61 (A102, 2026-09-27): a distress event that comes while her voice is busy is not lost: the turn is owed and asked at her
    first free tick while the child is still face down; owed no longer once it is not face down; the debt survives a save"""
    con = C.Conduct(seed=3, transcriber=Transcriber(None), stage=1, imperfect=False)
    _no_sets(con)
    s0 = con.tick(0, P(0, child_target="duck"), token=LX.WORD_ID["duck"])     # a right name: she answers, her voice busy for a while
    for t in range(1, 6):
        s = con.tick(t, P(t))
        if s.line is not None:
            break
    assert not con.fast.voice_free(t + 1), "her voice should be busy"
    s = con.tick(t + 1, P(t + 1, events=(("distress", None),), face_down=True))   # the distress while she speaks
    assert s.line is None and con.distress_due == t + 1, (s.line, con.distress_due)
    st = con.state(); con2 = C.Conduct(seed=3, transcriber=Transcriber(None), stage=1, imperfect=False); con2.load_state(st)
    assert con2.distress_due == t + 1
    got = None
    for u in range(t + 2, t + 60):
        s = con.tick(u, P(u, face_down=True))
        if s.line is not None and s.line.intent == "turn_over":
            got = u; break
    assert got is not None and C.Act("turn", "child") in s.acts and con.distress_due == t + 1, (got, con.distress_due)   # owed still
    again = None                                                                    # (its turn may fail): asked again once no turn
    for u in range(got + 1, got + 200):                                             # act runs and it still lies face down, at the
        s = con.tick(u, P(u, face_down=True))                                       # lines' own spacing
        if s.line is not None and s.line.intent == "turn_over":
            again = u; break
    assert again is not None and again - got >= 10, (got, again)
    s = con.tick(again + 1, P(again + 1, face_down=False))                          # turned (by her or itself): nothing owed
    assert con.distress_due is None
    con3 = C.Conduct(seed=3, transcriber=Transcriber(None), stage=1, imperfect=False)
    _no_sets(con3)
    con3.tick(0, P(0, child_target="duck"), token=LX.WORD_ID["duck"])
    for t3 in range(1, 6):
        if con3.tick(t3, P(t3)).line is not None:
            break
    con3.tick(t3 + 1, P(t3 + 1, events=(("distress", None),), face_down=True))
    assert con3.distress_due is not None
    s = con3.tick(t3 + 2, P(t3 + 2, face_down=False))                          # it rolled by itself: nothing owed
    assert con3.distress_due is None and (s.line is None or s.line.intent != "turn_over")
    print(f"61 a distress event during her line is owed: the turn asked at her first free tick ({got - (t + 1)} ticks later), again",
          f"{again - got} ticks after while it still lies face down, dropped once it does not; the debt saved (A102)")

def test_the_stage_advance():
    """lang 66 (C142, 2026-09-30): her stage advances at a dawn once the child has said STAGE2_WORDS distinct words her ear accepted
    EXACT_UNTIL times each (right names or echoes, the vocal book); two such words: stage 1 still; the third: stage 2, logged, and the
    stage survives a save; a stage-2 conduct never goes back. Life day 38 was the 38th day in stage 1: nothing advanced it"""
    con = C.Conduct(seed=3, transcriber=Transcriber(None), stage=1, imperfect=False)
    con.vocal_book = {"duck": K.EXACT_UNTIL, "ball": K.EXACT_UNTIL + 4, "cup": K.EXACT_UNTIL - 1}
    assert not con.dawn(24000) and con.stage == 1, (con.stage, con.vocal_book)
    con.vocal_book["mama"] = K.EXACT_UNTIL
    assert con.dawn(48000) and con.stage == 2 and con.book_log[-1][:3] == (48000, "stage", 2), (con.stage, con.book_log[-1])
    con2 = C.Conduct(seed=3, transcriber=Transcriber(None), stage=1, imperfect=False); con2.load_state(con.state())
    assert con2.stage == 2 and not con2.dawn(72000) and con2.stage == 2, con2.stage
    print(f"lang 66 (C142): two words said right {K.EXACT_UNTIL} times: stage 1; the third: stage 2 at the dawn, logged {con.book_log[-1][3]!r}, saved")


def test_the_word_rung():
    """lang 67 (C145, 2026-09-30): stage 2's rung between babble and a right name. The child says "duck" 45 times with nothing
    attended (no duck in its hand or look): each is echoed and the n-th earns WORTH_WORD e^(-n/HABIT_TAU), none under HABIT_FLOOR
    (30 smiles of 45), labelled "word", never a right name; the vocal book counts them, so a right name of the duck afterwards is
    worn to the same book; in stage 1 the same turns earn no word smile; while her name ask is open, none. Life day 39, stage 2's
    first day: her smiles summed 0.3 over the day"""
    import math as _m
    duck = LX.WORD_ID["duck"]
    N = 45
    toks = {40 + 20 * i: duck for i in range(N)}
    stream = [P(t, child_target=None) for t in range(40 + 20 * N + 10)]
    con = C.Conduct(seed=3, transcriber=Transcriber(None), stage=2, imperfect=False); _no_sets(con)
    said, judg = run(con, stream, tokens=toks)
    worths = [j[1] for j in judg if j[2] == "word"]
    want = [K.WORTH_WORD * _m.exp(-n / K.HABIT_TAU) for n in range(N)]
    want = [w for w in want if w >= K.HABIT_FLOOR]
    assert len(worths) == len(want) and all(abs(a - b) < 1e-9 for a, b in zip(worths, want)), (len(worths), len(want), worths[:3])
    assert not [j for j in judg if j[2] in ("right_name", "vocal_turn")], [j for j in judg if j[2] != "word"][:3]
    assert con.vocal_book["duck"] == len(want), con.vocal_book
    con1 = C.Conduct(seed=3, transcriber=Transcriber(None), stage=1, imperfect=False); _no_sets(con1)
    _s1, judg1 = run(con1, stream, tokens=toks)
    assert not [j for j in judg1 if j[2] == "word"], judg1[:3]
    print(f"lang 67 (C145): stage 2, 'duck' said {N} times with nothing attended: {len(worths)} word smiles, {worths[0]:.2f} to {worths[-1]:.3f},",
          f"then echoed without one; no right name; stage 1: none")


def test_the_fetch_by_memory():
    """lang 68 (C157, 2026-09-30): a lesson's fetch of a toy she knows the place of but does not see begins without a line: the duck out
    of her view and its body in her room, set_near(duck) asked: no line names it, her motion is asked bring_back duck and the look back to
    its eyes, and not the naming word's look (no word); with no duck body known: dropped "unseen" as before. Life days 40 to 42: 13, 12
    and 10 lesson requests a day dropped "unseen" after C121 picked the toys she knows the place of, each costing the play gap"""
    seen_no_duck = tuple(s for s in TOYS if s.id != "duck")
    stream = [P(t, seen=seen_no_duck, child_target=None) for t in range(80)]
    con = C.Conduct(seed=3, transcriber=Transcriber(None), stage=2, imperfect=False); _no_sets(con)
    con.motion.toys = {"duck": 1, "ball": 2}
    con.request("set_near", o="duck")
    said, _j = run(con, stream)
    kinds = [(a[2], a[3]) for a in con.motion.acts]                        # every act asked of her motion (the stub's log)
    assert ("bring_back", "duck") in kinds and ("look", "child_eyes") in kinds and ("look", "duck") not in kinds, kinds
    assert not [s for s in said if "duck" in s[1].text], [s[1].text for s in said][:3]
    assert not [r for r in con.fast.refused if "unseen: 'duck'" in str(r)], con.fast.refused[-3:]
    assert [r for r in con.fast.refused if "fetched by memory" in str(r)], con.fast.refused[-3:]
    con2 = C.Conduct(seed=3, transcriber=Transcriber(None), stage=2, imperfect=False); _no_sets(con2)
    con2.request("set_near", o="duck")
    run(con2, stream)
    kinds2 = [(a[2], a[3]) for a in con2.motion.acts]
    assert ("bring_back", "duck") not in kinds2 and [r for r in con2.fast.refused if "unseen: 'duck'" in str(r)], (kinds2, con2.fast.refused[-3:])
    print(f"lang 68 (C157): the duck out of her view, its place known: bring_back asked of her motion without a line; unknown: dropped unseen")


def test_the_words_recover():
    """lang 69 (C159, 2026-09-30): the words' habituation recovers over a night: at a dawn each word's count in the vocal book falls to
    HABIT_KEEP of itself (30 -> 15: the next day pays smiles from e^(-15/10) down to the floor again; 1 -> 0), the stage's advance
    judged first and the book untouched in stage 1 (it is that stage's record, C142). Life days 40 to 42: word smiles 533, 181 and 112
    by the middle of the day, the pool draining"""
    con = C.Conduct(seed=3, transcriber=Transcriber(None), stage=2, imperfect=False)
    con.vocal_book = {"duck": 30, "ball": 3, "cup": 1}
    want = {k: int(v * K.HABIT_KEEP) for k, v in con.vocal_book.items()}
    assert not con.dawn(48000) and con.vocal_book == want, con.vocal_book
    con1 = C.Conduct(seed=3, transcriber=Transcriber(None), stage=1, imperfect=False)
    con1.vocal_book = {"duck": K.EXACT_UNTIL, "ball": K.EXACT_UNTIL, "mama": K.EXACT_UNTIL}
    assert con1.dawn(48000) and con1.stage == 2 and con1.vocal_book["duck"] == int(K.EXACT_UNTIL * K.HABIT_KEEP), (con1.stage, con1.vocal_book)
    con0 = C.Conduct(seed=3, transcriber=Transcriber(None), stage=1, imperfect=False)
    con0.vocal_book = {"duck": 30}
    assert not con0.dawn(48000) and con0.vocal_book == {"duck": 30}, con0.vocal_book      # stage 1: the book is the stage's record
    print(f"lang 69 (C159): the vocal book at a stage-2 dawn {dict(duck=30, ball=3, cup=1)} -> {want}; the stage advanced first on a stage-1 book, then halved; stage 1 untouched")


def test_the_grunt_not_echoed():
    """lang 70 (C172, 2026-09-30): the child's "oh" is not echoed back and her social reply no longer opens with "oh": the child says "oh" 30
    times with the duck attended and 30 times with nothing attended; none of her replies echoes the grunt ("oh! oh.") or opens the social reply
    with it ("oh? hi pip."), the ones with the duck attended name the duck (the object reply "oh! the duck." keeps its measured form); a real
    word ("duck") said with nothing attended is still echoed. Life day 44: "oh" was 42% of the child's tokens and 38% of every word it heard"""
    oh = LX.WORD_ID["oh"]; duck = LX.WORD_ID["duck"]
    toks = {40 + 20 * i: oh for i in range(60)}
    stream = [P(t, child_target=("duck" if 40 <= t < 640 else None)) for t in range(40 + 20 * 60 + 10)]
    con = C.Conduct(seed=3, transcriber=Transcriber(None), stage=2, imperfect=False); _no_sets(con)
    said, _j = run(con, stream, tokens=toks)
    texts = [s[1].text for s in said]
    assert texts, "she must reply"
    ws_ = [set(w.strip("!.,?") for w in x.split()) for x in texts]
    bad = [x for x, w_ in zip(texts, ws_) if w_ <= {"oh"} or (("hi" in w_ or "mama" in w_) and "oh" in w_)]   # the echo of a grunt, the social
    assert not bad, bad[:5]                                                                          # reply opening with it (the object reply
    assert any("duck" in x for x in texts), texts[:8]                                                # "oh! the duck." is the peak table's: kept)
    con2 = C.Conduct(seed=3, transcriber=Transcriber(None), stage=2, imperfect=False); _no_sets(con2)
    said2, _j2 = run(con2, [P(t, child_target=None) for t in range(200)], tokens={40: duck, 80: duck, 120: duck})
    assert any("duck" in s[1].text for s in said2), [s[1].text for s in said2][:5]
    print(f"lang 70 (C172): 60 grunts: {len(texts)} replies, none opening with or echoing 'oh' ({texts[0]!r}, {texts[-1]!r}); a word is still echoed")


def test_the_function_word_not_echoed():
    """lang 71 (C197, 2026-10-01): the child's function word is not parroted back. The child says "the" 30 times with the duck attended and
    30 times with nothing attended, then "is" the same: none of her replies is the word alone or doubled ("the. the!", "is! is."), the ones
    with the duck attended name the duck, and a noun ("duck") said with nothing attended is still echoed. Life day 53: a fifth of her replies
    parroted the child's "the", "a" and "is", her day's speech 53 distinct words to the child's 76"""
    the = LX.WORD_ID["the"]; is_ = LX.WORD_ID["is"]; duck = LX.WORD_ID["duck"]
    toks = {40 + 20 * i: (the if i < 60 else is_) for i in range(120)}
    stream = [P(t, child_target=("duck" if (40 <= t < 640 or 1240 <= t < 1840) else None)) for t in range(40 + 20 * 120 + 10)]
    con = C.Conduct(seed=3, transcriber=Transcriber(None), stage=2, imperfect=False); _no_sets(con)
    said, _j = run(con, stream, tokens=toks)
    texts = [s[1].text for s in said]
    assert texts, "she must reply"
    ws_ = [[w.strip("!.,?") for w in x.split()] for x in texts]
    bad = [x for x, w_ in zip(texts, ws_) if set(w_) <= {"the"} or set(w_) <= {"is"} or set(w_) <= {"the", "yes"} or set(w_) <= {"is", "yes"}]
    assert not bad, bad[:6]
    assert any("duck" in x for x in texts), texts[:8]
    con2 = C.Conduct(seed=3, transcriber=Transcriber(None), stage=2, imperfect=False); _no_sets(con2)
    said2, _j2 = run(con2, [P(t, child_target=None) for t in range(200)], tokens={40: duck, 80: duck, 120: duck})
    assert any("duck" in s[1].text for s in said2), [s[1].text for s in said2][:5]
    print(f"lang 71 (C197): 120 function words: {len(texts)} replies, none the word alone or doubled ({texts[0]!r}, {texts[-1]!r}); the duck named when attended; a noun still echoed")


def test_the_planner_unseen_line_is_a_show():
    """lang 74 (C206): a planner (steer) row 'you see the block.' with the block out of the child's view (the test room's, on the
    sofa) is refused as untrue and becomes her show of the block, once; a row about a toy in its view is said as it was"""
    room = tuple(o for o in TOYS if o.id != "block_red")                 # one block in the room, on the sofa, out of its view
    con = _pair_con()
    _run_to(con, 200, seen_=room)
    assert con.fast.add_steer("you see the block.", "any", 200)[0]
    out = _run_to(con, 6, t0=200, seen_=room)
    assert any("a show of it instead (C206)" in r[2] for r in con.fast.refused), con.fast.refused[-3:]
    assert [tuple(a[1:3]) for a in con.acts_open if a[1] == "show"] == [("show", "block")], con.acts_open
    n_req = sum(1 for r in con.fast.refused if "(C206)" in r[2])
    _run_to(con, 30, t0=206, seen_=room)
    assert sum(1 for r in con.fast.refused if "(C206)" in r[2]) == n_req == 1, [r for r in con.fast.refused if "C206" in r[2]]
    con2 = _pair_con()
    _run_to(con2, 200)
    assert con2.fast.add_steer("you see the duck.", "any", 200)[0]
    said2 = [getattr(s.line, "text", s.line) for t, s in _run_to(con2, 12, t0=200, target=lambda t, c: "duck") if s.line is not None]   # it attends the duck: a follow-in
    assert said2 and all("duck" in ln for ln in said2) and not any("C206" in r[2] for r in con2.fast.refused) \
        and not [a for a in con2.acts_open if a[1] == "show"], (said2, con2.fast.refused[-3:], con2.acts_open)   # named, not shown
    print(f"lang 74: the planner's 'you see the block.' (unseen) became her show of the block, once ({n_req} request); the duck in its view named, not shown {said2}")


def test_the_unseen_label_is_a_show():
    """lang 73 (C201): a requested label of a toy the child cannot see (the block on the sofa in the test's room) becomes her show of
    it: the show act opened, a show frame said, the request logged; a label of a toy in its view stays a label"""
    def frames_of(intent, oid):
        o = [x for x in TOYS if x.id == oid][0]
        return {TP.key(TP.fill(fr, o=o)[0]) for fr in TP.FRAMES[intent]}
    con = _pair_con()
    _run_to(con, 200)                                                    # past the set gap the test room starts under (_no_sets)
    con.request("label", o="block")
    out = _run_to(con, 3, t0=200)
    assert [tuple(a[1:3]) for a in con.acts_open if a[1] == "show"] == [("show", "block")], (con.acts_open, con.fast.refused[-3:])
    assert any("a show of it instead (C201)" in r[2] for r in con.fast.refused), con.fast.refused[-3:]
    out += _run_to(con, 12, t0=203)
    said = [getattr(s.line, 'text', s.line) for t, s in out if s.line is not None]
    assert said and all(TP.key(ln) in frames_of("show", "block") for ln in said), said
    con2 = _pair_con()
    _run_to(con2, 200)
    con2.request("label", o="duck")
    out2 = _run_to(con2, 3, t0=200)
    assert not [a for a in con2.acts_open if a[1] == "show"] and not any("C201" in r[2] for r in con2.fast.refused), (con2.acts_open, con2.fast.refused[-3:])
    said2 = [getattr(s.line, 'text', s.line) for t, s in out2 + _run_to(con2, 12, t0=203) if s.line is not None]
    assert said2 and all(TP.key(ln) in frames_of("label", "duck") for ln in said2), said2
    print(f"lang 73: a label of the block the child cannot see became her show of it {said}; a label of the duck in its view stayed one {said2}")


def test_the_worn_word_not_echoed():
    """lang 72 (C198, 2026-10-01): a worn word is not parroted back. The child says "duck" 40 times with nothing attended: her first replies echo
    it ("duck! duck." as imitation), and once the vocal book holds HABIT_TAU of it (her smile worn) no reply is the bare word any more (the
    social line); with the duck attended the worn word is answered by naming the thing in view. Life day 54: "mama! mama." 222 lines, the
    child saying "mama" 703 times"""
    from body.sim.lang import consts as K
    duck = LX.WORD_ID["duck"]
    toks = {40 + 20 * i: duck for i in range(40)}
    stream = [P(t, child_target=None) for t in range(40 + 20 * 40 + 10)]
    con = C.Conduct(seed=3, transcriber=Transcriber(None), stage=2, imperfect=False); _no_sets(con)
    said, _j = run(con, stream, tokens=toks)
    texts = [s[1].text for s in said]
    def _txt(x):
        return x.text if hasattr(x, "text") else (x[0] if isinstance(x, tuple) else str(x))
    parrots = {_txt(TP.fill(f_, w="duck")) for k_ in ("echo", "recast", "echo_word", "recast_word") for f_ in TP.FRAMES[k_]}   # her bare echoes of the word
    bare = [i for i, x in enumerate(texts) if x in parrots]
    assert bare and bare[0] < 12, (bare[:5], texts[:5], sorted(parrots))              # a fresh word is echoed as imitation
    assert int(con.vocal_book.get("duck", 0)) >= int(K.HABIT_TAU), con.vocal_book     # worn by its smiles
    late = texts[-10:]
    assert not any(x in parrots for x in late), (late, sorted(parrots))                # a worn word is not parroted
    con2 = C.Conduct(seed=3, transcriber=Transcriber(None), stage=2, imperfect=False); _no_sets(con2)
    con2.vocal_book["duck"] = int(K.HABIT_TAU) + 5
    said2, _j2 = run(con2, [P(t, child_target="duck") for t in range(400)], tokens={40 + 20 * i: duck for i in range(15)})
    t2 = [s[1].text for s in said2]
    echoes = {_txt(TP.fill(f_, w="duck")) for k_ in ("echo", "echo_word") for f_ in TP.FRAMES[k_]}   # (a right name's confirm line shares the recast's words: allowed)
    assert t2 and any("duck" in x for x in t2) and not any(x in echoes for x in t2), t2[:6]
    print(f"lang 72 (C198): 'duck' said 40 times: echoed as imitation at first (reply {bare[0]}: {texts[bare[0]]!r}), the book at {con.vocal_book.get('duck')}, "
          f"the last ten replies none the bare word ({late[-1]!r}); worn and attended: named in a line ({t2[0]!r})")


TESTS = [test_the_worn_word_not_echoed, test_the_function_word_not_echoed, test_frames_and_birth_lines, test_line_check_refuses, test_compose_from_percept, test_variation_sets_and_repeats,
         test_replies_and_judgments, test_talk_over_and_turns, test_new_word_and_night, test_steer, test_transcriber,
         test_ear_rules, test_transcriber_with_ear, test_ledger_standing, test_replay_exact, test_cost, test_ear_templates_exact,
         test_ear_exact_across_threads, test_approximation_in_context, test_asks_answered, test_cut_words_not_said,
         test_new_words_in_a_day, test_acts_carry_the_object, test_line_check_truth, test_small_items,
         test_replay_across_processes, test_claude_never_asks, test_name_ask_answered_after_its_question,
         test_calls_out_of_sight, test_introduce_only_what_she_can_show, test_give_before_its_word, test_ear_deliberate_writes,
         test_gaze_leak_closed, test_new_word_on_its_peak, test_reads_trunk_and_hands, test_held_pairs_a55, test_echoes,
         test_claude_shows_and_forms, test_imperfect_parent_and_talk_over, test_sets_distinct_in_words,
         test_asks_she_can_judge, test_reading_held_three_ticks, test_low_items_fourth, test_low_items_fifth,
         test_trial_protocol, test_trial_chance_and_counterbalance, test_name_trial_foil, test_everyday_asks_teaching_only,
         test_trial_property, test_trial_low_items, test_understood_controlled, test_name_foils_matched,
         test_low_items_ninth, test_trial_one_timeline, test_trial_invariance, test_trial_window_share,
         test_trial_carrier_phrase, test_trial_acceptance_a60b, test_trial_levels_no_feedback, test_motor_judgments, test_her_lessons_and_hands, test_vocal_habituation, test_distress_owed, test_the_stage_advance, test_the_word_rung, test_the_fetch_by_memory, test_the_words_recover, test_the_grunt_not_echoed]

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
