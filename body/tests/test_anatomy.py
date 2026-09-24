"""the anatomy declared (docs/SIM_DESIGN.md 8.2 and 8.3; the core refactor's steps R1 to R6 and R9). Run: python3 -m body.tests.test_anatomy
(the organ tests run these too). `LanguageAnatomy(tok, cfg)` must rebuild exactly the symbols a life derived from its tokenizer before
R2, under every constant that moves them and on a tokenizer laid out otherwise, and building it must leave the body untouched (R1). The
life is built with it and reads its symbols and its text there, never the tokenizer; a life given the anatomy in the tokenizer's place
is the same life; an anatomy declared under other constants, or not a language one, is refused (R2). The tick's reward is the
anatomy's reward sources, felt in their declared order and added one at a time in it: bit for bit the reward `_sense` summed before
R3, on every tick, under the switches the pinned digests do not reach (cost_in_reward, world_mask off, own_store) (R3). The cortex's
input is the anatomy's channel codes summed in its declared order, bit for bit the sum before R4 for the diary's (gradients too); the
window holds each channel under its declared field and every reader goes through it; a later channel's forecast head is built last
and taught; all eleven places the input is made pass the anatomy's channels (R4). The voice is effector 0 with its organs' names, its
ear and its cost, and the diary gains nothing from the effectors' wiring; its gate's lesson is bit for bit the lesson before R5; the
defect fixes 4, 5 and 8 are switches off by their absence, each the lesson with only its fix written in, and live; a body with later
effectors has the diary's organs and striatum bit for bit beside its effectors' (built last, their blocks appended), draws the voice
first each tick, reads each effector's joints, hears each act after its own sound, teaches each gate and actor, rests them at night
and keeps them through a save; all eleven places the input is made pass the effectors' acts (R5); each later effector's act adds its
declared cost to the fatigue, a one-joint effector's reserved acts are never drawn and the later effectors' traces decay every tick under
actor_trace_tick (the R5 verifier's three). The body lives in a world (R9): the diary's world is today's queue and face, and a life
through the world loop, its sleep switch pausing the world, is the whole life before R9; a stub of the simulated world ticks a body of
frames (its words, its face, a sense of its frames, two effectors acting on it; every learning rate at 0) through a day, a night that
pauses it and a morning; the deadline switch is off, and on lets the world run on its own clock; the pace log records the ticks and the
nights and changes nothing. The motor timing part (R6), on a tiny arm in a stub world (two joints, its body sense their velocities, a
touch of pain and the parent's hand, a wall, demonstrations): built last from the body's seed and absent from the language body;
act_inv learning online on its own acts alone, its reliability Cohen's kappa per joint (labels blind to the act read 0 however skewed
the acts' own rates); act_pred's targets position by position
(its own acts, act_inv's reading of what moved it where it rested at act_inv's reliability, the rest for an effector with no inverse
model), its gradient and its learning, what it learns scaling with its labels' reliability under its own optimizer (GatedAdam); the forward half foreseeing each tick's body sense, its error correcting the proposal and
steering a chunk; the learned stops (act_pred's rest, the gate's own no, the reflex, the declared end, chunk_max) and the reflex's
tick (its act to the world, no draw, no credit, no eligibility). The striatum's rows per joint (R5b): a body of 34 joints as eight
limbs builds within memory (the language block first and as the diary's) and lives."""
import collections
import copy
import math
import os
import pickle
import random
import sys
import tempfile
import time
import types

import torch
from tokenizers import Tokenizer, models

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)   # this tree's body, not a fixed one
from body.life import Life, PHYSIOLOGY  # noqa: E402
from body.core.anatomy import (Anatomy, Channel, EarChannel, Effector, EffortReward, FaceChannel, FaceReward,  # noqa: E402
                                LanguageAnatomy, RewardSource, VoiceEffector, WorldWordsReward, anatomy_for)
from body.model import Organs  # noqa: E402
from body.core.world import Frame  # noqa: E402

TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
FIELDS = ("sil", "nl", "space_id", "eot", "end_id", "reserved", "bans")


def _born(tok, cfg):
    return Life.birth(tok, device="cpu", d=64, layers=2, heads=2, window=32, cfg=cfg, seed=0)


def _served_cfg():
    """the served save's constants as the determinism check's pins froze them (tools/pins/served_cfg.pkl), the physiology's keys"""
    p = os.path.join(ROOT, "tools", "pins", "served_cfg.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return {k: v for k, v in pickle.load(f).items() if k in PHYSIOLOGY}


def _toy_tok():
    """a tokenizer laid out otherwise: the rest not at 0, a special after the letters, no display symbol"""
    return Tokenizer(models.WordLevel(vocab={"a": 0, "<rest>": 1, " ": 2, "<z>": 3, "b": 4, "<end>": 5, "c": 6}, unk_token="<z>"))


def _same(a, life, label):
    for f in FIELDS:
        got, want = getattr(a, f), getattr(life, f)
        assert type(got) is type(want) and got == want, f"{label}: {f} is {got!r}, the life derives {want!r}"
    assert a.vocab == life.m.vocab == a.tok.get_vocab_size(), f"{label}: vocab {a.vocab}, the organs' {life.m.vocab}"
    ear, face, voice = a.channel("ear"), a.channel("face"), a.effectors[0]
    assert (ear.kind, ear.size, ear.rest_id, ear.end_id, ear.reserved, ear.partner) == ("symbol", life.m.vocab, life.sil, life.end_id, life.reserved, True), label
    assert a.partner is ear and a.channels == [ear, face] and (face.kind, face.size, face.partner) == ("vector", 2, False), label
    assert (type(ear), ear.organ, ear.field, ear.forecast, type(face), face.organ, face.field, face.forecast, a.inner_at, a.words) == \
        (EarChannel, "E", "x", True, FaceChannel, "face_in", "face", False, 2, ear), label
    assert (voice.name, voice.factors, voice.rest_id, voice.end_id, voice.reserved) == ("voice", [life.m.vocab], life.sil, life.space_id, life.bans), label
    assert (type(voice), voice.organ, voice.gate, voice.actor, voice.field, a.effectors) == (VoiceEffector, "E", "mouth_gate", "actor", "xo", [voice]), label
    assert [s.name for s in a.rewards] == ["face", "world_r", "cost"], label
    assert [type(s) for s in a.rewards] == [FaceReward, WorldWordsReward, EffortReward], label
    assert [s.clip for s in a.rewards] == [2, None, None], label
    assert [s.keys for s in a.rewards] == [(), ("world_r", "world_mask"), ("cost_in_reward", "symbol_cost", "gate_fatigue")], label
    a.check()


def test_language_anatomy_equals_the_tokenizers_fields():
    """anatomy 1: under the physiology's defaults, the served constants and every constant that names a symbol, on the served tokenizer
    and on one laid out otherwise, LanguageAnatomy's symbols are the life's own (built from the cfg given, and from the life's cfg)"""
    cases = [("the physiology", TOK, {}), ("end_symbol eot", TOK, dict(end_symbol="eot")),
             ("another rest, no display symbol", TOK, dict(rest_token="<eot_model>", display_token="\t", end_symbol="eot")),
             ("a rest the tokenizer lacks", TOK, dict(rest_token="<none>")),
             ("the toy tokenizer", _toy_tok(), dict(rest_token="<rest>", end_token="<end>", end_symbol="eot")),
             ("the toy tokenizer, rest as the end", _toy_tok(), dict(rest_token="<rest>", end_token="<end>", display_token="c"))]
    sc = _served_cfg()
    if sc is not None:
        cases.append(("the served constants (tools/pins/served_cfg.pkl)", TOK, sc))
    for label, tok, cfg in cases:
        life = _born(tok, cfg)
        _same(LanguageAnatomy(tok, cfg), life, label)
        _same(LanguageAnatomy(tok, life.cfg), life, label + " (the life's cfg)")
    # the served tokenizer today, written out: 107 symbols, the rest <pad> at 0, <eot_human> at 1, the space 105, the newline 106,
    # the reserved the ten other <...> tokens and the newline; the end the rest under the physiology (end_symbol "rest")
    a = LanguageAnatomy(TOK, {})
    assert (a.vocab, a.sil, a.eot, a.space_id, a.nl, a.end_id) == (107, 0, 1, 105, 106, 0), (a.vocab, a.sil, a.eot, a.space_id, a.nl, a.end_id)
    assert a.reserved == a.bans == list(range(1, 11)) + [106] and a.reserved is not a.bans, (a.reserved, a.bans)
    print("anatomy 1: LanguageAnatomy's symbols equal the life's under", len(cases), "constant sets and two tokenizers;",
          "today: vocab 107, rest 0, eot 1, space 105, newline 106, reserved 1-10 and 106" + ("" if sc is not None else " (no served pins here)"))


def test_language_anatomy_is_inert():
    """anatomy 2 (SIM_DESIGN.md 8.3 item 4): building the language anatomy draws no random number, builds no module and moves no
    part of a life born beside it"""
    life = _born(TOK, {})
    g0, l0 = torch.get_rng_state().clone(), life.gen.get_state().clone()
    sd0 = {k: v.clone() for k, v in life.m.state_dict().items()}; v0 = sorted(vars(life))
    a = LanguageAnatomy(TOK, life.cfg)
    assert torch.equal(g0, torch.get_rng_state()) and torch.equal(l0, life.gen.get_state()), "building the anatomy drew a random number"
    assert sorted(vars(life)) == v0 and all(torch.equal(sd0[k], v) for k, v in life.m.state_dict().items()), "building the anatomy moved the life"
    parts = [a, *a.channels, *a.effectors, *a.rewards]
    mods = [(type(p).__name__, k) for p in parts for k, v in vars(p).items() if isinstance(v, (torch.nn.Module, torch.Tensor))]
    assert not mods, f"the anatomy holds modules or tensors before a life binds them: {mods}"
    print("anatomy 2: building it draws no random number, builds no module, moves nothing of the life")


def test_anatomy_check():
    """anatomy 3: the declaration's own check refuses two partners, a rest that is reserved, a symbol outside the alphabet, an
    unknown kind, a name twice, no effector, a reward source reading an unknown constant, no reward source and a clip not above zero;
    a bare RewardSource declares no feeling (a bare Channel observes the world's frames since step R9: anatomy 21)"""
    def ok():
        return Anatomy([Channel("ear", "symbol", 5, organ="E", forecast=True, rest_id=0, end_id=1, reserved=[4], partner=True),
                        Channel("face", "vector", 2, organ="face_in")],
                       [VoiceEffector("voice", [5], rest_id=0, end_id=2, reserved=[4])], [RewardSource("face"), RewardSource("cost", ("symbol_cost",))])
    ok().check()
    bad = []
    a = ok(); a.channels[1].partner = True; bad.append(("two partners", a))
    a = ok(); a.channels[0].reserved = [0]; bad.append(("the rest reserved", a))
    a = ok(); a.channels[0].end_id = 5; bad.append(("a symbol outside", a))
    a = ok(); a.effectors[0].reserved = [7]; bad.append(("an act outside", a))
    a = ok(); a.channels[1].kind = "image"; bad.append(("an unknown kind", a))
    a = ok(); a.channels[1].name = "ear"; bad.append(("a name twice", a))
    a = ok(); a.effectors = []; bad.append(("no effector", a))
    a = ok(); a.rewards[1].keys = ("symbol_cots",); bad.append(("an unknown constant", a))
    a = ok(); a.rewards = []; bad.append(("no reward source", a))
    a = ok(); a.rewards[0].clip = 0; bad.append(("a clip of zero", a))
    a = ok(); a.channels = []; bad.append(("no channel", a))                                        # step R4's refusals
    a = ok(); a.channels.reverse(); bad.append(("channel 0 not the words", a))
    a = ok(); a.channels[0].organ = "face_in"; bad.append(("the words not read through the lexicon", a))
    a = ok(); a.channels[0].forecast = False; bad.append(("the words without their forecast", a))
    a = ok(); a.channels[0].partner = False; a.channels[1].partner = True; bad.append(("the partner not channel 0", a))
    a = ok(); a.channels[1].field = "ear"; bad.append(("a window field twice", a))
    a = ok(); a.channels[1].field = "xo"; bad.append(("a window field the core writes", a))
    a = ok(); a.channels[1].organ = None; bad.append(("a channel with no organ", a))
    a = ok(); a.channels[1].name = "the face"; bad.append(("a name that cannot name a head", a))
    a = ok(); a.inner_at = 3; bad.append(("the bundle after more channels than there are", a))
    for label, a in bad:
        try:
            a.check()
        except ValueError:
            continue
        raise AssertionError(f"the check let pass {label}")
    try:
        RewardSource("r").felt(Frame(0, {"ear": 0}, 0.0), None)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("a bare RewardSource felt something")
    print("anatomy 3: the check refuses", len(bad), "faulty declarations; a bare reward source feels nothing")


def _differs(A, B):
    """the first difference between two lives' weights, symbols, working attributes' names, random streams and pages, or None"""
    sa, sb = A.m.state_dict(), B.m.state_dict()
    if sorted(sa) != sorted(sb):
        return "the organs' names"
    for k in sa:
        if not torch.equal(sa[k], sb[k]):
            return f"the organ {k}"
    for f in FIELDS:
        if getattr(A, f) != getattr(B, f):
            return f"the symbol {f}"
    if sorted(vars(A)) != sorted(vars(B)):
        return "the working attributes' names"
    if not torch.equal(A.gen.get_state(), B.gen.get_state()):
        return "the life's random stream"
    if [e[:2] for e in A.page] != [e[:2] for e in B.page]:
        return "the page"
    return None


def _live(life, lines=("what do you want?", "I want milk"), faces={9: 2.0, 10: 0.0}, ticks=60):
    for t in range(ticks):
        if t % 30 == 0 and t // 30 < len(lines):
            life.type_text(lines[t // 30], who="parent")
        if t in faces:
            life.set_face(faces[t])
        life.tick()


def test_life_reads_its_anatomy():
    """anatomy 4 (step R2): a life born of a tokenizer holds the diary's anatomy, built under the life's own constants; its symbols
    are the anatomy's, in lists of its own (nothing the life does can move the declaration); the tokenizer is no attribute of the
    life, only the anatomy's (read through `life.tok`); the anatomy is still unbound (no module, no tensor)"""
    for label, cfg in (("the physiology", {}), ("rest as the end, another display symbol", dict(end_symbol="rest", display_token="\t")),
                       ("the served constants", _served_cfg() or {})):
        life = _born(TOK, cfg); a = life.anatomy
        assert type(a) is LanguageAnatomy and a.symbols() == LanguageAnatomy(TOK, life.cfg).symbols(), label
        _same(a, life, label)
        assert life.reserved is not a.reserved and life.bans is not a.bans and life.bans is not life.reserved, f"{label}: the life shares the anatomy's lists"
        assert "tok" not in vars(life) and "anatomy" in vars(life) and life.tok is TOK and a.tok is TOK, f"{label}: the tokenizer is not the anatomy's alone"
        parts = [a, *a.channels, *a.effectors, *a.rewards]
        bound = [(type(p).__name__, k) for p in parts for k, v in vars(p).items() if isinstance(v, (torch.nn.Module, torch.Tensor))]
        assert not bound, f"{label}: the life bound modules or tensors into its anatomy: {bound}"
    try:
        life.tok = TOK
    except AttributeError:
        pass
    else:
        raise AssertionError("life.tok could be set: the tokenizer is the anatomy's, read-only")
    print("anatomy 4: the life holds the diary's anatomy under its own constants and reads its symbols there, in its own lists;",
          "the tokenizer is only the anatomy's; the anatomy is unbound")


def test_an_anatomy_in_the_tokenizers_place():
    """anatomy 5 (step R2): `Life.birth` and `Life.load` given a LanguageAnatomy in the tokenizer's place give the same life as the
    tokenizer does (the same weights, symbols, attributes, random streams, and the same page after a minute of the script), and the
    same random streams after the birth; an anatomy declared under other constants is refused, and one not of language is refused
    (the core's words are a language's, read through a tokenizer; a language anatomy with later channels and effectors is taken:
    anatomies 12 and 18, and since step R9 lives in a world of frames: anatomy 21)"""
    import tempfile
    for label, cfg in (("the physiology", {}), ("the served constants", _served_cfg() or {})):
        torch.manual_seed(123); A = _born(TOK, cfg); gA = torch.get_rng_state().clone()
        torch.manual_seed(123); B = _born(LanguageAnatomy(TOK, cfg), cfg); gB = torch.get_rng_state().clone()
        assert torch.equal(gA, gB), f"{label}: a birth from the anatomy left the global random stream elsewhere"
        assert _differs(A, B) is None, f"{label}: born from the anatomy, {_differs(A, B)} differs"
        _live(A); _live(B)
        assert _differs(A, B) is None and len(A.page) == 120, f"{label}: a minute lived, {_differs(A, B)} differs"
        fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
        try:
            A.save(path)
            torch.manual_seed(7); C = Life.load(path, TOK, save_path=None)          # a load builds its organs from the global stream
            torch.manual_seed(7); D = Life.load(path, LanguageAnatomy(TOK, C.cfg), save_path=None)   # (the parts a save does not hold)
        finally:
            os.remove(path)
        assert _differs(C, D) is None and C.ticks == D.ticks == 60, f"{label}: loaded from the anatomy, {_differs(C, D)} differs"
    refused = []
    for what, call in (("an anatomy under another rest", lambda: _born(LanguageAnatomy(TOK, dict(rest_token="<eot_model>")), {})),
                       ("an anatomy under another end", lambda: _born(LanguageAnatomy(TOK, dict(end_symbol="eot")), dict(end_symbol="rest"))),
                       ("an anatomy with the life's constants given later", lambda: Life(A.m, LanguageAnatomy(TOK, {}), cfg=dict(display_token="\t")))):
        try:
            call()
        except ValueError:
            refused.append(what); continue
        raise AssertionError(f"the life took {what}")
    other = Anatomy([Channel("words", "symbol", 5, organ="E", forecast=True, rest_id=0, partner=True), Channel("eye", "vector", 4, organ="face_in")],
                    [VoiceEffector("voice", [5], rest_id=0), Effector("arm", [5, 5], rest_id=12)], [RewardSource("face")]).check()
    for call in (lambda: anatomy_for(other, {}), lambda: _born(other, {}), lambda: Life(A.m, other)):
        try:
            call()
        except NotImplementedError:
            continue
        raise AssertionError("a life was built on an anatomy not of language (its words read through no tokenizer)")
    print("anatomy 5: born and loaded from a LanguageAnatomy in the tokenizer's place, the same life under 2 constant sets;",
          f"refused: {len(refused)} anatomies declared under other constants, and one not of language")


class _CountingAnatomy(LanguageAnatomy):
    """the diary's anatomy whose text passes through counters, its tokenizer held apart (so the life's own reaches can be refused)"""

    def __init__(self, tok, cfg=None):
        super().__init__(tok, cfg)
        self._t = tok; self.calls = collections.Counter()

    def symbol(self, ch):
        self.calls["symbol"] += 1
        return self._t.token_to_id(ch)

    def decode(self, ids):
        self.calls["decode"] += 1
        return self._t.decode(ids)


class _NoTokenizer:
    """stands where the tokenizer was: any reach for it is an error"""

    def __getattr__(self, name):
        raise AssertionError(f"the body reached for the tokenizer ({name}) past its anatomy")


def test_the_body_reads_text_through_its_anatomy():
    """anatomy 6 (step R2): with the anatomy's tokenizer taken away (only its `symbol` and `decode` keep one), a life types, hears,
    speaks, sleeps a night on its utterances and a corpus (the report's examples) and lives on: the text passes through the anatomy
    alone, and the page reads as the tokenizer would write it"""
    import tempfile
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.write("The cat sat on the mat. It was a good day.\n"); f.close()
    try:
        cfg = dict(wake_ticks=80, wake_every=8, gate_every=8, night_rounds=1, night_starts=4, night_batch=4, rem_dreams=2, rem_steps=2,
                   dream_source="utterances", dream_who=1, night_load=0.0, write_floor=1e-30, dream_corpus_n=2, dream_corpus_file=f.name)
        a = _CountingAnatomy(TOK, cfg)
        life = _born(a, cfg)
        assert life.anatomy is a
        a.tok = _NoTokenizer()
        _live(life, lines=("go up", "we go up"), ticks=90)
        assert life.nights == 1 and not (life.last_night or {}).get("error"), (life.nights, (life.last_night or {}).get("error"))
        assert (life.last_night or {}).get("examples"), "the night's report wrote no examples"
    finally:
        os.remove(f.name)
    assert a.calls["symbol"] >= len("go up") + len("we go up") and a.calls["decode"] >= 2 * 90, dict(a.calls)
    heard = "".join(e[0] for e in life.page if e[1] == 0)
    assert heard.startswith("go up") and "we go up" in heard, heard
    assert len(life._corpus_pool()) == 2 and a.calls["symbol"] > len("go up") + len("we go up"), dict(a.calls)
    print(f"anatomy 6: a day, a night on the utterances and a corpus, and a morning with the tokenizer out of reach: the text went",
          f"through the anatomy ({a.calls['symbol']} symbols typed or read, {a.calls['decode']} decoded)")


def _reward_before_r3(self, u):
    """the reward as `_sense` summed it before step R3 (body/core/senses.py at commit 96f72a9; the ring and the actor's lines are
    anatomy 8's): the reference the sources are held to. Returns the felt event, the face's term and the reward."""
    # the face: a change is felt; a held face is silence; easing off is not an event
    lvl = max(-6, min(6, int(self.face_now)))
    felt = 0
    if lvl != self.level:
        if abs(lvl) > abs(self.level) or lvl * self.level < 0:
            felt = lvl
        self.level = lvl
    r = float(max(-2, min(2, felt)))                    # the world's reward: the felt face, clipped like a press
    r_face = r
    if u != self.sil:
        wr = float(self.cfg.get("world_r", 0.0))        # the world's words as reward (0 = off)
        if wr and not (int(self.cfg.get("world_mask", 0)) and getattr(self, "_acted_last", False)):
            r += wr                                     # not heard over its own voice (world_mask)
    if self.cfg.get("cost_in_reward") and getattr(self, "_acted_last", False):
        r -= float(self.cfg["symbol_cost"]) * (1.0 + (self.fatigue / float(self.cfg["gate_fatigue"])) ** 2)
    return felt, r_face, r


def _reward_by_sources(a, life, u):
    """the anatomy's sources felt on one frame, in their order, and summed in it (as `_sense` sums them)"""
    frame = Frame(0, {"ear": u}, life.face_now)
    judge, *others = a.rewards
    felt = judge.felt(frame, life)
    r_face = r = judge.term(felt)
    for s_ in others:
        v_ = s_.felt(frame, life)
        if v_ is not None:
            r += s_.term(v_)
    return felt, r_face, r


def _same_bits(w, g):
    return type(w) is type(g) and (w.hex() == g.hex() if isinstance(w, float) else w == g)


def test_reward_sources_feel_todays_rule():
    """anatomy 7 (step R3): the diary's reward sources, felt in their order, give bit for bit (the sign of a zero too) the felt event,
    the face's term and the reward `_sense` summed before R3, and leave the held face level where it left it, over 40000 random
    states: faces rising, held, easing, changing sign, past +-6 and fractional; the world's symbol or its rest; world_r off, on,
    negative and tiny; world_mask; an act on the last tick, none, or none recorded yet; the effort on or off, at random costs and
    fatigue"""
    rng = random.Random(0)
    a = LanguageAnatomy(TOK, {}); sil = a.sil
    seen = collections.Counter()
    for i in range(40000):
        cfg = dict(PHYSIOLOGY, world_r=rng.choice([0.0, -0.0, 0.3, -0.25, 1e-17, rng.uniform(-3, 3)]), world_mask=rng.choice([0, 1]),
                   cost_in_reward=rng.choice([0, 1, 0.0, True]), symbol_cost=rng.choice([0.12, 0.0, rng.uniform(0, 1)]),
                   gate_fatigue=rng.choice([10.0, rng.uniform(0.5, 30)]))
        st = dict(sil=sil, level=rng.randint(-6, 6), fatigue=rng.choice([0.0, rng.uniform(0, 30), rng.expovariate(1.0)]),
                  face_now=rng.choice([float(rng.randint(-6, 6)), rng.uniform(-8, 8), -0.5, 0.99, 6.0, -6.0]))
        acted = rng.choice([True, False, None])
        if acted is not None:
            st["_acted_last"] = acted
        u = rng.choice([sil, sil, 5, 42])
        A = types.SimpleNamespace(cfg=cfg, **st); B = types.SimpleNamespace(cfg=dict(cfg), **st)
        want, got = _reward_before_r3(A, u), _reward_by_sources(a, B, u)
        for w, g, what in zip(want, got, ("the felt event", "the face's term", "the reward")):
            assert _same_bits(w, g), f"state {i}: {what} {g!r}, before R3 {w!r} (u {u}, {st}, world_r {cfg['world_r']}, mask {cfg['world_mask']}, cost {cfg['cost_in_reward']})"
        assert A.level == B.level and type(A.level) is type(B.level), f"state {i}: the held level {B.level!r}, before R3 {A.level!r}"
        world_ = u != sil and cfg["world_r"] and not (cfg["world_mask"] and acted)
        cost_ = bool(cfg["cost_in_reward"]) and bool(acted)
        seen["felt"] += want[0] != 0; seen["clipped"] += abs(want[0]) > 2; seen["world"] += bool(world_); seen["effort"] += cost_
        seen["both"] += bool(world_) and cost_; seen["eased"] += want[0] == 0 and A.level != st["level"]
    assert min(seen.values()) >= 1000, dict(seen)
    print("anatomy 7: the sources feel today's rule bit for bit over 40000 states:", ", ".join(f"{k} {v}" for k, v in seen.items()))


def _sense_before_r3(self):
    """`_sense` as it was before step R3 (body/core/senses.py at commit 96f72a9), verbatim: the reference a life lives beside"""
    m = self.m
    u = self.queue.popleft() if self.queue else self.sil
    who = (self.queue_who.popleft() if self.queue_who else "") if u != self.sil else ""
    # THE OFFSET: the world quiet for offset_ticks after its utterance, once per pause, whatever the body is
    # saying meanwhile (with the body's silence required too, a babbling body never let it fire: run 41 held
    # two turn-end memories after six days)
    off = int(self.cfg.get("offset_ticks", 0)); settle_form = str(self.cfg.get("offset_form", "count")) == "settle"
    ps_ = self._pace_mode()                                    # under the sensed pace (2) the pause outlasted (M2, _pace_hear) replaces the count
    if u == self.sil and off > 0 and not self._offset_done and not settle_form and ps_ < 2 and self.ticks - self._last_world >= off:
        self._offset(settled=False); self._offset_done = True
    first_after_pause = (u != self.sil and self._offset_done)  # the first symbol after a perceived pause begins an utterance
    if u != self.sil:
        if ps_:
            self._pace_heard(ps_ >= 2)                          # the gap just ended is a heard event (before the world's last symbol moves)
        self._last_world = self.ticks; self._offset_done = False; self._turn_open = False; self._ear_release_at = None
    # the face: a change is felt; a held face is silence; easing off is not an event
    lvl = max(-6, min(6, int(self.face_now)))
    felt = 0
    if lvl != self.level:
        if abs(lvl) > abs(self.level) or lvl * self.level < 0:
            felt = lvl
        self.level = lvl
    r = float(max(-2, min(2, felt)))                    # the world's reward: the felt face, clipped like a press
    self._ring_r.append(r)
    if self._act_pending:                               # the actor's reliability: the reward of the ticks after each act, on its vote for that act
        H = int(self.cfg.get("actor_horizon", 16))
        for p_ in self._act_pending:
            p_[2] += r
        while self._act_pending and self.ticks - self._act_pending[0][0] >= H:
            t0_, v_, g_ = self._act_pending.popleft(); self._arel_update(v_, g_)
    if u != self.sil:
        wr = float(self.cfg.get("world_r", 0.0))        # the world's words as reward (0 = off)
        if wr and not (int(self.cfg.get("world_mask", 0)) and getattr(self, "_acted_last", False)):
            r += wr                                     # not heard over its own voice (world_mask)
    if self.cfg.get("cost_in_reward") and getattr(self, "_acted_last", False):
        # THE EFFORT IN THE REWARD: the cost of the last act is felt as the next tick's reward, so both critics
        # predict it and the gate reads their error alone. Added to the act's credit outside the critics (the
        # earlier form, with a tonic drive of 0.25 cancelling it) it was never predicted away and, with the drive
        # gone, held every act at a loss; with both gone the gate saturated at 0.98 (runs 69-72).
        r -= float(self.cfg["symbol_cost"]) * (1.0 + (self.fatigue / float(self.cfg["gate_fatigue"])) ** 2)
    if int(self.cfg.get("own_store", 0)):
        thr = float(self.cfg.get("own_store_r", 1.0))
        if float(r) >= thr and not getattr(self, "_own_stored", False) and self.ticks - getattr(self, "_own_store_tick", -10 ** 9) >= int(self.cfg.get("own_store_gap", 40)):
            self._own_stored = True; self._own_store_tick = self.ticks; self._consolidate_own(float(r) * float(self.cfg.get("own_store_gain", 0.3)))
        elif float(r) < 0.5 * thr:
            self._own_stored = False
    return u, who, felt, r, off, settle_form, first_after_pause



def test_a_life_feels_as_before():
    """anatomy 8 (step R3): two lives born alike live the same script with the same faces (rising, held, easing, changing sign, past
    +-2), one feeling through its anatomy's sources (`_sense`), one through `_sense` as it was before R3; under the served constants
    with the effort in the reward and the own store on, and under the physiology with the world's words unmasked and the effort on,
    every tick's felt event and reward (bit for bit), the anticipation ring, the actor's pending rewards and reliability (the served
    constants' actor), the tick's record, the page and the whole life are equal"""
    faces = {5: 2.0, 6: 2.0, 7: 4.0, 9: 2.0, 12: -2.0, 14: 0.0, 20: 6.0, 22: -3.5, 25: 0.0, 33: 1.0, 40: -6.0, 41: 0.0, 50: 3.0,
             52: 0.0, 70: 2.0, 71: 0.0, 88: -2.0, 89: 0.0, 100: 5.0, 101: 0.0}
    lines = ("what do you want?", "I want milk", "do you see the ball?", "yes. the ball is red")
    sc = _served_cfg()
    cases = [("the physiology, the world's words unmasked, the effort", dict(world_r=0.25, world_mask=0, cost_in_reward=1), False)]
    if sc is not None:
        cases.insert(0, ("the served constants, the effort, the own store", dict(sc, cost_in_reward=1, own_store=1, own_store_r=0.2), True))
    for label, cfg, actor_ in cases:
        lives = []
        for sense in (None, _sense_before_r3):
            torch.manual_seed(5); L = _born(TOK, cfg); rec = []
            f = types.MethodType(sense, L) if sense else L._sense

            def rec_sense(f=f, rec=rec, L=L):
                out = f()
                rec.append((out[2], out[3].hex(), L._ring_r[-1].hex(), [list(p) for p in L._act_pending], list(L._arel)))
                return out
            L._sense = rec_sense
            _live(L, lines=lines, faces=faces, ticks=120)
            lives.append((L, rec))
        (A, ra), (B, rb) = lives
        bad = next((t for t in range(max(len(ra), len(rb))) if t >= min(len(ra), len(rb)) or ra[t] != rb[t]), None)
        assert bad is None, f"{label}: tick {bad} felt {ra[bad] if bad < len(ra) else None}, before R3 {rb[bad] if bad < len(rb) else None}"
        assert _differs(A, B) is None, f"{label}: {_differs(A, B)} differs"
        assert A.last == B.last and [e for e in A.page] == [e for e in B.page] and A.level == B.level, f"{label}: the record, the page or the level differs"
        others_ = sum(1 for x in ra if x[1] != x[2]); felt_ = sum(1 for x in ra if x[0]); pend_ = sum(1 for x in ra if x[3])
        assert others_ >= 5 and felt_ >= 8 and (pend_ >= 5 or not actor_), f"{label}: the script reached too little: {others_} ticks with other terms, {felt_} felt, {pend_} with acts pending"
        print(f"anatomy 8 ({label}): 120 ticks felt as before R3, bit for bit; {felt_} felt faces, {others_} ticks with other terms, {pend_} with acts pending")


class _Const(RewardSource):
    """a source felt at one value every tick (None: always silent)"""

    def __init__(self, name, v, clip=None):
        super().__init__(name, (), clip)
        self.v = v

    def felt(self, frame, life):
        return self.v


def test_the_declared_order_is_the_sums():
    """anatomy 9 (step R3): the tick's reward is the anatomy's sources added one at a time in their declared order (1 + 1e-16 + 1e-16
    is 1, where a sum of the later terms taken first gives 1 + 2e-16; 1 + 1e-16 - 1 and 1 - 1 + 1e-16 differ as their orders do); a
    silent source adds nothing and a clipped one its clip; source 0, the judgment, alone is the felt event and the reward the ring and
    the actor's pending acts read"""
    a = LanguageAnatomy(TOK, {}); life = _born(a, {}); judge = a.rewards[0]
    assert life.anatomy is a

    def sense(face, others):
        a.rewards = [judge, *others]; life.level = 0; life.set_face(face)
        out = life._sense()
        return out[2], out[3], life._ring_r[-1]
    cases = [("the later terms one at a time", 1.0, [_Const("x", 1e-16), _Const("y", 1e-16)], 1, 1.0, 1.0),
             ("1 + 1e-16 - 1", 0.0, [_Const("x", 1.0), _Const("y", 1e-16), _Const("z", -1.0)], 0, 0.0, 0.0),
             ("1 - 1 + 1e-16", 0.0, [_Const("x", 1.0), _Const("z", -1.0), _Const("y", 1e-16)], 0, 1e-16, 0.0),
             ("a silent, a clipped, a negative", 4.0, [_Const("q", None), _Const("c", 5.0, clip=2), _Const("n", -0.5)], 4, 3.5, 2.0),
             ("the face alone", -3.0, [], -3, -2.0, -2.0)]
    for what, face, others, felt, r, ring in cases:
        got = sense(face, others)
        assert got[0] == felt and _same_bits(got[1], float(r)) and _same_bits(got[2], float(ring)), f"{what}: felt, reward, ring {got}, want {(felt, r, ring)}"
    life._act_pending.append([life.ticks, 0.5, 0.0])
    got = sense(2.0, [_Const("x", 0.75)])
    assert got == (2, 2.75, 2.0) and life._act_pending[-1][2] == 2.0, f"the actor's pending act read {life._act_pending[-1][2]}, the tick {got}"
    print("anatomy 9: the reward is the sources in their declared order, one at a time; a silent source adds nothing; the judgment's",
          "term alone reaches the ring and the actor's pending acts")


# ---------------- step R4: the channels ----------------

def _inputs_before_r4(m, xs, xos, faces, bundles, reads):
    """`Organs.inputs` as it was before step R4 (body/model.py at commit cba7179), verbatim: the reference the ordered sum is held to"""
    u = m.E(xs) + m.face_in(faces) + m.bundle_in(bundles.reshape(*bundles.shape[:-2], -1))   # [T, nb, d] or [B, T, nb, d]
    if xos is not None and m.sil_id is not None:
        # its own sound attenuated (corollary discharge, own_gain 0.5) and superposed on the tick's
        # position. Heard at full weight with the lessons hearing the world only (run 22), the cortex
        # alone was worse at day 6 ("big big big"); the loops that motivated that change were an
        # instrument's fault (a store holding only the cue), not the cortex's.
        own = (xos != m.sil_id).to(u.dtype).unsqueeze(-1)
        u = u + float(m.own_gain) * own * m.E(xos)
    return m.in_ln(u)


class _Level(torch.nn.Module):
    """an organ whose code is one level in every dimension, whatever it is shown (the float order's probe)"""

    def __init__(self, d, v):
        super().__init__()
        self.d, self.v = d, v

    def forward(self, obs):
        return torch.full((*obs.shape[:-1], self.d), self.v)


def test_the_input_is_the_channels_in_order():
    """anatomy 10 (step R4): the cortex's input is the anatomy's channel codes summed one at a time in the declared order, the ladder's
    bundle and its own sound joining after the first inner_at channels. For the diary it is bit for bit the sum before R4 (verbatim
    in this test), its gradients too, on random windows (one and a batch, its own sound or none) with every term alive, and on a
    lived window read per channel. On toy organs whose codes are levels (1 and 4e-8 twice, float32): the channels in the order
    declared give 1, in the reverse order 1 + 2^-23; the bundle (a level of 1) joining after no channel gives 1, after all of them
    1 + 2^-23: the declared order is the float order"""
    life = _born(TOK, _served_cfg() or {}); m = life.m; a = life.anatomy
    torch.manual_seed(11)
    with torch.no_grad():                               # every term away from its birth (face_in is born at zero)
        for p in list(m.face_in.parameters()) + list(m.bundle_in.parameters()) + list(m.in_ln.parameters()):
            p.add_(torch.randn_like(p) * 0.5)
    nb, d, V = len(m.clocks), m.d, m.vocab
    g = torch.Generator().manual_seed(3); n = 0
    for shape in ((17,), (5, 9), (1,)):
        xs = torch.randint(0, V, shape, generator=g); xs[..., ::3] = life.sil
        xos = torch.randint(0, V, shape, generator=g); xos[..., 1::2] = life.sil
        faces = torch.randn(*shape, 2, generator=g); bundles = torch.randn(*shape, nb, d, generator=g); reads = torch.randn(*shape, d, generator=g)
        for own in (xos, None):
            grads = []
            for f in (lambda: _inputs_before_r4(m, xs, own, faces, bundles, reads), lambda: m.inputs(a, {"ear": xs, "face": faces}, own, bundles)):
                m.zero_grad(set_to_none=True)
                u = f(); (u.pow(3).sum()).backward()
                grads.append((u.detach(), {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}))
            (u0, g0), (u1, g1) = grads
            assert torch.equal(u0, u1), f"shape {shape}, own {own is not None}: the input differs from the sum before R4"
            assert sorted(g0) == sorted(g1) and all(torch.equal(g0[k], g1[k]) for k in g0), f"shape {shape}: a gradient differs"
            n += 1
    m.zero_grad(set_to_none=True)
    _live(life, lines=("what do you want?", "I want milk", "do you see the ball?"), ticks=90)
    win = list(life.win); obs, whos, bundles, reads = life._window_tensors()
    xs_old = torch.tensor([w["x"] for w in win]); faces_old = torch.stack([w["face"] for w in win])
    assert list(obs) == ["ear", "face"] and torch.equal(obs["ear"], xs_old) and torch.equal(obs["face"], faces_old)
    with torch.no_grad():
        assert torch.equal(m.inputs(a, obs, whos, bundles), _inputs_before_r4(m, xs_old, whos, faces_old, bundles, reads)), "the lived window"
    # the float order, on toy organs whose codes are levels
    torch.manual_seed(0)
    om = Organs(4, d=8, layers=1, heads=1, window=8)
    with torch.no_grad():
        om.E.weight.zero_(); om.bundle_in.weight.zero_(); om.bundle_in.bias.zero_()
    om.in_ln = torch.nn.Identity()
    om.big = _Level(8, 1.0); om.eps1 = _Level(8, 4e-8); om.eps2 = _Level(8, 4e-8)

    def toy(order, inner_at):
        chans = [Channel("w", "symbol", 4, organ="E", forecast=True, rest_id=0)] + [Channel(n_, "vector", 1, organ=n_) for n_ in order]
        return Anatomy(chans, [VoiceEffector("v", [4], rest_id=0)], [RewardSource("r")], inner_at=inner_at).check()
    x = torch.zeros(3, dtype=torch.long); o = {k_: torch.zeros(3, 1) for k_ in ("big", "eps1", "eps2")}; o["w"] = x
    bund = torch.zeros(3, len(om.clocks), 8)
    one_up = float(torch.tensor(1.0) + torch.tensor(2.0 ** -23))
    fwd, rev = om.inputs(toy(["big", "eps1", "eps2"], 4), o, None, bund), om.inputs(toy(["eps1", "eps2", "big"], 4), o, None, bund)
    assert bool((fwd == 1.0).all()) and bool((rev == one_up).all()), (fwd[0, 0].item(), rev[0, 0].item())
    with torch.no_grad():
        om.bundle_in.bias.fill_(1.0)                    # the bundle a level of 1
    first, last = om.inputs(toy(["eps1", "eps2"], 0), o, None, bund), om.inputs(toy(["eps1", "eps2"], 3), o, None, bund)
    assert bool((first == 1.0).all()) and bool((last == one_up).all()), (first[0, 0].item(), last[0, 0].item())
    print(f"anatomy 10: the diary's input is the sum before R4 bit for bit ({n} random windows, gradients too, and a lived one); the declared",
          "order is the float order (1 against 1 + 2^-23, the channels and the bundle's place)")


class _Renamed(LanguageAnatomy):
    """the diary's anatomy with its channels' window fields renamed: the same body, its window keyed otherwise"""

    def __init__(self, tok, cfg=None):
        super().__init__(tok, cfg)
        self.channels[0].field = "heard"; self.channels[1].field = "seen"


def test_the_window_holds_each_channel_under_its_field():
    """anatomy 11 (step R4): each window position holds every channel's observation under the channel's declared field (the diary's
    "x" and "face", the keys the window always had) beside the core's own keys, and the window's tensors are per channel, equal to
    the window read as before R4; a dream's other channels are quiet, as its face always was (zeros). Two lives born alike, one with
    the fields renamed, live a day, a night (on its utterances, batched) and a morning of script and faces as the same life: every
    reader of the window goes through the declared field"""
    cfg = dict(wake_ticks=80, wake_every=8, gate_every=8, night_rounds=1, night_starts=4, night_batch=4, rem_dreams=2, rem_steps=2,
               dream_source="utterances", night_load=0.0, write_floor=1e-30, offset_ticks=8)
    lines = ("what do you want?", "I want milk", "do you see the ball?", "yes. the ball is red")
    faces = {20: 2.0, 21: 0.0, 50: -2.0, 51: 0.0, 100: 4.0, 101: 0.0}
    torch.manual_seed(5); A = _born(TOK, cfg)
    torch.manual_seed(5); B = _born(_Renamed(TOK, cfg), cfg)
    _live(A, lines=lines, faces=faces, ticks=120); _live(B, lines=lines, faces=faces, ticks=120)
    assert A.nights == B.nights == 1 and not (A.last_night or {}).get("error") and not (B.last_night or {}).get("error"), (A.last_night or {}).get("error")
    assert _differs(A, B) is None and A.last == B.last and A.page == B.page and A.utts == B.utts, f"renamed fields: {_differs(A, B)} differs"
    kA = set().union(*(w.keys() for w in A.win)); kB = set().union(*(w.keys() for w in B.win))
    core = {"xo", "bundle", "read", "r"}
    assert core | {"x", "face"} <= kA <= core | {"x", "face", "end"}, kA
    assert core | {"heard", "seen"} <= kB <= core | {"heard", "seen", "end"}, kB
    ends = sum(1 for w in A.win if w.get("end"))
    for L in (A, B):
        win = list(L.win); obs, whos, bundles, reads = L._window_tensors()
        f0, f1 = L.anatomy.channels[0].field, L.anatomy.channels[1].field
        assert list(obs) == ["ear", "face"] and obs["ear"].dtype == torch.long and obs["face"].shape == (len(win), 2)
        assert torch.equal(obs["ear"], torch.tensor([w[f0] for w in win])) and torch.equal(obs["face"], torch.stack([w[f1] for w in win]))
        assert torch.equal(whos, torch.tensor([w["xo"] for w in win]))
    for shape in ((7,), (3, 5)):
        xs = torch.randint(0, A.m.vocab, shape)
        d_ = A._dream_obs(xs)
        assert list(d_) == ["ear", "face"] and d_["ear"] is xs and d_["face"].dtype == torch.float32
        assert torch.equal(d_["face"], torch.zeros(*shape, 2)), shape
    print(f"anatomy 11: the window holds each channel under its field ({ends} offset marks); renamed, the same life through a day, a night",
          "and a morning; the window's tensors per channel as before; a dream's face quiet")


class _Touch(Channel):
    """a toy sense for the tests: two numbers that move with the tick, encoded by the face's own map (so its save needs no organ the
    diary lacks)"""

    def observe(self, life, x, who, still=False):
        t = float(life.ticks)
        return torch.tensor([math.sin(t / 7.0), math.cos(t / 5.0)], device=life.dev)


class _Probe(torch.nn.Module):
    """a later channel's own organ (a linear map) that records each read: whether the gradient was on, and over how many positions"""

    def __init__(self, n, d):
        super().__init__()
        self.lin = torch.nn.Linear(n, d); self.calls = []

    def forward(self, obs):
        self.calls.append((torch.is_grad_enabled(), int(obs.shape[0])))
        return self.lin(obs)


class _TouchAnatomy(LanguageAnatomy):
    """the diary's anatomy and a later channel, after its own sound in the input sum, with a forecast head of its own"""

    def __init__(self, tok, cfg=None):
        super().__init__(tok, cfg)
        self.channels.append(_Touch("touch", "vector", 2, organ="face_in", forecast=True))


def test_a_later_channel():
    """anatomy 12 (step R4): the diary's organs are built as before (no head beyond latent_pred, the same draws); a later channel's
    forecast head is built after every other organ (the rest bit for bit the diary's at the same seed), born unsure; a life with such a
    channel holds it in its window, adds its code last (after its own sound), teaches its head by the waking lesson (the target, the
    next position's code, read with the gradient off), dreams it quiet, sleeps a night and keeps the head through a save and a load;
    organs without a declared head, with an undeclared head, or without a channel's organ are refused"""
    V = LanguageAnatomy(TOK, {}).vocab
    torch.manual_seed(9); o1 = Organs(V, d=64, layers=2, heads=2, window=32); r1 = torch.get_rng_state()
    torch.manual_seed(9); o2 = Organs(V, d=64, layers=2, heads=2, window=32, channels=LanguageAnatomy(TOK, {}).channels); r2 = torch.get_rng_state()
    torch.manual_seed(9); o3 = Organs(V, d=64, layers=2, heads=2, window=32, channels=_TouchAnatomy(TOK, {}).channels)
    s1, s2, s3 = o1.state_dict(), o2.state_dict(), o3.state_dict()
    assert torch.equal(r1, r2) and list(s1) == list(s2) and all(torch.equal(s1[k], s2[k]) for k in s1), "the diary's organs are built otherwise"
    assert "chan_pred" not in vars(o2)["_modules"] and not hasattr(o2, "chan_pred")
    assert list(s3) == list(s1) + ["chan_pred.touch.weight", "chan_pred.touch.bias"] and all(torch.equal(s1[k], s3[k]) for k in s1), \
        "the later channel's head was not built last"
    assert float(o3.chan_pred["touch"].bias.detach().abs().max()) == 0.0 and float(o3.chan_pred["touch"].weight.detach().std()) < 1e-3
    cfg = dict(wake_ticks=80, wake_every=8, gate_every=8, night_rounds=1, night_starts=4, night_batch=4, rem_dreams=2, rem_steps=2,
               write_floor=1e-30)
    a = _TouchAnatomy(TOK, cfg)
    torch.manual_seed(5); L = _born(a, cfg); m = L.m
    assert L.anatomy is a and m.head(a, 0) is m.latent_pred and m.head(a, 2) is m.chan_pred["touch"]
    h0 = m.chan_pred["touch"].weight.detach().clone()
    _live(L, lines=("go up", "we go up", "up we go"), faces={10: 2.0, 11: 0.0, 40: -2.0, 41: 0.0}, ticks=120)
    assert L.nights == 1 and not (L.last_night or {}).get("error"), (L.last_night or {}).get("error")
    assert not torch.equal(h0, m.chan_pred["touch"].weight), "the waking lesson did not teach the later channel's head"
    win = list(L.win); obs, whos, bundles, reads = L._window_tensors()
    assert all("touch" in w for w in win) and list(obs) == ["ear", "face", "touch"] and obs["touch"].shape == (len(win), 2)
    diary = LanguageAnatomy(TOK, L.cfg); ln = m.in_ln; m.in_ln = torch.nn.Identity()
    try:
        with torch.no_grad():
            u_t, u_d, code = m.inputs(a, obs, whos, bundles), m.inputs(diary, obs, whos, bundles), m.face_in(obs["touch"])
    finally:
        m.in_ln = ln
    assert torch.equal(u_t, u_d + code) and not torch.equal(u_t, u_d), "the later channel's code is not the sum's last term"
    d_ = L._dream_obs(torch.tensor([3, 4, 5]))
    assert list(d_) == ["ear", "face", "touch"] and torch.equal(d_["touch"], torch.zeros(3, 2))
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    try:
        L.save(path)
        C = Life.load(path, _TouchAnatomy(TOK, L.cfg), save_path=None)
    finally:
        os.remove(path)
    assert torch.equal(C.m.chan_pred["touch"].weight, m.chan_pred["touch"].weight) and C.ticks == L.ticks
    # the head's target is the next position's code held still: a later channel with an organ of its own, which records whether it is
    # read with the gradient on; in one waking lesson it is read twice, the window's input with it and the target without
    pa = _TouchAnatomy(TOK, cfg); pa.channels[2].organ = "touch_in"
    torch.manual_seed(5); po = Organs(pa.vocab, d=64, layers=2, heads=2, window=32, channels=pa.channels)
    po.touch_in = _Probe(2, 64)
    P = Life(po, pa, cfg=cfg)
    _live(P, lines=("go up",), faces={}, ticks=24)
    po.touch_in.calls.clear(); out = P._wake_lesson()
    T_ = min(len(P.win), int(P.cfg["wake_window"]))
    assert out and "latent_cos" in out and po.touch_in.calls == [(True, T_), (False, T_ - 1)], (out, po.touch_in.calls, T_)
    refused = []
    bad_organ = _TouchAnatomy(TOK, {}); bad_organ.channels[2].organ = "tongue"
    for what, call in (("organs without the declared head", lambda: Life(o2, _TouchAnatomy(TOK, {}))),
                       ("organs with a head the anatomy does not declare", lambda: Life(o3, TOK)),
                       ("a channel whose organ the organs lack", lambda: Life(o3, bad_organ))):
        try:
            call()
        except ValueError:
            refused.append(what); continue
        raise AssertionError(f"the life took {what}")
    print("anatomy 12: the diary's organs built as before; a later channel's head built last, taught awake, kept through a save;",
          "its code the sum's last term, quiet in dreams; refused:", "; ".join(refused))


def test_every_call_site_passes_the_channels():
    """anatomy 13 (step R4): the eleven places the cortex's input is made (the stream now, the waking lesson, imagination for choice,
    a dream one at a time and in lockstep, the night's lesson batched and one at a time, REM imagined and rolled out, the gauge batched
    and one at a time) are all reached by a life with a later channel, and each passes the life's anatomy and all its channels"""
    import body.model as BM
    cfg = dict(wake_ticks=100000, wake_every=8, gate_every=8, night_rounds=1, night_starts=6, night_batch=4, rem_dreams=2, rem_steps=2,
               rem_rounds=1, write_floor=1e-30, fast_rls=1, fast_input="striatum", face_form="foresee", rem_form="forecast", rem_temp=1.0)
    torch.manual_seed(5); L = _born(_TouchAnatomy(TOK, cfg), cfg)
    names = [c.name for c in L.anatomy.channels]
    seen = collections.Counter(); bad = []
    orig = BM.Organs.inputs

    def spy(self, anatomy, obs, xos, bundles):
        where = sys._getframe(1).f_code.co_name + ("[batch]" if bundles.dim() == 4 and sys._getframe(1).f_code.co_name == "night" else "")
        seen[where] += 1
        if anatomy is not L.anatomy or list(obs) != names or any(obs[k].shape[:len(xos.shape)] != xos.shape for k in names):
            bad.append(where)
        return orig(self, anatomy, obs, xos, bundles)
    BM.Organs.inputs = spy
    try:
        _live(L, lines=("what do you want?", "I want milk", "do you see the ball?"), faces={20: 2.0, 21: 0.0}, ticks=90)
        L._imagine_value(TOK.token_to_id("a"), 3)
        dreams = L.dreams(6)
        assert dreams and max(len(d_) for d_ in dreams) >= 4, dreams
        L.gauge(dreams)                                   # batched (night_batch 4): _gauge_batched, _dream_batch
        L.cfg["night_batch"] = 0; L.gauge(dreams)         # one at a time: gauge, _dream_inputs
        L._frel_corr = 0.5                                # the face organ's proven share, so imagination counts
        ids = max(dreams, key=len)
        L._rem_imagine(ids); L._rem_rollout(ids, 0.0)
        L.cfg["night_batch"] = 4; r1 = L.night()          # the night batched, REM rolled out
        L.cfg["night_batch"] = 0; r2 = L.night()          # the night one at a time
    finally:
        BM.Organs.inputs = orig
    assert not r1.get("error") and not r2.get("error"), (r1.get("error"), r2.get("error"))
    want = {"_stream_now", "_wake_lesson", "_imagine_value", "_dream_inputs", "_dream_batch", "night[batch]", "night", "_rem_imagine",
            "_rem_rollout", "_gauge_batched", "gauge"}
    assert set(seen) == want, f"reached {sorted(seen)}, want {sorted(want)}"
    assert not bad, f"these passed another anatomy or not all its channels: {sorted(set(bad))}"
    print("anatomy 13: all eleven call sites reached with a later channel, each passing the anatomy's channels:",
          ", ".join(f"{k} {v}" for k, v in sorted(seen.items())))


def _window_tensors_before_r4(self, win=None):
    """`_window_tensors` as it was before step R4 (body/core/cortex.py at commit cba7179), verbatim"""
    win = list(self.win if win is None else win)
    xs = torch.tensor([w["x"] for w in win], device=self.dev)
    whos = torch.tensor([w["xo"] for w in win], device=self.dev)   # its own symbols, one per tick
    faces = torch.stack([w["face"] for w in win])
    bundles = torch.stack([w["bundle"] for w in win])
    reads = torch.stack([w["read"] for w in win])
    return xs, whos, faces, bundles, reads


def _imagine_value_before_r4(self, first, h):
    """`_imagine_value` as it was before step R4 (body/core/mouth.py at commit cba7179), verbatim, its window read and its input
    summed by the verbatim copies above"""
    m = self.m
    with torch.no_grad():
        win = list(self.win); line = m.stri_line.clone(); sym = int(first); zero_read = torch.zeros(m.d, device=self.dev)
        face = torch.tensor([self.face_now / 6.0, 0.0], device=self.dev)
        said = []
        for step in range(int(h)):
            said.append(sym)
            win.append({"x": self.sil, "xo": sym, "face": face, "bundle": self.bands, "read": zero_read, "r": 0.0})
            if len(win) > m.window:
                win = win[-m.window:]
            if step == int(h) - 1:
                break
            xs, whos, faces, bundles, reads = _window_tensors_before_r4(self, win)
            C = m.stream(_inputs_before_r4(m, xs, whos, faces, bundles, reads))[-1]
            lg = m.readout(m.forecast(C, zero_read)); lg[self.sil] = float("-inf"); lg[self.bans] = float("-inf")
            sym = int(lg.argmax())
            if int(self.cfg.get("plan_boundary", 1)) and sym == self.space_id:   # the word as the unit only under the old rule
                said.append(sym); break                       # the word ends: value the line here
        width = 2 * m.vocab + 3
        for sy in said:
            line = torch.roll(line, 1); line[0] = m.vocab + int(sy)      # its own symbols, kind 1
        z = m.stri_b.clone()
        for p_ in range(line.numel()):
            e = int(line[p_])
            if e >= 0:
                z += m.stri_W[p_ * width + e]
        z = torch.relu(z)
        if getattr(m, "stri_wm", 0):
            z = torch.cat([z, m.wm_slot * m.wm_on])
        return float(m.fast_value(z))


def test_imagination_as_before():
    """anatomy 14 (step R4): imagination for choice (the planner's `_imagine_value`) imagines its positions with the world quiet and
    the face held where it is (its change 0), every channel under its field: bit for bit the value it gave before R4 (verbatim in this
    test), for every first symbol, on a lived body whose face has just changed (its face map moved off its birth), under both word
    rules; and its imagined positions read, per channel, the world's rest and [face/6, 0]"""
    cfg = dict(_served_cfg() or {}, fast_rls=1, fast_input="striatum", wake_ticks=100000)
    torch.manual_seed(5); L = _born(TOK, cfg)
    _live(L, lines=("what do you want?", "I want milk", "do you see the ball?"), faces={20: 2.0, 21: 0.0, 50: -2.0}, ticks=70)
    torch.manual_seed(1)
    with torch.no_grad():                               # the face's map alive (its change's column too), and a forecast that reads the stream
        L.m.face_in.weight.add_(torch.randn_like(L.m.face_in.weight))
        L.m.latent_pred.weight.add_(torch.randn_like(L.m.latent_pred.weight) * 0.5)
    L.set_face(4.0)                                     # the face changed since the tick before: its change is not 0
    assert L.face_now != L.face_prev
    n = 0; moved = 0; face = L.anatomy.channels[1]
    for pb in (0, 1):
        L.cfg["plan_boundary"] = pb
        for c in [i for i in range(L.m.vocab) if i != L.sil and i not in L.bans][::7]:
            w_, g_ = _imagine_value_before_r4(L, c, 4), L._imagine_value(c, 4)
            assert w_.hex() == g_.hex(), f"plan_boundary {pb}, first {c}: {g_!r}, before R4 {w_!r}"
            face.observe = lambda life, x, who, still=False: FaceChannel.observe(face, life, x, who, False)   # the probe's reach: the face moving
            try:
                moved += L._imagine_value(c, 4) != g_
            finally:
                del face.observe
            n += 1
    assert moved >= 3, f"the probe cannot tell a held face from a moving one ({moved} of {n})"
    held = {c_.field: c_.observe(L, L.sil, 1, still=True) for c_ in L.anatomy.channels}
    assert set(held) == {"x", "face"} and held["x"] == L.sil and torch.equal(held["face"], torch.tensor([4.0 / 6.0, 0.0]))
    print(f"anatomy 14: imagination for choice as before R4, bit for bit ({n} first symbols, both word rules, the face just changed;",
          f"imagined with the face moving, {moved} of them would differ)")


# ---------------- step R5: the effectors ----------------

class _Arm(LanguageAnatomy):
    """the diary's anatomy and two later effectors: an arm of two joints of five settings (its rest the middle of each, act 12), and a
    grip of three settings (its rest 1) whose gate reads a second input of its own (a touch that moves with the tick)"""

    def __init__(self, tok, cfg=None):
        super().__init__(tok, cfg)
        self.effectors += [Effector("arm", [5, 5], rest_id=12, effort=0.05), _Grip("grip", [3], rest_id=1, n_in=2, effort=0.03)]


class _Grip(Effector):
    def gate_inputs(self, frame, life, state):
        return [1.0 if state["acted_last"] else 0.0, math.sin(frame.tick / 5.0)]


class _Tap(LanguageAnatomy):
    """the diary's anatomy and a later effector of one joint of four settings, its rest 1 and its act 3 reserved"""

    def __init__(self, tok, cfg=None):
        super().__init__(tok, cfg)
        self.effectors += [Effector("tap", [4], rest_id=1, reserved=[3], effort=0.02)]


def test_the_voice_is_effector_0():
    """anatomy 15 (step R5): the diary's effector 0 is the voice, naming the organs it always had (the lexicon E it shares with the
    ear, mouth_gate, actor, its act the window's xo); its gate's own inputs are today's ear (read through the effector) and its cost
    symbol_cost. The check refuses an anatomy whose effector 0 is not the voice and a later effector that names other organs than its
    own, shares a channel's name or a window field, has no rest, reserves acts of a factored alphabet or declares negative inputs. The
    diary's organs, striatum, life, window and dreams gain nothing from the effectors' wiring"""
    for label, cfg in (("the physiology", dict(gate_ear=1, gate_ear_decay=0.9, gate_ear_gain=1.5)), ("the served constants", _served_cfg() or {})):
        L = _born(TOK, cfg); v = L.anatomy.effectors[0]
        assert type(v) is VoiceEffector and (v.organ, v.gate, v.actor, v.field) == ("E", "mouth_gate", "actor", "xo")
        assert v.cost(0, None, L) == float(L.cfg["symbol_cost"])
        _live(L, ticks=40)
        u = TOK.token_to_id("w"); fr = Frame(L.ticks, {"ear": u}, L.face_now)
        keep = (getattr(L, "_ear_trace", None), getattr(L, "_ear_now", None), getattr(L, "_ear_release_at", None))
        got = v.gate_inputs(fr, L)
        L._ear_trace, L._ear_now, L._ear_release_at = keep
        want = L._voice_ear(u)
        assert got is not None and torch.equal(got, want) and got.shape == (2,), (label, got, want)
    L0 = _born(TOK, {}); assert L0.anatomy.effectors[0].gate_inputs(Frame(0, {"ear": 5}, 0.0), L0) is None   # no ear, no inputs
    # the check
    def ok():
        return _Arm(TOK, {})
    ok().check()
    bad = []
    a = ok(); a.effectors = a.effectors[1:] + a.effectors[:1]; bad.append(("the voice not first", a))
    a = ok(); a.effectors[0] = Effector("voice", [a.vocab], rest_id=a.sil); bad.append(("the voice a plain effector", a))
    a = ok(); a.effectors[0].factors = [a.vocab - 1]; bad.append(("the voice not over the words", a))
    a = ok(); a.effectors[1].gate = "mouth_gate"; bad.append(("a later effector naming the voice's gate", a))
    a = ok(); a.effectors[1].organ = "E"; bad.append(("a later effector naming the lexicon", a))
    a = ok(); a.effectors[1].name = "face"; a.effectors[1].organ, a.effectors[1].gate, a.effectors[1].actor = "acts.face", "gates.face", "actors.face"; bad.append(("an effector with a channel's name", a))
    a = ok(); a.effectors[1].field = "x"; bad.append(("an effector's field a channel's", a))
    a = ok(); a.effectors[1].field = "xo"; bad.append(("an effector's field the core's", a))
    a = ok(); a.effectors[2].field = "arm"; bad.append(("two effectors' fields alike", a))
    a = ok(); a.effectors[1].rest_id = None; bad.append(("a later effector without a rest", a))
    a = ok(); a.effectors[1].reserved = [3]; bad.append(("a factored alphabet's reserved act", a))
    a = ok(); a.effectors[2].n_in = -1; bad.append(("negative inputs", a))
    a = ok(); a.effectors[1].rest_id = 25; bad.append(("a rest outside the alphabet", a))
    for label, a in bad:
        try:
            a.check()
        except ValueError:
            continue
        raise AssertionError(f"the check let pass {label}")
    # the diary's organs, striatum and life gain nothing
    V = LanguageAnatomy(TOK, {}).vocab; eff = LanguageAnatomy(TOK, {}).effectors
    torch.manual_seed(9); o1 = Organs(V, d=64, layers=2, heads=2, window=32); r1 = torch.get_rng_state()
    torch.manual_seed(9); o2 = Organs(V, d=64, layers=2, heads=2, window=32, effectors=eff, born_seed=3); r2 = torch.get_rng_state()
    s1, s2 = o1.state_dict(), o2.state_dict()
    assert torch.equal(r1, r2) and list(s1) == list(s2) and all(torch.equal(s1[k], s2[k]) for k in s1), "the diary's organs are built otherwise"
    assert not any(hasattr(o2, n_) for n_ in ("acts", "gates", "actors", "stri_mline", "stri_blocks"))
    torch.manual_seed(4); o1.striatum_init(8, 64, seed=2, wm=1); r1 = torch.get_rng_state()
    torch.manual_seed(4); o2.striatum_init(8, 64, seed=2, wm=1, effectors=eff); r2 = torch.get_rng_state()
    s1, s2 = o1.state_dict(), o2.state_dict()
    assert torch.equal(r1, r2) and list(s1) == list(s2) and all(torch.equal(s1[k], s2[k]) for k in s1), "the diary's striatum is born otherwise"
    assert not hasattr(o2, "stri_blocks") and vars(o1).keys() == vars(o2).keys()
    L = _born(TOK, dict(_served_cfg() or {}, wake_ticks=100000)); _live(L, ticks=60)
    assert not any(hasattr(L, n_) for n_ in ("motor", "opt_motor")), "the diary's life gained the later effectors' state"
    assert all(set(w) <= {"x", "face", "xo", "bundle", "read", "r", "end"} for w in L.win) and list(L._dream_obs(torch.tensor([3, 4]))) == ["ear", "face"]
    assert all(len(r_) == 7 for r_ in L.gate_buf) and "acts" not in L.last and "effectors" not in L.insides()
    print(f"anatomy 15: the voice is effector 0 with its organs' names, its ear and its cost; the check refuses {len(bad)} faulty effectors;",
          "the diary's organs, striatum, life, window, dreams, gate rows and record gain nothing")


def _gate_lesson_before_r5(self):
    """`_gate_lesson` as it was before step R5 (body/core/mouth.py at commit e948e7b), verbatim"""
    buf = list(self.gate_buf)
    K = int(self.cfg["elig_ticks"]); dec = float(self.cfg["elig_decay"])
    n = len(buf) - K
    if n < 4:
        return
    cost = float(self.cfg["symbol_cost"]); f0 = float(self.cfg["gate_fatigue"]); w_int = float(self.cfg["gate_int"])
    tonic = float(self.cfg["gate_tonic"]); vig = float(self.cfg["gate_vigor"])
    feats = torch.stack([b[0] for b in buf[:n]]).to(self.dev)
    acts = torch.tensor([1.0 if b[1] else 0.0 for b in buf[:n]])
    G = torch.zeros(n)
    for t in range(n):
        g = sum((dec ** k) * float(buf[t + k][2]) for k in range(K))     # the dopamine that followed
        if buf[t][1]:
            # acting pays a tonic drive (babble is its own reward, not contingent on confidence) plus
            # the belief it had in its choice (habituating), minus an effort cost convex in fatigue
            # (linear, 0.59 at fatigue's ceiling never beat a confident recitation's drive of 0.7:
            # run 19, gate 0.97 all day, fatigue pinned at 40; convex, the mouth speaks in bouts).
            # With the effort in the reward (cost_in_reward) the cost is the critics' to predict, not the act's
            drive_t = tonic + (float(self.cfg.get("gate_tonic_rate", 0.0)) * float(buf[t][5]) if len(buf[t]) > 5 else 0.0)   # THE DRIVE FOLLOWS THE REWARD RATE
            g += drive_t + w_int * float(buf[t][3]) - (0.0 if self.cfg.get("cost_in_reward") else cost * (1.0 + (float(buf[t][4]) / f0) ** 2))
        G[t] = g
    # the credit is taken against a running baseline (dopamine is an error, not a value)
    base = getattr(self, "_g_base", None)
    if base is None:
        base = float(G.mean())
    A = G - base
    self._g_base = float(self.cfg["gate_baseline"]) * base + (1.0 - float(self.cfg["gate_baseline"])) * float(G.mean())
    if float(A.abs().max()) < 1e-4:
        return
    self.m.mouth_gate.train()
    z = self.m.mouth_gate(feats).squeeze(-1)
    fl = float(self.cfg["gate_floor"])
    # the probability the gate actually acted with (stress divisor and all), carried in the buffer (review 2026-09-06: recomputed
    # here without the divisor, (act - p) was biased with stress); older samples without it fall back to the recomputation
    p = torch.tensor([float(b[6]) if len(b) > 6 else float("nan") for b in buf[:n]], device=self.dev)
    p = torch.where(torch.isnan(p), (fl + (1.0 - fl) * torch.sigmoid(z)).detach(), p)
    # THE THREE-FACTOR RULE: credit x (action - p) has expectation cov(credit, acting), what a policy
    # must learn (Go for acts that paid, NoGo for acts that cost); plus vigor: the average credit
    # itself, tonic dopamine setting the rate of acting whatever it did
    elig = (acts.to(self.dev) - p) + vig
    loss = -(A.to(self.dev) * elig * z).mean()
    self.opt_gate.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(self.m.mouth_gate.parameters(), 1.0)
    self.opt_gate.step(); self.m.mouth_gate.eval()
    self._gate_last = {"n": n, "credit_mean": round(float(G.mean()), 4), "baseline": round(float(base), 4),
                       "acted": round(float(sum(1 for b in buf[:n] if b[1]) / n), 3), "tick": self.ticks}
    for _ in range(min(len(self.gate_buf), n)):                 # the samples the lesson consumed (review: popping gate_every left a third to be learned twice)
        self.gate_buf.popleft()


def _lesson_fixed(self, buf_, gate_, opt_, base, own, s, cost_of):
    """the lesson before R5 with only the two fixes written in, on any gate: the eligibility's act the recorded draw b[7] (own, defect 4)
    and the credit from buf[t + s + k] (s 1, defect 8), the effort cost_of(row); returns (the new baseline, the report)"""
    buf = list(buf_)
    K = int(self.cfg["elig_ticks"]); dec = float(self.cfg["elig_decay"])
    n = len(buf) - K - s
    if n < 4:
        return base, None
    f0 = float(self.cfg["gate_fatigue"]); w_int = float(self.cfg["gate_int"])
    tonic = float(self.cfg["gate_tonic"]); vig = float(self.cfg["gate_vigor"])
    feats = torch.stack([b[0] for b in buf[:n]]).to(self.dev)
    acts = torch.tensor([1.0 if (b[7] if own else b[1]) else 0.0 for b in buf[:n]])
    G = torch.zeros(n)
    for t in range(n):
        g = sum((dec ** k) * float(buf[t + s + k][2]) for k in range(K))
        if buf[t][1]:
            drive_t = tonic + (float(self.cfg.get("gate_tonic_rate", 0.0)) * float(buf[t][5]) if len(buf[t]) > 5 else 0.0)
            g += drive_t + w_int * float(buf[t][3]) - (0.0 if self.cfg.get("cost_in_reward") else cost_of(buf[t]) * (1.0 + (float(buf[t][4]) / f0) ** 2))
        G[t] = g
    if base is None:
        base = float(G.mean())
    A = G - base
    new_base = float(self.cfg["gate_baseline"]) * base + (1.0 - float(self.cfg["gate_baseline"])) * float(G.mean())
    if float(A.abs().max()) < 1e-4:
        return new_base, None
    gate_.train()
    z = gate_(feats).squeeze(-1)
    fl = float(self.cfg["gate_floor"])
    p = torch.tensor([float(b[6]) if len(b) > 6 else float("nan") for b in buf[:n]], device=self.dev)
    p = torch.where(torch.isnan(p), (fl + (1.0 - fl) * torch.sigmoid(z)).detach(), p)
    elig = (acts.to(self.dev) - p) + vig
    loss = -(A.to(self.dev) * elig * z).mean()
    opt_.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(gate_.parameters(), 1.0)
    opt_.step(); gate_.eval()
    last = {"n": n, "credit_mean": round(float(G.mean()), 4), "baseline": round(float(base), 4),
            "acted": round(float(sum(1 for b in buf[:n] if b[1]) / n), 3), "tick": self.ticks}
    for _ in range(min(len(buf_), n)):
        buf_.popleft()
    return new_base, last


def _rows(g, n, width, kind):
    """n random rows of a gate's buffer: kind 7 today's, 8 with the gate's draw, 9 a later effector's (draw and cost), "old" a mix of the
    older rows (no p, no reward trace); acted implies drawn; a draw that ended in a rest is acted 0; some p at 1 (a program's letters)"""
    out = []
    for t in range(n):
        drew = bool(torch.rand(1, generator=g) < 0.6); acted = drew and bool(torch.rand(1, generator=g) < 0.8)
        p = float(torch.rand(1, generator=g)) if t % 7 else 1.0
        row = [torch.randn(width, generator=g), acted, float(torch.randn(1, generator=g)), float(torch.rand(1, generator=g)) * 0.3,
               float(torch.rand(1, generator=g)) * 20.0, float(torch.randn(1, generator=g)) * 0.1, p]
        if kind == "old":
            row = row[:5 + (t % 3)]
        elif kind >= 8:
            row.append(drew)
            if kind == 9:
                row.append(float(torch.rand(1, generator=g)) * 0.1)
        out.append(row)
    return out


def _gate_state(L, gate_):
    return ({k: v.clone() for k, v in gate_.state_dict().items()}, L.opt_gate.state_dict() if gate_ is L.m.mouth_gate else L.opt_motor.state_dict())


def _same_opt(a, b):
    sa, sb = a["state"], b["state"]
    return sorted(sa) == sorted(sb) and all(sorted(sa[k]) == sorted(sb[k]) and all(
        (torch.equal(sa[k][f], sb[k][f]) if torch.is_tensor(sa[k][f]) else sa[k][f] == sb[k][f]) for f in sa[k]) for k in sa)


def test_the_gate_lesson_as_before():
    """anatomy 16 (step R5): the gate's lesson, now one lesson for every effector, is for the voice with the switches off bit for bit the
    lesson before R5 (verbatim in this test): the gate's weights, its optimizer's state, the baseline, the report and the rows left,
    over two lessons in a row, under SGD and Adam, the effort in the reward or not, the drive on the reward rate, no vigor, and rows of
    today's form and older forms; a lived day's rows are today's seven numbers"""
    n_cases = 0
    for label, extra in (("sgd", {}), ("adam", dict(gate_opt="adam")), ("the effort in the reward", dict(cost_in_reward=1)),
                         ("the drive on the rate, no vigor", dict(gate_tonic_rate=0.7, gate_vigor=0.0, gate_int=0.4)),
                         ("the served constants", dict(_served_cfg() or {}))):
        cfg = dict(extra, wake_ticks=100000, gate_every=10 ** 9)
        lives = []
        for _ in range(2):
            torch.manual_seed(5); L = _born(TOK, cfg); _live(L, ticks=40); lives.append(L)
        A, B = lives
        assert all(len(r_) == 7 for r_ in A.gate_buf) and len(A.gate_buf) == 40
        width = A.m.mouth_gate.in_features
        for kind in (7, "old"):
            for L in (A, B):
                L.gate_buf.clear(); L.gate_buf.extend(_rows(torch.Generator().manual_seed(17), 60, width, kind))
            for _ in range(2):
                _gate_lesson_before_r5(A); B._gate_lesson()
                (wa, oa), (wb, ob) = _gate_state(A, A.m.mouth_gate), _gate_state(B, B.m.mouth_gate)
                assert all(torch.equal(wa[k], wb[k]) for k in wa) and _same_opt(oa, ob), f"{label}, rows {kind}: the gate moved otherwise"
                assert A._g_base == B._g_base and A._gate_last == B._gate_last, (label, kind, A._gate_last, B._gate_last)
                assert len(A.gate_buf) == len(B.gate_buf) and all(ra[1:] == rb[1:] and torch.equal(ra[0], rb[0]) for ra, rb in zip(A.gate_buf, B.gate_buf))
                A.gate_buf.extend(_rows(torch.Generator().manual_seed(18), 30, width, kind)); B.gate_buf.extend(_rows(torch.Generator().manual_seed(18), 30, width, kind))
                n_cases += 1
    print(f"anatomy 16: the voice's gate lesson as before R5, bit for bit ({n_cases} lessons: SGD, Adam, the effort in the reward, the drive on",
          "the rate, the served constants; today's rows and older ones)")


def test_the_switches():
    """anatomy 17 (step R5; ops/review_2026-09-22.md section 1): the three defect fixes are switches, off by their absence (a life's cfg
    holds none unless set). gate_own_draw (defect 4): each row carries the gate's own draw, a go that ended in a rest is recorded as the
    go, a letter run without the gate as a go at p 1, and the lesson's eligibility takes the draw; elig_from 1 (defect 8): the credit
    sums the dopamine from the tick after the act; the lesson under each is the lesson before R5 with only that fix written in, for the
    voice and for a later effector (its cost its own row's); actor_trace_tick (defect 5): the actor's trace (and the chooser's)
    decays by dopamine's discount on every tick, where without it a tick with no act leaves it as it was"""
    from body.core.physiology import SWITCHES
    assert sorted(SWITCHES) == ["actor_trace_tick", "elig_from", "gate_own_draw"] and all(v == 0 for v in SWITCHES.values())
    assert not any(k in _born(TOK, {}).cfg for k in SWITCHES) and not any(k in PHYSIOLOGY for k in SWITCHES)
    # the lessons: the voice, then a later effector, under each switch, against the lesson with the fix written in
    n = 0
    for own, s in ((1, 0), (0, 1), (1, 1)):
        cfg = dict(wake_ticks=100000, gate_every=10 ** 9, gate_own_draw=own, elig_from=s, fast_rls=1, fast_input="striatum", actor=1)
        torch.manual_seed(5); A = _born(_Arm(TOK, cfg), cfg); torch.manual_seed(5); B = _born(_Arm(TOK, cfg), cfg)
        _live(A, ticks=30); _live(B, ticks=30)
        assert all(len(r_) == 7 + own for r_ in A.gate_buf) and all(len(r_) == 10 for r_ in A.motor[0]["buf"])   # R6: the reflex's flag last
        for i in (0, 1, 2):
            gA = A.m.mouth_gate if i == 0 else A.m.get_submodule(A.anatomy.effectors[i].gate)
            gB = B.m.mouth_gate if i == 0 else B.m.get_submodule(B.anatomy.effectors[i].gate)
            bufA = A.gate_buf if i == 0 else A.motor[i - 1]["buf"]; bufB = B.gate_buf if i == 0 else B.motor[i - 1]["buf"]
            kind = (7 + own) if i == 0 else 9
            for buf_ in (bufA, bufB):
                buf_.clear(); buf_.extend(_rows(torch.Generator().manual_seed(21 + i), 60, gA.in_features, kind))
            cost_of = (lambda b: float(A.cfg["symbol_cost"])) if i == 0 else (lambda b: float(b[8]))
            baseA = getattr(A, "_g_base", None) if i == 0 else A.motor[i - 1]["g_base"]
            baseA, lastA = _lesson_fixed(A, bufA, gA, A.opt_gate if i == 0 else A.opt_motor, baseA, own, s, cost_of)
            B._gate_lesson(i)
            baseB = getattr(B, "_g_base", None) if i == 0 else B.motor[i - 1]["g_base"]; lastB = B._gate_last if i == 0 else B.motor[i - 1]["last"]
            (wa, oa), (wb, ob) = _gate_state(A, gA), _gate_state(B, gB)
            assert lastA is not None and all(torch.equal(wa[k], wb[k]) for k in wa) and _same_opt(oa, ob), f"own {own}, from {s}, effector {i}: the gate moved otherwise"
            assert baseA == baseB and lastA == lastB and len(bufA) == len(bufB) == 60 - lastA["n"] and lastA["n"] == 60 - 12 - s, (own, s, i, lastA, lastB)
            n += 1
    # the draw recorded, live: the plain mouth with the rest as the end, its probe readout leaning to the rest (every go draws the rest),
    # and the served chunk mouth (its turn-taking reflexes off), its probe leaning to the rest inside a word (every letter after the
    # first is the rest)
    rows = {}
    for mouth, base_cfg in (("plain", dict(end_rest=1)), ("chunk", dict(_served_cfg() or {}, actor_form="chunk", gate_listen=0.0, gate_yield=0.0,
                                                                         gate_quiet_tau=0, pace_sense=0))):
        for own in (0, 1):
            cfg = dict(base_cfg, wake_ticks=100000, gate_every=10 ** 9, gate_floor=0.9, gate_own_draw=own)
            torch.manual_seed(5); L = _born(TOK, cfg); got = []
            ro_ = L.m.readout; bump_ = torch.zeros(L.m.vocab); bump_[L.sil] = 1000.0

            def lean(pred, prior=None, ro_=ro_, bump_=bump_, L=L, mouth=mouth):
                inside = getattr(L, "_acted_last", False) and getattr(L, "_own_last", None) not in (None, L.space_id)
                return ro_(pred, prior) + (bump_ if (mouth == "plain" or inside) else 0.0)
            L.m.readout = lean
            f = L._feel_and_learn

            def rec(*a, f=f, got=got, L=L):
                f(*a); got.append(list(L.gate_buf[-1][1:]))
            L._feel_and_learn = rec
            _live(L, lines=("what do you want?", "I want milk", "do you see the ball?", "yes. the ball is red", "go up"), ticks=150)
            rows[(mouth, own)] = got
    assert all(len(r_) == 6 for k_, rs_ in rows.items() if not k_[1] for r_ in rs_), "the rows' form without the switch"
    assert all(len(r_) == 7 for k_, rs_ in rows.items() if k_[1] for r_ in rs_), "the rows' form with the switch"
    r1 = rows[("plain", 1)] + rows[("chunk", 1)]
    assert all(r_[6] or not r_[0] for r_ in r1) and all(r_[6] for r_ in r1 if r_[5] == 1.0), "an act without a draw, or a program's letter not a go"
    ended = sum(1 for r_ in rows[("plain", 1)] if r_[6] and not r_[0] and r_[5] < 1.0)
    letters = sum(1 for r_ in rows[("chunk", 1)] if r_[6] and not r_[0] and r_[5] == 1.0)
    before = sum(1 for r_ in rows[("chunk", 0)] if not r_[0] and r_[5] == 1.0)     # without the switch: the gate's no at p 1, never decided
    acted = sum(1 for r_ in r1 if r_[0])
    assert ended >= 20 and letters >= 10 and before >= 10 and acted >= 10, (ended, letters, before, acted)
    # the actor's trace, per act or per tick, under the add form (the actor's) and the softmax form (the chooser's)
    moved = {}
    for form, name in (("add", "_e_actor"), ("softmax", "_e_chooser")):
        for tt in (0, 1):
            cfg = dict(wake_ticks=100000, fast_rls=1, fast_input="striatum", actor=1, actor_form=form, gate_floor=0.3, actor_margin=8.0,
                       actor_trace_tick=tt)
            torch.manual_seed(5); L = _born(TOK, cfg); gam = float(L.m.gammas()[int(L.cfg["dopamine_band"])]); seen = []
            f = L._act

            def rec(*a, f=f, seen=seen, L=L, name=name):
                prev = getattr(L, name, None); prev = prev.clone() if prev is not None else None
                out = f(*a); now = getattr(L, name, None)
                seen.append((prev, now.clone() if now is not None else None, bool(a[5] and a[11] and not L._chunk_cont))); return out
            L._act = rec
            _live(L, lines=("what do you want?", "I want milk", "do you see the ball?"), ticks=90)
            quiet = [(p_, n_) for p_, n_, act_ in seen if p_ is not None and not act_]
            assert len(quiet) >= 10 and sum(1 for _, _, a_ in seen if a_) >= 3, (form, tt, len(quiet))
            if tt:
                assert all(torch.equal(n_, gam * p_) for p_, n_ in quiet), f"{form}: a quiet tick did not decay the trace by the discount"
            else:
                assert all(torch.equal(n_, p_) for p_, n_ in quiet), f"{form}: without the switch a quiet tick moved the trace"
            moved[(form, tt)] = len(quiet)
    # THE LATER EFFECTORS' TRACES (the R5 verifier's third): each later effector's actor trace decays by dopamine's discount on every
    # tick it did not act under actor_trace_tick, and stands still on such a tick without it, as the voice's
    for tt in (0, 1):
        cfg = dict(wake_ticks=100000, fast_rls=1, fast_input="striatum", actor=1, gate_floor=0.3, actor_trace_tick=tt)
        torch.manual_seed(5); L = _born(_Arm(TOK, cfg), cfg); seen = []
        f = L._act_effectors

        def rec(u, stri, gam, tick_tr, f=f, seen=seen, L=L):
            prev = [st_["e_actor"].clone() if st_["e_actor"] is not None else None for st_ in L.motor]
            went = [bool(st_["now"]["acted"] and st_["now"]["act_on"]) for st_ in L.motor]
            out = f(u, stri, gam, tick_tr)
            g_ = float(gam[int(L.cfg["dopamine_band"])])
            seen.append([(p_, st_["e_actor"].clone() if st_["e_actor"] is not None else None, a_, g_) for p_, st_, a_ in zip(prev, L.motor, went)])
            return out
        L._act_effectors = rec
        _live(L, lines=("what do you want?", "I want milk", "do you see the ball?"), ticks=90)
        for i_, e_ in enumerate(L.anatomy.effectors[1:]):
            quiet = [(p_, n_, g_) for row_ in seen for p_, n_, a_, g_ in (row_[i_],) if p_ is not None and not a_]
            went = sum(1 for row_ in seen if row_[i_][2])
            assert len(quiet) >= 10 and went >= 3, (e_.name, tt, len(quiet), went)
            if tt:
                assert all(torch.equal(n_, g_ * p_) for p_, n_, g_ in quiet), f"{e_.name}: a quiet tick did not decay its trace by the discount"
            else:
                assert all(torch.equal(n_, p_) for p_, n_, g_ in quiet), f"{e_.name}: without the switch a quiet tick moved its trace"
            moved[(e_.name, tt)] = len(quiet)
    print(f"anatomy 17: the switches off by their absence; {n} lessons under gate_own_draw and elig_from equal the lesson with the fix",
          f"written in (the voice and two later effectors); live, {ended} drawn rests at a word's start and {letters} program letters that",
          f"were the rest recorded as goes (without the switch {before} such letters read as the gate's no at p 1), every program letter",
          f"a go at p 1; the actor's and the chooser's traces decay every tick under actor_trace_tick ({moved[('add', 1)]} and",
          f"{moved[('softmax', 1)]} quiet ticks) and stand still without it, and so do the later effectors' ({moved[('arm', 1)]} and",
          f"{moved[('grip', 1)]} quiet ticks)")


def test_a_later_effector():
    """anatomy 18 (step R5): a body of the diary's anatomy and two later effectors (an arm of two joints of five, a grip of three whose
    gate reads an input of its own). Its organs: the diary's bit for bit (the global random stream left where the diary's leaves it),
    then each effector's table (unit rows per joint from the body's seed alone), gate (born as the voice's, over its declared inputs)
    and actor; its striatum: the language block, thresholds and heads as the diary's, the effectors' blocks appended (since R5b rows
    per joint: an act adds its joints' settings' rows), their lines read after the language line's. The per-joint readout and the acts' rows; each tick the voice draws first (its choice the diary's on the
    first tick), then each effector its gate and its joints; the window holds each act under its field and the input adds each act's
    row after the voice's own sound; every effector's gate and actor learn; the night rests them; a save keeps them; organs that do not
    match the anatomy are refused"""
    cfg = dict(wake_ticks=120, wake_every=8, gate_every=8, night_rounds=1, night_starts=4, night_batch=4, rem_dreams=2, rem_steps=2,
               write_floor=1e-30, fast_rls=1, fast_input="striatum", actor=1, stri_k=8, stri_m=64, gate_floor=0.3)
    a = _Arm(TOK, cfg); V = a.vocab
    assert anatomy_for(a, cfg) is a and [e.n_acts for e in a.effectors[1:]] == [25, 3]
    # the organs
    torch.manual_seed(9); o1 = Organs(V, d=64, layers=2, heads=2, window=32); r1 = torch.get_rng_state()
    torch.manual_seed(9); o3 = Organs(V, d=64, layers=2, heads=2, window=32, effectors=a.effectors, born_seed=3); r3 = torch.get_rng_state()
    torch.manual_seed(1); o4 = Organs(V, d=64, layers=2, heads=2, window=32, effectors=a.effectors, born_seed=3)
    torch.manual_seed(9); o5 = Organs(V, d=64, layers=2, heads=2, window=32, effectors=a.effectors, born_seed=4)
    s1, s3 = o1.state_dict(), o3.state_dict()
    new = [k for k in s3 if k not in s1]
    assert torch.equal(r1, r3) and [k for k in s3 if k in s1] == list(s1) and all(torch.equal(s1[k], s3[k]) for k in s1), "the diary's organs are built otherwise"
    assert new == ["stri_mline", "acts.arm.rows", "acts.grip.rows", "gates.arm.weight", "gates.arm.bias", "gates.grip.weight", "gates.grip.bias",
                   "actors.arm.weight", "actors.arm.bias", "actors.grip.weight", "actors.grip.bias",
                   "timing.arm.pred.weight", "timing.arm.pred.bias", "timing.grip.pred.weight", "timing.grip.pred.bias"], new   # act_pred since R6
    assert torch.equal(o3.acts["arm"].rows, o4.acts["arm"].rows) and not torch.equal(o3.acts["arm"].rows, o5.acts["arm"].rows), "the tables not the body's seed's alone"
    assert o3.acts["arm"].rows.shape == (10, 64) and torch.allclose(o3.acts["arm"].rows.norm(dim=-1), torch.ones(10))
    assert (o3.gates["arm"].in_features, o3.gates["grip"].in_features) == (64 + 5 + 1, 64 + 5 + 2)
    assert float(o3.gates["grip"].weight.detach().abs().max()) == 0.0 and torch.equal(o3.gates["grip"].bias, o3.mouth_gate.bias)
    # the table: digits and flat acts, rows, the per-joint readout
    t = o3.acts["arm"]
    allacts = torch.arange(25)
    dg = t.digits(allacts)
    assert all(t.flat(dg[i].tolist()) == i for i in range(25)) and dg[12].tolist() == [2, 2] and dg[7].tolist() == [1, 2]
    assert torch.equal(t(allacts), t.rows[dg[:, 0]] + t.rows[5 + dg[:, 1]]) and t(torch.tensor([[3, 4]])).shape == (1, 2, 64)
    pred = torch.randn(64)
    lg = t.logits(pred, 7.0)
    assert len(lg) == 2 and torch.equal(lg[0], 7.0 * (pred @ t.rows[:5].t())) and torch.equal(lg[1], 7.0 * (pred @ t.rows[5:].t()))
    assert all(torch.equal(x_, torch.zeros(5)) for x_ in t.logits(None, 7.0))
    # the striatum: the diary's block, thresholds and heads, then the effectors' blocks; their lines read after the language line's
    torch.manual_seed(4); o1.striatum_init(8, 64, seed=2, wm=1); q1 = torch.get_rng_state()
    torch.manual_seed(4); o3.striatum_init(8, 64, seed=2, wm=1, effectors=a.effectors); q3 = torch.get_rng_state()
    nl = 8 * (2 * V + 3)
    assert torch.equal(q1, q3) and o3.stri_W.shape == (nl + 8 * 10 + 8 * 3, 64) and torch.equal(o3.stri_W[:nl], o1.stri_W)
    assert torch.equal(o3.stri_b, o1.stri_b) and torch.equal(o3.actor.weight, o1.actor.weight) and torch.equal(o3.vfast.weight, o1.vfast.weight)
    assert o3.stri_blocks == [(nl, (5, 5)), (nl + 80, (3,))] and o3.actors["arm"].weight.shape == (10, 64 * 2) and o3.actors["grip"].weight.shape == (3, 128)
    # the rows per joint (step R5b), drawn after the thresholds from the striatum's own generator: the arm's 8 positions x (5 + 5)
    # settings at 1/sqrt(8 x 2), then the grip's 8 x 3 at 1/sqrt(8 x 1)
    g_ = torch.Generator().manual_seed(2 + 7919); torch.randn(nl, 64, generator=g_); torch.rand(64, generator=g_)
    assert torch.equal(o3.stri_W[nl:], torch.cat([torch.randn(80, 64, generator=g_) / math.sqrt(16.0), torch.randn(24, 64, generator=g_) / math.sqrt(8.0)]))
    for o_ in (o1, o3):
        o_.striatum_push(0, 5); o_.striatum_push(1, 7); o_.striatum_push(2, 1)
    o3.striatum_push_act(0, 7); o3.striatum_push_act(0, 12); o3.striatum_push_act(1, 2)
    z = o1.stri_b.clone()
    for p_ in range(8):
        e_ = int(o1.stri_line[p_])
        if e_ >= 0:
            z += o1.stri_W[p_ * (2 * V + 3) + e_]
    z0 = z.clone()
    for j_, (base_, fac_) in enumerate(o3.stri_blocks):          # each act's joints' settings (its table's digits), joint 0 first
        tab_ = o3.acts[a.effectors[1 + j_].name]
        for p_ in range(8):
            x_ = int(o3.stri_mline[j_, p_])
            if x_ >= 0:
                off_ = base_ + p_ * sum(fac_)
                for d_, k_ in zip(tab_.digits(torch.tensor(x_)).tolist(), fac_):
                    z += o3.stri_W[off_ + d_]; off_ += k_
    # by hand: the arm's 12 (2, 2) at position 0 and its 7 (1, 2) at position 1, the grip's 2 at position 0
    by_hand = z0 + o3.stri_W[nl + 2] + o3.stri_W[nl + 5 + 2] + o3.stri_W[nl + 10 + 1] + o3.stri_W[nl + 10 + 5 + 2] + o3.stri_W[nl + 80 + 2]
    assert o3.stri_mline[0, :3].tolist() == [12, 7, -1] and torch.equal(o3.striatum_read(), torch.relu(z)) and not torch.equal(o3.striatum_read(), o1.striatum_read())
    assert torch.allclose(z, by_hand, atol=1e-6)
    o3.striatum_reset(); assert int(o3.stri_mline.max()) == -1 and int(o3.stri_line.max()) == -1
    # the life: the voice draws first; its first choice is the diary's
    lines = ("what do you want?", "I want milk", "do you see the ball?", "yes. the ball is red")
    faces = {10: 2.0, 11: 0.0, 40: -2.0, 41: 0.0, 70: 2.0, 71: 0.0}
    torch.manual_seed(5); D = _born(TOK, cfg); torch.manual_seed(5); M = _born(_Arm(TOK, cfg), cfg)
    sd, sm = D.m.state_dict(), M.m.state_dict()
    assert all(torch.equal(sd[k], sm[k][:sd[k].shape[0]] if k == "stri_W" else sm[k]) for k in sd), "born beside later effectors, the diary's organs differ"
    D.type_text("hello", who="parent"); M.type_text("hello", who="parent"); D.tick(); M.tick()
    assert D._last_choice == M._last_choice, (D._last_choice, M._last_choice)
    calls = []; o_rand, o_mn = torch.rand, torch.multinomial

    def rand(*a_, generator=None, **k_):
        if generator is M.gen:
            f_ = sys._getframe(1); calls.append((M.ticks, f_.f_code.co_name, f_.f_locals.get("i", 0), "rand"))
        return o_rand(*a_, generator=generator, **k_)

    def mn(*a_, generator=None, **k_):
        if generator is M.gen:
            f_ = sys._getframe(1); calls.append((M.ticks, f_.f_code.co_name, f_.f_locals.get("i", 0), "draw"))
        return o_mn(*a_, generator=generator, **k_)
    torch.rand, torch.multinomial = rand, mn
    try:
        _live(M, lines=lines, faces=faces, ticks=119)
    finally:
        torch.rand, torch.multinomial = o_rand, o_mn
    assert M.nights == 1 and not (M.last_night or {}).get("error"), (M.last_night or {}).get("error")
    day = [c_ for c_ in calls if c_[1] in ("_choose", "_choose_effector")]
    by_tick = collections.defaultdict(list)
    for c_ in day:
        by_tick[c_[0]].append((c_[2], c_[3]))
    joints = {0: 1, 1: 2, 2: 1}; n_draws = collections.Counter()
    for tk, seq in by_tick.items():
        assert [x_ for x_ in seq if x_[1] == "rand"] == [(0, "rand"), (1, "rand"), (2, "rand")], (tk, seq)
        assert [x_[0] for x_ in seq] == sorted(x_[0] for x_ in seq), f"tick {tk}: an effector drew before the one before it: {seq}"
        for i_ in (0, 1, 2):
            k_ = sum(1 for x_ in seq if x_ == (i_, "draw")); n_draws[i_] += k_
            assert k_ in (0, joints[i_]), (tk, i_, seq)
    assert len(by_tick) == 119 and min(n_draws.values()) >= 5, (len(by_tick), n_draws)
    # the window, the input and the record (after the night: the morning's positions)
    _live(M, lines=("go up", "we go up"), ticks=40)
    win = list(M.win); obs, whos, bundles, reads = M._window_tensors()
    assert list(obs) == ["ear", "face", "arm", "grip"] and all("arm" in w and "grip" in w for w in win)
    assert torch.equal(obs["arm"], torch.tensor([w["arm"] for w in win])) and int((obs["arm"] != 12).sum()) >= 3 and int((obs["grip"] != 1).sum()) >= 3
    diary = LanguageAnatomy(TOK, M.cfg); ln = M.m.in_ln; M.m.in_ln = torch.nn.Identity()
    try:
        with torch.no_grad():
            u_m, u_d = M.m.inputs(M.anatomy, obs, whos, bundles), M.m.inputs(diary, obs, whos, bundles)
            g_ = float(M.m.own_gain)
            want = u_d + g_ * (obs["arm"] != 12).float().unsqueeze(-1) * M.m.acts["arm"](obs["arm"])
            want = want + g_ * (obs["grip"] != 1).float().unsqueeze(-1) * M.m.acts["grip"](obs["grip"])
    finally:
        M.m.in_ln = ln
    assert torch.equal(u_m, want) and not torch.equal(u_m, u_d), "the effectors' acts are not the sum's terms after its own sound"
    assert set(M.last["acts"]) == {"arm", "grip"} and set(M.insides()["effectors"]) == {"arm", "grip"}
    # one choice read closely: the gate's probability over the floor (the proposal's salience in it), and the per-joint readout of the
    # proposal (act_pred's since R6) with the actor's bias joint by joint
    st_ = M.motor[0]; saved_ = (M.m.actors["arm"].weight.detach().clone(), M.m.actors["arm"].bias.detach().clone(), M.gen.get_state())
    with torch.no_grad():
        M.m.actors["arm"].weight.zero_(); M.m.actors["arm"].bias.copy_(torch.arange(10, dtype=torch.float32) / 3.0 - 1.5)
    C1 = M._C_last; lvl = 0.25; fr = Frame(M.ticks, {"ear": M.sil}, M.face_now)
    M._choose_effector(1, fr, C1, lvl, True)
    now = st_["now"]; beta = float(M.cfg.get("actor_beta", 1.0)); b_ = beta * torch.tanh(M.m.actors["arm"].bias.detach())
    with torch.no_grad():
        pr_ = M.m.timing["arm"].pred(C1); lg_ = M.m.acts["arm"].logits(pr_, float(M.m.read_sharp))
    assert torch.allclose(now["probs"][0], torch.softmax(lg_[0] + b_[:5], -1)) and torch.allclose(now["probs"][1], torch.softmax(lg_[1] + b_[5:], -1)), now["probs"]
    assert float(lg_[0].abs().max()) > 0, "act_pred proposed nothing"
    with torch.no_grad():
        sal_ = float(M.cfg["gate_salience"]) * float(pr_.norm())
        f_ = torch.cat([C1 / math.sqrt(64.0), torch.tensor([M.fatigue / 10.0, M.mood / 6.0, M.stress / 10.0, sal_, lvl]), torch.tensor([1.0 if st_["acted_last"] else 0.0])])
        z_ = M.m.gates["arm"](f_.unsqueeze(0))[0, 0] / (1.0 + M.stress / 10.0)
    fl_ = float(M.cfg["gate_floor"]); assert abs(now["p_act"] - (fl_ + (1.0 - fl_) * float(torch.sigmoid(z_)))) < 1e-6 and torch.equal(now["feat"], f_)
    with torch.no_grad():
        M.m.actors["arm"].weight.copy_(saved_[0]); M.m.actors["arm"].bias.copy_(saved_[1])
    M.gen.set_state(saved_[2])
    assert all(len(r_) == 7 for r_ in M.gate_buf) and all(len(r_) == 10 and r_[9] is False for st_ in M.motor for r_ in st_["buf"])
    for e_, st_ in zip(M.anatomy.effectors[1:], M.motor):
        assert st_["last"] and "n" in st_["last"], (e_.name, st_["last"])
        assert float(M.m.gates[e_.name].weight.detach().abs().max()) > 0 and float(M.m.actors[e_.name].weight.detach().abs().max()) > 0, e_.name
    # the night rests them; a save keeps them; the loaded body lives on
    L2 = _born(_Arm(TOK, cfg), cfg); _live(L2, lines=lines, ticks=120)
    assert L2.nights == 1 and L2.sleep_pressure == 0, (L2.nights, L2.sleep_pressure)
    for st_ in L2.motor:
        assert len(st_["buf"]) == 0 and st_["e_actor"] is None and st_["now"] is None and not st_["acted_last"]
    assert int(L2.m.stri_mline.max()) == -1
    import contextlib
    import io
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    try:
        M.save(path)
        C = Life.load(path, _Arm(TOK, M.cfg), save_path=None)
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            Dl = Life.load(path, TOK, save_path=None)             # the diary's anatomy: the effectors' organs are said, not loaded
    finally:
        os.remove(path)
    note = [l_ for l_ in said.getvalue().splitlines() if "does not declare" in l_]
    assert len(note) == 1 and all(k_ in note[0] for k_ in ("acts.arm.rows", "gates.grip.weight", "actors.arm.weight", "stri_mline", "timing.arm.pred.weight")), said.getvalue()
    assert not hasattr(Dl, "motor") and not hasattr(Dl.m, "acts") and not hasattr(Dl.m, "timing") and not hasattr(Dl, "opt_inv") and Dl.m.stri_W.shape[0] == 8 * (2 * V + 3)
    sM, sC = M.m.state_dict(), C.m.state_dict()
    assert list(sM) == list(sC) and all(torch.equal(sM[k], sC[k]) for k in sM), [k for k in sM if not torch.equal(sM[k], sC[k])]
    _live(C, ticks=10); assert C.ticks == M.ticks + 10
    # THE EFFORT (the R5 verifier's first): each later effector's act adds its declared cost to the body's fatigue, after the voice's,
    # in the anatomy's order, one at a time; a tick it does not act adds nothing and records no cost
    cfg3 = dict(cfg, wake_ticks=100000, gate_floor=0.5)
    torch.manual_seed(5); L3 = _born(_Arm(TOK, cfg3), cfg3); rows = []
    f = L3._act_effectors

    def rec(u, stri, gam, tick_tr, f=f, rows=rows, L3=L3):
        before = L3.fatigue; out = f(u, stri, gam, tick_tr)
        rows.append((before, L3.fatigue, [(st_["now"]["acted"], st_["now"]["cost"]) for st_ in L3.motor])); return out
    L3._act_effectors = rec
    _live(L3, lines=lines, faces=faces, ticks=90)
    efforts = [e_.effort for e_ in L3.anatomy.effectors[1:]]
    for before, after, per in rows:
        want = before
        for (acted_, cost_), eff_ in zip(per, efforts):
            assert cost_ == (float(eff_) if acted_ else 0.0), (per, efforts)
            if acted_:
                want += cost_
        assert after == want, f"the fatigue went {before!r} -> {after!r}, the effectors' costs give {want!r}"
    n_cost = [sum(1 for r_ in rows if r_[2][i_][0]) for i_ in range(2)]
    assert len(rows) == 90 and min(n_cost) >= 5 and len(rows) - max(n_cost) >= 5, n_cost
    # THE RESERVED ACTS (the R5 verifier's second): a one-joint effector's reserved acts are never drawn, the actor's bias on them or not
    # (their probability exactly 0), while its other acts are
    cfg4 = dict(cfg, wake_ticks=100000, gate_floor=0.9)
    torch.manual_seed(5); L4 = _born(_Tap(TOK, cfg4), cfg4); drawn = collections.Counter(); n_draw = 0
    for t in range(150):
        if t % 30 == 0 and t // 30 < len(lines):
            L4.type_text(lines[t // 30], who="parent")
        if t in faces:
            L4.set_face(faces[t])
        L4.tick()
        now_ = L4.motor[0]["now"]
        assert float(now_["probs"][0][3]) == 0.0, now_["probs"]
        if now_["drew"]:
            n_draw += 1; drawn[now_["act"]] += 1
    assert drawn[3] == 0 and n_draw >= 100 and drawn[0] >= 10 and drawn[2] >= 10, (n_draw, drawn)
    assert L4.motor[0]["now"]["act_on"], "the actor's bias was not on the reserved act's draw"
    # organs that do not match the anatomy
    refused = []
    wide = _Arm(TOK, {}); wide.effectors[2].n_in = 3
    flat = _Arm(TOK, {}); flat.effectors[1].factors = [25]
    for what, call in (("organs without the effectors' organs", lambda: Life(o1, _Arm(TOK, {}))),
                       ("organs with an undeclared effector's", lambda: Life(o3, TOK)),
                       ("a gate narrower than its declaration", lambda: Life(o4, wide)),
                       ("a table of other joints", lambda: Life(o4, flat))):
        try:
            call()
        except ValueError:
            refused.append(what); continue
        raise AssertionError(f"the life took {what}")
    print(f"anatomy 18: the diary's organs and striatum as the diary's, the effectors' built last and appended; the per-joint readout;",
          f"the voice draws first on each of {len(by_tick)} ticks ({n_draws[0]} words drawn), then the arm ({n_draws[1]} acts, two joints",
          f"each) and the grip ({n_draws[2]}); the acts in the window and the sum after its own sound; gates and actors learn; the night",
          f"rests them; a save keeps them; each act's declared cost reaches the fatigue in order ({n_cost[0]} and {n_cost[1]} acts);",
          f"a one-joint effector's reserved act never drawn in {n_draw} draws ({drawn[0]}, {drawn[1]}, {drawn[2]} of its others);",
          f"refused: {'; '.join(refused)}")


def test_every_call_site_passes_the_effectors():
    """anatomy 19 (step R5): the eleven places the cortex's input is made all pass each later effector's acts beside the channels (a
    dream's, imagination's and REM's at the effector's rest), in the shape of its own sound"""
    import body.model as BM
    cfg = dict(wake_ticks=100000, wake_every=8, gate_every=8, night_rounds=1, night_starts=6, night_batch=4, rem_dreams=2, rem_steps=2,
               rem_rounds=1, write_floor=1e-30, fast_rls=1, fast_input="striatum", face_form="foresee", rem_form="forecast", rem_temp=1.0,
               actor=1, gate_floor=0.3)
    torch.manual_seed(5); L = _born(_Arm(TOK, cfg), cfg)
    names = [c.name for c in L.anatomy.channels] + [e.name for e in L.anatomy.effectors[1:]]
    seen = collections.Counter(); bad = []; rests = collections.Counter()
    orig = BM.Organs.inputs

    def spy(self, anatomy, obs, xos, bundles):
        where = sys._getframe(1).f_code.co_name + ("[batch]" if bundles.dim() == 4 and sys._getframe(1).f_code.co_name == "night" else "")
        seen[where] += 1
        if anatomy is not L.anatomy or list(obs) != names or any(obs[k].shape[:len(xos.shape)] != xos.shape for k in names):
            bad.append(where)
        if bool((obs["arm"] == 12).all()) and bool((obs["grip"] == 1).all()):
            rests[where] += 1
        return orig(self, anatomy, obs, xos, bundles)
    BM.Organs.inputs = spy
    try:
        _live(L, lines=("what do you want?", "I want milk", "do you see the ball?"), faces={20: 2.0, 21: 0.0}, ticks=90)
        L._imagine_value(TOK.token_to_id("a"), 3)
        dreams = L.dreams(6)
        L.gauge(dreams); L.cfg["night_batch"] = 0; L.gauge(dreams)
        L._frel_corr = 0.5
        ids = max(dreams, key=len)
        lines0 = (L.m.stri_line.clone(), L.m.stri_mline.clone())
        L._rem_imagine(ids)
        assert torch.equal(L.m.stri_line, lines0[0]) and torch.equal(L.m.stri_mline, lines0[1]) and int(lines0[1].max()) >= 0, "REM left the lines moved"
        L._rem_rollout(ids, 0.0)
        L.cfg["night_batch"] = 4; r1 = L.night()
        L.cfg["night_batch"] = 0; r2 = L.night()
    finally:
        BM.Organs.inputs = orig
    assert not r1.get("error") and not r2.get("error"), (r1.get("error"), r2.get("error"))
    want = {"_stream_now", "_wake_lesson", "_imagine_value", "_dream_inputs", "_dream_batch", "night[batch]", "night", "_rem_imagine",
            "_rem_rollout", "_gauge_batched", "gauge"}
    assert set(seen) == want, f"reached {sorted(seen)}, want {sorted(want)}"
    assert not bad, f"these passed another anatomy or not all its channels and effectors: {sorted(set(bad))}"
    dreamt = want - {"_stream_now", "_wake_lesson", "_imagine_value"}
    assert all(rests[w_] == seen[w_] for w_ in dreamt) and rests["_stream_now"] < seen["_stream_now"], (dict(rests), dict(seen))
    print("anatomy 19: all eleven call sites pass the effectors' acts beside the channels; the dreams' at rest, the lived window's acting")


# ---------------- step R9: the world loop ----------------

def _sense_before_r9(self):
    """`_sense` as it was before step R9 (body/core/senses.py at commit 0562112), verbatim"""
    m = self.m
    u = self.queue.popleft() if self.queue else self.sil
    who = (self.queue_who.popleft() if self.queue_who else "") if u != self.sil else ""
    # THE OFFSET: the world quiet for offset_ticks after its utterance, once per pause, whatever the body is
    # saying meanwhile (with the body's silence required too, a babbling body never let it fire: run 41 held
    # two turn-end memories after six days)
    off = int(self.cfg.get("offset_ticks", 0)); settle_form = str(self.cfg.get("offset_form", "count")) == "settle"
    ps_ = self._pace_mode()                                    # under the sensed pace (2) the pause outlasted (M2, _pace_hear) replaces the count
    if u == self.sil and off > 0 and not self._offset_done and not settle_form and ps_ < 2 and self.ticks - self._last_world >= off:
        self._offset(settled=False); self._offset_done = True
    first_after_pause = (u != self.sil and self._offset_done)  # the first symbol after a perceived pause begins an utterance
    if u != self.sil:
        if ps_:
            self._pace_heard(ps_ >= 2)                          # the gap just ended is a heard event (before the world's last symbol moves)
        self._last_world = self.ticks; self._offset_done = False; self._turn_open = False; self._ear_release_at = None
    # THE FELT REWARD (the core refactor's step R3, docs/SIM_DESIGN.md 8.4): the anatomy's reward sources (body/core/anatomy.py),
    # felt in their declared order on this tick's frame and added one at a time in that order, the float order of the sum. The
    # diary's: the face, then the world's words (world_r, not over its own voice under world_mask), then the effort (cost_in_reward).
    frame = Frame(self.ticks, {"ear": u}, self.face_now)   # this tick of the world, built here from the queue (no draw moves)
    judge, *others = self.anatomy.rewards
    # source 0, the world's judgment (the face: a change is felt; a held face is silence; easing off is not an event): its
    # feeling is the tick's felt event, and its term alone is the world's reward the ring and the actor's reliability read
    felt = judge.felt(frame, self)
    r = judge.term(felt)                                # the world's reward: the felt face, clipped like a press
    self._ring_r.append(r)
    if self._act_pending:                               # the actor's reliability: the reward of the ticks after each act, on its vote for that act
        H = int(self.cfg.get("actor_horizon", 16))
        for p_ in self._act_pending:
            p_[2] += r
        while self._act_pending and self.ticks - self._act_pending[0][0] >= H:
            t0_, v_, g_ = self._act_pending.popleft(); self._arel_update(v_, g_)
    for s_ in others:                                   # the reward's other terms, in their order; a silent source adds nothing
        v_ = s_.felt(frame, self)
        if v_ is not None:
            r += s_.term(v_)
    if int(self.cfg.get("own_store", 0)):
        thr = float(self.cfg.get("own_store_r", 1.0))
        if float(r) >= thr and not getattr(self, "_own_stored", False) and self.ticks - getattr(self, "_own_store_tick", -10 ** 9) >= int(self.cfg.get("own_store_gap", 40)):
            self._own_stored = True; self._own_store_tick = self.ticks; self._consolidate_own(float(r) * float(self.cfg.get("own_store_gain", 0.3)))
        elif float(r) < 0.5 * thr:
            self._own_stored = False
    return u, who, felt, r, off, settle_form, first_after_pause


def _sleep_now_before_r9(self):
    """`_sleep_now` as it was before step R9 (body/core/night.py at commit 0562112), verbatim"""
    self.queue.clear(); self.queue_who.clear()
    self.night()


def _canon(g, x, path):
    """every value fed to the hash with its type and shape (as tools/determinism_check.py's --full); an unknown kind stops the test"""
    if torch.is_tensor(x):
        t = x.detach().cpu().contiguous(); g.update(f"T{t.dtype}{tuple(t.shape)}".encode()); g.update(t.numpy().tobytes())
    elif isinstance(x, dict):
        g.update(b"{")
        for k in sorted(x, key=repr):
            g.update(repr(k).encode() + b":"); _canon(g, x[k], f"{path}.{k}")
        g.update(b"}")
    elif isinstance(x, collections.deque):
        g.update(f"Q{x.maxlen}[".encode())
        for i, v in enumerate(x):
            _canon(g, v, f"{path}[{i}]")
        g.update(b"]")
    elif isinstance(x, (list, tuple)):
        g.update(b"[" if isinstance(x, list) else b"(")
        for i, v in enumerate(x):
            _canon(g, v, f"{path}[{i}]")
        g.update(b"]")
    elif isinstance(x, (set, frozenset)):
        g.update(b"S"); _canon(g, sorted(x, key=repr), path)
    elif x is None or isinstance(x, (bool, int, float, str, bytes)):
        g.update((type(x).__name__ + repr(x) + ";").encode())
    elif isinstance(x, (torch.dtype, torch.device)):
        g.update(repr(x).encode())
    else:
        raise TypeError(f"no hash for {type(x).__name__} at {path}")


def _whole(L):
    """the whole life, section by section (the organs with their gradients and plain attributes, the store, the optimizers, the random
    streams, every working attribute but the interface objects, the constants, the page): a hash per section"""
    import hashlib
    m = L.m; OPTS = sorted(k for k, v in vars(L).items() if isinstance(v, torch.optim.Optimizer))
    org = [("sd", dict(m.state_dict())), ("grad", {n: p.grad for n, p in m.named_parameters()}), ("buf", dict(m.named_buffers()))]
    for mn, mod in m.named_modules():
        org.append(("attrs:" + (mn or "."), {k: v for k, v in vars(mod).items() if not k.startswith("_") and not isinstance(v, torch.nn.Module)}))
    SKIP = {"m", "store", "tok", "gen", "cfg", "save_path", "_t_feel", "anatomy", "effectors", "world"} | set(OPTS)
    out = {}
    for name, items in (("organs", org), ("store", sorted(vars(L.store).items())), ("optim", [(k, getattr(L, k).state_dict()) for k in OPTS]),
                        ("rng", [("gen", L.gen.get_state()), ("torch", torch.get_rng_state())]),
                        ("work", [(k, v) for k, v in sorted(vars(L).items()) if k not in SKIP]), ("cfg", [("cfg", L.cfg)])):
        g = hashlib.sha256()
        for k, v in items:
            g.update(k.encode() + b"="); _canon(g, v, name + "." + k)
        out[name] = g.hexdigest()[:16]
    return out


def test_the_diary_world():
    """anatomy 20 (step R9): the language body's world is the DiaryWorld, today's page queue and face reached where the life always held
    them: its frame is the queue's symbol (or the rest), who typed it and the face, at the life's tick; its pause lets the queue go, as
    the sleep switch always did; it keeps no body alive and is not saved with it; its state goes and comes back as bytes; a lapse lets
    queued symbols go by. Lived through the world loop (the frame taken inside the tick, the acts returned and applied, the sleep switch
    pausing the world before the night and resuming it after), under the physiology and the served constants, with symbols queued at
    dusk and a morning after, a life is the whole life a tick before R9 lived (its `_sense` and `_sleep_now` verbatim), section by
    section; its acts are the voice's symbols on its page; the pause falls once at dusk and the resume once after the night"""
    import gc
    from body.core.world import World, DiaryWorld, WorldLoop
    # the world itself
    L = _born(TOK, {}); w = L.world
    assert type(w) is DiaryWorld and isinstance(w, World) and w.life is L and w.now is None
    L.type_text("ab", who="parent"); L.set_face(3.0)
    f1 = w.frame(); f2 = w.frame(); f3 = w.frame()
    a_, b_ = TOK.token_to_id("a"), TOK.token_to_id("b")
    assert (f1.tick, f1.obs, f1.face, f1.truth) == (L.ticks, {"ear": a_}, 3.0, {"who": "parent"}), f1
    assert (f2.obs, f2.truth, f3.obs, f3.truth) == ({"ear": b_}, {"who": "parent"}, {"ear": L.sil}, {"who": ""}) and not L.queue and not L.queue_who
    L.type_text("hello", who="you"); blob = w.save_state(); w.frame(); w.frame(); L.set_face(-1.0)
    w.load_state(blob)
    assert [TOK.decode([i]) for i in L.queue] == list("hello") and list(L.queue_who) == ["you"] * 5 and L.face_now == 3.0
    w.lapse(2); assert [TOK.decode([i]) for i in L.queue] == list("llo") and len(L.queue_who) == 3
    w.pause(); assert not L.queue and not L.queue_who
    assert w.apply({"voice": a_}) is None and w.resume() is None and not L.queue and L.face_now == 3.0
    assert w.type_text("hi", who="parent") == {"queued": 2} and w.set_face(9.0) == {"you": 6.0} and L.face_now == 6.0
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    try:
        L.save(path); blob_ = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        os.remove(path)
    assert not any("world" in str(k_) or "now" == k_ for k_ in blob_["life"]), sorted(blob_["life"])
    w2 = _born(TOK, {}).world; gc.collect()
    try:
        w2.life
    except RuntimeError:
        pass
    else:
        raise AssertionError("the diary's world kept its body alive")
    try:
        Life(_born(TOK, {}).m, TOK, world=object())
    except TypeError:
        pass
    else:
        raise AssertionError("a life took a world that is no World")
    # the language body through the world loop against the body a tick before R9
    lines = ("what do you want?", "I want milk", "do you see the ball?", "yes. the ball is red", "what is cold?", "ice is cold")
    faces = {10: 2.0, 11: 0.0, 40: -2.0, 41: 0.0, 75: 2.0, 76: 4.0, 77: 0.0, 130: -2.0, 131: 0.0}
    tiny = dict(wake_ticks=100, wake_every=8, gate_every=8, night_rounds=1, night_starts=4, night_batch=4, rem_dreams=2, rem_steps=2,
                write_floor=1e-30)
    cases = [("the physiology", dict(tiny))]
    sc = _served_cfg()
    if sc is not None:
        cases.append(("the served constants", dict(sc, **tiny)))
    for label, cfg in cases:
        torch.manual_seed(5); A = _born(TOK, cfg); torch.manual_seed(5); B = _born(TOK, cfg)
        B._sense = types.MethodType(_sense_before_r9, B); B._sleep_now = types.MethodType(_sleep_now_before_r9, B)
        calls = []; pause_, resume_ = A.world.pause, A.world.resume

        def pause(pause_=pause_, calls=calls, A=A):
            calls.append(("pause", A.ticks, A.nights, len(A.queue), A.asleep)); pause_()

        def resume(resume_=resume_, calls=calls, A=A):
            calls.append(("resume", A.ticks, A.nights, len(A.queue), A.asleep)); resume_()
        A.world.pause, A.world.resume = pause, resume
        run = WorldLoop(A); acts = []; dusk_q = None
        for t in range(160):
            for L_ in (A, B):
                if t % 30 == 0 and t // 30 < len(lines):
                    L_.type_text(lines[t // 30], who="parent" if t // 30 % 2 == 0 else "other")
                if t == 95:
                    L_.type_text("are you warm now?", who="parent")      # still queued when the night falls, at the hundredth tick
                if t in faces:
                    L_.set_face(faces[t])
            if t == 99:
                dusk_q = len(A.queue)
            acts.append(run.step()); B.tick()
        del B._sense, B._sleep_now                                   # the verbatim methods were the life's own attributes: the class's again
        assert A.nights == B.nights == 1 and not (A.last_night or {}).get("error"), (A.nights, (A.last_night or {}).get("error"))
        assert _differs(A, B) is None, f"{label}: {_differs(A, B)} differs"
        wa, wb = _whole(A), _whole(B)
        assert wa == wb, f"{label}: the sections {[k for k in wa if wa[k] != wb[k]]} differ from the life before R9"
        assert A.page == B.page and A.last == B.last and list(A.queue) == list(B.queue)
        said = [e[0] for e in A.page if e[1] == 1]
        assert [set(a) for a in acts] == [{"voice"}] * 160 and [TOK.decode([a["voice"]]) if a["voice"] != A.sil else "" for a in acts] == said
        assert run.lived == 160 and run.lapsed == 0 and sum(1 for s_ in said if s_) >= 10
        assert dusk_q and [c[0] for c in calls] == ["pause", "resume"], calls
        assert calls[0][1:] == (100, 0, dusk_q - 1, False) and calls[1][1:] == (100, 1, 0, False), (calls, dusk_q)
        print(f"anatomy 20 ({label}): 160 ticks through the world loop, a night at the hundredth with {dusk_q} symbols queued at dusk,",
              f"the whole life as before R9 ({', '.join(sorted(wa))}); {sum(1 for s_ in said if s_)} symbols said, each the tick's act")
    print("anatomy 20: the diary's world is the queue, who typed and the face; its pause lets the queue go; it keeps no body alive and is",
          "not saved; its state goes and comes back; a lapse lets symbols go by")


class _FrameGrip(Effector):
    """a grip whose gate's own input reads the world's frame (the touch the world shows), and which records every frame it is given"""
    seen = []                                                    # the test's record, not the anatomy's (a class list the test clears)

    def gate_inputs(self, frame, life, state):
        _FrameGrip.seen.append(("gate", frame))
        return [1.0 if state["acted_last"] else 0.0, float(frame.obs["touch"][0])]

    def cost(self, act, frame, life):
        _FrameGrip.seen.append(("cost", frame))
        return float(self.effort)


class _WorldArm(LanguageAnatomy):
    """the diary's words and face, a sense of the world's frames (a touch of two numbers, a bare Channel encoded by the face's map),
    an arm of two joints of five and the frame-reading grip"""

    def __init__(self, tok, cfg=None):
        super().__init__(tok, cfg)
        self.channels.append(Channel("touch", "vector", 2, organ="face_in"))
        self.effectors += [Effector("arm", [5, 5], rest_id=12, effort=0.05), _FrameGrip("grip", [3], rest_id=1, n_in=2, effort=0.03)]


def _stub_world(clock=None):
    """a stub of the simulated world (the SimWorld interface; body/sim/world.py will be the MuJoCo scene): a touch that moves with its
    tick and the arm's acts, the parent's line on the words channel now and then, a smile now and then; it refuses to be seen or moved
    while paused; its state goes and comes back; `clock` (a fake clock's list) moves by each tick's cost in `cost`"""
    from body.core.world import SimWorld

    class Stub(SimWorld):
        LINE = "go up we go "

        def __init__(self):
            self.t = 0; self.x = 0.0; self.paused = False; self.log = []; self.shown = []; self.applied = []; self.lapses = 0
            self.cost = []; self.night_s = 0.0

        def frame(self):
            assert not self.paused, "a frame taken while the world is paused"
            obs = {"touch": [math.sin(self.t / 7.0), self.x]}
            k = self.t % 60 - 10
            if 0 <= k < len(self.LINE):
                obs["ear"] = TOK.token_to_id(self.LINE[k])
            f = Frame(self.t, obs, 2.0 if self.t % 45 == 30 else 0.0, {"who": "parent", "x": self.x})
            self.shown.append((f, clock[0] if clock is not None else None)); return f

        def apply(self, acts):
            assert not self.paused, "the world moved while paused"
            if not acts:
                self.lapses += 1
            else:
                self.applied.append(dict(acts))
                if clock is not None and self.cost:
                    clock[0] += self.cost.pop(0)
            a = int(acts.get("arm", 12)); self.x += 0.01 * (a // 5 - 2); self.t += 1

        def pause(self):
            self.paused = True; self.log.append(("pause", self.t))
            if clock is not None:
                clock[0] += self.night_s

        def resume(self):
            self.paused = False; self.log.append(("resume", self.t))

        def save_state(self):
            return pickle.dumps((self.t, self.x))

        def load_state(self, blob):
            self.t, self.x = pickle.loads(blob)
    return Stub()


def test_a_world_of_frames():
    """anatomy 21 (step R9): the SimWorld is an interface the sim must write whole (it and a half-written one cannot be made). A stub of
    it, with the diary's words and face, a sense of its frames and two later effectors, and every learning rate at 0 (plumbing, not a
    life): through the world loop one frame is taken and one apply given each tick, in order; the frame's words are the tick's world
    symbol on the page, its face is felt, its touch is in every window position the tick opens (a bare Channel observes the world's
    frames), the grip's gate input and each act's cost read the tick's frame, the world takes each tick's acts as the life recorded
    them; the night pauses the world at dusk and the morning resumes it where it stood, nothing seen or moved between; every
    parameter is as born after a day, a night and a morning; a frame handed to the tick is the one it lives; the world is not saved
    with the body, and a loaded body lives on in a world given to it"""
    from body.core.world import SimWorld, WorldLoop
    for bad in ("the interface", "half of it"):
        try:
            if bad == "the interface":
                SimWorld()
            else:
                type("Half", (SimWorld,), {"frame": lambda s: None, "apply": lambda s, a: None})()
        except TypeError:
            continue
        raise AssertionError(f"{bad} was made without the sim's methods")
    LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0)
    cfg = dict(LR0, wake_ticks=120, wake_every=8, gate_every=8, night_rounds=1, night_starts=4, night_batch=4, rem_dreams=2, rem_steps=2,
               write_floor=1e-30, gate_floor=0.3, fast_rls=0)
    w = _stub_world()
    torch.manual_seed(5); L = _born_in(_WorldArm(TOK, cfg), cfg, w)
    assert L.world is w and w.now is None
    p0 = {k: v.detach().clone() for k, v in L.m.named_parameters()}
    run = WorldLoop(L); _FrameGrip.seen.clear(); rec = []
    for t in range(200):
        last_pos = L.win[-1] if L.win else None; n_seen = len(_FrameGrip.seen)
        acts = run.step()
        f, _ = w.shown[-1]
        opened = bool(L.win) and L.win[-1] is not last_pos                # (the night empties the window)
        rec.append((f, acts, dict(L.last["acts"]), L.last["felt"], L.page[-2][0], L.page[-1][0], opened,
                    L.win[-1]["touch"].clone() if L.win else None, _FrameGrip.seen[n_seen:]))
    assert (L.ticks, L.nights, w.t, len(w.shown), len(w.applied), run.lived) == (200, 1, 200, 200, 200, 200), (L.ticks, L.nights, w.t, len(w.shown), len(w.applied))
    assert [f.tick for f, _ in w.shown] == list(range(200)) and w.lapses == 0
    for i, (f, acts, last, felt, heard, said, opened, touch, seen) in enumerate(rec):
        assert w.applied[i] == acts and set(acts) == {"voice", "arm", "grip"}, (i, acts)
        assert acts["arm"] == last["arm"]["act"] and acts["grip"] == last["grip"]["act"], (i, acts, last)
        assert said == (TOK.decode([acts["voice"]]) if acts["voice"] != L.sil else ""), (i, said, acts)
        u = f.obs.get("ear")
        assert heard == (TOK.decode([u]) if u is not None else ""), (i, heard, u)
        assert felt == (2 if f.tick % 45 == 30 else 0), (i, felt)
        if opened:
            assert torch.equal(touch, torch.tensor(f.obs["touch"], dtype=torch.float32)), (i, touch, f.obs["touch"])
        assert seen and all(fr_ is f for _, fr_ in seen) and seen[0][0] == "gate", f"tick {i}: the grip read another frame than the tick's"
        assert any(k_ == "cost" for k_, _ in seen) == (acts["grip"] != 1), (i, seen, acts)
    assert w.log == [("pause", 119), ("resume", 119)] and L.nights == 1, w.log
    heard_n = sum(1 for r_ in rec if r_[4]); opened_n = sum(1 for r_ in rec if r_[6]); felt_n = sum(1 for r_ in rec if r_[3])
    arm_n = sum(1 for r_ in rec if r_[1]["arm"] != 12); grip_n = sum(1 for r_ in rec if r_[1]["grip"] != 1)
    assert heard_n >= 30 and opened_n >= 100 and felt_n >= 4 and arm_n >= 10 and grip_n >= 10, (heard_n, opened_n, felt_n, arm_n, grip_n)
    moved = [k for k, v in L.m.named_parameters() if not torch.equal(v, p0[k])]
    assert not moved, f"with every learning rate at 0 these moved: {moved}"
    # a frame handed to the tick is the frame it lives
    fr = Frame(999, {"ear": TOK.token_to_id("z"), "touch": [0.5, -0.5]}, 0.0, {"who": "parent"})
    n_shown = len(w.shown); last_pos = L.win[-1]
    got = L.tick(fr)
    assert len(w.shown) == n_shown and w.now is fr and set(got) == {"voice", "arm", "grip"} and L.page[-2][0] == "z"
    assert L.win[-1] is not last_pos and torch.equal(L.win[-1]["touch"], torch.tensor([0.5, -0.5]))
    # the world is not saved with the body; a loaded body lives on in the world given it
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    try:
        L.save(path)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        w2 = _stub_world(); w2.load_state(w.save_state())
        C = Life.load(path, _WorldArm(TOK, L.cfg), save_path=None, world=w2)
    finally:
        os.remove(path)
    assert not any("world" in str(k_) for k_ in blob["life"]) and (w2.t, w2.x) == (w.t, w.x) and C.world is w2
    run2 = WorldLoop(C)
    for _ in range(10):
        run2.step()
    assert C.ticks == L.ticks + 10 and w2.t == w.t + 10 and len(w2.applied) == 10
    print(f"anatomy 21: SimWorld cannot be made unwritten; a stub world ticks a body of frames 200 ticks, a night inside, every learning",
          f"rate at 0: {heard_n} world symbols heard, {felt_n} smiles felt, the touch in {opened_n} opened positions, the arm acting",
          f"{arm_n} and the grip {grip_n} times, each act to the world as recorded, every frame the tick's; paused at dusk, resumed where",
          "it stood; no parameter moved; a frame handed in is lived; the world not saved, a loaded body lives on in the world given it")


def _born_in(anatomy, cfg, world):
    return Life.birth(anatomy, device="cpu", d=64, layers=2, heads=2, window=32, cfg=cfg, seed=0, world=world)


def test_the_loop_deadline_and_pace():
    """anatomy 22 (step R9): THE DEADLINE SWITCH is off by default: however long a tick takes the world waits (no tick lapses, one
    world tick a tick lived); it needs the world's period; on (a test switch), the world runs a tick a period on its own clock: at
    every pass the body lives the tick that is due (the ticks the world ran past lapse unseen, a body ahead of the clock waits), and
    the night stops the clock (nothing lapses for it). THE PACE LOG: a line every `every` ticks (their seconds inside the tick, the wall
    seconds from start to start, the ticks over the period) and one each morning (the night's seconds), the night and the interval
    across it left out of the tick lines; a log that cannot be written is said once and never stops the loop; a life with the log is
    the life without it"""
    from body.core.world import PaceLog, WorldLoop
    tiny = dict(wake_ticks=100000, wake_every=8, gate_every=8, night_rounds=1, night_starts=4, night_batch=4, rem_dreams=2, rem_steps=2,
                write_floor=1e-30)
    # lockstep: every tick five periods long, and the world waits
    now = [0.0]; w = _stub_world(now); w.cost = [5.0] * 30
    torch.manual_seed(5); L = _born_in(_WorldArm(TOK, tiny), tiny, w)
    run = WorldLoop(L, period=1.0, clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    for _ in range(30):
        run.step()
    assert not run.deadline and run.lapsed == 0 and w.lapses == 0 and w.t == 30 and now[0] == 150.0
    try:
        WorldLoop(L, deadline=True)
    except ValueError:
        pass
    else:
        raise AssertionError("the deadline switch ran without the world's period")
    # the deadline: ticks of scripted lengths (in periods), a night inside at the thirtieth
    cost = [0.5, 2.5, 0.2, 0.3, 3.7, 1.0, 0.1, 4.2, 0.4, 0.6, 1.5, 0.05, 2.0, 0.9] * 4
    now = [0.0]; w = _stub_world(now); w.cost = list(cost); w.night_s = 500.0
    cfg = dict(tiny, wake_ticks=30)
    torch.manual_seed(5); L = _born_in(_WorldArm(TOK, cfg), cfg, w)
    run = WorldLoop(L, period=1.0, deadline=True, clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    for _ in range(len(cost)):
        run.step()
    dusk_t = w.shown[29][0].tick; morning = 30; t_m = w.shown[morning][1]
    assert L.nights == 1 and w.log == [("pause", dusk_t), ("resume", dusk_t)] and w.shown[morning][0].tick == dusk_t + 1, (L.nights, w.log, dusk_t)
    for i, (f, t_) in enumerate(w.shown):
        if i < morning:
            assert f.tick == int(t_ // 1.0), f"pass {i}: the body lived the world's tick {f.tick} at {t_}, the tick due is {int(t_)}"
        else:
            assert f.tick - w.shown[morning][0].tick == int((t_ - t_m) // 1.0), (i, f.tick, t_)
    assert len(w.shown) == len(cost) and run.lapsed == w.lapses and w.t == run.lived + run.lapsed and run.lapsed >= 10, (run.lapsed, w.lapses)
    n_dead, n_lapsed = len(w.shown), run.lapsed
    # a quick body waits for the world's clock (every tick a tenth of a period)
    now = [0.0]; w = _stub_world(now); w.cost = [0.1] * 12
    torch.manual_seed(5); L = _born_in(_WorldArm(TOK, tiny), tiny, w)
    run = WorldLoop(L, period=1.0, deadline=True, clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    for _ in range(12):
        run.step()
    assert run.lapsed == 0 and [t_ for _, t_ in w.shown] == [float(k) for k in range(12)], [t_ for _, t_ in w.shown]
    # the pace log: a line every 5 ticks and one each morning, the night left out
    import json
    d_ = tempfile.mkdtemp(); path = os.path.join(d_, "logs", "pace.jsonl")
    try:
        cost = [0.1, 0.2, 0.05, 0.3, 0.1, 0.12, 0.08, 0.4, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        now = [0.0]; w = _stub_world(now); w.cost = list(cost); w.night_s = 40.0
        cfg = dict(tiny, wake_ticks=12)
        torch.manual_seed(5); L = _born_in(_WorldArm(TOK, cfg), cfg, w)
        pace = PaceLog(path, 0.15, every=5, stamp=lambda: 7.0)
        run = WorldLoop(L, period=0.15, pace=pace, clock=lambda: now[0])
        starts = []
        for _ in range(len(cost)):
            t0 = now[0]; starts.append(t0); run.step()
            now[0] += max(0.0, 0.15 - (now[0] - t0))            # the serve's padding to its period
        rows = [json.loads(l_) for l_ in open(path)]
    finally:
        import shutil
        shutil.rmtree(d_, ignore_errors=True)
    assert L.nights == 1 and [r_["kind"] for r_ in rows] == ["ticks", "ticks", "night", "ticks"], rows
    night_i = 11                                                    # the twelfth tick holds the night
    awake = [i for i in range(len(cost)) if i != night_i]
    for r_, idx in zip([r_ for r_ in rows if r_["kind"] == "ticks"], [awake[0:5], awake[5:10], awake[10:15], awake[15:19]]):
        if len(idx) < 5:
            break
        busy = sum(cost[i] for i in idx)
        gaps = [starts[i] - starts[i - 1] for i in idx if i - 1 >= 0 and i - 1 != night_i and i != night_i + 1 and (i - 1) in awake]
        assert r_["n"] == 5 and abs(r_["busy"] - busy) < 1e-6 and r_["over"] == sum(1 for i in idx if cost[i] > 0.15), (r_, busy)
        assert r_["periods"] == len(gaps) and abs(r_["wall"] - sum(gaps)) < 1e-6 and r_["period"] == 0.15 and r_["t"] == 7.0, (r_, gaps)
    nr = rows[2]
    assert nr["night"] == 1 and nr["tick"] == 12 and abs(nr["seconds"] - (40.0 + cost[night_i])) < 1e-6, nr
    bad = PaceLog(tempfile.gettempdir(), 0.15, every=1)               # a directory: no line can be written
    import contextlib
    import io
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        for k in range(3):
            bad.record(float(k), float(k) + 0.1, k, 0)
    assert bad.failed == 3 and bad.lines == 0 and said.getvalue().count("[pace]") == 1, said.getvalue()
    # a life with the pace log is the life without it (logging only)
    cfg = dict(tiny, wake_ticks=60)
    lives = []
    for with_pace in (False, True):
        torch.manual_seed(5); D = _born(TOK, cfg)
        fd, p_ = tempfile.mkstemp(suffix=".jsonl"); os.close(fd)
        try:
            run = WorldLoop(D, period=0.15, pace=PaceLog(p_, 0.15, every=10) if with_pace else None)
            for t in range(90):
                if t % 30 == 0:
                    D.type_text(("what do you want?", "I want milk", "go up")[t // 30], who="parent")
                run.step()
            n_lines = sum(1 for _ in open(p_))
        finally:
            os.remove(p_)
        lives.append((D, n_lines))
    (D0, n0), (D1, n1) = lives
    assert n0 == 0 and n1 == 9 and _differs(D0, D1) is None and _whole(D0) == _whole(D1), (n0, n1)
    print(f"anatomy 22: lockstep, the world waits (30 ticks of 5 periods, none lapsed); the deadline switch needs a period; on, the body",
          f"lives the due tick at each of {n_dead} passes ({n_lapsed} world ticks lapsed unseen), a quick body waits for the clock, the night",
          f"stops it; the pace log: {len(rows)} lines, the night's {nr['seconds']} s left out of the tick lines; a failing log said once;",
          "the life the same with the log and without")


# ---------------- step R6: the motor timing part and the learned stops ----------------

_STEPS = (-0.27, -0.09, 0.0, 0.09, 0.27)
_LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0,
            night_rounds=1, night_starts=4, night_batch=4, rem_dreams=2, rem_steps=2)


def _arm_world(wall=None, words=True):
    """a stub of the simulated world for the motor timing part: a planar arm of two joints, each tick moved by the arm's act (each
    joint's setting one of five steps, -0.27 to 0.27 rad); its body sense the joints' velocities (rad/s over the 0.15 s tick); a touch
    of [pain, the parent's hand on the forearm]; the parent's hand moves the resting arm through the acts queued in `guide` (a
    demonstration: the body is told nothing, it feels the hand and the motion); an optional wall on joint 0 at `wall` rad that stops
    the arm there and hurts; a short line on the words now and then (none without `words`). It records every frame it shows and every move (the act that
    moved the arm and who made it: "own" for the body's act to the world, "guide" for the parent's hand, "still")"""
    from body.core.world import SimWorld

    class ArmWorld(SimWorld):
        LINE = "up we go "

        def __init__(self):
            self.t = 0; self.q = [0.0, 0.0]; self.v = [0.0, 0.0]; self.pain = 0.0; self.hand = 0.0; self.paused = False
            self.guide = []; self.moved = []; self.shown = []; self.log = []

        def frame(self):
            assert not self.paused, "a frame taken while the world is paused"
            obs = {"body": [self.v[0], self.v[1]], "touch": [self.pain, self.hand]}
            k = self.t % 60 - 5
            if words and 0 <= k < len(self.LINE):
                obs["ear"] = TOK.token_to_id(self.LINE[k])
            f = Frame(self.t, obs, 0.0, {"who": "parent", "q": list(self.q)})
            self.shown.append(f); return f

        def apply(self, acts):
            assert not self.paused, "the world moved while paused"
            a = int(acts.get("arm", 12)); who = "own" if a != 12 else "still"; self.hand = 0.0
            if a == 12 and self.guide:
                a = int(self.guide.pop(0)); who = "guide"; self.hand = 1.0
            q0 = self.q[0] + _STEPS[a // 5]; q1 = self.q[1] + _STEPS[a % 5]; self.pain = 0.0
            if wall is not None and q0 > wall:
                q0 = wall; self.pain = 1.0
            self.v = [(q0 - self.q[0]) / 0.15, (q1 - self.q[1]) / 0.15]; self.q = [q0, q1]
            self.moved.append((a, who)); self.t += 1

        def pause(self):
            self.paused = True; self.log.append(("pause", self.t))

        def resume(self):
            self.paused = False; self.log.append(("resume", self.t))

        def save_state(self):
            return pickle.dumps((self.t, self.q, self.v, self.pain, self.hand, self.guide))

        def load_state(self, blob):
            self.t, self.q, self.v, self.pain, self.hand, self.guide = pickle.loads(blob)
    return ArmWorld()


class _Reach(Effector):
    """the tiny arm's reach with a withdrawal reflex (SIM_DESIGN.md 5.5, in small): on a tick of pain it reverses its last move for 2
    ticks (the reflex's count kept in its working state)"""

    def reflex(self, frame, life, state):
        if float(frame.obs["touch"][0]) > 0.5 and not state.get("rfx_n", 0):
            last = int(state["now"]["world"]) if state["now"] is not None else int(self.rest_id)
            state["rfx_act"] = (4 - last // 5) * 5 + (4 - last % 5); state["rfx_n"] = 2
        if state.get("rfx_n", 0):
            state["rfx_n"] -= 1
            return state["rfx_act"]
        return None


class _Timed(LanguageAnatomy):
    """the diary's words and face, the arm's body sense (the joints' velocities) and a touch (pain, the parent's hand), each a channel
    of the world's frames encoded by the face's map; an arm of two joints of five (its rest 12, both held) sensing its body, with an
    inverse model and (reflex) the withdrawal reflex; a grip of three (its rest 1) sensing the pain alone (touch's number 0), no inverse
    model, its end act `grip_end`; a tap of four (its rest 1, its act 3 reserved) with no body sense"""

    def __init__(self, tok, cfg=None, reflex=True, grip_end=None):
        super().__init__(tok, cfg)
        self.channels += [Channel("body", "vector", 2, organ="face_in"), Channel("touch", "vector", 2, organ="face_in")]
        self.effectors += [(_Reach if reflex else Effector)("arm", [5, 5], rest_id=12, effort=0.05, sense="body", inverse=True, inv_hidden=32),
                           Effector("grip", [3], rest_id=1, effort=0.03, sense="touch", sense_idx=[0], end_id=grip_end),
                           Effector("tap", [4], rest_id=1, reserved=[3], effort=0.02)]


def _row(L, name, act):
    """an act's row in its effector's table"""
    return L.m.acts[name](torch.tensor(int(act))).detach().clone()


def test_the_timing_part_is_built_last():
    """anatomy 23 (step R6; SIM_DESIGN.md 5.4, 5.8 and 8.3): each later effector's motor timing part (timing.<name>: act_pred, and the
    forward half with its correction where the effector declares a body sense, and act_inv where it declares one) is built after
    every other organ, from a generator of its own seeded by the body's seed: the diary's organs and the effectors' tables, gates and
    actors are born as before beside it, the global random stream is left where it was; act_pred and the forward half born unsure,
    the correction at zero, act_inv sized by its declaration. The declaration is checked (a sense that is no vector channel, its
    numbers out of range, an inverse model with no sense, a voice with a sense) and organs that do not match it are refused. The
    language body has none of it: no timing organ, no optimizer, no motor state, no key in its cfg or its save; the motor constants are
    absent from a body's cfg unless given, and known when given"""
    from body.model import ActTable
    from body.core.physiology import MOTOR, SWITCHES
    a = _Timed(TOK, {}); V = a.vocab
    assert a.check() is a and [a.sense_size(e) for e in a.effectors] == [0, 2, 1, 0]
    torch.manual_seed(9); o1 = Organs(V, d=64, layers=2, heads=2, window=32); r1 = torch.get_rng_state()
    torch.manual_seed(9); o3 = Organs(V, d=64, layers=2, heads=2, window=32, channels=a.channels, effectors=a.effectors, born_seed=3); r3 = torch.get_rng_state()
    torch.manual_seed(1); o4 = Organs(V, d=64, layers=2, heads=2, window=32, channels=a.channels, effectors=a.effectors, born_seed=3)
    torch.manual_seed(9); o5 = Organs(V, d=64, layers=2, heads=2, window=32, channels=a.channels, effectors=a.effectors, born_seed=4)
    s1, s3, s4, s5 = o1.state_dict(), o3.state_dict(), o4.state_dict(), o5.state_dict()
    assert torch.equal(r1, r3) and [k for k in s3 if k in s1] == list(s1) and all(torch.equal(s1[k], s3[k]) for k in s1), "the diary's organs are built otherwise"
    new = [k for k in s3 if k not in s1]
    tim = [k for k in new if k.startswith("timing.")]
    assert new[:len(new) - len(tim)] == ["stri_mline", "acts.arm.rows", "acts.grip.rows", "acts.tap.rows", "gates.arm.weight", "gates.arm.bias",
                                          "gates.grip.weight", "gates.grip.bias", "gates.tap.weight", "gates.tap.bias", "actors.arm.weight",
                                          "actors.arm.bias", "actors.grip.weight", "actors.grip.bias", "actors.tap.weight", "actors.tap.bias"], new
    assert tim == ["timing.arm.pred.weight", "timing.arm.pred.bias", "timing.arm.fwd.weight", "timing.arm.fwd.bias", "timing.arm.cor.weight",
                   "timing.arm.inv.0.weight", "timing.arm.inv.0.bias", "timing.arm.inv.2.weight", "timing.arm.inv.2.bias",
                   "timing.grip.pred.weight", "timing.grip.pred.bias", "timing.grip.fwd.weight", "timing.grip.fwd.bias", "timing.grip.cor.weight",
                   "timing.tap.pred.weight", "timing.tap.pred.bias"] and new[-len(tim):] == tim, tim
    # the effectors' tables as step R5 drew them (their generator's draws, in their order, untouched by the timing part's)
    g_ = torch.Generator().manual_seed(3 + 15485863)
    for e in a.effectors[1:]:
        assert torch.equal(o3.acts[e.name].rows, ActTable(e.factors, 64, g_).rows), e.name
    assert all(torch.equal(s3[k], s4[k]) for k in tim) and not any(torch.equal(s3[k], s5[k]) for k in tim if "weight" in k and "cor" not in k), \
        "the timing part is not the body's seed's alone"
    t_ = o3.timing["arm"].requires_grad_(False)
    assert float(t_.pred.weight.std()) < 1e-3 and float(t_.pred.bias.abs().max()) == 0.0 and float(t_.fwd.weight.std()) < 1e-3
    assert float(t_.cor.weight.abs().max()) == 0.0 and t_.fwd.weight.shape == (2, 64) and t_.cor.weight.shape == (64, 2)
    assert t_.inv[0].weight.shape == (32, 4) and t_.inv[2].weight.shape == (10, 32) and (t_.factors, t_.sense_n, t_.inverse) == ((5, 5), 2, True)
    assert not hasattr(o3.timing["grip"], "inv") and not hasattr(o3.timing["tap"], "fwd") and o3.timing["grip"].sense_n == 1
    lg = t_.inverse_logits(torch.randn(3, 2), torch.randn(3, 2))
    assert [x.shape for x in lg] == [(3, 5), (3, 5)]
    t_.requires_grad_(True)
    # the declaration checked
    bad = {"a sense on a symbol channel": dict(sense="ear"), "a sense on no channel": dict(sense="nose"), "numbers out of range": dict(sense_idx=[0, 2]),
           "numbers twice": dict(sense_idx=[1, 1]), "numbers of no sense": dict(sense=None, inverse=False, sense_idx=[0]),
           "an inverse model with no sense": dict(sense=None), "an inverse model of no unit": dict(inv_hidden=0)}
    for what, ch in bad.items():
        b = _Timed(TOK, {})
        for k_, v_ in ch.items():
            setattr(b.effectors[1], k_, v_)
        try:
            b.check()
        except ValueError:
            continue
        raise AssertionError(f"the anatomy took {what}")
    b = _Timed(TOK, {}); b.effectors[0].sense = "body"
    try:
        b.check()
    except ValueError:
        pass
    else:
        raise AssertionError("the anatomy took a voice with a body sense")
    # organs that do not match the declaration
    cfg = dict(wake_ticks=100000)
    refused = []
    other = _Timed(TOK, cfg); other.effectors[1].inv_hidden = 16
    blind = _Timed(TOK, cfg); blind.effectors[2].sense = None; blind.effectors[2].sense_idx = None
    noinv = _Timed(TOK, cfg); noinv.effectors[1].inverse = False
    o6 = Organs(V, d=64, layers=2, heads=2, window=32, channels=a.channels, effectors=a.effectors, born_seed=3)
    del o6.timing["tap"]
    for what, call in (("an inverse model of other units", lambda: Life(o3, other)), ("a sense the effector does not declare", lambda: Life(o3, blind)),
                       ("an inverse model the effector does not declare", lambda: Life(o3, noinv)), ("no timing part", lambda: Life(o6, _Timed(TOK, cfg)))):
        try:
            call()
        except ValueError:
            refused.append(what); continue
        raise AssertionError(f"the life took {what}")
    # the language body has none of it
    D = _born(TOK, cfg)
    assert not hasattr(D.m, "timing") and not hasattr(D, "opt_inv") and not hasattr(D, "opt_pred") and not hasattr(D, "motor")
    assert not any(k in D.cfg for k in MOTOR) and not any(k in PHYSIOLOGY or k in SWITCHES for k in MOTOR)
    torch.manual_seed(9); o7 = Organs(V, d=64, layers=2, heads=2, window=32, channels=LanguageAnatomy(TOK, {}).channels,
                                      effectors=LanguageAnatomy(TOK, {}).effectors); r7 = torch.get_rng_state()
    assert torch.equal(r7, r1) and list(o7.state_dict()) == list(s1) and not hasattr(o7, "timing")
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    try:
        _live(D, ticks=20); D.save(path); blob = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        os.remove(path)
    assert "motor" not in blob["life"] and not any(k.startswith("timing.") for k in blob["organs"])
    import contextlib
    import io
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        L = _born(_Timed(TOK, dict(cfg, act_inv_lr=0.01, act_inv_tau=500)), dict(cfg, act_inv_lr=0.01, act_inv_tau=500))
    assert "unknown" not in said.getvalue() and L.cfg["act_inv_lr"] == 0.01 and L._motor_const("act_inv_tau") == 500
    assert abs(L.opt_inv.param_groups[0]["lr"] - 0.01) < 1e-12 and [id(p_) for p_ in L.opt_inv.param_groups[0]["params"]] == [id(p_) for p_ in L.m.timing["arm"].inv.parameters()]
    L2 = _born(_Timed(TOK, cfg), cfg)
    assert L2._motor_const("act_inv_lr") == MOTOR["act_inv_lr"] and "act_inv_lr" not in L2.cfg
    print(f"anatomy 23: the timing part built last from the body's seed ({len(tim)} tensors: act_pred for three effectors, the forward",
          f"half for two, act_inv for the arm), the diary's organs and the effectors' tables as before, the global stream untouched;",
          f"the declaration checked ({len(bad) + 1} faults refused); refused: {'; '.join(refused)}; the language body has none of it")


def test_act_inv_learns_online():
    """anatomy 24 (step R6; SIM_DESIGN.md 5.4): act_inv learns online from birth on the body's own acts, the efference copy the label:
    one lesson on the tick after each own act, on the pair of body senses the act moved between (the frames the world showed) and the
    act the world applied; never after a rest, a demonstration (the parent's hand) or a night; the grip, with no inverse model, none.
    Its reliability is Cohen's kappa per joint over its running confusion, the label it gave before each lesson against the efference
    copy, the joints' mean clipped (recomputed here, equal to the bit), zero until 64 acts. With every other learning rate at 0 only
    act_inv moves. It learns: its labels right on nearly every joint of the last own acts, its reliability high. The reliability (the
    confusion, the kappas) survives the night and a save and a load"""
    from body.core.world import WorldLoop
    cfg = dict(_LR0, wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.85, fast_rls=0, act_inv_lr=1e-2, act_inv_tau=2000)
    w = _arm_world(); torch.manual_seed(5); L = _born_in(_Timed(TOK, cfg), cfg, w)
    p0 = {k: v.detach().clone() for k, v in L.m.named_parameters()}
    lessons = []; pairs = []
    f_les, f_rel = L._inverse_lesson, L._inv_rel_update

    def les(i, s0, s1, act):
        with torch.no_grad():
            lab = [int(x.argmax()) for x in L.m.timing["arm"].inverse_logits(s0, s1)]
        out = f_les(i, s0, s1, act)
        lessons.append((L.ticks, i, s0.clone(), s1.clone(), act, lab, L.motor[0]["inv_gain"])); return out

    def rel(e, st, label, true):
        pairs.append((e.name, list(label), list(true))); return f_rel(e, st, label, true)
    L._inverse_lesson = les; L._inv_rel_update = rel
    run = WorldLoop(L); rec = []
    for t in range(1500):
        if t % 50 == 20:
            w.guide.extend([7, 13, 8])                            # the parent's hand, now and then, on the resting arm
        run.step(); rec.append(dict(L.last["acts"]["arm"]))
    guided = [t for t, (a_, who) in enumerate(w.moved) if who == "guide"]
    want = [t + 1 for t in range(len(rec) - 1) if rec[t]["acted"]]
    assert [l_[0] for l_ in lessons] == want and len(guided) >= 20, (len(lessons), len(want), len(guided))
    for tk, i, s0, s1, act, lab, g_ in lessons:
        t = tk - 1
        assert i == 1 and act == rec[t]["act"] == w.moved[t][0] and w.moved[t][1] == "own", (tk, act, rec[t], w.moved[t])
        assert torch.equal(s0, torch.tensor(w.shown[t].obs["body"], dtype=torch.float32)) and torch.equal(s1, torch.tensor(w.shown[t + 1].obs["body"], dtype=torch.float32))
    assert not any(t + 1 in set(want) for t in guided), "a demonstration was taken for an own act"
    # the reliability: Cohen's kappa per joint over the running confusion of the labels given before each lesson, recomputed
    assert [p_[1] for p_ in pairs] == [l_[5] for l_ in lessons] and all(p_[0] == "arm" for p_ in pairs)
    assert all(p_[2] == [l_[4] // 5, l_[4] % 5] for p_, l_ in zip(pairs, lessons))
    conf = [[[0.0] * 5 for _ in range(5)] for _ in range(2)]; d_ = 1.0 - 1.0 / 2000.0; gains = []
    for _, label, true in pairs:
        kap = []
        for j in range(2):
            Cj = conf[j]
            for r_ in Cj:
                for k in range(5):
                    r_[k] *= d_
            Cj[label[j]][true[j]] += 1.0
            n = sum(sum(r_) for r_ in Cj); po = sum(Cj[k][k] for k in range(5)) / n
            pe = sum(sum(Cj[k]) * sum(Cj[r][k] for r in range(5)) for k in range(5)) / (n * n)
            kap.append((po - pe) / (1.0 - pe) if 1.0 - pe > 1e-9 else 0.0)
        gains.append((sum(max(0.0, min(1.0, k)) for k in kap) / 2.0 if n > 64 else 0.0, [k if n > 64 else 0.0 for k in kap]))
    st = L.motor[0]
    assert st["inv_conf"] == conf and (st["inv_gain"], st["inv_kappa"]) == gains[-1] and [l_[6] for l_ in lessons] == [g_[0] for g_ in gains]
    warm = next(k for k, g_ in enumerate(gains) if g_[1] != [0.0, 0.0])
    assert all(g_ == (0.0, [0.0, 0.0]) for g_ in gains[:warm]) and warm == 65, (warm, gains[60:68])   # n > 64 at the 66th act (tau 2000)
    assert st["inv_n"] == len(lessons) and L.motor[1]["inv_n"] == 0 and L.motor[1]["inv_conf"] is None and L.motor[1]["inv_kappa"] == [0.0]
    assert not hasattr(L.m.timing["grip"], "inv")
    # only act_inv moved
    moved = sorted(k for k, v in L.m.named_parameters() if not torch.equal(v, p0[k]))
    assert moved == ["timing.arm.inv.0.bias", "timing.arm.inv.0.weight", "timing.arm.inv.2.bias", "timing.arm.inv.2.weight"], moved
    # it learned
    hits = [sum(1 for l_ in lessons[-100:] if l_[5][j] == [l_[4] // 5, l_[4] % 5][j]) / 100.0 for j in (0, 1)]
    first = [sum(1 for l_ in lessons[:50] if l_[5][j] == [l_[4] // 5, l_[4] % 5][j]) / 50.0 for j in (0, 1)]
    assert min(hits) >= 0.95 and st["inv_gain"] >= 0.8 and max(first) < 0.8, (first, hits, st["inv_gain"])   # kappa over tau 2000 acts: the early labels still weigh
    rep = L.insides()["effectors"]["arm"]["timing"]
    assert rep["inv_gain"] == round(st["inv_gain"], 3) and rep["inv_n"] == len(lessons)
    # the night keeps the reliability and leaves no pair across it (the sleep switch's night, then a night by hand); a save and a load
    # keep it
    L.cfg["wake_ticks"] = L.sleep_pressure + 3; n_les = len(lessons); kept = []
    f_night = L.night

    def night_spy():
        kept.append((copy.deepcopy(st["inv_conf"]), st["inv_gain"], list(st["inv_kappa"]), st["inv_n"])); out = f_night()
        kept.append((copy.deepcopy(st["inv_conf"]), st["inv_gain"], list(st["inv_kappa"]), st["inv_n"])); return out
    L.night = night_spy
    while L.nights == 0:
        run.step(); rec.append(dict(L.last["acts"]["arm"]))
    dusk = L.ticks - 1; L.cfg["wake_ticks"] = 100000
    assert len(kept) == 2 and kept[0] == kept[1] and st["sense"] is None and st["now"] is None and not (L.last_night or {}).get("error")
    for _ in range(6):
        run.step(); rec.append(dict(L.last["acts"]["arm"]))
    assert len(lessons) > n_les and all(l_[0] != dusk + 1 for l_ in lessons[n_les:]), "a lesson on the pair across the night"
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    try:
        L.save(path); blob = torch.load(path, map_location="cpu", weights_only=False)
        w2 = _arm_world(); w2.load_state(w.save_state())
        C = Life.load(path, _Timed(TOK, L.cfg), save_path=None, world=w2)
    finally:
        os.remove(path)
    assert sorted(blob["life"]["motor"]) == ["arm", "grip", "tap"] and blob["life"]["motor"]["arm"]["inv_conf"] == st["inv_conf"]
    assert blob["life"]["motor"]["grip"]["inv_conf"] is None and C.motor[1]["inv_conf"] is None
    assert C.motor[0]["inv_conf"] == st["inv_conf"] and (C.motor[0]["inv_gain"], C.motor[0]["inv_kappa"], C.motor[0]["inv_n"]) == (st["inv_gain"], st["inv_kappa"], st["inv_n"])
    sL, sC = L.m.state_dict(), C.m.state_dict()
    assert all(torch.equal(sL[k], sC[k]) for k in sL if k.startswith("timing.")) and C.opt_inv is not None
    # a save of R6's first form (the critics' running moments under "inv_m"): loaded, said once, the reliability earned again
    import contextlib
    import io
    old_ = copy.deepcopy(blob)
    old_["life"]["motor"] = {k_: {"inv_m": [1.0] * 6, "inv_gain": 0.5, "inv_corr": 0.5, "inv_n": v_["inv_n"]} for k_, v_ in blob["life"]["motor"].items()}
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd); said = io.StringIO()
    try:
        torch.save(old_, path)
        with contextlib.redirect_stdout(said):
            O = Life.load(path, _Timed(TOK, L.cfg), save_path=None, world=_arm_world())
    finally:
        os.remove(path)
    assert O.motor[0]["inv_conf"] == [[[0.0] * 5 for _ in range(5)] for _ in range(2)] and O.motor[0]["inv_gain"] == 0.0 and O.motor[0]["inv_n"] == st["inv_n"]
    assert said.getvalue().count("act_inv reliability") == 1 and "earned again" in said.getvalue(), said.getvalue()
    run2 = WorldLoop(C)
    for _ in range(5):
        run2.step()
    print(f"anatomy 24: act_inv learned online on {len(lessons)} own acts, one lesson the tick after each (never after a rest or one of",
          f"{len(guided)} demonstrations), its pair the frames' senses; its reliability Cohen's kappa per joint over its running confusion,",
          f"equal to the bit, zero for the first {warm} acts; only act_inv moved; its labels right on {first} of the first 50 joints and",
          f"{hits} of the last 100, its kappas {[round(k_, 3) for k_ in st['inv_kappa']]}, its reliability {st['inv_gain']:.3f}; kept through",
          f"the night (no pair across it) and a save and a load")


def _timing_ref(L, i, C, obs, gain):
    """act_pred's and the forward half's lesson recomputed position by position (the reference): returns (the proposal's loss, the
    forward loss, the targets, the weights, each position's kind). The weighted errors averaged over the positions (the weights
    absolute; the reference shared the lesson's old division by the weights' sum until the R6 verifier's first finding)"""
    e = L.anatomy.effectors[i]; tm = L.m.timing[e.name]; tab = L.m.get_submodule(e.organ)
    acts = [int(x) for x in obs[e.name]]; T = len(acts); rest = int(e.rest_id)
    s = obs[e.sense][:, list(e.sense_idx) if e.sense_idx is not None else slice(None)].float() if e.sense is not None else None
    num = torch.zeros(()); tg = []; ws = []; kinds = []
    with torch.no_grad():
        for t in range(1, T):
            p = tm.pred(C[t - 1])
            if s is not None:
                p = p + tm.cor(s[t] - tm.fwd(C[t - 1]))
            if acts[t] != rest:
                tgt, wt, kd = acts[t], 1.0, "own"
            elif e.inverse and t < T - 1:
                tgt, wt, kd = tab.flat([int(x.argmax()) for x in tm.inverse_logits(s[t], s[t + 1])]), float(gain), "read"
            elif e.inverse:
                tgt, wt, kd = rest, 0.0, "last"
            else:
                tgt, wt, kd = rest, 1.0, "rest"
            num = num + wt * 0.5 * ((p - tab(torch.tensor(tgt))) ** 2).sum()
            tg.append(tgt); ws.append(wt); kinds.append(kd)
        lf = None
        if s is not None:
            lf = sum(0.5 * float(((tm.fwd(C[t]) - s[t + 1]) ** 2).sum()) for t in range(T - 1)) / (T - 1)
    return float(num) / (T - 1), lf, tg, ws, kinds


def test_act_pred_targets():
    """anatomy 25 (step R6; SIM_DESIGN.md 5.4): act_pred's lesson over the waking window, recomputed position by position: the proposal
    at position t from the stream at t-1 and the forward half's error at t; its target the act at t where the effector acted (its
    efference copy, weight 1); where it rested (still, moved by the parent's hand, or by its reflex), act_inv's label for the senses at
    t and t+1 at act_inv's reliability (none at the last position, whose next sense is not felt yet); for the grip and the tap (no
    inverse model) their rest, weight 1. The window holds a reflex's tick as a rest (the world moved by the reflex's act). Once act_inv
    has learned, its label at a demonstrated position is the act the parent's hand made. The lesson's gradient reaches act_pred, the
    correction, the forward half and the cortex, never act_inv; the cortex's the same at every reliability (act_inv's labels reach
    act_pred and the correction alone: the R6 verifier's fourth look, anatomy 31). A window in which the world never spoke still
    teaches them; and the lesson teaches: on a window held fixed act_pred's and the forward half's errors fall and act_pred's best
    guesses come to match"""
    from body.core.world import WorldLoop
    cfg = dict(_LR0, wake_ticks=100000, wake_every=10 ** 6, gate_every=8, write_floor=1e-30, gate_floor=0.5, fast_rls=0, act_inv_lr=1e-2, act_inv_tau=2000)
    w = _arm_world(wall=0.45); torch.manual_seed(5); L = _born_in(_Timed(TOK, cfg), cfg, w)
    run = WorldLoop(L); rec = []
    for t in range(1600):
        if t >= 1600 - 40:
            w.guide[:] = [[6, 7, 11, 13, 8, 2][t % 6]]                 # the parent's hand on every resting tick of the last forty (never into the wall)
        elif t % 40 == 5:
            w.guide.extend([6, 11])
        run.step(); rec.append(dict(L.last["acts"]["arm"]))
    rfx = [t for t, r_ in enumerate(rec) if r_["world"] != r_["act"]]
    assert len(rfx) >= 4 and all(rec[t]["act"] == 12 and not rec[t]["acted"] for t in rfx), len(rfx)
    obs, whos, bundles, reads = L._window_tensors(); T = obs["arm"].shape[0]; t0 = L.ticks - T
    assert T == 32 and [int(x) for x in obs["arm"]] == [rec[t0 + k]["act"] for k in range(T)], "the window's acts are not the efference copies"
    assert torch.equal(obs["body"], torch.tensor([w.shown[t0 + k].obs["body"] for k in range(T)], dtype=torch.float32))
    with torch.no_grad():
        C = L.m.stream(L.m.inputs(L.anatomy, obs, whos, bundles))
    kinds_all = collections.Counter()
    for i in (1, 2, 3):
        for gain in (0.37, 0.0, 1.0):
            if i == 1:
                L.motor[0]["inv_gain"] = gain
            lt, rp, lb = L._timing_loss(i, C, obs)
            lt = lt if lb is None else lt + lb                    # the whole lesson: the own acts' sample and act_inv's labels' (R6 fix 5)
            lp_ref, lf_ref, tg, ws, kinds = _timing_ref(L, i, C, obs, gain if i == 1 else 0.0)
            assert abs(rp["pred"] - round(lp_ref, 4)) <= 1e-4 and (lf_ref is None) == ("fwd" not in rp), (i, gain, rp, lp_ref, lf_ref)
            want = lp_ref + (lf_ref or 0.0)
            assert abs(float(lt.detach()) - want) <= 1e-5 * max(1.0, abs(want)), (i, gain, float(lt.detach()), want)
            if lf_ref is not None:
                assert abs(rp["fwd"] - round(lf_ref, 4)) <= 1e-4, (rp, lf_ref)
            if gain == 0.37:
                kinds_all.update(f"{L.anatomy.effectors[i].name}:{k_}" for k_ in kinds)
    L.motor[0]["inv_gain"] = 0.37
    # a window that ends at a rest: its last position has no label (its next sense not felt yet), weight 0
    k_end = max(k for k in range(T) if int(obs["arm"][k]) == 12 and k >= 8)
    obs_c = {k_: v_[:k_end + 1] for k_, v_ in obs.items()}
    lt, rp, lb = L._timing_loss(1, C[:k_end + 1], obs_c)
    lt = lt if lb is None else lt + lb
    lp_ref, lf_ref, tg, ws, kinds = _timing_ref(L, 1, C[:k_end + 1], obs_c, 0.37)
    assert kinds[-1] == "last" and ws[-1] == 0.0 and abs(float(lt.detach()) - (lp_ref + lf_ref)) <= 1e-5 * max(1.0, lp_ref + lf_ref), (float(lt.detach()), lp_ref, lf_ref)
    kinds_all.update(["arm:last"])
    assert kinds_all["arm:own"] >= 5 and kinds_all["arm:read"] >= 5 and kinds_all["grip:rest"] >= 5 and kinds_all["tap:rest"] >= 5, kinds_all
    # the demonstrations: act_inv, learned, reads the parent's hand's act where the arm rested and was guided (the last forty ticks)
    tm = L.m.timing["arm"]; s = obs["body"]
    dem = [k for k in range(T - 1) if w.moved[t0 + k][1] == "guide"]
    with torch.no_grad():
        right = sum(1 for k in dem if L.m.acts["arm"].flat([int(x.argmax()) for x in tm.inverse_logits(s[k], s[k + 1])]) == w.moved[t0 + k][0])
    assert len(dem) >= 5 and right >= 0.9 * len(dem), (len(dem), right)
    # the gradient: act_pred, the correction, the forward half and the cortex; never act_inv; the cortex's the same at every reliability
    grads = {}
    for gain in (0.0, 0.37, 1.0):
        L.motor[0]["inv_gain"] = gain
        L.m.zero_grad(set_to_none=True)
        C2 = L.m.stream(L.m.inputs(L.anatomy, obs, whos, bundles))
        lt, _, lb = L._timing_loss(1, C2, obs); (lt if lb is None else lt + lb).backward()
        grads[gain] = {k: (v.grad.detach().clone() if v.grad is not None else None) for k, v in L.m.named_parameters()}
    gr = {k: (v is not None and float(v.abs().max()) > 0) for k, v in grads[0.37].items()}
    assert gr["timing.arm.pred.weight"] and gr["timing.arm.cor.weight"] and gr["timing.arm.fwd.weight"] and gr["blocks.0.attn.in_proj_weight"], gr
    assert not any(gr[k] for k in gr if k.startswith("timing.arm.inv.")) and not any(gr[k] for k in gr if k.startswith("timing.grip."))
    cx_ = [k for k in grads[0.37] if not k.startswith("timing.") and grads[0.37][k] is not None]
    assert cx_ and all(torch.equal(grads[g_][k], grads[0.37][k]) for g_ in (0.0, 1.0) for k in cx_), "act_inv's labels reached the cortex"
    assert not torch.equal(grads[0.0]["timing.arm.pred.weight"], grads[1.0]["timing.arm.pred.weight"])
    L.motor[0]["inv_gain"] = 0.37
    L.m.zero_grad(set_to_none=True)
    # the lesson teaches, on a window held fixed (the waking rate raised for the test, the day's and act_pred's, both samples'): 80
    # lessons (60 until R6 fix 5: the own acts, 15 of the window's 31 positions, now step at their share, 0.48 of a step, where the one
    # sample stepped them at the lesson's mean label weight, 0.67 at this reliability; at 60 the best guess matched 13 and 11 of the 15
    # own acts, pinned and threaded, at 80 all 15 in both)
    win0 = list(L.win)
    for g_ in L.opt_day.param_groups + L.opt_pred.param_groups + L.opt_lab.param_groups:
        g_["lr"] = 3e-3
    first = last = None
    for k in range(80):
        L.win.clear(); L.win.extend(win0)
        out = L._wake_lesson()
        if k == 0:
            first = out["motor"]
        last = out["motor"]
    assert last["arm"]["pred"] < 0.5 * first["arm"]["pred"] and last["arm"]["fwd"] < 0.5 * first["arm"]["fwd"] and last["grip"]["pred"] < 0.5 * first["grip"]["pred"], (first, last)
    L.win.clear(); L.win.extend(win0)
    with torch.no_grad():
        C3 = L.m.stream(L.m.inputs(L.anatomy, obs, whos, bundles))
        _, _, tg, ws, kinds = _timing_ref(L, 1, C3, obs, 0.37)
        fw = tm.fwd(C3[:-1]); P = tm.pred(C3[:-1]) + tm.cor(s[1:] - fw)
        best = [L._best_guess(L.anatomy.effectors[1], P[k]) for k in range(T - 1)]
    own_k = [k for k in range(T - 1) if kinds[k] == "own"]
    match = sum(1 for k in own_k if best[k] == tg[k])
    assert match >= 0.8 * len(own_k), (match, len(own_k))
    # a window in which the world never spoke still teaches the motor heads
    w3 = _arm_world(words=False); torch.manual_seed(5); L3 = _born_in(_Timed(TOK, cfg), cfg, w3); run3 = WorldLoop(L3)
    for _ in range(40):
        run3.step()
    out3 = L3._wake_lesson()
    assert out3 is not None and out3["n_world"] == 0 and set(out3["motor"]) == {"arm", "grip", "tap"}, out3
    print(f"anatomy 25: act_pred's lesson recomputed position by position for the arm (at three reliabilities), the grip and the tap:",
          f"{dict(kinds_all)}; {len(rfx)} reflex ticks recorded as rests (the window's acts the efference copies); act_inv read {right} of {len(dem)} demonstrations as the",
          f"parent's hand's act; the gradient reaches act_pred, the correction, the forward half and the cortex (the same at 0, 0.37 and",
          f"1: {len(cx_)} tensors), never act_inv; on a",
          f"fixed window act_pred's error {first['arm']['pred']} -> {last['arm']['pred']}, the forward half's {first['arm']['fwd']} ->",
          f"{last['arm']['fwd']}, the best guess matching {match} of {len(own_k)} own acts; a window with no word teaches them")


def _closed_arm(L):
    """the arm's gate shut (its weights at zero, its bias far below, the spontaneous floor 0): the arm never acts, it only rests"""
    with torch.no_grad():
        L.m.gates["arm"].weight.zero_(); L.m.gates["arm"].bias.fill_(-60.0)
    L.cfg["gate_floor"] = 0.0


def _lesson_at(L, obs, whos, bundles, gain):
    """the arm's timing lesson on a window at act_inv's reliability `gain`: (the loss, act_pred's and the correction's gradient, the
    forward half's gradient, the report)"""
    L.motor[0]["inv_gain"] = gain
    L.m.zero_grad(set_to_none=True)
    C = L.m.stream(L.m.inputs(L.anatomy, obs, whos, bundles))
    lt, rp, lb = L._timing_loss(1, C, obs)
    lt = lt if lb is None else lt + lb                            # the whole lesson: the own acts' sample and act_inv's labels' (R6 fix 5)
    lt.backward()
    tm = L.m.timing["arm"]
    g_pred = torch.cat([tm.pred.weight.grad.flatten(), tm.pred.bias.grad.flatten(), tm.cor.weight.grad.flatten()]).clone()
    g_fwd = torch.cat([tm.fwd.weight.grad.flatten(), tm.fwd.bias.grad.flatten()]).clone()
    L.m.zero_grad(set_to_none=True)
    return float(lt.detach()), g_pred, g_fwd, rp


def test_demonstrations_count_as_earned():
    """anatomy 28 (the R6 verifier's first finding; SIM_DESIGN.md 5.4): act_pred's lesson weighs a rest's label by act_inv's
    reliability ABSOLUTELY, the weighted errors averaged over the window's positions, so a demonstration counts only as far as act_inv
    has earned. On a window of rests alone (the arm's gate shut, nothing moving it) and on a window of pure demonstration (the gate
    shut, the parent's hand moving the arm at every tick), the lesson's proposal term and act_pred's gradient (with the correction's)
    scale exactly with the reliability: none at 0, a hundredth of the full lesson at 0.01, a quarter at 0.25; the forward half's
    lesson does not depend on it. On a window with own acts beside the rests the lesson is linear in the reliability, its own acts'
    part the same at every reliability. (Divided by the weights' sum, as it was, the reliability only reweighted: at 0.01 a window of
    rests taught as at 1.) What act_pred then learns under its optimizer is anatomy 31's"""
    from body.core.world import WorldLoop
    cfg = dict(_LR0, wake_ticks=100000, wake_every=10 ** 6, gate_every=10 ** 6, write_floor=1e-30, gate_floor=0.5, fast_rls=0,
               act_inv_lr=1e-2, act_inv_tau=2000)
    w = _arm_world(); torch.manual_seed(5); L = _born_in(_Timed(TOK, cfg), cfg, w)
    run = WorldLoop(L)
    for _ in range(400):                                          # act_inv learns on the arm's own acts
        run.step()
    mixed = L._window_tensors()
    assert int((mixed[0]["arm"] != 12).sum()) >= 5 and int((mixed[0]["arm"] == 12).sum()) >= 5
    _closed_arm(L)
    for _ in range(40):                                           # rests alone: nothing moves the arm
        run.step()
    rests = L._window_tensors()
    assert all(r_ == (12, "still") for r_ in w.moved[-32:]) and int((rests[0]["arm"] != 12).sum()) == 0
    for _ in range(40):                                           # pure demonstration: the parent's hand at every tick
        w.guide[:] = [[6, 7, 11, 13, 8, 16, 17, 18][len(w.moved) % 8]]
        run.step()
    demo = L._window_tensors()
    assert all(who == "guide" for _, who in w.moved[-32:]) and int((demo[0]["arm"] != 12).sum()) == 0
    gains = (0.0, 0.01, 0.25, 1.0); out = {}
    for name, (obs, whos, bundles, reads) in (("rests", rests), ("demonstration", demo), ("mixed", mixed)):
        res = {g: _lesson_at(L, obs, whos, bundles, g) for g in gains}
        lt0, gp0, gf0, _ = res[0.0]; lt1, gp1, gf1, rp1 = res[1.0]
        full = lt1 - lt0                                          # the rests' part of the lesson at reliability 1 (the forward half's cancels)
        assert full > 1e-3, (name, full)
        for g in gains:
            lt_, gp_, gf_, rp_ = res[g]
            assert abs((lt_ - lt0) - g * full) <= 1e-5 * max(1.0, full), (name, g, lt_ - lt0, g * full)
            assert torch.allclose(gp_ - gp0, g * (gp1 - gp0), rtol=1e-4, atol=1e-7), (name, g, float((gp_ - gp0).norm()), g * float((gp1 - gp0).norm()))
            assert torch.equal(gf_, gf0), (name, g, "the forward half's lesson moved with the reliability")
            assert rp_["demo_w"] == round(g, 3)
        if name != "mixed":                                       # no own act: nothing is taught at reliability 0
            assert float(gp0.abs().max()) == 0.0 and abs(lt0 - rp1["fwd"]) <= 1e-4, (name, float(gp0.abs().max()))
            ratio = float(res[0.01][1].norm()) / float(gp1.norm())
            assert abs(ratio - 0.01) <= 1e-4, (name, ratio)
        else:
            assert float(gp0.abs().max()) > 0.0                   # its own acts teach at every reliability
        out[name] = (round(full, 4), round(float(gp1.norm()), 4), round(float(res[0.01][1].norm()), 6))
    print(f"anatomy 28: act_pred's lesson scales with act_inv's reliability (the weights absolute): the rests' part of the lesson and",
          f"act_pred's gradient at 1 and at 0.01 (lesson, |grad| at 1, |grad| at 0.01): rests alone {out['rests']}, pure demonstration",
          f"{out['demonstration']}, own acts beside rests {out['mixed']} (linear in the reliability, the own acts' part fixed); the forward",
          f"half's lesson the same at every reliability")


def test_act_inv_reliability_is_kappa():
    """anatomy 29 (the R6 verifier's second finding; SIM_DESIGN.md 5.4): act_inv's reliability is Cohen's kappa per joint over its
    running confusion, so the acts' own rates are never read as skill. On acts whose joints hold 60%, 80% and 95% of the time (the
    other settings evenly), labels that carry nothing of the act read about 0: one that always says 'hold' (exactly 0), one drawn at
    the acts' own rates, one drawn evenly; the critics' estimator R6 used, recomputed here for the record, read the first 0.51 at 60%
    and 0.75 at 80%. A label right 90% of the time reads each joint's batch kappa of its counts (the decay made negligible); a label
    right on one joint and blind on the other reads half that joint's; a joint whose labels and acts never varied reads 0. Zero until
    64 acts, on from the 65th. A learner blind at first ('hold' always) and then right 90% reads about 0 while blind and rises after"""
    from body.core.mouth import MouthMixin
    from body.core.timing import TimingMixin

    class _Rel(TimingMixin):
        def __init__(self, tau):
            self.cfg = {"act_inv_tau": tau}

    arm = Effector("arm", [5, 5], rest_id=12, sense="body", inverse=True)
    HOLD = 2

    def act_(rng, p):
        return HOLD if rng.random() < p else rng.choice([0, 1, 3, 4])

    def run(tau, n, p, labeler, joint1=None, seed=1):
        rng = random.Random(seed); life = _Rel(tau); st = MouthMixin._motor_state_new(arm); pairs = []; hist = []
        for _ in range(n):
            true = [act_(rng, p), act_(rng, p) if joint1 is None else joint1]
            label = labeler(rng, true, p)
            life._inv_rel_update(arm, st, label, true); pairs.append((label, true)); hist.append(st["inv_gain"])
        return st, pairs, hist

    def batch_kappa(pairs, j):
        cnt = [[0] * 5 for _ in range(5)]
        for lab, tru in pairs:
            cnt[lab[j]][tru[j]] += 1
        n = float(len(pairs)); po = sum(cnt[k][k] for k in range(5)) / n
        pe = sum(sum(cnt[k]) * sum(cnt[r][k] for r in range(5)) for k in range(5)) / (n * n)
        return (po - pe) / (1.0 - pe) if 1.0 - pe > 1e-12 else 0.0

    def critics_(pairs, tau=8192.0):
        """the estimator R6 used: the critics' running moments over every setting of every joint pooled, the slope clipped"""
        d = 1.0 - 1.0 / tau; m = [0.0] * 6
        for label, true in pairs:
            for j in range(2):
                for k in range(5):
                    v = 1.0 if label[j] == k else 0.0; g = 1.0 if true[j] == k else 0.0
                    m[0] = d * m[0] + 1.0; m[1] = d * m[1] + v; m[2] = d * m[2] + g; m[3] = d * m[3] + v * v; m[4] = d * m[4] + g * g; m[5] = d * m[5] + v * g
        n = m[0]; mv, mg = m[1] / n, m[2] / n; var_v = m[3] / n - mv * mv; cov = m[5] / n - mv * mg
        return max(0.0, min(1.0, cov / max(var_v, 1e-9)))

    blind = {"always hold": lambda rng, t, p: [HOLD, HOLD], "at the acts' rates": lambda rng, t, p: [act_(rng, p), act_(rng, p)],
             "evenly": lambda rng, t, p: [rng.randrange(5), rng.randrange(5)]}
    right90 = lambda rng, t, p: [t[j] if rng.random() < 0.9 else rng.randrange(5) for j in range(2)]   # noqa: E731
    table = {}
    for p in (0.6, 0.8, 0.95):
        for name, lab in blind.items():
            st, pairs, _ = run(8192, 4000, p, lab)
            assert st["inv_gain"] <= 0.06 and all(abs(k_) <= 0.06 for k_ in st["inv_kappa"]), (p, name, st["inv_gain"], st["inv_kappa"])
            if name == "always hold":
                assert all(abs(k_) <= 1e-9 for k_ in st["inv_kappa"]), (p, st["inv_kappa"])
            table[(p, name)] = (round(st["inv_gain"], 3), round(critics_(pairs), 3))
        st, pairs, hist = run(1e12, 4000, p, right90)
        bk = [batch_kappa(pairs, j) for j in range(2)]
        assert all(abs(a_ - b_) <= 1e-6 for a_, b_ in zip(st["inv_kappa"], bk)) and abs(st["inv_gain"] - sum(bk) / 2.0) <= 1e-6, (p, st["inv_kappa"], bk)
        assert all(h_ == 0.0 for h_ in hist[:64]) and hist[64] > 0.0, (p, hist[62:66])
        table[(p, "right 90%")] = (round(st["inv_gain"], 3), round(critics_(pairs), 3))
    assert table[(0.6, "always hold")][1] >= 0.45 and table[(0.8, "always hold")][1] >= 0.7, table   # what R6's estimator read
    assert table[(0.6, "right 90%")][0] >= 0.8, table
    # one joint right, the other blind: half the first joint's
    st, pairs, _ = run(1e12, 3000, 0.6, lambda rng, t, p: [t[0] if rng.random() < 0.9 else rng.randrange(5), HOLD])
    k0 = batch_kappa(pairs, 0)
    assert abs(st["inv_kappa"][0] - k0) <= 1e-6 and abs(st["inv_kappa"][1]) <= 1e-9 and abs(st["inv_gain"] - k0 / 2.0) <= 1e-6, (st["inv_kappa"], k0)
    half = (round(k0, 3), round(st["inv_gain"], 3))
    # a joint that never varied (the acts and the labels always 'hold') has shown nothing: 0
    st, _, _ = run(8192, 500, 0.6, lambda rng, t, p: [t[0], HOLD], joint1=HOLD)
    assert st["inv_kappa"][0] == 1.0 and st["inv_kappa"][1] == 0.0 and st["inv_gain"] == 0.5, st["inv_kappa"]
    # an inverse model that learns (as act_inv does at birth): 3000 acts of labels that always say 'hold', then labels right 90% of the
    # time, the acts holding 80% throughout, the counts decaying over 2000 acts: about 0 while blind (R6's estimator read 0.75 there),
    # then rising as the informative labels replace the blind ones in the counts
    seen = [0]

    def learner(rng, t, p):
        seen[0] += 1
        return [HOLD, HOLD] if seen[0] <= 3000 else right90(rng, t, p)
    st, pairs, hist = run(2000, 9000, 0.8, learner)
    blind_ = max(hist[64:3000]); rise = [round(hist[k_], 3) for k_ in (2999, 3499, 3999, 5999, 8999)]
    assert blind_ <= 0.06 and critics_(pairs[:3000], 2000.0) >= 0.7, (blind_, critics_(pairs[:3000], 2000.0))
    assert all(a_ < b_ for a_, b_ in zip(rise, rise[1:])) and rise[-1] >= 0.7, rise
    print("anatomy 29: act_inv's reliability is Cohen's kappa per joint (kappa, and R6's critics' estimator beside it):",
          "; ".join(f"hold {p:.0%} {nm} {v_[0]} ({v_[1]})" for (p, nm), v_ in table.items()),
          f"; one joint right (its kappa {half[0]}) and one blind: {half[1]}; a joint that never varied 0; zero until 64 acts; a learner",
          f"blind for 3000 acts (at most {blind_:.3f}) then right 90%: {rise} at acts 3000, 3500, 4000, 6000, 9000")


def test_the_forward_half():
    """anatomy 26 (step R6; SIM_DESIGN.md 5.8: the motor timing part extended to foresee each effector's next body sense): after each
    tick's own step the forward half foresees the next tick's body sense from the stream there (the act just taken in it); at the next
    tick the error is the sense felt less the sense foreseen, and the proposal is act_pred's plus the correction of that error; the
    morning's first tick has no foresight and no correction; an effector with no sense has neither. The correction steers a chunk
    under way: a reach the world blocked (the sense falling short of the foreseen) moves act_pred's best guess, and the chunk's act
    with it"""
    from body.core.world import WorldLoop
    cfg = dict(_LR0, wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.5, fast_rls=0, chunk_gate=1, chunk_max=6)
    w = _arm_world(wall=0.45); torch.manual_seed(5); L = _born_in(_Timed(TOK, cfg), cfg, w)
    with torch.no_grad():
        L.m.timing["arm"].cor.weight.copy_(torch.randn(64, 2, generator=torch.Generator().manual_seed(3)) * 0.1)   # a correction to read
        L.m.timing["grip"].cor.weight.copy_(torch.randn(64, 1, generator=torch.Generator().manual_seed(4)) * 0.1)
    seen = []; props = []; after = []
    f_fore, f_prop = L._timing_foresee, L._timing_propose

    def fore(i):
        out = f_fore(i); st_ = L.motor[i - 1]
        # after the tick's own step: the window's last position is this tick's (this frame's body sense, the tick's own acts in it) and
        # the stream the forward half read is that position's
        pos = L.win[-1]
        with torch.no_grad():
            now_ = L._stream_now()
        after.append(torch.equal(pos["body"], torch.tensor(w.shown[L.ticks].obs["body"], dtype=torch.float32)) and
                     all(s_["now"] is not None and pos[e_.field] == s_["now"]["act"] for e_, s_ in zip(L.anatomy.effectors[1:], L.motor)) and torch.equal(L._C_last, now_))
        seen.append((L.ticks, i, None if st_["fwd"] is None else st_["fwd"].clone(), L._C_last.clone())); return out

    def prop(e, C):
        st_ = L.motor[L.anatomy.effectors.index(e) - 1]
        out = f_prop(e, C)
        props.append((L.ticks, e.name, C.clone(), None if st_["err"] is None else st_["err"].clone(), None if st_["sense"] is None else st_["sense"].clone(), out.clone()))
        return out
    L._timing_foresee = fore; L._timing_propose = prop
    run = WorldLoop(L)
    for _ in range(200):
        run.step()
    L.cfg["wake_ticks"] = L.sleep_pressure + 2
    while L.nights == 0:
        run.step()
    dusk = L.ticks - 1; L.cfg["wake_ticks"] = 100000
    for _ in range(10):
        run.step()
    fwd_at = {(tk, i): (f_, C_) for tk, i, f_, C_ in seen}
    n_err = n_none = 0
    for tk, name, C, err, sense, out in props:
        i = [e.name for e in L.anatomy.effectors].index(name); tm = L.m.timing[name]; e = L.anatomy.effectors[i]
        with torch.no_grad():
            want = tm.pred(C)
            if e.sense is None:
                assert err is None and (tk, i) in fwd_at and fwd_at[(tk, i)][0] is None
            else:
                idx = e.sense_idx if e.sense_idx is not None else [0, 1]
                assert torch.equal(sense, torch.tensor([w.shown[tk].obs[e.sense][j] for j in idx], dtype=torch.float32)), (tk, name)
                prev = fwd_at.get((tk - 1, i))
                if tk == dusk + 1 or prev is None:
                    assert err is None, (tk, name); n_none += 1
                else:
                    f_, C_ = prev
                    assert torch.equal(f_, tm.fwd(C_)) and torch.equal(err, sense - f_), (tk, name)
                    want = want + tm.cor(err); n_err += 1
        assert torch.equal(out, want), (tk, name)
    assert all(after) and len(after) == 3 * (L.ticks), (sum(after), len(after), L.ticks)
    assert n_err >= 350 and n_none == 4, (n_err, n_none)                  # the first tick of life and the morning's, for the arm and the grip
    assert L.motor[2]["fwd"] is None and L.motor[2]["err"] is None and not hasattr(L.m.timing["tap"], "cor")
    # the correction steers a chunk under way
    e = L.anatomy.effectors[1]; st = L.motor[0]; tm = L.m.timing["arm"]
    R = L.m.acts["arm"].rows
    with torch.no_grad():
        tm.pred.weight.zero_(); tm.pred.bias.copy_(R[3] + R[5 + 2])                   # act_pred: joint 0 up by 0.09, joint 1 held (act 17)
        tm.cor.weight.zero_(); tm.cor.weight[:, 0] = 2.0 * (R[3] - R[1]) / 1.8          # a shortfall on joint 0 turns its best guess down
        L.m.gates["arm"].weight.zero_(); L.m.gates["arm"].bias.fill_(30.0)
    acts = {}
    for label, felt in (("as foreseen", [1.8, 0.0]), ("blocked", [0.0, 0.0])):
        prev_now = dict(st["now"] or {}, act=17, world=17, acted=True)
        st["now"] = prev_now; st["acted_last"] = True; st["chunk"] = 2; st["sense"] = None; st["fwd"] = torch.tensor([1.8, 0.0]); st["rfx_n"] = 0
        fr = Frame(L.ticks, {"body": felt, "touch": [0.0, 0.0]}, 0.0); L.world.now = fr
        L._choose_effector(1, fr, L._C_last, 0.0, False)
        acts[label] = (st["now"]["act"], st["now"]["cont"], [round(float(x), 3) for x in st["err"]])
    assert acts["as foreseen"][:2] == (17, True) and acts["blocked"][:2] == (7, True), acts
    print(f"anatomy 26: the forward half foresaw each tick's body sense from the stream after the act, the error the sense less the",
          f"foresight on {n_err} proposals (none on the first tick of life and the morning's), each proposal act_pred's plus the",
          f"correction; the tap has neither; within a chunk the reach as foreseen goes on ({acts['as foreseen'][0]}) and a blocked one",
          f"turns ({acts['blocked'][0]}, error {acts['blocked'][2]})")


def _draw_spy(L, calls):
    """record every draw on the life's stream from the effectors' choices: (tick, effector, kind)"""
    o_rand, o_mn = torch.rand, torch.multinomial

    def rand(*a_, generator=None, **k_):
        if generator is L.gen:
            f_ = sys._getframe(1)
            if f_.f_code.co_name in ("_choose", "_choose_effector"):
                calls.append((L.ticks, f_.f_locals.get("i", 0), "rand"))
        return o_rand(*a_, generator=generator, **k_)

    def mn(*a_, generator=None, **k_):
        if generator is L.gen:
            f_ = sys._getframe(1)
            if f_.f_code.co_name in ("_choose", "_choose_effector"):
                calls.append((L.ticks, f_.f_locals.get("i", 0), "draw"))
        return o_mn(*a_, generator=generator, **k_)
    torch.rand, torch.multinomial = rand, mn
    return o_rand, o_mn


def test_the_learned_stops():
    """anatomy 27 (step R6; SIM_DESIGN.md 5.3 and 5.4): under chunk_gate a later effector's acts run in chunks. A chunk begins with a
    fresh decision (the gate's draw, each joint drawn); while it is under way each act is act_pred's best guess (no joint drawn), the
    gate's own draw deciding whether it goes on (recorded as the draw); it ends where act_pred's best guess is the rest (the learned
    end, the gate having said yes), where the gate's own draw says no, where the reflex fires, where the effector's declared end act
    closed it, or at chunk_max (a ceiling: the next act is a fresh decision). Only a chunk's first act is the actor's to credit. A
    reflex takes the tick: its act to the world, the effector's own act its rest (the window's), no draw on the body's stream, its
    cost to the fatigue, no striatal event, no actor credit, and no eligibility in the gate's lesson (the lesson equal to the lesson
    with those rows' eligibility zero). Without chunk_gate every act is a fresh decision, as before R6. The night ends any chunk"""
    from body.core.world import WorldLoop
    base = dict(_LR0, wake_ticks=100000, wake_every=8, gate_every=10 ** 9, write_floor=1e-30, gate_floor=0.0, fast_rls=1, fast_input="striatum",
                actor=1, chunk_gate=1, chunk_max=4, gate_own_draw=1)

    def born(act_arm, gate_arm, cfg=base, wall=None, grip_end=None, act_grip=None, scale=1.0):
        w_ = _arm_world(wall=wall); torch.manual_seed(5); L_ = _born_in(_Timed(TOK, cfg, grip_end=grip_end), cfg, w_)
        with torch.no_grad():
            for name, act in (("arm", act_arm), ("grip", act_grip)):
                if act is not None:                                   # act_pred set to propose one act (its best guess), at `scale`
                    tm_ = L_.m.timing[name]; tm_.pred.weight.zero_(); tm_.pred.bias.copy_(scale * _row(L_, name, act))
            L_.m.gates["arm"].weight.zero_(); L_.m.gates["arm"].bias.fill_(gate_arm)
        return L_, w_, WorldLoop(L_)

    def live(L_, run_, n, calls=None, extra=None):
        rows = []
        o_ = _draw_spy(L_, calls) if calls is not None else None
        try:
            for _ in range(n):
                ea = [None if st_["e_actor"] is None else st_["e_actor"].clone() for st_ in L_.motor]
                run_.step()
                rows.append((dict(L_.last["acts"]["arm"]), dict(L_.motor[0]["now"]), ea, [None if st_["e_actor"] is None else st_["e_actor"].clone() for st_ in L_.motor],
                             dict(L_.last["acts"]["grip"])))
                if extra is not None:
                    extra(L_, rows)
        finally:
            if o_ is not None:
                torch.rand, torch.multinomial = o_
        return rows
    # A. the chunk runs act_pred's best guess to its ceiling: act 17 (joint 0 up by 0.09), the gate open
    calls = []; L, w, run = born(17, 30.0)
    rows = live(L, run, 60, calls)
    by = collections.defaultdict(list)
    for tk, i, kind in calls:
        if i == 1:
            by[tk].append(kind)
    k_run = 0; starts = 0
    for tk, (rec, now, ea0, ea1, _) in enumerate(rows):
        assert rec["acted"] and now["drew"], (tk, rec)                                   # the gate open (p about 1)
        if rec["cont"]:
            k_run += 1
            assert rec["act"] == 17 and by[tk] == ["rand"] and 1 <= k_run <= 3 and now["p_choice"] == float(now["probs"][0][3]) * float(now["probs"][1][2]), (tk, rec, by[tk])
            assert torch.equal(ea0[0], ea1[0]), f"tick {tk}: a continuation credited the actor"
        else:
            assert by[tk] == ["rand", "draw", "draw"] and (tk == 0 or k_run == 3), (tk, by[tk], k_run)
            assert rec["stop"] == (None if tk == 0 else "max"), (tk, rec)
            assert ea1[0] is not None and (ea0[0] is None or not torch.equal(ea0[0], ea1[0])), f"tick {tk}: a chunk's start not credited"
            k_run = 0; starts += 1
    st = L.motor[0]
    assert starts == 15 and st["stops"]["max"] == 14 and st["chunks"] == 15, (starts, st["stops"], st["chunks"])
    # B. the learned end: act_pred's best guess is the rest (a weak proposal, so a fresh draw still acts); each chunk is its first act,
    # then the gate's yes with no act
    L, w, run = born(12, 30.0, scale=0.05)
    rows = live(L, run, 60)
    n_rest = 0
    for tk, (rec, now, ea0, ea1, _) in enumerate(rows):
        if tk and rows[tk - 1][0]["acted"]:
            assert rec["cont"] and not rec["acted"] and now["drew"] and rec["act"] == 12 and rec["stop"] == "rest", (tk, rec)
            n_rest += 1
        else:
            assert not rec["cont"] and rec["stop"] is None, (tk, rec)
    assert all(b_[1] is False and b_[7] is True for k_, b_ in enumerate(L.motor[0]["buf"]) if rows[k_][0]["cont"]), "the gate's yes not recorded at the learned end"
    assert n_rest >= 20 and L.motor[0]["stops"]["rest"] == n_rest, (n_rest, L.motor[0]["stops"])
    # C. the gate's own draw closes it: act_pred's guess 17, the gate at even odds
    calls = []; L, w, run = born(17, 0.0)
    rows = live(L, run, 300, calls)
    n_gate = n_go = 0
    for tk, (rec, now, ea0, ea1, _) in enumerate(rows):
        if rec["cont"]:
            if now["drew"]:
                assert rec["act"] == 17 and rec["acted"] and rec["stop"] is None; n_go += 1
            else:
                assert rec["act"] == 12 and not rec["acted"] and rec["stop"] == "gate"; n_gate += 1
    assert n_gate >= 15 and n_go >= 15 and L.motor[0]["stops"]["gate"] == n_gate, (n_gate, n_go)
    # D. the reflex: the arm reaches up into a wall, it hurts, the reflex reverses the last move for 2 ticks
    calls = []; L, w, run = born(17, 30.0, wall=0.3)
    pushes = []; o_push = L.m.striatum_push_act
    L.m.striatum_push_act = lambda j, act: (pushes.append((L.ticks, j)), o_push(j, act))[1]
    fat = []; f_act = L._act_effectors

    def act_rec(u, stri, gam, tick_tr):
        before = L.fatigue; out = f_act(u, stri, gam, tick_tr)
        fat.append((before, L.fatigue, [(st_["now"]["acted"], st_["now"]["reflex"], st_["now"]["cost"]) for st_ in L.motor])); return out
    L._act_effectors = act_rec
    wins = []
    rows = live(L, run, 80, calls, extra=lambda L_, rows_: wins.append(int(L_.win[-1]["arm"])))
    by = collections.defaultdict(list)
    for tk, i, kind in calls:
        if i == 1:
            by[tk].append(kind)
    rfx = [tk for tk, r_ in enumerate(rows) if r_[1]["reflex"]]
    assert len(rfx) >= 6, len(rfx)
    for tk in rfx:
        rec, now, ea0, ea1, _ = rows[tk]
        assert rec["act"] == 12 and not rec["acted"] and not now["drew"] and not rec["cont"] and rec["world"] != 12 and w.moved[tk] == (rec["world"], "own"), (tk, rec, w.moved[tk])
        assert by[tk] == [] and wins[tk] == 12 and (tk, 0) not in pushes, (tk, by[tk], wins[tk])
        assert (ea0[0] is None and ea1[0] is None) or torch.equal(ea0[0], ea1[0])
        before, after, per = fat[tk]
        assert per[0] == (False, True, 0.05) and abs(after - before - sum(c_ for a_, r_, c_ in per if a_ or r_)) < 1e-12, fat[tk]
    first_rfx = [tk for tk in rfx if not rows[tk - 1][1]["reflex"]]
    for tk in first_rfx:
        pw = rows[tk - 1][1]["world"]
        assert rows[tk][0]["stop"] == "reflex" and w.shown[tk].obs["touch"][0] == 1.0 and rows[tk][0]["world"] == (4 - pw // 5) * 5 + (4 - pw % 5), (tk, rows[tk][0], pw)
        assert rows[tk + 1][1]["reflex"] and rows[tk + 1][0]["world"] == rows[tk][0]["world"] if tk + 1 < len(rows) else True
    assert all(bool(b_[9]) == rows[len(rows) - len(L.motor[0]["buf"]) + k_][1]["reflex"] for k_, b_ in enumerate(L.motor[0]["buf"]))
    # the gate's lesson on those rows: no eligibility on a reflex's tick (the lesson equal to the lesson with those rows' eligibility zero)
    for own_, s_ in ((1, 0), (0, 1)):
        cfg2 = dict(base, gate_own_draw=own_, elig_from=s_, gate_lr=0.05)
        LA, wA, runA = born(17, 2.0, cfg=cfg2, wall=0.3); LB, wB, runB = born(17, 2.0, cfg=cfg2, wall=0.3)   # the gate short of sure (its p under 1)
        live(LA, runA, 70); live(LB, runB, 70)
        bufA, bufB = LA.motor[0]["buf"], LB.motor[0]["buf"]
        n_rf = sum(1 for b_ in bufB if b_[9])
        assert n_rf >= 4 and [b_[9] for b_ in bufA] == [b_[9] for b_ in bufB]
        assert all(abs(float(b_[6]) - float(LA.cfg["gate_vigor"])) > 0.05 for b_ in bufB if b_[9]), "a reflex row whose eligibility is zero anyway"
        vig = float(LA.cfg["gate_vigor"])
        ref = [(list(b_[:6]) + [vig, False] + list(b_[8:])) if b_[9] else list(b_) for b_ in bufA]
        assert all(not b_[1] for b_ in bufA if b_[9])
        bufA.clear(); bufA.extend(ref)
        gA, gB = LA.m.gates["arm"], LB.m.gates["arm"]
        baseA, lastA = _lesson_fixed(LA, bufA, gA, LA.opt_motor, LA.motor[0]["g_base"], own_, s_, lambda b: float(b[8]))
        LB._gate_lesson(1)
        (wa, oa), (wb, ob) = _gate_state(LA, gA), _gate_state(LB, gB)
        assert lastA is not None and all(torch.equal(wa[k], wb[k]) for k in wa) and _same_opt(oa, ob), f"own {own_}, from {s_}: the reflex's rows kept eligibility"
        assert baseA == LB.motor[0]["g_base"] and lastA == LB.motor[0]["last"]
    # E. the declared end: the grip's end act 0 closes its chunk, the next act a fresh decision
    cfgE = dict(base)
    L, w, run = born(None, 30.0, cfg=cfgE, grip_end=0, act_grip=0)
    with torch.no_grad():
        L.m.gates["grip"].weight.zero_(); L.m.gates["grip"].bias.fill_(30.0)
    rows = live(L, run, 40)
    g_ = [r_[4] for r_ in rows]
    for tk in range(1, len(g_)):
        if g_[tk - 1]["act"] == 0 and g_[tk - 1]["acted"]:
            assert not g_[tk]["cont"] and g_[tk]["stop"] == "end", (tk, g_[tk - 1], g_[tk])
        elif g_[tk - 1]["acted"]:
            assert g_[tk]["cont"] and g_[tk]["act"] == 0, (tk, g_[tk])
    assert L.motor[1]["stops"]["end"] >= 10
    # F. without chunk_gate every act is a fresh decision (as before R6); the night ends any chunk
    calls = []; L, w, run = born(17, 30.0, cfg=dict(base, chunk_gate=0))
    rows = live(L, run, 30, calls)
    assert not any(r_[0]["cont"] for r_ in rows) and all(sorted(k_ for t_, i_, k_ in calls if t_ == tk and i_ == 1) == ["draw", "draw", "rand"] for tk in range(30))
    assert L.motor[0]["chunks"] == 0 and L.motor[0]["chunk"] == 0 and sum(L.motor[0]["stops"].values()) == 0
    L, w, run = born(17, 30.0); live(L, run, 20)
    assert L.motor[0]["chunk"] >= 1 and L.motor[0]["sense"] is not None and L.motor[0]["fwd"] is not None
    L.night()
    assert all(st_["chunk"] == 0 and st_["sense"] is None and st_["fwd"] is None and st_["err"] is None for st_ in L.motor)
    rows = live(L, run, 1)
    assert not rows[0][0]["cont"] and rows[0][0]["stop"] is None
    print(f"anatomy 27: under chunk_gate the arm's chunks ran act_pred's best guess with only the gate drawing, {starts} chunks of",
          f"chunk_max 4 each re-decided at the ceiling, the actor credited at the starts alone; the learned end ({n_rest} stops at the rest,",
          f"the gate's yes recorded); the gate's own no ({n_gate} stops against {n_go} goes); {len(rfx)} reflex ticks (the reversal to",
          f"the world, a rest in the window, no draw, its cost, no striatal event, no credit, no eligibility in the lesson, equal to the",
          f"lesson with it zero); the grip's declared end; without chunk_gate every act fresh; the night ends the chunk")


# ---------------- step R5b: the striatum's rows per joint ----------------

class _Humanoid(LanguageAnatomy):
    """the diary's words and face, a body sense of 34 joints' velocities (a channel of the world's frames encoded by its own map, body_in), and
    34 joints of five settings (the middle, 2, holds) as eight limbs: two legs of six, a waist of three, two arms of seven (each with an
    inverse model), a head of three, two grips of one; each limb senses its own joints and rests with every joint held"""
    LIMBS = (("leg_l", 6), ("leg_r", 6), ("waist", 3), ("arm_l", 7), ("arm_r", 7), ("head", 3), ("grip_l", 1), ("grip_r", 1))

    def __init__(self, tok, cfg=None):
        super().__init__(tok, cfg)
        self.channels.append(Channel("body", "vector", 34, organ="body_in"))
        j0 = 0
        for name, J in self.LIMBS:
            self.effectors.append(Effector(name, [5] * J, rest_id=(5 ** J - 1) // 2, effort=0.01, sense="body", sense_idx=list(range(j0, j0 + J)),
                                           inverse=name.startswith("arm"), inv_hidden=16))
            j0 += J


def _body_world():
    """a stub of the simulated world for a body of 34 joints: each joint's velocity the step its setting in the tick's act makes (a
    rest holds every joint), a short line on the words now and then, a smile now and then"""
    from body.core.world import SimWorld

    class BodyWorld(SimWorld):
        LINE = "up we go "

        def __init__(self):
            self.t = 0; self.v = [0.0] * 34; self.paused = False; self.applied = []

        def frame(self):
            assert not self.paused, "a frame taken while the world is paused"
            obs = {"body": list(self.v)}
            k = self.t % 40 - 5
            if 0 <= k < len(self.LINE):
                obs["ear"] = TOK.token_to_id(self.LINE[k])
            return Frame(self.t, obs, 2.0 if self.t % 30 == 20 else 0.0, {"who": "parent"})

        def apply(self, acts):
            assert not self.paused, "the world moved while paused"
            self.applied.append(dict(acts)); j0 = 0
            for name, J in _Humanoid.LIMBS:
                a = int(acts.get(name, (5 ** J - 1) // 2))
                for j in range(J - 1, -1, -1):                             # joint 0 the most significant digit
                    self.v[j0 + j] = 0.1 * (a % 5 - 2); a //= 5
                j0 += J
            self.t += 1

        def pause(self):
            self.paused = True

        def resume(self):
            self.paused = False

        def save_state(self):
            return pickle.dumps((self.t, self.v))

        def load_state(self, blob):
            self.t, self.v = pickle.loads(blob)
    return BodyWorld()


def _stri_ref(o, effectors):
    """the striatal expansion by hand: the language line's events, then each later effector's acts, position by position, each act's
    joints' settings (its table's digits) at their rows in the effector's block, joint 0 first"""
    V = o.vocab; k = o.stri_line.numel(); z = o.stri_b.clone()
    for p_ in range(k):
        e_ = int(o.stri_line[p_])
        if e_ >= 0:
            z += o.stri_W[p_ * (2 * V + 3) + e_]
    base = k * (2 * V + 3)
    for j_, e in enumerate(effectors[1:]):
        S = sum(e.factors)
        for p_ in range(k):
            x_ = int(o.stri_mline[j_, p_])
            if x_ >= 0:
                off = base + p_ * S
                for d_, K in zip(o.acts[e.name].digits(torch.tensor(x_)).tolist(), e.factors):
                    z += o.stri_W[off + d_]; off += K
        base += k * S
    return torch.relu(z)


def test_a_34_joint_body_builds():
    """anatomy 30 (step R5b): the striatum holds a later effector's events by joint, not by flat act, so a body of 34 joints builds
    within memory. At the served striatum (8 events, 2048 units, the working-memory slot) the body's organs hold the language block,
    thresholds and heads bit for bit as the diary's, first (the global random stream left where the diary's leaves it), then a block
    per limb of 8 x its joints' settings rows at 1/sqrt(8 J): 1360 rows, 11.1 MB, where a row per flat act would have needed 1.5
    million rows (12.3 GB; not built); an act's event is its joints' settings' rows. Born as a life at those sizes it lives in a stub
    world: every limb acts and its acts enter its own line, the striatal read equal to the expansion by hand, the arms' act_inv
    counting one confusion per joint"""
    from body.core.world import WorldLoop
    k, m = 8, 2048
    a = _Humanoid(TOK); V = a.vocab; motor = a.effectors[1:]
    J = [len(e.factors) for e in motor]; S = [sum(e.factors) for e in motor]
    assert sum(J) == 34 and len(motor) == 8 and sum(S) == 170
    torch.manual_seed(9); o1 = Organs(V, d=64, layers=2, heads=2, window=32); o1.striatum_init(k, m, seed=2, wm=1); r1 = torch.get_rng_state()
    torch.manual_seed(9); o3 = Organs(V, d=64, layers=2, heads=2, window=32, channels=a.channels, effectors=a.effectors, born_seed=3)
    o3.striatum_init(k, m, seed=2, wm=1, effectors=a.effectors); r3 = torch.get_rng_state()
    nl = k * (2 * V + 3)
    assert torch.equal(r1, r3) and torch.equal(o3.stri_W[:nl], o1.stri_W) and torch.equal(o3.stri_b, o1.stri_b)
    assert torch.equal(o3.vfast.weight, o1.vfast.weight) and torch.equal(o3.actor.weight, o1.actor.weight) and o3.wm_slot.shape == o1.wm_slot.shape
    starts = [nl + k * sum(S[:j_]) for j_ in range(len(motor))]
    assert o3.stri_W.shape == (nl + k * 170, m) and o3.stri_blocks == [(s_, tuple(e.factors)) for s_, e in zip(starts, motor)]
    for (s_, fac_), j_ in zip(o3.stri_blocks, J):                       # each limb's rows at 1/sqrt(k J)
        sd_ = float(o3.stri_W[s_:s_ + k * sum(fac_)].std())
        assert abs(sd_ * math.sqrt(k * j_) - 1.0) < 0.03, (fac_, sd_)
    motor_mb = (o3.stri_W.shape[0] - nl) * m * o3.stri_W.element_size() / 1e6
    flat_gb = k * sum(e.n_acts for e in motor) * m * 4 / 1e9            # a row per flat act (R5's layout): computed, never built
    assert abs(motor_mb - 11.14) < 0.01 and flat_gb > 12.0, (motor_mb, flat_gb)
    assert all(o3.actors[e.name].weight.shape == (sum(e.factors), 2 * m) for e in motor)
    # an act's event: its joints' settings' rows (the left arm's seven joints, settings 0..4 and 2, 2 at position 0)
    arm = motor.index(next(e for e in motor if e.name == "arm_l"))
    act = o3.acts["arm_l"].flat([0, 1, 2, 3, 4, 2, 2])
    before = o3.striatum_read().clone(); o3.striatum_push_act(arm, act)
    rows = starts[arm] + torch.tensor([0, 5 + 1, 10 + 2, 15 + 3, 20 + 4, 25 + 2, 30 + 2])
    assert torch.allclose(o3.striatum_read(), torch.relu(o3.stri_b + o3.stri_W[rows].sum(0)), atol=1e-6) and not torch.equal(o3.striatum_read(), before)
    del o1, o3
    # born as a life at those sizes, it lives
    cfg = dict(wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.5, fast_rls=1, fast_input="striatum", actor=1,
               stri_k=k, stri_m=m, wm=1)
    w = _body_world(); t0 = time.time(); torch.manual_seed(0); ha = _Humanoid(TOK, cfg)
    ho = Organs(ha.vocab, d=64, layers=2, heads=2, window=32, channels=ha.channels, effectors=ha.effectors, born_seed=0)
    ho.body_in = torch.nn.Linear(34, 64)                                # the body sense's own map (the sim's encoder is a later step)
    L = Life(ho, ha, cfg=cfg, device="cpu", seed=0, world=w); built = time.time() - t0
    assert L.m.stri_W.shape == (nl + k * 170, m) and L.m.stri_blocks == [(s_, tuple(e.factors)) for s_, e in zip(starts, motor)]
    state_mb = sum(t_.numel() * t_.element_size() for t_ in L.m.state_dict().values()) / 1e6
    assert state_mb < 256, state_mb                                     # the fast critic's evidence (4097^2 float64) is most of it
    run = WorldLoop(L); acted = [0] * len(motor)
    for _ in range(40):
        run.step()
        for j_, st_ in enumerate(L.motor):
            acted[j_] += int(st_["now"]["acted"])
    assert all(n_ > 0 for n_ in acted) and all(int((L.m.stri_mline[j_] >= 0).sum()) == min(k, n_) for j_, n_ in enumerate(acted)), acted
    assert torch.allclose(L.m.striatum_read(), _stri_ref(L.m, L.anatomy.effectors), atol=1e-5)
    arms = [L.motor[j_] for j_, e in enumerate(motor) if e.inverse]
    assert len(arms) == 2 and all(len(st_["inv_conf"]) == 7 and st_["inv_n"] > 0 for st_ in arms)
    assert len(w.applied) == 40 and all(set(e.name for e in motor) <= set(x_) for x_ in w.applied)
    print(f"anatomy 30: a body of 34 joints of five as eight limbs, at the served striatum (8 events, 2048 units): the language block",
          f"first and as the diary's, then 1360 rows per joint, {motor_mb:.1f} MB (a row per flat act would need {flat_gb:.1f} GB, not",
          f"built); an act's event its joints' rows; born as a life in {built:.1f} s (its organs {state_mb:.0f} MB), 40 ticks lived, the",
          f"limbs acting {acted} times, the striatal read equal to the expansion by hand, the arms' act_inv counting 7 joints each")


# ---------------- the R6 verifiers' third and fourth findings: what act_pred learns, gated by its labels' reliability ----------------

def _proposal_error(L, win):
    """the arm's proposal (act_pred and the correction, through the cortex as it is now) against act_inv's labels at a window's
    labelled rests (every rested position but the last, whose next sense is not felt): (the mean of 0.5 |proposal - the label's row|^2
    as the lesson weighs it, the number of those positions)"""
    L.win.clear(); L.win.extend(win)
    obs, whos, bundles, _ = L._window_tensors()
    tm = L.m.timing["arm"]; tab = L.m.acts["arm"]
    with torch.no_grad():
        C = L.m.stream(L.m.inputs(L.anatomy, obs, whos, bundles))
        s = obs["body"].float(); acts = obs["arm"]; T = int(acts.shape[0])
        lab = tab.flat_many([lg.argmax(-1) for lg in tm.inverse_logits(s[:-1], s[1:])])
        P = tm.pred(C[:-1]) + tm.cor(s[1:] - tm.fwd(C[:-1]))
        pos = [t for t in range(1, T - 1) if int(acts[t]) == 12]
        return sum(0.5 * float(((P[t - 1] - tab(lab[t].unsqueeze(0))[0]) ** 2).sum()) for t in pos) / len(pos), len(pos)


def _arm_gated(L):
    """the arm's act_pred and correction, flattened"""
    tm = L.m.timing["arm"]
    return torch.cat([p.detach().flatten().clone() for p in list(tm.pred.parameters()) + list(tm.cor.parameters())])


def _stream_params(L):
    """the parameters the waking lesson's Adam steps (all but act_pred's and the corrections' since R6 fix 3), copied"""
    return [p_.detach().clone() for g_ in L.opt_day.param_groups for p_ in g_["params"]]


def _snap(L):
    """the organs and every optimizer the body has, copied"""
    return copy.deepcopy(L.m.state_dict()), {k: copy.deepcopy(v.state_dict()) for k, v in sorted(vars(L).items()) if isinstance(v, torch.optim.Optimizer)}


def _lessons_from(L, snap, win, gain, n):
    """n waking lessons on a window held fixed at act_inv's reliability `gain` (a number, or lesson i's: gain(i)), from a saved state
    (copied in again: an optimizer's loaded state shares its tensors with what it was loaded from, so its steps would change the saved
    state)"""
    L.m.load_state_dict(snap[0])
    for k, sd in snap[1].items():
        getattr(L, k).load_state_dict(copy.deepcopy(sd))
    out = None
    for i in range(n):
        L.motor[0]["inv_gain"] = float(gain(i) if callable(gain) else gain)
        L.win.clear(); L.win.extend(win); out = L._wake_lesson()
    return out


def _waking_rate(L, lr):
    """the waking lesson's rate, the day's Adam's and act_pred's, both samples' (a body from before R6 fix 3 has no opt_pred, one from
    before R6 fix 5 no opt_lab)"""
    for k in ("opt_day", "opt_pred", "opt_lab"):
        for g_ in (getattr(L, k).param_groups if hasattr(L, k) else []):
            g_["lr"] = float(lr)


def _curve_from(L, snap, win, gain, n, at):
    """n waking lessons on a window held fixed from a saved state (as `_lessons_from`), the proposal's error to act_inv's labels at
    the window's labelled rests after each count of lessons in `at`: {count: error}"""
    L.m.load_state_dict(snap[0])
    for k, sd in snap[1].items():
        getattr(L, k).load_state_dict(copy.deepcopy(sd))
    out = {}
    for i in range(n):
        L.motor[0]["inv_gain"] = float(gain(i) if callable(gain) else gain)
        L.win.clear(); L.win.extend(win); L._wake_lesson()
        if i + 1 in at:
            out[i + 1] = _proposal_error(L, win)[0]
    return out


def test_act_pred_learns_by_reliability():
    """anatomy 31 (the R6 verifiers' third finding, fourth and fifth looks, 2026-09-24; SIM_DESIGN.md 5.4): WHAT act_pred LEARNS from a lesson
    scales with the reliability of the lesson's labels, under the real optimizers, from birth and once it has learned. R6 fix 1 scaled
    the lesson's loss by act_inv's reliability, and the waking lesson's Adam, which divides each parameter's step by its recent gradient
    size, undid the scale (at 0.01 act_pred learned 0.44 to 0.95 of what it learned at 1); fix 3 gave act_pred and the correction an
    optimizer whose step carries the gain (GatedAdam), but the share of act_inv's labels that reached the stream through the proposal was
    still taught whole by the day's Adam once act_pred had learned (9 to 21 times its earned share), and act_pred's gradient, bounded
    with the stream's, set the stream's steps. Now act_inv's labels reach act_pred and the correction alone, each optimizer's gradient
    has its own bound, and the own acts and act_inv's labels step as two samples, each with moments of its own (fix 5). The tiny arm (act_inv learned on 400 ticks of its own acts) at the served waking rate (live_lr), through the real
    `_wake_lesson`, on three windows held fixed: rests alone (the gate shut, nothing moving the arm), the parent's guidance alone (the
    gate shut, the hand on every tick) and a mixed window (own acts beside rests); 300 lessons on each at act_inv's reliability 0, 0.01,
    0.05, 0.25 and 1, from three states: fresh; own acts first (100 lessons of the mixed window, its rests unlabelled, at the served
    rate); and trained (300 lessons of the arm's own acts at a hundred times the served rate). Measured: the proposal's error (act_pred
    and the correction, through the stream as the lessons left it) to act_inv's labels at the labelled rests; the share learned at g is
    (err(0) - err(g)) / (err(0) - err(1)). Asserted, from every state:
    - THE SHARE rises strictly with the reliability and is at most five times it at 0.01 and at 0.05. THE BOUND, and why: the stream
      learns the same at every reliability (below), so the share is act_pred's and the correction's; a lesson at g steps g of a whole
      step, so N lessons at g learn about what the whole lessons learn in their first g N, which is more than g of their N as the curve
      bends (learning slows as the error falls): up to 3.7 g here, at 300 lessons. Five times leaves room for that bend and stays
      under what the old codes learned at 0.01 (fix 1 alone 0.21 to 0.95; fix 3, once act_pred had learned, 0.12 and 0.34, where
      it learned 0.002 and 0.003 now). Over a longer horizon the bend grows (the verifier read 7 times at 1000 lessons), so:
    - THE SAME LABEL MASS GIVEN WHOLE TEACHES THE SAME: 300 lessons at 0.05 against 300 lessons of which every twentieth is at 1 and the
      others at 0 (the same windows, steps and stream at every lesson; only how the labels' weight is spread differs): the ratio of what
      they taught is within 0.5 to 2 on rests and on guidance from every state. An honest gate reads about 1 (measured 0.87 to 1.00: a
      barely earned lesson's direction enters the momentum in proportion, so the spread labels are the slower); a re-inflating one
      more (fix 3 from the trained state 3.7 and 6.2; fix 1 alone 1.5 to 5.8).
    - THE STREAM NEVER LEARNS act_inv's LABELS: after the 300 lessons on rests alone or guidance alone the day's parameters are the same
      to the bit at every reliability, from every state. (Beside own acts they are not: the own acts teach the stream through act_pred,
      whose weights the labels move as far as their reliability allows, so the stream's lesson from its own acts differs by that much.)
    - act_pred and the correction do not move at 0 where only the rests' labels teach, and move at most five times the reliability's
      share of the whole move at 0.01 and 0.05.
    THE MIXED WINDOW BESIDE FITTED OWN ACTS (the R6 verifier's fifth look, R6 fix 5): own acts beside act_inv's labelled rests, the
    life as lived, once act_pred has learned own acts; as one sample the lesson's gain was its mean label weight, which the own acts
    dominate, and once they were predicted well Adam re-inflated the labels to that gain whatever their reliability (the verifier: from
    mixfit, 1000 lessons at 0.01 taught 0.18 of what they taught at 1, the same mass given whole 2 to 3 times less). Three states:
    mixfit (2000 lessons of the mixed window, its rests unlabelled, at a hundred times the served rate: its own acts fitted), trained
    (above) and own acts long (3000 lessons of the arm's own acts at the served rate); 1000 lessons at 0, 0.01, 0.05, 1 and the same
    label mass given whole (every hundredth lesson at 1, every twentieth), read after 300 and 1000. Measured with the stream held
    still over those lessons (the day's rate 0; each state built with it learning): beside own acts the stream learns them through
    act_pred, whose path the labels move, and on a body still learning its own acts that coupling moves the error at the labelled rests
    as much as the labels teach at 0.01 (own acts long, the stream learning: a reliability of 0.001 moved the share by up to 0.016, one
    of 0.0001 by 0.009; at 0.01 it read 0.036 or 0.002 by the thread count, the same mass 0.05 to 1.6); held still, a reliability of
    0.0001 moves it by 0.0000 and one of 0.001 by 0.0011 at most, in every state. There,
    at 300 and at 1000 lessons, from every state: the whole lesson teaches, the share rises with the reliability and is at most five
    times it at 0.01 and 0.05, and the same mass given whole teaches the same within 0.5 to 2 (measured 0.60 to 0.97; the one-sample
    gate, 440bad3, from mixfit 0.058 at 0.01 after 300 and 0.104 after 1000, the same mass 2.2 to 3.4). And from mixfit (its own acts
    fitted, the coupling quiet: 0.0001 moves the share by 0.0001 at most) through the whole waking lesson, the stream learning: the
    share at most five times the reliability and the same mass given whole at most twice as much, after 300 and 1000 (measured 0.0105
    to 0.061, 0.55 to 0.96; 440bad3 0.10 to 0.57 and 2.2 to 3.3). The lower side is read where the stream is still: through the whole
    lesson the three whole lessons at 0.01 of the first 300 each take a first, sign-sized step on label moments barely formed, where the
    spread lessons average the labels' gradient as the own acts move act_pred beneath them (0.55 there).
    Then the optimizers: act_pred and the corrections (the arm's and the grip's; the tap senses nothing, its act_pred alone) in opt_pred,
    a group per effector, the own acts' sample; the arm's (the one inverse model) again in opt_lab, act_inv's labels' sample, moments
    of its own; every other parameter in the day's Adam as before; each sample's gain its labels' mean weight over the positions (the
    own acts' share, the reliability times the labels' share, their sum the lesson's mean label weight); one lesson's first moment in
    opt_pred along the own acts' gradient alone and in opt_lab along the labels' alone; at weight 1 on every lesson GatedAdam is Adam to
    the bit (every tensor of the body after 20 lessons on a window of own acts alone, against act_pred and the corrections in a torch
    Adam under the same bounds); at gain 0 act_pred, the correction and both samples' moments do not move; on a constant gradient
    scaled by g (a lesson's loss at reliability g) Adam moves n whole steps whatever g, GatedAdam n g; one step at gain g moves the
    moments (1 - beta) g of their way to the gradient per unit weight; the language body has no opt_pred and no opt_lab and its day's
    Adam holds every parameter, as before. And THE NIGHT does not teach act_pred: a night at the night's rate moves the stream and
    leaves act_pred, the corrections and both samples' moments as the day left them (R8, which replays act_pred's targets, must step
    them through their gate, body/core/night.py)"""
    from body.core.world import WorldLoop
    LR = float((_served_cfg() or PHYSIOLOGY)["live_lr"])
    cfg = dict(_LR0, live_lr=LR, wake_ticks=100000, wake_every=10 ** 6, gate_every=10 ** 6, write_floor=1e-30, gate_floor=0.5, fast_rls=0,
               act_inv_lr=1e-2, act_inv_tau=2000)
    w = _arm_world(); torch.manual_seed(5); L = _born_in(_Timed(TOK, cfg), cfg, w)
    run = WorldLoop(L)
    for _ in range(400):                                          # act_inv learns on the arm's own acts (no waking lesson: wake_every)
        run.step()
    mixed = list(L.win)
    _closed_arm(L)
    for _ in range(40):                                           # rests alone: nothing moves the arm
        run.step()
    rests = list(L.win)
    assert all(r_ == (12, "still") for r_ in w.moved[-32:])
    for _ in range(40):                                           # the parent's guidance alone: the hand on every tick
        w.guide[:] = [[6, 7, 11, 13, 8, 16, 17, 18][len(w.moved) % 8]]
        run.step()
    guide = list(L.win)
    assert all(who == "guide" for _, who in w.moved[-32:])
    with torch.no_grad():
        L.m.gates["arm"].bias.fill_(60.0)                         # the gate wide open: the arm acts (a chunk may end at its rest)
    for _ in range(40):
        run.step()
    own = list(L.win)
    assert sum(1 for _, who in w.moved[-32:] if who == "own") >= 24, w.moved[-32:]
    n_lab = {k: _proposal_error(L, v)[1] for k, v in (("rests", rests), ("guidance", guide), ("mixed", mixed))}
    assert n_lab["rests"] == n_lab["guidance"] == 30 and 5 <= n_lab["mixed"] <= 25 and L.motor[0]["inv_gain"] > 0.3, (n_lab, L.motor[0]["inv_gain"])
    fresh = _snap(L)
    _lessons_from(L, fresh, mixed, 0.0, 100)                      # own acts first: the moments built by its own acts (the rests unlabelled)
    built = _snap(L)
    L.m.load_state_dict(fresh[0])
    for k, sd in fresh[1].items():
        getattr(L, k).load_state_dict(copy.deepcopy(sd))
    _waking_rate(L, 100 * LR)                                     # trained: 300 lessons of its own acts at a hundred times the rate
    for _ in range(300):
        L.motor[0]["inv_gain"] = 1.0; L.win.clear(); L.win.extend(own); L._wake_lesson()
    _waking_rate(L, LR)
    trained = _snap(L)
    N = 300; gains = (0.0, 0.01, 0.05, 0.25, 1.0); shares = {}; moves = {}; clean = {}; same = {}
    for reg, snap in (("fresh", fresh), ("own acts first", built), ("trained", trained)):
        for name, win in (("rests", rests), ("guidance", guide), ("mixed", mixed)):
            L.m.load_state_dict(snap[0]); th0 = _arm_gated(L)
            res = {}; st = {}
            for g in gains:
                _lessons_from(L, snap, win, g, N)
                res[g] = (_proposal_error(L, win)[0], float((_arm_gated(L) - th0).norm())); st[g] = _stream_params(L)
            full = res[0.0][0] - res[1.0][0]
            shares[(reg, name)] = (full, {g: (res[0.0][0] - res[g][0]) / full for g in gains})
            moves[(reg, name)] = {g: res[g][1] / res[1.0][1] for g in gains}
            same[(reg, name)] = all(torch.equal(a_, b_) for g in gains[1:] for a_, b_ in zip(st[0.0], st[g]))
            if name != "mixed":                                   # the same label mass given whole: every twentieth lesson at 1, the rest at 0
                _lessons_from(L, snap, win, lambda i: 1.0 if i % 20 == 19 else 0.0, N)
                clean[(reg, name)] = (res[0.0][0] - res[0.05][0]) / (res[0.0][0] - _proposal_error(L, win)[0])
    table = {k: {g: round(v[1][g], 3) for g in gains[1:-1]} for k, v in shares.items()}
    said = (table, {k: round(v, 3) for k, v in clean.items()})
    for key, (full, share) in shares.items():
        assert full > 0.05, (key, "the whole lesson does not teach", full, said)
        assert all(share[a] < share[b] for a, b in zip(gains, gains[1:])), (key, "the share learned does not rise", said)
        assert share[0.01] <= 5 * 0.01 and share[0.05] <= 5 * 0.05, (key, "learned more than the reliability allows", said)
    for key, r_ in clean.items():
        assert 0.5 <= r_ <= 2.0, (key, "the spread labels taught otherwise than the same mass given whole", said)
    assert all(v_ for k_, v_ in same.items() if k_[1] != "mixed"), ("the stream learned act_inv's labels", [k for k, v in same.items() if not v], said)
    for key, mv in moves.items():
        if key[1] != "mixed":                                     # only the rests' labels teach: act_pred moves by the reliability
            assert mv[0.0] == 0.0 and mv[0.01] <= 5 * 0.01 and mv[0.05] <= 5 * 0.05, (key, mv)
    # THE MIXED WINDOW ONCE THE OWN ACTS ARE FITTED (the R6 verifier's fifth look): own acts beside act_inv's labelled rests, the life as
    # lived, from three states whose act_pred has learned own acts: mixfit (2000 lessons of the mixed window, its rests unlabelled, at a
    # hundred times the served rate), trained (above) and own acts long (3000 lessons of the arm's own acts at the served rate); 1000
    # lessons at 0, 0.01, 0.05 and 1, and the same label mass given whole (every hundredth lesson at 1, every twentieth), the error read
    # after 300 and after 1000. With the stream held still over the measured lessons (the day's rate 0; each state built with it
    # learning), so the proposal's error moves by act_pred's and the correction's two samples alone; and, from mixfit, through the
    # whole waking lesson, the stream learning its own acts as it lives
    for reg, n_, lr_, win_, g_ in (("mixfit", 2000, 100 * LR, mixed, 0.0), ("own acts long", 3000, LR, own, 1.0)):
        L.m.load_state_dict(fresh[0])
        for k, sd in fresh[1].items():
            getattr(L, k).load_state_dict(copy.deepcopy(sd))
        _waking_rate(L, lr_)
        for _ in range(n_):
            L.motor[0]["inv_gain"] = g_; L.win.clear(); L.win.extend(win_); L._wake_lesson()
        _waking_rate(L, LR)
        if reg == "mixfit":
            mixfit = _snap(L)
        else:
            ownlong = _snap(L)

    def still(snap):
        """a state's copy whose day's Adam steps at rate 0: the stream held still, the lesson otherwise whole"""
        c_ = (snap[0], copy.deepcopy(snap[1]))
        for g_ in c_[1]["opt_day"]["param_groups"]:
            g_["lr"] = 0.0
        return c_
    AT = (300, 1000); fit = {}
    for reg, snap in (("mixfit", still(mixfit)), ("trained", still(trained)), ("own acts long", still(ownlong)), ("mixfit, the stream learning", mixfit)):
        e_ = {g: _curve_from(L, snap, mixed, g, 1000, AT) for g in (0.0, 0.01, 0.05, 1.0)}
        if reg == "mixfit":                                       # held still: after 1000 lessons at 1 the day's parameters as the state left them
            day_ = {id(q_) for g_ in L.opt_day.param_groups for q_ in g_["params"]}
            assert all(torch.equal(p_.detach(), snap[0][k_]) for k_, p_ in L.m.named_parameters() if id(p_) in day_), "the stream moved"
        for g, k_ in ((0.01, 100), (0.05, 20)):
            e_[("mass", g)] = _curve_from(L, snap, mixed, lambda i, k_=k_: 1.0 if i % k_ == k_ - 1 else 0.0, 1000, AT)
        for n_ in AT:
            full = e_[0.0][n_] - e_[1.0][n_]
            fit[(reg, n_)] = (full, {g: (e_[0.0][n_] - e_[g][n_]) / full for g in (0.01, 0.05)},
                              {g: (e_[0.0][n_] - e_[g][n_]) / (e_[0.0][n_] - e_[("mass", g)][n_]) for g in (0.01, 0.05)})
    said_fit = {f"{k[0]} {k[1]}": (round(v[0], 3), {g: round(x, 4) for g, x in v[1].items()}, {g: round(x, 3) for g, x in v[2].items()})
                for k, v in fit.items()}
    for key, (full, share, ratio) in fit.items():
        assert full > 0.05, (key, "the whole lesson does not teach", said_fit)
        assert share[0.01] <= 5 * 0.01 and share[0.05] <= 5 * 0.05, (key, "beside fitted own acts the labels learned more than the reliability allows", said_fit)
        assert all(r_ <= 2.0 for r_ in ratio.values()), (key, "beside fitted own acts the spread labels taught more than the same mass given whole", said_fit)
        if "learning" not in key[0]:                              # (the lower side, read where the stream is still: see the docstring)
            assert 0.0 < share[0.01] < share[0.05], (key, "the share learned does not rise", said_fit)
            assert all(0.5 <= r_ for r_ in ratio.values()), (key, "beside fitted own acts the spread labels taught less than the same mass given whole", said_fit)
    # THE OPTIMIZERS: act_pred and the corrections in opt_pred (the own acts' sample), a group per effector, and the arm's (the one with
    # an inverse model) again in opt_lab (act_inv's labels' sample, moments of its own); every other parameter in the day's Adam as before
    from body.core.timing import GatedAdam
    tim = L.m.timing
    want = {"arm": [tim["arm"].pred.weight, tim["arm"].pred.bias, tim["arm"].cor.weight], "grip": [tim["grip"].pred.weight, tim["grip"].pred.bias,
            tim["grip"].cor.weight], "tap": [tim["tap"].pred.weight, tim["tap"].pred.bias]}
    assert isinstance(L.opt_pred, GatedAdam) and [g_["name"] for g_ in L.opt_pred.param_groups] == ["arm", "grip", "tap"]
    assert [[id(p_) for p_ in g_["params"]] for g_ in L.opt_pred.param_groups] == [[id(p_) for p_ in want[n_]] for n_ in ("arm", "grip", "tap")]
    assert isinstance(L.opt_lab, GatedAdam) and L.opt_lab is not L.opt_pred and [g_["name"] for g_ in L.opt_lab.param_groups] == ["arm"]
    assert [id(p_) for p_ in L.opt_lab.param_groups[0]["params"]] == [id(p_) for p_ in want["arm"]]
    gated_ids = {id(p_) for v_ in want.values() for p_ in v_}
    assert len(L.opt_day.param_groups) == 1 and [id(p_) for p_ in L.opt_day.param_groups[0]["params"]] == [id(p_) for p_ in L.m.parameters() if id(p_) not in gated_ids]
    d_, p_ = L.opt_day.param_groups[0], L.opt_pred.param_groups[0]
    assert all(g_["lr"] == LR for g_ in L.opt_pred.param_groups + L.opt_lab.param_groups) and d_["lr"] == LR
    assert (tuple(d_["betas"]), d_["eps"]) == (p_["betas"], p_["eps"]) == (L.opt_lab.param_groups[0]["betas"], L.opt_lab.param_groups[0]["eps"])
    # each sample's gain its labels' mean weight over the window's positions: rests alone at reliability 0.25, 30 labels of 31
    # positions, all act_inv's (the own acts' sample none); on the mixed window the own acts' share and the labels' at the
    # reliability, their sum the lesson's mean label weight; the grip and the tap (no inverse model) 1, their own acts' alone
    out = _lessons_from(L, fresh, rests, 0.25, 1)
    a_ = out["motor"]["arm"]
    assert (a_["w"], a_["w_own"], a_["w_lab"]) == (round(0.25 * 30 / 31, 4), 0.0, round(0.25 * 30 / 31, 4)), out["motor"]
    assert all((out["motor"][k_]["w"], out["motor"][k_]["w_own"], out["motor"][k_]["w_lab"]) == (1.0, 1.0, 0.0) for k_ in ("grip", "tap")), out["motor"]
    T_ = len(mixed); n_own_ = sum(1 for x_ in mixed[1:] if int(x_[L.anatomy.effectors[1].field]) != 12)
    n_lab_ = T_ - 1 - n_own_ - (1 if int(mixed[-1][L.anatomy.effectors[1].field]) == 12 else 0)
    a_ = _lessons_from(L, fresh, mixed, 0.37, 1)["motor"]["arm"]
    assert (a_["w_own"], a_["w_lab"]) == (round(n_own_ / (T_ - 1), 4), round(0.37 * n_lab_ / (T_ - 1), 4)) and abs(a_["w"] - a_["w_own"] - a_["w_lab"]) <= 1e-4, (a_, n_own_, n_lab_)
    # at gain 0 nothing of the arm's moves, both samples' moments included (rests alone at reliability 0, after the own acts' lessons)
    _lessons_from(L, built, rests, 0.0, 1)
    ia = [k_ for k_ in range(3)]                                  # the arm's parameters come first in opt_pred, alone in opt_lab
    assert torch.equal(_arm_gated(L), torch.cat([built[0][k_].flatten() for k_ in ("timing.arm.pred.weight", "timing.arm.pred.bias", "timing.arm.cor.weight")]))
    for k_o in ("opt_pred", "opt_lab"):
        sb = built[1][k_o]["state"]; sn = getattr(L, k_o).state_dict()["state"]
        assert sorted(sb) == sorted(sn) and all(torch.equal(sb[k_]["m"], sn[k_]["m"]) and torch.equal(sb[k_]["v"], sn[k_]["v"]) and (sb[k_]["q1"], sb[k_]["q2"]) == (sn[k_]["q1"], sn[k_]["q2"]) for k_ in ia if k_ in sb), k_o
    # EACH SAMPLE'S MOMENTS HOLD ITS OWN GRADIENT: one lesson from fresh on the mixed window at reliability 0.37; the arm's first moment
    # in opt_pred lies along the own acts' gradient alone, in opt_lab along act_inv's labels' alone (each recomputed apart on the same
    # state; the lesson's scale and each sample's bound scale them, not their direction)
    L.m.load_state_dict(fresh[0]); L.motor[0]["inv_gain"] = 0.37; L.win.clear(); L.win.extend(mixed)
    obs_, whos_, bundles_, _ = L._window_tensors()
    L.m.train(); L.m.zero_grad(set_to_none=True)
    lt_, _, lb_ = L._timing_loss(1, L.m.stream(L.m.inputs(L.anatomy, obs_, whos_, bundles_)), obs_)
    g_lab = torch.cat([x_.flatten() for x_ in torch.autograd.grad(lb_, want["arm"], retain_graph=True)])
    lt_.backward(); g_own = torch.cat([p_.grad.flatten() for p_ in want["arm"]]).clone(); L.m.zero_grad(set_to_none=True); L.m.eval()
    _lessons_from(L, fresh, mixed, 0.37, 1)
    m_own = torch.cat([L.opt_pred.state[p_]["m"].flatten() for p_ in want["arm"]]); m_lab = torch.cat([L.opt_lab.state[p_]["m"].flatten() for p_ in want["arm"]])
    cos_ = {k_: round(float(a_ @ b_ / (a_.norm() * b_.norm())), 6) for k_, (a_, b_) in (("own", (m_own, g_own)), ("labels", (m_lab, g_lab)), ("apart", (g_own, g_lab)))}
    assert cos_["own"] > 0.9999 and cos_["labels"] > 0.9999 and abs(cos_["apart"]) < 0.99, cos_
    # at weight 1 on every lesson it is Adam: 20 lessons on a window of own acts alone (the open gate's window, its rests replaced by an
    # act of the arm's own: every position of every effector an own act or an uninverted rest, weight 1, no label), then again with act_pred
    # and the corrections in a torch Adam of the same groups under the same bounds: every tensor of the body the same
    f_ = L.anatomy.effectors[1].field
    allown = [dict(x_, **{f_: (13 if int(x_[f_]) == 12 else int(x_[f_]))}) for x_ in own]
    out = _lessons_from(L, fresh, allown, 1.0, 20)
    assert all(v_["w"] == v_["w_own"] == 1.0 and v_["w_lab"] == 0.0 for v_ in out["motor"].values()), out["motor"]
    sA = copy.deepcopy(L.m.state_dict())
    keep = L.opt_pred
    L.m.load_state_dict(fresh[0]); L.opt_day.load_state_dict(copy.deepcopy(fresh[1]["opt_day"]))
    L.opt_pred = torch.optim.Adam([dict(params=list(g_["params"]), name=g_["name"], gain=0.0) for g_ in keep.param_groups], lr=LR)
    for _ in range(20):
        L.win.clear(); L.win.extend(allown); L._wake_lesson()
    sB = L.m.state_dict(); L.opt_pred = keep
    same_t = [k_ for k_ in sA if torch.equal(sA[k_], sB[k_])]
    assert len(same_t) == len(sA), [k_ for k_ in sA if k_ not in same_t]
    # on a constant gradient scaled by g: Adam re-inflates it to whole steps, GatedAdam steps g of them
    G = torch.tensor([0.5, -2.0, 1e-3]); n = 50; walk = {}
    for g in (1.0, 0.25, 0.01):
        pa = torch.nn.Parameter(torch.zeros(3)); pg = torch.nn.Parameter(torch.zeros(3))
        oa = torch.optim.Adam([pa], lr=LR); og = GatedAdam([{"params": [pg], "name": "x"}], lr=LR)
        for _ in range(n):
            pa.grad = g * G; oa.step(); pg.grad = g * G; og.param_groups[0]["gain"] = g; og.step()
        sa = (-pa.detach() / (LR * G.sign())); sg = (-pg.detach() / (LR * G.sign()))
        walk[g] = (round(float(sa.min()), 2), round(float(sg.max()), 3))
        assert float(sa.min()) >= 0.99 * n and float(sa.max()) <= 1.001 * n, (g, sa)
        assert float((sg - n * g).abs().max()) <= 1e-3 * n * g + 1e-6, (g, sg)
    # the moments move in proportion, on the gradient per unit weight: after 50 whole steps on G, one step at gain g on g G2 (a lesson's
    # gradient at reliability g) moves the first moment (1 - beta1) g of its way to G2 and the second (1 - beta2) g of its way to G2^2
    # (a barely earned lesson's direction barely enters the momentum later whole lessons step along)
    G2 = torch.tensor([-1.0, 0.5, 2.0])
    for g in (1.0, 0.25, 0.01):
        pg = torch.nn.Parameter(torch.zeros(3)); og = GatedAdam([{"params": [pg], "name": "x"}], lr=LR)
        for _ in range(n):
            pg.grad = G.clone(); og.param_groups[0]["gain"] = 1.0; og.step()
        m0_, v0_ = og.state[pg]["m"].clone(), og.state[pg]["v"].clone()
        pg.grad = g * G2; og.param_groups[0]["gain"] = g; og.step()
        assert torch.allclose(og.state[pg]["m"], m0_ + (1.0 - 0.9) * g * (G2 - m0_), rtol=1e-5, atol=1e-8), ("the first moment did not move (1 - beta1) g of its way", g, og.state[pg]["m"], m0_)
        assert torch.allclose(og.state[pg]["v"], v0_ + (1.0 - 0.999) * g * (G2 * G2 - v0_), rtol=1e-5, atol=1e-10), ("the second moment did not move (1 - beta2) g of its way", g, og.state[pg]["v"], v0_)
    # the language body: no opt_pred and no opt_lab, its day's Adam holds every parameter as before
    D = _born(TOK, dict(wake_ticks=100000))
    assert not hasattr(D, "opt_pred") and not hasattr(D, "opt_lab") and len(D.opt_day.param_groups) == 1 and [id(p_) for p_ in D.opt_day.param_groups[0]["params"]] == [id(p_) for p_ in D.m.parameters()]
    # THE NIGHT does not teach act_pred (R8 will, through the gate): a night at the night's rate, after the trained state's lessons (the
    # open gate's window at reliability 1: both samples' moments built), leaves act_pred, the corrections and both samples' moments
    _lessons_from(L, trained, own, 1.0, 1)
    gp_ = [p_ for g_ in L.opt_pred.param_groups for p_ in g_["params"]]
    g0_ = [p_.detach().clone() for p_ in gp_]; o0_ = {k_o: copy.deepcopy(getattr(L, k_o).state_dict()) for k_o in ("opt_pred", "opt_lab")}; s0_ = _stream_params(L)
    assert all(len(o0_[k_o]["state"]) >= 3 for k_o in o0_), "the guard would be empty: a sample has no moments"
    L.cfg["night_lr"] = 1e-3
    rep_ = L.night()
    assert all(torch.equal(a_, p_.detach()) for a_, p_ in zip(g0_, gp_)), "the night moved act_pred or a correction"
    for k_o in ("opt_pred", "opt_lab"):
        o1_ = getattr(L, k_o).state_dict()
        assert all(torch.equal(o0_[k_o]["state"][k_]["m"], o1_["state"][k_]["m"]) and torch.equal(o0_[k_o]["state"][k_]["v"], o1_["state"][k_]["v"])
                   and (o0_[k_o]["state"][k_]["q1"], o0_[k_o]["state"][k_]["q2"]) == (o1_["state"][k_]["q1"], o1_["state"][k_]["q2"]) for k_ in o0_[k_o]["state"]), k_o
    nmv_ = sum(1 for a_, b_ in zip(s0_, _stream_params(L)) if not torch.equal(a_, b_))
    assert nmv_ > 0, ("the night moved nothing: the guard would be empty", rep_)
    print(f"anatomy 31: what act_pred learns scales with its labels' reliability, at the served rate {LR} through the real waking lesson",
          f"(300 lessons on each window held fixed; the share learned at 0.01, 0.05, 0.25): {table}; the same label mass given whole",
          f"(300 lessons at 0.05 against every twentieth at 1): {said[1]}; the stream the same to the bit at every reliability",
          f"({sum(1 for k_ in same if k_[1] != 'mixed')} windows and states with no own act); THE MIXED WINDOW BESIDE FITTED OWN ACTS (1000",
          f"lessons; the whole lesson's gain, the share at 0.01 and 0.05, the same mass given whole at 0.01 and 0.05, after 300 and 1000):",
          f"{said_fit}; opt_pred holds act_pred and the corrections (a group per effector; the own acts' sample), opt_lab the arm's again",
          f"(act_inv's labels' sample, moments of its own: its first moment along the labels' gradient, cosines {cos_}), the day's Adam",
          f"everything else, each bounded by its own norm; each sample's gain its labels' mean weight; at weight 1 Adam to the bit",
          f"({len(same_t)} tensors after 20 lessons of own acts); at gain 0 nothing moves; a constant gradient scaled by g, n = {n} steps",
          f"(Adam's least, GatedAdam's most, in steps of the rate): {walk}; a night moved {nmv_} of the stream's tensors and none of",
          f"act_pred's or the corrections' or either sample's moments")


def test_a_striatum_saved_before_r5b_loads():
    """anatomy 32 (the R5b verifier's finding, 2026-09-24): a save of a body with later effectors taken before R5b holds their striatal
    block as a row for every flat act, so its striatum's shape is not the body's and the loader kept none of it, the voice's heads, its
    line and its slot included. Now the save's language block (the first k (2V + 3) rows), thresholds, lines, heads (the fast critic's
    and the voice's actor), working-memory slot and the effectors' actors are kept, said once, and only the effectors' rows are born
    again per joint, as the body's birth draws them; a save of the present layout loads whole as before; the loaded body lives on"""
    import contextlib
    import io
    cfg = dict(wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, fast_rls=1, fast_input="striatum", actor=1, stri_k=8,
               stri_m=64, wm=1, gate_floor=0.3)
    L = _born(_Arm(TOK, cfg), cfg); _live(L, ticks=40); V = L.m.vocab; k = 8
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():                                          # every held or learned piece at values of its own, so a copy shows
        for t_ in (L.m.vfast.weight, L.m.vfast.bias, L.m.actor.weight, L.m.actor.bias, L.m.actors["arm"].weight, L.m.actors["arm"].bias,
                   L.m.actors["grip"].weight, L.m.actors["grip"].bias, L.m.wm_slot, L.m.stri_b):
            t_.copy_(torch.randn(t_.shape, generator=g))
        L.m.wm_on.fill_(1.0); L.m.wm_age.fill_(3.0)
        L.m.stri_line.copy_(torch.randint(0, 2 * V + 3, (k,), generator=g))
        L.m.stri_mline.copy_(torch.tensor([[int(x) for x in torch.randint(0, 25, (k,), generator=g)], [int(x) for x in torch.randint(0, 3, (k,), generator=g)]]))
    nl = k * (2 * V + 3)
    assert L.m.stri_W.shape == (nl + k * (10 + 3), 64)
    held = ["stri_b", "stri_line", "vfast.weight", "vfast.bias", "actor.weight", "actor.bias", "wm_slot", "wm_on", "wm_age", "stri_mline",
            "actors.arm.weight", "actors.arm.bias", "actors.grip.weight", "actors.grip.bias"]
    sL = {k_: v_.detach().clone() for k_, v_ in L.m.state_dict().items()}
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd); fd2, path2 = tempfile.mkstemp(suffix=".pt"); os.close(fd2)
    said = io.StringIO()
    try:
        L.save(path); blob = torch.load(path, map_location="cpu", weights_only=False)
        old = copy.deepcopy(blob)                                  # the layout before R5b: the language block, then k rows per flat act
        old["organs"]["stri_W"] = torch.cat([blob["organs"]["stri_W"][:nl], torch.randn(k * (25 + 3), 64, generator=g)])
        torch.save(old, path2)
        with contextlib.redirect_stdout(said):
            C = Life.load(path, _Arm(TOK, L.cfg), save_path=None)
        now_said = said.getvalue(); said = io.StringIO()
        with contextlib.redirect_stdout(said):
            O = Life.load(path2, _Arm(TOK, L.cfg), save_path=None)
    finally:
        os.remove(path); os.remove(path2)
    sC, sO = C.m.state_dict(), O.m.state_dict()
    assert all(torch.equal(sL[k_], sC[k_]) for k_ in sL) and "R5b" not in now_said, "the present layout does not load whole"
    note = [l_ for l_ in said.getvalue().splitlines() if "before R5b" in l_]
    assert len(note) == 1 and f"{nl + k * 28} rows" in note[0] and f"({nl + k * 13} rows)" in note[0], said.getvalue()
    assert O.m.stri_W.shape == L.m.stri_W.shape and torch.equal(sO["stri_W"][:nl], sL["stri_W"][:nl]), "the language block"
    assert torch.equal(sO["stri_W"][nl:], sL["stri_W"][nl:]), "the effectors' rows are not born again per joint as the birth draws them"
    assert all(torch.equal(sO[k_], sL[k_]) for k_ in held), [k_ for k_ in held if not torch.equal(sO[k_], sL[k_])]
    assert float(sL["vfast.weight"].abs().max()) > 0 and float(sL["actor.weight"].abs().max()) > 0      # (zeros, as born, before the fix)
    assert all(torch.equal(sO[k_], sL[k_]) for k_ in sL), [k_ for k_ in sL if not torch.equal(sO[k_], sL[k_])]
    _live(O, ticks=10); assert O.ticks == L.ticks + 10
    print(f"anatomy 32: a striatum saved before R5b ({nl + k * 28} rows, a row per flat act) loads with its language block ({nl} rows),",
          f"thresholds, lines, heads, slot and actors kept ({len(held)} tensors) and the effectors' rows born again per joint ({k * 13} rows,",
          f"as the birth drew them), said once; the present layout loads whole; the loaded body lives on")


def test_act_preds_moments_survive_a_reload():
    """anatomy 33 (R6 fix 5, the R6 verifier's fifth look): a body with later effectors saves act_pred's and the correction's moments,
    each sample's (the own acts' in opt_pred, act_inv's labels' in opt_lab), under its life["motor"]; a reload gives them back to the
    bit (the first and second moments and each one's share still at birth, q1 and q2), and the reloaded body's next lesson is the saved
    body's to the bit. A save from before them (R6 fix 5, 2026-09-24: its motor state without "moments") loads with a note, said once,
    its moments born again, and lives on. The language body's save has no motor state, as before"""
    import contextlib
    import io
    from body.core.world import WorldLoop
    cfg = dict(_LR0, live_lr=1e-3, wake_ticks=100000, wake_every=10 ** 6, gate_every=10 ** 6, write_floor=1e-30, gate_floor=0.5, fast_rls=0,
               act_inv_lr=1e-2, act_inv_tau=2000)
    w = _arm_world(); torch.manual_seed(5); L = _born_in(_Timed(TOK, cfg), cfg, w)
    run = WorldLoop(L)
    for _ in range(80):
        run.step()
    win = list(L.win)
    for _ in range(5):                                            # both samples step: own acts beside act_inv's labelled rests
        L.motor[0]["inv_gain"] = 0.4; L.win.clear(); L.win.extend(win); out = L._wake_lesson()
    assert out["motor"]["arm"]["w_own"] > 0 and out["motor"]["arm"]["w_lab"] > 0, out["motor"]
    names = {e_.name: L._gated_names(e_) for e_ in L.anatomy.effectors[1:]}
    assert names["arm"] == ["pred.weight", "pred.bias", "cor.weight"] and names["tap"] == ["pred.weight", "pred.bias"]

    def moments(B):
        return {(k_o, e_.name, n_): (st_["m"].clone(), st_["v"].clone(), st_["q1"], st_["q2"])
                for k_o in ("opt_pred", "opt_lab") for g_ in getattr(B, k_o).param_groups for e_ in B.anatomy.effectors[1:] if g_["name"] == e_.name
                for n_, p_ in zip(B._gated_names(e_), g_["params"]) for st_ in [getattr(B, k_o).state.get(p_)] if st_}
    mL = moments(L)
    assert len(mL) == 3 + 3 + 2 + 3, sorted(mL)                   # the arm's, the grip's and the tap's own acts; the arm's labels
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd); fd2, path2 = tempfile.mkstemp(suffix=".pt"); os.close(fd2)
    said = io.StringIO()
    try:
        L.save(path); blob = torch.load(path, map_location="cpu", weights_only=False)
        assert set(blob["life"]["motor"]["arm"]["moments"]) == {"own", "labels"} and set(blob["life"]["motor"]["grip"]["moments"]) == {"own"}
        old = copy.deepcopy(blob)                                  # a save from before R6 fix 5: no moments
        for v_ in old["life"]["motor"].values():
            v_.pop("moments")
        torch.save(old, path2)
        with contextlib.redirect_stdout(said):
            C = Life.load(path, _Timed(TOK, L.cfg), save_path=None, world=_arm_world())
        now_said = said.getvalue(); said = io.StringIO()
        with contextlib.redirect_stdout(said):
            O = Life.load(path2, _Timed(TOK, L.cfg), save_path=None, world=_arm_world())
    finally:
        os.remove(path); os.remove(path2)
    mC = moments(C)
    assert sorted(mC) == sorted(mL) and all(torch.equal(mC[k_][0], mL[k_][0]) and torch.equal(mC[k_][1], mL[k_][1]) and mC[k_][2:] == mL[k_][2:] for k_ in mL)
    assert "moments" not in now_said, now_said
    note = [l_ for l_ in said.getvalue().splitlines() if "moments were not saved" in l_]
    assert len(note) == 1 and "R6 fix 5" in note[0] and "'arm'" in note[0], said.getvalue()
    assert moments(O) == {}, "a save without moments loaded some"
    # the reloaded body's next lesson is the saved body's, to the bit (the organs and both samples' moments given back; the day's Adam
    # born again in both, as every load has it: the saved body's is reset here to match): every tensor the load gave back (the cortex
    # and the timing organs among them; this body has no striatum, so its heads are born again as before) the same after the lesson
    sL_, sC_ = L.m.state_dict(), C.m.state_dict(); a0_ = L.m.timing["arm"].pred.weight.detach().clone()
    back = [k_ for k_ in sL_ if torch.equal(sL_[k_], sC_[k_])]
    assert all(k_ in back for k_ in sL_ if k_.startswith("timing.") or k_.startswith("blocks.")), [k_ for k_ in sL_ if k_ not in back]
    for B in (L, C):
        B.opt_day.state.clear()
        B.motor[0]["inv_gain"] = 0.4; B.win.clear(); B.win.extend(win); B._wake_lesson()
    sL_, sC_ = L.m.state_dict(), C.m.state_dict()
    assert not torch.equal(a0_, sL_["timing.arm.pred.weight"]) and all(torch.equal(sL_[k_], sC_[k_]) for k_ in back), [k_ for k_ in back if not torch.equal(sL_[k_], sC_[k_])]
    for B in (C, O):
        for _ in range(5):
            B.tick()
    O.motor[0]["inv_gain"] = 0.4; O.win.clear(); O.win.extend(win); out = O._wake_lesson()
    assert out["motor"]["arm"]["w_lab"] > 0 and len(moments(O)) == len(mL), (out["motor"], sorted(moments(O)))
    D = _born(TOK, dict(wake_ticks=100000))
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    try:
        D.save(path); bd = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        os.remove(path)
    assert "motor" not in bd["life"]
    print(f"anatomy 33: act_pred's and the corrections' moments, each sample's ({len(mL)} tensors' moments: the own acts' of the arm, the",
          f"grip and the tap, act_inv's labels' of the arm), saved and given back to the bit, the reloaded body's next lesson the saved",
          f"body's to the bit; a save from before them loads with the note said once, its moments born again, and lives on; the language",
          f"body's save has no motor state")


ANATOMY_TESTS = [test_language_anatomy_equals_the_tokenizers_fields, test_language_anatomy_is_inert, test_anatomy_check,
                 test_life_reads_its_anatomy, test_an_anatomy_in_the_tokenizers_place, test_the_body_reads_text_through_its_anatomy,
                 test_reward_sources_feel_todays_rule, test_a_life_feels_as_before, test_the_declared_order_is_the_sums,
                 test_the_input_is_the_channels_in_order, test_the_window_holds_each_channel_under_its_field, test_a_later_channel,
                 test_every_call_site_passes_the_channels, test_imagination_as_before, test_the_voice_is_effector_0,
                 test_the_gate_lesson_as_before, test_the_switches, test_a_later_effector, test_every_call_site_passes_the_effectors,
                 test_the_diary_world, test_a_world_of_frames, test_the_loop_deadline_and_pace,
                 test_the_timing_part_is_built_last, test_act_inv_learns_online, test_act_pred_targets, test_the_forward_half,
                 test_the_learned_stops, test_demonstrations_count_as_earned, test_act_inv_reliability_is_kappa,
                 test_a_34_joint_body_builds, test_act_pred_learns_by_reliability,
                 test_a_striatum_saved_before_r5b_loads, test_act_preds_moments_survive_a_reload]

if __name__ == "__main__":
    t0 = time.time(); failed = 0
    for t in ANATOMY_TESTS:
        try:
            t()
        except AssertionError as e:
            failed += 1; print("FAIL", t.__name__, ":", e)
        except Exception as e:
            failed += 1; print("ERROR", t.__name__, ":", type(e).__name__, str(e)[:300])
    print(f"{len(ANATOMY_TESTS) - failed}/{len(ANATOMY_TESTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if failed else 0)
