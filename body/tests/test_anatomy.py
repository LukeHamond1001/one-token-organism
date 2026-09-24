"""the anatomy declared (docs/SIM_DESIGN.md 8.2 and 8.3; the core refactor's steps R1 to R4). Run: python3 -m body.tests.test_anatomy
(the organ tests run these too). `LanguageAnatomy(tok, cfg)` must rebuild exactly the symbols a life derived from its tokenizer before
R2, under every constant that moves them and on a tokenizer laid out otherwise, and building it must leave the body untouched (R1). The
life is built with it and reads its symbols and its text there, never the tokenizer; a life given the anatomy in the tokenizer's place
is the same life; an anatomy declared under other constants, or not a language one, is refused (R2). The tick's reward is the
anatomy's reward sources, felt in their declared order and added one at a time in it: bit for bit the reward `_sense` summed before
R3, on every tick, under the switches the pinned digests do not reach (cost_in_reward, world_mask off, own_store) (R3). The cortex's
input is the anatomy's channel codes summed in its declared order, bit for bit the sum before R4 for the diary's (gradients too); the
window holds each channel under its declared field and every reader goes through it; a later channel's forecast head is built last
and taught; all eleven places the input is made pass the anatomy's channels (R4)."""
import collections
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
                                LanguageAnatomy, RewardSource, WorldWordsReward, anatomy_for)
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
    a bare RewardSource declares no feeling"""
    def ok():
        return Anatomy([Channel("ear", "symbol", 5, organ="E", forecast=True, rest_id=0, end_id=1, reserved=[4], partner=True),
                        Channel("face", "vector", 2, organ="face_in")],
                       [Effector("voice", [5], rest_id=0, end_id=2, reserved=[4])], [RewardSource("face"), RewardSource("cost", ("symbol_cost",))])
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
    for call in (lambda: Channel("x", "symbol", 3).observe(None, 0, 0), lambda: Effector("v", [3]).gate_inputs(None, None),
                 lambda: Effector("v", [3]).cost(0, None)):
        try:
            call()
        except NotImplementedError:
            continue
        raise AssertionError("a method not yet wired answered")
    try:
        RewardSource("r").felt(Frame(0, {"ear": 0}, 0.0), None)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("a bare RewardSource felt something")
    print("anatomy 3: the check refuses", len(bad), "faulty declarations; the methods of later steps are not yet wired;",
          "a bare reward source feels nothing")


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
    until step R5"""
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
                    [Effector("arm", [5, 5], rest_id=12)], [RewardSource("face")]).check()
    for call in (lambda: anatomy_for(other, {}), lambda: _born(other, {}), lambda: Life(A.m, other)):
        try:
            call()
        except NotImplementedError:
            continue
        raise AssertionError("a life was built on an anatomy not of language before step R5")
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
        return Anatomy(chans, [Effector("v", [4], rest_id=0)], [RewardSource("r")], inner_at=inner_at).check()
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


ANATOMY_TESTS = [test_language_anatomy_equals_the_tokenizers_fields, test_language_anatomy_is_inert, test_anatomy_check,
                 test_life_reads_its_anatomy, test_an_anatomy_in_the_tokenizers_place, test_the_body_reads_text_through_its_anatomy,
                 test_reward_sources_feel_todays_rule, test_a_life_feels_as_before, test_the_declared_order_is_the_sums,
                 test_the_input_is_the_channels_in_order, test_the_window_holds_each_channel_under_its_field, test_a_later_channel,
                 test_every_call_site_passes_the_channels, test_imagination_as_before]

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
