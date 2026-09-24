"""the anatomy declared (docs/SIM_DESIGN.md 8.2 and 8.3; the core refactor's steps R1 and R2). Run: python3 -m body.tests.test_anatomy
(the organ tests run these too). `LanguageAnatomy(tok, cfg)` must rebuild exactly the symbols a life derived from its tokenizer before
R2, under every constant that moves them and on a tokenizer laid out otherwise, and building it must leave the body untouched (R1). The
life is built with it and reads its symbols and its text there, never the tokenizer; a life given the anatomy in the tokenizer's place
is the same life; an anatomy declared under other constants, or not a language one, is refused (R2)."""
import collections
import os
import pickle
import sys
import time

import torch
from tokenizers import Tokenizer, models

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)   # this tree's body, not a fixed one
from body.life import Life, PHYSIOLOGY  # noqa: E402
from body.core.anatomy import Anatomy, Channel, Effector, LanguageAnatomy, RewardSource, anatomy_for  # noqa: E402

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
    assert (voice.name, voice.factors, voice.rest_id, voice.end_id, voice.reserved) == ("voice", [life.m.vocab], life.sil, life.space_id, life.bans), label
    assert [s.name for s in a.rewards] == ["face", "world_r", "cost"], label
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
    unknown kind, a name twice, no effector, and a reward source reading an unknown constant"""
    def ok():
        return Anatomy([Channel("ear", "symbol", 5, rest_id=0, end_id=1, reserved=[4], partner=True), Channel("face", "vector", 2)],
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
    for label, a in bad:
        try:
            a.check()
        except ValueError:
            continue
        raise AssertionError(f"the check let pass {label}")
    for call in (lambda: Channel("x", "symbol", 3).encode(1), lambda: Effector("v", [3]).gate_inputs(None, None),
                 lambda: Effector("v", [3]).cost(0, None), lambda: RewardSource("r").felt(None, None)):
        try:
            call()
        except NotImplementedError:
            continue
        raise AssertionError("a method not yet wired answered")
    print("anatomy 3: the check refuses", len(bad), "faulty declarations; the methods of later steps are not yet wired")


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
    until steps R4-R5"""
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
    other = Anatomy([Channel("eye", "vector", 4)], [Effector("arm", [5, 5], rest_id=12)], [RewardSource("face")]).check()
    for call in (lambda: anatomy_for(other, {}), lambda: _born(other, {}), lambda: Life(A.m, other)):
        try:
            call()
        except NotImplementedError:
            continue
        raise AssertionError("a life was built on an anatomy not of language before steps R4-R5")
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


ANATOMY_TESTS = [test_language_anatomy_equals_the_tokenizers_fields, test_language_anatomy_is_inert, test_anatomy_check,
                 test_life_reads_its_anatomy, test_an_anatomy_in_the_tokenizers_place, test_the_body_reads_text_through_its_anatomy]

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
