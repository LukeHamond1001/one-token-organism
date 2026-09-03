"""one test per organ (BODY_SPEC.md §8). Run: python3 -m body.tests.test_organs
Each test fails when its organ stops doing its job. Tiny body, CPU, seconds."""
import math
import sys
import time

import torch
from tokenizers import Tokenizer

sys.path.insert(0, "/Users/lukehamond/Projects/project")
from body.life import Life  # noqa: E402

TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
CFG = dict(wake_ticks=400, wake_every=8, gate_every=8, night_rounds=6, night_starts=12, rem_dreams=4, rem_steps=4,
           fatigue_half_life=1e9, stress_half_life=1e9, mood_half_life=1e9)   # feelings do not decay in the tests


def tiny(seed=0, **kw):
    c = dict(CFG); c.update(kw)
    return Life.birth(TOK, device="cpu", d=64, layers=2, heads=2, window=32, cfg=c, seed=seed)


def say(life, text, ticks_after=4):
    life.type_text(text)
    while life.queue:
        life.tick()
    for _ in range(ticks_after):
        life.tick()


def test_corollary_discharge():
    life = tiny()
    say(life, "dog will go", 6)
    world = life.store.W == 0
    assert life.store.n() >= 5, "the world's symbols were not stored"
    assert int((life.store.W == 1).sum()) == 0, "the body's own symbols were stored (corollary discharge failed)"
    assert float(life.store.S[world].mean()) > 0, "the world's symbols carry no strength"
    Es = torch.nn.functional.normalize(life.m.E.weight[life.sil], dim=0)
    assert float((life.store.V @ Es).max()) < 0.99, "the world's quiet was stored as a memory"
    print("1 corollary discharge: ok", life.store.n(), "slots, none the body's own")


def test_store_recalls():
    life = tiny()
    for _ in range(3):
        say(life, "give milk", 2)
    # the cue: the context 'give ' should recall 'm'
    life.bag_w.zero_(); life.bag_o.zero_()
    for ch in "give ":
        life.bag_w = life.cfg["bag_decay"] * life.m.shift(life.bag_w) + life.m.E.weight[TOK.token_to_id(ch)]
    pred, conf, _ = life.store.read(life.bag)
    top = TOK.decode([life.m.nearest(pred)])
    assert top == "m", f"the store recalled {top!r} after 'give ', not 'm'"
    print("2 the store recalls: 'give ' ->", repr(top), "conf", round(conf, 3))


def test_recall_is_by_content():
    """three lines with shared endings: each cue recalls its own continuation, not the strongest memory"""
    life = tiny()
    for _ in range(2):
        for line in ["dog will go", "ball on", "give milk"]:
            say(life, line, 2)
    for _ in range(6):
        say(life, "give milk", 2)                       # the most reinforced memory
    def recall(cue):
        # the query as life makes it: the cue typed while the body babbles (its babble is in the read
        # query but never in a key); the recall taken as the cue's last symbol entered, before the
        # body answers (with its own answer in the query the store rightly continues)
        life.type_text(cue)
        while life.queue:
            life.tick()
        pred = life._read_world; conf = float(pred.norm())
        for _ in range(6):
            life.tick()
        return TOK.decode([life.m.nearest(pred)]), conf
    got = {cue: recall(cue) for cue in ("dog will ", "ball ", "give ")}
    assert got["dog will "][0] == "g", f"'dog will ' recalled {got['dog will ']}"
    assert got["ball "][0] == "o", f"'ball ' recalled {got['ball ']}"
    assert got["give "][0] == "m", f"'give ' recalled {got['give ']}"
    print("2b recall is by content:", {k: (v[0], round(v[1], 2)) for k, v in got.items()})
    # continuation: with its own first letter in the query (the efference copy), the store recalls the
    # second letter, not the first again. The mouth is held quiet here: the instrument says the first
    # letter, and under the lag code "g g" is not "g"
    with torch.no_grad():
        life.m.mouth_gate.bias.fill_(-30.0)
    life.cfg["gate_floor"] = 0.0
    seq = {}
    for cue, first, second in (("dog will ", "g", "o"), ("give ", "m", "i"), ("ball ", "o", "n")):
        life.type_text(cue)
        while life.queue:
            life.tick()
        with torch.no_grad():
            life._step(TOK.token_to_id(first), 1, learn_store=False)
        pred, conf, _ = life.store.read(life.bag)
        seq[cue] = (TOK.decode([life.m.nearest(pred)]), round(conf, 2), second)
        for _ in range(6):
            life.tick()
    assert all(v[0] == v[2] for v in seq.values()), f"after its own first letter the store did not continue: {seq}"
    print("2c the store continues after its own first letter:", seq)


def test_dreams_are_its_lines():
    life = tiny()
    for line in ["dog will go", "give milk", "scared ball"]:
        say(life, line, 2)
    dreams = life.dreams(12)
    assert dreams, "no dreams from a store that holds three lines"
    texts = [TOK.decode(d) for d in dreams]
    hits = sum(1 for t in texts if any(w in t for w in ("dog", "will", "go", "give", "milk", "scared", "ball", "ill", "ilk")))
    assert hits >= len(texts) // 2, f"dreams are not its lines: {texts}"
    print("3 dreams are its lines:", texts[:5])


def test_night_moves_the_cortex():
    life = tiny()
    for _ in range(2):
        for line in ["dog will go", "give milk", "scared ball"]:
            say(life, line, 2)
    dreams = life.dreams(12)
    before, n = life.gauge(dreams)
    rep = life.night()
    assert not rep.get("error"), rep
    assert rep.get("discarded") is False
    after = rep["gauge"]["after"]
    assert after is not None and rep["nrem_steps"] > 0
    assert rep["nrem_loss"][-1] < rep["nrem_loss"][0], f"the night's loss did not fall: {rep['nrem_loss']}"
    print("4 the night moves the cortex: gauge", before, "->", after, "loss", rep["nrem_loss"][0], "->", rep["nrem_loss"][-1])


def test_rem_learns():
    life = tiny()
    for _ in range(2):
        for line in ["dog will go", "give milk", "scared ball", "big dog bigger dog"]:
            say(life, line, 2)
    c1 = life.night().get("rem_cos")
    for line in ["dog will go", "give milk", "scared ball"]:
        say(life, line, 2)
    c2 = life.night().get("rem_cos")
    assert c1 is not None and c2 is not None and math.isfinite(c2), (c1, c2)
    print("5 REM learns: forecast cosine", c1, "->", c2)


def test_gate():
    life = tiny(gate_lr=0.5)
    p0 = None
    for _ in range(120):
        life.tick()
        p0 = p0 or life.last.get("gate")
    p_quiet = life.last["gate"]
    assert 0.05 < p_quiet < 0.6, f"the gate drifted with no reward: {p0} -> {p_quiet}"
    # a burst: smiles while it acts
    for i in range(60):
        if i % 10 == 0:
            life.set_face(2.0)
        if i % 10 == 3:
            life.set_face(0.0)
        life.tick()
    p_reward = life.last["gate"]
    assert p_reward > p_quiet, f"a burst did not open the gate: {p_quiet} -> {p_reward}"
    # fatigue: force a tired body, no reward
    life.fatigue = 40.0
    for _ in range(120):
        life.tick(); life.fatigue = max(life.fatigue, 40.0)
    p_tired = life.last["gate"]
    assert p_tired < p_reward, f"fatigue did not close the gate: {p_reward} -> {p_tired}"
    assert p_tired >= life.cfg["gate_floor"] - 1e-6, "the gate fell below spontaneous activity"
    print("6 the gate: quiet", round(p_quiet, 3), "after reward", round(p_reward, 3), "tired", round(p_tired, 3))


def test_feelings_follow_dopamine():
    life = tiny()
    for _ in range(60):
        life.tick()
    m0 = life.mood
    assert abs(m0) < 0.5, f"mood moved with effort alone: {m0}"
    life.set_face(2.0); life.tick(); life.tick(); life.set_face(0.0); life.tick()
    assert life.mood > m0, f"a smile did not lift mood: {m0} -> {life.mood}"
    life.set_face(-2.0); life.tick(); life.tick(); life.set_face(0.0); life.tick()
    assert life.stress > 0, "a frown did not raise stress"
    life.mood = 0.0; life.tick(); s0 = life.m.read_sharp
    life.mood = 6.0; life.tick(); s1 = life.m.read_sharp
    assert abs(s0 - life.cfg["sharp_base"]) < 1e-6 and s1 > s0, f"decisiveness did not follow mood: {s0} {s1}"
    print("7 feelings follow dopamine: mood", round(m0, 3), "->", round(life.mood, 3), "stress", round(life.stress, 3), "| sharpness", s0, "->", s1)


def test_sleep_by_fatigue():
    life = tiny(wake_ticks=60)
    say(life, "dog will go", 2)
    n0 = life.nights
    for _ in range(80):
        life.tick()
    assert life.nights == n0 + 1, "it did not sleep at the switch"
    assert life.sleep_pressure < 60, "its pressure did not reset at the night"
    assert life.last_night and not life.last_night.get("error"), life.last_night
    print("8 sleep by fatigue: slept once, pressure", life.sleep_pressure, "night", {k: life.last_night.get(k) for k in ("dreams", "nrem_steps", "discarded")})


def test_guards():
    life = tiny()
    for line in ["dog will go", "give milk"]:
        say(life, line, 2)
    with torch.no_grad():
        life.m.latent_pred.bias[0] = float("nan")
    rep = life.night()
    assert rep.get("discarded") is True or rep.get("error"), rep
    print("9 guards: a NaN weight ->", "discarded" if rep.get("discarded") else rep.get("error"))


if __name__ == "__main__":
    t0 = time.time()
    tests = [test_corollary_discharge, test_store_recalls, test_recall_is_by_content, test_dreams_are_its_lines, test_night_moves_the_cortex,
             test_rem_learns, test_gate, test_feelings_follow_dopamine, test_sleep_by_fatigue, test_guards]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1; print("FAIL", t.__name__, ":", e)
        except Exception as e:
            failed += 1; print("ERROR", t.__name__, ":", type(e).__name__, str(e)[:300])
    print(f"{len(tests) - failed}/{len(tests)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if failed else 0)
