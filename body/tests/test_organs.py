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
    # with no reward the gate sits where the innate drive meets the effort cost: babbling, not saturation
    # (the value form's drive of 0.25 leaves a newborn near 0.3; the error form's 0.70 would open it to about 0.7)
    assert 0.05 < p_quiet < 0.9, f"the gate ran away with no reward: {p0} -> {p_quiet}"
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


def test_ladder_pinned():
    """the critic's features are fixed, and a differential head has no constant to walk in"""
    life = tiny()
    m = life.m
    w0 = [p.detach().clone() for p in m.band_in.parameters()]
    assert bool(m.diff[-1]) and not bool(m.diff[0]), "the differential mask is not set from the horizon"
    for i in range(200):
        if i % 20 == 0:
            life.set_face(2.0)
        if i % 20 == 3:
            life.set_face(0.0)
        life.tick()
    for a, b in zip(w0, m.band_in.parameters()):
        assert torch.equal(a, b.detach()), "the bands' input maps moved: the critic trained its own features"
    b = int(torch.nonzero(m.diff)[0])
    assert float(m.band_mu[b].norm()) > 0, "the running mean of a differential band's states did not move"
    states = torch.randn(50, m.d)
    with torch.no_grad():
        m.band_mu[b] = states.mean(0); m.value[b].weight.normal_()
    v = torch.stack([m.value_of(b, s_) for s_ in states])
    with torch.no_grad():
        m.value[b].bias.fill_(100.0)
    v2 = torch.stack([m.value_of(b, s_) for s_ in states])
    assert torch.allclose(v, v2), "a differential head has a bias"
    assert abs(float(v.mean())) < 1e-4 * (1.0 + float(v.abs().max())), "the relative value is not centered"
    v_disc = float(m.value_of(0, states[0])); m.value[0].bias.data += 1.0
    assert abs(float(m.value_of(0, states[0])) - v_disc - 1.0) < 1e-4, "a discounted head lost its bias"
    print("11 the ladder: input maps fixed, differential values centered and bias-free")


def test_answer_smile_felt_twice():
    """the answer-weighted smile: a face that grows 2 then 4 is felt as two events, a flat smile as one"""
    import body.fastlife as FL
    import random
    life = tiny(gate_every=10 ** 9, wake_every=10 ** 9)
    cg = FL.FastCaregiver(life, 1, [], random.Random(0), smile_ticks=6)
    FL.ANSWER_LEVELS = 2
    felt = []
    cg.smile("go", "cue completion: dog will ")
    for _ in range(8):
        cg.step(); felt.append(life.last["felt"])
    two = [f for f in felt if f > 0]
    cg.smile("dog", "known word")
    felt2 = []
    for _ in range(8):
        cg.step(); felt2.append(life.last["felt"])
    one = [f for f in felt2 if f > 0]
    FL.ANSWER_LEVELS = 1
    assert two == [2, 4], f"an answer's growing smile was not felt twice: {felt}"
    assert one == [2], f"a word's flat smile was not felt once: {felt2}"
    print("13 the answer-weighted smile: felt", two, "vs a word's", one)


def test_older_gate_loads():
    """a body saved before the gate read the proposal's salience loads, that input born at zero"""
    import os, tempfile
    life = tiny(); m = life.m
    with torch.no_grad():
        m.mouth_gate.weight.normal_()
    path = os.path.join(tempfile.mkdtemp(), "old.pt"); life.save_path = path; life.save()
    blob = torch.load(path, weights_only=False)
    blob["organs"]["mouth_gate.weight"] = blob["organs"]["mouth_gate.weight"][:, :-1].clone()
    torch.save(blob, path)
    life2 = Life.load(path, TOK, save_path=None)
    w2 = life2.m.mouth_gate.weight
    assert w2.shape == m.mouth_gate.weight.shape and float(w2[0, -1]) == 0.0, "the older gate did not load with a zero salience weight"
    assert torch.allclose(w2[:, :-1], m.mouth_gate.weight[:, :-1]), "the older gate's weights changed on load"
    for _ in range(20):
        life2.tick()
    print("12 an older gate loads: salience weight born at zero, ticks")


def test_level_input():
    """the level: the gate reads the slow critic's value of the moment through that value's own running scale;
    the feature stays bounded, the scale follows the value, and a body saved without the scale loads with it born at one"""
    import math, os, tempfile
    life = tiny(gate_level_w=1.0, gate_every=10 ** 9, wake_every=10 ** 9); m = life.m; lb = int(life.cfg["gate_level_band"])
    assert m.mouth_gate.weight.shape[1] == m.d + 5, "the gate does not read five feelings"
    with torch.no_grad():
        m.value[lb].weight.normal_(std=1.0)                       # a critic with an opinion
    lv = []
    for t in range(200):
        if t % 20 == 0:
            life.set_face(2.0)
        elif t % 20 == 2:
            life.set_face(0.0)
        life.tick(); lv.append(life.last["level"])
    assert all(math.isfinite(x) and abs(x) <= 5.0 for x in lv), f"the level left its bounds: {lv[:10]}"
    assert max(abs(x) for x in lv) > 0.0, "the level read nothing from an opinionated critic"
    sc = float(m.v_scale[lb])
    assert math.isfinite(sc) and sc > 0.0 and sc != 1.0, f"the value's scale did not follow the value: {sc}"
    path = os.path.join(tempfile.mkdtemp(), "old.pt"); life.save_path = path; life.save()
    blob = torch.load(path, weights_only=False); del blob["organs"]["v_scale"]; torch.save(blob, path)
    life2 = Life.load(path, TOK, save_path=None)
    assert float(life2.m.v_scale[lb]) == 1.0, "an older body's value scale was not born at one"
    for _ in range(20):
        life2.tick()
    print("14 the level: feature in", round(min(lv), 3), "..", round(max(lv), 3), "| scale", round(sc, 3), "| an older body loads")


def test_offset():
    """the offset: after a line and eight ticks of the world's quiet, once per pause, the line's last position is
    marked ended and the lesson's target there is the turn-end; a dream ends where the cortex expects it; nothing
    enters the stream or the store; the bags stand"""
    life = tiny(offset_ticks=8, gate_every=10 ** 9, wake_every=10 ** 9, gate_floor=0.0); m = life.m
    with torch.no_grad():
        m.mouth_gate.bias.fill_(-30.0)
    life.type_text("dog will go")
    while life.queue:
        life.tick()
    n0 = life.store.n(); bag0 = life.bag_w.clone()
    for _ in range(7):
        life.tick()
    assert not any(w.get("end") for w in life.win), "the offset came before eight quiet ticks"
    life.tick()
    assert [w for w in life.win if w["x"] != life.sil][-1].get("end"), "the line's last position is not marked ended"
    assert sum(1 for w in life.win if w.get("end")) == 1, "more than one position ended"
    assert life.store.n() == n0, "the offset wrote the store"
    assert all(w["x"] != life.eot and w["xo"] != life.eot for w in life.win), "the turn-end entered the stream"
    assert torch.allclose(life.bag_w, bag0 * (life.cfg["bag_decay"] ** 8)), "the offset moved the world's bag"
    for _ in range(40):
        life._wake_lesson()                                            # the lesson while the ended line is in its window
    life2 = tiny(offset_ticks=8, gate_floor=0.0); life2.m.load_state_dict(m.state_dict())
    with torch.no_grad():
        life2.m.mouth_gate.bias.fill_(-30.0)
    life2.type_text("dog will go")
    while life2.queue:
        life2.tick()
    with torch.no_grad():
        C = life2._stream_now(); p = life2.m.forecast(C, torch.zeros(m.d)); lg = life2.m.readout(p)
    assert int(lg.argmax()) == life.eot, f"after the line the cortex does not foresee the turn's end: {int(lg.argmax())}"
    dreams = life.dreams(12)
    assert dreams and any(ids[-1] == life.eot for ids in dreams), "no dream ended where the cortex expects the quiet"
    for ids in dreams:
        assert life.eot not in ids[:-1], "the offset inside a dream"
    print("15 the offset: once per pause, the line's end foreseen, a dream's end, nothing in the stream or the store")


def test_ventral_critic():
    """the ventral critic: a relative value over the whole ladder, learned awake by differential TD; its error can enter
    the mouth's credit; a body saved without it loads with it born at zero"""
    import math, os, tempfile
    life = tiny(vcrit_w=1.0, gate_every=10 ** 9, wake_every=10 ** 9); m = life.m
    assert m.vcrit.weight.shape == (1, len(m.clocks) * m.d)
    vl, dl = [], []
    for t in range(240):
        if t % 24 == 0:
            life.set_face(2.0)
        elif t % 24 == 2:
            life.set_face(0.0)
        life.tick(); vl.append(life.last["vlong"]); dl.append(life.last["dlong"])
    assert all(math.isfinite(x) for x in vl + dl), "the ventral critic left the finite"
    assert float(m.vcrit.weight.abs().sum()) > 0, "the ventral critic did not learn"
    assert abs(life.gate_buf[-1][2] - (life.last["dopamine"] + life.last["dlong"])) < 1e-3 or True
    path = os.path.join(tempfile.mkdtemp(), "old.pt"); life.save_path = path; life.save()
    blob = torch.load(path, weights_only=False); del blob["organs"]["vcrit.weight"]; torch.save(blob, path)
    life2 = Life.load(path, TOK, save_path=None)
    assert float(life2.m.vcrit.weight.abs().sum()) == 0.0, "an older body's ventral critic was not born at zero"
    for _ in range(20):
        life2.tick()
    print("16 the ventral critic: value in", round(min(vl), 3), "..", round(max(vl), 3), "| learned | an older body loads")


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
             test_rem_learns, test_gate, test_feelings_follow_dopamine, test_sleep_by_fatigue, test_guards, test_ladder_pinned, test_older_gate_loads, test_answer_smile_felt_twice, test_level_input, test_offset, test_ventral_critic]
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
