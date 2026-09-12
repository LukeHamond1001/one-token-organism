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
    marked ended and the lesson's target there is the turn-end; the store's slot for that symbol carries the boundary
    mark and the first memory kept after a pause the start mark; a dream runs from a start to an end; nothing enters
    the stream; the bags stand"""
    life = tiny(offset_ticks=8, offset_form="count", end_symbol="eot", gate_every=10 ** 9, wake_every=10 ** 9, gate_floor=0.0); m = life.m   # the old forms, deliberately
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
    assert abs(float(life.bag_w.norm() / bag0.norm()) - life.cfg["bag_decay"] ** 8) < 1e-4 and float(torch.cosine_similarity(life.bag_w, bag0, dim=0)) > 0.9999, "the offset moved the world's bag"
    assert int(life.store.B.sum()) == 1 and bool(life.store.B[-1]), "the last symbol's memory does not carry the boundary"
    d0 = life.dreams(12)
    assert d0 and any(ids[-1] == life.eot for ids in d0), "no untaught dream ended at the memory's boundary"
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
    # starts: a second utterance after the pause, and one after a long pause (its first symbol's context faded to nothing)
    for _ in range(4):
        life.tick()
    life.type_text("give milk")
    while life.queue:
        life.tick()
    for _ in range(8):
        life.tick()
    assert int(life.store.Bs.sum()) >= 1, "the first symbol after a pause was not marked a start"
    for _ in range(300):
        life.tick()
    life.type_text("where ball? ball under")
    while life.queue:
        life.tick()
    for _ in range(8):
        life.tick()
    j = int(torch.nonzero(life.store.Bs).flatten()[-1]); v = TOK.decode([life.m.nearest(life.store.V[j])])
    k = TOK.decode([life.m.nearest(life.store.K[j])])
    assert (k, v) == ("w", "h"), f"after a long pause the start mark fell on {k!r}->{v!r}, not the second symbol under the first"
    d2 = life.dreams(12)
    assert any(len(ids) >= 8 and ids[-1] == life.eot for ids in d2), f"no whole line dreamed from a start to an end: {[len(i) for i in d2]}"
    w = TOK.token_to_id("w")
    assert any(ids[0] == w and len(ids) >= 6 for ids in d2), "no dream began with the line's first symbol, read from the onset's key"
    for ids in d2:
        assert life.eot not in ids[:-1], "the offset inside a dream"
    print("15 the offset: once per pause, the line's end foreseen, the memory marked at both ends, a dream a whole line, nothing in the stream")


def test_ventral_critic():
    """the ventral critic: a relative value over the whole ladder, learned awake by differential TD; its error can enter
    the mouth's credit; a body saved without it loads with it born at zero"""
    import math, os, tempfile
    life = tiny(vcrit_w=1.0, gate_every=10 ** 9, wake_every=10 ** 9); m = life.m
    assert m.vcrit.weight.shape == (1, len(m.clocks) * m.d + len(m.clocks) + 1)  # the bands' states, the eight tonic traces, the clock
    vl, dl = [], []
    for t in range(240):
        if t % 24 == 0:
            life.set_face(2.0)
        elif t % 24 == 2:
            life.set_face(0.0)
        life.tick(); vl.append(life.last["vlong"]); dl.append(life.last["dlong"])
    assert all(math.isfinite(x) for x in vl + dl), "the ventral critic left the finite"
    assert float(m.vcrit.weight.abs().sum()) > 0, "the ventral critic did not learn"
    assert any(abs(x) > 1e-6 for x in dl), "the ventral critic never erred"
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



def _watched(seed=0, **kw):
    """the watched recipe at toy size: the striatal input with a working-memory slot, the decorrelated fast critic, the planning actor,
    the ventral critic on the clock and the tonic traces alone"""
    c = dict(fast_rls=1, fast_input="striatum", stri_k=4, stri_m=64, stri_quiet=1, wm=1, wm_max=64, actor=1, actor_form="plan", plan_k=3,
             plan_h=2, plan_beta=4.0, vcrit_w=0.3, vcrit_traces=1, vcrit_clock=1, vcrit_bands="-", own_target_decay=0.7,
             gate_every=10 ** 9, wake_every=10 ** 9)
    c.update(kw)
    return tiny(seed=seed, **c)


def _live(life, ticks):
    """a stretch of life: the world says a word now and then, smiles now and then"""
    for t in range(ticks):
        if t % 25 == 0:
            life.set_face(2.0)
        elif t % 25 == 2:
            life.set_face(0.0)
        if t % 40 == 0:
            life.type_text("go ")
        life.tick()


def test_striatum():
    """the striatal input: a delay line of the last k events (heard, own, face, quiet) through a born expansion the heads read;
    the expansion never learns"""
    life = _watched(); m = life.m
    V = m.vocab; k = int(life.cfg["stri_k"]); M = int(life.cfg["stri_m"])
    assert m.stri_W.shape == (k * (2 * V + 3), M) and m.stri_line.numel() == k
    m.striatum_reset(); z0 = m.striatum_read()
    assert z0.shape == (M,) and float(z0.min()) >= 0
    m.striatum_push(0, 5); z1 = m.striatum_read(); assert not torch.equal(z0, z1), "a heard symbol left no trace"
    m.striatum_push(1, 5); z2 = m.striatum_read(); assert not torch.equal(z1, z2), "own symbol indistinguishable from heard"
    assert int(m.stri_line[0]) == V + 5 and int(m.stri_line[1]) == 5, "the line does not keep the order of events"
    m.striatum_push(2, 1); assert int(m.stri_line[0]) == 2 * V + 1, "the frown is not an event"
    m.striatum_push(3, 0); assert int(m.stri_line[0]) == 2 * V + 2, "a quiet tick is not an event"
    for _ in range(k + 2):
        m.striatum_push(0, 1)
    assert int(m.stri_line[-1]) == 1, "the line does not forget the oldest event"
    assert m.stri_in().shape == (2 * M,), "the heads do not read the slot beside the line"
    W0, b0 = m.stri_W.clone(), m.stri_b.clone(); _live(life, 80)
    assert torch.equal(W0, m.stri_W) and torch.equal(b0, m.stri_b), "the born expansion moved (it must not learn)"
    print("17 the striatum: line of", k, "events x", 2 * V + 3, "kinds ->", M, "units | ordered | forgets | born, unlearned")


def test_working_memory():
    """working memory: a slot latched at the world's utterance end, cleared by the felt reward or by age; read beside the line"""
    life = _watched(wm_max=30, wm_burst=100.0); m = life.m
    assert float(m.wm_on) == 0
    say(life, "hi ", ticks_after=16)                       # the world speaks and falls quiet: the offset latches
    assert float(m.wm_on) == 1, "the offset did not latch the slot"
    assert float(m.stri_in()[int(life.cfg["stri_m"]):].abs().sum()) > 0, "the latched slot is empty"
    life.set_face(2.0)
    for _ in range(3):
        life.tick()
    assert float(m.wm_on) == 0, "the felt reward did not clear the slot"
    life.set_face(0.0); m.wm_latch(m.striatum_read()); assert float(m.wm_on) == 1
    for _ in range(34):
        life.tick()
    assert float(m.wm_on) == 0, "age did not clear the slot"
    print("18 working memory: latched at the offset | cleared by the reward | cleared by age", int(life.cfg["wm_max"]))


def test_planning_actor():
    """the planning actor: at a word boundary when about to speak, the cortex's few candidates are imagined through the world
    model and valued by the striatal critic; the imagination leaves the body as it was"""
    life = _watched(actor_margin=1e9); m = life.m          # every speakable symbol within the margin: the cap alone shortens the list
    line = m.stri_line.clone(); n = len(life.win); v = life._imagine_value(TOK.token_to_id("a"), 2)
    assert math.isfinite(float(v)) and torch.equal(line, m.stri_line) and len(life.win) == n, "imagining moved the body"
    pl = None
    for t in range(600):
        if t % 25 == 0:
            life.set_face(2.0)
        elif t % 25 == 2:
            life.set_face(0.0)
        if t % 40 == 0:
            life.type_text("go ")
        life.tick(); pl = getattr(life, "_plan_last", None)
        if pl:
            break
    assert pl, "the planner never ran (the body never spoke at a boundary)"
    assert len(pl["cands"]) == int(life.cfg["plan_k"]), "the shortlist is not the cortex's few"
    assert all(math.isfinite(float(x)) for x in pl["vals"].values()), "an imagined value left the finite"
    print("19 the planning actor: shortlist of", len(pl["cands"]), "| imagined values", {c: round(float(v), 3) for c, v in pl["vals"].items()})


def test_new_organs_round_trip():
    """save/load keeps the striatal expansion, its line, the fast evidence and head, the actor, the slot, and the parent's environment"""
    import os, tempfile
    life = _watched(); m = life.m; _live(life, 200); m.wm_latch(m.striatum_read())
    assert float(m.vf_A.abs().sum()) > 0, "no fast evidence gathered"
    path = os.path.join(tempfile.mkdtemp(), "w.pt"); life.save_path = path; life.save()
    life2 = Life.load(path, TOK, save_path=None); m2 = life2.m
    for k_ in ("stri_W", "stri_b", "stri_line", "vf_A", "vf_b", "vf_mu", "vf_var", "vf_n", "wm_slot", "wm_on", "wm_age"):
        assert torch.equal(getattr(m, k_), getattr(m2, k_)), f"{k_} did not survive the save"
    p1, p2 = dict(m.named_parameters()), dict(m2.named_parameters())
    for k_ in ("vfast.weight", "vfast.bias", "actor.weight", "actor.bias"):
        assert torch.equal(p1[k_], p2[k_]), f"{k_} did not survive the save"
    assert isinstance(torch.load(path, weights_only=False).get("env"), dict), "the save does not record the parent's environment"
    life2.tick()
    print("20 the new organs survive the save: striatum, fast evidence", tuple(m.vf_A.shape), "heads, slot, env")


def test_chain_closes():
    """the eleventh defect: the value's source and target are the same instant one tick apart, so the temporal-difference
    evidence is symmetric (the fast head's up to its trace); before the fix 0.16 and 0.30"""
    life = _watched(); m = life.m; _live(life, 400)

    def asym(A):
        return float((A - A.T).norm() / max(float(A.norm()), 1e-12))
    af, av = asym(m.vf_A[:-1, :-1]), asym(m.vc_A[:-1, :-1])
    assert av < 0.02, f"the ventral chain does not close (asymmetry {av:.3f})"
    assert af < 0.08, f"the fast chain does not close (asymmetry {af:.3f})"
    assert int((m.vf_A.diagonal() < 0).sum()) == 0 and int((m.vc_A.diagonal() < 0).sum()) == 0, "negative evidence on the diagonal"
    print(f"21 the chain closes: evidence asymmetry fast {af:.4f} ventral {av:.4f} | no negative diagonal")



def test_night_warmup():
    """the plasticity ramps: with night_warm W the night's first W steps run at 1/W, 2/W, ... of the rate, then the whole rate
    (a fresh optimizer's first full-rate step shoves a wide cortex); with night_warm 0 every step runs at the rate"""
    for warm in (4, 0):
        life = tiny(night_warm=warm, night_rounds=6, night_starts=2, rem_dreams=2, rem_rounds=1, rem_steps=1); seen = []
        orig = life._night_step
        def spy(opt, _seen=seen, _orig=orig):
            _seen.append(round(opt.param_groups[0]["lr"] / float(life.cfg["night_lr"]), 3)); return _orig(opt)
        life._night_step = spy
        say(life, "the dog sat on the hill ", 8); say(life, "the cat ran up the tree ", 8)
        rep = life.night() or getattr(life, "last_night", {}) or {}
        assert rep.get("dreams"), "the night dreamed nothing"
        nrem_lrs = seen[:6]
        if warm:
            assert nrem_lrs[:4] == [0.25, 0.5, 0.75, 1.0] and all(x == 1.0 for x in nrem_lrs[4:]), f"the ramp is wrong: {nrem_lrs}"
            ramp = nrem_lrs
        else:
            assert all(x == 1.0 for x in nrem_lrs), f"without the ramp the rate must be whole: {nrem_lrs}"
    print("22 the night's plasticity ramps: rates", ramp, "with night_warm 4; whole without")


def test_plan_boundary():
    """the planner's boundary: with plan_boundary 1 it plans only after a pause or a space; with plan_boundary 0 it plans
    whenever it acts and the cortex is torn, whatever the last symbol was (no fact about text needed)"""
    counts = {}
    for pb in (1, 0):
        life = _watched(actor_margin=1e9, plan_boundary=pb); n = 0; mid = 0
        prev_acted, prev_own = False, None
        for t in range(400):
            if t % 25 == 0:
                life.set_face(2.0)
            elif t % 25 == 2:
                life.set_face(0.0)
            if t % 40 == 0:
                life.type_text("go ")
            life._plan_last = None; life.tick()
            if getattr(life, "_plan_last", None):
                n += 1
                if prev_acted and prev_own not in (None, life.space_id):
                    mid += 1                                                   # a plan inside its own word
            prev_acted = bool(getattr(life, "_acted_last", False)); prev_own = getattr(life, "_own_last", None)
        counts[pb] = (n, mid)
    assert counts[1][1] == 0, f"with the space rule a plan fired inside a word: {counts[1]}"
    assert counts[0][0] >= counts[1][0] and counts[0][1] > 0, f"without the space rule the planner did not reach inside words: {counts}"
    print("23 the planner's boundary: plans", counts[1][0], "with the space rule (none mid-word);", counts[0][0], "without it,", counts[0][1], "mid-word")


def test_exploration_drive():
    """the exploration drive: with explore_gain the gate's floor rises after the world surprises it and settles as the
    world repeats; with the gain at zero the floor is the constant, whatever the world does"""
    floors = {}
    for gain in (0.0, 1.0):
        life = tiny(explore_gain=gain, explore_tau=8, gate_every=10 ** 9, wake_every=10 ** 9); fl = []
        for _ in range(3):
            say(life, "zq xj vk pw ", 2); fl.append(getattr(life, "_floor_now", None))
        floors[gain] = fl
    assert all(f == float(life.cfg["gate_floor"]) for f in floors[0.0]), f"without the drive the floor moved: {floors[0.0]}"
    assert floors[1.0][0] > float(life.cfg["gate_floor"]), f"the drive did not raise the floor after surprise: {floors[1.0]}"
    assert max(floors[1.0]) <= 0.5, "the floor passed its cap"
    print("24 the exploration drive: floor", round(float(life.cfg["gate_floor"]), 3), "->", [round(f, 3) for f in floors[1.0]], "after a strange line, three times")


def test_offset_by_settling():
    """the event's end by the law: with offset_form settle the offset fires when the surprise, high while the world speaks,
    settles after it stops, long before the count would; the count stays as the floor for a body whose surprise is flat"""
    life = tiny(offset_form="settle", offset_ticks=400, offset_settle=0.5, offset_fast=3, offset_slow=48, gate_every=10 ** 9, wake_every=10 ** 9)
    fired = []; orig = life._offset
    def spy():
        fired.append(life.ticks); return orig()
    life._offset = spy
    orig_step = life._step
    def step(x, who, **kw):                                   # a synthetic surprise: high while the world's symbols arrive, low at rest
        out = orig_step(x, who, **kw)
        if who == 0:
            life._surp_tick = 1.0 if x != life.sil else 0.05
        return out
    life._step = step
    for rep in range(4):
        t0 = life.ticks; life.type_text("the dog sat on the hill ")
        while life.queue:
            life.tick()
        t_end = life.ticks
        for _ in range(40):
            life.tick()
        assert any(t_end < f <= t_end + 20 for f in fired), f"the offset did not fire within 20 quiet ticks after the world stopped: {fired}"
        assert not any(t0 < f <= t_end for f in fired), "the offset fired inside an utterance"
    assert len(fired) == 4, f"once per pause: {fired}"
    flat = tiny(offset_form="settle", offset_ticks=8, gate_every=10 ** 9, wake_every=10 ** 9); fired2 = []; o2 = flat._offset
    flat._offset = lambda: (fired2.append(flat.ticks), o2())[1]
    flat.type_text("go "); 
    while flat.queue:
        flat.tick()
    for _ in range(30):
        flat.tick()
    assert len(fired2) == 1, f"the count floor did not end the event for a flat newborn: {fired2}"
    print("25 the event's end by the law: fired", [f - 0 for f in fired][:4], "ticks in, within 20 quiet ticks, no count; the newborn's floor at 8")


def test_end_as_rest():
    """the world's stop as rest (end_symbol rest): the offset marks the line's end, the lesson's target there is the rest
    itself, a dream ends with the rest, and the chat token appears nowhere; the old form still targets the token"""
    outs = {}
    for form in ("eot", "rest"):
        life = tiny(end_symbol=form, offset_ticks=8, gate_every=10 ** 9, wake_every=10 ** 9, gate_floor=0.0)
        with torch.no_grad():
            life.m.mouth_gate.bias.fill_(-30.0)                              # the mouth held silent: the world's line alone
        life.type_text("dog will go")
        while life.queue:
            life.tick()
        for _ in range(8):
            life.tick()                                                      # the offset after eight quiet ticks
        ended = [i for i, w in enumerate(life.win) if w.get("end")]
        assert len(ended) == 1, f"one position should be ended: {ended}"
        xs = torch.tensor([w["x"] for w in life.win]); y = xs.clone(); y[ended[-1]] = life.end_id
        d = life.dreams(12)
        outs[form] = (int(y[ended[-1]]), [ids[-1] for ids in d if ids])
        assert all(w["x"] != life.eot and w["xo"] != life.eot for w in life.win), "the turn-end entered the stream"
    assert outs["eot"][0] == life.eot and outs["rest"][0] == life.sil, f"the end's target: {outs}"
    assert any(e == life.eot for e in outs["eot"][1]), f"under the token form no dream ended at the boundary: {outs['eot'][1]}"
    assert any(e == life.sil for e in outs["rest"][1]), f"under the rest form no dream ended with the rest: {outs['rest'][1]}"
    assert not any(e == life.eot for e in outs["rest"][1]), "the chat token appeared in a rest-form dream"
    print("26 the world's stop as rest: the end's target", outs["rest"][0], "(the rest) | rest-form dreams end with", sorted(set(outs["rest"][1]))[:4], "| token form with", sorted(set(outs["eot"][1]))[:4])


def test_explore_in_the_choice():
    """the drive in the choice: with explore_choice the planner's temperature over its candidates rises after the world
    surprises it and is one when the drive is off; the gate's floor is untouched by it"""
    temps = {}
    for ec in (0.0, 2.0):
        life = _watched(actor_margin=1e9, explore_choice=ec, explore_tau=8); seen = []
        for t in range(300):
            if t % 25 == 0:
                life.set_face(2.0)
            elif t % 25 == 2:
                life.set_face(0.0)
            if t % 40 == 0:
                life.type_text("zq xj vk ")                                       # a strange line: surprise
            life._plan_last = None; life.tick()
            pl = getattr(life, "_plan_last", None)
            if pl and "temp" in pl:
                seen.append(pl["temp"])
        temps[ec] = seen
        assert getattr(life, "_floor_now", life.cfg["gate_floor"]) == float(life.cfg["gate_floor"]), "the choice drive moved the gate's floor"
    assert temps[0.0] and all(t == 1.0 for t in temps[0.0]), f"with the drive off the temperature must be one: {temps[0.0][:5]}"
    assert temps[2.0] and max(temps[2.0]) > 1.05, f"with the drive on the temperature never rose: {temps[2.0][:8]}"
    print("27 the drive in the choice: temperature", round(max(temps[2.0]), 2), "at most after strange lines, 1.0 with the drive off; the floor untouched")


def test_prefrontal_ceiling():
    """the prefrontal voice's ceiling: fixed = vcrit_w times its reliability; earned = the reliability itself, so a proven
    long value weighs as much as the fast critic and a wrong one nothing; the weight is read where the credit is built"""
    weights = {}
    for form in ("fixed", "earned"):
        life = tiny(vcrit_w=0.3, vcrit_auto=1, vcrit_ceiling=form, gate_every=10 ** 9, wake_every=10 ** 9)
        life._vrel_update = lambda *a, **k: None                              # the reliability held still for the reading
        life._vrel_gain = 0.8                                                 # a long value proved right eight tenths of the way
        life.set_face(2.0); life.tick(); life.set_face(0.0)
        for _ in range(6):
            life.tick()
        weights[form] = float(getattr(life, "_vw_now", float("nan")))
    assert abs(weights["fixed"] - 0.24) < 1e-6, f"fixed: {weights}"
    assert abs(weights["earned"] - 0.8) < 1e-6, f"earned: {weights}"
    print("28 the prefrontal voice's ceiling: fixed", weights["fixed"], "| earned", weights["earned"], "at reliability 0.8")

def test_calibrated_sharpness():
    """the readout's sharpness as a law: fed a world whose symbols come from a flatter reading of its own forecast, the base
    falls; from a sharper one, it rises; no number is set by hand"""
    ends = {}
    for star in (6.0, 60.0):
        life = tiny(sharp_form="calibrated", sharp_rate=0.5, gate_every=10 ** 9, wake_every=10 ** 9)
        torch.manual_seed(0)
        fc = torch.randn(life.m.d) * 0.3                                    # a forecast held still
        life._fc_prev = fc
        for _ in range(1000):
            life.m.read_sharp = life.sharp_cal                               # the reading the world is scored against (no mood)
            lg = life.m.readout(fc).clone(); lg[life.bans] = float("-inf"); lg[life.sil] = float("-inf")
            p = torch.softmax(lg / life.m.read_sharp * star, dim=0)         # the world drawn from the star reading
            life._sharp_calibrate(int(torch.multinomial(p, 1)))
        ends[star] = life.sharp_cal
    assert ends[6.0] < 25.0 - 3 and ends[60.0] > 25.0 + 3, ends
    print("29 the calibrated sharpness: from 25, under a world read at 6 ->", round(ends[6.0], 1), "| at 60 ->", round(ends[60.0], 1))


def test_evidence_survives_the_load():
    """the thirteenth defect: the prefrontal voice's evidence (the reliability's moments and buffers) and the calibrated
    sharpness are the body's, saved and loaded with it"""
    import os, tempfile
    life = tiny(vcrit_auto=1)
    life._vrel = [10.0, 1.0, 2.0, 3.0, 4.0, 5.0]; life._vrel_gain = 0.6; life._vrel_corr = 0.7; life.sharp_cal = 17.5
    life._vbuf_v.extend([0.1, 0.2, 0.3]); life._vbuf_r.extend([1.0, 0.0, 2.0])
    path = os.path.join(tempfile.mkdtemp(), "ev.pt"); life.save_path = path; life.save()
    life2 = Life.load(path, TOK, device="cpu", save_path=path)
    assert life2._vrel == life._vrel and abs(life2._vrel_gain - 0.6) < 1e-9 and abs(life2._vrel_corr - 0.7) < 1e-9, (life2._vrel, life2._vrel_gain)
    assert list(life2._vbuf_v) == [0.1, 0.2, 0.3] and list(life2._vbuf_r) == [1.0, 0.0, 2.0] and abs(life2.sharp_cal - 17.5) < 1e-9
    print("30 the evidence survives the load: the reliability's moments and buffers, the slope", life2._vrel_gain, "| the calibrated sharpness", life2.sharp_cal)


def test_face_foresees():
    """the face organ foresees (§5c): a smile that always follows one line is foreseen before it is felt, and the organ's slope,
    the felt reward on its foresight, comes out positive"""
    life = tiny(face_form="foresee", face_tau=2000, face_every=16, gate_every=10 ** 9, wake_every=10 ** 9)
    fores = []
    for rep in range(40):
        say(life, "dog will go", 1)
        fores.append(float(life._fpred_now))                                # what it foresees for the next tick
        life.set_face(2.0); life.tick(); life.set_face(0.0)                   # the smile, felt on this tick
        for _ in range(6):
            life.tick()
    first, late = fores[0], sum(fores[-5:]) / 5                            # before any smile, and after forty
    assert abs(first) < 0.3 and late > 0.5, (first, late)
    assert life._frel_gain > 0.3, life._frel_gain
    print("31 the face organ foresees: before the smile", round(first, 2), "->", round(late, 2), "| its slope", round(life._frel_gain, 2))


def test_rem_imagines():
    """REM as imagination (§5c): imagined transitions enter the fast critic's evidence weighted by the face organ's slope; at slope
    zero nothing enters; the lived delay line is restored after"""
    life = _watched(); m = life.m
    assert m.vf_A.numel() > 0, "the fast critic's evidence is not kept in this configuration"
    life.cfg["rem_form"] = "imagine"; life.cfg["face_form"] = "foresee"; life.cfg["rem_temp"] = 1.0; life.cfg["rem_steps"] = 6
    for _ in range(3):
        say(life, "dog will go", 2); life.set_face(2.0); life.tick(); life.set_face(0.0); say(life, "give milk", 2)
    ids = [TOK.token_to_id(c) for c in "dog will go"]
    line0 = m.stri_line.clone(); A0 = m.vf_A.clone(); b0 = m.vf_b.clone()
    life._frel_corr = 0.0; n0, _ = life._rem_imagine(ids)
    assert n0 == 0 and torch.equal(m.vf_b, b0), "imagination counted with a face organ that has proved nothing"
    life._frel_corr = 0.5; n1, _ = life._rem_imagine(ids)
    assert n1 == 6 and not torch.equal(m.vf_A, A0), (n1,)
    assert torch.equal(m.stri_line, line0), "the lived delay line was not restored"
    rep = life.night()
    assert rep.get("rem_imagined") and rep["rem_imagined"]["rounds"] >= 1, rep.get("rem_imagined")
    print("32 REM imagines: transitions at correlation 0 ->", n0, "| at 0.5 ->", n1, "| the night:", rep["rem_imagined"])


def test_actor_earned_voice():
    """the actor's earned voice: its reliability is the slope of the reward that follows an act on its vote for it; the voice is silent
    at slope zero and applied at its slope; the reading is from inside"""
    life = _watched(); m = life.m
    life.cfg["actor_voice"] = "earned"; life.cfg["actor_horizon"] = 4; life.cfg["actor_tau"] = 1000
    for _ in range(3):
        say(life, "dog will go", 2); life.set_face(2.0); life.tick(); life.set_face(0.0); say(life, "give milk", 2)
    n_acts = len(life._act_agree)
    assert n_acts > 0 and life._act_pending is not None, "no acts were recorded"
    for i in range(300):                                                  # votes followed by reward in proportion: a reliable actor
        v = (i % 5) / 4.0; life._arel_update(v, 2.0 * v + 0.1 * ((i * 7) % 3 - 1))
    assert life._arel_gain > 0.5, life._arel_gain
    z = m.stri_in(); a_bias = torch.tanh(m.actor(z))
    assert float(a_bias.abs().max()) <= 1.0
    d = life.insides(); assert d["actor_voice"] == "earned" and d["actor_slope"] > 0.5 and d["acts"] == n_acts, d
    print("33 the actor's earned voice: acts read", n_acts, "| agreement", d["actor_agree"], "| slope after reliable votes", d["actor_slope"])


def test_face_on_striatum():
    """the face organ reading the striatal input (face_input striatum, the review of 2026-09-10): its evidence is sized to what the fast
    critic reads, it foresees a smile that follows one line, and imagination scores with it"""
    life = _watched(); m = life.m
    life.cfg["face_form"] = "foresee"; life.cfg["face_input"] = "striatum"; life.cfg["face_every"] = 16; life.cfg["face_tau"] = 2000
    n = int(m.stri_in().numel()); life._fh_n = n
    life._fh_A = torch.zeros(n + 1, n + 1, dtype=torch.float64); life._fh_b = torch.zeros(n + 1, dtype=torch.float64); life._fh_w = torch.zeros(n + 1, dtype=torch.float64)
    fores = []
    for rep in range(30):
        say(life, "dog will go", 1); fores.append(float(life._fpred_now))
        life.set_face(2.0); life.tick(); life.set_face(0.0)
        for _ in range(6):
            life.tick()
    late = sum(fores[-5:]) / 5
    assert abs(fores[0]) < 0.3 and late > 0.3, (fores[0], late)
    assert life._frel_gain > 0.2, life._frel_gain
    d = life.insides(); assert d["face_input"] == "striatum" and d["torn_frac"] is not None
    print("34 the face organ on the striatal input: before the smile", round(fores[0], 2), "->", round(late, 2), "| slope", round(life._frel_gain, 2), "| torn", d["torn_frac"], "entropy", d["ent_mean"])


def test_page_tags_who():
    """the page marks who typed each symbol (the visitor page of 2026-09-11): the parent's and a visitor's symbols are told apart on
    the page, the silence carries no tag, and nothing inside reads the tag"""
    life = tiny()
    life.type_text("hi", who="you"); life.tick(); life.tick()
    life.type_text("go", who="parent"); life.tick(); life.tick(); life.tick()
    ev = [e for e in life.page if e[1] == 0]
    assert [e[4] for e in ev[:5]] == ["you", "you", "parent", "parent", ""], [e[4] for e in ev[:5]]
    assert [e[0] for e in ev[:5]] == ["h", "i", "g", "o", ""], [e[0] for e in ev[:5]]
    assert len(life.state(0)["page"][0]) == 5 and not life.queue_who
    print("35 the page tags who typed:", [e[4] or "-" for e in ev[:5]])


def test_typist_yields():
    """the parent yields to a visitor (2026-09-11): a symbol typed by anyone but the parent holds the typist's lines and faces for
    yield_ticks, its rows say yield and resume, and it scores again after"""
    import json
    import os
    import random
    from body.teacher import Teacher, Corpus, FixedPlanner
    life = tiny()
    log = "/private/tmp/claude-501/-Users-lukehamond-Projects-project/a22528f8-bc83-4acb-9044-d5917dc9456c/scratchpad/yield_test.jsonl"
    if os.path.exists(log):
        os.remove(log)

    class Stub(Teacher):                                   # the page in-process: two ticks a poll, as a served loop would
        def req(self, path, data=None, timeout=30):
            if path.startswith("/state"):
                life.tick(); life.tick()
                return life.state(int(path.split("since=")[1]))
            if path == "/type":
                return life.type_text(data["text"], who=data.get("who", "you"))
            if path == "/face":
                return life.set_face(data["expr"])
            return {}
    t = Stub("", 1, log, Corpus(None), FixedPlanner(random.Random(0)), period=4, quiet=2, cap=8, tick=0.001, listen=2, yield_ticks=20)
    t.poll(); t.finalized = t.maxtick - 1

    def child_says(word):                                  # the child's word on the page, finished, seen now
        m = t.maxtick
        for k in range(m - 8, m):
            t.its[k] = ""; t.tobs[k] = time.time()
        for j, ch in enumerate(word):
            t.its[m - 6 + j] = ch
        t.finalized = m - 9; t.scan()

    life.type_text("hi", who="you"); t.poll()
    assert t.yielding(), "the visitor's symbol did not make the parent yield"
    child_says("dog")
    vt = t.visitor_tick
    t.event("dog will go", "line")                         # the parent's line waits through the yield
    assert not t.yielding()
    parent_ticks = [i // 2 for i, e in enumerate(life.page) if e[1] == 0 and e[0] and e[4] == "parent"]
    assert parent_ticks and min(parent_ticks) >= vt + 20, (vt, parent_ticks[:3])
    child_says("dog")
    rows = [json.loads(l) for l in open(log)]
    kinds = [r["action"] for r in rows]
    assert "yield" in kinds and "resume" in kinds and kinds.index("yield") < kinds.index("resume"), kinds
    smiles = [i for i, k in enumerate(kinds) if k == "smile"]
    assert len(smiles) == 1 and smiles[0] > kinds.index("resume"), kinds
    print("36 the typist yields to a visitor: the line waited", min(parent_ticks) - vt, "ticks; rows", [k for k in kinds if k in ("yield", "resume", "smile", "line")])


def test_night_scales_with_the_day():
    """sleep need scales with the day's plasticity (night_load, 2026-09-11): a day that wrote many memories starts more dreams than
    night_starts, a quiet day starts night_starts, and the count is remembered across a save"""
    life = tiny(night_starts=4, night_load=0.5, night_starts_max=40, night_rounds=2)
    for line in ["dog will go", "give milk", "ball under", "big dog", "I saw dog", "you had ball", "first milk then ball", "scared ball"]:
        say(life, line, 2)
    n_before = life.store.n()
    life._store_after_night = 0                            # everything in the store is the day's
    rep = life.night()
    assert rep["new_slots"] == n_before and rep["dreams"] >= 5, (rep.get("new_slots"), rep.get("dreams"), n_before)
    assert rep["dreams"] <= 40 and life._store_after_night == life.store.n()
    say(life, "dog", 1)                                    # a quiet day
    rep2 = life.night()
    assert rep2["dreams"] == 4, rep2["dreams"]
    import os
    path = "/private/tmp/claude-501/-Users-lukehamond-Projects-project/a22528f8-bc83-4acb-9044-d5917dc9456c/scratchpad/night_load_test.pt"
    life.save(path); life2 = Life.load(path, TOK, device="cpu"); os.remove(path)
    assert life2._store_after_night == life._store_after_night
    print("37 the night scales with the day: a full day ->", rep["dreams"], "dreams of", rep["new_slots"], "new slots; a quiet day ->", rep2["dreams"])


def test_repetition_suppression():
    """the store's repetition suppression (store_sat, 2026-09-11): a memory repeated two hundred times grows like the log of its
    repetitions, the dreams no longer collapse onto it, and a store from before the law is converted once at the load"""
    import os
    life = tiny(store_sat=1)
    assert life.store.saturate
    for _ in range(12):
        say(life, "give milk", 8)                        # one line, repeated after a pause each time: its onset strengthens sub-linearly
    for line in ["dog will go", "ball under", "big dog", "I saw dog", "you had ball", "scared ball", "first milk then ball", "dog had ball"]:
        say(life, line, 8)
    assert int(life.store.Bs.sum()) >= 6, int(life.store.Bs.sum())
    S = life.store.S[:life.store.n()]
    assert float(S.max()) < 6.0 * float(S.mean()), (float(S.max()), float(S.mean()))
    life.gen.manual_seed(0); starts = life.store.sample_starts(64, gen=life.gen)
    top = max(starts.count(j) for j in set(starts)) / 64.0
    assert top < 0.5, top
    # the linear law's skew, converted once at the load
    life2 = tiny(store_sat=0)
    for _ in range(40):
        say(life2, "give milk", 8)
    say(life2, "dog will go", 8)
    S2 = life2.store.S[:life2.store.n()].clone(); m2 = float(S2.mean())
    path = "/private/tmp/claude-501/-Users-lukehamond-Projects-project/a22528f8-bc83-4acb-9044-d5917dc9456c/scratchpad/sat_test.pt"
    life2.save(path); life3 = Life.load(path, TOK, device="cpu", cfg=dict(store_sat=1)); os.remove(path)
    S3 = life3.store.S[:life3.store.n()]
    assert life3.store.sat_done and torch.allclose(S3, m2 * torch.log1p(S2 / m2), atol=1e-5), "the conversion is not m ln(1 + S/m)"
    assert torch.equal(torch.argsort(S2), torch.argsort(S3)), "the order of the strengths changed"
    life3.save(path); life4 = Life.load(path, TOK, device="cpu", cfg=dict(store_sat=1)); os.remove(path)
    assert torch.allclose(life4.store.S[:life4.store.n()], S3), "converted twice"
    skew = lambda t: float(t.max()) / float(t.mean())
    print("38 repetition suppression: max/mean strength", round(skew(S), 2), "| the most drawn start", round(top, 2), "of the dreams | a linear store converted once at the load:", round(skew(S2), 2), "->", round(skew(S3), 2))


def test_dreams_follow_the_episode():
    """the store's sequence links (store_chain, 2026-09-12): a dream from an onset runs the utterance as it was lived, to its end,
    instead of a pattern completion that stops at the first ambiguity; the links survive a save and a pruning"""
    import os
    life = tiny(store_chain=1)
    lines = ["dog will go up", "give milk now", "scared ball under", "big dog bigger dog"]
    for line in lines:
        say(life, line, 10)
    assert int((life.store.N[:, 0] >= 0).sum()) > 20, int((life.store.N[:, 0] >= 0).sum())
    life.gen.manual_seed(0); dreams = life.dreams(8)
    texts = [TOK.decode([i for i in d if i != life.end_id]) for d in dreams]
    lens = [len(t) for t in texts]
    whole = sum(1 for t in texts if any(t.strip() == l for l in lines))
    assert sum(lens) / len(lens) > 8 and whole >= len(texts) // 2, (texts,)
    path = "/private/tmp/claude-501/-Users-lukehamond-Projects-project/a22528f8-bc83-4acb-9044-d5917dc9456c/scratchpad/chain_test.pt"
    life.save(path); life2 = Life.load(path, TOK, device="cpu"); os.remove(path)
    assert torch.equal(life2.store.N, life.store.N)
    keep = torch.arange(life2.store.n() - 3)                 # a pruning: the links follow the slots that stay
    life2.store._keep(keep)
    N = life2.store.N; assert int(N.max()) < life2.store.n() and int((N >= life2.store.n()).sum()) == 0
    # the branch: one frame heard with three continuations is replayed with more than one of them
    life3 = tiny(store_chain=1)
    for _ in range(2):
        for line in ["the man had the web", "the man had the nut", "the man had the hut"]:
            say(life3, line, 10)
    life3.gen.manual_seed(1); ends = {TOK.decode([i for i in d if i != life3.end_id]).split()[-1] for d in life3.dreams(24) if len(d) > 8}
    assert len(ends & {"web", "nut", "hut"}) >= 2, ends
    print("39 dreams follow the episode:", texts[:4], "| mean length", round(sum(lens) / len(lens), 1), "| whole lines", whole, "of", len(texts), "| the branch replays", sorted(ends & {"web", "nut", "hut"}))


def test_actor_chunks():
    """the action chunk (actor_form chunk, 2026-09-12): inside a word the mouth says the cortex's own continuation with no gate
    decision and no sampling, the word ends at a space or the rest, and the actor's credit is taken once per word, not per letter"""
    life = _watched(actor_form="chunk", chunk_max=6, gate_floor=0.9)
    for _ in range(3):
        for line in ["dog will go", "give milk", "big dog"]:
            say(life, line, 6)
    words0 = getattr(life, "_chunk_words", 0); pend0 = len(life._act_pending); n0 = len(life.gate_buf)
    for _ in range(120):
        life.tick()
    words = getattr(life, "_chunk_words", 0) - words0; ticks_in = getattr(life, "_chunk_ticks", 0)
    assert words >= 3 and ticks_in >= words, (words, ticks_in)
    own = [e for e in life.page[-240:] if e[1] == 1 and e[0]]
    assert own, "it said nothing"
    # inside a word the gate had nothing to credit (p_act 1.0 on the program's ticks); a word's start was a decision (p_act < 1)
    rows = list(life.gate_buf)[-120:]
    conts = sum(1 for row in rows if row[1] and abs(float(row[6]) - 1.0) < 1e-9)
    starts = sum(1 for row in rows if row[1] and float(row[6]) < 1.0 - 1e-9)
    assert conts >= starts >= 1 and conts + starts == sum(1 for row in rows if row[1]), (conts, starts)
    # one credit per word: the actor's pending acts grew by the words begun, not by the letters said
    said = sum(1 for e in life.page[-240:] if e[1] == 1 and e[0])
    grew = len(life._act_pending) - pend0
    assert grew <= words + 2 and said > words, (grew, words, said)
    d = life.insides(); assert d["actor_form"] == "chunk" and d["chunk_words"] >= 3
    text = "".join(e[0] if e[0] else "_" for e in life.page[-120:] if e[1] == 1)
    print("40 the actor chunks:", words, "words in", ticks_in, "program ticks,", said, "symbols said; its speech:", repr(text[-60:]))


def test_second_voice():
    """the second voice (2026-09-12): a queue line marked "b:" is typed under the page tag "other" right after the parent's line,
    with no pace before it; the typist does not step aside for it; the corpus counts it as heard; the row says which voice"""
    import json
    import os
    import random
    from body.teacher import Teacher, Corpus, QueuePlanner
    life = tiny()
    log = "/private/tmp/claude-501/-Users-lukehamond-Projects-project/a22528f8-bc83-4acb-9044-d5917dc9456c/scratchpad/voice_test.jsonl"
    q = "/private/tmp/claude-501/-Users-lukehamond-Projects-project/a22528f8-bc83-4acb-9044-d5917dc9456c/scratchpad/voice_queue.jsonl"
    for f in (log, q, q + ".pos"):
        if os.path.exists(f):
            os.remove(f)
    open(q, "a").write(json.dumps({"say": ["where is the dog?", "b: the dog is in the box", "put dog in"]}) + "\n")

    class Stub(Teacher):
        def req(self, path, data=None, timeout=30):
            if path.startswith("/state"):
                life.tick(); life.tick()
                return life.state(int(path.split("since=")[1]))
            if path == "/type":
                return life.type_text(data["text"], who=data.get("who", "you"))
            if path == "/face":
                return life.set_face(data["expr"])
            return {}
    planner = QueuePlanner(q, random.Random(0), pos=0)
    t = Stub("", 1, log, Corpus(None), planner, period=4, quiet=2, cap=8, tick=0.001, listen=2, yield_ticks=20)
    t.poll(); t.finalized = t.maxtick - 1
    items = [planner.next(t) for _ in range(3)]
    assert items[1] == ("the dog is in the box", "line", "other") and len(items[0]) == 2, items
    for it in items:
        t.event(it[0], it[1], it[2] if len(it) > 2 else "parent")
    tags = [e[4] for e in life.page if e[1] == 0 and e[0]]
    assert "other" in tags and "parent" in tags and not t.yielding(), (set(tags), t.yielding())
    rows = [json.loads(l) for l in open(log)]
    voices = [r.get("voice") for r in rows if r["action"] in ("line", "cue")]
    assert voices == ["a", "b", "a"], voices
    assert t.corpus.lines.get("the dog is in the box", 0) == 1 and "box" in t.corpus.words
    print("41 the second voice: tags", sorted(set(tags)), "| voices", voices, "| the typist did not yield")


def test_own_speech_target():
    """the own-speech target (own_target_form recall, 2026-09-12): at its own positions the waking lesson targets the recall's
    continuation of what it said, weighted by the recall's confidence; under the world form its target there is the world's next
    symbol, the same at every own position (the "t t t" of the probe)"""
    life = tiny(own_target_form="recall", wake_every=10 ** 9)
    for _ in range(4):
        say(life, "give milk", 8)
    # a window: the world's line, a pause, then its own "give " with the recall after each symbol
    life.win.clear(); life.bag_w.zero_(); life.bag_o.zero_(); life.n_own = 0
    zero = torch.zeros(life.m.d)
    for ch in "give milk":
        life.win.append({"x": TOK.token_to_id(ch), "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
        life.bag_w = life.cfg["bag_decay"] * life.m.shift(life.bag_w) + life.m.E.weight[TOK.token_to_id(ch)]
    for _ in range(8):
        life.win.append({"x": life.sil, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
    life.bag_w.zero_()
    for ch in "give ":
        i = TOK.token_to_id(ch)
        life.bag_w = life.cfg["bag_decay"] * life.m.shift(life.bag_w) + life.m.E.weight[i]
        rd, conf, _ = life.store.read(life.bag_w)
        life.win.append({"x": life.sil, "xo": i, "face": torch.zeros(2), "bundle": life.bands, "read": rd, "r": 0.0})
    # the lesson's own targets: the recall after "give " says "m"
    xs, whos, faces, bundles, reads = life._window_tensors(list(life.win))
    T = xs.shape[0]; own = [t for t in range(T) if int(whos[t]) != life.sil]
    e_pos = own[-2]                                            # its own 'e' of "give ": the target there is the recall after "give "
    conf_last = float(reads[e_pos + 1].norm()); tgt = TOK.decode([int(life.m.nearest(reads[e_pos + 1]))])
    life._wake_recall_targets = 0
    life.cfg["wake_window"] = T; out = life._wake_lesson()
    assert life._wake_recall_targets >= 2, life._wake_recall_targets
    assert tgt == "m" or conf_last < 0.3, (tgt, conf_last)
    print("40b the own-speech target: the recall after its own 'give ' says", repr(tgt), "at confidence", round(conf_last, 2), "| recall targets in the lesson:", life._wake_recall_targets)


if __name__ == "__main__":
    t0 = time.time()
    tests = [test_corollary_discharge, test_store_recalls, test_recall_is_by_content, test_dreams_are_its_lines, test_night_moves_the_cortex,
             test_rem_learns, test_gate, test_feelings_follow_dopamine, test_sleep_by_fatigue, test_guards, test_ladder_pinned, test_older_gate_loads, test_answer_smile_felt_twice, test_level_input, test_offset, test_ventral_critic]
    tests += [test_striatum, test_working_memory, test_planning_actor, test_new_organs_round_trip, test_chain_closes, test_night_warmup, test_plan_boundary, test_exploration_drive, test_offset_by_settling, test_end_as_rest, test_explore_in_the_choice, test_prefrontal_ceiling, test_calibrated_sharpness, test_evidence_survives_the_load, test_face_foresees, test_rem_imagines, test_actor_earned_voice, test_face_on_striatum, test_page_tags_who, test_typist_yields, test_night_scales_with_the_day, test_repetition_suppression, test_dreams_follow_the_episode, test_actor_chunks, test_second_voice, test_own_speech_target]
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
