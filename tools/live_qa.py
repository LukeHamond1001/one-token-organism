"""THE QUESTION ASKED OF THE LIVE MOUTH (a supervisor's instrument, 2026-09-15): the demo's form. Each question is typed as the parent
types it (type_text, the ticks as lived: the gate, the sampled choice, the store writing), then the parent is silent for a window of
ticks and the child's own symbols on the page are read; the question counts as answered when a content word of the fact that the
question lacks appears in that turn (qa_by_gap's rule). Several passes, since the choice samples. The copy's sleep pressure is
zeroed so the copy does not fall asleep under the questions.
usage: python3 tools/live_qa.py BODY.pt --flags FLAGS.txt [--set facts|novel] [--reps 3] [--window 60] [--gap 30]"""
import sys, os, re, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tokenizers import Tokenizer
from body.life import Life, PHYSIOLOGY
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def arg(name, default):
    names = {"--" + name, "--" + name.replace("_", "-")}
    for i, a in enumerate(sys.argv[1:], 1):
        if a in names and i + 1 < len(sys.argv):
            return type(default)(sys.argv[i + 1])
    return default
def parse_flags(s):
    toks = s.split(); cfg = {}; i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            k = t[2:].replace("-", "_")
            if k in PHYSIOLOGY:
                cfg[k] = type(PHYSIOLOGY[k])(toks[i + 1])
            i += 2
        else:
            i += 1
    return cfg
NOVEL = [("what is sour?", "a lemon is sour"), ("what is loud?", "a drum is loud"), ("what is a pear?", "a pear is a fruit"),
         ("what is salty?", "the sea is salty"), ("what is slow?", "a snail is slow"), ("what do owls say?", "owls hoot"),
         ("what gives light?", "a lamp gives light"), ("what has wool?", "a sheep has wool"), ("what has a horn?", "a goat has a horn"),
         ("what is steep?", "a hill is steep")]
path = sys.argv[1]; qset = arg("set", "facts"); reps = arg("reps", 3); window = arg("window", 60); gap = arg("gap", 30)
quiet = arg("quiet", 8); max_wait = arg("max_wait", 200); trace = arg("trace", 0); smile = arg("smile", 0); interleave = arg("interleave", 0); day_ = arg("day", 235)   # --interleave 1: an ordinary exchange of the logged day (A then B) before each question, as the real day goes
# --smile 1: the parent smiles as the caregiver does: at a known word said in the child's turn (+2 for twelve ticks; the same word not
# within 120 ticks, the face returned five ticks before the next), and the answer's growing smile (+2 then +4) when the fact's word
# comes; without it the copy's mood sinks to the readout's floor through a run, which no parent's child lives under
KNOWN = set(w for w in re.findall(r"[a-z]+", open(os.path.join(ROOT, "data/teach_queue_w2.jsonl")).read().lower()) if len(w) >= 2) if smile else set()                    # --trace N: the first N questions tick by tick                       # the parent waits for the child's quiet before the next question, as the typist does
cfg = parse_flags(open(arg("flags", "")).read()) if arg("flags", "") else {}
for a in sys.argv[1:]:                                                          # any physiology constant may be overridden on the line
    if a.startswith("--") and "=" not in a:
        k = a[2:].replace("-", "_")
        if k in PHYSIOLOGY:
            cfg[k] = type(PHYSIOLOGY[k])(sys.argv[sys.argv.index(a) + 1])
TOK = Tokenizer.from_file(os.path.join(ROOT, "data/tok_char.json"))
life = Life.load(path, TOK, device="cpu", cfg=cfg); life.save_path = None; life.sleep_pressure = 0
pairs = NOVEL if qset == "novel" else [tuple(s.strip() for s in l.split("|")[:2]) for l in open(os.path.join(ROOT, "tools/facts_stage5.txt")) if "|" in l]
STOP = set("is are the a an in on and to do we i you it of my your they can what where who how does".split())
print(f"body {os.path.basename(path)}: nights {life.nights} store {life.store.n()} | {qset} questions, {reps} passes, the child's turn {window} ticks then {gap} of quiet", flush=True)
hits = {q: 0 for q, _ in pairs}; first = {}; t0 = time.time(); junk = 0; own_total = 0; talked_over = 0; word_tick = {}; smiles_given = [0]; last_face_end = -10 ** 9
ordinary = []
if interleave:
    import json
    R_ = [json.loads(l) for l in open(os.path.join(ROOT, "data/watch2_caregiver.jsonl")) if l.strip()]
    L_ = [(r["text"], "other" if r.get("voice") == "b" else "you") for r in R_ if r.get("day") == day_ and r["action"] == "line"]
    ordinary = [(L_[i], L_[i + 1]) for i in range(0, len(L_) - 1, 2) if L_[i][1] == "you" and L_[i + 1][1] == "other" and not any(f in L_[i][0] for f, _ in pairs)]
    print(f"interleaving {len(ordinary)} ordinary exchanges of day {day_} between the questions", flush=True)
def wait_quiet():
    silent = 0; waited = 0
    while silent < quiet and waited < max_wait:
        life.tick(); waited += 1
        silent = silent + 1 if int(life.win[-1]["xo"]) == life.sil else 0
    return silent < quiet
def say_line(text, who):
    wait_quiet(); life.type_text(text, who=who)
    while life.queue: life.tick()
    for _ in range(20): life.tick()                                             # the typist's four seconds before the other voice
for rep in range(reps):
    n = 0
    for qi, (q, fact) in enumerate(pairs):
        keys = [w for w in re.findall(r"[a-z]+", fact.lower()) if w not in STOP and w not in q.lower()]
        if ordinary:
            (a_, aw), (b_, bw) = ordinary[(rep * len(pairs) + qi) % len(ordinary)]; say_line(a_, aw); say_line(b_, bw)
        silent = 0; waited = 0
        while silent < quiet and waited < max_wait:                             # the typist's rule: no line over the child's speech
            life.tick(); waited += 1
            silent = silent + 1 if int(life.win[-1]["xo"]) == life.sil else 0
        talked_over += int(silent < quiet)
        life.type_text(q, who="you")
        tr = rep == 0 and qi < trace; rows = []
        def note(tag):
            c = getattr(life, "_last_choice", None); w = life.win[-1]
            if c: rows.append(f"{tag}{TOK.decode([int(w['x'])]) if int(w['x']) != life.sil else '_'}{TOK.decode([int(w['xo'])]) if int(w['xo']) != life.sil else '_'} act {c['p_act']:.2f}{'!' if c['acted'] else ' '} say {TOK.decode([c['nxt']]) if c['nxt'] != life.sil else '_'} p {c['p_choice']:.2f} top {TOK.decode([c['top']]) if c['top'] != life.sil else '_'} |pred| {c['norm']:.2f} sharp {c['sharp']:.0f}")
        got = []
        while life.queue:
            life.tick()
            if tr: note("q ")
            if not life.queue:                                                  # the question's last tick: a symbol said in it belongs to the answer
                w = life.win[-1]
                if int(w["xo"]) != life.sil: got.append(TOK.decode([int(w["xo"])]))
        answered_at = -1; face_until = -1; face_then = None; seen_words = 0
        for t_ in range(window):
            life.tick(); w = life.win[-1]; tick_now = life.ticks
            if int(w["xo"]) != life.sil: got.append(TOK.decode([int(w["xo"])]))
            if tr: note("  ")
            if smile:
                text_ = "".join(got).lower(); words_ = re.findall(r"[a-z]+", text_)
                done_ = words_[:-1] if text_ and text_[-1].isalpha() else words_          # a word is said when its last letter is followed by a non-letter
                if face_until >= 0 and tick_now >= face_until:
                    if face_then is not None and tick_now < face_then[1]:
                        life.set_face(face_then[0]); face_until = face_then[1]; face_then = None
                    else:
                        life.set_face(0.0); face_until = -1; last_face_end = tick_now
                if face_until < 0 and tick_now - globals().get("last_face_end", -10 ** 9) >= 5:
                    if answered_at < 0 and any(k in text_ for k in keys):                  # the answer's growing smile: 2 then 4
                        life.set_face(2.0); face_until = tick_now + 12; face_then = (4.0, tick_now + 24); answered_at = t_; smiles_given[0] += 1
                    elif len(done_) > seen_words:
                        wd = done_[seen_words]; seen_words = len(done_)
                        if wd in KNOWN and tick_now - word_tick.get(wd, -10 ** 9) > 120:      # a known word, not smiled at within 120 ticks
                            life.set_face(2.0); face_until = tick_now + 12; word_tick[wd] = tick_now; smiles_given[0] += 1
                    else:
                        seen_words = max(seen_words, len(done_))
        if tr:
            print(f"TRACE {q!r} (world symbol, own symbol; act = the gate's probability, ! = it acted; say = the sampled symbol at its probability; top = the readout's argmax):", flush=True)
            for r in rows: print("   " + r, flush=True)
        life.set_face(0.0); face_until = -1
        for _ in range(gap):
            life.tick()
        text = "".join(got); own_total += len(text); junk += sum(1 for ch in text if not (ch.islower() or ch in " .?!'"))
        ok = any(k in text.lower() for k in keys); n += int(ok); hits[q] += int(ok)
        if rep == 0: first[q] = (ok, text[:24])
    print(f"LIVE pass {rep + 1}: {n}/{len(pairs)} answered in the child's turn ({time.time() - t0:.0f}s)", flush=True)
ever = sum(1 for q in hits if hits[q] > 0)
print(f"LIVE {qset} (sharp_base {life.cfg['sharp_base']}, read_sharp {float(life.m.read_sharp):.1f}, mood {life.mood:.1f}, smiles {'on' if smile else 'off'}, rest {life.cfg.get('bag_rest_decay')}, sharp_conf {life.cfg.get('sharp_conf')}): answered in {sum(hits.values()) / max(1, reps):.1f} of {len(pairs)} per pass, {ever} ever | own symbols {own_total}, junk {junk}, smiles given {smiles_given[0]}, questions typed over its speech {talked_over} | " + " ".join(f"{'*' if ok else ' '}{q!r}->{t!r}" for q, (ok, t) in first.items()), flush=True)
