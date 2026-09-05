"""fast days: the same body, the same page, the same raw caregiver rules, read on the body's own
clock (ticks) inside one process. A day of 12,000 ticks runs in about a minute.

  python3 -m body.fastlife data/body2_fast.pt --days 10          # continue a body
  python3 -m body.fastlife data/body2_fast.pt --days 10 --birth  # a new one

The caregiver here decides from the page only (CURRICULUM.md, BODY_SPEC.md §6): lines and cues
every 240 ticks after 12 ticks of its quiet (or 180 ticks regardless), a smile 0->2 for 6 ticks
within 12 ticks of a known word of two or more letters bounded by space, edge or rest, at a cue's
completion or its first two letters, an expansion at a letter run's third return, a frown only at a
run of a repeated non-letter mark that is not the space. Never at rest, never at babble.
"""
import argparse
import json
import os
import random
import sys
import time

import torch
from tokenizers import Tokenizer

sys.path.insert(0, "/Users/lukehamond/Projects/project")
from body.life import Life  # noqa: E402
from body.caregiver import KNOWN2, ANSWERS, EXPAND  # noqa: E402

LINES = ["dog will go", "I will go up", "you will go in", "scared dog", "scared ball", "what? scared dog", "give milk", "give ball",
         "give book", "ball under", "ball on", "where ball? ball under", "I had milk", "you had ball", "dog had ball",
         "first milk then ball", "first up then in", "big dog bigger dog", "bigger dog up", "I saw dog", "you saw dog",
         "why dog up? because big dog"]
CUES = ["dog will ", "scared ", "give ", "where ball? ", "I had ", "first milk then ", "big dog bigger ", "why dog up? "]
# THE ANSWER-WEIGHTED SMILE (CURRICULUM.md, in force on the user's word of 2026-09-03 if the slow critics read nothing
# after the teacher): at a cue's completion the smile grows, 2 then 4, and the body feels each rise as an event (its
# felt reward is clipped at 2 per event, so a bigger smile must be a growing one). 1 = the flat smile of every word.
ANSWER_LEVELS = int(os.environ.get("ANSWER_LEVELS", "1"))
# THE PARENT (on the user's word of 2026-09-03, "mold the caregiver to a real parent"): attention that moves.
# An engagement level e, decided from the page alone, rises at an answer or a known word (more at a word new
# today), falls at babble and drifts down in silence; and the parent behaves by it: a known word gets its
# smile with probability e (a distracted parent misses words; a parent stops cheering the hundredth "dog"), the
# parent talks faster when engaged and slower when not, poses more cues when engaged, answers a smiled word
# with a line that holds it, and below a floor turns away for a while (the still face) until it comes back.
# A minute's smiles then depend on the body's own last minutes, which is what a critic at long horizons needs.
# 0 = the flat rules. Nothing reads the body's insides; the answer smile is always given.
PARENT = int(os.environ.get("PARENT", "0"))
# THE PARENT WANTS A REPLY (on the user's word of 2026-09-04): once its cue is answered, each further word the child
# adds before the parent's next turn wears its attention and gets no smile, unless the words go on completing the
# cued line ('where ball? ' 'ball under'); and a word said over the parent's own typing does the same. Decided from
# the page alone; the answer smile is always given. Measured first on fresh seeds (runs 63/64) before the served body.
REPLY = int(os.environ.get("REPLY", "0"))
# THE REPLY WITHHELD (on the user's word of 2026-09-04, "both are your call"): the parent who wants a reply answers
# when the child has finished. Its smile for the answer and its reply (the cued line in full, the recast a parent
# gives) come after the child's quiet of REPLY_QUIET ticks, and every symbol the child adds before that postpones
# them: a parent does not praise over a child still talking, and a child who talks over its parent gets no reply
# until it stops (the contingent response infants work for, Goldstein and West 2003). After the cap it replies
# anyway; the answer smile is always given. Decided from the page alone. The environment's road to turn-taking:
# the run-on's consequence, near a tenth of a smile under the reply rule alone, becomes the delay of the smile and
# of the reply, at the fast critic's horizon. 0 = the reply rule as before (the smile at once).
WAIT = int(os.environ.get("WAIT", "0"))
# The parent's second: a served body of 33 days speaks on half of all ticks and one rest in a hundred reaches eight
# ticks (a quiet of eight within 180 ticks after the parent's utterance 35 percent of the time, median 116 ticks; a
# quiet of four every time, median 22), and infant-parent turn transitions are near a second (Gratier 2015).
REPLY_QUIET = int(os.environ.get("REPLY_QUIET", "4"))
E0, E_FLOOR, E_TAU, E_AWAY, AWAY_TICKS = 0.6, 0.3, 600.0, 0.15, 200   # start, resting level, decay ticks, still-face


class FastCaregiver:
    def __init__(self, life, day, log, rng, period=240, quiet=12, cap=180, smile_ticks=6, react=12):
        self.L, self.day, self.log, self.rng = life, day, log, rng
        self.period, self.quiet, self.cap, self.smile_ticks, self.react = period, quiet, cap, smile_ticks, react
        self.face_until = -1; self.face_val = 0.0
        self.last_smile_tick = -10 ** 9; self.last_frown_tick = -10 ** 9; self.last_word = None
        self.letter_runs = {}; self.expanded = set(); self.pending = []
        self.cue = None; self.smiles = 0; self.frowns = 0; self.words = []
        self.tok_start = None; self.tok_buf = []
        self.last_write_tick = -1
        self.face_plan = []                                   # (tick, level): the growing smile's next rise
        self.e = E0; self.said_today = {}; self.away_until = -1; self.aways = 0; self.expand_next = None
        self.reply_cue = None; self.reply_tokens = []; self.answered = False; self.past = False   # the reply to a cue
        self.typing_span = (-1, -1)                                                            # the parent's own turn, in ticks
        self.reply_pending = None; self.reply_line = None; self.own_since = 0                    # the reply withheld (WAIT)

    def row(self, obj):
        obj = dict(obj); obj.setdefault("day", self.day); obj.setdefault("tick", self.L.ticks)
        self.log.append(obj)

    # one tick of the world: the face as set, then the body ticks, then the page is read
    def pace(self):
        return int(self.period * (1.6 - self.e)) if PARENT else self.period

    def step(self):
        L = self.L
        if PARENT:
            self.e += (E_FLOOR - self.e) / E_TAU                 # attention drifts down in silence
            if self.e < E_AWAY and L.ticks >= self.away_until:
                self.away_until = L.ticks + AWAY_TICKS; self.aways += 1
                self.row({"action": "away", "e": round(self.e, 3)}); self.e = 0.35
        if self.face_plan and L.ticks >= self.face_plan[0][0]:
            _, v = self.face_plan.pop(0); self.face_val = float(v); L.set_face(v)
        if L.ticks >= self.face_until and self.face_val != 0.0:
            self.face_val = 0.0; L.set_face(0.0); self.face_plan = []
        L.tick()
        sym = L.last.get("said", "")
        t = L.ticks
        if sym != "":
            self.last_write_tick = t
            if self.reply_pending is not None:
                self.own_since += 1
        # tokens in its stream: maximal runs of symbols that are not rest and not space
        if sym in ("", " "):
            if self.tok_buf:
                self.on_token("".join(self.tok_buf), self.tok_start, t - 1)
                self.tok_buf = []
        else:
            if not self.tok_buf:
                self.tok_start = t
            self.tok_buf.append(sym)
        rp = self.reply_pending
        if rp is not None:
            yielded = t - self.last_write_tick >= REPLY_QUIET
            if yielded or t - rp["end"] >= self.cap:
                self.reply_pending = None
                self.smile(rp["tok"], rp["why"], extra={"waited": t - rp["end"], "run_on": self.own_since, "yielded": yielded})
                self.reply_line = rp["line"]

    def hold_reply(self, tok, why, cue, answer, b):
        lines = [l for l in LINES if l.startswith(cue) and (l[len(cue):].split() or [""])[0] == answer]
        self.reply_pending = {"tok": tok, "why": why, "end": b, "line": self.rng.choice(lines) if lines else (cue + answer).strip()}
        self.own_since = 0

    def set_face(self, v, ticks):
        self.face_val = float(v); self.L.set_face(v); self.face_until = self.L.ticks + ticks

    def smile(self, on, why, extra=None):
        levels = ANSWER_LEVELS if why.startswith("cue") else 1
        self.set_face(2.0, self.smile_ticks); self.smiles += 1; self.last_smile_tick = self.L.ticks; self.last_word = on
        if levels >= 2:
            self.face_plan = [(self.L.ticks + self.smile_ticks // 2, 4.0)]
        self.words.append((self.L.ticks, on, why))
        self.row(dict({"action": "smile", "on": on, "why": why, "levels": levels, "e": round(self.e, 3), "gate": self.L.last.get("gate"), "mood": round(self.L.mood, 3)}, **(extra or {})))

    def on_token(self, tok, a, b):
        t = self.L.ticks
        if len(tok) >= 3 and len(set(tok.lower())) == 1:
            if tok[0].isalpha():
                k = tok[0].lower(); self.letter_runs[k] = self.letter_runs.get(k, 0) + 1
                if self.letter_runs[k] >= 3 and k not in self.expanded and k in EXPAND:
                    self.expanded.add(k); self.pending.append(EXPAND[k])
            elif tok[0] != " " and t - self.last_frown_tick > 240:
                self.set_face(-2.0, self.smile_ticks); self.frowns += 1; self.last_frown_tick = t
                self.row({"action": "frown", "on": tok}); return
        low = tok if (tok == "I" or tok.islower()) else ""
        if PARENT and t < self.away_until:
            return                                                # the parent is turned away
        if PARENT and REPLY:
            ts, te = self.typing_span
            if a < te and b >= ts:                                # said over the parent's own turn
                self.e = max(0.0, self.e - 0.04); self.row({"action": "missed", "on": tok, "why": "talked over", "e": round(self.e, 3)}); return
        c = self.cue
        if c and t <= c["until"] and not c["done"] and len(tok) >= 2 and low:
            if low in c["full"]:
                c["done"] = True; self.answered = True; self.reply_tokens.append(low)
                self.e = min(1.0, self.e + 0.25)
                if WAIT:
                    self.hold_reply(tok, "cue completion: " + c["text"], c["text"], low, b); return
                self.smile(tok, "cue completion: " + c["text"]); return
            if low[:2] in [x[:2] for x in c["full"]]:
                c["done"] = True; self.answered = True; self.reply_tokens.append(low)
                self.e = min(1.0, self.e + 0.15)
                if WAIT:
                    full = [x for x in c["full"] if x[:2] == low[:2]][0]
                    self.hold_reply(tok, "cue prefix: " + c["text"], c["text"], full, b); return
                self.smile(tok, "cue prefix: " + c["text"]); return
        if PARENT and REPLY and self.answered:
            # past its answer: the parent asked and got a speech (the words may go on completing the cued line)
            said = (self.reply_cue + " ".join(self.reply_tokens + [tok])).strip()
            if self.past or not any(l.startswith(said) for l in LINES):
                self.past = True; self.e = max(0.0, self.e - 0.04)
                self.row({"action": "missed", "on": tok, "why": "past its answer", "e": round(self.e, 3)}); return
            self.reply_tokens.append(tok)
        if low in KNOWN2 and t - b <= self.react:
            if self.last_word == low and t - self.last_smile_tick < 48:
                return
            if t - self.last_smile_tick >= 8:
                if PARENT:
                    n = self.said_today.get(low, 0); self.said_today[low] = n + 1
                    p = self.e * (0.95 ** max(0, n - 5))        # attention, and the fiftieth "dog" (0.1 by then)
                    self.e = min(1.0, self.e + (0.1 if n == 0 else 0.05))
                    if self.rng.random() > p:
                        self.row({"action": "missed", "on": tok, "why": "distracted", "e": round(self.e, 3)}); return
                    self.expand_next = low
                self.smile(tok, "known word")
        elif PARENT and low and len(low) >= 3 and low not in KNOWN2 and not any(w.startswith(low) for w in KNOWN2):
            self.e = max(0.0, self.e - 0.04)                      # babble wears the parent's attention

    def wait_gate(self):
        t0 = self.L.ticks
        while True:
            if self.L.asleep or self.L.sleep_pressure >= self.L.cfg["wake_ticks"]:
                return False
            if (self.L.ticks - self.last_write_tick >= self.quiet and self.L.ticks - t0 >= 2) or self.L.ticks - t0 >= self.cap:
                return True
            self.step()

    def event(self, text, kind):
        if not self.wait_gate():
            return False
        if kind == "cue":
            self.cue = {"text": text, "until": self.L.ticks + 96, "full": ANSWERS.get(text, []), "done": False}
        self.reply_cue = text if kind == "cue" else None; self.reply_tokens = []; self.answered = False; self.past = False
        self.L.type_text(text)
        t_start = self.L.ticks; its = []
        self.typing_span = (t_start + 1, 10 ** 9)                  # the parent's turn: from its first symbol to its last
        while self.L.queue:
            self.step()
            if not self.L.queue:                                # the tick the last symbol entered: its answer may begin here
                its.append(self.L.last.get("said", "") or "_")
        t_end = self.L.ticks
        self.typing_span = (t_start + 1, t_end)
        for _ in range(24):
            self.step(); its.append(self.L.last.get("said", "") or "_")
        self.row({"action": kind, "text": text, "its_after": "".join(its), "own": self.L.last.get("own"), "gate": self.L.last.get("gate"),
                  "fatigue": round(self.L.fatigue, 2), "mood": round(self.L.mood, 3), "store": self.L.store.n(), "t": t_start})
        if kind == "cue":
            for _ in range(8):
                self.step()
            self.cue = None
        return True

    def pace_wait(self, last_t, pace):
        """ticks until last_t + pace; under WAIT a reply that comes due is spoken at once and the pace restarts
        from it. Returns the new last_t, or None if the body fell asleep."""
        L = self.L
        while L.ticks < last_t + pace:
            if L.sleep_pressure >= L.cfg["wake_ticks"]:
                break
            if WAIT and self.reply_line:
                line = self.reply_line; self.reply_line = None
                last_t = L.ticks
                if not self.event(line, "line"):
                    return None
                continue
            self.step()
        return last_t

    def run_day(self):
        L = self.L
        self.row({"action": "day_start", "sleep_pressure": L.sleep_pressure, "nights": L.nights})
        order = self.rng.sample(LINES, len(LINES)); plan = []; ci = 0
        for i, line in enumerate(order):
            plan.append(("line", line))
            if i % 2 == 1 and ci < len(CUES):
                plan.append(("cue", CUES[ci])); ci += 1
        last_t = None; slept = False; pi = 0
        while pi < len(plan) or (PARENT and L.sleep_pressure < L.cfg["wake_ticks"] - 600):
            if pi < len(plan):
                kind, text = plan[pi]; pi += 1
            else:
                kind, text = "line", self.rng.choice(LINES)          # an engaged parent keeps talking till the night
            while self.pending:
                self.event(self.pending.pop(0), "line")
            if PARENT and self.expand_next and kind == "line":
                holds = [l for l in LINES if self.expand_next in l.split()]
                if holds:
                    text = self.rng.choice(holds)                  # answering its word with a line that holds it
                self.expand_next = None
            if last_t is not None:
                last_t = self.pace_wait(last_t, self.pace())
                if last_t is None:
                    slept = True; break
            if PARENT and L.ticks < self.away_until:
                while L.ticks < self.away_until:
                    self.step()
            last_t = L.ticks
            if not self.event(text, kind):
                slept = True; break
        if not slept:
            self.event("bye", "line")
        # the night comes when the pressure crosses; the body sleeps inside tick()
        n0 = L.nights
        while L.nights == n0:
            self.step()
        self.reply_line = None                                # a reply due at the night's edge is not given
        night = L.last_night
        self.row({"action": "night", "record": night})
        # the post-night cues, a line between each pair
        lines2 = self.rng.sample(LINES, len(LINES)); j = 0; last_t = None
        for i, c in enumerate(CUES):
            for text, kind in ([(c, "cue")] + ([(lines2[j % len(lines2)], "line")] if i % 2 == 1 else [])):
                if last_t is not None:
                    last_t = self.pace_wait(last_t, self.period) or L.ticks
                last_t = L.ticks; self.event(text, kind)
                if kind == "line":
                    j += 1
        self.row({"action": "day_end", "smiles": self.smiles, "frowns": self.frowns, "aways": self.aways, "e": round(self.e, 3), "words": self.words[-20:]})
        return night


def summarize(log, night, day):
    cues = [r for r in log if r.get("action") == "cue" and r.get("day") == day]
    smiles = [r for r in log if r.get("action") == "smile" and r.get("day") == day]
    held = [r for r in smiles if "run_on" in r]
    reply = (f" | reply: run-on {sum(r['run_on'] for r in held) / len(held):.1f} waited {sum(r['waited'] for r in held) / len(held):.0f}"
             f" yielded {sum(1 for r in held if r['yielded'])}/{len(held)}") if held else ""
    g = night.get("gauge", {}) if night else {}
    print(f"day {day}: smiles {len(smiles)} ({', '.join(sorted(set(r['on'] for r in smiles)))[:60]}) | cues {len(cues)}: "
          + " ".join(repr(r['its_after'].replace('_', '')[:6]) for r in cues[-8:])
          + f" | night: dreams {night.get('dreams')} len {night.get('mean_len')} nrem {(night.get('nrem_loss') or ['?'])[0]}->{(night.get('nrem_loss') or ['?'])[-1]} gauge {g.get('before')}->{g.get('after_nrem')}->{g.get('after')} "
          f"cos {g.get('cos_before')}->{g.get('cos_after')} rem {night.get('rem_cos_first')}->{night.get('rem_cos')} discarded {night.get('discarded')} "
          f"| gate {cues[-1].get('gate') if cues else None} store {night.get('store_slots')} vrel {night.get('vrel')}{reply} | dreams e.g. {[e[:14] for e in (night.get('examples') or [])[:3]]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("body"); ap.add_argument("--tok", default="/Users/lukehamond/Projects/project/data/tok_char.json")
    ap.add_argument("--days", type=int, default=5); ap.add_argument("--birth", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log", default=None)
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=6)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tok)
    if a.birth:
        life = Life.birth(tok, device="cpu", d=a.d, layers=a.layers, heads=4, window=64, seed=a.seed, save_path=a.body); life.save()
    else:
        life = Life.load(a.body, tok, device="cpu", save_path=a.body)
    rng = random.Random(a.seed)
    log = []
    for k in range(a.days):
        day = life.day_n + 1
        t0 = time.time()
        cg = FastCaregiver(life, day, log, rng)
        night = cg.run_day()
        summarize(log, night, day)
        life.save()
        if a.log:
            with open(a.log, "a") as f:
                for r in log:
                    if r.get("day") == day and r.get("action") != "night":
                        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"   ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
