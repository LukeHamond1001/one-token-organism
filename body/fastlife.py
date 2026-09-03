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

    def row(self, obj):
        obj = dict(obj); obj.setdefault("day", self.day); obj.setdefault("tick", self.L.ticks)
        self.log.append(obj)

    # one tick of the world: the face as set, then the body ticks, then the page is read
    def step(self):
        L = self.L
        if L.ticks >= self.face_until and self.face_val != 0.0:
            self.face_val = 0.0; L.set_face(0.0)
        L.tick()
        sym = L.last.get("said", "")
        t = L.ticks
        if sym != "":
            self.last_write_tick = t
        # tokens in its stream: maximal runs of symbols that are not rest and not space
        if sym in ("", " "):
            if self.tok_buf:
                self.on_token("".join(self.tok_buf), self.tok_start, t - 1)
                self.tok_buf = []
        else:
            if not self.tok_buf:
                self.tok_start = t
            self.tok_buf.append(sym)

    def set_face(self, v, ticks):
        self.face_val = float(v); self.L.set_face(v); self.face_until = self.L.ticks + ticks

    def smile(self, on, why):
        self.set_face(2.0, self.smile_ticks); self.smiles += 1; self.last_smile_tick = self.L.ticks; self.last_word = on
        self.words.append((self.L.ticks, on, why))
        self.row({"action": "smile", "on": on, "why": why, "gate": self.L.last.get("gate"), "mood": round(self.L.mood, 3)})

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
        c = self.cue
        if c and t <= c["until"] and not c["done"] and len(tok) >= 2 and low:
            if low in c["full"]:
                c["done"] = True; self.smile(tok, "cue completion: " + c["text"]); return
            if low[:2] in [x[:2] for x in c["full"]]:
                c["done"] = True; self.smile(tok, "cue prefix: " + c["text"]); return
        if low in KNOWN2 and t - b <= self.react:
            if self.last_word == low and t - self.last_smile_tick < 48:
                return
            if t - self.last_smile_tick >= 8:
                self.smile(tok, "known word")

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
        self.L.type_text(text)
        t_start = self.L.ticks; its = []
        while self.L.queue:
            self.step()
            if not self.L.queue:                                # the tick the last symbol entered: its answer may begin here
                its.append(self.L.last.get("said", "") or "_")
        t_end = self.L.ticks
        for _ in range(24):
            self.step(); its.append(self.L.last.get("said", "") or "_")
        self.row({"action": kind, "text": text, "its_after": "".join(its), "own": self.L.last.get("own"), "gate": self.L.last.get("gate"),
                  "fatigue": round(self.L.fatigue, 2), "mood": round(self.L.mood, 3), "store": self.L.store.n(), "t": t_start})
        if kind == "cue":
            for _ in range(8):
                self.step()
            self.cue = None
        return True

    def run_day(self):
        L = self.L
        self.row({"action": "day_start", "sleep_pressure": L.sleep_pressure, "nights": L.nights})
        order = self.rng.sample(LINES, len(LINES)); plan = []; ci = 0
        for i, line in enumerate(order):
            plan.append(("line", line))
            if i % 2 == 1 and ci < len(CUES):
                plan.append(("cue", CUES[ci])); ci += 1
        last_t = None; slept = False
        for kind, text in plan:
            while self.pending:
                self.event(self.pending.pop(0), "line")
            if last_t is not None:
                while L.ticks < last_t + self.period:
                    if L.sleep_pressure >= L.cfg["wake_ticks"]:
                        break
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
        night = L.last_night
        self.row({"action": "night", "record": night})
        # the post-night cues, a line between each pair
        lines2 = self.rng.sample(LINES, len(LINES)); j = 0; last_t = None
        for i, c in enumerate(CUES):
            for text, kind in ([(c, "cue")] + ([(lines2[j % len(lines2)], "line")] if i % 2 == 1 else [])):
                if last_t is not None:
                    while L.ticks < last_t + self.period:
                        self.step()
                last_t = L.ticks; self.event(text, kind)
                if kind == "line":
                    j += 1
        self.row({"action": "day_end", "smiles": self.smiles, "frowns": self.frowns, "words": self.words[-20:]})
        return night


def summarize(log, night, day):
    cues = [r for r in log if r.get("action") == "cue" and r.get("day") == day]
    smiles = [r for r in log if r.get("action") == "smile" and r.get("day") == day]
    g = night.get("gauge", {}) if night else {}
    print(f"day {day}: smiles {len(smiles)} ({', '.join(sorted(set(r['on'] for r in smiles)))[:60]}) | cues {len(cues)}: "
          + " ".join(repr(r['its_after'].replace('_', '')[:6]) for r in cues[-8:])
          + f" | night: dreams {night.get('dreams')} len {night.get('mean_len')} nrem {(night.get('nrem_loss') or ['?'])[0]}->{(night.get('nrem_loss') or ['?'])[-1]} gauge {g.get('before')}->{g.get('after_nrem')}->{g.get('after')} "
          f"cos {g.get('cos_before')}->{g.get('cos_after')} rem {night.get('rem_cos_first')}->{night.get('rem_cos')} discarded {night.get('discarded')} "
          f"| gate {cues[-1].get('gate') if cues else None} store {night.get('store_slots')} | dreams e.g. {[e[:14] for e in (night.get('examples') or [])[:3]]}", flush=True)


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
