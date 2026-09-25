"""the night over frames (a mixin of `Life`, body/life.py; the core refactor's step R8, docs/SIM_DESIGN.md 3.6, 3.7, 5.4, 7.3, 7.4 item 2, 8's
R8 row, 9, 10; A46): the night of a body that lives in the world's frames (the sim), under physiology.py's SLEEP switches, which the
diary's cfg does not hold, so none of it runs for the language body.

STEP R8a, THE DAY'S TAPE AND THE EPISODES (7.4 item 2, 9's tape; the switch `night_frames`, which needs `frames`):
- THE TAPE (`_tape_tick`, at each awake tick's end, after R7b's record): the tick's window position as the cortex received it, the
  frame's "stored codes": every vector channel's numbers at fp16 (the G1's 3,833: the face, both cochleas, the periphery, the fovea, the
  body, touch, the inertial units, the charge), every symbol channel's symbol (the words'), whether the offset ended an utterance there
  (the waking lesson's end target; marked when the offset falls, `_tape_mark_end`), the voice's symbol (its own sound, "xo") and every
  motor effector's act (the efference copies the cortex heard: a rest where it did not act), in blocks of TAPE_BLOCK rows. A row is a
  tick, and the tick's record (R7b's surprise, dopamine, tag and net reward) is the same row of the day's record. About 7.8 KB a tick at
  the G1's sizes (7.7 KB of fp16, SIM_DESIGN.md 9). The day's tape is the day's working state, saved with its day (A70).
- THE EPISODES (`_episodes_nightfall`, as the night begins, after the night has ended every event): the day cut at the frames' event
  ends (R7b's `_rec_ends`: the first settled tick after an unsettled one begins the next episode; the night ends the last); over the
  day's record, the tag reaching back (tag*, `tag_star_fast`, equal to R7d's `tag_star` to the bit) and each episode's entry and T_e
  (R7d's `episode_entries`: [its mean surprise x (1 + |dopamine|)] x (1 + T_e) over the bias-corrected running mean of entry_tau
  entries, T_e the largest tag* over it, so reaching up to tag_reach ticks past its end); its DREAM WINDOW (7.4 item 2: "the window
  dreamt ends at the episode's peak tag, so cause and outcome are both inside; an episode whose tag never reaches 0.1 is dreamt to its
  end"): the cortex's window of ticks (`m.window`) ending at the tick whose discounted tag gives T_e (the tag's own tick where the
  outcome falls inside the episode, or up to tag_reach - 1 ticks past its end: the outcome the tag reached back from), or at the
  episode's last tick where T_e is under peak_floor, reaching back before the episode's start within the day (the cause may begin in
  the event before: THE BUILDER'S READING, for the lead: 7.4 names the window's end and not its start; a window of the cortex's
  length, the length a dream's lockstep batch runs, from the day's tape wherever that falls); and each row's REPLAYED DOPAMINE'S CREDIT
  G (7.4 item 2: "G_t is the replayed dopamine's credit over the next 64 ticks"): the sum over k = 1..tag_reach of 0.9375^(k-1) x the
  dopamine k ticks later, dopamine's own discount from the tick after the act (the act's own consequences: defect 8's `elig_from`),
  within the day (THE BUILDER'S READING, for the lead: 7.4 gives the span and not the weights; the amygdala's horizon, the tag's reach
  and dopamine's discount are one law, so the credit is summed on it). A window shorter than two ticks teaches nothing and is not kept.
- THE REELS: the day's tape becomes a reel (its rows with G), each episode naming its window's rows in it; the episodes are kept across
  nights (the first design's "the sim's utterance memory": "a rewarded episode can be replayed on later nights"), each entry fading as
  the store's strengths do (store_fade, 0.9 a night: 7.4's "entries fade 0.9 a night"), and after each night the weakest give way until
  the kept windows hold at most episode_cap distinct ticks of tape (section 10's "episode cap", 9's "inside the episodes (cap 24,000
  ticks)"; the lowest entry first, of equals the oldest: the store's rule), each reel then compacted to the rows its kept windows
  read (`_episodes_fade`). THE BUILDER'S READING, for the lead: an episode keeps only the rows its dreams read (its window), never its
  whole span, so the cap counts what a night can dream; about 184 MB at the cap.
- THE ENTRIES' RUNNING MEAN, the reels, the episodes and their serial count are the body's memory across nights: saved in its life
  (life["episodes"]) as the utterances heard are, not in its day (body/core/persistence.py).
The night's draw, its passes over frames and the live, dark night are R8b and R8c (below)."""
import math

import torch

from .amygdala import episode_entries, night_draw  # noqa: F401  (night_draw: R8b's)
from .physiology import FRAMES, SLEEP

TAPE_BLOCK = 512                                                         # rows a block of the day's tape holds (a save keeps its last block
                                                                         # whole: at most 511 rows unused, about 4 MB at the G1's sizes)


def tag_star_fast(tags, gamma, reach):
    """THE TAG REACHES BACK, vectorized (R7d's `tag_star`, the same numbers to the bit: each candidate is float(gamma) ** k times the
    tag k ticks later in double precision, and the largest is kept): `tags` a float64 tensor [T]; returns tag* [T] (float64)"""
    t = tags.to(torch.float64)
    T = int(t.numel())
    out = torch.zeros(T, dtype=torch.float64)
    for k in range(min(int(reach), T)):
        cand = float(gamma) ** k * t[k:]
        out[:T - k] = torch.maximum(out[:T - k], cand)
    return out


def credit_after(delta, gamma, reach):
    """THE REPLAYED DOPAMINE'S CREDIT of each tick's act (7.4 item 2; the module's doc): G_t = the sum over k = 1..reach of gamma^(k-1) x
    delta_(t+k) within the record (float64, the terms added in k's order); `delta` [T]"""
    d = delta.to(torch.float64)
    T = int(d.numel())
    G = torch.zeros(T, dtype=torch.float64)
    for k in range(1, int(reach) + 1):
        if k >= T:
            break
        G[:T - k] += float(gamma) ** (k - 1) * d[k:]
    return G


def peak_of(tags, a, b, gamma, reach):
    """THE EPISODE'S PEAK (the module's doc): the tick r in [a, min(T, b - 1 + reach)) whose tag discounted from the episode's end,
    gamma^max(0, r - (b - 1)) x tag_r, is the largest (the first of equals): the outcome T_e reached back from; (r, its value)"""
    T = int(tags.numel()); hi = min(T, b - 1 + int(reach))
    best, at = -1.0, b - 1
    for r in range(a, hi):
        v = (float(gamma) ** max(0, r - (b - 1))) * float(tags[r])
        if v > best:
            best, at = v, r
    return at, best


class SleepMixin:
    # ---------------- the switches ----------------
    def _sleep_const(self, k):
        """a constant of the night over frames: the body's cfg when it was given, else SLEEP's"""
        return self.cfg.get(k, SLEEP[k])

    def _night_frames_on(self):
        """the switch `night_frames` (step R8): 0 for the diary, whose cfg holds no such key"""
        return bool(int(self.cfg.get("night_frames", SLEEP["night_frames"])))

    def _sleep_check(self):
        """THE SWITCHES THAT NEED WHAT THEY READ (step R8), at birth and at a load: the night over frames cuts R7b's record and event ends,
        so it needs `frames`; refused otherwise. The diary holds none of the keys, so nothing is refused"""
        if self._night_frames_on() and not self._frames_on():
            raise ValueError("Life: the night over frames (night_frames 1) cuts the frames' record into episodes and needs the frames (frames 1)")

    # ---------------- step R8a: the day's tape ----------------
    def _tape_layout(self):
        """the tape's row, from the anatomy: its symbol channels (the words first), its vector channels with their sizes, in the declared
        order, and its motor effectors"""
        a = self.anatomy
        sym = [c for c in a.channels if c.kind == "symbol"]
        vec = [c for c in a.channels if c.kind == "vector"]
        return sym, vec, list(a.motors)

    def _tape_block(self):
        sym, vec, mot = self._tape_layout()
        V = sum(int(c.size) for c in vec)
        return {"s": torch.zeros(TAPE_BLOCK, len(sym), dtype=torch.long), "v": torch.zeros(TAPE_BLOCK, V, dtype=torch.float16),
                "xo": torch.zeros(TAPE_BLOCK, dtype=torch.long), "a": torch.zeros(TAPE_BLOCK, len(mot), dtype=torch.long),
                "end": torch.zeros(TAPE_BLOCK, dtype=torch.bool)}

    def _tape_tick(self):
        """THE TICK TAPED (the module's doc), at the tick's end: this tick's window position (the last), its row the day's next, kept in
        the position ("tape") for the offset's end mark"""
        w = self.win[-1]
        tape = getattr(self, "_tape", None)
        if tape is None:
            tape = []; self._tape = tape
        n = int(getattr(self, "_tape_n", 0))
        b, r = divmod(n, TAPE_BLOCK)
        if b == len(tape):
            tape.append(self._tape_block())
        blk = tape[b]
        sym, vec, mot = self._tape_layout()
        with torch.no_grad():
            if vec:
                blk["v"][r] = torch.cat([torch.as_tensor(w[c.field]).reshape(-1).float().cpu() for c in vec]).to(torch.float16)
            for j, c in enumerate(sym):
                blk["s"][r, j] = int(w[c.field])
            blk["xo"][r] = int(w["xo"])
            for j, e in enumerate(mot):
                blk["a"][r, j] = int(w[e.field])
        blk["end"][r] = bool(w.get("end", False))
        w["tape"] = n
        self._tape_n = n + 1

    def _tape_mark_end(self, row):
        """the offset ended an utterance at this row's symbol (body/core/senses.py `_offset`): the waking lesson's end target, kept for the
        night"""
        tape = getattr(self, "_tape", None)
        if tape is None or not 0 <= int(row) < int(getattr(self, "_tape_n", 0)):
            return
        b, r = divmod(int(row), TAPE_BLOCK)
        tape[b]["end"][r] = True

    def _tape_rows(self, n):
        """the day's first n rows of tape as one reel's tensors (a copy)"""
        tape = self._tape or []
        cat = lambda k: torch.cat([blk[k] for blk in tape])[:n].clone()   # noqa: E731
        return {"s": cat("s"), "v": cat("v"), "xo": cat("xo"), "a": cat("a"), "end": cat("end")}

    # ---------------- step R8a: the episodes ----------------
    def _episodes_nightfall(self, rep):
        """THE DAY CUT INTO EPISODES (the module's doc), as the night begins (after every event has ended): the day's reel, each
        episode's entry, T_e, window and credit; the day's episodes join the kept ones. Returns the day's episodes' indices among the
        kept"""
        n = int(getattr(self, "_tape_n", 0))
        if int(getattr(self, "_rec_n", 0)) != n:
            raise RuntimeError(f"the night over frames: the day's tape holds {n} rows and its record {int(getattr(self, '_rec_n', 0))}")
        reels = getattr(self, "_reels", None)
        if reels is None:
            reels = []; self._reels = reels
        eps = getattr(self, "_episodes", None)
        if eps is None:
            eps = []; self._episodes = eps
        mean = getattr(self, "_ep_mean", None)
        if mean is None:
            mean = [0.0, 0]; self._ep_mean = mean
        if n == 0:
            rep["episodes"] = {"day": 0, "kept_before": len(eps)}
            return []
        rec = self._rec[:n].to(torch.float64)
        g = self._tag_gamma(); reach = int(self.cfg.get("tag_reach", FRAMES["tag_reach"]))
        tags = rec[:, 2]
        tstar = tag_star_fast(tags, g, reach)
        cuts = sorted({int(e) for e in (getattr(self, "_rec_ends", None) or []) if 0 < int(e) < n} | {n})
        spans, a = [], 0
        for e in cuts:
            if e > a:
                spans.append((a, e)); a = e
        ent = episode_entries(spans, rec[:, 0].tolist(), rec[:, 1].tolist(), tstar.tolist(), mean, tau=int(self._sleep_const("entry_tau")))
        G = credit_after(rec[:, 1], g, reach)
        reel = dict(self._tape_rows(n), G=G, day=int(self.nights))
        W = int(self.m.window); floor = float(self._sleep_const("peak_floor"))
        ri = len(reels); reels.append(reel)
        day_idx = []; serial = int(getattr(self, "_ep_serial", 0))
        for (a_, b_), (entry, T_e) in zip(spans, ent):
            if T_e >= floor:
                p, pv = peak_of(tags, a_, b_, g, reach)
            else:
                p, pv = b_ - 1, None
            w0 = max(0, p - W + 1)
            serial += 1
            if p - w0 + 1 < 2:
                continue                                                 # a window of one tick teaches nothing: not kept
            eps.append({"serial": serial, "day": int(self.nights), "reel": ri, "w0": w0, "w1": p, "entry": float(entry), "T_e": float(T_e),
                        "a": a_, "b": b_, "peak": p})
            day_idx.append(len(eps) - 1)
        self._ep_serial = serial
        rep["episodes"] = {"day": len(day_idx), "spans": len(spans), "ticks": n, "kept_before": len(eps) - len(day_idx),
                           "tagged": sum(1 for i in day_idx if eps[i]["T_e"] >= 1.0),
                           "at_peak": sum(1 for i in day_idx if eps[i]["T_e"] >= floor),
                           "entry_mean": (round(sum(eps[i]["entry"] for i in day_idx) / len(day_idx), 4) if day_idx else None),
                           "T_e_max": (round(max(eps[i]["T_e"] for i in day_idx), 4) if day_idx else None)}
        return day_idx

    def _episodes_fade(self, rep):
        """AFTER THE NIGHT (the module's doc): every kept entry fades as the store's strengths do; the weakest give way until the kept
        windows hold at most episode_cap distinct ticks of tape; each reel compacted to the rows its kept windows read (the windows'
        rows remapped), a reel no window reads let go"""
        eps = getattr(self, "_episodes", None) or []
        reels = getattr(self, "_reels", None) or []
        f = float(self.cfg["store_fade"])
        for ep in eps:
            ep["entry"] = ep["entry"] * f
        cap = int(self._sleep_const("episode_cap"))
        refs = [torch.zeros(int(r_["v"].shape[0]), dtype=torch.long) for r_ in reels]
        for ep in eps:
            refs[ep["reel"]][ep["w0"]:ep["w1"] + 1] += 1
        held = sum(int((c > 0).sum()) for c in refs)
        order = sorted(range(len(eps)), key=lambda i: (eps[i]["entry"], i))   # the weakest first, of equals the oldest
        gone = set(); k = 0
        while held > cap and k < len(order):
            i = order[k]; k += 1; ep = eps[i]
            c = refs[ep["reel"]][ep["w0"]:ep["w1"] + 1]
            c -= 1
            held -= int((c == 0).sum())
            gone.add(i)
        kept = [ep for i, ep in enumerate(eps) if i not in gone]
        new_reels, remap = [], {}
        for ri, r_ in enumerate(reels):
            rows = torch.nonzero(refs[ri] > 0).flatten()
            if rows.numel() == 0:
                continue
            pos = torch.full((int(r_["v"].shape[0]),), -1, dtype=torch.long); pos[rows] = torch.arange(int(rows.numel()))
            remap[ri] = (len(new_reels), pos)
            new_reels.append({k_: (v_.index_select(0, rows) if torch.is_tensor(v_) else v_) for k_, v_ in r_.items()})
        for ep in kept:
            ni, pos = remap[ep["reel"]]
            ep["reel"] = ni; ep["w0"] = int(pos[ep["w0"]]); ep["w1"] = int(pos[ep["w1"]])
        self._episodes = kept; self._reels = new_reels
        rep.setdefault("episodes", {}).update({"kept": len(kept), "gave_way": len(gone), "rows": held, "reels": len(new_reels)})

    def _episode_rows(self, ep):
        """an episode's dream window: its reel's rows w0..w1 (views)"""
        r_ = self._reels[ep["reel"]]; s = slice(ep["w0"], ep["w1"] + 1)
        return {k_: r_[k_][s] for k_ in ("s", "v", "xo", "a", "end", "G")}

    def _tape_nightfall_reset(self):
        """the day's tape let go with the rest of the day's working state (the episodes keep what the nights read)"""
        self._tape = []; self._tape_n = 0

    def _episodes_back(self, saved):
        """THE EPISODES GIVEN BACK at a load (body/core/persistence.py; saved in the life, life["episodes"]): the reels, the episodes, the
        entries' running mean and the serial count, for a body whose night is over frames and whose reels are this anatomy's rows (its
        symbol and vector channels' widths and its motor effectors); else said once and not read"""
        sym, vec, mot = self._tape_layout()
        V = sum(int(c.size) for c in vec)
        reels = list(saved.get("reels") or [])
        fits = all(tuple(r_["v"].shape[1:]) == (V,) and tuple(r_["s"].shape[1:]) == (len(sym),) and tuple(r_["a"].shape[1:]) == (len(mot),)
                   for r_ in reels)
        if not self._night_frames_on() or not fits:
            print(f"load: the save holds {len(saved.get('episodes') or [])} episodes of the night over frames, "
                  f"{'which this body does not dream (night_frames 0)' if fits else 'taped by another anatomy'}: not read", flush=True)
            return
        self._reels = reels
        self._episodes = [dict(ep) for ep in (saved.get("episodes") or [])]
        m_ = saved.get("mean") or [0.0, 0]
        self._ep_mean = [float(m_[0]), int(m_[1])]
        self._ep_serial = int(saved.get("serial", 0))
