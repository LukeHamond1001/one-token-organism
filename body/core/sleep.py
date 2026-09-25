"""the night over frames (a mixin of `Life`, body/life.py; the core refactor's step R8, docs/SIM_DESIGN.md 3.6, 3.7, 5.4, 7.3, 7.4 item 2, 8's
R8 row, 9, 10; A46): the night of a body that lives in the world's frames (the sim), under physiology.py's SLEEP switches, which the
diary's cfg does not hold, so none of it runs for the language body.

STEP R8a, THE DAY'S TAPE AND THE EPISODES (7.4 item 2, 9's tape; the switch `night_frames`, which needs `frames`):
- THE TAPE (`_tape_tick`, at each awake tick's end, after R7b's record): the tick's window position as the cortex received it, the
  frame's "stored codes": every vector channel's numbers at fp16 (the G1's 3,833: the face, both cochleas, the periphery, the fovea, the
  body, touch, the inertial units, the charge), every symbol channel's symbol (the words'), whether the offset ended an utterance there
  (the waking lesson's end target; marked when the offset falls, `_tape_mark_end`), the voice's symbol (its own sound, "xo") and every
  motor effector's act (the efference copies the cortex heard: a rest where it did not act), and the ladder's bands as the cortex
  received them there (the position's bundle, fp16: THE LEAD'S DECISION of 2026-09-25, item 1 of the PFC-maturation study, below), in
  blocks of TAPE_BLOCK rows. A row is a tick, and the tick's record (R7b's surprise, dopamine, tag and net reward) is the same row of the
  day's record. About 7.8 KB a tick at the G1's sizes without the bands (7.7 KB of fp16, SIM_DESIGN.md 9), and the bands' 8 KB at d 512
  (8 clocks x 512 at fp16) beside them while the day lasts: the day's tape is the day's working state, saved with its day (A70).
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
  whole span, so the cap counts what a night can dream; about 184 MB at the cap. Of the day's bands each episode keeps only its
  window's first (8 KB at d 512): where its dreams begin.
- THE ENTRIES' RUNNING MEAN, the reels, the episodes and their serial count are the body's memory across nights: saved in its life
  (life["episodes"]) as the utterances heard are, not in its day (body/core/persistence.py).

STEP R8b, THE NIGHT OVER FRAMES (`_night_frames`, night()'s passes for a body under `night_frames`; the words' night is `_night_words`,
body/core/night.py, as it always was):
- THE DRAW (7.4 item 2): as many dreams as the words' night draws (a dream for each memory the day added, night_load, between
  night_starts and night_starts_max), over the kept episodes: every episode of the day with T_e >= 1 once, highest first, at most half
  the night, then the rest by entry (R7d's `night_draw`, on the body's own stream). HOW FRAMES JOIN THE DREAMS (R7's report asked R8 to
  decide it; THE BUILDER'S READING, for the lead, from 7.3, 7.4 items 2, 7, 8 and 12 and A46): the episodes are the sim's dreams, the
  words among their channels as the day heard them; the store's pattern-completion dreams of the words (and R7b's words-only copy of the
  store) are not drawn for a body in frames. 7.4 draws "the night's dreams" by the episodes' entries alone, "every dreamt position"
  teaches "the cortex's forecasts" (the words' among them), and the first design named the tape's episodes "the sim's utterance memory"
  ("the night draws episodes by tag, as the language night draws utterances"); a words' dream drawn beside them would take its share of
  the night by the store's strengths, which the tag never set. The store still recalls by day and fades each night.
- NREM (8's R8 row, "per-channel batches from stored codes"; `_frames_batch`, `_frames_lesson`): the dreams shuffled each round and run in
  lockstep batches of night_batch windows, each channel's observations from the reels, the ladder's bands run along each window (as the
  words' dreams run them) FROM THE DAY'S BAND STATES AT THE WINDOW'S START (THE LEAD'S DECISION of 2026-09-25, the PFC-maturation study
  (docs/audit/pfc_maturation.md, risk 3): the words' dreams start with the bands at zero, so a night from zero would teach act_pred and
  the cortex without their slow context, context-free habits, the A-not-B error's latent trace; of the two choices the lead left, the
  day's states as they stood at the window's first tick (taped, above) and the states the night keeps (night_keep_bands), THE BUILDER
  TOOK THE FIRST: each window is replayed in the slow context it was lived in, a morning's episode in the morning's), the cortex with the
  gradient over the windows as the day received them. It learns what the
  waking lesson teaches, at every dreamt position at weight 1: the words' forecast (their next symbol, the offset's end: the waking
  lesson's targets), every later channel's head (its next code, over its own running mean: err_scale), the forward half (through the
  stream, as by day); and every motor effector's acts replayed as efference copies (in the input) and as act_pred's targets (act_inv's
  labels at its reliability where it rested, as by day), act_pred's error at each position weighted clip(1 + G, 0, 1) by the replayed
  dopamine's credit (R7d's `act_pred_night_weight`: acts followed by net harm are not taught as acts to make) and read from the stream
  detached (`_timing_loss(night=True)`: a weighted target never reaches the stream, whose Adam would teach it whole). TWO OPTIMIZERS:
  the night's Adam (night_lr, night_beta2, night_warm, as the words' night) over every parameter but act_pred's, its correction's and
  the recall maps', its gradient bounded by its own norm; act_pred, its correction and the maps by their own plain step (opt_pred,
  GatedDescent at the day's constants: the builder's reading, `_night_frames`), once a batch, never the Adam (the night's note since R6
  fix 7). The recall maps take no gradient at night: the store's recall is not an input of a dream (the words' night's rule: the cortex
  must carry the sequence itself), so the tape keeps no recall. ACT_INV (`_frames_inverse`) is replayed over the windows' own acts, one
  step of its own optimizer a batch, its reliability left as it was (a replayed pair was learned by day: a fit, not a prediction).
- REM ON FRAMES (REM stays on: the owner's ruling; `_rem_frames`): rem_rounds rounds over the night's first rem_dreams dreams, each run
  free from its first positions (its bands from the day's at its start, as NREM's) (the words drawn at rem_temp, every other vector channel's code its head's mean forecast, every effector
  at rest), the prefrontal forecast heads learning along the free run, the stream held (as the words' REM, `_rem_rollout`).
- Then the value ladder's replay, the gauge after (the words' next symbols and the channels' scaled errors on the night's first
  windows, an instrument) and, as the words' night, only a non-finite lesson reloads the evening's organs.
STEP R8c, THE LIVE, DARK NIGHT (SIM_DESIGN.md 3.6, 3.7, 5.4, A46, C74; the switch `twitch`, which needs `night_frames`):
- THE NIGHT AT THE TICK'S END (C74; `tick_end`): a body under the night over frames sleeps at the end of the tick its sleep switch fired
  in (`_sleep_now` marks it due), after the world has applied that tick's acts and run the cerebellum's sub-steps, so its night's own
  save is written between ticks, where the core's save is exact; the world loop calls it (body/core/world.py), and a night still due
  runs at the next tick's start for a caller that lives ticks by hand. The diary's night stays inside its tick (its save path unchanged).
- THE WORLD'S NIGHT (World.live_night, dusk, dawn): a world that runs through the night is stepped dark by the night (`_live_night`),
  night_ticks ticks of it (24,000 for the sim: as long as the day, B19's compressed hour), after the night's replay; a world that does
  not is paused and resumed around the night, as the diary's (the stub worlds of the tests; nothing twitches then).
- EVERY EFFECTOR AT REST (its gates closed, its servos at rest under the resting law: the world's) but THE BRAINSTEM'S TWITCHES: no cord
  pattern (the pattern generator's amplitude is its gate's readiness, and the born cry answers a felt pain, which the sleeping body does
  not feel: nothing sensed at night reaches a reward, a critic, the amygdala or a gate, 5.4) and no VOR (the eyes are off): THE BUILDER'S
  READING, for the lead (5.4 names the withdrawal alone, which is the world's). THE BORN TWITCH GENERATOR (`_twitch_schedule`): in
  active sleep (each sleep cycle of sleep_cycle ticks, 47 minutes, its first sleep_active half: the newborn's sleep begins in active
  sleep) each tick twitches with probability twitch_rate (10 a minute of active sleep), one joint (uniform among every joint of the
  effectors that declare `twitch`: the G1's waist, arms, hands and legs, 43 joints) one small step (the setting beside its hold: 0.09
  rad on the G1's limbs), its sign drawn; from a generator seeded by the organs' twitch_seed and the night's number, so a night's
  twitches are a function of the body's seed and the night alone, and nothing of them is saved (a save falls between ticks, never
  inside a night). Each twitch is logged as reflex (the cord's counts), and the night's event lines (pain, touch) are counted, logged
  never felt. The cerebellum learns below each night tick through the world's hook, as by day: it learns wherever the world runs.
- THE PAIRS TEACH (`_twitch_lessons`, after the replay: the fixed order a night whose physics runs on another core merges in, SIM_DESIGN.md
  9's lever): act_inv on each twitch's pair (the sense before and after, the twitch its label), its reliability first with the chance the
  generator gave each setting (the kappa correction's law); the twitching limb's forward half from the stream at the twitch (the night's
  frame, the twitch its efference copy, the bands the night holds), the stream held.
- The live night's physics and the replay touch nothing of each other (the physics steps the world and the cerebellum with the born
  generator's acts; the replay teaches the cortex, act_pred and act_inv from the reels), so they may run on separate cores and merge
  before the twitch lessons in this fixed order (A64's lever): not built here, written down for R10."""
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
        so it needs `frames`; the twitches (R8c) are the night over frames' live night, and need joints that declare them and the organs'
        born generator's seed (a body is born with its switches: A20); each refused otherwise. The diary holds none of the keys, so
        nothing is refused"""
        if self._night_frames_on() and not self._frames_on():
            raise ValueError("Life: the night over frames (night_frames 1) cuts the frames' record into episodes and needs the frames (frames 1)")
        if int(self.cfg.get("twitch", SLEEP["twitch"])):
            if not self._night_frames_on():
                raise ValueError("Life: the twitches (twitch 1) are the live night of the night over frames and need it (night_frames 1)")
            if not self._twitch_joints() or "twitch_seed" not in self.m._buffers:
                raise ValueError("Life: the twitches are switched on (twitch 1) and no motor effector declares them (Effector.twitch), or the "
                                 "organs hold no born twitch generator (twitch_seed; built by Organs(..., effectors=anatomy.effectors))")

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
                "end": torch.zeros(TAPE_BLOCK, dtype=torch.bool),
                "bands": torch.zeros(TAPE_BLOCK, len(self.m.clocks), int(self.m.d), dtype=torch.float16)}

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
            blk["bands"][r] = w["bundle"].to("cpu", torch.float16)   # the ladder's bands the cortex received at this tick (the lead's item 1)
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

    def _tape_bands(self, row):
        """the ladder's bands at this row of the day's tape (fp16 [bands, d], a copy): the band states the cortex received at that tick"""
        b, r = divmod(int(row), TAPE_BLOCK)
        return self._tape[b]["bands"][r].clone()

    def _tape_rows(self, n):
        """the day's first n rows of tape as one reel's tensors (a copy; the bands apart: each episode keeps its window's first)"""
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
                        "a": a_, "b": b_, "peak": p, "bands0": self._tape_bands(w0)})
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

    # ---------------- step R8b: the night over frames ----------------
    def _night_frames(self, rep):
        """THE NIGHT OVER FRAMES (step R8b; the module's doc): the draw over the kept episodes (the day's tagged first), NREM on their
        windows in per-channel batches, REM on frames, the value ladder's replay, the gauge before and after, a non-finite lesson
        reloading the evening's organs (as the words' night). The report's fields go into `rep`. Returns the night's Adam and its
        parameters, which the live night's twitch lessons step after it (R8c). THE DAY'S PENDING PAIRS of act_inv (gathered before dusk)
        are learned first, at nightfall, before the night's lessons (the words' night learns them after its passes, R6h's place; for a body
        in frames the builder's reading, for the lead: they were gathered before the night, so they are learned before it)"""
        m = self.m
        for i_, st_ in enumerate(getattr(self, "motor", ()), 1):
            if st_.get("inv_batch"):
                self._inverse_batch(i_)                               # act_inv's pairs gathered before dusk, learned at nightfall (R6h)
        # THE NIGHT'S TWO OPTIMIZERS (8's R8 row; the night's note since R6 fix 7): the night's Adam (night_lr, night_beta2, warmed over
        # night_warm steps, as the words' night) over every parameter but act_pred's, its correction's and the recall maps', each step's
        # gradient bounded by its own norm over them; act_pred, its correction and the maps by their own plain step (GatedDescent,
        # opt_pred, at the day's constants: THE BUILDER'S READING, for the lead: "stepped by act_pred's plain step", the day's lr_motor
        # and bound; at night_lr its product with the lesson's curvature would pass 2 at d 512, where the day's has about 8 times headroom)
        gated = {id(p_) for e_ in self.anatomy.motors for p_ in self._gated_params(e_)}
        params = [p_ for p_ in m.parameters() if id(p_) not in gated]
        opt = torch.optim.Adam(params, lr=float(self.cfg["night_lr"]), betas=(0.9, float(self.cfg.get("night_beta2", 0.999))))
        eps = getattr(self, "_episodes", None) or []
        day = int(self.nights)
        # --- how many dreams: as the words' night, the day's new memories (frames and words) between night_starts and night_starts_max ---
        n_new = (self.store.n() - int(self._store_after_night)) if self._store_after_night is not None else 0
        n_new = max(n_new, int(getattr(self, "_writes_today", 0)))
        load = float(self.cfg.get("night_load", 0.0))
        n = int(self.cfg["night_starts"])
        if load > 0.0:
            n = int(min(int(self.cfg.get("night_starts_max", 192)), max(int(self.cfg["night_starts"]), round(load * max(0, n_new)))))
        firsts = [float(ep["T_e"]) if int(ep["day"]) == day else 0.0 for ep in eps]   # the day's tagged come first (7.4 item 2)
        draw = night_draw([float(ep["entry"]) for ep in eps], firsts, n, self.gen) if eps else []
        n_first = min(sum(1 for f_ in firsts if f_ >= 1.0), n // 2)   # night_draw's: every one once, at most half the night
        rep.update({"dreams": len(draw), "new_slots": int(n_new), "tagged_first": n_first,
                    "draw_serials": [int(eps[i]["serial"]) for i in draw],
                    "dreamt_days": sorted({int(eps[i]["day"]) for i in draw}),
                    "mean_len": (round(sum(eps[i]["w1"] - eps[i]["w0"] + 1 for i in draw) / len(draw), 1) if draw else 0)})
        if not draw:
            rep["note"] = "no episode to dream"
            return opt, params
        dreams = [eps[i] for i in draw]
        self._night_away()                                        # the night's lessons on night_dev (the draw made at home)
        g_dreams = dreams[:32]                                    # the gauge's windows: the night's first (the tagged first among them)
        before = self._frames_gauge(g_dreams)
        warm_ = int(self.cfg.get("night_warm", 0)); base_lr_ = float(self.cfg["night_lr"]); nstep_ = 0
        m.train()
        nb = max(1, int(self.cfg.get("night_batch", 0)))          # night_batch 0: a dream a batch (THE BUILDER'S READING: windows of frames
        losses = []; nrem = 0; wstats = [0, 0, 0]; inv_pairs = 0  # run in lockstep batches; SIM_CFG's 16)
        for _ in range(int(self.cfg["night_rounds"])):
            order = torch.randperm(len(dreams), generator=self.gen).tolist(); tot = 0.0; ok = 0
            for i0 in range(0, len(order), nb):
                opt.zero_grad(set_to_none=True); self.opt_pred.zero_grad(set_to_none=True)
                batch = [dreams[j] for j in order[i0:i0 + nb]]
                loss, parts = self._frames_lesson(batch)
                if loss is None or not bool(torch.isfinite(loss.detach())):
                    continue
                loss.backward()
                nstep_ += 1
                if warm_:
                    for g_ in opt.param_groups:
                        g_["lr"] = base_lr_ * min(1.0, nstep_ / warm_)
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if bool(torch.isfinite(gn)):
                    opt.step()
                    self.opt_pred.step()                          # act_pred, its correction (and the maps): the plain step, never the Adam
                opt.zero_grad(set_to_none=True); self.opt_pred.zero_grad(set_to_none=True)
                inv_pairs += self._frames_inverse(parts)          # act_inv replayed over the batch's own acts (its own optimizer)
                for k_ in range(3):
                    wstats[k_] += parts["w"][k_]
                tot += float(loss.detach()); ok += 1; nrem += 1
            losses.append(round(tot / max(1, ok), 4))
        mid = self._frames_gauge(g_dreams)
        # --- REM on frames (REM stays on: the owner's ruling): the cortex runs free from each dream's first positions, the words drawn from
        # its readout and every other channel's code its head's forecast, and the prefrontal forecast heads learn along the free run ---
        rem_cos = []; rem_steps = 0
        for _ in range(int(self.cfg["rem_rounds"])):
            opt.zero_grad(set_to_none=True); rc = []; k_n = 0
            use = dreams[:int(self.cfg["rem_dreams"])]
            for ep in use:
                fl, fc = self._rem_frames(ep)
                if fl is None or not bool(torch.isfinite(fl.detach())):
                    continue
                (fl / max(1, len(use))).backward(); rc.append(fc); k_n += 1
            if rc:
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if bool(torch.isfinite(gn)):
                    opt.step()
                rem_steps += 1; rem_cos.append(sum(rc) / len(rc))
            opt.zero_grad(set_to_none=True)
        self._night_home()                                        # home before the value replay, the gauge after, the fade and the save
        self._value_replay()
        m.eval()
        finite = all(bool(torch.isfinite(p_).all()) for p_ in m.parameters())
        after = self._frames_gauge(g_dreams)
        rep["discarded"] = not finite; rep["undone_for"] = "non-finite" if not finite else None
        if rep["discarded"] and self.save_path and __import__("os").path.exists(self.save_path):
            sd = torch.load(self.save_path, map_location="cpu", weights_only=False)
            m.load_state_dict(sd["organs"]); m.to(self.dev)
            after = self._frames_gauge(g_dreams)
        pos = max(1, wstats[0])
        rep.update({"nrem_steps": nrem, "nrem_loss": losses[:3] + (["..."] if len(losses) > 6 else []) + losses[-3:],
                    "nrem_curve": [round(float(x), 4) for x in losses], "rem_steps": rem_steps,
                    "rem_cos": (round(rem_cos[-1], 3) if rem_cos else None), "rem_cos_first": (round(rem_cos[0], 3) if rem_cos else None),
                    "act_pred_weight": {"positions": wstats[0], "below_1": round(wstats[1] / pos, 4), "zero": round(wstats[2] / pos, 4)},
                    "act_inv_pairs": inv_pairs,
                    "gauge": {"before": before, "after_nrem": mid, "after": after}})
        return opt, params

    def _frames_batch(self, batch):
        """A NIGHT'S BATCH OF WINDOWS (step R8b; "per-channel batches from stored codes"): each channel's observations [B, T, ..] from the
        reels (a vector channel's fp16 numbers as float32), the voice's symbol and every motor effector's act [B, T], right-padded
        (a symbol channel's rest, a vector channel's zeros, every effector's rest: a causal cortex never sees the padding after a
        position); the ladder's bands run along each window from fresh (no gradient), as the words' lockstep dreams run them; the end
        marks, each row's credit and the mask of the real positions [B, T]; each window's length"""
        m = self.m; dev = self.dev
        sym, vec, mot = self._tape_layout()
        rows = [self._episode_rows(ep) for ep in batch]
        B = len(rows); lens = [int(r_["v"].shape[0]) for r_ in rows]; T = max(lens)
        obs = {}
        for j, c in enumerate(sym):
            x = torch.full((B, T), int(c.rest_id), dtype=torch.long)
            for b, r_ in enumerate(rows):
                x[b, :lens[b]] = r_["s"][:, j]
            obs[c.name] = x.to(dev)
        off = 0
        for c in vec:
            x = torch.zeros(B, T, int(c.size))
            for b, r_ in enumerate(rows):
                x[b, :lens[b]] = r_["v"][:, off:off + int(c.size)].float()
            obs[c.name] = x.to(dev); off += int(c.size)
        for j, e in enumerate(mot):
            x = torch.full((B, T), int(e.rest_id), dtype=torch.long)
            for b, r_ in enumerate(rows):
                x[b, :lens[b]] = r_["a"][:, j]
            obs[e.name] = x.to(dev)
        xos = torch.full((B, T), int(self.sil), dtype=torch.long)
        ends = torch.zeros(B, T, dtype=torch.bool); G = torch.zeros(B, T, dtype=torch.float64); mask = torch.zeros(B, T)
        for b, r_ in enumerate(rows):
            xos[b, :lens[b]] = r_["xo"]; ends[b, :lens[b]] = r_["end"]; G[b, :lens[b]] = r_["G"]; mask[b, :lens[b]] = 1.0
        xos = xos.to(dev)
        nbands = len(m.clocks)
        bundles = torch.zeros(B, T, nbands, m.d, device=dev)
        bands = torch.stack([ep["bands0"].float() for ep in batch]).to(dev)   # each window from the day's bands at its start (the lead's item 1)
        with torch.no_grad():
            cache = [None] * len(m.blocks)
            for t in range(T):
                bundles[:, t] = bands
                if t + 1 < T:
                    C = m.stream_step(m.inputs(self.anatomy, {k_: v_[:, t] for k_, v_ in obs.items()}, xos[:, t], bundles[:, t]), cache)
                    bands = m.band_update_b(bands, C)
        return obs, xos, bundles, ends, G, mask.to(dev), lens

    def _word_targets(self, xs, ends, lens):
        """THE WORDS' TARGETS OF A NIGHT'S WINDOWS, the waking lesson's (body/core/cortex.py `_wake_lesson`): at each position the world's
        next symbol within the window (weight 1; none where no symbol follows), the end of an utterance where the offset marked it (its
        end symbol, weight 1); the padding none. `xs` [B, T] the words, `ends` [B, T]"""
        B, T = xs.shape
        y = torch.full((B, T), int(self.sil), dtype=torch.long); w = torch.zeros(B, T)
        xl = xs.tolist(); el = ends.tolist()
        for b in range(B):
            last = -1; L_ = lens[b]
            nxt = [-1] * L_
            for t in range(L_ - 1, -1, -1):
                nxt[t] = last
                if int(xl[b][t]) != int(self.sil):
                    last = t
            for t in range(L_):
                y[b, t] = int(xl[b][max(0, nxt[t])])
                w[b, t] = 1.0 if nxt[t] >= 0 else 0.0
                if el[b][t]:
                    y[b, t] = int(self.end_id); w[b, t] = 1.0
        return y.to(xs.device), w.to(xs.device)

    def _frames_lesson(self, batch):
        """ONE NREM LESSON ON A BATCH OF WINDOWS (step R8b; 7.4 item 2's "what the replay teaches"): the cortex over the windows as the
        day received them (with the gradient); the words' forecast its next symbol and every later channel's head its next code, each
        channel's error over its own running mean as by day (err_scale); every motor effector's timing lesson on each window (the day's
        `_timing_loss`: its acts the efference copies in the input and act_pred's targets, act_inv's labels at its reliability where it
        rested, the forward half at weight 1 through the stream), act_pred's error at each position weighted clip(1 + G, 0, 1) by the
        replayed dopamine's credit and read from the stream detached (the night form); each window weighs alike. Returns (the loss, the
        parts the rest of the night reads: the batch, the act weights' counts)"""
        m = self.m
        obs, xos, bundles, ends, G, mask, lens = self._frames_batch(batch)
        C = m.stream(m.inputs(self.anatomy, obs, xos, bundles))
        words = self.anatomy.words
        y, w = self._word_targets(obs[words.name], ends, lens)
        ll, _ = m.latent_loss(m.latent_pred(C), y, w=w)
        es = self._err_scales()
        pair = mask[:, 1:]
        npair = float(pair.sum())
        for i_, c_ in enumerate(self.anatomy.channels):
            if i_ and c_.forecast and npair > 0:
                with torch.no_grad():
                    tgt = c_.encode(m, obs[c_.name][:, 1:])
                lc_ = (0.5 * ((m.head(self.anatomy, i_)(C[:, :-1]).float() - tgt.float()) ** 2).sum(-1) * pair).sum() / npair
                if es and c_.name in es:
                    lc_ = lc_ / es[c_.name]
                ll = ll + lc_
        W_ = (1.0 + G).clamp(0.0, 1.0)                            # act_pred_night_weight, position by position
        wst = [0, 0, 0]
        for b, L_ in enumerate(lens):
            if L_ < 2:
                continue
            wb = W_[b, :L_]
            wst[0] += L_ - 1; wst[1] += int((wb[1:] < 1.0).sum()); wst[2] += int((wb[1:] == 0.0).sum())
        for i_ in range(1, len(self.anatomy.motors) + 1):
            tot = None
            for b, L_ in enumerate(lens):
                if L_ < 2:
                    continue
                ob = {k_: v_[b, :L_] for k_, v_ in obs.items()}
                lt, rt, lb = self._timing_loss(i_, C[b, :L_], ob, wpos=W_[b, :L_], night=True)
                if lt is None:
                    continue
                if lb is not None and rt["rel"] > 0.0:
                    lt = lt + rt["rel"] * lb
                tot = lt if tot is None else tot + lt
            if tot is not None:
                ll = ll + tot / float(len(lens))
        return ll, {"obs": obs, "lens": lens, "w": wst}

    def _frames_inverse(self, parts):
        """ACT_INV REPLAYED (8's R8 row: "act_inv and the forward half replayed over the day's transitions"): for each motor effector with an
        inverse model, the batch's own acts' transitions (the sense at t and t + 1, the act at t where it acted, both inside the window),
        one step of its own optimizer on the mean of the joints' summed cross-entropy (the day's batched lesson), its reliability left as
        it is (a replayed pair was learned by day: a fit, not a prediction). Returns the pairs taught"""
        obs, lens = parts["obs"], parts["lens"]; n_ = 0
        for i_, e in enumerate(self.anatomy.motors, 1):
            if not e.inverse:
                continue
            S0, S1, A = [], [], []
            for b, L_ in enumerate(lens):
                if L_ < 2:
                    continue
                s = self._window_sense(e, {e.sense: obs[e.sense][b, :L_]})
                a = obs[e.name][b, :L_]
                idx = torch.nonzero(a[:-1] != int(e.rest_id)).flatten()
                if idx.numel():
                    S0.append(s[idx]); S1.append(s[idx + 1]); A.append(a[idx])
            if not S0:
                continue
            S0 = torch.cat(S0); S1 = torch.cat(S1); A = torch.cat(A)
            tm = self.m.timing[e.name]; tab = self.m.get_submodule(e.organ)
            tt = tab.digits(A.cpu())
            with torch.enable_grad():
                self.opt_inv.zero_grad(set_to_none=True)
                loss = None
                for j_, l_ in enumerate(tm.inverse_logits(S0, S1)):
                    ce = torch.nn.functional.cross_entropy(l_, tt[:, j_].to(l_.device))
                    loss = ce if loss is None else loss + ce
                loss.backward()
                self.opt_inv.step()
                self.opt_inv.zero_grad(set_to_none=True)
            n_ += int(A.numel())
        return n_

    def _frames_gauge(self, dreams):
        """THE FRAMES' GAUGE (an instrument, as the words' night's): on the night's first windows, with no gradient, the share of the
        words' next symbols the cortex forecasts itself (argmax, where a target is owed) and the mean over the later channels of each
        head's error over its running mean"""
        if not dreams:
            return None
        m = self.m; hits = 0.0; nw = 0.0; errs = []; es = self._err_scales() or {}
        nb = max(1, int(self.cfg.get("night_batch", 0)))
        bans = [b_ for b_ in self.bans if b_ != self.eot]
        with torch.no_grad():
            for i0 in range(0, len(dreams), nb):
                obs, xos, bundles, ends, G, mask, lens = self._frames_batch(dreams[i0:i0 + nb])
                C = m.stream(m.inputs(self.anatomy, obs, xos, bundles))
                y, w = self._word_targets(obs[self.anatomy.words.name], ends, lens)
                lg = m.readout(m.latent_pred(C)); lg[..., bans] = float("-inf")
                hits += float(((lg.argmax(-1) == y).float() * w).sum()); nw += float(w.sum())
                pair = mask[:, 1:]; npair = float(pair.sum())
                for i_, c_ in enumerate(self.anatomy.channels):
                    if i_ and c_.forecast and npair > 0:
                        tgt = c_.encode(m, obs[c_.name][:, 1:])
                        e_ = float((0.5 * ((m.head(self.anatomy, i_)(C[:, :-1]).float() - tgt.float()) ** 2).sum(-1) * pair).sum() / npair)
                        errs.append(e_ / es[c_.name] if c_.name in es else e_)
        return {"words": (round(hits / nw, 3) if nw else None), "channels": (round(sum(errs) / len(errs), 4) if errs else None)}

    def _rem_frames(self, ep, k=3):
        """REM ON FRAMES (step R8b; REM stays on: the owner's ruling; the first design's "REM samples discrete channels, takes the mean
        forecast for vectors"): from the window's first k positions as the day received them, the cortex runs free for rem_steps
        positions: the words drawn from its readout at the REM temperature (the rest among them: on frames a position is a tick, and the
        words' rest is their usual state; THE BUILDER'S READING, for the lead: the words' dreams draw a symbol at every position, as a
        dream of words has no quiet), every later vector channel's code its head's forecast (the mean forecast), every effector at rest
        and no sound of its own (the body is still in REM; its twitches are the world's, R8c). The prefrontal heads learn along the free
        run, the stream held (as the words' REM: `_rem_rollout`). Returns (loss, cos) or (None, None)"""
        m = self.m; dev = self.dev; L = int(self.cfg["rem_steps"])
        r_ = self._episode_rows(ep)
        T0 = min(int(k), int(r_["v"].shape[0]))
        if T0 < 1:
            return None, None
        sym, vec, mot = self._tape_layout()
        words = self.anatomy.words
        idx = {c_.name: i_ for i_, c_ in enumerate(self.anatomy.channels)}
        syms = {c_.name: [int(r_["s"][t, j]) for t in range(T0)] for j, c_ in enumerate(sym)}
        codes = {}; off = 0
        with torch.no_grad():
            for c_ in vec:
                codes[c_.name] = [c_.encode(m, r_["v"][t, off:off + int(c_.size)].float().to(dev)) for t in range(T0)]
                off += int(c_.size)
        bands = ep["bands0"].float().to(dev); bundles, Cs, bnext = [], [], []   # from the day's bands at the window's start (the lead's item 1)
        bans = list(self.bans)
        with torch.no_grad():
            for step in range(T0 + L):
                if step >= T0:
                    lg = m.readout(m.latent_pred(Cs[-1])).clone(); lg[bans] = float("-inf")
                    rt = self._rem_temperature()
                    syms[words.name].append(int(torch.multinomial(torch.softmax(lg / rt, 0).cpu(), 1, generator=self.gen)) if rt > 0 else int(lg.argmax()))
                    for c_ in sym:
                        if c_.name != words.name:
                            syms[c_.name].append(int(c_.rest_id))
                    for c_ in vec:
                        codes[c_.name].append(m.head(self.anatomy, idx[c_.name])(Cs[-1]) if c_.forecast else torch.zeros(m.d, device=dev))
                bundles.append(bands.clone())
                n = step + 1
                ob = {c_.name: torch.tensor(syms[c_.name][:n], dtype=torch.long, device=dev) for c_ in sym}
                for e in mot:
                    ob[e.name] = torch.full((n,), int(e.rest_id), dtype=torch.long, device=dev)
                u = m.inputs(self.anatomy, ob, torch.full((n,), int(self.sil), dtype=torch.long, device=dev), torch.stack(bundles),
                             codes={c_.name: torch.stack(codes[c_.name][:n]) for c_ in vec})
                C = m.stream(u)[-1]; Cs.append(C)
                bands = m.band_update(bands, C)
                bnext.append(bands.clone())
        C_free = torch.stack(Cs[T0 - 1:-1]); B_next = torch.stack(bnext[T0 - 1:-1])
        if C_free.shape[0] < 2:
            return None, None
        return m.forecast_loss(C_free, B_next, sig=0.0)

    # ---------------- step R8c: the live, dark night ----------------
    def _twitch_on(self):
        """the switch `twitch` (step R8c): 0 for the diary, whose cfg holds no such key"""
        return bool(int(self.cfg.get("twitch", SLEEP["twitch"])))

    def tick_end(self):
        """THE TICK'S END (step R8c; SIM_DESIGN.md C74), called by the world loop after the world has applied the tick's acts and run the
        cerebellum's sub-steps, between ticks: a body in frames whose sleep switch fired in the tick sleeps here (`_sleep_now` defers it),
        so its night's own save is written between ticks, where the core's save is exact; a world that runs through the night
        (World.live_night) is stepped dark through it by the night (`_live_night`), any other paused and resumed around it. The diary's
        night stays inside its tick, so nothing is due here for it. A caller that lives ticks by hand without calling it meets a night
        still due at the next tick's start (`tick`), before that tick's frame: the same place between ticks"""
        if not getattr(self, "_night_due", False):
            return
        self._night_due = False
        live = bool(getattr(self.world, "live_night", False))
        if not live:
            self.world.pause()
        try:
            self.night()
        finally:
            if not live:
                self.world.resume()

    def _sleep_cycle_active(self, t):
        """whether the night's tick t is in active sleep (the module's doc): its place in its sleep cycle (sleep_cycle ticks, active sleep
        first) below sleep_active of it"""
        cyc = float(self._sleep_const("sleep_cycle"))
        return (float(t) % cyc) < float(self._sleep_const("sleep_active")) * cyc

    def _twitch_joints(self):
        """every joint that twitches: (its motor effector's place among the motor effectors plus one, the joint), in the declared order"""
        return [(i, j) for i, e in enumerate(self.anatomy.motors, 1) if getattr(e, "twitch", False) for j in range(len(e.factors))]

    def _twitch_schedule(self, N):
        """THE BORN TWITCH GENERATOR'S NIGHT (the module's doc): {night tick: ((effector place, joint), sign 0 or 1)} over the night's N ticks,
        from a generator of its own seeded by the organs' twitch_seed and the night's number (a function of the body's seed and the night
        alone): each tick of active sleep twitches with probability twitch_rate, the joint uniform among every twitching joint, the sign
        uniform"""
        joints = self._twitch_joints()
        if not joints or N <= 0:
            return {}
        g = torch.Generator().manual_seed(int(self.m.twitch_seed) + 1000003 * int(self.nights))
        u = torch.rand(N, generator=g, dtype=torch.float64)
        pick = torch.randint(len(joints), (N,), generator=g)
        sign = torch.randint(2, (N,), generator=g)
        p = float(self._sleep_const("twitch_rate"))
        return {t: (joints[int(pick[t])], int(sign[t])) for t in range(N) if float(u[t]) < p and self._sleep_cycle_active(t)}

    def _twitch_act(self, e, j, sign):
        """a twitch's act: joint j of effector e one small step (the setting beside its hold: below it for sign 0, above for 1), every other
        joint holding; (the flat act, its settings)"""
        dig = [(int(k_) - 1) // 2 for k_ in e.factors]
        dig[int(j)] += 1 if int(sign) else -1
        return int(self.m.get_submodule(e.organ).flat(dig)), dig

    def _twitch_chance(self, e):
        """THE CHANCE THE TWITCH GENERATOR GAVE EACH SETTING (act_inv's kappa correction on a twitch's pair, as `_act_chance` gives a drawn
        act's): given that the twitch fell on effector e, per joint: its hold 1 - 1/J, each small step 1/(2J), every other setting 0"""
        J = len(e.factors); out = []
        for K in e.factors:
            q = torch.zeros(int(K)); h = (int(K) - 1) // 2
            q[h] = 1.0 - 1.0 / J; q[h - 1] = 0.5 / J; q[h + 1] = 0.5 / J
            out.append(q)
        return out

    def _frame_sense(self, e, frame):
        """effector e's body sense in a night's frame (float32 [sense_n]; its channel's zeros where the frame names it not)"""
        c = self.anatomy.channel(e.sense)
        o_ = frame.obs.get(c.name)
        x = torch.zeros(int(c.size)) if o_ is None else torch.as_tensor(o_, dtype=torch.float32).reshape(int(c.size))
        if e.sense_idx is not None:
            x = x[torch.tensor([int(k_) for k_ in e.sense_idx])]
        return x.to(self.dev)

    def _live_night(self, rep, opt, params):
        """THE LIVE, DARK NIGHT (step R8c; the module's doc; SIM_DESIGN.md 5.4, A46): dusk; night_ticks ticks of the world stepped with every
        effector at rest (no cord pattern: the gates are closed in sleep, and the pattern generator's amplitude is its gate's readiness;
        no VOR: the eyes are off) but the born generator's twitches in active sleep; each twitch's pair kept (the frame before, its act,
        the frame after), the night's event lines counted (pain and falls at night are logged, never felt: nothing sensed at night reaches
        a reward, a critic, the amygdala or a gate); dawn; then the pairs' lessons (`_twitch_lessons`). The cerebellum learns below each
        night tick through the world's hook, as by day"""
        from .frames import read_event_lines
        from .world import Acts
        m = self.m; w = self.world; N = int(self.cfg.get("night_ticks", 0))
        mot = list(self.anatomy.motors)
        sched = self._twitch_schedule(N) if self._twitch_on() else {}
        rest = {e_.name: int(e_.rest_id) for e_ in self.anatomy.effectors}
        ev = list(self.anatomy.events or ()); counts = {ln.name: 0 for ln in ev}
        cb0 = int(m.cereb.n_sub) if "cereb" in m._modules else None
        pairs = []; pending = None; by = {}; active = 0
        w.dusk()
        try:
            for t in range(N):
                f = w.frame()
                if pending is not None:
                    pairs.append(pending + (f,)); pending = None
                if ev:
                    for ln, v in zip(ev, read_event_lines(ev, f.obs)):
                        counts[ln.name] += int(v > 0.0)
                active += int(self._sleep_cycle_active(t))
                acts = Acts(rest)
                tw = sched.get(t)
                if tw is not None:
                    (i, j), sg = tw; e = mot[i - 1]
                    a, dig = self._twitch_act(e, j, sg)
                    acts[e.name] = a
                    pending = (i, a, j, f)
                    cn = self.motor[i - 1]["cord_n"]; cn["twitch"] = int(cn.get("twitch", 0)) + 1   # logged as reflex (3.7)
                    by[e.name] = by.get(e.name, 0) + 1
                w.apply(acts)
            if pending is not None:
                pairs.append(pending + (w.frame(),))
        finally:
            w.dawn()
        les = self._twitch_lessons(pairs, opt, params)
        rep["live"] = {"ticks": N, "active": active, "twitches": len(pairs), "by": by, "events": {k_: v_ for k_, v_ in counts.items() if v_},
                       "cereb_sub": (int(m.cereb.n_sub) - cb0 if cb0 is not None else None), **les}

    def _twitch_lessons(self, pairs, opt, params):
        """THE TWITCHES TEACH (step R8c; SIM_DESIGN.md 3.6, A46: "each twitch and its reafference is a clean single-joint pair for act_inv
        and the forward half"), after the night's replay (the fixed order a parallel night merges in: SIM_DESIGN.md 9, A64): ACT_INV on each
        pair in the order they came (the sense before and after, the twitch's act its label), its reliability updated first on its label
        (a prediction, not a fit) with the chance the generator gave each setting, one step of its own optimizer every act_inv_every pairs
        (its day's batched lesson, `_inverse_batch`); THE FORWARD HALF of the twitching effector, from the stream at the twitch's tick (the
        night's frame, the twitch its efference copy; the ladder's bands as the night holds them: THE BUILDER'S READING, for the lead: a
        night's frame was never lived awake, so its bands are the evening's, the states the night keeps) to its sense at the next tick, the
        stream held (the cortex does not learn from the night's frames: 5.4 names act_inv, the forward half and the cerebellum), one step of
        the night's Adam a batch of night_batch pairs. The cerebellum learned from them below the tick as the world ran"""
        mot = list(self.anatomy.motors); m = self.m
        if not pairs:
            return {"act_inv_twitch": 0, "fwd_steps": 0}
        every = max(1, int(self._motor_const("act_inv_every")))
        n_inv = 0
        for i, e in enumerate(mot, 1):
            if not e.inverse:
                continue
            st = self.motor[i - 1]; mine = [p_ for p_ in pairs if p_[0] == i]
            ch = self._twitch_chance(e)
            for k0 in range(0, len(mine), every):
                keep = st["inv_batch"]
                st["inv_batch"] = [(self._frame_sense(e, p_[3]), self._frame_sense(e, p_[4]), int(p_[1]), [q.clone() for q in ch])
                                   for p_ in mine[k0:k0 + every]]
                self._inverse_batch(i)
                st["inv_batch"] = keep
                n_inv += len(mine[k0:k0 + every])
        sym, vec, _ = self._tape_layout()
        nb = max(1, int(self.cfg.get("night_batch", 0))); steps = 0
        for k0 in range(0, len(pairs), nb):
            chunk = pairs[k0:k0 + nb]; B = len(chunk)
            obs = {}
            for c_ in sym:
                obs[c_.name] = torch.tensor([[int(c_.rest_id) if p_[3].obs.get(c_.name) is None else int(p_[3].obs[c_.name])] for p_ in chunk],
                                            dtype=torch.long, device=self.dev)
            for c_ in vec:
                obs[c_.name] = torch.stack([(torch.zeros(int(c_.size)) if p_[3].obs.get(c_.name) is None else
                                             torch.as_tensor(p_[3].obs[c_.name], dtype=torch.float32).reshape(int(c_.size))) for p_ in chunk]
                                           ).unsqueeze(1).to(self.dev)
            for i, e in enumerate(mot, 1):
                obs[e.name] = torch.tensor([[int(p_[1]) if p_[0] == i else int(e.rest_id)] for p_ in chunk], dtype=torch.long, device=self.dev)
            xos = torch.full((B, 1), int(self.sil), dtype=torch.long, device=self.dev)
            bundles = self.bands.detach().to(self.dev).unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).clone()
            with torch.no_grad():
                C0 = m.stream(m.inputs(self.anatomy, obs, xos, bundles))[:, 0]
            loss = None
            for b, p_ in enumerate(chunk):
                e = mot[p_[0] - 1]
                if e.sense is None:
                    continue
                s1 = self._frame_sense(e, p_[4])
                le = 0.5 * ((m.timing[e.name].fwd(C0[b]).float() - s1.float()) ** 2).sum()
                loss = le if loss is None else loss + le
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            (loss / float(B)).backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if bool(torch.isfinite(gn)):
                opt.step()
            opt.zero_grad(set_to_none=True); steps += 1
        return {"act_inv_twitch": n_inv, "fwd_steps": steps}

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
