"""the second body alive (BODY_SPEC.md §1, §3, §4): ticks, feelings, the one reward system,
the waking lesson, sleep by fatigue, the night. The caregiver reaches it only through THE DIARY
protocol in serve.py; nothing here reads the caregiver's mind or edits the body's words."""
import collections
import math
import os
import time

import torch
import torch.nn.functional as F

from .model import Organs, Store, CLOCKS

PHYSIOLOGY = dict(
    symbol_cost=0.12, fatigue_half_life=240, stress_half_life=240, mood_half_life=1200,   # in ticks: the body lives on its clock
    wake_ticks=12000, elig_ticks=12, elig_decay=0.8, store_fade=0.9, store_floor_rel=0.1, store_temp=0.02, heard_decay=0.999,
    bag_decay=0.8, bag_own_weight=1.0, night_lr=1e-4, night_rounds=24, night_starts=48, rem_steps=8, rem_dreams=8, rem_rounds=6, sigreg=0.0,
    dream_max=24, dream_floor_rel=0.5, end_rest=0, cost_in_reward=0, gate_slow_lr=0.0, dream_adapt=0.2, dream_recover=0.97, dream_exhaust=0.1, gate_baseline=0.9, wake_every=24, wake_window=32, live_lr=1e-5, value_lr=1e-3, band_lr=1e-5, face_lr=1e-3,
    gate_lr=0.05, birth_act=0.25, gate_habit=0.9, gate_fatigue=10.0, gate_int=0.5, gate_tonic=0.25, gate_vigor=1.0, gate_every=24,
    gate_salience=0.0,    # the forecast's certainty as an input of the gate (the proposal's salience); 0 until measured (run 30)
    # THE MOTIVATIONAL DOPAMINE: the gate's credit may carry the error of a slow band too (ventral striatal dopamine, the long
    # horizons of the discount gradient, driving vigor; the fast band's error selects the act). gate_slow_w 0 = off (runs 37/38)
    gate_slow_band=5, gate_slow_w=0.0,
    # THE VENTRAL CRITIC in the mouth's credit: its error (the act's effect on the long-run prospect) added to the fast
    # band's error with weight vcrit_w. Its horizon is definite, discounted at vcrit_gamma (1024 ticks): as a differential
    # (average-reward) head over fast features it computed the day-scale relative value, swinging by a hundred within a
    # day (runs 49/50: the integral of reward above its wandering average), and that drift entered the mouth's twelve-tick
    # credit ten times the size of the fast error and shut one seed's gate. 0 = off until measured (runs 51/52; candidate 1.0)
    vcrit_w=1.0, vcrit_gamma=1.0 - 1.0 / 1024,
    # THE CRITIC'S ELIGIBILITY TRACE: TD(0) bootstrapped over a thousand steps sits at a fixed point whose error the
    # horizon amplifies (Tsitsiklis and Van Roy: by (1 - lambda gamma) / (1 - gamma), a thousandfold at lambda 0), and
    # run 67's ventral critic read the return that followed at -0.28 over four days while the pinned 4096-tick band
    # read +0.86. TD(lambda) with the trace decaying at the critic's own horizon (lambda = gamma, the factor near 2)
    # is the backward view of the discounted return itself: a trace of the critic's inputs on its weights, captured
    # by the error as it arrives (the synaptic tag on the critic's side). 0 = TD(0); candidate 1 - 1/1024 (the night
    # clears the trace)
    vcrit_lambda=0.0,
    # THE LEVEL (Pavlovian-instrumental transfer): the gate reads the slow band's value, the state's long-run promise, through
    # a divisive normalization by that value's own running scale (semi-saturation 1), and its own three-factor lesson sets
    # the weight. A cue that promises reward invigorates the act (general PIT: the amygdala's Pavlovian value onto the
    # striatum's vigor). gate_level_w 0 = off until measured (runs 39/40)
    gate_level_band=5, gate_level_w=0.0,
    # THE OFFSET: the world's quiet after its utterance is an event. After offset_ticks of the world's quiet, once per
    # pause, the line's end (the turn-end, <eot_human>, the tokenizer's end of the human's turn) becomes the waking
    # lesson's target for the line's last symbol (the cortex learns "then quiet" instead of the next line's first
    # letter); the store's slot for that symbol carries a boundary mark and the first memory kept after a pause a start
    # mark: dreams run from a start and end at a marked memory (the cortex's own expectation of the quiet ended a young
    # body's dreams at two symbols and is not used); the mouth may never say it. Nothing enters the stream (as a
    # stream symbol it wiped the body's own context mid-answer: runs 43/44, day 1, "go n Z") and the store does not
    # hold it (written there, the quiet after a cue blended with the answer and the mouth read junk: runs 45/46).
    # 0 = off (before it, the cortex learned the seam between utterances: after "dog will go down" the next line's
    # first letter at probability 1, the mouth's "downg"; served body, day 10)
    offset_ticks=8,       # THE RECIPE (runs 47/48): two seconds at four ticks a second; within a line the world types a symbol a
    # tick, and 8 + the lesson's cadence of 24 keeps the ended position inside the lesson's 32. 0 = off
    # THE INTRINSIC CREDIT: "value" = the forecast's belief in what it said x novelty habituating by repetition (the recipe; with
    # gate_tonic 0.25). "error" = belief minus that syllable's usual belief (the songbird's performance error, Gadagkar 2016) with
    # gate_tonic 0.70 (the mean the value form gives a grown body): run 31 matched the value form's seeds at days 6 and 15 and
    # fell to 39/48 at day 20 on one cue's stutter; a second seed (run 34) decides. Not the recipe until it does.
    gate_int_form="value",
    gate_floor=0.05,      # spontaneous activity never stops: p(act) = floor + (1 - floor) sigmoid(z); no absorbing silence
    read_sharp=25.0, sharp_base=25.0, sharp_gain=25.0, burst=0.5, mood_gain=0.25, stress_gain=0.5, v_buf=32,
    dopamine_band=2,      # the band whose TD error is dopamine: clock 16, discount 0.9375 per tick (a four-second horizon)
    diff_horizon=1024,    # bands with clocks at or above this learn average-reward TD (no discount, the reward rate as baseline)
)


class Life:
    def __init__(self, organs, tok, cfg=None, device="cpu", seed=0, save_path=None):
        self.m = organs.to(device); self.m.eval()
        self.tok = tok; self.dev = device
        self.cfg = dict(PHYSIOLOGY); self.cfg.update(cfg or {})
        self.m.read_sharp = float(self.cfg["read_sharp"])
        self.sil = tok.token_to_id("<pad>")
        self.m.sil_id = self.sil                          # the cortex's inputs know its rest
        self.nl = tok.token_to_id("\n")
        self.eot = tok.token_to_id("<eot_human>")         # the world's turn ended: the offset (§2), never the mouth's
        self.bans = [i for i in range(11) if i != self.sil] + ([self.nl] if self.nl is not None else [])
        self._last_world = -10 ** 9; self._offset_done = True; self._last_write = None; self._start_pending = False
        self.store = Store(self.m.d, temp=float(self.cfg["store_temp"]), device=device)
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.save_path = save_path
        nb, d, W = len(self.m.clocks), self.m.d, self.m.window
        # working state
        self.bands = torch.zeros(nb, d, device=device)
        # THE CONTEXT: two bags, the world's (decaying per world symbol) and its own (per own symbol),
        # summed into the memory key; a pause moves neither. Decayed per tick, the babble between the
        # world's symbols shifted the world's weights in every key (collision test at own weight 0.6).
        self.bag_w = torch.zeros(d, device=device)
        self.bag_o = torch.zeros(d, device=device); self.n_own = 0
        self.win = collections.deque(maxlen=W)            # per tick: dict(x the world's, xo its own, face, bundle, read, r)
        self.pred_prev = None                              # the forecast made at the last step (surprise)
        self.v_prev = None                                 # V_b of the previous tick's states
        self.v_buf = {b: collections.deque(maxlen=int(self.cfg["v_buf"])) for b in range(nb)}
        # THE REWARD RATE, one estimate: a running mean of the reward at the differential horizon (tonic
        # dopamine), the baseline of every differential band. Estimated per band at the band's own clock,
        # the slowest band's baseline (one part in 16384 a tick) could not track the rate within a day and
        # its undiscounted value integrated raw reward (run 27, day 6: 643 against a return of 131).
        self.rbar = 0.0
        self._differential = [int(c) >= int(self.cfg["diff_horizon"]) for c in self.m.clocks]
        with torch.no_grad():
            self.m.diff[:] = torch.tensor(self._differential, dtype=torch.bool, device=self.m.diff.device)
        # feelings and clocks
        self.fatigue = 0.0; self.stress = 0.0; self.mood = 0.0
        self._t_feel = time.time()
        self.sleep_pressure = 0; self.ticks = 0; self.nights = 0; self.day_n = 0
        self.asleep = False; self.last_night = None
        # the face
        self.face_now = 0.0; self.level = 0; self.face_prev = 0.0
        # the mouth's gate
        self.gate_buf = collections.deque(maxlen=96)
        self.sym_freq = {}; self.perf = {}                      # habituation tables of the intrinsic credit (per symbol)
        self.heard = torch.zeros(self.m.vocab, device=device)   # the symbols the world has said (a slow tally)
        self._gate_last = None; self._wake_last = None
        self.n_bursts = 0
        # the page and the two hands
        self.queue = collections.deque()
        self.page = []; self.page_base = 0
        self.stream = collections.deque(maxlen=96)       # (id, who)
        self.last = {}
        self.credit = collections.deque(maxlen=64)
        # optimizers: the day's (the cortex and its forecasts), the striatum's (the gate), the critic's
        self.opt_day = torch.optim.Adam(self.m.parameters(), lr=float(self.cfg["live_lr"]))
        self.opt_gate = torch.optim.SGD(self.m.mouth_gate.parameters(), lr=float(self.cfg["gate_lr"]))
        # the critic's optimizer: the value heads and the Go/NoGo gates. The bands' input maps are fixed
        # (born): trained by the critic's own bootstrapped error they are the deadly triad, and at any
        # rate they ran away (1e-3: saturated by day 6, run 26; 1e-5: saturated by day 15, run 28) while
        # learning nothing this world offers to learn (the stream carries no reward at their horizons).
        # Linear heads on fixed features under on-policy TD converge (Tsitsiklis and Van Roy).
        self.opt_value = torch.optim.Adam(list(self.m.value.parameters()) + list(self.m.band_gate.parameters()) + list(self.m.vcrit.parameters()),
                                          lr=float(self.cfg["value_lr"]))
        for p_ in self.m.band_in.parameters():
            p_.requires_grad_(False)
        self.opt_face = torch.optim.Adam(self.m.face_head.parameters(), lr=float(self.cfg["face_lr"]))

    # ---------------- feelings ----------------
    def _decay_feelings(self):
        """feelings recover on the body's own clock (per tick), so its physiology does not change
        with the speed the serve happens to run at"""
        self.fatigue *= 0.5 ** (1.0 / float(self.cfg["fatigue_half_life"]))
        self.stress *= 0.5 ** (1.0 / float(self.cfg["stress_half_life"]))
        self.mood *= 0.5 ** (1.0 / float(self.cfg["mood_half_life"]))

    # ---------------- one step of the body ----------------
    def _step(self, x, who, r=0.0, learn_store=True, dopamine=0.0):
        """one symbol enters (x: id, who: 0 world / 1 body). Returns the stream C [d] and the forecast."""
        m = self.m
        with torch.no_grad():
            ex = m.E.weight[x]
            # surprise of what arrived, against the forecast made a step ago (embedding space)
            surp = 0.0
            if self.pred_prev is not None:
                surp = float(1.0 - F.cosine_similarity(self.pred_prev, ex, dim=0))
            if who == 1:
                surp = 0.0                                  # corollary discharge: its own symbol was foretold
            if who == 0 and x != self.sil:
                self.heard *= float(self.cfg["heard_decay"]); self.heard[x] += 1.0
            # the hippocampus: write what came next under the context before it
            if self.cfg.get("store_off"):
                learn_store = False                              # an instrument: the cortex alone, no hippocampus
            if learn_store and who == 0 and x != self.sil and self.key.norm() > 1e-6:
                self.store.write(self.key, ex, surp * (1.0 + abs(dopamine)), who)   # the world's quiet is not a memory
                self._last_write = (self.key.clone(), ex.clone())                    # for the boundary mark at the offset
                if getattr(self, "_start_pending", False):
                    self.store.mark_start(self.key, ex); self._start_pending = False  # the utterance's first kept memory
            # the context moves on: both bags fade with time (a pause ends a context, as working memory
            # does), the world's symbols entering the world's bag, its own symbols its own
            # (the bags are content alone: a speaker embedding summed into every key was a constant all
            # keys shared, which pushed every cosine toward 1 and let strangers crowd an exact match)
            if who == 0:
                self.bag_w = float(self.cfg["bag_decay"]) * self.bag_w
                self.bag_o = float(self.cfg["bag_decay"]) * self.bag_o
            if x != self.sil:
                if who == 0:
                    # a world symbol: the world's context shifts a lag and takes it; what it said since
                    # the last world symbol leaves the query (it is not in any key)
                    self.bag_w = m.shift(self.bag_w) + ex
                    self.bag_o = torch.zeros_like(self.bag_o); self.n_own = 0
                else:
                    self.bag_o = m.shift(self.bag_o) + ex; self.n_own += 1
            read, conf, _ = (self.store.read(self.bag) if not self.cfg.get("store_off") else (torch.zeros(m.d, device=self.dev), 0.0, -1))
            self._read = read                                  # the latest recall (an instrument's hook)
            face = torch.tensor([self.face_now / 6.0, (self.face_now - self.face_prev) / 6.0], device=self.dev)
            # THE TICK'S POSITION. The world's symbol opens it. The world's quiet opens nothing yet: the
            # forecast the mouth reads is then the one made at the last filled position, the one
            # holding its own last symbol, which is trained to foresee what follows that symbol. (Read
            # at a freshly appended rest, the forecast was of what follows a pause, and alone the mouth
            # looped on the cue's last word: runs 20 to 22.) Its own half then fills the open position
            # or, the world quiet, opens one of its own; a tick with nothing sounded leaves a rest.
            entry = {"face": face, "bundle": self.bands.clone(), "read": read.clone(), "r": float(r)}
            if who == 0:
                if x != self.sil or not self.win:
                    self.win.append({"x": int(x), "xo": self.sil, **entry}); self._pos_open = True
                else:
                    self._pos_open = False
            else:
                if getattr(self, "_pos_open", False):
                    self.win[-1]["xo"] = int(x)                # its own sound joins the world's time step
                else:
                    self.win.append({"x": self.sil, "xo": int(x), **entry})
                self._pos_open = False
            if who == 0 and not self._pos_open and getattr(self, "_C_last", None) is not None:
                C = self._C_last                               # the last filled position's stream, and its forecast
            else:
                C = self._stream_now()
            self.bands = m.band_update(self.bands, C)
            self._C_last = C
            pred = m.forecast(C, read)
            self.pred_prev = F.normalize(pred, dim=0)
        return C, pred, surp, conf

    def _offset(self):
        """THE OFFSET (§2): the world's quiet after its utterance, once per pause. The last world position is
        marked ended, so the waking lesson's target there is the turn-end and not the next line's first letter;
        a dream ends where the cortex, so taught, expects the quiet. Nothing enters the stream, the bags and the
        mouth's context stand, and the store keeps only what the world said next: written there too, the quiet
        after a cue blended with the answer's memory and the mouth read junk (runs 45 and 46, day 1)."""
        for w in reversed(self.win):
            if w["x"] != self.sil:
                w["end"] = True; break
        if self._last_write is not None and not self.cfg.get("store_off"):
            self.store.mark_boundary(*self._last_write)                # the memory of the last symbol carries the boundary

    @property
    def bag(self):
        """the read query: the world's context, shifted by as many lags as it has said since, plus the
        efference copy of what it said (the plan is known to the sequencing system in full; it is the
        hearing of it that is suppressed): the query after its own "b" is the key the world's "b"
        would have made"""
        w = self.bag_w
        for _ in range(min(int(self.n_own), 64)):
            w = self.m.shift(w)
        return w + float(self.cfg["bag_own_weight"]) * self.bag_o

    @property
    def key(self):
        """the write key: the world's context alone. Corollary discharge suppresses the response to
        self-produced sound, so a memory of the world's sequence is stored under the world's context,
        never under its own babble (own symbols in the key at 0.6 broke recall by content; at 0.5 the
        stale key, the cue alone, outmatched the continuation key after its own first letter, cosine
        0.96 to 0.94, and the mouth stuttered the first letter: run 16, day 2)"""
        return self.bag_w

    def _window_tensors(self, win=None):
        win = list(self.win if win is None else win)
        xs = torch.tensor([w["x"] for w in win], device=self.dev)
        whos = torch.tensor([w["xo"] for w in win], device=self.dev)   # its own symbols, one per tick
        faces = torch.stack([w["face"] for w in win])
        bundles = torch.stack([w["bundle"] for w in win])
        reads = torch.stack([w["read"] for w in win])
        return xs, whos, faces, bundles, reads

    def _stream_now(self):
        xs, whos, faces, bundles, reads = self._window_tensors()
        u = self.m.inputs(xs, whos, faces, bundles, reads)
        return self.m.stream(u)[-1]

    # ---------------- the tick ----------------
    def tick(self):
        m = self.m
        self._decay_feelings()
        u = self.queue.popleft() if self.queue else self.sil
        # THE OFFSET: the world quiet for offset_ticks after its utterance, once per pause, whatever the body is
        # saying meanwhile (with the body's silence required too, a babbling body never let it fire: run 41 held
        # two turn-end memories after six days)
        off = int(self.cfg.get("offset_ticks", 0))
        if u == self.sil and off > 0 and not self._offset_done and self.ticks - self._last_world >= off:
            self._offset(); self._offset_done = True
        first_after_pause = (u != self.sil and self._offset_done)  # the first symbol after a perceived pause begins an utterance
        if u != self.sil:
            self._last_world = self.ticks; self._offset_done = False
        # the face: a change is felt; a held face is silence; easing off is not an event
        lvl = max(-6, min(6, int(self.face_now)))
        felt = 0
        if lvl != self.level:
            if abs(lvl) > abs(self.level) or lvl * self.level < 0:
                felt = lvl
            self.level = lvl
        r = float(max(-2, min(2, felt)))                    # the world's reward: the felt face, clipped like a press
        if self.cfg.get("cost_in_reward") and getattr(self, "_acted_last", False):
            # THE EFFORT IN THE REWARD: the cost of the last act is felt as the next tick's reward, so both critics
            # predict it and the gate reads their error alone. Added to the act's credit outside the critics (the
            # earlier form, with a tonic drive of 0.25 cancelling it) it was never predicted away and, with the drive
            # gone, held every act at a loss; with both gone the gate saturated at 0.98 (runs 69-72).
            r -= float(self.cfg["symbol_cost"]) * (1.0 + (self.fatigue / float(self.cfg["gate_fatigue"])) ** 2)
        # --- the ear's half: the world's symbol (or its quiet) enters ---
        v_before = m.values(self.bands.detach()) if self.v_prev is None else self.v_prev
        if off > 0 and u != self.sil:
            # THE START MARK falls on the memory whose context holds the utterance's first symbol: the second
            # symbol's, whatever the pause did to the bag (a short pause left the first symbol a memory under a faded
            # context and a long one no memory at all, so the mark fell a symbol apart between the two, runs 53/54)
            if first_after_pause:
                self._start_armed = True; self._start_pending = False
            elif getattr(self, "_start_armed", False):
                self._start_pending = True; self._start_armed = False
        C1, pred1, surp1, conf1 = self._step(u, 0, r=r, dopamine=getattr(self, "_dopa", 0.0))
        self._read_world = getattr(self, "_read", None)        # the recall as the world's symbol entered
        # --- dopamine: the fast band's error of the world's reward; the critic learns at every band ---
        with torch.no_grad():
            v_now = m.values(self.bands)
            # THE LEVEL: the slow critic's value of this moment, read by the gate below, scaled by its own
            # running root mean square (divisive normalization; born at one so a newborn's noise reads small)
            lb = int(self.cfg["gate_level_band"]); vb = float(v_now[lb])
            m.v_scale[lb] += (1.0 / float(self.cfg["diff_horizon"])) * (vb * vb - float(m.v_scale[lb]))
            level = float(self.cfg["gate_level_w"]) * max(-5.0, min(5.0, vb / (1.0 + math.sqrt(max(0.0, float(m.v_scale[lb]))))))
        gam = m.gammas()
        with torch.enable_grad():
            m.train()
            v_prev_live = m.values(self._bands_prev.detach()) if getattr(self, "_bands_prev", None) is not None else None
            if v_prev_live is not None:
                # SEMI-GRADIENT TD, the convergent form: the target r + gamma V(s_now) is detached and
                # only the heads learn; the bands' states are fixed features of the stream (their input
                # maps are born, not trained by this error). The differential heads read their states
                # centered on a running mean at the reward rate's horizon (adaptation), so the relative
                # value's free constant has no direction to walk in.
                with torch.no_grad():
                    eta = 1.0 / float(self.cfg["diff_horizon"])
                    for b in range(len(gam)):                          # every band's mean (the ventral critic centers on all)
                        m.band_mu[b] += eta * (self._bands_prev[b].detach() - m.band_mu[b])
                # DISCOUNTED TD below the differential horizon, AVERAGE-REWARD (differential) TD at and
                # above it: with the discount near 1 the bootstrapped value ran away (the slowest band
                # read 7000 against a true return near 85, correlation -0.995: run 21, day 20). The
                # baseline is the reward rate estimated at the band's own clock, tonic dopamine.
                td = torch.stack([(r + gam[b] * v_now[b].detach() - v_prev_live[b]) if not self._differential[b]
                                  else (r - float(self.rbar) + v_now[b].detach() - v_prev_live[b]) for b in range(len(gam))])
                self.rbar += (1.0 / float(self.cfg["diff_horizon"])) * (r - self.rbar)   # the reward rate, tonic dopamine
                # THE VENTRAL CRITIC: discounted TD at a definite long horizon over the whole ladder's states (semi-gradient,
                # the target detached; linear on fixed features, convergent)
                with torch.no_grad():
                    vl_now = m.value_long(self.bands)
                td_long = r + float(self.cfg.get("vcrit_gamma", 1.0 - 1.0 / 1024)) * vl_now.detach() - m.value_long(self._bands_prev.detach())
                lam = float(self.cfg.get("vcrit_lambda", 0.0))
                if lam > 0.0:
                    # THE CRITIC'S ELIGIBILITY TRACE (TD(lambda), backward view): the trace of the critic's inputs
                    # decays at gamma * lambda; the error captures it. The surrogate's gradient is -error x trace.
                    with torch.no_grad():
                        x_prev = torch.cat([(self._bands_prev.detach() - m.band_mu).reshape(-1), torch.ones(1, device=self.dev)])
                        tr = getattr(self, "_vtrace", None)
                        self._vtrace = (float(self.cfg.get("vcrit_gamma", 1.0 - 1.0 / 1024)) * lam * tr if tr is not None else torch.zeros_like(x_prev)) + x_prev
                    loss_vl = -(td_long.detach() * (m.vcrit.weight[0] @ self._vtrace[:-1] + m.vcrit.bias[0] * self._vtrace[-1]))
                else:
                    loss_vl = td_long ** 2
                loss_v = (td ** 2).mean() + loss_vl
                # Go/NoGo on the bands' own updates: a positive error pulls the gate open, a negative one shut
                gates = torch.stack([torch.sigmoid(m.band_gate[b](self._bands_prev[b].detach())).squeeze()
                                     for b in range(len(gam))])
                tgt = (td.detach() > 0).float()
                loss_g = (td.detach().abs() * F.binary_cross_entropy(gates.clamp(1e-6, 1 - 1e-6), tgt, reduction="none")).mean()
                self.opt_value.zero_grad(set_to_none=True)
                (loss_v + 0.01 * loss_g).backward()
                self.opt_value.step()
                # DOPAMINE: the TD error of the band whose discount matches dopamine's (clock 16,
                # gamma 0.9375): an expected reward fires before it lands, a missed one dips
                delta = float(td[int(self.cfg["dopamine_band"])].detach())
                delta_slow = float(td[int(self.cfg["gate_slow_band"])].detach())
                delta_long = float(td_long.detach()); vlong = float(vl_now)
                for b in range(len(gam)):
                    self.v_buf[b].append((self._bands_prev[b].detach().cpu(), r, self.bands[b].detach().cpu()))
            else:
                delta = r; delta_slow = r; delta_long = r; vlong = 0.0
            m.eval()
        self._dopa = delta
        # THE SYNAPTIC TAG: every act (or rest) leaves a tag on the gate's weights, (act - p) x the gate's input, that
        # decays at the ventral critic's own horizon; the ventral error, as it arrives over the following minutes,
        # captures the tags (Frey and Morris 1997: a tag set by activity, captured by later dopamine). The expected
        # update is the sum over acts of (act - p) x (the long return that followed minus the critic's estimate): the
        # policy gradient at the critic's horizon, where the twelve-tick sum of the lesson could not reach the parent's
        # attention. gate_slow_lr 0 = off.
        slr = float(self.cfg.get("gate_slow_lr", 0.0))
        if slr > 0.0 and getattr(self, "_gate_tag", None) is not None:
            with torch.no_grad():
                m.mouth_gate.weight += slr * delta_long * self._gate_tag[:-1].unsqueeze(0)
                m.mouth_gate.bias += slr * delta_long * self._gate_tag[-1:]
        # --- its face learns from yours (a readout) ---
        with torch.enable_grad():
            f_pred = m.face_head(C1.detach()).squeeze() * 6.0
            lf = (f_pred - torch.tensor(float(self.face_now), device=self.dev)) ** 2
            self.opt_face.zero_grad(set_to_none=True); lf.backward(); self.opt_face.step()
        its_face = float(f_pred.detach())
        # --- the mouth's half: whether (the gate), then what (the lexicon) ---
        # DECISIVENESS from tonic dopamine (songbirds: variability is high when unrewarded and falls as
        # reward comes; mood is the body's tonic dopamine): the readout's sharpness = base + gain x mood/6
        m.read_sharp = float(self.cfg["sharp_base"]) + float(self.cfg["sharp_gain"]) * max(0.0, min(6.0, self.mood)) / 6.0
        with torch.no_grad():
            sal = float(self.cfg["gate_salience"]) * float(pred1.norm())      # the proposal's salience
            feat = torch.cat([C1.detach() / math.sqrt(float(m.d)),
                              torch.tensor([self.fatigue / 10.0, self.mood / 6.0, self.stress / 10.0, sal, level], device=self.dev)])
            z = m.mouth_gate(feat.unsqueeze(0))[0, 0] / (1.0 + self.stress / 10.0)   # stress flattens the choice
            fl = float(self.cfg["gate_floor"])
            p_act = fl + (1.0 - fl) * float(torch.sigmoid(z))                          # spontaneous activity as the floor
            acted = bool(torch.rand(1, generator=self.gen).item() < p_act)
            logits = m.readout(pred1).clone()
            if self.cfg.get("end_rest"):
                # THE END IS A REST: the forecast's vote for the turn's end (a symbol the mouth can never say) is its
                # vote for silence; banned outright, a sure forecast of the end raised the proposal's salience and then
                # the next-best symbol was said in its place
                logits[self.sil] = logits[self.eot]
            else:
                logits[self.sil] = float("-inf")
            logits[self.bans] = float("-inf")
            probs = torch.softmax(logits, -1)
            ent = float(-(probs * (probs + 1e-9).log()).sum() / math.log(probs.numel()))
            if acted:
                nxt = int(torch.multinomial(probs.cpu(), 1, generator=self.gen))
                p_choice = float(probs[nxt])
                if nxt == self.sil:
                    acted, p_choice = False, 0.0                  # it chose the rest: the turn is the other's
            else:
                nxt, p_choice = self.sil, 0.0
        int_t = 0.0
        if acted:
            hab = float(self.cfg["gate_habit"])
            if str(self.cfg.get("gate_int_form", "value")) == "error":
                # THE PERFORMANCE ERROR: the forecast's belief in what it said against that syllable's usual
                # belief (a running mean per symbol), positive when it did better than usual, negative when
                # worse, habituating as the expectation catches up (Gadagkar 2016: dopamine neurons encode
                # the singing bird's performance error, and deafened birds do not learn)
                pbar = self.perf.get(nxt, 0.0)
                int_t = p_choice - pbar
                self.perf[nxt] = pbar + (1.0 - hab) * (p_choice - pbar)
            else:
                novelty = 1.0 - self.sym_freq.get(nxt, 0.0)
                int_t = p_choice * max(0.0, novelty)
                for k_ in list(self.sym_freq):
                    self.sym_freq[k_] *= hab
                    if self.sym_freq[k_] < 1e-3:
                        del self.sym_freq[k_]
                self.sym_freq[nxt] = self.sym_freq.get(nxt, 0.0) + (1.0 - hab)
            self.fatigue += float(self.cfg["symbol_cost"])
            self._step(nxt, 1, r=0.0, dopamine=delta)          # its own symbol enters the stream
        else:
            self._step(self.sil, 1, r=0.0, learn_store=False)   # its rest enters as an empty tick
        self._bands_prev = self.bands.clone()
        self._acted_last = bool(acted)
        if float(self.cfg.get("gate_slow_lr", 0.0)) > 0.0:
            with torch.no_grad():
                g_ = float(self.cfg.get("vcrit_gamma", 1.0 - 1.0 / 1024))
                tag_in = torch.cat([feat.detach(), torch.ones(1, device=self.dev)])
                prev = getattr(self, "_gate_tag", None)
                self._gate_tag = ((g_ * prev) if prev is not None else torch.zeros_like(tag_in)) + (float(acted) - p_act) * tag_in
        # --- feelings from dopamine ---
        self.mood = max(-6.0, min(6.0, self.mood + float(self.cfg["mood_gain"]) * delta))
        self.stress = min(30.0, self.stress + float(self.cfg["stress_gain"]) * max(0.0, -delta))
        if abs(delta) >= float(self.cfg["burst"]):
            self.n_bursts += 1
        # --- the gate's buffer and lesson ---
        self.gate_buf.append([feat.cpu(), acted, delta + float(self.cfg["gate_slow_w"]) * delta_slow + float(self.cfg.get("vcrit_w", 0.0)) * delta_long, int_t, self.fatigue])
        if self.ticks > 0 and self.ticks % int(self.cfg["gate_every"]) == 0 and len(self.gate_buf) >= 16 + int(self.cfg["elig_ticks"]):
            try:
                self._gate_lesson()
            except Exception as e:
                self._gate_last = {"error": str(e)[:120]}
        # --- the waking cortex ---
        if self.ticks > 0 and self.ticks % int(self.cfg["wake_every"]) == 0:
            try:
                self._wake_lesson()
            except Exception as e:
                self._wake_last = {"error": str(e)[:120]}
        # --- bookkeeping ---
        self.stream.append((int(u), 0)); self.stream.append((int(nxt), 1))
        self.page.append(((self.tok.decode([int(u)]) if u != self.sil else ""), 0, round(self.face_now, 2), round(its_face, 2)))
        self.page.append(((self.tok.decode([int(nxt)]) if nxt != self.sil else ""), 1, round(self.face_now, 2), round(its_face, 2), False))
        if len(self.page) > 40000:
            del self.page[:20000]; self.page_base += 20000
        self.face_prev = self.face_now
        self.ticks += 1; self.sleep_pressure += 1
        self.last = {"tick": self.ticks, "you": round(self.face_now, 2), "face": round(its_face, 2),
                     "mood": round(self.mood, 2), "cort": round(self.fatigue, 2), "fatigue": round(self.fatigue, 2),
                     "stress": round(self.stress, 2), "ent": round(ent, 2), "felt": felt,
                     "said": (self.tok.decode([int(nxt)]) if nxt != self.sil else ""),
                     "gate": round(p_act, 3), "dopamine": round(delta, 3), "doses": self.n_bursts, "level": round(level, 3),
                     "vlong": round(vlong, 3), "dlong": round(delta_long, 3),
                     "store": self.store.n(), "store_conf": round(conf1, 3), "surprise": round(surp1, 3),
                     "own": [self.tok.decode([int(probs.argmax())]), round(float(probs.max()), 3)],
                     "gate_lesson": self._gate_last, "wake": self._wake_last}
        if self.sleep_pressure >= int(self.cfg["wake_ticks"]) and len(self.win) >= 8:
            self._sleep_now()

    # ---------------- the gate's lesson (the striatum's opponent rule) ----------------
    def _gate_lesson(self):
        buf = list(self.gate_buf)
        K = int(self.cfg["elig_ticks"]); dec = float(self.cfg["elig_decay"])
        n = len(buf) - K
        if n < 4:
            return
        cost = float(self.cfg["symbol_cost"]); f0 = float(self.cfg["gate_fatigue"]); w_int = float(self.cfg["gate_int"])
        tonic = float(self.cfg["gate_tonic"]); vig = float(self.cfg["gate_vigor"])
        feats = torch.stack([b[0] for b in buf[:n]]).to(self.dev)
        acts = torch.tensor([1.0 if b[1] else 0.0 for b in buf[:n]])
        G = torch.zeros(n)
        for t in range(n):
            g = sum((dec ** k) * float(buf[t + k][2]) for k in range(K))     # the dopamine that followed
            if buf[t][1]:
                # acting pays a tonic drive (babble is its own reward, not contingent on confidence) plus
                # the belief it had in its choice (habituating), minus an effort cost convex in fatigue
                # (linear, 0.59 at fatigue's ceiling never beat a confident recitation's drive of 0.7:
                # run 19, gate 0.97 all day, fatigue pinned at 40; convex, the mouth speaks in bouts).
                # With the effort in the reward (cost_in_reward) the cost is the critics' to predict, not the act's
                g += tonic + w_int * float(buf[t][3]) - (0.0 if self.cfg.get("cost_in_reward") else cost * (1.0 + (float(buf[t][4]) / f0) ** 2))
            G[t] = g
        # the credit is taken against a running baseline (dopamine is an error, not a value)
        base = getattr(self, "_g_base", None)
        if base is None:
            base = float(G.mean())
        A = G - base
        self._g_base = float(self.cfg["gate_baseline"]) * base + (1.0 - float(self.cfg["gate_baseline"])) * float(G.mean())
        if float(A.abs().max()) < 1e-4:
            return
        self.m.mouth_gate.train()
        z = self.m.mouth_gate(feats).squeeze(-1)
        fl = float(self.cfg["gate_floor"])
        p = (fl + (1.0 - fl) * torch.sigmoid(z)).detach()
        # THE THREE-FACTOR RULE: credit x (action - p) has expectation cov(credit, acting), what a policy
        # must learn (Go for acts that paid, NoGo for acts that cost); plus vigor: the average credit
        # itself, tonic dopamine setting the rate of acting whatever it did
        elig = (acts.to(self.dev) - p) + vig
        loss = -(A.to(self.dev) * elig * z).mean()
        self.opt_gate.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.m.mouth_gate.parameters(), 1.0)
        self.opt_gate.step(); self.m.mouth_gate.eval()
        self._gate_last = {"n": n, "credit_mean": round(float(G.mean()), 4), "baseline": round(float(base), 4),
                           "acted": round(float(sum(1 for b in buf[:n] if b[1]) / n), 3), "tick": self.ticks}
        for _ in range(min(len(self.gate_buf), int(self.cfg["gate_every"]))):
            self.gate_buf.popleft()

    # ---------------- the waking cortex ----------------
    def _wake_lesson(self):
        win = list(self.win)[-int(self.cfg["wake_window"]):]
        if len(win) < 8:
            return None
        xs, whos, faces, bundles, reads = self._window_tensors(win)
        T = xs.shape[0]
        # THE TARGET IS THE WORLD'S NEXT SYMBOL at every position: its own symbols and rests are inputs
        # only (one predicts the environment; one's own actions are not the environment). With the
        # immediate next input as the target, the forecast the mouth reads (made as the world's symbol
        # enters) was never trained, since what follows it is always its own step: measured on run 9,
        # day 10, as an off-by-one ("a" after "where ball? ", "o" after "big dog bigger ").
        nxt = [-1] * T; last = -1
        for t in range(T - 1, -1, -1):
            nxt[t] = last
            if int(xs[t]) != self.sil:
                last = t
        tgt_pos = [max(0, i) for i in nxt]
        w = torch.tensor([1.0 if i >= 0 else 0.0 for i in nxt], device=self.dev)
        y = xs[torch.tensor(tgt_pos, device=self.dev)].clone()
        for t in range(T):
            if win[t].get("end"):                                  # THE OFFSET: after this symbol the world went quiet
                y[t] = self.eot; w[t] = 1.0
        if float(w.sum()) < 1:
            return None
        m = self.m; m.train()
        try:
            self.opt_day.zero_grad(set_to_none=True)
            u = m.inputs(xs, whos, faces, bundles, reads)     # the window as lived: its own sound in it, attenuated
            C = m.stream(u)
            # the cortex is trained on ITS OWN forecast, day and night alike (predictive coding: each
            # area learns from its own error); recall is a parallel contribution the mouth reads, never
            # a term in the cortex's error (with the sum in the loss the day taught only the residual
            # the store missed and undid the night: run 13, day 4)
            pred = m.latent_pred(C)
            ll, lc = m.latent_loss(pred, y, w=w)
            fl, fc = m.forecast_loss(C[:-1].detach(), bundles[1:], sig=0.0)   # the PFC's heads learn from the stream, not through it
            loss = (ll + fl) * (1.0 + self.stress / 10.0)      # stress raises plasticity
            if not bool(torch.isfinite(loss.detach())):
                return {"skipped": "non-finite"}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            self.opt_day.step()
            out = {"latent_cos": round(lc, 3), "forecast_cos": round(fc, 3), "n_world": int(w.sum()), "tick": self.ticks}
        finally:
            self.opt_day.zero_grad(set_to_none=True); m.eval()
        self._wake_last = out
        return out

    # ---------------- the night ----------------
    def dreams(self, n=None):
        """dreams start where the store is strongest and run by pattern completion until the recall
        is half as sure as a memory of its own (relative to the store, not a constant)"""
        n = int(self.cfg["night_starts"] if n is None else n)
        starts = self.store.sample_starts(n, gen=self.gen)
        d_ = float(self.cfg["bag_decay"]); ref = self.store.self_confidence(qnorm=(1.0 / (1.0 - d_ * d_)) ** 0.5)   # a full context's norm
        floor = float(self.cfg["dream_floor_rel"]) * ref
        out = []
        a_hit, a_rec = float(self.cfg["dream_adapt"]), float(self.cfg["dream_recover"])
        with torch.no_grad():
            for j in starts:
                bag = self.store.K[j].clone()                           # a dream's context: per symbol, as the keys are
                # the dream begins with the context's own last symbol, read from the key (a key is the bag before the
                # memory's symbol, its newest term whole): at an onset that is the utterance's first symbol, which the
                # store never kept as a memory of its own (dreams began 'og will go', and the cortex lost every line's
                # first symbols: its trace fell from 60 to 38 of 82, runs 53/54)
                ids = [self.m.nearest(bag)]
                adapt = torch.ones(self.store.n(), device=self.dev)      # neural adaptation: a recalled memory tires
                s_floor = float(self.cfg["store_floor_rel"]) * float(self.store.S.mean())
                for _ in range(int(self.cfg["dream_max"])):
                    pred, conf, win = self.store.read(bag, adapt=adapt)
                    if conf < floor or (win >= 0 and (float(self.store.S[win] * adapt[win]) < s_floor
                                                      or float(adapt[win]) < float(self.cfg["dream_exhaust"]))):
                        break                                             # unsure, or the memory is exhausted (a slot fires at most twice)
                    lg = self.m.readout(pred).clone()
                    lg[self.bans] = float("-inf"); lg[self.sil] = float("-inf")
                    nid = int(lg.argmax())
                    ids.append(nid)
                    if int(self.cfg.get("offset_ticks", 0)) > 0 and win >= 0 and bool(self.store.B[win]):
                        ids.append(self.eot); break                       # the memory ends where the world went quiet
                    adapt = 1.0 - a_rec * (1.0 - adapt)                   # recovery toward 1
                    # the recalled memory tires fully each time it fires, and every slot tires in
                    # proportion to how much it fired (neural adaptation), so a cycle exhausts itself
                    # even when the attention is spread over near-duplicate memories of one context
                    adapt = adapt * (1.0 - (1.0 - a_hit) * self.store._last_w)
                    if win >= 0:
                        adapt[win] *= a_hit
                    bag = float(self.cfg["bag_decay"]) * self.m.shift(bag) + self.m.E.weight[nid]
                if len(ids) >= 3:
                    out.append(ids)
        return out

    def _dream_inputs(self, ids, mem_on):
        """a dream as a window: fresh bands (a night's working state), the store leading if mem_on"""
        m = self.m
        bands = torch.zeros_like(self.bands); bag = torch.zeros_like(self.bag_w)
        xs = [self.sil] + list(ids[:-1]); reads, bundles = [], []
        xos = torch.full((len(xs),), self.sil, dtype=torch.long, device=self.dev)   # a dream: no own sound
        with torch.no_grad():
            for x in xs:
                bag = float(self.cfg["bag_decay"]) * (m.shift(bag) if x != self.sil else bag) + (m.E.weight[x] if x != self.sil else 0.0)
                rd = self.store.read(bag)[0] if mem_on else torch.zeros(m.d, device=self.dev)
                reads.append(rd); bundles.append(bands.clone())
                n = len(reads)
                u = m.inputs(torch.tensor(xs[:n], device=self.dev), xos[:n],
                             torch.zeros(n, 2, device=self.dev), torch.stack(bundles), torch.stack(reads))
                C = m.stream(u)[-1]
                bands = m.band_update(bands, C)
        T = len(xs)
        return (torch.tensor(xs, device=self.dev), xos,
                torch.zeros(T, 2, device=self.dev), torch.stack(bundles), torch.stack(reads), torch.tensor(ids, device=self.dev))

    def gauge(self, dreams):
        """the cortex alone (store off), teacher-forced on the dreams: the share of next symbols it
        forecasts itself (argmax), and the mean cosine of its forecast to the embedding received
        (the finer instrument: it moves before the argmax does)"""
        hits = n = 0; cos_sum = 0.0
        with torch.no_grad():
            for ids in dreams:
                xs, whos, faces, bundles, reads, y = self._dream_inputs(ids, mem_on=False)
                C = self.m.stream(self.m.inputs(xs, whos, faces, bundles, reads))
                pred = self.m.latent_pred(C)
                lg = self.m.readout(pred); lg[:, [b for b in self.bans if b != self.eot]] = float("-inf"); lg[:, self.sil] = float("-inf")
                hits += int((lg.argmax(-1) == y).sum()); n += int(y.numel())   # a dream's end (the turn's) counts as a target
                cos_sum += float(F.cosine_similarity(pred, self.m.E.weight[y], dim=-1).sum())
        self._gauge_cos = (round(cos_sum / n, 3) if n else None)
        return (round(hits / n, 3) if n else None), n

    def _night_step(self, opt):
        gn = torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0)
        if bool(torch.isfinite(gn)):
            opt.step()
        opt.zero_grad(set_to_none=True)

    def night(self):
        self.asleep = True
        m = self.m
        rep = {"night": self.nights + 1, "tick": self.ticks}
        try:
            dreams = self.dreams()
            rep["dreams"] = len(dreams)
            rep["examples"] = [self.tok.decode(d)[:32] for d in dreams[:8]]
            rep["mean_len"] = round(sum(len(d) for d in dreams) / len(dreams), 1) if dreams else 0
            if not dreams:
                rep["note"] = "the store holds nothing to dream"
            else:
                before, nsym = self.gauge(dreams); before_cos = self._gauge_cos
                opt = torch.optim.Adam(m.parameters(), lr=float(self.cfg["night_lr"]))   # sleep's own plasticity
                sig = float(self.cfg["sigreg"])
                m.train()
                nrem = 0; losses = []
                for _ in range(int(self.cfg["night_rounds"])):
                    opt.zero_grad(set_to_none=True); tot = 0.0; ok = 0
                    for ids in dreams:
                        # the hippocampus replays the sequence; the cortex must carry it itself (the read
                        # is not an input to the lesson, or the cortex learns to copy the recall and the
                        # gauge, taken alone, stays flat: run 6, day 4)
                        xs, whos, faces, bundles, reads, y = self._dream_inputs(ids, mem_on=False)
                        C = m.stream(m.inputs(xs, whos, faces, bundles, reads))
                        ll, _ = m.latent_loss(m.latent_pred(C), y)          # no SIGReg: with a fixed lexicon and the PFC's
                                                                             # objective off the trunk, nothing can collapse
                        if not bool(torch.isfinite(ll.detach())):
                            continue
                        (ll / len(dreams)).backward(); tot += float(ll.detach()) / len(dreams); ok += 1
                    if ok:
                        self._night_step(opt); nrem += 1; losses.append(round(tot, 3))
                mid, _ = self.gauge(dreams); mid_cos = self._gauge_cos      # the gauge after NREM, before REM
                # REM: the cortex runs free from each dream's first symbols on its own readout,
                # a quarter of the night in rounds (biology's share), each round one batched step
                rem_cos = []; rem_steps = 0
                for _ in range(int(self.cfg["rem_rounds"])):
                    opt.zero_grad(set_to_none=True); rc = []
                    for ids in dreams[:int(self.cfg["rem_dreams"])]:
                        fl, fc = self._rem_rollout(ids, sig)
                        if fl is None or not bool(torch.isfinite(fl.detach())):
                            continue
                        (fl / max(1, min(len(dreams), int(self.cfg["rem_dreams"])))).backward(); rc.append(fc)
                    if rc:
                        self._night_step(opt); rem_steps += 1; rem_cos.append(sum(rc) / len(rc))
                # the value ladder replays its lived pairs once
                self._value_replay()
                m.eval()
                del opt
                finite = all(bool(torch.isfinite(p).all()) for p in m.parameters())
                rep["discarded"] = not finite
                if not finite and self.save_path and os.path.exists(self.save_path):
                    sd = torch.load(self.save_path, map_location="cpu", weights_only=False)
                    m.load_state_dict(sd["organs"]); m.to(self.dev)
                after, _ = self.gauge(dreams); after_cos = self._gauge_cos
                rep.update({"nrem_steps": nrem, "nrem_loss": losses[:3] + (["..."] if len(losses) > 6 else []) + losses[-3:],
                            "rem_steps": rem_steps, "rem_cos": (round(rem_cos[-1], 3) if rem_cos else None),
                            "rem_cos_first": (round(rem_cos[0], 3) if rem_cos else None),
                            "gauge": {"before": before, "after_nrem": mid, "after": after, "symbols": nsym,
                                      "cos_before": before_cos, "cos_after_nrem": mid_cos, "cos_after": after_cos}})
            # the rest: the store fades, the working state wakes fresh, the body is saved
            rep["store_dropped"] = self.store.fade(float(self.cfg["store_fade"]), float(self.cfg["store_floor_rel"]))
            rep["store_slots"] = self.store.n()
            self.bands.zero_(); self.bag_w.zero_(); self.bag_o.zero_(); self.n_own = 0; self.win.clear(); self.pred_prev = None
            self._bands_prev = None; self._C_last = None; self.v_prev = None
            self.stream.clear(); self.gate_buf.clear(); self._g_base = None; self._gate_tag = None; self._vtrace = None
            self.fatigue = 0.0
            self.sleep_pressure = 0
            self.nights += 1; self.day_n += 1
            self.last_night = rep
            if self.save_path:
                self.save()
        except Exception as e:
            rep["error"] = str(e)[:200]
            self.sleep_pressure = int(self.cfg["wake_ticks"]) // 2
            self.last_night = rep
        finally:
            self.asleep = False
        return rep

    def _rem_rollout(self, ids, sig, k=3):
        m = self.m
        L = int(self.cfg["rem_steps"])
        if len(ids) < k + 1:
            return None, None
        bands = torch.zeros_like(self.bands); bag = torch.zeros_like(self.bag_w)
        xs = [self.sil] + list(ids[:k])
        reads, bundles, Cs, bnext = [], [], [], []
        for step in range(len(xs) + L):
            if step >= len(xs):
                # its imagined next symbol, read off its forecast (no gradient through the choice),
                # heard as the world's in the next time step
                with torch.no_grad():
                    lg = m.readout(m.latent_pred(Cs[-1])).clone(); lg[self.bans] = float("-inf"); lg[self.sil] = float("-inf")
                    xs.append(int(lg.argmax()))
            with torch.no_grad():
                bag = float(self.cfg["bag_decay"]) * (m.shift(bag) if xs[step] != self.sil else bag) + (m.E.weight[xs[step]] if xs[step] != self.sil else 0.0)
            reads.append(torch.zeros(m.d, device=self.dev)); bundles.append(bands.clone())
            n = step + 1
            u = m.inputs(torch.tensor(xs[:n], device=self.dev), torch.full((n,), self.sil, dtype=torch.long, device=self.dev),
                         torch.zeros(n, 2, device=self.dev), torch.stack(bundles), torch.stack(reads))
            C = m.stream(u)[-1]
            Cs.append(C)
            with torch.no_grad():
                bands = m.band_update(bands, C.detach())
            bnext.append(bands.clone())
        # the stream at t forecasts the bundle handed over at t+1, along the free-running part. THE
        # STREAM IS DETACHED: the PFC's forecast heads learn from the cortex, they do not rewrite it
        # (each area learns from its own error). Trained through the trunk, six REM rounds undid a third
        # of NREM's gain on the next-symbol forecast (scratch night on run 19's day-2 body: gauge 0.66 ->
        # 0.85 after NREM -> 0.78 after REM, 0.85 with the stream detached; SIGReg was not the cause).
        C_free = torch.stack(Cs[len(ids[:k]):-1]).detach(); B_next = torch.stack(bnext[len(ids[:k]):-1])
        if C_free.shape[0] < 2:
            return None, None
        return m.forecast_loss(C_free, B_next, sig=0.0)

    def _value_replay(self):
        m = self.m; gam = m.gammas()
        terms = []
        for b in range(len(gam)):
            pairs = list(self.v_buf[b])
            if len(pairs) < 4:
                continue
            hp = torch.stack([p[0] for p in pairs]).to(self.dev); R = torch.tensor([p[1] for p in pairs], device=self.dev)
            hn = torch.stack([p[2] for p in pairs]).to(self.dev)
            with torch.no_grad():
                vn = m.value_of(b, hn)
            vp = m.value_of(b, hp)
            terms.append((((R - float(self.rbar) + vn - vp) if self._differential[b] else (R + gam[b] * vn - vp)) ** 2).mean())
        if terms:
            self.opt_value.zero_grad(set_to_none=True)
            torch.stack(terms).mean().backward(); self.opt_value.step()

    def _sleep_now(self):
        self.queue.clear()
        self.night()

    # ---------------- the hands ----------------
    def type_text(self, s):
        n = 0
        for ch in s:
            i = self.tok.token_to_id(ch)
            if i is not None and i >= 11 and i != self.nl and len(self.queue) < 600:
                self.queue.append(i); n += 1
        return {"queued": n}

    def set_face(self, expr):
        self.face_now = max(-6.0, min(6.0, float(expr)))
        return {"you": self.face_now}

    def state(self, since=0):
        i = max(0, int(since) - self.page_base)
        return {"page": self.page[i:], "n": self.page_base + len(self.page), "last": self.last,
                "queued": len(self.queue), "asleep": self.asleep, "sleep_pressure": self.sleep_pressure,
                "wake_ticks": int(self.cfg["wake_ticks"]), "nights": self.nights, "last_night": self.last_night,
                "store": self.store.n()}

    # ---------------- save / load ----------------
    def save(self, path=None):
        path = path or self.save_path
        blob = {"organs": self.m.state_dict(), "store": self.store.state_dict(), "cfg": self.cfg,
                "arch": {"vocab": self.m.vocab, "d": self.m.d, "layers": len(self.m.blocks), "heads": self.m.blocks[0].attn.num_heads,
                         "window": self.m.window, "clocks": list(self.m.clocks)},
                "life": {"ticks": self.ticks, "nights": self.nights, "day_n": self.day_n, "sleep_pressure": self.sleep_pressure, "heard": self.heard.cpu(),
                         "fatigue": self.fatigue, "stress": self.stress, "mood": self.mood, "n_bursts": self.n_bursts,
                         "sym_freq": self.sym_freq, "perf": {int(k): float(v) for k, v in self.perf.items()}, "last_night": self.last_night, "rbar": float(self.rbar)}}
        torch.save(blob, path + ".tmp"); os.replace(path + ".tmp", path)
        return {"saved": path}

    @classmethod
    def load(cls, path, tok, device="cpu", cfg=None, seed=0, save_path=None):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        a = blob["arch"]
        organs = Organs(a["vocab"], d=a["d"], layers=a["layers"], heads=a["heads"], window=a["window"], clocks=tuple(a["clocks"]))
        w = blob["organs"].get("mouth_gate.weight")
        if w is not None and w.shape[1] < organs.mouth_gate.weight.shape[1]:
            # an older body's gate had fewer inputs (no salience, no level): those weights are born at zero
            blob["organs"]["mouth_gate.weight"] = torch.cat([w, torch.zeros(w.shape[0], organs.mouth_gate.weight.shape[1] - w.shape[1])], 1)
        missing = organs.load_state_dict(blob["organs"], strict=False)
        if missing.missing_keys:
            print("load: organs without", missing.missing_keys, "(an older recipe; born fresh where missing)")
        c = dict(blob.get("cfg") or {})
        c.setdefault("gate_int_form", "value")            # an older body keeps the value form and its own drive unless told
        c.update(cfg or {})
        life = cls(organs, tok, cfg=c, device=device, seed=seed, save_path=save_path or path)
        life.store.load_state_dict(blob["store"])
        L = blob.get("life") or {}
        for k in ("ticks", "nights", "day_n", "sleep_pressure", "fatigue", "stress", "mood", "n_bursts", "last_night"):
            if k in L:
                setattr(life, k, L[k])
        life.sym_freq = dict(L.get("sym_freq") or {})
        life.perf = {int(k): float(v) for k, v in (L.get("perf") or {}).items()}
        if L.get("rbar") is not None:
            rb = L["rbar"]; life.rbar = float(rb.mean()) if torch.is_tensor(rb) else float(rb)
        if L.get("heard") is not None:
            life.heard = L["heard"].to(device)
        return life

    @classmethod
    def birth(cls, tok, device="cpu", d=256, layers=6, heads=4, window=64, cfg=None, seed=0, save_path=None):
        torch.manual_seed(int(seed))
        organs = Organs(tok.get_vocab_size(), d=d, layers=layers, heads=heads, window=window, birth_act=float((cfg or {}).get("birth_act", PHYSIOLOGY["birth_act"])))
        return cls(organs, tok, cfg=cfg, device=device, seed=seed, save_path=save_path)
