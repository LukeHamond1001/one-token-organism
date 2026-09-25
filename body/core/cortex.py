"""the cortex (a mixin of `Life`, body/life.py): one step of the body (`_step`: a symbol enters, its surprise, the store's write
under the key, the contexts move, the waking read, the stream and the forecast), the window as tensors and the stream now, the
waking lesson (`_wake_lesson`), and the readout's calibrated sharpness (`_sharp_calibrate`).

Moved verbatim from body/life.py (review 2026-09-22 section 4, step 2). Since the core refactor's step R4 (docs/SIM_DESIGN.md 8.4) a
window position holds each of the anatomy's channels under its field, the window's tensors are per channel, the stream's input is
their codes summed in the anatomy's order (`Organs.inputs`), and the waking lesson teaches a later channel's own forecast head. Since step
R6 it also teaches each later effector's motor timing part (act_pred and the forward half, body/core/timing.py `_timing_loss`), and a body
with later effectors has its lesson whether or not the world spoke in the window (their acts are targets on every position); act_pred and
the correction step with an optimizer of their own, plain steps on the lesson's gradient with its labels at their reliability
(`GatedDescent`, R6 fix 7), the labels act_inv reads reach them and never the stream, and each optimizer's gradient has its own bound
(the language body's is as it was)."""
import torch
import torch.nn.functional as F


class CortexMixin:
    # ---------------- one step of the body ----------------
    def _step(self, x, who, r=0.0, learn_store=True, dopamine=0.0):
        """one symbol enters (x: id, who: 0 world / 1 body). Returns the stream C [d] and the forecast."""
        m = self.m
        if who == 0 and getattr(self, "pred_prev", None) is not None:      # the tick's surprise, the rest included: the event's end by the law
            with torch.no_grad():
                self._surp_tick = float(1.0 - F.cosine_similarity(self.pred_prev, m.E.weight[int(x)], dim=0))
            if str(self.cfg.get("sharp_form", "fixed")) in ("calibrated", "world") and int(x) != self.sil:
                self._sharp_calibrate(int(x))       # on the world's spoken symbols only: on its quiet ticks the forecast is of the mouth's own next
                                                    # letter, and scored against the rest the gradient was negative whatever the reading (20:50)
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
            seam = bool(getattr(self, "_seam_pending", False)) and who == 0 and x != self.sil
            if seam:
                self._seam_pending = False                       # the first symbol's turn, kept or not
            if who == 0 and x != self.sil:
                self._utt_cur.append(int(x))                     # the utterance as heard, symbol by symbol (dream_source utterances)
                self._utt_felt = float(getattr(self, "_utt_felt", 0.0)) + surp * (1.0 + abs(dopamine))   # its felt strength, the store's law (utt_entry felt)
            # THE KEY OF A MEMORY (key_form; 2026-09-14, the twenty-seventh defect): "bag", the world's last symbols as a decayed, shifted sum
            # (the last five characters, in effect: "what is hot?" and "is your yam hot?" wrote under one key, and the store answered
            # nine of thirty fact questions however the pause or the horizon was set); "cortex", the stream's state at the position
            # before this symbol, the query of the tick before (the hippocampus indexes the cortex's pattern, not the sense data)
            key_ = self._q_prev if str(self.cfg.get("key_form", "bag")) == "cortex" else self.key
            if key_ is None:
                key_ = torch.zeros(m.d, device=self.dev)
            # THE WRITE FLOOR (2026-09-14, 21:15): the key's norm had to exceed 1e-6 for a memory to be written, a guard against the empty
            # bag at birth; but the world's bag fades 0.8 a tick through the child's turn, and after 62 ticks of the child talking the
            # first symbol of the other voice's answer fell under the floor and was never written: the answer's onset, the very memory
            # the question must find. The direction of a faded bag is the question's still; the floor is a constant (write_floor).
            if learn_store and who == 0 and x != self.sil and key_.norm() > float(self.cfg.get("write_floor", 1e-6)):
                if self.store.write(key_, ex, surp * (1.0 + abs(dopamine)), who):   # the world's quiet is not a memory
                    self._writes_today = int(getattr(self, "_writes_today", 0)) + 1   # the day's kept writes (the night's count; the store's growth stops at the capacity)
                rm_ = getattr(self.store, "last_remap", None)
                if rm_ is not None:
                    # THE THIRTY-SECOND DEFECT (2026-09-18, read at the rekey; confirmed on a tiny body: at the capacity 25 of 52 links joined
                    # the wrong slots, under it 138 of 138): a write beyond the capacity evicts the weakest slot and re-sorts the store by
                    # strength, so the index of the symbol before (the chain's link) and of the episode being followed went stale, and
                    # since the store reached the capacity (night 256) about half the chain links were made from a neighbouring slot. The
                    # store now reports its remap and the indices held across the write follow it.
                    self._prev_slot = int(rm_[self._prev_slot]) if 0 <= self._prev_slot < rm_.numel() else -1
                    if self._follow is not None and 0 <= self._follow[0] < rm_.numel():
                        s_ = int(rm_[self._follow[0]]); self._follow = ((s_,) + tuple(self._follow[1:])) if s_ >= 0 else None
                if int(self.cfg.get("store_chain", 0)) and self.store.last_idx >= 0:
                    if self._prev_slot >= 0:
                        self.store.link(self._prev_slot, self.store.last_idx, tag=self.store.episode)   # the episode's order: this symbol followed that one
                    self._prev_slot = self.store.last_idx
                if int(self.cfg.get("episode_chain", 0)) and self.store.last_idx >= 0:
                    self.store.note(self.store.episode, self.store.last_idx)           # the utterance's own ordered chain
                self._last_write = (key_.clone(), ex.clone())                       # for the boundary mark at the offset
                if getattr(self, "_start_pending", False):
                    self.store.mark_start(key_, ex); self._start_pending = False     # the utterance's first kept memory
                if seam:
                    self.store.mark_seam(key_, ex)               # the first symbol under the last line's faded context
            # the context moves on: both bags fade with time (a pause ends a context, as working memory
            # does), the world's symbols entering the world's bag, its own symbols its own
            # (the bags are content alone: a speaker embedding summed into every key was a constant all
            # keys shared, which pushed every cosine toward 1 and let strangers crowd an exact match)
            if who == 0:
                # THE HOLD OF WORKING MEMORY (bag_rest_decay; 2026-09-15, the live ruler's trace): the world's context faded 0.8 a tick
                # whether or not a symbol came, so a question was gone from the recall's cue eight ticks after its mark, and the gate,
                # taught caution by the talk-over frowns, opened twelve to forty ticks after: the child answered "yes." to the recall's
                # faded mean while the greedy readout two ticks after the question had said "the sun is hot". A context fades a symbol
                # at a time as new symbols displace it; in the quiet, working memory holds it (the delay-period activity of prefrontal
                # cortex holds a cue for seconds). The fast bags fade by bag_decay per symbol of their own kind and by bag_rest_decay
                # per quiet tick (0 = the old rule, the one rate for both).
                rest_ = float(self.cfg.get("bag_rest_decay", 0.0)) or float(self.cfg["bag_decay"])
                self.bag_w = (float(self.cfg["bag_decay"]) if x != self.sil else rest_) * self.bag_w
                self.bag_o = rest_ * self.bag_o                   # its own symbol, if one comes this tick, brings its own bag to the symbol rate (take_own)
            if x != self.sil:
                if who == 0:
                    # a world symbol: the world's context shifts a lag and takes it; what it said since
                    # the last world symbol leaves the query (it is not in any key)
                    self.take_world(int(x))
                else:
                    self.take_own(int(x))
            end_vec = F.normalize(m.E.weight[self.end_id], dim=0) if int(self.cfg.get("recall_end", 0)) else None
            rt_ = float(self.cfg.get("read_tire", 0.0))
            tire_ = self.store.A if (rt_ > 0.0 and self.store.A.numel() == self.store.n()) else None
            cortex_key = str(self.cfg.get("key_form", "bag")) == "cortex"
            if cortex_key:
                read = self._read_prev if getattr(self, "_read_prev", None) is not None else torch.zeros(m.d, device=self.dev)
                conf, win_ = 0.0, -1                           # the recall of the tick before enters the cortex (the return path's delay)
            else:
                read, conf, win_ = (self._recall(self.bag, end_vec=end_vec, tire=tire_) if not self.cfg.get("store_off") else (torch.zeros(m.d, device=self.dev), 0.0, -1))
                self._tire(win_, rt_)
                self._read = read                              # the latest recall (an instrument's hook)
            # THE POSITION'S CHANNELS (the core refactor's step R4, docs/SIM_DESIGN.md 8.4): each of the anatomy's channels observed at the
            # position this step opens, under the channel's window field (the diary's: the ear "x", the world's symbol, its rest at a position
            # its own sound opens; the face "face", [face/6, its change/6])
            obs_ = {c_.field: c_.observe(self, x, who) for c_ in self.anatomy.channels}
            # THE TICK'S POSITION. The world's symbol opens it. The world's quiet opens nothing yet: the
            # forecast the mouth reads is then the one made at the last filled position, the one
            # holding its own last symbol, which is trained to foresee what follows that symbol. (Read
            # at a freshly appended rest, the forecast was of what follows a pause, and alone the mouth
            # looped on the cue's last word: runs 20 to 22.) Its own half then fills the open position
            # or, the world quiet, opens one of its own; a tick with nothing sounded leaves a rest.
            entry = {"bundle": self.bands.clone(), "read": read.clone(), "r": float(r)}
            # THE LATER EFFECTORS' ACTS (step R5): each effector after the voice holds its act under its window field, the efference copy
            # the cortex hears beside its own sound: its act this tick at the tick's own step (its rest where it did not act), its rest
            # at a position the world opens. The diary declares none: its positions are as before.
            acts_ = ({e_.field: (int(st_["now"]["act"]) if who == 1 else int(e_.rest_id)) for e_, st_ in zip(self.anatomy.motors, self.motor)}
                     if len(self.anatomy.effectors) > 1 else None)
            if who == 0:
                if x != self.sil or not self.win:
                    self.win.append({**obs_, "xo": self.sil, **entry}); self._pos_open = True
                    if acts_:
                        self.win[-1].update(acts_)
                else:
                    self._pos_open = False
            else:
                if getattr(self, "_pos_open", False):
                    self.win[-1]["xo"] = int(x)                # its own sound joins the world's time step
                else:
                    self.win.append({**obs_, "xo": int(x), **entry})
                if acts_:
                    self.win[-1].update(acts_)                 # the later effectors' acts join it too
                self._pos_open = False
            if who == 0 and not self._pos_open and getattr(self, "_C_last", None) is not None:
                C = self._C_last                               # the last filled position's stream, and its forecast
            else:
                C = self._stream_now()
            if who == 0:                                       # once a tick (review 2026-09-06: twice doubled every band's rate)
                self.bands = m.band_update(self.bands, C)
            self._C_last = C
            if cortex_key:
                q = self.query_from(C, learn=True)
                read, conf, win_ = (self._recall(q, end_vec=end_vec, tire=tire_) if not self.cfg.get("store_off") else (torch.zeros(m.d, device=self.dev), 0.0, -1))
                self._tire(win_, rt_)
                self._read = read; self._read_prev = read; self._q_prev = q
            pred = m.forecast(C, read)
            self._fc_prev = pred.detach()
            self.pred_prev = F.normalize(pred, dim=0)
        return C, pred, surp, conf

    def _window_tensors(self, win=None):
        """THE WINDOW AS TENSORS, PER CHANNEL (step R4): obs, each of the anatomy's channels by name, its field at every position ([T]
        symbols of a symbol channel, [T, size] of a vector channel: the diary's ear [T] and face [T, 2]), and each later effector's acts
        by its name ([T], step R5); whos [T] its own symbol at each position; bundles [T, nb, d]; reads [T, d]"""
        win = list(self.win if win is None else win)
        obs = {c_.name: (torch.tensor([w[c_.field] for w in win], device=self.dev) if c_.kind == "symbol" else torch.stack([w[c_.field] for w in win]))
               for c_ in self.anatomy.channels}
        for e_ in self.anatomy.motors:                          # each later effector's acts, by its name (step R5)
            obs[e_.name] = torch.tensor([w[e_.field] for w in win], device=self.dev)
        whos = torch.tensor([w["xo"] for w in win], device=self.dev)   # its own symbols, one per tick
        bundles = torch.stack([w["bundle"] for w in win])
        reads = torch.stack([w["read"] for w in win])
        return obs, whos, bundles, reads

    def _stream_now(self):
        obs, whos, bundles, reads = self._window_tensors()
        u = self.m.inputs(self.anatomy, obs, whos, bundles)
        return self.m.stream(u)[-1]

    # ---------------- the waking cortex ----------------
    def _wake_lesson(self):
        win = list(self.win)[-int(self.cfg["wake_window"]):]
        if len(win) < 8:
            return None
        obs, whos, bundles, reads = self._window_tensors(win)
        xs = obs[self.anatomy.words.name]                          # the words (channel 0): the world's symbols, the lesson's targets
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
        odc = float(self.cfg.get("own_target_decay", 0.0))
        if odc > 0.0:
            # THE OWN-BABBLE TARGET FADES WITH DISTANCE (own_target_decay; 2026-09-06): at its own positions the target is the
            # world's next symbol, so a long babble targets the first letter of the parent's next line at every position and the
            # forecast locks on it (the second body's mouth, 'w' for an hour on a day of w-lines). A prediction is owed only
            # where one is possible: an own position's weight decays by the distance to the next world symbol.
            for t_ in range(T):
                if nxt[t_] >= 0 and int(whos[t_]) != 0:
                    w[t_] = w[t_] * (odc ** max(0, nxt[t_] - t_ - 1))
        y = xs[torch.tensor(tgt_pos, device=self.dev)].clone()
        for t in range(T):
            if win[t].get("end"):                                  # THE OFFSET: after this symbol the world went quiet
                y[t] = self.end_id; w[t] = 1.0
        if str(self.cfg.get("own_target_form", "world")) == "recall":
            cf = float(self.cfg.get("own_target_conf", 0.3)); n_rec = 0
            for t in range(T - 1):
                if int(whos[t]) != self.sil:                       # its own position: the target is the recall's continuation of what it said
                    rv = reads[t + 1]; c = float(rv.norm())
                    if c > cf:
                        y[t] = int(self.m.nearest(rv)); w[t] = min(1.0, c); n_rec += 1
                    else:
                        w[t] = 0.0                                 # unsure: nothing owed
            self._wake_recall_targets = getattr(self, "_wake_recall_targets", 0) + n_rec
        motor_ = len(self.anatomy.effectors) > 1                   # step R6: a later effector's acts are targets whether or not the world spoke
        if float(w.sum()) < 1 and not motor_:
            return None
        m = self.m; m.train()
        try:
            self.opt_day.zero_grad(set_to_none=True)
            if motor_:
                self.opt_pred.zero_grad(set_to_none=True)          # act_pred's and the corrections' (step R6)
            u = m.inputs(self.anatomy, obs, whos, bundles)     # the window as lived: its own sound in it, attenuated
            C = m.stream(u)
            # the cortex is trained on ITS OWN forecast, day and night alike (predictive coding: each
            # area learns from its own error); recall is a parallel contribution the mouth reads, never
            # a term in the cortex's error (with the sum in the loss the day taught only the residual
            # the store missed and undid the night: run 13, day 4)
            pred = m.latent_pred(C)
            ll, lc = m.latent_loss(pred, y, w=w)
            # THE LATER CHANNELS' FORECASTS (step R4): each later channel that declares a forecast is foreseen by its own head, the
            # channel's code at the next position as the target (held still: the head learns to foresee the code, not the code to
            # meet the head), by the squared error latent_loss uses; the diary's face declares none, so its lesson is as before
            es_ = self._err_scales()                                    # step R7c: each channel's error over its own running mean (err_scale)
            for i_, c_ in enumerate(self.anatomy.channels):
                if i_ and c_.forecast:
                    with torch.no_grad():
                        tgt_ = c_.encode(m, obs[c_.name][1:])
                    lc_ = 0.5 * ((m.head(self.anatomy, i_)(C[:-1]).float() - tgt_.float()) ** 2).sum(-1).mean()
                    if es_ and c_.name in es_:
                        lc_ = lc_ / es_[c_.name]
                    ll = ll + lc_
            # THE LATER EFFECTORS' TIMING PARTS (step R6): act_pred taught the act at each position from the stream before it (its own
            # act, or where it rested what act_inv reads moved it, weighted by act_inv's reliability), the forward half its next body sense;
            # the whole lesson the own acts' errors plus act_inv's labels' at their reliability; act_pred and the correction step with
            # opt_pred (body/core/timing.py GatedDescent: one plain step a lesson on this gradient, each element bounded, R6 fix 7), the
            # rest here; act_inv's labels reach act_pred and the correction alone, never the stream (`_timing_loss`)
            mrep = {}
            for i_ in range(1, len(self.anatomy.motors) + 1):
                lt_, rt_, lb_ = self._timing_loss(i_, C, obs)
                if lt_ is not None:
                    ll = ll + lt_; mrep[self.anatomy.motors[i_ - 1].name] = rt_
                    if lb_ is not None and rt_["rel"] > 0.0:
                        ll = ll + rt_["rel"] * lb_                      # act_inv's labels at their reliability (they reach act_pred alone)
            if str(self.cfg.get("rem_form", "forecast")) == "imagine":
                fl, fc = torch.zeros((), device=self.dev), 1.0            # the forecast heads retired (§5c: their target was the stream's own dynamics)
            else:
                fl, fc = m.forecast_loss(C[:-1].detach(), bundles[1:], sig=0.0)   # the PFC's heads learn from the stream, not through it
            # THE DAY'S PLASTICITY GATED (wake_base, wake_dopa, wake_novel; 2026-09-15): the waking write into cortex at the ordinary
            # tick's share, raised by the dopamine of the moment and by the running surprise (the rewarded and the novel are written,
            # the rest weakly); at the defaults (1, 0, 0) the lesson is as before
            gate_ = float(self.cfg.get("wake_base", 1.0)) + float(self.cfg.get("wake_dopa", 0.0)) * abs(float(getattr(self, "_dopa", 0.0))) + float(self.cfg.get("wake_novel", 0.0)) * float(getattr(self, "_surp_run", 0.0))
            loss = (ll + fl) * (1.0 + self.stress / 10.0) * gate_      # stress raises plasticity (the day's; act_pred's plain step divides it out)
            if not bool(torch.isfinite(loss.detach())):
                return {"skipped": "non-finite"}
            loss.backward()
            if motor_:
                # EACH OPTIMIZER'S GRADIENT BOUNDED APART (the R6 verifier's fourth look): the day's parameters by their norm here,
                # act_pred's and the corrections' element by element in their plain step (_timing_step), so their labels' reliability
                # never sets the stream's step
                torch.nn.utils.clip_grad_norm_([p_ for g_ in self.opt_day.param_groups for p_ in g_["params"]], 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            self.opt_day.step()
            if motor_:
                # act_pred and the corrections: one plain step on the lesson's own gradient, the plasticity scale above divided out
                # (R6 fix 8, the R6 verifier's eighth look: kept, it made act_pred the one waking parameter stress reached, four times at
                # the ceiling, and at a hundred times the served rate its bound held every element and the arm diverged), each element bounded
                self._timing_step((1.0 + self.stress / 10.0) * gate_)
            out = {"latent_cos": round(lc, 3), "forecast_cos": round(fc, 3), "n_world": int(w.sum()), "tick": self.ticks}
            if motor_:
                out["motor"] = {k_: dict(v_, w=round(v_["w"], 4), w_own=round(v_["w_own"], 4), w_lab=round(v_["w_lab"], 4),
                                         s_lab=round(v_["s_lab"], 4), rel=round(v_["rel"], 4)) for k_, v_ in mrep.items()}           # the timing parts' lesson (step R6)
        finally:
            self.opt_day.zero_grad(set_to_none=True)
            if motor_:
                self.opt_pred.zero_grad(set_to_none=True)
            m.eval()
        self._wake_last = out
        return out

    def _sharp_calibrate(self, x):
        """THE CALIBRATED READOUT (2026-09-08): the sharpness is the one number in the readout that theory does not give, and a
        readout is well set when its confidence matches how often it is right. Each world symbol is a sample of the truth the
        forecast was read against: the gradient of the log-likelihood of what arrived, with respect to the sharpness, is the
        forecast's score of what arrived minus its expected score under its own reading (temperature scaling, Guo et al. 2017,
        as a running law). Over-confident readings are pushed flatter, under-confident ones sharper, by the body's own hits and
        misses; dopamine's sharpening (sharp_gain x mood) rides on the calibrated base. The reserved symbols are outside the
        readout's support, so a reserved arrival teaches nothing."""
        fc = getattr(self, "_fc_prev", None)
        if fc is None or x in self.bans:
            return
        m = self.m
        with torch.no_grad():
            s = max(1e-3, float(m.read_sharp))
            lg = m.readout(fc); q = lg / s
            lg = lg.clone(); lg[self.bans] = float("-inf")
            p = torch.softmax(lg, dim=0)
            grad = float(q[x] - (p * q).sum())
            lo, hi = float(self.cfg.get("sharp_min", 2.0)), float(self.cfg.get("sharp_max", 100.0))
            self.sharp_cal = float(min(hi, max(lo, self.sharp_cal + float(self.cfg.get("sharp_rate", 0.05)) * grad)))
