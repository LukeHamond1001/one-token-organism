"""the senses (a mixin of `Life`, body/life.py): the feelings' own recovery each tick (`_decay_feelings`); the tick's first two
phases, `_sense` (the world's frame, since step R9 of the core refactor the one its world shows, the diary's its symbol or its quiet off
the queue as always; the offset by the count; the reward: the anatomy's reward sources felt in their order on the tick's frame, the
face first, since step R3, docs/SIM_DESIGN.md 8.4) and `_hear` (the start
mark, the world's symbol enters the stream, the offset by the settle law, the sensed pace's end, the striatal events); the offset
itself (`_offset`: the last world position marked ended, the utterance memory's entry, the working memory's latch); the world's two
hands on the page (`type_text`, `set_face`; the DiaryWorld reaches them); and the frame this tick is lived on (`_tick_frame`).

Moved verbatim from body/life.py (review 2026-09-22 section 4, step 2). Not every attribute they touch is born in `Life.__init__`:
twenty are first set later, by a phase of the tick or by the night, and thirteen of those are read here with a `getattr` default."""
import torch

from .world import Frame


class SensesMixin:
    # ---------------- feelings ----------------
    def _decay_feelings(self):
        """feelings recover on the body's own clock (per tick), so its physiology does not change
        with the speed the serve happens to run at"""
        self.fatigue *= 0.5 ** (1.0 / float(self.cfg["fatigue_half_life"]))
        for st_ in getattr(self, "motor", ()):                         # step R6h: each motor effector's own fatigue, at the same half-life
            st_["fatigue"] = float(st_["fatigue"]) * 0.5 ** (1.0 / float(self.cfg["fatigue_half_life"]))
        self.stress *= 0.5 ** (1.0 / float(self.cfg["stress_half_life"]))
        self.mood *= 0.5 ** (1.0 / float(self.cfg["mood_half_life"]))

    def _offset(self, settled=True):
        """THE OFFSET (§2): the world's quiet after its utterance, once per pause. The last world position is
        marked ended, so the waking lesson's target there is the turn-end and not the next line's first letter;
        a dream ends where the cortex, so taught, expects the quiet. Nothing enters the stream, the bags and the
        mouth's context stand, and the store keeps only what the world said next: written there too, the quiet
        after a cue blended with the answer's memory and the mouth read junk (runs 45 and 46, day 1)."""
        if int(self.cfg.get("wm", 0)) and getattr(self.m, "stri_wm", 0):
            with torch.no_grad():                            # WORKING MEMORY latches at the world's utterance end (a salience event)
                self.m.wm_latch(self.m.striatum_read())
        f_ = self.anatomy.words.field                                  # the words' window field (the diary's "x"; step R4)
        for w in reversed(self.win):
            if w[f_] != self.sil:
                w["end"] = True; break
        if self._last_write is not None and not self.cfg.get("store_off"):
            self.store.mark_boundary(*self._last_write)                # the memory of the last symbol carries the boundary
        self._prev_slot = -1                                           # the utterance ended: the next symbol begins a new chain
        self.note_offset()                                             # and the slow context's utterance closes with it
        self._follow = None                                            # and the recall's episode is let go
        live_ = self._pace_mode() >= 2
        self._turn_open = bool(settled) or live_                       # THE TURN'S READINESS (gate_turn): a settled end opens the slot, a count's end does not
                                                                       # (under the sensed pace, M5: either kind of end opens the child's slot)
        rel_ = int(self.cfg.get("gate_ear_release", 1))
        if live_:
            # THE SENSED PACE (pace_sense 2): an end the body foresaw waits for the reply's readiness (M4, _pace_hear) before the held ear lets
            # go; an end by the pause outlasted (M2) lets it go at once. No trace, no gain: the ear is held or it is not.
            if settled:
                self._ready_E = self.ticks; self._pred_ready = None; self._d_max = 0.0
            else:
                self._ear_held = False; self._ready_E = None; self._pred_ready = None
        elif settled and rel_ > 1:
            self._ear_release_at = self.ticks + rel_ - 1               # THE REPLY'S READINESS (2026-09-20, item 48): released the tick after, when the forecast has moved from the rest to the reply
        if settled and rel_ == 1 and not live_:
            self._ear_trace = 0.0                                      # THE EAR STOPS RINGING AT THE UTTERANCE'S END (2026-09-19 15:50): the trace held the gate shut
                                                                       # after a finished line too (the answers at 2.5 s and half as many, day 343); the listener is
                                                                       # released when the utterance is perceived to have ended BY THE SETTLE LAW (the cortex expected
                                                                       # the quiet), and held when the count alone ended it (a long pause the cortex did not expect:
                                                                       # a person thinking; the copy at 16:10: released on the count too, three words over a line)
        if len(self._utt_cur) >= 2:
            # THE REWARD'S TAG ON THE LINE BEFORE (reward_gain; 2026-09-15): the smiles' dopamine felt since the last utterance was kept
            # raises that utterance's strength, so the night replays the rewarded exchanges more (dopamine tags what preceded it)
            rg = float(self.cfg.get("reward_gain", 0.0))
            if rg > 0.0 and self.utt_S:
                self.utt_S[-1] = float(self.utt_S[-1]) + rg * max(0.0, float(getattr(self, "_dopa_since_utt", 0.0)))
            self._dopa_since_utt = 0.0
            self._utt_serial += 1
            entry_ = 1.0
            if str(self.cfg.get("utt_entry", "flat")) == "felt" and int(self.cfg.get("tag_trace", 0)):
                # DEFECT 6 AND 7 FIXED (tag_trace, step R7c; body/core/frames.py `_tag_entry`): the received tag over the utterance, reaching
                # back from the 64 ticks after it, at a saved and bias-corrected running mean
                entry_ = self._tag_entry(float(getattr(self, "_utt_felt", 0.0)) / max(1, len(self._utt_cur)))
            elif str(self.cfg.get("utt_entry", "flat")) == "felt":
                # THE TAG AT ENTRY (utt_entry felt; 2026-09-22, item 50): every utterance had entered the night's draw at 1.0, so the night
                # replayed by recency alone and a line heard once, however new, had the same few dreams as the tenth hearing of a drill.
                # The hippocampus tags an experience at encoding by its novelty and the reward around it and replays the tagged more;
                # the store already writes each symbol at surprise x (1 + |dopamine|). The utterance enters at the mean of that over its
                # symbols, relative to the running mean over utterances (utt_entry_tau), so the average entry stays near 1.0.
                f_ = float(getattr(self, "_utt_felt", 0.0)) / max(1, len(self._utt_cur))
                mu_ = getattr(self, "_utt_felt_mu", None)
                mu_ = f_ if mu_ is None else float(mu_)
                entry_ = f_ / max(1e-6, mu_) if mu_ > 1e-6 else 1.0
                self._utt_felt_mu = mu_ + (f_ - mu_) / max(1.0, float(self.cfg.get("utt_entry_tau", 64)))
            self.utts.append(list(self._utt_cur)); self.utt_S.append(entry_); self.utt_N.append(self._utt_serial)   # the utterance kept whole, at its entry strength, in its turn
            cap = int(self.cfg.get("utt_cap", 4096))
            if len(self.utts) > cap:                                   # the weakest (the oldest, faded) gives way
                i = min(range(len(self.utt_S)), key=lambda k: self.utt_S[k]); del self.utts[i]; del self.utt_S[i]; del self.utt_N[i]
        self._utt_cur = []; self._utt_felt = 0.0

    def _sense(self, frame=None):
        """the world's frame (the diary's: its symbol, or its quiet, off the queue), the offset by the count, the reward felt from the
        anatomy's sources in their order (the face first, then the reward's other terms). THE WORLD'S FRAME (the core refactor's step R9,
        docs/SIM_DESIGN.md 8.2): `frame`, when the loop hands one in, or the one the life's world shows now (`world.frame()`: the diary's
        pops the queue here, where it was always read, and reads the face its hand holds, so nothing moves); kept as the world's `now`,
        the frame this tick is lived on (a channel of the world's frames observes it; the mouth reads it for the effectors). The words
        channel's symbol in it is the tick's world symbol (the rest where the frame names none); who sent it is the frame's truth."""
        m = self.m
        if frame is None:
            frame = self.world.frame()
        self.world.now = frame
        u = frame.obs.get(self.anatomy.words.name)
        u = self.sil if u is None else int(u)
        who = frame.truth.get("who", "") if u != self.sil else ""
        # THE OFFSET: the world quiet for offset_ticks after its utterance, once per pause, whatever the body is
        # saying meanwhile (with the body's silence required too, a babbling body never let it fire: run 41 held
        # two turn-end memories after six days)
        off = int(self.cfg.get("offset_ticks", 0)); settle_form = str(self.cfg.get("offset_form", "count")) == "settle"
        ps_ = self._pace_mode()                                    # under the sensed pace (2) the pause outlasted (M2, _pace_hear) replaces the count
        if u == self.sil and off > 0 and not self._offset_done and not settle_form and ps_ < 2 and self.ticks - self._last_world >= off:
            self._offset(settled=False); self._offset_done = True
        first_after_pause = (u != self.sil and self._offset_done)  # the first symbol after a perceived pause begins an utterance
        if u != self.sil:
            if ps_:
                self._pace_heard(ps_ >= 2)                          # the gap just ended is a heard event (before the world's last symbol moves)
            self._last_world = self.ticks; self._offset_done = False; self._turn_open = False; self._ear_release_at = None
        # THE FELT REWARD (the core refactor's step R3, docs/SIM_DESIGN.md 8.4): the anatomy's reward sources (body/core/anatomy.py),
        # felt in their declared order on this tick's frame and added one at a time in that order, the float order of the sum. The
        # diary's: the face, then the world's words (world_r, not over its own voice under world_mask), then the effort (cost_in_reward).
        judge, *others = self.anatomy.rewards
        # source 0, the world's judgment (the face: a change is felt; a held face is silence; easing off is not an event): its
        # feeling is the tick's felt event, and its term alone is the world's reward the ring and the actor's reliability read
        felt = judge.felt(frame, self)
        r = judge.term(felt)                                # the world's reward: the felt face, clipped like a press
        terms_ = {judge.name: r} if (getattr(self, "motor", None) or self._tags_on()) else None   # step R6h: each source's term this tick, for a body
                                                                             # with motor effectors (the born cry's pain: body/core/cord.py) or that
                                                                             # keeps the received tag (R7c-d)
        self._ring_r.append(r)
        if self._act_pending:                               # the actor's reliability: the reward of the ticks after each act, on its vote for that act
            H = int(self.cfg.get("actor_horizon", 16))
            for p_ in self._act_pending:
                p_[2] += r
            while self._act_pending and self.ticks - self._act_pending[0][0] >= H:
                t0_, v_, g_ = self._act_pending.popleft(); self._arel_update(v_, g_)
        for s_ in others:                                   # the reward's other terms, in their order; a silent source adds nothing
            v_ = s_.felt(frame, self)
            if v_ is not None:
                r += s_.term(v_)
                if terms_ is not None:
                    terms_[s_.name] = s_.term(v_)
        if terms_ is not None:
            self._terms_now = terms_
            if self._tags_on():
                self._rtag_now = self._received_tag(terms_)             # step R7c-d: what was felt this tick, the tag's received part
                if int(self.cfg.get("tag_trace", 0)):
                    self._tag_boosts()                                  # defect 6: the received tag reaching back onto the entries (body/core/frames.py)
        if int(self.cfg.get("own_store", 0)):
            thr = float(self.cfg.get("own_store_r", 1.0))
            if float(r) >= thr and not getattr(self, "_own_stored", False) and self.ticks - getattr(self, "_own_store_tick", -10 ** 9) >= int(self.cfg.get("own_store_gap", 40)):
                self._own_stored = True; self._own_store_tick = self.ticks; self._consolidate_own(float(r) * float(self.cfg.get("own_store_gain", 0.3)))
            elif float(r) < 0.5 * thr:
                self._own_stored = False
        return u, who, felt, r, off, settle_form, first_after_pause

    def _hear(self, u, r, felt, off, settle_form, first_after_pause):
        """the ear's half: the start mark, the world's symbol enters the stream and the store writes what surprised it, the offset by the settle law, the striatal events"""
        m = self.m
        # --- the ear's half: the world's symbol (or its quiet) enters ---
        if off > 0 and u != self.sil:
            # THE START MARK falls on the memory whose context holds the utterance's first symbol: the second
            # symbol's, whatever the pause did to the bag (a short pause left the first symbol a memory under a faded
            # context and a long one no memory at all, so the mark fell a symbol apart between the two, runs 53/54)
            if first_after_pause:
                self._start_armed = True; self._start_pending = False; self._seam_pending = True
            elif getattr(self, "_start_armed", False):
                self._start_pending = True; self._start_armed = False
                self.store.episode += 1                                    # a new utterance of the world's: the links it writes carry its tag
        fr_ = self._frames_on()                                              # step R7b: the words' error on the frame (a forecast made)
        fw_ = fr_ and getattr(self, "pred_prev", None) is not None
        C1, pred1, surp1, conf1 = self._step(u, 0, r=r, dopamine=getattr(self, "_dopa", 0.0))
        if fr_:
            self._fw_err = float(self._surp_tick) if fw_ else None
        if self._utt_cur and int(self.cfg.get("tag_trace", 0)):          # defect 6 (tag_trace): the utterance heard carries the tags felt over it
            self._utt_tag = max(float(getattr(self, "_utt_tag", 0.0)), float(self._rtag_now))
        ps_ = self._pace_mode()
        if settle_form and off > 0 and ps_ < 2:                              # THE EVENT'S END BY THE LAW: two running averages of the
            st_ = float(getattr(self, "_surp_tick", 0.0))                     # tick's surprise; the world stops, the surprise jumps and
            af_ = 1.0 / max(1.0, float(self.cfg.get("offset_fast", 4))); as_ = 1.0 / max(1.0, float(self.cfg.get("offset_slow", 64)))
            self._surp_fast = (1 - af_) * getattr(self, "_surp_fast", st_) + af_ * st_
            self._surp_slow = (1 - as_) * getattr(self, "_surp_slow", st_) + as_ * st_
            settled_ = self._surp_fast <= float(self.cfg.get("offset_settle", 0.5)) * max(1e-6, self._surp_slow)
            fp_ = float(self.cfg.get("offset_foresee", 0.0))
            if fp_ > 0.0 and u == self.sil and not settled_:
                # THE QUIET FORESEEN (offset_foresee; 2026-09-20, item 48): on the copy the surprise's averages never settled, at no line's end
                # and in no pause of twenty one-handed lines (the count ended every utterance at eight ticks), so the ear's release and the
                # turn's readiness, both keyed to a settled end, never came. The cortex is taught at every offset to expect the rest after a
                # complete utterance; its readout's probability of the rest at a quiet tick is that expectation read directly. A listener
                # projects the other's turn-end from what has been said; a pause mid-line, the quiet not foreseen, ends nothing.
                with torch.no_grad():
                    self._p_rest = float(torch.softmax(m.readout(pred1), 0)[self.sil])
                settled_ = self._p_rest >= fp_
            if u == self.sil and not self._offset_done and self.ticks - self._last_world >= 1 and \
                    (settled_ or self.ticks - self._last_world >= off):         # the law, or the senses' own adaptation as the floor
                self._offset(settled=bool(settled_)); self._offset_done = True   # a newborn's flat surprise still ends events by the count
        if ps_:
            self._pace_hear(u, pred1, ps_ >= 2)                              # THE SENSED PACE: the end foreseen or outlasted, the reply's readiness
        if u != self.sil and (float(self.cfg.get("explore_gain", 0.0)) > 0 or float(self.cfg.get("explore_choice", 0.0)) > 0):   # arousal follows novelty: a running surprise at the world's symbols
            a_ = 1.0 / max(1.0, float(self.cfg.get("explore_tau", 64)))
            self._surp_run = (1.0 - a_) * getattr(self, "_surp_run", 0.0) + a_ * float(surp1)
        stri = str(self.cfg.get("fast_input", "band")) == "striatum" and int(self.cfg.get("fast_rls", 0))
        if stri:
            if felt:
                m.striatum_push(2, 0 if felt > 0 else 1)          # the felt face is an event of the stream
            if u != self.sil:
                m.striatum_push(0, int(u))
            if self.anatomy.events:
                self._events_push(u)                               # step R7a: the tick's fired event lines (body/core/frames.py)
            self._z_now = m.stri_in()
        self._read_world = getattr(self, "_read", None)        # the recall as the world's symbol entered
        return C1, pred1, surp1, conf1, stri

    # ---------------- the hands ----------------
    def type_text(self, s, who=""):
        """the world's hand: each symbol enters the page in its turn, tagged with who typed it (the visitor page of 2026-09-11:
        the parent yields to a visitor it can see on the page; the tag is on the page, never inside)"""
        n = 0; who = str(who)[:8]
        for ch in s:
            i = self.anatomy.symbol(ch)
            if i is not None and i != self.sil and i not in self.reserved and len(self.queue) < 600:   # the reserved symbols are not typed
                self.queue.append(i); self.queue_who.append(who); n += 1
        return {"queued": n}

    def set_face(self, expr):
        self.face_now = max(-6.0, min(6.0, float(expr)))
        return {"you": self.face_now}

    def _tick_frame(self, u):
        """the frame this tick is lived on (the world's `now`, since `_sense` took it; step R9), or, where the tick's senses kept none,
        the diary's frame of the world's symbol `u` as the tick built it before R9 (only the words are read from it there)"""
        f = self.world.now
        return f if f is not None else Frame(self.ticks, {self.anatomy.words.name: u}, self.face_now)
