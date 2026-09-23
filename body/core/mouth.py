"""the mouth (a mixin of `Life`, body/life.py): imagination for choice (`_imagine_value`); the sensed turn-taking (pace_sense,
M1-M5: the partner's pace as running quantiles, the end foreseen or the pause outlasted, the ear held and let go, the reply's
readiness, the wait and the floor; `_pace_*`); the tick's `_choose` (whether to speak, then what) and `_act`; `_feel_and_learn`
(the feelings from dopamine, the gate's buffer and the turns of the gate's and the waking lesson); and the gate's lesson.

Moved verbatim from body/life.py (review 2026-09-22 section 4, step 2). The review's Reflexes object is not made here: it would
change method bodies, which this step does not."""
import math

import torch
import torch.nn.functional as F


class MouthMixin:
    def _imagine_value(self, first, h):
        """IMAGINATION FOR CHOICE: say `first`, then h-1 more symbols as the cortex would (greedy), on a copy of the window; the
        striatal critic's value of the imagined line (with the working-memory slot as it stands). Nothing in the body changes."""
        m = self.m
        with torch.no_grad():
            win = list(self.win); line = m.stri_line.clone(); sym = int(first); zero_read = torch.zeros(m.d, device=self.dev)
            face = torch.tensor([self.face_now / 6.0, 0.0], device=self.dev)
            said = []
            for step in range(int(h)):
                said.append(sym)
                win.append({"x": self.sil, "xo": sym, "face": face, "bundle": self.bands, "read": zero_read, "r": 0.0})
                if len(win) > m.window:
                    win = win[-m.window:]
                if step == int(h) - 1:
                    break
                xs, whos, faces, bundles, reads = self._window_tensors(win)
                C = m.stream(m.inputs(xs, whos, faces, bundles, reads))[-1]
                lg = m.readout(m.forecast(C, zero_read)); lg[self.sil] = float("-inf"); lg[self.bans] = float("-inf")
                sym = int(lg.argmax())
                if int(self.cfg.get("plan_boundary", 1)) and sym == self.space_id:   # the word as the unit only under the old rule
                    said.append(sym); break                       # the word ends: value the line here
            width = 2 * m.vocab + 3
            for sy in said:
                line = torch.roll(line, 1); line[0] = m.vocab + int(sy)      # its own symbols, kind 1
            z = m.stri_b.clone()
            for p_ in range(line.numel()):
                e = int(line[p_])
                if e >= 0:
                    z += m.stri_W[p_ * width + e]
            z = torch.relu(z)
            if getattr(m, "stri_wm", 0):
                z = torch.cat([z, m.wm_slot * m.wm_on])
            return float(m.fast_value(z))

    # ---------------- the sensed turn-taking (pace_sense) ----------------
    # THE SENSED TURN-TAKING (pace_sense; 2026-09-23, ops' pace design): the reflexes of items 41-48 counted ticks (the count's 8, the ear's
    # 0.9 a tick and its release at 9, the slot's 40, the drive's 1200), and a count is a fact about one typist at one speed. Each becomes a
    # percentile of this partner's own silences, learned one heard event at a time: a running quantile in log units, q <- q + eta (p - [ln x
    # <= q]), so a fast hand with 1-4 s pauses and a robot on 30 Hz frames are met the same way. Typing speed is never read. The body's own
    # latency (the reply's readiness, M4) stays counted in cortex steps. Mechanisms (since = ticks since the world's last symbol):
    #   M1 THE END FORESEEN: a quiet tick of an open utterance at which the readout's most likely symbol (the reserved masked) is the end
    #      symbol the offset teaches (end_id: the rest, or the turn-end token under end_symbol eot) (listeners project a turn's end from what
    #      has been said); no constant: sharpness and mood cannot move an argmax.
    #   M2 THE PAUSE OUTLASTED: once the silence is longer than P (the partner's 99th-percentile pause: since + 1 > P, the gap it will close
    #      already past P) it ends the utterance unforeseen and lets the ear go at once.
    #   M3 THE EAR HELD: any world symbol holds the ear at 1 (gain 1) until an end lets it go, however long the pause (vocal suppression
    #      while a partner's call is under way); the learned weight on the ear shuts the gate through the whole line.
    #   M4 THE REPLY READY: after a foreseen end at E, the ear lets go at the first t > E where the end symbol is not the argmax and the
    #      forecast's change d_t = 1 - cos(pred_t, pred_t-1) is at most ready_ratio of its largest since E (a reply is launched when its plan
    #      settles); a reply never ready lapses when the slot closes (since > R_lo), or before the returns are known when the pause is outlasted.
    #   M5 TURN, WAIT, ALONE: after either end the slot (since <= R_lo) has the whole floor; past it the ear reads 1 until R_hi, then fades
    #      to 0 at 2 R_hi, and the floor returns as gate_floor x clip(since / R_hi - 1, 0, 1) (an infant's reply window follows the
    #      caregiver's usual latency; vocalizing returns when it is left alone). The same drive rules the floor while a line is under way
    #      (nothing below R_hi; nothing before the returns are known), as the babble drive it replaces did, whatever gate_listen is.
    # THE EVENTS (the review of 2026-09-23): a gap in which an end was foreseen is a return, unless the world came back within P (a false
    # end, heard as the pause it was: one false 3-tick "return" had undone nineteen true ones, R_lo then fell and P behind it, the body
    # talking over lines and never answering). R_lo and R_hi settle from the first 1/eta returns (the step's own horizon) as their sample
    # quantiles, a return shorter than P struck as P grows (early on P is small, and a word's gap foreseen falsely would otherwise stand as
    # the partner's quickest return); until they settle the pause tracker hears every silence not foreseen, as before the first return, so
    # a misheard early return can neither set R_lo nor fence P beneath it. After that each is a running quantile: it follows a partner
    # in its fast direction within about 1/eta events, and in its slow one at eta x min(p, 1 - p) a step (R_lo up, R_hi down, P down).
    # At 1 (shadow) the trackers run and the rules are computed on a shadow end and logged, the served reflexes deciding; at 2 they decide.
    # The pace rides on the utterance's ends: with offset_ticks 0 (no ends) it is off whatever pace_sense says (_pace_mode).
    @staticmethod
    def _pace_day_new():
        """the day's instruments of the sensed pace (the night report's "pace", then begun afresh)"""
        return {"fore": 0, "pause": 0, "false": 0, "false_fore": 0, "ret": 0, "sil": 0, "rel": [], "lapsed": 0, "cancel": 0}

    @staticmethod
    def _pace_q(xs, p):
        """the sample p-quantile of xs, interpolated between the order statistics (the returns' settling)"""
        s = sorted(xs); h = (len(s) - 1) * float(p); i = int(math.floor(h))
        return s[i] if i + 1 >= len(s) else s[i] + (h - i) * (s[i + 1] - s[i])

    def _pace_mode(self):
        """the sensed pace in force: pace_sense (0 off, 1 shadow, 2 live), or 0 when offset_ticks is 0: the pace senses and decides the
        utterance's ends, and with the ends switched off nothing would ever let the held ear go"""
        return int(self.cfg.get("pace_sense", 0)) if int(self.cfg.get("offset_ticks", 0)) > 0 else 0

    def _pace(self):
        """(P, R_lo, R_hi) in ticks: the partner's usual longest pause, its quickest and its longest usual return (R_lo and R_hi None
        until a return is heard: no slot limit, no wait, the whole floor)"""
        pq = self._pq
        P = math.exp(float(pq["pause"]))
        if pq.get("lo") is None or pq.get("hi") is None:
            return P, None, None
        lo = math.exp(float(pq["lo"]))
        return P, lo, max(lo, math.exp(float(pq["hi"])))

    def _pace_update(self, g):
        """one heard event, the gap g (ticks) the world's symbol ended: a gap of one tick is the line's own rhythm, not a silence; a gap in
        which a foreseen end fired and the world stayed away at least P is a return (R_lo, R_hi); one it closed within P was a false end
        and is a pause; any other gap under R_lo is a pause (P), those the pause outlasted included, so the tracker never sees only the
        gaps under its own threshold. Until the returns have settled (1/eta of them past P, "warm" in the trackers) R_lo and R_hi are the
        sample quantiles of those heard, any shorter than P struck as P grows, and the pause tracker hears every silence not foreseen"""
        if g < 2:
            return
        pq = self._pq; d = self._pace_day; lx = math.log(float(g)); eta = float(self.cfg.get("pace_eta", 0.05))
        p_lo = float(self.cfg.get("pace_ret_lo", 0.05)); p_hi = float(self.cfg.get("pace_ret_hi", 0.95))
        P, lo, hi = self._pace()
        warm = pq.get("warm")
        fore = bool(self._gap_foreseen)
        if fore and g < P:
            fore = False; d["false_fore"] += 1                   # the end foreseen, and the world back within its usual longest pause: a pause
        if fore:
            pq["n_ret"] = int(pq.get("n_ret", 0)) + 1; d["ret"] += 1
            if warm is not None:
                warm.append(lx)
            else:
                pq["lo"] = float(pq["lo"]) + eta * (p_lo - (1.0 if lx <= float(pq["lo"]) else 0.0))
                pq["hi"] = float(pq["hi"]) + eta * (p_hi - (1.0 if lx <= float(pq["hi"]) else 0.0))
        elif warm is not None or lo is None or g < lo:
            pq["pause"] = float(pq["pause"]) + eta * (float(self.cfg.get("pace_pause_p", 0.99)) - (1.0 if lx <= float(pq["pause"]) else 0.0))
            pq["n_pause"] = int(pq.get("n_pause", 0)) + 1; d["sil"] += 1
            if self._gap_paused and lo is not None and g < lo:
                d["false"] += 1                                  # the pause outlasted, and the world back before its quickest return: a false end
        if warm is not None:
            warm[:] = [x for x in warm if x >= float(pq["pause"])]    # a return shorter than the partner's usual longest pause, as P has grown: struck
            if warm:
                pq["lo"] = self._pace_q(warm, p_lo); pq["hi"] = self._pace_q(warm, p_hi)
            else:
                pq["lo"] = None; pq["hi"] = None
            if len(warm) >= max(1, int(round(1.0 / max(eta, 1e-9)))):
                pq["warm"] = None                                # settled: from here the running quantiles

    def _pace_heard(self, live):
        """a world symbol: the gap it ends is a heard event (the day's first, from the night's sentinel, is not); the gap's labels begin
        afresh, a reply's readiness still pending is cancelled, and (live) the ear is held"""
        if self._last_world >= 0:
            self._pace_update(self.ticks - self._last_world)
        if self._ready_E is not None:
            self._pace_day["cancel"] += 1
        fq_ = float(self.cfg.get("pace_fore_q", 0.0))
        gm_ = getattr(self, "_gap_pend_max", None)
        if fq_ > 0.0 and gm_ is not None and not self._gap_foreseen and not self._gap_paused:
            # M1 BY ITS OWN MEASURE: a gap the world closed with no end in it was a mid-line pause; the highest log-probability of the end
            # the readout gave in it is one sample of what this body predicts mid-line for this partner (the running quantile, as P)
            pq = self._pq; eta = float(self.cfg.get("pace_eta", 0.05))
            n_warm = max(1, int(round(1.0 / max(1e-9, 1.0 - fq_))))      # the sample quantile needs 1/(1-q) pauses to see its tail: 100 at 0.99
            if pq.get("fore_q") is None:
                # THE WARM START (2026-09-23 05:15; the copies of 03:32): started from one sample and stepping 0.05 up, the running 99th
                # percentile sat low all day and ends were foreseen mid-line (4.4 words over a line). As the returns settle, the first
                # pauses are kept and the quantile is their sample quantile; the running quantile goes on from there.
                fw = pq.get("fore_warm")
                if not isinstance(fw, list):
                    fw = []; pq["fore_warm"] = fw
                fw.append(float(gm_))
                if len(fw) >= n_warm:
                    pq["fore_q"] = float(self._pace_q(fw, fq_)); pq["fore_warm"] = None
            else:
                pq["fore_q"] = float(pq["fore_q"]) + eta * (fq_ - (1.0 if gm_ <= float(pq["fore_q"]) else 0.0))
            pq["n_mid"] = int(pq.get("n_mid", 0)) + 1
        self._gap_pend_max = None
        self._gap_foreseen = False; self._gap_paused = False; self._sh_done = False
        self._ready_E = None; self._pred_ready = None; self._d_max = 0.0
        if live:
            self._ear_held = True

    def _pace_top(self, pred):
        """the readout's most likely symbol, the reserved masked but the end symbol (M1's and M4's reading; no sampling, no RNG)"""
        with torch.no_grad():
            lg = self.m.readout(pred).clone()
            lg[[b for b in self.bans if b != self.end_id]] = float("-inf")
            return int(lg.argmax())

    def _pace_release(self, live, since=None):
        """the reply is ready (M4), or its slot has passed: the pending readiness ends and (live) the held ear lets go"""
        if since is not None:
            rel = self._pace_day["rel"]; rel.append(int(since)); del rel[:-4096]
        self._ready_E = None; self._pred_ready = None; self._d_max = 0.0
        if live:
            self._ear_held = False

    def _pace_hear(self, u, pred1, live):
        """M1 and M2, the utterance's end (live: the offset itself; shadow: a shadow end, logged), then M4, the reply's readiness"""
        d = self._pace_day; since = self.ticks - self._last_world; top = None
        open_ = (not self._offset_done) if live else (not self._sh_done)
        if u == self.sil and open_ and since >= 1:
            top = self._pace_top(pred1); fore = top == self.end_id
            fq_ = float(self.cfg.get("pace_fore_q", 0.0))
            if fq_ > 0.0:
                with torch.no_grad():
                    lg_ = self.m.readout(pred1).clone(); lg_[[b for b in self.bans if b != self.end_id]] = float("-inf")
                    lp_ = float(torch.log_softmax(lg_, 0)[self.end_id])
                self._gap_pend_max = lp_ if getattr(self, "_gap_pend_max", None) is None else max(float(self._gap_pend_max), lp_)
                q_ = self._pq.get("fore_q")
                if not fore and q_ is not None and lp_ > float(q_):
                    fore = True                                                   # above what it predicts in 99 of the partner's 100 mid-line pauses
            if fore or since + 1 > self._pace()[0]:                             # M2: the gap this silence will close is already longer than P
                if fore:
                    self._gap_foreseen = True; d["fore"] += 1
                else:
                    self._gap_paused = True; d["pause"] += 1
                if live:
                    self._offset(settled=fore); self._offset_done = True      # a foreseen end sets the readiness pending (_offset); an outlasted one lets the ear go
                else:
                    self._sh_done = True
                    self._ready_E = self.ticks if fore else None; self._pred_ready = None; self._d_max = 0.0
        if self._ready_E is None:
            return
        P, lo, hi = self._pace()
        if (since > lo) if lo is not None else (since + 1 > P):
            d["lapsed"] += 1; self._pace_release(live)                          # the slot passed with the reply never ready (before the returns are known,
                                                                                # the pause outlasted): the wait rules the ear
        elif int(self.cfg.get("ready_law", 1)):
            if self.ticks > self._ready_E and self._pred_ready is not None:
                with torch.no_grad():
                    dt_ = float(1.0 - F.cosine_similarity(pred1, self._pred_ready, dim=0))
                self._d_max = max(self._d_max, dt_)
                if top is None:
                    top = self._pace_top(pred1)
                if top != self.end_id and dt_ <= float(self.cfg.get("ready_ratio", 0.5)) * self._d_max:
                    self._pace_release(live, since)
                    return
            self._pred_ready = pred1.detach()
        elif int(self.cfg.get("gate_ear_release", 1)) >= 1 and self.ticks >= self._ready_E + int(self.cfg.get("gate_ear_release", 1)) - 1:
            self._pace_release(live, since)                                     # ready_law 0: the fixed release (gate_ear_release, 1 = at the end's own tick) after the end,
                                                                                # the body's recall time; 0 = none, the slot's close lets it go

    def _pace_wait(self, since, lo, hi):
        """M5's wait: the ear's input after the end with the ear let go: 0 in the slot (since <= R_lo), 1 until R_hi, then 2 - since / R_hi,
        0 from 2 R_hi (none before the first return)"""
        if lo is None or since <= lo:
            return 0.0
        if since <= hi:
            return 1.0
        return max(0.0, 2.0 - float(since) / hi)

    def _pace_ear(self, u):
        """M3 and M5: the ear's world input to the learned gate under the sensed pace (no trace, no gain)"""
        if u != self.sil or self._ear_held:
            return 1.0
        if not self._offset_done:
            return 0.0
        P, lo, hi = self._pace()
        return self._pace_wait(self.ticks - self._last_world, lo, hi)

    def _pace_floor(self, fl, since):
        """M5: after the end, the whole floor in the child's slot (since <= R_lo, or before the first return), then the drive, gate_floor x
        clip(since / R_hi - 1, 0, 1): nothing while the partner is merely slow, whole again by 2 R_hi. While the world's line is under way
        the drive alone (nothing below R_hi, nothing before a return is known): the babble drive it replaces held the floor at zero as the
        world spoke, so the line's floor does not rest on gate_listen (the review of 2026-09-23: 0.05 on a slow line at gate_listen 0)"""
        P, lo, hi = self._pace()
        if not self._offset_done:
            return 0.0 if hi is None else fl * min(1.0, max(0.0, float(since) / hi - 1.0))
        if lo is None or (getattr(self, "_turn_open", False) and since <= lo):
            return fl
        return fl * min(1.0, max(0.0, float(since) / hi - 1.0))

    def _pace_report(self):
        """the sensed pace's instruments: the trackers now, the day's ends, false ends, returns, the reply's releases, the ear's weight"""
        P, lo, hi = self._pace(); d = self._pace_day; rel = sorted(d["rel"]); n = len(rel)
        g = self.m.mouth_gate.weight; ew = round(float(g[0, self.m.d + 5].detach()), 3) if g.shape[1] >= self.m.d + 7 else None
        return {"mode": self._pace_mode(), "P": round(P, 2), "R_lo": (round(lo, 2) if lo is not None else None),
                "R_hi": (round(hi, 2) if hi is not None else None), "pauses_heard": int(self._pq.get("n_pause", 0)), "returns_heard": int(self._pq.get("n_ret", 0)),
                "returns_settled": self._pq.get("warm") is None,
                "ends_foreseen": d["fore"], "ends_by_pause": d["pause"], "false_pause_ends": d["false"], "false_foreseen_ends": d.get("false_fore", 0),
                "returns": d["ret"], "silences": d["sil"],
                "releases": n, "release_median": ((rel[(n - 1) // 2] + rel[n // 2]) / 2.0 if n else None), "release_lapsed": d["lapsed"],
                "release_cancelled": d["cancel"], "ear_w": ew}

    def _choose(self, C1, pred1, u, level, stri):
        """the mouth's half: whether to speak (the gate), then what (the readout at the mood's sharpness; the actor's chunk, plan or vote)"""
        m = self.m
        # --- the mouth's half: whether (the gate), then what (the lexicon) ---
        # DECISIVENESS from tonic dopamine (songbirds: variability is high when unrewarded and falls as
        # reward comes; mood is the body's tonic dopamine): the readout's sharpness = base + gain x mood/6
        # THE MOUTH'S DECISIVENESS (the spec's law, both sides of zero: 25 x (1 + mood/6); the code had clamped a bad day at zero, so it
        # never widened the babble: the review of 2026-09-08), floored where the lexicon's own noise wins (sharp_min 8: below about ten
        # a symbol at probability 0.5 no longer outweighs fifty strangers at their noise). The base is the readout's anatomy under the
        # fixed and world forms; under "calibrated" it is the world-calibrated base, which on the served body fell 25 -> 8 in three
        # minutes (2026-09-08, 20:47): the world's next symbol is far less predictable than the mouth's own, so the world's calibration
        # cannot set the mouth's decisiveness (perception and production are two readouts in biology too). Under "world" the
        # calibration runs as a reading and sets REM's sampling temperature: the dreams as varied as the world proved to be.
        base = float(self.sharp_cal) if str(self.cfg.get("sharp_form", "fixed")) == "calibrated" else float(self.cfg["sharp_base"])
        ratio = float(self.cfg["sharp_gain"]) / max(1e-6, float(self.cfg["sharp_base"]))
        m.read_sharp = max(float(self.cfg.get("sharp_min", 8.0)), base * (1.0 + ratio * max(-6.0, min(6.0, self.mood)) / 6.0))
        with torch.no_grad():
            sal = float(self.cfg["gate_salience"]) * float(pred1.norm())      # the proposal's salience
            feat = torch.cat([C1.detach() / math.sqrt(float(m.d)),
                              torch.tensor([self.fatigue / 10.0, self.mood / 6.0, self.stress / 10.0, sal, level], device=self.dev)])
            if int(self.cfg.get("gate_center", 0)):
                # THE ADAPTED INPUT: the gate's inputs relative to their running mean
                if getattr(self, "_feat_mu", None) is None or self._feat_mu.shape != feat.shape:
                    self._feat_mu = torch.zeros_like(feat)
                d_mu = (feat - self._feat_mu) / float(self.cfg.get("gate_center_tau", 1024))
                self._feat_mu += d_mu
                if int(self.cfg.get("gate_center_keep", 0)):
                    # THE FUNCTION KEPT UNDER THE MOVING MEAN (2026-09-05): the gate's function is w . x + c with c the
                    # uncentered intercept the lesson owns; the bias the centered forward pass uses is c + w . mu,
                    # recomputed from the current w and mu at every tick, so centering changes the lesson's coordinates
                    # and never the function. (The first form added w . d_mu to the bias each tick, which keeps the
                    # function only while w stands still; as the lesson moved w the increments stopped summing to
                    # w . mu and runs 131/132 drifted into a gate pointing against its mean feature, w . mu −4.1.)
                    # Without it a body switched to the adapted input mid-life lost w . mu (−2.06 on the served body's
                    # day 44) and opened its gate until its lesson refit.
                    wmu = float(m.mouth_gate.weight[0, : self._feat_mu.numel()] @ self._feat_mu)
                    if getattr(self, "_gate_c", None) is None:
                        # born: mu is near zero and c is the bias; loaded: the saved bias was c + w . mu of the saved mean
                        self._gate_c = float(m.mouth_gate.bias[0]) - wmu
                    else:
                        # the lesson may have moved the bias since the last tick: what it moved is the intercept's
                        self._gate_c += float(m.mouth_gate.bias[0]) - self._gate_wmu_last - self._gate_c
                    m.mouth_gate.bias.fill_(self._gate_c + wmu); self._gate_wmu_last = wmu
                feat = feat - self._feat_mu
            live_ = self._pace_mode() >= 2
            if int(self.cfg.get("gate_ear", 0)) and live_:
                # THE EAR UNDER THE SENSED PACE (M3, M5): held at 1 from the world's symbol until an end lets it go (M2 at once, M4 when the
                # reply is ready), then the wait; no trace, no gain
                ear_w = self._pace_ear(u); self._ear_now = ear_w
                feat = torch.cat([feat, torch.tensor([ear_w, 1.0 if getattr(self, "_acted_last", False) else 0.0], device=self.dev)])
            elif int(self.cfg.get("gate_ear", 0)):
                # THE EAR: the world's symbol this tick, its own act last tick (sensed, not inferred)
                ear_w = 1.0 if u != self.sil else 0.0
                ed_ = float(self.cfg.get("gate_ear_decay", 0.0))
                ra_ = getattr(self, "_ear_release_at", None)
                if ra_ is not None and self.ticks >= ra_:
                    self._ear_trace = 0.0; self._ear_release_at = None   # the ear released the tick after the perceived end (gate_ear_release 2)
                if ed_ > 0.0:
                    # THE EAR'S TRACE (gate_ear_decay; 2026-09-19, the review and the diagnostic probe of item 45): with the ear reading
                    # the tick alone, the learned gate (its ear weight -49 on a symbol's tick, +14 on its own act) was shut on the ticks a
                    # symbol arrived and free on the quiet ticks between a slow typist's keystrokes, and every word said over a
                    # one-handed line came through it with the floor shut. A sense persists past its stimulus (the auditory trace,
                    # the simplest sensory memory): the ear's world input decays by gate_ear_decay a tick from each symbol instead of
                    # falling to zero, so the gate's learned weight keeps it shut while a person is still typing at any pace and
                    # frees it as the ringing fades. A disclosed constant; nothing about content; the typist's fast lines unchanged.
                    self._ear_trace = max(ed_ * float(getattr(self, "_ear_trace", 0.0)), ear_w); ear_w = self._ear_trace
                ear_w = ear_w * float(self.cfg.get("gate_ear_gain", 1.0))     # THE EAR'S GAIN (2026-09-20): the learned weight on the ear (-49, built over weeks) moved by 0.05 in five days of frowns; the input's scale is the constant that sets how hard the ringing ear holds the gate
                feat = torch.cat([feat, torch.tensor([ear_w, 1.0 if getattr(self, "_acted_last", False) else 0.0], device=self.dev)])
            z = m.mouth_gate(feat.unsqueeze(0))[0, 0] / (1.0 + self.stress / 10.0)   # stress flattens the choice
            gy_ = float(self.cfg.get("gate_yield", 0.0))
            if gy_ > 0.0 and not live_:                                       # under the sensed pace M5's wait holds the gate instead
                # THE YIELD (gate_yield; 2026-09-20, the user's word: no blabber between the lines of a conversation): after the
                # world's line the child has its slot (gate_yield_after ticks); past it, with the world still quiet, the learned gate
                # is held by gate_yield, the hold fading over gate_quiet_tau as the babble drive returns, so a child alone for minutes
                # babbles again. Turn-taking: I answer, then I wait for you. The silence probe of 11:20 on the copy: 27 words in a
                # 240-tick silence after an answer, none of them through the floor, the learned gate running on its own answer.
                since_ = self.ticks - self._last_world
                after_ = int(self.cfg.get("gate_yield_after", 40))
                if since_ > after_:
                    tau_y = float(self.cfg.get("gate_quiet_tau", 0) or 300)
                    z = z - gy_ * (1.0 - min(1.0, float(since_ - after_) / tau_y))
            fl = float(self.cfg["gate_floor"])
            # THE LISTENING REFLEX (gate_listen; 2026-09-17, item 41): the learned gate had shut itself during the parent's lines (the ear's
            # weight -48) and the talk-overs came from what no learned weight reaches, the spontaneous floor starting a word on a
            # twentieth of the ticks and the chunk then running it with no gate decision. While the world's utterance is open (from its
            # symbol until the offset, the event's end the body computes) the floor is scaled by (1 - gate_listen) and a running word
            # is cut: the vocal suppression while hearing speech, innate before turn-taking is learned; the learned weights still decide.
            listening = float(self.cfg.get("gate_listen", 0.0)) > 0.0 and not self._offset_done and self.ticks - self._last_world < 10 ** 6
            if listening:
                fl = fl * (1.0 - float(self.cfg.get("gate_listen", 0.0)))
            qt_ = int(self.cfg.get("gate_quiet_tau", 0))
            if live_:
                fl = self._pace_floor(fl, self.ticks - self._last_world)        # M5: the slot's whole floor, then the drive, the line's floor too (the babble drive, the turn and the sure proposal retired)
            elif qt_ > 0:
                # THE BABBLE DRIVE (gate_quiet_tau; 2026-09-17, item 41, the user's word: "talk less until the teacher leaves it alone long
                # enough"): the urge to vocalize on its own returns with silence; the spontaneous floor is zero as the world speaks and
                # rebuilds toward gate_floor over gate_quiet_tau ticks of the world's silence. The learned gate is untouched: a sure proposal
                # (an answer) opens it whatever the floor, as the trace of 2026-09-17 showed (p_act 0.99 a tick after the line, the floor 0).
                ramp = min(1.0, max(0.0, float(self.ticks - self._last_world)) / float(qt_))
                if int(self.cfg.get("gate_turn", 0)) and getattr(self, "_turn_open", False) and self._offset_done and \
                        self.ticks - self._last_world <= int(self.cfg.get("gate_yield_after", 40)):
                    # THE TURN'S READINESS (gate_turn; 2026-09-20, item 48): the sure proposal opened the floor on nearly every forecast (the
                    # norm passes 0.45 and 0.9 alike), and its tries were the words in a one-handed line's pauses and the babble in a thinking
                    # silence, as well as what started a turn. The readiness to respond follows the other's utterance being perceived as
                    # complete (the settle law: the cortex expected the quiet), for the slot's length; a pause the count ended (a person
                    # thinking mid-line) opens no turn, and past the slot the drive's ramp rules the silence. Reads no content.
                    ramp = 1.0
                    fl = max(fl, float(self.cfg.get("gate_turn_floor", 0.0)))   # THE TURN'S FLOOR: the reply's readiness above the resting floor (day 358: the answers 69 percent, the gate alone too seldom opening in the slot)
                sure_ = float(self.cfg.get("gate_quiet_sure", 0.0))
                if sure_ > 0.0:
                    # THE SURE PROPOSAL (gate_quiet_sure; 2026-09-17, 21:20, read in the chair): the drive held the floor at zero for the first
                    # minute after a question and the answers went unsaid; the learned gate opened (p 0.99 the tick after the line) but the
                    # mouth then sampled the rest, and it was the floor's forty tries a turn that had let an answer out. The readiness to act
                    # rises with the strength of the proposal (the striatum driven harder by a stronger cortical input): a forecast whose
                    # norm reaches gate_quiet_sure has the floor whole at once, a flat one waits for the silence to rebuild it.
                    ramp = max(ramp, min(1.0, float(pred1.norm()) / sure_))
                fl = fl * ramp
            eg_ = float(self.cfg.get("explore_gain", 0.0))
            if eg_ > 0:                                                             # THE EXPLORATION DRIVE: readiness to act, not a
                fl = min(0.5, fl + eg_ * getattr(self, "_surp_run", 0.0))            # reward; the floor climbs where the world surprises
            self._floor_now = fl
            p_act = fl + (1.0 - fl) * float(torch.sigmoid(z))                          # spontaneous activity as the floor
            acted = bool(torch.rand(1, generator=self.gen).item() < p_act)
            # DECISIVENESS BY CERTAINTY (sharp_conf; 2026-09-15, the live ruler): each word's first symbol is sampled from this readout, and
            # at a fixed sharpness a three-word answer needed three lucky starts where the forecast's margin was thin (the live mouth
            # answered 3 of 30 questions the greedy readout answered 20 of). A selection's noise falls as its evidence rises (the
            # basal ganglia's threshold; a decision's variance at the bound); the forecast's norm is its certainty, the same the
            # gate's salience reads. The sharpness here is the readout's times (1 + sharp_conf x that norm): a sure forecast is read
            # decisively, an unsure one as before. The readouts of the probes, the gauge and the dreams are untouched.
            sc_ = float(self.cfg.get("sharp_conf", 0.0)); self._sharp_eff = float(m.read_sharp) * (1.0 + sc_ * float(pred1.norm()))
            logits = m.readout(pred1).clone() * (1.0 + sc_ * float(pred1.norm()))
            act_on = bool(int(self.cfg.get("actor", 0)) and stri and getattr(self, "_z_now", None) is not None)
            self._cands_now = None
            if int(self.cfg.get("actor", 0)) and str(self.cfg.get("actor_form", "add")) == "softmax":
                # THE CHOOSER AT A TORN MOMENT: the candidates are the mouth's top few (the cortex's forecast with the recall in it) and
                # the cortex's own top two; where the best two lie within the margin the chooser votes among them, a softmax of its
                # scores on the cortex's state (the running mean taken out, as a key's would be), at the earned gain
                with torch.no_grad():
                    act_on = True
                    c_ = C1.detach().float(); self._c_n += 1; a_ = max(1.0 / self._c_n, 1.0 - 0.9995); self._c_mu = self._c_mu + a_ * (c_ - self._c_mu)
                    za = F.normalize(c_ - self._c_mu, dim=0) * float(self.cfg.get("key_scale", 2.5)); self._za_now = za
                    spk0 = logits.clone(); spk0[self.sil] = float("-inf"); spk0[self.bans] = float("-inf")
                    kk = int(self.cfg.get("chooser_k", 4)); top = spk0.topk(kk).indices.tolist()
                    lc0 = m.readout(m.forecast(C1, torch.zeros_like(pred1))); lc0[self.sil] = float("-inf"); lc0[self.bans] = float("-inf")
                    top2 = lc0.topk(2).indices.tolist()
                    cands = sorted(set(int(i) for i in top + top2 if spk0[int(i)] > float("-inf")))
                    vals = spk0[cands]; torn = len(cands) > 1 and bool((vals.max() - vals.topk(2).values[-1]) <= float(self.cfg.get("actor_margin", 4.0)))
                    if torn:
                        sc = m.chooser(za)[cands] / float(self.cfg.get("actor_temp", 1.0)); pa = torch.softmax(sc, 0)
                        vote = torch.log(pa + 1e-9) - math.log(1.0 / len(cands))          # zero-mean over the candidates
                        ab = torch.zeros_like(logits); ab[cands] = vote; self._a_bias_now = ab
                        self._cands_now = cands; self._pa_now = pa
                        if str(self.cfg.get("actor_voice", "off")) == "earned" and self._arel_gain > 0.0:
                            logits[cands] = logits[cands] + float(self._arel_gain) * float(self.cfg.get("actor_beta", 1.0)) * vote
                    else:
                        self._a_bias_now = None
            elif act_on:
                with torch.no_grad():                                  # the striatum disposes: its bias on the cortex's proposal
                    a_bias = float(self.cfg.get("actor_beta", 1.0)) * torch.tanh(m.actor(self._z_now))
                    self._a_bias_now = a_bias
                    form_ = str(self.cfg.get("actor_form", "add"))
                    if str(self.cfg.get("actor_voice", "off")) == "earned" and form_ not in ("add", "select") and self._arel_gain > 0.0:
                        logits = logits + float(self._arel_gain) * a_bias                 # the earned voice: as loud as it has proved right
                    spk = logits.clone(); spk[self.sil] = float("-inf"); spk[self.bans] = float("-inf")   # the speakable proposals: not the rest, not the reserved
                    if form_ == "select":
                        short = spk >= (spk.max() - float(self.cfg.get("actor_margin", 4.0)))   # the cortex's shortlist
                        logits = torch.where(short, logits + a_bias, torch.full_like(logits, float("-inf")))
                    elif form_ in ("plan", "chunk"):
                        # THE BOUNDARY (plan_boundary 1): the tick after a pause in its own speech, or its last symbol the space (a
                        # fact about text written in). plan_boundary 0 (2026-09-08): no symbol, no pause test; the planner runs
                        # whenever it is about to act and the cortex is torn (more than one candidate within the margin), which is
                        # what the shortlist test below already asks. Inside a word the cortex is rarely torn; at a word's start it is.
                        # Deliberation where there is doubt: body-general, and it carries to a body without a space.
                        if int(self.cfg.get("plan_boundary", 1)) or form_ == "chunk":   # the chunk form deliberates at the word's start only
                            boundary = (not getattr(self, "_acted_last", False)) or getattr(self, "_own_last", None) in (None, self.space_id)
                        else:
                            boundary = True
                        if boundary and acted:                                 # imagination only when it is about to speak
                            short = spk >= (spk.max() - float(self.cfg.get("actor_margin", 4.0)))   # within the margin of the best speakable
                            cands = [int(i) for i in torch.nonzero(short).flatten().tolist() if int(i) != self.sil and int(i) not in self.bans]
                            if len(cands) > int(self.cfg.get("plan_k", 4)):        # the cortex's top few, as many as a choice can weigh
                                cands = sorted(cands, key=lambda c: -float(logits[c]))[: int(self.cfg.get("plan_k", 4))]
                            self._torn_now = len(cands) > 1
                            if len(cands) > 1:
                                vals = {c: self._imagine_value(c, int(self.cfg.get("plan_h", 4))) for c in cands}
                                self._plan_last = {"cands": cands, "vals": vals, "cortex": {c: float(logits[c]) for c in cands}}
                                planned = torch.full_like(logits, float("-inf"))
                                for c in cands:
                                    planned[c] = logits[c] + float(self.cfg.get("plan_beta", 4.0)) * vals[c]
                                ec_ = float(self.cfg.get("explore_choice", 0.0)); temp_ = 1.0
                                if ec_ > 0:                                     # THE DRIVE IN THE CHOICE: where the world is new the choice
                                    temp_ = 1.0 + ec_ * float(getattr(self, "_surp_run", 0.0))   # among the candidates widens; familiar, it narrows
                                    for c in cands:
                                        planned[c] = planned[c] / temp_
                                self._plan_last["temp"] = round(temp_, 3)
                                logits = planned
                    else:
                        logits = logits + a_bias
            if self.cfg.get("end_rest"):
                # THE END IS A REST: the forecast's vote for the turn's end (a symbol the mouth can never say) is its
                # vote for silence; banned outright, a sure forecast of the end raised the proposal's salience and then
                # the next-best symbol was said in its place
                if self.end_id != self.sil:
                    logits[self.sil] = logits[self.eot]                   # under the rest form the vote for the end is the rest's own logit
            else:
                logits[self.sil] = float("-inf")
            logits[self.bans] = float("-inf")
            probs = torch.softmax(logits, -1)
            ent = float(-(probs * (probs + 1e-9).log()).sum() / math.log(probs.numel())); self._ent_now = ent
            chunk_form = act_on and str(self.cfg.get("actor_form", "add")) == "chunk"
            self._chunk_cont = False
            if chunk_form and getattr(self, "_acted_last", False) and getattr(self, "_own_last", None) not in (None, self.space_id) \
                    and getattr(self, "_chunk_len", 0) < int(self.cfg.get("chunk_max", 12)) and not listening:   # a running word is cut while the world speaks
                # THE CHUNK RUNS: inside a word (its last own symbol not the space, its turn unbroken) the cortex's own continuation is
                # said, the most likely symbol, with no gate decision (p_act 1: nothing to credit) and no sampling; the word ends at
                # the space or at the rest (the turn's end); chunk_max symbols force a new decision
                if int(self.cfg.get("chunk_gate", 0)):
                    # THE CONTINUATION GATED (chunk_gate; 2026-09-22, item 50): the review found the face's credit could not reach a word
                    # under way: its letters ran with no decision (p 1, so the lesson's (act - p) was 0 for all of them) and the talked-over
                    # frown, felt at the word's end, reached only its first symbol five to ten ticks back. An action under way is itself
                    # gated (the stop pathway: cortex to subthalamus halts a program in progress): here the learned gate's own draw of this
                    # tick decides whether the program goes on, the letter still the cortex's continuation; a no stops the word.
                    if acted:
                        self._chunk_cont = True
                        nxt = int(torch.argmax(logits)); p_choice = float(probs[nxt]); self._chunk_len = getattr(self, "_chunk_len", 0) + 1
                        self._chunk_ticks = getattr(self, "_chunk_ticks", 0) + 1
                        if nxt == self.sil:
                            acted, p_choice = False, 0.0
                    else:
                        nxt, p_choice = self.sil, 0.0
                        self._chunk_stops = getattr(self, "_chunk_stops", 0) + 1
                else:
                    acted = True; p_act = 1.0; self._chunk_cont = True
                    nxt = int(torch.argmax(logits)); p_choice = float(probs[nxt]); self._chunk_len = getattr(self, "_chunk_len", 0) + 1
                    self._chunk_ticks = getattr(self, "_chunk_ticks", 0) + 1
                    if nxt == self.sil:
                        acted, p_choice = False, 0.0
            elif acted:
                nxt = int(torch.multinomial(probs.cpu(), 1, generator=self.gen))
                p_choice = float(probs[nxt])
                if nxt == self.sil:
                    acted, p_choice = False, 0.0                  # it chose the rest: the turn is the other's
                elif chunk_form:
                    self._chunk_len = 1                           # a word begins: the act; its letters follow as a program
                    if nxt != self.space_id:
                        self._chunk_words = getattr(self, "_chunk_words", 0) + 1
            else:
                nxt, p_choice = self.sil, 0.0
            self._last_choice = {"p_act": float(p_act), "acted": bool(acted), "nxt": int(nxt), "p_choice": float(p_choice), "norm": float(pred1.norm()),
                                 "top": int(torch.argmax(logits)), "sharp": float(getattr(self, "_sharp_eff", m.read_sharp))}   # the tick's choice, for the instruments
        return acted, nxt, p_act, p_choice, probs, feat, ent, act_on

    def _act(self, u, felt, stri, gam, delta, acted, nxt, p_act, p_choice, probs, feat, act_on):
        """the act: the actor's credit, the intrinsic credit, its own symbol (or its rest) enters the stream, the gate's tag"""
        m = self.m
        int_t = 0.0
        if acted and act_on and not self._chunk_cont:            # the actor's act and credit: once per word under the chunk form
            self._ring_torn.append(1.0 if self._torn_now else 0.0); self._ring_ent.append(float(self._ent_now)); self._torn_now = False
            if getattr(self, "_a_bias_now", None) is not None:
                ab_ = self._a_bias_now
                self._act_pending.append([self.ticks, float(ab_[nxt]), 0.0])   # its vote for the act taken; the reward that follows is gathered
                spk_ = ab_.clone(); spk_[self.sil] = float("-inf"); spk_[self.bans] = float("-inf")
                self._act_agree.append(1.0 if int(spk_.argmax()) == nxt else 0.0)
            with torch.no_grad():                                      # the actor's eligibility: what it said against what it expected, on this input
                if str(self.cfg.get("actor_form", "add")) == "softmax":
                    self._chooser_credit(nxt, float(gam[int(self.cfg["dopamine_band"])]))
                else:
                    oh = torch.zeros_like(probs); oh[nxt] = 1.0
                    e_new = torch.outer(oh - probs.detach(), self._z_now)
                    ea = getattr(self, "_e_actor", None)
                    self._e_actor = (float(gam[int(self.cfg["dopamine_band"])]) * ea if ea is not None else torch.zeros_like(e_new)) + e_new
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
        self._own_last = int(nxt) if acted else None
        if stri:
            if acted:
                m.striatum_push(1, int(nxt))                      # its own symbol is an event of the stream
            elif u == self.sil and not felt and int(self.cfg.get("stri_quiet", 0)):
                m.striatum_push(3, 0)                             # a tick of quiet is an event too (the line carries time)
        self._acted_last = bool(acted)
        if float(self.cfg.get("gate_slow_lr", 0.0)) > 0.0:
            with torch.no_grad():
                g_ = float(self.cfg.get("vcrit_gamma", 1.0 - 1.0 / 1024))
                tag_in = torch.cat([feat.detach(), torch.ones(1, device=self.dev)])
                prev = getattr(self, "_gate_tag", None)
                self._gate_tag = ((g_ * prev) if prev is not None else torch.zeros_like(tag_in)) + (float(acted) - p_act) * tag_in
        return int_t

    def _feel_and_learn(self, delta, delta_slow, delta_long, feat, acted, int_t, p_act):
        """the feelings from dopamine; the gate's buffer and its lesson; the waking cortex lesson"""
        m = self.m
        # --- feelings from dopamine ---
        self.mood = max(-6.0, min(6.0, self.mood + float(self.cfg["mood_gain"]) * delta))
        self.stress = min(30.0, self.stress + float(self.cfg["stress_gain"]) * max(0.0, -delta))
        if abs(delta) >= float(self.cfg["burst"]):
            self.n_bursts += 1
        # --- the gate's buffer and lesson ---
        if str(self.cfg.get("vcrit_ceiling", "fixed")) == "earned" and int(self.cfg.get("vcrit_auto", 0)):
            vw = float(self._vrel_gain)                                   # the voice is exactly as loud as it has proved right
        else:
            vw = float(self.cfg.get("vcrit_w", 0.0)) * (self._vrel_gain if int(self.cfg.get("vcrit_auto", 0)) else 1.0)
        self._vw_now = vw
        self.gate_buf.append([feat.cpu(), acted, delta + float(self.cfg["gate_slow_w"]) * delta_slow + vw * delta_long, int_t, self.fatigue,
                              float(m.r_tr[int(self.cfg.get("gate_tonic_clock", 4))]), p_act])   # the felt-reward trace at the tick, for the drive; the probability it acted with
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
                drive_t = tonic + (float(self.cfg.get("gate_tonic_rate", 0.0)) * float(buf[t][5]) if len(buf[t]) > 5 else 0.0)   # THE DRIVE FOLLOWS THE REWARD RATE
                g += drive_t + w_int * float(buf[t][3]) - (0.0 if self.cfg.get("cost_in_reward") else cost * (1.0 + (float(buf[t][4]) / f0) ** 2))
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
        # the probability the gate actually acted with (stress divisor and all), carried in the buffer (review 2026-09-06: recomputed
        # here without the divisor, (act - p) was biased with stress); older samples without it fall back to the recomputation
        p = torch.tensor([float(b[6]) if len(b) > 6 else float("nan") for b in buf[:n]], device=self.dev)
        p = torch.where(torch.isnan(p), (fl + (1.0 - fl) * torch.sigmoid(z)).detach(), p)
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
        for _ in range(min(len(self.gate_buf), n)):                 # the samples the lesson consumed (review: popping gate_every left a third to be learned twice)
            self.gate_buf.popleft()
