"""the mouth (a mixin of `Life`, body/life.py): imagination for choice (`_imagine_value`); the sensed turn-taking (pace_sense,
M1-M5: the partner's pace as running quantiles, the end foreseen or the pause outlasted, the ear held and let go, the reply's
readiness, the wait and the floor; `_pace_*`); the tick's `_choose` (whether to speak, then what) and `_act`; `_feel_and_learn`
(the feelings from dopamine, the gate's buffer and the turns of the gate's and the waking lesson); and the gate's lesson.

Moved verbatim from body/life.py (review 2026-09-22 section 4, step 2). The review's Reflexes object is not made here: it would
change method bodies, which this step does not.

THE EFFECTORS (the core refactor's step R5, docs/SIM_DESIGN.md 8.4): `_choose`, `_act` and the gate's lessons run over the anatomy's
effectors. The voice is effector 0 and keeps its code and names (its ear, now the gate inputs it declares, is `_voice_ear`); `_choose`
also returns its gate's own draw. Each later effector follows it: `_choose_effector` (its gate, its draw, its joints read and drawn),
`_act_effectors` (its actor's trace, its cost, its striatal event) and `_gate_lesson(i)`, the voice's lesson on its own gate. The
defect fixes 4 (gate_own_draw), 5 (actor_trace_tick) and 8 (elig_from) are switches, off by their absence (physiology.py SWITCHES).
Since step R9 the frame the effectors' gate inputs and costs read is the world's, the one `_sense` took (`_tick_frame`). STEP R6 (the
motor timing part, body/core/timing.py): a later effector's proposal is act_pred's, corrected by its forward half's error; under
chunk_gate its acts run in chunks with learned stops (act_pred's best guess the rest, its gate's own draw a no, its reflex, its declared
end, chunk_max); a reflex's act goes to the world with no gate draw, no efference copy, no actor credit and no gate eligibility."""
import collections
import math

import torch
import torch.nn.functional as F


# A104: kappa's established scale (Landis and Koch 1977, Biometrics 33:159): below 0.2 "slight", 0.2-0.4 "fair", 0.4-0.6 "moderate",
# 0.6-0.8 "substantial", above 0.8 "almost perfect". A later effector's decisiveness is earned from "fair" agreement and complete at
# "almost perfect" (_motor_sharp)
KAPPA_FAIR = 0.2
KAPPA_ALMOST_PERFECT = 0.8

class MouthMixin:
    def _imagine_value(self, first, h):
        """IMAGINATION FOR CHOICE: say `first`, then h-1 more symbols as the cortex would (greedy), on a copy of the window; the
        striatal critic's value of the imagined line (with the working-memory slot as it stands). Nothing in the body changes."""
        m = self.m
        with torch.no_grad():
            win = list(self.win); line = m.stri_line.clone(); sym = int(first); zero_read = torch.zeros(m.d, device=self.dev)
            held = {c_.field: c_.observe(self, self.sil, 1, still=True) for c_ in self.anatomy.channels}   # an imagined position: the world quiet, the face held (step R4)
            for e_ in self.anatomy.motors:
                held[e_.field] = int(e_.rest_id)                  # the later effectors at rest while it imagines speaking (step R5)
            said = []
            for step in range(int(h)):
                said.append(sym)
                win.append({**held, "xo": sym, "bundle": self.bands, "read": zero_read, "r": 0.0})
                if len(win) > m.window:
                    win = win[-m.window:]
                if step == int(h) - 1:
                    break
                obs, whos, bundles, reads = self._window_tensors(win)
                C = m.stream(m.inputs(self.anatomy, obs, whos, bundles))[-1]
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
            m.striatum_acts(z)                                    # the later effectors' lines as they stand (step R5; none for the diary)
            m.striatum_events(z)                                  # and the event lines' (step R7a; none for the diary)
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
        utterance's ends, and with the ends switched off nothing would ever let the held ear go. STEP R7c, THE PACE ON THE PARTNER CHANNEL
        (SIM_DESIGN.md 8's R7 row; the first design's 8.1 seam D: "turn-taking only on the voice against the declared partner
        channel"): 0 too for a body whose anatomy declares no partner (`Anatomy.partner`), since turn-taking is the voice's against the
        partner's pauses and returns; the partner, when declared, is channel 0, the words (the anatomy's check), so the symbols the pace
        hears are the partner's (THE BUILDER'S READING, for the lead)"""
        if self.anatomy.partner is None:
            return 0
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

    def _voice_ear(self, u):
        """THE VOICE'S OWN EAR (step R5: what VoiceEffector.gate_inputs reads, the gate's inputs beyond the stream and the feelings; the
        code as it stood in `_choose`): under gate_ear, [the ear's world input, its own act last tick] (the ear under the sensed pace, or
        the world's symbol this tick with its trace and gain), or None without the ear"""
        live_ = self._pace_mode() >= 2
        if int(self.cfg.get("gate_ear", 0)) and live_:
            # THE EAR UNDER THE SENSED PACE (M3, M5): held at 1 from the world's symbol until an end lets it go (M2 at once, M4 when the
            # reply is ready), then the wait; no trace, no gain
            ear_w = self._pace_ear(u); self._ear_now = ear_w
            return torch.tensor([ear_w, 1.0 if getattr(self, "_acted_last", False) else 0.0], device=self.dev)
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
            return torch.tensor([ear_w, 1.0 if getattr(self, "_acted_last", False) else 0.0], device=self.dev)
        return None

    def _choose(self, C1, pred1, u, level, stri):
        """the tick's choice for every effector, in the anatomy's order: the voice (effector 0) first, whether to speak (the gate), then
        what (the readout at the mood's sharpness; the actor's chunk, plan or vote), its draws on self.gen the tick's first, exactly as
        before step R5; then each later effector (`_choose_effector`: its gate, then its act, joint by joint), its draws after the voice's.
        Returns the voice's choice and the gate's own draw (`drew`: the gate's yes or no before the readout's choice could make a yes a
        rest; a chunk's letter run without the gate, a yes at p 1), which the lesson takes as the act under gate_own_draw (defect 4)"""
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
            # THE VOICE'S OWN EAR (step R5): the gate's inputs beyond the stream and the feelings are the effector's own, declared by it
            # (VoiceEffector.gate_inputs: `_voice_ear` below, on this tick's frame); appended after the adapted input, as always
            frame = self._tick_frame(u)                                  # this tick of the world, the one _sense took (step R9; no draw)
            ear_ = self.anatomy.voice.gate_inputs(frame, self)
            if ear_ is not None:
                feat = torch.cat([feat, ear_])
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
            drew = acted                                                               # the gate's own draw (defect 4: gate_own_draw)
            # DECISIVENESS BY CERTAINTY (sharp_conf; 2026-09-15, the live ruler): each word's first symbol is sampled from this readout, and
            # at a fixed sharpness a three-word answer needed three lucky starts where the forecast's margin was thin (the live mouth
            # answered 3 of 30 questions the greedy readout answered 20 of). A selection's noise falls as its evidence rises (the
            # basal ganglia's threshold; a decision's variance at the bound); the forecast's norm is its certainty, the same the
            # gate's salience reads. The sharpness here is the readout's times (1 + sharp_conf x that norm): a sure forecast is read
            # decisively, an unsure one as before. The readouts of the probes, the gauge and the dreams are untouched.
            sc_ = float(self.cfg.get("sharp_conf", 0.0)); self._sharp_eff = float(m.read_sharp) * (1.0 + sc_ * float(pred1.norm()))
            logits = m.readout(pred1).clone() * (1.0 + sc_ * float(pred1.norm()))
            if getattr(self.anatomy, "grounding", None) is not None:
                gs_ = self._ground_say(self.world.now)                  # A202: the word whose look fills the fovea, a prior on its logit
                self._ground_say_now = None if gs_ is None else [int(gs_[0]), float(gs_[1])]   # (body/core/grounding.py: GROUND_SAY x the margin)
                if gs_ is not None:
                    from body.core.grounding import GROUND_SAY
                    logits[int(gs_[0])] = logits[int(gs_[0])] + GROUND_SAY * float(gs_[1])
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
                    acted = True; p_act = 1.0; self._chunk_cont = True; drew = True     # the program runs with no gate decision: a yes at p 1
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
            top_ = int(torch.argmax(logits))
            self._last_choice = {"p_act": float(p_act), "acted": bool(acted), "nxt": int(nxt), "p_choice": float(p_choice), "norm": float(pred1.norm()),
                                 "top": top_, "p_top": float(probs[top_]), "sharp": float(getattr(self, "_sharp_eff", m.read_sharp))}   # the tick's choice, for the
                                                                                                                                       # instruments and the inner word (A137)
        if len(self.anatomy.effectors) > 1 and self._recall_on():
            self._frame_recall(C1)                                         # step R7f: the recall into action for this tick's proposals (frames.py)
        for i_ in range(1, len(self.anatomy.motors) + 1):                  # THE LATER EFFECTORS (step R5), after the voice, in their order
            self._choose_effector(i_, frame, C1, level, stri)              # (step R6h: the motor effectors, wherever the voice stands)
        return acted, nxt, p_act, p_choice, probs, feat, ent, act_on, drew

    @staticmethod
    def _motor_state_new(e):
        """a later effector's working state (step R5; one per effector after the voice, life.motor; e its declaration): its gate's
        buffer (the voice's gate_buf's twin), its lesson's baseline and report, its act last tick, its actor's eligibility trace, and
        the tick's choice. Step R6, its timing part's: the chunk's length (0: none under way), its body sense last tick, the sense its
        forward half foresaw for this tick and the error now; act_inv's reliability (its running confusion per joint, None for an
        effector with no inverse model, each joint's kappa and their mean clipped, the reliability; kept through the night and saved
        with the body), its lessons and the last; the chunks begun and their stops. Step R6h's: its performance error's running means
        (when it declares the term), its own fatigue, the movement unit's held settings, act_inv's pairs gathered for the next batch, the
        cord's patterns' counts and the born cry's breath clock; C54's, where its pattern generator stands in its rhythm (the cycle under
        way and the next: a function of the tick and the organs' spg_seed and spg_phase, so it is found again from birth after a load and
        never saved; body/core/cord.py `_spg_where`). Since A70 (2026-09-25) a save keeps all of it but that cache: life["motor"] its
        reliability, performance means, fatigue and act_inv's pending pairs, the body's day the rest (body/core/persistence.py)"""
        return {"buf": collections.deque(maxlen=96), "g_base": None, "last": None, "acted_last": False, "e_actor": None, "now": None,
                "a_tag": None, "a_upd": [0.0, 0.0],   # A142: the actor's synaptic tag, and the day's summed update norms (fast, slow: a ruler)
                "chunk": 0, "sense": None, "fwd": None, "err": None,
                "inv_conf": ([[[0.0] * int(K) for _ in range(int(K))] for K in e.factors] if e.inverse else None),
                "inv_kappa": [0.0] * len(e.factors), "inv_gain": 0.0, "inv_n": 0, "inv_last": None,
                "chunks": 0, "stops": {"rest": 0, "gate": 0, "reflex": 0, "end": 0, "max": 0},
                # step R6h: its performance error's running means (A41), when it declares one; its own fatigue (own_fatigue); the movement
                # unit's held settings (unit_margin); act_inv's pairs gathered for the next batch (act_inv_every); the cord's patterns'
                # ticks (logged as reflex) and the born cry's breath clock; C54: its pattern generator's place in its rhythm (none yet)
                "perf": ([[0.0] * int(K) for K in e.factors] if getattr(e, "intrinsic", False) else None),
                "fatigue": 0.0, "unit": None, "inv_batch": [], "cord_n": {}, "cry_t": 0, "breath_t": 0, "spg_cyc": None}   # (breath_t: A173)

    def _choose_effector(self, i, frame, C1, level, stri):
        """A LATER EFFECTOR'S CHOICE (step R5; effector i > 0, after the voice's, its draws on self.gen after the voice's): whether (its
        own gate, born as the voice's, on the stream, the feelings, its proposal's salience, the level and its own inputs, flattened by
        stress, over the spontaneous floor; its draw), then what (the per-joint readout of its proposal at the mood's sharpness, the
        striatal actor's bias added joint by joint under the actor, each joint drawn in turn; the act its joints' settings, a draw of its
        rest no act). The voice's reflexes (the ear, the pace, the listening, the babble drive, the chunk) are the voice's alone. Kept in
        its state for the act, the lesson and the instruments.
        STEP R6: first its timing part's senses (`_timing_sense`: act_inv's lesson on the tick before, the forward half's error now); its
        proposal is act_pred's, corrected by that error (`propose`). ITS SPINAL REFLEX, when it fires (`reflex`), takes the tick: its act
        goes to the world, the effector's own act is its rest, its gate does not draw. THE LEARNED STOPS (chunk_gate): while a chunk is
        under way (it acted last tick and its last act was not its declared end) and has run fewer than chunk_max acts, its gate's own
        draw decides whether it goes on, and the act is act_pred's best guess (no draw); the chunk ends where that guess is its rest or
        the gate says no. Otherwise (a chunk at its ceiling, its end, no chunk under way, chunk_gate 0) the choice is a fresh decision,
        as before R6, and under chunk_gate an act of it begins a chunk.
        STEP R6h: `i` is its place among the motor effectors plus one (the voice may stand anywhere); its gate reads its own fatigue
        (own_fatigue); the born orienting bias joins its logits where it declares one (body/core/cord.py); under unit_margin a unit
        under way holds its act joint by joint (`_unit_hold`) in place of act_pred's best guess; after the choice the cord's patterns
        for this tick (the spinal pattern generator, the born cry) are kept for the world (st["now"]["cord"])."""
        m = self.m; e = self.anatomy.motors[i - 1]; st = self.motor[i - 1]; tab = m.get_submodule(e.organ)
        self._timing_sense(i)                                             # act_inv's lesson on the last tick, the forward error now (step R6)
        prev = st["now"]
        with torch.no_grad():
            pred = e.propose(self, C1)                                    # act_pred's, corrected by the forward half's error (step R6)
            sal = float(self.cfg["gate_salience"]) * (float(pred.norm()) if pred is not None else 0.0)
            own = torch.as_tensor(e.gate_inputs(frame, self, st), dtype=torch.float32, device=self.dev).reshape(-1)
            if own.numel() != int(e.n_in):
                raise ValueError(f"the effector {e.name!r} declares {e.n_in} gate inputs and gave {own.numel()}")
            fat_ = float(st["fatigue"]) if int(self._motor_const("own_fatigue")) else self.fatigue   # step R6h: its own fatigue (own_fatigue)
            feat = torch.cat([C1.detach() / math.sqrt(float(m.d)),
                              torch.tensor([fat_ / 10.0, self.mood / 6.0, self.stress / 10.0, sal, level], device=self.dev), own])
            g_mod = m.get_submodule(e.gate)
            z_raw = g_mod(feat.unsqueeze(0))[0, 0]
            if int(self._motor_const("gate_scaling")):
                # A183 (2026-10-03): THE GATE'S SYNAPTIC SCALING. The dawn-75 copy's read (p1/a182_measure.py): every limb's gate and the
                # gaze's stood at a logit of +10 to +15 (acting on 97 to 100% of ticks; the voice's at -2.9): past the sigmoid's range its
                # slope is 1e-5, the three-factor rule's (act - p) is 0 on every sample (no tick of rest to compare), so the gate could
                # not learn when NOT to act, and the amygdala's freeze before a forecast pain (A71, A138: a logit or less) moved nothing:
                # 75 days of the wrists' pain (12 ticks a thousand) with no fall. A147's law for the actors, for the gates: a neuron
                # whose drive runs past its set point scales all its synapses down together (Turrigiano 2008; Turrigiano and Nelson
                # 2004). The set point is the gate's own floor read the other way: its certainty to act no greater than the floor's
                # certainty not to stay silent, |z| <= logit(1 - gate_floor) (2.94 at the floor 0.05; no new constant); past it the
                # gate's weights are multiplied by (z_max / |z|) ** (1 / tag_reach) each tick it is read (the log of the drive decays
                # at the tag's reach, 64 ticks: +13 is back at 3 in some 300 ticks), down only: a gate within its range is left alone
                fl0 = float(self.cfg["gate_floor"])
                if 0.0 < fl0 < 0.5:
                    zmax = math.log((1.0 - fl0) / fl0); az = abs(float(z_raw))
                    if az > zmax:
                        from .physiology import FRAMES as _FR
                        H_ = float(self.cfg.get("tag_reach", _FR["tag_reach"]))
                        c_ = float((zmax / az) ** (1.0 / H_))
                        for p_ in g_mod.parameters():
                            p_.mul_(c_)
                        st["gate_scaled"] = int(st.get("gate_scaled", 0)) + 1
            z = z_raw / (1.0 + self.stress / 10.0)
            if self._amyg_on() and int(self._amyg_const("amyg_pav")) and getattr(self, "_amyg_now", None) is not None:
                # APPROACH AND AVOID (step R7e, amyg_pav; SIM_DESIGN.md 7.4 item 4): the gate's logit gains beta x clip(N, -c, c), a Go bias
                # toward good and a freeze toward bad (Guitart-Masip et al. 2012); built and off at birth (physiology.py AMYG)
                c_ = float(self._amyg_const("amyg_pav_clip"))
                if str(self._amyg_const("amyg_pav_form")) == "earned":
                    # A71 (the lead's decision; physiology.py AMYG amyg_pav_form): born on, its weight the aversive heads' largest earned
                    # reliability (0 at birth, grown by use)
                    z = z + float(self._amyg_const("amyg_pav_beta")) * float(self._amyg_now.get("rho_bad", 0.0)) * max(-c_, min(c_, float(self._amyg_now["N"])))
                else:
                    z = z + float(self._amyg_const("amyg_pav_beta")) * max(-c_, min(c_, float(self._amyg_now["N"])))
            if int(self.cfg.get("imagine_pav", 0)) and getattr(self, "_imag_N", None) is not None and self._amyg_on() \
                    and getattr(self, "_amyg_now", None) is not None:
                # A138 (imagine_pav): the amygdala's forecast on the future the body has just imagined (sleep.py _imagine, fading) joins
                # the gate's logit as the felt one does under amyg_pav's earned form: beta x the aversive heads' earned reliability x
                # clip(the imagined net valence): a go toward an imagined good, a freeze before an imagined bad
                c_ = float(self._amyg_const("amyg_pav_clip"))
                z = z + float(self._amyg_const("amyg_pav_beta")) * float(self._amyg_now.get("rho_bad", 0.0)) * max(-c_, min(c_, float(self._imag_N)))
            fl = float(self.cfg["gate_floor"])
            if int(self._motor_const("gate_ceiling")):
                # A183: A CEILING SYMMETRIC TO THE FLOOR. Spontaneous activity never stops (the floor); neither does spontaneous rest: the
                # gate acts with probability floor + (1 - 2 floor) sigmoid(z), at most 1 - floor, so every effector rests on at least a
                # floor's share of its ticks and the three-factor rule has both choices to compare (the covariance of credit with
                # acting is undefined at p = 1; a policy that cannot sometimes rest cannot learn when to rest). No new constant
                p_act = fl + (1.0 - 2.0 * fl) * float(torch.sigmoid(z))
            else:
                p_act = fl + (1.0 - fl) * float(torch.sigmoid(z))
            rfx = e.reflex(frame, self, st)                               # its spinal reflex this tick (step R6), or None
            rest = int(e.rest_id)
            going = bool(int(self.cfg.get("chunk_gate", 0))) and bool(st["acted_last"]) and int(st["chunk"]) > 0   # a chunk under way
            ended = e.end_id is not None and prev is not None and int(prev["act"]) == int(e.end_id)            # closed by its declared end
            cont = going and not ended and rfx is None and int(st["chunk"]) < int(self.cfg.get("chunk_max", 12))
            stop = None
            drew = bool(torch.rand(1, generator=self.gen).item() < p_act) if rfx is None else False   # no draw on a reflex's tick
            sharp_e = self._motor_sharp(e, st)                            # A97: its decisiveness earned by its inverse model
            logits = tab.logits(pred, sharp_e, earned=self._motor_earned(e, st))   # C148: and its proposal's certainty with it
            act_on = bool(int(self.cfg.get("actor", 0)) and stri and getattr(self, "_z_now", None) is not None)
            if e.orient and int(self._reflex_const("orient")):            # STEP R6h: the born orienting bias (body/core/cord.py; 3.7, A43)
                ob_ = self._orient_bias(e, frame, tab)
                if ob_ is not None:
                    logits = [lg + b_ for lg, b_ in zip(logits, ob_)]
            proposal = logits                                             # A176: the cortex's proposal with the born orienting bias, the
            if act_on:                                                    # striatum's bias not in it: what a unit under way is held against
                # the striatum disposes: its bias on each joint's proposal
                a_bias = float(self.cfg.get("actor_beta", 1.0)) * torch.tanh(m.get_submodule(e.actor)(self._z_now))
                logits = [lg + b_ for lg, b_ in zip(logits, tab.split(a_bias))]
            if e.reserved:                                                # a one-joint alphabet's reserved acts are never drawn
                logits[0] = logits[0].clone(); logits[0][list(e.reserved)] = float("-inf")
                if proposal is not logits:
                    proposal[0] = proposal[0].clone(); proposal[0][list(e.reserved)] = float("-inf")
            probs = [torch.softmax(lg, -1) for lg in logits]
            if rfx is not None:                                           # THE REFLEX: its act to the world; the effector's own, its rest
                act = rest; acted = False; p_choice = 0.0
                dig = [int(x_) for x_ in tab.digits(torch.tensor(act)).tolist()]
                if going:
                    stop = "reflex"
            elif cont:                                                    # THE CHUNK GOES ON while its gate's own draw says so
                if drew:
                    um_ = self._motor_const("unit_margin")
                    if um_ is None:
                        act = self._best_guess(e, pred)                   # act_pred's best guess, no draw (R6)
                    else:
                        act = self._unit_hold(e, st, proposal, float(um_))   # STEP R6h: the movement unit holds its act (the persistence margin);
                                                                            # A176 (2026-10-02): against the cortex's proposal, not the striatum's
                                                                            # standing bias (life day 65: the left arm's saturated actor, +-1 on
                                                                            # every joint, led the held setting by more than log 4 and took every
                                                                            # unit back to one act within a tick: 75% of the day one posture, the
                                                                            # arm pinned at its range, no flexion for the pull-to-sit). The striatum
                                                                            # chose at the unit's start (the draw); mid-unit the program runs unless
                                                                            # the cortex changes its mind (the basal ganglia select, the brainstem
                                                                            # and cord carry the unit: Grillner 2006; von Hofsten's units)
                    dig = [int(x_) for x_ in tab.digits(torch.tensor(act)).tolist()]
                    acted = act != rest
                    p_choice = 1.0
                    for p_, a_ in zip(probs, dig):
                        p_choice *= float(p_[a_])
                    if not acted:
                        p_choice = 0.0; stop = "rest"                     # the learned end: its best guess is its rest
                else:
                    act = rest; acted = False; p_choice = 0.0; stop = "gate"   # its gate's own draw closes it
                    dig = [int(x_) for x_ in tab.digits(torch.tensor(act)).tolist()]
            else:
                if going:
                    stop = "end" if ended else "max"                      # closed by its end, or at its ceiling: a fresh decision
                if drew:
                    dig = [int(torch.multinomial(p_.cpu(), 1, generator=self.gen)) for p_ in probs]
                    act = tab.flat(dig); p_choice = 1.0
                    for p_, a_ in zip(probs, dig):
                        p_choice *= float(p_[a_])
                    acted = act != rest                                   # a draw of its rest: no act
                else:
                    act = rest; acted = False; p_choice = 0.0
                    dig = [int(x_) for x_ in tab.digits(torch.tensor(act)).tolist()]
            st["unit"] = list(dig) if acted else None                    # step R6h: the settings a unit under way holds
            if int(self.cfg.get("chunk_gate", 0)):
                if acted:
                    st["chunk"] = int(st["chunk"]) + 1 if cont else 1
                    if not cont:
                        st["chunks"] = int(st["chunks"]) + 1              # a chunk begins
                else:
                    st["chunk"] = 0
            if stop is not None:
                st["stops"][stop] = int(st["stops"].get(stop, 0)) + 1
        st["now"] = {"act": int(act), "acted": bool(acted), "drew": bool(drew), "p_act": float(p_act), "p_choice": float(p_choice),
                     "digits": dig, "probs": probs, "feat": feat.cpu(), "act_on": act_on, "cost": 0.0, "sharp": float(sum(sharp_e) / len(sharp_e)) if isinstance(sharp_e, list) else float(sharp_e),
                     "cont": bool(cont), "stop": stop, "reflex": rfx is not None, "world": int(rfx) if rfx is not None else int(act), "int": 0.0}
        # STEP R6h: THE CORD'S PATTERNS (body/core/cord.py): its spinal pattern generator and its born cry, added below the gate to the act
        # this tick (the world adds them to the targets it re-anchors: acts.cord); the gate's draw and the act's eligibility are the gate's
        st["now"]["cord"] = self._cord(i, frame, p_act, dig, rfx is not None) if (e.spg or e.cry or (e.orient and e.vor)) else None

    def _act(self, u, felt, stri, gam, delta, acted, nxt, p_act, p_choice, probs, feat, act_on, drew=None):
        """the act: the actor's credit, the intrinsic credit, its own symbol (or its rest) enters the stream, the gate's tag; each later
        effector's credit, cost and striatal event after the voice's (step R5: their acts enter the stream in the voice's own step)"""
        m = self.m
        int_t = 0.0
        tick_tr = int(self.cfg.get("actor_trace_tick", 0))
        if tick_tr:
            # DEFECT 5 FIXED (actor_trace_tick): the actor's eligibility decays by dopamine's discount every tick (the lesson multiplies
            # it by the dopamine of every tick), so an act's credit fades with time, not with the acts that follow it
            g_tr = float(gam[int(self.cfg["dopamine_band"])])
            if getattr(self, "_e_actor", None) is not None:
                self._e_actor = g_tr * self._e_actor
            if getattr(self, "_e_chooser", None) is not None:
                self._e_chooser = g_tr * self._e_chooser
        if acted and act_on and not self._chunk_cont:            # the actor's act and credit: once per word under the chunk form
            self._ring_torn.append(1.0 if self._torn_now else 0.0); self._ring_ent.append(float(self._ent_now)); self._torn_now = False
            if getattr(self, "_a_bias_now", None) is not None:
                ab_ = self._a_bias_now
                self._act_pending.append([self.ticks, float(ab_[nxt]), 0.0])   # its vote for the act taken; the reward that follows is gathered
                spk_ = ab_.clone(); spk_[self.sil] = float("-inf"); spk_[self.bans] = float("-inf")
                self._act_agree.append(1.0 if int(spk_.argmax()) == nxt else 0.0)
            with torch.no_grad():                                      # the actor's eligibility: what it said against what it expected, on this input
                g_act = 1.0 if tick_tr else float(gam[int(self.cfg["dopamine_band"])])   # decayed per tick above under actor_trace_tick
                if str(self.cfg.get("actor_form", "add")) == "softmax":
                    self._chooser_credit(nxt, g_act)
                else:
                    oh = torch.zeros_like(probs); oh[nxt] = 1.0
                    e_new = torch.outer((oh - probs.detach()) * self._actor_squash_grad(m.actor), self._actor_input())   # A147: through the tanh; A148: at unit power
                    ea = getattr(self, "_e_actor", None)
                    self._e_actor = (g_act * ea if ea is not None else torch.zeros_like(e_new)) + e_new
        if acted:
            hab = float(self.cfg["gate_habit"])
            if not self.anatomy.voice.intrinsic:
                pass                                              # STEP R6h (C61): a voice that declares no intrinsic term carries none (the G1's words)
            elif str(self.cfg.get("gate_int_form", "value")) == "error":
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
        if len(self.anatomy.effectors) > 1:
            self._act_effectors(u, stri, gam, tick_tr)            # the later effectors (step R5), after the voice
        self._acted_last = bool(acted)
        if float(self.cfg.get("gate_slow_lr", 0.0)) > 0.0:
            with torch.no_grad():
                g_ = float(self.cfg.get("vcrit_gamma", 1.0 - 1.0 / 1024))
                tag_in = torch.cat([feat.detach(), torch.ones(1, device=self.dev)])
                prev = getattr(self, "_gate_tag", None)
                a_tag = float(drew) if (drew is not None and int(self.cfg.get("gate_own_draw", 0))) else float(acted)   # the gate's own draw (defect 4)
                self._gate_tag = ((g_ * prev) if prev is not None else torch.zeros_like(tag_in)) + (a_tag - p_act) * tag_in
        return int_t

    def _actor_tag_step(self, st, e_new):
        """A142 (the lead, 2026-09-30; C153 the same evening): THE ACTOR'S SYNAPTIC TAG. Each later effector's eligibility (the one-hot of its
        settings less their probabilities, on the striatal input: what the fast lesson credits over dopamine's 16 ticks) is also summed
        into a tag that decays at the tag's reach (1 - 1/tag_reach, 64 ticks) and is captured each tick by phasic dopamine (critics:
        weight += actor_slow_lr x delta x tag), as a tag set by activity is captured by later dopamine (Frey and Morris 1997). Why: a reach and a grasp take
        seconds of coordinated steps and her smile comes after; the fast lesson's credit had faded (0.9375 a tick) before it arrived, so
        the actor learned from pain (dense, a tick after the act) and hardly from her face (life day 40: 533 word smiles, 21 acts on things
        smiled at, the arms' choices unmoved). e_new None: the decay alone. Off (actor_slow_lr 0): nothing kept"""
        if float(self.cfg.get("actor_slow_lr", 0.0)) <= 0.0:
            return
        # C153 (2026-09-30, day 42 read at noon): the tag decays at THE TAG'S REACH (1 - 1/tag_reach, 64 ticks: the horizon over which R7d
        # credits an act with the dopamine that follows) and is captured by PHASIC dopamine (critics: delta), not by the long critic's
        # error over its 1,024-tick horizon. As first built (A142, dawn 42) the capture was delta_long, a dense drifting signal: whenever
        # the long outlook dipped it punished whatever the body had done in the last minute, mostly holding its wrists, so the shoulders'
        # big swings rose from 8% to 52% of ticks by noon, the wrists' pain doubled (12 to 24 a thousand ticks), acts smiled at fell by
        # two thirds and the reward ran -206 in half a day. This form is the eligibility trace of TD(lambda) at a 64-tick horizon
        g_l = 1.0 - 1.0 / float(self.cfg.get("tag_reach", 64))
        tag = st.get("a_tag")
        if e_new is None:
            if tag is not None:
                tag.mul_(g_l)
            return
        if tag is None:
            st["a_tag"] = e_new.clone()                                     # the first act's eligibility begins it
        else:
            tag.add_(e_new)                                                 # (the tick's decay came first, above)

    def _motor_sharp(self, e, st):
        return self._motor_read(e, st)[0]

    def _motor_earned(self, e, st):
        """C148: the per-joint exponents that earned the sharpness (rel x variety; rel alone without sharp_per_joint; 1 for an effector
        without an inverse model), for ActTable.logits' `earned` when sharp_earned_norm is on; None when it is off"""
        return self._motor_read(e, st)[1] if int(self._motor_const("sharp_earned_norm")) else None

    def _motor_read(self, e, st):
        """-> (sharpness, earned exponents), each one for every joint or one per joint. A97 (the lead, 2026-09-26): A LATER EFFECTOR'S DECISIVENESS IS EARNED. Its proposal (act_pred's forecast of its own next act)
        is read at 1 + (the mouth's sharpness - 1) x its inverse model's reliability (st["inv_gain"]: the joints' mean kappa, 0 until 64
        acts and 0 whenever its acts have stopped varying, body/core/timing.py), so a limb whose cortex has not yet shown that it knows
        what its acts do proposes softly (sharpness 1: the forecast's own spread, the cord's patterns and the born biases weigh against
        it) and one that has proposes as the mouth does. Why: on life 1's second day every limb had fallen into a fixed point, the forecast
        of its own next act read at the mouth's 25 (the act it took, predicted, taken again: the same seven settings on 89% of its acts,
        every joint at its range limit, the cord's kick cancelled), because forecasting one's own act is trivially certain and says
        nothing of the world. The corticospinal system's say over movement grows with development and with use (Martin 2005; Eyre 2007),
        and the design's own law for a later readout is to be born small and weighted by earned reliability (docs/audit/pfc_maturation.md;
        act_pred's lesson weighs a label by act_inv's kappa). Kappa needs variety (a joint whose acts never vary has shown nothing), so
        a fixed point cannot hold: it softens the readout that made it. An effector without an inverse model (the gaze) reads as before
        (C82). No new constant: 1 is the readout's own scale."""
        s = float(self.m.read_sharp)
        if not getattr(e, "inverse", False):
            return s, 1.0
        if int(self.cfg.get("sharp_per_joint", 0)):
            # A140 (C117, 2026-09-29): EACH JOINT'S DECISIVENESS IS ITS OWN, AND EARNED BY VARIETY. On life day 27 every arm joint sat at a
            # stop (the left elbow 581 ticks of 600) while the actor sent big steps into the stop: the effector's mean kappa stayed "fair"
            # on the other joints and on a near-constant act (a 1% sampling variety keeps the chance-corrected kappa high), so the
            # readout stayed sharp and the fixed point A97 meant to soften held. Here joint j reads at s ** (rel_j x variety_j): rel_j
            # from its own kappa (inv_kappa[j]) and variety_j the normalized spread of its true settings over the inverse model's
            # horizon (its confusion's column marginals: 0 for one setting always, 1 for all alike; no new constant: the docstring's own
            # "a joint whose acts never vary has shown nothing" made literal)
            kap = st.get("inv_kappa") or []
            conf = st.get("inv_conf") or []
            out = []
            for j in range(len(e.factors)):
                k_j = float(kap[j]) if j < len(kap) else 0.0
                rel = max(0.0, min(1.0, (k_j - KAPPA_FAIR) / (KAPPA_ALMOST_PERFECT - KAPPA_FAIR)))
                var = 0.0
                if j < len(conf) and conf[j]:
                    Cj = conf[j]; K = len(Cj)
                    cols = [sum(Cj[r][c] for r in range(K)) for c in range(K)]; n = sum(cols)
                    if n > 0 and K > 1:
                        var = (1.0 - max(cols) / n) / (1.0 - 1.0 / K)
                out.append(s ** (rel * max(0.0, min(1.0, var))))
            g_ = [math.log(o_) / math.log(s) if s > 1.0 else 1.0 for o_ in out]          # C148: the exponents that earned each sharpness
            return out, g_
        kappa = float(st.get("inv_gain", 0.0) or 0.0)
        rel = max(0.0, min(1.0, (kappa - KAPPA_FAIR) / (KAPPA_ALMOST_PERFECT - KAPPA_FAIR)))
        return (s ** rel, rel) if True else None                                                      # A104: the temperature's own (geometric) scale, on kappa's
                                                                            # established one: nothing below "fair" agreement, the
                                                                            # mouth's at "almost perfect" (Landis and Koch 1977). A97's
                                                                            # 1 + (s - 1) kappa gave sharpness 3-5 at kappa 0.1 ("slight"),
                                                                            # and every limb's top probability was back above 0.9 within
                                                                            # half of life day 3, the day's earning gone with the variety

    def _actor_input(self):
        """A148 (2026-10-01): THE ACTOR'S LESSON AT THE INPUT'S UNIT POWER. The actor's weights step by lr x dopamine x (one-hot - p) x
        beta(1 - tanh^2) outer z (A147), so one step moves the bias's pre-activation by lr x dopamine x |z|^2: on the G1's striatal
        expansion (4,096 units, 1,700 of them active near 3, |z|^2 some 22,000) a single tick of dopamine moved it by hundreds, the
        bias slammed to its rail at every reward or pain (life day 49's first thousand ticks under A147: the actors' drive, scaled to
        1 within a minute, read 20 to 50 on the running mean as each lesson threw it back up). The eligibility is taken on z / max(|z|^2,
        1): the normalized LMS step (Haykin; divisive normalization of the input's power, the gain the RLS critics already carry), so a
        unit of dopamine moves the pre-activation by at most lr x beta whatever the line's load; the readout w . z is unchanged. The
        floor 1 is the unit (an empty line, z = 0, learns nothing either way)"""
        z = self._z_now
        return z / max(float(torch.dot(z, z)), 1.0)

    def _actor_squash_grad(self, actor):
        """A147 (2026-10-01): THE ACTOR'S ELIGIBILITY GOES THROUGH ITS SQUASHING. The striatum's bias on a logit is beta x tanh(w . z)
        (`_choose`, `_act_effectors`' readout), so the gradient of the act's log-probability with respect to w is
        (one-hot - p) x beta x (1 - tanh^2(w . z)) x z; until now the eligibility was (one-hot - p) x z, the squashing's derivative and
        its gain left out, so a unit driven past its range went on growing: on life day 48 every effector's actor read pre-activations
        of 2,000 to 11,000 against a range of 1 (the voice's 550), every setting's bias at +-1, a bang-bang policy dopamine could no
        longer move (the left shoulder's pitch at +small on 85% of its draws, the positional pain 142 onsets a day for five days).
        The factor per setting, [the actor's rows]; 1 where no striatal input stands"""
        if getattr(self, "_z_now", None) is None:
            return 1.0
        with torch.no_grad():
            t_ = torch.tanh(actor(self._z_now))
            return float(self.cfg.get("actor_beta", 1.0)) * (1.0 - t_ * t_)

    def _act_effectors(self, u, stri, gam, tick_tr):
        """THE LATER EFFECTORS' ACTS (step R5), each after the voice's, in the anatomy's order: its actor's eligibility (per act, or
        decayed every tick under actor_trace_tick), the one-hot of each joint's setting against that joint's probabilities, on the
        striatal input, as the voice's actor's over which symbol; its cost (its declaration's) to the body's fatigue; its act an event of
        its own striatal line; its act last tick. Its act entered the stream in the tick's own step (`_step`, its window field).
        Step R6: a chunk's continuation is act_pred's program, not the actor's choice, so only a chunk's first act is the actor's to
        credit (as the voice's words); a reflex's act costs its effort and is no act of the gate's (no credit, no striatal event); then
        the forward half foresees the next tick's body sense from the stream after this tick's own step (`_timing_foresee`)."""
        m = self.m; g_ = float(gam[int(self.cfg["dopamine_band"])])
        frame = self._tick_frame(u)                                      # this tick of the world, the one _sense took (step R9)
        for i_, (e_, st_) in enumerate(zip(self.anatomy.motors, self.motor)):
            now = st_["now"]
            if tick_tr and st_["e_actor"] is not None:
                st_["e_actor"] = g_ * st_["e_actor"]
            self._actor_tag_step(st_, None)                          # A142: the tag decays at the long critic's horizon every tick
            if now["acted"] and now["act_on"] and not now["cont"]:
                with torch.no_grad():
                    oh = torch.cat([F.one_hot(torch.tensor(a_), int(k_)).to(p_.dtype).to(p_.device) - p_
                                    for a_, k_, p_ in zip(now["digits"], e_.factors, now["probs"])])
                    oh = oh * self._actor_squash_grad(m.get_submodule(e_.actor))   # A147: the gradient through the bias's tanh
                    e_new = torch.outer(oh, self._actor_input())                 # A148: on the striatal input at unit power
                    ea = st_["e_actor"]
                    st_["e_actor"] = ((1.0 if tick_tr else g_) * ea if ea is not None else torch.zeros_like(e_new)) + e_new
                    self._actor_tag_step(st_, e_new)                 # A142: and takes this act's eligibility
            if now["acted"] and e_.intrinsic:
                now["int"] = self._perf_error(e_, st_, now)       # step R6h: its performance error, when it declares one (A41, C61)
            if now["acted"] or now["reflex"]:
                now["cost"] = float(e_.cost(now["world"], frame, self))
                if int(self._motor_const("own_fatigue")):
                    st_["fatigue"] = float(st_["fatigue"]) + now["cost"]  # step R6h: its own fatigue (SIM_DESIGN.md 3.5)
                else:
                    self.fatigue += now["cost"]
                if now["acted"] and stri:
                    m.striatum_push_act(i_, now["act"])           # its act is an event of its own line
            st_["acted_last"] = bool(now["acted"])
            self._timing_foresee(i_ + 1)                          # the forward half: the next tick's body sense (step R6)

    def _perf_error(self, e, st, now):
        """THE PERFORMANCE ERROR OF A MOTOR EFFECTOR THAT DECLARES IT (step R6h; SIM_DESIGN.md 3.5, A41 and C61; Gadagkar et al. 2016: a
        singing bird's dopamine neurons encode its performance against its own expectation), on a tick it acted: per joint, the belief
        its choice gave the setting it made (that joint's probability of it) less that setting's usual belief (a running mean per joint
        and setting, moving (1 - gate_habit) of the way a tick it is made: the tract's 10 articulators x 5 settings, 50 means), averaged
        over its joints. The voice's own form, on its symbol, joint by joint; zero-mean once its expectations catch up, and it compares
        the effector with its own past, never with anyone. It enters its gate's credit at gate_int (the lesson's w_int x e_t) and nothing
        else: not the reward, the critics, dopamine or any other gate (under gate_int_form "error", the one form a motor effector carries:
        Life refuses another)"""
        hab = float(self.cfg["gate_habit"]); perf = st["perf"]; errs = []
        for j, (p_, a_) in enumerate(zip(now["probs"], now["digits"])):
            b_ = float(p_[int(a_)]); m_ = float(perf[j][int(a_)])
            errs.append(b_ - m_)
            perf[j][int(a_)] = m_ + (1.0 - hab) * (b_ - m_)
        return float(sum(errs) / len(errs))

    def _feel_and_learn(self, delta, delta_slow, delta_long, feat, acted, int_t, p_act, drew=None):
        """the feelings from dopamine; the gate's buffer and its lesson, for every effector (the voice's, then each later one's: the
        same credit, the effector's own act, draw and cost); the waking cortex lesson"""
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
        credit = delta + float(self.cfg["gate_slow_w"]) * delta_slow + vw * delta_long
        r_tr = float(getattr(self, "_r_pos_tr", 0.0))                  # A156: the appetitive rate (the felt reward's positive part traced at the drive's clock, critics.py)
        row = [feat.cpu(), acted, credit, int_t, self.fatigue, r_tr, p_act]   # the reward-rate trace at the tick, for the drive; the probability it acted with
        if drew is not None and int(self.cfg.get("gate_own_draw", 0)):
            row.append(bool(drew))                                     # the gate's own draw (defect 4)
        self.gate_buf.append(row)
        for st_ in getattr(self, "motor", ()):                         # the later effectors' rows (step R5): its draw and its act's cost always;
            n_ = st_["now"]                                             # step R6: whether a reflex took the tick (no eligibility)
            st_["buf"].append([n_["feat"], n_["acted"], credit, n_["int"], (float(st_["fatigue"]) if int(self._motor_const("own_fatigue")) else self.fatigue),
                               r_tr, n_["p_act"], n_["drew"], n_["cost"], n_["reflex"]])
        lesson_now = self.ticks > 0 and self.ticks % int(self.cfg["gate_every"]) == 0
        if lesson_now and len(self.gate_buf) >= 16 + int(self.cfg["elig_ticks"]):
            try:
                self._gate_lesson()
            except Exception as e:
                self._gate_last = {"error": str(e)[:120]}
        for i_, st_ in enumerate(getattr(self, "motor", ()), 1):
            if lesson_now and len(st_["buf"]) >= 16 + int(self.cfg["elig_ticks"]):
                try:
                    self._gate_lesson(i_)
                except Exception as e:
                    st_["last"] = {"error": str(e)[:120]}
        # --- the waking cortex ---
        if self.ticks > 0 and self.ticks % int(self.cfg["wake_every"]) == 0:
            try:
                self._wake_lesson()
            except Exception as e:
                self._wake_last = {"error": str(e)[:120]}

    # ---------------- the gate's lesson (the striatum's opponent rule) ----------------
    def _gate_lesson(self, i=0):
        """THE GATE'S LESSON for effector i (step R5: the lesson is the same for every effector, each on its own gate, buffer, baseline
        and report: the voice's (i = 0) m.mouth_gate with opt_gate, gate_buf, _g_base and _gate_last as always; a later effector's
        gates[name] with opt_motor and its state in life.motor, its effort its act's recorded cost where the voice's is symbol_cost).
        Under gate_own_draw (defect 4) the eligibility's act is the gate's own draw, recorded in the row; under elig_from 1 (defect 8)
        the credit sums the dopamine from the tick after the act on. Step R6: a later effector's tick its reflex took gives its gate
        no eligibility (SIM_DESIGN.md 5.5)."""
        if i == 0:
            buf_, gate_, opt_, st_ = self.gate_buf, self.m.mouth_gate, self.opt_gate, None
        else:
            buf_, gate_, opt_, st_ = self.motor[i - 1]["buf"], self.m.get_submodule(self.anatomy.motors[i - 1].gate), self.opt_motor, self.motor[i - 1]
        buf = list(buf_)
        K = int(self.cfg["elig_ticks"]); dec = float(self.cfg["elig_decay"])
        s_ = 1 if int(self.cfg.get("elig_from", 0)) else 0              # the credit from the act's own tick (0), or from the tick after it (1)
        n = len(buf) - K - s_
        if n < 4:
            return
        cost = float(self.cfg["symbol_cost"]); f0 = float(self.cfg["gate_fatigue"]); w_int = float(self.cfg["gate_int"])
        tonic = float(self.cfg["gate_tonic"]); vig = float(self.cfg["gate_vigor"])
        feats = torch.stack([b[0] for b in buf[:n]]).to(self.dev)
        own_ = int(self.cfg.get("gate_own_draw", 0))
        acts = torch.tensor([1.0 if (b[7] if (own_ and len(b) > 7) else b[1]) else 0.0 for b in buf[:n]])   # the gate's own draw (defect 4), or the act
        G = torch.zeros(n)
        for t in range(n):
            g = sum((dec ** k) * float(buf[t + s_ + k][2]) for k in range(K))     # the dopamine that followed
            if buf[t][1]:
                # acting pays a tonic drive (babble is its own reward, not contingent on confidence) plus
                # the belief it had in its choice (habituating), minus an effort cost convex in fatigue
                # (linear, 0.59 at fatigue's ceiling never beat a confident recitation's drive of 0.7:
                # run 19, gate 0.97 all day, fatigue pinned at 40; convex, the mouth speaks in bouts).
                # With the effort in the reward (cost_in_reward) the cost is the critics' to predict, not the act's (the reward feels the
                # voice's effort alone: a later effector's then reaches neither; EffortReward, body/core/anatomy.py)
                c_t = cost if i == 0 else float(buf[t][8])                       # the voice's symbol_cost; a later effector's act's own cost
                drive_t = tonic + (float(self.cfg.get("gate_tonic_rate", 0.0)) * float(buf[t][5]) if len(buf[t]) > 5 else 0.0)   # THE DRIVE FOLLOWS THE REWARD RATE
                g += drive_t + w_int * float(buf[t][3]) - (0.0 if self.cfg.get("cost_in_reward") else c_t * (1.0 + (float(buf[t][4]) / f0) ** 2))
            G[t] = g
        # the credit is taken against a running baseline (dopamine is an error, not a value)
        base = getattr(self, "_g_base", None) if i == 0 else st_["g_base"]
        if base is None:
            base = float(G.mean())
        A = G - base
        g_base_new = float(self.cfg["gate_baseline"]) * base + (1.0 - float(self.cfg["gate_baseline"])) * float(G.mean())
        if i == 0:
            self._g_base = g_base_new
        else:
            st_["g_base"] = g_base_new
        if float(A.abs().max()) < 1e-4:
            return
        gate_.train()
        z = gate_(feats).squeeze(-1)
        fl = float(self.cfg["gate_floor"])
        # the probability the gate actually acted with (stress divisor and all), carried in the buffer (review 2026-09-06: recomputed
        # here without the divisor, (act - p) was biased with stress); older samples without it fall back to the recomputation
        p = torch.tensor([float(b[6]) if len(b) > 6 else float("nan") for b in buf[:n]], device=self.dev)
        top_ = (1.0 - 2.0 * fl) if (i and int(self._motor_const("gate_ceiling"))) else (1.0 - fl)   # A183: a motor gate's ceiling
        p = torch.where(torch.isnan(p), (fl + top_ * torch.sigmoid(z)).detach(), p)
        # THE THREE-FACTOR RULE: credit x (action - p) has expectation cov(credit, acting), what a policy
        # must learn (Go for acts that paid, NoGo for acts that cost); plus vigor: the average credit
        # itself, tonic dopamine setting the rate of acting whatever it did
        elig = (acts.to(self.dev) - p) + vig
        if i:
            elig = elig * torch.tensor([0.0 if (len(b) > 9 and b[9]) else 1.0 for b in buf[:n]], device=self.dev)   # a reflex's tick: none (step R6)
        loss = -(A.to(self.dev) * elig * z).mean()
        opt_.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(gate_.parameters(), 1.0)
        opt_.step(); gate_.eval()
        last = {"n": n, "credit_mean": round(float(G.mean()), 4), "baseline": round(float(base), 4),
                "acted": round(float(sum(1 for b in buf[:n] if b[1]) / n), 3), "tick": self.ticks}
        if i == 0:
            self._gate_last = last
        else:
            st_["last"] = last
        for _ in range(min(len(buf_), n)):                         # the samples the lesson consumed (review: popping gate_every left a third to be learned twice)
            buf_.popleft()
