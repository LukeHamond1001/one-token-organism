"""the body in frames (a mixin of `Life`, body/life.py; the core refactor's step R7, docs/SIM_DESIGN.md 7.2, 7.4, 7.6, 8's R7 row, 10): a
body that lives in the world's frames (the sim) rather than on the page alone.

STEP R7a, THE EVENT LINES (7.2, 7.4's low road; A37, A43): the anatomy's born event lines (`Anatomy.events`, body/core/anatomy.py
`EventLine`), read once a tick from the frame the tick is lived on (`_event_lines`): a line fires (1) when any of its numbers is above 0,
on its side where it declares one (both sides where the direction has none), and not where a line further along its limb's chain fires
(isolation: the furthest group that feels a contact names it); else 0. One declaration, read by the striatal expansion (the critics:
`_events_push` puts the tick's fired lines into the striatum's event line, whose born rows body/model.py appends after the effectors'
blocks) and by the amygdala (R7d). The diary declares none, so nothing here runs for it.

STEP R7b, THE FRAMES (7.4's "the fast memory's writes", 9's store, 10's "event end"; physiology.py FRAMES, the switch `frames`, which the
diary's cfg does not hold), at each tick's end (`_frame_tick`, after the tick's lessons and before its bookkeeping):
- THE FRAME'S SURPRISE (`_frame_surprise`): each forecasting channel's error on the frame that came: the words' the tick's surprise (the
  forecast's cosine distance to the symbol heard, the rest included: the language body's own, `_surp_tick`), every later channel's its
  head's squared error to the code that came (0.5 |head(C_t-1) - code_t|^2: the waking lesson's own loss, on the forecast made at the
  last tick's end); each divided by its own running mean (1 / min(n, err_tau): the critics' statistics' horizon), and their mean: every
  channel weighs alike, and a frame of ordinary surprise reads 1 (THE BUILDER'S READING, for the lead: 7.4 and 10 say "the summed
  forecast error", the first design "all channels weighted equally"; the mean is that sum weighted equally, and it keeps a frame's
  write strength in the words' units, a typical error reading 1, so the frames and the words share the store's floors and its
  eviction on one scale).
- THE EVENT'S END FOR FRAMES (`_frame_settle`; 10's "event end"): today's settle law on it, a fast and a slow running average (offset_fast
  4, offset_slow 64 ticks), settled while the fast is at most offset_settle 0.5 of the slow; an event ends on the first settled tick
  after one that was not (its surprise has risen and fallen back). There the store's last frame written is marked as an event's end
  (its boundary, `Store.mark_boundary`), the next frame written as an event's start (`mark_start`), and the tick is kept in the day's
  list of ends (R8 cuts the day's tape into episodes there). The night ends every event: an event still open at nightfall ends there.
- THE WRITE (`_frame_gate`, `_frame_write`): a frame is written when its surprise x (1 + tag) passes its own running write_q (0.9)
  quantile (the tag the amygdala's, R7d; 0 until then), so about a tenth of the ticks are written whatever the scale; the key is the
  stream's state at the tick before (the context before the frame, pattern-separated as the store's cortex key is: `query_from`), the
  value the frame's codes (every channel's code, summed in the anatomy's order: the frame as the cortex's input receives it), the
  strength surprise x (1 + |dopamine|) x (1 + tag), the store's own law for a heard word with the tag; the slot's `who` is 2 (a frame:
  the words' dreams never start at one; the night over frames is R8's).
- THE TICK'S RECORD (`_record_tick`): (the frame's surprise, dopamine, the tag, the net reward received), float32, 16 B a tick, in the
  day's record (`_rec`, its first `_rec_n` rows), R8's tape beside the codes; the day's ends beside it (`_rec_ends`). The night lets
  both go (R8 cuts and keeps them first).
Every state here is a working attribute of the life, saved with a motor body's day (A70).

STEP R7f, RECALL INTO ACTION (7.6, A45; the switch `recall`, and `wm_frames`; physiology.py FRAMES):
- THE HEADING (`_heading_step`): the trunk's yaw integrated from the torso gyro since birth, a head-direction signal by path integration
  (McNaughton et al. 2006): each tick, from the anatomy's `heading` source in the frame (the G1's `imu_torso`, raw), the gyro's rate about
  the direction its accelerometer's specific force gives as up (which way is down standing for the orientation, A67), times the tick
  (THE BUILDER'S READING, for the lead: 7.6 says "the trunk's yaw" and names no axis; the vertical the body can sense is the specific
  force's, as the cerebellum's mossy input reads it). A real gyro drifts (Woodman 2007), so the heading drifts: disclosed and reported
  (C51), never corrected from world truth.
- THE FRAME'S KEY: the cortex's stream plus the heading: the stream's pattern-separated direction (the store's cortex key) and the
  heading's born code (cos and sin of the heading through two fixed unit rows from the body's seed, the organs' head_code) summed at equal
  weight, at the key's scale (`query_from`, body/core/memory.py; the words' cortex key too, so frames and words share one search). THE
  BUILDER'S READING, for the lead: 7.6 names the two parts and no weight; equal is the one that prefers neither. And the words' key
  carries the heading too (the builder's reading, for the lead): the store holds words and frames and searches them with one query, and
  a key of one form would meet a query of the other only in part.
- THE FRAME'S VALUE: its codes (R7b) plus the efference copy of what the body did (every effector's act of the tick, its acts' row: a
  motor effector's unit rows summed over its joints, the voice's lexicon row; a rest adds nothing): the value is what came next after the
  key's context.
- RECALL (`_frame_recall`, at the choice, before the motor effectors choose): the store's nearest keys give back their values for the
  stream the choice reads, through the store's own search: under key_form "cortex" the words' read of this tick itself (the same query),
  else a read of its own with that query (no tiring); kept for the tick (`_frec_now`) and in the tick's window position ("frec"). (THE
  BUILDER'S READING, for the lead: 7.6's "as it gives back words today" read as the store's own search; the words' waking read tires
  what it recalls, and a second read of the same store in the same tick does not tire it again.)
- THE MAPS (`_recall_term`; one per motor effector, `recall_spec`): each motor effector's proposal (act_pred's, a direction in the
  stream's d, read against its acts' rows R [its settings, d] for its logits) gains M((r R^T) R) R, for the recalled value r: the
  recalled value's part in its acts' rows (the recalled act's embedding, (r R^T) R), through its map M born at zero (the organs'
  recall[name], d -> a score per setting), read back through its rows into the proposal (as the proposal's own terms are). It
  learns through act_pred's own lesson (`_timing_loss`: the same targets, the same weights, act_pred's plain step, GatedDescent):
  recall moves an act only as far as recalled acts have predicted the acts made (Lengyel and Dayan 2007: episodic control, the
  hippocampus's "third way" into action). The step's constants are act_pred's (lr_motor, the bound: its fan-in d, the map's too; the
  builder's reading, for the lead: 7.6 says "act_pred's plain lesson" and no rate of its own); the map's input, a recalled act's
  embedding, is smaller than the LayerNorm'd stream act_pred reads, so it steps less per lesson, by the size of what is recalled, as
  the plain step does everywhere.
- THE WORKING-MEMORY LATCH (wm_frames): working memory latches the striatal expansion at the frames' event ends (R7b) in place of the
  utterances' ends; the language body's latch is unchanged.
- C51'S INSTRUMENT (step R8d; `heading_drift`): the heading against the world's true yaw (the frame's truth, never the body's), in the
  night's report at dusk and in `insides`: reported, never corrected."""
import math

import torch
import torch.nn.functional as F
from .memory import GOAL_TAU                                        # A130: the held word's time constant

from .physiology import FRAMES


def recall_spec(anatomy, cfg):
    """WHAT THE ORGANS BUILD FOR RECALL INTO ACTION (Organs(..., recall=)): None while the switch is off (the diary's); else each motor
    effector's name and its settings' count (its map's outputs). The switch needs `frames` (the frames the store recalls). A map per
    MOTOR effector, nine for the G1: the words' effector's proposal is the cortex's forecast read through the lexicon, which already
    reads the store's recall through the hippocampal pathway (7.6: "before R7f, recall reached only the words"), and it has no act_pred
    for a map to learn beside (THE BUILDER'S READING, for the lead: 7.6's "ten small maps" counts the words' effector too)"""
    c = cfg or {}
    if not int(c.get("recall", FRAMES["recall"])):
        return None
    if not int(c.get("frames", FRAMES["frames"])):
        raise ValueError("recall into action (recall 1) needs the frames it recalls (frames 1)")
    from .anatomy import motor_effectors
    return dict(effectors=[(e.name, sum(int(k) for k in e.factors)) for e in motor_effectors(anatomy.effectors)])


def read_event_lines(events, obs):
    """THE BORN RULE (step R7a; the module's doc) on one frame's observations `obs` ({name: numbers}): per declared line, in order, 1.0 where
    it fires and 0.0 where not: any of its numbers above 0; on its side where it declares one (the direction's sense times the line's
    above 0), both lines of a pair where the direction is exactly 0 (no side); and none of the lines further along its limb's chain
    firing (isolation: those are judged by the first two rules alone)"""
    raw = {}
    for ln in events:
        o_ = obs.get(ln.obs)
        f_ = o_ is not None and any(float(o_[int(i)]) > 0.0 for i in ln.fired)
        if f_ and ln.side is not None:
            v_ = float(o_[int(ln.side[0])])
            f_ = v_ == 0.0 or float(ln.side[1]) * v_ > 0.0              # its side, or no side at all (both lines of the pair)
        raw[ln.name] = f_
    return tuple(1.0 if raw[ln.name] and not any(raw[d_] for d_ in ln.distal) else 0.0 for ln in events)


class FramesMixin:
    # ---------------- step R7a: the event lines ----------------
    def _event_lines(self, frame=None):
        """THE TICK'S EVENT LINES (the module's doc; `read_event_lines`): one number per declared line, 1.0 where it fires and 0.0 where
        not, in the declared order; read once a tick from the frame the tick is lived on (`frame`, the tick's own readers pass it) and
        kept for the tick; with no frame (an instrument's read, between ticks) the lines of the tick last read, never read afresh (the
        tick's count has moved on then while the world's frame has not); () for a body that declares none"""
        ev = self.anatomy.events
        if not ev:
            return ()
        now_ = getattr(self, "_events_now", None)
        if frame is None:
            return now_[1] if now_ is not None else tuple(0.0 for _ in ev)
        if now_ is not None and now_[0] == self.ticks:
            return now_[1]
        out = read_event_lines(ev, frame.obs)
        self._events_now = (self.ticks, out)
        return out

    def _events_push(self, u):
        """the tick's fired event lines enter the striatum's event line (step R7a; called in `_hear` with the heard events, before the
        striatal input is read): one event, their bits, when any fired; under stri_quiet an empty one when none did (the line carries
        time, as the language line's quiet does)"""
        ev = self._event_lines(self._tick_frame(u))
        mask = 0
        for i_, v_ in enumerate(ev):
            if v_:
                mask |= 1 << i_
        if mask:
            self.m.striatum_push_events(mask)
        elif int(self.cfg.get("stri_quiet", 0)):
            self.m.striatum_push_events(0)

    # ---------------- step R7b: the frames ----------------
    def _frame_const(self, k):
        """a frame constant: the body's cfg when it was given, else FRAMES'"""
        return self.cfg.get(k, FRAMES[k])

    def _frames_on(self):
        """the switch `frames` (step R7b): the body lives in frames (0 for the diary, whose cfg holds no such key)"""
        return bool(int(self.cfg.get("frames", FRAMES["frames"])))

    def _frame_codes(self, u):
        """THE FRAME AS THE CORTEX RECEIVES IT: each channel's code of this tick's frame ({name: [d]}, detached; the words' the row of the
        symbol heard, the rest's in the quiet) and their sum in the anatomy's order (the cortex's input before its LayerNorm, without the
        ladder's bundle and the efference copies)"""
        m = self.m; codes = {}; tot = None
        with torch.no_grad():
            for c_ in self.anatomy.channels:
                o_ = c_.observe(self, u, 0)
                x_ = torch.tensor(int(o_), device=self.dev) if c_.kind == "symbol" else o_
                code = c_.encode(m, x_).detach()
                codes[c_.name] = code
                tot = code if tot is None else tot + code
        return codes, tot

    def _frame_surprise(self, codes):
        """THE FRAME'S SURPRISE (the module's doc): the mean over the forecasting channels of each one's error on this frame over its own
        running mean (each mean taking this tick's error first); None when no channel foresaw it (after birth or a night)"""
        st = getattr(self, "_ferr", None)
        if st is None:
            st = {}; self._ferr = st
        fc = getattr(self, "_ffc", None) or {}
        fw = getattr(self, "_fw_err", None)
        tau = float(self._frame_const("err_tau")); vals = []
        for i_, c_ in enumerate(self.anatomy.channels):
            if i_ == 0:
                e = fw
            elif c_.name in fc:
                with torch.no_grad():
                    e = float(0.5 * ((fc[c_.name].float() - codes[c_.name].float()) ** 2).sum())
            else:
                e = None
            if e is None:
                continue
            n_, mu = st.get(c_.name, [0, 0.0])
            n_ += 1; mu = mu + (float(e) - mu) / min(float(n_), tau)
            st[c_.name] = [n_, mu]
            if mu > 0.0:
                vals.append(float(e) / mu)
        return (sum(vals) / len(vals)) if vals else None

    def _frame_settle(self, s):
        """THE EVENT'S END FOR FRAMES (the module's doc): today's settle law on the frame's surprise `s`; True on the first settled tick
        after one that was not"""
        af = 1.0 / max(1.0, float(self.cfg.get("offset_fast", 4))); as_ = 1.0 / max(1.0, float(self.cfg.get("offset_slow", 64)))
        fast = getattr(self, "_fs_fast", None); slow = getattr(self, "_fs_slow", None)
        fast = float(s) if fast is None else (1.0 - af) * fast + af * float(s)
        slow = float(s) if slow is None else (1.0 - as_) * slow + as_ * float(s)
        self._fs_fast, self._fs_slow = fast, slow
        settled = fast <= float(self.cfg.get("offset_settle", 0.5)) * max(1e-6, slow)
        ended = settled and not getattr(self, "_fs_settled", True)
        self._fs_settled = settled
        return ended

    def _frame_end(self):
        """an event ends (the module's doc): the last frame written marked as its end, the next to be written as the next one's start,
        the tick kept in the day's ends"""
        lw = getattr(self, "_flast_write", None)
        if lw is not None:
            self.store.mark_boundary(*lw)
        self._flast_write = None; self._fstart_armed = True
        if int(self._frame_const("wm_frames")) and int(self.cfg.get("wm", 0)) and getattr(self.m, "stri_wm", 0):
            with torch.no_grad():                                      # R7f: working memory latches at the frames' event end (wm_frames)
                self.m.wm_latch(self.m.striatum_read())
        ends = getattr(self, "_rec_ends", None)
        if ends is None:
            ends = []; self._rec_ends = ends
        ends.append(int(getattr(self, "_rec_n", 0)))

    def _frame_gate(self, g):
        """THE WRITE GATE (the module's doc): whether `g` (the frame's surprise x (1 + tag)) passes its own running write_q quantile, in
        log units, stepped by write_eta after the test; its first 1/write_eta samples settle it as their sample quantile, and nothing
        passes while it settles"""
        if not g > 0.0:
            return False
        lx = math.log(float(g)); eta = float(self._frame_const("write_eta")); p = float(self._frame_const("write_q"))
        q = getattr(self, "_fq", None)
        if q is None:
            w_ = getattr(self, "_fq_warm", None)
            if w_ is None:
                w_ = []; self._fq_warm = w_
            w_.append(lx)
            if len(w_) >= max(1, int(round(1.0 / max(eta, 1e-9)))):
                self._fq = float(self._pace_q(w_, p)); self._fq_warm = None
            return False
        passed = lx > float(q)
        self._fq = float(q) + eta * (p - (1.0 if lx <= float(q) else 0.0))
        return passed

    def _frame_write(self, key, value, strength, base=None, tag_w=0.0):
        """the frame written (the module's doc): its codes under the key of the stream before it, at `strength`, who 2; its start mark
        when an event ended since the last write; kept as the event's last frame for its end mark; under the amygdala (R7d) its later
        boosts pending (`base` its strength without the tag, `tag_w` the tag it was written with). True when the store kept it"""
        st = self.store
        if not st.write(key, value, float(strength), 2):
            return False                                                # merged into a slot it already had: nothing new
        self._frame_novel = float(strength)                             # A127: a frame the store kept as NEW (the hippocampal mismatch,
        self._boosts_remap(st.last_remap)                             # Lisman and Grace 2005): the novelty source reads it next tick
        self._flast_write = (key.clone(), value.clone())
        if getattr(self, "_fstart_armed", False):
            st.mark_start(key, value); self._fstart_armed = False
        self._fwrites = int(getattr(self, "_fwrites", 0)) + 1
        if base is not None and self._amyg_on() and st.last_idx >= 0:
            bs = getattr(self, "_fboosts", None)
            if bs is None:
                bs = []; self._fboosts = bs
            bs.append([int(st.last_idx), float(base), float(tag_w), int(self.ticks)])
        return True

    def _boosts_remap(self, remap):
        """THE BOOSTS FOLLOW THE SLOTS (R7d; 7.4: "the boost follows last_remap; a dropped slot gets nothing"): a write's eviction remap
        (old index -> new, -1 dropped) applied to every pending boost's slot, a dropped slot's boost let go (called after every store
        write while boosts are pending: the frames' here, the words' in body/core/cortex.py and body/core/memory.py)"""
        bs = getattr(self, "_fboosts", None)
        if remap is None or not bs:
            return
        keep = []
        for b in bs:
            j = int(b[0])
            j = int(remap[j]) if 0 <= j < int(remap.numel()) else -1
            if j >= 0:
                b[0] = j; keep.append(b)
        self._fboosts = keep

    def _frame_boosts(self, tag):
        """THE LATER BOOSTS (R7d; SIM_DESIGN.md 7.4 item 1): for tag_reach ticks after a frame's write, each larger g^dt x tag (g dopamine's
        discount, dt the ticks since the write, `tag` this tick's) adds s0 x (its increase in (1 + g^dt tag)) to the slot, s0 the write's
        strength without the tag, through the store's own saturating merge (store_sat: s x m / (m + S), m the store's mean strength; else
        added whole); the factor never passes 3 (the tag's cap is 2)"""
        bs = getattr(self, "_fboosts", None)
        if not bs:
            return
        st = self.store; g = self._tag_gamma(); reach = int(self._frame_const("tag_reach")); keep = []
        for b in bs:
            j, s0, T, t0 = int(b[0]), float(b[1]), float(b[2]), int(b[3])
            dt = int(self.ticks) - t0
            if dt >= reach:
                continue
            if dt >= 1:
                c = (g ** dt) * float(tag)
                if c > T:
                    if 0 <= j < st.n():
                        ds = s0 * (c - T)
                        with torch.no_grad():
                            if st.saturate:
                                m_ = float(st.S.mean())
                                st.S[j] += ds * m_ / (m_ + float(st.S[j]))
                            else:
                                st.S[j] += ds
                    b[2] = c
            keep.append(b)
        self._fboosts = keep

    def _record_tick(self, s, delta, tag, r):
        """THE TICK'S RECORD (the module's doc): (surprise, dopamine, tag, the net reward received) as float32, a row of the day's record"""
        rec = getattr(self, "_rec", None); n_ = int(getattr(self, "_rec_n", 0))
        if rec is None or n_ >= int(rec.shape[0]):
            new = torch.zeros(max(4096, 2 * (int(rec.shape[0]) if rec is not None else 0)), 4, dtype=torch.float32)
            if rec is not None:
                new[:n_] = rec[:n_]
            rec = new; self._rec = rec
        rec[n_] = torch.tensor([float(s), float(delta), float(tag), float(r)], dtype=torch.float32)
        self._rec_n = n_ + 1

    def _frame_key(self, C):
        """the key a frame is written under: the stream's state `C` pattern-separated (the store's cortex key, `query_from`); under
        key_form "cortex" the very key the tick's own step made (its running mean moved there), else made here, the running mean moving
        once a tick"""
        if str(self.cfg.get("key_form", "bag")) == "cortex" and getattr(self, "_q_prev", None) is not None:
            return self._q_prev
        return self.query_from(C, learn=True)

    def _frame_foresee(self):
        """at the tick's end, from the stream after its own step: each later forecasting channel's forecast of the next frame's code, and
        the key the next frame will be written under (none after a night, before the first step)"""
        C = getattr(self, "_C_last", None)
        if C is None:
            self._ffc = None; self._fkey_prev = None
            return
        with torch.no_grad():
            self._ffc = {c_.name: self.m.head(self.anatomy, i_)(C).detach() for i_, c_ in enumerate(self.anatomy.channels) if i_ and c_.forecast}
        self._fkey_prev = self._frame_key(C)

    def _tag_now(self):
        """the tag of this tick and the tag at a write (this tick's received part plus the forecast made the tick before, capped at the
        judgment's clip): the amygdala's (R7d, body/core/amygdala.py); 0 while it is off"""
        now = getattr(self, "_amyg_now", None) if self._amyg_on() else None
        if now is None:
            return 0.0, 0.0
        cap = self.anatomy.rewards[0].clip
        tw = float(now["R"]) + float(getattr(self, "_amyg_prev", 0.0))
        return float(now["tag"]), (min(float(cap), tw) if cap is not None else tw)

    def _frame_tick(self, u, delta, r, nxt=None):
        """THE FRAME'S TICK (the module's doc), at the tick's end: the frame's codes and surprise, the event's end, the gated write (its
        value with the efference copies under recall, R7f; `nxt` the voice's act this tick), the tick's record, the forecast and the key
        for the next frame"""
        codes, total = self._frame_codes(u)
        if self._recall_on():
            total = self._frame_value(total, self.sil if nxt is None else nxt)
        s = self._frame_surprise(codes)
        tag, tag_w = self._tag_now()
        self._frame_boosts(tag)                                       # R7d: the tag of this tick reaching back onto the frames written
        if s is not None:
            if self._frame_settle(s):
                self._frame_end()
            key = getattr(self, "_fkey_prev", None)
            if key is not None and self._frame_gate(float(s) * (1.0 + float(tag_w))):
                base = float(s) * (1.0 + abs(float(delta)))
                self._frame_write(key, total, base * (1.0 + float(tag_w)), base=base, tag_w=tag_w)
        self._record_tick(s if s is not None else 0.0, delta, tag, r)
        self._goal_trace(nxt)                                         # A130: the word it said this tick held for the next keys
        self._frame_foresee()
        if self._night_frames_on():
            self._tape_tick()                                         # step R8: the tick taped beside its record (body/core/sleep.py)

    def _goal_trace(self, nxt):
        """A130 (goal_key, the brain sprint): THE HELD WORD, private speech as a key. The body's own said word (the voice's symbol this
        tick, not silence and not the word boundary) is held as a unit direction in the stream's space, its lexicon row, and fades with
        the time constant GOAL_TAU ticks (body/core/memory.py); while it holds, every key the frames make carries it (`query_from`), so
        what is written and what is recalled into the effectors' proposals (R7f) is conditioned on the word it last said: the prefrontal
        trace steering hippocampal retrieval (Vygotsky's private speech; Miller and Cohen 2001). A new word replaces the last; the night
        lets it go. Off (goal_key 0, the sim at birth) nothing is kept"""
        if not (int(self.cfg.get("goal_key", 0)) and self._recall_on()):
            return
        g = getattr(self, "_goal", None)
        if nxt is not None and int(nxt) != int(self.sil) and int(nxt) != int(getattr(self, "space_id", -1)):
            with torch.no_grad():
                self._goal = F.normalize(self.m.E.weight[int(nxt)].detach().float(), dim=0)
        elif g is not None:
            with torch.no_grad():
                g.mul_(1.0 - 1.0 / float(GOAL_TAU))
                if float(g.norm()) < 1e-3:
                    self._goal = None

    def _frames_nightfall(self):
        """THE NIGHT ENDS EVERY EVENT (the module's doc; called as the night begins): an event still open ends at nightfall"""
        if not getattr(self, "_fs_settled", True):
            self._frame_end()

    def _words_store(self):
        """THE STORE'S WORDS ALONE (step R7b), for the words' dreams of a body in frames: a copy of the store without its frames (who 2),
        its links, marks and episodes remapped as the store's own compaction remaps them (`Store._keep`); the store itself untouched. The
        frames are the night over frames' (R8's), never a words' dream: read as words they ran the pattern completion on through them
        (THE BUILDER'S READING, for the lead: until R8, a body in frames dreams its words as a body without frames would)"""
        from ..model import Store
        st = self.store
        ws = Store(st.d, cap=st.cap, temp=st.temp, device=st.dev, read_strength=st.read_strength, links=st.NK)
        ws.load_state_dict(st.state_dict()); ws.saturate = st.saturate
        ws._keep(torch.nonzero(st.W != 2).flatten())
        ws.last_remap = None
        return ws

    def _frames_night(self):
        """the frames' working state wakes fresh after the night (called with the rest of it): no forecast, no key and no last frame
        carried over, the next frame written the morning's start; the day's record and its ends let go (R8 cuts them into episodes
        first); the running means, the settle law's averages and the write gate's quantile kept"""
        self._ffc = None; self._fkey_prev = None; self._fw_err = None; self._flast_write = None; self._fstart_armed = True
        self._rec = None; self._rec_n = 0; self._rec_ends = []; self._goal = None; self._ctx = None   # A130/A128: the held word and the held
                                                                                                        # context let go with the day
        if getattr(self, "_fboosts", None) is not None:
            self._fboosts = []                                        # R7d: the later boosts end at the night (its fade remaps the slots)

    # ---------------- step R7c: the error scales, the received tag and its reach onto an utterance (tag_trace, defect 6) ----------------
    def _err_scales(self):
        """STEP R7c (err_scale): each channel's running mean of its forecast error, {name: mean}, by which the waking lesson divides that
        channel's head's error (body/core/cortex.py); None while err_scale is off (the diary's), a channel with no mean yet absent"""
        if not int(self._frame_const("err_scale")):
            return None
        st = getattr(self, "_ferr", None) or {}
        return {k: float(v[1]) for k, v in st.items() if float(v[1]) > 0.0}

    def _tags_on(self):
        """whether this body keeps the received tag each tick: under tag_trace (defect 6, R7c) or with the amygdala on (R7d)"""
        return bool(int(self.cfg.get("tag_trace", 0))) or bool(int(self.cfg.get("amyg", 0)))

    def _received_tag(self, terms):
        """THE RECEIVED TAG (SIM_DESIGN.md 7.4: R_t = the sum of |term| over the sources that reach the amygdala, what was felt this tick,
        capped at the judgment's clip, source 0's): `terms` the tick's terms by source"""
        cap = self.anatomy.rewards[0].clip
        tot = 0.0
        for s_ in self.anatomy.rewards:
            if s_.amyg and s_.name in terms:
                tot += abs(float(terms[s_.name]))
        return min(float(cap), tot) if cap is not None else tot

    def _tag_gamma(self):
        """dopamine's own discount a tick (the dopamine band's: 0.9375), the tag's reach back (7.4: no new time constant)"""
        return float(self.m.gammas()[int(self.cfg["dopamine_band"])])

    def _tag_entry(self, f_):
        """DEFECT 6 FIXED (tag_trace, R7c; physiology.py SWITCHES), an utterance's entry at its end: x = its felt strength `f_` x (1 + T), T
        the largest received tag over its ticks (and this tick's); relative to the bias-corrected running mean of the entries before it
        (defect 7: saved, and corrected as it forms, m_n = b m_n-1 + (1 - b) x_n over (1 - b^n), b = 1 - 1/utt_entry_tau), the first
        entry 1; kept for the reach back (`_tag_boosts`) while its 64 ticks run"""
        T = max(float(getattr(self, "_utt_tag", 0.0)), float(getattr(self, "_rtag_now", 0.0)))
        x = float(f_) * (1.0 + T)
        beta = 1.0 - 1.0 / max(1.0, float(self.cfg.get("utt_entry_tau", 64)))
        m_, n_ = float(getattr(self, "_utt_felt_m", 0.0)), int(getattr(self, "_utt_felt_n", 0))
        mhat = m_ / (1.0 - beta ** n_) if n_ > 0 else 0.0
        entry = x / mhat if mhat > 1e-6 else 1.0
        self._utt_felt_m = beta * m_ + (1.0 - beta) * x; self._utt_felt_n = n_ + 1
        self._utt_tag = 0.0
        if mhat > 1e-6:
            bs = getattr(self, "_utt_boosts", None)
            if bs is None:
                bs = []; self._utt_boosts = bs
            bs.append([int(self._utt_serial), float(f_), float(mhat), float(T), int(self.ticks)])
        return entry

    def _tag_boosts(self):
        """THE TAG REACHES BACK (tag_trace): for the tag_reach ticks after an utterance's end, a received tag g^dt x R larger than its T
        raises its entry to f (1 + that) / the mean it was entered against (g dopamine's discount); after them, or at the night, none"""
        bs = getattr(self, "_utt_boosts", None)
        if not bs:
            return
        g = self._tag_gamma(); R = float(getattr(self, "_rtag_now", 0.0)); reach = int(self._frame_const("tag_reach")); keep = []
        for b in bs:
            serial, f_, mhat, T, t0 = b
            dt = int(self.ticks) - int(t0)
            if dt >= reach:
                continue
            if dt >= 1:
                c = (g ** dt) * R
                if c > T:
                    b[3] = c
                    if serial in self.utt_N:
                        self.utt_S[self.utt_N.index(serial)] = float(f_) * (1.0 + c) / float(mhat)
            keep.append(b)
        self._utt_boosts = keep

    def _frames_check(self):
        """THE SWITCHES THAT NEED THE FRAMES (step R7), checked at birth and at a load: err_scale divides each channel's error by the
        frames' own running means (R7c) and wm_frames latches working memory at the frames' event ends (R7f), so either without `frames`
        would be a silent no-op (and wm_frames would silence the utterances' latch): refused, as recall is (`recall_spec`). The diary
        holds none of the three keys, so nothing is refused"""
        if self._frames_on():
            return
        on = [k_ for k_ in ("err_scale", "wm_frames") if int(self._frame_const(k_))]
        if on:
            raise ValueError(f"Life: {' and '.join(on)} need the frames they read (frames 1): without them "
                             f"{'the scale has no running means' if 'err_scale' in on else ''}"
                             f"{' and ' if len(on) == 2 else ''}{'working memory would never latch' if 'wm_frames' in on else ''}")

    # ---------------- step R7f: recall into action ----------------
    def _recall_on(self):
        """the switch `recall` (step R7f): 0 for the diary, whose cfg holds no such key"""
        return bool(int(self.cfg.get("recall", FRAMES["recall"])))

    def _recall_attach(self):
        """BORN OR LOADED WITH THE SWITCH ON: the organs hold a map per motor effector (its settings wide) and the heading's code; with it
        off, organs that hold them are refused (a body is born with its switches: A20)"""
        has = "recall" in self.m._modules
        if not self._recall_on():
            if has:
                raise ValueError("Life: the organs hold recall's maps (m.recall) and the life's constants switch recall off (recall 0)")
            return
        spec = recall_spec(self.anatomy, self.cfg)
        got = [(n_, int(mp.out_features)) for n_, mp in self.m.recall.items()] if has else None
        if got != [(n_, int(k_)) for n_, k_ in spec["effectors"]] or "head_code" not in self.m._buffers:
            raise ValueError(f"Life: the organs' recall maps {got} are not the ones the anatomy declares ({spec['effectors']}; built by "
                             f"Organs(..., recall=recall_spec(anatomy, cfg)))")

    def _heading_step(self, frame):
        """THE HEADING (the module's doc): the trunk's yaw integrated from the torso gyro since birth, from the anatomy's heading source in
        this tick's frame: + dt x (the gyro's rate about the accelerometer's up); nothing on a tick the frame names no such source"""
        hd = self.anatomy.heading
        if hd is None:
            return
        o_ = frame.obs.get(hd.obs) if frame is not None else None
        if o_ is None:
            return
        a = [float(o_[int(i)]) for i in hd.acc]; w = [float(o_[int(i)]) for i in hd.gyro]
        n = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
        if n > 0.0:
            self._heading = float(getattr(self, "_heading", 0.0)) + float(hd.dt) * (w[0] * a[0] + w[1] * a[1] + w[2] * a[2]) / n

    def heading_drift(self):
        """C51'S INSTRUMENT (SIM_DESIGN.md 7.6, A45, C51; the core refactor's step R8d): the heading's drift against the world's true yaw, read
        from the frame the body lives this tick: its `truth` (the instruments' and the teacher's alone, never the body's) holds the world's
        own yaw of the torso since birth, unwrapped, in rad (`truth["yaw"]`: C73's list, the world's to supply); the drift is the heading
        less it, and wrapped to (-pi, pi]. Reported (the night's report at dusk, `insides`), NEVER CORRECTED: nothing of the body reads it.
        None where the body keeps no heading (recall off) or the frame carries no true yaw"""
        if not self._recall_on() or self.anatomy.heading is None:
            return None
        f_ = getattr(self.world, "now", None)
        y_ = None if f_ is None else f_.truth.get("yaw")
        if y_ is None:
            return None
        h_ = float(getattr(self, "_heading", 0.0)); d_ = h_ - float(y_)
        return {"heading": h_, "true_yaw": float(y_), "drift": d_, "drift_wrapped": math.atan2(math.sin(d_), math.cos(d_))}

    def _heading_code(self):
        """the heading's born code, a unit direction [d]: cos(heading) and sin(heading) through the organs' two fixed unit rows; None for a
        body without recall"""
        if not self._recall_on():
            return None
        h = float(getattr(self, "_heading", 0.0))
        hc = self.m.head_code
        return F.normalize(math.cos(h) * hc[0] + math.sin(h) * hc[1], dim=0)

    def _frame_value(self, total, nxt):
        """THE FRAME'S VALUE under recall (the module's doc): the frame's codes `total` plus the efference copy of every effector's act this
        tick (a motor effector's own act, its rows' sum, where not its rest; the voice's symbol `nxt`, its lexicon row, where not the rest)"""
        m = self.m; v = total.clone()
        with torch.no_grad():
            if int(nxt) != self.sil:
                v = v + m.E.weight[int(nxt)]
            for e_, st_ in zip(self.anatomy.motors, self.motor):
                a_ = int(st_["now"]["act"])
                if a_ != int(e_.rest_id):
                    v = v + m.acts[e_.name](torch.tensor(a_, device=self.dev))
        return v

    def _frame_recall(self, C1):
        """RECALL (the module's doc), at the choice: the recalled value for the stream the choice reads, the store's own search (under
        key_form "cortex" the words' read of this tick, else a read of its own with the same query), kept for the tick"""
        if str(self.cfg.get("key_form", "bag")) == "cortex":
            r = getattr(self, "_read_prev", None)
            r = torch.zeros(self.m.d, device=self.dev) if r is None else r
        else:
            r = self.store.read(self.query_from(C1, learn=False))[0] if self.store.n() > 0 else torch.zeros(self.m.d, device=self.dev)
        self._frec_now = r.detach().clone()
        return self._frec_now

    def _recall_term(self, e, r):
        """THE MAP'S TERM IN MOTOR EFFECTOR e's PROPOSAL (the module's doc) for recalled values `r` [.., d]: the recalled act's embedding (the
        recall's part in e's rows, R^T R r), through its map (d -> its settings), read back through its rows: [.., d]"""
        R = self.m.acts[e.name].rows                                   # [its settings, d]
        emb = (r @ R.t()) @ R
        return self.m.recall[e.name](emb) @ R
