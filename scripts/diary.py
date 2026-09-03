#!/usr/bin/env python3
"""diary.py — the diary body's serve (DIARY_BODY.md): two writers, one
page, one symbol per tick. No turns, no end marks, no breath, no hush:
silence is a symbol the body chooses. Reuses the organism's organs
(scripts/organism.py: the model and its memory, the face lesson,
cortisol, mood, doses, nights, save, reset) and replaces the turn
protocol with a clock.

  python3 scripts/diary.py data/organism_diary_0p5b.pt data/tok_char.json --dev mps \\
      --temp 0.05 --store-read-beta 1.0 --store-boost 16 --store-boost-min 0.15 \\
      --live-lr 1e-6 --store-decay 0.9 --save data/organism_diary_0p5b.pt --port 8018

Endpoints: GET / (the page), GET /state?since=N, GET /pulse;
POST /type {"text"}, /face {"expr"}, /sleep, /save, /reset.
"""
import collections
from os import environ as _os_env
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch  # noqa: E402
import scripts.organism as O  # noqa: E402


class Diary(O.Organism):
    def __init__(self, a):
        super().__init__(a)
        self.queue = collections.deque()          # your symbols waiting for ticks
        self.page = []                            # every shown symbol: (text, who, your face, its face)
        self.level = 0                            # the face level felt now
        self.credit = collections.deque(maxlen=64)  # the mouth's recent symbols: [id, credit]
        self.last = {}
        self.ticks = 0
        self.awake_ticks = True
        self.lock = threading.RLock()
        self.period = float(getattr(a, "diary_period", 0.5))
        self.who0 = torch.tensor([[0]], device=self.dev)
        self.who1 = torch.tensor([[1]], device=self.dev)   # the mouth's noise
        self.who2 = torch.tensor([[2]], device=self.dev)   # the mouth from memory (part of the thought)
        self._bans = [i for i in range(11) if i != self.sil]   # it may choose silence, never a mark
        self._nl = self.tok.token_to_id("\n")
        if self._nl is not None:
            self._bans.append(int(self._nl))                  # the newline is a page mark, not language (2026-09-02)
        if getattr(a, "sil_decay", None):
            self.m.kc_sil_decay = float(a.sil_decay)      # how long a thought lasts in silence
        # THREE LAWS REMOVED (2026-09-02, the user's order): a new line no longer ends a
        # thought (the bag only fades), your quiet is no longer stored as a memory, and
        # memory's voice is no longer raised over silence (no trust, no sure-memory boost).
        # The mouth chooses from its own belief, memory's vote inside it as the organ reads
        # it; only stamina leans it to silence.
        self.m.kc_break_ids = set()
        self.m.sil_id = None
        self.stream = collections.deque(maxlen=96)        # (id, who) of what was written, both hands
        self.page_base = 0                                # items dropped from the front of the page
        _ex = ((self.state_meta.get("life") or {}).get("extra") or {})
        # THE NIGHT (2026-09-02, the user's law): the cortex learns only from hippocampal
        # traces, and no index is kept on the body's behalf — no page, no transcript, no
        # list of lines. A dream starts where the memory itself is strongest (the store's
        # own key directions, below), and what follows is what the store holds.
        self.night_no_page = True
        # SLEEP BY ITS OWN FATIGUE (2026-09-02, the user's order): sleep pressure rises with
        # every waking tick (Process S, adenosine) and the body falls asleep by itself when
        # it crosses the switch; nobody posts its night. Persisted across restarts.
        self.wake_ticks = int(getattr(a, "wake_ticks", 12000) or 12000)
        self.sleep_pressure = int(_ex.get("sleep_pressure", 0) or 0) if isinstance(_ex, dict) else 0
        self.nights = int(_ex.get("nights", 0) or 0) if isinstance(_ex, dict) else 0
        self.asleep = False
        self.last_night = None
        # THE MOUTH'S GO/NO-GO (2026-09-03): the gate organ (in the model, saved with the body)
        # has the striatum's own plasticity: a small optimizer of its own, a lesson every few
        # ticks on detached features, credit = dopamine minus the cost of acting
        self.gate_on = bool(int(getattr(a, "gate", 1))) and hasattr(self.m, "mouth_gate")
        # FATIGUE, STRESS, MOOD (2026-09-03, the user's question "is stamina in the wrong category?"):
        # under --affect split the effort cost is fatigue (it recovers with rest and never touches
        # mood by itself), stress is a leaky integral of the world's dopamine dips (expectations
        # that failed), and mood is a leaky integral of dopamine, both signs. Constants disclosed:
        # stress +0.5 per unit of dip, mood +0.25 per unit of dopamine, both with the fatigue's
        # half-life of two minutes for stress and ten for mood.
        self.affect_split = str(getattr(a, "affect", "old")) == "split"
        self.stress = 0.0
        self.gate_buf = collections.deque(maxlen=96)     # (feat [d+2] cpu, acted, dopamine at that tick)
        # the striatum's rule is local and plain: Hebbian steps (SGD) on normalized features
        self.opt_gate = torch.optim.SGD(self.m.mouth_gate.parameters(), lr=float(getattr(a, "gate_lr", 0.05))) \
            if self.gate_on else None
        self._gate_p = None
        self._gate_last = None
        self.m.reset_bag(self.st) if hasattr(self.m, "reset_bag") else None
        threading.Thread(target=self._loop, daemon=True).start()

    def _wake_lesson(self):
        """THE WAKING CORTEX (2026-09-03, the user: the cortex should always be predicting the next
        state it receives): one step on the last window of the stream at the live rate —
        cross-entropy on the symbols the world wrote (the mouth's own are inputs, never targets:
        what it wrote itself was foretold by its efference copy), plus the forecast of the next
        band state with SIGReg, the same organ REM trains. A non-finite lesson is skipped; the
        gradient is clipped."""
        seq = list(self.stream)[-int(getattr(self.a, "wake_window", 32) or 32):]
        if len(seq) < 8:
            return None
        ids = [i for i, _ in seq]; who = [w for _, w in seq]
        x = torch.tensor([ids[:-1]], device=self.dev); wx = torch.tensor([who[:-1]], device=self.dev)
        y = torch.tensor([ids[1:]], device=self.dev)
        tgt_world = torch.tensor([[1.0 if w == 0 and t != self.sil else 0.0 for t, w in zip(ids[1:], who[1:])]],
                                 device=self.dev)   # the world's symbols only, and not its quiet
        if float(tgt_world.sum()) < 1:
            return None
        m = self.m
        m.train()
        try:
            self.opt.zero_grad(set_to_none=True)
            st_d = m.init_state(1, self.dev)
            lg, st_d, _ = m(x, st_d, None, who=wx)
            own = getattr(m, "_last_logits_own", None)
            own = lg if own is None else own
            ce = torch.nn.functional.cross_entropy(own[0].float(), y[0], reduction="none")
            loss = (ce * tgt_world[0]).sum() / tgt_world.sum()
            fw = float(getattr(self.a, "wake_forecast", 1.0) or 0.0)
            fl, cosv = (None, None)
            if fw > 0 and hasattr(m, "rem_pfc_loss"):
                fl, cosv = m.rem_pfc_loss(sigreg=float(getattr(self.a, "night_sigreg", 0.1) or 0.0), slot_from=1)
                if fl is not None and bool(torch.isfinite(fl.detach())):
                    loss = loss + fw * fl
            self._pop_side_losses()
            if not bool(torch.isfinite(loss.detach())):
                self.opt.zero_grad(set_to_none=True)
                return {"skipped": "non-finite", "tick": self.ticks}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            self.opt.step(); self.n_steps += 1
            self.opt.zero_grad(set_to_none=True)
            out = {"ce_world": round(float((ce * tgt_world[0]).sum() / tgt_world.sum()), 3),
                   "n_world": int(tgt_world.sum()), "forecast_cos": (None if cosv is None else round(float(cosv), 3)),
                   "tick": self.ticks}
        finally:
            m.eval()
        self._wake_last = out
        return out

    def _gate_feat(self):
        """what the gate reads: the cortex stream at the last position (detached, normalized)
        and interoception (stress, mood)"""
        C = getattr(self.m, "_last_C_live", None)
        if C is None:
            f = torch.zeros(self.m.d, device=self.dev)
        else:
            with torch.no_grad():
                f = self.m.lnf(C[0, -1].float().detach()).to(self.dev) / math.sqrt(float(self.m.d))   # unit norm
        io = torch.tensor([self.cortisol / 10.0, self.mood / 6.0, self.stress / 10.0], device=self.dev, dtype=f.dtype)
        return torch.cat([f, io]).detach()

    def _gate_lesson(self):
        """the actor's lesson: each choice to act or rest in the buffer takes credit from the
        dopamine that followed it (twelve ticks, 0.8 per tick) minus the cost of a symbol if it
        acted, and the gate is pushed toward what paid — the striatum's local rule, on detached
        cortex features (no force reaches the council)"""
        buf = list(self.gate_buf)
        if len(buf) < 16:
            return
        cost = float(getattr(self.a, "gate_cost", 0.12) or 0.0)
        w_int = float(getattr(self.a, "gate_int", 1.0) or 0.0)
        f0 = float(getattr(self.a, "gate_fatigue", 10.0) or 0.0)
        n = len(buf) - 12                                 # only choices whose eligibility window is complete
        feats = torch.stack([b[0] for b in buf[:n]]).to(self.dev)
        acts = torch.tensor([1.0 if b[1] else 0.0 for b in buf[:n]], device=self.dev)
        G = torch.zeros(n, device=self.dev)
        for t in range(n):
            g = 0.0
            for k_ in range(12):
                g += (0.8 ** k_) * float(buf[t + k_][2])   # the world's dopamine that followed
            if buf[t][1]:
                # its own reward at the symbol, and the symbol's cost, which a tired body pays more
                c_t = cost * (1.0 + (float(buf[t][4]) / f0 if f0 > 0 else 0.0))
                g += w_int * float(buf[t][3]) - c_t
            G[t] = g
        if float(G.abs().max()) < 1e-4:
            return
        # THE OPPONENT RULE (D1/D2): a burst strengthens Go for the context it came in, whatever
        # the body happened to do; a dip strengthens NoGo. The gate's logit moves with the sign
        # of the credit at each tick (a policy gradient instead credited the rests that filled a
        # smile's window and closed the gate for good: measured 2026-09-03 08:20)
        self.m.mouth_gate.train()
        z = self.m.mouth_gate(feats).squeeze(-1)
        loss = -(G.detach() * z).mean()
        self.opt_gate.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.m.mouth_gate.parameters(), 1.0)
        self.opt_gate.step()
        self.m.mouth_gate.eval()
        self._gate_last = {"n": n, "credit_mean": round(float(G.mean()), 4), "acted": round(float(acts.mean()), 3),
                           "int_mean": round(float(sum(b[3] for b in buf[:n] if b[1]) / max(1, sum(1 for b in buf[:n] if b[1]))), 4),
                           "tick": self.ticks}
        for _ in range(min(len(self.gate_buf), int(getattr(self.a, "gate_every", 24) or 24))):
            self.gate_buf.popleft()

    # ---- the clock ----
    def _loop(self):
        while True:
            t0 = time.time()
            if self.awake_ticks:
                try:
                    with self.lock:
                        self.tick()
                except Exception as e:  # never let one tick stop the clock
                    self.last = {"error": str(e)[:200], "tick": self.ticks}
            time.sleep(max(0.0, self.period - (time.time() - t0)))

    def tick(self):
        """one tick: your symbol (or silence) enters as the ear; its symbol
        (or silence) is chosen and enters as the mouth"""
        self._decay_mood(); self._decay_cort()
        u = self.queue.popleft() if self.queue else self.sil
        face = float(self.face_now)
        lvl = max(-6, min(6, int(face)))
        felt = 0
        if lvl != self.level:
            # only a change of face is felt, and relaxing toward neutral is not an
            # event: a face that grows or flips sign speaks, one that eases off is silent
            if abs(lvl) > abs(self.level) or (lvl * self.level < 0):
                felt = lvl
            self.level = lvl
        pl = None
        if felt:
            pl = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
            pl[0, 0] = min(felt, 2) if felt > 0 else 2 - max(felt, -2)   # 1..4 = +1 +2 -1 -2
        aff = self._aff(1)
        # the mouth has the floor when its previous symbol was a letter: then your silence
        # is its turn to write, not the end of your thought
        self.st["mouth_floor"] = bool(getattr(self, "_prev_mouth", self.sil) != self.sil)
        with torch.no_grad():
            lg, self.st, _ = self.m(torch.tensor([[u]], device=self.dev), self.st,
                                    press_levels=pl, affect=aff, who=self.who0)
        _rs = getattr(self.m, "_rpe_signed", None)
        delta_ear = float(_rs[0, -1]) if _rs is not None and _rs.numel() else 0.0   # the world's reward error at the ear's step
        its_face = self._face_lesson(face)
        # the mouth chooses: its own logits, memory's vote inside them
        v = lg[0, -1].float().clone()
        v[self._bans] = float("-inf")
        _lv = getattr(self.m, "_last_votes", None)
        own_ent = float(getattr(self.m, "_last_own_ent", 0.0) or 0.0)
        mem_max = float(_lv[0][0]) if _lv and _lv[0] else 0.0
        p_act = None
        if self.gate_on:
            # WHETHER to act is the gate's (basal ganglia): the content softmax never holds rest,
            # and no stress lean is written by hand — stress is an input the gate can learn from
            with torch.no_grad():
                feat = self._gate_feat()
                p_act = float(torch.sigmoid(self.m.mouth_gate(feat.unsqueeze(0)))[0, 0])
            v[self.sil] = float("-inf")
        elif self.cortisol > 0:                    # the old brake: stress leans the content softmax to silence
            v[self.sil] = v[self.sil] + float(getattr(self.a, "cort_k", 0.5)) * self.cortisol
        # no hand-written voice for memory (removed 2026-09-02): its vote is already in the
        # logits as the organ reads it, and the mouth chooses from that belief alone
        p1 = torch.softmax(v, -1)
        ent = float(-(p1 * (p1 + 1e-9).log()).sum() / math.log(max(2, p1.numel())))
        pr = torch.softmax(v / max(0.02, float(self.a.temp)), -1).cpu()
        int_t = 0.0
        if self.gate_on:
            acted = bool(torch.rand(1, generator=self.gen).item() < p_act)
            nxt = int(torch.multinomial(pr, 1, generator=self.gen)) if acted else self.sil
            if acted:
                # its own reward at the symbol: the belief it had in what it chose (producing the
                # sound it expected; the drive to babble, and to speak when it knows what it is
                # about to say). Absolute, 0..1: a reward relative to its own running mean fed on
                # runs of babble (measured 2026-09-03 08:10: one smile, then 98 percent acting).
                # HABITUATION (2026-09-03 09:05, after the flood): the reward is scaled by how
                # rarely this symbol was just chosen — a run pays less each time, a fresh symbol
                # pays in full (stimulus-specific adaptation; what keeps babble varied).
                # Never a lesson on content.
                hab = float(getattr(self.a, "gate_habit", 0.9) or 0.0)
                freq = getattr(self, "_sym_freq", None)
                if freq is None:
                    freq = self._sym_freq = {}
                novelty = 1.0 - freq.get(nxt, 0.0) if hab > 0 else 1.0
                int_t = float(pr[nxt]) * max(0.0, novelty)
                if hab > 0:
                    for k_ in list(freq.keys()):
                        freq[k_] *= hab
                        if freq[k_] < 1e-3:
                            del freq[k_]
                    freq[nxt] = freq.get(nxt, 0.0) + (1.0 - hab)
        else:
            nxt = int(torch.multinomial(pr, 1, generator=self.gen))
            acted = nxt != self.sil
        self._gate_p = p_act
        backed = bool(_lv and _lv[1] and int(_lv[1][0]) == nxt and nxt != self.sil
                      and mem_max >= float(getattr(self.a, "store_boost_min", 0.0) or 0.0))
        with torch.no_grad():
            # its symbol enters the stream as its own (speaker 1), whatever it came from
            _, self.st, _ = self.m(torch.tensor([[nxt]], device=self.dev), self.st, affect=aff, who=self.who1)
        self._prev_mouth = nxt
        if nxt != self.sil:
            # speaking costs: each symbol adds to the effort variable (half-life 120 s). Under the old
            # affect it is called stress, brakes the mouth and weighs on mood; under the split it is
            # fatigue, the gate's cost and input, and mood is dopamine's business
            self.cortisol += float(getattr(self.a, "diary_cost", 0.08))     # the cost of one symbol
            if not self.affect_split:
                self.mood = max(-6.0, min(6.0, self.mood - 0.002 * self.cortisol))
        self.stream.append((u, 0))
        self.stream.append((nxt, 2 if (backed and u == self.sil) else 1))
        # the day's record, as lived: both hands, silence included (a run of silence is kept
        # to one tick so the night sees pacing without drowning in it)
        if u != self.sil or not self.day_buf or self.day_buf[-1] != self.sil:
            self._who_now = 0; self.day_buf.append(u); self._rec_face(1)
        if nxt != self.sil or self.day_buf[-1] != self.sil:
            self._who_now = 1; self.day_buf.append(nxt); self._rec_face(1); self._who_now = 0
        self.credit.append([nxt, 0.0, bool(backed and u == self.sil)])   # every tick is a choice, silence included; memory-backed noted
        if felt and not self.affect_split:
            self.mood = max(-6.0, min(6.0, self.mood + 0.5 * felt))
        # DOPAMINE DOSES (2026-09-02, the user's aim: reward at every timescale): the fast
        # band's prediction error of the world's reward is the dopamine. At a felt face it
        # is the face minus what was expected; at a predictor of a face it fires before
        # any face (a secondary reinforcer, once the ladder has learned). Either way it
        # spreads over the last twelve ticks (six seconds, 0.8 per tick), and a burst
        # (|error| at least 0.5, the size of half a small smile) pays the lesson.
        rs = getattr(self.m, "_rpe_signed", None)
        delta = delta_ear + (float(rs[0, -1]) if rs is not None and rs.numel() else 0.0)   # both halves of the tick
        if self.affect_split:
            # mood is the leaky integral of dopamine, stress that of its dips: feelings from the
            # one reward system, not from effort
            self.mood = max(-6.0, min(6.0, self.mood + 0.25 * delta))
            self.stress = min(30.0, self.stress + 0.5 * max(0.0, -delta))
            self.stress *= 0.5 ** (self.period / 120.0)
        if abs(delta) >= 1e-3:
            for k_, item in enumerate(reversed(list(self.credit)[-12:])):
                item[1] += delta * (0.8 ** k_)
        if abs(delta) >= 0.5:
            self._dose_choices(level=int(pl[0, 0]) if pl is not None else 0)
        if int(getattr(self.a, "wake_lesson", 0) or 0) and self.ticks > 0 \
                and self.ticks % int(getattr(self.a, "wake_every", 24) or 24) == 0:
            try:
                self._wake_lesson()
            except Exception as e:
                self._wake_last = {"error": str(e)[:120], "tick": self.ticks}
        if self.gate_on:
            # [features, acted, the world's dopamine this tick, its own reward at the symbol, fatigue then]
            self.gate_buf.append([feat.detach().cpu(), acted, float(delta), int_t, float(self.cortisol)])
            if len(self.gate_buf) >= 16 + 12 and self.ticks % int(getattr(self.a, "gate_every", 24) or 24) == 0:
                try:
                    self._gate_lesson()
                except Exception as e:
                    self._gate_last = {"error": str(e)[:120], "tick": self.ticks}
        self.ticks += 1
        self.sleep_pressure += 1
        if self.sleep_pressure >= self.wake_ticks and len(self.day_buf) >= 65:
            # the switch flips: it falls asleep on its own, and wakes when the night is done
            self._sleep_now()
        if len(self.page) > 40000:                 # a bounded page: the front falls away, indices stay absolute
            drop = 20000; del self.page[:drop]; self.page_base += drop
        self.page.append(((self.tok.decode([u]) if u != self.sil else ""), 0, round(face, 2),
                          None if its_face is None else round(its_face, 2)))
        self.page.append(((self.tok.decode([nxt]) if nxt != self.sil else ""), 1, round(face, 2),
                          None if its_face is None else round(its_face, 2), bool(backed)))
        self.last = {"tick": self.ticks, "you": round(face, 2),
                     "face": None if its_face is None else round(its_face, 2),
                     "mood": round(self.mood, 2), "cort": round(self.cortisol, 2), "ent": round(ent, 2),
                     "mem": [[self.tok.decode([int(i_)]), round(float(v_), 2)] for v_, i_ in zip(*_lv)] if _lv else None,
                     "felt": felt, "said": self.tok.decode([nxt]) if nxt != self.sil else "", "backed": backed,
                     "own": ([self.tok.decode([self.m._last_own_top[0]]) if self.m._last_own_top[0] >= 11 else "<sil>",
                              round(self.m._last_own_top[1], 3)] if getattr(self.m, "_last_own_top", None) else None),
                     "dose": getattr(self, "_last_dose", None), "doses": getattr(self, "n_doses", 0),
                     "gate": (None if p_act is None else round(p_act, 3)), "gate_lesson": getattr(self, "_gate_last", None),
                     "fatigue": round(self.cortisol, 2), "stress": round(self.stress, 2), "affect": ("split" if self.affect_split else "old"),
                     "wake": getattr(self, "_wake_last", None)}

    def _dose_choices(self, level=0):
        """the only teacher is your face on what it actually did. GRADED (2026-09-02, no
        thresholds): every choice of the last twelve ticks (letters or silences) takes a
        step scaled by its own credit — up for credit, down for blame (its probability
        pushed toward a floor, never a hand-written replacement). The same felt face is a
        reward: it enters the lesson's forward as the press it was, so the value ladder
        learns from it at every band's timescale, and the basal-ganglia gate learns from
        the value's error. One system."""
        items = list(self.credit)
        if not items or max(abs(it[1]) for it in items) < 1e-3:
            return
        seq = list(self.stream)[-26:]                # the last thirteen ticks as lived, silences included
        if len(seq) < 4:
            for it in items: it[1] = 0.0
            return
        ids = [i for i, _ in seq]; who = [w for _, w in seq]
        n_m = len(items)
        # the mouth's recent choices sit at the end of the stream in order; map credit to them
        mouth_pos = [j for j in range(1, len(ids)) if who[j] != 0][-n_m:]
        wts = {}
        for k_, j in enumerate(mouth_pos):
            idx = n_m - len(mouth_pos) + k_
            c = float(items[idx][1])
            if abs(c) >= 1e-3:
                wts[j] = max(-2.0, min(2.0, c))
        if not wts:
            for it in items: it[1] = 0.0
            return
        x = torch.tensor([ids[:-1]], device=self.dev); wx = torch.tensor([who[:-1]], device=self.dev)
        y = torch.tensor([ids[1:]], device=self.dev)
        w = torch.zeros(1, len(ids) - 1, device=self.dev)
        for j, c in wts.items():
            if self.gate_on and ids[j] == self.sil:
                continue                                  # a rest is the gate's choice, not content
            w[0, j - 1] = c
        if self.gate_on and float(w.abs().sum()) < 1e-6:
            for it in items: it[1] = 0.0
            return
        pl_t = torch.zeros(1, len(ids) - 1, dtype=torch.long, device=self.dev)
        if level:
            pl_t[0, -1] = int(level)                  # the felt face, as the reward it was, at the tick it was felt
        vw = float(getattr(self.a, "value_w", 0.5) or 0.0)
        self.m.train()
        try:
            self.opt.zero_grad(set_to_none=True)
            st_d = self.m.init_state(1, self.dev)
            lg, st_d, _ = self.m(x, st_d, None, press_levels=pl_t, who=wx)
            ce = torch.nn.functional.cross_entropy(lg[0].float(), y[0], reduction="none")
            up = (ce * torch.relu(w[0])).sum()
            down = (torch.relu(3.0 - ce) * torch.relu(-w[0])).sum()   # blame pushes the symbol's loss to at least 3 nats
            loss = (up + down) / w.abs().sum().clamp(min=1e-3)
            vl = self.m.pop_value_loss() if hasattr(self.m, "pop_value_loss") else None
            bl = self.m.buffered_value_loss(self.st) if hasattr(self.m, "buffered_value_loss") else None
            if vw > 0:
                if vl is not None:
                    loss = loss + vw * vl
                if bl is not None:
                    loss = loss + vw * bl
            bg = self.m.pop_bg_loss() if hasattr(self.m, "pop_bg_loss") else None
            if bg is not None:
                loss = loss + bg
            self._pop_side_losses()
            loss.backward()
            self.opt.step()
        finally:
            self.m.eval()
        self._last_dose = {"up": int((w > 0).sum()), "down": int((w < 0).sum()), "level": int(level),
                           "value": (round(float(vl.detach()), 3) if vl is not None else None),
                           "ladder": (round(float(bl.detach()), 3) if bl is not None else None), "tick": self.ticks}
        self.n_doses = getattr(self, "n_doses", 0) + 1
        for it in items: it[1] = 0.0

    # ---- your hand ----
    def type_text(self, s):
        n = 0
        for ch in s:
            i = self.tok.token_to_id(ch)
            if i is not None and i >= 11 and i != self._nl and len(self.queue) < 600:   # a bounded queue; the newline is a page mark and never enters
                self.queue.append(i); n += 1
        return {"queued": len(self.queue), "took": n}

    def set_face(self, expr):
        try:
            self.face_now = max(-6.0, min(6.0, float(expr)))
        except Exception:
            self.face_now = 0.0
        return {"face": self.face_now}

    def state(self, since=0):
        if True:   # no lock: the night holds it for minutes, and a parent must still see the page and its sleep
            k = max(0, int(since) - self.page_base)
            return {"page": self.page[k:], "n": self.page_base + len(self.page), "last": self.last,
                    "queued": len(self.queue), "awake": self.awake_ticks, "period": self.period,
                    "asleep": self.asleep, "sleep_pressure": int(self.sleep_pressure), "wake_ticks": self.wake_ticks,
                    "nights": int(self.nights), "last_night": self.last_night,
                    "lived": len(self.day_buf)}

    def _life_extra(self):
        return {"sleep_pressure": int(self.sleep_pressure), "nights": int(self.nights)}

    # ---- the night ----
    def _dream_starts(self, n):
        """where a dream starts, from the memory itself: the strongest key directions of
        each band's store (its own structure; nothing is kept on its behalf), each turned
        back into a context the store can be queried with, ranked by the strength of the
        vote it draws. Spontaneous reactivation, as near as a matrix memory allows."""
        m = self.m
        M = self.st.get("M") if isinstance(self.st, dict) else None
        if not M:
            return []
        E = m.embed.weight.detach().float().cpu()
        floor = float(getattr(self.a, "store_boost_min", 0.0) or 0.0)
        norms = {k: float(v.abs().sum()) for k, v in M.items()}
        tot = sum(norms.values()) or 1.0
        starts = []
        for k, Mk in M.items():
            share = int(round(n * norms[k] / tot)) if norms[k] > 0 else 0
            if share <= 0 or str(k) not in m.stores:
                continue
            A = Mk[0].detach().float().cpu()                       # [d, D]: values by lifted keys
            q = max(1, min(share, min(A.shape) - 2))
            try:
                U, S, V = torch.svd_lowrank(A, q=q, niter=4)       # V [D, q]: the key directions
            except Exception:
                continue
            stn = m.stores[str(k)]
            P = stn.proj.detach().float().cpu(); ph = stn.phase.detach().float().cpu()
            for j in range(V.shape[1]):
                best = None
                for sign in (1.0, -1.0):                            # a singular direction has no sign of its own
                    u = sign * V[:, j]
                    x = torch.nn.functional.normalize(P.t() @ u, dim=0).clone().requires_grad_(True)
                    opt = torch.optim.Adam([x], lr=0.05)
                    for _ in range(40):                             # the store's key, turned back into a context
                        lift = torch.nn.functional.normalize(torch.cos(P @ torch.nn.functional.normalize(x, dim=0) + ph), dim=0)
                        loss = 1.0 - (lift * u).sum()
                        opt.zero_grad(); loss.backward(); opt.step()
                    with torch.no_grad():
                        xn = torch.nn.functional.normalize(x, dim=0)
                        lift = torch.nn.functional.normalize(torch.cos(P @ xn + ph), dim=0)
                        votes = (A @ lift) @ E.t()
                        votes[:11] = float("-inf")
                        v = float(votes.max())
                    if best is None or v > best[0]:
                        best = (v, xn.detach().clone())
                if best is not None and best[0] >= floor:
                    starts.append((best[0], k, best[1]))
        starts.sort(key=lambda s_: -s_[0])
        return starts[:n]

    def _dream_trace(self, key, max_len=48):
        """the hippocampus replays: from a context, memory's top vote is taken as the next
        symbol, the thought advances with it, and the trace ends where memory has nothing.
        No trunk vote, no stamina, no writes: what the store holds, in its own words."""
        m = self.m
        st = m.lane_state(self.st, 0)
        key = key.to(self.dev)
        edt = m.embed.weight.dtype
        st["bag_state"] = key.unsqueeze(0).to(edt)
        st["bag_prev"] = torch.nn.functional.normalize(key.unsqueeze(0), dim=-1).to(edt)
        st["mouth_floor"] = False
        st["ear_was_word"] = torch.zeros(1, dtype=torch.bool, device=self.dev)
        wo = getattr(m, "store_write_off", False); m.store_write_off = True
        ro = m.store_read_off; m.store_read_off = False
        floor = float(getattr(self.a, "store_boost_min", 0.0) or 0.0)
        ids, strengths = [], []
        fatigue = {}                                     # neural adaptation: a symbol replayed again and again tires
        try:
            with torch.no_grad():
                x, who = torch.tensor([[self.sil]], device=self.dev), self.who0
                for _ in range(max_len):
                    _, st, _ = m(x, st, who=who)
                    rd = getattr(m, "_last_rd_full", None)
                    if rd is None:
                        break
                    v = rd.clone(); v[:11] = float("-inf")
                    for s_, n_ in fatigue.items():
                        v[s_] = v[s_] - 0.5 * n_                 # each repeat costs half a vote: no attractor replays forever
                    top = int(v.argmax()); val = float(v[top])
                    if val < floor:
                        break
                    ids.append(top); strengths.append(round(val, 2))
                    fatigue = {s_: n_ * 0.7 for s_, n_ in fatigue.items()}   # and recovers as others speak
                    fatigue[top] = fatigue.get(top, 0.0) + 1.0
                    x, who = torch.tensor([[top]], device=self.dev), self.who0   # replayed as heard
        finally:
            m.store_write_off = wo; m.store_read_off = ro
        return ids, strengths

    def _student(self, ids, mem_on):
        """the cortex over a trace, teacher-forced, the trace as heard (the ear's hand);
        memory on for NREM (the hippocampus leads, the lesson lands on the trunk's own
        logits), memory set aside for REM (the PFC alone drives what the cortex must
        foresee). No face rides along: the face is learned by day."""
        m = self.m
        x = torch.tensor([[self.sil] + ids[:-1]], device=self.dev)
        y = torch.tensor(ids, device=self.dev)
        st = m.lane_state(self.st, 0) if mem_on else m.init_state(1, self.dev)
        m.reset_bag(st)
        st["mouth_floor"] = False
        ro, wo = m.store_read_off, getattr(m, "store_write_off", False)
        m.store_read_off, m.store_write_off = (not mem_on), True
        try:
            lg, _, _ = m(x, st, who=torch.zeros_like(x))
        finally:
            m.store_read_off, m.store_write_off = ro, wo
        return lg, y

    def _pop_side_losses(self):
        for name in ("pop_value_loss", "pop_face_loss", "pop_plan_aux", "pop_ponder_loss", "pop_route_aux", "pop_bg_loss"):
            if hasattr(self.m, name):
                getattr(self.m, name)()

    def _gauge(self, traces, cap=24):
        """the morning gauge: over each trace, the trunk ALONE (memory set aside, teacher-
        forced) — the fraction of the hippocampus's symbols it predicts itself (uptake)."""
        m = self.m; m.eval()
        ro, wo = m.store_read_off, getattr(m, "store_write_off", False)
        m.store_read_off, m.store_write_off = True, True
        hits_all, n_all = 0, 0
        hits_w, n_w = 0, 0                                # over traces with at least three distinct symbols
        k = 3
        try:
            with torch.no_grad():
                for ids in traces[:cap]:
                    if len(ids) < k + 2:
                        continue
                    rich = len(set(ids)) >= 3
                    h0, n0 = hits_all, n_all
                    st = m.init_state(1, self.dev); st["mouth_floor"] = False
                    x = torch.tensor([[self.sil] + ids[:k]], device=self.dev)
                    lg, st, _ = m(x, st, who=torch.zeros_like(x))
                    for j in range(k, len(ids)):
                        v = lg[0, -1].float().clone(); v[:11] = float("-inf")
                        hits_all += int(int(v.argmax()) == ids[j]); n_all += 1
                        x = torch.tensor([[ids[j]]], device=self.dev)
                        lg, st, _ = m(x, st, who=torch.zeros_like(x))
                    if rich:
                        hits_w += hits_all - h0; n_w += n_all - n0
        finally:
            m.store_read_off, m.store_write_off = ro, wo
        return {"uptake": (round(hits_all / n_all, 3) if n_all else None), "symbols": n_all,
                "uptake_rich": (round(hits_w / n_w, 3) if n_w else None), "symbols_rich": n_w}

    def _night_lessons(self, traces, scale, a, m, F, cos_log, sig):
        """the night's three lessons (NREM, REM, the value ladder), counted into _night_counts"""
        nrem = rem = 0
        batch = bool(int(getattr(a, "night_batch", 0) or 0))
        _dbg = bool(_os_env.get("IGA_NIGHT_DEBUG"))
        for _r in range(int(getattr(a, "night_rounds", 2) or 0)):
            if batch:
                self.opt.zero_grad(set_to_none=True)
            _tot = 0.0
            for ids in traces:
                # NREM: the hippocampus leads — its read is on, driving the council as by day —
                # and the lesson lands on the cortex's OWN logits, memory's vote left out, so
                # the trunk must come to carry the trace itself
                lg, y = self._student(ids, mem_on=bool(int(getattr(a, "nrem_mem", 1))))
                own = getattr(m, "_last_logits_own", None)
                own = lg if own is None else own
                if not batch:
                    self.opt.zero_grad(set_to_none=True)
                own0 = own[0].float()
                if int(getattr(a, "night_sil_mask", 0) or 0):
                    # a dream holds no rest, so rest is not a candidate in it: the silence logit is
                    # taken out of the softmax and receives no lesson (measured 2026-09-03: with it
                    # in, every night pushed rest down and the mouth filled its ticks at stress 11)
                    own0 = own0.clone(); own0[:, self.sil] = -1e4
                    keep = y != self.sil
                    loss = (F.cross_entropy(own0[keep], y[keep]) if bool(keep.any()) else own0.sum() * 0.0) * scale
                else:
                    loss = F.cross_entropy(own0, y) * scale
                self._pop_side_losses()
                if not bool(torch.isfinite(loss.detach())):
                    # a dream whose lesson is not a number teaches nothing (never a NaN step)
                    self.opt.zero_grad(set_to_none=True)
                    continue
                if batch:
                    # many replays, one consolidation: the gradients of every trace in the
                    # round are summed (mean over traces) and the weights move once
                    (loss / len(traces)).backward()
                else:
                    loss.backward(); self._night_step(); nrem += 1
                _tot += float(loss.detach()) / len(traces)
            if batch:
                self._night_step(); nrem += 1
            if _dbg:
                print(f"[night] round {_r} CE {_tot:.3f} lr {[g['lr'] for g in self.opt.param_groups]} batch {batch}", flush=True)
        rem_traces = traces[:int(getattr(a, "night_rem", 8) or 0)]
        if batch:
            self.opt.zero_grad(set_to_none=True)
        for ids in rem_traces:
            # REM: the hippocampus is silent — memory set aside, the PFC (the bands) drives —
            # and the cortex stream forecasts the band states it receives next along the trace
            self._student(ids, mem_on=False)
            loss, cosv = m.rem_pfc_loss(sigreg=sig, slot_from=1) if hasattr(m, "rem_pfc_loss") else (None, None)
            self._pop_side_losses()
            if loss is None:
                continue
            if not bool(torch.isfinite(loss.detach())):
                continue                                 # never a NaN step (night 7, 2026-09-03: REM went NaN)
            if batch:
                # many dreams, one consolidation (a single dream's step at the night's rate
                # undid NREM's lesson: measured 2026-09-03 01:40)
                (loss * scale / len(rem_traces)).backward(); rem += 1
            else:
                self.opt.zero_grad(set_to_none=True)
                (loss * scale).backward(); self._night_step(); rem += 1
            cos_log.append(cosv)
        if batch and rem:
            self._night_step()
        # the value ladder replays the day's lived pairs once more (reward at every timescale)
        vw = float(getattr(a, "value_w", 0.5) or 0.0)
        vsteps = 0
        if vw > 0 and hasattr(m, "buffered_value_loss"):
            bl = m.buffered_value_loss(self.st)
            if bl is not None and bool(torch.isfinite(bl.detach())):
                self.opt.zero_grad(set_to_none=True)
                (vw * bl).backward(); self._night_step(); vsteps = 1
        self._night_counts = (nrem, rem, vsteps)

    def _night_step(self):
        """one change of the weights in the night: the gradient is clipped to a norm of 1 and a
        step whose gradient is not a number is not taken (night 7, 2026-09-03, left the body NaN)"""
        gn = torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0)
        if bool(torch.isfinite(gn)):
            self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.n_steps += 1

    def _dream_night(self):
        """THE NIGHT (2026-09-02, the user's law). The cortex learns only from hippocampal
        traces, and nothing is kept on its behalf: a dream starts where the store itself is
        strongest, and what follows is what the store holds. NREM runs the body over each
        trace with the hippocampus leading (its read on) and teaches the trunk's own logits
        the trace; REM runs it with the hippocampus silent (memory set aside) and teaches the
        cortex stream to forecast the band states it receives at the next symbol (targets
        stop-grad, SIGReg as the collapse guard). The gauge is taken before and after."""
        import torch.nn.functional as F
        a = self.a
        m = self.m; m.eval()
        starts = self._dream_starts(int(getattr(a, "night_starts", 48) or 48))
        if not starts:
            return {"starts": 0, "note": "the store holds nothing to replay"}
        traces, empty = [], 0
        for strength, k, x in starts:
            ids, _ = self._dream_trace(x)
            if len(ids) >= 2:
                traces.append(ids)
            else:
                empty += 1
        if not traces:
            return {"starts": len(starts), "traces": 0, "empty": empty, "note": "the store's strongest keys drew no trace"}
        gauge_before = self._gauge(traces)
        scale = float(getattr(a, "night_scale", 1.0) or 1.0)
        nrem = rem = 0
        m.train()
        # sleep's plasticity is its own physiology: the optimizer runs at --night-lr while it
        # sleeps (the live rate when unset) and is restored before it wakes, whatever happens
        _lr0 = [g["lr"] for g in self.opt.param_groups]
        _nlr = getattr(a, "night_lr", None)
        _opt_day = self.opt
        if str(getattr(a, "night_opt", "shared")) == "own":
            # sleep's plasticity has its own state: a fresh optimizer for this night alone,
            # at the night's rate, freed at waking (the day's moments are untouched)
            self.opt = torch.optim.Adam(m.parameters(), lr=float(_nlr or a.live_lr))
        elif _nlr:
            for g in self.opt.param_groups:
                g["lr"] = float(_nlr)
        try:
            self._night_lessons(traces, scale, a, m, F, cos_log := [], sig := float(getattr(a, "night_sigreg", 0.1) or 0.0))
        finally:
            if self.opt is not _opt_day:
                del self.opt
            self.opt = _opt_day
            for g, lr0 in zip(self.opt.param_groups, _lr0):
                g["lr"] = lr0
        nrem, rem, vsteps = self._night_counts
        m.eval()
        with torch.no_grad():
            _finite = all(bool(torch.isfinite(p).all()) for p in m.parameters())
        if not _finite:
            # the night broke the weights: it is discarded, the body wakes as it slept (the file on
            # disk is the last save, before this night) — plumbing against a numerical failure
            sd = torch.load(a.save, map_location="cpu", weights_only=False)
            m.load_state_dict(sd["model"]); del sd
            m.eval()
            print(f"[diary] night {self.nights + 1}: weights not finite after the lessons; reloaded {a.save}", flush=True)
            self._night_discarded = True
        else:
            self._night_discarded = False
        gauge_after = self._gauge(traces)
        dec = lambda ids: self.tok.decode(list(ids))
        return {"starts": len(starts), "traces": len(traces), "empty": empty,
                "mean_len": round(sum(len(i_) for i_ in traces) / len(traces), 1),
                "examples": [dec(i_)[:32] for i_ in traces[:8]],
                "nrem_steps": nrem, "rem_steps": rem, "value_steps": vsteps,
                "rem_cos": (round(sum(cos_log) / len(cos_log), 3) if cos_log else None),
                "discarded": bool(getattr(self, "_night_discarded", False)),
                "gauge": {"before": gauge_before, "after": gauge_after}}

    def _sleep_now(self):
        """the body's own night: called from the tick when its sleep pressure crosses"""
        try:
            self.queue.clear()                       # a sleeping child hears nothing typed at it
            self.night()
        except Exception as e:
            self.last = {"error": "night: " + str(e)[:160], "tick": self.ticks}
            # a night that failed (memory, a fault) must not be retried on every tick: the body
            # stays awake with half a day's pressure and tries again later
            self.sleep_pressure = int(self.wake_ticks * 0.5)
            print(f"[diary] night failed at tick {self.ticks}: {str(e)[:200]}", flush=True)

    def night(self):
        self.awake_ticks = False
        self.asleep = True
        try:
            with self.lock:
                if not self.session:
                    self.session = [{"diary": True}]     # the night's bookkeeping wants a day
                dream = self._dream_night() if len(self.day_buf) >= 65 else None
                _p0 = self.sleep_pressure
                self.sleep_pressure = 0                  # before the night's autosave: a restored body wakes rested
                self.nights += 1                         # counted before the autosave, so a restored body knows its nights
                res = self.sleep()
                if not (isinstance(res, dict) and not res.get("error")):
                    self.sleep_pressure = _p0
                    self.nights -= 1
                if isinstance(res, dict) and dream is not None:
                    res["dream"] = dream
                    # the top-level counters name the night that actually happened
                    res["nrem"] = int(dream.get("nrem_steps", 0) or 0)
                    res["rem"] = int(dream.get("rem_steps", 0) or 0)
                if isinstance(res, dict) and not res.get("error"):
                    self.last_night = {"tick": self.ticks, "night": self.nights,
                                       "dream": res.get("dream"), "lived_tokens": res.get("lived_tokens"),
                                       "store_carried": res.get("store_carried")}
                if hasattr(self.m, "reset_bag"):
                    self.m.reset_bag(self.st)
                self.level = 0
                self.credit.clear()
        finally:
            self.asleep = False
            self.awake_ticks = True
        return res


PAGE = """<!doctype html><meta charset=utf-8><title>the diary</title>
<style>
body{margin:0;background:#f5f1e6;color:#222;font:16px/1.6 Georgia,serif}
#pg{white-space:pre-wrap;padding:32px 40px 120px;min-height:70vh;max-width:820px;margin:0 auto}
.u{color:#1a1a1a}.m{color:#7a2e0e}.n{color:#c9b8a5}
#bar{position:fixed;left:0;right:0;bottom:0;background:#eae4d3;border-top:1px solid #cbbfa3;padding:10px 40px;font:13px ui-monospace,monospace;display:flex;gap:18px;flex-wrap:wrap;align-items:center}
#bar b{font-weight:600}
button{font:12px ui-monospace,monospace;padding:4px 10px;background:#f5f1e6;border:1px solid #cbbfa3;cursor:pointer}
#hint{color:#6b6250;font:13px Georgia,serif;padding:12px 40px 0;max-width:820px;margin:0 auto}
</style>
<div id=hint>Type anywhere: your letters go in as you type, one per tick. Its letters from memory appear in brown, its noise in pale sand. Arrow up / down: your face (&minus;6..6): frown at noise, smile at quiet and at echoes. Enter: a new line. Nothing is edited; this is a diary.</div>
<div id=pg></div>
<div id=bar>
 <span>you <b id=you>0.0</b></span><span>its face <b id=face>–</b></span><span>mood <b id=mood>–</b></span>
 <span>stress <b id=cort>–</b></span><span>uncertainty <b id=ent>–</b></span><span>memory <b id=mem>–</b></span>
 <span>tick <b id=tick>0</b></span><span>queued <b id=q>0</b></span>
 <button onclick="post('/sleep',{}).then(r=>{document.getElementById('night').textContent=JSON.stringify(r).slice(0,160)})">sleep</button>
 <button onclick="post('/save',{})">save</button><span id=night></span>
</div>
<script>
let face=0,seen=0;
const pg=document.getElementById('pg');
function post(p,b){return fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}).then(r=>r.json())}
document.addEventListener('keydown',e=>{
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  if(e.key==='ArrowUp'){face=Math.min(6,face+0.5);post('/face',{expr:face});e.preventDefault();return}
  if(e.key==='ArrowDown'){face=Math.max(-6,face-0.5);post('/face',{expr:face});e.preventDefault();return}
  if(e.key==='Enter'){post('/type',{text:'\\n'});e.preventDefault();return}
  if(e.key.length===1){post('/type',{text:e.key});e.preventDefault()}
});
function render(items){
  for(const [t,who,_y,_f,backed] of items){
    if(!t)continue;
    const s=document.createElement('span');s.className=who?(backed?'m':'n'):'u';s.textContent=t;pg.appendChild(s);
  }
  window.scrollTo(0,document.body.scrollHeight);
}
async function poll(){
  try{
    const r=await fetch('/state?since='+seen).then(r=>r.json());
    render(r.page);seen=r.n;
    const l=r.last||{};const f=(x,d)=>x==null?'–':Number(x).toFixed(d);
    document.getElementById('you').textContent=f(face,1);
    document.getElementById('face').textContent=f(l.face,2);
    document.getElementById('mood').textContent=f(l.mood,2);
    document.getElementById('cort').textContent=f(l.cort,2);
    document.getElementById('ent').textContent=f(l.ent,2);
    document.getElementById('mem').textContent=l.mem?l.mem.slice(0,2).map(x=>JSON.stringify(x[0])+' '+x[1]).join('  '):'–';
    document.getElementById('tick').textContent=l.tick||0;
    document.getElementById('q').textContent=r.queued;
  }catch(e){}
  setTimeout(poll,250);
}
poll();
</script>"""

ORG = None
LOCK = threading.Lock()


class DH(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/state":
            since = int((parse_qs(u.query).get("since") or ["0"])[0])
            self._json(ORG.state(since))
            return
        if u.path == "/pulse":
            self._json({"awake": ORG.awake_ticks, "tick": ORG.ticks, "mood": round(ORG.mood, 2)})
            return
        b = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        try:
            if self.path == "/type":
                self._json(ORG.type_text(str(body.get("text", ""))))
            elif self.path == "/face":
                self._json(ORG.set_face(body.get("expr", 0)))
            elif self.path == "/sleep":
                self._json(ORG.night())
            elif self.path == "/save":
                with ORG.lock:
                    self._json(ORG.save())
            elif self.path == "/reset":
                with ORG.lock:
                    ORG.reset(); ORG.page = []; ORG.page_base = 0; ORG.queue.clear(); ORG.credit.clear(); ORG.level = 0
                    ORG.stream.clear()
                    if hasattr(ORG.m, "reset_bag"):
                        ORG.m.reset_bag(ORG.st)
                    self._json({"reset": True})
            else:
                self._json({"error": "unknown path"}, 404)
        except Exception as e:
            self._json({"error": str(e)[:300]}, 500)


def main():
    global ORG
    a = O.build_parser().parse_args()
    print("[diary] two writers, one page — no assists exist in this build", file=sys.stderr)
    ORG = Diary(a)
    print(f"[diary] the page is open on http://localhost:{a.port} ({a.ckpt} on {ORG.dev}, tick {ORG.period}s)",
          file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", a.port), DH).serve_forever()


if __name__ == "__main__":
    main()
