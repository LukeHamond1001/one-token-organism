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
    symbol_cost=0.12, fatigue_half_life_s=120.0, stress_half_life_s=120.0, mood_half_life_s=600.0,
    wake_ticks=12000, elig_ticks=12, elig_decay=0.8, store_fade=0.9, store_floor_rel=0.1, store_temp=0.05,
    bag_decay=0.7, night_lr=1e-4, night_rounds=24, night_starts=48, rem_steps=8, rem_dreams=8, sigreg=0.1,
    dream_max=24, dream_floor_rel=0.5, dream_adapt=0.5, dream_recover=0.7, gate_baseline=0.98, wake_every=24, wake_window=32, live_lr=1e-5, value_lr=1e-3, face_lr=1e-3,
    gate_lr=0.05, birth_act=0.25, gate_habit=0.9, gate_fatigue=10.0, gate_int=0.5, gate_every=24,
    read_sharp=10.0, burst=0.5, mood_gain=0.25, stress_gain=0.5, v_buf=32,
)


class Life:
    def __init__(self, organs, tok, cfg=None, device="cpu", seed=0, save_path=None):
        self.m = organs.to(device); self.m.eval()
        self.tok = tok; self.dev = device
        self.cfg = dict(PHYSIOLOGY); self.cfg.update(cfg or {})
        self.m.read_sharp = float(self.cfg["read_sharp"])
        self.sil = tok.token_to_id("<pad>")
        self.nl = tok.token_to_id("\n")
        self.bans = [i for i in range(11) if i != self.sil] + ([self.nl] if self.nl is not None else [])
        self.store = Store(self.m.d, temp=float(self.cfg["store_temp"]), device=device)
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.save_path = save_path
        nb, d, W = len(self.m.clocks), self.m.d, self.m.window
        # working state
        self.bands = torch.zeros(nb, d, device=device)
        self.bag = torch.zeros(d, device=device)
        self.win = collections.deque(maxlen=W)            # per step: dict(x, who, face, bundle, read, r)
        self.pred_prev = None                              # the forecast made at the last step (surprise)
        self.v_prev = None                                 # V_b of the previous tick's states
        self.v_buf = {b: collections.deque(maxlen=int(self.cfg["v_buf"])) for b in range(nb)}
        # feelings and clocks
        self.fatigue = 0.0; self.stress = 0.0; self.mood = 0.0
        self._t_feel = time.time()
        self.sleep_pressure = 0; self.ticks = 0; self.nights = 0; self.day_n = 0
        self.asleep = False; self.last_night = None
        # the face
        self.face_now = 0.0; self.level = 0; self.face_prev = 0.0
        # the mouth's gate
        self.gate_buf = collections.deque(maxlen=96)
        self.sym_freq = {}
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
        self.opt_value = torch.optim.Adam(list(self.m.value.parameters()) + list(self.m.band_gate.parameters()),
                                          lr=float(self.cfg["value_lr"]))
        self.opt_face = torch.optim.Adam(self.m.face_head.parameters(), lr=float(self.cfg["face_lr"]))

    # ---------------- feelings ----------------
    def _decay_feelings(self):
        now = time.time(); dt = max(0.0, now - self._t_feel); self._t_feel = now
        self.fatigue *= 0.5 ** (dt / float(self.cfg["fatigue_half_life_s"]))
        self.stress *= 0.5 ** (dt / float(self.cfg["stress_half_life_s"]))
        self.mood *= 0.5 ** (dt / float(self.cfg["mood_half_life_s"]))

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
            # the hippocampus: write what came next under the context before it
            if learn_store and who == 0 and x != self.sil and self.bag.norm() > 1e-6:
                self.store.write(self.bag, ex, surp * (1.0 + abs(dopamine)), who)   # the world's quiet is not a memory
            # the context moves on
            self.bag = float(self.cfg["bag_decay"]) * self.bag + ex + m.who_emb.weight[who]
            read, conf, _ = self.store.read(self.bag)
            face = torch.tensor([self.face_now / 6.0, (self.face_now - self.face_prev) / 6.0], device=self.dev)
            self.win.append({"x": int(x), "who": int(who), "face": face, "bundle": self.bands.clone(),
                             "read": read.clone(), "r": float(r)})
            C = self._stream_now()
            self.bands = m.band_update(self.bands, C)
            pred = m.latent_pred(C)
            self.pred_prev = F.normalize(pred, dim=0)
        return C, pred, surp, conf

    def _window_tensors(self, win=None):
        win = list(self.win if win is None else win)
        xs = torch.tensor([w["x"] for w in win], device=self.dev)
        whos = torch.tensor([w["who"] for w in win], device=self.dev)
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
        # the face: a change is felt; a held face is silence; easing off is not an event
        lvl = max(-6, min(6, int(self.face_now)))
        felt = 0
        if lvl != self.level:
            if abs(lvl) > abs(self.level) or lvl * self.level < 0:
                felt = lvl
            self.level = lvl
        r = float(max(-2, min(2, felt)))                    # the world's reward: the felt face, clipped like a press
        # --- the ear's half: the world's symbol (or its quiet) enters ---
        v_before = m.values(self.bands.detach()) if self.v_prev is None else self.v_prev
        C1, pred1, surp1, conf1 = self._step(u, 0, r=r, dopamine=getattr(self, "_dopa", 0.0))
        # --- dopamine: the fast band's error of the world's reward; the critic learns at every band ---
        with torch.no_grad():
            v_now = m.values(self.bands)
        gam = m.gammas()
        with torch.enable_grad():
            m.train()
            v_prev_live = m.values(self._bands_prev.detach()) if getattr(self, "_bands_prev", None) is not None else None
            if v_prev_live is not None:
                td = torch.stack([r + gam[b] * v_now[b].detach() - v_prev_live[b] for b in range(len(gam))])
                loss_v = (td ** 2).mean()
                # Go/NoGo on the bands' own updates: a positive error pulls the gate open, a negative one shut
                gates = torch.stack([torch.sigmoid(m.band_gate[b](self._bands_prev[b].detach())).squeeze()
                                     for b in range(len(gam))])
                tgt = (td.detach() > 0).float()
                loss_g = (td.detach().abs() * F.binary_cross_entropy(gates.clamp(1e-6, 1 - 1e-6), tgt, reduction="none")).mean()
                self.opt_value.zero_grad(set_to_none=True)
                (loss_v + 0.01 * loss_g).backward()
                self.opt_value.step()
                delta = float(td[0].detach())              # dopamine: the fast band's signed error
                for b in range(len(gam)):
                    self.v_buf[b].append((self._bands_prev[b].detach().cpu(), r, self.bands[b].detach().cpu()))
            else:
                delta = r
            m.eval()
        self._dopa = delta
        # --- its face learns from yours (a readout) ---
        with torch.enable_grad():
            f_pred = m.face_head(C1.detach()).squeeze() * 6.0
            lf = (f_pred - torch.tensor(float(self.face_now), device=self.dev)) ** 2
            self.opt_face.zero_grad(set_to_none=True); lf.backward(); self.opt_face.step()
        its_face = float(f_pred.detach())
        # --- the mouth's half: whether (the gate), then what (the lexicon) ---
        with torch.no_grad():
            feat = torch.cat([C1.detach(), torch.tensor([self.fatigue / 10.0, self.mood / 6.0, self.stress / 10.0], device=self.dev)])
            z = m.mouth_gate(feat.unsqueeze(0))[0, 0] / (1.0 + self.stress / 10.0)   # stress flattens the choice
            p_act = float(torch.sigmoid(z))
            acted = bool(torch.rand(1, generator=self.gen).item() < p_act)
            logits = m.readout(pred1).clone(); logits[self.bans] = float("-inf"); logits[self.sil] = float("-inf")
            probs = torch.softmax(logits, -1)
            ent = float(-(probs * (probs + 1e-9).log()).sum() / math.log(probs.numel()))
            if acted:
                nxt = int(torch.multinomial(probs.cpu(), 1, generator=self.gen))
                p_choice = float(probs[nxt])
            else:
                nxt, p_choice = self.sil, 0.0
        int_t = 0.0
        if acted:
            hab = float(self.cfg["gate_habit"])
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
        # --- feelings from dopamine ---
        self.mood = max(-6.0, min(6.0, self.mood + float(self.cfg["mood_gain"]) * delta))
        self.stress = min(30.0, self.stress + float(self.cfg["stress_gain"]) * max(0.0, -delta))
        if abs(delta) >= float(self.cfg["burst"]):
            self.n_bursts += 1
        # --- the gate's buffer and lesson ---
        self.gate_buf.append([feat.cpu(), acted, delta, int_t, self.fatigue])
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
                     "gate": round(p_act, 3), "dopamine": round(delta, 3), "doses": self.n_bursts,
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
        feats = torch.stack([b[0] for b in buf[:n]]).to(self.dev)
        G = torch.zeros(n)
        for t in range(n):
            g = sum((dec ** k) * float(buf[t + k][2]) for k in range(K))     # the dopamine that followed
            if buf[t][1]:
                g += w_int * float(buf[t][3]) - cost * (1.0 + float(buf[t][4]) / f0)
            G[t] = g
        # the striatum learns from the error against what it expected: a constant cost, or a constant
        # drive, teaches nothing; a burst, a dip, rising fatigue do. The baseline is a running mean.
        base = getattr(self, "_g_base", None)
        if base is None:
            base = float(G.mean())
        A = G - base
        self._g_base = float(self.cfg["gate_baseline"]) * base + (1.0 - float(self.cfg["gate_baseline"])) * float(G.mean())
        if float(A.abs().max()) < 1e-4:
            return
        self.m.mouth_gate.train()
        z = self.m.mouth_gate(feats).squeeze(-1)
        loss = -(A.to(self.dev) * z).mean()                 # bursts push Go, dips push NoGo, whatever it did
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
        w = torch.tensor([1.0 if (whos[t + 1] == 0 and int(xs[t + 1]) != self.sil) else 0.0 for t in range(T - 1)], device=self.dev)
        if float(w.sum()) < 1:
            return None
        m = self.m; m.train()
        try:
            self.opt_day.zero_grad(set_to_none=True)
            u = m.inputs(xs, whos, faces, bundles, reads)
            C = m.stream(u)
            pred = m.latent_pred(C[:-1])
            ll, lc = m.latent_loss(pred, xs[1:], w=w, sig=float(self.cfg["sigreg"]))
            fl, fc = m.forecast_loss(C[:-1], bundles[1:], sig=0.0)
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
        ref = self.store.self_confidence()
        floor = float(self.cfg["dream_floor_rel"]) * ref
        out = []
        a_hit, a_rec = float(self.cfg["dream_adapt"]), float(self.cfg["dream_recover"])
        with torch.no_grad():
            for j in starts:
                bag = self.store.K[j].clone(); ids = []
                adapt = torch.ones(self.store.n(), device=self.dev)      # neural adaptation: a recalled memory tires
                for _ in range(int(self.cfg["dream_max"])):
                    pred, conf, win = self.store.read(bag, adapt=adapt)
                    if conf < floor:
                        break
                    lg = self.m.readout(pred).clone(); lg[self.bans] = float("-inf"); lg[self.sil] = float("-inf")
                    nid = int(lg.argmax())
                    ids.append(nid)
                    adapt = 1.0 - a_rec * (1.0 - adapt)                   # recovery toward 1
                    if win >= 0:
                        adapt[win] *= a_hit                               # the winner tires
                    bag = float(self.cfg["bag_decay"]) * bag + self.m.E.weight[nid] + self.m.who_emb.weight[0]
                if len(ids) >= 2:
                    out.append(ids)
        return out

    def _dream_inputs(self, ids, mem_on):
        """a dream as a window: fresh bands (a night's working state), the store leading if mem_on"""
        m = self.m
        bands = torch.zeros_like(self.bands); bag = torch.zeros_like(self.bag)
        xs = [self.sil] + list(ids[:-1]); reads, bundles = [], []
        with torch.no_grad():
            for x in xs:
                bag = float(self.cfg["bag_decay"]) * bag + m.E.weight[x] + m.who_emb.weight[0]
                rd = self.store.read(bag)[0] if mem_on else torch.zeros(m.d, device=self.dev)
                reads.append(rd); bundles.append(bands.clone())
                u = m.inputs(torch.tensor(xs[:len(reads)], device=self.dev), torch.zeros(len(reads), dtype=torch.long, device=self.dev),
                             torch.zeros(len(reads), 2, device=self.dev), torch.stack(bundles), torch.stack(reads))
                C = m.stream(u)[-1]
                bands = m.band_update(bands, C)
        T = len(xs)
        return (torch.tensor(xs, device=self.dev), torch.zeros(T, dtype=torch.long, device=self.dev),
                torch.zeros(T, 2, device=self.dev), torch.stack(bundles), torch.stack(reads), torch.tensor(ids, device=self.dev))

    def gauge(self, dreams):
        """the cortex alone (store off), teacher-forced on the dreams: the share of next symbols it forecasts itself"""
        hits = n = 0
        with torch.no_grad():
            for ids in dreams:
                xs, whos, faces, bundles, reads, y = self._dream_inputs(ids, mem_on=False)
                C = self.m.stream(self.m.inputs(xs, whos, faces, bundles, reads))
                lg = self.m.readout(self.m.latent_pred(C)); lg[:, self.bans] = float("-inf"); lg[:, self.sil] = float("-inf")
                hits += int((lg.argmax(-1) == y).sum()); n += int(y.numel())
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
                before, nsym = self.gauge(dreams)
                opt = torch.optim.Adam(m.parameters(), lr=float(self.cfg["night_lr"]))   # sleep's own plasticity
                sig = float(self.cfg["sigreg"])
                m.train()
                nrem = 0; losses = []
                for _ in range(int(self.cfg["night_rounds"])):
                    opt.zero_grad(set_to_none=True); tot = 0.0; ok = 0
                    for ids in dreams:
                        xs, whos, faces, bundles, reads, y = self._dream_inputs(ids, mem_on=True)
                        C = m.stream(m.inputs(xs, whos, faces, bundles, reads))
                        ll, _ = m.latent_loss(m.latent_pred(C), y, sig=sig)
                        if not bool(torch.isfinite(ll.detach())):
                            continue
                        (ll / len(dreams)).backward(); tot += float(ll.detach()) / len(dreams); ok += 1
                    if ok:
                        self._night_step(opt); nrem += 1; losses.append(round(tot, 3))
                # REM: the cortex runs free from each dream's first symbols on its own readout
                rem_cos = []
                opt.zero_grad(set_to_none=True)
                for ids in dreams[:int(self.cfg["rem_dreams"])]:
                    fl, fc = self._rem_rollout(ids, sig)
                    if fl is None or not bool(torch.isfinite(fl.detach())):
                        continue
                    (fl / max(1, min(len(dreams), int(self.cfg["rem_dreams"])))).backward(); rem_cos.append(fc)
                if rem_cos:
                    self._night_step(opt)
                # the value ladder replays its lived pairs once
                self._value_replay()
                m.eval()
                del opt
                finite = all(bool(torch.isfinite(p).all()) for p in m.parameters())
                rep["discarded"] = not finite
                if not finite and self.save_path and os.path.exists(self.save_path):
                    sd = torch.load(self.save_path, map_location="cpu", weights_only=False)
                    m.load_state_dict(sd["organs"]); m.to(self.dev)
                after, _ = self.gauge(dreams)
                rep.update({"nrem_steps": nrem, "nrem_loss": losses[:3] + (["..."] if len(losses) > 6 else []) + losses[-3:],
                            "rem_steps": len(rem_cos), "rem_cos": (round(sum(rem_cos) / len(rem_cos), 3) if rem_cos else None),
                            "gauge": {"before": before, "after": after, "symbols": nsym}})
            # the rest: the store fades, the working state wakes fresh, the body is saved
            rep["store_dropped"] = self.store.fade(float(self.cfg["store_fade"]), float(self.cfg["store_floor_rel"]))
            rep["store_slots"] = self.store.n()
            self.bands.zero_(); self.bag.zero_(); self.win.clear(); self.pred_prev = None
            self._bands_prev = None; self.v_prev = None; self.stream.clear(); self.gate_buf.clear()
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
        bands = torch.zeros_like(self.bands); bag = torch.zeros_like(self.bag)
        xs = [self.sil] + list(ids[:k]); whos = [0] * len(xs)
        reads, bundles, Cs, bnext = [], [], [], []
        for step in range(len(xs) + L):
            if step >= len(xs):
                # its own next symbol, read off its forecast (no gradient through the choice)
                with torch.no_grad():
                    lg = m.readout(m.latent_pred(Cs[-1])).clone(); lg[self.bans] = float("-inf"); lg[self.sil] = float("-inf")
                    xs.append(int(lg.argmax())); whos.append(1)
            x = xs[-1] if step >= len(reads) else xs[step]
            with torch.no_grad():
                bag = float(self.cfg["bag_decay"]) * bag + m.E.weight[xs[step]] + m.who_emb.weight[whos[step]]
            reads.append(torch.zeros(m.d, device=self.dev)); bundles.append(bands.clone())
            u = m.inputs(torch.tensor(xs[:step + 1], device=self.dev), torch.tensor(whos[:step + 1], device=self.dev),
                         torch.zeros(step + 1, 2, device=self.dev), torch.stack(bundles), torch.stack(reads))
            C = m.stream(u)[-1]
            Cs.append(C)
            with torch.no_grad():
                bands = m.band_update(bands, C.detach())
            bnext.append(bands.clone())
        # the stream at t forecasts the bundle handed over at t+1, along the free-running part
        C_free = torch.stack(Cs[len(ids[:k]):-1]); B_next = torch.stack(bnext[len(ids[:k]):-1])
        if C_free.shape[0] < 2:
            return None, None
        return m.forecast_loss(C_free, B_next, sig=sig)

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
                vn = m.value[b](hn).squeeze(-1)
            vp = m.value[b](hp).squeeze(-1)
            terms.append(((R + gam[b] * vn - vp) ** 2).mean())
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
                "life": {"ticks": self.ticks, "nights": self.nights, "day_n": self.day_n, "sleep_pressure": self.sleep_pressure,
                         "fatigue": self.fatigue, "stress": self.stress, "mood": self.mood, "n_bursts": self.n_bursts,
                         "sym_freq": self.sym_freq, "last_night": self.last_night}}
        torch.save(blob, path + ".tmp"); os.replace(path + ".tmp", path)
        return {"saved": path}

    @classmethod
    def load(cls, path, tok, device="cpu", cfg=None, seed=0, save_path=None):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        a = blob["arch"]
        organs = Organs(a["vocab"], d=a["d"], layers=a["layers"], heads=a["heads"], window=a["window"], clocks=tuple(a["clocks"]))
        organs.load_state_dict(blob["organs"])
        c = dict(blob.get("cfg") or {}); c.update(cfg or {})
        life = cls(organs, tok, cfg=c, device=device, seed=seed, save_path=save_path or path)
        life.store.load_state_dict(blob["store"])
        L = blob.get("life") or {}
        for k in ("ticks", "nights", "day_n", "sleep_pressure", "fatigue", "stress", "mood", "n_bursts", "last_night"):
            if k in L:
                setattr(life, k, L[k])
        life.sym_freq = dict(L.get("sym_freq") or {})
        return life

    @classmethod
    def birth(cls, tok, device="cpu", d=256, layers=6, heads=4, window=64, cfg=None, seed=0, save_path=None):
        torch.manual_seed(int(seed))
        organs = Organs(tok.get_vocab_size(), d=d, layers=layers, heads=heads, window=window, birth_act=float((cfg or {}).get("birth_act", PHYSIOLOGY["birth_act"])))
        return cls(organs, tok, cfg=cfg, device=device, seed=seed, save_path=save_path)
