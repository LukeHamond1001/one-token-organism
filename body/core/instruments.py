"""the instruments (a mixin of `Life`, body/life.py): the tick's last phase (`_bookkeep`: the stream, the page, the tick's record
for the instruments, the sleep switch), the cortex alone gauged on the dreams (`gauge`, `_gauge_batched`), the page's own endpoint
(`state`: what a parent may see), and the supervisor's readers (`anticipation`, `insides`).

Moved verbatim from body/life.py (review 2026-09-22 section 4, step 2)."""
import torch
import torch.nn.functional as F


class InstrumentsMixin:
    def _bookkeep(self, u, who, nxt, its_face, felt, ent, p_act, delta, level, r, vlong, delta_long, conf1, surp1, probs):
        """the stream, the page, the tick's record for the instruments, the sleep switch"""
        m = self.m
        # --- bookkeeping ---
        self.stream.append((int(u), 0)); self.stream.append((int(nxt), 1))
        self.page.append(((self.anatomy.decode([int(u)]) if u != self.sil else ""), 0, round(self.face_now, 2), round(its_face, 2), who))
        self.page.append(((self.anatomy.decode([int(nxt)]) if nxt != self.sil else ""), 1, round(self.face_now, 2), round(its_face, 2), False))
        if len(self.page) > 40000:
            del self.page[:20000]; self.page_base += 20000
        self.face_prev = self.face_now
        self.ticks += 1; self.sleep_pressure += 1
        self.last = {"tick": self.ticks, "you": round(self.face_now, 2), "face": round(its_face, 2), "vrel": round(self._vrel_corr, 3),
                     "mood": round(self.mood, 2), "cort": round(self.fatigue, 2), "fatigue": round(self.fatigue, 2),
                     "stress": round(self.stress, 2), "ent": round(ent, 2), "felt": felt,
                     "said": (self.anatomy.decode([int(nxt)]) if nxt != self.sil else ""),
                     "gate": round(p_act, 3), "dopamine": round(delta, 3), "doses": self.n_bursts, "level": round(level, 3), "r": round(float(r), 3),
                     "vlong": round(vlong, 3), "dlong": round(delta_long, 3),
                     "store": self.store.n(), "store_conf": round(conf1, 3), "surprise": round(surp1, 3),
                     "own": [self.anatomy.decode([int(probs.argmax())]), round(float(probs.max()), 3)],
                     "gate_lesson": self._gate_last, "wake": self._wake_last}
        if len(self.anatomy.effectors) > 1:                        # the later effectors' acts this tick and their gates (step R5)
            self.last["acts"] = {e_.name: {"act": st_["now"]["act"], "acted": st_["now"]["acted"], "gate": round(st_["now"]["p_act"], 3)}
                                 for e_, st_ in zip(self.anatomy.effectors[1:], self.motor)}
        if self.sleep_pressure >= int(self.cfg["wake_ticks"]) and len(self.win) >= 8:
            self._sleep_now()

    def _gauge_batched(self, dreams, owns=None, bs=32):
        """gauge() over lockstep batches: the same count, many dreams at once; owns (dream_who): its own symbols are not counted"""
        hits = 0.0; n = 0.0; cos_sum = 0.0
        bans = [b for b in self.bans if b != self.eot]
        with torch.no_grad():
            for i in range(0, len(dreams), bs):
                obs, xos, bundles, reads, y, w = self._dream_batch(dreams[i:i + bs], owns[i:i + bs] if owns is not None else None)
                pred = self.m.latent_pred(self.m.stream(self.m.inputs(self.anatomy, obs, xos, bundles)))
                lg = self.m.readout(pred); lg[..., bans] = float("-inf")
                if self.end_id != self.sil:
                    lg[..., self.sil] = float("-inf")
                hits += float(((lg.argmax(-1) == y).float() * w).sum()); n += float(w.sum())
                cos_sum += float((F.cosine_similarity(pred, self.m.E.weight[y], dim=-1) * w).sum())
        self._gauge_cos = (round(cos_sum / n, 3) if n else None)
        return (round(hits / n, 3) if n else None), int(n)

    def gauge(self, dreams, owns=None):
        """the cortex alone (store off), teacher-forced on the dreams: the share of next symbols it
        forecasts itself (argmax), and the mean cosine of its forecast to the embedding received
        (the finer instrument: it moves before the argmax does); owns (who said each symbol) needs the lockstep path"""
        if int(self.cfg.get("night_batch", 0)) > 0 and dreams:
            return self._gauge_batched(dreams, owns)
        hits = n = 0; cos_sum = 0.0
        with torch.no_grad():
            for ids in dreams:
                obs, whos, bundles, reads, y = self._dream_inputs(ids, mem_on=False)
                C = self.m.stream(self.m.inputs(self.anatomy, obs, whos, bundles))
                pred = self.m.latent_pred(C)
                lg = self.m.readout(pred); lg[:, [b for b in self.bans if b != self.eot]] = float("-inf")
                if self.end_id != self.sil:
                    lg[:, self.sil] = float("-inf")                       # under the rest form the rest is a target the cortex may hit
                hits += int((lg.argmax(-1) == y).sum()); n += int(y.numel())   # a dream's end (the turn's) counts as a target
                cos_sum += float(F.cosine_similarity(pred, self.m.E.weight[y], dim=-1).sum())
        self._gauge_cos = (round(cos_sum / n, 3) if n else None)
        return (round(hits / n, 3) if n else None), n

    def state(self, since=0):
        """THE PAGE, and nothing else (2026-09-08, the review): what a parent may see. The words, the faces, whether it sleeps,
        how many nights it has lived. No reading from inside reaches the one who decides the face."""
        i = max(0, int(since) - self.page_base)
        return {"page": self.page[i:], "n": self.page_base + len(self.page), "base": self.page_base, "queued": len(self.queue),
                "asleep": self.asleep, "nights": self.nights}

    def anticipation(self):
        """the digest's number read from inside: the fast critic's rise over the seven ticks before a felt smile, as a percent of a smile"""
        vf = list(self._ring_vf); rr = list(self._ring_r); n = min(len(vf), len(rr)); vf, rr = vf[-n:], rr[-n:]
        idx = [i for i in range(8, n) if rr[i] > 0]
        if n < 100 or not idx:
            return None
        mv = sum(vf) / n
        pre = sum(vf[i] - mv for i in idx) / len(idx); far = sum(vf[i - 7] - mv for i in idx) / len(idx)
        return {"rise_pct": round((pre - far) / 2 * 100, 1), "n": len(idx), "mean_vf": round(mv, 3), "ticks": n}

    def insides(self):
        """the supervisor's instrument, never the caregiver's: the readings from inside"""
        d = {"last": self.last, "sleep_pressure": self.sleep_pressure, "wake_ticks": int(self.cfg["wake_ticks"]),
                "nights": self.nights, "last_night": self.last_night, "store": self.store.n(),
                "mood": round(float(self.mood), 3), "fatigue": round(float(self.fatigue), 3), "stress": round(float(self.stress), 3),
                "ticks": self.ticks, "vrel_slope": round(float(self._vrel_gain), 3), "vrel_corr": round(float(self._vrel_corr), 3),
                "vw_now": round(float(getattr(self, "_vw_now", 0.0)), 4), "floor_now": round(float(getattr(self, "_floor_now", 0.0)), 4),
                "sharp_now": round(float(self.m.read_sharp), 2), "sharp_eff": round(float(getattr(self, "_sharp_eff", self.m.read_sharp)), 2), "sharp_cal": round(float(self.sharp_cal), 2), "sharp_form": str(self.cfg.get("sharp_form", "fixed")),
                "anticipation": self.anticipation(), "face_form": str(self.cfg.get("face_form", "read")), "face_slope": round(float(self._frel_gain), 3),
                "face_corr": round(float(self._frel_corr), 3), "face_pred": round(float(self._fpred_now), 3), "rem_form": str(self.cfg.get("rem_form", "forecast")),
                "actor_voice": str(self.cfg.get("actor_voice", "off")), "actor_slope": round(float(self._arel_gain), 3), "actor_corr": round(float(self._arel_corr), 3),
                "actor_form": str(self.cfg.get("actor_form", "add")), "chunk_words": int(getattr(self, "_chunk_words", 0)), "chunk_ticks": int(getattr(self, "_chunk_ticks", 0)),
                "own_stored": int(getattr(self, "_own_stored_n", 0)),
                "actor_agree": (round(sum(self._act_agree) / len(self._act_agree), 3) if self._act_agree else None), "acts": len(self._act_agree),
                "face_input": str(self.cfg.get("face_input", "cortex")), "torn_frac": (round(sum(self._ring_torn) / len(self._ring_torn), 3) if self._ring_torn else None),
                "ent_mean": (round(sum(self._ring_ent) / len(self._ring_ent), 3) if self._ring_ent else None)}
        if int(self.cfg.get("pace_sense", 0)):
            d["pace"] = self._pace_report()                         # the sensed pace's instruments, the day so far
        if len(self.anatomy.effectors) > 1:                        # the later effectors (step R5): the act last tick, the gate's last lesson
            d["effectors"] = {e_.name: {"acted_last": st_["acted_last"], "gate_lesson": st_["last"]} for e_, st_ in zip(self.anatomy.effectors[1:], self.motor)}
        return d
