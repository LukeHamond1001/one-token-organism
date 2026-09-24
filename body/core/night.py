"""the night (a mixin of `Life`, body/life.py): the corpus as the body hears it, the dreams (`dreams`), the dreams as windows and
lockstep batches, `night()` itself (NREM, REM, the value replay, the gauge before and after, the store's fade, the report, the save),
the night on another device, REM's imagination and rollout, the value replay, and the sleep switch's call (`_sleep_now`: since the
core refactor's step R9 the world is paused before the night and resumed after it).

Moved verbatim from body/life.py (review 2026-09-22 section 4, step 2)."""
import os

import torch


class NightMixin:
    # ---------------- the night ----------------
    def _corpus_pool(self):
        """the corpus as the body hears it (item 44): the file named by dream_corpus_file, lower case, commas, quotes, colons and
        semicolons dropped, split at . ? and !, sentences of 8 symbols to the window less the end symbol, of the tokenizer's letters
        only; read once and kept as lists of ids"""
        if getattr(self, "_corpus_cache", None) is not None:
            return self._corpus_cache
        path = str(self.cfg.get("dream_corpus_file", "") or "")
        pool = []
        if path and os.path.exists(path):
            import re
            W_ = int(self.m.window) - (1 if int(self.cfg.get("offset_ticks", 0)) > 0 else 0)
            ok = set("abcdefghijklmnopqrstuvwxyz .?!'")
            raw = open(path, encoding="utf-8", errors="ignore").read().lower()
            raw = raw.replace("<|endoftext|>", " ").replace('"', "").replace(",", "").replace(";", "").replace(":", "").replace("\n", " ")
            for s_ in re.split(r"(?<=[.?!])\s+", raw):
                s_ = re.sub(r"\s+", " ", s_).strip()
                if 8 <= len(s_) <= W_ and all(ch in ok for ch in s_):
                    ids = [self.anatomy.symbol(ch) for ch in s_]
                    if all(i is not None for i in ids):
                        pool.append(ids)
        self._corpus_cache = pool
        return pool

    def dreams(self, n=None, with_who=False):
        """dreams start where the store is strongest and run by pattern completion until the recall
        is half as sure as a memory of its own (relative to the store, not a constant).
        THE DREAM KNOWS WHO SPOKE (dream_who, 0 = off; 2026-09-13, the twenty-first defect): the store keeps who said each memory
        (its own song written at a smile, W = 1), and on the served body 74% of the onsets and 58% of a night's dreams were its own
        garbled utterances, replayed as if the world had said them: the night, once strong, taught the cortex the child's babble
        as the language. With dream_who on, dreams start where the WORLD spoke (the world's onsets), run through its own replies as
        lived, and with_who returns, beside each dream, who said each symbol; the lesson then enters its own symbols as its own
        sound (the corollary discharge, as awake) and owes no forecast of them (one predicts the environment). A disclosed constant."""
        n = int(self.cfg["night_starts"] if n is None else n)
        if str(self.cfg.get("dream_source", "store")) == "utterances" and self.utts:
            # the utterances heard, whole, drawn by strength (the recent, still strong, more), with replacement when fewer than asked
            S_ = torch.tensor(self.utt_S, dtype=torch.float); p_ = S_.clamp_min(1e-6) / S_.clamp_min(1e-6).sum()
            # THE OLD IN THE DRAW (dream_old_share; 2026-09-15): a share of the night's dreams drawn uniformly over the whole memory,
            # the faded utterances of earlier days as likely as the fresh ones (replay reaches remote memories too); the rest by
            # strength as before. Without it every night leaned to the newest days' style and the held-out lines drifted.
            n_old = int(round(n * float(self.cfg.get("dream_old_share", 0.0)))); n_new = n - n_old
            idx = torch.multinomial(p_, n_new, replacement=bool(len(self.utts) < n_new), generator=self.gen).tolist() if n_new > 0 else []
            if n_old > 0:
                idx += torch.randint(0, len(self.utts), (n_old,), generator=self.gen).tolist()
            # THE DRAW'S SERIALS (2026-09-16, night 226): which utterances the night dreamt, by their serials, kept in the night's report
            # so a night can be re-run exactly on a copy (night 226 diverged on its draws and the copy's draws were others)
            self._last_draw = [int(self.utt_N[i]) if i < len(self.utt_N) else -1 for i in idx]
            end_ = [self.end_id] if int(self.cfg.get("offset_ticks", 0)) > 0 else []
            # THE EXCHANGE REPLAYED (dream_pair, the utterances that followed; dream_gap rests between, the pause compressed as replay
            # compresses it): a dream is the utterance and its successor in time when the memory still holds it
            pair = int(self.cfg.get("dream_pair", 0)); gap = [self.sil] * max(0, int(self.cfg.get("dream_gap", 1)))
            out = []
            for i in idx:
                d = list(self.utts[i]); j = i
                for _ in range(pair):
                    if j + 1 < len(self.utts) and self.utt_N[j + 1] == self.utt_N[j] + 1:
                        d = d + gap + list(self.utts[j + 1]); j += 1
                    else:
                        break
                # THE THIRTY-THIRD DEFECT (2026-09-18, night 276 failed: "the stream's cache outgrew the window"): an utterance longer
                # than the cortex's window (two lines with no offset between them, 72 symbols) cannot be dreamt whole; the lockstep
                # batch is as long as its longest dream and the stream's cache holds a window at most. A dream is clipped to the
                # window; the first night to draw that utterance, the first with 2048 dreams, failed whole and reset the day.
                W_ = int(self.m.window) - len(end_)
                if len(d) > W_:
                    out.append(d[:W_])                                 # cut, not ended: the turn did not end there (the review of 2026-09-19)
                else:
                    out.append(d + end_)
            # THE NIGHT READS (dream_corpus_n; 2026-09-18, item 44, the user's word: pretrain it, all local and live): a number of
            # sentences of a corpus drawn uniformly into the night's dreams, through the same lesson, as words overheard; the
            # corpus is read once, as the parent reads it to the child by day (lower case, no commas or quotes, sentences of the
            # window's length), and kept as symbol ids. A disclosed constant; the held-out ruler reads its effect each morning.
            n_corp = int(self.cfg.get("dream_corpus_n", 0))
            if n_corp > 0:
                pool = self._corpus_pool()
                if pool:
                    idx_c = torch.randint(0, len(pool), (n_corp,), generator=self.gen).tolist()
                    out += [list(pool[k]) + end_ for k in idx_c]
                    self._n_corpus_dreams = len(idx_c)
            return (out, [[False] * len(d) for d in out]) if with_who else out
        who_on = int(self.cfg.get("dream_who", 0)) > 0 and self.store.n() > 0
        starts = self.store.sample_starts(n, gen=self.gen, mask=(self.store.W != 1) if who_on else None)
        outw = []
        d_ = float(self.cfg["bag_decay"]); ref = self.store.self_confidence(qnorm=(1.0 / (1.0 - d_ * d_)) ** 0.5)   # a full context's norm
        floor = float(self.cfg["dream_floor_rel"]) * ref
        out = []
        a_hit, a_rec = float(self.cfg["dream_adapt"]), float(self.cfg["dream_recover"])
        chain = int(self.cfg.get("store_chain", 0)) and self.store.N.shape[0] == self.store.n()
        with torch.no_grad():
            for j in starts:
                if chain and int((self.store.N[j] >= 0).sum()) > 0:
                    # THE EPISODE AS LIVED: the onset's first symbol, then the slots in the order they were written, to the utterance's
                    # end; at a branch (a frame heard with several continuations) a draw by strength, the recent and the rewarded more
                    ids = [self.m.nearest(self.store.K[j])]; k = int(j); seen = {k}
                    who = [bool(self.store.W[j] == 1)]                    # who said each symbol: the onset's first by its own slot
                    tag = None
                    if int(self.cfg.get("dream_tag", 0)) > 0:
                        # THE DREAM FOLLOWS ONE UTTERANCE (dream_tag; the twenty-second defect): at the onset a continuation is drawn by
                        # strength as before, and the dream then follows the utterance that wrote it, slot by slot, ending where its
                        # trace ends; without the tag a chain drew a successor from another utterance at every shared slot, and half
                        # a night's dream text was stitched across lines after four symbols. A link from before the tags follows as before.
                        _, t_ = self.store.draw_link(j, gen=self.gen); tag = int(t_) if t_ >= 0 else None
                    for _ in range(int(self.cfg["dream_max"])):
                        ids.append(self.m.nearest(self.store.V[k])); who.append(bool(self.store.W[k] == 1))
                        nk = self.store.successor(k, gen=self.gen, tag=tag)
                        if nk < 0 or nk in seen:
                            if bool(self.store.B[k]) and int(self.cfg.get("offset_ticks", 0)) > 0:
                                ids.append(self.end_id); who.append(bool(self.store.W[k] == 1))   # the memory ends where the world went quiet
                            break
                        k = nk; seen.add(k)
                    if len(ids) >= 2:
                        out.append(ids); outw.append(who)
                    continue
                bag = self.store.K[j].clone()                           # a dream's context: per symbol, as the keys are
                # the dream begins with the context's own last symbol, read from the key (a key is the bag before the
                # memory's symbol, its newest term whole): at an onset that is the utterance's first symbol, which the
                # store never kept as a memory of its own (dreams began 'og will go', and the cortex lost every line's
                # first symbols: its trace fell from 60 to 38 of 82, runs 53/54)
                ids = [self.m.nearest(bag)]
                adapt = torch.ones(self.store.n(), device=self.dev)      # neural adaptation: a recalled memory tires
                fa_ = float(self.cfg.get("store_floor_abs", 0.0))      # the forgetting floor: absolute when set (the thirty-first defect)
                s_floor = fa_ if fa_ > 0 else float(self.cfg["store_floor_rel"]) * float(self.store.S.mean())
                for _ in range(int(self.cfg["dream_max"])):
                    pred, conf, win = self.store.read(bag, adapt=adapt)
                    if conf < floor or (win >= 0 and (float(self.store.S[win] * adapt[win]) < s_floor
                                                      or float(adapt[win]) < float(self.cfg["dream_exhaust"]))):
                        break                                             # unsure, or the memory is exhausted (a slot fires at most twice)
                    lg = self.m.readout(pred).clone()
                    lg[self.bans] = float("-inf")
                    if self.end_id != self.sil:
                        lg[self.sil] = float("-inf")
                    nid = int(lg.argmax())
                    if nid == self.sil:                                   # the rest form: the recall itself expects the quiet
                        ids.append(self.sil); break
                    ids.append(nid)
                    if int(self.cfg.get("offset_ticks", 0)) > 0 and win >= 0 and bool(self.store.B[win]):
                        ids.append(self.end_id); break                    # the memory ends where the world went quiet
                    adapt = 1.0 - a_rec * (1.0 - adapt)                   # recovery toward 1
                    # the recalled memory tires fully each time it fires, and every slot tires in
                    # proportion to how much it fired (neural adaptation), so a cycle exhausts itself
                    # even when the attention is spread over near-duplicate memories of one context
                    adapt = adapt * (1.0 - (1.0 - a_hit) * self.store._last_w)
                    if win >= 0:
                        adapt[win] *= a_hit
                    bag = float(self.cfg["bag_decay"]) * self.m.shift(bag) + self.m.E.weight[nid]
                if len(ids) >= 3:
                    out.append(ids); outw.append([False] * len(ids))     # a completed dream: the world's, as before
        return (out, outw) if with_who else out

    def _dream_obs(self, xs):
        """A DREAM'S OBSERVATIONS BY CHANNEL (the core refactor's step R4): the dream's symbols `xs` ([T], or [B, T] for a batch) on the
        words (channel 0), every other channel quiet over the same positions (its rest; a vector channel's zeros, the face a dream has
        always had). Each later effector rests over them (step R5: its acts its rest; its own acts replayed are step R8). The night over
        frames, each channel's own stored codes, is step R8."""
        obs = {c_.name: (xs if i_ == 0 else c_.quiet(tuple(xs.shape), self.dev)) for i_, c_ in enumerate(self.anatomy.channels)}
        for e_ in self.anatomy.effectors[1:]:
            obs[e_.name] = torch.full(tuple(xs.shape), int(e_.rest_id), dtype=torch.long, device=self.dev)
        return obs

    def _dream_inputs(self, ids, mem_on):
        """a dream as a window: fresh bands (a night's working state), the store leading if mem_on. Returns obs (by channel: the words
        the dream's symbols, the others quiet), xos (no own sound), bundles, reads and y the targets"""
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
                u = m.inputs(self.anatomy, self._dream_obs(torch.tensor(xs[:n], device=self.dev)), xos[:n], torch.stack(bundles))
                C = m.stream(u)[-1]
                bands = m.band_update(bands, C)
        return (self._dream_obs(torch.tensor(xs, device=self.dev)), xos, torch.stack(bundles), torch.stack(reads),
                torch.tensor(ids, device=self.dev))

    def _dream_batch(self, dream_list, own_list=None):
        """THE DREAMS IN LOCKSTEP (2026-09-13): a list of dreams as one right-padded batch, the bands run along each as _dream_inputs
        runs them one at a time (a causal cortex: a dream's positions never see the padding after them). Returns obs (by channel: the
        words xs [B, T], every other channel quiet: the diary's face [B, T, 2] of zeros), xos [B, T], bundles [B, T, nb, d], reads
        [B, T, d] (zeros: the store is off in the lesson), y [B, T] the targets and w [B, T] their weights (1 on a dream's own
        positions, 0 on the padding)"""
        m = self.m; B = len(dream_list); T = max(len(ids) for ids in dream_list); nb = len(m.clocks)
        xs = torch.full((B, T), self.sil, dtype=torch.long, device=self.dev)
        y = torch.full((B, T), self.sil, dtype=torch.long, device=self.dev)
        w = torch.zeros(B, T, device=self.dev)
        xos = torch.full((B, T), self.sil, dtype=torch.long, device=self.dev)      # a dream: no own sound, unless the dream knows who spoke
        for i, ids in enumerate(dream_list):
            L = len(ids); own = own_list[i] if own_list is not None else None
            y[i, :L] = torch.tensor(ids, dtype=torch.long, device=self.dev)
            if own is None:
                if L > 1:
                    xs[i, 1:L] = torch.tensor(ids[:-1], dtype=torch.long, device=self.dev)
                w[i, :L] = 1.0
            else:
                for t in range(1, L):                                  # its own symbols enter as its own sound (dream_who)
                    (xos if own[t - 1] else xs)[i, t] = ids[t - 1]
                w[i, :L] = torch.tensor([0.0 if o else 1.0 for o in own], device=self.dev)   # no forecast owed of its own act
        obs = self._dream_obs(xs); reads = torch.zeros(B, T, m.d, device=self.dev)
        bundles = torch.zeros(B, T, nb, m.d, device=self.dev); bands = torch.zeros(B, nb, m.d, device=self.dev)
        with torch.no_grad():
            cache = [None] * len(m.blocks)                                        # the stream's keys and values so far, per block
            for t in range(T):
                bundles[:, t] = bands
                if t + 1 < T:
                    C = m.stream_step(m.inputs(self.anatomy, {k_: v_[:, t] for k_, v_ in obs.items()}, xos[:, t], bundles[:, t]), cache)
                    bands = m.band_update_b(bands, C)
        return obs, xos, bundles, reads, y, w

    def _night_step(self, opt):
        gn = torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0)
        if bool(torch.isfinite(gn)):
            opt.step()
        opt.zero_grad(set_to_none=True)

    def night(self):
        """the night, in order: the dreams drawn from the store or the utterance memory (dreams()); NREM, the cortex learning on
        them with the recall off as its input; REM, the cortex running free and the forecast heads learning; the value ladder's
        replay; the gauge before and after; then the store fades, the working state wakes fresh and the body is saved. A night is
        kept whatever it does; only a non-finite lesson reloads the evening's organs."""
        self.asleep = True
        m = self.m
        rep = {"night": self.nights + 1, "tick": self.ticks}
        if not self._offset_done:
            # THE NIGHT ENDS EVERY UTTERANCE (the review of 2026-09-19): the switch fires on the tick count, blind to a line in progress;
            # a line the night fell inside was kept open until the morning, glued to the first line of the day under the dusk's tag
            self._offset(); self._offset_done = True
        try:
            # SLEEP NEED SCALES WITH THE DAY'S PLASTICITY (night_load, 0 = off; 2026-09-11, nights 114-115): with the parent talking
            # twice as much, the day wrote twice the memories and the night, dreaming its fixed 48 starts, consolidated less far (the
            # gauge after it 0.88 -> 0.71, the loss ending 0.09 -> 0.18). Slow-wave activity in a brain grows with the plasticity of
            # the wake before it (the synaptic homeostasis of Tononi and Cirelli); here the number of dreams a night starts grows
            # with the memories the day added to the store, night_load dreams per new slot, never fewer than night_starts and never
            # more than night_starts_max. A disclosed constant, not a rule about content; the store's own count, nothing read from
            # the parent.
            # --- the dreams drawn: as many as the day's new memories ask for, between night_starts and night_starts_max ---
            n_new = (self.store.n() - int(self._store_after_night)) if self._store_after_night is not None else 0
            n_new = max(n_new, int(getattr(self, "_writes_today", 0)))   # at the capacity the store's count stops growing (the review of 2026-09-19); the day's writes are the measure
            load = float(self.cfg.get("night_load", 0.0)); n_starts = None
            if load > 0.0:
                n_starts = int(min(int(self.cfg.get("night_starts_max", 192)), max(int(self.cfg["night_starts"]), round(load * max(0, n_new)))))
            who_on = int(self.cfg.get("dream_who", 0)) > 0 and int(self.cfg.get("night_batch", 0)) > 0
            self._n_corpus_dreams = 0
            if who_on:
                dreams, owns = self.dreams(n_starts, with_who=True)
            else:
                dreams = self.dreams(n_starts); owns = None
            n_own_ = len(dreams) - int(getattr(self, "_n_corpus_dreams", 0))       # the gauge reads the body's own dreams only (the review of 2026-09-19): comparable across nights with and without reading
            g_dreams = dreams[:n_own_] if n_own_ > 0 else dreams
            g_owns = (owns[:len(g_dreams)] if owns is not None else None)
            rep["corpus_dreams"] = int(getattr(self, "_n_corpus_dreams", 0))
            rep["dreams"] = len(dreams); rep["new_slots"] = int(n_new); rep["draw_serials"] = list(getattr(self, "_last_draw", []))
            if owns is not None:                                        # its own symbols in capitals, to be read
                rep["examples"] = ["".join(self.anatomy.decode([i]).upper() if o else self.anatomy.decode([i]) for i, o in zip(d, w_))[:32] for d, w_ in zip(dreams[:8], owns[:8])]
                rep["own_share"] = round(sum(sum(w_) for w_ in owns) / max(1, sum(len(w_) for w_ in owns)), 3)
            else:
                rep["examples"] = [self.anatomy.decode(d)[:32] for d in dreams[:8]]
            rep["mean_len"] = round(sum(len(d) for d in dreams) / len(dreams), 1) if dreams else 0
            if not dreams:
                rep["note"] = "the store holds nothing to dream"
            else:
                # --- NREM: sleep's own optimizer; the dreams in batches (night_batch > 0) or one step per round ---
                self._night_away()                                  # the night's lessons on night_dev (the dreams already drawn at home)
                before, nsym = self.gauge(g_dreams, g_owns); before_cos = self._gauge_cos
                # THE MOMENT'S HORIZON (night_beta2; 2026-09-16, night 226): a fresh optimizer's second moment forms over about a thousand
                # steps at 0.999, so at the third round an outlier gradient on a parameter whose moment is still small is normalised
                # into a step many times the rate, which the global clip does not bound (night 226's loss rose 0.23 -> 0.34 in one
                # round and the cortex was left in a basin the rate cannot climb). At 0.99 the moment forms in a hundred steps, before
                # the rounds where the night's outliers arrive. A disclosed constant; 0.999 = as before.
                # act_pred and the corrections take no gradient at night (anatomy 31 guards it). R8's replay of act_pred's targets, each
                # weighted by the replayed dopamine's credit, must step them through their gate (body/core/timing.py GatedAdam, at the
                # replayed weight), and whatever share of a weighted target reaches the stream would be taught whole by this Adam, as by
                # the day's (it divides each parameter's step by its recent gradient size): the waking lesson keeps act_inv's labels out
                # of the stream for that reason (`_timing_loss`)
                opt = torch.optim.Adam(m.parameters(), lr=float(self.cfg["night_lr"]), betas=(0.9, float(self.cfg.get("night_beta2", 0.999))))   # sleep's own plasticity
                sig = float(self.cfg["sigreg"])
                # THE PLASTICITY RAMPS (night_warm, 0 = off; 2026-09-06): a fresh optimizer's first steps move every weight
                # by the whole rate at once, and on a wide, deep cortex that first step is a shove (the 179M body's NREM loss
                # doubled or tripled at step one every night and spent the night recovering; the 32M reference never did).
                # Sleep's plasticity in a brain rises over the first minutes of NREM; here the rate climbs linearly over the
                # first night_warm steps, then holds. A disclosed constant, not a rule about content.
                warm_ = int(self.cfg.get("night_warm", 0)); base_lr_ = float(self.cfg["night_lr"]); nstep_ = 0
                m.train()
                nrem = 0; losses = []
                # A SYNAPTIC CHANGE PER RIPPLE, NOT PER NIGHT (night_batch, 0 = off; 2026-09-13, the twentieth defect): with one step
                # per round, a night of 512 dreams and three rounds was three weight updates, and the cortex's accuracy on the parent's
                # unreplayed lines stood at 0.52 for a hundred nights while its recall of the few replayed ones read 0.86. A sharp-wave
                # ripple induces its plasticity as it happens, thousands a night, the replays interleaved (the complementary learning
                # systems of McClelland, McNaughton and O'Reilly); here the optimizer steps after every night_batch dreams, the dreams
                # shuffled each round and run in lockstep. A disclosed constant, not a rule about content.
                nbatch_ = int(self.cfg.get("night_batch", 0))
                for _ in range(int(self.cfg["night_rounds"]) if nbatch_ > 0 else 0):
                    order = torch.randperm(len(dreams), generator=self.gen).tolist(); tot = 0.0; ok = 0
                    for i0 in range(0, len(order), nbatch_):
                        opt.zero_grad(set_to_none=True)
                        obs, xos, bundles, reads, y, w = self._dream_batch([dreams[j] for j in order[i0:i0 + nbatch_]],
                                                                           [owns[j] for j in order[i0:i0 + nbatch_]] if owns is not None else None)
                        C = m.stream(m.inputs(self.anatomy, obs, xos, bundles))
                        ll, _ = m.latent_loss(m.latent_pred(C), y, w=w)
                        if not bool(torch.isfinite(ll.detach())):
                            continue
                        ll.backward(); tot += float(ll.detach()); ok += 1; nstep_ += 1
                        if warm_:
                            for g_ in opt.param_groups:
                                g_["lr"] = base_lr_ * min(1.0, nstep_ / warm_)
                        self._night_step(opt); nrem += 1
                    losses.append(round(tot / max(1, ok), 3))
                for _ in range(int(self.cfg["night_rounds"]) if nbatch_ <= 0 else 0):
                    opt.zero_grad(set_to_none=True); tot = 0.0; ok = 0
                    for ids in dreams:
                        # the hippocampus replays the sequence; the cortex must carry it itself (the read
                        # is not an input to the lesson, or the cortex learns to copy the recall and the
                        # gauge, taken alone, stays flat: run 6, day 4)
                        obs, whos, bundles, reads, y = self._dream_inputs(ids, mem_on=False)
                        C = m.stream(m.inputs(self.anatomy, obs, whos, bundles))
                        ll, _ = m.latent_loss(m.latent_pred(C), y)          # no SIGReg: with a fixed lexicon and the PFC's
                                                                             # objective off the trunk, nothing can collapse
                        if not bool(torch.isfinite(ll.detach())):
                            continue
                        (ll / len(dreams)).backward(); tot += float(ll.detach()) / len(dreams); ok += 1
                    if ok:
                        nstep_ += 1
                        if warm_:
                            for g_ in opt.param_groups:
                                g_["lr"] = base_lr_ * min(1.0, nstep_ / warm_)
                        self._night_step(opt); nrem += 1; losses.append(round(tot, 3))
                mid, _ = self.gauge(g_dreams, g_owns); mid_cos = self._gauge_cos      # the gauge after NREM, before REM
                # --- REM: the cortex runs free from each dream's first symbols on its own readout,
                # a quarter of the night in rounds (biology's share), each round one batched step
                rem_cos = []; rem_steps = 0; rem_imag = None
                if str(self.cfg.get("rem_form", "forecast")) == "imagine":
                    rem_imag = self._rem_imagine_rounds(dreams); rem_steps = rem_imag["rounds"]
                else:
                  for _ in range(int(self.cfg["rem_rounds"])):
                    opt.zero_grad(set_to_none=True); rc = []
                    for ids in dreams[:int(self.cfg["rem_dreams"])]:
                        fl, fc = self._rem_rollout(ids, sig)
                        if fl is None or not bool(torch.isfinite(fl.detach())):
                            continue
                        (fl / max(1, min(len(dreams), int(self.cfg["rem_dreams"])))).backward(); rc.append(fc)
                    if rc:
                        self._night_step(opt); rem_steps += 1; rem_cos.append(sum(rc) / len(rc))
                # --- the value ladder replays its lived pairs once; the gauge after; a non-finite night reloads the evening's organs ---
                self._night_home()                                  # home before the value replay, the gauge after, the fade and the save
                self._value_replay()
                m.eval()
                del opt
                finite = all(bool(torch.isfinite(p).all()) for p in m.parameters())
                after, _ = self.gauge(g_dreams, g_owns); after_cos = self._gauge_cos
                # THE NIGHT STANDS (the user's word, 2026-09-11 20:00: 'remove night rollback'): from night 106 to 115 a night whose dream recall
                # fell by more than 0.15 was discarded and the organs returned to the evening's save; nothing in biology does that, and it
                # never fired after the night it was built for. A night is kept whatever it does. Only a non-finite lesson (the arithmetic
                # broken, not a learning outcome) reloads the evening's organs, so the body is not left with NaN for weights.
                rep["discarded"] = not finite; rep["undone_for"] = "non-finite" if not finite else None
                if rep["discarded"] and self.save_path and os.path.exists(self.save_path):
                    sd = torch.load(self.save_path, map_location="cpu", weights_only=False)
                    m.load_state_dict(sd["organs"]); m.to(self.dev)
                    after, _ = self.gauge(g_dreams, g_owns); after_cos = self._gauge_cos
                rep.update({"nrem_steps": nrem, "nrem_loss": losses[:3] + (["..."] if len(losses) > 6 else []) + losses[-3:],
                            "nrem_curve": [round(float(x), 4) for x in losses],   # the whole curve (2026-09-12): to read where the rounds stop paying
                            "rem_steps": rem_steps, "rem_cos": (round(rem_cos[-1], 3) if rem_cos else None),
                            "rem_cos_first": (round(rem_cos[0], 3) if rem_cos else None), "rem_imagined": rem_imag,
                            "gauge": {"before": before, "after_nrem": mid, "after": after, "symbols": nsym,
                                      "cos_before": before_cos, "cos_after_nrem": mid_cos, "cos_after": after_cos}})
            # --- the rest: the store fades, the working state wakes fresh, the body is saved ---
            if getattr(self, "_store_fresh", False):
                # A REBUILT STORE LIVES A DAY BEFORE ITS FIRST FADE (the ledger's rule, now the body's; the review of 2026-09-19): raw
                # single surprises under the relative floor lose the middles of the facts (nights 252 and 262, a third of the store)
                rep["store_dropped"] = 0; self._store_fresh = False
            else:
                rep["store_dropped"] = self.store.fade(float(self.cfg["store_fade"]), float(self.cfg["store_floor_rel"]), float(self.cfg.get("store_floor_abs", 0.0)))
            self.utt_S = [v * float(self.cfg["store_fade"]) for v in self.utt_S]   # the utterances heard fade as the store does
            rep["utterances"] = len(self.utts)
            rep["store_slots"] = self.store.n(); rep["vrel"] = round(self._vrel_corr, 3)
            self._store_after_night = self.store.n()
            if not int(self.cfg.get("night_keep_bands", 0)):
                self.bands.zero_()                                    # the slow state kept across sleep when the flag is on
            if int(self.cfg.get("vcrit_norm_wake", 0)) and self.m.vc_mu.numel():
                self.m.vc_n.fill_(float(int(self.cfg.get("vcrit_norm_tau", 0)) / 32.0))   # the statistics re-form at wake
            self.bag_w.zero_(); self.bag_o.zero_(); self.n_own = 0; self.win.clear(); self.pred_prev = None; self._follow = None
            self._prev_slot = -1; self._last_write = None; self._start_armed = False; self._start_pending = False; self._seam_pending = False
            self._last_world = -10 ** 9; self._offset_done = True; self._utt_cur = []; self._utt_felt = 0.0; self._ear_trace = 0.0; self._writes_today = 0; self._turn_open = False; self._ear_release_at = None
            if int(self.cfg.get("pace_sense", 0)) and "pace" not in rep:
                rep["pace"] = self._pace_report(); self._pace_day = self._pace_day_new()   # the day's instruments of the sensed pace, then a new day's
            self._gap_foreseen = False; self._gap_paused = False; self._sh_done = True     # the gap's labels cleared, the trackers kept (the partner is the same)
            self._ear_held = False; self._ready_E = None; self._pred_ready = None; self._d_max = 0.0
            self._bands_prev = None; self._C_last = None; self.v_prev = None
            self._z_prev = None; self._z_now = None; self._e_actor = None
            for st_ in getattr(self, "motor", ()):                 # the later effectors' working state begins afresh, as the voice's (step R5)
                st_["buf"].clear(); st_["g_base"] = None; st_["e_actor"] = None; st_["acted_last"] = False; st_["now"] = None
                st_["chunk"] = 0; st_["sense"] = None; st_["fwd"] = None; st_["err"] = None     # step R6: no chunk, no sense or foresight carried over
            if getattr(self.m, "stri_wm", 0):
                self.m.wm_clear()
            if self.m.stri_W.numel() > 0:
                self.m.striatum_reset()                            # the delay line empties for the night
            self.stream.clear(); self.gate_buf.clear(); self._g_base = None; self._gate_tag = None; self._vtrace = None; self._vc_e = None
            self._after_night = True
            self._vf_e = None
            self.fatigue = 0.0
            self.sleep_pressure = 0
            self.nights += 1; self.day_n += 1
            self.last_night = rep
            if self.save_path:
                self.save()
        except Exception as e:
            rep["error"] = str(e)[:200]
            print(f"[body] night {rep.get('night')} error: {e!r}", flush=True)
            self.sleep_pressure = int(self.cfg["wake_ticks"]) // 2
            if int(self.cfg.get("pace_sense", 0)) and "pace" not in rep:
                rep["pace"] = self._pace_report(); self._pace_day = self._pace_day_new()   # a night that failed ends the day's instruments too (the day goes on
                                                                                         # awake: the next report holds the ticks from here, none counted twice)
            self.last_night = rep
        finally:
            self._night_home()                                         # a failed night never leaves the day on the night's device
            self.asleep = False
        return rep

    def _night_away(self):
        """THE NIGHT ON ANOTHER DEVICE (night_dev; 2026-09-22): the cortex and the day's small state move for the night's lessons; the
        store, the critics' float64 evidence (kept on the host by Organs.to) and every random draw stay home"""
        nd = str(self.cfg.get("night_dev", "")).strip()
        if not nd or torch.device(nd) == torch.device(self.dev):
            return
        self._home_dev = self.dev; self.m.to(nd); self.dev = nd
        self.bands, self.bag_w, self.bag_o = self.bands.to(nd), self.bag_w.to(nd), self.bag_o.to(nd)

    def _night_home(self):
        home = getattr(self, "_home_dev", None)
        if home is None:
            return
        self.m.to(home); self.dev = home; self._home_dev = None
        self.bands, self.bag_w, self.bag_o = self.bands.to(home), self.bag_w.to(home), self.bag_o.to(home)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _rem_temperature(self):
        """REM's sampling temperature: rem_temp, and under the world form scaled by the readout's base over its world-calibrated base,
        so the dreams are drawn at the sharpness the world proved (a base of 25 calibrated to 8 samples at temperature 3)"""
        rt = float(self.cfg.get("rem_temp", 0.0))
        if rt > 0 and str(self.cfg.get("sharp_form", "fixed")) == "world" and int(self.cfg.get("rem_world_temp", 0)) and float(self.sharp_cal) > 0:
            rt = rt * float(self.cfg["sharp_base"]) / float(self.sharp_cal)   # rem_world_temp 1: the dreams at the world's proved sharpness (off until the reading is trusted)
        return rt

    def _rem_imagine_rounds(self, dreams):
        """REM AS IMAGINATION (§5c), the night's share: each round, each dream imagined once; the fast critic re-solved from the evidence"""
        m = self.m; rounds = 0; n_tr = 0; rsum = 0.0
        w = float(self.cfg.get("rem_weight", 1.0)) * self._face_weight()
        for _ in range(int(self.cfg["rem_rounds"])):
            n_round = 0
            for ids in dreams[:int(self.cfg["rem_dreams"])]:
                n_, r_ = self._rem_imagine(ids); n_round += n_; rsum += r_
            if n_round:
                rounds += 1; n_tr += n_round
                m.fast_rls_solve(int(self.cfg["dopamine_band"]), prior=getattr(self, "_vf_delta", None))
        return {"rounds": rounds, "transitions": n_tr, "mean_abs_rhat": (round(rsum / n_tr, 3) if n_tr else None), "weight": round(w, 3),
                "face_slope": round(float(self._frel_gain), 3), "face_corr": round(float(self._frel_corr), 3)}

    def _rem_imagine(self, ids, k=3):
        """REM AS IMAGINATION (2026-09-08, §5c): the cortex runs free from a dream's first symbols on its sampled readout; the imagined
        events advance the striatal delay line as heard ones do; the face organ scores each imagined tick with the felt reward it
        foresees; and the fast critic's evidence takes each imagined transition exactly as it takes a lived one (the same accumulators,
        no forgetting), weighted by the face organ's proven slope: imagination counts for as much as the imaginer has proved right.
        The lived delay line and working memory are restored after. Returns (transitions taken, the sum of |foreseen reward|)."""
        m = self.m
        if (m.stri_W.numel() == 0 or not int(self.cfg.get("fast_rls", 0)) or len(ids) < k + 1
                or str(self.cfg.get("face_form", "read")) != "foresee"):
            return 0, 0.0
        w = float(self.cfg.get("rem_weight", 1.0)) * self._face_weight()
        if w <= 0.0:
            return 0, 0.0
        L = int(self.cfg["rem_steps"]); gf = float(m.gammas()[int(self.cfg["dopamine_band"])])
        line_saved = m.stri_line.clone()
        mline_saved = m.stri_mline.clone() if "stri_mline" in m._buffers else None   # the later effectors' lines (step R5), restored after
        wm_saved = (m.wm_slot.clone(), m.wm_on.clone(), m.wm_age.clone()) if getattr(m, "stri_wm", 0) else None
        m.striatum_reset()
        if wm_saved is not None:
            m.wm_clear()
        bands = torch.zeros_like(self.bands); xs = [self.sil] + list(ids[:k])
        bundles, Cs = [], []; z_prev = None; f_prev = 0.0; e = None; n_up = 0; rsum = 0.0
        one = torch.ones(1, dtype=torch.float64)
        with torch.no_grad():
            for step in range(len(xs) + L):
                if step >= len(xs):
                    lg = m.readout(m.latent_pred(Cs[-1])).clone(); lg[self.bans] = float("-inf")   # the rest may be imagined: the world's stop
                    rt = self._rem_temperature()
                    xs.append(int(torch.multinomial(torch.softmax(lg / rt, 0).cpu(), 1, generator=self.gen)) if rt > 0 else int(lg.argmax()))   # drawn on the host: the CPU generator, the same stream on any device
                x = xs[step]
                m.striatum_push(0 if x != self.sil else 3, x if x != self.sil else 0)
                bundles.append(bands.clone())
                n = step + 1
                u = m.inputs(self.anatomy, self._dream_obs(torch.tensor(xs[:n], device=self.dev)),
                             torch.full((n,), self.sil, dtype=torch.long, device=self.dev), torch.stack(bundles))
                C = m.stream(u)[-1]; Cs.append(C)
                bands = m.band_update(bands, C)
                z_now = m.stri_in()
                if step > k and z_prev is not None:                       # the free-running part: imagined transitions
                    xa_p = torch.cat([z_prev.detach().cpu().double(), one]); xa_n = torch.cat([z_now.detach().cpu().double(), one])
                    e = (gf * e if e is not None else torch.zeros_like(xa_p)) + xa_p
                    m.vf_A.addr_(e, xa_p - gf * xa_n, alpha=w); m.vf_b.add_(e * (w * f_prev))
                    n_up += 1; rsum += abs(f_prev)
                z_prev = z_now; f_prev = self._foresee(z_now.detach().cpu().double() if str(self.cfg.get("face_input", "cortex")) == "striatum" else C.detach().cpu().double())   # the felt reward foreseen for the next imagined tick
        m.stri_line.copy_(line_saved)
        if mline_saved is not None:
            m.stri_mline.copy_(mline_saved)
        if wm_saved is not None:
            m.wm_slot.copy_(wm_saved[0]); m.wm_on.copy_(wm_saved[1]); m.wm_age.copy_(wm_saved[2])
        return n_up, rsum

    def _rem_rollout(self, ids, sig, k=3):
        m = self.m
        L = int(self.cfg["rem_steps"])
        if len(ids) < k + 1:
            return None, None
        bands = torch.zeros_like(self.bands); bag = torch.zeros_like(self.bag_w)
        xs = [self.sil] + list(ids[:k])
        bundles, Cs, bnext = [], [], []
        for step in range(len(xs) + L):
            if step >= len(xs):
                # its imagined next symbol, read off its forecast (no gradient through the choice),
                # heard as the world's in the next time step
                with torch.no_grad():
                    lg = m.readout(m.latent_pred(Cs[-1])).clone(); lg[self.bans] = float("-inf"); lg[self.sil] = float("-inf")
                    rt = self._rem_temperature()
                    # THE DREAM SAMPLED, NOT TAKEN AT ITS MODE (rem_temp > 0; 2026-09-06): biology's REM is noisy; the greedy
                    # continuation reproduces the store's most frequent lines and consolidates nothing new. 0 = greedy (as before).
                    xs.append(int(torch.multinomial(torch.softmax(lg / rt, 0).cpu(), 1, generator=self.gen)) if rt > 0 else int(lg.argmax()))   # drawn on the host: the CPU generator, the same stream on any device
            with torch.no_grad():
                bag = float(self.cfg["bag_decay"]) * (m.shift(bag) if xs[step] != self.sil else bag) + (m.E.weight[xs[step]] if xs[step] != self.sil else 0.0)
            bundles.append(bands.clone())
            n = step + 1
            u = m.inputs(self.anatomy, self._dream_obs(torch.tensor(xs[:n], device=self.dev)),
                         torch.full((n,), self.sil, dtype=torch.long, device=self.dev), torch.stack(bundles))
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
            if len(pairs) < 4 or (int(self.cfg.get("fast_rls", 0)) and b == int(self.cfg["dopamine_band"])):   # the fast head is solved, not replayed
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
        """THE SLEEP SWITCH'S CALL (the tick's last phase): the world pauses (the core refactor's step R9, body/core/world.py: the diary's
        lets its queue go, what this call always did first), the night runs, and the morning resumes the world where it stood. A night
        called by hand (a tool's, on a copy) leaves the world as it is, as it left the queue."""
        self.world.pause()
        try:
            self.night()
        finally:
            self.world.resume()
