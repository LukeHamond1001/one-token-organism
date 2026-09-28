"""the hippocampus's side of a life (a mixin of `Life`, body/life.py): the recall's query under the cortex key (`query_from`,
pattern separation), the waking read and its tiring (`_recall`, `_tire`), the two fast bags and the slow context that make the read
query and the write key (`bag`, `key`, `_ctx_scaled`, `take_world`, `take_own`, `note_offset`, `rest_tick`), and the body's own last
utterance written as an episode (`_consolidate_own`). The store is body/model.py's `FastStore`; the world's symbols are written in
`_step` (cortex.py) and the utterances heard enter their memory at `_offset` (senses.py).

Moved verbatim from body/life.py (review 2026-09-22 section 4, step 2)."""
import torch
import torch.nn.functional as F

# A130 (goal_key): the held word. GOAL_TAU, the trace's time constant in ticks: 60 ticks, 9 s at the sim's 150 ms tick, so the held word
# is at e^-2 by 18 s, where Peterson and Peterson (1959) found recall without rehearsal near its floor (a prefrontal delay trace of
# seconds, Funahashi et al. 1989). GOAL_SCALE, its weight in the key beside the stream's and the heading's unit directions: 1.0, equal
# weight, R7f's own convention for the heading (ours).
GOAL_TAU = 60
GOAL_SCALE = 1.0
# A128 (ctx_key): the held context, the temporal context model's drifting context (Howard and Kahana 2002): a leaky integral of the
# stream's pattern-separated direction, so a thing that just left the senses stays in the key for a while and the frames it was in
# keep coming back into the acts (object permanence's first substrate: the representation outlasts the sight of it). CTX_TAU 60 ticks,
# 9 s: the A-not-B delay an infant tolerates grows from 2 s at 7.5 months to 10 s at 12 months (Diamond 1985); GOAL_TAU's own span.
# CTX_SCALE 1.0: equal weight with the stream's direction and the heading (R7f's convention; ours).
CTX_TAU = 60
CTX_SCALE = 1.0
# A137 (inner_speech): the INNER word. The voice chooses a word every tick and its gate decides whether to sound it; with inner_speech
# on, a word chosen and not sounded is held in the same trace when the voice was sure of it (its probability at least INNER_P): private
# speech gone covert (Vygotsky: inner speech grows out of speech to oneself, the articulators stilled), the same key. INNER_P 0.5: the
# word likelier than every other together (ours).
INNER_P = 0.5
# A138 (imagine_key): IMAG_SCALE, the imagined future's weight in the key beside the stream's direction, the heading's, the held word's
# and the held context's: 1.0, equal weight (R7f's convention; ours).
IMAG_SCALE = 1.0


class MemoryMixin:
    def query_from(self, C, learn=False):
        """the recall's query under the cortex key: the stream's state, its running mean taken out (pattern separation), as a unit
        direction at the norm a full context's bag would have (the query's norm is the recall's inverse temperature; key_scale, a
        disclosed constant near the bag's own norm). With learn, the running mean takes this state in first."""
        c = C.detach().float()
        if learn:
            self._c_n += 1; a = max(1.0 / self._c_n, 1.0 - 0.9995)
            self._c_mu = self._c_mu + a * (c - self._c_mu)
        if int(self.cfg.get("recall", 0)):
            # STEP R7f (recall into action; body/core/frames.py): the key is the stream plus the heading, their unit directions at equal
            # weight, at the key's scale
            d = F.normalize(c - self._c_mu, dim=0)
            k = d + self._heading_code().to(c.dtype)
            if int(self.cfg.get("ctx_key", 0)):
                # A128 (ctx_key, the brain sprint): THE HELD CONTEXT JOINS THE KEY. The stream's direction integrated with the time constant
                # CTX_TAU (moved once a tick, with the running mean: learn), at CTX_SCALE: the key lags the senses, so a frame is written
                # under, and recalled by, the context it came out of (the temporal context model); the night lets it go
                ctx = getattr(self, "_ctx", None)
                if learn:
                    ctx = d.clone() if ctx is None else ctx + (d - ctx) / float(CTX_TAU)
                    self._ctx = ctx
                if ctx is not None:
                    k = k + CTX_SCALE * ctx
            g = getattr(self, "_goal", None)
            if g is not None and int(self.cfg.get("goal_key", 0)):
                # A130 (goal_key, the brain sprint): THE HELD WORD JOINS THE KEY. The body's own last said word, held as a fading unit
                # direction (body/core/frames.py _goal_trace, time constant GOAL_TAU), at GOAL_SCALE: the frames written while it holds
                # and the frames recalled into the effectors' proposals are conditioned on what it said (Vygotsky's private speech; the
                # prefrontal trace biasing hippocampal retrieval, Miller and Cohen 2001)
                k = k + GOAL_SCALE * g.to(c.dtype)
            im = getattr(self, "_imag", None)
            if im is not None and int(self.cfg.get("imagine_key", 0)):
                k = k + IMAG_SCALE * im.to(c.dtype)                  # A138: the imagined future (sleep.py _imagine) joins the key
            return F.normalize(k, dim=0) * float(self.cfg.get("key_scale", 2.5))
        return F.normalize(c - self._c_mu, dim=0) * float(self.cfg.get("key_scale", 2.5))

    def _tire(self, win_, rt_):
        if rt_ > 0.0 and self.store.A.numel() == self.store.n():
            if win_ >= 0:
                self.store.A[win_] *= (1.0 - rt_)                            # the winner tires
            if int(self.cfg.get("tire_recover", 0)):
                # DEFECT 1 FIXED (tire_recover, step R7c): the recovery written in place, as the tiring is, so the store's own buffer keeps
                # it through the next write of a new slot (physiology.py SWITCHES)
                self.store.A.copy_(1.0 - float(self.cfg.get("read_recover", 0.97)) * (1.0 - self.store.A))
            else:
                self.store.A = 1.0 - float(self.cfg.get("read_recover", 0.97)) * (1.0 - self.store.A)   # all recover toward rest

    def _recall(self, bag, end_vec=None, tire=None):
        """the waking read, carrying the episode it is in when read_follow is on (the gain, > 1): after a read whose winner continues
        the episode followed, the same tag is kept; after a read that landed elsewhere, the winner's newest link names the episode"""
        fw = float(self.cfg.get("read_follow", 0.0))
        if int(self.cfg.get("episode_chain", 0)) and fw > 1.0:
            # THE EPISODE KEPT PER UTTERANCE: the follow is (slot, tag, position); the next element of that episode is easier to recall;
            # a read that lands on it continues the episode, a read elsewhere joins the newest episode the winner belongs to
            st = self.store; fo = self._follow if (self._follow is not None and len(self._follow) == 3) else None
            nxt = st.next_in(fo[1], fo[2]) if fo is not None else -1
            read, conf, win_ = st.read(bag, end_vec=end_vec, tire=tire, follow=None, follow_gain=fw, boost=(nxt if nxt >= 0 else None))
            if win_ >= 0:
                if fo is not None and int(win_) == nxt:
                    self._follow = (int(win_), fo[1], fo[2] + 1)
                else:
                    eps = st.episodes_of(int(win_))
                    self._follow = (int(win_), eps[-1][0], eps[-1][1]) if eps else None
            else:
                self._follow = None
            return read, conf, win_
        follow = self._follow if (fw > 1.0 and self._follow is not None and len(self._follow) == 2) else None
        read, conf, win_ = self.store.read(bag, end_vec=end_vec, tire=tire, follow=follow, follow_gain=fw)
        if fw > 1.0:
            st = self.store
            if win_ >= 0 and st.N.shape[0] == st.n():
                tag = -1
                if follow is not None:
                    slot, t_ = follow
                    row = st.N[slot]
                    if bool(((row >= 0) & (st.NE[slot] == int(t_)) & (row == int(win_))).any()):
                        tag = int(t_)                                       # the read continued the episode: keep following it
                if tag < 0 and int(st.N[win_][0]) >= 0:
                    tag = int(st.NE[win_][0])                               # elsewhere: the winner's newest continuation names the episode
                self._follow = (int(win_), tag) if tag >= 0 else None
            else:
                self._follow = None
        return read, conf, win_

    @property
    def bag(self):
        """the read query: the world's context, shifted by as many lags as it has said since, plus the
        efference copy of what it said (the plan is known to the sequencing system in full; it is the
        hearing of it that is suppressed): the query after its own "b" is the key the world's "b"
        would have made"""
        w = self.bag_w
        for _ in range(min(int(self.n_own), 64)):
            w = self.m.shift(w)
        if int(self.cfg.get("bag_own_fade", 0)) == 2 and self.n_own > 0:
            # form 2: the world's context in the query fades by the symbol rate for each symbol the body said since the world's
            # last (as the shift advances its lag), the state itself untouched (the keys are the world's alone)
            rest_ = float(self.cfg.get("bag_rest_decay", 0.0)) or float(self.cfg["bag_decay"])
            w = w * (float(self.cfg["bag_decay"]) / rest_) ** min(int(self.n_own), 64)
        q = w + float(self.cfg["bag_own_weight"]) * self.bag_o
        lam = float(self.cfg.get("key_ctx", 0.0))
        if lam > 0.0:                                          # the slow context: the world's latest utterance, whole (the question)
            q = q + lam * self._ctx_scaled(self.ctx_cur, q)
        return q

    @property
    def key(self):
        """the write key: the world's context alone. Corollary discharge suppresses the response to
        self-produced sound, so a memory of the world's sequence is stored under the world's context,
        never under its own babble (own symbols in the key at 0.6 broke recall by content; at 0.5 the
        stale key, the cue alone, outmatched the continuation key after its own first letter, cosine
        0.96 to 0.94, and the mouth stuttered the first letter: run 16, day 2). With key_ctx, the slow
        context of the world's recent symbols joins it (its norm brought to the fast bag's)."""
        lam = float(self.cfg.get("key_ctx", 0.0))
        if lam <= 0.0:
            return self.bag_w
        return self.bag_w + lam * self._ctx_scaled(self.ctx_prev, self.bag_w)     # keyed by the utterance before this one

    def _ctx_scaled(self, ctx, bag):
        """the slow context at the fast bag's norm, so key_ctx is a plain ratio between the two"""
        n = float(ctx.norm())
        return ctx * (float(bag.norm()) / n) if n > 1e-6 else torch.zeros_like(ctx)

    def take_world(self, i):
        """a world symbol enters the contexts (the tick's rule, and the probes'): the fast bag shifts and takes it; the slow context
        is the utterance's own order-free bag (recency-weighted by ctx_decay a symbol), begun afresh at the utterance's first symbol,
        the finished one kept beside it as the context the next utterance is keyed by; what the body said since leaves the fast bag"""
        ex = self.m.E.weight[int(i)]
        self.bag_w = self.m.shift(self.bag_w) + ex
        if not self._utt_open:                                       # the utterance's first symbol: the last one becomes the key's context
            self.ctx_prev = self.ctx_cur.clone(); self.ctx_cur = torch.zeros_like(self.ctx_cur); self._utt_open = True
        rho = float(self.cfg.get("ctx_decay", 0.95))
        if str(self.cfg.get("ctx_form", "bag")) == "shifted":      # the utterance with its order (the shift a symbol): "what is hot?" and
            self.ctx_cur = rho * self.m.shift(self.ctx_cur) + ex     # "what do we see at night?" no longer share most of their letters' weight
        else:
            self.ctx_cur = rho * self.ctx_cur + ex
        self.bag_o = torch.zeros_like(self.bag_o); self.n_own = 0

    def take_own(self, i):
        """its own symbol enters its own fast context (the efference copy the query reads); the slow context is the world's alone.
        The tick faded its own bag at the quiet rate; a symbol of its own brings the fade to the symbol rate (bag_decay) before it enters"""
        ex = self.m.E.weight[int(i)]
        rest_ = float(self.cfg.get("bag_rest_decay", 0.0)) or float(self.cfg["bag_decay"])
        self.bag_o = (float(self.cfg["bag_decay"]) / rest_) * self.bag_o
        if int(self.cfg.get("bag_own_fade", 0)) == 1:
            # THE WORLD'S CONTEXT FADES BY ITS OWN SYMBOLS TOO (bag_own_fade 1; 2026-09-15, night 207): the query shifts the world's
            # context a lag for every symbol it says (the efference copy), so its fade must advance by the symbol rate for every own
            # symbol as well, as it would have for the other voice's; under the hold alone the world's context faded at the quiet
            # rate while the child answered, and the query's geometry during its answer no longer matched the keys written while the
            # other voice answered (the questions at two rests 21 -> 16). 0 = the form served from night 206; 1 = the symbol rate.
            # THE FLAW OF FORM 1 (night 209): fading the state itself let the child's speech shape the KEYS: when it answered before
            # the other voice, the onset of that voice's answer was written under a context faded to nothing, and one fact's
            # retellings landed in two key forms (the ten retaught that day answered 3 of 10). Form 2 leaves the state to the
            # world's own timing and puts the symbol-rate fade in the query alone (the bag property), where the efference copy's
            # shift already lives: the keys never depend on what the body said; the query reads as if the world had said it.
            self.bag_w = (float(self.cfg["bag_decay"]) / rest_) * self.bag_w
        self.bag_o = self.m.shift(self.bag_o) + ex; self.n_own += 1

    def note_offset(self):
        """the world's utterance ended (the offset): the next world symbol begins a new one"""
        self._utt_open = False

    def rest_tick(self, world=False):
        """a tick's fading of the fast bags (the world half of every tick): the world's bag by the symbol rate when a world symbol
        follows (world=True), else by the quiet rate; its own bag by the quiet rate (take_own brings a symbol's tick to the symbol
        rate); the slow context fades by symbols, not ticks"""
        rest_ = float(self.cfg.get("bag_rest_decay", 0.0)) or float(self.cfg["bag_decay"])
        self.bag_w = (float(self.cfg["bag_decay"]) if world else rest_) * self.bag_w; self.bag_o = rest_ * self.bag_o

    def _consolidate_own(self, strength):
        """the body's last utterance (its own symbols in the stream, the last run between pauses of three or more ticks) written
        into the store as an episode: keys as the world's would have been (the faded world context, then the utterance itself),
        the chain linked, the first symbol a start; nothing when the run is shorter than three symbols"""
        own = [i for (i, w) in self.stream if w == 1]
        runs = []; cur = []; gap = 0
        for i in own:
            if i == self.sil:
                gap += 1
                if gap >= 3 and cur:
                    runs.append(cur); cur = []
            else:
                gap = 0; cur.append(int(i))
        if cur:
            runs.append(cur)
        run = runs[-1] if runs else []
        run = run[-int(self.cfg.get("own_store_len", 12)):]           # the last symbols: the word rewarded and what led to it
        if len(run) < 3:
            return 0
        m = self.m; d_ = float(self.cfg["bag_decay"]); chain = int(self.cfg.get("store_chain", 0))
        with torch.no_grad():
            bag = self.bag_w.clone(); prev = -1; n = 0
            self.store.episode += 1                                        # its own utterance: an episode of its own
            for i in run:
                if i in self.bans or i == self.sil:
                    continue
                ex = m.E.weight[i]
                if bag.norm() > 1e-6 and self.store.write(bag, ex, float(strength), 1):
                    if getattr(self, "_fboosts", None):
                        self._boosts_remap(self.store.last_remap)     # step R7d: the frames' pending boosts follow the slots (body/core/frames.py)
                    j = self.store.last_idx; n += 1
                    if n == 2:
                        self.store.mark_start(bag, ex)         # the start mark on the second symbol's slot, as the world's onsets are marked
                    if chain and prev >= 0:
                        self.store.link(prev, j, tag=self.store.episode)
                    prev = j
                bag = d_ * m.shift(bag) + ex
        self._own_stored_n = getattr(self, "_own_stored_n", 0) + n
        return n
