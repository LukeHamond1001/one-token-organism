"""the second body alive (BODY_SPEC.md §1, §3, §4): ticks, feelings, the one reward system,
the waking lesson, sleep by fatigue, the night. The caregiver reaches it only through THE DIARY
protocol in serve.py; nothing here reads the caregiver's mind or edits the body's words.

HOW TO READ THIS FILE. `PHYSIOLOGY` holds every constant, grouped by organ; the served body's effective set is its save plus
ops/BASE_FLAGS.txt (ops/served_cfg.py prints it), and the history of every value is in BODY_SPEC.md's appendix and ITERATIONS.md,
not here. `Life.__init__` wires the organs (body/model.py: `Organs` the learned parts, `Store` the hippocampus) and the state a tick
touches. `tick()` is one moment of the body's clock in eight phases, each a method in this order: `_sense` (the world's symbol or
its quiet, the offset by the count, the face felt as reward), `_hear` (the symbol enters the stream, the store writes what surprised
it, the offset by the settle law), `_learn_values` (every critic learns; the fast band's error is dopamine), `_own_face`, `_choose`
(whether to speak, then what: the readout at the mood's sharpness, the actor's chunk or plan), `_act` (its own symbol or rest enters
the stream, the credits), `_feel_and_learn` (the feelings, the gate's lesson, the waking cortex lesson), `_bookkeep` (the page, the
record, the sleep switch).
`night()` runs NREM (the dreams from `dreams()`, batched, the cortex learning with the recall off as its input), REM
(`_rem_imagine`, `_rem_rollout`), the value replay, the store's fade, the save. `type_text`, `set_face`, `state` and `insides`
are the page's endpoints; `save`, `load` and `birth` are the body on disk."""
import collections
import math
import os
import time

import torch
import torch.nn.functional as F

from .model import Organs, Store

PHYSIOLOGY = dict(
    # The disclosed constants, grouped by organ. The served body's effective set is its save plus ops/BASE_FLAGS.txt (ops/served_cfg.py
    # prints it); the ledger (ITERATIONS.md) records every change and the spec (BODY_SPEC.md, the appendix on the constants' history)
    # keeps the derivations that used to sit here. A constant at 0 or "off" is an instrument kept in the code, measured and not adopted.
    # --- the body's clock and its feelings (in ticks; the body lives on its own clock, not the serve's) ---
    symbol_cost=0.12,   # fatigue per symbol said
    fatigue_half_life=240,
    stress_half_life=240,
    mood_half_life=1200,
    wake_ticks=12000,   # sleep pressure crosses the switch here: the body sleeps by itself
    burst=0.5,
    mood_gain=0.25,
    stress_gain=0.5,
    # --- reward and dopamine: the face becomes the fast band's error; the ladder learns at every band ---
    dopamine_band=2,   # the band whose TD error is dopamine: clock 16, discount 0.9375 a tick, a four-second horizon
    elig_ticks=12,
    elig_decay=0.8,
    world_r=0.0,   # each symbol the world types felt as reward beside the face
    world_mask=0,   # corollary discharge: the world's word is not felt as reward on a tick after the mouth acted
    cost_in_reward=0,
    reward_gain=0.0,
    diff_horizon=1024,   # bands with clocks at or above this learn average-reward TD (no discount, the reward rate as baseline)
    value_lr=1e-3,
    band_lr=1e-5,
    v_buf=32,
    # --- the hippocampal store: what is written, how it is read, how it is forgotten ---
    store_cap=8192,   # the slot count; over it the weakest gives way
    store_fade=0.9,
    store_floor_rel=0.1,   # a slot below this fraction of the store's mean strength is forgotten at night
    store_floor_abs=0.0,   # an absolute forgetting floor; 0 = the relative floor (falsified on the served body, item 2)
    store_temp=0.02,
    store_links=4,
    store_sat=0,
    store_chain=0,
    write_floor=1e-6,   # the key's norm must exceed this for a memory to be written (a constant, not a threshold on a fading quantity)
    read_follow=0.0,   # the recall carries the episode: the next slot of the followed utterance is easier to recall (a gain)
    read_tire=0.0,   # a slot that wins the read loses this much availability (synaptic depression), recovering by read_recover a tick
    read_recover=0.97,
    recall_end=0,
    episode_chain=0,
    heard_decay=0.999,
    # --- the key of a memory and the working context (the fast bag of the last symbols; the slow context of the last utterance) ---
    key_form="bag",
    key_scale=2.5,
    key_ctx=0.0,   # the previous utterance's order-free bag in the key and the query, at this weight
    ctx_decay=0.95,
    ctx_form="bag",
    bag_decay=0.8,
    bag_rest_decay=0.0,   # the world's context fades at this rate per quiet tick (0 = at bag_decay)
    bag_own_fade=0,   # 0/1/2: whether the body's own symbols fade the world's context (2 = in the query alone)
    bag_own_weight=1.0,
    # --- the own song: the body's own speech as a target and in memory ---
    own_gain=0.5,
    own_target_decay=0.0,
    own_target_form="world",   # "recall" = the own-speech target is the hippocampus's continuation of what it said; "world" = the world's next symbol
    own_target_conf=0.3,
    own_store=0,   # a rewarded own utterance written into the store as an episode (off: the corollary discharge keeps babble out)
    own_store_r=1.0,
    own_store_len=12,
    own_store_gain=0.3,
    own_store_gap=40,
    # --- the waking lesson of the cortex ---
    wake_every=24,
    wake_window=32,
    live_lr=1e-5,
    wake_base=1.0,
    wake_dopa=0.0,
    wake_novel=0.0,
    sigreg=0.0,
    # --- the night: NREM on the dreams, then REM; the utterance memory it replays ---
    night_lr=1e-4,
    night_warm=0,
    night_beta2=0.999,   # the night's Adam second-moment horizon
    night_rounds=24,
    night_batch=0,
    night_starts=48,
    night_starts_max=192,
    night_load=0.0,
    night_keep_bands=0,   # the slow bands are not zeroed at night
    night_ticks=0,
    utt_cap=4096,   # the utterance memory replayed at night, whole utterances
    dream_source="store",   # "utterances" = the night dreams the utterances heard whole; "store" = pattern completion from the store
    dream_who=0,
    dream_tag=0,
    dream_pair=0,
    dream_gap=1,
    dream_old_share=0.0,
    dream_max=24,
    dream_floor_rel=0.5,
    dream_adapt=0.2,
    dream_recover=0.97,
    dream_exhaust=0.1,
    rem_steps=8,
    rem_dreams=8,
    rem_rounds=6,
    rem_temp=0.0,
    rem_form="forecast",
    rem_weight=1.0,
    rem_world_temp=0,
    # --- the event's end (the offset after the world's quiet) and the marks ---
    offset_ticks=8,
    offset_form="settle",   # "settle" = the offset fires when the surprise settles (offset_fast/offset_slow, offset_settle); "count" = after offset_ticks
    offset_settle=0.5,
    offset_fast=4,
    offset_slow=64,
    end_rest=0,
    end_symbol="rest",   # "rest" = the turn ends at the first rest after a symbol; "eot" = the chat token marks it
    rest_token="<pad>",
    end_token="<eot_human>",
    display_token="\n",
    # --- the gate: whether to speak, learned from the same reward ---
    gate_lr=0.05,
    birth_act=0.25,
    gate_habit=0.9,
    gate_fatigue=10.0,
    gate_int=0.0,
    gate_int_form="value",
    gate_tonic=0.25,
    gate_tonic_rate=0.0,   # the tonic drive follows the felt-reward trace at clock gate_tonic_clock
    gate_tonic_clock=4,
    gate_vigor=1.0,
    gate_every=24,
    gate_baseline=0.9,
    gate_floor=0.05,   # spontaneous activity never stops: p(act) = floor + (1 - floor) sigmoid(z)
    gate_listen=0.0,   # THE LISTENING REFLEX (item 41): while the world's utterance is open (before the offset) the floor is scaled by (1 - gate_listen) and a running word is cut; 0 = off
    gate_salience=0.0,   # the forecast's certainty as an input of the gate; 0 = off
    gate_slow_band=5,
    gate_slow_w=0.0,
    gate_slow_lr=0.0,
    gate_level_band=5,
    gate_level_w=0.0,
    gate_center=0,   # the gate reads its inputs relative to their running mean
    gate_center_tau=1024,
    gate_center_keep=0,
    gate_ear=0,   # two more inputs to the gate: the world's symbol this tick and its own act last tick
    gate_opt="sgd",
    gate_adam_lr=1e-3,
    # --- the mouth's decisiveness: the readout's sharpness, and exploration ---
    read_sharp=25.0,
    sharp_base=25.0,
    sharp_gain=25.0,
    sharp_form="fixed",
    sharp_rate=0.05,
    sharp_min=8.0,
    sharp_max=100.0,
    sharp_conf=0.0,   # the choice's sharpness x (1 + sharp_conf x the forecast's norm); 0 = off (item 13: falsified)
    explore_gain=0.0,
    explore_tau=64,
    explore_choice=0.0,   # novelty widens the planner's choice among its candidates
    # --- the striatum, working memory and the fast critic ---
    fast_input="band",   # the fast critic's input: "striatum" = a delay line of the stream's last events through a born expansion; "band" = the dopamine band's state
    stri_k=8,
    stri_m=1024,
    stri_quiet=0,
    fast_rls=0,
    fast_rls_forget=36000,
    fast_rls_prior=3.0,
    fast_rls_every=64,
    wm=0,   # working memory: latches the line's expansion at a dopamine burst, clears at a reward or after wm_max ticks
    wm_burst=0.5,
    wm_max=512,
    # --- the ventral critic (the slow prospect) ---
    vcrit_w=0.0,
    vcrit_gamma=1.0 - 1.0 / 1024,
    vcrit_ceiling="fixed",   # "fixed" = vcrit_w x reliability; "earned" = the reliability itself
    vcrit_lambda=0.0,
    vcrit_lr=0.0,
    vcrit_tau=0.0,
    vcrit_diff=0,
    vcrit_bands="5,6,7",   # the ventral critic reads these bands ("-" = none)
    vcrit_traces=0,
    vcrit_clock=0,   # sleep pressure over the wake threshold as a critic input
    vcrit_center=1,
    vcrit_auto=0,
    vcrit_forget=0,
    vcrit_rls=0,   # the ventral head learns by recursive least-squares TD(lambda) from accumulated evidence
    vcrit_rls_delta=100.0,
    vcrit_rls_every=64,
    vcrit_norm_tau=0,
    vcrit_rls_prior=0.0,
    vcrit_norm_wake=0,
    # --- the actor and the planner: what to say at a word's start ---
    actor=0,
    actor_lr=0.02,
    actor_beta=1.0,
    actor_forget=36000,
    actor_form="add",   # "chunk" = one act per word, the letters inside not choices; "plan", "select", "add" the earlier forms
    actor_margin=4.0,
    chunk_max=12,   # a word runs as a motor program for at most this many symbols
    actor_voice="off",
    actor_horizon=16,
    actor_tau=36000,
    actor_input="striatum",
    actor_wmax=3.0,
    actor_temp=1.0,
    chooser_k=4,
    plan_h=2,
    plan_beta=4.0,
    plan_k=4,
    plan_boundary=0,   # 0 = the planner acts at the cortex's doubt; 1 = at the space (the old rule)
    # --- the face organ: its own face, learned from yours ---
    face_form="read",   # the face organ: "foresee" = it foresees the face to come; "read" = a readout of the felt face
    face_tau=36000,
    face_ridge=0.1,
    face_every=64,
    face_input="cortex",
    face_lr=1e-3,
)


class Life:
    def __init__(self, organs, tok, cfg=None, device="cpu", seed=0, save_path=None):
        unknown = sorted(k_ for k_ in (cfg or {}) if k_ not in PHYSIOLOGY)
        if unknown:
            print("physiology: unknown keys (ignored):", unknown, flush=True)     # review 2026-09-06: a typo was a silent no-op for 21 days
        self.m = organs.to(device); self.m.eval()
        self.tok = tok; self.dev = device
        self.cfg = dict(PHYSIOLOGY); self.cfg.update(cfg or {})
        vb = str(self.cfg.get("vcrit_bands", "") or "").strip()
        if vb:
            with torch.no_grad():
                organs.vcrit_mask.zero_()
                for b_ in vb.split(","):
                    if b_.strip() and b_.strip() != "-":
                        organs.vcrit_mask[int(b_)] = 1.0
        self.m.vcrit_center = bool(int(self.cfg.get("vcrit_center", 1)))
        self.m.vcrit_traces = bool(int(self.cfg.get("vcrit_traces", 0))); self.m.vcrit_clock = bool(int(self.cfg.get("vcrit_clock", 0)))
        with torch.no_grad():
            nb_ = organs.vcrit_mask.numel(); d_ = (organs.vcrit.weight.shape[1] - nb_ - 1) // nb_
            full = torch.cat([organs.vcrit_mask.cpu()[:, None].expand(nb_, d_).reshape(-1), torch.ones(nb_) if self.m.vcrit_traces else torch.zeros(nb_),
                              torch.ones(1) if self.m.vcrit_clock else torch.zeros(1)])
            self._vc_idx = full.nonzero().squeeze(1)
            if int(self.cfg.get("vcrit_rls", 0)):
                k = int(self._vc_idx.numel()) + 1
                pri = float(self.cfg.get("vcrit_rls_prior", 0.0) or 0.0); vf0 = float(self.cfg.get("vcrit_forget", 0) or 0) or 12000.0
                self._vc_delta = pri * vf0 if pri > 0 else float(self.cfg.get("vcrit_rls_delta", 100.0))
                norm_on = bool(int(self.cfg.get("vcrit_norm_tau", 0)))
                if organs.vc_A.numel() != k * k:
                    organs.vc_A = torch.eye(k, dtype=torch.float64, device="cpu") * (0.0 if norm_on else self._vc_delta)
                    organs.vc_b = torch.zeros(k, dtype=torch.float64, device="cpu")
                if norm_on and organs.vc_mu.numel() != k - 1:
                    # born with a prior on the input's scale (one unit, weighing tau/32 ticks), so the first hours' few samples
                    # do not inflate the standardized input before the statistics have formed
                    organs.vc_mu = torch.zeros(k - 1, dtype=torch.float64, device="cpu")
                    organs.vc_var = torch.ones(k - 1, dtype=torch.float64, device="cpu")
                    # THE FAST START (2026-09-06): the scale set by the first samples (n from 1), not born at one unit weighing tau/32. With the evidence in raw coordinates the statistics
                    # shape only the prior, so a fast start risks nothing; the slow one left the traces' variance at the residue of
                    # its birth value (0.0099 against a true 0.0001) after six days, a prior a hundred times too strong, the head crushed.
                    organs.vc_n = torch.full((), 1.0, dtype=torch.float64, device="cpu")   # the first sample sets the scale
        self._vc_e = None
        if int(self.cfg.get("fast_rls", 0)):
            with torch.no_grad():
                fb = int(self.cfg["dopamine_band"])
                if str(self.cfg.get("fast_input", "band")) == "striatum":
                    k_, m_ = int(self.cfg["stri_k"]), int(self.cfg["stri_m"])
                    wm_ = int(self.cfg.get("wm", 0))
                    if (organs.stri_W.numel() == 0 or organs.stri_line.numel() != k_ or organs.stri_W.shape[1] != m_ or organs.stri_W.shape[0] != k_ * (2 * organs.vocab + 3)
                            or organs.vfast.weight.shape[1] != m_ * (1 + wm_)):
                        organs.striatum_init(k_, m_, seed=seed, wm=wm_)        # born (or re-born at a new size)
                    kf = m_ * (1 + wm_) + 1
                else:
                    kf = int(organs.value[fb].weight.shape[1]) + 1
                dev_ = "cpu"
                if organs.vf_A.shape != (kf, kf):
                    organs.vf_A = torch.zeros(kf, kf, dtype=torch.float64, device=dev_); organs.vf_b = torch.zeros(kf, dtype=torch.float64, device=dev_)
                    organs.vf_mu = torch.zeros(kf - 1, dtype=torch.float64, device=dev_); organs.vf_var = torch.ones(kf - 1, dtype=torch.float64, device=dev_)
                    organs.vf_n = torch.full((), 1.0, dtype=torch.float64, device=dev_)
            self._vf_delta = float(self.cfg["fast_rls_prior"]) * float(self.cfg["fast_rls_forget"])
        self._vf_e = None
        self.m.read_sharp = float(self.cfg["read_sharp"])
        # THE COROLLARY DISCHARGE (own_gain, 2026-09-12): the cortex hears its own sound at this fraction (measured in cortex at a third
        # to a half); the "ttle tin" loop is the cortex forecasting from its own babble in the window, so the fraction is the lever
        self.m.own_gain = float(self.cfg.get("own_gain", 0.5))
        # THE BODY'S OWN SYMBOLS, DECLARED (anatomy, 2026-09-08): its rest, the world's turn-end, and a display symbol the world
        # never types, named in the physiology by the body that is born, never found by a fixed string in the code.
        self.sil = tok.token_to_id(str(self.cfg.get("rest_token", "<pad>")))
        self.m.sil_id = self.sil                          # the cortex's inputs know its rest
        self.nl = tok.token_to_id(str(self.cfg.get("display_token", "\n")))
        self.space_id = tok.token_to_id(" ")                  # the word boundary the planning actor decides at
        self.eot = tok.token_to_id(str(self.cfg.get("end_token", "<eot_human>")))
        self.end_id = self.sil if str(self.cfg.get("end_symbol", "eot")) == "rest" else self.eot   # what the offset teaches the cortex to expect         # the world's turn ended: the offset (§2), never the mouth's
        # THE RESERVED SYMBOLS (anatomy, declared, 2026-09-08): the lexicon's control tokens, every `<...>` the tokenizer
        # defines (the world's turn-end, the old face tokens), except the rest; and this body's newline, a display symbol the
        # world never types. Neither the mouth nor the typing admits them. Declared from the tokenizer, never counted.
        _vocab = tok.get_vocab(); _specials = sorted(i for s_, i in _vocab.items() if s_.startswith("<") and s_.endswith(">"))
        self.reserved = [i for i in _specials if i != self.sil] + ([self.nl] if self.nl is not None else [])
        self.bans = list(self.reserved)
        self._last_world = -10 ** 9; self._offset_done = True; self._last_write = None; self._start_pending = False
        self._prev_slot = -1                                   # the slot of the world's last symbol in this utterance (the sequence's link)
        self._follow = None                                    # the episode the waking recall is in: (slot, tag), or none (read_follow)
        # THE UTTERANCES HEARD (dream_source "utterances"; 2026-09-13, the twenty-fifth defect): the world's utterances as heard, whole,
        # between pauses, each with a strength that fades by night; the night dreams them whole. The slot store's chains gave the
        # night ten-symbol fragments that began mid-word, and seven nights of them read -0.01 on the held-out lines; one night of the
        # parent's last thousand lines, whole, read +0.13. A hippocampus keeps an episode as the sequence it was (CA3's chain, the
        # time cells); this memory keeps it plainly. The recall at the mouth still reads the slot store.
        self.utts = []; self.utt_S = []; self._utt_cur = []
        # THE EXCHANGE REPLAYED (dream_pair; 2026-09-14, the twenty-sixth defect): each utterance carries its serial number, so the night
        # can dream it together with the one that followed it (the question, the pause, the answer): the sequence replay of sleep,
        # compressed. Utterances dreamt one by one from rest taught the cortex no line from the line before it: the thirty facts were
        # completed from their prefixes (24 of 26) and answered from their questions in conversation once a day (5%).
        self.utt_N = []; self._utt_serial = 0
        self._q_prev = None; self._read_prev = None            # the cortex key: the query of the tick before, the read of the tick before
        # PATTERN SEPARATION (2026-09-14): the stream's states all point one way (mean pairwise cosine 0.986 on the served body; 0.002
        # once the running mean is taken out), so a key that is the raw state matches every other key and the recall is a blur (the
        # first cortex-keyed copy answered 1 of 30). The dentate gyrus decorrelates the cortical pattern before CA3 stores it; here
        # the running mean of the state is subtracted before the key is made. A plain average for the first two thousand ticks, then
        # a slow exponential one (0.9995 a tick, about an hour), saved with the body.
        self._c_mu = torch.zeros(self.m.d, device=device); self._c_n = 0
        # THE STORE'S CAPACITY (store_cap; 2026-09-15, the twenty-ninth defect): at 8192 slots a day's twelve hundred new memories evicted the
        # weakest twelve hundred, and the weakest are the memories heard once (strength 0.55 against a mean of 0.85 for the repeated):
        # the facts told on one day were gone by the next dusk (the prefixes 20 -> 10 of 26 within day 228). A hippocampus holds weeks
        # of episodes; the nightly fade, not the cap, is meant to do the forgetting (a once-heard memory falls under the fade's floor
        # in about nineteen nights, twenty-four thousand writes at this day's rate). The capacity is a constant of the organ.
        self.store = Store(self.m.d, cap=int(self.cfg.get("store_cap", 8192)), temp=float(self.cfg["store_temp"]), device=device, links=int(self.cfg.get("store_links", 4)))
        self.store.saturate = bool(int(self.cfg.get("store_sat", 0)))   # repetition suppression (store_sat)
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.save_path = save_path
        nb, d, W = len(self.m.clocks), self.m.d, self.m.window
        # working state
        self.bands = torch.zeros(nb, d, device=device)
        # THE CONTEXT: two bags, the world's (decaying per world symbol) and its own (per own symbol),
        # summed into the memory key; a pause moves neither. Decayed per tick, the babble between the
        # world's symbols shifted the world's weights in every key (collision test at own weight 0.6).
        self.bag_w = torch.zeros(d, device=device)
        # THE SLOW CONTEXT (key_ctx, the twenty-eighth defect; 2026-09-14): the fast bag holds the last five symbols, so every "the sun "
        # of four facts wrote into one slot and the episode's chain broke there (a common slot's sixteen links hold only the newest
        # utterances' tags). A second, order-free context of the last twenty or so WORLD symbols (ctx_decay a symbol, not a tick: it
        # holds through the pause) enters the key and the query at key_ctx times its weight: "the sun " after "what is hot?" and after
        # "what makes us warm?" are then different memories, each with its own chain, found by the question that stands in the
        # context. The temporal context of Howard and Kahana; the dentate's separation by context. Own symbols enter the query's
        # context as the world's would have (the efference copy), never the key's (corollary discharge).
        self.ctx_cur = torch.zeros(d, device=device); self.ctx_prev = torch.zeros(d, device=device); self._utt_open = False
        self.bag_o = torch.zeros(d, device=device); self.n_own = 0
        self.win = collections.deque(maxlen=W)            # per tick: dict(x the world's, xo its own, face, bundle, read, r)
        self.pred_prev = None                              # the forecast made at the last step (surprise)
        self.v_prev = None                                 # V_b of the previous tick's states
        self.v_buf = {b: collections.deque(maxlen=int(self.cfg["v_buf"])) for b in range(nb)}
        # THE REWARD RATE, one estimate: a running mean of the reward at the differential horizon (tonic
        # dopamine), the baseline of every differential band. Estimated per band at the band's own clock,
        # the slowest band's baseline (one part in 16384 a tick) could not track the rate within a day and
        # its undiscounted value integrated raw reward (run 27, day 6: 643 against a return of 131).
        self.rbar = 0.0
        # the reliability gain's buffers: the head's value and the reward per tick, the running moments of (value, return)
        self._vbuf_v = collections.deque(maxlen=4096); self._vbuf_r = collections.deque(maxlen=4096)
        self._vrel = [0.0] * 6; self._vrel_gain = 0.0; self._vrel_corr = 0.0
        self.sharp_cal = float(self.cfg["sharp_base"])       # THE CALIBRATED SHARPNESS (sharp_form "calibrated"): the readout's base, set by its own hits
        self._fc_prev = None                                  # the forecast made at the last step, unnormalized (its norm is its certainty)
        self._frel = [0.0] * 6; self._frel_gain = 0.0; self._frel_corr = 0.0   # THE FACE ORGAN'S RELIABILITY (face_form "foresee", §5c): the slope of the felt reward on its foresight
        self._C1_prev = None; self._fpred_now = 0.0
        self._arel = [0.0] * 6; self._arel_gain = 0.0; self._arel_corr = 0.0; self._act_pending = collections.deque()   # the actor's reliability
        self._store_after_night = None                    # the store's size when the last night ended: the day's new memories are counted from it
        self._a_bias_now = None; self._act_agree = collections.deque(maxlen=int(self.cfg["wake_ticks"]) + 16)
        # THE FACE ORGAN'S INPUT (face_input, the review of 2026-09-10): "cortex" reads the stream, which carries the coming reward at 0.11;
        # "striatum" reads what the fast critic reads, the delay line's expansion (and the working-memory slot beside it), which carries it at 0.47
        _fi = str(self.cfg.get("face_input", "cortex"))
        self._fh_n = int(self.m.stri_in().numel()) if (_fi == "striatum" and self.m.stri_W.numel() > 0) else int(self.m.d)
        self._fh_A = torch.zeros(self._fh_n + 1, self._fh_n + 1, dtype=torch.float64); self._fh_b = torch.zeros(self._fh_n + 1, dtype=torch.float64)   # the foreseeing face organ's evidence (least squares, as the fast critic's)
        self._fh_w = torch.zeros(self._fh_n + 1, dtype=torch.float64); self._fin_prev = None
        self._ring_torn = collections.deque(maxlen=int(self.cfg["wake_ticks"]) + 16); self._ring_ent = collections.deque(maxlen=int(self.cfg["wake_ticks"]) + 16)   # the readings of 2026-09-10: torn ticks, the readout's entropy
        self._torn_now = False; self._ent_now = 0.0
        _wk = int(self.cfg["wake_ticks"]) + 16                # the day's ring: the fast value before each tick and the felt reward at it (the anticipation reading)
        self._ring_vf = collections.deque(maxlen=_wk); self._ring_r = collections.deque(maxlen=_wk)
        self._differential = [int(c) >= int(self.cfg["diff_horizon"]) for c in self.m.clocks]
        with torch.no_grad():
            self.m.diff[:] = torch.tensor(self._differential, dtype=torch.bool, device=self.m.diff.device)
        # feelings and clocks
        self.fatigue = 0.0; self.stress = 0.0; self.mood = 0.0
        self._t_feel = time.time()
        self.sleep_pressure = 0; self.ticks = 0; self.nights = 0; self.day_n = 0
        self.asleep = False; self.last_night = None
        # the face
        self.face_now = 0.0; self.level = 0; self.face_prev = 0.0
        # the mouth's gate
        self.gate_buf = collections.deque(maxlen=96)
        self.sym_freq = {}; self.perf = {}                      # habituation tables of the intrinsic credit (per symbol)
        self.heard = torch.zeros(self.m.vocab, device=device)   # the symbols the world has said (a slow tally)
        self._gate_last = None; self._wake_last = None
        self.n_bursts = 0
        # the page and the two hands
        self.queue = collections.deque()
        self.queue_who = collections.deque()             # who typed each queued symbol ("parent", "you"): the page shows it, nothing inside reads it
        self.page = []; self.page_base = 0
        self.stream = collections.deque(maxlen=96)       # (id, who)
        self.last = {}
        self.credit = collections.deque(maxlen=64)
        # optimizers: the day's (the cortex and its forecasts), the striatum's (the gate), the critic's
        self.opt_day = torch.optim.Adam(self.m.parameters(), lr=float(self.cfg["live_lr"]))
        if int(self.cfg.get("gate_ear", 0)) and self.m.mouth_gate.in_features == self.m.d + 5:
            self.m.widen_gate(2)                                # THE EAR: two inputs, born at zero
        if str(self.cfg.get("gate_opt", "sgd")) == "adam":
            self.opt_gate = torch.optim.Adam(self.m.mouth_gate.parameters(), lr=float(self.cfg.get("gate_adam_lr", 1e-3)))
        else:
            self.opt_gate = torch.optim.SGD(self.m.mouth_gate.parameters(), lr=float(self.cfg["gate_lr"]))
        # the critic's optimizer: the value heads and the Go/NoGo gates. The bands' input maps are fixed
        # (born): trained by the critic's own bootstrapped error they are the deadly triad, and at any
        # rate they ran away (1e-3: saturated by day 6, run 26; 1e-5: saturated by day 15, run 28) while
        # learning nothing this world offers to learn (the stream carries no reward at their horizons).
        # Linear heads on fixed features under on-policy TD converge (Tsitsiklis and Van Roy).
        vlr = float(self.cfg["value_lr"]); vclr = float(self.cfg.get("vcrit_lr", 0.0)) or vlr
        self.opt_value = torch.optim.Adam([{"params": list(self.m.value.parameters()) + list(self.m.band_gate.parameters()), "lr": vlr},
                                           {"params": list(self.m.vcrit.parameters()), "lr": vclr}], lr=vlr)
        for p_ in self.m.band_in.parameters():
            p_.requires_grad_(False)
        self.opt_face = torch.optim.Adam(self.m.face_head.parameters(), lr=float(self.cfg["face_lr"]))

    # ---------------- feelings ----------------
    def _decay_feelings(self):
        """feelings recover on the body's own clock (per tick), so its physiology does not change
        with the speed the serve happens to run at"""
        self.fatigue *= 0.5 ** (1.0 / float(self.cfg["fatigue_half_life"]))
        self.stress *= 0.5 ** (1.0 / float(self.cfg["stress_half_life"]))
        self.mood *= 0.5 ** (1.0 / float(self.cfg["mood_half_life"]))

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
                self.store.write(key_, ex, surp * (1.0 + abs(dopamine)), who)   # the world's quiet is not a memory
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
            face = torch.tensor([self.face_now / 6.0, (self.face_now - self.face_prev) / 6.0], device=self.dev)
            # THE TICK'S POSITION. The world's symbol opens it. The world's quiet opens nothing yet: the
            # forecast the mouth reads is then the one made at the last filled position, the one
            # holding its own last symbol, which is trained to foresee what follows that symbol. (Read
            # at a freshly appended rest, the forecast was of what follows a pause, and alone the mouth
            # looped on the cue's last word: runs 20 to 22.) Its own half then fills the open position
            # or, the world quiet, opens one of its own; a tick with nothing sounded leaves a rest.
            entry = {"face": face, "bundle": self.bands.clone(), "read": read.clone(), "r": float(r)}
            if who == 0:
                if x != self.sil or not self.win:
                    self.win.append({"x": int(x), "xo": self.sil, **entry}); self._pos_open = True
                else:
                    self._pos_open = False
            else:
                if getattr(self, "_pos_open", False):
                    self.win[-1]["xo"] = int(x)                # its own sound joins the world's time step
                else:
                    self.win.append({"x": self.sil, "xo": int(x), **entry})
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

    def query_from(self, C, learn=False):
        """the recall's query under the cortex key: the stream's state, its running mean taken out (pattern separation), as a unit
        direction at the norm a full context's bag would have (the query's norm is the recall's inverse temperature; key_scale, a
        disclosed constant near the bag's own norm). With learn, the running mean takes this state in first."""
        c = C.detach().float()
        if learn:
            self._c_n += 1; a = max(1.0 / self._c_n, 1.0 - 0.9995)
            self._c_mu = self._c_mu + a * (c - self._c_mu)
        return F.normalize(c - self._c_mu, dim=0) * float(self.cfg.get("key_scale", 2.5))

    def _tire(self, win_, rt_):
        if rt_ > 0.0 and self.store.A.numel() == self.store.n():
            if win_ >= 0:
                self.store.A[win_] *= (1.0 - rt_)                            # the winner tires
            self.store.A = 1.0 - float(self.cfg.get("read_recover", 0.97)) * (1.0 - self.store.A)   # all recover toward rest

    def _chooser_credit(self, nxt, gamma):
        """the chooser's eligibility: the log-softmax's gradient over the candidates for the one said, on the cortex's state; decays by
        dopamine's discount; nothing new when the moment was not torn"""
        m = self.m
        e = getattr(self, "_e_chooser", None)
        e = (gamma * e) if e is not None else torch.zeros(m.vocab, m.d, device=self.dev)
        cands = getattr(self, "_cands_now", None)
        if cands is not None and int(nxt) in cands:
            pa = self._pa_now; za = self._za_now
            for j, c in enumerate(cands):
                e[c] += ((1.0 if c == int(nxt) else 0.0) - float(pa[j])) * za
        self._e_chooser = e

    def _chooser_learn(self, delta):
        """dopamine times the eligibility on the chooser's head; every row bounded (actor_wmax) so no candidate can saturate the vote"""
        e = getattr(self, "_e_chooser", None)
        if e is None or abs(float(delta)) < 1e-9:
            return
        with torch.no_grad():
            W = self.m.chooser.weight
            W.add_(float(self.cfg.get("actor_lr", 0.02)) * float(delta) * e)
            n = W.norm(dim=1, keepdim=True); wmax = float(self.cfg.get("actor_wmax", 3.0))
            W.mul_(torch.clamp(wmax / (n + 1e-9), max=1.0))

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

    def _offset(self):
        """THE OFFSET (§2): the world's quiet after its utterance, once per pause. The last world position is
        marked ended, so the waking lesson's target there is the turn-end and not the next line's first letter;
        a dream ends where the cortex, so taught, expects the quiet. Nothing enters the stream, the bags and the
        mouth's context stand, and the store keeps only what the world said next: written there too, the quiet
        after a cue blended with the answer's memory and the mouth read junk (runs 45 and 46, day 1)."""
        if int(self.cfg.get("wm", 0)) and getattr(self.m, "stri_wm", 0):
            with torch.no_grad():                            # WORKING MEMORY latches at the world's utterance end (a salience event)
                self.m.wm_latch(self.m.striatum_read())
        for w in reversed(self.win):
            if w["x"] != self.sil:
                w["end"] = True; break
        if self._last_write is not None and not self.cfg.get("store_off"):
            self.store.mark_boundary(*self._last_write)                # the memory of the last symbol carries the boundary
        self._prev_slot = -1                                           # the utterance ended: the next symbol begins a new chain
        self.note_offset()                                             # and the slow context's utterance closes with it
        self._follow = None                                            # and the recall's episode is let go
        if len(self._utt_cur) >= 2:
            # THE REWARD'S TAG ON THE LINE BEFORE (reward_gain; 2026-09-15): the smiles' dopamine felt since the last utterance was kept
            # raises that utterance's strength, so the night replays the rewarded exchanges more (dopamine tags what preceded it)
            rg = float(self.cfg.get("reward_gain", 0.0))
            if rg > 0.0 and self.utt_S:
                self.utt_S[-1] = float(self.utt_S[-1]) + rg * max(0.0, float(getattr(self, "_dopa_since_utt", 0.0)))
            self._dopa_since_utt = 0.0
            self._utt_serial += 1
            self.utts.append(list(self._utt_cur)); self.utt_S.append(1.0); self.utt_N.append(self._utt_serial)   # the utterance kept whole, at full strength, in its turn
            cap = int(self.cfg.get("utt_cap", 4096))
            if len(self.utts) > cap:                                   # the weakest (the oldest, faded) gives way
                i = min(range(len(self.utt_S)), key=lambda k: self.utt_S[k]); del self.utts[i]; del self.utt_S[i]; del self.utt_N[i]
        self._utt_cur = []

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

    def _window_tensors(self, win=None):
        win = list(self.win if win is None else win)
        xs = torch.tensor([w["x"] for w in win], device=self.dev)
        whos = torch.tensor([w["xo"] for w in win], device=self.dev)   # its own symbols, one per tick
        faces = torch.stack([w["face"] for w in win])
        bundles = torch.stack([w["bundle"] for w in win])
        reads = torch.stack([w["read"] for w in win])
        return xs, whos, faces, bundles, reads

    def _stream_now(self):
        xs, whos, faces, bundles, reads = self._window_tensors()
        u = self.m.inputs(xs, whos, faces, bundles, reads)
        return self.m.stream(u)[-1]

    # ---------------- the tick ----------------
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

    def fast_value(self):
        """the fast critic's value now (the striatal head on the delay line, or the dopamine band's head on its state)"""
        with torch.no_grad():
            if str(self.cfg.get("fast_input", "band")) == "striatum" and self.m.stri_W.numel() > 0:
                return float(self.m.fast_value(self.m.stri_in()))
            return float(self.m.values(self.bands)[int(self.cfg["dopamine_band"])])

    def tick(self):
        """one moment of the body's clock, in eight phases (each a method below, in this order)"""
        self._ring_vf.append(self.fast_value())            # the fast critic's value before this tick (the anticipation reading; the supervisor's, never the body's)
        self._decay_feelings()
        u, who, felt, r, off, settle_form, first_after_pause = self._sense()
        C1, pred1, surp1, conf1, stri = self._hear(u, r, felt, off, settle_form, first_after_pause)
        delta, delta_slow, delta_long, vlong, level, gam = self._learn_values(r, felt, stri)
        its_face = self._own_face(C1, r)
        acted, nxt, p_act, p_choice, probs, feat, ent, act_on = self._choose(C1, pred1, u, level, stri)
        int_t = self._act(u, felt, stri, gam, delta, acted, nxt, p_act, p_choice, probs, feat, act_on)
        self._feel_and_learn(delta, delta_slow, delta_long, feat, acted, int_t, p_act)
        self._bookkeep(u, who, nxt, its_face, felt, ent, p_act, delta, level, r, vlong, delta_long, conf1, surp1, probs)

    def _sense(self):
        """the world's symbol (or its quiet) off the queue, the offset by the count, the face felt as reward, the reward's other terms"""
        m = self.m
        u = self.queue.popleft() if self.queue else self.sil
        who = (self.queue_who.popleft() if self.queue_who else "") if u != self.sil else ""
        # THE OFFSET: the world quiet for offset_ticks after its utterance, once per pause, whatever the body is
        # saying meanwhile (with the body's silence required too, a babbling body never let it fire: run 41 held
        # two turn-end memories after six days)
        off = int(self.cfg.get("offset_ticks", 0)); settle_form = str(self.cfg.get("offset_form", "count")) == "settle"
        if u == self.sil and off > 0 and not self._offset_done and not settle_form and self.ticks - self._last_world >= off:
            self._offset(); self._offset_done = True
        first_after_pause = (u != self.sil and self._offset_done)  # the first symbol after a perceived pause begins an utterance
        if u != self.sil:
            self._last_world = self.ticks; self._offset_done = False
        # the face: a change is felt; a held face is silence; easing off is not an event
        lvl = max(-6, min(6, int(self.face_now)))
        felt = 0
        if lvl != self.level:
            if abs(lvl) > abs(self.level) or lvl * self.level < 0:
                felt = lvl
            self.level = lvl
        r = float(max(-2, min(2, felt)))                    # the world's reward: the felt face, clipped like a press
        self._ring_r.append(r)
        if self._act_pending:                               # the actor's reliability: the reward of the ticks after each act, on its vote for that act
            H = int(self.cfg.get("actor_horizon", 16))
            for p_ in self._act_pending:
                p_[2] += r
            while self._act_pending and self.ticks - self._act_pending[0][0] >= H:
                t0_, v_, g_ = self._act_pending.popleft(); self._arel_update(v_, g_)
        if u != self.sil:
            wr = float(self.cfg.get("world_r", 0.0))        # the world's words as reward (0 = off)
            if wr and not (int(self.cfg.get("world_mask", 0)) and getattr(self, "_acted_last", False)):
                r += wr                                     # not heard over its own voice (world_mask)
        if self.cfg.get("cost_in_reward") and getattr(self, "_acted_last", False):
            # THE EFFORT IN THE REWARD: the cost of the last act is felt as the next tick's reward, so both critics
            # predict it and the gate reads their error alone. Added to the act's credit outside the critics (the
            # earlier form, with a tonic drive of 0.25 cancelling it) it was never predicted away and, with the drive
            # gone, held every act at a loss; with both gone the gate saturated at 0.98 (runs 69-72).
            r -= float(self.cfg["symbol_cost"]) * (1.0 + (self.fatigue / float(self.cfg["gate_fatigue"])) ** 2)
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
        C1, pred1, surp1, conf1 = self._step(u, 0, r=r, dopamine=getattr(self, "_dopa", 0.0))
        if settle_form and off > 0:                                          # THE EVENT'S END BY THE LAW: two running averages of the
            st_ = float(getattr(self, "_surp_tick", 0.0))                     # tick's surprise; the world stops, the surprise jumps and
            af_ = 1.0 / max(1.0, float(self.cfg.get("offset_fast", 4))); as_ = 1.0 / max(1.0, float(self.cfg.get("offset_slow", 64)))
            self._surp_fast = (1 - af_) * getattr(self, "_surp_fast", st_) + af_ * st_
            self._surp_slow = (1 - as_) * getattr(self, "_surp_slow", st_) + as_ * st_
            settled_ = self._surp_fast <= float(self.cfg.get("offset_settle", 0.5)) * max(1e-6, self._surp_slow)
            if u == self.sil and not self._offset_done and self.ticks - self._last_world >= 1 and \
                    (settled_ or self.ticks - self._last_world >= off):         # the law, or the senses' own adaptation as the floor
                self._offset(); self._offset_done = True                     # a newborn's flat surprise still ends events by the count
        if u != self.sil and (float(self.cfg.get("explore_gain", 0.0)) > 0 or float(self.cfg.get("explore_choice", 0.0)) > 0):   # arousal follows novelty: a running surprise at the world's symbols
            a_ = 1.0 / max(1.0, float(self.cfg.get("explore_tau", 64)))
            self._surp_run = (1.0 - a_) * getattr(self, "_surp_run", 0.0) + a_ * float(surp1)
        stri = str(self.cfg.get("fast_input", "band")) == "striatum" and int(self.cfg.get("fast_rls", 0))
        if stri:
            if felt:
                m.striatum_push(2, 0 if felt > 0 else 1)          # the felt face is an event of the stream
            if u != self.sil:
                m.striatum_push(0, int(u))
            self._z_now = m.stri_in()
        self._read_world = getattr(self, "_read", None)        # the recall as the world's symbol entered
        return C1, pred1, surp1, conf1, stri

    def _learn_values(self, r, felt, stri):
        """dopamine: every critic learns from the felt reward; the fast band's error is dopamine; the chain closes; the synaptic tag is captured"""
        m = self.m
        # --- dopamine: the fast band's error of the world's reward; the critic learns at every band ---
        with torch.no_grad():
            v_now = m.values(self.bands)
            # THE LEVEL: the slow critic's value of this moment, read by the gate below, scaled by its own
            # running root mean square (divisive normalization; born at one so a newborn's noise reads small)
            lb = int(self.cfg["gate_level_band"]); vb = float(v_now[lb])
            m.v_scale[lb] += (1.0 / float(self.cfg["diff_horizon"])) * (vb * vb - float(m.v_scale[lb]))
            level = float(self.cfg["gate_level_w"]) * max(-5.0, min(5.0, vb / (1.0 + math.sqrt(max(0.0, float(m.v_scale[lb]))))))
        gam = m.gammas()
        with torch.no_grad():                                    # THE CLOCK advances: the day's fraction, the night at 1
            m.vc_clock_prev.copy_(m.vc_clock); m.vc_clock.fill_(min(1.0, float(self.sleep_pressure) / max(1.0, float(self.cfg["wake_ticks"]))))
        with torch.enable_grad():
            # the organs hold no dropout or batch statistics, so train()/eval() changed nothing but cost 12% of the tick in
            # Python (a recursive mode flip over every module twice a tick); the body stays in eval mode from construction
            v_prev_live = m.values(self._bands_prev.detach()) if getattr(self, "_bands_prev", None) is not None else None
            if v_prev_live is not None:
                # SEMI-GRADIENT TD, the convergent form: the target r + gamma V(s_now) is detached and
                # only the heads learn; the bands' states are fixed features of the stream (their input
                # maps are born, not trained by this error). The differential heads read their states
                # centered on a running mean at the reward rate's horizon (adaptation), so the relative
                # value's free constant has no direction to walk in.
                with torch.no_grad():
                    eta = 1.0 / float(self.cfg["diff_horizon"])
                    for b in range(len(gam)):                          # every band's mean (the ventral critic centers on all)
                        m.band_mu[b] += eta * (self._bands_prev[b].detach() - m.band_mu[b])
                # DISCOUNTED TD below the differential horizon, AVERAGE-REWARD (differential) TD at and
                # above it: with the discount near 1 the bootstrapped value ran away (the slowest band
                # read 7000 against a true return near 85, correlation -0.995: run 21, day 20). The
                # baseline is the reward rate estimated at the band's own clock, tonic dopamine.
                N_ = int(self.cfg.get("night_ticks", 0)); an_ = bool(N_) and bool(getattr(self, "_after_night", False)); self._after_night = False
                nf = [(gam[b] ** N_) if an_ else 1.0 for b in range(len(gam))]      # THE NIGHT TAKES TIME
                td = torch.stack([(r + gam[b] * nf[b] * v_now[b].detach() - v_prev_live[b]) if not self._differential[b]
                                  else (r - float(self.rbar) + v_now[b].detach() - v_prev_live[b]) for b in range(len(gam))])
                self.rbar += (1.0 / float(self.cfg["diff_horizon"])) * (r - self.rbar)   # the reward rate, tonic dopamine
                # THE VENTRAL CRITIC: discounted TD at a definite long horizon over the whole ladder's states (semi-gradient,
                # the target detached; linear on fixed features, convergent)
                with torch.no_grad():
                    vl_now = m.value_long(self.bands, m.r_tr, m.vc_clock)
                gl = float(self.cfg.get("vcrit_gamma", 1.0 - 1.0 / 1024)); nfl = (gl ** N_) if an_ else 1.0   # THE NIGHT TAKES TIME
                if int(self.cfg.get("vcrit_diff", 0)):
                    td_long = (r - float(self.rbar)) + vl_now.detach() - m.value_long(self._bands_prev.detach(), m.r_tr_prev, m.vc_clock_prev)
                    gl_tr = 1.0                                       # the trace decays at lambda alone
                else:
                    td_long = r + gl * nfl * vl_now.detach() - m.value_long(self._bands_prev.detach(), m.r_tr_prev, m.vc_clock_prev)
                    gl_tr = gl
                lam = float(self.cfg.get("vcrit_lambda", 0.0)); tau = float(self.cfg.get("vcrit_tau", 0.0))
                rls = int(self.cfg.get("vcrit_rls", 0))
                if rls:
                    # THE DECORRELATED CRITIC: the statistics of the trace against the state's discounted change and the reward
                    with torch.no_grad():
                        one = torch.ones(1, dtype=torch.float64, device="cpu")
                        x_p = m.vcrit_input(self._bands_prev.detach(), m.r_tr_prev, m.vc_clock_prev).detach().cpu()[self._vc_idx].double(); x_n = m.vcrit_input(self.bands.detach(), m.r_tr, m.vc_clock).detach().cpu()[self._vc_idx].double()
                        ntau = int(self.cfg.get("vcrit_norm_tau", 0))
                        if ntau:
                            # THE PRIOR AS THE METRIC (2026-09-05 21:40): the homeostatic statistics take the state but do NOT transform
                            # it. The evidence A, b is accumulated in the raw coordinates, which never drift; the statistics act only on
                            # the prior at the solve (delta x sd_i^2 on each weight, the level free), which is the standardized prior
                            # expressed in raw coordinates. Standardizing the inputs themselves while the statistics formed put every
                            # tick's evidence in different coordinates: on a recorded lived day the body's own head read -0.59 where the
                            # same evidence in fixed coordinates read +0.71 (nt_worlds3/4).
                            m.vcrit_norm_update(x_n, ntau)
                        xa_prev = torch.cat([x_p, one]); xa_now = torch.cat([x_n, one])
                        e = getattr(self, "_vc_e", None)
                        self._vc_e = (gl_tr * lam * e if e is not None else torch.zeros_like(xa_prev)) + xa_prev
                        vf_ = float(self.cfg.get("vcrit_forget", 0) or 0); beta = (1.0 - 1.0 / vf_) if vf_ > 0 else 1.0
                        dlt = xa_prev - gl * (nfl if not int(self.cfg.get("vcrit_diff", 0)) else 1.0) * xa_now
                        m.vc_A.mul_(beta).addr_(self._vc_e, dlt)
                        if beta < 1.0 and not ntau:
                            m.vc_A.diagonal().add_((1.0 - beta) * self._vc_delta)   # the constant prior inside A (the form without statistics)
                        m.vc_b.mul_(beta).add_(self._vc_e * float(r))
                        if self.ticks % int(self.cfg.get("vcrit_rls_every", 64)) == 0:
                            m.vcrit_rls_solve(self._vc_idx, prior=(self._vc_delta if ntau else None))
                if int(self.cfg.get("fast_rls", 0)) and (not stri or getattr(self, "_z_prev", None) is not None):
                    # THE FAST CRITIC DECORRELATED: the dopamine band's evidence on its own state (or the striatal input), at its own horizon
                    with torch.no_grad():
                        fb = int(self.cfg["dopamine_band"]); gf = float(gam[fb]); one = torch.ones(1, dtype=torch.float64, device="cpu")
                        if stri:
                            xf_p = self._z_prev.detach().cpu().double(); xf_n = self._z_now.detach().cpu().double()
                        else:
                            xf_p = self._bands_prev[fb].detach().cpu().double(); xf_n = self.bands[fb].detach().cpu().double()
                        m.fast_norm_update(xf_n, float(self.cfg.get("fast_rls_forget", 36000)))
                        xa_p = torch.cat([xf_p, one]); xa_n = torch.cat([xf_n, one])
                        ef = getattr(self, "_vf_e", None)
                        self._vf_e = (gf * ef if ef is not None else torch.zeros_like(xa_p)) + xa_p           # the trace at gamma (lambda 1)
                        bf = 1.0 - 1.0 / float(self.cfg.get("fast_rls_forget", 36000))
                        m.vf_A.mul_(bf).addr_(self._vf_e, xa_p - gf * xa_n); m.vf_b.mul_(bf).add_(self._vf_e * float(r))
                        if self.ticks % int(self.cfg.get("fast_rls_every", 64)) == 0:
                            m.fast_rls_solve(fb, prior=self._vf_delta)
                with torch.no_grad():
                    x_prev = torch.cat([m.vcrit_input(self._bands_prev.detach(), m.r_tr_prev, m.vc_clock_prev), torch.ones(1, device=self.dev)])
                    if lam > 0.0:
                        # THE CRITIC'S ELIGIBILITY TRACE (TD(lambda), backward view): the trace of the critic's inputs
                        # decays at gamma * lambda; the error captures it
                        tr = getattr(self, "_vtrace", None)
                        self._vtrace = (gl_tr * lam * tr if tr is not None else torch.zeros_like(x_prev)) + x_prev
                    e_in = self._vtrace if lam > 0.0 else x_prev
                if rls:
                    loss_vl = td_long.detach() * 0.0                   # the decorrelated head learns above, not by the gradient
                elif tau > 0.0:
                    # THE NORMALIZED STEP at the critic's time constant: the input's energy tracked over a horizon
                    with torch.no_grad():
                        en = float((e_in * e_in).sum())
                        self._vcrit_energy = en if getattr(self, "_vcrit_energy", None) is None else (1.0 - (1.0 - gl)) * self._vcrit_energy + (1.0 - gl) * en
                        step = ((1.0 - gl) / tau) / (self._vcrit_energy + 1e-6) * float(td_long.detach())
                        m.vcrit.weight += step * e_in[:-1].unsqueeze(0); m.vcrit.bias += step * e_in[-1:]
                    loss_vl = td_long.detach() * 0.0
                elif lam > 0.0:
                    loss_vl = -(td_long.detach() * (m.vcrit.weight[0] @ self._vtrace[:-1] + m.vcrit.bias[0] * self._vtrace[-1]))   # gradient -error x trace
                else:
                    loss_vl = td_long ** 2
                if int(self.cfg.get("fast_rls", 0)):
                    mask_ = torch.ones_like(td); mask_[int(self.cfg["dopamine_band"])] = 0.0; loss_v = ((td * mask_) ** 2).mean() + loss_vl
                else:
                    loss_v = (td ** 2).mean() + loss_vl
                # Go/NoGo on the bands' own updates: a positive error pulls the gate open, a negative one shut
                gates = torch.stack([torch.sigmoid(m.band_gate[b](self._bands_prev[b].detach())).squeeze()
                                     for b in range(len(gam))])
                tgt = (td.detach() > 0).float()
                loss_g = (td.detach().abs() * F.binary_cross_entropy(gates.clamp(1e-6, 1 - 1e-6), tgt, reduction="none")).mean()
                self.opt_value.zero_grad(set_to_none=True)
                (loss_v + 0.01 * loss_g).backward()
                self.opt_value.step()
                vf = float(self.cfg.get("vcrit_forget", 0) or 0)
                if vf > 0 and not rls:
                    with torch.no_grad():
                        m.vcrit.weight.mul_(1.0 - 1.0 / vf)       # the forgetting head
                # DOPAMINE: the TD error of the band whose discount matches dopamine's (clock 16,
                # gamma 0.9375): an expected reward fires before it lands, a missed one dips
                if stri and getattr(self, "_z_prev", None) is not None:
                    with torch.no_grad():                        # the dopamine from the striatal head
                        fb_ = int(self.cfg["dopamine_band"])
                        delta = float(r + float(gam[fb_]) * m.fast_value(self._z_now) - m.fast_value(self._z_prev))
                else:
                    delta = float(td[int(self.cfg["dopamine_band"])].detach())
                if int(self.cfg.get("actor", 0)) and str(self.cfg.get("actor_form", "add")) == "softmax":
                    self._chooser_learn(delta)
                elif int(self.cfg.get("actor", 0)) and getattr(self, "_e_actor", None) is not None:
                    with torch.no_grad():                        # THE ACTOR'S LESSON: dopamine times the eligibility, the weights forgetting
                        m.actor.weight.mul_(1.0 - 1.0 / float(self.cfg.get("actor_forget", 36000))).add_(float(self.cfg.get("actor_lr", 0.02)) * delta * self._e_actor)
                if stri and int(self.cfg.get("wm", 0)) and getattr(m, "stri_wm", 0):
                    with torch.no_grad():                        # WORKING MEMORY: latch at a burst, clear at the reward or with age
                        m.wm_tick()
                        if felt > 0 or float(m.wm_age) > float(self.cfg.get("wm_max", 512)):
                            m.wm_clear()
                        elif delta > float(self.cfg.get("wm_burst", 0.5)):
                            m.wm_latch(m.striatum_read()); self._z_now = m.stri_in()
                delta_slow = float(td[int(self.cfg["gate_slow_band"])].detach())
                delta_long = float(td_long.detach()); vlong = float(vl_now)
                if int(self.cfg.get("vcrit_auto", 0)):
                    self._vrel_update(vlong, r)
                for b in range(len(gam)):
                    self.v_buf[b].append((self._bands_prev[b].detach().cpu(), r, self.bands[b].detach().cpu()))
            else:
                delta = r; delta_slow = r; delta_long = r; vlong = 0.0
            # THE CHAIN CLOSES (the eleventh defect, found by review 2026-09-06): the state that was this update's target is the
            # next update's source. Before, the source was the end of the previous tick (after its own symbol) and the target the
            # middle of this one (after the world's), so the body's own step fell in a gap no transition covered and every
            # critic's evidence matrix was asymmetric (16-30 percent) where a closed chain's is symmetric.
            self._bands_prev = self.bands.clone()
            if stri:
                self._z_prev = self._z_now
        with torch.no_grad():                                    # THE TONIC TRACES advance with this tick's felt reward
            m.r_tr_prev.copy_(m.r_tr); m.r_tr += (float(r) - m.r_tr) / torch.tensor([float(c) for c in m.clocks], device=m.r_tr.device)
        self._dopa = delta
        self._dopa_since_utt = float(getattr(self, "_dopa_since_utt", 0.0)) + max(0.0, float(delta))   # the reward since the last utterance kept
        # THE SYNAPTIC TAG: every act (or rest) leaves a tag on the gate's weights, (act - p) x the gate's input, that
        # decays at the ventral critic's own horizon; the ventral error, as it arrives over the following minutes,
        # captures the tags (Frey and Morris 1997: a tag set by activity, captured by later dopamine). The expected
        # update is the sum over acts of (act - p) x (the long return that followed minus the critic's estimate): the
        # policy gradient at the critic's horizon, where the twelve-tick sum of the lesson could not reach the parent's
        # attention. gate_slow_lr 0 = off.
        slr = float(self.cfg.get("gate_slow_lr", 0.0))
        if slr > 0.0 and getattr(self, "_gate_tag", None) is not None:
            with torch.no_grad():
                m.mouth_gate.weight += slr * delta_long * self._gate_tag[:-1].unsqueeze(0)
                m.mouth_gate.bias += slr * delta_long * self._gate_tag[-1:]
        return delta, delta_slow, delta_long, vlong, level, gam

    def _own_face(self, C1, r):
        """its own face, learned from yours: a foresight of the felt reward (face_form "foresee") or a readout of it"""
        m = self.m
        # --- its face learns from yours (a readout) ---
        if str(self.cfg.get("face_form", "read")) == "foresee":
            # THE FACE ORGAN FORESEES (2026-09-08, §5c): from the stream a tick ago it predicts the felt reward of this tick; its
            # reliability, the slope of the felt reward on that foresight, is measured, so imagination can be weighed by it
            # a least-squares readout (as the fast critic's): decorrelated, in reward units, its evidence forgetting over face_tau
            # ticks and solved every face_every; tick-by-tick gradient steps on a target that is zero on most ticks swung it wildly
            with torch.no_grad():
                xin = self._face_input_vec(C1)
                if getattr(self, "_fin_prev", None) is not None and xin is not None:
                    x = torch.cat([self._fin_prev, torch.ones(1, dtype=torch.float64)])
                    f_prev = float(x @ self._fh_w)                                   # the foresight made a tick ago, before this evidence
                    bf = 1.0 - 1.0 / float(self.cfg.get("face_tau", 36000))
                    self._fh_A.mul_(bf).addr_(x, x); self._fh_b.mul_(bf).add_(x * float(r))
                    self._frel_update(f_prev, float(r))
                    if (self.ticks + 32) % int(self.cfg.get("face_every", 64)) == 0:   # offset from the fast critic's solve
                        self._face_solve()
                f_pred = self._foresee(xin) if xin is not None else 0.0            # the felt reward it foresees for the next tick
            self._fin_prev = xin; self._C1_prev = C1.detach()
        else:
            with torch.enable_grad():
                f_pred = m.face_head(C1.detach()).squeeze() * 6.0
                lf = (f_pred - torch.tensor(float(self.face_now), device=self.dev)) ** 2
                self.opt_face.zero_grad(set_to_none=True); lf.backward(); self.opt_face.step()
        its_face = float(f_pred.detach()) if torch.is_tensor(f_pred) else float(f_pred); self._fpred_now = its_face
        return its_face

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
            if int(self.cfg.get("gate_ear", 0)):
                # THE EAR: the world's symbol this tick, its own act last tick (sensed, not inferred)
                feat = torch.cat([feat, torch.tensor([1.0 if u != self.sil else 0.0, 1.0 if getattr(self, "_acted_last", False) else 0.0], device=self.dev)])
            z = m.mouth_gate(feat.unsqueeze(0))[0, 0] / (1.0 + self.stress / 10.0)   # stress flattens the choice
            fl = float(self.cfg["gate_floor"])
            # THE LISTENING REFLEX (gate_listen; 2026-09-17, item 41): the learned gate had shut itself during the parent's lines (the ear's
            # weight -48) and the talk-overs came from what no learned weight reaches, the spontaneous floor starting a word on a
            # twentieth of the ticks and the chunk then running it with no gate decision. While the world's utterance is open (from its
            # symbol until the offset, the event's end the body computes) the floor is scaled by (1 - gate_listen) and a running word
            # is cut: the vocal suppression while hearing speech, innate before turn-taking is learned; the learned weights still decide.
            listening = float(self.cfg.get("gate_listen", 0.0)) > 0.0 and not self._offset_done and self.ticks - self._last_world < 10 ** 6
            if listening:
                fl = fl * (1.0 - float(self.cfg.get("gate_listen", 0.0)))
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

    def _bookkeep(self, u, who, nxt, its_face, felt, ent, p_act, delta, level, r, vlong, delta_long, conf1, surp1, probs):
        """the stream, the page, the tick's record for the instruments, the sleep switch"""
        m = self.m
        # --- bookkeeping ---
        self.stream.append((int(u), 0)); self.stream.append((int(nxt), 1))
        self.page.append(((self.tok.decode([int(u)]) if u != self.sil else ""), 0, round(self.face_now, 2), round(its_face, 2), who))
        self.page.append(((self.tok.decode([int(nxt)]) if nxt != self.sil else ""), 1, round(self.face_now, 2), round(its_face, 2), False))
        if len(self.page) > 40000:
            del self.page[:20000]; self.page_base += 20000
        self.face_prev = self.face_now
        self.ticks += 1; self.sleep_pressure += 1
        self.last = {"tick": self.ticks, "you": round(self.face_now, 2), "face": round(its_face, 2), "vrel": round(self._vrel_corr, 3),
                     "mood": round(self.mood, 2), "cort": round(self.fatigue, 2), "fatigue": round(self.fatigue, 2),
                     "stress": round(self.stress, 2), "ent": round(ent, 2), "felt": felt,
                     "said": (self.tok.decode([int(nxt)]) if nxt != self.sil else ""),
                     "gate": round(p_act, 3), "dopamine": round(delta, 3), "doses": self.n_bursts, "level": round(level, 3), "r": round(float(r), 3),
                     "vlong": round(vlong, 3), "dlong": round(delta_long, 3),
                     "store": self.store.n(), "store_conf": round(conf1, 3), "surprise": round(surp1, 3),
                     "own": [self.tok.decode([int(probs.argmax())]), round(float(probs.max()), 3)],
                     "gate_lesson": self._gate_last, "wake": self._wake_last}
        if self.sleep_pressure >= int(self.cfg["wake_ticks"]) and len(self.win) >= 8:
            self._sleep_now()

    # ---------------- the gate's lesson (the striatum's opponent rule) ----------------
    def _vrel_update(self, vlong, r):
        """THE RELIABILITY GAIN: every 256 ticks, once the buffer holds 4096, the return at the head's horizon for the
        oldest 256 ticks (3072 rewards each, 95 percent of the discounted mass) against the head's value then; the
        moments decay over 8192 samples; the gain is max(0, corr)"""
        self._vbuf_v.append(float(vlong)); self._vbuf_r.append(float(r))
        if len(self._vbuf_r) < 4096 or self.ticks % 256 != 0:
            return
        gl = float(self.cfg.get("vcrit_gamma", 1.0 - 1.0 / 1024))
        rs = torch.tensor(list(self._vbuf_r)); vs = torch.tensor(list(self._vbuf_v))
        disc = gl ** torch.arange(3072, dtype=torch.float32)
        G = torch.stack([(rs[i:i + 3072] * disc).sum() for i in range(256)]); V = vs[:256]
        d = 1.0 - 1.0 / 65536; m = self._vrel                     # ~5 days of samples (review: 8192 held ~8 independent returns)
        for v_, g_ in zip(V.tolist(), G.tolist()):
            m[0] = d * m[0] + 1.0; m[1] = d * m[1] + v_; m[2] = d * m[2] + g_
            m[3] = d * m[3] + v_ * v_; m[4] = d * m[4] + g_ * g_; m[5] = d * m[5] + v_ * g_
        n = m[0]; mv, mg = m[1] / n, m[2] / n
        var_v, var_g = m[3] / n - mv * mv, m[4] / n - mg * mg; cov = m[5] / n - mv * mg
        corr = cov / math.sqrt(max(var_v, 1e-9) * max(var_g, 1e-9))
        self._vrel_corr = float(corr); self._vrel_gain = float(max(0.0, min(1.0, cov / max(var_v, 1e-9))))   # the slope (review: corr flickered at random)

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

    # ---------------- the waking cortex ----------------
    def _wake_lesson(self):
        win = list(self.win)[-int(self.cfg["wake_window"]):]
        if len(win) < 8:
            return None
        xs, whos, faces, bundles, reads = self._window_tensors(win)
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
        if float(w.sum()) < 1:
            return None
        m = self.m; m.train()
        try:
            self.opt_day.zero_grad(set_to_none=True)
            u = m.inputs(xs, whos, faces, bundles, reads)     # the window as lived: its own sound in it, attenuated
            C = m.stream(u)
            # the cortex is trained on ITS OWN forecast, day and night alike (predictive coding: each
            # area learns from its own error); recall is a parallel contribution the mouth reads, never
            # a term in the cortex's error (with the sum in the loss the day taught only the residual
            # the store missed and undid the night: run 13, day 4)
            pred = m.latent_pred(C)
            ll, lc = m.latent_loss(pred, y, w=w)
            if str(self.cfg.get("rem_form", "forecast")) == "imagine":
                fl, fc = torch.zeros((), device=self.dev), 1.0            # the forecast heads retired (§5c: their target was the stream's own dynamics)
            else:
                fl, fc = m.forecast_loss(C[:-1].detach(), bundles[1:], sig=0.0)   # the PFC's heads learn from the stream, not through it
            # THE DAY'S PLASTICITY GATED (wake_base, wake_dopa, wake_novel; 2026-09-15): the waking write into cortex at the ordinary
            # tick's share, raised by the dopamine of the moment and by the running surprise (the rewarded and the novel are written,
            # the rest weakly); at the defaults (1, 0, 0) the lesson is as before
            gate_ = float(self.cfg.get("wake_base", 1.0)) + float(self.cfg.get("wake_dopa", 0.0)) * abs(float(getattr(self, "_dopa", 0.0))) + float(self.cfg.get("wake_novel", 0.0)) * float(getattr(self, "_surp_run", 0.0))
            loss = (ll + fl) * (1.0 + self.stress / 10.0) * gate_      # stress raises plasticity
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
                out.append(d + end_)
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

    def _dream_inputs(self, ids, mem_on):
        """a dream as a window: fresh bands (a night's working state), the store leading if mem_on"""
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
                u = m.inputs(torch.tensor(xs[:n], device=self.dev), xos[:n],
                             torch.zeros(n, 2, device=self.dev), torch.stack(bundles), torch.stack(reads))
                C = m.stream(u)[-1]
                bands = m.band_update(bands, C)
        T = len(xs)
        return (torch.tensor(xs, device=self.dev), xos,
                torch.zeros(T, 2, device=self.dev), torch.stack(bundles), torch.stack(reads), torch.tensor(ids, device=self.dev))

    def _dream_batch(self, dream_list, own_list=None):
        """THE DREAMS IN LOCKSTEP (2026-09-13): a list of dreams as one right-padded batch, the bands run along each as _dream_inputs
        runs them one at a time (a causal cortex: a dream's positions never see the padding after them). Returns xs, xos [B, T],
        faces [B, T, 2], bundles [B, T, nb, d], reads [B, T, d] (zeros: the store is off in the lesson), y [B, T] the targets and
        w [B, T] their weights (1 on a dream's own positions, 0 on the padding)"""
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
        faces = torch.zeros(B, T, 2, device=self.dev); reads = torch.zeros(B, T, m.d, device=self.dev)
        bundles = torch.zeros(B, T, nb, m.d, device=self.dev); bands = torch.zeros(B, nb, m.d, device=self.dev)
        with torch.no_grad():
            cache = [None] * len(m.blocks)                                        # the stream's keys and values so far, per block
            for t in range(T):
                bundles[:, t] = bands
                if t + 1 < T:
                    C = m.stream_step(m.inputs(xs[:, t], xos[:, t], faces[:, t], bundles[:, t], reads[:, t]), cache)
                    bands = m.band_update_b(bands, C)
        return xs, xos, faces, bundles, reads, y, w

    def _gauge_batched(self, dreams, owns=None, bs=32):
        """gauge() over lockstep batches: the same count, many dreams at once; owns (dream_who): its own symbols are not counted"""
        hits = 0.0; n = 0.0; cos_sum = 0.0
        bans = [b for b in self.bans if b != self.eot]
        with torch.no_grad():
            for i in range(0, len(dreams), bs):
                xs, xos, faces, bundles, reads, y, w = self._dream_batch(dreams[i:i + bs], owns[i:i + bs] if owns is not None else None)
                pred = self.m.latent_pred(self.m.stream(self.m.inputs(xs, xos, faces, bundles, reads)))
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
                xs, whos, faces, bundles, reads, y = self._dream_inputs(ids, mem_on=False)
                C = self.m.stream(self.m.inputs(xs, whos, faces, bundles, reads))
                pred = self.m.latent_pred(C)
                lg = self.m.readout(pred); lg[:, [b for b in self.bans if b != self.eot]] = float("-inf")
                if self.end_id != self.sil:
                    lg[:, self.sil] = float("-inf")                       # under the rest form the rest is a target the cortex may hit
                hits += int((lg.argmax(-1) == y).sum()); n += int(y.numel())   # a dream's end (the turn's) counts as a target
                cos_sum += float(F.cosine_similarity(pred, self.m.E.weight[y], dim=-1).sum())
        self._gauge_cos = (round(cos_sum / n, 3) if n else None)
        return (round(hits / n, 3) if n else None), n

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
            load = float(self.cfg.get("night_load", 0.0)); n_starts = None
            if load > 0.0:
                n_starts = int(min(int(self.cfg.get("night_starts_max", 192)), max(int(self.cfg["night_starts"]), round(load * max(0, n_new)))))
            who_on = int(self.cfg.get("dream_who", 0)) > 0 and int(self.cfg.get("night_batch", 0)) > 0
            if who_on:
                dreams, owns = self.dreams(n_starts, with_who=True)
            else:
                dreams = self.dreams(n_starts); owns = None
            rep["dreams"] = len(dreams); rep["new_slots"] = int(n_new); rep["draw_serials"] = list(getattr(self, "_last_draw", []))
            if owns is not None:                                        # its own symbols in capitals, to be read
                rep["examples"] = ["".join(self.tok.decode([i]).upper() if o else self.tok.decode([i]) for i, o in zip(d, w_))[:32] for d, w_ in zip(dreams[:8], owns[:8])]
                rep["own_share"] = round(sum(sum(w_) for w_ in owns) / max(1, sum(len(w_) for w_ in owns)), 3)
            else:
                rep["examples"] = [self.tok.decode(d)[:32] for d in dreams[:8]]
            rep["mean_len"] = round(sum(len(d) for d in dreams) / len(dreams), 1) if dreams else 0
            if not dreams:
                rep["note"] = "the store holds nothing to dream"
            else:
                # --- NREM: sleep's own optimizer; the dreams in batches (night_batch > 0) or one step per round ---
                before, nsym = self.gauge(dreams, owns); before_cos = self._gauge_cos
                # THE MOMENT'S HORIZON (night_beta2; 2026-09-16, night 226): a fresh optimizer's second moment forms over about a thousand
                # steps at 0.999, so at the third round an outlier gradient on a parameter whose moment is still small is normalised
                # into a step many times the rate, which the global clip does not bound (night 226's loss rose 0.23 -> 0.34 in one
                # round and the cortex was left in a basin the rate cannot climb). At 0.99 the moment forms in a hundred steps, before
                # the rounds where the night's outliers arrive. A disclosed constant; 0.999 = as before.
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
                        xs, xos, faces, bundles, reads, y, w = self._dream_batch([dreams[j] for j in order[i0:i0 + nbatch_]],
                                                                                 [owns[j] for j in order[i0:i0 + nbatch_]] if owns is not None else None)
                        C = m.stream(m.inputs(xs, xos, faces, bundles, reads))
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
                        xs, whos, faces, bundles, reads, y = self._dream_inputs(ids, mem_on=False)
                        C = m.stream(m.inputs(xs, whos, faces, bundles, reads))
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
                mid, _ = self.gauge(dreams, owns); mid_cos = self._gauge_cos      # the gauge after NREM, before REM
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
                self._value_replay()
                m.eval()
                del opt
                finite = all(bool(torch.isfinite(p).all()) for p in m.parameters())
                after, _ = self.gauge(dreams, owns); after_cos = self._gauge_cos
                # THE NIGHT STANDS (the user's word, 2026-09-11 20:00: 'remove night rollback'): from night 106 to 115 a night whose dream recall
                # fell by more than 0.15 was discarded and the organs returned to the evening's save; nothing in biology does that, and it
                # never fired after the night it was built for. A night is kept whatever it does. Only a non-finite lesson (the arithmetic
                # broken, not a learning outcome) reloads the evening's organs, so the body is not left with NaN for weights.
                rep["discarded"] = not finite; rep["undone_for"] = "non-finite" if not finite else None
                if rep["discarded"] and self.save_path and os.path.exists(self.save_path):
                    sd = torch.load(self.save_path, map_location="cpu", weights_only=False)
                    m.load_state_dict(sd["organs"]); m.to(self.dev)
                    after, _ = self.gauge(dreams, owns); after_cos = self._gauge_cos
                rep.update({"nrem_steps": nrem, "nrem_loss": losses[:3] + (["..."] if len(losses) > 6 else []) + losses[-3:],
                            "nrem_curve": [round(float(x), 4) for x in losses],   # the whole curve (2026-09-12): to read where the rounds stop paying
                            "rem_steps": rem_steps, "rem_cos": (round(rem_cos[-1], 3) if rem_cos else None),
                            "rem_cos_first": (round(rem_cos[0], 3) if rem_cos else None), "rem_imagined": rem_imag,
                            "gauge": {"before": before, "after_nrem": mid, "after": after, "symbols": nsym,
                                      "cos_before": before_cos, "cos_after_nrem": mid_cos, "cos_after": after_cos}})
            # --- the rest: the store fades, the working state wakes fresh, the body is saved ---
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
            self._bands_prev = None; self._C_last = None; self.v_prev = None
            self._z_prev = None; self._z_now = None; self._e_actor = None
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
            self.sleep_pressure = int(self.cfg["wake_ticks"]) // 2
            self.last_night = rep
        finally:
            self.asleep = False
        return rep

    def _face_solve(self):
        """the foreseeing face organ from its evidence: w = (A + ridge)^-1 b, the ridge scaled to the evidence's own size"""
        m = self.m
        with torch.no_grad():
            A = self._fh_A; lam = float(self.cfg.get("face_ridge", 1.0)) * float(A.diagonal()[:-1].mean().clamp_min(1e-9))
            R = torch.full((A.shape[0],), lam, dtype=A.dtype); R[-1] = 0.0
            try:
                w = torch.linalg.solve(A + torch.diag(R), self._fh_b)
            except Exception:
                w = torch.linalg.lstsq(A + torch.diag(R), self._fh_b.unsqueeze(1)).solution.squeeze(1)
            self._fh_w = w
            if w.numel() == m.face_head.weight.numel() + 1:
                m.face_head.weight[0].copy_(w[:-1].to(m.face_head.weight)); m.face_head.bias[0] = w[-1].to(m.face_head.bias)

    def _arel_update(self, v, g):
        """the actor's reliability: the running moments of (its vote for the act taken, the reward of the ticks after) over actor_tau acts"""
        d = 1.0 - 1.0 / float(self.cfg.get("actor_tau", 36000)); m = self._arel
        m[0] = d * m[0] + 1.0; m[1] = d * m[1] + v; m[2] = d * m[2] + g; m[3] = d * m[3] + v * v; m[4] = d * m[4] + g * g; m[5] = d * m[5] + v * g
        n = m[0]; mv, mg = m[1] / n, m[2] / n
        var_v, var_g = m[3] / n - mv * mv, m[4] / n - mg * mg; cov = m[5] / n - mv * mg
        self._arel_corr = float(cov / math.sqrt(max(var_v, 1e-9) * max(var_g, 1e-9))) if n > 64 else 0.0
        self._arel_gain = float(max(0.0, min(1.0, cov / max(var_v, 1e-9)))) if n > 64 else 0.0

    def _rem_temperature(self):
        """REM's sampling temperature: rem_temp, and under the world form scaled by the readout's base over its world-calibrated base,
        so the dreams are drawn at the sharpness the world proved (a base of 25 calibrated to 8 samples at temperature 3)"""
        rt = float(self.cfg.get("rem_temp", 0.0))
        if rt > 0 and str(self.cfg.get("sharp_form", "fixed")) == "world" and int(self.cfg.get("rem_world_temp", 0)) and float(self.sharp_cal) > 0:
            rt = rt * float(self.cfg["sharp_base"]) / float(self.sharp_cal)   # rem_world_temp 1: the dreams at the world's proved sharpness (off until the reading is trusted)
        return rt

    def _face_input_vec(self, C1):
        """the face organ's input this tick, as a double vector: the stream, or the striatal input the fast critic reads"""
        if str(self.cfg.get("face_input", "cortex")) == "striatum" and self.m.stri_W.numel() > 0:
            z = getattr(self, "_z_now", None)
            return None if z is None else z.detach().cpu().double()
        return C1.detach().cpu().double()

    def _foresee(self, xin):
        """the felt reward the face organ foresees for the next tick, from its input"""
        if xin is None or xin.numel() != self._fh_n:
            return 0.0
        return float(torch.cat([xin, torch.ones(1, dtype=torch.float64)]) @ self._fh_w)

    def _frel_update(self, f, r):
        """the face organ's reliability: the running moments of (foresight, felt reward) over face_tau ticks; the slope, clipped to [0, 1]"""
        d = 1.0 - 1.0 / float(self.cfg.get("face_tau", 36000)); m = self._frel
        m[0] = d * m[0] + 1.0; m[1] = d * m[1] + f; m[2] = d * m[2] + r; m[3] = d * m[3] + f * f; m[4] = d * m[4] + r * r; m[5] = d * m[5] + f * r
        n = m[0]; mf, mr = m[1] / n, m[2] / n
        var_f, var_r = m[3] / n - mf * mf, m[4] / n - mr * mr; cov = m[5] / n - mf * mr
        self._frel_corr = float(cov / math.sqrt(max(var_f, 1e-9) * max(var_r, 1e-9))) if n > 64 else 0.0
        self._frel_gain = float(max(0.0, min(1.0, cov / max(var_f, 1e-9)))) if n > 64 else 0.0

    def _face_weight(self):
        """how much an imagined tick counts: the face organ's correlation with the felt reward, clipped at zero. Not its slope: a
        predictor of small variance clips its slope at one while discriminating nothing (night 50: slope 1.0, correlation 0.15, a
        mean foreseen reward of 0.03 over 336 imagined transitions at full weight). The correlation is the share it has proved."""
        return max(0.0, float(self._frel_corr))

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
        wm_saved = (m.wm_slot.clone(), m.wm_on.clone(), m.wm_age.clone()) if getattr(m, "stri_wm", 0) else None
        m.striatum_reset()
        if wm_saved is not None:
            m.wm_clear()
        bands = torch.zeros_like(self.bands); xs = [self.sil] + list(ids[:k])
        reads, bundles, Cs = [], [], []; z_prev = None; f_prev = 0.0; e = None; n_up = 0; rsum = 0.0
        one = torch.ones(1, dtype=torch.float64)
        with torch.no_grad():
            for step in range(len(xs) + L):
                if step >= len(xs):
                    lg = m.readout(m.latent_pred(Cs[-1])).clone(); lg[self.bans] = float("-inf")   # the rest may be imagined: the world's stop
                    rt = self._rem_temperature()
                    xs.append(int(torch.multinomial(torch.softmax(lg / rt, 0), 1)) if rt > 0 else int(lg.argmax()))
                x = xs[step]
                m.striatum_push(0 if x != self.sil else 3, x if x != self.sil else 0)
                reads.append(torch.zeros(m.d, device=self.dev)); bundles.append(bands.clone())
                n = step + 1
                u = m.inputs(torch.tensor(xs[:n], device=self.dev), torch.full((n,), self.sil, dtype=torch.long, device=self.dev),
                             torch.zeros(n, 2, device=self.dev), torch.stack(bundles), torch.stack(reads))
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
        reads, bundles, Cs, bnext = [], [], [], []
        for step in range(len(xs) + L):
            if step >= len(xs):
                # its imagined next symbol, read off its forecast (no gradient through the choice),
                # heard as the world's in the next time step
                with torch.no_grad():
                    lg = m.readout(m.latent_pred(Cs[-1])).clone(); lg[self.bans] = float("-inf"); lg[self.sil] = float("-inf")
                    rt = self._rem_temperature()
                    # THE DREAM SAMPLED, NOT TAKEN AT ITS MODE (rem_temp > 0; 2026-09-06): biology's REM is noisy; the greedy
                    # continuation reproduces the store's most frequent lines and consolidates nothing new. 0 = greedy (as before).
                    xs.append(int(torch.multinomial(torch.softmax(lg / rt, 0), 1)) if rt > 0 else int(lg.argmax()))
            with torch.no_grad():
                bag = float(self.cfg["bag_decay"]) * (m.shift(bag) if xs[step] != self.sil else bag) + (m.E.weight[xs[step]] if xs[step] != self.sil else 0.0)
            reads.append(torch.zeros(m.d, device=self.dev)); bundles.append(bands.clone())
            n = step + 1
            u = m.inputs(torch.tensor(xs[:n], device=self.dev), torch.full((n,), self.sil, dtype=torch.long, device=self.dev),
                         torch.zeros(n, 2, device=self.dev), torch.stack(bundles), torch.stack(reads))
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
                    j = self.store.last_idx; n += 1
                    if n == 2:
                        self.store.mark_start(bag, ex)         # the start mark on the second symbol's slot, as the world's onsets are marked
                    if chain and prev >= 0:
                        self.store.link(prev, j, tag=self.store.episode)
                    prev = j
                bag = d_ * m.shift(bag) + ex
        self._own_stored_n = getattr(self, "_own_stored_n", 0) + n
        return n

    def _sleep_now(self):
        self.queue.clear(); self.queue_who.clear()
        self.night()

    # ---------------- the hands ----------------
    def type_text(self, s, who=""):
        """the world's hand: each symbol enters the page in its turn, tagged with who typed it (the visitor page of 2026-09-11:
        the parent yields to a visitor it can see on the page; the tag is on the page, never inside)"""
        n = 0; who = str(who)[:8]
        for ch in s:
            i = self.tok.token_to_id(ch)
            if i is not None and i != self.sil and i not in self.reserved and len(self.queue) < 600:   # the reserved symbols are not typed
                self.queue.append(i); self.queue_who.append(who); n += 1
        return {"queued": n}

    def set_face(self, expr):
        self.face_now = max(-6.0, min(6.0, float(expr)))
        return {"you": self.face_now}

    def state(self, since=0):
        """THE PAGE, and nothing else (2026-09-08, the review): what a parent may see. The words, the faces, whether it sleeps,
        how many nights it has lived. No reading from inside reaches the one who decides the face."""
        i = max(0, int(since) - self.page_base)
        return {"page": self.page[i:], "n": self.page_base + len(self.page), "base": self.page_base, "queued": len(self.queue),
                "asleep": self.asleep, "nights": self.nights}

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
        return {"last": self.last, "sleep_pressure": self.sleep_pressure, "wake_ticks": int(self.cfg["wake_ticks"]),
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

    # ---------------- save / load ----------------
    def save(self, path=None):
        path = path or self.save_path
        blob = {"organs": self.m.state_dict(), "store": self.store.state_dict(), "cfg": self.cfg, "env": {k_: os.environ.get(k_, "") for k_ in ("PARENT", "REPLY", "WAIT", "TALKOVER_FROWN", "WORD_SMILES", "ROOM", "ANSWER_LEVELS", "REPLY_QUIET")},
                "arch": {"vocab": self.m.vocab, "d": self.m.d, "layers": len(self.m.blocks), "heads": self.m.blocks[0].attn.num_heads,
                         "window": self.m.window, "clocks": list(self.m.clocks)},
                "life": {"ticks": self.ticks, "nights": self.nights, "day_n": self.day_n, "sleep_pressure": self.sleep_pressure, "heard": self.heard.cpu(),
                         "fatigue": self.fatigue, "stress": self.stress, "mood": self.mood, "n_bursts": self.n_bursts,
                         "sym_freq": self.sym_freq, "perf": {int(k): float(v) for k, v in self.perf.items()}, "last_night": self.last_night, "rbar": float(self.rbar),
                         "feat_mu": (self._feat_mu.cpu() if getattr(self, "_feat_mu", None) is not None else None),
                         "vrel": list(self._vrel), "vrel_gain": float(self._vrel_gain), "vrel_corr": float(self._vrel_corr),
                         "vbuf_v": list(self._vbuf_v), "vbuf_r": list(self._vbuf_r), "sharp_cal": float(self.sharp_cal),
                         "frel": list(self._frel), "frel_gain": float(self._frel_gain), "frel_corr": float(self._frel_corr),
                         "fh_A": self._fh_A.clone(), "fh_b": self._fh_b.clone(), "fh_w": self._fh_w.clone(),
                         "arel": list(self._arel), "arel_gain": float(self._arel_gain), "arel_corr": float(self._arel_corr),
                         "store_after_night": self._store_after_night, "utts": self.utts, "utt_S": self.utt_S,
                         "utt_N": self.utt_N, "utt_serial": int(self._utt_serial), "c_mu": self._c_mu.clone(), "c_n": int(self._c_n), "ctx_cur": self.ctx_cur.clone(), "ctx_prev": self.ctx_prev.clone(), "utt_open": bool(self._utt_open)}}
        torch.save(blob, path + ".tmp"); os.replace(path + ".tmp", path)
        return {"saved": path}

    @classmethod
    def load(cls, path, tok, device="cpu", cfg=None, seed=0, save_path=None):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        a = blob["arch"]
        organs = Organs(a["vocab"], d=a["d"], layers=a["layers"], heads=a["heads"], window=a["window"], clocks=tuple(a["clocks"]))
        w = blob["organs"].get("mouth_gate.weight")
        if w is not None and w.shape[1] > organs.mouth_gate.weight.shape[1]:
            organs.widen_gate(w.shape[1] - organs.mouth_gate.weight.shape[1])   # a body with the ear
        if w is not None and w.shape[1] < organs.mouth_gate.weight.shape[1]:
            # an older body's gate had fewer inputs (no salience, no level): those weights are born at zero
            blob["organs"]["mouth_gate.weight"] = torch.cat([w, torch.zeros(w.shape[0], organs.mouth_gate.weight.shape[1] - w.shape[1])], 1)
        vw = blob["organs"].get("vcrit.weight")
        if vw is not None and vw.shape[1] < organs.vcrit.weight.shape[1]:   # an older critic without the trace inputs: those weights born at zero
            blob["organs"]["vcrit.weight"] = torch.cat([vw, torch.zeros(vw.shape[0], organs.vcrit.weight.shape[1] - vw.shape[1])], 1)
        vf_saved = {k_: blob["organs"].pop(k_) for k_ in ("vf_A", "vf_b", "vf_mu", "vf_var", "vf_n") if k_ in blob["organs"]}     # the fast head's evidence, sized by the life below
        st_saved = {k_: blob["organs"].pop(k_) for k_ in ("stri_W", "stri_b", "stri_line", "vfast.weight", "vfast.bias", "actor.weight", "actor.bias", "wm_slot", "wm_on", "wm_age") if k_ in blob["organs"]}   # the striatal input, sized by the life below
        vc_saved = {k_: blob["organs"].pop(k_) for k_ in ("vc_A", "vc_b", "vc_mu", "vc_var", "vc_n", "vc_form") if k_ in blob["organs"]}   # sized by the life below
        missing = organs.load_state_dict(blob["organs"], strict=False)
        if [k_ for k_ in missing.missing_keys if not (k_.startswith("vc_") or k_.startswith("vf_") or k_.startswith("stri_") or k_.startswith("vfast.") or k_.startswith("actor.") or k_.startswith("wm_"))]:
            print("load: organs without", [k_ for k_ in missing.missing_keys if not (k_.startswith("vc_") or k_.startswith("vf_") or k_.startswith("stri_") or k_.startswith("vfast.") or k_.startswith("actor.") or k_.startswith("wm_"))], "(an older recipe; born fresh where missing)")
        c = dict(blob.get("cfg") or {})
        c.setdefault("gate_int_form", "value")            # an older body keeps the value form and its own drive unless told
        c.update(cfg or {})
        life = cls(organs, tok, cfg=c, device=device, seed=seed, save_path=save_path or path)
        saved_norm = vc_saved.get("vc_mu") is not None and vc_saved["vc_mu"].numel() > 0; norm_on = int(c.get("vcrit_norm_tau", 0)) > 0
        saved_form = float(vc_saved["vc_form"]) if vc_saved.get("vc_form") is not None else 1.0
        if vc_saved and int(c.get("vcrit_rls", 0)) and vc_saved.get("vc_A") is not None and vc_saved["vc_A"].shape == life.m.vc_A.shape and saved_norm == norm_on and (not norm_on or saved_form == float(life.m.vc_form)):
            # the decorrelated critic's memory, in the units it was accumulated in; in other units it begins again from the prior
            life.m.vc_A.copy_(vc_saved["vc_A"].cpu()); life.m.vc_b.copy_(vc_saved["vc_b"].cpu())
            if norm_on:
                life.m.vc_mu.copy_(vc_saved["vc_mu"].cpu()); life.m.vc_var.copy_(vc_saved["vc_var"].cpu()); life.m.vc_n.copy_(vc_saved["vc_n"].cpu())
        if vf_saved and int(c.get("fast_rls", 0)) and vf_saved.get("vf_A") is not None and vf_saved["vf_A"].shape == life.m.vf_A.shape:
            # the fast critic's memory (the ninth defect: until 2026-09-06 the loader sized fresh zeros here and dropped the
            # saved evidence, so every reloaded fast-critic body met its prior with no evidence and its head was crushed
            # at the first solve; the running body was never affected, only its copies, stalks and restarts)
            for k_ in ("vf_A", "vf_b", "vf_mu", "vf_var", "vf_n"):
                getattr(life.m, k_).copy_(vf_saved[k_].cpu())
        if st_saved and life.m.stri_W.numel() > 0 and st_saved.get("stri_W") is not None and st_saved["stri_W"].shape == life.m.stri_W.shape:
            with torch.no_grad():                                  # the striatal input as born, its line, and its head
                life.m.stri_W.copy_(st_saved["stri_W"].to(device)); life.m.stri_b.copy_(st_saved["stri_b"].to(device)); life.m.stri_line.copy_(st_saved["stri_line"].to(device))
                life.m.vfast.weight.copy_(st_saved["vfast.weight"].to(device)); life.m.vfast.bias.copy_(st_saved["vfast.bias"].to(device))
                if st_saved.get("actor.weight") is not None and st_saved["actor.weight"].shape == life.m.actor.weight.shape:
                    life.m.actor.weight.copy_(st_saved["actor.weight"].to(device)); life.m.actor.bias.copy_(st_saved["actor.bias"].to(device))
                if st_saved.get("wm_slot") is not None and st_saved["wm_slot"].shape == life.m.wm_slot.shape:
                    life.m.wm_slot.copy_(st_saved["wm_slot"].to(device)); life.m.wm_on.copy_(st_saved["wm_on"].to(device)); life.m.wm_age.copy_(st_saved["wm_age"].to(device))
        life.store.load_state_dict(blob["store"])
        life.store.saturate = bool(int(life.cfg.get("store_sat", 0)))
        if life.store.saturate and not life.store.sat_done:
            life.store.compress()                                    # a save from before the law: converted once
        L = blob.get("life") or {}
        for k in ("ticks", "nights", "day_n", "sleep_pressure", "fatigue", "stress", "mood", "n_bursts", "last_night"):
            if k in L:
                setattr(life, k, L[k])
        if L.get("feat_mu") is not None:
            life._feat_mu = L["feat_mu"].to(device)                  # the gate's adapted input, its running mean
        if L.get("vrel") is not None:                                # THE THIRTEENTH DEFECT (2026-09-08): the prefrontal voice's evidence was dropped at every load
            life._vrel = [float(v) for v in L["vrel"]]; life._vrel_gain = float(L.get("vrel_gain", 0.0)); life._vrel_corr = float(L.get("vrel_corr", 0.0))
            life._vbuf_v.extend(float(v) for v in (L.get("vbuf_v") or [])); life._vbuf_r.extend(float(v) for v in (L.get("vbuf_r") or []))
        if L.get("sharp_cal") is not None:
            life.sharp_cal = float(L["sharp_cal"])
        if L.get("utts"):
            life.utts = [list(u) for u in L["utts"]]; life.utt_S = [float(v) for v in L.get("utt_S", [1.0] * len(L["utts"]))]
            life.utt_N = [int(v) for v in (L.get("utt_N") or range(1, len(life.utts) + 1))]   # a save from before the serials: taken as consecutive
            life._utt_serial = int(L.get("utt_serial", max(life.utt_N) if life.utt_N else 0))
        if L.get("ctx_cur") is not None and tuple(L["ctx_cur"].shape) == tuple(life.ctx_cur.shape):
            life.ctx_cur.copy_(L["ctx_cur"]); life.ctx_prev.copy_(L["ctx_prev"]); life._utt_open = bool(L.get("utt_open", False))
        if L.get("c_mu") is not None and tuple(L["c_mu"].shape) == tuple(life._c_mu.shape):
            life._c_mu.copy_(L["c_mu"]); life._c_n = int(L.get("c_n", 0))
        if L.get("frel") is not None:
            life._frel = [float(v) for v in L["frel"]]; life._frel_gain = float(L.get("frel_gain", 0.0)); life._frel_corr = float(L.get("frel_corr", 0.0))
        if L.get("fh_A") is not None and tuple(L["fh_A"].shape) == tuple(life._fh_A.shape):
            life._fh_A.copy_(L["fh_A"]); life._fh_b.copy_(L["fh_b"])
            if L.get("fh_w") is not None and tuple(L["fh_w"].shape) == tuple(life._fh_w.shape):
                life._fh_w.copy_(L["fh_w"])
        if L.get("arel") is not None:
            life._arel = [float(v) for v in L["arel"]]; life._arel_gain = float(L.get("arel_gain", 0.0)); life._arel_corr = float(L.get("arel_corr", 0.0))
        if L.get("store_after_night") is not None:
            life._store_after_night = int(L["store_after_night"])
        elif isinstance(L.get("last_night"), dict) and L["last_night"].get("store_slots") is not None:
            life._store_after_night = int(L["last_night"]["store_slots"])   # a save from before the count: the last night's report holds it
        life.sym_freq = dict(L.get("sym_freq") or {})
        life.perf = {int(k): float(v) for k, v in (L.get("perf") or {}).items()}
        if L.get("rbar") is not None:
            rb = L["rbar"]; life.rbar = float(rb.mean()) if torch.is_tensor(rb) else float(rb)
        if L.get("heard") is not None:
            life.heard = L["heard"].to(device)
        return life

    @classmethod
    def birth(cls, tok, device="cpu", d=256, layers=6, heads=4, window=64, cfg=None, seed=0, save_path=None):
        torch.manual_seed(int(seed))
        organs = Organs(tok.get_vocab_size(), d=d, layers=layers, heads=heads, window=window, birth_act=float((cfg or {}).get("birth_act", PHYSIOLOGY["birth_act"])))
        return cls(organs, tok, cfg=cfg, device=device, seed=seed, save_path=save_path)
