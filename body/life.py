"""the second body alive (BODY_SPEC.md §1, §3, §4): ticks, feelings, the one reward system,
the waking lesson, sleep by fatigue, the night. The caregiver reaches it only through THE DIARY
protocol in serve.py; nothing here reads the caregiver's mind or edits the body's words.

HOW TO READ THIS FILE. `PHYSIOLOGY` holds every constant, grouped by organ; the served body's effective set is its save plus
ops/BASE_FLAGS.txt (ops/served_cfg.py prints it), and the history of every value is in BODY_SPEC.md's appendix and ITERATIONS.md,
not here. `Life.__init__` wires the organs (body/model.py: `Organs` the learned parts, `Store` the hippocampus) and the state a
tick touches. `tick()` is one moment of the body's clock in eight phases, each a method in this order: `_sense` (the world's symbol
or its quiet, the offset by the count, the face felt as reward, then the reward's other terms: the anatomy's reward sources in
their order), `_hear` (the symbol enters the stream, the store writes what surprised it, the offset by the settle law),
`_learn_values` (every critic learns; the fast band's error is dopamine), `_own_face`, `_choose` (whether to speak, then what: the
readout at the mood's sharpness, the actor's chunk or plan), `_act` (its own symbol or rest enters the stream, the credits),
`_feel_and_learn` (the feelings, the gate's lesson, the waking cortex lesson), `_bookkeep` (the page, the record, the sleep
switch).
`night()` runs NREM (the dreams from `dreams()`, batched, the cortex learning with the recall off as its input), REM
(`_rem_imagine`, `_rem_rollout`), the value replay, the store's fade, the save. `type_text`, `set_face`, `state` and `insides`
are the page's endpoints; `save`, `load` and `birth` are the body on disk.

WHERE THE METHODS LIVE (the split of 2026-09-23, review 2026-09-22 section 4, step 2): this file keeps `Life.__init__`,
`tick()` and the read-only `tok` (the anatomy's tokenizer; the core refactor's step R2, docs/SIM_DESIGN.md 8.4); every other method
was moved verbatim into a mixin in body/core/ (its __init__.py has the map): senses, memory, cortex, mouth, critics, actor, night,
persistence, instruments and (step R6c) the cerebellum, with `PHYSIOLOGY` in body/core/physiology.py, re-exported here (a body whose
switch `cereb` is on has a cerebellum below the tick, body/core/cerebellum.py: the organs' m.cereb, its constants CEREB, absent unless
given; the life sets its world's sub-tick hook and the world calls it every 10 ms inside `apply`; nothing of the tick reads it, and the
diary has none of it); the anatomy, the body's senses,
effectors and reward sources declared, is body/core/anatomy.py, and the frame, the world at one tick, body/core/world.py. Since step
R4 the window holds each of the anatomy's channels under its field and the cortex's input is their codes summed in the anatomy's
order (`Organs.inputs(anatomy, obs, own, bundles)`). Since step R5 the tick's choice, act and gate lessons run over the anatomy's
effectors: the voice first (effector 0, today's code and names; `_choose` returns its gate's own draw too), then each later effector
(its working state in `motor`, its gates' optimizer `opt_motor`; neither exists for the diary). Since step R9 the body lives in a world
(`life.world`, body/core/world.py; the diary's DiaryWorld, today's queue and face, unless one is given): the tick takes the world's frame
at its senses' phase (or the frame the loop hands in) and returns its acts for the world; the sleep switch pauses the world for the
night; body/serve.py runs it all through a WorldLoop. Since step R6 each later effector has its motor timing part (body/core/timing.py,
the organs' m.timing[name]: act_pred its proposal, the forward half its correction, act_inv learning online with its own optimizer
`opt_inv`; act_pred and the correction learning in the waking lesson with `opt_pred`, plain steps on the lesson's gradient, its
labels at their reliability, each element bounded, since R6 fix 7) and, under chunk_gate, its chunks and learned stops;
the diary has none of it. Since step R6h the voice may stand at any place among the effectors (the G1 numbers its vocal tract 0 and the
words' output 1: body/sim/anatomy.py), every other effector a motor effector in the declared order (`anatomy.motors`); a motor effector
may carry the performance error (declared), its own fatigue, movement units, act_inv's batched lessons with the kappa correction, its
forward error at its gate (body/core/timing.py, physiology.py's MOTOR), the born patterns summed at the cord and the born biases
(body/core/cord.py, `CordMixin`, physiology.py's REFLEX); a vector channel may be read through its born code (body/model.py
`BornCode`, encs.<name>); the tick's acts are then an `Acts` (body/core/world.py), with the cord's steps and the VOR's constants beside
them. The diary has none of it."""
import collections
import math  # noqa: F401  (math, os and F: module names body.life had before the split; the moved methods import their own)
import os  # noqa: F401
import time

import torch
import torch.nn.functional as F  # noqa: F401

from .model import Organs, Store, FastStore  # noqa: F401  (Organs and Store: the names body.life always offered)
from .core.physiology import PHYSIOLOGY, SWITCHES, MOTOR, CEREB, REFLEX, FRAMES, AMYG, SLEEP
from .core.anatomy import anatomy_for
from .core.world import World, DiaryWorld, Acts
from .core.cerebellum import CerebellumMixin
from .core.senses import SensesMixin
from .core.memory import MemoryMixin
from .core.cortex import CortexMixin
from .core.mouth import MouthMixin
from .core.critics import CriticsMixin
from .core.actor import ActorMixin
from .core.night import NightMixin
from .core.persistence import PersistenceMixin
from .core.instruments import InstrumentsMixin
from .core.timing import TimingMixin, GatedDescent
from .core.cord import CordMixin
from .core.frames import FramesMixin
from .core.amygdala import AmygdalaMixin
from .core.sleep import SleepMixin

# `from body.life import *` gives exactly the names it gave before the split (the mixins stay reachable as attributes)
__all__ = ["collections", "math", "os", "time", "torch", "F", "Organs", "Store", "FastStore", "PHYSIOLOGY", "Life"]


class Life(SensesMixin, MemoryMixin, CortexMixin, MouthMixin, CriticsMixin, ActorMixin, NightMixin, PersistenceMixin, InstrumentsMixin, TimingMixin,
           CerebellumMixin, CordMixin, FramesMixin, AmygdalaMixin, SleepMixin):
    def __init__(self, organs, tok, cfg=None, device="cpu", seed=0, save_path=None, world=None):
        unknown = sorted(k_ for k_ in (cfg or {}) if k_ not in PHYSIOLOGY and k_ not in SWITCHES and k_ not in MOTOR and k_ not in CEREB and k_ not in REFLEX
                         and k_ not in FRAMES and k_ not in AMYG and k_ not in SLEEP)   # the switches, the motor's, the cerebellum's, (R6h) the born patterns', (R7) the frames', (R7d) the amygdala's and (R8) the night's constants are known, absent unless given
        if unknown:
            print("physiology: unknown keys (ignored):", unknown, flush=True)     # review 2026-09-06: a typo was a silent no-op for 21 days
        self.m = organs.to(device); self.m.eval()
        self.dev = device
        self.cfg = dict(PHYSIOLOGY); self.cfg.update(cfg or {})
        # THE BODY'S ANATOMY (the core refactor's step R2, docs/SIM_DESIGN.md 8.4): what it senses, how it acts, what it feels as reward
        # (body/core/anatomy.py). `tok` is what the callers always passed: a tokenizer builds the diary's LanguageAnatomy under this
        # life's constants; an anatomy may stand in its place. The tokenizer stays inside the language anatomy, for text only.
        self.anatomy = anatomy_for(tok, self.cfg)
        # ITS CHANNELS IN THE ORGANS (step R4): each channel's code is made by the organ it names, and each later channel that declares
        # a forecast has its head (Organs(..., channels=anatomy.channels) builds them); read here, nothing kept
        for i_, c_ in enumerate(self.anatomy.channels):
            try:
                org_ = getattr(organs, c_.organ, None) if "." not in c_.organ else organs.get_submodule(c_.organ)   # step R6h: encs.<name>
            except AttributeError:
                org_ = None
            if not isinstance(org_, torch.nn.Module):
                raise ValueError(f"Life: the channel {c_.name!r} is encoded by the organ {c_.organ!r}, which these organs do not have")
            if i_ and c_.forecast and c_.name not in getattr(organs, "chan_pred", {}):
                raise ValueError(f"Life: the channel {c_.name!r} declares a forecast, and these organs have no head for it "
                                 f"(built by Organs(..., channels=anatomy.channels))")
        _undeclared = sorted(set(getattr(organs, "chan_pred", {}).keys()) - {c_.name for c_ in self.anatomy.channels[1:] if c_.forecast})
        if _undeclared:
            raise ValueError(f"Life: the organs hold forecast heads for channels the anatomy does not declare: {_undeclared}")
        # ITS EFFECTORS IN THE ORGANS (step R5): each effector's table, gate and actor are the organs its declaration names (the voice's
        # the lexicon E, mouth_gate and actor; a later effector's built by Organs(..., effectors=anatomy.effectors)); read here, nothing kept
        v_at_ = self.anatomy.voice_at                           # step R6h: the voice may stand at any place; every other effector is a motor one
        for i_, e_ in enumerate(self.anatomy.effectors):
            mot_ = i_ != v_at_
            for what_, nm_ in (("table", e_.organ), ("gate", e_.gate), ("actor", e_.actor)):
                try:
                    organs.get_submodule(nm_)
                except AttributeError:
                    raise ValueError(f"Life: the effector {e_.name!r}'s {what_} is the organ {nm_!r}, which these organs do not have "
                                     f"(a later effector's are built by Organs(..., effectors=anatomy.effectors))") from None
            if mot_ and organs.get_submodule(e_.gate).in_features != organs.d + 5 + int(e_.n_in):
                raise ValueError(f"Life: the effector {e_.name!r}'s gate reads {organs.get_submodule(e_.gate).in_features} inputs, its declaration "
                                 f"{organs.d + 5 + int(e_.n_in)}")
            if mot_ and tuple(getattr(organs.get_submodule(e_.organ), "factors", ())) != tuple(int(k_) for k_ in e_.factors):
                raise ValueError(f"Life: the effector {e_.name!r}'s table has the joints {getattr(organs.get_submodule(e_.organ), 'factors', None)}, "
                                 f"its declaration {list(e_.factors)}")
            if mot_:                                                # step R6: its motor timing part as declared (joints, body sense, inverse model)
                tm_ = organs.timing[e_.name] if (hasattr(organs, "timing") and e_.name in organs.timing) else None
                want_ = (tuple(int(k_) for k_ in e_.factors), self.anatomy.sense_size(e_), bool(e_.inverse))
                if tm_ is None or (tm_.factors, tm_.sense_n, tm_.inverse) != want_ or \
                        (e_.inverse and tm_.inv[0].out_features != int(e_.inv_hidden)):
                    raise ValueError(f"Life: the effector {e_.name!r}'s motor timing part (timing.{e_.name}) is "
                                     f"{None if tm_ is None else (tm_.factors, tm_.sense_n, tm_.inverse)}, its declaration (joints, sense, inverse) {want_} "
                                     f"(built by Organs(..., channels=anatomy.channels, effectors=anatomy.effectors))")
        _undeclared = sorted((set(getattr(organs, "acts", {}).keys()) | set(getattr(organs, "timing", {}).keys())) - {e_.name for e_ in self.anatomy.motors})
        if _undeclared:
            raise ValueError(f"Life: the organs hold the organs of effectors the anatomy does not declare: {_undeclared}")
        # STEP R7a: the event lines' striatal delay line is the organs' when the anatomy declares event lines, and only then
        if bool(self.anatomy.events) != ("stri_eline" in organs._buffers):
            raise ValueError(f"Life: the anatomy declares {len(self.anatomy.events or ())} event lines and the organs "
                             f"{'hold' if 'stri_eline' in organs._buffers else 'hold no'} delay line for them (built by Organs(..., events=anatomy.events))")
        # STEP R6h (A41, C61): a motor effector's intrinsic term is the performance error per joint, its one form: under another form with
        # a weight on it, refused (the voice keeps both forms, as always)
        _int_ = [e_.name for e_ in self.anatomy.motors if e_.intrinsic]
        if _int_ and float(self.cfg.get("gate_int", 0.0)) != 0.0 and str(self.cfg.get("gate_int_form", "value")) != "error":
            raise ValueError(f"Life: the motor effectors {_int_} declare the intrinsic term, whose one form is the performance error "
                             f"(gate_int_form 'error'); this body's is {self.cfg.get('gate_int_form')!r} at gate_int {self.cfg.get('gate_int')}")
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
                    ne_ = len(self.anatomy.events or ())             # step R7a: the event lines' block after the effectors' (none for the diary)
                    rows_ = k_ * (2 * organs.vocab + 3) + sum(k_ * sum(int(f_) for f_ in e_.factors) for e_ in self.anatomy.motors) + k_ * ne_   # the language block, then the later effectors' per joint (steps R5, R5b), then the event lines' (R7a)
                    if (organs.stri_W.numel() == 0 or organs.stri_line.numel() != k_ or organs.stri_W.shape[1] != m_ or organs.stri_W.shape[0] != rows_
                            or organs.vfast.weight.shape[1] != m_ * (1 + wm_)):
                        organs.striatum_init(k_, m_, seed=seed, wm=wm_, effectors=self.anatomy.effectors, events=ne_)   # born (or re-born at a new size)
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
        # never types, named in the physiology by the body that is born, never found by a fixed string in the code. Derived by the
        # anatomy from its tokenizer since step R2 (LanguageAnatomy, body/core/anatomy.py, derives them as this method did).
        a_ = self.anatomy
        self.sil = a_.sil
        self.m.sil_id = self.sil                          # the cortex's inputs know its rest
        self.nl = a_.nl
        self.space_id = a_.space_id                           # the word boundary the planning actor decides at
        self.eot = a_.eot
        self.end_id = a_.end_id                               # what the offset teaches the cortex to expect (the rest under end_symbol "rest", else the world's turn-end): the offset (§2), never the mouth's
        # THE RESERVED SYMBOLS (anatomy, declared, 2026-09-08): the lexicon's control tokens, every `<...>` the tokenizer
        # defines (the world's turn-end, the old face tokens), except the rest; and this body's newline, a display symbol the
        # world never types. Neither the mouth nor the typing admits them. Declared from the tokenizer, never counted. The life's
        # own copies: nothing the life does can move the anatomy's declaration.
        self.reserved = list(a_.reserved)
        self.bans = list(a_.bans)
        self._last_world = -10 ** 9; self._offset_done = True; self._last_write = None; self._start_pending = False
        # THE SENSED TURN-TAKING (pace_sense): the partner's pace in log ticks (saved as life["pace"]): its pauses (P) and its returns (R_lo,
        # R_hi, from the first return heard); the gap's labels (a foreseen end in it, an end by the pause), the ear held by a line, the reply's
        # readiness (the end's tick, the forecast a step ago, the largest change since), the shadow's own end; the day's instruments
        self._pq = {"pause": 0.0, "lo": None, "hi": None, "n_pause": 0, "n_ret": 0, "warm": [], "fore_q": None, "n_mid": 0, "fore_warm": []}   # warm: the returns heard while they settle (None once settled)
        self._gap_foreseen = False; self._gap_paused = False; self._sh_done = True
        self._ear_held = False; self._ready_E = None; self._pred_ready = None; self._d_max = 0.0
        self._pace_day = self._pace_day_new()
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
        self.store = FastStore(self.m.d, cap=int(self.cfg.get("store_cap", 8192)), temp=float(self.cfg["store_temp"]), device=device, links=int(self.cfg.get("store_links", 4)))   # the copy-free store (2026-09-19)
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
        # THE WORLD (the core refactor's step R9, body/core/world.py): what the body lives in, frame by frame. The diary's (DiaryWorld)
        # unless one is given: the page's queue and the face its hand holds, read where they always were (the queue above, face_now);
        # the tick takes its frame from it at the senses' phase and the sleep switch pauses it for the night. Not saved with the body.
        if world is not None and not isinstance(world, World):
            raise TypeError(f"Life: the world is a body/core/world.py World, not {type(world).__name__}")
        self.world = DiaryWorld(self) if world is None else world
        # THE CEREBELLUM (step R6c; body/core/cerebellum.py): a body whose switch is on has its organs' cerebellum checked against its
        # anatomy's declaration and its world's sub-tick hook set; organs that hold one under a switch that is off are refused. The
        # diary's cfg has no switch and its organs no cerebellum: nothing runs, nothing is added
        if int(self.cfg.get("cereb", CEREB["cereb"])) or "cereb" in self.m._modules:
            self._cereb_attach()
        # THE AMYGDALA (step R7d; body/core/amygdala.py): a body whose switch is on has its organs' amygdala checked against its anatomy;
        # organs that hold one under a switch that is off are refused. The diary's cfg has no switch and its organs no amygdala
        if self._amyg_on() or "amyg" in self.m._modules:
            self._amyg_attach()
        # RECALL INTO ACTION (step R7f; body/core/frames.py): a body whose switch is on has its organs' maps and the heading's code checked
        if self._recall_on() or "recall" in self.m._modules:
            self._recall_attach()
        self._frames_check()                             # step R7: err_scale and wm_frames need the frames (body/core/frames.py)
        self._sleep_check()                              # step R8: the night over frames needs the frames (body/core/sleep.py)
        self.stream = collections.deque(maxlen=96)       # (id, who)
        self.last = {}
        self.credit = collections.deque(maxlen=64)
        # optimizers: the day's (the cortex and its forecasts), the striatum's (the gate), the critic's. A body with later effectors:
        # the day's holds every parameter but act_pred's and the corrections', which step with opt_pred (below; the diary has none)
        gated_ = {id(p_) for e_ in self.anatomy.motors for p_ in self._gated_params(e_)}
        self.opt_day = torch.optim.Adam([p_ for p_ in self.m.parameters() if id(p_) not in gated_] if gated_ else self.m.parameters(), lr=float(self.cfg["live_lr"]))
        if int(self.cfg.get("gate_ear", 0)) and self.m.mouth_gate.in_features == self.m.d + 5:
            self.m.widen_gate(2)                                # THE EAR: two inputs, born at zero
        if str(self.cfg.get("gate_opt", "sgd")) == "adam":
            self.opt_gate = torch.optim.Adam(self.m.mouth_gate.parameters(), lr=float(self.cfg.get("gate_adam_lr", 1e-3)))
        else:
            self.opt_gate = torch.optim.SGD(self.m.mouth_gate.parameters(), lr=float(self.cfg["gate_lr"]))
        # THE LATER EFFECTORS (step R5): each one's working state (its gate's buffer, its lesson's baseline and report, its act last tick,
        # its actor's trace, the tick's choice) in life.motor, in the anatomy's order after the voice; their gates' optimizer, of the
        # voice's kind, one for all (each lesson steps its own gate alone). The diary declares none: its life gains nothing.
        if len(self.anatomy.effectors) > 1:
            self.motor = [self._motor_state_new(e_) for e_ in self.anatomy.motors]
            gp_ = [p_ for e_ in self.anatomy.motors for p_ in self.m.get_submodule(e_.gate).parameters()]
            if str(self.cfg.get("gate_opt", "sgd")) == "adam":
                self.opt_motor = torch.optim.Adam(gp_, lr=float(self.cfg.get("gate_adam_lr", 1e-3)))
            else:
                self.opt_motor = torch.optim.SGD(gp_, lr=float(self.cfg["gate_lr"]))
            # THE INVERSE MODELS' OPTIMIZER (step R6): act_inv learns online from its own acts, a step a tick it acted, one optimizer for
            # every effector that declares one (each lesson steps its own alone); the waking lesson never reaches it
            ip_ = [p_ for e_ in self.anatomy.motors if e_.inverse for p_ in self.m.timing[e_.name].inv.parameters()]
            if ip_:
                self.opt_inv = torch.optim.Adam(ip_, lr=float(self.cfg.get("act_inv_lr", MOTOR["act_inv_lr"])))
            # ACT_PRED'S PLASTICITY GATED BY ITS LABELS' RELIABILITY (the R6 verifier's third finding; R6 fix 7, the lead's decision):
            # act_pred and the correction learn in the waking lesson by plain gradient descent, a group per later effector, one step a
            # lesson on the lesson's gradient (its labels at their reliability), no state: the step act_pred_rate x sqrt(d) x the waking
            # rate per unit of gradient, each element's at most act_pred_bound x sqrt(d) such steps (body/core/timing.py GatedDescent;
            # MOTOR); the labels act_inv reads reach these alone, never the stream (the verifier's fourth look)
            fan_ = float(self.m.d) ** 0.5                                # act_pred's fan-in, the stream's width d: sqrt(d)
            self.opt_pred = GatedDescent([{"params": self._gated_params(e_), "name": e_.name} for e_ in self.anatomy.motors],
                                         lr=float(self.cfg["live_lr"]), rate=float(self._motor_const("act_pred_rate")) * fan_,
                                         bound=float(self._motor_const("act_pred_bound")) * fan_ * float(self._motor_const("act_pred_rate")) * fan_)
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

    @property
    def tok(self):
        """the language anatomy's tokenizer, read-only (step R2: the body reads its anatomy; the tools and tests that turn text into
        symbols or back still find the tokenizer here)"""
        return self.anatomy.tok

    # ---------------- the tick ----------------
    def tick(self, frame=None):
        """one moment of the body's clock, in eight phases (each a method below, in this order). THE WORLD LOOP (step R9): the tick is
        lived on the world's frame, `frame` when the loop hands one in, else the one its world shows at the senses' phase (the diary's:
        built there from the queue, as always); it returns the tick's acts for the world, {effector name: act} (the voice's symbol or
        its rest, then each later effector's act or its rest; step R6: its reflex's act on a tick a reflex took), taken before the sleep
        switch's night rests them"""
        self._ring_vf.append(self.fast_value())            # the fast critic's value before this tick (the anticipation reading; the supervisor's, never the body's)
        self._decay_feelings()
        self.world.now = None                               # the frame of this tick is the one its senses take
        u, who, felt, r, off, settle_form, first_after_pause = self._sense() if frame is None else self._sense(frame)
        C1, pred1, surp1, conf1, stri = self._hear(u, r, felt, off, settle_form, first_after_pause)
        delta, delta_slow, delta_long, vlong, level, gam = self._learn_values(r, felt, stri)
        its_face = self._own_face(C1, r)
        if self._amyg_on():                                 # step R7d: the amygdala, after the critics and the face organ, before the choice
            self._amygdala(C1, self._tick_frame(u))
        acted, nxt, p_act, p_choice, probs, feat, ent, act_on, drew = self._choose(C1, pred1, u, level, stri)
        int_t = self._act(u, felt, stri, gam, delta, acted, nxt, p_act, p_choice, probs, feat, act_on, drew)
        self._feel_and_learn(delta, delta_slow, delta_long, feat, acted, int_t, p_act, drew)
        if self._frames_on():                               # step R7b: the frame's surprise, the event's end, the gated write, the record
            self._frame_tick(u, delta, r, nxt)
        acts = {self.anatomy.voice.name: int(nxt)}
        if len(self.anatomy.effectors) > 1:                 # step R6h: every effector's act in the declared order, the voice at its place
            wa_ = {e_.name: int(st_["now"]["world"]) for e_, st_ in zip(self.anatomy.motors, self.motor)}
            acts = Acts({e_.name: (acts[e_.name] if e_.name in acts else wa_[e_.name]) for e_ in self.anatomy.effectors})
            acts.cord = {e_.name: tuple(float(x_) for x_ in st_["now"]["cord"]) for e_, st_ in zip(self.anatomy.motors, self.motor)
                         if st_["now"].get("cord") is not None}              # the cord's patterns this tick (body/core/cord.py)
            acts.vor = self._vor_acts()                                      # the VOR's born constants (body/core/cord.py)
        self._bookkeep(u, who, nxt, its_face, felt, ent, p_act, delta, level, r, vlong, delta_long, conf1, surp1, probs)
        return acts
