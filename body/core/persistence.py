"""the body on disk (a mixin of `Life`, body/life.py): `save`, `load` (a reload: the save's own constants, then the caller's) and
`birth`. `load` and `birth` are classmethods: `cls` is `Life`. Their `tok` is what it always was, a tokenizer, or an anatomy in its
place (the core refactor's step R2): the life builds its anatomy from it (body/core/anatomy.py `anatomy_for`), and a birth sizes the
organs' alphabet from that anatomy. Both build the anatomy before the organs (a load under the save's constants, then the caller's), so
the organs build the forecast heads its later channels declare (step R4; the diary declares none). Both take the world the life lives
in (`world`, step R9; the diary's DiaryWorld when none is given): the world is not saved with the body. A body with later effectors
keeps their act_inv's reliability (step R6; its running confusion per joint since 2026-09-24, the kappas and the reliability) under
the save's life["motor"], a key only such a body's save has; their timing organs are in the organs' state (timing.<name>). act_pred
and the correction step plainly since R6 fix 7, with no optimizer state to keep: the moments of R6 fix 5 and 6's Adam that their
saves hold are not read (said once).

Moved verbatim from body/life.py (review 2026-09-22 section 4, step 2). Since step R6c a body whose cerebellum is on has it built by the
organs at birth and at load (`cerebellum_spec`: the anatomy's declaration and the born sizes, under the save's constants at a load), and
its every weight, its eligibility and its counters are in the organs' state (cereb.*): nothing of it is in the save's life dict. A load
whose switch differs from the save's organs (a cerebellum saved and switched off, or none saved and switched on) is refused (A20).
Since step R6h a motor effector's life["motor"] entry also keeps, where it has them, its performance error's running means ("perf"), its
own fatigue ("fatigue"), the kappa correction's running agreement and chance ("inv_ch") and act_inv's pairs gathered for the next batch
("inv_batch"); the born codes (encs.*) and the pattern generators' born phases (spg_phase) are in the organs' state, and since C54
their rhythms' seed (spg_seed): a rhythm is a function of the tick and these, so nothing of it is in the life dict and a load goes on
with it where the save left it (body/core/cord.py).
THE DAY (A70, the lead's decision of 2026-09-25: every state a save left out, saved; exact replay across a save and a night is the law).
A body with motor effectors also saves its whole working state (blob["day"]), so a life continued from a save made anywhere (inside a
day, mid-cry or mid-chunk; at a night's boundary) goes on as the life that went on, bit for bit (body/tests/test_motor.py, motor 12):
  - the life's random stream (its generator's state; the load's seed then seeds nothing of it);
  - the optimizers' moments (each one's state; its settings are the load's constants);
  - every working attribute of the life but the interface objects and the body's constants born from its anatomy and cfg (DAY_NOT):
    the window, the bags, the buffers, the page, the day's instruments, and among them the born orienting's memory of the cues' last
    fire, which runs on across a night (A70's finding; body/core/cord.py);
  - each motor effector's working state beyond what life["motor"] keeps: R6h's rest (the born cry's breath clock, the movement unit's
    held settings, the cord's counts, the pooled kappa) and R5's and R6's own (a chunk under way, the act last tick and the tick's choice,
    the forward half's foresight and error, the gate's buffer, the lesson's baseline and report, act_inv's last lesson, the chunks and
    their stops). The pattern generator's cache of its cycle is left out: its rhythm is the tick's and is found again from birth.
Before A70 a load began all of it afresh (the stream from the load's seed, the moments born again, the day as a night leaves it); a save
at a night's end, where the night has begun most of the day afresh, lost less, but still the stream, the moments and what a night keeps
(the orienting's memory, the value buffers, the day's rings). A motor body's save from before holds no day: the load says so once and
begins it as before (R6h's rule for older saves). A day lived by another anatomy (its channels or joints) is said and not read. The
day's tensors come back on the host (the sim's torch runs on the CPU, SIM_DESIGN.md 6.5). The language body's save holds none of it and
loads as it always did (its digests are the guard's)."""
import collections
import os

import torch

from ..model import Organs
from .anatomy import anatomy_for
from .cerebellum import cerebellum_spec
from .amygdala import amygdala_spec
from .frames import recall_spec
from .physiology import PHYSIOLOGY

# THE DAY (A70; the module's doc): what of the life's attributes the day does not hold, being held elsewhere or not the life's state: the
# organs, the store, the random stream, the optimizers (each by its moments apart) and the motor effectors' states (apart); the world
# and the save's path; the constants born with the life from its anatomy and cfg (its symbols, the critics' index and scales, the bands'
# kinds, the face organ's width); the device; the night's flag (a save inside a night wakes at its load); the page's queue (the diary
# world's hands); the feelings' wall clock (never read)
DAY_NOT = frozenset({"m", "store", "gen", "cfg", "save_path", "anatomy", "world", "motor", "dev", "asleep", "_t_feel", "queue", "queue_who",
                     "sil", "nl", "space_id", "eot", "end_id", "reserved", "bans", "_differential", "_fh_n", "_vc_idx", "_vc_delta",
                     "_vf_delta"})
# each motor effector's keys life["motor"] keeps in its own form (its rules for older saves stay theirs), and its pattern generator's
# cache of its cycle (found again from birth): not in the day
DAY_MOTOR_NOT = frozenset({"inv_conf", "inv_kappa", "inv_gain", "inv_n", "perf", "fatigue", "inv_ch", "inv_batch", "spg_cyc"})


def _anatomy_sig(anatomy):
    """what the day's working state is shaped by: each channel's name and size and each effector's name and joints"""
    return ([(c_.name, int(c_.size)) for c_ in anatomy.channels], [(e_.name, [int(k_) for k_ in e_.factors]) for e_ in anatomy.effectors])


class PersistenceMixin:
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
                         "utt_N": self.utt_N, "utt_serial": int(self._utt_serial), "c_mu": self._c_mu.clone(), "c_n": int(self._c_n), "ctx_cur": self.ctx_cur.clone(), "ctx_prev": self.ctx_prev.clone(),
                         "bands": self.bands.detach().cpu().clone(), "writes_today": int(getattr(self, "_writes_today", 0)), "store_fresh": bool(getattr(self, "_store_fresh", False)), "utt_open": bool(self._utt_open),
                         "pace": {k_: (list(v_) if isinstance(v_, list) else v_) for k_, v_ in self._pq.items()}, "pace_day": {k_: (list(v_) if isinstance(v_, list) else v_) for k_, v_ in self._pace_day.items()}}}
        if int(self.cfg.get("tag_trace", 0)):                         # defect 7 (tag_trace, R7c): the felt entries' running mean saved (a key only
            blob["life"]["utt_felt"] = {"m": float(getattr(self, "_utt_felt_m", 0.0)), "n": int(getattr(self, "_utt_felt_n", 0))}   # under the switch)
        if len(self.anatomy.effectors) > 1:                         # the later effectors' act_inv reliability (step R6); the diary's save has no such key
            # (act_pred's and the correction's plain step has no state since R6 fix 7: nothing of opt_pred is saved; R6 fix 5 and 6
            # saved their Adam's moments here, "moments"; the day's Adam and the others were born again at every load, as they always
            # were (review 2026-09-22 item 20), until A70: a motor body's day keeps their moments, below)
            blob["life"]["motor"] = {e_.name: {"inv_conf": (None if st_["inv_conf"] is None else [[[float(x_) for x_ in r_] for r_ in c_] for c_ in st_["inv_conf"]]),
                                               "inv_kappa": [float(x_) for x_ in st_["inv_kappa"]], "inv_gain": float(st_["inv_gain"]),
                                               "inv_n": int(st_["inv_n"]),
                                               **({"perf": [[float(x_) for x_ in r_] for r_ in st_["perf"]]} if st_.get("perf") is not None else {}),
                                               # step R6h: its own fatigue; the kappa correction's running agreement and chance; act_inv's
                                               # pairs gathered for the next batch (each key only where it has something)
                                               **({"fatigue": float(st_["fatigue"])} if float(st_.get("fatigue", 0.0)) != 0.0 else {}),
                                               **({"inv_ch": [[float(x_) for x_ in r_] for r_ in st_["inv_ch"]]} if st_.get("inv_ch") is not None else {}),
                                               **({"inv_batch": [(p_[0].tolist(), p_[1].tolist(), int(p_[2]), (None if p_[3] is None else [q_.tolist() for q_ in p_[3]]))
                                                                 for p_ in st_["inv_batch"]]} if st_.get("inv_batch") else {})}
                                     for e_, st_ in zip(self.anatomy.motors, self.motor)}
            # THE DAY (A70; the module's doc): the stream, the optimizers' moments, the life's working attributes (a test's instrument
            # wrapped round a method is code, not state: never saved) and each motor effector's working state beyond life["motor"]
            opts_ = {k_: v_ for k_, v_ in vars(self).items() if isinstance(v_, torch.optim.Optimizer)}
            blob["day"] = {"anatomy": _anatomy_sig(self.anatomy), "gen": self.gen.get_state(),
                           "optim": {k_: o_.state_dict()["state"] for k_, o_ in opts_.items()},
                           "life": {k_: v_ for k_, v_ in vars(self).items() if k_ not in DAY_NOT and k_ not in opts_ and not callable(v_)},
                           "motor": {e_.name: {k_: v_ for k_, v_ in st_.items() if k_ not in DAY_MOTOR_NOT}
                                     for e_, st_ in zip(self.anatomy.motors, self.motor)}}
        torch.save(blob, path + ".tmp"); os.replace(path + ".tmp", path)
        return {"saved": path}

    @classmethod
    def load(cls, path, tok, device="cpu", cfg=None, seed=0, save_path=None, world=None):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        a = blob["arch"]
        c = dict(blob.get("cfg") or {})
        c.setdefault("gate_int_form", "value")            # an older body keeps the value form and its own drive unless told
        c.update(cfg or {})
        anatomy = anatomy_for(tok, c)                     # the body's anatomy under the save's constants, then the caller's (no draw), before
                                                          # the organs: a later channel's forecast head is built with them (step R4)
        organs = Organs(a["vocab"], d=a["d"], layers=a["layers"], heads=a["heads"], window=a["window"], clocks=tuple(a["clocks"]),
                        channels=anatomy.channels, effectors=anatomy.effectors,   # a later effector's organs too (step R5; the tables from the save)
                        cerebellum=cerebellum_spec(anatomy, c),                     # and the cerebellum when its switch is on (step R6c; from the save)
                        events=anatomy.events,                                      # and the event lines' striatal line when declared (step R7a)
                        amygdala=amygdala_spec(anatomy, c),                         # and the amygdala when its switch is on (step R7d; from the save)
                        recall=recall_spec(anatomy, c))                             # and recall's maps when its switch is on (step R7f; from the save)
        # A BODY IS BORN WITH ITS SWITCHES (SIM_DESIGN.md A20; the R6c verifier's fourth finding): a save whose organs hold a cerebellum
        # loads only with the switch on, and one whose organs hold none only with it off, every one of the organ's entries from the save
        # (none born fresh at a load, none dropped); the language body's save holds none and its constants no switch, so nothing changes
        cb_saved = sorted(k_ for k_ in blob["organs"] if k_.split(".")[0] == "cereb")
        cb_built = sorted("cereb." + k_ for k_ in organs.cereb.state_dict()) if "cereb" in organs._modules else []
        if cb_saved != cb_built:
            raise ValueError(f"load: the save's organs hold {len(cb_saved)} entries of a cerebellum and its constants under this load switch it "
                             f"{'on' if cb_built else 'off'} ({len(cb_built)} entries): a body is born with its switches (SIM_DESIGN.md A20), so a "
                             f"cerebellum is neither dropped nor grown at a load")
        rc_saved = sorted(k_ for k_ in blob["organs"] if k_.split(".")[0] in ("recall", "head_code"))   # step R7f: recall's maps likewise (A20)
        rc_built = sorted(["head_code"] + ["recall." + k_ for k_ in organs.recall.state_dict()]) if "recall" in organs._modules else []
        if rc_saved != rc_built:
            raise ValueError(f"load: the save's organs hold {len(rc_saved)} entries of recall into action and its constants under this load switch "
                             f"it {'on' if rc_built else 'off'} ({len(rc_built)} entries): a body is born with its switches (SIM_DESIGN.md A20)")
        am_saved = sorted(k_ for k_ in blob["organs"] if k_.split(".")[0] == "amyg")          # step R7d: the amygdala likewise (A20)
        am_built = sorted("amyg." + k_ for k_ in organs.amyg.state_dict()) if "amyg" in organs._modules else []
        if am_saved != am_built:
            raise ValueError(f"load: the save's organs hold {len(am_saved)} entries of an amygdala and its constants under this load switch it "
                             f"{'on' if am_built else 'off'} ({len(am_built)} entries): a body is born with its switches (SIM_DESIGN.md A20), so an "
                             f"amygdala is neither dropped nor grown at a load")
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
        st_saved.update({k_: blob["organs"].pop(k_) for k_ in [k_ for k_ in blob["organs"] if k_ in ("stri_mline", "stri_eline") or k_.startswith("actors.")]})   # the later effectors' (step R5), the event lines' (R7a)
        vc_saved = {k_: blob["organs"].pop(k_) for k_ in ("vc_A", "vc_b", "vc_mu", "vc_var", "vc_n", "vc_form") if k_ in blob["organs"]}   # sized by the life below
        missing = organs.load_state_dict(blob["organs"], strict=False)
        motor_ = {e_.name for e_ in anatomy.motors}   # a later channel's head or a later effector's organs the anatomy does not declare: said, not loaded
        dropped = sorted([k_ for k_ in missing.unexpected_keys if k_.split(".")[0] in ("chan_pred", "acts", "gates", "timing", "encs", "spg_phase", "spg_seed")]
                         + [k_ for k_ in st_saved if (k_.startswith("actors.") and k_.split(".")[1] not in motor_) or (k_ == "stri_mline" and not motor_)
                            or (k_ == "stri_eline" and not anatomy.events)])
        if dropped:
            print("load: the save holds organs of channels or effectors this anatomy does not declare (not loaded):", dropped, flush=True)
        if [k_ for k_ in missing.missing_keys if not (k_.startswith("vc_") or k_.startswith("vf_") or k_.startswith("stri_") or k_.startswith("vfast.") or k_.startswith("actor.") or k_.startswith("actors.") or k_.startswith("wm_"))]:
            print("load: organs without", [k_ for k_ in missing.missing_keys if not (k_.startswith("vc_") or k_.startswith("vf_") or k_.startswith("stri_") or k_.startswith("vfast.") or k_.startswith("actor.") or k_.startswith("actors.") or k_.startswith("wm_"))], "(an older recipe; born fresh where missing)")
        life = cls(organs, anatomy, cfg=c, device=device, seed=seed, save_path=save_path or path, world=world)
        born_ = set(vars(life))                                      # the working attributes a life is born with (the day's, A70, below)
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
        same_ = bool(st_saved) and life.m.stri_W.numel() > 0 and st_saved.get("stri_W") is not None and st_saved["stri_W"].shape == life.m.stri_W.shape
        # A LATER EFFECTOR'S STRIATUM SAVED BEFORE ITS ROWS PER JOINT (step R5b; the R5b verifier's finding, 2026-09-24): the save's
        # effectors' block holds a row for every flat act, so the striatum's shape is not this body's, and until now nothing of it was
        # kept, the voice's heads and lines included. Its language block (the first k (2V + 3) rows), thresholds, lines, heads, slot and
        # actors are this body's: kept; only the effectors' rows are born again, per joint (as the life's birth of the striatum drew them).
        # Its fast critic's evidence and its heads were learned on units whose effectors' part the new rows change; the voice's is as saved.
        # ITS LIMIT (the R6 verifier's fourth look): the layouts are told apart by the striatum's shape alone, so a save of an anatomy
        # whose effectors' flat acts number as many as their joints' settings (a product equal to a sum: only an effector of two joints
        # of two settings, 2 x 2 = 2 + 2, beside effectors of one joint) loads by the present layout, its flat acts' rows read as its
        # joints'. No such save exists.
        nl_ = int(life.m.stri_line.numel()) * (2 * int(life.m.vocab) + 3)
        pre_r5b_ = (not same_ and bool(st_saved) and len(life.anatomy.effectors) > 1 and life.m.stri_W.numel() > 0 and st_saved.get("stri_W") is not None
                    and tuple(st_saved["stri_W"].shape) == (nl_ + sum(int(life.m.stri_line.numel()) * int(e_.n_acts) for e_ in life.anatomy.motors),
                                                            int(life.m.stri_W.shape[1]))
                    and st_saved.get("stri_line") is not None and st_saved["stri_line"].shape == life.m.stri_line.shape
                    and st_saved.get("vfast.weight") is not None and st_saved["vfast.weight"].shape == life.m.vfast.weight.shape)
        if pre_r5b_:
            print(f"load: the striatum was saved with a row per flat act (before R5b, 2026-09-24; {int(st_saved['stri_W'].shape[0])} rows): its language "
                  f"block, thresholds, lines, heads and actors are kept, the effectors' rows born again per joint ({int(life.m.stri_W.shape[0])} rows)", flush=True)
        if same_ or pre_r5b_:
            with torch.no_grad():                                  # the striatal input as born, its line, and its head
                if same_:
                    life.m.stri_W.copy_(st_saved["stri_W"].to(device))
                else:
                    life.m.stri_W[:nl_].copy_(st_saved["stri_W"][:nl_].to(device))   # the language block; the effectors' rows as born (R5b)
                life.m.stri_b.copy_(st_saved["stri_b"].to(device)); life.m.stri_line.copy_(st_saved["stri_line"].to(device))
                life.m.vfast.weight.copy_(st_saved["vfast.weight"].to(device)); life.m.vfast.bias.copy_(st_saved["vfast.bias"].to(device))
                if st_saved.get("actor.weight") is not None and st_saved["actor.weight"].shape == life.m.actor.weight.shape:
                    life.m.actor.weight.copy_(st_saved["actor.weight"].to(device)); life.m.actor.bias.copy_(st_saved["actor.bias"].to(device))
                if st_saved.get("wm_slot") is not None and st_saved["wm_slot"].shape == life.m.wm_slot.shape:
                    life.m.wm_slot.copy_(st_saved["wm_slot"].to(device)); life.m.wm_on.copy_(st_saved["wm_on"].to(device)); life.m.wm_age.copy_(st_saved["wm_age"].to(device))
                if st_saved.get("stri_mline") is not None and "stri_mline" in life.m._buffers and st_saved["stri_mline"].shape == life.m.stri_mline.shape:
                    life.m.stri_mline.copy_(st_saved["stri_mline"].to(device))   # the later effectors' lines and actors (step R5)
                if st_saved.get("stri_eline") is not None and "stri_eline" in life.m._buffers and st_saved["stri_eline"].shape == life.m.stri_eline.shape:
                    life.m.stri_eline.copy_(st_saved["stri_eline"].to(device))   # the event lines' line (step R7a)
                for e_ in life.anatomy.motors:
                    w_, b_ = st_saved.get(e_.actor + ".weight"), st_saved.get(e_.actor + ".bias")
                    a_ = life.m.get_submodule(e_.actor)
                    if w_ is not None and b_ is not None and w_.shape == a_.weight.shape:
                        a_.weight.copy_(w_.to(device)); a_.bias.copy_(b_.to(device))
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
        if L.get("bands") is not None and tuple(L["bands"].shape) == tuple(life.bands.shape):
            life.bands.copy_(L["bands"].to(device))                  # the slow bands survive a reload as they survive the night (the review of 2026-09-19: zeroed at every reload, every morning a birth)
        life._writes_today = int(L.get("writes_today", 0)); life._store_fresh = bool(L.get("store_fresh", False))
        if isinstance(L.get("pace"), dict):                          # the sensed pace's trackers (a save from before them: the newborn's start values)
            for k_ in ("fore_q",):
                if L["pace"].get(k_) is not None:
                    life._pq[k_] = float(L["pace"][k_])
            fw_ = L["pace"].get("fore_warm")
            life._pq["fore_warm"] = [float(x) for x in fw_] if isinstance(fw_, (list, tuple)) else None
            life._pq["n_mid"] = int(L["pace"].get("n_mid", 0))
            for k_ in ("pause", "lo", "hi"):
                if k_ in L["pace"]:
                    life._pq[k_] = (float(L["pace"][k_]) if L["pace"][k_] is not None else None)
            for k_ in ("n_pause", "n_ret"):
                life._pq[k_] = int(L["pace"].get(k_, 0))
            w_ = L["pace"].get("warm", None if life._pq["lo"] is not None else [])   # a save from before the settling: its running quantiles go on
            life._pq["warm"] = [float(x) for x in w_] if isinstance(w_, (list, tuple)) else None
            if life._pq["lo"] is None or life._pq["hi"] is None:
                life._pq["lo"] = None; life._pq["hi"] = None
                if life._pq["warm"] is None:
                    life._pq["warm"] = []                            # no returns known: they settle afresh
        if isinstance(L.get("pace_day"), dict):
            for k_, v_ in L["pace_day"].items():
                if k_ in life._pace_day:
                    life._pace_day[k_] = list(v_) if isinstance(v_, list) else v_
        if life._pace_mode() >= 2 and life._pq.get("warm") is not None:
            # THE SHADOW COMES FIRST (the review of 2026-09-23): live on returns not yet settled, R_lo and R_hi are few or none (no slot limit, no
            # wait, the whole floor) and P may still be the newborn's one tick; the design's shadow day (pace_sense 1) warms them. Said, not refused.
            P_, lo_, hi_ = life._pace()
            print(f"load: pace_sense 2 (live) with the partner's returns not settled ({len(life._pq['warm'])} of {max(1, int(round(1.0 / max(float(life.cfg.get('pace_eta', 0.05)), 1e-9))))}"
                  f" heard past its pauses; {int(life._pq.get('n_ret', 0))} returns, {int(life._pq.get('n_pause', 0))} pauses in all; P {P_:.1f} ticks):"
                  f" the shadow day (pace_sense 1) comes first", flush=True)
        if isinstance(L.get("motor"), dict):                          # the later effectors' act_inv reliability (step R6), for those this anatomy declares
            adam_ = []
            for e_, st_ in zip(life.anatomy.motors, getattr(life, "motor", ())):
                mv_ = L["motor"].get(e_.name)
                if not isinstance(mv_, dict):
                    continue
                st_["inv_n"] = int(mv_.get("inv_n", 0))
                st_["fatigue"] = float(mv_.get("fatigue", 0.0))   # step R6h: its own fatigue
                ch_ = mv_.get("inv_ch")                           # the kappa correction's running agreement and chance
                if e_.inverse and isinstance(ch_, list) and len(ch_) == len(e_.factors):
                    st_["inv_ch"] = [[float(x_) for x_ in r_] for r_ in ch_]
                if e_.inverse and isinstance(mv_.get("inv_batch"), list):   # act_inv's pairs gathered for the next batch
                    st_["inv_batch"] = [(torch.tensor(p_[0], dtype=torch.float32, device=device), torch.tensor(p_[1], dtype=torch.float32, device=device), int(p_[2]),
                                         (None if p_[3] is None else [torch.tensor(q_, dtype=torch.float32) for q_ in p_[3]])) for p_ in mv_["inv_batch"]]
                pf_ = mv_.get("perf")                             # step R6h: its performance error's running means, when it declares one
                if st_.get("perf") is not None and isinstance(pf_, list) and [len(r_) for r_ in pf_] == [int(k_) for k_ in e_.factors]:
                    st_["perf"] = [[float(x_) for x_ in r_] for r_ in pf_]
                # ACT_PRED'S MOMENTS, SAVED BY R6 FIX 5 OR 6 (2026-09-24): their Adam's (R6 fix 5's two samples, "own" and "labels"; R6 fix
                # 6's one set, "lesson"); act_pred steps plainly since R6 fix 7, with no state to give back: not read (said once)
                if isinstance(mv_.get("moments"), dict) and any(mv_["moments"].values()):
                    adam_.append(e_.name)
                cf_ = mv_.get("inv_conf")
                if e_.inverse and isinstance(cf_, list) and [len(c_) for c_ in cf_] == [int(k_) for k_ in e_.factors] and \
                        all(len(r_) == len(c_) for c_ in cf_ for r_ in c_) and len(mv_.get("inv_kappa") or []) == len(e_.factors):
                    st_["inv_conf"] = [[[float(x_) for x_ in r_] for r_ in c_] for c_ in cf_]
                    st_["inv_kappa"] = [float(x_) for x_ in mv_["inv_kappa"]]; st_["inv_gain"] = float(mv_.get("inv_gain", 0.0))
                elif e_.inverse and "inv_m" in mv_:              # a save from before 2026-09-24: the critics' estimator, which read base rates as skill
                    print(f"load: {e_.name}'s act_inv reliability was saved as the critics' estimator (before kappa, 2026-09-24): it is earned "
                          f"again from its next act (it read {float(mv_.get('inv_gain', 0.0)):.3f})", flush=True)
                elif e_.inverse and cf_ is not None:             # counts of other joints than this anatomy declares
                    print(f"load: {e_.name}'s act_inv confusion was saved for other joints than its declaration {list(e_.factors)}: it is "
                          f"earned again from its next act", flush=True)
            if adam_:
                print(f"load: act_pred's and the corrections' moments were saved by R6 fix 5 or 6 (2026-09-24, their Adam's) for {adam_}: "
                      f"not read, act_pred steps plainly since R6 fix 7 (no state)", flush=True)
        if isinstance(L.get("utt_felt"), dict) and int(life.cfg.get("tag_trace", 0)):   # defect 7 (tag_trace, R7c): the felt entries' mean
            life._utt_felt_m = float(L["utt_felt"].get("m", 0.0)); life._utt_felt_n = int(L["utt_felt"].get("n", 0))
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
        if len(life.anatomy.effectors) > 1:                         # THE DAY (A70; the module's doc), for a body with motor effectors
            day_ = blob.get("day")
            if isinstance(day_, dict) and day_.get("anatomy") == _anatomy_sig(life.anatomy):
                life._day_back(day_, born_)
            elif isinstance(day_, dict):
                print("load: the save's day was lived by another anatomy (its channels or joints): not read; the working state begins as a "
                      "night leaves it", flush=True)
            elif isinstance(L.get("motor"), dict):
                print("load: the save holds no day (a motor body's save from before A70, 2026-09-25): the working state begins as a night "
                      "leaves it (no cry, unit or chunk under way, no act last tick, no foresight, the cord's counts from zero, no orienting "
                      "cue seen before), the stream from the load's seed and the optimizers' moments born again: R6h's rule for older "
                      "saves", flush=True)
        return life

    def _day_back(self, day, born=()):
        """THE DAY GIVEN BACK (A70; the module's doc): the random stream's state, each optimizer's moments under this load's settings, the
        working attributes (a buffer kept at this life's length), and each motor effector's working state beyond life["motor"]. A working
        attribute the load itself made after the birth of this life (`born`: the attributes it was born with) that the saved life did not
        have is taken away again (the store's two day marks, read as absent: the day's own are the saved life's)"""
        self.gen.set_state(day["gen"])
        for k_, st_ in (day.get("optim") or {}).items():
            o_ = getattr(self, k_, None)
            if isinstance(o_, torch.optim.Optimizer):
                sd_ = o_.state_dict(); sd_["state"] = st_; o_.load_state_dict(sd_)
        life_ = day.get("life") or {}
        for k_, v_ in life_.items():
            cur_ = getattr(self, k_, None)
            if isinstance(cur_, collections.deque) and isinstance(v_, collections.deque) and cur_.maxlen != v_.maxlen:
                v_ = collections.deque(v_, maxlen=cur_.maxlen)
            setattr(self, k_, v_)
        for k_ in [k_ for k_, v_ in vars(self).items() if k_ not in life_ and k_ not in born and k_ not in DAY_NOT
                   and not isinstance(v_, torch.optim.Optimizer) and not callable(v_)]:
            delattr(self, k_)
        for e_, st_ in zip(self.anatomy.motors, self.motor):
            st_.update((day.get("motor") or {}).get(e_.name) or {})

    @classmethod
    def birth(cls, tok, device="cpu", d=256, layers=6, heads=4, window=64, cfg=None, seed=0, save_path=None, world=None):
        torch.manual_seed(int(seed))
        anatomy = anatomy_for(tok, cfg)                 # the body's anatomy (a tokenizer's: the diary's); built with no draw, before the organs
        organs = Organs(anatomy.vocab, d=d, layers=layers, heads=heads, window=window, birth_act=float((cfg or {}).get("birth_act", PHYSIOLOGY["birth_act"])),
                        channels=anatomy.channels, effectors=anatomy.effectors, born_seed=seed,   # a later channel's forecast head and a later effector's
                        cerebellum=cerebellum_spec(anatomy, cfg),   # organs built last (steps R4, R5; their tables from the body's seed), then the
                        events=anatomy.events,                      # cerebellum when its switch is on (step R6c), the event lines' striatal line (R7a),
                        amygdala=amygdala_spec(anatomy, cfg),       # the amygdala when its switch is on (R7d, the last), recall's maps (R7f);
                        recall=recall_spec(anatomy, cfg))           # the diary declares none of them
        return cls(organs, anatomy, cfg=cfg, device=device, seed=seed, save_path=save_path, world=world)
