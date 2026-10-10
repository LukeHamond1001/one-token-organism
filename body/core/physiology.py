"""the physiology table (moved from body/life.py, review 2026-09-22 section 4, step 2): `PHYSIOLOGY`, every disclosed constant,
grouped by organ; `SWITCHES`, the core refactor's defect-fix switches, declared and off by their absence (docs/SIM_DESIGN.md 8.4); and
`MOTOR`, the motor timing part's constants (step R6; step R6h's movement units, act_inv's batches, the kappa correction and fatigue per
effector among them), absent from a body's cfg unless given; `CEREB`, the cerebellum's switch and constants (step R6c), absent likewise;
`REFLEX`, the born patterns summed at the cord and the born biases (step R6h: the spinal pattern generator, the born cry, orienting,
the VOR), absent likewise; `FRAMES`, the body in frames (step R7: the frame writes, the event ends, the tick's record), absent
likewise; `AMYG`, the amygdala's switch and constants (step R7d), absent likewise; and `SLEEP`, the night over frames and the live,
dark night (step R8: the tape, the episodes, the twitches), absent likewise. body/life.py re-exports PHYSIOLOGY
(`from body.life import PHYSIOLOGY` holds). The served body's effective set is its save plus ops/BASE_FLAGS.txt (ops/served_cfg.py
prints it); the history of every value is in BODY_SPEC.md's appendix and ITERATIONS.md."""

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
    night_keep_bands=0,   # the slow bands are not zeroed at night. The sim: 1, the served value; A71 (the PFC study): matured by use, born at full strength
    night_ticks=0,
    utt_cap=4096,   # the utterance memory replayed at night, whole utterances
    utt_entry="flat",   # THE TAG AT ENTRY (2026-09-22, item 50): "flat" = every utterance enters the night's draw at 1.0; "felt" = at the mean of its symbols' write strengths (surprise x (1 + |dopamine|), the store's own law) over their running mean, so the novel and the rewarded are replayed more (tagging at encoding)
    utt_entry_tau=64,   # the running mean's horizon for the felt entry, in utterances
    night_dev="",   # THE NIGHT'S DEVICE (2026-09-22): "" = the body's own; e.g. "mps" = the cortex moves to the GPU for the night's lessons (NREM and REM) after the dreams are drawn and comes home before the value replay, the fade and the save; the day stays where it is
    dream_source="store",   # "utterances" = the night dreams the utterances heard whole; "store" = pattern completion from the store
    dream_who=0,
    dream_tag=0,
    dream_pair=0,
    dream_gap=1,
    dream_old_share=0.0,
    dream_corpus_n=0,       # THE NIGHT READS (2026-09-18, item 44): this many sentences of a corpus among each night's dreams (0 = off)
    dream_corpus_file="",   # the corpus, a text file; read as the parent reads it (lower case, no commas, sentences of the window's length)
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
    rem_limbs=0,                  # A132 (the sim's brain sprint): the limbs dream: in REM over frames each motor effector's act along the
                                  # free run is its imagined one (body/core/sleep.py _rem_limb_act), the seed positions the day's own acts
    # --- the event's end (the offset after the world's quiet) and the marks ---
    offset_ticks=8,
    offset_form="settle",   # "settle" = the offset fires when the surprise settles (offset_fast/offset_slow, offset_settle); "count" = after offset_ticks
    offset_foresee=0.0,     # THE QUIET FORESEEN (2026-09-20): under "settle" the utterance is also perceived complete when the readout's probability of the rest reaches this (the cortex expecting the quiet); 0 = off
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
    gate_quiet_tau=0,  # THE BABBLE DRIVE (item 41): the spontaneous floor rebuilds from 0 toward gate_floor over this many ticks of the world's silence; 0 = off
    gate_quiet_sure=0.0,   # THE SURE PROPOSAL (item 41): under the drive the floor is whole at once when the forecast's norm reaches this (an answer waits for no quiet); 0 = off
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
    gate_ear_decay=0.0,   # THE EAR'S TRACE (2026-09-19): the ear's world input persists between the world's symbols, decaying by this each tick (0 = the symbol this tick only); a slow typist's pauses leave the ear ringing
    gate_ear_gain=1.0,    # THE EAR'S GAIN (2026-09-20): the ear's world input to the learned gate, scaled; the gate's weight on it is built over weeks and moves little in days
    gate_ear_release=1,   # THE EAR'S RELEASE (2026-09-20): at an utterance perceived complete the ear stops ringing this many ticks on: 1 = at the end's own tick (the 19th's form), 2 = the tick after (the forecast is the rest at the end's tick and the reply the tick after: the reply's readiness), 0 = never, the ring fading by gate_ear_decay
    gate_yield=0.0,       # THE YIELD (2026-09-20): past its slot after the world's line (gate_yield_after ticks), the learned gate is held by this much, the hold fading over gate_quiet_tau as the babble drive returns; 0 = off
    gate_yield_after=40,  # the slot for its turn after the world's last symbol, in ticks
    gate_turn=0,          # THE TURN'S READINESS (2026-09-20): under the babble drive the floor is whole for the slot (gate_yield_after ticks) after a world utterance perceived to have ended by the settle law; a pause the count ended opens no turn; 0 = off
    gate_turn_floor=0.0,  # the spontaneous floor inside the slot after a foreseen end (2026-09-20 evening): the readiness to reply stronger than the resting floor; 0 = gate_floor
    # --- THE SENSED TURN-TAKING (pace_sense, 2026-09-23): each reflex that counted ticks becomes a percentile of the partner's own silences,
    # learned one heard event at a time (running quantiles in log units); the body's own latency stays counted in cortex steps ---
    pace_sense=0,         # 0 = off; 1 = shadow (the partner's pace tracked, the rules computed and logged, behaviour unchanged); 2 = live (M1-M5 replace
                          # offset_foresee and the surprise law, offset_ticks' count, gate_ear_decay/gain, gate_yield, the babble drive, the turn and the sure proposal)
    pace_eta=0.05,        # the trackers' step in log units per heard event: a new partner followed within about 1/eta (twenty) events in each tracker's
                          # fast direction (P up, R_lo down, R_hi up), at eta x min(p, 1 - p) a step in its slow one; R_lo and R_hi first settle as the
                          # sample quantiles of the first 1/eta returns heard
    pace_pause_p=0.99,    # P, the pause outlasted: this quantile of the partner's silences under R_lo (one pause in a hundred taken for an end)
    pace_ret_lo=0.05,     # R_lo: this quantile of the partner's returns (its quickest usual return: the child's slot, and what counts as a pause)
    pace_ret_hi=0.95,     # R_hi: this quantile of the partner's returns (its usual longest silence: beyond it the child is alone)
    ready_law=1,          # M4 the reply ready: 1 = the held ear released when the forecast has settled off the rest (cortex steps); 0 = gate_ear_release ticks after the foreseen end
    ready_ratio=0.5,      # M4: settled when the forecast's change is at most this fraction of its largest change since the end (the settle law's own ratio)
    pace_fore_q=0.0,      # M1 BY ITS OWN MEASURE (2026-09-23): >0 = the end is also foreseen when the readout's probability of the end symbol is above this quantile of its highest in each of the partner's mid-line pauses (a running quantile, like P); 0 = the argmax alone
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
    fast_input="band",   # the fast critic's input: "striatum" = a delay line of the stream's last events through a born expansion; "band" = the dopamine band's state. The sim: "striatum", the served value; A71 (the PFC study): matured by use, born at full strength
    stri_k=8,   # the sim: 8, the served value; A71 (the PFC study): matured by use, born at full strength
    stri_m=1024,   # the sim: 2048, the served value; A71 (the PFC study): matured by use, born at full strength
    stri_quiet=0,   # the sim: 1, the served value; A71 (the PFC study): matured by use, born at full strength
    fast_rls=0,   # the sim: 1, the served value; A71 (the PFC study): matured by use, born at full strength
    fast_rls_forget=36000,
    fast_rls_prior=3.0,   # the sim: 0.3, the served value; A71 (the PFC study): matured by use, born at full strength
    fast_rls_every=64,   # the sim: 256 (SIM_DESIGN.md 7.2, 10: the critics' solves every 256 ticks, a sim constant; the served 64)
    wm=0,   # working memory: latches the line's expansion at a dopamine burst, clears at a reward or after wm_max ticks. The sim: 1, the served value; A71 (the PFC study): matured by use, born at full strength
    wm_burst=0.5,   # the sim: 0.5, the served value; A71 (the PFC study): matured by use, born at full strength
    wm_max=512,   # the sim: 512, the served value (ours, unsourced: 77 s, likely rarely binding, since each event end overwrites the slot); A71 (the PFC study): matured by use, born at full strength
    # --- the ventral critic (the slow prospect) ---
    vcrit_w=0.0,
    vcrit_gamma=1.0 - 1.0 / 1024,
    vcrit_ceiling="fixed",   # "fixed" = vcrit_w x reliability; "earned" = the reliability itself. The sim: "earned", the served value; A71 (the PFC study): matured by use, born at full strength
    vcrit_lambda=0.0,   # the sim: 1.0, the served value; A71 (the PFC study): matured by use, born at full strength
    vcrit_lr=0.0,
    vcrit_tau=0.0,
    vcrit_diff=0,
    vcrit_bands="5,6,7",   # the ventral critic reads these bands ("-" = none). The sim: "0,1,2" (SIM_DESIGN.md 7.2: the continuous scene through the fast ladder bands, a sim constant; the served "-"); A71
    vcrit_traces=0,   # the sim: 1, the served value; A71 (the PFC study): matured by use, born at full strength
    vcrit_clock=0,   # sleep pressure over the wake threshold as a critic input. The sim: 1, the served value; A71 (the PFC study): matured by use, born at full strength
    vcrit_center=1,   # the sim: 0, the served value; A71 (the PFC study): matured by use, born at full strength
    vcrit_auto=0,   # the sim: 1, the served value (the voice exactly as loud as it has proved right); A71 (the PFC study): matured by use, born at full strength
    vcrit_forget=0,   # the sim: 36000, the served value; A71 (the PFC study): matured by use, born at full strength
    vcrit_rls=0,   # the ventral head learns by recursive least-squares TD(lambda) from accumulated evidence. The sim: 1, the served value; A71 (the PFC study): matured by use, born at full strength
    vcrit_rls_delta=100.0,
    vcrit_rls_every=64,   # the sim: 256 (SIM_DESIGN.md 7.2, 10: the critics' solves every 256 ticks, a sim constant; the served 64)
    vcrit_norm_tau=0,   # the sim: 36000, the served value; A71 (the PFC study): matured by use, born at full strength
    vcrit_rls_prior=0.0,   # the sim: 3.0, the served value; A71 (the PFC study): matured by use, born at full strength
    vcrit_norm_wake=0,
    # --- the actor and the planner: what to say at a word's start ---
    actor=0,
    actor_lr=0.02,
    actor_slow_lr=0.0,   # A142/C153 (2026-09-30): the actor's synaptic tag: each later effector's eligibility summed at the tag's reach
                         # (1 - 1/tag_reach, 64 ticks) and captured by phasic dopamine each tick (Frey and Morris 1997), so an act's credit
                         # reaches back a minute, not 16 ticks; 0 off (the diary); the G1 sets 1e-3 (ours)
    actor_beta=1.0,
    actor_forget=36000,
    dopamine_adapt=0,     # A172 (2026-10-02): the actors' dopamine over its running RMS (Tobler, Fiorillo and Schultz 2005: adaptive coding); 0 off
                          # (the diary, which has no actor); the G1 sets 1 (ours)
    dopamine_adapt_tau=256,     # ... the RMS's horizon, ticks (38 s: the adaptation within a block of trials, Tobler et al. 2005; ours)
    dopamine_adapt_floor=1e-4,  # ... the mean square's floor (an RMS of 0.01 at least: a silent line amplifies nothing without bound; ours)
    value_forget=0,      # A146: the ladder's value heads' forgetting (1 - 1/value_forget a tick; 0 = none, the diary's)
    actor_form="add",   # "chunk" = one act per word, the letters inside not choices; "plan", "select", "add" the earlier forms
    actor_margin=4.0,
    chunk_max=12,   # a word runs as a motor program for at most this many symbols
    chunk_gate=0,   # THE CONTINUATION GATED (2026-09-22, item 50): 1 = inside a word the learned gate decides at every symbol whether the program goes on (the basal ganglia's stop pathway can halt an action under way), so the face's credit reaches the choices it followed; 0 = a word runs unchosen once begun (p 1, no credit)
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

# THE CORE REFACTOR'S SWITCHES (docs/SIM_DESIGN.md 8.4; the defect fixes of ops/review_2026-09-22.md section 1, each a switch; defects 1
# and 6 since step R7c): declared
# here and OFF BY THEIR ABSENCE. A life's cfg holds one only when it is set (`cfg.get(name, off)` reads it), so the language body's
# constants, its save and its pinned digests stay what they were while the switch is off (8.3: the language life gains no key); a body
# born with one on (the sim is born with them on, 5.3) keeps it in its save. Adopting one on the language body is a measurement on a
# copy, applied at a boundary (8.6). Life.__init__ and serve.py know them as they know PHYSIOLOGY's keys.
SWITCHES = dict(
    # defect 4 (step R5): 1 = the gate's lesson and its synaptic tag take the gate's own draw as the act, so a word that ends in a
    # sampled rest, or a chunk's continuation that is the rest, is no longer recorded as the gate saying no (acted 0 at p 1: an
    # eligibility of -1 on a tick the gate never decided); a chunk's letters run without the gate (chunk_gate 0) are its draw at p 1
    gate_own_draw=0,
    # defect 5 (step R5): 1 = the actor's eligibility trace (and the chooser's) decays by dopamine's discount every tick, not once per
    # act, so its credit no longer smears across about sixteen words and through the silences between them
    actor_trace_tick=0,
    # defect 8 (step R5): 1 = the gate's credit sums the dopamine from the tick after the act on (the act's own consequences), not from
    # the act's own tick, whose dopamine was computed before the act (unbiased, but about 21 percent more noise)
    elig_from=0,
    # defect 1 (step R7c): 1 = the tired memories recover: the waking read's recovery toward rest is written into the store's own
    # availability in place, as its tiring is; without it the recovery went into a new tensor, which the next write of a new slot
    # dropped (FastStore's views), so the store's buffer kept the tiring of every read and none of the recovery, and the most-read
    # memories were pushed down all day (about -13.8 logits; ops/review_2026-09-22.md item 1: "gives exactly the behaviour of the Store
    # used before 09-19")
    tire_recover=0,
    # defect 6 (step R7c; SIM_DESIGN.md 7.4 "the tag reaches back", 10's `tag_trace`): 1 = the received tag reaches back onto the felt
    # entry of an utterance (utt_entry "felt"): its entry is its symbols' mean write strength times (1 + T), T the largest received tag
    # (the reward felt this tick, min(the judgment's clip, the sum of |term| over the sources that reach the amygdala)) over the
    # utterance's ticks and, reaching back, over the 64 ticks after its end at dopamine's discount a tick (later tags raise the entry
    # already made); over a running mean of utt_entry_tau entries, saved with the body and bias-corrected (defect 7, the felt mean born
    # cold and never saved, fixed with it). Before, the felt entry saw only the dopamine of the tick before each heard symbol, and the
    # typist never typed within a tick of a smile: smiles tagged nothing (ops/review_2026-09-22.md items 6 and 7). Off; on the language
    # body it is measured on a copy first (SIM_DESIGN.md 8)
    tag_trace=0,
)

# THE MOTOR TIMING PART'S CONSTANTS (the core refactor's step R6, docs/SIM_DESIGN.md 5.4 and 5.8; body/core/timing.py): declared here
# and ABSENT FROM A BODY'S CFG unless given (`cfg.get(name, MOTOR[name])` reads them), so the language body, which has no motor
# effector, gains no key (8.3, item 4) and its pinned digests stay what they were. A body with motor effectors (the sim) may set them
# at birth and keeps them in its save. Life.__init__ knows them as it knows PHYSIOLOGY's keys and the SWITCHES.
MOTOR = dict(
    # act_inv's online rate (Adam, one step a tick of its own act): the inverse model learns from the body's own acts from birth
    act_inv_lr=1e-3,
    # act_inv's running reliability: Cohen's kappa per joint over its running confusion (its label against its efference copy), the
    # counts decaying over this many own acts (since 2026-09-24; before, the critics' estimator's samples, one per setting of each
    # joint of an act, so the memory in acts was this over the joints' settings); zero until 64 acts
    act_inv_tau=8192,
    # ACT_PRED'S PLAIN STEP (R6 fix 7; body/core/timing.py GatedDescent): act_pred and its correction learn by plain gradient descent,
    # the step per unit of the lesson's gradient act_pred_rate x sqrt(d) x the waking rate (live_lr; d act_pred's fan-in, the stream's
    # width); each element's step bounded by act_pred_bound x sqrt(d) such steps. Derived and measured in GatedDescent's docstring
    act_pred_rate=2.0,
    act_pred_bound=1.0,
    # --- STEP R6h (SIM_DESIGN.md 3.5, 3.6, 10; each absent, R6's law, unless given: the sim is born with them, SIM_CFG in
    # body/sim/anatomy.py) ---
    # THE MOVEMENT UNIT'S PERSISTENCE MARGIN (3.6; ours, disclosed): under chunk_gate a unit under way holds its act joint by joint, a
    # joint taking another setting only where the choice's logits (the proposal at the readout's sharpness, the striatal actor's bias
    # and the born orienting bias in them) prefer it to the held setting by more than this; None: R6's continuation, act_pred's best
    # guess (each joint's argmax, which at birth, act_pred knowing nothing, is a fresh random act every tick: 3.6's evidence, 0 rolls).
    # The sim: log 4 (a setting four times as likely as the held one takes the joint). No roll count set it; none will tune it
    unit_margin=None,
    # ACT_INV'S LESSONS BATCHED (3.6): one step of its optimizer every this many ticks, on the mean over the pairs gathered since (each
    # labelled, and its reliability updated, by act_inv as it stood before the step); 1: R6's step a tick. The sim: 8 (unbatched the
    # lessons cost 4-6 ms a tick at the humanoid's size, section 9)
    act_inv_every=1,
    # THE KAPPA CORRECTION (R6's verifiers' finding, 6d6d246's proposal; 8's R6h row): act_inv's reliability takes each label's chance
    # agreement from the act's own choice, the probability the choice gave the label's setting (a drawn act's per-joint probabilities,
    # conditioned on the draw not being the rest; 1 or 0 for an act chosen without a draw: a movement unit's held act, a chunk's
    # continuation), kappa_j = (p_o - mean chance) / (1 - mean chance) over the same horizon; 0: R6's Cohen's kappa, its chance from
    # the pooled rates (which reads a label following the regime's mode as skill when the acts' rates shift). The sim: 1
    act_inv_chance=0,
    sharp_per_joint=0,   # A140: each joint's decisiveness its own, earned by its kappa and its settings' variety (mouth._motor_sharp); the G1 sets 1
    habit_by_credit=0,   # A141 (C149): by day act_pred's lesson at each position is weighted clip(1 + G, 0, 1) as the night's replay weights it
                         # (G dopamine's credit over the following ticks, sleep.credit_after on the day's record): an act followed by net harm
                         # is not cloned into the habit by day either; the G1 sets 1 (life day 40: the day cloned every act at weight 1)
    sharp_earned_norm=0, # C148: the proposal's certainty (its norm) is earned too: a joint's logits read s^g x cosine x |pred|^g with g its earned
                         # exponent (rel x variety), so a joint that has shown nothing draws from the cosines alone, however sure the forecast of its
                         # own habit; the G1 sets 1 (life day 40: three arm joints at one setting 76 to 98% of ticks under sharpness 1.0 to 1.5)
    # FATIGUE PER EFFECTOR (3.5; section 10's fatigue): each motor effector's acts' cost is its own fatigue, recovering at the body's
    # half-life, read by its own gate and weighed by its own lesson; 0: R5's, every cost added to the body's one fatigue (which would
    # add up nine limbs' costs and silence the voice). The sim: 1
    own_fatigue=0,
    # A183 (2026-10-03): THE MOTOR GATES' HOMEOSTASIS (body/core/mouth.py _choose_effector). gate_ceiling 1: a motor effector's gate acts
    # with probability floor + (1 - 2 floor) sigmoid(z), at most 1 - gate_floor (rest is sampled as activity is); gate_scaling 1: a gate
    # whose logit stands past logit(1 - gate_floor) has its weights scaled down at the tag's reach (synaptic scaling, as the actors',
    # A147). 0: R5's gate, floor + (1 - floor) sigmoid(z), unscaled. The sim: 1 and 1
    gate_ceiling=0, gate_scaling=0,
)

# THE CEREBELLUM'S SWITCH AND CONSTANTS (the core refactor's step R6c, docs/SIM_DESIGN.md 7.5, A44 and C50; body/core/cerebellum.py):
# declared here and ABSENT FROM A BODY'S CFG unless given (`cfg.get(name, CEREB[name])` reads them), so the language body, whose anatomy
# declares no cerebellar interface, gains no key, builds no organ and runs none of it (8.3, item 4), and its pinned digests stay what
# they were. A body born with the switch on (the sim: SIM_DESIGN.md 10's switches at birth) keeps it in its save, and its anatomy
# declares what the world feeds the organ below the tick (body/core/anatomy.py `Cerebellar`). Every value is ours, after the sources
# named; C50 settles the expansion's size and sparsity and the rates from their sources before birth (the citations here are recalled,
# not re-read, for this build).
CEREB = dict(
    # the switch: 1 = the organs build the cerebellum (Organs(..., cerebellum=)) and the life sets its world's sub-tick hook (World.below)
    cereb=0,
    # the loop's period in sim time: the world calls the hook every 10 ms (5 physics steps of 2 ms: 15 a tick). SIM_DESIGN.md 7.5; ours,
    # the rate the real robot's computer runs it at beside the servo loop (100 Hz)
    cereb_period_s=0.010,
    # the granule layer's size: a born expansion of the mossy input about 27 times wider than it (4,096 for the humanoid's about 150
    # numbers); ours, after Marr 1969 and Albus 1971's expansion recoding (an adaptive filter's basis: Fujita 1982)
    cereb_granule=4096,
    # mossy fibres per granule unit: the granule cell's four dendrites (Eccles, Ito and Szentagothai 1967), Marr 1969's codon; the
    # in-degree found near optimal for such an expansion (Litwin-Kumar et al. 2017; Billings et al. 2014)
    cereb_fan_in=4,
    # the fraction of granule units active at each sub-step, held constant by the Golgi cells' inhibition (Marr 1969; Albus 1971): the
    # units above the threshold that leaves this fraction active pass their excess, the rest are silent; ours ("about 10%", 7.5)
    cereb_coding=0.1,
    # the limbs' Purkinje readouts' step, least mean squares normalized by the granule layer's activity (w += rate e g / |g|^2), each
    # sub-step: at a state held constant the lesson alone moves the torque toward what its teacher asks with a time constant of 1 / rate
    # = 100 sub-steps (1 s; with the leak below, 1 / (rate + leak) = 75 sub-steps), about 25 times the servo's damping time (0.04 s), so
    # the lesson is slow beside the loop that teaches it, the separation of time scales feedback-error learning assumes (Kawato and Gomi
    # 1992); ours
    cereb_rate=0.01,
    # the limbs' readouts' leak, each sub-step, on the synapses of the granule units active in it (w <- (1 - leak) w before the lesson):
    # leaky least mean squares (Widrow and Stearns 1985), the leakage an adaptive controller needs so its weights stay bounded when its
    # error cannot be reduced by what it learns (the sigma-modification: Ioannou and Kokotovic 1983); in the cerebellar cortex, parallel
    # fibre activity without a climbing fibre potentiates as their conjunction depresses (Lev-Ram et al. 2003; Jorntell and Hansel 2006),
    # so a synapse the teacher no longer drives drifts back. Without it the law is a pure integrator of the servo's correction at a
    # still state: a limb pressed on a table wound its readout without bound (-531 N m after a 60 s press, the servo's authority gone)
    # and a newborn's rested limb was locked in place within 1.5 s (the R6c verifier). At a still state the readout's torque follows
    # tau <- (1 - leak) tau + rate e, so a load held constant is carried rate / (rate + leak) by the readout at its asymptote and the rest
    # stays the servo's correction, and a torque no teacher drives fades with a time constant of 1 / leak sub-steps (3.0 s). The value is
    # rate / 3.0: 3.0 is the asymptotic compensation per unit of residual error of the two-rate model fitted to people adapting to a
    # constant force field (Smith, Ghazizadeh and Shadmehr 2006: A = 0.992, B = 0.02 and A = 0.59, B = 0.21 a trial; B / (1 - A)
    # summed, 2.50 + 0.51 = 3.01; their fitted values read from the paper for this build), so the readout carries 0.75 of a constant
    # load, as their subjects' adaptation did; only the ratio is taken, the pace stays the rate's (a trial's length is not a sub-step's).
    # The flocculus has none: its slip is always reduced by what it learns (the window's counter-shift moves the image) and its gain is
    # to be kept between the turns that teach it. Ours, after the sources named
    cereb_leak=0.0033,
    # the flocculus's step, once a tick (retinal slip is seen once a tick), least mean squares over its two regressors (the head's turn
    # for the gain, 1 for the offset) normalized by their power and the granule layer's activity: an offset held in a constant context is
    # learned with a time constant of 1 / rate = 20 ticks (3 s), the gain with 1 / (rate turn^2) ticks; ours
    cereb_vor_rate=0.05,
)

# THE BORN PATTERNS SUMMED AT THE CORD, AND THE BORN BIASES (the core refactor's step R6h, docs/SIM_DESIGN.md 3.5-3.7, 10, A43, A47, A48,
# C53, C54; body/core/cord.py): each a switch or a constant ABSENT FROM A BODY'S CFG unless given (`cfg.get(name, REFLEX[name])`), so the
# language body, whose effectors declare none of them, gains no key, runs none of it and keeps its pinned digests. The sim is born with
# every switch on (SIM_DESIGN.md 10's switches at birth; body/sim/anatomy.py SIM_CFG). The effectors declare where each acts (their
# `spg`, `cry`, `orient` and `vor`, body/core/anatomy.py); the constants here are the body's, the same for every limb.
REFLEX = dict(
    # THE SPINAL PATTERN GENERATOR (A48; 3.7; C54 closed): a half-centre generator per limb that declares one (Brown 1911; per-limb
    # rhythm generators, as the per-muscle oscillators that gave a simulated neonate its motor patterns: Kuniyoshi and Sangawa 2006),
    # summed at the cord with the limb's own act like the grasp (A35). ITS SHAPE, the measured newborn rhythm: each cycle is a short
    # movement, flexion for spg_flex ticks then extension for spg_ext ticks, then a pause; the movement's ticks are fixed and the pause
    # stretches, each cycle's length drawn from the body's seed (body/core/cord.py). 1 = on
    spg=0,
    # THE MOVEMENT (the lead's decision on C54): flexion 2 ticks (0.30 s): newborns' flexion lasts "a little over 300 msec" in both kicks
    # and steps (Thelen and Fisher 1982, Dev Psychol 18:760-775), about 320 ms at 2 and 4 weeks (Thelen and Fisher 1983, J Mot Behav
    # 15:353-372); extension 3 ticks (0.45 s): 420 ms in steps and 586 ms in kicks (Thelen and Fisher 1982). Fixed: the movement's phases
    # are temporally constrained (Thelen and Fisher 1983) and the pause is what varies (Thelen 1981, Dev Psychol 17:237-257: the pause's
    # coefficient of variation about 140%). At the body's tick of 0.15 s (SIM_DESIGN.md 10), as every constant here is written in ticks
    spg_flex=2,
    spg_ext=3,
    # THE CYCLE (the lead's decision on C54): each cycle's length, the movement and its pause, drawn per cycle from the body's seed from a
    # log-normal of mean 3.56 s and standard deviation 1.93 s: newborns' kicking at birth, 3.56 +- 1.93 s (Hinnekens et al. 2023, eLife
    # 12:e87463, n = 15); their stepping agrees: 3.29 +- 1.21 s there, 3.03 +- 1.07 s (La Scaleia et al. 2018), about 3.5 s (Sylos-Labini
    # et al. 2017), 0.34 Hz (Dewolf et al. 2022). Clipped to 1.0-8.5 s: per-infant means from under 1 s to over 8 s (Thelen, Bradshaw and
    # Ward 1981), Hinnekens et al.'s per-infant range 1.19-8.51 s. The log-normal's own mean and SD are the sources' (sigma 0.508, the
    # median 3.13 s); the clip moves its 1.2% below 1.0 s and 2.5% above 8.5 s to the bounds, so the drawn cycles average 3.514 s with
    # an SD of 1.742 s (the clipped law's moments, exactly), the pause 2.76 s on average, its coefficient of variation 0.63 (written
    # down: below Thelen 1981's about 1.4; the cycle's moments are Hinnekens et al.'s, never fitted to the pause's). In ticks of 0.15 s,
    # as written: 23.73, 12.87, 6.67 and 56.67
    spg_cycle=3.56 / 0.15,
    spg_cycle_sd=1.93 / 0.15,
    spg_cycle_min=1.0 / 0.15,
    spg_cycle_max=8.5 / 0.15,
    # its amplitude A: the limb's gate's p_act x this, rad a tick, + along the flexion joints' senses in the flexion (3.7: the gate's
    # tonic readiness drives it, as the brainstem's drive enables the cord's generator; at the born p_act 0.2875, 0.026 rad a tick, about
    # a small step's third); ours (the design's), kept from R6h. IN THE EXTENSION THE STEP IS -A x spg_flex / spg_ext (2A/3), THE LEAD'S
    # DECISION (2026-09-25): a kick is a flexion and a return, and the sources give the phases' durations (above), not their excursions,
    # so the extension's 3 ticks return the 2A the flexion's 2 moved and a cycle's net excursion is zero. R6h and C54 stepped A in both
    # phases, which moved the targets A toward extension every cycle. The durations are unchanged; no posture term is added
    spg_amp=0.09,
    # THE BORN CRY (A47; 3.7; Jurgens 2002: the cry is innate and patterned by the periaqueductal grey): on a pain tick, or while the
    # charge is below cry_charge, the tract's cry posture is added to its targets in breath groups, its own act overriding it
    # articulator by articulator. 1 = on
    cry=0,
    # the charge line: ours (C53), below the parent's feeding line (0.35), so the cry is the body's own alarm, never timed to her
    cry_charge=0.2,
    # its breath groups, in ticks: expiration 5 (0.75 s), inspiration 2 (0.30 s), from pain cries of healthy full-term newborns recorded in
    # their first two weeks: 57 breaths a minute, the inspiratory phase 27% of the cycle (Robb, Sinton-White and Kaipa 2011, Int J
    # Pediatr Otorhinolaryngol 75:1265), so a cycle of 1.05 s, 0.77 s out and 0.28 s in: 5 and 2 ticks of 0.15 s (a cycle of 7, its
    # inspiration 29%). Its expiration ends early when the reservoir runs empty (the tract's breath left at 0: its own physics)
    cry_expire=5,
    cry_inspire=2,
    # THE BORN BREATH (A173, 2026-10-02; the respiratory rhythm of the brainstem, the pre-Botzinger complex: Smith, Ellenberger, Ballanyi,
    # Richter and Feldman 1991, Science 254:726): while the tract does not cry, its lungs are driven in a tidal cycle below the gate,
    # breath_expire ticks pushing at breath_amp of the lungs' range, then breath_inspire ticks drawn back; the own act's lungs step adds to
    # it as the cry's does (the world's sum under the clip). A newborn breathes 30 to 60 times a minute (Fleming et al. 2011, Lancet
    # 377:1011: the median 44 at 0 to 3 months), a cycle of 1.4 s: 5 and 5 ticks of 0.15 s. The glottis is not touched: quiet breathing
    # is silent (the glottis open at the passive rest); a glottis the child's own act presses on an expiration phonates. 1 = on
    breath=0,
    breath_expire=5,
    breath_inspire=5,
    breath_amp=0.4,      # the lungs' drive on the expiration, a fraction of their range (4.8 cmH2O of the tract's 12 at full drive: above a
                         # phonation threshold pressure of 2 to 4 cmH2O, under the cry's 0.6; ours)
    # THE NEWBORN'S EXPIRATORY BRAKING (A179, 2026-10-02): the glottis narrowed through the expiration and opened again on the inspiration,
    # a step of breath_brake of its range a tick at the cord (the laryngeal adductors' expiratory activity that holds the newborn's lung
    # volume: Kosch and Stark 1984, J Appl Physiol 57:1126; Harding 1984, Annu Rev Physiol 46:645; the audible grunt and sigh of the
    # first days), where the anatomy declares the glottis (cry["glottis"]). The own act's glottis step replaces it as the cry's is replaced
    # (the glottis the child's where it acts). Why: on the day-66 copy the voice's inverse model had had 272,759 pairs and sat at chance on
    # nine articulators of ten (kappa -0.01 to 0.04): 95% of its acts sounded nothing, so the ears' change labelled nothing. 0 = off
    breath_brake=0.0,
    # ORIENTING (3.7, A43; Goren 1975, Johnson and Morton 1991: newborns prefer faces; Muir and Field 1979: they turn toward sounds;
    # Johnson 1990: they orient to peripheral visual onsets through the subcortical route): a born bias on the proposals of the joints
    # an effector declares (the gaze's yaw and pitch, the waist's yaw) toward each cue the anatomy declares (a face-like blob in the
    # periphery, a sound's side, a sudden local change), and a born gate input "a cue appeared". 1 = on
    orient=0,
    # the bias, in logits, on each setting stepping toward the cue (and against it on each stepping away; the hold none), times the
    # orienting gain (the amygdala's, exactly 1 at birth: 7.4, R7e): log 4, ours: the size of the movement unit's persistence margin,
    # so a cue can turn a held unit toward it (its toward-setting gains log 4 and the held away-setting loses log 4: 2 log 4 past the
    # margin) and a proposal the cortex has learned as strongly can outweigh it; never set on a looking rate
    orient_bias=1.3862943611198906,
    # A186 (2026-10-03): THE BORN SACCADE (body/core/cord.py _orient_saccade): on the eyes (an effector that orients and carries the VOR) a
    # cord step toward the leading cue, orient_saccade_gain x its offset from the fovea's centre (0.5: newborns' saccades fall short and
    # reach a target in steps of about half the distance, Aslin and Salapatek 1975), at most orient_saccade_max rad a tick (0.20: the
    # gaze's big step on the G1, ours), times the orienting gain. 1 = on; off at birth for every body that does not say so
    orient_saccade=0,
    orient_saccade_gain=0.5,
    orient_saccade_max=0.20,
    # A216 (2026-10-10): THE BORN APPROACH (body/core/cord.py _approach_step): on an effector that declares `approach` (the G1's locomotor
    # command), a cord step toward the leading standing cue it names (her face): the turn's step approach_turn_gain x the cue's bearing
    # from the body (the eyes' turn in the head plus the cue's offset from the fovea, rad), at most approach_turn_max a tick; and, with
    # the cue within approach_zone of straight ahead, the speed's step approach_go a tick; both times the orienting gain (the amygdala's
    # toward or away). The superior colliculus's crossed descending road turns head and body toward a target and drives approach, its
    # uncrossed road defence (Dean, Redgrave and Westby 1989, TINS 12:137); the newborn turns its head toward its mother's voice and face
    # (Muir and Field 1979; Goren, Sarty and Wu 1975) and the infant's approach to the caregiver is the attachment system's set goal
    # (Bowlby 1969). The sizes are ours, in the command's own units (world.LOCO_SETTINGS: m/s and rad/s a tick, decaying): a bearing of
    # 0.4 rad steps the turn at its cap, which the command's decay holds at 0.6 rad/s; facing her the speed settles at a third of a
    # metre a second. 1 = on; off at birth for every body that does not say so
    approach=0,
    approach_turn_gain=0.05,
    approach_turn_max=0.02,
    approach_go=0.01,
    approach_zone=0.35,
    # THE VOR (3.7, A23; brainstem, present at birth): the gaze's window counter-turns by the torso gyro's rotation in each camera's frame
    # (the world applies it through the tick at its samples of the gyro, as it applies the servo law); the body's born gain and the
    # quick phase's jump back, a fraction of the axis's reach, handed to the world with the tick's acts (Acts.vor); the flocculus's
    # learned correction and offset come through the sub-tick hook (7.5). 1 = on
    vor=0,
    vor_gain=1.0,                 # born gain 1 (3.7; ours: the reflex's ideal, which the flocculus tunes)
    vor_quick=0.5,                # at the reach, a jump back of half the reach, in the direction of the turn (A23; ours)
)

# THE BODY IN FRAMES (the core refactor's step R7, docs/SIM_DESIGN.md 7.2, 7.4, 7.6, 8's R7 row, 10; body/core/frames.py): each a switch
# or a constant ABSENT FROM A BODY'S CFG unless given (`cfg.get(name, FRAMES[name])`), so the language body, which lives on the page
# alone, gains no key, runs none of it and keeps its pinned digests. The sim is born with every switch on (SIM_DESIGN.md 10's switches at
# birth; body/sim/anatomy.py SIM_CFG).
FRAMES = dict(
    # STEP R7b, THE FRAMES (7.4's "the fast memory's writes", 9's store, 10's "event end"): 1 = at each tick's end the frame's surprise
    # (each forecasting channel's error, the words' the tick's surprise and every later channel's its head's squared error to the code
    # that came, each scaled by its own running mean, their mean: all channels weighted equally), the event's end by today's settle law
    # on it (offset_fast, offset_slow, offset_settle: the first settled tick after one that was not), the store's marks there (the last
    # frame written ends an event, the next begins one), the surprise-gated write of the frame (its codes under the key of the stream
    # before it) and the tick's record (surprise, dopamine, tag, the net reward received: 16 B a tick, R8's tape). 0 = off
    frames=0,
    # the write gate's share: a frame is written when its surprise x (1 + tag) passes this running quantile of the same (7.4, 10; the
    # design's 0.9: about 2,400 frames in a life day of 24,000 ticks, section 9's store)
    write_q=0.9,
    # the running quantile's step, in log units, a tick: q <- q + eta (write_q - [ln x <= q]); its first 1/eta samples settle it as
    # their sample quantile, and nothing is written while it settles (the body's running quantiles are the sensed pace's trackers'
    # law and step, pace_eta; ours)
    write_eta=0.05,
    # each channel's forecast error's running mean, the rate 1 / min(n, err_tau) (the first ticks exact averages): the critics' own
    # statistics' horizon (fast_rls_forget, vcrit_norm_tau, 36,000 ticks; ours)
    err_tau=36000,
    # THE TAG'S REACH BACK, in ticks (SIM_DESIGN.md 7.4, "the tag reaches back": tag* = max over k < 64 of 0.9375^k x the tag k ticks
    # later, dopamine's own discount a tick; 0.52 of a smile's weight after 10 ticks, 0.28 after 20: risk 1's 7-25-tick delay): tag_trace's
    # reach onto an utterance (R7c), the store's later boosts after a frame's write (R7d), R8's tag* over the day's record; ours
    tag_reach=64,
    # STEP R7c, EACH CHANNEL'S FORECAST ERROR SCALED BY ITS OWN RUNNING MEAN (10's "forecast heads"): 1 = in the waking lesson each later
    # channel's head's squared error to the next born code is divided by that channel's running mean of its error (the frames' own,
    # above), so every channel's forecast teaches the stream about as much as another, whatever its code's size (the eyes' 1,536 numbers
    # and the charge's 2 weigh alike, as the frames' surprise weighs them); 0 = the sum as R4 built it. A channel with no running mean
    # yet is taken as it is. Needs `frames` (its running means are the frames' own)
    err_scale=0,
    # STEP R7f, RECALL INTO ACTION (7.6, A45; body/core/frames.py): 1 = a frame's key is the cortex's stream plus a heading (the trunk's
    # yaw integrated from the torso gyro since birth, the anatomy's `heading`; McNaughton et al. 2006), the stream's pattern-separated
    # direction and the heading's born code (cos, sin through two fixed unit rows from the body's seed) weighing alike, the words' cortex
    # key too; its value the frame's codes and the efference copy of every effector's act; each tick the store's nearest keys give back
    # their values through the store's own search (under key_form "cortex" the words' read itself, else a read of its own), and each
    # motor effector's proposal gains its map, born at zero, from the recalled act's embedding (the recall's part in its acts' rows) to a
    # score per setting, read back through its rows into the proposal, learned by act_pred's own lesson and plain step (Lengyel and Dayan
    # 2007: episodic control). Needs `frames`. 0 = off
    recall=0,
    novelty=0,                    # A127 (the sim's brain sprint): the novelty drive, body/sim/anatomy.Novelty: a frame the store keeps as
    goal_key=0,                   # A130 (the sim's brain sprint): the held word as a recall key, body/core/frames.py _goal_trace and
                                  # memory.py query_from (GOAL_TAU, GOAL_SCALE): the body's own last said word joins the frames' key
    ctx_key=0,                    # A128 (the sim's brain sprint): the held context as a recall key, body/core/memory.py query_from (CTX_TAU,
                                  # CTX_SCALE): the stream's direction integrated over CTX_TAU ticks joins the frames' key
    inner_speech=0,               # A137 (the sim's brain sprint): the inner word: the voice's sure, unsounded choice held as the goal word too
    imagine_key=0,                # A138 (the sim's brain sprint): waking imagination at an event's end (sleep.py _imagine): the imagined
    imagine_pav=0,                # frames' direction in the recall key; the amygdala's forecast on them in the gates' approach-and-avoid bias
                                  # new pays NOVELTY_GAIN; a switch of the body, off at birth, measured on a day copy
    vte_act=0,                    # A190 (2026-10-04): the deliberation weighs one effector's proposed act against its rest (sleep.py _vte_act_think)
    imagine_vte=0,                # A182 (2026-10-03): vicarious trial and error (sleep.py _vte_think): at each waking imagining a second future
                                  # from the same moment, the amygdala's forecast weighing the two, the body leaning toward the better one's
                                  # first acts (timing._timing_propose's _vte_term); needs the amygdala and rem_limbs; off at birth
    competence=0,                 # A181 (2026-10-03): the competence drive, body/sim/anatomy.Competence: at each event's end (frames) the body's
                                  # own acts' consequences becoming foreseeable pays COMPETENCE_GAIN x the event's effort x the progress of the
                                  # own-act forecast's skill against the naive forecast (frames._ferr_own_progress; A184); a switch of the
                                  # body, off at birth, measured on a day copy
    # STEP R7f, THE WORKING-MEMORY LATCH ON THE FRAMES' EVENT ENDS (7.6, 10's "event end"): 1 = working memory (wm) latches the striatal
    # expansion at the frames' event ends (R7b) in place of the utterances' ends; 0 = at the utterances' ends (the language body's).
    # Needs `frames`
    wm_frames=0,
)

# THE AMYGDALA (the owner's decision 2; the core refactor's step R7d; docs/SIM_DESIGN.md 7.4, 10, A16; body/core/amygdala.py): its switch
# and constants, ABSENT FROM A BODY'S CFG unless given (`cfg.get(name, AMYG[name])`), so the language body gains no key, builds no organ
# and runs none of it. The sim is born with it on (SIM_DESIGN.md 10's switches at birth; body/sim/anatomy.py SIM_CFG). Its discount is
# dopamine's own (the dopamine band's, 0.9375 a tick: no new time constant), its cap the judgment's clip (source 0's: 2), its horizon
# and the tag's reach back FRAMES' tag_reach (64): none of them is a constant of its own. Every value is ours, fixed before birth and
# never fitted to a rate (A16)
AMYG = dict(
    # the switch: 1 = the organs build the amygdala (Organs(..., amygdala=): m.amyg, built last) and the tick runs it after the critics'
    # lesson and the face organ, before the choice (`_amygdala`); 0 = none
    amyg=0,
    # ITS MEMORY: its evidence forgets 1 - 1/tau_a a tick, tau_a the ladder's clock at this band: band 6's, 4,096 ticks, about ten
    # minutes of life, about eight times its 526 inputs so the fit is determined; few-trial learning comes from least squares taking a
    # full step along a direction it has rarely seen, not from tau_a, which sets how fast it follows a drifting cortex and a changed world
    # (7.4, A16; the old valence learner's forgetting of 0.998 remembered about 500 ticks, fewer than its inputs)
    amyg_clock=6,
    # ITS PRIOR: the ridge R_ii = amyg_prior x tau_a x each input's running variance (its rate 1 / min(n, tau_a)), the level free: the
    # critics' own prior (fast_rls_prior 0.3, 36,000 ticks there; 7.4, A16: a prior of 1.0 was measured, an event line learned faster and
    # stream cues slower, and not adopted)
    amyg_prior=0.3,
    # ITS SOLVE, every this many ticks: W = (A + R)^-1 b, a Cholesky solve over the inputs that have varied (an input that never has is
    # left out, its weight 0: the minimum-norm answer), falling back to least squares (7.4: 0.69 ms a solve, 0.19 ms a tick measured)
    amyg_every=8,
    # ITS RELIABILITY: each head's running correlation of its forecast with the realized target (finalized tag_reach ticks later, or at
    # nightfall), the moments decaying over the critics' 36,000 ticks (the face organ's own measure), clipped to 0-1 and 0 until this
    # many finalized pairs exist (7.4, A16): at birth every forecast reaches nothing
    amyg_rel_tau=36000,
    amyg_pairs=64,
    # THE ORIENTING GAIN (R7e; 7.4 item 3, A16): each born orienting cue's bias times clip(1 + N, lo, hi), N the net valence the reliable
    # forecasts carry; exactly 1 at birth. The upper bound at most doubles the born pull; the lower is the owner's "toward or away" at its
    # weakest, a turn away never stronger than half the born pull toward (risk 15)
    amyg_orient_lo=-0.5,
    amyg_orient_hi=2.0,
    # APPROACH AND AVOID ON THE LIMBS (R7e; 7.4 item 4, Guitart-Masip et al. 2012: a Go bias toward good and a freeze toward bad): 1 = each
    # motor gate's logit gains amyg_pav_beta x clip(N, -amyg_pav_clip, amyg_pav_clip); BUILT AND OFF AT BIRTH (7.4 gives why: the gates
    # already learn from the same outcomes; a born Pavlovian bias impairs learning to hold still for reward and to act to avoid harm;
    # infants' fear-driven inhibition comes after mobility; it would add to risk 5), tried on a copy once an aversive head's reliability
    # reaches 0.2
    amyg_pav=0,
    amyg_pav_beta=1.0,
    amyg_pav_clip=2.0,
    # ITS WEIGHT'S FORM (A71, the lead's decision of 2026-09-25 on the PFC-maturation study, docs/audit/pfc_maturation.md: "amyg_pav switched
    # on after birth (7.4) is a change of body under A20"): "fixed" = R7e's law, the weight amyg_pav_beta; "earned" = BORN ON, ITS WEIGHT
    # amyg_pav_beta x THE LARGEST RELIABILITY OF THE AMYGDALA'S AVERSIVE HEADS (the heads of negative sign: the G1's face -, pain -, charge
    # -), which is 0 at birth and grows by use, as the long critic's voice is earned (Daw et al. 2005; the study's (b)): so the bias is born
    # whole and silent and has its say as far as the organ has proved it can foresee the bad, with no switch after birth. THE BUILDER'S
    # READING, for the lead: "the aversive head's reliability" read as the largest among the aversive heads (7.4's trial condition was
    # "an aversive head's rho reaches 0.2"). The sim: "earned" with amyg_pav 1 (body/sim/anatomy.py SIM_CFG)
    amyg_pav_form="fixed",
)

# THE NIGHT OVER FRAMES AND THE LIVE, DARK NIGHT (the core refactor's step R8, docs/SIM_DESIGN.md 3.6, 3.7, 5.4, 7.3, 7.4 item 2, 8's R8
# row, 9, 10; A46; body/core/sleep.py): each a switch or a constant ABSENT FROM A BODY'S CFG unless given (`cfg.get(name, SLEEP[name])`),
# so the language body, which lives on the page alone, gains no key, runs none of it and keeps its pinned digests. The sim is born with
# both switches on (SIM_DESIGN.md 10's switches at birth; body/sim/anatomy.py SIM_CFG). The night's length is PHYSIOLOGY's night_ticks
# (the night takes time: the critics discount across it, and a live night steps the world that many ticks); the entries fade as the
# store fades (store_fade); the night's passes keep PHYSIOLOGY's night_* and rem_* constants
SLEEP = dict(
    # STEP R8, THE NIGHT OVER FRAMES (7.4 item 2, 8's R8 row, 9's tape): 1 = each awake tick's frame as the cortex received it (every
    # channel's numbers at fp16, the words' symbol and whether the offset ended an utterance there, the voice's symbol and every motor
    # effector's act) is taped beside the tick's record (R7b's); at nightfall the day's tape is cut into episodes at the frames' event
    # ends, each episode's entry [its mean surprise x (1 + |dopamine|)] x (1 + T_e) over the bias-corrected running mean of entry_tau
    # entries, T_e the largest tag* over it (the tag reaching back, over the day's record); the night draws every episode of the day with
    # T_e >= 1 once, highest first, at most half the night, then the rest by entry; each dream is the episode's window of the cortex's
    # length ending at its peak tag (at its end below peak_floor), replayed as per-channel batches (every channel's forecast, the forward
    # half and act_inv at weight 1; every effector's acts as efference copies and act_pred's targets, act_pred's weighted clip(1 + G, 0,
    # 1) by the replayed dopamine's credit and stepped by its own plain step, never the night's Adam); REM on frames; the episodes kept
    # across nights, fading as the store does, the weakest giving way past episode_cap. Needs `frames`. 0 = off
    night_frames=0,
    # THE REVERSE VALUE SWEEP (A93, the lead's build 2026-09-26 on the owner's word "get this thing learning"; biology: after reward the
    # hippocampus replays the path backwards, more so the larger the reward, Foster and Wilson 2006, Ambrose, Pfeiffer and Foster 2016;
    # replay prioritized by gain, Mattar and Daw 2018): 1 = in the frames' night, before the value replay, each band's critic (the solved
    # fast head apart) takes the day's transitions from the tape's bands and the record's net reward and sweeps them BACKWARDS in time in
    # chunks of night_reverse_chunk ticks, one TD(0) step a chunk with the head as the earlier chunk left it, so a reward's value reaches
    # the acts before it in one night (risk 1, credit across the delay); the day's tagged episodes' windows are swept first (the gain),
    # then the whole day. Needs night_frames. 0 = off (the language body: its pins hold)
    night_reverse=0,
    night_reverse_chunk=64,
    # THE EPISODES' CAP, in ticks of tape (section 10's "episode cap"; 9's tape "inside the episodes (cap 24,000 ticks)"): the kept
    # episodes' windows hold at most this many distinct ticks of tape (about 184 MB at the G1's 3,834 numbers a tick in fp16); past it
    # the weakest episode (the lowest entry; of equals the oldest) gives way, as the store's weakest slot does. A life day's ticks; ours
    episode_cap=24000,
    # THE ENTRIES' RUNNING MEAN, its horizon in episodes (7.4 item 2: "over a running mean of 64 episodes (saved and bias-corrected:
    # defect 7's fix)"); ours (the utterances' felt entry's, utt_entry_tau 64)
    entry_tau=64,
    # THE WINDOW'S FLOOR (7.4 item 2: "An episode whose tag never reaches 0.1 is dreamt to its end"): an episode whose T_e is below it is
    # dreamt in the window ending at its last tick, else in the window ending at its peak tag; ours (the design's)
    peak_floor=0.1,
    # STEP R8, THE TWITCHES OF ACTIVE SLEEP IN THE LIVE, DARK NIGHT (3.7, 5.4, A46): 1 = when its world runs through the night (the
    # sim's: World.live_night) the night steps it dark for night_ticks ticks, every effector at rest, and in active sleep the born twitch
    # generator moves one joint at a time one small step (its sign drawn; the joint among every joint of the effectors that declare
    # `twitch`: the G1's waist, arms, hands and legs, never the tract or the gaze), a function of the body's seed and the night alone;
    # each twitch's pair (the body sense before and after, the act) teaches act_inv (its reliability updated first, the twitch's chance
    # the generator's own law) and the forward half, and the cerebellum learns below the tick wherever the world runs. A world that
    # does not run through the night pauses as the diary's does, and nothing twitches. Needs `night_frames`. 0 = off
    twitch=0,
    # THE TWITCHES' RATE, per tick of active sleep: about 10 a minute (Sokoloff, Hickerson, Wen, Tobias, McMurray and Blumberg 2020, Dev
    # Psychobiol 62:697-710, Figure 5a: twitches per minute of active sleep across all body segments; the sessions of infants under two
    # months lie at about 6 to 21, near 10 at their middle; the paper tabulates no single rate, so this is the figure read, C52), at 400
    # ticks of 0.15 s a minute: 10 / 400. Each tick of active sleep twitches with this probability, independently (the source's bursts,
    # about half its intervals under 1-2 s, and its hands' and feet's larger share are not reproduced: written down, C52); ours as read
    twitch_rate=10.0 / 400.0,
    # THE SLEEP CYCLE, in ticks (5.4: "Its REM phases, about half of it"): 47 minutes, the mean sleep cycle of healthy term infants (Stern,
    # Parmelee, Akiyama, Schultz and Wenner 1969, Pediatrics 43:65-70; as quoted in the review Promoting and Protecting Infant Sleep, Adv
    # Neonatal Care 2012; the primary read at C52), 47 x 400 = 18,800 ticks of 0.15 s. Each cycle begins in active sleep (the newborn's
    # sleep begins in active sleep; the changeover to quiet sleep at its beginning comes later in the first year: Tarullo, Balsam and
    # Fifer 2011, Inf Child Dev 20:35-46, citing Fagioli and Salzarulo 1982 and Hoppenbrouwers et al. 1982) and spends sleep_active of it
    # there: half (Roffwarg, Muzio and Dement 1966; "over half" in Anders et al. 1995), then quiet sleep. A night of 24,000 ticks is then
    # active sleep on its first 9,400 ticks and its last 5,200: 0.61 of it; ours as read
    sleep_cycle=47.0 * 400.0,
    sleep_active=0.5,
)
