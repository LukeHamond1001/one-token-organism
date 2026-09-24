"""the physiology table (moved from body/life.py, review 2026-09-22 section 4, step 2): `PHYSIOLOGY`, every disclosed constant,
grouped by organ; and `SWITCHES`, the core refactor's defect-fix switches, declared and off by their absence (docs/SIM_DESIGN.md 8.4). body/life.py re-exports it (`from body.life import PHYSIOLOGY` holds). The served body's effective set is its save
plus ops/BASE_FLAGS.txt (ops/served_cfg.py prints it); the history of every value is in BODY_SPEC.md's appendix and ITERATIONS.md."""

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

# THE CORE REFACTOR'S SWITCHES (docs/SIM_DESIGN.md 8.4; the defect fixes of ops/review_2026-09-22.md section 1, each a switch): declared
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
)
