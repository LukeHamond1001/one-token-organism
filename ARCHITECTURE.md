# ARCHITECTURE: the language body as it stands on 2026-09-23

This is the architecture of the served body, one section per mechanism, written for a researcher and for the robot work that
follows. It was read from the code at commit 4959064 (body/ is unchanged in the working tree), from the served flags
(ops/BASE_FLAGS.txt), from the ledger (ITERATIONS.md, items 1-52) and from the review of 2026-09-22 (ops/review_2026-09-22.md).
BODY_SPEC.md holds the older derivations and the history of each constant, but it was last changed on 09-17 and does not
describe items 41-52. Where this document and the code disagree, the code is right and this document is wrong.

## How to read it

**Where the served constants come from.** The served body's constants are the constants stored in its save, then overridden by
the flags it was launched with. A flag left out of the file keeps the save's value (item 39: "the served constants are what the
save says, and the flags are only a delta on it"). The defaults in `PHYSIOLOGY` (body/core/physiology.py, re-exported by body/life.py) apply only where neither the save
nor the flags set a value. `ops/served_cfg.py` prints the effective set, but it loads the save, and this document was written
without loading the served save. Twenty-six constants are set only by the save (the review's count, §4). Their values below come
from a print of the effective constants made on 2026-09-22 at 17:38. No flag has touched those keys since. Where they are named,
the values agree with watch2's birth record (data/watch/watch2/digest.txt, line 1) and with the ledger. In the tables, the
source column says **flags**, **save** or **default**.

**Status.**
- **SERVED**: on in the served body and acting.
- **SERVED, INERT**: the flag is set, but under the served constants the code path never runs.
- **OFF**: built and switched off by default. It is kept as an instrument, measured or waiting to be measured, and not served.
- **RETIRED**: measured and rejected, or replaced by something else. The code is kept as an instrument unless said otherwise.

**Scope**, from the humanoid inventory in ops/review_2026-09-22.md §3:
- **BODY-GENERAL**: the mechanism carries to another body unchanged.
- **LANGUAGE-SPECIFIC**: it depends on text: one symbol per tick, the space, a character table, or a typist's pace.
- **MIXED**: the law is general, but the unit it runs on is shaped by text.

**The laws the architecture is held to** (the user's standing rules):
- Reward comes only from the ground, never from a hand-written rule or a cheat.
- Mechanisms are biology's. Every constant is disclosed, and every mechanism is judged by whether it survives a change of body.
- A change is measured on a copy and reaches the served body only at a night's save.
- A night is kept whatever it does. REM stays on.
- There are no baseline runs and no side seeds, and everything runs locally.
- Progress is judged by sitting with the body.
- No value may be fitted to the environment's pace (item 50, 2026-09-22).

## The served body

| fact | value | source |
|---|---|---|
| the life | watch2, born the evening of 2026-09-06, seed 1. Its first days were lived under tools/archive/watch_life.py's scripted parent, then under the served typist. For a time its nights also dreamt story sentences (item 44), a channel removed in item 52 | data/watch/watch2/digest.txt; DIARY_BODY.md |
| size | 179M parameters; d 1024, 12 transformer blocks, window 64 ticks (the head count is in the save's `arch`) | the birth digest; README.md |
| symbols | 107 characters (data/tok_char.json): ids 0-10 are bookkeeping marks the mouth never says (`<pad>` is the rest, `<eot_human>` the turn's end), then digits, letters and punctuation | the tokenizer |
| clock | `--period 0.15` s a tick, about 0.17 s in practice (compute-bound); a day of 24000 ticks is about 67 minutes, a night about 25 minutes on the CPU (the nights of 2026-09-23 took 24 to 36 minutes) | flags; DEMO_SCRIPT.md; item 52; the page log's night rows |
| process | `python3 -m body.serve --load data/watch2.pt` on port 8020 with ops/BASE_FLAGS.txt; the typist chain (`body.teacher`) and ops/night_cycle.sh beside it | ops/serve_command.txt, ops/typist_chain_command.txt |
| store | FastStore, capacity 65536 slots | flags |
| tests | 91 test functions in body/tests/test_organs.py | counted |

## Summary

| # | mechanism | status | scope | ledger items |
|---|---|---|---|---|
| 1 | the cortex | SERVED | BODY-GENERAL (its input format LANGUAGE-SPECIFIC) | 8, 14, 15, 38, 40, 44, 49, 52 |
| 2a | the fixed lexicon and the readout | SERVED | LANGUAGE-SPECIFIC | 13, 36, 37, 41, 42 |
| 2b | the bands (the prefrontal ladder) | SERVED | BODY-GENERAL | 38, 46 |
| 3 | the hippocampal store | SERVED | MIXED | Landed, 1, 2, 10, 12, 16, 35, 37, 39, 40, 42, 46, 47 |
| 4 | the utterance memory | SERVED | MIXED | 6, 7, 38, 40, 46, 50, 52 |
| 5 | the night | SERVED | BODY-GENERAL (sleep by tick count is a robot blocker) | 6, 7, 14, 15, 38, 44, 46, 52 |
| 6 | the critics and dopamine | SERVED | BODY-GENERAL (the striatal input grows with the vocabulary) | 11, 21, 22, 38, 46, 50 |
| 7 | the mouth gate and its three-factor lesson | SERVED | BODY-GENERAL rule; one gate for the whole body | 41, 42, 45, 46, 47, 48, 50 |
| 8 | the actor (chunk form) and the chooser | actor SERVED; chooser RETIRED | the word unit LANGUAGE-SPECIFIC; deliberation BODY-GENERAL | 3, 4, 13, 18, 36, 50, 52 |
| 9 | feelings: mood, stress, fatigue, sleep pressure | SERVED | BODY-GENERAL | 8, 13, 37, 41, 42, 52 |
| 10 | the sensed turn-taking, pace_sense M1-M5 | SERVED (live since night 323's save) | MIXED | 50, 51, 52 |
| 11 | the timing reflexes it replaced | RETIRED; their flags are still set and inert, except the listening reflex's word cut | LANGUAGE-SPECIFIC (fitted to one typist) | 41, 45, 46, 47, 48, 49 |
| 12 | the gated continuation (chunk_gate) | OFF | BODY-GENERAL | 50, 52 |
| 13 | the tag at entry (utt_entry) | OFF (flat) | MIXED | 7, 50 |
| 14 | the night's device (night_dev) | OFF (the CPU) | BODY-GENERAL plumbing | 44, 52 |
| 15 | the face as the reward | SERVED | the face organ BODY-GENERAL; the felt rule and world_r are not | 41, 42, 45, 46, 50 |

## The tick

Everything below hangs on one loop. `Life.tick()` (body/life.py) runs eight phases in order:

1. `_sense`: the world's next queued symbol or its quiet, and the face felt as reward.
2. `_hear`: the symbol enters the cortex's stream, the store writes what surprised it, and the utterance's end is decided (by the sensed pace, §10).
3. `_learn_values`: every critic learns, and the fast critic's error becomes dopamine.
4. `_own_face`: the face organ.
5. `_choose`: the gate decides whether to speak, then the readout, the actor and the planner decide what.
6. `_act`: the body's own symbol, or its rest, enters the stream; the actor's credit.
7. `_feel_and_learn`: feelings, the gate's lesson every 24 ticks, the cortex's waking lesson every 24 ticks.
8. `_bookkeep`: the page and the sleep switch.

The serve runs `tick()` under one lock at the flags' period. The night runs inside the tick in which sleep pressure crosses its
switch.

---

## 1. The cortex

**What it does.** The cortex is a causal transformer over the last 64 ticks, one position per tick. Each position's input is
the sum of four terms, passed through a LayerNorm (`Organs.inputs`):
- the world's symbol embedding;
- 0.3 times the body's own symbol embedding in the same tick (`own_gain`, the corollary discharge);
- a learned map of the caregiver's face and its change (`face_in`);
- a learned map of the eight band states (`bundle_in`).

Twelve pre-LayerNorm blocks give the stream C. The head `latent_pred(C)` forecasts the next unit embedding. It is trained by
squared error to the unit target, so its minimizer is the conditional mean of the next embedding and its norm is its certainty.
The mouth reads `latent_pred(C) + store_in(recall)` (`Organs.forecast`): the cortex's vote plus the hippocampus's vote.

Recall is never an input to the stream and never a term in the cortex's loss. Item 40 read this in the code: "the recall never
enters the stream". The cortex never learns from reward.

**The waking lesson** (`_wake_lesson`) runs every 24 ticks on the last 32 positions (Adam at 1e-5, gradient clip 1):
- The target at every position is the world's next symbol. The body's own symbols and rests are inputs only.
- At a position after which the utterance was perceived to end, the target is the end symbol, which is the rest (`end_symbol rest`).
- The body's own positions are weighted 0.7 to the power of the number of positions between them and the next world symbol
  (`own_target_decay`), so an own position just before a world symbol keeps weight 1.
- The loss is scaled by (1 + stress/10).

The recall form of the own-speech target, which taught the recall's continuation at the body's own positions, was retired in
item 49: it "ate the answers". Awake, the whole window is recomputed every tick with no cache. At night, the stream runs one
position at a time with a key-value cache (`stream_step`).

**Biology.** Predictive coding (each area learns from its own error); the corollary discharge that attenuates the hearing of
one's own voice; the slow cortical learner of complementary learning systems.

| constant | served | source | note |
|---|---|---|---|
| d, layers, window | 1024, 12, 64 | save (`arch`) | fixed at birth |
| own_gain | 0.3 | flags | the body's own sound in the stream |
| wake_every, wake_window | 24, 32 | default | |
| live_lr | 1e-5 | default | |
| wake_base, wake_dopa, wake_novel | 1.0, 0, 0 | flags, default, default | the gated day lesson failed (item 8) |
| own_target_form | world | flags | recall form retired (item 49) |
| own_target_decay | 0.7 | save | |
| sigreg | 0 | default | SIGReg retired (BODY_SPEC §5b) |

**Status.** SERVED. **Scope.** BODY-GENERAL. The review lists "the cortex" as already general. Its input format, one symbol per
tick on one world channel, is LANGUAGE-SPECIFIC and is the first robot blocker (review §3).
**Evidence.**
- Item 8: the gated waking plasticity, failed, and the held-out's drift by day.
- Items 14 and 15: not run.
- Item 38: the night's divergence.
- Item 40: the recall kept out of the stream.
- Items 44 and 52: the corpus, read and then removed.
- Item 49: the own-speech target.

## 2. The fixed lexicon and the bands

### 2a. The lexicon and the readout

**What it does.** One embedding table E, 107 rows of random normal vectors normalized to unit length, frozen at birth
(`requires_grad` False). It is both the cortex's input and the mouth's readout. No vocabulary softmax is ever trained.

The readout is `logits = sharpness × (pred · Ê)` (`Organs.readout`). With a squared-error forecast, pred · Ê_k behaves as the
probability of symbol k, and the forecast's norm sets how decisive the read is.

Mood sets the sharpness: `max(sharp_min, 25 × (1 + mood/6))`, so it runs from 20 to 50 (§9). At a word's start the mouth samples
from this readout. Inside a word it takes the argmax (§8). The reserved symbols (every `<...>` except the rest, and the newline)
are masked. A fixed random permutation of the d dimensions (`perm`) is the lag code the store's contexts are built with (§3).

Under `sharp_form world`, a calibrated sharpness (`sharp_cal`, temperature scaling against each world symbol) runs as a reading
only. It sets neither the mouth nor REM, because `rem_world_temp` is 0. Decisiveness by certainty (`sharp_conf`) was falsified
on the live answers (item 13).

**Biology.** A fixed random code has no trivial solution to predicting a learned target. The readout's decisiveness follows tonic
dopamine, as the songbird's vocal variability falls when reward comes.

| constant | served | source |
|---|---|---|
| vocab | 107 | the tokenizer |
| sharp_base, sharp_gain | 25, 25 | default |
| sharp_min | 20 | flags (raised from 8 in item 41, so a bad mood no longer turns words into letters) |
| sharp_form | world | flags |
| sharp_conf | 0 | flags (item 13) |
| end_symbol, end_rest | rest, 1 | flags |

**Status.** SERVED. **Scope.** LANGUAGE-SPECIFIC. The review says the symbol table is frozen and random, some organs grow with
the vocabulary, and "the tokenizer stands in for the anatomy" (§3).
**Evidence.**
- Item 13: decisiveness by certainty, falsified.
- Item 36: the live mouth against the greedy readout.
- Item 37: the readout's sharpness bound.
- Item 41: sharp_min 8 to 20.
- Item 42 (2): the temperature is 1/(sharpness × |pred|).

### 2b. The bands (the prefrontal ladder)

**What it does.** Eight leaky integrators of the stream, at clocks of 1, 4, 16, 64, 256, 1024, 4096 and 16384 ticks
(`CLOCKS`). Each tick:

`s ← s + (g/τ)(tanh(W_b C) − s)`

- W_b is fixed at birth. A map trained by the critic's own bootstrapped error is the deadly triad, and it saturated in earlier runs.
- g is a learned Go/NoGo gate on the band's own update, trained toward open when its TD error is positive and toward shut when negative.

The bundle of the eight states enters the cortex through `bundle_in`. Each band carries a value head, the critic at that
timescale (§6). The states are kept across the night (`night_keep_bands 1`) and saved; before item 46, fix 11, they were zeroed
at every reload. The forecast heads `pfc_pred` still exist, but their loss is off under `rem_form imagine` (BODY_SPEC §5c).

**Biology.** Prefrontal integration at many timescales, and basal-ganglia gating of prefrontal updates.

**Status.** SERVED. **Scope.** BODY-GENERAL: the review names "the value ladder" as already general.
**Evidence.**
- Item 38: each band's value scale after the divergence.
- Item 46, fix 11: the bands saved.
- Items 11 and 23 are deferred on the body-general list.

## 3. The hippocampal store

**What it does.** A table of slots (`FastStore`, body/model.py). Each slot holds:
- a unit key and a unit value (the embedding of the symbol that came next);
- a strength, and who said it;
- three marks: an utterance's start, its end, and the seam;
- sixteen tagged links to the slots written next in the same utterance;
- a short-term availability.

**The write law** (`Life._step`). Every symbol the world types (never a rest, never the body's own) writes `key → E[x]`. The
strength is surprise × (1 + |dopamine|):
- surprise is 1 minus the cosine between the last forecast and what arrived;
- dopamine is the previous tick's.

A write needs a key norm above `write_floor` and a strength above 1e-4. If a slot's key and value both match at cosine above
0.97, the write merges into it with repetition suppression: `S += s × m/(m + S)`, where m is the store's mean strength. Otherwise
it makes a new slot. Past the capacity, the single weakest slot is evicted and the last slot moves into its place. Every index
the body holds follows the move through `last_remap`; the thirty-second defect in item 40 was that they did not.

**The key.** The key is the world's own context:
- The fast bag: each world symbol shifts the bag through the permutation and adds its embedding. The bag decays by 0.8 per
  world symbol and by 0.97 per quiet tick. That is the hold of item 37: in the quiet, working memory keeps the question.
- Plus 0.5 times the previous utterance's ordered context (`ctx_form shifted`, decaying 0.95 per symbol), scaled to the bag's
  norm. This context swaps in at the next utterance's first symbol, after that symbol's own write.

The body's own speech never enters a key (the corollary discharge).

**The recall.** The recall runs twice a tick, once in the world's half and once in the body's own half. The query is:
- the world's bag, shifted one lag for each own symbol said since the world's last, and faded by (0.8/0.97) per own symbol
  (`bag_own_fade 2`, in the query only);
- plus the body's own bag, the efference copy;
- plus 0.5 times the current utterance's context.

The read is a softmax over K·q / 0.02. It is a dot product, not a cosine, so the query's norm is the inverse temperature and a
faded context recalls faintly. Two more terms enter the softmax:
- the log of each slot's availability: the winner tires by 0.2, and all slots recover by 0.97 per read;
- ln 20 on the followed episode's next slots (`read_follow`).

The read is by content alone; strength decides durability and replay, not which memory wins. Its output is the attended mean of
the unit values, whose norm is the confidence. It reaches the mouth through `store_in`, a learned d × d map that is the identity
at birth.

**The fade and the floors** (at night, `Store.fade`). All strengths are multiplied by 0.9. Slots under the absolute floor 0.07
are forgotten. The relative floor (0.1 × the mean) applies only when the absolute floor is 0, so with 0.07 set it is inert. A
store rebuilt from the utterance memory (tools/rekey_store.py) skips its first fade (`store_fresh`, item 46, fix 8).

**Biology.**
- Episodic memory, with encoding gated by novelty and dopamine.
- Repetition suppression.
- CA3's recurrent sequence chain.
- The temporal context model (Howard and Kahana).
- Theta-sequence lag coding (holographic reduced representations).
- Synaptic depression of a recalled memory.
- Pattern completion.

| constant | served | source | note |
|---|---|---|---|
| store_cap | 65536 | flags | item 2: 8192, then 32768, then 65536 |
| store_temp | 0.02 | default | |
| store_sat | 1 | flags | repetition suppression |
| store_chain, store_links | 1, 16 | flags | |
| write_floor | 1e-30 | flags | the Landed list: the answer's onset is written after long turns |
| key_ctx, ctx_form, ctx_decay | 0.5, shifted, 0.95 | flags, flags, default | item 40: adopted with the rekey at night 262 |
| bag_decay, bag_rest_decay | 0.8, 0.97 | default, flags | item 37 |
| bag_own_fade, bag_own_weight | 2, 1.0 | flags, default | item 37, form 2 |
| read_follow | 20 | flags | |
| read_tire, read_recover | 0.2, 0.97 | flags | the recovery is lost at every new write (defect 1 below) |
| store_fade | 0.9 | default | |
| store_floor_abs, store_floor_rel | 0.07, 0.1 | flags | the relative floor is inert while the absolute is set |

**Status.** SERVED. These forms are RETIRED:
- the cortex-state key (`key_form cortex`, 0 of 30; the ledger's "Failed" list);
- the per-utterance episode chain (item 1, neutral);
- the relative floor at 0.25 (item 2, falsified);
- dreams by pattern completion from the store (`dream_source store`);
- the rewarded own utterance written as an episode (`own_store`);
- the seam read as the turn's end (`recall_end`).

The absolute floor went on at night 237, was reverted at night 249, and returned at night 294 when the relative floor fed on
itself again (items 2 and 47).

**Scope.** MIXED. The write law, recall by content, the chain and the fade are BODY-GENERAL; the review lists "the store and its
cortex key" as general. The served key, though, is a bag of symbol embeddings with one write per symbol, and the review names
"one store write per symbol" as a robot blocker. The cortex key is the general form, but as the whole key it failed here
(0 of 30).
**Evidence.**
- The Landed list: the write floor.
- Item 1: the episode chain per utterance.
- Item 2: capacity and floors.
- Items 10 and 12: deferred.
- Item 16: noise, not yet run.
- Item 35: a fact told once is kept.
- Item 37: the hold.
- Item 39: the reload's inheritance.
- Item 40: the key's separation, the rekey, the thirty-second defect.
- Item 42 (3): the recall's collisions.
- Item 46: the copy-free store.
- Item 47: the absolute floor returned.

## 4. The utterance memory

**What it does.** The world's utterances are kept whole, as heard. An utterance is the world's symbols between one perceived end
and the next, and it is kept only if it has at least two symbols (`_offset`). Each carries:
- an entry strength: 1.0 under `utt_entry flat`, which is served; §13 describes the other form;
- a serial number.

The memory holds 4096 utterances, and past that the weakest gives way. Every strength is multiplied by `store_fade` (0.9) each
night. Item 7's text says 0.97 a night; the code uses `store_fade`. The night draws its dreams from here (`dream_source
utterances`, §5). The mouth's recall still reads the slot store. The utterance memory is also the material a rekey rebuilds the
store from (item 40).

**Biology.** The replay of an episode as the sequence it was (the twenty-fifth defect in BODY_SPEC: "this memory keeps it
plainly").

| constant | served | source |
|---|---|---|
| utt_cap | 4096 | flags |
| utt_entry, utt_entry_tau | flat, 64 | save/default, default |
| reward_gain | 0 | default (item 7: inconclusive) |
| dream_pair, dream_old_share | 0, 0 | default (the Failed list; item 6: rejected, and a no-op under this source) |

**Status.** SERVED. **Scope.** MIXED. Episodic replay is general. The unit, a line the world typed between two quiets, is not,
and the night "replays only the world's typed lines, never the body's own acts" (review §3).
**Evidence.**
- Items 6 and 7: the draw.
- Item 38: the draw's serials are logged.
- Item 40: the rebuild.
- Item 46: fix 7 (the night ends every utterance) and fix 9 (clipped dreams).
- Item 50: replay by recency alone.
- Item 52: the corpus channel removed.

## 5. The night

**When.** Sleep pressure rises by 1 a tick. At 24000 the typing queue is cleared and `night()` runs inside that tick. An
utterance still open is ended first.

**In order** (`Life.night`):

1. **The dreams drawn.** The count is `min(2048, max(1024, round(1.0 × the day's new memories)))`. The new memories are the
   larger of the store's growth and the day's kept writes. The writes include merges, so every night draws 2048 (defect 14
   below). Dreams are utterances drawn by strength, with replacement when there are fewer than asked. Each gets the end symbol
   appended and is clipped to 63 symbols (the thirty-third defect, recorded under item 40, fixed in `Life.dreams`, test 70).
2. **NREM.** A fresh Adam each night: rate 1e-5, betas 0.9 and 0.99, a linear warm-up over 8 steps, clip 1.
   - Six rounds, the dreams shuffled each round and run in lockstep batches of 16: 768 steps at 2048 dreams.
   - The store is off as input (the reads are zero); the hippocampus supplies the sequence and the cortex must carry it.
   - The bands run from zero along each dream.
3. **REM as imagination** (`_rem_imagine_rounds`). Six rounds of the first 8 dreams.
   - From each dream's first three symbols the cortex runs free for 8 steps, sampled at temperature 1.0.
   - The striatal delay line advances with the imagined events, and the face organ scores each imagined tick.
   - The fast critic's evidence takes each imagined transition exactly as it takes a lived one, weighted by
     `rem_weight × max(0, the face organ's correlation with the felt reward)`. The fast critic is re-solved after each round.
   - No gradient reaches the cortex in REM.
4. **The value ladder's replay** (`_value_replay`). One Adam step over each band's last 32 lived transitions, every band except
   the dopamine band, whose head is solved instead (§6).
5. **The gauge and the plumbing.** The gauge (the cortex alone, teacher-forced on the night's dreams) is read before NREM,
   after NREM and at the end. A non-finite weight reloads the organs from the last save on disk; nothing else is ever rolled
   back. "The night stands" is the user's word of 09-11.
6. **The fade.** The store fades as in §3; the utterance strengths are multiplied by 0.9.
7. **Waking.** The working state wakes fresh: the bags, the window, the gate's buffer, the striatal line, working memory and the
   traces. The bands are kept. Fatigue and sleep pressure return to 0. The day's pace instruments go into the night's report.
8. **The save** (`Life.save`): written to a temporary file, then renamed over the save.

`night_ticks 2200` discounts the critics' bootstrap across the night as 2200 elapsed ticks. It lands on the wrong transition
(defect 9).

**Biology.**
- Sleep need grows with the day's plasticity (Tononi and Cirelli).
- Replay in sharp-wave ripples, interleaved (the complementary learning systems of McClelland, McNaughton and O'Reilly).
- REM as hippocampal-striatal replay for the critics, in the manner of Dyna (Lansink 2009).
- Sleep's plasticity ramps up over the first minutes of NREM (the warm-up).

| constant | served | source |
|---|---|---|
| wake_ticks | 24000 | flags (item 40: the day doubled at night 275) |
| night_starts, night_starts_max, night_load | 1024, 2048, 1.0 | flags |
| night_batch, night_rounds | 16, 6 | flags |
| night_lr, night_warm, night_beta2 | 1e-5, 8, 0.99 | flags (item 38: beta2 0.99 against a night's divergence) |
| rem_form, rem_rounds, rem_dreams, rem_steps, rem_temp | imagine, 6, 8, 8, 1.0 | flags/default |
| rem_weight, rem_world_temp | 1.0, 0 | default |
| night_keep_bands, night_ticks | 1, 2200 | flags |
| dream_source, dream_corpus_n | utterances, 0 | flags (item 52) |
| dream_who, dream_tag | 1, 1 | flags; SERVED, INERT (they act only under `dream_source store`) |
| night_dev | "" (the CPU) | default (§14) |

**Status.** SERVED. RETIRED:
- REM training the forecast heads (`rem_form forecast`, BODY_SPEC §5c);
- the unbatched night;
- dreams from the store;
- the night reading a corpus (item 44 adopted it, item 52 removed it: "a training channel no brain has and no robot will").

**Scope.** BODY-GENERAL: the review names "the night" as already general. Two robot blockers sit in it: sleep comes by tick count
inside a tick, and the night replays only the world's lines.
**Evidence.**
- Items 6 and 7: the draw.
- Items 14 and 15: not run.
- Item 38: robustness, beta2.
- Item 44: reading, and the GPU clock.
- Item 46: the review's night fixes.
- Item 52: the reading off, and the night at about 25 minutes.

## 6. The critics and dopamine

**What it does.** There are five parts.

**The striatal delay line.** The last 8 events of the stream are kept: a heard symbol, an own symbol, a felt face (warm or cold),
or a tick of quiet. Each event is one of 2 × 107 + 3 kinds, and the line is read through a born random expansion into 2048
thresholded ReLU units (`striatum_init`). Working memory sits beside it: a slot of the same width.
- The slot latches the line at the world's utterance end and at a dopamine burst above 0.5.
- It clears at a warm face or after 512 ticks.

The heads read 4096 inputs.

**The fast critic.** A linear value on the striatal input, solved from its evidence by recursive least-squares TD:
- the trace at γ, with λ 1;
- forgetting over 36000 ticks;
- a ridge prior of `fast_rls_prior` × `fast_rls_forget` × each input's running variance (0.3 × 36000), so the prior is
  0.3 in standardized units;
- re-solved every 64 ticks.

Its discount is dopamine's: `dopamine_band 2`, clock 16, γ = 0.9375. **Dopamine** is its error:

`δ = r + 0.9375 V(z_t) − V(z_{t−1})`

**The band ladder's heads.** Semi-gradient TD, one head per band, with Adam at 1e-3. The TD is discounted below clock 1024 and
average-reward at 1024 and above, against `rbar`, the reward rate at a rate of 1/1024. The dopamine band's head is masked out,
because the fast critic replaces it.

**The tonic traces.** The felt reward averaged at each of the eight clocks.

**The ventral critic.** RLS TD(λ = 1) at γ = 1 − 1/1024, forgetting over 36000 ticks. Its ridge prior is 3.0 × 36000 × each
input's running variance, with the statistics kept at a horizon of 36000 ticks; the evidence stays in raw coordinates.
- It reads only the eight tonic traces and the sleep-pressure clock. `vcrit_bands` is "-", so no band state enters it.
- Its weight in the gate's credit is its own reliability: the slope of the realized 1024-tick return on its value, clipped to
  [0, 1] and updated every 256 ticks (`vcrit_ceiling earned`, `vcrit_auto 1`).
- The flag `vcrit_w 0.3` therefore has no effect, as the review found.

**Dopamine reaches:**
- mood and stress (§9);
- the store's write strength (§3);
- the gate's credit (§7);
- the actor's lesson (§8);
- working memory's latch.

It never reaches the cortex's lesson (`wake_dopa` 0).

**Biology.**
- Dopamine as the TD error of reward (Schultz).
- The striatum's sparse born expansion of its afferents.
- Prefrontal gating by dopamine (working memory).
- The ventral striatum's long prospect.
- TD in its least-squares and Kalman form (Xu and colleagues, 2002).
- Tonic dopamine as the reward rate.

| constant | served | source |
|---|---|---|
| dopamine_band | 2 (γ 0.9375) | default |
| fast_input, fast_rls | striatum, 1 | save |
| stri_k, stri_m, stri_quiet | 8, 2048, 1 | default, save, save |
| fast_rls_forget, fast_rls_prior, fast_rls_every | 36000, 0.3, 64 | default, save, default |
| wm, wm_burst, wm_max | 1, 0.5, 512 | save, default, default |
| value_lr, diff_horizon, v_buf | 1e-3, 1024, 32 | default |
| vcrit_rls, vcrit_lambda, vcrit_forget, vcrit_norm_tau, vcrit_rls_prior | 1, 1.0, 36000, 36000, 3.0 | save |
| vcrit_bands, vcrit_traces, vcrit_clock, vcrit_center, vcrit_auto | "-", 1, 1, 0, 1 | save |
| vcrit_gamma, vcrit_ceiling, vcrit_w | 1 − 1/1024, earned, 0.3 | default, flags, flags (no effect) |
| world_r, world_mask | 0.3, 1 | save (§15) |

**Status.** SERVED. RETIRED: the gradient ventral critics, the fast critic on the band state (`fast_input band`), the level input
(`gate_level_w` 0), the slow error in the credit (`gate_slow_w` 0) and the long synaptic tag (`gate_slow_lr` 0).
**Scope.** BODY-GENERAL: the review names the value ladder, "dopamine from the fast critic" and the ventral critic. The striatal
input's width is 2 × vocabulary + 3 per event, one of the "organs that grow with the vocabulary" (review §3).
**Evidence.**
- Items 11, 21 and 22: the body-general list.
- Item 38: the critic after the divergence.
- Item 46: fix 11, the bands saved.
- Item 50: the gate's credit.
- BODY_SPEC §3; §5b, the critic's defects 3 to 11; §5c.

## 7. The mouth gate and its three-factor lesson

**What it does.** The gate (`mouth_gate`, a linear unit 1031 wide) decides each tick whether to act. It reads:
- the stream (C/√d);
- fatigue/10, mood/6 and stress/10;
- the forecast's salience and the ventral level, both held at 0 by their constants;
- these inputs relative to their running mean over 1024 ticks (`gate_center 1`);
- two more inputs, the ear: the world's input (under the sensed pace, §10) and its own act on the last tick (the efference
  copy).

The logit z is divided by (1 + stress/10). Then:

`p(act) = floor + (1 − floor) σ(z)`

The spontaneous floor is 0.05, shaped by the sensed pace (§10). A random draw decides.

**The lesson** (`_gate_lesson`) runs every 24 ticks over the buffered ticks. The credit of tick t is:

`G_t = Σ_{k<12} 0.8^k (δ + w_v δ_long)_{t+k} + [acted] (0.25 − 0.12 (1 + (fatigue/10)²))`

- δ is dopamine;
- δ_long is the ventral critic's error, and w_v its earned reliability;
- 0.25 is the tonic drive, and the last term is the effort cost, convex in fatigue.

The advantage is G minus a running baseline (0.9). The update follows `−mean(A × (acted − p) × z)`: the three-factor rule
(eligibility × dopamine × input), whose expectation is the covariance of credit with acting. `gate_vigor` is 0 in the save, so
no sign-following term is added. The optimizer is Adam at 1e-3 with a clip of 1.

What it has learned (items 41, 47 and 50): a weight near −45 on the ear (−48.7 on 09-20, −44.99 at the save of night 320) and
+13.9 on its own last act. That second weight keeps the gate open once it has begun, which was the babble loop of item 48.

**Biology.** The basal ganglia's Go and NoGo paths learned by a three-factor rule; spontaneous vocal activity as the floor; vocal
suppression while hearing.

| constant | served | source |
|---|---|---|
| gate_every, elig_ticks, elig_decay | 24, 12, 0.8 | default |
| gate_tonic, symbol_cost, gate_fatigue | 0.25, 0.12, 10 | default |
| gate_baseline, gate_floor | 0.9, 0.05 | default |
| gate_vigor, gate_opt, gate_center, gate_ear | 0, adam, 1, 1 | save |
| gate_adam_lr | 1e-3 | default (gate_lr 0.05 unused under Adam) |
| gate_int, gate_slow_lr | 0, 0 | flags |
| gate_salience, gate_level_w, gate_slow_w | 0, 0, 0 | default |

**Status.** SERVED. **Scope.** The rule is BODY-GENERAL: the review names "the gate's three-factor lesson and tag". But "one gate
serves the whole body" (review §3), and a robot needs one per effector.
**Evidence.**
- Items 41, 45 and 46: the ear, and the talk-overs.
- Item 42 (4): the gate's math.
- Item 47: the ear's weight holds.
- Item 48: the babble's door is the gate.
- Item 50: "the lesson moves a weight about a unit a day".
- BODY_SPEC §5b: the gate's rule.
- Defects 4 and 8 below.

## 8. The actor and the chooser

**What it does** (`actor_form chunk`, served). A decision is made at a word's start: the tick after no act, or after its own
space. When the gate says act there:
1. The shortlist is every speakable symbol within 4 logits of the best, at most 4 of them.
2. With more than one, each candidate is imagined on a copy of the window as the first of 2 symbols (itself and one greedy
   continuation, recall off) and valued by the fast critic on the imagined striatal line.
3. The planned logit is the logit plus 4 × that value, divided by (1 + 1.0 × the running surprise at the world's symbols;
   `explore_choice`). The choice is sampled.

Then the word **runs**. Each tick the cortex's most likely continuation is said, with no gate decision (p = 1) and no sampling,
until the space, the rest, or 8 symbols. A running word is cut while the world's line is open (the listening reflex's cut, §11).

The **actor head** is a linear map from the striatal input to a bias over the next symbol, bounded by tanh. It is added to the
logits at its earned voice: the slope of the reward over the next 16 ticks on its vote for the act taken, clipped to [0, 1]. It
is silent until proven. Its lesson is dopamine times an eligibility (one-hot minus the probabilities, outer the striatal input),
at a rate of 0.2 (save), with its weights forgetting over 36000 ticks. The review found its vote under 0.09 logit and its trace
decaying per word rather than per tick (defect 5).

The save's `actor_input "cortex"` is read by no code: the head reads the striatal input.

**The chooser** (`actor_form softmax`) was a striatal head scoring the candidates at a torn moment. It is RETIRED: its first live
day collapsed, the mouth looping on the day's action prior (item 3). The additive and select forms were retired earlier
(BODY_SPEC §6).

**Biology.** The basal ganglia select and credit an action as a chunk; speech runs as motor programs; when the cortex is torn,
the choice is deliberated by hippocampal-prefrontal imagination.

| constant | served | source |
|---|---|---|
| actor, actor_form | 1, chunk | save, flags |
| chunk_max | 8 | flags |
| actor_margin, plan_k, plan_h, plan_beta, plan_boundary | 4, 4, 2, 4, 0 | default/flags |
| explore_choice, explore_tau | 1.0, 64 | flags, default |
| actor_voice, actor_horizon, actor_tau | earned, 16, 36000 | flags, default, default |
| actor_lr, actor_forget, actor_beta | 0.2, 36000, 1.0 | save, default, default |
| actor_input | cortex | save (unread) |
| chunk_gate | 0 | default (§12) |

**Status.** The actor and planner are SERVED; the chooser is RETIRED.
**Scope.** The unit, "a word ended by a space", is LANGUAGE-SPECIFIC (review §3). Deliberation when unsure is BODY-GENERAL (review
§3). Item 52 keeps the space "until a learned stop replaces it".
**Evidence.**
- Item 3: the chooser, failed.
- Item 4: moot.
- Item 13: decisiveness.
- Item 18: moot.
- Item 36: the live mouth.
- Item 50: only a word's first symbol is a gate decision.
- Item 52: the space kept.
- BODY_SPEC §6: the action chunk, the planning actor, the earned voice.

## 9. Feelings: mood, stress, fatigue, sleep pressure

**What they do.** All four run on the body's own clock, per tick, not on the serve's wall clock.

| feeling | how it moves | what it acts on | constants (served, source) |
|---|---|---|---|
| mood | + 0.25 δ each tick; half-life 1200 ticks; clipped to ±6 | the readout's sharpness 25 (1 + mood/6), floored at 20; a gate input | mood_gain 0.25, mood_half_life 1200 (default) |
| stress | + 0.1 × max(0, −δ); half-life 240 ticks; capped at 30 | divides the gate's logit by (1 + stress/10), flattening the choice; multiplies the waking lesson by (1 + stress/10); a gate input | stress_gain 0.1 (flags; 0.5 before 09-12), stress_half_life 240 (default) |
| fatigue | + 0.12 per own symbol; half-life 240 ticks; 0 after the night | a gate input; the effort cost 0.12 (1 + (fatigue/10)²) in the gate's credit | symbol_cost 0.12, fatigue_half_life 240, gate_fatigue 10 (default) |
| sleep pressure | + 1 per tick; 0 after the night | the night at 24000; the ventral critic's clock input | wake_ticks 24000 (flags) |

`burst` (0.5) only counts doses for the instruments.

**Biology.** Mood as the integral of dopamine (tonic dopamine); stress as the integral of its dips; sleep pressure as adenosine.
The mood-set sharpness is the songbird's law: variability falls as reward comes.

**Status.** SERVED. **Scope.** BODY-GENERAL: the review names "feelings".
**Evidence.**
- Item 8: the day's lesson against stress.
- Items 13 and 37: the copies' mood sank to the floor under replayed rewards, a confound in every copy measurement without a
  contingent parent.
- Item 41: the chair's mood falling +1.7 to −6.0, and sharp_min.
- Item 42 (2): "the junk is the product of certainty and mood".
- Item 52: the mood in the chair.
- Items 23 and 31: the body-general list.

## 10. The sensed turn-taking (pace_sense, M1-M5)

**What it does.** Each timing reflex that counted ticks (§11) becomes a quantity the body measures from its partner. The body
tracks three percentiles of the partner's own silences, learned one heard event at a time as running quantiles in log units:

`q ← q + η (p − [ln x ≤ q])`, η = 0.05

A heard event is the gap, in ticks, that a world symbol ends. Gaps under 2 ticks are the line's own rhythm and are ignored.

- **P** is the 0.99 quantile of the partner's pauses inside a line.
- **R_lo** and **R_hi** are the 0.05 and 0.95 quantiles of its returns: gaps in which an end was foreseen and the world stayed
  away at least P.
- A foreseen end followed by the world's return within P is a false end, and its gap is counted as a pause.
- R_lo and R_hi settle first as the sample quantiles of the first 20 returns (1/η); after that they run.
- The trackers are saved with the body. The loader warns if the live mode starts before the returns have settled (the shadow
  day comes first).

The mechanisms (`_pace_hear`, `_pace_ear`, `_pace_floor`):

- **M1, the end foreseen.** On a quiet tick of an open line, if the readout's most likely symbol is the end symbol (the
  reserved masked, the end kept), the line ends as foreseen. It has no constant; sharpness and mood cannot move an argmax.
- **M2, the pause outlasted.** Once the silence will be longer than P, the line ends unforeseen and the ear lets go at once.
- **M3, the ear held.** Any world symbol holds the ear input at 1 until an end lets it go, however slow the hand. The gate's
  learned ear weight (about −45) then keeps the gate shut through the whole line. There is no trace and no gain.
- **M4, the reply ready.** After a foreseen end at tick E, the ear lets go at the first later tick where two things hold: the
  end symbol is no longer the argmax, and the forecast's change, 1 − cos(pred_t, pred_{t−1}), is at most 0.5 of its largest
  since E. This is counted in cortex steps. A reply never ready lapses when the slot closes. `ready_law 0` would release after
  `gate_ear_release` ticks instead.
- **M5, the turn, the wait and alone.** After either kind of end:
  - within the child's slot (since the world's last symbol ≤ R_lo), the whole floor is open;
  - after the slot, the ear input reads 1 until R_hi, then fades to 0 at 2 R_hi;
  - the floor returns as `0.05 × clip(since/R_hi − 1, 0, 1)`.

  While a line is under way the floor follows the same drive: nothing below R_hi, and nothing before the first return is
  known.

**Biology.** A listener projects the turn's end from what has been said (M1). Vocal suppression holds while a partner's call is
under way (M3). A reply is launched when its plan settles (M4). An infant's reply window follows the caregiver's usual latency,
and it vocalizes again when left alone (M5).

| constant | served | source | meaning |
|---|---|---|---|
| pace_sense | 2 (live) | flags | 0 off, 1 shadow, 2 live |
| pace_eta | 0.05 | default | tracker step; follows a new partner within about 20 events |
| pace_pause_p, pace_ret_lo, pace_ret_hi | 0.99, 0.05, 0.95 | default | P, R_lo, R_hi |
| ready_law, ready_ratio | 1, 0.5 | default | M4 by the forecast's settling |
| pace_fore_q | 0 | default | M1 by its own measure: measured twice, not adopted (item 52) |
| offset_ticks | 8 | flags | under pace 2 only a switch: the pace runs only when it is above 0 |

**The first live day** (item 52; the trackers warmed by the shadow day):
- trackers at P 27.5, R_lo 51.6 and R_hi 91.7 ticks;
- 0.09 words said over a one-handed line, 94 percent of lines clean; no words over fast lines;
- answers to 84 percent of the questions, at a median of 2.4 s;
- 0.3 babbled words per silent minute; 4 frowns.

On copies at half the typist's speed it was too patient: the answers fell to 9 of 29, at a median of 4.2 s, in the last third
of the day.

**Status.** SERVED: shadow from the restart after night 322, live from night 323's save. The shadow's determinism digest matched
the switch off (76f1d51c, test 88).
**Scope.** MIXED. The law is body-general: the partner's own silences are measured, typing speed is never read, the body's own
latency is counted in cortex steps, and the code's comment says a robot on 30 Hz frames "met the same way". But it rides on the
world going quiet as both the event clock and the turn signal, which the review names as a robot blocker. M1 also needs the
cortex taught an end symbol at each utterance's end.
**Evidence.**
- Item 50: the user's bar.
- Item 51: the design, the shadow, 90 tests then.
- Item 52: copies at 1.0, 2.0 and 4.0 symbols a second; the live switch; the first live day; pace_fore_q and its warm start.

## 11. The timing reflexes it replaced

**What they were.** From items 41 to 49, each reflex fixed one interruption or one silence, and none of them reads content.
Their values were fitted on copies to this typist's pace, so by the user's bar of 09-22 they were tuning (item 50). Under
`pace_sense 2`, every path below is gated on the pace mode being under 2 and does not run. The flags still carry their values.

| reflex | flag (served value) | what it did | ledger | under pace_sense 2 |
|---|---|---|---|---|
| the listening reflex | gate_listen 1.0 | while the world's line is open, the floor scaled to 0 and a running word cut | 41 | the floor part is superseded by M5; **the word cut still acts** (the chunk runs only when not listening) |
| the babble drive | gate_quiet_tau 1200 | the floor rebuilt over this many quiet ticks | 41, 48 | inert (M5's drive) |
| the sure proposal | gate_quiet_sure 0 | the whole floor when the forecast's norm passed it | 41, 48 | off since item 48 |
| the count | offset_ticks 8 | a line ended after this many quiet ticks (30 for a day, item 45; back to 8, item 46) | 45, 46 | a switch only; M2 replaces the count |
| the settle law | offset_form settle (offset_settle 0.5, offset_fast 4, offset_slow 64) | a line ended when the surprise's fast average fell under half its slow one | 45, 48 | inert (M1); item 48 found it never fired on one-handed lines |
| the quiet foreseen | offset_foresee 0.2 | a line ended when the readout's probability of the rest reached this | 48, 49 | inert (M1's argmax) |
| the ear's trace | gate_ear_decay 0.9 | the ear input decayed by this per tick between symbols | 46 | inert (M3) |
| the ear's gain | gate_ear_gain 1.5 | the ear input's scale | 47 | inert (M3 uses gain 1) |
| the ear's release | gate_ear_release 9 | the ear released this many ticks after a settled end (1, 0, 2, 6, then 9) | 46, 48 | inert under ready_law 1 (M4) |
| the yield | gate_yield 20, gate_yield_after 40 | past the slot, the gate held, the hold fading over gate_quiet_tau | 48 | inert (M5's wait) |
| the turn's readiness | gate_turn 1 | the whole floor for the slot after a settled end | 48 | inert (M5's slot) |
| the turn's floor | gate_turn_floor 0 | a readiness above the resting floor | 48 | measured, no gain; 0 |

**Status.** RETIRED by pace_sense at night 323's save. The flags remain in ops/BASE_FLAGS.txt and are inert, except gate_listen's
word cut. The review's refactor plan gathers them into a Reflexes object (§4).
**Scope.** LANGUAGE-SPECIFIC: fitted to one typist's pace.
**Evidence.**
- Item 41: the reflex, the drive, the sure proposal.
- Item 45: a person's hand.
- Item 46: the ear's trace and its release.
- Item 47: the gain.
- Item 48: the yield, the quiet foreseen, the release sweeps.
- Item 49: the foreseen threshold at 0.2.
- Item 50: why they had to go.

## 12. The gated continuation (chunk_gate)

**What it does.** Inside a word, the learned gate's own draw decides at every symbol whether the program goes on. The letter is
still the cortex's continuation; a "no" stops the word. With it, the face's credit reaches the choices a word is made of. Without
it, the letters run at p = 1 and carry no eligibility, and a frown felt at the word's end reaches only its first symbol, 5 to 10
ticks back (item 50).

**Biology.** The stop pathway: cortex to subthalamic nucleus, which can halt an action already under way.

| constant | served | source |
|---|---|---|
| chunk_gate | 0 | default (the save holds 0) |

**Status.** OFF. It was measured on reflex-free copies in item 50. With it and the tag at entry (B1) against without both (A1):
- words said over fast lines: 7 over 71 lines, against 70 over 66;
- frowns: 146 against 281;
- words over one-handed lines: 7.9 each, against 9.1.

B1 carried both fixes, so the two are not separated. Before adoption, defect 4 must be fixed: a rest inside a word is logged as
the gate saying no, and under chunk_gate that would happen at every rest.
**Scope.** BODY-GENERAL: the review names chunk_gate, and item 52 calls it "the first half" of a learned stop.
**Evidence.** Items 50 and 52; defect 4.

## 13. The tag at entry (utt_entry)

**What it does.** It sets the strength with which an utterance enters the night's draw.
- Under `flat` (served), every utterance enters at 1.0, so the night replays by recency alone.
- Under `felt`, an utterance enters at the mean over its symbols of the store's own write strength, surprise × (1 + |dopamine|),
  divided by a running mean over utterances (τ 64 utterances). The average entry stays near 1.

**Biology.** A hippocampus tags an experience at encoding by its novelty and by the reward around it, and replays the tagged
more.

| constant | served | source |
|---|---|---|
| utt_entry, utt_entry_tau | flat, 64 | default (the save holds flat) |

**Status.** OFF. It was measured only inside B1 (item 50), where the next morning the ten facts told once the day before were
recalled 4 of 10 whole and 3 of 10 near. The ledger says "no matched flat-entry night was run, so the felt entry's share is not
yet separated". The review adds two defects:
- Smiles never reach the felt entry, since it sees only the dopamine of the tick before each heard symbol, so B1 tested tagging
  by surprise, not by reward (defect 6).
- The running mean starts cold and is not saved (defect 7).

Its predecessor, `reward_gain` (item 7: the tag on the line before a smile), was inconclusive and is off.
**Scope.** MIXED. Tagging at encoding is general; the unit tagged is the typed utterance.
**Evidence.** Items 7 and 50; defects 6 and 7.

## 14. The night's device (night_dev)

**What it does.** A device name. After the dreams are drawn, the cortex and the day's small state move to that device for the
night's lessons (NREM and REM), and they come home before the value replay, the fade and the save. The critics' float64 evidence
stays on the host (`Organs.to`), and so does every random draw. A failed night comes home in `finally`.

| constant | served | source |
|---|---|---|
| night_dev | "" (the body's own device, the CPU) | default |

**Status.** OFF. Measured in item 52 on the save of night 322, with the same 2048 dreams and 768 steps on both:
- the NREM curve agreed within 0.001, and the gauge was the same;
- the GPU night took 1250 s against 1502 s on the CPU, 0.83 of the time where the pass line asked for 0.65.

Not adopted; the nights stay on the CPU. Item 44 had found the waking tick no faster on the GPU.
**Scope.** BODY-GENERAL plumbing. The robot need behind it is a loop with a deadline and safe sleep (review §3).
**Evidence.** Items 44 and 52.

## 15. The face as the reward

**What it does.** The caregiver's face is a number from −6 to 6 (`POST /face`). The body feels a **change** of level, not a held
face (`_sense`):
- the level is the integer part;
- a rise in magnitude or a change of sign is felt as the new level, clipped to ±2;
- easing back toward 0 is not an event.

That felt value is the reward r that every critic learns from (§6).

Beside the face there is one more felt term, grounded and served: every symbol the world types is felt as +0.3, except on a tick
right after the mouth acted (`world_r 0.3`, `world_mask 1`, both in the save). BODY_SPEC §6 names it next to the face and
measured it at about two fifths of all felt reward on the reference days. So in this body the face is the only reward the
caregiver chooses, but it is not the only felt reward.

The face is also a sense, twice over:
- it enters the cortex's input through `face_in`;
- a felt face is an event in the striatal line.

**The face organ** (`_own_face`, `face_form foresee`, `face_input striatum`) is the body's own face. It is a least-squares readout
of the striatal input that foresees the next tick's felt reward (ridge 0.1, forgetting over 36000 ticks, solved every 64 ticks).
Its correlation with the felt reward is its reliability, and that correlation is what REM weights imagination by (§5). The page
shows it as "its face". It is a readout, not a lever.

**Who makes the face.** The served parent is `body.teacher`, whose lines come from a queue file (data/teach_queue_w2.jsonl). The
queue is fed by ops/night_cycle.sh (the taught pairs, story rows, revisits); the brief for anyone writing to it is
ops/parent_brief_human.txt. The parent decides its face from the page alone. With the served environment
(ops/typist_chain_command.txt):
- a word it knows, said in the child's turn: +2 held 5 ticks;
- a word of the coming answer: +2, then +4, then 0 (ANSWER_SMILE);
- a word said over the parent's line: −2 held 2.5 ticks, at most every 20 ticks (TALKOVER_FROWN, FROWN_GAP 20);
- no smile for a word said past the child's turn (TURN_ONLY_SMILE);
- no smile for the same word again within 120 ticks (HABIT_TICKS);
- when its attention runs low, the parent turns away for 200 ticks: the still face.

A word counts as known once the teacher has typed it three times. A third of the parent voice's lines (never the `b:` voice's)
are typed one-handed, at 2 symbols a second with thinking pauses (SLOW_SHARE 0.33). `/insides` is the supervisor's instrument
and never reaches the one who decides the face.

**Biology.** Social reward from a caregiver's face, felt as prediction error. The face organ is the body's model of whatever
grounded reward it has; BODY_SPEC §5c says "a humanoid keeps it unchanged".

| constant | served | source |
|---|---|---|
| face_form, face_input | foresee, striatum | flags |
| face_tau, face_ridge, face_every | 36000, 0.1, 64 | default |
| world_r, world_mask, cost_in_reward | 0.3, 1, 0 | save, save, default |

**Status.** SERVED.
**Scope.** The face organ is BODY-GENERAL (the review names it). Two parts are not:
- The felt rule, which feels only changes of face level, is a robot blocker: "a jittery face reader produced 77 false smiles in
  300 ticks" (review §3).
- `world_r`, words as reward, belongs to a language world.

**Evidence.**
- Item 41: the frowns for talk-overs.
- Item 42 (2): the mood as the integral of dopamine.
- Item 45: the one-handed parent.
- Item 46: the answer's smile window fixed; the frown weighted −2 from night 290.
- Item 50: the frown lands 5 to 10 ticks after the decision.
- Defects 2, 18 and 19.

---

## What a robot needs next

The review (§3) names what stands between this body and a humanoid. None of it blocks the diary's demo. It groups the work in five
areas:
1. an Anatomy object in place of the tokenizer;
2. a frame of several channels per tick, with a gate per effector;
3. event boundaries from prediction error, and a separate channel for whose turn it is;
4. motor programs that learn when to stop;
5. a loop with a deadline, safe sleep and a steadier reward signal.

The ledger's "Body only" list (items 19-34) names the organs a simulated body would test first. Read against the mechanisms
above:

| need | where the body stands today | what points the way |
|---|---|---|
| **Several senses per tick** | One symbol per tick on one world channel (§1). The tokenizer stands in for the anatomy. The store writes one memory per symbol under a bag of symbol embeddings (§3). The striatal line knows four kinds of event: heard, own, face, quiet (§6). | A frame of channels per tick, with the cortex's input, the store's key and the striatal events built from the frame (review areas 1-2). A learned sensory front end (19), sensory gating and attention (29), real sensor noise (33). The cortex-state key failed as the whole key here (0 of 30) and would need retesting on a frame. |
| **A motor alphabet** | The mouth says one of 107 characters, read from a frozen random table (§2a). The actor's output and the striatal input's width grow with the vocabulary (§6, §8). | Effector primitives declared by the anatomy, not a tokenizer. Continuous action with a forward model (the cerebellum), with reflexes and pattern generators beneath it (20). |
| **A gate per effector** | One learned gate serves the whole body, with one ear and one own-act input (§7). | The same three-factor lesson per effector, each gate with its own efference input (review area 2). The rule is already general. |
| **Learned stops replacing the space** | A word runs to the space, the rest, or 8 symbols (§8). The quiet is both the event clock and the turn signal (§10). chunk_gate is built and off (§12). | Motor programs that learn when to stop (area 4); chunk_gate is "the first half" (item 52). Event boundaries from prediction error (item 9, not run) and a turn channel apart from the quiet (area 3). |
| **Pain and needs as internal reward** | r is a change of face level clipped to ±2, plus 0.3 per heard symbol (§15). No internal need rewards anything. | Drives from the hypothalamus: energy and temperature as the ground of reward (27). Threat and pain as a fast value tag with one-trial fear (28). Interoception (30), a circadian clock beside fatigue (31), joint attention and social reward (32). Each would enter r beside the face and reach every critic through the same fast critic, as grounded signals rather than rules. A steadier face reader (area 5). |

Three more constraints from the review's list:
- The loop runs at 0.15 s with no deadline. Awake, it recomputes the whole window every tick with no cache, and inputs wait in
  a queue.
- Sleep comes by tick count, inside a tick, under the serve's lock.
- The night replays only the world's typed lines, never the body's own acts (§4, §5). The own-episode write, `own_store`, is
  retired here.

The review lists as already general:
- the tick's phases, the value ladder, dopamine from the fast critic and the ventral critic;
- the gate's three-factor lesson and tag, and the body's copy of its own acts;
- the cortex, the store and its cortex key, the night and feelings;
- the face organ, the settle law, deliberation when unsure, and chunk_gate;
- save and load with organ migration.

pace_sense came after the review. Its law is general and its input is not (§10).

## Known defects

**From the review of 2026-09-22.** The review ranked these by harm to the body's learning, and each was checked by a skeptic. All
are open unless noted. Locations are in the current code (body/, after the split of life.py into body/core/); the review's line numbers predate the pace_sense merge and the split. "Constant"
means a switch that is off by default, measured on a copy before use.

| # | defect | where | harm, as the review measured | proposed fix |
|---|---|---|---|---|
| 1 | Tired memories never recover. `_tire` rebinds `store.A` to a new tensor; each new slot's write points it back at FastStore's buffer, which holds only the tiring from write ticks. | body/core/memory.py `_tire` (lines 23-27) | medium, served: the most-used memories are pushed down all day; the best match's availability averaged 0.62 against 0.71 fixed | `store.A.copy_(1 − rc (1 − store.A))` behind a constant, plus a test that writes a new slot after tiring |
| 2 | Some smiles are never felt. Two posts land milliseconds apart with no tick between. | body/caregiver.py (review: line 331) | medium, served: about 12% of known-word smiles and the first step of about a quarter of answer smiles; the "late" misses are refractory drops | relabel the log rows; env constant FACE_REST_SEEN (one tick at face 0 before the next smile) |
| 3 | Long lines are silently dropped. `clean()` drops lines over 40 symbols or with an apostrophe. | body/teacher.py (review: line 279) | medium, served: 553 of 10000 lines in the last 2500 queue rows | **partly fixed**: the four long POOL questions shortened in ops/night_cycle.sh (commit b10dd84). Still open: ops/read_stories.py allows the apostrophe, and ops/queue_depth.py does not count through `clean()` |
| 4 | A rest is logged as the gate saying no. Inside a running word a sampled rest is stored as (acted 0, p 1); with gate_vigor 0 that is an eligibility of −1 on a tick the gate never decided. | body/core/mouth.py `_choose`, chunk branch (lines 507-512) | low now (2 words in 15 days); at every rest inside a word under chunk_gate 1 | record the gate's own draw and use it for eligibility, before chunk_gate |
| 5 | The actor's trace decays per word, not per tick, while dopamine multiplies it every tick. | core/mouth.py `_act` (lines 543-546) and core/critics.py `_learn_values` (lines 160-162) | low: credit smears over about 16 words; nothing measured about the actor can be trusted until fixed | constant `actor_trace_tick` |
| 6 | Smiles never reach the felt entry: it sees only the dopamine of the tick before each heard symbol. | core/cortex.py `_step` (line 39) | low: B1 tested tagging by surprise, not reward | constant `utt_entry_trace` (a tag lasting 32-64 ticks) |
| 7 | The felt entry's running mean starts cold and is not saved. | core/senses.py `_offset` (lines 72-76); absent from `save()` | low: the first facts of B1 entered 1.1-1.5 times too strong | save and restore the mean; warm-up behind `utt_entry_warm` |
| 8 | The gate's credit counts the dopamine from before the act. | core/mouth.py `_gate_lesson` | low: unbiased, about 21% more noise | constant `elig_from` |
| 9 | `night_ticks` discounts the wrong step, the first to second morning tick. No transition across the night has been learned. | core/critics.py `_learn_values` (lines 53-55) | low: one bad sample a day | drop `--night-ticks` at a boundary, or constant `night_bridge` |
| 10 | A working-memory latch breaks the fast critic's evidence chain. | core/critics.py `_learn_values` (line 169) | low: about 1% of transitions | constant `wm_chain` |
| 11 | The quiet foreseen trains on its own guess: the foreseen end becomes the end target the cortex learns. | review: offset_foresee (inert now) | low | constant `foresee_retract`. My reading of the code, not measured: M1 ends a line through the same `_offset`, which marks the window's last world position as ended, so the same loop applies to M1 |
| 12 | The end depends on mood: p(rest) is read at the mood-set sharpness. | offset_foresee | low | constant `foresee_sharp`. Does not apply to M1, whose argmax sharpness cannot move |
| 13 | The store reads and tires twice a tick, in the world's half and the own half. | `_step` via `_hear` and `_act` | documentation | correct the docs; `read_tire_once` only if the documented rate is wanted |
| 14 | Every night draws the maximum dreams, because the write counter includes merges. | core/cortex.py `_step` (lines 52-53); `FastStore.write` returns True on a merge | low: 2048 dreams every night | constant `night_count='slots'` |
| 15 | Faded keys are stored malformed: keys fainter than 1e-12 normalize to short vectors that can never be recalled. | body/model.py `FastStore.write` (line 884) | low: rare since 09-19 | constant `store_key_rescale` |
| 16 | A failed save causes an extra night 12000 ticks later, and both memories fade twice that day. | core/night.py `night` (lines 398-403) | never fired; the disk is 94% full | give `save()` its own try block |
| 17 | The line popped at nightfall is lost. | body/teacher.py (review: line 215) | 5 of the last 10 nights | env constant UNSAID_BACK |
| 18 | "hot." gets no smile, and it drains attention. | body/caregiver.py (review: line 321) | about 2% of smiles; a bias against the end mark | KNOWN_MARK (strip only a trailing mark) |
| 19 | The answer smile pays for echoing the question. | body/caregiver.py (review: line 316) | about a third of answer smiles | ANSWER_ASKED (the answer's words minus the question's, both "or" choices kept, fused words accepted) |
| 20 | Negligible. | core/night.py `night` (line 380); core/persistence.py `load` (line 101); body/life.py `__init__` (line 228) | none | `night_ends_word`, `load_ends_line`; the optimizers are not saved, and need not be |

**Not reached by the served body** (review):
- A NaN in the bands freezes the tick loop before any night.
- A skipped write keeps `last_remap`. In `FastStore.write` the early return comes before the clear.
- A loaded store ignores `--store-temp`: `Store.load_state_dict` takes the saved temperature.
- A NaN face becomes a full smile: `set_face` clips without a finiteness check.
- Two visitor problems exist on the unpaused page at port 8020 (/talk).
- A failed `/type` logs a night that never happened.

**The critic's extra findings** (not verified by the review):
- `sharp_cal` never settles under `sharp_form world`. It is an instrument only (§2a).
- REM samples at the evening's mood sharpness. This overlaps defect 12.
- The first poll of a new typist day numbers a trimmed page wrongly.
- The yes/no check matches "not", "nothing" and "now".

**The process defect** (review §4). tools/determinism_check.py runs on flags alone. The 26 served constants that live only in the
save are never exercised, so the check never runs:
- the served dopamine, the ventral critic or the actor;
- the word-level mouth or the ear;
- save and load, or the sleep switch.

Its hash leaves out the bands, the face organ and `rbar`, and the tests share the gap. Until a served profile, a hash of the full
state and a save/load round trip are added, no refactor can be shown to change nothing on the served body.
The check now has them (2026-09-23): `--profile served` and `--profile switches`, `--full` and `--roundtrip`
(tools/determinism_check.py's first lines). The round trip found one field a reload does not give back: a store born under
store_sat 1 is never marked as converted, so its first reload compresses strengths that were already saturating.

**Constants with no effect on the served body.** The review counts 67 of 204. The ones checked in the code for this document:
- `vcrit_w` 0.3 is overruled by `vcrit_ceiling earned` with `vcrit_auto 1`.
- The reflex flags of §11 are inert under `pace_sense 2`: offset_form, offset_foresee, gate_ear_decay, gate_ear_gain,
  gate_ear_release, gate_quiet_tau, gate_quiet_sure, gate_yield and gate_turn. offset_ticks still acts as a switch, and
  gate_listen still cuts a running word.
- `dream_who` and `dream_tag` act only under `dream_source store`.
- `store_floor_rel` does nothing while `store_floor_abs` is set.
- `dream_corpus_file` does nothing with `dream_corpus_n` 0.
- `gate_lr` is unused under Adam.
- `actor_input` and `band_lr` are read by no code.
- `key_scale` acts only under the cortex key and in the retired chooser (`actor_form softmax`).
- `heard_decay` feeds a tally that is saved but never read.

## Where the code lives

The mechanisms map onto these functions:

| mechanism | body/model.py | `Life` (body/life.py and its mixins in body/core/) |
|---|---|---|
| cortex | `Organs.inputs`, `stream`, `stream_step`, `forecast`, `latent_loss` | `_wake_lesson`, `_stream_now` |
| lexicon and readout | `Organs.E`, `readout`, `perm` | `_choose` (sharpness), `_sharp_calibrate` |
| bands | `band_update`, `band_update_b`, `value_of` | `_learn_values` |
| store | `Store`, `FastStore` | `_step`, `_recall`, `_tire`, `take_world`, `take_own`, `bag`, `key` |
| utterance memory | | `_offset`, `dreams` |
| night | `stream_step` | `night`, `_dream_batch`, `_rem_imagine_rounds`, `_value_replay`, `gauge`, `_night_away`, `_night_home` |
| critics and dopamine | `striatum_*`, `wm_*`, `fast_rls_solve`, `vcrit_rls_solve` | `_learn_values`, `_vrel_update` |
| gate | `mouth_gate`, `widen_gate` | `_choose`, `_gate_lesson` |
| actor | `actor`, `chooser` | `_choose`, `_act`, `_imagine_value`, `_arel_update` |
| feelings | | `_decay_feelings`, `_feel_and_learn` |
| pace_sense | | `_pace_*`, the ear and floor in `_choose` |
| face | `face_in`, `face_head` | `_sense`, `_own_face`, `_face_solve`, `set_face` |
| save and load | | `save`, `load`, `birth` |

The review's refactor plan (§4) splits life.py into mixins in this same order: physiology, senses, memory, cortex, mouth (with a
Reflexes object for §11), critics, actor, night, persistence and instruments. It advises no full refactor before the demo.
Step 2, the mechanical split, is done (2026-09-23, branch refactor-life). body/life.py keeps `Life.__init__` and `tick()`. Every
other method moved verbatim into a mixin in body/core/, which `Life` inherits in that order. The map is in body/core/__init__.py:
- senses.py: `_decay_feelings`, `_offset`, `_sense`, `_hear`, `type_text`, `set_face`
- memory.py: `query_from`, `_tire`, `_recall`, `bag`, `key`, `_ctx_scaled`, `take_world`, `take_own`, `note_offset`, `rest_tick`, `_consolidate_own`
- cortex.py: `_step`, `_window_tensors`, `_stream_now`, `_wake_lesson`, `_sharp_calibrate`
- mouth.py: `_imagine_value`, the `_pace_*` methods, `_choose`, `_act`, `_feel_and_learn`, `_gate_lesson`
- critics.py: `fast_value`, `_learn_values`, `_own_face`, `_vrel_update`, the face organ
- actor.py: `_chooser_credit`, `_chooser_learn`, `_arel_update`
- night.py: dreams, `night`, REM, `_value_replay`, `_sleep_now`
- persistence.py: `save`, `load`, `birth`
- instruments.py: `_bookkeep`, `gauge`, `state`, `anticipation`, `insides`

`PHYSIOLOGY` is in physiology.py, and `from body.life import Life, PHYSIOLOGY` still works. Every digest of the determinism check
reproduced unchanged. The Reflexes object was not made, because it would change method bodies.
