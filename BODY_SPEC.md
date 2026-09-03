# BODY_SPEC — the second body, from the first body's lessons

Written 2026-09-03 after the first lineage (twenty-three days of THE DIARY on a
0.5B body from nothing) was stopped on the user's word: "stop what we are doing
now and start from scratch." Everything below is either PHYSIOLOGY (a constant
of a bodily process, fixed for a life, with a biological reading), an ORGAN
(learned from its own signal), or PLUMBING (numerical safety). Nothing decides
behaviour or content for the body by hand. The environment is raw. One reward
system, created inside, received at every timescale.

The aim is an architecture that never has to change during a life. The only
thing that changes while it lives is what the world says to it.

## 1. The world

- THE DIARY, unchanged: two writers, one page, one symbol per tick. The
  caregiver's letters enter one per tick; the body's symbol (or its rest)
  enters on the same tick. Symbols are characters (`data/tok_char.json`);
  ids below 11 are bookkeeping marks the mouth can never choose; the newline
  is a page mark and never enters.
- The caregiver's face is a continuous sense (−6..6); a CHANGE of face is
  felt (a held face is silence). The speaker of every symbol is a sense
  (0 the world, 1 the body).
- Protocol and endpoints as before (`/type`, `/face`, `/state`, `/save`), so
  the curriculum and the caregiver tooling carry over.

## 2. Organs

**Lexicon.** One embedding table E (symbols × d). It is the input and the
readout: the mouth reads it by cosine. No vocabulary softmax is ever trained.

**Hippocampus (episodic store).** Slots of (key, value, strength, who). The
key is the decaying bag of the last symbols' embeddings plus the speaker
sense; the value is the embedding of the symbol that came next. Write: a new
slot or a merge into the nearest slot; strength = surprise × (1 + dopamine),
own symbols carry zero surprise (corollary discharge: what the mouth wrote
was foretold). Read: attention over keys at the organ's own temperature,
returning a predicted next embedding. It reaches the cortex through ONE
learned path, the slot `store_in`, initialized to identity (the pathway
exists at birth; the cortex learns to modulate it). No direct vote on the
mouth. Fade: strengths × 0.9 each night; slots below the store's own mean
strength × 0.1 are forgotten (relative, not a constant).

**PFC (the band ladder).** Bands with clocks 1, 4, 16, 64, 256, 1024, 4096,
16384 ticks: leaky integrators of the stream, each with a learned input map,
updated at its clock. Each band has a value head (the critic at that
timescale) and a Go/NoGo gate learned from the value's error. The bands'
states form the bundle the cortex reads and must foresee.

**Cortex.** A small transformer over the last W steps of [embedding, speaker,
face, bundle, hippocampal read] producing the stream C; from C, `latent_pred`
forecasts the next embedding it will receive and `pfc_pred` forecasts the
next bundle. Its lessons are prediction: one minus the cosine to the
embedding received (stop-grad), one minus the cosine to the bundle received
(stop-grad), SIGReg on the forecasts as the collapse guard. It learns awake
(every K ticks on the last window, the world's symbols as targets, its own
symbols inputs only) and asleep (below). It never learns from reward.

**Mouth.** Whether to act is the basal ganglia's: a gate on [C, fatigue,
mood, stress] giving p(act); zero weights and a birth bias at birth. What to
say is the lexicon read by cosine from the forecast, logits = s × cosine,
sampled. s is a physiology constant until it becomes an organ (a learned
decisiveness driven by tonic dopamine). The gate learns by the opponent rule
(a dopamine burst strengthens Go for the context whatever it did, a dip
strengthens NoGo), plus its own reward at a symbol (the belief it had in its
choice, habituating per symbol) minus a cost that grows with fatigue.

**Face organ.** Its face is a forecast of the caregiver's, learned every tick
(head only). A readout, not a lever.

## 3. One reward system

- The face becomes dopamine as the fast band's signed prediction error of
  reward. Dopamine spreads over the last twelve ticks (0.8 per tick).
- Dopamine acts on: the store's write strength (encoding), the gate (acting),
  the value ladder at every band (expectation at every timescale, learned by
  TD as the bands tick and replayed each night), the Go/NoGo band gates, and
  the waking lesson's weight (plasticity). Never on content.
- Reward is created inside: the ladder's own error fires before a face once
  it has learned (secondary reinforcers); the body's own reward at a symbol
  drives the gate.
- Feelings: fatigue (+cost per symbol, half-life 120 s), stress (a leaky
  integral of dopamine dips, half-life 120 s), mood (a leaky integral of
  dopamine, half-life 600 s), sleep pressure (+1 per tick). Stress raises the
  waking lesson's weight and flattens the gate (exploration).

## 4. Sleep

- Sleep pressure crosses the switch (wake ticks) and the body sleeps by
  itself; the typing queue empties; it wakes when the night is done.
- NREM: dreams start where the store is strongest (slots sampled by strength);
  each dream is pattern completion (value → next key → read → ...) until the
  read's confidence falls below the store's own mean; the cortex learns the
  latent lesson on every dream, all dreams summed into one step per round,
  R rounds, sleep's own optimizer and rate.
- REM: from each dream's first symbols the cortex runs free on its own
  readout, hippocampus decoupled, and learns to forecast the bundle it
  receives at its own next step (stop-grad, SIGReg).
- The value ladder replays its lived pairs once. The store fades. Working
  state wakes fresh. Save and back up.
- PLUMBING: a non-finite lesson is skipped; gradients clipped; a night that
  leaves a weight non-finite is discarded and the body reloaded; a failed
  night is retried after half a day.

## 5. Physiology (the disclosed constants)

symbol cost 0.12 · fatigue and stress half-life 120 s · mood half-life 600 s
· wake switch 12,000 ticks · eligibility 12 × 0.8 · store fade 0.9/night ·
store forget floor 0.1 × mean strength · night rate 1e-4, rounds 24, REM
steps 8, SIGReg 0.1 · waking lesson every 24 ticks on 32 symbols at 1e-5 ·
gate rate 0.05, birth p(act) 0.25, habituation 0.9/act, fatigue scaling /10,
own-reward weight 0.5, credit baseline 0.98 (the striatum learns from the
error against what it expected: a constant cost teaches nothing) · dream
recall adaptation 0.5 per recall, recovery 0.7 per step (a recalled memory
tires, so a dream moves on) · readout sharpness 10 (to become an organ) ·
band clocks 1..16384 · dose burst 0.5 (PLUMBING, a compute budget).

## 6. The environment (raw)

The caregiver decides from the page and its face row only: pace by its quiet
(six seconds) or ninety seconds, smile 0→2 within three seconds at a known
word of two or more letters bounded by space, edge or its rest, at a correct
completion after a cue, and at the first two letters of a right answer after a
cue; frown only at a run of a repeated non-letter mark that is not the space;
never at quiet, never at babble; a run of one letter expanded once at its
third return; no newline; it sleeps on its own. Proposed, awaiting the word:
answer a smiled word by repeating it and expanding it into a frame.

## 7. Instruments (the supervisor's, never the caregiver's)

- The gauge: the cortex alone (store off), teacher-forced on the night's
  dreams, before and after the lesson.
- The free-running probe: a cue typed, then the mouth alone, memory off and
  on, greedy.
- The sampled mouth: as it lives, N samples per cue; started / completed
  answers; quiet fraction.
- Per night: dreams (examples, lengths), NREM loss, REM cosine, gauge,
  discarded flag. Per day: known words, smiles felt, cue completions, gate
  rate, fatigue, stress, mood.

## 8. Tests (each fails when its organ stops doing its job)

1. Corollary discharge: the body's own symbols are stored with near-zero
   strength; the world's with strength proportional to surprise.
2. The store recalls: after one line, the cue's next embedding is nearest
   the right symbol.
3. Dreams come from the store: a night's dreams are its lines, not noise.
4. The night moves the cortex: the gauge rises on the dreams; the free
   probe changes.
5. REM learns: the forecast cosine rises across nights; no NaN.
6. The gate holds its birth rate with no reward, opens on a burst, closes as
   fatigue grows, and habituates on a run.
7. Feelings follow dopamine, not effort: a smile lifts mood; babble alone
   does not sink it.
8. Sleep by fatigue: it sleeps at the switch and wakes rested.
9. Guards: an injected NaN lesson is skipped; an injected NaN weight
   discards the night.

## 9. Milestones

1. The night moves the cortex without collapse (gauge, probe).
2. The free-running probe gives a frame, not one letter.
3. A cue completed awake.
4. Known words in its own babble, several an hour, in the contexts that
   earned them.

## 10. Scale

Small first: d 256, six cortex blocks, window 64, about five to ten million
parameters; a tick in tens of milliseconds, a night in a minute, a day in
ten. Scale only after milestone 3.
