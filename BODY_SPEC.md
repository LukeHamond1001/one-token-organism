# BODY_SPEC — the second body, from the first body's lessons

> The user's law (2026-09-03): an architecture with grounded reward, received
> at short and long timescales, no cheats inside the architecture or the
> environment; the environment raw; and perfect mathematics and biology to get
> there. Every change below is checked against this sentence.

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
context is a lag code: two decaying contexts of embeddings, content alone,
each fading 0.8 per tick (a pause ends a context, as working memory does), and
each symbol shifts the whole context through a fixed permutation of the
dimensions before entering at lag 0, so a symbol at lag 0 and the same symbol
at lag 1 are different directions (theta sequence coding; the mathematics of
holographic reduced representations). The world's symbols enter the world's
context, its own enter its own, and the read query shifts the world's context
by as many lags as it has said since (the query after its own "b" is the key
the world's "b" would have made); what it said leaves the query when the world
speaks again. The store reads by dot product: keys are unit directions, the
query is the context as it is, so its norm is the inverse temperature of
recall and a faded context recalls faintly. A memory is written under the world's bag alone (corollary discharge:
the hearing of self-produced sound is suppressed, so the world's sequence is
stored under the world's context, never under its own babble). A memory is
read with the world's bag plus its own bag in full (the efference copy: the
sequencing system knows what it just said). Measured on the collision test:
own symbols in the key at 0.6 broke recall by content; at 0.5 in the query,
the stale key (the cue alone) outmatched the continuation key after its own
first letter, cosine 0.96 to 0.94, and the mouth stuttered the first letter
(run 16, day 2); clean keys with the full efference copy in the query recall
every cue and continue after every own first letter at confidence about 1 and quiet only fading it; the value is the embedding of the symbol that came next. Write: a new
slot or a merge into the nearest slot; strength = surprise × (1 + dopamine),
own symbols carry zero surprise (corollary discharge: what the mouth wrote
was foretold). Read: attention over keys by content alone at the organ's own temperature
(strength decides durability and replay, never which memory a cue retrieves),
returning a predicted next embedding. Its recall reaches the mouth's forecast through ONE
path, `store_in`, identity at birth: what the mouth reads = the cortex's
forecast, whose norm is its certainty, + the recall as a unit direction weighted
by its confidence: two calibrated votes. The cortex is trained on its own forecast alone, day and
night (predictive coding: each area learns from its own error); recall is
never a term in that error (with the sum in the loss the day taught only the
residual the store missed and undid the night).
Recall is not an input to the cortex's stream (entered there it looked like
the current symbol and the trunk advanced it a step; the mouth read the
second letter of every answer). No direct vote on the mouth. Fade: strengths × 0.9 each night; slots below the store's own mean
strength × 0.1 are forgotten (relative, not a constant).

**PFC (the band ladder).** Bands with clocks 1, 4, 16, 64, 256, 1024, 4096,
16384 ticks: leaky integrators of the stream, each with a learned input map,
updated at its clock. Each band has a value head (the critic at that
timescale) and a Go/NoGo gate learned from the value's error. The bands'
states form the bundle the cortex reads and the PFC's forecast heads must
foresee. Each area learns from its own error: the cortex's trunk from the
next embedding it receives, the PFC's input maps from the temporal-difference
error taken with both ends live (so each band learns to hold what predicts
reward at its own horizon: dopamine shaping working memory), the PFC's
forecast heads from the bundle that follows the stream, by day and in REM,
with the stream detached: the heads learn from the cortex, they do not
rewrite it (through the trunk, six REM rounds undid a third of NREM's gain on
the next-symbol forecast: measured on run 19).

**Cortex.** A small transformer over the last W ticks, one position per
tick, of [the world's embedding + 0.5 × its own embedding in the same tick,
face, bundle] producing the stream C. All the sounds of a tick superpose in
one time step, its own attenuated by corollary discharge (measured in cortex
at a third to a half); a quiet listener's stream is then exactly the format
of the dreams the night trains on (with two positions per tick, the world's
then its own, the stream read "d . o . g ." awake and "d o g" asleep, and the
cortex forecast "d" after everything awake: run 17). No speaker sense in the
stream beyond the attenuation: what it learned after the world's "d" applies
after its own; from C, `latent_pred`
forecasts the next embedding it will receive and `pfc_pred` forecasts the
next bundle. Its lessons are prediction: the squared error to the unit
embedding received (stop-grad; the minimiser is the conditional mean of the
next embedding, so the forecast's norm is its certainty), one minus the cosine
to the bundle received (stop-grad), SIGReg on the stream as the collapse guard. It learns awake
(every K ticks on the last window, the target at every position being the
next symbol the world will say; its own symbols and rests are inputs only,
never targets, and never shift the target: one predicts the environment, and
one's own actions are not the environment) and asleep (below). It never learns from reward.

**Mouth.** Whether to act is the basal ganglia's: a gate on [C, fatigue,
mood, stress] giving p(act); zero weights and a birth bias at birth. What to
say is the lexicon read by cosine from the forecast, logits = s × cosine +
log prior, the prior being a slow tally of the symbols the world has said
(perceptual narrowing; Bayes), sampled. s is decisiveness driven by tonic dopamine: s = 5 + 5 × mood/6
(songbirds: vocal variability is high when unrewarded and falls as reward
comes), so babble is varied at birth and sharpens as smiles arrive. The gate learns by the opponent rule
(a dopamine burst strengthens Go for the context whatever it did, a dip
strengthens NoGo), plus its own reward at a symbol (the belief it had in its
choice, habituating per symbol) minus a cost that grows with fatigue.

**Face organ.** Its face is a forecast of the caregiver's, learned every tick
(head only). A readout, not a lever.

## 3. One reward system

- The face becomes dopamine as the signed prediction error of the band
  whose discount matches dopamine's (clock 16, γ 0.9375; §5b). Dopamine spreads over the last twelve ticks (0.8 per tick).
- Dopamine acts on: the store's write strength (encoding), the gate (acting),
  the value ladder at every band (expectation at every timescale, learned by
  TD as the bands tick and replayed each night), the Go/NoGo band gates, and
  the waking lesson's weight (plasticity). Never on content.
- Reward is created inside: the ladder's own error fires before a face once
  it has learned (secondary reinforcers); the body's own reward at a symbol
  drives the gate.
- Feelings: fatigue (+cost per symbol, half-life 240 ticks), stress (a leaky
  integral of dopamine dips, half-life 240 ticks), mood (a leaky integral of
  dopamine, half-life 1200 ticks), sleep pressure (+1 per tick). All on the
  body's own clock, so its physiology does not depend on the serve's speed. Stress raises the
  waking lesson's weight and flattens the gate (exploration).

## 4. Sleep

- Sleep pressure crosses the switch (wake ticks) and the body sleeps by
  itself; the typing queue empties; it wakes when the night is done.
- NREM: dreams start where the store is strongest (slots sampled by strength);
  each dream is pattern completion (value → next key → read → ...) until the
  read's confidence falls below the store's own mean; the cortex learns the
  latent lesson on every dream with the store's read OFF as input (the
  hippocampus supplies the sequence, the plasticity is intra-cortical; with the
  read on, the cortex learned to copy it and carried nothing alone), all dreams summed into one step per round,
  R rounds, sleep's own optimizer and rate.
- REM: from each dream's first symbols the cortex runs free on its own
  readout, hippocampus decoupled, and the PFC's forecast heads learn to
  foresee the bundle that follows the free-running stream (the stream
  detached: the trunk is the night's NREM work, and REM through the trunk
  undid it).
- The value ladder replays its lived pairs once. The store fades. Working
  state wakes fresh. Save and back up.
- PLUMBING: a non-finite lesson is skipped; gradients clipped; a night that
  leaves a weight non-finite is discarded and the body reloaded; a failed
  night is retried after half a day.

## 5. Physiology (the disclosed constants)

symbol cost 0.12, the gate's effort per symbol 0.12 (1 + (fatigue/10)²) · fatigue and stress half-life 240 ticks · mood half-life 1200 ticks
· wake switch 12,000 ticks · eligibility 12 × 0.8 · store fade 0.9/night ·
store forget floor 0.1 × mean strength · store read temperature 0.02 · heard tally decay 0.999 per world symbol · night rate 1e-4, rounds 24, REM
steps 8 in 6 rounds (a quarter of the night), SIGReg 0.1 · waking lesson every 24 ticks on 32 symbols at 1e-5 · value heads and Go/NoGo gates at 1e-3, the bands' input maps at 1e-5 (a slow PFC, so its states stay forecastable) ·
gate rate 0.05, birth p(act) 0.25, habituation 0.9/act, fatigue scaling /10,
own-reward weight 0.5, tonic drive 0.25, vigor weight 1.0, credit baseline 0.9 per lesson (reset at the night), spontaneous-activity floor p(act) ≥ 0.05, the stream feature scaled by 1/√d (the striatum learns from the
error against what it expected: a constant cost teaches nothing) · dream
recall adaptation 0.2 x activation per step, recovery 0.97 per step (recalled
memories tire in proportion to how much they fired; a dream ends when its
recall is unsure, or the recalled memory's adapted strength falls below the
store's forget floor, or the memory has fired to a tenth of itself (a slot
fires at most twice in a dream): a cycle exhausts itself) · readout sharpness 25 + 25 × mood/6 on the dot product forecast · lexicon, no prior (decisiveness from tonic dopamine; the lexicon's pairwise cosines of about 0.06 set the floor: at 25 a symbol at probability 0.5 outweighs fifty strangers at their noise) ·
band clocks 1..16384 · dose burst 0.5 (PLUMBING, a compute budget).

## 5b. Mathematics (the user: "not only biology, also your math")

- **The gate's rule.** A sign-following update Δz ∝ Σ A_t has expectation
  mean(A) − baseline → 0: it is vigor, not a policy. The three-factor rule
  Δz ∝ Σ A_t (a_t − p_t) has expectation cov(credit, acting), which is what
  a policy must learn. Both are kept: policy plus vigor (tonic dopamine).
  The credit is taken against a running baseline (dopamine is an error),
  which leaves the expectation unbiased and lowers its variance.
- **The absorbing state.** A policy gradient has zero expected update when
  p(act) → 0 (no acts are sampled), and above the fatigue equilibrium every
  act carries negative credit, so the gate can be driven shut and stay shut
  (fast day 3: p = 0.000). Biology's spontaneous activity never stops: p(act)
  = 0.05 + 0.95 σ(z). The floor is physiology; the learned part sits above it.
- **Scale of the gate's features (fast day 3, second failure).** The stream
  C has norm ≈ √d = 16 after LayerNorm; Hebbian steps on it moved the gate's
  logit by tens per lesson and a stale baseline (0.98 per lesson, ≈ 1,200
  ticks) kept the credit negative for hundreds of lessons after the night
  reset fatigue, so vigor drove the gate to its floor with no reward in
  sight. The feature is C/√d (unit scale), the baseline forgets in ten
  lessons and resets with the night, and the tonic drive is 0.25 so the
  no-reward equilibrium is babble, not silence.
- **Birth economics.** With a flat readout the confidence drive is ≈ 0 and
  acting pays −cost: cov < 0, silence. A tonic drive w0 = 0.25 > cost 0.12
  makes acting pay when fresh; the fatigue-scaled cost c(f) = 0.12(1 + f/10)
  sets the equilibrium fatigue f* = 10(w0 + w q)/0.12 − 10 (≈ 12 at birth, an
  acting rate near 0.3, an infant's babble):
  babble in bouts bounded by fatigue, smiles opening the contexts that earned
  them (cov > 0 there).
- **The lexicon.** If the embedding table is both input and target of the
  latent lesson, all rows drifting together makes prediction trivial
  (collapse) and the store's saved values go stale. The lexicon is fixed:
  107 random unit vectors in 256 dimensions, pairwise cosines ≈ 0.06 ± 0.06,
  so the loss has no trivial solution and the readout at sharpness 10 gives
  p(top) ≈ 0.99 for an aligned forecast of norm 1. Representations are learned
  in the cortex.
- **Calibration.** Trained by one minus the cosine, a forecast's norm meant
  nothing, so at the mouth a flat forecast and a sure one weighed the same and
  the prior's commonest letter ('l') won every tie (run 14). Trained by squared
  error to unit targets, the forecast is the conditional mean of the next
  embedding: norm 1 when one symbol follows, small when many can, and
  forecast · E_k is its probability of symbol k. The mouth reads
  sharpness × (forecast · E), the forecast being the cortex's mean plus the
  recall: two calibrated votes, and their agreement is sharp because the
  readout is a dot product. No prior is added: the unconditional mean is
  already the heard distribution, and a separate log prior counted it twice
  (with it the space, the commonest symbol, won every flat context: run 15).
- **The crowd.** A speaker embedding summed into every symbol of a bag was a
  constant every key shared, pushing every cosine toward 1: keys sharing only
  " ll " with "dog will " sat at 0.966 against the exact key's 0.998, and six
  of them outvoted it. The bags are content alone. And a memory merges into
  the best-matching slot among those that say the same: judged by the single
  best key, a slot with the same key and another value blocked the merge, and
  each hearing of "ball on" added a voter (run 17, day 6).
- **Order.** A bag of symbols is blind to order and count: after its own
  "ball" the query was nine-tenths "l", the "bal" key matched at 0.976 and
  the "ball" key at 0.962, and the mouth stuttered the l (run 18: "balll",
  "boookkk"). Under the lag code "l" at lag 0 and at lag 1 are orthogonal
  directions and "bal" and "ball" are far apart. And the read is a dot product,
  not a cosine: normalised, a context faded to norm 0.01 by 24 quiet ticks still
  recalled at confidence 0.87, its faint tail amplified into a full direction,
  and the mouth chained across lines through the pauses (run 17); as a dot
  product the confidence falls with the pause (0.88, 0.81 at six ticks, 0.30 at
  twelve).
- **Effort is convex.** The gate's cost per symbol was linear in fatigue,
  0.12 (1 + fatigue/10): at fatigue's ceiling (about 40 at a duty cycle near 1,
  production 0.12 a symbol against a half-life of 240 ticks) that is 0.59, and a
  confident, varied recitation earns a tonic 0.25 plus a novelty term near 0.45
  whatever the caregiver does. So the gate sat at 0.97 all day and the mouth
  never paused (run 19, days 2 to 6; mood only 0.3 to 0.9, the smiles being
  predicted). Effort cost in biology is convex; with (fatigue/10)² the cost
  passes the drive near fatigue 22, a duty cycle of about a half, and the mouth
  speaks in bouts bounded by fatigue, as §5b promised.
- **The recall's confidence.** The store's read is the attended mean of unit
  values, and its norm is the agreement among the memories attended. The
  largest attention weight is not: once the key carried its own bag, duplicate
  slots of one line no longer merged and split the mass eight ways at 0.11,
  every one of them saying the same letter (run 15, day 5). The store's own
  reference (the dream floor) is the same norm, each slot read by its own key.
- **Born unsure.** The forecast head starts near zero (weights at 4e-4), a
  forecast of norm about 0.1: the mouth babbles noise until the first lessons,
  rather than one deterministic junk symbol from an init of norm 4 to 9.
- **SIGReg retired.** The next-symbol forecast must hit discrete fixed
  targets; regularizing it toward a Gaussian fights the lesson. The one
  collapse risk was the bundle forecast, where prediction and target both
  derived from the stream (C constant ⇒ bundles constant ⇒ trivial forecast),
  so SIGReg stood on the stream there. With that objective detached from the
  trunk nothing can collapse the stream, and the guard has nothing to guard:
  the function stays, the term is zero. (REM without SIGReg dropped the gauge
  exactly as with it, 0.85 to 0.78; the stream detached held 0.85.)
- **The PFC's lesson.** TD with both ends live is residual-gradient TD,
  which converges to a biased fixed point (the Bellman residual); TD with a
  detached target and the previous state recomputed live one tick later is
  semi-gradient TD(0) with a learned state map, the standard convergent
  form. The bands' input maps learn from it; the stream stays detached.
- **Dopamine's horizon.** The error of the clock-1 band (γ = 0) is "reward
  now minus what was predicted for now": a smile foreseen three ticks ahead
  never fires before it lands, so no reward is ever created inside. Dopamine
  is the TD error of the clock-16 band, γ = 1 − 1/16 = 0.9375 per tick, a
  horizon of about four seconds at four ticks a second, matching dopamine's
  discount of ~0.9–0.98 per hundred milliseconds. The eligibility window
  (12 ticks at 0.8) is a separate discount, as λ is from γ.
- **The bands.** s ← s + (g/τ)(tanh(Wc) − s) is a stable leaky integrator
  for g/τ ≤ 1 (g ∈ (0,1), τ ≥ 1), time constant τ/g, states bounded in
  (−1, 1); value heads are linear on bounded states. Per-band γ_b = 1 − 1/τ_b
  gives each critic the horizon of its own clock.
- **Bounded steps.** The gate's step per lesson is ≤ rate × |credit| ×
  |eligibility|, about 0.05 × 2 × 2 = 0.2 in logit at the extreme; the night's
  optimizer starts fresh each night so its first steps are ≈ rate in
  magnitude, and every step is clipped at norm 1.
- **The store's read.** Attention over unit keys at temperature 0.02 (measured on run 7's body: 7 of 8 cues right at 0.02–0.03, diffuse at 0.05) with
  log-strength bias: for the right key (cos ≈ 1) against a near context
  (cos ≈ 0.5) the logit gap is 10, so recall is decisive; equal keys share.
  Merge at cos > 0.97 on both key and value: the same memory, stronger.

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
