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
strength × 0.1 are forgotten (relative, not a constant). Each slot carries
two marks the waking recall never reads: an end mark, set when the world's
quiet after an utterance is perceived (the offset) on the slot that holds the
utterance's last symbol under its context; and a start mark, set on the
memory whose context holds the utterance's first symbol (the second symbol's;
the first symbol itself enters under whatever the pause left of the old
context, nothing after a long one, and a memory of its own or none). Dreams
are drawn from the start-marked slots by strength (with replacement when the
starts are fewer than the night's dreams); a dream begins with its context's
own last symbol, read from the key (a key is the bag before the memory's
symbol, its newest term whole: at an onset that is the line's first symbol),
and ends where it recalls an end-marked memory: replay runs through an
episode from its first symbol to its end. (With ends alone, dreams drawn from
strong mid-line slots were four symbols long and the night's gauge lagged at
0.5; with the start mark on the wrong slot, two; drawn from onsets without
the first symbol, 'og will go', the cortex alone within lines fell from 60 to
38 of 82 on two seeds, runs 53/54, and with it read 54 and 51 on two more,
runs 59/60, the recipe's level in life; half the night drawn from any memory
instead, runs 57/58, moved that within seed noise and cost the mouth.)

**PFC (the band ladder).** Bands with clocks 1, 4, 16, 64, 256, 1024, 4096,
16384 ticks: leaky integrators of the stream, each through an input map
fixed at birth (like the lexicon), updated at its clock. Each band has a value
head (the critic at that timescale) and a Go/NoGo gate learned from the
value's error; the critic's error trains no features (features trained by a
bootstrapped error are the deadly triad, and they saturated: run 28, day 15).
The bands' states form the bundle the cortex reads and the PFC's forecast
heads must foresee. Beside the ladder's heads, the ventral critic: one value
head over all the bands' states (each centered on its running mean),
discounted at 1024 ticks, so that its error moves with the act itself (the
fast bands change within a tick) while it predicts the long run; its error
may enter the mouth's credit (vcrit_w; 0 in the recipe, §5b). Each area learns from its own error: the cortex's trunk
from the next embedding it receives, the PFC's heads and gates from the
temporal-difference error on the bands' states, the PFC's
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
one's own actions are not the environment; and where the world fell quiet for
offset_ticks after a symbol, the target at that symbol is the world's turn-end,
<eot_human>, the offset, so a line has an end and the next line's first letter
is not learned as its continuation) and asleep (below). It never learns from reward.

**Mouth.** Whether to act is the basal ganglia's: a gate on [C, fatigue,
mood, stress, salience, the level] (the last two present at zero) giving p(act); zero weights and a birth bias at birth. What to
say is the lexicon read by cosine from the forecast, logits = s × cosine +
log prior, the prior being a slow tally of the symbols the world has said
(perceptual narrowing; Bayes), sampled. s is decisiveness driven by tonic dopamine: s = 5 + 5 × mood/6
(songbirds: vocal variability is high when unrewarded and falls as reward
comes), so babble is varied at birth and sharpens as smiles arrive. The gate learns by the opponent rule
(a dopamine burst strengthens Go for the context whatever it did, a dip
strengthens NoGo), plus an innate drive to act and its own performance
dopamine at a symbol (the belief it had in its choice minus that syllable's
usual belief, a running mean per symbol: the songbird's performance error,
positive when it did better than usual, habituating as the expectation
catches up) minus a cost that grows with fatigue.

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
steps 8 in 6 rounds (a quarter of the night), SIGReg 0.1 · waking lesson every 24 ticks on 32 symbols at 1e-5 · value heads and Go/NoGo gates at 1e-3, the bands' input maps fixed at birth · the reward rate and the differential bands' state means at 1/1024 a tick ·
gate rate 0.05, birth p(act) 0.25, habituation 0.9/act, fatigue scaling /10,
the offset 8 ticks (the world's turn-end as the lesson's target after 8 ticks of the world's quiet; runs 47/48), the ventral critic discounted at 1 - 1/1024 with weight 0 in the mouth's credit (runs 51/52; 55/56 pending), the level input's weight 0 (off until runs 39/40 measure it; its scale a running root mean square at 1/1024, semi-saturation 1, clip 5), own-reward weight 0.5, innate drive 0.25 (the value form; the error form, measured on two seeds and not adopted, carried 0.70; bodies saved without a form key load as the value form), expectation rate 0.1 per act of the syllable, vigor weight 1.0, credit baseline 0.9 per lesson (reset at the night), spontaneous-activity floor p(act) ≥ 0.05, the stream feature scaled by 1/√d (the striatum learns from the
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
  lessons and resets with the night, and the innate drive (0.25 under the
  value form, 0.70 under the error form) is above the fresh cost so the
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
- **The intrinsic credit is an error.** Removed, the mouth breaks its words
  (runs 24 and 30: 19 and 21 of 48 cues finished against 45; the gate at
  0.26): once the critic predicts the smiles the external dopamine is small,
  and only an internal drive holds a bout open. Biology has it: dopamine
  neurons of a singing bird encode its performance error against its own
  expectation (Gadagkar et al. 2016), and deafened birds do not learn. The
  first form, belief × novelty habituating by repetition, was a value, and on a
  grown body it is a near-constant 0.90 per act (run 29, day 20: mean 0.90,
  spread 0.10), a drive rather than a signal. The form now is the error,
  belief minus the syllable's usual belief (a running mean per symbol at 0.1
  per act), zero-mean once expectations catch up, negative for a production
  below par; the innate drive carries the mean the value form had (0.25 +
  0.5 × 0.90 = 0.70). Run 31 against two seeds of the value form: day 6, 46 of
  48 finished against 46 and 44, the cortex's trace 63 awake against 56 and
  59; day 15, 44 against 43 and 46, the trace 63 against 62 and 59; the ladder
  bounded throughout. A newborn under the error form opens its gate to about
  0.7 before fatigue balances it (babbling); its first day's smiles matched
  the value form's. At day 20 it finished 39 of 48 against the value form's
  46 to 47 on three seeds: one cue, "dog will ", collapsed to a stutter of
  the recall ("gog will gog") in every sample, the other seven at 5 or 6
  of 6; the credit decides whether the mouth speaks, not which symbol, so
  this reads as the seed's memory, but one seed cannot settle it. The
  value form remains the recipe (innate drive 0.25). The second seed (run 34)
  decided: 38 of 48 at day 20 against the value form's 46 to 47; the error
  form is kept in the code and not adopted.
- **The slow error in the gate's credit.** The mouth's credit is the fast
  band's error; adding the 1024-tick band's error to it (runs 37 and 38, the
  parent world, weight 1) left the mouth where it was (44 and 42 of 48 at day
  20) and did not help the critics: under the corrected instrument (§7) the
  1024-tick critic read -0.19 and +0.27 at day 20 against the recipe's +0.26
  and +0.14 under the same parent (runs 35, 36; all within one day's spread).
  The slow error routed into the gate buys nothing; the weight stays zero.
- **The seam, and the offset.** The window holds a position per symbol and
  a rest per quiet tick, the waking lesson's target is the next world symbol
  wherever it stands, the store never writes the world's quiet, and a dream
  may not rest: four organs agreeing that an utterance has no end, and the
  cortex learned the seam between utterances (the served body at day 10:
  after "dog will go down" the next line's first letter at probability 1.0,
  the mouth's "downg", "dogive", "ing"). THE OFFSET: after offset_ticks (8)
  of the world's quiet, once per pause, the line's last position is marked
  ended and the waking lesson's target there is the world's turn-end
  (<eot_human>, the tokenizer's own symbol for it); a dream ends where the
  cortex alone, over the dream so far, expects the quiet; the mouth may never
  say it. Nothing enters the stream and the store keeps only what the world
  said next: three earlier forms failed on their first days (as a stream
  symbol on both sides' quiet it never fired under babble, run 41; as a stream
  symbol on the world's quiet it wiped the body's own context mid-answer, a
  world symbol clearing the own bag, runs 43/44, "go n Z"; written into the
  store it blended the quiet after a cue with the answer and the mouth read
  junk, runs 45/46). The fourth form, runs 47 and 48 (the parent world): the
  mouth 46/45/43 and 47/45/47 of 48 at days 6/15/20 against the recipe's four
  seeds at 40 to 46, 42 to 46 and 40 to 46; unheard combinations at day 20
  87 and 73 of 128 against 76 and 80; the cortex alone within lines 51 to 61
  of 82 against 56 to 64 (run 47 lower, run 48 in range); dreams shorter and
  ending where their lines end. The recipe (2026-09-04).
- **The boundary mark, and the raised body's night.** A body raised for
  thirteen days without the offset held the seam at probability 1.0, and one
  day's offset lessons could not undo it: its dreams still spliced (the
  cortex that would end them had not learned the quiet), and the night's
  lesson on spliced dreams, at ten times the waking rate, re-taught the seam
  each night. A body born with the offset never had that fight (its untrained
  cortex ended dreams early from night 1). So the store's slots carry the
  boundary marks and a dream runs from a start-marked memory to an end-marked
  one whatever the cortex expects (the cortex's own expectation of the quiet,
  used first to end dreams, ended a young body's at two symbols and is
  retired); the night then teaches the line's end at its own rate. Measured
  on runs 53/54 (fresh seeds) and on the raised body from day 17.
- **The ventral critic.** The ladder's slow heads read states that move a
  thousandth per tick, so an act's effect on the long-run prospect cannot
  show in their error within the mouth's twelve ticks of eligibility (runs
  37/38 fed that error and nothing moved). Biology's long-horizon critic
  predicts far ahead from the cue it sees now. The ventral critic: one head
  over the whole ladder. As a differential (average-reward) head it computed
  the day-scale relative value, the integral of reward above its wandering
  average, swinging by a hundred within a day, and that swing entered the
  credit ten times the fast error's size and shut one seed's gate (runs
  49/50); discounted at 1024 ticks it is bounded (spread 6 against returns'
  5 to 10) and its error after an act reads +0.10 against +0.01 after a rest.
  With its error in the credit at weight 1 (runs 51/52): the fixed cues
  equal (42, 43 of 48 at day 20), unheard combinations 81 and 101 of 128
  (the second the best of any run), and engagement up on both seeds: 275 and
  271 known words a day against 230 and 221, 106 and 124 smiles against 84
  and 88, the parent away 0.2 times a day against 1.9 and 2.6. The first
  thing the long timescale has bought. Against it, one seed's gate drifted
  from 0.55 to 0.38 over its last week (the differential form's collapse,
  slower), and its own content at 1024 ticks reads +0.20 and -0.24. Weight 0
  in the recipe until forty days on two seeds (runs 55/56) say whether the
  drift settles. They said: the gate wandered between 0.35 and 0.54 across
  forty days with the mouth at 40 to 47 of 48 and the unheard combinations
  72 to 87, no collapse; and under the parent who wants a reply the long
  return is the only path by which the parent's attention reaches the gate
  (runs 63/64 at weight 0: the run-on unmoved). Weight 1 is the recipe
  (2026-09-04, 12:20).
- **The run-on, and the end as a rest.** After a cue's answer every body
  runs on, nine or ten symbols in the next twenty-five ticks, resting under
  two first, on every recipe of this record (runs 47 to 60); the ventral
  credit at weight 1 trims a tenth. Measured causes: the store's next symbol
  after a line at confidence 1.0 from other lines sharing its last words,
  a true memory in corpora where every cue is a prefix of a longer line; the
  smile landing one to six ticks after the answer, crediting the run-on's
  first symbols, act or rest alike; and the fast parent's attention rising at
  every known word the body says and falling only at non-words, so a run-on
  of known words is rewarded at both timescales. The architecture does not
  suppress what its environment rewards; whether the parent should want a
  reply rather than a monologue is an environment decision, on the user's
  word. On the body's side: the forecast's vote for the turn's end, a symbol
  the mouth can never say, is its vote for the rest (end_rest; off in the
  recipe until runs 61/62 read), a mouth that draws the rest having not
  acted; and the night's gauge counts a dream's end as a target (it banned
  the end and read a ceiling near 0.9 before 2026-09-04 09:00).
- **The tonic drive, measured.** The gate's lesson adds to every act a tonic
  drive of 0.25 ("babble is its own reward") less an effort cost of 0.12
  convex in fatigue, beside the grounded credit, the dopamine of the next
  twelve ticks. In a lived day under the parent who wants a reply, with the
  ventral credit at weight 1 (run 67's day-6 body, scratchpad/runon_credit2.py):
  the answer's letters earned +0.105 grounded credit per tick, the run-on's
  letters −0.018, and resting after the answer +0.070. The parent's
  contingency is there, resting beating the run-on by 0.09 a tick, and the
  tonic drive's net +0.13 per act reverses it; so the run-on stood at ten
  symbols per cue on every recipe and every parent. Under the law the
  intrinsic act credit goes: runs 69/70 (the new parent) and 71/72 (the
  earlier parent) carry the gate on the grounded credit, the spontaneous floor
  and the body's own performance error alone. Adopted only if the mouth
  survives in both worlds and the run-on shrinks in the new one.
- **The level.** The gate reading the 1024-tick critic's value of the moment
  (divided by that value's running root mean square, a fifth feeling beside
  fatigue, mood, stress and salience; Pavlovian-instrumental transfer, the
  state's long-run promise invigorating the act) with a weight its own
  three-factor lesson sets: runs 39 and 40, the parent world. The mouth 40
  and 46 of 48 at day 20 (the recipe 43 and 41), smiles and the parent's
  attention seed noise, and the learned weight -0.72 and -0.27 at day 6, -0.03
  and -0.07 at day 20: the striatum found nothing in the slow value to act on.
  The 1024-tick critic read -0.15 and -0.04 at day 20 (corrected instrument).
  Not adopted; the input stays at zero like the salience.
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
- **The ladder, measured.** An instrument (one fast day on a copy; the reward
  and every band's value per tick; the realized return at each band's own
  horizon) on run 21's day-20 body: at horizons 1 to 256 ticks the value equals
  the mean return and correlates with the realized return at about zero (the
  critics learned constants); at 1024 to 16384 the values diverge (the slowest
  read 7072 against a true return near 85, correlation −0.995: semi-gradient TD
  with bootstrapping as the discount nears 1); a ridge fit from each band's
  state to its return, held out, reads nothing (R² at or below zero) up to 1024
  ticks. The corrections, each measured. Bands with clocks at or above 1024
  ticks learn average-reward (differential) TD, δ = r − r̄ + V(s') − V(s), in
  the tick's lesson and the night's replay; the discounted bands keep
  discounted TD. The reward rate r̄ is one running mean of the reward at the
  differential horizon (tonic dopamine), shared: estimated per band at the
  band's own clock, the slowest could not track the rate within a day and its
  value integrated raw reward (run 27). A gated write of the bands
  (s += g (target − s), the clock forgetting) let the states jump as the gates
  learned and the values on them diverged by day 15 (run 25); the update is
  the leaky average s += (g/clock)(target − s). The bands' input maps are fixed
  at birth: trained by the critic's bootstrapped error they are the deadly
  triad, and they ran away at every rate (1e-3: saturated by day 6 and hurt
  the cortex, run 26; 1e-5: two bands' states at the tanh ceiling by day 15
  with values past a thousand, run 28) while learning nothing this world
  offers, since the stream itself, read linearly at each clock with no
  projection, carries the return at no horizon from 1 to 256 ticks (R² ≈ 0
  held out). And a differential value is relative, defined up to a constant,
  and a linear head over raw states has two directions for that constant to
  walk in under the optimizer, its bias and the states' mean; so a differential
  head has no bias and reads its state centered on a running mean of the
  states at the reward rate's horizon (adaptation), and the gradient's
  persistent direction is gone. Linear heads on fixed features under
  on-policy TD converge (Tsitsiklis and Van Roy). Bounded through twenty
  days on every seed since (runs 29, 31 to 36).
- **Reward at long timescales has content only where the world has it.**
  Under the flat rules (a smile within seconds of any known word, a flat
  smile at a cue's completion) the critics at 256 to 4096 ticks correlate
  with their realized returns at about zero on every body (run 29 at day
  15: −0.11, −0.47, −0.34), and a smile that grows at an answer (run 33) does
  not change that: the reward is a rate, and a cue's timing is the
  caregiver's schedule. Under the parent (attention that rises at answers
  and known words, falls at babble, drifts down in silence, and sets the
  chance of a smile, the pace, and the still face; CURRICULUM.md, on the
  user's word of 2026-09-03), the critics gain content, and less of it than
  first read: single instrument days of 7200 ticks gave 0.84 and 0.91 at 1024
  and 4096 ticks on run 35's twentieth day, and those readings are withdrawn
  (a day holds seven independent windows of the 1024-tick return; the
  corrected instrument, §7, four days of 14376 ticks with the truncated tail
  cut, reads the same bodies at 0.29 and 0.38 at 256 ticks, 0.26 and 0.14 at
  1024, spreads of 0.1 to 0.3, runs 35 and 36 at day 20; the 4096-tick
  horizon cannot be judged inside a day, whose return falls as the parent
  habituates while the band's state climbs). The parent's attention is a
  quantity the body can only read through the parent's behavior, and the
  middle of the ladder carries a little of it as a value from nothing but the
  face and the page; nothing in the body acts on it yet (the slow error in
  the gate, the level: measured, not adopted). In the flat world the reward is about three hundred smiles a day at
  isolated known words, jittered within twelve ticks, against eight cue
  completions: no state predicts the next smile better than its rate. Reward at
  long timescales is a form the architecture has and a measurement the
  environment has not yet posed; a world whose reward depends on longer history
  than a word is the environment's design (CURRICULUM.md, the proposal not in
  force).
- **The PFC's lesson.** TD with both ends live is residual-gradient TD,
  which converges to a biased fixed point (the Bellman residual); TD with a
  detached target is semi-gradient TD(0), convergent with linear heads on
  fixed features. With the state map learned through the same error it was
  semi-gradient TD with nonlinear function approximation, no guarantee, and
  it diverged (runs 21, 25, 27, 28); the map is now born and kept.
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

THE PARENT WANTS A REPLY (the user's word of 2026-09-04, "permission
granted"): once its cue is answered, each further word the child adds before
the parent's next turn wears the parent's attention (the babble cost, 0.04)
and gets no smile, unless the words go on completing the cued line ('where
ball? ' 'ball under'); a word said over the parent's own typing does the
same. The answer smile is always given; nothing reads the body's insides.
Under the earlier parent a run-on of known words was rewarded at both
timescales (the run-on bullet in §5b); this is the environment's long
contingency for turn-taking, the ventral critic's to read. Measured first on
fresh seeds (runs 63/64, REPLY=1 in fastlife; --reply 1 in the served
typist) before the served body takes it at a day boundary.

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
- The value instrument (the ladder against its returns): on fresh scratch
  copies of a saved body, four instrument days of twice the plan (44 lines,
  16 cues, about 14,400 ticks) with a different caregiver seed and order each,
  the parent's rules as served; per band, corr(V_b(t), G_b(t)) with G the
  realized discounted return, counting only ticks with at least 86% of the
  horizon ahead (t < T - 2h); the mean and spread across days. One day's
  reading is not a measurement (seven windows at 1024 ticks; a single short
  day once read 0.84 where four days read 0.26); horizons of 4096 ticks and
  up are beyond a day's judgment (the return trends down through the day as
  the parent habituates, the band's state trends up as it integrates it, and
  the two anti-correlate whatever the head knows); a held-out linear "ceiling"
  from the band's 256 dimensions overfits and is not reported.
- Engagement per day (from the caregiver's log, the supervisor's read):
  smiles, the parent's turns away, its attention at the day's end and at
  smiles, the misses by kind, known words said, the parent's hit rate; and
  from saved copies the gate's weights on its feelings.
- The seam probe (scratchpad/seam_probe.py): after a heard line, the
  probability the forecast gives the turn's end, the cortex alone and with
  the store, and the next speakable symbols. The rest probe
  (scratchpad/rest_probe.py): at each cue and after each line, the forecast's
  mass on the answer's first letter against its mass on the end read as a
  rest. The run-on (scratchpad/runon.py, from the caregiver's log): in the
  twenty-five ticks after a cue, the symbols the body adds after its first
  word, and the rest it takes right after that word.

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
