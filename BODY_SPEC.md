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
  and the body's own performance error alone. They read: the gate saturated,
  0.87 to 0.98 by day 4 in both worlds. The arithmetic: once a critic predicts
  a smile, the act that earned it gets no more error, so the gate's level is
  held by whatever sits outside the prediction; the drive and the cost both
  did, cancelling; with both gone only the smiles' surprises and the vigor
  remained, which push up. The clean form, THE EFFORT IN THE REWARD
  (cost_in_reward): the cost of the last act, convex in fatigue, is part of
  the next tick's felt reward, both critics predict it, and the gate reads
  their error alone, with no tonic drive. Runs 73/74 (the new parent) and
  75/76 (the earlier) measured it at the cost the old form used, 0.12 a
  symbol convex in fatigue: the run-on fell to eight or nine in both worlds
  (the effort's doing, not the parent's), and the gates slid, 0.36, 0.18, 0.30
  and 0.41 by day 11, one seed mute with six smiles on its day 10. The
  arithmetic: at the rate the old form set, fatigue rests near 21, where a
  symbol costs 0.64 and a word more than its smile; the tonic drive had paid
  that, and with it gone every act is a loss the critics learn to predict
  and the gate learns to avoid. The scale is the one free number, how
  tiring a syllable is against a smile, and biology's answer is very: an
  infant babbles for hours. Runs 81 to 84 (the new parent) carry the effort
  at 0.03 a symbol, a word a tenth of a smile, 81/82 with the synaptic tag
  at four millionths and 83/84 without: the gates 0.90 to 0.97 on day 1, the
  babble the cost of 0.12 had turned to muteness. A knife-edge: once the
  critics predict the cost, the gate's rate goes to wherever a word's chance
  of a smile balances its cost, with no homeostat between mute and babble;
  the tonic drive against the convex fatigue was the homeostat, the mouth
  speaking in bouts at the fatigue where the two balance. The effort form is
  set aside; the tonic form stays the recipe.
- **Accepted for forty days (2026-09-04).** Runs 85/86 on the recipe as
  committed under the parent who wants a reply: gates 0.44 to 0.57 and 0.46
  to 0.55 across forty days, the mouths 45 and 42 and 48 and 46 of 48 at day
  40, the run-on 9.1 and 9.2, no collapse on either seed.
- **The critic's eligibility trace.** Run 67's value instrument at day 20:
  the ventral critic, the whole-ladder head at horizon 1024 whose error the
  gate's credit carries, read the return that followed at −0.28 (four days,
  −0.40 to −0.20) while the pinned 4096-tick band read +0.86. A theorem, not
  a surprise: TD(0) bootstrapped over a thousand steps sits at a fixed point
  whose error the horizon amplifies by (1 − λγ)/(1 − γ), a thousandfold at
  λ = 0 (Tsitsiklis and Van Roy 1997). TD(λ) with the trace of the critic's
  inputs decaying at its own horizon (λ = γ, the factor two), the backward
  view of the discounted return, captured by the error as it arrives: the
  synaptic tag on the critic's side. Behind vcrit_lambda (0 = TD(0)); the
  discounted head gains a bias for its level, born at zero in older bodies;
  the night clears the trace. Measured on run 67's day-20 body with the
  critic born at zero and carried over chained days, judged frozen at each
  day's start: TD(0) at the shared rate +0.33, +0.26, −0.10 on days 2 to 4
  as its weights' norm grew 37, 47, 54 (the runaway that brought run 67's
  head to −0.28); the trace −0.75, +0.20, +0.52; and a control with no
  ventral credit in the gate earning the same smiles (240 to 253 a day
  against 216 to 248), so run 67's 91 was the trained head's noise in the
  gate's credit. Adam at the horizon rate moved a twentieth of the level in
  a day; the normalized step at a time constant of four horizons
  (vcrit_tau) moved nothing, the total feature energy (101 a tick, 15 a
  band, 1 for the slowest) starving the bias and the slow bands, the very
  directions a 1024-tick value lives in. The ventral credit is withdrawn
  (vcrit_w 0) until a head at that horizon reads right; the theorem stands
  and the trace is kept behind its flag. The arithmetic of the consequence,
  corrected the same afternoon: a ten-symbol run-on takes the parent's
  attention from 0.7 to 0.3, and the next seven known words lose about two
  smiles over the following minutes, against a return of six to fifteen
  with a spread near five, a shift a dozen cues can show. The long-timescale
  reward is in the environment as it stands; it is the critic that has not
  read it, and the instrument that judged the critics one day at a time
  read one swing per day at those horizons. The instrument now pools the
  frozen head's readings over ten chained days, for the ventral head and
  the pinned 1024- and 4096-tick bands alike. Four rules over ten chained
  days, pooled: TD(0) at the shared rate +0.19, the discounted trace at the
  shared rate +0.01, the average-reward trace (vcrit_diff) at the horizon
  rate −0.22, the discounted trace at the horizon rate −0.10; the ladder's
  own slow heads near zero. THE CEILING (scratchpad/vceil_fit.py, a ridge
  fit of the return on the saved states held out day by day): all eight
  bands +0.38, the slow bands alone +0.51, the fast bands alone +0.33. The
  learned heads sit at a third of what the states allow; the remedy is a
  head on the slow bands learned with the trace at a slow rate, measured
  the same way. Measured (scratchpad/vpool_E/F/G.out): TD(0) at the shared
  rate on bands 5 to 7 alone reads +0.46 on nine frozen days (+0.38 to
  +0.51, none below), its weights settling near 89, against the ceiling of
  +0.51; the trace on the slow bands erratic (+0.33 pooled, −0.13 to +0.63
  by day), the trace at the horizon rate weak (+0.12). The ventral head now
  reads the slow bands (vcrit_bands "5,6,7"); its weight in the gate's
  credit stays 0: runs 87/88 (the head at weight 1 with the synaptic tag,
  twenty days) held their gates and mouths and moved nothing, and their
  day-20 instrument read the head +0.21 and −0.18, right-signed only on
  average, where run 67's body held fixed read +0.46 every day. THE AGE
  CHAINS: the same head learned from zero for five days on run 85's bodies
  held at ages 6, 16, 20, 30 and 40 read, frozen per day, +0.22 (erratic)
  at 6 and +0.67, +0.68, +0.69, +0.71 from 16 on: a newborn's four minutes
  are unforeseeable, a two-week-old's foreseeable at +0.7 by a head learned
  for one day. The live head's deficit is its newborn history or the loop
  of a head shaping the reward it predicts; runs 89/90 (the head learning
  from birth, out of the credit) separate the two: at day 20 they read
  −0.41 and −0.23 with the error never in the credit, so the history is
  the defect, twenty days of Adam steps on a newborn's noise that a day
  cannot undo. THE FORGETTING HEAD (vcrit_forget, ticks, 0 = off; candidate
  24000): the head's weights decay toward zero at a time constant of days,
  decoupled from the lesson, so it is always the last days' head, the one
  the chains showed reading +0.7 from day 16; runs 103/104 measure it from
  birth with the reliability gain at weight 1.
  And the parent's attention, read from the bands at +0.89,
  correlates +0.03 with the return: a known word raises it as readily as a
  word past the answer lowers it, so the run-on's consequence under the
  parent who wants a reply is near a tenth of a smile, below any critic's
  reach. The reply needs a parent whose reply is withheld while the child
  runs on, or a body that feels the world's words as reward when they come
  (the dopamine of information). THE WORD (2026-09-04, "both are your
  call"): both taken, behind flags. The environment's road, THE REPLY
  WITHHELD (§6: WAIT=1 in fastlife, --wait 4 in the served typist): the
  parent's smile for the answer and its recast of the cued line come after
  the child's quiet of four ticks, every symbol before that postpones them,
  the cap of 180 ticks replies anyway. The architecture's road, THE
  WORLD'S WORDS AS REWARD (world_r, 0 = off, candidate 0.1): each symbol
  the world types is felt as reward beside the face, the caregiver's voice
  as a primary reward (DeCasper and Fifer 1980; Abrams 2016; Goldstein and
  West 2003; Bromberg-Martin and Hikosaka 2009), so a turn given up pays in
  what is heard, and the parent's pace (its engagement) becomes a reward
  rate the slow critics can foresee. The arithmetic at the dopamine band
  (gamma 15/16) with the reply four ticks off: the answer smile 2 then 4
  and a recast of twenty symbols at 0.1 make a prospect near 6, a run-on
  symbol at the pause's start costs its one-tick delay, 0.29, and a symbol
  two ticks into the pause costs the reset, 0.64, against the act's own
  margin (tonic 0.25 less the effort, plus the belief credit) of 0.13 to
  0.48; so the pause, once begun, pays to keep. THE MEASUREMENT that set
  the four ticks: the served body at 33 days speaks on half of all ticks,
  one rest in a hundred reaches eight ticks, a quiet of eight comes within
  180 ticks of the parent's utterance 35 percent of the time (median 116)
  and a quiet of four every time (median 22); a newborn under an
  eight-tick wait hit the cap on two answers of three (run-on 87 and 96
  symbols). Runs 93/94 (the reply withheld) and 95/96 (and the world's
  words at 0.1), born 22:22, twenty days, the run-on and the symbols
  before the reply at days 6, 16 and 20, against runs 89/90 (the reply
  rule alone, 9.8 to 10.5 per cue at day 6).
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

- **The critic's input was the defect (2026-09-05).** The night-transfer
  instrument on the served body's day 40: the fast bands drift across a
  night (cosine at the same tick 0.70; 0.40 across ten nights), the slow
  bands do not (0.99, 1.00, 1.00; across ten nights 0.98, 1.00, 1.00). A
  ridge head from the raw slow bands reads the return at horizon 1024 at
  +0.50 on day 40's held-out ticks and +0.51, +0.49, +0.51 on the bodies of
  days 41, 38 and 30: a readout learned on one day is worth as much ten
  nights later, so the nightly cortex change was never the obstacle. The
  same ridge on the slow bands centered on a running mean at 1024 ticks,
  which is what the live critic was given (the differential heads' adaptation,
  applied to it on 2026-09-03), reads +0.01 that day and −0.44 the next. The
  live rule itself, semi-gradient TD(0) with Adam at 1e-3, reads +0.52 after
  one pass over a third of a day on the raw slow bands (+0.53 on the next
  night's body) and −0.56 on the centered ones (−0.55 the next night), the
  sign every live head has read since run 49. The arithmetic: a running mean
  at 1024 ticks tracks a band whose clock is 4096 or 16384 and leaves it a
  thousand-tick recency residual; a head on that residual learns recency,
  wrong-signed in a world that reverts. The runaway, the withdrawn weight,
  the trace, the normalized step, the reliability gain at zero and the
  forgetting head were all treatments of this one line. Discounted with a
  bias, the head needs no centering: vcrit_center 0 reads the raw state (1
  keeps older bodies' readings). Runs 125–128 measure it live from birth on
  the ear actor: 125/126 the head out of the credit with its reliability
  logged by day; 127/128 the head in the credit at 1.0 through its measured
  reliability.

- **Adam was the lockstep (2026-09-05).** The ear's listening weight and the
  bout weight grew in step at every Adam rate (net mid-bout inside a line
  +0.3 at 1e-3 and 3e-3; the bout ran away at 1e-2), and doubling the
  inside credit (the world's words at 0.6) changed nothing. The arithmetic:
  a three-factor rule's step is eligibility × credit, and the credit gap
  inside the parent's lines (0.19 for resting) is a hundred times the gap
  outside (0.002, the mouth at its credit-neutral rate); Adam divides each
  weight's step by its own gradient's running size and so moves the sparse,
  large-gap listening weight and the dense, tiny-gap bout weight at the same
  speed. The magnitudes carry the information a striatal synapse uses. Runs
  133–136 (plain SGD on the gate at 0.05 and 0.2, the vigor term off, the
  adapted input with the function kept, the ear): at day 2 the listening
  weight −0.52 and −0.43 against the bout +0.40 and +0.50 at 0.2, the first
  bodies whose listening weight leads. The cortex weights stay near birth
  under SGD (norm 0.9 against Adam's 3 to 6), the same rule at work. Day 6
  decides whether a body rests mid-word while the parent speaks.
- **The function kept under the moving mean (2026-09-05), withdrawn.** The
  adapted input subtracts a running mean μ from the gate's inputs; a gate fit
  to the raw inputs loses w·μ from its logit when centering begins (−2.06 on
  the served body switched at day 43: the gate opened, seven turnings-away).
  gate_center_keep 1 holds the function exactly: the lesson owns an
  uncentered intercept c and the bias is recomputed as c + w·μ at every tick
  (a first form that added w·Δμ to the bias each tick kept the function only
  while w stood still and drifted runs 131/132 into a gate pointing against
  its mean feature). Held exactly, it changes what the lesson does: SGD 0.2
  reached a bout weight of +3.05 by day 6 (the drifting form +1.1; the Adam
  bodies without it +1.5 at day 18), the gates split to 0.72 and 0.27, one
  mouth fell to 32/20. The mean's slow drift under uncompensated centering
  does something to the lesson's dynamics that is not derived yet, and a rule
  whose effect cannot be derived is not in the recipe: keep 0 is the form
  that lives. The compensation's right use is a one-time warm-up at a
  mid-life switch, the lesson held while μ adapts and the bias takes w·Δμ,
  exact because w stands still (not yet written).

- **The critic's second defect: one day is one ramp (2026-09-05).** On the
  cached features of the served body's day 40, the live rule (TD(0), Adam
  1e-3, raw slow bands) reads the return at +0.52 after one pass and −0.49
  after four, −0.52 after sixteen (weights 2.3 → 7.2, bias → +1.7); TD(λ)
  with the trace at the horizon holds through four passes and flips at
  sixteen the same way. A day's slow bands are one monotone ramp, and a
  768-weight head fit to it long enough learns the day's trend and carries
  it wrongly to the next day: overfitting an effective sample of one, which
  the ridge head (+0.50 across ten nights) escaped by its regularizer. The
  forgetting head on the raw bands escapes it too: weights decaying toward
  zero at 24,000, 12,000 or 6,000 ticks hold +0.52 after sixteen passes and
  +0.53 to +0.60 on the body ten nights before; at 96,000 they fail as
  without. So the long critic is three flags at once, each necessary, none
  sufficient alone: vcrit_center 0, vcrit_lambda = γ, vcrit_forget ≈ 12000.
  Runs 137–140 measure it live from birth (137/138 out of the credit,
  139/140 in it through the reliability gain).

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

THE REPLY WITHHELD (the user's word of 2026-09-04, "both are your call"):
the parent who wants a reply answers when the child has finished. Its
smile for the answer (2 then 4) and its reply, the cued line in full with
the answer the child gave (the recast a parent gives: 'why dog up? ' →
'because' → "why dog up? because big dog"), come after the child's quiet
of four ticks, a second, and every symbol the child adds before that
postpones them; after the cap (180 ticks, the typist's 45 seconds) the
parent replies anyway. The reply is spoken at once, outside the pace,
and the pace restarts from it. A parent does not praise over a child
still talking, and a child who talks over its parent gets no reply until
it stops (the contingent response infants work for, Goldstein and West
2003). Decided from the page alone; the answer smile is always given;
the smile row records the ticks waited, the symbols said before the
reply and whether the child yielded before the cap (the turn-taking
instrument, §7). WAIT=1 and REPLY_QUIET (4) in fastlife; --wait 4 in the
served typist (0 = the smile at once). Measured on runs 93/94 and 95/96
before the served body takes it at a day boundary.

## 7. Instruments (the supervisor's, never the caregiver's)

- **The critic's third defect was the learning rule, not the input.** Every reading of the critic before 2026-09-05 14:00 was within one page (the fit on a day's first part, the read on its last, or the same page through another night's body); such a head reads the parent's plan, and can read it at −0.96. The cross-page instrument (one body lives several days under different plans; a head fit on one is read on the others) gives the honest reading: on a young body a least-squares head reads unlived days at +0.66/+0.92/+0.71, the gradient head (Adam, one pass, any input normalization) at −0.49/−0.74/−0.60. A one-pass gradient head on correlated inputs reads their dominant common component (its first-order direction is XᵀG); the least-squares head is (XᵀX)⁻¹XᵀG. THE DECORRELATED CRITIC (vcrit_rls): recursive least-squares TD(λ) with forgetting, the trace carried through a precision matrix (the online LSTD; the Kalman form of TD; what a decorrelating inhibitory input layer computes), reads the unlived days at +0.63/+0.92/+0.71 from one day, one pass. Statistics A, b (float64, saved with the body), the prior vcrit_rls_delta·I kept constant under forgetting, the solve every vcrit_rls_every ticks.
- **The critic's fourth defect is the night.** The live decorrelated head matches its offline replica to three decimals, and the same estimator with the body's own statistics fit on one fresh day reads other fresh days at the ceiling (+0.65/+0.74); fit on the days the body lived it reads them at −0.7. A fresh day has no night in it; the live window always does. The night's lessons shift the slow bands coherently while the day's reward level changes, and a least-squares head learns that between-day shift as value. Arms under test (2026-09-05 evening): the head's window inside the day (vcrit_forget 4000), the homeostatic statistics re-formed at wake (vcrit_norm_wake), the bands zeroed at night as before (night_keep_bands 0). The homeostatic input (vcrit_norm_tau, slow: 36000) and the prior in units of evidence (vcrit_rls_prior) stand.
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
- The night-transfer instrument (scratchpad/night_transfer.py, 2026-09-05):
  one page lived by a saved body under the fast parent (9000 ticks, its
  world symbols, own symbols and felt rewards recorded per tick), then
  replayed teacher-forced through fresh copies of that body and of the same
  body on other nights, so every body sees the identical stream and the
  identical rewards; a ridge head from the bands to the return at horizon
  1024, fit on the first 60% of one body's ticks and read on the last 40% of
  every body's (same body = a linear head's ceiling; other bodies = what a
  head carried across those nights would read); and the bands' own drift,
  the cosine at the same tick. Its companion (scratchpad/nt_heads.py) runs
  learning rules on the cached features: ridge, TD(0) with Adam as the body
  learns, RLS, on raw and on running-mean-centered inputs. The sanity row
  (the replay's bands against the lived bands, cosine 0.98) says the replay
  is the life.
- The start/continuation probe (scratchpad/pact2.py): on saved copies living
  3000 ticks under the fast parent with the lesson off, the gate's acts
  inside the parent's typing against outside, split by the tick before (a
  START is an act after a rest, a CONTINUATION an act after an act). It is the
  yardstick the ear can move: a linear gate stops starting bouts inside the
  parent's lines long before it stops continuing them. The conjunction probe
  (scratchpad/conj_probe.py): Fisher discriminants from the cortex state
  alone for "the parent is typing", "I acted last tick" and their
  conjunction, held-out AUC, and "inside vs outside" among mid-bout ticks
  alone; it read 0.997 for the last, so the pattern turn-taking needs is
  linearly available and the limit is the lesson, not the form.
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
- **The pages instrument** (scratchpad/nt_pages.py + nt_heads7/9.py): one body lives the fast parent's plan under several seeds; heads fit on one page are read on the others. The only reading of a critic that counts; within-page readings (night_transfer + nt_heads) measure drift, not value.

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
