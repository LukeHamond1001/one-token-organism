# The first simulated body: design and plan (2026-09-23)

Status: a design, not built. Nothing in `body/` has changed and nothing is committed. The language body keeps running its current code.

Reviewed the same day against the project's laws and this Mac's limits. The whole tick, the RAM, the save and the disk were re-measured (t12, section 4.2) and corrected; the ops guard, the digest procedure, the night's episodes, the teacher's judgments and the demonstration rule were made exact; the scripted posture before sleep was removed.

Sources: three competing designs (A, B, C), a map of where language is wired into the core, the MuJoCo measurements on this Mac, and the checks run today (t11, t12, and the digest guard). The scratch scripts are listed in section 14.

## 0. The answer in brief

**Should we start the sim robot now?** Yes, beginning with the core refactor.
- The refactor is the only thing the sim waits on, and every candidate world needs it.
- It is done in a git worktree on its own branch. The served language body keeps running its current code until it moves at a night boundary (section 8.6).
- Its first step (pinning the digest guard) does not touch the served language body. The guard was re-run today: every pinned digest reproduces, and one run takes about 4 seconds.
- The review (ops/review_2026-09-22.md, section 4) deferred the refactor until after the demo. The film was cut on 2026-09-23. Another take can still be filmed during the refactor, since the served body does not change until the move.

**When are we done with the language model?** As far as the sim is concerned, now.
- The sim needs nothing more from the language body's training.
- Its 179M character weights are text only and transfer nothing to a robot. What carries over is the machinery (cortex, store, gates, critics, dopamine, feelings, night), and that is done.
- What the sim needs is the shared core refactored, with the language body's digests unchanged at every step:
  - a first sim tick is possible at about refactor day 8;
  - the refactor is complete at about day 12.
- During the refactor the language body's core code is frozen. Its teacher, scripts and flags may still change (section 8.3).
- After that the language body stays as body #1. The guard on the shared core is a tiny body that the check builds, not the live one, so the guard does not need the language body alive.
- Whether it keeps training ("training mode until talkable") is the owner's call. If "done" means talkable, that is the diary's own open question and this design does not forecast it; nothing in the sim waits for it. Stopping it would give the sim back about 1.2 cores (it uses 119% of one core), 2.5 GB of RAM, and the 1.3–1.6 GB its nightly save needs for its temporary copy.

**What we build.** Option C, "the high chair" (a planar two-joint arm, a sticky mitten, a pan/tilt head, four objects on a table, a teacher across the table), with parts taken from A and B (section 3).

**When it is born.** Birth means the moment the sim body starts living and learning with its one seed.
- About working day 19 (range 15–23) with one build session: about four calendar weeks at five working days a week.
- About day 14 if the sim world is built in a second session while the refactor runs.
- The earlier "about 2 weeks" was low: all three designs and the coupling map land at 3–5 weeks to birth.
- The next body (A's 3D arm with fingers) reuses the refactor. It costs a new world, anatomy and teacher (the S steps take 7 working days for the high chair, more for a 3D arm), plus its own life. It is not re-estimated here.

**The first thing a viewer can watch it learn.** Turning to its name and reaching for shown objects, in life days 1–5. That is about the first 3–5 wall hours after birth: a sim tick costs about 75–85 ms beside the language body (measured, section 6.2), so a life day with its night takes about 40–55 wall minutes.

**Other questions the owner will ask.**
- *What does it cost?* This Mac only: no pod, no GPU training, and no Opus call inside the tick loop.
- *Is the teacher a human teacher, as ruled on 2026-09-17?* In part.
  - The per-tick judgments (the smile, naming what is touched, tidying) are scripted. They must land within ticks, and the world waits for the child, not for Opus.
  - From the first boundary after birth, Opus listens as the diary's teacher does. It reads the scene's text log every wall minute and sets the teacher's focus (what to show, what to ask, when to demonstrate) through the queue/JSONL protocol.
  - The listener is built after birth, off the critical path (section 5.7).
- *Will it disturb the language body?* Three guards protect it:
  - the digests at every refactor commit (8.3);
  - the ops guard on the language body's own tick period (section 3, graft 5);
  - a disk rule that always leaves room for the language body's nightly save (6.4).
- *What if the Mac restarts?* The sim resumes from its last save (made each night). The save holds the body, the world, the teacher and every random stream, so only the ticks since that save are lost (6.5).

### Words used here

| word | meaning |
|---|---|
| tick | one moment of the body's clock: 150 ms of simulated time |
| life day | 24,000 ticks (one simulated hour), followed by a night |
| the core | the shared machinery in `body/core/` and `body/model.py`: cortex, store, gates, critics, dopamine, feelings, night |
| digest | the hash `tools/determinism_check.py` prints after a tiny body lives a fixed script; equal digests mean an edit changed nothing the body does |
| born code | a fixed random encoder, made once from the body's seed and never trained (like today's fixed random lexicon) |
| fovea / periphery | the sharp 15° centre of the eye / the wide, coarse 60° view |
| efference copy | the body's copy of its own last act, fed back into the cortex |
| gate | the learned "act now or not" switch; today the mouth has one, the sim gives each effector its own |
| effector | anything the body acts with: voice, arm, grip, gaze |
| working day | one day of one build session's work; five to a calendar week |
| worktree | a second checkout of the repository on its own branch (`git worktree`); the refactor happens there, never in the main tree the served body runs from |

## 1. The goal and the laws

**The goal.** A body-general architecture that ends in a humanoid robot. The first sim is body #2 on the same core as the language body. Its job is to prove the core survives a change of body: from one typed symbol per tick to sight, sound, touch, charge and several effectors at once, with a teacher.

**What "done" means for this first sim.** Seed 1 born on the refactored core, living days and nights, reaching milestone M1 (section 10) while the language digests stay exactly as pinned.

**The laws, and where this design obeys them.**

| law | what it means here | where it shows |
|---|---|---|
| Grounded reward only | Reward comes from the teacher's visible face, from contact physics (pain) and from the body's own need (charge). Nothing rewards getting closer, looking, or hearing words. | Section 5.6. `world_r` (0.3 per heard symbol) is dropped. |
| No hand-written rules or cheats in the body | Guided moves are labelled by a learned inverse model, not a hand-written rounding, and nothing tells the body it is being guided. Stops are learned; `chunk_max` is only a ceiling. The innate parts are few, named and disclosed (servo law, withdrawal reflex, weakness when empty). At night the world pauses where it is, with no scripted posture. | Sections 5.4, 5.5. A's quantiser is kept only as an instrument. |
| Survives a change of body | Each mechanism is written against the Anatomy (channels, effectors, reward sources), never against the tokenizer or this arm. | Section 8.2. The per-joint alphabet scales to 20–30 joints. |
| Disclosed constants | Every constant is in one table, with who sets it. | Section 7. |
| One seed per body | Seed 1 is the only sim life. Plumbing and timing runs have every learning rate at 0 (the full compute runs, nothing changes). The babbler is an instrument of the world, not a body, and no weight of the body ever learns from its data. | Sections 5.4, 9.2, 9.5. |
| Measure on copies, change at boundaries | The refactor lives in a git worktree on its own branch. The served language body moves to it only at a night boundary, after its digests match with the live served constants (no copy of the save is needed). The sim body's fixes follow the same rule after birth. | Sections 8.3, 8.6. |
| The environment's shape is the owner's; the teacher's method is ours | The table, objects, words, voice and charge rates are listed as owner decisions. The teacher's timings and priorities are ours and live in one file. | Sections 5.7, 12. |
| Nothing fitted to the environment's pace | The world waits for the child (lockstep). The pain threshold comes from the limb's declared force, not from babble. Store writes are gated by the body's own running quantile. | Sections 5.5, 6.1. |

## 2. The choice: three designs, one pick

- **A, an arm on a table:** a 4-joint 3D arm with a two-finger gripper, two eyes, a mocap teacher.
- **B, a creature in a room:** a wheeled body with a panning head in a 3 × 3 m room, four loose objects, a teacher with a hand, a face and a twin that demonstrates.
- **C, the high chair:** a planar 2-joint arm 8 cm above a rimmed table with a sticky mitten, a pan/tilt head, four objects, a charge pad, a teacher across the table.

| criterion | A: arm on a table | B: creature in a room | C: high chair |
|---|---|---|---|
| **Fidelity to the laws** | Clean (no `world_r`, smiles only for visible events, no distance shaping). But it labels demonstrations with a hand-written quantiser of the joint change, exact only for its own alphabet. | Keeps `world_r` (+0.3 per teacher word). Its teacher names whatever the child's gaze rests on, so that reward becomes shaping for gaze. Labels demonstrations as "the act nearest the measured wheel speed", hand-written. | The cleanest. No `world_r`. A learned inverse model, trained on the body's own acts, labels guided motion in the body's own alphabet. If the teacher's event rates fall short, the teacher changes, never the body. |
| **Speed to a visibly learning body** | Slowest: birth about day 20, 24 days in all. Reaches of 10–15 ticks leave the first move about 7% of the smile's credit. | Birth about day 17, but reward is sparse: a random child earned 1 smile in 7.5 simulated minutes. | Fastest: contact is dense (the mitten touches an object on 22% of ticks); two joints; a rim keeps objects on the table. |
| **Growth into a humanoid** | The most direct: 3D arm, gripper, gravity compensation, two eyes, a per-joint alphabet. | Weak: wheels lead neither to legs nor to hands. | A toy, until it borrows A's per-joint alphabet. |
| **This Mac, beside the language body** | Fits; three renders cost 10.7 ms a tick. | Fits; it measured the project's own `Organs` at 25.8M parameters: 6.9 ms a tick. | Fits; one render of 5–7 ms. The whole tick is about 75–85 ms (section 6.2). |
| **Main risk** | Motor credit and the schedule. | Sparse reward. | Weak object identity at 64 px. Tested today and largely fixed (section 4.1). |

**Pick: C.** It is the fastest to a body a viewer can watch learning, and the most faithful to the laws. Its two weaknesses are fixed by grafts: A's per-joint alphabet and servo law fix its path to the humanoid; B's split render (a true fovea) plus a higher-contrast table fix its vision.

## 3. What C takes from A and B, and what was left out

**From A (the arm):**
1. **A per-joint alphabet** replaces C's 19 whole-arm moves.
   - Each joint takes one of {−big, −small, 0, +small, +big}.
   - An act's embedding is the sum of fixed random per-joint rows, so the readout is one softmax per joint.
   - Striatal event rows are per joint, not one per combination.
   - This grows linearly to a humanoid's 20–30 joints. A list of whole-arm moves does not (3^7 moves for 7 joints).
2. **The servo law.** Each move starts from where the limb actually is, plus gravity compensation.
   - The target is the measured angle plus the step, re-anchored every tick.
   - A measured ticks spent pressing into obstacles fall from 90% (steps added to the old target) to 26% (steps from the measured angle) to 4.9% (with gravity compensation).
   - A guided limb stays where the teacher leaves it, with no hand-written "yield" rule.
   - The high chair's hinges are vertical, so gravity compensation is about zero at birth. It lives in SimWorld for the next body's pitch joint.
3. **The grip is its own effector, with its own gate:** {close, hold, open}. An adhesion pad at birth; fingers can later sit behind the same alphabet.
4. **Teacher rules:**
   - it starts asking only after it has itself seen each object touched (contingent, not by calendar);
   - it does not smile at the same (object, act) pair again within 120 ticks;
   - it never smiles for getting closer;
   - it frowns only for a hit on its hand above F_pain and for being talked over (the diary's authorized frown);
   - it never loosens what counts as a met ask to reach a rate; a shortfall changes what it asks and shows.
5. **The ops guard.** The language body must not slow down.
   - Its served process writes no tick timing today. Its log holds three start-up lines, and the timing is only on its page, whose ports are never touched.
   - So from R9 on, DiaryWorld appends one line every 1000 ticks to `logs/diary_pace.jsonl`: the ticks, the wall seconds, the ticks that ran over the period, and each night's length. This is logging only, outside `life`, so the digests do not move. It reaches the served body with its move (8.6).
   - The guard reads that file. The sim drops to 1× real time, and pauses if that is not enough, when either:
     - the language body's tick period over the last 1000 ticks rises more than 5% above its median (nights left out);
     - one of its nights runs more than 5% over its median (43 minutes today).
   - The sim is not born before the language body has moved and the file exists (9.5), unless the owner has stopped the language body.
6. **The pain threshold law:** F_pain = k × the limb's own declared maximum steady force, taken from the body's model file, never from the babble distribution. k = 5 gives about 20 N at the mitten.

**From B (the creature):**

7. **The eye: one 128×128 render split into two.**
   - Periphery: averaged 2×2 to 64×64, RGB plus inverse depth, over 60°.
   - Fovea: the centre 32×32 at full resolution, 15°, about 2.1 px per degree (A's central resolution).
   - One render gives both (5.7–6.7 ms measured), against A's three renders.
8. **A forecast head per sense.** Channel 0 is today's `latent_pred`, so the language body is unchanged. Equal weights across channels, disclosed. C's single superposed forecast becomes a measured alternative, not the default.
9. **The core's size from the project's own `Organs`:** d 512, 6 blocks, 8 heads, window 64 = 25.8M parameters, 6.9 ms a tick for the cortex alone (measured). The whole tick at this size is about 55–60 ms (t12, section 6.2). The awake attention cache is not needed at birth.
10. **The call.** The teacher says the child's name; turning to face it within 20 ticks earns a smile. It is the cheapest visible first act, and it brings the teacher's face into the fovea.
11. **Deadline mode.** A test switch where the world does not wait for the child. Never the default.

**Left out:**
- B's per-word reward (shaping for gaze, as above). Effort reaches reward only through charge.
- B's wheels.
- A's 4-joint 3D arm at birth: offered to the owner (section 12) and recommended for the next body.
- A's three renders.
- C's superposed forecast as the default.
- C's velocity servos: their "hold" brakes against the teacher's guiding hand. B measured a braked body moving 0.05 m under a 6 N hand against 0.44 m when limp.

## 4. What was measured on this Mac

All runs at `nice -n 19`, one thread, beside the running language body (load about 4). Treat the numbers as conservative.

### 4.1 Does a fovea fix C's vision? (t11, run today)

C's own scene and random babbler (with the teacher's tidy), 4500 ticks.
- **Label:** which object fills the fovea (at least 20 of its 1024 pixels), or none.
- **Score:** a small network (an instrument, not the body) reads the frozen visual code; 200-tick blocks, 2 for training to 1 for testing; balanced accuracy over 5 classes, chance 0.20.

| eye | C's scene (tan table) | pale grey table, objects 1.3× larger |
|---|---|---|
| periphery 64 px, born conv net (C's) | 0.46 | 0.57 |
| periphery 64 px, retinotopic code (C's) | 0.48 | 0.62 |
| fovea 32 px, born conv net | 0.56 | 0.65 |
| **fovea 32 px, retinotopic code** | 0.66 (ball 0.18) | **0.81** (ball 0.72, cube 0.85, duck 0.87, cup 0.76) |

- The frame was the bottleneck, not the code. The red ball nearly vanishes against a tan table.
- The cheapest code (fixed colour mixes per grid cell, about 0.01 ms) beat the random conv net on both eyes. The body uses it for both.
- Rendering at 128 px cost 5.7–6.7 ms, no more than 64 px (5.3–6.4 ms).
- The readouts had about 130 training examples per class; the cortex will see far more.

### 4.2 The simulator, senses and core

| what | measured | script |
|---|---|---|
| Rendering with no window | Works (CGL on macOS, no `MUJOCO_GL`). Depth warns `ARB_clip_control unavailable` but is accurate to 1.3 mm at 1.5–2 m. Sky reads 301 m, so depth needs a cap. | t8_depth.py |
| Physics, 2-link arm (2 ms steps) | 200k steps/s in a Python loop (400× real time) | t1_arm.py |
| Physics, gym humanoid (17 motors) | 17.9k steps/s (54× real time): a later humanoid is affordable | t3_legged.py |
| Full sim tick with a scripted teacher (B's room) | 8.2 ms per 150 ms tick (18× real time), 104 MB | t6_teacher.py |
| Core: `Organs` d512, 6 blocks, window 64 | 25.8M parameters; 6.9 ms a tick (full-window forward every tick, waking lesson every 24 ticks) | t10_core_step.py |
| The whole language tick at the sim's size (d 512, 6 blocks, 8 heads, window 64; the served save's constants; one thread; a small store) | mean 55–60 ms, median 37–45 ms, with spikes of 0.5–0.7 s about every 25 ticks; 26.3M parameters. Profiled: about a third of the time goes to the critics' float64 least squares (two 4097 × 4097 matrices), about a quarter to the cortex (its full window recomputed about 1.5 times a tick) | t12_life_tick.py, t12b_profile.py |
| Store read at capacity (× 512) | 32,768 slots: 2.9 ms; 65,536 slots: 5.3 ms | t12_life_tick.py |
| A save and RAM at that size | a save of 424 MB with an empty store (268 MB of it the two float64 matrices); peak RSS 1.3 GB | t12c_sizes.py |
| The digest guard | every pinned digest in section 8.3 reproduced today; one run takes about 4 s | tools/determinism_check.py |
| Generic cores, forward / learn, 1 CPU thread | 3M: 4.3 / 12 ms; 19M: 20 / 77 ms; 57M: 44 / 217 ms. Two threads were slower (the language body holds the cores). A 179M core cannot learn every tick beside the language body. | t5_encoders.py, t5b_mps.py |
| Teacher's voice | macOS `say` to a 16 kHz WAV, read with `wave`. Identical every time, so cache once (about 15 KB a word). Words last 0.34–0.62 s, about 3 ticks. | t4_audio.py, t4b_rates.py |
| Ear | 40-band log-mel in numpy: 15×40 = 600 numbers per tick in 0.04 ms | t4_audio.py |
| Telling words apart | One voice across speaking rates: 100%. Four voices: 56% (chance 17%). Start with one voice. | t4_audio.py |
| State and replay | Full sim state 2.3 KB a tick; replaying the same commands gives exactly the same result; frames redraw from states at 251/s | t7_state.py |
| Pain threshold needs a body's own law | B's bumper read over 5 N on 509 of 3000 random ticks | t6b_bump.py |

## 5. The body at birth

### 5.1 The world

MuJoCo 3.9, one file `body/sim/highchair.xml`, 2 ms physics steps, 75 per tick.
- **Table:** 0.8 × 0.64 m, pale grey, with a 3 cm rim.
- **Backdrop:** a room behind the table, so a random gaze does not fill the eye with black.
- **The child's arm:** planar, 2 joints (links 0.22 m and 0.19 m), 8 cm above the table. A dark downward mitten carries a touch sensor and the adhesion grip.
- **The child's head:** 0.36 m up; pan ±60°, tilt −75° to +20°.
- **Objects:** ball, cube, duck, cup; 60 g each, 1.3× C's size, saturated colours.
- **Charge pad:** a flat magenta disc of radius 4.5 cm (no object shares its colour). It charges while the mitten is over the disc.
- **The parent (the owner's decision, 2026-09-23: "the teacher a robot like the model itself, teaching it like a parent would"):** a second robot of the same body across the table: the same planar arm and sticky mitten, the same pan/tilt head, and a face (a two-capsule mouth that shows smile and frown). It is driven by the scripted SimTeacher below (motion by inverse kinematics inside the same per-joint alphabet), with Opus steering what to teach between minutes. It shows by doing: it pushes, takes and names with its own arm, in the child's view, so imitation is of a body like its own; it can still guide the child's forearm (the soft weld below). Its arm has physics like the child's, so its touches and pushes are real events in the world. (A raised body could one day be the parent of the next: out of scope for the first life.) The original mocap teacher's parts follow:
  - The charge drain counts only the arm's own actuator torque, capped by its declared limit, so being guided costs at most that cap.
  - The weld's constraint force is not a contact, so it is never pain.
  - S1 measures the tracking and the servo's torque under a scripted guide.
- **SimWorld:** `frame()`, `apply(acts)`, `pause()`/`resume()`, state save and restore (the world's and the teacher's random streams included), and an exact replay check.

### 5.2 Senses per tick

Each channel's code is born (fixed, random, from the body's seed), unit-scaled and projected to d = 512.

| channel | kind | raw | born code |
|---|---|---|---|
| `words` (the partner) | symbol | the teacher's word token, arriving on the tick its sound ends; rest otherwise | a fixed random lexicon: about 20 words, the child's name, rest and end; letter rows reserved for the fallback |
| `ear` | vector | 15×40 log-mel: the parent's cached voice clips scaled by distance, the child's own voice, and the world's sounds (the owner's decision, 2026-09-23): contact clicks, and each object's own sound when it is touched or moves (the duck squeaks, the ball taps, the cup clinks, the cube knocks; cached clips, scaled by distance and speed) | fixed random projection |
| `eye_p` (periphery) | vector | 64×64 RGB + inverse depth (capped) | 8×8 cells × 6 random colour-depth mixes = 384, then a fixed projection |
| `eye_f` (fovea) | vector | 32×32 RGB | 8×8 cells of 4 px × 6 mixes = 384, then a fixed projection |
| `body` | vector | sin/cos of the 4 joint angles (arm and head), velocities, grip state, servo effort | fixed linear map, scaled by the ranges in the model file |
| `touch` | vector | mitten force (log), contact, held object, a hand on the forearm | fixed linear map |
| `charge` | vector | h and Δh | fixed linear map |
| face | scalar | [face/6, Δface/6] through today's `face_in` | unchanged |
| own acts | symbol × 4 | each effector's act on the last tick, at `own_gain` | each effector's fixed table |

- **Cortex input:** the channel codes summed in a fixed order (words, face, bundle, own acts, then the new channels), then `in_ln`. This keeps the language body's float order.
- **One forward a tick.** The sim's own acts enter at the next tick's frame, as efference copies, so its cortex runs one full-window forward a tick. The language body keeps its two halves, with its own symbol entering in the same tick.
- **Forecasts:** one head per channel.
  - The words head is read by a dot product with its own lexicon, exactly as today. It keeps today's target: the partner's next word, and the end at an event's end.
  - Each vector head predicts its channel's born code at the next tick, by the squared error `latent_loss` already uses.
  - Own acts are inputs only, never targets (as today: one's own actions are not the environment).
- **Event ends:** today's settle law (`offset_fast` 4, `offset_slow` 64, `offset_settle` 0.5) applied to the tick's summed forecast error, all channels weighted equally. Store marks, episode cuts and the night's event boundaries fall there. Turn-taking (pace_sense) stays on the words channel alone.
- **World truth** (object poses, labels) goes to the teacher and the instruments only. It never enters the body.

### 5.3 Effectors

| effector | alphabet | rest | gate inputs besides [C/√d, fatigue, mood, stress, salience, level, charge] | cost |
|---|---|---|---|---|
| 0 `voice` (keeps the names `mouth_gate`, `opt_gate`, `gate_buf`) | the `words` lexicon, shared with the ear (so imitation comes free, as today) | quiet | the partner's word this tick or held, own act last tick; pace_sense M1–M5 run on `words` | fatigue 0.12 a word (today's `symbol_cost`); no charge beyond the resting drain |
| 1 `arm` | 2 joints × {±0.27, ±0.09, 0} rad per tick, per-joint | hold | touch onset, pain, a hand on the forearm, own act last tick | fatigue 0.12 × Σ τ²/Σ τ²_max per tick of motion; charge 3e-4 × Σ τ² per tick |
| 2 `grip` | {close, hold, open} | hold | mitten contact, own act last tick | fatigue 0.03 per close or open; no charge beyond rest |
| 3 `gaze` | pan and tilt × {±0.2, ±0.07, 0} rad per tick, per-joint | hold | onset on `words`, eye surprise, own act last tick | fatigue 0.03 per step; no charge beyond rest |

- Every effector has its own gate, gate buffer, optimizer and three-factor lesson: today's code, looped over effectors.
- The new effectors' striatal event blocks are appended after the language block.
- **Where proposals come from.** The voice reads the words forecast, as today. The arm, grip and gaze each read a new `act_pred` head, with the same readout law. The striatal actor biases every proposal, as today.
- **The voice's clips.** A word act plays its clip in the child's voice (about 3 ticks). A new word act cuts the clip that is playing. The teacher hears only finished words, so the body can learn not to cut itself off, with no rule against it.
- **Switches at birth.** The sim is born with the defect fixes #1, #4, #5, #6 and #8 switched on, and with `chunk_gate` 1 for every effector. The language body keeps them off until each is measured on a copy.

### 5.4 Learned stops and demonstration

- **`act_inv`:** a small network mapping (body_t, body_t+1) to the arm's per-joint act. It learns online from birth, on seed 1's own acts, with the efference copy as the label. It never sees the babbler's data.
- **`act_pred`'s target at every tick** is the act that explains the arm's motion from t to t+1:
  - its efference copy when its gate acted;
  - `act_inv`'s label when it rested, whatever moved it (the teacher's hand, a collision, or nothing, which reads as hold).
  - The label's weight is `act_inv`'s running reliability on the body's own recent acts (the reliability estimator the critics already use), so demonstrations count only as far as the inverse model has earned.
  - The grip and gaze have no inverse model at birth: when they rest, their target is their rest.
- **Kinesthetic demonstration:** the teacher's hand holds the forearm (section 5.1) and moves it at most 0.27 rad a tick per joint, inside the alphabet, saying the words.
  - Nothing tells the body it is being guided. It feels the hand on the forearm as touch, and the motion through its joint sense.
  - This solves the correspondence problem in a grounded way: the teacher moves the body's own limb.
  - A's nearest-move quantiser is kept as an instrument that scores `act_inv`, never as part of the body.
- **A chunk of acts continues until** (this generalises today's `chunk_gate`):
  - `act_pred`'s best guess is the effector's rest (the learned end);
  - the effector's own gate draw closes (with defect #4 fixed);
  - the withdrawal reflex fires;
  - `chunk_max` is reached. It is 8, the served language body's value (the physiology default is 12), and it is a ceiling, never the usual stop.
- **What teaches the stop:**
  - the teacher judges an ask met only once the effector that did it has come to rest on the result (section 5.7), as the diary's teacher smiles only at a word bounded by a rest;
  - demonstrations end in a hold at touch;
  - touch onset is a surprise, and an event's end falls there.

### 5.5 Innate mechanisms (disclosed; spinal or anatomical)

- **The servo law** (section 3, graft 2).
- **Withdrawal reflex:** when a link feels force above F_pain, the arm reverses its last move for 2 ticks. Those ticks are logged as reflex and give the gate no eligibility. The cortex sees them through joint sense and touch.
- **Weakness when empty:** torque limits × (0.3 + 0.7h).
- **Night:** `world.pause()` freezes the world exactly where it is. Nothing is moved, and the morning resumes from the same state. There is no scripted posture before sleep.

### 5.6 Reward (summed in this order)

1. **Face:** today's rule (senses.py:100-107). A rise in the face's size, or a change of its sign, is felt as the new level, clipped to ±2; a held face and its easing off are not felt. Grounded in the teacher's visible judgment of world events.
2. **Pain:** −1 on a tick whose peak contact force exceeds F_pain (about 20 N). C measured it on about 5% of contact ticks in babble. Grounded in contact physics.
3. **Charge:** 4 × [D(h_t) − D(h_t+1)], with D(h) = (1 − h)². This is drive reduction, so sitting on a full charge earns nothing.
   - Drain: 4e-5 per tick at rest, plus 3e-4 × torque². Babbling empties a full charge in about 8000 ticks, so about 3 recharges fit a life day.
   - The pad adds 0.01 per tick.
   - This is homeostatic reward (Keramati and Gutkin, 2014). Summed over a whole drain and refill it nets zero, so the need exists only through the critics' discounting: the discounted sum equals D(h_0) minus (1 − γ)/γ times the discounted drive. The need is therefore weak for the slow critics, and M2's "goes to the pad on its own" rests on it (risk 10).

Left out: `world_r`, any reward for getting closer, novelty bonuses, and anything read from the body's insides.

### 5.7 The teacher (SimTeacher, inside the loop)

It counts sim ticks and keeps every constant in one file, `body/sim/teacher_consts.py`.

**What it perceives:** object poses, the child's visible arm, head, grip and gaze, contacts, the charge light, and the child's finished words. Never gates, critics or the store.

**What it does, in priority order:**
1. **Tidy:** an object out of reach for 20 ticks is put back, with "here". Required: without it, all 4 objects were out of reach within 7.5 minutes.
2. **Call:** it says the child's name, at most once per 240 ticks and only while the child's gaze is off the teacher. Turning to face it within 20 ticks earns a smile. "Facing" means the teacher's face fills at least 20 of the fovea's 1024 pixels on 2 ticks (the label t11 used for objects).
3. **Name:** when the child's gaze stays within 10° of a visible object for 2 ticks, or the mitten touches or holds an object, it says the name. At most once per 20 ticks per object.
4. **Show:** at most once per 240 ticks, when nothing above is under way, it carries an object 0.3 m in front of the eye and names it twice.
5. **Ask:** "look cup", "push cube", later "take duck". Only after it has seen each object touched. It waits 40 ticks and judges from the world. An ask is met:
   - "look X": X fills at least 20 of the fovea's pixels on 2 ticks;
   - "push X": X has moved at least 3 cm while the mitten touched it, and is still within reach;
   - "take X": the mitten has held X for 3 ticks;
   - in each case, only once the effector that did it has come to rest (the act complete).
6. **Demonstrate:** on every second failed ask, it guides the arm through the act (section 5.4), saying the words.
7. **Answer:** when the child says an object's name in a pause, it brings that object to the child's hand. The word changes the world.
8. **Face:**
   - +2 for a met ask, or for a right name (the name of what is in the child's fovea or hand);
   - no repeat smile for the same (object, act) within 120 ticks;
   - a frown for talk-over, or for a hit on its hand above F_pain;
   - it holds the face 10 ticks, then eases off (today's rule: the easing is not felt).

**Its voice:** one voice (Samantha) at rates 140, 200 and 260; about 20 words; cached once, about 1 MB. The child's words are cached in a second voice.

**Its rates:** checked with the random babbler before birth, aiming for at least 1 smile per 200 ticks.
- If it falls short, the teacher changes what it asks and shows, and how often. It never loosens what counts as met.
- The babbler's smiles per ask are written down as the chance level the body must beat.

**Its log:** everything goes to a JSONL log, rotated at 100 MB.

**Opus as the listening teacher (the owner's ruling of 2026-09-17).** From the first boundary after birth, Opus reads a text log of the scene every wall minute, as `ops/parent_brief_human.txt` has it listen in the diary. It sets the teacher's focus (which object to show next, what to ask, when to demonstrate) through the queue/JSONL protocol.
- It never acts inside the tick loop. The per-tick judgments above stay scripted, because they must land within ticks.
- It is built after birth, off the critical path, and joins at a boundary like any change of method.

### 5.8 The owner's decisions of 2026-09-23 (evening): faces, looking, two ears, two new parts

These override the sections above where they differ. They add about 3–4 working days before birth (S1–S3).

- **Both robots have faces.**
  - **The parent's face** (a mouth that smiles and frowns, eyes, a head that turns) is the reward's only carrier. The number-beside-the-mouth scaffold of decision 3 is dropped.
  - **The child's face** shows its own face readout (the face organ, ARCHITECTURE.md §6: its forecast of the felt reward) as an expression on its own head. The parent sees it, as the diary's caregiver sees the page.
- **The child must look to be rewarded.** A smile or frown is felt only while the parent's face is in the child's fovea (the same test the teacher already uses for "facing": at least 20 of the fovea's 1024 pixels on 2 ticks). An unseen smile is not felt. This makes joint attention (look at the face, follow its gaze to the thing named) the road to reward.
- **Two innate mechanisms** (disclosed, like the withdrawal reflex):
  - **Orienting:** a born bias of the gaze effector toward a face-like blob in the periphery and toward the side a voice comes from (the two ears below). A bias on the gaze proposal, never a forced move; without it a newborn that never looks is never rewarded.
  - **A born expression reading:** when the parent's face is in the fovea, a fixed read of the mouth's curvature gives the smile or frown level that the face reward (section 5.6, item 1) feels. It reads the rendered mouth only; world truth still never enters the body.
- **Two new parts from birth:**
  - **A valence learner (fast, one-shot-ish):** it learns which cues predict the good and the bad (the parent's face turning, a voice's tone, what hurt), from the felt reward and pain. Its output biases orienting toward or away and raises the fast memory's write strength for those moments. It is disclosed as the one new organ; its learning rule is the critics' own least squares on a short horizon.
  - **A motor timing part:** the forward model (`act_pred`) and inverse model (`act_inv`) of section 5.4, named as one part and extended to predict each effector's next body sense, so reaching is corrected smoothly within a chunk.
  - Other parts (attention gating, a learned visual hierarchy, body-mapped motor and touch areas) are added only when the body shows it needs them, each measured.
- **Hearing, the human way at the level of function:**
  - **Two ears** on the child's head (left and right), each a fixed cochlea-shaped filterbank (gammatone-like, log energy, 40 bands), so the interaural level and time differences tell the side a sound comes from.
  - **The parent's voice** (synthesised, one voice at three rates) and **the world's sounds** (contacts; each object's own sound) reach both ears scaled by distance and direction.
  - **The words channel** (the parent's word tokens) stays at birth as a scaffold and is removed at a boundary; hearing words from sound alone is itself a milestone.
  - The child speaks cached word clips in its own voice; a vocal tract that babbles is a later body.
- **No live human teacher.** The parent robot raises it day and night, with Opus steering its focus between minutes. The owner does not teach live.

## 6. Tick, speed, size, compute and disk

### 6.1 The tick

- 150 ms of sim time, the same as the language body, so every constant counted in ticks carries over.
- The world waits for the child (lockstep), so nothing is fitted to wall-clock pace.
- A viewer's watch mode runs at 1×; the body sees the same sim time either way.
- Deadline mode (the world does not wait) is a test switch only.

### 6.2 Wall time per tick

The first estimate here (23–30 ms) counted the cortex and guessed 5–10 ms for the rest. The rest was measured today (t12) and is about 45–50 ms.

| part | ms | source |
|---|---|---|
| physics (75 steps) | 3.8–5.8 | measured (t9) |
| eye (one 128 px render, split) | 5.7–6.7 | measured (t11) |
| born codes and ear | under 0.5 | measured |
| the whole language tick at d 512 under the served constants: cortex, gate, critics, striatum, bands, feelings, a small store | mean 55–60 (median 37–45; spikes of 0.5–0.7 s about every 25 ticks) | measured (t12) |
| store read at 65,536 slots (the t12 run's store was small) | 5.3 | measured (t12) |
| three more gates, `act_pred`, `act_inv`, six more forecast heads | 2–5 | not measured (small linear layers) |
| **total** | **about 75–85 (mean)** | about 1.8–2.0× real time beside the language body |

- **Where the time goes (profiled).** About a third goes to the critics' float64 least squares: the fast critic's evidence `vf_A` and the face head's `_fh_A`, both 4097 × 4097 because their features are the 8 bands × 512. Each gets a rank-one update every tick and a full solve about every 25 ticks, and those solves are the spikes. About a quarter goes to the cortex.
- **Speed-ups.** Each is measured for compute only (every learning rate at 0), and none becomes a default except at a boundary:
  - the cached awake stream (8.5);
  - fewer bands in the critics' features (a physiology change of the sim body, disclosed).
  - The one-forward-a-tick rule (5.2) is already in the design. It saves about 3 ms against the measurement.
- **A life day.** 24,000 ticks take about 30–34 wall minutes awake.
- **The night.** Its cost is not yet measured. The language body's night (179M, 4 threads) takes 43 minutes. Scaled by the core's compute (about 7× smaller) and by one thread against four (about 2.5× slower), the sim's night should take about 10–20 minutes. S5 measures it.
- **A life day with its night** therefore takes about 40–55 wall minutes.

### 6.3 Learned parameters

| part | parameters |
|---|---|
| `Organs` (d 512, 6 blocks, 8 heads, window 64; measured with a lexicon of 60) | 25.8M |
| 7 channel forecast heads | 1.8M |
| 3 `act_pred` heads | 0.8M |
| `act_inv`, 4 gates, the striatal blocks | under 0.2M |
| **total learned** | **about 28M** |
| born codes (fixed, not learned) | about 0.7M |
| the critics' least-squares state (float64; solved, not trained by gradient): two 4097 × 4097 matrices | 268 MB, in RAM and in every save (measured) |

### 6.4 Memory and disk

- **Store:** 65,536 slots × 512, the served language body's capacity: 268 MB in RAM; a read costs 5.3 ms (measured).
  - **Writes:** every heard word, plus the frames whose surprise is above its own running 0.9 quantile (2,400 a life day). That is about 3,400–4,800 writes a life day before merges, depending on how much the teacher talks.
  - **The twenty-ninth defect's rule** (BODY_SPEC.md): the capacity should hold about 19 nights of writes, so that the nightly fade does the forgetting, not the cap. 65,536 slots hold about 14–19 life days at these rates.
  - **Setting it:** S3 and S5 measure the write rate. If 19 days of it exceed 65,536 slots, the cap rises before birth (disclosed; each 32,768 slots cost 134 MB of RAM and of every save).
  - Writing every tick would fill it in under 3 life days.
  - The first draft's 32,768 slots would have held 7–10 days, which is the fault the twenty-ninth defect named.
- **The day's tape and the episodes (the sim's utterance memory):**
  - The tape holds each tick's born codes and acts, about 3.6 KB a tick in fp16 (86 MB a life day).
  - At nightfall the tape is cut into episodes at event ends (5.2). Each episode is tagged by the reward felt and the surprise inside it (fix #6).
  - The night draws episodes by tag, as the language night draws utterances.
  - Episodes are kept up to `episode_cap`, 24,000 ticks (86 MB), with the weakest giving way. They are saved with the body, so a rewarded episode can be replayed on later nights.
  - Nothing is re-rendered and no pixels are stored.
- **Disk:**

| item | size | kept |
|---|---|---|
| the save, written as `.tmp` then renamed (the language body's own pattern, so a whole save always exists) | about 0.8 GB: 424 MB measured with an empty store, plus the store's 285 MB and the episodes' 86 MB. The `.tmp` takes another 0.8 GB while it writes | one file |
| sim states for replay | 55 MB a life day | last 3 days (165 MB) |
| the teacher's JSONL log | rotated at 100 MB | the last file |
| voice cache | about 1 MB | always |
| **peak** | **about 1.9 GB** | |

- **The language body's share of the same disk.**
  - Its save, `data/watch2.pt`, is 1.28 GB. Every night it writes a whole `.tmp` copy before renaming (persistence.py:33).
  - Its store held 25,405 of its 65,536 slots at its last load (d 1024), so the file grows by about 0.33 GB as the store fills, and the `.tmp` grows with it. It needs up to about 1.6 GB free each night.
  - If that save fails, the body schedules an extra night (review defect #16) and its memories fade twice. The sim must never cause that.
- **The budget.** 5.8 GB was free at 20:00 on 2026-09-23. Take away 0.33 GB (the language store filling) and 1.9 GB (the sim's peak): about 3.5 GB is left at the worst moment. The language save needs 1.6 GB of that, leaving about 1.9 GB to spare.
- **The rule.** The sim writes nothing (save, states or log) that would leave less free space than the language save's size plus 1 GB. It skips the write, logs why, and pauses at its next boundary.
- **No films while it lives.** No films or other large files are written while it lives. The eye's inputs alone would cost about 0.5 GB a life day (uint8), and the full 128 px renders about 1.2 GB.

### 6.5 The process

- One torch thread (the digest depends on the thread count), `nice 10`.
- **RAM:** about 1.9–2.1 GB. The t12 run measured 1.3 GB with an empty store; add the store's 0.29 GB, the tape and episodes' 0.17 GB, and about 0.1 GB for MuJoCo and the renderer. The language body holds 2.5 GB, and 16 GB leaves room.
- Its own page `/sim` on port 8030. Never 8020, 8021 or 9333.
- The ops guard (section 3, graft 5) protects the language body.
- **The GPU.** Torch runs on the CPU (no MPS). MuJoCo's offscreen renderer uses the GPU through CGL, for one 128 px render a tick. The language body does not use the GPU.
- **A restart** (a crash, a reboot, an update) resumes from the last save, made each night. The save must hold the body, the SimWorld state, the teacher's state and every random stream (the world's, the teacher's, the body's), so the lockstep world continues exactly and only the ticks since that save are lost. S1 and S5 test it.

## 7. The disclosed constants

| constant | value | kind | set by |
|---|---|---|---|
| tick | 150 ms of sim time | clock | ours (today's) |
| physics step | 2 ms, 75 per tick | world | ours |
| life day | 24,000 ticks | physiology | ours (today's) |
| arm step sizes | ±0.09, ±0.27 rad per tick | anatomy | ours (C's measured joint speeds) |
| gaze step sizes | ±0.07, ±0.2 rad per tick | anatomy | ours |
| servo law | target = measured angle + step, plus gravity compensation | innate | ours |
| F_pain | 5 × the mitten's declared stall force (about 20 N) | innate | ours (from the model file) |
| withdrawal reflex | reverse the last move for 2 ticks | innate | ours |
| weakness when empty | torque limits × (0.3 + 0.7h) | anatomy | ours |
| face reward | a rise in the face's size or a change of its sign, felt as the new level clipped to ±2; a held face and its easing off not felt | reward | ours (today's) |
| pain reward | −1 per tick over F_pain | reward | ours |
| charge reward | 4 × [D(h_t) − D(h_t+1)], D(h) = (1 − h)² | reward | ours |
| reward order | face, pain, charge | reward | ours |
| charge drain | 4e-5 per tick + 3e-4 × Σ torque² (the arm's own actuator torque, capped by its declared limit) | world | owner (decision 5) |
| pad refill | 0.01 per tick, with the mitten over the disc (radius 4.5 cm) | world | owner (decision 5) |
| fatigue per act | voice 0.12 a word (today's `symbol_cost`); arm 0.12 × Σ τ²/Σ τ²_max per tick of motion; grip 0.03 per close or open; gaze 0.03 per step | physiology | ours |
| forecast head weights | equal across channels; vector heads by squared error to the next born code | physiology | ours |
| event end | today's settle law (`offset_fast` 4, `offset_slow` 64, `offset_settle` 0.5) on the summed forecast error | physiology | ours (today's) |
| store capacity | 65,536 slots (raised before birth if 19 days of measured writes exceed it) | physiology | ours (the served value; the twenty-ninth defect's rule) |
| store write gate | surprise above its own running 0.9 quantile | physiology | ours (the pace technique) |
| episode cap | 24,000 ticks of episodes kept for the night | physiology | ours |
| `chunk_max` | 8, a ceiling only | physiology | ours (the served language body's value; the default is 12) |
| `act_pred`'s label weight when the arm rested | `act_inv`'s running reliability on the body's own acts | physiology | ours |
| switches at birth | fixes #1, #4, #5, #6, #8 on; `chunk_gate` 1 per effector | physiology | ours (off for the language body) |
| eye | 128 px render → 64 px periphery (60°) + 32 px fovea (15°) | anatomy | ours |
| born codes | retinotopic 8×8×6 per eye; random projections from the seed | anatomy | ours |
| teacher timers | tidy 20, call window 20, call gap 240, show gap 240, name gap 20 per object, ask wait 40, smile repeat 120, face hold 10 ticks; a demonstration on every second failed ask | teacher method | ours (one file) |
| teacher judgments | gaze on an object: within 10° for 2 ticks; "in the fovea": at least 20 of its 1024 pixels on 2 ticks; push met: moved at least 3 cm under the mitten and still within reach; take met: held 3 ticks; any ask met only once the effector rests; smile +2; frown for talk-over or a hit on its hand above F_pain | teacher method | ours (one file) |
| teacher's guide | a soft weld on the forearm; at most 0.27 rad a tick per joint | teacher method | ours (one file) |
| teacher words and voice | about 20 words + the child's name; one voice at 3 rates | world | owner (decision 4) |
| table and objects | pale grey, rim 3 cm, objects 1.3× | world | owner (decision 1) |

Every language-body constant keeps its current value and place (flags, physiology, or the save).

## 8. The core refactor

The core is already body-general in its machinery. Language enters it through about nine seams. The refactor puts four interfaces at those seams (Anatomy, Senses, Effectors, RewardSources) plus a World loop, and keeps the language body identical.

### 8.1 Where language is wired in today (line numbers at commit c58acbc; `body/` is unchanged since)

| seam | where today | becomes |
|---|---|---|
| A. The alphabet and the one-symbol page | `Life(organs, tok)` life.py:46-51; `type_text` senses.py:183-191; `_sense` senses.py:86-87; queue and page life.py:221-224; decoding in instruments.py:15-75; `vocab` in persistence.py:18-40, 157-159 | Anatomy declares the channels; `World.frame()` gives the latest frame (no queue for the sim) |
| B. Reserved symbols (rest, end) | life.py:110-121; `self.sil` 77 uses, `bans`/`reserved` 20, `end_id`/`eot` 19 | each channel and effector declares its own rest, end and reserved ids |
| C. The mouth as the only effector | `_choose` mouth.py:276-526; `_act` mouth.py:528-582; readout model.py:623-635; chooser model.py:516; actor model.py:726; eligibility actor.py:17; the space as word end | a list of Effectors, per-joint alphabets, `act_pred`, learned stops |
| D. Gate and pace_sense | gate features mouth.py:293-343; floor mouth.py:357-397; lesson mouth.py:613-661 (general); pace M1–M5 mouth.py:49-274; senses.py:92-99, 148-168; gate width model.py:541, life.py:229-230 | turn-taking only on the voice against the declared partner channel; motor gates get their own "ear" inputs |
| E. Store keys | memory.py:66-117; writes cortex.py:44-74; marks model.py:56-58, 267-308 | keys from frame codes; writes gated by surprise quantile; marks at event ends. The store engine is unchanged. |
| F. Cortex input and readout | `inputs` model.py:561-576; forecast model.py:582-588; `latent_loss` model.py:812-832; waking lesson cortex.py:159-231 | an ordered sum of channel codes; one forecast head per channel |
| G. Face and reward | `face_in` model.py:433, cortex.py:108; reward senses.py:100-124; `set_face` senses.py:193-195 | RewardSources in declared order; the face as a channel |
| H. Teacher and caregiver | caregiver.py:134-349; teacher.py:113-399 | SimTeacher inside the sim loop; the planner protocol carries over |
| I. The night's dreams | night.py:13-91, 181-211, 294-305, 453-499 | episodes of stored codes and own acts; REM samples discrete channels, takes the mean forecast for vectors |
| Also | striatal layout `k·(2V+3)` model.py:711-746 and mouth.py:36-43; awake full-window recompute cortex.py:153-156; sleep inside a tick under the serve's lock serve.py:114-116 | per-effector blocks appended; a cached awake stream for the sim; `world.pause()` |

**Already general (no change):** the store engine and FastStore, the band ladder, the value heads, ventral and fast critics, dopamine, working memory, the gate's three-factor lesson, feelings, the face organ, the night's skeleton and value replay, the reliability estimators, `stream` and `stream_step`, save/load with organ migration.

### 8.2 The interfaces (new files `body/core/anatomy.py` and `body/core/world.py`)

```python
class Channel:                  # one sense
    name: str
    kind: str                   # 'symbol' or 'vector'
    table: Tensor | None        # [K, d] fixed unit rows (symbol channels; the language ear is today's E)
    encoder: Module | None      # born and frozen (vector channels)
    rest_id: int | None
    end_id: int | None
    reserved: list[int]
    partner: bool               # the turn-taking source (at most one channel)
    def encode(self, obs) -> Tensor: ...        # [d], unit-scaled

class Effector:                 # one way of acting
    name: str
    table: Tensor               # [K, d]; per-joint alphabets sum per-joint rows; the voice shares the ear's table
    factors: list[int]          # e.g. [5, 5] for two joints; [K] for one softmax
    rest_id: int
    gate: Module                # the voice's is m.mouth_gate; its opt_gate and gate_buf keep their names
    def gate_inputs(self, frame, life) -> Tensor: ...   # its own "ear"
    def cost(self, act, frame) -> float: ...
    # plus its chunk and stop state

class RewardSource:
    name: str
    def felt(self, frame, life) -> float: ...

class Anatomy:
    channels: list[Channel]         # the order is the float order of the input sum
    effectors: list[Effector]       # effector 0 is the voice
    rewards: list[RewardSource]     # summed in this order

@dataclass
class Frame:
    tick: int
    obs: dict                       # channel name -> raw observation
    face: float                     # the teacher's face level (a scaffold at birth)
    truth: dict                     # for the teacher and instruments only; never enters the body

class World:
    def frame(self) -> Frame: ...
    def apply(self, acts: dict) -> None: ...     # effector name -> act
    def pause(self) -> None: ...                 # night
    def resume(self) -> None: ...
    def save_state(self) -> bytes: ...
    def load_state(self, blob: bytes) -> None: ...
```

- **`LanguageAnatomy(tok, cfg)`** rebuilds today's values exactly: `sil`, `eot`, `nl`, `end_id`, `space_id`, `reserved`, `bans`; the ear channel (today's `E`, partner); the face channel; the mouth; and the sources face → `world_r` (with `world_mask`) → cost (under `cost_in_reward`, which is 0 on the served body).
- **The signatures stay.** `Life(organs, tok, ...)`, `Life.birth(tok, ...)` and `Life.load(path, tok, ...)` keep their signatures, and a tokenizer builds `LanguageAnatomy` inside. The check, `serve.py` and the tools therefore run unchanged.
- **`DiaryWorld`** is today's queue and face; `serve.py` wraps it. Its `pause()` keeps today's behaviour exactly. From R9 it also writes the pace log (section 3, graft 5).
- **`SimWorld`** (`body/sim/world.py`) runs the MuJoCo scene.
- **The loop** becomes: `frame = world.frame(); acts = life.tick(frame); world.apply(acts)`.
  - For the language body the frame is still built inside the tick from the queue, so the draw order does not move.
  - For the sim, each effector's act enters at the next tick's frame, so the cortex runs one forward a tick (5.2).
- **`SimAnatomy`** declares 9 channels, 4 effectors and 3 reward sources. Its born codes come from a dedicated `torch.Generator` seeded from the body's seed and are saved with the body, so they never touch the global random stream.

### 8.3 Keeping the language body's digest

**The rule:** every refactor commit leaves every pinned language digest exactly equal. Nothing in the refactor is meant to change behaviour, so nothing is re-pinned, except when a language core fix is rebased in from main (item 5 below).

**Today's digests** (`tools/determinism_check.py`, ops/BASE_FLAGS.txt, the served save's constants). All were reproduced at 20:00 on 2026-09-23. "Threaded" is torch's default, which is 4 threads on this Mac and is what the served body runs with.

| profile | threaded | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` | notes |
|---|---|---|---|
| default | `7c54199e3c72cf77d45a8e94` | `fbe9e6b330f44e7424f113f3` | matches commit f61f12f; **the digest depends on the thread count** |
| served | `fc81c03dec606afb62c3fc35` | `f507fb9c6c55649ddd39f917` | constants `458595067029` |
| switches | `ab1ab46751b11b314dfc78c2` | `14de2b131ce4b2874ca0ad95` | constants `81de6471d00b` |
| served `--full --roundtrip` | `c4730b232091fe229ee286dc` | `823431e80a08f197ee086884` | round trip still differs in `store.S` and `store.sat` (known) |

**The procedure:**
1. **Where.** The refactor lives in a git worktree on the branch `core-refactor` (3.4 MB of tracked files), never in the main tree. The served body runs from the main tree, and a restart there would load half-refactored code.
   - The check runs from the main tree's directory, because its flags name `data/stories_valid.txt` relatively. It imports the body from the worktree.
   - For example: `cd /Users/lukehamond/Projects/project && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python3 <worktree>/tools/determinism_check.py --profile served --cfg tools/pins/served_cfg.pkl`.
2. **R1 edits the check, then pins.** The check gets two changes, and nothing else:
   - A `--cfg` option reads the served constants from a frozen pickle of the save's cfg (`tools/pins/served_cfg.pkl`, a few KB; constants `458595067029`). Today the served and switches profiles read them from the live save named in `ops/serve_command.txt`, so a change of serve flags would move the pins with no change of code.
   - The `--full` skip list gains the interface objects (`anatomy`, `effectors`, `world`). `--full` hashes every working attribute of the life, and these are declarations whose content `test_anatomy` checks instead. Without the skip, R2's first new attribute would change the digest (or stop the check, which refuses types it cannot hash).
   - R1 then confirms that every digest above is unchanged by the edit. It writes all eight (three profiles plus served `--full --roundtrip`, each threaded and thread-pinned) and the `--full` section digests to `tools/pins/digests.txt`.
3. **At every commit** of R2–R9, re-run all eight at `nice -n 19`: about 4 s each, under a minute in all. Equal: go on. Different: the `--full` section digests place the change; fix it or revert.
4. **New code is inert for language.** The language anatomy:
   - builds no new module;
   - draws no new random number;
   - runs no new branch;
   - adds no attribute to the life or its modules, no field to its window entries and no key to its save.
5. **The language body's core code is frozen on main during R1–R9** (`body/life.py`, `body/core/`, `body/model.py`).
   - Its teacher, scripts and flags may change, since the frozen constants keep the guard steady.
   - A core fix the language body cannot wait for lands on main first, measured on a copy as always. It is then rebased into the branch, with the pins re-taken at that commit. Each costs about half a day, and none is in the schedule.
6. **After S5,** a `sim` profile joins the check, so later core changes guard both bodies.
   - It runs a tiny body on the SimAnatomy, fed a recorded 200-tick script of raw observations (stored once in `tools/pins/`, about 4 MB). The guard therefore covers the core without depending on the GPU renderer.
   - SimWorld's own exact-replay test covers the physics.

**The hazards, and the rule for each:**

| hazard | rule |
|---|---|
| Organ build order: `Organs.__init__` draws from the global random stream in a fixed order (model.py:418-547: lexicon, `who_emb`, `perm`, `face_in`, …) | new modules only at the end of `__init__` (after model.py:549), and only when the anatomy declares them |
| `widen_gate` (life.py:229-230) draws when its Linear is built | leave it where it is for the voice |
| Draw order on `self.gen` (mouth.py:398, 514) | the voice draws first, exactly as now; other effectors draw after it |
| Float order of the input sum | ear + face + bundle + own, then new channels |
| Float order of the reward sum | clip the face, then add `world_r`, then subtract cost, one at a time |
| `stri_W` layout (model.py:714-716) | new effectors' blocks go after the language block |
| Names in the state_dict (`E`, `face_in`, `mouth_gate`, `actor`, `chooser`, `stri_*`) and the save's `life` keys (persistence.py:17-32) | keep every name, so that `Life.load` reads the served save (`data/watch2.pt`) unchanged |
| `--full` hashes every working attribute of the life (`vars(life)` less a named skip list), every plain attribute of every module, every window entry and every key of the save | the language life gains nothing in any of them; the interface objects join the skip list in R1, before the pins are taken |
| The check, `serve.py` and the tools call `Life(organs, tok)`, `Life.birth(tok, ...)` and `Life.load(path, tok, ...)` | keep the signatures; a tokenizer builds `LanguageAnatomy` inside |
| The served and switches profiles read their constants from the live save | frozen in R1 (`--cfg tools/pins/served_cfg.pkl`) |
| Least-squares float order in the critics | untouched by the refactor |
| Thread count | every before/after comparison uses the same thread settings, and both settings are pinned |

### 8.4 The refactor steps

| # | step | days | risk to the digest |
|---|---|---|---|
| R1 | The worktree; the check's two edits and the eight pins (section 8.3); a test that `LanguageAnatomy` equals the fields derived from the tokenizer | 0.5 | none |
| R2 | Anatomy replaces `tok` (life.py, senses, instruments, night, persistence) | 0.5 | none |
| R3 | RewardSources as an ordered list (senses.py:100-124) | 0.5 | float order |
| R4 | Frames: channel codes summed in order in `Organs.inputs`; per-channel window fields; all 11 `inputs()` call sites (cortex.py:155, 205; mouth.py:31; night.py:173, 209, 296, 313, 485, 523; instruments.py:41, 60); per-channel forecast heads with `latent_pred` as channel 0 | 2 | float order; build order |
| R5 | Effectors: the voice as effector 0 keeping its names; `_choose`, `_act` and `_gate_lesson` loop over effectors; per-joint readouts; striatal blocks appended; the defect fixes #4 (record the gate's own draw), #5 (actor trace per tick) and #8 (credit from the act on) as switches, off for language | 3.5 | draw order on `self.gen`; `stri_W` layout |
| R9 | The world loop: DiaryWorld (serve wraps it) with the pace log, the SimWorld interface, sleep calls `world.pause()`, the deadline switch | 1 | low |
| | **First sim tick possible (every learning rate at 0: plumbing, not a life): end of day 8** | | |
| R6 | `act_pred`, `act_inv`, learned stops (`chunk_gate` per effector) | 1 | none (built last, off for language) |
| R7 | Events and episodes: store writes gated by the surprise quantile; marks at event ends; offset and pace on the declared partner channel; defect fixes #1 (tired memories recover) and #6 (smiles reach the felt entry, so rewarded own episodes are tagged for replay) as switches | 1.5 | medium |
| R8 | The night over frames: per-channel batches from stored codes; own acts replayed as efference copies and `act_pred` targets; REM on frames (REM stays on) | 1.5 | medium |
| | **Refactor complete: about day 12 (range 9–14)** | | |

The defect numbers are those of ops/review_2026-09-22.md section 1. Optional before R4: delete the dead paths the seams pass through (review section 4), one per commit with the digests held. It shortens R4–R5 by about what it costs.

### 8.5 After the refactor: an awake cache for the sim

The awake cortex recomputes the whole 64-tick window every tick (cortex.py:153-156), and in the language tick about 1.5 times a tick (its own half). The cached `stream_step` (model.py:602-621) exists but only the night uses it. At 25.8M the full recompute fits (6.9 ms), so the cache is not needed at birth. It is the first speed-up if the core grows. The critics' 4097-wide least squares cost more than the cortex (6.2).

### 8.6 The language body during and after

- It keeps running its current code throughout the refactor, with its core code frozen on main (8.3).
- **The move** happens at a night boundary, after the full guard passes. The eight digests are run with the live served constants (the save's own, not the frozen pickle) on main's code and on the branch's code alike, threaded and thread-pinned. The check reads only the save's constants, so no copy of the 1.3 GB save is needed.
- **A second check before the move.** The branch loads the served save read-only and lives a scripted 400 ticks, and main's code does the same on the same script; the two state hashes must be equal.
  - `save_path` points into the scratchpad and nothing is saved: `Life.load` saves over the file it loaded by default, and a save would cost 2.6 GB with its `.tmp`.
  - It runs before the sim is born, while the disk has room.
- **The move is reversible.** A save written by the refactored code must load in main's old code (a round trip new → old), so the next boundary can go back.
- **The merge.** The branch merges into main at the move, and the pace log (section 3, graft 5) starts with it.
- Adopting any defect-fix switch is a separate measurement on a copy, applied at a boundary.
- After that it is body #1. If the owner stops its training, the guard still works (the check runs a tiny body, not the live one), and the sim gets back about 1.2 cores, 2.5 GB of RAM and the disk its nightly save reserves.

## 9. The build plan, day by day

### 9.1 Files to create

| path | what |
|---|---|
| `body/core/anatomy.py` | Channel, Effector, RewardSource, Anatomy, LanguageAnatomy |
| `body/core/world.py` | World, Frame, DiaryWorld |
| `body/sim/highchair.xml` | the scene |
| `body/sim/world.py` | SimWorld: physics, servo law, reflex, charge, grip, eye render and split, save/restore |
| `body/sim/senses.py` | born codes, log-mel ear, voice cache |
| `body/sim/anatomy.py` | SimAnatomy (9 channels, 4 effectors, 3 reward sources) |
| `body/sim/teacher.py`, `body/sim/teacher_consts.py` | SimTeacher and its constants |
| `body/sim/serve.py` | the `/sim` page on port 8030 |
| `body/sim/voice/` | cached `say` clips (about 1 MB) |
| `tools/sim_babble.py`, `tools/sim_rates.py`, `tools/sim_eye_check.py` | instruments: babble baseline, teacher rates, t11 on the final scene |
| `tools/pins/served_cfg.pkl`, `tools/pins/digests.txt`, `tools/pins/sim_script.npz` | the frozen served constants, the eight pinned digests, and the sim profile's recorded observations (about 4 MB) |
| `body/sim/guard.py` | the ops guard (reads `logs/diary_pace.jsonl`, which DiaryWorld writes) and the disk rule |
| `body/tests/test_anatomy.py`, `body/tests/test_sim_world.py` | LanguageAnatomy equality; exact replay; save and restore of the world, the teacher and every random stream |

### 9.2 The sim's build steps

| # | step | days |
|---|---|---|
| S1 | The world with all the grafts (grey table, objects 1.3×, magenta pad, backdrop, rim; per-joint servo law with gravity compensation; adhesion grip; reflex; charge and weakness); the soft weld and a scripted guide (tracking, servo torque at its cap, the weld never counted as contact); SimWorld save/restore and the replay check; the babble baseline with the per-joint alphabet (contact rate, pain rate at F_pain, escapes, tidies, pad visits) | 1.5 |
| S2 | Senses: the born codes; the log-mel ear; the voice cache (teacher at 3 rates, the child's words); contact clicks; the words channel; t11 re-run on the final scene (instrument) | 1 |
| S3 | SimTeacher and its constants file (the gaps and the judgments of section 5.7); the JSONL log with rotation; event rates with the babbler (at least 1 smile per 200 ticks, reached by changing what it asks and shows, never what counts as met; the babbler's smiles per ask written down as chance); the teacher's word rate for the store's capacity | 1.5 |
| S4 | The `/sim` page: overview camera, both eyes, the teacher's face, charge, the four gates, a two-voice transcript (t: push cube / c: cube / t: yes, cube!); instruments (gate rate per effector, stop lengths, store writes, pain, smiles, charge) | 1 |
| S5 | Wiring: SimAnatomy; a plumbing run with every learning rate at 0; a two-night timing with every learning rate at 0 (the full compute runs, nothing changes; never a second life); the mean tick, the night's length and the store's write rate written down; the ops guard and the disk rule beside the language body | 1 |
| S6 | **Birth (seed 1):** day 1, its first night, day 2, the save round trip; the first hour watched at 1× | 0.5–1 |

### 9.3 One session (serial): birth about day 19

| day | work | done when |
|---|---|---|
| 1 | R1 pin the guard; R2 Anatomy replaces `tok` | pins written; `test_anatomy` passes; digests equal |
| 2 | R3 reward list; R4 begins: Frame, channel list, ordered input sum | digests equal |
| 3 | R4: per-channel window fields; the 11 `inputs()` call sites | digests equal |
| 4 | R4 ends: per-channel forecast heads; R5 begins: Effector class, the voice as effector 0 | digests equal |
| 5 | R5: `_choose` and `_act` loop over effectors; per-joint readouts | digests equal |
| 6 | R5: per-effector gates, buffers, optimizers; `_gate_lesson` loop; striatal blocks appended | digests equal |
| 7 | R5 ends: switches #4, #5, #8 (off for language), each with a test | digests equal |
| 8 | R9: World, DiaryWorld under serve, sleep calls `world.pause()`, deadline switch; a stub SimWorld ticks with every learning rate at 0 | **first sim tick possible** |
| 9 | R6: `act_pred`, `act_inv`, `chunk_gate` per effector | digests equal |
| 10 | R7: surprise-gated store writes, marks at event ends, pace on the partner channel | digests equal |
| 11 | R7 ends: switches #1, #6; R8 begins: night batches per channel | digests equal |
| 12 | R8 ends: own acts replayed, REM on frames; the full guard including `--roundtrip` | **refactor complete** |
| 13 | The move (8.6): the live-constants digests, the 400-tick load check, the new → old round trip; the branch merged and the language body restarted on it at its night boundary. S1: the scene with the grafts; SimWorld save/restore | replay exact; the pace log running |
| 14 | S1 ends: babble baseline; S2 begins: born codes, ear | baseline numbers written |
| 15 | S2 ends: voice cache, words channel, eye check; S3 begins: SimTeacher | fovea identity at least 0.75 on the final scene |
| 16 | S3 ends: teacher acts, constants file, log, rates with the babbler | at least 1 smile per 200 ticks |
| 17 | S4: the `/sim` page and instruments | page shows every channel and gate |
| 18 | S5: wiring, plumbing run, two-night timing, budget and ops guard | the birth checklist (9.5) passes |
| 19 | **S6: birth (seed 1)** | a day, a night, a day, and a save round trip |

### 9.4 Two sessions (parallel): birth about day 14

The second session touches only `body/sim/`, `tools/sim_*` and the `/sim` page, and builds SimWorld against the interface in section 8.2. The first session runs the refactor exactly as in 9.3.

| day | session 1 (core) | session 2 (sim) |
|---|---|---|
| 1–2 | R1, R2, R3, R4 begins | S1: the scene, SimWorld, babble baseline |
| 3 | R4 | S2: senses, voice cache, eye check |
| 4–5 | R4 ends, R5 | S3: SimTeacher, rates with the babbler |
| 6 | R5 | S4: the `/sim` page |
| 7 | R5 ends | spare: disk and heat checks, ops guard |
| 8 | R9 | — |
| 9 | R6 | S5a: wiring and plumbing run with every learning rate at 0 (needs R4, R5, R9) |
| 10–12 | R7, R8 | re-run the plumbing run as each step lands |
| 13 | full guard; the move at the language body's night boundary (8.6) | S5b: two-night timing on the whole core, every learning rate at 0 |
| 14 | — | **S6: birth (seed 1)** |

Both sessions write tests at `nice -n 19`, small and short, and never touch the language body's ports.

### 9.5 The birth checklist

All must pass before seed 1 is born:
- the eight language digests equal the pinned ones (three profiles and served `--full --roundtrip`, threaded and thread-pinned);
- the language body has moved to the refactored code and its pace log is running (or the owner has stopped it);
- the sim's replay check is exact, and a save and reload continues it exactly (world, teacher and every random stream);
- the babble baseline and the teacher's rates are written down, with at least 1 smile per 200 ticks and the babbler's smiles per ask as chance;
- the plumbing run shows every channel arriving, every effector acting, rewards summed in order, a night running and a save round trip;
- the mean sim tick is at most 100 ms beside the language body (1.5× real time), and the language body's tick period, read from its pace log, stays within 5% of its median;
- the store's capacity holds 19 life days of the measured write rate;
- at least 5 GB of disk is free before birth: the sim's 1.9 GB peak, the language body's growth and its 1.6 GB nightly `.tmp`, and 1 GB of margin; the disk rule (6.4) is live;
- `act_inv` starts untrained, and no weight of the body has learned from the babbler.

After birth: live days with a report each day, judged by sitting with it. Defects are fixed on copies and applied at boundaries, never tuned.

## 10. Milestones a viewer can watch

Each milestone is judged by sitting with the body on `/sim` and by its own rulers, never by a baseline run.

| | what the viewer sees | its own rulers | when |
|---|---|---|---|
| **M0: it lives** | a day, a night and a day; the save round trip; the eyes, gates and face moving | its forecast of its own next body sense and next periphery code beats "nothing changes" | birth day |
| **M1: it looks and reaches** | it turns to its name and the teacher smiles; its gaze follows the showing hand; the mitten arrives at a shown object; it touches more gently | calls answered within 20 ticks; shown objects brought into the fovea; pain per contact | life days 1–5 (about the first 3–5 wall hours) |
| **M2: it does what it is asked** | after "push cube" it touches the cube first; it stops on the object; it goes to the pad on its own when its charge runs low | first object touched after an ask (chance 0.25); asks met without a demonstration, against the babbler's chance per ask; `act_pred` against `act_inv` on held-out demonstrations; recharges done alone | life days 5–20, about 4–18 wall hours (plausible, not promised) |
| **M3: it names and asks** | it says the name of what is in its fovea or hand; it says "ball" in a pause and the ball comes; it talks over the teacher less | the teacher's judgment of its names above 1 in 4; requests answered; talk-overs per exchange | life days 15–40, about 10–37 wall hours (open) |

Wall hours assume 40–55 minutes a life day (6.2). At the ops guard's 1× real time a life day takes about 70–80 minutes.

## 11. Out of scope for the first body

- Learned encoders and SIGReg (built, off); pretrained vision, unless the owner picks it.
- Hearing from audio alone: the word tokens run beside the audio at birth.
- A synthesised voice (the numpy vocal tract): the child speaks cached word clips.
- More than one teacher voice.
- Learning to imitate the parent robot by watching alone, with no guidance, is not assumed at birth: the parent's own arm makes it possible (the same body), and it is measured as a milestone, not built in.
- Wheels, legs, balance; a 3D arm, fingers, lifting (the next body).
- Deadline mode as the default.
- A learned reading of faces beyond the born expression reading (section 5.8).
- Opus in the tick loop (it listens between, section 5.7).
- Torch on the GPU (MPS). Only the renderer uses the GPU.
- Real sleep (sleep pressure, a safe posture): sleep is a pause of the world by tick count.
- Any transfer of the language weights.
- The full dead-code clean-up beyond the seams.

## 12. The owner's decisions (the world's shape, then two others)

Decided by the owner on 2026-09-23: the teacher is a parent robot with the same body (section 5.1); the world has sound (section 5.2: the objects' own sounds besides the voices and contacts). S1 grows by about 1 day (the parent's arm with physics and its inverse kinematics), S2 by about half a day (the object sounds). Later that evening (section 5.8): both robots have faces; the reward is felt only when the child looks at the parent's face; innate orienting and a born expression reading; a valence learner and a named motor timing part from birth; two ears with cochlea-shaped filterbanks; no live human teacher. About 3–4 more working days before birth.

1. **The body and the world.**
   - Option 1 (recommended): the high chair's 2-joint planar arm with a sticky mitten and a pan/tilt head. The fastest to learning.
   - Option 2: A's 4-joint 3D arm with a two-finger gripper. About a week more and slower to learn, but nearer the humanoid.
   - Either way: a pale grey table and objects 1.3× larger (identity 0.66 → 0.81, measured), a rim, a magenta pad, a backdrop, and night as a pause of the world.
2. **Vision:** born random codes only (recommended; measured adequate for M1–M2 with the fovea), or frozen pretrained features.
3. **The scaffolds at birth:** the word-token channel beside the audio, and the teacher's face as a number beside the visible mouth. Both are removed later; each removal is a change of body.
4. **The teacher's voice and words:** one voice (Samantha, 3 rates), about 20 words, and the child's name.
5. **Charge:** drain and refill set so that about 3 recharges fit a life day.

These are ours, not the owner's: the reward sources and their constants, the teacher's method, the sizes, the pain multiple and the tick.

Not the world's shape, but also the owner's; the plan works under the defaults:

6. **The language body.** Default: it keeps living, with its core code frozen during R1–R9 (about 12 working days). Stopping it frees about 1.2 cores, 2.5 GB of RAM and 1.6 GB of nightly disk headroom, and removes the need for the ops guard.
7. **The disk.** Default: nothing moves; the sim fits (6.4) with about 1.9 GB to spare at the worst moment. `data/backups` (12 GB) and `data/first_lineage` (23 GB) are the owner's files. Moving them off this Mac would remove the disk risk; the design never deletes them.

## 13. Risks, worst first

| # | risk | what to watch |
|---|---|---|
| 1 | The refactor changing the language digests (draw order, build order, float order, thread count, and any new attribute, window field or save key under `--full`) | all eight pins at every commit, run from a worktree |
| 2 | Credit reaching the start of a reach (3–8 ticks at these speeds). Carried by `act_pred` from demonstrations, the chunk actor, and the night replaying rewarded own episodes (needs defect #6 fixed) | `act_pred` against `act_inv` on held-out demonstrations, first |
| 3 | Vision is still only moderate (0.81 with an instrument's readout) | the teacher's naming against the body's naming; the fovea's forecast error on shown objects |
| 4 | Pain freezing the arm (learned helplessness) | the arm gate's open rate after painful days |
| 5 | The body relying on the word tokens and ignoring the audio | the ear head's forecast of the teacher's words |
| 6 | Disk on this Mac: 5.8 GB free; the sim's peak is 1.9 GB, and the language body's nightly save needs up to 1.6 GB beside it. A failed language save brings an extra night (review defect #16) | the disk rule (6.4); one sim save file; no films while it lives |
| 7 | Heat on this fanless MacBook Air (M4, 4 performance and 6 efficiency cores, 16 GB) under both bodies at once | the ops guard on the language body's pace and nights |
| 8 | The tick is about 3× the first estimate (t12): the critics' 4097-wide float64 least squares and the cortex's repeated window. The milestones move by wall hours, not by life days | the mean tick and its spikes; the speed-ups of 6.2, measured on compute only |
| 9 | The teacher's smiles drifting into shaping | smiles only for completed, visible acts; never a looser criterion to reach a rate; every constant in one file; every act logged |
| 10 | The charge need is weak: drive reduction nets zero over a full cycle, and the need lives only in the discount (5.6) | recharges done alone; the arm's weakness when empty |
| 11 | Sleep is a pause by tick count, not a robot's sleep | accepted for the first body |

## 14. Sources

- ops/review_2026-09-22.md: section 1 (the defect numbers used here), section 3 (the robot gap), section 4 (the refactor plan).
- ARCHITECTURE.md, BODY_SPEC.md, ITERATIONS.md, DIARY_BODY.md: the language body as it stands.
- Scratch scripts (not in the repository), in `/private/tmp/claude-501/-Users-lukehamond-Projects-project/81d92d50-4dd9-488b-8268-f1a474117bfc/scratchpad/sim_wf/`:
  - t1_arm.py, t3_legged.py: physics speed;
  - t2_room.py, t6_teacher.py, t6b_bump.py, t6c.py: B's room and its teacher;
  - t4_audio.py, t4b_rates.py: voice and ear;
  - t5_encoders.py, t5b_mps.py, t10_core_step.py, t10b_core_768.py: core cost;
  - t7_state.py, t8_depth.py: state, replay, depth;
  - t9_arm_table.py (A), t9_optionB.py, t9b_casters.py, t9c_limp_guide.py (B), t9_highchair.py (C), t9_probe*.py, t10_born_eye.py: the three designs and their eyes;
  - t11_fovea.py, t11b_record.py, t11c_diag.py, t11d_contrast.py: today's fovea and contrast check;
  - t12_life_tick.py, t12b_profile.py, t12c_sizes.py: the whole language tick at the sim's size under the served constants, its profile, the store read at capacity, the save's size and the RAM (the review).
- The digest guard re-run in the review: `tools/determinism_check.py`, every profile, threaded and thread-pinned.
