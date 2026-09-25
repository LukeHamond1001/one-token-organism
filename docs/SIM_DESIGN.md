# The first simulated body: the stock Unitree G1 on a play mat (design and plan, 2026-09-24)

Status: a design with its world built and measured. The stock G1, the parent and the living room exist as a MuJoCo scene in `body/sim/` (section 17). Nothing is committed. The core refactor continues on the branch `sim-core`. The language body is paused.

This amends the all-out design written this morning with the owner's nine decisions (the G1 amendment; section 14 lists each and where it lands). The custom child that design built is kept for reference as `body/sim/*_customchild.*` (section 15). The one-arm high chair of 2026-09-23 is kept as `body/sim/highchair_onearm.xml` and its maker; its document is in git history (commit e144afa).

Sources:
- the all-out studies (the world, the core at this scale, learning to move, the parent's language);
- four studies run today for the amendment, on this Mac: the G1 in the room, the parent's feelings and its limits with the G1, the vocal tract, the amygdala;
- the branch's commits.

Section 17 lists the files.

## 0. The answer in brief

**What we build.** The stock Unitree G1 humanoid, unchanged, born lying on its back on a play mat in a furnished living room. A human-shaped parent raises it, and the parent's face shows the parent's feelings.
- The child must learn everything it does: looking, rolling, reaching, grasping, sitting, and its first words, spoken through its own vocal tract.
- **It is the real robot's model:** 43 joints with the Dex3 three-finger hands, 34.4 kg, 1.32 m tall, and no neck.
- **Its senses sit where the real G1's are:** a stereo camera pair in the head, microphones, two inertial units (with the noise its own file declares, which the world adds), and each motor's own sensors. Nothing is added to its body. A software fovea moves inside each camera image.
  - **One sense goes beyond the real robot: touch.** The real G1 feels only in its Dex3 hands; here every link feels contact, and pain is read from it. That is a skin the robot lacks, and it is the owner's call (B18).
- **It has no balance, posture or walking controller.** Its innate mechanisms are these, and no others:
  - the servo law with tone (the muscle and stretch-reflex model), applied to the robot's own motors at load;
  - the tract's resting posture (a rest is silence, A26);
  - weakness when empty;
  - the movement units' persistence margin;
  - the software fovea's conjugate and vergence coupling;
  - the ears' brainstem lateral read;
  - the amygdala's born event lines (what it learns from them is learned);
  - four disclosed reflexes (withdrawal, grasp, orienting, the VOR with its quick phase) and the born expression reading (section 3.7).
- **The parent** moves by scripted inverse kinematics. It has a small scripted state of feeling shown on a graded face, and answers the child within a second. It follows the child's attention, speaks infant-directed sentences, scaffolds, and keeps routines. Claude steers what it teaches between wall minutes.
- **Reward** comes only from the parent's face, and only while the child looks at it, plus pain and a charge need.

**What already exists** (built and measured today; section 17 lists the files):
- **The G1 room.** `body/sim/make_g1room.py` includes the G1's file unchanged: it is byte-identical to the commit. The senses are added at load as cameras and sites only.
  - Under babble the physics runs at 10.9–12.2× real time.
  - The two eyes cost about 16 ms a tick without shadows and 38 ms with the sun's shadow, from render to fovea.
- **The parent's acts.** All 22 passed inverse kinematics inside human joint ranges around the custom child. Aimed at the G1, the parent kneels beside it, rests a hand on its trunk, shows a toy 40 cm before its eyes, and guides a forearm.
- **The parent's feelings and graded face** (`parent_feel.py`). Its self-test shows that nothing but a judged act can ever be felt.
- **The vocal tract** (`tract.py`). It babbles and hears itself. Eight of the nine target words can be said in a way the parent's ear accepts in context.
- **The amygdala** is specified and measured on a synthetic stream: 0.19 ms a tick.
- **Stills** are in `video/sim_look_g1/`.

**What the measurements changed:**
- **A person cannot move a 34 kg body the way a parent moves a baby.** A person's hands give about 160 N with both hands, or 200 N for up to 2 s.
  - Within that: holding the G1 once it is propped near upright; guiding a limb that yields; a brief turn onto its side.
  - Beyond it: sitting it up from lying (217–237 N), sliding it on the mat, lifting it (its weight is 337 N).
  - So the parent comes to the child, and the child's own acts must carry every rise (section 4.2).
- **Four of the ten toys cannot be held** by a Dex3 hand at any scale tried: the ball, duck, bear and drum. The cup is scaled to 0.8.
- **Sitting and crawling.** Sitting upright with the legs out tips backward; the G1 sits only leaning forward with its hands on its knees. On hands and knees, flat palms need 4× the wrists' 5 N·m; on fists it nearly holds.
- **The solver.** The prototype's settings crash MuJoCo 3.9 when a Dex3 hand grasps. The elliptic friction cone with multi-point collision detection off ran 360 grasp trials and all the babble clean.
- **Pain.** Under babble, single-step contact peaks (944–1,799 N) reach the G1's pain threshold (1,012 N). The 10 ms filter decides whether that is felt (C5).
- **Imitation starts near chance.** An inverse model learned from the child's own babble turns the parent's words into echoes the parent accepts for only 2 of 12 words. The parent's ear works only by listening for the words it expects, against a bank of the child's own babble.
- **The tick grows.** The G1's physics, its two eyes, the tract and the parent's ear add about 15–30 ms. The tick is now about 100–155 ms mean with the sun's shadow in the eyes.

**The core.** The refactored core on `sim-core` serves this body at the planned size: d 512, 6 blocks, about 35M parameters.
- R1–R6, R9 and R5b are committed, with R6 fixes 1, 2 and 3 (e48b284: `act_pred`'s plasticity gated by its labels' reliability) and the verifiers' three non-blocking findings (6d6d246). Every language digest held.
- About 9.5 working days of core work are left. The amygdala is part of R7 and R8 (section 8).

**When it is born:**
- About working day 13 (range 11–18) with three build sessions in parallel: the core; the world and the G1; the parent, its voice and ears, and the child's tract.
- About day 18 with two sessions, about day 28 with one.
- The disk has 20 GB free. One verification run filled it this morning, so copies must stay small (risk 11). Birth needs at least 8 GB free (section 9).

**What a viewer can watch first:**
- Life days 1–3 (about the first 1–5 wall hours): its fovea, then its trunk, turn to the parent's voice and face, and it keeps the smiling face in view.
- Life days 5–25: its first touches of a toy held over its chest.
- Life days 15–60: rolling toward the parent on purpose.
- A life day with its night takes about 55–90 wall minutes.
- Sitting alone, crawling, standing and its first word from its own tract are not forecast (section 12).

**What it costs.** This Mac only: no pod and no GPU training. Claude acts between wall minutes, never inside the tick. Nothing here loads the language body's save.

**Understanding.** The owner's goal is a human brain architecture that gains understanding. Every milestone carries a test by something never taught (section 12): a voice from a place the parent never called from, a toy never shown, known words in a combination never heard, a word used to get something.

**The honest risks** (section 13 says how we see each one):
1. **Credit across a delay.** The smile is felt about 7–25 ticks after a reach begins, partly outside the 12-tick eligibility window.
2. **Balance at a 150 ms tick, in a stiff body.** The G1 cannot sit upright, and under the resting servo law every posture sinks. A limp upper body would fall with a time constant of about 0.19 s, near the tick.
3. **Motor milestones by chance only.** Rolls under babble are not yet measured on the G1. The born body's movement units are also shorter than the babbler that justified them: its gates continue a unit on about a quarter of ticks, so units average about 1.4 ticks, not 4.5 (section 3.6).
4. **The parent's strength.** A person cannot sit up, slide or lift a 34 kg body, so the child's own acts must carry each rise. Sitting with help may come late.
5. **Thrashing, learned helplessness, and leaving the mat,** with no one able to carry it back.
6. **The critics' float64 least squares** take about half the tick.
7. **Solver stability** with the Dex3 hands.
8. **Phantom pain** from contact spikes near F_pain.
9. **Vision.** The fovea gives 1.5 px a degree. On its back the G1 sees mostly the ceiling and its own body; face down, only the mat; sitting or standing, only what is low and ahead of it (A22).
10. **Vocal imitation starts near chance,** and the word tokens are the easier road to reward.
11. **Disk and heat.**
12. **The tick budget** is over the old 125 ms target.

**The owner's decisions.** The nine decisions of the G1 amendment are folded in; section 14 lists where each one lands. The decision log's part B asks what the design still defaults, the four toys a Dex3 hand cannot hold first. Three are needed before birth: B18 (touch on every link, a skin the real G1 lacks), B3 (the eyes' render shortcuts, the sun's shadow among them) and B1 (the cup already shrunk to fit the hand).

### Words used here

| word | meaning |
|---|---|
| tick | one moment of the body's clock: 150 ms of simulated time (75 physics steps of 2 ms) |
| life day | 24,000 ticks (one simulated hour), followed by a night |
| the core | the shared machinery in `body/core/` and `body/model.py`: cortex, store, gates, critics, dopamine, the amygdala, feelings, night |
| digest | the hash `tools/determinism_check.py` prints after a tiny body lives a fixed script; equal digests mean an edit changed nothing the body does |
| born code | a fixed random encoder, made once from the body's seed and never trained |
| periphery | each eye's whole image, coarse (88° × 58°, 56 × 32 px) |
| software fovea | a sharp 32 × 32 px window (about 21°) inside each eye's native image, moved by the gaze effector; nothing on the robot moves |
| effector | a part the body acts with: the voice (the tract), the word output (the scaffold), the gaze, the waist, each arm, each hand, each leg |
| gate | an effector's learned "act now or not" switch; each effector has its own |
| movement unit | a chunk of ticks over which an effector holds one act, as infants' movements come in units |
| efference copy | the body's copy of its own last act, fed back into the cortex |
| the scaffold | the word-token channel beside the audio at birth (input), and the silent token output the parent reads; both removed later |
| the tract | the child's articulatory vocal tract: its voice |
| the amygdala | the core's valence tagger (section 7.4): it learns fast which cues predict good and bad, and tags each moment's weight |
| tag | the amygdala's mark on a moment: what the moment is expected to bring plus what it brought, in reward units |
| never-taught test | a milestone's test of understanding by something the parent never taught (section 12) |
| working day | one build session's day of work |
| worktree | a second checkout on its own branch; the refactor happens there |

## 1. The goal and the laws

**The goal** (the owner's words): "give it human brain architecture where it will actually gain understanding"; find the human brain's math structure in simulation, where a body can learn without breaking; the robot after. This sim is body #2 on the same core as the language body. The body is the real robot's own model, so what it learns is learned by a body that exists.

**What "done" means for this first body.** Seed 1 of the G1 is born on the refactored core and lives days and nights. It reaches milestone M1 and passes M1's never-taught test (section 12), while the language digests stay exactly as pinned.

| law | what it means here | where |
|---|---|---|
| Grounded reward only | The parent's face (felt only while the child looks at it), pain from contact physics, the charge need. Nothing rewards moving, getting closer, looking, hearing words, sounding like the parent, or novelty in itself. A met ask is the parent's judgment of an act it asked for, not a reward for looking (A2). | 6 |
| No hand-written rules, cheats or controllers in the body | No balance, posture, walking or gravity-compensation law. Guided moves are labelled by a learned inverse model. Stops are learned, and `chunk_max` is only a ceiling. The innate parts are few, named and disclosed. | 3.3, 3.6, 3.7 |
| The architecture biology uses | One gate per limb (parallel basal ganglia loops), movement units, forward and inverse models, a vestibulo-ocular reflex, cochlea-shaped filterbanks with a brainstem delay line, the amygdala as a named organ (a fast valence tagger), vocal learning through the body's own ears | 3, 4.9, 7 |
| Survives a change of body | Every mechanism is written against the Anatomy (channels, effectors, reward sources), never against these joints. The per-joint alphabet grows linearly with the joints. The robot's own model is the body. | 8 |
| Understanding, tested by what was never taught | Every milestone has a test the parent never taught toward, judged by the body's own rulers and by sitting with it, never by a baseline run | 12 |
| Disclosed constants | One table, saying who sets each | 10 |
| One seed per body | Seed 1 is the only life. Plumbing and timing runs have every learning rate at 0. The babbler is an instrument of the world, and no weight of the body learns from it. | 11 |
| Nothing fitted to the environment's pace | The world waits for the child (lockstep). The pain threshold comes from the body's declared mass. Store writes are gated by the body's own running quantile. The parent's timings are set before birth and never tuned to the child's rates. | 3, 4, 6 |
| The environment's shape is the owner's | Decided: the nine decisions of the G1 amendment, and before them the parent, the room, the ears and the reward's route. Everything else defaulted is flagged. | 14 |
| The teacher's method and the body's constants are ours | The parent's timings, feelings, priorities and force caps live in its constants files. The body's constants are in section 10. | 4, 10 |
| Measure on copies, change at boundaries | The refactor lives in a worktree. After birth, fixes are measured on copies and applied at night boundaries. | 8, 11 |

## 2. What changed, and what still holds

**From the all-out design of this morning** (the owner's nine decisions):

| part | the all-out design | now |
|---|---|---|
| the child | a custom child: 0.75 m, 9.45 kg, a three-joint neck, eyes that turn, a mitten hand, a screen face | the stock G1: 1.32 m, 34.4 kg, no neck, Dex3 hands, no face (3.1) |
| looking | turning eyes and a neck, with a VOR | a software fovea inside each camera image, then the waist and the whole body; the VOR acts on the window (3.4, 3.7) |
| the parent | five scripted faces; lifts, props, turns and slides the child; brings it back | a scripted state of feeling on a graded face; a behaviour system (contingency, joint attention, infant-directed speech, scaffolding, routines); a person's force caps; it comes to the child (4) |
| the voice | a synthesized child voice playing word tokens | an articulatory vocal tract it must learn to use; the token output becomes a silent scaffold (4.9) |
| the tagger | the "valence learner", a decision-log entry | the amygdala, a named organ of the core, in R7 and R8 (7.4) |
| understanding | the ledger's "understood" | a never-taught test at every milestone (12) |
| light | fixed daylight | day and night light across the life day (5.4) |
| the face reward | a rise felt as the new level | the increment rule, for a graded face (6) |
| the world | a 1.6 m mat; toys sized for a mitten | a 2.8 × 2.0 m mat; toys placed around the G1; the cup at 0.8; the elliptic friction cone (5) |

**From the one-arm design of 2026-09-23:**
- **The body:** a whole humanoid lying on a mat, instead of a planar two-joint arm at a table.
- **The parent:** a human-shaped figure moved kinematically, instead of a copy of the child's robot.
- **The world:** a living room with ten toys, instead of a rimmed table with four objects.
- **Gravity compensation is dropped.**
  - For a free-floating body it is computed as if the pelvis were bolted down, so it lifts resting limbs off the mat.
  - It is also a posture aid in disguise. With weightless links, babble rolled the custom child 5.4 times a minute instead of 1.1, and sat it upright on 18% of ticks instead of 0.1%.
- **Pain:** a threshold from the whole body's weight.
- **The eyes:** a periphery and a fovea from one render per eye. There is no depth channel; two eyes give disparity.
- **The striatum:** rows per joint (R5b); movement units; fatigue per effector.
- **The words:** 50 birth words plus 26 letters for everything after. The parent's sentences come from templates, with Claude steering. The voices come from AVSpeech.
- **The language body is paused.** The ops guard and the pace log matter again only if it runs beside the sim.

**Still holds** (from the design of 2026-09-23, the owner's decisions of that evening, and this morning's design):
- **Time:** the 150 ms tick, the 24,000-tick life day, lockstep, and deadline mode as a test switch only.
- **Reward:** its three sources; the face felt only while the child looks at it; the born expression reading; innate orienting.
- **Parts from birth:** the amygdala and the motor timing part.
- **Hearing:**
  - two ears with cochlea-shaped filterbanks, and each toy's own sound;
  - the word tokens as a scaffold removed later, with letters as the fallback;
  - no live human teacher; Claude steers between minutes.
- **Movement and night:** the servo law re-anchored on each act; the withdrawal reflex; weakness when empty; night as a pause of the world.
- **The refactor:** its interfaces (Anatomy, Channel, Effector, RewardSource, World, Frame) and the eight-digest guard.
- **Memory:** the store and the episodes; the night drawing episodes by tag; REM stays on.
- **The teacher's rules:**
  - smiles only for completed, visible acts;
  - never a looser criterion to reach a rate;
  - demonstration by moving the child's own limb;
  - the babbler's smiles per ask written down as chance.

## 3. The child: the stock Unitree G1

### 3.1 The body

- **The model:** MuJoCo Menagerie's `unitree_g1/g1_with_hands.xml` (BSD-3-Clause, the licence kept), committed in dd8640e "on the owner's yes". The room includes it unchanged; its file is byte-identical to the commit (shasum checked).
- **Size:** 1.32 m standing and 34.39 kg. The upper body (waist and above) is 16.2 kg, one arm 4.05 kg, one leg 7.19 kg.
- **Shape:**
  - pelvis;
  - legs: hip pitch, roll and yaw; knee; ankle pitch and roll;
  - waist: yaw, roll and pitch, to the torso;
  - **the head is part of the torso: there is no neck**;
  - arms: shoulder pitch, roll and yaw; elbow; wrist roll, pitch and yaw;
  - Dex3 hands: a thumb of 3 joints, an index finger and a middle finger of 2 each.
- **Degrees of freedom:** 43 hinges and a free root, 49 in all, driven by 43 position actuators.
- **Collision:** 51 collision shapes on 42 of its 44 links. Self-collision is on, as shipped. Its visual meshes have about 629,000 triangles.
- **At birth** it lies on its back on the mat, head toward −x: arms a little out with the elbows a little bent, hips and knees a little flexed and turned out, hands open. It settles for 1.5 s under its servos.
- **No face on it** (the owner's decision 7). The parent has the face; the parent reads the child only from what it does, with two disclosed exceptions: where its software fovea points (B14), and its charge (the battery gauge a carer would see).

**No neck: what it means.**
- It cannot turn its head without turning its torso. It looks first with the software fovea inside the camera images (±38° × ±20°). Beyond that it looks by turning the waist (yaw ±150°, roll and pitch ±30°) and the whole body.
- **Lying on its back,** its line of sight is 49.5° from vertical, toward its feet, with the eyes 0.145 m above the floor.
  - It sees the ceiling, the far wall and its own body.
  - It sees the parent only when she leans over its chest. In the showing pose, her face sits at yaw −24° and pitch +11° in the left eye.
- **Looking sideways while lying means rolling the upper trunk against the mat.** Orienting and rolling share the same muscles. That may bring head-led rolls sooner, or tangle looking with falling; W4 measures it.
- **The head's inertial unit is the torso's.** The vestibular sense reads the trunk's orientation directly; the custom child had to combine its head sense with the neck's angles.
- **Face down,** it cannot lift its head alone. It must lift its chest on its elbows: prone on its elbows holds at 12% of the elbow's limit, with the eyes 0.25 m up. Even then its eyes point 62–72° below the horizontal, back under its own chest: face down it sees only the mat (A22).
- **Sitting or standing,** its eyes point 48° or more below the horizontal (the real G1's camera is aimed at the ground for walking). It never sees a standing adult's face; a face is seen only low and ahead of it (A22).

### 3.2 Joints and strength (the model's own)

Ranges are in degrees and torque limits in N·m, for the left side. The right side mirrors roll, yaw and the fingers' signs.

| joint | range | torque limit |
|---|---|---|
| hip pitch / roll / yaw | −145..165 / −30..170 / ±158 | 88 / 139 / 88 |
| knee; ankle pitch / roll | −5..165; −50..30 / ±15 | 139; 50 / 50 |
| waist yaw / roll / pitch | ±150 / ±30 / ±30 | 88 / 50 / 50 |
| shoulder pitch / roll / yaw | −177..153 / −91..129 / ±150 | 25 each |
| elbow; wrist roll / pitch / yaw | −60..120; ±113 / ±92.5 / ±92.5 | 25; 25 / 5 / 5 |
| Dex3 thumb joints 0 / 1 / 2 | ±60 / −41.5..60 / 0..100 | 2.45 / 1.4 / 1.4 |
| Dex3 index and middle (2 joints each) | −90..0, −100..0 | 1.4 |

- **These are the real motors' limits** as Unitree publishes them, through Menagerie, with no scaling. The old risk "torque limits from memory" is gone.
- The legs and waist are strong. The wrists' pitch and yaw (5 N·m) and the fingers (1.4 N·m) are weak.
- Each joint has friction loss 0.3 N·m and armature 0.01, as shipped.

### 3.3 The servo law and tone

- **The file's servos are a placeholder.** Its actuators are position servos at kp 500 with critical damping, Menagerie's value; its README says the gains need tuning. They are not the real robot's gains.
- **The gains are the body's low-level law, and ours.** The real G1 takes a target, kp and kd for every motor on every command. So the servo gains are set at load (as the scene light is switched off at load), never in the file.
- **The law** (the same as the custom child's):
  - each servo reaches its torque limit at 0.25 rad of error, so kp = limit ÷ 0.25. That gives 352–556 N·m per rad for the hips and knees, 200–352 for the waist, 200 for the ankles, 100 for the shoulders, elbows and wrist roll, and 20 for the wrists' pitch and yaw;
  - the Dex3 joints reach their limits at 0.1 rad: 14–24.5 N·m per rad;
  - damping is 0.04 s × kp.
- **An act re-anchors the target:** target = the measured angle + the step, so a move starts where the limb is.
- **At rest the target relaxes toward the measured angle,** with a time constant of 3 ticks (ours, anatomy: muscle tone).
  - A posture held by acting holds; a posture left to rest sinks.
  - The postures in section 3.8 were measured with the stock servos locked on fixed targets. Under the resting law every one of them sinks: holding a posture is learned, by acting every tick, as an infant learns to hold its head. W1 measures the sink rates.
- **No gravity compensation and no balance law.**
- **Weakness when empty:** the torque limits × (0.3 + 0.7h), where h is the charge.
- **Measured again before birth (W4)** under this law: rolls under babble, the leaning sit's fall time, guided tracking, and pressing into obstacles.

### 3.4 Senses

Each channel's code is born fixed from the body's seed, unit-scaled and projected to d = 512. Each channel has its own forecast head.

World truth (object poses, labels, which source made a sound, the torso's orientation in the room) goes only to the parent and the instruments, never into the body. There are four disclosed exceptions. The first three are the reward's carriers or the body's own state:
- **The face channel.** Its level is the born expression reading of the parent's graded face (A1). It passes only while the face test holds; the test is a geometric ray test driven by where the child's own software fovea points. The amygdala's and the critics' event line "a face in the fovea" is **not** this test: it is the born three-blob face template (the one orienting uses) run on the fovea's own pixels, so the body's value and salience never read world truth beyond the reward's carrier. W3 measures the template's hits and false alarms in the fovea (C3).
- **Pain.** It is the contact force on a link.
- **Charge.** It is h.

The fourth is a label, and it is temporary:
- **The word scaffold (channel 0).** It is the parent's own label for the word it said, given as a token or as letters beside the sound, and only while the parent is audible. The ears still hear the same word. The scaffold is removed on a copy once the child's own hearing of words passes the tests in section 4.9.

Everything else the body gets comes from its sensors. Orienting's triggers are among these: the face template works on the periphery's pixels, and a sound's onset and side come from the cochlea and the born lateral read. They are never taken from the world's list of events.

**Where the G1's senses are.** They are added at load as cameras and sites: no shape, mass, joint or actuator.

| sense | on the real G1 | here |
|---|---|---|
| eyes | the head's RealSense D435: two imagers 50 mm apart, plus a colour camera | two colour pinhole cameras at the D435's imagers. The pose is Unitree's URDF `d435_joint`: 0.0576 / 0.0175 / 0.4299 m in the torso's frame, pitched 47.6° down. The right eye is 50 mm to the right, and each pinhole sits 3 mm in front of the head shell. The field is 88.3° × 58° (horizontal × vertical), at 168 × 96 px per eye |
| ears | a four-microphone array (its positions are not in the model) | two sites on the head's sides, 15.8 cm apart (assumed, B5) |
| balance | inertial units in the torso and the pelvis | the model's own `imu_in_torso` (it moves with the head) and `imu_in_pelvis`, each with a gyro and an accelerometer. The file declares their noise (gyro 5e-4, accelerometer 1e-2), but MuJoCo 3.9 does not apply it (checked: a still G1 reads exactly 0), so the world adds it from its own seeded stream |
| joint sense | each motor reports its angle, velocity and torque | the 43 joints' angles, velocities and efforts |
| touch | tactile arrays on the Dex3-1 hand; no skin elsewhere | the contact force summed per link (42 links), from the contacts. On the 14 Dex3 links this is the real hand's sense; on the other 28 links it is a sim skin the robot lacks (B18) |
| its voice | a loudspeaker (its place to check, B5) | the tract's sound from the head's front (4.9) |

- **Colour stereo is a sim choice** (B4). The real D435's two imagers are monochrome infrared, with a separate colour camera beside them.
- **There is no depth channel.** Two eyes give disparity; the fusion is learned (the owner's decision 4).

**The eyes and the software fovea.**
- Each eye is rendered once a tick, both into one buffer, with one read-back.
- **Periphery:** the whole field averaged 3 × 3, giving 56 × 32 px (0.64 px a degree).
- **Fovea:** a 32 × 32 window of the native image, about 21° wide at 1.5 px a degree.
  - Its place is a gaze state (yaw, pitch, vergence) that the gaze effector moves.
  - It reaches ±38° × ±20° from each camera's axis.
  - Vergence moves the two windows apart or together, so both eyes can fixate a near thing.
- **The window's place enters the body sense,** as the custom child's eye angles did, so the body knows where it looks.
- **Why in software.** The real G1 can run the same window on its own camera images, so nothing is added to the robot (the owner's decision 9 recommends it).

| # | channel | numbers a tick | what it is |
|---|---|---|---|
| 0 | words (the scaffold) | one symbol from 79 | The parent's word token, arriving on the tick its sound ends; letters for words after the first 50 (section 4.9). Its forecast is today's `latent_pred`. |
| 1 | face | 2 | The born expression reading of the parent's graded face: 2 × (smile − frown), its level and change. It updates only while the face passes the face test (A1). Out of view it holds its last value for 30 ticks, then reads neutral (A2). |
| 2 | ears | 1,557 | Two cochleas (15 frames × 40 bands each, per tick) plus the brainstem's delay lines (21 low bands × 17 lags). The spatializer's head radius is the G1's, 0.078 m. |
| 3 | eye_p (periphery) | 2 × 168 | Per eye: 56 × 32 px, coded retinotopically as 7 × 4 cells of 8 px (about 12.5° × 14.5°, near the custom child's 12.5°) × 6 fixed colour mixes. |
| 4 | eye_f (fovea) | 2 × 384 | Per eye: the 32 × 32 window, coded as 8 × 8 cells × 6 fixed colour mixes. |
| 5 | body (joint sense) | 199 | Per joint (43): sin and cos of the scaled angle, velocity, and servo effort (torque ÷ limit). The gaze state and its velocity (6). The tract's 10 positions and velocities, and breath left (21). |
| 6 | touch | 42 × 2 = 84 | Per link: log force and onset. |
| 7 | vestibular | 24 | Both inertial units: the accelerometer (which way is down) and the gyro (how the trunk turns), each as the tick's mean and peak. |
| 8 | charge | 2 | h and Δh (the body's own need). |

- About 2,970 numbers a tick, plus one symbol.
- **Keeping raw codes costs nothing measurable.** Storing each channel's raw code in the window and re-encoding every tick measured 58.8 against 58.0 ms (paired), so the window keeps the raw codes, which are smaller.

### 3.5 Effectors

Each joint's act is one of five settings {−big, −small, 0, +small, +big} per tick. An act's embedding is the sum of fixed per-joint rows, and its readout is one softmax per joint.

| # | effector | joints | step sizes per tick |
|---|---|---|---|
| 0 | voice: the vocal tract | 10 articulators (section 4.9) | −0.6, −0.2, 0, +0.2, +0.6 of each articulator's range |
| 1 | words: the scaffold's silent output | the born table: 50 words, 26 letters, a space, rest and end (79 rows) | — |
| 2 | gaze: the software fovea | yaw and pitch (both windows together), vergence | ±4° / ±11.5°; vergence ±1.7° / ±5.7° |
| 3 | waist | yaw, roll, pitch | ±0.09 / ±0.27 rad |
| 4, 5 | arm L / R | shoulder 3, elbow, wrist 3 | ±0.09 / ±0.27 rad |
| 6, 7 | hand L / R | the Dex3's 7 joints | ±0.09 / ±0.27 rad |
| 8, 9 | leg L / R | hip 3, knee, ankle 2 | ±0.09 / ±0.27 rad |

- **56 joint readouts** of 5 settings each, plus the 79-row table. The striatum's rows per joint (R5b) come to about 18 MB (8 positions × 56 joints × 5 settings × 2,048 units).
- **Why one effector per limb:**
  - not one per joint, because 40 separate gates would be 40 bandits with nothing coordinating them;
  - not one for the whole body, because one gate cannot move one arm alone.
  - The basal ganglia run parallel loops, one territory per body part. Reaching and grasping are separate effectors, as in the two visuomotor channels.
- **The gaze moves by conjugate and vergence commands** (Hering's law of equal innervation), so 3 commands move the two windows.
- **The gaze has no joints.** Its "joints" are the window's state, and its consequence sense is that state in the body channel, as any limb's is its joints.
- **Each gate is born like the voice's:** `Linear(d + 5 + n_in)`, starting at `birth_act` 0.25.
  - Each motor gate has 4 inputs of its own: its own act last tick, touch onset on the limb, pain on the limb, and the limb's forward-model error.
  - The tract's gate takes its own act last tick, the forward-model error on its hearing, and breath left.
  - There is one optimizer (`opt_motor`), and the same three-factor lesson every 24 ticks as the voice's.
- **Fatigue is per effector,** and each gate reads its own. One shared fatigue would add up nine limbs' costs and silence the voice.
- **Switches at birth:** fixes #4, #5 and #8 on (the gate's own draw recorded, the actor trace decaying per tick, credit from the act on), and `chunk_gate` 1 for every effector. Fix #1 and the amygdala join them when R7 builds them (section 10). The language body keeps them all off.

### 3.6 Movement units, stops and the motor timing part

- **Chunks and learned stops** (built in R6). While a chunk is under way, the gate's own draw decides whether it goes on, and the act is `act_pred`'s best guess. The chunk ends when:
  - the guess is the effector's rest;
  - the gate says no;
  - a reflex fires;
  - or `chunk_max` 8 is reached (a ceiling, never the usual stop).
- **Movement units** (new, R6h). A motor chunk holds its first act unless the proposal prefers another by more than a born persistence margin (log 4 in logits, disclosed). The evidence, from the custom child's babbler (on its back, 3 × 4,000 ticks):
  - acts drawn fresh every tick: 0 rolls at any strength;
  - each joint holding its step for 1–8 ticks: 1.10 rolls a minute at full strength, 0.43 at 0.3×.
  - At birth `act_pred` knows nothing, so without persistence every chunk would be a fresh random draw each tick, and the body would never reach a whole-body event such as a roll.
  - The biological counterparts are infant movement units (von Hofsten) and the smooth, seconds-long spontaneous general movements of newborns (Prechtl), which are never white noise.
  - No roll count set the margin, and none will ever tune it. W4 writes down the rolls it gives on the G1, as chance.
  - **The evidence does not yet transfer to the born body** (found in the review of the amendment). The babbler held each joint for its own 1–8 ticks (4.5 on average). The born body's units are per effector, and R6's code continues a unit only while the gate's own draw says go on, at the born p_act of about 0.29 (the floor 0.05 plus 0.95 × `birth_act` 0.25). So born units average about 1.4 ticks, and about 70% last one tick: close to the fresh draws that gave 0 rolls. The nearer babbler conditions gave 0.6 rolls a minute (whole-body units, full strength), 0.03 (whole-body, 0.3×) and 0.17 (60% of units at rest). The margin only acts while a unit goes on.
  - So W4 runs **the born body's own motor loop** (its gates at birth, the continuation draw, the margin, `chunk_max`, every learning rate 0), not the per-joint instrument, and writes down the unit lengths and the rolls it gives. If the born units are far shorter than newborns' movements, the unit's born length is decided before birth on that biology (C38), never on a roll count.
- **The motor timing part** (built in R6, one per later effector):
  - `act_pred`: the cortex's stream → its own next act, read joint by joint.
  - the forward half: the stream after the tick's own step → the effector's consequence sense at the next tick (the limbs' joints; the gaze's state; the voice's ears).
  - the correction: the forward error → a term added to the proposal.
  - `act_inv`: (the sense at t, the sense at t+1) → the per-joint act, with a running reliability (Cohen's kappa per joint, R6 fix 2).
  - `act_pred`'s target is the efference copy when the effector acted. When it rested and something moved it (the parent's hand, a bump), the target is `act_inv`'s label, weighted by that reliability. So a demonstration counts only as far as the inverse model has earned.
  - `act_pred`'s plasticity is gated by its labels' reliability (R6 fix 3, e48b284).
- **New for the humanoid (R6h):**
  - each limb's forward error feeds its gate. It tells a movement the child made from one done to it, and it sets how much a guided movement counts.
  - `act_inv`'s lessons are batched every 8 ticks; unbatched they cost 4–6 ms a tick.
  - an effector declares its consequence sense (the voice declares the ears), so the same part serves the limbs, the gaze and the voice.

### 3.7 Reflexes: kept and refused

Each kept reflex has a biological basis and is disclosed. Withdrawal and grasp act below the gate. Their ticks are logged as reflex and carry no gate eligibility; the cortex sees them through touch and joint sense. Orienting is only a bias that learning can outweigh, so it fades by learning, not by calendar.

| reflex | status | how it works, and why |
|---|---|---|
| withdrawal | kept | Spinal and lifelong: the flexor withdrawal (Sherrington). When a limb's link feels more than F_pain, that limb flexes away: one big flexion step of its joints a tick, for 2 ticks, whatever its last move was. The waist and torso have none. W1 fixes each G1 limb's flexion direction per joint from the model's axes. |
| palmar grasp | kept | Spinal, present at birth. A touch on the hand's base link (the palm) above 0.3 N closes all seven Dex3 joints one step a tick, unless the hand's own act that tick opens it (the cortex can override). It makes the parent's hand-over a real hold from day one. It fades only as the cortex learns to override it. Lying face down, it closes a palm pressed on the mat, which may hinder crawling; W4 counts how often. |
| orienting to faces and voices | kept | Newborns prefer faces (Goren 1975; Johnson and Morton 1991) and turn toward sounds (Muir and Field 1979). It is a born bias on the gaze's and the waist yaw's proposals toward a face-like blob in the periphery (a fixed three-blob template) and toward the side the ears' born lateral read gives. There is also a born gate input, "a face or a sound onset appeared", computed from the pixels and the cochlea. It is a bias, never a forced move; its gain is set by the amygdala (section 7.4) and is exactly 1 at birth. Without it, a newborn that never looks is never rewarded. |
| VOR | kept | Brainstem, present at birth. The software fovea's window counter-shifts by the torso gyro's rotation about each camera's own image axes (gain 1; the cameras are pitched 47.6° from the torso, so the gyro is rotated into each camera's frame first), so it stays on its target while the trunk turns. At the window's reach its quick phase jumps it back by half the reach, in the direction of the turn (A23). The gaze's own acts add on top. |
| born expression reading | kept | A fixed read of the parent's mouth corners while the face is in the fovea (A1). |
| screen face | removed | The G1 has no face (the owner's decision 7). |
| stepping | refused | A walking pattern generator, so a controller. Thelen showed it is the same pattern as kicking on the back, which babble already gives. |
| righting, parachute, equilibrium reactions | refused | Balance controllers. Infants mature these partly innately; the laws make ours learned, so our child's task is honestly harder than an infant's. |
| asymmetric tonic neck reflex | refused | A hand-written head-to-arm coupling, and the G1 has no neck. Measured on the custom child, it cut rolls from 1.10 to 0.77 a minute. |
| symmetric tonic neck, tonic labyrinthine | refused | Posture-tone rules |
| Moro (startle) | refused | No use here; a fall is already a large forecast error. |
| rooting, sucking | refused | Charging is not by mouth (section 5.3). |
| Galant, Babinski, placing | refused | No function in this body |
| a born cry | refused | The tract can learn to call. The parent reads distress from pain, long spells face down, thumps and the charge light (A13). |

### 3.8 What the body can do at birth (measured on the G1, no learning)

**Reach:**

| posture | toys in reach | notes |
|---|---|---|
| lying on its back, arms only | 3 of 10: block, cup, rattle | The hand reaches up to 0.83 m from the pelvis on the mat. The share of directions reached at 0.3 / 0.45 / 0.6 / 0.75 / 0.9 m is 33 / 38 / 42 / 29 / 0%. The waist barely helps: only 1–2% of waist poses clear the mat. |
| sitting, leaning forward | 1 of 10: rattle | The block is 2 cm short and the cup 4 cm short, even with the waist. It reaches 0.50 m with the arms and 0.66 m with the waist. |

- **Under physics,** the stock servos brought the hand to the planned pose within 0.8–3.8 cm in 1.8 s, with the shoulder at its 25 N·m limit. The swinging hand knocked the block 20 cm and the rattle 10 cm away.

**The torque to hold each posture** (the stock servos locked on the pose; the largest share of the actuator limit over the legs, waist, shoulders and elbows):

| posture | largest share | holds? |
|---|---|---|
| lying on its back | 36% (a shoulder, holding the arms at the pose) | yes |
| prone on its elbows | 12% (an elbow) | yes; the eyes 0.25 m up |
| sitting upright, legs out | — | no: it tips backward even with the limits × 10; with the real limits it catches itself on its hands |
| sitting leaning forward (hips 97°, knees 23°) | 41% a shoulder, 28% the waist | yes, with the eyes 0.61 m up; the wrists are at their limit where the hands press on the knees |
| hands and knees, flat palms | the wrists' pitch needs 4.0× its 5 N·m | no: the wrists sag 24° |
| hands and knees, on fists | 22% (the waist); the thumbs at 1.55× their limit | nearly: the thumbs sag 9° |
| standing | 11% | yes |

- On hands and knees the G1's feet cannot point backward (the ankle stops at 30°), so its toes carry the lower body, about 65 N each.
- These hold only because the stock servos lock the pose. Under the resting law every posture sinks (section 3.3).

**Babble** (the stock servos, 2 ms steps, 75 a tick; the machine's load average was about 3.7 from other sessions):

| case | steps a second | real-time factor |
|---|---|---|
| at rest | 8,460 | 16.9× (8.9 ms a tick) |
| babbling (two seeds) | 5,430–6,120 | 10.9–12.2× (12.3–13.8 ms a tick) |
| babbling with the parent kneeling down beside it | 5,670 | 11.3× for the physics; 8.5× with the parent's Python (4.3 ms a tick) |
| self-collision off (a test only) | 7,450 | 14.9× |

- There were no MuJoCo warnings in any of these runs.
- Under babble Σ τ² ÷ Σ τ²_max averages 0.063 (the custom child's was 0.043).
- The largest contact at the tick ends was 944 N and 1,799 N under babble. F_pain is about 1,012 N, so babble alone reaches pain (C5).
- With the kinematic parent kneeling, a contact reached 7,205 N: the G1's limbs hitting an immovable parent. The parent's yield rule (A4) is essential.
- **Not yet measured on the G1** (W4): rolls under babble, travel, time off the mat, and the leaning sit's fall time. The custom child's numbers are kept in section 15 for reference.

**Grasp** (360 trials under the final solver: a hand-over into the palm, 6 tries, and a top grasp from a surface, 3 tries, per toy per scale, at scales 1.0, 0.8, 0.7 and 0.6; none unstable):
- **Held at full size:** the block 9 of 9; the ring 6 of 6 hand-overs; the rattle 4 of 6 hand-overs; the car and the stacker 2 of 6 hand-overs and 3 of 3 top grasps.
- **The cup:** 0 of 9 at full size and 3 of 3 top grasps at 0.8, so it is scaled to 0.8 (6.7 cm across).
- **The ball, duck, bear and drum:** at most 1 in 9 at any scale from 0.6 to 1.0. They stay at full size and are flagged (B1).
- The Dex3's two fingers are 5.7 cm apart. Small objects slip between them, and round or large ones cannot be wrapped.

## 4. The parent

### 4.1 Its body and how it moves

- **A complete human figure at human proportions:** 16 segments (pelvis, spine, chest, neck, head; upper arms, forearms, hands; thighs, shins, feet).
- **An expressive face,** graded from its feelings (section 4.3): brows; lids; eyes with irises and pupils that look; cheeks; an upper and a lower lip; a mouth that smiles, frowns, rounds and opens with its voice.
- **Hands with shapes:** open, point, grip, curl.
- **A body, her trunk carried** (the lead's decisions of 2026-09-25: she is a body; and, after the first physical build fell onto the child and lay on it for minutes, her trunk is carried: A25b). Her 16 segments are dynamic MuJoCo bodies in one tree (`make_g1room.py`: a free pelvis; ball joints at the lumbar, the thorax, the neck, the shoulders, wrists, hips and ankles; hinges at the elbows, the forearms' pronation and the knees), at de Leva's (1996) female segment masses and inertias for her 62 kg (Harbo, Brincks and Andersen 2012's women's median). Her planner still makes the pose she means at each tick's end, by scripted forward kinematics, two-bone inverse kinematics for the arms and legs, and look-at for the head and eyes, all inside human joint ranges; the physics carries it out (`parent_body.py`):
  - **Her trunk is carried.** A person does not fall over, and balance is the environment's business, never the child's. Her pelvis is pulled toward the pelvis her plan means by a spring and a damper, and her weight is carried at her whole body's centre of mass; the support's force is at most 695.5 N and never points down, its torque at most 257.4 N·m (A25b). So she cannot be toppled by the child and never presses down on anything with more than her own weight; pushed along the floor past its cap she gives way, carried. Her spine and neck are stiff and limited (each spends its strength 5° off her plan), so her head follows her trunk.
  - **Her limbs are dynamic at a woman's strength.** Each axis of each joint has one MuJoCo actuator whose control and force ranges are her strength there, per direction (Harbo et al. 2012; Garces et al. 2002; Nordin et al. 1987: the table in A25b); its damping is inside that range, and nothing else of hers acts on her joints (no joint damping, no applied torque), so no torque of hers at a joint ever exceeds a woman's. Her command is her plan's spring (her strength over 20°), the damping's target velocity, her limbs' own weight (MuJoCo's bias force), her tone on an arm she holds still, and her holds' effort; resting, her legs lie where her carried pelvis puts them at a third of that stiffness.
  - **She never plans a pose into the floor, the furniture or the child.** Each planned pose is raised until no collision shape of her pelvis, legs or feet is under the floor or the mat's top (her floor lift: her feet, knees and shins then meet the floor by contact, the deepest just touching); every kneeling frame keeps her legs, trunk and head 3 cm from the child (A4), and each tick her planned legs, trunk and head are measured against the child where it is now and, nearer than 3 cm, she straightens and moves her base back off it (her standoff, A25b); her hands' targets on the child are its surface where it is now and her planned clearance (3 mm), re-planned as it moves.
  - **Her contacts are the physics'.** Her shapes are soft at contact priority 2 (MuJoCo's default time constant, 0.02; her hands' palm, fingers and thumb 0.006), so hers is every contact's with the G1 and the G1 is untouched (A21's mechanism). A4's care stays on her plan: a chain pressed past its cap for 2 steps stops where it is and backs off.
  - It holds toys through welds that start switched off. Its hand's collision proxy is off the toys while a weld holds (both collision bits; the prototype's first switch cleared only one).
  - **It holds the G1 only through capped springs** (section 4.2): a force at the held point, computed each physics step from her hand's pose and clipped at the act's cap, applied as an outside force on that link, and its reaction on her hand at her grip, her arm exerting it within her strength. A weld on the G1 refuses the world.
- **Its motions,** all passing the check with no failing frame:
  - a 4 m walk;
  - kneeling down (two step variants) and standing up, eased in and out at a person's pace: her pelvis at most 0.5 m/s, every other segment at most 0.8 m/s (A25b: carried, her body starts and stops only as fast as its support lets it);
  - a knee shuffle;
  - sitting on its heels or on the floor.
  - Kneeling needs about 0.6 m clear in front, so it kneels a step back and shuffles in on its knees.
  - **Beside the child she steps calmly** (A25b): she moves her legs (a step, kneeling down, a shuffle, a turn on her knees) on a tick when none of the child's shapes within 0.3 m of her legs moved more than 2 cm since the last tick, waiting at most 3 s in a phase for that, then going on.
- **Its cost.** One scripted pose takes 0.7–4 ms. Solving an act takes 1–430 ms, too slow for every tick. So an act is solved when it starts, and re-solved once a second while its target moves, warm-started from the last solution and interpolated in between (W2).
- It is the world's side only. Nothing of it enters the body except through the child's senses.

### 4.2 Its acts, and what a person can do for a 34 kg body

**Its acts:**

| act | status (measured today) |
|---|---|
| kneel beside the G1 and attend, one hand resting on its trunk | passes; at 0.78 m from its centre line, 0.10 m toward its feet, clear of its arm and the toys (no parent contact) |
| show a toy 40 cm before its eyes | passes; the toy held a little toward her side so her face is unblocked in both eyes (a ray test) |
| point, either hand | passes, including a drum 1.4 m away |
| hand a toy to the near or far hand | passes on the custom child (the grasp reflex held the rattle 3 s after release); on the G1, 6 of 10 toys can be held (section 3.8) |
| guide a forearm | passes, within the guide's cap only while the arm yields (below) |
| turn it from its front onto its back | a brief act within the caps (below); for distress face down only. There is no tummy time: face down, its eyes see only the mat (A7, A22) |
| pull to sit | only with the child's own flexion (A9); a limp sit-up needs more than a person gives |
| hold it propped near upright | passes within the caps (below) |
| pick up each of the 10 toys | passes |
| lift it, slide it on the mat, bring it back | refused: beyond a person's force |
| give the bottle, tidy | to build (W2) |

**A person's force caps** (ours, from ergonomic norms recalled from memory: the NIOSH patient-handling limit of 35 lb ≈ 156 N (Waters 2007), and the Snook and Ciriello push and pull tables; W2 checks them against the sources):
- one hand: 100 N sustained, 150 N for up to 2 s;
- both hands: 160 N sustained, 200 N for up to 2 s.

**What a person's hands can do for the G1** (34.39 kg, 337 N; forces are for both hands unless marked; "ramp" is a slowly growing push, "path" a scripted hand path):

| act | force needed | within the caps? |
|---|---|---|
| turn from its back to its side | ramp: 105 N to start, 149–169 N through; path: 420–464 N | within the caps by the ramp, but not used: there is no tummy time (A22) |
| turn from its front to its side | ramp: 94–170 N | as a brief act, for distress face down, narrated as her act (A7) |
| sit it up from lying (limp) | ramp: 111 N at 5° risen, 139 N at 20°, 217 N at 60°; path: 225–237 N peak | no: over 200 N. Only with the child's own flexion (A9) |
| hold it propped | 101 / 83 / 58 / 40 / 20 N at 60 / 45 / 30 / 20 / 10° from vertical | yes, within 30° of vertical |
| catch a fall from sitting | about 2.4 J to absorb (35° → 50°): stopped within 1–2 cm at 200 N (an estimate with no recorded source). The upper body's weight releases about 7 J from 35° to 50°; a limp trunk would pass 80° within her 300 ms reaction; the resting law's lag slows a sink to about 4° a second at 35° (from the hips' gains) | probably, for a sink under the resting law; not measured (C6) |
| guide a limp forearm | one hand: 37–44 N to raise the wrist 5–15 cm; 20 cm in 1.2 s: 139 N peak, 74 N to hold | yes, slowly |
| guide a forearm under the resting law, or stiffened | 201 / 211 N peak; 153 / 178 N to hold | no: the guide's cap stops it |
| guide against its resisting arm | 83–111 N | no: the cap is min(1.5 × its 65 N push, 100 N) = 98 N, so the guide stops. The child can refuse. |
| bend a leg | one hand, at the knee: 60–76 N | yes |
| slide it by the pelvis | μ × 337 N: 135 N at μ 0.4, 202 N at μ 0.6, 337 N at μ 1.0; 312–433 N measured on the mat | only on a slick floor (μ ≤ 0.47); never on the mat |
| lift it | 285–296 N for 1–5 cm | no |

- **Where she can put her hands.** A kneeling adult can place her hands at the start and the end of the roll, and at the end of the sit-up. The sit-up's starting grip needs a forearm twist of −115°, outside the human range, so the pull-to-sit holds the forearms instead.
- **A rigid scripted hold must never drive a 34 kg body.** A roll along a rigid hand path about the body's near edge reached 4–8 kN, driving the body through its own arm, leg and the mat. Every hold is a capped spring (below).

**Consequences:**
- **She comes to the child; she does not bring it back.** The all-out design's G1 line ("slides it … capped at 337 N") was beyond a person, and is replaced by these caps.
- **Every hold is a capped spring,** with each cap in the parent's constants file. A contact over the act's cap for 2 physics steps stops that segment and backs it off (A4).
- **Being held is felt.** Each hold's spring force is added to the touch of the link it holds, since it stands for the hand's grip. It counts toward pain under the same law as any force. Every cap (at most 200 N) is far under F_pain (1,012 N), so a hold is felt as touch, never as pain, and no exception to the pain law is needed.
- **The friction model.** MuJoCo's soft contacts let the G1 creep 5 cm at 146–199 N, below the true sliding force. W1 sets real frictions and a friction model that does not creep (C26).
  - Now the world's surfaces are 1.0, and the G1's own foot spheres are 0.6 at contact priority 1 (as shipped), so its feet already set their own friction on the mat.
  - **Real frictions are set on the world's side only:** the floor, the mat, the furniture and the toys take contact priority 2, so their friction and softness decide every contact with the G1 without touching its file. The G1's self-contacts keep its own values. The creep is fought with world options (the elliptic cone's `impratio`, or `noslip` iterations), never with the G1's geoms.
- **Clearing toys** before kneeling. The kneeling spot for the G1 is outside its leg sweep (W4 measures the sweep).
- **Her body is carried and strength-limited** (A25b). Her trunk is carried by a support capped at 695.5 N (never downward) and 257.4 N·m, so she can neither fall onto the child nor press down on it with more than her weight; each joint of hers is an actuator ranged at a woman's strength, damping included, so no act of hers exerts more than a woman can; her holds' effort is carried by her arms within that strength.
- **Her feet, knees and shins meet the floor by contact.** Each planned pose is raised until none of her pelvis's, legs' or feet's shapes is under the floor (her floor lift, A25b), so a leg is never planned into it.

### 4.3 Its feelings, its face, and what earns a smile

**Its feelings** (the owner's decision 1; `body/sim/parent_feel.py`). They are a small scripted state on the world's side, driven only by outward events the world logs.

| state | driven by | dynamics (ticks) |
|---|---|---|
| joy | a judgment of a completed, visible act (the worth table below), and nothing else | one pulse per judgment, of amplitude worth ÷ 2. It rises over 2 ticks, is held until seen (at most 20), then 10 ticks more, then eases off over 5. |
| displeasure | stage 2 only: the child's own act hitting her above her pain threshold (strength 0.5, reads −1), or talking over her (0.25, reads −0.5) | held 10 ticks, never waits for a look, eases off over 5. It ends any smile. |
| surprise | sudden events: a loud sound onset, a toy flying, a completed roll or first act, the peekaboo reveal | time constant 3; a same-kind event within 200 ticks surprises × 0.5 each time |
| concern | the child's pain (1.0) or distress (0.6) | time constant 40. It cancels any smile, and blocks new smiles while above 0.5: about 4 s after a pain, and all through distress. |
| attention | engagement: 1 in interplay, 0.4 during her own tasks, 0 away; and where she looks (the behaviour layer's choice) | time constant 5. A greeting eyebrow flash (3 ticks) at the child's first look after 200 ticks without one. |
| mood | +0.08 × worth, −0.15 × displeasure, −0.04 per child pain | time constant 4,000 |

**Its face.** It takes 12 graded parameters, each a FACS action unit at an intensity (`parent_kin.face_geoms_graded`):
- smile (AU12, the lips parting above 0.25) and cheek (AU6);
- frown (AU15);
- brows: inner raise (AU1), outer raise (AU2), lowerer (AU4);
- lids: raise or droop (AU5 or its negative), tighten (AU7);
- mouth: jaw drop (AU26), "oh" rounding (AU18/22), press (AU24);
- blink.
- A head tilt of 10° for the question face goes on the pose.

**How feelings map to the face:**
- the smile is the joy pulse; the cheek is s × (0.5 + 0.5s), where s is the smile;
- the frown is displeasure;
- the brow lowerer is max(0.9 × displeasure, 0.45 × concern);
- the brows rise with surprise, concern, the greeting flash and the question;
- the lids follow surprise, attention, winding down and low mood;
- the jaw follows surprise and speech;
- the "oh" follows surprise, never while smiling;
- the lips press with concern.

**The one law that keeps the reward honest.** The born reading is 2 × (smile − frown), in −2..+2. It reads only the mouth corners, and only joy pulses and frowns move the corners. Surprise, interest, concern, the question face, speech, the flash, drowsiness and mood all read 0. So looking at her, being near her, or surprising or worrying her can never be felt.

**When a smile starts.** A judged smile starts within the tick, unless one of two things holds:
- another smile is still on her face; or
- the child saw a positive face under 31 ticks ago and has not seen her neutral since (the born reading's disclosed 30-tick hold).

In either case the smile waits for 2 neutral ticks, and is dropped and logged if that takes over 40 ticks from the judgment. Only one smile can wait; the larger is kept. This rule uses only her own face and the face test's history.

**The self-test** (20,000 ticks, random gaze):
- **No judgments, anything else happening** (surprise, pain, questions, speech): the felt total was exactly 0, under sticky, glancing, always-looking and periodic gaze.
- **With judgments:** felt ≤ judged in every case. An always-looking child felt 179.0 of 179.0; glancing every 2–10 ticks, also 179.0; every 30 ticks, 128; every 45 ticks, 110.
- **Frowns:** each felt at most once, and only when seen.

**What the child's own pixels see.** Fovea pixels (32 px, 20.9°) that differ from her neutral face by more than 8 grey levels:

| distance | frown | +1 | +2 | surprise |
|---|---|---|---|---|
| 0.3 m | 88 | 107 | 155 | 147 |
| 0.6 m | 58 | 42 | 67 | 74 |
| 1.2 m | 15 | 15 | 18 | 21 |
| 2.0 m | 7 | 3 | 12 | 15 |

The grades are visible at the lean-in distance. Beyond about 1 m, only the born reading's world value carries them (A1, as disclosed).

**What earns a smile.** The worth is the reading's peak. The table is fixed before birth, only tightens, and is never loosened.

| act | worth |
|---|---|
| a met ask; a right name (exact); the call answered, until the name is understood | 2 |
| a first motor act: a whole roll of its own; its own reach and hold; sitting alone a moment | 2, falling with mastery as 1 + e^(−n/10) over its n earlier smiles, floor 1 (A2) |
| an approximation of a word (stage 2), until the exact word has been said 3 times | 1 |
| a stage-1 vocal turn in a pause while looking (at most once per 60 ticks); peekaboo answered by an act (A2: an act begun within 10 ticks of the reveal by an effector that had rested the 5 ticks before it) | 1 |
| a guided act the child repeats itself within 40 ticks | that act's worth |

- **Earns no smile:** a toy put into its hand, a guided act itself, part of an act, getting closer, looking, distress.
- "Slight warmth" (below 1) appears only while a smile eases off.

**Its gaze:**
- its eyes rest on the child's eyes while it talks;
- they go to the object during the naming word, then come back (the joint-attention cue);
- it leans over the G1's chest to be seen, its face never closer than 25 cm to the G1's eyes;
- when it calls or leans in, it places its face in the child's periphery, at least 15° off the fovea's line, and never moves it onto that line, so turning to it is the child's own act (A3);
- when feeding, it keeps its face in view, which pairs the face with the relief of the need.

### 4.4 Its voice

- **The engine:** macOS speech through a small Swift server (`AVSpeechSynthesizer.write`, with word markers).
  - It returns exact word onsets and is byte-identical on every run.
  - A sentence takes about 45 ms once warm (median 42–50 ms, 95th percentile 53–68 ms).
  - It runs at nice 10, off the tick loop. A cache miss makes the lockstep world wait about 50 ms of wall time and costs no sim time.
- **The voice:** compact Samantha, at rate 0.25 and pitch 1.15. None of the installed voices is at enhanced or premium quality.
- **Registers by intent** (Fernald 1989's contours), which carry real prosodic cues the amygdala can pick up:

| register | pitch | rate | contour | used for |
|---|---|---|---|---|
| plain | 1.15 | 0.25 | — | most lines |
| approval | 1.35 | 0.25 | rise–fall | "yes! the duck!" |
| comfort | 1.05 | 0.15 | falling | after pain or distress |
| calling | 1.25 | 0.25, +6 dB | rising | the child's name |
| question | plain | 0.25 | rising (the synthesizer raises the pitch on "?" itself) | asks |
| "no." (stage 2 only) | 1.0 | 0.3 | short, low | a hit or talk-over |

- **Infant-directed speech** (the owner's decision 6):
  - lines of at most 6 words, the focus word last (Fernald and Mazzie 1991);
  - the focus word on a pitch peak through SSML (`<prosody pitch="+30%" rate="70%">`). SSML was probed but its effect is not measured (P1). The fallback is a one-word echo at the approval pitch ("duck!");
  - the new word's lines at rate 0.2; P1 measures words a second, with a target of at most 3.
- **Its lines:** 4.0 words and 1.11 s (7.4 ticks) on average, 3.6 words a second.
  - At one symbol a tick, 95% of word tokens arrive on the tick their word ends and 5% one tick late.
- **The cache:** `body/sim/voice/cache/`, keyed by (line, register), with a 300 MB limit that drops the least recently used lines.
  - At each night boundary it pre-synthesizes every template line for the current vocabulary: 331 lines in 15 s, 14 MB.
  - It is saved beside the body, so a replay is exact.
- **Its mouth** opens each tick with the clip's loudness in that tick (the jaw parameter).

### 4.5 Its sentences

- **Two layers.**
  - **The fast layer** is scripted, deterministic and seeded. It picks short sentences from templates, filled from what the parent can see: what the child looks at, holds, or just said. It carries every judgment that must land within ticks. There are 331 distinct lines at birth.
  - **Claude, between wall minutes,** reads a plain-text digest of the last minute and writes one steering row: the focus toys, the next activity, fresh lines tied to situations, and the one new word to introduce.
  - Claude never controls a feeling, the face, the gaze or a judgment, and never sees anything inside the child. A14 lists what it may and may not do.
- **Intents, with examples:**

| intent | example |
|---|---|
| call | "pip. look at mama." |
| greet | "hi pip." |
| label (following the child's attention) | "it is a duck." / "you see the duck." |
| show | "look at the drum." / "see? a drum." |
| ask | "where is the ball?" / "what is this?" / "give me the cup." |
| confirm | "yes. the duck!" |
| recast | "ball. yes. the ball." |
| narrate | "uh oh. the ball is down." |
| body | "here is your foot." |
| motor | "sit. you sit." |
| feed | "here is your bottle." / "more?" |
| leave, return | "bye bye pip." / "hi pip! mama is here." |
| peekaboo, comfort, night | "peekaboo!" / "night night pip." |

- **Variation sets** (Küntay and Slobin; Onnis et al. 2008): 2–3 lines sharing the focus word, with frames differing by at least one word ("a duck." / "the duck!" / "you see the duck?"), 6 ticks apart. A set counts as one naming. At most one set per object per 120 ticks.
- **The line check** applies to every line, templates and Claude's alike:
  - at most 6 words;
  - only `. ? !` as punctuation;
  - only vocabulary words, plus that day's new word, placed last in the sentence;
  - never a held-out never-taught pair before its test (section 12).
- **The digest and steering.**
  - Every wall minute, `tools/sim_digest.py` writes about 40 lines of outward events only: posture; where the child's fovea rested, with the share on the parent's face; what it touched or held; what it said and when; the asks and their outcomes; the face events seen and unseen; pain; the ledger; the parent's last lines.
  - Claude appends a row to `data/sim_steer.jsonl`: {tick_from, ttl 2,000 ticks, focus, episode, task, away_ticks, introduce, lines, note}.
  - The fast layer checks each line, takes the row at the next utterance boundary, and uses each of Claude's lines at most 3 times.
  - Rows are logged by tick, so a replay is exact.
  - Without Claude, the fast layer runs alone. The brief is `ops/sim_parent_brief.txt`, in the manner of `ops/parent_brief_human.txt`.

### 4.6 Turn-taking and stages (in ticks)

- **Pauses:** 6 ticks between related lines; 20 ticks of expectant pause after a question, ask or call.
- **Judging:** 20 ticks for a gaze ask, 40 for an act ask. An act counts as met only once the effector that did it has come to rest on the result.
- **The child's turn:**
  - it ends when the tract has rested 2 ticks (silent) after sounding;
  - the parent replies 3 ticks later, answering what the child said: an expansion, a recast, an echo of its babble, or an answer (Goldstein et al. 2003; Goldstein and Schwade 2008, from memory).
- **Talk-over.** If the child starts sounding during the parent's clip, the parent finishes the current word (at most 3 ticks), stops, and looks at the child with a listening face.
- **Repeats:**
  - the call at most once per 240 ticks;
  - the same line not within 60 ticks;
  - the same object named at most once per 20 ticks.
- **Density:** about one line every 15–25 ticks while the child attends, about 1,500–2,500 words a life day, and at least 40% of play ticks free of the parent's voice.
- **Stage 1, until 5 words reach "says":** protoconversation.
  - A vocal turn taken in a pause while the child looks earns a small smile (+1), at most once per 60 ticks.
  - There is no talk-over frown.
- **Stage 2:**
  - among its sounds, smiles only for right words and met asks (the vocal turn's +1 ends); the worth table's motor rows are the same in both stages;
  - the talk-over frown returns;
  - an approximation the parent's ear accepts in context (section 4.9) earns a recast and a smile until the exact word has been said 3 times.
- Stages change at the next morning.

### 4.7 Its day (a life day of 24,000 ticks)

Anything urgent comes before the plan, in the behaviour system's order (section 4.10).

| episode | when | what the parent does |
|---|---|---|
| wake | ticks 0–300 | Morning light. It gets up from the sofa, kneels beside the mat, leans over the child's chest: "hi pip." (the greeting flash comes at its first look, as in 4.3), then the call. |
| meal (feed) | When the charge light is low (h < 0.35), not by the clock. With the G1's torques that comes about 5–6 times a day under babble and 1–2 times at rest (section 6). Each feed is 400–700 ticks of routine, of which the charge takes about 65 ticks of palm contact. | The routine: "hungry? here is your bottle." → the hold → "more?" → "all done." Stage 1: the bottle into the child's palm (the grasp reflex closes on it), her face in view, the comfort register. Stage 2 (after 20 holds over 2 days): the bottle within reach. Stage 3 (once it moves across the mat on purpose): the bottle on its dock, and she points. A bid while the light is low brings her at once; otherwise she checks within 200 ticks. |
| floor play | 3 blocks of 4,000–5,000 ticks | 2–3 focus toys; follow-in naming; toys shown beside her face and shaken for their sound; hand-overs of the holdable toys; the ball rolled toward its hands; asks; peekaboo; strays tidied. |
| motor time | 2 blocks of 1,000–1,500 ticks | By the child's visible stage, on the scaffolding ladders (section 4.10): toys at the edge of its reach; the pull-to-sit, rising only with its own flexion; the prop near upright, easing off and always catching the fall; calls from where its eyes can reach her (toward its feet when it lies on its back, A22). There is no tummy time (A7). It guides the child's own limbs while saying the action word, never doing the whole roll for it. |
| show time | about 1,500 ticks | Toys from the shelf one at a time: "what is this?", a pause, then the name. |
| away | 2–4 times, 400–1,200 ticks each | Never with the charge below 0.5, within 100 ticks of pain, or during distress. "bye bye pip." and a wave at the door; calls from the hall about every 600 ticks; answers bids from afar within 5 ticks ("mama is here."); comes back early after 3 bids in 40 ticks. |
| its own tasks | about 3,000 ticks | Tidies, sits on the sofa, eats at the low table, narrates; turns and answers when the child vocalizes. |
| winding down | the last 1,000 ticks | Dusk light; the lamp comes on. Comfort register, rate 0.15–0.2, at most one line per 40 ticks, no asks and no new toys, her lids lowered. |
| goodnight | the last 300 ticks | "night night pip.", the lamp dims, the parent goes to the sofa. No ask, no new word, no repositioning, so nothing is timed to the night. |
| night | after tick 24,000 | The world pauses in the dark; the parent sleeps on the sofa. |

**Turning and coming to it.**
- The parent turns the child onto its back when it is distressed face down for over 100 ticks: a brief act within the caps, narrated as her act.
- She does not bring it back when it scoots away. She comes to it, wherever it is (B7).
- She places herself where a roll would end facing her.

### 4.8 The words: the first 50, growth, and what "learned" means

**The first 50:**

| group | words |
|---|---|
| names | pip (the child), mama (the parent) |
| social | hi, bye, yes, no, good, uh, oh, night, peekaboo |
| toys | ball, duck, block, cup, bear, car, drum, bottle |
| body | hand, foot, head, tummy |
| room | mat, sofa, window |
| actions | look, give, roll, sit, up, down, crawl, more |
| function | the, a, is, you, it, here, there, on, in, your, where, what, this, at, me, see |

- **The growth queue** holds 76 words: rattle, book, red, blue, yellow, big, little, push, drop, shake, stand, eyes, mouth, nose, clap, wave, sleep, go, get, hold, want, all, done, and so on, plus inflections (sat, rolled, fell, got).
  - The toy the child handled most that day comes first, then first-words frequency.
  - The world's rattle, stacker and ring are named through the queue.
  - Colour words wait until two toys share a colour (the colour twins, B2).
- **The parent's ledger** uses outward events only:
  - **heard:** said with the referent in the child's view.
  - **understood:** after "where is the X?" or "look at the X", with X visible but not in the fovea, X lands in the fovea within 20 ticks and stays 2 ticks. For an action word, the act follows within 40 ticks.
    - It must hold on at least 5 of the last 10 asks, and beat the child's own base rate (the same test at random moments with no word said) with a one-sided binomial p < 0.05.
  - **says:** the child says X with X in its fovea or hand, or right after doing the act; not within 10 ticks of the parent saying it; 3 times over at least 2 life days. Anything said within those 10 ticks counts only as an echo.
    - Said by the silent token output: its token, or letters within edit distance 1.
    - Said by the tract: accepted by the parent's ear in context (section 4.9). The ledger keeps the two apart.
- **"Learned" means understood.** Saying the word is the next rung, and using it in a never-taught way is the test of understanding (section 12).
- **Growth pace.** About 12 content words are active at a time.
  - The next word enters when at least half of the active set is understood.
  - At least one new word every 2 life days, at most 3 a life day.
  - A new word is said sentence-final, in 3 lines within a minute (one variation set).

### 4.9 The child's voice: the vocal tract, and the word scaffold

The owner's decision 5: the voice is an articulatory vocal tract, so the child babbles, hears itself through its two ears, and can learn to match the parent's words. The word and letter channel stays as a scaffold at birth and is removed later. The first word from its own tract is a milestone.

**The tract** (`body/sim/tract.py`). It is the voice effector's physics, run on the world's side like MuJoCo. The body sends one step per articulator per tick and hears the result only through its own ears.
- **A source-filter articulatory synthesizer in numpy,** in the manner of Sondhi and Schroeter (1987). The sources are made in the time domain, and the tube's response is applied in the frequency domain.
  - Ten articulators set a 24-section tube 12 cm long (a child's), plus a nasal branch, updated every 10 ms.
  - Each articulator moves toward its target with damped muscle dynamics, so one tick's move finishes within 57–120 ms.
  - The tube's losses (viscous, thermal, soft walls), lip radiation and the nasal side branch follow Flanagan.
  - Airflow is quasi-static: the glottis and the narrowest point act as orifices in series. So voicing stops by itself during a closure.
  - One turbulence law (noise grows as Re² − Re_c² above Re_c 1,800) makes the breathy /h/, hisses and release bursts, at the glottis and at the narrowest point.
  - The voice is glottal pulses with pitch (180–546 Hz, resting about 265 Hz), jitter, shimmer and a breathy-to-pressed spectral tilt.
  - A breath reservoir (400 cm³: about 2.6 s of speech, refilled in 0.8 s at rest) makes breath groups.
- **No phoneme is written anywhere.** Vowels, the nasal murmur, hisses, bursts and a stop's silence all come from the tube's shape and the airflow.
- **Calibration** (all ours, disclosed):
  - the four corners of the tongue's range were fitted to children's vowel means (Peterson and Barney), with each constriction held at its anatomical place. The results: /i/ 344 / 2,719 Hz, /a/ 938 / 1,625 Hz, a rounded /u/ about 375 / 1,250 Hz;
  - an honest gap: the vowel space is about the size of an adult woman's. Front vowels fall short in F2 (2,700 against 3,200 Hz) and low vowels in F1 (about 900 against 1,030 Hz);
  - levels: /a/ matches the parent's speech level; a hiss sits at −14 dB and /h/ at −13 dB relative to /a/.
- **Cost** (measured on this Mac while other sessions shared it):

| case | ms per 150 ms tick |
|---|---|
| sounding (vowel, glide, fricative), a quieter moment | 1.8–3.0 |
| sounding, under heavy load | 7–10 |
| at rest | 0.12 |
| its own voice to both ears | +0.17 |
| the babble average (6% of ticks sound), under load | about 1.6 |

**The alphabet and the gate.**
- **The voice effector is 10 articulators:** lungs, glottis (which also decides voicing), pitch, jaw, tongue front/back, tongue height, tongue tip, lips, rounding and velum.
- **Each takes one of {−0.6, −0.2, 0, +0.2, +0.6} of its range per tick,** from where it is (the servo law): 50 per-joint striatal rows and one softmax per articulator.
  - The steps were enlarged on purpose. The first steps, ±0.1 / ±0.3, could not open the lips in one tick; in real speech the lips and tongue tip cross their range in about 70 ms.
- **The gate** is born like the others (`birth_act` 0.25). Its own inputs: its own act last tick, the forward error on its hearing, and breath left.
- **At rest the targets relax to a silent resting posture** (nose breathing, lips nearly closed, the velum down), with a time constant of 1 tick, so a rest means quiet.
- **Pace.** One articulator target per tick gives at most 3.3 syllables a second, about canonical babbling's rate. The accepted words took 4–6 ticks (0.6–0.9 s).
- **The born babble,** measured with movement units (seed 7, 4,000 ticks; seed 8, 12,000 ticks):
  - 6–7% of ticks sound, and only 21–22% of the ticks when the gate is open;
  - 6.4–7.8 utterances a minute, averaging 3.6 ticks;
  - 0.6–0.7 closure-then-voiced-opening transitions a minute (canonical-like syllables, by chance).
  - The born babble is sparse breaths, squeals and glides. So the first thing the voice must learn is to make sound at all, which stage 1's smile for a vocal turn rewards.

**How the child hears itself.**
- The tract is a software organ driving the G1's own loudspeaker, as the software fovea keeps the body stock. That the real G1 has a speaker, and where it sits, is from memory and still to check (B5).
- In the sim the sound comes from the head's front and is spatialized to the two microphones.
- It goes through the same cochleas and brainstem (channel 2). Its own voice arrives about 14 dB above the parent's at 1.5 m. There is no bone conduction.
- An act at tick t is heard at t+1, as with the limbs.
- The tract's 10 positions and velocities, plus breath left, join the body channel (21 numbers).

**Vocal learning within the laws** (proposed; only the inverse model's starting point was measured):
- **The voice declares the ears as its consequence sense,** as the limbs declare their joints, so this stays body-general.
  - Its forward half (the stream plus the step → the next ear code) and `act_inv` (the ear codes at t and t+1 → the steps) learn from its own babble, labelled by the efference copy.
  - The forward error tells its own sound from someone else's.
- **The parent's words are demonstrations under the arm's law.** While the voice rests and its ears change in a way it did not predict, `act_pred`'s target is `act_inv`'s reading of that change, weighted by `act_inv`'s reliability.
  - This is R6's rest law with the ears as the voice's consequence sense, nothing more. It cannot tell speech from other sounds and is never told: her words, a toy's squeak and a footstep are all read as acts it could have made. It acts from birth, and it is the only imitation route.
  - The mapping is learned from its own babble, so there is still no mirror module (A10).
- **Reward comes only through the face:**
  - stage 1: +1 for a vocal turn;
  - stage 2: smiles for words the parent's ear accepts: right names, met asks, and context approximations (a recast, then a smile until the exact form has been said 3 times).
  - Nothing rewards sounding like the parent. A songbird-style inner "matches the tutor" signal was refused as an intrinsic reward.
- **The measured starting point** (`t_imitate.py`):
  - a linear inverse model trained on 16,000 babble ticks (1,082 sounding) fits its own voice poorly. Held-out R²: glottis 0.92; lungs, pitch and lips about 0.5; jaw and tongue 0.2–0.37;
  - it read 12 of the parent's words into echoes with a median rank of 22 of 50 (chance 25.5). The parent's ear accepts 2 of 12 in context ("see" 100%, "ball" 28%).
  - So early imitation rests on the cortex learning to hear across voices. `act_inv`'s reliability is measured on its own voice, so it overstates on the parent's.
  - A method for the parent: echo the child's babble back (Goldstein and Schwade 2008).

**How the parent hears the child's words** (`body/sim/parent_ear.py`; its cochlea `body/sim/ears.py`).
- **Features:** the same cochlea → log bands → a shift of 0–4 bands (the parent adapting to a shorter tract) → cepstra c1–c12, mean-normalized → slope-constrained DTW (Itakura).
- **How well it recognizes, as measured:**
  - across voices of the same synthesizer: 94–100% right among the 50 birth words;
  - across 8 other voices, with templates in the parent's voice only: 14–36%. The 8 are other macOS voices (Flo, Sandy, Shelley, Eddy, Reed, Junior, Kathy, Fred): other synthesizers, not recordings of people (A27). (The earlier "66–80%" was within one synthesizer.)
  - with a template bank of many speakers (the parent plus 8 other voices; the tested speaker held out): 72–96% among 50, and 98% within a context set of 6.
- **The decision rule that works.** A word c is accepted when both hold:
  1. c is the nearest of the words the parent expects in this situation;
  2. d(c) − d(the child's own babble bank) < −m.
  - m is set so held-out babble passes as the context word 2% of the time: m = −0.35 on a bank of 193 babble utterances. At that setting, the other voices' words pass 100%.
  - A fixed distance cutoff fails: where 1% of babble passes, 0% of the other voices' words do.
  - The bank is recorded from the babbler before birth and kept fixed. Adding rejected near-words in life would lock those pronunciations out.
  - m is re-set for the real context set sizes (P3, P6).
- **Cost:** about 287 ms per utterance against the whole bank, on the world's side, at the utterance's end.
- **Chance:** about 0.13–0.16 accepted approximations a minute per context word, written down as chance.
- **The tract's reach** (acceptance in context over 50 random context sets each; the hand scores and the searched acts are instruments written by us, never given to the body):

| word | hand score | searched acts |
|---|---|---|
| mama | 100% | 100% |
| ball | 0% | 100% |
| duck | 0% | 100% |
| up | 100% | 0% |
| see | 100% | 0% |
| no | 0% | 82% |
| pip | 78% | 0% |
| hi | 68% | 0% |
| bye | 0% | 0% |

**The scaffold, and removing it.**
- **Input (channel 0):** a fixed born table of 50 word tokens, 26 letters, a space, rest and end (79 rows), shared by the ear and the output.
  - A word after the first 50 arrives as its letters, one a tick from its onset, then a space. There are no new rows during life.
  - Tokens arrive only while the parent is audible.
- **Output (effector 1):** the same 79 rows as a silent effector with its own gate. The parent reads it as an exact transcript, matching a letter string to a word when it is exact, within edit distance 1 (2 for words of 6 letters or more), or a prefix of at least 2 letters of what the child sees or holds. It replaces the synthesized child voice.
- **Words change the world** (from either output):
  - a toy's name said in a pause brings that toy;
  - "mama" brings the parent close;
  - "up" starts the pull-to-sit;
  - "more" gets more of the feed or the game;
  - "no" stops the parent's offer;
  - "peekaboo" starts a round.
- **Removing the input scaffold,** on a copy, at a night boundary. The removal only silences the channel: it reads "no token", as on every tick the parent is silent, and nothing in the body changes, so it is a change of the world, not the body (A29).
  - An earlier draft gave the words head a new target after removal: the voice's inverse model's reading of what was heard. That reading is articulator steps, not a word, so it cannot be a words target; and a rule for "heard speech with no token" needs the body to know which sounds are speech, which is world truth. It is dropped. After removal, the words live in the ears alone, which is what the two tests below measure.
  - Remove it when both hold:
    1. the words channel's own forecast (today's `latent_pred`), made on the tick before each token arrives and so from the sound alone, names the parent's word for at least 70% of the birth words;
    2. a copy day with the words channel silent keeps at least 80% of the token day's comprehension hits and right names.
- **Removing the token output** (ours, on a copy at a night boundary), when all three hold:
  1. the input test passes;
  2. at least 10 words reach "says" from the tract alone;
  3. a copy day without the token output keeps at least 80% of the right names.
- Re-test every 5 life days after M6 begins.
- **Risk:** tokens are the easier road to reward (risk 10).

**The milestone M6t, "first word from its own tract"** (section 12):
- the ledger's "says", judged on the tract's sound alone with the tokens ignored: accepted in context, not within 10 ticks of the parent saying it, 3 times over 2 life days;
- chance is its own rate with the referent absent, plus the babbler's 2%;
- a precursor rung, "first accepted echo", shows the imitation route working;
- its never-taught test: the word used in a never-paired place, or used to get the toy.
- Not forecast.

### 4.10 Its conduct: the behaviour system

The owner's decision 6 calls the parent the environment's most important part. Its conduct runs in four layers.

| layer | runs | does | latency after the child's act |
|---|---|---|---|
| L0 motion | every physics step | her muscles toward her interpolated plan, each within her strength; her trunk carried by its capped support (A25b); capped-spring holds; the yield (A4) | — |
| L1 reflexive | every tick | gaze to the child's act, to a sudden event, or to its gaze target; the face from the feelings; a hand withdrawn when hit | gaze ≤ 2 ticks (0.3 s); face ≤ 1 tick |
| L2 conduct | every tick, on events | judgments; replies; follow-in naming; offers and guides | voice 3–5 ticks; a hand act starts within 7 ticks (1 s). In lockstep, the solver's wall time costs no sim time. |
| L3 episode | at boundaries | the day plan's episode and its routine | — |

**Every tick, before her plan is carried out** (A25b): her planned pose is raised off the floor; her planned legs, trunk and head are measured against the child's body and legs where they are now, and nearer than 3 cm she straightens and moves back off it; beside the child she moves her legs only while its limbs near them are still (the calm step, at most 3 s a phase).

**Priority**, highest first: the child's pain (concern and comfort); being hit (withdraw); a low charge (the meal); the child's vocal turn (reply); finishing her word; judging a pending ask; joint attention; the episode's act; idle (watch, at most one line per 40 ticks).

**Contingency.**
- Every outward act of the child gets her gaze within 0.3 s.
- A judged act gets the smile and the approval word within a tick.
- A vocal turn gets a reply 3 ticks after it ends, often echoing and expanding its babble.
- P6's ruler: at least 90% of the child's acts answered within 7 ticks while she is within 3 m and not away (C24).

**Joint attention.**
- The child's target is the object the central ray of its software fovea hits (either eye) for 3 ticks running, or the object in its hand. She reads it from the fovea's window, as a person reads an infant's eyes (B14).
- She looks there within 2 ticks.
- If her voice is free, the object is nameable, and it has not been named in the last 20 ticks, she names it with a variation set. Her eyes are on the object during the name, then back on the child's eyes.
- She redirects ("look! the drum.", with a point) only after 40 ticks with no target. Follow-in naming outnumbers redirects at least 2 to 1 (Tomasello and Farrar 1986).
- Showing: the toy beside her own face at about 40 cm from the G1's eyes, shaken for its sound.
- With the tract: a babble her ear accepts as a visible referent's name is echoed and expanded within 3 ticks ("ball! the ball!"). There is no separate syllable matcher (A27).

**Scaffolding** (Wood and Middleton's contingent-shift rule, per task): after a success at help level k, the next attempt starts at k−1; after a failure, at k+1; every task starts at 1. Every top level ends with her letting go. The smile comes only from the child's own completion.

| task | ladder |
|---|---|
| reach and hold | 0: a toy at the reach boundary + 5 cm, on the side its fovea is on · 1: at 80% of its reach · 2: 20 cm from its hand, shaken · 3: touched to its palm (narration only, no smile) |
| roll | 0: she kneels where the roll would end · 1: plus a toy just beyond reach · 2: guides the far arm across (≤ 65 N) · 3: bends the far knee over (≤ 76 N), then lets go. She never does the whole roll. |
| sit | the pull-to-sit by the forearms, rising only with its own flexion (A9); the prop near upright, eased 100 / 70 / 40 / 20%, then hovering |
| on its front (its own roll; no tummy time) | 0: her voice from beside it · 1: a toy on the mat where its eyes look, under its chest · 2: after 100 ticks of distress, she turns it onto its back (A7). No lift is asked for: face down its eyes see only the mat (A22). |
| give and release | 0: an open hand plus the ask · 1: plus a point · 2: a touch on the toy. She never pulls. |

**Routines** (Bruner's formats: a fixed opening, order and closing): the wake, the meal, play and show, winding down, goodnight (section 4.7), and her own day (tidying, the sofa, her own meal at the low table).

**Leaving and returning:**
- she leaves with "bye bye pip." and a wave at the door;
- she calls from the hall about every 600 ticks, spatialized;
- she answers a bid from afar within 5 ticks ("mama is here.");
- she returns early after 3 bids in 40 ticks;
- the reunion is "hi pip! mama is here.", arriving in the child's periphery, and the flash at its first look. No smile without an act.

**Being hit.** Her own pain threshold is 150 N (a 10 ms mean) on any segment. This is ours, a person's, not the child's 1,012 N. The G1's arm pushes 65 N static, and its swings exceed that.
- Stage 1: surprise, "oh!", and she withdraws her hand. No frown.
- Stage 2: a frown (−1) and "no.", but only for the child's own motor act. A reflex she triggered is logged as her defect.

**What Claude steers between minutes** (A14, extended):
- **May:** choose the episode, the focus toys and which scaffolding task to work on; write situation lines and variation-set frames, each passing the line check; choose the day's new word and the away time; write notes; propose method changes, applied only at a night boundary after a test on a copy.
- **Must never:** set or nudge a feeling, the face, the gaze, a judgment, a worth, a help level or a force; write praise or use the approval register; loosen a criterion, or choose activities for their smile rate; see the child's insides (the digest reports face events, not feeling numbers); act inside a minute or at night; exceed the word pace, or name what cannot be shown within the minute.

## 5. The world

### 5.1 The living room

- **The room:** 5.2 × 4.6 m, 2.6 m high, with an oak floor.
  - **The play mat: 2.8 × 2.0 m of foam, 7 × 5 tiles of 0.4 m.** That is about 2.1 body lengths of the G1, the same ratio the 1.6 m mat gave the custom child (B11).
  - A sofa against the back wall, a low table, cube shelves with baskets.
  - A window with curtains, a doorway to a hall (the house to come), a floor lamp, a plant, wall art.
- **Lights:** a sun (directional, casting shadows) through the window, a key spot and a fill. Their colour and direction change across the life day (section 5.4).
  - A ceiling spot light blacked out the ceiling in MuJoCo 3.9's renderer on this Mac, so it was removed; the ceiling is most of what a child on its back sees.
  - The sun's shadow stays in the child's eyes. Dropping it would save about 21 ms a tick but take a piece of the owner's complete reality (decision 3), so it is the owner's call (B3), never decided by the eye check or by the tick.
  - The G1 file's own directional light (a Menagerie scene light, not part of the robot) is switched off at load.
- **The play space is the whole floor** (B7). The mat has no walls. The parent cannot carry the G1 back, so she comes to it.
- **The G1 in the file.** The room includes `assets/unitree_g1/g1_with_hands.xml` first, so its `stand` keyframe still addresses its own joints.
  - MuJoCo applies the last `compiler` element to the whole model, so the room is written in radians (every `euler` converted in one place), and the G1's mesh folder is given from `body/sim/`.
  - The room's contact defaults sit in their own class, so the G1's defaults are untouched.
- **Collision classes:** world 1 (floor, walls, mat, furniture); the G1 keeps its own contype 1 and conaffinity 1 (self-collision on, as shipped); toys 4; the parent 8, with conaffinity 1 so it touches the G1.
- **Physics:** MuJoCo 3.9, 2 ms steps with `implicitfast`, 75 steps a tick, **the elliptic friction cone, multi-point collision detection off.**
  - With the pyramidal cone, a block squeezed in a Dex3 hand stopped MuJoCo ("FactorizeHessian: rank-deficient sparse Hessian"); with the conjugate-gradient solver or a dense Jacobian it gave NaN accelerations and a reset.
  - With the elliptic cone and multi-point detection on (MuJoCo's default), 8 of 360 grasp trials blew up to NaN about 0.1 s into closing the fingers, and MuJoCo silently reset them. With it off: 0 of 360, and the babble ran clean.
  - The world catches `mujoco.FatalError`: two hands posed 3 cm into each other raised it inside `mj_forward` (A18). It also disables MuJoCo's auto-reset (`<flag autoreset="disable"/>`), so a bad state is never silently replaced.
  - The world's geoms (floor, mat, furniture, toys) take contact priority 2, so real frictions are set on the world's side (C26).

### 5.2 Toys and their sounds

Every sound comes from an event in the physics: a toy's own clip on its event, scaled by the impulse or speed; contact clicks; the parent's voice and footsteps (from its gait); the child's tract. Each source is placed in space for the two ears.

| toy | colour | size | held by a Dex3 hand? | its sound |
|---|---|---|---|---|
| ball | red | 12 cm | no (B1): pushed, rolled, named | a soft rubber bounce when it lands or is struck |
| block | blue | 7 cm cube | yes, 9 of 9 | a hollow wooden knock on contact |
| duck | yellow | 11 cm | no (B1) | a squeak when squeezed |
| cup | green | 0.8 scale, 6.7 cm across | yes, top grasp 3 of 3 | a plastic clink when it strikes something |
| rattle | purple | a handle and a head | yes, 4 of 6 hand-overs | beads shaking, louder with speed |
| car | orange | 14 cm long | yes | a wheel rattle while it moves |
| bear | brown | 12 cm | no (B1) | a soft bell inside when it moves |
| stacker | white, with coloured rings | 11 cm base | yes | the rings' plastic clack |
| drum | cyan | 15 cm | no (B1): hit | a boom when hit on top |
| ring (teether) | pink | 10 cm | yes, 6 of 6 hand-overs | a crinkle when handled |

- **Where they lie at birth:** around the G1 on the mat. The block, cup and rattle are within its lying reach. The rattle was moved 7 cm nearer the right hand, and the block and duck were moved out from under its arm and leg.
- **Colour twins** for the never-taught word test (B2): a blue ball, a red block, a yellow cup and a green car, each a copy of its toy in another colour, introduced before the colour words.

### 5.3 The charger

- **The bottle.** A toy-sized bottle charges the child while it touches either palm: 0.01 a tick, so from empty to full in 100 ticks.
  - It is sized for the Dex3 in W3 (the cup held only at 6.7 cm across).
  - Its dock is the prototype's white disc. The maker still has it at the mat's corner, far from the child; W3 moves it beside one hand (A17).
  - The dock moves out of reach only once moving across the mat toward things beats chance.
- **Why a bottle** (the owner's decision 6 calls charging the child's meal):
  - the palm and the grasp reflex make feeding possible from day one;
  - "bottle" is a first word;
  - the parent's feeding pairs her face with relief.
- **When the charge is low,** the parent brings the bottle. An empty child far from the bottle almost never reaches it, because weakness makes it worse.

### 5.4 Day, night and a complete reality

The owner's decision 3: sounds only from real events, day and night light, the parent's whole day as a parent, and no shortcuts in the body.
- **Light across the life day.** The sun moves across the window from morning to dusk over the 24,000 ticks, with its colour warming at the ends of the day. The lamp comes on at winding down and dims over goodnight's last 30 ticks. The night is dark. The morning's light returns over the first 30 ticks of the wake.
  - The eye check (C3) runs under morning, midday and dusk light. It reports what the fovea can tell apart in each light; it never changes the light. Making the world easier to see because the child's eyes find it hard would fit the world to the learner.
- **Night.** At the end of goodnight, `world.pause()` freezes the world exactly as it is. Nothing is moved, and the morning resumes from the same state. The parent sleeps on the sofa.
- **Sounds only from real events** (section 5.2). The room's echo is B9's default. The G1's own motors, which hum on the real robot as they work, are B20's.
- **No shortcuts in the body.** Every channel comes from a sensor where the real G1 has one, with the four disclosed exceptions of section 3.4.

## 6. Reward

Summed in this order.

1. **Face** (the increment rule, for the sim body). A rise of the reading's positive part is felt as positive, and a rise of its negative part as negative; no fall is ever felt. It is clipped to ±2.
   - It is felt only through the born expression reading, which updates only while the parent's face passes the face test on 2 consecutive ticks (A1). An unseen smile is not felt, and a face seen again unchanged is not felt again.
   - For steps from neutral it equals today's rule (a rise felt as the new level), which the language body keeps.
   - Why it changed: today's rule over-pays a graded onset. Measured on the graded face, a watching child felt 276.8 of 185.5 judged (+49%).
   - It is grounded in the parent's visible judgment of world events.
2. **Pain.** −1 on a tick when any link's force exceeds F_pain = 3 × the child's weight from the model file: 3 × 337.4 N ≈ 1,012 N.
   - A link's force for pain is the tick's largest 10 ms mean (5 physics steps) of its summed normal force (A12).
   - At rest the largest contact at the tick ends is 272 N. Under babble, single-step peaks at the tick ends reached 944 and 1,799 N; W4 measures the pain rate with the filter (C5).
   - Pain at a joint's limit is a candidate, not measured, and not at birth.
3. **Charge.** 4 × [D(h_t) − D(h_t+1)], with D(h) = (1 − h)². This is drive reduction (Keramati and Gutkin 2014), so a full charge earns nothing.
   - The drain is 4e-5 a tick plus 2e-3 × (Σ τ² ÷ Σ τ²_max), over the G1's own limits.
   - Under babble (0.063) a full charge empties in about 6,000 ticks and falls to the feeding level (0.35) in about 3,900, so about 5–6 feeds a day. At rest (0.004) it takes about 13,500 ticks, so 1–2 feeds a day.
   - Only the child's own actuator torque counts, so being guided costs at most its cap.

**Left out:** any reward for moving, getting closer, looking, hearing words, sounding like the parent, novelty, or anything read from the body's insides.

## 7. The core at this scale

### 7.1 It fits

- The branch's core serves a humanoid of about 40–56 joints at d 512, 6 blocks, 8 heads and window 64.
  - It measured 35.2M parameters (31.3M trained) at a table of 600 rows, against 28.8M for the language anatomy.
  - Peak memory with a small store was 1.68 GB, against 1.50 GB.
- **R5b is done** (0103f0a): the striatum holds a later effector's events by joint. A body of 34 joints of five settings needs 1,360 rows (11.1 MB); the G1's 56 joint readouts need about 2,240 rows (about 18 MB). Before R5b, one six-joint limb would have needed about 1 GB.

### 7.2 The critics set the cost, not the body

- The fast critic and the face organ each read the striatal expansion: 2,048 units plus the working-memory slot, 4,097 wide in float64.
- A rank-one update with forgetting costs 7–13 ms per matrix per tick. A solve costs 0.78–0.88 s per matrix every 64 ticks. Together that is about 40–50 ms a tick, half the tick or more.
- **So there is no striatal territory per limb for the critics to read.** Nine territories of 256 units would make the critics' input about 8,700 wide: about 4.5× the update cost and 9.5× the solve.
  - The limbs' events join the one expansion instead, with per-joint rows scaled by 1/√(k·J), so 56 joints do not swamp the words' line.
  - R7 adds event lines for touch onset by zone group, pain, a face in the fovea and a sound onset on each side. The value can then rise at the causal moment, not only when the smile is finally seen. The amygdala reads the same lines.
- **The continuous scene** (a hand near a toy) reaches the value through the ventral critic, which reads the fast ladder bands 0–2 (a sim constant). That is about 1,546 wide, about 3–4 ms a tick.
- **A sim constant:** solve both 4,097-wide matrices every 256 ticks instead of 64. It saves about 7–20 ms a tick. The weights then refresh every 38 s of sim time, and the evidence still accumulates every tick.

### 7.3 Credit across many limbs

- One scalar dopamine for all ten gates does not dilute the expected signal to the limb that caused a reward, because each gate learns the covariance of credit with its own acting.
- It does add variance, and it lets limbs that do not matter drift ("superstitious" movement). Infants show the same shape: in the mobile studies of Rovee-Collier and Thelen, kicking first rises in both legs, then only the tied leg keeps it.
- The delay is the real problem (risk 1). It is carried by:
  - movement units (fewer decisions per reward);
  - the critics' event lines;
  - the parent's spoken praise, which the child's orienting brings into view;
  - the switch fixes at birth;
  - coordination learned through `act_pred`: from the body's own acts, from the parent's guidance, and above all from the night's replay, where the amygdala's tag decides which episodes are dreamt first and the replayed dopamine decides which way their acts are taught (R8, section 7.4).
- Reward chooses which whole-body patterns are replayed; supervised learning spreads them over the joints. There is no per-limb reward and no intrinsic bonus.

### 7.4 The amygdala: the valence tagger (the owner's decision 2)

Status: specified, and measured on a synthetic stream; nothing is built. It replaces the "valence learner" of this morning's design (A16). It is on for the sim at birth. It is off by absence for the language body, whose digests hold.

**What it is.** A named organ of the core, in `body/core/amygdala.py` as `AmygdalaMixin`, with its constants in `AMYG` in `physiology.py`.
- From what the body senses now, it learns fast which cues predict good and bad, from each of the body's reward sources, over the next few seconds.
- It marks each moment with its emotional weight, the tag: what the moment is expected to bring, plus what it brought.
- The tag sets how strongly the moment is written to the fast memory, how likely the night is to replay it, and whether orienting turns toward or away.
- It never makes reward, changes reward, produces dopamine, enters any gate's credit or trains the cortex. It cannot pay the child for anything.

**Biology:**
- the amygdala as the Pavlovian cue-to-outcome learner, with LeDoux's fast thalamic "low road" and slower cortical "high road";
- separate positive-valence and negative-valence neurons in the basolateral amygdala (Paton et al. 2006; Namburi et al. 2015; Kim et al. 2016);
- the amygdala strengthening the storage of emotional events (McGaugh 2004);
- behavioural tagging: a strong event within a window also secures weak memories made around it (Frey and Morris 1997; Redondo and Morris 2011);
- rewarded and emotional experiences replayed more in sleep, the hippocampus and amygdala replaying together (Ambrose, Pfeiffer and Foster 2016; Girardeau et al. 2017);
- conditioned orienting through the central amygdala (Gallagher and Holland 1999);
- the learning law as the least-squares (Kalman) form of the Rescorla–Wagner rule (Dayan and Kakade 2001).

**What it reads.** Each tick, `x = [C/√d, the event lines, 1]`:

| road | what | numbers (sim) |
|---|---|---|
| high (cortical) | the cortex's stream C, read when the tick's choice reads it, and detached: everything the cortex has made of the eyes, the ears, the joints, touch, balance and the charge. A voice's tone, the parent's face turning or changing, and the charge falling are carried here. | 512 |
| low (thalamic) | R7's event lines, one declaration read by both the critics and the amygdala: touch onset in 8 zone groups (head and trunk together as the G1's torso, pelvis, each arm, each hand, each leg); pain on any link; a face in the fovea (the born three-blob template on the fovea's pixels, never the world's face test: section 3.4); a sound onset on the left and on the right (the cochlea's onset and the born lateral read) | 12 |
| level | 1 | 1 |

The language body declares no event lines.

**What it learns.** One head per reward source and sign; each `RewardSource` declares its signs and whether it reaches the amygdala.
- The sim has 5 heads: face +, face −, pain −, charge +, charge −.
- For head h, from source s with sign σ, what arrives is u = max(0, σ × the source's term), and the head forecasts y_t = Σ_{k≥1} γ^(k−1) u_(t+k) with γ = 0.9375: dopamine's own discount, so no new time constant is added. The sum is not cut off; beyond 64 ticks its weight is under 2%.
- The heads are split because the basolateral amygdala keeps good and bad apart. A single signed forecast would net to zero a cue that brings a smile and then pain, although that cue matters most.

**Its law: the critics' least squares, with a shorter memory.** It does not bootstrap, so the value learner's deadly triad does not apply.
- The eligibility trace: e_t = γ e_(t−1) + x_t.
- The evidence: b ← β b + e_(t−1) u_tᵀ; A ← β A + x_t x_tᵀ; β = 1 − 1/τ_a. This is the exact backward form of regressing y on x, so an outcome enters the evidence on the tick it is felt.
- The solve every 8 ticks: W = (A + R)⁻¹ b, with R_ii = 0.3 × τ_a × each input's running variance (the critics' prior); the level term is free; a Cholesky solve, falling back to lstsq.
- The forecast: ŷ = max(0, Wᵀx) per head.
- **τ_a = 4,096 ticks** (band 6's clock), about 10 minutes of life: eight times the input width, so the fit is determined.
  - Few-trial learning comes from least squares taking a full step along a direction it has rarely seen, not from τ_a. τ_a sets how fast the organ follows a drifting cortex and a changed world.
  - The old entry's forgetting of 0.998 remembered about 500 ticks, fewer than its 525 inputs.
- It learns only while awake. Its evidence is kept across the night; the morning's changed cortex is followed by the forgetting.
- **Cost, measured** at 525 inputs and 5 heads, float64 on one thread, at nice 19 under load: 0.10 ms an update and 0.69 ms a solve, so **0.19 ms a tick**. It adds about 2.3 MB to the save and draws no random numbers.

**Its reliability: the voice it has earned.** Each head's ρ is the running correlation of its forecast with the realized target. The target is finalized 64 ticks later (or at nightfall); the moments run over the critics' 36,000 ticks; ρ is clipped to 0–1 and is 0 until 64 finalized pairs exist. This is the face organ's own measure. At birth every ρ is 0, so the forecasts reach nothing until they have proved themselves.
- Anticipated good: A⁺ = Σ over positive heads of ρ ŷ.
- Anticipated bad: A⁻ = Σ over negative heads of ρ ŷ.
- Net valence: N = A⁺ − A⁻.

**The tag:** tag_t = min(2, A⁺ + A⁻ + R_t), where R_t = Σ |term| over the sources that reach the amygdala: what was felt this tick. The cap is source 0's clip. It is in reward units, the scale every critic learns in. (A tag scaled by its own running mean would put ordinary moments at half the cap, and would need a saved, warmed-up mean, defect 7's failure.)

| moment (sim) | tag |
|---|---|
| an ordinary tick: the charge's drain and its forecast | about 0.01; at low charge and hard effort, about 0.05 |
| a feed tick: its relief and its forecast | up to about 0.6 |
| a small smile (+1) or a pain tick (−1) | 1, plus any forecast of more |
| a big smile (+2), or pain still forecast to continue | 2, the cap |
| a cue that reliably predicts a big smile 5 ticks later, at ρ 0.5 | 0.5 × 2 × 0.9375⁴ ≈ 0.77 |

**The tag reaches back** (behavioural tagging; this is defect #6's fix): the tag a moment carries for memory is tag*_t = max over k = 0..63 of γ^k × tag_(t+k). So a reach that began up to 64 ticks before the smile it earned shares the smile's weight: 0.52 of it after 10 ticks, 0.28 after 20. That covers the 7–25-tick delay of risk 1.

**Where the tag goes:**
1. **The fast memory's writes** (R7):
   - the write gate tests surprise × (1 + tag) against its own running 0.9 quantile, so the write rate stays at the gate's share and tagged frames win places;
   - a write's strength is surprise × (1 + |dopamine|) × (1 + tag), with the tag at the write being this tick's received part plus the forecast made the tick before;
   - later boosts: for 64 ticks after a write, each larger γ^Δ × tag adds s₀ × (its increase in (1 + γ^Δ tag)) to the slot, through the store's own saturating merge. The boost follows `last_remap`; a dropped slot gets nothing. The factor never passes 3.
   - Dopamine's |δ| (the surprise of reward) and the tag (the size of the stakes) are different things, and both are kept.
2. **The night's replay priority** (R8):
   - an episode's entry is [its mean surprise × (1 + |δ|)] × (1 + T_e), over a running mean of 64 episodes (saved and bias-corrected: defect 7's fix). T_e is the largest tag* in the episode, reaching up to 64 ticks past its end. Entries fade 0.9 a night.
   - **The tagged come first:** every episode of the day with T_e ≥ 1 is dreamt once before the weighted draw, highest first, taking at most half the night's dreams. Without this, a strong event with about 1.2 expected draws is missed about 30% of nights.
   - The window dreamt ends at the episode's peak tag, so cause and outcome are both inside. An episode whose tag never reaches 0.1 is dreamt to its end.
   - **What the replay teaches.** The draw is by arousal, so falls are replayed as often as smiles. The cortex's forecasts, the forward half and `act_inv` learn from every dreamt position at weight 1. `act_pred`'s lesson at position t is weighted clip(1 + G_t, 0, 1), where G_t is the replayed dopamine's credit over the next 64 ticks. So acts followed by net harm are not taught as acts to make. (Weighting `act_pred` by the tag, as first written, would have taught the acts that led to a fall.)
3. **Orienting:** each born orienting trigger's bias (the face template in the periphery; a sound onset's side; acting on the gaze and the waist) is multiplied each tick by g = clip(1 + N, −0.5, 2).
   - A context that predicts one reward unit of good doubles the born pull. One that predicts 1.5 units of bad turns it into a weak turn away, as the owner's "toward or away" asks.
   - At birth ρ = 0, so g = 1 exactly. The daily report watches how often the child looks at a parent whose face has predicted frowns.
4. **Approach and avoid on the limbs** (switch `amyg_pav`, built and off at birth): each motor gate's logit gets z += 1.0 × clip(N, −2, 2), a Go bias toward good and a freeze toward bad (Guitart-Masip et al. 2012).
   - Off at birth, because the three-factor gates already learn from the same outcomes; because a born Pavlovian bias is known to impair learning to hold still for reward and to act to avoid harm; because infants' fear-driven inhibition comes after mobility (about 7–9 months); and because it would add to risk 5 before anything is measured.
   - Tried on a copy once an aversive head's ρ reaches 0.2, and adopted at a boundary only if pain ticks fall while no gate's open rate and no M2–M4 ruler falls to its chance.

**Nothing else takes the tag or the forecast:** not reward, dopamine, mood, stress, the gates' credit, the actor, working memory's latch, the cortex's loss, REM's scorer or anything the parent sees.

**The critics and the amygdala: no duplication.**

| organ | reads | forecasts | memory | used by |
|---|---|---|---|---|
| fast critic | the striatal expansion of the last 8 events (and the event lines), plus working memory | the discounted return of the summed reward at 0.9375 (TD, bootstrapped) | 36,000 | dopamine: the gates' credit, the actors, mood, stress, the store's (1 + \|δ\|), working memory's latch |
| ventral critic | the tonic traces and the clock | the return at 1 − 1/1024 | 36,000 | the gates' credit, at its earned reliability |
| face organ | the striatal expansion | the next tick's summed felt reward | 36,000 | REM's scorer (the custom child's screen face is gone) |
| **amygdala** | **the scene now: the stream and the born event lines** | **each source's good and bad separately, from the outcomes themselves** | **4,096** | **the tag (store writes, the night's draw) and the orienting gain** |

The critics learn from what happened what it is worth; they own dopamine, the only teaching signal for action. The amygdala learns quickly from what is here now what is coming and how much it matters; it owns salience: what is kept, what is replayed, and where the eyes go.

**Measured on a synthetic stream** (an instrument, not the body; `$S/g1/amyg/`). The stream stands in for the cortex: a 512-dimensional AR(1) process, 12 sparse event lines with background events, and a pairing every about 400 ticks with the outcome 5 ticks after the cue. Values are the forecast at the cue as a share of the discounted outcome (0.9375⁴), after 1, 3 and 9 pairings, at τ_a 4,096 and the critics' prior:

| cue | after 1 | after 3 | after 9 | note |
|---|---|---|---|---|
| a born event line alone (low road) | 0.30 | 0.40 | 0.52 | a never-paired line stays within ±0.12 |
| the stream moved 11 sd along a fixed direction for 3 ticks (high road) | 0.06 | 0.18 | 0.39 | at the critics' memory of 36,000: 0.02, 0.06, 0.21 |
| both together | 0.33 | 0.44 | 0.60 | at 36,000: 0.21, 0.28, 0.45 |
| an event line at the old memory of about 500 ticks | 0.17 | 0.31 | 0.35 | dips between pairings; falls to 0 within 5 unpaired |
| extinction: 10 pairings, then 10 unpaired | peak 0.49 | — | 0.17 after 10 unpaired | partial and gradual, as in animals |

- False alarms on background ticks: RMS 4.4% of the outcome, 99th percentile 12.6%. Reliability weighting keeps them out of the tag.
- A stronger prior for the stream (1.0 instead of 0.3) raised one-pairing learning of an event line to 0.41 but slowed stream cues. It is not adopted; it stays a later copy measurement.
- **Speed depends on the cue.** A cue with a born event line (pain, a touch, a face in the fovea) is learned in one pairing. A cue the cortex must represent (a voice's tone, the parent turning) took about 6–9 pairings to reach half its one-trial level. The real rate is measured in the first life days (C11).

**Tests on a tiny body** (`body/tests/test_amygdala.py`; a tiny anatomy of d 32 with a words channel, a 4-number cue channel, event lines "bump" and "chime", one 2-joint effector with an orienting trigger on "chime", reward sources face ± and pain −, and a scripted world of 2,000 ticks):
1. Inert for language: the eight digests equal; no `amyg*` attribute or save key; the same parameter count and state-dict keys.
2. The law, exact: the backward evidence equals Σ x_t y_t from the forward targets, and the solve equals numpy's ridge answer, each to 1e-9.
3. One pairing: "bump" then pain 5 ticks later; after one pairing, the pain head at a new "bump" is at least 0.2 of 0.9375⁴, and "chime" stays within ±0.1.
4. Split valence: a cue followed by a smile and then pain gives both heads above zero, |N| < A⁺ + A⁻, and a tag above an unpaired cue's.
5. Tag bounds: in [0, 2]; received-only while ρ = 0; 0 with nothing forecast or felt.
6. The later boost, exact, through the saturation law; it follows `last_remap`, and a dropped slot gets nothing.
7. The night's entry and the tagged-first draw, within 3σ of the binomial expectation over 10,000 seeded draws.
8. `act_pred` at night: a window followed by net harm (G < −1) gives exactly zero gradient on `act_pred`; the other losses are unchanged.
9. Orienting: g = 1 exactly at ρ 0, and follows clip(1 + N, −0.5, 2) with a forced reliable forecast; the gate's draw is identical.
10. `amyg_pav`: off, logits unchanged; on, z gains β × N.
11. Save round trip: every tensor and moment restored; the continuation's digest unchanged.
12. The `sim` profile digest is pinned with the amygdala on, and equal across two runs.
13. Cost at the sim's width under 0.5 ms a tick.

**Its rulers in life**, each day (C11): each head's ρ; the tag's distribution; the share of the night's dreams on episodes that carried an outcome, against the same tape drawn by surprise alone (offline, from the saved tape); the reliability over the first 4,096 ticks after a night against the evening's; the share of ticks looking at the parent's face after days with frowns. Its never-taught test (valence by prosody) is M1's, in section 12.

## 8. The refactor: where it stands, and the steps left

**Done on `sim-core`** (worktree `$S/wt_sim`), every language digest held at every commit:

| step | commit | what |
|---|---|---|
| R1 | d05a24f | the guard pinned, the anatomy declared |
| R2 | 96f72a9 | the anatomy replaces `tok` |
| R3 | cba7179 | reward as the anatomy's ordered sources |
| R4 | e948e7b | channels in the core: window per channel, the ordered input sum, a forecast head per channel |
| R5 | 0562112 | effectors: the voice as effector 0, per-joint readouts, striatal blocks appended, fixes #4, #5, #8 as switches |
| R9 | 2489e89 | the world loop: World, Frame, DiaryWorld with the pace log, the SimWorld interface, night pauses the world |
| R6 | 19a8832 | the motor timing part (act_pred, the forward half, its correction, act_inv with its reliability); learned stops with `chunk_gate` per effector; reflex hooks (a forced act with no eligibility); 118/118 organ tests and 27/27 anatomy tests |
| R6 fix 1 | fdeb605 | `act_pred`'s lesson weighs a rest's label by `act_inv`'s reliability absolutely |
| R6 fix 2 | 974d865 | `act_inv`'s reliability is Cohen's kappa per joint over its running confusion |
| R5b | 0103f0a | the striatum holds a later effector's events by joint, not by flat act |
| R6 fix 3 | e48b284 | `act_pred`'s plasticity gated by its labels' reliability (`GatedAdam`) |
| verifiers' findings | 6d6d246 | a striatum saved before R5b keeps all but its effectors' rows; `act_inv`'s kappa under shifting act rates measured, its correction left for R6h |

**Left:**

| step | work | days | risk to the digests |
|---|---|---|---|
| R6h | movement units (the persistence margin), with the born unit lengths measured under the continuation draw (C38); the kappa correction from 6d6d246; fatigue per effector; each limb's forward error into its gate; `act_inv`'s lessons batched every 8 ticks; born encoders `encs.<name>` from the body's seed; an effector's declared consequence sense (the voice's is the ears); the gaze effector with no joints (the window's state); the orienting and VOR hooks on the gaze and the waist | 2.5 | none (off for language) |
| R7 | R7a: the anatomy's event lines (`Anatomy.events`, a frame field read by the striatal expansion and the amygdala). R7b: surprise-gated writes at frames; marks at event ends; each tick's record (surprise, δ, tag, net received reward: 16 B a tick). R7c: fix #1 as a switch; fix #6 as the switch `tag_trace`; each channel's forecast error scaled by its own running mean; pace on the partner channel. R7d: the amygdala (switch `amyg`), built last in `Organs`, after `_learn_values` and `_own_face` and before `_choose`; its tests. R7e: the orienting gain; `amyg_pav` built and off. | 3 | medium |
| R8 | the night over frames: per-channel batches from stored codes; the episodes' entries (T_e from tag* over the tape at nightfall); the tagged dreamt first; the window at the peak; every effector's acts replayed as efference copies and `act_pred` targets, weighted by the replayed dopamine's credit; `act_inv` and the forward half replayed over the day's transitions; REM on frames (REM stays on); the entries' running mean saved; a tape of 6 KB a tick | 3 | medium |
| — | the full guard, with `--roundtrip`; the `sim` profile pinned | 1 | — |

That is about 9.5 working days of core work.

**The guard, unchanged.** Every refactor commit leaves all eight pinned language digests equal. They are in `tools/pins/digests.txt`, run from the main tree's directory and importing the branch's body:

| profile | threaded | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` |
|---|---|---|
| default | `7c54199e3c72cf77d45a8e94` | `fbe9e6b330f44e7424f113f3` |
| served | `fc81c03dec606afb62c3fc35` | `f507fb9c6c55649ddd39f917` |
| switches | `ab1ab46751b11b314dfc78c2` | `14de2b131ce4b2874ca0ad95` |
| served `--full --roundtrip` | `c4730b232091fe229ee286dc` | `823431e80a08f197ee086884` |

- **New code is inert for language.** The language anatomy builds no new module, draws no new random number, runs no new branch, and adds no attribute, window field or save key. The amygdala and `tag_trace` are absent from its settings.
- **New modules come last,** built from generators of their own seeded by the body's seed.
- **After the plumbing run,** a `sim` profile joins the check. It runs a tiny body on the SimAnatomy, fed a recorded 200-tick script of raw observations (about 4 MB in `tools/pins/`), so the guard covers the core without the renderer. SimWorld's own exact-replay test covers the physics.
- **Fix #6 for the language body** (`tag_trace` with the felt entry of an utterance) stays off unless measured on a copy; the review measured it only together with `chunk_gate`.

**The language body.** It is paused. It moves to the refactored code whenever it restarts, after:
- the eight digests under its live constants;
- a 400-tick read-only load check (no save path to `data/watch2.pt`);
- a save round trip from new to old.

Nothing in the sim waits for it.

## 9. Compute, memory and disk on this Mac

M4 MacBook Air (fanless): 10 cores, 16 GB RAM. One torch thread until a quiet window measures more. The body runs at `nice 10`.

**Wall time per tick** (measured under load from other sessions; conservative):

| part | ms | source |
|---|---|---|
| physics: 75 steps of the G1 under babble, about 28 contacts | 12.3–13.8 | measured |
| the parent's mocap and its Python, every step | 4.3 | measured |
| the parent's plan (solved per act and once a second, warm-started, interpolated) | 1–3 after W2 | estimate |
| two eyes, 168 × 96 each, the sun's shadow: render, read-back, the split into periphery and fovea | 37.6 (16.5 without shadows) | measured |
| ears: three sources, both cochleas, the brainstem | 2.3 | measured |
| the child's tract: the babble average, plus its sound to both ears | 1.6 + 0.2 | measured |
| the parent's ear: 287 ms per child utterance, at the born babble's 6.4–7.8 utterances a minute | about 5 on average (4.6–5.6; 287 ms at an utterance's end) | measured per utterance; the rate from the born babble |
| the parent's voice | 0 on a cache hit | measured |
| the core at the humanoid's size, at the critics' current solve rate | mean 56–84, median 39–53 | measured (paired against the language core: 84.4 against 74.3 mean) |
| the critics' solves every 256 ticks instead of 64 | −7 to −20 | measured parts |
| the amygdala | 0.19 | measured (synthetic) |
| not yet measured: per-step contact forces for touch and the pain filter; the face test's rays; the fovea's face template; the parent's conduct layers L1–L2 and feelings each tick; the room camera while watched | about 2–10 | estimate |
| **total** | **about 100–155 mean with the sun's shadow; about 80–135 without** | from real time to 1.5× faster |

- **Against this morning's design** (80–125 ms): the G1's physics adds 4–5 ms, its eyes 8–13 ms with the sun's shadow (render and read-back: 32.5 ms against the custom child's 24.2 ms in the same run), the tract about 2 ms, and the parent's ear about 2–5 ms.
- **Half the eyes' cost is the G1's own visual meshes** (629,000 triangles). Hiding them (a test only) gave 8.3 ms without shadows and 16.7 ms with the sun's. A lighter visual copy of the meshes, used only by the eyes, would save about 8–18 ms, but it changes how the G1 looks to itself (B3).
- **A twice-sharp fovea** (336 × 192 per eye) costs 16.9 ms without shadows and 38.3 ms with the sun's: resolution barely matters; the fixed overhead does.
- **Levers if the tick is too slow,** in order: the parent's ear scoring only the expected words and the bank (an estimate of −1 to −3 ms; the world's side, ours); then the owner's two render choices of B3, the lighter mesh copy (−8 to −18 ms) and the eyes without the sun's shadow (−21 ms). The shadow and the mesh are the owner's (decision 3), never ours to drop for speed; a slower life is the alternative, since lockstep makes the tick cost wall time only.
- **A life day:** 24,000 ticks take about 40–62 wall minutes awake. The night is estimated at 12–25 minutes (R8 is not built). So a life day with its night takes about 55–90 wall minutes.
- **Where the core's time goes.** The critics take about half the core's tick; the cortex's full window is recomputed every tick. The core's additions for a humanoid were profiled at about 10–12 ms a tick:

| part | ms a tick |
|---|---|
| `act_inv` and forward lessons, unbatched | 4–6 |
| nine more effector choices | 3.5 |
| encoders over the window | 2.7 |
| motor striatal lines | 2.2 |
| `act_pred`'s lesson (27–40 ms every 24 ticks) | 1.1–1.7 |
| nine more gate lessons | 0.3 |

- **The page's room camera** (1920 × 1080, 51 ms with both lights' shadows) is rendered only while someone watches, at most every 4th tick. Milestone films are redrawn afterwards from saved states, never recorded live.
- **Threads:** 4 against 1 was inconclusive under load (52–130 against 81–119 ms). This is decided at S5, in a quiet window.

**Memory:**
- The core is about 2.5 GB with a full store.
- MuJoCo and the renderers take about 0.3 GB with the G1's meshes; the synth server about 50 MB; the parent's ear bank a few MB.
- The language body is paused, so RAM is not a constraint.

**Store capacity.**
- Writes: about 2,500 heard words a day, plus about 2,400 surprise writes, so about 4,900 a life day before merges. The amygdala changes which frames win the gate, not the write rate.
- The twenty-ninth defect's rule (capacity for about 19 nights of writes) then asks for about 98,304 slots, against the served 65,536.
- S5b measures the real rate, and the cap is set before birth. Each 32,768 slots cost 134 MB of RAM and of every save.

**Disk** (12 GB free at 03:25 on 2026-09-24, 0 at 03:57, all taken by the refactor's verification copies; 20 GB after they were cleared; A18):

| item | size | kept |
|---|---|---|
| the save, written as `.tmp` then renamed | about 1.3–1.7 GB (up to 3.4 GB at peak with its `.tmp`); the amygdala adds 2.3 MB. The earlier 0.95 GB had no source and is below the language body's own save (1.28 GB on disk today), while the sim's core has more parameters (35.2M against 28.8M), a larger store (98,304 slots against 65,536: +134 MB) and the day's tape (141 MB). S5b measures it | one |
| the day's tape (about 3,000 numbers a tick, fp16: 6 KB) | about 141 MB a life day, inside the episodes (cap 24,000 ticks) | in the save |
| each tick's record for the amygdala (16 B) | 384 KB a life day | in the tape |
| world states for exact replay and redrawn films | about 7.6 KB a tick, 180 MB a life day | the last 3 days, 0.55 GB |
| the voice cache | at most 300 MB | always |
| the parent's JSONL log | rotated at 100 MB | the last file |
| **peak** | **about 4.4 GB** | |

- The tract's sound is not saved: the tract is deterministic from its seed and the acts, so a replay regenerates it.
- **The rule.** The sim writes nothing that would leave less free space than the larger of the two bodies' saves (the sim's, about 1.7 GB until S5b measures it) plus 1 GB: about 2.7 GB. It skips the write, logs why, and pauses at its next boundary.
- **Birth needs at least 8 GB free:** the peak of 4.4 GB plus the rule's floor of 2.7 GB, and a margin.
- **The owner's folders** `data/backups` and `data/first_lineage` are never touched.

## 10. The disclosed constants

| constant | value | kind | set by |
|---|---|---|---|
| tick; physics step | 150 ms; 2 ms (75 a tick), `implicitfast`, the elliptic friction cone, multi-point collision detection off, auto-reset disabled | clock, world | ours |
| life day | 24,000 ticks | physiology | ours (today's) |
| the child | the stock Unitree G1 with Dex3 hands (Menagerie `g1_with_hands.xml`, unchanged): 1.32 m, 34.39 kg | world | owner (decision 7) |
| torque limits and joint ranges | the model's own (section 3.2) | anatomy | the robot's |
| servo gains | set at load, never in the file: the limit reached at 0.25 rad of error (the Dex3's joints at 0.1 rad); damping 0.04 s × kp | anatomy | ours |
| servo law | an act: target = measured angle + step; at rest the target relaxes to the measured angle, time constant 3 ticks; no gravity compensation | innate | ours |
| step sizes | waist, arms, hands, legs ±0.09 / ±0.27 rad a tick; gaze ±4° / ±11.5°, vergence ±1.7° / ±5.7°; tract −0.6, −0.2, 0, +0.2, +0.6 of each articulator's range | anatomy | ours |
| persistence margin (movement units) | log 4 in logits | physiology | ours |
| `chunk_max` | 8, a ceiling only | physiology | ours (the served value) |
| F_pain | 3 × the child's weight from the model file (about 1,012 N), per link | innate | ours |
| pain force | per link, the tick's largest 10 ms mean (5 physics steps) of the summed normal force (A12) | innate | ours |
| withdrawal | the hurt limb takes one big flexion step a tick for 2 ticks; the waist and torso have none; each joint's flexion direction fixed in W1 | innate | ours |
| grasp reflex | a palm touch above 0.3 N closes the Dex3's 7 joints one step a tick, unless the hand's own act opens it | innate | ours |
| orienting | a born bias on the gaze's and the waist yaw's proposals toward a three-blob face template and the ears' lateral read; a born gate input; its gain from the amygdala | innate | ours |
| VOR | the fovea window counter-shifts by the torso gyro's rotation in each camera's frame, gain 1; its quick phase at the window's reach, a jump back of half the reach in the direction of the turn (A23) | innate | ours |
| weakness when empty | torque limits × (0.3 + 0.7h) | anatomy | ours |
| eyes | a colour stereo pair at the D435's imagers (50 mm apart), 88.3° × 58° (horizontal × vertical), 168 × 96 px each, pitched 47.6° down as on the real G1; periphery 3 × 3 averaged (56 × 32 px); fovea a 32 px window (about 21°) whose centre reaches ±38° × ±20°; vergence 0–12°; born centred and parallel; the window holds where it is left; RGB only; the sun's shadow kept | anatomy | ours (colour stereo: B4; the shadow and a lighter mesh copy: B3) |
| the IMUs' noise | the model file's own declared noise (gyro 5e-4, accelerometer 1e-2), added by the world from its own seeded stream, since MuJoCo 3.9 does not apply it | world | the robot's (its file) |
| the event line "a face in the fovea" | the born three-blob face template on each fovea's pixels (either eye), never the world's face test | innate | ours |
| contacts with the world | the world's geoms at contact priority 2, so their friction and softness decide contacts with the G1; the G1's file untouched (A21, C26) | world | ours |
| born codes | retinotopic: periphery 7 × 4 cells × 6 colour mixes per eye, fovea 8 × 8 × 6; ears 40-band gammatone, ERB-spaced 80–7,600 Hz, cube-root, delay lines ±8 samples in 21 bands, head radius 0.078 m; random projections from the seed | anatomy | ours |
| ears' places | two sites on the head's sides, 15.8 cm apart | anatomy | owner (default, B5) |
| the tract | 12 cm, 24 sections plus a nasal branch; pitch 180–546 Hz, resting about 265 Hz; breath 400 cm³, refilled in 0.8 s; rest relax 1 tick; its calibration (section 4.9) | anatomy | ours (identity: B6) |
| face reward | the increment rule: rises of the positive part felt positive, rises of the negative part felt negative, falls never felt, clipped ±2; felt only via the born reading while the face test passes on 2 consecutive ticks | reward | ours |
| born reading | 2 × (smile − frown), the mouth corners only; its last value for 30 ticks out of view, then neutral (A2) | innate | ours |
| face test | the mouth point in an eye's software fovea and reached by an unblocked ray; the face turned within 75°; its front (0.17 × 0.21 m) at least 20 fovea px × the cosine of the turn; either eye (A1) | innate | ours |
| the parent's feelings (FEEL) | joy pulses of worth ÷ 2 (rise 2, held until seen at most 20, then 10, ease 5); displeasure 0.5 / 0.25, held 10; surprise τ 3, × 0.5 per repeat within 200; concern 1.0 / 0.6, τ 40, blocks smiles above 0.5; attention τ 5, the flash after 200 ticks unseen; mood τ 4,000 | teacher method | ours |
| the face's start rule | a smile waits for 2 neutral ticks if another is on her face or a positive face was seen under 31 ticks ago; dropped after 40 | teacher method | ours |
| the worth table | section 4.3; mastery 1 + e^(−n/10), floor 1 | teacher method | ours |
| the parent's face placement | never closer than 25 cm, arriving at least 15° off the fovea's line (A3) | teacher method | ours |
| the parent's force caps | one hand 100 N sustained, 150 N for up to 2 s; both hands 160 N, 200 N for up to 2 s; every hold a capped spring | teacher method | ours (checked against sources in W2) |
| the guide's cap | min(1.5 × the limb's own push at that pose, 100 N); at most 8 ticks at up to 0.3 m/s; stops after 2 ticks at the cap (A10) | teacher method | ours |
| the parent's soft contact and yield | solref 0.02 on its collision shapes (MuJoCo's default), 0.006 on its hands' (A25b); a contact over the act's cap for 2 steps stops that segment and backs it off 2 cm a tick (A4) | teacher method | ours |
| the parent's body (A25b) | 16 dynamic segments at de Leva's female masses for 62 kg; one actuator per joint axis, its ranges her strength (the table in A25b), damping inside; her trunk carried at her plan by a support capped at 695.5 N (her weight ⊕ the child's steady push; never down) and 257.4 N·m (two hips' extension), 20,000 N/m and 2,000 N·m/rad; her spine and neck stiff (5°), her limbs 20°, resting legs 60°; her pose raised off the floor; her standoff 3 cm from the child's body and legs each tick; the calm step (0.3 m, 2 cm a tick, 3 s); kneeling at most 0.5 m/s at the pelvis | world (her body) | Harbo 2012, de Leva 1996, Garces 2002, Nordin 1987, the G1's file; the rest ours |
| the parent's own pain | 150 N (a 10 ms mean) on any segment | teacher method | ours |
| what her body bears at a contact (A25b) | her hands 140 N a tick / 280 N a 10 ms mean, her forearms 160 / 320 N; pressed in at most her region's compression at its bound (hands 3.7 mm, forearms 8.0, upper arms 10.0, trunk 11.2, legs 8.8, head 0.87) | tests | ISO/TS 15066:2016 Tables A.2 and A.3 |
| the prop's easing and catch | the pull-to-sit capped at the caps; the prop within 30° of vertical, easing 100, 70, 40, 20% then hovering 5 cm off, one step down per 10 ticks within 20° of vertical; the catch 2 ticks after the trunk passes 35° or the head drops faster than 0.5 m/s (A9); the 300 ms reaction is a person's and never changes (C6) | teacher method | ours |
| a peekaboo answer | an act begun within 10 ticks of the reveal by an effector that rested the 5 ticks before (A2) | teacher method | ours |
| the parent's paths | a 5 cm floor grid; clearances: furniture 0.15 m, the child 0.25 m (outside its leg sweep), toys 0.08 m; walking 0.8 m/s, shuffling on the knees 0.25 m/s (A6) | teacher method | ours |
| the parent's timings, judgments and conduct | sections 4.3–4.10, in `body/sim/lang/consts.py` and the parent's constants file | teacher method | ours |
| the parent's ear | the cochlea, a shift of 0–4 bands, c1–c12, Itakura DTW; a template bank of many voices; accept when nearest of the expected words and d − d(babble bank) < −m, m = −0.35, re-set per context size; exact when also nearest of all its words; the bank, the templates and the expected sets fixed before birth, the child's own productions never added (A27) | teacher method | ours |
| the tidy | a toy out of the child's reach and untouched for 2,000 ticks is put back (A5) | teacher method | ours |
| the amygdala | on for the sim, absent for the language body (switch `amyg`); input: the stream C/√d, the anatomy's event lines, a level; heads: one per reward source and sign (sim: face ±, pain −, charge ±) | physiology, anatomy | ours |
| its law | least squares with forgetting, the backward target identity; ridge 0.3 × τ_a × running variance, the level free; τ_a 4,096 ticks; solved every 8 ticks; horizon 0.9375 a tick, not cut off | physiology | ours (the critics') |
| its reliability | correlation with the realized target, finalized 64 ticks later, moments over 36,000 ticks, clipped 0–1, zero until 64 pairs | physiology | ours (the face organ's) |
| its tag and uses | min(2, A⁺ + A⁻ + received); reaching back 64 ticks at 0.9375; the store's gate surprise × (1 + tag) over its running 0.9 quantile, strength × (1 + tag) with later boosts, at most × 3; the night's entry × (1 + T_e) over a saved running mean of 64 episodes; T_e ≥ 1 dreamt first, at most half the night; the window ending at the peak tag; `act_pred` at night × clip(1 + G, 0, 1); orienting gain clip(1 + N, −0.5, 2) | physiology | ours |
| `amyg_pav` | 1 logit per reward unit of N, clipped ±2; off at birth | innate | ours |
| `tag_trace` (fix #6) | the received tag reaching back onto the language body's felt entry; defect 7 fixed with it; off | physiology (switch) | ours |
| bedtime feed | a feed first if h < 0.6 when goodnight begins; every hold released at tick 23,700 (A17) | teacher method | ours |
| pain reward | −1 per tick over F_pain | reward | ours |
| charge reward | 4 × [D(h_t) − D(h_t+1)], D(h) = (1 − h)² | reward | ours |
| reward order | face, pain, charge | reward | ours |
| charge drain; bottle | 4e-5 + 2e-3 × Σ τ² / Σ τ²_max a tick; +0.01 a tick while the bottle touches a palm | world | owner (default, B10) |
| fatigue | per effector; the tract 0.12 a sounding tick; limbs 0.12 × Σ τ² / Σ τ²_max per tick of motion; gaze 0.03 a step | physiology | ours |
| motor timing lessons | `act_inv` batched every 8 ticks; the MOTOR constants (`act_inv_lr` 1e-3, `act_inv_tau` 8192) | physiology | ours (R6) |
| critics' solves | every 256 ticks (sim), 64 (language) | physiology | ours |
| ventral critic's bands | fast ladder bands 0–2 (sim) | physiology | ours |
| forecast heads | one per channel; vector heads by squared error to the next born code, each error scaled by its own running mean | physiology | ours |
| event end | today's settle law (`offset_fast` 4, `offset_slow` 64, `offset_settle` 0.5) on the summed forecast error | physiology | ours (today's) |
| store | capacity set before birth to hold 19 life days of the measured writes | physiology | ours |
| episode cap | 24,000 ticks | physiology | ours |
| switches at birth | fixes #1, #4, #5, #8 on; `amyg` on; `chunk_gate` 1 per effector | physiology | ours (off for language) |
| voices | parent: compact Samantha, rate 0.25, pitch 1.15, and the registers; the child: its tract | world, anatomy | owner (default) / ours |
| light | the sun across the window from morning to dusk; the lamp at winding down; dark at night | world | owner (decision 3) |
| the room, toys, charger, names | sections 5 and 4.8 | world | owner (defaults) |

Every language-body constant keeps its value and place.

## 11. The build plan, day by day to birth

Three sessions in parallel. Each writes tests at `nice -n 19`, small and short, and never touches ports 8020, 8021 or 9333. One verification copy at a time across all sessions, deleted when its run ends (A18).
- **Core** works in the `sim-core` worktree.
- **World** works in `body/sim/`, `tools/sim_*` and the `/sim` page.
- **Parent** works in `body/sim/lang/`, `body/sim/voice/`, `body/sim/ears.py`, `body/sim/parent_*.py` and `body/sim/tract.py`.

| day | core (session 1) | world and the G1 (session 2) | parent, voice and ears (session 3) |
|---|---|---|---|
| 1 | R6h: movement units, and the born unit lengths under the continuation draw (C38) | W1: SimWorld from `g1scene.py`: the servo law set at load; weakness; withdrawal and grasp on the Dex3; touch and pain per link with the 10 ms filter; the IMUs' declared noise; `FatalError` caught and MuJoCo's auto-reset disabled (A18); save and restore with every random stream; the exact replay test | P1: the parent's voice: the Swift synth server, the cache, SSML emphasis and the word rate measured, the determinism test |
| 2 | R6h: fatigue per effector, forward error into the gates, `act_inv` batched | W1: the sink rates under the resting law; real frictions by the world's contact priority and a friction model that does not creep. W2: capped springs for every hold on the G1 (the prototype's welds on its torso, pelvis and elbows removed); the yield rule | P2: the ears (`ears.py`): the spatializer at the G1's head radius, two cochleas, the delay lines, the born lateral read; the tract's sound through them |
| 3 | R6h: born encoders; the declared consequence sense; the gaze effector; the orienting and VOR hooks | W2: the parent's acts on the G1 under the caps: the kneel outside the leg sweep, attend, show, hand over, guide, the brief turn, the pull-to-sit with its help, the prop near upright, the catch; the motor intents | P3: the fast layer: templates, the line check, intents, variation sets; the feelings and graded face (`parent_feel.py`) in the world |
| 4 | R7a–b: event lines; surprise-gated writes, marks at event ends, the tick's record | W3: the eyes in the world: the gaze effector, the VOR, the born codes, the face test's rays in the software fovea, the face template on the fovea's pixels; the eye check on the toys and the face under morning, midday and dusk light (reported, the light unchanged); the dock beside a hand; the bottle for the Dex3; B1 and B2 applied | P3: the behaviour system's L1–L2: contingency, joint attention from the software fovea, the scaffolding ladders; the ledger |
| 5 | R7c: fixes #1 and #6 (`tag_trace`), error scaling, pace on the partner channel | W4: the born body's own motor timing on the G1 (per effector: its gate's draw at the born p_act, the continuation draw, the margin, `chunk_max`; a replica on day 5, repeated on the real core at every learning rate 0 at S5a, each run discarded after), under the final law: unit lengths, rolls by direction, travel, time off the mat, pain with the filter, thumps, the leaning sit's fall time, bottle visits, hits on the parent. Written down as chance. | P3v: the tract as effector 0 in the world; the parent's ear (`parent_ear.py`): the babble bank recorded before birth, templates from P1, m re-set for the context sizes |
| 6 | R7d–e: the amygdala, its 13 tests, the orienting gain, `amyg_pav` off | W5: sounds from physical events; the room's echo (B9); the ears in the frame; the tract's sound from the head's front; the day's light and the lamp; every channel from the scene | P4: the day plan, routines, stages, leaving and returning; the never-taught pairs held out by the line check |
| 7 | R8: night batches from stored codes; the tape and episodes | W6: the `/sim` page on port 8030: the room camera, both eyes with their fovea windows, the parent's face, charge, the ten gates, a two-voice transcript, instruments | P5: the digest, `ops/sim_parent_brief.txt`, the steering check; P6: `tools/sim_parent_rates.py` with W4's babbler |
| 8 | R8: every effector's acts replayed; the entries, the tagged first, `act_pred`'s weight from the replayed dopamine | S5a (sessions 2 and 3): the SimAnatomy (9 channels, 10 effectors, 3 reward sources) on the core as it stands; a plumbing run at every learning rate 0 | (S5a) |
| 9 | R8: `act_inv` and the forward half replayed; REM on frames; the entries' mean saved | the parent's rates with the babbler: smiles per ask, calls answered, asks met, contingency (at least 0.9) and the voice-free share, written down as chance. There is no target rate. Before birth the method changes only for a stated reason (such as an ask no body could meet from where it lies), never to reach a number. | |
| 10 | the full guard with `--roundtrip`; the `sim` profile pinned | the plumbing re-run on R7 and R8; heat checks | |
| 11 | — | S5a again on the finished core: every channel, effector and reward source through a night | |
| 12 | — | S5b: two nights at every learning rate 0 on the whole core: the mean tick, the night's length and the store's write rate written down; the store's cap set; the birth checklist | |
| 13 | — | **S6: birth (seed 1)**: day 1, its first night, day 2, the save round trip; the first hour watched at 1× | |

- **Against this morning's plan** (birth about day 11 for the custom child, with the G1 and the tract each adding days unmeasured): the G1's room, senses, eyes and the parent's acts on it, the tract and the parent's ear now exist as prototypes. What remains is the servo law on the G1, the caps and the new acts, and the core's additions (the gaze and the declared consequence sense, the born units' lengths, the amygdala's extra half day in R8). R6 fix 3 is done (e48b284).
- **Range 11–18.** The widest unknowns are the tick (the eye check and the parent's ear), R8, and whether the pull-to-sit and the catch work under the caps.
- **Two sessions:** the world then the parent in session 2 (about 15 days), with the joint days after. Birth about day 18.
- **One session:** about day 28.

**The birth checklist:**
- The eight language digests equal their pins, and the `sim` profile is pinned with the amygdala on.
- The G1's model file is byte-identical to its commit.
- The sim's replay is exact, and a save and reload continues it exactly (the world, the parent, the tract and every random stream).
- The babble baseline, the parent's rates and the parent's ear's chance acceptances are written down, with the babbler's smiles per ask as chance.
- The plumbing run shows every channel arriving, every effector acting, rewards summed in order, a night with the tagged first, and a save round trip.
- The mean tick is at most 150 ms (1× real time) over two heat-soaked hours, or the owner has chosen between B3's render choices and a slower life; MuJoCo's auto-reset is disabled and its warning counters are checked every tick (A18).
- The store's capacity holds 19 life days of the measured writes.
- At least 8 GB of disk is free (the peak of about 4.4 GB, the rule's floor of about 2.7 GB, and a margin; section 9), and the disk rule is live.
- `act_inv` starts untrained, and no weight of the body has learned from the babbler.
- Seed 1 is born from its freshly built state, never from a plumbing or timing run's state: at every learning rate 0 the store's writes, the critics' and the amygdala's least-squares evidence and the running means still accumulate.
- The born movement units' lengths are written down (C38), and the owner has answered B18 (the skin), B3 (the renders) and B1 (the cup's scale).
- The parent's force caps are in its constants file and checked against their sources; no act drives the G1 over them.
- The babble bank for the parent's ear is recorded and fixed; the never-taught pairs are listed and held out.

After birth, it lives days with a report each day, judged by sitting with it. Defects are fixed on copies and applied at boundaries, never tuned.

## 12. Milestones a viewer can watch, and the tests of understanding

Each milestone is judged by sitting with the child on `/sim` and by its own rulers, never by a baseline run. A19 gives each ruler, its chance level and when it counts as reached. These are estimates: no life has been run. Wall hours assume 55–90 minutes per life day.

| | what the viewer sees | its own rulers | life days | wall hours |
|---|---|---|---|---|
| **M0: it lives** | a day, a night and a day; the fovea windows, gates and the parent's face moving | its forecasts of its own next body sense and periphery beat "nothing changes" | birth day | the first |
| **M1: it orients** | its fovea, then its trunk, turn to the parent's voice and name, and it holds the smiling face in view | turns within 20 ticks above the born rate; the share of ticks with the parent's face in the fovea | 1–3 | about 1–5 |
| **M2: it reaches** | its hand touches a toy held over its chest | touches above the babble rate | first 5–25; reliable 20–60 | 5–38; 18–90 |
| **M3: it grasps** | it reaches, then holds one of the six holdable toys (the reflex holds from hour 0 when a toy meets the palm) | reach then hold for at least 3 ticks, above chance | 15–60 | 14–90 |
| **M4: it rolls on purpose** | it rolls toward the parent or a toy | rolls by direction against the babbler's share under the same parent placement (A19); the trunk turning first | 15–60 | 14–90 |
| **M5: it sits with help** | it helps the pull-to-sit with its own flexion, then, held near upright, keeps its trunk up and looks at the face (her face low and ahead, where its eyes reach; in the leaning sit only by raising its trunk: A22) | its share of the rise; the trunk-up share and face looks while held; less hold needed over the days | 20–80 (open) | 18–120 |
| **M6: it names** | it names what is in its fovea or hand (by the scaffold's output first) | the ledger's "says"; right names above 1 in 4 | first right names 15–40; 1 in 4 at 30–80 (open) | 14–60; 28–120 |
| **M6t: its first word from its own tract** | a word the parent's ear accepts, from the tract alone | M6's ruler on the tract's sound; the precursor: its first accepted echo | not forecast | — |
| **M7: it sits alone** | the leaning sit, hands on its knees, without the parent's hands | seated time without her hands | not forecast | — |

- Crawling, standing and walking are not forecast. The G1 cannot crawl on flat palms (its wrists), and upright sitting tips it backward (section 3.8).
- Rolls under babble are not yet measured on the G1 (W4). Rolling on purpose is what the months buy.

**The tests of understanding** (the owner's decision 8). Each milestone has a test by something the parent never taught toward. It is judged by the same body's own rate at matched moments, one-sided p < 0.01, pooled by A19's rule. Each probe is also an unpaired trial, so the parent runs at most 3 probes of a kind a life day (a parent's-method item, ours). The report describes each probe that was seen.

| milestone | the never-taught test | chance |
|---|---|---|
| M0 | its forecasts beat "nothing changes" on ticks in a light it has not lived yet (its first dusk, its first lamp-lit evening) | "nothing changes" on those ticks |
| M1 | (a) the parent's voice from a place she has never called from (the doorway, the first time, while that place is within its eyes' reach: A22, A28) brings the face into the fovea within 20 ticks; (b) valence by prosody: once approval-register lines have preceded smiles, a line in the approval register with words never said in it, with no judgment pending, brings a look at her face within 20 ticks | (a) matched moments with no voice; (b) plain-register lines of the same length at matched moments |
| M2 | a toy never shown before (from the shelf, at its first showings) is touched within 40 ticks | matched moments with that toy in the same place, not shown |
| M3 | the hand opens by its own act before touching a holdable toy it has never held, then holds it at least 3 ticks | the same at matched moments |
| M4 | it rolls toward a toy on the side the parent has never knelt on (she kneels on the face side, A6) | the babbler's share under that placement |
| M5 | held near upright, it reaches for a shown toy: reaching was learned lying down, and the pairing was never taught | matched moments while held, no toy shown |
| M6 | (a) a known word finds its toy seen in a new place or from a new angle (upside down in her hand, or in a place within its view it has never seen the toy: A28); (b) known words in a combination never heard: "where is the red ball?" (a look), and "push the red ball" (an act, once "push" is understood), where "red" and "ball" were heard only in other pairings (the colour twins, B2; the held-out pairs in A28; the line check holds them out until the test); (c) a word used to get something: it names a toy that is out of reach, and the parent has not named it in the last 40 ticks | (a) per word, its own rate with the object absent; (b) its share of looks to the named twin when she asks with the noun alone ("where is the ball?") at matched moments (A28); (c) its own rate of that name at matched moments with the toy in view and not asked for |
| M6t | a word from its own tract used in a never-paired place, or used to get the toy | its own rate with the referent absent, plus the babbler's 2% |
| M7 | it sits alone off the mat, on the oak floor, where it never sat | none: an event, in a new place |

## 13. Risks and how we see them

The numbers follow section 0.

| # | risk | what we watch |
|---|---|---|
| 1 | Credit across the delay: eligibility is 12 ticks at decay 0.8, so a reward 8 ticks after the act reaches it at 0.17, and after 11 ticks at 0.09. The amygdala's tag reaches back 64 ticks for memory and the night, never for credit. | `act_pred` against `act_inv` on held-out demonstrations, first; the value's rise at the touch; each limb's gate rate conditional on the ask (the causal limb should keep acting and the others fall back) |
| 2 | Balance at a 150 ms tick, in a stiff body: upright sitting tips back; the leaning sit holds only on locked servos; under the resting law every posture sinks. A limp upper body (16.2 kg, its centre 0.25 m above the hips) falls with a time constant of about 0.19 s, near the tick; the resting law's lag slows a sink to about 4° a second at 35° (an estimate from the hips' gains), so the danger is the child's own big steps | the sink rates (W1); the leaning sit's fall time; falls and strikes per hour. If sitting has not come by M5's range, a faster learned motor loop below the tick is designed: a change of body, so a new body with its own seed (A20) |
| 3 | Motor milestones by chance only; rolls not yet measured on the G1; the born units average about 1.4 ticks, far shorter than the babbler that justified them (section 3.6) | the born units' lengths (C38); rolls per life day by direction (chance: the born loop's share under the same placement); the trunk-first order; face-down bout lengths and the chest-up share; the parent's turns a day |
| 4 | The parent's strength: a person cannot sit up, slide or lift a 34 kg body; the catch's energy estimate has no source | the pull-to-sit's share carried by the child; holds at their caps; the catch (C6); acts refused for force, logged |
| 5 | Thrashing, learned helplessness, leaving the mat with no one to carry it back | per effector: gate open rate, Σ τ² / max, step reversals a second, thumps a minute; open rates after painful days; long still bouts; low charge with falling torque use; time off the mat and where it ends up |
| 6 | The critics' cost | the mean tick and its spikes; the solve every 256 ticks |
| 7 | Solver stability with the Dex3 hands | MuJoCo's warning counters every tick; `FatalError` caught; the life pauses on any reset or NaN (A18) |
| 8 | Phantom pain: babble's single-step peaks (944–1,799 N) near F_pain (1,012 N) | pain on still ticks and under babble, with and without the 10 ms filter (C5) |
| 9 | Vision: 1.5 px a degree in the fovea; on its back it sees mostly the ceiling and its own body, and the parent only when she leans over its chest or stands toward its feet; face down only the mat; sitting or standing only what is low and ahead (A22); the light changes through the day | the eye check at W3 (fovea identity at least 0.75 on the ten toys and the face, under each light); the fovea's forecast error on shown toys |
| 10 | Vocal imitation near chance at birth (2 of 12 echoes accepted); tokens are the easier road to reward | the first accepted echo; the tract's share of "says"; the copy tests before removing either scaffold (section 4.9) |
| 11 | Disk and heat | the disk rule; one copy at a time; the mean tick over long runs |
| 12 | The tick over the old 125 ms target | the tick heat-soaked (C12); the levers in section 9 |
| 13 | Smiles drifting into shaping; the parent talking too much | the neutral resting face; the one law; the born reading's hold; `tools/sim_parent_rates.py` (at least 40% of play ticks free of her voice) |
| 14 | The charge need is weak (drive reduction nets zero over a cycle), and feeds are frequent under babble (about 5–6 a day) | feeds the child starts itself; the bottle's stages |
| 15 | The amygdala's "away" gain could teach it not to look at a parent whose face predicts frowns | the share of ticks on her face after days with frowns (7.4) |
| 16 | Four toys cannot be held | the grasp rate on the six holdable toys; B1 |
| 17 | Sleep is a pause by tick count, not a robot's sleep | accepted for the first body |
| 18 | Sim choices that differ from the real robot: colour stereo, the microphones' places, the speaker's place, the servo gains, touch on every link (a skin the robot lacks) and pain read from it | listed in B4, B5 and B18 and section 3; checked before the robot. On the real G1 the skin's place would be taken by an added skin or by contact estimated from the joints' torques, each a change of body |

## 14. The owner's decisions

**The nine decisions of the G1 amendment, and where each lands:**

| # | the owner's decision | where |
|---|---|---|
| 1 | The parent's face from its own feelings, graded and continuous | 4.3 (feelings, face, the one law); 6 (the increment rule) |
| 2 | The child's amygdala, a named organ that tags moments | 7.4; R7 and R8 in 8; its constants in 10 |
| 3 | A complete reality sim: sounds only from real events, day and night light, no shortcuts | 5.2, 5.4; the eye check under each light (C3) |
| 4 | Two eyes merged as in humans: a stereo pair, fusion learned, no depth channel | 3.4 |
| 5 | Unique sounds: an articulatory vocal tract as the voice; the word and letter channel a scaffold removed later | 4.9; M6t in 12 |
| 6 | A really good sim parent: contingent within about a second, joint attention, infant-directed speech, scaffolding, emotions, routines | 4.3–4.10 |
| 7 | The child is the stock G1, unchanged, with no face; the human-shaped parent has the face | 3; 15 |
| 8 | The goal: human brain architecture that gains understanding, tested at every milestone by something never taught | 1; 12 |
| 9 | The G1's facts: no neck; senses where the real G1's are; a movable fovea in software | 3.1, 3.4, 3.5 |

**Still defaulted, each with a recommendation** (the decision log's part B asks each as a plain question):
- the four toys a Dex3 hand cannot hold, and the cup already scaled to 0.8 so it can be held (B1);
- the colour twins for the never-taught word test (B2);
- the eyes' two render shortcuts: a lighter visual copy of the G1's meshes, and the sun's shadow (B3);
- colour stereo, the microphones' and the speaker's places (B4, B5);
- the child's voice identity (B6);
- the play space, now that no one can carry the child back (B7);
- the sofa's gap, the table's under-shelf and the morning basket (B8);
- the room's echo (B9);
- the charger (B10); the mat's size (B11);
- looking at the parent never raising her smile (B12);
- names, the parent's voice, no second adult (B13);
- how the parent reads the child's gaze (B14);
- the room's size for a 1.32 m child (B15);
- the parent's strength: a person's (B16);
- the word tokens, if the child's ears and voice never pass the tests for removing them (B17);
- touch on every link, a skin the real G1 lacks, and pain read from it (B18);
- a life day of one simulated hour, so the sun crosses the window in an hour (B19);
- the G1's own motor hum (B20).

**Ours, decided here and disclosed:**
- the servo law, its gains set at load, and the step sizes;
- the persistence margin;
- F_pain's multiple;
- the kept and refused reflexes, the VOR on the software fovea among them, with its quick phase from birth, decided on biology rather than by a measured rate (A23, C33);
- the event line "a face in the fovea" from the born template on the fovea's pixels (section 3.4);
- the IMUs' declared noise added by the world; the world's contact priority; MuJoCo's auto-reset disabled (A18, A21);
- a born cry refused;
- the amygdala's law and constants (A16);
- the software fovea's control (A23);
- the critics' solve interval;
- the parent's method: its feelings, timings, force caps, ear and conduct (A25, A27);
- the tract's anatomy, calibration and alphabet (A26);
- the tests of understanding and the scaffold's removal (A28, A29).

## 15. The stock G1: what it changed from the custom child

This morning's design built a custom child to a brief of about 02:00. The owner's later words (about 02:10–02:30, recorded in the project memory and in `allout/owner_amendment_0924.md`) made the child the stock G1. Commit dd8640e added the model "on the owner's yes". The custom child is kept for reference:
- `body/sim/make_livingroom_customchild.py` → `livingroom_customchild.xml` (byte-identical to this morning's `livingroom.xml` apart from its header line);
- `body/sim/scene_customchild.py`;
- its measurements in `$S/allout/` and `$S/allout/move/`.

| part | the custom child | the stock G1 |
|---|---|---|
| looking | eyes that turn (conjugate and vergence), a 3-joint neck, the VOR on the eyes | a software fovea (±38° × ±20°), then the waist and the body; the VOR on the window |
| its face | a screen face showing its face organ's readout; the parent read distress from it | none: the parent reads distress from pain, long spells face down, thumps and the charge light |
| size and weight | 0.75 m, 9.45 kg: the parent lifts, props and turns it as a baby | 1.32 m, 34.4 kg: a person turns it briefly, holds it near upright, and cannot sit it up, slide it or lift it |
| strength | adult reference × 0.08, from memory | the real motors' limits (section 3.2) |
| pain | 3 × 93 N ≈ 278 N | 3 × 337 N ≈ 1,012 N |
| hands | a tendon finger and a thumb (2 servos) | Dex3: three fingers, 7 joints; 6 of 10 toys holdable |
| the alphabet | 40 joint settings in 9 motor effectors, and a 79-row voice | 43 joints, 3 gaze and 10 tract articulators in 9 effectors, and the 79-row silent output |
| reach lying down | 0.53 m; 3 of 10 toys | 0.83 m; 3 of 10 toys |
| sitting | propped upright by the parent in 2.4 s | leaning forward only; the parent cannot sit it up alone |
| touch | every link, as on an infant's skin | every link: the Dex3's 14 links as on the real hand, the other 28 a sim skin (B18) |
| physics | 11–18× real time | 10.9–12.2× under babble |
| two eyes with the sun's shadow (render and read-back) | 24.2 ms | 32.5 ms |

**The custom child's babble numbers,** kept because the G1's are not yet measured (the movement study's model: 0.75 m, 8.7 kg, 32 hinges; 3 × 4,000 ticks per condition):
- rolls come only from babble correlated in time: 0 a minute with a fresh step every tick, 1.10 with chunks of 1–8 ticks, 0.17 when 60% of the chunks are rests; half-size steps give 0;
- torque does not limit rolling: it still rolls at 0.3× strength and stops at 0.15×;
- head turns do not steer rolls: 51% of rolls go toward the side the head turned;
- it never sits up by chance (the trunk upright on at most 0.14% of ticks);
- it travels 1.2–3.3 m in 10 minutes and leaves a 1.2 m mat within 1–10 minutes;
- propped and let go, it falls within 0.15–0.9 s, its head striking the mat at 260–890 N.

## 16. What grows next: the house

The room has a doorway to a hall. Each change of world is the owner's call and joins at a boundary.
1. **The hall and a kitchen.** The parent's own tasks happen there, and the bottle is filled there. Leaving and returning then have a place.
2. **A bedroom.** Sleep in a place of its own; how it lies at night becomes the child's own doing.
3. **Stairs.** Climbing, which the parent guards.
4. **More objects to name.** Books, a ball pit, food toys.
5. **A second adult voice.**
6. **After the first body: the real robot.** The body is already the real G1's model, its senses sit where the real ones are, and its fovea and tract are software organs the robot can run.

The render cost is mostly fixed overhead (the pixel count barely matters), and physics contacts arise only near the child, so more rooms cost little per tick. They are loaded from the same maker.

## 17. Files and sources

**In `body/sim/` (the starting point; not committed):**

| file | what |
|---|---|
| `make_g1room.py` → `g1room.xml`, `textures/room_*.png` | the maker and the generated scene: the stock G1 (included unchanged, by relative paths), the parent with its graded face, the living room (edit the maker, never the XML) |
| `g1scene.py` | loading the room and adding the G1's senses at load (the stereo pair at the D435, the ear sites); the parent's mocap and graded face; the welds (for toys; those on the G1's torso, pelvis and elbows are replaced by capped springs in W2, section 4.1) and the hand proxy; touch per link; the birth pose (cached to `g1_birth_state.npy`); SimWorld grows from it |
| `g1eyes.py` | the two eyes: one render each, one read-back; the periphery; the movable software fovea (yaw, pitch, vergence) |
| `g1acts.py` | the parent's acts aimed at the G1's parts (attend, show, guide, two hands on the body) |
| `parent_kin.py` | the parent's skeleton, forward and two-bone inverse kinematics, hand shapes, the face (the scalar path, and the graded FACS face with `face_reading`) |
| `parent_poses.py`, `parent_acts.py` | its scripted motions and acts, each with a report of joint ranges; an act never chooses the face |
| `parent_feel.py` | the parent's feelings, their display on the graded face, the born reading with the increment rule, and the no-farming self-test |
| `tract.py` | the child's articulatory vocal tract |
| `ears.py` | the two ears (the born cochlea, the brainstem's delay lines, the spatializer) |
| `parent_ear.py` | the parent's ear for the child's words, with the context and babble-bank decision rule |
| `make_livingroom_customchild.py` → `livingroom_customchild.xml`; `scene_customchild.py` | the custom child's room and loader, kept for reference (section 15) |
| `highchair_onearm.xml`, `make_highchair_onearm.py` | the one-arm high chair, kept for reference (`tools/sim_look.py` renders it) |
| `assets/unitree_g1/` | the stock G1 (commit dd8640e) |

**In `tools/`:** `sim_look_g1.py` renders the G1 room's stills into `video/sim_look_g1/` (`--out=` for elsewhere).

**To create** (section 11):
- `body/sim/world.py` (SimWorld), `body/sim/senses.py`, `body/sim/anatomy.py` (SimAnatomy);
- `body/sim/voice/`, `body/sim/lang/{lexicon,templates,conduct,ledger,day,transcriber,consts}.py`, `body/sim/serve.py` (`/sim` on port 8030);
- `body/core/amygdala.py`, `body/tests/test_amygdala.py`;
- `tools/sim_babble.py`, `tools/sim_parent_rates.py`, `tools/sim_eye_check.py`, `tools/sim_digest.py`;
- `ops/sim_parent_brief.txt`;
- `body/tests/test_sim_world.py`, `body/tests/test_sim_lang.py`;
- `tools/pins/sim_script.npz`.

**Stills and sounds:**
- `video/sim_look_g1/`: `room.png` (the parent kneeling by the G1, a hand on its chest), `kneel.png`, `show_toy.png` (the duck 40 cm before its eyes), `g1_eyes.png` (both eyes, peripheries and fovea windows, with the room at the same moment); `parent_faces_graded.png`, `parent_feelings_timeline.png`, `parent_face_fovea.png`.
- `video/sim_look_g1/voice/`: `voice_sheet.png` (15 s of babble; "ball" and "mama" as the parent, hand score, searched and echo; the vowel space) and the matching `.wav` files.
- `video/sim_look_allout/`: the custom child's stills.

**Scratch studies** (not in the repository), under `$S = /private/tmp/claude-501/-Users-lukehamond-Projects-project/81d92d50-4dd9-488b-8268-f1a474117bfc/scratchpad`:
- `$S/g1/`: the G1 in the room: `m_speed_g1`, `m_render_g1`, `m_reach_g1`, `m_postures_g1`, `m_grasp_g1`, `m_help_g1` (each a `.py` with its `.json`); `tmp/` (the solver side tests, the rolls, the showing composition; `t_view2.py` and `t_view3.py`, where the head camera's fovea can reach a face in each posture; `t_size.py`, the lying G1's size).
- `$S/g1/parent/`: the parent's limits with the G1 (`m_g1_ramp`, `m_g1_help`) and the face sheet.
- `$S/g1/voice/`: the tract's studies (`babble`, `search`, `t_imitate`, `t_reject`, `t_xbank`, `t_cost`, `t_calib2`, `t_fitcorners`, `sheet`) and results; `xvoices/` (8 extra speakers' clips, 22 MB).
- `$S/g1/amyg/`: the amygdala on a synthetic stream (`m_amyg.py` to `m_amyg4.py` and their `.json`).
- `$S/allout/`: the all-out world and the custom child; the core at this scale (`t20_humanoid_tick.py`, `t21_parts.py`, `t22_paired.py`, `t23_rss.py`, `prof20.py`); `move/` (learning to move); `lang/` (the parent's language: `synth.swift`, `voices.swift`, `ear2.py`, `parent_lang.py`, the tests); `owner_amendment_0924.md`.
- `$S/sim_wf/`: the old design's measurements that still hold.

**Sources for the G1's senses:** Unitree's `unitree_ros` `robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf` (the `d435_joint`); Unitree's G1 product page; the Menagerie `unitree_g1` README.

**Documents:** ARCHITECTURE.md, BODY_SPEC.md and ops/review_2026-09-22.md (the defect numbers).

## The decision log (2026-09-24, amended for the G1)

This log settles the design's open decisions and edge cases, in three parts:
- **(A) Decided here (ours):** each decision with its reason under the laws.
- **(B) The owner's calls:** plain questions, each with a recommended default.
- **(C) Open until measured:** what decides each one, and where it is measured.

Where a decision changes an earlier section, that section points here. W, P, R and S refer to the build plan (section 11).

**What the edge cases were found from:**
- The all-out world (this morning, the custom child):
  - the sofa's base stands 8 cm off the floor, and five toys fit under it, 0.88 m deep, beyond the parent's reach;
  - MuJoCo silently resets the whole state when it meets a bad acceleration.
- The G1 in the room (today):
  - the prototype's solver crashed with Dex3 grasps;
  - the parent's hand stayed collidable after its switch cleared only one bit;
  - babble's contact peaks at the tick ends reached 944–1,799 N, and 7,205 N against the kneeling kinematic parent;
  - a person's force is below most of what a baby's parent does.
- What the G1 amendment opened (this log's update; `$S/g1/tmp/t_view*.py`, `t_size.py`):
  - the real G1's head camera is pitched 47.6° down for walking, so face down it sees only the mat, and sitting or standing it sees only what is low and ahead (A22);
  - the lying G1's chest is 0.16 m high, 1 cm under the low table's decorative shelf (A24);
  - a fast guide of its arm passes the parent's cap (A25);
  - the parent's ear's "other speakers" were other macOS voices (A27);
  - removing the scaffold must not change the body, or it would be a new body (A29).
- The review of the amendment against the laws and this Mac (later on 2026-09-24; the worktree's code, the G1's file and MuJoCo 3.9 checked directly):
  - touch on every link is a sense the real G1 lacks (B18), and the sun's shadow and the cup's size had been decided without the owner (B3, B1);
  - the event line "a face in the fovea" read the world's face test, a world-truth feature in the body's value (section 3.4, A1);
  - the born movement units are about 1.4 ticks long under R6's continuation draw, not the babbler's 4.5 (section 3.6, C38);
  - a quick phase would have been added by a measured rate (A23), and the catch's reaction could have been changed to reach 95% (C6);
  - the scaffold's removal rule was ill-typed and needed a speech detector (A29);
  - the G1's holds were welds with no cap (4.1); its frictions and self-contacts were to be set on its own geoms (A21); MuJoCo 3.9 does not apply the IMU noise its file declares, and has a switch for its silent reset (A18);
  - the save's size (0.95 GB) had no source and was below the language save's 1.28 GB (section 9);
  - R6 fix 3 was already committed (e48b284).

### (A) Decided here (ours)

**A1. What counts as looking at the parent's face** (the reward's gate; sections 3.4 and 6)
- **The test.** It runs on each tick for each eye, and passes when all four hold:
  1. the parent's mouth point lies inside that eye's software fovea window (32 px, about ±10.5°);
  2. a ray from the eye to the mouth point hits nothing first. The ray runs over the shapes the eyes render, so the child's own hand, a toy, the parent's hand or its hair all block it;
  3. the parent's face is turned within 75° of the eye (in profile a smile still reads; from behind it does not);
  4. the face's front (an ellipse about 0.17 × 0.21 m) covers at least 20 fovea pixels, scaled by the cosine of that turn.
- Either eye passing counts. The test must pass on 2 consecutive ticks (300 ms), so a sweep of the window across the face is not a look.
- **The range this gives,** at 1.5 px a degree: about 3 m with the face turned toward the child. The parent on the sofa can be seen; from the hall, mostly not; closer than 25 cm, never (A3).
- **No mutual gaze is required.** Where the parent looks is its own cue, not part of the child's gate.
- **Why a ray test, not a segmentation render.** Nine rays cost microseconds, where a second render per eye costs milliseconds. C2 checks the rays against a segmentation render; if they disagree too often, the render is used.
- **The born reading reads the parent's expression from the world:** 2 × (smile − frown) of her graded face, not from pixels.
  - The grades are visible in the fovea's pixels at the lean-in distance (section 4.3), but from about 1 m the mouth is too few pixels for any born pixel reader.
  - This is disclosed as how the born reading works. The test above is where the child's own eyes decide.
- **The test gates the reward's carrier and nothing else.** The critics' and the amygdala's event line "a face in the fovea" comes from the born face template on the fovea's own pixels (section 3.4). Given this test instead, the body's value would read a perfect world-truth face detector: the one feature that best predicts a smile.

**A2. Glances and smile-farming** (sections 3.4 and 4.3)
- **The born reading holds its last value for 30 ticks out of view, then reads neutral.** 30 ticks is longer than any smile lasts once seen, so the same smile can never be felt twice, and an old smile never blocks the next.
- **The increment rule** (section 6) feels only rises; easing off is never felt.
- **No face can be farmed.** The parent's resting face is neutral; her other faces read 0 except the smile and the frown; every smile returns to neutral within 15 ticks of being seen. So looking away and back gains nothing. The self-test confirms it (section 4.3).
- **The call answered earns a smile only until the child's name is "understood"** (section 4.8).
  - After that, the answer to a call is the next activity (a toy shown), not a smile.
  - The call stays at most once per 240 ticks. It is made only when the parent's plan needs attention (a new episode or an ask), never on a timer of the child looking away.
- **A toy put into the hand earns no smile.** A handed-over toy held by the grasp reflex earns narration ("you have the rattle."). A held toy earns a smile only when the child reached it itself, with no hand-over in the last 40 ticks.
- **Peekaboo:** the reveal shows surprise, which reads 0. The smile comes only when the child answers the reveal with an act: a vocal turn, a reach, or kicking.
  - **An answer** is an act begun within 10 ticks of the reveal by an effector (the voice, an arm or a leg) that had rested for the 5 ticks before it. Babble already under way is not an answer, so ongoing movement is never paid as a reply. Its rate under the babbler is written down as chance (P6).
- **Mastered acts.** An unasked motor act's worth falls with mastery as 1 + e^(−n/10) over its n earlier smiles, floor 1. A met ask keeps +2. Nothing the child earns falls to 0.
- **No smile ever answers distress** (A13).
- **Frowns** are held 10 ticks and never wait for a look, and an unseen frown is not felt. A frown that waited for the look would teach the child not to look.

**A3. The parent's face timing** (section 4.3)
- A smile starts within a tick of the judgment, with the approval word, unless the start rule makes it wait (section 4.3). It is held until the child sees it, for at most 20 ticks; once seen, for 10 more; then it eases off over 5 ticks.
- Before it smiles, the parent moves any toy in its hand off the line between the child's eyes and her mouth.
- **Where her face goes:**
  - never closer than 25 cm to the child's eyes;
  - when she leans in or calls, her face arrives at least 15° off the fovea's current line and holds still there;
  - she never moves her face onto that line;
  - once the child looks, she keeps still, holding the mutual gaze.

**A4. Contacts between the two bodies** (section 4.2)
- **Which parts touch.** The parent's collision shapes (contype 8, conaffinity 1) touch the G1 (contype 1) and the toys. The planner keeps her legs, trunk and head at least 3 cm from the child in every planned frame; only her hands and forearms touch the child, and only during an act.
- **The switch.** Turning a hand's collision off clears both of its bits. The prototype's first switch cleared only contype, and a 2.2 kN contact made the hold forces read in kilonewtons.
- **A soft parent.** Her collision shapes get a soft contact (solref 0.05, like flesh and cloth).
- **The parent yields.** If any contact between her and the child is over the act's cap for 2 physics steps, that segment stops and backs off 2 cm a tick along the contact. The act resumes when the force is gone. With a kinematic parent kneeling into babble a contact reached 7,205 N, so this rule is essential. It is her care, on the world's side; nothing in the child's body.
- **Contact with the parent is touch.** It counts toward pain under the same law as any contact, and so do her holds (section 4.2). A pain tick she caused is logged as her defect.
- **The child may hold her hand** (the reflex closes on it). She withdraws slowly (0.1 m/s) and yields when pulled against.
- **Handing a toy over:** she releases when the child's palm touch is at least 0.3 N and its fingers are closed at least 30° for 2 ticks, or after 40 ticks. A dropped toy is fine, and makes its sound.
- **Taking a toy back:** she opens her hand under the toy and waits, closing only after the child's palm force has stayed under 0.3 N for 2 ticks. She never pulls a toy out of the child's hand.
- **The caps** are a person's (section 4.2): one hand 100 N sustained and 150 N for up to 2 s; both hands 160 N and 200 N. This replaces the all-out design's G1 line (caps up to 337 N).

**A5. Toys lost**
- **Under the sofa.** Its base stands 8 cm off the floor, and five toys fit under it, 0.88 m deep, beyond the parent's arm. The default is a skirt to the floor (B8).
- **Under the low table.** It has 0.37 m of clearance; the parent kneels and reaches under it. Its under-shelf at 0.18 m is decoration only. B8 removes it: made solid, it would sit 1 cm above the lying G1's chest and could wedge it (A24).
- **Into the hall.** The doorway is open, and the hall is walled at its far end. The parent fetches toys from it and goes to the child there.
- **The tidy.** The parent puts a toy back (on the shelf or at the mat's edge, narrated: "the car goes here.") when the child cannot reach it and has not touched it for 2,000 ticks. Toys near the child stay where the child put them.
- **Out of reach.** A toy the parent cannot reach from the floor she fetches standing. If that fails, it stays lost until the next morning, when the world returns it to its basket (B8). This is logged, and applies only to toys.
- **Physics faults** (a toy below the floor, outside the walls, or faster than 20 m/s) are world defects (A18).

**A6. The parent's walking and kneeling paths**
- **Walking.** A path on a 5 cm floor grid (A*), keeping clear of furniture by 0.15 m, the child's body by 0.25 m and toys by 0.08 m.
  - She never steps over the child, and never sets a foot within 5 cm of a toy.
  - A toy lying across the only path is picked up and set aside, narrated.
  - She walks at 0.8 m/s and shuffles on her knees at 0.25 m/s. Her footsteps make their sound.
- **Where she kneels:** about 0.72–0.78 m from the G1's torso centre line, beside its chest, on the side its eyes face (the side its torso is rolled toward), and outside its leg sweep (W4 measures the sweep). She arrives in its periphery.
  - Kneeling needs 0.6 m clear in front, so she kneels a step back and shuffles in.
  - Toys where she will kneel are cleared first.
- If no spot is free within 1 m (the child against a wall, under the table or in the hall), she kneels at its head or feet, or stands and bends; she never moves the child to make room.
- She comes toward the child from inside its field of view, unless she is calling from away.
- Plans are solved when an act starts and again once a second (W2). A frame outside human joint ranges is refused, and another spot is tried.

**A7. Turning, and no lifting or sliding**
- **No lift, no slide on the mat, no bringing back.** The G1 weighs 337 N; a person's caps are 160–200 N (section 4.2). She comes to the child.
- **No tummy time.** Face down the G1's eyes point 62–72° below the horizontal, back under its chest, so her face and toys ahead can never be seen (A22). Turning it onto its front would be distress with nothing to lift for. It lies on its front only by its own roll.
- **Turning:** onto its back when it is distressed face down for over 100 ticks. Two capped springs, on the pelvis and a shoulder, push by a slowly growing force within the caps for at most 2 s. If the turn does not complete within the caps, she stops, narrates, and helps by the roll's ladder instead. It is always narrated as her act, in words that pass the line check ("uh oh. mama is here."), never as a roll the child made.
- **No carrying for comfort.** A 34 kg child cannot be carried by the parent.

**A8. Help with rolling**
- The parent helps a roll only by guiding the limbs into its start: the far arm across the chest (at most 65 N), the far knee bent over (at most 76 N), while she says "roll". Then she lets go. The roll itself must come from the child's own acts.
- **A roll counts as the child's** when it is whole (back to front or front to back; ending on its side is part of a roll and earns nothing), it ends at least 3 ticks after the parent's last touch, and the child's own motor acts moved a limb in those ticks. Only then does she smile.
- She places herself where a roll would end facing her.

**A9. Help with sitting**
- **The pull-to-sit.** She kneels at the G1's feet and holds both forearms, pulling with a force that grows to the caps (160 N sustained, 200 N for up to 2 s). A limp G1 needs 217–237 N, so it rises only when its own waist and hip flexion carry the rest. If the pull sits at its cap for 2 ticks, she stops, lays it back gently and narrates. This is the pull-to-sit test infants are given, where the infant's own flexion is what is judged. Its hold and its inverse kinematics are solved in W2 (C7).
- **The prop.** Once its trunk is within 30° of vertical, she holds its torso with a capped spring (58 N at 30°, 20 N at 10°). The G1 sits only leaning forward with its hands on its knees; upright with the legs out, it tips back.
- **Easing off.** Each time the trunk has stayed within 20° of vertical for 10 ticks, the cap steps down: 100%, 70%, 40%, 20%, then her hands hover 5 cm away. When a fall starts, the cap goes back up one step.
- **The catch.** When the trunk passes 35° from vertical, or the head drops faster than 0.5 m/s, the hovering hands engage 2 ticks later (300 ms, a person's reaction), within the caps. The child feels the start of a fall through its vestibular sense, but not the strike. A failed catch is logged (C6).
  - What it must stop is not measured. The upper body (16.2 kg, its centre 0.25 m above the hips) releases about 7 J between 35° and 50°. Limp, it would pass 80° within the 300 ms reaction; under the resting law the hips' lag slows a sink to about 4° a second at 35°. The case that matters is a fall driven by the child's own big steps, which C6 measures on the G1.
- **Sitting alone for a moment** (a first motor act): at least 5 ticks at the hovering step, with no contact from her and the trunk within 30° of vertical.
- **When.** Only in the motor blocks, at most 2 × 1,500 ticks a day, and distress ends it at once. "up" said by the child also starts it.

**A10. Demonstrations**
- **Guiding is the only way a movement is demonstrated into the body.** The parent moves a forearm, hand or shin along a scripted path of at most 8 ticks (one movement unit), at up to 0.3 m/s, saying the action word.
- **The guide's cap** is min(1.5 × the limb's own push at that pose, 100 N): 98 N for the G1's arm (its push is 65 N). When the spring has sat at its cap for 2 ticks, because the child resists or the joint is at its range, she stops. Measured: a limp arm is guided slowly within it; an arm under the resting law or stiffened is not (section 4.2).
- **What the child learns from a guide** goes through `act_inv`'s reliability (section 3.6): a guide counts only as far as the inverse model has earned.
- **A guided act earns no smile.** A smile comes if the child repeats the act itself within 40 ticks.
- **Showing by doing** (shaking the rattle, rolling the ball, tapping the drum) reaches the child only as sights and sounds. The parent's spoken words reach the voice only through its ears and its own inverse model. There is no mirror module and no imitation shortcut.

**A11. The hands' grasp**
- **Holding is friction only:** no weld, no attach and no sticky rule for the child. The Dex3's collision shapes keep their friction as shipped until W1 sets real frictions (C26).
- **The grasp reflex** (section 3.7) closes all seven Dex3 joints one step a tick while the palm touch is at least 0.3 N, unless the hand's own act opens it that tick. The palm is the hand's base link; its touch is read from contacts (the real Dex3-1 has tactile arrays).
  - It never opens by itself. Letting go is the child's own act and is learned; the ask "give me the cup" needs it.
  - It fires on anything in the palm: a toy, the bottle, the parent's hand, or the mat. W4 counts closed fists on the mat.
- **Six toys can be held** (section 3.8). The other four are for looking, pushing, rolling, hitting and naming unless B1 changes them.

**A12. Pain for a body on the floor**
- F_pain stays at 3 × the body's weight per link: about 1,012 N for the G1.
- **A link's force for pain** is the tick's largest 10 ms mean (5 physics steps) of its summed normal force.
  - A single-step solver spike is not a blow. Real impacts in this contact model last 10–30 ms.
  - Babble's single-step peaks at the tick ends (944–1,799 N) make phantom pain a real risk. W4 counts pain with and without the filter (C5).
- **Self-contact counts:** a kick to its own leg hurts. MuJoCo already ignores contact between neighbouring links. Any other pair that presses into itself at rest is listed in the world (never by editing the G1's file), and disclosed.
  - The list removes that pair from touch and pain only, as a sensor's blind spot. It never excludes the pair from collision: the G1's self-collision stays as shipped, so the body's physics is the stock robot's.
- **Not at birth:** pain at a joint's range (C).
- **The parent:** a hold's spring force is touch on the held link under the same law as any other force. Her caps (at most 200 N) keep it far under F_pain, so no exception to the law is needed.
- **Being pinned** costs −1 a tick for as long as it lasts. The only source we know of is the parent, and the yield rule ends it.
- **What 1,012 N means for 34 kg.** The rule is the custom child's (3 × 93 N ≈ 278 N), carried to the G1's declared mass, not refitted:
  - the loads it carries are far below it: the largest resting contact lying is 272 N, and the pelvis carries about 250 N in the leaning sit;
  - the thump line the parent reads as distress is half of it, 506 N (A13);
  - the custom child's falls from the prop struck its head at 260–890 N at 9.45 kg; the G1's falls under the final servo law are not yet measured (C31). A fall that passes 1,012 N hurts, as it should.
- **One threshold for all 42 links,** not one per link:
  - a threshold from each link's own mass would put a finger link's at a few newtons, so its own weight on its hand, or any grasp, would hurt;
  - the skin's pressure threshold would need contact areas, which MuJoCo's contacts do not give reliably;
  - the real G1's damage limits per link are not known here (C37).
- **The robot after.** If any real G1 link is damaged below 1,012 N (C37), the pain law takes that link's limit for the next body, which is a new body with its own seed (A20), never a change to seed 1.

**A13. What the parent does when the child does not look, is distressed, or babbles**
- **When it does not look:**
  - she follows its attention, naming what its fovea rests on;
  - she shakes a toy that makes a sound beside her own face;
  - she calls, at most once per 240 ticks. After 3 calls unanswered in 720 ticks, she carries on with the activity where the child is looking.
  - She never frowns for not looking, never touches the child to make it look, and never moves her face onto its line of gaze.
- **When it is distressed.** The G1 has no face, so distress is read from outward events: a pain event in the last 40 ticks; over 100 ticks face down; thumps (limb strikes above half F_pain) at more than 3 in 40 ticks; the charge light low. She checks in this order:
  1. pain in the last 40 ticks: comfort;
  2. charge below 0.5: feed;
  3. face down for over 100 ticks: turn it onto its back (A7);
  4. a toy just lost from its hand: offer it back;
  5. otherwise: a comforting voice and a change of activity.
  - Her concern shows on her brows and pressed lips, which read 0. Distress is never answered with a smile.
- **When it babbles:**
  - **Stage 1.** A vocal turn in a pause while looking earns a small smile (at most once per 60 ticks) and a reply 3 ticks after the turn ends. While not looking, it earns only the reply.
  - **Sounding during her line:** she finishes the word, stops, and looks with a listening face. No frown in stage 1.
  - **Babble that never stops** (the tract sounding on more than 70% of 40 ticks): she waits 20 ticks, then speaks anyway, as parents do. No frown in either stage.
  - **Stage 2** frowns only when the child talks over her own line.
  - When both start on the same tick, the child has the turn.

**A14. What Claude (Opus) may and may not do** (sections 4.5 and 4.10)
- **May:**
  - choose the focus toys, the next episode and the scaffolding task, within the day plan's ranges;
  - write lines tied to situations (when it holds the duck: "you have the duck.") and variation-set frames, each passing the line check;
  - name the one new word to introduce, from the queue;
  - set how long the parent is away, within 400–1,200 ticks;
  - write a note for the daily report;
  - propose changes to the parent's method. A session applies them only at a night boundary, after testing on a copy.
- **May not:**
  - set or nudge any feeling, the face, the gaze, a judgment, a worth, a help level or a force;
  - write praise ("yes", "good", the approval register), which belongs to the fast layer's judgments;
  - loosen any criterion, or choose activities for their smile rate;
  - change any constant of the body, the parent's caps, timings or feelings, or the world;
  - see anything inside the child: weights, gates, values, forecasts, tags or the face organ's numbers;
  - act inside a minute, or at night;
  - introduce more than 1 word a minute or 3 a life day, or a word whose referent the parent cannot show within that minute;
  - say a held-out never-taught pair before its test.
- **When Claude is silent or fails,** the fast layer runs alone, and the log says so. Rows are logged by tick, so a replay is exact whatever Claude did.

**A15. Sentences and vocabulary growth** (section 4.8)
- A queue word enters only when its referent or act is in the world and the parent can show it within the minute. Colour words wait for the colour twins (B2).
- Claude may add queue words at a night boundary, under the same test. Otherwise the queue is fixed.
- **Unchanged:**
  - the line check: at most 6 words, `. ? !`, and the vocabulary plus that day's new word, placed last;
  - the pace: at least 1 new word every 2 life days (the world does not wait for the child), at most 3 a day.
- A queue word the child says before it is introduced is not recognised: the parent's transcript and her ear know only the vocabulary.
- **The never-taught pairs** (section 12) are listed before birth and held out by the line check until their test.

**A16. The amygdala's constants** (the owner's decision 2; section 7.4 and its rows in section 10). The organ is the owner's decision; its law and constants are the body's, so ours. Each is fixed before birth, and none is fitted to a rate. It replaces this morning's "valence learner", whose short memory (forgetting 0.998, about 500 ticks, fewer than its 525 inputs), single signed forecast, tag scaled by running means and tag-weighted `act_pred` lesson were each measured or reasoned wrong.
- **Its input:** the cortex's stream C/√d (the high road), the anatomy's event lines (the low road: 12 for the sim, the same declaration the critics read) and a level. No world truth.
- **Its heads:** one per reward source and sign; for the sim, face +, face −, pain −, charge + and charge −. The basolateral amygdala keeps good and bad apart; one signed forecast would cancel a cue that brings a smile and then pain.
- **Its horizon:** 0.9375 a tick, dopamine's own discount, so no new time constant.
- **Its memory:** τ_a = 4,096 ticks (band 6's clock), eight times its 525 inputs, so the fit is determined. Learning in one pairing comes from least squares, not from a short memory.
- **Its prior:** 0.3 × τ_a × each input's running variance, the critics' own. A prior of 1.0 was measured (an event line learned faster, stream cues slower) and not adopted; it stays a later copy measurement.
- **Its solve:** every 8 ticks: 0.69 ms a solve, 0.19 ms a tick.
- **Its reliability:** the face organ's measure (the correlation with the realized target, finalized 64 ticks later, zero until 64 pairs). At birth its forecasts reach nothing.
- **The tag:** what was received plus the reliability-weighted forecast, capped at 2 (the face's clip), in reward units. No running-mean scaling (defect 7).
- **Reaching back** 64 ticks at 0.9375 (behavioural tagging). It covers risk 1's 7–25-tick delay for memory and the night, never for credit.
- **The store:** the write gate tests surprise × (1 + tag) against its own running 0.9 quantile, so tags change which frames are written, not how many. A write's strength is × (1 + tag), with later boosts, at most × 3.
- **The night:** entries × (1 + T_e) over a saved running mean of 64 episodes; episodes with T_e ≥ 1 dreamt first, taking at most half the night; the window ending at the peak tag; `act_pred`'s lesson weighted clip(1 + G, 0, 1) by the replayed dopamine, never by the tag, which would teach the acts that led to a fall.
- **Orienting:** gain clip(1 + N, −0.5, 2), exactly 1 at birth. The upper bound at most doubles the born pull. The lower bound is the owner's "toward or away" at its weakest: a turn away never stronger than half the born pull toward (risk 15).
- **Awake only:** it learns only while awake; its evidence is kept across the night.
- **`amyg_pav`** is built and off at birth (section 7.4 gives why, and when it is tried on a copy).
- **Nothing else takes the tag or the forecast:** not reward, dopamine, the gates' credit, mood, the cortex's loss, or anything the parent sees.

**A17. The day, the night and the charge** (section 5.3)
- **The bottle's dock.** The maker's pad is at the mat's corner, far from the child at birth. W3 moves the dock beside one hand, within reach of the lying G1, and it moves to the mat's corner only when the bottle's stage 3 begins (section 4.7).
- **The bottle never empties.** It charges the child while it touches a palm, whoever is holding it.
- **A bedtime feed.** If h is below 0.6 when goodnight begins, the parent feeds first. So every night starts near full, and nothing about the night is timed to the child's rates.
- **Before the night pause:**
  - at tick 23,700 the parent releases every hold and steps back, so the paused world has no parent contact;
  - the child sleeps as it lies, and the morning resumes the same state (a fall in progress resumes as a fall);
  - the light follows the day (section 5.4).
- **No death and no shutdown.** At h = 0 the body keeps 30% of its strength. The parent feeds below 0.35 and checks within 200 ticks, so h below 0.1 is a parent defect.

**A18. Crashes, disk and heat**
- **After a crash, the lost part of the day replays exactly.** Every random stream, Claude's rows and the synthesized clips are logged, and the tract is deterministic, so the life resumes from the last save (the last night) and replays the lost part of the day exactly. A crash costs wall time only, so there are no saves in the middle of a day.
- **MuJoCo's silent reset and its fatal errors.** MuJoCo resets the state by itself on a bad acceleration, and counts it in its warning counters; it can also raise `mujoco.FatalError` inside `mj_forward`. MuJoCo 3.9 has a switch for the reset (`<flag autoreset="disable"/>`), so the world turns it off: a bad state then stays visible as NaN instead of being silently replaced by the start pose. The world also checks the counters every tick and catches the error. On any reset, NaN or fatal error:
  - the life pauses before that tick reaches the body;
  - the last good state is saved;
  - the fix is measured on a copy;
  - the life resumes from the last save with the fix, logged as a change of world at a boundary.
- **The solver** is the elliptic friction cone with multi-point collision detection off (section 5.1), the only setting that ran the Dex3 grasps clean.
- **The voice server.** A cache miss makes the world wait. If the server is down for 60 s, the life pauses at that tick. It never skips or swaps a line. If Claude is down, the fast layer runs alone (A14).
- **The Mac** runs `caffeinate -i` while the child lives. A closed lid or sleep only pauses the life.
- **Disk.**
  - The rule of section 9 stands (skip the write, log why, pause at the next boundary). It is checked every 1,000 ticks and before each save.
  - The disk filled at 03:57 (0 bytes free, ENOSPC), after 12 GB free at 03:25, 5.1 GB at 03:46 and 3.1 GB at 03:50. The refactor's verification copies in `$S/sim_refactor` had reached 19 GB. Clearing them left 20 GB.
  - **Rules for the build,** since three parallel sessions share one disk:
    - each session deletes its own verification copies when the run ends, whether it passed or not;
    - no session starts a copy unless the free disk after it would still be at least 8 GB;
    - one verification run at a time across all sessions;
    - scratch studies stay small (the G1 studies used about 40 MB in all).
  - The owner's folders are never touched.
  - Birth needs at least 8 GB free: the peak of about 4.4 GB plus the rule's floor of about 2.7 GB, and a margin (section 9; the earlier 6 GB rested on a save of 0.95 GB that had no source).
- **Heat.** In lockstep, heat slows only the wall clock, never the life. The birth checklist's mean tick is measured over two heat-soaked hours. One life at a time on this Mac.

**A19. How each milestone is judged** (section 12)
- **A milestone is reached** when its ruler holds on 2 consecutive life days and a watched session at 1× on `/sim` shows it at least 3 times. The report describes what was seen, with stills redrawn from saved states.
- **Understanding is judged as well:** the milestone's never-taught test (section 12) is reported beside it, passed or not yet, with each probe seen described.
- **Chance is always the same body's own rate:** its rate at matched random moments of the same day with no cue (as the ledger's base rate), or, for M1, its own first 2,000 ticks of life. The babbler's numbers, written down before birth, are chance for the motor rates. There is no trained baseline run.
- **The test** is one-sided (binomial or permutation), p < 0.01 over a life day.
  - A ruler with too few events in a day is pooled over the fewest consecutive days that give at least 20 events.
  - The pooling rule is fixed now, before birth.

| | ruler (outward events only) | chance |
|---|---|---|
| M0 | its forecasts of its own next body sense and periphery beat "nothing changes" on held-out ticks | "nothing changes" |
| M1 | after the parent's voice starts out of view: the face test (A1) passes within 20 ticks; the share of ticks with the face in the fovea while the parent is within 3 m | its own first 2,000 ticks (the born orienting included) |
| M2 | after a toy is shown over its chest: a hand touches it within 40 ticks | matched moments with a toy in the same place, not shown |
| M3 | the hand opens by its own act in the 5 ticks before touching a holdable toy, then holds it at least 3 ticks (a hold by the reflex alone counts for neither side) | the same at matched moments |
| M4 | whole rolls toward the side where the parent or a shown toy is, as a share of all rolls; the trunk turning first | the babbler's share under the same parent placement, measured in W4 and P6 with the born motor timing (C38), the born orienting and the VOR on (A22) (not 50%: she kneels on the side its eyes face) |
| M5 | in the pull-to-sit, the share of the rise its own flexion carries; while held, the share of ticks with the trunk up, looks at the face, and the easing step reached (A9) | its own first pull-to-sit day |
| M6 | the ledger's "says"; right names as a share of the names said while an object is in its fovea or hand | per word, its own rate of saying it with the object absent |
| M6t | M6's ruler on the tract's sound alone, the tokens ignored; the precursor, the first accepted echo | its own rate with the referent absent, plus the babbler's 2% |
| M7 | sitting (pelvis on the floor, trunk within 30° of vertical, no contact with the parent) for 67 ticks (10 s); the leaning sit with the hands on the knees counts, and a sit with the hands on the mat is counted apart | none: it is an event |

- Reaching a milestone changes no constant of the body. The parent's stages change at the next morning.

**A20. One seed, one body.** Seed 1 of the stock G1 is born once. Changing the body after birth would make a new body, with its own seed and its own life, never a continuation.

**A21. The G1 stays stock.**
- Its file is included unchanged and checked byte for byte at the birth checklist.
- Its senses are added at load, as cameras and sites (no shape, mass, joint or actuator).
- Its servo gains are set at load, as the real robot takes kp and kd with every command (section 3.3).
- The Menagerie scene light is switched off at load.
- Anything the world needs about the G1 is set in the world, never in its file, and never on its geoms at load either:
  - frictions and contact softness through the world's own geoms at contact priority 2 (C26);
  - a self-contact pair that presses at rest is blind to touch and pain, never excluded from collision (A12);
  - the IMUs' noise its file declares is added by the world, since MuJoCo does not apply it.
- **What does change at load, and why each is still the stock robot:** cameras and sites added (sensors only); the servos' kp and kd, which the real G1 takes with every command; the torque limit scaled by weakness (0.3 + 0.7h), a clip the real robot's command could apply. The last is a sim physiology with no counterpart in the robot's hardware, disclosed as such.
- **One sense goes beyond the robot:** touch on the 28 links outside the Dex3 hands (B18).

**A22. Looking without a neck** (the owner's decision 9; sections 3.1 and 3.4)
- **How it looks.** The gaze moves a software fovea inside each camera image (A23). Beyond the fovea's reach, the waist turns the trunk (yaw ±150°, roll and pitch ±30°), and then the whole body. Lying on its back, a waist yaw turns the upper trunk against the mat, the start of a roll.
- **Its eyes are aimed at the ground.** The real G1's head camera is pitched 47.6° down, for walking. We keep it where the real one is (the owner's decisions 7 and 9). Where its fovea can reach a face, measured on section 3.8's postures with the waist as posed and pitched fully back:

| posture | eyes' height | where the fovea can reach a face |
|---|---|---|
| on its back | 0.14 m | leaning over its chest (the showing pose: yaw −24°, pitch +11°); toward its feet, a kneeling face 0.5–1.5 m from its eyes and a standing face 1.0–3.0 m; never at its sides or beyond its head |
| on its front, on its elbows or on hands and knees | 0.25–0.48 m | nowhere: its eyes point 42–72° below the horizontal, back under its own body; it sees the mat |
| sitting upright (it tips back, section 3.8) | 0.77 m | straight ahead, below 0.49 m high at 0.4 m and below 0.35 m at 0.6 m; with the waist fully back, 0.44–0.76 m at 0.4 m |
| the leaning sit | 0.65 m | below 0.21 m high at 0.4 m, nothing at 0.6 m (its own legs fill the view); with the waist fully back, 0.36–0.68 m at 0.4 m |
| standing | 1.28 m | 0.30–1.07 m high at 0.4 m, up to 0.76 m at 1.0 m; with the waist fully back, up to 1.26–1.29 m; a standing adult's face never |

- **What follows** (ours; the parent's method):
  1. **She puts her face where its eyes can reach** in its current posture and place: on its back, leaning over its chest, or kneeling toward its feet outside its leg sweep; sitting, low and ahead of it. The poses are solved in W2 (C34). Where no pose in human ranges reaches, she does not expect a look.
  2. **Face down she never expects to be seen.** There is no tummy time (A7). She speaks from beside it and puts a toy on the mat where its eyes look (the ladder in section 4.10).
  3. **Every ask, call and judgment that needs a look** is made only while her face, or the object, is within this reach. She moves first. The never-taught probes follow the same rule (A28). An ask no body could meet from where it lies is never made.
  4. **M5's "looks at the face"** counts only while her face is within reach. In the leaning sit that needs the child to raise its trunk, which is the sitting skill itself.
  5. **Orienting's face template** can fire only where a face can appear: on its back, above its chest and toward its feet.
  6. **M4's chance.** The born orienting may turn the trunk toward her voice and tip a roll toward her. The babbler that sets M4's chance runs with the born motor timing (C38), the born orienting bias and the VOR on, since they are part of the born body, so born turns count as chance.
- **The parent reads where it looks** from the fovea's window (B14).

**A23. The software fovea's control** (sections 3.4, 3.5 and 3.7)
- **Its state:** yaw and pitch (both windows together) and vergence, in degrees in each camera's image. Born at 0, 0, 0: both windows centred, the eyes parallel.
- **Its steps:** ±4° or ±11.5° a tick for yaw and pitch; ±1.7° or ±5.7° for vergence. A 30° shift takes 3 ticks. A newborn's gaze shifts to far targets are also slow and come in steps, a series of small saccades (Aslin and Salapatek 1975), so no faster saccade is added.
- **Its reach:** each window's centre is kept inside its image, which puts it within ±38° × ±20° of the camera's axis. (`g1eyes.py`'s docstring says about ±33° × ±18°, a linear estimate; the code clips in pixels.)
- **Vergence:** 0–12°, from parallel to a point 0.25 m away, the nearest her face comes (A3). Anything nearer is seen double, as inside an infant's near point.
- **At rest the window stays where it was left.** Nothing in the image pulls it back to the centre.
- **The VOR** (section 3.7): each tick the window counter-shifts by the torso gyro's rotation over the tick about each camera's own image axes (the gyro rotated into the camera's frame, pitched 47.6° from the torso), at gain 1, and is clipped at its reach.
  - Roll about the line of sight is not compensated. The window is square and cannot turn; the human torsional VOR is weak too.
  - **Its quick phase is kept from birth** (changed in the review of the amendment). The first draft had none and would have added one before birth if the window sat pinned at its edge on more than 10% of ticks: a born reflex chosen by a measured rate, which the laws refuse. It is decided on biology instead: the VOR is a brainstem reflex of slow and quick phases together, and rotating a term newborn elicits nystagmus with both (from memory; W3 checks a source). So when the counter-shift would carry a window past its reach, the window jumps back, in the direction of the trunk's turn, by half its reach, within that tick. It is logged as reflex, with no gate eligibility. C33 counts the quick phases and only reports them.
- **Refused:** smooth pursuit, optokinetic following, a saliency map, or any rule that moves the fovea to the brightest or newest thing. Pursuit is learned, as infants' smooth pursuit develops over the first months, through `act_pred` and the gaze's forward model. The orienting bias is the only born pull.
- **One effector for both eyes.** Only conjugate and vergence commands move the windows (Hering's law), so one eye never looks away alone.
- **Its consequence sense** is the window's state (6 numbers in the body channel) and what the fovea then sees.
- **Its fatigue** is 0.03 a step; the gaze's gate reads its own.
- **One window serves all three:** what the fovea sees, the face test (A1) and the target the parent reads (B14).
- **On the real robot** the window is software on its own images, and nothing moves.

**A24. The G1's size on the mat and in the room** (section 5.1)
- **The body lying** (`t_size.py`): 1.32 m long and 0.71 m wide with its arms out. Its chest is 0.16 m high, its hands up to 0.30 m in the birth pose, and its eyes 0.14 m. On its elbows its chest rises to 0.37 m.
- **The mat** (B11): 2.8 × 2.0 m, centred at (0, −0.6). The G1 is born along its length with its pelvis at (−0.05, −0.62). That leaves about 0.65 m of mat beyond its head, 0.75 m beyond its feet and 1.0 m on each side. The parent's kneeling spot (0.72–0.78 m from its centre line) is on the mat at birth.
- **The room** (B15): 5.2 × 4.6 m. That is about 3.9 × 3.5 G1 lengths, against 6.9 × 6.1 for the custom child: for the G1 it is like a 2.9 × 2.6 m room for a baby.
  - The mat's front edge is 0.7 m from the front wall, and its back edge 0.33 m from the low table.
  - Its left end is 1.2 m from the window wall. Its right end is 0.8 m from the shelves' front and 1.2 m from the doorway's wall.
- **It will leave the mat.** Under babble its pelvis moved 0.16–0.42 m in 40 s (three runs), so leaving within a life day is likely. The floor is the play space (B7); C18 measures travel and time off the mat.
- **The doorway** is 0.9 m wide. The G1 passes it only lengthwise; the hall beyond is 1.9 × 2.4 m. She goes to it there (A6).
- **The low table:** the top's underside is 0.37 m up. Its legs are 0.87 m apart along its length and 0.35 m across, so a body rolling sideways meets them. A head-first entry could take its trunk under (C32).
  - The decorative under-shelf sits 0.17 m up, 1 cm above the lying chest. Made solid it could wedge a 34 kg body she cannot slide out, so B8 removes it.
- **The sofa's 8 cm gap** takes a hand or a foot, never the body (B8's skirt).
- **Nothing is moved to make room for it,** and the world never moves the child. Only its own acts, or her brief turn for distress (A7), change where it lies.

**A25. A person's strength, and how she helps** (section 4.2)
- **The caps are a person's:** one hand 100 N sustained and 150 N for up to 2 s; both hands 160 N and 200 N. They come from ergonomic norms recalled from memory; W2 checks them against their sources, and they may only tighten. They never scale with the child. No act is chosen for the force the child would need.
- **What they allow on the G1** (measured):
  - holding it within 30° of vertical (58 N at 30°);
  - guiding a limp forearm slowly (a ramp raised the wrist 15 cm at 44 N and 30 cm at 92 N);
  - bending a knee (60–76 N);
  - turning it from its front toward its back, as a brief act (a ramp of 94–170 N);
  - catching the start of a fall (the earlier "about 2.4 J at 200 N" has no source; A9 and C6).
- **What they refuse:**
  - lifting it (285–296 N);
  - sliding it on the mat (312–433 N measured);
  - sitting it up while it is limp (217–237 N);
  - guiding an arm that holds or resists (83–111 N, against the guide's cap of 98 N);
  - a fast guide (a 20 cm path in 1.2 s peaked at 139 N).
- **She helps in four ways only:**
  1. she comes to it, and places herself, toys and her face where its own acts pay (a toy at its reach boundary; her face where its eyes reach, A22);
  2. she guides a yielding limb into the start of a move, never through the whole move (A8, A10);
  3. she holds a posture it reached itself (the prop within 30°, A9);
  4. she catches the start of a fall (A9).
- **She never supplies the rise.** A hold or guide at its cap for 2 ticks stops. She lays the child back gently if needed and narrates. The act is logged as refused for force, and counts as her failure, not the child's. On the scaffolding ladder she never climbs to a rung that needs more force than the caps.
- **The guide's pace** is set in W2 on the limp arm, so a guide needs at most the arm's own push (65 N). That leaves the child room to stop her by resisting. The pace is never tuned to the child's learning.
- **Her holds are touch, never pain** (A12).
- **Why a person.** A real G1's carer is a person. A stronger parent would carry the child through the postures it has not reached, and B16 asks about that.

**A25b. The physical parent: her trunk carried, her limbs at a woman's strength** (the lead's structural decision of 2026-09-25, amending A25 and the first physical build; sections 4.1, 4.2, 4.10, A4, A6, C8; recorded here as A25b, renumbered at the merge)
- **Why.** The first physical build drove her whole body at a woman's strength and balanced it with a capped residual force on her pelvis. Under the babbling G1 she fell onto it and lay on it for minutes, her joints exerted more than her strength (MuJoCo's joint damping acted outside the clip), and the residual carried her gait with her floor contact off and lifted her while she touched the child. A person does not fall over: balance is the environment's business, never the child's.
- **Her trunk is carried** (`parent_body.Drive`). Every physics step her pelvis is pulled toward the pelvis her plan means (its place and its turn, interpolated through the tick) by a spring (20,000 N/m) and a damper (critical against her 62 kg: 2,227 N·s/m), with her planned acceleration fed forward and her weight carried at her whole body's centre of mass (so carrying it twists her not at all); her pelvis, abdomen and chest are each turned toward the turn her plan means for it by a spring (2,000 N·m/rad) and a damper (25 N·m·s/rad, the most an explicit damper bears on the pelvis's own inertia at 2 ms steps with a margin of two).
  - **Its caps, a person's.** The force is at most **695.5 N** and never points down: her weight, 608.2 N (62 kg, Harbo, Brincks and Andersen 2012's women's median), and along the floor the most the child can push anything with, its weight times the floor's friction (the G1's 34.39 kg from its model file × 9.81 × the mat's and floor's 1.0: 337.4 N), so |(608.2, 337.4)| = 695.5 N, 1.14 × her weight. Past the cap the correction is scaled down as a whole, her weight carried first. The torque is at most **257.4 N·m** (the three segments' torques' sizes together): her two hips' extension strength (2 × 128.7 N·m, Harbo 2012), what legs hold a pelvis with; the child's steady push at her chest's height (about 0.5 m) asks 169 N·m.
  - So she cannot be toppled by the child, never presses down on anything with more than her own weight (the support never pushes her down), and cannot press more than 337 N along the floor with her trunk. Pushed harder she gives way along the floor, carried upright (parent 22: pushed on her chest at 506 N she slid 1.1 m in 0.9 s, her chest within 2° of upright; pushed down at 1,216 N her support never pulled her down).
  - Her spine's two joints and her neck are stiff (each spends its strength 5° off her plan) and limited by their strength, so her head follows her trunk. The first carried build carried only her pelvis: pushed on her chest at 506 N, her spine (103 N·m) folded 126° back over it; the support now carries all three trunk segments.
- **Her limbs are fully dynamic, their limits enforced** (`make_g1room.parent_actuators`). One MuJoCo actuator per axis of each joint (39: a ball joint's three in its own frame, a hinge's one), a motor whose control range and force range are her strength there per direction; its damping (critical against the joint's rest inertia, MuJoCo's dof_M0) is the actuator's own and inside that range, taken implicitly by MuJoCo's implicitfast while unclamped. No joint damping, armature, friction loss or applied torque of hers acts on her joints. Parent 22 checks every actuator's ranges; parent 23 reads every actuator's force on every physics step under babble against its limits (never over: largest share 1.0). The limits (N·m, the two directions):

| joint group | limits | source |
|---|---|---|
| shoulder | flexion / extension 38.0 / 45.7; abduction / adduction 38.0 / 45.7; rotation 19.0 | Harbo 2012 (abduction isometric, adduction isokinetic); flexion and extension take them, rotation half: ours |
| elbow | flexion 26.5 (isometric), extension 27.2 | Harbo 2012 |
| forearm (pronation) | 6.0 | ours: the wrist's weakest |
| wrist | flexion 14.4 (isometric), extension 6.01; deviation 6.0; twist 3.0 | Harbo 2012; deviation and twist ours |
| hip | flexion 104.4 (isometric), extension 128.7; abduction and adduction 80; rotation 30 | Harbo 2012; ab-, adduction and rotation ours |
| knee | extension 166.6 (isometric), flexion 59.3 | Harbo 2012 |
| ankle | plantarflexion 76.4, dorsiflexion 27.5 (isometric); inversion 20; twist 10 | Harbo 2012; inversion and twist ours |
| trunk (lumbar, thorax) | flexion 64, extension 103; side bend 64; twist 32 | Nordin et al. 1987's ranges' midpoints (ours); side bend and twist ours |
| neck | flexion 16.6, extension 26.5; side bend and turn 16.6 | Garces et al. 2002; side bend and turn ours |

  - Her command is her plan's spring (her strength over 20°, over 5° at her spine and neck), the damping's target velocity, her own limbs' weight (MuJoCo's bias force), her tone (only on an arm she holds still: her trunk is carried and her legs only posed) and her holds' effort (J^T F). Resting (kneeling, standing or sitting where she is) her legs lie where her carried pelvis puts them at a third of that stiffness: held stiffly against the floor under a carried pelvis, a leg pressed it with a leg's strength (0.8–1 kN at the thigh).
- **She never plans a pose that intersects the child, the floor or the furniture** (`parent_motion`):
  - **Her floor lift:** each planned pose is raised until no collision shape of her pelvis, legs or feet is under the floor or the mat's top under it (the kneeling and walking poses were built for a kinematic parent, the thigh's shape 1.8 cm into the mat on her heels); the deepest shape then just meets it.
  - **Her standoff:** every kneeling frame keeps her legs, trunk and head 3 cm from the child when planned (A4), and each tick her planned legs, trunk and head are measured against the child's body and legs where they are now (MuJoCo's own distance); nearer than 3 cm she straightens (10° a step) and then moves her base back off it on the floor plan (3.75 cm a step, never into furniture), at most four steps a tick, her hands' targets on the child left where they are; clear again by 6.75 cm, her standoff comes back 1 cm a tick where it stays clear. Its arms are left out: a hand of hers on its chest is within its arms' reach, as a parent's is.
  - **Her hands' targets** are the child's surface where it is now and her planned clearance (3 mm), re-planned as it moves; contact is made by her hands and forearms at her own limb strength.
  - **Her kneeling down is a person's:** eased in and out over a fifth of its time at each end, timed so her pelvis never exceeds 0.5 m/s nor any other segment 0.8 m/s (KNEEL_TIME, measured on the lifted poses): timed by her fastest segment alone, her pelvis sat back onto her heels at 0.6 m/s and her carried body met the floor at 1.6 kN through her thighs.
  - **The calm step:** beside the child she moves her legs (a step, kneeling down, a shuffle, a turn on her knees) on a tick when none of the child's shapes within 0.3 m of her legs moved more than 2 cm since the last tick; she waits for that at most 3 s in a phase, then goes on.
- **Her contacts:** her hands' palm, fingers, thumb and capsule at contact time constant 0.006 (three steps; MuJoCo's contact stiffness scales with the pair's mass, and a babbling hip sank her fingertip 8.3 mm at 0.02 and 3.2 at 0.006); the rest of her at 0.02, MuJoCo's default (0.05 let a babbling knee sink 46 mm into her thigh). A4's care stays: a chain pressed past its cap for 2 steps stops where it is (for her trunk, her support holds her where she is) and her plan backs off.
- **What it removes:** the first physical build's balance (the still and moving residuals, their dead bands and easing), her gait carried with her floor contact off, her knees' hold on a tall kneel's lean, and the hands' 10 mm test.
- **The tests measure what matters** (parent 23, under babble on 8 seeds at p_rest 0.3 and 0.6, 600 ticks each; parent 16 on the still child): (a) no pain on the child from her by the joints' law (her contacts' and holds' outside torque on each of its joints, the world's truth its born observer estimates, as a 10 ms mean, never past that joint's own limit; nor F_pain on its base or a link); (b) her hands' and forearms' contact forces within ISO/TS 15066's body-region bounds (Table A.2: hands and fingers 140 N quasi-static, 280 N transient; lower arms 160 / 320 N); (c) her trunk and head never touching the child's body and never touching the child for more than her reaction time (2 ticks) in a row; (d) her hands reaching their planned contact on the still child (within 3 cm of the planned grip, touching it) and her touches made under babble; (e) no single step's contact over 1 kN, no contact of hers deeper than her region's compression at its ISO/TS 15066 bound (its Table A.2 force over its Table A.3 spring constant: hands 3.7 mm, forearms 8.0, upper arms 10.0, trunk 11.2, legs 8.8, head 0.87), and her upward force on it never its weight; (f) her work on it: no plan faults, and no contact of her body putting more net energy into it than a centimetre's slide.
- **Measured** (the table): see C8.

**A25c. She passes no contact to the child** (the lead's structural decision of 2026-09-25, after A25b's build; sections 4.1, 4.2, A4, A25b)
- **Why.** Over three physical rounds her kneeling thigh, her shins and her hands met the babbling G1 at up to 1.6 kN (10 ms mean): a person kneeling beside a thrashing child is not a wall, but a MuJoCo body driven toward its plan is one. No softness, standoff or yield closed it.
- **Decided:** her collision shapes touch the room (bit 16), the floor (bit 2) and the toys, never the G1 and never each other (`make_g1room.py`: her conaffinity 18; her palm's touch shapes likewise). Every force she puts on the child is one of her capped springs (a hold, a touch, a turn: `parent_motion`'s holds, applied at the held point within the act's cap, A10) or a toy she holds or sets down. A kick passes through her leg, which the child does not feel; her planner keeps her knees outside its leg sweep.
- **What she no longer does at birth** (`parent_motion.NOT_AT_BIRTH`): the guide, the knee over, the pull to sit and the prop. Each moves the child's body through a movement or a posture it did not make; they are refused ("not at birth") until opened by a later decision, and their tests run them opened.
- **Measured:** the world's 18 tests and the parent's 24 pass with it (on `sim-world`).

**A26. The voice's alphabet: the vocal tract** (section 4.9)
- **Ten articulators,** each taking one of {−0.6, −0.2, 0, +0.2, +0.6} of its range a tick, from where it is (the servo law): 50 striatal rows.
  - The steps were enlarged from ±0.1 / ±0.3, which could not open the lips in one tick. Real lips and tongue tips cross their range in about 70 ms.
- **A rest is silence.** At rest the targets relax, with a time constant of 1 tick, to a silent resting posture: nose breathing, the lips nearly closed, the velum down. The gate's "no" means quiet, as a limb's rest means still.
- **One target per articulator per tick** gives at most 3.3 syllables a second, about canonical babbling's rate. A word takes 4–6 ticks. Nothing sequences sounds inside a tick.
- **Physics, not rules.** The articulators move with damped muscle dynamics (57–120 ms), so sounds glide into each other; breath (400 cm³, about 2.6 s, refilled in 0.8 s) makes breath groups. No phoneme table exists anywhere.
- **Fitted to children's vowels, never to the teacher.** The tongue's corners were fitted to children's vowel means (Peterson and Barney). Nothing was fitted to the parent's words or voice.
- **Not tuned to reach a word.** "bye" was reached by no act tried, and "hi", "pip", "up" and "see" only by hand. The tract stays as it is, and the first 50 words stay as they are. A word it cannot yet say is still one it can understand, and name by tokens while the scaffold lasts.
- **It hears itself at t+1,** through its own ears, about 14 dB above the parent at 1.5 m, with no bone conduction.
- **The silent token output** (effector 1) is a second, separate effector with its own gate.
  - It never talks over her: a token made during her line is read when her line ends, and it never earns a frown.
  - The ledger keeps what is said by the tract and by tokens apart from birth.
- **Deterministic** from its seed and the acts; its sound is not saved.

**A27. The parent's ear: how she recognises the child's words** (section 4.9; `body/sim/parent_ear.py`)
- **It is in the world, never in the body.** It uses the same born cochlea as the child, a shift of 0–4 bands, cepstra c1–c12 and Itakura DTW.
- **Its templates:** the parent's own voice at plain and approval pitch; the same synthesizer at the old child pitch; and 8 other macOS voices (Flo, Sandy, Shelley, Eddy, Reed, Junior, Kathy, Fred). These are other synthesizers, not recordings of people.
- **Fixed before birth: the templates, the babble bank and the expected sets.**
  - The child's accepted productions are never added as templates (`add_template` stays unused in life). Each one added would widen what passes, so the child's own chance acceptances (2% of babble) would loosen the criterion over time.
  - The babble bank is the babbler's 193 utterances, recorded before birth.
- **What she expects** is listed per situation in her constants file before birth, and never widened:
  - the names of what is in the child's fovea or hand;
  - the word of a pending ask, and the focus word of her last line (anything said within 10 ticks of her saying it is an echo, section 4.8);
  - the words of the routine under way: a greeting (hi, mama), a feed (more, bottle), leaving (bye), the game (peekaboo);
  - "mama" while she is away or out of its view.
- **Accepted:** the word is the nearest of the expected words, and its distance is less than the nearest babble's plus 0.35 (m = −0.35). m was set so held-out babble passes 2% of the time. It is re-set before birth for each context set's size (P3, P6), and never loosened after birth.
- **Exact or approximate** (the worth table's two rows, section 4.3):
  - exact: accepted, and also the nearest of all her words, not only the expected ones. It is a right name (worth 2);
  - approximate: accepted only among the expected words. It earns a recast and a smile of 1, until that word has been said exactly 3 times.
  - For the token output, exact is its token, and approximate is letters within the edit distance of section 4.9.
- **Her echo of a babble** ("ball! the ball!") uses the same rule, with the visible referents as the expected words. There is no separate syllable matcher.
- **Its cost:** about 287 ms at an utterance's end, on the world's side. In lockstep that costs no sim time; she replies 3 ticks after the child's turn ends, whatever the wall clock did.
- **Instruments only:** the hand scores and the searched acts were written by us to measure the tract's reach. They never reach the body or the parent.

**A28. The tests of understanding** (the owner's decision 8; section 12)
- **Fixed before birth** (P4): each milestone's probes, the held-out pairs, the novel places and toys, and each chance measure. Nothing is added later to rescue a milestone.
- **The held-out pairs,** with B2's colour twins:
  - the pairs are (red, ball), (blue, block), (yellow, cup) and (green, car). Each held-out object is the one whose colour word is heard only on another toy: "red" on the red block, "blue" on the blue ball, "yellow" on the duck, "green" on the green cup;
  - a pair is tested only once both its words are "understood" alone (the ledger) and each has been heard in at least one other pairing;
  - the line check holds each pair out of every line: templates, Claude's lines, recasts and echoes.
- **The combination test:** "where is the red ball?" with both balls in view; the fovea lands on the red one within 20 ticks and stays 2 ticks. Chance is its share of landings on that twin when she asks with the noun alone ("where is the ball?") at matched moments. "push the red ball" (the owner's example) follows once "push" is understood.
- **One-shot novelty** (a new place for her voice, a toy never shown, a known toy in a new place or upside down):
  - each novel item is a probe only on its first 3 presentations; after that it has been taught;
  - the test pools over the items of a kind: the passes against the sum of each item's own chance (a Poisson-binomial), one-sided p < 0.01;
  - a kind with too few items to reach p < 0.01 at its chance is reported as "not testable yet". It is never pooled with another kind and never loosened (C36).
- **A probe is an ordinary moment to the child.** Its line passes the line check. Her judgment and smile follow as for any ask: she never withholds a smile to keep an item untaught, and never adds one.
- **Only within view.** A probe runs only while its object, place or her face is within the fovea's reach for the child's posture (A22). Otherwise it waits.
- **At most 3 probes of a kind a life day** (section 12).
- **After the scaffold is silenced** (A29), the production tests (M6(c), M6t) count only the tract.
- **Reported** beside each milestone: passed, not yet, or not testable yet, with each probe seen described.

**A29. Removing the scaffold** (the owner's decision 5; section 4.9)
- **Silenced, not removed.** The words channel stays in the anatomy and reads "no token", as on every tick the parent is silent. The token output stays an effector, and the parent stops reading it.
  - No rule waits for the silence. The first draft's rule for "heard speech with no token" is dropped (section 4.9): it gave the words head an articulator reading as a target, and it needed the body to know which sounds are speech. The voice's own demonstration law (R6's rest law on the ears) acts from birth on every unpredicted sound, and the silence changes nothing in it.
  - So the day the scaffold goes silent is a change of the world, not the body, and the child stays seed 1 (A20). A rule switched on at removal would make a new body.
- **The order:** the input first, then the output, each at its own night boundary, each on a copy first.
- **The input** is silenced when the words channel's own forecast, made the tick before each token arrives (from the sound alone), names the parent's word for at least 70% of the birth words, and a copy day with the channel silent keeps at least 80% of the token day's comprehension hits and right names. (The first draft named "the voice's inverse model" here; it reads articulator steps, not words.)
- **The output** stops being read when the input test has passed, at least 10 words reach "says" from the tract alone, and a copy day without it keeps at least 80% of the right names.
- **If a test fails,** the scaffold stays and is tested again every 5 life days after M6 begins. There is no deadline (B17).
- **The letters go with the words:** they are rows of the same channel and effector.
- **After the output is unread,** its gate earns nothing and fades by its own credit. Nothing forces it off.
- **Never brought back** to raise a rate. A fault in the change itself is fixed as a defect, on a copy, at a boundary.

### (B) The owner's calls, with recommended defaults

The nine decisions of the G1 amendment are settled (section 14). These are the environment's remaining shapes, each with the default the design uses until the owner says otherwise.

| # | question | recommended default | what it changes |
|---|---|---|---|
| B1 | The G1's hand cannot hold four toys: the ball, the duck, the bear and the drum. Leave them as they are? And we already made the cup smaller (0.8 of its size, 6.7 cm across) so the hand can hold it: keep that? | Yes to both: the four are for looking at, pushing, rolling, hitting and naming, and the grasp milestone uses the other six. The cup's size was changed to fit the robot's hand, which is the world fitted to the body, so it is your call. | A new shape (a handle, ears) would be measured again in W3. A full-size cup would leave five holdable toys. |
| B2 | May we add four colour twins (a blue ball, a red block, a yellow cup, a green car), so we can test "the red ball" when it has never heard those words together? | Yes. | Four more toys; colour words come in once two toys share a colour (A28). |
| B3 | If the tick is too slow, may its eyes (a) see a simpler copy of its own body, or (b) see the room without the sun's shadow? You would not see (a); (b) takes a piece of the complete reality you asked for. | Neither unless the heat-soaked tick is over 150 ms; then (a) first. If still too slow, we would rather ask again than drop the shadow: a slower life only costs wall time. | (a) saves about 8–18 ms a tick; (b) about 21 ms. |
| B4 | The real G1's two eye cameras see in grey; ours see in colour. Keep colour? | Yes. | On the real robot it would need its colour camera, or to learn in grey. |
| B5 | The model does not say where the G1's microphones and speaker are. Put its ears on the sides of its head and its voice at the front? | Yes, until Unitree's documents say where they are. | The ears' timing and loudness. |
| B6 | Keep the child's voice as built: a child-sized throat, pitch about 265 Hz? | Yes. | What you hear it say. |
| B7 | Nobody can carry the 34 kg child back, so the parent goes to it wherever it is. Keep the whole floor and the open hall as its play space, with no baby gate? | Yes. | Where it can end up. |
| B8 | Close the gap under the sofa, take away the low table's pretend shelf (the lying G1's chest is 1 cm under it), and put lost toys back in their basket each morning? | Yes to all three. | Two lines in the maker; one morning rule in the world. |
| B9 | Should the room echo? | Yes, if it costs under 1 ms a tick; otherwise no, and we report it. | The ears hear the room. |
| B10 | Keep charging as a bottle it holds in its hand, about 5–6 times a day while it moves a lot? | Yes. | Charging by mouth would need a mouth the G1 does not have. |
| B11 | Keep the mat at 2.8 × 2.0 m? | Yes. | — |
| B12 | Your decision 1 lists "looking at it" among what raises her happiness, and also says smiles come only for completed, visible acts. Should the child just looking at her ever make her smile? | No. Looking raises her attention (her eyes and a greeting flash, which the child cannot feel as reward); only its acts earn a smile. A look she asked for (a call answered, "look at the drum") is an act and does earn one. | Otherwise looking would pay. |
| B13 | Keep the names "pip" and "mama", the Samantha voice, and no second adult? | Yes. A better voice is a download you would install yourself. | — |
| B14 | Should she tell where the child is looking from its fovea, as people read a baby's eyes, though a real G1 shows no eyes? (Claude's minute digest reads the same window.) | Yes. | From its head's direction alone, she would name the wrong toy more often. |
| B15 | Keep the living room at 5.2 × 4.6 m? For the 1.32 m G1 that is like a 2.9 × 2.6 m room for a baby. | Yes. More room comes with the house (section 16). | How often it meets a wall or furniture (A24). |
| B16 | Keep the parent as strong as a real person, so she can never lift it, slide it or sit it up? | Yes. It must rise by itself, even if sitting comes late. | Stronger, she could carry it through postures it has not reached (A25). |
| B17 | If the child's own hearing and voice never pass the tests for dropping the word tokens, keep the tokens? | Yes. They stay and are tested again every 5 life days (A29). | With a fixed date instead, it may lose words it knows. |
| B18 | The real G1 feels touch only in its hands. Should our G1 feel touch (and pain) on its whole body, like a baby's skin, though the robot has no skin there? | Yes, disclosed as a sim sense: pain is one of its three rewards and needs it, and an infant has skin. On the real robot it would need an added skin, or contact estimated from its joints' torques. | Without it, touch and pain exist only in the hands, and a blow to the head is felt only through balance and joint effort. |
| B19 | A life day is one simulated hour, so the sun crosses the window in an hour, not a day. Keep that? | Yes. A real-length day would make each life day 24 times longer in wall time. | How fast the light changes. |
| B20 | The real G1's motors hum as they work, and its microphones hear it. Add that sound, growing with each motor's effort? | Yes, if it costs under 1 ms a tick; otherwise no, and we report it. | It hears its own effort; nothing rewards it. |
### (C) Open until measured

| # | open question | measured in | what decides it |
|---|---|---|---|
| C1 | Which child | — | settled: the stock G1 (the owner's decision 7) |
| C2 | Whether the face test's rays agree with a segmentation render inside the software fovea | W3: 2,000 babbled frames, the parent at 0.3–3.5 m | at least 95% agreement; below that, the render is used |
| C3 | Whether the fovea tells the toys and the face apart at 1.5 px a degree, under morning, midday and dusk light; the born face template's hits and false alarms in the fovea (section 3.4) | W3's eye check | identity at least 0.75; below that, the eye's own born constants (the fovea's size and code) are reconsidered before birth. The light and the shadow are never changed for it (5.4, B3) |
| C4 | Rolls under the final servo law on the G1, from the born body's own motor timing (C38), not the per-joint babbler | W4 | written down as chance; never tuned |
| C5 | Phantom pain from contact spikes, with and without the 10 ms filter | W4 | pain on still ticks under 0.01% |
| C6 | How many fall starts the catch stops on the G1, including falls driven by the child's own big steps (the earlier "about 2.4 J at 200 N" has no source; the upper body releases about 7 J from 35° to 50°, A9) | W2 and W4 | at least 95% caught; below that, the hover distance or the trigger angle changes before birth, for the stated reason that an uncaught fall hurts. The 300 ms reaction is a person's and never changes |
| C7 | The pull-to-sit with the child's help under the caps; the guide's cap for each limb; the guide's pace on the limp arm (a 20 cm path in 1.2 s peaked at 139 N); the brief turn from its front, completed within 2 s at 200 N (A25) | W2 | no hold at its cap for more than 2 ticks, no act over the caps, and a limp arm guided at no more than 65 N |
| C8 | Peak forces when the babbling G1 meets the soft, yielding parent; her kneeling spot outside its leg sweep | W2 and W4 | under her own 150 N threshold on most ticks, and under F_pain always |
| C9 | How often a babbling child answers a call by chance | P6, with the babbler | written down as chance |
| C10 | How often babble touches the bottle beside its hand | W4 | written down |
| C11 | The amygdala: its reliability per head, the night's draw against surprise alone, the morning drift, its speed on the body's own stream, and the share of episodes with T_e ≥ 1 (if most carry a smile, "tagged first" fills half of every night) | R7's plumbing run, then the first life days | its cost measured at 0.19 ms; the rest reported daily; the constants of A16 never tuned to these |
| C12 | The tick, heat-soaked | S5b | a mean of at most 150 ms over 2 hours |
| C13 | The store's write rate, and the capacity it needs | S5b | room for 19 life days of writes |
| C14 | How long a night takes (R8) | S5b | reported |
| C15 | Recognising words across voices, for removing the scaffolds; the tract's words in the parent's ear | P1–P3v, then after M6 begins | the tests of section 4.9 |
| C16 | Credit across the delay (risk 1) | the first life days | `act_pred` against `act_inv` on held-out guides; the value rising at the touch |
| C17 | Balance at a 150 ms tick: the leaning sit under the resting law | W1 and W4 | M5's range |
| C18 | The G1 on the mat: the sink rates under the resting law, the pain rate at 1,012 N, travel and time off the mat | W1–W4 | written down before birth |
| C19 | Toys lost each day, if the sofa's gap stays | the first life days | reported |
| C20 | Free disk for birth, and before every verification copy; the save's real size | before S5b; before each copy; S5b | at least 8 GB left after the copy (A18); the disk rule's floor and birth's need re-set from the measured save |
| C21 | Whether being held is felt: the hold's force in the held link's touch | W1–W2 | the touch channel shows every hold; every hold far under F_pain |
| C22 | The flexor withdrawal on each G1 limb, and whether it ever drives a limb into a worse contact | W1, then W4 | written down; no withdrawal that raises the pain it answers |
| C23 | The tick's parts not yet measured: per-step contact forces for touch and the pain filter, the face test's rays, the parent's ear at the babble rate | S5a | inside the 150 ms mean |
| C24 | The parent's contingency: the child's acts answered within 7 ticks | P6 | at least 90% |
| C25 | SSML's effect on per-word prosody, and the parent's word rate | P1 | at most 3 words a second on new words |
| C26 | Real frictions (the world's surfaces 1.0 now; the G1's feet 0.6 at priority 1, as shipped) and a friction model that does not creep (the G1 crept 5 cm at 146–199 N) | W1 | the measured sliding force matches μ × weight, set through the world's geoms at contact priority 2 and world options only; the G1's file and geoms untouched (A21) |
| C27 | The parent's ear in life: its cost at the babble rate, its false accepts, m at the real context sizes; how often an accepted word is also exact (A27); whether the fixed babble bank still rejects babble once the child's babble has changed | P3v and P6, then the first life days | chance written down before birth; the rule never loosened after it |
| C28 | The grasp on the six holdable toys under the resting servo law, and B1's choice | W3–W4 | written down |
| C29 | Early imitation: the first accepted echo | the first life days | reported (M6t's precursor) |
| C30 | Hits on the parent per hour under babble | W4 | written down; her withdrawals and "oh!" counted |
| C31 | The G1's falls under the final servo law: strike forces from the sits, from rolls and from a fall off its elbows, against F_pain (the custom child's head struck at 260–890 N at 9.45 kg) | W4 | written down; the pain law is not changed by them (A12) |
| C32 | Whether the lying G1 can get its trunk under the low table (head first, between its legs), and whether it gets itself out | W4, with B8's choice | written down; if it can wedge, the owner is asked about a closed base for the table before birth |
| C33 | How often the VOR's quick phase fires during trunk turns (A23) | W4, with the born VOR | reported only; the quick phase is decided on biology (A23), never by this rate |
| C34 | Whether the parent can put her face where the child's eyes reach in each posture (A22): leaning over its chest; kneeling toward its feet outside its leg sweep; low and ahead of a sitting child | W2 | a pose inside human ranges for each; where none exists, she never expects a look there |
| C35 | What the voice's demonstration law (R6's rest law on the ears, section 4.9) labels from birth: the parent's speech, the toys' sounds, footsteps; how reliable `act_inv`'s labels are on each | R6h and the plumbing run (S5a) | written down; the law is not changed by it. (The first draft's rule for "heard speech with no token" is dropped, A29) |
| C36 | The tests of understanding: how many items each kind of never-taught test has before its items are taught, and whether that can reach p < 0.01 at its chance (A28) | P4 (the items listed), then the first life days | a kind that cannot is reported "not testable yet"; more items are added only before birth |
| C37 | The real G1's damage limits per link (Unitree's documents) | before the robot | if any link is damaged below 1,012 N, the next body's pain law takes that limit (A12) |
| C38 | The born movement units on the G1: their lengths under the gate's continuation draw at the born p_act (about 0.29, so about 1.4 ticks on average), and the rolls, travel and reaches they give (section 3.6) | R6h, then W4 on a replica and at S5a on the real core, every learning rate 0 | written down as chance. If the units are far shorter than newborns' movements (general movements last seconds), the unit's born length is decided before birth on that biology, as a disclosed constant; never on a roll count, and never after birth |
