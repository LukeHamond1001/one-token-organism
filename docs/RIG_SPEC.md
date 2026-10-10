# THE RIG: the same life on the real G1 (stage 1 of the play; written 2026-10-09, 21:20, owed since 2026-10-09 morning)

The body's contract with any world is `body/core/world.py` `World`: `frame()` shows the world as it is, the body lives the tick on it,
`apply(acts)` takes the body's acts and moves the world on by a tick (150 ms); `sub_tick(SubFrame)` is called every 10 ms inside the tick
for the loop below it (the cerebellum); `save_state` / `load_state` keep the world's state beside the life's; `dusk()` / `dawn()` for a
world that runs through the night. The sim implements it as `G1World` (`body/sim/world.py`). THE RIG IS A SECOND IMPLEMENTATION OF THAT
CONTRACT, `RigWorld`, whose physics is the real robot. Nothing in the brain, the anatomy (`body/sim/anatomy.py`), the reward or the sleep
changes: the life that lived in the sim is loaded against the same anatomy and goes on. What follows is what `RigWorld` must do, piece
by piece, what runs where, and what we do not know until there is a machine.

## 1. The machine

- A Unitree G1 with the Dex3 hands (the sim's body is this file: `assets/unitree_g1/g1_with_hands.xml`), its head D435 and both IMUs.
- Its onboard computer for the loop below the tick (100 Hz) and the safety layer; a laptop beside it for the brain (7 Hz), the record
  and the saves, on a wired link. The brain's tick is 150 ms of work on this laptop; the 100 Hz loop must never wait for it.
- An overhead tether or gantry rated for the body (34 kg) with a load cell, and an e-stop in the owner's hand. The tether is the ghost's
  catch; the load cell is the strength's `share carried`.
- A room: a mat, toys the sim's names know (ball, block, duck, cup, rattle, car, bear, stacker, drum, ring, book, box, bucket), a sofa, you.

## 2. Senses in: the frame, channel by channel (the sizes are the anatomy's, `SIZES`)

| channel | size | the sim | the rig |
|---|---|---|---|
| `eye_p`, `eye_f` | 172, 1,536 | the D435's three imagers rendered, then `body/sim/eyes.py` in software: periphery cells, the fovea's born bank at the gaze window | THE SAME SOFTWARE on the real D435's two infrared imagers (336 x 192 after a crop/scale to the sim's field) and its colour camera; the gaze is still a software window (the robot has no eye muscles), the VOR counter-turn from the real head gyro. The real imagers see near-infrared the render did not (a disclosed gap closes) |
| `ears` | 1,725 | sounds placed in the room, two cochleas | two microphones at the ear sites, the same cochlea code on the real pressure; her voice is your voice |
| `body` | 242 | per joint sin, cos, velocity, effort (torque over the limit), temperature; the gaze's state; the tract | the joint encoders and motor currents from the SDK at 100 Hz, the tick's means; the gaze's state is the software window's; the tract's 21 numbers are the voice synthesizer's state as in the sim (the voice stays software, played through a speaker) |
| `touch` | 130 | the Dex3 arrays' 16 zones (log force, onset); the observer's outside torque per joint; the base's outside wrench | the Dex3's real tactile arrays; THE OBSERVER (`body/sim/observer.py`, the momentum observer from the robot's own sensors, A37) run on the real currents and encoders, as it was built to be; the base wrench from the pelvis IMU and the leg joints as the observer gives it |
| `vestibular` | 24 | both IMUs, mean and peak | the real torso and pelvis IMUs |
| `face` | 2 | the parent's graded face read by the world (A1's scaffold) | a camera on you, a smile reader giving `2 x (smile - frown)`; until the born face detector on the pixels works this stays the scaffold it is in the sim, disclosed |
| `words` | 1 token | the parent's word on the tick its sound ends | a speech recognizer on your voice, the lexicon's 79 symbols; the born table is the same |
| `pain` | 44 flags | the observer's outside torque past the joint's limit; the base's past 3 x the weight | the same rule on the real observer, with the limits set in section 5 |

The born cues beside the channels (`face_periph`, `sound_side`, `onset_periph`, `imu_torso`) come from the same software on the real
signals. The event lines are the anatomy's and do not change.

## 3. Acts out: the effectors

The brain's acts are per-joint steps in five settings (`SETTINGS`: -0.27, -0.09, 0, 0.09, 0.27 rad a tick) for the waist, arms, hands and
legs, the gaze's three settings, the tract's ten and the words' output. The sim's servo law re-anchors each joint's target to its measured
angle plus the step and lets the motor's PD pull it there within the joint's declared torque limit (`body/sim/world.py` `_set_servo_law`).
The rig does the same through the SDK's low-level joint command (target angle, kp, kd, feed-forward torque): the target is angle + step,
kp and kd set to the sim's servo gains per joint (the sim's gains are soft on purpose: the real motors can do more, and the gains are a
body constant to carry over, not raise), the torque limit the SDK's per-joint cap set to the anatomy's limits. The gaze moves no motor.
The voice is the sim's tract synthesizer through a speaker; the words' output is silent as in the sim.

## 4. The loop below the tick: the spinal layer runs on the robot

The 100 Hz loop in the sim (`apply`'s 75 physics steps with a `SubFrame` every 5) carries: the postural tone and the vestibulospinal reflex
(`reflexes.posture_step`), the standing and stepping reflexes (`reflexes.stand`), the grasp and dorsal reflexes on the hands' touch, the
withdrawal on pain, the traction response, the prone pattern, the tendon organ's inhibition, the resting tone of the arms, and the
cerebellum's torque (`body/core/cerebellum.py` through `Below`). On the rig these run on the robot's computer in one 100 Hz process that
owns the SDK's low-level command: it holds the last acts from the brain (7 Hz), re-anchors the targets, adds the reflexes' offsets exactly as
the sim's loop does, writes kp, kd, targets and the cerebellum's feed-forward torque, reads encoders, currents, IMUs and the tactile arrays,
runs the observer, and streams the 10 ms samples up to the laptop, which assembles the tick's frame (means, peaks, onsets) as `_sense_tick`
does. The cerebellum's `SubFrame` (mossy input, the servo's corrective torque as the teacher, the limits) is built on the robot and its
answer (the torque) applied there; the laptop holds the weights and ships the updated readout down each tick (its lesson is least mean
squares on a small matrix: this is cheap). A brain tick that arrives late changes nothing: the loop keeps the last acts and the reflexes,
which is what the sim does when the acts are rest.

## 5. The pain law on real motors: the safety layer

In the sim, pain is the observer's outside torque past a joint's declared limit (`tau_max`, the gear's holding figure) and the base's
force past three times the weight; the servo clips torque at the limit, the tendon organ relaxes a loaded joint for two ticks, the
withdrawal flexes a hurt limb. On the rig the same numbers are set from the real motors' continuous ratings (the G1's joint torque limits
in its own description), a safety margin under them (ours: 0.7 of the rated continuous torque is the pain line, the SDK's hard cap at the
rated peak), and three things the sim never needed: a motor temperature cutoff (the body channel carries temperature already), a
joint-velocity cap, and a watchdog that drops to damping mode if the 100 Hz loop misses three frames. The tether's load cell is read as
the base's outside wrench (the harness holds it: the trunk's touch zones, as her hands were) so the postural tone's share eases under a
real hold exactly as it did under hers (`_tone_w`). The e-stop cuts motor power and leaves the tether holding.

## 6. The ghost on the real body: Unitree's own controller, by the strength

The sim's ghost (W7, `body/sim/ghost.py`) holds the torso, rights the pelvis and, with the expert (W7b), walks the legs with Unitree's
pretrained policy by a strength in [0, 1]. On the real G1 that policy is the robot's own locomotion mode. The rig's ghost: the tether's
hold (the strength sets the harness's slack through a winch, or a person on the rope by the number on the screen at first), and the
locomotion controller's joint targets blended with the brain's by the strength at the 100 Hz loop, exactly as `_expert_step` blends them in
the sim (targets drawn toward the policy's by s; no feed-forward torque needed where the robot's own PD is stiff). The fade law is the
same (weight borne, read from the tether's load cell and the soles' estimate), the catch is the tether, and the lay-on-the-mat is you.
The record keeps the strength every tick as it does now.

## 7. Day and night, saves and the record

A life-day is 24,000 ticks (an hour of real time at 150 ms); the night is 24,000 more with the eyes and ears off: the robot in damping
mode lying on its mat, powered, the loop below the tick running the night's frames with the body's own senses, as the sim does
(`live_night`). The life's save (`life.pt`) and the world's state (`RigWorld.save_state`: the gaze window, the tract, the ears' delay
lines, the observer's state, the tick, the ghost's strength and edge) are written together every 1,000 waking ticks and at dawn, as now.
The record (`ticks.jsonl`) keeps the same rows plus the tether's load and the e-stop's state. A landing of code at a checkpoint is the same
procedure as now, with the robot in damping mode during the restart.

## 8. What we do not know until there is a machine (the honest list)

1. THE BODY MODEL IS WRONG. Everything the brain learned about its body is about the sim's: contact, friction, motor response, the
   observer's noise. Expect the first real days to look like the first sim days of the body change (the pre-robot rehearsal the owner
   planned, 2026-09-28): the reflexes carry it, the policy relearns. If relearning is slower than a new seed, start a new seed on the real
   body: the sim validated the architecture, not the weights.
2. THE SERVO LAW. The sim's soft gains were chosen for the sim; the real motors' PD at those gains may not hold the body the same way.
   Measured on the bench, joint by joint, before any life tick.
3. THE OBSERVER ON REAL CURRENTS. The momentum observer was built to need only the robot's own sensors; its noise floor on the real
   machine sets the touch onsets' threshold, which is the event lines' threshold. Measured on the bench.
4. THE EYES. The real imagers' exposure, noise and near-infrared change the periphery and fovea statistics; the brain has never seen
   real pixels. Expect the naming to drop and come back.
5. THE FACE READER. A smile reader on a camera is a scaffold with a different error than the sim's graded face. Disclosed, measured
   against your own report of what you did.
6. TIMING. The brain's tick on the laptop is 140 to 300 ms of compute for 150 ms of life. In the sim the clock is the sim's; on the rig
   the clock is real, so the brain must finish a tick in 150 ms or the loop below runs on the last acts. A GPU for the brain is the
   simplest fix; if there is none, the tick stretches to the brain's time and the body lives slower than real time, which the
   architecture allows (the loop below is what keeps it safe).

## 9. The order of work

1. Bench, no life: the 100 Hz process on the robot with the SDK; the servo law and the reflexes on one leg hanging; the observer's
   noise floor; the pain line's trip tested with a hand pushing a limb. The tether rigged and load-tested.
2. Bench, the senses: the frame built from the real sensors and compared channel by channel with a sim frame of the same pose
   (sizes, scales, onsets).
3. Standing under the tether with Unitree's controller at strength 1 and no brain; then the brain in the loop at strength 1; then the
   fade, under the tether, in the room, with you.
4. The first real life-day, recorded; its rulers against the sim's days; then the life goes on there.

The interface is the one the body already has. The work is an implementation of `World` whose physics is a robot, a safety layer the sim
did not need, and a tether. None of it costs the sim life a tick; all of it is testable on a bench before a single life tick is lived.
