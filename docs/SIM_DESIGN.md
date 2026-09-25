# The first simulated body: the stock Unitree G1 on a play mat (design and plan, 2026-09-24)

Status: a design whose first parts are built, verified and measured on branches, not yet merged into main (section 0's build status). The stock G1, the parent and the living room are a MuJoCo scene in `body/sim/` (section 17). The world (W1, W3) is built on `sim-world`, the parent's voice and the child's ears (P1, P2) on `sim-parent`, and the core refactor continues on `sim-core`. The language body is paused.

This amends the all-out design written this morning with the owner's nine decisions (the G1 amendment; section 14 lists each and where it lands). The custom child that design built is kept for reference as `body/sim/*_customchild.*` (section 15). The one-arm high chair of 2026-09-23 is kept as `body/sim/highchair_onearm.xml` and its maker; its document is in git history (commit e144afa).

It is amended again, on the evening of 2026-09-24, by the lead's decisions on the owner's bar: "robot need to be like we put human brain in g1 and sim is reality", and an architecture worth billions when all is complete. The owner's standing rule is to decide rather than ask, so the lead decided for him and reports; he may overrule any of it. Each decision is in the decision log with its reason and source (A36–A57). They answer B4, B14 and B18, reopen A23, reverse the refusals of the stepping generator and the born cry, and add a cerebellum (R6c). The audits they answer are in `docs/audit/`.

Later that evening it is amended once more, on the owner's word "humanoid with our human architecture going in sim training; we need it at learning speed and understanding level of human". Two decisions answer it, taken for him in the same way (A58, A59). The human-pace ledger (section 12) reports each milestone against an infant's waking hours to the same capacity; it is a ruler, never a target. The speed plan (section 9) says how fast the life can run without changing anything the body senses or does. Spending on a rented machine is the owner's call (B21).

On the morning of 2026-09-25 the owner set a new bar: "only local no cloud. i dont care about huuman pace i just want fast learninhg and for it to solve robotics that no archecture had done before". The lead's decisions answer it in the same way (A61–A66; A60 is the parent session's, on `sim-parent`), and the owner may overrule any of them:
- the life runs on this Mac only, with no cloud or rented machine and no spending, ever (his word; B21 answered, A61);
- fast learning is the body's sample efficiency, counted in its own life hours, plus the fastest local run that changes nothing it senses or does (A62);
- the firsts are named in section 12, each with its bar, the closest published work and what delivers it, and registered before birth (A63);
- the local speed levers have their build steps (section 9, A64), and B3 is answered: a lighter copy of its own meshes for its eyes, while the colour camera keeps its own view and the sun's shadow stays (A65);
- the human-pace ledger is now an optional report, still behind its firewall (A66).

Sources:
- the all-out studies (the world, the core at this scale, learning to move, the parent's language);
- four studies run today for the amendment, on this Mac: the G1 in the room, the parent's feelings and its limits with the G1, the vocal tract, the amygdala;
- the branch's commits;
- the first build's reports and its verifiers' verdicts (W1 and W3, P1 and P2, R6 fix 3), folded in on the afternoon of 2026-09-24;
- the audit against the owner's bar (`docs/audit/`: the roadmap, the brain systems map, the reality gap, understanding, value, and the skeptic's verdicts on them), and the lead's decisions on it (A36–A57);
- the firsts research and its skeptic's review (the morning of 2026-09-25, `$S/firsts/`), and the lead's decisions on the owner's new bar (A61–A66).

Section 17 lists the files.

## 0. The answer in brief

**What we build.** The stock Unitree G1 humanoid, unchanged, born lying on its back on a play mat in a furnished living room. A human-shaped parent raises it, and the parent's face shows the parent's feelings.
- The child must learn everything it does: looking, rolling, reaching, grasping, sitting, and its first words, spoken through its own vocal tract.
- **It is the real robot's model:** 43 joints with the Dex3 three-finger hands, 34.4 kg, 1.32 m tall, and no neck.
- **Its senses are the real G1's, sensor for sensor** (A36): the head's RealSense D435 (a grey stereo pair, with colour from the colour camera beside it, A38), microphones, two inertial units (with a drifting gyro, A39), each motor's angle, velocity, estimated torque and temperature (A39), and the Dex3 hands' tactile arrays (A37). Nothing is added to its body. A software fovea moves inside each camera image, and a software estimator reads contact from the joints' own efforts (A37).
  - **No sense goes beyond the real robot** (B18, answered by A37). Touch is felt where the real G1 feels it, in the Dex3 hands. Elsewhere, contact and pain are estimated from the joints' efforts, as a robot's collision detector estimates them (Haddadin et al. 2017). This replaces the sim skin that covered the other 28 links.
  - **So the real G1 is a change of world for seed 1, not a new body** (A20, A36). The scaffolds are the world's and are removed by their tests (the word tokens, A29; the world-truth smile, A49).
- **It has no balance, posture or walking controller.** Its innate mechanisms are these, and no others:
  - the servo law with tone (the muscle and stretch-reflex model), applied to the robot's own motors at load, with the gains Unitree publishes for them (A39);
  - the tract's resting posture (a rest is silence, A26);
  - weakness when empty;
  - the movement units' persistence margin;
  - the software fovea's conjugate and vergence coupling, and its born centre-surround and oriented bank (A42);
  - the ears' brainstem lateral read;
  - the contact estimator on the joints' efforts (A37);
  - the amygdala's born event lines (what it learns from them is learned);
  - the cerebellum's born expansion, whose weights are learned below the tick (A44);
  - a spinal pattern generator per limb, summed at the cord with the limb's own act (A48);
  - the gates' two disclosed drives: a tonic drive that follows the reward rate, and the tract's performance error (A41);
  - six disclosed reflexes (withdrawal, grasp, orienting to faces, voices and sudden visual change, the VOR with its quick phase, a born cry, brainstem twitches in sleep) and the born expression reading (section 3.7).
- **The parent** moves by scripted inverse kinematics. It has a small scripted state of feeling shown on a graded face of human proportions, and answers the child within about a second, imperfectly, as a person does (A52). It follows the child's attention as a person could read it (its trunk and hands, A40), speaks infant-directed sentences in varied voices (A50), scaffolds, copies its movements, and keeps routines. Claude steers what it teaches once every 400 ticks (A59): the one input to the life from off this Mac, logged by tick (A61).
- **Reward** comes only from the parent's face, and only while the child looks at it, plus pain and a charge need. The face is read by the child's own eyes once its born face detector works; until then it is read from the world, a disclosed scaffold with its removal test (A49).

**What already exists** (built and measured today; section 17 lists the files):
- **The G1 room.** `body/sim/make_g1room.py` includes the G1's file unchanged: it is byte-identical to the commit. The senses are added at load as cameras and sites only.
  - Under babble the physics runs at 10.9–12.2× real time.
  - The two eyes cost about 16 ms a tick without shadows and 38 ms with the sun's shadow, from render to fovea.
  - In the built world (W1 and W3, on `sim-world`): 15–18 ms a tick under babble (8.4–10× real time), and 48–53 ms with both eyes and the sun's shadow (2.8–3.1×), before the parent's face of human proportions made the eyes' render about 25–30% slower (section 9).
- **The parent's acts.** All 22 passed inverse kinematics inside human joint ranges around the custom child. Aimed at the G1, the parent kneels beside it, rests a hand on its trunk, shows a toy 40 cm before its eyes, and guides a forearm.
- **The parent's feelings and graded face** (`parent_feel.py`). Its self-test shows that nothing but a judged act can ever be felt.
- **The vocal tract** (`tract.py`). It babbles and hears itself. Eight of the nine target words can be said in a way the parent's ear accepts in context.
- **The amygdala** is specified and measured on a synthetic stream: 0.19 ms a tick.
- **Stills** are in `video/sim_look_g1/` (the parent's old face; redrawn after the merge).

**Build status** (2026-09-24; each part on its own branch, not yet merged into main; each checked by a separate verifier who re-ran its tests, the eight language digests and probes of their own):

| part | branch | commits | tests | verifier |
|---|---|---|---|---|
| W1, the world (`world.py`, `reflexes.py`, the room), and W3, the eyes (`eyes.py`), with the parent's face of human proportions (`parent_face.py`) | `sim-world` | 1ac403d, bd87dcc, 3c3b639; the merge of main b85abd3; fixes d1e35b7, 6003014, f29bb36 and 7f40cc2, b9b61bd and df05f4c | `test_sim_world` 18/18, `test_sim_eyes` 16/16, threaded and pinned | READY at df05f4c, after four rounds that each found blockers |
| P1, the parent's voice (`voice/`, `lang/lexicon.py`), and P2, the ears (`ears.py`), with the child's tract (`tract.py`) | `sim-parent` | 2e5e68a; fix 0f257d1; the merge of main 7b93e36; fixes 87a4194, 06b85b2 | `test_sim_voice` 13/13, `test_sim_ears` 18/18, threaded and single-threaded | READY at 06b85b2, after three rounds that each found blockers |
| R6 fix 3 (`act_pred`'s plasticity gated by its labels' reliability, `GatedAdam`) and the verifiers' three non-blocking findings | `sim-core` | e48b284, 6d6d246 | organ and anatomy tests 123/123, threaded and pinned | NOT READY: the gate holds only for a newborn body (section 3.6); R6 fix 4 answers it |

- Every language digest equals its pin on every commit above, and no language-body file changed.
- In verification, and not folded here: R6 fix 4 on `sim-core` (440bad3) and P3, the parent's fast layer, her ear and the ledger, on `sim-parent` (5ea4c4e). Each verifier's first look found blockers.
- What each build changed in this design is folded into the sections below; the questions its fix rounds decided are A30–A35.

**What the measurements changed:**
- **A person cannot move a 34 kg body the way a parent moves a baby.** A person's hands give about 160 N with both hands, or 200 N for up to 2 s.
  - Within that: holding the G1 once it is propped near upright; guiding a limb that yields; a brief turn onto its side.
  - Beyond it: sitting it up from lying (217–237 N), sliding it on the mat, lifting it (its weight is 337 N).
  - So the parent comes to the child, and the child's own acts must carry every rise (section 4.2).
- **Two of the ten toys cannot be held** by a Dex3 hand: the bear (2 of 9 tries) and the drum (0 of 9). The first measurements' four (the ball, duck, bear and drum) were MuJoCo's soft contacts creeping at impratio 1. At impratio 10, chosen by the contacts' physics (A32), the ball holds 9 of 9, the duck 7 of 9 and the cup at full size 9 of 9. The cup, shrunk to 0.8 for the creep, stays at 0.8 until the owner answers B1.
- **Sitting and crawling.** Sitting upright with the legs out tips backward; the G1 sits only leaning forward with its hands on its knees. On hands and knees, flat palms need 4× the wrists' 5 N·m; on fists it nearly holds.
- **The solver.** The prototype's settings crash MuJoCo 3.9 when a Dex3 hand grasps. The elliptic friction cone with multi-point collision detection off ran 360 grasp trials and all the babble clean. With impratio 10 the soft contacts' creep nearly stops (a box pushed below its sliding force slid 3.6–21 mm in 2 s at impratio 1, 0.4–1.8 mm at 10); the cost is that a pressed housing that slides reads harder (section 5.1).
- **Pain.** With the 10 ms filter and the newborn's withdrawal, babble is in pain on 6–12% of ticks (11.6% over 8 seeds × 400 ticks; 6.1% on a verifier's 4 seeds), mostly the G1's own housings pressed together: the thighs into the pelvis, the shoulders into the torso. At rest, none. No self-contact presses at rest, so every one is felt (A12). These figures were measured on the sim skin, which the lead's decisions remove: pain is now estimated from the joints' efforts (A37). W4 measures it again on the born loop, and counts how much of it is the collision hulls meeting rather than housings: 35% of the contacts over the old threshold came with every hinge more than 0.2 rad from its range's ends (A54).
- **The born face template barely sees a real face.** The parent's face was rebuilt to a real woman's proportions and a real face's photometry, and the template was left as it was (A30). It detected her face in 1 of 48 fovea readings at 0.3–2 m, and never in the periphery; its other matches were chance, as strong on the image upside down. As built, the event line "a face in the fovea" and orienting's face cue would run on chance. The template's constants are settled from its own sources before birth, by sim-face's study, on the fovea's new code at a newborn's acuity (C39, A42).
- **The withdrawal is a newborn's, and crude.** The generalized flexion raises the pain it answers on 26% of onsets, against 19% for resting and 48% for the babble's own acts. That is written down as the newborn's (C22); a learned tuning is an open item (C40).
- **Imitation starts near chance.** An inverse model learned from the child's own babble turns the parent's words into echoes the parent accepts for only 2 of 12 words. The parent's ear works only by listening for the words it expects, against a bank of the child's own babble.
- **The tick grows.** The G1's physics, its two eyes, the tract and the parent's ear add about 15–30 ms, and the parent's face of human proportions about 8–11 ms more in the eyes. The tick is now about 105–160 ms mean with the sun's shadow in the eyes. The lead's decisions add about 2–15 ms (estimates), and the colour camera, a third view to render, about 10–15 ms more: about 117–190 ms, past 150 ms at its upper end (section 9, A57). The exact levers and the lighter meshes for its eyes bring it to about 65–135 ms, by estimate (A64, A65).

**The core.** The refactored core on `sim-core` serves this body at the planned size: d 512, 6 blocks, about 35M parameters.
- R1–R6, R9 and R5b are committed, with R6 fixes 1, 2 and 3 (e48b284: `act_pred`'s plasticity gated by its labels' reliability) and the verifiers' three non-blocking findings (6d6d246). Every language digest held. R6 fix 3's verifier found that the gate holds only for a newborn body; R6 fix 4 (440bad3) is in verification (section 3.6).
- About 13 working days of core work are left: R6h, R7 and R8 with the lead's additions, and a new step, R6c, the cerebellum (section 8). The amygdala is part of R7 and R8.

**When it is born:**
- About working day 17 (range 15–22) with three build sessions in parallel: the core; the world and the G1; the parent, its voice and ears, and the child's tract. The lead's decisions moved it from day 13 (section 11, A57).
- About day 24 with two sessions, about day 41 with one.
- The disk has 18 GB free (the afternoon of 2026-09-24). One verification run filled it in the morning, so copies must stay small (risk 11). Birth needs at least 8 GB free (section 9).

**What a viewer can watch first:**
- Life days 1–3 (about the first 1–5 wall hours): its fovea, then its trunk, turn to the parent's voice and face, and it keeps the smiling face in view.
- Life days 5–25: its first touches of a toy held over its chest.
- Life days 15–60: rolling toward the parent on purpose.
- A life day with its night takes about 65–110 wall minutes as designed (55–90 before the lead's decisions: the tick grows, and the night is live), and about 38–79 with the exact levers and the lighter meshes for its eyes (estimates; section 9, A64, A65).
- Sitting alone, crawling, standing and its first word from its own tract are not forecast (section 12).

**What it costs.** This Mac only, with no cloud, ever: the owner's word of 2026-09-25 (A61; B21 answered). No pod, no rented machine, no GPU training and no spending. Claude acts once every 400 ticks (A59), never inside the tick, served on the owner's plan: its steering row is the one input to the life from off this Mac, and every row is logged by tick, so the life replays on this Mac alone (A61). Nothing here loads the language body's save.

**Fast learning and the firsts** (A61–A66; section 12):
- **Fast learning is sample efficiency** (A62): the life hours the body needs to reach each milestone (a life day's wake is one hour), the exposures it needed on the way (smiles felt, asks met, her guides, namings), and what a night adds. Beside it goes the fastest honest local run: the wall time a life hour takes on this Mac, with nothing changed that the body senses or does. Other methods' published sample counts are context, never a run or a target, and no "N times faster" figure is printed.
- **The robotics first is the setting, not a task** (A63). Each skill alone has been learned faster by some method, from a signal built for it. None has shown them together: in one life with no resets, from sparse feedback like a person's, with no reward built for any task, still learning afterwards, in a real robot's whole body.
  - **First 1:** from randomly initialised learned weights and a disclosed innate set, the stock G1's model learns to orient, reach and grasp (and, for the full claim, to roll on purpose), each within 60 life hours, from a scripted parent's sparse smiles, felt only while it looks at her. Her guides are counted and Claude's steering of her curriculum is disclosed beside it. Level 2 holds the skills on the smile read by its own camera, and it keeps learning for 200 life hours without losing them.
  - **First 2:** it passes at least 3 of 4 infant tests of understanding it was never taught (the ball behind the table, the toy under the cover, her silent head turn, a look to her at a new toy), each failed by its own born state.
  - Every bar is frozen before birth and reported whether it passes or fails. No life has run: every hour here is a bar, not a result.
- **This Mac, local only** (A61, A64, A65): about 0.8–1.3× real time as designed, 1.0–1.6× with the exact levers, and about 1.1–2.3× with the lighter meshes for its eyes too, all estimates (sustained speed on the fanless Air is unmeasured, C12). Running around the clock, First 1's 60 life hours then take about 1.6–3.3 calendar days and its 200 hours about 5–11, stretched by the owner's own use of his laptop.
- **The human-pace ledger** (section 12) stays only as an optional report the owner does not need: frozen before birth if kept, read by nothing, and never a headline (A58's firewall, A66).

**Understanding.** The owner's goal is a human brain architecture that gains understanding. Every milestone carries a test by something never taught (section 12): a voice from a place the parent never called from, a toy never shown, known words in a combination never heard, a word used to get something.

**The honest risks** (section 13 says how we see each one):
1. **Credit across a delay.** The smile is felt about 7–25 ticks after a reach begins, partly outside the 12-tick eligibility window.
2. **Balance at a 150 ms tick, in a stiff body.** The G1 cannot sit upright, and under the resting servo law every posture sinks. A limp upper body would fall with a time constant of about 0.19 s, near the tick.
3. **Motor milestones by chance only.** Rolls under babble are not yet measured on the G1. The born body's movement units are also shorter than the babbler that justified them: its gates continue a unit on about a quarter of ticks, so units average about 1.4 ticks, not 4.5 (section 3.6).
4. **The parent's strength.** A person cannot sit up, slide or lift a 34 kg body, so the child's own acts must carry each rise. Sitting with help may come late.
5. **Thrashing, learned helplessness, and leaving the mat,** with no one able to carry it back.
6. **The critics' float64 least squares** take about half the tick.
7. **Solver stability** with the Dex3 hands.
8. **Pain from its own housings,** now read through the joints' efforts (A37): 6–12% of babble ticks on the old skin, part of it the collision hulls meeting; measured again on the born loop (A54). The newborn's withdrawal can add to it.
9. **Vision.** The fovea is at a newborn's acuity through a born bank (3 px a degree, A42), in grey from the D435's imagers, with colour from one camera beside the left eye (A38). On its back the G1 sees mostly the ceiling and its own body; face down, only the mat; sitting or standing, only what is low and ahead of it (A22). The born face template, as built, rarely detects a real face (C39), and until it works the smile is read from the world (A49).
10. **Vocal imitation starts near chance,** and the word tokens are the easier road to reward.
11. **Disk and heat.**
12. **The tick budget:** about 117–190 ms as designed, with the lead's additions and the colour camera's own view; about 65–135 ms with the exact levers and the lighter meshes for its eyes (estimates, section 9). In lockstep a slower tick costs wall time only.

Section 13 adds risks 13–26; of those the lead's decisions brought, the one a viewer meets first is risk 19: **the parent sees only its trunk and hands** (A40). A look made by the fovea alone is invisible to her, as it would be on the real robot: fewer asks are met, and the ledger may run slow.

**The owner's decisions.** The nine decisions of the G1 amendment are folded in; section 14 lists where each one lands. The lead's decisions of 2026-09-24 on the owner's bar (A36–A59) are folded in too, each the owner's to overrule: they answer B18 (touch only where the real G1 has it), B4 (the D435's own sensors as the eyes) and B14 (the parent reads its trunk and hands). So are the lead's decisions of 2026-09-25 on his new bar (A61–A66): his own word answers B21 (no cloud, ever), and the lead answers B3 (a lighter copy of its own meshes for its eyes; the colour camera's own view and the sun's shadow kept). The decision log's part B asks what the design still defaults. One is needed before birth: B1 (the cup, shrunk for a creep that is gone). B5 (the microphones' and the speaker's places) must be read from Unitree's documents before birth, or the real robot would not be a change of world (A36). The build's fix rounds decided six questions under the laws, recorded as ours: the parent's face made real and the detector left alone, the newborn's withdrawal, impratio by physics, what decides the eye check, a new word on its pitch peak, and the grasp summed at the spinal cord (A30–A35).

### Words used here

| word | meaning |
|---|---|
| tick | one moment of the body's clock: 150 ms of simulated time (75 physics steps of 2 ms) |
| life day | 24,000 ticks (one simulated hour), followed by a night |
| the core | the shared machinery in `body/core/` and `body/model.py`: cortex, store, gates, critics, dopamine, the amygdala, feelings, night |
| digest | the hash `tools/determinism_check.py` prints after a tiny body lives a fixed script; equal digests mean an edit changed nothing the body does |
| born code | a fixed random encoder, made once from the body's seed and never trained |
| periphery | each grey eye's whole image, coarse (88° × 58°, 56 × 32 px), and the colour camera's in 5 × 3 cells |
| software fovea | a sharp 64 × 64 px window (about 21°, 3 px a degree, read through a born bank: A42) inside each grey eye's native image, with a colour window at the left eye's gaze, moved by the gaze effector; nothing on the robot moves (the first build's was 32 × 32 at 1.5 px a degree) |
| effector | a part the body acts with: the voice (the tract), the word output (the scaffold), the gaze, the waist, each arm, each hand, each leg |
| gate | an effector's learned "act now or not" switch; each effector has its own |
| movement unit | a chunk of ticks over which an effector holds one act, as infants' movements come in units |
| efference copy | the body's copy of its own last act, fed back into the cortex |
| the scaffold | the word-token channel beside the audio at birth (input), and the silent token output the parent reads; both removed later. The world-truth smile is a second scaffold of the same kind, removed when the child's own pixels carry the face (A49) |
| the tract | the child's articulatory vocal tract: its voice |
| the amygdala | the core's valence tagger (section 7.4): it learns fast which cues predict good and bad, and tags each moment's weight |
| tag | the amygdala's mark on a moment: what the moment is expected to bring plus what it brought, in reward units |
| never-taught test | a milestone's test of understanding by something the parent never taught (section 12) |
| the human-pace ledger | section 12's optional report of each milestone against the waking hours an infant has lived at the same capacity: never a target or a headline (A58, A66). "The ledger" alone is still the parent's ledger of words |
| waking-hour age | the infant age at which an infant at mean sleep has lived as many waking hours as the robot has lived life days (one waking hour each) to a milestone: the human-pace ledger's first figure |
| life hour | a life day's wake, 24,000 waking ticks: the body's unit of experience, in which fast learning is counted (A62) |
| a first | a claim no published architecture has shown, registered before birth with its bar, judged by A19's rule and reported whether it passes or fails (section 12, A63) |
| the intervention log | the count of every act of the parent's that moves the child's body or its world on its behalf: guides, holds, turns, placements, tidies, the morning basket (A63) |
| the observer | the born software estimator of contact from the joints' own efforts (a momentum observer, A37): what the real G1 can feel outside its hands |
| the cerebellum | the core's loop below the tick (R6c, 7.5): a born expansion whose weights learn load compensation and the VOR's gain, never balance |
| twitch | a brief single-joint act the brainstem makes in sleep, in the live, dark night (A46) |
| the lead | the session that decides for the owner under his standing word ("never ask, decide") and reports; the owner may overrule |
| working day | one build session's day of work |
| worktree | a second checkout on its own branch; the refactor happens there |

## 1. The goal and the laws

**The goal** (the owner's words): "give it human brain architecture where it will actually gain understanding"; find the human brain's math structure in simulation, where a body can learn without breaking; the robot after. This sim is body #2 on the same core as the language body. The body is the real robot's own model, so what it learns is learned by a body that exists.

**The bar** (the owner's words, 2026-09-24): "robot need to be like we put human brain in g1 and sim is reality", and the architecture worth billions when all is complete. So the sim body is the real G1's body, sense for sense and motor for motor, and seed 1 can go on in the real G1 as a change of world (A36). The brain is judged by an infant's first-year capacities (not its pace, since 2026-09-25: A66), and every organ added for it is decided before birth, since a body changed after birth is a new seed (A20).

**The bar of 2026-09-25** (the owner's words): "only local no cloud. i dont care about huuman pace i just want fast learninhg and for it to solve robotics that no archecture had done before". So the life runs on this Mac only (A61). Fast learning is the body's sample efficiency, counted in its own life hours, and the fastest local run that changes nothing it senses or does (A62). The goal is firsts no architecture has shown, each worded for what a life can show and registered before birth with its bar (section 12, A63). The earlier bars stand: the sim is the robot, the architecture is judged by understanding and not by association, and it is to be worth billions when all is complete.

**What "done" means for this first body.** Seed 1 of the G1 is born on the refactored core and lives days and nights. It reaches milestone M1 and passes M1's never-taught test (section 12), while the language digests stay exactly as pinned. Beyond that, the life is judged by the firsts registered before birth (section 12, A63).

| law | what it means here | where |
|---|---|---|
| Grounded reward only | The parent's face (felt only while the child looks at it), pain from contact physics (estimated from the joints' efforts, A37), the charge need. Nothing rewards moving, getting closer, looking, hearing words, sounding like the parent, or novelty in itself. A met ask is the parent's judgment of an act it asked for, not a reward for looking (A2). The gates' credit carries two disclosed intrinsic terms, neither a reward and neither reaching the critics or dopamine: a tonic drive that follows the reward rate, and the tract's performance error against its own usual acts (A41). | 6, 7.3 |
| The sim is the robot | Every sense and actuator of the body, and every constant of their physics, is the real G1's, or published for it: touch only in the Dex3 hands, contact elsewhere from the joints' efforts, the D435's own sensors, Unitree's servo gains and motor figures. The brain's constants (the core, the born codes, the reflexes, the cerebellum) are ours, disclosed in section 10, and run on the robot as software. What the real robot lacks lives in the world as a scaffold with a removal test (the word tokens, the world-truth smile). So the real G1 is a change of world for seed 1 (A36). | 3, A36 |
| No hand-written rules, cheats or controllers in the body | No balance, posture, walking or gravity-compensation law. Guided moves are labelled by a learned inverse model. Stops are learned, and `chunk_max` is only a ceiling. The innate parts are few, named and disclosed. The spinal pattern generator is a rhythm with no posture or balance term (A48), and the cerebellum's teacher is the servo law's own correction, which knows joint angles only, so it cannot learn a sit (A44); whatever it learns of a limb's own weight is learned, never written. | 3.3, 3.6, 3.7, 7.5 |
| The architecture biology uses | One gate per limb (parallel basal ganglia loops), movement units, forward and inverse models, a vestibulo-ocular reflex, cochlea-shaped filterbanks with a brainstem delay line, a born centre-surround and oriented bank in the fovea, the amygdala as a named organ (a fast valence tagger), a cerebellum below the tick, a spinal pattern generator, recall into action from the hippocampal store, twitches in active sleep, vocal learning through the body's own ears | 3, 4.9, 7 |
| Survives a change of body | Every mechanism is written against the Anatomy (channels, effectors, reward sources), never against these joints. The per-joint alphabet grows linearly with the joints. The robot's own model is the body. | 8 |
| Understanding, tested by what was never taught | Every milestone has a test the parent never taught toward, judged by the body's own rulers and by sitting with it, never by a baseline run | 12 |
| Local only | The life runs on this Mac: no cloud or rented machine and no spending, ever (the owner's word, A61). Claude's steering row every 400 ticks is the one input from off the Mac, served on his plan and logged by tick, so the life replays here alone | 0, 9 |
| Fast learning, honestly counted | Speed is the body's sample efficiency (life hours and exposures to criterion, and what a night adds) and the fastest local run that changes nothing the body senses or does. Other methods' published sample counts are context, never a run or a target, and no multiple is printed (A62) | 9, 12 |
| Firsts, registered before birth | Each claim is worded for what the life can show, frozen with its bar before birth, its tests' chance measured on the born state, the parent's interventions counted beside it, and reported whether it passes or fails (A63) | 12 |
| Human pace: an optional report | The human-pace ledger is kept only as a report the owner does not need, never a headline. Its firewall stands: frozen before birth if kept, and read by nothing in the body, the parent, the brief, the tests or the build (A58, A66) | 12 |
| Disclosed constants | One table, saying who sets each | 10 |
| One seed per body | Seed 1 is the only life. Plumbing and timing runs have every learning rate at 0. The babbler is an instrument of the world, and no weight of the body learns from it. | 11 |
| Nothing fitted to the environment's pace | The world waits for the child (lockstep). The pain thresholds come from the body's declared model: each joint's torque limit, and for the free base its mass (A37). Store writes are gated by the body's own running quantile. The parent's timings are set before birth and never tuned to the child's rates. | 3, 4, 6 |
| The environment's shape is the owner's | Decided: the nine decisions of the G1 amendment, and before them the parent, the room, the ears and the reward's route; and the lead's decisions of 2026-09-24 and 2026-09-25 for him, which he may overrule (A36–A59, A61–A66). Everything else defaulted is flagged. | 14 |
| The teacher's method and the body's constants are ours | The parent's timings, feelings, priorities and force caps live in its constants files. The body's constants are in section 10. | 4, 10 |
| Measure on copies, change at boundaries | The refactor lives in a worktree. After birth, fixes are measured on copies and applied at night boundaries. | 8, 11 |

## 2. What changed, and what still holds

**From the owner's word of 2026-09-25** (A61–A66):

| part | before | now |
|---|---|---|
| the machine | this Mac by default; a rented graphics machine the owner's call (B21) | this Mac only, no cloud, ever (his word; A61) |
| how speed is judged | an infant's waking hours to the same capacity, with the waking-hour age as headline (A58) | sample efficiency in the body's own life hours and exposures, and the fastest local run that changes nothing the body senses or does (A62); the human-pace ledger an optional report (A66) |
| what is claimed | the milestones and their never-taught tests | also two firsts, each worded for what a life can show and registered before birth with its bar (12, A63) |
| the speed levers | built on the core's free days 15–16 | each with its build step (R8, R10, W7, P7), birth waiting for none (9, 11, A64) |
| the eyes' render | three shortcuts offered to the owner (B3) | a lighter copy of its own meshes for its eyes, within its sensors' noise and a pixel; the colour camera's own view and the sun's shadow kept (A65) |

**From the G1 amendment** (the lead's decisions of the evening of 2026-09-24, on the owner's bar; A36–A57):

| part | the G1 amendment | now |
|---|---|---|
| touch and pain | 45 zones, one per link: a sim skin on the 28 links the real G1 does not feel with (B18); pain above 1,012 N on a zone | the Dex3 hands' tactile arrays (16 zones), and contact and pain everywhere estimated from the joints' efforts by a momentum observer; pain when an outside torque passes a joint's own limit (3.4, 6, A37) |
| the eyes | two colour pinholes at the D435's imagers, 168 × 96 px; the fovea's code the mean colour of 4-px cells, about 0.2 cycles a degree | the D435's own sensors: a grey global-shutter stereo pair and a colour camera beside the left one, each with its noise; the fovea at native pixels (3 px a degree) through a born centre-surround and oriented bank, a newborn's acuity (3.4, A38, A42) |
| the motors | servo gains ours (the limit at 0.25 rad); ideal actuators; a gyro with white noise only | Unitree's published gains; each motor's torque–speed envelope, torque estimated from current, encoder steps and heating; a drifting gyro; each motor's temperature as a sense, never a reward (3.3, A39) |
| how she reads it | from the software fovea's window (B14) | from its trunk and hands, with a person's error (4.10, A40) |
| the gates' credit | "born like the voice's", its drives unstated | a tonic drive following the reward rate (Niv et al. 2007) in every gate, and the performance error in the tract's gate only (Gadagkar et al. 2016), disclosed (3.5, 7.3, A41) |
| orienting | to a face-like blob and to a sound's side; visual onsets refused (A23) | also to a sudden local change in the periphery, habituating (Sokolov 1963; Johnson 1990) (3.7, A43) |
| below the tick | nothing | a scoped cerebellum: load compensation and the VOR's gain (R6c, 7.5, A44) |
| the store | recall reaches the words only | recall enters each effector's proposal through a map born at zero (R7f, 7.6, A45) |
| the night | the world paused | the world live and dark, with twitches in active sleep (5.4, R8, A46) |
| the voice's reflexes | a born cry refused | a born cry on pain or a low charge, which the tract's own acts override (3.7, A47) |
| the spinal cord | the stepping generator refused | a pattern generator per limb, summed at the cord like the grasp (3.7, A48) |
| the smile's carrier | the world's value of her face, gated by the face test | the child's own pixels, through a born mouth-corner reader, once its face detector works; until then the world's value as a disclosed scaffold (6, A49) |
| the parent | one waveform per line; her eyes to the object on every naming word, asks included; perfect contingency | eight voice variants per line; her eyes on the child in asks and probes; imperfect as a person, and copying its movements (4.3, 4.4, 4.10, A50–A52) |
| the room | ten flat-coloured toys, one of each kind | textures, containers, a cover, at least 3 examples per tested noun, new objects by calendar (5.2, A53) |
| the tests | one never-taught test per milestone | the roadmap's tests by milestone added, fixed before C36 (12, A55) |

**From the all-out design of this morning** (the owner's nine decisions):

| part | the all-out design | now |
|---|---|---|
| the child | a custom child: 0.75 m, 9.45 kg, a three-joint neck, eyes that turn, a mitten hand, a screen face | the stock G1: 1.32 m, 34.4 kg, no neck, Dex3 hands, no face (3.1) |
| looking | turning eyes and a neck, with a VOR | a software fovea inside each camera image, then the waist and the whole body; the VOR acts on the window (3.4, 3.7) |
| the parent | five scripted faces; lifts, props, turns and slides the child; brings it back | a scripted state of feeling on a graded face of human proportions; a behaviour system (contingency, joint attention, infant-directed speech, scaffolding, routines); a person's force caps; it comes to the child (4) |
| the voice | a synthesized child voice playing word tokens | an articulatory vocal tract it must learn to use; the token output becomes a silent scaffold (4.9) |
| the tagger | the "valence learner", a decision-log entry | the amygdala, a named organ of the core, in R7 and R8 (7.4) |
| understanding | the ledger's "understood" | a never-taught test at every milestone (12) |
| light | fixed daylight | day and night light across the life day (5.4) |
| the face reward | a rise felt as the new level | the increment rule, for a graded face (6) |
| the world | a 1.6 m mat; toys sized for a mitten | a 2.8 × 2.0 m mat; toys placed around the G1; the cup at 0.8 (B1); the elliptic friction cone with impratio 10 (5) |

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
- **Time:** the 150 ms tick, the 24,000-tick life day, lockstep, and deadline mode as a test switch only. (The night is no longer a pause: A46.)
- **Reward:** its three sources; the face felt only while the child looks at it; the born expression reading (from the world until the child's own pixels carry it, A49); innate orienting.
- **Parts from birth:** the amygdala and the motor timing part.
- **Hearing:**
  - two ears with cochlea-shaped filterbanks, and each toy's own sound;
  - the word tokens as a scaffold removed later, with letters as the fallback;
  - no live human teacher; Claude steers between minutes.
- **Movement and night:** the servo law re-anchored on each act (with Unitree's gains now, A39); the withdrawal reflex (triggered by the joints' pain now, A37); weakness when empty. Night as a pause of the world is replaced by a live, dark night (A46).
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
- **No face on it** (the owner's decision 7). The parent has the face; the parent reads the child only from what it does, with one disclosed exception: its charge (the battery gauge a carer would see; the real G1 reports it). Where it looks she reads from its trunk and hands, as a person would read a robot that shows no eyes (B14, answered by A40).

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

- **These are the model's own limits:** Menagerie's file, read at load and never changed there, with no scaling but weakness (A21). The old risk "torque limits from memory" is gone. They are not "the real motors' limits as Unitree publishes them", as this section first said: Unitree's own sources agree neither with the file nor with each other (read again 2026-09-24):
  - its URDF of this revision (`unitree_ros` `g1_29dof_with_hand_rev_1_0.urdf`) gives 35 N·m for the ankles' pitch and roll and the waist's roll and pitch, where the file gives 50;
  - its MJCF (`unitree_mujoco` `g1_29dof.xml`) gives those four 50, but the hip roll 88, where the file and the URDF give 139;
  - its G1 page gives the knee 90 N·m (the G1) or 120 (the EDU), where all three files give 139.
  - W1 first set 35 from the URDF. A verifier found that against this section and A21, and the file's limits were put back (so kp 200 at the ankles). The design keeps the model's (the owner's "keep it stock"); the real robot's are checked before the robot (C37).
- The legs and waist are strong. The wrists' pitch and yaw (5 N·m) and the fingers (1.4 N·m) are weak.
- Each joint has friction loss 0.3 N·m and armature 0.01, as shipped.

### 3.3 The servo law and tone

- **The file's servos are a placeholder.** Its actuators are position servos at kp 500 with critical damping, Menagerie's value; its README says the gains need tuning. They are not the real robot's gains.
- **The gains are the real robot's** (A39; changed by the lead's decisions). The real G1 takes a target, kp and kd for every motor on every command. So the servo gains are set at load (as the scene light is switched off at load), never in the file, and they are the gains Unitree publishes for the G1 and the Dex3 in its own low-level code (unitree_sdk2's G1 and Dex3 examples, unitree_rl_gym's G1 configuration), read from those files before birth (C46). Where Unitree publishes none for a joint, it takes the nearest published joint's gain per N·m of torque limit. None is chosen for a measured rate.
- **The first law, now retired** (the same as the custom child's; every W1 figure in this section and in 3.8 was measured under it):
  - each servo reached its torque limit at 0.25 rad of error, so kp = limit ÷ 0.25. That gave 352–556 N·m per rad for the hips and knees, 200–352 for the waist, 200 for the ankles, 100 for the shoulders, elbows and wrist roll, and 20 for the wrists' pitch and yaw;
  - the Dex3 joints reached their limits at 0.1 rad: 14–24.5 N·m per rad;
  - damping was 0.04 s × kp.
  - Unitree's published leg gains are recalled as several times softer than these; the sinks, the pain rate, the grasps, the guides and the catch are measured again under the published gains (C46).
- **An act re-anchors the target:** target = the measured angle + the step, so a move starts where the limb is.
- **At rest the target relaxes toward the measured angle,** with a time constant of 3 ticks (ours, anatomy: muscle tone).
  - A posture held by acting holds; a posture left to rest sinks.
  - The postures in section 3.8 were measured with the stock servos locked on fixed targets. Under the resting law they sink: holding a posture is learned, by acting every tick, as an infant learns to hold its head. Measured in W1 (`tools/sim_sink.py`, 3 s at rest under this law): lying at birth, a shoulder sags 12°; with the arms raised 45°, 29°; prone on its elbows, the elbows 12°; the leaning sit holds (6° at an elbow); standing tips past 20° after 2.85 s. These were measured at the first W1 build, with the ankles then at 35 N·m; a re-run after the file's limits were restored found no posture tipping 20° within 3 s.
- **No gravity compensation and no balance law.**
- **Weakness when empty:** the torque limits × (0.3 + 0.7h), where h is the charge.
- **The motors and the inertial units as the real ones** (A39; the world's physics, each from its own seeded stream, never a rule in the body; each constant from Unitree's documents or the part's datasheet, C46):
  - each motor's torque falls with its speed along its torque–speed envelope;
  - the torque the body senses is estimated from the motor's current, with that estimate's noise, not MuJoCo's exact force; the angle comes through the encoder's steps;
  - each motor heats with the square of its torque and cools to the room (a first-order thermal model). Its temperature is a sense in the body channel (3.4), as the real G1 reports it in each motor's state. It is never a reward and never a drive. Firmware that weakens a hot motor is modelled only if Unitree documents it;
  - each gyro's bias walks (a random walk per axis; Woodman 2007), on top of the white noise the file declares. The VOR's born gain meets the drift, and the cerebellum's flocculus learns to cancel it from retinal slip (7.5).
- **Measured again before birth (W4)** under this law, with the published gains and the motor models: the sink rates, rolls under babble, the leaning sit's fall time, guided tracking, and pressing into obstacles.

### 3.4 Senses

Each channel's code is born fixed from the body's seed, unit-scaled and projected to d = 512. Each channel has its own forecast head.

World truth (object poses, labels, which source made a sound, the torso's orientation in the room, every contact force) goes only to the parent and the instruments, never into the body. There were four disclosed exceptions; under the lead's decisions two remain, and both are scaffolds of the world, each removed by its own test (A36). Pain and charge are sensors now: pain is estimated from the joints' own efforts (A37), and the charge is the battery's gauge, which the real G1 reports.
- **The face channel, until the child's own pixels carry it** (A49). Its level is the born expression reading of the parent's graded face (A1). It passes only while the face test holds; the test is a geometric ray test driven by where the child's own software fovea points. The amygdala's and the critics' event line "a face in the fovea" is **not** this test: it is the born three-blob face template (the one orienting uses) run on the fovea's own pixels (built in W3: the frame's `face_fovea`, either eye), so the body's value and salience never read world truth beyond the reward's carrier. On the parent's face of human proportions the template, unchanged, detected her face in 1 of 48 fovea readings at 0.3–2 m, and its other matches were chance. As built, this event line would run on chance, so its constants are settled from its own sources before birth (C39).
  - Once that detector works, the face channel's level is the born mouth-corner reader's on the fovea's own pixels, gated by the detector (A49). If it works before birth, seed 1 is born so, and the world's value never enters the body. Otherwise the world's value is a scaffold: the reward's face term takes it while the world supplies it and the reader's reading when the world falls silent, a rule fixed from birth, so its removal is a change of the world, as the word scaffold's is (A29, A49).
- **The word scaffold (channel 0).** It is the parent's own label for the word it said, given as a token or as letters beside the sound, and only while the parent is audible. The ears still hear the same word. The scaffold is removed on a copy once the child's own hearing of words passes the tests in section 4.9.

Everything else the body gets comes from its sensors. Orienting's triggers are among these: the face template works on the periphery's pixels (the frame's `face_periph`: whether it matched, and where the best match lies from the window; it never matched her face there, C39), a sound's onset and side come from the cochlea and the born lateral read, and a sudden local change comes from the periphery's own cells (A43). They are never taken from the world's list of events.

**Where the G1's senses are.** They are added at load as cameras and sites: no shape, mass, joint or actuator. Each is the real G1's sensor, and each sensor's physics (its noise, its steps, its drift) is modelled on the world's side from the part's own figures (A36, A38, A39).

| sense | on the real G1 | here |
|---|---|---|
| eyes | the head's RealSense D435 (Intel D400 series datasheet): two monochrome global-shutter imagers 50 mm apart (OmniVision OV9282, 1,280 × 800), a rolling-shutter colour camera beside the left one (OmniVision OV2740, 1,920 × 1,080, 69.4° × 42.5°), and an infrared dot projector between them | the two grey imagers as its two eyes. The pose is Unitree's URDF `d435_joint`: 0.0576 / 0.0175 / 0.4299 m in the torso's frame, pitched 47.6° down. The right eye is 50 mm to the right, and each pinhole sits 3 mm in front of the head shell. The field is 88.3° × 58° (horizontal × vertical), at 336 × 192 px per eye (A42). The colour camera is a third view at its own place beside the left imager, 69.4° × 42.5° at about 238 × 134 px. The projector is off. Every image passes through its sensor's noise (A38) |
| ears | a four-microphone array (its positions are not in the model) | two sites on the head's sides, 15.8 cm apart (assumed, B5; read from Unitree's documents before birth, A36) |
| balance | inertial units in the torso and the pelvis | the model's own `imu_in_torso` (it moves with the head) and `imu_in_pelvis`, each with a gyro and an accelerometer. The file declares their noise (gyro 5e-4, accelerometer 1e-2), but MuJoCo 3.9 does not apply it (checked: a still G1 reads exactly 0), so the world adds it from its own seeded stream, and each gyro's bias walks (A39) |
| joint sense | each motor reports its angle, velocity, torque (estimated from its current) and temperature (unitree_hg `MotorState`) | the 43 joints' angles (through the encoder's steps), velocities, torques estimated from current, and temperatures (A39) |
| touch | tactile arrays on the Dex3-1 hand; no skin elsewhere | the Dex3 hands' 16 zones (each palm and each finger link), counting contact only on the faces where the real arrays lie and saturating at their range (C44); everywhere, the outside torque on each joint and the outside wrench on the free base, estimated from the joints' efforts (the observer, below; A37) |
| its voice | a loudspeaker (its place to check, B5) | the tract's sound from the head's front (4.9) |

- **Grey stereo, as on the robot** (B4, answered by A38). The first build gave both eyes colour; the real D435's two imagers are monochrome, and its colour comes from one separate camera.
- **There is no depth channel.** Two eyes give disparity; the fusion is learned (the owner's decision 4).

**The eyes and the software fovea** (A38, A42).
- **The three views** (the two grey eyes and the colour camera) are rendered once a tick into one buffer, with one read-back. The G1's own body is drawn in them from a lighter copy of its visual meshes, within its sensors' noise and a pixel (A65, C69); the physics and every other camera keep the full meshes.
  - **Grey:** each eye's image is the imager's response to the rendered light, its visible response weighting the render's red, green and blue (C45). The render has no near-infrared light, which the real imagers also see: a disclosed gap.
  - **Colour:** the colour camera's own view, 15 mm beside the left imager (RealSense's documentation gives 15 mm between the two centre-lines; its exact place and axis are read from the datasheet, C45). It covers the central 69.4° × 42.5° of the left eye's field. So colour is central and one-sided, as on the robot. It is never cut from the left eye's render, whose colour would miss the robot's parallax where its hands reach (A65).
  - **The camera model** (A38): an exposure loop, Poisson–Gaussian noise (Foi et al. 2008), blur from the head's rotation over each exposure (the physics' own motion, not the gyro's noisy reading: the blur is the world's), the colour camera's rows read one after another over its readout time, and gamma. Its constants come from the sensors' published figures (C45); its random numbers come from the world's seeded stream.
  - **The infrared projector is off.** A pattern of laser dots in its eyes would be a lamp on its own head, which the eyes refuse, as they refuse MuJoCo's headlight (5.1).
- **Periphery:** each grey eye's whole field averaged 6 × 6, giving 56 × 32 px (0.64 px a degree, as before); the colour image in 5 × 3 cells about 14° across.
- **Fovea:** a 64 × 64 window of each grey eye's native image, about 21° wide at 3 px a degree, a newborn's acuity (A42). The first build's 32 × 32 window at 1.5 px a degree, read as the mean of 4-px cells, carried about 0.19 cycles a degree, below a newborn's ~1 (Dobson and Teller 1978).
  - Its place is a gaze state (yaw, pitch, vergence) that the gaze effector moves.
  - It reaches ±38° × ±20° from each camera's axis.
  - Vergence moves the two windows apart or together, so both eyes can fixate a near thing.
  - **The colour window** is the same 21° at the left eye's gaze direction in the colour camera's image; what lies outside that camera's field reads nothing.
- **The fovea's born bank** (A42; orientation-selective cells are present in visually inexperienced kittens: Hubel and Wiesel 1963). On each grey fovea's native pixels:
  - centre-surround ON and OFF cells (a difference of Gaussians, the centre 1 px, 0.33°);
  - oriented cells at 4 orientations and 2 scales (periods of 3 and 6 px: 1.0 and 0.5 cycles a degree, a newborn's limit and half of it), each the energy of an even and an odd filter, as a complex cell;
  - both pooled over 8 × 8 px cells, the same grid of 8 × 8 cells of 2.6° as before, so an edge finer than a cell is kept as its energy.
  - The colour window is read as cell means of the colour camera's red–green and blue–yellow axes, ON and OFF.
  - The born face detector reads the centre-surround map at native pixels (C39). The bank's constants are ours, disclosed in section 10, fixed before birth, and never tuned to the eye check (C3, C48).
- **The window's place enters the body sense,** as the custom child's eye angles did, so the body knows where it looks.
- **Why in software.** The real G1 can run the same window and bank on its own camera images, so nothing is added to the robot (the owner's decision 9 recommends it).

**Touch where the real G1 feels, and contact from its joints** (B18, answered by A37).
- **The Dex3 hands** keep their 16 zones, one per palm and finger link, standing for the Dex3-1's tactile arrays: a zone counts contact only on the faces where the real arrays lie, and saturates at their range (their number, places and range from Unitree's Dex3-1 documentation, C44). The grasp reads the palm, as before.
- **Everywhere, contact is estimated from the joints' efforts** by a born observer: the momentum observer robots use to detect collisions without a skin (Haddadin et al. 2017). The residual between the generalized momentum the body shows and the momentum its motors and gravity account for is each joint's outside torque.
  - It reads only what the real robot's own sensors give: the encoders, the torques estimated from current (A39), and the pelvis's inertial unit, whose specific force and rotation stand for the free base. So the whole body's outside wrench is estimated too.
  - Its model is the robot's own file (its inertias), which the real robot's software carries too. In the sim that model is the physics itself, so only the sensors' models (A39) give it error; the real robot's inertias, friction and backlash differ from its file, so its observer errs more: a disclosed gap, measured on the robot before its first days (C43).
  - It runs every 5 physics steps (10 ms, the pain filter's window) in the world's sensor code, as software the real G1 can run on its own sensors, like the fovea.
- **What it can and cannot tell.** A contact shows on the joints between the pelvis and the touched link, so the furthest joint that feels it names the limb (Haddadin et al.'s isolation). A touch on the head cannot be told from one on the torso (they are one link), and a touch on the pelvis shows only in the base's wrench. A touch lighter than the estimate's noise is not felt: the real robot's condition.
- **The sim skin is gone from the body:** the 29 zones outside the hands, on 28 links (the head's zone is on the torso's link). The world still computes every contact force, for the instruments, the parent's own pain and perception (her thump line, A13) and the hull measure (A54); the body gets contact only through the observer and the arrays' model on the Dex3's faces.
- **Pain** (section 6) is read from the observer: a joint's outside torque past its own torque limit, or the base's outside force past 3 × the body's weight (A37).

| # | channel | numbers a tick | what it is |
|---|---|---|---|
| 0 | words (the scaffold) | one symbol from 79 | The parent's word token, arriving on the tick its sound ends; letters for words after the first 50 (section 4.9). Its forecast is today's `latent_pred`. |
| 1 | face | 2 | The born expression reading of the parent's graded face: 2 × (smile − frown), its level and change. It updates only while the face passes the face test (A1); once the child's pixels carry it, it is the born mouth-corner reader's, updating while the born detector finds her face (A49). Out of view it holds its last value for 30 ticks, then reads neutral (A2). |
| 2 | ears | 1,725 | Two cochleas (15 frames × 40 bands each, per tick) plus the brainstem's delay lines (21 low bands × 25 lags: ±12 samples, the head's own largest delay in those bands, 0.71 ms; ±8 would saturate the lateral read near 46° in the lowest bands). The spatializer is the exact rigid sphere through the two ear sites (radius 0.079 m, half their 15.8 cm), behind the ears' converter (flat to 7.6 kHz, gone by 8 kHz). Amended by P2 from 1,557 (`body/sim/ears.py`). |
| 3 | eye_p (periphery) | 2 × 56 + 60 = 172 | Per grey eye: 56 × 32 px, coded retinotopically as 7 × 4 cells of 8 px (about 12.5° × 14.5°, near the custom child's 12.5°) × luminance ON and OFF. The colour camera: 5 × 3 cells × red–green and blue–yellow, each ON and OFF (A38). The first build's colour eyes gave 2 × 168. |
| 4 | eye_f (fovea) | 2 × 640 + 256 = 1,536 | Per grey eye: the 64 × 64 window through the born bank, 8 × 8 cells × (centre-surround ON and OFF, and oriented energy at 4 orientations × 2 scales) (A42). The colour window: 8 × 8 cells × red–green and blue–yellow, ON and OFF. The first build's gave 2 × 384. |
| 5 | body (joint sense) | 242 | Per joint (43): sin and cos of the scaled angle (through the encoder's steps), velocity, servo effort (the torque estimated from current ÷ the limit), and the motor's temperature (A39). The gaze state and its velocity (6). The world gives these 221. The tract's 10 positions and velocities, and breath left (21), join from the voice lane. |
| 6 | touch | 16 × 2 + 43 × 2 + 6 × 2 = 130 | The Dex3 hands' 16 zones: log(1 + F / 1 N) of the tick's mean summed normal force on the arrays' faces, saturating at their range, and its onset (the rise since the last tick). The observer's outside torque on each of the 43 joints (÷ that joint's limit, signed) and the free base's outside wrench (force and torque in the pelvis's frame, ÷ the body's weight in N and N·m per metre), each with its onset (A37). The first build's 45 zones gave 90. |
| 7 | vestibular | 24 | Both inertial units: the accelerometer (which way is down) and the gyro (how the trunk turns, its bias walking), each as the tick's mean and peak. |
| 8 | charge | 2 | h and Δh (the body's own need; the battery's gauge on the real G1). |

- About 3,830 numbers a tick, plus one symbol (the ears' 1,725 in place of 1,557; under the lead's decisions the eyes' 1,708 in place of 1,104, the body's 242 in place of 199 and touch's 130 in place of 90: the core's born encoders, R6h day 3, take these sizes).
- **The born face template's two readings** come with the eyes in the frame: `face_fovea` (1: the event line "a face in the fovea", either eye) and `face_periph` (3: orienting's cue, whether it matched and where the best match lies from the window). R7's event lines and R6h's orienting hook read them; the template's constants are in section 10.
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
- **The gates' credit, disclosed** (A41; the code's own terms: the gate lesson in `body/core/mouth.py`, its constants in `physiology.py`). "Born like the voice's" left them unstated; they are on the reward path, so they are written here. For a tick on which a gate acted, the lesson's credit is
  - G_t = Σ_{k<12} 0.8^k δ_(t+k) + drive_t + w_int × e_t − c_t × (1 + (F_t ÷ 10)²),
  - the dopamine that followed, the tonic drive, the performance error, and the act's effort cost at its fatigue F; a rested tick's credit is the dopamine alone. The credit is taken against its running baseline (0.9), as the voice's is.
  - **The tonic drive follows the reward rate** (Niv et al. 2007: tonic dopamine as the average reward rate, the opportunity cost of time, setting vigor): drive_t = 0.25 + 4.66 × R̄_t, where R̄ is the felt reward's running mean at the ladder's 256-tick clock (`gate_tonic` 0.25, `gate_tonic_rate` 4.66, `gate_tonic_clock` 4). The 0.25 is the core's born drive and the served language body's: babble is its own reward at birth, and with no drive that body fell silent (BODY_SPEC.md). The 4.66 is Σ_{k<12} 0.8^k, the gate's own eligibility window: the reward the rate brings over the span the credit sums, so both terms are in the credit's units. It is the same for every gate.
  - So the drive rises under the parent's smiles and falls in a world that hurts. At a felt rate of −0.054 a tick (pain on about 5% of ticks, and nothing else) it reaches 0, and below that acting itself costs (risk 5, C47).
  - **The performance error, in the tract's gate only** (Gadagkar et al. 2016: a singing bird's dopamine neurons encode its performance against its own expectation). Per articulator, the forecast's belief in the setting it chose minus that setting's usual belief (a running mean per articulator and setting, 50 in all, moving 0.1 per act: `gate_habit` 0.9), averaged over the ten; `gate_int_form` "error", `gate_int` 0.5 for the tract's gate (the weight of the language body's interest term, "half its own confidence", under which its error form replaced the value form: BODY_SPEC.md; those runs raised `gate_tonic` to 0.70 to carry the value form's mean, and the served body ran with `gate_int` 0. The sim keeps the born 0.25, since its drive follows the reward rate; ours) and 0 for every other gate. It is zero-mean once its expectations catch up, and it compares the tract with itself, never with the parent (4.9).
  - **`gate_vigor` is 0** for the sim, as on the served language body (the core's default is 1.0): the drive already carries the reward rate, and a second route would count it twice.
  - None of these reaches the reward, the critics, dopamine or the amygdala (7.3).
  - In the code as built, the performance error is computed on effector 0's symbol, and every later effector's row carries 0; R6h computes it on the tract's gate and gives the token output none (C61).
- **Fatigue is per effector,** and each gate reads its own. One shared fatigue would add up nine limbs' costs and silence the voice.
- **Switches at birth:** fixes #4, #5 and #8 on (the gate's own draw recorded, the actor trace decaying per tick, credit from the act on), and `chunk_gate` 1 for every effector. Fix #1 and the amygdala join them when R7 builds them (section 10). The language body keeps them all off.

### 3.6 Movement units, stops and the motor timing part

- **Chunks and learned stops** (built in R6). While a chunk is under way, the gate's own draw decides whether it goes on, and the act is `act_pred`'s best guess. The chunk ends when:
  - the guess is the effector's rest;
  - the gate says no;
  - a reflex takes the tick (the withdrawal; the grasp sums with the hand's own act and ends nothing, 3.7);
  - or `chunk_max` 8 is reached (a ceiling, never the usual stop).
- **Movement units** (new, R6h). A motor chunk holds its first act unless the proposal prefers another by more than a born persistence margin (log 4 in logits, disclosed). The evidence, from the custom child's babbler (on its back, 3 × 4,000 ticks):
  - acts drawn fresh every tick: 0 rolls at any strength;
  - each joint holding its step for 1–8 ticks: 1.10 rolls a minute at full strength, 0.43 at 0.3×.
  - At birth `act_pred` knows nothing, so without persistence every chunk would be a fresh random draw each tick, and the body would never reach a whole-body event such as a roll.
  - The biological counterparts are infant movement units (von Hofsten) and the smooth, seconds-long spontaneous general movements of newborns (Prechtl), which are never white noise.
  - No roll count set the margin, and none will ever tune it. W4 writes down the rolls it gives on the G1, as chance.
  - **The evidence does not yet transfer to the born body** (found in the review of the amendment). The babbler held each joint for its own 1–8 ticks (4.5 on average). The born body's units are per effector, and R6's code continues a unit only while the gate's own draw says go on, at the born p_act of about 0.29 (the floor 0.05 plus 0.95 × `birth_act` 0.25). So born units average about 1.4 ticks, and about 70% last one tick: close to the fresh draws that gave 0 rolls. The nearer babbler conditions gave 0.6 rolls a minute (whole-body units, full strength), 0.03 (whole-body, 0.3×) and 0.17 (60% of units at rest). The margin only acts while a unit goes on.
  - So W4 runs **the born body's own motor loop** (its gates at birth, the continuation draw, the margin, `chunk_max`, every learning rate 0), not the per-joint instrument, and writes down the unit lengths and the rolls it gives. If the born units are far shorter than newborns' movements, the unit's born length is decided before birth on that biology (C38), never on a roll count.
  - **The spinal pattern generator** (A48) is the lead's answer on the same biology: newborns' kicks are rhythmic and their general movements seconds long (Thelen 1979; Prechtl 1990), and a spinal half-centre generator is the classic mechanism for such rhythms (Brown 1911), where a long born unit would be a constant with no source. It adds a slow rhythm to each limb's own acts below the gate (3.7), so the born loop W4 runs includes it, and C38's unit lengths and rolls are written down with it on.
- **The motor timing part** (built in R6, one per later effector):
  - `act_pred`: the cortex's stream → its own next act, read joint by joint.
  - the forward half: the stream after the tick's own step → the effector's consequence sense at the next tick (the limbs' joints; the gaze's state; the voice's ears).
  - the correction: the forward error → a term added to the proposal.
  - `act_inv`: (the sense at t, the sense at t+1) → the per-joint act, with a running reliability (Cohen's kappa per joint, R6 fix 2).
  - `act_pred`'s target is the efference copy when the effector acted. When it rested and something moved it (the parent's hand, a bump), the target is `act_inv`'s label, weighted by that reliability. So a demonstration counts only as far as the inverse model has earned.
  - `act_pred`'s plasticity is gated by its labels' reliability (R6 fix 3, e48b284). Its own optimizer, `GatedAdam`, takes each lesson as one sample weighted by its labels' mean weight (1 for an own act, `act_inv`'s reliability for a labelled rest, 0 for an unlabelled one), so own acts beside unlabelled rests learn at the share of the window they fill. At weight 1 it is Adam bit for bit.
  - Its verifier found the gate holds only for a newborn body: once `act_pred` has learned, unearned labels still reached the proposal through the cortex, whose Adam normalises each element. R6 fix 4 (440bad3: `act_inv`'s labels teach `act_pred` and the correction, never the stream; each optimizer bounds its own gradient) is in verification, and is folded here once a verifier passes it.
- **New for the humanoid (R6h):**
  - each limb's forward error feeds its gate. It tells a movement the child made from one done to it, and it sets how much a guided movement counts.
  - `act_inv`'s lessons are batched every 8 ticks; unbatched they cost 4–6 ms a tick.
  - an effector declares its consequence sense (the voice declares the ears), so the same part serves the limbs, the gaze and the voice.
- **Below the tick: the cerebellum** (R6c, section 7.5, A44). The motor timing part acts once a tick; the cerebellum acts every 10 ms inside it, adding learned torque to the servo law where a limb carries a load, and learning the VOR's gain. It never proposes an act, and its teacher cannot teach it a sit or a balance (7.5).
- **Recall into action** (R7f, section 7.6, A45). The store's recalled frames carry the efference copies of what the body did next; they enter each effector's proposal through a map born at zero, so recall moves an act only as far as it has predicted one.
- **Twitches teach the inverse model at night** (R8, A46). In the live, dark night the brainstem moves one joint at a time; each twitch and its reafference is a clean single-joint pair for `act_inv` and the forward half, and for the cerebellum.

### 3.7 Reflexes: kept and refused

Each kept reflex has a biological basis and is disclosed. The withdrawal acts below the gate: its ticks are logged as reflex and carry no gate eligibility. The grasp is summed at the spinal cord with the hand's own act (A35): the hand's gate draws every tick and its acts keep their eligibility, so letting go can be learned. The spinal pattern generator and the born cry sum the same way with the limb's or the tract's own act (A47, A48). The cortex sees them all through touch, joint sense, hearing and its forward model's error. Orienting is only a bias that learning can outweigh, so it fades by learning, not by calendar.

| reflex | status | how it works, and why |
|---|---|---|
| withdrawal | kept | Spinal and lifelong: the flexor withdrawal (Sherrington), generalized over the limb as a newborn's is (A31). When any joint of a limb is in pain (the observer's estimate, A37; a hand's joints count for its arm), that limb takes one big flexion step (0.27 rad) of its flexion joints a tick, for 2 ticks, wherever on the limb it hurts and whatever its last move was: a leg flexes the hip (pitch), the knee and the ankle (dorsiflexion); an arm flexes the shoulder (pitch) and the elbow, and a hand's pain withdraws its arm. No wrist joint takes part: the Dex3's fingers close across the wrist's pitch axis. Each sign was measured on the G1 (W1). The waist and torso have none, and pain in the base alone withdraws nothing. On the old skin it pressed a limb into a worse contact on 26% of onsets under babble (C22, measured again under the observer); a learned tuning is open (C40). |
| palmar grasp | kept | Spinal, present at birth. A touch on the palm above 0.3 N (its tick-mean force) closes six of the Dex3's seven joints one small step a tick (the thumb's rotation has no closing sense and keeps the hand's own setting), unless the hand's own act that tick opens it. It is summed at the spinal cord with the hand's own act, never taking its tick (A35). It makes the parent's hand-over a real hold from day one. It fades only as the cortex learns to override it. It fires on anything pressing the palm, its own fingers included: a fist closed on nothing stays closed until the hand opens it (under babble it fired on 14% of hand-ticks, 40% of those on its own fingers alone: C41). Lying face down, it closes a palm pressed on the mat, which may hinder crawling; W4 counts how often. |
| orienting to faces, voices and sudden change | kept | Newborns prefer faces (Goren 1975; Johnson and Morton 1991), turn toward sounds (Muir and Field 1979), and orient to peripheral visual onsets through the subcortical route (Johnson 1990). It is a born bias on the gaze's and the waist yaw's proposals toward a face-like blob in the periphery (a fixed three-blob template), toward the side the ears' born lateral read gives, and toward a sudden local change in the periphery (A43). There is also a born gate input, "a face, a sound onset or a sudden change appeared", computed from the pixels and the cochlea. It is a bias, never a forced move; its gain is set by the amygdala (section 7.4) and is exactly 1 at birth. Without it, a newborn that never looks is never rewarded. Measured on the parent's face of human proportions, the template never matched her face in the periphery: as built, the face cue would be chance, and the sound's side is the cue that works (C39). |
| orienting to sudden change: the cue | kept (A43; reopens A23) | A grey periphery cell whose luminance changed since the last tick by more than the periphery's median change plus a Weber fraction of 0.10 (the face template's contrast line) is a local onset: a change the whole image does not share, so the day's light and most of the trunk's own turn do not trigger it, and none fires while the gyro reads a turn above 10° a second. It habituates per cell (Sokolov 1963's orienting reflex, specific to the stimulus): each onset raises that cell's trace, which decays at the ladder's 256-tick clock, and the cue is the change × (1 − the trace), so a toy shaken again and again pulls less, and a new one in another place pulls fully. Its constants are settled from their sources before birth (C49). It feeds R7's event lines "a visual onset on the left / right" (7.4). |
| VOR | kept | Brainstem, present at birth. The software fovea's window counter-shifts by the torso gyro's rotation about each camera's own image axes (gain 1 at birth; the cameras are pitched 47.6° from the torso, so the gyro is rotated into each camera's frame first), so it stays on its target while the trunk turns. At the window's reach its quick phase jumps it back by half the reach, in the direction of the turn (A23). The gaze's own acts add on top. The cerebellum's flocculus learns the VOR's gain, and an offset that cancels the gyro's drifting bias, from retinal slip (7.5, A44; Ito 1982). |
| born expression reading | kept | A fixed read of the parent's mouth corners while the face is in the fovea (A1). Read from the world, a disclosed scaffold, until the born mouth-corner reader on the fovea's own pixels takes it over once the born face detector works (A49; Field et al. 1982: newborns tell happy, sad and surprised faces apart up close). |
| screen face | removed | The G1 has no face (the owner's decision 7). |
| spinal pattern generator | kept (A48; reverses the stepping refusal) | Spinal, present at birth: a half-centre oscillator per limb (Brown 1911), as the per-muscle oscillators that gave a simulated neonate its motor patterns (Kuniyoshi and Sangawa 2006). Its phase advances each tick; in its flexion half it adds a small step along the limb's flexion joints (the withdrawal's: a leg's hip pitch, knee and ankle pitch; an arm's shoulder pitch and elbow), in its extension half the opposite. The legs run in antiphase, as newborns' alternating kicks (Thelen 1979). Its amplitude is the limb's gate's p_act × 0.09 rad, so the gate's tonic readiness drives it, as the brainstem's drive enables the cord's generator. Summed at the cord like the grasp (A35): the gate draws every tick and its acts keep their eligibility, and an own act against the step cancels it. Its period, amplitude and the arms' coupling are read from their sources with C38's unit lengths, before birth, never on a roll count (C54). It has no posture, balance or gravity term: a rhythm, not a walking controller. |
| a born cry | kept (A47; reverses the G1 amendment's refusal) | Brainstem, present at birth: the cry is innate and patterned by the periaqueductal grey (Jürgens 2002). On a pain tick, or while the charge is below 0.2, the tract's born cry posture (the lungs pushing, the glottis pressed, the pitch raised, the jaw open) is added to the tract's targets, breathing in groups from its reservoir. Summed like the grasp: the tract's gate draws every tick, and its own act that tick overrides the cry's step, so the cortex can hush it. Its ticks are logged as reflex; a cry is never a vocal turn, and no smile answers it (A13). The charge line (0.2) is below the parent's feeding line (0.35), so the cry is the body's own alarm, never timed to her. Its pattern and line are settled from their sources before birth (C53). The parent hears it as distress (A13). |
| twitches in active sleep | kept (A46) | Brainstem, in active sleep from before birth (Blumberg, Marques and Iida 2013; in human infants, Sokoloff et al. 2020). In the live, dark night's REM phases, one joint at a time takes one small step (0.09 rad, its sign drawn), at about 10 a minute (Sokoloff et al. 2020's rate, read exactly before birth, C52); the tract and the gaze have none. The joint and the sign come from a born generator seeded by the body's seed. Its ticks are logged as reflex. Each twitch and its reafference teach `act_inv`, the forward half and the cerebellum (3.6, R8). |
| righting, parachute, equilibrium reactions | refused | Balance controllers. Infants mature these partly innately; the laws make ours learned, so our child's task is honestly harder than an infant's. |
| asymmetric tonic neck reflex | refused | A hand-written head-to-arm coupling, and the G1 has no neck. Measured on the custom child, it cut rolls from 1.10 to 0.77 a minute. |
| symmetric tonic neck, tonic labyrinthine | refused | Posture-tone rules |
| Moro (startle) | refused | No use here; a fall is already a large forecast error. |
| rooting, sucking | refused | Charging is not by mouth (section 5.3). |
| Galant, Babinski, placing | refused | No function in this body |

### 3.8 What the body can do at birth (measured on the G1, no learning)

Everything here was measured before the lead's decisions: at the stock or the first law's servo gains, with ideal motors and the sim skin. W4 measures what changes again under Unitree's published gains, the motor models and the observer (C43, C46); the reach and the postures' geometry do not change.

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
| babbling in the built world (W1: the servo law, touch and pain, the IMUs, the reflexes; load about 2) | — | 8.4–10× (14.9–17.9 ms a tick: physics 12.3–15.0, the world's Python 2.3–2.6) |
| the same with both eyes and the sun's shadow (W3), before the parent's face of human proportions | — | 2.8–3.1× (47.7–53.2 ms; the render 31.2–31.7) |

- There were no MuJoCo warnings in any of these runs.
- Under babble Σ τ² ÷ Σ τ²_max averages 0.063 (the custom child's was 0.043).
- The largest contact at the tick ends was 944 N and 1,799 N under babble. F_pain is about 1,012 N, so babble alone reaches pain (C5). With the 10 ms filter and the withdrawal, the built world is in pain on 11.6% of babble ticks (section 6).
- With the kinematic parent kneeling, a contact reached 7,205 N: the G1's limbs hitting an immovable parent. The parent's yield rule (A4) is essential.
- **Not yet measured on the G1** (W4): rolls under babble, travel, time off the mat, and the leaning sit's fall time. The custom child's numbers are kept in section 15 for reference.

**Grasp** (`tools/sim_grasp.py` on the built room: a Dex3 hand at its stock gains closing to the range's end; a hand-over into the palm, 6 tries, and a top grasp from a surface, 3 tries, per toy; none unstable). The prototype's 360 trials (scales 0.6–1.0) ran at impratio 1, where MuJoCo's soft contacts creep; the world runs at impratio 10, chosen by the contacts' physics (5.1, A32):

| toy | held at impratio 1 | at impratio 10 (the world) |
|---|---|---|
| ball | 0 of 9 | 9 of 9 |
| block | 9 of 9 | 9 of 9 |
| duck | 0 of 9 | 7 of 9 |
| cup at 0.8 | 5 of 9 | 9 of 9 |
| cup at full size | 0 of 9 | 9 of 9 |
| rattle | 5 of 6 hand-overs | 6 of 6 hand-overs (0 of 3 top grasps) |
| car | 5 of 9 | 8 of 9 |
| bear | 0 of 9 | 2 of 9 |
| stacker | 3 of 9 | 9 of 9 |
| drum | 0 of 9 | 0 of 9 |
| ring | 6 of 6 hand-overs | 6 of 6 hand-overs |

- **Only the bear and the drum cannot be held.** The cup was shrunk to 0.8 (6.7 cm across) for the creep; it holds at full size now, and stays at 0.8 until the owner answers B1.
- The grasp reflex's small steps under the body's servo law press less than the stock gains closing to the range's end; W4 measures the holds under the reflex and the resting law (C28).
- The Dex3's two fingers are 5.7 cm apart: small objects slip between them, and round or large ones are held by friction, never wrapped.

## 4. The parent

### 4.1 Its body and how it moves

- **A complete human figure at human proportions:** 16 segments (pelvis, spine, chest, neck, head; upper arms, forearms, hands; thighs, shins, feet).
- **A face of human proportions,** graded from its feelings (section 4.3): brows; lids; eyes with irises and pupils that look; cheeks; an upper and a lower lip; a mouth that smiles, frowns, rounds and opens with its voice. Built in W3's fix rounds (`parent_face.py`) to a real woman's norms and a real face's photometry, with the born template left as it was (A30):
  - **One smooth head sheet:** a thin-plate spline through the landmarks and her cranium, jaw and neck, the nose part of it, shaded by its own normals. MuJoCo draws the room's indirect light without shadowing, so her skin's shade of it is computed from her own geometry and baked into her texture (share 0.4). The eye openings are cut at the norms' outline.
  - **Measured on the drawn face** (`tools/sim_face_measure.py`, by rays on the geometry; the norm in brackets, mm): the eye opening 30.7 (30.7); the inner canthi 31.8 apart (31.8), the outer 87.8 (87.8); the fissure 10.5 tall (10.9); the pupils 61.7 apart (61.7); the iris 11.7 (11.7); the mouth 50.5 wide (50.2); the brow's lower edge 11.4 over the lid at the pupil (10.9), 18.0 over the inner canthus (17.5) and 18.9 over the outer (18.5), its top 24.5 over the pupil (25); nasion to subnasale 51.7 (50.6), nasion to chin 110.3 (111.8); the head 147.2 wide (146.6), the cheekbones 131.8 apart (130.0), the jaw's angles 93.0 (91.1). A verifier's own method (a segmentation render from the front) agreed within 1 mm.
  - **Sources:** Farkas et al.'s North American white women aged 18–25 (n 200), as tabulated by Husein et al. 2010 (J Plast Reconstr Aesthet Surg 63:1825–1831) and Virdi et al. 2019; Dodgson 2004 (the pupils); Rüfer, Schröder and Erb 2005 (the iris); Gao et al. 2025 and McKinney et al. 1991 (the brows); Yaremchuk's atlas (the brow 10 mm and the cheek 2 mm in front of the cornea); Park et al.'s head model (the head); EyeWiki's margin-to-reflex distances (the lid at rest). Recalled and flagged: the globe (24 mm) and a room-light pupil (4 mm). Ours: the anchors in her head's frame and the relief between the landmarks.
  - **The lids** turn about a hinge through the globe's centre, 12° back and −9° tilted: of a grid, the least turn that closes a blink. A blink closes the whole fissure, the lashes stop on the lower lid, and each lid stays 0.2 mm behind the skin through its sweep. Rigid lids show up to 1.5 mm² of eye at the fissure's ends in a blink seen from 35° (disclosed); from the front, none.
  - **The brows** are strips lying on the skin (0.1–1.1 mm over it in every expression); each lip is a tube of constant radius; nothing is white but the teeth, and nothing glows.
  - **Photometry** (Russell, Kramer and Jones 2017: young women without makeup, each feature's lightness contrast against the skin around it): measured under a studio lamp at her face's depth, eyes 0.152, brows 0.126, lips 0.092, the source's own.
  - Every graded expression still works, and the born reading still reads the mouth corners only. Cosmetic and open: at a full smile the lips are thin crescents round a flat dark oval, with no teeth.
  - The custom child's reference room keeps the old face (`parent_face_customchild.py`).
- **Her collision is her drawn face:** 102 convex pieces of her own sheet, 6 of her hair's cap, the head's solid, and a sphere over each eye out to the lashes. Over 83,068 rays the worst point was 2.8 mm behind the drawn face, and 0.16% of rays more than 0.5 mm behind; a 3 mm free sphere swept onto her face never got behind it. `mj_step` did not slow (0.33 ms). The hair hanging at her sides has no collision (a sphere passed 28–40 mm into it).
- **Hands with shapes:** open, point, grip, curl.
- **Kinematic.** It is posed every physics step by scripted forward kinematics, two-bone inverse kinematics for the arms and legs, and look-at for the head and eyes, all inside human joint ranges. The poses are written into 16 MuJoCo mocap bodies.
  - It touches the child and the toys as an immovable body, so its contacts need the yield rule (A4).
  - It holds toys through welds that start switched off. Its hand's collision proxy is off while a weld holds (both collision bits; the prototype's first switch cleared only one).
  - **It holds the G1 only through capped springs** (section 4.2): a force at the held point, computed each physics step from her hand's pose and clipped at the act's cap, applied as an outside force on that link. The prototype's welds on the torso, the pelvis and the elbows (`make_g1room.py`) are soft but have no cap, so W2 replaces them; a weld on the G1 would break the caps.
  - The room still excludes her hands' contacts with the G1's torso and pelvis (made for the prototype's welds), so a hand laid on its chest passes into it and is felt only through a weld. W2 removes the exclusion with the welds.
- **Its motions,** all passing the check with no failing frame:
  - a 4 m walk;
  - kneeling down (two step variants) and standing up;
  - a knee shuffle;
  - sitting on its heels or on the floor.
  - Kneeling needs about 0.6 m clear in front, so it kneels a step back and shuffles in on its knees.
- **Its cost.** One scripted pose takes 0.7–4 ms. Solving an act takes 1–430 ms, too slow for every tick. So an act is solved when it starts, and re-solved once a second while its target moves, warm-started from the last solution and interpolated in between (W2).
- It is the world's side only. Nothing of it enters the body except through the child's senses.

### 4.2 Its acts, and what a person can do for a 34 kg body

**Its acts:**

| act | status (measured today) |
|---|---|
| kneel beside the G1 and attend, one hand resting on its trunk | passes; at 0.78 m from its centre line, 0.10 m toward its feet, clear of its arm and the toys (no parent contact) |
| show a toy 40 cm before its eyes | passes; the toy held a little toward her side so her face is unblocked in both eyes (a ray test) |
| point, either hand | passes, including a drum 1.4 m away |
| hand a toy to the near or far hand | passes on the custom child (the grasp reflex held the rattle 3 s after release); on the G1, 8 of 10 toys can be held at impratio 10 (section 3.8) |
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
- **Being held is felt.** Each hold's spring force is an outside force on the link it holds, so the observer feels it through the joints, as the real robot would feel a person's hands (A37); on a Dex3 zone it is also the array's touch. It counts toward pain under the same law as any force. On the old skin every cap (at most 200 N) was far under F_pain (1,012 N). Under the joints' law a hold's force acts through a lever: a guide at its 98 N cap on a forearm about 0.25 m from the shoulder is near the shoulder's 25 N·m. So W2 and W4 check every hold and guide against it (C43); if any would hurt, her cap for that act tightens (the caps may only tighten, A25), and no exception to the pain law is made.
- **The friction model.** MuJoCo's soft contacts let the G1 creep 5 cm at 146–199 N, below the true sliding force. At impratio 10, chosen by the contacts' physics (5.1, A32), the G1 pushed at the pelvis with 100 / 150 / 200 N slides 0.4 / 0.7 / 1.9 cm in 2 s, against 1.9 / 3.1 / 4.4 cm at impratio 1. The surfaces' own frictions are still open (C26).
  - Now the world's surfaces are 1.0, and the G1's own foot spheres are 0.6 at contact priority 1 (as shipped), so its feet already set their own friction on the mat.
  - **Real frictions are set on the world's side only:** the floor, the mat, the furniture and the toys take contact priority 2 (built), so their friction and softness decide every contact with the G1 without touching its file. The G1's self-contacts keep its own values. The creep is fought with a world option, impratio 10; the noslip solver was refused (5.1). The G1's geoms are never touched.
- **Clearing toys** before kneeling. The kneeling spot for the G1 is outside its leg sweep (W4 measures the sweep).
- **Its feet do not physically meet the floor or the furniture.** Each placement is checked against them instead (the shoe within 5 mm of the floor).

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

**What the child's own pixels see.** Fovea pixels (32 px, 20.9°, the left eye) that differ from her neutral face by more than 8 grey levels, measured on her face of human proportions (W3's fourth fix; a scratch instrument, `$S/build/world/fix4/pixtable.py`, to be made a tool with W5's lights). Each cell is frown / +1 / +2 / surprise: the frown is frown 1 with the brows lowered 0.9 and the lids tightened 0.8; +1 a smile of 0.5; +2 a smile of 1; surprise the brows raised 0.95, the lids up 1, the jaw 0.5 and the "oh" 0.8.

| distance | midday | morning | dusk |
|---|---|---|---|
| 0.3 m | 28 / 19 / 37 / 101 | 78 / 33 / 80 / 167 | 61 / 26 / 70 / 149 |
| 0.6 m | 1 / 1 / 13 / 20 | 11 / 15 / 27 / 47 | 9 / 16 / 28 / 43 |
| 1.2 m | 2 / 1 / 3 / 2 | — | — |
| 2.0 m | 0 / 0 / 0 / 1 | — | — |

- The table depends on the light as much as on the face. In that placement her face reads about 36 of 255 at midday and about 60–77 under the morning and dusk lights, where the grades take about 2–16 times as many pixels at 0.6 m. It is given per light, never as a property of the face.
- The table it replaces was measured on the cartoon face (88 / 107 / 155 / 147 at 0.3 m; 58 / 42 / 67 / 74 at 0.6 m). On a real face the grades are plain in the child's own pixels only at the lean-in distance; beyond about 1 m only the born reading's world value carries them (A1, as disclosed). Once the child's own pixels carry the face (A49), a smile beyond that distance is not felt, as a newborn's acuity would not resolve it (Field et al. 1982 tested expressions up close): fewer smiles felt from afar, the real condition. The table was measured at 1.5 px a degree; at the fovea's new 3 px a degree (A42) it is measured again with the reader (C55).

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
- they go to the object only on the naming word of a label, a show or a confirm, then come back (the joint-attention cue);
- **in an ask or a probe** ("where is the X?", "look at the X", "give me the X", and every never-taught probe but M1's head-turn probe, whose stimulus is her silent turn) her head and eyes stay on the child, she does not point, and she does not turn until the ask is judged, as preferential-looking studies blind the parent (Golinkoff et al. 1987). Otherwise the child could pass by following her gaze. This closes the leak the audit found (A51);
- it leans over the G1's chest to be seen, its face never closer than 25 cm to the G1's eyes;
- when it calls or leans in, it places its face in the child's periphery, at least 15° off the fovea's line, and never moves it onto that line, so turning to it is the child's own act (A3);
- when feeding, it keeps its face in view, which pairs the face with the relief of the need.

### 4.4 Its voice

- **The engine:** macOS speech through a small Swift server (`AVSpeechSynthesizer.write`, with word markers).
  - It returns exact word onsets and is byte-identical on every run. (Not after an SSML `<break>`, which moves the next word's mark 90–210 ms before its sound; so no line has a break.)
  - A sentence takes about 45 ms once warm (median 42–50 ms, 95th percentile 53–68 ms).
  - It runs at nice 10, off the tick loop. A cache miss makes the lockstep world wait about 50 ms of wall time and costs no sim time.
- **The voice:** compact Samantha, at rate 0.25 and pitch 1.15. None of the installed voices is at enhanced or premium quality.
- **Eight variants of every line** (A50). The first build's clips were byte-identical each time a line was said, so a word could be learned as one waveform; 14-month-olds learn minimal pairs only across talkers' variation (Rost and McMurray 2009). Each line is made in 8 variants: its pitch moved by up to ±8% through the engine's own prosody, and its spectrum warped by a vocal-tract-length factor between 0.9 and 1.1 (Jaitly and Hinton 2013's VTLP range), resynthesized in numpy. The variant heard is drawn per utterance from the parent's seeded stream. Each variant is deterministic, so every clip is still made again bit for bit and held to its digest. The new word stays on its pitch peak in every variant (checked by frame, A34, C56). The lines made ahead grow to about 106 MB (331 lines × 8 × 40 KB), inside the 300 MB limit.
- **Registers by intent** (Fernald 1989's contours), which carry real prosodic cues the amygdala can pick up:

| register | pitch | rate | contour | used for |
|---|---|---|---|---|
| plain | 1.15 | 0.25 | — | most lines |
| approval | 1.35 | 0.25 | rise–fall | "yes! the duck!" |
| comfort | 1.05 | 0.15 | falling | after pain or distress |
| calling | 1.25 | 0.25, +6 dB | rising | the child's name |
| question | plain | 0.25 | rising (the synthesizer raises the pitch on "?" itself) | asks |
| "no." (stage 2 only) | 1.0 | 0.3 | short, low | a hit or talk-over |
| new word | 1.15 | 0.15 | the new word on its pitch peak and lengthened, on any ending | the day's new word (4.8) |

- **Infant-directed speech** (the owner's decision 6):
  - lines of at most 6 words, the focus word last (Fernald and Mazzie 1991: mothers put a focused new word in final position on the utterance's pitch peak in speech to infants, and not consistently in speech to adults);
  - the focus word on a pitch peak and lengthened through SSML: a prosody around the word and the punctuation after it, pitch +30% on the line's, rate 0.7 × the line's (lengthening: Albin and Echols 1996, word- and sentence-final lengthening in infant-directed speech). Every line goes to the engine as SSML, its register's pitch and rate as one prosody around it (a plain line so made is the same clip, bit for bit, as the utterance).
    - What the engine does, measured: a nested prosody's rate is read against the engine's default, not the line's; its pitch against the line's; a "." left outside the word's prosody is spoken aloud as "period" (bit for bit), so the voice refuses any word mark without a letter.
    - The first build (87a4194 and before) left the punctuation outside and wrote the rate as 70% of the default. Its "+37% F0, +91% length" was measured only on "." lines, where the parent said "period" after the word; on "?" and "!" lines the word came out 15% shorter.
    - Measured on every one of the 331 birth lines, the line's last word emphasized in the plain register: 1.17 times as long and F0 1.29–1.30 times the unemphasized line's, on ".", "?" and "!" lines alike;
  - the new word's lines in the new-word register (rate 0.15, the new word always emphasized; the voice refuses a new-word line without it). Measured over the 331 birth lines as new-word lines, the line's last word as the new word:

| line ends in | lines (distinct) | the new word's length, × the same line unemphasized (least) | × the plain line (least) | its F0, × unemphasized (least) | × the plain line (least) | words a second, pooled | lines over 3 |
|---|---|---|---|---|---|---|---|
| "." | 275 (258) | 1.14 (1.12) | 1.32 (1.28) | 1.30 (1.23) | 1.30 (1.22) | 2.93 | 116 |
| "?" | 37 (37) | 1.15 (1.12) | 1.33 (1.29) | 1.29 (1.24) | 1.29 (1.26) | 2.90 | 13 |
| "!" | 19 (19) | 1.14 (1.12) | 1.32 (1.29) | 1.30 (1.28) | 1.30 (1.27) | 2.51 | 2 |

- **The new word, measured further** (`tools/sim_voice_check.py`):
  - it lasts about 630 ms, against 552–554 ms unemphasized and 473–481 ms in the plain line; its F0 is about 195 Hz on "." and "!" lines and 232 Hz on questions, against 150 and 178;
  - C25 is pooled: 2.91 words a second over all 331 new-word lines, and 2.40–2.83 (2.61 on average) over the variation set for a new word ("a X." / "the X!" / "you see the X?", each of the 8 toys). Line by line, 131 of the 331 run over 3 (up to 3.88), four of them lines of 3 words ("what is it?" 3.70, "up. up. up!" 3.57, "up! sit up." 3.53, "good. the bear!" 3.16); no line of 1–2 words does. Rate alone cannot bring every line under 3: the engine gets no slower below 0.15 (C25);
  - at rate 0.2 (the parent spec's D3) the emphasized lines ran at 3.12 and 3.10 words a second on "." and "?" lines, over C25, so the register is 0.15. Questions keep the new word: the engine does it on every ending.
  - **On the line's pitch peak, measured** (06b85b2's verifier, the tool's own pitch tracker): the emphasized new word is the line's highest word peak on 233 of 273 "." lines, 34 of 36 "?" lines and 17 of 18 "!" lines of more than one word. It is on every line of the variation set and on "look. a X.". It is not on "this is a X." for 7 of the 8 toys ("this" peaks at 281 Hz, the noun at 258–271), on "this is your foot / hand / head." or on "the X is up.". So she introduces a new word only in frames measured to put it on the peak (A34), and `tools/sim_voice_check.py` is to measure the peak by frame (P3).
- **Its lines:** 4.0 words and 1.12 s (7.5 ticks) on average, 3.56 words a second (measured over the 331 birth lines; plain F0 203 Hz, approval 239, comfort 184 at 3.08 words a second, calling 219 at +6 dB).
  - At one symbol a tick, 95% of word tokens arrive on the tick their word ends and 5% one tick late.
- **The cache** is the life's voice folder, saved beside the body (never the source tree), keyed by the request: the voice, the line as SSML in its register, the clip format.
  - Every line the life hears is kept for good (`kept/`), so a replay reads the very samples heard, and a ledger keeps every clip's digest. A clip made again must equal its digest, or the voice refuses it and the life pauses (an OS update may change the voice). A ledger line cut short (a full disk, a kill) is dropped on load, and a line the ledger lost is restored from the heard clip's own record; a damaged clip so recorded is made again and held to that record. With the ledger lost, a heard clip can be checked only against its own stored record.
  - At each night boundary it pre-synthesizes every template line for the current vocabulary: 331 lines (314 distinct) in 8–15 s, 13.3 MB (40 KB a line); with the eight variants (A50), about 1–2 minutes and about 106 MB. Lines made ahead and not yet heard (`clips/`) have a 300 MB limit that drops the least recently used.
  - `kept/` grows by 40 KB for each line heard for the first time: at most about 25 MB a life day at 4.6's density even if every line were new, and the day's fresh lines in practice (section 9).
  - The server answers within 60 s or the life pauses at that tick (the decision log's rule, in `SynthServer`).
- **Its mouth** opens each tick with the clip's loudness in that tick (the jaw parameter).

### 4.5 Its sentences

- **Two layers.**
  - **The fast layer** is scripted, deterministic and seeded. It picks short sentences from templates, filled from what the parent can see: what the child looks at, holds, or just said. It carries every judgment that must land within ticks. There are 331 template lines at birth, 314 of them distinct (17 "." lines repeat).
  - **Claude, once every 400 ticks** (60 s of sim time; A59), reads a plain-text digest of the last window and writes one steering row: the focus toys, the next activity, fresh lines tied to situations, and the one new word to introduce.
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

- **Variation sets** (Küntay and Slobin; Onnis et al. 2008): 2–3 lines sharing the focus word, with frames differing by at least one word ("a duck." / "the duck!" / "you see the duck?"), 6 ticks apart. A set counts as one naming. At most one set per object per 120 ticks. A new word's set is in the new-word register, the word on its pitch peak and lengthened in each line whatever its ending: 2.40–2.83 words a second (C25).
- **The line check** applies to every line, templates and Claude's alike:
  - at most 6 words;
  - only `. ? !` as punctuation (the voice itself refuses a mark it would speak, a comma after the focus word among them, so a line past the check would stop the voice: the check is the guard);
  - only vocabulary words, plus that day's new word, placed last in the sentence, in a frame measured to put it on the line's pitch peak (A34);
  - never a held-out never-taught pair before its test (section 12).
- **The digest and steering.**
  - Every 400 ticks (A59; the first draft said every wall minute), `tools/sim_digest.py` writes about 40 lines of outward events only: posture; where its trunk faced and its hands went, as she reads them (A40), never its fovea's window, which a real G1 does not show; what it touched or held; what it said and when; the asks and their outcomes; the face events seen and unseen; pain; the ledger; the parent's last lines.
  - Claude appends a row to `data/sim_steer.jsonl`: {tick_from, ttl 2,000 ticks, focus, episode, task, away_ticks, introduce, lines, note}.
  - The fast layer checks each line, takes the row at its fixed tick (the digest's tick + 400, then the next utterance boundary), and uses each of Claude's lines at most 3 times. A row not ready by its tick makes the world wait; a row that fails leaves the fast layer alone for that window, and the log says so (A59).
  - Rows are logged by tick, so a replay is exact.
  - Without Claude, the fast layer runs alone. The brief is `ops/sim_parent_brief.txt`, in the manner of `ops/parent_brief_human.txt`.

### 4.6 Turn-taking and stages (in ticks)

- **Pauses:** 6 ticks between related lines; 20 ticks of expectant pause after a question, ask or call.
- **Judging:** 20 ticks for a gaze ask, 40 for an act ask. An act counts as met only once the effector that did it has come to rest on the result.
- **The child's turn:**
  - it ends when the tract has rested 2 ticks (silent) after sounding;
  - the parent replies 3 ticks later on average, answering what the child said: an expansion, a recast, an echo of its babble, or an answer (Goldstein et al. 2003; Goldstein and Schwade 2008, from memory). Her latency is jittered, and she sometimes misses a turn, at rates from human dyads (A52).
- **Talk-over.** If the child starts sounding during the parent's clip, the parent finishes the current word (at most 3 ticks), stops, and looks at the child with a listening face. A word that would need longer (2.8–2.9% of cuts over the birth lines: a long word said slowly, or a line's last word; 11.1% of cuts over the same lines as new-word lines, whose lengthened new word takes about 630 ms, 4–6 ticks to finish) is broken off at 3 ticks and not labelled as said: its token is withdrawn from channel 0, and a spelled word's letters already given are closed by its space.
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
| night | after tick 24,000 | The world runs on, dark, for 24,000 ticks (A46); the parent sleeps on the sofa; the child sleeps as it lies, with twitches in REM's phases. |

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
  - Colour words wait until two toys share a colour (the colour twins, B2), and each colour word is heard on at least 2 kinds of toy before its test (A55).
- **The parent's ledger** uses outward events only:
  - **heard:** said with the referent in the child's view.
  - **understood:** after "where is the X?" or "look at the X", with X visible but not where she reads it looking, its trunk turns to X or its hand reaches toward X within 20 ticks and holds 2 ticks, as she reads them (A40); her eyes stay on the child meanwhile (A51). For an action word, the act follows within 40 ticks.
    - It must hold on at least 5 of the last 10 asks, and beat the child's own base rate (the same test at random moments with no word said) with a one-sided binomial p < 0.05.
    - This is her ledger, which paces her words. It is not a claim of understanding: every claim rests on section 12's tests, one-sided p < 0.01 on fresh asks, with the word scaffold silenced for word claims (A19, A29, A55). The instruments, never she, read the fovea.
  - **says:** the child says X with X where she reads it looking or in its hand (A40), or right after doing the act; not within 10 ticks of the parent saying it; 3 times over at least 2 life days. Anything said within those 10 ticks counts only as an echo.
    - Said by the silent token output: its token, or letters within edit distance 1.
    - Said by the tract: accepted by the parent's ear in context (section 4.9). The ledger keeps the two apart.
- **"Learned" means understood.** Saying the word is the next rung, and using it in a never-taught way is the test of understanding (section 12).
- **Growth pace.** About 12 content words are active at a time.
  - The next word enters when at least half of the active set is understood.
  - At least one new word every 2 life days, at most 3 a life day.
  - A new word is said sentence-final, in 3 lines within a minute (one variation set), on the line's pitch peak and lengthened in every line of the set, whether it ends in ".", "?" or "!", and only in frames measured to put it on the peak (A34; 4.4: measured on all 331 birth lines by ending, the word 1.14 times as long and its F0 1.30 times the same line's unemphasized, 1.32 and 1.30 times the plain line's; the set at 2.40–2.83 words a second).

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
  - levels: /a/ at the parent's speech level, ANSI S3.5-1997's normal vocal effort, 62 dB SPL at 1 m (A26); a hiss sits at −14 dB and /h/ at −13 dB relative to /a/.
- **Cost** (measured on this Mac while other sessions shared it):

| case | ms per 150 ms tick |
|---|---|
| sounding (vowel, glide, fricative), a quieter moment | 1.8–3.0 |
| sounding, under heavy load | 7–10 |
| at rest | 0.12 |
| its own voice to both ears | +0.37 a sounding tick (0.75–0.95 under heavy load), +0.05 at rest |
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
- It goes through the same cochleas and brainstem (channel 2). Its own voice arrives about 19–20 dB above the same sound from 1.5 m in front (the exact sphere's level for a source on its surface: +19.8 dB at 500 Hz, +19.1 dB for a held /a/, 78.7 dB SPL; the earlier "about 14 dB" was the prototype's 0.3 m level floor), and reads at the midline. There is no bone conduction.
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
  - Nothing rewards sounding like the parent. A songbird-style inner "matches the tutor" signal was refused as an intrinsic reward, and stays refused (A56).
  - **The tract's gate does carry the songbird's performance error** (A41; Gadagkar et al. 2016): the forecast's belief in the act it made against that act's usual belief, in the gate's credit only, never the reward. It compares the tract with its own past, not with the parent, and it is zero-mean once its expectations catch up (3.5).
- **The born cry** (A47) sounds through the same tract on pain or a low charge, and the tract's own acts override it. The child hears its cries, as it hears its babble.
- **The measured starting point** (`t_imitate.py`):
  - a linear inverse model trained on 16,000 babble ticks (1,082 sounding) fits its own voice poorly. Held-out R²: glottis 0.92; lungs, pitch and lips about 0.5; jaw and tongue 0.2–0.37;
  - it read 12 of the parent's words into echoes with a median rank of 22 of 50 (chance 25.5). The parent's ear accepts 2 of 12 in context ("see" 100%, "ball" 28%).
  - So early imitation rests on the cortex learning to hear across voices. `act_inv`'s reliability is measured on its own voice, so it overstates on the parent's.
  - A method for the parent: echo the child's babble back (Goldstein and Schwade 2008).

**How the parent hears the child's words** (`body/sim/parent_ear.py`, the study's prototype). Every figure below was measured through the study's cochlea (filters about 1.6 ERB wide), which the prototype still uses (`body/sim/parent_ear_cochlea.py`). P3v moves the parent's ear onto the child's cochlea (`ears.cochlea`, 1.00 ERB, calibrated) and measures them again before any is used.
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
| L0 motion | every physics step | interpolated poses; capped-spring holds; the yield (A4) | — |
| L1 reflexive | every tick | gaze to the child's act, to a sudden event, or to its gaze target; the face from the feelings; a hand withdrawn when hit | gaze ≤ 2 ticks (0.3 s); face ≤ 1 tick |
| L2 conduct | every tick, on events | judgments; replies; follow-in naming; offers and guides | voice 3–5 ticks; a hand act starts within 7 ticks (1 s). In lockstep, the solver's wall time costs no sim time. |
| L3 episode | at boundaries | the day plan's episode and its routine | — |

**Priority**, highest first: the child's pain (concern and comfort); being hit (withdraw); a low charge (the meal); the child's vocal turn (reply); finishing her word; judging a pending ask; joint attention; the episode's act; idle (watch, at most one line per 40 ticks).

**Contingency.**
- Every outward act of the child gets her gaze within 0.3 s, unless she misses it (below).
- A judged act gets the smile and the approval word within a tick.
- A vocal turn gets a reply 3 ticks after it ends on average, often echoing and expanding its babble.
- **She is imperfect, as a person is** (A52; her method, ours). Real dyads are coordinated only part of the time, and mismatch and repair are how an infant learns to cope (Tronick and Gianino 1986); 5-month-olds look longer at a view of legs that does not follow their own movement than at the perfectly contingent live view of their own (Bahrick and Watson 1985), which Watson read as a turn, after about 3 months, from the perfect contingency of the self toward the imperfect contingency of people. So she misses a share of its acts, her reply latency is jittered, and she has spells of distraction during her own tasks, each from its own seeded stream at rates read from the human data before birth and never fitted to the child's rates (C57). No miss ever touches a judgment she has made: a judged act still gets its smile within a tick.
- P6's ruler: her answered share of the child's acts within 7 ticks, while she is within 3 m and not away, equals her declared rate within its binomial error (C24). It checks that her method is as written; it is not a target.

**Joint attention.**
- The child's target, as she reads it (A40; B14 answered): the nameable object nearest the line its head's camera faces, within the fovea's reach of that line (±38° × ±20°), held 3 ticks running; or the object its hand holds, or reaches toward (the hand's path over the last 3 ticks closing on it). She reads the camera's line with a person's error, drawn per reading from her seeded stream (its size from a source on judging another's head direction, before birth: C58). A real G1 shows no eyes, so she reads what a person would see: its trunk, its head and its hands. The fovea's window stays the child's own and the instruments'.
- She looks there within 2 ticks.
- If her voice is free, the object is nameable, and it has not been named in the last 20 ticks, she names it with a variation set. Her eyes are on the object during the naming word only, then back on the child's eyes.
- **She copies its movements** (A52; her method, ours). Parents imitate their infants' acts, and being imitated is what builds the infant's own mapping from seen acts to its acts (Ray and Heyes 2011). When its arm or hand makes a visible movement (a raise, a wave, a shake, an open hand) while she attends, she makes the matching movement with her own arm, mirrored as she faces it, within 1–2 s, at a rate from the same data (C57). It earns nothing and asks nothing: a movement she shows it, which its own eyes may or may not use.
- She redirects ("look! the drum.", with a point) only after 40 ticks with no target. A redirect is a show, never judged as an ask and never a probe, so it may point (A51). Follow-in naming outnumbers redirects at least 2 to 1 (Tomasello and Farrar 1986).
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

**What Claude steers between minutes** (A14, extended; each "minute" is a steering window of 400 ticks since A59):
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
- **Textures** (A53). The toys, the furniture, the rug and the parent's clothes carry textures, so that a thing's identity is not its colour alone. In the first build only the floor, the window and her face were textured, and the eye check read the toys mostly by colour (the nearest mean 0.43 against a small readout's 0.85). A texture barely changes the render's cost (section 9).
  - The sun's shadow stays in the child's eyes. Dropping it would save about 21 ms a tick but take a piece of the owner's complete reality (decision 3). B3 is answered, and the shadow stays (A65), never decided by the eye check or by the tick.
  - The G1 file's own directional light (a Menagerie scene light, not part of the robot) is switched off at load.
  - **No headlight.** MuJoCo's headlight is a lamp at the viewing camera: for the child's eyes, a lamp shining from its own head, which the G1 does not have. It is off, and the eyes refuse to render with one (a verifier found it on; C3's first figures were measured with it). MuJoCo computes no bounced light, so each light carries the room's indirect light as its ambient term, a third of its own diffuse (ROOM_INDIRECT, ours: about the prototype's headlight ambient carried onto the lights).
- **The play space is the whole floor** (B7). The mat has no walls. The parent cannot carry the G1 back, so she comes to it.
- **The G1 in the file.** The room includes `assets/unitree_g1/g1_with_hands.xml` first, so its `stand` keyframe still addresses its own joints.
  - MuJoCo applies the last `compiler` element to the whole model, so the room is written in radians (every `euler` converted in one place), and the G1's mesh folder is given from `body/sim/`.
  - The room's contact defaults sit in their own class, so the G1's defaults are untouched.
- **Collision classes:** world 1 (floor, walls, mat, furniture); the G1 keeps its own contype 1 and conaffinity 1 (self-collision on, as shipped); toys 4; the parent 8, with conaffinity 1 so it touches the G1.
- **Physics:** MuJoCo 3.9, 2 ms steps with `implicitfast`, 75 steps a tick, **the elliptic friction cone, multi-point collision detection off, and impratio 10.**
  - With the pyramidal cone, a block squeezed in a Dex3 hand stopped MuJoCo ("FactorizeHessian: rank-deficient sparse Hessian"); with the conjugate-gradient solver or a dense Jacobian it gave NaN accelerations and a reset.
  - With the elliptic cone and multi-point detection on (MuJoCo's default), 8 of 360 grasp trials blew up to NaN about 0.1 s into closing the fingers, and MuJoCo silently reset them. With it off: 0 of 360, and the babble ran clean.
  - **impratio 10, chosen by the physics of these contacts** (A32), never by pain rates or by which toys can be held. Rubber fingertips on plastic toys, toys and the body on a foam mat, and motor housings on housings all hold without a slide below their sliding force. MuJoCo's soft contacts creep there by design (Overview, "Softness and slip"), and its documentation names the remedy: "using the Newton solver with elliptic friction cones and large value of impratio is the recommended way of reducing slip" (ibid.); "When contact slip is a problem, the best way to suppress it is to use elliptic cones, large impratio, and the Newton algorithm with very small tolerance" (Modeling, "Solver settings"; both the stable documentation, read 2026-09-24). Menagerie's hand and gripper models (the Shadow hand, the Allegro hand, the Robotiq 2F-85, ALOHA) ship elliptic cones with impratio 10.
    - Measured (`tools/sim_friction.py`): a box on the mat pushed at 0.3–0.97 of its sliding force slid 3.6–21 mm in 2 s at impratio 1 and 0.4–1.8 mm at 10 (a real one would not slide); the resting G1 pushed at the pelvis with 100 / 150 / 200 N slid 1.9 / 3.1 / 4.4 cm at 1 and 0.4 / 0.7 / 1.9 cm at 10; above the sliding force both follow Coulomb's law within 1%.
    - **Its cost, disclosed** (C5, C22). The convex contact model couples a slip to the normal direction ("the only way to initiate slip is to generate some motion in the normal direction", Computation, "Physical realism and soft contacts"). A pressed box that starts to slide is pressed 22–24% harder than its load for about 10 ms, at either setting. In the G1's hip housings, pressed together and slid apart by the withdrawal, friction raises the normal force at 10 (1,795 N against 1,553 N with the pair frictionless) where at 1 it lowers it (1,333 against 1,646): a pressed housing that slides reads harder at 10.
    - The noslip solver, the documentation's next step, was refused: its cascade "is no longer solving a well-defined optimization problem (or any other problem); instead it is just an adhoc mechanism" (Modeling, "Solver settings"), and it nearly doubled the tick.
  - The world catches `mujoco.FatalError`: two hands posed 3 cm into each other raised it inside `mj_forward` (A18). It also disables MuJoCo's auto-reset (`<flag autoreset="disable"/>`), so a bad state is never silently replaced. Built (W1): every tick checks MuJoCo's warning counters and the end state against MuJoCo's own bound (`mjMAXVAL`, 1e10) on positions, velocities and accelerations, with the tick's closing forward pass inside the catch; on any fault the world puts back the state the tick began with and raises `WorldFault`, so the fault never reaches the body.
  - The world's geoms (floor, mat, furniture, toys) take contact priority 2, so real frictions are set on the world's side (built; the surfaces' own values are still 1.0, C26).
  - **The mat's collision box reaches 10 cm below its top,** sunk into the floor. In a thin mat a 5 mm foot sphere was trapped at 1,900 N and gave pain on 151 of 400 babble ticks; with the deep box, 2 of 400, both real blows.

### 5.2 Toys and their sounds

Every sound comes from an event in the physics: a toy's own clip on its event, scaled by the impulse or speed; contact clicks; the parent's voice and footsteps (from its gait); the child's tract. Each source is placed in space for the two ears.

| toy | colour | size | held by a Dex3 hand? (3.8, impratio 10) | its sound |
|---|---|---|---|---|
| ball | red | 12 cm | yes, 9 of 9; also pushed, rolled and named | a soft rubber bounce when it lands or is struck |
| block | blue | 7 cm cube | yes, 9 of 9 | a hollow wooden knock on contact |
| duck | yellow | 11 cm | yes, 7 of 9 | a squeak when squeezed |
| cup | green | 0.8 scale, 6.7 cm across (B1) | yes, 9 of 9 (and 9 of 9 at full size) | a plastic clink when it strikes something |
| rattle | purple | a handle and a head | yes, 6 of 6 hand-overs | beads shaking, louder with speed |
| car | orange | 14 cm long | yes, 8 of 9 | a wheel rattle while it moves |
| bear | brown | 12 cm | no, 2 of 9 (B1) | a soft bell inside when it moves |
| stacker | white, with coloured rings | 11 cm base | yes, 9 of 9 | the rings' plastic clack |
| drum | cyan | 15 cm | no, 0 of 9 (B1): hit | a boom when hit on top |
| ring (teether) | pink | 10 cm | yes, 6 of 6 hand-overs | a crinkle when handled |

- **Where they lie at birth:** around the G1 on the mat. The block, cup and rattle are within its lying reach. The rattle was moved 7 cm nearer the right hand, and the block and duck were moved out from under its arm and leg.
- **Colour twins** for the never-taught word test (B2): a blue ball, a red block, a yellow cup and a green car, each a copy of its toy in another colour, introduced before the colour words. The twins are the test's targets, and each colour word is heard on at least 2 kinds of toy before it (A28, A55).
- **A richer room** (A53; the lead's decision for the owner, whose room it is):
  - **Containers:** a hollow cup and open baskets built from box geoms, so that "in", a birth word, can happen.
  - **A cover:** a light cloth-like lid over a shallow tray that a Dex3 hand can lift (measured before birth, C60), for hiding games and M3's search test (section 12).
  - **At least 3 examples of every tested noun** (Quinn, Eimas and Rosenkrantz 1993: infants form perceptual categories from several exemplars at 3–4 months), differing in size, colour and texture, so that a noun cannot mean one object. No example repeats a held-out pair of A28 (no second blue ball, red block, yellow cup or green car), so the twins stay the only objects that pair their words.
  - **New objects by calendar:** a seeded inventory, fixed before birth, adds one new object every 3 life days and rearranges the clutter, by recompiling the room at a night boundary. The calendar never waits for, or hurries with, the child's progress. The first build's toys had already made some probes novel (A28); this makes novelty last.
  - Each new object's holds by a Dex3 hand, its sounds and the render's cost are measured before it joins (C60).

### 5.3 The charger

- **The bottle.** A toy-sized bottle charges the child while it touches either palm: 0.01 a tick, so from empty to full in 100 ticks.
  - It is sized for the Dex3 in W3 (not yet built; the hand now holds the cup at full size, 3.8).
  - Its dock is the prototype's white disc. The maker still has it at the mat's corner, far from the child; W3 moves it beside one hand (A17; not yet built).
  - The dock moves out of reach only once moving across the mat toward things beats chance.
- **Why a bottle** (the owner's decision 6 calls charging the child's meal):
  - the palm and the grasp reflex make feeding possible from day one;
  - "bottle" is a first word;
  - the parent's feeding pairs her face with relief.
- **When the charge is low,** the parent brings the bottle. An empty child far from the bottle almost never reaches it, because weakness makes it worse.

### 5.4 Day, night and a complete reality

The owner's decision 3: sounds only from real events, day and night light, the parent's whole day as a parent, and no shortcuts in the body.
- **Light across the life day.** The sun moves across the window from morning to dusk over the 24,000 ticks, with its colour warming at the ends of the day. The lamp comes on at winding down and dims over goodnight's last 30 ticks. The night is dark. The morning's light returns over the first 30 ticks of the wake.
  - The eye check (C3) runs under morning, midday and dusk light. It reports what the fovea can tell apart in each light; it never changes the light. Making the world easier to see because the child's eyes find it hard would fit the world to the learner. Its lights are stand-ins until W5 builds the day's; it runs again then.
- **Night: live and dark** (A46; changed by the lead's decisions: the G1 amendment froze the world with `world.pause()`). At the end of goodnight the world does not freeze. It runs on, dark: the lights dim to the night's, the parent sleeps on the sofa (her kinematic body still, touching nothing), and the child sleeps as it lies, its gates closed and its servos at rest under the resting law.
  - The night lasts 24,000 ticks of sim time (NIGHT_TICKS; one simulated hour, as long as the day, B19's compressed hour for the night as for the day). Its REM phases, about half of it (active sleep is about half of a newborn's sleep: Roffwarg, Muzio and Dement 1966), carry the brainstem's twitches (3.7), whose reafference teaches `act_inv`, the forward half and the cerebellum. The core's night passes (NREM replay, REM on frames) run as before; REM stays on (the owner's ruling).
  - Asleep, it neither sees nor hears: the eyes are not rendered at night and the ears are not run. The joints, the observer and the inertial units are, around each twitch. Nothing it senses at night reaches a reward, a critic, the amygdala or a gate. The withdrawal still acts (it is spinal and lifelong), and a pain at night is logged.
  - The charge's basal drain stops at night, and the twitches' effort drains as by day (ours, disclosed). At 4e-5 a tick, a night of 24,000 ticks would drain 0.96 of a full charge, so a child fed at bedtime (A17) would wake empty every morning, where the G1 amendment's paused night drained nothing; a real G1 rests docked or powered down. It is a rule of the world's night, fixed before birth, never set on the child's feeds.
  - The morning resumes from wherever the night left the world: a twitch may have moved a limb, and nothing else moves it.
  - The night's physics takes about 6–7 wall minutes (24,000 ticks at 8.4–10× real time, the eyes dark), measured at S5b with the core's night passes (C14, C52).
  - Measured for W5: with every light switched off, MuJoCo renders the scene unlit and brighter (a mean grey of 95 against 39 lit), so the night dims the lights rather than switching them off; the ceiling, the window, the lamp's shade and the dock are emissive and still glow. The save carries the lights' places and directions, and the eyes re-render when any light changes.
- **Sounds only from real events** (section 5.2). The room's echo is B9's default. The G1's own motors, which hum on the real robot as they work, are B20's.
- **No shortcuts in the body.** Every channel comes from a sensor the real G1 has, where it has it, with the two disclosed scaffolds of section 3.4 (the word tokens and, until the child's own pixels carry it, the world-truth smile).

## 6. Reward

Summed in this order.

1. **Face** (the increment rule, for the sim body). A rise of the reading's positive part is felt as positive, and a rise of its negative part as negative; no fall is ever felt. It is clipped to ±2.
   - It is felt only through the born expression reading, which updates only while the parent's face passes the face test on 2 consecutive ticks (A1). An unseen smile is not felt, and a face seen again unchanged is not felt again.
   - **Its carrier moves to the child's own pixels** (A49). Once the born face detector works (C39), the reading is the born mouth-corner reader's on the fovea's pixels, and it updates only while the detector finds her face in a fovea on 2 consecutive ticks; the world's value and the ray test leave the body. Until then the world's value is a disclosed scaffold: the face term takes it while the world supplies it and the reader's own reading when the world falls silent, a rule fixed from birth, and the world falls silent when the removal test passes (A49). A real G1 has no world truth to read, so the robot needs the pixels.
   - For steps from neutral it equals today's rule (a rise felt as the new level), which the language body keeps.
   - Why it changed: today's rule over-pays a graded onset. Measured on the graded face, a watching child felt 276.8 of 185.5 judged (+49%).
   - It is grounded in the parent's visible judgment of world events.
2. **Pain** (changed by the lead's decisions: A37). −1 on a tick when the observer's estimated outside torque on any joint passes that joint's own torque limit, or the free base's estimated outside force passes F_pain = 3 × the child's weight from the model file (3 × 337.4 N ≈ 1,012 N).
   - A load from outside larger than a joint's motor can hold back-drives its gear: the robot's own damage line, taken from its declared model (its file's limits) as F_pain was from its declared mass. The base has no motor, so F_pain's rule stays there.
   - Each estimate for pain is its tick's largest 10 ms mean (A12's filter, the observer's own 10 ms step).
   - It is read from the robot's own sensors (3.4), so pain is no longer world truth. The Dex3 hands' arrays carry touch, not pain: they saturate far below any damaging force, so a hand's pain comes from its joints, as everywhere else.
   - **On the old skin** (a link's summed normal force over F_pain): at rest the largest contact at the tick ends was 272 N; under babble, single-step peaks at the tick ends reached 944 and 1,799 N. Measured in the built world with the filter and the newborn's withdrawal (`tools/sim_pain.py`, 8 seeds × 400 ticks): pain on 11.6% of babble ticks at impratio 10 (11.7% at 1; 6.1% on a verifier's 4 seeds), mostly the thighs' housings driven into the pelvis and the shoulders into the torso; 0 of 400 ticks at rest (C5). Under the joints' law it is measured again on the born loop, with the share from hulls meeting (C43, A54).
   - **At a range's end.** A joint driven against its range's end meets the stop's reaction, an outside torque the observer reads as a real robot's would; it hurts only past the joint's own limit, under the same law. No separate pain for a joint held at its range is added; that stays a candidate, not at birth. W4 writes down the share of the joints' pain at the ranges' ends (C43, C59).
   - **Where a posture loads a joint past its limit, it hurts.** On hands and knees, flat palms need about 4× the wrists' 5 N·m (section 0), so crawling on flat palms would hurt at the wrists under this law, as the gear would be back-driven on the real robot; on fists it nearly holds. W4 writes down which postures and holds pass a joint's limit (C43).
3. **Charge.** 4 × [D(h_t) − D(h_t+1)], with D(h) = (1 − h)². This is drive reduction (Keramati and Gutkin 2014), so a full charge earns nothing.
   - The drain is 4e-5 a tick plus 2e-3 × (Σ τ² ÷ Σ τ²_max), over the G1's own limits. At night the 4e-5 stops, and only the twitches' effort drains (A46).
   - Under babble (0.063) a full charge empties in about 6,000 ticks and falls to the feeding level (0.35) in about 3,900, so about 5–6 feeds a day. At rest (0.004) it takes about 13,500 ticks, so 1–2 feeds a day.
   - Only the child's own actuator torque counts, so being guided costs at most its cap.

**Left out:** any reward for moving, getting closer, looking, hearing words, sounding like the parent, novelty, learning progress (A56), or anything read from the body's insides.

**Not a reward, and disclosed:** the gates' two intrinsic terms, the tonic drive that follows the reward rate and the tract's performance error, enter only the gates' own credit (3.5, 7.3, A41).

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
  - R7 adds event lines for touch onset by group (from the observer and the hands' arrays since A37), pain, a face in the fovea, a sound onset on each side and a visual onset on each side (A43). The value can then rise at the causal moment, not only when the smile is finally seen. The amygdala reads the same lines.
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
- Reward chooses which whole-body patterns are replayed; supervised learning spreads them over the joints. There is no per-limb reward.
- **Two intrinsic terms, disclosed** (A41; amended by the lead's decisions: this line said "no intrinsic bonus", and the code's gates carried a tonic drive all along, undisclosed). Every gate's credit carries a tonic drive that follows the reward rate: 0.25 + 4.66 × R̄ per act, Niv et al.'s (2007) opportunity cost of time. The tract's gate alone carries the performance error against its own usual act (Gadagkar et al. 2016). Neither is a reward: neither reaches the critics, dopamine, the amygdala, the store or the night's draw, and neither is paid for anything the parent sees (3.5). A learning-progress bonus was proposed and dropped (A56).

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
| low (thalamic) | R7's event lines, one declaration read by both the critics and the amygdala: touch onset in 7 groups (the trunk with the head and the pelvis, each arm, each hand, each leg), from the observer and the hands' arrays (A37; the observer cannot tell the head from the torso, so the first build's 8 zone groups become 7); pain (any joint, or the base); a face in the fovea (the born three-blob template on the fovea's pixels, never the world's face test: section 3.4; its constants open, C39); a sound onset on the left and on the right (the cochlea's onset and the born lateral read; a sound with nothing below about 760 Hz gets no side, C42); a visual onset in the periphery on the left and on the right (the born local-change cue, A43) | 13 |
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
- **τ_a = 4,096 ticks** (band 6's clock), about 10 minutes of life: about eight times the input width (525 inputs in the G1 amendment, 526 with the lead's 13 event lines), so the fit is determined.
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
3. **Orienting:** each born orienting trigger's bias (the face template in the periphery; a sound onset's side; a sudden local change in the periphery, A43; acting on the gaze and the waist) is multiplied each tick by g = clip(1 + N, −0.5, 2).
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

### 7.5 The cerebellum: a loop below the tick (R6c; A44)

Status: specified; nothing is built. It is decided before birth, because a loop below the tick added later would be a change of body, so a new seed (A20; risk 2).

**What it is.** A named organ of the core, `body/core/cerebellum.py` (`CerebellumMixin`), with its constants in `CEREB` in `physiology.py`. It runs every 5 physics steps (10 ms) inside the tick, through a sub-tick hook the world calls from `SimWorld.apply` (R6c adds the hook to the World interface; the diary's world never calls it). It is the Marr–Albus cerebellum (Marr 1969; Albus 1971), read as an adaptive filter (Fujita 1982):
- **Mossy fibres** (its input at each sub-step): for the 29 joints of the arms, the legs and the waist, their angles, velocities, estimated torques and the servo's current targets (the efference copy of the tick's act: 116); both inertial units (12); the hands' touch (16). About 150 numbers.
- **The granule layer:** a born random sparse expansion, fixed from the body's seed: 4,096 units, each reading 4 inputs, thresholded to about 10% active (ours, after Marr's codons and Albus's expansion; settled from their sources before birth, C50).
- **Purkinje cells:** per joint of the arms, the legs and the waist (29; the hands' joints have none), a linear readout of the granule layer, whose output is a torque added to the servo law's (inside the joint's limit and the weakness clip); and two readouts for the VOR (the flocculus): a gain correction and an offset on the window's counter-shift.
- **Climbing fibres, its two teachers:**
  - **the servo law's own corrective torque** (feedback-error learning: Kawato and Gomi 1992): the torque the servo spends pulling the joint back to its target. The readout learns to supply it before the error appears, so a limb carrying a load (its own weight when raised, a toy in the hand, her hold) tracks its target with less error;
  - **retinal slip** (Ito 1982): the fovea's image motion over a tick while the gaze held and the VOR acted, read from the grey fovea's centre-surround map (its shift between ticks). It teaches the VOR's gain, and an offset that cancels the gyro's drifting bias (A39). Slip is seen once a tick, so the flocculus learns once a tick.
- **Its law:** least mean squares, w ← w + η e x, with η normalized by the granule layer's activity (ours, C50). It learns wherever the world runs: awake, and on the live night's twitches.
- **What it cannot do.** Feedback-error learning needs an innate feedback controller for the task it learns (Kawato and Gomi 1992). Its teacher knows joint angles only, and righting and equilibrium reactions are refused (3.7). So it learns load compensation and the VOR's gain, never a sit or a balance: balance stays the cortex's to learn through the tick, and risk 2 stands.
- **What it may do, disclosed.** A limb left to rest sinks against the servo's lagging target (3.3), and that lag is a corrective torque, so its teacher is not silent at rest: the readout may learn part of a limb's own weight and slow a rested posture's sink. That is a learned compensation below the tick, never a written one, and never a balance. R6c's tests and W4 write down the sink rates with it learning (C50).
- **Nothing else:** it never proposes an act, never enters a gate's credit, never reads reward.
- **On the real robot** it runs at 100 Hz on the robot's own computer beside the servo loop, from the same sensors.
- **Replay:** it draws no random numbers after birth, its weights are in the save, and the world calls it at fixed sub-steps, so a replay is exact.
- **Cost:** est. about 1 ms a tick (15 sub-steps, a sparse 4,096-wide expansion, 29 joints' readouts); measured in R6c (C50).
- **Tests** (R6c, a tiny body): inert for language (the eight digests; no attribute, save key or random draw); the law exact against numpy; a load step on a test limb, where the servo's corrective torque shrinks as the readout learns; a biased gyro, where the slip falls toward zero; the save round trip; its cost.
- **Biology.** The cerebellum grows about 240% in the first year (Knickmeyer et al. 2008) and calibrates reaching, posture and the VOR. We give it only what its teachers can teach.

### 7.6 Recall into action (R7f; A45)

- R7b makes the store write frames. R7f gives a frame its key and its value:
  - **the key:** the cortex's stream, plus a heading, the trunk's yaw integrated from the torso gyro since birth (a head-direction signal by path integration: McNaughton et al. 2006). A real gyro drifts (Woodman 2007), so the heading drifts too: disclosed, and reported (C51);
  - **the value:** the next frame's codes, and the efference copy of what the body did next (every effector's act).
- **Recall.** Each tick the store's nearest keys give back their values, as it gives back words today. The recalled efference copies enter each effector's proposal through a map born at zero (one per effector, from the recalled act's embedding to its logits), which learns through `act_pred`'s own lesson. So recall moves an act only as far as recalled acts have predicted the acts made: episodic control, the hippocampus's "third way" into action (Lengyel and Dayan 2007). Before R7f, recall reached only the words.
- **The working-memory latch** moves from utterance ends to R7's event ends for the sim: the settle law on the summed forecast error (section 10's "event end"), which R7b defines for frames. The language body's latch is unchanged.
- **Cost:** est. under 1 ms a tick (the store's existing search; ten small maps); measured in R7.
- **Tests:** inert for language; at birth the maps leave every proposal exactly unchanged; in a scripted world where a recalled act predicts the next one, the map's weight grows.

## 8. The refactor: where it stands, and the steps left

**Done on `sim-core`** (worktree `$S/wt_sim`), every language digest held at every commit:

| step | commit | what |
|---|---|---|
| R1 | d05a24f | the guard pinned, the anatomy declared |
| R2 | 96f72a9 | the anatomy replaces `tok` |
| R3 | cba7179 | reward as the anatomy's ordered sources |
| R4 | e948e7b | channels in the core: window per channel, the ordered input sum, a forecast head per channel |
| R5 | 0562112 | effectors: the voice as effector 0, per-joint readouts, striatal blocks appended, fixes #4, #5, #8 as switches |
| R9 | 2489e89 | the world loop: World, Frame, DiaryWorld with the pace log, the SimWorld interface, night pauses the world (R8 makes the sim's night live and dark: A46) |
| R6 | 19a8832 | the motor timing part (act_pred, the forward half, its correction, act_inv with its reliability); learned stops with `chunk_gate` per effector; reflex hooks (a forced act with no eligibility); 118/118 organ tests and 27/27 anatomy tests |
| R6 fix 1 | fdeb605 | `act_pred`'s lesson weighs a rest's label by `act_inv`'s reliability absolutely |
| R6 fix 2 | 974d865 | `act_inv`'s reliability is Cohen's kappa per joint over its running confusion |
| R5b | 0103f0a | the striatum holds a later effector's events by joint, not by flat act |
| R6 fix 3 | e48b284 | `act_pred`'s plasticity gated by its labels' reliability (`GatedAdam`) |
| verifiers' findings | 6d6d246 | a striatum saved before R5b keeps all but its effectors' rows; `act_inv`'s kappa under shifting act rates measured, its correction left for R6h |

- **R6 fix 3's verifier said NOT READY.** `GatedAdam` is right, and Adam bit for bit at weight 1, but the gate held only for a newborn body: once `act_pred` had learned, 1,000 lessons at reliability 0.01 taught the proposal about ten times what the same label mass taught given whole, through `act_pred`'s gradient into the cortex. R6 fix 4 (440bad3) closes that route and gives each optimizer its own gradient bound; its verifier found one route still open (windows mixing own acts and labelled rests, once the own acts are fitted, through Adam's moments), with a remedy probed (separate moments for `act_inv`'s labels). Both are in progress on `sim-core` and are folded here when a verifier passes them.

**Left:**

| step | work | days | risk to the digests |
|---|---|---|---|
| R6h | movement units (the persistence margin), with the born unit lengths measured under the continuation draw (C38); the kappa correction from 6d6d246; fatigue per effector; each limb's forward error into its gate; `act_inv`'s lessons batched every 8 ticks; born encoders `encs.<name>` from the body's seed, at the new channel sizes (3.4); an effector's declared consequence sense (the voice's is the ears); the gaze effector with no joints (the window's state); the orienting and VOR hooks on the gaze and the waist. **Added by the lead's decisions:** the gates' drives set and disclosed (the tonic drive following the reward rate in every gate; the performance error on the tract's gate only, per articulator; `gate_vigor` 0: 3.5, A41, C61); the spinal pattern generator per limb and the born cry, both summed below the gate like the grasp, with their reflex ticks logged (A47, A48); the visual onset's orienting hook beside the face and sound cues (A43) | 3.25 | none (off for language) |
| R6c | **new:** the cerebellum (7.5, A44): the sub-tick hook in the World interface (`SimWorld.apply` calls the body every 5 physics steps; the diary's world never does); `CerebellumMixin` with its born expansion, its Purkinje readouts for the limbs, the waist and the VOR, its two teachers (the servo's corrective torque, retinal slip) and its law; its tests | 1 | none (absent for language) |
| R7 | R7a: the anatomy's event lines (`Anatomy.events`, a frame field read by the striatal expansion and the amygdala), 13 for the sim (7.4). R7b: surprise-gated writes at frames; marks at event ends, the event end defined for frames; each tick's record (surprise, δ, tag, net received reward: 16 B a tick). R7c: fix #1 as a switch; fix #6 as the switch `tag_trace`; each channel's forecast error scaled by its own running mean; pace on the partner channel. R7d: the amygdala (switch `amyg`), built last in `Organs`, after `_learn_values` and `_own_face` and before `_choose`; its tests. R7e: the orienting gain on all three born cues; `amyg_pav` built and off. **R7f, added:** recall into action (7.6, A45): the frame's key with the gyro's heading, its value with the efference copies, the maps born at zero into each proposal; the working-memory latch on R7b's event ends | 4 | medium |
| R8 | the night over frames: per-channel batches from stored codes; the episodes' entries (T_e from tag* over the tape at nightfall); the tagged dreamt first; the window at the peak; every effector's acts replayed as efference copies and `act_pred` targets, weighted by the replayed dopamine's credit; `act_inv` and the forward half replayed over the day's transitions; REM on frames (REM stays on); the entries' running mean saved; a tape of about 7.7 KB a tick. **Added:** the live, dark night (5.4, A46): the World interface's night steps the world dark instead of freezing it (the diary's world still pauses); the born twitch generator in REM's phases; each twitch's pair taught to `act_inv`, the forward half and the cerebellum | 4 | medium |
| — | the full guard, with `--roundtrip`; the `sim` profile pinned | 1 | — |

That is about 13.25 working days of core work (it was 9.5 before the lead's decisions: R6c adds 1, R6h 0.75, R7 1 and R8 1).

**The guard, unchanged.** Every refactor commit leaves all eight pinned language digests equal. They are in `tools/pins/digests.txt`, run from the main tree's directory and importing the branch's body:

| profile | threaded | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` |
|---|---|---|
| default | `7c54199e3c72cf77d45a8e94` | `fbe9e6b330f44e7424f113f3` |
| served | `fc81c03dec606afb62c3fc35` | `f507fb9c6c55649ddd39f917` |
| switches | `ab1ab46751b11b314dfc78c2` | `14de2b131ce4b2874ca0ad95` |
| served `--full --roundtrip` | `c4730b232091fe229ee286dc` | `823431e80a08f197ee086884` |

- **New code is inert for language.** The language anatomy builds no new module, draws no new random number, runs no new branch, and adds no attribute, window field or save key. The amygdala, `tag_trace`, the cerebellum, the pattern generator, the cry, the visual onset cue, recall into action and the twitches are absent from its settings, and its world never calls the sub-tick hook or runs a live night.
- **New modules come last,** built from generators of their own seeded by the body's seed.
- **After the plumbing run,** a `sim` profile joins the check. It runs a tiny body on the SimAnatomy, fed a recorded 200-tick script of raw observations (about 4 MB in `tools/pins/`), so the guard covers the core without the renderer. SimWorld's own exact-replay test covers the physics.
- **Fix #6 for the language body** (`tag_trace` with the felt entry of an utterance) stays off unless measured on a copy; the review measured it only together with `chunk_gate`.

**The language body.** It is paused. It moves to the refactored code whenever it restarts, after:
- the eight digests under its live constants;
- a 400-tick read-only load check (no save path to `data/watch2.pt`);
- a save round trip from new to old.

Nothing in the sim waits for it.

## 9. Compute, memory and disk on this Mac, and how fast the life can run

M4 MacBook Air (fanless): 10 cores, 16 GB RAM. One torch thread until a quiet window measures more. The body runs at `nice 10`.

**Wall time per tick.** These were measured under the moderate load of other sessions: about 3.7 where the load was recorded (3.8). They are conservative against that load only. At a load of 10–15 the eyes' render ran about 4× slower (the speed plan, below):

| part | ms | source |
|---|---|---|
| physics: 75 steps of the G1 under babble, about 28 contacts | 12.3–15.0 | measured (the prototype, and the built world) |
| the world's Python: the servo law, touch and pain per zone with the 10 ms filter, the IMUs, the reflexes | 2.3–2.6 | measured (W1) |
| the parent's mocap and its Python, every step | 4.3 | measured |
| the parent's plan (solved per act and once a second, warm-started, interpolated) | 1–3 after W2 | estimate |
| two eyes, 168 × 96 each, the sun's shadow: render and read-back | 31.2–31.7 with the parent's old face; her face of human proportions makes it about 25–30% slower (35 → 43–46 side by side), so about 40–45 (16.5 without shadows, the prototype) | measured (W3) |
| the eyes' split into periphery and fovea, the retina's code, the born face template (3.0) and the face test's rays | 0.7–3.5 | measured (W3) |
| ears: three sources, both cochleas, the brainstem | 1.4 (0.23 + about 0.37 a source; 2.8 under heavy load) | measured |
| the child's tract: the babble average, plus its sound to both ears | 1.6 + 0.1 (its sound: 0.37 a sounding tick, 0.05 at rest) | measured |
| the parent's ear: 287 ms per child utterance, at the born babble's 6.4–7.8 utterances a minute | about 5 on average (4.6–5.6; 287 ms at an utterance's end) | measured per utterance; the rate from the born babble |
| the parent's voice | 0 on a cache hit | measured |
| the core at the humanoid's size, at the critics' current solve rate | mean 56–84, median 39–53 | measured (paired against the language core: 84.4 against 74.3 mean) |
| the critics' solves every 256 ticks instead of 64 | −7 to −20 | measured parts |
| the amygdala | 0.19 | measured (synthetic) |
| not yet measured: the parent's conduct layers L1–L2 and feelings each tick; the room camera while watched | about 1–5 | estimate |
| **total before the lead's decisions** | **about 105–160 mean with the sun's shadow; about 80–135 without** | from about 0.9× to 1.4× real time |
| **added by the lead's decisions** (A57): the grey eyes at 336 × 192 each (A42) | +0.3–0.9 against the prototype's own render (the twice-sharp measure below: 16.9 and 38.3 ms); up to about +6 against the built world's 31.2–31.7 | measured on the prototype; on the built eyes, C48 |
| the colour camera's own view, a third render with the sun's shadow and her face (A38) | about +10–15: a third view's fixed overhead, from the two views' 40–45 | estimate (C48) |
| the fovea's born bank, the colour code, the camera model on three images | about 0.5–1.5 | estimate |
| the observer every 10 ms (A37) | about 0.3–1 | estimate (C43) |
| the motor models, the gyros' bias and the motors' heat (A39) | under 0.1 | estimate |
| the visual onset cue (A43) | about 0.1 | estimate |
| the cerebellum every 10 ms (A44) | about 0.5–1.5 | estimate (C50) |
| recall into action (A45) | under 1 | estimate |
| the pixel smile reader (A49) | 0.1–0.3 | estimate |
| a convex-decomposed collision copy, only if W4 needs it (A54) | +1–3 of physics | estimate |
| **total with the lead's decisions** | **about 117–190 mean with the sun's shadow; about 90–160 without** | from about 0.8× to 1.3× real time; the roadmap's +2–7 ms did not count the colour camera's view |
| **with the exact levers and the lighter meshes for its eyes** (A64, A65) | **about 65–135 mean with the sun's shadow** | from about 1.1× to 2.3× real time; estimates (the speed plan's case C, below) |

- **Against this morning's design** (80–125 ms): the G1's physics adds 4–5 ms, its eyes 8–13 ms with the sun's shadow (render and read-back: 32.5 ms against the custom child's 24.2 ms in the same run), the tract about 2 ms, and the parent's ear about 2–5 ms.
- **Her face costs the eyes 8–11 ms.** The parent's face of human proportions (a smooth sheet and its pieces: 859,000 mesh faces in the model against 680,000) made the two eyes' render about 25–30% slower in a side-by-side measure (35 → 43–46 ms); `mj_step` did not change (0.33 ms). Her face is the world's, not a render shortcut, so it is not one of B3's levers.
- **Half the eyes' cost is the G1's own visual meshes** (629,000 triangles). Hiding them (a test only) gave 8.3 ms without shadows and 16.7 ms with the sun's. A lighter visual copy of the meshes, used only by the eyes, would save about 8–18 ms over the two grey eyes, and about half as much again in the colour camera's view. It is taken, within the eyes' own noise and a pixel, so the G1 looks the same to itself (A65, C69).
- **A twice-sharp fovea** (336 × 192 per eye) costs 16.9 ms without shadows and 38.3 ms with the sun's: resolution barely matters; the fixed overhead does.
- **Levers for the tick,** in order: the parent's ear scoring only the expected words and the bank (an estimate of −1 to −3 ms; the world's side, ours); the exact levers of the speed plan below (A64); and B3's render choices, answered by the lead (A65). The lighter mesh copy is taken (about −12 to −27 ms over the three views). The colour camera's image cut from a colour render of the left eye is refused: it would save about 10–15 ms, but its 15 mm of misplacement is 2.9° of parallax at 0.3 m and 5.7° at 0.15 m, where its hands reach. The eyes without the sun's shadow are refused (−21 ms; decision 3). Past these, a slower life is the alternative, since lockstep makes the tick cost wall time only.
- **A life day:** 24,000 ticks take about 47–76 wall minutes awake at the new tick. The night is estimated at 12–25 minutes for the core's passes (R8 is not built), plus about 6–7 for the live, dark night's physics (5.4). So a life day with its night takes about 65–110 wall minutes (55–90 before the lead's decisions).
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
| the save, written as `.tmp` then renamed | about 1.3–1.7 GB (up to 3.4 GB at peak with its `.tmp`); the amygdala adds 2.3 MB. The earlier 0.95 GB had no source and is below the language body's own save (1.28 GB on disk today), while the sim's core has more parameters (35.2M against 28.8M), a larger store (98,304 slots against 65,536: +134 MB) and the day's tape (151 MB; 184 MB with the lead's channel sizes). S5b measures it | one |
| the day's tape (about 3,830 numbers a tick with the lead's channel sizes, fp16: 7.7 KB; 3,150 and 6.3 KB before) | about 184 MB a life day (151 before), inside the episodes (cap 24,000 ticks); the night keeps only the twitches' windows | in the save |
| each tick's record for the amygdala (16 B) | 384 KB a life day | in the tape |
| world states for exact replay and redrawn films | about 7.6 KB a tick, 180 MB a life day; the live night's states every 100th tick only (a night replays exactly from its start and its streams), about 2 MB | the last 3 days, 0.55 GB |
| the cerebellum's weights and recall's maps | about 1 MB | in the save |
| the voice cache (the life's folder) | lines made ahead: at most 300 MB. Lines heard: kept for good, 40 KB each (13.3 MB for the 331 birth lines), growing by at most about 25 MB a life day | always |
| the parent's JSONL log | rotated at 100 MB | the last file |
| **peak** | **about 4.5 GB** (4.4 before the lead's decisions: the tape's growth) | |

- The tract's sound is not saved: the tract is deterministic from its seed and the acts, so a replay regenerates it.
- The voice's kept lines grow with the life: at most about 25 MB a life day (0.75 GB over 30 days if every line were new), the day's fresh lines in practice. The disk rule counts them.
- **The rule.** The sim writes nothing that would leave less free space than the larger of the two bodies' saves (the sim's, about 1.7 GB until S5b measures it) plus 1 GB: about 2.7 GB. It skips the write, logs why, and pauses at its next boundary.
- **Birth needs at least 8 GB free:** the peak of about 4.5 GB plus the rule's floor of 2.7 GB, and a margin.
- **The owner's folders** `data/backups` and `data/first_lineage` are never touched.

### How fast the life can run on this Mac: the speed plan (A59, 2026-09-24; amended on 2026-09-25 by A61, A64 and A65)

Local only, with no cloud, ever (A61). Fast learning is counted in the body's own life hours (A62); this plan makes each life hour take as little wall time as this Mac can give without changing anything the body senses or does. The arithmetic is in `$S/pace/calc.py` and `fold.py` (section 17), with case C below recomputed for (a) alone. Every figure marked "estimate" is unmeasured.

**The real-time factor on this Mac:**
- **0.8–1.3× real time on a quiet, cool machine.** That follows from the tick of 117–190 ms above: a life day with its night takes 65–110 wall minutes.
  - Sustained speed is unmeasured. The Air is fanless and throttles under a long load, so S5b's two heat-soaked hours decide it (C12).
- **Under other sessions' load it is slower.** On the evening of 2026-09-24, at a load of 10–15, the prototype's two eyes with the sun's shadow took 142 ms to render: the median of 40 renders (`$S/pace/rbench.py`).
  - `mjv_updateScene` took 0.11 ms and the read-back 5 ms. Against the quiet 31–38 ms that is about 4× slower, and the whole cost is the GL draw (Apple deprecated OpenGL in 2018).
  - One scene update shared across both views gave bit-identical pixels but saved nothing. Repeated renders were bit-identical.
- **This is the owner's working laptop.** The calendar below assumes the life runs around the clock at its quiet tick, so it must be divided by the share of wall time the life actually gets (a duty factor).
- **In lockstep a slower tick costs wall time only,** never anything in the life (A18).

**Where the tick goes** (from the table above; 117–190 ms in all):

| part | ms a tick |
|---|---|
| the core at the 256-tick solves (about half of it the critics) | 36–77 |
| three renders | 50–66 |
| physics, the world's Python and the parent | 20–25 |
| everything else | 12–23 |

**The exact levers, decided, each with its build step** (A64). None of them changes anything the body senses or does.

| lever | saves (estimate) | built in |
|---|---|---|
| the critics' rank-one updates on a second performance core: they feed only the next solve, so they are joined before each solve and each save | 14–26 ms | R10 (core, day 15) |
| the parent's ear off the tick, joined at her reply tick | about 5 ms | P7 (parent, day 13) |
| the ears, the tract, touch and the observer computed while the GPU draws | 4–8 ms | W7 (world, day 13) |
| the world's per-step Python compiled, bit-equal to the Python it replaces | 2–4 ms | W7 (world, day 13) |
| the live night's physics and the replay on separate cores, merged in a fixed order | the night takes 12–25 wall minutes, not 18–32 | R8 (core, day 12) |

Three rules bind every lever:
1. Every asynchronous join happens at a fixed sim tick, and blocks if the work is late. Nothing is joined "when ready".
2. The BLAS thread count is pinned per operation.
3. A lever stays only if all eight language digests, the `sim` profile and SimWorld's exact replay are unchanged by it.

- **Birth waits for none.** An exact lever leaves the body bit-identical, so one not ready by S5b joins at a night boundary after birth, once its three rules hold on a copy. It is not a change of body (A20).
- **The machine's share is the largest local factor measured:** at a load of 10–15 the eyes' render ran about 4× slower (above). While the life runs, it has the Mac's first claim after the owner's own use: other sessions' sims, renders and verification copies run at `nice -n 19`, one at a time, the room camera renders only while someone watches, and the load is logged each life day (C72).
- The fanless Air's heat will take back part of what they save.

**The render: B3, answered by the lead** (A65). A render shortcut is taken only if what the body senses stays the same within its own sensor's noise and a pixel of its native image.
- **(a) Taken:** a lighter copy of the G1's own visual meshes, for its eyes only: the coarsest that passes the rule where its body is nearest its eyes, set at W3's reopening (C69). It saves about 8–18 ms over the two grey eyes and about half as much again in the colour view: about 12–27 ms in all (an estimate).
- **(c) Refused:** the colour camera keeps its own view, 15 mm beside the left imager. Cut from the left eye's render, its colour would miss the robot's parallax by 1.1–2.2 of the colour window's cells where its hands reach, and the real G1 would be a change of body.
- **(b) Refused:** the sun's shadow stays (decision 3).

**Refused.** Each would change the body or the world, or need a second seed. The new bar does not reopen them: fast learning is counted in life hours (A62), so a wall clock made faster by changing the body buys nothing the bar counts, and risks the one seed (A64):
- **float16, bfloat16 or TF32 on any GPU.** A change of precision is a change of body. The gain would be small anyway: at batch 1 the work is limited by kernel launches. Determinism on Apple's GPU backend (MPS) is also unproven.
- **A longer physics step, a looser solver, or self-collision off.** Each changes the world.
- **GPU physics.** MuJoCo's documentation says MJX on JAX "can be 10x slower than MuJoCo" for a single scene. MuJoCo Warp is unmeasured here, and it is a different pipeline, so it would be a different world.
- **Parallel lives.** There is one seed.
- **Each lesson applied a tick late, or the critics' solve applied at a fixed later tick.** Either would let learning run beside the next tick's physics. But each changes the body, for at most about 20% more speed.
- **Any cloud or rented machine.** The owner's word: no cloud, ever (A61).

**The steering keyed to ticks** (A59). As first written, Claude's row came every wall minute (4.5).
- At 1× a wall minute is about 400 ticks; at 4× it is 1,600. So the teaching would thin as the machine got faster, and it already drifted with heat.
- Now the digest is written every 400 ticks (60 s of sim time), and each row is applied at a fixed later tick: the digest's tick + 400.
- A late row makes the world wait, which costs wall time only. If Claude fails, the fast layer runs alone for that window and the log says so.
- Claude's round trip then caps the pace. At a round trip of R seconds, the life runs at most 60/R × real time: at the local plan's fastest estimate (2.3×) a window is 26 wall seconds. The round trip is measured at P5; if it would cap the pace, each row lands two windows after its digest, fixed before birth (A64, C71).
- **The one input from off this Mac** (A61). The rows come from Claude, served on the owner's plan and never the metered API, and every row is logged by tick, so the life replays here with no network. If the owner rules it out, the fast layer runs alone (4.5).

**Off this Mac: closed** (A61). The owner's word is no cloud, ever, so the study's rented-machine cases (2.2–4.7× by estimate, never measured), their conditions, the one-hour test (C63) and the prices are dropped. They stay in `$S/pace/speed.md` and in git history (7bdc141). Two facts from them stay, for the real G1 one day:
- **The real G1 runs at exactly 1×.** Reality supplies the physics and the renders there. The core must fit into 150 ms on the robot's own computer, which is unmeasured.
- **Another machine re-pins the digests.** MuJoCo's documentation promises exact reproducibility only "within a single version, on the same architecture", and PyTorch's does not promise it across platforms or between CPU and GPU. So the robot's computer gets its own digests on its own fingerprint, the Mac's pins are never overwritten, and exact replay holds per machine segment.

**Calendar days to the firsts' life hours.** A life day's wake is one life hour. The table assumes the life runs around the clock; the owner's own use of his laptop, fixes and pauses add to it (the duty factor above).

| case | real time | wall minutes a life day | life days a calendar day | calendar days to 60 life days (First 1's bar) | to 200 life days (its long-run clause) |
|---|---|---|---|---|---|
| A. this Mac as designed, quiet and cool | 0.8–1.3× | 65–110 | 13–22 | 2.7–4.6 | 9.0–15.3 |
| B. A with the exact levers | 1.0–1.6× | 49–84 | 17–30 | 2.0–3.5 | 6.8–11.7 |
| C. B with (a), the lighter meshes for its eyes | 1.1–2.3× | 38–79 | 18–38 | 1.6–3.3 | 5.3–11.0 |

- **This is a planning reference for wall time.** It never forecasts a milestone and is never a target (A62). Case C's tick of about 65–135 ms is B's 92–147 less (a)'s 12–27, an estimate; the night is 12–25 wall minutes in B and C.
- **Claude's steering:** 60 rows a life day at the 400-tick key, 3,600 to 60 life days and 12,000 to 200, served on the owner's plan (A61).

## 10. The disclosed constants

| constant | value | kind | set by |
|---|---|---|---|
| tick; physics step | 150 ms; 2 ms (75 a tick), `implicitfast`, the elliptic friction cone, multi-point collision detection off, impratio 10 (MuJoCo 3.9's documented remedy for slip, as Menagerie's hand models ship it: 5.1, A32), auto-reset disabled, the end state checked against `mjMAXVAL` each tick | clock, world | ours |
| life day | 24,000 ticks | physiology | ours (today's) |
| the child | the stock Unitree G1 with Dex3 hands (Menagerie `g1_with_hands.xml`, unchanged): 1.32 m, 34.39 kg | world | owner (decision 7) |
| torque limits and joint ranges | the model's own (Menagerie's file, section 3.2); Unitree's differing URDF (35 N·m for the ankles and the waist's roll and pitch), MJCF (the hip roll 88) and page (the knee 90 / 120) recorded beside them | anatomy | the robot's model |
| servo gains | set at load, never in the file: Unitree's published kp and kd for the G1 and the Dex3 (unitree_sdk2's examples, unitree_rl_gym's G1 configuration), read before birth; a joint with none takes the nearest published joint's gain per N·m of limit (A39, C46). Retired: the limit at 0.25 rad of error (the Dex3's at 0.1), damping 0.04 s × kp, under which W1 measured | anatomy | the robot's (published) |
| the motors and inertial units | each motor's torque–speed envelope; its torque sensed as estimated from current, with that estimate's noise; its angle through the encoder's steps; a first-order thermal model (heating with torque², cooling to the room); each gyro's bias a random walk per axis (Woodman 2007) on the file's white noise; all from the world's seeded streams, each constant from Unitree's documents or the part's datasheet (A39, C46) | world (the sensors' and motors' physics) | the robot's parts |
| servo law | an act: target = measured angle + step; at rest the target relaxes to the measured angle, time constant 3 ticks; no gravity compensation | innate | ours |
| step sizes | waist, arms, hands, legs ±0.09 / ±0.27 rad a tick; gaze ±4° / ±11.5°, vergence ±1.7° / ±5.7°; tract −0.6, −0.2, 0, +0.2, +0.6 of each articulator's range | anatomy | ours |
| persistence margin (movement units) | log 4 in logits | physiology | ours |
| `chunk_max` | 8, a ceiling only | physiology | ours (the served value) |
| pain | a joint's estimated outside torque past that joint's own torque limit (its file's), or the free base's estimated outside force past F_pain = 3 × the child's weight from the model file (about 1,012 N); the Dex3's arrays carry no pain (A37). Retired: F_pain per touch zone on the sim skin | innate | ours |
| pain force | the observer's estimates, each the tick's largest 10 ms mean (A12's filter, the observer's own 10 ms step) | innate | ours |
| the observer | the momentum observer (Haddadin et al. 2017) on the robot's own sensors (the encoders, the torques estimated from current, the pelvis's inertial unit for the free base) and its file's inertias; every 5 physics steps (10 ms); its gain ours, settled before birth (C43); in the world's sensor code, as software the real G1 can run | innate (a software sense) | ours |
| withdrawal | a limb with any joint in pain (the observer's; a hand's joints for its arm) takes one big flexion step (0.27 rad) of its flexion joints a tick for 2 ticks (WITHDRAW_TICKS), generalized over the limb: a leg its hip pitch, knee and ankle pitch; an arm its shoulder pitch and elbow (its hand's pain included; no wrist joint); the signs measured on the G1; the waist and torso have none | innate | ours (Andrews and Fitzgerald 1994; Cornelissen et al. 2013; Holmberg and Schouenborg 1996: A31) |
| grasp reflex | a palm touch above 0.3 N (GRASP_N, the palm's tick-mean force; the prototype's value) closes six of the Dex3's seven joints one small step a tick (the thumb's rotation has none), summed at the spinal cord with the hand's own act, unless that act opens the hand (A35) | innate | ours |
| orienting | a born bias on the gaze's and the waist yaw's proposals toward a three-blob face template, the ears' lateral read and a sudden local change in the periphery; a born gate input; its gain from the amygdala (the template's constants: C39) | innate | ours |
| the visual onset cue | a grey periphery cell's luminance change since the last tick beyond the periphery's median change by a Weber fraction of 0.10; none while the gyro reads a turn above 10° a second; habituating per cell, the trace decaying at the ladder's 256-tick clock (Sokolov 1963; Johnson 1990; A43); settled from sources before birth (C49) | innate | ours |
| VOR | the fovea window counter-shifts by the torso gyro's rotation in each camera's frame, born at gain 1, its gain and a bias offset then learned by the cerebellum's flocculus (A44); its quick phase at the window's reach, a jump back of half the reach in the direction of the turn (A23) | innate | ours |
| weakness when empty | torque limits × (0.3 + 0.7h) | anatomy | ours |
| eyes | the D435's own sensors (A38): a grey stereo pair at its imagers (50 mm apart), 88.3° × 58° (horizontal × vertical), 336 × 192 px each, pitched 47.6° down as on the real G1, grey as each imager's visible response to the render; the colour camera as a third view beside the left imager (its place and axis from the datasheet, C45), 69.4° × 42.5° at about 238 × 134 px; the infrared projector off; periphery 6 × 6 averaged (56 × 32 px), the colour image in 5 × 3 cells; fovea a 64 px window (about 21°, 3 px a degree, A42) whose centre reaches ±38.1° × ±20.3°, and a colour window at the left eye's gaze; vergence 0–11.4° (to the 25 cm near point); born centred and parallel; the window holds where it is left; the sun's shadow kept; its own body drawn for the eyes from a lighter copy of its visual meshes within the render's rule, and the colour camera its own view (A65, C69) | anatomy | the robot's sensors / ours (the render's rule: A65, B3 answered) |
| the camera model | an exposure loop, Poisson–Gaussian noise (Foi et al. 2008), blur from the head's rotation in the physics over each exposure, the colour camera's rows read over its readout time, gamma; constants from the OV9282's and OV2740's published figures (C45); the world's seeded stream | world (the sensors' physics) | the robot's parts |
| the fovea's born bank | on each grey fovea's native pixels: centre-surround ON and OFF (a difference of Gaussians, centre 1 px), and oriented energy at 4 orientations × 2 scales (periods 3 and 6 px: 1.0 and 0.5 cycles a degree), pooled over 8 × 8 px cells; the colour window as cell means of red–green and blue–yellow, ON and OFF (Hubel and Wiesel 1963; Dobson and Teller 1978; A42) | innate | ours |
| touch | the Dex3 hands' 16 zones (each palm and finger link), counting contact only on the arrays' faces and saturating at their range (C44), each log(1 + F / 1 N) of the tick's mean summed normal force and its onset; the observer's outside torque per joint (÷ its limit) and the base's outside wrench (÷ the body's weight), each with its onset (A37). Retired: the sim skin's 29 other zones | anatomy | the robot's sensors / ours |
| the IMUs' noise | the model file's own declared noise (gyro 5e-4, accelerometer 1e-2), added by the world from its own seeded stream, since MuJoCo 3.9 does not apply it; each gyro's bias walking on top (A39, C46) | world | the robot's (its file; the part's datasheet) |
| the event line "a face in the fovea" | the born three-blob face template on each fovea's pixels (either eye), never the world's face test | innate | ours |
| the born face template | CONSPEC's three dark blobs in a face-shaped ellipse (A1's face front, 0.17 × 0.21 m): eye discs 0.20 W across at (±0.22 W, +0.12 H), a mouth ellipse 0.36 W × 0.10 H at (0, −0.25 H); read on luminance at widths 8, 11, 16 and 23 px (at the first build's 1.5 px a degree; C39 restates the detector on the new code); a match at r ≥ 0.5 over the ellipse with the blobs darker than the face by at least 0.10 (Weber); its readings `face_fovea` (1) and `face_periph` (3) | innate | ours (Johnson and Morton 1991; Goren 1975; Farroni et al. 2005 for the polarity; the layout recalled from a person's proportions, never set on the parent); to be settled from its sources (C39) |
| contacts with the world | the world's geoms at contact priority 2, so their friction and softness decide contacts with the G1; the G1's file untouched (A21, C26) | world | ours |
| the mat's collision box | 10 cm deep, sunk into the floor (a thin one trapped a foot sphere at 1,900 N) | world | ours |
| born codes | retinotopic: periphery 7 × 4 cells × luminance ON and OFF per grey eye and 5 × 3 cells × red–green and blue–yellow, ON and OFF, for the colour camera; fovea 8 × 8 cells × the born bank's 10 per grey eye and × 4 colour channels for the colour window (A38, A42; the first build's colour eyes had 6 opponent channels per eye); ears 40-band gammatone (the power response, 1.00 ERB, calibrated in Pa²), ERB-spaced 80–7,600 Hz, cube-root re 60 dB SPL, delay lines ±12 samples in 21 bands (a correlation times the band's loudness), the exact rigid sphere of radius 0.079 m, the ears' converter (flat to 7.6 kHz, 60 dB down from 8 kHz); random projections from the seed | anatomy | ours |
| ears' places | two sites on the head's sides, 15.8 cm apart | anatomy | owner (default, B5) |
| the tract | 12 cm, 24 sections plus a nasal branch; pitch 180–546 Hz, resting about 265 Hz; breath 400 cm³, refilled in 0.8 s; rest relax 1 tick; its calibration (section 4.9) | anatomy | ours (identity: B6) |
| face reward | the increment rule: rises of the positive part felt positive, rises of the negative part felt negative, falls never felt, clipped ±2; felt only via the born reading while the face test passes on 2 consecutive ticks; once the child's pixels carry it, while the born detector finds her face in a fovea on 2 consecutive ticks (A49) | reward | ours |
| born reading | 2 × (smile − frown), the mouth corners only; its last value for 30 ticks out of view, then neutral (A2); from the world while the scaffold lasts | innate | ours |
| the born mouth-corner reader | on the fovea's centre-surround map, inside a face the born detector found: the mouth corners' height against the mouth's centre, read into −2..+2 as the born reading's scale; its constants from its sources (Field et al. 1982) once C39 settles the detector (A49, C55); born in every case, the face term taking the world's value while the world supplies it; the scaffold's removal test: the reader's sign agreeing with the world's on at least 90% of judged faces within the lean-in distance, and a smile of +1 or more on at most 1% of ticks with her face neutral or absent, tried every 5 life days until it passes (A49, C55) | innate | ours |
| face test | the mouth point in an eye's software fovea and reached by an unblocked ray (a first hit within 2 cm of the mouth point, or on her own lips, counts as the face: RAY_SLACK_M); the face turned within 75°; its front (0.17 × 0.21 m) at least 20 fovea px at 1.5 px a degree × the cosine of the turn, so 80 px at the fovea's 3 px a degree (A42); either eye (A1); a scaffold until the pixel carrier (A49) | innate | ours |
| the parent's feelings (FEEL) | joy pulses of worth ÷ 2 (rise 2, held until seen at most 20, then 10, ease 5); displeasure 0.5 / 0.25, held 10; surprise τ 3, × 0.5 per repeat within 200; concern 1.0 / 0.6, τ 40, blocks smiles above 0.5; attention τ 5, the flash after 200 ticks unseen; mood τ 4,000 | teacher method | ours |
| the face's start rule | a smile waits for 2 neutral ticks if another is on her face or a positive face was seen under 31 ticks ago; dropped after 40 | teacher method | ours |
| the worth table | section 4.3; mastery 1 + e^(−n/10), floor 1 | teacher method | ours |
| the parent's face placement | never closer than 25 cm, arriving at least 15° off the fovea's line (A3) | teacher method | ours |
| the parent's face | the norms and sources of 4.1; ours: her corneal plane and pupils' height in her head's frame, the medial canthus 3 mm behind the corneal apex, the nasion 9 mm over the pupils and 10 mm in front of the cornea, the relief between the landmarks | world (the parent's body) | the sources / ours |
| her lids and eyes | the hinge 12° back (HINGE_BACK_DEG) and −9° tilted (HINGE_TILT_DEG), of a grid the least turn that closes a blink; LID_CLEAR 0.2 mm; BLINK_OVERLAP 0.5 mm; BLINK_INNER (0.02, 0.98); LID_SPAN_DEG (110, 60); R_COV 13.3, R_COV_LO 13.45, R_COV_IN 12.45 and R_M 13.6 mm; LATERAL_TUCK 0.20; RIM_BAND 1 mm; the lashes LASH_R 0.55 mm over LASH_SPAN (0.06, 0.80); the pupil's disc seated 0.15 mm on the iris (PUPIL_SEAT); TEMPLE_PINCH 6 mm | world | ours |
| her collision | convex pieces of her own sheet (COLLIDE_GAP 4 mm, COLLIDE_DEPTH 12 mm, COLLIDE_MIN 5° and 6 mm), her hair's cap, a sphere over each eye to the lash front | world | ours |
| her photometry | Russell, Kramer and Jones 2017's contrasts, measured under a studio lamp (STUDIO 0.35, 0.65; STUDIO_SKIN_L 65) at her face's depth (FACE_DEPTH_M 0.52 m), with the 1 mm band round the lashes (LASH_BAND_MM) and a 5 mm skin ring (ANNULUS_MM); the colours so calibrated: the iris RGB (0.3137, 0.1881, 0.1045), the brows 37% hair over skin, the lips 32%, the sclera and the lashes; her skin's shade of the indirect light baked at share 0.4 | world | the source / ours (the calibration) |
| the face instruments | `sim_face_measure`'s LEAK_DEPTH 0.15 mm; `sim_face_template`'s detection: a match centred within a third of her face's width of her face's centre, its width within √2 of hers | instrument | ours |
| the parent's force caps | one hand 100 N sustained, 150 N for up to 2 s; both hands 160 N, 200 N for up to 2 s; every hold a capped spring | teacher method | ours (checked against sources in W2) |
| the guide's cap | min(1.5 × the limb's own push at that pose, 100 N); at most 8 ticks at up to 0.3 m/s; stops after 2 ticks at the cap (A10) | teacher method | ours |
| the parent's soft contact and yield | solref 0.05 on its collision shapes; a contact over the act's cap for 2 steps stops that segment and backs it off 2 cm a tick (A4) | teacher method | ours |
| the parent's own pain | 150 N (a 10 ms mean) on any segment | teacher method | ours |
| the prop's easing and catch | the pull-to-sit capped at the caps; the prop within 30° of vertical, easing 100, 70, 40, 20% then hovering 5 cm off, one step down per 10 ticks within 20° of vertical; the catch 2 ticks after the trunk passes 35° or the head drops faster than 0.5 m/s (A9); the 300 ms reaction is a person's and never changes (C6) | teacher method | ours |
| a peekaboo answer | an act begun within 10 ticks of the reveal by an effector that rested the 5 ticks before (A2) | teacher method | ours |
| the parent's paths | a 5 cm floor grid; clearances: furniture 0.15 m, the child 0.25 m (outside its leg sweep), toys 0.08 m; walking 0.8 m/s, shuffling on the knees 0.25 m/s (A6) | teacher method | ours |
| the parent's timings, judgments and conduct | sections 4.3–4.10, in `body/sim/lang/consts.py` and the parent's constants file | teacher method | ours |
| her gaze in asks and probes | her head and eyes on the child, no point, no turn until the ask is judged (M1's head-turn probe excepted, its stimulus her silent turn); to the object only on a label's, a show's or a confirm's naming word (Golinkoff et al. 1987; A51) | teacher method | ours |
| her reading of where it looks | the nameable object nearest its head camera's line within ±38° × ±20°, held 3 ticks, or the object its hand holds or closes on; the line read with a person's error from her seeded stream (A40, C58) | teacher method | ours |
| her imperfection and her copying | a share of its acts missed, her reply's latency jittered, spells of distraction, and her mirrored copies of its arm and hand movements within 1–2 s, each at a rate from human data (Tronick and Gianino 1986; Bahrick and Watson 1985; Ray and Heyes 2011), from her seeded stream, fixed before birth (A52, C57) | teacher method | ours |
| the parent's ear | the cochlea, a shift of 0–4 bands, c1–c12, Itakura DTW; a template bank of many voices; accept when nearest of the expected words and d − d(babble bank) < −m, m = −0.35, re-set per context size; exact when also nearest of all its words; the bank, the templates and the expected sets fixed before birth, the child's own productions never added (A27) | teacher method | ours |
| the tidy | a toy out of the child's reach and untouched for 2,000 ticks is put back (A5) | teacher method | ours |
| the amygdala | on for the sim, absent for the language body (switch `amyg`); input: the stream C/√d, the anatomy's event lines, a level; heads: one per reward source and sign (sim: face ±, pain −, charge ±) | physiology, anatomy | ours |
| its law | least squares with forgetting, the backward target identity; ridge 0.3 × τ_a × running variance, the level free; τ_a 4,096 ticks; solved every 8 ticks; horizon 0.9375 a tick, not cut off | physiology | ours (the critics') |
| its reliability | correlation with the realized target, finalized 64 ticks later, moments over 36,000 ticks, clipped 0–1, zero until 64 pairs | physiology | ours (the face organ's) |
| its tag and uses | min(2, A⁺ + A⁻ + received); reaching back 64 ticks at 0.9375; the store's gate surprise × (1 + tag) over its running 0.9 quantile, strength × (1 + tag) with later boosts, at most × 3; the night's entry × (1 + T_e) over a saved running mean of 64 episodes; T_e ≥ 1 dreamt first, at most half the night; the window ending at the peak tag; `act_pred` at night × clip(1 + G, 0, 1); orienting gain clip(1 + N, −0.5, 2) | physiology | ours |
| `amyg_pav` | 1 logit per reward unit of N, clipped ±2; off at birth | innate | ours |
| `tag_trace` (fix #6) | the received tag reaching back onto the language body's felt entry; defect 7 fixed with it; off | physiology (switch) | ours |
| bedtime feed | a feed first if h < 0.6 when goodnight begins; every hold released at tick 23,700 (A17) | teacher method | ours |
| pain reward | −1 per tick when any joint's estimated outside torque passes its limit, or the base's outside force passes F_pain (A37) | reward | ours |
| charge reward | 4 × [D(h_t) − D(h_t+1)], D(h) = (1 − h)² | reward | ours |
| reward order | face, pain, charge | reward | ours |
| charge drain; bottle | 4e-5 + 2e-3 × Σ τ² / Σ τ²_max a tick; +0.01 a tick while the bottle touches a palm; at night the 4e-5 stops (A46) | world | owner (default, B10); the night's, ours |
| the night | live and dark for NIGHT_TICKS 24,000 ticks (A46); the eyes and ears off; twitches in REM's phases (about half the night: Roffwarg, Muzio and Dement 1966) at about 10 a minute (Sokoloff et al. 2020; read exactly before birth, C52), one joint and one 0.09 rad step each, from a born generator seeded by the body's seed; the tract and the gaze none | physiology, world | ours |
| fatigue | per effector; the tract 0.12 a sounding tick; limbs 0.12 × Σ τ² / Σ τ²_max per tick of motion; gaze 0.03 a step | physiology | ours |
| the gates' tonic drive | per act of any gate: `gate_tonic` 0.25 + `gate_tonic_rate` 4.66 × the felt reward's running mean at the 256-tick clock (`gate_tonic_clock` 4); 4.66 = Σ_{k<12} 0.8^k, the gate's eligibility window (Niv et al. 2007; A41) | physiology | ours (the 0.25: the core's and the served language body's) |
| the performance error | the tract's gate only (`gate_int` 0.5, `gate_int_form` "error"; 0 for every other gate): per articulator, the belief in the chosen setting minus that setting's running mean (50 means, 0.1 per act, `gate_habit` 0.9), averaged over the ten (Gadagkar et al. 2016; A41) | physiology | ours (the 0.5: the language body's interest weight) |
| `gate_vigor` | 0 for the sim (the core's default 1.0; the served language body's 0) (A41) | physiology | ours |
| the spinal pattern generator | a half-centre oscillator per limb, the legs in antiphase; in its flexion half a step along the limb's flexion joints, in its extension half the opposite; amplitude the limb's gate's p_act × 0.09 rad; summed at the cord with the limb's own act; its period, amplitude and the arms' coupling from Thelen 1979 and Kuniyoshi and Sangawa 2006 with C38, before birth (A48, C54) | innate | ours |
| the born cry | on a pain tick or a charge below 0.2: the tract's cry posture (lungs pushing, glottis pressed, pitch raised, jaw open) added to its targets, the tract's own act overriding it; logged as reflex (Jürgens 2002; A47, C53) | innate | ours |
| motor timing lessons | `act_inv` batched every 8 ticks; the MOTOR constants (`act_inv_lr` 1e-3, `act_inv_tau` 8192) | physiology | ours (R6) |
| critics' solves | every 256 ticks (sim), 64 (language) | physiology | ours |
| ventral critic's bands | fast ladder bands 0–2 (sim) | physiology | ours |
| forecast heads | one per channel; vector heads by squared error to the next born code, each error scaled by its own running mean | physiology | ours |
| event end | today's settle law (`offset_fast` 4, `offset_slow` 64, `offset_settle` 0.5) on the summed forecast error; for the sim it also latches working memory, in place of utterance ends (R7f, A45) | physiology | ours (today's) |
| recall into action | the frame's key: the stream and the trunk's yaw integrated from the torso gyro; its value: the next frame and every effector's efference copy; into each effector's proposal through a map born at zero, learned by `act_pred`'s lesson (Lengyel and Dayan 2007; McNaughton et al. 2006; A45) | physiology | ours |
| the cerebellum | every 5 physics steps through the world's sub-tick hook; a born sparse expansion of 4,096 granule units, 4 inputs each, about 10% active; per-joint Purkinje readouts (the arms, the legs and the waist: 29 joints) adding torque inside the limit, and the flocculus's VOR gain and offset; taught by the servo's corrective torque (every 10 ms) and retinal slip (once a tick); LMS normalized by the expansion's activity; settled from sources before birth (Marr 1969; Albus 1971; Fujita 1982; Kawato and Gomi 1992; Ito 1982; A44, C50) | physiology, innate (the expansion) | ours |
| store | capacity set before birth to hold 19 life days of the measured writes | physiology | ours |
| episode cap | 24,000 ticks | physiology | ours |
| switches at birth | fixes #1, #4, #5, #8 on; `amyg` on; `chunk_gate` 1 per effector; the cerebellum, the pattern generator, the cry, the visual onset cue, recall into action and the twitches on | physiology | ours (off for language) |
| voices | parent: compact Samantha, rate 0.25, pitch 1.15, and the registers (the new word's at rate 0.15); 8 variants of every line, pitch within ±8% and a vocal-tract-length warp within 0.9–1.1, drawn per utterance from her seeded stream (Jaitly and Hinton 2013; A50); the focus word's prosody: pitch +30% on the line's, rate 0.7 × the line's, holding its punctuation (Fernald and Mazzie 1991; Albin and Echols 1996); the child: its tract | world, anatomy | owner (default) / ours |
| light | the sun across the window from morning to dusk; the lamp at winding down; dark through the live night | world | owner (decision 3) |
| the room's indirect light | each light's ambient a third of its diffuse (ROOM_INDIRECT); no headlight at the child's eyes | world | ours |
| the room, toys, charger, names | sections 5 and 4.8; textures, containers, a cover, at least 3 examples of each tested noun, and one new object every 3 life days from an inventory fixed before birth (A53) | world | owner (defaults; the richer room decided by the lead for him) |
| the G1's collision shapes | its own, as shipped; replaced at load by a convex decomposition of its own meshes (Wei et al. 2022) only if W4 finds pain from hulls meeting, decided before birth (A54) | world | the robot's model / ours |

Every language-body constant keeps its value and place.

## 11. The build plan, day by day to birth

Three sessions in parallel. Each writes tests at `nice -n 19`, small and short, and never touches ports 8020, 8021 or 9333. One verification copy at a time across all sessions, deleted when its run ends (A18).
- **Core** works in the `sim-core` worktree.
- **World** works in `body/sim/`, `tools/sim_*` and the `/sim` page.
- **Parent** works in `body/sim/lang/`, `body/sim/voice/`, `body/sim/ears.py`, `body/sim/parent_*.py` and `body/sim/tract.py`.

Re-estimated for the lead's decisions (A57): the new steps are R6c; the additions to R6h, R7 and R8; W3 and W1 reopened; W4's hull pain; W5's live night and W5b's richer room; P1b's voice variants; P3's closed gaze leak, her imperfection and her copying; P4's tests; and the pixel smile reader. The decisions of 2026-09-25 add the levers' steps (R10, W7 and P7, and R8's parallel night: A64), the mesh copy for the eyes in W3 (A65), and the firsts' registration in P4 and S5b (A63). None moves birth, and birth waits for no lever. Rows marked "built" are done and verified (section 0).

| day | core (session 1) | world and the G1 (session 2) | parent, voice and ears (session 3) |
|---|---|---|---|
| 1 | R6h: movement units, and the born unit lengths under the continuation draw (C38); the gates' drives set and disclosed (3.5, A41) | W1 (built): SimWorld from `g1scene.py`: the servo law set at load; weakness; withdrawal and grasp on the Dex3; touch and pain per link with the 10 ms filter; the IMUs' declared noise; `FatalError` caught and MuJoCo's auto-reset disabled (A18); save and restore with every random stream; the exact replay test | P1 (built): the parent's voice: the Swift synth server, the cache, SSML emphasis and the word rate measured, the determinism test |
| 2 | R6h: fatigue per effector, forward error into the gates, `act_inv` batched; the spinal pattern generator and the born cry, summed below the gate (A47, A48) | W1 (built): the sink rates under the resting law; real frictions by the world's contact priority and a friction model that does not creep. W2: capped springs for every hold on the G1 (the prototype's welds on its torso, pelvis and elbows removed); the yield rule | P2 (built): the ears (`ears.py`): the spatializer at the G1's head radius, two cochleas, the delay lines, the born lateral read; the tract's sound through them |
| 3 | R6h: born encoders at the new channel sizes; the declared consequence sense; the gaze effector; the orienting hooks (face, sound, sudden change: A43) and the VOR hook | W2: the parent's acts on the G1 under the caps: the kneel outside the leg sweep, attend, show, hand over, guide, the brief turn, the pull-to-sit with its help, the prop near upright, the catch; the motor intents | P3: the fast layer: templates, the line check, intents, variation sets; the feelings and graded face (`parent_feel.py`) in the world; the gaze leak closed, her eyes on the child in asks and probes (A51) |
| 4 | R6h's last quarter; R6c: the cerebellum, the sub-tick hook in the World interface, its tests (7.5, A44) | W3 (built): the eyes, the gaze, the VOR, the retina's code, the face test's rays, the template on the fovea's pixels, the eye check under three lights. W3 reopened (A38): the grey pair and the colour camera as three views; the camera model; the dock beside a hand; the bottle for the Dex3; B1 and B2 applied | P3: the behaviour system's L1–L2: contingency with her declared imperfection (A52), joint attention from its trunk and hands (A40), her copying of its movements, the scaffolding ladders; the ledger as she reads it |
| 5 | R6c's last quarter; R7a–b: the 13 event lines; surprise-gated writes, the event end for frames, the tick's record | W3 reopened (A42, A65): the fovea's born bank at 3 px a degree; the lighter copy of its own meshes for the eyes, by the render's rule (C69); the eye check again at the core's d, on the copy (C3, C48); the visual onset cue measured (C49) | P1b: the eight variants of every line (A50), measured through the parent's ear and for the new word's peak by frame (C56) |
| 6 | R7c: fixes #1 and #6 (`tag_trace`), error scaling, pace on the partner channel | W1 reopened (A37, A39): the observer and the hands' arrays; pain from the joints; the withdrawal's trigger; Unitree's gains; the motor, gyro and heat models; the sink rates again (C43, C44, C46) | P3v: the tract as effector 0 in the world; the parent's ear (`parent_ear.py`): the babble bank recorded before birth, templates from P1, m re-set for the context sizes; the cry heard as distress (A13) |
| 7 | R7d–e: the amygdala, its tests, the orienting gain on the three cues, `amyg_pav` off | W4: the born body's own motor timing on the G1, with the pattern generator and the cry (per effector: its gate's draw at the born p_act, the continuation draw, the margin, `chunk_max`; a replica here, repeated on the real core at every learning rate 0 at S5a, each run discarded after), under the final law: unit lengths, rolls by direction, travel, time off the mat, pain from the joints with the share from hulls meeting (A54, C59), thumps, the leaning sit's fall time, bottle visits, hits on the parent; every hold and guide against the joints' pain law (C43). Written down as chance. | P4: the day plan, routines, stages, leaving and returning; the never-taught pairs held out by the line check; the tests of understanding by milestone listed with their items and chances (A55, C36); the firsts' claims, bars, items and instruments registered (A62, A63, C70) |
| 8 | R7f: recall into action; the working-memory latch on event ends (7.6, A45) | W4 continued: if the hulls carry the pain, the convex-decomposed collision copy loaded world-side and W4's pain measured again (A54) | P5: the digest (its trunk and hands, never its fovea: A40), `ops/sim_parent_brief.txt`, the steering check; the digest and the row keyed to every 400 ticks, and Claude's round trip measured (A59), the row's lag set by it (C71); P6: `tools/sim_parent_rates.py` with W4's babbler, her declared rates checked (C24, C57, C58) |
| 9 | R7f's last quarter; R8: night batches from stored codes; the tape and episodes | W5: sounds from physical events; the room's echo (B9); the ears in the frame; the tract's sound from the head's front; the day's light and the lamp; the live, dark night's world (the dim, the eyes and ears off, the parent asleep: A46); every channel from the scene | W3r: the born mouth-corner reader on the fovea's map, with sim-face's detector once C39 settles; the scaffold's removal test fixed (A49, C55) |
| 10 | R8: every effector's acts replayed; the entries, the tagged first, `act_pred`'s weight from the replayed dopamine | W5b: the richer room: textures, containers, the cover, at least 3 examples of each tested noun, the inventory and its calendar (A53, C60); the eye check again under W5's lights | W5b with session 2: the new objects' sounds, holds by a Dex3 hand and the cover's lift (C60) |
| 11 | R8: `act_inv` and the forward half replayed; REM on frames; the entries' mean saved | W6: the `/sim` page on port 8030: the room camera, the three views with their fovea windows, the parent's face, charge, the ten gates, a two-voice transcript, instruments | (S5a) |
| 12 | R8: the live, dark night: the world's night, the twitches and their lessons (A46); its physics and the replay on separate cores, merged in a fixed order (A64) | S5a (sessions 2 and 3): the SimAnatomy (9 channels, 10 effectors, 3 reward sources) on the core as it stands; a plumbing run at every learning rate 0 | (S5a) |
| 13 | R8's last quarter | W7 (A64): the ears, the tract, touch and the observer computed while the GPU draws; the world's per-step Python compiled, bit-equal to what it replaces; each kept only under the levers' three rules (section 9) | the parent's rates with the babbler: smiles per ask, calls answered, asks met, her declared contingency and the voice-free share, written down as chance. There is no target rate. Before birth the method changes only for a stated reason (such as an ask no body could meet from where it lies), never to reach a number. P7 (A64): her ear off the tick, joined at her reply tick |
| 14 | the full guard with `--roundtrip`; the `sim` profile pinned | the plumbing re-run on R7 and R8; heat checks | |
| 15 | R10 (A64): the critics' rank-one updates on a second performance core, joined before each solve and each save; each lever kept only if the eight digests, the `sim` profile and SimWorld's replay are unchanged | S5a again on the finished core: every channel, effector and reward source through a live night | |
| 16 | R10's last quarter, then S5b with session 2, with every lever that holds its rules | S5b: two days and nights at every learning rate 0 on the whole core: the mean tick with the three views and the levers (C12, C72), the night's length and the store's write rate written down; the store's cap set; the firsts' tests run on the born state for their chance (A63, C70); the birth checklist | |
| 17 | — | **S6: birth (seed 1)**: day 1, its first night, day 2, the save round trip; the first hour watched at 1× | |

**Where the build stands** (section 0's build status):
- **Built and verified:** W1 (day 1's world; day 2's sink rates and friction model, impratio 10), W3 (the eyes, the gaze, the VOR with its quick phase, the retina's code, the face test and the born template), P1 and P2.
- **Left in their rows:** the surfaces' own frictions (C26); the dock beside a hand, the bottle for the Dex3, and B1 and B2 applied (W3); the born template's constants from its sources (C39, before birth).
- **Reopened or added by the lead's decisions:** W3 (three views, the camera model, the born bank; the eye check again), W1 (the observer, the arrays, the joints' pain, Unitree's gains, the motor and gyro models), W4 (the hull share and, if needed, a decomposed collision copy), W5 (the live night) and W5b (the richer room); P1b (the variants), P3 (the leak, her imperfection, her copying, her reading of its trunk and hands), P4 (the tests by milestone) and W3r (the pixel smile reader); R6c, and additions to R6h, R7 and R8.
- **Added to later rows by the build:**
  - W2 removes the room's exclusion of her hands' contacts with the G1's torso and pelvis, with the welds, and builds the face channel's 2-tick gate and 30-tick hold (with S5a);
  - W4 counts fists closed on its own fingers (C41) and the withdrawal's pressing on the born loop (C22);
  - W5 dims the night's lights rather than switching them off (5.4), weighs the room's echo against the ears' cost per source (B9), and writes down a sound with no low bands (C42); the eye check runs again under its lights, at the core's d (A33).
- **Merging:** each branch's trial merge into main was clean at its verified commit. P1–P2's `ears.py` and `tract.py` replace main's prototypes, and main's prototype ear moved to `parent_ear_cochlea.py`; W3's `eyes.py` replaces `g1eyes.py`, which stays for `tools/sim_look_g1.py`.

- **Against this morning's plan** (birth about day 11 for the custom child, with the G1 and the tract each adding days unmeasured): the G1's room, senses, eyes and the parent's acts on it, the tract and the parent's ear now exist as prototypes. What remains is the servo law on the G1, the caps and the new acts, and the core's additions (the gaze and the declared consequence sense, the born units' lengths, the amygdala's extra half day in R8). R6 fix 3 is committed (e48b284); its verifier's finding is answered by R6 fix 4 (440bad3), in verification.
- **Against the G1 amendment's plan** (birth about day 13, range 11–18): the lead's decisions add about 3.75 days of core work (R6c 1; R6h 0.75; R7 1; R8 1), which is the critical path, and about 4.5 days each to the world and the parent, which fit beside it. Birth moves to about day 17 (the roadmap estimated 15–16; it counted neither the colour camera's view nor the pixel reader's day).
- **Range 15–22.** The widest unknowns are R8 with its live night, whether W4 finds pain from hulls meeting (a decomposed copy adds about a day), the reopened eye check, the observer's pain rate under the caps, and whether the pull-to-sit and the catch work under the caps. The tick no longer holds birth: B3 is answered and no 150 ms bar binds a lockstep life (A65).
- **Two sessions:** the core in session 1 (about 13.25 days), then its help with the rest; the world and the parent in session 2 (about 22 days); the joint days after. Birth about day 24.
- **One session:** about day 41.

**The birth checklist:**
- The eight language digests equal their pins, and the `sim` profile is pinned with the amygdala on.
- The G1's model file is byte-identical to its commit.
- The sim's replay is exact, and a save and reload continues it exactly (the world, the parent, the tract and every random stream).
- The babble baseline, the parent's rates and the parent's ear's chance acceptances are written down, with the babbler's smiles per ask as chance.
- The plumbing run shows every channel arriving, every effector acting, rewards summed in order, a night with the tagged first, and a save round trip.
- The mean tick, with the three views, the eyes' lighter mesh copy, every organ of the lead's decisions and every lever that holds its rules, is measured over two heat-soaked hours and written down (C12, C72). No 150 ms bar binds a lockstep life, since B3 is answered (A65). MuJoCo's auto-reset is disabled and its warning counters are checked every tick (A18).
- The store's capacity holds 19 life days of the measured writes.
- At least 8 GB of disk is free (the peak of about 4.5 GB, the rule's floor of about 2.7 GB, and a margin; section 9), and the disk rule is live.
- `act_inv` starts untrained, and no weight of the body has learned from the babbler.
- Seed 1 is born from its freshly built state, never from a plumbing or timing run's state: at every learning rate 0 the store's writes, the critics' and the amygdala's least-squares evidence and the running means still accumulate.
- The born movement units' lengths are written down (C38), with the spinal pattern generator on, its constants read from their sources (C54), and the owner has answered B1 (the cup's scale). B3 was answered for him by the lead (A65), as were B18, B4 and B14 (A37, A38, A40); B21 by his own word (A61).
- The sim is the robot, checked sensor by sensor (A36): the microphones' and the speaker's places read from Unitree's documents (B5); Unitree's servo gains, the motors', gyros' and heat's constants (C46), the Dex3-1 arrays' places and range (C44) and the D435's sensors' figures (C45) read from their sources.
- The observer's error and the pain rate from the joints are written down on the born loop, with the share from hulls meeting, and every hold and guide is checked against the joints' pain law (C43, C59); if a decomposed collision copy was needed, it is loaded world-side and the G1's file is still byte-identical (A54).
- The cerebellum's tests pass and its constants are read from their sources (C50); the visual onset cue's and the born cry's constants likewise (C49, C53).
- The born face detector's constants are settled (C39) and the born mouth-corner reader is built; if the detector does not work at birth, the world-truth smile is a scaffold whose removal test is fixed (A49, C55).
- The tests of understanding by milestone (section 12) are listed with their items and chances before C36 (A55).
- The parent's imperfection and copying rates and her reading error are read from their sources (C57, C58); her eyes stay on the child in asks and probes (A51); her voice has its eight variants (A50).
- The live night's length, twitch rate and wall cost are measured at S5b (C52).
- The born face template's constants are set from its sources and measured on her face, upright and turned, under each light (C39).
- The eye check passes by A33's readout at the core's d, under W5's lights, on the grey pair, the colour camera and the born bank (C3, C48).
- The parent's force caps are in its constants file and checked against their sources; no act drives the G1 over them.
- The babble bank for the parent's ear is recorded and fixed; the never-taught pairs are listed and held out.
- The firsts are registered (A63, C70): each claim's wording, bar, rulers, items and chance levels frozen with a digest; each registered test run on the born state at every learning rate 0, and any it passes at p < 0.01 removed from its claim; the intervention log and A62's instruments (life hours, exposures to criterion by kind, the overnight gain) live.
- The eyes' lighter mesh copy passes the render's rule (C69), and the G1's file is still byte-identical.
- The human-pace ledger is optional (A66). Only if it is to be reported at all, it is frozen before birth (A58):
  - its recalled rows are read at their sources (C62);
  - its rows, sources, criteria and conversion are committed with a digest;
  - nothing in the body, the parent, the brief, the tests or the build reads it.
  A ledger not frozen before birth is never reported, and birth does not wait for it.
- The machine is this Mac: no cloud, ever (A61; B21 answered). The steering is keyed to ticks (A59), with its lag set (C71).

After birth, it lives days with a report each day, judged by sitting with it. Defects are fixed on copies and applied at boundaries, never tuned.

## 12. Milestones a viewer can watch, the tests of understanding, and the firsts

Each milestone is judged by sitting with the child on `/sim` and by its own rulers, never by a baseline run. A19 gives each ruler, its chance level and when it counts as reached. These are estimates: no life has been run. Wall hours assume 55–90 minutes per life day; at the 65–110 minutes the lead's decisions bring (section 9), each is about a fifth longer, and at the local plan's 38–79 minutes (section 9's case C) about a fifth shorter.

A life day's wake is one life hour, so the life-days column is also the body's sample: the experience it needs, in which fast learning is counted (A62). The last column names the first each milestone serves (the firsts, below). An infant's waking hours for each row are in the human-pace ledger, an optional report (A66, below).

| | what the viewer sees | its own rulers | life days | wall hours | the first it serves |
|---|---|---|---|---|---|
| **M0: it lives** | a day, a night and a day; the fovea windows, gates and the parent's face moving | its forecasts of its own next body sense and periphery beat "nothing changes" | birth day | the first | — |
| **M1: it orients** | its fovea, then its trunk, turn to the parent's voice and name, and it holds the smiling face in view | turns within 20 ticks above the born rate; the share of ticks with the parent's face in the fovea | 1–3 | about 1–5 | First 1; First 2's head-turn and occluded-ball tests |
| **M2: it reaches** | its hand touches a toy held over its chest | touches above the babble rate | first 5–25; reliable 20–60 | 5–38; 18–90 | First 1; First 2's look to her at a new toy |
| **M3: it grasps** | it reaches, then holds one of the eight holdable toys (the reflex holds from hour 0 when a toy meets the palm) | reach then hold for at least 3 ticks, above chance | 15–60 | 14–90 | First 1; First 2's cover |
| **M4: it rolls on purpose** | it rolls toward the parent or a toy | rolls by direction against the babbler's share under the same parent placement (A19); the trunk turning first | 15–60 | 14–90 | First 1's full claim |
| **M5: it sits with help** | it helps the pull-to-sit with its own flexion, then, held near upright, keeps its trunk up and looks at the face (her face low and ahead, where its eyes reach; in the leaning sit only by raising its trunk: A22) | its share of the rise; the trunk-up share and face looks while held; less hold needed over the days | 20–80 (open) | 18–120 | — |
| **M6: it names** | it names what is in its fovea or hand (by the scaffold's output first) | the ledger's "says"; right names above 1 in 4 | first right names 15–40; 1 in 4 at 30–80 (open) | 14–60; 28–120 | — (compositional words were done before; its tests stay) |
| **M6t: its first word from its own tract** | a word the parent's ear accepts, from the tract alone | M6's ruler on the tract's sound; the precursor: its first accepted echo | not forecast | — | — (done before, by Elija) |
| **M7: it sits alone** | the leaning sit, hands on its knees, without the parent's hands | seated time without her hands | not forecast | — | — |

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
| M6 | (a) a known word finds its toy seen in a new place or from a new angle (upside down in her hand, or in a place within its view it has never seen the toy: A28); (b) known words in a combination never heard: "where is the blue ball?" (a look), and "push the blue ball" (an act, once "push" is understood; the owner's example, "push the red ball", in the twins' colours), where the target is the twin and "blue" and "ball" were heard only in other pairings, each colour word on at least 2 kinds (the colour twins, B2; the held-out pairs in A28; the line check holds them out until the test; A55); (c) a word used to get something: it names a toy that is out of reach, and the parent has not named it in the last 40 ticks | (a) per word, its own rate with the object absent; (b) its share of looks to the named twin when she asks with the noun alone ("where is the ball?") at matched moments (A28); (c) its own rate of that name at matched moments with the toy in view and not asked for |
| M6t | a word from its own tract used in a never-paired place, or used to get the toy | its own rate with the referent absent, plus the babbler's 2% |
| M7 | it sits alone off the mat, on the oak floor, where it never sat | none: an event, in a new place |

**The tests of understanding by milestone** (the roadmap's, added by the lead's decisions: A55). They show understanding the way infants show it, by where they look and what they do unpaid, and they are fixed before C36 (P4): the items, the chance measures and the pooling. Each is judged one-sided at p < 0.01 by A19's rule, on fresh items (a novel item is a probe only on its first 3 presentations, A28), and a claim about words is made only with the word scaffold silenced (A29). No claim rests on the parent's ledger and its 10-ask window at p < 0.05 (4.8). The instruments read the fovea; the parent never does (A40). A probe waits until its object is within the fovea's reach for the child's posture (A22), and her eyes stay on the child through it (A51), except in M1's head-turn probe, whose stimulus is her silent turn to the toy.

| M | the test | chance | source |
|---|---|---|---|
| M0 | the touch forecast's error at its own hand's touch against her hand's, matched by the touched limb and the force (the hands' arrays, or the observer's group) | a ratio of 1 | Blakemore, Wolpert and Frith 1998 |
| M0 | the ear forecast's error at the drum's boom with the strike in the fovea against out of view, and at her speech onsets with her face in the fovea against not | equal errors on matched clips, by permutation | Spelke 1976; Kuhl and Meltzoff 1982 |
| M1 | after a toy's event breaks the eye forecast (beyond the body's own 0.99 quantile of eye error), the toy's share of fovea ticks and touches over the next 200 ticks | that share after an ordinary event at matched visibility | Stahl and Feigenson 2015 |
| M1 | after her silent head turn to a toy (no word, no point), the fovea on that toy within 20 ticks | matched windows with no turn | Scaife and Bruner 1975 |
| M1–M2 | the ball rolled behind the low table: the fovea at its far edge first; the eye error when it reappears | landings there in matched windows with no roll; an unheralded toy appearing at the same place | Johnson, Amso and Slemmer 2003; Baillargeon 1987 |
| M2 | a look to her face within 20 ticks of a never-seen toy's appearance | matched moments | Walden and Ogan 1988 |
| M3 | the arm's act rate while it holds the rattle, which sounds with speed; the ear forecast's error at rattle sounds its own arm made against her shakes at matched loudness | the rattle's own first holds (the block differs in mass and grasp, so it is not the chance); equal errors | Rovee and Rovee 1969 |
| M3 | the cover lifted within 40 ticks of a toy hidden under it | matched moments with nothing hidden | Piaget 1954 |
| M4–M5 | the fovea on the goal toy before her reaching hand arrives (her tidying, her hand-overs) | matched windows with no reach | Falck-Ytter, Gredebäck and von Hofsten 2006 |
| M6 | "where is the X?" with a never-seen exemplar of X beside a known non-X; at least 3 examples of each tested noun (5.2) | its landings on the new exemplar when she names the other object | Quinn, Eimas and Rosenkrantz 1993 |
| M6 | the colour twins as targets, each colour word heard on at least 2 kinds (M6(b) above); combinations of colour and noun come after the first year in infants, so a pass is not expected in year 1 | the noun-alone rate on that twin | Wagner, Dobkins and Barner 2013; Fernald, Thorpe and Marchman 2010 |
| M6 | it says "more" at a charge below 0.35, against above 0.7 | its rate of "more" at a charge above 0.7 | ours (the understanding audit's): a word used for its own need |

### The firsts: what no architecture has shown, and the bar for each (A61–A63)

The owner's word (2026-09-25): "for it to solve robotics that no archecture had done before". A first here is a claim no published architecture has shown, checked against the state of the art in September 2026 (the firsts research and its skeptic's review, `$S/firsts/`). It is registered before birth with its bar, judged by A19's rule, and reported whether it passes or fails (A63). No life has run: every hour below is a bar, not a result.

**The robotics first is the setting, not a task.** Every skill below has been learned faster by some method, from a signal built for it. None has shown them together: in one life with no resets, from sparse feedback like a person's, with no reward built for any task, still learning afterwards, in a real robot's whole body. It becomes a robotics result on the real G1 (the research's F9; section 16).

**The state of the art, September 2026** (the closest published results; context, never a run and never a target: A62):

| line of work | the closest published results | what they learned from |
|---|---|---|
| a real robot, from scratch | an A1 rolled, stood and walked within 1 h, and a UR5 picked and placed from pixels in about 8 h (DayDreamer: Wu et al. 2022); an A1 walked in 20 min (Smith, Kostrikov and Levine 2022) | a task reward per skill |
| real manipulation | 25–50 min a policy (SERL: Luo et al. 2024); 1–2.5 h a task (HIL-SERL: Luo et al. 2025) | demonstrations, a success classifier as the reward, human corrections |
| many simulations at once | ANYmal walked after 20 wall minutes with 4,096 simulated robots (Rudin et al. 2022), the G1 after 15 on one RTX 4090 (Seo et al. 2025); the Rubik's cube hand used 13,000 simulated years (OpenAI 2019); grasping from pixels took 580,000 attempts over 800 robot hours (QT-Opt: Kalashnikov et al. 2018) | task rewards, parallel copies |
| world models | one configuration across more than 150 tasks (DreamerV3: Hafner et al. 2023); over a million hours of video and 62 h of robot video (V-JEPA 2, 2025) | task rewards; passive video |
| foundation models | about 10,000 h of teleoperation (π0: Black et al. 2024); 780,000 synthetic trajectories, about 6,500 h (GR00T N1, 2025); improvement from deployment, task by task, with human corrections (π*0.6, 2025) | demonstrations |
| developmental robotics | an iCub from babbling to skilled reaching in under 3 h, through staged maps (Law et al. 2014); MIMo, the infant simulator, rolling after 10⁶ steps from a dense orientation reward without vision, one agent per skill (Philipp et al. 2026); staged grasping and imitation with a caregiver's help (Ugur et al. 2015); a simulated infant with a caregiver and infant tests set up (SEDRo 2020, a proposal; Doyle et al. 2023) | engineered maps, rewards built per skill |
| a person or a face as the reward | a face in grid worlds (Broekens 2007), or through a reward model trained on recorded faces (EMPATHIC: Cui et al. 2021); a person's preferences taught simulated motor skills (Christiano et al. 2017); joint attention, gaze following and social referencing, each trained as the skill tested (Nagai et al. 2003; Triesch et al. 2006; Boucenna et al. 2014) | one skill each |
| understanding | surprise at impossible events learned from passive video, with no body (PLATO: Piloto et al. 2022; Garrido et al. 2025); embodied agents mostly failed held-out permanence tests (Animal-AI: Crosby et al. 2020) | passive video; task rewards |
| lifelong learning | deep networks lose plasticity over long streams (Dohare et al. 2024); several tasks without resets on a real robot (Gupta et al. 2021); policies that keep learning on the robot (Smith et al. 2022); deployed humanoid policies stay static (a 2026 survey: Nguyen et al.) | task rewards |

**First 1: one life, many skills, sparse social feedback, the stock G1's whole body** (the thesis).
- **The claim,** registered before birth: "The stock G1 model, from randomly initialised learned weights and a disclosed innate set (section 10), in one life of one stream with no resets and no parallel copies, learned to orient, reach and grasp [the full claim: and to roll on purpose]. It learned from a scripted parent's sparse smiles for its own completed acts, felt only while it looked at her, with pain and a charge need as its only other rewards, and it kept learning for 200 life hours without losing those skills."
  - Printed beside it, always: the intervention log (her guides, holds, turns and placements, counted: A63), and her curriculum steered every 400 ticks by a language model that never sets a feeling, a judgment, a smile, a help level or a force (A14, A61).
- **A full claim and a lesser one,** both registered: M1–M4, and M1–M3, since rolls under babble are not yet measured on the G1 (risk 3).
- **Two levels.** Level 1 holds on the world-truth smile, the disclosed scaffold (A49). Level 2 holds on the smile read by its own camera: A49's removal test passes, and M2 and M3 hold for 10 life days after it. Level 2 is the research's F3, a face read by the robot's own eyes as its reward. In the sim that face is a scripted avatar, so it becomes a person's only on the real G1.
- **Bar:**
  - each milestone reaches its ruler by A19's rule within 60 life hours (1.44M waking ticks: the pessimistic end of the forecasts above);
  - the never-taught tests of M2 and M3 (and M4, for the full claim) pass, and M1's where C36 shows they can reach p < 0.01 (M1(a) rests on a single first call from the doorway);
  - the log shows no episodic reset, and no reward but the face, pain and the charge (section 6);
  - **the long-run clause** (the research's F5, on its skeptic's stronger bar), at 200 life hours:
    - savings: a skill left unasked and unscaffolded for a span registered before birth is relearned in fewer exposures than it was first learned in;
    - plasticity: new words' namings to criterion, and new objects' showings to a first grasp, are no higher in the last 50 hours than in the first 50;
    - M1–M3's rulers are no lower than on their criterion days.
  - 200 hours is about ten times the store's 19 nights, so the clause tests what the night keeps.
- **Its cost, reported and never claimed** (the research's F2): its life hours to each criterion, its exposures to criterion by kind, and the overnight gain (A62), beside the table above with each method's signal. No multiple is printed.
- **Closest:** DayDreamer (a task reward per skill); HIL-SERL (demonstrations, a classifier and corrections, one task at a time); MIMo (one agent per skill, from a reward built for it, in an infant's body); the iCub (engineered maps, no person); Christiano et al. (a person's preferences, one simulated skill at a time); Gupta et al. (no resets, task rewards). SEDRo and Doyle et al. set such a world up, so the novelty must come from the result, not from the setting.
- **The reviewer's first question:** why about 60 hours, when HIL-SERL takes 1–2.5 a task? Because here there is no teleoperation, no classifier trained for a task, no reset and no reward that names a task, and the skills are kept afterwards.
- **What delivers it:** the grounded reward (6); a gate per limb with movement units (3.6); credit through the amygdala's tags and the night's tagged replay (7.3, 7.4, R8); the cerebellum (7.5); the spinal pattern generator (A48); recall into action (7.6); REM's twitches (A46); the parent's scaffolding (4.10).
- **Achievable here:** yes, if credit crosses the delay (risk 1) and movement units last (risk 3). At section 9's case C, 60 life hours take about 1.6–3.3 calendar days running around the clock, and 200 about 5–11, stretched by the Mac's share for the life.

**First 2: understanding, shown in the body by tests it was never taught.**
- **The claim:** its own life passes at least 3 of a hard set of 4 infant tests of understanding, none of them trained toward, each failed by its born state.
- **The hard set** (from the tests above, A55):
  - the ball rolled behind the low table, its fovea at the far edge first (Johnson, Amso and Slemmer 2003);
  - the cover lifted within 40 ticks of a toy hidden under it (Piaget 1954);
  - its fovea on the toy within 20 ticks of her silent head turn (Scaife and Bruner 1975);
  - a look to her face within 20 ticks of a never-seen toy (Walden and Ogan 1988).
- **What does not count:** a test its born state passes at p < 0.01 (measured at S5b, A63), and the tests this design passes by construction: its own touch against hers and its own rattle's sound (the efference copy feeds the forecast), and exploring what broke an expectation (the born orienting to sudden change, A43). They stay tests; they are not this claim.
- **Clever Hans:** at a probe she behaves exactly as at the matched moments, the stimulus apart, with her eyes on the child (A51), so it cannot pass by reading her.
- **Bar:** 3 of the 4 pass by A19's rule (one-sided p < 0.01, fresh items, pooled), reported at 60 and at 200 life hours and at each night boundary after. Infants pass them at 6–12 months, so no hour is promised.
- **Closest:** PLATO and Garrido et al. (surprise learned from passive video, with no body); Animal-AI (embodied agents mostly failed held-out permanence tests); Triesch et al. and Boucenna et al. (the skill tested was the skill trained). The review found no embodied learner with a sparse social reward that passed a battery registered in advance.
- **What delivers it:** the cortex's forecasts of its own senses (a broken expectation is their surprise), the amygdala, orienting, recall into action (A45), the store and the night.
- **Achievable here:** medium-low by 60 life hours, medium by 200.

**Dropped as firsts** (A63), each kept where it serves:
- **Speed alone** (the research's F2): single skills have been learned in minutes to hours from signals built for them. The body's hours are First 1's reported cost.
- **A face as the reward, on its own** (F3): real faces have served as rewards before (Broekens 2007; Veeriah et al. 2016; Li et al. 2020; EMPATHIC). In the sim the face is a scripted avatar read by a born reader, so here it is First 1's second level, and a first only with a person and the real G1.
- **Lifelong learning, on its own** (F5): learning without resets on a robot (Gupta et al. 2021), continued learning on hardware (Smith et al. 2022) and sleep-like consolidation of robot skills in simulation (Jayasinghe et al. 2026) did parts of it. It is First 1's long-run clause.
- **Grounded, compositional words** (F6): embodied agents understood held-out colour–shape pairs (Hermann et al. 2017; Chaplot et al. 2018). M6(b) stays a test.
- **One core, two bodies** (F7): false as worded, since the sim turns on organs the language body lacks (the amygdala, the event lines, the cerebellum, recall into action) and sim-only constants; and DreamerV3 used one configuration across bodies. The language digests held stay a disclosed property.
- **A first word from its own tract** (F8): Elija learned words from caregivers through a synthesized vocal tract (Howard and Messum 2014), and M6t is not forecast.
- **Seed 1 on the real G1** (F9): it needs the robot, and each step is the owner's call (section 16).
- **Not yet reviewed:** a word told once and acted on the next morning by the body, the robot's version of the language body's kiwi (named in `docs/ALIGNMENT.md`). It is registered only after a skeptic's review against the state of the art, before birth; the nearest known is fast mapping on an iCub with pre-built features (Twomey et al. 2016).

### The human-pace ledger: an optional report (A58; demoted by A66)

It reports the G1's milestones against infant norms, in waking hours, as the owner's bar of 2026-09-24 asked: "learning speed and understanding level of human". On 2026-09-25 he said he does not care about human pace (A66), so it is kept only as an optional report he does not need: never a headline, and nothing waits for it, birth included. Its firewall stands in full (below).
- **It is a ruler, never a target.** Nothing in the body, the parent, the brief, the tests or the build reads it.
- **It maps no month onto a life day inside the body** (A56).
- **Its arithmetic is in the study's `$S/pace/hours.py`.** If the ledger is kept, a tool that reproduces it is committed with the ledger's digest before birth.

**The ruler:**
- **Robot hours, H_r(m).** Count the waking ticks from birth to the end of the first of A19's two consecutive days on milestone m's ruler, then multiply by 0.15 s.
  - For a never-taught test, count to the day it first passes.
  - A life day's wake is 24,000 ticks, which is one hour (B19). Nights are excluded, as an infant's sleep is.
- **Infant hours, H_i(m).** These are the waking hours from birth to the norm's age: 24 h minus the mean 24-hour sleep at each age, integrated.
  - The sleep figures are from Galland et al. 2012, Table 2: 14.6 h at 0–2 months, 13.6 at about 3, 12.9 at about 6, 12.6 at about 9, 12.9 at about 12 and 12.6 at 1–2 years.
  - They are placed at 1, 3, 6, 9, 12 and 18 months, linear between and flat outside, over months of 30.4375 days.
  - The table's ±1.96 SD limits run from 9.3–20.0 h at 0–2 months to 10.1–15.8 h at 12. They scale H_i by about ×0.5–1.5 at 4 months and ×0.6–1.4 at 12.
- **Its first figure is the waking-hour age, A_r.** This is the infant age at which an infant at mean sleep has lived H_r waking hours, with its range under the sleep limits.
  - A_r is placed in the milestone's own age distribution, using WHO's percentiles where they exist.
  - For M7 those are: 1st percentile 3.8 months, 5th 4.3, 25th 5.2, 50th 5.9, 75th 6.7, 95th 8.0 and 99th 9.2. In waking hours: 1,144, 1,306, 1,602, 1,837, 2,108, 2,552 and 2,967.
- **Second, the ratio S = H_i / H_r,** printed at the median age, over the age window, and over the sleep limits.
  - S = 1 means an infant's pace per waking hour.
  - One seed gives one H_r against a population, so S carries false precision. It is printed beside A_r, never alone.
  - It is never printed for M1, whose infant range runs from a newborn's orienting to the CDC's 4-month item.
- **The body's own exposures to criterion** (the right column of the second table below) are part of fast learning's measure (A62), with or without the infant column.
- **"Understanding level" is answered only by which never-taught tests pass,** and at what A_r. A motor row never answers it.

**Waking hours by age.** The first row is at mean sleep. The second is under the sleep limits. The third is the hours with a face in view (see the exposures table below), interpolated linearly between 1 and 11 months and flat outside them:

| age | 1 day | 1 week | 1 month | 3 months | 4 months | 6 months | 9 months | 12 months | 18 months |
|---|---|---|---|---|---|---|---|---|---|
| waking hours | 9 | 66 | 286 | 889 | 1,209 | 1,870 | 2,898 | 3,925 | 5,979 |
| under the sleep limits | 4–15 | 28–103 | 122–447 | 432–1,339 | 625–1,787 | 1,035–2,700 | 1,729–4,060 | 2,478–5,362 | 4,030–7,909 |
| with a face in view | 2 | 16 | 72 | 212 | 279 | 400 | 545 | 642 | 814 |

**The ledger.** The kinds of criterion:
- **first:** a first occurrence, an earliest age;
- **mean** or **median:** attainment;
- **≥75%:** the age by which at least 75% attain (the CDC's items: Zubler et al. 2022), an upper bound on the median;
- **group:** the age of the lab group tested, which gives no onset age.

H_i is the median age's hours at mean sleep, with the age window in brackets.

| M | infant capacity | age | criterion | source | H_i |
|---|---|---|---|---|---|
| M0 test | tells its own touch from another's | 24 h | group | Rochat and Hespos 1997† | 9 |
| M0 test | matches a sound to its sight | 4–4.5 months | group | Spelke 1976†; Kuhl and Meltzoff 1982† | 1,209–1,372 |
| M1 | turns to a voice | within its first days; the learned turn returns after a dip at 2–3 months; 4 months | group; ≥75% | Muir and Field 1979†; Muir, Clifton and Clarkson 1989†; the CDC ("turns head toward the sound of your voice") | from 9–66 up to at most 1,209; no single figure |
| M1 test | explores the object that broke its expectation | 11 months | group | Stahl and Feigenson 2015† | 3,586 |
| M1 test | follows her head turn | a first stage at about 6 months (a third of trials under 4 months; nearly all at 11–14) | groups | Butterworth and Jarrett 1991†; Scaife and Bruner 1975 | 1,870 [under 1,209 to 4,604] |
| M1–M2 test | the occluded ball: surprise at its violation; an anticipatory look | 3.5–4.5 months; 6 months | groups | Baillargeon 1987†; Johnson, Amso and Slemmer 2003† | 1,048–1,372; 1,870 |
| M2 | first reach contact | a mean of 12.3 weeks (n 7); first reaches at 12–22 weeks (n 4); "reaches to grab a toy he wants" by 6 months | first; ≥75% | Clifton et al. 1993; Thelen et al. 1993; the CDC | 835 [813–1,870] |
| M2 test | looks to the parent's face at a new toy | the 10–13-month group; 12 months | group | Walden and Ogan 1988; Sorce et al. 1985† | 3,243–4,264 |
| M3 | first grasp after a reach | a mean of 16.0 weeks (n 7) | first | Clifton et al. 1993 | 1,106 |
| M3 test | acts for a contingent effect | 10 weeks | group | Rovee and Rovee 1969† | 671 |
| M3 test | the hand prepares for the grasp before touch | 5–6 months was the youngest group tested, so there is no onset age; whether that group closed the hand before touch is read at the source (C62) | group | von Hofsten and Rönnqvist 1988 | at most 1,536–1,870 if it did |
| M3 test | searches under a cover | from about 7.5 months; "looks for things he sees you hide, like a toy under a blanket" by 12 months | group; ≥75% | Diamond 1985; the CDC | 2,381 [to at most 3,925] |
| M4 | rolls back to front | 5.1 months, SD 1.5 (n 240) [±2 SD: 2.1–8.1]; "rolls from tummy to back" by 6 months | mean; ≥75% | Nelson et al. 2004 (Hong Kong); the CDC | 1,569 [610–2,587] |
| M4–M5 test | its eyes reach another's goal before her hand | 12 months, not at 6 | groups | Falck-Ytter, Gredebäck and von Hofsten 2006† | 3,925 (not by 1,870) |
| M5 | leans on its hands when sitting; head steady when held | by 6 months; by 4 months | ≥75% | the CDC | at most 1,870; at most 1,209 |
| M6 | calls a parent "mama" or "dada"; tries 1 or 2 other words | by 12 months; by 15 months | ≥75% | the CDC | at most 3,925; at most 4,945 |
| M6 tests | understands common nouns; forms a category from exemplars; points to ask for something; combines a colour with a noun | 6–9 months; 3–4 months; by 15 months; after year 1 | groups; ≥75% | Bergelson and Swingley 2012; Quinn, Eimas and Rosenkrantz 1993†; the CDC; Wagner, Dobkins and Barner 2013† | 1,870–2,898; 889–1,209; at most 4,945; over 3,925 |
| M6t | cooing; vowels imitated; canonical babbling; a first word | 1–4 months; 12–20 weeks; onset by 10 months in hearing infants, usually from 6 (a mode of about 7); as M6 | groups; onset | Oller 2000†; Kuhl and Meltzoff 1996†; Oller and Eilers 1988†; Eilers and Oller 1994 | 286–1,209; 813–1,404; at most 3,243 [1,870–3,243] |
| M7 | sits without support (no arm support allowed) | a median of 5.9 months [3.8–9.2, 1st–99th percentiles] (n 816) | median | WHO Multicentre Growth Reference Study Group 2006, Table II | 1,837 [1,144–2,967] |

**Exposures per waking hour, on both sides.** These are reported beside each row:

| exposure | an infant | the robot |
|---|---|---|
| words heard | about 12,300 adult words in a 12-hour recorded day (15,439 at 2 months, SD 8,234), with 221–271 conversational turns at 2–6 months (LENA: Gilkerson et al. 2017). The count includes overheard speech and the recording's naps, so it is neither per waking hour nor child-directed | the parent's 1,500–2,500 directed words a life day (4.6), all inside its waking hour. The ratio to an infant's directed words is unknown. It is reported, and never moved toward any norm |
| a face in view | about 15 minutes a waking hour at 1 month, about 5 at 11 months (22 infants' head cameras: Jayaraman, Fausey and Smith 2015) | its own share of ticks with her face in its fovea (M1's second ruler), measured, never assumed |
| contingent replies | a caregiver's, at the rates of human dyads | her declared rates (A52, C24, C57), and the logged ones |
| sleep per waking hour | 1.55 h at 0–2 months, 1.16 at 12 months (Galland's means) | 1.0: a night of 24,000 ticks for each day of 24,000 (A46) |

**Exposures to criterion: the second ledger.** Each entry is a lab paradigm, set against the body's own count of the same kind of exposure:

| paradigm | an infant | the body's own count |
|---|---|---|
| a contingent effect (conjugate reinforcement) | learned within one session at 10 weeks (Rovee and Rovee 1969†) | the rattle's holds and its arm's acts until M3's rattle test passes |
| a word from labelings | a new word from 9 labelings at 13 months (Woodward, Markman and Fitzsimmons 1994†) | the parent's namings of a word (a variation set counts as one, 4.5) until its looks on "where is the X?" pass A19's test for that word |
| words found in running speech | segmented after 2 minutes of speech at 8 months (Saffran, Aslin and Newport 1996†) | none yet. It stays only if an instrument for it is fixed before C36 (A55); otherwise it is dropped before the freeze |

**Reading it:**
- **The criteria differ.**
  - A first occurrence is an earliest age.
  - The CDC's ages are upper bounds on the median.
  - A lab group's age gives no onset.
  - A19's rule (two consecutive days at p < 0.01) is stricter than a first occurrence, so it reads the robot late against those rows.
- **Motor rows mix learning with growth.** Infants' motor milestones wait on growth and maturation. This body is born at full strength and never grows. The never-taught tests are the fairer comparison.
- **The tasks differ.**
  - The G1 has no neck.
  - Its sit leans with its hands on its knees. WHO's criterion excludes arm support, so M7's ruler is the easier task and the WHO row flatters the robot. The CDC's "leans on hands" is the nearer norm (M5).
  - It rolls from its back. Western infants usually roll front to back first; Nelson's sample rolled back to front first.
- **Waking is not alert.** A newborn's waking includes feeding, crying and drowsiness, so the early H_i overstate the usable hours. The face-in-view row partly corrects this.
- **Hours before birth.** Infants hear before birth (DeCasper and Fifer 1980). The robot's born parts stand in for gestation and evolution, disclosed as innate (section 10), and no weight learns before birth (section 11). Neither side counts these hours.
- **The samples.** Clifton (n 7) and Thelen (n 4) are small; WHO (n 816) and Nelson (n 240) are large.
  - Galland's limits are individual ±1.96 SD, so holding one limit across all of infancy widens the range.
  - Galland's Asian samples slept about 59 minutes less.
- **What was read at the source.** The Galland, WHO and CDC tables, and the Clifton, Thelen, Nelson, Eilers and Oller, Jayaraman, Gilkerson, Diamond, Bergelson and Swingley, Walden and Ogan, von Hofsten and Rönnqvist, and Scaife and Bruner summaries. The rows marked † are recalled and are read before the freeze (C62).

**The firewall** (A58):
- **Demoted, not removed** (A66). The firewall binds whether or not the ledger is ever printed. A ledger not frozen before birth is never reported.
- **The freeze.** The ledger, if kept, is frozen before birth and committed with a digest: its rows, sources, criteria and conversion. After birth no row is added, removed or re-sourced.
- **Every row is reported,** passes and failures alike.
- **Nothing reads it.** No parent brief, planner prompt, test, build step or constant reads the ledger or quotes an infant age.
- **The parent's word rate and timings are never moved** toward LENA or any other norm.
- **The owner's bar is answered by architecture alone.**
  - A low reading can lead only to a change of the learning machinery, justified on general grounds. After birth, that is a new seed (A20).
  - It never leads to a change of the parent, the room, the pace, or a constant chosen to move a row.

**The language body on this ruler** (an illustration; no S is printed for it):
- **Its hours: 197.5.** The save written after life day 404 holds 4,740,012 waking ticks; the counter in `body/core/instruments.py` moves only in the waking tick.
- **"404 life days of about an hour"** (`video/film5_design/life_montage.md`) is about twice too high.
  - Days were 7,200 ticks for a stretch (DIARY_BODY.md), and 24,000 only from 2026-09-18 (`ops/teaching_method.md` item 10).
  - Days 1–10 held 5.2 hours.
- **A child reaches 197.5 waking hours at about 21 days old** (14–48 under the sleep limits).
- **What it heard:** about 235,000 words in 51,066 lines, about 1,190 a waking hour. LENA's 1,025 a recorded hour is not comparable (overheard speech; naps).
- **Why no S.** It typed from a small alphabet, with no articulator and no body, and it was heavily drilled: "a lemon is sour" was typed 423 times.

## 13. Risks and how we see them

The numbers follow section 0.

| # | risk | what we watch |
|---|---|---|
| 1 | Credit across the delay: eligibility is 12 ticks at decay 0.8, so a reward 8 ticks after the act reaches it at 0.17, and after 11 ticks at 0.09. The amygdala's tag reaches back 64 ticks for memory and the night, never for credit. | `act_pred` against `act_inv` on held-out demonstrations, first; the value's rise at the touch; each limb's gate rate conditional on the ask (the causal limb should keep acting and the others fall back) |
| 2 | Balance at a 150 ms tick, in a stiff body: upright sitting tips back; the leaning sit holds only on locked servos; under the resting law every posture sinks. A limp upper body (16.2 kg, its centre 0.25 m above the hips) falls with a time constant of about 0.19 s, near the tick; the resting law's lag slows a sink to about 4° a second at 35° (an estimate from the hips' gains), so the danger is the child's own big steps | the sink rates (W1, again under the published gains: C46); the leaning sit's fall time; falls and strikes per hour. The cerebellum is born below the tick (R6c), but its teacher knows joint angles only, so it compensates loads and never sits (A44). If sitting has not come by M5's range, a balance loop below the tick would still be a change of body, so a new body with its own seed (A20) |
| 3 | Motor milestones by chance only; rolls not yet measured on the G1; the born units average about 1.4 ticks, far shorter than the babbler that justified them (section 3.6) | the born units' lengths (C38); rolls per life day by direction (chance: the born loop's share under the same placement); the trunk-first order; face-down bout lengths and the chest-up share; the parent's turns a day |
| 4 | The parent's strength: a person cannot sit up, slide or lift a 34 kg body; the catch's energy estimate has no source | the pull-to-sit's share carried by the child; holds at their caps; the catch (C6); acts refused for force, logged |
| 5 | Thrashing, learned helplessness, leaving the mat with no one to carry it back. The tonic drive follows the reward rate, so in a world that hurts on more than about 5% of ticks it turns negative and acting itself costs (A41) | the drive's value per life day (C47); per effector: gate open rate, Σ τ² / max, step reversals a second, thumps a minute; open rates after painful days; long still bouts; low charge with falling torque use; time off the mat and where it ends up |
| 6 | The critics' cost | the mean tick and its spikes; the solve every 256 ticks |
| 7 | Solver stability with the Dex3 hands | MuJoCo's warning counters every tick; `FatalError` caught; the life pauses on any reset or NaN (A18) |
| 8 | Pain from its own housings, now read through the joints (A37): on the old skin 6–12% of babble ticks, mostly the thighs' housings into the pelvis and the shoulders into the torso, part of it the collision hulls meeting (A54); a pressed housing that slides reads harder at impratio 10 (5.1); the newborn's withdrawal raised the pain it answered on 26% of onsets (C22). Under the joints' law a person's hold can pass a joint's limit through its lever (4.2), and a posture that loads a joint past its limit hurts (flat palms on hands and knees, section 6) | pain on still ticks and under babble from the joints (C5, C43); the share from hulls meeting (C59); the withdrawal's rate against rest and babble (C22); holds and guides against the law (C43) |
| 9 | Vision: a newborn's acuity in the fovea (3 px a degree through the born bank, A42), in grey, with colour from one camera beside the left eye (A38); on its back it sees mostly the ceiling and its own body, and the parent only when she leans over its chest or stands toward its feet; face down only the mat; sitting or standing only what is low and ahead (A22); the light changes through the day; the born face template, as built, rarely detects a real face, so the face event line and orienting's face cue may run on chance (C39) | the eye check at W3, again on the new code (fovea identity at least 0.75 on the ten toys and the face, under each light, by A33's readout: C3, C48); the fovea's forecast error on shown toys; the template's detections of her face (C39) |
| 10 | Vocal imitation near chance at birth (2 of 12 echoes accepted); tokens are the easier road to reward | the first accepted echo; the tract's share of "says"; the copy tests before removing either scaffold (section 4.9) |
| 11 | Disk and heat | the disk rule; one copy at a time; the mean tick over long runs |
| 12 | The tick: about 117–190 ms as designed, with the lead's additions and the colour camera's own view; about 65–135 ms with the exact levers and the eyes' lighter mesh copy, estimates (section 9). In lockstep a slower tick costs wall time only | the tick heat-soaked with the three views (C12, C48); each lever's saving under its rules and the machine's share (C72); B3 answered (A65). Other sessions' load slows it: at a load of 10–15 the eyes' render alone ran about 4× slower (section 9) |
| 13 | Smiles drifting into shaping; the parent talking too much | the neutral resting face; the one law; the born reading's hold; `tools/sim_parent_rates.py` (at least 40% of play ticks free of her voice) |
| 14 | The charge need is weak (drive reduction nets zero over a cycle), and feeds are frequent under babble (about 5–6 a day) | feeds the child starts itself; the bottle's stages |
| 15 | The amygdala's "away" gain could teach it not to look at a parent whose face predicts frowns | the share of ticks on her face after days with frowns (7.4) |
| 16 | Two toys cannot be held (the bear and the drum) | the grasp rate on the eight holdable toys; B1 |
| 17 | Sleep is a night by tick count, not a robot's sleep: live and dark since the lead's decisions (A46), no longer a pause | accepted for the first body; the night's length and cost (C52) |
| 18 | Sim choices that differ from the real robot. The lead's decisions removed most: colour stereo (A38), the servo gains (A39), touch on every link and pain read from it (A37), the parent reading a fovea the robot does not show (A40). Left: the microphones' and the speaker's places (B5), the imagers' near-infrared (3.4), the colour camera's place until read (C45), and every sensor model's constants until read (C44–C46) | the birth checklist's sensor-by-sensor check (A36); B5 read from Unitree's documents before birth |
| 19 | The parent sees only its trunk and hands (A40): a look made by the fovea alone is invisible to her, so asks are met less often, the ledger runs slow, and follow-in naming names the wrong toy more often | her reading against the fovea's target under the babbler (C58); met asks and the ledger per life day; the word pace's floor (a new word every 2 life days) holds whatever the ledger does |
| 20 | The observer misreads: light touches below its noise are not felt, a touch on the head is a touch on the torso, and its error under fast rolls is unmeasured | its error against the world's true contact torques, at rest and under babble (an instrument, C43) |
| 21 | The pixel reader never works, so the world-truth smile stays and the robot cannot be reached | C39 and C55 before birth; the scaffold's removal test on copies after birth (A49) |
| 22 | An imperfect parent pays fewer smiles, and credit across the delay (risk 1) gets harder | her declared rates against her logged ones (C24, C57); the felt reward per life day |
| 23 | The live night: its wall cost, and a twitch pressing housings into pain while no one is awake | the night's wall minutes and its pain ticks (C52) |
| 24 | The human-pace ledger, now an optional report (A66), read as a target or quoted as a headline: a norm pulling the parent's pace, a brief, a test or a constant toward it; a guessed multiple reaching the film | the firewall (A58, A66): if kept, the ledger frozen with its digest before birth, read by nothing that acts, every row reported, passes and failures alike; a ledger not frozen before birth never reported |
| 25 | A first claimed past what the life shows: "a person as the only reward" while she is scripted, smiles by a worth table and guides its limbs; understanding from a test that a reflex or the efference copy passes; speed printed as a multiple of methods that each learned one skill from a signal built for it | the registration before birth (A63, C70): the claim's wording, the born state's chance, the intervention log and the steering disclosed, the exact replay released; no claim leaves the project before its bar has passed and a skeptic has read it against the log |
| 26 | Speed bought by changing the body or the world: a lever that is not exact, a render shortcut past its rule, or a refused lever reopened for the new bar | the levers' three rules (A59, A64); the render's rule and C69's measure (A65); the eight digests, the `sim` profile and SimWorld's replay on every commit |

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

**The lead's decisions for the owner, 2026-09-24, on his bar** ("robot need to be like we put human brain in g1 and sim is reality"; an architecture worth billions when all is complete), **and on 2026-09-25, on his new word** (row 10). Under his standing rule ("never ask, decide"), the lead decided and reports; the owner may overrule any of them. Each is in the decision log with its reason and source.

| # | the lead's decision | where | log |
|---|---|---|---|
| 1 | The sim body is the real robot's body, so seed 1 can move to the real G1 as a change of world: touch only in the Dex3 hands, contact and pain elsewhere from the joints' efforts (B18); the D435's own sensors, with a camera model (B4); Unitree's servo gains, actuator and gyro models; motor temperature as a sense only; the parent reads its trunk and hands (B14) | 0, 1, 3.1, 3.3, 3.4, 4.10, 6, 10 | A36–A40 |
| 2 | The brain before birth: the gates' drives disclosed (a tonic drive following the reward rate; the performance error on the tract only); newborn acuity through a born bank; orienting to sudden visual change, habituating (A23 reopened); a scoped cerebellum (R6c); recall into action in R7; REM twitches in a live, dark night in R8; a born cry; a spinal pattern generator per limb | 3.5–3.7, 5.4, 7.3, 7.5, 7.6, 8, 10 | A41–A48 |
| 3 | The smile read from pixels by a born mouth-corner reader once the born face detector works; until then the world-truth smile a disclosed scaffold with its removal test | 3.4, 3.7, 6, 10 | A49 |
| 4 | The parent and the room: eight voice variants per line; the gaze leak closed; an imperfect parent who copies its movements; a richer room; hull pain measured first, then a decomposed collision copy if needed | 4.3, 4.4, 4.6, 4.8, 4.10, 5.1, 5.2, 10 | A50–A54 |
| 5 | Understanding: the roadmap's tests by milestone, with their chance levels, fixed before C36 | 12 | A55 |
| 6 | Dropped: a learning-progress reward; noradrenaline, acetylcholine and serotonin controllers; thalamic gain; a breath rhythm; an auditory-target reward; an acuity schedule and plasticity windows in year 1; a balance law | 6, 7.3 | A56 |
| 7 | The build plan re-estimated: birth about day 17 (range 15–22); the tick re-checked, about 117–190 ms | 0, 9, 11 | A57 |
| 8 | Human pace (later that evening): each milestone reported against an infant's waking hours to the same capacity, a ruler frozen before birth and never a target | 1, 12 | A58 |
| 9 | The speed plan: exact levers that change nothing the body senses or does, the steering keyed to ticks, reduced precision and a looser world refused, a rented machine only under conditions | 9, 4.5 | A59 |
| 10 | The owner's word of 2026-09-25 ("only local no cloud. i dont care about huuman pace i just want fast learninhg and for it to solve robotics that no archecture had done before"): local only, no cloud ever (B21, answered by his word); fast learning as sample efficiency in life hours plus the fastest honest local run; two firsts registered before birth, and the rest dropped with reasons; the local levers given build steps and the machine's share; B3 answered (the lighter meshes for its eyes; the colour camera's own view and the sun's shadow kept); the human-pace ledger an optional report | 0, 1, 2, 9, 11, 12, 13 | A61–A66 |

**Still defaulted, each with a recommendation** (the decision log's part B asks each as a plain question):
- the two toys a Dex3 hand cannot hold (the bear and the drum), and the cup, scaled to 0.8 for a creep that impratio 10 removed (B1);
- the colour twins for the never-taught word test (B2);
- the microphones' and the speaker's places, to be read from Unitree's documents before birth (B5; B4 was answered by the lead, A38);
- the child's voice identity (B6);
- the play space, now that no one can carry the child back (B7);
- the sofa's gap, the table's under-shelf and the morning basket (B8);
- the room's echo (B9);
- the charger (B10); the mat's size (B11);
- looking at the parent never raising her smile (B12);
- names, the parent's voice, no second adult (B13);
- the room's size for a 1.32 m child (B15);
- the parent's strength: a person's (B16);
- the word tokens, if the child's ears and voice never pass the tests for removing them (B17);
- a life day of one simulated hour, so the sun crosses the window in an hour (B19);
- the G1's own motor hum (B20).

B3 and B21 were answered on 2026-09-25: B21 by the owner's own word (no cloud, ever: A61), B3 by the lead for him (A65).

**Ours, decided here and disclosed:**
- the servo law, its gains set at load, and the step sizes;
- the persistence margin;
- F_pain's multiple;
- the kept and refused reflexes, the VOR on the software fovea among them, with its quick phase from birth, decided on biology rather than by a measured rate (A23, C33);
- the event line "a face in the fovea" from the born template on the fovea's pixels (section 3.4; its constants settled from its sources, C39);
- the IMUs' declared noise added by the world; the world's contact priority; MuJoCo's auto-reset disabled (A18, A21);
- a born cry kept, reversing the G1 amendment's refusal (A47, the lead's for the owner);
- the amygdala's law and constants (A16);
- the software fovea's control (A23);
- the critics' solve interval;
- the parent's method: its feelings, timings, force caps, ear and conduct (A25, A27);
- the tract's anatomy, calibration and alphabet (A26);
- the tests of understanding and the scaffold's removal (A28, A29).
- the parent's face made real and the detector left alone (A30); the newborn's generalized withdrawal (A31); impratio 10 by the contacts' physics (A32); what decides the eye check (A33); a new word on its pitch peak, on every ending (A34); the grasp summed at the spinal cord (A35);
- the stock model's torque limits, with Unitree's differing values recorded (3.2).
- the constants of the lead's decisions, each ours and fixed before birth: the observer's gain and the joints' pain line (A37); the camera model's form (A38); the gates' drive and error weights (A41); the fovea's bank (A42); the visual onset cue (A43); the cerebellum's expansion and law (A44); recall's maps (A45); the night's length and twitch step (A46); the cry's posture and line (A47); the pattern generator's form (A48); the mouth-corner reader and the scaffold's removal test (A49); the voice variants' ranges (A50); the parent's gaze in asks, her imperfection, her copying and her reading error (A51, A52, A40); the room's calendar (A53); the human-pace ledger's conversion and its firewall (A58); the steering's 400-tick key and the exact levers' rules (A59); the instruments of fast learning (A62); the firsts' wording, bars and registration (A63); the levers' build steps, the machine's share and the steering's lag (A64); the render's rule and the eyes' mesh copy (A65).

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
| strength | adult reference × 0.08, from memory | the model's own limits (section 3.2) |
| pain | 3 × 93 N ≈ 278 N | 3 × 337 N ≈ 1,012 N |
| hands | a tendon finger and a thumb (2 servos) | Dex3: three fingers, 7 joints; 8 of 10 toys holdable at impratio 10 |
| the alphabet | 40 joint settings in 9 motor effectors, and a 79-row voice | 43 joints, 3 gaze and 10 tract articulators in 9 effectors, and the 79-row silent output |
| reach lying down | 0.53 m; 3 of 10 toys | 0.83 m; 3 of 10 toys |
| sitting | propped upright by the parent in 2.4 s | leaning forward only; the parent cannot sit it up alone |
| touch | every link, as on an infant's skin | the Dex3 hands' 16 zones, as on the real hand; elsewhere contact and pain estimated from the joints' efforts (A37; the G1 amendment's sim skin on the other 28 links is gone) |
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
6. **After the first body: the real robot.** The body is the real G1's, sensor for sensor (A36): its senses are the real ones, its motors run at Unitree's gains, and its fovea, bank, observer, cerebellum and tract are software the robot can run. So seed 1 can go on in the real G1 as a change of world, once the smile is read from its own pixels (A49), the critics' solves run off the tick at fixed ticks, the tick is timed on the robot's own computer, and its first days are on the mat, eyes first. Each step is the owner's call. There the firsts become robotics results: First 1's second level meets a real person's face, and its long-run clause runs on hardware (the research's F9, section 12). The colour camera keeps its real place in the sim for this reason (A65).

The render cost is mostly fixed overhead (the pixel count barely matters), and physics contacts arise only near the child, so more rooms cost little per tick. They are loaded from the same maker.

## 17. Files and sources

**In `body/sim/` (on main since 1eb268b as the starting point; built further on the branches of section 0, not yet merged):**

| file | what |
|---|---|
| `make_g1room.py` → `g1room.xml`, `textures/room_*.png` | the maker and the generated scene: the stock G1 (included unchanged, by relative paths), the parent with its graded face, the living room (edit the maker, never the XML) |
| `g1scene.py` | loading the room and adding the G1's senses at load (the stereo pair at the D435, the ear sites); the parent's mocap and graded face; the welds (for toys; those on the G1's torso, pelvis and elbows are replaced by capped springs in W2, section 4.1) and the hand proxy; touch per zone; the birth pose (cached to `g1_birth_state.npy`); SimWorld grows from it. On `sim-world`: the parent drawn at birth (`born_parent`), and `Scene.pose` saved with the world |
| `g1eyes.py` | the prototype's two eyes (one render each, one read-back; the periphery; the movable software fovea); on `sim-world` `eyes.py` replaces it, and it stays so `tools/sim_look_g1.py` loads |
| `g1acts.py` | the parent's acts aimed at the G1's parts (attend, show, guide, two hands on the body) |
| `parent_kin.py` | the parent's skeleton, forward and two-bone inverse kinematics, hand shapes, the face (the scalar path, and the graded FACS face with `face_reading`); on `sim-world` its face is `parent_face.py`'s, re-exported |
| `parent_poses.py`, `parent_acts.py` | its scripted motions and acts, each with a report of joint ranges; an act never chooses the face |
| `parent_feel.py` | the parent's feelings, their display on the graded face, the born reading with the increment rule, and the no-farming self-test |
| `tract.py` | the child's articulatory vocal tract |
| `ears.py` | the two ears (built, P2): the spatializer (the exact rigid sphere behind the ears' converter), two calibrated cochleas, the brainstem's delay lines, the born lateral read, the onsets |
| `parent_ear.py`, `parent_ear_cochlea.py` | the parent's ear for the child's words, with the context and babble-bank decision rule, and the study's cochlea it was measured through (until P3v moves it onto `ears.cochlea`) |
| `voice/`, `lang/lexicon.py` | built (P1): the parent's voice (the Swift server, SSML registers, the life's cache and ledger, playback with the talk-over stop) and the born table of 79 rows with the words channel |
| `world.py`, `reflexes.py` | built (W1, `sim-world`): `G1World`, the core's `SimWorld` over MuJoCo in lockstep: the servo law set at load, weakness, touch and pain per zone with the 10 ms filter, the IMUs' declared noise, the gaze and the VOR, faults caught, save and restore with every random stream; the newborn's withdrawal and the grasp summed at the spinal cord |
| `eyes.py` | built (W3, `sim-world`): both eyes rendered into one buffer, the periphery and the fovea, the retina's code, the face test (truth only), the born face template |
| `parent_face.py`, `parent_face_customchild.py`, `assets/parent/`, `textures/parent_face_ao.png` | built (`sim-world`): the parent's face of human proportions, its lids, photometry, baked shade and collision pieces; the old face kept for the custom child's room |
| `make_livingroom_customchild.py` → `livingroom_customchild.xml`; `scene_customchild.py` | the custom child's room and loader, kept for reference (section 15) |
| `highchair_onearm.xml`, `make_highchair_onearm.py` | the one-arm high chair, kept for reference (`tools/sim_look.py` renders it) |
| `assets/unitree_g1/` | the stock G1 (commit dd8640e) |

**In `tools/`:** `sim_look_g1.py` renders the G1 room's stills into `video/sim_look_g1/` (`--out=` for elsewhere). On `sim-world`: `sim_babble.py` (the babbler and the speed tool), `sim_sink.py`, `sim_friction.py`, `sim_pain.py`, `sim_grasp.py`, `sim_eye_check.py` (C2, C3), `sim_face_measure.py`, `sim_face_photometry.py`, `sim_face_template.py`. On `sim-parent`: `sim_voice_check.py`. Tests: `body/tests/test_sim_world.py` and `test_sim_eyes.py` (`sim-world`), `test_sim_voice.py` and `test_sim_ears.py` (`sim-parent`).

**To create** (section 11):
- `body/sim/senses.py`, `body/sim/anatomy.py` (SimAnatomy);
- `body/sim/lang/{templates,conduct,ledger,day,transcriber,consts}.py` (P3, in verification on `sim-parent`), `body/sim/serve.py` (`/sim` on port 8030);
- `body/core/amygdala.py`, `body/tests/test_amygdala.py`;
- `tools/sim_parent_rates.py`, `tools/sim_digest.py`;
- `ops/sim_parent_brief.txt`;
- `body/tests/test_sim_lang.py`;
- `tools/pins/sim_script.npz`;
- for the lead's decisions: `body/core/cerebellum.py` and `body/tests/test_cerebellum.py` (R6c); the pattern generator, the cry and the twitch generator beside the reflexes (R6h, R8); `body/sim/observer.py` (the contact observer), `body/sim/motors.py` (the motor, gyro and heat models) and `body/sim/camera.py` (the camera model), with the three views and the born bank in `eyes.py` (W1 and W3 reopened); the variants in `body/sim/voice/` (P1b); the richer room in `make_g1room.py` (W5b); `tools/sim_hull_pain.py` (W4);
- for the decisions of 2026-09-25: the levers in their sessions' own files (R10, W7, P7, R8: A64); the eyes' lighter mesh copy, loaded world-side, and its check (W3, C69); the firsts' registration, committed with its digest (P4, C70).

**Stills and sounds:**
- `video/sim_look_g1/`: `room.png` (the parent kneeling by the G1, a hand on its chest), `kneel.png`, `show_toy.png` (the duck 40 cm before its eyes), `g1_eyes.png` (both eyes, peripheries and fovea windows, with the room at the same moment); `parent_faces_graded.png`, `parent_feelings_timeline.png`, `parent_face_fovea.png`. They show the parent's old face, and are redrawn after the merge.
- `video/sim_look_g1/voice/`: `voice_sheet.png` (15 s of babble; "ball" and "mama" as the parent, hand score, searched and echo; the vowel space) and the matching `.wav` files.
- `video/sim_look_allout/`: the custom child's stills.

**Scratch studies** (not in the repository), under `$S = /private/tmp/claude-501/-Users-lukehamond-Projects-project/81d92d50-4dd9-488b-8268-f1a474117bfc/scratchpad`:
- `$S/g1/`: the G1 in the room: `m_speed_g1`, `m_render_g1`, `m_reach_g1`, `m_postures_g1`, `m_grasp_g1`, `m_help_g1` (each a `.py` with its `.json`); `tmp/` (the solver side tests, the rolls, the showing composition; `t_view2.py` and `t_view3.py`, where the head camera's fovea can reach a face in each posture; `t_size.py`, the lying G1's size).
- `$S/g1/parent/`: the parent's limits with the G1 (`m_g1_ramp`, `m_g1_help`) and the face sheet.
- `$S/g1/voice/`: the tract's studies (`babble`, `search`, `t_imitate`, `t_reject`, `t_xbank`, `t_cost`, `t_calib2`, `t_fitcorners`, `sheet`) and results; `xvoices/` (8 extra speakers' clips, 22 MB).
- `$S/g1/amyg/`: the amygdala on a synthetic stream (`m_amyg.py` to `m_amyg4.py` and their `.json`).
- `$S/allout/`: the all-out world and the custom child; the core at this scale (`t20_humanoid_tick.py`, `t21_parts.py`, `t22_paired.py`, `t23_rss.py`, `prof20.py`); `move/` (learning to move); `lang/` (the parent's language: `synth.swift`, `voices.swift`, `ear2.py`, `parent_lang.py`, the tests); `owner_amendment_0924.md`.
- `$S/sim_wf/`: the old design's measurements that still hold.
- `$S/build/world/` (`fix` to `fix4`): the world track's measurements (friction, pain, grasps, the template, C2 and C3, the face's pixels); `$S/build/parent/`: the voice and ears track's logs.
- `$S/pace/`: the human-pace study and the speed plan (A58, A59, B21):
  - `norms.md`, `speed.md` and the skeptic's review folded into sections 9 and 12;
  - `hours.py`: the waking hours by age;
  - `calc.py` and `fold.py`: the calendar days and the costs;
  - `rbench.py`: the render measured under load;
  - the tables read: `galland2012.pdf`, `who2006.pdf`, `zubler2022.pdf`.
- `$S/firsts/`: the firsts research (`research.md`: the state of the art in September 2026, the candidate firsts and their sources). Its skeptic's review was returned to the lead and not saved as a file; both are folded into section 12's firsts and A61–A66.

**Sources for the G1's senses and limits:** Unitree's `unitree_ros` `robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf` (the `d435_joint`; its torque limits); Unitree's `unitree_mujoco` `unitree_robots/g1/g1_29dof.xml`; Unitree's G1 product page; the Menagerie `unitree_g1` README.

**Sources for the world, the parent's face and her voice:** MuJoCo 3.9's documentation (Overview, "Softness and slip"; Modeling, "Solver settings"; Computation, "Physical realism and soft contacts"); the face's norms and photometry (section 4.1: Farkas via Husein et al. 2010 and Virdi et al. 2019, Dodgson 2004, Rüfer et al. 2005, Gao et al. 2025, McKinney et al. 1991, Yaremchuk, Park et al., EyeWiki, Russell, Kramer and Jones 2017); the withdrawal's (section 3.7, A31); the voice's (Fernald and Mazzie 1991; Albin and Echols 1996, through Soderstrom and Bortfeld's review).

**Sources for the lead's decisions:** the Intel RealSense D400 series datasheet (the D435's imagers, colour camera and projector); Unitree's unitree_sdk2 G1 and Dex3 examples, unitree_rl_gym's G1 configuration, the unitree_hg `MotorState` and the Dex3-1 documentation (to read before birth: C44–C46); and the references cited in A36–A57.

**Sources for the human-pace ledger and the speed plan:**
- the norms: Galland et al. 2012, Table 2; the WHO Multicentre Growth Reference Study Group 2006, Table II; Zubler et al. 2022 (the CDC's milestones); Gilkerson et al. 2017 (LENA); Jayaraman, Fausey and Smith 2015; and the rest cited in section 12 (those marked † recalled, C62);
- the speed plan: MuJoCo's and PyTorch's documentation on reproducibility; RunPod's, AWS's and Hetzner's price pages, read on 2026-09-24; Anthropic's API prices (as of 2026-06).

**Sources for the firsts and the render's rule** (section 12, A62–A65; the research's links are in `$S/firsts/research.md`):
- the state of the art: Wu et al. 2022 (DayDreamer); Smith, Kostrikov and Levine 2022; Luo et al. 2024 (SERL) and 2025 (HIL-SERL); Rudin et al. 2022; Seo et al. 2025; OpenAI 2019; Kalashnikov et al. 2018 (QT-Opt); Hafner et al. 2023 (DreamerV3); V-JEPA 2 (2025); Black et al. 2024 (π0); GR00T N1 (2025); π*0.6 (2025); Law et al. 2014; Philipp et al. 2026 (MIMo); Ugur et al. 2015; SEDRo (2020); Doyle et al. 2023; Broekens 2007; Cui et al. 2021 (EMPATHIC); Veeriah et al. 2016; Li et al. 2020; Christiano et al. 2017; Nagai et al. 2003; Triesch et al. 2006; Boucenna et al. 2014; Piloto et al. 2022 (PLATO); Garrido et al. 2025; Crosby et al. 2020 (Animal-AI); Dohare et al. 2024; Gupta et al. 2021; Smith et al. 2022; Jayasinghe et al. 2026; Nguyen et al. 2026 (the survey); Hermann et al. 2017; Chaplot et al. 2018; Howard and Messum 2014 (Elija); Twomey et al. 2016;
- the render: RealSense's help centre, "Extrinsic Camera Calibration" (15 mm between the RGB sensor's and the left imager's centre-lines), read on 2026-09-25; Garland and Heckbert 1997 (quadric edge collapse).

**Documents:** ARCHITECTURE.md, BODY_SPEC.md and ops/review_2026-09-22.md (the defect numbers); `docs/audit/` (the roadmap, the brain systems map, the reality gap, understanding, value, the skeptic's verdicts).

## The decision log (2026-09-24, amended for the G1, for the first build, and for the owner's bar)

This log settles the design's open decisions and edge cases, in three parts:
- **(A) Decided here (ours):** each decision with its reason under the laws.
- **(B) The owner's calls:** plain questions, each with a recommended default.
- **(C) Open until measured:** what decides each one, and where it is measured.

Where a decision changes an earlier section, that section points here. W, P, R and S refer to the build plan (section 11). A36–A57 were decided by the lead for the owner, on his bar, under his standing rule to decide rather than ask; the owner may overrule any of them. A58 and A59, the human-pace ledger and the speed plan, were decided the same way later that evening, on his bar of human learning speed and understanding; B21 asked him about spending. A61–A66 answer his word of the morning of 2026-09-25 ("only local no cloud. i dont care about huuman pace i just want fast learninhg and for it to solve robotics that no archecture had done before"), decided the same way; his word answers B21, and the lead answers B3 (A65). A60 and C64–C68 are the parent session's, on `sim-parent` and not yet merged; the numbers here follow them.

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
- The first build's verification rounds (later on 2026-09-24; W1 and W3 on `sim-world`, P1 and P2 on `sim-parent`):
  - the grasp as a core hook took the hand's tick, so a touched hand could never let go (A35);
  - world truth (the face test) had entered the body's observations, the parent was never drawn at birth, and a lamp shone from the child's eyes (A1, 5.1);
  - her cartoon face was far from a real face's proportions and photometry, and the born template, measured on a real one, still barely fired (A30, C39);
  - a withdrawal tuned by a born copy of the body was an adult's reflex (A31); impratio needed a physical reason (A32); the eye check's verdict turned on the readout and on the views (A33);
  - the parent's new-word emphasis was measured only on "." lines, where the engine spoke the "." as "period" (A34);
  - the torque limits "as Unitree publishes them" were Menagerie's, and Unitree's own sources disagree (3.2).
- The audit against the owner's bar (the evening of 2026-09-24; `docs/audit/`: the roadmap, four studies and the skeptic's verdicts on them):
  - under A20 the real G1 would be a new life: the sim body differed from the robot's in its skin, its colour stereo, its servo gains and its face reward (A36);
  - the gates carried drives in the code (`gate_tonic` 0.25, `gate_int`, `gate_int_form`) that the design denied ("no intrinsic bonus", 7.3) (A41);
  - the fovea read 4-px colour means, about 0.19 cycles a degree, below a newborn's ~1 (A42);
  - her eyes went to the object on the naming word in asks too, so the child could pass by following her gaze (A51);
  - every spoken line was one byte-identical waveform (A50);
  - 35% of the contacts over F_pain came with every hinge more than 0.2 rad from its range's ends: collision hulls meeting, measured on the per-joint instrument, not the born loop (A54);
  - the colour-twin test's chance could sit near its ceiling (A28, A55);
  - no organ acted below the tick, and a loop added there later would be a new body (A44).
- The human-pace study (later on the evening of 2026-09-24; `$S/pace/`, reviewed by a skeptic):
  - the language body's "404 life days of about an hour" was about twice its waking hours (197.5) (A58);
  - a column of what the ratio would read if the forecasts held turned guesses into headline multiples before any life (A58);
  - Claude's steering by wall minute tied the teaching's density to the machine's speed and heat (A59);
  - at a load of 10–15 from other sessions, the eyes' render ran about 4× slower than section 9's figure (section 9).
- The firsts research and its skeptic's review (the morning of 2026-09-25; `$S/firsts/`):
  - "a person as the only reward" is not what the design runs: her smiles follow a fixed worth table, her hands guide its limbs and place its toys, and a language model steers her curriculum (A61, A63);
  - speed alone is no first: single skills have been learned in minutes to hours from signals built for them (A62);
  - three tests of understanding pass by construction: the efference copy predicts its own touch and its own rattle, and the born orienting explores what broke an expectation (A63);
  - three candidates had been done before: compositional words (Hermann et al. 2017), one configuration across bodies (DreamerV3), words through a synthesized vocal tract (Elija) (A63);
  - "no cloud" against Claude's steering, a hosted model's row inside the life (A61);
  - the colour camera's place, 15 mm from the left imager, against the colour window's cells where its hands reach (A65).

### (A) Decided here (ours)

**A1. What counts as looking at the parent's face** (the reward's gate; sections 3.4 and 6)
- **The test.** It runs on each tick for each eye, and passes when all four hold:
  1. the parent's mouth point lies inside that eye's software fovea window (64 px at 3 px a degree since A42, about ±10.5°);
  2. a ray from the eye to the mouth point hits nothing first. The ray runs over the shapes the eyes render, so the child's own hand, a toy, the parent's hand or its hair all block it;
  3. the parent's face is turned within 75° of the eye (in profile a smile still reads; from behind it does not);
  4. the face's front (an ellipse about 0.17 × 0.21 m) covers at least 20 fovea pixels at the first build's 1.5 px a degree, scaled by the cosine of that turn: the same solid angle is 80 pixels at the fovea's 3 px a degree (A42), so the test's reach is unchanged.
- Either eye passing counts. The test must pass on 2 consecutive ticks (300 ms), so a sweep of the window across the face is not a look.
- **The range this gives,** at 1.5 px a degree (and the same at 3, the criterion kept in angle): about 3 m with the face turned toward the child. The parent on the sofa can be seen; from the hall, mostly not; closer than 25 cm, never (A3).
- **No mutual gaze is required.** Where the parent looks is its own cue, not part of the child's gate.
- **Why a ray test, not a segmentation render.** Nine rays cost microseconds, where a second render per eye costs milliseconds. C2 checks the rays against a segmentation render; if they disagree too often, the render is used.
- **The born reading reads the parent's expression from the world:** 2 × (smile − frown) of her graded face, not from pixels.
  - The grades are visible in the fovea's pixels only at the lean-in distance, and on her face of human proportions in fewer pixels than on the cartoon's, differing by light (section 4.3); from about 1 m the mouth is too few pixels for any born pixel reader.
  - This is disclosed as how the born reading works. The test above is where the child's own eyes decide.
  - **Amended by A49:** the world's value is now a disclosed scaffold. Once the born face detector works, the reading comes from the born mouth-corner reader on the fovea's pixels, and this test and the world's value leave the body.
- **The test gates the reward's carrier and nothing else.** The critics' and the amygdala's event line "a face in the fovea" comes from the born face template on the fovea's own pixels (section 3.4). Given this test instead, the body's value would read a perfect world-truth face detector: the one feature that best predicts a smile.
- **Built (W3), and measured:**
  - A first hit within 2 cm of the mouth point, or on her own lips, counts as the face (RAY_SLACK_M). A ray grazing her lip corner on a face turned past about 84° now reads "turned away", not "blocked".
  - C2: over 2,000 babbled frames with her head kept inside the room, the ray agrees with an 8× segmentation render on 99.8% of the 975 frames whose geometry passes (98.7–100% in every distance bin from 0.3 to 3.5 m). The rays stay.
  - The born template on her face of human proportions, over the same frames: 12 hits in the 453 where the test passes (6 of 82 at 0.3–0.6 m, none beyond 1.5 m), and false alarms on 4 of 736 frames with her face not seen and 3 of 401 with no face. Placed facing the child at 0.3–2 m, it detected her face (centred on it, at its size) in 1 of 48 fovea readings; its other matches were chance, as strong on the window upside down; in the periphery, never (C39).

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
- **Her copying of its movements** (A52) is the parent's method, on the world's side: she mirrors what its arms do. The child's mapping from seen acts to its own is learned, as being imitated builds it in infants (Ray and Heyes 2011); nothing in the body is added for it.

**A11. The hands' grasp**
- **Holding is friction only:** no weld, no attach and no sticky rule for the child. The Dex3's collision shapes keep their friction as shipped; the world's geoms set the other side of each contact at priority 2 (C26).
- **The grasp reflex** (section 3.7) closes six of the Dex3's seven joints (the thumb's rotation has no closing sense) one small step a tick while the palm's touch is at least 0.3 N, unless the hand's own act opens it that tick. The palm is its own touch zone; its touch is read from contacts (the real Dex3-1 has tactile arrays). It is summed at the spinal cord with the hand's own act (A35), so the hand's gate draws every tick and its opening acts keep their eligibility.
  - It never opens by itself. Letting go is the child's own act and is learned; the ask "give me the cup" needs it.
  - It fires on anything in the palm: a toy, the bottle, the parent's hand, the mat, or its own fingers. W4 counts closed fists on the mat, and fists closed on its own fingers (C41).
- **Eight toys can be held** at impratio 10 (section 3.8). The bear and the drum are for looking, pushing, hitting and naming unless B1 changes them.

**A12. Pain for a body on the floor**
- **Amended by A37:** the body's pain is now read from the joints' efforts: a joint's outside torque past its own limit, or the base's outside force past F_pain. What follows was decided for the sim skin; its filter, its one-threshold reasoning, the self-contact rule and the robot's damage limits carry over, and its measured rates are the old skin's.
- F_pain stays at 3 × the body's weight per link: about 1,012 N for the G1 (now for the free base only).
- **A link's force for pain** is the tick's largest 10 ms mean (5 physics steps) of its summed normal force.
  - A single-step solver spike is not a blow. Real impacts in this contact model last 10–30 ms.
  - Babble's single-step peaks at the tick ends (944–1,799 N) made phantom pain a real risk. Measured with the filter in the built world: 11.6% of babble ticks, 0 at rest (section 6, C5).
- **Self-contact counts:** a kick to its own leg hurts. MuJoCo already ignores contact between neighbouring links. Any other pair that presses into itself at rest is listed in the world (never by editing the G1's file), and disclosed.
  - The list removes that pair from touch and pain only, as a sensor's blind spot. It never excludes the pair from collision: the G1's self-collision stays as shipped, so the body's physics is the stock robot's.
  - **Built (`rest_blind`): the list is empty.** For the G1 as born no pair presses at rest, so every self-contact is felt. A wider rule, blind to a joint's inner links pressing each other, was built and removed (6003014): it claimed those contacts were "the range, not a blow", but 35% of them above F_pain came more than 0.2 rad from every range end. They are convex hulls meeting, as motor housings would on the real robot, and they carry most of babble's pain. How much of the body has skin was the owner's (B18); the lead answered it for him: none outside the hands (A37). The skeptic's reading of the same figure (the other 65% near a range's end, measured on the per-joint instrument) is A54's.
- **Not at birth:** a separate pain at a joint's range (C). Under A37 a stop's reaction is an outside torque the observer reads, and hurts past the joint's limit like any other (section 6).
- **The parent:** a hold's spring force is touch on the held link under the same law as any other force. Her caps (at most 200 N) kept it far under F_pain on the skin, so no exception to the law was needed. Under the joints' law (A37) a hold acts through a lever and may reach a joint's limit, so every hold and guide is checked (C43); one that would hurt tightens its cap, and still no exception is made.
- **Being pinned** costs −1 a tick for as long as it lasts. The only source we know of is the parent, and the yield rule ends it.
- **What 1,012 N means for 34 kg.** The rule is the custom child's (3 × 93 N ≈ 278 N), carried to the G1's declared mass, not refitted:
  - the loads it carries are far below it: the largest resting contact lying is 272 N, and the pelvis carries about 250 N in the leaning sit;
  - the thump line the parent reads as distress is half of it, 506 N (A13);
  - the custom child's falls from the prop struck its head at 260–890 N at 9.45 kg; the G1's falls under the final servo law are not yet measured (C31). A fall that passes 1,012 N hurts, as it should.
- **One threshold for all 45 touch zones,** not one per zone (the skin's reasoning; under A37 the joints' limits are each joint's own, from the file, and F_pain is kept for the free base):
  - a threshold from each link's own mass would put a finger link's at a few newtons, so its own weight on its hand, or any grasp, would hurt;
  - the skin's pressure threshold would need contact areas, which MuJoCo's contacts do not give reliably;
  - the real G1's damage limits per link are not known here (C37).
- **The robot after.** If any real G1 link is damaged below 1,012 N (C37), the pain law takes that link's limit for the next body, which is a new body with its own seed (A20), never a change to seed 1.
- **The friction cone's share** (5.1): at impratio 10 a pressed housing that slides reads harder. It is disclosed as impratio's cost, never a reason to change the pain law (C5).

**A13. What the parent does when the child does not look, is distressed, or babbles**
- **When it does not look:**
  - she follows its attention, naming what its fovea rests on;
  - she shakes a toy that makes a sound beside her own face;
  - she calls, at most once per 240 ticks. After 3 calls unanswered in 720 ticks, she carries on with the activity where the child is looking.
  - She never frowns for not looking, never touches the child to make it look, and never moves her face onto its line of gaze.
- **When it is distressed.** The G1 has no face, so distress is read from outward events: a pain event in the last 40 ticks; its born cry (A47); over 100 ticks face down; thumps (limb strikes above half F_pain, as the world measures them, the parent's own perception) at more than 3 in 40 ticks; the charge light low. She checks in this order:
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
- **Amended by A59:** "a minute" in this entry is now a steering window of 400 ticks (60 s of sim time). Each row is applied at a fixed tick, so the teaching's density no longer drifts with the machine's speed or heat.

**A15. Sentences and vocabulary growth** (section 4.8)
- A queue word enters only when its referent or act is in the world and the parent can show it within the minute. Colour words wait for the colour twins (B2).
- Claude may add queue words at a night boundary, under the same test. Otherwise the queue is fixed.
- **Unchanged:**
  - the line check: at most 6 words, `. ? !`, and the vocabulary plus that day's new word, placed last;
  - the pace: at least 1 new word every 2 life days (the world does not wait for the child), at most 3 a day.
- A queue word the child says before it is introduced is not recognised: the parent's transcript and her ear know only the vocabulary.
- **The never-taught pairs** (section 12) are listed before birth and held out by the line check until their test.

**A16. The amygdala's constants** (the owner's decision 2; section 7.4 and its rows in section 10). The organ is the owner's decision; its law and constants are the body's, so ours. Each is fixed before birth, and none is fitted to a rate. It replaces this morning's "valence learner", whose short memory (forgetting 0.998, about 500 ticks, fewer than its 525 inputs), single signed forecast, tag scaled by running means and tag-weighted `act_pred` lesson were each measured or reasoned wrong.
- **Its input:** the cortex's stream C/√d (the high road), the anatomy's event lines (the low road: 13 for the sim since the lead's decisions, the same declaration the critics read) and a level. No world truth.
- **Its heads:** one per reward source and sign; for the sim, face +, face −, pain −, charge + and charge −. The basolateral amygdala keeps good and bad apart; one signed forecast would cancel a cue that brings a smile and then pain.
- **Its horizon:** 0.9375 a tick, dopamine's own discount, so no new time constant.
- **Its memory:** τ_a = 4,096 ticks (band 6's clock), about eight times its 525 inputs (526 with the 13 event lines of the lead's decisions), so the fit is determined. Learning in one pairing comes from least squares, not from a short memory.
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
- **Before the night** (live and dark since A46):
  - at tick 23,700 the parent releases every hold and steps back, so the night's world has no parent contact;
  - the child sleeps as it lies; the world runs dark through the night, and only its twitches move it (a fall in progress goes on as a fall);
  - the light follows the day (section 5.4).
- **No death and no shutdown.** At h = 0 the body keeps 30% of its strength. The parent feeds below 0.35 and checks within 200 ticks, so h below 0.1 is a parent defect.

**A18. Crashes, disk and heat**
- **After a crash, the lost part of the day replays exactly.** Every random stream, Claude's rows and the synthesized clips are logged, and the tract is deterministic, so the life resumes from the last save (the last night) and replays the lost part of the day exactly. A crash costs wall time only, so there are no saves in the middle of a day.
- **MuJoCo's silent reset and its fatal errors.** MuJoCo resets the state by itself on a bad acceleration, and counts it in its warning counters; it can also raise `mujoco.FatalError` inside `mj_forward`. MuJoCo 3.9 has a switch for the reset (`<flag autoreset="disable"/>`), so the world turns it off: a bad state then stays visible as NaN instead of being silently replaced by the start pose. The world also checks the counters every tick and catches the error. On any reset, NaN or fatal error: Built in W1: the flag checked at load; each tick's end state checked against `mjMAXVAL`; the closing forward pass inside the catch; on a fault the world puts back the tick's starting state and raises `WorldFault` (tested with a velocity of 1e11 injected on a tick's 75th step).
  - the life pauses before that tick reaches the body;
  - the last good state is saved;
  - the fix is measured on a copy;
  - the life resumes from the last save with the fix, logged as a change of world at a boundary.
- **The solver** is the elliptic friction cone with multi-point collision detection off (section 5.1), the only setting that ran the Dex3 grasps clean, with impratio 10 (A32).
- **The voice server.** A cache miss makes the world wait. If the server is down for 60 s, the life pauses at that tick. It never skips or swaps a line. If Claude is down, the fast layer runs alone (A14). Built: `SynthServer` gives up at 60 s and raises `VoiceDown`.
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
  - Birth needs at least 8 GB free: the peak of about 4.5 GB plus the rule's floor of about 2.7 GB, and a margin (section 9; the earlier 6 GB rested on a save of 0.95 GB that had no source).
- **Heat.** In lockstep, heat slows only the wall clock, never the life. The birth checklist's mean tick is measured over two heat-soaked hours. One life at a time on this Mac.

**A19. How each milestone is judged** (section 12)
- **A milestone is reached** when its ruler holds on 2 consecutive life days and a watched session at 1× on `/sim` shows it at least 3 times. The report describes what was seen, with stills redrawn from saved states.
- **Understanding is judged as well:** the milestone's never-taught test (section 12) is reported beside it, passed or not yet, with each probe seen described.
- **Chance is always the same body's own rate:** its rate at matched random moments of the same day with no cue (as the ledger's base rate), or, for M1, its own first 2,000 ticks of life. The babbler's numbers, written down before birth, are chance for the motor rates. There is no trained baseline run.
- **The test** is one-sided (binomial or permutation), p < 0.01 over a life day. No claim rests on the parent's ledger and its 10-ask window at p < 0.05 (4.8); claims about words are made with the word scaffold silenced (A29, A55).
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
- Its torque limits are its file's, never another source's (section 3.2: Unitree's own differ from the file and from each other).
- The Menagerie scene light is switched off at load.
- Anything the world needs about the G1 is set in the world, never in its file, and never on its geoms at load either:
  - frictions and contact softness through the world's own geoms at contact priority 2 (C26), and the world's options (impratio 10, A32) and the mat's deep collision box;
  - a self-contact pair that presses at rest is blind to touch and pain, never excluded from collision (A12; none presses at rest for the G1 as born);
  - the IMUs' noise its file declares is added by the world, since MuJoCo does not apply it.
- **What does change at load, and why each is still the stock robot:** cameras and sites added (sensors only); the servos' kp and kd, which the real G1 takes with every command, now Unitree's published gains (A39); the torque limit scaled by weakness (0.3 + 0.7h), a clip the real robot's command could apply. The last is a sim physiology with no counterpart in the robot's hardware, disclosed as such.
- **No sense goes beyond the robot** since A37 (the G1 amendment's skin on the zones outside the Dex3 hands is gone).
- **Amended by A54:** if W4 finds pain from its collision hulls meeting, its collision shapes may be replaced at load by a convex decomposition of its own meshes, loaded world-side: the same robot's geometry, closer, with the file still byte-identical. It is decided before birth.

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
  5. **Orienting's face template** can fire only where a face can appear: on its back, above its chest and toward its feet. As built it rarely fires even there, and her face in the kneeling attend pose lies sideways in its fovea (C39).
  6. **M4's chance.** The born orienting may turn the trunk toward her voice and tip a roll toward her. The babbler that sets M4's chance runs with the born motor timing (C38), the born orienting bias and the VOR on, since they are part of the born body, so born turns count as chance.
- **The parent reads where it looks** from its trunk and hands (B14, answered by A40), never from the fovea's window.

**A23. The software fovea's control** (sections 3.4, 3.5 and 3.7)
- **Its state:** yaw and pitch (both windows together) and vergence, in degrees in each camera's image. Born at 0, 0, 0: both windows centred, the eyes parallel.
- **Its steps:** ±4° or ±11.5° a tick for yaw and pitch; ±1.7° or ±5.7° for vergence. A 30° shift takes 3 ticks. A newborn's gaze shifts to far targets are also slow and come in steps, a series of small saccades (Aslin and Salapatek 1975), so no faster saccade is added.
- **Its reach:** each window's centre is kept inside its image: ±38.1° × ±20.3° of the camera's axis (built; `g1eyes.py`'s docstring said about ±33° × ±18°, a linear estimate).
- **Vergence:** 0–11.4° (2 atan(0.025 / 0.25), the 50 mm baseline), from parallel to a point 0.25 m away, the nearest her face comes (A3). Anything nearer is seen double, as inside an infant's near point.
- **At rest the window stays where it was left.** Nothing in the image pulls it back to the centre.
- **The VOR** (section 3.7): each tick the window counter-shifts by the torso gyro's rotation over the tick about each camera's own image axes (the gyro rotated into the camera's frame, pitched 47.6° from the torso), at gain 1, and is clipped at its reach.
  - Roll about the line of sight is not compensated. The window is square and cannot turn; the human torsional VOR is weak too.
  - **Its quick phase is kept from birth** (changed in the review of the amendment). The first draft had none and would have added one before birth if the window sat pinned at its edge on more than 10% of ticks: a born reflex chosen by a measured rate, which the laws refuse. It is decided on biology instead: the VOR is a brainstem reflex of slow and quick phases together, and rotating a term newborn elicits nystagmus with both (from memory; W3 checks a source). So when the counter-shift would carry a window past its reach, the window jumps back, in the direction of the trunk's turn, by half its reach, within that tick. It is logged as reflex, with no gate eligibility. C33 counts the quick phases and only reports them. Built (d1e35b7): with the trunk turned 50° and the window at its edge, it gave 2 quick phases and no tick pinned.
- **Refused:** smooth pursuit, optokinetic following, a saliency map, or any rule that moves the fovea to the brightest thing. Pursuit is learned, as infants' smooth pursuit develops over the first months, through `act_pred` and the gaze's forward model. The orienting bias is the only born pull.
  - **Reopened by A43:** the refusal of "the newest thing" cited no biology, and newborns do orient to peripheral onsets through the subcortical route (Johnson 1990). So a sudden local change in the periphery joins the face and the sound as a third born orienting cue: a bias of the same kind, habituating, never a forced move and never a saliency map.
- **One effector for both eyes.** Only conjugate and vergence commands move the windows (Hering's law), so one eye never looks away alone.
- **Its consequence sense** is the window's state (6 numbers in the body channel) and what the fovea then sees.
- **Its fatigue** is 0.03 a step; the gaze's gate reads its own.
- **One window serves what the fovea sees and the face test (A1).** The parent no longer reads it (A40); the instruments do.
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
- **Her holds are touch, never pain** (A12): on the skin by their caps; under the joints' law by checking each hold and guide and tightening a cap that would hurt (A37, C43).
- **Why a person.** A real G1's carer is a person. A stronger parent would carry the child through the postures it has not reached, and B16 asks about that.

**A26. The voice's alphabet: the vocal tract** (section 4.9)
- **Ten articulators,** each taking one of {−0.6, −0.2, 0, +0.2, +0.6} of its range a tick, from where it is (the servo law): 50 striatal rows.
  - The steps were enlarged from ±0.1 / ±0.3, which could not open the lips in one tick. Real lips and tongue tips cross their range in about 70 ms.
- **A rest is silence.** At rest the targets relax, with a time constant of 1 tick, to a silent resting posture: nose breathing, the lips nearly closed, the velum down. The gate's "no" means quiet, as a limb's rest means still.
- **One target per articulator per tick** gives at most 3.3 syllables a second, about canonical babbling's rate. A word takes 4–6 ticks. Nothing sequences sounds inside a tick.
- **Physics, not rules.** The articulators move with damped muscle dynamics (57–120 ms), so sounds glide into each other; breath (400 cm³, about 2.6 s, refilled in 0.8 s) makes breath groups. No phoneme table exists anywhere.
- **Fitted to children's vowels, never to the teacher.** The tongue's corners were fitted to children's vowel means (Peterson and Barney). Nothing was fitted to the parent's words or voice. Its level (GAIN) puts a steady /a/ at 62 dB SPL at 1 m, ANSI S3.5-1997's "normal" vocal effort. It is computed from the parent's engine level (`t_calib2.py`), which is defined as that same 62 dB, so in pascals it is the standard's level, not a fit to her voice; giving a held vowel the level of running speech is our choice.
- **Not tuned to reach a word.** "bye" was reached by no act tried, and "hi", "pip", "up" and "see" only by hand. The tract stays as it is, and the first 50 words stay as they are. A word it cannot yet say is still one it can understand, and name by tokens while the scaffold lasts.
- **It hears itself at t+1,** through its own ears, about 19–20 dB above the same sound from 1.5 m (the exact sphere's level for its speaker on the head's front), with no bone conduction.
- **The silent token output** (effector 1) is a second, separate effector with its own gate.
  - It never talks over her: a token made during her line is read when her line ends, and it never earns a frown.
  - The ledger keeps what is said by the tract and by tokens apart from birth.
- **Deterministic** from its seed and the acts; its sound is not saved.

**A27. The parent's ear: how she recognises the child's words** (section 4.9; `body/sim/parent_ear.py`)
- **It is in the world, never in the body.** It uses the same born cochlea as the child (from P3v; the prototype still hears through the study's), a shift of 0–4 bands, cepstra c1–c12 and Itakura DTW.
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
  - **amended by A55:** the pairs are (blue, ball), (red, block), (yellow, cup) and (green, car): the twins are the targets. Each colour word is heard on at least 2 kinds of toy, never with the held-out noun: "blue" on the blue block and another blue kind, "red" on the red ball and another, "yellow" on the duck and another, "green" on the green cup and another. The richer room's examples (A53) never repeat a held-out pair. (The first form targeted the familiar originals, where the noun-alone chance could sit near 0.8, and then even 20 of 20 gives p = 0.0115, failing A19's 0.01.) A colour and a noun together are understood after the first year in infants (Wagner, Dobkins and Barner 2013; Fernald, Thorpe and Marchman 2010), so this test is not expected to pass in year 1;
  - a pair is tested only once both its words are "understood" alone (the ledger) and each has been heard in at least one other pairing;
  - the line check holds each pair out of every line: templates, Claude's lines, recasts and echoes.
- **The combination test:** "where is the blue ball?" with both balls in view; the fovea lands on the blue twin within 20 ticks and stays 2 ticks. Chance is its share of landings on that twin when she asks with the noun alone ("where is the ball?") at matched moments. "push the blue ball" (the owner's example, in the twins' colours) follows once "push" is understood. Her eyes stay on the child throughout (A51).
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

A30–A35 were decided in the first build's fix rounds, for the owner, under his standing word to decide toward the laws rather than ask. Each is ours to answer for.

**A30. The parent's face: the world made real, the detector left alone** (decided in the W1 verifier's third and fourth rounds; sections 4.1 and 4.3, C39)
- The born template never fired on her cartoon face: irises 19.6 mm against a real 11.7, eye whites 27 mm tall against about 10, eyes 72 mm apart against about 62, and no dark eye region. Changing her face or the template until it fired would fit one to the other.
- **Decided: fix the world, not the detector.** Her face is built to a real woman's cited proportions and a real face's photometry at low spatial frequency (4.1), with her expressions kept and her look warm. The template, its sizes and its threshold are untouched, and nothing on her face was tuned to cross them.
- **The result is a finding:** the template detects her face in 1 of 48 fovea readings at 0.3–2 m, and never in the periphery; its other matches are chance. Real faces are not what the template, as written, finds, so its own sources decide it (C39).
- Her collision became her drawn face, and her pixels at each distance and light were measured again (4.3).

**A31. The withdrawal is a newborn's** (decided in the W1 verifier's third round; section 3.7, C22, C40)
- W1's second fix made the withdrawal precise: a "local sign" computed on an exact kinematic copy of the body, moving the joints that carry the hurt spot away, from a new afferent (`pain_site`, where on the link it pressed and the contact's normal). That is an adult's tuned reflex, born; and its normal was the contact's, not the skin's (26% of pressing contacts more than 30° from the link's own surface).
- **Decided: the design's generalized flexion, as a newborn has it.**
  - Human newborns' withdrawal is evoked from the whole limb at low thresholds, and its spatial tuning grows with age (Andrews and Fitzgerald 1994, Pain 56:95–101).
  - Young infants' nociceptive flexion reflexes are bilateral and non-specific (Cornelissen et al. 2013, PLoS ONE 8:e76470).
  - Newborn rats' withdrawals are functionally unadapted and often move the limb toward the stimulus; the adult, site-specific organisation emerges over the first three postnatal weeks (Holmberg and Schouenborg 1996, J Physiol 493:239–252).
  - That tuning is learned from the tactile feedback of the body's own movements, spontaneous twitches in sleep among them (Petersson et al. 2003, Nature 424:72–75; Waldenström et al. 2003, J Neurosci 23:7719–7725).
- `pain_site`, the local sign and its constant `RF_HALF` are gone; the frame carries pain alone.
- **C22 is written down as the newborn's,** not held to a bar only a tuned reflex meets. A learned tuning would be its biology's next stage; it is an open design item (C40), not built.

**A32. impratio 10, by the contacts' physics** (decided in the W1 verifier's third round; section 5.1)
- W1 set impratio 10 to stop MuJoCo's creep, and a verifier asked for its physical reason: the choice had been read against pain rates and grasps.
- **Decided on physics and MuJoCo's guidance, never on pain rates or on which toys are held:** the contacts here (rubber fingertips on plastic, toys and the body on foam, housings on housings) hold without a slide below their sliding force; MuJoCo's soft contacts creep by design, and its documentation names elliptic cones with a large impratio and the Newton solver as the remedy; Menagerie's hand models ship impratio 10 (5.1).
- Measured at 1 and 10: the creep, the slide's onset, pain with the filter, and the ten toys' holds (5.1, 3.8, section 6). **B1's premise follows from the physics:** only the bear and the drum cannot be held.
- **Its cost, disclosed:** a pressed housing that slides reads harder at 10 (5.1, C5).

**A33. What decides the eye check (C3)** (decided here, after the W1 verifier's fourth round)
- The identity score turned on the readout and on the views. Linear readouts reach only 0.52–0.64. A small nonlinear readout on the full code reached 0.840–0.860 at 67 training views a class. On the code through the core's projection it reached 0.794–0.836 there, but 0.74–0.78 at 44 training views, 0.65–0.72 at 22, and 0.711 in one setting (dusk, no shadow) at 40.
- **Decided:** C3 is judged by a small nonlinear readout (one hidden layer) on the code as the core receives it: the channel's born projection into the cortex's d and the input's LayerNorm. The cortex reads the eye channel through that projection into its learned nonlinear blocks, so a linear readout is only a lower bound, and the full code is more than the core receives.
- **At the tool's full training set,** 67 views a class (of 100), under each light with and without the shadow. C3 asks whether the fovea's code carries a toy's identity at 1.5 px a degree. The child sees each toy for thousands of ticks, so a readout starved of views measures the readout, not the eye.
- This was decided after the curve by views was measured. The curve is reported beside every verdict, so the choice can be judged; the bar (0.75) and the light are never changed for it.
- The tool projected into 256, and the sim's d is 512 (7.1), so the check runs again at 512 and under W5's lights (C3).

**A34. A new word on its pitch peak, on every ending** (the parent's method, ours; sections 4.4 and 4.8, C25)
- P1's first build claimed its new-word emphasis lengthened the word (+91%) and slowed the line. That held only on "." lines, where the engine spoke the "." aloud as "period"; on "?" and "!" lines the word came out 15% shorter.
- **Decided:** she introduces a new word as infant-directed speech does: in final position, on the line's pitch peak and lengthened (Fernald and Mazzie 1991; Albin and Echols 1996), on ".", "?" and "!" lines alike. She uses only the engine's own SSML: a prosody around the word and its punctuation, pitch +30% on the line's, rate 0.7 × the line's, in the new-word register at rate 0.15. Had the engine failed on an ending, she would not have used that ending for a new word; it manages all three.
- **Only in frames that put the new word on the peak.** Measured, the emphasized word is the line's highest peak on every line of the variation set and on "look. a X.", but not on "this is a X." (7 of the 8 toys), "this is your X." or "the X is up.", where an earlier word peaks higher. She does not introduce a new word in those frames, and the tool measures the peak by frame (P3).

**A35. The grasp summed at the spinal cord** (decided in the W1 verifier's first round; sections 3.6, 3.7 and A11)
- Built first as a core reflex hook, the grasp took the hand's tick. The hook is decided before the gate draws and sends the effector's rest, and the reflex read last tick's act, which was its own closing. With a ball in the palm it took 40 of 40 ticks, and the hand's gate drew 0 times: the child could never let go.
- **Decided: the spinal summation,** as in the cord, where a reflex and the descending command meet at the same motor neurons. The hand's gate draws every tick, and the reflex adds a small closing step to the hand's own act unless that act opens the hand. The hand's acts keep their eligibility, so letting go is learned (A11). The withdrawal keeps the hook, with no eligibility on its ticks.
- Measured with the ball in the palm: the gate drew on 24 of 40 ticks and passed 23 opening acts (test world 15).

A36–A57 were decided by the lead for the owner on the evening of 2026-09-24, on his bar ("robot need to be like we put human brain in g1 and sim is reality"; an architecture worth billions when all is complete), from the audits in `docs/audit/`. His standing rule is to decide rather than ask; the lead decided and reports, and the owner may overrule any of them. Each constant they add is ours, fixed before birth, and never tuned to a measured rate.

**A36. The sim body is the real robot's body** (the lead's decision 1; sections 0, 1, 3; the value audit's fifth demonstration and the skeptic's first strategic item)
- **Decided:** every sense, actuator and constant of the body is the real G1's, or published for it, so that seed 1 can go on in the real G1 as a change of world, as the scaffold's removal is a change of world (A29), and not as a new life under A20.
- **Why.** Under A20 a change of body after birth is a new seed. The G1 amendment's body differed from the robot's in its skin (B18), its colour stereo (B4), its servo gains (3.3) and its face reward (A1). Moving to the robot would then have been a new life, babbling on hardware: the breakage the owner built the sim to avoid, with only the code carried over.
- **What that asks,** each in its own entry: touch only where the real G1 has it, and contact from the joints' efforts (A37); the D435's own sensors (A38); Unitree's gains and the motors' and sensors' physics (A39); the parent reading only what a person would see (A40); the smile from the child's own pixels (A49).
- **The line between body and world.** The body is everything that runs on the robot: the core, the born codes and bank, the fovea, the observer, the reflexes, the cerebellum, the tract's synthesis, the weakness clip. The world is everything reality replaces: MuJoCo's physics, the models of the motors and the sensors (their noise, steps, drift and heat), the room, the parent, her voice, the scaffolds, lockstep. A sensor model is the world's stand-in for the sensor's physics, so on the robot the real sensor's physics replaces it.
- **What still differs, disclosed:** the microphones' and the speaker's places (B5), read from Unitree's documents before birth, or the move would be a change of body; the imagers' near-infrared, which the render lacks (3.4); every sensor model's constants until read from their sources (C44–C46).
- **What the robot still needs, each the owner's call:** the smile from pixels (A49); the critics' solves off the tick thread, each landing at a fixed tick so a replay stays exact; the tick timed on the robot's own computer; first days on the mat, eyes first.

**A37. Touch where the real G1 feels, and contact and pain from its joints** (B18, answered; the lead's decision 1a; sections 3.4, 3.7, 6, A12)
- **Decided:** the Dex3 hands keep their 16 touch zones, standing for the Dex3-1's tactile arrays: contact counts only on the faces where the arrays lie, saturating at their range (C44). The sim skin on the 29 other zones (28 links: the head's zone is on the torso's link) is removed from the body. Everywhere, contact is estimated from the joints' efforts by a born momentum observer (Haddadin et al. 2017).
- **The observer.** The residual between the generalized momentum the body shows and the momentum its motors and gravity account for is each joint's outside torque. It reads only the robot's own sensors: the encoders, the torques estimated from current (A39), and the pelvis's inertial unit, whose specific force and rotation stand for the free base, so the whole body's outside wrench is estimated too. Its model is the robot's own file. It runs every 5 physics steps, in the world's sensor code, as software the real G1 can run. Its gain is ours, settled before birth from the method's own analysis (C43).
- **Pain:** −1 on a tick when a joint's outside torque, as its tick's largest 10 ms mean, passes that joint's own torque limit, or the base's outside force passes F_pain (3 × the body's weight). A load from outside larger than a joint's motor can hold back-drives its gear: the robot's own damage line, taken from its declared model as F_pain was from its declared mass. The base has no motor, so F_pain's rule stays there. The Dex3's arrays carry touch, not pain: they saturate far below any damaging force, so a hand's pain comes from its joints. (The lead's text puts pain from the joints "elsewhere"; one law over every joint keeps the hands under it too, and is recorded here as ours.)
- **What follows:** pain is no longer world truth (3.4); the withdrawal fires on a limb's joint in pain (3.7); the event lines' touch groups become 7, the head merged with the trunk (7.4); a person's hold is felt through the joints, and every hold and guide is checked against the joints' pain law (4.2, C43).
- **Why not keep the skin.** B18's default kept it because pain is one of three rewards and an infant has skin. But the bar is that the sim is the robot; a skin the robot lacks would make the robot a new body. The joints' estimate is what the robot can feel, and a robot's collision detector is built this way (Haddadin et al. 2017).
- **Measured before birth** (C43): the observer's error against the world's true contact torques (an instrument) at rest and under babble; the pain rate on the born loop; the withdrawal's rate (C22 again); phantom pain on still ticks (C5 again).

**A38. The eyes are the D435's own sensors** (B4, answered; the lead's decision 1b; section 3.4)
- **What the sensors give** (the Intel RealSense D400 series datasheet): the stereo pair is two monochrome global-shutter imagers (OmniVision OV9282, 1,280 × 800), 50 mm apart, with a depth field of about 87° × 58°; they have no infrared-cut filter, so they see visible and near-infrared light. The colour camera is one rolling-shutter sensor (OmniVision OV2740, 1,920 × 1,080) beside the left imager, 69.4° × 42.5°. An infrared projector casts a dot pattern between them.
- **Decided:** the child's two eyes are the grey pair, each image the imager's visible response to the render. Colour comes only from the colour camera, rendered as a third view at its own place, and enters the body as colour-opponent cells over its own field and in a colour window at the left eye's gaze direction. So colour is central and one-sided, as on the robot. The projector is off: a pattern of laser dots in its eyes would be a lamp on its own head, refused as MuJoCo's headlight is (5.1).
- **The camera model** (the lead's decision; Foi et al. 2008): an exposure loop, Poisson–Gaussian noise, blur from the head's rotation over each exposure (from the physics), the colour camera's rows read over its readout time, and gamma, from the world's seeded stream. Its constants come from the sensors' published figures (C45).
- **Why.** B4's default kept colour in both eyes; on the robot that would need its colour camera or grey learning, a change of body. The real sensors are the body.
- **Its cost, disclosed:** the third view costs about 10–15 ms a tick (section 9). The roadmap's estimate of +2–7 ms did not count it. Cutting the colour image from a colour render of the left eye would save it, at about 15 mm of misplacement: a render shortcut, so it is offered to the owner under B3, never taken by us. (B3 was answered by the lead on 2026-09-25: not taken, since its parallax fails the render's rule where its hands reach, A65.)
- **Open:** the colour camera's exact place and axis, and the imagers' spectral response, from the datasheet (C45); recalled here as about 15 mm beside the left imager.

**A39. The motors and the inertial units as the real ones** (the lead's decision 1c–d; section 3.3)
- **Servo gains:** Unitree's published kp and kd for the G1 and the Dex3 (unitree_sdk2's examples, unitree_rl_gym's G1 configuration), read before birth; a joint with none takes the nearest published joint's gain per N·m of limit. They replace the first law's "limit at 0.25 rad", which was ours. Recalled: Unitree's leg gains are several times softer than ours were, so the sink rates, the pain rate, the grasps, the guides and the catch are measured again (C46).
- **The actuators and the gyro** (Hwangbo et al. 2019: the actuators are the main gap between a legged robot's simulation and reality; Woodman 2007 for a gyro's bias): each motor's torque–speed envelope, its torque sensed as estimated from current with that estimate's noise, its angle through the encoder's steps, and each gyro's bias walking. Each constant from Unitree's documents or the part's datasheet (C46); none fitted.
- **Motor temperature, as a sense only:** a first-order thermal model per motor; the temperature joins the body channel, as the real G1 reports it in each motor's state. It is never a reward and never a drive (a heat drive was proposed and not taken: A56). Firmware that weakens a hot motor is modelled only if Unitree documents it.
- **Why.** The owner's bar: the sim is reality. These are what the real motors and sensors do; leaving them ideal would teach the body a robot that does not exist.
- **What follows:** the VOR's born gain meets a drifting gyro, and the cerebellum's flocculus learns to cancel it (A44).

**A40. The parent reads its trunk and hands** (B14, answered; the lead's decision 1e; sections 3.1, 4.8, 4.10)
- **Decided:** she reads where the child looks as a person would read a robot that shows no eyes: the object nearest the line its head's camera faces, within the fovea's reach of that line, held 3 ticks; or the object its hand holds or closes on. She reads the line with a person's error, drawn from her seeded stream, its size from a source on judging another's head direction (C58). Her judgments of asks, her ledger, her follow-in naming and Claude's digest all read this, never the fovea's window.
- **Why.** A real G1 shows no fovea: a parent who reads one is a world the robot cannot have, so the robot would be a change of world she could not follow. The skeptic named the choice: a visible gaze (a change of body) or a parent who reads the trunk. The body is not changed.
- **What it costs, disclosed** (risk 19): a look made by the fovea alone is invisible to her, so asks are met less often, the ledger runs slower, and she names the wrong toy more often, as B14's default warned. The word pace's floor holds whatever the ledger does. To be understood, the child must turn its trunk or reach: what a real G1 must do.
- **The instruments** (section 12's rulers and tests, the page) still read the fovea; they are the experimenter's, and reach neither the body nor the parent.

**A41. The gates' drives, disclosed** (the lead's decision 2; sections 3.5, 6, 7.3, 4.9; amends "no intrinsic bonus")
- **Found:** the code's gates carried drives the design denied. Every gate's lesson adds, for each act, `gate_tonic` (0.25 by default) plus `gate_tonic_rate` × the felt reward's trace, and `gate_int` × an intrinsic term (`gate_int_form` "value" or "error"), minus the act's effort cost; `gate_vigor` adds a rate term. Section 7.3 said "no intrinsic bonus", 4.9 refused a songbird signal, and 3.5 said each gate is "born like the voice's" without these values.
- **Decided:**
  - **a tonic drive that follows the reward rate, in every gate** (Niv et al. 2007: tonic dopamine as the average reward rate, the opportunity cost of time, setting vigor): 0.25 + 4.66 × R̄ per act, R̄ the felt reward's mean at the 256-tick clock. The 0.25 is the core's and the served language body's born drive (babble is its own reward; the language body fell silent without one). The 4.66 is Σ_{k<12} 0.8^k, the gate's eligibility window, so the drive is the reward the rate brings over the span the credit sums, in the credit's units;
  - **the performance error, in the tract's gate only** (Gadagkar et al. 2016: a singing bird's dopamine neurons encode its performance against its own expectation): per articulator, the belief in the chosen setting minus that setting's running mean, averaged, at weight 0.5 (the language body's interest weight, "half its own confidence"). It compares the tract with its own past, never with the parent; the refused tutor match stays refused (A56);
  - **`gate_vigor` 0**, so the reward rate is not counted twice.
- **Why these and not a learning-progress reward:** they are what the code carries, disclosed and given a source; a learning-progress reward would be a new reward, against section 6 (A56).
- **What it does:** the drive rises under her smiles and falls where the body hurts; below a felt rate of about −0.054 a tick it turns negative and acting costs (risk 5). None of it reaches the reward, the critics, dopamine or the amygdala.
- **Open:** in the code the error is computed on effector 0's symbol, and later effectors carry 0; R6h computes it for the tract and gives the token output none (C61). The drive's value is reported daily (C47).

**A42. Newborn acuity: the fovea at native pixels through a born bank** (the lead's decision 2; section 3.4)
- **Found:** the fovea's code was the mean colour of 4-px cells at 1.5 px a degree, about 0.19 cycles a degree with no edges (the audit), below a newborn's ~1 cycle a degree (Dobson and Teller 1978). A lab would laugh at "human brain" over eyes coarser than a newborn's.
- **Decided:** the grey eyes render at 336 × 192 (3 px a degree at the centre); the fovea is a 64 × 64 window of native pixels; a born bank reads it: centre-surround ON and OFF cells, and oriented energy at 4 orientations and 2 scales (1.0 and 0.5 cycles a degree), pooled over the same 8 × 8 grid of cells, so an edge finer than a cell is kept as its energy. Orientation-selective cells are present in visually inexperienced kittens (Hubel and Wiesel 1963): the bank is born, as the cochlea is. It feeds the born face detector (C39).
- **Why no rising acuity:** an acuity schedule needs a mapping from infant months onto life days that no source gives, and a window that closes wrongly costs the only seed (A56).
- **Its cost:** +0.3–0.9 ms against the prototype's render, up to about +6 ms against the built world's (section 9); the bank under a millisecond. The eye check runs again on the new code (C3, C48).

**A43. Orienting to sudden visual change, habituating** (the lead's decision 2; reopens A23; section 3.7)
- **Decided:** a third born orienting cue beside the face and the sound: a sudden local change in the grey periphery, beyond the whole image's median change, suppressed while the trunk turns fast, habituating per place (Sokolov 1963: the orienting reflex habituates to a repeated stimulus and recovers for a new one). It is a bias on the gaze's and the waist yaw's proposals, with the amygdala's gain, exactly 1 at birth; it adds the event lines "a visual onset on the left / right".
- **Why.** Newborns orient to peripheral visual onsets through the subcortical route (Johnson 1990). A23's refusal ("the newest thing") cited no biology, and a sound onset turned the child where a visual one could not. It is the same kind of born bias as the kept face and sound orienting, not a saliency map and not a reward.
- **Its constants** (the change's threshold, the turn that suppresses it, the habituation's clock) are settled from their sources before birth (C49).

**A44. A scoped cerebellum** (the lead's decision 2; new step R6c; section 7.5)
- **Decided:** a cerebellum below the tick: a born granule expansion (Marr 1969; Albus 1971; an adaptive filter: Fujita 1982), Purkinje weights learned by least mean squares every 10 ms, taught by the servo law's own corrective torque (feedback-error learning: Kawato and Gomi 1992) and by retinal slip (the flocculus: Ito 1982). It learns load compensation and the VOR's gain, and an offset that cancels the gyro's drift. It never proposes an act. Since the servo's lagging target at rest is a correction too, it may learn part of a limb's own weight and slow a rested posture's sink: learned, disclosed, and written down (7.5, C50). Its sub-tick hook is added to the World interface in R6c.
- **Why now.** A loop below the tick added after birth would be a change of body, so a new seed (A20). The cerebellum grows about 240% in the first year (Knickmeyer et al. 2008).
- **Why not sitting** (the skeptic's finding): feedback-error learning needs an innate feedback controller for the task. The servo law corrects joint angles only, and righting and equilibrium reactions are refused, so this cerebellum can learn what the servo law and the eyes can teach, and not a balance. Risk 2 stands.
- **Its constants** (the expansion's size and sparsity, the learning rate) are settled from their sources before birth (C50).

**A45. Recall into action** (the lead's decision 2; R7f; section 7.6)
- **Decided:** a frame's key in the store is the stream plus a heading integrated from the gyro (McNaughton et al. 2006); its value is the next frame and the efference copies; recalled efference copies enter each effector's proposal through a map born at zero, learned by `act_pred`'s own lesson (Lengyel and Dayan 2007). The working-memory latch fires at R7's event ends, once R7b defines them for frames, in place of utterance ends.
- **Why.** Recall reached only the words, never a limb. Deferred imitation and object permanence need what was done to come back into what is done. Born at zero, it changes nothing until it predicts.
- **Disclosed:** the heading drifts with a real gyro, and the cerebellum's offset corrects the VOR, not the heading (C51).

**A46. REM twitches in a live, dark night** (the lead's decision 2; R8; sections 3.7, 5.4, 6; the owner's ruling that REM stays on)
- **Decided:** the world is not frozen at night. It runs dark for 24,000 ticks, the parent asleep, the child's gates closed, its eyes and ears off. In REM's phases (about half of a newborn's sleep: Roffwarg, Muzio and Dement 1966) the brainstem makes twitches, one joint and one small step at a time, at about 10 a minute (Sokoloff et al. 2020, read exactly before birth). Each twitch and its reafference teach `act_inv`, the forward half and the cerebellum. The core's night passes run as before; REM stays on.
- **Why.** Twitches in active sleep give clean single-joint pairs of act and consequence (Blumberg, Marques and Iida 2013), and `act_inv`'s reliability gates every guide; the withdrawal's learned tuning (C40) grows from such feedback too (Petersson et al. 2003).
- **Ours, disclosed:** the night's length (as long as the day, B19's compressed hour); the twitch's step (0.09 rad); the charge's basal drain stops at night, since a night of 24,000 ticks at 4e-5 would drain 0.96 of a full charge where the paused night drained nothing (5.4); nothing sensed at night reaches a reward, a critic, the amygdala or a gate. The night's cost and pain are measured at S5b (C52).
- **It changes** 5.4's night as a pause, A17, risk 17 and the World interface's night (R8).

**A47. A born cry** (the lead's decision 2; reverses the G1 amendment's refusal; sections 3.7, 4.9, A13)
- **Decided:** on a pain tick, or while the charge is below 0.2, the tract's born cry posture is added to its targets, and the tract's own act overrides it. Its ticks are logged as reflex; a cry is never a vocal turn, and no smile answers it. The parent hears it as distress.
- **Why.** The cry is innate and patterned in the brainstem (Jürgens 2002); a newborn cries before it learns anything. The refusal said "the tract can learn to call", but the first thing a newborn's voice does is cry. It costs nothing a tick.
- **Its pattern and its charge line are ours,** from its sources before birth (C53); the line sits below the parent's feeding line so the cry is the body's alarm, never timed to her.

**A48. A spinal pattern generator per limb** (the lead's decision 2; reverses the stepping refusal; sections 3.6, 3.7; tied to C38)
- **Decided:** a half-centre oscillator per limb (Brown 1911), summed at the cord with the limb's own act like the grasp (A35), the legs in antiphase, its amplitude the limb's gate's readiness. It has no posture, balance or gravity term.
- **Why.** The stepping refusal rested on "kicking, which babble already gives". It fails by construction, not only by measurement (the skeptic): Thelen's argument assumes a kicking generator, and babble is gated draws of about 1.4 ticks, close to the fresh draws that gave 0 rolls (C38). Per-muscle oscillators gave a simulated neonate its motor patterns (Kuniyoshi and Sangawa 2006).
- **Its period, amplitude and the arms' coupling** are read from their sources with C38's unit lengths, before birth, never on a roll count (C54).

**A49. The smile read from the child's own pixels** (the lead's decision 3; sections 3.4, 3.7, 6, A1)
- **Decided:** once the born face detector works (C39), the reward's carrier is the child's own perception of her face: a born mouth-corner reader on the fovea's centre-surround map, inside the face the detector found, read to the born reading's −2..+2 (newborns tell happy, sad and surprised faces apart up close: Field et al. 1982). The ray test and the world's value leave the body.
- **Until then, a disclosed scaffold.** The reader is born in every case. The face term takes the world's value while the world supplies it and the reader's reading when the world falls silent: a rule fixed from birth, so the silence is a change of the world, as A29's is. If the detector works before birth, the world never supplies it.
- **The removal test** (on a copy, at a night boundary): on a copy day, for judged faces seen within the lean-in distance, the reader's sign agrees with the world's on at least 90%, and it reads a smile of +1 or more on at most 1% of ticks where her face is neutral or absent. Fixed before birth (C55); if it fails, it is tried again every 5 life days, as the word scaffold is.
- **Disclosed:** from pixels, a smile beyond about 1 m is not felt (4.3's table): fewer smiles felt from afar, the real condition, never a reason to keep the scaffold.
- **Why.** On the real G1 nothing reads a person's smile from world truth: without this the reward has no carrier on the robot (the reality audit's second item, a transfer blocker).

**A50. Eight voice variants per line** (the lead's decision 4; section 4.4)
- **Decided:** every line is made in 8 variants, its pitch within ±8% by the engine's prosody and its spectrum warped by a vocal-tract-length factor within 0.9–1.1 (Jaitly and Hinton 2013), the variant drawn per utterance; each deterministic and held to its digest.
- **Why.** One waveform per line lets a word be learned as a waveform; infants learn words across talkers' variation (Rost and McMurray 2009), and the measured bottleneck of imitation is hearing across voices (4.9).
- **Measured before birth** (C56): the parent's ear on the variants, and the new word on its pitch peak in every variant (A34).

**A51. The gaze leak closed** (the lead's decision 4; sections 4.3, 4.10, A28)
- **Decided:** her eyes go to the object only on the naming word of a label, a show or a confirm. In asks and probes her head and eyes stay on the child, she does not point, and she does not turn until the ask is judged. The one exception is M1's gaze-following probe, whose stimulus is her silent head turn to a toy, with no word and no point (section 12).
- **Why.** Her eyes went to the object on every naming word, "where is the ball?" included, so a child could pass by following her gaze: Clever Hans. Preferential-looking studies blind the parent for this reason (Golinkoff et al. 1987).

**A52. An imperfect parent, who copies its movements** (the lead's decision 4; the parent's method, ours; section 4.10)
- **Decided:** she misses a share of its acts, her reply's latency is jittered, and she has spells of distraction, each at rates from human dyads (Tronick and Gianino 1986: interactions are coordinated only part of the time, and mismatch and repair are how the infant learns; Bahrick and Watson 1985: 5-month-olds turn from the perfectly contingent view of their own legs, read as a preference for the imperfect contingency of people). She copies its arm and hand movements, mirrored, within 1–2 s, as parents imitate their infants (Ray and Heyes 2011). All from her seeded streams, fixed before birth, never fitted to the child's rates (C57).
- **What it changes:** P6's contingency ruler becomes her declared rate, not 90% (C24); a judged act still gets its smile within a tick.

**A53. A richer room** (the lead's decision 4, for the owner, whose room it is; sections 5.1, 5.2)
- **Decided:** textures on the toys, the furniture and her clothes; containers (a hollow cup, open baskets); a cover a Dex3 hand can lift; at least 3 examples of every tested noun (Quinn, Eimas and Rosenkrantz 1993); one new object every 3 life days from an inventory fixed before birth, by a calendar that never follows the child's progress.
- **Why.** Identity was mostly colour, "in" could not happen, nothing could be hidden and found, and a noun heard on one object may mean that object.
- **Measured before each object joins** (C60): its holds, its sounds and its cost.

**A54. Hull pain measured first** (the lead's decision 4; sections 3.4, 6, A21)
- **Decided:** W4 measures pain on the born loop under the joints' law, and the share of it from the collision hulls meeting rather than housings (contacts over the line with every hinge more than 0.2 rad from its range's ends). If the hulls carry it, a convex decomposition of the G1's own collision meshes (Wei et al. 2022) is loaded world-side, and pain is measured again. The G1's file stays byte-identical. It is decided before birth.
- **Why.** The skeptic found the artifact overstated: 35% of the compound-joint contacts over F_pain were far from the ranges' ends, the rest may be housings meeting, and the run used the per-joint instrument, not the born loop. Measure before claiming. Pain from a modelling artifact would teach "hip roll hurts" through a sense the robot lacks.
- **Its cost:** est. +1–3 ms of physics, only if needed (C59).

**A55. The tests of understanding by milestone** (the lead's decision 5; section 12, A19, A28)
- **Decided:** the roadmap's tests (section 12's second table) join the never-taught tests, with their chance levels, fixed before C36: self against other in touch, the senses linked, surprise at a broken expectation, following her head turn, the occluded ball, social referencing, cause and effect with the rattle (its own first holds as chance), search under the cover, looking ahead of her reach, new exemplars of a noun, the colour twins as targets with each colour word on at least 2 kinds, and "more" for its own need. One-sided p < 0.01, fresh items, the word scaffold silenced for word claims; no claim on the ledger's window.
- **Why.** Infants show understanding by where they look and what they do unpaid; one test per milestone did not reach curiosity, cause and effect, permanence, intention, categories or self against other.

**A56. Dropped, and why** (the lead's decision 6)
- **A learning-progress reward:** it would be a new reward, against section 6's law, and the biology cited for it does not show one: Redgrave and Gurney (2006) is short-latency dopamine to unexpected salient events, for discovering agency; Bromberg-Martin and Hikosaka (2009) is a preference for advance information about reward; Oudeyer et al. (2007) is a robot algorithm. The disclosed drive (A41) takes its place.
- **Noradrenaline, acetylcholine and serotonin controllers** (Doya 2002's map): three meta-controllers with no learning rule and no sourced constants: names without rules.
- **Thalamic gain:** no learning signal was named for it.
- **A breath rhythm:** breathing is silent and the tract already has its breath reservoir; no function.
- **An auditory-target reward:** it is tutor matching, which 4.9 refuses; and the measured bottleneck is hearing across voices (rank 22 of 50 against a chance of 25.5), which the variants address (A50).
- **An acuity schedule and plasticity windows in year 1:** mapping infant months onto life days needs an unsourced constant, a window that closes wrongly costs the only seed, and the evidence for starting blurred is contested (Vogelsang et al. 2018).
- **A balance law:** the laws refuse it, and the cerebellum's teacher cannot teach one (A44).
- **A heat drive:** motor temperature is a sense only (A39).

**A57. The build plan and the tick, re-estimated** (the lead's decision 7; sections 0, 9, 11)
- **The steps:** R6c (1 day); R6h +0.75 (the drives, the pattern generator, the cry); R7 +1 (the visual onset, recall into action, the latch); R8 +1 (the live night and its twitches); W1 and W3 reopened; W4's hull share; W5's live night and W5b's richer room; P1b's variants; P3's leak, imperfection, copying and reading; P4's tests; the pixel reader. Core work rises from 9.5 to about 13.25 days, the critical path. Birth moves from about day 13 to about day 17 (range 15–22) with three sessions, about 24 with two, about 41 with one.
- **The tick:** about 2–15 ms for the additions (the roadmap's +2–7 ms was their lower half) and about 10–15 ms for the colour camera's own view: about 117–190 ms with the sun's shadow. Its upper end is past 150 ms, so the birth checklist's rule stands: at most 150 ms heat-soaked, or the owner chooses between B3's render choices and a slower life. (Amended by A65 on 2026-09-25: B3 is answered, and the tick is measured, with no 150 ms bar.)

**A58. The human-pace ledger: infant norms in waking hours, a ruler never a target** (decided for the owner on his bar, "humanoid with our human architecture going in sim training; we need it at learning speed and understanding level of human", as A36–A57 were; section 12; the study in `$S/pace/`, reviewed by a skeptic)
- **Decided:** each milestone and each never-taught test is reported against an infant's waking hours to the same capacity (section 12's human-pace ledger).
  - The robot's hours are its waking ticks to the milestone × 0.15 s.
  - The infant's hours run from birth to the norm's age, from Galland et al. 2012's mean sleep by age, with its sleep limits as the range.
  - The headline is the waking-hour age A_r, placed in the milestone's own age distribution. The ratio S = H_i / H_r is secondary, and is never printed for M1.
  - Beside each row go the exposures per waking hour on both sides (words, a face in view, contingent replies, sleep).
  - A second table counts exposures to criterion, the honest measure of learning speed.
  - "Understanding level" is answered only by which never-taught tests pass, and at what A_r.
- **The firewall** (section 12):
  - frozen before birth with a digest, never re-sourced after it;
  - every row reported, passes and failures alike;
  - read by no brief, prompt, test, build step or constant;
  - the parent's word rate and timings never moved toward a norm.
  - A low reading can lead only to a change of the learning machinery on general grounds, and after birth that is a new seed (A20). It never leads to a change of the parent, the room, the pace, or a constant chosen to move a row.
- **Why:**
  - The bar asks for a human's learning speed and understanding. Wall time depends on the machine, and a life day is the body's own unit. Waking hours match what both sides live awake, with sleep excluded on both: the robot sleeps 1.0 hour per waking hour, an infant 1.55 at 0–2 months and 1.16 at 12.
  - The laws allow published norms as a reference and forbid a baseline run, so the norms are a ruler: never run, never targeted (section 1).
  - It maps no month onto a life day inside the body, so A56's refusal of an acuity schedule stands.
- **Dropped in review:**
  - a column of what S would read if section 12's forecasts held, which turned guesses into headline multiples of 33–167× before any life;
  - what S = 1 costs in wall days (it stays in section 9 as life days, with no milestone attached);
  - a single S for M1;
  - a words ratio of 1.5–2.4×, since LENA's count includes overheard speech and naps;
  - a deferred-imitation row and crawling, which no ruler of the body's reads.
- **Found by it:** the language body lived 197.5 waking hours, not 404: about a three-week-old's (21 days; 14–48 under the sleep limits).
- **Open:** the recalled rows are read at their sources before the freeze (C62).
- **Amended by A66** (2026-09-25): an optional report the owner does not need, never a headline. Its firewall stands in full, and C62 is done only if it is kept.

**A59. The speed plan: exact levers, the steering keyed to ticks, and the refusals** (decided for the owner in the same way; section 9, B21)
- **Decided:**
  - **The exact levers of section 9,** each under its three rules: joins at fixed ticks that block if the work is late, the BLAS thread count pinned per operation, and each lever kept only if the eight digests, the `sim` profile and SimWorld's replay are unchanged. They are built in the core session's free days 15–16.
  - **The steering keyed to ticks.** The digest is written every 400 ticks, and each row is applied at the digest's tick + 400. The world waits for a late row; a failed row leaves the fast layer alone, logged. This amends 4.5, 4.10 and A14: "a minute" there is now a window of 400 ticks, 60 s of sim time.
  - **The refusals:** reduced precision (float16, bfloat16, TF32), a longer step or a looser solver, GPU physics, parallel lives, and the lesson lag and the late solve, which would change the body for at most about 20% more speed.
  - **Off this Mac only under section 9's conditions,** with the machine chosen before birth and birth on it. The spending is the owner's (B21). **Closed on 2026-09-25** by his word: no machine but this Mac, ever (A61). The levers' build steps, the machine's share and the steering's lag are A64's, and B3 is answered by A65.
- **Why:**
  - Lockstep makes speed a matter of wall time only, so nothing the body senses or does may change for it (the laws: one seed, exact determinism, nothing fitted to the environment's pace).
  - Steering by wall minute let the machine's speed and heat set how dense the teaching was, which tied the environment's pace to the hardware. A row keyed to ticks is the same at any speed, and replays exactly.
- **What it costs:** a late row costs wall time. Claude's round trip caps the pace at 60/R × real time (R in seconds), measured at P5.

A61–A66 were decided by the lead for the owner on the morning of 2026-09-25, on his word "only local no cloud. i dont care about huuman pace i just want fast learninhg and for it to solve robotics that no archecture had done before", from the firsts research and its skeptic's review (`$S/firsts/`). His standing rule is to decide rather than ask; the lead decided and reports, and the owner may overrule any of them. B21 is answered by his own word.

**A61. Local only: no cloud, ever** (the owner's word; B21 answered; sections 0, 1, 9)
- **Decided (his):** seed 1 is born and lives on this Mac. No cloud or rented machine, no paid test and no spending, ever. C63 is closed, and section 9's rented-machine cases, their conditions and their prices are dropped; they stay in git history (7bdc141).
- **The one input from off this Mac, read by the lead:** Claude's steering row every 400 ticks (4.5, A14, A59). It carries the owner's word of 2026-09-17 for a teacher ("we need a human teacher; you have to get Opus to become that", said of the language body's parent), and it is served on his plan, never the metered API, so it costs no money. Every row is logged by tick, so the life replays on this Mac with no network. The lead reads "no cloud" as no cloud machine and no spending, and keeps the steering.
  - If the owner means no call off this Mac at all, the fast layer runs alone (4.5 allows it), its choices of toys, episodes, tasks and new words made by P4's day plan, and nothing in the body changes.
- **Disclosed beside every first** (A63): the steering chooses her curriculum (the focus toys, the episodes, the scaffolding tasks, the day's new word), and never a feeling, a judgment, a smile, a help level or a force (A14).
- **What it costs:** calendar days only. In lockstep the machine sets wall time and nothing the body senses or does (A59).

**A62. Fast learning, defined** (the owner's word, "i just want fast learninhg"; sections 0, 1, 9, 12)
- **Decided:** fast learning is two things, reported apart.
  1. **Sample efficiency, the headline:** the experience the body needs, in its own units.
     - Its life hours to each milestone's criterion: the waking ticks from birth to the first of A19's two days, × 0.15 s. A life day's wake is one hour.
     - Its exposures to criterion, by kind: the smiles it felt and the asks it met before criterion; her guides, holds, turns and placements (the intervention log, A63); the namings of a word before its test of understanding passes (a variation set counts once, 4.5); the holds before the rattle's test passes.
     - The overnight gain: each skill's ruler in a day's first floor-play block against the day before's last, with nothing practised between but the night's replay and twitches.
  2. **The fastest honest local run:** the wall time a life hour takes on this Mac, from the exact levers (A64), the render as decided before birth (A65) and the machine's share, with nothing changed that the body senses or does.
- **Context, never a run and never a target:** the closest methods' published sample counts, each with the signal it learned from (section 12's firsts). No multiple ("N times faster than X") is printed, since each of those methods learned one skill from a signal built for it.
- **What can move it:** only architecture decided before birth: credit across the delay, the night's replay per waking hour, recall into action. Never the parent's pace, a looser ruler or an easier world. A faster wall clock brings the firsts' life hours sooner on the calendar, and changes none of them.
- **Why:** the body's hours do not depend on the machine, and they are the units other methods' sample counts are counted in. Wall time does depend on the machine, so it is reported beside them, never as learning.

**A63. The firsts, registered before birth** (the owner's word, "for it to solve robotics that no archecture had done before"; section 12)
- **Decided:** two firsts are registered (section 12).
  - **First 1:** one life, many skills, sparse social feedback, the stock G1's whole body, in a full claim (M1–M4) and a lesser one (M1–M3), at two levels, with a long-run clause and its cost reported.
  - **First 2:** understanding shown by a hard set of four tests it was never taught, claimed at 3 of 4.
- **The three safeguards the laws allow** (one seed and no comparison leave only "it can be done" claims):
  - each claim's wording, bar, rulers, items and chance levels frozen with a digest before birth;
  - each test's chance measured on seed 1's born state at every learning rate 0 (S5b); a test it passes at p < 0.01 leaves the claim;
  - the exact replay released: the commit, seed 1's born state and the logs, so anyone can replay the life to any tick.
- **The intervention log:** every act of hers that moves its body or its world on its behalf (guides, holds, turns onto its back, the pull-to-sit, props, catches, toys placed on the ladders, tidies, the morning basket), counted per life day and printed beside each first. "No reset" means no episodic reset: the world never restarts, and her interventions are counted, not hidden.
- **Dropped as firsts** (section 12 gives each reason): speed alone (the research's F2, now First 1's reported cost); a face as the reward on its own (F3, now First 1's second level); lifelong learning on its own (F5, now its long-run clause); grounded compositional words (F6); one core, two bodies (F7); a first word from its own tract (F8). Seed 1 on the real G1 (F9) is the endgame and the owner's call (section 16).
- **Why:** a first holds only if it is worded for what the life can show. A scripted parent whose smiles follow a worth table and whose hands guide its limbs is not "a person as the only reward". A test that a reflex or the efference copy passes by construction is not understanding. Speed against methods that each learned one skill from a signal built for it is not a like-for-like figure.
- **No claim leaves the project** before the life has passed its registered bar and a skeptic has read the claim against the log (risk 25).

**A64. The local speed levers, with their build steps** (the owner's word; amends A59's schedule; sections 9, 11)
- **Decided:** all five exact levers of section 9, each built by the session that owns its code:
  - **R10** (core, day 15): the critics' rank-one updates on a second performance core, joined before each solve and each save (14–26 ms);
  - **P7** (parent, day 13): her ear off the tick, joined at her reply tick (about 5 ms);
  - **W7** (world, day 13): the ears, the tract, touch and the observer computed while the GPU draws (4–8 ms), and the world's per-step Python compiled, bit-equal to the Python it replaces (2–4 ms);
  - **R8** (core, day 12): the live night's physics and the replay on separate cores, merged in a fixed order (the night 12–25 wall minutes, not 18–32).
- **Under the digest rules** (A59): every join at a fixed sim tick, blocking if the work is late; the BLAS thread count pinned per operation; each lever kept only if the eight language digests, the `sim` profile and SimWorld's exact replay are unchanged.
- **Birth waits for none.** An exact lever leaves the body bit-identical, so one not ready by S5b joins at a night boundary after birth, once its three rules hold on a copy. It is not a change of body (A20).
- **The machine's share.** The largest local factor measured is other work on this Mac: at a load of 10–15 the eyes' render ran about 4× slower (section 9). While the life runs, it has the Mac's first claim after the owner's own use: other sessions' sims, renders and verification copies run at `nice -n 19`, one at a time; the room camera renders only while someone watches; and the load is logged each life day (C72).
- **The steering's lag.** Claude's round trip caps the pace at 60/R × real time (A59). If P5 measures a round trip longer than a window's wall time at the pace the levers reach (26 s at 2.3×), each row is applied two windows after its digest (the digest's tick + 800), fixed before birth (C71). It is a constant of the parent's method: it changes when a row lands, never what the body senses or does, and it is never set by the child's rates.
- **Not reopened:** reduced precision, a longer step or a looser solver, GPU physics, parallel lives, and the lesson lag or the late solve (A59). Fast learning is counted in life hours (A62): a faster wall clock bought by changing the body buys nothing the bar counts, and risks the only seed.

**A65. B3 answered: the lighter meshes for its eyes; the colour camera's own view and the sun's shadow kept** (B3, answered by the lead for the owner; sections 3.4, 5.1, 9, 10)
- **The rule:** a render shortcut is taken only if what the body senses stays the same within its own sensor's noise and a pixel of its native image. The render is the world's, which reality replaces on the robot (A36's line); this keeps the owner's complete reality (decision 3) at the eyes' own resolution, and every sensor where the robot has it.
- **(a) Yes: a lighter copy of its own visual meshes, for its eyes only.**
  - Each of the G1's visual meshes is decimated (quadric edge collapse: Garland and Heckbert 1997) and loaded world-side for the three views alone. The G1's file stays byte-identical, and the physics, the collision geoms, the room camera and the films keep the full meshes.
  - The coarsest copy that passes the rule, at the poses where its own body is nearest its eyes (a hand before its face, an arm across its view), is used. It is measured at W3's reopening and set before birth, never after (C69).
  - It saves about 8–18 ms over the two grey eyes, since half their render was its own meshes (hiding them, a test only: section 9), and about half as much again in the colour view: about 12–27 ms in all, an estimate.
- **(c) No: the colour camera keeps its own view at its own place.** The evidence:
  - Its RGB sensor's centre-line is 15 mm from the left imager's (RealSense's documentation, read 2026-09-25; the datasheet is still read at C45).
  - Cut from the left eye's render, its colour would sit exactly on the left eye's grey. On the robot it is displaced by the parallax: 2.9° at 0.3 m and 5.7° at 0.15 m, where its hands and the toys it reaches for are. That is 1.1 and 2.2 of the colour window's 2.6° cells, or 9–17 of the fovea's pixels: far past the rule.
  - By A36's line, a sensor seen from another place makes the real G1 a change of body, as B5 says of the microphones.
  - Reprojecting the left eye's colour through its depth to the camera's place would keep the parallax, but not the band a near hand uncovers against the room (about 2.4° beside a hand at 0.3 m and 3.9° at 0.2 m, with a wall 2 m behind), so it fails too.
  - Its saving, about 10–15 ms (an estimate), is smaller after (a), which lightens the colour view's own draw.
- **(b) No: the sun's shadow stays** (the owner's decision 3). A shadow changes the light far past the rule.
- **What it changes:** with the exact levers and (a) the tick is about 65–135 ms, 1.1–2.3× real time (section 9's case C, an estimate that C12 and C48 measure). No 150 ms bar binds a lockstep life, since a slower tick costs wall time only, so the birth checklist's tick rule becomes a measurement.
- **Why the lead answers it:** the design asked B3 only if the tick passed 150 ms, a bar the owner's word removes. Answered by this rule, the answer changes nothing the body could sense beyond its own noise, and keeps every sensor where the robot has it.

**A66. The human-pace ledger, an optional report** (the owner's word, "i dont care about huuman pace"; amends A58; sections 0, 1, 12)
- **Decided:** the ledger stays in section 12 as an optional report the owner does not need. It is never a headline, and nothing waits for it: birth does not, and C62's reading of its recalled rows is done only if it is kept.
- **The firewall stands in full,** printed or not. If it is reported at all, it is the ledger frozen with its digest before birth, every row, passes and failures alike. Nothing in the body, the parent, the brief, the tests or the build reads it, and the parent's pace is never moved toward a norm. A ledger not frozen before birth is never reported.
- **In the headlines instead:** fast learning in the body's own units (A62), and the firsts (A63).
- **Kept from it:** the body's own exposures to criterion (its second table's right column) join A62's measure; the infant column is optional with the rest.
- **Why:** the owner's word. A ruler he does not want should not shape the reports, and the firewall costs nothing to keep.

### (B) The owner's calls, with recommended defaults

The nine decisions of the G1 amendment are settled (section 14). These are the environment's remaining shapes, each with the default the design uses until the owner says otherwise. B4, B14 and B18 were answered by the lead for the owner on 2026-09-24 (A37, A38, A40); he may overrule each. On 2026-09-25 B21 was answered by the owner's own word (A61), and B3 by the lead for him (A65).

| # | question | recommended default | what it changes |
|---|---|---|---|
| B1 | At first the G1's hand could not hold four toys (the ball, the duck, the bear, the drum), so we made the cup smaller (0.8 of its size, 6.7 cm across) to let the hand hold it. That was the simulator's contacts creeping. With the contact setting chosen by physics (impratio 10, A32), the hand holds the ball, the duck and the full-size cup; only the bear (2 of 9 tries) and the drum (0 of 9) cannot be held. May we put the cup back to its own size, and leave the bear and the drum as they are? | Yes to both. The cup's shrink fitted the world to the body, and now has no reason; the bear and the drum are for looking at, pushing, hitting and naming. Until you say, the cup stays at 0.8. | Eight toys are holdable either way; the full-size cup held 9 of 9. |
| B2 | May we add four colour twins (a blue ball, a red block, a yellow cup, a green car), so we can test "the red ball" when it has never heard those words together? | Yes. | Four more toys; colour words come in once two toys share a colour (A28). |
| B3 | If the tick is too slow, may its eyes (a) see a simpler copy of its own body, (b) see the room without the sun's shadow, or (c) take the colour camera's picture from the left eye's view, about 15 mm from where the real colour camera sits? You would not see (a) or (c); (b) takes a piece of the complete reality you asked for. | **Answered by the lead for the owner (A65):** (a) yes, a lighter copy of its own meshes for its eyes only, the coarsest whose images stay within its sensors' noise and a pixel where its body is nearest its eyes (C69); (b) no, the sun's shadow stays (decision 3); (c) no, on the evidence: the RGB sensor sits 15 mm from the left imager's centre-line (RealSense's documentation), so colour cut from the left eye's render would miss the robot's parallax by 2.9° at 0.3 m and 5.7° at 0.15 m, 1.1–2.2 of the colour window's cells where its hands reach, and the real G1 would be a change of body (A36). (The first default was neither unless the heat-soaked tick passed 150 ms; then (a) first, then (c).) | (a) saves about 8–18 ms over the two grey eyes and about half as much again in the colour view: about 12–27 ms a tick (an estimate); with the exact levers the tick is about 65–135 ms, 1.1–2.3× (section 9's case C). (b) would have saved about 21 ms and (c) about 10–15. No 150 ms bar binds a lockstep life: a slower tick costs wall time only. |
| B4 | The real G1's two eye cameras see in grey; ours see in colour. Keep colour? | **Answered by the lead for the owner (A38):** no. Its two eyes are the D435's grey imagers, and its colour comes from the D435's own colour camera beside the left one, each with its noise. (The first default was yes.) | The robot's eyes are the sim's, so the move to the robot is a change of world. The colour camera costs a third view (B3). |
| B5 | The model does not say where the G1's microphones and speaker are. Put its ears on the sides of its head and its voice at the front? | Yes, until Unitree's documents say where they are; they are read before birth, since a different place on the robot would make the move a change of body (A36). | The ears' timing and loudness. |
| B6 | Keep the child's voice as built: a child-sized throat, pitch about 265 Hz? | Yes. | What you hear it say. |
| B7 | Nobody can carry the 34 kg child back, so the parent goes to it wherever it is. Keep the whole floor and the open hall as its play space, with no baby gate? | Yes. | Where it can end up. |
| B8 | Close the gap under the sofa, take away the low table's pretend shelf (the lying G1's chest is 1 cm under it), and put lost toys back in their basket each morning? | Yes to all three. | Two lines in the maker; one morning rule in the world. |
| B9 | Should the room echo? | Yes, if it costs under 1 ms a tick; otherwise no, and we report it. | The ears hear the room. Measured since (P2): each sound source costs about 0.37–0.4 ms a tick through the ears' spatializer, so six first-order echoes (the walls, the floor and the ceiling as image sources) would cost about 2.2–2.4 ms. Unless W5 finds a cheaper echo, the default gives none, reported. |
| B10 | Keep charging as a bottle it holds in its hand, about 5–6 times a day while it moves a lot? | Yes. | Charging by mouth would need a mouth the G1 does not have. |
| B11 | Keep the mat at 2.8 × 2.0 m? | Yes. | — |
| B12 | Your decision 1 lists "looking at it" among what raises her happiness, and also says smiles come only for completed, visible acts. Should the child just looking at her ever make her smile? | No. Looking raises her attention (her eyes and a greeting flash, which the child cannot feel as reward); only its acts earn a smile. A look she asked for (a call answered, "look at the drum") is an act and does earn one. | Otherwise looking would pay. |
| B13 | Keep the names "pip" and "mama", the Samantha voice, and no second adult? | Yes. A better voice is a download you would install yourself. | — |
| B14 | Should she tell where the child is looking from its fovea, as people read a baby's eyes, though a real G1 shows no eyes? (Claude's minute digest reads the same window.) | **Answered by the lead for the owner (A40):** no. She reads its trunk and hands with a person's error, as she could on the real robot, and so does the digest. (The first default was yes.) | She names the wrong toy more often, and asks are met less often: a look must show in its trunk or hands (risk 19). |
| B15 | Keep the living room at 5.2 × 4.6 m? For the 1.32 m G1 that is like a 2.9 × 2.6 m room for a baby. | Yes. More room comes with the house (section 16). | How often it meets a wall or furniture (A24). |
| B16 | Keep the parent as strong as a real person, so she can never lift it, slide it or sit it up? | Yes. It must rise by itself, even if sitting comes late. | Stronger, she could carry it through postures it has not reached (A25). |
| B17 | If the child's own hearing and voice never pass the tests for dropping the word tokens, keep the tokens? | Yes. They stay and are tested again every 5 life days (A29). | With a fixed date instead, it may lose words it knows. |
| B18 | The real G1 feels touch only in its hands. Should our G1 feel touch (and pain) on its whole body, like a baby's skin, though the robot has no skin there? | **Answered by the lead for the owner (A37):** no. Touch is felt in the Dex3 hands, as on the robot, and contact and pain elsewhere are estimated from the joints' efforts, as a robot's collision detector does (Haddadin et al. 2017). (The first default was yes, with an added skin or the joints' estimate named as the robot's options.) | Pain stays one of its three rewards, read from what the robot can feel; a blow is felt through the joints it loads. The old skin's rates (6–12% of babble ticks, mostly its own housings) are measured again under the joints' law, with the share from hulls meeting (C43, A54). |
| B19 | A life day is one simulated hour, so the sun crosses the window in an hour, not a day. Keep that? | Yes. A real-length day would make each life day 24 times longer in wall time. | How fast the light changes. |
| B20 | The real G1's motors hum as they work, and its microphones hear it. Add that sound, growing with each motor's effort? | Yes, if it costs under 1 ms a tick; otherwise no, and we report it. | It hears its own effort; nothing rewards it. |
| B21 | On this Mac the life runs at about 0.8–1.3× real time (on a quiet, cool machine; sustained speed is unmeasured). So an infant's first year of waking hours, about 3,925 life days, would take about 177–300 days of running around the clock, or 133–228 with our exact levers. A rented Linux machine with a graphics card might run it at 2.2–4.7× (49–106 days), but that is an estimate, unmeasured. May we spend on (1) a one-hour test on such a machine, about $1–2, and then, if it reaches 2.5× or better, (2) renting it for the life? | **Answered by the owner's own word, 2026-09-25 (A61):** "only local no cloud". No cloud or rented machine, no paid test and no spending, ever: seed 1 is born and lives on this Mac. (The first default was no spending now, and a paid hour only on his yes.) Claude's steering stays, served on his plan, never the metered API; if he means no call off this Mac at all, the fast layer runs alone (4.5). | Only how many calendar days the life takes: nothing the body senses or does changes (A59). On this Mac, with the exact levers and the lighter meshes for its eyes, 60 life days take about 1.6–3.3 calendar days of running around the clock and 200 about 5–11 (section 9). The rented machines' estimates and prices stay in git history (7bdc141). C63 is closed. |
### (C) Open until measured

| # | open question | measured in | what decides it |
|---|---|---|---|
| C1 | Which child | — | settled: the stock G1 (the owner's decision 7) |
| C2 | Whether the face test's rays agree with a segmentation render inside the software fovea | W3: 2,000 babbled frames, the parent at 0.3–3.5 m | at least 95% agreement; below that, the render is used. Measured: 99.8% of the 975 frames whose geometry passes, her head kept inside the room (98.7–100% in each distance bin); the rays stay |
| C3 | Whether the fovea tells the toys and the face apart at 1.5 px a degree, under morning, midday and dusk light | W3's eye check (`tools/sim_eye_check.py`), again under W5's lights | identity at least 0.75 by A33's readout (a small nonlinear readout on the code through the channel's born projection into the core's d, at 67 training views a class, each light with and without the sun's shadow), the curve by views reported beside it; below that, the eye's own born constants (the fovea's size and code) are reconsidered before birth. The light and the shadow are never changed for it (5.4, B3). Measured on stand-in lights through the tool's projection into 256: 0.794–0.836, passing in all six settings (0.74–0.78 at 44 training views, 0.65–0.72 at 22); on the full code 0.840–0.860; linear readouts 0.52–0.64. To run at the sim's d 512. The born template's hits and false alarms are C39's. Again on the new eyes (A38, A42): the grey pair and the colour camera through the born bank at 3 px a degree, at the core's d, under W5's lights (C48) |
| C4 | Rolls under the final servo law on the G1, from the born body's own motor timing (C38), not the per-joint babbler | W4 | written down as chance; never tuned |
| C5 | Phantom pain from contact spikes, with and without the 10 ms filter | W4 | pain on still ticks under 0.01%. Measured in W1: 0 of 400 ticks at rest; 0 on the 8–9 still ticks babble gave, too few to test the bar (babble almost never rests every effector), so W4 counts still ticks from the born loop at rest; under babble 11.6% of ticks at impratio 10, 11.7% at 1. The slide's spike at impratio 10 is disclosed (5.1). Under the joints' law (A37), phantom pain is the observer's noise and error: counted on the born loop's still ticks, the same bar (C43) |
| C6 | How many fall starts the catch stops on the G1, including falls driven by the child's own big steps (the earlier "about 2.4 J at 200 N" has no source; the upper body releases about 7 J from 35° to 50°, A9) | W2 and W4 | at least 95% caught; below that, the hover distance or the trigger angle changes before birth, for the stated reason that an uncaught fall hurts. The 300 ms reaction is a person's and never changes |
| C7 | The pull-to-sit with the child's help under the caps; the guide's cap for each limb; the guide's pace on the limp arm (a 20 cm path in 1.2 s peaked at 139 N); the brief turn from its front, completed within 2 s at 200 N (A25) | W2 | no hold at its cap for more than 2 ticks, no act over the caps, and a limp arm guided at no more than 65 N |
| C8 | Peak forces when the babbling G1 meets the soft, yielding parent; her kneeling spot outside its leg sweep | W2 and W4 | under her own 150 N threshold on most ticks, and never pain by the joints' law (A37, C43; F_pain on the old skin) |
| C9 | How often a babbling child answers a call by chance | P6, with the babbler | written down as chance |
| C10 | How often babble touches the bottle beside its hand | W4 | written down |
| C11 | The amygdala: its reliability per head, the night's draw against surprise alone, the morning drift, its speed on the body's own stream, and the share of episodes with T_e ≥ 1 (if most carry a smile, "tagged first" fills half of every night) | R7's plumbing run, then the first life days | its cost measured at 0.19 ms; the rest reported daily; the constants of A16 never tuned to these |
| C12 | The tick, heat-soaked, with the three views, the eyes' lighter mesh copy, every organ of the lead's decisions and every lever that holds its rules (section 9: about 117–190 ms as designed, about 65–135 with the levers and the copy, estimated) | S5b | written down (C72). No bar binds a lockstep life since B3 is answered (A65), and nothing the body senses or does is changed for the tick; the first bar was a mean of at most 150 ms over 2 hours |
| C13 | The store's write rate, and the capacity it needs | S5b | room for 19 life days of writes |
| C14 | How long a night takes (R8), the live, dark night's physics included (A46, C52) | S5b | reported |
| C15 | Recognising words across voices, for removing the scaffolds; the tract's words in the parent's ear | P1–P3v, then after M6 begins | the tests of section 4.9 |
| C16 | Credit across the delay (risk 1) | the first life days | `act_pred` against `act_inv` on held-out guides; the value rising at the touch |
| C17 | Balance at a 150 ms tick: the leaning sit under the resting law | W1 and W4 | M5's range. W1: the leaning sit holds 3 s at rest (6° at an elbow); standing tipped past 20° after 2.85 s with the ankles at 35 N·m, and not within 3 s at the file's 50 (3.3) |
| C18 | The G1 on the mat: the sink rates under the resting law, the pain rate at 1,012 N, travel and time off the mat | W1–W4 | written down before birth. W1: the sink rates (3.3) and the pain rate (section 6); travel and time off the mat are W4's |
| C19 | Toys lost each day, if the sofa's gap stays | the first life days | reported |
| C20 | Free disk for birth, and before every verification copy; the save's real size | before S5b; before each copy; S5b | at least 8 GB left after the copy (A18); the disk rule's floor and birth's need re-set from the measured save |
| C21 | Whether being held is felt: the hold's force in the held link's touch (under A37: the observer's outside torques, and a Dex3 zone's touch where she holds a hand) | W1–W2, again when W1 reopens | the touch channel shows every hold; no hold or guide passes the joints' pain law (C43; on the old skin, every hold far under F_pain) |
| C22 | The flexor withdrawal on each G1 limb: how often the newborn's generalized flexion presses a limb into a worse contact | W1 (measured), then W4 on the born loop | written down as the newborn's (A31), with the limb resting and the babble's own acts as the chance comparisons; never tuned. W1 (8 seeds × 400 babble ticks at impratio 10, 121 onsets, either of its 2 ticks counted): it raised the pain it answered in 31 (26%: 28 on its first tick, 13 on its second), resting in 23 (19%), the babble's acts in 58 (48%); it pressed harder than resting in 60. A verifier's 4 seeds: 8 of 32, resting 8, babble 15. The first bar, "no withdrawal that raises the pain it answers", asked a newborn's reflex for a tuned one's (A31); the rate falls only if a learned tuning is built (C40). Measured again under the joints' law on the born loop, where a limb's pain is its joints' (A37, C43) |
| C23 | The tick's parts not yet measured: the parent's ear at the babble rate, and the parent's conduct layers each tick | S5a | written down with the tick (C12; no 150 ms bar since A65). Measured since (W1, W3): touch and the pain filter within the world's 2.3–2.6 ms, the face test's rays and the template within the eyes' 0.7–3.5 ms (section 9) |
| C24 | The parent's contingency: the child's acts answered within 7 ticks, while she is within 3 m and not away | P6 | her declared rate from human data (A52), within its binomial error; the first bar, at least 90%, was a perfect parent's |
| C25 | SSML's effect on per-word prosody, and the parent's word rate | P1 | at most 3 words a second on new words, pooled over the lines (a set, or an ending's lines). Measured on all 331 birth lines as new-word lines (rate 0.15, the new word emphasized), by ending: the new word 1.14–1.15 times as long as unemphasized and 1.32–1.33 times the plain line's, its F0 1.29–1.30 times both, on ".", "?" and "!" lines alike; 2.93, 2.90 and 2.51 words a second pooled (2.91 over all), the variation set 2.40–2.83; line by line, 131 lines over 3, up to 3.88 (lines of 3–6 words). The first build's "+37% F0, +91% length, 2.84 words a second" was "." lines only, with "period" spoken. Open: the limit of 3 is ours, from the parent spec, with no cited source, and it was written as pooled after the measure; a source for it, and whether it binds per line, are decided before birth (rate alone cannot bring every line under 3). The new word on the line's pitch peak is measured by frame (A34) |
| C26 | Real frictions (the world's surfaces 1.0 now; the G1's feet 0.6 at priority 1, as shipped) and a friction model that does not creep (the G1 crept 5 cm at 146–199 N) | W1 | the measured sliding force matches μ × weight, set through the world's geoms at contact priority 2 and world options only; the G1's file and geoms untouched (A21). The model is done: impratio 10 (A32), sliding within 1% of Coulomb above μMg, a creep of 0.4–1.8 mm in 2 s below it. The surfaces' own values are still 1.0: the sources found give wood on metal 0.2–0.6, polystyrene on polystyrene 0.5, rubber on concrete 0.6–0.85, and only a weak 0.63 for EVA foam (Engineering ToolBox); setting them means measuring the grasps again (C28) |
| C27 | The parent's ear in life: its cost at the babble rate, its false accepts, m at the real context sizes; how often an accepted word is also exact (A27); whether the fixed babble bank still rejects babble once the child's babble has changed | P3v and P6, then the first life days | chance written down before birth; the rule never loosened after it |
| C28 | The grasp on the eight holdable toys under the grasp reflex and the resting servo law (3.8's impratio-10 holds were at stock gains), and B1's choice | W3–W4 | written down |
| C29 | Early imitation: the first accepted echo | the first life days | reported (M6t's precursor) |
| C30 | Hits on the parent per hour under babble | W4 | written down; her withdrawals and "oh!" counted |
| C31 | The G1's falls under the final servo law: strike forces from the sits, from rolls and from a fall off its elbows, against F_pain (the custom child's head struck at 260–890 N at 9.45 kg) | W4 | written down; the pain law is not changed by them (A12) |
| C32 | Whether the lying G1 can get its trunk under the low table (head first, between its legs), and whether it gets itself out | W4, with B8's choice | written down; if it can wedge, the owner is asked about a closed base for the table before birth |
| C33 | How often the VOR's quick phase fires during trunk turns (A23) | W4, with the born VOR | reported only; the quick phase is decided on biology (A23), never by this rate |
| C34 | Whether the parent can put her face where the child's eyes reach in each posture (A22): leaning over its chest; kneeling toward its feet outside its leg sweep; low and ahead of a sitting child | W2 | a pose inside human ranges for each; where none exists, she never expects a look there |
| C35 | What the voice's demonstration law (R6's rest law on the ears, section 4.9) labels from birth: the parent's speech, the toys' sounds, footsteps; how reliable `act_inv`'s labels are on each | R6h and the plumbing run (S5a) | written down; the law is not changed by it. (The first draft's rule for "heard speech with no token" is dropped, A29) |
| C36 | The tests of understanding: how many items each kind of never-taught test has before its items are taught, and whether that can reach p < 0.01 at its chance (A28) | P4 (the items listed), then the first life days | a kind that cannot is reported "not testable yet"; more items are added only before birth; the roadmap's tests by milestone (section 12, A55) are counted the same way |
| C37 | The real G1's damage limits per link and per joint (Unitree's documents) | before the robot, and before birth where they bear on the pain law | if any joint is damaged below its torque limit, or the base below 1,012 N, the pain law takes that limit: before birth for seed 1, or for the next body after it (A12, A37) |
| C38 | The born movement units on the G1: their lengths under the gate's continuation draw at the born p_act (about 0.29, so about 1.4 ticks on average), and the rolls, travel and reaches they give (section 3.6) | R6h, then W4 on a replica and at S5a on the real core, every learning rate 0 | written down as chance. If the units are far shorter than newborns' movements (general movements last seconds), the unit's born length is decided before birth on that biology, as a disclosed constant; never on a roll count, and never after birth. Measured with the spinal pattern generator on, as part of the born body (A48, C54) |
| C39 | The born face template's constants, from its own sources. On the parent's face of human proportions it detected her face in 1 of 48 fovea readings at 0.3–2 m (4 of 810 in a verifier's sweep, all at 1.1–1.2 m, where her face is about the smallest size's width) and never in the periphery; its other matches were chance, as strong on the window upside down. Its sizes (8–23 px) fit her face in the fovea only at about 0.63–1.8 m, and in her kneeling attend pose her face lies sideways in the fovea of the G1 on its back, where an upright template cannot match. As built, the event line "a face in the fovea" and orienting's face cue would run on chance | before birth: the template's layout, sizes, contrast polarity, orientation and threshold read from CONSPEC's sources (Johnson and Morton 1991; Goren 1975; Farroni et al. 2005 on polarity) at a newborn's acuity; then measured on her face, upright and turned, under each light | set from the sources alone, never on her face or on a hit rate, and her face is never changed toward it (A30). Until then the event line and the periphery cue stay the template as built, never the world's face test (A1); orienting's sound cue is unaffected. Sim-face's study is settling it from its sources now, on the fovea's new code (the centre-surround map at 3 px a degree, A42). It also gates the born mouth-corner reader, the reward's carrier once it works (A49, C55) |
| C40 | A learned tuning of the withdrawal: the adult's local sign, learned from the tactile feedback of the body's own movements, twitches in sleep among them (Petersson et al. 2003; Waldenström et al. 2003) | a design item, not built | designed only as learning from the body's own movements, in the core's terms; never a born kinematic copy (A31). The live night's twitches (A46) now give the body such feedback; the tuning itself stays unbuilt |
| C41 | Fists closed on its own fingers: under babble the grasp fired on 14% of hand-ticks, 40% of those with only its own fingers on the palm (`palm_own_N`) | W4 | written down; if they hinder reaching or letting go, the reflex's trigger is decided on its biology before birth, never on a rate |
| C42 | A sound with nothing above threshold below about 760 Hz gets no side from the born lateral read (it reads the 80–757 Hz bands), so both "sound on the left / right" event lines fire | W5, with the toys' sounds | written down; a side for such sounds would change the born read's design, decided on the brainstem's biology before birth |
| C43 | The observer (A37): its error against the world's true contact torques (an instrument), at rest, under babble and under fast rolls; its gain; the pain rate from the joints on the born loop; phantom pain on still ticks; the withdrawal's rate (C22); every hold, guide and the pull-to-sit against the joints' pain law | W1 reopened, W2 and W4 | written down before birth; its gain from the method's own analysis (Haddadin et al. 2017), never from a pain rate. A hold or guide that would hurt tightens its cap (A25). On the real G1 its error, with the file's inertias against the robot's, is measured again before its first days |
| C44 | The Dex3-1's tactile arrays: their number, places on the palm and the finger pads, and range | before W1 reopens (Unitree's Dex3-1 documentation) | the 16 zones count contact only on the arrays' faces and saturate at their range; if a link has no array, its zone goes, and the observer alone feels it |
| C45 | The D435's sensors: the colour camera's place and axis beside the left imager (recalled as about 15 mm; RealSense's documentation gives 15 mm between the two centre-lines, read 2026-09-25, A65's evidence), the imagers' visible spectral response, and the camera model's constants (full well, read noise, exposure loop, the colour camera's readout time) | before W3 reopens (the Intel D400 series datasheet; the OV9282's and OV2740's figures) | set from those sources; where a figure is not published, a published characterisation of the D435; never fitted to the eye check |
| C46 | Unitree's servo gains for the G1 and the Dex3; each motor's torque–speed envelope, torque-estimate noise, encoder steps and thermal constants; the IMUs' bias constants; whether the firmware weakens a hot motor. Then the sink rates, rolls, the pain rate, the grasps (C28), the guides (C7) and the catch (C6) again under them | before W1 reopens (unitree_sdk2, unitree_rl_gym, Unitree's motor and IMU figures), then W1 and W4 | set from the sources, never from a measured rate; where unitree_sdk2's examples and unitree_rl_gym's configuration give different gains, the choice and its reason are written down before birth, never chosen by a measured rate; everything section 3 measured under the first law is written down again |
| C47 | The gates' drive in life: 0.25 + 4.66 R̄ per act; how often it falls below 0; the tract's performance error's mean and spread | the first life days | reported daily; never tuned (A41). A drive below 0 for whole days is watched under risk 5 |
| C48 | The new eyes: the render's cost with the grey pair at 336 × 192 and the colour camera's third view, with the sun's shadow, her face and the eyes' lighter mesh copy; the bank's cost; the eye check (C3) on the new code | W3 reopened, S5b | written down; B3 is answered (A65): the mesh copy by the render's rule (C69), the colour camera's own view and the sun's shadow kept |
| C49 | The visual onset cue's constants: the change's threshold, the turn that suppresses it, the habituation's clock; its rate of firing on its own hands and on the parent | before W3 reopens (Sokolov 1963; Johnson 1990; infant habituation studies), then W4 | set from the sources before birth; its rates written down, never tuned |
| C50 | The cerebellum's constants (the expansion's size and sparsity, the learning rate); its tests; its cost; the servo's corrective torque under a held toy and the VOR's slip in life; how much it slows a rested posture's sink (7.5) | R6c, W4, then the first life days | the constants from their sources before birth (Marr 1969; Albus 1971; Fujita 1982; the adaptive-filter literature); the sink rates written down with it learning; the rest reported |
| C51 | Recall into action: the heading's drift against the world's true yaw (an instrument), the maps' weights, recall's share of each proposal | R7's plumbing run, then the first life days | reported; the heading is never corrected from world truth |
| C52 | The live night: Sokoloff et al. (2020)'s twitch rate read exactly; the night's wall minutes; pain and falls at night; the twitch pairs' share of `act_inv`'s lessons | before R8 (the source), then S5b and the first nights | the rate from the source; the length (24,000 ticks) kept unless its wall cost breaks the day's, then the owner is told |
| C53 | The born cry: its pattern (the articulators' posture, its breath groups) and its charge line; its rate on the born loop | before R6h ends (Jürgens 2002; newborn cry acoustics), then W4 | set from the sources; its rate written down |
| C54 | The spinal pattern generator: its period, amplitude and the arms' coupling | before R6h ends (Thelen 1979; Kuniyoshi and Sangawa 2006), with C38 | set from the sources with the born units' lengths; never on a roll count |
| C55 | The born mouth-corner reader: its constants from its sources once the detector is settled (C39); the pixel table (4.3) again at 3 px a degree; the scaffold's removal test's numbers (90% sign agreement within the lean-in distance; at most 1% false smiles) | W3r before birth; the removal test on copies after | the constants from the sources, never on her face or a felt rate (A30); the test fixed before birth and never loosened (A49) |
| C56 | The voice variants: the parent's ear on them (C27), the new word on its pitch peak in each (A34), the registers kept apart across them (M1(b)'s test reads the approval register: 1.35 against the plain 1.15, ±8% each), their cost at night | P1b | written down; a variant that loses the peak is not used for a new word, as A34's frames are not |
| C57 | The parent's imperfection and copying: the share of acts missed, the latency's spread, the spells of distraction, her copying's rate and delay | before P3 (Tronick and Gianino 1986; Bahrick and Watson 1985; Ray and Heyes 2011, and the studies they cite), then P6 | read from human data before birth and never fitted to the child (A52) |
| C58 | Her reading of where it looks: the error of a person judging another's head direction; how often her reading agrees with the fovea's target under the babbler | before P3 (a source), then P6 | the error from the source; the agreement written down as what she can see (A40) |
| C59 | Hull pain: the share of the joints' pain from contacts with every hinge more than 0.2 rad from its range's ends, on the born loop | W4 | if the hulls carry it, the decomposed collision copy is loaded world-side and pain measured again (A54); the pain law is not changed by it |
| C60 | The richer room: each new object's holds by a Dex3 hand, its sounds, and the render's cost; the cover lifted by a Dex3 hand | W5b, and before each object joins | written down; an object is never shaped to be easier for the child (the cup's lesson, B1) |
| C61 | The performance error's route: in the code it is computed on effector 0's symbol and every later effector carries 0; which effector the SimAnatomy numbers 0 (section 3.5 numbers the tract 0; `anatomy_for` makes a language's symbol voice effector 0) | R6h | the error reaches the tract's gate and no other; the token output gets none (A41) |
| C62 | The human-pace ledger's recalled rows (marked † in section 12): each age read at its source. Among them: whether von Hofsten and Rönnqvist's youngest group closed the hand before touch; canonical babbling's onset range; Muir and Field 1979's newborn orienting; Woodward, Markman and Fitzsimmons 1994's 9 labelings; Saffran, Aslin and Newport 1996's 2 minutes | before birth (P4, with the tests' items) | each row read at its source; a row that cannot be sourced is removed before the freeze, never after; then the ledger is frozen with its digest (A58). Only if the ledger is kept (A66): a ledger not frozen before birth is never reported |
| C63 | The tick on a rented graphics machine: the full tick with the three views, EGL's pixels bit-identical on a repeat run, every digest pinned on its own fingerprint | closed (A61) | closed by the owner's word, "only local no cloud": no cloud machine is tested, and seed 1 is born on this Mac |
| C69 | The eyes' lighter mesh copy (A65): the coarsest decimation of the G1's visual meshes whose images, at the poses where its body is nearest its eyes (a hand before its face, an arm across its view), under each light with the sun's shadow, differ from the full meshes' within the camera model's own noise and move no silhouette by more than a native pixel; its saving in each of the three views | W3 reopened | set by the rule before birth, never on the eye check, a learning rate or a tick target; the G1's file byte-identical |
| C70 | The firsts' registration (A63): each claim's wording, bar, rulers, items and chance levels; the savings span; A62's instruments (life hours, exposures to criterion by kind, the overnight gain) and the intervention log; each registered test run on seed 1's born state at every learning rate 0 | P4 (the items), then S5b (the born state), before birth | frozen with a digest before birth; a test the born state passes at p < 0.01 leaves its claim; a kind C36 finds untestable is reported so; nothing registered changes after birth |
| C71 | Claude's round trip against a steering window (A64): its spread at P5, against a window's wall time at the pace the levers reach (26 s at 2.3×) | P5, then S5b | if it would cap the pace, each row lands two windows after its digest, set before birth and never after; otherwise one |
| C72 | The local pace in life (A62, A64): each lever's measured saving under its three rules; the heat-soaked tick with them (C12); the load from other work and the share of wall time the life gets; the wall minutes a life day | the levers' steps, S5b, then every life day | written down daily; nothing the body senses or does is changed for it |
