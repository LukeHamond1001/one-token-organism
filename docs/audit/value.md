# Value audit: what would make this architecture worth billions

SD = docs/SIM_DESIGN.md; BS = BODY_SPEC.md; IT = ITERATIONS.md; wt_sim, wt_world = the scratchpad worktrees.

## Verdict

Today the value is an option, not an asset. The language body has real evidence: facts added without displacing old ones, and a fact learned after one telling (told five times in all that day) and recalled after a sleep. The G1 design is faithful and disclosed, and a life replays bit for bit. But no humanoid life has been run (SD l.1360), R7 and R8 are unbuilt (SD §8), and under the design's own law A20 nothing the sim child learns can reach the real G1 (SD l.1835). A lab would price it on the five demonstrations below; the fifth makes it a robotics company.

## What a top lab or investor checks

| test | where it stands | gap |
|---|---|---|
| body generality | the anatomy is declared (wt_sim body/core/anatomy.py:1-50); 8 language digests held through R1–R6 (SD §8) | one body has lived; `anatomy_for` refuses any anatomy that is not a language's (anatomy.py:509-514) |
| sample efficiency | about 3,500 world symbols a day (IT:24), so about 1.2 M characters over its ~334 days | no motor result; the G1's milestones are forecasts (SD §12) |
| continual learning | 30 facts in six days while held-out lines stayed at 0.666–0.702 (BS:1161-1170); "kiwi" learned after one telling (told five times in all before the night), recalled after a sleep (DEMO_SCRIPT.md:28-35; the correction in docs/audit/first2_word_clause.md) | days, not months; the sim store holds about 19 nights (SD §9) |
| grounded language | never-taught tests fixed before birth (SD §12, A19) | channel 0 is the parent's own label (SD §3.4); imitation 2 of 12 (SD l.52) |
| safety, interpretability | named organs; one constants table (SD §10); the amygdala cannot make reward (SD §7.4); reward only from face, pain and charge (SD §6) | on hardware: a 34 kg learner with no balance law (SD §3.3) |
| sim-to-real | the stock G1 file, byte-identical; real torque limits; senses at the real sensors (SD §3.1–3.4, A21) | skin the robot lacks (B18), colour stereo (B4), placeholder servos (SD l.228), a face reward gated by world truth (wt_world body/sim/world.py:51-52) |
| scaling | 35 M parameters at d 512 (SD §7.1) | one life at about 1× real time (SD l.1201); the critics' cost grows quadratically with width and already ruled out per-limb territories (SD §7.2) |
| evidence | rulers and chance fixed before birth (A19); digests and exact replay (SD §8) | one seed and no comparisons, by law |

## The real advantages

1. **One learner for a whole life, measured not to forget.** The facts entered the cortex over six nights (0.356 → 0.703) while the held-out lines held (BS:1161-1170). Networks trained in sequence forget (McCloskey and Cohen 1989) and lose plasticity (Dohare et al. 2024). Complementary learning systems (McClelland et al. 1995) are the textbook answer; here they run live.
2. **A child's data, not a dataset's.** π0 trained on about 10,000 hours from 7 robot configurations (Black et al. 2024). BabyLM caps its training data at 10–100 M words (Warstadt et al. 2023). The diary heard about 1.2 M characters. If the G1 reaches M2–M4 in 5–60 life hours (SD §12), that is the headline.
3. **Reproducible to the bit.** Eight digests guard every commit (SD §8), and the world saves and replays exactly (wt_sim body/core/world.py:165-166). A skeptic can replay seed 1 to any tick.
4. **Interpretable and honest by construction.** Each tick's surprise, δ, tag and reward are recorded (R7b, SD §8); every constant names its source (SD §10); the physics is measured, not hoped (a person cannot lift it, SD §0).

## The weakest claims

1. **"It will learn to move."** Nothing has lived. Rolls under babble are unmeasured. The born units average about 1.4 ticks, close to the fresh draws that gave 0 rolls (SD §3.6). Credit reaches an act 8 ticks old at 0.17 (SD §13, risk 1).
2. **"The robot after."** Under A20 a change of body is a new life (SD l.1835). The sim body differs from the real one in its skin, its stereo, its servo gains and its face test, so moving to the robot is a change of body. The real G1 would be born again and would babble on hardware: the breakage the owner built the sim to avoid. Only the code transfers, not the learned skill.
3. **"Real time on the robot."** The world waits for the child (wt_sim body/core/world.py:154-155), and deadline mode is only a test switch (SD l.164). One critic solve takes 0.78–0.88 s (SD l.981): about six ticks stalled on a robot. The tick runs at 6.7 Hz. A learned humanoid policy runs at 50 Hz over a 1 kHz PD loop (Radosavovic et al. 2024), and a limp trunk falls in about 0.19 s (SD §13, risk 2). The G1 EDU carries a Jetson Orin NX 16 GB (Unitree 2024), and nothing has been timed on it.
4. **"Body-general."** Only one body has lived, and the core's words must be a language's (anatomy.py:509-514). DreamerV3 used one set of hyperparameters across more than 150 tasks (Hafner et al. 2023), and DayDreamer used one set on four robots (Wu et al. 2022).
5. **"Grounded language."** At birth a word's meaning rides a label channel (SD §3.4), and the parent's ear hears only the words it expects (SD l.52). Vong et al. (2024) learned word–referent maps that generalize to new instances from 61 hours of one child's headcam with a generic model. That is the bar.
6. **One seed, no comparison.** SD, ARCHITECTURE.md and BS cite none of Dreamer, DayDreamer, iCub or Kuniyoshi (grep). Labs will ask.

## Where it sits in the field

| line of work | its best result | what it lacks that this has |
|---|---|---|
| world models | DayDreamer: a real A1 rolled over in 5 minutes, stood in 20 and walked in about an hour, with no sim (Wu et al. 2022) | reward from a parent, language, sleep, lifelong memory |
| robot foundation models | π0 (Black et al. 2024); RT-2 (Brohan et al. 2023); Open X-Embodiment (2024) | learning from its own life after deployment |
| sim-to-real RL | humanoids walking zero-shot after massively parallel sim (Rudin et al. 2022; Radosavovic et al. 2024), with domain randomization (Tobin et al. 2017) | one life; understanding |
| developmental robotics | iCub (Metta et al. 2010); simulated neonates' motor patterns emerging from body dynamics (Kuniyoshi and Sangawa 2006); the survey by Asada et al. (2009) | one core running both language and a full humanoid |

Each wins its own axis today; none has the last column together. That is the thesis to prove.

## The five demonstrations that would most raise its value

1. **M1–M4 on the G1 in one life, filmed live at 1×, with the never-taught tests passed** (SD §12, A19). It takes R6h, R7 and R8 (about 9.5 days, SD §8), W4's born unit lengths under C38, and a watch on the credit delay (risk 1).
2. **The same core and constants on a body with no words.** It takes an `anatomy_for` that accepts a non-language anatomy (anatomy.py:509-514) with the digests still held, then a third body with its own seed (A20), judged by the same rulers: for example a Dex3 hand at a table. Three bodies, three seeds: replication within the laws.
3. **Months without forgetting, in the body.** Re-probe week 1's skills (orienting, reaching, first names) in week 6 on their own rulers, and add a plasticity ruler (Dohare et al. 2024). It takes R8's night over frames and the store's capacity rule (SD §9). Longer lives in the house (SD §16) are the scaling evidence the laws permit.
4. **Grounded, compositional words.** M6's never-taught tests ("push the red ball", after hearing its words only in other pairings) passed after the scaffold is removed, then M6t from the tract (SD §4.9, §12). It takes the scaffold's removal tests, and imitation raised well above 2 of 12.
5. **Seed 1 continues on the real G1.** It takes these steps, in order:
   - Make the sim body the robot, sensor for sensor, so the move is a change of world, as the scaffold's removal is (SD l.1977). Touch only in the Dex3 hands, plus contact read from the joint efforts already in channel 5 (the route SD l.1414 names; Haddadin et al. 2017). Mono infrared stereo (B4). Measured servo gains.
   - Read the face from pixels with the born reader, not from the ray test.
   - A real-time mode with the critics' solves off the tick thread, timed on an Orin NX.
   - A first hardware life that cannot hurt the robot: lying on the mat, eyes and gaze only (M1), with a human's real face as the reward, then reaching.

   Each step changes the body or the world, so each is the owner's call (SD §14; SD l.170 has no live human teacher).
