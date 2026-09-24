# The skeptic's audit of the four audits

SD = docs/SIM_DESIGN.md:line; BS = BODY_SPEC.md; `wt_sim/`, `wt_world/` = the scratchpad worktrees.

## Five things the audits got wrong or missed

1. **The pain artifact is overstated.** Pelvis↔hip-roll pairs dominate the pain list (build/world/fix3/m_pain_imp10.json), but the W1 verifier found that only 35% of the compound-joint contacts over F_pain came with every hinge more than 0.2 rad from its range's ends (`wt_world/body/sim/world.py:216-218`). The rest came near a range's end and may be real housings meeting. The run also used the per-joint instrument (p_rest 0.45), not the born loop: measure before claiming. The same file shows something new: withdrawal raised the pain it answered in 31 of 121 onsets (C22). The protective reflex flexes the leg into the pelvis.
2. **The core already carries an intrinsic credit, and the design does not say so.** The served gate carried a tonic drive and an interest term (BS:820), and the core has the songbird performance error (`wt_sim/body/core/mouth.py:693-700`; BS:284-300). Yet SD:998 says "no intrinsic bonus", SD:764 refuses a songbird tutor signal, and SD:316 says every gate is born "like the voice's" without giving `gate_tonic`, `gate_int` or `gate_int_form` for the G1. Undisclosed constants on the reward path.
3. **The biology cited for learning progress is misapplied.**
   - Redgrave and Gurney (2006) is about short-latency dopamine to unexpected salient events, for discovering agency.
   - Bromberg-Martin and Hikosaka (2009) is about a preference for advance information about reward.
   - Oudeyer et al. (2007) is a robot algorithm.
   - None shows a learning-progress reward.
4. **The stepping refusal's premise has not "failed by measurement".** Rolls are still unmeasured on the G1 (SD:1399). It fails by construction: Thelen's argument (SD:362) assumes a kicking generator, and babble is gated random draws of about 1.4 ticks (SD:336).
5. **The proposed cerebellum's teacher cannot teach balance.** Feedback-error learning needs an innate feedback controller for the task (Kawato and Gomi 1992). The servo law corrects joint angles only, and righting and equilibrium reactions are refused (SD:363). So it can learn load compensation and VOR gain from retinal slip (Ito 1982), not a sit.

## Verdicts: the brain systems map

| proposal | verdict |
|---|---|
| a born centre-surround and oriented bank; fovea at native pixels | **keep.** The fovea is 4-px colour means (`eyes.py:63,98-102`), below a newborn's ~1 c/° (Dobson and Teller 1978). It is the eye's counterpart of the cochlea. A twice-sharp render costs ~+6 ms with the sun's shadow (SD:1197-1199). |
| rising acuity; plasticity windows | **drop for year 1.** Mapping infant months onto life days needs an unsourced constant, and a window that closes wrongly costs the only seed. Vogelsang et al. (2018) is contested. |
| slip sense on the Dex3 | defer |
| working-memory latch on event ends | **fix:** R7 has not defined an "event end" yet |
| hippocampal key with heading; recall into proposals | **keep** as R7's specification (Lengyel and Dayan 2007). Cheap; on hardware a gyro heading drifts (Woodman 2007). |
| cerebellum | **fix:** limit its scope to load compensation and VOR gain (finding 5). Decide it now: a loop below the tick added later is a new seed (SD:1398; A20, SD:1835). It needs a sub-tick hook in `SimWorld.apply` (`wt_sim/body/core/world.py:150-163`); est. ~1 ms. |
| thalamic (TRN) gain | **drop:** no learning signal is named |
| visual-onset orienting (superior colliculus) | **keep**, merged with understanding 1a, with habituation (Sokolov 1963). It is the same kind of born bias as the kept face and sound orienting (SD:358), and A23's refusal (SD:1879) cites no biology. |
| a breath rhythm | **drop:** breathing is silent; no function |
| a born cry | **keep as the owner's ruling** (it reverses SD:369): innate at birth (Jürgens 2002), costs nothing. |
| a spinal CPG | **keep as the owner's ruling**, tied to C38 (finding 4). Precedent: per-muscle oscillators in a simulated neonate (Kuniyoshi and Sangawa 2006). |
| a heat drive | **fix:** add motor temperature as a sense only (the real G1 reports it), with no new reward |
| sleep by processes S and C | defer |
| noradrenaline, acetylcholine, serotonin (Doya 2002) | **drop:** three meta-controllers with no rule and no sourced constants. |
| REM twitches in a live, dark world | **keep.** It adds to REM; the rate is from Sokoloff et al. (2020); twitches give `act_inv` clean single-joint labels (SD:344). It needs the night unpaused (SD:945), which is the owner's call. Physics is cheap (SD:401-402). |
| auditory targets (DIVA) | **drop:** SD:764 refused tutor matching, and the measured bottleneck is hearing across voices: rank 22 of 50 against a chance of 25.5 (SD:767-768) |
| separate Go and NoGo weights | defer |

## Verdicts: understanding

| # | verdict |
|---|---|
| 1a | **keep** (above) |
| 1b learning-progress term | **drop as proposed.** It breaks §6 (SD:967) and rests on misapplied biology (finding 3). Replace it with finding 2's decision. |
| 1 test | **keep** |
| 2 cause-and-effect tests | **keep. Fix** the chance: the rattle and the block differ in mass and grasp, so use the rattle's own first holds as chance. |
| 3 cover toy, hiding game, look-ahead | **keep.** World-side; before C36. |
| 4 the parent copies its movements | **keep.** It is the parent's method, so it is ours (Ray and Heyes 2011). It helps only after the acuity fix. |
| 5 ≥3 examples per tested noun; C3 judges the fovea's code | **keep** |
| 6 colour twins | **fix as proposed** (0.8^20 = 0.0115 is correct). Mark colour combinations as beyond year 1 (Wagner, Dobkins and Barner 2013; Fernald, Thorpe and Marchman 2010). |
| 7 gaze-following and social-referencing tests | **keep the tests.** Her feelings about objects need the pixel face reader (reality 2). |
| 8 linking the senses; 9 self versus other | **keep.** Cheap and impressive. Test 9 is Blakemore, Wolpert and Frith (1998) in a robot. |
| 10a the gaze leak | **keep, first.** Her eyes go to the object on the naming word (SD:560, 853). |
| 10b–c | **keep.** No claim rests on the ledger's p < 0.05 window (SD:696). Claims use A19's p < 0.01 with the scaffold silenced (A29). |

## Verdicts: the reality gap

| # | verdict |
|---|---|
| 1 pain from collision hulls | **fix:** measure first (finding 1), on a decomposed collision copy loaded world-side (Wei et al. 2022). The G1's file stays untouched. |
| 2 the reward and the gaze read from world truth | **keep.** The biggest transfer blocker. A born mouth reader on pixels works only up close (SD:1603-1605), which matches Field et al. (1982). B14 has no honest counterpart on the robot. The owner must choose between a visible gaze (a change of body) and a parent who reads the trunk. |
| 3 dynamic arms for the parent | **fix: lower priority.** A4's yield already bounds her contacts after 2 physics steps (SD:1635). |
| 4 voice variants | **keep.** No cost per tick (Rost and McMurray 2009). |
| 5 real actuators; IMU drift | **keep, with sourced constants only** (Hwangbo et al. 2019). |
| 6 textures, containers, new objects | **keep.** "Nothing new after day 1" overclaims: A28 has novel toys (SD:1964). New objects arrive on a calendar, never paced by the child's progress. |
| 7 a real camera model | **keep.** Fold it into B4. |
| 8 reverberation and noise | **keep, low priority** |
| 9 an imperfect parent | **keep**, with rates from human data (Tronick and Gianino 1986) |

## Verdicts: value

| demonstration | verdict |
|---|---|
| 1 M1–M4 filmed live | **keep** |
| 2 a body with no words | **fix:** generalize `anatomy_for` now (`anatomy.py:509-514`). A third life is the owner's call, given his "no small side runs" rule. |
| 3 a plasticity ruler | **keep** (Dohare et al. 2024) |
| 4 compositional words | **keep** |
| 5 the sim body is the robot's body | **keep; the first strategic item.** Under A20 the robot would otherwise be a new life (SD:1414, 1835). The critics' solves (0.78–0.88 s, SD:981) can move off the tick thread only if each lands at a fixed tick, so exact replay still holds. |
| a related-work section | **keep.** Citing Dreamer is not a baseline run. DayDreamer's abstract gives 1 hour for rolling, standing and walking together (Wu et al. 2022), so drop the 5- and 20-minute figures. |

## What a lab would laugh at, and what would impress

- **Laugh:** the smile read from world truth; byte-identical voice clips; the naming gaze leak; a fovea coarser than a newborn's under "human brain"; pain unchecked against collision hulls; neuromodulator names with no rules.
- **Impress:** facts learned without forgetting (BS:1161-1170); exact replay; tests fixed before birth against the body's own rate; the self-versus-other test; a sim body matching the robot sensor for sensor.

## The ten to do first

1. Close the gaze leak: her head and eyes stay on the child during asks and probes.
2. Decide before birth that the sim body is the robot's body: B18, B4, the servo gains and B14.
3. Measure the hull pain, then fix it.
4. Read the face reward from pixels.
5. Disclose and decide the gates' intrinsic terms.
6. Put the fovea at native pixels with a born filter bank.
7. Add the voice variants.
8. Add visual-onset orienting with habituation.
9. Decide the organ below the tick (the cerebellum, scoped as in finding 5) and the CPG before birth.
10. Fix the test set before C36 does: the colour twins, ≥3 examples per noun, the cover toy, and the self-versus-other and linking-the-senses tests.
