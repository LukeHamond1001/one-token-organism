# Roadmap to the owner's bar

SD = docs/SIM_DESIGN.md at e1bb0b6, committed after the audits (theirs cite 1eb268b); BS = BODY_SPEC.md; `wt_sim/`, `wt_world/` = the scratchpad worktrees; "est." = my estimate.

## 1. The honest state today

The language body learned thirty facts in six nights (cortex 0.356 → 0.703) while held-out lines stayed at 0.666–0.702 (BS:1161–1170). The G1 design is disclosed for the stock robot (SD §0, §10); W1, W3, P1 and P2 are verified (SD:1425–1426), and the core holds every digest (SD §8). It is not yet a life: R6h–R8 remain, and no motor result exists (SD §12). It is not yet a human brain: it has no cerebellum; its fovea reads 4-pixel colour means (`wt_world/body/sim/eyes.py:5,63`), below newborn acuity (Dobson and Teller 1978); its face template finds her face in 1 of 48 readings (SD:2218); its gates carry drives (`wt_sim/body/core/physiology.py:128–131`) the design denies (SD:1072). It is not yet the robot: the smile is world truth (SD:1719), its skin is one the G1 lacks (SD:2173), and under A20 the real G1 would be a new life (SD:1957).

## 2. The brain: ranked additions

All decided before birth: a body changed after it is a new seed (SD:1957, 1498).

| # | addition (source) | where | cost/tick | unlocks |
|---|---|---|---|---|
| 1 | **Disclose the gates' drives.** Default: a tonic drive that follows the reward rate (Niv et al. 2007), plus the performance error on the tract only (Gadagkar et al. 2016). Amend SD:1072 and 831 | R6h, SD §10; the owner confirms (SD:1041) | 0 | honest credit |
| 2 | **Newborn acuity:** a twice-sharp fovea (SD:1278) through a born centre-surround and oriented bank (Hubel and Wiesel 1963), feeding C39's template (SD:2218) | W3; R6h; C3 again (SD:2182) | +0.3–0.9 ms (SD:37, 1278); bank est. <1 ms | M1–M2, faces |
| 3 | **Orienting to sudden visual change,** habituating (Johnson 1990; Sokolov 1963), beside the face and sound biases (SD:380). Reopens A23, ours (SD:2002) | R7a, R7e | est. 0.1 ms | M1, new toys |
| 4 | **A scoped cerebellum:** granule expansion, LMS Purkinje weights below the tick (Marr 1969; Albus 1971), taught by the servo law's corrective torque (Kawato and Gomi 1992) and retinal slip (Ito 1982): load compensation and VOR gain, not sitting (SD:385) | new R6c; a sub-tick hook (`wt_sim/body/core/world.py:151–171`) | est. ~1 ms | M2–M3, gyro drift |
| 5 | **Recall into action:** key = stream + gyro heading (McNaughton et al. 2006); value = next frame + efference copy; into proposals through a zero-born map (Lengyel and Dayan 2007); working memory latched at event ends | R7b (SD:1224) | est. <1 ms | object permanence |
| 6 | **REM twitches in a live, dark night** (Sokoloff et al. 2020; Blumberg, Marques and Iida 2013). They train `act_inv`, which gates guides (SD:1805), and C40 (SD:2219). The owner's call (SD:1018) | R8 (SD:1225) | night only | M2 |
| 7 | **A spinal CPG,** summed at the cord like A35's grasp (SD:2145) (Brown 1911; Kuniyoshi and Sangawa 2006). SD:384 assumed babble kicks; born units last about 1.4 ticks (SD:357) | R6h after C38; the owner's call | ~0 | M4 |
| 8 | **A born cry** on pain or low charge, overridable (Jürgens 2002) | R6h; the owner's call (SD:391) | 0 | first sounds |
| 9 | **Motor heat as a sense.** The real G1 reports it (unitree_hg `MotorState`) | channel 5 | <0.1 ms | transfer |

## 3. Understanding: tests by milestone

**First, close the gaze leak:** her eyes go to the object on the naming word (SD:609, 920); in asks and probes they stay on the child (Golinkoff et al. 1987). Tests are fixed before C36 (SD:2215); one-sided p < 0.01, fresh items, scaffold silenced (SD:1935, 2097).

| M | test | chance |
|---|---|---|
| M0 | touch-forecast error, its own hand against hers (Blakemore, Wolpert and Frith 1998) | ratio 1 |
| M0 | ear-forecast error at the drum seen or unseen, and at her speech with her face seen or unseen (Spelke 1976; Kuhl and Meltzoff 1982) | equal, by permutation |
| M1 | the toy's share of fovea time over 200 ticks after an event beyond its 0.99 eye-error quantile (Stahl and Feigenson 2015) | an ordinary event |
| M1 | fovea on the toy within 20 ticks of her silent head turn (Scaife and Bruner 1975) | no turn |
| M1–M2 | ball behind the table: the fovea at its far edge first (Johnson, Amso and Slemmer 2003); the eye error when it reappears (Baillargeon 1987) | no roll; an unannounced toy |
| M2 | a look to her face within 20 ticks of a never-seen toy (Walden and Ogan 1988) | matched moments |
| M3 | act rate holding the rattle (Rovee and Rovee 1969); ear error at its own shakes against hers | first holds; equal |
| M3 | a cover lifted within 40 ticks of hiding (Piaget 1954) | nothing hidden |
| M4–M5 | fovea on the toy before her reaching hand (Falck-Ytter, Gredebäck and von Hofsten 2006) | no reach |
| M6 | "where is the X?": a never-seen X beside a known non-X; at least 3 examples per noun (Quinn, Eimas and Rosenkrantz 1993) | landings on it when she names the other |
| M6 | twins as targets, each colour on at least 2 kinds (fixes SD:2082–2086); beyond year 1 (Wagner, Dobkins and Barner 2013) | noun-alone rate |
| M6 | "more" at charge below 0.35, against above 0.7 | high-charge rate |

## 4. Reality: ranked sim upgrades

**Make the sim body the robot's body,** so the real G1 is a change of world for seed 1, like the scaffold's removal (SD:2100). The owner's calls before birth:
- **B18:** touch in the Dex3 hands, plus contact from joint efforts (Haddadin et al. 2017; SD:1514).
- **B4:** the D435's real imagers (Intel D400 datasheet; SD:2159).
- **Servo gains:** measured (SD:248).
- **B14:** she reads its trunk and hand (SD:2169).

| # | upgrade | cost/tick | effect |
|---|---|---|---|
| 1 | **The smile from pixels:** a born mouth-corner reader (Field et al. 1982), once C39 works, replacing world truth (SD:1719) | est. 0.1–0.3 ms | reward on a robot |
| 2 | **Hull pain:** 11.6% of babble ticks (SD:2184); 35% of over-threshold joint contacts away from range ends (`wt_world/body/sim/world.py:216–218`). Measure on the born loop, then a convex-decomposed collision copy (Wei et al. 2022), the G1's file untouched (SD:1959) | est. +1–3 ms | pain means damage |
| 3 | **Actuators and gyro bias,** from Unitree's constants (Hwangbo et al. 2019; Woodman 2007) | <0.1 ms | transfer |
| 4 | **Eight variants per voice line,** with vocal-tract-length warps (Jaitly and Hinton 2013); about 106 MB (SD:656) | 0 | words across voices (Rost and McMurray 2009; SD:834) |
| 5 | **A richer room:** textures, containers, the cover, 3+ examples per noun, new objects by calendar | ~0 | categories, "in" |
| 6 | **A camera model** (Foi et al. 2008), with B4 | est. 0.3 ms | transfer |
| 7 | **An imperfect parent** (Tronick and Gianino 1986) | 0 | repair |
| 8 | **Dynamic parent arms** (A25) | est. 1–2 ms | low (SD:1755) |
| 9 | **Reverberation and noise** (SD:2175) | est. 0.3–0.8 ms | low |

Before birth, §2 and items 1–7 add an estimated 2–7 ms to 105–160 ms (SD:1273). That may need SD:1279's first lever.

## 5. Value: five demonstrations, in order

1. **M1–M4 in one life,** filmed at 1×, never-taught tests passed. Needs R6h–R8, §2's 1–7, C38, a credit-delay watch (SD:1497).
2. **Grounded, compositional words** with the scaffold silenced, then M6t (SD §12). Needs the leak closed, the twins, the variants, imitation above 2 of 12 (SD:834); the bar is Vong et al. (2024).
3. **Months without forgetting.** Week-1 skills re-probed in week 6, with a plasticity ruler (Dohare et al. 2024). Needs R8, the store's capacity (SD §9), the house (SD §16).
4. **The same core, a wordless body.** Needs `anatomy_for` generalized (`wt_sim/body/core/anatomy.py:500–514`), digests held; its own seed, the owner's call.
5. **Seed 1 on the real G1.** Needs §4's decision, the smile from pixels, the critics' 0.78–0.88 s solves (SD:1055) off the tick at fixed ticks, timing on the Jetson Orin NX (Unitree 2024), first days on the mat, eyes only.

## 6. The build plan (SD §11)

**Before birth** (about day 15–16, not 13; est.):
- **Core:** R6h adds §2's 1, 7, 8; a new R6c adds 4 (est. 1 day); R7 adds 3 and 5; R8 adds 6; `anatomy_for` generalized, inert for language (est. 0.5 day).
- **World:** W4 measures hull pain; W3 reopens for §2's 2 and §4's 1; §4's 2, 3, 5, 6.
- **Parent:** P1 the variants; P3 the leak closed, her copying of its movements (Ray and Heyes 2011), her imperfection; P4 §3's tests.
- **Checklist:** the owner's answers; hull pain measured; the plasticity ruler; the tick re-measured.

**After birth**, on copies at night boundaries (SD §1): §4's 8–9; new objects and visitors; a related-work section (DayDreamer's A1 rolled, stood and walked within an hour: Wu et al. 2022). Then demonstrations 3–5.

## 7. What not to do

- No learning-progress reward (it breaks SD:1041; its biology is misapplied), neuromodulator names without rules (Doya 2002), thalamic gain, breath rhythm, auditory-target reward (SD:831), acuity schedule or plasticity windows.
- No balance law; the cerebellum will not make it sit (SD:385).
- No blinded contact pairs (`world.py:216–218`) or edits to the G1's file (SD:1959).
- No body change after birth, side seeds or baselines (citing Dreamer, Hafner et al. 2023, is not a run); no parent tuned to the child (SD §1).
- No "billions" claim before demonstration 1.
