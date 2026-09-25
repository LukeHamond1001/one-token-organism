# Does the prefrontal ladder mature? A design study (2026-09-25)

Research only, by an Opus 5.5 study for the lead. Read against SIM_DESIGN 3.5, 7.2–7.4, 7.6, 10, 12, A20, A56, A58, and `sim-core` at
23e0fff. Abstracts read are in `$S/pfc_study/`. No life has run.

## The answer in brief

Yes, by use, not by a clock. Where age and experience have been pulled apart, infants' prefrontal gains followed experience, and the
brain's integration windows start long and shorten. So the ladder, the horizons, the hold and the gates are born whole. What matures
is their voice in action, by the core's two existing laws: readouts born small that learn (c), and voices weighted by their proven
reliability (b). Nothing needs adding to the body. Before birth: the born config (`SIM_CFG`) lacks the mechanisms the design assumes;
the slow bands must survive the night; the dreams should carry the day's band states (R8); `amyg_pav`'s switch after birth conflicts
with A20; and the tests of §4 must be registered.

## 1. What matures in the first year

| faculty | measured | source |
|---|---|---|
| working memory, reach | the A-not-B error's delay rises about 2 s a month, from under 2 s at 7.5 months to over 10 s at 12; delayed response follows it; 7.5–9-month-olds fail at 2–5 s like monkeys with dorsolateral PFC lesions | Diamond 1985; Diamond & Doar 1989; Diamond & Goldman-Rakic 1989 |
| working memory, gaze | 1–2 s from about 5.5 months, 3–5 s at 6; flat to 8 months, then linear gains to 12; reaching perseverates most | Reznick et al. 2004; Gilmore & Johnson 1995; Pelphrey et al. 2004 |
| capacity | one item in the youngest, up to four in older infants (4–13 months) | Ross-Sheehy et al. 2003 |
| perseveration | absent at 5 months, present at 7–8 | Clearfield et al. 2006 |
| inhibition | measured at 6 months, coherent at 9; 9-month-olds learn within a session to withhold looks to distractors; clear-barrier detours improve over 7.5–12 months | Holmboe et al. 2018, 2008; Diamond 1990 |
| expectation, attention | anticipatory looks to a regular sequence at 3.5 months; an omitted event timed at 4; focused attention in play rises from 10 months | Haith et al. 1988; Colombo & Richman 2002; Ruff & Capozzoli 2003 |
| credit across a delay | at 6–8 months, contingencies are learned when immediate, not at 3 s | Millar & Watson 1979 |
| planning | intentional means–end at 7 months, scaled to distance by 8 | Willatts 1999 |

**The mechanisms.** Prefrontal synaptic density peaks after 15 months (auditory near 3) and prunes into adolescence (Huttenlocher &
Dabholkar 1997); in macaque visual cortex synaptogenesis is keyed to conception, unmoved by preterm birth and extra light, while
experience acts on which synapses stay (Bourgeois et al. 1989). Myelin reaches the frontal lobes at 6–8 months (Deoni et al. 2011),
association cortex matures after the sensory cortex it integrates (Gogtay et al. 2004), and in mice prefrontal myelin needs social
experience in a critical window (Makinodan et al. 2012). Dorsolateral D1 protein rises into adulthood (Rothmond et al. 2012), and low
prefrontal dopamine (treated PKU) impairs these tasks from 6–12 months (Diamond et al. 1997). Higher-order networks knit together over
two years (Gao et al. 2015). Integration windows run the other way: infants' intrinsic timescales are longer than adults' everywhere,
without the adult gradient (Truzzi & Cusack 2023; Yates et al. 2022), and shorten from 6 to 16 months (Truzzi et al. 2026).

## 2. Calendar, gate, or emergent

**(a) A calendar.** Synapse numbers and the myelin order are largely gene-timed. But where calendar and experience were separated,
behaviour followed experience: preterm infants equal full-term ones on A-not-B by age since birth and beat them by age since
conception (Matthews et al. 1996; 20 infants); locomotor experience, even in a walker, predicts search at equal age (Kermoian & Campos
1988); weekly practice from 6.5 months beat age-matched controls at object retrieval (Bojczyk & Corbetta 2004); training raised
11-month-olds' attentional control, not their working memory, with dose (Wass et al. 2011). Even the visual critical period is paced
by input (Mower 1991). A growing-memory schedule helped one network (Elman 1993) but was unneeded, and harmful, with realistic input
(Rohde & Plaut 1999). The robot has no counterpart of the calendar parts (its synapses exist at birth; A20 forbids adding any), so no
calendar, as A56 refused one for acuity.

**(b) A reliability gate.** Control passes between prefrontal and habitual systems by each one's reliability, proposed by Daw et al.
(2005) and found in lateral and frontopolar PFC (Lee et al. 2014): the organ is present, its voice earned. The core already does this:
the long critic's "prefrontal voice", `act_pred`'s labels at kappa, the amygdala's ρ. The language body showed why: a long critic
wired in from birth read wrong, and twenty days of steps on a newborn's noise could not be undone (BODY_SPEC, runs 87–90).

**(c) Emergent.** The A-not-B error is an active trace losing to a latent habit; a stronger active trace ends it (Munakata 1998, who
set that strength by hand), and networks that learn to predict acquire such traces (Munakata et al. 1997). Prefrontal updating is
modelled as gates trained by dopamine (O'Reilly & Frank 2006), as the ladder's band gates are. The perseverative reach is built by the
history of reaching (Smith et al. 1999) and appears with skilled reaching (Clearfield et al. 2006).

**So:** (c) for delay tolerance, inhibition and attention, on gene-timed hardware; integration windows born long; (b) for the long
horizon's say in action.

## 3. This architecture, faculty by faculty

As built, a band is s ← s + (g/τ)(tanh(W_b C) − s): τ = 1 … 16,384 ticks, W_b fixed at birth, g a learned Go/NoGo gate born at 0.88.
By the design, bands 3–7 reach everything that acts only through the cortex's bundle map (`bundle_in`, born at std 0.02, learned by
prediction); the ventral critic reads bands 0–2, the tonic traces and the clock (7.2). The cortex's 64-tick window (9.6 s) is itself a
hold.

| faculty | born | what grows, by what |
|---|---|---|
| the bands | 8 clocks, open gates, fixed maps | the cortex's reading of each (c) |
| the band gates | open | when each band updates (c) |
| the fast critic | γ 0.9375 (2.4 s), matching dopamine's seconds-scale discount (Kobayashi & Schultz 2008) and the infant's under-3 s span (Millar & Watson 1979) | its reach back to earlier cues, by TD (c) |
| the long critic | γ 1 − 1/1024 (2.6 min) | its voice in the gates' credit (b): the slope of the realized return on its value, held to 0–1, every 256 ticks once 4,096 are buffered |
| the slot's latch | at event ends and dopamine above 0.5 | nothing |
| the slot's hold | one item, to a reward or 512 ticks (77 s; ours, unsourced, likely rarely binding, since each event end overwrites it) | its weight in value, by least squares (c) |
| the gates' inhibition | the no, the stop each tick, overridable born biases, a floor of 0.05 | the no, and the override of biases and habits (c) |
| `amyg_pav` | off | off for the life, or its weight the aversive head's ρ, 0 at birth (b) |

**The config, found.** `SIM_CFG` at 23e0fff sets none of the switches these rows need, and the defaults differ: the bands zeroed each
night (`night_keep_bands` 0), no striatal fast critic, no slot (`wm` 0), and the long critic mute (`vcrit_w` 0, `vcrit_auto` 0) and
reading bands 5–7; the tests add them by hand. Zeroed nightly, band 7 reaches only about 72% of its settled level by the day's end at
its born gate (1 − e^(−24,000 × 0.88 / 16,384)), and bands 6–7 become a clock of time since waking (BODY_SPEC's pages instrument).

## 4. The tests that would show it

Under section 12's rules (A19; fresh items; her eyes on the child; at most 3 probes of a kind a day; the born state's chance first; no
feedback in test trials, P3's amendment of A28). Delays of 13, 33, 67 and 133 ticks (2–20 s) straddle the window, so the long ones
test the bands and the slot. The covers come within reach only at the delay's end, as an infant's hands are held (the gates' floor of
0.05 would reach during it). A trials carry no smile, so their habit is the reach's own history (Smith et al. 1999).

| test | protocol | infant | source |
|---|---|---|---|
| T1 delayed response, gaze (from M1) | a toy at one of two places in the fovea's reach, both covered, side random; the fovea's place at the delay's end | 1–2 s from about 5.5 months, 3–5 s at 6; gains from 8 to 12 | Reznick 2004; Gilmore & Johnson 1995; Pelphrey 2004 |
| T2 A-not-B, gaze (from M1) | hidden at A until two correct first looks, then at B | ahead of reaching from 5 to 8 months, equal by 9–10 | Cuevas & Bell 2010 |
| T3 A-not-B and delayed response, reach (after M3's cover test) | the first cover touched | under 2 s at 7.5 months, about +2 s a month, over 10 s at 12 | Diamond 1985; Diamond & Doar 1989 |
| T4 a clear barrier (after M3) | a toy in a clear box open on one side; reaching through the opening | 7.5–12 months | Diamond 1990 |
| T5 expectation in time (from M1) | her face left and right on a beat, then irregularly: the fovea moving first; the forecast error at an omitted appearance | 3.5 months; 4 | Haith 1988; Colombo & Richman 2002 |

Maturation shown: the born state fails all five, and the longest passing delay rises with life hours. Perseveration rising before it
falls is expected (Munakata 1998; Clearfield et al. 2006) and reported, never treated: a fix after birth is a new seed (A20). For the
optional ledger only: T3's delay of d seconds reads as 7.5 + (d − 2)/2 months within 7.5–12 (Diamond's mean slope, wide spread).
Nothing reads it (A58).

## 5. The risks

1. **A hidden curriculum.** Scheduling hold, horizon or bands by life day writes T1–T3's curve and needs an unsourced months-to-days
   constant (A56). Refused.
2. **A band that never comes online.** Zeroed nightly, band 7 never settles; a band gate learning shut freezes its band (its clock
   τ/g, with no floor); a slow band whose block in the bundle map never grows is off in effect; the long critic stays mute if the
   world has no structure at its horizon (the language body's finding). Instruments see these; no rule corrects them.
3. **The night.** Dreams start with the bands at zero (`_dream_inputs`), so the night teaches `act_pred` and the cortex without their
   slow context: context-free habits, A-not-B's latent trace. R8's night over frames should start each window from the day's bands.
4. **The solve.** The long critic's voice needs 4,096 buffered ticks and forgets over 65,536 samples; the slot's weight enters at
   256-tick solves.
5. **`amyg_pav` switched on after birth** (7.4) is a change of body under A20.

## 6. A decision draft (the lead numbers it)

**The prefrontal parts mature by use, not by calendar** (the lead's decision; sections 3.5, 7.2–7.4, 7.6, 10, 12; this study).
- **Decided:** no calendar in the ladder, the horizons, working memory or the gates: it needs an unsourced constant (A56), infants'
  gains followed experience where it was separated from age (Matthews 1996; Kermoian & Campos 1988), and it would write the tests'
  answers.
- **Born whole:** eight bands at clocks 1–16,384, open gates, fixed maps; the 64-tick window; the fast critic at 0.9375 and the long
  at 1 − 1/1024; one slot latching at event ends and bursts; the gates with their floor, stop and overridable biases.
- **What matures:** the long critic's voice at its earned reliability (b); the bands' reach through the learned bundle map (c); the
  slot's weight by least squares (c); the gates' no and overrides by the three-factor lesson (c).
- **The born config** (`SIM_CFG`, at the served values, `tools/pins/served_cfg.pkl` on `sim-core`): `night_keep_bands` 1; `fast_input`
  "striatum", `fast_rls` 1, `stri_k` 8, `stri_m` 2,048; `wm` 1, `wm_burst` 0.5, `wm_max` 512 (ours); `vcrit_rls` 1, `vcrit_auto` 1,
  `vcrit_ceiling` "earned", `vcrit_forget` 36,000, `vcrit_traces` 1, `vcrit_clock` 1; the sim's `vcrit_bands` "0,1,2" (7.2).
- **R8:** each dreamt window starts from the day's band states at its start.
- **`amyg_pav`:** off for the life, or born with its weight at the aversive head's ρ; no switch after birth.
- **Instruments, daily, read by nothing that acts:** per band, the mean gate, τ/g, its block's norm in the bundle map and its value
  against its return; the long critic's voice; the slot's latches, holds and weight.
- **Tests:** T1–T5 join section 12, fixed before C36, their born chance measured at S5b.
- **Disclosed:** biology's calendar parts (synapse numbers, the myelin order, receptor courses) have no counterpart here; the robot is
  born with an adult's gradient of clocks and a hold beyond a young infant's (its window alone spans about an 11-month-old's A-not-B
  delay). Its prefrontal growth is growth in use; the tests report only that.
- **Cost:** config and instruments; the slot and the striatal critic are costed in 7.2.
- **Open:** the tests' born chance; the slow bands' blocks over the first 20 life days; the long critic's voice at 20 and 60 life
  hours.

## Sources

Bojczyk & Corbetta 2004 10.1037/0012-1649.40.1.54; Bourgeois et al. 1989 10.1073/pnas.86.11.4297; Clearfield et al. 2006
10.1016/j.infbeh.2006.03.001; Colombo & Richman 2002 10.1111/1467-9280.00484; Cuevas & Bell 2010 10.1037/a0020185; Daw et al. 2005
10.1038/nn1560; Deoni et al. 2011 10.1523/JNEUROSCI.2106-10.2011; Diamond 1985 10.1111/j.1467-8624.1985.tb00160.x; Diamond 1990
10.1111/j.1749-6632.1990.tb48913.x; Diamond & Doar 1989 10.1002/dev.420220307; Diamond & Goldman-Rakic 1989 10.1007/BF00248277;
Diamond et al. 1997 10.2307/1166208; Elman 1993 10.1016/0010-0277(93)90058-4; Gao et al. 2015 10.1007/s00429-014-0710-3; Gilmore &
Johnson 1995 10.1006/jecp.1995.1019; Gogtay et al. 2004 10.1073/pnas.0402680101; Haith et al. 1988 10.1111/j.1467-8624.1988.tb01481.x;
Holmboe et al. 2008 10.1016/j.jecp.2007.09.004; Holmboe et al. 2018 10.1111/desc.12690; Huttenlocher & Dabholkar 1997
10.1002/(sici)1096-9861(19971020)387:2<167::aid-cne1>3.0.co;2-z; Kermoian & Campos 1988 10.1111/j.1467-8624.1988.tb03244.x; Kobayashi
& Schultz 2008 10.1523/JNEUROSCI.1600-08.2008; Lee et al. 2014 10.1016/j.neuron.2013.11.028; Makinodan et al. 2012
10.1126/science.1220845; Matthews et al. 1996 10.1111/j.1467-8624.1996.tb01881.x; Millar & Watson 1979
10.1111/j.1467-8624.1979.tb02423.x; Mower 1991 10.1016/0165-3806(91)90001-Y; Munakata 1998 10.1111/1467-7687.00021; Munakata et al.
1997 10.1037/0033-295X.104.4.686; O'Reilly & Frank 2006 10.1162/089976606775093909; Pelphrey et al. 2004 10.1037/0012-1649.40.5.836;
Reznick et al. 2004 10.1207/s15327078in0601_7; Rohde & Plaut 1999 10.1016/S0010-0277(99)00031-1; Ross-Sheehy et al. 2003
10.1046/j.1467-8624.2003.00639.x; Rothmond et al. 2012 10.1186/1471-2202-13-18; Ruff & Capozzoli 2003 10.1037/0012-1649.39.5.877;
Smith et al. 1999 10.1037/0033-295X.106.2.235; Truzzi & Cusack 2023 10.1016/j.neuroimage.2023.120155; Truzzi et al. 2026
10.1093/cercor/bhag077; Wass et al. 2011 10.1016/j.cub.2011.08.004; Willatts 1999 10.1037/0012-1649.35.3.651; Yates et al. 2022
10.1073/pnas.2200257119.
