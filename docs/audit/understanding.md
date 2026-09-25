# Audit: understanding

Design = docs/SIM_DESIGN.md (by line); $S = the scratchpad.

**Verdict.** The proof is disciplined: every claim is a never-taught probe against the body's own rate (§12; A19:1813; A28:1957). But infants show understanding by where they look and what they do unpaid, and this body has no reason to look at or act on anything but her face: nothing rewards novelty or its insides (§1:122, §6:967, §7.3:998), and gaze's only born pulls are faces and sound onsets (§3.7:358; A23:1879 refuses "the newest thing").

First flaw to fix: her eyes go to the object during the naming word (§4.3:560), "where is the ball?" included, so the child can pass by following her gaze (Clever Hans).

Ranked, most lacking first:

### 1. Curiosity, and a response to the unexpected: absent
- **Infants.** Collicular orienting to peripheral onsets at birth (Johnson 1990); novelty preference by 2 months (Fantz 1964); at 11 months, exploring the object that broke an expectation (Stahl & Feigenson 2015).
- **Design.** Surprise shapes memory only: store writes and replay (§7.4:1064–1069).
- **Missing.**
  - (a) A born, habituating orienting to peripheral onset or motion (Sokolov 1963): a bias like the kept face orienting, not a reward. It reopens A23, whose refusal cites no biology.
  - (b) The owner's call: a learning-progress term in dopamine (Oudeyer, Kaplan & Hafner 2007; Redgrave & Gurney 2006; Bromberg-Martin & Hikosaka 2009), against §6:967. The language body's novelty credit became a tonic drive and was removed (BODY_SPEC.md:284–300); only progress decaying to zero fits, since raw error chases noise (Burda et al. 2019).
- **Test.** After a toy's event breaks the eye forecast (above the body's own 0.99 error quantile), the toy's share of fovea ticks and touches over 200 ticks. **Chance:** that share after an ordinary event at matched visibility. Today's design predicts chance.

### 2. Cause and effect: learnable, never acted on
- **Infants.** 10-week-olds kick more when kicking moves a mobile (Rovee & Rovee 1969; Watson & Ramey 1972). Launching is perceived at 6 months (Leslie & Keeble 1987).
- **Design.** Each channel's forecast reads a stream holding its efference copies ($S/wt_sim/body/core/cortex.py:240–247; $S/wt_sim/body/model.py:704–735), so "my shake makes the rattle sound" can be learned. But the sound earns nothing; a handed-over toy earns narration only (A2:1615).
- **Tests.**
  - Arm act rate holding the rattle, which sounds with speed (§5.2), against holding the block, silent in the hand. **Chance:** the block's rate.
  - Ear-forecast error at rattle onsets its own arm caused, against her shakes at matched loudness. **Chance:** equal errors (permutation).

### 3. Object permanence: no occluder play
- **Infants.** Surprise at a vanished hidden object at 3.5–4.5 months (Baillargeon 1987); looking ahead to an occluder's exit at 4–6 months (Johnson, Amso & Slemmer 2003); search by hand at 8–9 months (Piaget 1954).
- **Design.** The window is 64 ticks, 9.6 s (§7.1:973); longer occlusions rest on the leaky bands (ARCHITECTURE.md:183). Peekaboo hides only her face (A2:1616).
- **Missing.** No toy is an occluder a Dex3 can lift (§5.2:910–924), there is no hiding game, and nothing rewards tracking a toy (#1).
- **Tests.**
  - The ball rolls behind the low table: the fovea reaches the exit edge first. **Chance:** landings there in matched windows with no roll.
  - Eye-forecast error at its reappearance, against an unheralded appearance at the same place. **Chance:** equal.

### 4. Intention reading: no route from her acts to its own
- **Infants.** At 6 months they encode a reach's goal, not its path (Woodward 1998). At 12 months, not 6, they look ahead to another's goal (Falck-Ytter, Gredebäck & von Hofsten 2006), tracking their own grasping (Kanakogi & Itakura 2011).
- **Design.** No mirror module (A10:1682–1687). She echoes its babble (§4.6:637) but never mirrors its limbs.
- **Missing.** Her contingent motor imitation, as parents do (Ray & Heyes 2011); matched experience builds the mapping associatively (Heyes 2010). It is the teacher's method, so ours.
- **Test.** During her reaches (tidying, hand-overs), the fovea lands on the goal toy before her hand does. **Chance:** matched windows without a reach.

### 5. Categories: one exemplar a word, and eyes coarser than a newborn's
- **Infants.** Perceptual categories at 3–4 months (Quinn, Eimas & Rosenkrantz 1993). Naming promotes categories from 3 months (Ferry, Hespos & Waxman 2010; Waxman & Markow 1995).
- **Design.** Each toy is one of a kind (§5.2). The fovea's code is 8 × 8 cells over 21° ($S/wt_world/body/sim/eyes.py:10, 63): about 0.2 cycles a degree, against a newborn's ~1 (Dobson & Teller 1978). All ~2,970 inputs sum into one 512-wide vector (model.py:704–735). Toys will read as colour blobs.
- **Missing.** At least 3 exemplars per tested noun, fixed before birth (C36:2051); C3 (2018) should judge the code.
- **Test.** "Where is the X?" with a never-seen X and a known non-X in view. **Chance:** landings on the new exemplar when she names the other object.

### 6. Compositional generalisation: the colour-twin test is mis-staged
- **Infants.** Word order understood at about 17 months (Hirsh-Pasek & Golinkoff 1996). Adjective–noun phrases at about 30 months (Fernald, Thorpe & Marchman 2010). Colour words come slowly (Wagner, Dobkins & Barner 2013).
- **Design** (A28:1959–1963).
  - Each colour word is heard on one object, so "red" could mean "the red block".
  - For (red, ball) and (blue, block) the target is the familiar original (§5.2:914). The noun-alone chance may then sit near ceiling: at 0.8, even 20 of 20 gives p = 0.0115, failing A19's 0.01.
- **Fix before birth.** Target the twins, (blue, ball) and (red, block). Hear each colour word on 2 or more kinds. Add an earlier verb–noun pair held out ("give" + "block"). **Chance:** the noun-alone rate, as designed.

### 7. Joint attention and social referencing: only the parent's half
- **Infants.** Following head turns at 3–6 months (Scaife & Bruner 1975; Butterworth & Jarrett 1991). Coordinated joint attention at 9–15 months (Carpenter, Nagell & Tomasello 1998). Social referencing at 12 months (Sorce et al. 1985).
- **Design.** Follow-in naming (§4.10:850). The amygdala scales orienting by her face's learned valence (§7.4:1073).
- **Missing.**
  - Following her head pays only through asks, and gaze following is learned only when it pays (Triesch et al. 2006).
  - Her irises are under one cell of the code; only her head turn is readable.
  - No object-directed affect: every face but smile and frown reads 0 (§4.3:521).
- **Tests.**
  - After her silent head turn to a toy, the fovea lands on it within 20 ticks. **Chance:** matched windows with no turn.
  - Looks back to her face within 20 ticks of a never-shown toy (Walden & Ogan 1988). **Chance:** matched moments.

### 8. Cross-modal binding: ready, untested
- **Infants.** A felt shape matched to a seen one at 1 month (Meltzoff & Borton 1979). Soundtracks matched to events at 4 months (Spelke 1976). Vowels matched to lips at 4.5 months (Kuhl & Meltzoff 1982).
- **Design.** One stream for every sense; toys' own sounds; her jaw follows her loudness (§4.4:592).
- **Test.** Ear-forecast error at the drum's boom with the strike in the fovea against out of view, and at her speech onsets with her face in view against not. **Chance:** equal error on matched clips.

### 9. Self versus other: the mechanisms are in place
- **Infants.** Newborns root less to their own touch (Rochat & Hespos 1997). 5-month-olds tell live from replayed views of their legs (Bahrick & Watson 1985). A forward model attenuates self-made touch (Blakemore, Wolpert & Frith 1998).
- **Design.** Efference copies enter the stream (model.py:725–733). The limb and voice forward errors separate its own movement and sound from others' (§3.6:346; §4.9:757).
- **Test.** Touch-forecast error at its own hand's touch against hers, matched by link and force. **Chance:** a ratio of 1.

### 10. Grounded words: the best specified, with one leak
- **Infants.** Common nouns understood at 6–9 months (Bergelson & Swingley 2012; Tincoff & Jusczyk 1999). First words near 12 months (Fenson et al. 1994).
- **Design.** The ledger (§4.8:695), base rates, and the scaffold's silencing (A29).
- **Flaws.**
  - (a) The gaze leak, which the no-word base rate does not model. Her head and eyes should stay on the child through asks and probes, as preferential-looking studies blind the parent (Golinkoff et al. 1987).
  - (b) "Understood" is judged on a sliding 10-ask window at p < 0.05 (§4.8:696). Repeated checks inflate false positives (Simmons, Nelson & Simonsohn 2011); confirm on fresh asks at p < 0.01.
  - (c) Claims should rest on the silenced-scaffold copy; the tokens give each word's identity free (§3.4).
  - (d) No feeling word among the first 50; the feed line "hungry?" (§4.7:660) fails the line check (§4.5:622).
- **Test.** It says "more" at a charge below 0.35 against above 0.7. **Chance:** its rate at high charge.

**Before birth** (C36 fixes test items then): blind her gaze in asks; retarget the twins; add exemplars and an occluder; her motor mirroring; reopen A23; the owner's call on progress; C3 judges the fovea's code.
