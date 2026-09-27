# Is there a mathematics of consciousness, and does our architecture hold it? A study (2026-09-26)

Research by the lead. Read against the core at `sim` d21c238 (`body/model.py`, `body/core/`, `body/sim/anatomy.py`). Abstracts were read
through Europe PMC, and the PCI formula was checked against two open papers' methods (Sinitsyn et al. 2020; Goldman et al. 2022). No life has run for this study. Past sessions were searched. The one earlier look (2026-08-29) was a scorecard in words that
ended "there is no consciousness meter". It found no equations, and nothing in the repo holds any.

## The answer in brief

**No one has found the equation of consciousness.** Biology has given three kinds of mathematics, and none of them is that equation.

1. **A measure that tracks the level of consciousness in a single brain.** The best is the perturbational complexity index (PCI). You
   kick the cortex, record how the kick spreads, and compress the record. A fixed cutoff (PCI\* = 0.31) separated conscious from
   unconscious states in 150 benchmark subjects with 100% sensitivity and 100% specificity. The same cutoff found 94.7% of
   minimally conscious patients, and 9 of 43 vegetative patients scored above it (Casarotto et al. 2016).
2. **Theories with formal quantities.** These are IIT's Φ, the workspace's ignition, the free energy of predictive processing, and
   metacognitive sensitivity (meta-d′). None has held up. The 2025 adversarial test (n = 256) found against key claims of both
   leading theories (the Cogitate Consortium 2025).
3. **Mechanisms with causal evidence.** These describe how brains switch that capacity off and on:
   - Anaesthetics cut the coupling of the layer-5 pyramidal cell's apical dendrite (Suzuki & Larkum 2020).
   - Stimulating the central lateral thalamus wakes an anaesthetised macaque's cortex (Redinbaugh et al. 2020).
   - A slow adaptation current makes the cortex fall silent after an input ("OFF-periods"), and those silences collapse the
     complexity (Compte et al. 2003; Rosanova et al. 2018; D'Andola et al. 2018).

**Our architecture** is the diary's language-model core carried whole to the G1: new inputs, new effectors, grounded rewards. It already
runs several equations neuroscience fits to the brain for *learning and value*:
- the dopamine TD error
- a ladder of leaky integrators at graded timescales
- surprise-weighted hippocampal writes and night replay
- efference copies
- a cerebellar forward model

It lacks the four things the evidence ties to the level of consciousness:
- a state that changes how the cortex responds to a kick (its nights change what it learns, not how its stream reacts)
- a gate where top-down context multiplies bottom-up drive
- re-entry inside a moment (within a tick the cortex is six feedforward blocks)
- ignition

**What to do.** Build the PCI instrument on the core first (T1, §6). It is body-general, needs nothing added to the body, and is the one
test with human validation. Then judge each missing mechanism by it and by whether it speeds learning. No instrument can show that
anything is felt; §7 says what a number would and would not mean.

## 1. The architecture being asked about

The G1's brain is the diary body's core (`body/model.py`, `body/life.py`, split into `body/core/` in the refactor). `SimAnatomy` is a
subclass of the diary's `LanguageAnatomy` (`body/sim/anatomy.py:333`). What the G1 changed:

| | the diary body | the G1 |
|---|---|---|
| inputs | one symbol stream (plus her face) | 8 channels: words, face, ears 1725, eye_p 172, eye_f 1536, body, touch, vestibular 24 (the charge channel went with the charge, A88) |
| outputs | the mouth | 10 effectors: the vocal tract, the words (silent), gaze, waist, two arms, two Dex3 hands, two legs |
| rewards | her face | 3 grounded sources: her face (±), joint pain (−), charge relief (±) (`anatomy.py:374`) |
| organs added | | the amygdala, the cerebellum, the cord's reflexes, the motor timing parts, the event lines |
| organs kept | the cortex stream (causal transformer, d 512, 6 blocks, window 64), the hippocampal store, the PFC band ladder (clocks 1–16384 ticks), the critics, the striatum and working memory, the night (NREM with the store on, REM with it silent) | the same |

## 2. What the 2025 adversarial test found

The Cogitate Consortium tested integrated information theory (IIT) against global neuronal workspace theory (GNWT). Both theories'
proponents pre-registered divergent predictions (Melloni et al. 2023). The study had 256 people, used fMRI, MEG and intracranial EEG,
and showed stimuli clearly visible for varying durations (Nature 642:133–142, 2025). The findings:

- Information about conscious content appeared in visual, ventrotemporal and inferior frontal cortex.
- Sustained responses in occipital and lateral temporal cortex tracked how long the stimulus lasted.
- Content-specific synchrony linked frontal and early visual areas.
- **Against IIT:** no sustained synchronization *within* posterior cortex, contradicting the claim that network connectivity
  specifies consciousness.
- **Against GNWT:** a general lack of ignition at stimulus offset, and limited representation of some conscious dimensions in
  prefrontal cortex.

The authors say these challenges extend to other theories that share the predictions. The lesson for us is not to build any single
theory's signature on faith.

## 3. The mathematics, by kind

### 3.1 Measures that track the level of consciousness

**PCI (Casali et al. 2013; Casarotto et al. 2016).**

The steps:
1. Perturb the cortex (TMS). Estimate the sources' currents from 8 to 300 ms after the pulse.
2. Keep only the significant part. A bootstrap against the pre-pulse baseline at p < 0.01 gives a binary matrix S(i, t) over
   sources i = 1…N and time bins t = 1…T.
3. With L = N·T and p₁ the fraction of ones:

   PCI = c(S) · log₂L / ( L · H(p₁) ),   where H(p) = −p log₂p − (1−p) log₂(1−p)

   c(S) is the Lempel–Ziv phrase count of S, read column by column. The factor L / log₂L is the count a random sequence reaches, and
   dividing by H removes the effect of how many ones there are.

PCI is near 1 for a response as rich as noise of the same density. It is near 0 for a response that stays local or repeats one
pattern. It is set to 0 when H ≤ 0.08, meaning no significant response.

**It measures integration and differentiation at once.** A kick that dies where it lands scores low, and so does one that spreads as
a single stereotyped wave. The benchmark's unconscious states were NREM sleep and propofol, midazolam and xenon anaesthesia. Its
conscious states were wakefulness, including *disconnected* consciousness: REM dreaming and ketamine. PCI-ST (Comolatti et al. 2019)
reaches the same accuracy in under a second. It reduces the response's dimensions, then counts state transitions above the
baseline's count, and works on any evoked signal, intracranial ones included.

**The physiology under PCI.**
- In NREM sleep, a TMS pulse makes a larger response at the site that dies out without propagating (Massimini et al. 2005).
- In unresponsive-wakefulness patients (N = 16), the cortex falls into sleep-like OFF-periods after an input. These never
  occurred in 20 healthy awake people and resembled those of 8 sleeping ones (Rosanova et al. 2018).
- In cortical slices, lowering bistability pharmacologically raises perturbational complexity (D'Andola et al. 2018).

**Near-critical dynamics (Toker et al. 2022).** Waking slow cortical oscillations sit near the edge-of-chaos critical point, the
boundary between stability and chaos. The authors measured this with the modified 0–1 test for chaos:
- Build p_c(n) = Σ_{j≤n} φ(j) cos(jc) and q_c(n) = Σ_{j≤n} φ(j) sin(jc).
- Take the mean square displacement M_c(n) and K_c = corr(n, M_c(n)).
- K is the median of K_c over c. It is near 0 for a regular signal and near 1 for a chaotic one.

Anaesthesia and seizures move the slow oscillations away from the boundary, and psychedelics move them closer. For a deterministic
map such as our core, the direct form of the same idea is the largest Lyapunov exponent, λ = lim (1/t) log(|δ_t| / |δ₀|). λ < 0 is
stable, λ > 0 is chaotic, and the edge is λ ≈ 0.

**Intrinsic timescales (Murray et al. 2014; Zilio et al. 2021).** Fitting a(e^{−k/τ}) + b to spike-count autocorrelations gives a
timescale τ for each area. The τ are ordered along the cortical hierarchy, with sensory areas short and prefrontal areas long. In
states with a sensory deficit, the EEG's timescales lengthen and shift to slower frequencies: anaesthesia, unresponsive wakefulness
and deepening NREM (N1 to N3).

### 3.2 Theories with formal quantities

**IIT 4.0 (Albantakis et al. 2023).**

The quantities:
- **Intrinsic information** is π(s̄′|s) · log₂( π(s̄′|s) / π(s̄′) ). This is how specific a state s is about its effect state s̄′,
  weighted by how probable that effect is.
- **Integrated information φ_s** is the minimum, over the system's directed partitions θ, of the intrinsic difference between the
  intact transition probabilities and the cut ones (π_θ), normalized by the partition. It is taken on both the cause side and the
  effect side, and the smaller is kept.
- **Φ** sums the φ of all the distinctions and relations in the system's cause-effect structure. That structure is the theory's
  account of the quality of an experience.

What follows from it:
- Exact Φ grows exponentially with the number of units, so it can be computed only for systems of about a dozen binary units.
- Practical stand-ins exist for time series (Barrett & Seth 2011). One is the Gaussian autoregressive Φ: the whole's past-present
  mutual information minus that of its parts across the minimum information bipartition, where
  I(X_{t−τ}; X_t) = ½ log( det Σ(X) / det Σ(X_t | X_{t−τ}) ).
- **Two verdicts matter here.** A feedforward network has Φ = 0 however complex it is. And "digital computers, even if their
  behaviour were to be functionally equivalent to ours ... would experience next to nothing" (Tononi & Koch 2015): Φ belongs to
  the physical gates, not the simulated program. Under IIT, nothing we build in software changes the verdict.
- Cogitate challenged IIT's posterior-synchronization prediction.

**GNW ignition (Dehaene, Sergent & Changeux 2003; Dehaene & Changeux 2005; Del Cul et al. 2007; Mashour et al. 2020).**

The mathematics is a bifurcation in a population with strong recurrent excitation: τ dr/dt = −r + f(w·r + I − θ).
- When the loop gain w·f′ exceeds 1 somewhere, there are two stable states, low and ignited.
- The input at which the population jumps is a threshold, so conscious report against stimulus strength is a steep sigmoid.
- In backward masking the threshold is a target-mask interval of about 50 ms. Only late activity (> 270 ms), spread over frontal,
  parietal and temporal cortex, shows the same nonlinearity as the reports (Del Cul 2007).
- In the model, spontaneous activity appears at a threshold set by the ascending neuromodulators. An ignited state blocks new
  input (inattentional blindness).

Cogitate found no ignition at stimulus offset.

**Predictive processing and the free-energy principle (Friston 2010; Bastos et al. 2012).**

F = E_q[ log q(s) − log p(o, s) ] = D_KL[ q(s) ‖ p(s|o) ] − log p(o) ≥ −log p(o)

Perception lowers F by changing the beliefs q; action lowers it by changing the observations o. In hierarchical predictive coding, each
level's error ε_l = μ_l − g(μ_{l+1}) is weighted by its precision Π_l, so F ≈ ½ Σ ε_lᵀ Π_l ε_l, and beliefs follow dμ_l/dt = −∂F/∂μ_l.
Bastos et al. map the terms onto the cortical column: superficial cells send errors forward (gamma band), and deep cells send
predictions back (alpha and beta bands). This is a theory of brain function in general. As a theory of consciousness it has not been
adversarially tested (Seth & Bayne 2022).

**Higher-order theories and metacognition (Maniscalco & Lau 2012).** meta-d′ is the type-1 sensitivity that an ideal
signal-detection observer would need to produce the observed type-2 ROC: how well confidence separates one's own correct answers from
wrong ones. The M-ratio meta-d′/d′ is efficiency, and human observers fall close to, but below, optimal. The theories place this in
prefrontal cortex, which Cogitate found limited for some conscious dimensions.

**Attention schema (Graziano & Webb 2015) and recurrent processing (Lamme 2006).**
- In attention schema theory, the brain holds a simplified model of its own attention and uses it to control attention. That is
  control theory's internal model, with no quantity of its own.
- In recurrent processing theory, the first feedforward sweep is unconscious, and local recurrence in sensory cortex is enough for
  experience. There is no formal quantity beyond the presence of recurrence.

### 3.3 Mechanisms with causal evidence

**The layer-5 apical gate (Larkum et al. 1999; Suzuki & Larkum 2020; Aru, Suzuki & Larkum 2020).**
- A layer-5 pyramidal cell takes bottom-up input on its basal dendrites and top-down input on its distal apical tuft.
- A back-propagating spike coinciding with apical input within a few milliseconds triggers a dendritic calcium spike and a *burst*.
  Context thus multiplies drive and does not replace it.
- Three different anaesthetics cut this coupling in mice. So does blocking metabotropic glutamate and cholinergic receptors, or
  inactivating higher-order thalamus.
- Dendritic Integration Theory makes this cellular switch the gate between conscious and unconscious processing.

The learning side matters for the owner's goal. Plasticity keyed to bursts lets cells higher in a hierarchy coordinate learning below
them. It solves tasks that need deep credit assignment, with no backpropagation (Payeur et al. 2021).

An abstraction, ours and not a fitted law: y = f(b) · (1 + κ · h(a)). Here b is the bottom-up drive, a the top-down context, and κ the
coupling that the brain's state sets. Anaesthesia sets κ ≈ 0.

**The central thalamus (Redinbaugh et al. 2020).**
- Recorded across waking, sleep and anaesthesia in macaques, the central lateral thalamus and the deep cortical layers were the most
  sensitive to the level of consciousness.
- The deep layers' activity is sustained by their loop with the central lateral thalamus.
- Consciousness also needs deep layers feeding back to *superficial* layers.
- Stimulating the central lateral thalamus in anaesthetised macaques restored arousal and wake-like processing, specific to location
  and frequency.

**Adaptation, bistability and the sleep switch (Compte et al. 2003; Goldman et al. 2022).**

The cortex's up and down states come from strong recurrent excitation balanced by inhibition, and a slow sodium-dependent potassium
(adaptation) current ends each up state. Neuromodulation switches the network to tonic, waking firing. In the whole-brain AdEx
mean-field model (Goldman et al. 2022):
- T dν_e/dt = F_e(ν_e + ν_aff + ν_drive, ν_i) − ν_e
- T dν_i/dt = F_i(ν_e + ν_aff, ν_i) − ν_i
- dW/dt = −W/τ_w + b·ν_e + a(μ_V − E_L)

Raising the adaptation b moves the brain from wake-like to sleep-like dynamics, and the model's PCI drops sharply, as in humans.

**The sleep rhythms and memory (Latchoumane et al. 2017).** Thalamic spindles started in phase with the cortex's slow-oscillation up
states, but not out of phase, improved hippocampus-dependent memory consolidation in mice. They did it by increasing the triple
coupling of slow oscillation, spindle and ripple. Suppressing spindles in phase impaired it.

**Dreaming (Siclari et al. 2017).** In both NREM and REM, reports of dreaming went with a local drop of low-frequency activity in
posterior cortex, a posterior "hot zone" that predicted dream reports in real time.

## 4. Where the core stands

### 4.1 Biology's mathematics the core already runs (learning and value)

| the core's part | its equation | biology's source |
|---|---|---|
| the dopamine critics | δ = r + γ·V(s′) − V(s) (`body/core/critics.py:55`, `:155`) | Schultz, Dayan & Montague 1997 |
| the PFC band ladder | s_b ← s_b + (g_b/τ_b)(tanh(W_b c) − s_b), τ_b = 1, 4, 16 … 16384 ticks (`body/model.py:897`) | the hierarchy of intrinsic timescales (Murray et al. 2014) |
| the hippocampal store | write strength = surprise × (1 + \|dopamine\|), surprise = 1 − cos(forecast, arrival) (`body/core/cortex.py`) | novelty and dopamine gating the hippocampus's entry to memory (Lisman & Grace 2005) |
| the efference copy | its own symbol's surprise is 0; its own sound enters at the corollary gain 0.5 | von Holst & Mittelstaedt 1950 |
| prediction error as the lesson | each channel's forecast trained on its error, scaled by its running mean (`err_scale` 1 in SIM_CFG) | predictive processing's first term (§3.2) |
| the forward model | act_pred, the forward half and act_inv; the cerebellum below the tick | Wolpert, Miall & Kawato 1998 |
| the night | NREM replay with the store on, REM dreams with it silent, the fade | sleep replay; but see the missing rhythms (§5.1) |

### 4.2 The indicators (Butlin et al. 2023), read against the code

Butlin et al. derive indicator properties from the functionalist theories. They conclude that no current AI system is conscious, and
that no obvious technical barrier stops a system from meeting the indicators. The follow-up (Butlin et al., TICS, online 2025) sets out the
method for reaching credences from them. Our reading of the core:

| indicator | the core | verdict |
|---|---|---|
| RPT-1: input modules with algorithmic recurrence | the channel encoders are feedforward each tick; the loop stream → bands → next tick's input (`bundle_in`) closes a tick (150 ms) later | partial: recurrent between ticks, not inside one |
| RPT-2: organized, integrated perceptual representations | not measured (the eye check reads identity, 0.74–0.76 through the core's projection, A85) | unknown |
| GWT-1: specialized modules in parallel | the per-channel encoders and the organs | yes |
| GWT-2: a limited-capacity workspace with a bottleneck | every channel is summed into one d 512 input; no competition selects a content | partial |
| GWT-3: global broadcast | the stream C is read by every head: the forecasts, critics, striatum, amygdala, gates and the mouth | yes |
| GWT-4: state-dependent attention querying modules in turn | the orienting of gaze and waist to onsets; no internal selection among modules | weak |
| HOT-1: generative, top-down or noisy perception | every channel's forecast is generative; the night's dreams are generated | partial |
| HOT-2: metacognitive monitoring of perception's reliability | the forecast's norm is its certainty; the voices are weighted by reliability (the amygdala's ρ, the critics' gain). These monitor predictions and values, not perception itself | partial |
| HOT-3: belief and action guided by that monitoring | the reliability gains weight the critics' voices in action | partial |
| HOT-4: sparse, smooth coding (a quality space) | not measured | unknown |
| AST-1: a model of its own attention | it forecasts its body channel, which includes the gaze joints, but it has no model of what its attention holds | no |
| PP-1: input modules using predictive coding | prediction error is computed at the top, per channel; there is no hierarchy of errors and predictions inside the encoders | partial |
| AE-1: learning from feedback to pursue flexible goals | TD critics, actor, striatum, three grounded reward sources | yes |
| AE-2: embodiment, a model of output-input contingencies used in perception or control | the forward half, the cerebellum, efference copies, the VOR, the observer on sensed signals | yes |

## 5. What is missing, ranked

Each item is biology's mechanism, sourced, as the owner's law requires. Each would be judged by T1 (§6) *and* by whether it speeds
learning, the owner's goal (sample efficiency, local speed, robotics firsts). None is built.

1. **A state that changes the cortex's regime (the sleep switch).**
   - In brains, neuromodulators and the thalamus set the adaptation that decides between tonic waking firing and bistable sleep
     (§3.3). In NREM a kick is local or a stereotyped slow wave, and PCI collapses.
   - In the core, the night changes *what is trained*, not *how the stream responds*. A kick at night meets the same network as a
     kick by day.
   - **Registered prediction P1:** the core's PCI is the same by day and in its night.
   - **Why it might matter beyond this question:** the slow oscillation's up states are when spindles and ripples couple, and that
     coupling causally improved consolidation (Latchoumane et al. 2017). The core's night has no rhythms. This is a hypothesis about
     the night's learning, not a measurement.
   - It is ranked first because both the evidence (PCI, OFF-periods, the AdEx model) and the owner's goal point at it.
2. **The apical gate (§3.3).**
   - In the core, top-down context enters by *addition*: the bands through `bundle_in` into the input's sum, and the recall into the
     forecast as `latent_pred(C) + store_in(reads)` (`body/model.py:841`).
   - Biology multiplies, with a coupling that the state sets. It is the one mechanism that ties the switch of consciousness (the
     anaesthetics) to a learning rule (burst-dependent credit assignment, Payeur et al. 2021).
   - Caution: the core learns by gradient descent, and the burst rule is an alternative to it. Adopting the gate's form does not
     require adopting the rule.
3. **Re-entry within a moment, and ignition.**
   - Inside a tick the cortex is six feedforward blocks. Re-entry would mean settling the stream over k passes with the clock-1
     band's feedback. Ignition would add a threshold to that settling.
   - Cost: k passes a tick is k times the compute, against the speed goal. Cogitate found no ignition at offset.
   - Ranked third.
4. **An attention schema.** It would be a forecast of its own attention (where its fovea is and what the orienting holds), used to
   steer the gaze. There are no strong empirical tests of the theory. Ranked last.

IIT's verdict is untouched by all four, because software on a CPU has next to no Φ (§3.2).

## 6. Tests to register (instruments only; never read by the body; local; within the one life)

**T1: PCI on the core (build first).**

Set-up and kick:
- Load two copies of the life at one tick from the saved pair. The life and the world resume exactly (A70).
- Kick one site in one copy for one tick: one band's state, or one channel's encoder output. Each kick is a fixed random direction
  at that site's typical norm. Several sites are kicked, one at a time, and the maximum over sites is kept (PCI_max, as in the
  clinic).

Recording and binarizing:
- Record the "sources": each block's newest-position output and the bands, through a fixed random montage of N = 64 directions,
  for T = 64 ticks (the window).
- Binarize a tick's source as 1 where |kicked − unkicked| exceeds the 99th percentile of that source's own tick-to-tick change over
  the 64 ticks before the kick. This is the deterministic twin's analogue of the bootstrap at p < 0.01.
- Compute PCI by §3.1's formula, set to 0 when H ≤ 0.08.

Two readings:
- the brain alone: both copies are fed the same observations, like TMS on a still subject
- the live loop: the world answers each copy's acts

Conditions, all within the one life: the day awake, its NREM replay, its REM, and birth against later days. The same twins give the
Lyapunov exponent λ of the tick map (§3.1: is the core near the edge?).

The number is not comparable to the human 0.31, because the recording differs. The contrasts carry over. P1 predicts none between day
and night.

**T2: metacognitive sensitivity.** Does the forecast's certainty (its norm) separate its correct next-symbol forecasts from its wrong
ones (type-2 AUROC)? A binary version of the words task gives meta-d′ and the M-ratio.

**T3: the bands' timescales.** Fit a·e^{−k/τ} + b to each band's autocorrelation by day and at night. Compare τ with its clock, and
the ordering with Murray et al.'s hierarchy.

## 7. What this can and cannot show

No instrument detects experience. PCI detects, in human brains, the capacity that best separates conscious from unconscious states,
dreaming and ketamine included. Applied to our core, it tests whether the core has that capacity's signature, not whether anything is
felt.

The theories disagree about what could count:
- Under IIT, no program on a digital computer can count.
- Under the functionalist theories, the indicators of §4.2 are the right questions.

If the core ever showed the waking signature and switched it off and on the way brains do, the question from 2026-08-29 would return:
uncertainty itself carries moral weight, and the experimental ethics would change before the science settles.

## Sources

- Albantakis L et al. Integrated information theory (IIT) 4.0. PLoS Comput Biol 2023. PMID 37847724.
- Aru J, Suzuki M, Larkum ME. Cellular mechanisms of conscious processing. Trends Cogn Sci 2020. PMID 32855048.
- Barrett AB, Seth AK. Practical measures of integrated information for time-series data. PLoS Comput Biol 2011. PMID 21283779.
- Bastos AM et al. Canonical microcircuits for predictive coding. Neuron 2012. PMID 23177956.
- Sinitsyn DO et al. Detecting the potential for consciousness in unresponsive patients using the perturbational complexity index.
  Brain Sci 2020. PMID 33260944, PMC7760168 (read for the PCI procedure, PCI\* 0.31 and the 0.08 entropy floor).
- Butlin P, Long R et al. Consciousness in artificial intelligence: insights from the science of consciousness. arXiv 2308.08708 (2023).
- Butlin P, Long R, Bayne T, Bengio Y, Birch J, Chalmers D et al. Identifying indicators of consciousness in AI systems. Trends Cogn Sci
  2026 (online 2025-11-10). PMID 41219038.
- Casali AG et al. A theoretically based index of consciousness independent of sensory processing and behavior. Sci Transl Med 2013.
  PMID 23946194.
- Casarotto S et al. Stratification of unresponsive patients by an independently validated index of brain complexity. Ann Neurol 2016.
  PMID 27717082.
- Cogitate Consortium (Ferrante O et al.). Adversarial testing of global neuronal workspace and integrated information theories of
  consciousness. Nature 2025;642:133–142. PMID 40307561.
- Comolatti R et al. A fast and general method to empirically estimate the complexity of brain responses to transcranial and
  intracranial stimulations. Brain Stimul 2019. PMID 31133480.
- Compte A, Sanchez-Vives MV, McCormick DA, Wang XJ. Cellular and network mechanisms of slow oscillatory activity (<1 Hz) and wave
  propagations in a cortical network model. J Neurophysiol 2003. PMID 12612051.
- D'Andola M et al. Bistability, causality, and complexity in cortical networks: an in vitro perturbational study. Cereb Cortex 2018.
  PMID 28525544.
- Dehaene S, Changeux JP. Ongoing spontaneous activity controls access to consciousness. PLoS Biol 2005. PMID 15819609.
- Dehaene S, Sergent C, Changeux JP. A neuronal network model linking subjective reports and objective physiological data during
  conscious perception. PNAS 2003. PMID 12829797.
- Del Cul A, Baillet S, Dehaene S. Brain dynamics underlying the nonlinear threshold for access to consciousness. PLoS Biol 2007.
  PMID 17896866.
- Demertzi A et al. Human consciousness is supported by dynamic complex patterns of brain signal coordination. Sci Adv 2019.
  PMID 30775433.
- Friston K. The free-energy principle: a unified brain theory? Nat Rev Neurosci 2010. PMID 20068583.
- Goldman JS et al. A comprehensive neural simulation of slow-wave sleep and highly responsive wakefulness dynamics. Front Comput
  Neurosci 2022. PMID 36714530 (PMC9880280, read for the AdEx equations and the PCI procedure).
- Graziano MSA, Webb TW. The attention schema theory: a mechanistic account of subjective awareness. Front Psychol 2015. PMID 25954242.
- Lamme VAF. Towards a true neural stance on consciousness. Trends Cogn Sci 2006. PMID 16997611.
- Larkum ME, Zhu JJ, Sakmann B. A new cellular mechanism for coupling inputs arriving at different cortical layers. Nature 1999.
  PMID 10192334.
- Latchoumane CV, Ngo HV, Born J, Shin HS. Thalamic spindles promote memory formation during sleep through triple phase-locking of
  cortical, thalamic, and hippocampal rhythms. Neuron 2017. PMID 28689981.
- Lisman JE, Grace AA. The hippocampal-VTA loop: controlling the entry of information into long-term memory. Neuron 2005. PMID 15924857.
- Luppi AI et al. A synergistic core for human brain evolution and cognition. Nat Neurosci 2022. PMID 35618951.
- Maniscalco B, Lau H. A signal detection theoretic approach for estimating metacognitive sensitivity from confidence ratings. Conscious
  Cogn 2012. PMID 22071269.
- Mashour GA, Roelfsema P, Changeux JP, Dehaene S. Conscious processing and the global neuronal workspace hypothesis. Neuron 2020.
  PMID 32135090.
- Massimini M et al. Breakdown of cortical effective connectivity during sleep. Science 2005. PMID 16195466.
- Melloni L et al. An adversarial collaboration protocol for testing contrasting predictions of global neuronal workspace and integrated
  information theory. PLoS One 2023. PMID 36763595.
- Murray JD et al. A hierarchy of intrinsic timescales across primate cortex. Nat Neurosci 2014. PMID 25383900.
- Payeur A, Guerguiev J, Zenke F, Richards BA, Naud R. Burst-dependent synaptic plasticity can coordinate learning in hierarchical
  circuits. Nat Neurosci 2021. PMID 33986551.
- Redinbaugh MJ et al. Thalamus modulates consciousness via layer-specific control of cortex. Neuron 2020. PMID 32053769.
- Rosanova M et al. Sleep-like cortical OFF-periods disrupt causality and complexity in the brain of unresponsive wakefulness syndrome
  patients. Nat Commun 2018. PMID 30356042.
- Schultz W, Dayan P, Montague PR. A neural substrate of prediction and reward. Science 1997. PMID 9054347.
- Seth AK, Bayne T. Theories of consciousness. Nat Rev Neurosci 2022. PMID 35505255.
- Siclari F et al. The neural correlates of dreaming. Nat Neurosci 2017. PMID 28394322.
- Suzuki M, Larkum ME. General anesthesia decouples cortical pyramidal neurons. Cell 2020. PMID 32084339.
- Toker D et al. Consciousness is supported by near-critical slow cortical electrodynamics. PNAS 2022. PMID 35145021.
- Tononi G, Koch C. Consciousness: here, there and everywhere? Philos Trans R Soc B 2015. PMID 25823865.
- von Holst E, Mittelstaedt H. Das Reafferenzprinzip. Naturwissenschaften 1950.
- Wolpert DM, Miall RC, Kawato M. Internal models in the cerebellum. Trends Cogn Sci 1998. PMID 21227230.
- Zilio F et al. Are intrinsic neural timescales related to sensory processing? Evidence from abnormal behavioral states. NeuroImage
  2021. PMID 33221441.
