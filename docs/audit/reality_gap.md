# The sim's reality gap (audit, 2026-09-24)

**The verdict.** The robot is honest: the stock G1 with its published torques (SIM_DESIGN.md 3.2). The ears are near exact: a rigid-sphere head, within 0.19 µs of the true interaural delay (wt_parent ears.py:4–45). Three gaps would teach the child things false in reality; two would stop it on the real G1. Ranked by effect on learning, then transfer. Costs are per 150 ms tick; "est." means not measured.

**1. Pain comes mostly from a collision-shape artifact (physics, touch). Learning: highest.**
- **Have:**
  - F_pain is 1,012 N per link, filtered over 10 ms (6, :959).
  - The G1's collision geoms are meshes (g1_with_hands.xml:14–15, 85, 98). MuJoCo collides meshes as their convex hulls (MuJoCo docs, "mesh").
- **Gap:**
  - Under the instrument babbler, 11.6% of ticks are pain (372 of 3,200).
  - About three quarters of the listed pain pairs are pelvis↔hip-roll housings (build/world/fix3/m_pain_imp10.json). The W1 verifier called these the "hulls meeting" (world.py:216–218).
  - C5's still-tick test passes (0 of 8), so it misses this.
  - One of three rewards would teach "hip roll hurts" from a modelling artifact, through a skin the robot lacks (B18, :2009).
- **Fix:** convex decomposition of the G1's collision meshes (Wei et al. 2022, CoACD), loaded world-side; the owner's call under A21 (:1837).
- **Cost:** est. +1–3 ms of physics (self-collision costs about 2–4 ms, 3.8).

**2. The reward and joint attention read world truth (the parent, vision). Transfer: blocker.**
- **Have:**
  - The born reading takes 2 × (smile − frown) from the world, not from pixels (A1, :1603).
  - The parent reads the child's software fovea window (B14, :2005).
- **Gap:** on the real G1 nothing reads a human smile and no one sees a software window: the reward's carrier is missing.
- **Fix:**
  - A born mouth-corner reader on the fovea's pixels, beside the CONSPEC template (eyes.py:146). The grades are visible at the lean-in distance (4.3 table, :534–541). Newborns discriminate expressions up close (Field et al. 1982).
  - The parent reads gaze only from the trunk's direction and the reaching hand, with a person's error.
- **Cost:** est. 0.1–0.3 ms. Fewer smiles felt from afar: the real condition.

**3. The parent is an immovable wall (physics, the parent's body). Learning: high.**
- **Have:** 16 kinematic mocap bodies (4.1, :425), soft contacts (A4, :1634) and a yield after 2 physics steps (:1635).
- **Gap:**
  - Babble against her reached 7,205 N (:409): people never give and can cause pain. A human arm yields as an impedance (Hogan 1985).
- **Fix:** dynamic hands and forearms with human masses, tracking the inverse-kinematics targets through actuators whose force limits are her caps (A25). Every contact is bounded each step; the yield comes from physics.
- **Cost:** about 12 degrees of freedom, est. +1–2 ms.

**4. Every spoken line is one fixed waveform (audio, the voice). Learning: high.**
- **Have:** compact Samantha (:571). Clips are byte-identical and cached by line and register (:568, :589).
- **Gap:**
  - Each line is the same waveform every time: words can be memorized as waveforms.
  - 14-month-olds learn minimal pairs only across talker variability (Rost and McMurray 2009).
  - Infant-directed speech hyperarticulates vowels (Kuhl et al. 1997).
- **Fix:**
  - Seeded variation per utterance: pitch ±8% and emphasis placement.
  - A vocal-tract-length warp in numpy (Jaitly and Hinton 2013).
  - Eight variants per line, made at night: 331 × 8 lines, about 112 MB, which fits the 300 MB cache (:590).
  - Visitors in the installed voices (A27).
- **Cost:** 0 per tick; about 2 min a night.

**5. The actuators and the proprioceptive senses are ideal (physics, proprioception, vestibular). Transfer: high.**
- **Have:**
  - Position servos with a hard torque clamp (3.3).
  - Exact angles, velocities and torques (world.py:484).
  - IMU white noise only (world.py:679–681).
  - The VOR integrates the gyro at gain 1 (:359).
- **Gap:**
  - The real G1 estimates torque from current, with torque–speed limits, backlash, heating and bus latency: the main sim-to-real gap for legged robots (Hwangbo et al. 2019; Tan et al. 2018).
  - A MEMS gyro drifts (Woodman 2007); with a real bias the window walks to its edge, and the design has no VOR adaptation, which the flocculus provides (Ito 1982).
- **Fix:**
  - Per joint: a torque–speed envelope, a first-order thermal model, encoder quantization and noise on the torque estimate.
  - A gyro bias random walk.
  - Motor figures from Unitree's documents. This exposes a missing organ (cerebellar gain learning): architecture, not tuning.
- **Cost:** under 0.1 ms, numpy over 43 joints.

**6. The world is poor in things and affordances (richness, novelty, soft bodies). Learning: high for understanding.**
- **Have:**
  - Flat-coloured primitives. Only the floor, the window and the face are textured (g1room.xml:53–124).
  - A solid cylinder cup (:505) and a rigid bear (:526).
  - Baskets and curtains without collision (:323–336, :372).
  - 14 objects, put back each morning (A5, :1647), and no second person (B13).
- **Gap:**
  - Identity is mostly colour. The eye check reads nearest-mean 0.43 and MLP 0.85 (m_eye_check_3lights_final.json). Real objects share colours and carry texture.
  - "in" is a birth word (4.8), but nothing can contain anything.
  - Nothing is new after day 1: the never-taught tests (12) run on a closed world.
- **Fix:**
  - Textures on toys, furniture and clothes. Resolution barely changes the eyes' cost (:1199).
  - A hollow cup and open baskets built from box geoms.
  - A squashable bear through softer contact.
  - A seeded inventory growing at nights (a new object every few life days, rearranged clutter) by MjSpec recompile; a visitor.
- **Cost:** about 0 per tick; seconds per night.

**7. The camera is not the real camera (vision). Transfer: high.**
- **Have:**
  - One instantaneous pinhole frame per tick, with no noise, exposure or blur (eyes.py).
  - Colour placed at the infrared imagers (B4, :267).
- **Gap:**
  - The D435's colour sensor is a separate rolling-shutter OV2740 at 69.4° × 42.5°. Its imagers are monochrome global-shutter OV9282 (Intel D400 datasheet).
  - The G1's head also carries a Livox MID-360 lidar, which is not modelled.
  - Trunk turns blur real frames; auto-exposure shifts brightness.
- **Fix:**
  - A numpy camera model after read-back: an auto-exposure loop, Poisson–Gaussian noise (Foi et al. 2008), rotational blur taken from the gyro, and gamma.
  - Decide B4 toward a real sensor.
- **Cost:** est. 0.3 ms.

**8. The room has no reverberation and no noise (audio, acoustics). Transfer: medium-high.**
- **Have:**
  - Anechoic ears (ears.py:47–52; B9, :2000).
  - No ambient noise, and no sound of the robot's own motors (B20, :2011).
  - Toy sounds are canned clips (5.2).
- **Gap:**
  - Real rooms reverberate; localization there rests on the precedence effect (Litovsky et al. 1999).
  - A robot hears its own fans and motors (Nakadai et al. 2000).
- **Fix:**
  - One diffuse late tail per ear on the mix (a feedback delay network: Jot and Chaigne 1991).
  - Image sources for the loudest source only (Allen and Berkley 1979).
  - Seeded ambient noise, and motor hum from the actuators' effort.
  - Modal contact sounds driven by the impulses (van den Doel et al. 2001).
- **Cost:** est. 0.3–0.8 ms (a source costs 0.37–0.40 ms, ears.py:52).

**9. The parent is too perfect as a person (her timing, face and imperfection). Learning: medium-high.**
- **Have:**
  - Her gaze arrives within 2 ticks, and her reply exactly 3 ticks after the child's turn (:845–852; lang/consts.py:20).
  - She always follows the child's attention and never errs.
  - Her feelings are scripted pulses (4.3). Only her blinks are random (parent_feel.py:51).
- **Gap:**
  - Real dyads are coordinated only part of the time, with mismatch and repair (Tronick and Gianino 1986).
  - 5-month-olds prefer imperfect contingency to perfect (Bahrick and Watson 1985).
  - Natural naming is mostly referentially ambiguous (Medina et al. 2011).
- **Fix:**
  - Seeded latencies, misses and distraction from these human data, never fitted to the child's rates (the parent's method is ours).
  - Small face movements and gaze aversion.
- **Cost:** 0.

**10. Time and the need (day, night, charge). Learning: low. Transfer: medium.**
- **Have:**
  - A life day is one hour, with the sun crossing in it (B19, :2010).
  - Night is a pause (risk 17, :1413).
  - The palm charges fully in 100 ticks (:930). A full charge empties in about 6,000 ticks under babble (:964).
- **Gap:**
  - An hour awake is near a young infant's wake window; the 24× light sweep matters little.
  - The real G1 runs about 2 h per battery and charges by cable or swap (Unitree spec): the bottle has no counterpart.
- **Fix:** keep and disclose; plan a dock for the robot.
- **Cost:** 0.

**Order:** 1–3 before birth (they change felt reward); 4 and 6 cost nothing per tick; 5, 7 and 8 together est. under 1.5 ms of the 100–155 ms tick (9).

Web sources: [Intel D400 datasheet](https://www.mouser.com/pdfdocs/Intel_D400_Series_Datasheet.pdf); [D435i RGB field of view](https://support.intelrealsense.com/hc/en-us/community/posts/360049401533-D435i-RGB-Sensor-FOV-); [Bahrick and Watson 1985](https://infantlab.fiu.edu/publications/publications-by-date/publications-1975-1989/1985_bahrickwatson_dp_detection-of-intermodal-proprioceptive-visual-contingency.pdf); [G1 sensors](https://robotsguide.com/robots/unitree-g1).
