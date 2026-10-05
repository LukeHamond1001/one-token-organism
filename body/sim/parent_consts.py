"""THE PARENT'S BODY: HER CONSTANTS FILE (docs/SIM_DESIGN.md 4.1, 4.2, 4.10, A3, A4, A6, A7, A9, A10, A22, A25; section 10's rows
"the parent's force caps", "the guide's cap", "the parent's soft contact and yield", "the parent's own pain", "the prop's easing and
catch", "the parent's paths"; the build plan's W2). Every constant of her motion is here, with who set it and why. They are the
teacher's method (section 1: "the parent's timings, feelings, priorities and force caps live in its constants files"), set before
birth, never tuned to the child's rates, never scaled to the child (A25), and her caps may only tighten. Her talk's constants are
the fast layer's (body/sim/lang/consts.py, P3).

HER STRENGTH: A PERSON'S (A25; the owner's B16: she can never lift, slide or sit up the 34 kg G1; it must rise by itself). The
design wrote the caps "from ergonomic norms recalled from memory: the NIOSH patient-handling limit of 35 lb (Waters 2007), and the
Snook and Ciriello push and pull tables; W2 checks them against the sources". Checked 2026-09-24:
  - Waters 2007 (Am J Nurs 107(8):53-58, "When is it safe to manually lift a patient?"): the revised NIOSH lifting equation "yields a
    recommended 35-lb. maximum weight limit for use in patient-handling tasks": 35 lb = 15.88 kg = 155.7 N. The design's 160 N
    sustained was that limit rounded UP; a cap may only tighten, so both hands' sustained cap is the source's 156 N.
  - Snook and Ciriello 1991 (Ergonomics 34(9):1197-1213, "The design of manual handling tasks: revised tables of maximum acceptable
    weights and forces"; the tables as reprinted in "Using the Snook Push/Pull Tables", ergonomiesite.be, read 2026-09-24; Garg,
    Waters, Kapellusch and Karwowski 2014, Int J Ind Ergon 44:281-291, find the 1991 female values still valid): two-handed forces
    acceptable to women, at the lowest handle height tabulated (64 cm; a kneeling woman's hands at a body on the floor are lower
    still), over 2.1 m, one exertion every 5 min / 30 min / 8 h:
        sustained pull  75% of women 13 / 14 / 18 kg (128 / 137 / 177 N); 50% of women 17 / 18 / 23 kg (167 / 177 / 226 N)
        sustained push  75% of women 13 / 14 / 17 kg (128 / 137 / 167 N); 50% of women 17 / 18 / 23 kg (167 / 177 / 226 N)
        initial pull    75% of women 26 / 27 / 28 kg (255 / 265 / 275 N)
        initial push    75% of women 19 / 20 / 21 kg (186 / 196 / 206 N); 50% of women 23 / 24 / 25 kg (226 / 235 / 245 N)
    So the sustained 156 N lies between the 75th-percentile woman's acceptable force (128-177 N by frequency) and the median
    woman's (167-226 N): a person of ordinary strength, not a strong one; which percentile the parent is is written down for the
    design (the owner's "a person of real strength"). The brief 200 N (an initial force, held at most 2 s) lies within the
    75th-percentile woman's initial push (186-206 N) and under her initial pull (255-275 N): kept.
  - ONE HAND: the tables are two-handed; no one-handed acceptable force was found. Garg et al. 2014 cite one-handed dynamic pulling
    STRENGTHS (Garg and Beller 1990, men: a mean 180-360 N at 1.1-0.7 m/s; women 65% of men: Garg et al. 1988, Fothergill et al.
    1991), so a woman's one-handed dynamic pull averages about 117-234 N. The one-hand caps (100 N sustained, 150 N for 2 s) are
    under that: kept as ours, flagged (no acceptable-force source).
The caps bound HER effort, whatever the direction (a squeeze counts as a push): the sum of her holds' force magnitudes and of her
body's contact forces on the G1 (each contact's normal and friction force together), so her holds give way to what her body
already presses (the W2 verifier's finding: holds alone were counted, and friction never).
Every constant that shapes her behaviour is here; parent_motion.py names only its own guards (a planning fault, a cache)."""

# ------------------------------------------------------------------------------------------------------ HER BODY (the lead's decision)
# SHE IS A BODY (the lead's decision of 2026-09-25, after the W2 verifier's third round: a kinematic parent of infinite mass against
# the 34 kg G1 gave kN spikes, hands and a forearm inside it, levitation from the blow rule and path refusals). Her 16 segments are
# dynamic MuJoCo bodies in one tree (a free pelvis; ball joints at the lumbar, the thorax, the neck, the shoulders, wrists, hips
# and ankles; hinges at the elbows, the forearms' pronation and the knees), with a person's masses and inertias.
# HER TRUNK IS CARRIED (the lead's structural decision of 2026-09-25, A25b, after the first physical build fell onto the child and
# lay on it for minutes): a person does not fall over, and balance is the environment's business, never the child's. Her pelvis is
# held toward the pelvis her planner means by a force-limited support (HER SUPPORT, below; parent_body.Drive), her spine and neck
# stiff and limited, so her head follows her trunk; her limbs stay fully dynamic, each joint driven toward her plan by a MuJoCo
# actuator whose force range IS her strength there (HER STRENGTH, below), its damping inside that range, so no torque of hers at a
# joint ever exceeds a woman's. Her feet, knees and shins touch the floor through contact, always; every contact with the child is
# resolved by the physics.
BODY_MASS_KG = 62.0             # kg: Harbo, Brincks and Andersen 2012 (Eur J Appl Physiol 112:267-275), their 85 women's median
                                # body mass (range 46-105 kg) at a median height of 1.67 m (her stature parent_kin.H: 1.68 m)
# Segment masses as fractions of body mass, their centres of mass along the segment from its proximal end, and their radii of
# gyration (sagittal, transverse, longitudinal) as fractions of its length: de Leva 1996 (J Biomech 29:1223-1230, "Adjustments to
# Zatsiorsky-Seluyanov's segment inertia parameters"), the FEMALE columns, as reproduced in C-Motion's Visual3D documentation
# ("Adjusted Zatsiorsky-Seluyanov's segment inertia parameters", read 2026-09-25): head 0.0668, trunk 0.4257, upper arm 0.0255,
# forearm 0.0138, hand 0.0056, thigh 0.1478, shank 0.0481, foot 0.0129 (they sum to 0.9999 with two of each limb). The trunk's
# split into upper, middle and lower trunk (0.1545 / 0.1465 / 0.1247, summing to the trunk's 0.4257) is de Leva's table as
# recalled, not re-read: her chest, abdomen and pelvis. The head's centre of mass is given from the vertex (0.4841 of the vertex to
# the neck); the foot's from the heel.
SEG_MASS_FRAC = dict(pelvis=0.1247, abdomen=0.1465, chest=0.1545, head=0.0668, upper_arm=0.0255, forearm=0.0138, hand=0.0056,
                     thigh=0.1478, shin=0.0481, foot=0.0129)
SEG_COM_FRAC = dict(head=0.4841, upper_arm=0.5754, forearm=0.4559, hand=0.7474, thigh=0.3612, shin=0.4352, foot=0.4014)
SEG_GYR = dict(head=(0.271, 0.295, 0.261), upper_arm=(0.278, 0.260, 0.148), forearm=(0.261, 0.257, 0.094),
               hand=(0.631, 0.454, 0.335), thigh=(0.369, 0.364, 0.162), shin=(0.267, 0.263, 0.092), foot=(0.299, 0.279, 0.139))
SEG_LEN = dict(head=0.266, upper_arm=0.30, forearm=0.25, hand=0.094, thigh=0.41, shin=0.41, foot=0.254)
                                # m: her own skeleton's (parent_kin: L_UA, L_FA, L_TH, L_SH), her head from the neck joint to the
                                # top of its solid, her hand from the wrist to the knuckle line (de Leva's hand ends at the third
                                # metacarpale), her foot from the shoe's heel to its toe (parent_poses.SHOE_HEEL, SHOE_TOE): ours.
                                # Her trunk's three parts take their centres of mass at the centres of their collision ellipsoids
                                # and their inertias as solid ellipsoids of those shapes (ours: de Leva's trunk parts are measured
                                # between landmarks her trunk does not have)
# HER STRENGTH: each joint's torque limit, N m, per direction, a woman's. Harbo, Brincks and Andersen 2012 (above; their women's
# mean peak torques, Tables 3 and 4, the isometric figure where one was measured, else the isokinetic at 60-90 deg/s): shoulder
# abduction 38.0 (isometric), adduction 45.7; elbow flexion 26.5 (isometric), extension 27.2; wrist flexion 14.4 (isometric),
# extension 6.01; hip flexion 104.4 (isometric), extension 128.7; knee extension 166.6 (isometric), flexion 59.3; ankle
# dorsiflexion 27.5 (isometric; their "ankle extension"), plantarflexion 76.4 (their "ankle flexion"). The neck: Garces et al. 2002
# (Med Sci Sports Exerc 34:464-470), women's peak isometric torque, flexion 16.6, extension 26.5 (their abstract, read through a
# search index 2026-09-25). The trunk: Nordin et al. 1987 (Spine 12:105-111), 101 women's isometric trunk flexion 19-109 N m and
# extension 38-168 N m (the ranges, read through a search index): the midpoints 64 and 103, ours. OURS, flagged (no source read):
# the shoulder's flexion and extension take its abduction and adduction figures; its rotation half the abduction's; pronation
# the wrist's weakest (extension, 6.0); the wrist's deviation 6.0 and its twist 3.0; the neck's side bend and turn its flexion's;
# the trunk's side bend its flexion's and its twist half; the hip's abduction and adduction 80 and rotation 30; the ankle's
# inversion 20 and twist 10. Each limit is her actuator's force range at that joint and direction (make_g1room.parent_actuators:
# MuJoCo's forcerange, and ctrlrange the same), so it binds her whole torque there, her plan's spring, her damping, her posture's
# feedforward, her tone and her holds' effort together: MuJoCo clamps the actuator's force, the damping included, and nothing
# else of hers acts on her joints (no joint damping, no applied force on them). A ball joint's three actuators each bind one axis
# of its own frame (the per-axis figures above; a torque along a diagonal may reach the axes' figures together, as the per-axis
# measurements allow). body/tests/test_sim_parent.py reads her actuators' forces on every physics step under babble against
# these limits.
STRENGTH = dict(shoulder=dict(flex=38.0, ext=45.7, abd=38.0, add=45.7, rot=19.0), elbow=dict(flex=26.5, ext=27.2),
                pron=dict(rot=6.0), wrist=dict(flex=14.4, ext=6.01, dev=6.0, rot=3.0),
                neck=dict(flex=16.6, ext=26.5, lat=16.6, rot=16.6), trunk=dict(flex=64.0, ext=103.0, lat=64.0, rot=32.0),
                hip=dict(flex=104.4, ext=128.7, abd=80.0, add=80.0, rot=30.0), knee=dict(flex=59.3, ext=166.6),
                ankle=dict(dorsi=27.5, plantar=76.4, inv=20.0, rot=10.0))
STIFF_SAT_DEG = 5.0             # her spine's two joints and her neck (her trunk carried above her pelvis, her head following it: the
                                # lead's decision, "a stiff neck, limited"): each spends its whole strength 5 deg off her plan (ours:
                                # four times SAT_DEG's stiffness; the strength still binds) ...
LEG_REST_SAT_DEG = 60.0         # her legs' stiffness while her base is still (kneeling, standing or sitting where she is): they rest
                                # where her carried pelvis puts them, on the floor, as a person's relaxed legs do (ours: at SAT_DEG a
                                # leg planned on the floor pressed it with up to a leg's strength whenever her pelvis sat a centimetre
                                # off her plan, 0.8-1 kN at the thigh, A25b's build) ...
SAT_DEG = 20.0                  # ... every other joint's stiffness toward her plan: its strength (the mean of its two directions) over this
                                # angle, so it spends its whole strength 20 deg off its plan (ours). The arm's endpoint stiffness it
                                # gives (about 400 N/m at the shoulder's 120 N m/rad, the hand 0.55 m out) is of the order a person's
                                # arm holds a posture with, a few hundred N/m (Mussa-Ivaldi, Hogan and Bizzi 1985, J Neurosci
                                # 5:2732-2743, recalled)
TONE_S = 0.3                    # s: her tone's time constant: each joint's error integrated at its stiffness over this, within its
                                # strength, so she holds the pose she means against its load (ours: her kneeling body had rested
                                # 4 cm off its plan on its springs alone, and her hand with it)
TONE_SHARE = 0.5                # her tone reaches at most this share of each joint's strength (ours: a plan the floor makes
                                # unreachable, her heel's last degree, must not wind her whole strength up against it)
TONE_STILL_RPS = 0.2            # rad/s: ... settling only on a joint whose planned speed this tick is under this (ours: a moving
                                # joint's lag is her springs' and damping's; integrated, it wound up and threw her reaching hand
                                # 30 cm past its target)
DAMP_RATIO = 1.0                # each joint's damping, critical against its stiffness and the inertia it moves at her rest pose
                                # (MuJoCo's dof_M0; ours)
# HER SUPPORT: HER TRUNK CARRIED (the lead's structural decision of 2026-09-25, A25b; amends A25 and the first physical build's
# balance, whose residual force had carried her gait with her floor contact off, lifted her at up to 1,308 N while she touched the
# child and let her fall onto it). A person does not fall over: balance is the environment's business, never the child's. Every
# physics step her pelvis is pulled toward the pelvis her planner means (its place and its turn, from her plan at the last tick's
# end to her plan at this one's) by a spring and a damper on it, and her weight is carried where it is (a force equal to her
# weight, up, at her whole body's centre of mass: it carries her weight without twisting her). Its force together (her weight
# carried and the spring's) is capped at SUP_F_MAX and never points down, and its torque is capped at SUP_T_MAX (parent_body.Drive:
# computed each step as plain numbers, applied as an outside force on her pelvis). So she cannot be toppled by the child, cannot
# press more than her own weight down through her trunk onto anything (the support never pushes her down), and cannot press more
# than SUP_F_MAX along the floor; pushed harder than that she gives way along the floor, carried, and never falls. Her planner keeps
# her trunk clear of the child (CLEAR_M, re-planned as it moves: parent_motion._standoff_tick), so the support carries her toward
# no contact with it; where the child moves into her the physics resolves it and her plan backs off (A4).
# The caps, a person's:
SUP_F_MAX = 695.5               # N: her weight carried (BODY_MASS_KG x g: 62 kg x 9.81 = 608.2 N) and, along the floor, the most the
                                # child can push anything with: its weight on the floor times the floor's friction (the G1's 34.39 kg
                                # from its model file x 9.81 = 337.4 N; the mat and the floor at friction 1.0, section 4.2), so
                                # |(608.2, 337.4)| = 695.5 N, 1.14 x her weight: the child's greatest steady push never topples her
                                # (a kick's impulse is more for a few milliseconds; her body's mass takes it). Ours, from those
                                # three sourced numbers (A25b)
SUP_T_MAX = 257.4               # N m: the torque that holds her pelvis's turn is at most what her legs hold a pelvis with, her two
                                # hips' extension strength (2 x 128.7 N m: Harbo, Brincks and Andersen 2012, the women's mean, as
                                # STRENGTH's); the child's steady push (337.4 N) at her chest's height over her pelvis (about 0.5 m)
                                # asks 169 N m (ours, A25b)
SUP_K = 20000.0                 # N/m: the support's spring on her pelvis's place (ours: 1 cm off her plan per 200 N; her planner's
                                # standoff and A4's yield do the rest) ...
SUP_ZETA = 1.0                  # ... damped critically against her whole body's mass (2 sqrt(SUP_K x BODY_MASS_KG): 2,227 N s/m) ...
SUP_KR = 2000.0                 # N m/rad: ... its spring on her pelvis's turn (ours: 1 deg per 35 N m) ...
SUP_CR = 25.0                   # N m s/rad: ... and its damping (ours: the most an explicit damper on her pelvis's own inertia, about
                                # 0.06 kg m^2 at 2 ms steps, bears with a margin of two: 0.06 / 0.002 / 2 = 15-30; her spine's and
                                # hips' actuators damp the rest of her body's turn, implicitly)
# ------------------------------------------------------------------------------------------------------ her caps (A25; ours, checked)
CAP_ONE = 100.0                 # N, one hand, sustained (ours; under a woman's one-handed dynamic pulling strength, see above)
CAP_ONE_BRIEF = 150.0           # N, one hand, for at most BRIEF_S (ours; the same)
CAP_TWO = 156.0                 # N, both hands together, sustained: NIOSH's 35 lb patient-handling limit (Waters 2007); the
                                # design's 160 tightened to its own source
CAP_TWO_BRIEF = 200.0           # N, both hands, for at most BRIEF_S: within Snook and Ciriello's initial push for 75% of women
BRIEF_S = 2.0                   # s: the brief caps' longest use (A25); after it, the sustained cap until she has rested
BRIEF_REST_S = 300.0            # s under the sustained caps before a brief cap is hers again: Snook and Ciriello's initial forces
                                # (the source of CAP_TWO_BRIEF) are for one exertion every 5 min at most, their lowest tabulated
                                # frequency (the W2 verifier's finding: 2 s, the burst's own length, was looser than the source)

# ------------------------------------------------------------------------------------------------------------- the capped spring
HOLD_K = 2000.0                 # N/m: her hand's stiffness on a held point (ours: the order of a human arm's endpoint stiffness
                                # in a steady effort, Mussa-Ivaldi, Hogan and Bizzi 1985, Perreault et al. 2001, recalled, not
                                # checked). The cap bounds the force; K only sets how far a hold gives before the cap
HOLD_C = 60.0                   # N s/m: its damping on the held point's velocity relative to her hand (ours; stable at 2 ms
                                # steps on the lightest held link, measured by body/tests/test_sim_parent.py)
AT_CAP = 0.98                   # a hold is "at its cap" on a step when its force is at least this share of the cap (ours)
AT_CAP_TICKS = 2                # a hold at its cap on every step of this many ticks stops (A10, A25: "a hold or guide at its cap
                                # for 2 ticks stops")
HOLD_SLIP_M = 0.12              # m: a held point this far from her hand's reach has slipped from her grip (ours: a hand's length)
GRIP_TOL_M = 0.03               # m: her hand within this of its grip on the held point holds with the spring's whole force; farther,
                                # less, down to none at HOLD_SLIP_M (ours: she is a body, and her grip is where her hand is: an arm
                                # that cannot carry a load is pushed off it)

# ------------------------------------------------------------------------------------------------------ her acts' own caps (A8-A10)
TOUCH_N = 5.0                   # N: a hand resting on the child (attend, "touch"): the relaxed hand's weight, 0.6% of a 60 kg
                                # woman's body weight (Winter's segment table: the hand 0.006 M), with some of the forearm's
                                # (ours)
ATTEND_PARTS = "chest|tummy|leg|foot"   # attend's resting hand: on its trunk (4.2), its chest first (out of the fold of its hips:
                                # its thighs close on its belly, 88 N m at the hip, and a hand there is caught between them: the
                                # first physical build's stopped fix round, kept), or, where only its feet leave her room to kneel
                                # (A6: a child in a corner), on its leg or its foot, the first her hand reaches (ours)
TOUCH_DEPTH_M = 0.02            # m: a resting hand's target lies this far inside the surface along its normal (the cap decides
                                # the force; ours)
GUIDE_CAP_FACTOR = 1.5          # the guide's cap: min(1.5 x the limb's own push at that pose, CAP_ONE) (A10; ours)
GUIDE_MAX_TICKS = 8             # a guide's path lasts at most one movement unit (A10)
GUIDE_MAX_SPEED = 0.30          # m/s, the guide's fastest pace (A10); its pace is set on the yielding arm (A25): GUIDE_SPEED
GUIDE_SPEED = 0.30              # m/s: the guide's pace, set in W2 on the limp arm A25 names (tools/sim_parent_motion.py guide_pace:
                                # the arm's actuators zeroed, the near forearm raised GUIDE_RAISE_M), the fastest of 0.30 / 0.25 /
                                # 0.20 / 0.15 / 0.12 / 0.10 / 0.08 m/s whose peak stayed within the arm's own push (93.9 N there):
                                # peaks 53.5 / 50.3 / 50.3 / 47.2 / 47.1 / 47.0 / 36.1 N, all within it and within the design's
                                # 65 N, so A10's fastest. Under the resting servo law the same guides peak 99.0 / 98.5 / 98.5 /
                                # 93.4 / 90.1 / 88.7 / 76.9 N against its 94.2 N push: an arm that holds meets her cap and stops
                                # her (A10, A25). The W2 verifier's finding: 0.15 m/s had been set on the resting arm, with a
                                # 1.3 N margin. Never tuned to the child's learning (A25)
ROLL_ARM_N = 65.0               # N: the far arm guided across the chest (A8)
ROLL_KNEE_N = 76.0              # N: the far knee bent over, one hand (A8; section 4.2's "bend a leg: 60-76 N")
CAP_RAMP_NPS = 100.0            # N/s: a pull or a turn grows its force at this rate, so it reaches CAP_TWO_BRIEF at BRIEF_S
                                # (ours: "a slowly growing force", A7, A9)
PULL_MAX_S = 4.0                # s: a pull that has not reached her cap in this long (the ramp's 2 s and as long again) is laid back
                                # (ours: her grip or her arms gave out; she is a body)
PLAN_LEAN_MAX = 60              # C256: the plan's reach test (_reachable: the put's place, the pick from where she kneels, the turn's grips)
PLAN_SPINE_MAX = 35             # ... allows this much lean and spine flexion, ten degrees inside her limits (70, 45): a reach planned at the
                                # edge of her stretch misses by the centimetre the test did not see; a person plans inside it (ours)
REST_VIA_M = (0.30, 0.25)       # C259: a hand coming to rest whose way would pass within the forearm's length of her shoulder (its way
                                # lifted over the child comes down from right above the hand hanging below the shoulder) comes down in front
                                # of her first: through a point this far in front of the shoulder and this far below it, in her chest's frame
                                # (ours: the forearm's length in front, the hand at the elbow's height: the arm bent at a right angle before it
                                # hangs, as an arm comes down from a raise)
SWIVEL_STEP_DEG = 45.0          # C255: her elbow's swing about the shoulder-wrist line moves at most this a tick toward the swing the placing
                                # chose when the search ran wide (a target moved round her arm, the relax's carried arm): 45 deg on an elbow's
                                # circle of 0.25 m is 0.2 m a tick, under MAX_JUMP_M; before, 60 to 120 deg flips of 0.3 to 0.6 m (ours)
TURN_REPLANS = 2                # C250: a turn whose grips lie out of her reach when she gets there (the child moved while she came) is approached
                                # again from where it lies now, at most this many times (as the hand-over's HANDOVER_REPLANS) (ours)
TURN_RISE_DEG = 1.0             # C241: the turn's clock counts no tick where its chest has turned this much further from face down than its best so
                                # far (as PULL_RISE_DEG for the pull, C235); the stop is TURN_MAX_S without a turn, or TURN_TOTAL_S in all (ours)
TURN_TOTAL_S = 10.0             # s: the turn's longest effort in all, at the sustained caps after BRIEF_S (the brief clock's law, A25) (ours)
TURN_MAX_S = 4.0                # s: the turn from its front lasts at most this (A7 gave it 2 s; A101: a person's roll of a heavy child
                                # takes 3-4 s: the brief caps carry its first 2 s (BRIEF_S), the sustained caps the rest, by her caps'
                                # own law; ours, disclosed)
TURN_RAMP_NPS = 400.0           # N/s: the turn's force builds to the brief caps within half a second (A101; ours: the initial push of
                                # Snook and Ciriello's tables is exerted from the first moment, and CAP_RAMP_NPS spent the whole 2 s
                                # climbing to it, so the roll never had its force)
GUIDE_RAISE_M = 0.12            # m: a guide raises the near forearm this far (ours: within one movement unit at GUIDE_SPEED,
                                # 8 ticks x 0.30 m/s = 0.36 m, and inside the arm's reach off the mat)
FOREARM_HOLD = (0.07, 0.0, 0.0) # where she takes a G1 forearm (its elbow link's frame): 7 cm down it from the elbow, ...
FOREARM_HOLD_N = (0.0, 0.0, 1.0)    # ... her palm on its surface along this normal (the link's +z: its top as it lies; ours)
ROLL_ARM_M = 0.15               # m: the far arm drawn this far toward her and up across its chest (A8; ours)
KNEE_OVER_M = 0.15              # m: the far knee drawn this far toward her and up (A8; ours)
PULL_RISE_DEG = 1.0             # deg a tick: the trunk coming up by this much or more under the pull is a heavy child rising, not one
                                # resisting at her cap, and the cap's stop (AT_CAP_TICKS) counts no such tick (C235; ours: a tenth of the
                                # trunk's swing at her hands' pace, GUIDE_SPEED over its length)
PULL_LEAD_M = 0.30              # m: the pull's spring target lies this far ahead of the held forearm, so its force is set by the
                                # ramp and her caps, never by how far the spring stretches (HOLD_K x 0.30 = 600 N, over every cap;
                                # ours)
TURN_LIFT_M = 0.12              # m: the turn's spring target lies this far along its push (ours: as PULL_LEAD_M, HOLD_K x
                                # 0.12 = 240 N, over the one-hand brief cap it is given) ...
TURN_OVER_UP = 0.4              # ... its push straight up while the chest faces down, then this much up with the rest toward
                                # its back, away from her (ours) ...
TURN_PAST_Z = 0.3               # C167 (2026-09-30): the turn is DONE only once its chest's normal has risen past this beyond its side (0 is its
                                # side; ours: about 17 deg past, where a body on its side settles rather than tips back), or it lies on its
                                # back. Until then "past its side" at 0.0 was done, and under A143 (the passive end-range stiffness) the
                                # child turned from beside its chest lay back prone after her hands slipped at its side: a false done in her
                                # log, the turn never re-asked (parent 30's second route, read 2026-09-30 22:15)
TURN_SIDE_Z = 0.3               # ... switched when its chest's normal rises past this (a unit vector's z: face down -1, on its
                                # side 0; ours). It has turned when it lies on its back, or on its side with its chest past this

# ------------------------------------------------------------------------------------------------------ the prop and the catch (A9)
SITTING_DEG = 45.0              # the posture 'sitting': the trunk within this of vertical (parent_motion Child.posture; ours), and the catch
                                # line of a sit held by the forearms (C237)
HAND_SIT_MAX_DEG = 40.0         # C237: a child sitting held by its forearms leans forward (a baby holding a parent's hands; tripod sitting): the
                                # held sit is kept with the trunk within this (ours), steadied within PROP_MAX_DEG, caught past SITTING_DEG
PROP_MAX_DEG = 30.0             # the prop engages only with the trunk within this of vertical (A9)
PROP_EASE = (1.0, 0.7, 0.4, 0.2)    # the prop's cap steps (x CAP_TWO), then hovering (A9)
PROP_STEADY_DEG = 20.0          # the trunk within this of vertical ...
PROP_STEADY_TICKS = 10          # ... for this many ticks steps the cap down (A9)
PROP_WATCH_TICKS = 200          # C238: hovering over a child that sits by itself, she watches it this long (30 s; ours: a parent spots a new
                                # sitter half a minute, the catch hers all the while) before the prop is done and she goes on
HOVER_M = 0.05                  # m: the hovering hands' gap (A9)
CATCH_DEG = 35.0                # the catch: the trunk past this from vertical ... (A9)
CATCH_HEAD_MPS = 0.5            # ... or the head dropping faster than this (m/s) (A9)
REACTION_TICKS = 2              # her reaction: 300 ms, a person's; never changes (A9, C6)
LAY_BACK_S = 2.0                # s: laying the child back gently eases her hold to nothing over this (ours)
# The catch's springs pull at once, up to her brief caps while she has them (else her sustained ones), toward where the trunk sat
# most upright while she propped it (a person catching pushes back; she does not merely stop where her hands met it); they pull
# at most BRIEF_S, and a trunk not back within PROP_MAX_DEG by then is laid back gently. C6 counts a fall as caught when it is
# stopped short of CATCH_STOP_DEG (back within PROP_MAX_DEG, or held until she lays it back) and neither its trunk, head nor
# pelvis strikes the mat, the floor, the furniture or a toy at F_pain (A12's 10 ms filter) while it falls, never counting its own
# links pressing each other (tools/sim_parent_motion.py catch, Strike)
CATCH_STOP_DEG = 60.0           # a fall stopped short of this is caught (ours: past 60 deg its upper body lies closer to the mat
                                # than to upright, and its head is within about 0.2 m of the mat)

# ---------------------------------------------------------------------------------------------- her contacts with the child (A4)
SOFT_SOLREF = (0.02, 1.0)       # her collision shapes' contact: MuJoCo's default time constant (the G1's own), taken by every
                                # contact with the G1 through contact priority SOFT_PRIORITY (the world's own mechanism, A21), the G1
                                # untouched. A4's 0.05 ("soft, like flesh and cloth") softened a kinematic body of infinite mass; her
                                # body now gives by itself: at 0.05 her hand sank 9 mm into the child landing on it at 26 N, and a
                                # babbling foot 39 mm into her kneeling thigh at 1.6 kN (10 ms mean), where at 0.02 the worst of the
                                # same 8 runs was 0.9 kN (the physical build, 2026-09-25): a design change the lead is asked to ratify
HAND_SOLREF = (0.006, 1.0)      # her palm's, fingers', thumb's and hand capsule's contact: three physics steps (MuJoCo's guidance: at
                                # least two). MuJoCo's contact is stiff in acceleration, so in force its stiffness scales with the
                                # pair's effective mass, and a light hand sinks deepest: at 0.02 a babbling hip sank her fingertip 8.3
                                # mm, at 0.006 3.2 mm (seed 18 at p_rest 0.6, the carried build, 2026-09-25). At 0.006 MuJoCo's
                                # contact (20 / 0.006^2 per kg at its impedance 0.95) has ISO/TS 15066's hand stiffness, 75 N/mm (its
                                # Table A.3), at 0.135 kg, a hand's effective mass at its fingers
# (Tried: her shapes' impedance rising to 0.99 at 4 mm in, against MuJoCo's contact scaling to the pair's mass, under which a heavy
# link of the child's sinks her light fingertip 17 mm at 25 N: the G1's pinch points (its hip's and knee's links closing on her hand
# or face) then squeezed at 2 to 30 kN where the default let them sink at 1 to 2 kN; kept at MuJoCo's default, 2026-09-25)
SOFT_PRIORITY = 2
# SHE IS A BODY (the lead's decision of 2026-09-25): the child's push moves her as far as her strength, her mass and her footing let
# it, a blow included; no rule moves her. The kinematic parent's blow, give and rise rules (the W2 fix's, never ratified) are gone.
# What stays is A4's care, on her plan:
YIELD_STEPS = 2                 # a contact over the act's cap for 2 physics steps stops that chain where it is (A4) ...
YIELD_M_PER_TICK = 0.02         # ... and, while it still presses, her plan for it is where the child pushed her, backing off 2 cm
                                # a tick along the contact (A4)
YIELD_MAX_M = 0.12              # m: backed off this far in one push and still pressed on, she gives the act up (ours: a hand's
                                # length)
YIELD_BACK_M_PER_TICK = 0.01    # m a tick: a chain her plan backed off comes back at half the yield's rate once the child has left
                                # it alone for her reaction time (REACTION_TICKS), and only where it stays CLEAR_M from the child
                                # (A4: "the act resumes when the force is gone"; ours)
# Her body backs off on the floor plan away from the child's centre of mass, turned toward the contacts' own way out as far as they
# agree (the W2 verifier's seed 6 had two contacts pointing opposite ways, a leg between her shins, flip her way out every step);
# her arms along their contacts (an arm low by the floor drawn out along it); never into furniture (along it)
DRAW_IN_MAX_M = 0.35            # m: a pressed arm draws its hand in toward her shoulder at most this far in one push (ours: from
                                # her arm's reach, 0.6 m, to 0.25 m from her shoulder)
YIELD_LEAN_DEG_PER_TICK = 10.0  # deg a tick: pressed on her body as she leans over the child, she straightens up this fast (her
                                # knees on the floor do not carry her back: her trunk does; ours: a babbling G1's straight leg had
                                # pinned her leaning face at 1.5-2 kN for 2 s while her base backed off 2 cm a tick)
PATIENCE_TICKS = 40             # ticks: an act held up this long in a row by the child pressing against her is given up (6 s; ours)
HAND_CLEAR_M = 0.003            # m: her hand planned this far outside the G1's collision surface (its convex hulls) where it rests
                                # on it or holds it (ours); her palm, fingers and thumb touch the child as the rest of her does, so
                                # they never pass into it
HAND_FREE_M = 0.01              # m: a hand this near the link it reaches for has arrived on it (ours)
SETTLE_ABOVE_M = 0.05           # m: a toy handed into the child's palm is held this far above it first ... (ours)
SETTLE_TOL_M = 0.015            # m: ... until her real hand is within this of where she means it ...
SETTLE_TICKS = 6                # ... at most this long (0.9 s) ...
SETTLE_TOY_TICKS = 24           # C223: ... and this long (3.6 s) when it is the toy she settles over the place (its hang off her grip is aimed out
                                # by sight before it is lowered; ours)
SETTLE_LOWER_TICKS = 4          # ... then lowered into the palm over this many ticks (0.6 s; ours: at a physical arm's speed its
                                # fingers met the toy first and the grasp closed on nothing)
AIM_GAIN = 0.5                  # her aim at a place moves this share of what her real hand still misses there, a tick (ours) ...
AIM_MAX_M = 0.10                # ... at most this far from the place (ours)
AIM_MAX_TOY_M = 0.20            # C223: ... and this far when it is the TOY she aims (a carried toy hangs up to 14 cm off her grip, and her hand's
                                # rotation error moves it by as much; ours)
ARRIVE_WAIT_TICKS = 14          # ticks: her hand not yet on the link as its reach's time ends is waited for this long before the
                                # hold is refused (2.1 s; ours: her hand is a body, and follows its plan a moment behind). C166
                                # (2026-09-30): 4 ticks (0.6 s) until life day 44, when the body unfrozen by A143 began to wriggle under
                                # her hands: 15 of her 22 turns that day refused "did not arrive on it (13 to 46 cm off: it moved)",
                                # none done, the child prone a third of the day. Her reach tracks the link each tick (the hand target
                                # of kind "link" is resolved at the tick's end), so the wait is the chase: a person following a
                                # wriggling baby's shoulder keeps after it for a couple of seconds before giving up
HOLD_SHAPE = (0.10, 0.0)        # her hand's (curl, thumb) on the child: open and flat, the fingers along its surface (a curled
                                # grip's fingers passed 3.6 cm into its torso: the W2 verifier's finding; ours)
TRUNK_STILL_M = 0.01            # m: her trunk holds still (the report's trunk field, P3) while her chest moves less than this ...
TRUNK_STILL_DEG = 2.0           # ... and turns less than this over a tick (ours: a body at rest sways less; a lean or a push more)
FACE_CHILD_DEG = 60.0           # deg: her trunk faces the child while her chest's forward lies within this of the way to its torso
                                # on the floor plan (ours: kneeling beside it or at its feet, the child lies in front of her)
HER_PAIN_N = 150.0              # N: her own pain, on any segment, as a 10 ms mean (4.10; a person's, ours)
# WHAT A PERSON'S BODY BEARS AT ITS CONTACTS (the lead's babble criteria b and e, A25b): ISO/TS 15066:2016 ("Robots and robotic devices:
# collaborative robots"), Annex A, read 2026-09-25 from the standard's text. Table A.2, the biomechanical limits (the pain onset of
# 100 healthy adults, force values from a study of 188 sources; transient contact, where the body part can recoil, at least twice
# the quasi-static): hands and fingers 140 N, lower arms and wrist joints 160 N, upper arms and elbows 150 N, chest 140 N, abdomen
# 110 N, pelvis 180 N, thighs and knees 220 N, lower legs 130 N, skull and forehead 130 N (no transient allowed), face 65 N (none).
# Table A.3, each region's effective spring constant: hands and fingers 75 N/mm, lower arms 40, upper arms 30, chest 25, abdomen
# 10, pelvis 25, thighs and knees 50, lower legs 60, face 75, skull and forehead 150. Her contacts with the child are held to them:
HAND_N = (140.0, 280.0)         # N: her hand's contact force on the child, a tick's mean (quasi-static) and a 10 ms mean (transient)
FOREARM_N = (160.0, 320.0)      # N: her forearm's, the same
DEPTH_M = dict(hand=280.0 / 75e3, forearm=320.0 / 40e3, upper_arm=300.0 / 30e3, trunk=280.0 / 25e3, legs=440.0 / 50e3,
               head=130.0 / 150e3)   # m: how far a region of hers is ever pressed in by the child: that region's compression at its
                                # transient bound (its skull's at its quasi-static one: no transient contact there), 3.7 mm at her
                                # hands, 8.0 at her forearms, 10.0 at her upper arms, 11.2 at her trunk (the chest's), 8.8 at her
                                # legs (the thighs'), 0.87 at her head (ISO/TS 15066 Tables A.2 and A.3; the lead's "no penetration
                                # deeper than the solver's margin", read as the depth a person's body gives at its bound)
TRUNK_STANDOFF_M = 0.0          # m: her trunk and head (their shapes as built) never touch the child's body (its trunk's links), as the
                                # physics has them at four steps a tick (A25b: her plan keeps CLEAR_M; this is what her carried body
                                # does with it) ...
REST_TICKS = 2                  # ... and never touch the child at all for more than her reaction time in a row (REACTION_TICKS: a
                                # touch longer than that is her resting on it)
HER_PAIN_STEPS = 5              # the 10 ms mean: 5 physics steps (the child's pain filter's length, A12)
WITHDRAW_M = 0.12               # m: a hand withdrawn when hit draws back this far (ours)
SUPPORT_BEND_DEG = 45.0         # leaning (lean + spine) this far, her free hand rests on her own thigh (a hand hanging from a
                                # far lean reached down onto the child; the floor beside her knee was where its arm lay: W2; ours)
SUPPORT_OFF_DEG = 35.0          # ... and lets it hang again once she straightens past this (ours: 10 deg under, so it does not
                                # flicker at the edge)
OFFER_HOLD_TICKS = 40           # ticks: her open hand held out for a toy (the give ask's level 0), or her hands over her face
                                # (peekaboo), wait at most this long for the next act (ours: A4's own patience with the child's
                                # hand, 40 ticks for a hand-over's release); the act is under way while they wait (P3's contract)
GESTURE_HOLD_TICKS = 4          # ticks: a hand held out (an open hand, a raised arm, a fist) stays this long before it rests
                                # (ours: 0.6 s, long enough to be seen at the child's 150 ms tick)
DO_WALK_M = 1.0                 # m: 'do walk' (a verb's showing): a few steps, about three strides, away and turned back to the
                                # child (ours)
IDLE_RELAX_TICKS = 10           # a hand left out by a finished act relaxes after this long unless it holds something (ours)
CLEAR_M = 0.03                  # m: her legs, trunk and head kept this far from the child in every planned frame (A4), and each tick
                                # from the child's body where it is now (the lead's standoff, A25b: parent_motion._standoff) ...
STANDOFF_M_PER_TICK = 0.0375    # ... her base moved back off it this far a step (her knee shuffle's pace, SHUFFLE_MPS x a tick) ...
STANDOFF_STEPS = 4              # ... at most this many steps a tick (15 cm; ours)
# THE CALM STEP (A25b; C8: "her kneeling spot outside its leg sweep"; a parent beside a kicking baby waits for it to settle): beside the
# child she moves her legs (a step, kneeling down, a shuffle on her knees, a turn on them) only on a tick when no shape of the child's
# within CALM_NEAR_M of her legs moved more than CALM_MOVE_M since the last tick; otherwise her base waits where it is, carried, at
# most CALM_WAIT_TICKS in a phase, then goes on (measured on the carried build, seeds 1-5 at p_rest 0.3 and 0.6: most ticks over a
# joint's limit from her were its hands and feet meeting her shins, thighs and feet as she walked, knelt down or shuffled beside it;
# waiting without end, she did two acts in 600 ticks)
CALM_NEAR_M = 0.30              # m: its shapes this near her legs (a babbling limb's step in a tick, 0.27 rad at its hip or shoulder,
                                # moves its hand or foot about 0.15 m: twice that; ours) ...
CALM_MOVE_M = 0.02              # ... that moved this far in the last tick (about 0.13 m/s: a limb at rest drifts less; ours) ...
CALM_WAIT_TICKS = 20            # ... she waits for, at most this long in a phase (3 s; ours)

# ------------------------------------------------------------------------------------------------------------- her paths (A6)
GRID_M = 0.05                   # the floor grid of her paths (A6)
CLEAR_FURNITURE_M = 0.15        # a path keeps this far from furniture and walls (A6)
CLEAR_CHILD_M = 0.25            # ... from the child's body (A6: outside its leg sweep)
CLEAR_TOY_M = 0.08              # ... from toys (A6)
CLEAR_CHILD_TIGHT_M = 0.10      # m: where the child's limbs close every way at A6's clearance (a babbling G1 in the middle of the
                                # mat: the W2 verifier's finding 6), she threads past them this far from its shapes, watching her way
                                # as ever (ours; a departure from A6's 0.25 m, for the lead to ratify)
BODY_R_M = 0.20                 # m: her own half-width on the floor plan, walking (ours: a woman's hip breadth, about 0.37 m)
APPROACH_CHILD_M = 0.8          # m: her last steps to a spot beside the child, within this of it, come no nearer the child than the
                                # spot itself (the spot lies within A6's walking clearance, a step back from where she kneels; the
                                # babbling child's limbs had closed every way to it: the W2 verifier's finding 6; ours)
SLOW_NEAR_CHILD_M = 0.7         # m: walking with her centre this near the child's shapes she goes at half WALK_MPS (ours: a person
                                # steps carefully by a baby; a foot swung at her full pace into a babbling limb that moved into
                                # her way did 5 J of work on it, the physical build's first babble runs)
WALK_MPS = 0.8                  # walking (A6); parent_poses.walk's gait runs at its own SPEED and is timed to this
SHUFFLE_MPS = 0.25              # shuffling on the knees (A6)
TURN_DEG_PER_S = 180.0          # turning in place (ours: a half turn in a second)
KNEEL_DOWN_S = 2.4              # s: standing to sitting on her heels at the least (ours: parent_poses.kneel_down's three parts)
STAND_UP_S = 2.4                # s: the way back up at the least (ours)
KNEEL_PELVIS_MPS = 0.5          # m/s: her pelvis's fastest kneeling down, sitting back onto her heels or getting up (A25b: her body is
                                # carried, and a carried body starts and stops only as fast as its support lets it: 88 N over her
                                # weight stops a 0.46 m/s drop in 7 cm, and the first carried build's drop onto her heels hit the floor
                                # at 1.6 kN through her thighs). Ours: of the order of the centre of mass's vertical speed in rising
                                # from a chair, about half a metre a second, recalled; her kneeling down eased in and out over a
                                # fifth of its time at each end, its fastest moment at this pace (parent_motion.KNEEL_TIME, _ease)
KNEEL_SEG_MPS = 0.8             # m/s: no segment of hers moves faster than this kneeling down or getting up (ours: her stepping foot
                                # and her knee coming down set the pace; with parent_poses.kneel_down's own path, 3.1 s all the way)
KNEEL_OFF_M = (0.72, 0.78)      # m: her kneeling spot from the G1's torso centre line, beside its chest (A6)
KNEEL_OFF_TRY = (0.75, 0.72, 0.78, 0.84, 0.90)   # the offsets she tries in turn: A6's range first, then a little further out when the
PLACE_SPOT_R = (0.50, 0.60, 0.42)   # C260: a put planned anew for a place her side spots cannot reach kneels round the PLACE itself: her
                                    # pelvis this far from it, facing it, beyond the place from the child first, then round it by 45 deg; a
                                    # floor point half a metre ahead of a heels kneel is inside the put's reach and clear of her knees (ours:
                                    # A6's range for a floor reach from the heels)
STAY_SIDE_M = 1.1               # m: kneeling within this of a supine child's middle, she keeps her side of it for the next approach
                                # (A112, C96: _beside_now's own distance; a person does not walk round a baby that rolled a little)
                                # child's arm lies where her knees would go (W2: its arms sink and move out from the birth pose)
KNEEL_ALONG_M = 0.10            # m: ... toward its feet from its trunk's middle (section 4.2's measured spot)
HIPS_ALONG_M = (0.30, 0.40, 0.20)   # m toward its feet from its trunk's middle, tried in turn: beside its hips, where she kneels
                                # to bend its far knee over (A8) or to take both its forearms (A9): from beside its chest the
                                # far knee lies beyond her reach (W2: 12 cm short). Ours
GATHER_UP_M = 0.25              # A165: the gathering point for the pull-to-sit, this far above its chest (where one trunk of hers holds both forearms; ours)
GATHER_MAX_M = 0.40             # A165: a forearm drawn at most this far toward the gathering point in one gather (ours; the guide's pace sets the ticks)
SIDE_SHOULDER_DZ_M = 0.12       # A170: a child is on its side for the turn when one shoulder stands this much higher than the other (a rocking
                                # prone child reads 'side' for a tick with its shoulders level, and keeps the face-down grips; ours)
GATHER_RETRIES = 2              # A166: a gather whose hand did not arrive (the arm moved) is planned again where the arm is, this many times (ours)
PULL_TIDY_M = 0.70              # C244: toys lying within this of the point beyond its feet where she kneels for the pull are set aside first when no
                                # spot is free of them (the kneel-down, its knees' way and her step back lie within it; ours) ...
PULL_TIDY_TOYS = 2              # ... at most this many a pull asked (ours)
PULL_FEET_M = (0.35, 0.40, 0.45, 0.50, 0.55)   # m beyond its feet (0.70 m from its pelvis along it) where she kneels on her
                                # heels for the pull-to-sit, facing its head (A9: "she kneels at the G1's feet and holds both
                                # forearms"), nearest first: her knees about 0.34 m ahead of her heels' spot, so from just beyond
                                # its soles; a spot is taken only where every kneeling frame keeps 3 cm from it (A4) and one trunk
                                # of hers reaches both its forearms (C7). Ours (the W2 verifier's finding: W2 had knelt beside its
                                # hips, which A9 does not say, and refused even there)
KNEEL_CLEAR_FRONT_M = 0.6       # m: kneeling down needs this clear in front, so she kneels a step back and shuffles in (4.1)
KNEEL_FOOTPRINT_R = 0.35        # m: toys within this of her knees' line where she kneels are cleared first (ours)
TOY_ASIDE_M = 0.35              # m: a cleared toy is set down this far beyond her footprint, away from the child (ours)
TRUNK_DEG_PER_S = 30.0          # deg/s: her trunk's unhurried lean, down or up (ours; the lean phase's pace, and her return to
                                # upright when her hands come back to rest: at the old fixed 4 ticks a 67 deg lean came back at
                                # 110 deg/s and her arm swung 0.6 m in a tick, W2)
REACH_MPS = 0.5                 # m/s: her hand's unhurried reach (ours)
REACH_MIN_S = 0.45              # s: the shortest hand move (ours: 3 ticks)
OVER_CHILD_M = 0.15             # m: a hand crossing over the child goes this far above its shapes' tops (ours: her forearm hangs
                                # below her hand; at the kinematic parent's 8 cm her physical forearm dragged over its knee)
APPROACH_M = 0.10               # m: a hand comes onto the child, or a toy, along the surface's normal from this far out (ours)

# ------------------------------------------------------------------------------------------------------------ her face (A3, A22)
FACE_MIN_M = 0.25               # her face never closer than this to the child's eyes (A3)
FACE_ON_LINE_DEG = 8.0          # A94: when she smiles at an act her mouth goes ONTO its fovea's line, within 8 deg (inside the 64 px window
                                # at 3 px a degree: about 10 deg to its edge), the en-face position parents take (Stern 1974)
FACE_OFF_LINE_DEG = 15.0        # when she leans in or calls, her face arrives at least this far off the fovea's line (A3)
LEAN_DIST_M = (0.30, 0.60)      # m: where she puts her face when she leans in: from the child's eyes (ours: the lean-in
                                # distance, where the grades are visible in its fovea, section 4.3). She kneels where a pose inside
                                # human ranges puts it there (beside its chest or its shoulders), the least bend first; never
                                # widened after measuring
LEAN_MAX_DEG = 70               # her kneeling lean's range in her searches (hip flexion kneeling tall; ours: parent_kin's range)
LEAN_ALONG_M = (-0.10, 0.0, 0.10, -0.20)   # m toward its feet from its trunk's middle: the spots she leans in from, beside its chest
                                # and shoulders first (her face within LEAN_DIST_M at the least bend there, W2: a lean of 50 deg
                                # beside its shoulders, 60 beside its chest; ours)

# ------------------------------------------------------------------------------------------------------ her plans (4.1: "W2")
REPLAN_TICKS = 7                # an act's plan is solved when it starts and again about once a second (7 ticks, 1.05 s) while
                                # its target moves, warm-started from the last solution, interpolated in between (4.1)
REPLAN_MOVED_M = 0.02           # m: the target has moved when it is this far from where the last solve put it (ours)
FOCUS_TICKS = 3                 # ticks: a look "during the focus word" holds on the object this long, then the next look (ours:
                                # a focus word lasts 2-3 ticks at her 3.6 words a second, 4.4)
LOOK_TICKS = 2                  # her gaze reaches its target within 2 ticks (4.10's L1)
SHOW_SHAKE_M = 0.02             # m: a shown toy is shaken this far either way ... (4.10: "shaken for its sound"; ours)
SHOW_SHAKE_HZ = 2.5             # ... at this rate (ours)
SHOW_DIST_M = 0.40              # m: a toy shown before the G1's eyes (4.2, 4.10)
HANDOVER_PALM_N = 0.3           # the hand-over's release: the child's palm touch at least 0.3 N ... (A4)
HANDOVER_CLOSED_DEG = 30.0      # ... and its fingers closed at least 30 deg (A4) ...
HANDOVER_HOLD_TICKS = 2         # ... for 2 ticks, ...
HANDOVER_MAX_TICKS = 40         # ... or after 40 ticks (A4)
HANDOVER_PRESS_M = 0.005        # C222: the toy lowered this much further into the palm a tick (3 cm/s) while she feels no palm on it ... (ours)
HANDOVER_PRESS_MAX_M = 0.06     # ... at most this far past where she planned it (the uncertainty of her reading of its palm; ours)
HANDOVER_OUT_M = 0.02           # C222: a toy is within the hand's reach when its centre lies no further out along the palm's normal than its own
HANDOVER_CLEAR_M = 0.01         # C232: a hand is clear for a toy when its grasp point stands the toy's half-extent along gravity plus this over the floor (ours)
                                # half-extent plus this beyond the grasp point (a fist's knuckles stand 5 to 6 cm out of the face; ours)
HANDOVER_AT_M = 0.04            # C240: the offer's press runs only while the toy is within this of its place at the palm (the toy's own miss,
                                # _aim_fix, less what she has pressed it); off it, she follows the hand (A4's forty ticks run on) ... (ours)
HANDOVER_FOLLOW_TICKS = 20      # C240: ... and off it this long (3 s) with the hand beyond her reach from where she kneels, she kneels again
HANDOVER_REPLANS = 2            #       where it is, at most this many times in a hand-over (ours)
HANDOVER_PALM_RISE_MIN = -0.2   # C240: a palm whose normal's rise is at least this (level, or facing the floor by no more than 12 deg: the toy
                                # pressed along its normal, the fingers close on it; the born child's palms lie level, rise -0.03) is given to
                                # before one facing the floor further (the copy's -0.30 and -0.62: the toy pressed onto the back of the hand) (ours)
TAKE_BACK_N = 0.3               # taking a toy back: she closes only after the child's palm force stayed under 0.3 N ... (A4)
TAKE_BACK_TICKS = 2             # ... for 2 ticks

# ---------------------------------------------------------------- C268 (2026-10-04): the child stood up, held, and walked to her
STAND_CHEST_M = 1.06            # m: its upper chest's height standing (the G1's; the held point's goal over its feet)
STAND_PELVIS_M = 0.70           # m: its pelvis above this with the trunk within STAND_DEG is standing (ours; the G1's stands at 0.79)
STAND_DEG = 20.0                # deg: the trunk within this of vertical counts as upright in her hands (the prop's steady line)
RAISE_MPS = 0.08                # m/s: her hands lead its chest up and over its feet at this speed (ours; the day-84 copy: on its feet
                                # in 9 s on 100 to 135 N with its own legs' thrust, A193)
RAISE_MAX_TICKS = 260           # a raise that has not stood it in this long (24 s) is given up and it is laid back (ours)
STAND_HOLD_TICKS = 40           # she steadies it standing this long (6 s) before the walk or the sitting down (ours)
WALK_LEAD_M = 0.18              # m: her hands lead its chest this far ahead of its feet's middle (ours; C274: 0.10 until then, the stance hip then
                                # extended only to about the swing's trigger and steps were rare)
STAND_WALK_MPS = 0.04                 # m/s: and no faster than this (ours; the copy: 0.78 m in 45 s)
STAND_FOLLOW_MPS = 0.30 # m/s: her hands go with its chest this fast once its feet are ahead of them (C274; ours)
ROCK_M = 0.10 # m: its chest led this far to the side of its feet's middle, over one foot then the other (C274; ours: about half the distance between its feet)
ROCK_TICKS = 10 # ticks a side (1.5 s; ours: a swing takes 7)
ROCK_MPS = 0.15                 # m/s: the sway's pace (ours)
WALK_TURN_M = 1.0               # m: farther than this from the mat's centre, the held walk turns toward it (C274; ours)
STAND_TURN_DPS = 3.0            # deg/s: the walk's way turns this fast (ours: a quarter turn in half a minute; at 15 it swung out of her
                                # hands, at 6 it turned on past the mat; the day-85 copy)
WALK_STOP_M = 0.45              # m: its feet this near her knees, the walk is done (ours: her own reach's comfort)
WALK_MAX_TICKS = 400            # a walk's longest (60 s; ours)
STAND_FALL_M = 0.50             # m: its pelvis under this while she holds it standing: it has sunk; she lowers it (ours)
LOWER_MPS = 0.10                # m/s: her hands lower its chest to sitting height at this speed (ours)
SIT_CHEST_M = 0.45              # m: its upper chest's height sitting on the mat (the G1's)
STAND_SHUFFLE_MPS = 0.40        # m/s: her knee-shuffle beside the child while she raises and walks it (ours: the shuffle's own pace)
STAND_LEAD_M = 0.15             # m: her hands' led point at most this far from its chest (ours: twice the cap's stretch at HOLD_K)
STAND_CAP_SHARE = 0.8           # of the child's weight, both hands together, while she raises it to its feet (ours; see _ctl_stand)
WALK_FAR_M = 1.0                # m: she walks it this far, then sits it down (ours)
SIT_BACK_M = 0.55               # m: its chest is led this far behind its feet as she sits it down (the G1's sitting: its pelvis behind its heels)
LOWER_MAX_TICKS = 120           # the sitting down's longest (18 s; ours)
WALK_STALL_TICKS = 80           # a walk with no 2 cm of progress in this long (4.5 s) is over (ours)
