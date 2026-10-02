"""THE PARENT'S MOTION (docs/SIM_DESIGN.md 4.1, 4.2, 4.10's L0 and the motor side of L1-L2, A3, A4, A6-A10, A22, A25; the build
plan's W2). How her body carries out what her conduct asks, on the world's side. SHE IS A BODY (the lead's decision of 2026-09-25):
her 16 segments are dynamic MuJoCo bodies with a woman's masses (body/sim/parent_body.py, make_g1room.py), her planner here makes
the pose she means at each tick's end (parent_kin's forward and two-bone inverse kinematics and look-at, inside human joint ranges),
and her joints are driven toward it with a woman's strength; every contact with the child is the physics'. A person's strength:
she comes to the child; she can never lift, slide or sit up the 34 kg G1 (the owner's B16); it must rise by itself.

THE INTERFACE her conduct calls: P3's contract as StubMotion writes it (body/sim/lang/conduct.py on sim-parent, 1387a44). An act is
anything with `kind`, `target`, `during` and `thing` (conduct.Act). request(act, tick) -> id; status(id, tick) -> exactly
'running' (under way, or waiting its turn), 'done', 'refused' or 'cancelled'; cancel(id, tick) (from that tick: at once for the
world's next tick, later for a later one); report(tick) -> dict(eyes, head, left, right, trunk, face, acts), her attention log's
fields (A51) for the tick just lived: where her eyes, head and each hand physically point (a toy's id, a place,
'child_eyes' / 'child' / 'child_periphery', the part of the child a hand touches by its word, 'mama' for her own face, '?' at
nothing she can name, None at rest, her hands resting on her thighs or hanging), her trunk ('child' while it faces the child and
holds still, the place she walks to, '?' while it leans, turns, shifts or is pushed), her face ('mama' on a tick her expression is
not its neutral set or moves, her jaw with her speech and her blinks apart; None while still), and {id: status} for every act asked
of her until it has been reported ended; bind_conduct(conduct) (she reads its eyes_on_child and its still every tick: while an ask
is pending her eyes and head stay on the child's eyes, L1 glances at nothing, and no act starts but PENDING_OK's, the others waiting
their turn; while still, a formal trial's settle and window, no act starts at all, her hands go to rest on her thighs and stay, and
her face is drawn in its neutral set but her jaw) or set_eyes_on_child(flag) / set_still(flag); glance(target, tick) (L1's gaze to a
sudden event); state() / load_state(s); why(id), the reason an act was refused or how it ended. An act stays under way while it
directs the child's eyes (a toy shown, a hand held out, a point) and ends as her body comes back. Acts run on two channels: her gaze
("look") and her body (every other kind), each in the order asked; an act that cannot be done is refused with its reason, never
faked. KINDS lists every kind with what it does; the task's motor intents map onto them (approach, kneel = "attend", lean in =
"lean_in", hold up a toy = "show", hand over = "hand_over", guide a forearm = "guide", prop = "prop", the brief capped turn =
"turn", bring back what rolled away = "bring_back", point = "point", leave = "walk" to "door", return =
"walk" to "child", sofa = "walk" to "sofa"). DOES (a class attribute, read by P3's conduct and templates.showable) lists what 'do'
carries out ('show' and 'pick_up' on the toy the Act's thing names); 'copy' makes the movement P3's conduct copies (A52:
'kind:side').

L0, EVERY PHYSICS STEP (the world's apply calls before_step and after_step around each of its 75 mj_steps):
  - HER MUSCLES drive her joints toward the tick's targets, from her plan at the last tick's end to her plan at this one's, each
    joint axis one MuJoCo actuator whose force range is her strength there, damping included (parent_body.Drive; parent_consts.
    STRENGTH, a woman's, per joint and direction); HER TRUNK IS CARRIED (A25b): her pelvis, abdomen and chest held toward her plan
    by a support capped at SUP_F_MAX (never down) and SUP_T_MAX, her weight carried at her centre of mass, giving way to the
    child's push on her body and never letting her weight rest on it; her feet, knees and shins meet the floor by contact;
  - EVERY HOLD ON THE G1 IS A CAPPED SPRING (4.1, 4.2, A25), never a weld: a force at the held point, K (where she means the
    point to be - the point) + C (its velocity - the point's), clipped at the hold's own cap (a resting hand's weight, the guide's, the prop's step), then all her holds together within what
    HER caps leave after her body's own contacts (one hand 100 N sustained, 150 N for up to 2 s; both hands 156 N, 200 N for up to
    2 s: body/sim/parent_consts.py, checked against their sources), applied as an outside force on that link at the held point,
    and its reaction on her own hand at her grip, her arm and trunk exerting it within her strength (Newton: she feels what she
    pulls). Her hand is planned on the held point, open and flat on its surface; her palm, fingers and thumb touch the child as
    the rest of her does, so they never pass into it. Each hold's force is added to the held link's touch (4.2: being held is
    felt), under the same pain law as any force (every cap is far under F_pain). A hold whose held point has left her hand by a
    hand's length (HOLD_SLIP_M) has slipped;
  - HER CONTACTS WITH THE CHILD are the physics' (her shapes at MuJoCo's default contact, solref 0.02, at contact priority 2 so
    hers is the contact's, the G1 untouched: A21's mechanism): the child pushes her body, and her body gives as far as her
    strength, mass and footing let it; no rule moves her. A4, her care, on her plan: per chain (each arm, and the rest of her), a
    contact force (normal and friction) over the act's cap (a resting hand's weight for her body; an arm reaching onto or holding
    the child, its hold's cap, and its hand on the held link past that cap) for 2 physics steps STOPS that chain where it is (its
    joints' targets become their positions and its tone lets go, so she presses no further; her legs keep the tone that carries
    her own weight), and an arm stopped so stops her trunk too while her plan moves it (a lean, her base),
    which holds there while the arm is pressed. While it still presses: a chain she moves (her gait, a lean, a reaching arm) backs
    off 2 cm a tick along the contact (her body on the floor plan away from the child's centre of mass, turned toward the
    contacts' own way out as far as they agree, never into furniture, and her trunk, bent over the child, straightens 10 deg a
    tick; an arm pressed on its upper arm or forearm, or with its way out barred, draws its hand in toward her shoulder), up to a
    hand's length (backed off that far, her body gives the act up); a chain she holds still stays: her body holds the pose it had
    as the push began and her plan is where the push put her, an arm goes where the push has put it, so she never springs back
    into the child; a chain resting some of her weight on the child moves off it. The act waits while a chain it moves is
    pressed, and is given up after PATIENCE_TICKS; it resumes when
    the force is gone, the chain coming back only where it stays 3 cm clear of the child. A plan never runs ahead of her body:
    after a stop, she reaches the planned pose again before the plan goes on. Walking, she watches her way (where the child has
    moved within A6's clearance of it she finishes the step under way and plans the trip again) and beside it walks at half her
    pace. Her body never pushes the child: only her holds do, each within its cap. She sees where her hand is: a hand held at a
    place is aimed where her real hand arrives there (_aim_fix), and waits a moment to arrive before a grasp or a hold;
  - HER PAIN: a segment's 10 ms mean contact force from the G1 over 150 N is a hit (4.10), logged with its tick for her conduct.
L1, EVERY TICK (tick_begin): her planned pose is raised off the floor (_floor_lift) and kept clear of the child where it is now (her
standoff, _standoff: her legs, trunk, head and free arms 3 cm from its body and legs); beside it she moves her legs only while its
limbs near them are still (the calm step, _restless_near); her acts advance, each a list of phases (walk, turn, kneel down, stand up, shuffle on the knees, lean,
hand moves, grasps and releases of toys, holds), from which her pose at the tick's end is computed: her base (standing, walking,
kneeling on her heels or tall, sitting) by parent_poses, each hand by two-bone IK to its grip point, her head and eyes by look-at,
her face from her feelings (set_face: parent_feel's graded face, P3; drawn only when it changed, 24 ms a drawing). Plans are solved
when an act starts and re-solved about once a second (REPLAN_TICKS) while its target moves, warm-started from the last solution
(the search begins at the last lean and spine and widens), the trunk interpolated from the old solution to the new.

HER PATHS (A6): A* on a 5 cm floor grid, keeping 0.15 m from furniture and walls, 0.25 m from the child's body, 0.08 m from toys
(beyond her own half-width); where the babbling child's limbs close every way at 0.25 m, 0.10 m (CLEAR_CHILD_TIGHT_M), and within
APPROACH_CHILD_M of her spot beside it no nearer the child than the spot itself; a toy across her only way is picked up and set aside; walking at 0.8 m/s, shuffling on her knees at
0.25 m/s, kneeling down eased in and out at a person's pace (KNEEL_TIME: her pelvis at most KNEEL_PELVIS_MPS, every other segment
KNEEL_SEG_MPS). She kneels beside the child's chest on the side it faces
(0.72-0.78 m from its torso's centre line, then 0.84 and 0.90 m where its arm lies in the way), then its other side, its head and
its feet; she kneels down a step back where the child is nearer than her step ahead and shuffles in, or, with no room behind her,
kneels down along it and turns on her knees to face it; she checks again before she shuffles in and stops short if it moved. Toys
where she will kneel are cleared first (a toy the child touches is its own: A4), set aside within her reach away from it. When no
spot is free the act is refused (she never moves the child to make room, A6); a spot is taken only where the act can be done from
it (her hand reaches its trunk, or where only its feet leave room its leg; her face reaches its view within LEAN_DIST_M). Every
planned kneeling frame keeps her legs, trunk and head 3 cm from the child (A4), checked by MuJoCo's own geometry (mj_geomDistance)
on a scratch copy of the state, and checked again as she kneels down (its arms spread as they sink). Getting up is planned the same
way: up onto her knees, a shuffle back or a turn on her knees where rising there would touch it, then onto her feet; refused
(she stays kneeling) when every way would. Leaning far over, her free hands rest on her thighs.

WHAT SHE CAN DO FOR A 34 KG BODY (4.2, A7-A10, A25): touch (a hand resting, TOUCH_N); guide a forearm (the guide's cap min(1.5 x
the arm's own push at that pose, 100 N), a path of at most 8 ticks at GUIDE_SPEED; stopped after 2 ticks at its cap, so the child
can refuse); the roll's help (the far arm across within 65 N, the far knee over within 76 N, then she lets go); prop within 30 deg
of vertical (easing 100/70/40/20% of CAP_TWO, then hovering; the catch 2 ticks after the trunk passes 35 deg or the head drops
faster than 0.5 m/s; laid back gently past 30 deg after a catch); the brief turn from its front (its near shoulder and hip lifted
straight up, then over, the force growing at 100 N/s, at most 2 s); the pull-to-sit by both forearms (only from where one trunk of
hers reaches both: C7; growing to the brief cap; at the cap for 2 ticks she stops and lays it back gently). None of these supplies
a rise, and none can lift or slide it: each is bounded by her caps (156-200 N against its 337 N weight, and 312-433 N to slide it
on the mat), and tools/sim_parent_motion.py measures each act's peak force against its cap, the G1's rise, slide and trunk, her
reach and her timing.

EXACT DETERMINISM. Nothing here draws a random number; every choice is a deterministic search. state() carries every act, phase,
hold, stop, timer, warm start and her joints' targets as plain numbers (her body itself is the physics', saved with the world), so
a world saved while she acts and restored anywhere continues bit for bit (body/tests/test_sim_parent.py)."""
import heapq
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from . import extras as X                      # A129: the bucket's sizes (BUCKET_H) for the hide game

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import g1scene as G  # noqa: E402
import parent_consts as K  # noqa: E402
import parent_body as PB  # noqa: E402
import parent_kin as kin  # noqa: E402
import parent_poses as P  # noqa: E402
from parent_kin import unit  # noqa: E402

TICK_S, STEPS = 0.150, 75
CHAINS = ("arm_L", "arm_R", "core")
SEG_CHAIN = {s: ("arm_L" if s.endswith("_L") and s.split("_")[0] in ("upper", "forearm", "hand") else
                 "arm_R" if s.endswith("_R") and s.split("_")[0] in ("upper", "forearm", "hand") else "core") for s in kin.SEGS}
GRIP_LOCAL = {sd: np.array([0, -kin.side_sign(sd) * .045, -.10]) for sd in "LR"}   # her grip point in her hand's frame (P.reach)
HAND_SEGS = {"L": 0, "R": 1}
HAND_TOUCH_AFFINITY = 18                    # her palm, fingers and thumb touch what has contype 2 or 16: the room and the (A25c: never the G1)
                                            # floor, never a toy (whose contype is 4; make_g1room.HAND_TOUCH)
BODY_AFFINITY = 18                          # her body's shapes touch the room (bit 16) and the floor's bit 2, never the G1 (A25c; make_g1room.FLOOR;
                                            # A25b: her trunk is carried, so no gait of hers goes through the floor)
MOVING = ("walk", "turn", "kneel_down", "shuffle", "sofa", "lie")   # her base modes that move her base (her plan moves her pelvis)
STEPPING = ("walk", "turn", "kneel_down", "shuffle", "knee_turn")   # the phases that move her legs (the calm step: _restless_near)
HAND_BOX = (0.0, 0.0, -0.08, 0.12, 0.12, 0.15)   # her hand's frame: a box holding her hand in any shape (centre, half-sizes; its
                                            # fingers reach about 0.2 m from her wrist); MuJoCo's midphase tests her hand's shapes in it
FOREARM_TOP, UP_LOCAL = K.FOREARM_HOLD, K.FOREARM_HOLD_N
THIGH_REST = np.array([0.06, 0.0, -0.22])   # her grip on her own thigh (its frame: +x its front, -z toward the knee): the palm on
                                            # its surface (the thigh's 7 cm radius, the grip 4.5 cm off the palm), mid-thigh

# what each kind does (P3's ACT_KINDS, as the motion carries them out; the design section in brackets)
KINDS = {
    "look": "her head and eyes to the target within LOOK_TICKS (4.10 L1); during='focus' holds FOCUS_TICKS, then the next look",
    "lean_in": "come to the child and put her face where its eyes can reach (the head camera's 47.6 deg pitch; A22, C34), at least "
               "25 cm from its eyes and 15 deg off its fovea's line (A3); refused where no pose inside human ranges reaches (A22)",
    "attend": "come to it, kneel beside its chest and attend, one hand resting on its trunk (a hold at TOUCH_N; 4.2)",
    "approach": "come to it wherever it is: walk, kneel a step back, clear toys, shuffle in on her knees (A6, B7)",
    "show": "fetch the toy if she does not hold it, come to the child, hold it SHOW_DIST before its eyes beside her face, shaken (4.10)",
    "point": "point at the target with the hand that can, from where she is (4.2)",
    "open_hand": "an open hand held out before the child, palm up, for a toy (4.10's give ladder, level 0)",
    "hand_over": "fetch the toy, come to the child, bring it into the near hand's palm; release by A4's rule (the palm pressed and the "
                 "fingers closed for 2 ticks, or after 40 ticks)",
    "touch": "a hand resting on the named part of the child (a hold at TOUCH_N)",
    "withdraw": "the hand she was hit on drawn back WITHDRAW_M at once (4.10)",
    "guide": "guide the named forearm along a path of at most 8 ticks within the guide's cap (A10); 'far_arm' across its chest "
             "within 65 N (A8)",
    "knee_over": "the far knee bent over within 76 N, then let go (A8)",
    "pull_to_sit": "both forearms held, the pull growing to her brief cap; at the cap for 2 ticks she stops and lays it back (A9)",
    "prop": "the trunk held where it reached, within 30 deg of vertical, easing and hovering, the catch (A9)",
    "turn": "the brief turn from its front toward its back: springs on its pelvis and a shoulder, growing within the caps, at most "
            "2 s (A7)",
    "bring_back": "fetch a toy that rolled away and set it down within the child's reach (4.10's reach ladder, level 1)",
    "hide": "fetch a toy and let it go into the bucket in the child's view (A129, the hide game on the bucket A126: the object permanence "
            "test proper; the child's hand into the bucket after it is the find)",
    "bring_far": "fetch a toy and set it down beside the child's far shoulder, level with its head, ROLL_BEYOND_M past the reach of the "
                 "arm on that side, from a kneel on its far side (the roll rung's setup, A109)",
    "clear": "a toy moved out of where she will kneel (A6)",
    "wave": "a wave of her hand",
    "walk": "walk to the target: 'door' (the hall, leaving), 'sofa' (she sits on it), 'child' (return: she comes to it), a point",
    "cover_face": "her hands over her face (peekaboo)",
    "reveal_face": "her hands away from her face (the reveal)",
    "do": "her body does the named act (DOES: wave, clap, stand, walk, open_hand, close_hand, show, pick_up); others are refused "
          "(templates.showable keeps the word waiting)",
    "copy": "her own arm or hand makes the movement the child just made, mirrored as she faces it (target 'kind:side', the side "
            "hers: arm_raise, wave, shake, open_hand; A52), asking nothing and earning nothing",
    "stand": "stand up where she is",
}
# NOT AT BIRTH (A25c, the lead's decision of 2026-09-25; amended by A90, 2026-09-26): the acts that move the child's body through a
# movement or a posture it did not make. At A25c all four were closed: the guides of its limbs (A8, A10), the knee over, the pull-to-sit
# and the prop with its catch (A9), because a guide is a demonstration and First 1's claim was "no demonstrations", and because a
# person cannot sit up or lift a 34 kg body (risk 4). A90 (the teacher's build 2d; docs/audit/teacher_of_reality.md 2d) reopens the
# guide and the knee over: a coach guides a hand, within the force a person can use (65 N, 76 N), and the smile goes to the child's own
# repeat, never to the guided act (the conduct judges nothing while her hands move it: HANDS_ON), every guide counted beside First 1
# (A63). The pull-to-sit and the prop stay closed: 34 kg. Refused when asked, with this reason, and logged. Kept in KINDS (their
# controllers stay built) so a later decision can open them at a boundary, never silently.
NOT_AT_BIRTH = ()                           # A161 (2026-10-01, the lead's decision at a boundary, as the note above asks): the pull-to-sit and the
                                            # prop are OPEN. Life day 56: the child on its back for seven days (sitting changes everything it
                                            # sees: the room, the toys, her face at its eyes' level); the pull rises only with its own flexion
                                            # (A9: at her brief cap 2 ticks she lays it back), so the posture stays its own to make (A25c);
                                            # the prop holds a trunk it has brought within 30 deg of vertical. Before A161: ("pull_to_sit", "prop")
_OPENED = [False]                           # the controllers' own tests open them (opened()); kept for the tests that name it


class opened:
    """a context in which NOT_AT_BIRTH's acts are carried out (their built controllers' own tests only: body/tests/test_sim_parent.py)"""

    def __enter__(self):
        _OPENED[0] = True

    def __exit__(self, *exc):
        _OPENED[0] = False
DOES = ("wave", "clap", "stand", "walk", "open_hand", "close_hand", "show", "pick_up")   # what 'do' carries out (P3's conduct reads
                                            # it: templates.showable() keeps a verb whose act is not here waiting)

PENDING_OK = (("look", "child_eyes"), ("open_hand", "child"), ("lean_in", "child_periphery"), ("withdraw", "child"))
# the acts an ask of her conduct allows while it is pending (P3's body/sim/lang/conduct.py PENDING_OK, A51; read from the bound
# conduct's module when one is bound, this copy otherwise)
PART_WORD = {"tummy": "trunk", "chest": "trunk", "head": "head", "hand": "hand", "foot": "foot", "arm": "arm", "leg": "leg",
             "far_arm": "arm", "L": "arm", "R": "arm", "trunk": "trunk"}   # a part of the child as her attention log names it (P3)
LINK_WORD = {**{n: "trunk" for n in ("torso_link", "pelvis", "waist_yaw_link", "waist_roll_link", "waist_pitch_link")},
             **{f"{s_}_{j}": w_ for s_ in ("left", "right") for j, w_ in
                (("shoulder_pitch_link", "arm"), ("shoulder_roll_link", "arm"), ("shoulder_yaw_link", "arm"), ("elbow_link", "arm"),
                 ("wrist_roll_link", "hand"), ("wrist_pitch_link", "hand"), ("wrist_yaw_link", "hand"), ("hip_pitch_link", "leg"),
                 ("hip_roll_link", "leg"), ("hip_yaw_link", "leg"), ("knee_link", "leg"), ("ankle_pitch_link", "foot"),
                 ("ankle_roll_link", "foot"))}}
BODY_PARTS = {"hand": ("left_wrist_yaw_link", "right_wrist_yaw_link"), "foot": ("left_ankle_roll_link", "right_ankle_roll_link"),
              "head": ("torso_link",), "tummy": ("torso_link",), "arm": ("left_elbow_link", "right_elbow_link"),
              "leg": ("left_knee_link", "right_knee_link")}
DONE_STATES = ("done", "refused", "cancelled")
# the room's fixtures she can walk to or point at (world points: the make_g1room layout)
FIXTURES = {"door": (2.6, -1.5, 1.0), "hall": (3.3, -1.5, 1.0), "window": (-2.6, -0.1, 1.45), "sofa": (0.15, 1.82, 0.45),
            "table": (0.15, 0.95, 0.40), "shelf": (2.39, 0.55, 0.45), "mat": (0.0, -0.6, 0.012), "light": (0.0, 0.3, 2.58),
            "floor": (1.2, -1.9, 0.0), "lamp": (-1.25, 1.95, 1.5)}


def fixtures_of(m):
    """A133 (the changed room): the fixtures' places from the model: the sofa, the table, the shelf and the lamp where their geoms stand
    (sofa_base, table_top, shelf, lamp_shade; their heights FIXTURES'), the rest FIXTURES' (the door, the hall, the window and the light
    are the walls', the mat does not move). What she looks at, points at, walks to and sits on follows the furniture"""
    out = dict(FIXTURES)
    for name, geom in (("sofa", "sofa_base"), ("table", "table_top"), ("shelf", "shelf"), ("lamp", "lamp_shade")):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if gid >= 0:
            out[name] = (float(m.geom_pos[gid][0]), float(m.geom_pos[gid][1]), float(FIXTURES[name][2]))
    return out
HALL_SPOT = (3.35, -1.5)                    # where she stands in the hall when away (the doorway's line, 0.75 m past it)
SOFA_SPOT_OFF = (0.71, -0.18)               # her seat on the sofa from the sofa's centre (A133: the sofa where the room has it; in the room
                                            # of birth (0.86, 1.64)): its right end, where she can stand before it (the low table stands
                                            # 0.13 m from the sofa's front along x -0.43..0.73: no one can stand before the middle)
SOFA_SEAT_Z = 0.52                          # the seat cushion's top (make_g1room: the cushions' tops at about 0.52 m)
MAT_BOX = (-1.40, 1.40, -1.60, 0.40, 0.012) # the play mat: x and y extents and its top (make_g1room.MAT_CENTER, MAT_HX, MAT_HY, MAT_T)
PALM_GRASP_OUT = 0.016                      # the Dex3's grasp point (g1acts.GRASP_LOCAL) lies this far out of its palm's surface
CARRY = {"k": "carry"}                      # a toy carried before her, at her waist
MAX_JUMP_M = 0.25                           # her pelvis never moves more than this in a tick (walking: 0.12 m; kneeling down:
MAX_LIMB_JUMP_M = 0.60                      # about 0.1 m), nor any segment more than this (a walking foot's swing: up to 0.4 m);
FETCH_OFF_TRY = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)   # A117 (C102): her kneeling spot's distance from a toy she fetches, tried in
                                            # turn: 0.45 m first (the pick from above at her knees, A5), then farther out where
                                            # furniture or a wall stands on the ring (the cup under the low table: free spots from 0.55 m;
                                            # the ball in the corner: from 0.75 m; measured on the dawn-15 pair), as far as a tall
                                            # kneel's reach carries (_reachable_at's 1.0 m: arm 0.65, trunk 0.35). Ours
ROLL_BEYOND_M = 0.10                        # A109 (C91): the roll rung's toy set this far past the reach of the arm on its far side: two
                                            # steps of the reach ladder (dayplan.LESSON_STEP), a full roll carries its body about that far
PUT_AHEAD_M = 0.35                          # A109: the roll rung's toy set down this far before her pelvis (where set_near's toy lies
                                            # from her kneel beside the child: her knees at HEELS_BACK, the toy a hand beyond them)
REBASE_TOL_M = 0.02                         # A108: her restored plan drawing her pelvis farther than this from where her body was
TURN_HEAD_OFFS = (0.55, 0.65, 0.75)          # A136: her kneel at a prone child's head for the roll by its far shoulder, this far from its eyes
HAND_HIGH_M = 0.06                              # C139: a pick whose hand stopped this far above the toy met something on the way down (the duck against
                                                # the front wall, day 37: 8 hand-overs refused 11 cm off): the toy is beyond her reach from above, left
                                                # where it lies (conduct.left), and the morning tidy puts it back (ours)
HIDE_REACH_M = 0.55                             # C138: the hide brings the bucket to within this of one of the child's hands (a G1 arm reaches 0.55, lane.arm_reach; ours)
TP_OPEN_CONTAINERS = frozenset({"bucket"})      # (templates.OPEN_CONTAINERS, named here without the import: the lang package imports this module)
NEVER_FETCHED = frozenset({"bucket"})          # C119: what she never carries: the hide game's container (day 27: ten shows of it refused at the put, 45 to 50 cm off)
                                            # (ours: the far shoulder's grip in her tall reach from 0.55 to 0.65 m in the rig)
                                            # last drawn is rebased onto the drawn pose (2 cm: under it her drive absorbs the difference)
                                            # more is a planning fault, refused (ours)
FACE_REDRAW_DEG = 1.5                       # her eyes are drawn again when her gaze has turned this far in her head (ours: a face
                                            # costs about 24 ms to draw; her irises move 0.3 mm for 1.5 deg)
KNEE_ROUTE_M = 0.9                          # A96: a new kneeling spot this near (m) and within KNEE_ROUTE_DEG of her facing is reached on
KNEE_ROUTE_DEG = 100                        # her knees (up onto the tall kneel, a turn on them, the shuffle) rather than by standing up
                                            # and walking (0.6 m and 25 deg before: the way round a lying child, its side to its head,
                                            # is about 0.8 m and a quarter turn, and a walk there took 70 s of re-planned trips)
MAX_NEED_TRIES = 24                         # the spots on which an act's need (a trunk solve, a face search) is tried before she
                                            # C170 (2026-09-30): 6 until the table left the mat's north (C169): the spots north of a child on
                                            # the mat, no longer behind furniture, were tried first for a put's reach and spent the six on the
                                            # far side of the put; the spot beside its hand came twentieth (19 failed reach tries, 2.4 s of planning)
                                            # gives up choosing (a guard on a planning tick's cost, ours)
MAX_PLANS = 40                              # an act whose plans do not settle in this many is given up (a guard, ours)
KEEP_ACTS, KEEP_OLD = 64, 1024             # the acts kept whole in her state, and the final statuses of older ones (her state is
                                            # captured every tick for the world's fault roll-back, so it stays small)
CHILD_BODY = ("pelvis", "waist_yaw_link", "waist_roll_link", "torso_link")   # the child's body (its trunk; its head is part of its
                                            # torso's link): her trunk keeps its standoff from these (A25b: _standoff)
HEELS_BACK = 0.338 - 0.03                   # parent_poses: the heels kneel's pelvis lies this far behind the tall kneel's
STAND_BACK = 0.30                           # parent_poses.kneel_down: its standing start lies this far behind the tall kneel's pelvis


class _Plain:
    """an act asked by the world itself (the night's walk to the sofa and back), in the shape of P3's Act"""

    def __init__(self, kind, target=None, during=None, thing=None):
        self.kind, self.target, self.during, self.thing = kind, target, during, thing


def _body_or_none(m, name):
    try:
        return m.body(name).id
    except KeyError:
        return None


class Refuse(Exception):
    """an act she cannot do, with the reason (it is refused, never faked)"""


MOUTH_LOCAL = np.array([kin.head_surface_x(0, kin.MOUTH_Z) + .0015, 0.0, kin.MOUTH_Z])   # the face test's mouth point (eyes.py)
FACE_CENTRE = np.array([kin.head_surface_x(0, 0.15), 0.0, 0.15])


def E_FACE_TURN_DEG():
    """A1's face test: her face turned within this of the eye (body/sim/eyes.FACE_TURN_DEG; imported late)"""
    from body.sim import eyes as _e
    return _e.FACE_TURN_DEG


def W_EYE_F():
    """the eyes' focal length in px (body/sim/world.EYE_F_PX; imported late: world imports this module)"""
    from body.sim import world as _w
    return _w.EYE_F_PX


class NatR:
    """a natural hand rotation: the palm facing `palm`, the fingers along her forearm (as parent_poses.reach with fingers None);
    with toy_c and off, the grip follows from a held toy's centre at toy_c (its centre `off` in her hand's frame)"""

    def __init__(self, palm, toy_c=None, off=None, bend=False):
        self.palm = unit(np.asarray(palm, float)); self.toy_c = toy_c; self.off = off
        self.bend = bend                    # the palm turned toward `palm` only as far as a straight wrist allows (a grasp from above:
                                            # the palm faces the toy across her forearm, as a hand reaching down to the floor does)


_FRAME_CACHE = {}


def frame_segs(kind, param, at, yaw):
    """her segments' world frames for a whole-body kneeling frame, placed at `at` facing `yaw`: parent_poses.kneel_down(u) or
    kneel(mode) at the origin facing +x, computed once and moved rigidly (both are rigid in their spot and facing, so the kneeling
    plan's frame checks cost a transform, not a pose)"""
    key = (kind, param)
    if key not in _FRAME_CACHE:
        pose = P.kneel_down((0.0, 0.0), 0.0, param) if kind == "kneel_down" else P.kneel((0.0, 0.0), 0.0, param)
        _FRAME_CACHE[key] = kin.fk(pose)
    Rz = kin.rz(yaw)
    t = np.array([at[0], at[1], 0.0])
    return {s_: (Rz @ p_ + t, Rz @ R_) for s_, (p_, R_) in _FRAME_CACHE[key].items()}


def trunk_pose(at, yaw, mode, lean, spine, twist, lift=0.0):
    """parent_poses.kneel's pelvis and spine alone (no legs, no arms), raised by `lift` (her floor lift there: ParentMotion._floor_lift):
    the chest's pose her arms reach from"""
    hip_z = P.KNEE_FLOOR + kin.L_TH * .995 if mode == "tall" else P.KNEE_FLOOR + .23
    p = P.base(at, yaw, hip_z + lift)
    p.R = kin.rz(yaw) @ kin.ry(math.radians(lean))
    kin.spine(p, lumbar=(spine * .55, 0, twist * .3), chest=(spine * .45, 0, twist * .7))
    return p


# parent_poses.kneel_down's pace (A25b: her carried body kneels at a person's pace): the time her kneeling down takes from standing
# (u 0) to u, at u = 0, 0.05, ... 3 (s), each step of u as slow as its slowest-allowed segment needs (her pelvis at KNEEL_PELVIS_MPS,
# every other segment at KNEEL_SEG_MPS), measured on the pose raised off the floor as her plan raises it (tools/sim_parent_motion.py
# kneel_time; body/tests/test_sim_parent.py measures it again). Her kneeling down runs along it eased in and out (a trapezoid: _ease),
# its fastest moment at these speeds. (The first physical build's KNEEL_PATH timed it by her fastest segment alone, so her pelvis sat
# back onto her heels at 0.6 m/s and her carried body met the floor at 1.6 kN)
KNEEL_TIME = (0.0, 0.0869, 0.254, 0.4302, 0.617, 0.792, 0.9236, 0.9721, 0.9812, 1.0066, 1.0453, 1.0944, 1.1558, 1.2167, 1.2791,
              1.3473, 1.4037, 1.4528, 1.4915, 1.5169, 1.526, 1.5517, 1.5636, 1.6068, 1.6997, 1.8444, 2.0258, 2.1198, 2.166, 2.2193,
              2.2788, 2.3428, 2.4097, 2.4776, 2.5442, 2.6072, 2.6642, 2.7127, 2.7501, 2.7741, 2.7826, 2.7918, 2.8069, 2.8278, 2.8546,
              2.8871, 2.9245, 2.9659, 3.0106, 3.058, 3.1072, 3.1578, 3.2093, 3.2605, 3.3112, 3.3608, 3.4079, 3.4504, 3.4853, 3.5089,
              3.5174)
KNEEL_U = tuple(round(0.05 * i, 2) for i in range(61))
EASE_IN = 0.2                               # her kneeling down eased in over its first fifth and out over its last (a constant
EASE_PEAK = 1.0 / (1.0 - EASE_IN)           # acceleration there), its fastest moment 1.25 x its mean (_ease)
KNEEL_CHECK_U = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0)   # kneel_down's frames checked 3 cm clear of the child from
                                            # standing to her tall kneel (her stepping foot swings forward in the first half: a sparser
                                            # check let it brush the child's hand, W2)


def _kneel_time(u):
    return float(np.interp(u, KNEEL_U, KNEEL_TIME))


def _kneel_u(T):
    return float(np.interp(T, KNEEL_TIME, KNEEL_U))


def _seg_dist(p, a, b):
    """a floor point's distance to the segment a-b"""
    p, a, b = (np.asarray(v, float)[:2] for v in (p, a, b))
    ab = b - a; L = float(ab @ ab)
    u = 0.0 if L < 1e-12 else min(1.0, max(0.0, float((p - a) @ ab) / L))
    return float(np.linalg.norm(p - (a + ab * u)))


def _along(pts, u):
    """the point a share u of the way along a polyline, by length"""
    L = [float(np.linalg.norm(b - a)) for a, b in zip(pts[:-1], pts[1:])]
    tot = sum(L)
    if tot < 1e-9:
        return pts[-1].copy()
    s = u * tot
    for (a, b), l in zip(zip(pts[:-1], pts[1:]), L):
        if s <= l or l == L[-1] and b is pts[-1]:
            return a + (b - a) * (min(1.0, s / l) if l > 1e-12 else 1.0)
        s -= l
    return pts[-1].copy()


def floor_z(xy):
    """the floor's height at a floor point: the mat's top on the mat, else the floor"""
    x0, x1, y0, y1, t = MAT_BOX
    return t if (x0 < xy[0] < x1 and y0 < xy[1] < y1) else 0.0


def sit_chair(at, yaw, u=1.0, seat_z=SOFA_SEAT_Z, lean=0.0):
    """sitting on a seat (the sofa): the pelvis on the seat at `at`, the thighs forward, the shins down, the feet flat on the floor
    in front; u in [0, 1] from standing in front of the seat (0) to seated (1), the trunk leaning forward on the way down"""
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-fwd[1], fwd[0]])
    feet_xy = {sd: np.asarray(at) + fwd * 0.50 + left * sg * 0.12 for sd, sg in (("L", 1), ("R", -1))}
    stand_xy = np.asarray(at) + fwd * 0.42
    e = _smooth(u)
    xy = stand_xy + (np.asarray(at) - stand_xy) * e
    z = (kin.HIP_Z - 0.015) + (seat_z + 0.07 - (kin.HIP_Z - 0.015)) * e
    p = P.base(xy, yaw, z)
    bend = 30 * math.sin(math.pi * e) + lean
    p.R = kin.rz(yaw) @ kin.ry(math.radians(bend * 0.6))
    kin.spine(p, lumbar=(bend * 0.25, 0, 0), chest=(bend * 0.15, 0, 0))
    feet = {sd: np.r_[feet_xy[sd], P.ANKLE_H] for sd in "LR"}
    P.legs_to(p, feet, {sd: np.r_[fwd, 0.4] for sd in "LR"}, foot_R={sd: P.flat_foot_R(yaw) for sd in "LR"})
    P.arms_relaxed(p, bend=25 + 20 * e, abd=10)
    return p


PUT_HEAD_OFFS = (0.65, 0.75, 0.85)          # A125: her heels' spot from a prone child's head to set a toy before its face (the toy 0.25 m
                                            # before its eyes: 0.4 to 0.6 m from her, within one trunk's reach). Ours
PUT_TOL_M = 0.10                            # m: a toy set down farther than this from where she meant it is reached for again (a person
PUT_RETRIES = 2                             # sets a toy where she means it), this many times, then the act is refused. Ours
LURE_MAX_M = 1.0                            # C168 (2026-09-30): a lesson's toy she cannot set beside the child's hand (no spot she can kneel
LURE_CLEAR_M = 0.45                         # at reaches the place: life day 44, the child 13,662 ticks under the coffee table, every hand-over
LURE_EDGE_M = 0.2                           # and show refused) goes instead to the nearest free floor point within LURE_MAX_M of its near hand,
LURE_MIN_M = 0.25                           # (the lure is taken when no spot she can kneel at reaches the natural place, or that place lies within
                                            # LURE_EDGE_M of the furniture's footprint: a toy set against the table's lip lands short of where she
                                            # meant it, 11 cm off at the edge of her reach, and lies out of the child's reach under the lip)
                                            # LURE_CLEAR_M clear of furniture and walls and at least LURE_MIN_M from the hand, from a spot she
                                            # can kneel at: the child must move to it (a parent lures a baby out from under the table with a
                                            # toy; A6 stands: she never moves the child). Ours. The clearance is her own body's: she must be
                                            # able to kneel right beside the point (at 0.2 the put landed at the edge of her reach, 11 cm off)
CRAWL_AHEAD_M = 0.15                        # A125 (the crawl rung): a lesson toy set this far before a prone child's eyes, plus her lesson's
                                            # distance: a stretch of its arm forward, then a crawl's length as the ladder rises (ours)
LIE_DOWN_S = 3.0                            # A124 (C107): her way down from the tall kneel onto her front, and up again, in this (ours:
                                            # a person's unhurried lying down; every segment under KNEEL_SEG_MPS on the way)
LIE_CHEST_UP = (35.0, 30.0, 25.0)           # the chest's extension lying, tried in turn (parent_poses.lie_prone: 35 keeps the forearms on
                                            # the floor and the head at 0.29 m)
LIE_HEAD_UP_M = 0.15                        # m: a prone child's eyes rise about this when it lifts its head (lane.HEAD_UP_M's head-up, the
                                            # G1's neckless head on its trunk; ours): where her lying face waits to be seen
LIE_OFFS = (1.35, 1.25, 1.45, 1.15, 1.55)   # m: her heels' spot from a prone child's head along its axis for the tummy-time lean-in: the
                                            # tall kneel HEELS_BACK nearer, her lying face 0.60 m nearer again, so her mouth lands 0.30 to
                                            # 0.60 m before its eyes (LEAN_DIST_M); tried nearest the middle first. Ours


def lie_pose(at, yaw, u=1.0, chest_up=35.0):
    """A124: from the tall kneel at `at` facing yaw (u 0) down onto her front (u 1): the pelvis sinks and slides a little back as the
    trunk pitches forward to the floor, the legs go out straight behind, the arms come down to the floor ahead; at 1 it is
    parent_poses.lie_prone with its chest raised chest_up"""
    e = _smooth(float(np.clip(u, 0.0, 1.0)))
    if e >= 1.0:
        return P.lie_prone(at, yaw, chest_up=chest_up)
    fwd = np.array([math.cos(yaw), math.sin(yaw)]); left = np.array([-fwd[1], fwd[0]])
    hip0 = P.KNEE_FLOOR + kin.L_TH * .995
    z = hip0 + (0.10 - hip0) * e
    p = P.base(at, yaw, z)
    p.R = kin.rz(yaw) @ kin.ry(math.radians(90.0 * e))
    ext = -abs(chest_up) * e
    lo_l = kin.LIM_DEG["lumbar_flex"][0] + 2; lo_c = kin.LIM_DEG["chest_flex"][0] + 2
    kin.spine(p, lumbar=(max(lo_l, ext * .55), 0, 0), chest=(max(lo_c, ext * .45), 0, 0))
    feet, knees = {}, {}
    for sd, sg in (("L", 1), ("R", -1)):
        k0 = np.asarray(at) + fwd * .03 + left * sg * .115                  # the tall kneel's feet: behind the knees on the floor
        f0 = np.r_[k0 - fwd * kin.L_SH * .95, P.ANKLE_H * .8]
        f1 = np.r_[np.asarray(at) - fwd * (kin.L_TH + kin.L_SH) * .98 + left * sg * .10, P.ANKLE_H * .8]
        feet[sd] = f0 + (f1 - f0) * e
        knees[sd] = np.r_[fwd * (.5 - .35 * e), -1.0]
    P.legs_to(p, feet, knees)
    if e < 0.35:
        P.arms_relaxed(p, bend=25, abd=12)
    else:
        segs = kin.fk(p)
        w = (e - 0.35) / 0.65
        for sd, sg in (("L", 1), ("R", -1)):
            cp, cR = segs["chest"]
            sh = cp + cR @ kin.OFFSET[f"upper_arm_{sd}"]
            grip = np.r_[sh[:2] + fwd * .26 + left * sg * .10, .035 + (1 - w) * .25]
            P.reach(p, sd, grip, np.array([0.0, 0.0, -1.0]), pole=np.r_[-fwd * .2 + left * sg * .35, -1.0], curl=.15, thumb=.2)
    kin.look(p, np.r_[np.asarray(at) + fwd * 1.2, .12])
    return p


def _nlerp(q0, q1, a):
    q1 = q1 if float(q0 @ q1) >= 0 else -q1
    q = q0 + a * (q1 - q0)
    return q / np.linalg.norm(q)


def _mat_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).reshape(-1))
    return q


def _quat_to_mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def _smooth(x):
    x = min(max(float(x), 0.0), 1.0)
    return x * x * x * (10 - 15 * x + 6 * x * x)            # minimum jerk


def _ease(x):
    """a trapezoid's progress: speeding up evenly over EASE_IN of the way's time, steady, slowing down evenly over the last EASE_IN"""
    x = min(max(float(x), 0.0), 1.0); r = EASE_IN; v = 1.0 / (1.0 - r)
    if x < r:
        return 0.5 * v * x * x / r
    if x > 1.0 - r:
        return 1.0 - 0.5 * v * (1.0 - x) * (1.0 - x) / r
    return 0.5 * v * r + v * (x - r)


def _ang(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _lst(v):
    return None if v is None else [float(x) for x in np.asarray(v, float).reshape(-1)]


# ------------------------------------------------------------------------------------------------ the child as she sees it
class Child:
    """The G1's geometry as the parent sees it (world truth, for her planner only; nothing here reaches the body): its eyes, trunk,
    posture, the side it faces, its footprint on the floor plan, and the named points her hands go to."""

    def __init__(self, m, d, g1_set):
        b = lambda n: m.body(n).id
        c = lambda n: m.camera(n).id
        self.eye = {sd: d.cam_xpos[c(f"eye_{sd}")].copy() for sd in "LR"}
        self.cam_R = {sd: d.cam_xmat[c(f"eye_{sd}")].reshape(3, 3).copy() for sd in "LR"}
        self.eyes = (self.eye["L"] + self.eye["R"]) / 2
        self.axis = -self.cam_R["L"][:, 2]                                   # the optical axis
        self.torso = d.xpos[b("torso_link")].copy()
        self.torso_R = d.xmat[b("torso_link")].reshape(3, 3).copy()
        self.pelvis = d.xpos[b("pelvis")].copy()
        self.pelvis_R = d.xmat[b("pelvis")].reshape(3, 3).copy()
        self.com = d.subtree_com[b("pelvis")].copy()
        self.head = d.xpos[b("torso_link")] + self.torso_R @ np.array([0.0, 0.0, 0.43])
        up = self.torso_R[:, 2]
        self.trunk_deg = math.degrees(math.acos(float(np.clip(up[2], -1, 1))))    # the spine from vertical
        fz = float(self.torso_R[2, 0])                                       # the chest's normal, up or down
        if self.trunk_deg < 45:
            self.posture = "sitting"
        elif fz > 0.6:
            self.posture = "back"
        elif fz < -0.6:
            self.posture = "front"
        else:
            self.posture = "side"
        self.rolled = float(self.torso_R[2, 1])                              # > 0: its left side up (it faces its right)
        self.len_axis = unit((self.pelvis - self.torso) * [1, 1, 0]) if np.linalg.norm((self.pelvis - self.torso)[:2]) > .05 \
            else unit(-self.torso_R[:, 2] * [1, 1, 0])                        # toward its feet on the floor plan
        self.lat = np.array([-self.len_axis[1], self.len_axis[0], 0.0])     # its left side on the floor plan when on its back
        if self.posture == "front":
            self.lat = -self.lat
        key = id(m)
        if key not in _G1_SHAPES:
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] in g1_set and (m.geom_contype[g] or m.geom_conaffinity[g])]
            _G1_SHAPES[key] = (np.array(gs), np.array([m.geom_rbound[g] if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH else _mesh_r(m, g)
                                                      for g in gs]))
        gs, rad = _G1_SHAPES[key]
        self.foot_pts = np.column_stack([d.geom_xpos[gs], rad])              # (x, y, z, radius) of its collision shapes
        self.grasp = {}
        self.palm_n = {}
        for sd, nm in (("L", "left"), ("R", "right")):
            wy = b(f"{nm}_wrist_yaw_link")
            Rw = d.xmat[wy].reshape(3, 3)
            self.grasp[sd] = d.xpos[wy] + Rw @ np.array([.13, -.06 if sd == "L" else .06, 0.0])
            self.palm_n[sd] = Rw @ np.array([0, -1.0 if sd == "L" else 1.0, 0])       # out of its palm
        self.body = {n: (d.xpos[b(n)].copy(), d.xmat[b(n)].reshape(3, 3).copy()) for n in
                     ("left_elbow_link", "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_knee_link",
                      "right_knee_link", "left_shoulder_roll_link", "right_shoulder_roll_link", "left_ankle_roll_link",
                      "right_ankle_roll_link", "torso_link", "pelvis")}

    def face_side(self):
        """the side of it she kneels on (A6): the side its torso is rolled toward, on its back; its left when it lies flat"""
        if self.posture == "back":
            return "R" if self.rolled > 0.15 else "L"
        if self.posture == "side":
            return "R" if self.rolled > 0 else "L"
        return "L"

    def clearance_xy(self, xy):
        """the floor-plan distance from a point to its body's shapes (their radii taken off)"""
        d = np.linalg.norm(self.foot_pts[:, :2] - np.asarray(xy)[None, :2], axis=1) - self.foot_pts[:, 3]
        return float(d.min())


_MESH_R = {}
_G1_SHAPES = {}


_MESH_V = {}


def _mesh_verts(m, g):
    """a mesh geom's vertices in the geom's frame (MuJoCo compiles them there), cached by geom (C221)"""
    key = (id(m), g)
    if key not in _MESH_V:
        mid = int(m.geom_dataid[g])
        adr, num = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        _MESH_V[key] = np.asarray(m.mesh_vert[adr:adr + num], float)
    return _MESH_V[key]


def _mesh_r(m, g):
    """a mesh geom's radius on the floor plan: its bounding box's half-diagonal in x and y (not the full bounding sphere)"""
    key = (id(m), g)
    if key not in _MESH_R:
        a = m.geom_aabb[g][3:]
        _MESH_R[key] = float(math.hypot(a[0], a[1]) * 0.85)
    return _MESH_R[key]


# ------------------------------------------------------------------------------------------------ the capped spring
class Hold:
    """a capped spring on a G1 link (4.1, A25): the held point is `local` in the link's frame; her hand's target moves from t0 to t1
    through the tick; the force K (target - p) + C (v_target - v) is clipped at `cap`, then at her own caps with her other holds.
    Its state is plain numbers (state())."""

    def __init__(self, name, body, local, side, cap, brief=False, kind="hold", n_dir=None):
        self.name, self.body, self.local, self.side = name, int(body), np.asarray(local, float), side
        self.cap, self.brief, self.kind = float(cap), bool(brief), kind
        self.t0 = self.t1 = None
        self.n_dir = None if n_dir is None else np.asarray(n_dir, float)     # its surface normal in the link's frame (touch)
        self.force = np.zeros(3)
        self.peak = 0.0
        self.at_cap_steps = 0
        self.at_cap_ticks = 0
        self.steps = 0
        self.ctl = {}                                                       # its controller's plain state
        self.next = None

    def point(self, d):
        return d.xpos[self.body] + d.xmat[self.body].reshape(3, 3) @ self.local

    def state(self):
        return dict(name=self.name, body=self.body, local=_lst(self.local), side=self.side, cap=self.cap, brief=self.brief,
                    kind=self.kind, t0=_lst(self.t0), t1=_lst(self.t1), n_dir=_lst(self.n_dir), force=_lst(self.force),
                    peak=self.peak, at_cap_steps=self.at_cap_steps, at_cap_ticks=self.at_cap_ticks, steps=self.steps, ctl=_plain(self.ctl))

    @classmethod
    def from_state(cls, s):
        h = cls(s["name"], s["body"], s["local"], s["side"], s["cap"], s["brief"], s["kind"], s["n_dir"])
        h.t0 = None if s["t0"] is None else np.array(s["t0"])
        h.t1 = None if s["t1"] is None else np.array(s["t1"])
        h.force = np.array(s["force"]); h.peak = float(s["peak"])
        h.at_cap_steps, h.at_cap_ticks, h.steps = int(s["at_cap_steps"]), int(s["at_cap_ticks"]), int(s["steps"])
        h.ctl = _unplain(s["ctl"])
        return h


# ------------------------------------------------------------------------------------------------ her floor plan (A6)
class FloorPlan:
    """the room's floor on a GRID_M grid for her paths: what stands on the floor (the world's shapes and the furniture's seen parts,
    as their bounding boxes) blocks it, the room's walls and the hall bound it. Static, built once per world."""
    X0, X1, Y0, Y1 = -2.6, 4.5, -2.7, 2.3

    def __init__(self, m, d):
        g = K.GRID_M
        self.nx, self.ny = int(round((self.X1 - self.X0) / g)) + 1, int(round((self.Y1 - self.Y0) / g)) + 1
        xs = self.X0 + g * np.arange(self.nx); ys = self.Y0 + g * np.arange(self.ny)
        self.xs, self.ys = xs, ys
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        inside = ((X > -2.6) & (X < 2.6) & (Y > -2.3) & (Y < 2.3)) | ((X >= 2.6) & (X < 4.5) & (Y > -2.65) & (Y < -0.35))
        self.dist = np.full((self.nx, self.ny), 9.0)             # each cell's distance to the nearest standing thing or wall
        boxes = []
        room = m.body("room").id
        for gg in range(m.ngeom):
            if int(m.geom_bodyid[gg]) != room or m.geom_type[gg] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            nm = m.geom(gg).name or ""
            if nm in ("mat", "ceiling", "floor") or nm.startswith("pad"):
                continue
            R = d.geom_xmat[gg].reshape(3, 3); c = d.geom_xpos[gg] + R @ m.geom_aabb[gg][:3]
            half = np.abs(R) @ m.geom_aabb[gg][3:]
            lo, hi = c - half, c + half
            if hi[2] < 0.02 or lo[2] > 1.8 or (hi[0] - lo[0]) > 5 or (hi[1] - lo[1]) > 5:
                continue
            boxes.append((lo[:2], hi[:2]))
        for lo, hi in boxes:
            dx = np.maximum(0, np.maximum(lo[0] - X, X - hi[0])); dy = np.maximum(0, np.maximum(lo[1] - Y, Y - hi[1]))
            self.dist = np.minimum(self.dist, np.hypot(dx, dy))
        # the room's back and front walls (the only shapes wider than 5 m); outside the room and the hall nothing is free
        inroom = (X > -2.6) & (X < 2.6) & (Y > -2.3) & (Y < 2.3)
        self.dist = np.minimum(self.dist, np.where(inroom, np.minimum(2.3 - Y, Y + 2.3), 9.0))
        self.dist = np.where(inside, self.dist, -1.0)

    def cell(self, xy):
        return (int(round((xy[0] - self.X0) / K.GRID_M)), int(round((xy[1] - self.Y0) / K.GRID_M)))

    def point(self, c):
        return np.array([self.X0 + K.GRID_M * c[0], self.Y0 + K.GRID_M * c[1]])

    def free(self, clear, child=None, toys=(), goal=None, goal_r=0.0, goal_clear=None, child_m=K.CLEAR_CHILD_M):
        """a boolean grid of the cells her centre may cross: furniture and walls at `clear`, the child's shapes at child_m (A6's
        CLEAR_CHILD_M) + her half-width (except within goal_r of the goal), toys at CLEAR_TOY_M + her feet's half-span"""
        X, Y = np.meshgrid(self.xs, self.ys, indexing="ij")
        ok = self.dist >= clear
        cd = None
        if child is not None:
            r = child_m + K.BODY_R_M
            cd = np.full(ok.shape, 9.0)
            for x, y, z, rad in child.foot_pts:
                cd = np.minimum(cd, np.hypot(X - x, Y - y) - rad)
            ok &= cd >= r
        tok = np.ones(ok.shape, dtype=bool)
        for (x, y) in toys:
            tok &= np.hypot(X - x, Y - y) >= K.CLEAR_TOY_M + 0.15
        ok &= tok
        if goal is not None and goal_r > 0:                                  # the last steps to her spot come as near furniture as
            near = np.hypot(X - goal[0], Y - goal[1]) <= goal_r                 # her own width and 5 cm (a spot by the table, the sofa)
            ok |= near & (self.dist >= (K.BODY_R_M + 0.05 if goal_clear is None else goal_clear))
        if goal is not None and cd is not None:                             # and, within APPROACH_CHILD_M of her spot, no nearer the
            gd = float(cd[self.cell(goal)])                                 # child than her spot itself (her spot lies within A6's
            near = np.hypot(X - goal[0], Y - goal[1]) <= K.APPROACH_CHILD_M    # walking clearance: she kneels beside it)
            ok |= near & tok & (cd >= min(r, gd - K.GRID_M)) & (self.dist >= K.BODY_R_M + 0.05)
        return ok

    def path(self, a, b, ok):
        """A* over the free cells, 8-connected, from a to b (floor points); the path's corners, simplified by line of sight; None
        when there is none"""
        s, t = self.cell(a), self.cell(b)
        nx, ny = self.nx, self.ny
        if not (0 <= s[0] < nx and 0 <= s[1] < ny and 0 <= t[0] < nx and 0 <= t[1] < ny):
            return None
        ok = ok.copy(); ok[s] = True; ok[t] = True
        g = {s: 0.0}; came = {}
        h = lambda c: math.hypot(c[0] - t[0], c[1] - t[1])
        heap = [(h(s), 0, s)]; n = 0
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.4142135623730951), (1, -1, 1.4142135623730951),
                 (-1, 1, 1.4142135623730951), (-1, -1, 1.4142135623730951)]
        done = set()
        while heap:
            _, _, c = heapq.heappop(heap)
            if c == t:
                break
            if c in done:
                continue
            done.add(c)
            for dx, dy, w in steps:
                q = (c[0] + dx, c[1] + dy)
                if not (0 <= q[0] < nx and 0 <= q[1] < ny) or not ok[q]:
                    continue
                ng = g[c] + w
                if ng < g.get(q, 1e18):
                    g[q] = ng; came[q] = c; n += 1
                    heapq.heappush(heap, (ng + h(q), n, q))
        if t not in g:
            return None
        cells = [t]
        while cells[-1] != s:
            cells.append(came[cells[-1]])
        cells = cells[::-1]
        pts = [np.asarray(a, float)[:2]]
        i = 0
        while i < len(cells) - 1:                                   # line of sight simplification
            j = len(cells) - 1
            while j > i + 1 and not self._los(cells[i], cells[j], ok):
                j -= 1
            pts.append(self.point(cells[j]))
            i = j
        pts[-1] = np.asarray(b, float)[:2]
        return [p for k, p in enumerate(pts) if k == 0 or np.linalg.norm(p - pts[k - 1]) > 1e-6]

    @staticmethod
    def _los(c0, c1, ok):
        n = int(max(abs(c1[0] - c0[0]), abs(c1[1] - c0[1]))) * 2 + 1
        for k in range(n + 1):
            u = k / n
            q = (int(round(c0[0] + (c1[0] - c0[0]) * u)), int(round(c0[1] + (c1[1] - c0[1]) * u)))
            if not ok[q]:
                return False
        return True


def _half_z(m, d, g):
    R = d.geom_xmat[g].reshape(3, 3)
    return float(np.abs(R[2]) @ m.geom_aabb[g][3:])


def _plain(x):
    """numbers and arrays as plain lists, recursively (for the save)"""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return {"__a__": x.tolist(), "shape": list(x.shape)}
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _canon(x):
    """a plain structure rebuilt so equal states always pickle to equal bytes: every string interned (pickle memoizes strings by
    identity, so equal strings must be one object), every container new (none shared), numbers as Python's (the world's _pose_state
    does the same for her pose)"""
    if isinstance(x, dict):
        return {(sys.intern(k) if isinstance(k, str) else k): _canon(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, str):
        return sys.intern(x)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    if isinstance(x, np.ndarray):
        return _canon(_plain(x))
    return x


def _unplain(x):
    if isinstance(x, dict):
        if "__a__" in x:
            return np.array(x["__a__"], dtype=np.float64).reshape(x["shape"])
        return {k: _unplain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_unplain(v) for v in x]
    return x


# ------------------------------------------------------------------------------------------------ her motion
class ParentMotion:
    """THE PARENT'S MOTION over a G1World (see the module's doc). The world owns it (G1World.parent) and calls tick_begin() before
    its tick's physics, before_step(s) / after_step(s) around each physics step and tick_end() after; her conduct calls request(),
    status(), cancel(); set_face() takes her feelings' graded face (parent_feel, P3). Its state rides in the world's save."""

    DOES = DOES                                                             # P3's conduct reads it (the acts 'do' carries out)

    def __init__(self, world):
        self.w = world
        self.conduct = None                                                 # her conduct, bound by bind_conduct (never her state)
        m, d = self.m, self.d = world.m, world.d
        sc = self.scene = world.scene
        self.bm = sc.bmap                                                   # her body's joints (parent_body.BodyMap) ...
        self.drive = PB.Drive(m, d, self.bm)                                # ... driven every physics step (L0)
        self.seg_chain = np.array([CHAINS.index(SEG_CHAIN[s]) for s in kin.SEGS])
        her_body = {m.body(f"parent_{s}").id: i for i, s in enumerate(kin.SEGS)}
        # HER HAND MEETS THE ROOM, NEVER THE CHILD (A25c, the lead's decision of 2026-09-25, replacing A4's "her whole hand touches the
        # child"): her palm, fingers and thumb as drawn touch the room and the floor and never a toy or the G1 (make_g1room.HAND_TOUCH:
        # contype 0, conaffinity 2 | 16, soft at priority 2, compiled collidable so MuJoCo's midphase keeps them); what she does to the
        # child is done by her capped springs (holds) and by the toys she holds
        self.finger_geoms = {}
        for sd in "LR":
            hb = m.body(f"parent_hand_{sd}").id; px = m.geom(f"parent_hand_{sd}").id
            self.finger_geoms[sd] = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == hb and g != px
                                     and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH]
            if not all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == HAND_TOUCH_AFFINITY for g in self.finger_geoms[sd]):
                raise ValueError("her palm, fingers and thumb must carry make_g1room.HAND_TOUCH's bits: regenerate g1room.xml")
            a0, n0 = int(m.body_bvhadr[hb]), int(m.body_bvhnum[hb])        # her fingers are drawn where her hand shape puts them (the
            m.bvh_aabb[a0:a0 + n0] = HAND_BOX                               # file's are placeholders), so MuJoCo's midphase boxes for
                                                                            # her hand, made at compile, are widened to all of it
        self.geom_seg = np.full(m.ngeom, -1, dtype=np.int64)                # her collision shapes' segment
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if b in her_body and (m.geom_contype[g] or m.geom_conaffinity[g]):
                self.geom_seg[g] = her_body[b]
        self.g1_geom = np.zeros(m.ngeom, dtype=bool)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in sc.g1_set and (m.geom_contype[g] or m.geom_conaffinity[g]):
                self.g1_geom[g] = True
        self.toys = {m.body(b).name[4:]: b for b in range(m.nbody) if m.body(b).name.startswith("toy_")}
        self.fixtures = fixtures_of(m)                                      # A133: the furniture where THIS room's model has it
        self.sofa_spot = tuple(float(v) for v in np.asarray(self.fixtures["sofa"][:2]) + np.asarray(SOFA_SPOT_OFF))
        self.toy_of_geom = np.full(m.ngeom, -1, dtype=np.int64)
        self.toy_names = sorted(self.toys)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            for i, k in enumerate(self.toy_names):
                if self.toys[k] == b and (m.geom_contype[g] or m.geom_conaffinity[g]):
                    self.toy_of_geom[g] = i
        self.toy_weld = {}
        for e in range(m.neq):
            nm = m.equality(e).name or ""
            if nm.startswith("hold_") and m.eq_type[e] == mujoco.mjtEq.mjEQ_WELD:
                _, sd, toy = nm.split("_", 2)
                self.toy_weld[(sd, toy)] = e
        self.hand_geom = {sd: m.geom(f"parent_hand_{sd}").id for sd in "LR"}
        self.proxy_on = {g: (int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in self.hand_geom.values()}
        self.body_geoms = np.array([g for g in range(m.ngeom) if self.geom_seg[g] >= 0 and int(m.geom_contype[g]) == 8
                                    and g not in self.hand_geom.values()], dtype=np.int64)   # her body's shapes on the floor's bit
        self.low_shapes = []                                                # her pelvis's, legs' and feet's collision shapes in their
        for g in range(m.ngeom):                                            # segments' frames (her floor lift: _floor_lift)
            sg = int(self.geom_seg[g])
            if sg < 0 or int(m.geom_contype[g]) != 8:
                continue
            seg = kin.SEGS[sg]
            if not seg.startswith(("pelvis", "thigh", "shin", "foot")):
                continue
            self.low_shapes.append((seg, int(m.geom_type[g]), m.geom_size[g].copy(), m.geom_pos[g].copy(), _quat_to_mat(m.geom_quat[g])))
        self._lift_cache = {}                                               # (a cache, never her state)
        self.leg_geoms = np.array([g for g in range(m.ngeom) if int(self.geom_seg[g]) >= 0 and int(m.geom_contype[g]) == 8
                                   and kin.SEGS[int(self.geom_seg[g])].startswith(("thigh", "shin", "foot"))], dtype=np.int64)
        self.plan = FloorPlan(m, d)
        self.scratch = mujoco.MjData(m)
        self.hand_idx = np.array([kin.SEGS.index("hand_L"), kin.SEGS.index("hand_R")])
        self.hand_idx_set = {int(x) for x in self.hand_idx}
        self.child_root = m.body("pelvis").id                               # the G1's root: its subtree's centre of mass is the child's
        self.grip_site = {sd: m.site(f"parent_hand_{sd}_grip").id for sd in "LR"}   # her grip point on each hand (GRIP_LOCAL)
        self._jac = np.zeros((3, m.nv))
        self._f6 = np.zeros(6)
        self._ft = np.zeros(6)
        self._hand_local = {}                                               # her hand's shapes per hand shape (a cache)
        self.hand_all = {sd: [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == m.body(f"parent_hand_{sd}").id
                              and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH] for sd in "LR"}   # palm, capsule, fingers, thumb
        self.hand_arr = {sd: np.array(v, dtype=np.int64) for sd, v in self.hand_all.items()}
        self.her_main = [int(g) for g in np.nonzero(self.geom_seg >= 0)[0] if (m.geom(g).name or "").startswith("parent_")
                         and not (m.geom(g).name or "").startswith(("parent_face", "parent_hair", "parent_eye"))]
        self.g1_geoms = [int(g) for g in np.nonzero(self.g1_geom)[0]]
        arm_bodies = {int(m.body(i).id) for i in range(m.nbody) if not m.body(i).name.startswith("parent")
                      and any(k in m.body(i).name for k in ("shoulder", "elbow", "wrist", "hand"))}   # C118: its arms and hands
        self.g1_arm_geoms = [g for g in self.g1_geoms if int(m.geom_bodyid[g]) in arm_bodies]
        self.g1_arr = np.array(self.g1_geoms, dtype=np.int64)
        trunk = {m.body(n).id for n in ("pelvis", "waist_yaw_link", "waist_roll_link", "torso_link") if _body_or_none(m, n) is not None}
        self.g1_trunk = np.array([g for g in self.g1_geoms if int(m.geom_bodyid[g]) in trunk], dtype=np.int64)   # A86: the child's
        # trunk, which her kneeling keeps CLEAR_M from; its limbs pass through her without force since A25c
        trunk = {m.body(n).id for n in CHILD_BODY}                          # the child's body: its trunk's links (its head is its torso's)
        self.g1_body_arr = np.array([g for g in self.g1_geoms if int(m.geom_bodyid[g]) in trunk], dtype=np.int64)
        kick = trunk | {m.body(f"{sd_}_{n}").id for sd_ in ("left", "right") for n in ("hip_pitch_link", "hip_roll_link", "hip_yaw_link",
                                                                                     "knee_link", "ankle_pitch_link", "ankle_roll_link")}
        self.g1_standoff_arr = np.array([g for g in self.g1_geoms if int(m.geom_bodyid[g]) in kick], dtype=np.int64)
        # her standoff: from its body and its legs (C8: "her kneeling spot outside its leg sweep"), not its arms: a hand of hers on its
        # chest is within its arms' reach, as a parent's is, and its arms' blows are its own (the babble test counts them)
        self.toy_rest = {}                                                  # each toy's lowest point above its centre, as born
        for k, b in self.toys.items():
            low = min(float(d.geom_xpos[g][2] - _half_z(m, d, g)) for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g])
            self.toy_rest[k] = float(d.xpos[b][2] - low)
        self.hold_zone = np.zeros(world.nz)                                 # this step's hold forces on each touch zone (world)
        self.child = Child(m, d, sc.g1_set)
        self._pose_cache = None                                             # her base pose for the last base (a cache, not state)
        self.kneel_reasons = []                                             # why the last kneel plan passed over each spot (instrument)
        self._reset()

    # ------------------------------------------------------------------ state
    def _reset(self):
        pose = self.scene.pose if self.scene.pose is not None else G.born_parent()
        yaw = math.atan2(pose.R[1, 0], pose.R[0, 0])
        self.acts = []                                                      # the recent acts (the last KEEP_ACTS and every one open)
        self.live = []                                                      # the ids asked of her not yet reported ended (report)
        self.cancels = []                                                   # [id, tick]: cancels for a tick not yet begun
        self.glances = []                                                   # L1's glances: [target, from tick, until tick]
        self.rep = None                                                     # (tick, report): the last report, given again if asked
        self.eoc_set = False                                                # eyes_on_child as the glue set it (no conduct bound)
        self.still_set = False                                              # still as the glue set it (no conduct bound)
        self.eoc = False                                                    # eyes_on_child as read this tick
        self.still = False                                                  # a formal trial holding her still, as read this tick
        self.aim = None; self.aim_f = {"L": 0.0, "R": 0.0}                  # the links her hands reach onto for a hold, and their touch
        self.arrived = {"L": False, "R": False}                            # her hands that met their link this tick ...
        self.arrived_now = {"L": False, "R": False}                        # ... and last tick (read by the reach)
        self.first_id = 0                                                   # the id of acts[0]
        self.old = {}                                                       # the final status of acts pruned from the list
        self.queue = {"body": [], "gaze": []}
        self.cur = {"body": None, "gaze": None}
        self.phases = []
        self.base = dict(mode="held", at=_lst(pose.pos[:2]), yaw=yaw, lean=0.0, spine=0.0, twist=0.0)   # the scene's pose, as placed
        self.arms = {sd: dict(mode="relaxed") for sd in "LR"}
        self.look = dict(target="ahead", until=0)
        self.face = dict(kin.FACE_NEUTRAL)
        self.holding = {"L": None, "R": None}
        self.carry = {"L": None, "R": None}                                 # a toy in her hand: its centre in her hand's frame
        self.lift = {"L": None, "R": None}                                  # where a hand lifts to when it lets go of the child
        self.holds = []
        self.asleep = False                                                 # the night's (sleep, wake)
        self.offset = {c: np.zeros(3) for c in CHAINS}                      # her plan moved where the child pushed her (A4), per chain
        self._offset_at_start = {c: np.zeros(3) for c in CHAINS}            # her offsets as this tick began (not state: set each tick)
        self._standoff_at_start = np.zeros(2)                               # (A108: her standoff likewise)
        self.over = {c: 0 for c in CHAINS}
        self.yielding = {c: False for c in CHAINS}
        self.stopped = {c: False for c in CHAINS}                           # a chain stopped in the last tick (A4): this tick starts
                                                                            # from where her body is, not from last tick's plan
        self.swivel = {"L": 0, "R": 0}                                      # each elbow's swing from her natural pole, as planned
        self.serial = 0; self.cur_serial = 0                                # her phases' numbers
        self.off_serial = {c: -1 for c in CHAINS}                           # the phase each chain's yield offset was made in
        self.prev = None                                                    # her base and arms as last planned
        self.blocked = None                                                 # the act given up by a push she backed off 12 cm from
        self.ymoved = {c: 0.0 for c in CHAINS}                              # how far each chain has backed off in this push
        self.quiet = {c: 0 for c in CHAINS}                                 # ticks since the child last pressed on each chain
        self.body_f = np.zeros(3)                                           # her body's contact force on the G1 per chain, last step
        self.body_hist = np.zeros((K.HER_PAIN_STEPS, 3))                    # ... and the last 5 steps
        self.ydir = {c: np.zeros(3) for c in CHAINS}
        self.touching = {c: 0.0 for c in CHAINS}
        self.last_touch = {c: 0.0 for c in CHAINS}
        self.pain_win = np.zeros((len(kin.SEGS), K.HER_PAIN_STEPS))
        self.pain_any = False
        self.hits = []
        self.lesson_dist = 0.10                 # her lesson's distance for bring_back (A90): the toy set this far out from its near hand
        chain = ("shoulder_roll_link", "shoulder_yaw_link", "elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link")
        self.arm_m = float(sum(np.linalg.norm(self.m.body_pos[self.m.body(f"left_{c}").id]) for c in chain)
                           + np.linalg.norm([0.13, 0.06, 0.0]))          # A109: its arm's reach from the shoulder, as the lane measures
                                                                            # child_can_reach (lane.arm_reach: the chain to the grasp point)
        self.brief_s = 0.0
        self.rest_s = K.BRIEF_REST_S
        self.drawn_face = None
        self.face_shown = None                                              # her face as drawn in the last tick (the report's face) ...
        self.face_prev = None                                               # ... and in the tick before it
        self.trunk_at = None                                                # her chest as the last tick ended (the report's trunk field)
        self.trunk_moved = False                                            # ... and whether it moved in that tick
        self.planned_trunk_moved = False                                    # her plan moved her trunk this tick (a lean, a turn)
        self.hold_touch = {"L": 0.0, "R": 0.0}                              # a holding hand's contact with its held link, last step
        self.toy_touch = {"L": False, "R": False}                           # a toy in her hand touching the child, last step
        self.lag = False
        self.push_on = {c: False for c in CHAINS}
        self.held_at = {c: None for c in CHAINS}                            # a still chain's pose as a push on it began (plain)
        self.on_child = {c: False for c in CHAINS}                          # a chain of hers resting its weight on the child
        self.arm_in = {c: False for c in CHAINS}                            # an arm pressed on its upper arm or forearm (it draws in)
        self.core_hold = None                                               # her trunk's pose where it stopped with a pressed arm (plain)
        self.child_prev = None                                              # the child's shapes' centres as the last tick began, and as
        self.child_pts = None                                               # this one did (her state: the calm step, _restless_near)
        self.trunk_kneel = False                                            # she kneels where only its trunk was clear (A86)
        self.standoff = np.zeros(2)                                         # her base moved back off the child's body (her standoff,
                                                                            # A25b: _standoff), on the floor plan
        self.clear_now = 0.2                                                # her planned trunk's clearance from the child's body (m)
        segs = kin.fk(pose)
        self.written = (np.array([segs[s][0] for s in kin.SEGS]), np.array([_mat_to_quat(segs[s][1]) for s in kin.SEGS]))
                                                                            # her plan's segments at the last tick's end (her body's own
                                                                            # are the physics')
        tg = self.bm.targets(segs)
        self.drive.set_tick(tg, tg)
        self.placed = False                                                 # an instrument placed her (tests): idle until an act
        self.xfrc_bodies = []                                               # the bodies her holds pushed on the last step
        self.start = self.end = None
        self.moving = False
        self.warm = {}
        self.tick = int(self.w.tick)
        self.stats = self._new_stats()
        self.effort_peak = 0.0
        self.solve_ms = {}                                                  # wall ms her trunk solves took per act (an instrument:
                                                                            # never in her state, which must replay bit for bit)

    @staticmethod
    def _new_stats():
        return dict(contact_peak={c: 0.0 for c in CHAINS}, pen_max=0.0, hold_peak=0.0, yield_ticks=0)

    def state(self):
        """every number of her motion, plain (her pose is the scene's, saved by the world); canonical, so equal states pickle alike"""
        return _canon(self._state())

    def _state(self):
        return dict(live=list(self.live), cancels=[list(c) for c in self.cancels], glances=_plain(self.glances),
                    rep=None if self.rep is None else [self.rep[0], _plain(dict(self.rep[1], acts=[[int(k), v] for k, v in self.rep[1]["acts"].items()]))],
                    eoc_set=self.eoc_set, still_set=self.still_set, eoc=self.eoc, still=self.still, asleep=bool(self.asleep),
                    aim=None if self.aim is None else {k: list(v) for k, v in self.aim.items()},
                    arrived=dict(self.arrived), arrived_now=dict(self.arrived_now),
                    acts=_plain(self.acts), first_id=self.first_id, old=[[int(k), v] for k, v in self.old.items()], queue={k: list(v) for k, v in self.queue.items()}, cur=dict(self.cur),
                    phases=[_plain(p) for p in self.phases], base=_plain(self.base), arms=_plain(self.arms), look=_plain(self.look),
                    face=dict(self.face), holding=dict(self.holding), carry=_plain(self.carry), lift=_plain(self.lift), holds=[h.state() for h in self.holds],
                    offset={c: _lst(v) for c, v in self.offset.items()},
                    over=dict(self.over), yielding=dict(self.yielding), stopped=dict(self.stopped), swivel=dict(self.swivel), serial=self.serial, cur_serial=self.cur_serial, blocked=self.blocked,
                    off_serial=dict(self.off_serial), ymoved=dict(self.ymoved), quiet=dict(self.quiet), body_f=_lst(self.body_f), body_hist=self.body_hist.tolist(),
                    prev=self.prev,
                    ydir={c: _lst(v) for c, v in self.ydir.items()}, touching=dict(self.touching), last_touch=dict(self.last_touch), pain_win=self.pain_win.tolist(),
                    hits=[list(h) for h in self.hits], brief_s=self.brief_s, rest_s=self.rest_s, lesson_dist=float(self.lesson_dist),
                    drawn_face=None if self.drawn_face is None else [list(self.drawn_face[0]), _lst(self.drawn_face[1])],
                    face_shown=None if self.face_shown is None else list(self.face_shown),
                    face_prev=None if self.face_prev is None else list(self.face_prev),
                    trunk_at=None if self.trunk_at is None else [_lst(self.trunk_at[0]), _lst(self.trunk_at[1])], trunk_moved=self.trunk_moved,
                    planned_trunk_moved=self.planned_trunk_moved, hold_touch=dict(self.hold_touch), toy_touch=dict(self.toy_touch), push_on=dict(self.push_on),
                    held_at={c: v for c, v in self.held_at.items()}, on_child=dict(self.on_child),
                    arm_in=dict(self.arm_in), core_hold=self.core_hold, standoff=_lst(self.standoff), clear_now=self.clear_now, trunk_kneel=bool(getattr(self, "trunk_kneel", False)),
                    child_pts=None if self.child_pts is None else [list(x) for x in self.child_pts],
                    written=None if self.written is None else [self.written[0].tolist(), self.written[1].tolist()],
                    drive=self.drive.state(),
                    moving=self.moving, placed=self.placed, xfrc_bodies=list(self.xfrc_bodies), warm=_plain(self.warm), tick=self.tick, stats=_plain(self.stats), effort_peak=self.effort_peak,
                    proxies={int(g): [int(self.m.geom_contype[g]), int(self.m.geom_conaffinity[g])]
                             for g in list(self.hand_geom.values()) + self.finger_geoms["L"] + self.finger_geoms["R"] + [int(x) for x in self.body_geoms]})

    def load_state(self, s):
        self.live = [int(i) for i in s.get("live", ())]
        self.cancels = [[int(i), int(t)] for i, t in s.get("cancels", ())]
        self.glances = [list(g) for g in _unplain(s.get("glances", []))]
        r = s.get("rep")
        self.rep = None if r is None else (int(r[0]), dict(_unplain(r[1]), acts={int(k): v for k, v in r[1]["acts"]}))
        self.eoc_set = bool(s.get("eoc_set", False)); self.eoc = bool(s.get("eoc", False))
        self.still_set = bool(s.get("still_set", False)); self.still = bool(s.get("still", False))
        self.asleep = bool(s.get("asleep", False))
        am = s.get("aim")
        self.aim = None if am is None else {k: (int(v[0]), float(v[1])) for k, v in am.items()}
        self.aim_f = {"L": 0.0, "R": 0.0}
        self.arrived = dict(s.get("arrived", {"L": False, "R": False})); self.arrived_now = dict(s.get("arrived_now", {"L": False, "R": False}))
        self.acts = _unplain(s["acts"]); self.first_id = int(s["first_id"]); self.old = {int(k): v for k, v in s["old"]}
        self.queue = {k: list(v) for k, v in s["queue"].items()}
        self.cur = dict(s["cur"])
        self.phases = [_unplain(p) for p in s["phases"]]
        self.base = _unplain(s["base"]); self.arms = _unplain(s["arms"]); self.look = _unplain(s["look"])
        self.face = dict(s["face"]); self.holding = dict(s["holding"]); self.carry = _unplain(s["carry"]); self.lift = _unplain(s["lift"])
        self.holds = [Hold.from_state(h) for h in s["holds"]]
        self.offset = {c: np.array(v, dtype=np.float64) for c, v in s["offset"].items()}
        self.over = dict(s["over"]); self.yielding = dict(s["yielding"]); self.swivel = dict(s["swivel"])
        self.stopped = dict(s.get("stopped", {c: False for c in CHAINS}))
        self.serial, self.cur_serial, self.off_serial = int(s["serial"]), int(s["cur_serial"]), dict(s["off_serial"])
        self.blocked = None if s["blocked"] is None else int(s["blocked"])
        self.ymoved = dict(s["ymoved"]); self.quiet = dict(s["quiet"]); self.body_f = np.array(s["body_f"], dtype=np.float64)
        self.body_hist = np.array(s.get("body_hist", np.zeros((K.HER_PAIN_STEPS, 3))), dtype=np.float64).reshape(K.HER_PAIN_STEPS, 3)
        self.prev = s["prev"]
        self.ydir = {c: np.array(v, dtype=np.float64) for c, v in s["ydir"].items()}
        self.touching = dict(s["touching"]); self.last_touch = dict(s["last_touch"])
        self.pain_win = np.array(s["pain_win"], dtype=np.float64).reshape(len(kin.SEGS), K.HER_PAIN_STEPS)
        self.pain_any = bool(self.pain_win.any())
        self.hits = [tuple(h) for h in s["hits"]]
        self.lesson_dist = float(s.get("lesson_dist", 0.10))
        self.brief_s, self.rest_s = float(s["brief_s"]), float(s["rest_s"])
        self.drawn_face = None if s["drawn_face"] is None else (tuple(tuple(x) for x in s["drawn_face"][0]), np.array(s["drawn_face"][1]))
        fs = s.get("face_shown")
        self.face_shown = None if fs is None else tuple(tuple(x) for x in fs)
        fs = s.get("face_prev")
        self.face_prev = None if fs is None else tuple(tuple(x) for x in fs)
        ta = s.get("trunk_at")
        self.trunk_at = None if ta is None else (np.array(ta[0], dtype=np.float64), np.array(ta[1], dtype=np.float64))
        self.trunk_moved = bool(s.get("trunk_moved", False))
        self.planned_trunk_moved = bool(s.get("planned_trunk_moved", False))
        self.hold_touch = {k: float(v) for k, v in s.get("hold_touch", {"L": 0.0, "R": 0.0}).items()}
        self.toy_touch = {k: bool(v) for k, v in s.get("toy_touch", {"L": False, "R": False}).items()}
        self.push_on = {k: bool(v) for k, v in s.get("push_on", {c: False for c in CHAINS}).items()}
        self.held_at = {k: (None if v is None else [list(x) for x in v]) for k, v in s.get("held_at", {c: None for c in CHAINS}).items()}
        self.on_child = {k: bool(v) for k, v in s.get("on_child", {c: False for c in CHAINS}).items()}
        self.arm_in = {k: bool(v) for k, v in s.get("arm_in", {c: False for c in CHAINS}).items()}
        ch = s.get("core_hold")
        self.core_hold = None if ch is None else [list(x) for x in ch]
        self.standoff = np.array(s.get("standoff", (0.0, 0.0)), dtype=np.float64)
        self.trunk_kneel = bool(s.get("trunk_kneel", False))
        cp = s.get("child_pts")
        self.child_pts = None if cp is None else np.array(cp, dtype=np.float64)
        self.child_prev = None
        self.clear_now = float(s.get("clear_now", 0.2))
        self.written = None if s["written"] is None else (np.array(s["written"][0], dtype=np.float64), np.array(s["written"][1], dtype=np.float64))
        self.drive.load_state(s["drive"])
        self.start = self.end = None
        self.moving = bool(s["moving"]); self.placed = bool(s["placed"]); self.xfrc_bodies = [int(b) for b in s["xfrc_bodies"]]
        self.warm = _unplain(s["warm"]); self.tick = int(s["tick"])
        self.stats = _unplain(s["stats"]); self.effort_peak = float(s["effort_peak"])
        for g, (ct, ca) in s["proxies"].items():
            self.m.geom_contype[int(g)] = ct; self.m.geom_conaffinity[int(g)] = ca

    # ------------------------------------------------------------------ the interface (P3's)
    def request(self, act, tick=None):
        """ask for an act (anything with kind, target, during and thing: conduct.Act); returns its id. Refused at once when its
        kind is unknown; otherwise it waits its turn on its channel (reported 'running') and starts in turn"""
        kind = str(getattr(act, "kind", act))
        target = getattr(act, "target", None)
        during = getattr(act, "during", None)
        thing = getattr(act, "thing", None)
        tick = self.w.tick if tick is None else int(tick)
        i = self.first_id + len(self.acts)
        chan = "gaze" if kind == "look" else "body"
        a = dict(id=i, tick=tick, kind=kind, target=None if target is None else str(target), during=during,
                 thing=None if thing is None else str(thing), chan=chan, status="queued", why="", start=None, end=None, info={})
        self.acts.append(a)
        self.live.append(i)
        if kind not in KINDS:
            self._end(a, "refused", f"no such act: {kind}")
        elif kind in NOT_AT_BIRTH and not _OPENED[0]:
            self._end(a, "refused", f"not at birth: she never moves its body through a movement or a posture it did not make (A25c)")
        else:
            self.queue[chan].append(i)
        self._prune()
        return i

    def _prune(self):
        """only the last KEEP_ACTS finished acts are kept whole; an older one keeps its final status and reason"""
        while len(self.acts) > KEEP_ACTS and self.acts[0]["status"] in DONE_STATES:
            a = self.acts.pop(0); self.first_id += 1
            self.old[a["id"]] = [a["status"], a["why"]]
            if len(self.old) > KEEP_OLD:
                del self.old[min(self.old)]
        if len(self.live) > KEEP_OLD:                                       # a world whose glue never reads her report: its oldest
            ended = [i for i in self.live if self._act(i)["status"] in DONE_STATES][:len(self.live) - KEEP_OLD]   # ended acts
            self.live = [i for i in self.live if i not in set(ended)]                                              # go unreported

    def _act(self, i):
        i = int(i)
        if i < self.first_id:
            st = self.old.get(i, ["done", "expired"])
            return dict(id=i, status=st[0], why=st[1])
        return self.acts[i - self.first_id]

    def status(self, i, tick=None):
        """P3's contract: exactly 'running' (under way, or waiting its turn), 'done', 'refused' or 'cancelled'"""
        st = self._act(i)["status"]
        return "running" if st == "queued" else st

    def why(self, i):
        return self._act(i)["why"]

    def info(self, i):
        return self._act(i)

    def cancel(self, i, tick=None):
        """the act stopped from `tick` (P3's contract; ticks as the conduct counts them: report(t) is made after the world's tick
        that leaves world.tick == t, so a cancel for tick t + 1 made after report(t) takes effect in the world's next tick, as
        does one with no tick; a later tick waits for it). A waiting act ends at once; one under way ends where it is (a
        transition finishes, her holds let go: _abort_body); a look's eyes go back to the child's eyes"""
        if tick is not None and int(tick) > self.w.tick + 1:
            self.cancels.append([int(i), int(tick)])
            return
        a = self._act(i)
        if int(i) < self.first_id or a["status"] in DONE_STATES:
            return
        if a["status"] == "queued":
            self.queue[a["chan"]].remove(a["id"])
            self._end(a, "cancelled", "cancelled before it began")
            return
        self._end(a, "cancelled", "cancelled")
        if a["chan"] == "body":
            self._abort_body()
        else:
            self._gaze_back(a)

    def sleep(self):
        """THE NIGHT (5.4, A46, A17; the world's dusk): every act of hers stops where it is, her holds let go, what she holds she
        sets aside (A95: life 1 slept holding two toys and woke with both hands full), and she walks to the sofa and sits there,
        touching nothing, until the morning (wake)"""
        for i in list(self.live):
            self.cancel(i)
        self.asleep = True
        for toy in [v for v in self.holding.values() if v is not None]:
            self.request(_Plain("clear", toy))                           # set down beside her, before she goes
        self.request(_Plain("walk", "sofa"))

    def wake(self):
        """THE MORNING (the world's dawn): she comes back to the child"""
        self.asleep = False
        self.request(_Plain("walk", "child"))

    def bind_conduct(self, conduct):
        """her conduct (P3's Conduct), whose eyes_on_child she reads every tick (A51): while an ask is pending her eyes and head
        stay on the child's eyes, L1 turns to nothing (no glance), and no act starts but those an ask allows (PENDING_OK); they
        wait their turn, reported 'running'. Not her state: the conduct saves its own"""
        self.conduct = conduct

    def set_eyes_on_child(self, flag):
        """the same, set by the world's glue each tick where no conduct is bound (saved: it rules the tick it is set for)"""
        self.eoc_set = bool(flag)

    def set_still(self, flag):
        """P3's Conduct.still, set by the world's glue where no conduct is bound: a formal trial's settle, sentence and window, when
        she starts no act, rests her hands on her thighs and shows her face's neutral set (saved)"""
        self.still_set = bool(flag)

    def _still(self):
        if self.conduct is not None:
            return bool(getattr(self.conduct, "still", False))
        return bool(self.still_set)

    def _eyes_on_child(self):
        if self.conduct is not None:
            return bool(getattr(self.conduct, "eyes_on_child", False))
        return bool(self.eoc_set)

    def _pending_ok(self):
        """the acts an ask allows (P3's conduct.PENDING_OK, read from the bound conduct's module; PENDING_OK here otherwise)"""
        if self.conduct is not None:
            mod = sys.modules.get(type(self.conduct).__module__)
            ok = getattr(mod, "PENDING_OK", None)
            if ok is not None:
                return tuple(tuple(x) for x in ok)
        return PENDING_OK

    def glance(self, target, tick=None, ticks=None):
        """L1's reflexive gaze (4.10; the world calls it for a sudden event): her eyes and head on `target` from the world's next
        tick for `ticks` ticks (LOOK_TICKS: her gaze reaches its target within 2 ticks), then back where they were; none while an
        ask is pending (A51)"""
        n = max(1, int(K.LOOK_TICKS if ticks is None else ticks))
        start = self.w.tick if tick is None else max(int(tick) - 1, self.w.tick)
        self.glances.append([target if isinstance(target, str) else _lst(target), int(start), int(start) + n])

    def report(self, tick):
        """P3's contract (StubMotion.report; A51): her attention log's fields for the tick just lived, made after the world's tick
        and before her conduct runs: where her eyes, her head and each hand physically point (a thing's id; 'child_eyes' or
        'child' for her eyes on its face, or 'child' for a hand held open or waved toward it touching nothing; the part of the
        child a hand touches or reaches onto, by its word: 'trunk', 'arm', 'hand', 'leg', 'foot', 'head'; a place word; 'mama'
        for her own face; '?' where it points at nothing she can name, as a walk's eyes ahead; None at rest), and `acts`, {id:
        status} for every act asked of her that she has not yet reported ended, each exactly 'running', 'done', 'refused' or
        'cancelled'; an act reported ended is not reported again. Asked again for the same tick, the same report"""
        if self.rep is not None and self.rep[0] == int(tick):
            r = self.rep[1]
            return dict(r, acts=dict(r["acts"]))
        f = self._fields()
        acts = {i: self.status(i) for i in self.live}
        self.live = [i for i in self.live if acts[i] not in DONE_STATES]
        r = dict(f, acts=acts)
        self.rep = (int(tick), dict(r, acts=dict(acts)))
        return dict(r, acts=dict(acts))

    def _word(self, t):
        """a target as her attention log names it: a toy's id, a place, the child itself, a part of the child by its word"""
        if t is None:
            return None
        if isinstance(t, (list, tuple, np.ndarray)):
            return "?"
        t = str(t)
        if t in ("child", "child_eyes", "child_periphery", "child_line", "mama") or t in self.toys or t in self.fixtures:
            return t
        return PART_WORD.get(t, "?")

    def _link_word(self, body):
        return LINK_WORD.get(self.m.body(int(body)).name, "?")

    def _fields(self):
        """where her eyes, head, hands and trunk point now, and whether her face moves (report)"""
        t = self.look.get("target")
        g = self._glance_now() if not self.eoc and self.cur["gaze"] is None else None
        if g is not None and self._resolve_point(g) is not None:
            t = g                                                           # L1's glance: her eyes on it for its ticks
        moving = self.base["mode"] in MOVING
        eyes = ("?" if moving else None) if t in (None, "ahead") else self._word(t)
        out = dict(eyes=eyes, head=eyes)
        cur = self._act(self.cur["body"]) if self.cur["body"] is not None else None
        for sd, fld in (("L", "left"), ("R", "right")):
            out[fld] = self._hand_word(sd, cur)
        out["trunk"] = self._trunk_word(cur)
        out["face"] = self._face_word()
        return out

    def _trunk_word(self, cur):
        """her trunk (P3's tenth round): 'child' while it faces the child and holds still (her body as the physics has it: her chest
        moved less than TRUNK_STILL_M and turned less than TRUNK_STILL_DEG over the tick, and her plan did not move it), the place
        she walks to while she walks to a named one, '?' while it leans, turns, shifts or is pushed, or faces elsewhere"""
        b = self.base
        if b["mode"] in MOVING:
            if cur is not None and cur["kind"] == "walk" and cur.get("target") is not None:
                return self._word(cur["target"])
            return "?"
        if self.trunk_moved or self.planned_trunk_moved:
            return "?"
        cp = self.d.xpos[self.bm.seg_body["chest"]]; cR = self.d.xmat[self.bm.seg_body["chest"]].reshape(3, 3)
        to = self.child.torso - cp
        fw, tw = cR[:2, 0], to[:2]
        nf, nt = float(np.linalg.norm(fw)), float(np.linalg.norm(tw))
        if nf < 1e-6 or nt < 1e-6:
            return "?"
        return "child" if float(fw @ tw) / (nf * nt) >= math.cos(math.radians(K.FACE_CHILD_DEG)) else "?"

    def _face_word(self):
        """her face (P3's contract): 'mama' on a tick her expression is not its neutral set, or moved since the last tick; her jaw
        (her mouth, with her speech) and her blinks apart; None while it is still"""
        key = self.face_shown
        if key is None:
            f = self._face_drawn()
            key = tuple(sorted((k, float(v)) for k, v in f.items() if k not in ("jaw", "blink")))
        neutral = tuple(sorted((k, float(v)) for k, v in kin.FACE_NEUTRAL.items() if k not in ("jaw", "blink")))
        moved = self.face_prev is not None and tuple(self.face_prev) != tuple(key)
        return "mama" if tuple(key) != neutral or moved else None

    def _face_drawn(self):
        """her face as it is drawn this tick: her feelings' graded face, or, while a formal trial holds her still, its neutral set but
        her jaw (her mouth moves with her speech; P3's Conduct.still)"""
        if self.still:
            f = dict(kin.FACE_NEUTRAL); f["jaw"] = float(self.face.get("jaw", 0.0))
            return f
        return dict(self.face)

    def _hand_word(self, sd, cur):
        toy = self.holding[sd]
        if toy is not None:
            return toy                                                     # a toy in her hand: the hand is at it
        a = self.arms[sd]
        mode = a.get("mode")
        if mode == "relaxed" or a.get("support"):
            return None
        if mode == "hold":
            h = self._hold(a["hold"])
            return "?" if h is None else self._link_word(h.body)
        if mode == "point":
            return self._word(a.get("target"))
        to = a.get("to") or {}
        k = to.get("k")
        if k in ("relaxed", "carry", "thigh"):
            return None
        if k in ("show", "above_toy", "toy_at", "toy_centre", "floor"):
            return to.get("toy", "?")
        if k == "palm":
            return "hand"
        if k == "open":
            return "child"
        if k == "link":
            return self._link_word(to["body"])
        if k == "rel" and cur is not None:
            kind, tg = cur["kind"], cur["target"]
            if kind == "wave" or (kind == "do" and tg in ("wave", "open_hand", "close_hand")):
                return "child"                                             # waved or held open toward it, touching nothing
            if kind in ("cover_face", "reveal_face"):
                return "mama"
            if kind == "copy":
                return None                                                # moving at nothing (P3's directs)
        return "?"

    def _glance_now(self):
        for g in self.glances:
            if g[1] <= self.tick < g[2]:
                return g[0]
        return None

    def set_face(self, params):
        """her face this tick: graded face parameters (parent_kin.FACE_NEUTRAL's keys; parent_feel's display)"""
        f = dict(kin.FACE_NEUTRAL); f.update({k: float(v) for k, v in dict(params).items()})
        self.face = f

    def truth(self):
        """her motion as the instruments and her conduct see it (world truth, never the body's): her holds and their forces, her
        effort against her caps, her contacts with the child and whether she yields, the hits she felt, her acts under way, her
        support (its force and torque on her pelvis) and her joints at her strength"""
        cur = self._act(self.cur["body"]) if self.cur["body"] is not None else None
        return dict(holds=[dict(name=h.name, kind=h.kind, side=h.side, body=self.m.body(h.body).name, N=float(np.linalg.norm(h.force)),
                                cap=h.cap, peak=h.peak, state=h.ctl.get("state"), mode=h.ctl.get("mode")) for h in self.holds],
                    effort_N=float(sum(np.linalg.norm(h.force) for h in self.holds)), brief_s=self.brief_s,
                    contact_N={c: float(self.last_touch.get(c, 0.0)) for c in CHAINS}, yielding=dict(self.yielding),
                    hits=[tuple(h) for h in self.hits if h[0] >= self.tick - 1], holding=dict(self.holding),
                    act=None if cur is None else dict(id=cur["id"], kind=cur["kind"], target=cur["target"],
                                                     phase=self.phases[0]["type"] if self.phases else None),
                    base=self.base["mode"], at=list(self.base["at"]), yaw=float(self.base["yaw"]),
                    support=self.drive.sup.tolist(), pelvis=self.d.xpos[self.bm.pelvis].tolist())

    def place(self, mode, at, yaw, lean=0.0, spine=0.0, twist=0.0):
        """AN INSTRUMENT'S PLACEMENT (tests and tools only, never her conduct): her body put at once in a base posture ('stand',
        'heels', 'tall') at rest, with no act under way, as if she had come there"""
        self.base = dict(mode=mode, at=_lst(at), yaw=float(yaw), lean=float(lean), spine=float(spine), twist=float(twist))
        self.arms = {sd: dict(mode="relaxed") for sd in "LR"}
        self.phases = []
        self.child = Child(self.m, self.d, self.scene.g1_set)
        self._put(self._pose())

    def _put(self, pose):
        """her body put in `pose` at rest (an instrument's placement), her plan and her joints' targets with it"""
        self.scene.set_parent(pose)
        mujoco.mj_forward(self.m, self.d)
        segs = kin.fk(pose)
        self.written = (np.array([segs[s][0] for s in kin.SEGS]), np.array([_mat_to_quat(segs[s][1]) for s in kin.SEGS]))
        tg = self.bm.targets(segs)
        self.drive.set_tick(tg, tg)
        self.drawn_face = (self._face_key(), self._gaze_head(pose, segs))
        self.trunk_at = None
        self.placed = False

    def give_toy(self, side, toy):
        """AN INSTRUMENT'S PLACEMENT (tests and tools only): a toy put in her hand at once, carried before her"""
        self.arms[side] = dict(mode="at", to=CARRY, shape1=dict(curl=.9, thumb=.8, index=None), stay=True)
        pose = self._pose()
        self._put(pose)
        hb = self.bm.seg_body[f"hand_{side}"]
        g = self.d.site_xpos[self.grip_site[side]].copy()
        b = self.toys[toy]
        adr = self.m.jnt_qposadr[self.m.body_jntadr[b]]
        self.d.qpos[adr:adr + 3] = g + np.array([0, 0, -0.01])
        self.d.qvel[self.m.jnt_dofadr[self.m.body_jntadr[b]]:self.m.jnt_dofadr[self.m.body_jntadr[b]] + 6] = 0
        mujoco.mj_forward(self.m, self.d)
        self.phases = [dict(type="grasp", side=side, toy=toy)]
        self._ph_grasp(None, self.phases[0])
        self.phases = []

    def busy(self):
        return self.cur["body"] is not None or bool(self.queue["body"])

    def _end(self, a, status, why=""):
        a["status"] = status; a["why"] = why; a["end"] = int(self.w.tick)
        if a["chan"] == "body" and self.cur["body"] == a["id"]:
            self.cur["body"] = None
        if a["chan"] == "gaze" and self.cur["gaze"] == a["id"]:
            self.cur["gaze"] = None


    # ------------------------------------------------------------------ L1: every tick
    def tick_begin(self):
        """her acts advance one tick; her pose at the tick's end is found and the steps' interpolation set up (the world calls it
        before the tick's physics)"""
        self._tick_begin()
        self._set_aim()

    def _tick_begin(self):
        m, d = self.m, self.d
        self.tick = int(self.w.tick)
        if self.cancels:                                                    # cancels for this tick (P3's cancel(id, tick))
            due = [c for c in self.cancels if c[1] <= self.tick + 1]
            self.cancels = [c for c in self.cancels if c[1] > self.tick + 1]
            for i, _t in due:
                self.cancel(i)
        self.eoc = self._eyes_on_child()                                    # an ask pending in her conduct (A51)
        self.still = self._still()                                          # a formal trial holding her still (4.8)
        self.arrived_now = dict(self.arrived); self.arrived = {"L": False, "R": False}   # her hands that met their link last tick
        self.child_prev = self.child_pts                                    # (the calm step's last look: the child as the last tick began)
        self.child = Child(m, d, self.scene.g1_set)
        self.child_pts = self.child.foot_pts[:, :3].copy()
        self._anchor_pressed()                                              # A4: pressed, she is where the child pushed her
        self.push_on = dict(self.yielding)                                  # the chains yielding as this tick begins
        self.lag = any(self.stopped.values())                               # a chain stopped last tick: her plan waits a tick for her
        self._offset_at_start = {c: self.offset[c].copy() for c in CHAINS}
        self._standoff_at_start = self.standoff.copy()                     # A108: and her standoff (the guard's way back)
        planned0 = self.written
        if self.placed:
            if not (self.queue["body"] or self.queue["gaze"] or self.cur["body"] is not None or self.cur["gaze"] is not None):
                self.moving = False
                self._set_hold_targets()
                self._drive_tick(None)
                return
            self.placed = False
        self._gaze_tick()
        self._body_tick()
        if self.cur["body"] is None and not self.phases and not any(self.yielding.values()):
            self._fold_core()                                               # at rest, where the child pushed her is where she is
        self._hold_tick()
        self._still_tick()
        self._support_tick()
        self._decay_offsets()
        active = (self.cur["body"] is not None or self.cur["gaze"] is not None or bool(self.holds) or bool(self.glances)
                  or self.look.get("target") not in (None, "ahead") or any(a["mode"] != "relaxed" for a in self.arms.values())
                  or self.base["mode"] not in ("held", "stand", "heels", "tall", "sofa", "lying") or any(np.any(self.offset[c]) for c in CHAINS)
                  or self.base.get("dirty", False) or (self.drawn_face is not None and self._face_key() != self.drawn_face[0])
                  or (self.drawn_face is None and self._face_key() != self._face_key_of(kin.FACE_NEUTRAL))
                  or any(self.stopped.values()) or any(self.yielding.values()))
        self.base["dirty"] = False
        if not active:
            self.moving = False
            self.planned_trunk_moved = False
            self._set_hold_targets()
            self._drive_tick(None)
            return
        pose = self._standoff(self._pose())
        segs = kin.fk(pose)
        pos1 = np.array([segs[s][0] for s in kin.SEGS])
        mv = np.linalg.norm(pos1 - planned0[0], axis=1)
        jump = float(mv[0])
        if (jump > MAX_JUMP_M or float(mv.max()) > MAX_LIMB_JUMP_M) and not self.lag:
            self.stats["jumps_refused"] = self.stats.get("jumps_refused", 0) + 1   # never a teleport: a plan that would move
            cur = self._act(self.cur["body"]) if self.cur["body"] is not None else None   # her body faster than a person is a
            if cur is not None:                                                  # bug; the act stops and she stays as she is
                self._end(cur, "refused", f"her body would have jumped {float(mv.max()):.2f} m in a tick (a planning fault: "
                                          f"{kin.SEGS[int(np.argmax(mv))]}, phase {self.phases[0]['type'] if self.phases else None})")
            self.phases = []
            if self.prev is not None:                                       # she stays as she was planned: her base and arms as they
                self.base = _unplain(self.prev["base"]); self.base["dirty"] = True   # were when last planned
                self.arms = _unplain(self.prev["arms"])
                for c in CHAINS:
                    self.offset[c] = self._offset_at_start[c].copy()
                so = self.prev.get("standoff")                              # A108: and her standoff as it was when that pose was drawn
                self.standoff = np.array(so if so is not None else self._standoff_at_start, dtype=np.float64)
                self._rebase_on_drawn(planned0[0][0])                       # A108: her plan agrees with the pose her body was drawn at
            self.phases = [ph for ph in self._settle_phases() if ph["type"] != "plan"]   # A105/A108: never left mid-transition (a
                                                                            # half kneel finishes; until A108 the phases asked for were
                                                                            # dropped, and she stood half knelt for the rest of the day)
            self.moving = False
            self.planned_trunk_moved = False
            self._set_hold_targets()
            self._drive_tick(None)
            return
        self.prev = dict(base=_plain(self.base), arms=_plain(self.arms), standoff=_lst(self.standoff))   # what this plan comes
                                                                            # from (the guard's way back; A108: with her standoff)
        quat1 = np.array([_mat_to_quat(segs[s][1]) for s in kin.SEGS])
        ci = kin.SEGS.index("chest")
        self.planned_trunk_moved = bool(float(np.linalg.norm(pos1[ci] - planned0[0][ci])) > K.TRUNK_STILL_M
                                        or abs(float(quat1[ci] @ planned0[1][ci])) < math.cos(math.radians(K.TRUNK_STILL_DEG) / 2))
        self.moving = not (np.array_equal(pos1, planned0[0]) and np.array_equal(quat1, planned0[1]))
        self.written = (pos1, quat1)
        fk_ = self._face_key()
        gh = self._gaze_head(pose, segs)
        dg = self.drawn_face[1] if self.drawn_face is not None else None
        turned = dg is None or (np.linalg.norm(gh) > 1e-9) != (np.linalg.norm(dg) > 1e-9) or \
            (np.linalg.norm(gh) > 1e-9 and float(unit(gh) @ unit(dg)) < math.cos(math.radians(FACE_REDRAW_DEG)))
        draw_face = self.drawn_face is None or fk_ != self.drawn_face[0] or turned
        self.scene.set_parent(pose, body=False, face=draw_face)
        if draw_face:
            self.drawn_face = (fk_, gh)
        self._set_hold_targets()
        self._drive_tick(self.bm.targets(segs))

    def _drive_tick(self, t1):
        """her joints' targets for this tick (parent_body.Drive): from her plan at the last tick's end (or, for a chain stopped
        then, from where her body is) to t1 (None: her plan unchanged), her pelvis's among them (her support carries her there)"""
        dr = self.drive
        t0 = dr.t1
        if t1 is None:
            t1 = t0
        for k, c in enumerate(CHAINS):                                      # a chain the child presses lets its tone go (her legs' kept)
            if self.yielding[c]:
                dr.relax(k)
        if any(self.stopped.values()):
            now = self.bm.read(self.d)
            t0 = [x.copy() for x in t0]
            for k, c in enumerate(CHAINS):
                if not self.stopped[c]:
                    continue
                sel = self.bm.ball_chain == k; t0[2][sel] = now[2][sel]
                selh = self.bm.hinge_chain == k; t0[3][selh] = now[3][selh]
                if c == "core":
                    t0[0], t0[1] = now[0], now[1]
            t0 = tuple(t0)
            self.stopped = {c: False for c in CHAINS}
        if self.core_hold is not None and (self.yielding["core"] or not (self.yielding["arm_L"] or self.yielding["arm_R"])):
            self.core_hold = None                                           # the arm it stopped with is free (or the child presses her
                                                                            # trunk itself: its own yield leads)
        held = {k: (PB.tg_unplain(self.held_at[c]) if c == "core" else None) for k, c in enumerate(CHAINS)
                if self.yielding[c] and not self._moving_chain(c) and self.held_at[c] is not None}
        if self.core_hold is not None:                                      # her trunk, stopped with a pressed arm it carries, holds
            held[2] = PB.tg_unplain(self.core_hold)                         # there while that arm yields (_carry_stop)
        if held:                                                            # a still chain the child presses: her body holds the pose it
            t1 = [x.copy() for x in t1]                                     # had as the push began (never back into it; never sagging
            now = self.bm.read(self.d)                                      # with it), an arm goes where the push has put it (its own
            for k, ha in held.items():                                      # weight carried: it never springs back into the child)
                ha = now if ha is None else ha
                sel = self.bm.ball_chain == k; t1[2][sel] = ha[2][sel]
                selh = self.bm.hinge_chain == k; t1[3][selh] = ha[3][selh]
                if k == 2:
                    t1[0], t1[1] = ha[0], ha[1]
            t1 = tuple(t1)
        for c in CHAINS:
            if not self.yielding[c]:
                self.held_at[c] = None
        dr.set_tick(t0, t1, legs_rest=self.base["mode"] not in ("walk", "turn"))   # her legs stiff only to step (A25b: carried, she
                                                                            # kneels, shuffles and rests with relaxed legs)

    def _moving_chain(self, c):
        """whether her plan moves chain c now: her body while her base moves or her trunk leans (a phase under way on them), an arm
        while it reaches or moves (A4 stops and backs off a chain she moves into the child; a still one the child presses gives as a
        body does)"""
        if self.on_child.get(c) or self.arm_in.get(c):                      # resting on the child is moving off it, and an arm pressed
            return True                                                     # on itself draws in
        if c == "core":
            ph = self.phases[0]["type"] if self.phases else None
            return self.base["mode"] in MOVING or ph in ("lean", "walk", "turn", "kneel_down", "shuffle", "knee_turn", "sit") or \
                (ph == "reach" and "lean_to" in self.phases[0])
        a = self.arms[c[-1]]
        return a.get("mode") in ("move", "point") or (a.get("mode") == "at" and a.get("shake"))

    def _anchor_pressed(self):
        """A4 on her body: each chain the child pressed past its cap in the last tick (yielding) is, in her plan, where the child
        pushed it (her pelvis on the floor plan for the rest of her, her hand's grip for an arm: her plan's offset takes up the
        difference), and backs off YIELD_M_PER_TICK along its way out while it still presses; backed off YIELD_MAX_M in one push and
        still pressed on, the act under way is given up"""
        for c in CHAINS:
            if not self.yielding[c]:
                continue
            if c == "core":
                act = self.d.xpos[self.bm.pelvis]; plan = self.written[0][0]
                dv = np.array([act[0] - plan[0], act[1] - plan[1], 0.0])
            else:                                                           # an arm: where the child pushed her hand, as the push
                sd = c[-1]                                                  # begins (after, her plan backs off from there: a hand
                dv = np.zeros(3)                                            # short of a plan the floor clamps is no push)
                if not self.push_on[c]:
                    dv = self.d.site_xpos[self.grip_site[sd]] - self._grip_now(sd)[0]
                    n = float(np.linalg.norm(dv))
                    if n > K.YIELD_MAX_M:
                        dv = dv * (K.YIELD_MAX_M / n)
            step = self.ydir[c] * K.YIELD_M_PER_TICK if self._moving_chain(c) and self.ymoved[c] < K.YIELD_MAX_M else np.zeros(3)
            if c != "core" and self.arm_in.get(c):                          # pressed on her arm itself: she draws her hand in toward her
                step = self._draw_in(c[-1])                                 # shoulder (moving her hand along the contact would swing her
                                                                            # elbow into it)
            if c == "core":                                                 # (still, she stays where it pushed her: A4's back-off is
                                                                            # for a chain she moves, a hand's length at most)
                step = np.array([step[0], step[1], 0.0])
                if not self._core_room(step + dv):
                    step = np.r_[self._core_slide(step)[:2], 0.0]
            elif step[2] < 0 and self._chain_low(c) < 0.20:
                step = step * [1.0, 1.0, 0.0]                               # never backed off into the floor
            if c != "core" and self._moving_chain(c) and float(np.linalg.norm(step)) < 0.25 * K.YIELD_M_PER_TICK \
                    and not any(h.side == c[-1] for h in self.holds):
                step = self._draw_in(c[-1])                                 # its way out barred (the child's arm lying on her hand by
                                                                            # the floor, or backed off as far as it goes and still
                                                                            # pressed): she draws her hand in from under it
            self.offset[c] = self.offset[c] + dv + step
            self.ymoved[c] += float(np.linalg.norm(step))
            b = self.base
            if c == "core" and b["mode"] in ("heels", "tall") and (b.get("lean") or b.get("spine")):   # pressed on as she leans
                d_ = K.YIELD_LEAN_DEG_PER_TICK                               # over it, she straightens up away from it (her knees on
                b["lean"] = max(0.0, float(b.get("lean", 0.0)) - d_)         # the floor do not back her off: her trunk does)
                b["spine"] = max(0.0, float(b.get("spine", 0.0)) - d_)
                b["dirty"] = True
            if c == "core" and self.ymoved[c] >= K.YIELD_MAX_M and self.blocked is None and self.cur["body"] is not None:
                self.blocked = int(self.cur["body"])                        # backed off as far as she will: that act is given up (an
                                                                            # arm pressed on waits, and gives up after PATIENCE_TICKS)

    def _set_aim(self):
        """the link each hand is reaching onto for a hold, and that hold's cap (from the hold phase ahead of it): {side: (body,
        cap)}; None when neither is"""
        aim = {}
        for sd in "LR":
            a = self.arms[sd]
            to = a.get("to") or {}
            if a.get("mode") in ("move", "at") and to.get("k") == "link":
                ph = next((ph for ph in self.phases if ph.get("type") == "hold" and ph.get("side") == sd), None)
                if ph is not None:                                          # a brief hold (the pull, the turn) grows to her brief cap
                    cap = K.CAP_TWO_BRIEF / 2 if ph.get("brief") else float(ph["cap"])   # a hand
                    aim[sd] = (int(to["body"]), max(cap, K.TOUCH_N))
        self.aim = aim or None

    def _face_key(self):
        return self._face_key_of(self.face)

    @staticmethod
    def _face_key_of(f):
        return tuple(sorted((k, float(v)) for k, v in f.items()))

    @staticmethod
    def _gaze_head(pose, segs):
        if pose.gaze is None:
            return np.zeros(3)
        hp, hR = segs["head"]
        return hR.T @ (np.asarray(pose.gaze) - hp)

    def before_step(self, s):
        """L0 before physics step s of the tick: her holds' forces (on the child, their reaction on her hands, her arms' effort),
        then her joints' torques and her balance (parent_body.Drive)"""
        if self.holds or self.xfrc_bodies:
            self._apply_holds(s)
        else:
            self.drive.effort[:] = 0.0                                      # no hold: no hold's effort in her joints (S5a: a finished
                                                                            # hold's effort had stayed in her command, and a replay
                                                                            # restored after a hold inherited it)
            self._effort(0.0, 0.0, float(self.body_f.sum()))                 # her brief caps' clock runs at rest too
        self.drive.step(s)

    def after_step(self, s):
        """L0 after physics step s: her contacts with the child read from the step's contacts (the stop, A4; her pain). Each
        contact's whole force (its normal and its friction, MuJoCo's mj_contactForce) counts toward her stop and her effort; its
        normal force toward her pain (the child's own pain law's measure, A12). A holding hand's contact with the link it holds is
        that hold's (her caps count it), never a push to stop for"""
        m, d = self.m, self.d
        nc = d.ncon
        mine0 = mine1 = None
        held = None
        self.hold_touch = {"L": 0.0, "R": 0.0}
        self.toy_touch = {"L": False, "R": False}
        if nc:
            geom = d.contact.geom
            gs = self.geom_seg[geom]
            g1_0 = self.g1_geom[geom[:, 0]]; g1_1 = self.g1_geom[geom[:, 1]]
            if self.holding["L"] is not None or self.holding["R"] is not None:
                held = {self.toy_names.index(t): ("arm_L" if sd == "L" else "arm_R") for sd, t in self.holding.items() if t is not None}
                hk = np.array(sorted(held))
                ty0 = self.toy_of_geom[geom[:, 0]]; ty1 = self.toy_of_geom[geom[:, 1]]
                mine0 = ((gs[:, 0] >= 0) | np.isin(ty0, hk)) & g1_1
                mine1 = ((gs[:, 1] >= 0) | np.isin(ty1, hk)) & g1_0
            else:
                mine0 = (gs[:, 0] >= 0) & g1_1
                mine1 = (gs[:, 1] >= 0) & g1_0
        self.drive.push = np.zeros(3)                                       # the child's push on her body this step (her support reads it)
        if mine0 is None or not (mine0.any() or mine1.any()):             # the common step: nothing of hers touches the child
            for c in CHAINS:
                self.over[c] = 0
                self.on_child[c] = False
                self.arm_in[c] = False
            self.body_f[:] = 0.0
            if self.body_hist.any():
                self.body_hist = np.roll(self.body_hist, -1, axis=0); self.body_hist[-1] = 0.0
            if self.pain_any:
                self._pain_step(np.zeros(len(kin.SEGS)))
            return
        force = np.zeros(3); away = np.zeros((3, 3)); seg_f = np.zeros(len(kin.SEGS)); fsum = np.zeros(3); up = np.zeros(3)
        armbody = np.zeros(2)
        self.aim_f = {"L": 0.0, "R": 0.0}
        hold_on = {h.side: h.body for h in self.holds}
        f6 = self._f6
        con = d.contact
        for i in np.nonzero(mine0 | mine1)[0]:
            c = con[i]
            if c.efc_address < 0:
                continue
            mujoco.mj_contactForce(m, d, int(i), f6)                       # in the contact's frame: its normal, then its friction
            fn = float(f6[0])
            fw = float(math.sqrt(f6[0] * f6[0] + f6[1] * f6[1] + f6[2] * f6[2]))
            n = np.array(c.frame[:3])
            side0 = bool(mine0[i])
            seg = int(gs[i, 0] if side0 else gs[i, 1])
            other = int(m.geom_bodyid[int(geom[i, 1] if side0 else geom[i, 0])])
            if seg >= 0:
                ch = CHAINS.index(SEG_CHAIN[kin.SEGS[seg]]); seg_f[seg] += fn
            else:
                ch = CHAINS.index(held[int(ty0[i] if side0 else ty1[i])])
                self.toy_touch[CHAINS[ch][-1]] = True                       # a toy in her hand touching the child (her aim holds)
            if c.dist < 0 and seg >= 0:
                self.stats["pen_max"] = max(self.stats["pen_max"], float(-c.dist))
            if seg in self.hand_idx_set:                                    # her hand on the link it holds, or arriving on the link it
                sd_ = kin.SEGS[seg][-1]                                     # reaches for (the hold it starts): that touch is her act's,
                aim = self.aim.get(sd_) if self.aim is not None else None   # counted in her effort against her caps (one hand's brief
                if hold_on.get(sd_) == other or (aim is not None and other == aim[0]):   # cap), never a push to stop for until it
                    self.hold_touch[sd_] += fw                              # passes that hold's own cap (below); the reach ends where
                    if aim is not None and other == aim[0]:                 # it arrives (_ph_reach)
                        self.aim_f[sd_] += fw; self.arrived[sd_] = True
                    continue
            force[ch] += fw
            if seg >= 0 and ch < 2 and not kin.SEGS[seg].startswith("hand"):
                armbody[ch] += fw                                           # pressed on her upper arm or forearm, not her hand
            nout = -n if side0 else n                                       # the way out of the child, at her side of the contact
            away[ch] += nout * fn
            if ch == 2 and seg >= 0:                                        # the child's force on her body (its normal and friction:
                self.drive.push += np.asarray(c.frame).reshape(3, 3).T @ (f6[:3] if not side0 else -f6[:3])   # on her, world)
            fsum[ch] += max(fn, 0.0)
            up[ch] += max(0.0, float(nout[2])) * fn                         # the child bearing her up: she rests on it
        self.body_f = force.copy()
        self.body_hist = np.roll(self.body_hist, -1, axis=0); self.body_hist[-1] = force
        for k, c in enumerate(CHAINS):                                      # she rests some of her weight on the child: she moves off
            self.on_child[c] = bool(up[k] > self._chain_cap(c))            # it, still or not (A4's back-off; never held there)
        for k in (0, 1):                                                    # an arm pressed on its upper arm or forearm draws itself in
            self.arm_in[CHAINS[k]] = bool(armbody[k] > self._contact_cap())
        for k, sd in ((0, "L"), (1, "R")):                                  # an arm touching the child builds no tone against it
            if force[k] > 0.0 or self.hold_touch[sd] > 0.0:
                self.drive.relax(k)
            over = self.hold_touch[sd] - self._touch_cap(sd)               # her hand pressing the link it holds or reaches for past
            if over > 0.0:                                                  # that hold's own cap presses the child (A4)
                force[k] += over
        if force[2] > 0.0:
            self.drive.relax(2)
        self._stop_steps(force, away, fsum)
        self._pain_step(seg_f)

    def _stop_steps(self, force, away, fsum):
        """A4, per chain: a contact over a resting hand's weight (her body's cap: no act of hers pushes the child with her body) for
        YIELD_STEPS steps stops the chain where it is (its joints' targets become their positions: parent_body.Drive.hold_chain);
        it yields (its plan follows where the child pushes it, backing off, _anchor_pressed) until the force is gone"""
        for k, c in enumerate(CHAINS):
            cap = self._chain_cap(c)
            fk = float(force[k])
            self.stats["contact_peak"][c] = max(self.stats["contact_peak"][c], fk)
            self.touching[c] = max(self.touching[c], fk)
            if fk > cap:
                self.over[c] += 1
                self.stats["over_run"] = max(self.stats.get("over_run", 0), self.over[c])
                if self.over[c] >= K.YIELD_STEPS:
                    if not self.push_on[c]:                                 # a push begun in this tick stops the chain (a chain
                        self.drive.hold_chain(k)                            # already yielding at the tick's start is backing off
                        self.stopped[c] = True                              # along its plan: _anchor_pressed)
                        if self.held_at[c] is None:
                            self.held_at[c] = PB.tg_plain(self.drive.stop[k])   # the pose it holds while still and pressed
                        if c != "core":
                            self._carry_stop()
                    self.yielding[c] = True
                    y = self._yield_dir(c, away[k], float(fsum[k]))
                    if y is not None:
                        self.ydir[c] = y
            else:
                self.over[c] = 0

    def _draw_in(self, sd):
        """a pressed hand's way out toward her shoulder, at twice the back-off's pace, never nearer the shoulder than 25 cm and
        never more than DRAW_IN_MAX_M in one push (a hand the child holds where it is stays: its plan drawn on without it had run
        46 cm off it and tripped the jump guard)"""
        if self.ymoved[f"arm_{sd}"] >= K.DRAW_IN_MAX_M:
            return np.zeros(3)
        g = self._grip_now(sd)[0]
        v = self._shoulder(sd, self.scene.pose) - g
        nv = float(np.linalg.norm(v))
        return v / nv * min(2 * K.YIELD_M_PER_TICK, max(0.0, nv - 0.25)) if nv > 1e-6 else np.zeros(3)

    def _carry_stop(self):
        """A4 up her body: an arm the child presses stops where it is, and so does her trunk, which carries it, when her plan is
        moving her trunk (a lean, her base): it holds there while the arm yields (_drive_tick), never carrying the stopped arm on
        into the child (the physical build's finding: her lean went on and pressed her stopped forearm down on the child's wrist at
        100 N for a tick, 10 J of work)"""
        kc = CHAINS.index("core")
        if self.core_hold is None and not self.yielding["core"] and self._moving_chain("core"):
            self.drive.hold_chain(kc)
            self.stopped["core"] = True
            self.core_hold = PB.tg_plain(self.drive.stop[kc])

    def _yield_dir(self, c, away, fsum=0.0):
        """the way a chain backs off: along its contacts (their normals weighted by their forces) as far as they agree (their
        resultant over their sum: 1 when every contact pushes it the same way, 0 when they cancel, as with a leg between her shins),
        the rest away from the child: from its centre of mass out through her pelvis (her body) or her hand (an arm). Her body backs
        off on the floor plan, never toward the child's centre (the W2 verifier's finding: two contacts pointing opposite ways had
        flipped her way out every step); an arm low by the floor never into it (a forearm under the child's hip is drawn out along
        the floor)"""
        com = self.d.subtree_com[self.child_root][:2]
        if c == "core":
            her = self.d.xpos[self.bm.pelvis][:2]
            h = away[:2] / fsum if fsum > 0 else np.zeros(2)
        else:
            her = self._grip_now(c[-1])[0][:2]
            h = away / fsum if fsum > 0 else np.zeros(3)
            if h[2] < 0 and self._chain_low(c) < 0.20:
                h = h * [1.0, 1.0, 0.0]
        v = her - com; nv = float(np.linalg.norm(v))
        if nv > 1e-6:
            out = v / nv
        else:                                                               # right over its centre: back the way she faces
            yaw = float(self.base["yaw"]); out = -np.array([math.cos(yaw), math.sin(yaw)])
        agree = min(1.0, float(np.linalg.norm(h)))
        if c == "core":
            if float(h @ out) <= 0:
                agree, h = 0.0, np.zeros(2)
            w_ = h + (1.0 - agree) * out
            nw = float(np.linalg.norm(w_))
            return np.array([w_[0] / nw, w_[1] / nw, 0.0]) if nw > 1e-9 else np.array([out[0], out[1], 0.0])
        w_ = h + (1.0 - agree) * np.array([out[0], out[1], 0.0])
        nw = float(np.linalg.norm(w_))
        return w_ / nw if nw > 1e-9 else np.array([out[0], out[1], 0.0])

    def _core_slide(self, step):
        """her body's back-off turned along the furniture behind her: the least turn (30 deg steps, at most 90 deg either way) with
        room; none, she stays"""
        for deg in (30, -30, 60, -60, 90, -90):
            r = math.radians(deg); c_, s_ = math.cos(r), math.sin(r)
            st = np.array([c_ * step[0] - s_ * step[1], s_ * step[0] + c_ * step[1], 0.0])
            if self._core_room(st):
                return st
        return np.zeros(3)

    def _core_room(self, step):
        """whether her body can back off by `step` on the floor plan: not into furniture or a wall (her half-width clear, or no
        nearer to it than she is)"""
        at0 = np.asarray(self.base["at"], float) + self.offset["core"][:2]
        at1 = at0 + step[:2]
        if not self._in_plan(at1):
            return False
        d1 = float(self.plan.dist[self.plan.cell(at1)])
        return d1 >= K.BODY_R_M or (self._in_plan(at0) and d1 >= float(self.plan.dist[self.plan.cell(at0)]))

    def _pain_step(self, seg_f):
        """her pain window: each segment's last 5 steps of force from the child; a 10 ms mean over HER_PAIN_N is a hit"""
        self.pain_win = np.roll(self.pain_win, -1, axis=1); self.pain_win[:, -1] = seg_f
        self.pain_any = bool(self.pain_win.any())
        mean = self.pain_win.mean(axis=1)
        for i in np.nonzero(mean > K.HER_PAIN_N)[0]:
            if not self.hits or self.hits[-1][0] != self.tick or self.hits[-1][1] != kin.SEGS[i]:
                self.hits.append((self.tick, kin.SEGS[i], round(float(mean[i]), 1)))
                del self.hits[:-50]

    def _chain_low(self, c):
        """the height of her hand's grip on an arm chain (m)"""
        if c == "core":
            return 1.0
        return float(self._grip_now(c[-1])[0][2])

    def tick_end(self):
        """after the tick's physics: the holds' tick counters, the stop's release, her trunk's motion over the tick (the report's
        trunk field)"""
        for h in self.holds:
            h.at_cap_ticks = h.at_cap_ticks + 1 if h.at_cap_steps >= STEPS else 0
            h.at_cap_steps = 0
        self.last_touch = dict(self.touching)
        for c in CHAINS:
            cap = self._chain_cap(c)
            if self.yielding[c] and self.touching[c] <= cap:
                self.yielding[c] = False                                    # the force is gone: the act resumes (A4)
            if not self.yielding[c]:
                self.ymoved[c] = 0.0                                        # the push is over
            self.quiet[c] = 0 if self.touching[c] > cap else self.quiet[c] + 1
            self.touching[c] = 0.0
        if any(self.yielding.values()):
            self.stats["yield_ticks"] += 1
        cb = self.bm.seg_body["chest"]
        now = (self.d.xpos[cb].copy(), _mat_to_quat(self.d.xmat[cb].reshape(3, 3)))
        if self.trunk_at is None:
            self.trunk_moved = False
        else:
            self.trunk_moved = bool(float(np.linalg.norm(now[0] - self.trunk_at[0])) > K.TRUNK_STILL_M
                                    or abs(float(now[1] @ self.trunk_at[1])) < math.cos(math.radians(K.TRUNK_STILL_DEG) / 2))
        self.trunk_at = now
        f = self._face_drawn()
        self.face_prev = self.face_shown
        self.face_shown = tuple(sorted((k, float(v)) for k, v in f.items() if k not in ("jaw", "blink")))

    def _chain_cap(self, c):
        """A4's "the act's cap" for a chain's contacts with the child: her body's is a resting hand's weight (no act of hers pushes
        the child with her body); an arm reaching onto the child for a hold, or holding it, may touch its other links on the way up
        to that hold's own cap (her hand on the link it holds or reaches for is the hold's own touch, counted against her caps); a
        resting hand's weight otherwise"""
        if c == "core":
            return self._contact_cap()
        sd = c[-1]
        cap = K.TOUCH_N
        if self.aim is not None and sd in self.aim:
            cap = max(cap, float(self.aim[sd][1]))
        for h in self.holds:
            if h.side == sd:
                cap = max(cap, K.CAP_TWO_BRIEF / 2 if h.brief else float(h.cap))
        return min(cap, K.CAP_ONE)

    def _touch_cap(self, sd):
        """what her hand may press the link it holds or reaches for with: that hold's own cap (a brief one's, her brief cap a hand),
        a resting hand's weight at least"""
        cap = K.TOUCH_N
        if self.aim is not None and sd in self.aim:
            cap = max(cap, float(self.aim[sd][1]))
        for h in self.holds:
            if h.side == sd:
                cap = max(cap, K.CAP_TWO_BRIEF / 2 if h.brief else float(h.cap))
        return min(cap, K.CAP_ONE_BRIEF)

    def _contact_cap(self):
        """her body's contact cap: no act of hers pushes the child with her body (only her holds push, each within its cap), so any
        other contact over a resting hand's weight is yielded (A4, tightened: section 4.2's act caps are her holds')"""
        return K.TOUCH_N

    def _decay_offsets(self):
        """a chain backed off comes back at YIELD_BACK_M_PER_TICK once the child has left it alone for her reaction time (A4: the act
        resumes when the force is gone), and only where it then stays CLEAR_M from the child: she never presses back into it (a
        hand whose proxy is off, on the child, is left out of that measure: its own guard keeps it outside)"""
        todo = [c for c in CHAINS if not self.yielding[c] and self.quiet[c] >= K.REACTION_TICKS and np.any(self.offset[c])]
        for c in todo:
            old = self.offset[c].copy()
            n = float(np.linalg.norm(old)); step = K.YIELD_BACK_M_PER_TICK
            self.offset[c] = np.zeros(3) if n <= step else old * (1 - step / n)
            if self._chain_clear(c) < K.CLEAR_M:
                self.offset[c] = old

    def _chain_clear(self, c):
        """the chain's least distance to the child with her pose as her offsets now make it (a trial pose: her elbows' swing and
        the reach errors are put back after it); a hand whose proxy is off (on the child, or holding a toy) is left out"""
        sw = dict(self.swivel); errs = {sd: self.arms[sd].get("err") for sd in "LR"}
        try:
            pose = self._pose(trial=True)
        finally:
            self.swivel = sw
            for sd in "LR":
                if errs[sd] is None:
                    self.arms[sd].pop("err", None)
                else:
                    self.arms[sd]["err"] = errs[sd]
        return self._clearance(pose, segs=(c,), skip_off=True)

    # ------------------------------------------------------------------ the capped springs
    def _set_hold_targets(self):
        """each hold's target moves through the tick from where it was to where her act puts it (Hold.next), or stays"""
        for h in self.holds:
            p = h.point(self.d)
            h.t0 = (h.t1 if h.t1 is not None else p).copy()
            nxt = getattr(h, "next", None)
            h.t1 = np.asarray(nxt if nxt is not None else h.t0, float).copy()
            h.next = None

    def _apply_holds(self, s):
        """her holds on the child this step (4.1, 4.2, A25): each a capped spring toward where she means the held point to be; within
        the hold's own cap,
        then her holds together within what her caps leave after her body's contacts and her holding hands' own touch; applied on
        the held link at the held point, its reaction on her hand at her grip, and her arm and trunk exerting it (J^T F on her
        joints, within her strength: parent_body.Drive)"""
        m, d = self.m, self.d
        for b in self.xfrc_bodies:
            d.xfrc_applied[b] = 0.0
        self.xfrc_bodies = []
        self.hold_zone[:] = 0.0
        self.drive.effort[:] = 0.0
        if not self.holds:
            self._effort(0.0, 0.0, float(self.body_f.sum()))
            return
        a = (s + 0.5) / STEPS
        jacp = self._jac
        raw, pts, grips = [], [], []
        for h in self.holds:
            p = h.point(d)
            mujoco.mj_jac(m, d, jacp, None, p, h.body)
            v = jacp @ d.qvel
            tgt = h.t0 + a * (h.t1 - h.t0)
            vt = (h.t1 - h.t0) / TICK_S
            g = d.site_xpos[self.grip_site[h.side]]
            f = K.HOLD_K * (tgt - p) + K.HOLD_C * (vt - v)
            n = float(np.linalg.norm(f))
            if n > h.cap:
                f = f * (h.cap / n)
            arm = self.arms.get(h.side, {})
            if arm.get("mode") == "hold" and arm.get("hold") == h.name:     # HER GRIP IS HER HAND: where the physics has put her hand
                Rl = d.xmat[h.body].reshape(3, 3)                           # off its grip on the held point (her arm could not carry
                e = float(np.linalg.norm(g - (p + Rl @ np.asarray(arm["goff"], float))))   # the load, or the child pulled away), the
                if e > K.GRIP_TOL_M:                                        # grip holds less, and nothing past a hand's length
                    f = f * max(0.0, 1.0 - (e - K.GRIP_TOL_M) / (K.HOLD_SLIP_M - K.GRIP_TOL_M))
            raw.append(f); pts.append(p); grips.append(g.copy())
        brief = any(h.brief for h in self.holds) and self.brief_s < K.BRIEF_S
        one = K.CAP_ONE_BRIEF if brief else K.CAP_ONE
        two = K.CAP_TWO_BRIEF if brief else K.CAP_TWO
        body = self.body_hist.max(axis=0)                                   # what her body already presses on the G1 (the most of the
        touch = self.hold_touch                                             # last 5 steps, per chain) and her holding hands' own touch
        scale = np.ones(len(raw))
        for sd in "LR":
            idx = [i for i, h in enumerate(self.holds) if h.side == sd]
            tot = sum(float(np.linalg.norm(raw[i])) for i in idx)
            lim = max(0.0, one - float(body[CHAINS.index(f"arm_{sd}")]) - float(touch.get(sd, 0.0)))
            if tot > lim:
                for i in idx:
                    scale[i] = lim / tot
        tot = sum(float(np.linalg.norm(raw[i])) * scale[i] for i in range(len(raw)))
        lim = max(0.0, two - float(body.sum()) - float(sum(touch.values())))
        if tot > lim:
            scale *= lim / tot
        effort = 0.0
        per_side = {"L": 0.0, "R": 0.0}
        dofs = self.bm.dofs
        for i, h in enumerate(self.holds):
            f = raw[i] * scale[i]
            n = float(np.linalg.norm(f))
            h.force = f; h.peak = max(h.peak, n); h.steps += 1
            if n >= K.AT_CAP * min(h.cap, h.cap * scale[i] if scale[i] < 1 else h.cap) and n > 0:
                h.at_cap_steps += 1
            elif scale[i] < 0.999 and n > 0:
                h.at_cap_steps += 1                                          # held back by her own caps: at her cap
            b = h.body
            d.xfrc_applied[b, :3] += f                                      # on the child, at the held point
            d.xfrc_applied[b, 3:] += np.cross(pts[i] - d.xipos[b], f)
            hb = self.bm.hand_body[h.side]                                  # its reaction on her hand, at her grip (Newton)
            d.xfrc_applied[hb, :3] -= f
            d.xfrc_applied[hb, 3:] -= np.cross(grips[i] - d.xipos[hb], f)
            mujoco.mj_jac(m, d, jacp, None, grips[i], hb)                   # her arm and trunk exert it (within her strength)
            self.drive.effort += jacp[:, dofs].T @ f
            self.xfrc_bodies += [b, hb]
            z = int(self.w.body_zone[b])
            if z >= 0:
                self.hold_zone[z] += n
            effort += n; per_side[h.side] += n
        self.stats["hold_peak"] = max(self.stats["hold_peak"], max((float(np.linalg.norm(h.force)) for h in self.holds), default=0.0))
        tt = float(sum(self.hold_touch.values()))
        self._effort(effort + tt, max(per_side[sd] + self.hold_touch.get(sd, 0.0) for sd in "LR"), effort + tt + float(self.body_f.sum()))

    def _effort(self, two, one, total):
        """her brief caps' clock: time her holds spend over a sustained cap counts toward BRIEF_S; BRIEF_REST_S under both gives it
        back (a blow from the child against her body is not her exertion). Her peak force on the G1 (her holds and her body's
        contacts together) is kept for the instruments"""
        dt = self.m.opt.timestep
        self.effort_peak = max(self.effort_peak, two)
        self.stats["total_peak"] = max(self.stats.get("total_peak", 0.0), total)
        if two > K.CAP_TWO or one > K.CAP_ONE:
            self.brief_s += dt; self.rest_s = 0.0
        else:
            self.rest_s += dt
            if self.rest_s >= K.BRIEF_REST_S:
                self.brief_s = 0.0

    # ------------------------------------------------------------------ her pose at the tick's end
    def _pose(self, base=None, arms=True, look=True, trial=False):
        """her pose from her base, her arms' specs, her look and her face (every IK inside parent_kin's human ranges; the report
        of each IK is kept on the pose)"""
        b = self.base if base is None else base
        oc = self.offset["core"] + np.r_[self.standoff, 0.0]               # (her standoff moves her base, never her hands' targets)
        at = np.asarray(b["at"], float) + oc[:2]
        mode = b["mode"]
        key = (mode, tuple(at), b.get("yaw"), b.get("lean"), b.get("spine"), b.get("twist"), b.get("u"), b.get("s"),
               tuple(b.get("p0") or ()), tuple(b.get("p1") or ()), b.get("seat_z"))
        if mode != "held" and self._pose_cache is not None and self._pose_cache[0] == key:
            p = self._pose_cache[1].copy()
        else:
            p = None
        if p is not None:
            pass
        elif mode == "held":
            p = self.scene.pose.copy() if self.scene.pose is not None else G.born_parent()
        elif mode in ("stand", "turn"):
            p = P.stand(at, b["yaw"])
        elif mode == "walk":
            p, _ = P.walk(np.asarray(b["p0"], float) + oc[:2], np.asarray(b["p1"], float) + oc[:2], b["s"] / P.SPEED)
        elif mode == "kneel_down":
            p = P.kneel_down(at, b["yaw"], b["u"])
        elif mode in ("heels", "tall"):
            p = P.kneel(at, b["yaw"], mode, lean=b["lean"], spine_flex=b["spine"], twist=b["twist"])
        elif mode == "shuffle":
            p = P.kneel_shuffle(np.asarray(b["p0"], float) + oc[:2], np.asarray(b["p1"], float) + oc[:2], b["yaw"], b["u"], "tall")
        elif mode == "sofa":
            p = sit_chair(at, b["yaw"], b["u"], b.get("seat_z", SOFA_SEAT_Z))
        elif mode in ("lie", "lying"):                                      # A124: down onto her front, or lying (tummy time)
            p = lie_pose(at, b["yaw"], b.get("u", 1.0), b.get("lean", LIE_CHEST_UP[0]))
        else:
            raise ValueError(mode)
        if mode not in ("held", "sofa", "lie", "lying") and (self._pose_cache is None or self._pose_cache[0] != key):
            p.pos = p.pos + np.array([0.0, 0.0, self._floor_lift(p)])        # never planned into the floor (A25b)
        if mode != "held" and (self._pose_cache is None or self._pose_cache[0] != key):
            q = p.copy(); q.report = {k: (dict(v) if isinstance(v, dict) else v) for k, v in p.report.items()}
            self._pose_cache = (key, q)
        if p is not None and self._pose_cache is not None:
            p.report = {k: (dict(v) if isinstance(v, dict) else v) for k, v in p.report.items()}
        if arms:
            for sd in "LR":
                if mode in ("lie", "lying") and self.arms[sd].get("mode", "relaxed") == "relaxed":
                    continue                                                # lying, a hand at rest stays on the floor (the pose's own)
                self._arm(p, sd, trial)
        if look:
            tgt = self._look_point()
            if tgt is not None:
                kin.look(p, tgt)
        p.expr = self._face_drawn()
        return p

    def _look_point(self):
        t = self.look.get("target")
        g = self._glance_now() if not self.eoc and self.cur["gaze"] is None else None
        if g is not None and self._resolve_point(g) is not None:
            return self._resolve_point(g)                                   # L1's glance: her eyes on it for its ticks
        if t in (None, "ahead"):
            return None
        return self._resolve_point(t)

    def _resolve_point(self, t):
        """a named target as a world point: the child's eyes, the child, a toy, a fixture, or a point [x, y, z]"""
        ch = self.child
        if isinstance(t, (list, tuple, np.ndarray)):
            return np.asarray(t, float)
        if t in ("child_eyes", "child_periphery", "child_line", "mama"):
            return ch.eyes.copy()
        if t == "child":
            return ch.torso.copy()
        if t in self.toys:
            return self.d.xpos[self.toys[t]].copy()
        if t in self.fixtures:
            return np.array(self.fixtures[t], float)
        if t in BODY_PARTS:
            return self.d.xpos[self.m.body(BODY_PARTS[t][0]).id].copy()
        return None

    def _arm(self, p, sd, trial=False):
        """her arm for the tick's end. Her plan's offsets (A4: where the child pushed her): her body's (core) moves with her whole
        pose and with any hand target fixed in the room or on the child; her arm's own moves that hand. A hand holding the child is
        planned on its held point, open and flat on its surface (its touch keeps it outside, as the rest of her)"""
        a = self.arms[sd]
        mode = a["mode"]
        oa = self.offset[f"arm_{sd}"]; oc = self.offset["core"]
        if mode == "relaxed":
            if np.any(oa):                                                  # a resting arm pushed away stays there
                hp, hR = kin.fk(p)[f"hand_{sd}"]
                a["err"] = float(self._place(p, sd, hp + hR @ GRIP_LOCAL[sd] + oa, hR, dict(p.hand[sd]), cont=True))
            return
        if mode == "point":
            tgt = self._resolve_point(a["target"])
            if tgt is not None:
                a["err"] = float(P.point_at(p, sd, tgt))
            return
        if mode == "hold":
            grip, R, shape = self._hand_now(p, sd, a)
            a["err"] = float(self._place(p, sd, grip + oa + oc, R, shape, cont=True))
            return
        grip, R, shape = self._hand_now(p, sd, a)
        if self._world_target(a["to"]):
            off = oa + oc
        elif mode == "move":                                                # from a point in the room to one on her own body
            off = oa + oc * (1.0 - _smooth(a["t"] / max(a["n"], 1)))
        else:
            off = oa
        want = grip + off
        if mode in ("at", "move") and self._world_target(a["to"]) and a["to"]["k"] != "link" and \
                (mode == "at" or a.get("t", 0) >= a.get("n", 1) - 1):
            want = self._aim_fix(sd, a, want, trial)                        # (never aimed on while it touches the child)
        a["err"] = float(self._place(p, sd, want, R, shape, cont=True))

    def _aim_fix(self, sd, a, want, trial):
        """SHE SEES WHERE HER HAND IS: a hand held at a place (or on the child) is planned where it must be for her real hand to
        arrive there, her plan's aim moved each tick by AIM_GAIN of what her hand as the physics has it still misses (her body
        rests and her arm hangs some centimetres off their plan), at most AIM_MAX_M; never while the hand touches the child (it
        would press on); the aim kept with the arm's plan"""
        fix = np.asarray(a.get("fix", (0.0, 0.0, 0.0)), float)
        if not trial:
            k = self.hand_idx[HAND_SEGS[sd]]
            if not self.pain_win[k, -1] > 0.0 and not self.toy_touch.get(sd, False):
                miss = want - self._grip_now(sd, actual=True)[0]
                cur = self.carry.get(sd)
                if cur is not None and a["to"].get("k") in ("palm", "floor", "show", "toy_at"):
                    # C222 (2026-10-02): SHE SEES WHERE THE TOY IS. A toy she carries hangs from her grip where the weld took it (up to 8 cm
                    # from her grip point, `_ph_grasp`), so a hand placed to its plan puts the toy where her planned hand's ROTATION would,
                    # and her real hand's rotation is off by tens of degrees: day 62's probe, her hand at its plan within 1 mm, the block 10
                    # cm above the palm it was meant for. When the target is the toy's place (the palm, the floor, the show, a toy's spot),
                    # what she still misses is the toy's miss, read from the toy itself
                    try:
                        c_ = np.asarray(self._resolve_hand(a["to"], sd)[0], float)
                        miss = c_ - self.d.xpos[self.toys[cur["toy"]]]
                    except Exception:
                        pass
                a["miss"] = float(np.linalg.norm(miss))
                fix = fix + K.AIM_GAIN * miss
                n = float(np.linalg.norm(fix))
                if n > K.AIM_MAX_M:
                    fix = fix * (K.AIM_MAX_M / n)
                a["fix"] = _lst(fix)
        return want + fix

    def _hand_now(self, p, sd, a):
        """the grip point, hand rotation and hand shape an arm spec asks for at the tick's end"""
        mode = a["mode"]
        if mode == "move":
            u = _smooth(a["t"] / max(a["n"], 1))
            g1, R1 = self._realize(p, sd, *self._resolve_hand(a["to"], sd, p))
            q0 = np.asarray(a["q0"], float)
            pts = [np.asarray(x, float) for x in a.get("path") or [a["g0"]]]
            if a.get("via") is not None:
                pts.append(g1 + self._approach_dir(a["to"], sd) * K.APPROACH_M)
            pts.append(g1)
            grip = _along(pts, u)
            R = _quat_to_mat(_nlerp(q0, _mat_to_quat(R1), u))
            s0, s1 = a["shape0"], a["shape1"]
            shape = {k: (None if s1[k] is None else (s1[k] if s0.get(k) is None else s0[k] + u * (s1[k] - s0[k]))) for k in s1}
            return grip, R, shape
        if mode == "at":
            g, R = self._realize(p, sd, *self._resolve_hand(a["to"], sd, p))
            if a.get("shake"):
                ph = 2 * math.pi * K.SHOW_SHAKE_HZ * a["shake_t"] * TICK_S
                g = g + np.asarray(a["shake"], float) * K.SHOW_SHAKE_M * math.sin(ph)
            return g, R, a["shape1"]
        if mode == "hold":
            h = self._hold(a["hold"])
            Rl = self.d.xmat[h.body].reshape(3, 3)
            hover = Rl @ h.n_dir * float(h.ctl.get("hover", 0.0))             # the prop's hovering hands: 5 cm off (A9)
            pt = h.point(self.d) + self._point_vel(h) * TICK_S              # the held point at the tick's end, at its velocity now:
            return pt + Rl @ np.asarray(a["goff"], float) + hover, Rl @ np.asarray(a["Rl"], float).reshape(3, 3), a["shape1"]   # no lag
        raise ValueError(mode)

    def _point_vel(self, h):
        """the held point's velocity now (m/s)"""
        jacp = np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.d, jacp, None, h.point(self.d), h.body)
        return jacp @ self.d.qvel

    def _hand_geoms_local(self, sd, shape):
        """her hand's shapes in its own frame for a hand shape: [(geom, pos, quat, half-length or None)] (the scene's hand drawing,
        parent_kin.hand_geoms, for the fingers and thumb; the palm and the collision capsule as built)"""
        key = (sd, float(shape.get("curl", .4)), float(shape.get("thumb", .35)), shape.get("index"))
        if key not in self._hand_local:
            hg = kin.hand_geoms(sd, key[1], key[2], key[3])
            self._hand_local[key] = [(self.scene.hand_ids[sd][n], np.asarray(p_, float), np.asarray(q_, float), float(hl))
                                     for n, (p_, q_, hl) in hg.items()]
        return self._hand_local[key]

    def _hand_clearance_at(self, sd, pos, R, shape, qpos=None, held=None):
        """the least signed distance (m, up to 0.1) from her hand segment posed at (pos, R) in `shape` (its palm, its collision
        capsule, its fingers and thumb) to the G1's collision shapes, the G1 at qpos (the world's now when None), on a scratch copy
        of the state (MuJoCo's mj_geomDistance: the G1's meshes by their convex hulls, which enclose them): her hand's shapes put
        where that pose and shape put them. held: a G1 link she holds; then (to it, to every other link of the G1)"""
        m, sc = self.m, self.scratch
        sc.qpos[:] = self.d.qpos if qpos is None else qpos
        mujoco.mj_kinematics(m, sc)
        R = np.asarray(R, float); pos = np.asarray(pos, float)
        for g, p_, q_, hl in self._hand_geoms_local(sd, shape):            # her fingers and thumb in this shape
            sc.geom_xpos[g] = pos + R @ p_
            sc.geom_xmat[g] = (R @ _quat_to_mat(q_)).ravel()
            m.geom_size[g, 1] = hl
        for g in (self.hand_geom[sd], m.geom(f"parent_hand_{sd}_palm").id):   # her palm and her hand's capsule as built
            sc.geom_xpos[g] = pos + R @ m.geom_pos[g]
            sc.geom_xmat[g] = (R @ _quat_to_mat(m.geom_quat[g])).ravel()
        best = [0.1, 0.1]
        gp = sc.geom_xpos; g1 = self.g1_arr; rb = m.geom_rbound
        other = m.geom_bodyid[g1] != (-1 if held is None else held)
        for g in self.hand_all[sd]:
            dc = np.linalg.norm(gp[g1] - gp[g], axis=1) - rb[g1] - rb[g]
            for j in np.nonzero(dc < max(best))[0]:
                k = int(other[j])
                best[k] = min(best[k], float(mujoco.mj_geomDistance(m, sc, int(g), int(g1[j]), best[k], self._ft)))
        return min(best) if held is None else (best[0], best[1])

    def _realize(self, p, sd, grip, R):
        """a target's concrete (grip, R) for her pose p: a natural rotation found by three passes of her arm's IK (the fingers set
        along the forearm each pass), a held toy's grip following its centre"""
        if not isinstance(R, NatR):
            return np.asarray(grip, float), R
        segs = kin.fk(p)
        cp, cR = segs["chest"]
        sh = cp + cR @ kin.OFFSET[f"upper_arm_{sd}"]
        pole = cR @ np.array([-.5, kin.side_sign(sd) * 1.0, -1.0])
        g = np.asarray(grip, float)
        palm = R.palm
        fa = unit(g - sh)
        if R.bend:
            palm = kin.perp(R.palm, fa)
        fing = kin.perp(fa, palm)
        Rm = P.hand_R_palm(sd, palm, fing)
        for _ in range(3):
            if R.toy_c is not None:
                g = np.asarray(R.toy_c, float) - Rm @ R.off + Rm @ GRIP_LOCAL[sd]
            kin.arm_ik(p, sd, g - Rm @ GRIP_LOCAL[sd], pole, Rm, segs=segs)
            fa = -p.world_override[f"forearm_{sd}"][:, 2]
            if R.bend:
                palm = kin.perp(R.palm, fa)
            fing = kin.perp(fa, palm)
            Rm = P.hand_R_palm(sd, palm, fing)
        if R.toy_c is not None:
            g = np.asarray(R.toy_c, float) - Rm @ R.off + Rm @ GRIP_LOCAL[sd]
        return g, Rm

    def _place(self, p, sd, grip, R, shape, cont=False):
        """her hand by two-bone IK so its grip point is at `grip` with rotation R (parent_poses.reach's pole: the elbow out, back
        and down; where that breaks a human range, the elbow swung about the shoulder-wrist line to the least excess, as
        parent_poses.reach_swivel does). With cont (her drawn pose, tick to tick) the swing nearest her elbow as drawn is kept among
        the equally good, so her elbow never flips; the reach error in m"""
        grip, R = self._realize(p, sd, grip, R)
        fz = floor_z(grip[:2]) + 0.012
        cur = self.carry.get(sd)
        if cur is not None:                                                 # nor a toy in her hand
            dz = float((R @ (np.asarray(cur["off"], float) - GRIP_LOCAL[sd]))[2])
            fz = max(fz, floor_z(grip[:2]) + self.toy_rest.get(cur["toy"], 0.03) - 0.004 - dz)
        if grip[2] < fz:                                                    # her hand never goes into the floor
            grip = np.array([grip[0], grip[1], fz])
        segs = kin.fk(p)
        cp, cR = segs["chest"]
        wrist = grip - R @ GRIP_LOCAL[sd]
        pole = cR @ np.array([-.5, kin.side_sign(sd) * 1.0, -1.0])
        err = kin.arm_ik(p, sd, wrist, pole, R, segs=segs)
        ex0 = P._excess(p.report[f"arm_{sd}"])
        elbow0 = self.written[0][kin.SEGS.index(f"forearm_{sd}")] if cont else None
        far0 = False
        if cont:                                                            # her natural pole's elbow far from where it is drawn (a
            sh0 = cp + cR @ kin.OFFSET[f"upper_arm_{sd}"]                   # target moved round her arm): the swing searched too
            far0 = float(np.linalg.norm(sh0 + p.world_override[f"upper_arm_{sd}"] @ np.array([0, 0, -kin.L_UA]) - elbow0)) > MAX_JUMP_M
        if ex0 > 0 or (cont and self.swivel[sd] != 0) or far0:
            sh = cp + cR @ kin.OFFSET[f"upper_arm_{sd}"]
            axis = unit(wrist - sh)
            best = None
            sws = (0, -30, 30, -60, 60, -90, 90, -120, 120)
            if cont and not far0:                                           # tick to tick her elbow swings at most 30 deg (never flips)
                sws = tuple(sorted({max(-120, min(120, self.swivel[sd] + dd)) for dd in (-30, -15, 0, 15, 30)}, key=abs))
            elbow_s = None
            if cont and far0:
                # C108 (A133): the drawn elbow lies where the LAST tick's wrist and shoulder put it; once the wrist has moved (a relax's
                # step) every swing of this tick's arm can look like a jump against it, and the range excess alone chose the swing: her
                # elbow leapt from +120 to -60 degrees in a tick (0.62 m, the jump guard). A second measure of continuity, like with
                # like: the elbow her LAST swing gives at THIS tick's wrist and shoulder. A swing continuous with EITHER counts as
                # continuous (the drawn elbow's continuity keeps its say, parent 12 and 13's turns; the last swing's saves the relax)
                pl0 = kin.axang(axis, math.radians(float(self.swivel[sd]))) @ pole
                kin.arm_ik(p, sd, wrist, pl0, R, segs=segs)
                elbow_s = sh + p.world_override[f"upper_arm_{sd}"] @ np.array([0, 0, -kin.L_UA])
            for sw in sws:
                pl = kin.axang(axis, math.radians(sw)) @ pole
                e2 = kin.arm_ik(p, sd, wrist, pl, R, segs=segs)
                ex = round(P._excess(p.report[f"arm_{sd}"]))
                jump = 0.0
                if cont:
                    el = sh + p.world_override[f"upper_arm_{sd}"] @ np.array([0, 0, -kin.L_UA])
                    jump = round(float(np.linalg.norm(el - elbow0)), 2)
                    if elbow_s is not None:
                        jump = min(jump, round(float(np.linalg.norm(el - elbow_s)), 2))
                key = (int(jump > MAX_JUMP_M), ex, jump, abs(sw))            # her elbow never flips across in a tick (more than her
                if best is None or key < best[0]:                             # pelvis may move) to ease a range: the W2 fix's babble
                    best = (key, pl, sw)                                      # seed 10 at p_rest 0.6 swung it 0.6 m
            err = kin.arm_ik(p, sd, wrist, best[1], R, segs=segs)
            if cont:
                self.swivel[sd] = best[2]
        elif cont:
            self.swivel[sd] = 0
        p.hand[sd] = dict(curl=shape.get("curl", .4), thumb=shape.get("thumb", .35), index=shape.get("index"))
        return err

    def _grip_now(self, sd, plan=False, actual=False):
        """her hand's grip point and rotation as her plan has it at the last tick's end (actual: as the physics has it now); plan:
        as a point in the room for a new target, her plan's offsets taken off (they are added back to every target in the room,
        _arm), so the hand does not jump"""
        if actual:
            hb = self.bm.hand_body[sd]
            return self.d.site_xpos[self.grip_site[sd]].copy(), self.d.xmat[hb].reshape(3, 3).copy()
        i = kin.SEGS.index(f"hand_{sd}")
        R = _quat_to_mat(self.written[1][i])
        g = self.written[0][i] + R @ GRIP_LOCAL[sd]
        if plan:
            g = g - self.offset[f"arm_{sd}"] - self.offset["core"]
        return g, R

    def _shape_now(self, sd):
        pose = self.scene.pose
        h = pose.hand[sd] if pose is not None else dict(curl=.25, thumb=.2, index=None)
        return dict(curl=float(h.get("curl", .25)), thumb=float(h.get("thumb", .2)), index=h.get("index"))

    # ------------------------------------------------------------------ where a hand goes
    def _her_fwd(self, p=None):
        yaw = self.base["yaw"]
        return np.array([math.cos(yaw), math.sin(yaw), 0.0])

    def _resolve_hand(self, to, sd, p=None):
        """a hand target spec (plain data, resolved every tick so it follows what it names) as (grip point, hand rotation); the
        rotation may be natural (NatR): the palm's facing fixed and the fingers along her forearm, found with her arm (_realize)"""
        k = to["k"]
        d, ch = self.d, self.child
        if k == "fixed":
            return np.asarray(to["grip"], float), np.asarray(to["R"], float).reshape(3, 3)
        if k == "above_toy":
            b = self.toys[to["toy"]]
            c = d.xpos[b].copy() if float(to.get("h", 0.0)) > 0.0 else self._grasp_point(to["toy"])   # C138: a pick takes a container by its rim
            return c + np.array([0, 0, 0.01 + to.get("h", 0.0)]), NatR([0, 0, -1.0], bend=True)
        if k in ("toy_at", "palm", "show", "floor", "open"):
            cur = self.carry.get(sd)
            if k == "palm":
                n = ch.palm_n[to["side"]]
                gap = to.get("gap")
                if gap is None:                                             # the toy's face on the palm, pressed 3 mm (the grasp point
                    gap = (self._toy_half_along(cur["toy"], n) if cur else 0.03) - PALM_GRASP_OUT - 0.003   # lies 1.6 cm out of it)
                c = ch.grasp[to["side"]] + n * gap + np.array([0.0, 0.0, float(to.get("lift", 0.0))])   # lift: her hand kept off it
                R = NatR([0, 0, -1.0], bend=True)                           # held from above, lowered into its hand (its palm faces
                                                                            # its side: there is no room for her hand beyond the toy)
            elif k == "show":
                c, n = self._show_point(sd)
                R = NatR(n)
            elif k == "floor":
                xy = np.asarray(to["xy"], float)
                zf = floor_z(xy)
                c = np.array([xy[0], xy[1], zf + (self._hold_z(cur["toy"]) if cur else 0.05) + 0.012])   # (C138: a container by its rim)
                R = NatR([0, 0, -1.0], bend=True)
            elif k == "open":
                c = np.asarray(self._resolve_point(to["at"]), float) + np.asarray(to.get("off", (0, 0, 0)), float)
                return c, NatR([0, 0, 1.0])                                   # palm up
            else:
                c = np.asarray(to["point"], float)
                R = np.asarray(to["R"], float).reshape(3, 3)
            if cur is None:
                return c, R
            off = np.asarray(cur["off"], float)                               # the toy's centre in her hand's frame
            if isinstance(R, NatR):
                return c, NatR(R.palm, toy_c=c, off=off, bend=R.bend)
            hand = c - R @ off
            return hand + R @ GRIP_LOCAL[sd], R
        if k == "link":
            body = int(to["body"]); Rl = d.xmat[body].reshape(3, 3)
            pt = d.xpos[body] + Rl @ np.asarray(to["local"], float)
            if "palm" in to:
                return pt + Rl @ np.asarray(to["goff"], float), NatR(Rl @ np.asarray(to["palm"], float))
            return pt + Rl @ np.asarray(to["goff"], float), Rl @ np.asarray(to["Rl"], float).reshape(3, 3)
        if k == "thigh":                                                    # her free hand resting on her own thigh, palm down, the
            pose = p if p is not None else self.scene.pose                  # fingers toward her knee (leaning over the child)
            tp, tR = kin.fk(pose)[f"thigh_{sd}"]
            return tp + tR @ THIGH_REST, tR @ P.hand_R_palm(sd, [-1.0, 0, 0], [0, 0, -1.0])
        if k == "toy_centre":                                               # a toy in her other hand: her grip onto its centre, her
            pose = p if p is not None else self.scene.pose                  # palm facing it from her own side
            cR = kin.fk(pose)["chest"][1]
            c = d.xpos[self.toys[to["toy"]]].copy()
            return c, NatR(cR @ np.array([0, -kin.side_sign(sd), 0.0]))
        if k == "up_from":
            g, R = self._grip_now(sd, plan=True)
            return g + np.array([0, 0, 0.08]), R
        if k == "off_surface":
            g, R = self._grip_now(sd, plan=True)
            return np.asarray(self.lift.get(sd) or _lst(g + np.array([0, 0, 0.08])), float), R
        if k == "carry":
            pose = p if p is not None else self.scene.pose
            yaw = self.base["yaw"]
            f3 = np.array([math.cos(yaw), math.sin(yaw), 0.0]); l3 = np.array([-f3[1], f3[0], 0.0])
            root = np.asarray(pose.pos, float)
            g = np.array([root[0], root[1], 0.0]) + f3 * 0.32 + l3 * kin.side_sign(sd) * 0.12 + np.array([0, 0, root[2] + 0.10])
            return g, NatR([0, 0, -1.0], bend=True)
        if k == "rel":
            pose = p if p is not None else self.scene.pose
            segs = kin.fk(pose)
            seg = to.get("seg", "chest")
            sp, sR = segs[seg]
            return sp + sR @ np.asarray(to["grip"], float), sR @ np.asarray(to["R"], float).reshape(3, 3)
        if k == "relaxed":
            q = self._pose(arms=False, look=False)
            segs = kin.fk(q)
            hp, hR = segs[f"hand_{sd}"]
            return hp + hR @ GRIP_LOCAL[sd], hR
        raise ValueError(k)

    def _over_child(self, g0, g1):
        """a hand's way from g0 toward g1 kept over the child: where the straight line passes within 0.25 m (on the floor plan) of
        its shapes, lifted OVER_CHILD_M above their tops before crossing (her hands and forearms never pass through it, A4); the
        waypoints from g0"""
        g0 = np.asarray(g0, float); g1 = np.asarray(g1, float)
        top = -1.0
        for x, y, z, r in self.child.foot_pts:
            if _seg_dist((x, y), g0, g1) < 0.25 + r:
                top = max(top, z + r)
        h = top + K.OVER_CHILD_M
        if top < 0 or min(g0[2], g1[2]) >= h:
            return [g0]
        return [g0, np.array([g0[0], g0[1], max(g0[2], h)]), np.array([g1[0], g1[1], max(g1[2], h)])]

    def _toy_half_along(self, toy, n):
        """a toy's half-extent along a direction, from its centre: the SUPPORT of each of its collision shapes along it (C221, 2026-10-02:
        a sphere's radius, a capsule's and a cylinder's by their axis, an ellipsoid's by its radii, a box's projection, a mesh's by its
        vertices). Until C221 it projected each shape's bounding box on the direction in the shape's own frame, which for a sphere rolled
        to any angle reads up to sqrt(3) times its radius: the hand-over's ball (6 cm) read 8.6 to 9.8 cm and its centre was planned 6.7 cm
        above the child's palm, so every hand-over of day 62's morning (18 of 18) hovered the toy over an open hand and let it go, 0 N on
        the palm (the probe on day 62's copy: her hand at its plan within 2 mm, the plan the fault)"""
        m, d = self.m, self.d
        b = self.toys[toy]
        c = d.xpos[b]
        n = unit(np.asarray(n, float))
        best = 0.0
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or not m.geom_contype[g]:
                continue
            R = d.geom_xmat[g].reshape(3, 3)
            nl = R.T @ n                                                   # the direction in the shape's frame
            size = m.geom_size[g]; kind = m.geom_type[g]
            if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                ext = float(size[0])
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                ext = float(size[0] + size[1] * abs(nl[2]))
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                ext = float(size[0] * math.sqrt(max(0.0, 1.0 - nl[2] ** 2)) + size[1] * abs(nl[2]))
            elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                ext = float(np.linalg.norm(np.asarray(size[:3], float) * nl))
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                ext = float(np.abs(nl) @ size[:3])
            elif kind == mujoco.mjtGeom.mjGEOM_MESH:
                V = _mesh_verts(m, g)
                ext = float((V @ nl).max()) if len(V) else float(np.abs(nl) @ m.geom_aabb[g][3:])
            else:
                ext = float(np.abs(nl) @ m.geom_aabb[g][3:])
            best = max(best, float((d.geom_xpos[g] - c) @ n) + ext)
        return best

    def _approach_dir(self, to, sd):
        k = to["k"]
        if k == "palm":
            return np.array([0, 0, 1.0])
        if k == "link":
            Rl = self.d.xmat[int(to["body"])].reshape(3, 3)
            return unit(Rl @ np.asarray(to["goff"], float))
        return np.array([0, 0, 1.0])

    def _shoulder(self, sd, p=None):
        pose = p if p is not None else self.scene.pose
        segs = kin.fk(pose)
        cp, cR = segs["chest"]
        return cp + cR @ kin.OFFSET[f"upper_arm_{sd}"]

    def _show_point(self, sd):
        """the toy's centre SHOW_DIST before the child's eyes along their axis, moved toward her side of its view so her face stays
        unblocked (4.2), her palm behind it facing the child"""
        ch = self.child
        her_head = self._head_now()
        side = unit(np.cross(ch.axis, np.cross(her_head - ch.eyes, ch.axis)))    # toward her, across the axis
        c = ch.eyes + ch.axis * K.SHOW_DIST_M + side * 0.10
        n = unit(ch.eyes - c)                                                    # her palm faces the child through the toy
        return c, n

    def _head_now(self):
        i = kin.SEGS.index("head")
        return self.written[0][i] + _quat_to_mat(self.written[1][i]) @ np.array([0, 0, 0.15])

    def _hold(self, name):
        for h in self.holds:
            if h.name == name:
                return h
        return None

    # ------------------------------------------------------------------ her gaze (L1)
    def _gaze_tick(self):
        """her gaze's acts in order (a look reaches its target within the tick, holds LOOK_TICKS, or FOCUS_TICKS during a focus
        word, then her eyes go back to the child's eyes: 4.10); while an ask is pending (eyes_on_child) her eyes stay on the
        child's eyes and a look elsewhere waits its turn (A51); L1's glances between"""
        self.glances = [g for g in self.glances if g[2] > self.tick]
        if self.eoc:
            self.look = dict(target="child_eyes")
        if self.cur["gaze"] is None and self.queue["gaze"]:
            ok = self._pending_ok()
            k = 0 if not self.eoc else next((j for j, i in enumerate(self.queue["gaze"])
                                              if (self._act(i)["kind"], self._act(i)["target"]) in ok), None)
            a = self._act(self.queue["gaze"][k]) if k is not None else None
            if a is not None:
                self.queue["gaze"].pop(k)
                a["status"] = "running"; a["start"] = self.tick
                self.cur["gaze"] = a["id"]
                if a["target"] not in (None, "ahead") and self._resolve_point(a["target"]) is None:
                    self._end(a, "refused", f"nothing to look at: {a['target']}")
                else:
                    self.look = dict(target=a["target"])
        if self.cur["gaze"] is not None:
            a = self._act(self.cur["gaze"])
            hold = K.FOCUS_TICKS if a["during"] == "focus" else K.LOOK_TICKS
            if self.tick - a["start"] + 1 >= hold:
                self._end(a, "done", "")
                self._gaze_back(a)
        if self.eoc:
            self.glances = []                                               # L1 turns to nothing while an ask is pending

    def _gaze_back(self, a):
        """a look ended (done or cancelled): her eyes back on the child's eyes (4.10: "her eyes are on the object during the name,
        then back on the child's eyes"; P3's contract: a look is under way as long as her eyes are held on its target)"""
        if self.look.get("target") == a.get("target") and a.get("target") not in (None, "ahead"):
            self.look = dict(target="child_eyes")

    # ------------------------------------------------------------------ her body's acts (L1-L2)
    def _next_body(self):
        """the next act her body starts: the first asked; while an ask is pending (eyes_on_child), the first an ask allows
        (PENDING_OK), the others waiting their turn (A51); none while a formal trial holds her still (P3's Conduct.still)"""
        if self.still:
            return None
        if not self.eoc:
            return 0 if self.queue["body"] else None
        ok = self._pending_ok()
        return next((k for k, i in enumerate(self.queue["body"]) if (self._act(i)["kind"], self._act(i)["target"]) in ok), None)

    def _still_tick(self):
        """while a formal trial holds her still (P3's Conduct.still: its settle, sentence and window) and no act of hers is under
        way: her holds on the child let go and her hands brought to rest (on her thighs when she kneels, hanging when she stands),
        at her own reaching pace; her face is drawn in its neutral set (_face_drawn)"""
        if not self.still or self.cur["body"] is not None or self.phases:
            return
        if self.holds:
            self.phases = self._let_go_phases()
            return
        kneel = self.base["mode"] in ("heels", "tall")
        n = int(math.ceil(K.REACH_MIN_S / TICK_S)) + 1
        for sd in "LR":
            arm = self.arms[sd]
            if arm.get("auto") or arm.get("support") or self.holding[sd] is not None:
                continue
            if kneel and arm["mode"] != "relaxed" or (kneel and arm["mode"] == "relaxed"):
                g0, R0 = self._grip_now(sd, plan=True)
                self.arms[sd] = dict(mode="move", to=dict(k="thigh"), g0=_lst(g0), path=[_lst(g0)], q0=_lst(_mat_to_quat(R0)), t=0,
                                     n=n, shape0=self._shape_now(sd), shape1=dict(curl=.2, thumb=.3, index=None), via=None, auto=True)
            elif arm["mode"] != "relaxed":
                self.phases.append(dict(type="relax", sides=sd))

    def _body_tick(self):
        nxt = self._next_body() if self.cur["body"] is None and not self.phases else None
        if nxt is not None:
            a = self._act(self.queue["body"].pop(nxt))
            a["status"] = "running"; a["start"] = self.tick
            a["info"] = dict(ticks=0, paused=0, paused_run=0, err=0.0, viol=0, hold_peak=0.0, hold_cap=0.0)
            self.cur["body"] = a["id"]
            self.blocked = None
            self._fold_core()                                               # she is where the child pushed her: plans start there
            if self.base["mode"] == "held":
                pose = self.scene.pose
                self.base = dict(mode="stand", at=_lst(pose.pos[:2]), yaw=math.atan2(pose.R[1, 0], pose.R[0, 0]), lean=0.0, spine=0.0,
                                 twist=0.0)
            try:
                self.phases = self._release_phases(a) + self._settle_phases() + self._plan(a)
            except Refuse as e:
                self._end(a, "refused", str(e)); self.phases = []
                return
        a = self._act(self.cur["body"]) if self.cur["body"] is not None else None
        if a is not None:
            a["info"]["ticks"] += 1
        self._idle_relax(a)
        if a is not None and self.blocked == a["id"]:
            self.blocked = None
            self._fail(a, "given up: the child is in her way (she backed off 12 cm and it still pressed on her)")
            return
        if any(self.yielding[c] and self._moving_chain(c) for c in CHAINS):   # a chain of hers moving into the child: the act waits
            if a is not None:                                               # (a still body the child presses on goes on: she is a
                a["info"]["paused"] += 1; a["info"]["paused_run"] += 1       # body, and it gives)
                if a["info"]["paused_run"] > K.PATIENCE_TICKS:
                    self._fail(a, "given up: the child pressed against her for 6 s")
            return
        if a is not None:
            a["info"]["paused_run"] = 0
        if self.lag:                                                        # a yield stopped her drawing short of the plan: this tick
            return                                                          # she reaches it again before her plan goes on
        ph0 = self.phases[0] if self.phases else None
        if ph0 is not None and ph0.get("type") in STEPPING and ph0.get("calm_wait", 0) < K.CALM_WAIT_TICKS and self._restless_near():
            ph0["calm_wait"] = ph0.get("calm_wait", 0) + 1                  # THE CALM STEP (A25b): beside the child she moves her legs
            self.stats["calm_waits"] = self.stats.get("calm_waits", 0) + 1   # (a step, kneeling down, a shuffle, a turn on her knees)
            return                                                          # only while its limbs near her legs are still: she waits
                                                                            # for them, as a person by a kicking baby does, at most
                                                                            # CALM_WAIT_TICKS in a phase, then goes on
        for _ in range(16):
            if not self.phases:
                if a is not None:
                    self._end(a, "done", a["why"])
                return
            ph = self.phases[0]
            r = self._phase(a, ph)
            if r == "run":
                return
            if r in ("next", "done"):
                if self.phases and self.phases[0] is ph:
                    self.phases.pop(0)
                if r == "done":
                    if not self.phases and a is not None:
                        self._end(a, "done", a["why"])
                    return
                continue
            if a is not None:
                self._fail(a, r)
            else:
                self.phases = []
            return

    def _phase(self, a, ph):
        b = self.base
        if ph.get("t", 0) == 0 and ph["type"] in ("shuffle", "kneel_down", "knee_turn", "walk", "turn", "sit") and \
                b["mode"] in ("heels", "tall") and (b.get("lean") or b.get("spine") or b.get("twist")):
            self.phases.insert(self.phases.index(ph), dict(type="lean", lean=0.0, spine=0.0, twist=0.0))
            return "next"                                                   # upright before she moves her base (never a lean dragged)
        if "serial" not in ph:
            self.serial += 1; ph["serial"] = self.serial                  # each phase begun gets its own number (a yield's offset
        self.cur_serial = ph["serial"]                                      # is kept through the phase it was made in)
        try:
            r = getattr(self, "_ph_" + ph["type"])(a, ph)
        except Refuse as e:
            return str(e)
        ph["t"] = ph.get("t", 0) + 1
        if a is not None:
            for sd in "LR":
                e = self.arms[sd].get("err")
                if e is not None:
                    a["info"]["err"] = max(a["info"]["err"], float(e))
        return r

    def _fail(self, a, why):
        self._end(a, "refused", why)
        self._abort_body()

    def _fold_core(self):
        """where the child pushed her made her place (at an act's start, her base at rest): the offset goes into where she kneels or
        stands, and a hand held at a point in the room keeps it as its own offset, so nothing she plans moves"""
        if np.any(self.standoff) and self.base["mode"] in ("heels", "tall", "stand", "turn"):   # her standoff is where she is now
            self.base["at"] = _lst(np.asarray(self.base["at"], float) + self.standoff)
            self.standoff = np.zeros(2)
        oc = self.offset["core"]
        if not np.any(oc) or self.base["mode"] not in ("heels", "tall", "stand", "turn"):
            return
        oc = np.array([oc[0], oc[1], 0.0])
        self.base["at"] = _lst(np.asarray(self.base["at"], float) + oc[:2])
        self.offset["core"] = np.zeros(3)
        for sd in "LR":
            arm = self.arms[sd]
            if arm["mode"] == "hold" or (arm["mode"] in ("at", "move") and self._world_target(arm["to"])):
                self.offset[f"arm_{sd}"] = self.offset[f"arm_{sd}"] + oc
            elif arm["mode"] == "move":
                arm["path"] = [_lst(np.asarray(x, float) + oc) for x in arm["path"]]
            if self.lift.get(sd) is not None:
                self.lift[sd] = _lst(np.asarray(self.lift[sd], float) + oc)

    @staticmethod
    def _world_target(to):
        """a hand target fixed in the room or on the child (a core offset moves it with her body), not one on her own body (which
        her body's pose already carries)"""
        return to["k"] not in ("carry", "rel", "relaxed", "thigh")

    def _abort_body(self):
        """an act stopped midway: a transition under way finishes (she never stops half knelt), her holds on the child let go, her
        free hands relax; a toy in her hand stays in it"""
        keep = []
        if self.phases:
            ph = self.phases[0]
            if ph["type"] in ("kneel_down", "shuffle", "sit", "turn"):
                keep = [ph]
            elif ph["type"] == "walk" and self.base["mode"] == "walk":
                b = self.base
                p0, p1 = np.asarray(b["p0"]), np.asarray(b["p1"])
                D = float(np.linalg.norm(p1 - p0)); sd = min(D, b["s"] + P.STEP_L / 2)
                ph = dict(ph); ph["p1"] = _lst(p0 + (p1 - p0) * (sd / D if D > 0 else 0)); ph["resume"] = True
                keep = [ph]
        self.phases = keep + self._let_go_phases() + [dict(type="relax", sides="LR")]

    def _release_phases(self, a):
        """before a new act: her holds on the child let go (unless the act goes on with them)"""
        if not self.holds:
            return []
        if a["kind"] in ("prop",) and all(h.kind == "prop" for h in self.holds):
            return []
        return self._let_go_phases()

    def _plan_let_go(self, a, names=None):
        return self._let_go_phases(names)

    def _let_go_phases(self, names=None):
        """her holds end: the springs let go, each hand lifted off along the surface it held, then relaxed"""
        hs = [h for h in self.holds if names is None or h.name in names]
        if not hs:
            return []
        out = [dict(type="let_go", names=[h.name for h in hs])]
        sides = sorted({h.side for h in hs})
        out.append(dict(type="reach", hands={sd: dict(k="off_surface", side=sd) for sd in sides},
                        shape={sd: dict(curl=.3, thumb=.3, index=None) for sd in sides}, n=3, solve=False))
        out.append(dict(type="relax", sides="".join(sides)))
        return out

    def _support_tick(self):
        """leaning far over (lean + spine at least K.SUPPORT_BEND_DEG) she rests a free hand on her own thigh (it would hang down
        onto the child otherwise), set there over a few ticks, and lets it hang again when she straightens (an arm no act uses:
        its moves advance here)"""
        b = self.base
        bend = b.get("lean", 0.0) + b.get("spine", 0.0) if b["mode"] in ("heels", "tall") else 0.0
        n = int(math.ceil(K.REACH_MIN_S / TICK_S)) + 1
        for sd in "LR":
            arm = self.arms[sd]
            if arm.get("auto"):
                arm["t"] = arm.get("t", 0) + 1
                if arm["t"] >= arm["n"]:
                    if arm["to"]["k"] == "thigh":
                        self.arms[sd] = dict(mode="at", to=dict(k="thigh"), shape1=dict(curl=.2, thumb=.3, index=None), support=True)
                    else:
                        self.arms[sd] = dict(mode="relaxed")
                continue
            free = self.holding[sd] is None and not self._hold_on(sd)
            if arm["mode"] == "relaxed" and free and bend >= K.SUPPORT_BEND_DEG:
                g0, R0 = self._grip_now(sd, plan=True)
                self.arms[sd] = dict(mode="move", to=dict(k="thigh"), g0=_lst(g0), path=[_lst(g0)], q0=_lst(_mat_to_quat(R0)), t=0, n=n,
                                     shape0=self._shape_now(sd), shape1=dict(curl=.2, thumb=.3, index=None), via=None, auto=True)
            elif arm.get("support") and (bend < K.SUPPORT_OFF_DEG or not free):
                g0, R0 = self._grip_now(sd, plan=True)
                self.arms[sd] = dict(mode="move", to=dict(k="relaxed"), g0=_lst(g0), path=[_lst(g0)], q0=_lst(_mat_to_quat(R0)), t=0, n=n,
                                     shape0=self._shape_now(sd), shape1=dict(curl=.35, thumb=.25, index=None), via=None, auto=True)

    def _idle_relax(self, a):
        """a hand left out by a finished act relaxes after K.IDLE_RELAX_TICKS, unless it holds a toy or the child"""
        if a is not None or self.phases:
            return
        for sd in "LR":
            arm = self.arms[sd]
            if arm["mode"] in ("at", "point") and self.holding[sd] is None and not arm.get("stay") and not arm.get("support"):
                arm["idle"] = arm.get("idle", 0) + 1
                if arm["idle"] >= K.IDLE_RELAX_TICKS:
                    self.phases.append(dict(type="relax", sides=sd))

    # ------------------------------------------------------------------ phases: her base
    def _stable(self, mode, at, yaw, lean=0.0, spine=0.0, twist=0.0):
        self.base = dict(mode=mode, at=_lst(at), yaw=float(yaw), lean=float(lean), spine=float(spine), twist=float(twist), dirty=True)

    def _ph_walk(self, a, ph):
        p0 = np.asarray(ph["p0"], float); p1 = np.asarray(ph["p1"], float)
        D = float(np.linalg.norm(p1 - p0))
        head = math.atan2(p1[1] - p0[1], p1[0] - p0[0]) if D > 1e-9 else self.base["yaw"]
        if ph.get("t", 0) == 0 and not ph.get("resume"):
            if D < 0.03:
                self._stable("stand", p1, self.base["yaw"])
                return "next"
            self.base = dict(mode="walk", p0=_lst(p0), p1=_lst(p1), s=0.0, at=_lst(p0), yaw=head, lean=0.0, spine=0.0, twist=0.0)
        b = self.base
        if not ph.get("resume") and ph.get("child", True) and self._way_blocked(ph, b, p0, p1, D):
            self._stop_walk(a, ph, b, p0, p1, D)                           # the child moved into her way: she finishes the step
            p1 = np.asarray(ph["p1"], float); D = float(np.linalg.norm(p1 - p0))   # under way and plans the trip again
        near = self.child.clearance_xy(b["at"]) < K.SLOW_NEAR_CHILD_M       # beside the child she walks at half her pace (a swinging
        b["s"] = min(D, b["s"] + K.WALK_MPS * TICK_S * (0.5 if near else 1.0))   # foot carries a quarter of the energy into a limb
        b["at"] = _lst(p0 + (p1 - p0) * (b["s"] / D if D > 0 else 1))              # that moves into her way)
        if b["s"] >= D - 1e-9:
            self._stable("stand", p1, head)
            return "done"
        return "run"

    def _way_blocked(self, ph, b, p0, p1, D):
        """whether the child has moved into her way (A6's clearance, kept while she walks: 4.1's plans are solved again when what
        they avoid moves): the next second of this walk (WALK_MPS) passes within CLEAR_CHILD_M and her half-width of the child's
        shapes (less a grid cell, the path's own resolution), nearer than it did when she planned it and nearer than she stands
        now (a way planned past it as it lay, or away from beside it, is not blocked)"""
        s0 = float(b["s"])
        dmin = self._way_clear(ph, p0, p1, s0 + K.GRID_M, min(D, s0 + K.WALK_MPS))
        need = K.CLEAR_CHILD_M + K.BODY_R_M - K.GRID_M
        return dmin < need and dmin < float(ph.get("clear0", 9.0)) - 0.02 and dmin < self.child.clearance_xy(b["at"]) - 0.01

    def _stop_walk(self, a, ph, b, p0, p1, D):
        """the walk stopped where the step under way ends (as parent_poses.walk blends to standing over half a step), or at once
        when that step's end would take her nearer the child; the rest of the trip dropped and planned again (`again`: an approach,
        a walk to a place, a fetch), or, with no act, she stays"""
        s_end = min(D, float(b["s"]) + P.STEP_L / 2)
        end = p0 + (p1 - p0) * (s_end / D)
        if self.child.clearance_xy(end) < min(self.child.clearance_xy(b["at"]), K.CLEAR_CHILD_M + K.BODY_R_M - K.GRID_M):
            s_end = float(b["s"]); end = p0 + (p1 - p0) * (s_end / D)
        ph["p1"] = _lst(end); b["p1"] = _lst(end)
        self.stats["walks_stopped"] = self.stats.get("walks_stopped", 0) + 1
        i = self.phases.index(ph) if ph in self.phases else 0
        rest = self.phases[i + 1:]
        again = ph.get("again")
        grp = ph.get("grp") if again is not None and again.get("grp") else None
        keep = [q for q in rest if q.get("trip") != ph.get("trip") and (grp is None or q.get("grp") != grp)]
        if a is None or again is None:
            self.phases[i + 1:] = [q for q in keep if q.get("trip") != ph.get("trip")]
            return
        a["info"]["walk_replans"] = a["info"].get("walk_replans", 0) + 1
        redo = dict(type="plan", what=again["what"], args=dict(again["args"]))
        self.phases[i + 1:] = [redo] + keep

    def _ph_turn(self, a, ph):
        y0 = self.base["yaw"]; y1 = float(ph["yaw"])
        dy = _ang(y1 - y0)
        step = math.radians(K.TURN_DEG_PER_S) * TICK_S
        if abs(dy) <= 1e-6:
            return "next"
        at = self.base["at"]
        if abs(dy) <= step:
            self._stable("stand", at, y1)
            return "done"
        self.base = dict(mode="turn", at=at, yaw=y0 + math.copysign(step, dy), lean=0.0, spine=0.0, twist=0.0)
        return "run"

    def _ph_kneel_down(self, a, ph):
        """parent_poses.kneel_down from u0 to u1 (0 standing, 2 the tall kneel, 3 on her heels) at the tall kneel's spot, eased in and
        out (_ease) and timed so that at its fastest no segment of hers moves faster than KNEEL_SEG_MPS and her pelvis no faster than
        KNEEL_PELVIS_MPS (KNEEL_TIME), and never quicker than KNEEL_DOWN_S / STAND_UP_S all the way"""
        u0, u1 = float(ph["u0"]), float(ph["u1"])
        if ph.get("t", 0) == 0:                                             # it begins from the pose she is in (never a stale plan)
            at0 = np.asarray(ph["at"], float); y0 = float(ph["yaw"]); fw = np.array([math.cos(y0), math.sin(y0)])
            b = self.base
            want = {0.0: ("stand", at0 - fw * STAND_BACK), 2.0: ("tall", at0), 3.0: ("heels", at0 - fw * HEELS_BACK)}.get(u0)
            if want is not None and (b["mode"] not in (want[0], "turn") or float(np.linalg.norm(np.asarray(b["at"], float) - want[1])) > 0.05
                                     or abs(_ang(b["yaw"] - y0)) > math.radians(5)):
                return f"a kneeling move not begun from her pose ({b['mode']} at {np.round(b['at'], 2).tolist()}: a stale plan)"
        L0, L1 = _kneel_time(u0), _kneel_time(u1)
        n = max(1, int(math.ceil(EASE_PEAK * abs(L1 - L0) / TICK_S)),
                int(round(abs(u1 - u0) / 3 * (K.KNEEL_DOWN_S if u1 > u0 else K.STAND_UP_S) / TICK_S)))
        t = ph.get("t", 0) + 1
        at = np.asarray(ph["at"], float); yaw = float(ph["yaw"])
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        if t >= n:
            if u1 >= 3:
                self._stable("heels", at - fwd * HEELS_BACK, yaw)
            elif u1 >= 2:
                self._stable("tall", at, yaw)
            else:
                self._stable("stand", at - fwd * STAND_BACK, yaw)
            return "done"
        self.base = dict(mode="kneel_down", at=_lst(at), yaw=yaw, u=_kneel_u(L0 + (L1 - L0) * _ease(t / n)), lean=0.0, spine=0.0,
                         twist=0.0)
        return "run"

    def _ph_shuffle(self, a, ph):
        if ph.get("t", 0) == 0:                                             # a shuffle starts where she kneels tall, facing its way
            b = self.base
            if b["mode"] != "tall" or float(np.linalg.norm(np.asarray(b["at"], float) - np.asarray(ph["p0"], float))) > 0.05 or \
                    abs(_ang(b["yaw"] - float(ph["yaw"]))) > math.radians(5):
                return "a shuffle not begun from her tall kneel (a stale plan)"
            ph["p0"] = list(b["at"])
        p0 = np.asarray(ph["p0"], float); p1 = np.asarray(ph["p1"], float); yaw = float(ph["yaw"])
        D = float(np.linalg.norm(p1 - p0))
        n = max(2, int(math.ceil(D / K.SHUFFLE_MPS / TICK_S)))
        t = ph.get("t", 0) + 1
        if D < 0.01:
            self._stable("tall", p1, yaw)
            return "next"
        if t >= n:
            self._stable("tall", p1, yaw)
            return "done"
        self.base = dict(mode="shuffle", p0=_lst(p0), p1=_lst(p1), yaw=yaw, u=t / n, at=_lst(p0 + (p1 - p0) * t / n), lean=0.0,
                         spine=0.0, twist=0.0)
        return "run"

    def _ph_knee_turn(self, a, ph):
        """turning on her knees (tall) where she knelt, a quarter turn in about a second and a half (ours)"""
        b = self.base
        if ph.get("t", 0) == 0 and (b["mode"] != "tall" or abs(_ang(b["yaw"] - float(ph["yaw0"]))) > math.radians(5)):
            return "a turn on her knees not begun from her tall kneel (a stale plan)"
        d_ = _ang(float(ph["yaw1"]) - float(ph["yaw0"]))
        n = max(2, int(math.ceil(abs(d_) / (math.pi / 2) * 1.5 / TICK_S)))
        t = ph.get("t", 0) + 1
        self._stable("tall", ph["at"], float(ph["yaw0"]) + d_ * _smooth(t / n))
        return "done" if t >= n else "run"

    def _ph_lean(self, a, ph):
        b = self.base
        if ph.get("t", 0) == 0:
            ph["from"] = [b.get("lean", 0.0), b.get("spine", 0.0), b.get("twist", 0.0)]
            ph["n"] = int(ph.get("n") or max(2, math.ceil(max(abs(ph["lean"] - b.get("lean", 0.0)), abs(ph["spine"] - b.get("spine", 0.0)),
                                                              abs(ph.get("twist", 0.0) - b.get("twist", 0.0))) / K.TRUNK_DEG_PER_S / TICK_S)))
        t = ph.get("t", 0) + 1; n = ph["n"]
        u = _smooth(t / n)
        f = ph["from"]
        b["lean"] = f[0] + (ph["lean"] - f[0]) * u
        b["spine"] = f[1] + (ph["spine"] - f[1]) * u
        b["twist"] = f[2] + (ph.get("twist", 0.0) - f[2]) * u
        b["dirty"] = True
        return "done" if t >= n else "run"

    def _ph_sit(self, a, ph):
        """sitting down on the sofa (u 0 to 1) or getting up from it (1 to 0)"""
        n = max(2, int(round(1.6 / TICK_S)))
        t = ph.get("t", 0) + 1
        u0, u1 = float(ph["u0"]), float(ph["u1"])
        at = np.asarray(ph["at"], float); yaw = float(ph["yaw"])
        u = u0 + (u1 - u0) * min(1.0, t / n)
        if t >= n and u1 <= 0:
            fwd = np.array([math.cos(yaw), math.sin(yaw)])
            self._stable("stand", at + fwd * 0.42, yaw)
            return "done"
        self.base = dict(mode="sofa", at=_lst(at), yaw=yaw, u=u, lean=0.0, spine=0.0, twist=0.0)
        return "done" if t >= n else "run"

    def _ph_lie(self, a, ph):
        """A124: lying down onto her front from the tall kneel at `at` (u 0 to 1) or getting up onto it again (1 to 0), over LIE_DOWN_S"""
        n = max(2, int(round(LIE_DOWN_S / TICK_S)))
        t = ph.get("t", 0) + 1
        u0, u1 = float(ph["u0"]), float(ph["u1"])
        at = np.asarray(ph["at"], float); yaw = float(ph["yaw"]); cu = float(ph.get("chest_up", LIE_CHEST_UP[0]))
        if t == 1 and u0 <= 0:
            b = self.base
            if b["mode"] != "tall" or float(np.linalg.norm(np.asarray(b["at"], float) - at)) > 0.05 or abs(_ang(b["yaw"] - yaw)) > math.radians(5):
                return f"lying down not begun from her tall kneel there ({b['mode']} at {np.round(b['at'], 2).tolist()}: a stale plan)"
        u = u0 + (u1 - u0) * min(1.0, t / n)
        if t >= n:
            if u1 >= 1:
                self._stable("lying", at, yaw, lean=cu)
            else:
                self._stable("tall", at, yaw)
            return "done"
        self.base = dict(mode="lie", at=_lst(at), yaw=yaw, u=u, lean=cu, spine=0.0, twist=0.0)
        return "run"

    def _ph_wait(self, a, ph):
        return "done" if ph.get("t", 0) + 1 >= int(ph["n"]) else "run"

    def _ph_hold_out(self, a, ph):
        """a hand (or both) held out to the child, or over her face, while nothing else is asked of her body, at most n ticks: the
        act stays under way as long as it directs the child's eyes (P3's contract); the next act asked ends it"""
        return "done" if self.queue["body"] or ph.get("t", 0) + 1 >= int(ph["n"]) else "run"

    def _ph_plan(self, a, ph):
        """a plan made when it is reached (the child and the toys where they are then), put in this phase's place"""
        if a is not None:
            a["info"]["plans"] = a["info"].get("plans", 0) + 1
            if a["info"]["plans"] > MAX_PLANS:
                return f"her plans for it did not settle ({MAX_PLANS} made)"
        new = getattr(self, "_plan_" + ph["what"])(a, **ph.get("args", {}))
        i = self.phases.index(ph)
        self.phases[i:i + 1] = new
        return "next"

    # ------------------------------------------------------------------ phases: her hands
    def _ph_reach(self, a, ph):
        """her hands to their targets (min-jerk, from where they are drawn), the trunk's lean solved for them when she kneels and
        re-solved about once a second while a target moves (warm-started); `via`: along the target's approach direction"""
        t = ph.get("t", 0)
        hands = ph["hands"]
        if t == 0:
            dist = 0.0
            pose = self.scene.pose
            for sd, to in hands.items():
                g0, R0 = self._grip_now(sd, plan=True)
                g1, R1 = self._resolve_hand(to, sd, pose)
                via = g1 + self._approach_dir(to, sd) * K.APPROACH_M if ph.get("via") else None
                path = self._over_child(g0, via if via is not None else g1)
                pts = [np.asarray(x) for x in path] + ([via] if via is not None else []) + [g1]
                dist = max(dist, sum(float(np.linalg.norm(b_ - a_)) for a_, b_ in zip(pts[:-1], pts[1:])))
                self.arms[sd] = dict(mode="move", to=to, g0=_lst(g0), path=[_lst(x) for x in path], q0=_lst(_mat_to_quat(R0)), t=0, n=1,
                                     shape0=self._shape_now(sd), shape1=dict(ph.get("shape", {}).get(sd, dict(curl=.4, thumb=.35, index=None))),
                                     via="auto" if ph.get("via") else None)
            n = max(int(math.ceil(K.REACH_MIN_S / TICK_S)), int(math.ceil(dist / K.REACH_MPS / TICK_S)))
            ph["n"] = int(ph.get("n") or n)
            for sd in hands:
                self.arms[sd]["n"] = ph["n"]
            if ph.get("solve", True) and self.base["mode"] in ("heels", "tall"):
                self._trunk_for(a, ph, hands, first=True)
        else:
            if ph.get("solve", True) and self.base["mode"] in ("heels", "tall") and t % K.REPLAN_TICKS == 0:
                self._trunk_for(a, ph, hands, first=False)
        t += 1
        for sd in hands:
            self.arms[sd]["t"] = min(t, ph["n"])
        met = [sd for sd, to in hands.items() if to.get("k") == "link" and self.arrived_now.get(sd) and self._near_aim(sd, to)]
        if met and len(met) == len(hands):                                  # her hand met the link it reaches onto (the child moved to
            for sd in met:                                                  # meet it): it has arrived, where it is on it
                to = dict(hands[sd]); body = int(to["body"])
                Rl = self.d.xmat[body].reshape(3, 3)
                g, _ = self._grip_now(sd, plan=True)                        # (its grip where it is drawn, her offsets kept apart)
                pt = self.d.xpos[body] + Rl @ np.asarray(to["local"], float)
                to["goff"] = _lst(Rl.T @ (g - pt))
                self.arms[sd] = dict(mode="at", to=to, shape1=self.arms[sd]["shape1"], stay=bool(ph.get("stay")))
            i = self.phases.index(ph) if ph in self.phases else -1          # the approach's last reach onto it is not needed now
            nxt = self.phases[i + 1] if 0 <= i < len(self.phases) - 1 else None
            if nxt is not None and nxt.get("type") == "reach" and set(nxt["hands"]) == set(met) and \
                    all(v.get("k") == "link" for v in nxt["hands"].values()):
                del self.phases[i + 1]
            self.stats["hands_met"] = self.stats.get("hands_met", 0) + 1
            return "done"
        if "lean_to" in ph:
            u = _smooth(min(1.0, (t - ph["lean_t0"]) / max(1, ph["lean_n"])))
            f, g = ph["lean_from"], ph["lean_to"]
            self.base["lean"] = f[0] + (g[0] - f[0]) * u; self.base["spine"] = f[1] + (g[1] - f[1]) * u
            self.base["twist"] = f[2] + (g[2] - f[2]) * u
        if t >= ph["n"]:
            for sd, to in hands.items():
                self.arms[sd] = dict(mode="at", to=to, shape1=self.arms[sd]["shape1"], stay=bool(ph.get("stay")),
                                     shake=ph.get("shake"), shake_t=0)
            return "done"
        return "run"

    def _near_aim(self, sd, to):
        """her hand (drawn, her offsets apart) within a hand's length (HOLD_SLIP_M) of where the hold ahead wants its grip on the
        link (met anywhere farther, it goes on: its touch keeps it outside, sliding over the child)"""
        hold = next((ph for ph in self.phases if ph.get("type") == "hold" and ph.get("side") == sd), None)
        if hold is None:
            return False
        body = int(hold["body"]); Rl = self.d.xmat[body].reshape(3, 3)
        want = self.d.xpos[body] + Rl @ (np.asarray(hold["local"], float) + np.asarray(hold["goff"], float))
        return float(np.linalg.norm(self._grip_now(sd, plan=True)[0] - want)) <= K.HOLD_SLIP_M

    def _trunk_for(self, a, ph, hands, first):
        """the lean and spine that let her hands reach their targets from where she kneels, nearest the last solution"""
        pose = self.scene.pose
        tg = {}
        for sd, to in hands.items():
            g, R = self._resolve_hand(to, sd, pose)
            tg[sd] = (g, R, self.arms[sd]["shape1"])
        for sd in "LR":                                                     # a hand already holding the child must still reach its
            if sd not in tg and self.arms[sd].get("mode") == "hold" and self._hold(self.arms[sd]["hold"]) is not None:   # held point
                g, R, sh = self._hand_now(pose, sd, self.arms[sd])
                tg[sd] = (g, R, sh)
        key = "trunk"
        if not first:
            last = ph.get("solved_at")
            if last is not None and max(float(np.linalg.norm(tg[sd][0] - np.asarray(last[sd]))) for sd in hands) < K.REPLAN_MOVED_M:
                return
        import time as _t
        t0 = _t.perf_counter()
        warm = self.warm.get(key)
        lean, spine, tw, ok = self._solve_trunk(tg, warm)
        if a is not None:                                                   # wall time: an instrument, never her state
            self.solve_ms[a["id"]] = self.solve_ms.get(a["id"], 0.0) + (_t.perf_counter() - t0) * 1e3
            if len(self.solve_ms) > KEEP_ACTS:
                del self.solve_ms[min(self.solve_ms)]
        if not ok:                                                          # no trunk inside human ranges reaches it: she does not
            lean, spine, tw = self.base.get("lean", 0.0), self.base.get("spine", 0.0), self.base.get("twist", 0.0)   # contort; the
            ph["short"] = True                                              # hand stops short, and the act knows (its reach error)
        self.warm[key] = [lean, spine, tw]
        ph["solved_at"] = {sd: _lst(tg[sd][0]) for sd in hands}
        ph["lean_from"] = [self.base.get("lean", 0.0), self.base.get("spine", 0.0), self.base.get("twist", 0.0)]
        ph["lean_to"] = [lean, spine, tw]
        ph["lean_t0"] = ph.get("t", 0)
        ph["lean_n"] = ph["n"] - ph.get("t", 0) if first else K.REPLAN_TICKS
        if not ok and a is not None:
            a["info"]["unreached"] = True

    def _solve_trunk(self, targets, warm=None, max_lean=70, max_spine=45, step=5, cold=False):
        """(lean, spine, twist, ok): her kneeling trunk so that every hand reaches its (grip, R) within 5 mm inside human ranges.
        The search runs over 5 deg steps of lean and spine (and the chest's turn, 0 then +-15 and +-30 deg) from the warm start
        outward, or from upright by total bend, on the chest's pose alone (the arms' reach does not depend on the legs), and keeps
        the first whose full kneeling pose is inside human ranges too"""
        b = self.base
        at = np.asarray(b["at"], float) + self.offset["core"][:2]
        if warm is None and step < 10:                                      # a cold solve: 10 deg steps first, then 5 near the answer
            lean, spine, tw, ok = self._solve_trunk(targets, None, max_lean, max_spine, step=10)
            if not ok:
                return lean, spine, tw, ok
            warm = [lean, spine, tw]
        lift = self._kneel_lift(at, b["yaw"], b["mode"])
        cands = [(l, s_, tw) for tw in ((0, -15, 15, -30, 30) if step <= 5 else (0, -30, 30)) for l in range(0, max_lean + 1, step)
                 for s_ in range(0, max_spine + 1, step)]
        if warm is not None:
            w3 = list(warm) + [0.0] * (3 - len(warm))
            cands.sort(key=lambda c: (abs(c[0] - w3[0]) + abs(c[1] - w3[1]) + abs(c[2] - w3[2]), c[0] + c[1] + abs(c[2]), c[0]))
        else:
            cands.sort(key=lambda c: (abs(c[2]) > 0, c[0] + c[1] + abs(c[2]), c[0]))
        if warm is not None:                                                # a warm re-plan looks near the last answer only
            cands = [c for c in cands if abs(c[0] - w3[0]) <= 15 and abs(c[1] - w3[1]) <= 15 and abs(c[2] - w3[2]) <= 15]
        best = None
        for lean, spine, tw in cands:
            p = trunk_pose(at, b["yaw"], b["mode"], lean, spine, tw, lift)
            errs = [self._place(p, sd, g, R, sh) for sd, (g, R, sh) in targets.items()]
            bad = sum(len(r["violations"]) for r in p.report.values() if isinstance(r, dict) and r.get("violations"))
            score = max(errs) + 0.05 * bad
            if best is None or score < best[0]:
                best = (score, lean, spine, tw)
            if max(errs) < 0.005 and not bad:
                full = P.kneel(at, b["yaw"], b["mode"], lean=lean, spine_flex=spine, twist=tw)
                full.pos = full.pos + np.array([0.0, 0.0, lift])
                if not any(r.get("violations") for k, r in full.report.items() if k.startswith("leg") and isinstance(r, dict)) \
                        and self._clearance(full, segs=("core",)) >= K.CLEAR_M:   # her legs, trunk and head CLEAR_M from the child
                    return float(lean), float(spine), float(tw), True                  # as she leans (A4; the lead's standoff, A25b)
        if warm is not None and not cold:                                   # nothing near the last answer: the whole search
            return self._solve_trunk(targets, None, max_lean, max_spine, step, cold=True)
        return float(best[1]), float(best[2]), float(best[3]), False

    def _ph_relax(self, a, ph):
        """her free hands back to rest (a hand holding a toy keeps it before her), the trunk upright"""
        sides = [sd for sd in ph["sides"] if self.arms[sd]["mode"] not in ("relaxed", "hold")]   # a hand holding the child is let go
                                                                            # first (_let_go_phases), never walked off its spring
        t = ph.get("t", 0)
        if t == 0:
            if not sides and not (self.base["mode"] in ("heels", "tall") and (self.base.get("lean") or self.base.get("spine") or self.base.get("twist"))):
                return "next"
            ph["from"] = [self.base.get("lean", 0.0), self.base.get("spine", 0.0), self.base.get("twist", 0.0)]
            n = int(math.ceil(K.REACH_MIN_S / TICK_S)) + 1                  # at her hand's reaching pace and her trunk's, never faster
            n = max(n, int(math.ceil(max(abs(x) for x in ph["from"]) / K.TRUNK_DEG_PER_S / TICK_S)))
            for sd in sides:
                g0, R0 = self._grip_now(sd, plan=True)
                to = CARRY if self.holding[sd] is not None else {"k": "relaxed"}
                g1, _ = self._resolve_hand(to, sd)
                n = max(n, int(math.ceil(float(np.linalg.norm(g1 - (g0 + self.offset[f"arm_{sd}"] + self.offset["core"])))
                                         / K.REACH_MPS / TICK_S)))
            ph["n"] = n
            for sd in sides:
                g0, R0 = self._grip_now(sd, plan=True)
                to = CARRY if self.holding[sd] is not None else {"k": "relaxed"}
                g1, _ = self._resolve_hand(to, sd)
                self.arms[sd] = dict(mode="move", to=to, g0=_lst(g0), path=[_lst(x) for x in self._over_child(g0, g1)], q0=_lst(_mat_to_quat(R0)),
                                     t=0, n=ph["n"], shape0=self._shape_now(sd),
                                     shape1=dict(curl=.35 if self.holding[sd] is None else .9, thumb=.25 if self.holding[sd] is None else .8,
                                                 index=None), via=None)
            ph["sides_moving"] = sides
        t += 1
        for sd in ph.get("sides_moving", []):
            self.arms[sd]["t"] = t
        if self.base["mode"] in ("heels", "tall"):
            u = _smooth(t / ph["n"])
            self.base["lean"] = ph["from"][0] * (1 - u); self.base["spine"] = ph["from"][1] * (1 - u)
            self.base["twist"] = ph["from"][2] * (1 - u)
            self.base["dirty"] = True
        if t >= ph["n"]:
            for sd in ph.get("sides_moving", []):
                self.arms[sd] = dict(mode="at", to=CARRY, shape1=dict(curl=.9, thumb=.8, index=None), stay=True) if self.holding[sd] \
                    else dict(mode="relaxed")
            return "done"
        return "run"

    def _ph_grasp(self, a, ph):
        """her hand closes on a toy: the toy's weld switched on where it is (no yank), her hand's collision proxy off the toys (4.1)"""
        sd, toy = ph["side"], ph["toy"]
        g, _ = self._grip_now(sd, actual=True)
        c = self._grasp_point(toy)                                          # (C138: a container's rim)
        if float(np.linalg.norm(c - g)) > 0.08:
            if ph.get("waited", 0) < K.ARRIVE_WAIT_TICKS:                   # her hand is a body: it may still be on its way (her aim
                ph["waited"] = ph.get("waited", 0) + 1                      # by sight brings it: _aim_fix)
                return "run"
            if float(g[2] - c[2]) > HAND_HIGH_M:                            # C139: her hand stopped above it: something in its way
                return (f"the {toy} was not under her hand ({100 * float(np.linalg.norm(c - g)):.0f} cm off: her hand stopped "
                        f"{100 * float(g[2] - c[2]):.0f} cm above it): beyond her reach from above")
            return f"the {toy} was not under her hand ({100 * float(np.linalg.norm(c - g)):.0f} cm off)"
        self._proxy(sd, "carry")
        self.scene.weld(f"hold_{sd}_{toy}", True)
        hb = self.bm.hand_body[sd]
        Rh = self.d.xmat[hb].reshape(3, 3)
        self.carry[sd] = dict(toy=toy, off=_lst(Rh.T @ (self.d.xpos[self.toys[toy]] - self.d.xpos[hb])))   # the toy's centre in her hand's frame
        self.holding[sd] = toy
        return "next"

    def _ph_release(self, a, ph):
        sd = ph["side"]
        toy = self.holding[sd]
        if toy is not None and ph.get("at") is not None:                     # A125: a toy set down where she meant it (the lesson's
            g, _ = self._grip_now(sd, actual=True)                          # place before a prone child's face landed 33 cm off and the
            over = ph.get("over")                                           # C136: a drop into the bucket is measured against the bucket
            if over is not None and over.get("toy") in self.toys:           # where it stands NOW (day 35: three hides refused 12 to 30 cm
                ph["at"] = _lst(self.d.xpos[self.toys[over["toy"]]][:2])    # off: the child beside it shoved the bucket after the plan)
            if toy in TP_OPEN_CONTAINERS:                                   # C138: a container held by its rim: its centre is what lands
                g = self.d.xpos[self.toys[toy]].copy()
            miss = float(np.linalg.norm(g[:2] - np.asarray(ph["at"], float)))   # act was done): her hand not there, she reaches again,
            if miss > PUT_TOL_M:                                            # PUT_RETRIES times; then the act is refused with the miss
                if ph.get("retries", 0) < PUT_RETRIES:
                    ph["retries"] = ph.get("retries", 0) + 1
                    i = self.phases.index(ph)
                    to = dict(over) if over is not None else dict(k="floor", xy=list(ph["at"]))   # C136: over the bucket as it stands
                    self.phases.insert(i, dict(type="reach", hands={sd: to}, via=True,
                                               shape={sd: dict(curl=.95, thumb=.85, index=None)}))
                    return "next"
                return f"the {toy} could not be set down where she meant it ({100 * miss:.0f} cm off, beyond her reach from here)"
        if toy is not None:
            self.scene.weld(f"hold_{sd}_{toy}", False)
        self.holding[sd] = None
        self.carry[sd] = None
        return "next"

    def _ph_proxy(self, a, ph):
        self._proxy(ph["side"], ph["on"] if ph["on"] == "carry" else bool(ph["on"]))
        return "next"

    def _proxy(self, sd, on):
        """her hand's collision proxy (4.1): on (it touches the room, the floor, the child and the toys); 'carry' (a toy held by
        its weld: it touches the room, the floor and the child, never the toy it holds: contype 0, its conaffinity kept); off (both
        bits cleared). Her palm, fingers and thumb always touch the child: she is a body"""
        g = self.hand_geom[sd]
        ct, ca = self.proxy_on[g]
        if on == "carry":
            ct = 0
        elif not on:
            ct, ca = 0, 0
        self.m.geom_contype[g] = ct
        self.m.geom_conaffinity[g] = ca

    def _ph_let_go(self, a, ph):
        """her holds on the child end: the hands stay where they were, then relax (a load she carries is eased off first by the act)"""
        names = set(ph["names"])
        for h in [h for h in self.holds if h.name in names]:
            sd = h.side
            g, R = self._grip_now(sd, plan=True)
            n = self.d.xmat[h.body].reshape(3, 3) @ h.n_dir
            self.lift[sd] = _lst(g + unit(n) * K.APPROACH_M)                 # where her hand lifts to: back out along the surface's normal
            self.arms[sd] = dict(mode="at", to=dict(k="fixed", grip=_lst(g), R=_lst(R)), shape1=self._shape_now(sd))
        self.holds = [h for h in self.holds if h.name not in names]
        return "next"

    # ------------------------------------------------------------------ planning: her clearance from the child (A4)
    def _standoff_clear(self, pose, segs):
        """her standoff's measure: her legs, trunk, head and free arms from the child's body and legs, and her legs from its arms and
        hands too (its arm lying between her kneeling legs was squeezed by them: A25b's build); while she kneels where only its trunk
        was clear (A86: beside a babbling child), from its trunk only, as that kneeling was planned"""
        if getattr(self, "trunk_kneel", False):
            return self._clearance(pose, segs=segs, child=self.g1_trunk)
        c1 = self._clearance(pose, segs=segs, child=self.g1_standoff_arr)
        if c1 < K.CLEAR_M:
            return c1
        return min(c1, self._clearance(pose, segs=("core",), child=self.g1_arr, only=("thigh", "shin", "foot")))

    def _standoff_chains(self):
        """her chains her standoff keeps clear of the child: the rest of her always, and an arm that neither reaches onto the child nor
        holds it (a free arm hanging from her leaning trunk was carried onto the child's torso at 967 N, A25b's build)"""
        out = ["core"]
        for sd in "LR":
            a = self.arms[sd]
            to = a.get("to") or {}
            free = a.get("mode") in ("relaxed", "point") or to.get("k") in ("thigh", "carry", "show", "rel", "relaxed")
            if free and self.holding[sd] is None and not any(h.side == sd for h in self.holds):
                out.append(f"arm_{sd}")                                     # (an arm reaching onto it, a toy by it or the floor, or
        return tuple(out)                                                   # holding it or a toy, does its own reaching)

    def _restless_near(self):
        """whether a shape of the child's within CALM_NEAR_M of her legs (her thighs', shins' and feet's collision shapes as they are)
        moved more than CALM_MOVE_M since the last tick (its limbs babbling beside where her legs move: A25b's calm step)"""
        prev = self.child_prev
        if prev is None:
            return False
        now = self.child.foot_pts[:, :3]
        if now.shape != np.asarray(prev).shape:
            return False
        moved = np.linalg.norm(now - np.asarray(prev), axis=1) > K.CALM_MOVE_M
        if not moved.any():
            return False
        legs = self.d.geom_xpos[self.leg_geoms]
        dist = np.linalg.norm(now[moved][:, None, :] - legs[None, :, :], axis=2) - self.child.foot_pts[moved, 3][:, None]
        return bool((dist < K.CALM_NEAR_M).any())

    def _standoff(self, pose):
        """HER TRUNK CLEAR OF THE CHILD'S BODY, RE-PLANNED AS IT MOVES (the lead's decision of 2026-09-25, A25b; A4's 3 cm): each tick
        her planned legs, trunk and head are measured against the child's body where it is now (its trunk's links, MuJoCo's own
        geometry); nearer than CLEAR_M, she straightens up (YIELD_LEAN_DEG_PER_TICK a step) and then moves her base back off it on
        the floor plan (STANDOFF_M_PER_TICK a step, never into furniture), until her plan is clear again, at most STANDOFF_STEPS
        steps a tick; her hands' targets on the child stay where they are. Clear by CLEAR_M + STANDOFF_M_PER_TICK, a standoff she
        took comes back at YIELD_BACK_M_PER_TICK where it stays clear. Her support carries her to that plan (parent_body): the
        child moving into her is resolved by the physics and A4"""
        b = self.base
        if b["mode"] in ("held", "sofa", "lie", "lying"):
            self.clear_now = 0.2
            return pose
        com = self.child.com
        if float(np.linalg.norm(np.asarray(pose.pos[:2]) - com[:2])) > 1.6:          # (far from it: nothing to keep clear of)
            self.clear_now = 0.2
            if np.any(self.standoff):                                       # A108: her standoff comes back at the pace it went out
                n = float(np.linalg.norm(self.standoff)); step = K.STANDOFF_M_PER_TICK   # (a knee shuffle's), never all at once: a
                self.standoff = np.zeros(2) if n <= step else self.standoff * (1.0 - step / n)   # 0.4 m return in a tick met the
                b["dirty"] = True                                           # jump guard, which put her base back but not her standoff
                pose = self._pose()
            return pose
        segs = self._standoff_chains()
        clr = self._standoff_clear(pose, segs)
        moved = False
        for _ in range(K.STANDOFF_STEPS):
            if clr >= K.CLEAR_M:
                break
            if b["mode"] in ("heels", "tall") and (b.get("lean") or b.get("spine")):
                d_ = K.YIELD_LEAN_DEG_PER_TICK
                b["lean"] = max(0.0, float(b.get("lean", 0.0)) - d_); b["spine"] = max(0.0, float(b.get("spine", 0.0)) - d_)
                ph = self.phases[0] if self.phases else None
                if ph is not None and "lean_to" in ph:                      # her trunk's plan for this reach keeps no more than that
                    ph["lean_to"] = [min(ph["lean_to"][0], b["lean"]), min(ph["lean_to"][1], b["spine"]), ph["lean_to"][2]]
                    ph["lean_from"] = [min(ph["lean_from"][0], b["lean"]), min(ph["lean_from"][1], b["spine"]), ph["lean_from"][2]]
            else:
                her = np.asarray(pose.pos[:2], float)
                v = her - com[:2]; nv = float(np.linalg.norm(v))
                out = v / nv if nv > 1e-6 else -np.array([math.cos(b["yaw"]), math.sin(b["yaw"])])
                step = out * K.STANDOFF_M_PER_TICK
                if not self._core_room(np.r_[step + self.standoff, 0.0]):
                    step = self._core_slide(np.r_[step, 0.0])[:2]
                    if not np.any(step):
                        break
                self.standoff = self.standoff + step
            b["dirty"] = True; moved = True
            pose = self._pose()
            clr = self._standoff_clear(pose, segs)
        if moved:
            self.stats["standoff_ticks"] = self.stats.get("standoff_ticks", 0) + 1
        elif np.any(self.standoff) and clr >= K.CLEAR_M + K.STANDOFF_M_PER_TICK:   # it left her room: her standoff comes back,
            old = self.standoff.copy()                                               # only where she stays clear of it
            n = float(np.linalg.norm(old))
            self.standoff = np.zeros(2) if n <= K.YIELD_BACK_M_PER_TICK else old * (1.0 - K.YIELD_BACK_M_PER_TICK / n)
            b["dirty"] = True
            trial = self._pose()
            c2 = self._standoff_clear(trial, segs)
            if c2 >= K.CLEAR_M:
                pose, clr = trial, c2
            else:
                self.standoff = old
                pose = self._pose()
        self.clear_now = float(clr)
        return pose

    def _floor_lift(self, pose):
        """SHE NEVER PLANS A POSE INTO THE FLOOR (the lead's decision of 2026-09-25, A25b: her trunk is carried, so a leg planned into
        the floor would press it with a leg's whole strength against her support, lifting her off it): how far `pose` must rise so
        that none of her pelvis's, legs' or feet's collision shapes (MuJoCo's own, posed by parent_kin's frames) is below the floor
        or the mat's top under it; the deepest one then just meets it, the others a few millimetres above (parent_poses places the
        knee at KNEE_FLOOR by the shin's radius, while her thigh's shape is thicker and her shoe's sits lower than its drawn sole). 0
        when none is below"""
        segs = kin.fk(pose)
        lift = 0.0
        for seg, typ, size, lp, lR in self.low_shapes:
            P0, R0 = segs[seg]
            c = P0 + R0 @ lp
            Rw = R0 @ lR
            if typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
                low = float(c[2]) - size[1] * abs(float(Rw[2, 2])) - size[0]
            elif typ == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                low = float(c[2]) - math.sqrt(float((Rw[2, 0] * size[0]) ** 2 + (Rw[2, 1] * size[1]) ** 2 + (Rw[2, 2] * size[2]) ** 2))
            elif typ == mujoco.mjtGeom.mjGEOM_SPHERE:
                low = float(c[2]) - size[0]
            else:                                                           # a box
                low = float(c[2]) - float(abs(Rw[2, 0]) * size[0] + abs(Rw[2, 1]) * size[1] + abs(Rw[2, 2]) * size[2])
            lift = max(lift, floor_z(c[:2]) - low)
        return lift

    def _kneel_lift(self, at, yaw, mode):
        """her floor lift kneeling at `at` facing `yaw` (the legs of parent_poses.kneel do not move with her lean): cached per spot"""
        key = (mode, round(float(at[0]), 4), round(float(at[1]), 4), round(float(yaw), 4))
        if key not in self._lift_cache:
            if len(self._lift_cache) > 256:
                self._lift_cache.clear()
            self._lift_cache[key] = self._floor_lift(P.kneel(np.asarray(at, float), yaw, mode))
        return self._lift_cache[key]

    def _clear_trunk(self, pose):
        """her kneeling's clearance from the child (A86): all of its shapes, as A4 plans it; on the second pass of a plan that found
        no way clear of all of them (`_trunk_only`), its trunk only (its pelvis, waist and torso): its limbs, which a babbling child
        sweeps through every spot beside it, pass through her body without force since A25c, as a thrashing baby's arm bumps a
        kneeling parent's knee and she stays"""
        return self._clearance(pose, child=self.g1_trunk) if getattr(self, "_trunk_only", False) else self._clearance(pose)

    def _two_pass(self, fn, *args, **kw):
        """A86: a plan tried clear of all of the child first; only if none is, clear of its trunk"""
        self._trunk_only = False
        try:
            out = fn(*args, **kw)
            if out is not None:
                return out
        except Refuse:
            pass
        self._trunk_only = True
        try:
            out = fn(*args, **kw)
            if out is not None:
                self.trunk_kneel = True                                     # her kneeling now clear of its trunk only: her standoff
            return out                                                      # measures that too (A86)
        finally:
            self._trunk_only = False

    def _clearance(self, pose, segs=("core",), skip_off=False, child=None, only=None):
        """the least distance (m, up to 0.2) from her collision shapes of the named chains, posed as `pose`, to the G1's (or to the
        G1 shapes `child` names), by MuJoCo's own geometry on a scratch copy of the state (skip_off: a shape whose collision is
        switched off now is left out)"""
        m, sd = self.m, self.scratch
        sd.qpos[:] = self.d.qpos
        fk = pose if isinstance(pose, dict) else kin.fk(pose)
        self.bm.write_qpos(sd.qpos, self.bm.targets(fk))
        mujoco.mj_kinematics(m, sd)
        chains = set(segs)
        mine = [g for g in np.nonzero(self.geom_seg >= 0)[0] if SEG_CHAIN[kin.SEGS[self.geom_seg[g]]] in chains
                and not (skip_off and not (m.geom_contype[g] or m.geom_conaffinity[g]))
                and (only is None or kin.SEGS[self.geom_seg[g]].startswith(only))]
        best = 0.2
        ft = np.zeros(6)
        gp = sd.geom_xpos
        g1 = self.g1_geoms if child is None else child
        rb = m.geom_rbound
        for g in mine:
            dc = np.linalg.norm(gp[g1] - gp[g], axis=1) - rb[g1] - rb[g]
            for j in np.nonzero(dc < best)[0]:
                r = mujoco.mj_geomDistance(m, sd, int(g), int(g1[j]), best, ft)
                best = min(best, float(r))
        return best

    # ------------------------------------------------------------------ planning: coming to the child (A6)
    def _spots(self, where=None, offs=None, alongs=None):
        """her kneeling spots (her pelvis on her heels, her facing) in the order she tries them: beside its chest on the side it
        faces, then the other side, then at its head and at its feet (A6)"""
        ch = self.child
        out = []
        if ch.posture == "sitting":
            fr = unit(ch.torso_R[:, 0] * [1, 1, 0])[:2]
            lat = np.array([-fr[1], fr[0]])
            for sg in ((1, -1) if where in (None, "side") else (1,)):
                for off in (offs or (0.75, 0.85, 0.70)):
                    for along in (alongs or (-0.10, 0.0, -0.20)):
                        H = ch.pelvis[:2] + lat * sg * off + fr * along
                        out.append((H, math.atan2(-lat[1] * sg, -lat[0] * sg), "L" if sg > 0 else "R"))
            if where in (None, "front"):
                for off in (0.95, 1.05):
                    H = ch.pelvis[:2] + fr * off
                    out.append((H, math.atan2(-fr[1], -fr[0]), "front"))
            return out
        mid = (ch.torso[:2] + ch.pelvis[:2]) / 2
        axis = ch.len_axis[:2]
        first = ch.face_side()
        if ch.posture == "back" and self.base["mode"] in ("heels", "tall"):   # A112 (C96): a child on its back sees her from either
            d = np.asarray(self.base["at"], float) - mid                        # side, so the side she already kneels on comes first (a
            if float(np.linalg.norm(d)) <= K.STAY_SIDE_M:                       # slight roll of its torso flipped face_side and sent her
                first = "L" if float(d @ ch.lat[:2]) > 0 else "R"               # round it: life day 6's 465 kneels and 221 walks)
        sides = [first, "R" if first == "L" else "L"]
        if where in ("L", "R"):
            sides = [where]
        if where in (None, "side", "L", "R"):
            for sd in sides:
                sg = 1 if sd == "L" else -1
                lat = ch.lat[:2] * sg
                for off in (offs or K.KNEEL_OFF_TRY):
                    for along in (alongs or (K.KNEEL_ALONG_M, 0.0, 0.20, -0.10, 0.30)):
                        out.append((mid + axis * along + lat * off, math.atan2(-lat[1], -lat[0]), sd))
        if where in (None, "feet", "pull"):
            feet = ch.pelvis[:2] + axis * 0.70
            for off in (offs if where in ("feet", "pull") and offs else (0.55, 0.65)):
                out.append((feet + axis * off, math.atan2(-axis[1], -axis[0]), "feet"))
        if where == "pull":
            # A163 (2026-10-01): THE PULL-TO-SIT FROM ITS SIDE WHEN ITS FEET ARE OUT OF HER REACH. A9 has her kneel at its feet; on this
            # body (a 1.3 m G1, its forearms at its hips) one trunk of hers reaches both forearms from there only in a lean her knees
            # cannot hold, so every pull-to-sit was refused before she went (parent 7 had recorded it; life day 56 tick 2,702,009: 'no
            # spot she can kneel at lets her do it (pull): her reach'). After the feet's spots, the spots beside its hips on either
            # side (the near forearm at hand, the far one across its chest), the pull's direction unchanged (up and toward its feet,
            # in its frame: a sit-up about its hips); a parent of a child too long to reach from its feet kneels beside its hips
            for sd in sides:
                sg = 1 if sd == "L" else -1
                lat = ch.lat[:2] * sg
                for off in K.KNEEL_OFF_TRY:
                    for along in (0.0, 0.10, -0.10, 0.20):
                        out.append((mid + axis * along + lat * off, math.atan2(-lat[1], -lat[0]), sd))
        if where in (None, "head"):
            head = ch.torso[:2] - axis * 0.45
            for off in (offs if where == "head" and offs else (0.60, 0.70)):
                out.append((head - axis * off, math.atan2(axis[1], axis[0]), "head"))
        return out

    def _toys_xy(self, exclude=()):
        ex = set(exclude) | {t for t in self.holding.values() if t is not None}
        return {k: self.d.xpos[b][:2].copy() for k, b in self.toys.items() if k not in ex}

    def _kneel_plan(self, where=None, offs=None, alongs=None, need=None):
        return self._two_pass(self._kneel_plan_once, where, offs, alongs, need)

    def _kneel_plan_once(self, where=None, offs=None, alongs=None, need=None):
        """the first spot she can kneel at: the final kneel 3 cm clear of the child (A4), from where the act can be done (`need`:
        _need_ok); kneeling down a step back where the child is too near ahead (her step, her knees and her standing spot clear of
        it, of toys and of furniture), the shuffle in clear; the toys where she will kneel cleared first, each free of the child (a
        toy it touches is its own: A4) and within her reach from where she kneels down. None when no spot is free (the reasons are
        kept in `kneel_reasons`, an instrument)."""
        toys = self._toys_xy()
        reasons = []
        reach_cache = {}
        own = {}
        tries = 0
        for H, yaw, tag in self._spots(where, offs, alongs):
            if need is not None and tries >= MAX_NEED_TRIES:
                reasons.append((tag, f"the act cannot be done from there ({need}): not tried, {MAX_NEED_TRIES} spots were"))
                continue
            fwd = np.array([math.cos(yaw), math.sin(yaw)]); left = np.array([-fwd[1], fwd[0]])
            if not self._in_plan(H) or self.plan.dist[self.plan.cell(H)] < K.BODY_R_M + 0.05:
                reasons.append((tag, "furniture")); continue
            if self._clear_trunk(frame_segs("kneel", "heels", H, yaw)) < K.CLEAR_M:
                reasons.append((tag, "the final kneel touches the child")); continue
            need_ok = None                                                  # asked only of a spot that passes every other check (it is
            T = H + fwd * HEELS_BACK                                        # the costly one: a trunk solve or a face search)
            if self._clear_trunk(frame_segs("kneel_down", 2.5, T, yaw)) < K.CLEAR_M:
                reasons.append((tag, "sitting back touches the child")); continue
            for back in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                T2 = T - fwd * back
                stand = T2 - fwd * STAND_BACK
                if not self._in_plan(stand) or self.plan.dist[self.plan.cell(stand)] < K.BODY_R_M + 0.05 or \
                        self.plan.dist[self.plan.cell(T2)] < K.BODY_R_M:
                    reasons.append((tag, back, "furniture at her step back")); continue
                spots = [T2, T2 + fwd * 0.40 + left * 0.12, T2 + fwd * 0.03 + left * 0.115, T2 + fwd * 0.03 - left * 0.115, stand]
                if any(np.linalg.norm(xy - q) < 0.15 for xy in toys.values() for q in spots):
                    reasons.append((tag, back, "a toy where she kneels down")); continue
                if not (all(self._clear_trunk(frame_segs("kneel_down", u, T2, yaw)) >= K.CLEAR_M for u in KNEEL_CHECK_U) and
                        all(self._clear_trunk(frame_segs("kneel", "tall", T2 + (T - T2) * u, yaw)) >= K.CLEAR_M for u in (0.5, 1.0) if back > 0)):
                    reasons.append((tag, back, "kneeling down or shuffling in touches the child")); continue
                clear, why = [], None
                for k, xy in toys.items():
                    knees = [T + fwd * 0.03 + left * sg * 0.115 for sg in (1, -1)] + [T2 + fwd * 0.03 + left * sg * 0.115 for sg in (1, -1)]
                    shins = [H - fwd * 0.05 + left * sg * 0.115 for sg in (1, -1)]
                    if min(float(np.linalg.norm(xy - q)) for q in knees + shins) >= 0.20 and _seg_dist(xy, T2, T) >= 0.20:
                        continue
                    if k not in own:
                        own[k] = self._child_has(k)
                    if own[k]:
                        why = f"the {k} there is the child's"; break
                    sd = "L" if float((xy - T2) @ left) > 0 else "R"
                    if self.holding[sd] is not None:                        # the hand that will clear it (_plan_clear_here)
                        sd = "R" if sd == "L" else "L"
                    key = (k, sd, round(float(T2[0]), 2), round(float(T2[1]), 2), round(yaw, 2))
                    if key not in reach_cache:
                        reach_cache[key] = self._reachable_at(sd, np.r_[xy, floor_z(xy) + 0.05], T2, yaw, "tall")
                    if not reach_cache[key]:
                        why = f"the {k} is beyond her reach from where she kneels down"; break
                    clear.append(k)
                if why is not None:
                    reasons.append((tag, back, why)); continue
                if need_ok is None:
                    need_ok = self._need_ok(need, H, yaw); tries += need is not None
                if not need_ok:
                    reasons.append((tag, f"the act cannot be done from there ({need})")); break
                if not self._can_walk_to(stand):                            # C131: a spot she cannot get to is no spot (life day 32)
                    reasons.append((tag, back, "no path on the floor to her standing spot")); continue
                self.kneel_reasons = reasons
                return dict(H=_lst(H), T=_lst(T), T2=_lst(T2), stand=_lst(stand), yaw=float(yaw), where=tag, clear=clear)
            if need_ok is False:
                continue
            # no room to kneel down facing it: kneel down at the spot facing along the child, then turn on her knees to face it
            for side_turn in (1, -1):
                y2 = yaw + side_turn * math.pi / 2
                f2 = np.array([math.cos(y2), math.sin(y2)])
                stand = T - f2 * STAND_BACK
                if not self._in_plan(stand) or self.plan.dist[self.plan.cell(stand)] < K.BODY_R_M + 0.05:
                    reasons.append((tag, "turn", "furniture where she would stand")); continue
                l2 = np.array([-f2[1], f2[0]])
                spots = [T, T + f2 * 0.40 + l2 * 0.12, T + f2 * 0.03 + l2 * 0.115, T + f2 * 0.03 - l2 * 0.115, stand]
                if any(np.linalg.norm(xy - q) < 0.15 for xy in toys.values() for q in spots):
                    reasons.append((tag, "turn", "a toy where she kneels down")); continue
                if not (all(self._clear_trunk(frame_segs("kneel_down", u, T, y2)) >= K.CLEAR_M for u in KNEEL_CHECK_U) and
                        all(self._clear_trunk(frame_segs("kneel", "tall", T, yaw + side_turn * a_)) >= K.CLEAR_M for a_ in (math.pi / 4,))):
                    reasons.append((tag, "turn", "kneeling down along it or turning touches the child")); continue
                if need_ok is None:
                    need_ok = self._need_ok(need, H, yaw); tries += need is not None
                if not need_ok:
                    reasons.append((tag, f"the act cannot be done from there ({need})")); break
                if not self._can_walk_to(stand):                            # C131
                    reasons.append((tag, "turn", "no path on the floor to her standing spot")); continue
                self.kneel_reasons = reasons
                return dict(H=_lst(H), T=_lst(T), T2=_lst(T), stand=_lst(stand), yaw=float(yaw), yaw_down=float(y2), where=tag, clear=[])
        self.kneel_reasons = reasons
        return None

    def _can_walk_to(self, stand):
        """C131 (life day 32): whether a path on the floor leads from where she is to a spot's standing point (A6's clearances, the
        toys in the way cleared as a walk would). The spot chooser passed a spot beside the child and the walk to it was refused
        "no path on the floor" 27 times in a day (the child lay by the doorway, the spot on its far side, the way round it closed by
        its own clearance and the wall's): a spot she cannot get to is no spot, and the next side is tried"""
        start = np.asarray(self.base["at"], float)
        if float(np.linalg.norm(np.asarray(stand, float) - start)) < 0.05:
            return True
        try:
            self._walk_phases0(start, np.asarray(stand, float), 0.0, goal_r=0.35, child=True)
            return True
        except Refuse:
            return False

    def _need_ok(self, need, H, yaw):
        """whether the act can be done from a kneeling spot (her pelvis on her heels at H, facing yaw): 'touch:<part>' her near hand
        reaches its surface there with some trunk inside human ranges; 'lean' a pose puts her face where its eyes reach within
        LEAN_DIST_M (face_reach); None anything"""
        if need is None:
            return True
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        if need == "lean":
            return self.face_reach(at=H, yaw=yaw, base_mode="heels") is not None
        if need == "lean_line":                                             # A96: a pose there puts her mouth on its fovea's line
            return self.face_reach(at=H, yaw=yaw, base_mode="heels", on_line=True) is not None
        if need == "lie":                                                   # A124 (C107): lying on her front from the tall kneel there,
            return self._lie_fit(np.asarray(H, float) + fwd * HEELS_BACK, yaw) is not None   # her face lands before a prone child's eyes
        if need.startswith("reach:"):                                      # A133: her hand reaches a floor point from there (a put's place),
            xy = np.array([float(v) for v in need.split(":", 1)[1].split(",")])   # from her heels, as a put reaches (an approach ends there)
            return any(self._reachable_at(sd, np.r_[xy, floor_z(xy) + 0.05], np.asarray(H, float), yaw, "heels") for sd in "LR")
        if need in ("turn_both", "turn_shoulder"):
            # A136 (C98, C109): the turn's far grips in her reach from there: both (A101's turn, beside its chest) or the far shoulder
            # alone (the roll by the shoulder from its head): each grip's point and palm from _turn_targets with her base set there
            T2 = np.asarray(H, float) + fwd * HEELS_BACK
            saved = self.base
            self.base = dict(saved, at=_lst(H), yaw=float(yaw), mode="heels", lean=0.0, spine=0.0, twist=0.0)
            try:
                tg = self._turn_targets(("shoulder", "hip") if need == "turn_both" else ("shoulder", "torso"))
                targets = {}
                for sd in ("R", "L"):
                    to, loc, nl, shape, b, toward = tg[sd]
                    g, R = self._resolve_hand(to, sd)
                    if float(np.linalg.norm(g[:2] - np.asarray(T2, float)[:2])) > 1.0:
                        return False                                        # beyond any lean of hers (_reachable_at's bound)
                    targets[sd] = (g, R, shape)
                # C144 (2026-09-30): ONE trunk for both grips, as the turn's reach solves it (_plan_turn: both hands from one trunk
                # pose, _trunk_for). Each grip was checked by a trunk of its own (_reachable_at), so a spot passed where the one lean
                # that reaches the shoulder leaves the other hand 29 to 32 cm short of the torso's far side, and the turn was refused
                # at its hold instead of tried from the next spot (life day 38 tick 1,840,101: four refusals in 60 ticks over a prone
                # child at (0.64, 0.30), the rig at that spot: 29 and 32 cm short from a spot the check had passed)
                self.base = dict(mode="tall", at=_lst(T2), yaw=float(yaw), lean=0.0, spine=0.0, twist=0.0)
                return bool(self._solve_trunk(targets, None, step=10)[3])
            finally:
                self.base = saved
        if need == "pull":                                                  # one trunk reaches both its forearms from her tall kneel, or
            saved = self.base                                               # each forearm alone (A169: the gather's reach)
            self.base = dict(mode="tall", at=_lst(np.asarray(H, float) + fwd * HEELS_BACK), yaw=float(yaw), lean=0.0, spine=0.0,
                             twist=0.0)
            try:
                return self._pull_pairing() is not None or self._gather_reach()
            finally:
                self.base = saved
        if need.startswith("hand:"):                                       # C133: the child's palm (side cs) within her hand's reach from
            _h, cs, toy = need.split(":", 2)                                # there, as the hand-over reaches it (_plan_hand_over)
            pt = self.child.grasp[cs] + self.child.palm_n[cs] * 0.04
            hands = [sd for sd in "LR" if self.holding[sd] is None or self.holding[sd] == toy or toy == "None"]   # a hand that can
            return any(self._reachable_at(sd, pt, np.asarray(H, float), yaw, "heels", palm=-self.child.palm_n[cs], bend=False)
                       for sd in hands)                                    # give it: holding it, or free to take it (_giving's swap)
        if need.startswith("touch:") and "|" in need:
            return any(self._need_ok(f"touch:{p_}", H, yaw) for p_ in need.split(":", 1)[1].split("|"))
        if need.startswith("touch:"):
            part = need.split(":", 1)[1]
            her = np.r_[H, 0.5]
            body = min((self.m.body(n).id for n in BODY_PARTS.get(part, ("torso_link",))), key=lambda b: float(np.linalg.norm(self.d.xpos[b] - her)))
            Rl = self.d.xmat[body].reshape(3, 3)
            if part in ("tummy", "child", "chest"):
                loc, nor = self._trunk_face(body, her, z=0.07 if part != "chest" else 0.20)
                n = unit(Rl @ np.asarray(nor, float))
                pt = self._surface(body, self.d.xpos[body] + Rl @ np.asarray(loc, float), n)
            else:
                pt, n = self._anchor(body, prefer=her)
            left = np.array([-fwd[1], fwd[0]])
            sd = "L" if float((pt[:2] - H) @ left) > 0 else "R"
            return self._reachable_at(sd, pt + n * 0.01, H + fwd * HEELS_BACK, yaw, "tall", palm=-n, bend=False)
        return True

    def _beside_now(self):
        """she kneels on her heels within 1.1 m of the child's trunk, facing it (within 45 deg), 3 cm clear of it (A4)"""
        b = self.base
        at = np.asarray(b["at"], float)
        to = (self.child.torso[:2] + self.child.pelvis[:2]) / 2 - at
        if float(np.linalg.norm(to)) > 1.1:
            return False
        if abs(_ang(math.atan2(to[1], to[0]) - b["yaw"])) > math.radians(45):
            return False
        return self._clearance(frame_segs("kneel", "heels", at, b["yaw"])) >= K.CLEAR_M

    def _in_plan(self, xy):
        c = self.plan.cell(xy)
        return 0 <= c[0] < self.plan.nx and 0 <= c[1] < self.plan.ny and self.plan.dist[c] >= 0

    def _plan_approach(self, a, where=None, offs=None, alongs=None, need=None):
        """her way to the child wherever it is (A6, B7): up if she kneels elsewhere, a walk around furniture, the child and toys,
        kneeling down a step back, the toys in her way cleared, a shuffle in on her knees, down onto her heels; to a spot from where
        the act can be done (need)"""
        b = self.base
        if where is None and offs is None and b["mode"] == "heels" and self._beside_now() and \
                self._need_ok(need, np.asarray(b["at"], float), float(b["yaw"])):
            return []                                                       # she already kneels beside it, clear of it: she stays
        self.trunk_kneel = False
        kp = self._kneel_plan(where, offs, alongs, need)
        if kp is None and need == "lean_line":                              # A96: no spot puts her face on its line (it looks at the
            a["info"]["line_spot"] = False                                  # floor, a toy): its periphery, as before A94
            need = "lean"
            kp = self._kneel_plan(where, offs, alongs, need)
        if kp is None:
            if need == "lean" and any("cannot be done" in str(r[-1]) for r in self.kneel_reasons):
                raise Refuse("no pose inside human ranges puts her face where its eyes can reach, from any spot she can kneel at "
                             "(A22, C34: she never expects a look there)")
            if need and any("cannot be done" in str(r[-1]) for r in self.kneel_reasons):
                raise Refuse(f"no spot she can kneel at lets her do it ({need}): her reach (A6)")
            raise Refuse("no spot to kneel beside the child: every side is blocked (A6: she never moves the child to make room)")
        a["info"]["spot"] = kp
        H, T, T2, stand, yaw = np.array(kp["H"]), np.array(kp["T"]), np.array(kp["T2"]), np.array(kp["stand"]), kp["yaw"]
        b = self.base
        out = []
        if b["mode"] in ("heels", "tall"):
            fwd0 = np.array([math.cos(b["yaw"]), math.sin(b["yaw"])])
            curT = np.asarray(b["at"], float) + (fwd0 * HEELS_BACK if b["mode"] == "heels" else 0)
            if float(np.linalg.norm(curT - T)) < 0.02 and abs(_ang(b["yaw"] - yaw)) < math.radians(3) and b["mode"] == "heels":
                return []
            if float(np.linalg.norm(curT - T)) < KNEE_ROUTE_M and abs(_ang(b["yaw"] - yaw)) < math.radians(KNEE_ROUTE_DEG) and \
                    not kp["clear"] and \
                    (b["mode"] == "tall" or self._clearance(frame_segs("kneel_down", 2.5, curT, b["yaw"])) >= K.CLEAR_M) and \
                    all(self._clearance(frame_segs("kneel", "tall", curT, b["yaw"] + _ang(yaw - b["yaw"]) * v)) >= K.CLEAR_M
                        for v in (0.25, 0.5, 0.75, 1.0)):
                if b["mode"] == "heels":
                    out.append(dict(type="kneel_down", at=_lst(curT), yaw=b["yaw"], u0=3.0, u1=2.0))
                if abs(_ang(b["yaw"] - yaw)) > math.radians(3):                # turned on her knees to the new spot's facing first
                    out.append(dict(type="knee_turn", at=_lst(curT), yaw0=float(b["yaw"]), yaw1=float(yaw)))
                out.append(dict(type="plan", what="shuffle_in", args=dict(T2=_lst(curT), T=_lst(T), yaw=yaw)))
                self.serial += 1
                for ph in out:
                    ph["grp"] = f"approach{self.serial}"
                return out
            return self._up_phases() + [dict(type="plan", what="approach", args=dict(where=where, offs=offs, alongs=alongs, need=need))]
        elif b["mode"] == "lying" and need == "lie" and \
                self._lie_fit(np.asarray(b["at"], float), float(b["yaw"])) is not None:
            return []                                                       # A124: she lies before its face already: she stays
        elif b["mode"] in ("sofa", "lying"):
            return self._up_phases() + [dict(type="plan", what="approach", args=dict(where=where, offs=offs, alongs=alongs, need=need))]
        else:
            start = np.asarray(b["at"], float)
        again = dict(where=where, offs=offs, alongs=alongs, need=need)     # the approach planned again if the child has moved into it
        redo = dict(what="approach", args=again, grp=True)
        if "yaw_down" in kp:                                                # kneeling down along the child, then turning to face it
            yd = kp["yaw_down"]
            out += self._walk_phases(start, stand, yd, goal_r=0.35, again=redo)
            out.append(dict(type="plan", what="kneel_here", args=dict(T2=_lst(T), yaw=yd, again=again)))
            out.append(dict(type="knee_turn", at=_lst(T), yaw0=yd, yaw1=yaw))
            out.append(dict(type="kneel_down", at=_lst(T), yaw=yaw, u0=2.0, u1=3.0))
            self.serial += 1
            for ph in out:
                ph["grp"] = f"approach{self.serial}"
            return out
        out += self._walk_phases(start, stand, yaw, goal_r=0.35, again=redo)
        out.append(dict(type="plan", what="kneel_here", args=dict(T2=_lst(T2), yaw=yaw, again=again)))
        if kp["clear"]:
            out.append(dict(type="plan", what="clear_here", args=dict(toys=list(kp["clear"]), T=_lst(T), T2=_lst(T2), yaw=yaw)))
        out.append(dict(type="plan", what="shuffle_in", args=dict(T2=_lst(T2), T=_lst(T), yaw=yaw)))
        self.serial += 1
        for ph in out:
            ph["grp"] = f"approach{self.serial}"
        return out

    def _plan_kneel_here(self, a, T2, yaw, again):
        """kneeling down where she stands, checked again against the child as it lies now (its arms sink and spread under the
        resting law while she walks there: a hand 7 cm nearer in 3 s, W2): every frame 3 cm clear, or the approach planned again"""
        T2 = np.asarray(T2, float)
        if all(self._clearance(frame_segs("kneel_down", u, T2, yaw)) >= K.CLEAR_M for u in KNEEL_CHECK_U):
            return [dict(type="kneel_down", at=_lst(T2), yaw=float(yaw), u0=0.0, u1=2.0)]
        if all(self._clearance(frame_segs("kneel_down", u, T2, yaw), child=self.g1_trunk) >= K.CLEAR_M for u in KNEEL_CHECK_U):
            self.trunk_kneel = True                                         # clear of its trunk only (A86)
            return [dict(type="kneel_down", at=_lst(T2), yaw=float(yaw), u0=0.0, u1=2.0)]
        a["info"]["replans"] = a["info"].get("replans", 0) + 1
        if a["info"]["replans"] > 3:
            raise Refuse("the child keeps moving where she would kneel (A4: 3 cm clear)")
        me = self.phases[0].get("grp") if self.phases else None
        if me is not None:                                                  # the rest of this approach goes with it
            self.phases = [self.phases[0]] + [q for q in self.phases[1:] if q.get("grp") != me]
        return [dict(type="plan", what="approach", args=dict(again))]

    def _plan_shuffle_in(self, a, T2, T, yaw):
        return self._two_pass(self._plan_shuffle_in_once, a, T2, T, yaw)

    def _plan_shuffle_in_once(self, a, T2, T, yaw):
        """the shuffle in from where she knelt down to her spot, checked again against the child as it lies now (its limbs move and
        sink): she stops short where her knees would come within 3 cm of it (A4)"""
        T2 = np.asarray(T2, float); T = np.asarray(T, float)
        for u in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
            Tu = T2 + (T - T2) * u
            if all(self._clear_trunk(frame_segs("kneel", "tall", T2 + (Tu - T2) * v, yaw)) >= K.CLEAR_M for v in (0.5, 1.0)) and \
                    self._clear_trunk(frame_segs("kneel_down", 2.5, Tu, yaw)) >= K.CLEAR_M and \
                    self._clear_trunk(frame_segs("kneel_down", 3.0, Tu, yaw)) >= K.CLEAR_M:
                a["info"]["shuffle_short_m"] = round(float(np.linalg.norm(T - Tu)), 3)
                return [dict(type="shuffle", p0=_lst(T2), p1=_lst(Tu), yaw=yaw), dict(type="kneel_down", at=_lst(Tu), yaw=yaw, u0=2.0, u1=3.0)]
        raise Refuse("the child now lies where she would kneel (A4: 3 cm clear)")

    def _walk_phases(self, start, goal, yaw_end, goal_r=0.3, child=True, goal_clear=None, toys=True, depth=0, skip=(), again=None):
        """turn, walk, turn along an A* path (A6); where toys close the only way, the first of them in her way is picked up and set
        aside first (A6: "a toy lying across the only path is picked up and set aside"), and the walk planned again. Each walk
        carries its trip's goal and `again`, the plan that makes the trip anew (with the phases of the trip it replaces: `grp`)
        when the child moves into her way (_way_blocked)"""
        out = self._walk_phases0(start, goal, yaw_end, goal_r, child, goal_clear, toys, depth, skip)
        self.serial += 1
        for ph in out:
            ph["trip"] = self.serial
            if ph["type"] == "walk":
                ph["goal"] = _lst(np.asarray(goal, float)[:2]); ph["goal_r"] = float(goal_r); ph["child"] = bool(child)
                p0, p1 = np.asarray(ph["p0"], float), np.asarray(ph["p1"], float)
                ph["clear0"] = self._way_clear(ph, p0, p1, 0.0, float(np.linalg.norm(p1 - p0)))   # as planned
                if again is not None:
                    ph["again"] = again
        return out

    def _way_clear(self, ph, p0, p1, s0, s1):
        """the least floor-plan distance from the child's shapes to a walk's points from s0 to s1 along it (every GRID_M), those
        within a grid cell of her trip's goal region left out (where only furniture bounds her last steps, as her path allowed);
        9 when none is left"""
        D = float(np.linalg.norm(p1 - p0))
        if D < 1e-9 or s1 <= s0:
            return 9.0
        hd = (p1 - p0) / D
        pts = p0[None, :] + hd[None, :] * np.arange(s0, s1 + 1e-9, K.GRID_M)[:, None]
        goal = ph.get("goal")
        if goal is not None:
            pts = pts[np.linalg.norm(pts - np.asarray(goal, float)[None, :], axis=1) > float(ph.get("goal_r", 0.0)) + K.GRID_M]
        if not len(pts):
            return 9.0
        fp = self.child.foot_pts
        return float((np.linalg.norm(fp[None, :, :2] - pts[:, None, :], axis=2) - fp[None, :, 3]).min())

    def _walk_phases0(self, start, goal, yaw_end, goal_r=0.3, child=True, goal_clear=None, toys=True, depth=0, skip=()):
        start = np.asarray(start, float); goal = np.asarray(goal, float)
        txy = self._toys_xy(exclude=skip) if toys else {}
        path = None
        for child_m in (K.CLEAR_CHILD_M, K.CLEAR_CHILD_TIGHT_M):           # A6's clearance first; where the child's limbs close every
            ok = self.plan.free(K.BODY_R_M + K.CLEAR_FURNITURE_M, self.child if child else None, list(txy.values()), goal=goal,
                                goal_r=goal_r, goal_clear=goal_clear, child_m=child_m)   # way, she threads past them nearer
            first = None
            if not ok[self.plan.cell(start)]:                              # she stands where the clearances do not hold (by the child,
                near = self.plan.free(K.BODY_R_M + 0.02, None, [], None)    # after kneeling): the path starts from where she is
                dxy = np.hypot(*np.meshgrid(self.plan.xs - start[0], self.plan.ys - start[1], indexing="ij"))
                ok = ok | (near & (dxy <= 0.6))
                if not ok[self.plan.cell(start)]:                          # wedged (A95: standing up from the sofa's corner, 12 cm from
                    cand = np.argwhere(ok & (dxy <= 0.6))                  # its arm): one step to the nearest free cell first, then
                    if len(cand):                                          # the path from there
                        k_ = cand[np.argmin(dxy[cand[:, 0], cand[:, 1]])]
                        first = np.array([float(self.plan.xs[k_[0]]), float(self.plan.ys[k_[1]])])
            path = self.plan.path(start if first is None else first, goal, ok)
            if path is not None and first is not None:
                path = [np.asarray(start, float)] + [np.asarray(p_, float) for p_ in path]
            if path is not None or not child or not toys:
                break
        if path is None and toys and depth < 3:
            bare = self._walk_phases0(start, goal, yaw_end, goal_r, child, goal_clear, toys=False, depth=depth + 1, skip=skip)
            way = [np.asarray(ph["p0"], float) for ph in bare if ph["type"] == "walk"] + [goal]
            blocking = []
            for k, xy in txy.items():
                dmin = min(_seg_dist(xy, a_, b_) for a_, b_ in zip(way[:-1], way[1:])) if len(way) > 1 else 9.0
                if dmin < K.CLEAR_TOY_M + 0.15:
                    along = min(range(len(way) - 1), key=lambda i: _seg_dist(xy, way[i], way[i + 1])) if len(way) > 1 else 0
                    blocking.append((along, float(np.linalg.norm(xy - way[along])), k))
            if not blocking:
                raise Refuse("no path on the floor to there (A6's clearances)")
            blocking.sort()
            toy = blocking[0][2]
            if self._child_has(toy):
                raise Refuse(f"the only way there is past the {toy}, and the child has it (A4)")
            return [dict(type="plan", what="clear_way", args=dict(toy=toy)),
                    dict(type="plan", what="walk_to", args=dict(xy=_lst(goal), yaw=float(yaw_end), child=child, goal_r=goal_r,
                                                                goal_clear=goal_clear))]
        if path is None:
            raise Refuse("no path on the floor to there (A6's clearances)")
        out = []
        head = float(self.base["yaw"])
        step = math.radians(K.TURN_DEG_PER_S) * TICK_S
        for p0, p1 in zip(path[:-1], path[1:]):
            if np.linalg.norm(p1 - p0) < 0.03:
                continue
            h1 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            if abs(_ang(h1 - head)) > step:                                 # a bend of less than a tick's turn she walks through
                out.append(dict(type="turn", yaw=h1))
            out.append(dict(type="walk", p0=_lst(p0), p1=_lst(p1)))
            head = h1
        out.append(dict(type="turn", yaw=float(yaw_end)))
        return out

    def _plan_clear_way(self, a, toy):
        """a toy across her only way: fetched (she kneels by it), set aside beside her, away from the way she goes, and she stands"""
        a["info"].setdefault("cleared_way", []).append(toy)
        return [dict(type="plan", what="fetch", args=dict(toy=toy)), dict(type="plan", what="set_aside", args=dict(toy=toy))] + \
            [dict(type="plan", what="stand_up", args={})]

    def _plan_stand_up(self, a):
        return self._two_pass(self._plan_stand_up_once, a)

    def _plan_stand_up_once(self, a):
        """her way up from where she kneels (or sits), planned against the child as it lies now (A4: her legs, trunk and head 3 cm
        clear of it in every planned frame): up onto her knees, then, where getting up there would touch it, a shuffle back on her
        knees, or a turn on her knees and a shuffle away, first; then up onto her feet. Refused when every way touches it (she
        stays kneeling; she never pushes it to get up)"""
        b = self.base
        if b["mode"] == "sofa":
            return [dict(type="sit", at=b["at"], yaw=b["yaw"], u0=1.0, u1=0.0)]
        if b["mode"] == "lying":                                            # A124: up onto her tall kneel, then as from a kneel
            return [dict(type="lie", at=b["at"], yaw=b["yaw"], u0=1.0, u1=0.0, chest_up=b.get("lean", LIE_CHEST_UP[0])),
                    dict(type="plan", what="stand_up", args={})]
        if b["mode"] not in ("heels", "tall"):
            return []
        yaw = float(b["yaw"]); fwd = np.array([math.cos(yaw), math.sin(yaw)])
        T = np.asarray(b["at"], float) + (fwd * HEELS_BACK if b["mode"] == "heels" else 0)
        out = []
        if b["mode"] == "heels":                                            # rising onto her knees (her knees stay where they are):
            now = self._clear_trunk(frame_segs("kneel", "heels", np.asarray(b["at"], float), yaw))   # never nearer to it than she is,
            if self._clear_trunk(frame_segs("kneel_down", 2.5, T, yaw)) < min(K.CLEAR_M, now - 0.005, 0.0):   # and never into it:
                raise Refuse("she cannot rise onto her knees: the child lies against them (A4)")          # a limb babbling against
                                                                            # her knees she rises from as a body does (A4 stops her
                                                                            # if it presses)
            out.append(dict(type="kneel_down", at=_lst(T), yaw=yaw, u0=3.0, u1=2.0))
        toys = self._toys_xy()
        why = []
        near = min(K.CLEAR_M, self._clear_trunk(frame_segs("kneel", "tall", T, yaw)) - 0.005)   # turning or shuffling away on her
        for turn in (0.0, math.pi / 2, -math.pi / 2, math.pi):                               # knees: never nearer than she is
            y2 = yaw + turn
            f2 = np.array([math.cos(y2), math.sin(y2)]); l2 = np.array([-f2[1], f2[0]])
            if turn and not all(self._clear_trunk(frame_segs("kneel", "tall", T, yaw + turn * v)) >= near for v in (0.5, 1.0)):
                why.append((round(math.degrees(turn)), "turning on her knees touches it")); continue
            for back in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
                T2 = T - f2 * back
                stand = T2 - f2 * STAND_BACK                                # where she stands: her own half-width from furniture (she
                if not (self._in_plan(T2) and self._in_plan(stand)) or self.plan.dist[self.plan.cell(stand)] < K.BODY_R_M or \
                        self.plan.dist[self.plan.cell(T2)] < K.BODY_R_M:        # walks away from there)
                    why.append((round(math.degrees(turn)), back, "furniture")); continue
                spots = [T2 + f2 * 0.40 + l2 * 0.12, T2 + f2 * 0.03 + l2 * 0.115, T2 + f2 * 0.03 - l2 * 0.115, stand]
                if any(np.linalg.norm(xy - q) < 0.15 for xy in toys.values() for q in spots):
                    why.append((round(math.degrees(turn)), back, "a toy where she steps")); continue
                if back > 0 and not all(self._clear_trunk(frame_segs("kneel", "tall", T + (T2 - T) * v, y2)) >= (near if v < 1 else K.CLEAR_M)
                                        for v in (0.5, 1.0)):
                    why.append((round(math.degrees(turn)), back, "shuffling back touches it")); continue
                if not all(self._clear_trunk(frame_segs("kneel_down", u, T2, y2)) >= K.CLEAR_M for u in KNEEL_CHECK_U[::-1]):
                    why.append((round(math.degrees(turn)), back, "getting up touches it")); continue
                if turn:
                    out.append(dict(type="knee_turn", at=_lst(T), yaw0=yaw, yaw1=y2))
                if back > 0:
                    out.append(dict(type="shuffle", p0=_lst(T), p1=_lst(T2), yaw=y2))
                out.append(dict(type="kneel_down", at=_lst(T2), yaw=y2, u0=2.0, u1=0.0))
                if a is not None:
                    a["info"]["up"] = dict(turn_deg=round(math.degrees(turn)), back_m=back)
                return out
        self.kneel_reasons = why
        raise Refuse("she cannot get up without touching the child (A4: she stays kneeling until there is room)")

    def _plan_clear_here(self, a, toys, T, T2, yaw):
        """from her tall kneel, the toys where she will kneel moved aside, away from the child (A6): each set down beside her and a
        little ahead, where she can reach it, clear of the child, of other toys and of the way her knees will shuffle"""
        out = []
        T = np.asarray(T, float); T2 = np.asarray(T2, float)
        fwd = np.array([math.cos(yaw), math.sin(yaw)]); left = np.array([-fwd[1], fwd[0]])
        placed = []
        if any(self._child_has(k) for k in toys):                              # the child now touches one (its limbs sink under the
            a["info"]["replans"] = a["info"].get("replans", 0) + 1            # resting law): it is the child's, so another spot
            if a["info"]["replans"] > 2:
                raise Refuse("no spot to kneel: the toys where she would kneel are the child's")
            me = self.phases[0].get("grp") if self.phases else None
            if me is not None:                                                # the rest of this approach goes with it
                self.phases = [self.phases[0]] + [q for q in self.phases[1:] if q.get("grp") != me]
            return [dict(type="plan", what="approach", args={})]
        for k in toys:
            xy = self.d.xpos[self.toys[k]][:2]
            lat = float((xy - T2) @ left)
            sd = "L" if lat > 0 else "R"
            if self.holding[sd] is not None:                                # a hand holding a toy (to show, to give) keeps it: the
                sd = "R" if sd == "L" else "L"                              # other clears the way
                if self.holding[sd] is not None:                            # both full (A95): one set aside, then the approach planned anew
                    me = self.phases[0].get("grp") if self.phases else None
                    if me is not None:                                        # the rest of this approach (its shuffle in) goes with it
                        self.phases = [self.phases[0]] + [q for q in self.phases[1:] if q.get("grp") != me]
                    return [dict(type="plan", what="set_aside", args=dict(toy=self.holding[sd])),
                            dict(type="plan", what="approach", args={})]
            aside = None
            for sg in ((1, -1) if lat > 0 else (-1, 1)):
                hand = sd
                for dl, df in ((0.35, 0.35), (0.40, 0.20), (0.30, 0.45), (0.45, 0.30), (0.40, 0.05), (0.50, 0.40)):
                    q = T2 + left * sg * dl + fwd * df
                    if not (self._in_plan(q) and self.plan.dist[self.plan.cell(q)] > 0.10 and self.child.clearance_xy(q) > 0.25):
                        continue
                    if any(np.linalg.norm(q - x2) < 0.14 for kk, x2 in self._toys_xy(exclude=(k,)).items()) or \
                            any(np.linalg.norm(q - x2) < 0.14 for x2 in placed):
                        continue
                    if min(_seg_dist(q, T2 + left * s2 * 0.115, T + left * s2 * 0.115) for s2 in (1, -1)) < 0.18:
                        continue
                    if not self._reachable_at(hand, np.r_[q, floor_z(q) + 0.06], T2, yaw, "tall"):
                        continue
                    aside = q; sd_put = hand; break
                if aside is not None:
                    break
            if aside is None:
                raise Refuse(f"nowhere within her reach to set the {k} aside")
            if not self._reachable_at(sd, np.r_[xy, floor_z(xy) + 0.05], T2, yaw, "tall"):
                raise Refuse(f"the {k} lies beyond her reach where she kneels")
            placed.append(aside)
            out += self._pick_phases(sd, k)
            out += self._put_phases(sd, aside)
            a["info"].setdefault("cleared", []).append(k)
        return out

    def _reachable_at(self, sd, grip, at, yaw, mode, palm=(0, 0, -1.0), bend=True):
        """whether her hand reaches a point (palm down) kneeling at a spot, with some trunk inside human ranges"""
        grip = np.asarray(grip, float)
        R = NatR(palm, bend=bend)
        if float(np.linalg.norm(grip[:2] - np.asarray(at, float)[:2])) > 1.0:
            return False                                                    # beyond any lean of hers (arm 0.65 m, trunk 0.35 m)
        saved = self.base
        self.base = dict(mode=mode, at=_lst(at), yaw=yaw, lean=0.0, spine=0.0, twist=0.0)
        try:
            lean, spine, tw, ok = self._solve_trunk({sd: (grip, R, dict(curl=.9, thumb=.8, index=None))}, None, step=10)
        finally:
            self.base = saved
        return ok

    def _pick_phases(self, sd, toy):
        """her hand onto a toy from above, closing on it, and lifting it"""
        sh_open = dict(curl=.2, thumb=.3, index=None); sh_hold = dict(curl=.95, thumb=.85, index=None)
        return [dict(type="reach", hands={sd: dict(k="above_toy", toy=toy, h=0.0)}, via=True, shape={sd: sh_open}),
                dict(type="grasp", side=sd, toy=toy),
                dict(type="plan", what="ease_out", args=dict(side=sd)),
                dict(type="reach", hands={sd: CARRY}, shape={sd: sh_hold})]

    def _plan_ease_out(self, a, side):
        """a toy picked up beside the child is first slid 7 cm on the floor the way that takes it clearest of the child (it may lie
        against or under its hand), then lifted"""
        g, R = self._grip_now(side, plan=True)
        toy = self.holding[side]
        if toy is None:
            return []
        up = np.array([0, 0, 0.12])
        if self._toy_clearance(toy, up) >= 0.015 and self._toy_clearance(toy, up / 2) >= 0.015:
            return []                                                       # it lifts straight up clear of the child
        best = None
        for dist in (0.07, 0.12):
            for deg in range(0, 360, 45):
                v = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg)), 0.0]) * dist + np.array([0, 0, 0.005])
                cl = min(self._toy_clearance(toy, v), self._toy_clearance(toy, v + up))
                if best is None or cl > best[0]:
                    best = (cl, v)
            if best[0] >= 0.015:
                break
        return [dict(type="reach", hands={side: dict(k="fixed", grip=_lst(g + best[1]), R=_lst(R))},
                     shape={side: self._shape_now(side)}, solve=False, n=6)]

    def _child_has(self, toy):
        """C118: the child HAS a toy when it lies within a centimetre of its hands or arms (A4: she never takes a toy from it). Until
        day 27 any contact with its body counted, and a ball lying against the trunk of a still child blocked five lessons in a morning
        (her hides and shows on it refused "the child has the ball"); the lane's own holds are its hands (lane._contacts)"""
        return self._toy_clearance(toy, np.zeros(3), geoms=self.g1_arm_geoms) < 0.01

    def _toy_clearance(self, toy, shift, geoms=None):
        """the least distance from a toy's shapes, moved by `shift`, to the child's (MuJoCo's own geometry, a scratch state); `geoms`:
        the child's shapes to measure against (all of them by default; C118: its arms' and hands' for "the child has it")"""
        m, sd = self.m, self.scratch
        sd.qpos[:] = self.d.qpos
        b = self.toys[toy]
        adr = m.jnt_qposadr[m.body_jntadr[b]]
        sd.qpos[adr:adr + 3] += shift
        mujoco.mj_kinematics(m, sd)
        best = 0.2
        ft = np.zeros(6)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or not m.geom_contype[g]:
                continue
            for h in (self.g1_geoms if geoms is None else geoms):
                if np.linalg.norm(sd.geom_xpos[h] - sd.geom_xpos[g]) - m.geom_rbound[h] - m.geom_rbound[g] < best:
                    best = min(best, float(mujoco.mj_geomDistance(m, sd, g, h, best, ft)))
        return best

    def _put_phases(self, sd, xy, check=False):
        """a toy in her hand set down on the floor at xy (a centimetre up, then let go) and her hand drawn up and back; with `check` the
        release checks the toy landed where she meant it (A125's landing check, a lesson's place: PUT_TOL_M); a toy set aside or cleared
        from her way lands where it lands (A133)"""
        sh_hold = dict(curl=.95, thumb=.85, index=None)
        return [dict(type="reach", hands={sd: dict(k="floor", xy=_lst(xy))}, via=True, shape={sd: sh_hold}),
                dict(type="release", side=sd, **({"at": _lst(xy)} if check else {})),
                dict(type="reach", hands={sd: dict(k="up_from", side=sd)}, shape={sd: dict(curl=.3, thumb=.3, index=None)}, n=3),
                dict(type="proxy", side=sd, on=True),
                dict(type="relax", sides=sd)]

    # ------------------------------------------------------------------ the acts: what each asks of her body
    def _plan(self, a):
        return getattr(self, "_act_" + a["kind"])(a, a["target"])

    def _plan_act(self, a):
        """the act's own plan, made once a transition her base was left in has finished (_settle_phases)"""
        return self._plan(a)

    def _rebase_on_drawn(self, drawn_pelvis):
        """A108 (2026-09-27, C90): her plan's state made to agree with the pose her body was last drawn at. The jump guard's way back
        restores what the last accepted plan came from; when that state no longer draws the pose it drew (her standoff zeroed at
        once by the far branch, a push under lag, a save from before A108), every later plan begins from a pose she is not in and
        is refused as a jump, forever: life day 6's stand-up froze at u 0.17 with her body drawn 0.40 m behind her base, and 260
        acts that day and 40 more the next morning were refused before a turn happened to be planned from the other end. The
        pelvis's gap on the floor plan between the drawn pose and the restored plan moves her base's floor points (at, and a walk's or
        shuffle's p0, p1) by that gap; nothing is drawn differently. Counted (stats['rebased'], the largest gap in 'rebased_m'):
        a rebase is the record of a fault upstream, never the plan"""
        b = self.base
        if b["mode"] in ("held", "sofa", "lie", "lying"):
            return
        gap = np.asarray(drawn_pelvis[:2], float) - np.asarray(self._pose().pos[:2], float)
        n = float(np.linalg.norm(gap))
        if n <= REBASE_TOL_M:
            return
        b["at"] = _lst(np.asarray(b["at"], float) + gap)
        for k in ("p0", "p1"):
            if b.get(k) is not None:
                b[k] = _lst(np.asarray(b[k], float) + gap)
        b["dirty"] = True
        self.stats["rebased"] = self.stats.get("rebased", 0) + 1
        self.stats["rebased_m"] = max(float(self.stats.get("rebased_m", 0.0)), n)

    def _settle_phases(self):
        """A105 (2026-09-27): a base left in a transition when an act starts (a shuffle, a turn on her knees, a walk or a turn on her
        feet stopped where it was; a kneel half done) is settled first, so the plan begins from a pose she is in. A shuffle or a knee
        turn stopped is her tall kneel where her pelvis is (the same height: no jump); a walk or a turn on her feet stopped is her
        standing there; a kneel half done finishes to its nearer end by its own phase, and the act is planned after it (a `plan act`
        phase). Why: on life day 5 her turn was refused 85 times in a morning, "her body would have jumped 0.45 m in a tick (a
        planning fault: foot_R, phase turn)": her base had been left in 'shuffle' by a stopped act, the approach planned from it as
        from standing, and every retry (A102) met the same guard"""
        b = self.base
        mode = b.get("mode")
        if mode in ("shuffle", "knee_turn"):
            self._stable("tall", b["at"], b["yaw"])
        elif mode in ("walk", "turn"):
            self._stable("stand", b["at"], b["yaw"])
        elif mode == "kneel_down":
            u = float(b.get("u", 2.0))
            if abs(u - round(u)) > 1e-6 or round(u) not in (0, 2, 3):
                u1 = 2.0 if 1.0 <= u < 2.5 else (0.0 if u < 1.0 else 3.0)
                self.stats["settled"] = self.stats.get("settled", 0) + 1
                return [dict(type="kneel_down", at=_lst(b["at"]), yaw=float(b["yaw"]), u0=u, u1=u1), dict(type="plan", what="act", args={})]
            self._stable({0: "stand", 2: "tall", 3: "heels"}[int(round(u))], b["at"], b["yaw"])
        else:
            return []
        self.stats["settled"] = self.stats.get("settled", 0) + 1
        return []

    def _near(self, a, where=None, offs=None, alongs=None, need=None):
        return [dict(type="plan", what="approach", args=dict(where=where, offs=offs, alongs=alongs, need=need))]

    def _act_approach(self, a, t):
        return self._near(a)

    def _act_attend(self, a, t):
        return self._near(a, need=f"touch:{K.ATTEND_PARTS}") + [dict(type="plan", what="touch", args=dict(part=K.ATTEND_PARTS))]

    def _act_touch(self, a, t):
        part = t if t in BODY_PARTS or t in ("child", "chest") else "tummy"
        return self._near(a, need=f"touch:{part}") + [dict(type="plan", what="touch", args=dict(part=part))]

    def _act_stand(self, a, t):
        return self._up_phases()

    def _up_phases(self):
        """getting up, planned when it is reached (_plan_stand_up)"""
        return [dict(type="plan", what="stand_up", args={})] if self.base["mode"] in ("heels", "tall", "sofa", "lying") else []

    def _standing_at(self):
        """where she will stand after getting up"""
        b = self.base
        if b["mode"] in ("heels", "tall"):
            fwd = np.array([math.cos(b["yaw"]), math.sin(b["yaw"])])
            T = np.asarray(b["at"], float) + (fwd * HEELS_BACK if b["mode"] == "heels" else 0)
            return T - fwd * STAND_BACK
        if b["mode"] == "sofa":
            return np.asarray(b["at"], float) + np.array([math.cos(b["yaw"]), math.sin(b["yaw"])]) * 0.42
        return np.asarray(b["at"], float)

    def _act_walk(self, a, t):
        if t in (None, "child", "mama"):
            return self._near(a)
        if t in ("door", "hall"):
            return self._up_phases() + [dict(type="plan", what="walk_to", args=dict(xy=list(HALL_SPOT), yaw=0.0, child=True))]
        if t == "sofa":
            fwd = -math.pi / 2                                          # she sits facing the room (-y)
            stand = np.array(self.sofa_spot) + np.array([0, -0.42])
            return self._up_phases() + [dict(type="plan", what="walk_to", args=dict(xy=_lst(stand), yaw=fwd, child=True, goal_r=0.45,
                                                                                goal_clear=0.11)),
                                        dict(type="sit", at=list(self.sofa_spot), yaw=fwd, u0=0.0, u1=1.0)]
        pt = self._resolve_point(t)
        if pt is None:
            raise Refuse(f"no such place: {t}")
        return self._up_phases() + [dict(type="plan", what="walk_to", args=dict(xy=_lst(pt[:2]), yaw=None, child=True))]

    def _plan_walk_to(self, a, xy, yaw=None, child=True, goal_r=0.3, goal_clear=None):
        start = self._standing_at() if self.base["mode"] not in ("stand", "turn") else np.asarray(self.base["at"], float)
        xy = np.asarray(xy, float)
        if yaw is None:
            yaw = math.atan2(xy[1] - start[1], xy[0] - start[0])
        redo = dict(what="walk_to", args=dict(xy=_lst(xy), yaw=float(yaw), child=child, goal_r=goal_r, goal_clear=goal_clear))
        return self._walk_phases(start, xy, yaw, goal_r=goal_r, child=child, goal_clear=goal_clear, again=redo)

    def _act_point(self, a, t):
        if self._resolve_point(t) is None:
            raise Refuse(f"nothing to point at: {t}")
        return [dict(type="point", target=t)]                               # the act ends as her arm comes back (P3's contract)

    def _ph_point(self, a, ph):
        """her arm to the pointing pose (moved there over a few ticks, then kept on the target), held 4 ticks"""
        t = ph.get("t", 0)
        if t == 0:
            tgt = self._resolve_point(ph["target"])
            pose = self._pose(arms=False, look=False)
            best = None
            for sd in ("R", "L"):
                q = pose.copy()
                err = P.point_at(q, sd, tgt)
                bad = len(q.report.get(f"arm_{sd}", {}).get("violations", []))
                if best is None or (err + 0.05 * bad) < best[0]:
                    best = (err + 0.05 * bad, sd, q)
            sd = best[1]
            segs = kin.fk(best[2]); hp, hR = segs[f"hand_{sd}"]
            ph["side"] = sd
            g0, R0 = self._grip_now(sd, plan=True)
            self.arms[sd] = dict(mode="move", to=dict(k="fixed", grip=_lst(hp + hR @ GRIP_LOCAL[sd]), R=_lst(hR)), g0=_lst(g0),
                                 q0=_lst(_mat_to_quat(R0)), t=0, n=3, shape0=self._shape_now(sd), shape1=dict(curl=1.45, thumb=.9, index=0.0), via=None)
        sd = ph["side"]
        if t + 1 < 3:
            self.arms[sd]["t"] = t + 1
            return "run"
        self.arms[sd] = dict(mode="point", target=ph["target"])
        if t + 1 >= 3 + 4:                                                  # held, then her arm back (P3's contract: the act ends as
            i = self.phases.index(ph) if ph in self.phases else -1          # her body stops pointing)
            self.phases.insert(i + 1, dict(type="relax", sides=sd))
            return "done"
        return "run"

    # ---- a hand resting on the child (4.2: attend; P3's "touch")
    def _anchor(self, body, prefer=None):
        """a point on a G1 link's surface facing her, and its outward normal (world): rays from her side at the link's collision
        shapes (MuJoCo's own ray on each shape)"""
        m, d = self.m, self.d
        com = d.xipos[body].copy()
        her = self._head_now() * 0 + self._shoulder("R") if prefer is None else np.asarray(prefer, float)
        best = None
        geoms = [g for g in self.g1_geoms if int(m.geom_bodyid[g]) == body]
        for up in (0.6, 0.3, 0.0):
            dvec = unit(unit(her - com) * (1 - up) + np.array([0, 0, 1.0]) * up)
            start = com + dvec * 0.30
            for g in geoms:
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    r = mujoco.mj_rayMesh(m, d, g, start, -dvec)
                else:
                    r = mujoco.mju_rayGeom(d.geom_xpos[g], d.geom_xmat[g], m.geom_size[g], start, -dvec, int(m.geom_type[g]))
                if r >= 0 and (best is None or r < best[0]):
                    best = (r, start - dvec * r, dvec)
            if best is not None:
                break
        if best is None:
            return com, unit(her - com)
        return best[1], best[2]

    def _plan_touch(self, a, part="tummy", side=None):
        """a hand resting on the named part of the child (a hold at TOUCH_N); 'a|b|c': the first of them her hand reaches from
        where she kneels (attend: its trunk, or where only its feet leave her room, its leg or its foot)"""
        if "|" in part:
            b = self.base
            opts = part.split("|")
            part = next((p_ for p_ in opts if b["mode"] not in ("heels", "tall")
                         or self._need_ok(f"touch:{p_}", np.asarray(b["at"], float) - (np.array([math.cos(b["yaw"]), math.sin(b["yaw"])])
                                                                                         * HEELS_BACK if b["mode"] == "tall" else 0), float(b["yaw"]))),
                        opts[0])
        names = BODY_PARTS.get(part, ("torso_link",))
        her = np.r_[np.asarray(self.base["at"], float), 0.5]
        body = min((self.m.body(n).id for n in names), key=lambda b: float(np.linalg.norm(self.d.xpos[b] - her)))
        sd = side or self._near_hand(self.d.xpos[body])
        if part in ("tummy", "child", "chest"):
            local, normal = self._trunk_face(body, her, z=0.07 if part != "chest" else 0.20)
            return self._hold_phases(a, sd, body, "touch", cap=K.TOUCH_N, local=local, normal=normal)
        return self._hold_phases(a, sd, body, "touch", cap=K.TOUCH_N)

    def _trunk_face(self, body, her, z=0.07):
        """the torso's surface she rests a hand on: its front, back or a side, whichever faces up and toward her most (the belly on
        its back, as the prototype's attend: the torso's front 8 cm out, 7 cm up)"""
        Rl = self.d.xmat[body].reshape(3, 3)
        opts = [((0.080, 0.0, z), (1.0, 0, 0)), ((-0.070, 0.0, z), (-1.0, 0, 0)), ((0.0, 0.105, z), (0, 1.0, 0)), ((0.0, -0.105, z), (0, -1.0, 0))]
        c = self.d.xpos[body]
        def score(o):
            n = Rl @ np.array(o[1]); pt = c + Rl @ np.array(o[0])
            return float(n[2]) + 0.5 * float(n @ unit(her - pt))
        loc, nor = max(opts, key=score)
        return list(loc), list(nor)

    def _near_hand(self, pt):
        yaw = self.base["yaw"]
        left = np.array([-math.sin(yaw), math.cos(yaw)])
        return "L" if float((np.asarray(pt)[:2] - np.asarray(self.base["at"], float)) @ left) > 0 else "R"

    def _surface(self, body, pt, n):
        """the point where a line along -n through pt, from 0.3 m out, first meets the G1's collision shapes there (the held link's
        or a neighbour's: the G1's forearm is a chain of short housings; its drawn surface, MuJoCo's ray on each mesh or primitive),
        or pt when it misses"""
        m, d = self.m, self.d
        start = np.asarray(pt, float) + n * 0.30
        best = None
        for g in self.g1_geoms:
            if float(np.linalg.norm(d.geom_xpos[g] - np.asarray(pt, float))) > m.geom_rbound[g] + 0.05:
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                r = mujoco.mj_rayMesh(m, d, g, start, -n)
            else:
                r = mujoco.mju_rayGeom(d.geom_xpos[g], d.geom_xmat[g], m.geom_size[g], start, -n, int(m.geom_type[g]))
            if r >= 0 and (best is None or r < best):
                best = r
        return np.asarray(pt, float) if best is None else start - n * best

    def _palm_depth(self, sd, shape):
        """how far her hand's shapes reach out of its palm side, from her hand's axis (m): the palm, the capsule, each finger and
        the thumb in this hand shape (her grip point lies GRIP_LOCAL's 4.5 cm out on that side)"""
        m = self.m
        sg = kin.side_sign(sd)
        deep = 0.0
        for g, p_, q_, hl in self._hand_geoms_local(sd, shape):
            R = _quat_to_mat(q_)
            ends = [p_ + R[:, 2] * hl, p_ - R[:, 2] * hl]
            deep = max(deep, max(float(-sg * e[1]) for e in ends) + float(m.geom_size[g, 0]))
        for g in self.hand_all[sd]:
            if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                deep = max(deep, float(-sg * m.geom_pos[g][1]) + float(m.geom_size[g, 1]))
            elif int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_CAPSULE and g == self.hand_geom[sd]:
                deep = max(deep, float(m.geom_size[g, 0]))
        return deep

    def _hold_target(self, sd, body, local=None, normal=None):
        """where her hand goes on a G1 link: a point on its surface (along the given normal from `local`, or a ray from her side,
        _anchor) with her open flat hand (HOLD_SHAPE) on it, palm to it, the fingers along her forearm; its grip set so her palm,
        fingers and thumb lie HAND_CLEAR_M outside the child's collision surface (their reach out of her palm, then MuJoCo's own
        distance to its convex hulls with her arm as it would reach, moved out along the normal by what it lacks).
        -> (hand target spec, the held point and normal in the link's frame, the hand shape)"""
        m, d = self.m, self.d
        Rl = d.xmat[body].reshape(3, 3)
        if local is None:
            pt, n = self._anchor(body, prefer=self._shoulder(sd))
            n = unit(n)
        else:
            n = unit(Rl @ np.asarray(normal, float))
            pt = self._surface(body, d.xpos[body] + Rl @ np.asarray(local, float), n)
        loc = Rl.T @ (pt - d.xpos[body]); nl = Rl.T @ n
        shape = dict(curl=K.HOLD_SHAPE[0], thumb=K.HOLD_SHAPE[1], index=None)
        g_in = float(abs(GRIP_LOCAL[sd][1])) - (self._palm_depth(sd, shape) + K.HAND_CLEAR_M)
        goff = -nl * g_in
        to = dict(k="link", body=int(body), local=_lst(loc), goff=_lst(goff), palm=_lst(-nl))
        base = self._pose(arms=False, look=False)
        for _ in range(3):                                                  # her hand as it would reach there, measured and moved out
            g, R = self._realize(base.copy(), sd, *self._resolve_hand(to, sd, base))
            cl = self._hand_clearance_at(sd, g - R @ GRIP_LOCAL[sd], R, shape)
            if cl >= K.HAND_CLEAR_M:
                break
            goff = goff + nl * (K.HAND_CLEAR_M - cl + 0.001)
            to = dict(to, goff=_lst(goff))
        return to, loc, nl, shape

    def _hold_phases(self, a, sd, body, kind, cap, brief=False, local=None, normal=None, ctl=None, tall_if_needed=True):
        """her hand onto a link of the child along its surface normal, then a capped spring there (the kind's controller runs it)"""
        to, loc, nl, shape = self._hold_target(sd, body, local, normal)
        goff = np.asarray(to["goff"], float)
        out = []
        if tall_if_needed and self.base["mode"] == "heels":
            g, R = self._resolve_hand(to, sd)
            lean, spine, tw, ok = self._solve_trunk({sd: (g, R, shape)}, self.warm.get("trunk"))
            if not ok:
                b = self.base
                fw = np.array([math.cos(b["yaw"]), math.sin(b["yaw"])])
                out.append(dict(type="kneel_down", at=_lst(np.asarray(b["at"], float) + fw * HEELS_BACK), yaw=b["yaw"], u0=3.0, u1=2.0))
        above = dict(to, goff=_lst(goff + nl * K.APPROACH_M))
        out.append(dict(type="reach", hands={sd: above}, shape={sd: shape}))              # her hand over the point, along its normal,
        out.append(dict(type="reach", hands={sd: to}, shape={sd: shape}, n=3))           # then onto it (its proxy on: it yields
                                                                                          # like the rest of her), HAND_CLEAR_M out;
                                                                                          # the hold switches it off (its grip)
        out.append(dict(type="hold", name=f"{kind}_{sd}", side=sd, body=int(body), local=_lst(loc), normal=_lst(nl), goff=_lst(goff),
                        kind=kind, cap=float(cap), brief=bool(brief), ctl=ctl or {}, shape=shape))
        return out

    def _ph_hold(self, a, ph):
        """a capped spring engaged at a held point (the hand is drawn there, its proxy off: the spring is its grip), then run by its
        kind's controller until it ends ('done'), stops at its cap ('stopped') or keeps holding after the act ('keep')"""
        if not ph.get("engaged"):
            if not ph.get("tried"):                                         # (an instrument: her holds tried and engaged, by kind)
                ph["tried"] = True
                ht = self.stats.setdefault("hold_tries", {}); ht[ph["kind"]] = ht.get(ph["kind"], 0) + 1
            e = self.arms[ph["side"]].get("err") or 0.0
            if e > K.HOLD_SLIP_M:                                           # a spring from a hand that is not there would be a force
                return f"her {'left' if ph['side'] == 'L' else 'right'} hand cannot reach it from here ({100 * e:.0f} cm short)"
            Rb = self.d.xmat[int(ph["body"])].reshape(3, 3)                 # her grip where it arrived, against where the hold wants
            g, _ = self._grip_now(ph["side"], actual=True)                    # it on the link now (the child moves while she reaches:
            want = self.d.xpos[int(ph["body"])] + Rb @ (np.asarray(ph["local"], float) + np.asarray(ph["goff"], float))   # a spring
            miss = float(np.linalg.norm(g - want))                          # from a hand that is not there would be a force from
            if miss > 0.03:                                                 # nothing: the W2 fix's babble seed 3, a hand 25 cm off);
                hb = self.bm.hand_body[ph["side"]]                          # a hand that met the link on its way (it moved to meet
                on = self._hand_clearance_at(ph["side"], self.d.xpos[hb], self.d.xmat[hb].reshape(3, 3),   # her hand)
                                             self._shape_now(ph["side"]), held=int(ph["body"]))[0] <= K.HAND_FREE_M
                if not (on and miss <= K.HOLD_SLIP_M):                      # is on it, within a hand's length of where she aimed;
                    if ph.get("waited", 0) < K.ARRIVE_WAIT_TICKS:            # her hand is a body: it may still be on its way (she
                        ph["waited"] = ph.get("waited", 0) + 1              # waits for it a moment), and takes it where it is,
                        return "run"                                        # within a hand's length (her grip holds the less the
                    if miss > K.HOLD_SLIP_M:                                # farther: GRIP_TOL_M)
                        c_ = ph.get("ctl") or {}
                        if ph["kind"] == "gather" and int(c_.get("gather_tries", 0)) < K.GATHER_RETRIES:
                            # A166 (2026-10-01): THE GATHER PLANNED AGAIN WHERE THE ARM IS. Life day 57's first pull-to-sit with the gather
                            # (tick 2,754,601): her hand reached for the forearm where it lay and the arm moved 31 cm before the hand
                            # arrived, and the act was refused. A parent reaching for a flailing arm reaches again where it is now: the
                            # gather's phases are made again from the forearm's present place, up to GATHER_RETRIES times
                            if a is not None:
                                a["info"]["gather_retries"] = a["info"].get("gather_retries", 0) + 1
                            new = self._gather_phases(a, c_["gather_cs"], ph["side"], int(c_.get("gather_tries", 0)) + 1)
                            i = self.phases.index(ph)
                            self.phases[i:i + 1] = new
                            return "next"
                        return f"her {'left' if ph['side'] == 'L' else 'right'} hand did not arrive on it ({100 * miss:.0f} cm off: it moved)"
            ph["engaged"] = True
            he = self.stats.setdefault("holds_engaged", {}); he[ph["kind"]] = he.get(ph["kind"], 0) + 1
            h = Hold(ph["name"], ph["body"], ph["local"], ph["side"], ph["cap"], ph["brief"], ph["kind"], ph["normal"])
            h.ctl = dict(ph["ctl"]); h.ctl["t"] = 0
            self.holds = [x for x in self.holds if x.name != h.name] + [h]
            Rb = self.d.xmat[h.body].reshape(3, 3)                          # her grip as her hand arrived: fixed on the link from now
            g, Rh = self._grip_now(ph["side"], actual=True)
            goff = _lst(Rb.T @ (g - h.point(self.d)))
            self.arms[ph["side"]] = dict(mode="hold", hold=h.name, goff=goff, goff0=list(goff), goff_plan=list(ph["goff"]),
                                         Rl=_lst(Rb.T @ Rh), shape1=ph["shape"])
            self._start_ctl(a, h)
            if not ph.get("wait", True):
                return "next"
        h = self._hold(ph["name"])
        if h is None:
            return "done"
        st = h.ctl.get("state", "run")
        if a is not None:
            a["info"]["hold_peak"] = max(a["info"]["hold_peak"], h.peak)
            a["info"]["hold_cap"] = max(a["info"]["hold_cap"], h.cap)
        if st in ("run",):
            return "run"
        if st == "keep":
            return "done"
        if st == "done":
            if a is not None and h.kind in ("guide", "knee") and "moved" in h.ctl:
                c = h.ctl                                                   # what the guide did, said as it was (the W2 verifier: a
                a["why"] = (f"its path ran its {c['n']} ticks: the held point moved {100 * c['moved']:.1f} of {100 * c['dist']:.1f} "
                            f"cm along it, at most {h.peak:.0f} N (her cap {h.cap:.0f} N, the limb's own push "   # 'done' at 99 N,
                            f"{c.get('push', 0.0):.0f} N)")                                                      # 5.7 of 12 cm)
            return "done"
        if st.startswith("stopped"):
            return st
        return "run"

    # ------------------------------------------------------------------ the holds' controllers (every tick)
    def _hold_tick(self):
        for h in list(self.holds):
            c = h.ctl
            c["t"] = int(c.get("t", 0)) + 1
            arm = self.arms[h.side]
            e = arm.get("err") if arm.get("mode") == "hold" else None
            if arm.get("mode") == "hold" and arm.get("hold") == h.name:   # her hand as the physics has it, off its grip on the held
                Rl = self.d.xmat[h.body].reshape(3, 3)                      # point by a hand's length (the child's own links pushed it
                g_act = self._grip_now(h.side, actual=True)[0]              # off, or it could not keep up): the grip is gone
                off = float(np.linalg.norm(g_act - (h.point(self.d) + Rl @ np.asarray(arm["goff"], float))))
                if h.kind in ("turn", "gather", "pull") and off > K.GRIP_TOL_M:   # C88 (A103): the turn's hand GRASPS the limb's root: while it
                    # A168 (2026-10-02): and so do the pull-to-sit's hands on its forearms (A9: 'takes both its forearms'), the gather's
                    # too. Day 58's second pull with the whole chain (tick 2,804,015): both forearms gathered, the pull begun at the
                    # gather's force, and both holds lost within a tick ('her hands lost their hold on it before she began'): a live
                    # child's arm jerks a hand's length in a tick and the hold, a spring with a slip tolerance, counted it slipped,
                    # where the same sequence runs to her cap on the rig's still child. A grasp around a forearm rides with it
                    hb = self.bm.hand_body[h.side]                          # still touches the held link it keeps its hold where the
                    on = self._hand_clearance_at(h.side, self.d.xpos[hb], self.d.xmat[hb].reshape(3, 3), self._shape_now(h.side),
                                                 held=int(h.body))[0] <= K.HAND_FREE_M
                    if on:                                                  # link has carried it (the grip re-anchored on the link),
                        arm["goff"] = _lst(Rl.T @ (g_act - h.point(self.d)))   # and slips only once it has left the link
                        off = 0.0
                e = max(e or 0.0, off)
            if e is not None and e > K.HOLD_SLIP_M:                         # the held point left her reach: it slipped from her grip
                if h.kind == "prop" and c.get("fall"):
                    self._fall_done(c, "out of her reach: it slipped from her hands")
                self.holds = [x for x in self.holds if x is not h]
                self.stats["slips"] = self.stats.get("slips", 0) + 1
                g, R = self._grip_now(h.side, plan=True)
                self.arms[h.side] = dict(mode="at", to=dict(k="fixed", grip=_lst(g), R=_lst(R)), shape1=self._shape_now(h.side))
                continue
            getattr(self, "_ctl_" + h.kind)(h, c)

    def _start_ctl(self, a, h):
        c = h.ctl
        p = h.point(self.d)
        c["p0"] = _lst(p)
        c["state"] = "run"
        c["hover"] = 0.0
        if h.kind in ("guide", "knee", "gather"):
            dirv = unit(np.asarray(c["dir"], float))
            dist = min(float(c["dist"]), K.GUIDE_SPEED * K.GUIDE_MAX_TICKS * TICK_S)
            c.update(dir=_lst(dirv), dist=dist, n=int(max(2, min(K.GUIDE_MAX_TICKS, math.ceil(dist / K.GUIDE_SPEED / TICK_S - 1e-9)))))
            push = self._limb_push(h, dirv, c["limb"])
            cap = min(K.GUIDE_CAP_FACTOR * push, K.CAP_ONE, float(c.get("max", K.CAP_ONE)))
            h.cap = cap; c["push"] = push
            if a is not None:
                a["info"]["limb_push"] = push; a["info"]["guide_cap"] = cap
        elif h.kind == "prop":
            c.update(mode="hold", k=0, steady=0, last_up=-99, best=self.child.trunk_deg, ref=_lst(p), head_z=float(self.child.head[2]),
                     react=0, alone=0, catches=0, falls=0)
            h.cap = K.PROP_EASE[0] * K.CAP_TWO / 2
        elif h.kind in ("pull", "turn"):
            c.update(ramp=0.0, top=0)
            h.cap = 0.0
            h.brief = True

    def _ctl_fixed(self, h, c):
        """an instrument's hold (tests): the target stays where the test put it (ctl target)"""
        h.next = np.asarray(c["target"], float)

    def _ctl_touch(self, h, c):
        n = self.d.xmat[h.body].reshape(3, 3) @ h.n_dir
        h.next = h.point(self.d) - n * K.TOUCH_DEPTH_M
        if c["state"] == "run" and c["t"] >= 2:
            c["state"] = "keep"

    def _ctl_guide(self, h, c):
        if c["state"] != "run":
            h.next = h.point(self.d)
            return
        u = _smooth(min(1.0, c["t"] / c["n"]))
        h.next = np.asarray(c["p0"], float) + np.asarray(c["dir"], float) * c["dist"] * u
        if h.at_cap_ticks >= K.AT_CAP_TICKS:
            c["state"] = "stopped: the guide sat at its cap for 2 ticks (the child resisted, or its joint is at its range: A10)"
            h.cap = 0.0
        elif c["t"] >= c["n"] + 2:
            c["state"] = "done"
            c["moved"] = float((h.point(self.d) - np.asarray(c["p0"], float)) @ np.asarray(c["dir"], float))

    _ctl_knee = _ctl_guide

    def _ctl_gather(self, h, c):
        """A165: the guide's path toward the gathering point, then the hold KEPT (the pull takes it over)"""
        self._ctl_guide(h, c)
        if c["state"] == "done":
            c["state"] = "keep"
            h.next = h.point(self.d)

    def _limb_push(self, h, dirv, limb):
        """the limb's own push at this pose along dirv at the held point: the largest force there its joints can resist, each at its
        limit this tick (weakness included): min over the limb's joints of limit / |J_i . dir| (A10's "the limb's own push")"""
        m, d = self.m, self.d
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, None, h.point(d), h.body)
        lim = self.w.limits_now()
        sl = self.w.eff_slices[limb]
        best = float("inf")
        for j in range(sl.start, sl.stop):
            lever = abs(float(jacp[:, self.w.dof[j]] @ dirv))
            if lever > 1e-4:
                best = min(best, float(lim[j]) / lever)
        return best

    def _ctl_prop(self, h, c):
        """the prop (A9): the trunk held where it reached (its most upright), the cap eased 100/70/40/20% of both hands' sustained cap
        each time the trunk stays within 20 deg of vertical for 10 ticks, back up a step when it leaves 20 deg, then her hands
        hovering 5 cm off; the catch 2 ticks after the trunk passes 35 deg or the head drops faster than 0.5 m/s, within her brief
        caps; beyond 30 deg after a catch she lays it back gently"""
        ch = self.child
        th = ch.trunk_deg
        vz = (float(ch.head[2]) - c["head_z"]) / TICK_S
        c["head_z"] = float(ch.head[2])
        p = h.point(self.d)
        mode = c["mode"]
        c["theta"] = th
        if mode == "hold":
            if th < c["best"]:
                c["best"] = th; c["ref"] = _lst(p)
            h.next = np.asarray(c["ref"], float)
            if th <= K.PROP_STEADY_DEG:
                c["steady"] += 1
                if c["steady"] >= K.PROP_STEADY_TICKS:
                    c["steady"] = 0; c["k"] += 1
                    if c["k"] >= len(K.PROP_EASE):
                        c["mode"] = "hover"; h.cap = 0.0; c["hover"] = K.HOVER_M; c["alone"] = 0
                        return
            else:
                c["steady"] = 0
                if c["k"] > 0 and c["t"] - c["last_up"] >= K.PROP_STEADY_TICKS:
                    c["k"] -= 1; c["last_up"] = c["t"]
            h.cap = K.PROP_EASE[min(c["k"], len(K.PROP_EASE) - 1)] * K.CAP_TWO / 2
            if th > K.CATCH_DEG and h.at_cap_ticks >= K.AT_CAP_TICKS:
                c["mode"] = "lay"; c["lay_t"] = 0; c["lay_cap"] = h.cap; c["falls"] += 1
        elif mode == "hover":
            h.next = p
            h.cap = 0.0
            if th > K.CATCH_DEG or vz < -K.CATCH_HEAD_MPS:
                c["mode"] = "react"; c["react"] = K.REACTION_TICKS
                c["fall"] = dict(t=int(self.tick), start_deg=round(th, 1), peak_deg=round(th, 1))
            else:
                c["alone"] += 1
                c["sat_alone"] = bool(c.get("sat_alone")) or (c["alone"] >= 5 and th <= K.PROP_MAX_DEG)
                if th < c["best"]:                                          # where it sat most upright: where a catch pushes it
                    c["best"] = th; c["ref"] = _lst(p)                      # back to (a spring settles short of its target)
        elif mode == "react":
            c["react"] -= 1
            h.next = p
            c["fall"]["peak_deg"] = round(max(c["fall"]["peak_deg"], th), 1)
            if c["react"] <= 0:                                             # her hands push at once toward where it sat (her brief caps
                c["mode"] = "catch"; c["hover"] = 0.0; c["catch_t"] = 0; c["catches"] += 1   # while she has them: A25), from this
                h.cap = K.CAP_TWO_BRIEF / 2; h.brief = True                                   # tick on: her 300 ms reaction
                h.next = np.asarray(c["ref"], float)
        elif mode == "catch":
            h.next = np.asarray(c["ref"], float)
            c["catch_t"] += 1
            c["fall"]["peak_deg"] = round(max(c["fall"]["peak_deg"], th), 1)
            if th <= K.PROP_MAX_DEG:
                self._fall_done(c, "caught: back within 30 deg")
                c.update(mode="hold", k=0, steady=0, best=th, ref=_lst(p)); h.brief = False
                h.cap = K.PROP_EASE[0] * K.CAP_TWO / 2
            elif c["catch_t"] * TICK_S >= K.BRIEF_S:                        # her brief effort spent: she lays it back gently
                self._fall_done(c, "laid back")
                c["mode"] = "lay"; c["lay_t"] = 0; c["lay_cap"] = h.cap; c["falls"] += 1
        elif mode == "lay":
            c["lay_t"] += 1
            n = K.LAY_BACK_S / TICK_S
            h.cap = c["lay_cap"] * max(0.0, 1 - c["lay_t"] / n)
            h.next = np.asarray(c["ref"], float)
            if c["lay_t"] >= n:
                c["state"] = "stopped: the trunk fell beyond 30 deg and she laid it back (A9, A25)"
                h.brief = False

    def _fall_done(self, c, how):
        """a fall from hovering ended: kept for the instruments (C6's count), its start, its deepest angle and how it ended"""
        f = dict(c.get("fall") or {}, how=how, end=int(self.tick))
        c["fall"] = None
        falls = self.stats.setdefault("falls", [])
        if falls and falls[-1].get("t") == f.get("t"):                      # both hands follow one fall: written once
            return
        falls.append(f)
        del falls[:-50]

    def _ctl_pull(self, h, c):
        """the pull-to-sit (A9): both forearms pulled toward her and up with a force growing at CAP_RAMP_NPS to her brief cap; the
        trunk risen within 30 deg of vertical is its own sit; the pull at its full cap for 2 ticks stops, and she lays it back gently"""
        p = h.point(self.d)
        ch = self.child
        mode = c.get("mode", "pull")
        if mode == "pull":
            if not c.get("go"):                                             # both hands on before the pull begins
                h.cap = 0.0; h.next = p
                return
            c["go_t"] = c.get("go_t", 0) + 1
            c["ramp"] = min(K.CAP_TWO_BRIEF, K.CAP_RAMP_NPS * c["go_t"] * TICK_S)
            h.cap = c["ramp"] / 2
            h.next = p + np.asarray(c["dir"], float) * K.PULL_LEAD_M
            full = c["ramp"] >= K.CAP_TWO_BRIEF - 1e-9 or self.brief_s >= K.BRIEF_S
            eff = sum(float(np.linalg.norm(x.force)) for x in self.holds if x.kind == "pull")
            cap_now = K.CAP_TWO_BRIEF if self.brief_s < K.BRIEF_S else K.CAP_TWO
            c["top"] = c["top"] + 1 if full and eff >= K.AT_CAP * min(cap_now, c["ramp"]) else 0
            c["theta"] = ch.trunk_deg
            if ch.trunk_deg <= K.PROP_MAX_DEG:
                c["state"] = "done"; c["sat"] = True
            elif c["top"] >= K.AT_CAP_TICKS * 1:
                c["mode"] = "lay"; c["lay_t"] = 0; c["lay_cap"] = h.cap; c["ref"] = _lst(p)
            elif c["go_t"] * TICK_S >= K.PULL_MAX_S:                        # her grip or her arms gave out short of her cap (she is a
                c["mode"] = "lay"; c["lay_t"] = 0; c["lay_cap"] = h.cap; c["ref"] = _lst(p)   # body: GRIP_TOL_M)
                c["gave_out"] = True
        else:
            c["lay_t"] += 1
            n = K.LAY_BACK_S / TICK_S
            h.cap = c["lay_cap"] * max(0.0, 1 - c["lay_t"] / n)
            h.next = np.asarray(c["ref"], float)
            if c["lay_t"] >= n:
                c["state"] = ("stopped: her arms could not carry the pull to her cap (her strength, her grip); laid back gently (A9)"
                              if c.get("gave_out") else
                              "stopped: the pull sat at her cap for 2 ticks; it rises only with its own flexion (A9), laid back gently")

    def _ctl_turn(self, h, c):
        """the brief turn from its front (A7): its near shoulder and hip lifted straight up (the body rolls about its far edge by
        itself; a push across it would slide it), then, past its side, over toward its back; the force growing at CAP_RAMP_NPS
        within her brief caps, at most TURN_MAX_S"""
        ch = self.child
        p = h.point(self.d)
        if not c.get("go"):                                                 # both hands on before the push begins: the hand that is
            h.cap = K.TOUCH_N; h.next = p                                   # on rests on its link with a resting hand's force and rides
            return                                                          # it as the child moves (A103; at 0 N it did not follow)
        c["go_t"] = c.get("go_t", 0) + 1
        up = np.array([0, 0, 1.0])
        toward = np.asarray(c["toward"], float) if "toward" in c else -np.asarray(c["away"], float)
        chest_z = float(ch.torso_R[2, 0])                                   # -1 face down, 0 on its side, +1 on its back
        dirv = unit(up + toward * 0.5) if chest_z < -K.TURN_SIDE_Z else unit(up * K.TURN_OVER_UP + toward)   # A101: the far side
                                                                            # up and toward her (about its near edge), then over
        h.next = p + dirv * K.TURN_LIFT_M
        c["ramp"] = min(K.CAP_TWO_BRIEF, K.TURN_RAMP_NPS * c["go_t"] * TICK_S)   # A101: the roll's force built within half a second
        h.cap = min(c["ramp"] / 2, K.CAP_ONE_BRIEF)
        c["posture"] = ch.posture
        if ch.posture == "back" or chest_z >= K.TURN_PAST_Z:                # onto its back, or past its side by TURN_PAST_Z (C167: at its
            c["state"] = "done"; c["turned"] = True                          # side alone it tipped back under A143's stiff limbs)
        elif c["go_t"] * TICK_S >= K.TURN_MAX_S:
            deg = math.degrees(math.acos(float(np.clip(-chest_z, -1.0, 1.0))))
            if chest_z >= K.TURN_PAST_Z:                                    # A136/C167: past its side by the margin at her cap's end is the
                c["state"] = "done"; c["turned"] = True                      # turn done; at its side alone it lies balanced and falls
            else:                                                           # back (or is helped over); the head route's two
                c["state"] = (f"stopped: not turned within her caps in {K.TURN_MAX_S:.0f} s (A7, A101; its chest turned {deg:.0f} deg from face down, "
                              "a side is 90): she narrates and helps by the roll's ladder instead")   # hands turn it 106 deg in the 4 s

    # ---- toys: fetching, showing, handing over, bringing back, clearing (4.10, A4, A5, A6)
    def _toy(self, t):
        if t not in self.toys:
            raise Refuse(f"no such toy in the room: {t}")
        return t

    def _fetch(self, a, toy):
        """the toy into her hand: nothing if she holds it; from where she kneels if it lies within her reach there; else she comes to
        it (kneeling beside it, the side away from the child) and picks it up"""
        if toy in self.holding.values():
            return []
        if toy in NEVER_FETCHED:                                            # C119: the bucket stays where it stands (the hide game's container,
            raise Refuse(f"the {toy} stays where it stands: she does not carry it (C119)")   # A129; her pick and put of it miss by 17 to 50 cm)
        return [dict(type="plan", what="fetch", args=dict(toy=toy))]

    def _plan_fetch(self, a, toy):
        if toy in self.holding.values():
            return []
        if toy in NEVER_FETCHED and a.get("info", {}).get("carry_ok") != toy:   # C125: every road to a fetch ends here (her way cleared, the
            raise Refuse(f"the {toy} stays where it stands: she does not carry it (C119)")   # nearest toy, a word's show or pick-up); C138: the hide carries it
        if all(v is not None for v in self.holding.values()):          # both hands full (A95): the toy she needs least set aside first
            other = self.holding[self._near_hand(self.d.xpos[self.toys[toy]])]
            other = other if other is not None else next(v for v in self.holding.values() if v is not None)
            return [dict(type="plan", what="set_aside", args=dict(toy=other)), dict(type="plan", what="fetch", args=dict(toy=toy))]
        if self._child_has(toy):                                        # C118: a toy at its hands or arms is its (A4); one lying
            raise Refuse(f"the child has the {toy}: she never takes a toy from it (A4)")   # against its trunk or legs is not
        c = self.d.xpos[self.toys[toy]].copy()
        if c[2] > 1.0 or not self._in_plan(c[:2]):
            raise Refuse(f"the {toy} is out of her reach (A5)")
        sd = self._free_hand(c)
        if self.base["mode"] in ("heels", "tall") and self._reachable(sd, dict(k="above_toy", toy=toy, h=0.0)):
            return self._pick_phases(sd, toy)
        if self.base["mode"] in ("heels", "tall", "sofa", "lying"):
            return self._up_phases() + [dict(type="plan", what="fetch", args=dict(toy=toy))]   # up first, then planned from there
        spot = self._toy_spot(c[:2])
        if spot is None:
            raise Refuse(f"nowhere to kneel by the {toy}")
        T2, yaw = spot
        stand = T2 - np.array([math.cos(yaw), math.sin(yaw)]) * STAND_BACK
        out = []
        start = self._standing_at()
        out += self._walk_phases(start, stand, yaw, goal_r=0.3, skip=(toy,), again=dict(what="fetch", args=dict(toy=toy), grp=True))
        out.append(dict(type="kneel_down", at=_lst(T2), yaw=yaw, u0=0.0, u1=2.0))
        out.append(dict(type="plan", what="pick", args=dict(toy=toy)))
        self.serial += 1
        for ph in out:
            ph["grp"] = f"fetch{self.serial}"
        return out

    def _grasp_point(self, toy):
        """where her hand closes on a toy: its body's origin, or for an open container (the bucket: its origin is its floor plate,
        extras.add_bucket) the top of the wall nearest her, so she carries it by its rim (C138: her pick of it aimed inside it and
        missed by 17 to 50 cm, C119)"""
        c = self.d.xpos[self.toys[toy]].copy()
        if toy not in TP_OPEN_CONTAINERS:
            return c
        Rt = self.d.xmat[self.toys[toy]].reshape(3, 3)
        her = np.asarray(self.base["at"], float)
        walls = [Rt @ np.array([sx * X.BUCKET_IN, sy * X.BUCKET_IN, X.BUCKET_WALL + X.BUCKET_H]) for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        return c + min(walls, key=lambda w: float(np.linalg.norm((c + w)[:2] - her)))

    def _hold_z(self, toy):
        """how high her grip on a toy sits above the toy's lowest point: its rest height, or a container's rim (C138)"""
        if toy in TP_OPEN_CONTAINERS:
            return X.BUCKET_WALL + X.BUCKET_H
        return float(self.toy_rest.get(toy, 0.05))

    def _plan_pick(self, a, toy):
        c = self.d.xpos[self.toys[toy]]
        return self._pick_phases(self._free_hand(c), toy)

    def _free_hand(self, pt):
        sd = self._near_hand(pt)
        if self.holding[sd] is not None or self._hold_on(sd):
            sd = "L" if sd == "R" else "R"
        if self.holding[sd] is not None:
            raise Refuse("both her hands are full")
        return sd

    def _hold_on(self, sd):
        return any(h.side == sd for h in self.holds)

    def _reachable(self, sd, to):
        g, R = self._resolve_hand(to, sd)
        lean, spine, tw, ok = self._solve_trunk({sd: (g, R, dict(curl=.3, thumb=.3, index=None))}, self.warm.get("trunk"))
        return ok

    def _toy_spot(self, xy):
        """where she kneels (tall) to pick a toy up: facing it, at FETCH_OFF_TRY's first distance the floor leaves her (0.45 m, the
        pick from above at her knees; farther out where furniture or a wall stands on that ring: A117, C102, the cup under the low
        table and the ball in the room's corner, reached from the table's edge or the corner's mouth as a person kneels there and
        reaches in), the spot farthest from the child among those at that distance whose kneeling frames are clear of it (A4) and
        of other toys, and, beyond the first ring, from which her hand reaches the toy from a tall kneel (_reachable_at)"""
        toys = self._toys_xy()
        xy = np.asarray(xy, float)
        for dist in FETCH_OFF_TRY:
            best = None
            for deg in range(0, 360, 20):
                ang = math.radians(deg)
                T2 = xy - np.array([math.cos(ang), math.sin(ang)]) * dist
                yaw = ang
                fwd = np.array([math.cos(yaw), math.sin(yaw)]); left = np.array([-fwd[1], fwd[0]])
                stand = T2 - fwd * STAND_BACK
                if not (self._in_plan(T2) and self._in_plan(stand)) or self.plan.dist[self.plan.cell(stand)] < K.BODY_R_M + 0.05 \
                        or self.plan.dist[self.plan.cell(T2)] < K.BODY_R_M:
                    continue
                feet = [T2 + fwd * 0.40 + left * 0.12, T2 + fwd * 0.03 + left * 0.115, T2 + fwd * 0.03 - left * 0.115, stand]
                if any(np.linalg.norm(q - v) < 0.14 for k, v in toys.items() if np.linalg.norm(v - xy) > 1e-6 for q in feet):
                    continue
                if np.linalg.norm(feet[0] - xy) < 0.12:
                    continue
                cl = self.child.clearance_xy(T2)
                if cl < 0.45:
                    continue
                if dist > FETCH_OFF_TRY[0]:                                 # beyond her knees: her hand must reach it from there
                    sd = "L" if float((xy - T2) @ left) > 0 else "R"
                    if not self._reachable_at(sd, np.r_[xy, floor_z(xy) + 0.05], T2, yaw, "tall"):
                        continue
                if best is None or cl > best[0]:
                    best = (cl, T2, yaw)
            if best is not None:
                T2, yaw = best[1], best[2]
                if self._clearance(frame_segs("kneel", "tall", T2, yaw)) >= K.CLEAR_M:
                    return T2, yaw
        return None

    def _act_show(self, a, t):
        toy = self._toy(t)
        if self._kneel_plan(None, None, None, None) is None:                  # C175 (2026-10-01): no spot to kneel beside the child (a wall, a
            a["info"]["fallback"] = "set_down"                              # corner: day 46's west edge): the toy is set down within its
            return self._act_bring_back(a, t)                               # reach instead (the lure, C168, when that too is out of reach)
        if self.child.posture == "front":                                   # C177 (2026-10-01): THE SHOW'S SPOT REACHES ITS PUT. A show ends by
            near = self._near(a, where="head", offs=PUT_HEAD_OFFS)          # setting the toy down within the child's reach (A100), but its
        else:                                                               # approach carried no need, so off the mat she knelt where the put
            xy = self._put_xy()                                             # then lay 48 to 57 cm beyond her reach (day 47: 13 shows, 13 refused
            near = self._near(a, need=f"reach:{xy[0]:.3f},{xy[1]:.3f}")     # "could not be set down where she meant it"). As bring_back's (A133)
        return self._fetch(a, toy) + near + [dict(type="plan", what="show", args=dict(toy=toy))]

    def _plan_show(self, a, toy):
        sd = [s_ for s_, v in self.holding.items() if v == toy]
        if not sd:
            raise Refuse(f"the {toy} is not in her hand")
        sd = sd[0]
        ch = self.child
        shake = unit(np.cross(ch.axis, [0, 0, 1.0]) if abs(ch.axis[2]) < 0.95 else ch.cam_R["L"][:, 0])
        return [dict(type="reach", hands={sd: dict(k="show")}, shape={sd: dict(curl=.9, thumb=.8, index=None)}, stay=True, shake=_lst(shake)),
                dict(type="shake", side=sd, n=4), dict(type="relax", sides=sd),  # back before her (P3's contract as it was), then
                dict(type="plan", what="put_near", args=dict(toy=toy))]          # A100 (C84): set down within its reach ("here"),
                                                                                  # beside its near hand at her lesson's distance, as the
                                                                                  # reach lesson does; until A100 the shown toy stayed in
                                                                                  # her hand (the rattle all of day 1: a hand lost to her,
                                                                                  # the toy to the child). The act ends as the toy is down

    def _ph_shake(self, a, ph):
        """the shown toy shaken for its sound (4.10) for n ticks; she keeps shaking gently while it is shown"""
        sd = ph["side"]
        arm = self.arms[sd]
        if arm["mode"] == "at":
            arm["shake_t"] = arm.get("shake_t", 0) + 1
        return "done" if ph.get("t", 0) + 1 >= ph["n"] else "run"

    def _act_hand_over(self, a, t):
        toy = self._toy(t)
        free = [x for x in "LR" if not self._hand_full(x)]
        if free and all(self._kneel_plan(cs, None, None, f"hand:{cs}:{toy}") is None for cs in free):
            # C175 (2026-10-01): A HAND-OVER WITH NO SPOT BECOMES A SET-DOWN. Life day 46: the child scooted off the mat's west edge within a
            # thousand ticks and lay along the wall and in the corner; 7 of her 9 hand-overs and all 11 shows were refused ("no spot to kneel
            # beside the child: every side is blocked", her hand 30 to 40 cm short), no act on a toy was smiled in 8,700 ticks. A person who
            # cannot kneel by the baby's hand puts the toy where the baby can reach it: the set-down (and its lure, C168) plans a spot that
            # reaches the floor beside the child, not its palm, and the child's own reach completes the lesson
            a["info"]["fallback"] = "set_down"
            return self._act_bring_back(a, t)
        return self._fetch(a, toy) + [dict(type="plan", what="near_free_hand", args=dict(toy=toy)),
                                      dict(type="plan", what="hand_over", args=dict(toy=toy))]

    def _plan_near_free_hand(self, a, toy=None):
        """she comes to the side of the child's free hand (a hand holding a toy is not given another)"""
        free = [x for x in "LR" if not self._hand_full(x)]
        if not free:
            raise Refuse("both its hands hold something")
        if self.base["mode"] == "heels" and self._beside_now():
            her = np.asarray(self.base["at"], float)
            cs = min(free, key=lambda x: float(np.linalg.norm(self.child.grasp[x][:2] - her)))
            if float(np.linalg.norm(self.child.grasp[cs][:2] - her)) < 0.75:
                return []                                                   # it is within her reach from where she kneels
        order = free if len(free) == 1 else sorted(free, key=lambda x: x != self.child.face_side())
        last = None
        for cs in order:                                                    # C133 (day 33): the spot must put its palm in her reach
            try:                                                            # (six hand-overs refused "beyond her reach" in 4,000
                return self._plan_approach(a, where=cs, need=f"hand:{cs}:{toy}")   # ticks: she knelt at its side, its hand lay far); its
            except Refuse as e:                                             # other free hand's side next
                last = e
        raise last

    def _plan_hand_over(self, a, toy):
        """the toy brought into the child's near palm along its normal; released by A4's rule (the palm pressed at least 0.3 N and
        the fingers closed at least 30 deg for 2 ticks, or after 40 ticks)"""
        her = np.asarray(self.base["at"], float)
        free = [x for x in "LR" if not self._hand_full(x)]
        if not free:
            raise Refuse("both its hands hold something")
        cs = min(free, key=lambda x: float(np.linalg.norm(self.child.grasp[x][:2] - her)))   # its nearer free hand
        sd, swap = self._giving(toy, self.child.grasp[cs])
        if not swap and not self._reachable_at(sd, self.child.grasp[cs] + self.child.palm_n[cs] * 0.04, self.base["at"], self.base["yaw"],
                                                 self.base["mode"] if self.base["mode"] in ("heels", "tall") else "heels",
                                                 palm=-self.child.palm_n[cs], bend=False):
            raise Refuse("its free hand is beyond her reach from where she kneels")
        return swap + [dict(type="reach", hands={sd: dict(k="palm", side=cs, lift=K.SETTLE_ABOVE_M)}, via=True,
                            shape={sd: dict(curl=.9, thumb=.8, index=None)}),   # the toy held a little above its palm until her real
                dict(type="settle", side=sd),                                   # hand is there (she sees it: _aim_fix), then lowered
                dict(type="reach", hands={sd: dict(k="palm", side=cs)}, shape={sd: dict(curl=.9, thumb=.8, index=None)},
                     n=K.SETTLE_LOWER_TICKS),                                   # slowly into it (a physical arm's hand arriving at speed
                dict(type="handover", side=sd, child=cs),                        # met its fingers first, and the grasp closed on nothing)
                dict(type="release", side=sd),
                dict(type="reach", hands={sd: dict(k="up_from", side=sd)}, shape={sd: dict(curl=.3, thumb=.3, index=None)}, n=3),
                dict(type="proxy", side=sd, on=True),
                dict(type="relax", sides=sd)]

    def _ph_settle(self, a, ph):
        """her hand held where it is planned until her real hand is there (within SETTLE_TOL_M: she sees it, _aim_fix), at most
        SETTLE_TICKS"""
        miss = self.arms[ph["side"]].get("miss")
        t = ph.get("t", 0) + 1
        return "done" if (miss is not None and miss <= K.SETTLE_TOL_M) or t >= K.SETTLE_TICKS else "run"

    def _ph_handover(self, a, ph):
        """A4's release: the child's palm touch at least 0.3 N and its fingers closed at least 30 deg for 2 ticks, or 40 ticks. The touch
        is the toy's push on the PALM'S SIDE of its hand (C219, 2026-10-02): on day 61's copy a cup set against the knuckles of a fist
        (74 deg closed on nothing, the cup 25 cm from its grasp point, 0 N on the palm) was 'closed on it' and released to fall"""
        cs = ph["child"]
        w = self.w
        toy = self.holding[ph["side"]]
        palm, inside = self._toy_in_hand(toy, cs) if toy is not None else (0.0, False)   # the child's hand on the toy, as she feels it
        closed = self._closure(cs)                                                        # through it, and the toy within its fingers' reach
        arm = self.arms.get(ph["side"]) or {}
        to = arm.get("to") if arm.get("mode") == "at" and isinstance(arm.get("to"), dict) else None
        if to is not None and to.get("k") == "palm" and toy is not None and self._toy_grip(toy, cs, palmar=True) < K.HANDOVER_PALM_N:
            # C222 (2026-10-02): THE TOY IS PRESSED INTO THE PALM UNTIL SHE FEELS IT THERE. Her reading of the child's palm (its grasp point
            # PALM_GRASP_OUT out of the surface) is a model; on day 62's copy the grasp point lay 0.9 to 3.5 cm out of the palm's face and the
            # ball planned to press 3 mm hung 1 cm clear of it. A person lowers the toy until the baby's hand takes its weight: while the
            # palm's side carries none of it (a touch on the hand's edge or fingers is not the palm's), the toy goes HANDOVER_PRESS_M further
            # in a tick, at most HANDOVER_PRESS_MAX_M past her plan
            if to.get("gap") is None:
                n_ = self.child.palm_n[to["side"]]
                to["gap"] = self._toy_half_along(toy, n_) - PALM_GRASP_OUT - 0.003
            ph.setdefault("gap0", float(to["gap"]))
            if to["gap"] > ph["gap0"] - K.HANDOVER_PRESS_MAX_M:
                to["gap"] = float(to["gap"]) - K.HANDOVER_PRESS_M
                ph["pressed"] = ph.get("pressed", 0.0) + K.HANDOVER_PRESS_M
        if a is not None and ph.get("pressed"):
            a["info"]["pressed_m"] = ph["pressed"]
        ok = palm >= K.HANDOVER_PALM_N and closed >= math.radians(K.HANDOVER_CLOSED_DEG) and inside
        ph["ok"] = ph.get("ok", 0) + 1 if ok else 0
        ph["palm"] = max(ph.get("palm", 0.0), palm)
        t = ph.get("t", 0) + 1
        if a is not None:
            a["info"]["palm_N"] = ph["palm"]; a["info"]["closure_deg"] = math.degrees(closed)
        if ph["ok"] >= K.HANDOVER_HOLD_TICKS:
            a["why"] = "the child's hand closed on it"
            return "done"
        if t >= K.HANDOVER_MAX_TICKS:                                       # A4: "a dropped toy is fine, and makes its sound"
            a["why"] = (f"released after 40 ticks (A4): its hand never closed on the {toy} for 2 ticks (its palm on it at most "
                        f"{ph['palm']:.1f} N of 0.3, its fingers now {math.degrees(closed):.0f} deg of 30): the {toy} was let go, not "
                        f"taken (A4: a dropped toy is fine)")
            a["info"]["held"] = False
            return "done"
        return "run"

    def _hand_full(self, cs):
        """a toy in the child's hand (pressing it from within its fingers' reach, or within 4 cm of its grasp point), as she sees it"""
        for toy, b in self.toys.items():
            if toy in self.holding.values():
                continue
            if float(np.linalg.norm(self.d.xpos[b] - self.child.grasp[cs])) < 0.04:
                return True
            grip, inside = self._toy_in_hand(toy, cs)
            if inside and grip > K.HANDOVER_PALM_N:
                return True
        return False

    def _toy_in_hand(self, toy, cs):
        """(the toy's whole push on the child's hand, N; whether the toy lies within the hand's reach: its centre no further out along the
        palm's normal than its own half-extent HANDOVER_OUT_M beyond the grasp point, as she reads it). C222 (2026-10-02): a toy on the
        knuckles of a fist sits 3 cm and more beyond that (the fist's knuckles stand 5 to 6 cm out of the palm's face); a toy within closed
        fingers pushes the palm across its plane as the hand turns on it (the rig's block at 62 deg closed, 13 N, read by the palm's side
        alone as nothing: C219's rule lost the release the tick the hand took it)"""
        n = self.child.palm_n[cs]
        depth = float((self.d.xpos[self.toys[toy]] - self.child.grasp[cs]) @ n)
        inside = depth <= self._toy_half_along(toy, n) + K.HANDOVER_OUT_M
        return self._toy_grip(toy, cs), inside

    def _toy_grip(self, toy, cs, palmar=False):
        """the normal force between a toy and the child's hand (its palm and fingers) now (N): what she feels of its grip through the
        toy she holds (A4's release reads the toy's own contact, never the palm's whole touch, which a fist closed on itself fills).
        `palmar` (C219): only the pushes from the palm's side, as she reads its hand: a push on a link of the hand directed AWAY from
        the hand's grasp centre (where a held toy's centre sits: HAND_DEPTH_M before the palm's face, which she reads as its grasp
        point out from the palm, A4) is the toy on the palm's face or within the fingers; one directed toward it is the toy against
        the back of the hand or a fist's knuckles (the world's own reading of the two sides, body/sim/world.py A171)"""
        m, d = self.m, self.d
        w = self.w
        hand = set(w.groups["hand_l" if cs == "L" else "hand_r"])
        ti = self.toy_names.index(toy)
        from .world import HAND_DEPTH_M, SIDE_COS as W_SIDE_COS             # (the world's constants; imported here, the world imports her)
        G = self.child.grasp[cs] + self.child.palm_n[cs] * (HAND_DEPTH_M - PALM_GRASP_OUT)
        f = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            g0, g1 = int(c.geom[0]), int(c.geom[1])
            if c.efc_address < 0:
                continue
            if self.toy_of_geom[g0] == ti and w.zone_of_geom[g1] in hand:
                into = np.array(c.frame[:3])                                # the normal, from the toy into the hand's link
            elif self.toy_of_geom[g1] == ti and w.zone_of_geom[g0] in hand:
                into = -np.array(c.frame[:3])
            else:
                continue
            if palmar:                                                      # (A171's rule, both along the palm's normal: the push's part and
                a_ = float(into @ self.child.palm_n[cs])                        # the contact's depth from the grasp centre; a push across the
                s_ = float((np.asarray(c.pos) - G) @ self.child.palm_n[cs])   # normal is no side's)
                if abs(a_) < W_SIDE_COS or a_ * s_ <= 0.0:
                    continue
            f += float(d.efc_force[c.efc_address])
        return f

    def _closure(self, cs):
        """how far the child's fingers have closed from open: the mean angle of the Dex3's flexing joints (the thumb's two, each
        finger's two) from their born, open angles"""
        side = "left" if cs == "L" else "right"
        js = [f"{side}_hand_{j}_joint" for j in ("thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
        q = [abs(float(self.d.qpos[self.m.jnt_qposadr[self.m.joint(j).id]])) for j in js]
        return float(np.mean(q))

    def _swap_phases(self, frm, to, toy):
        """a toy passed from one of her hands to the other before her chest: the holding hand brings it to her midline, the other
        takes it by its centre, the first lets go and rests"""
        sg = kin.side_sign(frm)
        R_from = P.hand_R_palm(frm, [0, -sg, 0.0], [1.0, 0, 0])            # in her chest's frame: the palm toward her other side
        hold = dict(curl=.9, thumb=.8, index=None)
        return [dict(type="reach", hands={frm: dict(k="rel", seg="chest", grip=[0.32, sg * 0.05, -0.08], R=_lst(R_from))},
                     shape={frm: hold}, solve=False),
                dict(type="reach", hands={to: dict(k="toy_centre", toy=toy)}, shape={to: dict(curl=.5, thumb=.5, index=None)}, solve=False, n=4),
                dict(type="grasp", side=to, toy=toy),
                dict(type="release", side=frm),
                dict(type="reach", hands={frm: dict(k="up_from", side=frm)}, shape={frm: dict(curl=.3, thumb=.3, index=None)}, n=2, solve=False),
                dict(type="proxy", side=frm, on=True),
                dict(type="relax", sides=frm)]

    def _giving(self, toy, pt):
        """(the hand that gives, and the phases passing the toy to it first): the hand on the side of the point she gives at"""
        sds = [s_ for s_, v in self.holding.items() if v == toy]
        if not sds:
            raise Refuse(f"the {toy} is not in her hand")
        want = self._near_hand(pt)
        if sds[0] == want or self.holding[want] is not None or self._hold_on(want):
            return sds[0], []
        return want, self._swap_phases(sds[0], want, toy)

    def _put_xy(self):
        """where a lesson's toy goes for a child not on its front (A90, `_plan_put_near`): on the floor beside its near hand, out from its
        body by her lesson's distance"""
        ch = self.child
        her = np.asarray(self.base["at"], float)
        cs = min("LR", key=lambda x: float(np.linalg.norm(ch.grasp[x][:2] - her)))
        out = unit((ch.grasp[cs] - ch.torso)[:2] * 1.0)
        return ch.grasp[cs][:2] + out * float(self.lesson_dist)

    def _act_bring_back(self, a, t):
        toy = self._toy(t)
        if self.child.posture == "front":                                 # A125 (the crawl rung): the toy goes before a prone child's
            near = self._near(a, where="head", offs=PUT_HEAD_OFFS)         # face, so she kneels at its head, where her hand reaches it
        else:                                                             # A133 (room b's lesson): she kneels where her hand reaches the
            xy = self._put_xy()                                           # put's place; a spot beside the child that cannot reach it
            need = f"reach:{xy[0]:.3f},{xy[1]:.3f}"                       # (the room of birth's table kept her nearer) is passed over
            c_ = self.plan.cell(xy); edge_ = (not self._in_plan(xy)) or float(self.plan.dist[c_]) < LURE_EDGE_M
            if edge_ or self._kneel_plan(None, None, None, need) is None:   # C168: no spot she can kneel at reaches the put's place (the child
                xy2 = self._lure_xy()                                     # under the coffee table): the toy goes to the nearest free floor
                if xy2 is not None:                                       # point in its reach instead, and the child must move to it
                    a["info"]["put_xy"] = _lst(xy2); a["info"]["lure"] = True
                    self.stats["lures"] = int(self.stats.get("lures", 0)) + 1; self.stats["lure_xy"] = _lst(xy2)   # (an instrument)
                    need = f"reach:{xy2[0]:.3f},{xy2[1]:.3f}"
            near = self._near(a, need=need)
        return self._fetch(a, toy) + near + [dict(type="plan", what="put_near", args=dict(toy=toy))]

    def _lure_xy(self):
        """C168: the nearest free floor point to the child's near hand (LURE_MIN_M to LURE_MAX_M from it, LURE_CLEAR_M clear of furniture and
        walls on the floor plan) that some spot she can kneel at reaches; None when none of the nearest LURE_TRIES does"""
        ch = self.child
        her = np.asarray(self.base["at"], float)
        cs = min("LR", key=lambda x: float(np.linalg.norm(ch.grasp[x][:2] - her)))
        hand = np.asarray(ch.grasp[cs][:2], float)
        c0 = self.plan.cell(hand); r = int(math.ceil(LURE_MAX_M / K.GRID_M))
        cands = []
        for i in range(max(0, c0[0] - r), min(self.plan.nx, c0[0] + r + 1)):
            for j in range(max(0, c0[1] - r), min(self.plan.ny, c0[1] + r + 1)):
                if self.plan.dist[i, j] < LURE_CLEAR_M:
                    continue
                p = self.plan.point((i, j)); d = float(np.linalg.norm(p - hand))
                if LURE_MIN_M <= d <= LURE_MAX_M:
                    cands.append((d, p))
        cands.sort(key=lambda x: x[0])
        for d, p in cands[:8]:
            if self._kneel_plan(None, None, None, f"reach:{p[0]:.3f},{p[1]:.3f}") is not None:
                return p
        return None

    def _act_hide(self, a, t):
        """A129 (the hide game, on the bucket A126): the toy fetched and let go INTO THE TUB in the child's view: she brings it over the
        rim and opens her hand, so it is gone from the child's eyes and still there (Piaget's stage 4, the object permanence test
        proper; the ruler is the child's hand into the bucket after it)"""
        toy = self._toy(t)
        if "bucket" not in self.toys:
            raise Refuse("no bucket in the room to hide it in")
        if toy == "bucket":
            raise Refuse("the bucket is what hides, not what is hidden")
        c = self.d.xpos[self.toys["bucket"]]
        near = min(float(np.linalg.norm(c[:2] - self.child.grasp[x][:2])) for x in "LR")
        if near > HIDE_REACH_M:                                             # C138: the bucket out of the child's reach (it crawled off, or
            a["info"]["carry_ok"] = "bucket"                                # the bucket was shoved): she brings the bucket beside it first,
            if self.child.posture == "front":                               # carried by its rim, set within its reach as a lesson's toy
                near = self._near(a, where="head", offs=PUT_HEAD_OFFS)      # is (_act_bring_back's road to the put), and the toy let go
            else:                                                           # into it there
                xy = self._put_xy()
                near = self._near(a, need=f"reach:{xy[0]:.3f},{xy[1]:.3f}")
            return self._fetch(a, toy) + [dict(type="plan", what="fetch", args=dict(toy="bucket"))] + near + \
                [dict(type="plan", what="put_near", args=dict(toy="bucket")), dict(type="plan", what="drop_in", args=dict(toy=toy))]
        return self._fetch(a, toy) + [dict(type="plan", what="drop_in", args=dict(toy=toy))]

    def _plan_drop_in(self, a, toy):
        """the toy in her hand carried over the bucket and let go (A129): from where she kneels if her hand reaches over the rim with it,
        else from a kneel beside the bucket (a fetch's spot, the side away from the child); the release checks her hand was over the bucket
        (PUT_TOL_M, as a put's landing)"""
        sds = [s_ for s_, v in self.holding.items() if v == toy]
        if not sds:
            raise Refuse(f"the {toy} is not in her hand")
        sd = sds[0]
        c = self.d.xpos[self.toys["bucket"]].copy()
        h = X.BUCKET_H + 2.0 * float(self.toy_rest.get(toy, 0.05)) + 0.02      # her grip over the rim by the toy's height and 2 cm: the
        to = dict(k="above_toy", toy="bucket", h=h)                            # toy hangs under her hand and clears the rim as it falls
        if self.base["mode"] in ("heels", "tall") and self._reachable(sd, to):
            return self._drop_phases(sd, to, c[:2])
        if self.base["mode"] in ("heels", "tall", "sofa", "lying"):
            return self._up_phases() + [dict(type="plan", what="drop_in", args=dict(toy=toy))]
        spot = self._toy_spot(c[:2])
        if spot is None:
            raise Refuse("nowhere to kneel by the bucket")
        T2, yaw = spot
        stand = T2 - np.array([math.cos(yaw), math.sin(yaw)]) * STAND_BACK
        out = self._walk_phases(self._standing_at(), stand, yaw, goal_r=0.3, skip=(toy,),
                                again=dict(what="drop_in", args=dict(toy=toy), grp=True))
        out.append(dict(type="kneel_down", at=_lst(T2), yaw=yaw, u0=0.0, u1=2.0))
        out.append(dict(type="plan", what="drop_in", args=dict(toy=toy)))
        self.serial += 1
        for ph in out:
            ph["grp"] = f"hide{self.serial}"
        return out

    def _drop_phases(self, sd, to, xy):
        """a toy in her hand brought over the bucket and let go, and her hand drawn up and back (A129)"""
        sh_hold = dict(curl=.95, thumb=.85, index=None)
        return [dict(type="reach", hands={sd: to}, via=True, shape={sd: sh_hold}),
                dict(type="release", side=sd, at=_lst(xy), over=dict(to)),          # C136: the release knows what it hangs over
                dict(type="reach", hands={sd: dict(k="up_from", side=sd)}, shape={sd: dict(curl=.3, thumb=.3, index=None)}, n=3),
                dict(type="proxy", side=sd, on=True),
                dict(type="relax", sides=sd)]

    def _far_side(self):
        """A109: the child's side away from the side she kneels on (Child.face_side), and the lateral direction toward it on the
        floor plan"""
        ch = self.child
        far = "R" if ch.face_side() == "L" else "L"
        return far, ch.lat[:2] * (1 if far == "L" else -1)

    def _far_xy(self):
        """A109: where the roll rung's toy goes: beside the child's far shoulder, level with its head, ROLL_BEYOND_M past the reach of
        the arm on that side -> (the far side, the lateral direction toward it, the spot, the far shoulder's floor point)"""
        ch = self.child
        far, lat = self._far_side()
        sh = ch.body[f"{'left' if far == 'L' else 'right'}_shoulder_roll_link"][0][:2]
        return far, lat, sh + lat * (self.arm_m + ROLL_BEYOND_M), sh

    def _act_bring_far(self, a, t):
        """A109 (C91): the roll rung's setup: the toy fetched, then from a kneel on the child's far side, the toy between her and the
        child, set down beside its far shoulder, level with its head, ROLL_BEYOND_M past the reach of the arm on that side, so a roll
        toward it brings it within reach and the toy is the reward (its own "got"); a person gives a rolling baby a reason to roll"""
        toy = self._toy(t)
        ch = self.child
        far, lat, xy, sh = self._far_xy()
        mid = (ch.torso[:2] + ch.pelvis[:2]) / 2
        d = float((xy - mid) @ lat)                                          # the spot's distance out from the child's middle
        a0 = float((sh - mid) @ ch.len_axis[:2])                             # and its place along the body (its shoulder's)
        offs = (d + PUT_AHEAD_M, d + PUT_AHEAD_M + 0.10, d + PUT_AHEAD_M - 0.10)   # she kneels beyond it: the toy PUT_AHEAD_M before
        alongs = (a0, a0 + 0.10, a0 - 0.10)                                  # her pelvis, as set_near's toy lies (A90)
        return self._fetch(a, toy) + self._near(a, where=far, offs=offs, alongs=alongs) + \
            [dict(type="plan", what="put_far", args=dict(toy=toy))]

    def _plan_put_far(self, a, toy):
        ch = self.child
        far, lat, xy, sh = self._far_xy()
        if not self._in_plan(xy) or self.plan.dist[self.plan.cell(xy)] < 0.10 or ch.clearance_xy(xy) < 0.05:
            raise Refuse(f"no room for the {toy} beside its far shoulder (A109: the mat's edge or the furniture)")
        sd, swap = self._giving(toy, np.r_[xy, 0.0])
        return swap + self._put_phases(sd, xy, check=True)

    def _plan_put_near(self, a, toy):
        """the toy set down within the child's reach: on the floor beside its near hand, out from its body by her lesson's distance
        (lesson_dist: 0.10 at birth, 4.10's reach ladder level 1; her day plan raises it 0.05 a mastered level, A90)"""
        ch = self.child
        her = np.asarray(self.base["at"], float)
        if ch.posture == "front":                                           # A125 (the crawl rung): a prone child's hands lie under and
            ahead = -unit(np.r_[ch.len_axis[:2], 0.0])[:2]                   # beside it; the toy goes before its face, CRAWL_AHEAD_M past
            xy = np.asarray(ch.eyes[:2], float) + ahead * (CRAWL_AHEAD_M + float(self.lesson_dist))   # its eyes and the lesson's distance
        else:                                                               # (tummy time: a person puts the toy just out of reach ahead)
            xy = np.asarray(a["info"]["put_xy"], float) if a["info"].get("put_xy") else self._put_xy()   # (C168: the lure's place)
        if toy not in self.holding.values() and a["info"].get("refetch", 0) < 2:   # C138: a toy set aside on her way here (both hands
            a["info"]["refetch"] = a["info"].get("refetch", 0) + 1                 # full, a toy where she kneels: _plan_clear_here) is
            a["info"]["carry_ok"] = toy                                             # picked up again and the put planned anew
            return [dict(type="plan", what="fetch", args=dict(toy=toy)), dict(type="plan", what="put_near", args=dict(toy=toy))]
        sd, swap = self._giving(toy, np.r_[xy, 0.0])
        if not a.get("re_near") and not self._reachable(sd, dict(k="floor", xy=_lst(xy))):
            # A133 (room b's lesson): the put's place is read from the child's near hand where she kneels NOW, not where she planned
            # from; when her hand does not reach it from here (a sitting child's far side, 45 to 67 cm off before the release check
            # refused it) she kneels again where it does (the reach need), once
            a["re_near"] = True                                             # (a child that lay down on its front since the plan: her
            need = f"reach:{xy[0]:.3f},{xy[1]:.3f}"                         # kneel at its head, the crawl rung's, A125)
            near = self._near(a, where="head", offs=PUT_HEAD_OFFS, need=need) if ch.posture == "front" else self._near(a, need=need)
            return near + [dict(type="plan", what="put_near", args=dict(toy=toy))]
        return swap + self._put_phases(sd, xy, check=True)

    def _act_clear(self, a, t):
        toy = self._toy(t)
        return self._fetch(a, toy) + [dict(type="plan", what="set_aside", args=dict(toy=toy))]

    def _plan_set_aside(self, a, toy):
        sds = [s_ for s_, v in self.holding.items() if v == toy]
        sd = sds[0]
        b = self.base
        fwd = np.array([math.cos(b["yaw"]), math.sin(b["yaw"])]); left = np.array([-fwd[1], fwd[0]])
        at = np.asarray(b["at"], float)
        for sg in (1, -1):
            q = at + left * sg * 0.45 - fwd * 0.05
            if self._in_plan(q) and self.plan.dist[self.plan.cell(q)] > 0.1 and self.child.clearance_xy(q) > 0.3:
                return self._put_phases(sd, q)
        raise Refuse(f"nowhere to set the {toy} aside")

    def _act_open_hand(self, a, t):
        return self._near(a) + [dict(type="plan", what="open_hand", args=dict(target=t))]

    def _plan_open_hand(self, a, target=None):
        """her open hand, palm up, held out 10 cm beside the child's near hand toward her (the give ask's level 0, 4.10)"""
        ch = self.child
        her = np.asarray(self.base["at"], float)
        cs = min("LR", key=lambda x: float(np.linalg.norm(ch.grasp[x][:2] - her)))
        to_her = unit(np.r_[her - ch.grasp[cs][:2], 0.0])
        sd = self._near_hand(ch.grasp[cs])
        if self.holding[sd] is not None:
            sd = "L" if sd == "R" else "R"
        spec = dict(k="open", at=_lst(ch.grasp[cs]), off=_lst(to_her * 0.10 + np.array([0, 0, -0.02])))
        return [dict(type="reach", hands={sd: spec}, shape={sd: dict(curl=.15, thumb=.2, index=None)}, stay=True),
                dict(type="hold_out", n=K.OFFER_HOLD_TICKS), dict(type="relax", sides=sd)]

    # ---- guiding, the roll's ladder, the pull-to-sit, the prop, the turn (A7-A10)
    def _act_guide(self, a, t):
        return self._near(a, where=self._guide_side(t)) + [dict(type="plan", what="guide", args=dict(which=t))]

    def _guide_side(self, t):
        if t in ("far_arm",):
            return None                                                     # beside its chest on the side it faces: the far arm is across
        if t in ("L", "R"):
            return t
        return None

    def _plan_guide(self, a, which=None):
        ch = self.child
        her = np.asarray(self.base["at"], float)
        near = min("LR", key=lambda x: float(np.linalg.norm(ch.grasp[x][:2] - her)))
        far = "R" if near == "L" else "L"
        cs = far if which == "far_arm" else (which if which in ("L", "R") else near)
        side = "left" if cs == "L" else "right"
        body = self.m.body(f"{side}_elbow_link").id
        to_her = unit(np.r_[her - ch.torso[:2], 0.0])
        if which == "far_arm":
            dirv = unit(to_her + np.array([0, 0, 0.6])); cap_max = K.ROLL_ARM_N; dist = K.ROLL_ARM_M
        else:
            dirv = np.array([0, 0, 1.0]); cap_max = K.CAP_ONE; dist = K.GUIDE_RAISE_M
        sd = self._near_hand(self.d.xpos[body])
        if self.holding[sd] is not None:
            sd = "L" if sd == "R" else "R"
        ctl = dict(dir=_lst(dirv), dist=dist, limb="arm_l" if cs == "L" else "arm_r", max=cap_max)
        return self._hold_phases(a, sd, body, "guide", cap=cap_max, local=FOREARM_TOP, normal=UP_LOCAL, ctl=ctl) + \
            [dict(type="plan", what="let_go", args=dict(names=[f"guide_{sd}"])), dict(type="relax", sides=sd)]

    def _act_knee_over(self, a, t):
        return self._near(a, alongs=K.HIPS_ALONG_M) + [dict(type="plan", what="knee", args={})]    # beside its hips (A8)

    def _plan_knee(self, a):
        ch = self.child
        her = np.asarray(self.base["at"], float)
        near = min("LR", key=lambda x: float(np.linalg.norm(ch.body[f"{'left' if x == 'L' else 'right'}_knee_link"][0][:2] - her)))
        far = "R" if near == "L" else "L"
        body = self.m.body(f"{'left' if far == 'L' else 'right'}_knee_link").id
        to_her = unit(np.r_[her - ch.pelvis[:2], 0.0])
        ctl = dict(dir=_lst(unit(to_her + np.array([0, 0, 1.0]))), dist=K.KNEE_OVER_M, limb="leg_l" if far == "L" else "leg_r",
                   max=K.ROLL_KNEE_N)
        sd = self._near_hand(self.d.xpos[body])
        return self._hold_phases(a, sd, body, "knee", cap=K.ROLL_KNEE_N, ctl=ctl) + \
            [dict(type="plan", what="let_go", args=dict(names=[f"knee_{sd}"])), dict(type="relax", sides=sd)]

    def _act_pull_to_sit(self, a, t):
        """A9: she kneels at its feet and takes both its forearms (from the spots at its feet from where one trunk of hers reaches
        both, PULL_FEET_M); refused, with the reason, where none does (C7)"""
        if self.child.posture != "back":
            raise Refuse("the pull-to-sit is from lying on its back (A9)")
        return self._near(a, where="pull", offs=K.PULL_FEET_M, need="pull") + [dict(type="plan", what="pull", args={})]   # A163: its feet, then its sides

    def _pull_pairing(self):
        """which of her hands takes which of its forearms so that one trunk of hers, kneeling where she is, reaches both (her palm
        on each forearm's top): the pairing, or None"""
        bodies = {cs: self.m.body(f"{'left' if cs == 'L' else 'right'}_elbow_link").id for cs in "LR"}
        for pair in ((("L", "L"), ("R", "R")), (("R", "L"), ("L", "R"))):
            tg = {}
            for sd, cs in pair:
                to, _, _, shape = self._hold_target(sd, bodies[cs], FOREARM_TOP, UP_LOCAL)
                tg[sd] = (*self._resolve_hand(to, sd), shape)
            if self._solve_trunk(tg, None)[3]:
                return pair
        return None

    def _gather_reach(self):
        """A169 (2026-10-02): from her tall kneel here, each forearm reachable by a hand of its own from a trunk pose of its own (the
        gather's requirement, A165: the nearer forearm by the hand nearer it, the other by the other hand). Life day 59's first
        pull-to-sit (tick 2,838,792) was refused before she went, 'no spot she can kneel at lets her do it (pull): her reach', because
        the spot check asked for one trunk reaching both forearms at once (the pairing) while the gather, built for the arms it cannot
        take at once, was never reached: a spot is good for the pull when she can take the forearms one at a time from it"""
        bodies = {cs: self.m.body(f"{'left' if cs == 'L' else 'right'}_elbow_link").id for cs in "LR"}
        order = sorted("LR", key=lambda cs: float(np.linalg.norm(self.d.xpos[bodies[cs]][:2] - np.asarray(self.base["at"], float))))
        used = set()
        for cs in order:
            sd = self._near_hand(self.d.xpos[bodies[cs]])
            if sd in used:
                sd = "L" if sd == "R" else "R"
            used.add(sd)
            to, _loc, _nl, shape = self._hold_target(sd, bodies[cs], FOREARM_TOP, UP_LOCAL)
            g, R = self._resolve_hand(to, sd)
            if float(np.linalg.norm(g[:2] - np.asarray(self.base["at"], float)[:2])) > 1.0:
                return False
            if not self._solve_trunk({sd: (g, R, shape)}, None, step=10)[3]:
                return False
        return True

    def _plan_pull(self, a):
        """the pull-to-sit (A9) from its feet: both its forearms held, pulled up and toward its feet (toward her), as a sit-up
        raises the trunk about the hips"""
        ch = self.child
        b = self.base
        if b["mode"] == "heels":                                            # she pulls kneeling tall (her reach, _need_ok's 'pull')
            fw = np.array([math.cos(b["yaw"]), math.sin(b["yaw"])])
            return [dict(type="kneel_down", at=_lst(np.asarray(b["at"], float) + fw * HEELS_BACK), yaw=b["yaw"], u0=3.0, u1=2.0),
                    dict(type="plan", what="pull", args={})]
        dirv = unit(np.array([0, 0, 1.0]) + ch.len_axis * 0.5)
        out = []
        bodies = {cs: self.m.body(f"{'left' if cs == 'L' else 'right'}_elbow_link").id for cs in "LR"}
        pairing = self._pull_pairing()
        if pairing is None:
            # A165 (2026-10-01): THE FOREARMS GATHERED ONE AT A TIME. Life day 56's second motor block (tick 2,708,768): the side spot
            # passed the reach check, she came, and at the forearms no one trunk of hers reached both (the child's arms lie wherever its
            # habits leave them, and move in the ticks she takes to come). A parent of a flailing infant takes one forearm, then the
            # other, brings them together over its chest, and pulls: each forearm is held by the hand nearer it from a trunk pose of its
            # own (a kept hold: the guide's path, then 'keep'), drawn toward the gathering point above its chest, and once both are
            # held the two holds become the pull's (kind 'pull', its controller from the start)
            if any(v is not None for v in self.holding.values()):
                raise Refuse("her hands are busy: the pull-to-sit takes both (A9, A165)")
            order = sorted("LR", key=lambda cs: float(np.linalg.norm(self.d.xpos[bodies[cs]][:2] - np.asarray(self.base["at"], float))))
            used = set()
            for cs in order:                                                 # the nearer forearm first
                sd = self._near_hand(self.d.xpos[bodies[cs]])
                if sd in used:
                    sd = "L" if sd == "R" else "R"
                used.add(sd)
                out += self._gather_phases(a, cs, sd, 0)
                out.append(dict(type="holds_wait", kind="gather"))
            out.append(dict(type="plan", what="pull_from_gather", args=dict(dir=_lst(dirv))))
            return out
        for sd, cs in pairing:
            body = bodies[cs]
            ctl = dict(dir=_lst(dirv))
            ph = self._hold_phases(a, sd, body, "pull", cap=0.0, brief=True, local=FOREARM_TOP, normal=UP_LOCAL, ctl=ctl,
                                   tall_if_needed=False)
            out += ph[:-1]
            out.append(dict(ph[-1], wait=False))
        out.append(dict(type="holds_wait", kind="pull"))
        out.append(dict(type="plan", what="let_go", args=dict(names=[f"pull_{s_}" for s_ in "LR"])))
        out.append(dict(type="relax", sides="LR"))
        return out

    def _ph_holds_wait(self, a, ph):
        hs = [h for h in self.holds if h.kind == ph["kind"]]
        if not hs:                                                          # her grip slipped before the act began (she is a body:
            cz = float(self.child.torso_R[2, 0])
            if ph["kind"] == "turn" and cz >= K.TURN_PAST_Z:                 # GRIP_TOL_M, HOLD_SLIP_M); A101: a turn whose hands
                if a is not None:                                           # slipped once it had rolled past its side is done (the far
                    a["why"] = "turned from its front past its side within her caps (A7, A101; her hands slipped as it rolled)"   # links
                return "done"                                               # roll up and over out of her palms there); C167: past it
            if ph["kind"] == "turn" and cz >= -0.5:                         # by TURN_PAST_Z, else it tips back (A143's stiff limbs): not done
                deg = math.degrees(math.acos(float(np.clip(-cz, -1.0, 1.0))))
                return (f"stopped: her hands slipped at its side (A7, A101; its chest turned {deg:.0f} deg from face down, a side is 90, "
                        f"done is past it): it lay back; the turn is asked again")
            return "her hands lost their hold on it before she began (it slipped from her grip)"
        for h in hs:
            h.ctl["go"] = True                                              # every hand of the act is on: it begins
        for h in hs:
            st = h.ctl.get("state", "run")
            if st.startswith("stopped"):
                return st
        if ph["kind"] == "gather" and all(h.ctl.get("state") == "keep" for h in hs):
            return "done"                                                   # A165: the forearm held and kept; the next phase follows
        if all(h.ctl.get("state") == "done" for h in hs):
            if a is not None and any(h.ctl.get("sat") for h in hs):
                a["why"] = "sat up with its own flexion (A9)"
            if a is not None and any(h.ctl.get("turned") for h in hs):
                a["why"] = "turned from its front past its side within her caps (A7, A101)"
            return "done"
        if a is not None:
            a["info"]["hold_peak"] = max([a["info"]["hold_peak"]] + [h.peak for h in hs])
            a["info"]["effort_peak"] = self.effort_peak
        return "run"

    def _act_prop(self, a, t):
        if self.child.trunk_deg > K.PROP_MAX_DEG:
            raise Refuse(f"the trunk is {self.child.trunk_deg:.0f} deg from vertical: she props only within 30 deg (A9)")
        return self._near(a) + [dict(type="plan", what="prop", args={})]

    def _plan_prop(self, a):
        if self.child.trunk_deg > K.PROP_MAX_DEG:
            raise Refuse(f"the trunk is {self.child.trunk_deg:.0f} deg from vertical: she props only within 30 deg (A9)")
        torso = self.m.body("torso_link").id
        her = np.r_[np.asarray(self.base["at"], float), 0.5]
        Rt = self.d.xmat[torso].reshape(3, 3)
        back = dict(local=[-0.075, 0.0, 0.22], normal=[-1.0, 0.0, 0.0]); front = dict(local=[0.085, 0.0, 0.22], normal=[1.0, 0.0, 0.0])
        pts = {k: self.d.xpos[torso] + Rt @ np.array(v["local"]) for k, v in (("back", back), ("front", front))}
        yaw = self.base["yaw"]; left = np.array([-math.sin(yaw), math.cos(yaw)])
        lat = {k: float((pts[k][:2] - her[:2]) @ left) for k in pts}
        order = sorted(("back", "front"), key=lambda k: -lat[k])             # the one more to her left takes her left hand
        out = []
        for sd, k in zip("LR", order):
            spec = back if k == "back" else front
            ph = self._hold_phases(a, sd, torso, "prop", cap=K.PROP_EASE[0] * K.CAP_TWO / 2, local=spec["local"], normal=spec["normal"],
                                   ctl={}, tall_if_needed=(sd == "L"))
            out += ph[:-1] + [dict(ph[-1], wait=False)]
        out.append(dict(type="holds_wait", kind="prop"))
        return out

    def _act_turn(self, a, t):
        if self.child.posture not in ("front", "side"):                     # A170: or on its side
            raise Refuse("the brief turn is for the child face down or on its side (A7, A170)")
        return [dict(type="plan", what="turn_approach", args={})]

    def _plan_turn_approach(self, a):
        """A136 (C98, C109), C98 fixed (2026-09-28): the turn's spot. At its HEAD first, both hands on the far shoulder and the far side
        of its torso, pulled across it (turn_shoulder); else beside its chest with the far shoulder and hip in reach (turn_both); else
        refused. The order was the other way round until C98's measurement on a rocking prone child (the babbler at rest 0.6, three
        seeds, twelve asks): from beside its chest the one-hand check passed and the two-hand reach then hung 30 to 49 cm short over
        its raised back (the trunk solve for both hands from one lean fails over a child up on its elbows), and where both grips did
        land, 200 N rolled it to 50 to 70 degrees and dragged it a hand's breadth toward her instead of over: 0 turns of 9 attempts,
        the same as day 10's 16 refusals. From its head the same child was turned onto its back in 3 of 4 asks (the fourth balanced
        on its side at 85 degrees and fell back within her 4 s). A still child turns from its head in under 400 ticks (parent 30)
        as from its side in under 300: the head end is the way that works for both, so it is her first way. (From a kneel beside a
        prone G1 the far shoulder and hip lay 0.93 to 0.98 m off for the still child of day 24, C109; A103 had tried a need for both
        grips from the chest side alone and dropped it)"""
        why = []
        for where, offs, need, mode in (("head", TURN_HEAD_OFFS, "turn_shoulder", "head"), (None, None, "turn_both", "side")):
            try:
                out = self._plan_approach(a, where=where, offs=offs, need=need)
            except Refuse as e:
                why.append(f"{need}: {str(e)[:70]}"); continue
            a["info"]["turn_mode"] = mode
            return out + [dict(type="plan", what="turn", args=dict(mode=mode))]
        raise Refuse("no spot she can kneel at puts a far grip of the turn in her reach (A136, C98: at its head for the shoulder and the "
                     "torso's far side, beside its chest for both): " + "; ".join(why))

    def _turn_targets(self, kinds=("shoulder", "hip")):
        """A101: the turn's two grips on the child's FAR shoulder and hip roll links (their far surfaces, her palms toward her):
        {hand: (target spec, held point and normal in the link's frame, hand shape, link)}. A136: from its head the far side of its
        torso stands in for the hip (kinds ("shoulder", "torso")): both hands within reach there, both pulling across it"""
        ch = self.child
        her = np.r_[np.asarray(self.base["at"], float), 0.0]
        toward = unit((her - ch.torso) * [1, 1, 0])
        near = "left" if float(ch.lat[:2] @ toward[:2]) > 0 else "right"
        far = "right" if near == "left" else "left"
        zs = {s_: float(self.d.xipos[self.m.body(f"{s_}_shoulder_roll_link").id][2]) for s_ in ("left", "right")}
        if ch.posture == "side" and max(zs.values()) - min(zs.values()) > K.SIDE_SHOULDER_DZ_M:
            # A170 (2026-10-02): A CHILD ON ITS SIDE IS TURNED ONTO ITS BACK: its UPPER shoulder (the higher of the two) and the upper
            # side of its torso, pushed toward its back (the chest's normal reversed, along the floor). Life day 60: the child rolled
            # onto its front and side and lay there 38% of the day; C216/C217's tummy time came due at tick 2,901,200 and the turn was
            # refused, 'the brief turn is for the child face down (A7)'. From its head the face-down rule took the far side as the
            # one its face points away from, which on a side-lying child is the LOWER shoulder, and pushed toward the face's side:
            # that rolls it onto its front. The turn's controller then runs from its side (past TURN_SIDE_Z already) over to its back
            far = max(zs, key=zs.get)                                        # (a rocking prone child reads 'side' for a tick with its
            toward = unit(-ch.torso_R[:, 0] * [1, 1, 0])                     # shoulders level: that one keeps the face-down grips)
        elif "torso" in kinds:                                               # A136: from its head "toward her" runs along it; the far
            toward = unit(-ch.lat * (1 if far == "left" else -1) * [1, 1, 0])   # side is the one her first hand reaches (face_side's other)
            far = "right" if ch.face_side() == "L" else "left"
            toward = unit((self.d.xipos[self.m.body(f"{'left' if far == 'right' else 'right'}_shoulder_roll_link").id]
                           - self.d.xipos[self.m.body(f"{far}_shoulder_roll_link").id]) * [1, 1, 0])
        out = {}
        bodies = {"shoulder": f"{far}_shoulder_roll_link", "hip": f"{far}_hip_roll_link", "torso": "torso_link"}
        for name in kinds:
            body = bodies[name]
            b = self.m.body(body).id
            sd = "R" if name == "shoulder" else "L"
            beyond = self.d.xipos[b] - toward * 0.60                        # a point beyond the child on its far side: the anchor's
            pt, n = self._anchor(b, prefer=beyond)                          # rays from there meet the link's far surface, its normal
            Rl = self.d.xmat[b].reshape(3, 3)                               # pointing away from her; her palm on it faces her
            local = Rl.T @ (pt - self.d.xpos[b]); normal = Rl.T @ unit(n)
            to, loc, nl, shape = self._hold_target(sd, b, _lst(local), _lst(normal))
            out[sd] = (to, loc, nl, shape, b, toward)
        return out

    def _plan_turn(self, a, mode="side"):
        """A101 (2026-09-26, C81): the turn from its front by its FAR shoulder and FAR hip, her hands reaching over its back onto their
        far surfaces (the palms toward her), both placed from ONE trunk pose, then pulled up and toward her together so the body rolls
        about its near edge onto its side and its back, as a person rolls a heavy child. Until A101 the near shoulder and hip were
        "lifted" from above, each hand placed by its own lean (the second lean pulled the first hand off its hold: `slips`), and a hand
        on top of a link cannot lift it: the spring moved nothing (0 N through every hold of `test_sim_parent` 16's turn), the act
        stopped at A7's 2 s and the child lay prone, on life day 3 for half a day (its eyes on the mat, no smile possible)"""
        above, onto, shapes, holds = {}, {}, {}, []
        kinds = ("shoulder", "hip") if mode == "side" else ("shoulder", "torso")   # A136: from its head, the far shoulder and the far
        for sd, (to, loc, nl, shape, b, toward) in self._turn_targets(kinds).items():   # side of its torso, both pulled ACROSS it
            goff = np.asarray(to["goff"], float)
            above[sd] = dict(to, goff=_lst(goff + nl * K.APPROACH_M)); onto[sd] = to; shapes[sd] = shape
            holds.append(dict(type="hold", name=f"turn_{sd}", side=sd, body=int(b), local=_lst(loc), normal=_lst(nl), goff=_lst(goff),
                              kind="turn", cap=0.0, brief=True, ctl=dict(toward=_lst(toward)), shape=shape, wait=False))
        out = []
        if not a["info"].get("turn_retry") and any(not self._reachable(sd, onto[sd]) for sd in onto):
            # A136 (C98): the child moved while she came (a rocking child crawls a hand's breadth a second): the grips she planned from
            # the spot are out of her reach from where she now kneels (the rig's misses: 30 to 74 cm short). Once, she approaches again
            # from where it lies now
            a["info"]["turn_retry"] = True
            return [dict(type="plan", what="turn_approach", args={})]
        if self.base["mode"] == "heels":                                    # up onto the tall kneel: the reach over its back needs it
            b_ = self.base; fw = np.array([math.cos(b_["yaw"]), math.sin(b_["yaw"])])
            out.append(dict(type="kneel_down", at=_lst(np.asarray(b_["at"], float) + fw * HEELS_BACK), yaw=b_["yaw"], u0=3.0, u1=2.0))
        out.append(dict(type="reach", hands=above, shape=shapes))           # both hands over their points, along the normals,
        out.append(dict(type="reach", hands=onto, shape=shapes, n=3))       # then onto them together (one trunk for both)
        out += holds
        out.append(dict(type="holds_wait", kind="turn"))
        out.append(dict(type="plan", what="let_go", args=dict(names=["turn_L", "turn_R"])))
        out.append(dict(type="relax", sides="LR"))
        return out

    # ---- her face where its eyes can reach (A3, A22, C34)
    def _act_lean_in(self, a, t):
        if self.child.posture == "front":                                 # A124 (C107): a prone child sees the floor: tummy time, her
            return self._near(a, where="head", offs=LIE_OFFS, need="lie") + [dict(type="plan", what="lie_in", args={})]   # face on it
        on_line = t == "child_line"                                       # A94: her face onto its line of sight (the smile she gives)
        need = "lean_line" if on_line else "lean"                          # A96: from a spot where a pose puts it ON the line (else, the
        return self._near(a, alongs=K.LEAN_ALONG_M, need=need) + [dict(type="plan", what="lean_in", args=dict(on_line=on_line))]   # periphery)

    def _lie_fit(self, T, yaw):
        """A124 (C107): the chest's extension (LIE_CHEST_UP, tried in turn) with which, lying on her front from the tall kneel at T
        facing yaw, her mouth lands LEAN_DIST_M from where a prone child's eyes will be when it lifts its head (LIE_HEAD_UP_M above
        them now: face down, its cameras look into the mat, C94, and see her only in a head-up), her face turned toward them within
        A1's limit, her body 3 cm clear of it (A4) and every joint inside its range; None when none does"""
        for cu in LIE_CHEST_UP:
            p = lie_pose(T, yaw, 1.0, cu)
            p.pos = p.pos + np.array([0.0, 0.0, self._floor_lift(p)])
            kin.look(p, self.child.eyes)
            if any(r.get("violations") for r in p.report.values() if isinstance(r, dict)):
                continue
            mouth, ffwd, centre = self.mouth_of(p)
            lifted = np.asarray(self.child.eyes, float) + np.array([0.0, 0.0, LIE_HEAD_UP_M])   # its eyes when it lifts its head
            lo, hi = K.LEAN_DIST_M                                          # (a prone G1's cameras look into the mat until it does,
            d = float(np.linalg.norm(mouth - lifted))                       # C94; her face waits where the lifted head will see it)
            if not (lo <= d <= hi) or float(ffwd @ unit(lifted - mouth)) < math.cos(math.radians(E_FACE_TURN_DEG())):
                continue
            if self._clearance(p) < K.CLEAR_M:
                continue
            return cu
        return None

    def _plan_lie_in(self, a):
        """A124 (C107, tummy time): from the tall kneel before a prone child's head (the approach's spot, need "lie") she lies down on
        her front, her face on the floor before its face, and looks at its eyes; already lying there, she only looks"""
        b = self.base
        if b["mode"] == "lying":
            return [dict(type="look_at", target="child_eyes")]
        yaw = float(b["yaw"]); fwd = np.array([math.cos(yaw), math.sin(yaw)])
        T = np.asarray(b["at"], float) + (fwd * HEELS_BACK if b["mode"] == "heels" else 0.0)
        cu = self._lie_fit(T, yaw)
        if cu is None:
            raise Refuse("no lying pose puts her face where its eyes can reach from here (A124, C107)")
        out = []
        if b["mode"] == "heels":
            out.append(dict(type="kneel_down", at=_lst(T), yaw=yaw, u0=3.0, u1=2.0))
        elif b["mode"] != "tall":
            raise Refuse(f"lying down begins from a kneel, not {b['mode']}")
        out.append(dict(type="lie", at=_lst(T), yaw=yaw, u0=0.0, u1=1.0, chest_up=float(cu)))
        out.append(dict(type="look_at", target="child_eyes"))
        return out

    def _plan_lean_in(self, a, on_line=False):
        sol = self.face_reach(on_line=on_line)
        if sol is None and on_line:
            sol = self.face_reach()                                       # no pose on its line from here: its periphery, as before
        a["info"]["face"] = sol
        if sol is None:
            raise Refuse("no pose inside human ranges puts her face where its eyes can reach from here (A22, C34)")
        out = []
        b = self.base
        if sol["mode"] != b["mode"]:
            fwd = np.array([math.cos(b["yaw"]), math.sin(b["yaw"])])
            if b["mode"] == "heels" and sol["mode"] == "tall":
                out.append(dict(type="kneel_down", at=_lst(np.asarray(b["at"], float) + fwd * HEELS_BACK), yaw=b["yaw"], u0=3.0, u1=2.0))
            elif b["mode"] == "tall" and sol["mode"] == "heels":
                out.append(dict(type="kneel_down", at=b["at"], yaw=b["yaw"], u0=2.0, u1=3.0))
        out.append(dict(type="lean", lean=sol["lean"], spine=sol["spine"], twist=sol["twist"]))
        out.append(dict(type="look_at", target="child_eyes"))
        return out

    def _gather_phases(self, a, cs, sd, tries):
        """A165/A166: the phases that take forearm cs with hand sd where the forearm lies NOW and draw it toward the gathering point
        above its chest (a kept hold); `tries` the gathers of this forearm so far (A166: a hand that did not arrive because the arm moved
        plans again from where the arm is, up to GATHER_RETRIES times)"""
        ch = self.child
        body = self.m.body(f"{'left' if cs == 'L' else 'right'}_elbow_link").id
        G = ch.torso + np.array([0, 0, 1.0]) * K.GATHER_UP_M                    # the gathering point: above its chest
        p0 = self.d.xpos[body] + self.d.xmat[body].reshape(3, 3) @ np.asarray(FOREARM_TOP, float)
        to_g = G - p0
        dist = float(np.linalg.norm(to_g))
        ctl = dict(dir=_lst(unit(to_g) if dist > 1e-6 else np.array([0, 0, 1.0])), dist=min(dist, K.GATHER_MAX_M),
                   limb="arm_l" if cs == "L" else "arm_r", max=K.CAP_ONE, gather_cs=cs, gather_tries=int(tries))
        return self._hold_phases(a, sd, body, "gather", cap=K.CAP_ONE, local=FOREARM_TOP, normal=UP_LOCAL, ctl=ctl)

    def _plan_pull_from_gather(self, a, dir=None):
        """A165: both forearms held (the gather's kept holds): they become the pull's holds and the pull runs as from its feet"""
        hs = [h for h in self.holds if h.kind == "gather"]
        if len(hs) < 2:
            raise Refuse("a forearm slipped from her hand before the pull began (A165)")
        for h in hs:
            cap0 = float(h.cap)                                              # the gather's holding force on this forearm
            h.kind = "pull"
            h.name = f"pull_{h.side}"
            h.ctl = dict(dir=list(dir))
            self._start_ctl(a, h)
            # A167 (2026-10-01): THE PULL TAKES OVER AT THE GATHER'S FORCE. The pull's controller ramps its cap from 0 (A9: 100 N/s to her
            # brief cap); handed two forearms held up over its chest, a cap of 0 let them fall from her hands before the ramp could carry
            # them (a forearm drops 40 cm in two ticks), and life day 57's second pull-to-sit (tick 2,757,977, 258 ticks in) ended 'her
            # hands lost their hold on it before she began (it slipped from her grip)'. The ramp now begins where the gather's hold stood
            # (both hands together: twice this hand's cap), and the pull is on at once
            c = h.ctl
            c["go"] = True
            c["ramp"] = min(K.CAP_TWO_BRIEF, 2.0 * cap0)
            c["go_t"] = int(math.ceil(c["ramp"] / (K.CAP_RAMP_NPS * TICK_S)))
            h.cap = c["ramp"] / 2
            h.next = h.point(self.d)
            if self.arms[h.side].get("hold") is not None:
                self.arms[h.side]["hold"] = h.name
        return [dict(type="holds_wait", kind="pull"),
                dict(type="plan", what="let_go", args=dict(names=[f"pull_{s_}" for s_ in "LR"])),
                dict(type="relax", sides="LR")]

    def _ph_look_at(self, a, ph):
        self.look = dict(target=ph["target"])
        return "next"

    def mouth_of(self, pose):
        """her mouth point (the face test's, body/sim/eyes.mouth_point) for a pose"""
        hp, hR = kin.fk(pose)["head"]
        return hp + hR @ MOUTH_LOCAL, hR[:, 0].copy(), hp + hR @ FACE_CENTRE

    def face_in_view(self, mouth, fwd, centre, gaze=None):
        """for each of the child's eyes: (reachable by its fovea, degrees off the fovea's current line, distance, turned deg): the
        mouth point where the fovea window's centre can be put (inside the image by half a window), at least FACE_MIN_M away"""
        ch, w = self.child, self.w
        gaze = w.gaze if gaze is None else gaze
        out = {}
        for sd in "LR":
            v = ch.cam_R[sd].T @ (mouth - ch.eye[sd])
            if v[2] >= 0:
                out[sd] = (False, 0.0, 0.0, 180.0); continue
            x = G.EYE_W / 2 + W_EYE_F() * v[0] / -v[2]; y = G.EYE_H / 2 - W_EYE_F() * v[1] / -v[2]
            half = 16 + 2
            inside = half <= x <= G.EYE_W - half and half <= y <= G.EYE_H - half
            yaw = gaze[0] + (gaze[2] / 2 if sd == "L" else -gaze[2] / 2)
            r = np.array([math.tan(yaw), math.tan(gaze[1]), -1.0]); r /= np.linalg.norm(r)
            to = v / np.linalg.norm(v)
            off = math.degrees(math.acos(float(np.clip(r @ to, -1, 1))))
            dist = float(np.linalg.norm(mouth - ch.eye[sd]))
            te = ch.eye[sd] - centre
            turn = math.degrees(math.acos(float(np.clip(fwd @ te / np.linalg.norm(te), -1, 1))))
            out[sd] = (bool(inside and dist >= K.FACE_MIN_M and turn <= E_FACE_TURN_DEG()), off, dist, turn)
        return out

    def face_reach(self, at=None, yaw=None, modes=("heels", "tall"), base_mode=None, on_line=False):
        """the kneeling trunk (mode, lean, spine, twist) at her spot that puts her mouth where the child's eyes can reach it with
        their foveae (A22): LEAN_DIST_M from its eyes (never nearer than FACE_MIN_M, A3), 15 deg or more off its fovea's current line
        (A3), her face turned within A1's 75 deg of the eye, her legs, trunk and head 3 cm clear of it (A4), her free hands too as
        she will hold them; the least bend first (A1: either eye counts; both eyes preferred at the same bend). None when none does
        (C34). base_mode: the mode `at` is given in (her base's own when omitted). on_line (A94): her mouth ON its fovea's current
        line instead, within FACE_ON_LINE_DEG, the en-face position a parent takes to smile (Stern 1974; Papousek and Papousek 1987)"""
        b = self.base
        at = np.asarray(b["at"] if at is None else at, float); yaw = b["yaw"] if yaw is None else yaw
        bm = b["mode"] if base_mode is None else base_mode
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        lo, hi = K.LEAN_DIST_M
        cands = sorted((lean + spine + abs(twist), lean, spine, twist, mode) for mode in modes for lean in range(0, K.LEAN_MAX_DEG + 1, 10)
                       for spine in (0, 15, 30, 45) for twist in (0, -15, 15, -30, 30))
        either = None
        for bend, lean, spine, twist, mode in cands:
            if either is not None and bend > either["bend"]:
                return either                                               # the least bend (A1: either eye counts)
            if mode == bm or bm not in ("heels", "tall"):
                at_m = at
            else:
                at_m = at + fwd * HEELS_BACK if mode == "tall" else at - fwd * HEELS_BACK
            p = P.kneel(at_m, yaw, mode, lean=lean, spine_flex=spine, twist=twist)
            p.pos = p.pos + np.array([0.0, 0.0, self._kneel_lift(at_m, yaw, mode)])
            kin.look(p, self.child.eyes)
            mouth, ffwd, centre = self.mouth_of(p)
            fv = self.face_in_view(mouth, ffwd, centre)
            ok = {s_: fv[s_][0] and (fv[s_][1] <= K.FACE_ON_LINE_DEG if on_line else fv[s_][1] >= K.FACE_OFF_LINE_DEG)
                  and lo <= fv[s_][2] <= hi for s_ in "LR"}
            if not any(ok.values()) or (either is not None and not all(ok.values())):
                continue
            if any(r.get("violations") for r in p.report.values() if isinstance(r, dict)):
                continue
            if self._clearance(p) < K.CLEAR_M:
                continue
            free = [sd for sd in "LR" if self.holding[sd] is None and not self._hold_on(sd)]
            if free and lean + spine >= K.SUPPORT_BEND_DEG:                 # her free hands on her thighs, as she will lean
                for sd in free:
                    self._place(p, sd, *self._resolve_hand(dict(k="thigh"), sd, p), dict(curl=.2, thumb=.3, index=None))
            if free and self._clearance(p, segs=tuple(f"arm_{sd}" for sd in free)) < K.CLEAR_M:
                continue
            sol = dict(mode=mode, lean=float(lean), spine=float(spine), twist=float(twist), at=_lst(at_m), both=all(ok.values()),
                       bend=bend, eyes={s_: [bool(ok[s_]), round(fv[s_][1], 1), round(fv[s_][2], 3), round(fv[s_][3], 1)] for s_ in "LR"})
            if sol["both"]:
                return sol
            either = sol                                                    # one eye: the least bend, unless both at the same bend
        return either

    # ---- gestures
    def _free_side(self, prefer="R"):
        for sd in (prefer, "L" if prefer == "R" else "R"):
            if self.holding[sd] is None and not self._hold_on(sd):
                return sd
        raise Refuse("both her hands are busy")

    def _act_wave(self, a, t, side=None):
        sd = self._free_side("R") if side is None else self._side_free(side)
        sg = kin.side_sign(sd)
        Rc = P.hand_R_palm(sd, [1.0, 0, 0], [0, 0, 1.0])
        to = dict(k="rel", seg="chest", grip=[0.12, sg * 0.30, 0.42], R=_lst(Rc))
        return [dict(type="reach", hands={sd: to}, shape={sd: dict(curl=.1, thumb=.2, index=None)}, solve=False, shake_rel=[0, sg, 0]),
                dict(type="wave", side=sd, n=10), dict(type="relax", sides=sd)]

    def _ph_wave(self, a, ph):
        """her open hand waved side to side by her head (2.5 Hz, 6 cm) for n ticks"""
        sd = ph["side"]
        arm = self.arms[sd]
        t = ph.get("t", 0)
        if arm["mode"] == "at":
            segs = kin.fk(self.scene.pose)
            arm["shake"] = _lst(segs["chest"][1][:, 1] * 3.0)
            arm["shake_t"] = t + 1
        return "done" if t + 1 >= ph["n"] else "run"

    def _act_do(self, a, t):
        """her body does the named act while she says its word (a verb's introduction; P3's templates.NEEDS ("act", kind)): the
        kinds in DOES are done, any other is refused (templates.showable() keeps its word waiting)"""
        if t == "wave":
            return self._act_wave(a, t)
        if t == "clap":
            for sd in "LR":
                if self.holding[sd] is not None or self._hold_on(sd):
                    raise Refuse("her hands are busy")
            out = []
            for k in range(3):
                for gap in (0.14, 0.035):
                    hands = {sd: dict(k="rel", seg="chest", grip=[0.32, kin.side_sign(sd) * gap, 0.02],
                                      R=_lst(P.hand_R_palm(sd, [0, -kin.side_sign(sd), 0], [0.3, 0, 1.0]))) for sd in "LR"}
                    out.append(dict(type="reach", hands=hands, shape={sd: dict(curl=.1, thumb=.2, index=None) for sd in "LR"}, solve=False, n=2))
            return out + [dict(type="relax", sides="LR")]
        if t == "stand":                                                    # she gets up (standing already, she is standing)
            return self._up_phases()
        if t == "walk":                                                     # up, and a few steps across the room and back to face it
            return self._up_phases() + [dict(type="plan", what="do_walk", args={})]
        if t in ("open_hand", "close_hand"):
            sd = self._free_side("R")
            sg = kin.side_sign(sd)
            R_up = P.hand_R_palm(sd, [0, 0, 1.0], [1.0, 0, 0])              # palm up, the fingers ahead of her, in her chest's frame
            to = dict(k="rel", seg="chest", grip=[0.36, sg * 0.14, 0.02], R=_lst(R_up))
            opened = dict(curl=.1, thumb=.25, index=None)
            if t == "open_hand":
                return [dict(type="reach", hands={sd: to}, shape={sd: opened}, solve=False), dict(type="wait", n=K.GESTURE_HOLD_TICKS),
                        dict(type="relax", sides=sd)]
            return [dict(type="reach", hands={sd: to}, shape={sd: opened}, solve=False),
                    dict(type="reach", hands={sd: to}, shape={sd: dict(curl=1.45, thumb=.9, index=None)}, solve=False, n=3),
                    dict(type="wait", n=K.GESTURE_HOLD_TICKS), dict(type="relax", sides=sd)]
        thing = a.get("thing")                                              # the toy the word's set is said of (P3's Act.thing): she
        if t in ("show", "pick_up") and thing is not None:                  # handles that toy and no other (templates.TOY_ACTS)
            toy = self._toy(thing)
            held = [sd for sd in "LR" if self.holding[sd] == toy]
            if t == "show":
                return self._shake_own(held[0]) if held else \
                    [dict(type="plan", what="fetch", args=dict(toy=toy)), dict(type="plan", what="shake_held", args=dict(toy=toy))]
            return [] if held else [dict(type="plan", what="fetch", args=dict(toy=toy))]
        if t == "show":                                                     # a toy shaken before her (the "shake" and "hold" words)
            held = [sd for sd in "LR" if self.holding[sd] is not None]
            if held:
                return self._shake_own(held[0])
            return [dict(type="plan", what="nearest_toy", args=dict(then="shake"))]
        if t == "pick_up":                                                  # the nearest toy that is not the child's, picked up
            return [dict(type="plan", what="nearest_toy", args=dict(then=None))]
        raise Refuse(f"her body does not show '{t}'")

    def _shake_own(self, sd):
        """the toy in her hand held up before her chest and shaken for its sound"""
        sg = kin.side_sign(sd)
        R_in = P.hand_R_palm(sd, [0, -sg, 0.0], [0, 0, 1.0])
        to = dict(k="rel", seg="chest", grip=[0.34, sg * 0.10, 0.18], R=_lst(R_in))
        return [dict(type="reach", hands={sd: to}, shape={sd: dict(curl=.9, thumb=.8, index=None)}, solve=False, stay=True,
                     shake=[0.0, 0.0, 1.0]),
                dict(type="shake", side=sd, n=6), dict(type="relax", sides=sd)]

    def _plan_nearest_toy(self, a, then=None):
        """the nearest toy that is not the child's (a toy it touches is its own: A4) and lies on the floor of the room, fetched"""
        her = np.asarray(self.base["at"], float)
        cands = sorted((float(np.linalg.norm(self.d.xpos[b][:2] - her)), k) for k, b in self.toys.items()
                       if k not in self.holding.values() and k not in NEVER_FETCHED      # (C125: never the bucket)
                       and self.d.xpos[b][2] < 1.0 and self._in_plan(self.d.xpos[b][:2]))
        for _, k in cands:
            if not self._child_has(k):
                out = [dict(type="plan", what="fetch", args=dict(toy=k))]
                if then == "shake":
                    out.append(dict(type="plan", what="shake_held", args=dict(toy=k)))
                a["info"]["toy"] = k
                return out
        raise Refuse("no toy she may take is in the room (the child has them, A4)")

    def _plan_shake_held(self, a, toy):
        sds = [sd for sd, v in self.holding.items() if v == toy]
        if not sds:
            raise Refuse(f"the {toy} is not in her hand")
        return self._shake_own(sds[0])

    def _plan_do_walk(self, a):
        """a few steps: to a free point about K.DO_WALK_M from where she stands (the first of eight ways round, from her facing, with a
        path on the floor that keeps A6's clearances), then turned to face the child"""
        at = np.asarray(self.base["at"], float); yaw = float(self.base["yaw"])
        ch = (self.child.torso[:2] + self.child.pelvis[:2]) / 2
        for k in range(8):
            ang = yaw + (math.pi / 4) * ((k + 1) // 2) * (1 if k % 2 else -1)
            q = at + K.DO_WALK_M * np.array([math.cos(ang), math.sin(ang)])
            if not self._in_plan(q) or self.plan.dist[self.plan.cell(q)] < K.BODY_R_M + K.CLEAR_FURNITURE_M or \
                    self.child.clearance_xy(q) < K.CLEAR_CHILD_M + K.BODY_R_M:
                continue
            face = math.atan2(ch[1] - q[1], ch[0] - q[0])
            try:
                return self._walk_phases(at, q, face, goal_r=0.3, again=dict(what="do_walk", args={}))
            except Refuse:
                continue
        raise Refuse("no room for her to walk a few steps here")

    def _side_free(self, side):
        sd = {"left": "L", "right": "R", "L": "L", "R": "R"}.get(str(side))
        if sd is None:
            raise Refuse(f"no such side: {side}")
        if self.holding[sd] is not None or self._hold_on(sd):
            raise Refuse(f"her {'left' if sd == 'L' else 'right'} hand is busy")
        return sd

    def _act_copy(self, a, t):
        """her own arm or hand makes the movement the child just made, mirrored as she faces it (P3's A52; target 'kind:side', the
        side hers): 'arm_raise' her arm raised above her shoulder and lowered, 'wave' a wave of that hand, 'shake' that hand
        shaken before her (a toy in it shaken for its sound), 'open_hand' that hand opened out before her, palm up. It asks nothing
        and earns nothing (her conduct's); a busy hand refuses it"""
        kind, _, side = str(t or "").partition(":")
        sd = {"left": "L", "right": "R", "L": "L", "R": "R"}.get(side)
        if sd is None or kind not in ("arm_raise", "wave", "shake", "open_hand"):
            raise Refuse(f"no such movement to copy: {t}")
        if kind == "shake" and self.holding[sd] is not None:
            return self._shake_own(sd)
        sd = self._side_free(sd)
        sg = kin.side_sign(sd)
        if kind == "wave":
            return self._act_wave(a, None, side=sd)
        if kind == "arm_raise":
            R_up = P.hand_R_palm(sd, [1.0, 0, 0], [0, 0, 1.0])
            to = dict(k="rel", seg="chest", grip=[0.10, sg * 0.24, 0.60], R=_lst(R_up))
            return [dict(type="reach", hands={sd: to}, shape={sd: dict(curl=.15, thumb=.2, index=None)}, solve=False),
                    dict(type="wait", n=K.GESTURE_HOLD_TICKS), dict(type="relax", sides=sd)]
        if kind == "shake":
            R_in = P.hand_R_palm(sd, [0, -sg, 0.0], [0, 0, 1.0])
            to = dict(k="rel", seg="chest", grip=[0.34, sg * 0.14, 0.14], R=_lst(R_in))
            return [dict(type="reach", hands={sd: to}, shape={sd: dict(curl=.6, thumb=.5, index=None)}, solve=False, stay=True,
                         shake=[0.0, 0.0, 1.0]),
                    dict(type="shake", side=sd, n=6), dict(type="relax", sides=sd)]
        R_up = P.hand_R_palm(sd, [0, 0, 1.0], [1.0, 0, 0])
        to = dict(k="rel", seg="chest", grip=[0.36, sg * 0.14, 0.02], R=_lst(R_up))
        return [dict(type="reach", hands={sd: to}, shape={sd: dict(curl=.1, thumb=.25, index=None)}, solve=False),
                dict(type="wait", n=K.GESTURE_HOLD_TICKS), dict(type="relax", sides=sd)]

    def _act_cover_face(self, a, t):
        hands = {}
        for sd in "LR":
            if self.holding[sd] is not None or self._hold_on(sd):
                raise Refuse("her hands are busy")
            sg = kin.side_sign(sd)
            hands[sd] = dict(k="rel", seg="head", grip=[0.085, sg * 0.030, float(kin.PUPIL_Z)], R=_lst(P.hand_R_palm(sd, [-1.0, 0, 0], [0, 0, 1.0])))
        return [dict(type="reach", hands=hands, shape={sd: dict(curl=.05, thumb=.1, index=None) for sd in "LR"}, solve=False, n=3, stay=True),
                dict(type="hold_out", n=K.OFFER_HOLD_TICKS)]

    def _act_reveal_face(self, a, t):
        hands = {}
        for sd in "LR":
            sg = kin.side_sign(sd)
            hands[sd] = dict(k="rel", seg="head", grip=[0.02, sg * 0.20, float(kin.PUPIL_Z) - 0.02], R=_lst(P.hand_R_palm(sd, [1.0, 0, 0], [0, 0, 1.0])))
        return [dict(type="reach", hands=hands, shape={sd: dict(curl=.05, thumb=.2, index=None) for sd in "LR"}, solve=False, n=2),
                dict(type="relax", sides="LR")]

    def _act_withdraw(self, a, t):
        seg = self.hits[-1][1] if self.hits else None
        if seg is not None and SEG_CHAIN.get(seg, "core") != "core":
            sd = SEG_CHAIN[seg][-1]
        else:
            her = np.asarray(self.base["at"], float)
            sd = min("LR", key=lambda x: self.child.clearance_xy(self._grip_now(x)[0][:2]))
        out = []
        if self._hold_on(sd):
            out += self._let_go_phases([h.name for h in self.holds if h.side == sd])
        g, R = self._grip_now(sd, plan=True)
        away = unit(np.r_[(g - self.child.torso)[:2], 0.0])
        to = dict(k="fixed", grip=_lst(g + away * K.WITHDRAW_M + np.array([0, 0, 0.05])), R=_lst(R))
        return out + [dict(type="reach", hands={sd: to}, shape={sd: self._shape_now(sd)}, solve=False, n=2), dict(type="relax", sides=sd)]
