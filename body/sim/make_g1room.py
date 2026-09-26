"""Writes body/sim/g1room.xml: the living room with the STOCK UNITREE G1 as the child (docs/SIM_DESIGN.md sections 3, 5 and
15; the G1 amendment of 2026-09-24; from the G1 prototype of the same day, a copy of make_livingroom_customchild.py with the custom child
taken out). The XML is generated: edit the numbers here and re-run, never the XML.

  THE CHILD   the stock Unitree G1 with the Dex3 hands (MuJoCo Menagerie unitree_g1/g1_with_hands.xml), INCLUDED
              UNCHANGED from body/sim/assets/unitree_g1/ (the owner's word: "keep it stock"): 43 position actuators, 34.4 kg,
              1.32 m, no neck (the head is part of torso_link), no camera, IMU sites imu_in_torso and imu_in_pelvis. Its
              senses (a stereo camera pair where the real head's RealSense D435 sits, two ear sites) are added at load time
              as cameras and sites only (g1scene.py: no geom, mass or joint), because a body in an included file cannot
              take new children in XML. Its servo gains are set at load by the body's servo law (body/sim/world.py), the
              file's kp 500 being Menagerie's placeholder ("needs tuning"). It is born lying on its back (g1scene.birth()).
  THE PARENT  a body (the lead's decision of 2026-09-25): 16 dynamic segments in one tree (parent_kin.py's skeleton, a free
              pelvis, ball joints and hinges: PARENT_JOINTS), each with de Leva's female mass and inertia at her body mass
              (parent_consts.py), driven by her motion (body/sim/parent_motion.py, body/sim/parent_body.py) through one actuator
              per joint axis whose force range is a woman's strength there (parent_actuators; the lead's structural decision of
              2026-09-25, A25b: her trunk carried by a capped support, her limbs dynamic within her strength); touches toys, the room, the floor and the child with soft shapes (solref 0.02 at contact priority 2); holds toys
              through welds that start switched off, and the child ONLY through capped springs (no weld on the G1, no contact
              exclusion: her hands and arms collide with it; SIM_DESIGN.md 4.1, 4.2, A25). Her face is
              built to a real face's proportions and photometry (parent_face.py, through parent_kin.py: one smooth head sheet
              with its eye openings, the moving lids, irises, lashes, brows and lips, which g1scene.Scene poses from the
              parent's feelings, the parent lane's parent_feel.py; its meshes, texture and rig written by build_assets at
              each build; the W1 verifier's third and fourth rounds).
  THE ROOM    as in make_livingroom_customchild.py, with the play mat enlarged for a 1.32 m body (2.8 x 2.0 m) and the toys placed
              around the G1.

Units: RADIANS in this file (the G1's compiler says angle="radian", and MuJoCo applies the last compiler element to
the whole model), so every euler below is converted from degrees when the XML is written. The G1 comes first in the
world so its "stand" keyframe addresses its own qpos. The room's geoms take their contact defaults from the class
"room", so the G1's own defaults are untouched. Every path in the XML is relative to body/sim/ (the G1's file and its
mesh folder, the textures), so the scene loads from any checkout.

THE SOLVER (the G1 study, 2026-09-24): the elliptic friction cone, multi-point CCD off, and impratio 10. With the pyramidal cone a block
squeezed in a Dex3 hand stopped MuJoCo 3.9 ("FactorizeHessian: rank-deficient sparse Hessian"); with multi-point CCD on,
8 of 360 grasp trials blew up to NaN about 0.1 s into the fingers' closing and MuJoCo silently reset them; elliptic with
multi-point CCD off gave 0 of 360 and a clean babble. IMPRATIO 10 IS CHOSEN BY THE PHYSICS OF THESE CONTACTS AND MUJOCO'S GUIDANCE
(the owner's decision of 2026-09-24, made for him in the W1 verifier's third round; never by pain rates or by which toys can be
held; tools/sim_friction.py, see the option's comment): rubber-soled fingertips on plastic toys, toys and the body on a foam mat,
and motor housings on housings are all contacts that hold without a slide below their sliding force; MuJoCo's soft contacts creep
there by design, and MuJoCo 3.9's documentation names the remedy (elliptic cones with a large impratio and the Newton solver).
MuJoCo's auto-reset is off (A18), so a bad state is never silently replaced by the start pose: the world finds it and stops the tick.

Collision bits: world 1 | 16 (walls, furniture), the floor and the mat 2 (with conaffinity 1: the G1 rests on them); the G1 keeps
its own contype 1 / conaffinity 1 (self-collision on, as shipped); toys 4 (conaffinity 1 | 2 | 4 | 8); the parent 8, with
conaffinity 2 | 16, so she touches the room, the floor and the toys, never the G1 (A25c) and never herself, her feet, knees and shins on the
floor always (A25b: her trunk is carried, so no gait of hers is ever carried through the floor). The world's geoms and the toys take contact priority 2
(WORLD_PRIORITY; 5.1, A21, C26), so their friction and softness decide every contact with the G1 (the stock feet's own
priority 1 and friction 0.6 included) and nothing on the G1 is changed. Their frictions are still the prototype's 1.0: the
real surfaces' values are C26's, open (the W1 fix's report). Run: python3 body/sim/make_g1room.py  (writes
g1room.xml and textures/room_*.png, the living room's own two textures, byte for byte the same)."""
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
G1_DIR = Path("assets/unitree_g1")             # relative to body/sim/, where g1room.xml is written
G1_XML = G1_DIR / "g1_with_hands.xml"          # included unchanged
import parent_kin as kin  # noqa: E402
import parent_consts as PK  # noqa: E402  (her body's masses, strength and joints: her constants file)

# ------------------------------------------------------------------ collision classes
# The world's geoms (the floor, walls, mat and furniture) and the toys take contact priority 2 (5.1, A21, C26): where one of them
# touches the G1 (whose feet carry priority 1 in the stock file, its other shapes 0), MuJoCo takes the contact's friction, condim
# and softness from the world's geom alone, so the world's surfaces decide how the stock robot meets them and the G1's file and
# geoms stay untouched. Two priority-2 geoms (a toy on the floor) combine as MuJoCo does (the larger friction; mixed softness).
WORLD_PRIORITY = 2
# The world's solids carry bits 1 and 16 (A25c, the lead's decision of 2026-09-25): bit 1 so the G1 (conaffinity 1) and the toys
# (conaffinity 15) meet them as before; bit 16 so the parent (conaffinity 2 | 16) meets them WITHOUT meeting the G1, whose
# contype is bit 1 alone.
WORLD = f'contype="17" conaffinity="0" priority="{WORLD_PRIORITY}"'
# THE FLOOR AND THE MAT (what she stands, kneels and walks on) carry their own bit, 2 (with conaffinity 1, so the G1 still rests on
# them): her body touches them through her conaffinity's bit 2, always, so her feet, knees and shins meet the floor by contact
# (the lead's decisions of 2026-09-25: she is a body, and her trunk is carried, A25b)
FLOOR = f'contype="2" conaffinity="1" priority="{WORLD_PRIORITY}"'
TOY = f'contype="4" conaffinity="15" priority="{WORLD_PRIORITY}"'   # the toys touch the floor's bit 2 as well
# the parent's collision shapes: MuJoCo's default contact time constant (solref 0.02, the G1's own), at contact priority 2 like the
# world's, so hers is every contact's with the G1 (priority 0, its feet 1) and nothing on the G1 changes (A21's mechanism;
# body/sim/parent_consts.SOFT_*). A4's soft parent (0.05) softened a kinematic body's infinite mass; her body now gives by itself: at
# 0.05 her hand's shapes sank 9 mm into the child landing on it at 26 N, and a babbling foot sank 39 mm into her kneeling thigh at
# 1.6 kN (10 ms mean) where at 0.02 the worst of the same runs was 0.9 kN (the physical build, 2026-09-25).
# She is a body that passes NO CONTACT TO THE CHILD (A25c, the lead's structural decision of 2026-09-25, after three physical
# rounds in which her kneeling thigh, her shins and her hands met the babbling G1 at up to 1.6 kN): her shapes touch the room
# (bit 16), the floor (bit 2) and the toys (their conaffinity has her bit 8), never the G1 (contype 1) and never each other.
# Every force she puts on the child is one of her capped springs (a hold, a touch, a turn: parent_motion's holds, applied at
# the held point within its cap) or a toy she holds or sets down; the child's kick passes through her leg (her planner keeps her
# knees outside its leg sweep, A3), which it does not feel.
PARENT = f'contype="8" conaffinity="18" priority="2" solref="{PK.SOFT_SOLREF[0]:g} {PK.SOFT_SOLREF[1]:g}"'
PARENT_LIMB = PARENT
PARENT_HAND = f'contype="8" conaffinity="18" priority="2" solref="{PK.HAND_SOLREF[0]:g} {PK.HAND_SOLREF[1]:g}"'   # her hand's capsule
DECOR = 'contype="0" conaffinity="0"'
HAND_TOUCH = f'contype="0" conaffinity="18" priority="2" solref="{PK.HAND_SOLREF[0]:g} {PK.HAND_SOLREF[1]:g}"'   # her palm, fingers and thumb: they touch
# the room and the floor, never the G1 (A25c: her touch on the child is her capped spring's) and never a toy (a toy's contype is
# 4), soft as the rest of her; compiled collidable so MuJoCo's
# midphase keeps them (a geom compiled with no collision bits is never tested, whatever its bits later: the W2 verifier's third
# finding, her fingers passing through a babbling limb); her motion switches them off with her hand's proxy while she holds
VIS0 = DECOR + ' density="0"'          # massless and seen only (on dynamic bodies)

MAT_CENTER = np.array([0.0, -0.60])
MAT_HX, MAT_HY, MAT_T = 1.40, 1.00, .012   # a 2.8 x 2.0 m foam mat (7 x 5 tiles of 0.4 m), 1.2 cm thick: about 2.1 body
                                           # lengths, as the 1.6 m mat was for the 0.75 m child
ROOM_X, ROOM_Y, ROOM_H = 2.6, 2.3, 2.6

# ------------------------------------------------------------------ the light (5.1, 5.4; the W1 verifier's fifth finding)
# THE ROOM'S LIGHTS are its only light: a sun through the window, a key spot and a fill (LIGHTS). There is NO HEADLIGHT: MuJoCo's
# headlight is a lamp at the viewing camera, which for the child's eyes would be a lamp shining from its own head, one the G1
# does not have, so its room would never be dark at night (the owner's decision 3). What a lamp at the eye stood in for is the
# room's own indirect light: the light the walls, floor and ceiling give back, which reaches a face bent over a child from
# below and every side. MuJoCo's renderer computes no bounced light, so each light carries it as its ambient term, ROOM_INDIRECT
# of its own diffuse (about the prototype's headlight ambient, 0.32 over the three lights' 0.94, carried onto the lights that
# make it: world, ours). It goes where its light goes: a light darkened takes its indirect light with it, and the day's light (W5)
# scales it with the diffuse (W5: darken the lights' terms, never switch every light off, which makes MuJoCo draw the scene
# unlit; the emissive surfaces still glow). The room's cameras for people (room, mat, top) see the same light.
ROOM_INDIRECT = 1 / 3
LIGHTS = (('sun', 'type="directional" pos="-2 -.1 3" dir=".62 .18 -.76" castshadow="true"', (.30, .285, .25), (.08, .08, .08)),
          ('key', 'type="spot" pos="1.6 -2.0 2.3" dir="-.5 .45 -.62" cutoff="45" exponent="2" castshadow="true" bulbradius=".2"',
           (.42, .41, .40), (.12, .12, .12)),
          ('fill', 'type="directional" pos="0 0 2.5" dir="-.2 .5 -.84" castshadow="false"', (.22, .22, .23), (0, 0, 0)))


def lights_xml():
    return "\n    ".join(f'<light name="{n}" {a} diffuse="{f(*dif)}" ambient="{f(*(ROOM_INDIRECT * np.array(dif)))}" specular="{f(*spe)}"/>'
                          for n, a, dif, spe in LIGHTS)

# her face's calibrated albedos (tools/sim_face_photometry.py --calibrate; Russell, Kramer and Jones 2017, under a studio's frontal
# light with her own shading in): the eyes by the iris's lightness (the prototype's brown, its hue kept), the brows and lips as her
# hair's and the prototype's lip colour over her skin at the found share (the hue kept, the strength found). Not calibrated, ours:
# the sclera's off-white and the lash line at half her hair's colour over her skin (sparse lashes)
SCLERA_RGB = ".88 .86 .82 1"      # a sclera's off-white (ours; not calibrated: the eyes' strength is the iris's)
LASH_RGB = ".55 .405 .32 1"        # the lash line: her hair's colour over her skin at half (sparse lashes; ours)
BROW_RGB = ".6316 .4721 .3779 1"
LIPS_RGB = ".7642 .5258 .4506 1"
IRIS_RGB = ".3137 .1881 .1045 1"   # a medium brown iris: the prototype's brown, its lightness found
MAT = dict(
    # the parent: skin, a teal sweater, denim trousers, white shoes, dark brown hair; face features
    skin=dict(rgba=".86 .66 .54 1", specular=".12", shininess=".25"),
    skin_d=dict(rgba=".80 .58 .47 1", specular=".1", shininess=".25"),
    hair=dict(rgba=".24 .15 .10 1", specular=".3", shininess=".6"),
    sweater=dict(rgba=".20 .56 .62 1", specular=".08", shininess=".15"),
    sweater_d=dict(rgba=".15 .45 .50 1", specular=".08", shininess=".15"),
    denim=dict(rgba=".22 .27 .38 1", specular=".1", shininess=".2"),
    shoe=dict(rgba=".95 .95 .94 1", specular=".2", shininess=".4"),
    sole=dict(rgba=".62 .60 .58 1", specular=".1", shininess=".2"),
    # HER FACE (body/sim/parent_face.py; the W1 verifier's fourth round). The sheet's skin carries her shade of the room's indirect
    # light as a texture (computed from her geometry: parent_face.ao_texture); the eye's parts, the lids and the caruncle carry
    # theirs in their albedo, baked when written (face_shaded below). The sclera, the brows and the lips are each at the albedo
    # whose CIE L* contrast against the skin around it is a young woman's under a studio's frontal light, her own shading in
    # (Russell, Kramer and Jones 2017, table 2, no cosmetics: the eyes 0.152, the brows 0.126, the lips 0.092), found by
    # tools/sim_face_photometry.py --calibrate (the sclera's grey; the brows as her hair's colour over her skin; the lips as the
    # prototype's lip colour over her skin: each keeps its hue, only its strength is found) and checked by body/tests/
    # test_sim_eyes.py eyes 13. No drawn highlight on the eyes (the W1 verifier's specks): the corneas' own specular gives the
    # room's lamps' reflections, dark at night.
    skin_face=dict(rgba=".86 .66 .54 1", texture="parent_face_ao", specular=".08", shininess=".2"),
    sclera=dict(rgba=SCLERA_RGB, specular=".3", shininess=".7"),
    iris=dict(rgba=IRIS_RGB, specular=".35", shininess=".85"),
    pupil=dict(rgba=".03 .03 .03 1", specular=".35", shininess=".85"),
    lash=dict(rgba=LASH_RGB, specular=".1", shininess=".3"),
    lid=dict(rgba=".86 .66 .54 1", specular=".08", shininess=".2"),         # her skin (the lids' shade is baked in when written)
    caruncle=dict(rgba=".78 .50 .47 1", specular=".25", shininess=".6"),
    fornix=dict(rgba=".52 .30 .29 1", specular=".1", shininess=".3"),
    lips=dict(rgba=LIPS_RGB, specular=".15", shininess=".4"),
    mouth_in=dict(rgba=".30 .06 .08 1", specular=".1", shininess=".2"),
    teeth=dict(rgba=".97 .96 .93 1", specular=".2", shininess=".4"),
    brow=dict(rgba=BROW_RGB, specular=".05", shininess=".2"),
    belt=dict(rgba=".30 .20 .13 1", specular=".1", shininess=".2"),
    # the room
    floor=dict(texture="floor_oak", texrepeat="4 2.2", specular=".12", shininess=".35", reflectance=".015"),
    wall=dict(rgba=".93 .91 .87 1", specular="0", shininess="0"),
    wall_accent=dict(rgba=".74 .80 .76 1", specular="0", shininess="0"),
    ceiling=dict(rgba=".97 .97 .96 1", emission=".25", specular="0"),
    trim=dict(rgba=".97 .97 .96 1", specular=".1", shininess=".3"),
    tile_a=dict(rgba=".88 .85 .79 1", specular=".05", shininess=".2"),
    tile_b=dict(rgba=".78 .82 .78 1", specular=".05", shininess=".2"),
    mat_edge=dict(rgba=".78 .76 .72 1", specular=".05"),
    sofa=dict(rgba=".47 .55 .63 1", specular=".05", shininess=".1"),
    sofa_d=dict(rgba=".41 .48 .56 1", specular=".05", shininess=".1"),
    pillow_a=dict(rgba=".85 .52 .38 1", specular="0"),
    pillow_b=dict(rgba=".94 .91 .84 1", specular="0"),
    wood=dict(rgba=".80 .64 .46 1", specular=".25", shininess=".4"),
    wood_d=dict(rgba=".52 .38 .26 1", specular=".2", shininess=".4"),
    shelf=dict(rgba=".96 .95 .93 1", specular=".15", shininess=".3"),
    basket=dict(rgba=".72 .66 .58 1", specular="0"),
    basket2=dict(rgba=".58 .60 .62 1", specular="0"),
    book1=dict(rgba=".45 .55 .70 1", specular=".1"), book2=dict(rgba=".85 .78 .62 1", specular=".1"),
    book3=dict(rgba=".62 .45 .42 1", specular=".1"), book4=dict(rgba=".50 .62 .55 1", specular=".1"),
    sky=dict(texture="window_sky", emission=".85", specular="0"),
    curtain=dict(rgba=".91 .87 .80 1", specular="0"),
    lamp_shade=dict(rgba="1 .95 .85 1", emission=".6", specular="0"),
    lamp_metal=dict(rgba=".25 .25 .27 1", specular=".6", shininess=".8"),
    pot=dict(rgba=".86 .82 .76 1", specular=".2", shininess=".4"),
    leaf=dict(rgba=".30 .48 .30 1", specular=".15", shininess=".3"),
    art_a=dict(rgba=".93 .74 .58 1", specular="0"), art_b=dict(rgba=".66 .74 .84 1", specular="0"),
    art_c=dict(rgba=".97 .94 .88 1", specular="0"), art_d=dict(rgba=".78 .82 .70 1", specular="0"),
    hall=dict(rgba=".80 .78 .74 1", specular="0"),
    pad=dict(rgba=".97 .97 .95 1", emission=".35", specular=".2", shininess=".5"),
    pad_ring=dict(rgba=".75 .95 .90 1", emission=".9", specular="0"),
    # the toys: saturated, one dominant hue each
    t_red=dict(rgba=".92 .12 .10 1", specular=".5", shininess=".7"),
    t_orange=dict(rgba=".99 .45 .05 1", specular=".5", shininess=".7"),
    t_yellow=dict(rgba="1 .84 .08 1", specular=".4", shininess=".6"),
    t_green=dict(rgba=".10 .72 .24 1", specular=".5", shininess=".7"),
    t_green_d=dict(rgba=".05 .42 .13 1", specular=".2", shininess=".5"),
    t_cyan=dict(rgba=".02 .70 .86 1", specular=".5", shininess=".7"),
    t_blue=dict(rgba=".10 .30 .98 1", specular=".4", shininess=".6"),
    t_purple=dict(rgba=".56 .16 .86 1", specular=".5", shininess=".7"),
    t_pink=dict(rgba=".98 .22 .62 1", specular=".4", shininess=".6"),
    t_brown=dict(rgba=".66 .38 .16 1", specular=".1", shininess=".2"),
    t_brown_l=dict(rgba=".88 .70 .50 1", specular=".1", shininess=".2"),
    t_white=dict(rgba=".97 .97 .96 1", specular=".3", shininess=".5"),
    t_black=dict(rgba=".05 .05 .06 1", specular=".6", shininess=".9"),
    t_milk=dict(rgba=".95 .94 .90 1", specular=".5", shininess=".7"),
    t_collar=dict(rgba=".72 .78 .86 1", specular=".4", shininess=".6"),
    t_teat=dict(rgba=".93 .76 .52 1", specular=".2", shininess=".3"),
    t_beak=dict(rgba="1 .52 .08 1", specular=".3", shininess=".5"),
)


def f(*v):
    return " ".join(f"{x:.5g}" for x in v)


def lookat_xyaxes(pos, target):
    fwd = kin.unit(np.subtract(target, pos))
    x = kin.unit(np.cross(fwd, [0, 0, 1]))
    y = np.cross(x, fwd)
    return f(*x, *y)


def mirror_axis(a):
    return (-a[0], a[1], -a[2])


def mirror_pos(p):
    return (p[0], -p[1], p[2])


# ================================================================== THE PARENT (a body: 16 dynamic segments)
# HER HAIR (a dark brown bob with a bun, the film's) fitted to her head of [P]'s breadth and length: the cap over her head from the
# hairline (parent_face.hair_mesh), a side-swept fringe, the bob's sides over the ears to the jaw (behind her cheeks), the nape, the
# bun (ours: the style). Her ears ([F]: 59.6 mm long, inclined 17.5 deg back) under the sides (ours: where).
# (name, type, centre, euler deg, size, material), m
HAIR = [("fringe", "ellipsoid", (.090, -.012, .246), (-14, -30, 0), (.022, .056, .013), "hair"),
        ("side_L", "ellipsoid", (-.010, .072, .152), (0, -10, 0), (.048, .014, .080), "hair"),
        ("side_R", "ellipsoid", (-.010, -.072, .152), (0, -10, 0), (.048, .014, .080), "hair"),
        ("bun", "sphere", (-.108, 0, .214), (0, 0, 0), (.040, .040, .040), "hair"),
        ("nape", "ellipsoid", (-.046, 0, .128), (0, 0, 0), (.054, .066, .052), "hair")]
HEAD_SOLID = ((.005, 0, .180), (.078, .064, .086))     # inside her sheet at every point (ours; checked by body/tests/test_sim_eyes.py)
EARS = [("ear_L", "ellipsoid", (.000, .0705, .150), (0, -17.5, 0), (.015, .006, .0298), "skin_d"),
        ("ear_R", "ellipsoid", (.000, -.0705, .150), (0, -17.5, 0), (.015, .006, .0298), "skin_d")]


def hair_shapes():
    """her hair as (centre, semi-axes, rotation) for her shade's occupancy (parent_face.Occupancy)"""
    from scipy.spatial.transform import Rotation as Rot
    out = []
    for n, t, c, e, sz, mt in HAIR:
        out.append((np.array(c), np.array(sz), Rot.from_euler("XYZ", e, degrees=True).as_matrix()))
    out.append((np.array(HEAD_SOLID[0]), np.array(HEAD_SOLID[1]), np.eye(3)))
    return out


def parent_segment_geoms(seg):
    """The parent's geoms in a segment's own frame. Collision geoms (bit 8) are the solid shapes; her palm, fingers and thumb
    touch the room and the floor, never the G1 (HAND_TOUCH, A25c); faces and trim are seen only."""
    sd = seg[-1] if seg[-2] == "_" else ""
    sg = 1 if sd == "L" else -1
    m = (lambda p: p) if sd != "R" else mirror_pos
    P = f"parent_{seg}"
    g = []
    if seg == "pelvis":
        g += [f'<geom name="{P}" type="ellipsoid" pos="-.015 0 .035" size=".11 .165 .115" material="denim" {PARENT}/>',
              f'<geom type="ellipsoid" pos="-.005 0 .105" size=".113 .158 .03" material="belt" {DECOR}/>']
    elif seg == "abdomen":
        g += [f'<geom name="{P}" type="ellipsoid" pos="-.005 0 .075" size=".098 .145 .12" material="sweater" {PARENT}/>']
    elif seg == "chest":
        g += [f'<geom name="{P}" type="ellipsoid" pos=".005 0 .11" size=".1 .155 .145" material="sweater" {PARENT}/>',
              f'<geom type="sphere" pos="0 .15 .2" size=".045" material="sweater" {DECOR}/>',
              f'<geom type="sphere" pos="0 -.15 .2" size=".045" material="sweater" {DECOR}/>',
              f'<geom type="cylinder" pos=".01 0 .243" size=".056 .014" material="sweater_d" {DECOR}/>']
    elif seg == "head":
        g += ['<camera name="parent_portrait" pos=".62 0 .145" xyaxes="0 1 0 0 0 1" fovy="30"/>',
              f'<geom type="capsule" fromto=".008 0 -.01 .02 0 .08" size=".041" material="skin" {DECOR}/>',
              f'<geom name="{P}" type="ellipsoid" pos="{f(*HEAD_SOLID[0])}" size="{f(*HEAD_SOLID[1])}" rgba="0 0 0 0" group="3" {PARENT}/>',
              # her head's core (collision): inside her sheet everywhere; her face's and her hair's pieces are its surface
              f'<geom type="ellipsoid" pos="{f(*HEAD_SOLID[0])}" size="{f(*HEAD_SOLID[1])}" material="skin" {DECOR}/>',   # her head's solid
              # inside the sheet: it casts her head's shadow and fills behind the sheet where it is clipped under her hair
              ] + [f'<geom name="parent_{n}" type="{t}" pos="{f(*c)}" euler="{f(*e)}" size="{f(*sz)}" material="{mt}" {DECOR}/>'
                   for n, t, c, e, sz, mt in HAIR + EARS] \
            + kin.face.static_xml(DECOR, PARENT) + kin.face.moving_xml(DECOR, kin.FACE_NEUTRAL)   # her face (parent_face.py)
    elif seg.startswith("upper_arm"):
        g += [f'<geom name="{P}" type="capsule" fromto="0 0 -.01 0 0 -.285" size=".041" material="sweater" {PARENT}/>']
    elif seg.startswith("forearm"):
        g += [f'<geom name="{P}" type="capsule" fromto="0 0 0 0 0 -.185" size=".0385" material="sweater" {PARENT_LIMB}/>',
              f'<geom type="cylinder" pos="0 0 -.19" size=".037 .012" material="sweater_d" {DECOR}/>',
              f'<geom type="capsule" fromto="0 0 -.19 0 0 -.245" size=".025" material="skin" {DECOR}/>']
    elif seg.startswith("hand"):
        g += [f'<geom name="{P}_palm" type="ellipsoid" pos="0 0 -.052" size=".043 .0165 .05" material="skin" {HAND_TOUCH}/>',
              f'<geom name="{P}" type="capsule" fromto="0 0 -.035 0 0 -.115" size=".022" rgba="0 0 0 0" group="3" {PARENT_HAND}/>']
        for i in range(4):
            for j in range(2):
                g.append(f'<geom name="parent_f{i}{j}_{sd}" type="capsule" size="{kin.FINGER_R - .0006 * i} .02" material="skin" {HAND_TOUCH}/>')
        for j in range(2):
            g.append(f'<geom name="parent_t{j}_{sd}" type="capsule" size=".0098 .016" material="skin" {HAND_TOUCH}/>')
        g.append(f'<site name="{P}_grip" pos="{f(*m((0, -.045, -.10)))}" size=".008" group="4"/>')
    elif seg.startswith("thigh"):
        g += [f'<geom name="{P}" type="capsule" fromto="0 0 -.02 0 0 -.39" size=".07" material="denim" {PARENT}/>']
    elif seg.startswith("shin"):
        g += [f'<geom name="{P}" type="capsule" fromto="0 0 0 0 0 -.35" size=".053" material="denim" {PARENT}/>',
              f'<geom type="cylinder" pos="0 0 -.385" size=".038 .022" material="shoe" {DECOR}/>']
    elif seg.startswith("foot"):
        g += [f'<geom name="{P}" type="ellipsoid" pos=".058 0 -.036" size=".128 .047 .036" material="shoe" {PARENT}/>',
              f'<geom type="box" pos=".058 0 -.064" size=".124 .044 .006" material="sole" {DECOR}/>',
              f'<geom type="ellipsoid" pos=".13 0 -.02" size=".05 .03 .012" material="sole" {DECOR}/>']
    return "\n      ".join(g)


from g1scene import PARENT_BIRTH as PARENT_START  # noqa: E402  (where the scene's birth stands her: one place for both)

# HER JOINTS (the lead's decision of 2026-09-25: she is a body): each segment's joints to its parent segment, named for the scene
# (g1scene.PARENT_JOINTS reads them): a ball, or hinges (axis in the segment's own frame, range in radians). Every segment's frame is
# parent_kin's (aligned with her root in the rest pose, a limb hanging along -z from its proximal joint), so a ball's quaternion is
# parent_kin's local rotation of that segment and the elbow's two hinges compose as parent_kin.arm_ik composes them (the flexion
# about -y, then the forearm's own turn about its axis), the knee's as leg_ik's (about +y). The hinges' ranges are parent_kin's human
# ranges (LIM_DEG: the elbow 0-150, pronation +-90, the knee 0-160 deg), so the child can never bend them past a person's.
PARENT_JOINTS = {"abdomen": [("lumbar", "ball")], "chest": [("thorax", "ball")], "head": [("neck", "ball")],
                 "upper_arm": [("shoulder", "ball")],
                 "forearm": [("elbow", "hinge", (0, -1, 0), "elbow"), ("pron", "hinge", (0, 0, 1), "pronation")],
                 "hand": [("wrist", "ball")], "thigh": [("hip", "ball")], "shin": [("knee", "hinge", (0, 1, 0), "knee")],
                 "foot": [("ankle", "ball")]}


def _ellipsoid_inertia(m, a, b, c):
    return (m / 5 * (b * b + c * c), m / 5 * (a * a + c * c), m / 5 * (a * a + b * b))


TRUNK_SHAPE = {"pelvis": ((-.015, 0, .035), (.11, .165, .115)), "abdomen": ((-.005, 0, .075), (.098, .145, .12)),
               "chest": ((.005, 0, .11), (.1, .155, .145))}   # her trunk's collision ellipsoids (parent_segment_geoms)


def parent_inertial(seg):
    """her segment's mass, centre of mass and principal inertias (parent_consts: de Leva 1996's female segments at her body mass;
    her trunk's parts as solid ellipsoids of their shapes, ours)"""
    kind = seg[:-2] if seg[-2] == "_" else seg
    mass = PK.SEG_MASS_FRAC[kind] * PK.BODY_MASS_KG
    if kind in TRUNK_SHAPE:
        c, (a, b, cc) = TRUNK_SHAPE[kind]
        return mass, c, _ellipsoid_inertia(mass, a, b, cc)
    L = PK.SEG_LEN[kind]; fr = PK.SEG_COM_FRAC[kind]; rs, rt, rl = PK.SEG_GYR[kind]
    if kind == "head":                                                  # from the vertex (the top of her head's solid) down
        c = (.005, 0.0, L * (1 - fr)); I = (mass * (rs * L) ** 2, mass * (rt * L) ** 2, mass * (rl * L) ** 2)
    elif kind == "foot":                                                # from the heel along the foot (its long axis: x)
        c = (-.068 + fr * L, 0.0, -.036); I = (mass * (rl * L) ** 2, mass * (rt * L) ** 2, mass * (rs * L) ** 2)
    else:                                                               # a limb segment hanging along -z from its joint
        c = (0.0, 0.0, -fr * L); I = (mass * (rs * L) ** 2, mass * (rt * L) ** 2, mass * (rl * L) ** 2)
    a, b, cc = I                                                        # de Leva's hand radii (0.631, 0.454, 0.335 of its length)
    if b + cc < a:                                                      # break a rigid body's triangle inequality (their axes are
        I = (a, b, a - b)                                               # not a flat hand's): its long-axis inertia raised to the
    return mass, c, I                                                   # least a body allows (ours)


def parent_joints_xml(seg):
    kind = seg[:-2] if seg[-2] == "_" else seg
    sd = seg[-1] if seg[-2] == "_" else ""
    out = []
    for spec in PARENT_JOINTS.get(kind, ()):
        name = f"parent_{spec[0]}" + (f"_{sd}" if sd else "")
        if spec[1] == "ball":
            out.append(f'<joint name="{name}" type="ball"/>')
        else:
            lo, hi = kin.LIM_DEG[spec[3]]
            out.append(f'<joint name="{name}" type="hinge" axis="{f(*spec[2])}" range="{math.radians(lo):.6f} {math.radians(hi):.6f}"/>')
    return out


def parent_actuators():
    """HER MUSCLES (the lead's structural decision of 2026-09-25, A25b): one MuJoCo actuator per axis of each of her joints (a
    ball's three, in its own frame: gear along x, y, z; a hinge's one), named parent_<joint>_<axis> or parent_<joint>, in her drive's
    order (parent_body.BALLS then HINGES). Each is a motor with its own damping: force = ctrl - kv x the joint's velocity along its
    axis (gainprm 1; biasprm 0, 0, -kv, kv set at load from the joint's inertia: parent_body.BodyMap), and its ctrlrange and
    forcerange are her strength there, per direction (parent_consts.STRENGTH through parent_body._strength), so MuJoCo clamps her
    whole torque at that joint, her damping included, to a woman's. MuJoCo takes an actuator's damping implicitly under implicitfast
    while its force is inside its range, and explicitly (none) when the range clamps it (MuJoCo 3.9, measured: a clamped actuator
    adds no implicit damping)"""
    import parent_body as PB                                            # her joints' order and strength (one place for both)
    out = []
    for n, _seg in PB.BALLS:
        for k, ax in enumerate("xyz"):
            lo, hi = PB._strength(n)[k]
            gear = " ".join("1" if j == k else "0" for j in range(3))
            out.append(f'<general name="parent_{n}_{ax}" joint="parent_{n}" gear="{gear} 0 0 0" gainprm="1" biastype="affine" '
                       f'biasprm="0 0 0" ctrllimited="true" ctrlrange="{lo:g} {hi:g}" forcelimited="true" forcerange="{lo:g} {hi:g}"/>')
    for n, _seg in PB.HINGES:
        (lo, hi), = PB._strength(n)
        out.append(f'<general name="parent_{n}" joint="parent_{n}" gainprm="1" biastype="affine" biasprm="0 0 0" ctrllimited="true" '
                   f'ctrlrange="{lo:g} {hi:g}" forcelimited="true" forcerange="{lo:g} {hi:g}"/>')
    return out


def parent():
    """HER BODY (the lead's decision of 2026-09-25): her 16 segments as one tree of dynamic bodies, a free pelvis, standing where
    the scene's birth stands her in parent_kin's rest pose (every joint at zero); each with de Leva's female mass and inertia at
    her body mass (parent_inertial)"""
    pose = kin.Pose(PARENT_START["pos"], kin.rz(PARENT_START["yaw"]))
    segs = kin.fk(pose)
    kids = {s: [c for c in kin.SEGS if kin.SEG_PARENT[c] == s] for s in kin.SEGS}

    def body(s, ind):
        pad = " " * ind
        if kin.SEG_PARENT[s] is None:
            p, R = segs[s]
            head = f'{pad}<body name="parent_{s}" childclass="room" pos="{f(*p)}" quat="{f(*kin.mjquat(R))}">'
            joints = [f'<freejoint name="parent_root"/>']
        else:
            head = f'{pad}<body name="parent_{s}" pos="{f(*kin.OFFSET[s])}">'
            joints = parent_joints_xml(s)
        mass, c, I = parent_inertial(s)
        lines = [head] + [f"{pad}  {j}" for j in joints]
        lines.append(f'{pad}  <inertial pos="{f(*c)}" mass="{mass:.4f}" diaginertia="{I[0]:.6g} {I[1]:.6g} {I[2]:.6g}"/>')
        lines.append(f"{pad}  " + parent_segment_geoms(s).replace("\n      ", f"\n{pad}  "))
        for k in kids[s]:
            lines.append(body(k, ind + 2))
        lines.append(f"{pad}</body>")
        return "\n".join(lines)
    return ('\n    <!-- ======================= THE PARENT: a body, 16 dynamic segments in one tree (parent_motion.py drives her) '
            '======================= -->\n' + body("pelvis", 4))


# ================================================================== THE TOYS
# name: (position on the floor or mat (x, y, yaw deg), sound) -- the sound each makes is a clip of its own (not built here).
# Placed around the G1 lying on its back (pelvis near (-0.05, -0.62), head toward -x, its left side toward +y): the
# rattle and the block beside its hands, the cup above its left shoulder, the ball and the duck beside its knees, the
# car beyond its feet, the bear and the stacker beyond its head, the ring off to its right, the drum off the mat by the
# table.
TOYS = dict(
    ball=((.45, -1.12, 0), "a soft rubber bounce when it lands or is struck"),
    block=((-.26, -.02, 20), "a hollow wooden knock on contact"),
    duck=((.50, .04, -130), "a squeak when squeezed"),
    cup=((-.58, -.12, 30), "a plastic clink when it strikes something"),
    rattle=((.08, -.97, 60), "beads shaking, louder with speed"),
    car=((1.02, -.45, 200), "a wheel rattle while it moves"),
    bear=((-1.08, -.38, -40), "a soft bell inside when it moves"),
    stacker=((-.98, -1.28, 0), "the rings' plastic clack"),
    drum=((1.62, .32, 0), "a boom when hit on top"),
    ring=((.98, -1.34, 0), "a crinkle when handled"),
    bottle=((-.10, -.97, 0), "a soft slosh when shaken"),
)
# THE BOTTLE AND ITS DOCK (5.3, A17; built at S5a): the child's charger, a toy-sized baby bottle standing on its dock beside the
# lying G1's right hand, within its reach (the dock moves to the mat's corner only when the bottle's stage 3 begins, 4.7). Its
# three shapes are named bottle*, so each charges the child while it touches a palm (world.CHARGER_PREFIX), whoever holds it.
# Its size a small 4-ounce bottle's (about 56 mm across and 12 cm tall with its teat; ours, from memory) and its mass with milk
# 0.12 kg (ours); its collar and teat as a bottle's. DOCK is the pad's place: the nearest place beside the right hand (the side
# she does not kneel on when it lies flat, face_side "L", so her hands never work across it) that the lying arm reaches with
# 5 cm to spare, at least 15 cm from every other toy, where she can kneel to it (her planner's own spot, 0.45 m from it, clear of
# the child and the toys), searched at birth's pose in 2 cm and 10 degree steps (S5a: 0.26 m from the hand, 0.35 m from the
# shoulder, its reach 0.55; beside the left hand it stood where she kneels to attend it, and blocked her hands).
DOCK = (-.10, -.97)


# Each toy's size relative to the all-out maker's (made for the 12-month-old's mitten): every toy at its own size. The G1 study
# (m_grasp_g1.py, now tools/sim_grasp.py: a stock Dex3 hand at its stock gains, a hand-over into the palm 6 tries and a top grasp
# from a surface 3, per toy) found at impratio 1 the ball, duck, bear and drum held at no scale from 0.6 to 1.0 and the cup only at
# 0.8, so the cup was made 0.8. That was the soft contacts' creep. With impratio chosen by physics (10, the option's comment;
# tools/sim_grasp.py on the built room, 2026-09-24, none unstable): at 1 the ball 0 of 9, the block 9, the duck 0, the cup at 0.8
# 5 and at its own size 0, the rattle 5, the car 5, the bear 0, the stacker 3, the drum 0, the ring 6 of 6 hand-overs; at 10 the
# ball 9, the block 9, the duck 7, the cup at 0.8 9 and at its own size 9, the rattle 6 of 6 hand-overs (0 of 3 top grasps), the
# car 8, the bear 2, the stacker 9, the drum 0, the ring 6 of 6 hand-overs. (Stock gains closing to the range's end; the grasp
# reflex's small steps under the body's servo law press less, W4.) THE CUP STAYS AT 0.8 (the W1 verifier's fourth finding): its
# size is the owner's call (SIM_DESIGN.md B1), and the design uses B1's default until the owner says otherwise. B1'S PREMISE
# FOLLOWS FROM THE PHYSICS: the hand holds the cup at its own size, so the shrink no longer has the reason it was made for (a
# world fitted to the body with no need), and of the four toys B1 names only the bear and the drum are not held.
TOY_SCALE = dict(ball=1.0, block=1.0, duck=1.0, cup=0.8, rattle=1.0, car=1.0, bear=1.0, stacker=1.0, drum=1.0, ring=1.0, bottle=1.0)
TOY_MESHES = dict(cup=("cup_rim", "cup_handle"), drum=("drum_rim",), ring=("teether",),
                  stacker=("ring0", "ring1", "ring2", "ring3", "ring4"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
if "toy-scale" in ARGS:                  # for the grasp study only: every toy at one scale, written to --out
    TOY_SCALE = {k: float(ARGS["toy-scale"]) for k in TOY_SCALE}


def _scale_attrs(xml, s):
    """Scale a geom's size, pos, fromto by s and its mass by s^3 (a toy's shape about its own frame)."""
    def num(attr, fac):
        def f(mt):
            return f'{attr}="' + " ".join(f"{float(v) * fac:.5g}" for v in mt.group(1).split()) + '"'
        return f
    for attr in ("size", "pos", "fromto"):
        xml = re.sub(rf'\b{attr}="([^"]+)"', num(attr, s), xml)
    return re.sub(r'\bmass="([^"]+)"', num("mass", s ** 3), xml)


def mesh_scale(name, base):
    for k, names in TOY_MESHES.items():
        if name in names:
            return f(*(np.array(base) * TOY_SCALE[k]))
    return f(*base)


def toy_geoms(k):
    z0, g = _toy_geoms(k)
    sc = TOY_SCALE[k]
    return z0 * sc, [_scale_attrs(x, sc) for x in g]


def _toy_geoms(k):
    fr = 'friction="1 .01 .001"'
    if k == "ball":
        return .06, [f'<geom name="ball" type="sphere" size=".06" mass=".12" material="t_red" condim="6" friction="1 .01 .002" {TOY}/>',
                     f'<geom type="sphere" pos="0 0 0" size=".0602" material="t_red" {VIS0} rgba="1 1 1 0"/>']
    if k == "block":
        return .035, [f'<geom name="block" type="box" size=".035 .035 .035" mass=".09" material="t_blue" {fr} {TOY}/>',
                      f'<geom type="box" pos="0 0 .0352" size=".022 .022 .0005" material="t_white" {VIS0}/>',
                      f'<geom type="box" pos=".0352 0 0" size=".0005 .022 .022" material="t_white" {VIS0}/>',
                      f'<geom type="box" pos="0 -.0352 0" size=".022 .0005 .022" material="t_white" {VIS0}/>']
    if k == "duck":
        return .036, [f'<geom name="duck" type="ellipsoid" size=".056 .042 .036" mass=".06" material="t_yellow" {fr} {TOY}/>',
                      f'<geom name="duck_head" type="sphere" pos=".032 0 .044" size=".027" mass=".015" material="t_yellow" {fr} {TOY}/>',
                      f'<geom type="ellipsoid" pos=".062 0 .040" size=".017 .014 .0055" material="t_beak" {VIS0}/>',
                      f'<geom type="sphere" pos=".05 .0155 .053" size=".0045" material="t_black" {VIS0}/>',
                      f'<geom type="sphere" pos=".05 -.0155 .053" size=".0045" material="t_black" {VIS0}/>',
                      f'<geom type="ellipsoid" pos="-.052 0 .02" euler="0 -35 0" size=".02 .017 .009" material="t_yellow" {VIS0}/>']
    if k == "cup":
        return .045, [f'<geom name="cup" type="cylinder" size=".042 .045" mass=".06" material="t_green" condim="6" friction="1 .01 .002" {TOY}/>',
                      f'<geom type="cylinder" pos="0 0 .0452" size=".036 .0006" material="t_green_d" {VIS0}/>',
                      f'<geom type="mesh" mesh="cup_rim" pos="0 0 .044" material="t_green" {VIS0}/>',
                      f'<geom type="mesh" mesh="cup_handle" pos=".047 0 0" euler="90 0 0" material="t_green" {VIS0}/>']
    if k == "rattle":   # a handle and a ball head: the handle fits the child's fist
        return .026, [f'<geom name="rattle" type="capsule" fromto="-.06 0 0 .03 0 0" size=".012" mass=".04" material="t_purple" {fr} {TOY}/>',
                      f'<geom name="rattle_head" type="sphere" pos=".062 0 0" size=".026" mass=".04" material="t_purple" condim="6" friction="1 .01 .002" {TOY}/>',
                      f'<geom type="cylinder" pos=".032 0 0" zaxis="1 0 0" size=".018 .004" material="t_white" {VIS0}/>',
                      f'<geom type="sphere" pos="-.064 0 0" size=".017" material="t_white" {VIS0}/>']
    if k == "car":
        g = [f'<geom name="car" type="box" pos="0 0 .01" size=".07 .036 .022" mass=".10" material="t_orange" {fr} {TOY}/>',
             f'<geom type="ellipsoid" pos="-.008 0 .036" size=".042 .032 .022" material="t_orange" {VIS0}/>',
             f'<geom type="ellipsoid" pos="-.004 0 .04" size=".034 .0325 .014" material="t_cyan" {VIS0} rgba=".75 .92 1 1"/>']
        for x in (-.042, .042):
            for y in (-.036, .036):
                g.append(f'<geom type="cylinder" pos="{x} {y} -.006" zaxis="0 1 0" size=".022 .008" material="t_black" {TOY} mass=".005"/>')
        return .028, g
    if k == "bear":     # a plush bear sitting up
        return .062, [f'<geom name="bear" type="ellipsoid" pos="0 0 0" size=".05 .058 .062" mass=".09" material="t_brown" {fr} {TOY}/>',
                     f'<geom name="bear_head" type="sphere" pos=".005 0 .095" size=".046" mass=".05" material="t_brown" {fr} {TOY}/>',
                     f'<geom type="sphere" pos=".0 .034 .132" size=".017" material="t_brown" {VIS0}/>',
                     f'<geom type="sphere" pos=".0 -.034 .132" size=".017" material="t_brown" {VIS0}/>',
                     f'<geom type="ellipsoid" pos=".045 0 .084" size=".016 .022 .016" material="t_brown_l" {VIS0}/>',
                     f'<geom type="sphere" pos=".06 0 .09" size=".0065" material="t_black" {VIS0}/>',
                     f'<geom type="sphere" pos=".043 .016 .107" size=".0045" material="t_black" {VIS0}/>',
                     f'<geom type="sphere" pos=".043 -.016 .107" size=".0045" material="t_black" {VIS0}/>',
                     f'<geom type="ellipsoid" pos=".04 0 .0" size=".012 .036 .04" material="t_brown_l" {VIS0}/>',
                     f'<geom type="capsule" fromto=".02 .05 .02 .05 .07 -.03" size=".018" material="t_brown" {VIS0}/>',
                     f'<geom type="capsule" fromto=".02 -.05 .02 .05 -.07 -.03" size=".018" material="t_brown" {VIS0}/>',
                     f'<geom type="capsule" fromto=".02 .03 -.045 .08 .045 -.05" size=".02" mass=".01" material="t_brown" {TOY}/>',
                     f'<geom type="capsule" fromto=".02 -.03 -.045 .08 -.045 -.05" size=".02" mass=".01" material="t_brown" {TOY}/>']
    if k == "stacker":  # a ring stacker: base, post, five graded rings (seen), one solid cone (touched)
        g = [f'<geom name="stacker" type="cylinder" pos="0 0 -.06" size=".055 .012" mass=".06" material="t_white" {fr} {TOY}/>',
             f'<geom name="stacker_stack" type="cylinder" pos="0 0 .0" size=".04 .05" mass=".08" material="t_white" {fr} {TOY} rgba="0 0 0 0" group="3"/>',
             f'<geom type="capsule" fromto="0 0 -.05 0 0 .085" size=".009" material="t_white" {VIS0}/>',
             f'<geom type="sphere" pos="0 0 .09" size=".017" material="t_yellow" {VIS0}/>']
        cols = ("t_red", "t_orange", "t_yellow", "t_green", "t_blue")
        for i, c in enumerate(cols):
            rr = .050 - .0065 * i
            g.append(f'<geom type="mesh" mesh="ring{i}" pos="0 0 {-.036 + i * .023:.4f}" material="{c}" {VIS0}/>')
        return .072, g
    if k == "drum":
        return .055, [f'<geom name="drum" type="cylinder" size=".075 .055" mass=".20" material="t_cyan" condim="6" friction="1 .01 .002" {TOY}/>',
                      f'<geom type="cylinder" pos="0 0 .0552" size=".07 .0006" material="t_white" {VIS0}/>',
                      f'<geom type="mesh" mesh="drum_rim" pos="0 0 .052" material="t_yellow" {VIS0}/>',
                      f'<geom type="mesh" mesh="drum_rim" pos="0 0 -.052" material="t_yellow" {VIS0}/>']
    if k == "ring":     # a teething ring: a torus (its hull is touched)
        return .016, [f'<geom name="ring" type="mesh" mesh="teether" mass=".04" material="t_pink" {fr} {TOY}/>']
    if k == "bottle":   # the charger (5.3): a standing bottle, its collar and its teat, every shape named bottle*
        return .045, [f'<geom name="bottle" type="cylinder" size=".028 .045" mass=".11" material="t_milk" condim="6" friction="1 .01 .002" {TOY}/>',
                      f'<geom name="bottle_collar" type="cylinder" pos="0 0 .051" size=".030 .008" mass=".006" material="t_collar" {fr} {TOY}/>',
                      f'<geom name="bottle_teat" type="capsule" fromto="0 0 .06 0 0 .072" size=".011" mass=".004" material="t_teat" {fr} {TOY}/>']
    raise KeyError(k)


def toys():
    out = []
    for k, ((x, y, yaw), snd) in TOYS.items():
        z0, g = toy_geoms(k)
        base = MAT_T if abs(x - MAT_CENTER[0]) < MAT_HX and abs(y - MAT_CENTER[1]) < MAT_HY else 0
        z = base + z0
        out.append(f'    <body name="toy_{k}" childclass="room" pos="{f(x, y, z + .001)}" euler="0 0 {yaw}"><freejoint name="toy_{k}"/>\n      '
                   + "\n      ".join(g) + "\n    </body>")
    return "\n".join(out)


# ================================================================== THE ROOM
def room():
    W, D, Hh = ROOM_X, ROOM_Y, ROOM_H
    g = []
    g.append(f'<geom name="floor" type="plane" size="{W} {D} .1" material="floor" {FLOOR}/>')
    g.append(f'<geom name="ceiling" type="box" pos="0 0 {Hh + .03}" size="{W} {D} .03" material="ceiling" {DECOR}/>')
    # walls: the back wall (sofa) in a soft sage; the others warm white. The right wall has a doorway to a hall.
    g.append(f'<geom name="wall_back" type="box" pos="0 {D + .05} {Hh / 2}" size="{W + .1} .05 {Hh / 2}" material="wall_accent" {WORLD}/>')
    g.append(f'<geom name="wall_front" type="box" pos="0 {-D - .05} {Hh / 2}" size="{W + .1} .05 {Hh / 2}" material="wall" {WORLD}/>')
    g.append(f'<geom name="wall_left" type="box" pos="{-W - .05} 0 {Hh / 2}" size=".05 {D + .1} {Hh / 2}" material="wall" {WORLD}/>')
    door_y0, door_y1, door_h = -1.95, -1.05, 2.05
    g.append(f'<geom name="wall_right_a" type="box" pos="{W + .05} {(door_y1 + D) / 2:.4g} {Hh / 2}" size=".05 {(D - door_y1) / 2:.4g} {Hh / 2}" material="wall" {WORLD}/>')
    g.append(f'<geom name="wall_right_b" type="box" pos="{W + .05} {(door_y0 - D) / 2:.4g} {Hh / 2}" size=".05 {(door_y0 + D) / 2:.4g} {Hh / 2}" material="wall" {WORLD}/>')
    g.append(f'<geom type="box" pos="{W + .05} {(door_y0 + door_y1) / 2:.4g} {(door_h + Hh) / 2:.4g}" size=".05 {(door_y1 - door_y0) / 2:.4g} {(Hh - door_h) / 2:.4g}" material="wall" {DECOR}/>')
    for yy in (door_y0, door_y1):
        g.append(f'<geom type="box" pos="{W - .005} {yy} {door_h / 2:.4g}" size=".012 .045 {door_h / 2:.4g}" material="trim" {DECOR}/>')
    g.append(f'<geom type="box" pos="{W - .005} {(door_y0 + door_y1) / 2:.4g} {door_h + .04:.4g}" size=".012 {(door_y1 - door_y0) / 2 + .045:.4g} .04" material="trim" {DECOR}/>')
    # the hall beyond the door (the house to come): a floor, a far wall, a side wall
    g.append(f'<geom type="box" pos="{W + 1.0} {(door_y0 + door_y1) / 2:.4g} -.001" size=".95 1.2 .001" material="floor" {DECOR}/>')
    g.append(f'<geom name="hall_end" type="box" pos="{W + 1.95} {(door_y0 + door_y1) / 2:.4g} {Hh / 2}" size=".05 1.2 {Hh / 2}" material="hall" {WORLD}/>')
    g.append(f'<geom name="hall_side_a" type="box" pos="{W + 1.0} {(door_y0 + door_y1) / 2 + 1.2:.4g} {Hh / 2}" size=".95 .05 {Hh / 2}" material="hall" {WORLD}/>')
    g.append(f'<geom name="hall_side_b" type="box" pos="{W + 1.0} {(door_y0 + door_y1) / 2 - 1.2:.4g} {Hh / 2}" size=".95 .05 {Hh / 2}" material="hall" {WORLD}/>')
    # skirting boards
    for pos, size in (((0, D - .012, .05), (W, .012, .05)), ((0, -D + .012, .05), (W, .012, .05)),
                      ((-W + .012, 0, .05), (.012, D, .05)), ((W - .012, (door_y1 + D) / 2, .05), (.012, (D - door_y1) / 2, .05)),
                      ((W - .012, (door_y0 - D) / 2, .05), (.012, (door_y0 + D) / 2, .05))):
        g.append(f'<geom type="box" pos="{f(*pos)}" size="{f(*size)}" material="trim" {DECOR}/>')
    # the window on the left wall, with a sill and curtains
    wx, wy, wz, ww, wh = -W + .005, -.1, 1.45, .80, .70
    g.append(f'<geom name="window" type="box" pos="{wx} {wy} {wz}" size=".004 {ww} {wh}" material="sky" {DECOR}/>')
    for yy, zz, sy, sz in ((wy, wz + wh, ww + .04, .035), (wy, wz - wh, ww + .04, .035), (wy - ww, wz, .035, wh + .04),
                           (wy + ww, wz, .035, wh + .04), (wy, wz, .02, wh), (wy, wz + .12, ww, .02)):
        g.append(f'<geom type="box" pos="{wx + .02:.4g} {yy:.4g} {zz:.4g}" size=".025 {sy:.4g} {sz:.4g}" material="trim" {DECOR}/>')
    g.append(f'<geom type="box" pos="{wx + .07:.4g} {wy} {wz - wh - .045:.4g}" size=".07 {ww + .1:.4g} .02" material="trim" {DECOR}/>')
    g.append(f'<geom type="capsule" fromto="{wx + .12:.4g} {wy - ww - .5:.4g} {wz + wh + .14:.4g} {wx + .12:.4g} {wy + ww + .5:.4g} {wz + wh + .14:.4g}" size=".012" material="wood_d" {DECOR}/>')
    for side in (-1, 1):
        yc = wy + side * (ww + .22)
        for i in range(7):
            yy = yc + (i - 3) * .05
            g.append(f'<geom type="capsule" fromto="{wx + .13:.4g} {yy:.4g} .03 {wx + .13:.4g} {yy:.4g} {wz + wh + .13:.4g}" size="{.028 + .006 * (i % 2):.3g}" material="curtain" {DECOR}/>')
    # the sofa against the back wall
    sx, sy = .15, D - .48
    g += [f'<geom name="sofa_base" type="box" pos="{sx} {sy} .24" size="1.05 .44 .16" material="sofa_d" {WORLD}/>',
          f'<geom name="sofa_back" type="box" pos="{sx} {sy + .34} .58" size="1.05 .12 .30" material="sofa" {WORLD}/>',
          f'<geom name="sofa_arm_l" type="box" pos="{sx - .96} {sy} .47" size=".11 .44 .14" material="sofa" {WORLD}/>',
          f'<geom name="sofa_arm_r" type="box" pos="{sx + .96} {sy} .47" size=".11 .44 .14" material="sofa" {WORLD}/>']
    for i in range(3):
        cx = sx + (i - 1) * .57
        g.append(f'<geom type="mesh" mesh="cushion" pos="{cx:.4g} {sy - .05:.4g} .455" material="sofa" {DECOR}/>')
        g.append(f'<geom type="mesh" mesh="back_cushion" pos="{cx:.4g} {sy + .20:.4g} .70" euler="-12 0 0" material="sofa" {DECOR}/>')
    g.append(f'<geom type="mesh" mesh="pillow" pos="{sx - .70:.4g} {sy + .08:.4g} .66" euler="-15 0 12" material="pillow_a" {DECOR}/>')
    g.append(f'<geom type="mesh" mesh="pillow" pos="{sx + .72:.4g} {sy + .08:.4g} .66" euler="-15 0 -10" material="pillow_b" {DECOR}/>')
    for xx in (-.98, .98):
        for yy in (-.36, .36):
            g.append(f'<geom type="cylinder" pos="{sx + xx:.4g} {sy + yy:.4g} .04" size=".025 .04" material="wood_d" {DECOR}/>')
    # the low table (rounded, baby-safe)
    tx, ty = .15, .95
    g.append(f'<geom name="table_top" type="mesh" mesh="table_top" pos="{tx} {ty} .40" material="wood" {WORLD}/>')
    g.append(f'<geom type="box" pos="{tx} {ty} .18" size=".48 .22 .012" material="wood" {DECOR}/>')
    for xx in (-.46, .46):
        for yy in (-.2, .2):
            g.append(f'<geom name="table_leg_{"l" if xx < 0 else "r"}{"f" if yy < 0 else "b"}" type="cylinder" pos="{tx + xx:.4g} {ty + yy:.4g} .19" size=".025 .19" material="wood" {WORLD}/>')
    g.append(f'<geom type="cylinder" pos="{tx + .25:.4g} {ty + .05:.4g} .43" size=".055 .012" material="pot" {DECOR}/>')
    g.append(f'<geom type="box" pos="{tx - .22:.4g} {ty - .02:.4g} .423" euler="0 0 12" size=".11 .08 .012" material="book2" {DECOR}/>')
    g.append(f'<geom type="box" pos="{tx - .22:.4g} {ty - .02:.4g} .443" euler="0 0 4" size=".1 .075 .009" material="book1" {DECOR}/>')
    # cube shelves on the right wall (4 x 2 cubbies), baskets and books; a plant on top
    shx, shy, cw, dp = W - .21, .55, .39, .195
    nx, nz = 4, 2
    wtot, htot = nx * cw, nz * cw
    g.append(f'<geom name="shelf" type="box" pos="{shx} {shy} {htot / 2 + .03:.4g}" size="{dp} {wtot / 2:.4g} {htot / 2 + .03:.4g}" material="shelf" {WORLD} rgba="0 0 0 0"/>')
    for i in range(nx + 1):
        yy = shy - wtot / 2 + i * cw
        g.append(f'<geom type="box" pos="{shx} {yy:.4g} {htot / 2 + .03:.4g}" size="{dp} .014 {htot / 2 + .03:.4g}" material="shelf" {DECOR}/>')
    for j in range(nz + 1):
        zz = .03 + j * cw
        g.append(f'<geom type="box" pos="{shx} {shy} {zz:.4g}" size="{dp} {wtot / 2 + .014:.4g} .014" material="shelf" {DECOR}/>')
    g.append(f'<geom type="box" pos="{shx + dp - .005:.4g} {shy} {htot / 2 + .03:.4g}" size=".005 {wtot / 2:.4g} {htot / 2 + .03:.4g}" material="shelf" {DECOR}/>')
    contents = {(0, 0): "basket", (1, 0): "books", (2, 0): "basket2", (3, 0): "basket", (0, 1): "books", (1, 1): "plant_small",
                (2, 1): "books2", (3, 1): "basket2"}
    for (i, j), what in contents.items():
        yy = shy - wtot / 2 + (i + .5) * cw
        z0 = .03 + j * cw + .014
        if what.startswith("basket"):
            g.append(f'<geom type="mesh" mesh="basket" pos="{shx - .01:.4g} {yy:.4g} {z0 + .14:.4g}" material="{what}" {DECOR}/>')
        elif what.startswith("books"):
            rng = np.random.default_rng(i * 7 + j)
            y = yy - cw / 2 + .05
            for b in range(7):
                th = rng.uniform(.018, .03)
                hh = rng.uniform(.22, .3)
                tilt = 0 if b < 6 else 14
                g.append(f'<geom type="box" pos="{shx - .02:.4g} {y + th:.4g} {z0 + hh / 2:.4g}" euler="{tilt} 0 0" size=".09 {th:.4g} {hh / 2:.4g}" material="book{1 + (b + i) % 4}" {DECOR}/>')
                y += 2 * th + .002
        else:
            g.append(f'<geom type="cylinder" pos="{shx - .02:.4g} {yy:.4g} {z0 + .06:.4g}" size=".06 .06" material="pot" {DECOR}/>')
            for a in range(6):
                ang = a * math.pi / 3
                g.append(f'<geom type="ellipsoid" pos="{shx - .02 + .04 * math.cos(ang):.4g} {yy + .04 * math.sin(ang):.4g} {z0 + .17:.4g}" euler="{30 * math.sin(ang):.3g} {30 * math.cos(ang):.3g} 0" size=".03 .03 .08" material="leaf" {DECOR}/>')
    # on top of the shelves: a big plant and two frames
    ytop, ztop = shy + .5, htot + .06
    g.append(f'<geom type="cylinder" pos="{shx - .02:.4g} {ytop:.4g} {ztop + .09:.4g}" size=".09 .09" material="pot" {DECOR}/>')
    rng = np.random.default_rng(11)
    for a in range(11):
        ang = a * 2 * math.pi / 11 + rng.uniform(-.2, .2)
        el = rng.uniform(20, 55)
        L = rng.uniform(.16, .26)
        cx, cy = math.cos(ang) * L * .5 * math.cos(math.radians(el)), math.sin(ang) * L * .5 * math.cos(math.radians(el))
        g.append(f'<geom type="ellipsoid" pos="{shx - .02 + cx:.4g} {ytop + cy:.4g} {ztop + .2 + L * .5 * math.sin(math.radians(el)):.4g}" '
                 f'zaxis="{math.cos(ang) * math.cos(math.radians(el)):.3g} {math.sin(ang) * math.cos(math.radians(el)):.3g} {math.sin(math.radians(el)):.3g}" size=".045 .016 {L / 2:.3g}" material="leaf" {DECOR}/>')
    g.append(f'<geom type="box" pos="{shx:.4g} {shy - .45:.4g} {ztop + .13:.4g}" euler="0 -8 0" size=".012 .11 .14" material="wood_d" {DECOR}/>')
    g.append(f'<geom type="box" pos="{shx - .014:.4g} {shy - .45:.4g} {ztop + .13:.4g}" euler="0 -8 0" size=".004 .09 .12" material="art_d" {DECOR}/>')
    # the floor lamp at the sofa's left end
    lx, ly = -1.25, D - .35
    g += [f'<geom type="cylinder" pos="{lx} {ly} .012" size=".14 .012" material="lamp_metal" {DECOR}/>',
          f'<geom type="capsule" fromto="{lx} {ly} .02 {lx} {ly} 1.45" size=".012" material="lamp_metal" {DECOR}/>',
          f'<geom type="cylinder" pos="{lx} {ly} 1.52" size=".19 .13" material="lamp_shade" {DECOR}/>']
    # a big plant in the front-left corner
    px, py = -W + .38, -D + .45
    g.append(f'<geom type="cylinder" pos="{px} {py} .19" size=".19 .19" material="pot" {DECOR}/>')
    for a in range(16):
        ang = a * 2 * math.pi / 16 + rng.uniform(-.2, .2)
        el = rng.uniform(35, 80)
        L = rng.uniform(.35, .7)
        dx, dy, dz = math.cos(ang) * math.cos(math.radians(el)), math.sin(ang) * math.cos(math.radians(el)), math.sin(math.radians(el))
        g.append(f'<geom type="ellipsoid" pos="{px + dx * L * .5:.4g} {py + dy * L * .5:.4g} {.38 + dz * L * .5:.4g}" zaxis="{dx:.3g} {dy:.3g} {dz:.3g}" size=".08 .02 {L / 2:.3g}" material="leaf" {DECOR}/>')
    # wall art above the sofa: two frames of soft shapes
    for k, (ax, aw, ah, c1, c2) in enumerate(((-.45, .34, .26, "art_a", "art_b"), (.62, .26, .34, "art_d", "art_a"))):
        ay = D - .012
        g.append(f'<geom type="box" pos="{ax} {ay} 1.45" size="{aw} .012 {ah}" material="wood_d" {DECOR}/>')
        g.append(f'<geom type="box" pos="{ax} {ay - .008:.4g} 1.45" size="{aw - .025:.4g} .012 {ah - .025:.4g}" material="art_c" {DECOR}/>')
        g.append(f'<geom type="cylinder" pos="{ax - aw * .3:.4g} {ay - .022:.4g} {1.45 + ah * .15:.4g}" zaxis="0 1 0" size="{ah * .45:.3g} .003" material="{c1}" {DECOR}/>')
        g.append(f'<geom type="box" pos="{ax + aw * .25:.4g} {ay - .024:.4g} {1.45 - ah * .2:.4g}" size="{aw * .35:.3g} .003 {ah * .35:.3g}" material="{c2}" {DECOR}/>')
    # the ceiling lamp
    g.append(f'<geom type="cylinder" pos="0 .3 {Hh - .02}" size=".22 .02" material="lamp_shade" {DECOR}/>')
    return "\n    ".join(g)


MAT_SOLID = .10     # the mat's collision box reaches this far below its top (its top at MAT_T, the rest sunk under the floor)


def play_mat():
    """The foam mat: its collision box's top face at the mat's height, the box sunk MAT_SOLID below it into the floor, where no
    one sees it. As a box only MAT_T (1.2 cm) thick, one of the G1's foot spheres (5 mm, the Menagerie file's four point feet)
    pressed into the soft foam passed the box's mid-plane, so the box's nearest face became its bottom and pushed the foot down
    into the floor, which pushed it up: a foot trapped between the two at 1,900 N, felt as pain on 151 of 400 babbled ticks (the
    world's measurement, 2026-09-24). A thick box's nearest face is always its top."""
    cx, cy = MAT_CENTER
    g = [f'<geom name="mat" type="box" pos="{cx} {cy} {MAT_T - MAT_SOLID / 2:.4g}" size="{MAT_HX} {MAT_HY} {MAT_SOLID / 2}" material="mat_edge" '
         f'friction="1 .01 .001" solref=".03 1" solimp=".85 .95 .004" {FLOOR}/>']
    tw = .4
    nx, ny = int(round(2 * MAT_HX / tw)), int(round(2 * MAT_HY / tw))
    for i in range(nx):
        for j in range(ny):
            x = cx - MAT_HX + (i + .5) * tw
            y = cy - MAT_HY + (j + .5) * tw
            g.append(f'<geom type="box" pos="{x:.4g} {y:.4g} {MAT_T / 2 + .0004:.4g}" size="{tw / 2 - .002:.4g} {tw / 2 - .002:.4g} {MAT_T / 2:.4g}" '
                     f'material="{"tile_a" if (i + j) % 2 == 0 else "tile_b"}" {DECOR}/>')
    # the charge pad, the bottle's dock (a low white disc with a soft mint glow; A17: beside the lying G1's right hand)
    px, py = DOCK                                                     # beside the lying G1's left hand (A17)
    g.append(f'<geom name="pad" type="cylinder" pos="{px:.4g} {py:.4g} {MAT_T + .004:.4g}" size=".13 .004" material="pad" {DECOR}/>')
    g.append(f'<geom name="pad_ring" type="mesh" mesh="pad_ring" pos="{px:.4g} {py:.4g} {MAT_T + .006:.4g}" material="pad_ring" {DECOR}/>')   # named: the pad (her floor plan steps over it)
    return "\n    ".join(g)


# ================================================================== textures
def floor_texture(path):
    """Light oak boards, 1024 x 1024, deterministic: 8 boards across, each broken once at a random length, a gentle
    grain, a thin dark seam."""
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(7)
    N, boards = 1024, 8
    img = np.zeros((N, N, 3), np.float32)
    base = np.array([.79, .64, .47])
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    w = N // boards
    for i in range(boards):
        cut = rng.integers(0, N)
        for seg in range(2):
            tone = base * rng.uniform(.94, 1.05) + rng.uniform(-.012, .012, 3)
            rows = ((yy - cut) % N < N // 2) if seg == 0 else ((yy - cut) % N >= N // 2)
            m = rows & (xx >= i * w) & (xx < (i + 1) * w)
            ph = rng.uniform(0, 6)
            grain = .022 * np.sin((xx - i * w) * .55 + 3.0 * np.sin(yy * .006 + ph) + ph) + .012 * np.sin((xx - i * w) * 1.9 + yy * .02)
            img[m] = tone[None, :] * (1 + grain[m, None])
            img[m & ((((yy - cut) % (N // 2))) < 2)] *= .86
        img[:, i * w:i * w + 2] *= .84
    im = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(.6))
    im.save(path)


def sky_texture(path):
    from PIL import Image, ImageFilter
    H_, W_ = 256, 256
    t = np.linspace(0, 1, H_)[:, None, None]
    top, hor = np.array([.62, .78, .95]), np.array([.93, .96, .99])
    img = np.broadcast_to(top * (1 - t) + hor * t, (H_, W_, 3)).copy()
    rng = np.random.default_rng(3)
    x = np.arange(W_)
    line = (H_ * .74 + 12 * np.sin(x / 23 + 1) + 7 * np.sin(x / 9 + rng.uniform(0, 6)) + 3 * np.sin(x / 4)).astype(int)
    for i in range(W_):
        img[line[i]:, i] = np.array([.66, .78, .66]) * (1 - .1 * (np.arange(H_ - line[i]) / 60).clip(0, 1))[:, None]
    for cx, cy, r in ((60, 50, 18), (80, 45, 22), (100, 52, 16), (190, 80, 14), (205, 76, 18)):
        yy, xx = np.mgrid[0:H_, 0:W_]
        m = ((xx - cx) ** 2 + ((yy - cy) * 1.6) ** 2) < r * r
        img[m] = img[m] * .3 + .7
    Image.fromarray((img * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5)).save(path)



def deg_to_rad_eulers(xml):
    """Every euler attribute in this maker's text is written in degrees; the model is compiled in radians (the G1's
    compiler), so convert them all here, in one place."""
    def conv(mt):
        return 'euler="' + " ".join(f"{math.radians(float(v)):.6g}" for v in mt.group(1).split()) + '"'
    return re.sub(r'euler="([^"]+)"', conv, xml)


def build():
    """write the textures and the scene (to --out=<path> if given, else body/sim/g1room.xml)"""
    out = Path(ARGS["out"]).resolve() if "out" in ARGS else HERE / "g1room.xml"
    (HERE / "textures").mkdir(exist_ok=True)
    floor_texture(HERE / "textures" / "room_floor_oak.png")
    sky_texture(HERE / "textures" / "room_window_sky.png")
    if "skip-face" not in ARGS:                                        # her face's meshes, texture and rig (parent_face.build_assets)
        info = kin.face.build_assets(HERE / "textures", hair=hair_shapes(), room_indirect=ROOM_INDIRECT)
        print("her face:", info)
    out.write_text(scene_xml(out.parent))
    return out


def face_shaded(mat):
    """the materials with the eye's parts' shade of the room's indirect light baked into their albedo (parent_face: computed from
    her geometry by the build, stored in assets/parent/rig.npz; never set): albedo x (1 - share x (1 - AO)), the share
    ROOM_INDIRECT / (ROOM_INDIRECT + 0.5) (parent_face.ao_share)"""
    rig = kin.face.rig()
    share = kin.face.ao_share(ROOM_INDIRECT)
    ao = {"eye": float(rig["ao_eye"]), "lid": float(rig["ao_lid"]), "caruncle": float(rig["ao_caruncle"])}
    parts = {"sclera": "eye", "iris": "eye", "pupil": "eye", "fornix": "eye", "lid": "lid", "caruncle": "caruncle"}   # the lash line's
    # colour is calibrated as seen (tools/sim_face_photometry.py), its shade in
    out = {k: dict(v) for k, v in mat.items()}
    for n, part in parts.items():
        rgba = [float(x) for x in out[n]["rgba"].split()]
        k = 1 - share * (1 - ao[part])
        out[n]["rgba"] = " ".join(f"{x * k:.4g}" for x in rgba[:3]) + f" {rgba[3]:g}"
    return out


def scene_xml(folder=HERE):
    """the scene's XML text, its paths relative to `folder` (where it is to be written)"""
    rel = lambda p: os.path.relpath(HERE / p, Path(folder).resolve())     # every path in the XML relative to the XML's own folder
    mats = "\n    ".join(f'<material name="{k}" ' + " ".join(f'{a}="{v}"' for a, v in d.items()) + "/>" for k, d in face_shaded(MAT).items())
    cams = dict(room=((2.45, -2.2, 1.7), (-.2, -.2, .35), 52), mat=((1.6, -2.0, 1.15), (-.1, -.55, .2), 42),
                top=((0, -.6, 3.2), (0, -.599, 0), 50))
    cam_xml = "\n    ".join(f'<camera name="{k}" pos="{f(*p)}" xyaxes="{lookat_xyaxes(p, t)}" fovy="{fv}"/>' for k, (p, t, fv) in cams.items())
    welds = []
    for hs in ("L", "R"):
        for k in TOYS:
            welds.append(f'<weld name="hold_{hs}_{k}" body1="parent_hand_{hs}" body2="toy_{k}" active="false" solref=".006 1"/>')
        # no weld on the G1 (W2): the prototype's welds on its torso, pelvis and elbows had no cap, and a weld on the G1 would break a
        # person's caps; every hold on it is a capped spring (body/sim/parent_motion.py)
    sounds = "\n    ".join(f'<text name="sound_{k}" data="{snd}"/>' for k, (_, snd) in TOYS.items())
    xml = f"""<!-- GENERATED by body/sim/make_g1room.py (docs/SIM_DESIGN.md): the stock Unitree G1 (included unchanged), the
     parent and the living room. Do not hand-edit. Units m, kg, s, RADIANS. Floor z = 0; the room spans x +-{ROOM_X},
     y +-{ROOM_Y}, {ROOM_H} m high. Load it through body/sim/g1scene.py, which adds the G1's senses (cameras and sites only). -->
<mujoco model="g1room">
  <include file="{rel(G1_XML)}"/>
  <!-- this compiler element comes after the G1's, so it is the one MuJoCo applies to the whole model: the same
       angle unit as the G1's (radian); the G1's own mesh folder given from this file's folder (the included file's
       own meshdir "assets" would resolve from here, not from its folder) -->
  <compiler angle="radian" meshdir="{rel(G1_DIR / 'assets')}" texturedir="{rel('textures')}" autolimits="true"/>
  <!-- the elliptic friction cone: with the pyramidal cone a block squeezed in a Dex3 hand stopped MuJoCo 3.9 (Newton:
       "FactorizeHessian: rank-deficient sparse Hessian"; CG or a dense Jacobian: NaN accelerations and a reset) -->
  <!-- multi-point CCD off: with it on (MuJoCo 3.9's default) 2-4 of 12 hand-overs of the block or the car into a Dex3
       hand blew up about 0.1 s into the fingers' closing (NaN accelerations, an automatic reset); off, none did -->
  <!-- impratio 10: friction's impedance ten times the normal's. CHOSEN BY THE PHYSICS OF THESE CONTACTS (the owner's decision
       of 2026-09-24, made for him; tools/sim_friction.py, measured 2026-09-24 at 1 and 10, never by pain rates or grasp counts):
       real rubber, plastic, foam and housings hold without a slide below mu N; MuJoCo's soft contacts creep there "by design"
       (MuJoCo 3.9 docs, Overview, "Softness and slip"), and the docs name the remedy: "using the Newton solver with elliptic
       friction cones and large value of impratio is the recommended way of reducing slip" (ibid.; Modeling, "Solver
       settings": "When contact slip is a problem, the best way to suppress it is to use elliptic cones, large impratio, and the
       Newton algorithm with very small tolerance"; both on the stable docs, read 2026-09-24); MuJoCo Menagerie's hand and gripper models (the Shadow hand, the
       Allegro hand, the Robotiq 2F-85, ALOHA) ship cone="elliptic" impratio="10". Measured: a 10 kg box on the mat's contact
       pushed at 0.3-0.97 mu M g slid 3.6-21 mm in 2 s at 1 and 0.4-1.8 mm at 10 (real: none); the resting G1 pushed at 100 /
       150 / 200 N at the pelvis slid 1.9 / 3.1 / 4.4 cm at 1 and 0.4 / 0.7 / 1.9 cm at 10; above mu M g both slide as Coulomb's
       law says (0.99-1.00 of its distance). THE COST, disclosed (C5, C22): the convex contact model couples a slip to the normal
       direction ("the only way to initiate slip is to generate some motion in the normal direction", Computation, "Physical realism
       and soft contacts"), so a
       pressed box that starts to slide is pressed 22-24% harder than its load for about 10 ms at either setting, steady 0.5-5%
       after; and in the G1's hip housings pressed together and slid apart by the newborn's flexion, friction raises the normal
       force at 10 (1,795 N against 1,553 N with the pair frictionless) where at 1 it lowers it (1,333 against 1,646): a
       pressed housing that slides reads harder at 10. The noslip solver (the docs' next step, "If that is not sufficient, enable
       the Noslip solver") would stop the slip, but its cascade "is no longer solving a well-defined optimization problem (or any
       other problem); instead it is just an adhoc mechanism" (Modeling, "Solver settings"), and it nearly doubled the tick. -->
  <!-- auto-reset off (A18): MuJoCo would otherwise put a state with a bad position, velocity or acceleration back to the
       start pose by itself and go on; off, the bad state stays as it is, and the world (body/sim/world.py) finds it on the
       tick it happens, before that tick's frame reaches the body -->
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10">
    <flag multiccd="disable" autoreset="disable"/>
  </option>
  <size memory="128M"/>
  <statistic center="0 -.5 .3" extent="2.5"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="45"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight active="0" ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/>
    <map znear=".001" zfar="30" shadowclip="2.6" shadowscale=".6"/>
  </visual>
  <default>
    <default class="room">
      <geom solref=".015 1" solimp=".9 .95 .001"/>
    </default>
  </default>
  <asset>
    <texture name="floor_oak" type="2d" file="room_floor_oak.png"/>
    <texture name="window_sky" type="2d" file="room_window_sky.png"/>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1=".92 .94 .97" rgb2=".75 .78 .82" width="256" height="256"/>
    {mats}
    <mesh name="cup_rim" builtin="supertorus" params="40 .10 1 1" scale="{mesh_scale("cup_rim", [0.039, 0.039, 0.039])}"/>
    <mesh name="cup_handle" builtin="supertorus" params="30 .24 1 1" scale="{mesh_scale("cup_handle", [0.02, 0.02, 0.02])}"/>
    <mesh name="drum_rim" builtin="supertorus" params="48 .06 1 1" scale="{mesh_scale("drum_rim", [0.076, 0.076, 0.076])}"/>
    <mesh name="teether" builtin="supertorus" params="40 .30 1 1" scale="{mesh_scale("teether", [0.05, 0.05, 0.05])}"/>
    <mesh name="pad_ring" builtin="supertorus" params="48 .05 1 1" scale=".13 .13 .13"/>
    <mesh name="ring0" builtin="supertorus" params="40 .40 1 1" scale="{mesh_scale("ring0", [0.05, 0.05, 0.05])}"/>
    <mesh name="ring1" builtin="supertorus" params="40 .40 1 1" scale="{mesh_scale("ring1", [0.044, 0.044, 0.044])}"/>
    <mesh name="ring2" builtin="supertorus" params="40 .40 1 1" scale="{mesh_scale("ring2", [0.038, 0.038, 0.038])}"/>
    <mesh name="ring3" builtin="supertorus" params="40 .40 1 1" scale="{mesh_scale("ring3", [0.032, 0.032, 0.032])}"/>
    <mesh name="ring4" builtin="supertorus" params="40 .40 1 1" scale="{mesh_scale("ring4", [0.026, 0.026, 0.026])}"/>
    <mesh name="cushion" builtin="supersphere" params="24 .3 .3" scale=".28 .40 .07"/>
    <mesh name="back_cushion" builtin="supersphere" params="24 .35 .35" scale=".28 .09 .24"/>
    <mesh name="pillow" builtin="supersphere" params="24 .45 .5" scale=".20 .07 .19"/>
    <mesh name="table_top" builtin="supersphere" params="32 .25 .2" scale=".58 .30 .035"/>
    <mesh name="basket" builtin="supersphere" params="24 .2 .15" scale=".17 .17 .14"/>
    {chr(10).join('    ' + x for x in kin.face.asset_xml(lambda q: os.path.relpath(q, HERE / G1_DIR / 'assets'))).strip()}
  </asset>
  <custom>
    {sounds}
  </custom>
  <worldbody>
    <!-- the room's lights, each carrying the room's indirect light as its ambient (ROOM_INDIRECT of its diffuse); no headlight -->
    {lights_xml()}
    {cam_xml}
    <body name="room" childclass="room">
    {room()}
    {play_mat()}
    </body>
{toys()}
{parent()}
  </worldbody>
  <actuator>
    <!-- HER MUSCLES: one per axis of each of her joints, their force ranges her strength (parent_consts.STRENGTH; A25b) -->
    {chr(10).join("    " + a for a in parent_actuators()).strip()}
  </actuator>
  <equality>
    {chr(10).join("    " + w for w in welds).strip()}
  </equality>
</mujoco>
"""
    return deg_to_rad_eulers(xml)


if __name__ == "__main__":
    p = build()
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(p))
    print(p, "nq", m.nq, "nv", m.nv, "nu", m.nu, "ngeom", m.ngeom, "nbody", m.nbody, "nmocap", m.nmocap, "neq", m.neq,
          "nsensor", m.nsensor, "nlight", m.nlight, "nkey", m.nkey, "G1 mass", round(float(m.body_subtreemass[m.body("pelvis").id]), 3))
