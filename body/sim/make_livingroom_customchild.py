"""Writes body/sim/livingroom_customchild.xml and its two textures: the CUSTOM CHILD's living room (kept for reference; the
child is now the stock Unitree G1, make_g1room.py; docs/SIM_DESIGN.md sections 15 and 17). From the 2026-09-24 prototype.

  THE CHILD   a physical humanoid about 12 months in size (0.75 m, 9.4 kg): pelvis, a three-joint lumbar and a
              three-joint thoracic spine, a three-joint neck, a head with two eyes that turn (a periphery camera and a
              fovea camera on each), two ears, a vestibule (accelerometer and gyro), a screen face; shoulders (3),
              elbows, forearm rotation, wrists, a mitten hand with a two-joint finger paddle pulled by one tendon and an
              opposing thumb; hips (3), knees, ankles (2). Position servos with honest torque limits; no controller.
  THE PARENT  kinematic: 16 mocap segments (see parent_kin.py) posed by scripted IK each step; touches toys and the child as
              an immovable body; holds toys, the child's forearm or its chest through welds that start switched off.
  THE ROOM    a living room: oak floor, a foam play mat, a sofa, a low table, cube shelves, a window with curtains, a
              doorway to a hall (the house to come), a floor lamp, a plant, wall art; ten toys with their own sounds.

Collision bits: world 1 (floor, walls, mat, furniture), child 2 (self-collision on), toys 4, parent 8 (touches the
child and the toys only). Run: python3 body/sim/make_livingroom_customchild.py  (writes livingroom_customchild.xml and textures/room_*.png; the XML is
generated: edit the numbers here and re-run, never the XML)."""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parent_kin as kin  # noqa: E402

# ------------------------------------------------------------------ collision classes
WORLD = 'contype="1" conaffinity="0"'
CHILD = 'contype="2" conaffinity="15"'
TOY = 'contype="4" conaffinity="13"'
PARENT = 'contype="8" conaffinity="0"'
DECOR = 'contype="0" conaffinity="0"'
VIS0 = DECOR + ' density="0"'          # massless and seen only (on dynamic bodies)

MAT_CENTER = np.array([0.0, -0.25])
MAT_HALF, MAT_T = .80, .012            # a 1.6 x 1.6 m foam mat (4 x 4 tiles of 0.4 m), 1.2 cm thick
ROOM_X, ROOM_Y, ROOM_H = 2.6, 2.3, 2.6

MAT = dict(
    # the child: a soft white shell, dark joints, a coral accent, a dark screen face with a cool glow
    shell=dict(rgba=".96 .95 .93 1", specular=".25", shininess=".5"),
    shell2=dict(rgba=".90 .89 .87 1", specular=".25", shininess=".5"),
    cjoint=dict(rgba=".27 .28 .31 1", specular=".3", shininess=".5"),
    coral=dict(rgba=".98 .55 .38 1", specular=".3", shininess=".5"),
    screen=dict(rgba=".06 .07 .09 1", specular=".8", shininess=".95"),
    glow=dict(rgba=".78 .96 1 1", emission="1", specular="0"),
    # the parent: skin, a teal sweater, denim trousers, white shoes, dark brown hair; face features
    skin=dict(rgba=".86 .66 .54 1", specular=".12", shininess=".25"),
    skin_d=dict(rgba=".80 .58 .47 1", specular=".1", shininess=".25"),
    hair=dict(rgba=".24 .15 .10 1", specular=".3", shininess=".6"),
    sweater=dict(rgba=".20 .56 .62 1", specular=".08", shininess=".15"),
    sweater_d=dict(rgba=".15 .45 .50 1", specular=".08", shininess=".15"),
    denim=dict(rgba=".22 .27 .38 1", specular=".1", shininess=".2"),
    shoe=dict(rgba=".95 .95 .94 1", specular=".2", shininess=".4"),
    sole=dict(rgba=".62 .60 .58 1", specular=".1", shininess=".2"),
    sclera=dict(rgba=".98 .98 .97 1", specular=".5", shininess=".8"),
    iris=dict(rgba=".30 .18 .10 1", specular=".6", shininess=".9"),
    pupil=dict(rgba=".03 .03 .03 1", specular=".8", shininess=".9"),
    glint=dict(rgba="1 1 1 1", emission=".8", specular="0"),
    lips=dict(rgba=".56 .24 .26 1", specular=".25", shininess=".5"),
    mouth_in=dict(rgba=".30 .06 .08 1", specular=".1", shininess=".2"),
    teeth=dict(rgba=".97 .96 .93 1", specular=".3", shininess=".5"),
    brow=dict(rgba=".30 .20 .13 1", specular=".1", shininess=".2"),
    blush=dict(rgba=".88 .60 .54 1", specular="0"),
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
    metal=dict(rgba=".25 .25 .27 1", specular=".6", shininess=".8"),
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


# ================================================================== THE CHILD
# Segment numbers (standing, pelvis frame at the hip joints' level, x forward, y left, z up). A 12-month-old: stature
# 0.75 m, sitting height about 0.47 m, head circumference 46 cm (a sphere of radius 7.4 cm), biacromial about 0.17 m.
# Masses by infant segment fractions (Jensen-style: head about 20 %, trunk 45 %, each thigh 6.5 %, shank 3.5 %,
# foot 1.3 %, upper arm 2.7 %, forearm 1.7 %, hand 0.8 %), rounded; the total is 9.4 kg.
C = dict(
    hip_y=.05, thigh_L=.145, shin_L=.150, ankle_h=.041,
    lumbar_z=.07, thor_z=.075, neck_z=.125, shoulder=(0, .082, .09),
    ua_L=.12, fa_L=.10, head_c=(.01, 0, .075), head_r=.074,
)
MASS = dict(pelvis=.95, abdomen=1.0, chest=2.45, head=1.90, upper_arm=.26, forearm=.16, palm=.05, finger1=.012,
            finger2=.008, thumb=.010, thigh=.62, shin=.33, foot=.12, eye=.004)

# Torque limits: adult reference peak isometric torques (rounded, approximate, from the standard strength literature)
# scaled by 0.08 = the mass ratio 9.4 / 70 (0.134, geometric similarity: torque ~ L^3 ~ mass) x 0.6 (an infant's smaller
# muscle fraction of body mass, about 25 % against 40 %). (negative-direction limit, positive-direction limit), N m.
# Positive is flexion / abduction / external rotation / dorsiflexion (the joint axes below are set so).
ADULT = dict(hip_flex=(200, 150), hip_abd=(120, 100), hip_rot=(50, 50), knee=(220, 110), ankle_dorsi=(140, 40),
             ankle_inv=(30, 30), lumbar_flex=(250, 150), lumbar_lat=(130, 130), lumbar_rot=(60, 60),
             thor_flex=(250, 150), thor_lat=(130, 130), thor_rot=(60, 60),
             neck_flex=(40, 25), neck_lat=(25, 25), neck_rot=(15, 15),
             shoulder_flex=(80, 70), shoulder_abd=(80, 60), shoulder_rot=(40, 40), elbow=(50, 70),
             pronation=(10, 10), wrist_flex=(10, 15))
SCALE = .08
TAU = {k: (-SCALE * a, SCALE * b) for k, (a, b) in ADULT.items()}
TAU["finger"] = (-.10, .30)     # the finger tendon (N m on each finger joint): flexors pull, extensors weaker
TAU["thumb"] = (-.06, .15)
TAU["eye"] = (-.02, .02)

# joint ranges, degrees (an infant's ranges, a little wider than an adult's where infants are known to be: hip
# flexion and abduction, ankle dorsiflexion)
RNG = dict(lumbar_flex=(-25, 50), lumbar_lat=(-25, 25), lumbar_rot=(-20, 20),
           thor_flex=(-20, 40), thor_lat=(-20, 20), thor_rot=(-35, 35),
           neck_flex=(-60, 60), neck_lat=(-45, 45), neck_rot=(-80, 80),
           eye_yaw=(-40, 40), eye_pitch=(-35, 35),
           shoulder_flex=(-45, 180), shoulder_abd=(-30, 160), shoulder_rot=(-70, 90), elbow=(0, 145),
           pronation=(-80, 85), wrist_flex=(-60, 75), finger1=(-10, 95), finger2=(0, 100), thumb=(0, 95),
           hip_flex=(-20, 130), hip_abd=(-30, 70), hip_rot=(-40, 50), knee=(0, 150),
           ankle_dorsi=(-50, 40), ankle_inv=(-35, 20))
KP_ERR = .25    # a servo reaches its torque limit at a 0.25 rad error; kv = 0.04 s x kp
JOINTS = []     # (name, range key, torque key, armature, damping) in order: the actuator list follows it


def jnt(name, axis, rk, tk=None, arm=.002, damp=.05, extra=""):
    lo, hi = RNG[rk]
    JOINTS.append((name, rk, tk if tk is not None else rk, arm, damp))
    return (f'<joint name="{name}" axis="{f(*axis)}" range="{lo} {hi}" armature="{arm}" damping="{damp}" {extra}/>')


def touch_site(name, typ, size, pos=(0, 0, 0), extra=""):
    TOUCH.append(name)
    return f'<site name="{name}" type="{typ}" size="{size}" pos="{f(*pos)}" group="4" rgba="1 .3 .3 .25" {extra}/>'


TOUCH = []


def child_arm(sd):
    """One arm; sd 'L' or 'R'. Built for the left and mirrored (y -> -y; axes (x, y, z) -> (-x, y, -z))."""
    m = (lambda p: p) if sd == "L" else mirror_pos
    ax = (lambda a: a) if sd == "L" else mirror_axis
    p = f"child_{sd}_"
    sh = m(C["shoulder"])
    return f"""
        <body name="{p}upper_arm" pos="{f(*sh)}">
          {jnt(p + "shoulder_flex", ax((0, -1, 0)), "shoulder_flex")}
          {jnt(p + "shoulder_abd", ax((1, 0, 0)), "shoulder_abd")}
          {jnt(p + "shoulder_rot", ax((0, 0, 1)), "shoulder_rot")}
          <geom name="{p}upper_arm" type="capsule" fromto="0 0 -.008 0 0 -.106" size=".0265" mass="{MASS['upper_arm']}" material="shell" {CHILD}/>
          <geom type="sphere" size=".03" material="coral" {VIS0}/>
          {touch_site(p + "upper_arm_touch", "capsule", ".031 .052", (0, 0, -.057))}
          <body name="{p}forearm" pos="0 0 {-C['ua_L']}">
            {jnt(p + "elbow", ax((0, -1, 0)), "elbow")}
            {jnt(p + "pronation", ax((0, 0, 1)), "pronation", arm=.001, damp=.02)}
            <geom type="sphere" size=".0235" material="cjoint" {VIS0}/>
            <geom name="{p}forearm" type="capsule" fromto="0 0 -.006 0 0 -.088" size=".022" mass="{MASS['forearm']}" material="shell" {CHILD}/>
            {touch_site(p + "forearm_touch", "capsule", ".026 .043", (0, 0, -.047))}
            <site name="{p}forearm_hold" pos="0 0 -.07" size=".006" group="4"/>
            <body name="{p}hand" pos="0 0 {-C['fa_L']}">
              {jnt(p + "wrist_flex", ax((-1, 0, 0)), "wrist_flex", arm=.001, damp=.02)}
              <geom type="sphere" size=".0175" material="coral" {VIS0}/>
              <geom name="{p}palm" type="box" pos="0 0 -.022" size=".022 .0115 .021" mass="{MASS['palm']}" group="3" material="shell2" condim="4" friction="1.1 .02 .001" {CHILD}/>
              <geom type="mesh" mesh="c_palm" pos="0 0 -.022" material="shell2" {VIS0}/>
              {touch_site(p + "palm_touch", "box", ".024 .016 .023", m((0, -.004, -.022)))}
              <site name="{p}grasp" pos="{f(*m((0, -.022, -.058)))}" size=".008" group="4"/>
              <body name="{p}finger1" pos="0 0 -.043">
                {jnt(p + "finger1", ax((-1, 0, 0)), "finger1", "finger", arm=.0005, damp=.004, extra='stiffness=".03" springref="20"')}
                <geom name="{p}finger1" type="box" pos="0 0 -.011" size=".021 .0095 .0115" mass="{MASS['finger1']}" group="3" material="shell2" condim="4" friction="1.1 .02 .001" {CHILD}/>
                <geom type="mesh" mesh="c_finger1" pos="0 0 -.011" material="shell2" {VIS0}/>
                {touch_site(p + "finger1_touch", "box", ".023 .013 .013", m((0, -.003, -.011)))}
                <body name="{p}finger2" pos="0 0 -.023">
                  {jnt(p + "finger2", ax((-1, 0, 0)), "finger2", "finger", arm=.0005, damp=.004, extra='stiffness=".03" springref="25"')}
                  <geom name="{p}finger2" type="box" pos="0 0 -.009" size=".020 .0085 .0095" mass="{MASS['finger2']}" group="3" material="shell2" condim="4" friction="1.1 .02 .001" {CHILD}/>
                  <geom type="mesh" mesh="c_finger2" pos="0 0 -.009" material="shell2" {VIS0}/>
                  {touch_site(p + "fingertip_touch", "box", ".022 .012 .012", m((0, -.003, -.009)))}
                </body>
              </body>
              <body name="{p}thumb" pos="{f(*m((.021, -.007, -.010)))}">
                {jnt(p + "thumb", ax((0, 0, -1)), "thumb", "thumb", arm=.0005, damp=.004, extra='stiffness=".02" springref="15"')}
                <geom name="{p}thumb" type="capsule" fromto="{f(*m((0, 0, 0)), *m((.010, 0, -.026)))}" size=".0078" mass="{MASS['thumb']}" material="shell2" condim="4" friction="1.1 .02 .001" {CHILD}/>
                {touch_site(p + "thumb_touch", "capsule", ".0105 .015", m((.005, 0, -.013)), f'zaxis="{f(*m((.36, 0, -.93)))}"')}
              </body>
            </body>
          </body>
        </body>"""


def child_leg(sd):
    m = (lambda p: p) if sd == "L" else mirror_pos
    ax = (lambda a: a) if sd == "L" else mirror_axis
    p = f"child_{sd}_"
    return f"""
      <body name="{p}thigh" pos="{f(*m((0, C['hip_y'], 0)))}">
        {jnt(p + "hip_flex", ax((0, -1, 0)), "hip_flex")}
        {jnt(p + "hip_abd", ax((1, 0, 0)), "hip_abd")}
        {jnt(p + "hip_rot", ax((0, 0, 1)), "hip_rot")}
        <geom name="{p}thigh" type="capsule" fromto="0 0 -.02 0 0 -.122" size=".041" mass="{MASS['thigh']}" material="shell" {CHILD}/>
        {touch_site(p + "thigh_touch", "capsule", ".045 .058", (0, 0, -.07))}
        <body name="{p}shin" pos="0 0 {-C['thigh_L']}">
          {jnt(p + "knee", ax((0, 1, 0)), "knee")}
          <geom type="sphere" size=".033" material="coral" {VIS0}/>
          <geom name="{p}shin" type="capsule" fromto="0 0 -.012 0 0 -.118" size=".030" mass="{MASS['shin']}" material="shell" {CHILD}/>
          {touch_site(p + "shin_touch", "capsule", ".035 .066", (0, 0, -.06))}
          <body name="{p}foot" pos="0 0 {-C['shin_L']}">
            {jnt(p + "ankle_dorsi", ax((0, -1, 0)), "ankle_dorsi", arm=.001, damp=.03)}
            {jnt(p + "ankle_inv", ax((1, 0, 0)), "ankle_inv", arm=.001, damp=.03)}
            <geom type="sphere" size=".022" material="cjoint" {VIS0}/>
            <geom name="{p}foot" type="box" pos=".03 0 -.025" size=".055 .028 .016" mass="{MASS['foot']}" group="3" material="shell2" friction="1 .02 .001" {CHILD}/>
            <geom type="ellipsoid" pos=".03 0 -.022" size=".059 .031 .019" material="shell2" {VIS0}/>
            <geom type="cylinder" pos=".03 0 -.0385" size=".026 .0025" material="coral" {VIS0}/>
            {touch_site(p + "sole_touch", "box", ".058 .031 .006", (.03, 0, -.041))}
          </body>
        </body>
      </body>"""


def child_eye(sd):
    """One eye: yaw then pitch about its own centre, which sits on the screen at the drawn eye; a periphery camera
    (fovy EYE_P_FOVY) and a fovea camera (EYE_F_FOVY) look along the head's +x."""
    m = (lambda p: p) if sd == "L" else mirror_pos
    p = f"child_{sd}_"
    pos = m(EYE_POS)
    return f"""
          <body name="{p}eye" pos="{f(*pos)}">
            {jnt(p + "eye_yaw", (0, 0, 1), "eye_yaw", "eye", arm=.0002, damp=.0005)}
            {jnt(p + "eye_pitch", (0, -1, 0), "eye_pitch", "eye", arm=.0002, damp=.0005)}
            <geom type="sphere" size=".004" mass="{MASS['eye']}" material="screen" contype="0" conaffinity="0" group="3"/>
            <camera name="{p}eye" pos="0 0 0" xyaxes="0 -1 0 0 0 1" fovy="{EYE_P_FOVY}"/>
            <camera name="{p}fovea" pos="0 0 0" xyaxes="0 -1 0 0 0 1" fovy="{EYE_F_FOVY}"/>
          </body>"""


EYE_POS = (.096, .022, .082)      # in the head frame (the head frame sits at the neck joint)
EYE_P_FOVY, EYE_F_FOVY = 100, 20   # a default (flagged): periphery 64 px over 100 deg, fovea 32 px over 20 deg
CHILD_MOUTH_N = 5


def child_face():
    hc, r = np.array(C["head_c"]), C["head_r"]
    G = 'group="5"'      # the child's own face and head shell: drawn for the audience, never in its own eyes (World.EYE_GROUPS)
    out = [f'<geom name="child_head_shell" type="mesh" mesh="c_head" pos="{f(*hc)}" material="shell" {G} {VIS0}/>',
           f'<geom name="child_bezel" type="mesh" mesh="c_bezel" pos="{f(hc[0] + .0745, 0, hc[2] + .002)}" material="shell2" {G} {VIS0}/>',
           f'<geom name="child_screen" type="mesh" mesh="c_screen" pos="{f(hc[0] + .0765, 0, hc[2] + .002)}" material="screen" {G} {VIS0}/>']
    for sd, sg in (("L", 1), ("R", -1)):
        out.append(f'<geom name="child_glow_eye_{sd}" type="ellipsoid" pos="{f(.0925, sg * .022, .082)}" size=".0015 .0088 .0118" material="glow" {G} {VIS0}/>')
    for i in range(CHILD_MOUTH_N - 1):
        out.append(f'<geom name="child_mouth{i}" type="capsule" size=".0034 .004" pos=".092 0 .052" material="glow" {G} {VIS0}/>')
    # two ears: coral discs on the sides; the ear sites are where each ear hears from
    for sd, sg in (("L", 1), ("R", -1)):
        out.append(f'<geom name="child_ear_{sd}" type="cylinder" pos="{f(hc[0] - .004, sg * (r - .002), hc[2] + .004)}" zaxis="0 1 0" size=".0125 .004" material="coral" {G} {VIS0}/>')
        out.append(f'<geom type="cylinder" pos="{f(hc[0] - .004, sg * (r + .0015), hc[2] + .004)}" zaxis="0 1 0" size=".008 .0015" material="cjoint" {G} {VIS0}/>')
        out.append(f'<site name="child_ear_{sd}" pos="{f(hc[0] - .004, sg * (r + .004), hc[2] + .004)}" size=".006" group="4"/>')
    return "\n          ".join(out)


def child():
    hc, r = C["head_c"], C["head_r"]
    return f"""
    <!-- ======================= THE CHILD: a humanoid about 12 months in size, born lying on the mat ======================= -->
    <body name="child_pelvis" pos="0 0 .4">
      <freejoint name="child_root"/>
      <geom name="child_pelvis" type="capsule" fromto="-.005 -.03 .004 -.005 .03 .004" size=".058" mass="{MASS['pelvis']}" material="shell" {CHILD}/>
      {touch_site("child_bottom_touch", "box", ".07 .11 .075", (-.03, 0, -.01))}
      <site name="child_pelvis_imu" pos="0 0 0" size=".005" group="4"/>
      <body name="child_abdomen" pos="0 0 {C['lumbar_z']}">
        {jnt("child_lumbar_flex", (0, 1, 0), "lumbar_flex")}
        {jnt("child_lumbar_lat", (1, 0, 0), "lumbar_lat")}
        {jnt("child_lumbar_rot", (0, 0, 1), "lumbar_rot")}
        <geom name="child_abdomen" type="capsule" fromto=".006 -.012 .032 .006 .012 .032" size=".062" mass="{MASS['abdomen']}" material="shell" {CHILD}/>
        {touch_site("child_belly_touch", "box", ".04 .1 .06", (.05, 0, .032))}
        {touch_site("child_lower_back_touch", "box", ".04 .1 .06", (-.04, 0, .032))}
        <body name="child_chest" pos="0 0 {C['thor_z']}">
          {jnt("child_thor_flex", (0, 1, 0), "thor_flex")}
          {jnt("child_thor_lat", (1, 0, 0), "thor_lat")}
          {jnt("child_thor_rot", (0, 0, 1), "thor_rot")}
          <geom name="child_chest" type="capsule" fromto="0 -.015 .052 0 .015 .052" size=".06" mass="{MASS['chest']}" material="shell" {CHILD}/>
          <geom type="cylinder" pos=".058 0 .07" zaxis="1 0 0" size=".016 .004" material="cjoint" {VIS0}/>
          <geom name="child_charge_light" type="cylinder" pos=".0615 0 .07" zaxis="1 0 0" size=".011 .002" material="glow" {VIS0}/>
          <geom type="cylinder" pos="0 0 .118" size=".034 .01" material="coral" {VIS0}/>
          {touch_site("child_chest_touch", "box", ".04 .105 .07", (.045, 0, .052))}
          {touch_site("child_back_touch", "box", ".04 .105 .07", (-.045, 0, .052))}
          <site name="child_prop_L" pos="0 .075 .045" size=".008" group="4"/>
          <site name="child_prop_R" pos="0 -.075 .045" size=".008" group="4"/>
          {child_arm("L")}
          {child_arm("R")}
          <body name="child_head" pos="0 0 {C['neck_z']}">
            {jnt("child_neck_flex", (0, 1, 0), "neck_flex", arm=.002, damp=.03)}
            {jnt("child_neck_lat", (1, 0, 0), "neck_lat", arm=.002, damp=.03)}
            {jnt("child_neck_rot", (0, 0, 1), "neck_rot", arm=.002, damp=.03)}
            <geom type="cylinder" fromto="0 0 -.004 0 0 .03" size=".026" material="cjoint" {VIS0}/>
            <geom name="child_head" type="sphere" pos="{f(*hc)}" size="{r}" mass="{MASS['head']}" material="shell" group="3" {CHILD}/>
            {touch_site("child_head_touch", "sphere", f"{r + .006}", hc)}
            <site name="child_vestibule" pos="{f(hc[0], 0, hc[2])}" size=".006" group="4"/>
            {child_face()}
            <camera name="child_portrait" pos=".36 0 .085" xyaxes="0 1 0 0 0 1" fovy="30"/>
            {child_eye("L")}
            {child_eye("R")}
          </body>
        </body>
      </body>
      {child_leg("L")}
      {child_leg("R")}
    </body>"""


def child_actuators():
    out = []
    fingers_done = set()
    for name, rk, tk, _, _ in JOINTS:
        if tk == "finger":
            sd = name.split("_")[1]
            if sd in fingers_done:
                continue
            fingers_done.add(sd)
            lo, hi = TAU["finger"]
            kp = max(-lo, hi) / .4
            out.append(f'<position name="child_{sd}_fingers" tendon="child_{sd}_fingers" kp="{kp:.4g}" kv="{kp * .03:.4g}" '
                       f'ctrlrange="{math.radians(-10):.4f} {math.radians(195):.4f}" forcelimited="true" forcerange="{lo:.3g} {hi:.3g}"/>')
            continue
        lo, hi = TAU[tk]
        kp = max(-lo, hi) / (KP_ERR if tk not in ("eye", "thumb") else .1)
        a, b = RNG[rk]
        out.append(f'<position name="{name}" joint="{name}" kp="{kp:.4g}" kv="{kp * .04:.4g}" ctrlrange="{math.radians(a):.4f} {math.radians(b):.4f}" '
                   f'forcelimited="true" forcerange="{lo:.3g} {hi:.3g}"/>')
    return "\n    ".join(out)


# ================================================================== THE PARENT (mocap segments)
def parent_segment_geoms(seg):
    """The parent's geoms in a segment's own frame. Collision geoms (bit 8) are the solid shapes; faces, fingers and
    trim are seen only."""
    sd = seg[-1] if seg[-2] == "_" else ""
    sg = 1 if sd == "L" else -1
    m = (lambda p: p) if sd != "R" else mirror_pos
    P = f"parent_{seg}"
    g = []
    if seg == "pelvis":
        g += [f'<geom name="{P}" type="ellipsoid" pos="-.015 0 .035" size=".11 .165 .115" material="denim" {PARENT}/>',
              f'<geom type="ellipsoid" pos="-.005 0 .105" size=".113 .158 .03" material="brow" {DECOR}/>']
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
              f'<geom name="{P}" type="ellipsoid" pos=".015 0 .16" size=".098 .084 .106" material="skin" {PARENT}/>',
              f'<geom name="parent_hair" type="ellipsoid" pos="-.022 0 .172" size=".108 .094 .112" material="hair" {DECOR}/>',
              f'<geom name="parent_fringe" type="ellipsoid" pos=".072 -.012 .226" euler="-18 -40 0" size=".034 .07 .02" material="hair" {DECOR}/>',
              f'<geom name="parent_side_L" type="ellipsoid" pos=".012 .078 .158" euler="0 -12 0" size=".058 .022 .09" material="hair" {DECOR}/>',
              f'<geom name="parent_side_R" type="ellipsoid" pos=".012 -.078 .158" euler="0 -12 0" size=".058 .022 .09" material="hair" {DECOR}/>',
              f'<geom name="parent_bun" type="sphere" pos="-.095 0 .215" size=".047" material="hair" {DECOR}/>',
              f'<geom name="parent_nape" type="ellipsoid" pos="-.045 0 .12" size=".06 .078 .06" material="hair" {DECOR}/>',
              f'<geom type="ellipsoid" pos="-.002 .083 .152" size=".018 .011 .027" material="skin_d" {DECOR}/>',
              f'<geom type="ellipsoid" pos="-.002 -.083 .152" size=".018 .011 .027" material="skin_d" {DECOR}/>',
              f'<geom name="parent_nose" type="ellipsoid" pos=".107 0 .147" size=".011 .0095 .016" material="skin_d" {DECOR}/>',
              f'<geom type="ellipsoid" pos=".093 .052 .132" euler="0 0 32" size=".0015 .011 .007" material="blush" {DECOR}/>',
              f'<geom type="ellipsoid" pos=".093 -.052 .132" euler="0 0 -32" size=".0015 .011 .007" material="blush" {DECOR}/>']
        for s2 in ("L", "R"):
            c = kin.EYE_C[s2]
            g += [f'<geom name="parent_sclera_{s2}" type="ellipsoid" pos="{f(*c)}" size=".0085 .0175 .0135" material="sclera" {DECOR}/>',
                  f'<geom name="parent_iris_{s2}" type="ellipsoid" pos="{f(*c)}" size=".0022 .0098 .0098" material="iris" {DECOR}/>',
                  f'<geom name="parent_pupil_{s2}" type="ellipsoid" pos="{f(*c)}" size=".0016 .0052 .0052" material="pupil" {DECOR}/>',
                  f'<geom name="parent_glint_{s2}" type="sphere" pos="{f(*c)}" size=".0021" material="glint" {DECOR}/>',
                  f'<geom name="parent_lid_lo_{s2}" type="ellipsoid" pos="{f(*c)}" size=".0082 .0195 .0065" material="skin" {DECOR}/>',
                  f'<geom name="parent_lid_up_{s2}" type="ellipsoid" pos="{f(*c)}" size=".0085 .0195 .005" material="skin" {DECOR}/>',
                  f'<geom name="parent_brow_{s2}" type="capsule" size=".0034 .015" material="brow" {DECOR}/>']
        for i in range(kin.FACE_MOUTH_N - 1):
            g.append(f'<geom name="parent_mouth{i}" type="capsule" size=".0048 .006" material="lips" {DECOR}/>')
        g += [f'<geom name="parent_mouth_open" type="ellipsoid" size=".004 .02 .006" material="mouth_in" {DECOR}/>',
              f'<geom name="parent_teeth" type="ellipsoid" size=".003 .015 .002" material="teeth" {DECOR}/>']
    elif seg.startswith("upper_arm"):
        g += [f'<geom name="{P}" type="capsule" fromto="0 0 -.01 0 0 -.285" size=".041" material="sweater" {PARENT}/>']
    elif seg.startswith("forearm"):
        g += [f'<geom name="{P}" type="capsule" fromto="0 0 0 0 0 -.185" size=".0385" material="sweater" {PARENT}/>',
              f'<geom type="cylinder" pos="0 0 -.19" size=".037 .012" material="sweater_d" {DECOR}/>',
              f'<geom type="capsule" fromto="0 0 -.19 0 0 -.245" size=".025" material="skin" {DECOR}/>']
    elif seg.startswith("hand"):
        g += [f'<geom name="{P}_palm" type="ellipsoid" pos="0 0 -.052" size=".043 .0165 .05" material="skin" {DECOR}/>',
              f'<geom name="{P}" type="capsule" fromto="0 0 -.035 0 0 -.115" size=".022" rgba="0 0 0 0" group="3" {PARENT}/>']
        for i in range(4):
            for j in range(2):
                g.append(f'<geom name="parent_f{i}{j}_{sd}" type="capsule" size="{kin.FINGER_R - .0006 * i} .02" material="skin" {DECOR}/>')
        for j in range(2):
            g.append(f'<geom name="parent_t{j}_{sd}" type="capsule" size=".0098 .016" material="skin" {DECOR}/>')
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


PARENT_START = dict(pos=(1.9, -1.5, kin.HIP_Z), yaw=math.radians(150))


def parent():
    pose = kin.Pose(PARENT_START["pos"], kin.rz(PARENT_START["yaw"]))
    segs = kin.fk(pose)
    out = ['\n    <!-- ======================= THE PARENT: kinematic, 16 mocap segments posed by parent_kin.py ======================= -->']
    for s in kin.SEGS:
        p, R = segs[s]
        out.append(f'    <body name="parent_{s}" mocap="true" pos="{f(*p)}" quat="{f(*kin.mjquat(R))}">\n      {parent_segment_geoms(s)}\n    </body>')
    return "\n".join(out)


# ================================================================== THE TOYS
# name: (position on the floor or mat (x, y, yaw deg), sound) -- the sound each makes is a clip of its own (not built here)
TOYS = dict(
    ball=((.30, -.55, 0), "a soft rubber bounce when it lands or is struck"),
    block=((-.42, -.52, 20), "a hollow wooden knock on contact"),
    duck=((.36, .02, -130), "a squeak when squeezed"),
    cup=((-.30, .08, 30), "a plastic clink when it strikes something"),
    rattle=((.05, -.62, 60), "beads shaking, louder with speed"),
    car=((.62, -.30, 200), "a wheel rattle while it moves"),
    bear=((-.62, -.05, -40), "a soft bell inside when it moves"),
    stacker=((-.66, -.78, 0), "the rings' plastic clack"),
    drum=((1.05, .45, 0), "a boom when hit on top"),
    ring=((.55, -.85, 0), "a crinkle when handled"),
)


def toy_geoms(k):
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
    raise KeyError(k)


def toys():
    out = []
    for k, ((x, y, yaw), snd) in TOYS.items():
        z0, g = toy_geoms(k)
        base = MAT_T if abs(x - MAT_CENTER[0]) < MAT_HALF and abs(y - MAT_CENTER[1]) < MAT_HALF else 0
        z = base + z0
        out.append(f'    <body name="toy_{k}" pos="{f(x, y, z + .001)}" euler="0 0 {yaw}"><freejoint name="toy_{k}"/>\n      '
                   + "\n      ".join(g) + "\n    </body>")
    return "\n".join(out)


# ================================================================== THE ROOM
def room():
    W, D, Hh = ROOM_X, ROOM_Y, ROOM_H
    g = []
    g.append(f'<geom name="floor" type="plane" size="{W} {D} .1" material="floor" {WORLD}/>')
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
    g += [f'<geom type="cylinder" pos="{lx} {ly} .012" size=".14 .012" material="metal" {DECOR}/>',
          f'<geom type="capsule" fromto="{lx} {ly} .02 {lx} {ly} 1.45" size=".012" material="metal" {DECOR}/>',
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


def play_mat():
    cx, cy = MAT_CENTER
    g = [f'<geom name="mat" type="box" pos="{cx} {cy} {MAT_T / 2}" size="{MAT_HALF} {MAT_HALF} {MAT_T / 2}" material="mat_edge" '
         f'friction="1 .01 .001" solref=".03 1" solimp=".85 .95 .004" {WORLD}/>']
    n = 4
    tw = 2 * MAT_HALF / n
    for i in range(n):
        for j in range(n):
            x = cx - MAT_HALF + (i + .5) * tw
            y = cy - MAT_HALF + (j + .5) * tw
            g.append(f'<geom type="box" pos="{x:.4g} {y:.4g} {MAT_T / 2 + .0004:.4g}" size="{tw / 2 - .002:.4g} {tw / 2 - .002:.4g} {MAT_T / 2:.4g}" '
                     f'material="{"tile_a" if (i + j) % 2 == 0 else "tile_b"}" {DECOR}/>')
    # the charge pad (a default: a low white disc with a soft mint glow at the mat's corner; its law is not built here)
    px, py = cx + MAT_HALF - .2, cy + MAT_HALF - .2
    g.append(f'<geom name="pad" type="cylinder" pos="{px:.4g} {py:.4g} {MAT_T + .004:.4g}" size=".13 .004" material="pad" {DECOR}/>')
    g.append(f'<geom type="mesh" mesh="pad_ring" pos="{px:.4g} {py:.4g} {MAT_T + .006:.4g}" material="pad_ring" {DECOR}/>')
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


def build():
    (HERE / "textures").mkdir(exist_ok=True)
    floor_texture(HERE / "textures" / "room_floor_oak.png")
    sky_texture(HERE / "textures" / "room_window_sky.png")
    mats = "\n    ".join(f'<material name="{k}" ' + " ".join(f'{a}="{v}"' for a, v in d.items()) + "/>" for k, d in MAT.items())
    JOINTS.clear(); TOUCH.clear()
    child_xml = child()
    cams = dict(room=((2.35, -2.05, 1.62), (-.2, .15, .45), 50), mat=((1.25, -1.55, .95), (0, -.2, .28), 38),
                top=((0, -.25, 2.4), (0, -.249, 0), 45))
    cam_xml = "\n    ".join(f'<camera name="{k}" pos="{f(*p)}" xyaxes="{lookat_xyaxes(p, t)}" fovy="{fv}"/>' for k, (p, t, fv) in cams.items())
    welds = []
    for hs in ("L", "R"):
        for k in TOYS:
            welds.append(f'<weld name="hold_{hs}_{k}" body1="parent_hand_{hs}" body2="toy_{k}" active="false" solref=".006 1"/>')
        welds.append(f'<weld name="prop_{hs}" body1="parent_hand_{hs}" body2="child_chest" torquescale="0" active="false" solref=".04 1" solimp=".9 .95 .001"/>')
        for cs in ("L", "R"):
            welds.append(f'<weld name="guide_{hs}_{cs}" body1="parent_hand_{hs}" body2="child_{cs}_forearm" torquescale="0" active="false" solref=".02 1"/>')
    tendons = []
    for sd in ("L", "R"):
        tendons.append(f'<fixed name="child_{sd}_fingers"><joint joint="child_{sd}_finger1" coef="1"/><joint joint="child_{sd}_finger2" coef="1"/></fixed>')
    sensors = [f'<touch name="{t[:-6] if t.endswith("_touch") else t}" site="{t}"/>' for t in TOUCH]
    sensors += ['<accelerometer name="child_vestibule_acc" site="child_vestibule"/>', '<gyro name="child_vestibule_gyro" site="child_vestibule"/>',
                '<framequat name="child_head_quat" objtype="site" objname="child_vestibule"/>']
    sounds = "\n    ".join(f'<text name="sound_{k}" data="{snd}"/>' for k, (_, snd) in TOYS.items())
    xml = f"""<!-- GENERATED by body/sim/make_livingroom_customchild.py (from the 2026-09-24 prototype): the custom child (kept for reference), the parent and the living room. Do not hand-edit.
     Units m, kg, s; angles in degrees in this file. Floor z = 0; the room spans x +-{ROOM_X}, y +-{ROOM_Y}, {ROOM_H} m high.
     Collision bits: world 1, child 2 (self-collision on), toys 4, parent 8 (mocap: touches the child and toys only). -->
<mujoco model="livingroom">
  <compiler angle="degree" texturedir="textures" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="pyramidal"/>
  <size memory="64M"/>
  <statistic center="0 -.2 .3" extent="2.5"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="45"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient=".32 .32 .33" diffuse=".18 .18 .18" specular=".02 .02 .02"/>
    <map znear=".003" zfar="30" shadowclip="2.6" shadowscale=".6"/>
  </visual>
  <default>
    <geom solref=".015 1" solimp=".9 .95 .001"/>
  </default>
  <asset>
    <texture name="floor_oak" type="2d" file="room_floor_oak.png"/>
    <texture name="window_sky" type="2d" file="room_window_sky.png"/>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1=".92 .94 .97" rgb2=".75 .78 .82" width="256" height="256"/>
    {mats}
    <mesh name="c_head" builtin="supersphere" params="48 .45 .45" scale=".078 .076 .08"/>
    <mesh name="c_palm" builtin="supersphere" params="24 .45 .45" scale=".0235 .013 .023"/>
    <mesh name="c_finger1" builtin="supersphere" params="24 .45 .45" scale=".0222 .0108 .0128"/>
    <mesh name="c_finger2" builtin="supersphere" params="24 .5 .5" scale=".0212 .0098 .0112"/>
    <mesh name="c_bezel" builtin="supersphere" params="32 .3 .3" scale=".006 .063 .054"/>
    <mesh name="c_screen" builtin="supersphere" params="32 .28 .28" scale=".0055 .058 .049"/>
    <mesh name="cup_rim" builtin="supertorus" params="40 .10 1 1" scale=".039 .039 .039"/>
    <mesh name="cup_handle" builtin="supertorus" params="30 .24 1 1" scale=".02 .02 .02"/>
    <mesh name="drum_rim" builtin="supertorus" params="48 .06 1 1" scale=".076 .076 .076"/>
    <mesh name="teether" builtin="supertorus" params="40 .30 1 1" scale=".05 .05 .05"/>
    <mesh name="pad_ring" builtin="supertorus" params="48 .05 1 1" scale=".13 .13 .13"/>
    <mesh name="ring0" builtin="supertorus" params="40 .40 1 1" scale=".05 .05 .05"/>
    <mesh name="ring1" builtin="supertorus" params="40 .40 1 1" scale=".044 .044 .044"/>
    <mesh name="ring2" builtin="supertorus" params="40 .40 1 1" scale=".038 .038 .038"/>
    <mesh name="ring3" builtin="supertorus" params="40 .40 1 1" scale=".032 .032 .032"/>
    <mesh name="ring4" builtin="supertorus" params="40 .40 1 1" scale=".026 .026 .026"/>
    <mesh name="cushion" builtin="supersphere" params="24 .3 .3" scale=".28 .40 .07"/>
    <mesh name="back_cushion" builtin="supersphere" params="24 .35 .35" scale=".28 .09 .24"/>
    <mesh name="pillow" builtin="supersphere" params="24 .45 .5" scale=".20 .07 .19"/>
    <mesh name="table_top" builtin="supersphere" params="32 .25 .2" scale=".58 .30 .035"/>
    <mesh name="basket" builtin="supersphere" params="24 .2 .15" scale=".17 .17 .14"/>
  </asset>
  <custom>
    <numeric name="child_torque_scale" data="{SCALE}"/>
    {sounds}
  </custom>
  <worldbody>
    <light name="sun" type="directional" pos="-2 -.1 3" dir=".62 .18 -.76" castshadow="true" diffuse=".30 .285 .25" specular=".08 .08 .08"/>
    <light name="key" type="spot" pos="1.6 -1.8 2.3" dir="-.5 .6 -.62" cutoff="45" exponent="2" castshadow="true" bulbradius=".2" diffuse=".42 .41 .40" specular=".12 .12 .12"/>
    <light name="fill" type="directional" pos="0 0 2.5" dir="-.2 .5 -.84" castshadow="false" diffuse=".22 .22 .23" specular="0 0 0"/>
    <!-- (a ceiling spot light was tried and removed: in MuJoCo 3.9's renderer on this Mac it blacked out the ceiling seen
         from below, which is most of what the newborn sees) -->
    {cam_xml}
    {room()}
    {play_mat()}
{toys()}
{child_xml}
{parent()}
  </worldbody>
  <contact>
    <!-- the parent's hand proxies (one capsule each) never push on the child's trunk: when a hand holds the chest, the
         prop weld is the grip (a capsule cannot wrap around a ribcage the way fingers do). World.hold_mode() also switches
         a hand's proxy off while it holds anything through a weld. -->
    <exclude body1="parent_hand_L" body2="child_chest"/>
    <exclude body1="parent_hand_R" body2="child_chest"/>
    <exclude body1="parent_hand_L" body2="child_abdomen"/>
    <exclude body1="parent_hand_R" body2="child_abdomen"/>
  </contact>
  <tendon>
    {chr(10).join("    " + t for t in tendons).strip()}
  </tendon>
  <equality>
    {chr(10).join("    " + w for w in welds).strip()}
  </equality>
  <actuator>
    {child_actuators()}
  </actuator>
  <sensor>
    {chr(10).join("    " + s for s in sensors).strip()}
  </sensor>
</mujoco>
"""
    out = HERE / "livingroom_customchild.xml"
    out.write_text(xml)
    return out


if __name__ == "__main__":
    p = build()
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(p))
    print(p, "nq", m.nq, "nv", m.nv, "nu", m.nu, "ngeom", m.ngeom, "nbody", m.nbody, "nmocap", m.nmocap, "neq", m.neq,
          "nsensor", m.nsensor, "child mass", round(float(m.body_subtreemass[m.body("child_pelvis").id]), 3))
