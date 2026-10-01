"""EXTRAS FOR THE ROOM (A115, 2026-09-27; docs/SIM_DESIGN.md C99, A28's novel toys): things added to the world at `g1scene.load_model`'s
`extra` hook, never to the room file, so a life born in the room can meet them later (tools/sim_migrate_world.py carries a saved
world across). Each is a function of the MjSpec that adds bodies before the compile.

THE BOOK (the first transfer test): a small board book, a black box 16 x 12 x 2.4 cm of 150 g, lying flat, with a white page edge.
A toy of a kind the child has never seen (its word "book" is in her growth queue, templates.GROWTH; black is no colour word of hers,
so no colour comes in with it: the test is the reach, not a word). Its contact attributes are the toys' (make_g1room.TOY), its friction the block's, its sound a soft slap (sounds.KINDS
"book"). Named `toy_book` with a free joint `toy_book`, as every toy is, so the lane, the parent and the sounds find it by name."""
import mujoco
import numpy as np

BOOK_HALF = (0.080, 0.060, 0.012)             # m: half-sizes of a board book, 16 x 12 x 2.4 cm (ours: a first board book)
BOOK_MASS = 0.15                              # kg (a board book of that size)
BOOK_RGBA = (0.05, 0.05, 0.06, 1.0)           # make_g1room's t_black (2026-09-27 17:45: a red book made "red" a colour word with two kinds,
                                              # B2's condition, and the colour line's stress fault C101 showed; the motor test wants no new word)
BOOK_EDGE_RGBA = (0.97, 0.97, 0.96, 1.0)      # the pages' edge, make_g1room's t_white
WORLD_PRIORITY = 2                            # make_g1room.WORLD_PRIORITY: the toys' contact priority


def add_book(xy=(0.0, -0.60), yaw_deg=0.0):
    """-> extra(spec): the book laid flat at xy on the mat (its centre 1.2 cm up plus the mat's 1.2 cm), turned yaw_deg"""
    def extra(spec):
        b = spec.worldbody.add_body(name="toy_book", pos=[float(xy[0]), float(xy[1]), 0.012 + BOOK_HALF[2] + 0.001])
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), np.radians(float(yaw_deg)))
        b.quat = q.tolist()
        b.add_freejoint(name="toy_book")
        g = b.add_geom(name="book", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(BOOK_HALF), mass=BOOK_MASS, rgba=list(BOOK_RGBA))
        g.contype = 4; g.conaffinity = 15; g.priority = WORLD_PRIORITY          # make_g1room.TOY
        g.friction = [1.0, 0.01, 0.001]                                          # the block's
        g.solref = [0.015, 1.0]; g.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]          # the room class's softness (default class "room")
        e = b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[BOOK_HALF[0] - 0.004, 0.0, 0.0], size=[0.0035, BOOK_HALF[1] - 0.006, BOOK_HALF[2] - 0.002],
                       rgba=list(BOOK_EDGE_RGBA))
        e.contype = 0; e.conaffinity = 0; e.density = 0.0                        # seen only (make_g1room.VIS0)
        for hs in ("L", "R"):                                                    # her hold of it: the weld each toy has (make_g1room's
            w = spec.add_equality()                                              # hold_{hand}_{toy}, inactive until her grasp closes;
            w.type = mujoco.mjtEq.mjEQ_WELD; w.objtype = mujoco.mjtObj.mjOBJ_BODY; w.name = f"hold_{hs}_book"          # parent_motion's scene.weld); life dawn 15's first
            w.name1 = f"parent_hand_{hs}"; w.name2 = "toy_book"                  # attempt crashed at tick 1 without them
            w.active = False; w.solref = [0.006, 1.0]
        spec.add_text(name="sound_book", data="a soft slap when it lands or is struck")   # the toy's sound described (make_g1room's texts)
    return extra


BOX_HALF = 0.045                              # m: a small cardboard box, a 9 cm cube (ours: a second novel toy, a shape unlike the book's slab)
BOX_MASS = 0.12                               # kg (a small filled cardboard box)
BOX_RGBA = (0.62, 0.58, 0.52, 1.0)            # cardboard grey-brown; "grey" in her inventory: no colour word of hers (templates.COLOURS
                                              # are blue, green, red, yellow), and no other kind of that colour (C101)


def add_box(xy=(0.3, -0.4), yaw_deg=0.0):
    """-> extra(spec): the box set at xy on the mat (A121, the second novel toy: its time to the child's first grasp against the
    book's 592 ticks), with the toys' contacts, its two hold welds and its sound text, as the book has"""
    def extra(spec):
        b = spec.worldbody.add_body(name="toy_box", pos=[float(xy[0]), float(xy[1]), 0.012 + BOX_HALF + 0.001])
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), np.radians(float(yaw_deg)))
        b.quat = q.tolist()
        b.add_freejoint(name="toy_box")
        g = b.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BOX_HALF] * 3, mass=BOX_MASS, rgba=list(BOX_RGBA))
        g.contype = 4; g.conaffinity = 15; g.priority = WORLD_PRIORITY          # make_g1room.TOY
        g.friction = [1.0, 0.01, 0.001]                                          # the block's
        g.solref = [0.015, 1.0]; g.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
        for hs in ("L", "R"):
            w = spec.add_equality()
            w.type = mujoco.mjtEq.mjEQ_WELD; w.objtype = mujoco.mjtObj.mjOBJ_BODY; w.name = f"hold_{hs}_box"
            w.name1 = f"parent_hand_{hs}"; w.name2 = "toy_box"
            w.active = False; w.solref = [0.006, 1.0]
        spec.add_text(name="sound_box", data="a hollow cardboard knock when it lands or is struck")
    return extra


BUCKET_IN = 0.09                                 # m: the bucket's inner half-width, an 18 cm square (ours: the duck's 7 cm, a G1 hand and a miss fit; A126)
BUCKET_WALL = 0.005                              # m: the wall's thickness (a thin plastic bucket)
BUCKET_H = 0.08                                  # m: the wall's height. C179 (2026-10-01): 0.12 until life day 48; a 7 cm toy inside lies 1 cm under the rim,
                                                 # hidden from a child beside it; the child's hand at the bucket rose to 6 cm under the old rim and never over
                                                 # it (days 44 to 48: the search where the toy vanished, the find denied by the wall), so the rim comes down to it
                                              # mat (eyes at 0.1 m prone, 0.35 m sitting, half a metre off: the near wall hides the whole floor)
BUCKET_MASS = 0.60                               # kg. C179 (2026-10-01): 0.20 until life day 48, when a touch of the child's hand tipped it (it lay on its
                                                 # side at day 47's tick 22,000, the hides spilled); a weighted base stays up under a hand's push (ours)
BUCKET_RGBA = (0.45, 0.55, 0.20, 1.0)            # olive: no colour word of hers, no other kind of it (the box's road: no colour lesson rides in)
BUCKET_YAW_DEG = 0.0


def add_bucket(xy=(-0.3, -0.45), yaw_deg=BUCKET_YAW_DEG):
    """-> extra(spec): the bucket set at xy on the mat (A126, the third novel object and the first hollow one: a toy can be put INTO it in
    the child's view and be gone from its eyes while it stays, the hide game's vessel, A129), with the toys' contacts, its two hold welds
    and its sound text, as the box has. An open-top box, a bucket: a floor plate and four walls, BUCKET_IN inside, BUCKET_H high"""
    def extra(spec):
        b = spec.worldbody.add_body(name="toy_bucket", pos=[float(xy[0]), float(xy[1]), 0.012 + BUCKET_WALL + 0.001])
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), np.radians(float(yaw_deg)))
        b.quat = q.tolist()
        b.add_freejoint(name="toy_bucket")
        out = BUCKET_IN + BUCKET_WALL
        parts = [("bucket", [out, out, BUCKET_WALL], [0.0, 0.0, 0.0])]                                     # the floor plate (the body's origin)
        for i, (sx, sy, px, py) in enumerate(((BUCKET_WALL, out, out - BUCKET_WALL, 0.0), (BUCKET_WALL, out, -(out - BUCKET_WALL), 0.0),
                                              (out, BUCKET_WALL, 0.0, out - BUCKET_WALL), (out, BUCKET_WALL, 0.0, -(out - BUCKET_WALL)))):
            parts.append((f"bucket_wall{i}", [sx, sy, BUCKET_H / 2.0], [px, py, BUCKET_WALL + BUCKET_H / 2.0]))
        for name, size, pos in parts:
            g = b.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, mass=BUCKET_MASS / len(parts), rgba=list(BUCKET_RGBA))
            g.contype = 4; g.conaffinity = 15; g.priority = WORLD_PRIORITY                          # make_g1room.TOY
            g.friction = [1.0, 0.01, 0.001]                                                          # the block's
            g.solref = [0.015, 1.0]; g.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
        for hs in ("L", "R"):
            w = spec.add_equality()
            w.type = mujoco.mjtEq.mjEQ_WELD; w.objtype = mujoco.mjtObj.mjOBJ_BODY; w.name = f"hold_{hs}_bucket"
            w.name1 = f"parent_hand_{hs}"; w.name2 = "toy_bucket"
            w.active = False; w.solref = [0.006, 1.0]
        spec.add_text(name="sound_bucket", data="a hollow plastic knock when it lands or is struck")
    return extra


def add_book_box(xy=(0.3, -0.4), yaw_deg=0.0):
    """-> extra(spec): the room with the book (where the model compiles it; a carried world puts it where its save has it) and the
    box at xy: the runner's --extra for a life that has met both (A121)"""
    book, box = add_book(), add_box(xy=xy, yaw_deg=yaw_deg)

    def extra(spec):
        book(spec); box(spec)
    return extra


def add_book_box_bucket(xy=(-0.3, -0.45), yaw_deg=BUCKET_YAW_DEG):
    """-> extra(spec): the room with the book and the box where the model compiles them (a carried world puts them where its save has
    them) and the bucket at xy: the runner's --extra for a life that has met all three (A126)"""
    both, bucket = add_book_box(), add_bucket(xy=xy, yaw_deg=yaw_deg)

    def extra(spec):
        both(spec); bucket(spec)
    return extra


EXTRAS = {"book": add_book, "box": add_box, "book_box": add_book_box, "bucket": add_bucket, "book_box_bucket": add_book_box_bucket}   # the runner's --extra names
NEW_TOY = {"book": "book", "box": "box", "book_box": "box", "bucket": "bucket", "book_box_bucket": "bucket"}   # the toy each extra brings in (the migration tool's report)
