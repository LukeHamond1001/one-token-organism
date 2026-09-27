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
    return extra


EXTRAS = {"book": add_book}                   # the runner's --extra names
