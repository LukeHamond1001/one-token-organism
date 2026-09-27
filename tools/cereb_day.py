"""THE CEREBELLUM OVER A LIFE DAY ON THE TEST LIMB (an instrument of step R6c; docs/SIM_DESIGN.md 7.5, C50): the test limb of
body/tests/test_cerebellum.py (a MuJoCo arm of two hinges in a vertical plane under the servo law, the instrument's driver moving it
through four postures in turn, a 0.5 kg weight added to its hand at tick 400) lives 24,000 ticks with the cerebellum off, or on with the
mossy numbers of each kind named (body/tests/test_cerebellum.py MOSSY_KINDS: "state", each joint's angle, velocity and servo target,
which the tests declare; "design", 7.5's list, adding each joint's estimated torque; "efference", the targets and the tick's steps;
"efference+angle", "efference+velocity", "efference+torque"), and prints the servo's corrective torque at the shoulder per 2,000 ticks,
the elbow's over the last 2,000, and the largest Purkinje weight. Every number is the body's CEREB unless --rate gives the limbs'
rate or --leak their leak. Measured 2026-09-24 (N m at the shoulder, off 4.8), under the pure law (--leak 0): state about 1.0 all
day; efference 1.1 -> 1.6; efference+angle and efference+velocity 1.0 -> 1.5; design unstable from about tick 2,000 (22 at the limit),
at --rate 0.002 from about tick 14,000; efference+torque unstable from about tick 4,000. Under the leak (CEREB's 0.0033, after the R6c
verifier): off 4.80; state 1.70 -> 1.83 (largest weight 0.023); design 1.73 -> 1.83 (0.048); efference+torque 1.73 -> 1.83 (0.130),
each flat all day. Run at nice 19, one thread:
  nice -n 19 python3 tools/cereb_day.py [--ticks 24000] [--rate 0.01] [--leak 0.0033] off state design ...
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from body.core.physiology import CEREB  # noqa: E402
from body.tests.test_cerebellum import MOSSY_KINDS, ArmWorld, OrganHook, arm_cerebellar, organ, run_limb  # noqa: E402


def day(kind, ticks, rate, leak):
    w = ArmWorld(mossy=kind if kind != "off" else "state")
    o = None
    if kind != "off":
        o = organ(arm_cerebellar(kind), seed=1); w.below = OrganHook(o, rate=rate, leak=leak)
    t0 = time.time()
    run_limb(w, ticks, load_at=400)
    x = np.array(w.teach_log)[:, 0]; y = np.array(w.teach_log)[:, 1]
    per = " ".join(f"{x[i:i + 2000].mean():.3f}" for i in range(0, ticks, 2000))
    wmax = 0.0 if o is None else float(o.pc_w.abs().max())
    print(f"{kind:20s} shoulder per 2000 ticks (N m): {per} | elbow last 2000 {y[-2000:].mean():.3f} | largest weight {wmax:.3f} | {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    args = sys.argv[1:]; ticks = 24000; rate = CEREB["cereb_rate"]; leak = CEREB["cereb_leak"]; kinds = []
    i = 0
    while i < len(args):
        if args[i] == "--ticks":
            ticks = int(args[i + 1]); i += 2
        elif args[i] == "--rate":
            rate = float(args[i + 1]); i += 2
        elif args[i] == "--leak":
            leak = float(args[i + 1]); i += 2
        elif args[i] == "off" or args[i] in MOSSY_KINDS:
            kinds.append(args[i]); i += 1
        else:
            sys.exit(f"cereb_day: unknown argument {args[i]!r}\n" + __doc__)
    for k in kinds or ["off", "state", "design"]:
        day(k, ticks, rate, leak)
