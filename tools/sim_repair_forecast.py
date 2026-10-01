"""THE FORECAST HEADS SCALED BACK (C193, the lead, 2026-10-01): on the night of life day 50 the reel's lesson against (A152's negative weights on
the squared-distance error, before A153 bounded it) drove every later effector's forecast norm from about 2.3 to some 3,600; the readout
(sharpness x cos x |P|^g, C148) then gave one setting a probability of 1 on every joint and the arms stood frozen at dawn 51. This tool loads
a pair (never the life's while it runs), measures each later effector's forecast norm over a few ticks, and scales its forecast head
(act_pred's `pred` and its correction `cor`, the lesson's two targets) by target / measured where the measured norm stands above the
target, then saves the life. The target is the forecast scale the body held for weeks (about 2: 1.3 to 2.0 on the day-27 pair, 2.3 on dawn 50), before the night that blew it up: a repair of
the lead's own defect's damage, not a lesson undone. Run: sim_repair_forecast.py DIR [--target 2.0] [--dry]"""
import argparse, os, sys, collections, importlib.util, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("sl", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_life.py")); sl = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(sl)
except SystemExit:
    pass
from body import model as M

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--target", type=float, default=2.0); ap.add_argument("--dry", action="store_true"); ap.add_argument("--ticks", type=int, default=12)
args = ap.parse_args()
torch.set_num_threads(2)
ns = argparse.Namespace(out=args.dir, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="a", body="a", no_eyes=False, set=[], lr0=False,
                        page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=2)
from body.core.world import WorldLoop
# 1. measure on one load, a few ticks lived (a pair saved at dawn has an empty window: nothing to forecast from until it lives)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
norms = collections.defaultdict(list); orig = M.ActTable.logits
tabs = {id(L.m.get_submodule(e_.organ)): e_.name for e_ in L.anatomy.motors}
def rec(self, pred, sharp, earned=None):
    if pred is not None and id(self) in tabs:
        norms[tabs[id(self)]].append(float(pred.norm()))
    return orig(self, pred, sharp, earned)
M.ActTable.logits = rec
run = WorldLoop(L)
for _ in range(args.ticks):
    run.step()
M.ActTable.logits = orig
t_meas = int(world.tick)
scale = {}
for e_ in L.anatomy.motors:
    v = sorted(norms.get(e_.name, [])); med = v[len(v) // 2] if v else None
    scale[e_.name] = ((args.target / med) if (med is not None and med > args.target) else 1.0, med)
del run, L, world, eyes, lane
print(f"measured over {args.ticks} ticks from the pair (to tick {t_meas}): the forecast norms (median) and the scale each head takes:")
for n_, (s_, med) in scale.items():
    print(f"  {n_:7s} |P| {med if med is not None else float('nan'):9.2f} -> scale {s_:.4f}")
if args.dry:
    print("dry run: nothing saved"); sys.exit(0)
# 2. a fresh load of the same pair (no tick lived), the heads scaled, the pair saved at its own tick
world, eyes, lane, L = sl.build(ns)
scaled = {}
with torch.no_grad():
    for e_ in L.anatomy.motors:
        s_ = scale[e_.name][0]
        if s_ < 1.0:
            tm = L.m.timing[e_.name]
            for mod in (tm.pred, tm.cor):
                mod.weight.mul_(s_)
                if mod.bias is not None:
                    mod.bias.mul_(s_)
            scaled[e_.name] = round(s_, 5)
if args.dry:
    print("dry run: nothing saved"); sys.exit(0)
if scaled:
    L.save()                                                            # life.pt (and world.pt beside it, as the runner saves them)
    print(f"saved: {len(scaled)} forecast heads scaled back to {args.target}: {scaled}")
else:
    print("nothing above the target: nothing saved")
