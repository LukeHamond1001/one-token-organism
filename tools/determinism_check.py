"""THE BEHAVIOUR GUARD (a supervisor's instrument, 2026-09-17; the served profiles, the full state and the round trip 2026-09-23, review
section 4): a tiny body born at a fixed seed lives a fixed script (lines typed as the parents type them, faces at fixed ticks), sleeps
one night and lives on; every weight, the store, the utterance memory, the page and the feelings are hashed. Run before and after an
edit of body/: an equal digest means the edit changed nothing the body does on that script; a different digest means it did. Imports
the body from its own tree, so it can guard a worktree. Run from the main tree (the flags' relative paths: data/stories_valid.txt).

Profiles (--profile):
  default   the flags alone, then the tiny night; the night called by hand (the guard as it was; its digest is unchanged)
  served    the served save's own constants first (read from the save's pickle, its tensors never read: the 26 constants the flags
            leave to the save), then the flags, then the tiny night; the tiny day ends at the tick's own sleep switch
            (wake_ticks = --ticks), so the served dopamine, ventral critic, actor, word-level mouth, ear and switch all run
  switches  served, plus the switches not yet served: chunk_gate 1, utt_entry felt, pace_fore_q 0.99
Options:
  --full       the digest also takes every optimizer's state, the random streams, the organs' every buffer, gradient and plain
               attribute, the store's every field, the whole saved blob (every counter in the saved life dict, the critics' float64
               matrices, the striatum, the pace trackers, the utterance memory, the feelings) and every working attribute of the
               life; a digest per section is printed beside it, so a change can be placed
  --roundtrip  halfway through the day the life is saved to a temporary file, loaded as a reload loads it (the save's own
               constants) and lives on; the night then saves there too
  --save PATH  the served save whose constants the profile takes (default: the --load of ops/serve_command.txt, else data/watch2.pt)
  --tmp DIR    where the round trip's and --full's temporary saves go (default: the system's temporary directory); removed after
usage: python3 tools/determinism_check.py [--flags ops/BASE_FLAGS.txt] [--ticks 400] [--profile default|served|switches] [--full]
       [--roundtrip] [--save data/watch2.pt] [--tmp DIR]"""
import sys, os, io, hashlib, pickle, zipfile, tempfile, shutil, collections
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import torch
from tokenizers import Tokenizer
from body.life import Life, PHYSIOLOGY
MAIN = "/Users/lukehamond/Projects/project"

def arg(name, default):
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--" + name and i + 1 < len(sys.argv): return type(default)(sys.argv[i + 1])
    return default
def has(name):
    return ("--" + name) in sys.argv[1:]
def parse_flags(path):
    toks = open(path).read().split(); cfg = {}; i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            k = t[2:].replace("-", "_")
            if k in PHYSIOLOGY: cfg[k] = type(PHYSIOLOGY[k])(toks[i + 1])
            i += 2
        else: i += 1
    return cfg

# ---------------- the served save's constants, read from its pickle with every tensor left unread (as tools/inproc_parent.py save_meta) ----------------
class _Stub:
    def __new__(cls, *a, **k): return object.__new__(cls)
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass
class _NoTensors(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.split(".")[0] in ("torch", "numpy"): return _Stub
        return super().find_class(mod, name)
    def persistent_load(self, pid): return None
def save_cfg(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n == "data.pkl" or n.endswith("/data.pkl"))
        blob = _NoTensors(io.BytesIO(z.read(name))).load()
    return dict(blob.get("cfg") or {})
def served_save():
    for root in (os.getcwd(), MAIN):
        try:
            toks = open(os.path.join(root, "ops", "serve_command.txt")).read().split()
            return os.path.join(root, toks[toks.index("--load") + 1])
        except Exception:
            pass
    return os.path.join(MAIN, "data", "watch2.pt")

_i = 1                                           # an argument not known stops the check: one ignored would hand back another profile's digest
while _i < len(sys.argv):
    _a = sys.argv[_i]
    if _a in ("--full", "--roundtrip"): _i += 1
    elif _a in ("--flags", "--ticks", "--profile", "--save", "--tmp", "--night_dev") and _i + 1 < len(sys.argv): _i += 2
    else: sys.exit(f"determinism_check: unknown argument {_a!r}\n" + __doc__.split("usage: ")[1])
profile = arg("profile", "default"); FULL = has("full"); ROUNDTRIP = has("roundtrip")
assert profile in ("default", "served", "switches"), f"unknown profile {profile!r}"
flags = arg("flags", os.path.join(HERE, "ops", "BASE_FLAGS.txt")); ticks = arg("ticks", 400)
save_only = None
if profile == "default":
    cfg = parse_flags(flags) if os.path.exists(flags) else {}
else:
    src = arg("save", "") or served_save()
    sc = save_cfg(src)
    # the save's keys the physiology no longer knows are read by nothing (Life.load passes them and prints them): left out
    cfg = {k: v for k, v in sc.items() if k in PHYSIOLOGY}
    cfg.setdefault("gate_int_form", "value")                       # as Life.load: an older save keeps the value form
    fl = parse_flags(flags) if os.path.exists(flags) else {}
    save_only = sorted(k for k, v in cfg.items() if k not in fl and v != PHYSIOLOGY[k])
    cfg.update(fl)                                                  # as Life.load: the save's constants, then the flags
    if profile == "switches":
        cfg.update(dict(chunk_gate=1, utt_entry="felt", pace_fore_q=0.99))
cfg.update(dict(night_starts=64, night_starts_max=64, night_rounds=2, night_batch=8, rem_dreams=4, rem_steps=4))
if profile != "default":
    cfg["wake_ticks"] = int(ticks)                                  # the tiny day ends at the tick's own sleep switch
cfg["night_dev"] = str(arg("night_dev", ""))   # the guard stays a CPU guard whatever the flags carry; --night_dev mps makes it a device smoke test   # a night a tiny body can afford
consts = hashlib.sha256(repr(sorted((k, type(v).__name__, repr(v)) for k, v in cfg.items())).encode()).hexdigest()[:12]
if save_only is not None:
    print(f"profile {profile}: the constants of {src} ({len(save_only)} left to the save by the flags), then {os.path.relpath(flags) if os.path.exists(flags) else flags}, then the tiny night", flush=True)
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
torch.manual_seed(0)
life = Life.birth(TOK, device="cpu", d=64, layers=2, heads=2, window=32, cfg=cfg, seed=0)
SCRIPT = [("what do you want?", "parent"), ("I want milk", "other"), ("do you see the ball?", "parent"), ("yes. the ball is red", "other"),
          ("what is cold?", "parent"), ("ice is cold", "other"), ("shall we go out?", "parent"), ("no. it is wet out", "other"),
          ("what do bees make?", "parent"), ("bees make honey", "other"), ("are you warm now?", "parent"), ("I am warm here", "other")]
FACES = {23: 2.0, 24: 2.0, 25: 0.0, 61: -2.0, 62: 0.0, 140: 2.0, 141: 4.0, 142: 0.0, 230: 2.0, 231: 0.0, 333: -2.0, 334: 0.0}
def run(n, t0=0):
    """ticks t0 .. t0+n-1 of a stretch of the script: a line every 30 ticks from the stretch's tick 0 (so the morning after the night
    hears the first lines again), the faces at their ticks; a stretch cut by the round trip goes on from t0 as if it had not been"""
    for t in range(t0, t0 + n):
        if t % 30 == 0 and t // 30 < len(SCRIPT):
            life.type_text(SCRIPT[t // 30][0], who=SCRIPT[t // 30][1])
        if t in FACES: life.set_face(FACES[t])
        life.tick()
def canon(g, x, path):
    """every value fed to the hash with its type and shape; anything of a kind not listed stops the check"""
    if torch.is_tensor(x):
        t = x.detach().cpu().contiguous()
        g.update(f"T{t.dtype}{tuple(t.shape)}".encode()); g.update(t.numpy().tobytes())
    elif isinstance(x, dict):
        g.update(b"{")
        for k in sorted(x, key=repr): g.update(repr(k).encode() + b":"); canon(g, x[k], f"{path}.{k}")
        g.update(b"}")
    elif isinstance(x, collections.deque):
        g.update(f"Q{x.maxlen}[".encode())
        for i, v in enumerate(x): canon(g, v, f"{path}[{i}]")
        g.update(b"]")
    elif isinstance(x, (list, tuple)):
        g.update(b"[" if isinstance(x, list) else b"(")
        for i, v in enumerate(x): canon(g, v, f"{path}[{i}]")
        g.update(b"]")
    elif isinstance(x, (set, frozenset)):
        g.update(b"S"); canon(g, sorted(x, key=repr), path)
    elif x is None or isinstance(x, (bool, int, float, str, bytes)):
        g.update((type(x).__name__ + repr(x) + ";").encode())
    elif isinstance(x, torch.dtype) or isinstance(x, torch.device):
        g.update(repr(x).encode())
    else:
        raise TypeError(f"--full: no hash for {type(x).__name__} at {path}")
tmpd = tempfile.mkdtemp(prefix="determinism_", dir=(arg("tmp", "") or None)) if (ROUNDTRIP or FULL) else None
try:
    half = ticks // 2 if ROUNDTRIP else ticks
    run(half)
    lost = None
    if ROUNDTRIP:
        life.save(os.path.join(tmpd, "life.pt"))
        life = Life.load(os.path.join(tmpd, "life.pt"), TOK, device="cpu", seed=0)   # the save's own constants, as a reload keeps them
        # THE FIXED POINT: the loaded life saved again at once must write what it was loaded from; a field the loader drops (the ninth
        # and thirteenth defects) or sizes afresh is named here, and the names enter the digest
        life.save(os.path.join(tmpd, "again.pt"))
        b1 = torch.load(os.path.join(tmpd, "life.pt"), map_location="cpu", weights_only=False)
        b2 = torch.load(os.path.join(tmpd, "again.pt"), map_location="cpu", weights_only=False); os.remove(os.path.join(tmpd, "again.pt"))
        lost = []
        for part in sorted(set(b1) | set(b2)):
            if part == "env": continue
            x1, x2 = b1.get(part), b2.get(part)
            keys = sorted(set(x1) | set(x2), key=repr) if isinstance(x1, dict) and isinstance(x2, dict) else [None]
            for k in keys:
                g1, g2 = hashlib.sha256(), hashlib.sha256()
                canon(g1, x1 if k is None else x1.get(k), part); canon(g2, x2 if k is None else x2.get(k), part)
                if g1.digest() != g2.digest(): lost.append(part if k is None else f"{part}.{k}")
        del b1, b2
        run(ticks - half, half)
    if profile == "default":
        rep = life.night()                           # as the tick's sleep switch calls it: with grad, the night's own optimizer
    else:
        assert life.nights == 1, f"the sleep switch did not bring the night (nights {life.nights}, sleep pressure {life.sleep_pressure}, last night {(life.last_night or {}).get('error')})"
        rep = life.last_night                        # the switch brought it, at the last tick of the day
    assert not rep.get("error"), rep.get("error")
    run(100)
    h = hashlib.sha256()
    def add(name, x):
        if torch.is_tensor(x): h.update(name.encode()); h.update(x.detach().cpu().contiguous().numpy().tobytes())
        else: h.update(name.encode()); h.update(repr(x).encode())
    for k, v in sorted(life.m.state_dict().items()): add(k, v)
    st = life.store
    for k in ("K", "V", "S", "W", "N", "NE", "B", "Bs", "Bq", "A"): add("store." + k, getattr(st, k))
    add("utts", [list(u) for u in life.utts]); add("utt_S", list(life.utt_S))
    add("page", "".join(e[0] for e in life.page)); add("feel", (round(float(life.mood), 6), round(float(life.fatigue), 6), round(float(life.stress), 6), int(life.ticks), int(life.nights)))
    if lost is not None: add("roundtrip.lost", lost)       # the round trip's fixed point: the fields a reload does not give back
    sections = []
    if FULL:
        def section(name, items):
            g = hashlib.sha256()
            for k, v in items: g.update(k.encode() + b"="); canon(g, v, name + "." + k)
            sections.append((name, g.hexdigest()[:12])); h.update(f"full.{name}".encode()); h.update(g.digest())
        m = life.m
        # the organs: every buffer (persistent or not), every parameter's plasticity switch and gradient, every plain attribute of every module
        org = [("buffers", dict(m.named_buffers())), ("params", dict(m.named_parameters())),
               ("requires_grad", {n: p.requires_grad for n, p in m.named_parameters()}), ("grad", {n: p.grad for n, p in m.named_parameters()})]
        for mn, mod in m.named_modules():
            org.append(("attrs:" + (mn or "."), {k: v for k, v in vars(mod).items() if not k.startswith("_") and not isinstance(v, torch.nn.Module)}))
        section("organs", org)
        section("store", [(k, v) for k, v in sorted(vars(life.store).items())])
        # the saved blob as the body writes it (every counter of the saved life dict), the environment it records left out
        sp = os.path.join(tmpd, "full.pt"); life.save(sp); blob = torch.load(sp, map_location="cpu", weights_only=False); os.remove(sp)
        section("saved", [(k, blob[k]) for k in sorted(blob) if k != "env"])
        OPTS = sorted(k for k, v in vars(life).items() if isinstance(v, torch.optim.Optimizer))
        section("optim", [(k, getattr(life, k).state_dict()) for k in OPTS])
        section("rng", [("life.gen", life.gen.get_state()), ("torch", torch.get_rng_state())])
        SKIP = {"m", "store", "tok", "gen", "cfg", "save_path", "_t_feel"} | set(OPTS)   # hashed above, or the wall clock and the file's name
        section("work", [(k, v) for k, v in sorted(vars(life).items()) if k not in SKIP] + [("cfg", life.cfg)])
finally:
    if tmpd: shutil.rmtree(tmpd, ignore_errors=True)
print(f"digest {h.hexdigest()[:24]} | ticks {life.ticks} nights {life.nights} store {st.n()} utts {len(life.utts)} own symbols {sum(1 for e in life.page if e[1] == 1 and e[0])} | dreams {rep.get('dreams')} dropped {rep.get('store_dropped')}"
      f" | profile {profile}{' full' if FULL else ''}{' roundtrip' if ROUNDTRIP else ''} consts {consts}")
if sections:
    print("full: " + " ".join(f"{n} {d}" for n, d in sections))
if lost is not None:
    print("roundtrip: saved again at once after the load, the save differs in " + (", ".join(lost) if lost else "nothing"))
