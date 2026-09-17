"""THE READ TRACED (a supervisor's instrument, 2026-09-17): a prefix fed to a saved body on a copy exactly as probe_lm feeds it, then
the store's read at that moment laid open: the slots that carry the weight, their strengths, the symbol each one's value says, and
the vote split by symbol; and how many keys sit near the query (cos > 0.9) with their strengths, the satellites a fact's retellings
leave. Two saves side by side tell whether a change of forgetting moved the vote.
usage: python3 tools/read_trace.py SAVE.pt "ice is " ["water is " ...] [--top 8]"""
import sys, torch
sys.path.insert(0, "/Users/lukehamond/Projects/project")
from tokenizers import Tokenizer
from body.life import Life

top = 8; args = []; skip = False
for i, a in enumerate(sys.argv[1:], 1):
    if skip: skip = False; continue
    if a == "--top": top = int(sys.argv[i + 1]); skip = True; continue
    if not a.startswith("--"): args.append(a)
path, prefixes = args[0], args[1:]
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
life = Life.load(path, TOK, device="cpu", cfg={}, seed=0); m = life.m; m.eval(); st = life.store
print(f"{path.rsplit('/', 1)[-1]}: nights {life.nights} store {st.n()} temp {st.temp} read_strength {st.read_strength}")
def sym_of(v):
    lg = m.readout(v.unsqueeze(0) if v.dim() == 1 else v); return TOK.decode([int(lg.argmax())])
for pre in prefixes:
    life.win.clear(); life.bag_w.zero_(); life.bag_o.zero_(); life.ctx_cur.zero_(); life.ctx_prev.zero_(); life._utt_open = False; life.n_own = 0; life._follow = None
    with torch.no_grad():
        for ch in pre:
            i = TOK.token_to_id(ch); life.rest_tick(world=True); life.take_world(i)
        q = life.bag.float()
        sims = st.K @ q; w = torch.softmax(sims / st.temp, 0)
        order = torch.argsort(w, descending=True)[:top]
        # the vote split by the value's symbol over the slots that hold 99% of the weight
        cum = 0.0; split = {}
        for j in torch.argsort(w, descending=True).tolist():
            s = sym_of(st.V[j]); split[s] = split.get(s, 0.0) + float(w[j]); cum += float(w[j])
            if cum > 0.99: break
        near = (sims > 0.9 * float(q.norm())).nonzero().flatten()
        print(f"\n'{pre}': query norm {float(q.norm()):.3f}; the vote by symbol (99% of the weight): " + ", ".join(f"{k!r} {v:.2f}" for k, v in sorted(split.items(), key=lambda x: -x[1])[:6]))
        print(f"   keys near the query (sim > 0.9 of the norm): {int(near.numel())}; their strengths: " + (", ".join(f"{float(st.S[j]):.2f}" for j in near.tolist()[:12]) if near.numel() else "-"))
        for j in order.tolist():
            print(f"   slot {j:6d} w {float(w[j]):.3f} sim {float(sims[j]):.3f} S {float(st.S[j]):.2f} value -> {sym_of(st.V[j])!r} key's last symbol {TOK.decode([int(m.nearest(st.K[j]))])!r}")
