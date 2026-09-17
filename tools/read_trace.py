"""THE READ TRACED (a supervisor's instrument, 2026-09-17): a prefix fed to a saved body on a copy exactly as probe_lm feeds it, then
the store's read at that moment laid open: the slots that carry the weight, their strengths, the symbol each one's value says, and
the vote split by symbol; and how many keys sit near the query (cos > 0.9) with their strengths, the satellites a fact's retellings
leave. Two saves side by side tell whether a change of forgetting moved the vote.
usage: python3 tools/read_trace.py SAVE.pt "ice is " ["water is " ...] [--top 8]
       python3 tools/read_trace.py SAVE.pt --qa [--pause 2]   (every fact's question fed as the question ruler feeds it: the answer's
       first symbol's share of the read and the winner, fact by fact)"""
import sys, torch
sys.path.insert(0, "/Users/lukehamond/Projects/project")
from tokenizers import Tokenizer
from body.life import Life

top = 8; args = []; skip = False; pause = 2
for i, a in enumerate(sys.argv[1:], 1):
    if skip: skip = False; continue
    if a == "--top": top = int(sys.argv[i + 1]); skip = True; continue
    if a == "--pause": pause = int(sys.argv[i + 1]); skip = True; continue
    if not a.startswith("--"): args.append(a)
path, prefixes = args[0], args[1:]
qa = "--qa" in sys.argv
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

if qa:
    # THE QUESTIONS, READ AS THE RULER READS THEM (the real read, life._recall, its weights from the store): two readings per fact.
    # The onset: after the question and the rests, the share of the answer SENTENCE's first symbol (an article or a word of the
    # question for most facts: a weak reading, said so). The middle: the sentence teacher-forced as the body's own symbols up to
    # its answer word (probe_lm's keys: the fact's content words not in the question), then the read: the share of the answer
    # word's first letter, the reading that the episode's chain must carry (the reviewer of 2026-09-17: fact[0] alone said little).
    import re
    STOP = set("is are the a an in on and to do we i you it of my your they can what where who how does".split())
    pairs = [tuple(x.strip() for x in l.split("|")[:2]) for l in open("/Users/lukehamond/Projects/project/tools/facts_stage5.txt") if "|" in l]
    def split_of():
        w = st._last_w; out = {}; cum = 0.0
        for j in torch.argsort(w, descending=True).tolist():
            s_ = sym_of(st.V[j]); out[s_] = out.get(s_, 0.0) + float(w[j]); cum += float(w[j])
            if cum > 0.99: break
        return out
    def top(d): return max(d.items(), key=lambda x: x[1]) if d else ("?", 0.0)
    zero = torch.zeros(m.d)
    n_on = n_mid = 0; t_on = t_mid = 0.0
    print(f"\nTHE QUESTIONS ({pause} rests): the onset = the sentence's first symbol's share of the real read | the middle = the answer word's first letter's share after the sentence is taken as its own up to that word")
    for q, fact in pairs:
        keys = [w_ for w_ in re.findall(r"[a-z]+", fact.lower()) if w_ not in STOP and w_ not in q.lower()]
        aw = keys[0] if keys else fact.split()[-1]; pos = fact.lower().find(aw); pre = fact[:pos]
        life.win.clear(); life.bag_w.zero_(); life.bag_o.zero_(); life.ctx_cur.zero_(); life.ctx_prev.zero_(); life._utt_open = False; life.n_own = 0; life._follow = None
        with torch.no_grad():
            for ch in q:
                i = TOK.token_to_id(ch); life.win.append({"x": i, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
                life.rest_tick(world=True); life.take_world(i)
            for _ in range(pause):
                life.win.append({"x": life.sil, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0}); life.rest_tick()
            rd, conf, _w = life._recall(life.bag); on = split_of(); a0 = fact[0]; on_share = on.get(a0, 0.0); on_top = top(on)
            for ch in pre:                                            # the sentence as its own, up to the answer word
                i = TOK.token_to_id(ch)
                life.win.append({"x": life.sil, "xo": i, "face": torch.zeros(2), "bundle": life.bands, "read": rd, "r": 0.0})
                life.rest_tick(); life.take_own(i); rd, conf, _w = life._recall(life.bag)
            mid = split_of(); m0 = aw[0]; mid_share = mid.get(m0, 0.0); mid_top = top(mid)
        n_on += int(on_top[0] == a0); t_on += on_share; n_mid += int(mid_top[0] == m0); t_mid += mid_share
        flag = " (opener in the question)" if fact.split()[0].lower() in q.lower() else ""
        print(f"   {q!r:26} onset {a0!r} {on_share:.2f} {'wins' if on_top[0] == a0 else 'loses to ' + repr(on_top[0]) + f' {on_top[1]:.2f}'}{flag} | middle {pre!r}->{m0!r} ({aw}) {mid_share:.2f} {'wins' if mid_top[0] == m0 else 'loses to ' + repr(mid_top[0]) + f' {mid_top[1]:.2f}'}")
    n = max(1, len(pairs)); print(f"   the onset wins {n_on} of {len(pairs)} (mean share {t_on / n:.3f}); THE MIDDLE, the answer word, wins {n_mid} of {len(pairs)} (mean share {t_mid / n:.3f})")
