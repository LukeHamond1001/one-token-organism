"""THE CEILING OF THE LINE: on two recorded days of a striatal body (tools/record_day.py with the raw line saved), how much
of the discounted return the delay line holds under different readouts: the live born expansion (ridge), the plain one-hot
line (ridge), a small trained nonlinear head on the one-hot line, and born sparse conjunctions (granule-cell form: each unit
the AND of a few random position-event pairs) with ridge.   python3 tools/ceiling_line.py DAY_A.pt DAY_B.pt [--h 16]
"""
import argparse, statistics as st, torch
ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--h", type=int, default=16); ap.add_argument("--epochs", type=int, default=40)
a = ap.parse_args(); torch.manual_seed(0); torch.set_num_threads(3)
def load(f):
    d = torch.load(f, map_location="cpu", weights_only=False); r = d["r"]; g = 1 - 1 / a.h; G = torch.zeros(len(r)); acc = 0.0
    for t in range(len(r) - 1, -1, -1): acc = float(r[t]) + g * acc; G[t] = acc
    return d, G, [i for i, x in enumerate(r.tolist()) if x > 0]
dA, GA, smA = load(a.a); dB, GB, smB = load(a.b); V = int(dA["vocab"]); W = 2 * V + 3; K = dA["line"].shape[1]
def corr(u, v):
    u = u - u.mean(); v = v - v.mean(); den = (u.norm() * v.norm()).item(); return float((u @ v) / den) if den > 1e-9 else float("nan")
def rise(Vv, sm, span=8):
    pre = [float(Vv[i - 1]) for i in sm if i >= 1]; far = [float(Vv[i - span]) for i in sm if i >= span]; return (st.mean(pre) - st.mean(far)) / 2 * 100
def onehot(d):
    L = d["line"].long(); T = L.shape[0]; X = torch.zeros(T, K * W)
    for p in range(K):
        e = L[:, p]; ok = e >= 0; X[torch.arange(T)[ok], p * W + e[ok]] = 1.0
    return X
XA, XB = onehot(dA), onehot(dB)
def ridge_report(FA, FB, name, lams=(1.0, 10.0, 100.0, 1000.0)):
    mu, sd = FA.mean(0), FA.std(0) + 1e-6; ZA = torch.cat([(FA - mu) / sd, torch.ones(len(FA), 1)], 1); ZB = torch.cat([(FB - mu) / sd, torch.ones(len(FB), 1)], 1)
    best = None
    for lam in lams:
        A = ZA.T @ ZA + lam * torch.eye(ZA.shape[1]); A[-1, -1] -= lam; w = torch.linalg.solve(A, ZA.T @ GA); VA = ZA @ w; VB = ZB @ w
        c = corr(VB, GB)
        if best is None or c > best[0]: best = (c, f"  {name:44s} ridge {lam:5.0f}: in-sample corr {corr(VA, GA):+.2f} rise {rise(VA, smA):+4.0f}% | held-out day corr {c:+.2f} rise before smiles {rise(VB, smB):+4.0f}% of a smile")
    print(best[1])
print(f"fit on {a.a.split('/')[-1]} ({len(GA)} ticks, {len(smA)} rewards) -> read on {a.b.split('/')[-1]} ({len(GB)} ticks, {len(smB)} rewards); horizon {a.h}; line {K} x {W}")
ridge_report(dA["Z"].float(), dB["Z"].float(), f"the live born expansion ({dA['Z'].shape[1]} units)")
ridge_report(XA, XB, f"the one-hot line ({K * W} dims, linear)")
print(f"  the body's own live head on the read day: corr {corr(dB['vf'], GB):+.2f} | rise before smiles {rise(dB['vf'], smB):+.0f}% | error at the reward tick {st.mean([float(dB['dopa'][i]) for i in smB]):+.2f}")
# a trained nonlinear head on the one-hot line
n = len(XA); cut = int(n * 0.8)
for hid in (256,):
    net = torch.nn.Sequential(torch.nn.Linear(K * W, hid), torch.nn.ReLU(), torch.nn.Linear(hid, 1)); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5); best = (1e9, None)
    for ep in range(a.epochs):
        perm = torch.randperm(cut)
        for i in range(0, cut, 256):
            b = perm[i:i + 256]; loss = ((net(XA[b]).squeeze(1) - GA[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): vl = float(((net(XA[cut:]).squeeze(1) - GA[cut:]) ** 2).mean())
        if vl < best[0]: best = (vl, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    with torch.no_grad(): VA = net(XA).squeeze(1); VB = net(XB).squeeze(1)
    print(f"  {'a trained MLP ' + str(hid) + ' on the one-hot line':44s}            : fit-day corr {corr(VA[:cut], GA[:cut]):+.2f} (tail {corr(VA[cut:], GA[cut:]):+.2f}) rise {rise(VA, smA):+4.0f}% | held-out day corr {corr(VB, GB):+.2f} rise before smiles {rise(VB, smB):+4.0f}% of a smile")
# born sparse conjunctions: each unit = AND of `fan` random (position, event) pairs from a bag of pairs that occur; fires when all present
torch.manual_seed(2); LA = dA["line"].long(); LB = dB["line"].long()
present = torch.zeros(K * W, dtype=torch.bool); present[XA.sum(0) > 0] = True; pool = torch.nonzero(present).squeeze(1)
for fan, M in ((2, 8000), (3, 16000)):
    idx = pool[torch.randint(0, len(pool), (M, fan))]
    def conj(X): 
        out = torch.ones(len(X), M)
        for j in range(fan): out = out * X[:, idx[:, j]]
        return out
    CA, CB = conj(XA), conj(XB); keep = CA.sum(0) > 2; CA, CB = CA[:, keep], CB[:, keep]
    ridge_report(CA, CB, f"born conjunctions of {fan} ({int(keep.sum())} live of {M})", lams=(10.0, 100.0, 1000.0, 10000.0))
