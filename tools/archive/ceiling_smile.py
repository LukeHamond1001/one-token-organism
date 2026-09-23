"""THE CEILING OF EXPECTATION: on two recorded days of one body (tools/record_day.py), how predictable the discounted return
at a horizon is (a) from the body's own last K symbols (one-hot, ridge): is the smile in the stream at all; (b) from the fast
bands through a small nonlinear head (an MLP, early-stopped on the fit day's tail): can a striatal readout find it where a
linear one cannot.   python3 tools/ceiling_smile.py DAY_A.pt DAY_B.pt [--h 16] [--k 8] [--bands 0,1,2]
"""
import argparse, statistics as st, torch
ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--h", type=int, default=16)
ap.add_argument("--k", type=int, default=8); ap.add_argument("--bands", default="0,1,2"); ap.add_argument("--epochs", type=int, default=60)
a = ap.parse_args(); torch.manual_seed(0); torch.set_num_threads(2)
def load(f):
    d = torch.load(f, map_location="cpu", weights_only=False); r = d["r"]; g = 1.0 - 1.0 / a.h; G = torch.zeros(len(r)); acc = 0.0
    for t in range(len(r) - 1, -1, -1): acc = float(r[t]) + g * acc; G[t] = acc
    return d, G, [i for i, rr in enumerate(r.tolist()) if rr > 0]
dA, GA, smA = load(a.a); dB, GB, smB = load(a.b)
def corr(u, v):
    u = u - u.mean(); v = v - v.mean(); den = (u.norm() * v.norm()).item(); return float((u @ v) / den) if den > 1e-9 else float("nan")
def rise(V, sm, span=8):
    pre = [float(V[i - 1]) for i in sm if i >= 1]; far = [float(V[i - span]) for i in sm if i >= span]
    return (st.mean(pre) - st.mean(far)) / 2 * 100 if pre and far else float("nan")
# (a) the body's own last K symbols
syms = sorted({s for d in (dA, dB) for s in d["said"]}); idx = {s: i for i, s in enumerate(syms)}; V = len(syms)
def sym_feats(d):
    S = d["said"]; T = len(S); X = torch.zeros(T, a.k * V + 1)
    for t in range(T):
        for j in range(a.k):
            if t - j >= 0: X[t, j * V + idx[S[t - j]]] = 1.0
        X[t, -1] = 1.0
    return X
XA, XB = sym_feats(dA), sym_feats(dB)
print(f"fit on {a.a.split('/')[-1]} ({len(GA)} ticks, {len(smA)} rewards) -> read on {a.b.split('/')[-1]} ({len(GB)} ticks, {len(smB)} rewards); horizon {a.h}; {V} symbols")
for lam in (1.0, 10.0, 100.0):
    A = XA.T @ XA + lam * torch.eye(XA.shape[1]); w = torch.linalg.solve(A, XA.T @ GA); VA = XA @ w; VB = XB @ w
    print(f"  (a) last {a.k} own symbols, ridge {lam:5.0f}: in-sample corr {corr(VA, GA):+.2f} rise {rise(VA, smA):+4.0f}% | held-out day corr {corr(VB, GB):+.2f} rise before smiles {rise(VB, smB):+4.0f}% of a smile")
# (b) a small nonlinear head on the fast bands
bands = [int(x) for x in a.bands.split(",")]
FA = dA["bands"][:, bands, :].float().reshape(len(GA), -1); FB = dB["bands"][:, bands, :].float().reshape(len(GB), -1)
mu, sd = FA.mean(0), FA.std(0) + 1e-6; ZA = (FA - mu) / sd; ZB = (FB - mu) / sd
n = len(ZA); cut = int(n * 0.8); tr, va = slice(0, cut), slice(cut, n)
for hid in (64, 256):
    net = torch.nn.Sequential(torch.nn.Linear(ZA.shape[1], hid), torch.nn.Tanh(), torch.nn.Linear(hid, 1)); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best = (1e9, None)
    for ep in range(a.epochs):
        perm = torch.randperm(cut)
        for i in range(0, cut, 256):
            b = perm[i:i + 256]; loss = ((net(ZA[b]).squeeze(1) - GA[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): vl = float(((net(ZA[va]).squeeze(1) - GA[va]) ** 2).mean())
        if vl < best[0]: best = (vl, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    with torch.no_grad(): VA = net(ZA).squeeze(1); VB = net(ZB).squeeze(1)
    print(f"  (b) MLP {hid} on bands {a.bands}: fit-day corr {corr(VA[tr], GA[tr]):+.2f} (its tail {corr(VA[va], GA[va]):+.2f}) rise {rise(VA, smA):+4.0f}% | held-out day corr {corr(VB, GB):+.2f} rise before smiles {rise(VB, smB):+4.0f}% of a smile")
