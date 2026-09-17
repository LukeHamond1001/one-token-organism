"""WHICH BANDS CARRY THE SMILE: ridge heads from chosen band sets to the discounted return at a horizon, fitted on one recorded
day and read on another day of the same body (tools/record_day.py). Reports the correlation with the return in and out of
sample and the rise of the fitted value in the ticks before a smile, as a fraction of the smile.
    python3 tools/fit_return.py DAY_A.pt DAY_B.pt [--h 16] [--sets 2;0;1;0,1;0,1,2;0,1,2,3;all]
"""
import argparse, statistics as st, torch
ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--h", type=int, default=16)
ap.add_argument("--sets", default="2;0;1;0,1;0,1,2;0,1,2,3;all"); ap.add_argument("--lams", default="1,10,100,1000"); ap.add_argument("--src", default="bands", help="bands | C (the cortex's stream vector) | C+bands | Z (the striatal input)")
a = ap.parse_args()
def load(f):
    d = torch.load(f, map_location="cpu", weights_only=False); X = d["bands"].float(); r = d["r"]; nb = X.shape[1]
    g = 1.0 - 1.0 / a.h; G = torch.zeros(len(r)); acc = 0.0
    for t in range(len(r) - 1, -1, -1): acc = float(r[t]) + g * acc; G[t] = acc
    sm = [i for i, rr in enumerate(r.tolist()) if rr > 0]
    return X, G, sm, nb, d
XA, GA, smA, nb, dA = load(a.a); XB, GB, smB, _, dB = load(a.b)
print(f"fit on {a.a.split('/')[-1]} ({len(GA)} ticks, {len(smA)} rewards) -> read on {a.b.split('/')[-1]} ({len(GB)} ticks, {len(smB)} rewards); horizon {a.h}; clocks {dA.get('clocks')}")
def corr(u, v):
    u = u - u.mean(); v = v - v.mean(); den = (u.norm() * v.norm()).item()
    return float((u @ v) / den) if den > 1e-9 else float("nan")
def feats(X, bands, d=None):
    F = X.reshape(X.shape[0], -1) if bands == "all" else X[:, bands, :].reshape(X.shape[0], -1)
    if a.src == "C": return d["C"].float()
    if a.src == "Z": return d["Z"].float()
    if a.src == "C+bands": return torch.cat([d["C"].float(), F], 1)
    return F
def rise(V, sm, span=8):
    pre = [float(V[i - 1]) for i in sm if i - 1 >= 0]; far = [float(V[i - span]) for i in sm if i - span >= 0]
    return (st.mean(pre) - st.mean(far)) if pre and far else float("nan")
for s in (["C"] if a.src in ("C", "Z") else a.sets.split(";")):
    bands = "all" if s in ("all", "C") else [int(x) for x in s.split(",")]
    FA = feats(XA, bands, dA); FB = feats(XB, bands, dB); mu = FA.mean(0); sd = FA.std(0) + 1e-6
    ZA = (FA - mu) / sd; ZB = (FB - mu) / sd; ZA1 = torch.cat([ZA, torch.ones(len(ZA), 1)], 1); ZB1 = torch.cat([ZB, torch.ones(len(ZB), 1)], 1)
    best = None
    for lam in [float(x) for x in a.lams.split(",")]:
        A = ZA1.T @ ZA1 + lam * torch.eye(ZA1.shape[1]); A[-1, -1] -= lam; w = torch.linalg.solve(A, ZA1.T @ GA)
        VA = ZA1 @ w; VB = ZB1 @ w
        line = f"  bands {s:9s} lam {lam:6.0f}: in-sample corr {corr(VA, GA):+.2f} rise {rise(VA, smA)/2*100:+4.0f}% | held-out day corr {corr(VB, GB):+.2f} rise before smiles {rise(VB, smB)/2*100:+4.0f}% of a smile"
        if best is None or corr(VB, GB) > best[0]: best = (corr(VB, GB), line)
        print(line)
    print("   best held-out:", best[1].strip())
print(f"the body's own fast head on the read day: corr with G {corr(dB['vf'], GB):+.2f} | rise before smiles {rise(dB['vf'], smB)/2*100:+.0f}% | error at the reward tick {st.mean([float(dB['dopa'][i]) for i in smB]):+.2f}")
