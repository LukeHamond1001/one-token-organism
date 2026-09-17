"""THE STORE'S STRENGTHS (a supervisor's instrument, 2026-09-17): a save's slot count, the strengths' mean, median, quantiles and
histogram, the counts under candidate floors, and the save's store constants.
usage: python3 tools/store_dist.py SAVE.pt"""
import torch, sys
p=sys.argv[1]
d=torch.load(p, map_location='cpu', weights_only=False)
def find(o, path=''):
    if isinstance(o, dict):
        if all(k in o for k in ('K','V','S')): return o, path
        for k,v in o.items():
            r=find(v, path+'/'+str(k))
            if r: return r
    return None
r=find(d)
if not r:
    print(type(d), list(d.keys())[:40] if isinstance(d,dict) else ''); sys.exit()
st,path=r
S=st['S'].float()
print('path',path,'n',S.numel(),'mean',round(S.mean().item(),4),'median',round(S.median().item(),4))
qs=[0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]
print('quantiles', {q: round(torch.quantile(S,q).item(),4) for q in qs})
print('rel floor 0.1*mean =', round(0.1*S.mean().item(),4))
for t in [0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3]:
    print(f'below {t}: {(S<t).sum().item()}')
if 'W' in st:
    W=st['W']; print('who counts', torch.bincount(W.long()).tolist())
for k in ('A','B','Bs','Bq'):
    if k in st:
        v=st[k]; print(k, 'dtype', v.dtype, 'sum', v.float().sum().item(), 'mean', round(v.float().mean().item(),4))
edges=[0.0,0.02,0.05,0.1,0.2,0.3,0.5,0.7,1.0,1.5,2,3,5,10,100]
h=torch.histogram(S, bins=torch.tensor(edges)).hist.tolist()
print('hist', list(zip(edges[:-1], [int(x) for x in h])))
cfg=d.get('cfg') if isinstance(d,dict) else None
if cfg: print('cfg store keys', {k:v for k,v in cfg.items() if 'store' in k or 'write' in k})
