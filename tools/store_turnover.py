"""THE STORE'S TURNOVER (a supervisor's instrument, 2026-09-17): two morning saves a day apart; which slots survived, how many were
written and dropped, the new slots' write strength, and the steady-state size under absolute forgetting floors (the fade 0.9 a
night; a slot written at s lives ln(F/s)/ln(0.9) nights above a floor F). The derivation of store_floor_abs (the ledger's item 2).
usage: python3 tools/store_turnover.py BEFORE.pt AFTER.pt"""
import torch, sys, math
def load_store(p):
    d=torch.load(p, map_location='cpu', weights_only=False)
    st=d['store']; K=st['K'].float(); S=st['S'].float()
    keys=[hash(r.numpy().tobytes()) for r in K]
    return keys, S
k0,S0=load_store(sys.argv[1]); k1,S1=load_store(sys.argv[2])
set0=set(k0); set1=set(k1)
surv=[i for i,k in enumerate(k1) if k in set0]; new=[i for i,k in enumerate(k1) if k not in set0]
dropped=sum(1 for k in k0 if k not in set1)
print(f"before {len(k0)} after {len(k1)}: survivors {len(surv)} new {len(new)} dropped {dropped}")
s_new=S1[new]/0.9   # the strength as written (one fade applied at the night)
qs=[0.05,0.1,0.25,0.5,0.75,0.9,0.99]
print("new slots' write strength quantiles", {q: round(torch.quantile(s_new,q).item(),3) for q in qs}, "mean", round(s_new.mean().item(),3))
# the strengthening of survivors (merges): the ratio S1/S0 for survivors vs the plain fade 0.9
idx0={k:i for i,k in enumerate(k0)}
r=torch.tensor([S1[i].item()/(0.9*S0[idx0[k1[i]]].item()) for i in surv])
print(f"survivors strengthened by a merge that day: {(r>1.001).sum().item()} of {len(surv)} (ratio mean among them {r[r>1.001].mean().item():.3f})")
f=0.9; ln=math.log(f)
print("floor F | drop now | drop tonight | lifetime of a new slot (nights, mean / median) | steady size = new/day x mean lifetime")
for F in [0.03,0.04,0.05,0.06,0.07,0.08,0.1]:
    now=(S1<F).sum().item(); tonight=((S1*f)<F).sum().item()
    life=torch.clamp(torch.ceil(torch.log(F/s_new)/ln),min=0)   # nights until faded below F
    print(f"{F:5.2f} | {now:6d} | {tonight:6d} | {life.mean().item():5.1f} / {life.median().item():4.0f} | {len(new)*life.mean().item():8.0f}")
print("the relative floor now:", round(0.1*S1.mean().item(),4))
