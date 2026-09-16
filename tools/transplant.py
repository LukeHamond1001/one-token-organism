"""A CORTEX TRANSPLANT ON A COPY (a supervisor's instrument, 2026-09-16, night 226): the current body's save with its cortex (the
transformer, its embedding, its input projections and its forecast head) replaced by the cortex of an earlier save, everything
else of the body kept as it is: the hippocampal store, the utterance memory, the critics, the gate, the face, the life's state.
For measuring on copies whether a damaged cortex is better repaired by the nights or by the earlier cortex; whether such a repair
is ever applied to the served body is the user's call, never the instrument's.
usage: python3 tools/transplant.py CURRENT.pt DONOR.pt OUT.pt"""
import sys, torch
cur = torch.load(sys.argv[1], map_location="cpu", weights_only=False); donor = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
CORTEX = ("blocks.", "E.", "latent_pred", "in_ln", "bundle_in", "store_in", "face_in", "perm")
moved = 0
for k, v in donor["organs"].items():
    if k.startswith(CORTEX) and k in cur["organs"] and cur["organs"][k].shape == v.shape:
        cur["organs"][k] = v.clone(); moved += 1
torch.save(cur, sys.argv[3])
print(f"transplanted {moved} cortex tensors from {sys.argv[2].split('/')[-1]} (nights {donor['life']['nights']}) into {sys.argv[1].split('/')[-1]} (nights {cur['life']['nights']}); the store ({cur['store']['K'].shape[0]} slots), the utterance memory ({len(cur['life']['utts'])}) and the life kept -> {sys.argv[3].split('/')[-1]}")
