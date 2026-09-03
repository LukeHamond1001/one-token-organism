"""the sequence probe (BODY_SPEC.md §7): after a cue, what the store and the cortex forecast, and
again after the body's own first letter of the answer. Read-only, the mouth held quiet.

  python3 -m body.sequence_probe data/body2_fast.pt
"""
import sys

import torch
from tokenizers import Tokenizer

sys.path.insert(0, "/Users/lukehamond/Projects/project")
from body.life import Life  # noqa: E402

CASES = [("dog will ", "go"), ("give ", "milk"), ("where ball? ", "ball"), ("big dog bigger ", "dog"),
         ("I had ", "milk"), ("scared ", "dog"), ("first milk then ", "ball"), ("why dog up? ", "because")]


def quiet_life(path, tok):
    L = Life.load(path, tok, save_path=None); L.save_path = None
    L.cfg["wake_ticks"] = 10 ** 9; L.cfg["wake_every"] = 10 ** 9; L.cfg["gate_every"] = 10 ** 9; L.cfg["gate_floor"] = 0.0
    with torch.no_grad():
        L.m.mouth_gate.bias.fill_(-30.0)
    return L


def forecast(L, p):
    prior = L.heard / L.heard.sum().clamp(min=1.0)
    lg = L.m.readout(p, prior=prior).clone(); lg[L.bans] = float("-inf"); lg[L.sil] = float("-inf")
    return L.tok.decode([int(lg.argmax())])


def main():
    path = sys.argv[1]
    tok = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
    right_store = right_cortex = right_seq = 0
    for cue, ans in CASES:
        L = quiet_life(path, tok); m = L.m
        L.type_text(cue)
        while L.queue:
            L.tick()
        # the forecast made as the cue's last symbol entered (what the mouth reads)
        with torch.no_grad():
            C = L._stream_now(); p0 = m.latent_pred(C)
        c0 = forecast(L, p0)
        pred, conf, _ = L.store.read(L.bag); s0 = tok.decode([m.nearest(pred)])
        # its own first letter of the answer enters, then the forecast again
        with torch.no_grad():
            C1, p1, _, _ = L._step(tok.token_to_id(ans[0]), 1, learn_store=False)
        c1 = forecast(L, p1)
        pred2, conf2, _ = L.store.read(L.bag); s1 = tok.decode([m.nearest(pred2)])
        right_store += int(s0 == ans[0]); right_cortex += int(c0 == ans[0]); right_seq += int(c1 == ans[1])
        print(f"{cue!r:>18} want {ans[0]!r}: store {s0!r}({conf:.2f}) cortex {c0!r} | after own {ans[0]!r} want {ans[1]!r}: store {s1!r}({conf2:.2f}) cortex {c1!r}")
    print(f"first letter: store {right_store}/{len(CASES)} cortex {right_cortex}/{len(CASES)} | second letter (cortex, after its own first): {right_seq}/{len(CASES)}")


if __name__ == "__main__":
    main()
