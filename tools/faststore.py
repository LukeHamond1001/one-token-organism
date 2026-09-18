"""A STORE WITH ROOM KEPT AHEAD (a supervisor's instrument, 2026-09-18): body/model.py's Store grows its slots by concatenation, a
copy of every slot at every write, which the served body pays once a symbol and a replay of a hundred thousand symbols cannot. The
same store, its rows kept in buffers allocated in blocks and exposed as views: reads, links, marks, fades and the eviction are the
body's own code, unchanged; only the append and the compaction differ. For the tools that rebuild or fill a store; never the body."""
import torch, torch.nn.functional as F
from body.model import Store

class FastStore(Store):
    BLOCK = 8192
    def _room(self, extra=1):
        n = self.n(); have = self._Kb.shape[0] if hasattr(self, "_Kb") else 0
        if not hasattr(self, "_Kb") or n + extra > have:
            new = max(have + self.BLOCK, n + extra)
            def grow(buf, shape1, fill, dtype):
                b = torch.full((new,) + shape1, fill, dtype=dtype, device=self.dev); 
                if n > 0: b[:n] = buf[:n]
                return b
            self._Kb = grow(getattr(self, "K", torch.zeros(0, self.d)), (self.d,), 0.0, torch.float32)
            self._Vb = grow(getattr(self, "V", torch.zeros(0, self.d)), (self.d,), 0.0, torch.float32)
            self._Sb = grow(getattr(self, "S", torch.zeros(0)), (), 0.0, torch.float32)
            self._Wb = grow(getattr(self, "W", torch.zeros(0, dtype=torch.long)), (), 0, torch.long)
            self._Bb = grow(getattr(self, "B", torch.zeros(0, dtype=torch.bool)), (), False, torch.bool)
            self._Bsb = grow(getattr(self, "Bs", torch.zeros(0, dtype=torch.bool)), (), False, torch.bool)
            self._Bqb = grow(getattr(self, "Bq", torch.zeros(0, dtype=torch.bool)), (), False, torch.bool)
            self._Nb = grow(getattr(self, "N", torch.zeros(0, self.NK, dtype=torch.long)), (self.NK,), -1, torch.long)
            self._NEb = grow(getattr(self, "NE", torch.zeros(0, self.NK, dtype=torch.long)), (self.NK,), -1, torch.long)
            self._Ab = grow(getattr(self, "A", torch.zeros(0)), (), 1.0, torch.float32)
            self._views(n)
    def _views(self, n):
        self.K, self.V, self.S, self.W = self._Kb[:n], self._Vb[:n], self._Sb[:n], self._Wb[:n]
        self.B, self.Bs, self.Bq = self._Bb[:n], self._Bsb[:n], self._Bqb[:n]
        self.N, self.NE, self.A = self._Nb[:n], self._NEb[:n], self._Ab[:n]
    @torch.no_grad()
    def write(self, k, v, strength, who, merge_cos=0.97):
        if strength <= 1e-4:
            return False
        k = F.normalize(k.to(self.dev).float(), dim=0); v = F.normalize(v.to(self.dev).float(), dim=0)
        n = self.n()
        if n > 0:
            sims = self.K @ k
            same = (sims > merge_cos) & ((self.V @ v) > merge_cos)
            if bool(same.any()):
                j = int(torch.where(same, sims, torch.full_like(sims, -2.0)).argmax())
                self.last_idx = j
                if self.saturate:
                    m_ = float(self.S.mean()) if n > 0 else 1.0
                    self.S[j] += float(strength) * m_ / (m_ + float(self.S[j]))
                else:
                    self.S[j] += float(strength)
                return True
        self._room(1)
        self._Kb[n] = k; self._Vb[n] = v; self._Sb[n] = float(strength); self._Wb[n] = int(who)
        self._Bb[n] = False; self._Bsb[n] = False; self._Bqb[n] = False; self._Nb[n] = -1; self._NEb[n] = -1; self._Ab[n] = 1.0
        self._views(n + 1); self.last_idx = n
        if self.n() > self.cap:
            keep = torch.argsort(self.S, descending=True)[: self.cap]
            self._keep(keep)
        return True
    def _keep(self, idx):
        super()._keep(idx)                                       # the body's compaction (new tensors); then back into the buffers
        n = self.n(); K, V, S, W, B, Bs, Bq, N, NE, A = self.K, self.V, self.S, self.W, self.B, self.Bs, self.Bq, self.N, self.NE, self.A
        self._Kb[:n] = K; self._Vb[:n] = V; self._Sb[:n] = S; self._Wb[:n] = W; self._Bb[:n] = B; self._Bsb[:n] = Bs; self._Bqb[:n] = Bq
        self._Nb[:n] = N; self._NEb[:n] = NE; self._Ab[:n] = A
        self._views(n)
    def n(self):
        return int(self.K.shape[0])
