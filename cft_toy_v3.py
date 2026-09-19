# cft_toy_v3.py — FIX: positional embedding inside chunks (model was blind to order!)
import random, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); random.seed(0)
D, N_SLOTS = 64, 8

SYMS = "ABCDEFGH"
TOK = {c: i+1 for i, c in enumerate(SYMS)}; TOK.update({">": 9, "<": 10, "?": 11})
def make_sample(L):
    order = random.sample(SYMS, L+1)
    facts = [(order[i], order[i+1]) for i in range(L)]
    i, j = sorted(random.sample(range(L+1), 2))
    qa, qb, lab = (order[i], order[j], 1) if random.random() < .5 else (order[j], order[i], 0)
    return facts, (qa, qb), lab
def ty(bs, op):
    t = torch.full((bs, 3), 1).long(); t[:, 1] = op; return t
def batch(bs, L):
    S = [make_sample(L) for _ in range(bs)]
    chunks, types = [], []
    for k in range(L):
        chunks.append(torch.tensor([[TOK[f[k][0]], 9, TOK[f[k][1]]] for f, _, _ in S])); types.append(ty(bs, 2))
    chunks.append(torch.tensor([[TOK[q[0]], 11, TOK[q[1]]] for _, q, _ in S])); types.append(ty(bs, 3))
    return chunks, types, torch.tensor([l for *_, l in S])

class BDHCQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.S = nn.Parameter(torch.randn(N_SLOTS, D)*.02)
        self.wk, self.wv, self.wr, self.wl = (nn.Linear(D, D, bias=False) for _ in range(4))
        self.wg, self.wg2, self.wo = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D, bias=False)
    def parts(self, x):                                    # BREAK DOWN -> RELATE
        A  = torch.softmax(self.S @ self.wk(x).transpose(1, 2)/D**.5, -1)
        P  = A @ self.wv(x); R = self.wl(P)
        return P + torch.softmax(self.wr(P) @ R.transpose(1, 2)/D**.5, -1) @ R
    def write(self, Pr, M, g):                             # REDUCE -> decay + gated write
        Md  = M * g
        Aw  = torch.softmax(Pr @ Md.transpose(1, 2)/D**.5, -1)
        Upd = Aw.transpose(1, 2) @ self.wo(Pr)
        G   = torch.sigmoid(self.wg(Upd) + self.wg2(Md))
        return Md + G * Upd

class CFTToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.typ = nn.Embedding(12, D), nn.Embedding(4, D)
        self.pos  = nn.Embedding(3, D)                     # <<< THE FIX: order within chunk
        self.local = nn.TransformerEncoderLayer(D, 4, 128, dropout=0.0, batch_first=True)
        self.bdh   = BDHCQ()
        self.M0    = nn.Parameter(torch.randn(N_SLOTS, D)*.1)
        self.g     = nn.Parameter(torch.zeros(N_SLOTS, 1))
        self.rq    = nn.Linear(D, D, bias=False)
        self.head  = nn.Sequential(nn.Linear(2*D, 64), nn.ReLU(), nn.Linear(64, 2))
        self.fid   = nn.Linear(N_SLOTS*D, D)
    def embed(self, c, t):
        return self.tok(c) + self.typ(t) + self.pos(torch.arange(c.size(1)))   # token+type+POSITION
    def forward(self, chunks, types, ablate=False):
        B = chunks[0].size(0); M = self.M0.unsqueeze(0).expand(B, -1, -1); g = torch.sigmoid(self.g)
        pools = []
        for c, t in zip(chunks[:-1], types[:-1]):
            x = self.local(self.embed(c, t)); pools.append(x.mean(1))
            M = self.bdh.write(self.bdh.parts(x), M, g)
        Pq = self.bdh.parts(self.local(self.embed(chunks[-1], types[-1])))
        Mr = torch.zeros_like(M) if ablate else M
        A  = torch.softmax(self.rq(Pq) @ Mr.transpose(1, 2)/D**.5, -1)
        return (self.head(torch.cat([Pq.mean(1), (A @ Mr).mean(1)], -1)),
                self.fid(M.flatten(1)), torch.stack(pools, 1).mean(1).detach())

model = CFTToy(); print("params:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.Adam(model.parameters(), 1e-3)

@torch.no_grad()
def acc(L, ablate=False, n=400):
    c, t, y = batch(n, L); lo, _, _ = model(c, t, ablate)
    return (lo.argmax(1) == y).float().mean().item()

for phase, (L, steps) in enumerate([(1, 2000), (2, 2500), (3, 1500)], 1):
    for step in range(steps):
        chunks, types, y = batch(128, L)
        logits, recon, tgt = model(chunks, types)
        loss = F.cross_entropy(logits, y) + .3*F.mse_loss(recon, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0: print(f"phase {phase} (len {L}) step {step:4d} | loss {loss.item():.3f}")
    print(f"--- phase {phase} complete | acc on len {L}: {acc(L):.2%}")

for L in (2, 3, 4):
    print(f"chain len {L} ({'seen ' if L < 4 else 'UNSEEN'}) accuracy: {acc(L):.2%}")
print(f"memory ABLATED (M zeroed), len 3: {acc(3, True):.2%}   <- must collapse to ~50%")