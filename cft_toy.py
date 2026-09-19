# cft_toy.py — Toy Micro-CFT: Atomic Embedding + BDH-CQ + Fleeting Memory + Fidelity Loss
# Task: transitive logic. Facts streamed ONE CHUNK AT A TIME into fixed-size memory,
# then the query is answered FROM MEMORY ALONE. Run: python cft_toy.py  (CPU, ~2-4 min)
import random, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); random.seed(0)
D, N_SLOTS = 64, 8                      # hidden dim, fleeting-memory slots (fixed forever = O(1))

# ---------- SECTION 1: SYNTHETIC DATA (A > B > C ... chains) ----------
SYMS = "ABCDEFGH"
TOK = {c: i+1 for i, c in enumerate(SYMS)}; TOK.update({">": 9, "<": 10, "?": 11})
def make_sample(L):
    order = random.sample(SYMS, L+1)                       # order[0] > order[1] > ...
    facts = [(order[i], order[i+1]) for i in range(L)]
    i, j = sorted(random.sample(range(L+1), 2))
    qa, qb, lab = (order[i], order[j], 1) if random.random() < .5 else (order[j], order[i], 0)
    return facts, (qa, qb), lab
def ty(bs, op):
    t = torch.full((bs, 3), 1).long(); t[:, 1] = op; return t   # types: 1=symbol, 2=rel, 3=query
def batch(bs, L):
    S = [make_sample(L) for _ in range(bs)]
    chunks, types = [], []
    for k in range(L):                                     # one 3-token chunk per fact
        chunks.append(torch.tensor([[TOK[f[k][0]], 9, TOK[f[k][1]]] for f, _, _ in S]))
        types.append(ty(bs, 2))
    chunks.append(torch.tensor([[TOK[q[0]], 11, TOK[q[1]]] for _, q, _ in S]))  # query chunk
    types.append(ty(bs, 3))
    return chunks, types, torch.tensor([l for *_, l in S])

# ---------- SECTION 2: THE TOY CFT ----------
class AtomicEmbedding(nn.Module):                          # token + type = grounded primitives
    def __init__(self):
        super().__init__(); self.tok = nn.Embedding(12, D); self.typ = nn.Embedding(4, D)
    def forward(self, i, t): return self.tok(i) + self.typ(t)

class BDHCQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.S = nn.Parameter(torch.randn(N_SLOTS, D)*.02) # cognitive slot queries
        self.wk, self.wv, self.wr, self.wl = (nn.Linear(D, D, bias=False) for _ in range(4))
        self.wg, self.wg2, self.wo = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D, bias=False)
    def parts(self, x):                                    # BREAK DOWN -> RELATE
        A  = torch.softmax(self.S @ self.wk(x).transpose(1, 2)/D**.5, -1)   # slots query input
        P  = A @ self.wv(x)                                                 # (B,N,D) decomposed parts
        R  = self.wl(P)
        Ar = torch.softmax(self.wr(P) @ R.transpose(1, 2)/D**.5, -1)        # relational graph
        return P + Ar @ R                                                   # parts + relations
    def write(self, Pr, M, g):                             # REDUCE -> decay + gated write
        Md = M * g                                         # synaptic decay of old memory
        Aw = torch.softmax(Pr @ Md.transpose(1, 2)/D**.5, -1)               # route parts->slots
        G  = torch.sigmoid(self.wg(Pr) + self.wg2(Md))                      # overwrite gate
        return Md + G * (Aw.transpose(1, 2) @ self.wo(Pr))                  # new fixed-size M

class CFTToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb   = AtomicEmbedding()
        self.local = nn.TransformerEncoderLayer(D, 4, 128, dropout=0.0, batch_first=True)
        self.bdh   = BDHCQ()
        self.g     = nn.Parameter(torch.zeros(N_SLOTS, 1)) # learnable decay rates
        self.rq    = nn.Linear(D, D, bias=False)           # query reads memory (cognitive query)
        self.head  = nn.Sequential(nn.Linear(2*D, 64), nn.ReLU(), nn.Linear(64, 2))
        self.fid   = nn.Linear(N_SLOTS*D, D)               # fidelity decoder
    def forward(self, chunks, types, ablate=False):
        B = chunks[0].size(0); M = torch.zeros(B, N_SLOTS, D); g = torch.sigmoid(self.g)
        pools = []
        for c, t in zip(chunks[:-1], types[:-1]):          # stream facts into fleeting memory
            x = self.local(self.emb(c, t)); pools.append(x.mean(1))
            M = self.bdh.write(self.bdh.parts(x), M, g)
        Pq = self.bdh.parts(self.local(self.emb(chunks[-1], types[-1])))
        Mr = torch.zeros_like(M) if ablate else M          # ablation probe: kill memory
        A  = torch.softmax(self.rq(Pq) @ Mr.transpose(1, 2)/D**.5, -1)
        logits = self.head(torch.cat([Pq.mean(1), (A @ Mr).mean(1)], -1))
        return logits, self.fid(M.flatten(1)), torch.stack(pools, 1).mean(1).detach()

# ---------- SECTION 3: TRAIN + PROOF ----------
model = CFTToy(); print("params:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.Adam(model.parameters(), 2e-3)
for step in range(1501):
    chunks, types, y = batch(128, random.choice([2, 3]))
    logits, recon, tgt = model(chunks, types)
    loss = F.cross_entropy(logits, y) + .3*F.mse_loss(recon, tgt)   # task + FIDELITY loss
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0: print(f"step {step:4d} | loss {loss.item():.3f}")

@torch.no_grad()
def acc(L, ablate=False, n=400):
    c, t, y = batch(n, L); lo, _, _ = model(c, t, ablate)
    return (lo.argmax(1) == y).float().mean().item()

for L in (2, 3, 4):
    print(f"chain len {L} ({'seen ' if L < 4 else 'UNSEEN'}) accuracy: {acc(L):.2%}")
print(f"memory ABLATED (M zeroed), len 3: {acc(3, True):.2%}   <- should collapse to ~50%")