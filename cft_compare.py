# cft_compare.py — CFT (streaming, fixed memory) vs standard Transformer (full context)
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
def batch(bs, L):                                   # chunked (for CFT)
    S = [make_sample(L) for _ in range(bs)]; chunks, types = [], []
    for k in range(L):
        chunks.append(torch.tensor([[TOK[f[k][0]], 9, TOK[f[k][1]]] for f, _, _ in S])); types.append(ty(bs, 2))
    chunks.append(torch.tensor([[TOK[q[0]], 11, TOK[q[1]]] for _, q, _ in S])); types.append(ty(bs, 3))
    return chunks, types, torch.tensor([l for *_, l in S])
def batch_flat(bs, L):                              # one long sequence (for baseline)
    ids, types, y = [], [], []
    for _ in range(bs):
        f, q, l = make_sample(L); ri, rt = [], []
        for a, b in f: ri += [TOK[a], 9, TOK[b]]; rt += [1, 2, 1]
        ri += [TOK[q[0]], 11, TOK[q[1]]]; rt += [1, 3, 1]
        ids.append(ri); types.append(rt); y.append(l)
    return torch.tensor(ids), torch.tensor(types), torch.tensor(y)

class BDHCQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.S = nn.Parameter(torch.randn(N_SLOTS, D)*.02)
        self.wk, self.wv, self.wr, self.wl = (nn.Linear(D, D, bias=False) for _ in range(4))
        self.wg, self.wg2, self.wo = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D, bias=False)
    def parts(self, x):
        A = torch.softmax(self.S @ self.wk(x).transpose(1, 2)/D**.5, -1)
        P = A @ self.wv(x); R = self.wl(P)
        return P + torch.softmax(self.wr(P) @ R.transpose(1, 2)/D**.5, -1) @ R
    def write(self, Pr, M, g):
        Md = M * g
        Aw = torch.softmax(Pr @ Md.transpose(1, 2)/D**.5, -1)
        Upd = Aw.transpose(1, 2) @ self.wo(Pr)
        return Md + torch.sigmoid(self.wg(Upd) + self.wg2(Md)) * Upd

class CFTToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.typ, self.pos = nn.Embedding(12, D), nn.Embedding(4, D), nn.Embedding(3, D)
        self.local = nn.TransformerEncoderLayer(D, 4, 128, dropout=0.0, batch_first=True)
        self.bdh = BDHCQ(); self.M0 = nn.Parameter(torch.randn(N_SLOTS, D)*.1)
        self.g = nn.Parameter(torch.zeros(N_SLOTS, 1)); self.rq = nn.Linear(D, D, bias=False)
        self.head = nn.Sequential(nn.Linear(2*D, 64), nn.ReLU(), nn.Linear(64, 2))
        self.fid = nn.Linear(N_SLOTS*D, D)
    def embed(self, c, t): return self.tok(c) + self.typ(t) + self.pos(torch.arange(c.size(1)))
    def forward(self, chunks, types):
        B = chunks[0].size(0); M = self.M0.unsqueeze(0).expand(B, -1, -1); g = torch.sigmoid(self.g)
        pools = []
        for c, t in zip(chunks[:-1], types[:-1]):
            x = self.local(self.embed(c, t)); pools.append(x.mean(1))
            M = self.bdh.write(self.bdh.parts(x), M, g)
        Pq = self.bdh.parts(self.local(self.embed(chunks[-1], types[-1])))
        A = torch.softmax(self.rq(Pq) @ M.transpose(1, 2)/D**.5, -1)
        return (self.head(torch.cat([Pq.mean(1), (A @ M).mean(1)], -1)),
                self.fid(M.flatten(1)), torch.stack(pools, 1).mean(1).detach())

class BaselineTF(nn.Module):                        # standard Transformer, sees EVERYTHING at once
    def __init__(self):
        super().__init__()
        self.tok, self.typ, self.pos = nn.Embedding(12, D), nn.Embedding(4, D), nn.Embedding(64, D)
        self.enc = nn.TransformerEncoderLayer(D, 4, 128, dropout=0.0, batch_first=True)
        self.head = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, ids, types):
        x = self.tok(ids) + self.typ(types) + self.pos(torch.arange(ids.size(1)))
        return self.head(self.enc(x).mean(1))

PHASES = [(1, 2000), (2, 2500), (3, 1500)]
cft, base = CFTToy(), BaselineTF()
print("params CFT / baseline:", sum(p.numel() for p in cft.parameters()), sum(p.numel() for p in base.parameters()))
oc, ob = torch.optim.Adam(cft.parameters(), 1e-3), torch.optim.Adam(base.parameters(), 1e-3)
for phase, (L, steps) in enumerate(PHASES, 1):
    for step in range(steps):
        ch, ty_, y = batch(128, L)
        lo, rec, tgt = cft(ch, ty_)
        lc = F.cross_entropy(lo, y) + .3*F.mse_loss(rec, tgt)
        oc.zero_grad(); lc.backward(); oc.step()
        i, t, y = batch_flat(128, L)
        lb = F.cross_entropy(base(i, t), y)
        ob.zero_grad(); lb.backward(); ob.step()
    print(f"phase {phase} (len {L}) done | cft loss {lc.item():.3f} | base loss {lb.item():.3f}")

@torch.no_grad()
def acc_cft(L, n=400):
    ch, t, y = batch(n, L); lo, _, _ = cft(ch, t); return (lo.argmax(1) == y).float().mean().item()
@torch.no_grad()
def acc_base(L, n=400):
    i, t, y = batch_flat(n, L); return (base(i, t).argmax(1) == y).float().mean().item()

print("\n--- ACCURACY ---")
for L in (2, 3, 4):
    print(f"len {L}: CFT {acc_cft(L):.2%} | Baseline(full context) {acc_base(L):.2%}")
print("\n--- INFERENCE MEMORY FOOTPRINT (fp32 bytes) ---")
print(f"{'chain':>5} | {'tokens seen':>11} | {'baseline KV cache':>17} | {'CFT fleeting memory':>21}")
for L in (2, 4, 8, 16, 64):
    T = 3*(L+1)
    print(f"{L:>5} | {T:>11} | {2*1*T*D*4:>17,} | {N_SLOTS*D*4:>21,}   (CFT never sees >3 tokens at once)")