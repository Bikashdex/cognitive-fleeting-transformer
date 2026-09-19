# cft_rule_world_colab.py — The "Rule World" Experiment on GPU
# Task: Learn "Red IS Fire, Fire CAUSES Danger -> Red IS Danger"
# Tests if CFT learns universal rules, not just memorized words.

import random, torch, torch.nn as nn, torch.nn.functional as F

# --- 1. SETUP: Auto-detect GPU (The Sports Car) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Running on: {device} (If it says 'cuda', your free GPU is active!)")

torch.manual_seed(42); random.seed(42)
D, N_SLOTS = 64, 8

# --- 2. RULE WORLD VOCABULARY ---
VOCAB = {
    "Red":1, "Blue":2, "Green":3, "Yellow":4, 
    "Fire":5, "Water":6, "Ice":7, "Lava":8, 
    "Danger":9, "Safe":10, "IS":11, "CAUSES":12, "?":13
}

# Ground truth rules of our fake world
RULES = {
    "Red": ("Fire", "Danger"), "Blue": ("Water", "Safe"),
    "Green": ("Ice", "Safe"), "Yellow": ("Lava", "Danger")
}

def make_rule_sample(train_entities=["Red", "Blue"], test_unseen=False):
    if test_unseen:
        entity = random.choice(["Green", "Yellow"]) # The "Never Seen" test
    else:
        entity = random.choice(train_entities)
        
    medium, outcome = RULES[entity]
    facts = [
        (VOCAB[entity], VOCAB["IS"], VOCAB[medium]),
        (VOCAB[medium], VOCAB["CAUSES"], VOCAB[outcome])
    ]
    random.shuffle(facts) 
    label = 0 if outcome == "Danger" else 1
    return facts, (VOCAB[entity], VOCAB["IS"], VOCAB["?"]), label

def ty(bs, op_type):
    t = torch.full((bs, 3), 4).long() 
    if op_type == 'fact': t[:, 0] = 1; t[:, 2] = 2
    elif op_type == 'rule': t[:, 0] = 2; t[:, 2] = 3
    elif op_type == 'query': t[:, 0] = 1; t[:, 2] = 5
    return t

def batch(bs, train_entities=["Red", "Blue"], test_unseen=False):
    samples = [make_rule_sample(train_entities, test_unseen) for _ in range(bs)]
    chunks, types = [], []
    IS_TOKEN = VOCAB["IS"] # FIX: Get ID once to prevent KeyError
    
    for k in range(2):
        op_type = 'fact' if samples[0][0][k][1] == IS_TOKEN else 'rule'
        chunks.append(torch.tensor([[s[0][k][0], s[0][k][1], s[0][k][2]] for s in samples]))
        types.append(ty(bs, op_type))
        
    chunks.append(torch.tensor([[s[1][0], s[1][1], s[1][2]] for s in samples]))
    types.append(ty(bs, 'query'))
    return chunks, types, torch.tensor([s[2] for s in samples])

# --- 3. THE PROVEN CFT ARCHITECTURE ---
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

class CFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.typ, self.pos = nn.Embedding(16, D), nn.Embedding(6, D), nn.Embedding(3, D)
        self.local = nn.TransformerEncoderLayer(D, 4, 128, dropout=0.0, batch_first=True)
        self.bdh = BDHCQ(); self.M0 = nn.Parameter(torch.randn(N_SLOTS, D)*.1)
        self.g = nn.Parameter(torch.zeros(N_SLOTS, 1)); self.rq = nn.Linear(D, D, bias=False)
        self.head = nn.Sequential(nn.Linear(2*D, 64), nn.ReLU(), nn.Linear(64, 2))
        self.fid = nn.Linear(N_SLOTS*D, D)
    def embed(self, c, t): return self.tok(c) + self.typ(t) + self.pos(torch.arange(c.size(1)))
    def forward(self, chunks, types, ablate=False):
        B = chunks[0].size(0); M = self.M0.unsqueeze(0).expand(B, -1, -1); g = torch.sigmoid(self.g)
        for c, t in zip(chunks[:-1], types[:-1]):
            x = self.local(self.embed(c, t))
            M = self.bdh.write(self.bdh.parts(x), M, g)
        Pq = self.bdh.parts(self.local(self.embed(chunks[-1], types[-1])))
        Mr = torch.zeros_like(M) if ablate else M
        A = torch.softmax(self.rq(Pq) @ Mr.transpose(1, 2)/D**.5, -1)
        return self.head(torch.cat([Pq.mean(1), (A @ Mr).mean(1)], -1)), self.fid(M.flatten(1))

# --- 4. TRAINING LOOP ---
model = CFT().to(device)
print(f" CFT Parameters: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.Adam(model.parameters(), 1e-3)

print("\n--- Training on Red & Blue Rules (2000 steps) ---")
for step in range(2001):
    chunks, types, y = batch(128, train_entities=["Red", "Blue"], test_unseen=False)
    chunks = [c.to(device) for c in chunks]; types = [t.to(device) for t in types]; y = y.to(device)
    
    logits, recon = model(chunks, types)
    loss = F.cross_entropy(logits, y) + 0.3 * F.mse_loss(recon, torch.zeros_like(recon))
    opt.zero_grad(); loss.backward(); opt.step()
    
    if step % 500 == 0: print(f"  Step {step}/2000 | Loss: {loss.item():.3f}")

# --- 5. THE MOMENT OF TRUTH ---
@torch.no_grad()
def test_acc(test_unseen, ablate=False, n=400):
    c, t, y = batch(n, train_entities=["Red", "Blue"], test_unseen=test_unseen)
    c = [x.to(device) for x in c]; t = [x.to(device) for x in t]; y = y.to(device)
    lo, _ = model(c, t, ablate)
    return (lo.argmax(1) == y).float().mean().item()

print("\n---  RULE WORLD RESULTS ---")
seen_acc = test_acc(test_unseen=False)
unseen_acc = test_acc(test_unseen=True)
ablated_acc = test_acc(test_unseen=True, ablate=True)

print(f"✅ Accuracy on SEEN entities (Red/Blue): {seen_acc:.2%}")
print(f"🌟 Accuracy on UNSEEN entities (Green/Yellow): {unseen_acc:.2%}  <-- THE HOLY GRAIL")
print(f" Accuracy with Memory ABLATED (Zeroed): {ablated_acc:.2%}  <-- PROOF OF REASONING")
print("\n(If Unseen is > 70%, your model learned the universal rule of transitivity!)")