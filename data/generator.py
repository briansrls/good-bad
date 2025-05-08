"""Synthetic contradiction‑rich KB generator."""
import random, torch
from torch.nn import functional as F

N_ATOMS = 10
CONTR_RATE = 0.20
LABEL_NOISE = 0.15

def _random_kb():
    atoms = [f"a{i}" for i in range(N_ATOMS)]
    kb = {}
    for a in atoms:
        val = random.random() > 0.5
        if random.random() < CONTR_RATE:
            val = not val                      # inject contradiction
        kb[a] = val
    return kb

def _encode(kb, q):
    if q in kb:
        g_target = 1.0 if kb[q] else 0.0
        b_target = 1.0 if not kb[q] else 0.0
    else:  # q not in kb
        g_target = 0.0
        b_target = 0.0
    
    # Apply label noise
    g = min(max(random.gauss(g_target, LABEL_NOISE), 0), 1)
    b = min(max(random.gauss(b_target, LABEL_NOISE), 0), 1)
    return torch.tensor([g, b])

def build_split(n_samples: int):
    xs, ys = [], []
    for _ in range(n_samples):
        kb = _random_kb()
        q  = random.choice(list(kb))
        x  = F.one_hot(torch.tensor(int(q[1:])), N_ATOMS).float()
        y  = _encode(kb, q)
        xs.append(x); ys.append(y)
    return torch.stack(xs), torch.stack(ys) 