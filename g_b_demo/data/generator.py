"""Synthetic contradiction‑rich KB generator."""
import random, torch
import builtins # Import the builtins module
from torch.nn import functional as F

N_ATOMS = 10
CONTR_RATE = 0.0  # DIAGNOSTIC: No contradictions
LABEL_NOISE = 0.0 # DIAGNOSTIC: No label noise

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
    else:
        g_target = 0.0
        b_target = 0.0
    
    # Apply label noise (currently LABEL_NOISE = 0.0)
    g_val = random.gauss(g_target, LABEL_NOISE)
    g = builtins.min(builtins.max(g_val, 0), 1)
    b_val = random.gauss(b_target, LABEL_NOISE)
    b = builtins.min(builtins.max(b_val, 0), 1)
    return torch.tensor([g, b], dtype=torch.float32)

def build_split(n_samples, cfg=None, existing_kb=None):
    xs, ys = [], []
    
    # Use existing_kb if provided, otherwise generate a new one.
    # This 'current_kb' will be used for generating all samples in this split
    # and will also be returned.
    current_kb = existing_kb if existing_kb is not None else _random_kb()
    
    possible_queries = list(current_kb.keys())

    for _ in range(n_samples):
        # Query a random atom from the current_kb
        q  = random.choice(possible_queries) 
        x  = F.one_hot(torch.tensor(int(q[1:])), N_ATOMS).float()
        y  = _encode(current_kb, q) # Use the current_kb
        xs.append(x); ys.append(y)
    X = torch.stack(xs)
    Y = torch.stack(ys)

    return X, Y, current_kb # Always return the knowledge base used 