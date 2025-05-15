"""Synthetic contradiction‑rich KB generator."""
import random, torch
import builtins # Import the builtins module
from torch.nn import functional as F

N_ATOMS = 300
# CONTR_RATE is not directly applicable to independent fuzzy scores in the same way.
# If contradictions are needed, a different mechanism for fuzzy targets would be required.
# CONTR_RATE = 0.0 
LABEL_NOISE = 0.0 # DIAGNOSTIC: No label noise

def _random_kb():
    atoms = [f"a{i}" for i in range(N_ATOMS)]
    kb = {}
    for a in atoms:
        # Assign independent random fuzzy scores for goodness and badness
        good_score = random.random() # Generates a float between 0.0 and 1.0
        bad_score = random.random()  # Generates a float between 0.0 and 1.0
        kb[a] = (good_score, bad_score) # Store as a tuple
    return kb

def _encode(kb, q):
    if q in kb:
        g_target_fuzzy, b_target_fuzzy = kb[q]
    else:  # q not in kb (should not happen if q is chosen from kb.keys())
        g_target_fuzzy = 0.0 # Default if atom somehow not in kb
        b_target_fuzzy = 0.0
    
    # Apply label noise (adds Gaussian noise then clips to [0,1])
    # If LABEL_NOISE is 0, g = g_target_fuzzy and b = b_target_fuzzy
    g_val = random.gauss(g_target_fuzzy, LABEL_NOISE)
    g = builtins.min(builtins.max(g_val, 0), 1)
    b_val = random.gauss(b_target_fuzzy, LABEL_NOISE)
    b = builtins.min(builtins.max(b_val, 0), 1)
    return torch.tensor([g, b], dtype=torch.float32)

def build_split(n_samples, cfg=None, existing_kb=None):
    xs, ys = [], []
    
    current_kb = existing_kb if existing_kb is not None else _random_kb()
    
    possible_queries = list(current_kb.keys())

    if not possible_queries: # Handle empty KB case
        # This case should ideally not happen if N_ATOMS > 0
        # but as a safeguard:
        for _ in range(n_samples):
            x = torch.zeros(N_ATOMS).float() # Dummy input
            y = torch.tensor([0.5, 0.5], dtype=torch.float32) # Neutral fuzzy target
            xs.append(x); ys.append(y)
        X = torch.stack(xs)
        Y = torch.stack(ys)
        return X, Y, current_kb


    for _ in range(n_samples):
        q  = random.choice(possible_queries) 
        x  = F.one_hot(torch.tensor(int(q[1:])), N_ATOMS).float()
        y  = _encode(current_kb, q)
        xs.append(x); ys.append(y)
    X = torch.stack(xs)
    Y = torch.stack(ys)

    return X, Y, current_kb 