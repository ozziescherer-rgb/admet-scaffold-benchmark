"""
Random vs. scaffold split evaluation for molecular solubility prediction.

Trains a LightGBM model on Morgan count fingerprints (ESOL dataset, 1,128
molecules) and scores it under two train/test splits:

  random    molecules assigned to train/test at random  (optimistic)
  scaffold  no Bemis-Murcko scaffold appears in both     (realistic)

A predict-the-mean baseline is included as the floor every model must beat.
Each configuration is repeated over 3 seeds so every number has a std.

Run:  python benchmark.py        ->  prints the table, writes results/results.csv
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
import lightgbm as lgb

RDLogger.DisableLog("rdApp.*")

N_SEEDS = 3
TEST_FRAC = 0.2

# 1. Load ESOL (Delaney) solubility dataset
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
df = pd.read_csv(url)
smiles = df["smiles"].tolist()
y = df["measured log solubility in mols per litre"].to_numpy()
print(f"Loaded {len(smiles)} molecules")

# 2. Featurize: SMILES -> Morgan count fingerprints (radius 2, 2048 bits)
mols = [Chem.MolFromSmiles(s) for s in smiles]
assert all(m is not None for m in mols), "a SMILES failed to parse"
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
X = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
print(f"Feature matrix: {X.shape}")


# 3a. Random split
def random_split(n, test_frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    return idx[n_test:], idx[:n_test]


# 3b. Scaffold split: whole Bemis-Murcko scaffold groups go to train OR test
def scaffold_split(smiles, test_frac, seed):
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        scaf = MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
        groups[scaf].append(i)
    scaffolds = list(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffolds)
    n_test_target = int(len(smiles) * test_frac)
    train, test = [], []
    for scaf in scaffolds:
        members = groups[scaf]
        if len(test) + len(members) <= n_test_target:
            test.extend(members)
        else:
            train.extend(members)
    return np.array(train), np.array(test)


def scaffold_overlap(train_idx, test_idx):
    """Fraction of test molecules whose scaffold also appears in training."""
    scaf = lambda i: MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles[i], includeChirality=False)
    train_scafs = {scaf(i) for i in train_idx}
    return float(np.mean([scaf(i) in train_scafs for i in test_idx]))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# 4. Train + evaluate over splits and seeds
rows = []
for split_name in ["random", "scaffold"]:
    for seed in range(N_SEEDS):
        if split_name == "random":
            train_idx, test_idx = random_split(len(smiles), TEST_FRAC, seed)
        else:
            train_idx, test_idx = scaffold_split(smiles, TEST_FRAC, seed)
        overlap = scaffold_overlap(train_idx, test_idx)

        baseline_pred = np.full(len(test_idx), y[train_idx].mean())
        rows.append({"split": split_name, "model": "predict_mean", "seed": seed,
                     "rmse": rmse(y[test_idx], baseline_pred),
                     "test_scaffold_overlap": overlap})

        model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                                  random_state=seed, verbose=-1)
        model.fit(X[train_idx], y[train_idx])
        rows.append({"split": split_name, "model": "lightgbm_fingerprints", "seed": seed,
                     "rmse": rmse(y[test_idx], model.predict(X[test_idx])),
                     "test_scaffold_overlap": overlap})

# 5. Summarize
results = pd.DataFrame(rows)
results.to_csv("results/results.csv", index=False)

summary = results.groupby(["model", "split"])["rmse"].agg(["mean", "std"]).round(3)
overlap = results.groupby("split")["test_scaffold_overlap"].mean().round(3)

print("\nRMSE (lower is better), mean and std over", N_SEEDS, "seeds:\n")
print(summary)
print("\nFraction of test molecules whose scaffold also appears in training:")
print(overlap.to_string())
print("\nSaved results/results.csv")
