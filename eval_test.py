"""Evaluation on the test split.

Usage :
    python eval_test.py                 # model_gru_1000.pt
    python eval_test.py lstm            # model_lstm_1000.pt
    python eval_test.py stgcn 100       # model_stgcn_100.pt

Requires : models/model_<arch>_<vocab>.pt
           features_<vocab>/manifest_test.csv

Produces : confusions/confusion_<arch>_<vocab>.npy
           (confusion matrix, rows = true class)
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch

from sign_lang.data import resample, to_graph
from sign_lang.model import SignGRU, STGCN

ROOT = Path(__file__).resolve().parent

CELL = sys.argv[1].lower() if len(sys.argv) > 1 else "gru"
VOCAB = sys.argv[2] if len(sys.argv) > 2 else "1000"  # "100" or "1000"

FEAT = ROOT / f"features_{VOCAB}"
CKPT = ROOT / "models" / f"model_{CELL}_{VOCAB}.pt"
CONF_OUT = ROOT / "confusions" / f"confusion_{CELL}_{VOCAB}.npy"

BATCH = 256  # batched evaluation: 14 000 sequences at once saturate the VRAM
TOP_CONF = 25  # number of confused pairs displayed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)

classes = ckpt["classes"]
mean, std = ckpt["mean"], ckpt["std"]
SEQ_LEN = ckpt["seq_len"]
LABEL_COL = ckpt["label_col"]  # evaluate on the same label used for training

IS_STGCN = ckpt.get("arch") == "stgcn"

if IS_STGCN:
    model = STGCN(ckpt["in_channels"], len(classes)).to(DEVICE)
else:
    model = SignGRU(
        ckpt["dim"],
        len(classes),
        hidden=ckpt["hidden"],
        layers=ckpt["layers"],
        cell=ckpt.get("cell", "gru"),
    ).to(DEVICE)

model.load_state_dict(ckpt["state_dict"])
model.eval()

cls_idx = {c: i for i, c in enumerate(classes)}
N_CLASSES = len(classes)
TOPK = min(5, N_CLASSES)

print(f"model     : {CKPT.name}")
print(f"vocab     : {VOCAB}")

if IS_STGCN:
    print(
        f"arch      : st-gcn ({ckpt['in_channels']}, {SEQ_LEN}, {ckpt['num_joints']})"
    )
else:
    print(f"cell      : {ckpt.get('cell', 'gru')}")
print(f"label     : {LABEL_COL}")
print(f"classes   : {N_CLASSES}")
print(f"device    : {DEVICE}")

# ----------------------------------------------------------------------------
# Test data
# ----------------------------------------------------------------------------
manifest = FEAT / "manifest_test.csv"
if not manifest.exists():
    sys.exit(
        f"missing manifest: {manifest}\n"
        f"run first: python video_extracter.py test"
    )

X, y = [], []
missing = unknown = 0

with open(manifest, encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        npy = FEAT / r["npy"]
        if not npy.exists():
            missing += 1
            continue
        label = r[LABEL_COL]
        if label not in cls_idx:
            unknown += 1
            continue
        seq = resample(np.load(npy).astype(np.float32), SEQ_LEN)
        X.append(to_graph(seq) if IS_STGCN else seq)
        y.append(cls_idx[label])

if not X:
    sys.exit("no test sequence loaded")

X = np.stack(X)  # (N, 3, T, 57) for st-gcn, (N, T, 223) otherwise
y = np.asarray(y, dtype=np.int64)

msg = f"test      : {len(X)} sequences"
if missing:
    msg += f"  ({missing} .npy missing)"
if unknown:
    msg += f"  ({unknown} classes outside the model)"
print(msg)

# Normalisation with the statistics FROM THE CHECKPOINT.
# Recomputing them on the test split would leak information.
# For st-gcn, mean/std have shape (3, 1, 57): broadcasting covers N and T.
X = torch.from_numpy((X - mean) / std)
y_true = torch.from_numpy(y)

# ----------------------------------------------------------------------------
# Batched inference
# ----------------------------------------------------------------------------
pred1 = torch.empty(len(X), dtype=torch.int64)
hit_k = torch.zeros(len(X), dtype=torch.bool)

with torch.no_grad():
    for i in range(0, len(X), BATCH):
        xb = X[i : i + BATCH].to(DEVICE)
        yb = y_true[i : i + BATCH].to(DEVICE)

        logits = model(xb)
        topk = logits.topk(TOPK, dim=1).indices

        pred1[i : i + BATCH] = topk[:, 0].cpu()
        hit_k[i : i + BATCH] = (topk == yb[:, None]).any(dim=1).cpu()

top1 = (pred1 == y_true).float().mean().item()
topn = hit_k.float().mean().item()

print()
print(f"top1  : {top1 * 100:.2f}%")
print(f"top{TOPK}  : {topn * 100:.2f}%")

# ----------------------------------------------------------------------------
# Confusion matrix
# ----------------------------------------------------------------------------
conf = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int32)
np.add.at(conf, (y_true.numpy(), pred1.numpy()), 1)
np.save(CONF_OUT, conf)

support = conf.sum(axis=1)  # number of test examples per class
seen = support > 0
per_class = np.zeros(N_CLASSES)
per_class[seen] = np.diag(conf)[seen] / support[seen]

print()
print(f"classes evaluated  : {int(seen.sum())} / {N_CLASSES}")
print(
    f"examples per class : min={support[seen].min()} "
    f"median={int(np.median(support[seen]))} max={support[seen].max()}"
)
print(
    f"per-class accuracy (unweighted mean) : "
    f"{per_class[seen].mean() * 100:.2f}%"
)
print(f"classes never recognised : {int((per_class[seen] == 0).sum())}")

# Most confused pairs
off = conf.copy()
np.fill_diagonal(off, 0)
flat = np.argsort(off.ravel())[::-1][:TOP_CONF]

print()
print(f"{TOP_CONF} most confused pairs:")
print(f"{'true class':<22} {'predicted':<22} {'n':>5}  {'% of the class':>14}")
for f in flat:
    t, p = divmod(int(f), N_CLASSES)
    n = off[t, p]
    if n == 0:
        break
    part = 100 * n / support[t] if support[t] else 0.0
    print(f"{classes[t]:<22} {classes[p]:<22} {n:>5}  {part:>13.0f}%")

print()
print(f"matrix saved to : {CONF_OUT.name}  ({N_CLASSES}x{N_CLASSES})")
