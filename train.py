"""Train a sign classifier on the extracted features.

Usage :
    python train.py                # GRU, 1000 words
    python train.py lstm           # LSTM, 1000 words
    python train.py stgcn          # ST-GCN, 1000 words
    python train.py stgcn 100      # ST-GCN, 100 words

Training is always done on the gloss, never on the word.
Reads features_<vocab>/manifest_{train,val}.csv
Produces : models/model_<arch>_<vocab>.pt  (weights + normalisation + classes)

The ST-GCN only consumes dimensions 0 to 170 of the .npy files, that is 57
joints x 3 channels. The 48 derived distances and the 4 flags are dropped:
outside the graph formalism.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sign_lang.data import resample, to_graph
from sign_lang.model import NUM_JOINTS, STGCN, SignGRU

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

ARCH = sys.argv[1].lower() if len(sys.argv) > 1 else "gru"  # gru | lstm | stgcn
VOCAB = sys.argv[2] if len(sys.argv) > 2 else "1000"  # "100" or "1000"

# Always train on the gloss: one class per distinct sign. DOG1 and DOG2 are two
# different signs, merging them would hide a real recognition error. The word
# level belongs further down the chain, where the LLM needs an English word.
LABEL_COL = "gloss"

if ARCH not in ("gru", "lstm", "stgcn"):
    sys.exit(f"unknown architecture: {ARCH}  (gru | lstm | stgcn)")

IS_STGCN = ARCH == "stgcn"
FEAT = ROOT / f"features_{VOCAB}"
CKPT = ROOT / "models" / f"model_{ARCH}_{VOCAB}.pt"

SEQ_LEN = 32  # every sequence is brought to this length
BATCH = 32
BATCH_EVAL = 256  # evaluating in a single pass would saturate the VRAM
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-2
SEED = 0

CHANNELS = 3  # st-gcn: x, y, z
HIDDEN = 256  # rnn only
LAYERS = 2  # rnn only
DROPOUT = 0.5 if IS_STGCN else 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_split(split, classes=None):
    """Loads the whole split into memory. Returns (X, labels)."""
    path = FEAT / f"manifest_{split}.csv"
    if not path.exists():
        sys.exit(
            f"missing manifest: {path}\n"
            f"run first: python video_extracter.py {split}"
        )

    X, labels = [], []
    missing = unknown = 0

    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            npy = FEAT / r["npy"]
            if not npy.exists():
                missing += 1
                continue
            label = r[LABEL_COL]
            if classes is not None and label not in classes:
                unknown += 1
                continue
            seq = resample(np.load(npy).astype(np.float32), SEQ_LEN)
            X.append(to_graph(seq) if IS_STGCN else seq)
            labels.append(label)

    if not X:
        sys.exit(f"no sequence loaded for {split}")

    msg = f"{split:<5} : {len(X)} sequences"
    if missing:
        msg += f"  ({missing} .npy missing)"
    if unknown:
        msg += f"  ({unknown} classes outside train)"
    print(msg)

    return np.stack(X), labels


print(f"label     : {LABEL_COL}")
print(f"arch      : {ARCH}")
print(f"vocab     : {VOCAB}")
print(f"device    : {DEVICE}")

X_train, y_train_lab = load_split("train")
classes = sorted(set(y_train_lab))
cls_idx = {c: i for i, c in enumerate(classes)}

X_val, y_val_lab = load_split("val", classes=set(classes))

y_train = np.array([cls_idx[c] for c in y_train_lab], dtype=np.int64)
y_val = np.array([cls_idx[c] for c in y_val_lab], dtype=np.int64)

N_CLASSES = len(classes)
TOPK = min(5, N_CLASSES)
DIM = None if IS_STGCN else X_train.shape[2]

print(f"classes   : {N_CLASSES}")
if IS_STGCN:
    print(f"input     : ({CHANNELS}, {SEQ_LEN}, {NUM_JOINTS})")
else:
    print(f"dim       : {DIM}")

# Normalisation computed on the TRAIN split only.
#   st-gcn : per (channel, joint), averaging over N and T
#   rnn    : per feature, averaging over N and T
if IS_STGCN:
    mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1, 3, 1, 57)
    std = X_train.std(axis=(0, 2), keepdims=True)
else:
    mean = X_train.reshape(-1, DIM).mean(axis=0)
    std = X_train.reshape(-1, DIM).std(axis=0)
std[std < 1e-6] = 1.0

X_train = torch.from_numpy((X_train - mean) / std)
X_val = torch.from_numpy((X_val - mean) / std)
y_train = torch.from_numpy(y_train)
y_val = torch.from_numpy(y_val)

if IS_STGCN:
    mean, std = mean[0], std[0]  # (3, 1, 57), ready for the checkpoint


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
if IS_STGCN:
    model = STGCN(CHANNELS, N_CLASSES, dropout=DROPOUT).to(DEVICE)
else:
    model = SignGRU(
        DIM, N_CLASSES, hidden=HIDDEN, layers=LAYERS, cell=ARCH, dropout=DROPOUT
    ).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params / 1e6:.2f} M")
print()

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def checkpoint():
    """Contents of the .pt file. The arch-specific keys are disjoint."""
    d = {
        "state_dict": model.state_dict(),
        "classes": classes,
        "mean": mean,
        "std": std,
        "label_col": LABEL_COL,
        "seq_len": SEQ_LEN,
    }
    if IS_STGCN:
        d.update({"in_channels": CHANNELS, "num_joints": NUM_JOINTS, "arch": "stgcn"})
    else:
        d.update({"dim": DIM, "hidden": HIDDEN, "layers": LAYERS, "cell": ARCH})
    return d


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate():
    model.eval()
    total_loss = 0.0
    ok1 = okk = 0

    for i in range(0, len(X_val), BATCH_EVAL):
        xb = X_val[i : i + BATCH_EVAL].to(DEVICE)
        yb = y_val[i : i + BATCH_EVAL].to(DEVICE)

        logits = model(xb)
        total_loss += criterion(logits, yb).item() * len(yb)

        topk = logits.topk(TOPK, dim=1).indices
        ok1 += (topk[:, 0] == yb).sum().item()
        okk += (topk == yb[:, None]).any(dim=1).sum().item()

    n = len(X_val)
    return total_loss / n, ok1 / n, okk / n


best = 0.0
n_train = X_train.shape[0]
t0 = time.perf_counter()

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(n_train)
    total_loss = 0.0

    for i in range(0, n_train, BATCH):
        idx = perm[i : i + BATCH]
        xb = X_train[idx].to(DEVICE)
        yb = y_train[idx].to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(idx)

    scheduler.step()
    train_loss = total_loss / n_train
    val_loss, top1, topn = evaluate()

    flag = ""
    if top1 > best:
        best = top1
        torch.save(checkpoint(), CKPT)
        flag = "  <- best"

    print(
        f"epoch {epoch:3d}/{EPOCHS}  "
        f"train {train_loss:.3f}  val {val_loss:.3f}  "
        f"top1 {top1 * 100:5.2f}%  top{TOPK} {topn * 100:5.2f}%{flag}"
    )

print()
print(f"best top1 : {best * 100:.2f}%")
print(f"duration  : {(time.perf_counter() - t0) / 60:.1f} min")
print(f"saved to  : {CKPT}")
