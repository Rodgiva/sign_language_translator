"""Sign classification models: a recurrent baseline and an ST-GCN."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignGRU(nn.Module):
    """Projection -> bidirectional RNN -> mean+max pooling -> classification.

    `cell` is either "gru" or "lstm". It is a parameter rather than a global
    constant, so that the class stays usable from several scripts.
    """

    def __init__(
        self, dim_in, n_classes, hidden=256, layers=2, cell="gru", dropout=0.3
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        rnn_cls = nn.LSTM if cell.lower() == "lstm" else nn.GRU
        self.gru = rnn_cls(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, n_classes),
        )

    def forward(self, x):  # x: (B, T, DIM)
        h = self.proj(x)
        out, _ = self.gru(h)  # (B, T, 2*hidden)
        pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.head(pooled)


# ----------------------------------------------------------------------------
# Graph topology: 57 joints
#     0  .. 20 : left hand (MediaPipe topology)
#     21 .. 41 : right hand
#     42 .. 56 : pose (order of data.POSE_KEEP)
# ----------------------------------------------------------------------------
NUM_JOINTS = 57
L_HAND0, R_HAND0, POSE0 = 0, 21, 42

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # little
    (0, 17),                                  # palm
]

# LOCAL indices inside POSE_KEEP:
# 0 NOSE, 1 L_EYE, 2 R_EYE, 3 L_EAR, 4 R_EAR, 5 MOUTH_L, 6 MOUTH_R,
# 7 L_SHOULDER, 8 R_SHOULDER, 9 L_ELBOW, 10 R_ELBOW,
# 11 L_WRIST, 12 R_WRIST, 13 L_HIP, 14 R_HIP
POSE_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # nose - eyes - ears
    (0, 5), (0, 6), (5, 6),                   # nose - mouth
    (0, 7), (0, 8), (7, 8),                   # head - shoulders
    (7, 9), (9, 11),                          # left arm
    (8, 10), (10, 12),                        # right arm
    (7, 13), (8, 14), (13, 14),               # torso
]

# Bridges between each hand and the matching pose wrist. Without them the graph
# has three disjoint components and no information flows between hands and body.
BRIDGES = [
    (L_HAND0 + 0, POSE0 + 11),
    (R_HAND0 + 0, POSE0 + 12),
]

# Centre of gravity of the graph: both shoulders.
CENTER = [POSE0 + 7, POSE0 + 8]


def _edge_list():
    edges = []
    for a, b in HAND_EDGES:
        edges.append((L_HAND0 + a, L_HAND0 + b))
        edges.append((R_HAND0 + a, R_HAND0 + b))
    for a, b in POSE_EDGES:
        edges.append((POSE0 + a, POSE0 + b))
    return edges + BRIDGES


def _hop_distance(A):
    """Distance to the centre, counted in edges."""
    dist = np.full(NUM_JOINTS, np.inf)
    frontier = list(CENTER)
    for c in CENTER:
        dist[c] = 0
    d = 0
    while frontier:
        nxt = []
        for u in frontier:
            for v in np.flatnonzero(A[u]):
                if np.isinf(dist[v]):
                    dist[v] = d + 1
                    nxt.append(int(v))
        frontier = nxt
        d += 1
    return dist


def _normalize(A):
    """Normalisation by in-degree (columns)."""
    deg = A.sum(axis=0)
    deg[deg == 0] = 1.0
    return A / deg


def build_adjacency():
    """A of shape (3, V, V): spatial configuration partitioning.

    0 : the joint itself
    1 : neighbours closer to the centre  (centripetal)
    2 : neighbours further from the centre (centrifugal)
    """
    A = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    for a, b in _edge_list():
        A[a, b] = A[b, a] = 1.0

    dist = _hop_distance(A)

    a_self = np.eye(NUM_JOINTS, dtype=np.float32)
    a_in = np.zeros_like(A)
    a_out = np.zeros_like(A)

    # The einsum gathers x[v] into output[w], so A[v, w] describes edge v -> w.
    for v in range(NUM_JOINTS):
        for w in range(NUM_JOINTS):
            if A[v, w]:
                if dist[v] < dist[w]:
                    a_in[v, w] = 1.0
                else:
                    a_out[v, w] = 1.0

    return np.stack([_normalize(a_self), _normalize(a_in), _normalize(a_out)])


# ----------------------------------------------------------------------------
# ST-GCN
# ----------------------------------------------------------------------------
class GraphConv(nn.Module):
    """Spatial convolution: 1x1 per subset, then aggregation over the graph."""

    def __init__(self, in_channels, out_channels, num_subsets):
        super().__init__()
        self.num_subsets = num_subsets
        self.conv = nn.Conv2d(in_channels, out_channels * num_subsets, kernel_size=1)

    def forward(self, x, A):  # x: (N, C, T, V)
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.num_subsets, kc // self.num_subsets, t, v)
        return torch.einsum("nkctv,kvw->nctw", x, A)


class STGCNBlock(nn.Module):
    """Spatial graph convolution, then a 9x1 temporal convolution."""

    def __init__(self, in_channels, out_channels, num_subsets,
                 stride=1, residual=True, dropout=0.5, temporal_kernel=9):
        super().__init__()
        pad = (temporal_kernel - 1) // 2

        self.gcn = GraphConv(in_channels, out_channels, num_subsets)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(temporal_kernel, 1),
                      stride=(stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = 0 if self.residual is None else self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


class STGCN(nn.Module):
    """Input: (N, C, T, V). Output: logits (N, n_classes)."""

    LAYERS = [
        # (in, out, stride, residual)
        (64, 64, 1, True),
        (64, 64, 1, True),
        (64, 64, 1, True),
        (64, 128, 2, True),
        (128, 128, 1, True),
        (128, 128, 1, True),
        (128, 256, 2, True),
        (256, 256, 1, True),
        (256, 256, 1, True),
    ]

    def __init__(self, in_channels, n_classes, dropout=0.5, edge_importance=True):
        super().__init__()

        A = torch.from_numpy(build_adjacency())
        self.register_buffer("A", A)
        num_subsets = A.size(0)
        v = A.size(1)

        self.data_bn = nn.BatchNorm1d(in_channels * v)

        blocks = [STGCNBlock(in_channels, 64, num_subsets,
                             stride=1, residual=False, dropout=dropout)]
        for c_in, c_out, stride, res in self.LAYERS:
            blocks.append(STGCNBlock(c_in, c_out, num_subsets,
                                     stride=stride, residual=res,
                                     dropout=dropout))
        self.blocks = nn.ModuleList(blocks)

        if edge_importance:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones_like(A)) for _ in self.blocks]
            )
        else:
            self.edge_importance = [1.0] * len(self.blocks)

        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):  # x: (N, C, T, V)
        n, c, t, v = x.size()

        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()

        for block, importance in zip(self.blocks, self.edge_importance):
            x = block(x, self.A * importance)

        x = F.avg_pool2d(x, x.size()[2:]).view(n, -1)
        return self.fc(x)
