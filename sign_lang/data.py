"""Feature definition and MediaPipe access.

Single source of truth for the feature vector layout.
"""

import re
from pathlib import Path

import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# ----------------------------------------------------------------------------
# Feature vector definition
# ----------------------------------------------------------------------------
PL = vision.PoseLandmark
L_SH, R_SH = int(PL.LEFT_SHOULDER), int(PL.RIGHT_SHOULDER)

POSE_KEEP = [
    int(PL.NOSE),
    int(PL.LEFT_EYE),
    int(PL.RIGHT_EYE),
    int(PL.LEFT_EAR),
    int(PL.RIGHT_EAR),
    int(PL.MOUTH_LEFT),
    int(PL.MOUTH_RIGHT),
    L_SH,
    R_SH,
    int(PL.LEFT_ELBOW),
    int(PL.RIGHT_ELBOW),
    int(PL.LEFT_WRIST),
    int(PL.RIGHT_WRIST),
    int(PL.LEFT_HIP),
    int(PL.RIGHT_HIP),
]

FACE_ANCHORS = [10, 168, 1, 13, 14, 152, 61, 291, 33, 263, 234, 454]

N_HAND, N_POSE, N_ANCH = 21, len(POSE_KEEP), len(FACE_ANCHORS)
DIM = 2 * N_HAND * 3 + N_POSE * 3 + N_ANCH * 2 * 2 + 4  # = 223
WRIST = 0

# Active span segmentation
GAP = 4  # fills detection gaps shorter than 4 frames
MIN_LEN = 10  # rejects spans shorter than 10 frames
MARGIN = 3  # margin added on each side


def base_word(gloss):
    """ENVELOPE3 -> ENVELOPE. Strips the numeric variant suffix."""
    return re.sub(r"\d+$", "", gloss)


def split_hands(hand_result):
    """Splits the detected hands into (left, right) using handedness."""
    left = right = None
    for lms, hd in zip(hand_result.hand_landmarks, hand_result.handedness):
        name = hd[0].category_name
        if name == "Left" and left is None:
            left = lms
        elif name == "Right" and right is None:
            right = lms
    return left, right


def extract(hand_result, pose_result, face_result):
    """One float32 vector of DIM values, for a single frame."""
    pose = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
    face = face_result.face_landmarks[0] if face_result.face_landmarks else None
    left, right = split_hands(hand_result)

    # Body frame: origin = shoulder midpoint, scale = shoulder width
    if pose is not None:
        ls, rs = pose[L_SH], pose[R_SH]
        ox, oy = (ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0
        scale = float(np.hypot(ls.x - rs.x, ls.y - rs.y))
    elif face is not None:  # fallback: inter-ocular distance
        a, b = face[33], face[263]
        ox, oy = face[1].x, face[1].y
        scale = float(np.hypot(a.x - b.x, a.y - b.y))
    else:
        return np.zeros(DIM, dtype=np.float32)

    if scale < 1e-6:
        return np.zeros(DIM, dtype=np.float32)

    def norm(p):
        return [(p.x - ox) / scale, (p.y - oy) / scale, p.z / scale]

    f = []

    # 1. Both hands in the body frame                    -> 126
    for hand in (left, right):
        if hand is None:
            f.extend([0.0] * (N_HAND * 3))
        else:
            for p in hand:
                f.extend(norm(p))

    # 2. Upper body                                      -> 45
    if pose is None:
        f.extend([0.0] * (N_POSE * 3))
    else:
        for i in POSE_KEEP:
            f.extend(norm(pose[i]))

    # 3. Wrist -> face anchor vectors (x, y)              -> 48
    for idx in FACE_ANCHORS:
        for hand in (left, right):
            if face is None or hand is None:
                f.extend([0.0, 0.0])
            else:
                a, w = face[idx], hand[WRIST]
                f.extend([(w.x - a.x) / scale, (w.y - a.y) / scale])

    # 4. Detection flags                                  -> 4
    f.extend(
        [
            1.0 if left is not None else 0.0,
            1.0 if right is not None else 0.0,
            1.0 if face is not None else 0.0,
            1.0 if pose is not None else 0.0,
        ]
    )

    return np.asarray(f, dtype=np.float32)


def resample(seq, length):
    """Temporally interpolates a sequence to `length` frames."""
    n = seq.shape[0]
    if n == length:
        return seq
    src = np.linspace(0.0, 1.0, num=n)
    dst = np.linspace(0.0, 1.0, num=length)
    out = np.empty((length, seq.shape[1]), dtype=np.float32)
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(dst, src, seq[:, c])
    return out


def active_span(has_hand, gap=GAP, min_len=MIN_LEN, margin=MARGIN):
    """Returns (start, end) of the span where a hand is present, or None."""
    mask = np.asarray(has_hand, dtype=bool)
    if not mask.any():
        return None

    idx = np.flatnonzero(mask)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap:
            mask[a:b] = True

    idx = np.flatnonzero(mask)
    start, end = int(idx[0]), int(idx[-1]) + 1

    if end - start < min_len:
        return None

    return max(0, start - margin), min(len(mask), end + margin)


def safe_name(s):
    """Safe folder name. WARNING: not injective, never use it to rebuild the
    label. Use the manifest instead."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def npy_rel_path(gloss, video_name):
    """Output path, relative to OUT_DIR."""
    return Path(safe_name(gloss)) / (Path(video_name).stem + ".npy")


def make_detectors(model_dir, running_mode=vision.RunningMode.VIDEO):
    """Creates the three detectors. Returns (hand, pose, face)."""
    model_dir = Path(model_dir)

    def make(cls, opt_cls, model_name, **kw):
        return cls.create_from_options(
            opt_cls(
                base_options=BaseOptions(model_asset_path=str(model_dir / model_name)),
                running_mode=running_mode,
                **kw,
            )
        )

    hand = make(
        vision.HandLandmarker,
        vision.HandLandmarkerOptions,
        "hand_landmarker.task",
        num_hands=2,
        min_hand_detection_confidence=0.15,
        min_hand_presence_confidence=0.15,
        min_tracking_confidence=0.15,
    )
    pose = make(
        vision.PoseLandmarker,
        vision.PoseLandmarkerOptions,
        "pose_landmarker_full.task",
        num_poses=1,
        output_segmentation_masks=False,
    )
    face = make(
        vision.FaceLandmarker,
        vision.FaceLandmarkerOptions,
        "face_landmarker.task",
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return hand, pose, face


def to_graph(seq):
    """(T, DIM) -> (3, T, 57). Keeps the raw coordinates only.

    The 48 derived distances and the 4 flags are dropped: they have no place
    in a graph formalism.
    """
    t = seq.shape[0]
    n_h, n_p = 2 * N_HAND * 3, N_POSE * 3
    hands = seq[:, :n_h].reshape(t, 2 * N_HAND, 3)
    pose = seq[:, n_h : n_h + n_p].reshape(t, N_POSE, 3)
    joints = np.concatenate([hands, pose], axis=1)  # (T, 57, 3)
    return np.transpose(joints, (2, 0, 1)).astype(np.float32)
