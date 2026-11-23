"""Real-time sign recognition from the webcam, then sentence generation.

Usage :
    python sign_reader.py                 # models/model_gru_1000.pt
    python sign_reader.py lstm            # models/model_lstm_1000.pt
    python sign_reader.py stgcn 100       # models/model_stgcn_100.pt

The architecture is read from the checkpoint, not from the argument: the latter
only serves to build the file name.

End of sentence: wrists below the hips for REST_FRAMES frames, or SPACE.
Requires : models/model_<arch>_<vocab>.pt, landmarks/*.task,
           and a running Ollama server.
"""

import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks.python import vision

from sign_lang.data import (
    GAP,
    MARGIN,
    MIN_LEN,
    extract,
    make_detectors,
    resample,
    to_graph,
)
from sign_lang.llm import to_sentence
from sign_lang.model import SignGRU, STGCN

ROOT = Path(__file__).resolve().parent

CELL = sys.argv[1].lower() if len(sys.argv) > 1 else "gru"  # "gru", "lstm", "stgcn"
VOCAB = sys.argv[2] if len(sys.argv) > 2 else "100"  # "100" or "1000"
CKPT = ROOT / "models" / f"model_{CELL}_{VOCAB}.pt"

COOLDOWN = 15  # frames to wait after a prediction
CONF_MIN = 0.30  # below this, the sign is not kept
REST_FRAMES = 36  # ~1.2 s with wrists below the hips -> end of sentence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PL = vision.PoseLandmark
L_WR, R_WR = int(PL.LEFT_WRIST), int(PL.RIGHT_WRIST)
L_HIP, R_HIP = int(PL.LEFT_HIP), int(PL.RIGHT_HIP)

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
classes = ckpt["classes"]
mean, std = ckpt["mean"], ckpt["std"]
SEQ_LEN = ckpt["seq_len"]

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

print(f"model    : {len(classes)} classes, label={ckpt['label_col']}")
print(f"device   : {DEVICE}")
print("SPACE = send the sentence   q = quit")


@torch.no_grad()
def predict(buffer):
    seq = resample(np.stack(buffer), SEQ_LEN)
    if IS_STGCN:
        # (T, 223) -> (3, T, 57), normalised per (channel, joint)
        seq = (to_graph(seq) - mean) / std
    else:
        seq = (seq - mean) / std
    x = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(x), dim=1)[0]
    top = probs.topk(min(3, len(classes)))
    return [(classes[i], float(p)) for p, i in zip(top.values, top.indices)]


# ----------------------------------------------------------------------------
# LLM stage, on a separate thread
# ----------------------------------------------------------------------------
# The call takes 1 to 3 s, and far more the first time since Ollama loads the
# model. Doing it inside the loop would freeze the image: keep it aside.
lock = threading.Lock()
english = ""  # last sentence produced
pending = False  # a call is in flight


def llm_worker(signs):
    global english, pending
    try:
        text = to_sentence(signs)
    except Exception as exc:  # Ollama down, model missing, timeout
        text = f"[llm unavailable: {exc}]"
    with lock:
        english = text
        pending = False
    print(f"-> {text}")


def send(signs):
    """Starts the sentence generation. Does nothing if a call is in flight."""
    global pending
    with lock:
        if pending:
            return False
        pending = True
    threading.Thread(target=llm_worker, args=(signs,), daemon=True).start()
    return True


# ----------------------------------------------------------------------------
# Real-time loop
# ----------------------------------------------------------------------------
hand_lm, pose_lm, face_lm = make_detectors(ROOT / "landmarks")

cam = cv.VideoCapture(0)

pre_roll = deque(maxlen=MARGIN)  # frames just before the sign starts
buffer = []  # segment in progress
recording = False
no_hand = 0
cooldown = 0
rest = 0  # consecutive frames in rest position
last = []  # top-3 of the last sign
phrase = []  # one entry per kept sign, each one [(gloss, conf), ...]

while cam.isOpened():
    ok, frame = cam.read()
    if not ok:
        break

    # DO NOT mirror the image: MediaPipe would infer the wrong hand
    # and the features would no longer match the training data.
    h = frame.shape[0]
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts = int(time.perf_counter() * 1000)

    hand_result = hand_lm.detect_for_video(mp_image, ts)
    pose_result = pose_lm.detect_for_video(mp_image, ts)
    face_result = face_lm.detect_for_video(mp_image, ts)

    feats = extract(hand_result, pose_result, face_result)
    has_hand = len(hand_result.hand_landmarks) > 0

    # ---- Rest position ------------------------------------------------------
    # Ask the POSE rather than the hand detection: the pose returns the wrists
    # reliably, which is not the case for the hand detector.
    at_rest = False
    if pose_result.pose_landmarks:
        p = pose_result.pose_landmarks[0]
        hip_y = (p[L_HIP].y + p[R_HIP].y) / 2.0
        at_rest = p[L_WR].y > hip_y and p[R_WR].y > hip_y  # y grows downwards
    rest = rest + 1 if at_rest else 0

    # ---- Segmentation -------------------------------------------------------
    if cooldown > 0:
        cooldown -= 1
        pre_roll.append(feats)

    elif not recording:
        if has_hand:
            recording = True
            buffer = list(pre_roll) + [feats]  # pick up the leading margin
            no_hand = 0
        else:
            pre_roll.append(feats)

    else:
        buffer.append(feats)
        no_hand = 0 if has_hand else no_hand + 1

        if no_hand >= GAP:
            # Drop the hand-free tail, keeping MARGIN frames
            keep = len(buffer) - GAP + MARGIN
            segment = buffer[: max(keep, 0)]

            if len(segment) >= MIN_LEN:
                last = predict(segment)
                label, conf = last[0]
                if conf >= CONF_MIN:
                    phrase.append(last)  # the whole top-3, for the LLM
                    print(f"{label:<16} {conf * 100:5.1f}%   ({len(segment)} frames)")
                else:
                    print(
                        f"uncertain        {conf * 100:5.1f}%   "
                        f"({len(segment)} frames)"
                    )

            recording = False
            buffer = []
            pre_roll.clear()
            cooldown = COOLDOWN

    # ---- End of sentence on a long pause ------------------------------------
    if phrase and rest >= REST_FRAMES:
        if send(phrase):
            phrase = []
            rest = 0

    # ---- Display ------------------------------------------------------------
    for hand_landmarks in hand_result.hand_landmarks:
        vision.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style(),
        )

    if recording:
        state, colour = f"REC {len(buffer)}", (0, 0, 255)
    elif cooldown > 0:
        state, colour = "pause", (0, 165, 255)
    elif phrase and rest:
        state, colour = f"rest {rest}/{REST_FRAMES}", (0, 255, 255)
    else:
        state, colour = "ready", (180, 180, 180)
    cv.putText(frame, state, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

    for k, (label, conf) in enumerate(last):
        c = (0, 255, 0) if k == 0 and conf >= CONF_MIN else (160, 160, 160)
        cv.putText(
            frame,
            f"{label}  {conf * 100:.0f}%",
            (10, 70 + k * 28),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            c,
            2,
        )

    if phrase:
        cv.putText(
            frame,
            " ".join(cands[0][0] for cands in phrase[-6:]),
            (10, h - 50),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

    with lock:
        shown, busy = english, pending
    if busy:
        cv.putText(
            frame, "...", (10, h - 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
        )
    elif shown:
        cv.putText(
            frame, shown, (10, h - 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

    cv.imshow("Signs", frame)

    key = cv.waitKey(5) & 0xFF
    if key == ord(" ") and phrase:  # manual send
        if send(phrase):
            phrase = []
            rest = 0
    elif key == ord("\b"):
        phrase = []
        rest = 0
    elif key == ord("q"):
        break

hand_lm.close()
pose_lm.close()
face_lm.close()
cam.release()
cv.destroyAllWindows()
