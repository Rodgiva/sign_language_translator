"""ASL Citizen extraction: videos -> coordinate sequences (.npy).

Usage :
    python extract.py train 100     # 100 WORDS drawn at random, all variants
    python extract.py val 100       # reuses the word list already written
    python extract.py test 100

The vocabulary size is required on every split: it selects the output folder.

Label levels :
    gloss : ENVELOPE3   -> training class (visually coherent sign)
    word  : ENVELOPE    -> translator output
Both are stored in features/manifest_<split>.csv.
"""

import csv
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2 as cv
import numpy as np
import mediapipe as mp

from sign_lang import data

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ROOT = Path(r"D:\ASL_Citizen")  # dataset (read-only)
MODEL_DIR = Path(__file__).resolve().parent / "landmarks" # where the .task files are

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"

# Number of WORDS to keep. Every variant and every video of those words is
# extracted. Required argument: it selects the output folder, so a default
# would silently write one split into the folder of another vocabulary.
if len(sys.argv) < 3:
    sys.exit(
        "usage: python video_extracter.py <split> <vocab_size>\n"
        "the vocabulary size is required for every split, since it selects\n"
        "the output folder (features_100, features_1000, ...)"
    )
VOCAB_SIZE = int(sys.argv[2])

# One output folder per vocabulary size: features_100, features_1000, ...
OUT_DIR = MODEL_DIR.parent / f"features_{VOCAB_SIZE}"

MIN_VIDEOS = 13  # minimum number of videos per WORD (all variants together)
SEED = 0  # sampling seed: same value -> same word list

WORKERS = 8  # 8 physical cores


# ----------------------------------------------------------------------------
# Detectors: one set per process
# ----------------------------------------------------------------------------
_DET = {}


def init_worker():
    """Called once per child process."""
    hand, pose, face = data.make_detectors(MODEL_DIR)
    _DET["hand"], _DET["pose"], _DET["face"] = hand, pose, face
    _DET["offset"] = 0


def process_one(task):
    """Processes one video. Returns (status, gloss, frames_read, frames_kept)."""
    gloss, video_name = task
    video_path = ROOT / "videos" / video_name
    out_path = OUT_DIR / data.npy_rel_path(gloss, video_name)

    if out_path.exists():
        return ("skip", gloss, 0, 0)
    if not video_path.exists():
        return ("missing", gloss, 0, 0)

    hand, pose, face = _DET["hand"], _DET["pose"], _DET["face"]

    cap = cv.VideoCapture(str(video_path))
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0

    sequence = []
    has_hand = []
    n = 0
    ts = _DET["offset"]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = _DET["offset"] + int(n * 1000 / fps)

        hand_result = hand.detect_for_video(mp_image, ts)
        pose_result = pose.detect_for_video(mp_image, ts)
        face_result = face.detect_for_video(mp_image, ts)

        sequence.append(data.extract(hand_result, pose_result, face_result))
        has_hand.append(len(hand_result.hand_landmarks) > 0)
        n += 1

    cap.release()
    _DET["offset"] = ts + 1000  # increasing timestamps from one video to the next

    if n == 0:
        return ("empty", gloss, 0, 0)

    span = data.active_span(has_hand)
    if span is None:
        return ("empty", gloss, n, 0)

    s, e = span
    arr = np.stack(sequence)[s:e]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)

    return ("done", gloss, n, int(arr.shape[0]))


# ----------------------------------------------------------------------------
# Word selection
# ----------------------------------------------------------------------------
def choose_words():
    """Returns the set of WORDS to process (or None for all of them)."""
    words_file = OUT_DIR / "words.txt"

    if words_file.exists():
        keep = {w for w in words_file.read_text(encoding="utf-8").split("\n") if w}
        print(f"list read back from {words_file.name}: {len(keep)} words")
        return keep

    if VOCAB_SIZE is None:
        print("no restriction: every word")
        return None

    # Counting ALWAYS on train.csv: the list is identical for every split
    with open(ROOT / "splits" / "train.csv", encoding="utf-8") as fh:
        per_word = Counter()
        variants = defaultdict(set)
        for r in csv.DictReader(fh):
            w = data.base_word(r["Gloss"])
            per_word[w] += 1
            variants[w].add(r["Gloss"])

    eligible = sorted(w for w, c in per_word.items() if c >= MIN_VIDEOS)
    if len(eligible) < VOCAB_SIZE:
        sys.exit(f"only {len(eligible)} words have >= {MIN_VIDEOS} videos")

    # Reproducible random draw: the dataset being balanced, sorting by frequency
    # brings nothing and gives an arbitrary sample.
    keep = set(random.Random(SEED).sample(eligible, VOCAB_SIZE))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    words_file.write_text("\n".join(sorted(keep)), encoding="utf-8")

    n_var = sum(len(variants[w]) for w in keep)
    multi = sorted(
        (w for w in keep if len(variants[w]) > 1), key=lambda w: -len(variants[w])
    )
    print(
        f"{len(keep)} words drawn (seed={SEED}, min {MIN_VIDEOS} videos), "
        f"written to {words_file.name}"
    )
    print(f"-> {n_var} glosses (variants included)")
    if multi:
        apercu = ", ".join(f"{w}({len(variants[w])})" for w in multi[:6])
        print(f"-> {len(multi)} words with several variants: {apercu}")

    return keep


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main():
    csv_path = ROOT / "splits" / f"{SPLIT}.csv"
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    keep = choose_words()
    if keep is not None:
        rows = [r for r in rows if data.base_word(r["Gloss"]) in keep]

    tasks = [(r["Gloss"], r["Video file"]) for r in rows]

    glosses = {g for g, _ in tasks}
    words = {data.base_word(g) for g in glosses}

    print(f"split   : {SPLIT}")
    print(f"words   : {len(words)}")
    print(f"glosses : {len(glosses)}")
    print(f"videos  : {len(tasks)}")
    print(f"workers : {WORKERS}")
    print(f"dim     : {data.DIM}")
    print(f"output  : {OUT_DIR}")

    if tasks:
        per_gloss = sorted(Counter(g for g, _ in tasks).values())
        per_word = sorted(Counter(data.base_word(g) for g, _ in tasks).values())
        print(
            f"videos per gloss : min={per_gloss[0]} "
            f"median={per_gloss[len(per_gloss) // 2]} max={per_gloss[-1]}"
        )
        print(
            f"videos per word  : min={per_word[0]} "
            f"median={per_word[len(per_word) // 2]} max={per_word[-1]}"
        )

    if not tasks:
        sys.exit("no video to process")

    # Manifest: the only reliable source of labels.
    # Folder names are sanitised (HURDLE/TRIP1 -> HURDLE_TRIP1).
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = OUT_DIR / f"manifest_{SPLIT}.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["npy", "gloss", "word", "participant", "video"])
        for r in rows:
            w.writerow(
                [
                    data.npy_rel_path(r["Gloss"], r["Video file"]).as_posix(),
                    r["Gloss"],
                    data.base_word(r["Gloss"]),
                    r["Participant ID"],
                    r["Video file"],
                ]
            )
    print(f"manifest : {manifest.name} ({len(rows)} rows)")
    print()

    # ---- Processing ---------------------------------------------------------
    counts = {"done": 0, "skip": 0, "missing": 0, "empty": 0}
    total_read = total_kept = 0
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker) as pool:
        for i, result in enumerate(pool.map(process_one, tasks, chunksize=4), 1):
            status, gloss, n_read, n_kept = result
            counts[status] += 1
            total_read += n_read
            total_kept += n_kept

            if i % 50 == 0 or i == len(tasks):
                elapsed = time.perf_counter() - t0
                rate = elapsed / i
                left = (len(tasks) - i) * rate / 60
                print(
                    f"[{i}/{len(tasks)}] {gloss:<16} "
                    f"{rate:.2f} s/video   ~{left:.0f} min left"
                )

    elapsed = time.perf_counter() - t0

    print()
    print(f"saved        : {counts['done']}")
    print(f"already done : {counts['skip']}")
    print(f"missing      : {counts['missing']}")
    print(f"no sign      : {counts['empty']}")
    if total_read:
        pct = 100 * total_kept / total_read
        print(f"frames       : {total_read} read -> {total_kept} kept ({pct:.0f} %)")
    print(f"duration     : {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
