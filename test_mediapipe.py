import time
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

FACE_ANCHORS = [10, 168, 1, 13, 14, 152, 61, 291, 33, 263, 234, 454]

PL = vision.PoseLandmark

POSE_KEEP = {
    int(PL.LEFT_SHOULDER),
    int(PL.RIGHT_SHOULDER),
    int(PL.LEFT_ELBOW),
    int(PL.RIGHT_ELBOW),
    int(PL.LEFT_WRIST),
    int(PL.RIGHT_WRIST),
    int(PL.LEFT_HIP),
    int(PL.RIGHT_HIP),
}

POSE_CONNECTIONS = [
    c
    for c in vision.PoseLandmarksConnections.POSE_LANDMARKS
    if c.start in POSE_KEEP and c.end in POSE_KEEP
]

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="landmarks/hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="landmarks/pose_landmarker_full.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="landmarks/face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

cam = cv.VideoCapture(0)

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        print("Camera Frame not available")
        break

    height, width = frame.shape[:2]
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.perf_counter() * 1000)

    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
    face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

    for pose_landmarks in pose_result.pose_landmarks:
        vision.drawing_utils.draw_landmarks(
            frame,
            pose_landmarks,
            POSE_CONNECTIONS,
            landmark_drawing_spec=None,
        )

        for i in POSE_KEEP:
            lm = pose_landmarks[i]
            if lm.visibility is not None and lm.visibility < 0.5:
                continue
            cv.circle(
                frame, (int(lm.x * width), int(lm.y * height)), 4, (255, 0, 0), -1
            )

    for hand_landmarks in hand_result.hand_landmarks:
        vision.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style(),
        )

    if face_result.face_landmarks:
        face = face_result.face_landmarks[0]
        for anchor_index in FACE_ANCHORS:
            landmark = face[anchor_index]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv.circle(frame, (x, y), 3, (0, 255, 0), -1)
            cv.putText(
                frame,
                str(anchor_index),
                (x + 5, y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
            )

    cv.imshow("Show Video", frame)

    if cv.waitKey(20) & 0xFF == ord("q"):
        break

hand_landmarker.close()
pose_landmarker.close()
face_landmarker.close()
cam.release()
cv.destroyAllWindows()
