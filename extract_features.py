import os
import cv2
import numpy as np
import mediapipe as mp

# ==========================================
# CONFIGURATION
# ==========================================
RAW_VIDEOS_DIR     = os.path.join(os.getcwd(), 'raw_videos')
EXTRACTED_DIR      = os.path.join(os.getcwd(), 'extracted_data')
SEQUENCE_LENGTH    = 30
FEATURES_PER_FRAME = 720  # 345 coords + 345 velocity + 15 lh_angles + 15 rh_angles

SELECTED_FACE_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    46, 53, 52, 65, 55, 285, 295, 282, 283, 276,
    33, 133, 362, 263,
    1, 2, 98, 327, 152, 148
]

mp_holistic = mp.solutions.holistic


# ==========================================
# JOINT ANGLE COMPUTATION
# ==========================================
def compute_joint_angles(hand_landmarks_flat):
    if np.all(hand_landmarks_flat == 0):
        return np.zeros(15)
    pts = hand_landmarks_flat.reshape(21, 3)
    triplets = [
        (0,1,2),(1,2,3),
        (0,5,6),(5,6,7),(6,7,8),
        (0,9,10),(9,10,11),(10,11,12),
        (0,13,14),(13,14,15),(14,15,16),
        (0,17,18),(17,18,19),(18,19,20),
        (5,9,13),
    ]
    angles = []
    for a, b, c in triplets:
        v1, v2 = pts[a]-pts[b], pts[c]-pts[b]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        angles.append(0.0 if n1 < 1e-6 or n2 < 1e-6
                      else np.arccos(np.clip(np.dot(v1,v2)/(n1*n2), -1., 1.)))
    return np.array(angles)


# ==========================================
# KEYPOINT EXTRACTION
# ==========================================
def extract_keypoints(results):
    """
    Returns 345-dim nose-normalized coordinate vector.
    pose(99) + face(120) + lh(63) + rh(63) = 345
    """
    pose = (np.array([[r.x,r.y,r.z] for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(99))
    face = (np.array([[results.face_landmarks.landmark[i].x,
                       results.face_landmarks.landmark[i].y,
                       results.face_landmarks.landmark[i].z]
                      for i in SELECTED_FACE_INDICES]).flatten()
            if results.face_landmarks else np.zeros(120))
    lh = (np.array([[r.x,r.y,r.z] for r in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(63))
    rh = (np.array([[r.x,r.y,r.z] for r in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(63))

    raw = np.concatenate([pose, face, lh, rh])  # 345-dim

    if results.pose_landmarks:
        nx = results.pose_landmarks.landmark[0].x
        ny = results.pose_landmarks.landmark[0].y
        nz = results.pose_landmarks.landmark[0].z
        raw[0::3] -= nx
        raw[1::3] -= ny
        raw[2::3] -= nz
    return raw


# ==========================================
# EXTRACTION LOOP
# ==========================================
def process_videos():
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    total_saved = total_skipped = 0

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
        for category_folder in sorted(os.listdir(RAW_VIDEOS_DIR)):
            cat_path = os.path.join(RAW_VIDEOS_DIR, category_folder)
            if not os.path.isdir(cat_path): continue

            for sign_word in sorted(os.listdir(cat_path)):
                word_path = os.path.join(cat_path, sign_word)
                if not os.path.isdir(word_path): continue

                target_dir = os.path.join(EXTRACTED_DIR, sign_word)
                os.makedirs(target_dir, exist_ok=True)
                videos_found = False

                for video_file in sorted(os.listdir(word_path)):
                    if not video_file.lower().endswith(valid_extensions): continue
                    videos_found = True

                    video_path = os.path.join(word_path, video_file)
                    save_path  = os.path.join(target_dir, video_file.rsplit('.',1)[0]+'.npy')

                    if os.path.exists(save_path):
                        total_skipped += 1
                        continue

                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames == 0:
                        print(f"  [ERROR] No frames: {video_file}")
                        cap.release(); continue

                    frame_indices  = set(np.unique(
                        np.linspace(0, total_frames-1, SEQUENCE_LENGTH, dtype=int)))
                    sequence_data  = []
                    prev_keypoints = None
                    current_frame  = 0

                    while cap.isOpened() and len(sequence_data) < SEQUENCE_LENGTH:
                        ret, frame = cap.read()
                        if not ret: break

                        if current_frame in frame_indices:
                            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image.flags.writeable = False
                            results = holistic.process(image)

                            keypoints = extract_keypoints(results)
                            velocity  = (np.zeros(345) if prev_keypoints is None
                                         else keypoints - prev_keypoints)
                            prev_keypoints = keypoints.copy()

                            lh_flat = (np.array([[r.x,r.y,r.z] for r in
                                        results.left_hand_landmarks.landmark]).flatten()
                                       if results.left_hand_landmarks else np.zeros(63))
                            rh_flat = (np.array([[r.x,r.y,r.z] for r in
                                        results.right_hand_landmarks.landmark]).flatten()
                                       if results.right_hand_landmarks else np.zeros(63))

                            combined = np.concatenate([
                                keypoints, velocity,
                                compute_joint_angles(lh_flat),
                                compute_joint_angles(rh_flat)
                            ])  # 720-dim
                            sequence_data.append(combined)

                        current_frame += 1

                    cap.release()

                    while len(sequence_data) < SEQUENCE_LENGTH:
                        sequence_data.append(np.zeros(FEATURES_PER_FRAME))

                    np.save(save_path, np.array(sequence_data))
                    total_saved += 1
                    print(f"  OK -> {sign_word} / {video_file}")

                if not videos_found:
                    print(f"  [WARNING] No videos: {word_path}")

    print(f"\nExtraction complete. Saved:{total_saved}  Skipped:{total_skipped}")


if __name__ == "__main__":
    print(f"Starting extraction | seq_len={SEQUENCE_LENGTH} | features={FEATURES_PER_FRAME}")
    process_videos()