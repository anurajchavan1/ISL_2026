"""
extract_features.py

Extracts 720-dim feature vectors from each video:
  345 nose-normalised coordinates (pose + face + hands)
+ 345 frame-to-frame velocity
+  15 left hand joint angles
+  15 right hand joint angles
= 720 total

Run AFTER dataset_cleaner.py and BEFORE train_bilstm.py / train_tcn.py.
"""

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
FEATURES_PER_FRAME = 720  # 345 coords + 345 velocity + 15 lh + 15 rh angles

SELECTED_FACE_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0,
    267, 269, 270, 409, 46, 53, 52, 65, 55, 285, 295, 282, 283, 276,
    33, 133, 362, 263, 1, 2, 98, 327, 152, 148
]

VALID_EXT   = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
mp_holistic = mp.solutions.holistic


# ==========================================
# JOINT ANGLES
# ==========================================
def compute_joint_angles(hand_flat):
    """15 joint angles from 21 hand landmarks — finger curl + spread."""
    if np.all(hand_flat == 0):
        return np.zeros(15)
    pts = hand_flat.reshape(21, 3)
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
                      else float(np.arccos(np.clip(
                          np.dot(v1,v2)/(n1*n2), -1., 1.))))
    return np.array(angles)


# ==========================================
# KEYPOINT EXTRACTION
# ==========================================
def extract_keypoints(results):
    """
    345-dim nose-anchor normalised coordinate vector.
    pose(99) + face(120) + lh(63) + rh(63) = 345
    """
    pose = (np.array([[r.x,r.y,r.z]
                      for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(99))
    face = (np.array([[results.face_landmarks.landmark[i].x,
                       results.face_landmarks.landmark[i].y,
                       results.face_landmarks.landmark[i].z]
                      for i in SELECTED_FACE_INDICES]).flatten()
            if results.face_landmarks else np.zeros(120))
    lh = (np.array([[r.x,r.y,r.z]
                    for r in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(63))
    rh = (np.array([[r.x,r.y,r.z]
                    for r in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(63))

    raw = np.concatenate([pose, face, lh, rh])  # 345

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
    saved = skipped = errors = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:

        for cat in sorted(os.listdir(RAW_VIDEOS_DIR)):
            cat_path = os.path.join(RAW_VIDEOS_DIR, cat)
            if not os.path.isdir(cat_path):
                continue

            for word in sorted(os.listdir(cat_path)):
                word_path = os.path.join(cat_path, word)
                if not os.path.isdir(word_path):
                    continue

                target_dir = os.path.join(EXTRACTED_DIR, word)
                os.makedirs(target_dir, exist_ok=True)

                for video_file in sorted(os.listdir(word_path)):
                    if not video_file.lower().endswith(VALID_EXT):
                        continue

                    save_path = os.path.join(
                        target_dir,
                        video_file.rsplit('.',1)[0] + '.npy'
                    )
                    if os.path.exists(save_path):
                        skipped += 1
                        continue

                    video_path = os.path.join(word_path, video_file)
                    cap        = cv2.VideoCapture(video_path)
                    n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    if n_frames == 0:
                        print(f"  [ERROR] {word}/{video_file}")
                        cap.release()
                        errors += 1
                        continue

                    frame_indices  = set(np.unique(
                        np.linspace(0, n_frames-1, SEQUENCE_LENGTH, dtype=int)
                    ))
                    seq            = []
                    prev_kp        = None
                    cur_frame      = 0

                    while cap.isOpened() and len(seq) < SEQUENCE_LENGTH:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if cur_frame in frame_indices:
                            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img.flags.writeable = False
                            res = holistic.process(img)

                            kp  = extract_keypoints(res)
                            vel = (np.zeros(345) if prev_kp is None
                                   else kp - prev_kp)
                            prev_kp = kp.copy()

                            lh_flat = (
                                np.array([[r.x,r.y,r.z]
                                          for r in res.left_hand_landmarks.landmark]).flatten()
                                if res.left_hand_landmarks else np.zeros(63))
                            rh_flat = (
                                np.array([[r.x,r.y,r.z]
                                          for r in res.right_hand_landmarks.landmark]).flatten()
                                if res.right_hand_landmarks else np.zeros(63))

                            combined = np.concatenate([
                                kp, vel,
                                compute_joint_angles(lh_flat),
                                compute_joint_angles(rh_flat)
                            ])  # 720
                            seq.append(combined)

                        cur_frame += 1

                    cap.release()

                    while len(seq) < SEQUENCE_LENGTH:
                        seq.append(np.zeros(FEATURES_PER_FRAME))

                    np.save(save_path, np.array(seq, dtype=np.float32))
                    saved += 1
                    print(f"  OK  {word} / {video_file}")

    print(f"\nDone.  Saved:{saved}  Skipped:{skipped}  Errors:{errors}")


if __name__ == "__main__":
    print(f"Extracting features | seq={SEQUENCE_LENGTH} | dims={FEATURES_PER_FRAME}")
    process_videos()