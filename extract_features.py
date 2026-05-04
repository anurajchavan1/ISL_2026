import os
import cv2
import numpy as np
import mediapipe as mp

# ==========================================
# 1. CONFIGURATION
# ==========================================
RAW_VIDEOS_DIR = os.path.join(os.getcwd(), 'raw_videos')
EXTRACTED_DIR = os.path.join(os.getcwd(), 'extracted_data')
SEQUENCE_LENGTH = 30 

SELECTED_FACE_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 
    46, 53, 52, 65, 55, 285, 295, 282, 283, 276, 
    33, 133, 362, 263, 
    1, 2, 98, 327, 152, 148 
]

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    
    if results.face_landmarks:
        face = np.array([[results.face_landmarks.landmark[i].x, 
                          results.face_landmarks.landmark[i].y, 
                          results.face_landmarks.landmark[i].z] for i in SELECTED_FACE_INDICES]).flatten()
    else:
        face = np.zeros(40*3)
        
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # --- PHASE 2: HAND WEIGHTING MULTIPLIER ---
    # This forces the AI to pay twice as much attention to the fingers
    lh = lh * 2.0 
    rh = rh * 2.0 
    
    raw_features = np.concatenate([pose, face, lh, rh])
    
    # NORMALIZATION (The Nose-Anchor Fix)
    if results.pose_landmarks:
        nose_x = results.pose_landmarks.landmark[0].x
        nose_y = results.pose_landmarks.landmark[0].y
        nose_z = results.pose_landmarks.landmark[0].z
        
        raw_features[0::3] -= nose_x 
        raw_features[1::3] -= nose_y 
        raw_features[2::3] -= nose_z 
        
    return raw_features

# ==========================================
# 2. EXTRACTION LOOP
# ==========================================
def process_videos():
    if not os.path.exists(EXTRACTED_DIR):
        os.makedirs(EXTRACTED_DIR)

    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for category_folder in os.listdir(RAW_VIDEOS_DIR):
            cat_path = os.path.join(RAW_VIDEOS_DIR, category_folder)
            if not os.path.isdir(cat_path): continue
            
            for sign_word in os.listdir(cat_path):
                word_path = os.path.join(cat_path, sign_word)
                if not os.path.isdir(word_path): continue
                
                target_word_dir = os.path.join(EXTRACTED_DIR, sign_word)
                os.makedirs(target_word_dir, exist_ok=True)
                
                videos_found_in_folder = False
                
                for video_file in os.listdir(word_path):
                    if not video_file.lower().endswith(valid_extensions): 
                        continue
                        
                    videos_found_in_folder = True
                    video_path = os.path.join(word_path, video_file)
                    save_path = os.path.join(target_word_dir, video_file.rsplit('.', 1)[0] + '.npy')
                    
                    if os.path.exists(save_path):
                        continue
                    
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    if total_frames == 0: 
                        print(f"  [ERROR] OpenCV cannot read frames from: {video_file}")
                        cap.release()
                        continue
                        
                    frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
                    sequence_data = []
                    current_frame = 0
                    
                    while cap.isOpened() and len(sequence_data) < SEQUENCE_LENGTH:
                        ret, frame = cap.read()
                        if not ret: break
                            
                        if current_frame in frame_indices:
                            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image.flags.writeable = False
                            results = holistic.process(image)
                            
                            keypoints = extract_keypoints(results)
                            
                            if len(sequence_data) == 0:
                                velocity = np.zeros(345)
                            else:
                                velocity = keypoints - sequence_data[-1][:345]
                                
                            combined_features = np.concatenate([keypoints, velocity])
                            sequence_data.append(combined_features)
                            
                        current_frame += 1
                        
                    cap.release()
                    
                    while len(sequence_data) < SEQUENCE_LENGTH:
                        sequence_data.append(np.zeros(690))
                        
                    np.save(save_path, np.array(sequence_data))
                    print(f"SUCCESS -> Extracted: {sign_word} / {video_file}")

                if not videos_found_in_folder:
                    print(f"  [WARNING] No video files found inside folder: {word_path}")

if __name__ == "__main__":
    print("Starting Extraction with Enhanced Hand Tracking...")
    process_videos()