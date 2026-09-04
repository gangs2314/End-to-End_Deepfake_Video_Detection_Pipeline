import cv2
import os
import sys
import mediapipe as mp

if len(sys.argv) < 2:
    print("ERROR: Please provide a video path.")
    sys.exit(1)

video_path = sys.argv[1]
output_dir = "temp_frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"ERROR: Cannot open video file {video_path}")
    sys.exit(1)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
# model_selection=1 is better for faces further away/smaller
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

frame_count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract 1 frame per second (assuming ~30fps)
    if frame_count % 30 == 0:
        # MediaPipe requires RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            # We just take the first detected face to match your original logic
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            
            # Convert relative coordinates to absolute pixel values
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            # Add a small margin (padding) around the face
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            
            x_start = max(0, x - margin_x)
            y_start = max(0, y - margin_y)
            x_end = min(iw, x + w + margin_x)
            y_end = min(ih, y + h + margin_y)

            # Crop the face
            face = frame[y_start:y_end, x_start:x_end]

            if face.size > 0:
                face_path = os.path.join(output_dir, f"frame_{saved}.jpg")
                cv2.imwrite(face_path, face)
                saved += 1

    frame_count += 1

cap.release()
face_detection.close()
# Do not print anything here unless you want it mixed into standard output