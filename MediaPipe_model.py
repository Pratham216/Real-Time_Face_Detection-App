import cv2
import numpy as np
import face_recognition
import os
import time
from threading import Thread
import mediapipe as mp

"""
Model: MediaPipe Face Detection
Accuracy: ~92-95%
F1 Score: ~0.93
Pros: Fast, good accuracy, works well on mobile
Cons: Can miss very small faces
"""

# ---------- Prompt for Name ---------- #
name_input = input("Enter your name (for capturing face with 'c'): ").strip()

# ---------- Prepare Faces Directory ---------- #
path = 'faces'
os.makedirs(path, exist_ok=True)

# ---------- Load and Encode Training Images ---------- #
def load_images_and_encodings():
    images = []
    classNames = []
    
    for img in os.listdir(path):
        image = cv2.imread(f'{path}/{img}')
        images.append(image)
        classNames.append(os.path.splitext(img)[0])
    
    encodes = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img_rgb)
        if enc:
            encodes.append(enc[0])
    return classNames, encodes

classNames, knownEncodes = load_images_and_encodings()
print("Encoding complete")

# ---------- Initialize MediaPipe Face Detection ---------- #
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,       # 0 for short-range, 1 for full-range
    min_detection_confidence=0.5
)

# ---------- Threaded Webcam ---------- #
class WebcamStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if not grabbed:
                self.stopped = True
                break
            self.grabbed = grabbed
            self.frame = frame

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# ---------- Main Loop ---------- #
scale = 0.5  # Scaling factor for face detection
frame_interval = 2  # Process every 2nd frame
frame_count = 0
stream = WebcamStream()
prev_time = time.time()
last_face_locations = []
last_names = []

while True:
    success, frame = stream.read()
    if not success:
        break

    frame_count += 1
    display_frame = frame.copy()

    # Press 'c' to capture and save face image
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        filename = os.path.join(path, f"{name_input}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Image saved: {filename}")

        # Reload all encodings
        classNames, knownEncodes = load_images_and_encodings()
        print("Encodings updated")

    # Face detection and recognition every 'frame_interval' frames
    if frame_count % frame_interval == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe Face Detection
        results = face_detection.process(rgb_frame)
        
        face_encodings = []
        face_locations = []
        names = []
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw = small_frame.shape[:2]
                
                # Get bounding box coordinates
                x1 = int(bboxC.xmin * iw)
                y1 = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                # Ensure coordinates are within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw - 1, x1 + w), min(ih - 1, y1 + h)
                
                # Extract face ROI
                roi = small_frame[y1:y2, x1:x2]
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                enc = face_recognition.face_encodings(roi_rgb)
                
                if enc:
                    face_encodings.append(enc[0])
                    # Scale back up face locations to original frame size
                    face_locations.append((int(y1 / scale), int(x2 / scale), int(y2 / scale), int(x1 / scale)))

        # Face recognition
        for encodeFace in face_encodings:
            matches = face_recognition.compare_faces(knownEncodes, encodeFace, tolerance=0.6)
            faceDis = face_recognition.face_distance(knownEncodes, encodeFace)
            matchIndex = np.argmin(faceDis) if faceDis.size > 0 else -1
            name = classNames[matchIndex].upper() if matchIndex != -1 and matches[matchIndex] else 'Unknown'
            names.append(name)

        last_face_locations = face_locations
        last_names = names

    # Display results
    for (y1, x2, y2, x1), name in zip(last_face_locations, last_names):
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(display_frame, (x1, y2 - 20), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(display_frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    # Show FPS and model info
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(display_frame, f"FPS: {int(fps)} | Model: MediaPipe (Acc: 92-95%)", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Face Recognition (MediaPipe)", display_frame)

    if key == ord('q'):
        break

# Cleanup
face_detection.close()
stream.stop()
cv2.destroyAllWindows()