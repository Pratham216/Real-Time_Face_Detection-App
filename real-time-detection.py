import cv2
import numpy as np
import face_recognition
import os
import time
from threading import Thread

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

# ---------- Haar Cascade ---------- #
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------- Threaded Webcam ---------- #
class WebcamStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        # Initialize attributes before starting the thread
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        # Start the thread after initializing attributes
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
scale = 0.3
frame_interval = 3
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

    if frame_count % frame_interval == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        face_encodings = []
        face_locations = []
        names = []

        for (x, y, w, h) in faces:
            x1, y1, x2, y2 = x, y, x + w, y + h
            roi = small_frame[y1:y2, x1:x2]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            enc = face_recognition.face_encodings(roi_rgb)
            if enc:
                face_encodings.append(enc[0])
                face_locations.append((int(y1 / scale), int(x2 / scale), int(y2 / scale), int(x1 / scale)))

        for encodeFace in face_encodings:
            matches = face_recognition.compare_faces(knownEncodes, encodeFace, tolerance=0.6)
            faceDis = face_recognition.face_distance(knownEncodes, encodeFace)
            matchIndex = np.argmin(faceDis) if faceDis.size > 0 else -1
            name = classNames[matchIndex].upper() if matchIndex != -1 and matches[matchIndex] else 'Unknown'
            names.append(name)

        last_face_locations = face_locations
        last_names = names

    for (y1, x2, y2, x1), name in zip(last_face_locations, last_names):
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(display_frame, (x1, y2 - 20), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(display_frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    # Show FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow("Face Recognition", display_frame)

    if key == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()