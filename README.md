# 👁️ **REAL-TIME FACE DETECTION APP** 🚀

### *Precision Detection Meets High-Performance Computer Vision*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FF00?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-0078D4?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)

---

## 🌟 **Overview**

This project is a comprehensive **Real-Time Face Detection and Recognition System** that implements multiple state-of-the-art computer vision models. From lightweight DNNs to the highly accurate YOLOv8 and MTCNN, this app provides a versatile environment for face detection, image capturing, and automated attendance logging.

---

## ✨ **Core Features**

- ⚡ **Multi-Model Support**: Switch between DNN, MTCNN, MediaPipe, and YOLOv8 seamlessly.
- 🕒 **Real-Time Processing**: Ultra-low latency face detection at high FPS.
- 📸 **Image Capture**: Automated face collection for recognition training.
- 📑 **Smart Attendance**: Automatic logging of detected faces into `attendance.csv`.
- 📊 **Accuracy Metrics**: Visual confidence scores for every detected face.

---

## 🧠 **Supported Models**

| Model | Script | Strengths |
| :--- | :--- | :--- |
| **YOLOv8** | `YOLOv8_model.py` | State-of-the-art speed & accuracy |
| **MTCNN** | `MTCNN_model.py` | Accurate face & feature alignment |
| **MediaPipe** | `MediaPipe_model.py` | Google's lightning-fast lightweight model |
| **DNN (OpenCV)** | `DNN_model.py` | Reliable & built-in OpenCV performance |

---

## 🛠️ **Installation & Usage**

### 1. Clone the Repository
```bash
git clone https://github.com/Pratham216/Real-Time_Face_Detection-App.git
cd Real-Time_Face_Detection-App
```

### 2. Install Dependencies
```bash
pip install opencv-python ultralytics mediapipe mtcnn pandas
```

### 3. Run the App
To start the real-time detection:
```bash
python real-time-detection.py
```

To capture images for your database:
```bash
python capture-image.py
```

---

## 📂 **Project Structure**

```text
├── faces/               # Local database of face images
├── attendance.csv       # Automated attendance log
├── YOLOv8_model.py      # YOLOv8 implementation
├── MTCNN_model.py       # MTCNN implementation
├── MediaPipe_model.py   # MediaPipe implementation
├── DNN_model.py         # OpenCV DNN implementation
├── capture-image.py     # Tool for face data collection
└── real-time-detection.py # Main entry point
```

---

## 🤝 **Contributing**

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 **License**

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
  <h3>Built with ❤️ by [Pratham](https://github.com/Pratham216)</h3>
  <p>If you like this project, please give it a ⭐️!</p>
</div>
