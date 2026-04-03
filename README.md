# 🚗 Smart Traffic AI System (YOLOv8 + DeepSORT)

A complete **Computer Vision pipeline** for real-time traffic analysis, built using **YOLOv8** and **DeepSORT**.
This system performs **vehicle detection, tracking, counting, direction analysis, and speed estimation**.

---

## 📌 Features

* 🚘 Vehicle Detection using YOLOv8
* 🧠 Multi-Object Tracking using DeepSORT
* 🔢 Accurate Vehicle Counting (Line Crossing)
* 🔄 Direction Detection (UP / DOWN)
* ⚡ Speed Estimation (per vehicle)
* 📊 Real-time GUI + Terminal Logs

---

## 🧠 System Architecture

```
Video Input
   ↓
YOLOv8 (Detection)
   ↓
DeepSORT (Tracking + IDs)
   ↓
Line Crossing Logic
   ↓
Direction Detection
   ↓
Vehicle Counting
   ↓
Speed Estimation
```

---

## 📁 Project Structure

```
traffic-ai/
│
├── main.py              # Final main file
├── yolov8n.pt           # YOLO model
├── video.mp4            # Input video
├── venv/                # Virtual environment
```

---

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository

```
git clone <your-repo-link>
cd traffic-ai
```

---

### 2️⃣ Create Virtual Environment

```
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```
pip install ultralytics opencv-python deep-sort-realtime numpy
```

---

### 4️⃣ Add Required Files

Make sure you have:

```
yolov8n.pt
fixed.mp4
```

---

## ▶️ Run the Project

```
python main.py
```

---

## 🧪 Output

### 🖥 GUI:

* Vehicle bounding boxes
* Unique tracking IDs
* Speed (km/h) per vehicle
* Counting zone
* UP / DOWN / TOTAL counters

### 🧾 Terminal Logs:

```
DOWN ID 7 Speed: 54
DOWN ID 12 Speed: 48
```

---

## ⚙️ Configuration

### 📍 Counting Line

```
LINE_Y = 250
OFFSET = 25
```

### ⚡ Speed Calibration

```
PIXEL_TO_METER = 0.02
```

> ⚠️ Adjust this value to match realistic speeds.

---

## ⚠️ Limitations

* Speed estimation is **approximate** (no real-world calibration)
* Accuracy depends on:

  * Camera angle
  * Video quality
  * Detection confidence

---

## 📊 Performance

| Component        | Status              |
| ---------------- | ------------------- |
| Detection        | ✅ High              |
| Tracking         | ✅ Stable (DeepSORT) |
| Counting         | ✅ Accurate          |
| Direction        | ✅ Correct           |
| Speed Estimation | ⚠️ Approximate      |

---

## 🚀 Future Improvements

* 🚨 Speed Violation Detection
* 📊 Dashboard (Streamlit / React)
* 🪖 Helmet Detection
* ⚡ Real-world calibration
* 📈 Data export (CSV / Database)

---

## 🎯 Use Cases

* Smart Traffic Monitoring
* Vehicle Flow Analysis
* Road Safety Systems
* AI Surveillance

---

## ⭐ Acknowledgements

* Ultralytics YOLOv8
* DeepSORT Tracking
* OpenCV

---

## 📌 Final Note

This project demonstrates a **complete real-world computer vision pipeline**, combining detection, tracking, analytics, and estimation into a single system.
