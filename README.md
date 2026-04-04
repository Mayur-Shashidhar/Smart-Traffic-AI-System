# 🚗 Smart Traffic AI System (YOLOv8 + DeepSORT)

A complete **Computer Vision pipeline** for real-time traffic analysis, built using **YOLOv8** and **DeepSORT**.
This system performs **vehicle detection, tracking, counting, direction analysis, speed estimation, and model evaluation**.

---

## 📌 Features

* 🚘 Vehicle Detection using YOLOv8
* 🧠 Multi-Object Tracking using DeepSORT
* 🔢 Accurate Vehicle Counting (Line Crossing)
* 🔄 Direction Detection (UP / DOWN)
* ⚡ Speed Estimation (per vehicle)
* 📊 Real-time GUI + Terminal Logs
* 📈 **Model Accuracy Comparison (YOLOv8n vs YOLOv8x)**

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
Speed Estimation / Accuracy Evaluation
```

---

## 📁 Project Structure

```
traffic-ai/
│
├── main.py              # Base system (detection + tracking + speed)
├── accuracy.py          # Model comparison + accuracy evaluation
├── video.mp4            # Input video
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

## 📦 Model Weights

Model weights are **NOT included in this repository** due to GitHub size limits.

They will be **automatically downloaded** when you run the code using
Ultralytics.

### Models Used:

* `yolov8n.pt` → Fast model (Model A)
* `yolov8x.pt` → Accurate model (Model B)

No manual download required ✅

---

## ▶️ Run the Project

### 🔹 Base System

```
python main.py
```

👉 Includes:

* Detection
* Tracking
* Counting
* Speed estimation

---

### 🔹 Accuracy System

```
python accuracy.py
```

👉 Includes:

* Dual model comparison
* Unique vehicle counting
* Accuracy calculation

---

## 🧪 Output

### 🖥 GUI:

* Vehicle bounding boxes
* Unique tracking IDs
* Counting zone
* Model A vs Model B counts

### 🧾 Terminal Logs:

```
--- Frame 32 ---
A COUNT → ID 7
B COUNT → ID 9
Running Total → A: 12 | B: 14
```

### 📊 Final Report:

```
Model A: 27
Model B: 26
Difference: 1
Accuracy: 96.29%
```

---

## ⚙️ Configuration

### 📍 Counting Line

```
LINE_Y = 150
OFFSET = 25
```

---

### ⚡ Speed Calibration

```
PIXEL_TO_METER = 0.02
```

> ⚠️ Adjust for realistic speeds based on camera setup.

---

## ⚠️ Limitations

* Speed estimation is **approximate** (no real-world calibration)
* Accuracy is **relative (model vs model)**, not absolute ground truth
* Performance depends on:

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
| Accuracy Eval    | ✅ Relative          |

---

## 🚀 Future Improvements

* 🚨 Speed Violation Detection
* 📊 Dashboard (Streamlit / React)
* 🪖 Helmet Detection
* ⚡ Real-world calibration
* 📈 CSV / Database logging
* 📉 Precision / Recall evaluation

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

This project demonstrates a **complete real-world computer vision pipeline**, combining detection, tracking, analytics, and evaluation into a single system.

It evolves from a basic detection system (`main.py`) to a **self-evaluating AI pipeline (`accuracy.py`)**, showcasing practical ML engineering workflows.
