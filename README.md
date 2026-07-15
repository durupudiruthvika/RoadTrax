# 🛣️ RoadTrax: Real-Time Pothole Detection & Geospatial Mapping using YOLOv8

## 🌐 Live Demo

🚀 **Try the Application Here**

**🔗 Live Website:** https://babaameer-roadtrax.hf.space/

> **Note:** This application is deployed on **Hugging Face Spaces**. The first request may take **30–60 seconds** if the Space is waking up.

---

## 📑 Table of Contents

1. [📋 Project Overview](#-overview)
2. [🌐 Live Demo](#-live-demo)
3. [🎯 Key Features](#-features)
4. [⚙️ Tech Stack & Dataset](#technologies-used)
5. [🚀 Installation & Usage](#-usage)
6. [🔄 Project Workflow](#-project-workflow)
7. [📊 Sample Outputs](#-sample-outputs)
8. [📈 Model Performance](#-evaluation-metrics)
9. [🔮 Future Enhancements](#-future-enhancements)
10. [👥 Team Members](#-team-members)
11. [👨‍🏫 Academic Mentor](#-mentor)
12. [🙏 Acknowledgment](#-acknowledgment)

---

# 📋 Overview

**RoadTrax** is an AI-powered road inspection platform that performs **real-time pothole detection, severity classification, and geospatial mapping** using **YOLOv8** and **OpenCV**. The system enables road authorities, municipalities, and citizens to automatically identify road surface damages from images, videos, or live camera feeds while storing their GPS locations for visualization on an interactive map.

RoadTrax minimizes manual inspection efforts by providing intelligent road condition monitoring with high detection accuracy, helping improve road maintenance planning and public safety.

---

# ✨ Features

## 🛣️ Intelligent Pothole Detection

- Real-time pothole detection
- Image upload detection
- Video analysis
- Live webcam detection
- Multiple pothole detection

---

## 📍 GPS & Geospatial Mapping

- Automatic GPS coordinate extraction
- OpenStreetMap integration
- Interactive map visualization
- Location-based pothole reporting

---

## 📊 Severity Classification

- Small
- Medium
- Large

Severity is estimated based on detected pothole dimensions to help prioritize road repairs.

---

## 📈 Analytics Dashboard

- Detection statistics
- Severity distribution
- Location-wise reports
- Detection history
- Interactive visualizations

---

## ☁️ Cloud Integration

- Database storage
- Cloud deployment
- Report generation
- Data visualization

---

## 🔐 User Authentication

- Google OAuth Login
- Secure user authentication
- Personalized dashboard

---

# ⚙️ Technologies Used

## Programming Languages

- Python
- HTML5
- CSS3
- JavaScript

---

## AI & Computer Vision

- YOLOv8 (Ultralytics)
- OpenCV
- BoT-SORT Tracker
- NumPy

---

## Backend

- Flask
- REST APIs

---

## Frontend

- HTML
- CSS
- Bootstrap
- JavaScript
- Leaflet.js

---

## Database

- SQLite
- Supabase (Cloud Storage)

---

## Maps

- OpenStreetMap
- Leaflet.js

---

## Deployment

- Hugging Face Spaces
- Docker

---

## Dataset

- **RDD2022 (Road Damage Dataset)**
- **Roboflow Annotated Dataset**

---

# 🚀 Usage

## Clone Repository

```bash
git clone https://github.com/yourusername/RoadTrax.git

cd RoadTrax
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
python app.py
```

---

## Open Browser

```
http://127.0.0.1:5000
```

---

# 🔄 Project Workflow

### 1️⃣ Data Input

- Image Upload
- Video Upload
- Live Camera Feed

↓

### 2️⃣ AI Detection

- YOLOv8 Object Detection
- Bounding Box Generation
- Confidence Score Calculation

↓

### 3️⃣ Severity Analysis

- Small
- Medium
- Large

↓

### 4️⃣ GPS Mapping

- Capture Location
- Store Coordinates
- Plot on Interactive Map

↓

### 5️⃣ Dashboard

- Reports
- Analytics
- Detection History
- Severity Charts

---

# 📊 Sample Outputs

The application generates:

- 🛣️ Detected pothole images
- 🎥 Annotated video outputs
- 📍 GPS-based road maps
- 📊 Detection dashboard
- 📈 Severity analysis charts
- 📋 Historical reports
- 📍 Interactive map visualization
- 📄 Downloadable reports

---

# 📈 Evaluation Metrics

| Metric | Performance |
|---------|-------------|
| Precision | **87%** |
| Recall | **84%** |
| mAP@0.5 | **86.8%** |
| Inference Speed | **30 FPS** |
| GPS Accuracy | **3–5 m** |
| Image Detection Time | **~1.8 sec** |
| Dashboard Load Time | **~1.2 sec** |

---

# 🔮 Future Enhancements

## 🚧 Planned Improvements

- Mobile application
- Android & iOS support
- Real-time municipality alerts
- Road repair tracking
- Citizen complaint portal
- Offline detection mode

---

## 🤖 AI Improvements

- Road crack detection
- Speed breaker detection
- Waterlogging detection
- Semantic road segmentation
- Multi-class road damage detection
- Explainable AI (XAI)
- Edge AI deployment

---

# 👥 Team Members

**B.Tech CSE (2022–2026)**

**Amrita School of Computing, Bengaluru**

**Amrita Vishwa Vidyapeetham**

- **Naga Ruthvika Durupudi**
- **Baba Ameer Shaik**
- **Adapala Rishi Manikanta**
- **Tegulapalle Venkateswara Reddy**

---

# 👨‍🏫 Mentor

**Dr. Nandu C. Nair**

Assistant Professor

Department of Computer Science and Engineering

Amrita School of Computing, Bengaluru

Amrita Vishwa Vidyapeetham

---

# 🙏 Acknowledgment

RoadTrax was developed as part of the **B.Tech Computer Science and Engineering (2022–2026)** curriculum at **Amrita School of Computing, Bengaluru**, under the guidance of **Dr. Nandu C. Nair**.

We express our sincere gratitude to our mentor, faculty members, and Amrita Vishwa Vidyapeetham for their continuous support, valuable guidance, and encouragement throughout the development of this project.

---

# 🏛️ Developed At

**Amrita School of Computing, Bengaluru**

**Amrita Vishwa Vidyapeetham, India**
