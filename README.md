# Bamboo Growth Habit Classification

A simple yet powerful **YOLOv5s-based** desktop application that classifies bamboo images into **Running** or **Clumping** growth habits using a trained model and a user-friendly Tkinter interface.

Created by:
Dustin Jeiondre A. Ventura
Carla C. Ty
BS Computer Engineering Students
Mapua University
---

## Features

- Choose bamboo images from your computer
- Detects and draws bounding boxes on bamboo
- Displays confidence percentage for each detection
- Saves processed results
- Works with custom-trained YOLOv5 models

---

## Project Structure
```bash
BambooClassifier/
├── BambooClassifier.py              # Main application
├── yolov5/                          # YOLOv5 source code
|   ├── runs/train/.../weights/      # Trained model directory /exp/weights/best.pt
├── detected/                        # Processed Images
├── requirements.txt                 # Dependency Requirements
└── README.md                        # Step-by-step Tutorial
```

## How to Run 

### 1. Clone or Download the Project
```bash
git clone 
cd BambooClassifier
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Terminal
source .venv/Scripts/activate
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt

# In the case of hidden dependencies from YOLOv5 not being installed
cd yolov5
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python BambooClassifier.py
```

- Note: You can play around with 3 trained models under /runs with varying result parameters.
