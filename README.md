# YOLO-Classroom-Detection# YOLO Classroom Object Detection Project 🚀

## 📌 Project Overview

This **Deep Learning Project** utilizes **YOLOv8** to detect **Persons, Desks, Tables, and Windows** in classroom environments. Addressing the lack of pre-existing data, the project implements a **semi-automated annotation pipeline** where pre-trained models auto-label raw video frames. This establishes an efficient end-to-end workflow covering data extraction, auto-labeling, custom model training, and real-time inference.

## 🛠️ Installation

1.  **Clone the repository**

    ```bash
    git clone https://github.com/mchtyldzx/YOLO-Classroom-Detection.git
    cd YOLO-Classroom-Detection
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Workflow Steps

### 1. Data Collection (`extract_frames.py`)

Extracts image frames from a source video (`classroom.mp4`) to create a raw dataset.

```bash
python extract_frames.py
```

### 2. Auto-Labeling (`auto_label.py`)

Uses a pre-trained YOLOv8 model to automatically detect and label "person" and "table" objects in the raw images. This significantly reduces manual labeling effort.

```bash
python auto_label.py
```

### 3. Dataset Splitting (`split_data.py`)

Splits the labeled dataset into Training (80%) and Validation (20%) sets, organizing them into the required YOLO folder structure.

```bash
python split_data.py
```

### 4. Training (`train.py`)

Trains a custom YOLOv8 model on the prepared dataset for 50 epochs.

```bash
python train.py
```

_The best model weights will be saved in `runs/detect/custom_model/weights/best.pt`._

### 5. Inference (`main.py`)

Runs the trained custom model on a video file to detect objects in real-time or save the output.

```bash
python main.py
```

## 📁 Repository Structure

- `auto_label.py`: Script for model-assisted labeling.
- `extract_frames.py`: Script to extract images from video.
- `split_data.py`: Script to organize dataset into train/val.
- `train.py`: Training script using UltraLytics YOLO.
- `main.py`: Inference script for video detection.
- `data.yaml`: YOLO dataset configuration file.

## 📝 Notes

- Since the base model does not recognize "Windows" natively, they may need to be labeled manually using a tool like [MakeSense.ai](https://www.makesense.ai/) if auto-labeling is irrelevant for that class.
- Ensure that input video is named correctly in the scripts or updated accordingly.

---
