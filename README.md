#  Safe Plate Guardian: Smart Enforcement for Helmet Compliance on the Road




## Table of Contents
- [Overview](#overview)
- [Key Goals](#key-goals)
- [Features](#features)
- [High-Level-Flow](#high-level-flow)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)

## Overview
Safe Plate Guardian is an automated system that leverages deep learning to detect motorcycle riders who are not wearing helmets and automatically extract their vehicle license plate details. This project improves road safety by enforcing helmet compliance in real time.

## Key Goals
- **Automated Detection:** Identify motorcyclists without helmets using advanced object detection.
- **License Plate Extraction:** Perform OCR on the identified license plates for enforcement.
- **Real-Time Alerts:** Quickly issue challans or notifications to violators.
- **Scalable:** Seamlessly integrate with existing CCTV or traffic camera infrastructure.

## Features

### Helmet Detection
Uses YOLO-based deep learning models (YOLOv2, YOLOv3, YOLOv5) for accurate detection of motorcyclists and helmet usage.

### License Plate Recognition
Integrates Optical Character Recognition (OCR) to capture license plate details.

### Multiple Object Tracking
Monitors multiple motorcyclists simultaneously in real-time video streams.

### Automated Enforcement
Generates alerts or challans based on detected violations.

### Data Management
Optionally stores violation data in a database (e.g., SQL) for record-keeping and reporting.

## High-Level Flow

ASCII Diagram:
    [ Video Feed / Image Input ]
            |
            v
    [ YOLO-based Object Detection ] --> Detect Motorcyclist & Helmet
            |
       (if no helmet)
            v
    [ License Plate Detection + OCR ] --> Extract Plate Number
            |
            v
    [ Enforcement Actions ] --> Issue Challan / Log Violation

Mermaid Diagram:
    mermaid
    flowchart TD
        A[Video Feed / Image Input] --> B[YOLO-based Object Detection]
        B --> C{Is Helmet Present?}
        C -- Yes --> D[No Action]
        C -- No --> E[License Plate Detection + OCR]
        E --> F[Extract Plate Number]
        F --> G[Enforcement Actions]

## Technologies Used
- **Python 3.x:** Main programming language.
- **Deep Learning:**
  - YOLOv2 / YOLOv3 / YOLOv5 for object detection.
  - TensorFlow or PyTorch for model training/inference.
- **Computer Vision:**
  - OpenCV for image and video stream handling.
- **OCR:**
  - Tesseract or any preferred OCR engine.
- **Database (Optional):**
  - SQL (MySQL, PostgreSQL, or SQLite) for managing violation records.

## Project Structure
    safe-plate-guardian/
    ├── data/                 # Datasets and pre-trained models
    ├── src/                  # Source code for detection and OCR
    ├── results/              # Processed images, logs, and outputs
    ├── requirements.txt      # List of dependencies
    └── README.md             # This file

## Installation

1. Clone the Repository:
       git clone https://github.com/saiguru-2005/Smart-Enforcement-for-Helmet-Compliance-on-the-Road.git
       cd safe-plate-guardian

2. Install Dependencies:
       pip install --upgrade pip
       pip install -r requirements.txt

3. Set Up Pre-trained Models:
   - Download YOLO weights (e.g., YOLOv5) and place them in data/pre-trained-models/.
   - Adjust paths in your configuration scripts if necessary.

4. Configure OCR:
   - Install Tesseract or another OCR engine.
   - Update any OCR-related paths in ocr.py or relevant config files.

## Usage

- **Run the Main Application:**
      python src/main.py --input path_to_video_or_camera

  Note: The --input parameter can be a video file path or a camera index (e.g., 0 for the default webcam).

- **Training Custom Models (Optional):**
      python src/train.py --epochs 50 --batch_size 8 --data data_config.yaml

  Ensure your dataset is properly prepared under the data/ directory.

- **Output and Logs:**
  - Processed images or violation snapshots are saved in results/output_images/.
  - Logs (and optionally a violation database) are stored in the results/ directory.

Happy coding! Let's make our roads safer by ensuring helmet compliance in real time.




You can get whole project here ---->         https://drive.google.com/file/d/1JjP5KiWm44um6F5geaUdiX9eHsELI6vT/view?usp=sharing


