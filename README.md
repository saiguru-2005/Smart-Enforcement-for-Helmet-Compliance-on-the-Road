#  Safe Plate Guardian: Smart Enforcement for Helmet Compliance on the Road


Table of Contents :
Overview
Features
High-Level Flow
Technologies Used
Project Structure
Installation
Usage


Overview
Safe Plate Guardian is an automated system that leverages deep learning to detect motorcycle riders who are not wearing helmets and automatically extract their vehicle license plate details. This project aims to improve road safety by enforcing helmet compliance in real time.

Key goals:

Automated Detection: Identify motorcyclists without helmets using advanced object detection.
License Plate Extraction: Perform OCR on the identified license plates for enforcement.
Real-Time Alerts: Quickly issue challans or notifications to violators.
Scalable: Can be integrated with existing CCTV or traffic camera infrastructure.

Features
Helmet Detection
Uses YOLO-based deep learning models (YOLOv2, YOLOv3, YOLOv5) for accurate detection of motorcyclists and helmet usage.

License Plate Recognition
Integrates Optical Character Recognition (OCR) to read and record vehicle license plates from detected frames.

Multiple Object Tracking
Monitors multiple motorcyclists simultaneously in real-time video streams.

Automated Enforcement
Generates alerts or challans based on detected violations, streamlining the penalty process.

Data Management
Optionally stores violation data in a database (e.g., SQL) for record-keeping and reporting.


High-Level Flow

<details> <summary>**Example with an ASCII Diagram**</summary>

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
[ Enforcement Actions ] --> Issue challan / log violation
</details>


Technologies Used:

Python 3.x Main programming language.

Deep Learning
 *YOLOv2 / YOLOv3 / YOLOv5 for object detection.
 *TensorFlow or PyTorch (depending on your chosen framework) for model training/inference.

Computer Vision
 * OpenCV for image and video stream handling.

OCR
 *Tesseract or any preferred OCR engine for reading license plates.

Database (Optional)
 *SQL (MySQL, PostgreSQL, or SQLite) for storing and managing violation records.



Installation1:
Clone the Repository:

git clone https://github.com/saiguru-2005/Smart-Enforcement-for-Helmet-Compliance-on-the-Road.git



cd safe-plate-guardian


Install Dependencies:
Make sure you have Python 3.x installed, then run:

pip install --upgrade pip
pip install -r requirements.txt


Set Up Pre-trained Models
Download YOLO weights (e.g., YOLOv5) and place them in data/pre-trained-models/.
Adjust paths in your configuration scripts if needed.

Configure OCR:
Install Tesseract or another OCR engine if not included in your environment.
Update any OCR-related paths in ocr.py or relevant config files.


Usage:
Run the Main Application

python src/main.py --input path_to_video_or_camera
         * --input can be a video file path or a camera index (e.g., 0 for default webcam).
Training Custom Models (Optional):
  If you wish to train or fine-tune YOLO on your own dataset:
python src/train.py --epochs 50 --batch_size 8 --data data_config.yaml

Make sure your dataset is prepared under data/ and labeled appropriately.


Output and Logs:
 Processed images or violation snapshots are saved in results/output_images/.
 Logs (and optionally a database of violations) are stored in the results/ directory.



You can get whole project here ---->         https://drive.google.com/file/d/1JjP5KiWm44um6F5geaUdiX9eHsELI6vT/view?usp=sharing


