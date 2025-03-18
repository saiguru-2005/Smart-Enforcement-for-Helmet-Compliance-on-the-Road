# Smart-Enforcement-for-Helmet-Compliance-on-the-Road


Safe Plate Guardian: Smart Enforcement for Helmet Compliance on the Road
Overview
Safe Plate Guardian is an automated system designed to enhance road safety by detecting helmet violations among motorcyclists in real time. By leveraging advanced deep learning techniques and computer vision, the system accurately identifies riders without helmets and extracts their vehicle license plate numbers for immediate enforcement actions.

Features
Automated Helmet Detection: Utilizes state-of-the-art deep learning models (YOLOv2, YOLOv3, and YOLOv5) to detect motorcyclists and assess helmet compliance.
License Plate Recognition: Integrates Optical Character Recognition (OCR) to extract vehicle registration numbers from images and videos.
Real-time Processing: Designed for real-time surveillance to ensure swift identification and enforcement.
Multiple Object Tracking: Monitors detected violations continuously across video frames.
Automated Enforcement: Facilitates quick challan issuance by linking detected violations with extracted vehicle data.
Technologies Used
Deep Learning: YOLOv2, YOLOv3, YOLOv5 for object detection.
Computer Vision: OpenCV for image and video processing.
OCR: Optical Character Recognition libraries for license plate extraction.
Programming Language: Python
Frameworks: TensorFlow and Keras (as applicable)
Project Structure
bash
Copy
Edit
safe-plate-guardian/
├── data/           # Datasets used for training and testing
├── models/         # Pre-trained models and configuration files
├── src/            # Source code for helmet detection, license plate recognition, and tracking
├── results/        # Output images, videos, and generated reports
├── requirements.txt# Project dependencies
└── README.md       # This file
Installation
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/yourusername/safe-plate-guardian.git
cd safe-plate-guardian
Install Dependencies:

Ensure you have Python 3.x installed. Then, install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
Configure the Environment:

Set up any necessary environment variables or configuration files as described in the project documentation.

Usage
Training the Model:

To train the object detection and recognition models, run the training script:

bash
Copy
Edit
python src/train.py
Real-time Detection:

To start real-time helmet detection and license plate recognition (using video input or camera feed):

bash
Copy
Edit
python src/main.py --input video_file_or_camera
Viewing Results:

Processed images, violation snapshots, and reports are saved in the results/ directory.

Contributing
Contributions are welcome! If you have suggestions, bug fixes, or improvements, please fork the repository and submit a pull request.
