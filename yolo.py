import numpy as np
import cv2 as cv
import subprocess
import time
import os
from yoloDetection import detectObject, displayImage
import sys

global class_labels
global cnn_model
global cnn_layer_names

def loadLibraries():
    global class_labels
    global cnn_model
    global cnn_layer_names

    class_labels = open(r"C:\\Users\\saine\\Downloads\\Major Project (2)\\Major Project\\code\\yolov3model\\yolov3-labels").read().strip().split("\n")
    print(str(class_labels) + " == " + str(len(class_labels)))

    cnn_model = cv.dnn.readNetFromDarknet(
        r"C:\\Users\\saine\\Downloads\\Major Project (2)\\Major Project\\code\\yolov3model\\yolov3.cfg",
        r"C:\\Users\\saine\\Downloads\\Major Project (2)\\Major Project\\code\\yolov3model\\yolov3.weights"
    )

    cnn_layer_names = cnn_model.getLayerNames()
    cnn_layer_names = [cnn_layer_names[i - 1] for i in cnn_model.getUnconnectedOutLayers().flatten()]  # ✅ Fixed indexing

def detectFromImage(imagename):
        label_colors = (0,255,0)
        try:
                image = cv.imread(imagename) #image reading
                image_height, image_width = image.shape[:2] #converting image to two dimensional array
        except:
                raise 'Invalid image path'
        finally:
                image, _, _, _, _ = detectObject(cnn_model, cnn_layer_names, image_height, image_width, image, label_colors, class_labels, indexno)
                displayImage(image, 0) #display image with detected objects label

def detectFromVideo(videoFile):
        label_colors = (0,255,0)
        indexno = 0
        try:
                video = cv.VideoCapture(videoFile)
                frame_height, frame_width = None, None
                video_writer = None
        except:
                raise 'Unable to load video'
        finally:
                while True:
                        frame_grabbed, frames = video.read()
                        if not frame_grabbed:
                                break
                        if frame_width is None or frame_height is None:
                                frame_height, frame_width = frames.shape[:2]
                        frames, _, _, _, _ = detectObject(cnn_model, cnn_layer_names, frame_height, frame_width, frames, label_colors, class_labels, indexno)
                        print(indexno)
                        if indexno == 5:
                            video.release()    
                            break

        print("Releasing resources")
        video.release()


if __name__ == '__main__':
        loadLibraries()
        print("sample commands to run code with image or video")
        print("python yolo.py image input_image_path")
        print("python yolo.py video input_video_path")
        if len(sys.argv) == 3:
                if sys.argv[1] == 'image':
                        detectFromImage(sys.argv[2])
                elif sys.argv[1] == 'video':
                        detectFromVideo(sys.argv[2])
                else:
                        print("invalid input")
        else:
                print("follow sample command to run code")
