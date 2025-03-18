import tkinter as tk
from tkinter import *
from tkinter import filedialog, simpledialog, messagebox, ttk
from tkinter.filedialog import askopenfilename
import os
import sys
import time
import subprocess
import sqlite3
import numpy as np
import pandas as pd
import cv2 as cv
import pytesseract as tess
from PIL import Image, ImageTk

from yoloDetection import detectObject, displayImage
from chellan_issue import issue_challan

# ✅ Using only TensorFlow's Keras (no standalone keras)
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical



global filename
global class_labels
global cnn_model
global cnn_layer_names

frame_count = 0
frame_count_out = 0

confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416
global option
labels_value = []

frame_count = 0
frame_count_out = 0

confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416

labels_value = []

import sqlite3
import os

# Setup Database Path
db_path = os.path.abspath("database.db")
print("Database setup path:", db_path)

# Connect to Database
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Check Existing Table Structure
c.execute("PRAGMA table_info(riders);")
columns = c.fetchall()
print("Existing Columns:", columns)  # Debugging Print

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.saving import register_keras_serializable

# Register Sequential model to avoid deserialization issues
@register_keras_serializable()
class CustomSequential(tf.keras.models.Sequential):
    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Check if the model has at least one layer.
        if "layers" in config and len(config["layers"]) > 0:
            first_layer = config["layers"][0]
            # If the first layer has a batch_input_shape defined, check its dimensions.
            if "config" in first_layer and "batch_input_shape" in first_layer["config"]:
                bis = first_layer["config"]["batch_input_shape"]
                # If the batch_input_shape has 5 elements and the second element is None,
                # remove that extra dimension.
                if isinstance(bis, (list, tuple)) and len(bis) == 5 and bis[1] is None:
                    new_bis = (bis[0],) + tuple(bis[2:])  # Result will be (None, 64, 64, 3)
                    first_layer["config"]["batch_input_shape"] = new_bis
                    config["layers"][0] = first_layer
        return super(CustomSequential, cls).from_config(config, custom_objects)



# Rename Column (Only if "number_plate" Exists)
column_names = [col[1] for col in columns]
if "number_plate" in column_names:
    try:
        c.execute("ALTER TABLE riders RENAME COLUMN number_plate TO license_plate;")
        conn.commit()
        print("Column renamed successfully!")
    except sqlite3.Error as e:
        print(f"Error: {e}")
else:
    print("Column 'number_plate' does not exist. Skipping rename.")

# Create Table if Not Exists
try:
    c.execute('''
        CREATE TABLE IF NOT EXISTS riders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            license_plate TEXT UNIQUE,
            name TEXT,
            phone_number TEXT,
            email TEXT,
            due_challan INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    print("Table 'riders' created or already exists.")
except sqlite3.Error as e:
    print(f"Error creating table: {e}")

# Close Connection
conn.close()



# Load label names
labels_value = []
with open(r"C:\Users\saine\Downloads\Major Project (2)\Major Project\code\Models\labels.txt", "r") as file:
    for line in file:
        line = line.strip('\n').strip()
        labels_value.append(line)

# Load YOLO model from JSON and weights
import json
with open(r"C:\Users\saine\Downloads\Major Project (2)\Major Project\code\Models\model.json", "r") as json_file:
    loaded_model_json = json_file.read()

config = json.loads(loaded_model_json)
if "layers" in config and len(config["layers"]) > 0:
    first_layer = config["layers"][0]
    if "config" in first_layer and "batch_input_shape" in first_layer["config"]:
        bis = first_layer["config"]["batch_input_shape"]
        if isinstance(bis, list) and len(bis) == 5 and bis[1] is None:
            first_layer["config"]["batch_input_shape"] = (bis[0],) + tuple(bis[2:])
            config["layers"][0] = first_layer
loaded_model_json = json.dumps(config)

    
# Use CustomSequential to deserialize properly
plate_detecter = model_from_json(
    loaded_model_json,
    custom_objects={"Sequential": CustomSequential, "CustomSequential": CustomSequential}
)


# Load model weights
plate_detecter.load_weights(r"C:\Users\saine\Downloads\Major Project (2)\Major Project\code\Models\model_weights.h5")

# Load YOLO configuration
classesFile = r"C:\Users\saine\Downloads\Major Project (2)\Major Project\code\Models\obj.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# YOLO configuration and weights
modelConfiguration = r"C:\Users\saine\Downloads\Major Project (2)\Major Project\code\Models\yolov3-obj.cfg"
modelWeights = r"C:\Users\saine\Downloads\Major Project (2)\Major Project\code\Models\yolov3-obj_2400.weights"

# Initialize YOLO object detection
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)






# Now you can use plate_detecter for prediction without _make_predict_function()
# Example: Make a prediction
# You can prepare your image here (e.g., preprocess and reshape)
# image = cv.imread('path_to_image')
# prediction = plate_detecter.predict(image)  # Call predict with your input

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

'''from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense'''




import sqlite3

conn = sqlite3.connect("riders.db")  # Ensure the correct database file
c = conn.cursor()

c.execute("PRAGMA table_info(riders);")
columns = c.fetchall()

print("Current columns in 'riders' table:")
for col in columns:
    print(col)

conn.close()

def exit_program():
    print("Exiting program due to an error.")
    exit(1)  # Terminates the program with status code 1

def load_models():
    global class_labels_lp, plate_detecter

    try:
        # Recreate the model architecture:
        plate_detecter = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),  
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),  
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),      
            Dense(20, activation='softmax')       
        ])

        plate_detecter.load_weights(r"C:\Users\saine\Downloads\Major Project (2)\Major Project\code\Models\model_weights.h5")  # Load the weights

    except Exception as e:
        print(f"Error loading license plate models: {e}")
        exit_program(root)


def loadLibraries():  # function to load yolov3 model weight and class labels
    global class_labels, cnn_model, cnn_layer_names

    try:
        # Use raw strings and full paths:
        with open(r"C:\Users\saine\Downloads\Major Project (2)\Major Project\code\yolov3model\yolov3-labels", 'r') as f:
            class_labels = f.read().rstrip('\n').split('\n')  # Correct way to read labels

        print(str(class_labels) + " == " + str(len(class_labels)))

        cnn_model = cv.dnn.readNetFromDarknet(
            r"C:\Users\saine\Downloads\Major Project (2)\Major Project\code\yolov3model\yolov3.cfg",
            r"C:\Users\saine\Downloads\Major Project (2)\Major Project\code\yolov3model\yolov3.weights"
        )  # reading model

        layer_names = cnn_model.getLayerNames()  # Get all layer names first
        cnn_layer_names = [layer_names[i - 1] for i in cnn_model.getUnconnectedOutLayers()]  # Correct way to get unconnected layers

    except FileNotFoundError as fnf_error:  # Handle file not found
        print(f"Error loading YOLO files: {fnf_error}")
        exit_program(root) # Exit if files not found

    except Exception as e:  # Catch other potential errors
        print(f"An error occurred: {e}")
        exit_program(root)
        

def upload(): 
    global filename
    #filename = filedialog.askopenfilename(initialdir="bikes")
    filename = filedialog.askopenfilename(initialdir="bikes", title="Select Image",
                                      filetypes=(("Image files", "*.jpg;*.png"), ("All files", "*.*")))

    messagebox.showinfo("File Information", "image file loaded")
    detectBike()
    


def detectBike():
    global option
    option = 0
    indexno = 0
    label_colors = (0,255,0)
    try:
        image = cv.imread(filename)
        image_height, image_width = image.shape[:2]
    except:
        raise 'Invalid image path'
    finally:
        image, ops = detectObject(cnn_model, cnn_layer_names, image_height, image_width, image, label_colors, class_labels,indexno)
        if ops == 1:
            displayImage(image,0)
            option = 1
        else:
            displayImage(image,0)    
    detectHelmet()

# Main application
root = tk.Tk()
root.title("Safe Plate Guardian")
root.geometry("800x600")
    
'''def detectBike():
    global option
    option = 0
    try:
        image = cv.imread(filename)
        if image is None:
            raise ValueError("Failed to load image. Check the file path.")
        image_height, image_width = image.shape[:2]
    except Exception as e:
        messagebox.showerror("Error", f"Error loading image: {e}")
        return
    detectHelmet()'''

    


'''def drawPred(classId, conf, left, top, right, bottom,frame,option):
    global frame_count
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    label_name,label_conf = label.split(':')
    print(label_name+" === "+str(conf)+"==  "+str(option))
    if label_name == 'Helmet' and conf > 0.50:
        if option == 0 and conf > 0.90:
            cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
            frame_count+=1
        if option == 0 and conf < 0.90:
            cv.putText(frame, "Helmet Not detected", (10, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            frame_count+=1
            img = cv.imread(filename)
            img = cv.resize(img, (64,64))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1,64,64,3)
            X = np.asarray(im2arr)
            X = X.astype('float32')
            X = X/255
            preds = plate_detecter.predict(X)
            predict = np.argmax(preds)


            license_plate_number = labels_value[predict]
            license_plate_number = license_plate_number.replace(" ","")
            output_folder = os.path.join('static', 'outputImages')
            os.makedirs(output_folder, exist_ok=True)

            output_filename = os.path.join(output_folder, f"{license_plate_number}.jpg")
            cv.imwrite(output_filename, frame)

            textarea.insert(END,filename+"\n\n")
            textarea.insert(END,"Number plate detected as "+str(labels_value[predict]))  
            print(labels_value[predict])  
            issue_chellan(labels_value[predict])

        if option == 1:
            cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
            frame_count+=1    
    
        
    if(frame_count> 0):
        return frame_count
'''
def drawPred(classId, conf, left, top, right, bottom,frame,option):
    global frame_count
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    label_name,label_conf = label.split(':')
    print(label_name+" === "+str(conf)+"==  "+str(option))
    if label_name == 'Helmet' and conf > 0.50:
        if option == 0 and conf > 0.90:
            cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
            frame_count+=1
        if option == 0 and conf < 0.90:
            cv.putText(frame, "Helmet Not detected", (10, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            frame_count+=1
            # *** Start of corrected license plate prediction ***
            try:
                # Extract the region of interest (ROI) containing the license plate
                plate_roi = frame[top:bottom, left:right] # Corrected slicing
                
                if plate_roi.size > 0:  # Check if ROI is valid
                    img = cv.resize(plate_roi, (64, 64))
                    img = img.astype('float32') / 255.0  # Normalize
                    img = np.expand_dims(img, axis=0)  # Add batch dimension

                    preds = plate_detecter.predict(img)
                    predict = np.argmax(preds)
                    license_plate_number = labels_value[predict]
                    license_plate_number = license_plate_number.replace(" ","")

                    output_folder = os.path.join('static', 'outputImages')
                    os.makedirs(output_folder, exist_ok=True)
                    output_filename = os.path.join(output_folder, f"{license_plate_number}.jpg")
                    cv.imwrite(output_filename, frame)

                    textarea.insert(END,filename+"\n\n")
                    textarea.insert(END,"Number plate detected as "+str(license_plate_number))  # Use the cleaned number
                    print(license_plate_number)  # Print the cleaned number
                    issue_challan(license_plate_number) #Pass correct number to issue_chellan

                else:
                    print("Error: Invalid ROI for license plate.")

            except Exception as e:
                print(f"Error during license plate prediction: {e}")
            # *** End of corrected license plate prediction ***
        if option == 1:
            cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
            frame_count+=1    
    
        
    if(frame_count> 0):
        return frame_count



def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

def postprocess(frame, outs, confidence_threshold=0.5):
    height, width = frame.shape[:2]
    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices = cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"Helmet Detected" if class_ids[i] == 0 else "Helmet Not Detected"
        color = (0, 255, 0) if class_ids[i] == 0 else (0, 0, 255)
        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

'''

def postprocess(frame, outs, option):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    global frame_count_out
    frame_count_out=0
    classIds = []
    confidences = [] 
    boxes = []
    classIds = []
    confidences = []
    boxes = []
    cc = 0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                print(classIds)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    count_person=0 # for counting the classes in this loop.
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        frame_count_out = drawPred(classIds[i], confidences[i], left, top, left + width, top + height,frame,option)
        my_class='Helmet'      
        unknown_class = classes[classId]
        print("===="+str(unknown_class))
        if my_class == unknown_class:
            count_person += 1
            
    print(str(frame_count_out))
    if count_person == 0 and option == 1:
        cv.putText(frame, "Helmet Not detected", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
    if count_person >= 1 and option == 0:
        path = 'test_out/'
        cv.imwrite(str(path)+str(cc)+".jpg", frame)     # writing to folder.
        cc = cc + 1
        frame = cv.resize(frame,(500,500))
        cv.imshow('img',frame)
        cv.waitKey(50)
'''

def detectHelmet():
    textarea.delete('1.0', END)
    global option
    if option == 1:
        frame = cv.imread(filename)
        frame_count =0
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        postprocess(frame, outs,0)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        print(label)
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        print(label)
    else:
        messagebox.showinfo("Person & Motor bike not detected in uploaded image", "Person & Motor bike not detected in uploaded image")





import cv2 as cv
import os
from tkinter import filedialog, messagebox

def videoHelmetDetect():
    videofile = filedialog.askopenfilename()
    if not videofile:
        return  # Exit if no file is selected

    video = cv.VideoCapture(videofile)
    if not video.isOpened():
        messagebox.showerror("Error", "Unable to open video file")
        return

    # Use absolute paths
    yolo_path = r"C:\Users\saine\Downloads\Major Project (2)\Major Project\code\yolov3model"
    weights_path = os.path.join(yolo_path, "yolov3.weights")
    config_path = os.path.join(yolo_path, "yolov3.cfg")

    # Load YOLO model
    net = cv.dnn.readNet(weights_path, config_path)

    layer_names = getOutputsNames(net)
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        blob = cv.dnn.blobFromImage(frame, 1/255, (416, 416), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(layer_names)
        postprocess(frame, outs, 0.5)
        
        cv.imshow("Helmet Detection", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()

'''
def videoHelmetDetect():
    global filename
    videofile = askopenfilename(initialdir = "videos")
    video = cv.VideoCapture(videofile)
    while(True):
        ret, frame = video.read()
        if ret == True:
            frame_count = 0
            filename = "temp.png"
            cv.imwrite("temp.png",frame)
            blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
            net.setInput(blob)
            outs = net.forward(getOutputsNames(net))
            postprocess(frame, outs,0)
            t, _ = net.getPerfProfile()
            label=''
            cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv.imshow("Predicted Result", frame)
            if cv.waitKey(5) & 0xFF == ord('q'):
                break  
        else:
            break
    video.release()
    cv.destroyAllWindows()
'''
'''
def videoOn():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access the camera!")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv.imshow("Camera Feed", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

'''

import cv2 
import numpy as np 
import os
import tkinter.messagebox as messagebox

import cv2
import numpy as np
import os

def videoOn():
    # Define Paths
    weights_path =r"C:/Users/saine/Downloads/Major Project (2)/Major Project/code/yolov3model/yolov3.weights"
    config_path = r"C:/Users/saine/Downloads/Major Project (2)/Major Project/code/yolov3model/yolov3.cfg"
    labels_path = r"C:/Users/saine/Downloads/Major Project (2)/Major Project/code/yolov3model/yolov3-labels"

    # Check if model files exist
    for path in [weights_path, config_path, labels_path]:
        if not os.path.exists(path):
            print(f"Error: '{path}' not found!")
            return

    # Load YOLO Model
    try:
        net = cv2.dnn.readNet(weights_path, config_path)
    except cv2.error as e:
        print(f"Error loading YOLO model: {e}")
        return

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load Labels
    with open(labels_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Open Webcam
    cap = cv2.VideoCapture(0)  # Change index if you have multiple cameras
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        height, width, _ = frame.shape

        # YOLO Preprocessing
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        detected_classes = []
        confidences = []
        boxes = []

        # Process Detection Results
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id >= len(classes):  # Prevent IndexError
                    continue

                if confidence > 0.3:  # Confidence threshold
                    center_x, center_y, w, h = map(int, detection[:4] * [width, height, width, height])
                    x, y = center_x - w // 2, center_y - h // 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    detected_classes.append(classes[class_id])

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

        helmet_detected = False
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                x, y, w, h = boxes[i]
                label = detected_classes[i]
                color = (0, 255, 0) if label == "helmet" else (0, 0, 255)

                if label == "helmet":
                    helmet_detected = True

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display Helmet Detection Result
        status = "Helmet Detected" if helmet_detected else "No Helmet Detected"
        color = (0, 255, 0) if helmet_detected else (0, 0, 255)
        cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow("Helmet Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def exit():
    global root
    root.destroy()
    

loadLibraries()


import tkinter as tk
from tkinter import messagebox, scrolledtext
import sqlite3
from PIL import Image, ImageTk

def show_home_page():
    global current_page
    if current_page:
        current_page.destroy()

    current_page = tk.Frame(root, bg="#f0f0f0")  # Set background color
    current_page.pack(fill="both", expand=True)

    tk.Label(current_page, text="Home Page", font=("Helvetica", 30), bg="#f0f0f0").pack(pady=50)  # Increase font size and padding


    # Admin Login Button
    admin_login_btn = tk.Button(current_page, text="Admin Login", command=admin_login, bg="#007bff", fg="white", activebackground="#0056b3", activeforeground="white", padx=20, pady=10, bd=0, font=("Helvetica", 16))  # Increase font size and padding
    admin_login_btn.pack(pady=20)
   

    # Check Challans Button
    check_challans_btn = tk.Button(current_page, text="Check Challans", command=show_user_page, bg="#28a745", fg="white", activebackground="#218838", activeforeground="white", padx=20, pady=10, bd=0, font=("Helvetica", 16))  # Increase font size and padding
    check_challans_btn.pack(pady=20)



def admin_login():
    admin_username = "admin"
    admin_password = "password"  # Store securely in a real-world scenario

    login_window = tk.Toplevel(root)
    login_window.title("Admin Login")
    login_window.geometry("300x200")
    login_window.configure(background="#f0f0f0")

    tk.Label(login_window, text="Username:", bg="#f0f0f0", font=("Helvetica", 14)).pack(pady=10)
    username_entry = tk.Entry(login_window, font=("Helvetica", 14))
    username_entry.pack()

    tk.Label(login_window, text="Password:", bg="#f0f0f0", font=("Helvetica", 14)).pack(pady=10)
    password_entry = tk.Entry(login_window, show="*", font=("Helvetica", 14))
    password_entry.pack()

    def check_login():
        username = username_entry.get().strip()
        password = password_entry.get().strip()
        print(f"Entered Username: {username}, Password: {password}")  # Debugging
        if username == admin_username and password == admin_password:
            messagebox.showinfo("Login Successful", "Welcome, Admin!")
            login_window.destroy()
            show_admin_page()  # Navigate to the Admin page
        elif username == "user":  # Example for user login
            messagebox.showinfo("Login Successful", "Welcome, User!")
            login_window.destroy()
            show_user_page()  # Navigate to the User page
        else:
            messagebox.showerror("Login Failed", "Invalid username or password.")


    login_button = tk.Button(login_window, text="Login", command=check_login, bg="#007bff", fg="white", font=("Helvetica", 14))
    login_button.pack(pady=10)


def show_admin_page():
    global current_page, textarea
    if current_page:
        current_page.destroy()

    current_page = tk.Frame(root, bg="#f0f0f0")  # Set background color
    current_page.pack(fill="both", expand=True)

    tk.Label(current_page, text="Admin Page", font=("Helvetica", 30), bg="#f0f0f0").pack(pady=50)  # Increase font size and padding

    # Add Text Area
    textarea = scrolledtext.ScrolledText(current_page, width=40, height=10, font=("Helvetica", 14))  # Increase font size
    textarea.pack(side="right", pady=20)

    button_frame = tk.Frame(current_page, bg="#f0f0f0")  # Set background color for the button frame
    button_frame.pack(side="left", padx=20)  # Set padding for the button frame

    tk.Button(button_frame, text="Upload Image", bg="#007bff", fg="white", activebackground="#0056b3", activeforeground="white", padx=20, pady=10, bd=0, font=("Helvetica", 16) ,command=upload).pack(pady=10)  # Increase font size and padding
    tk.Button(button_frame, text="Upload Video", bg="#007bff", fg="white", activebackground="#0056b3", activeforeground="white", padx=20, pady=10, bd=0, font=("Helvetica", 16) ,command=videoHelmetDetect).pack(pady=10)  # Increase font size and padding
    tk.Button(button_frame, text="Camera", bg="#007bff", fg="white", activebackground="#0056b3", activeforeground="white", padx=20, pady=10, bd=0, font=("Helvetica", 16) ,command=videoOn).pack(pady=10)  # Increase font size and padding
    tk.Button(button_frame, text="Log out", command=show_home_page, bg="#dc3545", fg="white", activebackground="#c82333", activeforeground="white", padx=20, pady=10, bd=0, font=("Helvetica", 16)).pack(pady=10)  # Increase font size and padding

def show_user_page():
    global current_page, license_plate_entry
    if current_page:
        current_page.destroy()

    current_page = tk.Frame(root, bg="#f0f0f0")  # Set background color
    current_page.pack(fill="both", expand=True)

    tk.Label(current_page, text="User Page", font=("Helvetica", 30), bg="#f0f0f0").pack(pady=50)  # Increase font size and padding

    tk.Label(current_page, text="Enter License Plate Number:", bg="#f0f0f0", font=("Helvetica", 16)).pack(pady=10)  # Increase font size and padding
    license_plate_entry = tk.Entry(current_page, font=("Helvetica", 14))  # Increase font size
    license_plate_entry.pack(pady=10)

    check_btn = tk.Button(current_page, text="Check", command=check_challans, bg="#007bff", fg="white", activebackground="#0056b3", activeforeground="white", padx=20, pady=10, bd=0, font=("Helvetica", 16))  # Increase font size and padding
    check_btn.pack(pady=10)
    
    back_btn = tk.Button(current_page, text="Back", command=show_home_page, bg="#dc3545", fg="white", activebackground="#c82333", activeforeground="white", padx=20, pady=10, bd=0, font=("Helvetica", 16))  # Increase font size and padding
    back_btn.pack(pady=10)
    

'''def check_challans():
    global current_page, license_plate_entry
    license_plate = license_plate_entry.get()
    
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Execute a SELECT query to fetch data for the given license plate
    #c.execute("SELECT * FROM riders WHERE license_plate=?", (license_plate,))
    c.execute("SELECT * FROM riders WHERE license_plate=?", (license_plate,))

    data = c.fetchone()
    
    # Close the database connection
    conn.close()

    # If data is found for the given license plate
    if data:
        id, license_plate, name, phone_number, email, due_challan = data
        challan_details = f"License Plate: {license_plate}\nOwner: {name}\nDue Challan: {due_challan}"

        # Destroy the current page if it exists
        if current_page:
            current_page.destroy()
        
        # Create a new frame for displaying challan details
        current_page = tk.Frame(root)
        current_page.pack(fill="both", expand=True)
        
        # Display the challan details
        tk.Label(current_page, text="Challan Details", font=("Helvetica", 18)).pack(pady=20)
        tk.Label(current_page, text=challan_details, wraplength=400, font=("Helvetica", 14)).pack(pady=10)
        
        # Display the image
        image_path = f"static/outputImages/{license_plate}.jpg"  # Assuming the image file name is the same as the license plate number
        try:
            image = Image.open(image_path)
            image = image.resize((200, 200))  # Resize the image if needed
            photo = ImageTk.PhotoImage(image)

            image_label = tk.Label(current_page, image=photo)
            image_label.image = photo
            image_label.pack(pady=20)
        except FileNotFoundError:
            messagebox.showinfo("Image Info", "Image not found.")
        
        # Add a back button to return to the home page
        back_btn = tk.Button(current_page, text="Back", command=show_user_page, bg="#dc3545", fg="white", activebackground="#c82333", activeforeground="white", padx=20, pady=10, bd=0, font=("Helvetica", 16))
        back_btn.pack(pady=10)
    else:
        messagebox.showinfo("Challans Info", "No challans found for the provided license plate.")
'''

import os
import sqlite3

db_path = os.path.abspath("database.db")
print("Testing database at:", db_path)

try:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM riders")  # Try to select
    print("Successfully selected from 'riders' table (if no error).")
    conn.close()
except sqlite3.OperationalError as e:
    print(f"Database Error: {e}")  # Print the exact error!
except Exception as e:
    print(f"Other Error: {e}")
    


import os
from PIL import Image, ImageTk
import sqlite3
import tkinter as tk  # Make sure tkinter is imported
from tkinter import messagebox

def check_challans():
    
    global current_page, license_plate_entry, root  # Avoid global if possible

    license_plate = license_plate_entry.get().strip()  # Remove spaces
    db_path = os.path.abspath("database.db")

    print(f"Database path in check_challans: {db_path}")
    print(f"License plate from entry (stripped): '{license_plate}'")

    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Use '=' for exact match, and ensure single-element tuple has a comma
        c.execute("SELECT * FROM riders WHERE license_plate = ? COLLATE NOCASE", (license_plate,))
        data = c.fetchone()

        print(f"Query: SELECT * FROM riders WHERE license_plate = '{license_plate}' COLLATE NOCASE")
        print("Query Result:", data)

        if data is None:
            messagebox.showinfo("Challans Info", "No challans found for the provided license plate.")
            return

        # Handle missing name/phone_number
        if len(data) == 5:
            id, license_plate, issued_chellan, name, phone_number = data
        elif len(data) == 3:
            id, license_plate, issued_chellan = data
            name, phone_number = "N/A", "N/A"

        challan_details = f"License Plate: {license_plate}\nOwner: {name}\nIssued Challan: {issued_chellan}\nPhone Number: {phone_number}"

        if current_page:
            current_page.destroy()

        current_page = tk.Frame(root)
        current_page.pack(fill="both", expand=True)

        tk.Label(current_page, text="Challan Details", font=("Helvetica", 18)).pack(pady=20)
        tk.Label(current_page, text=challan_details, wraplength=400, font=("Helvetica", 14)).pack(pady=10)

        # Corrected image path
        image_path = os.path.join(
            "C:",
            "Users",
            "saine",
            "Downloads",
            "Major Project (2)",
            "Major Project",
            "code",
            "static",
            "outputImages",
            f"{license_plate}.jpg"
        )

        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            messagebox.showinfo("Image Info", f"Image not found at: {image_path}")
            return
        except Exception as e:
            print(f"Error opening image: {e}")
            messagebox.showinfo("Image Error", f"Error opening image: {e}")
            return

        image = image.resize((200, 200))
        photo = ImageTk.PhotoImage(image)

        image_label = tk.Label(current_page, image=photo)
        image_label.image = photo  # Keep a reference!
        image_label.pack(pady=20)

        back_btn = tk.Button(current_page, text="Back", command=show_user_page,
                             bg="#dc3545", fg="white", activebackground="#c82333",
                             activeforeground="white", padx=20, pady=10, bd=0,
                             font=("Helvetica", 16))
        back_btn.pack(pady=10)

    except sqlite3.Error as e:
        print(f"Database error in check_challans: {e}")
        messagebox.showerror("Error", "A database error occurred. Please try again.")

    finally:
        if 'conn' in locals() and conn:
            conn.close()


'''
def check_challans():
    global current_page, license_plate_entry

    license_plate = license_plate_entry.get()

    db_path = os.path.abspath("database.db")
    print("Database path in check_challans:", db_path)  # Print the path

    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute("SELECT * FROM riders WHERE license_plate=?", (license_plate,))# Corrected line

        data = c.fetchone()

        if data:
            id, license_plate,issued_challan = data  # Use number_plate
            challan_details = f"License Plate: {license_plate}\nOwner: {name or 'N/A'}\nPhone: {phone_number or 'N/A'}\nEmail: {email or 'N/A'}\nDue Challan: {due_challan or 0}"

            if current_page:
                current_page.destroy()

            current_page = tk.Frame(root)
            current_page.pack(fill="both", expand=True)

            tk.Label(current_page, text="Challan Details", font=("Helvetica", 18)).pack(pady=20)
            tk.Label(current_page, text=challan_details, wraplength=400, font=("Helvetica", 14)).pack(pady=10)

            image_path = os.path.join(
                "C:",
                "Users",
                "saine",
                "Downloads",
                "Major Project (2)",
                "Major Project",
                "code",
                "static",
                "outputImages",
                f"{license_plate}.jpg"  # Use license_plate from entry
            )
            print("Image path:", image_path)

            try:
                image = Image.open(image_path)
                image = image.resize((200, 200))
                photo = ImageTk.PhotoImage(image)

                image_label = tk.Label(current_page, image=photo)
                image_label.image = photo
                image_label.pack(pady=20)

            except FileNotFoundError:
                messagebox.showinfo("Image Info", f"Image not found at: {image_path}")
            except Exception as e:
                messagebox.showinfo("Image Error", f"Error opening image: {e}")

            back_btn = tk.Button(current_page, text="Back", command=show_user_page,
                                 bg="#dc3545", fg="white", activebackground="#c82333",
                                 activeforeground="white", padx=20, pady=10, bd=0,
                                 font=("Helvetica", 16))
            back_btn.pack(pady=10)

        else:
            messagebox.showinfo("Challans Info", "No challans found for the provided license plate.")

    except sqlite3.Error as e:
        print(f"Database error in check_challans: {e}")
        messagebox.showerror("Error", "A database error occurred. Please try again.")

    finally:
        if 'conn' in locals() and conn:
            conn.close()'''



if __name__ == "__main__":
    root = tk.Tk()
    root.title("Safe Plate Guardian")
    root.geometry("800x700")
    root.configure(background="#f0f0f0")

    # # Load and resize logos
    # logo_left_img = Image.open("logo_left.png")  # Adjust path as needed
    # logo_left_img = logo_left_img.resize((100, 100), Image.ANTIALIAS)
    # logo_left = ImageTk.PhotoImage(logo_left_img)

    # logo_right_img = Image.open("logo_right.png")  # Adjust path as needed
    # logo_right_img = logo_right_img.resize((100, 100), Image.ANTIALIAS)
    # logo_right = ImageTk.PhotoImage(logo_right_img)

    current_page = None
    textarea = None
    license_plate_entry = None

    show_home_page()

    root.mainloop()




