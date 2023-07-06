import cv2
import numpy as np
import csv
import pymysql
import pyrebase
import datetime
import os
from pymongo import MongoClient

# Load YOLO model
net = cv2.dnn.readNet("yolo-tiny.weights", "yolo-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up Firebase configuration
firebase_config = {
    "apiKey": "YOUR_API_KEY",
    "authDomain": "YOUR_AUTH_DOMAIN",
    "databaseURL": "YOUR_DATABASE_URL",
    "projectId": "YOUR_PROJECT_ID",
    "storageBucket": "YOUR_STORAGE_BUCKET",
    "messagingSenderId": "YOUR_MESSAGING_SENDER_ID",
    "appId": "YOUR_APP_ID"
}

# Connect to Firebase
firebase = pyrebase.initialize_app(firebase_config)
db = firebase.database()

# Connect to SQL database
connection = pymysql.connect(
    host="localhost",
    user="yourusername",
    password="yourpassword",
    database="yourdatabase"
)
cursor = connection.cursor()

# Connect to MongoDB
mongo_client = MongoClient("mongodb://localhost:27017")
mongo_db = mongo_client["yourmongodb"]
mongo_collection = mongo_db["detections"]

# Create folder for saving cropped images
os.makedirs("cropped_images", exist_ok=True)

# Function to save detection in CSV file
def save_to_csv(detections):
    with open("detections.csv", "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(detections)

# Function to save detection in SQL database
def save_to_sql(detections):
    sql = "INSERT INTO detections (label, confidence, timestamp) VALUES (%s, %s, %s)"
    cursor.execute(sql, detections)
    connection.commit()

# Function to save detection in MongoDB
def save_to_mongodb(detections, image):
    data = {
        "label": detections[0],
        "confidence": detections[1],
        "timestamp": datetime.datetime.now(),
        "image": image.tolist()
    }
    mongo_collection.insert_one(data)

# Function to perform object detection and save cropped images
def perform_object_detection(image):
    height, width, channels = image.shape

    # Perform forward pass through the network
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected
                label = classes[class_id]
                detections = [label, confidence, datetime.datetime.now()]

                # Save cropped image
                x, y, w, h = detection[0:4] * np.array([width, height, width, height])
                x = int(x - w / 2)
                y = int(y - h / 2)
                cropped_image = image[y:y+h, x:x+w]
                cv2.imwrite(f"cropped_images/{label}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", cropped_image)

                # Save detection information
                save_to_csv(detections)
                save_to_sql(detections)
                save_to_mongodb(detections, cropped_image)

                # Store detection in Firebase
                db.child("detections").push({
                    "label": detections[0],
                    "confidence": detections[1],
                    "timestamp": str(detections[2])
                })

# Perform object detection on live webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
    perform_object_detection(frame)
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

