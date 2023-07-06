import cv2
import numpy as np
import csv
import datetime
import pymongo
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from sqlalchemy import create_engine, Table, Column, Integer, String, DateTime, MetaData

# Path to YOLO files
weights_path = 'yolo-tiny.weights'
config_path = 'yolo-tiny.cfg'
class_names_path = 'coco.names'

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# Load class names
with open(class_names_path, 'r') as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Set up MongoDB connection
mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
mongo_db = mongo_client['object_detection']
mongo_collection = mongo_db['detections']

# Set up Firebase credentials
cred = credentials.Certificate('firebase_credentials.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://your-database-url.firebaseio.com/'
})
firebase_ref = db.reference('/detections')

# Set up SQLite connection
engine = create_engine('sqlite:///detections.db')
metadata = MetaData()
detections_table = Table('detections', metadata,
                         Column('id', Integer, primary_key=True),
                         Column('class_name', String),
                         Column('timestamp', DateTime),
                         Column('image_path', String))
metadata.create_all(engine)
conn = engine.connect()

# Helper function for non-maximum suppression
def non_max_suppression(boxes, scores, threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, threshold - 0.1)
    return indices.flatten()

# Perform object detection on an image
def detect_objects(image):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    height, width = image.shape[:2]
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = non_max_suppression(boxes, confidences, 0.3)

    cropped_images = []
    detections = []

    for i in indices:
        x, y, w, h = boxes[i]
        crop = image[y:y + h, x:x + w]
        cropped_images.append(crop)

        class_name = class_names[class_ids[i]]
        timestamp = datetime.datetime.now()
        detection = {
            'class_name': class_name,
            'timestamp': timestamp,
            'image_path': f'detections/{timestamp.strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
        }
        detections.append(detection)

        cv2.imwrite(detection['image_path'], crop)

    return cropped_images, detections

# Perform detection on live webcam feed
def detect_objects_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped_images, detections = detect_objects(frame)

        # Save detections in MongoDB
        mongo_collection.insert_many(detections)

        # Save detections in Firebase
        firebase_ref.update(detections)

        # Save detections in SQLite
        for detection in detections:
            conn.execute(detections_table.insert().values(detection))

        # Log detections in CSV
        with open('detections.csv', 'a') as csv_file:
            writer = csv.writer(csv_file)
            for detection in detections:
                writer.writerow([
                    detection['class_name'],
                    detection['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                    detection['image_path']
                ])

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run object detection on live webcam feed
detect_objects_webcam()

