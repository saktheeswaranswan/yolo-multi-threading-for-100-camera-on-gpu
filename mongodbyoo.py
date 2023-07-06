import cv2
import numpy as np
import time
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['detection_results']
collection = db['detections']

# Load YOLOv3-tiny
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Define output folder for saving images
output_folder = 'detected_images/'

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path

while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width, _ = frame.shape

    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Log detections in MongoDB
    detections = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            detection = {
                'timestamp': time.time(),
                'label': label,
                'confidence': confidence,
                'image_path': f'{output_folder}detected_{time.time()}.jpg'
            }
            detections.append(detection)
            collection.insert_one(detection)

            # Crop and save detected image
            detected_image = frame[y:y + h, x:x + w]
            cv2.imwrite(detection['image_path'], detected_image)

            # Display detection on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame with detections
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture andsave any remaining detections in MongoDB.
cap.release()
cv2.destroyAllWindows()
client.close()

To install MongoDB, follow the steps below:

1. Visit the MongoDB download page: https://www.mongodb.com/try/download/community
2. Select the appropriate version and operating system for your machine.
3. Download the installer and run it.
4. Follow the installation instructions provided by the installer.
5. Once the installation is complete, MongoDB will be installed on your system.
6. You can start the MongoDB server by running the `mongod` command in your terminal or command prompt.
7. To interact with MongoDB, you can use the MongoDB Shell by running the `mongo` command in your terminal or command prompt.

Note: Make sure to install the PyMongo package (`pip install pymongo`) before running the code to connect and interact with MongoDB using Python.

