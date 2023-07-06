import cv2
import numpy as np
import datetime
import mysql.connector

# Load YOLOv3-tiny model
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

# Load COCO class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    database="detection_results"
)
cursor = db.cursor()

# Create table if it doesn't exist
cursor.execute("CREATE TABLE IF NOT EXISTS detections (id INT AUTO_INCREMENT PRIMARY KEY, class VARCHAR(255), timestamp DATETIME)")

# Create a folder for saving cropped images
output_folder = "cropped_images"
os.makedirs(output_folder, exist_ok=True)

# Initialize webcam or video file
cap = cv2.VideoCapture(0)  # Change to 0 for webcam or provide video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Process detection results
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])

    # Non-maximum suppression to remove overlapping detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Log results in SQL and save cropped images
    for i in indices:
        i = i[0]
        class_id = class_ids[i]
        label = classes[class_id]
        confidence = confidences[i]
        x, y, w, h = boxes[i]

        # Log detection in SQL database
        timestamp = datetime.datetime.now()
        query = "INSERT INTO detections (class, timestamp) VALUES (%s, %s)"
        values = (label, timestamp)
        cursor.execute(query, values)
        db.commit()

        # Save cropped image
        crop_img = frame[y:y+h, x:x+w]
        image_name = f"{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(output_folder, image_name), crop_img)

        # Display detection on frame
        cv2.rectangle(frame, (x, y),Here's the continuation of the code:

```python
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

