import cv2
import numpy as np
import datetime
import csv

# Load YOLO model with tiny weights and config
net = cv2.dnn.readNet("yolo-tiny.weights", "yolo-tiny.cfg")

# Load COCO dataset names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set confidence threshold and non-maximum suppression threshold
conf_threshold = 0.5
nms_threshold = 0.4

# Load CSV file with times and classes
with open("times_and_classes.csv", "r") as f:
    reader = csv.reader(f)
    times_and_classes = list(reader)

# Get system time
now = datetime.datetime.now()
current_time = now.strftime("%H:%M")

# Set crop images folder
crop_folder = "crop_images/"

# Initialize video stream
cap = cv2.VideoCapture(0)

# Start video stream
while True:
    ret, frame = cap.read()
    if ret:
        # Resize frame
        frame_resized = cv2.resize(frame, (416, 416))
        # Convert frame to blob
        blob = cv2.dnn.blobFromImage(frame_resized, 1 / 255, (416, 416), swapRB=True, crop=False)
        # Set input to YOLO model
        net.setInput(blob)
        # Get output layers
        output_layers = net.getUnconnectedOutLayersNames()
        # Forward pass
        layer_outputs = net.forward(output_layers)
        # Apply non-maximum suppression
        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x, center_y, w, h = (detection[0:4] * np.array([416, 416, 416, 416])).astype(int)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        # Check if any class matches with the one in the CSV file and the time is correct
        for i in indices:
            i = i[0]
            if classes[class_ids[i]] in [row[1] for row in times_and_classes if row[0] == current_time]:
                # Save cropped image
                x, y, w, h = boxes[i]
                crop_img = frame[y:y+h, x:x+w]
                crop_filename = crop_folder + classes[class_ids[i]] + "_" + str(now.timestamp()) + ".jpg"
                cv2.imwrite(crop_filename, crop_img)
        # Display frame
        cv2.imshow("Live Stream", frame)
        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break
    else:
        break

# Release video stream and destroy windows
cap.release()
cv2.destroyAllWindows()

