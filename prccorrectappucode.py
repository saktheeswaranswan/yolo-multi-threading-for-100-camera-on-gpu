import cv2
import numpy as np
import datetime
import os
import csv

# Load YOLOv3 model and configuration
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')

# Load class labels
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load thresholds from threshold.csv
thresholds = {}
with open('threshold.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        class_name = row[0]
        threshold = int(row[1])
        thresholds[class_name] = threshold

# Initialize class values
class_values = {}
with open('classes.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        class_name = row[0]
        value = int(row[1])
        class_values[class_name] = value

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change 0 to the desired camera index if using a different camera

# Set video dimensions and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

# Create a folder to store the cropped images
output_folder = 'cropped_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize variables for counting
counters = {class_name: 0 for class_name in classes}
start_time = datetime.datetime.now()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects in the frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(net.getUnconnectedOutLayersNames())

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] in class_values:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)

                # Calculate the coordinates for the bounding box
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Process the detected objects
    for i in indices:
        i = i
        box = boxes[i]
        x, y, width, height = box

        # Crop the detected object
        cropped_object = frame[y:y+height, x:x+width]

        # Get the class name and increment the counter
        class_id = class_ids[i]
        class_name = classes[class_id]
        counters[class_name] += class_values[class_name]

        # Check if the counter exceeds the threshold
        if counters[class_name] >= thresholds[class_name]:
            # Save the cropped image with timestamp and location
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            location = "bus_station_village_district"  # Replace with the actual location
            image_name = f"{class_name}_{timestamp}_{location}.jpg"
            cv2.imwrite(os.path.join(output_folder, image_name), cropped_object)

            # Reset the counter
            counters[class_name] = 0

        # Draw bounding box and label on the frame
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        text = f"{class_name}: {counters[class_name]}"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)
    out.write(frame)

    # Check for key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()


