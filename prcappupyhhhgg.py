import cv2
import csv
import time
import datetime
import os
import numpy as np

# Paths to the configuration files and weights
weights_path = "yolov3-tiny.weights"
config_path = "yolov3-tiny.cfg"
names_path = "coco.names"

# Load the COCO class names
with open(names_path, "r") as f:
    class_names = f.read().splitlines()

# Load the classes.csv and threshold.csv files
classes_file = "classes.csv"
threshold_file = "threshold.csv"

# Load the threshold values from threshold.csv
threshold_values = {}
with open(threshold_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        class_name = row[0]
        threshold = int(row[1])
        threshold_values[class_name] = threshold

# Create a folder to store the cropped images
output_folder = "cropped_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Capture video using OpenCV
cap = cv2.VideoCapture(0)

# Get the current time
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Initialize a dictionary to store the counts for each class
class_counts = {}

# Tabulate the detected objects and threshold reached in separate CSV files
detected_objects_file = "detected_objects.csv"
threshold_reached_file = "threshold_reached.csv"

# Create the CSV files if they don't exist
if not os.path.exists(detected_objects_file):
    with open(detected_objects_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Class", "Count"])

if not os.path.exists(threshold_reached_file):
    with open(threshold_reached_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Class", "Threshold Reached"])

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Get the current time in seconds
    current_seconds = int(time.time())

    # Load the classes and their corresponding values from classes.csv
    classes = {}
    with open(classes_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            class_name = row[0]
            value = int(row[1])
            classes[class_name] = value + current_seconds

    # Perform object detection on the frame
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Initialize lists to store detected objects and their classes
    class_ids = []
    confidences = []
    boxes = []

    # Iterate over each detection from the output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

                            # Filter detections with a minimum confidence threshold of 0.5
            if confidence > 0.5:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    # Calculate the top-left corner of the bounding box
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    # Store the class, confidence, and bounding box coordinates
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, width, height])    

      # Apply non-maximum suppression to remove redundant overlapping boxes
      indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

      for i in range(len(boxes)):
         if i in indices:
        class_id = class_ids[i]
        class_name = class_names[class_id]
        confidence = confidences[i]
        box = boxes[i]

        # Check if the class is in the classes.csv and threshold.csv
        if class_name in classes and class_name in threshold_values:
            current_value = classes[class_name]
            threshold_value = threshold_values[class_name]

            # Compare the current value with the threshold value
            if current_value >= threshold_value:
                # Crop the image
                cropped_img = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

                # Check if the cropped image is valid
                if cropped_img.size != 0:
                    # Update the class count
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    count = class_counts[class_name]

                    # Check if the count exceeds the threshold
                    if count >= threshold_value:
                        # Save the cropped image with the timestamp
                        output_class_folder = os.path.join(output_folder, class_name)
                        if not os.path.exists(output_class_folder):
                            os.makedirs(output_class_folder)

                        image_count = len(os.listdir(output_class_folder))
                        cropped_img_name = f"{class_name}_{image_count + 1}.jpg"
                        cv2.imwrite(os.path.join(output_class_folder, cropped_img_name), cropped_img)

                        # Display the cropped image with the count
                        cv2.putText(frame, f"Count: {count}", (box[0], box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        # Append the data to the detected_objects.csv file
                        with open(detected_objects_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([current_time, class_name, count])

                        # Check if the count equals or exceeds the threshold
                        if count == threshold_value:
                            # Append the data to the threshold_reached.csv file
                            with open(threshold_reached_file, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([current_time, class_name, threshold_value])

# Display the resulting frame
cv2.imshow("Object Detection", frame)

# Break the loop if 'q' is pressed
if cv2.waitKey(1) == ord("q"):
    break

# Release the video
cap.release()
cv2.destroyAllWindows()


# Display the resulting frame
cv2.imshow("Object Detection", frame)

# Break the loop if 'q' is pressed
if cv2.waitKey(1) == ord("q"):
    break

# Release the video
cap.release()
cv2.destroyAllWindows()

