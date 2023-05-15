import cv2
import numpy as np
import os

# Load YOLO model
net = cv2.dnn.readNet("face-yolov3-tiny_41000.weights", "face-yolov3-tiny.cfg")

# Load COCO names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize counter for naming cropped images
count = 1

while True:
    # Read frame from video
    ret, frame = cap.read()

    # Get image dimensions
    height, width, channels = frame.shape

    # Create blob from image
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    # Set input blob for the network
    net.setInput(blob)

    # Forward pass through the network
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Get class IDs, confidence scores, and bounding boxes
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
                x = center_x - w // 2
                y = center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Create directory to store cropped images
    if not os.path.exists("cropped_images"):
        os.makedirs("cropped_images")

    # Crop and save images
    for i in indices:
        i = i
        box = boxes[i]
        x, y, w, h = box
        cropped_img = frame[y:y+h, x:x+w]
        if not cropped_img.any():
            continue
        cv2.imwrite(f"cropped_images/{count}.jpg", cropped_img)
        count += 1

    # Draw bounding boxes and labels
    for i in indices:
        i = i
        box = boxes[i]
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, classes[class_ids[i]], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display image with detections
    cv2.imshow("Live Video", frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and destroy


# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()

