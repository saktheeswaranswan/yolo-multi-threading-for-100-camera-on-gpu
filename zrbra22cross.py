import cv2
import numpy as np

# Load YOLOv3-tiny configuration and weights
net = cv2.dnn.readNet('yolov3-tiny.cfg', 'yolov3-tiny.weights')

# Load COCO class names
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set the minimum probability threshold for detecting objects
conf_threshold = 0.5

# Create a window to display the output stream
cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)

# Create a folder to store cropped images
import os
if not os.path.exists('cropped_images'):
    os.mkdir('cropped_images')

# Open a live video stream
cap = cv2.VideoCapture('/home/josva/Music/helmetbbrc/catt.mp4')

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Create a blob from the input frame and pass it through the YOLOv3-tiny network
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize lists for detected object class IDs, confidences, and bounding boxes
    class_ids = []
    confidences = []
    boxes = []

    # Loop over each of the detected objects and filter out objects that don't meet the minimum confidence threshold
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                # Convert the YOLOv3-tiny output coordinates to the format used by OpenCV
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                # Add the detected object class ID, confidence, and bounding box coordinates to their respective lists
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Apply non-maximum suppression to remove duplicate detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)

    # Loop over each of the detected objects and draw the bounding box and label on the frame
    for i in indices:
        i = i
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        label = f'{classes[class_ids[i]]}: {confidences[i]:.2f}'

        if classes[class_ids[i]] == 'cat':
            # Crop the person in the image
            person_box = [left, top, width, height]
            person_img = frame[person_box[1]:person_box[1]+person_box[3], person_box[0]:person_box[0]+person_box[2]]

            # Save the cropped person image to a file
            filename = f'cropped_images/person_{i}.jpg'
            cv2.imwrite(filename, person_img)

        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow('Object Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
