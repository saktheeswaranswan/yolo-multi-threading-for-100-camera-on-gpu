import cv2
import numpy as np
import os
import time

# Load YOLOv3 Tiny model and COCO class names
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set minimum confidence level and non-maximum suppression threshold
conf_threshold = 0.5
nms_threshold = 0.4

# Create output folder
output_folder = 'output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Start live video capture
cap = cv2.VideoCapture('/home/josva/Music/helmetbbrc/carrts.mp4')
while True:
    ret, frame = cap.read()
    
    # Prepare image for YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get detections
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Initialize lists to store bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                # Object detected
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Add bounding box, confidence, and class ID to respective lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-maximum suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Check if cat is detected, if not crop person and save image
    cat_detected = False
    for i in indices:
        if classes[class_ids[i]] == 'cat':
            cat_detected = True
            break
    if not cat_detected:
        for i in indices:
            if classes[class_ids[i]] == 'person':
                x, y, w, h = boxes[i]
                crop_img = frame[y:y+h, x:x+w]
                img_name = 'person_' + str(time.time()) + '.jpg'
                cv2.imwrite(os.path.join(output_folder, img_name), crop_img)

    # Draw bounding boxes on original frame
    for i in indices:
        if classes[class_ids[i]] == 'cat':
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, classes[class_ids[i]], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, classes[class_ids[i]], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Show output frame
    cv2.imshow('Object Detection', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()

