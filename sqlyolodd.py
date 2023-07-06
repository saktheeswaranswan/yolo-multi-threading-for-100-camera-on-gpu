import cv2
import numpy as np
import time
import mysql.connector

# Load the YOLOv3-tiny model
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')

# Load the COCO class labels
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the SQL database connection
mydb = mysql.connector.connect(
  host="localhost",
  user="your_username",
  password="your_password",
  database="object_detection"
)
mycursor = mydb.cursor()

# Create a table for storing the detection results
mycursor.execute("CREATE TABLE IF NOT EXISTS detections (timestamp DATETIME, object_class VARCHAR(255))")

# Create a folder for storing cropped images
output_folder = 'output_images/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to log the detection results in the SQL database
def log_detection(timestamp, object_class):
    sql = "INSERT INTO detections (timestamp, object_class) VALUES (%s, %s)"
    val = (timestamp, object_class)
    mycursor.execute(sql, val)
    mydb.commit()

# Function to crop the image and save it to the output folder
def save_cropped_image(image, box, output_path):
    x, y, w, h = box
    cropped_image = image[y:y+h, x:x+w]
    cv2.imwrite(output_path, cropped_image)

# Function to perform object detection on the live video stream
def detect_objects():
    cap = cv2.VideoCapture(0)  # Use the first webcam device (change the parameter if you have multiple cameras)

    while True:
        ret, frame = cap.read()

        # Resize the frame to the input size required by YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

        # Set the input blob for the network
        net.setInput(blob)

        # Forward pass through the network
        start_time = time.time()
        layer_outputs = net.forward(['yolo_82', 'yolo_94'])

        # Get the bounding boxes, class labels, and confidences
        boxes = []
        confidences = []
        class_ids = []
        H, W = frame.shape[:2]

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype('int')

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression to remove overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5(adjust the threshold as needed)

        # Process the detected objects
        if len(indices) > 0:
            for i in indices.flatten():
                class_id = class_ids[i]
                object_class = classes[class_id]
                confidence = confidences[i]
                box = boxes[i]

                # Log the detection in the SQL database
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                log_detection(timestamp, object_class)

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
                label = f'{object_class}: {confidence:.2f}'
                cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            # Cat not detected, crop the image to focus on the person
            save_cropped_image(frame, (0, 0, W, H), os.path.join(output_folder, f'person_{timestamp}.jpg'))

        # Display the frame
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

# Run the object detection on the live video stream
detect_objects()

