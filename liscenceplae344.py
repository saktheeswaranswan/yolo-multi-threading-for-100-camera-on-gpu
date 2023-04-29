#https://www.whatfontis.com/blog/number-plates-colors-in-india/
import cv2
import numpy as np
import random
import os

# Define the state codes and corresponding fonts
state_codes = {
    "AN": "ANPRNum.ttf",
    "AP": "AP_NumberPlate.ttf",
    "AR": "AR Number Plate.ttf",
    "AS": "AS number plate.ttf",
    "BR": "BR Number Plate.ttf",
    "CH": "Ch number plate.ttf",
    "DD": "DD Number Plate.ttf",
    "DL": "India Number Plate.ttf",
    "DN": "DN Number Plate.ttf",
    "GA": "GA Number Plate.ttf",
    "GJ": "GJ Number Plate.ttf",
    "HR": "HR Number Plate.ttf",
    "HP": "HP Number Plate.ttf",
    "JH": "JH Number Plate.ttf",
    "JK": "JK Number Plate.ttf",
    "KA": "KA Number Plate.ttf",
    "KL": "KL number plate.ttf",
    "LD": "LD Number Plate.ttf",
    "MH": "MH Number Plate.ttf",
    "ML": "ML Number Plate.ttf",
    "MN": "MN Number Plate.ttf",
    "MP": "MP Number Plate.ttf",
    "MZ": "MZ Number Plate.ttf",
    "NL": "NL Number Plate.ttf",
    "OD": "OD Number Plate.ttf",
    "PB": "Pb Number Plate.ttf",
    "PY": "PY Number Plate.ttf",
    "RJ": "RJ Number Plate.ttf",
    "SK": "SK Number Plate.ttf",
    "TN": "TN number plate.ttf",
    "TR": "TR Number Plate.ttf",
    "TS": "TS Number Plate.ttf",
    "UK": "UK Number Plate.ttf",
    "UP": "UP Number Plate.ttf",
    "WB": "WB Number Plate.ttf"
}

# Create the output folder if it doesn't exist
if not os.path.exists("output"):
    os.mkdir("output")

# Loop through each state and generate random license plates
for state in state_codes:
    # Load the state font
    font_path = os.path.join("fonts", state_codes[state])
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Generate 10 random license plates
    for i in range(10):
        # Generate a random license plate number
        number = random.randint(1000, 9999)
        alpha = chr(random.randint(65, 90))
        license_plate = alpha + str(number)

        # Create a blank image
        img = np.zeros((100, 300, 3), np.uint8)

        # Write the license plate number on the image
        cv2.putText(img, license_plate, (10, 70), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw the YOLO annotation box around the license plate number
        x, y, w, h = cv2.boundingRect(np.array([[[10, 70]], [[10, 150]], [[280, 150]], [[280, 70]]], dtype=np.int32))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the image and the corresponding label
        file_name = state + "_" + str(i+1)
        img_path = os.path.join("output", file_name + ".jpg")
        label_path = os.path.join("output", file_name + ".txt")

        cv2.imwrite(img_path, img)

        # Calculate
        x_center = (x + w/2) / img.shape[1]
        y_center = (y + h/2) / img.shape[0]
        width = w / img.shape[1]
        height = h / img.shape[0]

        # Write the label file in YOLO format
        with open(label_path, "w") as f:
             f.write("0 " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height))


