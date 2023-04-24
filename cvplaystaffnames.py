from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import pygame  # Install pygame library
import os
import glob
import time

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Initialize pygame mixer
pygame.mixer.init()

# Load mp3 files
mp3_files = []
for i in range(len(class_names)):
    mp3_files.append(pygame.mixer.Sound("audio_file_" + str(i) + ".mp3"))

# Specify folder path for input images
folder_path = "/home/josva/Music/yolo-face/cropped_images/"

while True:
    # Get the last updated image in the folder
    list_of_files = glob.glob(folder_path + "*.jpg")
    latest_file = max(list_of_files, key=os.path.getctime)

    # Read the image from the folder
    image = cv2.imread(latest_file)

    # Resize the image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image_array = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_array = (image_array / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Show the detected image with class name
    image_copy = image.copy()
    cv2.putText(image_copy, "Class: " + class_name[2:], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Detected Image", image_copy)

    # Play mp3 file for 5 seconds if specific class name is detected
    if "saktheeswaran" in class_name:
        mp3_files[0].play()
        pygame.time.wait(5000)
    elif "shamini" in class_name:
        mp3_files[1].play()
        pygame.time.wait(5000)
    # Add more elif statements for other class names and corresponding audio files

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

cv2.destroyAllWindows()

