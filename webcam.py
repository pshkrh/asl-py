import numpy as np
import cv2
from fastai.vision import *

# Setting CPU as the default device for inference
defaults.device = torch.device('cpu')

# Change these paths to fit your system
path = '/home/pk/Projects/asl-classification/'
imgpath = '/home/pk/Projects/asl-classification/pred-image.jpg'

# Set the WebCam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Set Font
font = cv2.FONT_HERSHEY_SIMPLEX

# Set Learner Object
learn = load_learner(path)

# Predict the image
def predict():
    img = open_image(imgpath)
    pred_class, pred_idx, outputs = learn.predict(img)
    return pred_class


def webcam_stream():

    # Initialize fps to 0
    fps = 0

    # Initialize prediction string, title and help text
    prediction = ''
    title = 'ASL Classification'
    help_text = 'Press Q to quit'

    # Run till exit key not pressed - this will capture a video from the webcam
    while True:
        # Capture each frame
        ret, frame = cap.read()

        if fps == 18:  # This needs to be adjusted, currently too fast for a user to change sign without garbage characters being added in between
            image = frame[50:300, 50:300]
            cv2.imwrite('pred-image.jpg', image)
            pred = predict()
            temp = str(pred)
            if temp == "space":
                prediction += " "
            elif temp == "del":
                prediction = prediction[:-1]
            elif temp == "nothing":
                prediction += ""
            else:
                prediction += temp
            fps = 0

        fps += 1

        # Display Title
        cv2.putText(frame, title, (180, 30), font,
                    1, (0, 0, 0), 2, cv2.LINE_AA)

        # Display the prediction underneath the region of interest
        cv2.putText(frame, prediction, (50, 400), font,
                    2, (255, 255, 255), 2, cv2.LINE_AA)

        # Display Help text on the bottom
        cv2.putText(frame, help_text, (180, 450), font,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw the region of interest and name the video capture window
        cv2.rectangle(frame, (50, 50), (300, 300), (250, 0, 0), 2)
        cv2.imshow("ASL Prediction", frame)

        # Exit when q key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    webcam_stream()

# Release and destroy the window after exiting the loop
cap.release()
cv2.destroyWindow("ASL Prediction")
