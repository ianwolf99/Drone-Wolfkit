
import cv2
import numpy as np
import time

# Load video feed from drone
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold image to find bright areas
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Find contours in image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours and draw rectangle around fires
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display video feed
    cv2.imshow('frame', frame)

    # Wait for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture and close window
cap.release()
cv2.destroyAllWindows()
