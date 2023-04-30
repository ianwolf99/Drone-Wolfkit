
import cv2
import numpy as np

# Read the video feed from the drone
cap = cv2.VideoCapture("drone_feed.mp4")

# Define the threshold values for flood detection
lower_threshold = np.array([0, 127, 0])
upper_threshold = np.array([255, 255, 255])

# Loop over the frames in the video feed
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to detect the flooded areas
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Apply a mask to filter out non-flooded areas
        mask = cv2.inRange(frame, lower_threshold, upper_threshold)
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        
        # Find contours of flooded areas
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes around the detected flooded areas
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
        # Display the resulting frame
        cv2.imshow('Flood Detection',frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
