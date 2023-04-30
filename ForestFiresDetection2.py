import cv2
import numpy as np

# Define temperature threshold for fire detection
TEMP_THRESHOLD = 60  # in degrees Celsius

# Initialize the video capture device (use 0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame from the video feed
    ret, frame = cap.read()

    # Convert the frame to grayscale for easier processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to create a binary mask of high-temperature areas
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours of high-temperature areas
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around the high-temperature areas and classify as fire if temperature exceeds threshold
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        temperature = np.mean(gray[y:y+h, x:x+w])
        if temperature > TEMP_THRESHOLD:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Fire Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the processed frame with fire detection results
    cv2.imshow("Fire Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
