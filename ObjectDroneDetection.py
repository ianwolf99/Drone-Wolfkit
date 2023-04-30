
#requires python
import cv2

# Load the YOLOv4 object detection model
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

# Define the classes to detect
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set the input size for the network
input_size = (416, 416)

# Start the video capture from the drone
cap = cv2.VideoCapture('rtsp://drone_ip_address/live')

while True:
    # Read the frame from the video feed
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame to the input size for the network
    resized_frame = cv2.resize(frame, input_size)
    
    # Normalize the image
    normalized_frame = cv2.dnn.blobFromImage(resized_frame, 1/255, input_size, (0,0,0), True, crop=False)
    
    # Set the input for the network
    net.setInput(normalized_frame)
    
    # Run forward pass and get the detections
    detections = net.forward()
    
    # Loop through the detections
    for detection in detections:
        # Get the class ID and confidence score
        class_id = np.argmax(detection[5:])
        confidence = detection[class_id + 5]
        
        # If the confidence score is above the threshold, detect the object
        if confidence > 0.5:
            # Get the object's position and size
            box = detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, w, h) = box.astype('int')
            
            # Draw a bounding box around the object and label it
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, classes[class_id], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Live Feed', frame)
    
    # Wait for a key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
