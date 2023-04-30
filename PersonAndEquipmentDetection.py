#requires python

import cv2

# Load YOLOv3 pre-trained model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Set threshold for object detection confidence
threshold = 0.6

# Load class names for YOLO model
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load video file
cap = cv2.VideoCapture("video.mp4")

while True:
    # Read frame from video file
    ret, frame = cap.read()

    if ret:
        # Create blob from image frame
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

        # Set input for YOLO model
        net.setInput(blob)

        # Get output layer names from YOLO model
        output_layers = net.getUnconnectedOutLayersNames()

        # Run forward pass through YOLO model
        outputs = net.forward(output_layers)

        # Process YOLO model output
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Check if detected object is person or equipment
                if classes[class_id] in ["person", "equipment"] and confidence > threshold:
                    # Draw bounding box around object
                    box = detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    x, y, w, h = box.astype("int")
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display processed frame
        cv2.imshow("Frame", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release video file and close windows
cap.release()
cv2.destroyAllWindows()
