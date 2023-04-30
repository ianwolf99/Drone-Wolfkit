
import cv2

# load the image
image = cv2.imread('drone_image.jpg')

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# apply adaptive thresholding to create a binary image
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# apply morphological operations to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# apply connected component analysis to identify flooded areas
_, labels = cv2.connectedComponents(morph)

# count the number of connected components (excluding the background label 0)
num_labels = len(set(labels.flatten())) - 1

# display the flood detection results
print("Number of flooded areas detected: {}".format(num_labels))

cv2.imshow('Flood Detection', morph)
cv2.waitKey(0)
cv2.destroyAllWindows()
