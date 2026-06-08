import cv2
import os
import numpy as np
from skimage import morphology, measure

img = cv2.imread('bacteria_img/SAMPLE_2.png')
name = "SAMPLE_2"

# Save original
cv2.imwrite(os.path.join('output', f"{name}_original.png"), img)

# Preprocess
img = cv2.resize(img, (500, 500))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# cv2.imshow("Grayscale Image",gray)
# cv2.waitKey(0)

# Detect Petri dish
circles = cv2.HoughCircles(
    gray, cv2.HOUGH_GRADIENT, 1.2, 200,
    param1=50, param2=30, minRadius=200, maxRadius=450
)
if circles is None:
    print(f"No dish found in {name}")

x, y, r = np.uint16(np.around(circles))[0][0]

# Mask dish
mask = np.zeros_like(gray)
cv2.circle(mask, (x, y), r, 255, -1)
masked = cv2.bitwise_and(gray, gray, mask=mask)

cv2.imwrite(os.path.join('output', f"{name}_masked.png"), masked)

cv2.imshow("Masked Image",masked)
cv2.waitKey(0)

# Remove tiny specks
# kernel = np.ones((3,3), np.uint8)
# clean = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel, iterations=2)

# cv2.imshow("Cleaned Image",clean)
# cv2.waitKey(0)

# cv2.imwrite(os.path.join('output', f"{name}_cleaned.png"), clean)

# --- Threshold colonies ---
thresh = cv2.adaptiveThreshold(
    masked, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    51, 5
)

Thresholded = morphology.remove_small_objects(thresh.astype(bool), min_size=50)
Thresholded = (Thresholded * 255).astype(np.uint8)

cv2.imwrite(os.path.join('output', f"{name}_thresholded.png"), Thresholded)

cv2.imshow("Thresholded Image",Thresholded)
cv2.waitKey(0)
