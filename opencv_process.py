import cv2
import os
import numpy as np
from skimage import morphology

img = cv2.imread('bacteria_img/SAMPLE_13.jpg')
name = "SAMPLE_13"

# Save original
cv2.imwrite(os.path.join('output', f"{name}_original.jpg"), img)


# Grayscale image
def Grayscale_image(image):
    h,w = img.shape[:2]
    new_width = 500
    ratio = new_width/w
    new_height = int(h*ratio)
    image = cv2.resize(image, (new_width,new_height))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    return gray

gray = Grayscale_image(img)

cv2.imshow("Grayscale Image",gray)
cv2.waitKey(0)

# Mask dish
def detect_dish(image):
    # Detect Petri dish using Hough Circle Transform
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp = 1.2,
        minDist = image.shape[0]//2,
        param1=50, param2=25, minRadius=100, maxRadius=image.shape[0]//2
    )

    if circles is None:
        print(f"No dish found in {name}")

    x, y, r = np.uint16(np.around(circles))[0][0]
    return (x,y,r)

x,y,r = detect_dish(gray)

def apply_mask(image,x,y,r):

    # Create mask with detected circle and apply to image
    mask = np.zeros_like(image)
    cv2.circle(mask, (x, y), r, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)

    cv2.imwrite(os.path.join('output', f"{name}_masked.png"), masked)

    return masked

masked = apply_mask(gray,x,y,r)

cv2.imshow("Masked Image",masked)
cv2.waitKey(0)

# Remove tiny specks
def clean_image(image):
    blur = cv2.bilateralFilter(image, 9, 50, 50)  # preserves edges
    kernel = np.ones((2,2), np.uint8)
    clean = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel, iterations=1)

    # kernel = np.ones((2,2), np.uint8)
    # clean = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel, iterations=2)

    cv2.imwrite(os.path.join('output', f"{name}_cleaned.png"), clean)

    return clean

clean = clean_image(masked)
cv2.imshow("Cleaned Image",clean)
cv2.waitKey(0)

# Threshold colonies
def threshold_image(image):
    # Apply adaptive thresholding to segment colonies
    thresh = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 2
    )

    thresholded = morphology.remove_small_objects(thresh.astype(bool), min_size=50)
    thresholded = (thresholded*255).astype(np.uint8)

    cv2.imwrite(os.path.join('output', f"{name}_thresholded.png"), thresholded)

    return thresholded

thresholded = threshold_image(clean)

cv2.imshow("Thresholded Image",thresholded)
cv2.waitKey(0)

x_t,y_t,r_t = detect_dish(thresholded)
Thresholded_inner = apply_mask(thresholded,x_t,y_t,r_t-45)
cv2.imshow("Inner Thresholded Image",Thresholded_inner)
cv2.waitKey(0)

"""
Thresholded_inner
↓
Remove large artifacts like glare
↓
Distance transform
↓
Sure foreground
↓
Sure background
↓
Watershed markers
↓
Watershed segmentation
↓
Count separated colonies
"""


