import cv2
import os
import numpy as np
import copy
from skimage import morphology


class ColonyCounter:
    

    def __init__(self,img_path):
        self.img_name = img_path
        self.img = None
        self.img_gray_scale = None
        self.img_masked =  None
        self.img_clean =  None
        self.img_thresholded = None
        self.img_thresholded_inner = None
        self.img_contour = None

        self.x = None
        self.y = None
        self.r = None

        self.contours = None
        self.hierarchy = None
        self.filtered_contours = []


    def load_image(self):
        print(self.img_name)
        self.img = cv2.imread(self.img_name)
        

    def preprocess_image(self, new_width=500):
        h,w = self.img.shape[:2]
        ratio = new_width/w
        new_height = int(h*ratio)
        self.img = cv2.resize(self.img, (new_width,new_height))
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        
        self.img_gray_scale = gray
        

    def detect_dish(self):
        # Detect Petri dish using Hough Circle Transform
        circles = cv2.HoughCircles(
            self.img_gray_scale,
            cv2.HOUGH_GRADIENT,
            dp = 1.2,
            minDist = self.img_gray_scale.shape[0]//2,
            param1=50, param2=25, minRadius=100, maxRadius=self.img_gray_scale.shape[0]//2
        )

        if circles is None:
            print(f"No dish found in {name}")

        self.x, self.y, self.r = np.uint16(np.around(circles))[0][0]
        return self.x, self.y, self.r

    def apply_mask(self, image, radius_margin=0):
         # Create mask with detected circle and apply to image
        mask = np.zeros_like(image)
        
        radius = int(self.r) - radius_margin
        cv2.circle(mask, (int(self.x), int(self.y)), radius, 255, -1)
        
        masked = cv2.bitwise_and(image, image, mask=mask)

        return masked
    
    def mask_dish(self):
        self.img_masked = self.apply_mask(
            self.img_gray_scale,
            radius_margin=0
        )

    
    def clean_image(self):
        blur = cv2.bilateralFilter(self.img_masked, 9, 50, 50)  # preserves edges
        kernel = np.ones((2,2), np.uint8)
        clean = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel, iterations=1)

        self.img_clean = clean


    def threshold_image(self):
        # Apply adaptive thresholding to segment colonies
        thresh = cv2.adaptiveThreshold(
            self.img_clean, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31, 2
        )

        thresholded = morphology.remove_small_objects(thresh.astype(bool), min_size=50)
        thresholded = (thresholded*255).astype(np.uint8)

        self.img_thresholded = thresholded

    def apply_inner_mask_to_threshold(self,radius_margin=45):
        self.img_thresholded_inner =  self.apply_mask(
            self.img_thresholded,
            radius_margin=radius_margin
        ) 
    
    def find_contours(self):
        self.contours, self.hierarchy = cv2.findContours(
            self.img_thresholded_inner,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        return self.contours, self.hierarchy
    
    
    def filter_colonies(self, min_area=25, max_area=30000,
                    min_circularity=0.05, max_aspect_ratio=2):

        filtered_cnts = []

        if self.hierarchy is None:
            return filtered_cnts

        for i, c in enumerate(self.contours):
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)

            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)

            x, y, w, h = cv2.boundingRect(c)

            if w == 0 or h == 0:
                continue

            aspect_ratio = max(w, h) / min(w, h)

            if (
                min_area < area < max_area and
                circularity > min_circularity and
                aspect_ratio < max_aspect_ratio
            ):
                self.filtered_contours.append(c)

        return self.filtered_contours
    
    def draw_contours(self):
        
        contours_img = cv2.cvtColor(self.img_thresholded_inner.copy(), cv2.COLOR_GRAY2BGR)

        PINK_COLOR = (255, 0, 255)

        cv2.drawContours(
            contours_img,
            self.contours,
            contourIdx=-1,
            color=PINK_COLOR,
            thickness=2
        )

        self.img_contour = contours_img

        return self.img_contour
    
    def count_colonies(self):
        return len(self.filtered_contours)

    def show_results(self,display_name,img):
        cv2.imshow(display_name,img)
        cv2.waitKey(0)

    def run_pipeline(self):
        self.load_image()
        self.preprocess_image(new_width=500)
        self.show_results("Pre Process",self.img)
        
        self.detect_dish()
        self.mask_dish()
        self.show_results("Masked",self.img_masked)
        
        self.clean_image()
        self.show_results("Clean",self.img_clean)

        self.threshold_image()
        self.show_results("Thresholded",self.img_thresholded)

        self.apply_inner_mask_to_threshold(radius_margin=45)
        self.show_results("Inner Thresholded",self.img_thresholded_inner)

        self.find_contours()
        self.filter_colonies()
        
        result_img = self.draw_contours()
        self.show_results("Contours",self.img_contour)
        
        colonies_count = self.count_colonies()
        
        return colonies_count, result_img
    
counter = ColonyCounter('output/SAMPLE_13.jpg')
#counter.show_results("First Pic",counter.img)

count, result_img = counter.run_pipeline()
print(f"Colony Count: {count}")
