import cv2

image = cv2.imread('bacteria_img/IMG_6227.jpg')

h,w = image.shape[:2]
print(f"heigh: {h}, w: {w}")

(B,G,R) = image[100,100]
print(f"R:{R}, G:{G}, B:{B}")


b = image[100,100,2]
print(b)

print("Display ROI: Region of Interest")
roi = image[100:500,200:700]
#cv2.imshow("ROI", roi)
#cv2.waitKey(0)

print("resize image")
resize = cv2.resize(image,(500,500))
#cv2.imshow("Resized Image",resize)
#cv2.waitKey(0)

ratio = 800/ w

dim= (800, int(h*ratio))

resize_aspect = cv2.resize(image,dim)
#cv2.imshow("Resized image",resize_aspect)
#cv2.waitKey(0)

print("Drawing a rectangle")
output = image.copy()
rectangle = cv2.rectangle(output,(1500,900),
                          (600,400),(255,0,0),2)
#cv2.imshow("Image with Recatnalge",rectangle)
#cv2.waitKey(0)

output = image.copy()
text = cv2.putText(output,"OpenCV Demo",(500,550),
                   cv2.FONT_HERSHEY_SIMPLEX,4,(255,0,0),2)
cv2.imshow("Image with text",text)
cv2.waitKey(0)