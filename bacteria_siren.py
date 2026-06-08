import cv2 as cv
import os
SAMPLE_BASE_NAME = "SAMPLE_"

# Get the last image name
# Set the next image sample with the next number
def set_file_name():
    folder_path ="bacteria_img"
    files = os.listdir(folder_path)
    file_name = SAMPLE_BASE_NAME
    if len(files) == 0:
        file_name = file_name +"0" + ".png"
        return file_name
    largest_value = -1
    for file in files:  
        print(file)
        tokens = file.split("_")
        dot_index = tokens[1].find('.')
        curr_file_number = tokens[1][:dot_index]
        curr_file_number = int(curr_file_number)
        if curr_file_number > largest_value:
            largest_value = curr_file_number
        
    next_file_number = int(largest_value) + 1
    file_name = file_name +str(next_file_number)+".png"
    print(file_name)
    return file_name

# Image Capturing 
cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()
    frame = cv.flip(frame,1)
    cv.imshow("Bacteria Capture",frame)
    key = cv.waitKey(1)

    if key == 13: # Enter Key
        file_name = set_file_name()
        cv.imwrite("bacteria_img/"+file_name,frame)
        cam.release()
        break
    elif key == 27: # Escape key
        cam.release()
        break

cv.destroyAllWindows()  