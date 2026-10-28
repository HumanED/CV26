import cv2
import os

OBJECTS = ['']      # List all object names here (undecided)
OBJECT = OBJECTS[0] # Index object that the video is from 

cam = cv2.VideoCapture("/CV26/dataset/video/" + OBJECT) # Indexes into video of selected object

PATH = '/CV26/dataset/frames'

try:
    if not os.path.exists(PATH):
        os.makedirs(PATH)
except OSError:
    print ('Error: Creating directory of data')

curr_frame = 0

while(True):
    ret, frame = cam.read()
    
    if ret:
        # if video, continue creating images
        name = PATH + str(curr_frame) + '.jpg'
        cv2.imwrite(name, frame)
        curr_frame += 1
    else:
        break
    
cam.release()
cv2.destroyAllWindows()