import cv2
import imutils
import numpy as np

def find_marker(image):
    gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    gray = cv2.GaussianBlur( gray, (5,5), 0 )
    edged = cv2.Canny( gray, 35, 125 )
    
    cnts = cv2.findContours( edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    cnts = imutils.grab_contours( cnts )
    c = max( cnts, key = cv2.contourArea )
    marker = cv2.minAreaRect( c )
    fl = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
    return fl

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 24
# initialize the known object width, which in this case, my face
# is 8 inches wide
KNOWN_WIDTH = 11
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the face in the image, and initialize
# the focal length
image = cv2.imread("mark-2ft.jpg")
focalLength = find_marker(image)
print( focalLength )