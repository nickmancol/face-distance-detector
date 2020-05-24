import cv2
import sys
import imutils
import numpy as np

def get_edged( image ):
    gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    gray = cv2.GaussianBlur( gray, (5,5), 0 )
    edged = cv2.Canny( gray, 35, 125 )

    return edged 

def get_marker( allContours ):
    sortedCountours = sorted(allContours, key = cv2.contourArea, reverse = True)[:5]
    
    # select the largest 4 point perimeter as the marker contour
    for c in sortedCountours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx

def largest_marker_focal_len(image_path, ref_distance, ref_width):
    # open reference image
    image = cv2.imread( image_path )
    edged = get_edged( image )
    # detect all contours over the gray image 
    allContours = cv2.findContours( edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    allContours = imutils.grab_contours( allContours )
    markerContour = get_marker( allContours )
    
    # use the marker width to calculate the focal length of the camera
    pixels = (cv2.minAreaRect( markerContour ))[1][0]
    fl = ((pixels * ref_distance) / ref_width)
    
    return fl


def annotate_faces( img, faces, ref_distance, focal_length ):
    for (x, y, w, h) in faces:        
        # draw a rectangle over each detected face
        cv2.rectangle( img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # calculate the approximate distance based on the width in pixels of the face detected
        cm =  (ref_distance * focal_length) /  w
        # put the distance as green text over the face's rectangle
        cv2.putText( img, "%.2fcm" % (cm),
            (x, h -10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1)
        cv2.imshow('img', img )

def faces_dist(classifier, ref_width, ref_distance, focal_length):
    cap = cv2.VideoCapture(0)
    while True:        
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect all the faces on the image
        faces = classifier.detectMultiScale( gray, 1.1, 4)
        annotate_faces( img, faces, ref_distance, focal_length)

        k = cv2.waitKey(30) & 0xff
        # use ESC to terminate
        if k==27:
            break
    # release the camera
    cap.release()

# legal paper size in cm
if len(sys.argv) < 4:
    print("This requires 3 args: reference_img_path ref_distance_cm ref_marker_with_cm")
    exit(1)


classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#classifier = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')

IMG_SRC = sys.argv[1]
REF_DISTANCE = float(sys.argv[2])
REF_WIDTH = float(sys.argv[3])
FL = largest_marker_focal_len( IMG_SRC, REF_DISTANCE, REF_WIDTH )
faces_dist(classifier, REF_WIDTH, REF_WIDTH, FL)