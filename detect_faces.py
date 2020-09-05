import cv2
import sys
import math
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

def reference_pixels(image_path, ref_distance, ref_width):
    # open reference image
    image = cv2.imread( image_path )
    edged = get_edged( image )
    # detect all contours over the gray image 
    allContours = cv2.findContours( edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    allContours = imutils.grab_contours( allContours )
    markerContour = get_marker( allContours )
    
    # use the marker width to calculate the focal length of the camera
    pixels = (cv2.minAreaRect( markerContour ))[1][0]
    
    return pixels


def annotate_faces( img, faces, ratio_px_cm ):
    points = []
    for (x, y, w, h) in faces:        
        center = (x+(int(w/2)), y+(int(h/2)))
        cv2.circle( img, center, 2, (0,255,0),2)
        for p in points:
            ed = euclidean_dist( p, center ) * ratio_px_cm
            color = (0,255,0) 
            if ed < MIN_DIST:
                color = (0,0,255)
            # draw a rectangle over each detected face
            cv2.rectangle( img, (x, y), (x+w, y+h), color, 2)            
            # put the distance as text over the face's rectangle
            cv2.putText( img, "%scm" % (ed),
                (x, h -10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
            # draw a line between the faces detected
            cv2.line( img, center, p, color, 5)
        points.append( center )

    cv2.imshow('img', img )


def euclidean_dist( pA, pB ):
    return math.sqrt( (pA[0]-pB[0])**2 + (pA[1]-pB[1])**2 )

def faces_dist(classifier, ref_width, ref_pix):
    ratio_px_cm = ref_width / ref_pix
    cap = cv2.VideoCapture(0)
    while True:        
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect all the faces on the image
        faces = classifier.detectMultiScale( gray, 1.1, 4)
        annotate_faces( img, faces, ratio_px_cm)

        k = cv2.waitKey(30) & 0xff
        # use ESC to terminate
        if k==27:
            break
    # release the camera
    cap.release()

# legal paper size in cm
if len(sys.argv) < 4:
    print("This requires 3 args: reference_img_path ref_distance_cm ref_marker_witdth_cm")
    exit(1)


classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#classifier = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')

MIN_DIST = 60
IMG_SRC = sys.argv[1]
REF_DISTANCE = float(sys.argv[2])
REF_WIDTH = float(sys.argv[3])
REF_PIX = reference_pixels( IMG_SRC, REF_DISTANCE, REF_WIDTH )
faces_dist(classifier, REF_WIDTH, REF_PIX)