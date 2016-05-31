import cv2
import os
import sys
from string import Template

# first argument is the haarcascades path
face_cascade_path = sys.argv[1]
face_cascade = cv2.CascadeClassifier(os.path.expanduser(face_cascade_path))

eyes_cascade_path = sys.argv[2]
eyes_cascade = cv2.CascadeClassifier(os.path.expanduser(eyes_cascade_path))

smile_cascade_path = sys.argv[3]
smile_cascade = cv2.CascadeClassifier(os.path.expanduser(smile_cascade_path))

scale_factor = 1.1
min_neighbors = 3
min_size = (100, 100)
flags = cv2.cv.CV_HAAR_SCALE_IMAGE

for infname in sys.argv[3:]:
    image_path = os.path.expanduser(infname)
    image = cv2.imread(image_path)
    faces = face_cascade.detectMultiScale(image, scaleFactor = scale_factor, minNeighbors = min_neighbors, minSize = min_size, flags = flags)
    print len(faces)
    for( x, y, w, h ) in faces:
        face = image[y:y+h, x:x+w]
        
        eyes = eyes_cascade.detectMultiScale(face[0:h/2,0:w], scaleFactor = scale_factor, minNeighbors = min_neighbors, minSize = (20, 20), flags = flags)
        print "eyes: %d\n" % len(eyes)
        for( x2, y2, w2, h2 ) in eyes:
            cv2.rectangle(image, (x+x2, y+y2), (x+x2 + w2, y+y2 + h2), (255, 0, 255), 2)

        smiles = smile_cascade.detectMultiScale(face[h/2:h,0:w], scaleFactor = scale_factor, minNeighbors = 20, minSize = (100, 50), flags = flags)
        print "smiles: %d\n" % len(smiles)
        for( x3, y3, w3, h3 ) in smiles:
            cv2.rectangle(image, (x+x3, y+y3 + h/2), (x+x3 + w3, y+y3 + h3 + h/2), (0, 255, 255), 2)
        #        if (len(eyes) == 2 and len(smiles) == 1):
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    outfname = "./%s.faces.jpg" % os.path.basename(infname)
    cv2.imwrite(os.path.expanduser(outfname), image)
