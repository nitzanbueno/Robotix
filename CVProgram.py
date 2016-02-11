import cv2
import urllib2
import numpy as np
import math

# Opens the Axis IP camera with password - can be substituted for VideoCapture
urll = "http://172.17.3.33/axis-cgi/mjpg/video.cgi"
username = "root"
password = "mann"

passman = urllib2.HTTPPasswordMgrWithDefaultRealm()
passman.add_password(None, urll, username, password)
authhandler = urllib2.HTTPBasicAuthHandler(passman)

urlll = urllib2.build_opener(authhandler)

urllib2.install_opener(urlll)

stream = urllib2.urlopen(urll)



def drawPoint(image, point, color, thickness=2):
    x, y = int(point[0]), int(point[1])
    if 0 <= x <= 600 and 0 <= y <= 800:
        cv2.rectangle(image, (x-thickness//2,y-thickness//2), (x+thickness//2,y+thickness//2), color)


def drawText(image, text):
    text = str(text)
    y0, dy = 50, 30
    for i, line in enumerate(text.splitlines()):
        y = y0 + i*dy
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)

def numpyTest(rvecs, tvecs):
    rmat, jacobian = cv2.Rodrigues(rvecs[0])
    tvec = tvecs[0]
    # rtmat = np.concatenate((rmat, tvec), axis=1)
    # rtmat = np.concatenate((rtmat, np.array([[0,0,0,1]])), axis=0)
    # cameramat = np.linalg.inv(rtmat)
    # camR = cameramat[0:3, 0:3]
    # camT = cameramat[0:3, 3]
    camR = rmat.T
    camT = -np.dot(rmat, tvec)
    return camR, camT


# This is the point whose HSL values are displayed on screen as well as the point itself as a rectangle
track = (100,100)
np.set_printoptions(precision=1)
np.set_printoptions(suppress=True)
points = np.array([[[268.5,305,0], [-268.5, 305, 0], [-268.5, 0, 0], [268.5, 0, 0]]]).astype("float32")
ding = True
bites=''
while True:
    # More code to open the image - I just copied this from StackOverflow
    bites+=stream.read(1024)
    a = bites.find('\xff\xd8')
    b = bites.find('\xff\xd9')
    if a!=-1 and b!=-1:
        jpg = bites[a:b+2]
        bites= bites[b+2:]
        image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
    else:
        continue
    # Now image is stored in "image"
    # Rotate image because our camera was sideways
    rows,cols, k = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
    image = cv2.warpAffine(image,M,(cols,rows))
    # Get HSL values (I know this is HLS but it's just convenience)
    hslimage = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Filter HSL values of the reflectors
    lower = np.array([30,40,60])
    upper = np.array([80,160,255])
    filtered = cv2.inRange(hslimage, lower, upper)
    #cv2.imshow("Filtered", filtered)

    # Find corners of U: Find contour and then do a convex hull and approxPolyDP, then we'll have a 4-gon of corners.
    contours, hierarchy = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cont = contours[0]
        maxarea = cv2.contourArea(cont)
        for c in contours:
            area = cv2.contourArea(c)
            if area > maxarea:
                maxarea = area
                cont = c

        convex = cv2.convexHull(cont, clockwise=True)
        epsilon = 0.1*cv2.arcLength(convex, True)
        poly = cv2.approxPolyDP(convex, epsilon, True)
        # Now poly is a 4-gon of corners, and we have to find a relative point.
        # We find the top right point of the U by doing the following check:
        # If the midpoint between the next point (in a counter-clockwise rotation (for some reason it's not clockwise))
        # and the current point is NOT in the contour, it's the valley part of the U (the "hole" of the U), and that
        # only happens in the top-right corner.
        #try:
        if len(poly) == 4:
            trindex = 0
            for i in range(len(poly)):
                p0 = poly[i, 0]
                p1 = poly[(i+1)%len(poly), 0]
                mid = (int((p0[0] + p1[0])/2), int((p0[1] + p1[1])/2))
                dist = cv2.pointPolygonTest(cont,mid,False)
                if dist < 0:
                    trindex = i
                    break
            cv2.drawContours(image, [poly], -1, (255, 255, 255), thickness=4)
            drawPoint(image, poly[trindex, 0], (255, 0, 0), thickness=10)
            drawPoint(image, poly[(trindex+1)%4, 0], (0,255,255), thickness=10)
            # Now comes the problematic part - the camera calibration (or position location).
            corners = np.array([[poly[(i+trindex)%len(poly),0] for i in range(len(poly))]]).astype("float32")
            retval, matrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(points, corners, (600,800), None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
            tvec = tvecs[0]
            rvec = rvecs[0]
            camR, camT = numpyTest(rvecs, tvecs)
            dist = np.linalg.norm(camT)
            camRVec, jacobian = cv2.Rodrigues(camR)
            drawText(image, camRVec*10)
            origin = np.array([[0],[2],[0]])
            rmat, jacobian = cv2.Rodrigues(rvec)
            center, jacobian = cv2.projectPoints(np.float32([[0,0,0]]), rvec, tvec, matrix, distCoeffs)
            drawPoint(image, center[0,0], (0, 255, 255), 10)
            p = (sum([i[0]**2 for i in tvec]))**(0.5)
            p = int(p)
            drawPoint(image, tvec, (255, 0, 0))
            drawPoint(image, np.add(np.dot(rmat, origin), tvec), (255, 255, 255))

    # Show results
    #cv2.putText(image,str(hslimage[track[1], track[0]]),(10,50), font, 1,(255,255,255),2)
    #drawPoint(image, track, (0,255,0))
    cv2.imshow('Image',image)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        exit(0)

    # This code moves the tracked point with the WASD keys.
    if k == ord("d"):
        track = (track[0] + 10, track[1])
    if k == ord("a"):
        track = (track[0] - 10, track[1])
    if k == ord("s"):
        track = (track[0], track[1] + 10)
    if k == ord("w"):
        track = (track[0], track[1] - 10)
