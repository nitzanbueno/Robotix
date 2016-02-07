import cv2
import urllib2
import numpy as np

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


# This is the point whose HSL values are displayed on screen as well as the point itself as a rectangle
track = (100,100)


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
    # Draw the tracked point on screen
    cv2.rectangle(image, track, (track[0]+2,track[1]+2), (0,255,0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,str(hslimage[track[1], track[0]]),(10,50), font, 1,(255,255,255),2)

    # Filter HSL values of the reflectors
    lower = np.array([30,90,100])
    upper = np.array([80,160,255])
    filtered = cv2.inRange(hslimage, lower, upper)

    # Show results
    cv2.imshow('Image',image)
    cv2.imshow('Filtered',filtered)
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
