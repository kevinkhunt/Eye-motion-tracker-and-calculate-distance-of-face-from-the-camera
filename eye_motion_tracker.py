import cv2
import dlib
import numpy as np
import math
from collections import deque
from math import hypot
import time

# we have used here two models :
# 1) dlib facial landmarks detector is used to detect all face parts , but we have used this model to detect the eyes only.
# 2) cv2.dnn.readNetFromCaffe model which is pretrained in model for detecting only face . We have used this model for detecting
#     the face because this model is much accurate then dlib facial landmarks detector.

known_height = 300



# The dlib face landmark detector will return a shape  object containing the 68 (x, y)-coordinates of the facial landmark regions.

# Using the shape_to_np  function, we cam convert this object to a NumPy array, allowing it to “play nicer” with our Python code.

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords



# This function will create the mask on the eye
def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


def contouring(thresh, mid, img, right=False):
    # Here we are finding the contours on the eye.
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        # detect the area which is detected maximum by contours.
        cnt = max(cnts, key=cv2.contourArea)

        # Image moments help you to calculate some features like center of mass of the object, area of the object etc
        # The function cv2.moments() gives a dictionary of all moment values calculated
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # This cx and cy is the center of the pupil detected.



        # This i=0 is for the left eye and i=1 is for the right eye.
        i = 0
        if right:
            i = 1
            cx = mid + cx
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), 1)

        # Here we are calculating the distance between the pupil and the different points.
        # So here our basic idea is we have drawn a square inside our eye.and inside our eye we have detected the pupil.
        # and then we calculate the distance between the pupil point and the points of the square.
        # So whichever distance is minimum , we will let to know that eye is seeing which side .
        # Here this idea works very well but if you want that the eye is seeing on specific left center point then you
        # Have to use infrared ray sensors which is a hardware component.

        left_eye_point_left = distance_calculate(left_eye[i][0], [cx, cy])
        left_eye_point_right = distance_calculate(left_eye[i][1], [cx, cy])
        left_eye_point_top = distance_calculate(left_eye[i][2], [cx, cy])
        left_eye_point_bottom = distance_calculate(left_eye[i][3], [cx, cy])
        left_eye_point_left_top_cor = distance_calculate(left_eye[i][4], [cx, cy]) / 1.414
        left_eye_point_left_bottom_cor = distance_calculate(left_eye[i][5], [cx, cy]) / 1.414
        left_eye_point_right_top = distance_calculate(left_eye[i][6], [cx, cy]) / 1.414
        left_eye_point_right_bottom = distance_calculate(left_eye[i][7], [cx, cy]) / 1.414


        minimum = min(left_eye_point_left, left_eye_point_right)

        if (left_eye_point_left == minimum or left_eye_point_left_top_cor == minimum or left_eye_point_left_bottom_cor == minimum):
            cv2.putText(img, "OK and you are looking left side", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(img, "Press y to capture image", (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        1)
        else:
            # (left_eye_point_right == minimum):
            cv2.putText(img, "OK and you are looking right side", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(img, "Press y to capture image", (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        1)

    except:
        pass




# initializes dlib’s pre-trained face detector so we can detect the face
detector = dlib.get_frontal_face_detector()

# loads the facial landmark predictor using the path to the supplied for detecting the landmarks points
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

# Function which calculates the distance between the points of the square and the pupil point
def distance_calculate(q1, q2):
    distance = math.sqrt(((int(q1[0]) - int(q2[0])) ** 2) + ((int(q1[1]) - int(q2[1])) ** 2))
    return distance

# Function which returns the square points. Through square points only we can able to know the distance between the pupil
# and the points . So here we are basically determines the square inside our eye.
def get_lines(eye_points, shape):
    left_point = (shape.part(eye_points[0]).x, shape.part(eye_points[0]).y)
    right_point = (shape.part(eye_points[3]).x, shape.part(eye_points[3]).y)
    center_top = midpoint(shape.part(eye_points[1]), shape.part(eye_points[2]))
    center_bottom = midpoint(shape.part(eye_points[5]), shape.part(eye_points[4]))

    # hor_line = cv2.line(img, left_point, right_point, (0, 255, 0), 1)
    ver_line = cv2.line(img, center_top, center_bottom, (0, 255, 0), 1)
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ver_line_length1 = (ver_line_length / 2)

    midx, midy = (center_top[0] + center_bottom[0]) / 2, (center_top[1] + center_bottom[1]) / 2

    center_left_pt = ((int(midx - ver_line_length1)), int(midy))
    center_right_pt = ((int(midx + ver_line_length1)), int(midy))
    # left_line=cv2.line(img,(midx),(midy),(0,255,0),1,)
    hor_line = cv2.line(img, center_left_pt, center_right_pt, (0, 255, 0), 1)
    left_top_cor = (center_left_pt[0], center_left_pt[1] + ver_line_length1)
    left_bottom_cor = (center_left_pt[0], center_left_pt[1] - ver_line_length1)
    right_top_cor = (center_right_pt[0], center_right_pt[1] + ver_line_length1)
    right_bottom_cor = (center_right_pt[0], center_right_pt[1] - ver_line_length1)

    # print(left_point)
    return center_left_pt, center_right_pt, center_top, center_bottom, left_top_cor, left_bottom_cor, right_top_cor, right_bottom_cor


# Function which will help to determines the square points .
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def nothing(x):
    pass

# We have to set threshold trackbar because of different lightining conditions.
# If we set the threshold to a specific value for all lightining conditions then it can't detect the perfect pupil point .
# So we have to set threshold trackbar for different lightining conditions.
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

 # we load our model using our --prototxt  and --model  file paths. We store the model as net
 # for detecting the face. we have detect the face and calculate the distance same way as previous
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
img_counter=0

lst1=[(3, 5),(3, 250),(3, 470),(630, 5),(630, 240),(630, 470)]
z=0
while (True):
    key = cv2.waitKey(1)
    ret, img = cap.read()
    max_distance = 0


    img = cv2.flip(img, +1)

    cv2.circle(img, lst1[z], 3, (255, 255, 255), 3)
    '''cv2.circle(img, (3, 250), 3, (255, 255, 255), 3)
    cv2.circle(img, (3, 470), 3, (255, 255, 255), 3)
    cv2.circle(img, (630, 5), 3, (255, 255, 255), 3)
    cv2.circle(img, (630, 240), 3, (255, 255, 255), 3)
    cv2.circle(img, (630, 470), 3, (255, 255, 255), 3)'''

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Here we are detecting the facial landmarks on the gray frame
    rects = detector(gray, 1)

    max_distance = 0

    (height1, width1) = img.shape[:2]


    # Here we have created the blob
    # The dnn.blobFromImage  takes care of pre-processing which includes setting the blob  dimensions and normalization.

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Here we are applying face detection
    #To detect faces, we pass the blob  through the net
    net.setInput(blob)
    detections = net.forward()

    # And from here we’ll loop over the detections and draw boxes around the detected faces:
    #  loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([width1, height1, width1, height1])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(startX, 0)
            startY = max(startY, 0)
            endX = min(endX, width1 - 1)
            endY = min(endY, height1 - 1)
            h = endY - startY
            text = "Face height: {}".format(h)


            distance = known_height / h * 0.51

            # draw the bounding box of the face along with the associated
            # probability

            # print(distance)

            if distance < 1.00:
                cv2.putText(img, "keep your distance greater than 100cm", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 1)


            else :

                if key == ord('y') or key == ord('Y'):

                    mask = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)[1][:, :, 0]
                    dst = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)

                    img_name = "opencv_frame_{}_{}.png".format(time.time(),z)

                    cv2.imwrite(img_name, dst)

                    z+=1

                    if z>5:
                        z=0









            # Now the lines  is used to print the distance(cm)  on the frame.
            cv2.putText(img, "%.2fcm" % (distance * 100), (img.shape[1] - 300, img.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)



    # This for loop is for the second model of face detection (The model which is used to detect the eyes)
    for rect in rects:

        # Determines the facial landmarks on the gray frame
        shape = predictor(gray, rect)


        left_eye = get_lines([36, 37, 38, 39, 40, 41], shape)
        right_eye = get_lines([42, 43, 44, 45, 46, 47, 48], shape)
        # Here we are creating the list of both the eyes
        left_eye = list((left_eye, right_eye))


        shape = shape_to_np(shape)

        (height, width) = img.shape[:2]

        # Here we are creating the mask as the same size of the frame .
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)


        # cv2. erode() method is used to perform erosion on the image. The basic idea of erosion is just like soil erosion only,
        # it erodes away the boundaries of foreground object (Always try to keep foreground in white).

        # dilation  increases the white region in the image or size of foreground object increases.
        # Normally, in cases like noise removal, erosion is followed by dilation.

        # for more information you can visit this link : https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html

        # This all stuf we have to do because we have to detect the pupil of the eye in different lightining conditions.
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)  # 1
        thresh = cv2.dilate(thresh, None, iterations=4)  # 2
        thresh = cv2.medianBlur(thresh, 3)  # 3
        thresh = cv2.bitwise_not(thresh)

        if distance > 1.00:

            contouring(thresh[:, 0:mid], mid, img)
            contouring(thresh[:, mid:], mid, img, True)




    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)

    if  key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()