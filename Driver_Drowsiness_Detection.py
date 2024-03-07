import imutils
import dlib
import cv2
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import numpy as np

# -------Warning Sound-----#
mixer.init()
sound = mixer.Sound('jagte.mp3')

# -------EYE_ASPECT_RATIO_CALCULATION-----#
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

eye_threshold = 0.25
consecutive_frames = 30

# -------Dlib and Video capture-----#
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:

        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        left_eye_asp_ratio = eye_aspect_ratio(leftEye)
        right_eye_asp_ratio = eye_aspect_ratio(rightEye)

        eye_asp_ratio = (left_eye_asp_ratio + right_eye_asp_ratio) / 2.0

        # -------shape-----#
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -2, (135, 206, 250), 1)
        cv2.drawContours(frame, [rightEyeHull], -2, (135, 206, 250), 1)

        # -------EYE-----#
        if eye_asp_ratio < eye_threshold:
            count = count + 1

            if count >=consecutive_frames:
                sound.play()
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            count = 0
            sound.stop()
            cv2.putText(frame, "****************EYE OPEN****************", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "****************EYE OPEN****************", (10, 325),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cv2.destroyAllWindows()