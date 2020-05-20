# dlib is only compatible with python 3.6 and lower
# Steps to run on anacoda using virtual env - https://www.deciphertechnic.com/install-dlib-python-api-on-windows/
# 1. connda activate env_dlib
# 2. cd C:\Users\sinha\Desktop\Face Landmarks Detection
# 3. python facial_landmark_detection.py

#!python --version

import dlib
import cv2

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
  ret, frame = cap.read()

  if ret == False:
    continue

  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

  faces = detector(gray)

  for face in faces:
    left = face.left()
    right = face.right()
    top = face.top()
    bottom = face.bottom()

    #cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),3)

    landmarks = predictor(gray,face)

    for n in range(0,68):
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      cv2.circle(frame,(x,y),3,(0,0,255),-1)


    cv2.imshow("Color Frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

video_capture.release()
cv2.destroyAllWindows()