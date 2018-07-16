import os
import cv2

global face_cascade, eye_cascade, model_path, trainimagedir, testimagedir
face_cascade = cv2.CascadeClassifier('C:\\Users\\AL2041\\FaceDetecion\\cascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\AL2041\\FaceDetecion\\cascades\\haarcascade_eye.xml')

trainimagedir = os.path.join(os.getcwd(),'TrainImages')

testimagedir = os.path.join(os.getcwd(),'TestImages')
if not os.path.exists(testimagedir):
    os.mkdir(testimagedir)
    print("TestImges folder created")

model_path = os.path.join(os.getcwd(), 'Model')
if not os.path.exists(model_path):
    os.mkdir(model_path)