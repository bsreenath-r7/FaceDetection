import settings
import cv2
from glob import glob
import os
import sys

print("Taking test images")
cap = cv2.VideoCapture(0)

while True:
    try:
        empid = int(input("Enter employee ID: \n"))
    except ValueError:
        print("Sorry, I didn't understand that. \n")
    else:
        break

id_sr_no = {}
image_paths = [y for x in os.walk(os.path.join(os.getcwd(),'TestImages')) for y in glob(os.path.join(x[0], '*.jpg'))]
for image_path in image_paths:
    idno = int(os.path.split(image_path)[1].split("_")[0])
    id_sr_no.setdefault(idno, [])
    srno = int(os.path.split(image_path)[1].split("_")[1].split('.')[0])
    id_sr_no[idno].append(srno)

if empid in id_sr_no:
    if len(id_sr_no[empid])==0:
        img_no=1
    else:
        img_no = max(id_sr_no[empid])+1
else:
    img_no=1


while 1:
    ret, img = cap.read()

#    if ret==False:
#          print ('cannot read frame')
#          break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = settings.face_cascade.detectMultiScale(gray, 1.3, 5)

    print("Found {0} faces!".format(len(faces)))
    if len(faces) > 1:   #len(faces) == 0 or
        print("found more than one face") #break
    else:
        pass
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        roi_gray = gray[y-20:y+h+20, x-10:x+w+10]
        roi_color = img[y:y+h, x:x+w]

        eyes = settings.eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)

    # Display the resulting frame
    cv2.imshow('Test Image',img)
    k = cv2.waitKey(1) & 0xFF

    #Wait to press 's' key for capturing
    if k == ord('s') and len(faces) == 1:
        while not os.path.exists(os.path.join(os.path.join(os.getcwd(),'TestImages'), str(empid) + '_' + str(img_no) + '.jpg')):
            cv2.imwrite(os.path.join(os.path.join(os.getcwd(),'TestImages'), str(empid) + '_' + str(img_no) + '.jpg'), roi_gray)
        img_no += 1

    #Wait to press 'q' key to quit
    elif k == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
cap.release()




# For face recognition we will the the LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_path = [y for x in os.walk(os.getcwd()) for y in glob(os.path.join(x[0], '*.xml'))]
recognizer.read(model_path[0])
print("Applying pre trained model")

test_image_paths = [y for x in os.walk(settings.testimagedir) for y in glob(os.path.join(x[0], '*.jpg'))]
if len(test_image_paths)==0:
    sys.exit("TestImages folder has no images")

for image_path in test_image_paths:
    predict_image_cv2 = cv2.imread(image_path)
    predict_image = cv2.cvtColor(predict_image_cv2, cv2.COLOR_BGR2GRAY)
    faces = settings.face_cascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        employee_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w]) #predict_image[y-20:y+h+20, x-10:x+w+10]
        employee_actual = int(os.path.split(image_path)[1].split("_")[0])
        if employee_actual == employee_predicted:
            print("{} is Correctly Recognized with confidence {}".format(employee_actual, conf))
        else:
            print("{} is Incorrect Recognized as {} with confidence {}".format(employee_actual, employee_predicted, conf))
        #cv2.imshow("Recognizing Face", predict_image[y-20:y+h+20, x-10:x+w+10]) #predict_image[y: y + h, x: x + w]
        #cv2.waitKey(1)