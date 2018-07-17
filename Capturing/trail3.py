import cv2
import os
import dlib

face_cascade = cv2.CascadeClassifier('C:\\Users\\AL2041\\FaceDetecion\\cascades\\haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('C:\\Users\\AL2041\\FaceDetecion\\cascades\\haarcascade_eye.xml')

#Create the tracker we will use
tracker = dlib.correlation_tracker()

#The variable we use to keep track of the fact whether we are
#currently using the dlib tracker
trackingFace = 0
cap = cv2.VideoCapture(0)
cwd = os.getcwd()
empid = str(input("Enter employee ID: \n"))
emp_directory = os.path.join(cwd,empid )

if not os.path.exists(emp_directory):
   os.mkdir(emp_directory)

img_no = 1
rectangleColor = (0,165,255)

while 1:
    ret, img = cap.read()
    if not trackingFace:
        if ret==False:
              print ('cannot read frame')
              break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        print("Found {0} faces!".format(len(faces)))
        if len(faces) > 1:   #len(faces) == 0 or
            print("found more than one face")
            break
        else:
            pass


    #For now, we are only interested in the 'largest' face, and we
    #determine this based on the largest area of the found
    #rectangle. First initialize the required variables to 0

        maxArea = 0
        x = 0
        y = 0
        w = 0
        h = 0


        #Loop over all faces and check if the area for this face is
        #the largest so far
        for (_x,_y,_w,_h) in faces:
            if  _w*_h > maxArea:
                x = _x
                y = _y
                w = _w
                h = _h
                maxArea = w*h

            #If one or more faces are found, draw a rectangle around the
            #largest face present in the picture
            if maxArea > 0 :
                cv2.rectangle(img,  (x-10, y-20),(x + w+10 , y + h+20),rectangleColor,2)

        # Display the resulting frame
        cv2.imshow('img',img)

        # write on the live stream video
        text_color = (255,255,255)
        cv2.putText(img, "Press S to save image", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, thickness=2)

        k = cv2.waitKey(10) & 0xFF
        #Wait to press 's' key for capturing
        if k == ord('s'):
            cv2.imwrite(os.path.join(empid, str(empid) + '_' + str(img_no) + '.jpg'), roi_gray)
            img_no += 1

        #Wait to press 'q' key to quit
        elif k == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
