import os
import cv2
import os.path
import settings
from glob import glob



class CaptureImages():

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cwd = os.getcwd()
        while True:
            try:
                self.empid = str(input("Enter employee ID: \n"))
            except ValueError:
                print("Sorry, I didn't understand that. \n")
            else:
                break
        self.testimagedir = os.path.join(self.cwd,'TrainImages')
        self.emp_directory = os.path.join(self.cwd,'TrainImages', self.empid)

    def ClickImages(self):

        if not os.path.exists(self.testimagedir):
           os.mkdir(self.testimagedir)
        elif not os.path.exists(self.emp_directory ):
            os.mkdir(self.emp_directory)


        img_no_set = []
        image_paths = [y for x in os.walk(self.emp_directory) for y in glob(os.path.join(x[0], '*.jpg'))]
        for image_path in image_paths:
            nbr = int(os.path.split(image_path)[1].split("_")[1].split('.')[0])
            img_no_set.append(nbr)
        if len(img_no_set)==0:
            img_no=1
        else:
            img_no = max(img_no_set)+1


        while 1:
            ret, img = self.cap.read()

            if ret==False:
                  print ('cannot read frame')
                  break

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
            cv2.imshow('Train Image',img)
            k = cv2.waitKey(1) & 0xFF

            #Wait to press 's' key for capturing
            if k == ord('s') and len(faces) == 1:
                while not os.path.exists(os.path.join(self.emp_directory, str(self.empid) + '_' + str(img_no) + '.jpg')):
                    cv2.imwrite(os.path.join(self.emp_directory, str(self.empid) + '_' + str(img_no) + '.jpg'), roi_gray)
                img_no += 1

            #Wait to press 'q' key to quit
            elif k == ord('q'):
                break


        self.cap.release()
        cv2.destroyAllWindows()
