import cv2
import settings
import FaceImageCapture as fic
import ImageLables as il
import numpy as np



print("Now capturing images")
gim = fic.CaptureImages()
gim.ClickImages()

trainimagedir = settings.trainimagedir
images, labels = il.get_images_and_labels(trainimagedir)




# For face recognition we will the the LBPH Face Recognizer
print("Now doing training")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Perform the tranining
recognizer.train(images, np.array(labels))

recognizer.save(settings.model_path+'\model.xml')

print("pretrained model saved to:" + "\n"+settings.model_path+"\model.xml")