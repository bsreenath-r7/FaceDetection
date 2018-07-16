import os
import os.path
import numpy as np
from PIL import Image
from glob import glob

def get_images_and_labels(trainimagedir):
    #global face_cascade
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training

    image_paths = [y for x in os.walk(trainimagedir) for y in glob(os.path.join(x[0], '*.jpg'))]

    # images will contains face images
    images = []

    # labels will contains the label that is assigned to the image
    labels = []

    for image_path in image_paths:
        image_pil = Image.open(image_path)
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split("_")[0])

        images.append(image)
        labels.append(nbr)

        # Detect the face in the image
#        faces = settings.face_cascade.detectMultiScale(image, 1.3, 5)
#        # If face is detected, append the face to images and the label to labels
#        for (x, y, w, h) in faces:
#            images.append(image[y: y + h, x: x + w])
#            labels.append(nbr)
#            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
#            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels