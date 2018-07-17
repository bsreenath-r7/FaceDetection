import face_recognition as FR
known_image = FR.load_image_file("img1.png")
unknown_image = FR.load_image_file("img2.png")
 
biden_encoding = FR.face_encodings(known_image)[0]
unknown_encoding = FR.face_encodings(unknown_image)[0]
 
results = FR.compare_faces([biden_encoding], unknown_encoding)