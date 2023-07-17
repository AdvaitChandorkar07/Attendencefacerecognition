import csv
import cv2
import numpy as np
import pandas as pd
import face_recognition
import os
import dlib


def target_face(directory_path):
    known_faces = []
    face_parameters = []
    for filename in os.listdir(directory_path):
        image_path = os.path.join(directory_path, filename)
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) > 0:
            face_encodings = face_recognition.face_encodings(image, face_locations)
            known_faces.extend(face_encodings)
            face_landmarks = face_recognition.face_landmarks(image, face_locations)
            for landmarks in face_landmarks:
                parameters = [(p[0], p[1]) for p in landmarks.values()]
                face_parameters.append(parameters)
    return known_faces, face_parameters

def is_face_in_frame(frame, face_encodings):
    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) > 0:
        frame_encodings = face_recognition.face_encodings(frame, face_locations)
        for face_encoding in frame_encodings:
            matches = face_recognition.compare_faces(face_encodings, face_encoding)
            if True in matches:
                return 1
    return 0


known_faces, parameters = target_face("/Users/advait/Desktop/tp")
print(parameters)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    if is_face_in_frame(frame, known_faces):
        print("Advait is found")
    else:
        print("Advait is not found")
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()