import cv2 as cv
import os
from util import get_face_landmarks
import numpy as np 

data_dir="PATH TO DATASET"
output=[]
for emotion_idx, emotion in enumerate(os.listdir(data_dir)):
    for image_path_ in os.listdir(os.path.join(data_dir,emotion)):
        image_path = os.path.join(data_dir,emotion,image_path_)
        image=cv.imread(image_path)
        face_landmarks= get_face_landmarks(image)

        if len(face_landmarks) ==1404:
            face_landmarks.append(int(emotion_idx))
            output.append(face_landmarks)

np.savetxt("data.txt",np.asarray(output))            
            

