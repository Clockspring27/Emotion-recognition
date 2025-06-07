import pickle

import cv2

from util import get_face_landmarks


emotions = ['HAPPY', 'SAD', 'SURPRISED']

with open('C:/Users/Admin/Downloads/projects/Emotion_detection/model_smote_new', 'rb') as f:
    model = pickle.load(f)

def image(img):
    img=cv2.imread(img)
    img=cv2.resize(img,(512,512))
    face_landmarks = get_face_landmarks(img, draw=True, static_image_mode=False)
    output = model.predict([face_landmarks])
    cv2.putText(img,
                emotions[int(output[0])],
                (10, img.shape[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 255, 0),
                5)
    print(emotions[int(output[0])])
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image("PATH TO IMAGE")    



