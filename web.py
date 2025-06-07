import streamlit as st
import cv2
import numpy as np
import pickle
from util import get_face_landmarks

# Emotion labels
emotions = ['HAPPY', 'SAD', 'SURPRISED']

# Load the model
with open('PATH TO MODEL', 'rb') as f:
    model = pickle.load(f)

# Streamlit UI
st.title('Emotion Detector')
st.header('Upload an Image')

# Upload file
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

if file is not None:
    # Convert file to image array
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)  # BGR format

    # Extract landmarks
    face_landmarks = get_face_landmarks(img, draw=True, static_image_mode=False)

    if face_landmarks is not None:
        # Predict emotion
        output = model.predict([face_landmarks])
        predicted_emotion = emotions[int(output[0])]  # Assuming model outputs 0, 1, or 2

        # Annotate image
        cv2.putText(img,
                    predicted_emotion,
                    (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f'Predicted Emotion: {predicted_emotion}')
    else:
        st.error("No face landmarks detected.")



    
    
