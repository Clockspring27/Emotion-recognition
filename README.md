# Face Emotion Recognition

This project focuses on developing a facial emotion recognition system using deep learning techniques. It includes data preparation (**using runwayml/stable-diffusion-v1-5**), model training (**NCA +KNN**), evaluation, and deployment via **Gradio**.

## Features

- **95.38% Accuracy** on test data
- **Real-time inference** via web interface
- **Three emotion classes**: Happy, Sad, Surprised
- **Advanced feature extraction** using geometric facial landmarks
- **Efficient model** using only 2.14% of most important features

## web demo:
You can try the live demo ![here](https://huggingface.co/spaces/Clocksp/face-emotion-recognition)


## Model Performance

The emotion recognition model achieved an overall **accuracy of 95.38%** on the test dataset using a pipeline composed of:
- **Neighborhood Components Analysis (NCA)** for dimensionality reduction
- **K-Nearest Neighbors (KNN)** classifier for emotion prediction

## Classification Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0.0   | 1.0000    | 1.0000 | 1.0000   | 22      |
| 1.0   | 0.9091    | 0.9524 | 0.9302   | 21      |
| 2.0   | 0.9524    | 0.9091 | 0.9302   | 22      |

### Accuracy Metrics

- **Overall Accuracy**: **95.38%**
- **Macro Average F1-Score**: **0.9535**
- **Weighted Average F1-Score**: **0.9538**

### Confusion Matrix Summary

![alt text](https://github.com/Clockspring27/Emotion-recognition/blob/main/confusion_matrix.png)

### Report
Please click ![here](https://github.com/Clockspring27/Emotion-recognition/blob/main/Face_Emotion_Recognition_Report.pdf) for model Results
