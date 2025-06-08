# Emotion Recognition with Dimensionality Reduction and KNN

This project focuses on developing a facial emotion recognition system using deep learning techniques. It includes data preparation, model training, evaluation, and deployment via a web interface.

## Features

- Reads and processes emotion data from TXT files
- Reduces feature dimensions using Neighborhood Components Analysis
- Classifies emotions using KNeighborsClassifier

## Requirements

Install required libraries using:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/Clockspring27/Emotion-recognition.git
cd Emotion-recognition
```
2. Test the model:
```bash
python test.py
```
3. Run the web demo:
```bash
streamlit web.py
```

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

These results demonstrate strong and balanced classification performance across all three emotion classes.

```
## License

This project is licensed under the MIT License. See `LICENSE` for details.
