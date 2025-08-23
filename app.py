import gradio as gr
import cv2
import numpy as np
import pickle
from functools import lru_cache

try:
    from util import get_face_landmarks
except Exception as e:
    raise ImportError(
        "Could not import 'get_face_landmarks' from util.py. "
        "Make sure util.py exists and defines get_face_landmarks(img, draw: bool, static_image_mode: bool)."
    ) from e


# ---- App Config ----
EMOTIONS = ["HAPPY", "SAD", "SURPRISED"]
MODEL_PATH = "model.pkl"
APP_TITLE = "Emotion Detector"
APP_DESC = (
    "Upload an image or use your webcam. Toggle 'Draw Landmarks' for visualization. "
)


# ---- Model Loader (cached) ----
@lru_cache(maxsize=1)
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


# ---- Core Inference ----
def predict_emotion(image, draw_toggle):
    """
    image: PIL.Image (from gr.Image with type='pil')
    draw_toggle: 'OFF' or 'ON'
    """
    if image is None:
        return {"Status": 1.0}, None, "Please upload an image."

    draw = (draw_toggle == "ON")

    # Convert PIL -> OpenCV BGR
    img_rgb = np.array(image)
    if img_rgb.ndim == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Extract landmarks
    landmarks = get_face_landmarks(img_bgr, draw=draw, static_image_mode=True)

    # Handle no-face case
    if landmarks is None or (hasattr(landmarks, "__len__") and len(landmarks) == 0):
        return {"No face detected": 1.0}, img_rgb, "No face detected in the image."

    # Load model
    model = load_model()

    # Predict
    output = model.predict([landmarks])
    pred_idx = int(output[0])
    pred_label = EMOTIONS[pred_idx] if 0 <= pred_idx < len(EMOTIONS) else str(pred_idx)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([landmarks])[0]
        confidence = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}
    else:
        confidence = {pred_label: 1.0}

    # If draw_toggle is ON, landmarks drawn on img_bgr by util
    img_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if draw else img_rgb

    status = f" Detected emotion: {pred_label}"
    return confidence, img_out, status


# ---- Gradio UI ----
with gr.Blocks(theme="default") as demo:
    gr.Markdown(f"# {APP_TITLE}\n{APP_DESC}")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Examples",
                sources=["upload", "webcam"],
                interactive=True,
            )
            draw_toggle = gr.Radio(
                choices=["OFF", "ON"],
                value="OFF",
                label="Draw Landmarks",
                interactive=True,
            )
            

        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=3, label="Predicted Emotion & Confidence")
            image_output = gr.Image(type="numpy", label="Image Output")
            status_output = gr.Textbox(label="Status", interactive=False)

    gr.Examples(
        examples=[
            ["examples/happy.png", "OFF"],
            ["examples/sad.png", "OFF"],
            ["examples/surprised.png", "OFF"],
        ],
        inputs=[image_input, draw_toggle],
        label="Try examples",
    )

    # Real-time: change triggers inference
    image_input.change(
        fn=predict_emotion,
        inputs=[image_input, draw_toggle],
        outputs=[label_output, image_output, status_output],
        queue=False,
    )
    draw_toggle.change(
        fn=predict_emotion,
        inputs=[image_input, draw_toggle],
        outputs=[label_output, image_output, status_output],
        queue=False,
    )

if __name__ == "__main__":
    demo.launch()
