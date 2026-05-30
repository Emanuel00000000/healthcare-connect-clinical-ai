
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import io

app = FastAPI(title="Lung CNN Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "vgg16_mlp_pipeline_with_dropout.keras"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["Normal", "Benigno", "Maligno"]
MALIGNANT_THRESHOLD = 0.711


@app.get("/")
def home():
    return {"status": "API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((226, 226))

    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    probs = model.predict(arr)[0]
    pred_idx = int(np.argmax(probs))

    malignant_prob = float(probs[2])

    if malignant_prob >= MALIGNANT_THRESHOLD:
        risk = "High"
        recommendation = "Specialist review required"
    elif pred_idx == 1:
        risk = "Medium"
        recommendation = "Medical follow-up recommended"
    else:
        risk = "Low"
        recommendation = "No urgent escalation suggested"

    return {
        "prediction": class_names[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "normal_probability": round(float(probs[0]), 4),
        "benign_probability": round(float(probs[1]), 4),
        "malignant_probability": round(float(probs[2]), 4),
        "risk_level": risk,
        "recommendation": recommendation
    }
