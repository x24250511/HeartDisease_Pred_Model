from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

from models.load_models import load_cnn_model, load_ptb_model, load_tabular_model
from fusion.fusion_engine import fusion_decision

app = FastAPI(title="Heart Disease Multimodal API")

cnn_model = load_cnn_model()
ptb_model, ptb_le = load_ptb_model()
tabular_model = load_tabular_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@app.get("/")
def home():
    return {"message": "Heart Disease ML API running."}


@app.post("/predict_combined")
def predict_combined(
    cnn_pred: str,
    ptb_pred: str,
    tabular_pred: str
):

    label, message = fusion_decision(cnn_pred, ptb_pred, tabular_pred)
    return {
        "final_label": label,
        "message": message
    }


@app.post("/predict_ecg")
async def predict_ecg(file: UploadFile = File(...)):

    image = Image.open(file.file)

    # Crop header/footer
    width, height = image.size
    top_crop = int(0.18 * height)
    bottom_crop = int(0.92 * height)
    left_crop = int(0.02 * width)
    right_crop = int(0.98 * width)

    image = image.crop((left_crop, top_crop, right_crop, bottom_crop))

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = cnn_model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).item()

    label = "ABNORMAL" if pred == 0 else "NORMAL"

    return {
        "prediction": label,
        "confidence_abnormal": float(probs[0][0]),
        "confidence_normal": float(probs[0][1])
    }
