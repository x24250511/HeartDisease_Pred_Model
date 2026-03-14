import os
import torch
import joblib
from torchvision import models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_cnn_model():
    model_path = os.path.join(BASE_DIR, "ecg_cnn_model.pth")

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model


def load_ptb_model():
    model_path = os.path.join(BASE_DIR, "ptb_xgb_model.pkl")
    le_path = os.path.join(BASE_DIR, "ptb_label_encoder.pkl")

    model = joblib.load(model_path)
    label_encoder = joblib.load(le_path)

    return model, label_encoder


def load_tabular_model():
    model_path = os.path.join(BASE_DIR, "tabular_model.pkl")
    model = joblib.load(model_path)

    return model
