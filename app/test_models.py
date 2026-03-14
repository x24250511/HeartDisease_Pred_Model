import torch
from models.load_models import load_cnn_model

cnn = load_cnn_model()

for param in cnn.parameters():
    print(param.abs().mean())
    break
