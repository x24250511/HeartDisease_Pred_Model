import torch
from models.load_models import load_cnn_model, load_ptb_model, load_tabular_model
from fusion.fusion_engine import fusion_decision


cnn_model = load_cnn_model()
ptb_model, ptb_le = load_ptb_model()
tabular_model = load_tabular_model()


def predict_ecg(ecg_tensor):
    with torch.no_grad():
        output = cnn_model(ecg_tensor)
        pred = torch.argmax(output, dim=1).item()
    return "ABNORMAL" if pred == 1 else "NORMAL"


def predict_ptb(ptb_features):
    pred = ptb_model.predict(ptb_features)
    return ptb_le.inverse_transform(pred)[0]


def predict_tabular(tabular_features):
    pred = tabular_model.predict(tabular_features)
    return "HIGH_RISK" if pred[0] == 1 else "LOW_RISK"


def run_combined(ecg_tensor, ptb_features, tabular_features):
    cnn_pred = predict_ecg(ecg_tensor)

    if cnn_pred == "NORMAL":
        tab_pred = predict_tabular(tabular_features)
        return fusion_decision(cnn_pred, None, tab_pred)

    ptb_pred = predict_ptb(ptb_features)
    tab_pred = predict_tabular(tabular_features)

    return fusion_decision(cnn_pred, ptb_pred, tab_pred)
