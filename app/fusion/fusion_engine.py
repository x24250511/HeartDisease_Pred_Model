def fusion_decision(cnn_pred, ptb_pred=None, tabular_pred=None):
    if cnn_pred == "NORMAL":
        if tabular_pred == "HIGH_RISK":
            return " CLINICAL_RISK_DETECTED", \
                "ECG normal but clinical risk elevated."
        return "NORMAL", "No ECG abnormality detected."

    if cnn_pred == "ABNORMAL":
        if ptb_pred == "MI" and tabular_pred == "HIGH_RISK":
            return "HIGH_RISK_MI", \
                "High-risk myocardial infarction suspected."
        if ptb_pred == "MI" and tabular_pred == "LOW_RISK":
            return "POSSIBLE_MI", \
                "Possible myocardial infarction detected."

        if ptb_pred == "OTHER_ABNORMAL" and tabular_pred == "HIGH_RISK":
            return "HIGH_RISK_ABNORMAL", \
                "Abnormal ECG with elevated clinical risk."
        if ptb_pred == "OTHER_ABNORMAL" and tabular_pred == "LOW_RISK":
            return "POSSIBLE_ABNORMAL", \
                "Abnormal ECG detected, Monitoring advised."

    return "UNCERTAIN", "Further evaluation needed to clarify risk."
