from fusion.fusion_engine import fusion_decision

print(fusion_decision("ABNORMAL", "MI", "HIGH_RISK"))
print(fusion_decision("NORMAL", None, "LOW_RISK"))
