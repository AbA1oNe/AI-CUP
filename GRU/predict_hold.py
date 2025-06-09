# predict_hold.py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from model_hold import GRUClassifier, load_txt

def cut_equal(seq, n=27):
    points = np.linspace(0, len(seq), n + 1, dtype=int)
    return [seq[points[i]:points[i+1]] for i in range(n)]

def predict_one(model, swings):
    model.eval()
    with torch.no_grad():
        preds = []
        for swing in swings:
            if len(swing) == 0:
                continue
            x = torch.tensor(swing, dtype=torch.float32).unsqueeze(0)
            pred = model(x).item()
            preds.append(pred)
        return sum(preds) / len(preds) if preds else 0.5

def main():
    model = GRUClassifier()
    model.load_state_dict(torch.load("model_hold.pth"))
    model.eval()

    rows = []
    for file in Path("test_data").glob("*.txt"):
        uid = int(file.stem)
        seq = load_txt(file)
        swings = cut_equal(seq)
        pred = predict_one(model, swings)
        rows.append([uid, pred])

    df = pd.DataFrame(rows, columns=["unique_id", "hold racket handed"])
    df = df.sort_values("unique_id")
    df.to_csv("submission_hold.csv", index=False, float_format="%.6f")
    print("✅ 預測完成 ➜ submission_hold.csv")

if __name__ == '__main__':
    main()
