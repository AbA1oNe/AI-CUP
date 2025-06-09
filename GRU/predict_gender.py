# predict_gender.py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from model_gender import GRUClassifier, load_txt

def cut_equal(seq, n=27):
    points = np.linspace(0, len(seq), n + 1, dtype=int)
    return [seq[points[i]:points[i+1]] for i in range(n)]

def predict_one(model, swings):
    model.eval()
    with torch.no_grad():
        probs = []
        for swing in swings:
            if len(swing) == 0:
                continue
            x = torch.tensor(swing, dtype=torch.float32).unsqueeze(0)
            prob = model(x).item()
            probs.append(prob)
        return sum(probs) / len(probs) if probs else 0.5

def main():
    model = GRUClassifier()
    model.load_state_dict(torch.load("model_gender.pth"))
    model.eval()

    test_dir = Path("test_data")
    rows = []
    for file in test_dir.glob("*.txt"):
        uid = int(file.stem)
        seq = load_txt(file)
        swings = cut_equal(seq)
        prob = predict_one(model, swings)
        rows.append([uid, prob])

    df = pd.DataFrame(rows, columns=["unique_id", "gender"])
    df = df.sort_values("unique_id")
    df.to_csv("submission_gender.csv", index=False, float_format='%.6f')
    print("✅ 預測完成 ➜ submission_gender.csv")

if __name__ == '__main__':
    main()
