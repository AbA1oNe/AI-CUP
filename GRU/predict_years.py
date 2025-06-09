# predict_years.py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from model_years import GRUClassifier, load_txt

def cut_equal(seq, n=27):
    points = np.linspace(0, len(seq), n + 1, dtype=int)
    return [seq[points[i]:points[i+1]] for i in range(n)]

def predict_avg(model, swings):
    model.eval()
    pred_sum = torch.zeros(3)
    with torch.no_grad():
        for s in swings:
            if len(s) == 0: continue
            x = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_sum += probs[0]
    return (pred_sum / len(swings)).tolist()

def main():
    model = GRUClassifier(output_size=3)
    model.load_state_dict(torch.load("model_years.pth"))
    model.eval()

    rows = []
    for file in Path("test_data").glob("*.txt"):
        uid = int(file.stem)
        seq = load_txt(file)
        swings = cut_equal(seq)
        prob = predict_avg(model, swings)
        rows.append([uid] + prob)

    df = pd.DataFrame(rows, columns=[
        "unique_id", "play years_0", "play years_1", "play years_2"
    ])
    df = df.sort_values("unique_id")
    df.to_csv("submission_years.csv", index=False, float_format="%.6f")
    print("✅ submission_years.csv created")

if __name__ == '__main__':
    main()
