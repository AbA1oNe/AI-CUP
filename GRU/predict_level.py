# predict_level.py
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# ✅ 從 model_level.py 引入模型和載入 .txt 函式
from model_level import GRUClassifier, load_txt

def cut_equal(sequence, n=27):
    """將序列等分為 n 段"""
    points = np.linspace(0, len(sequence), n + 1, dtype=int)
    return [sequence[points[i]:points[i+1]] for i in range(n)]

def predict_avg(model, swings):
    """對 27 段 swing 平均 softmax 結果"""
    model.eval()
    pred_sum = torch.zeros(4)
    with torch.no_grad():
        for s in swings:
            if len(s) == 0: continue
            x = torch.tensor(s, dtype=torch.float32).unsqueeze(0)  # [1, len, 6]
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_sum += probs[0]
    return (pred_sum / len(swings)).tolist()

def main():
    model = GRUClassifier(output_size=4)
    model.load_state_dict(torch.load("model_level.pth"))
    model.eval()

    rows = []
    for file in Path("test_data").glob("*.txt"):
        uid = int(file.stem)
        seq = load_txt(file)
        swings = cut_equal(seq)
        pred = predict_avg(model, swings)
        rows.append([uid] + pred)

    df = pd.DataFrame(rows, columns=[
        "unique_id", "level_2", "level_3", "level_4", "level_5"
    ])
    df = df.sort_values("unique_id")
    df.to_csv("submission_level.csv", index=False, float_format="%.6f")
    print("✅ 預測完成 ➜ submission_level.csv")

if __name__ == '__main__':
    main()
