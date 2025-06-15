import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 載入模型與任務資訊
model = load_model("multi_output_lstm.h5")
targets = ['gender', 'hold racket handed', 'play years', 'level']
encoders = {
    t: np.load(f"encoder_{t}.npy", allow_pickle=True)
    for t in targets
}

# 整理測試集
test_info = pd.read_csv("test_info.csv")
DATA_DIR = "tabular_data_test"

x_test, id_order = [], []
for row in test_info.itertuples():
    uid = row.unique_id
    path = Path(f"{DATA_DIR}/{uid}.csv")
    if path.exists():
        df = pd.read_csv(path)
        if len(df) == 27:
            x_test.append(df.values)
            id_order.append(uid)

x_test = np.array(x_test)
n, seq_len, feat_dim = x_test.shape

scaler = MinMaxScaler()
x_test = x_test.reshape(n * seq_len, feat_dim)
x_test = scaler.fit_transform(x_test).reshape(n, seq_len, feat_dim)

# 預測
y_preds = model.predict(x_test)
if len(targets) == 1:
    y_preds = [y_preds]

# 組裝 submission
submission = pd.DataFrame()
submission["unique_id"] = id_order

for i, t in enumerate(targets):
    probs = y_preds[i]
    class_names = encoders[t]

    if probs.shape[1] == 1:
        submission[t] = probs[:, 0]
    else:
        for j, cname in enumerate(class_names):
            submission[f"{t}_{cname}"] = probs[:, j]

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv 已產出，格式符合 sample_submission.csv")
