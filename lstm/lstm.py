import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
import tensorflow as tf

# 自訂 focal loss（適用於 binary）
def focal_loss(gamma=2., alpha=0.25):
    def focal_binary_crossentropy(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt)
    return focal_binary_crossentropy

# 參數
DATA_DIR = "tabular_data_train"
INFO_CSV = "train_info.csv"
GROUP_SIZE = 27

# 1. 讀取 train_info 資訊
targets = ['gender', 'hold racket handed', 'play years', 'level']
info = pd.read_csv(INFO_CSV)

# 2. 整理訓練特徵與標籤
x_data = []
y_data = {t: [] for t in targets}

for f in Path(DATA_DIR).glob("*.csv"):
    unique_id = int(f.stem)
    row = info[info['unique_id'] == unique_id]
    if row.empty:
        continue
    df = pd.read_csv(f)
    if len(df) != GROUP_SIZE:
        continue
    x_data.append(df.values)
    for t in targets:
        y_data[t].append(row[t].values[0])

x_data = np.array(x_data)
n_samples, seq_len, feat_dim = x_data.shape

# 3. 標準化
scaler = MinMaxScaler()
x_data = x_data.reshape(n_samples * seq_len, feat_dim)
x_data = scaler.fit_transform(x_data).reshape(n_samples, seq_len, feat_dim)

# 4. 編碼標籤
encoders = {t: LabelEncoder().fit(y_data[t]) for t in targets}
y_outputs = {}

for t in targets:
    le = encoders[t]
    y = le.transform(y_data[t])
    if len(le.classes_) == 2:
        y_outputs[t] = np.array(y).reshape(-1, 1)
    else:
        y_outputs[t] = to_categorical(y, num_classes=len(le.classes_))

# 5. 建構模型
input_layer = Input(shape=(seq_len, feat_dim))
x = LSTM(256, return_sequences=True)(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = LSTM(128, return_sequences=True)(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = LSTM(64, return_sequences=True)(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = GlobalMaxPooling1D()(x)

output_layers = []
losses = {}
metrics = {}

for t in targets:
    name = t.replace(" ", "_")
    if y_outputs[t].shape[1] == 1:
        output = Dense(1, activation='sigmoid', name=name)(x)
        losses[name] = focal_loss()
        metrics[name] = ["accuracy", AUC(name=f"auc_{name}")]
    else:
        output = Dense(y_outputs[t].shape[1], activation='softmax', name=name)(x)
        losses[name] = 'categorical_crossentropy'
        metrics[name] = "accuracy"
    output_layers.append(output)

model = Model(inputs=input_layer, outputs=output_layers)
model.compile(optimizer=Adam(1e-3), loss=losses, metrics=metrics)

# 6. 建立 sample_weight（對 binary 輸出加 class_weight，其他給 ones）
sample_weights = []

for t in targets:
    y = y_outputs[t].flatten()
    if y_outputs[t].shape[1] == 1:
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        weight_map = {i: w for i, w in enumerate(class_weights)}
        weights = np.array([weight_map[label] for label in y])
        sample_weights.append(weights)
    else:
        sample_weights.append(np.ones(len(y)))

# 7. 訓練 with EarlyStopping
callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

model.fit(
    x_data,
    [y_outputs[t] for t in targets],
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    sample_weight=sample_weights,
    callbacks=callbacks
)

# 8. 儲存模型與 encoder
model.save("multi_output_lstm.h5")
for t in targets:
    np.save(f"encoder_{t}.npy", encoders[t].classes_)

print("\n✅ 模型訓練完成，已儲存")
