import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def inference():
    group_size = 27
    model_dir = './AICUP/models/LightGBM_randomSearch'
    datapath = './AICUP/tabular_data_test'
    datalist = list(Path(datapath).glob('**/*.csv'))
    
    x_test = pd.DataFrame()
    uid_list = []

    for file in datalist:
        uid = int(Path(file).stem)
        data = pd.read_csv(file)
        x_test = pd.concat([x_test, data], ignore_index=True)
        uid_list += [uid] * len(data)

    # 特徵標準化
    scaler = joblib.load(f'{model_dir}/scaler.pkl')
    X_test_scaled = scaler.transform(x_test)

    # LabelEncoder
    encoders = {}
    for col in ['gender', 'hold racket handed', 'play years', 'level']:
        encoders[col] = joblib.load(f'{model_dir}/le {col}.pkl')

    # 預測任務
    predictions = {'unique_id': sorted(set(uid_list))}
    def predict_binary(model_file, label_name):
        clf = joblib.load(f'{model_dir}/{model_file}')
        pred_proba = clf.predict_proba(X_test_scaled)
        prob = [p[0] for p in pred_proba]  # 取類別 0 的機率
        num_groups = len(prob) // group_size

        results = []
        for i in range(num_groups):
            group_prob = prob[i * group_size:(i + 1) * group_size]
            avg_prob = sum(group_prob) / group_size
            results.append(float(avg_prob))
        
        predictions[label_name] = results
        print(predictions)

    def predict_multiary(model_file, label_name):
        clf = joblib.load(f'{model_dir}/{model_file}')
        pred_proba = clf.predict_proba(X_test_scaled)
        num_groups = len(pred_proba) // group_size
        classes = encoders[label_name].classes_

        for class_idx, class_name in enumerate(classes):
            results = []
            for i in range(num_groups):
                group = pred_proba[i * group_size:(i + 1) * group_size]
                avg_prob = np.mean([row[class_idx] for row in group])
                results.append(float(avg_prob))
            col_name = f'{label_name}_{class_name}'
            predictions[col_name] = results

    predict_binary('gender_binary.pkl', 'gender')
    predict_binary('hold racket handed_binary.pkl', 'hold racket handed')
    predict_multiary('play years_multiary.pkl', 'play years')
    predict_multiary('level_multiary.pkl', 'level')
    
    
    df = pd.DataFrame(predictions)
    df.to_csv('./AICUP/LightGBM/submission2.csv', index=False, float_format="%.6f")
    print("✅ 推論完成，機率值已儲存至 submission.csv")

if __name__ == '__main__':
    inference()
