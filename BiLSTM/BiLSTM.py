from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import TensorBoard
import datetime

def reshape_groups(X, group_size):
    n_samples, n_feats = X.shape
    n_groups = n_samples // group_size
    return X[:n_groups * group_size].reshape(n_groups, group_size, n_feats)

def group_labels(y_series, group_size):
    arr = y_series.values
    n_groups = arr.shape[0] // group_size
    # take label of first row in each group
    return arr.reshape(n_groups, group_size)[:,0]

def build_bi_lstm(input_shape, num_classes=1, binary=True):
    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape=input_shape))
    if binary:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(0.0001),
                      metrics=['AUC'])
    else:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(0.0001),
                      metrics=['accuracy'])
    return model

def FFT(xreal, ximag):    
    n = 2
    while n * 2 <= len(xreal):
        n *= 2

    p = int(math.log(n, 2))
    
    # Bit-reversal permutation
    for i in range(n):
        a = i
        b = 0
        for j in range(p):
            b = int(b * 2 + a % 2)
            a = a / 2
        if b > i:
            xreal[i], xreal[b] = xreal[b], xreal[i]
            ximag[i], ximag[b] = ximag[b], ximag[i]
            
    wreal = []
    wimag = []
    
    arg = float(-2 * math.pi / n)
    treal = float(math.cos(arg))
    timag = float(math.sin(arg))
    
    wreal.append(1.0)
    wimag.append(0.0)
    
    # Compute the twiddle factors
    for j in range(1, int(n / 2)):
        wreal.append(wreal[-1] * treal - wimag[-1] * timag)
        wimag.append(wreal[-1] * timag + wimag[-1] * treal)
        
    m = 2
    while m < n + 1:
        for k in range(0, n, m):
            for j in range(int(m / 2)):
                index1 = k + j
                index2 = int(index1 + m / 2)
                t = int(n * j / m)
                treal_temp = wreal[t] * xreal[index2] - wimag[t] * ximag[index2]
                timag_temp = wreal[t] * ximag[index2] + wimag[t] * xreal[index2]
                ureal = xreal[index1]
                uimag = ximag[index1]
                xreal[index1] = ureal + treal_temp
                ximag[index1] = uimag + timag_temp
                xreal[index2] = ureal - treal_temp
                ximag[index2] = uimag - timag_temp
        m *= 2
        
    return n, xreal, ximag   
    
def FFT_data(input_data, swinging_times):   
    txtlength = swinging_times[-1] - swinging_times[0]
    a_mean = [0] * txtlength
    g_mean = [0] * txtlength
       
    for num in range(len(swinging_times) - 1):
        a = []
        g = []
        for swing in range(swinging_times[num], swinging_times[num + 1]):
            # Compute magnitude for accelerometer and gyroscope
            a.append(math.sqrt(math.pow((input_data[swing][0] + input_data[swing][1] + input_data[swing][2]), 2)))
            g.append(math.sqrt(math.pow((input_data[swing][3] + input_data[swing][4] + input_data[swing][5]), 2)))

        a_mean[num] = sum(a) / len(a)
        g_mean[num] = sum(a) / len(a)   # Likely a bug: g_mean using a list

    return a_mean, g_mean

def feature(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer):
    allsum = []
    mean = []
    var = []
    rms = []
    XYZmean_a = 0
    a = []
    g = []
    a_s1 = 0
    a_s2 = 0
    g_s1 = 0
    g_s2 = 0
    a_k1 = 0
    a_k2 = 0
    g_k1 = 0
    g_k2 = 0
    
    # Calculate overall sum and magnitude arrays for accelerometer and gyroscope
    for i in range(len(input_data)):
        if i == 0:
            allsum = input_data[i]
            a.append(math.sqrt(math.pow((input_data[i][0] + input_data[i][1] + input_data[i][2]), 2)))
            g.append(math.sqrt(math.pow((input_data[i][3] + input_data[i][4] + input_data[i][5]), 2)))
            continue
        
        a.append(math.sqrt(math.pow((input_data[i][0] + input_data[i][1] + input_data[i][2]), 2)))
        g.append(math.sqrt(math.pow((input_data[i][3] + input_data[i][4] + input_data[i][5]), 2)))
       
        allsum = [allsum[j] + input_data[i][j] for j in range(len(input_data[i]))]
        
    mean = [allsum[j] / len(input_data) for j in range(len(input_data[i]))]
    
    # Calculate variance and root mean square (RMS)
    for i in range(len(input_data)):
        if i == 0:
            var = input_data[i]
            rms = input_data[i]
            continue

        var = [var[j] + math.pow((input_data[i][j] - mean[j]), 2) for j in range(len(input_data[i]))]
        rms = [rms[j] + math.pow(input_data[i][j], 2) for j in range(len(input_data[i]))]
        
    var = [math.sqrt(var[j] / len(input_data)) for j in range(len(input_data[i]))]
    rms = [math.sqrt(rms[j] / len(input_data)) for j in range(len(input_data[i]))]
    
    # Accelerometer and gyroscope basic statistics
    a_max = [max(a)]
    a_min = [min(a)]
    a_mean = [sum(a) / len(a)]
    g_max = [max(g)]
    g_min = [min(g)]
    g_mean = [sum(g) / len(g)]
    
    a_var = math.sqrt(math.pow((var[0] + var[1] + var[2]), 2))
    
    # Calculate 4th, 2nd, and 3rd order moments for kurtosis and skewness
    for i in range(len(input_data)):
        a_s1 += math.pow((a[i] - a_mean[0]), 4)
        a_s2 += math.pow((a[i] - a_mean[0]), 2)
        g_s1 += math.pow((g[i] - g_mean[0]), 4)
        g_s2 += math.pow((g[i] - g_mean[0]), 2)
        a_k1 += math.pow((a[i] - a_mean[0]), 3)
        g_k1 += math.pow((g[i] - g_mean[0]), 3)
    
    a_s1 = a_s1 / len(input_data)
    a_s2 = a_s2 / len(input_data)
    g_s1 = g_s1 / len(input_data)
    g_s2 = g_s2 / len(input_data)
    a_k2 = math.pow(a_s2, 1.5)
    g_k2 = math.pow(g_s2, 1.5)
    a_s2 = a_s2 * a_s2
    g_s2 = g_s2 * g_s2
    
    a_kurtosis = [a_s1 / a_s2]
    g_kurtosis = [g_s1 / g_s2]
    a_skewness = [a_k1 / a_k2]
    g_skewness = [g_k1 / g_k2]
    
    a_fft_mean = 0
    g_fft_mean = 0
    cut = int(n_fft / swinging_times)
    a_psd = []
    g_psd = []
    entropy_a = []
    entropy_g = []
    e1 = []
    e3 = []
    e2 = 0
    e4 = 0
    
    # Compute FFT and power spectral density (PSD) related features
    for i in range(cut * swinging_now, cut * (swinging_now + 1)):
        a_fft_mean += a_fft[i]
        g_fft_mean += g_fft[i]
        a_psd.append(math.pow(a_fft[i], 2) + math.pow(a_fft_imag[i], 2))
        g_psd.append(math.pow(g_fft[i], 2) + math.pow(g_fft_imag[i], 2))
        e1.append(math.pow(a_psd[-1], 0.5))
        e3.append(math.pow(g_psd[-1], 0.5))
        
    a_fft_mean = a_fft_mean / cut
    g_fft_mean = g_fft_mean / cut
    
    a_psd_mean = sum(a_psd) / len(a_psd)
    g_psd_mean = sum(g_psd) / len(g_psd)
    
    for i in range(cut):
        e2 += math.pow(a_psd[i], 0.5)
        e4 += math.pow(g_psd[i], 0.5)
    
    for i in range(cut):
        entropy_a.append((e1[i] / e2) * math.log(e1[i] / e2))
        entropy_g.append((e3[i] / e4) * math.log(e3[i] / e4))
    
    a_entropy_mean = sum(entropy_a) / len(entropy_a)
    g_entropy_mean = sum(entropy_g) / len(entropy_g)       
       
    # Combine all features into one output list and write to CSV
    output = (mean + var + rms + a_max + a_mean + a_min +
              g_max + g_mean + g_min + [a_fft_mean] + [g_fft_mean] +
              [a_psd_mean] + [g_psd_mean] + a_kurtosis + g_kurtosis +
              a_skewness + g_skewness + [a_entropy_mean] + [g_entropy_mean])
    writer.writerow(output)

def data_generate():
    datapath = './test_data'
    tar_dir = 'tabular_data_test'
    # Create target directory if it doesn't exist

    pathlist_txt = Path(datapath).glob('**/*.txt')

    # Process each .txt file in the train_data folder
    for file in pathlist_txt:
        f = open(file)

        All_data = []

        count = 0
        for line in f.readlines():
            # Skip empty lines and header line
            if line == '\n' or count == 0:
                count += 1
                continue
            num = line.split(' ')
            if len(num) > 5:
                tmp_list = []
                for i in range(6):
                    tmp_list.append(int(num[i]))
                All_data.append(tmp_list)
        
        f.close()

        # Generate indices for swings (features are divided based on these indices)
        swing_index = np.linspace(0, len(All_data), 28, dtype=int)

        headerList = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean', 
                      'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 
                      'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms', 
                      'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min', 
                      'a_fft', 'g_fft', 'a_psd', 'g_psd', 'a_kurt', 'g_kurt', 
                      'a_skewn', 'g_skewn', 'a_entropy', 'g_entropy']                
        
        with open(f'./{tar_dir}/{Path(file).stem}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headerList)
            try:
                a_fft, g_fft = FFT_data(All_data, swing_index)
                a_fft_imag = [0] * len(a_fft)
                g_fft_imag = [0] * len(g_fft)
                n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
                for i in range(len(swing_index)):
                    if i == 0:
                        continue
                    feature(All_data[swing_index[i - 1]: swing_index[i]], i - 1, len(swing_index) - 1,
                            n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer)
            except:
                print(Path(file).stem)
                continue

def main():
    import matplotlib.pyplot as plt

    # TensorBoard setup
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # If features haven't been generated, run data_generate() to create CSV feature files
    #data_generate()
    
    # Read training information and split data into 80% training and 20% testing based on player_id
    info = pd.read_csv('train_info.csv')
    unique_players = info['player_id'].unique()
    train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)
    
    # Read feature CSV files from the folder "./tabular_data_test"
    datapath = './tabular_data_train'
    datalist = list(Path(datapath).glob('**/*.csv'))
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']
    
    # Initialize DataFrames for train and test datasets
    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_mask)
    x_test = pd.DataFrame()
    y_test = pd.DataFrame(columns=target_mask)
    
    for file in datalist:
        file_stem = Path(file).stem
        if not file_stem.isdigit():
            print(f"Skipping file with non-numeric stem: {file_stem}")
            continue
        unique_id = int(file_stem)
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            continue
        # Extract player_id and target values from the row
        player_id = row['player_id'].iloc[0]
        data = pd.read_csv(file)
        target = row[target_mask]
        target_repeated = pd.concat([target] * len(data))
        target_repeated['unique_id'] = unique_id
        if player_id in train_players:
            x_train = pd.concat([x_train, data], ignore_index=True)
            y_train = pd.concat([y_train, target_repeated], ignore_index=True)
        elif player_id in test_players:
            x_test = pd.concat([x_test, data], ignore_index=True)
            y_test = pd.concat([y_test, target_repeated], ignore_index=True)
    
    # Feature normalization using MinMaxScaler and encoding labels using LabelEncoder
    scaler = MinMaxScaler()
    le = LabelEncoder()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)
    
    group_size = 27

    # Reshape the data into groups of size 27
    X_train_scaled = reshape_groups(X_train_scaled, group_size)
    X_test_scaled = reshape_groups(X_test_scaled, group_size)

    # gender (binary classification)
    y_train_gender = le.fit_transform(group_labels(y_train['gender'], group_size))
    y_test_gender = le.transform(group_labels(y_test['gender'], group_size))

    model_gender = build_bi_lstm(
        input_shape=(group_size, X_train_scaled.shape[2]),
        binary=True)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # Train the model with early stopping and TensorBoard
    model_gender.fit(
        X_train_scaled, y_train_gender,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop, tensorboard_callback]
    )
    
    pred_gender = model_gender.predict(X_test_scaled).ravel()
    print("gender AUC:", roc_auc_score(y_test_gender, pred_gender))

    # Confusion matrix for gender
    y_pred_gender = (pred_gender > 0.5).astype(int)
    cm_gender = confusion_matrix(y_test_gender, y_pred_gender)
    disp_gender = ConfusionMatrixDisplay(confusion_matrix=cm_gender, display_labels=le.classes_)
    disp_gender.plot()
    plt.title("Confusion Matrix - Gender")
    plt.show()

    # hold racket handed (binary classification)
    y_train_hold = le.fit_transform(group_labels(y_train['hold racket handed'], group_size))
    y_test_hold = le.transform(group_labels(y_test['hold racket handed'], group_size))

    model_hold = build_bi_lstm(
        input_shape=(group_size, X_train_scaled.shape[2]),
        binary=True
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # Train the model with early stopping and TensorBoard
    model_hold.fit(
        X_train_scaled, y_train_hold,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop, tensorboard_callback]
    )

    pred_hold = model_hold.predict(X_test_scaled).ravel()
    print("hold racket handed AUC:", roc_auc_score(y_test_hold, pred_hold))

    # Confusion matrix for hold racket handed
    y_pred_hold = (pred_hold > 0.5).astype(int)
    cm_hold = confusion_matrix(y_test_hold, y_pred_hold)
    disp_hold = ConfusionMatrixDisplay(confusion_matrix=cm_hold, display_labels=le.classes_)
    disp_hold.plot()
    plt.title("Confusion Matrix - Hold Racket Handed")
    plt.show()

    # years (multi classification)
    y_train_years = le.fit_transform(group_labels(y_train['play years'], group_size))
    y_test_years = le.transform(group_labels(y_test['play years'], group_size))
    n_classes = len(np.unique(y_train_years))

    model_years = build_bi_lstm(
        input_shape=(group_size, X_train_scaled.shape[2]),
        num_classes=n_classes,
        binary=False
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model_years.fit(
        X_train_scaled, y_train_years,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop, tensorboard_callback]
    )

    # Use predicted probabilities for multiclass ROC AUC
    y_score_years = model_years.predict(X_test_scaled)
    print("play years AUC:", roc_auc_score(y_test_years, y_score_years, multi_class='ovr'))

    # Confusion matrix for play years
    y_pred_years = np.argmax(y_score_years, axis=1)
    cm_years = confusion_matrix(y_test_years, y_pred_years)
    disp_years = ConfusionMatrixDisplay(confusion_matrix=cm_years, display_labels=le.classes_)
    disp_years.plot()
    plt.title("Confusion Matrix - Play Years")
    plt.show()

    # level (multi-class classification)
    y_train_level = le.fit_transform(group_labels(y_train['level'], group_size))
    y_test_level = le.transform(group_labels(y_test['level'], group_size))
    n_classes = len(np.unique(y_train_level))

    model_level = build_bi_lstm(
        input_shape=(group_size, X_train_scaled.shape[2]),
        num_classes=n_classes,
        binary=False)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # Train the model with early stopping and TensorBoard
    model_level.fit(
        X_train_scaled, y_train_level,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop, tensorboard_callback]
    )
    # Use predicted probabilities for multiclass ROC AUC
    y_score_level = model_level.predict(X_test_scaled)
    print("level AUC:", roc_auc_score(y_test_level, y_score_level, multi_class='ovr'))

    # Confusion matrix for level
    y_pred_level = np.argmax(y_score_level, axis=1)
    cm_level = confusion_matrix(y_test_level, y_pred_level)
    disp_level = ConfusionMatrixDisplay(confusion_matrix=cm_level, display_labels=le.classes_)
    disp_level.plot()
    plt.title("Confusion Matrix - Level")
    plt.show()

    unique_id = y_test['unique_id'].unique()

    # Initialize DataFrames for test dataset
    info = pd.read_csv('test_info.csv')
    datapath = './tabular_data_test'
    datalist = list(Path(datapath).glob('**/*.csv'))
    target_mask = ['mode','cut_point']

    x_test = pd.DataFrame()
    y_test = pd.DataFrame(columns=target_mask)

    for file in datalist:
        unique_id = int(Path(file).stem)
        row = info[info['unique_id'] == unique_id]
        # Skip files without info or without any rows
        # if row.empty:
        #     continue
        data = pd.read_csv(file)
        # if the file is empty, create a “dummy” block of 27 rows of zeros
        if data.empty:
            data = pd.DataFrame(
            np.zeros((27, len(data.columns))),
            columns=data.columns
            ) 
        # Extract target values and repeat per row
        target = row[target_mask]
        target_repeated = pd.concat([target] * len(data), ignore_index=True)
        target_repeated['unique_id'] = unique_id
        x_test = pd.concat([x_test, data], ignore_index=True)
        y_test = pd.concat([y_test, target_repeated], ignore_index=True)
    print(x_test.shape)
    # Feature normalization using MinMaxScaler and encoding labels using LabelEncoder
    X_test_scaled = scaler.fit_transform(x_test)
    
    group_size = 27

    # Reshape the data into groups of size 27
    X_test_scaled = reshape_groups(X_test_scaled, group_size)

    predictions = {
        'unique_id': group_labels(y_test['unique_id'], group_size)
    }

    def predict_binary(model, label_name):
        # one probability per group already
        pred = model.predict(X_test_scaled).ravel()
        predictions[label_name] = pred.tolist()

    def predict_multiclass(model, label_name):
        # multiclass probabilities per group
        pred = model.predict(X_test_scaled)
        offset = 0
        if label_name == 'level':
            offset = 2  # start numbering at 2 for level
        for class_idx in range(pred.shape[1]):
            col = pred[:, class_idx].tolist()
            predictions[f'{label_name}_{class_idx + offset}'] = col

    predict_binary(model_gender, 'gender')
    predict_binary(model_hold, 'hold racket handed')
    predict_multiclass(model_years, 'play years')
    predict_multiclass(model_level, 'level')

    df = pd.DataFrame(predictions)
    df.to_csv('./submission.csv', index=False)
    print("✅ Submission done.csv")


if __name__ == '__main__':
    main()