# merge_predictions.py
import pandas as pd

def main():
    # 讀入四個子任務的預測結果
    df_gender = pd.read_csv("submission_gender.csv")
    df_hold = pd.read_csv("submission_hold.csv")
    df_years = pd.read_csv("submission_years.csv")
    df_level = pd.read_csv("submission_level.csv")

    # 根據 unique_id 合併
    df = df_gender.merge(df_hold, on="unique_id") \
                  .merge(df_years, on="unique_id") \
                  .merge(df_level, on="unique_id")

    # 確保欄位順序
    columns = [
        "unique_id", "gender", "hold racket handed",
        "play years_0", "play years_1", "play years_2",
        "level_2", "level_3", "level_4", "level_5"
    ]
    df = df[columns]

    df.to_csv("submission.csv", index=False, float_format="%.6f")
    print("✅ 成功合併為 submission.csv")

if __name__ == "__main__":
    main()
