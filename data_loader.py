# data_loader.py
import os
import pandas as pd

def load_and_combine_datasets(data_dir="data"):
    combined = []

    # Dataset 1: Sentiment Analysis
    file1 = os.path.join(data_dir, "finance1.csv")
    if os.path.exists(file1):
        try:
            df1 = pd.read_csv(file1)
            df1 = df1.rename(columns={df1.columns[0]: "label", df1.columns[1]: "text"})
            df1 = df1[["text", "label"]].dropna()
            combined.append(df1)
            print("✅ Loaded finance1.csv")
        except Exception as e:
            print(f"❌ Error loading finance1.csv: {e}")

    # Dataset 2: Fraud Detection
    file2 = os.path.join(data_dir, "synthetic_log.csv")
    if os.path.exists(file2):
        try:
            df2 = pd.read_csv(file2)
            if all(col in df2.columns for col in ["type", "nameOrig", "nameDest", "amount", "isFraud"]):
                df2["text"] = (
                    "Transaction of type " + df2["type"].astype(str) +
                    " from " + df2["nameOrig"].astype(str) +
                    " to " + df2["nameDest"].astype(str) +
                    " of amount $" + df2["amount"].astype(str)
                )
                df2["label"] = df2["isFraud"].astype(str)
                df2 = df2[["text", "label"]].dropna()
                combined.append(df2)
                print("✅ Loaded synthetic_log.csv")
            else:
                print("⚠️ Missing required columns in synthetic_log.csv")
        except Exception as e:
            print(f"❌ Error loading synthetic_log.csv: {e}")

    # Dataset 3: Financial QA
    file3 = os.path.join(data_dir, "financial_qa.csv")
    if os.path.exists(file3):
        try:
            df3 = pd.read_csv(file3)
            if all(col in df3.columns for col in ["question", "context", "answer"]):
                df3["text"] = "Q: " + df3["question"].astype(str) + "\nContext: " + df3["context"].astype(str)
                df3["label"] = df3["answer"].astype(str)
                df3 = df3[["text", "label"]].dropna()
                combined.append(df3)
                print("✅ Loaded financial_qa.csv")
            else:
                print("⚠️ Missing required columns in financial_qa.csv")
        except Exception as e:
            print(f"❌ Error loading financial_qa.csv: {e}")

    if not combined:
        raise ValueError("❌ No valid datasets loaded. Please check your files and formats.")

    final_df = pd.concat(combined, ignore_index=True)
    print(f"📊 Combined dataset shape: {final_df.shape}")
    return final_df