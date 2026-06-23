"""
Predictive Caching System — Markov Chain Training Script
Xây dựng transition matrix từ trace data để dự đoán block tiếp theo.
Class MarkovChainModel được import từ markov_chain_model.py (KHÔNG định nghĩa
ở đây) để pickle có thể load lại đúng class khi dùng ở simulate_cache.py.
"""

import pandas as pd
import numpy as np
import glob
import os
import pickle
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

from markov_chain_model import MarkovChainModel

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR      = "./collect/data"
TOP_K         = 50
MODEL_PATH    = "./model/markov_model.pkl"
ENCODER_PATH  = "./model/block_encoder.pkl"   # dùng chung encoder với RF


# ─── TRAIN PIPELINE ───────────────────────────────────────────────────────────
def load_and_prepare(data_dir, top_k=TOP_K, window_size=5):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy CSV trong {data_dir}")

    dfs = [pd.read_csv(f, parse_dates=["Session_ID"]) for f in sorted(csv_files)]
    df  = pd.concat(dfs, ignore_index=True).sort_values("Session_ID").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {df['Block_ID'].nunique():,} unique blocks")

    # Filter top-K
    top_blocks = df["Block_ID"].value_counts().head(top_k).index.tolist()
    df = df[df["Block_ID"].isin(top_blocks)].copy()

    # Dùng encoder đã có (từ RF) nếu tồn tại, để class indices khớp nhau
    if os.path.exists(ENCODER_PATH):
        with open(ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        mask = df["Block_ID"].isin(le.classes_)
        df = df[mask].copy()
        df["block_enc"] = le.transform(df["Block_ID"])
        print(f"  Dùng encoder RF: {len(le.classes_)} classes")
    else:
        le = LabelEncoder()
        df["block_enc"] = le.fit_transform(df["Block_ID"])
        print(f"  Tạo encoder mới: {df['Block_ID'].nunique()} classes")

    # Build feature matrix (window)
    block_seq = df["block_enc"].values
    io_size   = df["IO_Size"].values
    io_offset = df["IO_Offset"].values

    records = []
    for i in range(window_size, len(block_seq)):
        window = block_seq[i - window_size: i]
        freq = {}
        for b in window:
            freq[b] = freq.get(b, 0) + 1

        record = {
            **{f"prev_block_{j}": window[j] for j in range(window_size)},
            "io_size"         : io_size[i],
            "io_offset"       : io_offset[i],
            "window_unique"   : len(set(window)),
            "most_freq_block" : max(freq, key=freq.get),
            "last_block_count": freq.get(block_seq[i-1], 0),
            "block_delta"     : int(block_seq[i-1]) - int(block_seq[i-2]) if window_size >= 2 else 0,
            "target"          : block_seq[i],
        }
        records.append(record)

    feat_df = pd.DataFrame(records)
    feature_cols = [c for c in feat_df.columns if c != "target"]
    X = feat_df[feature_cols].values
    y = feat_df["target"].values

    return X, y, le, feat_df


def evaluate_markov(model, X_test, y_test):
    from sklearn.metrics import f1_score, recall_score

    y_pred = model.predict(X_test)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*50}")
    print(f"  Markov Chain — Evaluation")
    print(f"  F1-Score  (weighted): {f1:.4f}")
    print(f"  Recall    (weighted): {recall:.4f}")
    print(f"  Accuracy            : {model.score(X_test, y_test):.4f}")
    print(f"{'='*50}")
    return f1, recall


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    print("=" * 55)
    print("  Predictive Cache — Markov Chain Training Pipeline")
    print("=" * 55)

    print("\n[1] Loading & preparing data...")
    X, y, le, feat_df = load_and_prepare(DATA_DIR, top_k=TOP_K, window_size=5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train={len(X_train):,}  Test={len(X_test):,}")

    print("\n[2] Training Markov Chain (order=1)...")
    model = MarkovChainModel(order=1)
    model.fit(X_train, y_train)

    print("\n[3] Evaluating...")
    evaluate_markov(model, X_test, y_test)

    print("\n[4] Saving model...")
    os.makedirs("./model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved → {MODEL_PATH}")
    print("\n✓ Done! Chạy python simulate_cache.py để so sánh.")