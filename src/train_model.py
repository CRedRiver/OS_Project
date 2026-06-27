"""
Predictive Caching System — Feature Engineering + Random Forest Training
Dùng với data Baleen traces đã parse (Session_ID, Block_ID, IO_Offset, IO_Size, Op_Name, User)
"""

import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR    = "./collect/train_data"          # thư mục chứa các file CSV
WINDOW_SIZE = 5                         # số request lịch sử để predict tiếp theo
TOP_K       = 50                        # chỉ predict top K block phổ biến nhất
MODEL_PATH  = "./model/rf_model.pkl"
ENCODER_PATH= "./model/block_encoder.pkl"

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
def load_all_traces(data_dir):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV trong {data_dir}")

    dfs = []
    for f in sorted(csv_files):
        df = pd.read_csv(f, parse_dates=["Session_ID"])
        df["source_file"] = os.path.basename(f)
        dfs.append(df)
        print(f"  Loaded {os.path.basename(f)}: {len(df):,} rows")

    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values("Session_ID").reset_index(drop=True)
    return data

# ─── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────
def engineer_features(df, window_size=WINDOW_SIZE, top_k=TOP_K):
    print(f"\n[Feature Engineering] Window={window_size}, Top-K blocks={top_k}")

    # Chỉ giữ top K block phổ biến nhất (để bài toán classification khả thi)
    top_blocks = df["Block_ID"].value_counts().head(top_k).index.tolist()
    df = df[df["Block_ID"].isin(top_blocks)].copy()
    print(f"  Sau lọc top-{top_k}: {len(df):,} rows, {df['Block_ID'].nunique()} unique blocks")

    # Encode Block_ID thành số nguyên
    le = LabelEncoder()
    df["block_enc"] = le.fit_transform(df["Block_ID"])

    # Tạo features từ sliding window
    records = []
    block_seq = df["block_enc"].values
    io_size   = df["IO_Size"].values
    io_offset = df["IO_Offset"].values

    for i in range(window_size, len(block_seq)):
        window_blocks = block_seq[i - window_size : i]   # W block trước đó
        target        = block_seq[i]                       # block tiếp theo cần predict

        # Frequency features: mỗi block xuất hiện bao nhiêu lần trong window
        freq = {}
        for b in window_blocks:
            freq[b] = freq.get(b, 0) + 1

        record = {
            # Window features
            **{f"prev_block_{j}": window_blocks[j] for j in range(window_size)},
            # IO size và offset của request hiện tại
            "io_size"         : io_size[i],
            "io_offset"       : io_offset[i],
            # Thống kê trong window
            "window_unique"   : len(set(window_blocks)),
            "most_freq_block" : max(freq, key=freq.get),
            "last_block_count": freq.get(block_seq[i-1], 0),
            # Inter-arrival: khoảng cách giữa 2 block liên tiếp
            "block_delta"     : int(block_seq[i-1]) - int(block_seq[i-2]) if window_size >= 2 else 0,
            # Target
            "target"          : target,
        }
        records.append(record)

    feat_df = pd.DataFrame(records)
    print(f"  Feature matrix: {feat_df.shape}")
    return feat_df, le

# ─── 3. TRAIN RANDOM FOREST ───────────────────────────────────────────────────
def train_random_forest(feat_df):
    feature_cols = [c for c in feat_df.columns if c != "target"]
    X = feat_df[feature_cols].values
    y = feat_df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[Training] Train={len(X_train):,}  Test={len(X_test):,}")

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        n_jobs=-1,          # dùng toàn bộ CPU
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    print("  Model trained!")

    # ─── Đánh giá ───
    y_pred = clf.predict(X_test)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*50}")
    print(f"  F1-Score  (weighted): {f1:.4f}")
    print(f"  Recall    (weighted): {recall:.4f}")
    print(f"{'='*50}")
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Feature importance top 10
    feature_names = [c for c in feat_df.columns if c != "target"]
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("[Top 10 Feature Importances]")
    for idx in top_idx:
        print(f"  {feature_names[idx]:<25} {importances[idx]:.4f}")

    return clf, f1, recall

# ─── 4. SIMULATE CACHE HIT RATE ───────────────────────────────────────────────
def simulate_cache_hit_rate(clf, feat_df, le, cache_size=10, top_n_prefetch=3):
    """
    Mô phỏng đơn giản: với mỗi request, model predict top_n block tiếp theo.
    Nếu block thực sự nằm trong cache → HIT.
    """
    feature_cols = [c for c in feat_df.columns if c != "target"]
    X = feat_df[feature_cols].values
    y_true = feat_df["target"].values

    # Predict xác suất cho tất cả class
    proba = clf.predict_proba(X)

    cache = []
    hits = 0
    total = len(y_true)

    for i in range(total):
        actual_block = y_true[i]

        # Kiểm tra hit
        if actual_block in cache:
            hits += 1
        else:
            # Miss → thêm vào cache
            cache.append(actual_block)
            if len(cache) > cache_size:
                cache.pop(0)   # FIFO evict

        # Prefetch: lấy top_n block có xác suất cao nhất
        top_n = np.argsort(proba[i])[::-1][:top_n_prefetch]
        for b in top_n:
            if b not in cache:
                cache.append(b)
                if len(cache) > cache_size + top_n_prefetch:
                    cache.pop(0)

    hit_rate = hits / total
    return hit_rate

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Predictive Cache — Random Forest Training Pipeline")
    print("=" * 55)

    # 1. Load
    print("\n[1] Loading traces...")
    df = load_all_traces(DATA_DIR)
    print(f"  Total rows: {len(df):,} | Unique blocks: {df['Block_ID'].nunique():,}")

    # 2. Features
    print("\n[2] Engineering features...")
    feat_df, le = engineer_features(df)

    # 3. Train
    print("\n[3] Training Random Forest...")
    clf, f1, recall = train_random_forest(feat_df)

    # 4. Simulate
    print("\n[4] Simulating Cache Hit Rate...")
    hit_rate = simulate_cache_hit_rate(clf, feat_df, le, cache_size=20, top_n_prefetch=3)
    print(f"  Cache Hit Rate (size=20, prefetch=3): {hit_rate:.2%}")

    # 5. Save model
    os.makedirs("./model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"\n[5] Model saved → {MODEL_PATH}")
    print(f"    Encoder saved → {ENCODER_PATH}")
    print("\n✓ Done! Tiếp theo chạy: python simulate_cache.py")