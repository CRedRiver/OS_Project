"""
Predictive Caching System — Analysis Report Generator
Đọc sim_results.json + chạy model evaluation → xuất report HTML đẹp.
Chạy SAU khi đã có: train_model.py, markov_model.py, simulate_cache.py
"""

import json
import os
import glob
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
import warnings
warnings.filterwarnings("ignore")

# Import bắt buộc để pickle.load() resolve được class MarkovChainModel
from markov_chain_model import MarkovChainModel

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR     = "./collect/data"
RF_MODEL     = "./model/rf_model.pkl"
MK_MODEL     = "./model/markov_model.pkl"
ENCODER_PATH = "./model/block_encoder.pkl"
SIM_JSON     = "./report/sim_results.json"
REPORT_PATH  = "./report/analysis_report.html"
WINDOW_SIZE  = 5
TOP_K        = 50


# ─── LOAD HELPERS ─────────────────────────────────────────────────────────────
def load_data():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    dfs = [pd.read_csv(f, parse_dates=["Session_ID"]) for f in sorted(csv_files)]
    df = pd.concat(dfs, ignore_index=True).sort_values("Session_ID").reset_index(drop=True)
    return df


def build_features(df, le, window_size=WINDOW_SIZE, top_k=TOP_K):
    top_blocks = df["Block_ID"].value_counts().head(top_k).index.tolist()
    df = df[df["Block_ID"].isin(top_blocks)].copy()
    mask = df["Block_ID"].isin(le.classes_)
    df = df[mask].copy()
    df["block_enc"] = le.transform(df["Block_ID"])

    block_seq = df["block_enc"].values
    io_size   = df["IO_Size"].values
    io_offset = df["IO_Offset"].values

    records = []
    for i in range(window_size, len(block_seq)):
        window = block_seq[i - window_size: i]
        freq = {}
        for b in window:
            freq[b] = freq.get(b, 0) + 1
        records.append({
            **{f"prev_block_{j}": window[j] for j in range(window_size)},
            "io_size"         : io_size[i],
            "io_offset"       : io_offset[i],
            "window_unique"   : len(set(window)),
            "most_freq_block" : max(freq, key=freq.get),
            "last_block_count": freq.get(block_seq[i-1], 0),
            "block_delta"     : int(block_seq[i-1]) - int(block_seq[i-2]) if window_size >= 2 else 0,
            "target"          : block_seq[i],
        })

    feat_df = pd.DataFrame(records)
    feat_cols = [c for c in feat_df.columns if c != "target"]
    return feat_df[feat_cols].values, feat_df["target"].values


def eval_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    return {
        "name"     : name,
        "f1"       : round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall"   : round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "accuracy" : round(float(np.mean(y_pred == y_test)), 4),
    }


# ─── HTML REPORT ──────────────────────────────────────────────────────────────
def bar(val, max_val=1.0, color="#3B8BD4", height=16):
    pct = min(val / max_val, 1.0) * 100
    return (
        f'<div style="background:#f0f0f0;border-radius:4px;height:{height}px;overflow:hidden">'
        f'<div style="background:{color};width:{pct:.1f}%;height:100%;border-radius:4px"></div></div>'
    )


def color_rank(val, values, high_is_good=True):
    """Màu xanh = tốt nhất, đỏ = kém nhất."""
    ranks = sorted(values, reverse=high_is_good)
    idx = ranks.index(val)
    colors = ["#1D9E75", "#EF9F27", "#E24B4A", "#888780"]
    return colors[min(idx, len(colors) - 1)]


def generate_report(sim_results, model_metrics, data_stats):
    cfg = sim_results.get("config", {})
    res = sim_results.get("results", {})

    strategies = list(res.keys())
    hit_rates  = [res[s]["hit_rate"] for s in strategies]
    amats      = [res[s]["amat"]     for s in strategies]
    dts        = [res[s]["dt"]       for s in strategies]

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Sim table rows ────────────────────────────────────────────────────────
    sim_rows = ""
    for s in strategies:
        r  = res[s]
        hr = r["hit_rate"]
        am = r["amat"]
        dt = r["dt"]
        best_hr = max(hit_rates) == hr
        best_am = min(amats) == am
        best_dt = min(dts) == dt
        badge = ' <span style="background:#1D9E75;color:#fff;font-size:10px;padding:2px 6px;border-radius:10px;vertical-align:middle">BEST</span>' if best_hr else ""
        sim_rows += f"""
        <tr style="{'background:#f8fff8' if best_hr else ''}">
          <td style="font-weight:{'700' if best_hr else '400'}">{s}{badge}</td>
          <td>
            <div style="display:flex;align-items:center;gap:8px">
              <span style="color:{color_rank(hr,hit_rates,True)};font-weight:700;min-width:54px">{hr:.2%}</span>
              {bar(hr,max(hit_rates),color_rank(hr,hit_rates,True))}
            </div>
          </td>
          <td>
            <div style="display:flex;align-items:center;gap:8px">
              <span style="color:{color_rank(am,amats,False)};font-weight:700;min-width:60px">{am:.2f} ns</span>
              {bar(am,max(amats),color_rank(am,amats,False))}
            </div>
          </td>
          <td>
            <div style="display:flex;align-items:center;gap:8px">
              <span style="color:{color_rank(dt,dts,False)};font-weight:700;min-width:90px">{dt:,.0f} ns</span>
              {bar(dt,max(dts),color_rank(dt,dts,False))}
            </div>
          </td>
          <td style="text-align:center">{'✓' if best_am else ''}</td>
          <td style="text-align:center">{'✓' if best_dt else ''}</td>
        </tr>"""

    # ── Model metric rows ─────────────────────────────────────────────────────
    metric_rows = ""
    for m in model_metrics:
        metric_rows += f"""
        <tr>
          <td style="font-weight:600">{m['name']}</td>
          <td><span style="color:{color_rank(m['f1'],[x['f1'] for x in model_metrics],True)};font-weight:700">{m['f1']:.4f}</span></td>
          <td><span style="color:{color_rank(m['recall'],[x['recall'] for x in model_metrics],True)};font-weight:700">{m['recall']:.4f}</span></td>
          <td><span style="color:{color_rank(m['precision'],[x['precision'] for x in model_metrics],True)};font-weight:700">{m['precision']:.4f}</span></td>
          <td><span style="color:{color_rank(m['accuracy'],[x['accuracy'] for x in model_metrics],True)};font-weight:700">{m['accuracy']:.4f}</span></td>
        </tr>"""

    # ── Insight box ───────────────────────────────────────────────────────────
    best_s     = strategies[hit_rates.index(max(hit_rates))]
    best_dt_s  = strategies[dts.index(min(dts))]
    rf_metrics = next((m for m in model_metrics if "RF" in m["name"] or "Random" in m["name"]), None)
    mk_metrics = next((m for m in model_metrics if "Markov" in m["name"]), None)

    rf_better = rf_metrics and mk_metrics and rf_metrics["f1"] > mk_metrics["f1"]
    insight = f"""
    <li>Chiến lược tốt nhất về hit rate: <strong>{best_s}</strong> ({max(hit_rates):.2%})</li>
    <li>Chiến lược tốt nhất về Disk-head Time: <strong>{best_dt_s}</strong> ({min(dts):,.0f} ns)</li>
    <li>{'RF (F1=' + str(rf_metrics['f1']) + ') dự đoán chính xác hơn Markov (F1=' + str(mk_metrics['f1']) + ')' if rf_better else 'Markov Chain cạnh tranh tốt với RF'} trên tập test</li>
    <li>Cache size = {cfg.get('cache_size', '?')}  ·  Simulation = {cfg.get('sim_rows', '?'):,} requests</li>
    <li>Dữ liệu: {data_stats['total_rows']:,} rows  ·  {data_stats['unique_blocks']:,} unique blocks  ·  {data_stats['n_files']} file(s)</li>
    """

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predictive Cache — Analysis Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f5;color:#222;font-size:14px;line-height:1.6}}
  .wrap{{max-width:960px;margin:0 auto;padding:32px 24px}}
  h1{{font-size:24px;font-weight:700;color:#111;margin-bottom:4px}}
  h2{{font-size:16px;font-weight:600;color:#333;margin:28px 0 12px;padding-left:10px;border-left:3px solid #1D9E75}}
  .meta{{color:#888;font-size:12px;margin-bottom:28px}}
  .card{{background:#fff;border-radius:10px;padding:20px 24px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.07)}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{background:#f7f7f7;padding:8px 12px;text-align:left;font-weight:600;border-bottom:2px solid #e5e5e5;white-space:nowrap}}
  td{{padding:8px 12px;border-bottom:1px solid #f0f0f0;vertical-align:middle}}
  tr:last-child td{{border-bottom:none}}
  .stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:8px}}
  .stat{{background:#f7fdf9;border-radius:8px;padding:14px 16px;border:1px solid #d5eee4}}
  .stat .val{{font-size:22px;font-weight:700;color:#0F6E56;line-height:1.2}}
  .stat .lbl{{font-size:11px;color:#666;margin-top:2px}}
  .insight{{background:#fffbe6;border-left:4px solid #EF9F27;border-radius:4px;padding:12px 16px}}
  .insight li{{margin:4px 0 4px 16px;font-size:13px}}
  .badge{{display:inline-block;background:#1D9E75;color:#fff;font-size:10px;padding:2px 7px;border-radius:10px;vertical-align:middle;margin-left:6px}}
  .section-note{{font-size:12px;color:#888;margin-top:8px}}
  @media(max-width:600px){{.wrap{{padding:16px}}.stat-grid{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body>
<div class="wrap">
  <h1>Predictive Caching System — Analysis Report</h1>
  <div class="meta">Generated: {now}  ·  OS Project</div>

  <!-- Data stats -->
  <div class="card">
    <h2>Dataset Overview</h2>
    <div class="stat-grid">
      <div class="stat"><div class="val">{data_stats['total_rows']:,}</div><div class="lbl">Total I/O requests</div></div>
      <div class="stat"><div class="val">{data_stats['unique_blocks']:,}</div><div class="lbl">Unique blocks</div></div>
      <div class="stat"><div class="val">{data_stats['n_files']}</div><div class="lbl">Trace files</div></div>
      <div class="stat"><div class="val">{data_stats['top_k']}</div><div class="lbl">Top-K blocks (training)</div></div>
    </div>
  </div>

  <!-- Simulation results -->
  <div class="card">
    <h2>Cache Simulation Results
      <span style="font-size:12px;font-weight:400;color:#888;margin-left:8px">
        n={cfg.get('sim_rows',0):,} requests · cache size={cfg.get('cache_size',0)}
      </span>
    </h2>
    <table>
      <thead>
        <tr>
          <th>Strategy</th>
          <th>Cache Hit Rate ↑</th>
          <th>AMAT (ns) ↓</th>
          <th>Disk-head Time (ns) ↓</th>
          <th>Best AMAT</th>
          <th>Best DT</th>
        </tr>
      </thead>
      <tbody>{sim_rows}</tbody>
    </table>
    <p class="section-note">
      AMAT = Hit_rate×1ns + (1−Hit_rate)×100ns  ·  
      DT = seek_cost per miss (base 500ns + distance×0.05ns/sector)
    </p>
  </div>

  <!-- Model metrics -->
  <div class="card">
    <h2>Model Evaluation (test split 20%)</h2>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>F1-Score ↑</th>
          <th>Recall ↑</th>
          <th>Precision ↑</th>
          <th>Accuracy ↑</th>
        </tr>
      </thead>
      <tbody>{metric_rows}</tbody>
    </table>
    <p class="section-note">All metrics weighted-average. Higher is better (↑).</p>
  </div>

  <!-- Insights -->
  <div class="card">
    <h2>Key Findings</h2>
    <div class="insight"><ul>{insight}</ul></div>
  </div>

  <!-- Methodology -->
  <div class="card">
    <h2>Methodology</h2>
    <ul style="padding-left:20px;font-size:13px;line-height:2">
      <li><strong>Data</strong>: Baleen traces (Meta) + blktrace local traces, merged into unified 6-column CSV schema</li>
      <li><strong>Feature engineering</strong>: Sliding window (size={WINDOW_SIZE}), frequency stats, inter-arrival delta, IO size/offset</li>
      <li><strong>Models</strong>: Random Forest (100 trees, balanced class weights) · Markov Chain (order-1 transition matrix)</li>
      <li><strong>Simulator</strong>: Discrete Event Simulation — LRU, LFU, Predictive RF, Predictive Markov run on same trace</li>
      <li><strong>Prefetch</strong>: top-{3} predicted blocks pre-loaded into cache on each miss</li>
      <li><strong>Metrics</strong>: Cache Hit Rate · AMAT · Disk-head Time (seek cost model)</li>
    </ul>
  </div>
</div>
</body>
</html>"""
    return html


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Predictive Cache — Analysis Report Generator")
    print("=" * 55)

    # ── Check prerequisites ────────────────────────────────────────────────────
    missing = [p for p in [RF_MODEL, MK_MODEL, ENCODER_PATH, SIM_JSON] if not os.path.exists(p)]
    if missing:
        print(f"[ERROR] Thiếu files: {missing}")
        print("  Chạy theo thứ tự: train_model.py → markov_model.py → simulate_cache.py")
        exit(1)

    # ── Load sim results ───────────────────────────────────────────────────────
    with open(SIM_JSON) as f:
        sim_results = json.load(f)
    print("[OK] sim_results.json loaded")

    # ── Load models & encoder ──────────────────────────────────────────────────
    with open(RF_MODEL,     "rb") as f: rf_clf = pickle.load(f)
    with open(MK_MODEL,     "rb") as f: mk_clf = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f: le     = pickle.load(f)
    print("[OK] Models loaded")

    # ── Build feature matrix for eval ─────────────────────────────────────────
    print("[*] Loading data & building features...")
    df = load_data()
    data_stats = {
        "total_rows"   : len(df),
        "unique_blocks": df["Block_ID"].nunique(),
        "n_files"      : len(glob.glob(os.path.join(DATA_DIR, "*.csv"))),
        "top_k"        : TOP_K,
    }

    X, y = build_features(df, le)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Test set: {len(X_test):,} samples")

    # ── Evaluate models ────────────────────────────────────────────────────────
    print("[*] Evaluating models...")
    model_metrics = [
        eval_model(rf_clf, X_test, y_test, "Random Forest"),
        eval_model(mk_clf, X_test, y_test, "Markov Chain"),
    ]
    for m in model_metrics:
        print(f"  {m['name']:<20} F1={m['f1']:.4f}  Recall={m['recall']:.4f}  "
              f"Precision={m['precision']:.4f}  Acc={m['accuracy']:.4f}")

    # ── Generate HTML ──────────────────────────────────────────────────────────
    print("[*] Generating HTML report...")
    html = generate_report(sim_results, model_metrics, data_stats)

    os.makedirs("./report", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✓ Report saved → {REPORT_PATH}")
    print("  Mở file trong browser: xdg-open ./report/analysis_report.html")