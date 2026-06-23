"""
Predictive Caching System — DES Simulator v2
So sánh: LRU  vs  LFU  vs  Predictive (Random Forest)  vs  Markov Chain
Metrics: Cache Hit Rate · AMAT · Disk-head Time (DT)
Hiển thị kết quả với rich + plotext
"""

import pandas as pd
import numpy as np
import pickle
import glob
import os
from collections import OrderedDict, Counter

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    from rich import box
    import plotext as plt
    RICH_OK = True
except ImportError:
    RICH_OK = False
    plt = None
    print("[WARNING] rich/plotext chưa cài. Chạy: pip install rich plotext")

# Import bắt buộc để pickle.load() resolve được class MarkovChainModel
# khi đọc markov_model.pkl. Phải import cùng đường dẫn module đã dùng
# lúc train (markov_model.py cũng import từ markov_chain_model.py).
from markov_chain_model import MarkovChainModel

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR       = "./collect/data"
RF_MODEL_PATH  = "./model/rf_model.pkl"
MK_MODEL_PATH  = "./model/markov_model.pkl"
ENCODER_PATH   = "./model/block_encoder.pkl"

CACHE_SIZE     = 50      # tăng từ 20 → 50 để thấy sự khác biệt rõ hơn
WINDOW_SIZE    = 5
TOP_K          = 50
SIM_ROWS       = 20000   # tăng từ 5000 → 20000 để kết quả ổn định hơn
TOP_N_PREFETCH = 3

# AMAT constants (ns): Hit=1ns, Miss=100ns (disk seek ~8ms → normalized)
AMAT_HIT_NS  = 1.0
AMAT_MISS_NS = 100.0

# Disk-head time: giả định mỗi miss phải seek, chi phí tỷ lệ với khoảng cách sector
DT_SEEK_BASE_NS  = 500.0   # overhead cơ bản mỗi lần seek (ns, normalized)
DT_PER_SECTOR_NS = 0.05    # chi phí theo khoảng cách sector

console = Console() if RICH_OK else None


# ─── LRU CACHE ────────────────────────────────────────────────────────────────
class LRUCache:
    def __init__(self, capacity):
        self.cap   = capacity
        self.cache = OrderedDict()

    def access(self, block):
        hit = block in self.cache
        if hit:
            self.cache.move_to_end(block)
        else:
            self.cache[block] = True
            if len(self.cache) > self.cap:
                self.cache.popitem(last=False)
        return hit


# ─── LFU CACHE ────────────────────────────────────────────────────────────────
class LFUCache:
    def __init__(self, capacity):
        self.cap   = capacity
        self.cache = {}
        self.freq  = Counter()

    def access(self, block):
        hit = block in self.cache
        if hit:
            self.freq[block] += 1
        else:
            if len(self.cache) >= self.cap:
                evict = min(self.freq, key=self.freq.get)
                del self.cache[evict]
                del self.freq[evict]
            self.cache[block] = True
            self.freq[block]  = 1
        return hit


# ─── GENERIC PREDICTIVE CACHE ─────────────────────────────────────────────────
class PredictiveCache:
    """
    Dùng được với bất kỳ model nào có predict_proba(X) interface.
    RF và Markov đều tương thích.
    """

    def __init__(self, capacity, clf, le,
                 window_size=WINDOW_SIZE,
                 top_n=TOP_N_PREFETCH,
                 label="Predictive"):
        self.cap     = capacity
        self.clf     = clf
        self.le      = le
        self.win     = window_size
        self.top_n   = top_n
        self.label   = label
        self.cache   = OrderedDict()
        self.history = []   # encoded block history

    def _make_features(self, io_size, io_offset):
        if len(self.history) < self.win:
            return None
        window = self.history[-self.win:]
        freq = {}
        for b in window:
            freq[b] = freq.get(b, 0) + 1
        features = (
            list(window)
            + [io_size, io_offset]
            + [len(set(window))]
            + [max(freq, key=freq.get)]
            + [freq.get(window[-1], 0)]
            + [int(window[-1]) - int(window[-2]) if self.win >= 2 else 0]
        )
        return np.array(features).reshape(1, -1)

    def _prefetch(self, io_size, io_offset):
        feat = self._make_features(io_size, io_offset)
        if feat is None:
            return []
        try:
            proba   = self.clf.predict_proba(feat)[0]
            top_idx = np.argsort(proba)[::-1][:self.top_n]
            return list(top_idx)
        except Exception:
            return []

    def access(self, block, io_size=0, io_offset=0):
        # Encode block
        if hasattr(self.le, 'classes_') and block in self.le.classes_:
            enc = int(self.le.transform([block])[0])
        else:
            enc = -1

        hit = block in self.cache
        if hit:
            self.cache.move_to_end(block)
        else:
            prefetch_blocks = self._prefetch(io_size, io_offset)
            self.cache[block] = True
            if len(self.cache) > self.cap:
                self.cache.popitem(last=False)
            # Prefetch
            for pb in prefetch_blocks:
                if pb != enc:
                    try:
                        real_block = self.le.inverse_transform([pb])[0]
                        if real_block not in self.cache:
                            self.cache[real_block] = True
                            if len(self.cache) > self.cap + self.top_n:
                                self.cache.popitem(last=False)
                    except Exception:
                        pass

        if enc >= 0:
            self.history.append(enc)
            if len(self.history) > self.win + 10:
                self.history.pop(0)

        return hit


# ─── DISK-HEAD TIME METRIC ────────────────────────────────────────────────────
def compute_disk_head_time(blocks, hits, offsets):
    """
    Tính Disk-head Time (DT): tổng chi phí vật lý phục vụ các miss requests.
    Bao gồm: seek time (tỷ lệ với khoảng cách sector) + base overhead.
    """
    total_dt = 0.0
    last_sector = 0
    for block, hit, offset in zip(blocks, hits, offsets):
        if not hit:
            sector = int(offset) // 512 if offset > 0 else int(block)
            seek_dist = abs(sector - last_sector)
            dt = DT_SEEK_BASE_NS + seek_dist * DT_PER_SECTOR_NS
            total_dt += dt
            last_sector = sector
    return total_dt


# ─── SIMULATION ───────────────────────────────────────────────────────────────
def run_simulation(df, rf_clf, mk_clf, le):
    df_sim = df.head(SIM_ROWS).copy()

    lru  = LRUCache(CACHE_SIZE)
    lfu  = LFUCache(CACHE_SIZE)
    pred_rf = PredictiveCache(CACHE_SIZE, rf_clf, le, label="RF")
    pred_mk = PredictiveCache(CACHE_SIZE, mk_clf, le, label="Markov")

    stats = {
        "LRU"            : [],
        "LFU"            : [],
        "Predictive (RF)": [],
        "Markov Chain"   : [],
    }

    blocks   = df_sim["Block_ID"].values
    io_sizes = df_sim["IO_Size"].values
    offsets  = df_sim["IO_Offset"].values

    iterator = range(len(blocks))
    if RICH_OK:
        iterator = track(iterator,
                         description="[cyan]Simulating...[/cyan]",
                         total=len(blocks))

    for i in iterator:
        b  = blocks[i]
        sz = int(io_sizes[i])
        of = int(offsets[i])

        stats["LRU"].append(int(lru.access(b)))
        stats["LFU"].append(int(lfu.access(b)))
        stats["Predictive (RF)"].append(int(pred_rf.access(b, sz, of)))
        stats["Markov Chain"].append(int(pred_mk.access(b, sz, of)))

    return stats, blocks, offsets


# ─── REPORT ───────────────────────────────────────────────────────────────────
def print_report(stats, blocks, offsets):
    results = {}
    for name, hits in stats.items():
        n        = len(hits)
        hit_arr  = np.array(hits, dtype=bool)
        hit_rate = hit_arr.mean()
        amat     = hit_rate * AMAT_HIT_NS + (1 - hit_rate) * AMAT_MISS_NS
        dt       = compute_disk_head_time(blocks[:n], hit_arr, offsets[:n])
        results[name] = {
            "hit_rate": hit_rate,
            "amat"    : amat,
            "dt"      : dt,
            "hits"    : hits,
        }

    if RICH_OK:
        # ── Bảng kết quả ──────────────────────────────────────────────────────
        table = Table(
            title=f"Cache Strategy Comparison  (n={SIM_ROWS:,}, cache={CACHE_SIZE})",
            box=box.ROUNDED, show_lines=True
        )
        table.add_column("Strategy",       style="bold cyan",   min_width=18)
        table.add_column("Hit Rate",       style="bold green",  justify="right")
        table.add_column("AMAT (ns)",      style="bold yellow", justify="right")
        table.add_column("DT Score",       style="bold magenta",justify="right")
        table.add_column("Total Hits",     style="dim",         justify="right")

        best_name = max(results, key=lambda k: results[k]["hit_rate"])
        best_dt   = min(results, key=lambda k: results[k]["dt"])

        for name, r in results.items():
            is_best = name == best_name
            style   = "bold white on dark_green" if is_best else ""
            dt_mark = " ★" if name == best_dt else ""
            table.add_row(
                name,
                f"{r['hit_rate']:.2%}",
                f"{r['amat']:.2f}",
                f"{r['dt']:,.0f}{dt_mark}",
                f"{sum(r['hits']):,}",
                style=style,
            )

        console.print()
        console.print(table)
        console.print(
            "[dim]  ★ = lowest Disk-head Time  |  "
            "DT = total seek cost for all misses (ns, normalized)[/dim]\n"
        )

        # ── Hit Rate Over Time ─────────────────────────────────────────────────
        console.print(Panel(
            "[bold]Cache Hit Rate Over Time  (rolling window = 500)[/bold]",
            style="dim"
        ))
        window = 500
        colors = ["red", "blue", "green", "yellow"]
        for (name, r), color in zip(results.items(), colors):
            hits = r["hits"]
            rolling = [
                sum(hits[max(0, i - window): i + 1]) / min(i + 1, window)
                for i in range(0, len(hits), 50)
            ]
            plt.plot(rolling, label=name, color=color)

        plt.title("Hit Rate (rolling avg per 500 requests)")
        plt.xlabel("Request batch (×50)")
        plt.ylabel("Hit Rate")
        plt.show()

        # ── AMAT Bar ──────────────────────────────────────────────────────────
        console.print(Panel("[bold]AMAT Comparison[/bold]", style="dim"))
        names  = list(results.keys())
        amats  = [results[n]["amat"] for n in names]
        plt.bar(names, amats)
        plt.title("Average Memory Access Time (ns) — lower is better")
        plt.ylabel("ns")
        plt.show()

        # ── Summary ───────────────────────────────────────────────────────────
        best = results[best_name]
        vs_lru_delta = (best["hit_rate"] - results["LRU"]["hit_rate"]) * 100

        console.print(Panel(
            f"[bold green]✓ Best Hit Rate: {best_name}[/bold green]  "
            f"[yellow]{best['hit_rate']:.2%}[/yellow]  "
            f"([cyan]+{vs_lru_delta:.2f}%[/cyan] vs LRU)\n"
            f"[bold green]✓ Best DT: {best_dt}[/bold green]  "
            f"[yellow]{results[best_dt]['dt']:,.0f} ns[/yellow]  "
            f"AMAT: [yellow]{results[best_dt]['amat']:.2f} ns[/yellow]",
            title="Result", border_style="green"
        ))

    else:
        for name, r in results.items():
            print(f"{name}: Hit={r['hit_rate']:.2%}  AMAT={r['amat']:.2f}ns  "
                  f"DT={r['dt']:,.0f}ns")

    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if RICH_OK:
        console.print(Panel(
            "[bold cyan]Predictive Cache — DES Simulator v2[/bold cyan]\n"
            f"Strategies: LRU · LFU · Predictive (RF) · Markov Chain\n"
            f"Cache size={CACHE_SIZE}  |  Simulating {SIM_ROWS:,} requests",
            border_style="cyan"
        ))

    # ── Load models ───────────────────────────────────────────────────────────
    missing = []
    for p in [RF_MODEL_PATH, MK_MODEL_PATH, ENCODER_PATH]:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        print(f"[ERROR] Thiếu model files: {missing}")
        print("  Chạy: python train_model.py && python markov_model.py")
        exit(1)

    with open(RF_MODEL_PATH,  "rb") as f: rf_clf = pickle.load(f)
    with open(MK_MODEL_PATH,  "rb") as f: mk_clf = pickle.load(f)
    with open(ENCODER_PATH,   "rb") as f: le     = pickle.load(f)
    if RICH_OK:
        console.print("[green]✓ Models loaded[/green]")

    # ── Load data ─────────────────────────────────────────────────────────────
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"[ERROR] Không có CSV trong {DATA_DIR}")
        exit(1)
    dfs = [pd.read_csv(f, parse_dates=["Session_ID"]) for f in sorted(csv_files)]
    df  = pd.concat(dfs, ignore_index=True).sort_values("Session_ID").reset_index(drop=True)
    if RICH_OK:
        console.print(f"[green]✓ Data loaded:[/green] {len(df):,} rows")

    # ── Simulate ──────────────────────────────────────────────────────────────
    stats, blocks, offsets = run_simulation(df, rf_clf, mk_clf, le)

    # ── Report ────────────────────────────────────────────────────────────────
    results = print_report(stats, blocks, offsets)

    # ── Save results JSON for analysis_report.py ──────────────────────────────
    import json
    os.makedirs("./report", exist_ok=True)
    summary = {
        name: {
            "hit_rate": float(r["hit_rate"]),
            "amat"    : float(r["amat"]),
            "dt"      : float(r["dt"]),
            "n_hits"  : int(sum(r["hits"])),
            "n_total" : len(r["hits"]),
        }
        for name, r in results.items()
    }
    with open("./report/sim_results.json", "w") as f:
        json.dump({"config": {"cache_size": CACHE_SIZE, "sim_rows": SIM_ROWS},
                   "results": summary}, f, indent=2)
    if RICH_OK:
        console.print("[dim]Results saved → ./report/sim_results.json[/dim]")