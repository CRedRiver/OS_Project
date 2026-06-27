#!/usr/bin/env python3
"""
Benchmark I/O thực tế qua PredictiveFS.
Đo thời gian đọc lặp lại một tập file theo pattern tuần tự,
để xem prefetch có thực sự giảm thời gian truy cập hay không.

Cách dùng (SAU KHI đã mount predictive_fs.py ở terminal khác):
  python benchmark_fs.py <mountpoint>

Ví dụ:
  Terminal 1: python predictive_fs.py ~/pfs_source ~/pfs_mount
  Terminal 2: python benchmark_fs.py ~/pfs_mount
"""

import os
import sys
import time
import random


def create_test_files(source_dir, n_files=20, size_kb=512):
    """Tạo bộ file test nếu source_dir còn trống."""
    os.makedirs(source_dir, exist_ok=True)
    existing = [f for f in os.listdir(source_dir) if f.startswith("file_")]
    if len(existing) >= n_files:
        print(f"[*] Đã có {len(existing)} file test, bỏ qua tạo mới.")
        return

    print(f"[*] Tạo {n_files} file test ({size_kb} KB mỗi file) tại {source_dir}...")
    for i in range(n_files):
        path = os.path.join(source_dir, f"file_{i:03d}.bin")
        with open(path, "wb") as f:
            f.write(os.urandom(size_kb * 1024))
    print("[*] Xong.")


def run_benchmark(mountpoint, n_files=20, n_rounds=3, sequential=True):
    """
    Đọc lặp lại n_files theo thứ tự cố định, n_rounds lần.
    Vòng 1: cold (chưa có gì trong cache, không có pattern để predict)
    Vòng 2+: predictor đã học pattern từ vòng 1, prefetch nên có hit
    """
    files = sorted(f for f in os.listdir(mountpoint) if f.startswith("file_"))[:n_files]
    if not files:
        print(f"[ERROR] Không tìm thấy file test trong {mountpoint}")
        print("  Chạy create_test_files() trước, hoặc kiểm tra mountpoint đúng chưa.")
        return

    print(f"[*] Benchmark: {len(files)} file, {n_rounds} vòng, "
          f"thứ tự {'tuần tự' if sequential else 'ngẫu nhiên'}")
    print(f"{'Vòng':<8}{'Tổng thời gian (s)':<22}{'Trung bình/file (ms)':<22}")
    print("-" * 52)

    round_times = []
    for round_num in range(1, n_rounds + 1):
        order = files if sequential else random.sample(files, len(files))
        start = time.perf_counter()
        for fname in order:
            full_path = os.path.join(mountpoint, fname)
            with open(full_path, "rb") as f:
                f.read()
        elapsed = time.perf_counter() - start
        round_times.append(elapsed)
        avg_ms = (elapsed / len(files)) * 1000
        print(f"{round_num:<8}{elapsed:<22.4f}{avg_ms:<22.2f}")

    if len(round_times) >= 2:
        improvement = (round_times[0] - round_times[-1]) / round_times[0] * 100
        print("-" * 52)
        print(f"[KẾT QUẢ] Vòng cuối nhanh hơn vòng đầu: {improvement:.1f}%")
        if improvement > 0:
            print("  → Predictive prefetch đang hoạt động: pattern lặp lại")
            print("    được học và cache trước, giảm thời gian đọc thực tế.")
        else:
            print("  → Chưa thấy cải thiện rõ — có thể do file quá nhỏ,")
            print("    cache miss nhiều, hoặc cần nhiều vòng lặp hơn để model học.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Cách dùng: python benchmark_fs.py <mountpoint>")
        sys.exit(1)

    mountpoint = sys.argv[1]
    if not os.path.isdir(mountpoint):
        print(f"[ERROR] mountpoint không tồn tại: {mountpoint}")
        sys.exit(1)

    run_benchmark(mountpoint, n_files=20, n_rounds=3, sequential=True)