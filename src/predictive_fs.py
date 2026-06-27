#!/usr/bin/env python3
"""
PredictiveFS — FUSE passthrough filesystem với predictive caching.

Đây là "develop a file system" đúng nghĩa: mount một filesystem thật,
chặn (intercept) mọi request đọc file LIVE, và dùng model ML để
dự đoán + prefetch file tiếp theo vào RAM cache.

KHÁC với simulate_cache.py (offline DES trên trace có sẵn):
  - PredictiveFS chạy LIVE, chặn request thật của ứng dụng thật
  - Áp dụng prediction NGAY khi request đến, không phải replay log

Cách dùng:
  python predictive_fs.py <source_dir> <mountpoint>

Ví dụ:
  mkdir -p ~/pfs_source ~/pfs_mount
  echo "hello" > ~/pfs_source/a.txt
  python predictive_fs.py ~/pfs_source ~/pfs_mount
  # ở terminal khác:
  cat ~/pfs_mount/a.txt
  # Ctrl+C ở terminal chạy predictive_fs.py để unmount, hoặc:
  fusermount -u ~/pfs_mount

Yêu cầu: pip install fusepy --break-system-packages
         sudo apt install fuse3   (nếu chưa có /dev/fuse trên WSL)
"""

import os
import sys
import csv
import time
import errno
import pickle
import threading
from collections import OrderedDict, defaultdict

try:
    from fuse import FUSE, FuseOSError, Operations
except ImportError:
    print("[ERROR] Thiếu fusepy. Chạy: pip install fusepy --break-system-packages")
    sys.exit(1)


# ─── CONFIG ───────────────────────────────────────────────────────────────────
CACHE_CAPACITY_BYTES = 64 * 1024 * 1024   # 64 MB RAM cache cho prefetch
ACCESS_LOG_PATH      = "./access_log.csv"
PREFETCH_TOP_N       = 2                   # số file dự đoán để prefetch mỗi lần
MIN_HISTORY_TO_PREDICT = 1                 # cần ít nhất 1 access trước đó để dự đoán


# ─── SIMPLE FILE-LEVEL MARKOV PREDICTOR ───────────────────────────────────────
class FileAccessPredictor:
    """
    Markov bậc 1 ở mức TÊN FILE (không phải block/sector như simulate_cache.py).
    Học online: mỗi lần một file được đọc, cập nhật P(next_file | current_file).
    Đơn giản, nhẹ, phù hợp chạy live trong vòng read() của FUSE mà không
    block ứng dụng đang đọc.
    """

    def __init__(self):
        self.transition = defaultdict(lambda: defaultdict(int))  # {file: {next_file: count}}
        self.last_accessed = None
        self.lock = threading.Lock()

    def record_access(self, filepath):
        """Cập nhật transition matrix khi filepath vừa được đọc."""
        with self.lock:
            if self.last_accessed is not None and self.last_accessed != filepath:
                self.transition[self.last_accessed][filepath] += 1
            self.last_accessed = filepath

    def predict_next(self, filepath, top_n=PREFETCH_TOP_N):
        """Trả về danh sách top-N file có khả năng được đọc tiếp theo."""
        with self.lock:
            candidates = self.transition.get(filepath)
            if not candidates:
                return []
            ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
            return [f for f, _ in ranked[:top_n]]

    def save(self, path):
        with self.lock:
            with open(path, "wb") as fh:
                pickle.dump(dict(self.transition), fh)

    def load(self, path):
        if os.path.exists(path):
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            with self.lock:
                self.transition = defaultdict(lambda: defaultdict(int))
                for k, v in data.items():
                    self.transition[k] = defaultdict(int, v)


# ─── RAM CACHE (LRU theo dung lượng) ──────────────────────────────────────────
class RAMCache:
    """LRU cache đơn giản, giới hạn theo tổng số bytes thay vì số entry."""

    def __init__(self, capacity_bytes):
        self.capacity = capacity_bytes
        self.data = OrderedDict()   # {filepath: bytes}
        self.current_size = 0
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.prefetches = 0

    def get(self, filepath):
        with self.lock:
            if filepath in self.data:
                self.data.move_to_end(filepath)
                self.hits += 1
                return self.data[filepath]
            self.misses += 1
            return None

    def put(self, filepath, content, is_prefetch=False):
        with self.lock:
            size = len(content)
            if size > self.capacity:
                return  # file quá lớn để cache, bỏ qua
            if filepath in self.data:
                self.current_size -= len(self.data[filepath])
                del self.data[filepath]
            while self.current_size + size > self.capacity and self.data:
                _, evicted = self.data.popitem(last=False)
                self.current_size -= len(evicted)
            self.data[filepath] = content
            self.current_size += size
            if is_prefetch:
                self.prefetches += 1

    def stats(self):
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "prefetches": self.prefetches,
                "hit_rate": hit_rate,
                "cached_files": len(self.data),
                "cache_used_mb": self.current_size / (1024 * 1024),
            }


# ─── FUSE PASSTHROUGH FILESYSTEM ──────────────────────────────────────────────
class PredictiveFS(Operations):
    """
    Passthrough FUSE filesystem: mọi thao tác được chuyển tiếp tới source_dir,
    nhưng read() được chặn để:
      1. Thử trả từ RAM cache trước (nếu đã được prefetch)
      2. Nếu miss, đọc từ đĩa thật như bình thường
      3. Log access, cập nhật predictor, prefetch file dự đoán kế tiếp
    """

    def __init__(self, source_dir):
        self.source_dir = os.path.abspath(source_dir)
        self.cache = RAMCache(CACHE_CAPACITY_BYTES)
        self.predictor = FileAccessPredictor()
        self._init_access_log()
        print(f"[PredictiveFS] Passthrough gốc: {self.source_dir}")
        print(f"[PredictiveFS] RAM cache: {CACHE_CAPACITY_BYTES // (1024*1024)} MB")

    def _init_access_log(self):
        is_new = not os.path.exists(ACCESS_LOG_PATH)
        self.log_file = open(ACCESS_LOG_PATH, "a", newline="")
        self.log_writer = csv.writer(self.log_file)
        if is_new:
            self.log_writer.writerow(["timestamp", "filepath", "size_bytes", "cache_hit"])
        self.log_lock = threading.Lock()

    def _full_path(self, partial):
        if partial.startswith("/"):
            partial = partial[1:]
        return os.path.join(self.source_dir, partial)

    def _log_access(self, filepath, size, cache_hit):
        with self.log_lock:
            self.log_writer.writerow([time.time(), filepath, size, int(cache_hit)])
            self.log_file.flush()

    def _prefetch_async(self, filepath):
        """Chạy prefetch trong background thread để không block read() hiện tại."""
        def _worker():
            predicted = self.predictor.predict_next(filepath)
            for pred_file in predicted:
                full = self._full_path(pred_file)
                if os.path.exists(full) and os.path.isfile(full):
                    try:
                        with open(full, "rb") as fh:
                            content = fh.read()
                        self.cache.put(pred_file, content, is_prefetch=True)
                    except OSError:
                        pass
        threading.Thread(target=_worker, daemon=True).start()

    # ── Filesystem metadata operations (passthrough thuần) ─────────────────────
    def getattr(self, path, fh=None):
        full = self._full_path(path)
        if not os.path.exists(full):
            raise FuseOSError(errno.ENOENT)
        st = os.lstat(full)
        return {key: getattr(st, key) for key in (
            "st_atime", "st_ctime", "st_gid", "st_mode",
            "st_mtime", "st_nlink", "st_size", "st_uid")}

    def readdir(self, path, fh):
        full = self._full_path(path)
        entries = [".", ".."]
        if os.path.isdir(full):
            entries += os.listdir(full)
        for e in entries:
            yield e

    def open(self, path, flags):
        full = self._full_path(path)
        if not os.path.exists(full):
            raise FuseOSError(errno.ENOENT)
        return os.open(full, flags)

    def release(self, path, fh):
        return os.close(fh)

    # ── READ — đây là nơi predictive caching xảy ra ─────────────────────────────
    def read(self, path, length, offset, fh):
        rel_path = path[1:] if path.startswith("/") else path

        # 1. Thử cache trước
        cached = self.cache.get(rel_path)
        if cached is not None:
            self._log_access(rel_path, len(cached), cache_hit=True)
            self.predictor.record_access(rel_path)
            self._prefetch_async(rel_path)
            return cached[offset:offset + length]

        # 2. Cache miss → đọc thật từ đĩa (passthrough)
        os.lseek(fh, offset, os.SEEK_SET)
        data = os.read(fh, length)

        # Nếu đọc toàn bộ file (offset=0, length đủ lớn), cache lại để lần sau hit
        full = self._full_path(path)
        try:
            file_size = os.path.getsize(full)
            if offset == 0 and length >= file_size:
                self.cache.put(rel_path, data, is_prefetch=False)
        except OSError:
            file_size = -1

        self._log_access(rel_path, len(data), cache_hit=False)
        self.predictor.record_access(rel_path)
        self._prefetch_async(rel_path)

        return data

    def write(self, path, data, offset, fh):
        os.lseek(fh, offset, os.SEEK_SET)
        result = os.write(fh, data)
        rel_path = path[1:] if path.startswith("/") else path
        self.cache.data.pop(rel_path, None)  # invalidate cache nếu file bị ghi
        return result

    def truncate(self, path, length, fh=None):
        full = self._full_path(path)
        with open(full, "r+") as f:
            f.truncate(length)

    def create(self, path, mode, fi=None):
        full = self._full_path(path)
        return os.open(full, os.O_WRONLY | os.O_CREAT, mode)

    def unlink(self, path):
        full = self._full_path(path)
        os.unlink(full)

    def mkdir(self, path, mode):
        full = self._full_path(path)
        os.mkdir(full, mode)

    def rmdir(self, path):
        full = self._full_path(path)
        os.rmdir(full)

    def destroy(self, path):
        """Gọi khi unmount — in thống kê và lưu predictor."""
        stats = self.cache.stats()
        print("\n" + "=" * 50)
        print("  PredictiveFS — Thống kê phiên làm việc")
        print("=" * 50)
        print(f"  Cache hits      : {stats['hits']}")
        print(f"  Cache misses    : {stats['misses']}")
        print(f"  Hit rate        : {stats['hit_rate']:.2%}")
        print(f"  Files prefetched: {stats['prefetches']}")
        print(f"  RAM cache used  : {stats['cache_used_mb']:.2f} MB")
        print(f"  Access log saved: {ACCESS_LOG_PATH}")
        self.predictor.save("./model/file_predictor.pkl")
        self.log_file.close()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Cách dùng: python predictive_fs.py <source_dir> <mountpoint>")
        print("Ví dụ:")
        print("  mkdir -p ~/pfs_source ~/pfs_mount")
        print("  python predictive_fs.py ~/pfs_source ~/pfs_mount")
        sys.exit(1)

    source_dir = sys.argv[1]
    mountpoint = sys.argv[2]

    if not os.path.isdir(source_dir):
        print(f"[ERROR] source_dir không tồn tại: {source_dir}")
        sys.exit(1)
    if not os.path.isdir(mountpoint):
        print(f"[ERROR] mountpoint không tồn tại, tạo trước bằng: mkdir -p {mountpoint}")
        sys.exit(1)

    os.makedirs("./model", exist_ok=True)
    fs = PredictiveFS(source_dir)
    fs.predictor.load("./model/file_predictor.pkl")

    print(f"[PredictiveFS] Mounting tại {mountpoint} ... (Ctrl+C để unmount)")
    FUSE(fs, mountpoint, foreground=True, nothreads=False)