"""
Markov Chain Model — định nghĩa class dùng chung.
File này KHÔNG được chạy trực tiếp — chỉ để import.
Cả markov_model.py (lúc train) và simulate_cache.py (lúc load .pkl)
đều phải import class từ ĐÚNG file này để pickle hoạt động chính xác.
"""

import numpy as np
from collections import defaultdict


class MarkovChainModel:
    """
    Mô hình Markov bậc 1: P(next_block | current_block).
    Hỗ trợ predict_proba() để plug-in thẳng vào PredictiveCache,
    tương thích với cùng interface như sklearn classifier.
    """

    def __init__(self, order=1):
        self.order = order          # bậc của Markov (1 = dựa vào 1 block trước)
        self.transition_ = {}       # {state: {next_state: count}}
        self.classes_ = None        # danh sách class (block_enc) theo thứ tự
        self.n_classes_ = 0
        self._proba_cache = {}      # cache row đã normalize

    # ── Train ──────────────────────────────────────────────────────────────────
    def fit(self, X, y):
        """
        X: array (n_samples, window_size+) — chỉ dùng cột cuối (block liền trước)
        y: array (n_samples,) — block tiếp theo (encoded)
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        for prev, nxt in zip(X[:, -1].astype(int), y.astype(int)):
            if prev not in self.transition_:
                self.transition_[prev] = defaultdict(int)
            self.transition_[prev][nxt] += 1

        print(f"  Markov: {len(self.transition_)} unique states, "
              f"{self.n_classes_} unique targets")
        return self

    def _get_proba_row(self, state):
        if state in self._proba_cache:
            return self._proba_cache[state]

        row = np.zeros(self.n_classes_)
        if state in self.transition_:
            for next_enc, cnt in self.transition_[state].items():
                idx = np.searchsorted(self.classes_, next_enc)
                if idx < self.n_classes_ and self.classes_[idx] == next_enc:
                    row[idx] = cnt
            total = row.sum()
            if total > 0:
                row /= total
            else:
                row[:] = 1.0 / self.n_classes_
        else:
            row[:] = 1.0 / self.n_classes_

        self._proba_cache[state] = row
        return row

    def predict_proba(self, X):
        result = np.zeros((len(X), self.n_classes_))
        for i, row in enumerate(X):
            state = int(row[-1])
            result[i] = self._get_proba_row(state)
        return result

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)