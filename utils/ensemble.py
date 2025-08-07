# ensemble.py

import numpy as np

class AverageEnsemble:
    """
    最简单的“概率平均”集成：
    pipelines: list of 已 fit 好的 Pipeline，每个 pipeline 必须有 predict_proba() 方法。
    threshold: 二分类阈值，默认 0.5，predict() 直接根据 avg_prob >= threshold 判正类。
    """
    def __init__(self, pipelines, threshold=0.5):
        if len(pipelines) == 0:
            raise ValueError("pipelines 列表不能为空")
        self.pipelines = pipelines
        self.threshold = threshold

    def predict_proba(self, X):
        probas = []
        for mdl in self.pipelines:
            p_i = mdl.predict_proba(X)[:, 1].reshape(-1, 1)
            probas.append(p_i)
        probas_mat = np.hstack(probas)
        avg_pos = np.mean(probas_mat, axis=1)
        avg_neg = 1.0 - avg_pos
        return np.vstack([avg_neg, avg_pos]).T

    def predict(self, X):
        avg_pos = self.predict_proba(X)[:, 1]
        return (avg_pos >= self.threshold).astype(int)
