
from typing import Dict
import numpy as np

def _ensure_1d(a):
    a = np.asarray(a).reshape(-1)
    return a

def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_true = _ensure_1d(y_true)
    y_prob = _ensure_1d(y_prob)
    y_hat = (y_prob >= threshold).astype(int)
    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    tn = int(((y_hat == 0) & (y_true == 0)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(len(y_true), 1)
    prec1 = tp / max((tp + fp), 1)
    rec1 = tp / max((tp + fn), 1)
    prec0 = tn / max((tn + fn), 1)
    rec0 = tn / max((tn + fp), 1)
    # AUROC and AUPRC try sklearn, else fallback
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
        f1 = float(f1_score(y_true, y_hat))
    except Exception:
        auroc = _roc_auc(y_true, y_prob)
        auprc = _pr_auc(y_true, y_prob)
        f1 = 2 * prec1 * rec1 / max((prec1 + rec1), 1e-8)
    return {
        "acc": float(acc),
        "prec0": float(prec0),
        "prec1": float(prec1),
        "rec0": float(rec0),
        "rec1": float(rec1),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(f1),
    }

def _roc_auc(y_true, y_prob):
    # simple Mann-Whitney U estimate
    y_true = _ensure_1d(y_true)
    y_prob = _ensure_1d(y_prob)
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    sum_ranks_pos = ranks[: len(pos)].sum() + len(pos)  # 1-based
    U = sum_ranks_pos - len(pos) * (len(pos) + 1) / 2
    return float(U / (len(pos) * len(neg)))

def _pr_auc(y_true, y_prob, points: int = 200):
    y_true = _ensure_1d(y_true)
    y_prob = _ensure_1d(y_prob)
    thresholds = np.linspace(0.0, 1.0, points)
    prec, rec = [], []
    for th in thresholds:
        y_hat = (y_prob >= th).astype(int)
        tp = ((y_hat == 1) & (y_true == 1)).sum()
        fp = ((y_hat == 1) & (y_true == 0)).sum()
        fn = ((y_hat == 0) & (y_true == 1)).sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        prec.append(p)
        rec.append(r)
    # integrate precision with respect to recall
    order = np.argsort(rec)
    rec = np.array(rec)[order]
    prec = np.array(prec)[order]
    return float(np.trapz(prec, rec))
