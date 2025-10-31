
"""
Robust binary metrics for ConCare training.

Exposes:
    - binary_metrics(y_true, y_prob)
    - threshold_metrics(y_true, y_prob, thr)
    - minpse(y_true, y_prob, *, num_thresholds=1001)

Notes:
  * Inputs are 1-D numpy arrays (or array-likes).
  * Handles degenerate cases gracefully (all-positive or all-negative labels).
  * Clamps probabilities into [0, 1] and replaces NaNs/Infs.
  * MinPSE = max_t min(Precision(t), Sensitivity/Recall(t)), i.e., the best
    achievable operating point where both precision and recall are simultaneously high.
"""
from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    _HAVE_SK = True
except Exception:  # very defensive
    _HAVE_SK = False


# ---------- utils ----------

def _to_1d_float(a):
    a = np.asarray(a).reshape(-1).astype(np.float64, copy=False)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    return a


def _sanitize_probs(p):
    p = _to_1d_float(p)
    # If logits were mistakenly passed, apply a numerically safe sigmoid
    if p.min() < 0.0 or p.max() > 1.0:
        p = 1.0 / (1.0 + np.exp(-np.clip(p, -80.0, 80.0)))
    # Finally clamp to [0, 1]
    return np.clip(p, 0.0, 1.0)


def _sanitize_labels(y):
    y = _to_1d_float(y)
    # Accept {0,1}, {False,True}, or probabilities close to 0/1
    return (y >= 0.5).astype(np.int32)


def _degenerate_defaults(y_true):
    n = y_true.size
    pos = int(y_true.sum())
    prevalence = pos / max(n, 1)
    return {
        "auroc": 0.5,                 # random classifier
        "auprc": float(prevalence)    # expected AP equals prevalence under random ranking
    }


# ---------- public APIs ----------

def binary_metrics(y_true, y_prob):
    """Compute AUROC and AUPRC from probabilities.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground truth labels in {0,1} (or probabilities close to 0/1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        {'auroc': float, 'auprc': float}
    """
    y_true = _sanitize_labels(y_true)
    y_prob = _sanitize_probs(y_prob)

    n = y_true.size
    pos = int(y_true.sum())
    neg = int(n - pos)

    # Degenerate cases: all pos or all neg
    if pos == 0 or neg == 0:
        return _degenerate_defaults(y_true)

    metrics = {}

    # AUROC
    try:
        if _HAVE_SK:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        else:
            # Minimal fallback via rank-based AUC (Mann–Whitney U)
            order = np.argsort(y_prob)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, n + 1, dtype=np.float64)
            r_pos = ranks[y_true == 1].sum()
            auc = (r_pos - pos * (pos + 1) / 2.0) / (pos * neg)
            metrics["auroc"] = float(auc)
    except Exception:
        metrics["auroc"] = 0.5

    # AUPRC (Average Precision)
    try:
        if _HAVE_SK:
            metrics["auprc"] = float(average_precision_score(y_true, y_prob))
        else:
            # Simple PR area approximation via interpolation
            idx = np.argsort(-y_prob)
            y_sorted = y_true[idx]
            tp = 0.0
            fp = 0.0
            precisions = []
            recalls = []
            for i in range(n):
                if y_sorted[i] == 1:
                    tp += 1.0
                else:
                    fp += 1.0
                precisions.append(tp / max(tp + fp, 1.0))
                recalls.append(tp / pos if pos > 0 else 0.0)
            # Compute area under PR using rectangle rule
            ap = 0.0
            prev_r = 0.0
            for p, r in zip(precisions, recalls):
                ap += p * max(r - prev_r, 0.0)
                prev_r = r
            metrics["auprc"] = float(ap)
    except Exception:
        metrics["auprc"] = float(pos / max(n, 1))

    return metrics


def threshold_metrics(y_true, y_prob, thr: float):
    """Compute thresholded metrics at a given probability threshold.

    Returns
    -------
    dict with keys: acc, precision, recall, f1, specificity, tp, fp, tn, fn, thr
    """
    y_true = _sanitize_labels(y_true)
    y_prob = _sanitize_probs(y_prob)
    thr = float(thr)

    y_pred = (y_prob >= thr).astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "thr": thr,
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def minpse(y_true, y_prob, *, num_thresholds: int = 1001):
    """Compute MinPSE = max_t min(Precision(t), Recall/Sensitivity(t)).

    Also returns the threshold that attains the maximum.

    Parameters
    ----------
    y_true : array-like
    y_prob : array-like
    num_thresholds : int, optional
        If sklearn is unavailable, evaluate on this many evenly-spaced thresholds in [0,1].

    Returns
    -------
    dict
        {'minpse': float, 'best_thr': float, 'precision': float, 'recall': float}
    """
    y_true = _sanitize_labels(y_true)
    y_prob = _sanitize_probs(y_prob)

    pos = int(y_true.sum())
    neg = int(y_true.size - pos)
    if pos == 0 or neg == 0:
        # With a single class, precision/recall curves are undefined in a useful way.
        # Define MinPSE to 0.0 and select mid threshold.
        return {"minpse": 0.0, "best_thr": 0.5, "precision": 0.0, "recall": 0.0}

    if _HAVE_SK:
        # sklearn returns precision, recall for thresholds sorted increasing
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        # precision_recall_curve returns an extra (P=1 at R=0) point;
        # thresholds has length len(prec)-1. Align by discarding the first point.
        prec_ = prec[1:]
        rec_ = rec[1:]
        if thr.size == 0:
            # all scores identical; pick thr=0.5
            p = float(prec[-1])
            r = float(rec[-1])
            return {"minpse": float(min(p, r)), "best_thr": 0.5, "precision": p, "recall": r}
        m = np.minimum(prec_, rec_)
        i = int(np.argmax(m))
        best_thr = float(thr[i])
        return {"minpse": float(m[i]), "best_thr": best_thr, "precision": float(prec_[i]), "recall": float(rec_[i])}
    else:
        # Fallback: scan thresholds uniformly in [0,1]
        best = {"minpse": -1.0, "best_thr": 0.5, "precision": 0.0, "recall": 0.0}
        for thr in np.linspace(0.0, 1.0, num_thresholds):
            tm = threshold_metrics(y_true, y_prob, thr)
            m = min(tm["precision"], tm["recall"])
            if m > best["minpse"]:
                best = {"minpse": float(m), "best_thr": float(thr),
                        "precision": float(tm["precision"]), "recall": float(tm["recall"])}
        return best


__all__ = [
    "binary_metrics",
    "threshold_metrics",
    "minpse",
]
