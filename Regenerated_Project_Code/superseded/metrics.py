# metrics.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


@dataclass
class BinaryMetrics:
    auroc: float
    auprc: float
    acc: Optional[float] = None
    prec1: Optional[float] = None
    rec1: Optional[float] = None
    f1: Optional[float] = None
    prec0: Optional[float] = None
    rec0: Optional[float] = None
    thr: Optional[float] = None
    loss: Optional[float] = None


def _ensure_prob1(pred: np.ndarray) -> np.ndarray:
    """
    Returns the probability for the positive class p1 with shape (N,).
    Accepts shapes:
      - (N,) interpreted as p1
      - (N, 1) interpreted as p1
      - (N, 2) interpreted as [p0, p1]
    """
    pred = np.asarray(pred)
    if pred.ndim == 1:
        return pred
    if pred.ndim == 2 and pred.shape[1] == 1:
        return pred[:, 0]
    if pred.ndim == 2 and pred.shape[1] == 2:
        return pred[:, 1]
    raise ValueError(f"Unsupported prediction shape {pred.shape}. Expect (N,), (N,1), or (N,2).")


def _stack_p0p1(p1: np.ndarray) -> np.ndarray:
    """
    Build a two column array [p0, p1] from p1.
    """
    p1 = np.asarray(p1).reshape(-1)
    p0 = 1.0 - p1
    return np.stack([p0, p1], axis=1)


def get_binary_metrics(
    y_true: Iterable[int],
    y_pred: np.ndarray,
    *,
    papers_metrics_mode: bool = False,
    find_best_f1_threshold: bool = True,
    provided_loss: Optional[float] = None,
) -> BinaryMetrics:
    """
    Compute AUROC and AUPRC for binary classification.
    Two modes are supported.

    papers_metrics_mode=True:
      - Match the authors’ script style
      - Convert predictions to two columns [p0, p1] if needed
      - Compute labels with argmax([p0, p1]) which is equivalent to p1 >= 0.5
      - Report acc, prec1, rec1, f1 from those labels
      - Also report prec0 and rec0 from the confusion matrix

    papers_metrics_mode=False:
      - Use p1 to compute AUROC and AUPRC
      - Optionally select a threshold that maximizes F1 on the validation set
      - Then report acc, prec1, rec1, f1 at that threshold
      - Also report prec0 and rec0
    """
    y_true = np.asarray(list(y_true)).astype(int)
    p1 = _ensure_prob1(np.asarray(y_pred))

    # Always compute threshold free metrics with p1
    auroc = float(roc_auc_score(y_true, p1)) if len(np.unique(y_true)) > 1 else 0.0
    auprc = float(average_precision_score(y_true, p1)) if len(np.unique(y_true)) > 1 else 0.0

    if papers_metrics_mode:
        # Build two column scores and predict via argmax like the authors
        probs2 = _stack_p0p1(p1)
        y_hat = probs2.argmax(axis=1)
        cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        acc = float(accuracy_score(y_true, y_hat))
        prec1 = float(precision_score(y_true, y_hat, pos_label=1, zero_division=0))
        rec1 = float(recall_score(y_true, y_hat, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true, y_hat, pos_label=1, zero_division=0))
        # Class 0 metrics
        prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # precision for class 0
        rec0  = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # recall for class 0

        return BinaryMetrics(
            auroc=auroc,
            auprc=auprc,
            acc=acc,
            prec1=prec1,
            rec1=rec1,
            f1=f1,
            prec0=prec0,
            rec0=rec0,
            thr=0.5,  # implicit from argmax
            loss=provided_loss,
        )

    # Baseline mode with optional best F1 threshold search
    if find_best_f1_threshold:
        pr_thr, best_f1 = _best_f1_threshold(y_true, p1)
        thr = pr_thr
    else:
        thr = 0.5

    y_hat = (p1 >= thr).astype(int)
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    acc = float(accuracy_score(y_true, y_hat))
    prec1 = float(precision_score(y_true, y_hat, pos_label=1, zero_division=0))
    rec1 = float(recall_score(y_true, y_hat, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_hat, pos_label=1, zero_division=0))
    prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return BinaryMetrics(
        auroc=auroc,
        auprc=auprc,
        acc=acc,
        prec1=prec1,
        rec1=rec1,
        f1=f1,
        prec0=prec0,
        rec0=rec0,
        thr=float(thr),
        loss=provided_loss,
    )


def _best_f1_threshold(y_true: np.ndarray, p1: np.ndarray) -> Tuple[float, float]:
    """
    Search for the probability threshold that maximizes F1 on the given data.
    Ties are broken by choosing the smallest threshold that achieves the best F1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, p1)
    # precision_recall_curve returns precision and recall for thresholds plus
    # an extra point at threshold infinity. Align lengths accordingly.
    f1s = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], a_min=1e-12, a_max=None)
    if f1s.size == 0:
        return 0.5, 0.0
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def print_binary_classification_report(
    metrics_obj: BinaryMetrics,
    header: Optional[str] = None,
    *,
    papers_metrics_mode: bool = False,
) -> str:
    """
    Return a formatted string. Keep names consistent with our logs.
    """
    lines = []
    if header:
        lines.append(header)

    # Threshold free block
    lines.append("Threshold free")
    lines.append(f"   auroc: {metrics_obj.auroc:.4f}")
    lines.append(f"   auprc: {metrics_obj.auprc:.4f}")
    if metrics_obj.loss is not None:
        lines.append(f"    loss: {metrics_obj.loss:.4f}")

    # Thresholded block
    if papers_metrics_mode:
        thr_note = "0.50 via argmax"
    else:
        thr_note = f"{metrics_obj.thr:.2f}" if metrics_obj.thr is not None else "n/a"
    lines.append(f"At thr={thr_note}")
    if metrics_obj.acc is not None:
        lines.append(f"    acc: {metrics_obj.acc:.4f}")
    if metrics_obj.prec0 is not None:
        lines.append(f"  prec0: {metrics_obj.prec0:.4f}")
    if metrics_obj.prec1 is not None:
        lines.append(f"  prec1: {metrics_obj.prec1:.4f}")
    if metrics_obj.rec0 is not None:
        lines.append(f"   rec0: {metrics_obj.rec0:.4f}")
    if metrics_obj.rec1 is not None:
        lines.append(f"   rec1: {metrics_obj.rec1:.4f}")
    if metrics_obj.f1 is not None:
        lines.append(f"     f1: {metrics_obj.f1:.4f}")

    return "\n".join(lines)


# Convenience helpers used by training loops

def compute_and_log_metrics(
    y_true: Iterable[int],
    y_pred_probs: np.ndarray,
    *,
    papers_metrics_mode: bool = False,
    find_best_f1_threshold: bool = True,
    provided_loss: Optional[float] = None,
) -> Tuple[BinaryMetrics, str]:
    """
    Compute metrics and return both the object and a formatted string.
    """
    m = get_binary_metrics(
        y_true,
        y_pred_probs,
        papers_metrics_mode=papers_metrics_mode,
        find_best_f1_threshold=find_best_f1_threshold,
        provided_loss=provided_loss,
    )
    txt = print_binary_classification_report(m, papers_metrics_mode=papers_metrics_mode)
    return m, txt
