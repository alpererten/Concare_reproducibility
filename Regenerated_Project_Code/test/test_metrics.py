import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np

try:
    from sklearn import metrics as _sk_metrics  # noqa: F401
    _HAVE_SK = True
except Exception:  # pragma: no cover - defensive guard
    _HAVE_SK = False


_ROOT = Path(__file__).resolve().parents[2]
_AUTHORS_METRICS_PATH = _ROOT / "Regenerated_Project_Code" / "metrics_authors.py"
_REGEN_METRICS_PATH = _ROOT / "Regenerated_Project_Code" / "metrics.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _format_diff(name: str, ours: float, authors: float) -> str:
    delta = ours - authors
    return f"{name}: ours={ours:.6f}, authors={authors:.6f}, delta={delta:+.3e}"


def _is_close(a: float, b: float, *, rel_tol: float = 1e-6, abs_tol: float = 1e-8) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


@unittest.skipUnless(_HAVE_SK, "scikit-learn metrics are required for parity comparison")
class MetricsParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.authors_metrics = _load_module("authors_metrics", _AUTHORS_METRICS_PATH)
        cls.regen_metrics = _load_module("regenerated_metrics", _REGEN_METRICS_PATH)

        rng = np.random.default_rng(seed=1337)
        y_true = rng.integers(0, 2, size=512).astype(np.int32)
        y_prob = rng.random(size=512)

        # Guarantee both classes appear; the legacy code assumes that.
        if y_true.sum() == 0:
            y_true[0] = 1
        if y_true.sum() == y_true.size:
            y_true[0] = 0

        cls.y_true = y_true
        cls.y_prob = y_prob

    def test_binary_metrics_alignment(self):
        # The authors' helper interprets the first column as P(class 0) and the
        # second as P(class 1); when fed a single vector it assumes that vector
        # is P(class 0). Our regenerated helpers use P(class 1), so flip here.
        authors = self.authors_metrics.print_metrics_binary(
            self.y_true, 1.0 - self.y_prob, verbose=0
        )
        regen_binary = self.regen_metrics.binary_metrics(self.y_true, self.y_prob)
        regen_minpse = self.regen_metrics.minpse(self.y_true, self.y_prob)
        regen_thr = self.regen_metrics.threshold_metrics(self.y_true, self.y_prob, thr=0.5)

        tn, fp, fn, tp = (
            regen_thr["tn"],
            regen_thr["fp"],
            regen_thr["fn"],
            regen_thr["tp"],
        )
        prec0 = tn / max(tn + fn, 1)

        differences = []
        comparisons = [
            ("auroc", regen_binary["auroc"], authors["auroc"]),
            ("auprc", regen_binary["auprc"], authors["auprc"]),
            ("minpse", regen_minpse["minpse"], authors["minpse"]),
            ("acc", regen_thr["acc"], authors["acc"]),
            ("precision_class1", regen_thr["precision"], authors["prec1"]),
            ("precision_class0", prec0, authors["prec0"]),
            ("recall_class1", regen_thr["recall"], authors["rec1"]),
            ("recall_class0", regen_thr["specificity"], authors["rec0"]),
            ("f1_score", regen_thr["f1"], authors["f1_score"]),
        ]

        for name, ours, theirs in comparisons:
            if not _is_close(ours, theirs):
                differences.append(_format_diff(name, ours, theirs))

        if differences:
            self.fail("Metric calculations diverged:\n" + "\n".join(differences))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
