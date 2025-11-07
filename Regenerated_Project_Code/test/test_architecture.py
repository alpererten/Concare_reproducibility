import json
import math
import unittest
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

try:
    import nbformat
except Exception:  # pragma: no cover - optional dependency
    nbformat = None


_ROOT = Path(__file__).resolve().parents[2]
_AUTH_NOTEBOOK = _ROOT / "ConCare-master-authors-repo" / "concare-notebook_trained_251020_4.ipynb"
_ARCH_REPORT = _ROOT / "Regenerated_Project_Code" / "arch_difference.md"


def _load_authors_concare():
    if nbformat is None:
        raise unittest.SkipTest("nbformat is required to parse the authors' notebook")

    nb = nbformat.read(_AUTH_NOTEBOOK, as_version=4)
    source = None
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code" and "class ConCare" in "".join(cell.get("source", "")):
            source = "".join(cell["source"])
            break
    if source is None:
        raise AssertionError("ConCare definition not found in notebook")

    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "np": np,
        "copy": __import__("copy"),
        "Variable": Variable,
        "device": torch.device("cpu"),
    }
    exec(source, namespace)
    return namespace["ConCare"]


def _load_regenerated_concare():
    module = import_module("Regenerated_Project_Code.model_codes.ConCare_Model_v3")
    return module.ConCare


def _describe_model(model):
    summary = {}
    summary["module_order"] = list(model._modules.keys())
    summary["num_grus"] = len(model.GRUs)
    summary["gru_hidden_size"] = model.GRUs[0].hidden_size
    summary["per_feature_attention_cls"] = type(model.LastStepAttentions[0]).__name__
    summary["per_feature_attention_time_aware"] = bool(
        getattr(model.LastStepAttentions[0], "time_aware", False)
    )
    summary["has_positional_encoding"] = hasattr(model, "PositionalEncoding")
    summary["multihead_heads"] = model.MHD_num_head
    summary["d_model"] = model.d_model
    summary["d_ff"] = model.d_ff
    summary["keep_prob"] = model.keep_prob
    return summary


def _diff_summaries(authors_summary, regen_summary):
    diffs = []
    keys = sorted(set(authors_summary) | set(regen_summary))
    for key in keys:
        aval = authors_summary.get(key)
        rval = regen_summary.get(key)
        if aval != rval:
            def _fmt(v):
                if isinstance(v, (list, dict)):
                    return json.dumps(v, sort_keys=True)
                return str(v)
            diffs.append(f"- {key}: authors={_fmt(aval)}, regenerated={_fmt(rval)}")
    return diffs


@unittest.skipIf(nbformat is None, "nbformat is required to introspect the authors' notebook")
class ArchitectureComparisonTest(unittest.TestCase):
    maxDiff = None

    def test_architecture_differences_documented(self):
        authors_cls = _load_authors_concare()
        regen_cls = _load_regenerated_concare()

        kwargs = dict(input_dim=76, hidden_dim=64, d_model=64, MHD_num_head=4, d_ff=256, output_dim=1, keep_prob=0.8)
        authors_model = authors_cls(**kwargs)
        regen_model = regen_cls(demographic_dim=12, **kwargs)

        authors_summary = _describe_model(authors_model)
        regen_summary = _describe_model(regen_model)
        differences = _diff_summaries(authors_summary, regen_summary)

        report = _ARCH_REPORT.read_text(encoding="utf-8")
        missing = [line for line in differences if line not in report]
        if missing:
            self.fail("Architecture differences are not documented:\n" + "\n".join(missing))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
