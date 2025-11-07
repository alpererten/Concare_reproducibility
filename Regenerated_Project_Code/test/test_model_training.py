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


def _instantiate_models():
    kwargs = dict(
        input_dim=76,
        hidden_dim=64,
        d_model=64,
        MHD_num_head=4,
        d_ff=256,
        output_dim=1,
        keep_prob=0.8,
    )
    authors_cls = _load_authors_concare()
    regen_cls = _load_regenerated_concare()
    torch.manual_seed(20231105)
    authors_model = authors_cls(**kwargs)
    torch.manual_seed(20231105)
    regen_model = regen_cls(demographic_dim=12, **kwargs)
    return authors_model, regen_model


def _synthetic_batch(batch_size=4, time_steps=48, features=76, demo_dim=12):
    g = torch.Generator().manual_seed(20231105)
    x = torch.randn(batch_size, time_steps, features, generator=g)
    demo = torch.randn(batch_size, demo_dim, generator=g)
    y = torch.randint(0, 2, (batch_size, 1), generator=g).float()
    return x, demo, y


def _forward_outputs(model, x, demo):
    model.eval()
    with torch.no_grad():
        out, decov = model(x, demo)
    return out, decov


def _train_model(model, x, demo, y, *, steps=3, lr=1e-3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    initial_loss = None
    loss_value = None
    for step in range(steps):
        opt.zero_grad()
        out, decov = model(x, demo)
        loss = criterion(out, y)
        if torch.is_tensor(decov):
            loss = loss + 1e-3 * decov
        if initial_loss is None:
            initial_loss = float(loss.item())
        loss.backward()
        opt.step()
        loss_value = float(loss.item())
    return initial_loss, loss_value


@unittest.skipIf(nbformat is None, "nbformat is required to introspect the authors' notebook")
class ModelForwardSmokeTest(unittest.TestCase):
    def test_forward_outputs_and_doc_alignment(self):
        authors_model, regen_model = _instantiate_models()
        x, demo, _ = _synthetic_batch()

        authors_out, authors_decov = _forward_outputs(authors_model, x, demo)
        regen_out, regen_decov = _forward_outputs(regen_model, x, demo)

        self.assertEqual(authors_out.shape, (4, 1))
        self.assertEqual(regen_out.shape, (4, 1))
        self.assertTrue(torch.all((authors_out >= 0) & (authors_out <= 1)))
        self.assertTrue(torch.all((regen_out >= 0) & (regen_out <= 1)))
        self.assertTrue(torch.isfinite(authors_decov))
        self.assertTrue(torch.isfinite(regen_decov))

        mean_abs_diff = torch.mean(torch.abs(authors_out - regen_out)).item()
        summary_line = f"- forward_mean_abs_diff (B=4,T=48 synthetic batch): {mean_abs_diff:.6f}"

        report = _ARCH_REPORT.read_text(encoding="utf-8")
        if summary_line not in report:
            self.fail("Forward-pass finding missing from arch_difference.md:\n" + summary_line)


@unittest.skipIf(nbformat is None, "nbformat is required to introspect the authors' notebook")
class TrainingLoopComparisonTest(unittest.TestCase):
    def test_short_training_runs_and_doc_alignment(self):
        authors_model, regen_model = _instantiate_models()
        x, demo, y = _synthetic_batch()

        auth_initial, auth_final = _train_model(authors_model, x, demo, y)
        regen_initial, regen_final = _train_model(regen_model, x, demo, y)

        self.assertLess(auth_final, auth_initial)
        self.assertLess(regen_final, regen_initial)

        summary_line = (
            "- training_loss (3 steps, lr=1e-3 on synthetic batch): "
            f"authors_initial={auth_initial:.6f}, authors_final={auth_final:.6f}; "
            f"regen_initial={regen_initial:.6f}, regen_final={regen_final:.6f}"
        )

        report = _ARCH_REPORT.read_text(encoding="utf-8")
        if summary_line not in report:
            self.fail("Training-loop finding missing from arch_difference.md:\n" + summary_line)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
