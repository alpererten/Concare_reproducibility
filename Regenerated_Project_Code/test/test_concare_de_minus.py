import unittest

import torch

from model_codes.ConCare_DE_minus import ConCareDEMinus


class ConCareDEMinusForwardTest(unittest.TestCase):
    def test_forward_shapes_and_ignores_demographics(self):
        model = ConCareDEMinus(
            input_dim=76,
            hidden_dim=64,
            d_model=64,
            MHD_num_head=4,
            d_ff=256,
            output_dim=1,
            keep_prob=0.5,
            demographic_dim=12,
        )
        model.eval()
        x = torch.randn(2, 24, 76)
        demo = torch.randn(2, 12)
        demo_clone = demo.clone()
        with torch.no_grad():
            out, decov = model(x, demo)
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue(torch.all((out >= 0.0) & (out <= 1.0)))
        self.assertTrue(torch.isfinite(decov).all())
        # Ensure ablation does not mutate demographics in-place
        self.assertTrue(torch.allclose(demo, demo_clone))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
