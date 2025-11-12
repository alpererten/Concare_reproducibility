import unittest

import torch

from Regenerated_Project_Code.model_codes.ConCare_MC_minus import ConCareMCMinus


class ConCareMCMinusForwardTest(unittest.TestCase):
    def test_forward_shapes_and_decov_zero(self):
        model = ConCareMCMinus(
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
        x = torch.randn(3, 48, 76)
        demo = torch.randn(3, 12)
        with torch.no_grad():
            out, decov = model(x, demo)
        self.assertEqual(out.shape, (3, 1))
        self.assertTrue(torch.all((out >= 0.0) & (out <= 1.0)))
        self.assertTrue(torch.allclose(decov, torch.zeros_like(decov)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
