import unittest

import torch

from Regenerated_Project_Code.model_codes.ConCare_Model_v3 import ConCare


def _synthetic(batch_size=2, time_steps=12, features=4, demo_dim=3):
    g = torch.Generator().manual_seed(20231109)
    x = torch.randn(batch_size, time_steps, features, generator=g)
    demo = torch.randn(batch_size, demo_dim, generator=g)
    return x, demo


class TimeAblationTest(unittest.TestCase):
    def test_forward_without_time_aware_attention(self):
        x, demo = _synthetic()
        model = ConCare(
            input_dim=x.shape[-1],
            hidden_dim=8,
            d_model=8,
            MHD_num_head=2,
            d_ff=16,
            output_dim=1,
            keep_prob=0.9,
            demographic_dim=demo.shape[-1],
            time_aware_attention=False,
        )
        out, decov = model(x, demo)

        self.assertEqual(out.shape, (x.shape[0], 1))
        self.assertTrue(torch.all((out >= 0) & (out <= 1)))
        self.assertTrue(torch.isfinite(decov))
        self.assertFalse(model.LastStepAttentions[0].time_aware)


if __name__ == "__main__":
    unittest.main()
