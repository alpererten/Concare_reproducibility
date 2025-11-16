import json
import tempfile
import unittest

import torch

from Regenerated_Project_Code.model_codes.ConCare_Model_v3 import (
    ConCare,
    FinalAttentionQKV,
    MissingAwareTemporalAttention,
)


class MissingAwareAttentionTests(unittest.TestCase):
    def test_missing_temporal_prefers_observed(self):
        attn = MissingAwareTemporalAttention(hidden_dim=2, attention_hidden_dim=2)
        with torch.no_grad():
            attn.query_proj.weight.copy_(torch.eye(2))
            attn.query_proj.bias.zero_()
            attn.key_proj.weight.copy_(torch.eye(2))
            attn.key_proj.bias.zero_()
            attn.obs_bias.fill_(2.0)
            attn.miss_bias.fill_(0.0)

        H = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        mask = torch.tensor([[1.0, 0.0, 1.0]])
        _, weights = attn(H, mask)

        self.assertGreater(weights[0, 0].item(), weights[0, 1].item())
        self.assertGreater(weights[0, 2].item(), weights[0, 1].item())

    def test_final_attention_mask_bias(self):
        attn = FinalAttentionQKV(hidden_dim=4, attention_hidden_dim=4, dropout=0.0, use_mask_bias=True)
        with torch.no_grad():
            for module in (attn.W_q, attn.W_k, attn.W_v, attn.W_out):
                module.weight.zero_()
                if module.bias is not None:
                    module.bias.zero_()
            attn.mask_scale.fill_(1.0)

        x = torch.ones(1, 2, 4)
        feature_mask = torch.tensor([[0.0, 1.0]])
        _, weights = attn(x, feature_mask)

        self.assertGreater(weights[0, 1].item(), weights[0, 0].item())

    def test_concare_forward_with_masks(self):
        cfg = {
            "id_to_channel": ["feat_a", "feat_b"],
            "is_categorical_channel": {"feat_a": False, "feat_b": False},
            "possible_values": {"feat_a": [], "feat_b": []},
            "normal_values": {"feat_a": 0.0, "feat_b": 0.0},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = f"{tmpdir}/disc.json"
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)

            model = ConCare(
                input_dim=4,
                hidden_dim=4,
                d_model=4,
                MHD_num_head=2,
                d_ff=8,
                output_dim=1,
                keep_prob=0.9,
                demographic_dim=2,
                mask_dim=2,
                base_value_dim=2,
                config_path=cfg_path,
                enable_missing_aware=True,
            )

            X = torch.zeros(1, 3, 4)
            # Value streams
            X[0, :, 0] = torch.tensor([1.0, 0.5, 0.0])
            X[0, :, 1] = torch.tensor([0.0, -0.5, -1.0])
            # Masks
            X[0, :, 2] = torch.tensor([1.0, 1.0, 0.0])
            X[0, :, 3] = torch.tensor([0.0, 1.0, 1.0])
            D = torch.zeros(1, 2)

            out, decov = model(X, D)

            self.assertEqual(out.shape, (1, 1))
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.isfinite(decov))

            ctx, masks, _ = model(X, D, return_context=True)
            self.assertEqual(ctx.shape[:2], (1, 3))  # 2 value dims + 1 demo
            self.assertIsNotNone(masks)
            self.assertEqual(masks.shape, (1, 3))

    def test_return_context_without_missing_extension(self):
        model = ConCare(
            input_dim=2,
            hidden_dim=4,
            d_model=4,
            MHD_num_head=2,
            d_ff=8,
            output_dim=1,
            keep_prob=0.9,
            demographic_dim=1,
            mask_dim=0,
            enable_missing_aware=False,
        )
        X = torch.randn(1, 3, 2)
        D = torch.randn(1, 1)
        ctx, masks, _ = model(X, D, return_context=True)
        self.assertEqual(ctx.shape[:2], (1, 3))
        self.assertIsNone(masks)


if __name__ == "__main__":
    unittest.main()
