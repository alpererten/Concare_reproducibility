# Architecture Differences (Authors vs. Regenerated)

We load the original ConCare implementation from `concare-notebook_trained_251020_4.ipynb` and the reproduced `model_codes/ConCare_Model_v3.py`, instantiate both with the paper hyper-parameters (`input_dim=76`, `hidden_dim=64`, `d_model=64`, `MHD_num_head=4`, `d_ff=256`, `keep_prob=0.8`, `demographic_dim=12`), and compare their module layouts. The unit test in `test/test_architecture.py` verifies that the differences below stay in sync with the actual code.

- has_positional_encoding: authors=True, regenerated=False
- module_order: authors=["PositionalEncoding", "GRUs", "LastStepAttentions", "FinalAttentionQKV", "MultiHeadedAttention", "SublayerConnection", "PositionwiseFeedForward", "demo_proj_main", "demo_proj", "output0", "output1", "dropout", "tanh", "softmax", "sigmoid", "relu"], regenerated=["GRUs", "LastStepAttentions", "demo_proj_main", "demo_proj", "MultiHeadedAttention", "SublayerConnection", "PositionwiseFeedForward", "FinalAttentionQKV", "output0", "output1", "dropout", "tanh", "sigmoid", "relu"]
- per_feature_attention_cls: authors=SingleAttention, regenerated=SingleAttentionPerFeatureNew

Interpretation:

- The authors keep a `PositionalEncoding` module (even though it is not used later), while the regenerated model omits it entirely.
- Our reproduction swaps the generic `SingleAttention` class for a specialized `SingleAttentionPerFeatureNew` implementation. The new module now exposes the `time_aware` toggle (and a CLI flag `--no_time_aware_attention`) so we can reproduce ConCare's time-decay ablation directly, despite reusing a different internal implementation.

## Forward & Training Smoke Tests

Small synthetic experiments (batch=4, 48 time steps, paper hyper-parameters) highlight the runtime differences:

- forward_mean_abs_diff (B=4,T=48 synthetic batch): 0.028644
- training_loss (3 steps, lr=1e-3 on synthetic batch): authors_initial=0.726901, authors_final=0.700444; regen_initial=0.695841, regen_final=0.679561

Both models produce valid probabilities and their losses decrease over a few optimization steps, but the regenerated model consistently yields slightly lower loss on this toy batch.
