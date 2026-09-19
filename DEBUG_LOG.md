cft_compare_v2: HEAD-TO-HEAD vs full-attention baseline (same curriculum).
Accuracy: len2 CFT 98.25% vs base 59.50% | len3 96.00% vs 100% | len4 UNSEEN 84.00% vs 56.00%.
Baseline shows catastrophic forgetting (len2 collapse) and no length extrapolation.
CFT: stable skills + compositional generalization + constant 2,048-byte memory vs baseline's linear growth (99,840 B at chain 64).