"""FLARE verifier package: V3-MC critic + WM-terminal outcome predictor.

Layout:
    data.py     — ChunkDataset, label/featurization helpers
    models.py   — V3MC, WMTerminal architectures
    train_v3_mc.py        — train V3-MC (discounted Monte Carlo critic)
    train_wm_terminal.py  — train WM-terminal (nano outcome predictor)
    score.py    — inference-time scoring (V1, V2_outcome, V3-MC, WM-terminal)
    build_v2_outcome_cache.py — kNN cache for V2_outcome
    extract_encoder_features.py — Day 10 post-hoc encoder feature extraction (remote)
"""
