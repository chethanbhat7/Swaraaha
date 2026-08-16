# Comparative Study Report

| Arm | Type | Params | Test F1@0.5 | Test F1@tuned | Boli F1@0.5 | Boli F1@tuned |
|---|---|---|---|---|---|---|
| arm01_5x_w2v2 | classifier | 94569090 | 0.5111 | 0.5183 | 0.1093 | 0.1535 |
| arm02_mt_w2v2_frz3 | multitask | 97332362 | 0.4896 | 0.5215 | 0.1125 | 0.1595 |
| arm03_mt_w2v2_frz20 | multitask | 97332362 | 0.1417 | 0.3388 | 0.0346 | 0.2951 |
| arm04_cnn_pool | multitask | 342030 | 0.1880 | 0.2532 | 0.4384 | 0.3588 |
| arm05_cnn_single | multitask_single | 274950 | 0.2214 | 0.2531 | 0.4505 | 0.4414 |
| arm06_cnn_lstm | multitask | 457614 | 0.2481 | 0.2585 | 0.4257 | 0.5206 |
| arm07_cnn_tf | multitask | 540302 | 0.2550 | 0.2644 | 0.4020 | 0.4771 |

Honest labeling: "in-distribution held-out (same-speaker overlap)" for test; "cross-corpus held-out" for Boli. Single seed 42; thresholds tuned on val only.
