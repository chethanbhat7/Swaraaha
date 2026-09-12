# Comparative Study Report

| Arm | Type | Params | Test F1@0.5 | Test F1@tuned | Boli F1@0.5 | Boli F1@tuned |
|---|---|---|---|---|---|---|
| arm01_5x_w2v2 | classifier | 94569090 | 0.5730 | 0.5704 | 0.2540 | 0.2920 |
| arm02_mt_w2v2_frz3 | multitask | 97332362 | 0.5415 | 0.5557 | 0.1621 | 0.2384 |
| arm03_mt_w2v2_frz20 | multitask | 97332362 | 0.1413 | 0.3781 | 0.0457 | 0.2138 |
| arm04_cnn_pool | multitask | 342030 | 0.1492 | 0.2556 | 0.3507 | 0.3315 |
| arm05_cnn_single | multitask_single | 274950 | 0.2116 | 0.2617 | 0.4501 | 0.4756 |
| arm06_cnn_lstm | multitask | 457614 | 0.2418 | 0.2489 | 0.4508 | 0.5077 |
| arm07_cnn_tf | multitask | 540302 | 0.2694 | 0.2714 | 0.4214 | 0.3451 |

Honest labeling: "in-distribution held-out (same-speaker overlap)" for test; "cross-corpus held-out" for Boli. Single seed 42; thresholds tuned on val only.
