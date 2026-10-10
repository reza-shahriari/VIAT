# Autopilot benchmark (2026-10-01 12:10)

Detectors: yolov8x-worldv2.pt, sam3.1_l.pt - quality `high` - judges: clip, sam_zoom, learned - up to 600 frames per video.

Ground truth is your hand labels. Objects your labeling policy skips (background cars, the turret commander...) count as false positives here, so precision is a lower bound.

## Per video

| Video | Frames | Tracks (real) | Raw P / R | Auto-reject P / R | After reviewing flagged P / R | Flagged |
|---|---|---|---|---|---|---|
| CV90_BrazilTank_train.mp4 | 600 | 33 (17) | 0.546 / 1.000 | 0.593 / 1.000 | 0.902 / 1.000 | 16 |
| DroneRunChicago1.mp4 | 600 | 11 (1) | 0.351 / 1.000 | 0.415 / 1.000 | 1.000 / 1.000 | 10 |
| ISrealEitanAPC1.mp4 | 600 | 18 (14) | 0.787 / 0.958 | 0.787 / 0.958 | 0.952 / 0.958 | 6 |
| INKAS_M1MilitaryVehicle.mp4 | 600 | 33 (13) | 0.574 / 1.000 | 0.708 / 1.000 | 1.000 / 1.000 | 20 |
| Israeli_bulldozers_manoeuver.mp4 | 600 | 39 (6) | 0.297 / 0.991 | 0.316 / 0.991 | 0.730 / 0.862 | 37 |
| ImaginiK2TankTrain.mp4 | 600 | 34 (14) | 0.556 / 0.988 | 0.605 / 0.988 | 0.909 / 0.986 | 20 |
| BMW_M_DRAG.mp4 | 600 | 28 (20) | 0.774 / 0.998 | 0.815 / 0.998 | 0.943 / 0.998 | 7 |
| EpicRunDroneView.mp4 | 600 | 6 (5) | 0.859 / 0.869 | 0.967 / 0.869 | 0.967 / 0.869 | 2 |
| **All** | | | **0.496 / 0.981** | **0.543 / 0.981** | **0.900 / 0.955** | |

Auto-reject column: tracks whose combined judge score is <= 0.3 removed (the default `reject_below`, as the pipeline does it).

## Judges

AUC: chance a real track scores above a false one (0.5 = useless, 1.0 = perfect). For each reject threshold t: share of rejected boxes that were really false (precision), share of all false boxes removed, and real boxes lost.

### clip: AUC 0.938 over 202 tracks

| t | rejected tracks | precision | false removed | real boxes lost |
|---|---|---|---|---|
| 0.05 | 22 | 0.906 | 0.265 | 170 |
| 0.1 | 40 | 0.935 | 0.395 | 170 |
| 0.15 | 65 | 0.874 | 0.651 | 579 |
| 0.2 | 82 | 0.881 | 0.777 | 647 |
| 0.25 | 97 | 0.885 | 0.849 | 679 |
| 0.3 | 103 | 0.872 | 0.864 | 782 |
| 0.4 | 115 | 0.838 | 0.913 | 1093 |
| 0.5 | 126 | 0.789 | 0.948 | 1568 |

Suggested reject threshold (>= 95% precision): None

### sam_zoom: AUC 0.859 over 202 tracks

| t | rejected tracks | precision | false removed | real boxes lost |
|---|---|---|---|---|
| 0.05 | 8 | 1.000 | 0.049 | 0 |
| 0.1 | 9 | 1.000 | 0.055 | 0 |
| 0.15 | 10 | 1.000 | 0.061 | 0 |
| 0.2 | 11 | 1.000 | 0.066 | 0 |
| 0.25 | 13 | 1.000 | 0.082 | 0 |
| 0.3 | 18 | 1.000 | 0.117 | 0 |
| 0.4 | 26 | 1.000 | 0.187 | 0 |
| 0.5 | 35 | 1.000 | 0.311 | 0 |

Suggested reject threshold (>= 95% precision): 0.5

### combined: AUC 0.968 over 202 tracks

| t | rejected tracks | precision | false removed | real boxes lost |
|---|---|---|---|---|
| 0.05 | 2 | 1.000 | 0.011 | 0 |
| 0.1 | 8 | 1.000 | 0.045 | 0 |
| 0.15 | 11 | 1.000 | 0.068 | 0 |
| 0.2 | 14 | 1.000 | 0.108 | 0 |
| 0.25 | 21 | 1.000 | 0.157 | 0 |
| 0.3 | 26 | 1.000 | 0.174 | 0 |
| 0.4 | 43 | 1.000 | 0.377 | 0 |
| 0.5 | 54 | 1.000 | 0.435 | 0 |

Suggested reject threshold (>= 95% precision): 0.5

### learned: AUC 0.974 over 202 tracks

| t | rejected tracks | precision | false removed | real boxes lost |
|---|---|---|---|---|
| 0.05 | 76 | 0.958 | 0.470 | 127 |
| 0.1 | 91 | 0.957 | 0.620 | 172 |
| 0.15 | 96 | 0.962 | 0.698 | 172 |
| 0.2 | 98 | 0.947 | 0.700 | 240 |
| 0.25 | 101 | 0.952 | 0.769 | 240 |
| 0.3 | 103 | 0.952 | 0.779 | 240 |
| 0.4 | 106 | 0.953 | 0.796 | 240 |
| 0.5 | 108 | 0.923 | 0.798 | 410 |

Suggested reject threshold (>= 95% precision): 0.4

## Judge time

- clip: 3 s total
- sam_zoom: 92 s total

## Learned judge

Learned judge trained on 373 decisions (165 real): cross-validated AUC 0.971. Strongest features: p25_score (+0.97), clip (+0.93), support (+0.92), log_size (+0.84), max_score (+0.82), agree (-0.70)

