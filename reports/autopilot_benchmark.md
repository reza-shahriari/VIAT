# Autopilot benchmark (2026-10-01 11:28)

Detectors: yolov8x-worldv2.pt, sam3.1_l.pt - quality `high` - judges: clip, sam_zoom - up to 600 frames per video.

Ground truth is your hand labels. Objects your labeling policy skips (background cars, the turret commander...) count as false positives here, so precision is a lower bound.

## Per video

| Video | Frames | Tracks (real) | Raw P / R | Auto-reject P / R | After reviewing flagged P / R | Flagged |
|---|---|---|---|---|---|---|
| AchzaritTank.mp4 | 600 | 3 (1) | 0.912 / 1.000 | 0.972 / 1.000 | 1.000 / 1.000 | 2 |
| 20_INCREDIBLE_ARMORED_VEHICLES.mp4 | 600 | 73 (19) | 0.440 / 1.000 | 0.507 / 1.000 | 0.852 / 0.978 | 66 |
| BMW_X6_UavView.mp4 | 600 | 3 (3) | 0.968 / 1.000 | 0.968 / 1.000 | 0.968 / 1.000 | 0 |
| CarsInSnowDroneView.mp4 | 600 | 2 (2) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0 |
| DroneViewHuman.mp4 | 301 | 3 (1) | 0.756 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 2 |
| IDF_D9_armored_bulldozer.mp4 | 600 | 11 (4) | 0.476 / 0.930 | 0.531 / 0.930 | 0.684 / 0.930 | 10 |
| Danish_troops_firing.mp4 | 600 | 14 (7) | 0.845 / 0.781 | 0.845 / 0.781 | 0.940 / 0.779 | 10 |
| DifferentVanets.mp4 | 600 | 30 (11) | 0.598 / 0.998 | 0.598 / 0.998 | 0.965 / 0.998 | 22 |
| Challenger3Tank.mp4 | 600 | 31 (26) | 0.864 / 0.997 | 0.913 / 0.997 | 0.945 / 0.997 | 3 |
| AN_MPQ-64 Sentinel RADAR - Andrew Dutton (720p, h2 | 600 | 1 (1) | 0.963 / 0.048 | 0.963 / 0.048 | 0.963 / 0.048 | 1 |
| **All** | | | **0.700 / 0.782** | **0.744 / 0.782** | **0.918 / 0.780** | |

Auto-reject column: tracks whose combined judge score is <= 0.3 removed (suggested threshold, see below).

## Detectors alone vs union (all videos)

| Detectors | Precision | Recall |
|---|---|---|
| yolov8x-worldv2.pt | 0.869 | 0.684 |
| sam3.1_l.pt | 0.579 | 0.790 |
| union | 0.700 | 0.782 |

## Judges

AUC: chance a real track scores above a false one (0.5 = useless, 1.0 = perfect). For each reject threshold t: share of rejected boxes that were really false (precision), share of all false boxes removed, and real boxes lost.

### clip: AUC 0.876 over 171 tracks

| t | rejected tracks | precision | false removed | real boxes lost |
|---|---|---|---|---|
| 0.05 | 30 | 0.902 | 0.257 | 75 |
| 0.1 | 49 | 0.809 | 0.377 | 238 |
| 0.15 | 67 | 0.818 | 0.540 | 322 |
| 0.2 | 79 | 0.764 | 0.658 | 544 |
| 0.25 | 95 | 0.708 | 0.754 | 834 |
| 0.3 | 104 | 0.702 | 0.838 | 952 |
| 0.4 | 113 | 0.617 | 0.875 | 1454 |
| 0.5 | 120 | 0.462 | 0.889 | 2773 |

Suggested reject threshold (>= 95% precision): None

### sam_zoom: AUC 0.769 over 171 tracks

| t | rejected tracks | precision | false removed | real boxes lost |
|---|---|---|---|---|
| 0.05 | 4 | 1.000 | 0.028 | 0 |
| 0.1 | 6 | 1.000 | 0.055 | 0 |
| 0.15 | 8 | 1.000 | 0.067 | 0 |
| 0.2 | 8 | 1.000 | 0.067 | 0 |
| 0.25 | 13 | 1.000 | 0.139 | 0 |
| 0.3 | 18 | 1.000 | 0.201 | 0 |
| 0.4 | 25 | 0.901 | 0.276 | 81 |
| 0.5 | 29 | 0.762 | 0.302 | 253 |

Suggested reject threshold (>= 95% precision): 0.3

### combined: AUC 0.910 over 171 tracks

| t | rejected tracks | precision | false removed | real boxes lost |
|---|---|---|---|---|
| 0.05 | 3 | 1.000 | 0.026 | 0 |
| 0.1 | 5 | 1.000 | 0.040 | 0 |
| 0.15 | 8 | 1.000 | 0.067 | 0 |
| 0.2 | 15 | 1.000 | 0.170 | 0 |
| 0.25 | 19 | 1.000 | 0.213 | 0 |
| 0.3 | 22 | 1.000 | 0.232 | 0 |
| 0.4 | 31 | 0.912 | 0.315 | 81 |
| 0.5 | 43 | 0.824 | 0.443 | 253 |

Suggested reject threshold (>= 95% precision): 0.3

## Judge time

- clip: 3 s total
- sam_zoom: 110 s total

## Learned judge

Learned judge trained on 171 decisions (75 real): cross-validated AUC 0.954. Strongest features: p25_score (+1.55), span_sec (+1.21), support (+1.14), log_size (+1.04), clip (+0.94), ext_frac (+0.69)

