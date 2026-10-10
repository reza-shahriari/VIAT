# Autopilot benchmark (2026-10-08 13:09)

Detectors: yolov8x-worldv2.pt, sam3.1_l.pt, IDEA-Research/grounding-dino-base, google/owlv2-large-patch14-ensemble - quality `high` - judges: clip - up to 600 frames per video.

Ground truth is your hand labels. Objects your labeling policy skips (background cars, the turret commander...) count as false positives here, so precision is a lower bound.

## Per video

| Video | Frames | Tracks (real) | Raw P / R | Auto-reject P / R | After reviewing flagged P / R | Flagged |
|---|---|---|---|---|---|---|
| AchzaritTank.mp4 | 600 | 16 (1) | 0.205 / 1.000 | 0.512 / 1.000 | 1.000 / 1.000 | 15 |
| 20_INCREDIBLE_ARMORED_VEHICLES.mp4 | 600 | 118 (18) | 0.236 / 1.000 | 0.255 / 0.138 | 0.787 / 0.858 | 117 |
| BMW_X6_UavView.mp4 | 600 | 4 (3) | 0.872 / 1.000 | 0.872 / 1.000 | 0.968 / 1.000 | 1 |
| CarsInSnowDroneView.mp4 | 600 | 2 (2) | 1.000 / 1.000 | 0.000 / 0.000 | 1.000 / 1.000 | 1 |
| DroneViewHuman.mp4 | 301 | 4 (1) | 0.644 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 3 |
| IDF_D9_armored_bulldozer.mp4 | 600 | 15 (1) | 0.154 / 1.000 | 0.191 / 1.000 | 0.535 / 0.538 | 15 |
| Danish_troops_firing.mp4 | 600 | 37 (9) | 0.369 / 0.879 | 0.648 / 0.771 | 0.690 / 0.825 | 34 |
| DifferentVanets.mp4 | 600 | 49 (11) | 0.381 / 0.999 | 0.595 / 0.895 | 0.821 / 0.834 | 44 |
| Challenger3Tank.mp4 | 600 | 59 (26) | 0.452 / 0.998 | 0.507 / 0.924 | 0.911 / 0.994 | 33 |
| AN_MPQ-64 Sentinel RADAR - Andrew Dutton (720p, h2 | 600 | 8 (4) | 0.524 / 0.785 | 0.791 / 0.763 | 0.814 / 0.761 | 8 |
| CV90_BrazilTank_train.mp4 | 600 | 65 (16) | 0.288 / 1.000 | 0.506 / 0.963 | 0.997 / 0.960 | 50 |
| DroneRunChicago1.mp4 | 600 | 13 (1) | 0.309 / 1.000 | 0.977 / 1.000 | 1.000 / 1.000 | 12 |
| ISrealEitanAPC1.mp4 | 600 | 28 (11) | 0.442 / 1.000 | 0.512 / 0.910 | 0.807 / 0.873 | 19 |
| INKAS_M1MilitaryVehicle.mp4 | 600 | 81 (15) | 0.196 / 1.000 | 0.364 / 0.930 | 0.733 / 0.938 | 72 |
| Israeli_bulldozers_manoeuver.mp4 | 600 | 65 (5) | 0.146 / 0.992 | 0.197 / 0.552 | 0.808 / 0.563 | 65 |
| ImaginiK2TankTrain.mp4 | 600 | 62 (13) | 0.275 / 0.985 | 0.538 / 0.978 | 0.840 / 0.959 | 48 |
| BMW_M_DRAG.mp4 | 600 | 40 (21) | 0.582 / 0.998 | 0.809 / 0.873 | 0.860 / 0.998 | 22 |
| EpicRunDroneView.mp4 | 600 | 15 (6) | 0.601 / 0.900 | 0.665 / 0.807 | 0.885 / 0.900 | 12 |
| **All** | | | **0.314 / 0.958** | **0.498 / 0.784** | **0.832 / 0.866** | |

Auto-reject column: tracks whose combined judge score is <= 0.3 removed (the default `reject_below`, as the pipeline does it).

## Judges

AUC: chance a real track scores above a false one (0.5 = useless, 1.0 = perfect). For each reject threshold t: share of rejected boxes that were really false (precision), share of all false boxes removed, and real boxes lost.

### clip: AUC 0.814 over 681 tracks

| t | rejected tracks | precision | false removed | real boxes lost |
|---|---|---|---|---|
| 0.05 | 142 | 0.972 | 0.215 | 196 |
| 0.1 | 238 | 0.936 | 0.354 | 767 |
| 0.15 | 290 | 0.916 | 0.455 | 1322 |
| 0.2 | 337 | 0.901 | 0.530 | 1845 |
| 0.25 | 371 | 0.907 | 0.605 | 1973 |
| 0.3 | 397 | 0.888 | 0.643 | 2578 |
| 0.4 | 458 | 0.844 | 0.733 | 4326 |
| 0.5 | 498 | 0.784 | 0.774 | 6782 |

Suggested reject threshold (>= 95% precision): 0.05

### combined: AUC 0.814 over 681 tracks

| t | rejected tracks | precision | false removed | real boxes lost |
|---|---|---|---|---|
| 0.05 | 142 | 0.972 | 0.215 | 196 |
| 0.1 | 238 | 0.936 | 0.354 | 767 |
| 0.15 | 290 | 0.916 | 0.455 | 1322 |
| 0.2 | 337 | 0.901 | 0.530 | 1845 |
| 0.25 | 371 | 0.907 | 0.605 | 1973 |
| 0.3 | 397 | 0.888 | 0.643 | 2578 |
| 0.4 | 458 | 0.844 | 0.733 | 4326 |
| 0.5 | 498 | 0.784 | 0.774 | 6782 |

Suggested reject threshold (>= 95% precision): 0.05

### learned: AUC 0.964 over 681 tracks

| t | rejected tracks | precision | false removed | real boxes lost |
|---|---|---|---|---|
| 0.05 | 333 | 0.995 | 0.416 | 62 |
| 0.1 | 385 | 0.996 | 0.510 | 62 |
| 0.15 | 407 | 0.991 | 0.531 | 160 |
| 0.2 | 420 | 0.983 | 0.549 | 309 |
| 0.25 | 439 | 0.973 | 0.573 | 509 |
| 0.3 | 447 | 0.971 | 0.594 | 561 |
| 0.4 | 471 | 0.955 | 0.654 | 972 |
| 0.5 | 487 | 0.956 | 0.709 | 1035 |

Suggested reject threshold (>= 95% precision): 0.5

## Judge time

- clip: 15 s total

