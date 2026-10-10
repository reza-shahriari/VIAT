# Autopilot benchmark (2026-10-08 13:03)

Detectors: google/owlv2-large-patch14-ensemble - quality `high` - judges: none - up to 600 frames per video.

Ground truth is your hand labels. Objects your labeling policy skips (background cars, the turret commander...) count as false positives here, so precision is a lower bound.

## Per video

| Video | Frames | Tracks (real) | Raw P / R | Auto-reject P / R | After reviewing flagged P / R | Flagged |
|---|---|---|---|---|---|---|
| AchzaritTank.mp4 | 600 | 32 (2) | 0.103 / 1.000 | - | 0.414 / 1.000 | 31 |
| 20_INCREDIBLE_ARMORED_VEHICLES.mp4 | 600 | 146 (17) | 0.212 / 0.978 | - | 0.683 / 0.753 | 134 |
| BMW_X6_UavView.mp4 | 600 | 12 (3) | 0.258 / 0.790 | - | 0.543 / 0.440 | 12 |
| CarsInSnowDroneView.mp4 | 600 | 2 (2) | 1.000 / 1.000 | - | 1.000 / 1.000 | 2 |
| DroneViewHuman.mp4 | 301 | 4 (2) | 0.469 / 0.970 | - | 0.612 / 0.970 | 4 |
| IDF_D9_armored_bulldozer.mp4 | 600 | 14 (0) | 0.126 / 0.997 | - | 0.000 / 0.000 | 14 |
| Danish_troops_firing.mp4 | 600 | 52 (7) | 0.293 / 0.868 | - | 0.866 / 0.717 | 50 |
| DifferentVanets.mp4 | 600 | 55 (13) | 0.384 / 0.993 | - | 0.693 / 0.961 | 45 |
| Challenger3Tank.mp4 | 600 | 70 (22) | 0.337 / 0.846 | - | 0.669 / 0.830 | 43 |
| AN_MPQ-64 Sentinel RADAR - Andrew Dutton (720p, h2 | 600 | 7 (2) | 0.461 / 0.577 | - | 0.698 / 0.526 | 5 |
| CV90_BrazilTank_train.mp4 | 600 | 79 (18) | 0.211 / 0.957 | - | 0.672 / 0.925 | 64 |
| DroneRunChicago1.mp4 | 600 | 15 (2) | 0.228 / 1.000 | - | 0.516 / 1.000 | 15 |
| ISrealEitanAPC1.mp4 | 600 | 50 (10) | 0.238 / 0.948 | - | 0.414 / 0.783 | 26 |
| INKAS_M1MilitaryVehicle.mp4 | 600 | 108 (15) | 0.159 / 0.989 | - | 0.629 / 0.913 | 99 |
| Israeli_bulldozers_manoeuver.mp4 | 600 | 52 (5) | 0.169 / 0.991 | - | 0.597 / 0.596 | 51 |
| ImaginiK2TankTrain.mp4 | 600 | 77 (14) | 0.217 / 0.972 | - | 0.604 / 0.932 | 59 |
| BMW_M_DRAG.mp4 | 600 | 38 (12) | 0.326 / 0.521 | - | 0.635 / 0.366 | 27 |
| EpicRunDroneView.mp4 | 600 | 6 (5) | 0.828 / 0.822 | - | 0.936 / 0.780 | 4 |
| **All** | | | **0.238 / 0.862** | - | **0.641 / 0.704** | |

No judge scores, so no auto-reject column.

## Judges

AUC: chance a real track scores above a false one (0.5 = useless, 1.0 = perfect). For each reject threshold t: share of rejected boxes that were really false (precision), share of all false boxes removed, and real boxes lost.

### learned: AUC 0.896 over 819 tracks

| t | rejected tracks | precision | false removed | real boxes lost |
|---|---|---|---|---|
| 0.05 | 378 | 0.987 | 0.317 | 180 |
| 0.1 | 452 | 0.972 | 0.385 | 474 |
| 0.15 | 490 | 0.971 | 0.427 | 538 |
| 0.2 | 514 | 0.951 | 0.454 | 997 |
| 0.25 | 535 | 0.930 | 0.477 | 1525 |
| 0.3 | 551 | 0.930 | 0.500 | 1602 |
| 0.4 | 588 | 0.923 | 0.559 | 1984 |
| 0.5 | 608 | 0.915 | 0.610 | 2407 |

Suggested reject threshold (>= 95% precision): 0.2

