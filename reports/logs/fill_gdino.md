# Autopilot benchmark (2026-10-08 06:44)

Detectors: IDEA-Research/grounding-dino-base - quality `high` - judges: none - up to 600 frames per video.

Ground truth is your hand labels. Objects your labeling policy skips (background cars, the turret commander...) count as false positives here, so precision is a lower bound.

## Per video

| Video | Frames | Tracks (real) | Raw P / R | Auto-reject P / R | After reviewing flagged P / R | Flagged |
|---|---|---|---|---|---|---|
| AchzaritTank.mp4 | 600 | 34 (1) | 0.220 / 1.000 | - | 1.000 / 1.000 | 33 |
| 20_INCREDIBLE_ARMORED_VEHICLES.mp4 | 600 | 47 (18) | 0.491 / 0.926 | - | 0.877 / 0.911 | 34 |
| BMW_X6_UavView.mp4 | 600 | 4 (3) | 0.872 / 1.000 | - | 0.968 / 1.000 | 1 |
| CarsInSnowDroneView.mp4 | 600 | 4 (2) | 0.726 / 1.000 | - | 1.000 / 1.000 | 2 |
| DroneViewHuman.mp4 | 301 | 5 (1) | 0.592 / 1.000 | - | 0.758 / 1.000 | 3 |
| IDF_D9_armored_bulldozer.mp4 | 600 | 9 (2) | 0.262 / 0.876 | - | 0.507 / 0.521 | 9 |
| Danish_troops_firing.mp4 | 600 | 31 (8) | 0.439 / 0.927 | - | 0.882 / 0.908 | 28 |
| DifferentVanets.mp4 | 600 | 19 (11) | 0.751 / 0.999 | - | 0.878 / 0.999 | 6 |
| Challenger3Tank.mp4 | 600 | 32 (26) | 0.843 / 1.000 | - | 0.949 / 1.000 | 10 |
| AN_MPQ-64 Sentinel RADAR - Andrew Dutton (720p, h2 | 600 | 11 (4) | 0.516 / 0.755 | - | 0.947 / 0.627 | 11 |
| CV90_BrazilTank_train.mp4 | 600 | 59 (18) | 0.301 / 0.994 | - | 0.854 / 0.994 | 40 |
| DroneRunChicago1.mp4 | 600 | 8 (1) | 0.313 / 1.000 | - | 0.976 / 1.000 | 6 |
| ISrealEitanAPC1.mp4 | 600 | 14 (13) | 0.903 / 0.958 | - | 0.954 / 0.958 | 1 |
| INKAS_M1MilitaryVehicle.mp4 | 600 | 47 (13) | 0.380 / 1.000 | - | 0.916 / 1.000 | 34 |
| Israeli_bulldozers_manoeuver.mp4 | 600 | 52 (4) | 0.223 / 0.914 | - | 0.674 / 0.487 | 50 |
| ImaginiK2TankTrain.mp4 | 600 | 61 (12) | 0.267 / 0.945 | - | 0.780 / 0.945 | 46 |
| BMW_M_DRAG.mp4 | 600 | 23 (20) | 0.885 / 0.998 | - | 0.917 / 0.998 | 5 |
| EpicRunDroneView.mp4 | 600 | 30 (4) | 0.364 / 0.900 | - | 0.989 / 0.826 | 26 |
| **All** | | | **0.427 / 0.941** | - | **0.881 / 0.873** | |

No judge scores, so no auto-reject column.

## Judges

AUC: chance a real track scores above a false one (0.5 = useless, 1.0 = perfect). For each reject threshold t: share of rejected boxes that were really false (precision), share of all false boxes removed, and real boxes lost.

### learned: AUC 0.973 over 490 tracks

| t | rejected tracks | precision | false removed | real boxes lost |
|---|---|---|---|---|
| 0.05 | 257 | 0.985 | 0.567 | 170 |
| 0.1 | 288 | 0.986 | 0.653 | 177 |
| 0.15 | 298 | 0.979 | 0.669 | 278 |
| 0.2 | 304 | 0.980 | 0.703 | 278 |
| 0.25 | 310 | 0.962 | 0.704 | 553 |
| 0.3 | 316 | 0.951 | 0.705 | 723 |
| 0.4 | 328 | 0.938 | 0.750 | 969 |
| 0.5 | 336 | 0.911 | 0.769 | 1483 |

Suggested reject threshold (>= 95% precision): 0.3

