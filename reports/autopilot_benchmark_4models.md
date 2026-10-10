# Autopilot benchmark (2026-10-08 13:08)

Detectors: yolov8x-worldv2.pt, sam3.1_l.pt, IDEA-Research/grounding-dino-base, google/owlv2-large-patch14-ensemble - quality `high` - judges: clip, sam_zoom, learned - up to 600 frames per video.

Ground truth is your hand labels. Objects your labeling policy skips (background cars, the turret commander...) count as false positives here, so precision is a lower bound.

## Per video

| Video | Frames | Tracks (real) | Raw P / R | Auto-reject P / R | After reviewing flagged P / R | Flagged |
|---|---|---|---|---|---|---|
| AchzaritTank.mp4 | 600 | 16 (1) | 0.205 / 1.000 | 0.512 / 1.000 | 0.519 / 1.000 | 14 |
| 20_INCREDIBLE_ARMORED_VEHICLES.mp4 | 600 | 118 (18) | 0.236 / 1.000 | 0.327 / 1.000 | 0.783 / 0.874 | 113 |
| BMW_X6_UavView.mp4 | 600 | 4 (3) | 0.872 / 1.000 | 0.872 / 1.000 | 0.968 / 1.000 | 1 |
| CarsInSnowDroneView.mp4 | 600 | 2 (2) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0 |
| DroneViewHuman.mp4 | 301 | 4 (1) | 0.644 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 3 |
| IDF_D9_armored_bulldozer.mp4 | 600 | 15 (1) | 0.154 / 1.000 | 0.195 / 1.000 | 0.535 / 0.538 | 15 |
| Danish_troops_firing.mp4 | 600 | 37 (9) | 0.369 / 0.879 | 0.631 / 0.771 | 0.690 / 0.825 | 33 |
| DifferentVanets.mp4 | 600 | 49 (11) | 0.381 / 0.999 | 0.438 / 0.999 | 0.808 / 0.834 | 40 |
| Challenger3Tank.mp4 | 600 | 59 (26) | 0.452 / 0.998 | 0.579 / 0.998 | 0.943 / 0.994 | 32 |
| AN_MPQ-64 Sentinel RADAR - Andrew Dutton (720p, h2 | 600 | 8 (4) | 0.524 / 0.785 | 0.000 / 0.000 | 0.814 / 0.761 | 7 |
| CV90_BrazilTank_train.mp4 | 600 | 65 (16) | 0.288 / 1.000 | 0.397 / 0.997 | 0.880 / 0.989 | 46 |
| DroneRunChicago1.mp4 | 600 | 13 (1) | 0.309 / 1.000 | 0.393 / 1.000 | 0.604 / 1.000 | 11 |
| ISrealEitanAPC1.mp4 | 600 | 28 (11) | 0.442 / 1.000 | 0.503 / 1.000 | 0.853 / 0.873 | 18 |
| INKAS_M1MilitaryVehicle.mp4 | 600 | 81 (15) | 0.196 / 1.000 | 0.327 / 1.000 | 0.766 / 0.938 | 72 |
| Israeli_bulldozers_manoeuver.mp4 | 600 | 65 (5) | 0.146 / 0.992 | 0.206 / 0.924 | 0.808 / 0.563 | 63 |
| ImaginiK2TankTrain.mp4 | 600 | 62 (13) | 0.275 / 0.985 | 0.459 / 0.985 | 0.795 / 0.959 | 47 |
| BMW_M_DRAG.mp4 | 600 | 40 (21) | 0.582 / 0.998 | 0.703 / 0.998 | 0.838 / 0.998 | 19 |
| EpicRunDroneView.mp4 | 600 | 15 (6) | 0.601 / 0.900 | 0.885 / 0.900 | 0.885 / 0.900 | 12 |
| **All** | | | **0.314 / 0.958** | **0.429 / 0.857** | **0.783 / 0.868** | |

Auto-reject column: tracks whose combined judge score is <= 0.3 removed (the default `reject_below`, as the pipeline does it).

## Detectors alone vs union (all videos)

| Detectors | Precision | Recall |
|---|---|---|
| yolov8x-worldv2.pt | 0.780 | 0.696 |
| sam3.1_l.pt | 0.446 | 0.866 |
| IDEA-Research/grounding-dino-base | 0.427 | 0.941 |
| google/owlv2-large-patch14-ensemble | 0.238 | 0.862 |
| union | 0.314 | 0.958 |

## Judges

AUC: chance a real track scores above a false one (0.5 = useless, 1.0 = perfect). For each reject threshold t: share of rejected boxes that were really false (precision), share of all false boxes removed, and real boxes lost.

### clip: AUC 0.866 over 681 tracks

| t | rejected tracks | precision | false removed | real boxes lost |
|---|---|---|---|---|
| 0.05 | 115 | 0.947 | 0.196 | 346 |
| 0.1 | 209 | 0.964 | 0.348 | 414 |
| 0.15 | 288 | 0.941 | 0.472 | 940 |
| 0.2 | 353 | 0.932 | 0.554 | 1291 |
| 0.25 | 391 | 0.909 | 0.632 | 2017 |
| 0.3 | 434 | 0.897 | 0.718 | 2623 |
| 0.4 | 477 | 0.835 | 0.792 | 5000 |
| 0.5 | 505 | 0.818 | 0.832 | 5881 |

Suggested reject threshold (>= 95% precision): 0.1

### sam_zoom: AUC 0.886 over 681 tracks

| t | rejected tracks | precision | false removed | real boxes lost |
|---|---|---|---|---|
| 0.05 | 95 | 1.000 | 0.166 | 0 |
| 0.1 | 145 | 0.860 | 0.230 | 1188 |
| 0.15 | 173 | 0.883 | 0.280 | 1188 |
| 0.2 | 195 | 0.897 | 0.325 | 1188 |
| 0.25 | 212 | 0.885 | 0.368 | 1521 |
| 0.3 | 231 | 0.879 | 0.415 | 1817 |
| 0.4 | 269 | 0.895 | 0.489 | 1817 |
| 0.5 | 304 | 0.912 | 0.593 | 1817 |

Suggested reject threshold (>= 95% precision): 0.05

### combined: AUC 0.929 over 681 tracks

| t | rejected tracks | precision | false removed | real boxes lost |
|---|---|---|---|---|
| 0.05 | 48 | 1.000 | 0.066 | 0 |
| 0.1 | 109 | 0.934 | 0.147 | 332 |
| 0.15 | 162 | 0.929 | 0.243 | 588 |
| 0.2 | 193 | 0.946 | 0.327 | 588 |
| 0.25 | 229 | 0.893 | 0.398 | 1521 |
| 0.3 | 254 | 0.890 | 0.460 | 1817 |
| 0.4 | 301 | 0.903 | 0.534 | 1817 |
| 0.5 | 351 | 0.897 | 0.665 | 2424 |

Suggested reject threshold (>= 95% precision): 0.05

### learned: AUC 0.974 over 681 tracks

| t | rejected tracks | precision | false removed | real boxes lost |
|---|---|---|---|---|
| 0.05 | 326 | 1.000 | 0.422 | 0 |
| 0.1 | 377 | 0.996 | 0.495 | 62 |
| 0.15 | 405 | 0.996 | 0.546 | 62 |
| 0.2 | 421 | 0.993 | 0.579 | 137 |
| 0.25 | 432 | 0.993 | 0.607 | 137 |
| 0.3 | 442 | 0.989 | 0.624 | 217 |
| 0.4 | 461 | 0.964 | 0.655 | 779 |
| 0.5 | 477 | 0.960 | 0.697 | 934 |

Suggested reject threshold (>= 95% precision): 0.5

## Judge time

- clip: 7 s total
- sam_zoom: 212 s total

