# Tracking robustness, automatic failure detection, and offline track repair (for unattended video auto-labeling)

Scope note: covers (a) making SAM 2 / SAM 3-style mask propagation and SOT trackers drift less, (b) flagging likely-bad track segments without ground truth, and (c) repairing fragmented / switched / gappy tracks offline, plus the preconditions (shot cuts, camera motion, aerial small objects). Pipeline architecture (Grounded-SAM-2, DEVA, MASA, SAM2MOT as a pipeline), detector verification, review-triage UX, data engines and tool surveys belong to other researchers and appear here only where they intersect. All dates are as of September 2026. Primary sources were read directly (arXiv PDFs and official repos) unless noted.

## Q1. Training-free / drop-in memory improvements for SAM 2 / SAM 2.1 / SAM 3 tracking (SAMURAI, DAM4SAM/SAM2.1++, SAM2Long, 2025–2026 successors, EfficientTAM/EdgeTAM)

### Takeaway
Training-free memory-management add-ons reliably improve SAM 2.1 on single-object tracking benchmarks at little speed cost: SAMURAI +5.7 LaSOT AUC with runtime unchanged; DAM4SAM/SAM2.1++ +5.1 LaSOT AUC and +0.048 VOT2020 EAO at about 10–20% slowdown; SAM2Long +3.0 J&F on average and up to +5.3, at a 14% FPS cost. Three caveats matter more for VIAT. First, SAM 3 (Nov 2025), which VIAT already has, beats all of these add-ons on VOS benchmarks by itself (MOSEv2 60.3 vs 51–54 for the add-ons). Second, the motion-prior variants can hurt on complex-motion VOS benchmarks. Third, most are single-object designs, and licenses differ: SAM2Long is CC-BY-NC (non-commercial), and SAM 3 uses the custom "SAM License". For multi-object propagation, the relevant options are DAM4SAM's multi-target d4sm, SAM2MOT, LiAM-SAM (Sep 2026: +10.5 HOTA and 96% fewer ID switches than a naive detector+SAM2) and SAM 3/3.1's detection-anchored disambiguation.

### Cited Findings

#### SAMURAI (Yang et al., arXiv 2411.11922, v1 18 Nov 2024; Apache-2.0; code github.com/yangchris11/samurai)
- Mechanism: a linear Kalman filter on the box. Among SAM 2's three candidate masks, SAMURAI picks the one maximizing α_kf·IoU(KF-predicted box, mask) + (1−α_kf)·mask-affinity score. A frame enters the memory bank only if mask affinity, object score and motion (KF-IoU) score all exceed thresholds (τ_mask, τ_obj, τ_kf) — [SAMURAI paper](https://arxiv.org/abs/2411.11922)
- α_kf sensitivity on LaSOT (AUC): 0.00 → 72.67, 0.15 → 74.23 (best), 0.25 → 73.76, 0.50 → 72.92 — [SAMURAI paper](https://arxiv.org/abs/2411.11922)
- Ablation (LaSOT AUC): baseline 68.32; motion modeling alone 70.81; motion-aware memory selection alone 72.67; both 74.23 — [SAMURAI paper](https://arxiv.org/abs/2411.11922)
- Results: LaSOT AUC SAM2.1-L 68.54 → SAMURAI-L 74.23 (+5.69); LaSOT_ext 58.55 → 61.03 (+2.48). Gains at every size (T +2.58, S +3.57, B +4.68 LaSOT AUC). GOT-10k AO 81.7. For comparison, supervised OSTrack384 scores LaSOT 71.1, LaSOT_ext 50.5, GOT-10k AO 73.7 — [SAMURAI paper](https://arxiv.org/abs/2411.11922)
- Cost: "runtime measurements conducted on one NVIDIA RTX 4090 GPU remain consistent with the baseline"; described as real-time — [SAMURAI paper](https://arxiv.org/abs/2411.11922)
- Repo: zero-shot on SAM 2.1 weights; VOS inference script added 27 Jan 2025 and multi-GPU script 18 Feb 2025; no streaming/webcam input. The README documents single-object (box-prompted VOT) and VOS scripts and does not document multi-object support — [SAMURAI repo](https://github.com/yangchris11/samurai)
- VOS caveat: in the SAM 3 paper's VOS table, SAMURAI scores below plain SAM 2.1-L on MOSEv1 (72.6 vs 77.9), DAVIS17 (89.9 vs 90.7) and YTVOS19 (88.3 vs 89.3), but above it on MOSEv2 (51.1 vs 47.9†, zero-shot) and LVOSv2 (84.2 vs 79.6) — [SAM 3 paper](https://arxiv.org/abs/2511.16719). SAM2Dual's own evaluation puts SAMURAI below SAM2 on MOSEv2 (47.82 vs 49.33) and PUMaVOS (79.4 vs 81.0) — [SAM2Dual](https://arxiv.org/abs/2608.18640). Baselines differ across papers (for example, SAM2.1-L LVOSv2 is 79.6 in the SAM 3 paper but 84.1 in SeC's repo table — [SeC repo](https://github.com/OpenIXCLab/SeC)).

#### DAM4SAM / "SAM2.1++" (Videnović, Lukežič, Kristan; CVPR 2025, arXiv 2411.17576; IJCV 2026 extension arXiv 2509.13864; Apache-2.0; code github.com/jovanavidenovic/DAM4SAM)
- Mechanism: SAM 2's FIFO memory is replaced by (i) a Recent Appearance Memory (3 most recent appearances, updated every 5 frames and only when the predicted mask is non-empty) and (ii) a Distractor-Resolving Memory (initial frame plus 3 anchor frames). The DRM is updated only when SAM 2's predicted IoU > 0.8 and the mask area is within 20% of the median area of the last 10 frames. It is triggered when a distractor is detected (bounding-box ratio of the chosen vs. alternative masks < 0.7) — [DAM4SAM paper (HTML)](https://arxiv.org/html/2411.17576)
- Key observation: "in the frames before the failure occurs, SAM2 in fact detects such distractors in the alternative two predicted output masks"; memory corruption from empty masks during occlusion is a named failure mode — [DAM4SAM paper (HTML)](https://arxiv.org/html/2411.17576)
- Results, SAM2.1 → SAM2.1++: VOT2020 EAO 0.681 → 0.729; VOT2022 EAO 0.692 → 0.753; VOTS2024 quality 0.661 → 0.711 (S3-Track 0.722 is higher); LaSOT AUC 70.0 → 75.1; LaSOT_ext 56.9 → 60.9; GOT-10k AO 80.7 → 81.1. On the new distractor-distilled DiDi set: quality 0.649 → 0.694, vs SAMURAI 0.680, SAM2.1Long 0.646, ODTrack 0.608, Cutie 0.575 — [DAM4SAM paper (HTML)](https://arxiv.org/html/2411.17576)
- Speed: 11 FPS vs 13.3 FPS for SAM2.1 on an A100 40GB (≈20% overhead); VOT2022-RT EAO 0.635 under the 20 FPS real-time protocol — [DAM4SAM paper (HTML)](https://arxiv.org/html/2411.17576)
- IJCV extension: beats SAM2.1 on 13 benchmarks and sets SOTA on 10. Large model 13.2 → 11.8 FPS with 1.78 GB inference memory footprint; "a moderate 10% speed reduction"; speed does not depend on the number of distractors. The DAM module transfers to efficient trackers: EfficientTAM-S DiDi quality 0.587 → 0.653 (+11%) at 25 FPS. The abstract says this "matches tracking quality of the non-real-time SAM2.1-L on multiple tracking and segmentation benchmarks"; the text reports a robustness score of 0.938 vs SAM2.1-L's 0.946 with 6.5× fewer parameters. Other transfers: EdgeTAM 0.566 → 0.591 (+4%); EfficientTAM-S VOT2022 0.585 → 0.630, VOST 0.509 → 0.537, LVOSv2 0.805 → 0.817 — [DAM4SAM IJCV version](https://arxiv.org/abs/2509.13864)
- Multi-object: a multi-target implementation, d4sm (Apache-2.0 + BSD-3-Clause, SAM2.1-L/T), is public. Its README reports VOTS2025 quality 0.710 (SAM2.1-L) and 0.626 in real-time mode; no SAM 3 support is mentioned — [d4sm repo](https://github.com/alanlukezic/d4sm); pointer from [DAM4SAM repo](https://github.com/jovanavidenovic/DAM4SAM)
- On fast-moving-object datasets (FMOX), a Dec 2025 benchmark of SAM2, EfficientTAM, DAM4SAM and SAMURAI found "DAM4SAM and SAMURAI perform well on more challenging sequences" — [FMOX benchmark](https://arxiv.org/abs/2512.09633)

#### SAM2Long (Ding et al., arXiv 2410.16268, Oct 2024; ICCV 2025; license CC-BY-NC 4.0, non-commercial; SAM 2 parts Apache-2.0; code github.com/Mark12Ding/SAM2Long)
- Mechanism: P memory pathways; each branches into SAM 2's 3 candidates, with cumulative score S = Σ log(IoU_pred). The top-P pathways are kept. When max |occlusion score| < δ_conf (= 2), deliberately distinct masks are kept for diversity. Memory frames are admitted with IoU threshold δ_iou = 0.3. The best pathway is chosen after the final frame — [SAM2Long paper](https://arxiv.org/abs/2410.16268)
- Results: +3.0 J&F averaged over 24 head-to-head comparisons, up to +5.3 on long-video benchmarks — [SAM2Long paper](https://arxiv.org/abs/2410.16268). The repo lists SA-V (SAM 2-L) val 76.3 → 80.8 and test 75.5 → 80.8, and LVOS v2 83.0 → 85.2 — [SAM2Long repo](https://github.com/Mark12Ding/SAM2Long)
- Cost: in the paper's pathway ablation (RTX 3090), P = 1 (= SAM 2 baseline) gives SA-V 76.3 / LVOS 83.0 at 22 FPS and 5.1 GB. P = 3 gives 80.8 / 85.4 at 19 FPS and 5.3 GB, i.e. "only a 14% FPS slowdown, an 8% increase in GFlops, and a 4% rise in memory usage". P = 4 gives no further gain — [SAM2Long paper](https://arxiv.org/abs/2410.16268)
- Limitation stated by the authors: "primarily focuses on single-object tracking rather than carefully designed multi-object scenarios" (the appendix shows multi-object cases) — [SAM2Long paper](https://arxiv.org/abs/2410.16268)

#### 2025–2026 successors
- HiM2SAM (arXiv 2507.07603, Jul 2025; code github.com/LouisFinner/HiM2SAM): training-free hierarchical motion estimation (linear KF plus selective non-linear point-tracker refinement) and separate long-/short-term memory. LaSOT / LaSOT_ext AUC (L): 75.1 / 62.8, vs SAMURAI-L 74.2 / 61.0 and DAM4SAM-L 75.1 / 60.9. That is 9.6% / 7.2% relative over SAM2 while "preserv[ing] the real-time capability" (A6000) — [HiM2SAM](https://arxiv.org/abs/2507.07603)
- SAM2Dual (Kim & Lim, arXiv 2608.18640, 19 Aug 2026, IEEE TIP format): training-free dual memory (short-term plus interval-sampled long-term, gated fusion) plus text-aware memory reweighting. MOSEv2 49.33 → 50.65; LVOSv2 82.1 → 83.1; PUMaVOS 80.6, slightly below SAM2's 81.0. In the same table HiM2SAM scores 45.90 / 81.7 / 78.5, i.e. below SAM2 — [SAM2Dual](https://arxiv.org/abs/2608.18640)
- SAM2RL (arXiv 2507.08548, Jul 2025): RL-learned memory control. In a per-video overfitting setup its gain "exceeds by more than three times the gains of existing heuristics". This shows headroom in memory control, not a deployable method — [SAM2RL](https://arxiv.org/abs/2507.08548)
- MA-SAM2 (arXiv 2507.09577): training-free "context-aware and occlusion-resilient memory" for surgical video — [MA-SAM2](https://arxiv.org/abs/2507.09577)
- SeC (arXiv 2507.15852, Jul 2025; ICLR 2026; Apache-2.0; SeC-4B LVLM): builds an object "concept" with an LVLM, invoked only when a new scene appears. SeCVOS 70.0 vs SAM 2.1 58.2 (+11.8), SAM2.1Long 62.3, SAMURAI 62.2; MOSEv2 53.8; SA-V val 82.7; LVOSv2 86.5 — [SeC repo](https://github.com/OpenIXCLab/SeC); [SeC paper](https://arxiv.org/abs/2507.15852)
- LiAM-SAM (arXiv 2609.28078, 23 Sep 2026; contracted by Toyota Motor Europe) is a detector-prompted SAM2 MOT with lifecycle-aware memory, targeting three failure stages: (i) erroneous or duplicate track initiation, via contrastive track initiation; (ii) memory drift during close interactions, via motion- and geometry-grounded memory correction; and (iii) unreliable re-identification after long occlusion, via adaptive context memory. The tracker itself is training-free — [LiAM-SAM](https://arxiv.org/abs/2609.28078)
  - DanceTrack test: HOTA 79.6 / IDF1 83.6 / AssA 72.6, vs the naive SAM2 + RF-DETR baseline at 64.3 / 67.0 / 52.1 (+15.25 HOTA, 96% fewer IDSW). With ByteTrack's YOLOX detector: HOTA 76.5 vs MOTRv2 69.9. SportsMOT HOTA 81.8 / IDF1 86.4; BDD100K val IDF1 73.4 — [LiAM-SAM](https://arxiv.org/abs/2609.28078)
  - Ablation (DanceTrack val): IDSW 36,210 → 1,416. The motion-grounding component alone cuts IDSW by 87.8–95.0% — [LiAM-SAM](https://arxiv.org/abs/2609.28078)
  - Cost: 5 FPS with 4 objects and 2 FPS with 30 objects on an A100. Token pruning (SASP) raises this to 10 / 5 FPS at −1.9 HOTA (78.2 → 76.3) — [LiAM-SAM](https://arxiv.org/abs/2609.28078)
- SAM2MOT (Huawei Cloud, arXiv 2504.04519, v5 Nov 2025; code github.com/TripleJoy/SAM2MOT): zero-shot, detector-prompted SAM2.1-L MOT. DanceTrack HOTA 75.5–75.8 / IDF1 83.4–83.9 (+2.1 HOTA, +4.5 IDF1 over the prior best) — [SAM2MOT](https://arxiv.org/abs/2504.04519). LiAM-SAM notes that SAM2MOT "reports an offline optimization with video-specific thresholds" — [LiAM-SAM](https://arxiv.org/abs/2609.28078)

#### SAM 3 / SAM 3.1 (Meta; SAM 3 arXiv 2511.16719, Nov 2025; SAM 3.1 released 27 Mar 2026; custom "SAM License" for code and checkpoints; checkpoints gated on Hugging Face)
- VOS (J&F), SAM 3 vs SeC / SAM2Long / SAMURAI / SAM 2.1-L: MOSEv2 60.3 vs 53.8 / 51.5 / 51.1 / 47.9†; LVOSv2 88.5 vs 86.5 / 85.9 / 84.2 / 79.6; SA-V test 84.4 vs 81.7 / 81.2 / 80.0 / 78.4 — [SAM 3 paper](https://arxiv.org/abs/2511.16719)
- Robustness logic inside SAM 3's video pipeline (masklet detection score, confirmation delay, suppression, periodic and detection-guided re-prompting) is detailed under Q2/Q3. Turning all of it off costs 2.6–4.9 cgF1 on SA-Co/VEval val splits — [SAM 3 paper](https://arxiv.org/abs/2511.16719)
- SAM 3.1 "Object Multiplex" groups objects into fixed-capacity buckets (up to 16 objects per forward pass). It gives ~7× speedup at 128 objects on one H100 vs SAM 3 and is a drop-in replacement. Release-note metrics SAM 3 → 3.1: YT-Temporal-1B cgF1 50.8 → 52.9, MOSE v2 83.5 → 83.8, DAVIS17 84.4 → 85.1, LVOS v2 89.7 → 89.3, YTVIS21 mAP 57.4 → 56.6 — [SAM 3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md). These release-note numbers do not match the SAM 3 paper's VOS table (e.g. MOSEv2 60.3, DAVIS17 92.2), so they presumably use a different protocol — treat with caution.
- License and access: "SAM License" for code and checkpoints; access requires approval on Hugging Face — [SAM 3 repo](https://github.com/facebookresearch/sam3)
- Field report (multi-animal, Nov 2025):
  - Zero-shot SAM 3 matches the authors' SAM2MOT-based Grounding-DINO + SAM 2 pipeline in HOTA (ChimpAct 58.9 vs 58.6; BFT 75.4 vs 74.8). However, SAM 3 ran out of memory on an H100 with 96 GB even with offload_video_to_cpu and offload_state_to_cpu on some AnimalTrack sequences — [Zero-shot multi-animal tracking](https://arxiv.org/abs/2511.02591)
  - SAM 3 also "must be run n times for n text prompts", which prevents cross-class duplicate resolution, and is sensitive to prompt wording — [Zero-shot multi-animal tracking](https://arxiv.org/abs/2511.02591)

#### Efficiency variants (speed/VRAM, not robustness)
- SAM 2 baseline on A100: tiny 91.2, small 84.8, base+ 64.1, large 39.5 FPS; Apache-2.0; SAM 2.1 released 29 Sep 2024; supports independent per-object inference — [SAM 2 repo](https://github.com/facebookresearch/sam2)
- EfficientTAM (arXiv 2411.18933, Nov 2024; ICCV 2025; Apache-2.0): roughly SAM 2 accuracy (SA-V test 74.5 vs 74.7) with ~2× speedup on A100 and ~2.4× fewer parameters; ~10 FPS on iPhone 15 Pro Max — [EfficientTAM paper](https://arxiv.org/abs/2411.18933); [repo](https://github.com/yformer/EfficientTAM)
- EdgeTAM (arXiv 2501.07256; CVPR 2025; Apache-2.0): J&F 87.7 DAVIS17, 70.0 MOSE, 72.3 SA-V val, 71.7 SA-V test; 16 FPS on iPhone 15 Pro Max (22× over SAM 2); 150.9 FPS on A100; SAM 2-like predictor API — [EdgeTAM paper](https://arxiv.org/abs/2501.07256); [repo](https://github.com/facebookresearch/EdgeTAM)
- Efficient-SAM2 (ICLR 2026, arXiv 2602.08224): 1.68× speedup of SAM2.1-L with a 1.0% SA-V test drop, using sparse window routing plus sparse memory retrieval; needs "minimal training" — [Efficient-SAM2](https://arxiv.org/abs/2602.08224)

#### Registry (one-line summary; details and sources above)
| Method | Date / venue | Code / license | Multi-object | Headline gain | Cost |
|---|---|---|---|---|---|
| SAMURAI | Nov 2024 arXiv | yangchris11/samurai, Apache-2.0 | single-object documented | LaSOT AUC +5.7 (L) | ≈ baseline speed (RTX 4090) |
| DAM4SAM / SAM2.1++ | CVPR 2025; IJCV 2026 | DAM4SAM Apache-2.0; d4sm Apache-2.0/BSD-3 | yes (d4sm) | VOT2020 EAO +0.048; LaSOT +5.1 | −10–20% FPS; 1.78 GB (L) |
| SAM2Long | ICCV 2025 | CC-BY-NC 4.0 (non-commercial) | mainly single | +3.0 J&F avg, ≤ +5.3 | −14% FPS, 5.3 GB (3090) |
| HiM2SAM | Jul 2025 arXiv | LouisFinner/HiM2SAM (license not verified) | single | LaSOT_ext AUC 62.8 | "real-time preserved" |
| SAM2Dual | Aug 2026 arXiv | code not found | VOS | MOSEv2 +1.3 | not reported |
| SeC | ICLR 2026 | Apache-2.0, 4B LVLM | VOS | SeCVOS +11.8 vs SAM 2.1 | LVLM (heavy) |
| LiAM-SAM | Sep 2026 arXiv | code/license not found | yes (MOT) | DanceTrack +15.25 HOTA, −96% IDSW | 2–5 FPS (A100) |
| SAM 3 / 3.1 | Nov 2025 / Mar 2026 | SAM License, gated | yes (native) | MOSEv2 60.3 J&F | 3.1: ~7× faster at 128 objects (H100) |

### Inferences
- VIAT already has SAM 3 native, which outscores every SAM 2.1 add-on on VOS benchmarks. So for prompt→track in VIAT, upgrading to SAM 3.1 plus its detection-anchored logic probably matters more than porting SAMURAI/DAM4SAM. The add-ons remain relevant for the SAM2-TRT backend and for single-object box-prompted tracking, where SAM2-TRT is fast. This is an inference; no paper applies SAMURAI/DAM4SAM on top of SAM 3's tracker.
- DAM4SAM is the most practical add-on for VIAT: permissive license, multi-target d4sm code, 13-benchmark validation, and it also transfers to EfficientTAM/EdgeTAM. Its update gate (predicted IoU > 0.8 AND area within 20% of the 10-frame median) doubles as a ready-made per-frame reliability flag for Q2.
- SAM2Long's design (keep K hypotheses, pick the best path at the end of the video) is naturally offline and fits VIAT's whole-video setting. However, its CC-BY-NC license blocks commercial use of that code, so VIAT would need a clean-room reimplementation of the idea.
- Kalman/motion priors in image coordinates (SAMURAI, HiM2SAM) help on LaSOT-style SOT but are unproven, and sometimes harmful, on complex-motion VOS. On drone footage the camera moves, so any motion prior or velocity-outlier flag should run on camera-motion-compensated coordinates (see Q6).
- Cost scales with object count for SAM-style trackers (LiAM-SAM: 2 FPS at 30 objects; SAM 3 OOM on dense animal videos at 96 GB). For VisDrone-style scenes with hundreds of small objects, SAM-based propagation for every object is likely impractical. A better plan is detector-driven box MOT plus offline repair (Q4), with SAM propagation reserved for flagged or gappy segments. SAM 3.1 multiplexing should be benchmarked on VIAT's GPUs before deciding.

### Gaps
- No published evaluation of SAMURAI / DAM4SAM / SAM2Long-style memory policies grafted onto SAM 3 or SAM 3.1.
- No aerial/drone-specific evaluation (e.g., UAV123, VisDrone-SOT) of these SAM 2 memory variants was found.
- Code and license could not be found for LiAM-SAM or SAM2Dual; HiM2SAM and SAM2MOT licenses were not verified.
- SAM 3.1 VRAM usage is not published, and its release-note metrics conflict with the SAM 3 paper's protocol.
- No reliability or failure-rate analysis was found for VIAT's other SOT backends (OSTrack, E.T.Track, OpenCV CSRT/KCF) as label propagators, beyond OSTrack's LaSOT/GOT-10k numbers above.

## Q2. Automatic failure detection without ground truth (forward-backward / cycle consistency, SAM 2/3 internal scores, geometry jumps, velocity outliers, appearance drift, detector agreement)

### Takeaway
The only signal family with published precision/recall for failure detection is the classic forward-backward (time-reversibility) check. Kalal et al. 2010 report 95% recall and 96% precision for identifying correct point trajectories at a 1-pixel threshold. SAM 2's own predicted-IoU head is demonstrably unreliable as a stand-alone error detector: about 0.74 peak accuracy at telling whether true IoU > 0.5 on VOST, with a mis-calibrated best threshold of ~0.8. Working systems therefore combine several signals:
- temporal statistics of the model's logits (variance, dwell time below thresholds);
- agreement with an independent detector over a window (SAM 3's masklet detection score; SAM2BMP's key-frame-ratio filter, +2.7 AP);
- geometric consistency (area vs. running median; frame-to-frame IoU and size jumps);
- within-track appearance consistency (ReID-embedding clustering to split ID-switched tracklets, as in GTA).

No paper found reports precision/recall for these combined signals on SAM-style mask tracks.

### Cited Findings

#### Forward-backward / time-reversibility (classics)
- Forward-Backward error (Kalal, Mikolajczyk, Matas, ICPR 2010) tracks a point forward k steps, then backward, and measures the distance between the two trajectories.
  - Result: with a 1 px threshold, "recall is of 95% and precision of 96%" at detecting correct trajectories (inlier base rate 65%). SSD "was unable to detect inliers for small thresholds, its precision starts below 70%" — [Kalal et al. 2010 (PDF)](http://kahlan.eps.surrey.ac.uk/featurespace/tld/Publications/2010_icpr.pdf); [IEEE Xplore](https://ieeexplore.ieee.org/document/5596017/)
  - FB error is complementary to NCC and underlies the Median Flow tracker, which uses the 50% most reliable points — [Kalal et al. 2010](http://kahlan.eps.surrey.ac.uk/featurespace/tld/Publications/2010_icpr.pdf)
  - Caveat: this was evaluated for point trajectories, not masks or boxes.
- Time-reversed Markov chain (Wu, Sankaranarayanan, Chellappa; CVPR 2007 / TPAMI 2010) evaluates a particle-filter tracker online by filtering back to t = 0 from the posterior at time t. The distance between the back-filtered density and the initial prior is the failure statistic, because occlusion, low SNR or modeling error break time-reversibility — [ML Anthology (CVPR 2007)](https://mlanthology.org/cvpr/2007/wu2007cvpr-situ/); [TPAMI 2010 (PubMed)](https://pubmed.ncbi.nlm.nih.gov/20558876/)

#### Reliability of SAM 2 / SAM 3 internal signals
- Live Interactive Training (Cornell, arXiv 2603.26929, 27 Mar 2026) tested SAM2's predicted IoU as an automatic error detector against ground-truth IoU at 0.5 — [LIT paper](https://arxiv.org/html/2603.26929v1)
  - Peak accuracy was "only ~0.74" on VOST. The best predicted-IoU threshold (~0.8) was far from 0.5; FPR rises sharply at high thresholds and FNR is high at low thresholds.
  - Using predicted IoU both to trigger and to validate corrections reduced correction efficiency relative to full user supervision. The authors conclude "SAM2 lacks a robust internal quality estimator".
  - Separately, LIT's own online adaptation cut corrections by 18–34% and annotation time by 22.1% (41.92% fewer corrections in a user study).
- SAM2Long uses the SAM 2 occlusion score (o_t > 0 means the object is present); |o_t| < δ_conf = 2 is treated as "uncertain" — [SAM2Long](https://arxiv.org/abs/2410.16268)
  - Its memory admission uses predicted-IoU ≥ δ_iou. On SA-V: no filter (δ_iou = 0) gives 80.0; thresholds 0.1–0.7 give 80.6–81.0; 0.8 gives 80.0 and 0.9 gives 77.8 (the SAM 2 baseline is 76.3). Predicted IoU works as a coarse junk filter, not a precise quality score.
- DAM4SAM does not trust predicted IoU alone either: its memory gate additionally requires mask area within 20% of the 10-frame median. Its distractor detector compares the chosen mask against SAM 2's alternative masks (bbox ratio < 0.7) — [DAM4SAM (HTML)](https://arxiv.org/html/2411.17576)
- SAM2MOT turns SAM2 object-score logits into track states — [SAM2MOT](https://arxiv.org/abs/2504.04519)
  - States: reliable (> 8.0), pending (6.0–8.0; triggers "quality reconstruction" from a matched high-confidence detection), suspicious (2.0–6.0) and lost (≤ 2.0). An object is removed after 25 frames lost.
  - When two masks overlap by more than 0.8, it computes the variance of each object's logits over the last N = 10 frames. A sudden drop (high variance) identifies the object that was wrongly taken over, and that object's memory update is skipped.
- SAM 3's Masklet Detection Score (MDS) uses an independent detector as the check — [SAM 3 paper](https://arxiv.org/abs/2511.16719)
  - Definition: Δ_i(τ) = +1 if some detection has IoU > threshold with masklet i on frame τ, else −1; MDS is the sum over a window. With a confirmation delay of T = 15 frames (≈0.5 s at 30 fps), a new masklet is removed if MDS < 0 in its first window, i.e. it must match a detection in at least half of those frames.
  - Duplicates (two masklets matching the same detection with IoU ≥ 0.1 for ≥ ⌈T/2⌉ frames) are removed, keeping the older one. Confirmed masklets whose lifetime MDS drops below 0 are suppressed (output zeroed) but kept in state.
  - Ablation of all temporal-disambiguation strategies together: SA-Co/VEval SA-V val cgF1 24.4 → 29.3, pHOTA 57.4 → 60.7; YT-Temporal-1B val cgF1 47.0 → 50.2, pHOTA 68.6 → 70.5; SmartGlasses val cgF1 30.9 → 33.5, pHOTA 58.1 → 60.2. A detector plus a tracking-by-detection tracker instead of the SAM 3 tracker reaches only pHOTA 57.6 on SA-V val.
- SAM2BMP (ICCVM 2026) filters with a detector-support ratio — [SAM2BMP](https://iccvm.org/2026/files/papers/517.pdf)
  - A "Key Frame Ratio Filter" removes trajectories whose share of frames with high-confidence detector support is below 0.1. It adds +2.7 AP on YouTube-VIS 2021 (R50 Mask2Former + SAM2-L).
  - Tracklet NMS (IoU > 0.5 over ≥ 6 frames, keeping the track with more key frames) adds +0.4 AP.
  - Raising the detector foreground threshold from 0.60 to 0.95 raised AP from 49.9 to 52.7, but 0.97 lowered it to 52.0.

#### Geometry, velocity and appearance statistics
- The AWS SageMaker Ground Truth video-tracking QA recipe (22 Apr 2021) flags three things, and routes flagged frames to targeted audit jobs — [AWS ML blog](https://aws.amazon.com/blogs/machine-learning/quality-assessment-for-sagemaker-ground-truth-video-multi-frame-object-tracking-annotations-using-statistical-analysis/)
  - rolling IoU between consecutive boxes below ~0.5;
  - box-size change > 20% between frames;
  - ResNet-18 (ImageNet) embedding distance between consecutive crops > 2σ from the mean.
  - It reports no effectiveness numbers, only a qualitative demo on injected errors.
- GTA's "tracklet splitter" detects ID-switched (impure) tracklets by DBSCAN clustering of per-frame OSNet ReID embeddings inside each tracklet (min samples 5, ε = 0.6, max 3 clusters) — [GTA paper](https://arxiv.org/abs/2411.08216)
  - On top of the connector, the splitter adds +0.2 to +1.1 HOTA and +0.3 to +4.5 IDF1 across SORT / ByteTrack / Deep-EIoU on SportsMOT and SoccerNet (e.g. SoccerNet ByteTrack IDF1 78.22 → 82.67).
- Learned tracklet scoring without ground truth: Maksai & Fua (CVPR 2019) train a bi-LSTM scoring function over box coordinates, confidences and motion shifts as an explicit proxy for IDF1 that "can be computed without ground truth" — [Maksai & Fua, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Maksai_Eliminating_Exposure_Bias_and_Metric_Mismatch_in_Multiple_Object_Tracking_CVPR_2019_paper.pdf)
- Failure taxonomy for detector-prompted SAM2 MOT: (1) erroneous or duplicate track births, (2) memory drift, mask bleeding and ID swaps during close interactions, (3) broken re-ID after long occlusion or exit/re-entry — [LiAM-SAM](https://arxiv.org/abs/2609.28078)
- UAV motion statistics differ: FOLT shows mean relative acceleration (normalized by object size) is much higher on VisDrone than on MOT17/20, and VisDrone objects are smaller — [FOLT (ACM MM 2023)](https://arxiv.org/abs/2308.07207)
- Track-level human verification is effective: tracklets built by MHT over low-threshold detections were verified by operators. This reduced annotation workload "up to 96%" on the AUTH Multidrone dataset — [Ince et al. 2021](https://arxiv.org/abs/2101.06977)

### Inferences
- A practical, ground-truth-free risk score for VIAT tracks could combine the signals below. Each has published use as a component, but none has published precision/recall as an error detector for mask tracks:
  - (a) running MDS against any independent detector VIAT already has (YOLO / YOLO-World / SAM 3 detector), with long negative runs flagged;
  - (b) SAM logit state and its short-window variance (SAM2MOT thresholds as starting values);
  - (c) area or aspect-ratio deviation from a running median (DAM4SAM's 20%/10-frame rule) and box-IoU / size jumps (AWS's 0.5 / 20% rules);
  - (d) velocity/acceleration outliers computed after camera-motion compensation;
  - (e) within-track appearance multimodality (GTA-style DBSCAN on ReID or foundation embeddings) and distance to the track's first high-confidence template;
  - (f) forward–backward mask IoU (see Q3).
- Treat SAM 2's predicted IoU and object score only as weak features. Their thresholds are dataset-dependent (best predicted-IoU cut ~0.8 on VOST, while SAM2Long's memory gate works best near 0.1–0.7).
- Rank review items at the track-segment level, not per frame. The Multidrone result (96% workload reduction from tracklet-level verification) and SAM 3's windowed MDS both argue for aggregating evidence over windows of about 15–30 frames before asking a human.
- Thresholds should be adapted per sequence. The multi-animal SAM2MOT study found per-sequence adaptive detection thresholds beat any single global threshold (ChimpAct HOTA 56.7 vs 54.6 at 0.4) — [Zero-shot multi-animal tracking](https://arxiv.org/abs/2511.02591). This matters for unattended runs over heterogeneous footage.

### Gaps
- No precision/recall or AUROC was found for these as predictors of real tracking errors on modern benchmarks: detector-support ratio or MDS, area/aspect jumps, velocity outliers, embedding drift against the initial template (DINOv2/DINOv3, CLIP/SigLIP, OSNet), or forward-backward mask IoU with SAM 2/3. Searches for DINOv3-based ReID for MOT returned only general claims, not quantified comparisons with OSNet.
- The reliability of SAM 3's own per-masklet scores as error predictors has not been studied (the LIT analysis covers SAM 2 only).
- The confidence outputs of OSTrack / E.T.Track / CSRT (e.g., peak score, PSR) were not researched here; no source was gathered.
- The Maksai & Fua PDF could not be downloaded, so the accuracy of its IDF1 proxy is not reported here (only its existence, from the search snippet and CVF listing).

## Q3. Bidirectional / ensemble propagation: forward from first appearance and backward from last; re-seeding from high-confidence keyframes; how the two directions are merged

### Takeaway
Propagating in both temporal directions and anchoring on high-confidence keyframes is consistently beneficial wherever it has been measured:
- CTRL's bidirectional track extension cut totally-missed vehicles by ~76% (21.4k → 5.1k).
- DetZero fuses forward- and reverse-order tracker outputs with weighted box fusion.
- In 2D, SAM2BMP's backward SAM 2 completion was its single largest gain (+5.3 AP on YouTube-VIS 2021).

Merging is usually simple:
- concatenation of two unidirectional passes (SAMannot);
- stop-at-anchor rules (MiVOS stops propagation at previously corrected frames);
- a learned difference-aware fusion, which adds only +0.3–0.5 AUC-J&F over linear blending (MiVOS);
- detection-anchored re-seeding: SAM 3 re-prompts every 16 frames from detections with IoU ≥ 0.8 and scores ≥ 0.8, and re-conditions whenever tracker/detector box IoU < 0.85.

### Cited Findings
- SAM2BMP ("Video Instance Segmentation via Bidirectional Mask Propagation with SAM2", ICCVM 2026) — [SAM2BMP](https://iccvm.org/2026/files/papers/517.pdf)
  - Method: a forward phase propagates SAM2 masks and matches them to Mask2Former detections by mask IoU; a backward phase propagates each track's mask back from its first detection, because "small or partially occluded objects may not produce high-confidence detections initially". Low-confidence detections become new tracks only if IoU < 0.3 with all tracks.
  - Ablation on YTVIS21 (R50 + SAM2-L): 43.2 AP → +backward completion 48.5 (+5.3) → +key-frame-ratio filter (+2.7) → +low-confidence proposals (+1.1) → +tracklet NMS (+0.4) → 52.7 (+9.5 total).
  - Throughput with an R50 detector: ~7–7.7 FPS and ~5.8–6.3 (memory units as reported) for SAM2 T/B+/L. Swin-L 66.7 / 62.0 AP and ViT-L 69.2 / 66.6 AP on YTVIS19/21.
- CTRL ("Once Detected, Never Lost", Fan et al., arXiv 2304.12315, Apr 2023; code github.com/tusen-ai/SST) is modeled on how human annotators work: they first label clear frames of a track, then use temporal coherence to label obscure ones. After forward tracking with an "immortal" tracker, tracks are extended backward with a backward motion model, and the whole track is refined at once — [CTRL](https://arxiv.org/abs/2304.12315)
  - On Waymo (L2) vehicles, totally-missed FNs fell from 21.4k (2.05%) with refinement only, to 9.1k (0.87%) with forward extension, to 5.1k (0.48%) with bidirectional extension. APH went 80.1 → 80.7 → 81.0; pedestrians' totally-missed rate fell from 2.20% to 0.53%.
- DetZero (arXiv 2306.06023, Jun 2023; code github.com/PJLab-ADG/DetZero) uses "forward and reverse order tracking fusion": the offline tracker runs in both time orders, tracks are paired by a location-aware similarity, and pairs are fused with WBF "to further ameliorate the missing boxes and stable the motion state". Tracks that are too short skip refinement — [DetZero](https://arxiv.org/abs/2306.06023)
- MiVOS (Cheng, Tai, Tang, arXiv 2103.07941, 2021): from each user-corrected frame, masks are propagated in two independent passes (forward and backward), stopping when they reach a previously interacted frame — [MiVOS](https://arxiv.org/abs/2103.07941)
  - Conflicts between rounds are resolved by a learned "difference-aware fusion" that uses the positive and negative mask changes of the user's correction, aligned by the space-time memory affinity.
  - Ablation (DAVIS interactive, AUC-J&F): learnable fusion 87.6 vs 87.4 before it (linear-weighting baseline); difference-aware 87.9. "Perfect interaction" upper bound: 90.2.
- XMem++ (arXiv 2307.15958, 2023) adds a "permanent memory" so that multiple user-annotated keyframes with differing appearances are all used during propagation. An attention-based frame-suggestion mechanism proposes "the next best frame for annotation". It also introduced the PUMaVOS benchmark — [XMem++](https://arxiv.org/abs/2307.15958)
- SAMannot (arXiv 2601.11301, Jan 2026; MIT; local SAM2 annotation tool) supports forward, backward and bidirectional propagation; bidirectional "executes two separate unidirectional passes and concatenates the results". Users can set checkpoints that block propagation at specific frames. A ≥ 6 GB VRAM GPU is recommended — [SAMannot](https://arxiv.org/abs/2601.11301)
- SAM 3 anchor-based re-seeding — [SAM 3 paper](https://arxiv.org/abs/2511.16719)
  - Periodic re-prompting: every N = 16 frames the tracker is re-initialized from a detection's mask if IoU(detection, tracked mask) ≥ 0.8 and both detection and masklet scores ≥ 0.8. It is "most effective on frames where objects are not occluded and are fully visible".
  - Detection-guided re-prompting: re-condition when the best-matching detection has box IoU < 0.85 with the tracked mask, to fix drift and leaky masks.
- SAM2MOT "quality reconstruction": when a track is in the "pending" logit band and matches a high-confidence detection, that detection box replaces the track's keyframe prompt information — [SAM2MOT](https://arxiv.org/abs/2504.04519)
- Seq-NMS (Han et al., arXiv 1602.08465, 2016 classic) links detections across the whole clip by dynamic programming and rescores each linked sequence. On ImageNet VID full val (VGG): NMS 44.9 → Seq-NMS (avg rescoring) 51.4 → combination 52.2 mAP — [Seq-NMS](https://arxiv.org/abs/1602.08465)

### Inferences
- For VIAT's offline mode, a sound default is:
  1. find high-confidence anchor frames per object (detector score and SAM score ≥ ~0.8, unoccluded — SAM 3's re-prompt criteria);
  2. propagate from each anchor in both directions up to the neighboring anchors (MiVOS / SAMannot "stop at checkpoint" semantics);
  3. in each inter-anchor interval, compare the forward mask from the left anchor with the backward mask from the right anchor.
  The per-frame IoU between the two directions is a direct mask analog of Kalal's FB error. Where the directions agree, accept; where they disagree, prefer the nearer anchor or the one with detector support, and raise a review flag.
- Backward completion from the first confident detection should be the default for objects entering the scene. In both SAM2BMP (2D) and CTRL (3D) it was the largest single improvement, which fits VIAT's pain point with objects entering and leaving.
- Learned fusion adds little over simple rules (+0.3–0.5 in MiVOS), so a rule-based merge (agreement check, nearest anchor, detector support) is a reasonable first implementation.

### Gaps
- No 2D paper quantifies forward-vs-backward mask disagreement as a predictor of annotation errors (precision/recall).
- No study was found of multi-tracker voting ensembles (e.g., SAM 2 + SAM 3 + OSTrack + CSRT) for label QA.
- DetZero's ablation does not isolate the forward/reverse fusion gain from its other modules.
- The bidirectional logic of commercial labeling tools (CVAT, Encord, V7 and others) was not examined here; that belongs to the tools-survey researcher.

## Q4. Offline MOT post-processing: tracklet stitching / global association, interpolation, smoothing, short-tracklet removal, class voting, ReID-based ID-switch repair — with measured gains

### Takeaway
Cheap offline repairs give measured, largely additive gains:
- **Interpolation.** Linear interpolation gives about +0.8–2.0 MOTA and +0.2–1.3 IDF1. Gaussian-smoothed interpolation (GSI) adds 0 to +1.9 MOTA and up to +1.0 IDF1 on top of linking, and stabilizes velocity.
- **Appearance-free global linking.** AFLink gives +0.2–3.7 IDF1 at about 1.7 ms/frame, helping weak trackers most, but it cannot split ID-switched tracks.
- **ReID split-and-connect.** GTA gives +3.7 to +10.2 HOTA and +6.3 to +18.5 IDF1 on sports data.
- **Full offline graph trackers.** SUSHI and NOOUGAT beat online trackers on association (NOOUGAT offline DanceTrack HOTA 68.4 vs SUSHI 63.3; SportsMOT 85.6).
- **Class consistency.** Video-level class suppression and rescoring give +2–7 mAP on ImageNet VID (classic).

Licenses vary and matter: StrongSORT is GPL-3.0; BoxMOT and Ultralytics are AGPL-3.0; GTA, SUSHI, supervision and torchreid are MIT; FastReID is Apache-2.0.

### Cited Findings

#### Interpolation, smoothing and appearance-free linking (StrongSORT++, TMM 2023; GPL-3.0)
- AFLink links two tracklets using only spatio-temporal inputs: a two-branch 1D-conv classifier over frame, x, y — [StrongSORT paper](https://arxiv.org/abs/2202.13514)
  - It is trained in "just over 10 seconds" on MOT17 tracklets cut with noise. Candidate pairs are gated at 30 frames and 75 px, and a pair is linked if the score > 0.95 (0.9 on DanceTrack).
  - Cost: 1.7 ms per image, reported in the repo as 591.9 Hz on MOT17 — [StrongSORT repo](https://github.com/dyhBUPT/StrongSORT)
- GSI fits Gaussian-process regression to each linearly interpolated trajectory with adaptive smoothness λ = τ·log(τ³/l), τ = 10. The maximum gap is 20 frames (5 on DanceTrack). Cost: 7.1 ms per image (140.9 Hz) — [StrongSORT paper](https://arxiv.org/abs/2202.13514)
- Plug-and-play ablation on MOT17 val (IDF1 / MOTA / HOTA) — [StrongSORT paper](https://arxiv.org/abs/2202.13514):

| Tracker | Base | +AFLink | +AFLink+GSI (Δ vs base) |
|---|---|---|---|
| StrongSORTv1 | 79.5 / 76.8 / 67.8 | 80.0 / 76.8 / 68.1 | 80.4 / 78.2 / 68.9 (+0.9 / +1.4 / +1.1) |
| StrongSORTv6 | 82.3 / 77.1 / 69.6 | 82.5 / 77.1 / 69.6 | 83.3 / 78.7 / 70.8 (+1.0 / +1.6 / +1.2) |
| CenterTrack | 64.6 / 66.8 / 55.3 | 68.3 / 66.9 / 57.2 | 68.4 / 66.9 / 57.6 (+3.8 / +0.1 / +2.3) |
| TransTrack | 68.6 / 67.7 / 58.1 | 69.1 / 67.7 / 58.3 | 69.9 / 69.6 / 59.4 (+1.3 / +1.9 / +1.3) |
| FairMOT | 72.7 / 69.1 / 57.3 | 73.2 / 69.2 / 57.6 | 74.2 / 71.1 / 59.0 (+1.5 / +2.0 / +1.7) |

- "Poorer trackers tend to benefit more from AFLink", while GSI "works better on stronger trackers". Stated limitation: AFLink "is helpless against false association problems… cannot split mixed-up ID trajectories" — [StrongSORT paper](https://arxiv.org/abs/2202.13514)
- MOT17 test: StrongSORT HOTA 63.5 / IDF1 78.5 / MOTA 78.3 / IDs 1,446; +AFLink 63.7 / 79.0 / 78.3 / 1,401; +AFLink+GSI (StrongSORT++) 64.4 / 79.5 / 79.6 / 1,194. DanceTrack test StrongSORT++: HOTA 55.6, IDF1 55.2 — [StrongSORT paper](https://arxiv.org/abs/2202.13514)
- Linear interpolation alone:
  - ByteTrack (ECCV 2022) with gaps up to σ = 20 frames: MOTA 76.6 → 78.3, IDF1 79.3 → 80.2 on MOT17 val — [ByteTrack](https://arxiv.org/abs/2110.06864)
  - On MOT17 test, without → with linear interpolation: ByteTrack HOTA 62.8 → 63.2, IDF1 77.2 → 77.4, MOTA 78.9 → 79.7; OC-SORT 61.7 → 63.2, 76.2 → 77.5, 76.0 → 78.0 — [StrongSORT paper, Table IV](https://arxiv.org/abs/2202.13514)

#### ReID-based split-and-connect (GTA, ACCV 2024 workshop, arXiv 2411.08216; MIT; code github.com/sjc042/gta-link)
- Model-agnostic refinement of any MOT-format output — [GTA](https://arxiv.org/abs/2411.08216); [repo](https://github.com/sjc042/gta-link)
  - Splitter: DBSCAN on per-detection OSNet embeddings (OSNet trained on SportsMOT) separates multi-identity tracklets.
  - Connector: iteratively merges tracklet pairs by average cosine distance (threshold α = 0.4) with a spatial factor.
- Gains (HOTA / IDF1 / IDs) — [GTA](https://arxiv.org/abs/2411.08216):
  - SportsMOT: SORT +10.24 HOTA / +18.54 IDF1 (IDs 5,180 → 3,547); ByteTrack +6.28 / +12.40 (3,147 → 2,107); Deep-EIoU +3.83 / +6.70, reaching HOTA 81.04.
  - SoccerNet: SORT +6.84 / +12.68; ByteTrack +4.67 / +9.45; Deep-EIoU 79.41 → 83.11 HOTA (+3.70) / +6.26 IDF1.
  - The connector alone gives most of the gain (e.g. SORT SportsMOT +9.15 HOTA); the splitter adds +0.2–1.1 HOTA.
- GTATrack (arXiv 2602.00484, Jan 2026; code github.com/ron941/GTATrack-STC2025) combines Deep-EIoU online association, GTA refinement and detector pseudo-labeling. It won the SoccerTrack Challenge 2025 with HOTA 0.60 and cut false positives to 982 — [GTATrack](https://arxiv.org/abs/2602.00484)

#### Offline / global graph trackers
- SUSHI (Cetintas, Brasó, Leal-Taixé; CVPR 2023; MIT; code github.com/dvl-tum/SUSHI) — [SUSHI paper](https://arxiv.org/abs/2212.03038); [repo](https://github.com/dvl-tum/SUSHI)
  - Splits long clips into a hierarchy of subclips and uses GNNs at each temporal scale.
  - Results (HOTA / IDF1 / MOTA): MOT17 private 66.5 / 83.1 / 81.1; MOT20 64.3 / 79.8 / 74.3; DanceTrack 63.3 / 63.4 / 88.7; BDD100K mIDF1 60.0.
  - Uses ByteTrack YOLOX detections and a fast-reid BoT R50-ibn ReID model; pretrained models are available.
- NOOUGAT (Missaoui, Cetintas, Brasó, Meinhardt, Leal-Taixé; arXiv 2509.02111, Sep 2025): one GNN tracker for any temporal horizon. It processes non-overlapping subclips and fuses them with an Autoregressive Long-term Tracking layer — [NOOUGAT](https://arxiv.org/abs/2509.02111)
  - Online AssA gains: +2.3 DanceTrack, +9.2 SportsMOT, +5.0 MOT20.
  - Offline (HOTA / IDF1 / AssA): DanceTrack 68.4 / 72.7 / 58.7 vs SUSHI 63.3 / 63.4 / 50.1; SportsMOT 85.6 / 92.3 / 83.0; MOT20 66.1 / 83.0 / 70.7 vs SUSHI 64.3 / 79.8 / 67.5; MOT17 66.9 / 83.9 / 68.5 vs SUSHI 66.5 / 83.1 / 67.8.
  - Offline mode adds "an extra 3.8 AssA and 2.1 IDF1" on DanceTrack over its own online mode.

#### Short/unsupported tracklet removal and duplicate suppression
- SAM2BMP: key-frame-ratio filter (drop tracks whose detector-supported frames are < 10% of their length) +2.7 AP; tracklet NMS +0.4 AP — [SAM2BMP](https://iccvm.org/2026/files/papers/517.pdf)
- SAM 3: unconfirmed-masklet removal, duplicate-masklet removal and MDS-based suppression (definitions in Q2) — [SAM 3 paper](https://arxiv.org/abs/2511.16719)
- DetZero does not run refinement on "too short tracks"; their boxes are merged directly into the output — [DetZero](https://arxiv.org/abs/2306.06023)

#### Track-level class consistency (voting / rescoring)
- T-CNN (Kang et al., arXiv 1604.02532; ILSVRC2015 VID winner) — [T-CNN](https://arxiv.org/abs/1604.02532)
  - Multi-context suppression (MCS) exploits the fact that a video contains few classes. Classes in the top 0.0003 of all boxes in a video are "high-confidence"; the scores of every other class are reduced by 0.4.
  - Gains on ImageNet VID val: MCS +2.2–3.8% mAP, tubelet rescoring +0.4–1.2%, motion-guided propagation +0.8–2.5%; up to +6.7% combined.
- Seq-NMS whole-sequence rescoring: +6.5 mAP (44.9 → 51.4, VGG, ImageNet VID full val) — [Seq-NMS](https://arxiv.org/abs/1602.08465)

#### Libraries and licenses
- BoxMOT is AGPL-3.0, with Python and C++ trackers — [BoxMOT repo](https://github.com/mikel-brostrom/boxmot)
  - Trackers: OccluBoost, BoTSORT, BoostTrack, StrongSORT, DeepOCSORT, ByteTrack, HybridSORT, OCSORT, SFSort.
  - MOT17 ablation: OccluBoost HOTA 71.10 / MOTA 78.50 / IDF1 85.28 vs ByteTrack 67.68 / 78.04 / 79.16.
  - The README excerpt did not document GSI/AFLink post-processing.
- Ultralytics is AGPL-3.0 or a paid Enterprise license — [Ultralytics repo](https://github.com/ultralytics/ultralytics); [tracking docs](https://docs.ultralytics.com/modes/track/)
  - Six trackers: BoT-SORT, ByteTrack, OC-SORT, Deep OC-SORT, FastTracker and TrackTrack (the default).
  - `gmc_method` options: sparseOptFlow (default), orb, sift, ecc, none. `with_reid` enables appearance matching.
  - The docs mention no offline interpolation or post-processing.
- supervision (Roboflow, MIT) provides a ByteTrack integration and a `DetectionsSmoother` — [supervision repo](https://github.com/roboflow/supervision)
- ReID backbones:
  - torchreid / OSNet and OSNet-AIN (instance-normalization variant aimed at cross-domain generalization), MIT, ONNX/OpenVINO/TFLite export — [torchreid](https://github.com/KaiyangZhou/deep-person-reid)
  - FastReID, Apache-2.0, person/vehicle/cross-domain ReID with ONNX/TensorRT conversion; last changelog entry 2021 — [FastReID](https://github.com/JDAI-CV/fast-reid)

### Inferences
- The repair operators suggest a natural order: (1) split impure tracks (appearance clustering, GTA splitter); (2) link fragments (AFLink-style spatio-temporal gating plus appearance, GTA connector); (3) drop unsupported or duplicate tracks (key-frame ratio, MDS, tracklet NMS); (4) fill gaps (≤ ~20 frames by linear interpolation or GSI; longer or non-linear gaps by bidirectional SAM propagation between the bounding detections, as in Q3); (5) smooth (GSI); (6) confidence-weighted class vote per track (T-CNN/Seq-NMS evidence); (7) emit a per-track "repair log" (how many frames interpolated, merges, splits, class flips) as review-priority features. The ordering is an inference; each step's gain is measured separately in the sources above.
- License-wise, GSI is a small GP-regression routine and AFLink a tiny classifier. Reimplementing them avoids importing GPL-3.0 (StrongSORT) or AGPL-3.0 (BoxMOT) code into VIAT. GTA (MIT) can be used or adapted directly, with torchreid (MIT) or FastReID (Apache-2.0) embeddings.
- For VisDrone-like aerial data, appearance ReID is weak for small, similar vehicles and pedestrians. FOLT's authors even state "appearance-matching strategy is not important in UAV-MOT scenes" — [FOLT](https://arxiv.org/abs/2308.07207). So linking should rely mainly on motion after camera-motion compensation, with appearance as a secondary cue. GTA's gains were measured on sports players at much larger pixel sizes and may not transfer.

### Gaps
- No modern MOT paper isolates frame-to-track class majority voting as its own module with IDF1/HOTA or class-accuracy numbers; the evidence is from 2016-era video object detection (ImageNet VID).
- No evaluation was found of AFLink, GSI or GTA on VisDrone-MOT or UAVDT.
- NOOUGAT code availability was not verified.
- The effect of DINOv2/DINOv3 or SAM 3 features vs OSNet for offline stitching was not found quantified. A 2026 paper uses multi-stage SAM 3 feature fusion for zero-shot ReID in egocentric kitchens (arXiv 2605.26383), but it was not read.

## Q5. Sequence-level auto-labeling that exploits whole sequences (3DAL, Auto4D, DetZero, CTRL) and what transfers to 2D video

### Takeaway
Offboard, whole-sequence auto-labeling in LiDAR reached human-level quality. Waymo's 3DAL (2021) came within ~1 AP of human labelers at 3D AP@0.7 and beat them on BEV AP@0.8. DetZero and CTRL (2023) then exceeded human label consistency, e.g. DetZero 3D AP@0.7 90.24 vs human 86.45. They share three moves:
1. Prioritize track completeness (high-recall detection plus offline or bidirectional tracking).
2. Refine each object using all of its frames at once (object- or track-centric, with separate handling of static and moving objects).
3. Train students on 10% human plus 90% auto labels to get about 100%-human-label performance.

These ideas transfer to 2D video as keyframe-anchored bidirectional completion, per-track size/shape priors, trajectory smoothing, and static-object consolidation in stabilized coordinates.

### Cited Findings
- 3DAL (Qi et al., Waymo; arXiv 2103.05073, Mar 2021) — [3DAL](https://arxiv.org/abs/2103.05073)
  - Pipeline: an offboard multi-frame detector and tracker, then extraction of each object's full track data, motion-state classification, and a static-object model (one box per object) or a dynamic-object model (per-frame boxes using the whole track).
  - Human study (5 WOD val sequences, 3 experienced labelers): human 3D AP@0.7 / 0.8 = 86.45 / 60.49, BEV 93.86 / 86.27; 3DAL 85.37 / 56.93, BEV 92.80 / 87.55. Auto labels were "only around 1 point lower than the average labeler" at 0.7 and exceeded humans by 1.28 BEV AP@0.8; the gap at 3D AP@0.8 "is mostly caused by inaccurate heights".
  - The full pipeline adds +14.52 3D AP@0.8 over the single-frame detector and +8.39 over the multi-frame detector.
  - Semi-supervised (student MVF++): 10% human labels → 64.3 3D AP; 10% human + 90% 3DAL → 70.0; 100% human → 71.2. Cross-domain: human 59.4, self-annotation 60.3, 3DAL labels 64.2.
- Auto4D (Yang et al., Uber ATG/Waabi, arXiv 2101.06586, 2021) splits each 4D label into an object size fixed over time for rigid objects and a motion path. It iteratively refines online detections plus tracks, re-estimating size and smoothing the path using the entire trajectory. It reports "a 25% reduction of human annotation efforts" and shows annotator-in-the-loop use — [Auto4D](https://arxiv.org/abs/2101.06586)
- DetZero (Ma et al., arXiv 2306.06023, Jun 2023; code github.com/PJLab-ADG/DetZero) — [DetZero](https://arxiv.org/abs/2306.06023)
  - Diagnosis: onboard trackers produce incomplete trajectories, and motion state complicates object-centric refinement. Response: a high-recall multi-frame detector, an offline tracker with forward/reverse fusion, and attention-based refinement with decomposed regression.
  - Ranked 1st on the Waymo 3D detection leaderboard (85.15 mAPH L2) and 1st on Waymo 3D tracking (75.05 MOTA L2, +9.97).
  - Against the 3DAL human study: 3D AP@0.7 / 0.8 90.24 / 67.61 vs human 86.45 / 60.49; BEV 95.15 / 90.04. The authors say this is "the first time that the offboard 3D detector model can outperform the average human labels".
  - Semi-supervised: 10% human + 90% DetZero gives vehicle AP 74.44 vs 75.41 with 100% human; pedestrian AP 78.92 vs 77.51.
- CTRL (Fan et al., arXiv 2304.12315, Apr 2023; code in github.com/tusen-ai/SST) — [CTRL](https://arxiv.org/abs/2304.12315)
  - Its "track-centric" design mimics annotators, who label easy samples first and propagate them with temporal cues.
  - The bidirectional tracking module is followed by multiple-in-multiple-out refinement: the whole track's cropped points are refined jointly in one forward pass. It reports surpassing human-level annotation accuracy on Waymo without model ensembles. Missed-object reductions are in Q3.

### Inferences
- In 2D, the analogs are:
  - Completeness first: detect at a low threshold, track offline and bidirectionally, then filter by track-level evidence such as key-frame ratio or MDS rather than per-frame confidence.
  - Whole-track refinement: compute per-track robust statistics from confident frames (median width/height/aspect for rigid objects in near-nadir drone views, where scale changes slowly) and use them to regularize or flag per-frame boxes, then smooth centers (GSI-style).
  - Static-object consolidation: after camera-motion compensation (Q6), parked cars in VisDrone can get one stabilized box re-projected per frame. This mirrors 3DAL's static branch.
- The 10% human / 90% auto-label result suggests VIAT's triage target could be "a small human-verified subset plus auto labels", but that is the data-engine researcher's domain.

### Gaps
- No 2D-video study was found that measures auto-label quality against human re-labeling consistency with the rigor of 3DAL's human study; 2D evidence stays benchmark-metric-based.
- The 3D refinement networks (3DAL, DetZero, CTRL) are trained on large labeled LiDAR sets. No training-free 2D equivalent with measured gains was found; the heuristic analogs above are untested.

## Q6. Preconditions for unattended runs: shot/scene-cut detection, camera-motion compensation, and small-object / dense-scene issues in aerial MOT

### Takeaway
Shot-boundary detection is mature, cheap and permissively licensed, so it should run first and hard-split tracks at cuts:
- TransNet V2 (MIT): F1 96.2 on BBC, 93.9 on RAI, 77.9 on ClipShots.
- PySceneDetect (BSD-3): AdaptiveDetector F1 91.6 on BBC, but only ~74 on short-form AutoShot and ~56 on ClipShots hard cuts.

Camera-motion compensation gives modest but consistent association gains on pedestrian benchmarks (+1.6 IDF1 in BoT-SORT's ablation) and matters more on UAV footage, where objects are smaller and accelerate more. GMC can fail when few static background keypoints exist. On UAVDT, a SAM2-based MOT beat ByteTrack with the same detector (IDF1 74.4 vs 67.8; IDSW 141 vs 452). Dense scenes, however, stress SAM-style memory and runtime.

### Cited Findings
- TransNet V2 (Souček & Lokoč, arXiv 2008.04838, 2020): F1 ClipShots 77.9, BBC 96.2, RAI 93.9, vs TransNet 73.5 / 92.9 / 94.3 and Hassanien et al. 75.9 / 92.6 / 93.9 — [TransNet V2 paper](https://arxiv.org/abs/2008.04838). MIT license; TensorFlow with a PyTorch inference port; pretrained weights, no training needed — [TransNetV2 repo](https://github.com/soCzech/TransNetV2)
- AutoShot (arXiv 2304.06116, 2023): an architecture-searched shot-boundary model on a new short-video SHOT dataset. It beats TransNetV2 by 4.2% F1 on SHOT and by 3.5% precision at TransNetV2's recall, and also exceeds it on ClipShots, BBC and RAI — [AutoShot](https://arxiv.org/abs/2304.06116)
- PySceneDetect (BSD-3-Clause; v0.7.1, 21 Jul 2026): Content, Adaptive, Threshold, Hash and Histogram detectors, with a Python API — [PySceneDetect repo](https://github.com/Breakthrough/PySceneDetect)
  - Official benchmark F1: BBC Planet Earth hard cuts — Adaptive 91.59, Content 86.69, Hash 83.10, Histogram 79.96. AutoShot hard cuts — Adaptive 73.86, Content 69.26. ClipShots hard cuts — Content 55.84, Adaptive 55.75. ClipShots fades — Histogram 75.33 — [PySceneDetect benchmarks](https://www.scenedetect.com/benchmarks/)
- BoT-SORT (Aharon et al., arXiv 2206.14651, 2022) estimates global motion from image keypoints tracked with sparse optical flow, then an affine fit, and corrects the Kalman state with it — [BoT-SORT](https://arxiv.org/abs/2206.14651)
  - MOT17 val ablation (MOTA / IDF1 / HOTA): ByteTrack* 77.66 / 79.77 / 67.88; +updated KF 77.67 / 79.89 / 68.12; +CMC 78.31 / 81.51 / 69.06 (CMC alone +1.62 IDF1, +0.94 HOTA); +ReID 78.46 / 82.07 / 69.17.
  - Stated limitations: "In scenes with a high density of dynamic objects, the estimation of the camera motion may fail due to lack of background keypoints", and GMC can be slow on large images (it can be multi-threaded).
- StrongSORT's ECC-based camera motion compensation gave only +0.2 IDF1 and +0.3 MOTA on MOT17 val — [StrongSORT](https://arxiv.org/abs/2202.13514)
- UCMCTrack (arXiv 2312.08952; code github.com/corfyi/UCMCTrack) uses motion only. It applies one set of compensation parameters for the whole sequence instead of per-frame CMC, runs a ground-plane Kalman filter, and replaces IoU with a Mapped Mahalanobis Distance. UCMCTrack+ reaches MOT17 test HOTA 65.8 — [UCMCTrack](https://arxiv.org/abs/2312.08952)
- FOLT (ACM MM 2023) uses lightweight optical flow for flow-guided feature augmentation and motion prediction — [FOLT](https://arxiv.org/abs/2308.07207)
  - VisDrone test: ByteTrack MOTA 35.7 / IDF1 37.0 / IDs 2,168 at 27.0 FPS; OC-SORT 39.6 / 50.4 / 986; FOLT 42.1 / 56.9 / 800 at 29.4 FPS.
  - UAVDT: ByteTrack 41.6 / 59.1 vs FOLT 48.5 / 68.3.
  - VisDrone objects are smaller and move faster (higher size-normalized acceleration) than in MOT17/20.
- AMOT (arXiv 2508.01730, Aug 2025) combines an appearance-motion consistency matrix with motion-aware track continuation that reactivates unmatched tracks. VisDrone2019: IDF1 61.4, MOTA 46.0 at 36.4 FPS — [AMOT](https://arxiv.org/abs/2508.01730)
- SAM2MOT on UAVDT (Co-DINO-L detector, no fine-tuning), MOTA / IDF1 / IDSW at the IoU 0.5 standard — [SAM2MOT](https://arxiv.org/abs/2504.04519)
  - SAM2MOT 55.6 / 74.4 / 141 vs ByteTrack 52.4 / 67.8 / 452 and OC-SORT 48.5 / 64.5 / 444 with the same detector; published FOLT 48.5 / 68.3; DroneMOT 50.1 / 69.6.
  - "Considering annotation noise in the UAVDT ground truth", the authors also report IoU 0.4 (66.1 MOTA / 79.3 IDF1).
- Dense scenes and SAM-style trackers: SAM 3 ran out of memory on an H100 (96 GB) with CPU offload on some dense animal sequences — [Zero-shot multi-animal tracking](https://arxiv.org/abs/2511.02591). LiAM-SAM drops from 5 FPS at 4 objects to 2 FPS at 30 objects on an A100 — [LiAM-SAM](https://arxiv.org/abs/2609.28078). SAM 3.1 multiplexing targets exactly this (~7× faster at 128 objects on an H100) — [SAM 3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)

### Inferences
- A pre-flight for VIAT unattended runs:
  1. Run shot detection (TransNet V2 for edited or compiled footage; PySceneDetect AdaptiveDetector as a CPU fallback) and forbid any track or propagation from crossing a cut.
  2. Estimate a per-frame global homography or affine transform (sparse optical flow or ECC, as in Ultralytics' `gmc_method`) and store it with an inlier count. Low-inlier frames are "GMC-unreliable" and should raise the risk score for motion-based flags.
  3. Run all velocity and acceleration outlier tests and Kalman gating in stabilized coordinates.
- For VisDrone-style scenes (many tiny objects, drone ego-motion), a lower-cost default is detector-driven box MOT with CMC (BoT-SORT/ByteTrack/OC-SORT-class trackers) plus the offline repairs in Q4. SAM-based mask propagation would be invoked selectively for flagged, gappy or mask-required objects. SAM2MOT's UAVDT result shows SAM-based association can clearly beat ByteTrack in aerial footage when affordable, so a GPU-budget switch is warranted.
- UAVDT ground-truth noise (SAM2MOT's reason for also reporting IoU 0.4) implies that small-object box precision is inherently limited. Review triage should weight gross errors (misses, ID switches, class errors) above sub-pixel box jitter for tiny objects.

### Gaps
- No verified throughput for TransNet V2 was found, and no shot-detector evaluation on drone or aerial footage.
- No study quantifies how often GMC fails on VisDrone/UAVDT, or how much CMC improves SAM-based propagation on drone video.
- A 2025–2026 journal report (YOLO-Ro-KCF, PMC) claims VisDrone-2019 MOTA 70.6 / HOTA 72.1 / IDF1 73.5 at 43 FPS on a Jetson AGX Xavier — [YOLO-Ro-KCF](https://pmc.ncbi.nlm.nih.gov/articles/PMC13493796/). These figures are far above other recent reports (AMOT IDF1 61.4 / MOTA 46.0), probably due to a different protocol or subset. They are unverified and should not be used as a reference point.
- MOSAIC-Tracker (ISPRS 2025) and STCMOT (arXiv 2409.11234) claim VisDrone gains, but their numbers were not retrieved.
