# Data engines and self-training: auto-label → verify → fine-tune → re-label loops (SAM-family engines, industrial auto-labeling, foundation→specialist distillation, pseudo-label self-training for detection and video)

*Research date: 2026-09-30. Where a line says "(GitHub API)", the license (SPDX id) and repo-creation date were read from api.github.com on 2026-09-30. "Classic" marks pre-2024 work that is included as background. **API-only / paid** and restrictive licenses (SAM License, AGPL, GPL, non-commercial) are flagged inline.*

---

## 1. SAM (2023), SAM 2 (2024) and SAM 3 (2025–26) data engines: phases, what was automated, time per mask/masklet, speed-ups, AI annotators/verifiers, and what this means for a small team with one workstation

### Takeaway
All three Meta engines climb the same steps. First, humans annotate with the model in the loop. Next, the model pre-fills and humans only add what is missing or fix what is wrong. Finally, the model labels on its own and automated filters or verifiers, plus targeted human checks, gate quality. The model is retrained many times along the way: SAM 6+5 times, SAM 2 2+5 times, SAM 3 6+7 times plus 3 verifier retrains. Human time fell from 34 s to 14 s per mask (SAM) and from 37.8 s to 4.5 s per video frame (SAM 2, ~8.4×). SAM 3 handed most human *verification* to fine-tuned Llama 3.2 verifiers, which roughly doubled throughput: about 5× faster on negative prompts and 36% faster on positives. People were then focused on failure cases such as crowded scenes and tracking failures. No engine ran with zero humans: SA-V's automatic masklets and SAM 3's video phase both kept a human verification gate.

### Cited Findings

#### SAM (Segment Anything, Meta, arXiv 2304.02643, April 2023; classic)
- **Stage 1, "assisted-manual":** professional annotators clicked foreground/background points in a browser tool powered by SAM and refined masks with brush and eraser tools. They labeled objects "in order of prominence" and were told to move on after 30 s per mask. The model was retrained **6 times** and the image encoder scaled from ViT-B to ViT-H. Output: **4.3M masks from 120k images**. Average time per mask fell **from 34 s to 14 s**, and masks per image rose from 20 to 44 — [SAM paper](https://arxiv.org/html/2304.02643)
- 14 s/mask "is 6.5× faster than mask annotation for COCO and only 2× slower than bounding-box labeling with extreme points" — [SAM paper](https://arxiv.org/html/2304.02643)
- **Stage 2, "semi-automatic":** a box detector trained on stage-1 masks detected confident masks automatically. Annotators received pre-filled images and labeled only the remaining unannotated objects, to increase diversity. The model was retrained **5 times**. Output: **5.9M masks in 180k images** (10.2M total). Time per mask rose back to **34 s** (excluding the automatic masks) because the remaining objects were harder. Masks per image went from 44 to 72 — [SAM paper](https://arxiv.org/html/2304.02643)
- **Stage 3, "fully automatic":** the model was prompted with a **32×32 point grid**, producing several candidate masks per point. It kept only confident masks (via IoU prediction) and *stable* masks, where "thresholding the probability map at 0.5−δ and 0.5+δ results in similar masks". NMS removed duplicates, and overlapping zoomed-in crops were processed to catch small masks. Output: **1.1B masks on 11M images**, 99.1% of them generated fully automatically — [SAM paper](https://arxiv.org/html/2304.02643)
- **QA:** 500 images (~50k masks) were professionally corrected. Of the auto vs. corrected pairs, **94% had IoU > 90%** and 97% had IoU > 75%. Prior estimates of inter-annotator consistency are 85–91% IoU — [SAM paper](https://arxiv.org/html/2304.02643)
- **Ablation:** training on automatic masks only was about **0.5 mIoU** below training on all three stages, so SAM's default training uses only the automatic masks. 1M images (~10% of SA-1B) gave results comparable to the full set, while 0.1M images gave a large drop — [SAM paper](https://arxiv.org/html/2304.02643)
- License: the model is Apache 2.0, and SA-1B was released under a research-use agreement — [SAM paper](https://arxiv.org/html/2304.02643). The repo is Apache-2.0 (GitHub API) — [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

#### SAM 2 (Meta, arXiv 2408.00714; repo created 2024-07-29)

| Phase | Tool in the loop | Time / frame | Speed-up vs P1 | Masklets | % frames edited | Clicks per clicked frame | Alignment with Phase-1 masks* | Model updates |
|---|---|---|---|---|---|---|---|---|
| 1 | SAM per frame + pixel tools | 37.8 s | 1× | 16K (1.4K videos) | 100% | 4.80 | — | — |
| 2 | SAM + "SAM 2 Mask" (mask-only prompts, temporal propagation) | 7.4 s | ~5.1× | 63.5K | 23.25% | 3.61 | 86.4% | retrained twice |
| 3 | Full SAM 2 (points + masks, memory) | 4.5 s | ~8.4× | 197.0K | 19.04% | 2.68 | 89.1% | updated 5 times |

*Source for all table values:* [SAM 2 paper](https://arxiv.org/html/2408.00714v2). (*The alignment column is the paper's Table 1 measure of agreement with Phase-1-style manual masks.)
- **Quality gate:** "A separate set of annotators are tasked with verifying the quality of each annotated masklet as 'satisfactory' or 'unsatisfactory'." Unsatisfactory masklets went back for refinement, and poorly defined objects were rejected — [SAM 2 paper](https://arxiv.org/html/2408.00714v2)
- **Auto masklets:** SAM 2 was prompted with a grid, and the results went through the **same human verification**. Accepted auto masklets added diversity. Rejected ones were given to annotators as model-failure cases to refine — [SAM 2 paper](https://arxiv.org/html/2408.00714v2)
- **Final SA-V:** 50.9K videos (average 14 s), **642.6K masklets (190.9K manual + 451.7K automatic)**, 35.5M masks, 4.2M frames / 196 h, from 47 countries — [SAM 2 paper](https://arxiv.org/html/2408.00714v2)
- **Gain per data-engine increment** (J&F, 3-click). SA-V val: 50.0 (VOS + SA-1B only) → 53.0 (+Phase 1) → 58.8 (+Phase 2) → 62.5 (+Phase 3) → 63.2 (+Auto). Average over 9 zero-shot video datasets: 62.5 → 66.9 → 70.9 → 71.2 → 71.5 — [SAM 2 paper](https://arxiv.org/html/2408.00714v2)
- Licenses: training and demo code Apache 2.0, SA-V CC BY 4.0 — [SAM 2 paper](https://arxiv.org/html/2408.00714v2). The repo is Apache-2.0 (GitHub API) — [facebookresearch/sam2](https://github.com/facebookresearch/sam2)

#### SAM 3 (Meta, arXiv 2511.16719, Nov 2025; v2 dated 28 Mar 2026) and SAM 3.1 (27 Mar 2026)
- **Phase 1, human verification:** SAM 2 plus an open-vocabulary detector proposed masks for noun phrases (NPs). Humans performed two verification tasks: **mask verification (MV)** and **exhaustivity verification (EV)**. Output: 4.3M image–NP pairs — [SAM 3 paper](https://arxiv.org/html/2511.16719)
- **Phase 2, human + AI verification:** Llama 3.2 models were fine-tuned on the **Phase-1 human accept/reject labels** for MV and EV. This "roughly doubles the data engine's throughput vs. human annotators". NP proposal was upgraded to a Llama-based pipeline that also proposes **hard-negative NPs adversarial to SAM 3**. SAM 3 was retrained **6 times**. Output: +122M image–NP pairs — [SAM 3 paper](https://arxiv.org/html/2511.16719)
- **Phase 3, scaling and domain expansion:** 15 datasets/domains and a 22.4M-node ontology built from Wikidata. SAM 3 was iterated **7 times** and the AI verifiers **3 times**. EV used domain-specific human supervision. Output: +19.5M image–NP pairs — [SAM 3 paper](https://arxiv.org/html/2511.16719)
- **Phase 4, video:** SA-Co/VIDEO has 52.5K videos and 467K masklets. Humans were concentrated "on likely failures by favoring clips with many crowded objects and tracking failures" — [SAM 3 paper](https://arxiv.org/html/2511.16719)
- **Meta blog on the engine:** the AI annotators were based on Llama 3.2v and "specifically trained to match or surpass human accuracy". The hybrid pipeline "more than double[d]" throughput vs. a human-only pipeline. It was **~5× faster than humans on negative prompts** (concept absent) and **36% faster on positive prompts** "even in challenging fine-grained domains". The dataset has over 4M unique concepts — [Meta AI blog](https://ai.meta.com/blog/segment-anything-model-3/)
- **Datasets:**
  - SA-Co/HQ: 5.2M images, 4M unique NPs, 52M masks (phases 1–3).
  - SA-Co/SYN: 38M phrases and 1.4B masks, produced by the mature engine "without human involvement".
  - SA-Co/EXT: 15 external datasets enriched with hard negatives.
  - SA-Co/VIDEO: 52.5K videos, 24.8K unique NPs, 134K video–NP pairs, averaging 84.1 frames at 6 fps.
  - Source: [SAM 3 paper](https://arxiv.org/html/2511.16719)
- **Ablations:**
  - Training-data mix (cgF1): EXT only 23.7 → EXT+SYN 32.8 → EXT+HQ 45.5 → all three 47.4.
  - Hard negatives, 0 vs 30 per image: cgF1 28.3 → 43.0, IL_MCC 0.44 → 0.68.
  - AI verifiers applied to SAM 3 outputs: cgF1 54.0 → 61.2 (EV verifier replaces SAM 3's presence score, +7.2) → 62.3 (+MV, +1.1).
  - Source: [SAM 3 paper](https://arxiv.org/html/2511.16719). "AI verifiers close half of the gap" between SAM 3 and human performance (72.8 cgF1 on SA-Co/Gold), and "the MV AI verifier performs well zero-shot" on new domains — [SAM 3 paper v2](https://arxiv.org/html/2511.16719v2)
- Synthetic data generated by "SAM 3 + AI verifiers" showed "similar scaling behavior as human-annotated (HQ) data" on a held-out domain (Fig. 9) — [SAM 3 paper](https://arxiv.org/html/2511.16719)
- **Speed:** "30 milliseconds for a single image with more than 100 detected objects on an H200". Video latency scales with the number of objects, staying near real-time for about five concurrent objects — [Meta AI blog](https://ai.meta.com/blog/segment-anything-model-3/)
- **SAM 3.1 (27 Mar 2026):** a drop-in checkpoint that adds "Object Multiplex", a shared-memory approach to joint multi-object tracking, handling up to **16 objects per forward pass**. It roughly doubles speed for videos with a medium number of objects, **from 16 to 32 fps on one H100** — [Meta AI blog](https://ai.meta.com/blog/segment-anything-model-3/); [sam3 repo](https://github.com/facebookresearch/sam3)
  - A secondary analysis says accuracy is flat and speed is 7.2× faster with 128 objects — [note.com analysis](https://note.com/ai_driven/n/n1784eaf09356?hl=en) (secondary; unverified).
  - A **paid SAM 3.1 API** at $2.5 per 1,000 images and $0.2 per 1,000 video frames was reported by a news aggregator — [KuCoin news](https://www.kucoin.com/news/flash/meta-launches-sam-3-1-api-for-image-and-video-segmentation) (secondary). The weights remain downloadable per the repo.
- **License and access:**
  - SAM 3 uses the custom **"SAM License"** (GitHub API: NOASSERTION). It permits commercial use and derivative works, but prohibits **military/warfare**, nuclear, espionage and ITAR-regulated uses, and requires trade-control compliance and attribution — [SAM 3 LICENSE](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
  - Checkpoints are **gated on Hugging Face** and need an access request. The code requires Python 3.12+, PyTorch 2.7+ and CUDA 12.6+ — [sam3 repo](https://github.com/facebookresearch/sam3)
- **Fine-tuning code:**
  - Uses Hydra configs. **Single-GPU is supported** (`--num-gpus 1`), as is multi-node with up to 8 GPUs per node.
  - Includes examples for full fine-tuning on Roboflow100-VL with 100 images, and for ODinW13 10-shot runs with 3 seeds.
  - Caveat from the README: "a small number of jobs may diverge during training, in which case we just use the last checkpoint's result before it diverges".
  - Source: [SAM 3 README_TRAIN](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md)

### Inferences
- **The human role moves from drawing to handling rejects.** Across the engines it goes: draw → correct → add missing → verify → handle rejects only. VIAT already covers the first two (SAM-assisted drawing; auto-annotate then fix manually). Three pieces are missing:
  - a *verification-first* review mode: accept/reject per object or per track instead of redrawing;
  - automatic quality filters that run before any human sees a label, such as SAM's stability, IoU-prediction and NMS filters and SAM 3's MV/EV verifiers;
  - **logging every human accept/reject decision** so a verifier can later be trained on those logs. SAM 3's AI verifiers were built exactly this way, from Phase-1 human decisions.
- **Retrain often.** Meta retrained after each chunk of data: 11 SAM retrains across two stages, and 13 SAM 3 retrains plus 3 verifier retrains. For VIAT this means fine-tuning after every batch of verified corrections (for example per video, or per few hundred corrected frames), not once at the end.
- **Expect diminishing returns.** Gains per increment shrink: SAM 2 gained +3.0, +5.8, +3.7 and +0.7 J&F on SA-V val. Automatic-only data matched human-heavy data only once the model was already strong in-domain (SAM: −0.5 mIoU after ~10M human-verified masks). For a new domain like drone footage, expect the first 1–2 verified rounds to matter most.
- **Automate "absent" decisions first.** AI verification helped most on *negatives* (5× faster vs. +36% on positives), and hard negatives drove large gains (IL_MCC 0.44 → 0.68). VIAT should automate "not present / false positive" decisions first and store rejected detections (shadows, clutter, look-alikes) as explicit hard negatives. Humans stay on positive boundary and identity decisions.
- **"Unattended" still kept a human gate.** At Meta that gate was a sampled or flagged subset: a 500-image professional audit (SAM), every masklet verified (SAM 2), and humans on crowded or tracking-failure clips (SAM 3 video). A realistic VIAT target is human attention proportional to the flagged-failure rate plus a small random audit, not zero.
- **Licensing:** SAM 3/3.1's license bans military uses, which matters for drone and aerial users in defense work. SAM and SAM 2 (Apache-2.0) carry no such restriction.

### Gaps
- Per-task human times for SAM 3 (seconds per MV/EV decision) and per-task accuracy of the AI verifiers vs. humans were not found. Only throughput multipliers and cgF1 deltas are published in the sources read.
- The total human-hours spent by any of the three engines is not published in the material found.
- SAM 3.1's accuracy vs. SAM 3 and the "7.2× at 128 objects" figure come only from a secondary blog. SAM 3.1 API pricing comes only from a news aggregator.

---

## 2. Industrial practice: Tesla auto-labeling, Waymo offboard auto-labeling, and vendor statistics on model-assisted labeling

### Takeaway
The industrial auto-labeling that removed the most human effort (Tesla, Waymo) runs **offboard**. Large offline models see whole clips: past and future frames, multiple cameras, and even multiple trips over the same place. They produce labels that match or beat human labelers, and humans move to verification and cleanup. Published numbers come from keynotes and press:
- Tesla: 10k clips auto-labeled in a week vs. "several months" manually (2021).
- Tesla (secondary recap): "5 million hours of manual labeling replaced by 12 hours on the cluster for 10k trips" for lanes (2022).
- Waymo: auto labels on par with humans in a human-label study.

Vendor statistics on model-assisted labeling are marketing-grade, typically "up to 50%" time savings.

### Cited Findings
**Tesla**
- **AI Day 2021:**
  - Tesla built a 1,000-person in-house labeling team.
  - Labeling moved from per-image 2D labels to labels in "4D vector space" reconstructions from 8 cameras over time: "each unitary labeling effort is now worth 100x what was previously".
  - Source: [TDS AI Day 2021 review](https://towardsdatascience.com/tesla-ai-day-2021-review-part-2-training-data-how-does-a-car-learn-e8863ba3f5b0/)
- **Offline auto-labeling pipeline:**
  - Clips carry video, IMU, GPS and odometry.
  - Offline neural networks extract depth, segmentation and point matches, with **access to future frames** ("cheat and look ahead" to resolve occlusions and kinematics).
  - Labels are then generated algorithmically from those features.
  - Source: [TDS AI Day 2021 review](https://towardsdatascience.com/tesla-ai-day-2021-review-part-2-training-data-how-does-a-car-learn-e8863ba3f5b0/)
- **Radar-removal project:** 10,000 clips were auto-labeled in about a week, work that Ashok Elluswamy said would have taken several months with humans — [WRAL archive of CNN article](https://www.wral.com/archive/20353852/); [TDS review](https://towardsdatascience.com/tesla-ai-day-2021-review-part-2-training-data-how-does-a-car-learn-e8863ba3f5b0/)
- Musk: "Without the auto labeling I think we would not be able to solve the self-driving problem" — [WRAL/CNN](https://www.wral.com/archive/20353852/)
- **June 2022 layoffs:** Tesla laid off "a significant number" of Autopilot data-annotation staff as part of a 10% salaried cut; labelers worked in San Mateo, CA and Buffalo, NY. The in-house team had been "more than 1,000 people". Tesla kept posting annotation jobs, and humans may still verify auto-labels — [WRAL/CNN](https://www.wral.com/archive/20353852/) (the original CNN URL returned HTTP 451 to the fetcher)
- **AI Day 2022:** an "auto labeling machine" based on **multi-trip reconstruction** for lanes; "5 million hours of manual labeling replaced by 12 hours on the cluster for 10k trips" — [nocode.ai recap](https://www.nocode.ai/recap-tesla-ai-day-2022/) (secondary recap, not a transcript)

**Waymo**
- **"Offboard 3D Object Detection from Point Cloud Sequences"** (Qi, Zhou, Najibi, Sun, Vo, Deng, Anguelov; CVPR 2021; arXiv 2103.05073):
  - The "3D Auto Labeling" pipeline uses temporal point clouds through multi-frame detection plus object-centric refinement.
  - It shows significant gains over state-of-the-art onboard detectors, with auto labels "on par with human labels verified through a human label study".
  - The auto labels were also used for semi-supervised learning.
  - Source: [arXiv](https://arxiv.org/abs/2103.05073)
  - The human label study found auto labels "on par or even slightly better" than human labels on the selected test segments — [paper PDF](https://arxiv.org/pdf/2103.05073)

**Vendors and studies of model-assisted labeling**
- **Labelbox claims** that model-assisted labeling (pre-labels) reduces "labeling time and costs by up to 50%". It cites a customer with a 160% gain in labeling efficiency, and a 2018 UC Berkeley / Georgia Tech / Peking U / Uber AI experiment in which "when the model was completely accurate 40% of the time, the average time for labeling was reduced by 60%" — [Labelbox blog](https://labelbox.com/blog/use-automation-to-reduce-your-labeling-time-and-spend/). These are vendor claims seen via search snippet; the page itself refused connection.
- **Voxel51 "Verified Auto Labeling"** claims "up to 95% of human-level performance" at "up to 100,000×" lower labeling cost, doable "in just a few hours on a single GPU" — [Voxel51 blog](https://voxel51.com/blog/zero-shot-auto-labeling-rivals-human-performance) (vendor). The underlying paper's numbers are in §3.
- **Desmond et al. (IBM; ACM IUI 2021, DOI 10.1145/3397481.3450698):** a 54-participant study found that AI label suggestions (semi-supervised predictions plus a reduced label space) improved both labeler accuracy and speed — [arXiv 2104.04122](https://arxiv.org/abs/2104.04122). This was category labeling, not boxes; no numbers were available in the abstract.
- **Classic: Papadopoulos, Uijlings, Keller, Ferrari (CVPR 2016), "We don't need no bounding-boxes":**
  - Annotators only answer yes/no on detector-proposed boxes, inside a loop of retrain → relocalize → verify.
  - Total annotation time was cut **6–9×** vs. drawing boxes.
  - Detectors trained this way were "almost as good" as fully supervised ones on PASCAL VOC 2007.
  - Source: [arXiv 1602.08405](https://arxiv.org/abs/1602.08405)

### Inferences
- **Use look-ahead.** The most transferable idea for VIAT is offboard processing. VIAT works offline, so the label for frame *t* can use frames after *t*: bidirectional propagation or tracking and track-level smoothing. That is the same "look ahead" advantage Tesla and Waymo exploit. This overlaps with the tracker / offline track-repair researcher's scope.
- **Verify once per track.** Tesla's "label once in vector space, reuse across frames and cameras" maps to verifying once per *track*. The review unit should be a track or tubelet, with one human decision covering many frames, rather than per-frame boxes. The classic verification-only result (6–9× cheaper than drawing) supports yes/no review as the default human action.
- **Humans stay as verifiers.** Both companies kept humans in a verification and cleanup role. The practical end state is humans auditing a sample and handling flagged failures, which matches the Meta engines.

### Gaps
- No reliable, first-party Scale AI statistics on model-assisted or auto-labeling speed-ups were found; only third-party reviews surfaced.
- Waymo's exact 3D AP numbers for auto labels vs. human labelers were not extracted (the abstract gives only qualitative "on par").
- The Tesla AI Day 2022 "5M hours → 12 h" figure comes from a secondary recap, not a primary transcript. The 2021 figures come via press and a TDS review.

---

## 3. Foundation model → specialist distillation, and how few human-verified images are needed to adapt open-vocabulary models

### Takeaway
Training a small detector purely on **zero-shot auto-labels** reaches ~93–95% of the mAP50 of the same detector trained on human labels for common-object datasets (VOC, COCO). It reaches only ~62–68% on domain-shifted or long-tail data (BDD100K driving, LVIS), which is the regime drone footage falls into. For a student, a low auto-label confidence threshold (0.2) works best. A handful of verified examples changes the picture:
- **10-shot fine-tuning roughly doubles open-vocabulary accuracy** on out-of-distribution data: Grounding DINO 15.7 → 33.6 mAP and SAM 3 15.2 → 36.5 AP on RF100-VL.
- Fully supervised specialists reach ~55–62 mAP on the same benchmark.

The practical recipe is therefore: auto-label → verify a small class-balanced set → fine-tune the labeler → re-label → distill into a fast specialist. The specialist can be Apache-2.0 (RF-DETR, D-FINE, RT-DETR) or AGPL-3.0 (YOLO26).

### Cited Findings

#### Autodistill and Roboflow's "foundation model → small model" workflows
- **Autodistill** (Roboflow; repo created 2023-06-06; core Apache-2.0, per-plugin licenses vary) "uses big, slower foundation models to train small, faster supervised models" — [autodistill repo](https://github.com/autodistill/autodistill)
  - Concepts: **base model**, **ontology** (mapping prompts to classes), auto-labeled **dataset**, **target model**, and **distilled model**.
  - The README's detection base models include Grounded SAM 2, GroundedSAM, GroundingDINO, DETIC, OWL-ViT, OWLv2, SAM-CLIP, LLaVA-1.5, Kosmos-2 and CoDet, plus cloud APIs (Azure Custom Vision, AWS Rekognition, Google Vision; **paid APIs**).
  - Target models are YOLOv8, YOLO-NAS, YOLOv5 and DETR.
  - It publishes no quantitative results, and "human in the loop support" is only on the roadmap.
- **Roboflow's documented pattern:** run SAM 3 over the images, review and correct in Roboflow Annotate, then train RF-DETR on the result; "Autodistill automates this process" — [Roboflow blog](https://blog.roboflow.com/build-a-promptable-object-detection-project/) (via search snippet)
- **Roboflow Rapid** (launched 9 Dec 2025): a text prompt plus "a handful of images or a short video" leads to automatic annotation, background RF-DETR training and deployment. It is a cloud service billed in credits and publishes no accuracy numbers — [Roboflow blog](https://blog.roboflow.com/roboflow-rapid/) (**cloud/paid**)
- **Roboflow-hosted SAM 3 fine-tuning** (guide dated 15 May 2026): runs in Roboflow's cloud, billed in credits (the free tier is included). Data is resized to 1008×1008, and the guide recommends no augmentation on the first run. It gives no before/after metrics — [Roboflow blog](https://blog.roboflow.com/fine-tune-sam3/) (**cloud/paid**; local fine-tuning uses the open sam3 code instead, §1)

#### Controlled study: detectors trained on zero-shot auto-labels vs. human labels (Voxel51 — Griffin, Gangwar, Sela, Corso; arXiv 2506.02359, June 2025)
- **Setup:**
  - Auto-labelers: **YOLO-World** (72.9M params; **197.2 s** to label VOC), **YOLOE** (35.2M; **204.9 s**) and **Grounding DINO-T** (172.2M; **2,290.3 s**).
  - Datasets: VOC, COCO, LVIS and BDD100K.
  - Students: YOLO11 n–x and RT-DETR.
  - Source: [paper](https://arxiv.org/html/2506.02359)
- **YOLO11n mAP50, human labels vs. auto-labels:** VOC 0.756 vs 0.715; COCO 0.496 vs 0.460; LVIS 0.087 vs 0.059; **BDD 0.434 vs 0.271** — [paper](https://arxiv.org/html/2506.02359). From these raw numbers (my computation), auto-labels reach ≈94.6% (VOC), 92.7% (COCO), 67.8% (LVIS) and 62.4% (BDD) of human-label mAP50. The relative-gap percentages returned by the fetch tool were inconsistent, so I recomputed them.
- **Threshold:** an auto-label confidence threshold of **α = 0.2 gave the best downstream models** on every dataset. α = 0.8 was worst despite having the highest precision. The single recommended configuration is "YOLOW-0.2". Switching from YOLO11n trained on human labels to YOLO11s trained on auto-labels "increases VOC mAP 2–5%" — [paper](https://arxiv.org/html/2506.02359)
- **Cost and time across the four datasets:** humans took **6,703 h and $124,092.54**; auto-labeling took **1.27 h and $1.18** (5,279× faster, 105,105× cheaper). VOC alone: 78 h and $1,442.09 vs. 0.06 h and $0.05 — [paper](https://arxiv.org/html/2506.02359)
- **Failure modes:**
  - rare classes ("steak knife", "car battery", "eye dropper");
  - "small or partially occluded objects in heavy traffic";
  - domain shift (the driving viewpoint gave the largest gap);
  - long vocabularies (LVIS) causing Grounding DINO memory limits and ≈300× longer labeling.
  - The authors say human labeling remains preferable for specialized domains such as BDD and LVIS.
  - Source: [paper](https://arxiv.org/html/2506.02359)

#### How few verified labels adapt an open-vocabulary model (ODinW-13, RF100-VL)
- **RF100-VL** (Robicheaux, Popov, Madan, Robinson, Nelson, Ramanan, Peri; NeurIPS 2025 Datasets & Benchmarks; arXiv 2505.20612; 100 datasets including aerial and medical; code Apache-2.0 per GitHub API; data CC BY 4.0). Source for all numbers: [RF100-VL paper](https://arxiv.org/html/2505.20612); [abs](https://arxiv.org/abs/2505.20612); [repo](https://github.com/roboflow/rf100-vl)

  | Setting | Model | mAP on RF100-VL |
  |---|---|---|
  | Zero-shot | Grounding DINO | 15.7 |
  | Zero-shot | OWLv2 | 13.6 |
  | Zero-shot | MQ-GLIP-Text | 12.2 |
  | Zero-shot | Gemini 2.5 Pro | 11.6 |
  | Zero-shot | Detic | 9.5 |
  | Zero-shot | Qwen2.5-VL-72B | 5.6 |
  | 10-shot (10 instances per class) | **Grounding DINO** | **33.6** |
  | 10-shot | Detic w/ federated loss | 22.8 |
  | 10-shot | YOLOv8s | 20.7 |
  | 10-shot | YOLOv8m | 20.3 |
  | Fully supervised | YOLOv8n | 55.4 |
  | Fully supervised | YOLO11n | 56.1 |
  | Fully supervised | YOLOv8s | 56.5 |
  | Fully supervised | YOLO11s | 57.0 |
  | Fully supervised | LW-DETR-s | 57.4 |
  | Fully supervised | LW-DETR-m | 59.8 |
  | Semi-supervised, 10% labels | YOLOv8s + STAC | 43.1 |
  | Semi-supervised, 10% labels | YOLOv8m + STAC | 43.3 |
  | **Aerial subset, zero-shot** | Grounding DINO | 21.5 |
  | Aerial subset, zero-shot | Detic | 12.2 |
  | Aerial subset, zero-shot | Qwen2.5-VL | 4.6 |

  Zero-shot accuracy is under 2% on hard medical sets.
- **More RF100-VL findings:** the winner of the CVPR 2025 Foundational FSOD challenge (BEATON) scored **50.4 mAP on Roboflow20-VL, +17 over the baseline**. "Models that achieved higher performance on COCO did not necessarily improve real-world performance on RF100-VL". Annotator instructions gave multimodal LLMs inconsistent benefit — [RF100-VL paper](https://arxiv.org/html/2505.20612)
- **SAM 3 few-shot:** ODinW13 goes from **61.0 AP zero-shot to 71.8 AP 10-shot**, and RF100-VL from **15.2 to 36.5** — [SAM 3 paper](https://arxiv.org/html/2511.16719)
  - Exemplars instead of training: three interactive exemplar prompts add **+21.6 cgF1** over text-only, plateauing after about 4 — [SAM 3 paper](https://arxiv.org/html/2511.16719)
  - RF-DETR's README lists SAM 3 at 61.6 AP50:95 on RF100-VL, a number it attributes to the SAM 3 authors (probably full fine-tuning; setting not verified) — [rf-detr repo](https://github.com/roboflow/rf-detr)
- **YOLOE** ("Real-Time Seeing Anything", THU-MIG; arXiv 2503.07465, March 2025; repo created 2025-03-10; **AGPL-3.0**). Source: [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe/); license: [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe) (GitHub API)
  - Prompt modes: text, visual (example boxes), and prompt-free (a 4,585-name built-in vocabulary).
  - Ultralytics fine-tuning uses `YOLOEPESegTrainer`/`YOLOEPETrainer`, which "fuse your class names into the head and fine-tune from there".
  - YOLOE-v8-L transferred to COCO with "nearly 4× less training time" than YOLOv8-L and +0.6 box AP.
  - YOLOE-v8s beats YOLO-Worldv2-S by 3.5 AP on LVIS.
  - Visual-prompt results come back labeled "object0, object1…" and must be remapped to class names.
- **YOLOE-26** (Ultralytics, 2026): YOLOE-26x reaches 40.6 AP on LVIS minival with text prompts and 38.5 with visual prompts — [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe/); [YOLO26 docs](https://docs.ultralytics.com/models/yolo26/) (**AGPL-3.0**)
- **YOLO-World:** the original repo (Tencent AILab-CVC, created 2024-01-29) is **GPL-3.0** — [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World). The Ultralytics package that ships YOLO-World, YOLOE and YOLO26 is **AGPL-3.0** — [ultralytics](https://github.com/ultralytics/ultralytics) (both GitHub API)
- **Grounding DINO** (IDEA Research; ECCV 2024; **Apache-2.0**, repo created 2023-03-09) — [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) (GitHub API)
  - **MM-Grounding-DINO** (Zhao et al., arXiv 2401.02361, Jan 2024) is an open reimplementation with training and fine-tuning code, built in MMDetection (Apache-2.0); MM-G-DINO-Tiny outperforms G-DINO-Tiny — [arXiv](https://arxiv.org/abs/2401.02361)
  - Maintenance risk: MMDetection's last push was 2024-08-21 — [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) (GitHub API)
- **API-only:** Grounding DINO 1.5 Pro/Edge and DINO-X are reached through a paid API (DeepDataSpace). DINO-X Pro scores zero-shot 56.0 AP on COCO, 59.8 on LVIS-minival and 52.4 on LVIS-val — [Grounding-DINO-1.5-API](https://github.com/IDEA-Research/Grounding-DINO-1.5-API); [DINO-X-API](https://github.com/idea-research/dino-x-api)
- **OWLv2 / OWL-ST** (Minderer et al., Google; arXiv 2306.09683, June 2023): self-training in which an existing detector pseudo-annotates web image–text pairs, scaled to more than 1B examples. The key challenges were "the choice of label space, pseudo-annotation filtering, and training efficiency". **LVIS rare-class AP rose from 31.2% to 44.6%** (43% relative) for L/14 — [arXiv](https://arxiv.org/abs/2306.09683). Code lives in google-research/scenic (**Apache-2.0**, GitHub API) — [scenic](https://github.com/google-research/scenic)
- **Florence-2** (Microsoft; arXiv 2311.06242, Nov 2023): the FLD-5B dataset (5.4B annotations on 126M images) was built with "an iterative strategy of automated image annotation and model refinement", i.e. a data engine — [arXiv](https://arxiv.org/abs/2311.06242)
- **LAE-DINO** ("Locate Anything on Earth", AAAI 2025; repo **MIT**, GitHub API): the "LAE-Label Engine" collects, auto-annotates and unifies 10 remote-sensing datasets into **LAE-1M** (1M instances across 1,600 categories), which trains an open-vocabulary remote-sensing detector — [arXiv 2408.09110](https://arxiv.org/abs/2408.09110); [repo](https://github.com/jaychempan/LAE-DINO); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32672). It is *not* the same model as VIAT's existing transformers-based `LocateAnythingDetector` (`viat/utils/zero_shot_manager.py`, which installs via `locate_anything_req.txt`), per a local check of the repo.

#### Specialist ("student") detectors to distill into

| Model | Release / venue | License | Accuracy | Latency | Training notes | Source |
|---|---|---|---|---|---|---|
| **RF-DETR** | Repo 2025-03-19; arXiv 2511.09554; ICLR 2026 | **Apache-2.0** for N–L detection and all segmentation sizes; **PML 1.0** for XL/2XL detection and `rfdetr_plus` | COCO: L 56.5 AP, 2XL 60.1; Seg-2XL 49.9 mask AP; RF100-VL: **L 62.2 AP50:95** (88.2 AP50) | L 6.8 ms, 2XL 17.2 ms, Seg-2XL 21.8 ms (T4, TensorRT FP16, bs 1) | COCO-JSON or YOLO format; `model.train(dataset_dir=…)` | [rf-detr repo](https://github.com/roboflow/rf-detr) |
| RF-DETR training cost | — | — | — | — | "around an hour to train a model with 2,000 images for 15 epochs using an A100"; ≥50 epochs recommended for production; bs 16 on A100, bs 4 × grad-accum 4 on T4 | [Roboflow blog](https://blog.roboflow.com/train-rf-detr-on-a-custom-dataset/) (via search snippet) |
| **D-FINE** | ICLR 2025 Spotlight; released 18 Oct 2024 | **Apache-2.0** | COCO AP: N 42.8, S 48.5, M 52.3, L 54.0, X 55.8; Objects365-pretrained weights available | 2.12 / 3.49 / 5.62 / 8.07 / 12.89 ms (T4) | Custom COCO-format fine-tuning; example command uses 4 GPUs | [D-FINE repo](https://github.com/Peterande/D-FINE) |
| **RT-DETR** | 2023 (repo 2023-05-10) | **Apache-2.0** | Used as a student in the Voxel51 auto-label study | — | — | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) (GitHub API); [Voxel51 paper](https://arxiv.org/html/2506.02359) |
| **YOLO26** | Ultralytics; docs say released **January 2026** (an arXiv review, [2509.25164](https://arxiv.org/abs/2509.25164), and search results date the announcement to Sept 2025) | **AGPL-3.0** or Enterprise | COCO: n 40.9, s 48.6, m 53.1, l 55.0, x 57.5 AP; DOTA-v1.0 OBB +3.4 mAP vs YOLO11; **no VisDrone numbers in the docs** | 1.7–11.8 ms (T4, TensorRT) | NMS-free end-to-end, DFL removed, MuSGD optimizer, ProgLoss, **STAL (small-target-aware label assignment)**; docs say "YOLO27 coming soon" | [YOLO26 docs](https://docs.ultralytics.com/models/yolo26/) |
| YOLO26 STAL | — | — | — | — | STAL "adjusts assignment priors and spatial tolerance" so tiny, occluded or low-contrast instances get a supervisory signal | [Ultralytics blog](https://www.ultralytics.com/blog/how-ultralytics-yolo26-trains-smarter-with-progloss-stal-and-musgd) (via search snippet) |

### Inferences
- **Drone footage will behave like BDD/LVIS, not VOC.** In the Voxel51 study, domain shift plus small objects is the hard regime, so expect zero-shot auto-labels alone to yield only ~60–70% of human-label student quality. RF100-VL's aerial zero-shot score of 21.5 mAP for Grounding DINO supports this. A verification plus few-shot fine-tuning step *before* distillation is therefore essential for VIAT's aerial data.
- **Spend human effort on the labeler first.** The highest-leverage human input is ~10 verified instances per class used to fine-tune the *labeler* (SAM 3 or Grounding DINO: roughly 2× mAP on RF100-VL). Then re-label, then distill into a fast specialist.
- **Two different thresholds.**
  - Labels used to *train a student from scratch* should favor recall (α ≈ 0.2 was best).
  - Labels *accepted as ground truth* in VIAT's dataset need precision (a verifier or human gate).
  - VIAT should store every candidate with its score and provenance, and choose thresholds by sweeping them against a verified hold-out set, as the Voxel51 authors did.
- **License-driven model stack for a distributable desktop tool:**
  - Apache-2.0: SAM/SAM 2.1, Grounding DINO / MM-GDINO, OWLv2, RF-DETR N–L, D-FINE, RT-DETR, Autodistill.
  - AGPL-3.0: Ultralytics YOLO26 / YOLOE / YOLO-World.
  - GPL-3.0: the original YOLO-World.
  - Custom SAM License: SAM 3/3.1, which bans military use and has gated weights.
  - If VIAT's PyInstaller packaging (`installation_tools/`) bundles Ultralytics, AGPL obligations apply; flag this to whoever owns packaging.
- **The student doubles as a fast re-labeler.** RF-DETR-L runs at ~6.8 ms/image (T4, TensorRT) vs. ~30 ms/image for SAM 3 on an H200. Re-labeling passes over long videos can use the student, with SAM 3 reserved for flagged frames and keyframes.

### Gaps
- No controlled study was found that trains RF-DETR, D-FINE or YOLO26 on **SAM 3** pseudo-labels vs. human labels. The Voxel51 study used YOLO-World, YOLOE and Grounding DINO as labelers and YOLO11 and RT-DETR as students.
- No published few-shot numbers were found for YOLO-World prompt-tuning or fine-tuning, or for YOLOE on RF100-VL / ODinW-13. ODinW-13 numbers for models other than SAM 3 were not collected.
- The VRAM and time needed for SAM 3 fine-tuning are not documented in README_TRAIN.
- Florence-2's per-iteration refinement gains, and any accuracy metrics for Roboflow Rapid or Autodistill, were not found.

---

## 4. Semi-supervised detection with pseudo-labels (image and video): methods, measured gains, thresholds, failure modes and safeguards

### Takeaway
Teacher–student pseudo-labeling reliably adds **+3 to +6 mAP** when a decent labeled set exists:
- Soft Teacher: +3.6 on full COCO.
- MixPL: +4.3 to +6.4.
- Self-training (Zoph et al.): +1.3 to +3.4.

It adds much more in the low-label regime: Unbiased Teacher is about +10 mAP over supervised training at 0.5–2% of COCO. The biggest gains come with **1–50 labeled shots per class**, with diminishing returns beyond ~100–150.

Main failure modes and their mitigations:
- Confirmation bias: keep labeled samples in every batch, use soft labels.
- Class imbalance: class-balance loss, resampling of tail classes.
- Systematic under-labeling of small objects: size-aware thresholds, mosaic on pseudo-labeled images.
- Oscillating pseudo-targets: EMA teacher, adaptive (GMM) thresholds.
- Across all of these: a verified hold-out set to measure progress.

In video, tracking and optical-flow consistency recover missed detections and remove short-lived false positives. Tracker-filled boxes should carry soft or down-weighted labels.

### Cited Findings

#### Teacher–student SSOD methods (2021–2024)

| Method | Venue | Code / license | Mechanism | Measured results | Source |
|---|---|---|---|---|---|
| Unbiased Teacher (classic) | ICLR 2021 | [facebookresearch/unbiased-teacher](https://github.com/facebookresearch/unbiased-teacher), **MIT** (GitHub API; v2 repo also MIT) | Jointly trained student and gradually progressing (EMA) teacher; "class-balance loss to downweight overly confident pseudo-labels" | **+6.8 mAP** over prior SOTA at 1% COCO; ~**+10 mAP** over supervised at 0.5/1/2% COCO | [arXiv 2102.09480](https://arxiv.org/abs/2102.09480) |
| Soft Teacher (classic) | ICCV 2021 | [microsoft/SoftTeacher](https://github.com/microsoft/SoftTeacher), **MIT** (GitHub API) | Classification loss of each unlabeled box weighted by the teacher's score; box jittering selects reliable pseudo-boxes for regression | Full COCO baseline 40.9 → **44.5 mAP (+3.6)** with 123K unlabeled images; Swin detector 60.4 test-dev (+1.5), 52.4 mask AP (+1.2); with Objects365 pretraining 61.3 / 53.0 | [arXiv 2106.09018](https://arxiv.org/abs/2106.09018) |
| Consistent-Teacher | CVPR 2023 (Highlight) | [Adamdad/ConsistentTeacher](https://github.com/Adamdad/ConsistentTeacher): code **Apache-2.0** (GitHub API); arXiv text is CC BY-NC-SA 4.0 | Adaptive anchor assignment, 3D feature alignment, **GMM-based dynamic pseudo-box threshold** against "oscillating pseudo-targets" | 10% COCO, ResNet-50: **40.0 mAP** (~+3 over prior pseudo-label baselines); full COCO + unlabeled: 47.7 | [arXiv 2209.01589](https://arxiv.org/abs/2209.01589) |
| MixPL | arXiv Dec 2023 | [Czm369/MixPL](https://github.com/Czm369/MixPL), **Apache-2.0**, built on MMDetection | Pseudo-Mixup (false negatives "blend into background"); pseudo-Mosaic (more small/medium instances); labeled resampling of tail classes. **Thresholds: 0.7 for CE-loss detectors (Faster/Cascade R-CNN), 0.4 for RetinaNet, 0.3 for FCOS** | COCO-standard, Faster R-CNN: 25.06 / 33.48 / 37.16 at 1 / 5 / 10%. COCO-full + unlabeled2017: Faster R-CNN 46.2 (+5.2), FCOS 47.6 (+6.4), DINO 55.2 (+4.3); DINO Swin-L 60.2 | [arXiv 2312.07006](https://arxiv.org/html/2312.07006) |
| Efficient Teacher | arXiv Feb 2023 (Alibaba) | [AlibabaResearch/efficientteacher](https://github.com/AlibabaResearch/efficientteacher): code **GPL-3.0** (GitHub API; arXiv text CC BY 4.0) | First SSOD for **YOLOv5-style** one-stage anchor-based detectors: Dense Detector, Pseudo Label Assigner, Epoch Adaptor | Claims SOTA on VOC/COCO "with fewer FLOPs" (numbers not extracted) | [arXiv 2302.07577](https://arxiv.org/abs/2302.07577) |
| Sparse Semi-DETR | CVPR 2024 | not verified | Query Refinement Module plus Reliable Pseudo-Label Filtering; targets small and occluded objects | COCO-Partial 10%: **44.3 mAP**; COCO-Full: **51.3** (+0.9 over Semi-DETR); small-object AP +1.2 / +0.9 / +1.7 over Semi-DETR at 1 / 5 / 10% | [CVPR 2024 open access](https://openaccess.thecvf.com/content/CVPR2024/html/Shehzadi_Sparse_Semi-DETR_Sparse_Learnable_Queries_for_Semi-Supervised_Object_Detection_CVPR_2024_paper.html) (numbers via search snippet) |

- MixPL's diagnosis: pseudo-labels "amplify detector strengths while accentuating weaknesses… particularly for small and tail category objects". The number of small-object pseudo-labels "consistently lags behind ground truth" — [MixPL paper](https://arxiv.org/html/2312.07006)

#### 2026 practitioner evaluation
- **"Practical Insights into Semi-Supervised Object Detection Approaches"** (Wang, Balasubramaniyam, Sangem, Guevara, Caragea; arXiv 2601.13380, Jan 2026). Source for all points: [arXiv](https://arxiv.org/html/2601.13380)
  - Setup: MixPL, Semi-DETR and Consistent-Teacher at k-shot (1–150 labeled shots per class) on COCO, VOC and a 7-class beetle dataset.
  - COCO, 150-shot: **41.6 / 42.2 / 33.42 mAP**. VOC, 150-shot: 63.1 / 65.1 / 68.7.
  - Cost: Consistent-Teacher runs at ~10–15 ms/image with a 370–387 MB model; the DETR-based methods need 37–43 ms and 884–920 MB.
  - "Most pronounced improvements occur in the low-shot range (1–50 shots), with diminishing returns beyond 100 shots."
  - Per-class balanced labeling beats percentage-based sampling at similar cost; 100–150 shots per category are recommended for near-saturated performance on complex datasets.
  - In places, performance *decreased* as labeled instances were added, attributed to "noise in labeled or pseudo-labeled data".

#### Self-training fundamentals and confirmation bias (classic)
- **Zoph et al., "Rethinking Pre-training and Self-training"** (NeurIPS 2020):
  - Self-training gave **+1.3 to +3.4 AP on COCO across all dataset sizes**, reaching 54.3 AP (+1.5 over SpineNet); PASCAL segmentation reached 90.5 mIoU.
  - Self-training "works well exactly on the same setup that pre-training does not work".
  - Source: [arXiv 2006.06882](https://arxiv.org/abs/2006.06882)
- **Noisy Student** (Xie et al., CVPR 2020):
  - Loop: the teacher pseudo-labels 300M unlabeled images; an **equal-or-larger student is trained with noise** (dropout, stochastic depth, RandAugment) and then becomes the next teacher.
  - Result: 88.4% ImageNet top-1, 2.0 points better than the prior SOTA that used 3.5B weakly-labeled images.
  - Robustness: ImageNet-A 61.0 → 83.7%, ImageNet-C mCE 45.7 → 28.3, ImageNet-P mean flip rate 27.8 → 12.2.
  - Source: [arXiv 1911.04252](https://arxiv.org/abs/1911.04252)
- **Arazo et al.** (arXiv 1908.02983, 2019/2020): "naive pseudo-labeling overfits to incorrect pseudo-labels due to the so-called confirmation bias". Mitigations: soft pseudo-labels, mixup, and a **minimum number of labeled samples per mini-batch** — [arXiv](https://arxiv.org/abs/1908.02983)

#### Aerial / tiny-object SSOD
- **STOD** (Neural Computing and Applications, 2024):
  - A dense IoU-aware head scores localization quality.
  - GMM-based multi-threshold mining splits pseudo-boxes into strong and weak positives.
  - On **VisDrone-Partial**: **+3.1 mAP, +5.7 AP50, +3.0 AP75** over the baseline.
  - Source: [Springer](https://link.springer.com/article/10.1007/s00521-024-09936-z) (abstract via search)
- **S³OD, "Rethinking Scale Imbalance in SSOD for Aerial Images"** (Zhang et al., arXiv 2310.14718, Oct 2023; journal version in ISPRS J. 2025):
  - Identifies three size-related imbalances: pseudo-label, label-assignment and negative-learning.
  - Fixes: **size-aware adaptive thresholding**, size-rebalanced label assignment, and teacher-guided negative learning.
  - Evaluated on DOTA-v1.5; code "will be released soon".
  - Source: [arXiv](https://arxiv.org/abs/2310.14718); [ISPRS](https://www.sciencedirect.com/science/article/abs/pii/S0924271625000425)

#### Video-specific self-training: temporal consistency and tracking
- **Classic: RoyChowdhury et al. (CVPR 2019), "Automatic adaptation of object detectors to new domains using self-training":**
  - Pseudo-labels from unlabeled video combine high-confidence detections with **hard examples found by a tracker**, i.e. objects the detector missed.
  - **Soft labels via a modified distillation loss beat hard labels.**
  - Domains: a face detector adapted from WIDER to surveillance video, and a pedestrian detector adapted from BDD100K clear daytime to rain, fog and night.
  - Source: [arXiv 1904.07305](https://arxiv.org/abs/1904.07305)
- **SSVOD** (Mahmud, Liu, Yaman, Marculescu; WACV 2024):
  - Works from **sparse video annotations**.
  - **Flow-warped predictions from nearby frames** estimate temporal consistency; cross-IoU selects robust boxes and cross-divergence selects class labels.
  - Mixes hard and soft pseudo-labels to balance "confirmation bias and uncertainty noise".
  - Evaluated on ImageNet-VID, Epic-KITCHENS and YouTube-VIS.
  - Source: [arXiv 2309.01391](https://arxiv.org/abs/2309.01391)
- **OYSTER** (Zhang et al., Waabi; CVPR 2023; LiDAR):
  - Uses temporal consistency "to filter out noisy unsupervised detections".
  - Loop: **track, refine long tracks, discard short tracks, retrain on the refined pseudo-labels, repeat**.
  - The model continues to "self-improve given more rounds of iterative self-training".
  - Evaluated on PandaSet and Argoverse 2.
  - Source: [arXiv 2311.02007](https://arxiv.org/abs/2311.02007)
- **SAM2Auto** (Rocky & Wu; arXiv 2506.07850, June 2025):
  - Components: SMART-OD (automatic masks plus open-world detection), FLASH (frame-level video instance segmentation with cross-frame consistency), and statistical filtering of false positives.
  - Claims "no human intervention or dataset-specific training" and accuracy "comparable to manual annotation"; the abstract gives no numbers.
  - Source: [arXiv](https://arxiv.org/abs/2506.07850)
- **ST-SAM** (ACM MM 2025): turns pseudo-labels into hybrid prompts for SAM, "reducing error accumulation during self-training iterations" (camouflaged-object segmentation). With 1% labels it matches fully supervised methods — [arXiv 2507.23307](https://arxiv.org/abs/2507.23307)
- A 2025 UAV paper reports that SAM "struggles in novel and specialized domains such as aerial surveillance" — [arXiv 2507.05229](https://arxiv.org/pdf/2507.05229) (via search snippet; full text not verified)

### Inferences
- **Prefer offline rounds for VIAT.** The simplest fit for a desktop tool is offline, STAC-style rounds: the teacher (SAM 3 or an ensemble, plus a tracker) labels once, the student trains on fixed pseudo-labels, and the cycle repeats. That is effectively what the Meta engines do. Online EMA-teacher frameworks are tied to MMDetection or Detectron2 (MMDetection unpushed since Aug 2024) or to YOLOv5 (Efficient Teacher, GPL-3.0, last pushed 2023), which means high integration cost for modest extra gain.
- **Threshold policy by use:**
  - For a student trained offline on pseudo-labels: low threshold (~0.2–0.3) plus score-weighted loss or ignore regions.
  - For labels written into the dataset as accepted ground truth: high threshold plus a verifier.
  - For online SSOD: ~0.7 for CE-loss detectors, 0.3–0.4 for focal-loss detectors, or adaptive GMM and size-aware thresholds for tiny objects.
- **Safeguard checklist against error amplification:**
  - keep verified labels in every batch (Arazo);
  - soft or down-weighted labels for uncertain and tracker-filled boxes (Soft Teacher, RoyChowdhury);
  - strong augmentation or noise on the student (Noisy Student);
  - class resampling or class-balance loss (MixPL, Unbiased Teacher);
  - size-aware thresholds (S³OD, STOD);
  - an EMA or frozen teacher inside a round;
  - a verified hold-out set split *by video* (my suggestion, to avoid near-duplicate adjacent frames leaking between splits);
  - stop when the hold-out score stalls.
- **Video pseudo-labeling recipe** (combining OYSTER, RoyChowdhury and SSVOD):
  - Aggregate at track level: keep tracks with at least N frames and mean score ≥ τ.
  - Treat very short or flickering tracks as likely false positives (hard negatives).
  - Fill gaps inside confirmed tracks with tracker or SAM 2 propagated boxes at reduced loss weight.
  - Send frames where detector and tracker disagree to human review or hard-example mining.

### Gaps
- Numeric results for Efficient Teacher and SSVOD were not extracted (abstracts only).
- No 2024–2026 peer-reviewed paper with numbers on tracker-based pseudo-label filling specifically for **drone video** (VisDrone-VID, UAVDT) was found. An MDPI Electronics 2024 paper on "Bi-Directional Pseudo Label Recovery" for self-training — [MDPI](https://www.mdpi.com/2079-9292/13/12/2230) — returned HTTP 403 and could not be read.
- Per-round gains for multi-round self-training (Noisy Student iterations, OYSTER rounds) were not captured.
- Surveys such as "Semi-Supervised Object Detection: A Survey on Progress from CNN to Transformer" — [arXiv 2407.08460](https://arxiv.org/abs/2407.08460) — exist but were not mined.

---

## 5. Loop design for a single-workstation tool: number of iterations, stopping criteria, feeding back human corrections, aerial/drone domain adaptation, compute and wall-clock per iteration

### Takeaway
A practical VIAT loop is **2–4 rounds** of the same cycle:
- auto-label with an ensemble (SAM 3 text prompts plus YOLOE / YOLO-World, tiled for small objects);
- filter at track level;
- AI verification, using a verifier trained on logged human accept/reject decisions;
- human review of flagged tracks only, plus a small random audit;
- fine-tune the labeler on ~10–150 verified instances per class and the specialist on all verified plus pseudo-labels;
- re-label.

Stop when the gain on a frozen, video-split, human-verified hold-out drops below about one mAP point between rounds, or when the audited correction rate falls under target. Every published anchor points to minutes-to-an-hour of inference and roughly 1–4 GPU-hours of fine-tuning per round for a few thousand training frames on one high-end GPU. That is an extrapolation, see Inferences.

### Cited Findings
- **Retrain cadence in Meta's engines:** SAM retrained 6× (stage 1) + 5× (stage 2) — [SAM paper](https://arxiv.org/html/2304.02643). SAM 2: 2× (phase 2) + 5× (phase 3) — [SAM 2 paper](https://arxiv.org/html/2408.00714v2). SAM 3: 6× (phase 2) + 7× (phase 3), with the verifiers retrained 3× — [SAM 3 paper](https://arxiv.org/html/2511.16719)
- **Diminishing returns:** SAM 2 gained +3.0 / +5.8 / +3.7 / +0.7 J&F on SA-V val for successive data increments — [SAM 2 paper](https://arxiv.org/html/2408.00714v2). SSOD gains are largest at 1–50 labeled shots per class and diminish past 100 — [Practical Insights (2026)](https://arxiv.org/html/2601.13380)
- **Corrections as prompts vs. as training data:**
  - SAM 3 exemplar prompts: +21.6 cgF1 after 3 exemplars, plateau after about 4.
  - SAM 3 10-shot fine-tuning: ODinW13 61.0 → 71.8 AP (+10.8) and RF100-VL 15.2 → 36.5 (+21.3; deltas are my arithmetic).
  - Source: [SAM 3 paper](https://arxiv.org/html/2511.16719)
  - YOLOE visual prompts need "one example box per target" — [YOLOE docs](https://docs.ultralytics.com/models/yoloe/)
- **Verifier trained from human logs:** SAM 3's MV/EV verifiers were fine-tuned on Phase-1 human accept/reject labels and "roughly double" throughput — [SAM 3 paper](https://arxiv.org/html/2511.16719)
- **Audit sampling:** SAM professionally corrected a 500-image (~50k-mask) sample to certify its automatic masks — [SAM paper](https://arxiv.org/html/2304.02643)
- **Compute anchors:**
  - SAM 3: 30 ms/image with >100 objects on an H200 — [Meta AI blog](https://ai.meta.com/blog/segment-anything-model-3/)
  - SAM 3.1: 32 fps on an H100 for a medium number of objects, 16 objects per forward pass — [Meta AI blog](https://ai.meta.com/blog/segment-anything-model-3/)
  - Auto-labeling VOC: YOLO-World 197.2 s, YOLOE 204.9 s, Grounding DINO-T 2,290.3 s — [Voxel51 paper](https://arxiv.org/html/2506.02359)
  - RF-DETR: ~1 h for 2,000 images × 15 epochs on an A100 — [Roboflow blog](https://blog.roboflow.com/train-rf-detr-on-a-custom-dataset/)
  - YOLOE-v8-L: ~4× less training time than YOLOv8-L on COCO transfer — [YOLOE docs](https://docs.ultralytics.com/models/yoloe/)
  - SAM 3 fine-tuning runs on one GPU but some runs diverge — [README_TRAIN](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md)
  - Inference latency: RF-DETR-L 6.8 ms (T4) — [rf-detr](https://github.com/roboflow/rf-detr); YOLO26 1.7–11.8 ms (T4) — [YOLO26 docs](https://docs.ultralytics.com/models/yolo26/)
- **Aerial / small-object levers:**
  - Zero-shot on RF100-VL's aerial subset: Grounding DINO 21.5 mAP — [RF100-VL](https://arxiv.org/html/2505.20612)
  - SAM's automatic stage used "multiple overlapping zoomed-in image crops" for small masks — [SAM paper](https://arxiv.org/html/2304.02643)
  - Small-object pseudo-labels lag ground truth; pseudo-Mosaic adds small instances — [MixPL](https://arxiv.org/html/2312.07006)
  - STOD: +3.1 mAP on VisDrone-Partial — [Springer](https://link.springer.com/article/10.1007/s00521-024-09936-z)
  - S³OD: size-aware adaptive thresholding — [arXiv 2310.14718](https://arxiv.org/abs/2310.14718)
  - YOLO26: STAL — [YOLO26 docs](https://docs.ultralytics.com/models/yolo26/)
  - LAE-1M: auto-annotated remote-sensing data engine — [LAE-DINO](https://arxiv.org/abs/2408.09110)
- **Hard negatives:** going from 0 to 30 hard negatives per image raised cgF1 from 28.3 to 43.0 — [SAM 3 paper](https://arxiv.org/html/2511.16719)
- **Threshold selection:** α = 0.2 was best for training students on auto-labels — [Voxel51 paper](https://arxiv.org/html/2506.02359)
- **Hold-out realism:** better COCO accuracy did not predict better RF100-VL accuracy — [RF100-VL](https://arxiv.org/html/2505.20612). Adding labels sometimes *reduced* accuracy because of label noise — [Practical Insights](https://arxiv.org/html/2601.13380)

### Inferences
*These are design proposals synthesized from the cited findings, not sourced facts.*

**Proposed VIAT loop, one "round":**
1. **Round 0: build a gold set.** Hand-verify a class-balanced seed of ~10 instances per class to start, growing toward ~100–150 per class for hard classes. Build a **frozen hold-out split by video/flight** covering altitudes, times of day and scenes. Never train on the hold-out and never overwrite it by re-labeling.
2. **Auto-label.**
   - Run SAM 3 text prompts, plus YOLOE / YOLO-World as a second opinion, on keyframes (e.g. 1–5 fps) with overlapping tiles for drone footage (SAM's zoomed-crop precedent).
   - Keep *all* candidates with scores, model name, round number and prompt, i.e. full provenance.
   - Propagate or associate across frames with SAM 2/3 or the fast SOT/MOT trackers. Tracking design belongs to the other researchers.
3. **Temporal filter.**
   - Track-level voting: drop very short or flickering tracks, which become hard-negative candidates.
   - Back-fill gaps inside confirmed tracks and mark those boxes `tracker_filled`, with lower training weight.
   - Flag detector–tracker disagreements.
4. **AI verification.**
   - Crop-level verifier(s) answer "is this box a correct instance of class c?" (MV-like) and "are objects missing in this frame or tile?" (EV-like). Verifier model choice is another researcher's scope.
   - Start with off-the-shelf checks, then fine-tune the verifier on VIAT's own accept/reject logs once a few thousand decisions exist, as SAM 3 did.
5. **Human triage.**
   - Show humans only flagged tracks: low verifier score, disagreement, crowded scenes, track breaks (SAM 3 Phase 4 focus), and new or rare classes.
   - Add a small random audit (e.g. 1–2% of tracks) to estimate residual error.
   - The UI should be yes / no / fix per **track**, since verification is 6–9× cheaper than drawing (Papadopoulos).
6. **Feed corrections back through three channels:**
   - (a) *Immediately*, as exemplar or visual prompts (SAM 3 exemplars, YOLOE visual prompts) to fix the current video without training; gains plateau after ~4 exemplars.
   - (b) *Next round*, as training data: fine-tune the labeler (SAM 3 or Grounding DINO on verified crops, 10-shot scale) and the specialist (RF-DETR / D-FINE / YOLO26 on verified plus weighted pseudo-labels).
   - (c) *Meta-data*: accept/reject logs train the verifier and calibrate per-class thresholds, and rejected boxes become hard negatives (shadows, rooftops, look-alikes).
7. **Re-label and diff.** Re-run the improved labeler or student on the unlabeled pool, never on human-verified items. Report label churn (added / removed / changed boxes) per class and per size bucket.
8. **Stopping criteria**, for example: stop when any of these holds for two consecutive rounds:
   - hold-out mAP50-95 improves by less than ~1 point;
   - churn is below ~1–2% of boxes;
   - the audited correction rate is below the target (e.g. < 2–5% of tracks needing edits).

   The SAM 2 increments (+5.8 → +3.7 → +0.7) suggest 2–4 rounds will capture most of the gain. Hard caps on rounds avoid drift.

**Error-amplification monitors per round:**
- per-class and per-size-bucket precision/recall on the hold-out, to catch class collapse or small-object erosion (MixPL);
- pseudo-label *counts* per class and size vs. the gold set's distribution;
- teacher–student disagreement rate;
- label versioning with rollback.

**Aerial / drone specifics:**
- Tile both labeling and training.
- Use size-aware or class-wise thresholds rather than one global threshold.
- Mine hard negatives such as shadows (VIAT has an in-progress `viat/utils/shadow_filter.py` per git status) and feed them back as negatives.
- Fine-tune the labeler on in-domain drone crops before relying on zero-shot outputs (RF100-VL aerial zero-shot is only ~21.5 mAP for Grounding DINO).
- LAE-DINO/LAE-1M is a remote-sensing option. Whether it transfers to low-altitude, oblique drone video is untested in the material found.

**Rough compute per round on one high-end GPU** (extrapolated from the cited anchors; not measured):
- **Auto-label.** Example: 10 videos × 3 min × 30 fps = 54k frames, sampled at 5 fps gives about 9k frames. At ~30 ms/image (SAM 3 on an H200), 9k frames take about 4.5 min before tiling, and ~20 min with 4× tiling. A workstation GPU may be several times slower, so plan for up to ~1 h. Dense drone scenes with many objects slow SAM 3 *video* propagation, since latency scales with object count; SAM 3.1 multiplexing (16 objects per pass) helps. The alternative is per-frame detection plus a fast tracker, with SAM reserved for keyframes.
- **Fine-tune the specialist.** RF-DETR runs ~1 h per 2,000 images × 15 epochs on an A100. About 3,000 de-duplicated training frames (1–2 fps sampling) × 30 epochs comes to ≈3 h. Smaller models (RF-DETR-N/S, YOLO26s) or fewer epochs from COCO / Objects365 weights would be faster.
- **Fine-tune the labeler** (SAM 3 10-shot): time and VRAM undocumented, so budget a separate run and watch for divergence.
- **Total:** roughly a half-day to one day of unattended GPU time per round.

**License-aware defaults for an unattended local loop:**
- Labeler: SAM 3/3.1 (SAM License, gated, no military use), or an Apache-2.0 fallback: Grounding DINO / MM-GDINO + SAM 2.1.
- Student: RF-DETR N–L or D-FINE (Apache-2.0), or YOLO26 / YOLOE if AGPL is acceptable.
- **Avoid for a local tool:** API-only labelers (Grounding DINO 1.5 / DINO-X API, Roboflow Rapid, paid SAM 3.1 API).

### Gaps
- No published end-to-end benchmark of a *single-workstation* data engine (rounds × wall-clock × accuracy) was found. The compute estimates above are extrapolations.
- VRAM and time for SAM 3 fine-tuning on 24–48 GB consumer or workstation GPUs are undocumented, and no timing numbers for RF-DETR, D-FINE or YOLO26 training on consumer GPUs (e.g. RTX 4090) were found.
- No study was found that establishes stopping criteria (rounds, churn thresholds) for iterative pseudo-labeling in detection. The criteria above are heuristics.
- No quantitative comparison was found of "correction as exemplar prompt" vs. "correction as fine-tuning data" on the same data, beyond SAM 3's separate exemplar and 10-shot results.
- Nothing measured how well zero-shot foundation labelers do on **low-altitude drone video with VisDrone-style tiny objects** specifically (e.g. SAM 3 or YOLOE on VisDrone). RF100-VL's aerial subset and a secondary claim that SAM struggles on UAV footage are the closest evidence.
