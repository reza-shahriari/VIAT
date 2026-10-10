# Detector Error Reduction and Verification for Prompt-Based (Open-Vocabulary) Auto-Labels

Scope: per-image/per-frame label correctness (false positives, misses, wrong class, duplicates, loose boxes) for text/visual-prompted detectors and segmenters, and how to verify auto-labels automatically with a second model. Current as of 2026-09-30. Tool licenses were checked through the GitHub/Hugging Face APIs on that date. "Older classic" marks pre-2024 work.

---

## Q1. Which 2025–2026 open-vocabulary detectors/segmenters are strongest, and how do they fail?

### Takeaway
The strongest open model for prompt→detect/segment is **SAM 3 / SAM 3.1**, released under the custom SAM License, which allows commercial use with restrictions. On COCO/LVIS it matches or beats the API-only DINO-X, and it is far ahead on "concept present or absent" data that includes many negatives (SA-Co/Gold). It still reaches only about 74% of human performance and collapses zero-shot on aerial/remote-sensing text prompts (about 12 mAP on DIOR text-only). **YOLOE / YOLOE-26** (AGPL-3.0) is the fast real-time option with native cross-image visual prompts. **Rex-Omni** is the best MLLM-style detector on dense/small-object sets such as VisDrone, but its license is non-commercial. MLLMs (Qwen3-VL, Gemini, GPT-5) ground well on referring expressions but produce no confidence scores and score low on detection mAP. **All models fail badly on aerial imagery with large vocabularies**, mainly because of semantic confusion between overlapping class names, small-object score bias and false-positive saturation.

### Cited Findings

**SAM 3 (Meta, Nov 2025; arXiv 2511.16719, v2 28 Mar 2026)**
- The task is "Promptable Concept Segmentation": given a short noun phrase and/or image exemplars (positive or negative boxes), the model detects, segments and tracks *all* instances. Prompts are restricted to simple noun phrases. — [SAM 3 paper](https://arxiv.org/pdf/2511.16719)
- Size: about 850M parameters (about 450M vision encoder, 300M text encoder, 100M detector plus tracker). — [SAM 3 paper](https://arxiv.org/pdf/2511.16719)
- Speed: "On an H200 GPU, SAM 3 runs in 30 ms for a single image with 100+ detected objects". In video, latency scales with the number of objects, with near real-time for about 5 concurrent objects. — [SAM 3 paper (HTML v1)](https://arxiv.org/html/2511.16719v1)
- Real-time 30 FPS video needed multiple GPUs: up to 10 objects on 2 H200s, 28 objects on 4 H200s and 64 objects on 8 H200s. Cost scales linearly with the number of tracked objects. — [SAM 3 paper, §B Limitations](https://arxiv.org/pdf/2511.16719)
- Zero-shot results (Table 1): LVIS box AP 53.6, COCO box AP 56.4, COCO-O 55.7, LVIS mask AP 48.5. The intro text says "48.8 … vs. the current best of 38.5", a minor internal inconsistency. SA-Co/Gold cgF1 is 54.1 (masks) and 55.7 (boxes), against human 72.8 and 74.0, i.e. about 74% of human. — [SAM 3 paper](https://arxiv.org/pdf/2511.16719)
- On SA-Co/Gold (box cgF1), the same table scores the baselines as follows: OWLv2⋆ 24.5, DINO-X (API) 22.5, APE-D 17.3, Gemini 2.5 Flash 14.4, LLMDet-L 6.8 and Grounding DINO-T 3.4. SA-Co/Gold image–phrase pairs are 84.4% negatives (the concept is absent). — [SAM 3 paper](https://arxiv.org/pdf/2511.16719)
- ODinW13: 61.0 AP zero-shot and 71.8 AP 10-shot (Grounding DINO 1.5 Pro gets 67.9 10-shot). Roboflow100-VL: 15.2 AP zero-shot and 36.5 AP 10-shot. On RF100-VL zero-shot, SAM 3 is behind Grounding DINO-T (15.7). — [SAM 3 paper, Table 2](https://arxiv.org/pdf/2511.16719)
- Counting: CountBench 93.8% accuracy (MAE 0.12); PixMo-Count 86.2% (MAE 0.21). — [SAM 3 paper (HTML v1)](https://arxiv.org/html/2511.16719v1)
- Hallucination control: a "presence head" decouples "is the concept present?" from localization. A proposal's final score is its own score multiplied by the presence score. The presence head adds +1.5 cgF1 and +0.05 IL_MCC. Training with 30 hard negatives per image raises image-level IL_MCC from 0.44 to 0.68. Hard negatives are phrases that are absent from an image but that an earlier SAM 3 predicted masks for. — [SAM 3 paper](https://arxiv.org/pdf/2511.16719)
- Stated limitations:
  - SAM 3 "struggles to generalize to fine-grained out-of-domain concepts (e.g., aircraft types, medical terms)", especially in niche visual domains such as thermal imagery.
  - It does not support multi-attribute queries beyond one or two attributes, or referring expressions.
  - Video has "no shared object-level contextual information to aid in resolving ambiguities in multi-object tracking".
  - [SAM 3 paper, §B](https://arxiv.org/pdf/2511.16719)
- License: "SAM License", a custom Meta license last updated Nov 19, 2025.
  - It grants a "non-exclusive, worldwide, non-transferable and royalty-free limited license" to use, reproduce, distribute and create derivative works, so commercial use is allowed.
  - It forbids "end uses prohibited by Trade Controls, including those related to military or warfare purposes, nuclear industries or applications, espionage, or the development or use of guns or illegal weapons" and ITAR-controlled activities.
  - Redistribution must include a copy of the agreement.
  - Hugging Face weights are gated with manual approval.
  - [sam3 LICENSE](https://github.com/facebookresearch/sam3/blob/main/LICENSE); [HF facebook/sam3](https://huggingface.co/facebook/sam3)
- VisDrone is one of the SA-Co/Bronze evaluation domains; no per-domain VisDrone number was found (see Gaps). — [SAM 3 paper, Table 27](https://arxiv.org/pdf/2511.16719)

**SAM 3.1 (Mar 27, 2026)**
- "Object Multiplex" groups objects into fixed-capacity buckets and processes them jointly. It gives about a 7× speedup at 128 objects on one H100. Other optimizations: fewer CPU–GPU syncs, better `torch.compile` and batched post-processing. — [SAM 3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
- Accuracy changes are mixed rather than a uniform improvement:
  - Video PCS (text) cgF1: SA-V 30.3→30.5; YT-Temporal-1B 50.8→52.9; SmartGlasses 36.4→36.3; LVVIS 36.3→34.3.
  - VOS: MOSEv2 60.3→62.3; DAVIS17 84.4→85.1; SA-V test 89.7→89.3.
  - [SAM 3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
- The checkpoint is on Hugging Face as `facebook/sam3.1` (created 2026-03-26), gated, license "other". — [HF facebook/sam3.1](https://huggingface.co/facebook/sam3.1)

**SAM 3 on aerial / remote sensing (third-party evaluations)**
- "Promptable Concept Segmentation from Above" (arXiv, Jul 2026):
  - DIOR detection: text-only 11.67 mAP (16.79 AP50); box-only one-shot exemplar 66.55 mAP; text+box 63.79 mAP.
  - iSAID instance segmentation: text-only 11.85 mAP; box-only 32.77; text+box 35.54.
  - iSAID "Small Vehicle" reaches only 6.76 mAP with text and 9.15 at best.
  - Text prompts confuse windmills with airplanes. Shadows cost windmills and vehicles 2–4 mAP.
  - Concepts that need ground-level context (schools, churches) score near 0.
  - [arXiv 2607.09583](https://arxiv.org/html/2607.09583v1)
- Remote-sensing adaptations of SAM 3 (semantic segmentation, not boxes):
  - SegEarth-OV3 (Dec 2025) averages 53.4% mIoU over 20 remote-sensing segmentation datasets, 12.7 points above CorrCLIP. — [arXiv 2512.08730](https://arxiv.org/abs/2512.08730)
  - ProC-SAM3 (Jun 2026) averages 56.1% mIoU on 8 datasets, 3.9 points above SegEarth-OV3. — [arXiv 2606.21863](https://arxiv.org/html/2606.21863)

**YOLOE (THU-MIG; arXiv Mar 2025) and YOLOE-26 (Ultralytics, 2026)**
- Supports text prompts (RepRTA), visual prompts (SAVPE) and a prompt-free mode (LRPC, with a built-in vocabulary of 4,585 names). — [YOLOE paper](https://arxiv.org/abs/2503.07465); [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe)
- YOLOE-v8-L on LVIS minival (Fixed AP):
  - Text prompt: 35.9 AP (APr 33.2). Visual prompt: 34.2 AP. Speed: 102.5 FPS on T4 with TensorRT.
  - YOLOE-v8-S beats YOLO-Worldv2-S by 3.5 AP, with 3× less training cost and 1.4×/1.3× inference speedups on T4 and iPhone 12.
  - With visual prompts it beats T-Rex2 by 3.3 APr and 0.9 APc, using 2× less training data.
  - [YOLOE paper](https://arxiv.org/abs/2503.07465)
- YOLOE-26 on LVIS minival:
  - YOLOE-26x: 40.6 AP (text) and 38.5 AP (visual). Prompt-free variants: 29.9–31.1 (x) and 21.4–22.6 (s). YOLOE-26s: 30.8 (text) / 28.6 (visual).
  - Prompt-free variants trail text prompting by 8–11 AP.
  - Latency grows about 19% when the prompt set goes from 80 to 1,203 classes. Rare classes underperform. Prompts "work best for appearance, not relationships or context".
  - [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe)
- YOLOE-26S beats YOLO-World-S by 11.4 AP on LVIS, and YOLOE-26L beats YOLO-World-L by 10.0 AP. — [Ultralytics / search summary of YOLO26 docs](https://docs.ultralytics.com/models/yolo26)
- YOLO26 technical report (arXiv, Jun 2, 2026): 40.9–57.5 COCO mAP at 1.7–11.8 ms on T4 TensorRT. It uses an NMS-free dual head and "STAL", a label assignment "that guarantees positive coverage for small objects". — [arXiv 2606.03748](https://arxiv.org/abs/2606.03748)
- License: AGPL-3.0 for the THU-MIG/yoloe repo and ultralytics; Ultralytics also sells an Enterprise license. — [GitHub THU-MIG/yoloe](https://github.com/THU-MIG/yoloe); [Ultralytics docs](https://docs.ultralytics.com/models/yoloe)

**YOLO-World v2 (Tencent AILab, CVPR 2024; GPL-3.0)**
- YOLO-Worldv2-L: 35.5 LVIS minival AP at 80 FPS (T4, TensorRT). YOLO-Worldv2-S: 24.4 AP at 216.4 FPS. — [YOLOE paper, Table 1](https://arxiv.org/abs/2503.07465); license from [GitHub AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World)

**Grounding DINO 1.5/1.6, DINO-X (IDEA-Research; API-only for Pro/Edge)**
- DINO-X Pro (Nov 2024), zero-shot: 56.0 COCO, 59.8 LVIS-minival, 52.4 LVIS-val; rare classes 63.3 and 56.5. Grounding DINO 1.6 Pro: 55.4, 57.7, 51.1. — [DINO-X paper](https://arxiv.org/abs/2411.14347)
- DINO-X Edge at 640 px: 48.7 COCO and 44.5 LVIS-minival; 138.6 FPS with TensorRT FP32 on A100; 20.1 FPS FP16 on Orin NX. — [DINO-X paper, Table 7](https://arxiv.org/abs/2411.14347)
- Grounding DINO 1.5 Edge: 36.2 AP LVIS-minival at 75.2 FPS (TensorRT, A100). — [Grounding DINO 1.5 paper](https://arxiv.org/abs/2405.10300)
- Access is through the DINO-X API / DeepDataSpace cloud platform, which offers text, visual and prompt-free modes. — [DINO-X-API repo](https://github.com/IDEA-Research/DINO-X-API); [DINO-X platform](https://cloud.deepdataspace.com/en)
- The original open Grounding DINO is Apache-2.0. — [GitHub IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- Aerial is the weakest ODinW domain:
  - "AerialDrone" zero-shot AP: Grounding DINO 1.5 Pro 19.0 (vs its 58.7 ODinW13 average); Grounding DINO-L 12.6; OWL-ST 19.9; T-Rex2 visual prompt 16.0.
  - On ODinW35 "AerialMaritimeDrone_tiled", Grounding DINO 1.5 Pro gets 18.2 AP zero-shot vs 45.0 AP after fine-tuning.
  - [Grounding DINO 1.5 paper](https://arxiv.org/abs/2405.10300)

**OWLv2 (Google, NeurIPS 2023; older classic; Apache-2.0)**
- Self-training (OWL-ST) raised LVIS rare-class AP from 31.2% to 44.6%. — [OWLv2 paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/e6d58fc68c0f3c36ae6e0e64478a69c0-Paper-Conference.pdf)
- In SAM 3's evaluation, OWLv2⋆ scores 45.5 LVIS box AP and 46.1 COCO. It is the strongest baseline on SA-Co/Gold at 24.5 cgF1. — [SAM 3 paper](https://arxiv.org/pdf/2511.16719)
- Weights are Apache-2.0. — [HF google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble)

**LLMDet (CVPR 2025 highlight; Apache-2.0)**
- Reaches 50.6 AP on LVIS minival with Swin-L, +3.3/3.8/14.3 AP over its baseline for Swin-T/B/L. — [LLMDet paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Fu_LLMDet_Learning_Strong_Open-Vocabulary_Object_Detectors_under_the_Supervision_of_CVPR_2025_paper.pdf); [repo](https://github.com/iSEE-Laboratory/LLMDet)
- In SAM 3's harness LLMDet-L gets 42.0 LVIS box AP and 55.6 COCO AP, but only 6.8 cgF1 on SA-Co/Gold, which has many absent concepts. — [SAM 3 paper](https://arxiv.org/pdf/2511.16719)

**T-Rex2 (IDEA, ECCV 2024; API)**
- Visual-prompt LVIS minival: 37.4 AP (APr 29.9). — [YOLOE paper, Table 1](https://arxiv.org/abs/2503.07465)
- ODinW13 with visual prompts: 55.7 AP. — [Grounding DINO 1.5 paper](https://arxiv.org/abs/2405.10300)
- FSC147 counting: F1 91.5, MAE 10.9. — [Rex-Omni paper](https://arxiv.org/html/2510.12798v1)
- The repo license is not SPDX-identified. — [GitHub IDEA-Research/T-Rex](https://github.com/IDEA-Research/T-Rex)

**Rex-Omni (IDEA, Oct 15, 2025; CVPR 2026)**
- A 3B MLLM built on Qwen2.5-VL-3B that predicts quantized coordinates (0–999) as tokens. Trained with SFT on 22M samples plus GRPO reinforcement post-training. — [Rex-Omni paper](https://arxiv.org/html/2510.12798v1)
- Zero-shot F1@IoU0.5:
  - COCO 72.0 (Grounding DINO-T 69.8); LVIS 64.3 (Grounding DINO 47.7).
  - VisDrone 61.6 (Grounding DINO-T 55.2, SEED1.5-VL 55.9, Qwen2.5-VL-7B 34.5); Dense200 78.4.
  - COCO F1 drops to 15.9 at IoU 0.95, i.e. boxes are coarse at strict IoU.
  - [Rex-Omni paper](https://arxiv.org/html/2510.12798v1)
- MLLM failure modes it documents:
  - Duplicates: removing duplicates adds +15.3% on VisDrone for the SFT-only model.
  - "Large box" predictions covering many objects: 20.5% of Dense200 predictions for SFT, reduced to 3.5% after GRPO.
  - [Rex-Omni paper](https://arxiv.org/html/2510.12798v1)
- License: **IDEA License 1.0 plus the Qwen RESEARCH LICENSE** (non-commercial base). AWQ-quantized weights and a vLLM backend are available. — [Rex-Omni GitHub](https://github.com/IDEA-Research/Rex-Omni)

**General VLMs used as detectors**
- Qwen3-VL (Sep–Oct 2025; Apache-2.0), ODinW-13 mAP:
  - 235B-A22B-Instruct 48.6; 30B-A3B-Instruct 47.5; 32B-Instruct 46.6; 8B-Instruct 44.7; 4B-Instruct 48.2; 2B-Instruct 43.4.
  - Gemini 2.5 Pro scores 33.7–34.5 on the same benchmark.
  - RefCOCO-avg: Qwen3-VL-235B-Instruct 91.9 vs Gemini 2.5 Pro 74.6 and GPT-5 (high) 66.8.
  - [Qwen3-VL technical report](https://arxiv.org/pdf/2511.21631); license from [HF Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- Qwen3-VL uses coordinates normalized to [0, 1000], unlike Qwen2.5-VL. To compute mAP, its authors "set confidence scores to 1.0", i.e. the MLLM outputs have no usable detection confidence. — [Qwen3-VL technical report](https://arxiv.org/pdf/2511.21631)
- Qwen2.5-VL licenses differ by size: 3B uses the Qwen Research license (non-commercial), 7B is Apache-2.0, 72B uses the Qwen license. — [HF API: 3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [72B](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)
- Roboflow100-VL zero-shot mAP:
  - Grounding DINO 15.7 (aerial 21.8), Gemini 2.5 Pro 13.3 (aerial 8.4), MQ-GLIP 12.0, OWLv2 10.1 (aerial 15.3), Detic 9.5, Qwen2.5-VL-72B 5.4 (aerial 4.6).
  - Grounding DINO drops from 49.2 on ODinW-13 to 16.0 on RF100-VL.
  - Specialist open-vocabulary detectors beat MLLMs "partly because generalist models lack proper confidence scoring and non-maximal suppression". Annotator instructions give "inconsistent benefit".
  - [RF100-VL paper](https://arxiv.org/html/2505.20612v1)
- Moondream 3 Preview (Sep 2025): 9B MoE with 2B active, native detection output. License is BSL 1.1 with an additional use grant; commercial use is allowed except for competing paid services. — [HF moondream3-preview](https://huggingface.co/moondream/moondream3-preview); [Moondream blog](https://moondream.ai/blog/moondream-3-preview)
- Florence-2 (Microsoft; older, CVPR 2024) is MIT-licensed. InternVL3.5-8B (Aug 2025) and Molmo-7B-D (Sep 2024) are Apache-2.0. — [HF Florence-2-large](https://huggingface.co/microsoft/Florence-2-large); [HF InternVL3_5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B); [HF Molmo-7B-D](https://huggingface.co/allenai/Molmo-7B-D-0924)

**Aerial transfer of open-vocabulary detectors (the most relevant failure study for VisDrone-style data)**
- "Do Open-Vocabulary Detectors Transfer to Aerial Imagery?" (arXiv, Jan 2026). Setup: LAE-80C, 3,592 images, 86,558 instances, 80 classes from DIOR/DOTA-v2/FAIR1M/xView. Results:
  - OWLv2 has the best F1, 27.6% (P 31.3%, R 24.7%), but produced 47,058 false positives against 21,408 true positives.
  - LLMDet: F1 12.5% (P 44.1%, R 7.3%). Grounding DINO in Oracle mode: F1 7.4% (R 3.9%). YOLOE-11: 3.9%. YOLO-World: 2.8%.
  - Grounding DINO in "Global" mode (all 80 classes prompted) found only 218 of 86,558 objects.
  - Cutting the vocabulary from 80 to about 3.2 classes per image gave a "15× improvement". The authors identify semantic confusion, not visual extraction, as the main bottleneck.
  - Adding an "aerial view of" prefix raised precision but lowered recall; synonym expansion lowered recall.
  - [arXiv 2601.22164](https://arxiv.org/html/2601.22164v1)
- Scale bias: foundation-model open-vocabulary detectors give small objects systematically lower confidence. For Grounding DINO the mean is 0.180 for small objects vs 0.520 for large; the log(area)–confidence correlation is r = 0.36–0.64 across Grounding DINO, OWL-ViT and YOLO-World. — [arXiv 2607.10993](https://arxiv.org/html/2607.10993v1)

### Inferences
- For VIAT's general video, SAM 3/3.1 (text prompts, masks and boxes, tracking) is the strongest local default. Its presence head and hard-negative training are the main reason it hallucinates far less on absent concepts than grounding-trained detectors: compare LLMDet-L at 6.8 and Grounding DINO-T at 3.4 cgF1 on negative-heavy SA-Co/Gold. The cgF1 metric penalizes predictions on negative pairs, so this gap measures false positives on absent concepts, not general accuracy.
- For drone footage with many small objects, text-only zero-shot prompting is not good enough with any model: about 12 mAP for SAM 3 text-only on DIOR, and F1 below 28% for all open-vocabulary detectors on LAE-80C. The pipeline has to rely on exemplars or few-shot adaptation (Q2), per-scene vocabulary restriction, tiling (Q3) and scale-aware thresholds (Q6).
- YOLOE-26 is the practical companion for high-throughput prompting and cross-image visual prompts, but AGPL-3.0 has distribution implications for a desktop tool. Rex-Omni is technically attractive for dense small objects, but the Qwen Research base makes it non-commercial.
- MLLMs such as Qwen3-VL are better used as verifiers (Q4) than as primary box generators. Their boxes have no confidence, can be coarse at strict IoU and duplicate in dense scenes.

### Gaps
- No per-domain SA-Co/Bronze number for VisDrone was found in the SAM 3 paper. I found no published VisDrone box AP for SAM 3, SAM 3.1 or YOLOE-26.
- Exact release date of YOLOE-26 / YOLO26 weights: the arXiv report is dated June 2026, but the weights may predate it; not verified.
- DINO-X / Grounding DINO 1.6 API pricing: the pricing page exists but numbers could not be retrieved.
- No verified zero-shot detection numbers for Florence-2, Moondream 3 or InternVL3.5 were collected.
- VRAM requirements for SAM 3 / SAM 3.1 on consumer GPUs (e.g., 24 GB cards) were not found in primary sources.

---

## Q2. "Correct once, fix everywhere": how do visual exemplars turn a few corrections into a better prompt across a dataset or video?

### Takeaway
Exemplars give the largest measured single-step accuracy gains in this whole subtopic. Within one image, SAM 3 gains **+18 to +21 AP+** over T-Rex2 from one exemplar box. Interactive exemplar clicks add **+21.6 cgF1 after 3 clicks** over text-only, because a positive or negative box generalizes to all similar instances. On remote sensing, one box exemplar lifts SAM 3 on DIOR from **11.7 to 63.8–66.6 mAP**. However, **SAM 3 exemplars are per-image** (ROI-pooled from the current image) and are not natively transferable across images. Cross-image "fix everywhere" needs one of four things: YOLOE `refer_image`, OWLv2 image queries, a feature memory bank (DINOv2/DINOv3 prototypes), or, most powerful, **few-shot fine-tuning** on the corrected labels (SAM 3 RF100-VL goes from 15.2 to 36.5 AP with 10 shots).

### Cited Findings
- SAM 3 exemplars are "a bounding box and an associated binary label (positive or negative)". A positive box on one dog makes the model "detect all dogs in the image". Each exemplar is encoded from "an embedding for the position, an embedding for the label, and ROI-pooled visual features". — [SAM 3 paper](https://arxiv.org/pdf/2511.16719)
- Exemplars "are particularly useful when the model initially misses some instances, or when the concept is rare". Prompts must stay category-consistent: "fish" cannot be refined with exemplars of just the tail, and the text should be updated instead. — [SAM 3 paper](https://arxiv.org/pdf/2511.16719)
- One exemplar, AP+ on positive images only:

  | Benchmark | Text only | Image exemplar only | Text + image | T-Rex2 (image) |
  |---|---|---|---|---|
  | COCO | 58.8 | 76.8 | 78.1 | 58.5 |
  | LVIS | 54.7 | 76.0 | 78.4 | 65.8 |
  | ODinW13 | 63.1 | 82.2 | 81.8 | 61.8 |

  — [SAM 3 paper, Table 3](https://arxiv.org/pdf/2511.16719)
- Interactive loop: "missed ground truths are candidate positive prompts, false positive detections are candidate negative prompts".
  - After 3 clicks, interactive PCS beats text-only by +21.6 cgF1 and beats ideal per-instance box-to-mask correction (PVS) by +2.0.
  - Gains plateau after about 4 clicks "as exemplars cannot fix poor-quality masks". Switching to per-instance correction after that yields further gains.
  - [SAM 3 paper, Fig. 7](https://arxiv.org/pdf/2511.16719)
- An exemplar alone is ambiguous: given a dog, should the model find all dogs or only that breed? SAM 3 therefore does worse with a visual-only prompt than with text on SA-Co/Gold, and the paper recommends text plus exemplar. — [SAM 3 paper, §F.2](https://arxiv.org/pdf/2511.16719)
- Remote sensing (SAM 3, one exemplar box):
  - DIOR: 11.67 → 63.79 mAP (text+box) or 66.55 (box only). Airport class: 0 → 92.25 mAP.
  - iSAID: 11.85 → 35.54 mAP (text+box).
  - Combining text and box was 2.76 mAP worse than box-only on DIOR ("cross-modal interference"). Exemplars erase the "small vehicle" vs "large vehicle" size distinction that text preserves.
  - [arXiv 2607.09583](https://arxiv.org/html/2607.09583v1)
- Cross-image transfer in SAM 3 is not native. An Ultralytics maintainer says: "If you want to stay strictly in SAM 3, you'd currently need to run each image independently". Suggested alternatives are YOLOE's `refer_image` or a "shared canvas" hack (concatenate the reference and target images side by side; FSS-SAM3 repo). — [Ultralytics discussion #23968](https://github.com/orgs/ultralytics/discussions/23968)
- Ultralytics `SAM3SemanticPredictor` exposes text and exemplar-box concept segmentation, with positive or negative boxes. — [Ultralytics SAM 3 docs](https://docs.ultralytics.com/models/sam-3)
- YOLOE visual prompts:
  - Boxes on a reference image (`refer_image`) define the prompt, which then applies to other images; "prompts from one image apply consistently to other frames in video workflows". — [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe)
  - Visual-prompt AP is close to text-prompt AP: YOLOE-v8-L 34.2 vs 35.9. With visual prompts YOLOE-v8-L beats T-Rex2 by 3.3 APr. — [YOLOE paper](https://arxiv.org/abs/2503.07465)
  - Outputs are generic "object0, object1" labels that the user must map back. — [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe)
- OWLv2 image-guided (one-shot) detection:
  - A query image replaces the text prompt. In Hugging Face transformers, `embed_image_query` takes the box with the highest objectness as the query embedding. — [HF transformers PR #49200](https://github.com/huggingface/transformers/pull/49200)
  - Users have reported one-shot results not matching the official Colab (reproducibility pitfall). — [scenic issue #1096](https://github.com/google-research/scenic/issues/1096)
- Memory-bank / prototype approaches that need no training:
  - "No time to train!" (Jul 2025) stores DINOv2 features of a few reference masks in a memory bank, aggregates them into class prototypes and matches SAM proposals against them. Results: COCO FSOD 36.8 nAP, PASCAL VOC few-shot 71.2 nAP50, cross-domain FSOD 22.4 nAP. — [arXiv 2507.02798](https://arxiv.org/abs/2507.02798); [code](https://github.com/miquel-espinosa/no-time-to-train)
  - ConceptBank (Feb 2026) is a parameter-free calibration for SAM 3. It builds class-wise visual prototypes from target-domain evidence and suppresses outliers under data and concept drift, including remote sensing. Code is released. — [arXiv 2602.06333](https://arxiv.org/abs/2602.06333)
- Few-shot fine-tuning as the strongest "fix everywhere" lever:
  - SAM 3: ODinW13 61.0 → 71.8 AP (10-shot); RF100-VL 15.2 → 36.5 AP (10-shot). Few-shot training disables mask losses for box-only data, runs 40 epochs at a tenth of the learning rate with batch size 2. — [SAM 3 paper](https://arxiv.org/pdf/2511.16719)
  - Grounding DINO on RF100-VL: 15.7 → 33.3 mAP with 10 shots. MLLMs barely improve from in-context few-shot examples: Qwen2.5-VL 7.5 and Gemini 2.5 Pro 9.2 mAP. — [RF100-VL paper](https://arxiv.org/html/2505.20612v1)
- DINO-X also supports visual-prompt and prompt-free modes, via API. — [DINO-X platform](https://cloud.deepdataspace.com/en)

### Inferences
- A practical "correct once, fix everywhere" design for VIAT:
  1. **Within an image or frame**: turn each user deletion into a SAM 3 negative exemplar and each added box into a positive exemplar, then re-run the concept prompt. The paper's simulation implies about 3 clicks recover most of the achievable gain.
  2. **Across a video**: SAM 3's text prompt is global to all frames, while exemplars are frame-specific. VIAT would need to re-inject exemplars on frames where errors recur, or keep an exemplar crop bank and re-apply it per frame.
  3. **Across a dataset**: store verified crops as prototypes (DINOv2/DINOv3/SigLIP 2 embeddings) and use them to filter or re-score new detections (Q5). YOLOE `refer_image` is a second path, and a short few-shot fine-tune of SAM 3 or YOLOE on accumulated corrections is a third.
- For aerial data, the evidence says exemplars matter more than prompt wording: box exemplars far outperform text. Prompt engineering such as "aerial view of" or synonym lists did not help (Q1).
- The "shared canvas" concatenation trick is an unvalidated community hack. It halves effective resolution per image, which is likely harmful for small objects (inference).

### Gaps
- No published measurement was found for negative-exemplar-only gains, i.e. how much adding negatives reduces false positives separately from positives. SAM 3's Fig. 7 mixes both.
- No study was found that quantifies cross-frame reuse of SAM 3 exemplars in video, or cross-image transfer accuracy of YOLOE visual prompts vs same-image prompts.

---

## Q3. Ensembles and agreement: WBF vs NMS/soft-NMS, cross-model agreement, TTA consistency, SAHI slicing

### Takeaway
**WBF helps only when fusing different, reasonably accurate models.** On COCO it adds about +1.3 mAP for two EfficientDets and reaches 56.1 mAP from a best single model at 52.1 with a 7-model ensemble plus flip TTA. Flip TTA alone adds about +0.4–0.5 mAP. Replacing NMS with WBF on a *single* model *hurts*. **SAHI slicing** gives +5–7 AP inference-only on VisDrone and +13–15 AP with slice-aware fine-tuning, the largest documented gain for small objects. No rigorous study was found that measures WBF or cross-model agreement specifically for open-vocabulary detector ensembles. Agreement as a confidence or triage signal is plausible but unquantified for this setting.

### Cited Findings
- WBF (Solovyev, Wang, Gabruseva; Image and Vision Computing 2021; MIT license) averages coordinates of overlapping boxes weighted by confidence instead of discarding boxes. — [WBF paper](https://arxiv.org/abs/1910.13302); [repo](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
- Two-model COCO ensemble (EfficientDet-B6 0.513 and B7 0.521 mAP), mAP 0.5:0.95:
  - NMS 0.5269; soft-NMS 0.5239; NMW 0.5285; **WBF 0.5344** (about +1.3 over the best single model).
  - Optimal IoU thresholds were about 0.75–0.77 at mAP 0.5:0.95.
  - [WBF paper, Table 1](https://arxiv.org/abs/1910.13302)
- Flip TTA on one model (EffDet-B7 0.521): NMS 0.5233, soft-NMS 0.5210, NMW 0.5250, WBF 0.5262. — [WBF paper, Table 2](https://arxiv.org/abs/1910.13302)
- A 7-model ensemble with flip TTA (EfficientDet B4–B7, 2× DetectoRS, YOLOv5x) reached 56.1 mAP on COCO val and 56.4 on test-dev with WBF, vs 52.1 for the best single model. — [WBF paper, Table 3](https://arxiv.org/abs/1910.13302)
- Single-model caveat: "replacing the NMS method at the output of the individual model with WBF results in the degradation of the model performance … due to the excessive number of low scored wrong predictions". WBF "gives worse results than NMS" when there are many overlapping boxes with different scores. — [WBF paper](https://arxiv.org/abs/1910.13302)
- In practice, open-vocabulary ensembles for pseudo-labeling (Grounding DINO, OV-DINO, YOLO-World, OWLv2) lower each model's confidence threshold to keep more candidates, then concatenate boxes and apply same-class NMS at IoU 0.4. This is as summarized from the ROVI dataset paper; the details were not independently read. — [ROVI (arXiv 2508.01008)](https://arxiv.org/pdf/2508.01008)
- SAHI (ICIP 2022; older classic; MIT license): slicing-aided inference on VisDrone adds +6.8, +5.1 and +5.3 AP for FCOS, VFNet and TOOD. With slicing-aided fine-tuning the cumulative gains are +12.7, +13.4 and +14.5 AP. It works on top of any detector without retraining. — [SAHI paper](https://arxiv.org/abs/2202.06934); [repo](https://github.com/obss/sahi)
- Tiling appears in open-vocabulary benchmarks, but its zero-shot benefit is modest there: Grounding DINO 1.5 Pro scores 18.2 AP zero-shot vs 45.0 fine-tuned on ODinW35 "AerialMaritimeDrone_tiled". — [Grounding DINO 1.5 paper](https://arxiv.org/abs/2405.10300)
- For MLLM detectors, duplicate removal matters: the SFT-only Rex-Omni gained +15.3 F1 points on VisDrone just from duplicate removal. — [Rex-Omni paper](https://arxiv.org/html/2510.12798v1)
- TTA-based consensus is being used for *label-free* quality estimation. The "Cumulative Consensus Score" (arXiv Sep 2025) evaluates deployed detectors without labels through augmentation consistency. I did not read it in detail. — [arXiv 2509.12871](https://arxiv.org/pdf/2509.12871)
- SAM 3 itself removes duplicates in video: two masklets that both overlap the same detection (IoU ≥ 0.1) for at least ⌈T/2⌉ frames count as duplicates, and the later-starting one is dropped. — [SAM 3 paper, §C.3](https://arxiv.org/pdf/2511.16719)

### Inferences
- For VIAT, WBF is useful when running genuinely different detectors, for example SAM 3 plus YOLOE-26 plus a fine-tuned closed-set YOLO. It should not be a drop-in replacement for per-model NMS. Keep class-aware NMS within each model and use WBF across models.
- Cross-model disagreement makes a cheap triage signal. Examples: a box found by only one of three models, a class mismatch between models, or a large IoU spread between their boxes for the same object. These are strong candidates for the "small ranked set of likely errors". The ranking value is inferred, not measured, for open-vocabulary detectors (see Gaps).
- For VisDrone-style frames, SAHI-style tiling combined with a fast model (YOLOE-26) or SAM 3 is the most evidence-backed way to reduce misses of small objects. Tiling multiplies per-frame cost by the number of tiles, and cross-tile duplicates need merging with NMS/WBF (SAHI does this).
- Flip/multi-scale TTA is cheap and gives small mAP gains (about +0.5). Its bigger value for auto-labeling is probably as a per-box stability or uncertainty feature for triage (inference).

### Gaps
- I found no paper that quantifies WBF vs NMS for fusing *open-vocabulary* detectors (e.g., SAM 3 + YOLOE + OWLv2), or that measures how well cross-model agreement predicts label errors (AUROC or precision@k) in this setting.
- I found no study of SAHI with SAM 3 or YOLOE-26 on VisDrone.

---

## Q4. VLM verification: using a second (vision-language) model to verify, re-classify or check exhaustivity

### Takeaway
The best-documented system is **SAM 3's data engine**. Llama 3.2 was fine-tuned on about 10M human verification labels per task, after pre-training on 200M+ pairs, to produce a **mask verifier (MV)** and an **exhaustivity verifier (EV)**. These **match or beat single human verifiers**: MV accuracy 77.7% vs human 75.0%; EV 81.1% vs 81.8%. They **double data-engine throughput**, with 5× faster handling of negative phrases. **Used at test time on SAM 3's own outputs, they add +8.3 cgF1** (54.0 → 62.3, closing about half the gap to human). Zero-shot, general VLMs used as judges give smaller but real gains:
- GPT-4o review of Grounding DINO labels: +3.1 AP downstream.
- Gemini 2.5 Flash judging industrial-diagram detections: mAP 0.728 → 0.925.

These judges struggle with dense, small and closely packed objects. Local 7B-class VLMs on vLLM run roughly **13–21 image requests/s on one A100**, which makes per-detection or per-track verification feasible offline.

### Cited Findings

**SAM 3 AI verifiers (the most relevant precedent)**
- Pipeline: SAM 3 proposes masks, then MV has annotators "accept or reject masks based on their quality and relevance to the NP". EV checks "if all instances of the NP have been masked". Pairs that fail go to manual correction. — [SAM 3 paper, §5](https://arxiv.org/pdf/2511.16719)
- Training: Phase-1 human accept/reject labels were used to "fine-tune Llama 3.2 … to create AI verifiers". Training data also came from the correction task: every manually added mask is a positive MV example, and every exhaustively finished job is a positive EV example. The mixed data (over 200M image–text pairs) pre-trained a "foundational AI verifier", then about 10M high-quality examples fine-tuned one verifier per task. — [SAM 3 paper, §D](https://arxiv.org/pdf/2511.16719)
- Input design:
  - **MV**: the verifier sees "two crops of the image, a zoomed-out view where the object is highlighted via a box and a zoomed-in view where the mask is highlighted". The paper says this "allows better visualization of small objects, and avoids color confusion from mask overlay". There are 5 choices: Accept, Reject, Accept as text, Flag label, Whole image. — [SAM 3 paper, Table 21](https://arxiv.org/pdf/2511.16719)
  - **EV**: red boxes for all current detections are drawn on the full image, and the model picks one of 6 options, including "Reject" (missing objects that can be separated) and "Reject but unseparated". — [SAM 3 paper, Table 22](https://arxiv.org/pdf/2511.16719)
  - Scores come from the "logits of the choice tokens … soft-max over the logits to get their probabilities", not free text. — [SAM 3 paper, Tables 21–22](https://arxiv.org/pdf/2511.16719)
  - The EV verifier doubles as a presence detector: "presence score … defined as 1 − P(Accept | no boxes as input)". — [SAM 3 paper, Table 22](https://arxiv.org/pdf/2511.16719)
- Accuracy vs humans (Table 23; ground truth is the majority vote of other annotators, averaged over 7 domains):
  - MV: human 75.0%, AI 77.7%. EV: human 81.8%, AI 81.1%.
  - Per-domain, AI MV beats humans in all 7 domains. AI EV is slightly below humans in Crowded (81.3 vs 82.4), Food&Drink, Sports Equipment and Wiki-Common (88.4 vs 91.5).
  - [SAM 3 paper, Table 23](https://arxiv.org/pdf/2511.16719)
- Domain transfer: "the MV AI verifier is typically on par with human verifiers without any domain specific data; the EV AI annotator is typically worse than human in a zero-shot evaluation, but can reach human performance with only thousands of domain specific data points". — [SAM 3 paper](https://arxiv.org/pdf/2511.16719)
- Test-time use on SAM 3 outputs (SA-Co/Gold cgF1):

  | Configuration | Average | Food&Drink | Attributes | Wiki-Common | Crowded |
  |---|---|---|---|---|---|
  | SAM 3 | 54.0 | 56.1 | 54.2 | 42.9 | 60.3 |
  | + EV presence score | 61.2 | – | – | – | – |
  | + EV and MV | 62.3 | 68.0 | 70.9 | 52.3 | 62.7 |
  | Human | 72.8 | – | – | – | – |

  Replacing SAM 3's presence score with the EV score gives +7.2, "from both IL_MCC and pmF1", because it "serves as a better calibration of object scores". Removing MV-rejected masks gives +1.1 more. — [SAM 3 paper, Tab. 9d and Tab. 14](https://arxiv.org/pdf/2511.16719)
- Throughput, image engine on a new fine-grained domain with verifiers never trained on it (seconds per image–phrase pair):

  | Setting | Mixed data | Positive phrase | Negative phrase |
  |---|---|---|---|
  | Human from phrase | 90 | 236 | 71 |
  | Human from masks | 86 | 205 | 70 |
  | Engine, all human | 50 | 207 | 30 |
  | Engine with AI verifiers | 23 | 152 | 6 |

  This is "a 5x speed up on negative phrases and a 36% speed up for positive phrases" and about 2× overall. — [SAM 3 paper, Table 15](https://arxiv.org/pdf/2511.16719)
- Video caveat: starting human annotators from pseudo-label masklets was **40% slower** than annotating from the noun phrase (3,221 s vs 2,307 s per positive video–phrase pair) but produced 9% more masklets. — [SAM 3 paper, Table 16](https://arxiv.org/pdf/2511.16719)
- Verification also enables human-free domain adaptation: training on "SA-Co/SYN-Food" (SAM 3 pseudo-labels cleaned by MV/EV, no humans) scales similarly to human-verified data and "eventually" catches up. Unverified pseudo-labels (PL-Food) lag far behind. — [SAM 3 paper, §A.3](https://arxiv.org/pdf/2511.16719)
- SAM 3 Agent (an MLLM calling SAM 3 as a tool, inspecting returned masks and iterating) scores zero-shot on ReasonSeg val (gIoU):
  - Qwen2.5-VL 7B: 62.2. Llama4 Maverick: 68.5. Qwen2.5-VL 72B: 74.6. Gemini 2.5 Pro: 77.0. With Gemini 2.5 Pro it reaches 45.3 on OmniLabel.
  - [SAM 3 paper, Table 8](https://arxiv.org/pdf/2511.16719); [HTML v1](https://arxiv.org/html/2511.16719v1)

**Zero-shot general VLMs as judges**
- DART (Expert Systems with Applications; code public):
  - GPT-4o reviewed Grounding DINO pseudo-labels with boxes and class labels "drawn on top of the image", compressed to 512 px on the long side. It answered three checks: does each box enclose one object, are all objects localized, and are the boxes tight.
  - Review was limited to about 5K images that had multiple annotations or scores below 0.5.
  - Downstream YOLOv8n AP went from 0.771 to 0.802 with review; the full pipeline reached 0.832 on a 23-class heavy-machinery dataset of more than 15K images.
  - [DART (arXiv 2407.09174)](https://arxiv.org/html/2407.09174)
- "VLM as a Judge for Object Detection in Industrial Diagrams" (arXiv, Oct 2025):
  - Gemini 2.5 Flash and Gemma 3 judged completeness (missed objects), false detections, localization and class.
  - Grounding strategies compared: colored boxes with text references; removing already-detected objects and showing the remaining image; text tags; explicit coordinates; and combinations. Combined cues worked best.
  - Missing-object assessment F1: 0.876 for Gemini 2.5 Flash (P 0.833, R 0.924) and 0.815 for Gemma 3.
  - After judge-guided correction, mAP rose from 0.728 to 0.925 (Gemini) and 0.882 (Gemma 3).
  - Cost: about 6M Gemini input tokens per pass over 500 images.
  - Failures in "dense areas where multiple symbols are closely packed" and weak fine-grained geometric reasoning.
  - [arXiv 2510.03376](https://arxiv.org/html/2510.03376)
- VLM-PL used a vision LLM (Ferret) as a binary verifier of pseudo ground truth for class-incremental detection (secondary summary). — [VLM-PL (IEEE)](https://ieeexplore.ieee.org/document/10678395/)
- Hallucination on yes/no object existence: on the POPE benchmark, embedding the instruction into the image ("Prompt-in-Image") raised Qwen2.5-VL accuracy from 80.2% to 84.3%, reportedly helping with "small, unusual, or partially hidden objects". The paper's title ("Cure or Poison?") signals the effect is model-dependent. — [arXiv 2508.01678](https://arxiv.org/pdf/2508.01678)
- API frontier models are weak at box grounding even when strong at VQA: RefCOCO-avg is 66.8 for GPT-5 (high) and 74.6 for Gemini 2.5 Pro, vs 91.9 for Qwen3-VL-235B. — [Qwen3-VL report](https://arxiv.org/pdf/2511.21631)

**Throughput and cost**
- vLLM on one A100 40GB (bf16, ShareGPT4V images):
  - Qwen2.5-VL-7B: 13.24 req/s at unlimited rate (TTFT 130 ms) and 20.89 req/s at concurrency 50.
  - InternVL3.5-2B: 26.3–31.9 req/s. InternVL3.5-4B: 15.6 req/s. MiniCPM-V-4: 17.6 req/s.
  - [vLLM issue #24728](https://github.com/vllm-project/vllm/issues/24728)
- Gemini 2.5 Flash API: $0.30 per 1M input tokens (text, image or video) and $2.50 per 1M output tokens, including thinking tokens. — [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)

### Inferences
- **Verifier design to copy from SAM 3:**
  1. Frame verification as multiple choice and read the choice-token logits, which gives a continuous, rankable score instead of free-text yes/no.
  2. For per-object checks, show two views: context with the box drawn, plus a zoomed crop with the mask or box. This targets small objects directly.
  3. For exhaustivity, draw all current boxes in red on the full frame and ask whether any instance is unlabeled.
  4. Use the "zero boxes" EV query as an image-level presence gate that suppresses hallucinated detections for absent concepts.
- The SAM 3 verifiers were fine-tuned on about 10M in-domain decisions. Zero-shot general VLMs are much weaker and noisier, as the DART and industrial-diagram gains show. A realistic VIAT path is therefore: start with an off-the-shelf local VLM (Apache-2.0 Qwen3-VL-8B or InternVL3.5), log every human accept/reject in the UI, and periodically fine-tune a small verifier on those logs. The data engine and self-training loops are covered by another researcher.
- Cost estimates (arithmetic from the figures above, not a measured benchmark):
  - At about 13–21 req/s, a local 7B VLM verifies about 10,000 detection crops in roughly 8–13 minutes on one A100, before accounting for crop preparation and longer outputs. Verifying once per *track* instead of per frame cuts this by the average track length.
  - Gemini 2.5 Flash at about 6M input tokens per 500 images is roughly $1.80 per verification pass per 500 images.
- EV-style exhaustivity checks catch misses that per-box verification cannot. The industrial-diagram study's "remove already-detected objects, then ask what remains" strategy suits VIAT: mask out labeled boxes and ask the VLM to find remaining instances.
- In dense VisDrone crowds, whole-frame exhaustivity judgments will be unreliable: both the SAM 3 EV verifier's weakest domains and the industrial study's failures involve crowded or dense scenes. Run exhaustivity checks per tile (e.g., SAHI tiles) rather than per full frame (inference).

### Gaps
- SAM 3's MV/EV verifier weights are not public as far as I found. The repo documents SAM 3 checkpoints, and I found no verifier release, so VIAT cannot use them directly (unconfirmed).
- I found no peer-reviewed benchmark of local VLMs (Qwen3-VL, InternVL3.5, Molmo) as per-detection yes/no verifiers on natural or aerial detection data. None reports precision/recall of accept/reject, calibration, or a tiny-object breakdown. The crop-vs-drawn-box-on-full-image question is answered only indirectly: SAM 3 uses both views, and the industrial study found combined cues best.
- No measured throughput for SAM 3-style two-view verification prompts, and no VLM throughput on H100 or consumer GPUs (only the A100 numbers above).

---

## Q5. Lightweight crop verifiers: CLIP / SigLIP 2 / DINOv2 / DINOv3 embeddings with kNN or linear probes; background rejection; track-level aggregation

### Takeaway
Embedding-based verifiers are the cheap first-stage filter, orders of magnitude cheaper than a VLM. The strongest recent evidence is indirect but encouraging: a **DINOv2 memory bank of a few reference masks matched against SAM proposals** reaches **36.8 nAP on COCO few-shot detection with no training**. SAM 3's own video logic, the **Masklet Detection Score with a 15-frame confirmation delay**, is a working template for **verifying a whole track instead of every frame**. No head-to-head benchmark of CLIP vs SigLIP 2 vs DINOv3 kNN as open-vocabulary false-positive filters was found.

### Cited Findings
- Crop context (older classic): ViLD encodes region crops with CLIP and ensembles embeddings from 1× and 1.5×-enlarged crops, because tight crops lose context. — [ViLD (ICLR 2022)](https://arxiv.org/pdf/2104.13921)
- SigLIP 2 (Google, Feb 2025; Apache-2.0) adds decoder and self-distillation/masked-prediction losses for "improved … localization, and dense features". It also has a NaFlex variant that keeps native aspect ratio and variable resolution, which suits odd-shaped crops. — [SigLIP 2 paper](https://arxiv.org/abs/2502.14786); [HF siglip2-so400m](https://huggingface.co/google/siglip2-so400m-patch14-384)
- DINOv3 (Meta, Aug 2025) is released under the custom "DINOv3 License" with gated Hugging Face access. — [HF facebook/dinov3-vitl16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m); [repo](https://github.com/facebookresearch/dinov3)
- "No time to train!" (Jul 2025): DINOv2 features of reference masks go into a memory bank, are aggregated into class prototypes and are matched to SAM proposals by "semantic-aware soft merging". Results: COCO FSOD 36.8 nAP, VOC few-shot 71.2 nAP50, cross-domain FSOD 22.4 nAP, all state of the art among training-free methods. — [arXiv 2507.02798](https://arxiv.org/abs/2507.02798)
- ConceptBank (Feb 2026) builds class-wise visual prototypes from target-domain evidence to recalibrate SAM 3 prompts under drift, including remote sensing. — [arXiv 2602.06333](https://arxiv.org/abs/2602.06333)
- The SAM 3 MV verifier renders a zoomed-out view with a box and a zoomed-in view with the mask, "to better visualize small objects". — [SAM 3 paper, Table 21](https://arxiv.org/pdf/2511.16719)
- Track-level temporal verification in SAM 3 video:
  - Output is delayed by T = 15 frames, about 0.5 s at 30 FPS, to validate candidates.
  - A new masklet is removed unless its Masklet Detection Score over the window is at least 0. That means it must match a detection in at least half of the window's frames, which "helps reject some false positive detections".
  - Confirmed masklets whose lifetime score drops below 0 are suppressed. Example: an object entering the frame edge that is never confirmed.
  - Every 16 frames the tracker is re-prompted from detections when detection-to-track IoU ≥ 0.8 and both scores ≥ 0.8.
  - [SAM 3 paper, §C.3](https://arxiv.org/pdf/2511.16719)
- Cheap models trained on auto-labels already approach human-label quality. YOLO11s trained on YOLO-World labels at α = 0.2 reached 0.768 mAP50 on VOC vs 0.817 with human labels, and 0.538 vs 0.588 on COCO. — [Voxel51 auto-labeling paper](https://arxiv.org/abs/2506.02359)

### Inferences
- **Suggested cheap-verifier stack for VIAT** (design inference):
  1. Embed each detection crop with SigLIP 2 (Apache-2.0; text-alignable, so class names give a zero-shot fallback) or DINOv2/DINOv3 (better for fine-grained visual similarity). Use a 1.0× crop and a 1.5× context crop, following the ViLD precedent.
  2. Keep per-class banks of human-accepted and human-rejected crops. Rejected crops cover shadows, reflections, background and wrong-class confusers.
  3. Score each new detection by kNN margin (similarity to accepted minus similarity to rejected). Add a "background / not an object" pseudo-class built from rejected crops for open-set rejection.
  4. Send only low-margin or disagreeing detections to the VLM (Q4), then to the human.
- **Track-level aggregation**: compute the embedding score on K sampled crops per track (e.g., sharpest or largest, and spread in time), aggregate by mean or median, and accept or reject the whole track. SAM 3's MDS rule (matched in at least half of a window's frames) is a ready-made heuristic for temporal consistency. Track repair itself belongs to another researcher.

### Gaps
- No 2024–2026 paper was found that quantifies kNN or linear-probe crop verification (CLIP/SigLIP 2/DINOv3) as a false-positive filter for open-vocabulary detectors: precision gain vs recall loss, with how many labeled examples. The expected gains above are extrapolations from few-shot detection results.
- DINOv3 license terms (commercial scope) were not read in full, only identified as the custom gated "dinov3-license".

---

## Q6. Confidence calibration, threshold selection, box tightening and shadow/reflection false positives

### Takeaway
Open-vocabulary detector scores are **not comparable across classes, prompts or object sizes**. Small objects get about 3× lower scores (Grounding DINO: 0.18 vs 0.52). Generic prompts get about 2× lower scores than specific ones. Global high thresholds therefore silently delete small objects. The Voxel51 study found that **auto-label F1 peaks at a low-to-mid threshold (average α ≈ 0.33)**, the best downstream training labels came from **α = 0.2** (YOLO-World), and **high thresholds collapse recall**. SAM 3 is explicitly calibrated for a fixed **0.5** threshold, and its **presence score** can be further improved by a verifier (+7.2 cgF1). MLLM detectors have no scores at all. Mask-derived boxes and SAM-based refinement tighten boxes. Shadows cost SAM 3 about 2–4 mAP on aerial vehicles and windmills.

### Cited Findings
- Scale and semantic bias, from "Confidence Scores in Open-Vocabulary Detection Are a Biased Mixture of Scale and Semantics" (arXiv, Jul 13, 2026):
  - log(area) correlates with confidence at r ∈ [0.36, 0.64] across Grounding DINO-SwinT, OWL-ViT-B/32 and YOLO-World-S.
  - Grounding DINO: small objects average 0.180 confidence vs 0.520 for large.
  - Generic queries ("an object") score 0.194 vs 0.403 for the most specific.
  - A parameter-free, variance-matched temperature scaling gives +19.6% small-object Recall@10 and +4.1% overall Recall@10, but costs −0.025 AP.
  - No code was mentioned.
  - [arXiv 2607.10993](https://arxiv.org/html/2607.10993v1)
- ProCal (arXiv, Jul 2026) is an inference-time "proposal calibration" that combines a localization-aware foreground score with a background-suppression score. Details beyond the abstract were not verified. — [arXiv 2607.01759](https://arxiv.org/pdf/2607.01759)
- Threshold selection for auto-labeling (Voxel51; arXiv Jun 2025):
  - Tested YOLO-World, YOLOE and Grounding DINO-T on VOC, COCO, LVIS and BDD. "All F1 peaks occur at mid-to-low α values with an average of 0.33". "We caution against using high confidence thresholds that leave too many objects unlabeled."
  - Grounding DINO has "the greatest sensitivity to changes in α". YOLO-World and YOLOE have wider α ranges near peak F1.
  - Best auto-label F1: VOC 0.785 (YOLO-World at α = 0.5), COCO 0.640, BDD 0.499 (YOLO-World at 0.05), LVIS 0.215 (YOLOE at 0.3).
  - [Voxel51 paper](https://arxiv.org/abs/2506.02359)
- Auto-labeling cost (same paper): 3.45M objects across four datasets in 1.27 GPU-hours on one L40S (about $1.18 at $0.93/h). The human-labeling equivalent is about 6,703 hours, or $124,092 at SageMaker's $0.036 per box. The press release frames this as "up to 95% of human-level performance". — [Voxel51 paper](https://arxiv.org/abs/2506.02359); [PR Newswire](https://www.prnewswire.com/news-releases/voxel51-research-reveals-auto-labeling-achieves-up-to-95-of-human-level-performance-while-cutting-costs-by-100-000x-302473005.html)
- SAM 3 evaluation "uses predictions with confidence above 0.5" to enforce calibration and downstream usability. — [SAM 3 paper (HTML v1)](https://arxiv.org/html/2511.16719v1)
- The EV verifier's presence score "serves as a better calibration of object scores", improving pmF1 as well as IL_MCC (+7.2 cgF1 in total). — [SAM 3 paper](https://arxiv.org/pdf/2511.16719)
- MLLM detectors: Qwen3-VL sets all confidences to 1.0 for mAP. RF100-VL attributes MLLMs' weak mAP partly to missing confidence scores and NMS. — [Qwen3-VL report](https://arxiv.org/pdf/2511.21631); [RF100-VL](https://arxiv.org/html/2505.20612v1)
- Loose boxes and localization quality: Rex-Omni's COCO F1 falls from 72.0 at IoU 0.5 to 15.9 at IoU 0.95. The SFT model produced oversized "group" boxes in 20.5% of Dense200 predictions. — [Rex-Omni paper](https://arxiv.org/html/2510.12798v1)
- Box tightening with SAM: SAMRefiner (arXiv Feb 2025) uses SAM with multi-prompt excavation to refine coarse masks and boxes. It "can constantly boost the quality of pseudo masks", with up to +10.3% for PointWSSIS at 1% annotations. It also notes that tight-box prompts are sensitive to errors in the coarse mask. — [SAMRefiner](https://arxiv.org/abs/2502.06756)
- The SAM 3 MV verifier explicitly judges mask quality, and the Correction stage lets annotators "add, remove or edit masks". — [SAM 3 paper](https://arxiv.org/pdf/2511.16719)
- Shadows and confusers in aerial imagery (SAM 3): windmills and vehicles lose 2–4 mAP when shadows are present, because the model "cannot decouple object boundaries from cast shadows". Harbor masks bleed into adjacent water and structures (4.00 mAP text-only). — [arXiv 2607.09583](https://arxiv.org/html/2607.09583v1)
- Prompt phrasing trades precision for recall: an "aerial view of" prefix raised Grounding DINO precision from 60.4% to 92.6% while recall fell from 3.9% to 2.3%. — [arXiv 2601.22164](https://arxiv.org/html/2601.22164v1)
- YOLOE-26 and YOLO26 heads are NMS-free (end-to-end), which removes one tunable (NMS IoU) but not the score threshold. — [arXiv 2606.03748](https://arxiv.org/abs/2606.03748); [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe)

### Inferences
- **Thresholding policy for VIAT** (design inference):
  1. Never use one global high threshold. Use per-class and per-size-bin thresholds, fit on a small human-verified sample per project, as a very small calibration set.
  2. Detect at a low threshold (about 0.2–0.35) for recall, then let verifiers remove false positives instead of the threshold.
  3. Apply scale-aware score correction for small objects, following the temperature-scaling result. That may cost some AP, so it belongs in triage ranking rather than final acceptance.
  4. For SAM 3, keep the 0.5 operating point it was calibrated for, but gate image-level presence with a verifier.
- **Loose boxes**: derive boxes from SAM/SAM 3 masks, for example by prompting SAM with the detector box and taking the mask's bounding box. The mask-vs-box IoU then serves as a "looseness" signal for triage; this is not validated.
- **Shadows and reflections**: mask-derived boxes inherit shadow bleed. Mitigations: negative exemplars on shadow regions (SAM 3), rejected-crop banks with a "shadow" pseudo-class (Q5), and a verifier question about whether the box contains only the object or also its shadow (Q4). Note: the VIAT working tree contains an untracked `viat/utils/shadow_filter.py`; its design is outside this subtopic and was not examined.

### Gaps
- No 2024–2026 study was found on per-class threshold fitting or temperature scaling specifically for SAM 3 or YOLOE-26 scores. No localization-quality (IoU-prediction) head study for these models was found either.
- No quantitative study was found on shadow or reflection false-positive rates for open-vocabulary detectors on drone video (VisDrone-like). The only numbers are the 2–4 mAP remote-sensing shadow effect above.
- No quantitative before/after study was found for "SAM mask → box" tightening on detector pseudo-labels (box AP at strict IoU). SAMRefiner reports mask-quality gains in weakly supervised settings.
