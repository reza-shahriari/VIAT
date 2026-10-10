# Detection-driven track birth/death + propagation + identity keeping for fully automatic video annotation

Scope note: this file covers pipeline architecture and association logic only. Tracker memory/drift, offline tracklet stitching and interpolation, detector error reduction, VLM verification, review triage, data engines and the tools survey are covered by other researchers; they appear here only where they intersect. "Code-verified" means the fact was read directly from source code, not from a paper or README. The SAM 3 and SAM 3.1 source was read from the copy vendored in this repo at `/home/hunter/Vision/VIAT/sam3_src/`, which includes the SAM 3.1 release. Citations point to the matching upstream files in facebookresearch/sam3; line-level details were checked against the vendored copy only.

---

## Q1. SAM 3 / SAM 3.1 video mode: how the detector and tracker interact, the APIs, wrappers, speed, limits and results

### Takeaway
SAM 3's video mode is a hand-engineered tracking-by-detection loop wrapped around a SAM 2-style mask tracker:
- **Detection and association:** the DETR-style concept detector runs on every frame. Detections are matched to propagated masklets by mask IoU. Unmatched high-score detections spawn new masklets.
- **Output hold-back:** outputs are delayed by a 15-frame "hotstart" buffer, so spurious or duplicate masklets can be deleted before they are ever emitted.
- **Reconditioning:** every 16 frames, high-confidence detections overwrite the tracker's memory.
- **No retirement:** after hotstart, masklets are never deleted, only hidden.

This lifecycle is excellent for offline annotation of a moderate number of objects. It is weak for long, dense, small-object drone video: cost and memory grow with the number of objects ever tracked, SAM 3.1 caps objects at 16 by default, and there is no appearance re-ID. The detector cannot be swapped out. On a July 2026 UAV benchmark, native SAM 3 scored 25.6 HOTA against 55.4 for an external "detect every 25 frames + short-segment SAM 3 propagation" pipeline.

### Cited Findings

**Paper-level description (SAM 3, arXiv 2511.16719, Nov 2025)**
- The paper formalizes the per-frame video step as: propagate the previous masklets; detect on the current frame with the prompt; then match and update. It uses "a simple IoU based matching function", and "new masklets are spawned for all newly detected objects that are not matched." — [SAM 3 paper (HTML)](https://arxiv.org/html/2511.16719)
- A "masklet detection score" measures how consistently a masklet has been matched to detections within a temporal window, and masklets below a threshold are suppressed. The tracker is also "periodically re-prompt[ed] with high-confidence detection masks", replacing its own predictions. The paper's appendix §C.3 lists "Removal of Unconfirmed Masklets", "Removal of Duplicate Masklets" and "Masklet Suppression". — [SAM 3 paper](https://arxiv.org/html/2511.16719)
- The detector runs on every frame to find new instances, and the tracker runs per object, so cost grows with the number of masklets. — [SAM 3 paper](https://arxiv.org/html/2511.16719)
- Speed: "On an H200 GPU, SAM 3 runs in 30 ms for a single image with 100+ detected objects. In video, the inference latency scales with the number of objects, sustaining near real-time performance for ∼5 concurrent objects." No VRAM figures are given. — [SAM 3 paper](https://arxiv.org/html/2511.16719); same claim in the [Meta blog](https://ai.meta.com/blog/segment-anything-model-3/)

**Exact algorithm (code-verified, `Sam3VideoBase._det_track_one_frame` → `run_tracker_update_planning_phase`)** — [sam3_video_base.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_video_base.py)
- **Step 1, detect.** The backbone and detector run on frame t. Detections below `score_threshold_detection` are dropped, then NMS is applied at `det_nms_thresh`. The shared backbone features are cached for the tracker.
- **Step 2, propagate.** All existing masklets are propagated to frame t (a SAM 2-style step, without memory encoding yet).
- **Step 3, associate** (`_associate_det_trk_compilable`). Compute the N×M mask IoU between detection masks and tracker masks at low resolution. SAM 3.1 can use IoM (intersection over minimum) for reconditioning.
  - A track counts as matched if any detection has IoU ≥ `trk_assoc_iou_thresh` (0.5). Non-empty tracks that are not matched are "unmatched".
  - A detection is new only if its score ≥ `new_det_thresh` **and** its IoU is below `assoc_iou_thresh` (loose, 0.1) with **every** track. Many-to-one is allowed: any detection that overlaps a track by ≥ 0.1 is treated as already explained, which is the main duplicate guard.
  - Every track with IoU ≥ 0.1 to a detection is recorded under that detection (`det_to_matched_trk_obj_ids`). This list feeds keep-alive counters and duplicate detection.
  - Hungarian one-to-one matching (`o2o_matching_masklets_enable`) exists in code but is asserted off ("Temporarily disabled").
- **Object cap.** If the existing count plus new detections exceeds `max_num_objects`, only the highest-scoring new detections are kept; the rest are silently dropped with a log warning. New IDs are `max_obj_id + 1, …`, so IDs are never reused.
- **Hotstart** (`_process_hotstart`):
  - A new object gets `trk_keep_alive = init_trk_keep_alive`. Each matched frame adds +1 (capped at max); each unmatched frame subtracts 1 (floored at min).
  - An object is **removed** if it first appeared within the last `hotstart_delay` frames and has accumulated ≥ `hotstart_unmatch_thresh` unmatched frames.
  - When one detection matches ≥ 2 masklets, the later-born masklet accrues "overlap" frames against the earlier one. It is **removed** once that count reaches `hotstart_dup_thresh` while it is still within its hotstart window.
  - If `suppress_unmatched_only_within_hotstart=False`, masklets with keep-alive ≤ 0 are **hidden** (not deleted) on that frame.
- **Reconditioning** (`_recondition_masklets`):
  - Every `recondition_every_nth_frame` frames, each track with a unique detection is re-seeded with that detection's mask via `tracker.add_new_mask`. The detection must have score ≥ 0.8 and IoU ≥ 0.8 (SAM 3.1: IoM ≥ 0.5); ambiguous one-to-many matches are excluded. The tracker's own object score must also be > 0.8.
  - Reconditioned objects' outputs are overridden with the detection masks.
  - Optional trigger (`reconstruction_bbox_iou_thresh > 0`): recondition when the box IoU between the detection and the tracker's mask-derived box falls below a threshold.
- **Occlusion-aware overlap suppression.** For masklet pairs with mask IoU ≥ 0.7, the object that was most recently occluded is set to "no object" before memory encoding. A masklet removed by hotstart is always suppressed if it overlaps any other object.
- **Paper vs. shipped code.** The only keep-alive hiding path in `Sam3VideoBase` (the single `suppressed_obj_ids.add` call) is gated by `not suppress_unmatched_only_within_hotstart`. The shipped SAM 3 (November 2025) builder sets that flag to True, so hiding is off. An established masklet (older than the hotstart window) that stops being matched by detections therefore keeps being output for as long as the tracker mask is non-empty. This differs from the paper's "if a masklet's detection score falls below a threshold, we suppress it." The SAM 3.1 builder sets the flag to False, turning hiding on. — [sam3_video_base.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_video_base.py), [model_builder.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py)
- **Masklet confirmation** (optional, `update_masklet_confirmation_status`). A masklet becomes CONFIRMED after 3 consecutive frames matched to a detection (or created by one). Unconfirmed masklets are hidden. Output looks 2 frames ahead to decide.
- **Boundary suppression** (optional, `_suppress_detections_close_to_boundary`). New detections whose box centre lies within 2.5% of the image border are not spawned.
- **Deletion paths.** Objects are only ever deleted by the hotstart rules or by a user `remove_object`. No age- or emptiness-based retirement was found; `decrease_trk_keep_alive_for_empty_masklets` only lowers the keep-alive counter. — [sam3_video_base.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_video_base.py), [sam3_multiplex_tracking.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_multiplex_tracking.py)
- **Output stage** (`propagate_in_video` / `_postprocess_output`):
  - With hotstart on, outputs are buffered and emitted `hotstart_delay` frames late. IDs removed by hotstart are hidden everywhere they would have appeared.
  - Suppressed and unconfirmed IDs and zero-area masks are dropped.
  - A pixel-wise non-overlap constraint assigns each pixel to the object with the highest tracker score.
  - Boxes are computed from masks and returned as normalized xywh.
  - — [sam3_video_inference.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_video_inference.py)

**Shipped thresholds (code-verified)** — [model_builder.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py), [sam3_multiplex_base.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_multiplex_base.py)

| Parameter | SAM 3 (Nov 2025) `build_sam3_video_predictor` | SAM 3.1 (Mar 2026) `build_sam3_multiplex_video_predictor` |
|---|---|---|
| score_threshold_detection | 0.5 | 0.4 |
| det_nms_thresh | 0.1 (IoU) | 0.1, `det_nms_use_iom=True` |
| assoc_iou_thresh (det "explained" by a track) | 0.1 | 0.1 |
| trk_assoc_iou_thresh (track "matched") | 0.5 (class default) | 0.5 (class default) |
| new_det_thresh (spawn) | 0.7 | 0.65 |
| hotstart_delay / unmatch / dup | 15 / 8 / 8 | 15 / 8 / 8 |
| suppress_unmatched_only_within_hotstart | True | False (keep-alive hiding active everywhere) |
| init / max / min trk_keep_alive | 30 / 30 / −1 | 0 / 8 / −4 (class defaults) |
| recent-occlusion overlap suppression | IoU 0.7 | IoU 0.7 |
| recondition_every_nth_frame | 16 (IoU ≥ 0.8) | 16 (IoM ≥ 0.5) |
| masklet_confirmation (3 consecutive dets) | off | on |
| suppress_det_close_to_boundary | off | on (2.5% margin) |
| max_num_objects | unlimited (10000) | **16 by default in the builder**; the class default is 128 |
| image size | 1008 | 1008 |

- There is also an ablation config with all heuristics disabled: hotstart 0 and no reconditioning. — [model_builder.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py)
- The Hugging Face Transformers port (`Sam3VideoConfig`) uses the same SAM 3 defaults (0.5 / 0.1 / 0.1 / 0.5 / 0.7, hotstart 15/8/8, keep-alive 30/30/−1, occlusion 0.7, recondition every 16, `high_conf_thresh=0.8`, `high_iou_thresh=0.8`, `max_num_objects=10000`). It adds `recondition_on_trk_masks=True` ("use tracked masks … detector serves as validation signal"). — [HF Transformers SAM3 Video docs](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_video)

**Official video API** — [facebookresearch/sam3 README](https://github.com/facebookresearch/sam3), [sam3_base_predictor.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_base_predictor.py), [model_builder.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py)
- **Builders:** `build_sam3_video_predictor()`, `build_sam3_multiplex_video_predictor()`, and `build_sam3_predictor(version="sam3"|"sam3.1")`. Checkpoints auto-download from Hugging Face after access is granted.
- **Request types for `handle_request(...)`:**
  - `start_session`: `resource_path` is a JPEG folder or MP4; options `offload_video_to_cpu` and `offload_state_to_cpu`.
  - `add_prompt`: takes `frame_index` plus one of `text`; `bounding_boxes` + `bounding_box_labels` (exemplar prompts for the concept); or `points` + `point_labels` + `obj_id` (a SAM 2-style instance prompt to refine an existing object or add a new one). Also takes `output_prob_thresh` (default 0.5).
  - `remove_object`, `reset_session`, `cancel_propagation`, `close_session`.
- **Streaming:** `handle_stream_request({"type":"propagate_in_video", "propagation_direction": "both"|"forward"|"backward", "start_frame_index", "max_frame_num_to_track"})`.
- The same API serves SAM 3 and SAM 3.1.
- **"both" direction semantics (code-verified).** It runs forward from the prompt frame to the end, then backward from the prompt frame to frame 0. After refinement edits, only the edited objects are re-propagated (`propagation_partial`) and merged with the cached predictions. — [sam3_base_predictor.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_base_predictor.py), [sam3_video_inference.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_video_inference.py)
- **Multi-GPU (code-verified).** The detector shards frames round-robin across GPUs. Masklets are distributed across GPUs, with GPU 0 planning births and deaths and broadcasting the plan. The multi-GPU predictor takes a `gpus_to_use` argument. — [sam3_video_base.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_video_base.py), [sam3_video_predictor.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_video_predictor.py)
- **Multi-concept use.** The official 3.1 notebook resets the session before switching to another text prompt. — [sam3.1_video_predictor_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/sam3.1_video_predictor_example.ipynb)
- By contrast, HF `Sam3VideoProcessor.add_text_prompt` accepts a list of prompts, detects all of them in one pass, and returns `prompt_to_obj_ids`. — [HF docs](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_video)
- The paper warns: "All prompts must be consistent in their category definition, or the model's behavior is undefined." — [SAM 3 paper](https://arxiv.org/html/2511.16719)
- **HF streaming mode caveat.** Streaming inference "disables hotstart heuristics that remove unmatched and duplicate objects, as these require access to future frames … may result in more false positive detections and duplicate object tracks". Custom resolutions (e.g. 560) are possible but "may degrade accuracy"; the model is meant for 1008 px. — [HF docs](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_video)

**SAM 3.1 "Object Multiplex" (27 March 2026)**
- It groups objects into fixed-capacity buckets and processes them jointly instead of running a per-object forward pass.
- Reported gains: "~7x speedup at 128 objects on a single H100" versus SAM 3, less CPU–GPU synchronization in detection–tracker association, and batched postprocessing.
- Accuracy is "mixed" on SA-Co/VEval, with YT-Temporal-1B up 2.1 cgF1 and VOS improved on 6 of 7 benchmarks (MOSEv2 +2.0). — [RELEASE_SAM3p1.md](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
- Meta: "up to 16 objects in a single forward pass"; throughput for a "medium number of objects" rises from 16 to 32 FPS on one H100. — [Meta AI blog](https://ai.meta.com/blog/segment-anything-model-3/)
- Secondary sources, not verified against Meta: 1 object goes from 26.5 to 33.8 FPS, and 128 objects from 1.6 to 11.5 FPS on H100. — [Labellerr blog](https://www.labellerr.com/blog/sam-3-vs-sam-3-1-performance-comparison/), [Substack summary](https://farukalamai.substack.com/p/sam-31-is-here-7-faster-at-multi), [note.com analysis](https://note.com/ai_driven/n/n1784eaf09356?hl=en)

**Reported video results (test split unless noted)** — [RELEASE_SAM3p1.md](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md), [SAM 3 paper](https://arxiv.org/html/2511.16719)
- **SA-Co/VEval (cgF1 / pHOTA):**
  - SAM 3: SA-V 30.3/58.0, YT-Temporal-1B 50.8/69.9, SmartGlasses 36.4/63.6.
  - SAM 3.1: 30.5/58.7, 52.9/70.7, 36.3/64.4.
- **Public benchmarks, SAM 3 → SAM 3.1:** LVVIS test mAP 36.3 → 34.3, BURST test HOTA 44.5 → 43.3, YTVIS21 val mAP 57.4 → 56.6, OVIS val mAP 60.5 → 61.5.
- **VOS, SAM 3 → SAM 3.1:** MOSEv2 J&F 60.3 → 62.3, DAVIS17 92.2 → 92.7, SA-V test 84.4 → 85.1.
- **Baselines on SA-Co/VEval SA-V (cgF1 / pHOTA):** SAM 3 Detector + T-by-D (the same detector with SAM 3's tracker replaced by a tracking-by-detection association module) 25.7/55.7; LLMDet + SAM 3 Tracker 2.3/30.1; GLEE (one noun phrase at a time) 0.1/11.8; full SAM 3 30.3/58.0. — [SAM 3 paper](https://arxiv.org/html/2511.16719)

**Ultralytics wrapper**
- `SAM3VideoSemanticPredictor` does text-prompted video concept tracking. Example: `predictor(source=video, text=["person","bicycle"], stream=True)` with `conf=0.25` and `half`/`quantize` options.
- `SAM3VideoPredictor` takes box/visual prompts.
- Weights (`sam3.pt`) are "not automatically downloaded"; users must request access on Hugging Face.
- SAM 3.1 (`sam3.1_multiplex.pt`) works for images, but "Object Multiplex video tracking is not supported yet, so keep using `sam3.pt`" for video. — [Ultralytics SAM 3 docs](https://docs.ultralytics.com/models/sam-3/)
- The integration landed in ultralytics 8.3.237 (PR #22897). — [Ultralytics PR #22897](https://github.com/ultralytics/ultralytics/pull/22897)
- Search-snippet level, not verified in source: the Ultralytics reference docs list `new_det_thresh=0.7`, `hotstart_delay=15`, `hotstart_unmatch_thresh=8` and `hotstart_dup_thresh=8`, i.e. Meta's heuristics. They also note that per-frame image inference assigns fresh IDs every frame, so video predictors must be used for consistent IDs. — [Ultralytics SAM 3 docs](https://docs.ultralytics.com/models/sam-3)
- Known bug (March 2026): the video predictor produced a spurious ID absent from image-mode output on MOT17. It was marked fixed via PR #24249. — [Ultralytics issue #23941](https://github.com/ultralytics/ultralytics/issues/23941)

**Memory and long-video problems**
- The OOM in the video notebook on longer videos is attributed to `_ALL_INFERENCE_STATES` growing during `propagate_in_video`. `async_loading_frames=True` and `offload_video_to_cpu=True` help, but "not enough for longer videos". — [sam3 issue #169](https://github.com/facebookresearch/sam3/issues/169); see also [issue #305 "Not all memory released in video predictions"](https://github.com/facebookresearch/sam3/issues/305)
- SAM 3.1 hit OOM on an RTX 4090 (24 GB) in `propagate_in_video`, where SAM 3 had not (opened 3 April 2026). — [sam3 issue #511](https://github.com/facebookresearch/sam3/issues/511)
- A May 2026 thread aggregates 3.1 problems (device, memory, loading). A community re-implementation ("MuggledSAM") claims 50 fps on an RTX 3090 at 336 px while tracking 4 objects, with no memory buildup. — [sam3 issue #554](https://github.com/facebookresearch/sam3/issues/554)
- AeroTrack paper (UAV-OVVIS, July 2026): "Native SAM3 continuously accumulates memory in long sequences." On a stress video with 80+ vehicles on an RTX 3090 (24 GB), native SAM 3 could complete the run only with a target budget of N ≤ 7 objects, while segment-reset variants handled more than 10× larger budgets at comparable FPS. — [UAV-OVVIS / AeroTrack](https://arxiv.org/html/2607.08075)
- **Aerial accuracy.** On AeroVIS (9 UAV categories, 8,279 trajectories), native SAM 3 scored HOTA 25.61, with DetA 12.29, AssA 56.83 and Mask J 67.18. AeroTrack with Grounding DINO detection plus SAM 3 propagation scored HOTA 55.39 (DetA 46.44, AssA 66.89, Mask J 58.65). The authors attribute native SAM 3's weakness to "target budget, memory growth, and re-detection capability in long UAV videos with dense targets". — [AeroTrack](https://arxiv.org/html/2607.08075)

**License**
- SAM 3 and 3.1 code and weights are under the custom "SAM License" (last updated 19 November 2025).
- It grants a royalty-free license to use, modify and distribute, but prohibits use subject to ITAR or "end uses prohibited by Trade Controls, including those related to military or warfare purposes". It also requires acknowledgement in publications and forbids reverse engineering. — [sam3 LICENSE](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
- The vendored `pyproject.toml` nonetheless carries an "OSI Approved :: MIT License" classifier. — [sam3 pyproject](https://github.com/facebookresearch/sam3/blob/main/pyproject.toml)

### Inferences
- **Architecture.** SAM 3's loop is a textbook tentative → confirmed → hidden lifecycle, with:
  - a loose "explained-by-existing-track" gate (IoU ≥ 0.1) to stop duplicates;
  - a strict "confirmed-by-detection" gate (IoU ≥ 0.5) to feed keep-alive counters;
  - a look-ahead buffer (hotstart) that deletes tentative mistakes retroactively.

  For offline annotation latency is irrelevant, so VIAT should always run offline, hotstart-on mode. It could even lengthen the look-ahead, because hotstart only protects the first 15 frames of each track. This follows from the code and the HF note that streaming disables hotstart.
- **Death is not really implemented.** Masklets are hidden, not retired. Tracker compute and memory therefore scale with the cumulative number of objects ever spawned, not the number currently visible. With SAM 3.1's builder default `max_num_objects=16`, once the cap is reached new arrivals are silently dropped. For drone traffic, where objects continuously enter and leave, this explains the budget and memory failures AeroTrack observed. An annotation pipeline must retire dead tracks, reset or segment the tracker state, or raise the cap and accept the cost.
- **Drifted tracks persist in SAM 3.** With the shipped SAM 3 config, a masklet that drifts onto background after its first 15 frames is never hidden for lack of detection support; only overlap suppression or a user `remove_object` stops it. SAM 3.1's keep-alive hiding (after about 8 unmatched frames, given max 8 and min −4) partly fixes this. Either way, "track output but not supported by detections for n frames" is a cheap, high-value flag for VIAT's review queue.
- **Re-identification** relies entirely on SAM 2-style memory. If an object's mask goes empty and it is later re-detected away from its old mask, it no longer overlaps any track, so the re-detection spawns a new ID. VIAT needs an explicit lost-track bank or offline stitching for re-entries.
- **Late births are never back-filled.** Births are late by construction: score ≥ 0.65–0.7, plus 3-frame confirmation and the 2.5% border margin in 3.1. The "both" direction only propagates from the prompt frame, so a second, per-track backward pass is needed to recover each object's true entry frames.
- **The detector is not pluggable.** Only SAM 3's own concept detector drives births. YOLO, YOLO-World or a fine-tuned drone detector cannot be substituted without writing a custom loop, although SAM 3's tracker can still be driven per object with mask or point prompts.
- **Masks versus box-only association.** Swapping SAM 3's tracker for tracking-by-detection with the same detector cost 4.6 cgF1 and 2.3 pHOTA on SA-V. Mask propagation helps, but the larger lever on aerial data is detection recall: native SAM 3's AeroVIS DetA of 12.3 versus AssA of 56.8 points to missed small objects rather than association as the main failure.
- **Licensing.** The SAM License's military and Trade-Controls exclusion is material for drone and aerial annotation work. SAM 2.1 (Apache-2.0) avoids it.

### Gaps
- The exact rows of the paper's heuristic ablation (Table 38, "Strategies on Temporal Disambiguation") could not be extracted; the HTML fetch was truncated and the PDF exceeded the fetch size. The quantitative value of hotstart, reconditioning and suppression individually is therefore unknown here.
- The paper does not specify which tracking-by-detection association the "SAM 3 Detector + T-by-D" baseline uses.
- No official VRAM-per-object or VRAM-per-frame figures exist for SAM 3 or 3.1. Only anecdotal OOM reports and AeroTrack's budget curve were found.
- I could not verify the Ultralytics `SAM3VideoSemanticPredictor` source (the fetch of `predict.py` did not contain it). Whether it ports every Meta heuristic (confirmation, boundary suppression, reconditioning) is unverified.
- The HF port defaults `recondition_on_trk_masks=True`, whereas Meta's code re-conditions with detection masks. It is not clear whether this is an intentional divergence.

---

## Q2. Grounded-SAM-2 "continuous ID" and tracking-with-new-objects demos

### Takeaway
Grounded-SAM-2's continuous-ID demos use the simplest detect-every-K-frames design:
- Every 20 frames, Grounding DINO boxes are turned into masks by the SAM 2 image predictor.
- Each detection is greedily matched to the propagated masks by mask IoU > 0.8 and inherits that ID; otherwise it gets a new ID.
- SAM 2 is reset and re-seeded with only the currently detected objects, then propagated 20 frames.

Death is therefore implicit (anything not re-detected at a key frame is dropped), re-ID does not exist, and the strict 0.8 IoU makes ID switches likely under shape change. The maintainers themselves describe the feature as unstable. It is useful as a baseline and a template, not as a production labeler.

### Cited Findings
- **Scripts and history.**
  - Scripts: `grounded_sam2_tracking_demo_with_continuous_id.py`; `..._gd1.5.py` (Grounding DINO 1.5, API token required); `..._plus.py` (adds reverse tracking "to simply cover the whole lifetime of the object"); and a camera/stream variant.
  - Continuous ID was released 9 August 2024, DINO-X support 2 December 2024, and the dds-cloudapi-sdk V2 update 20 April 2025. SAM 2.1 has been supported since 10 October 2024. — [Grounded-SAM-2 README](https://github.com/IDEA-Research/Grounded-SAM-2)
- **Detector options.** Grounding DINO (local or HF); Grounding DINO 1.5/1.6 and DINO-X (**API-only, paid tokens** via dds-cloudapi-sdk); Florence-2. — [Grounded-SAM-2 README](https://github.com/IDEA-Research/Grounded-SAM-2)
- **Main loop (code-verified from the raw script):**
  - `step = 20`.
  - At each key frame, Grounding DINO runs with `threshold=0.25, text_threshold=0.25`, then `image_predictor.predict(box=input_boxes, multimask_output=False)` produces the masks.
  - `objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)`.
  - Then `video_predictor.reset_state(inference_state)`, `add_new_mask(...)` for each detected object, and `propagate_in_video(..., max_frame_num_to_track=step, start_frame_idx=start_frame_idx)`.
  - `sam2_masks` is set to the last propagated frame's masks.
  - If nothing is detected, empty masks and JSON are saved for the step and the loop continues.
  - — [grounded_sam2_tracking_demo_with_continuous_id.py](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_tracking_demo_with_continuous_id.py)
- **`update_masks` (code-verified).** For each **new detection mask** (empty masks skipped), iterate over the propagated masks. On the first one with binary mask IoU > 0.8, inherit its `instance_id` and `break`. Otherwise `objects_count += 1` and assign a new ID. Results are stored in a dict keyed by ID, so two detections that match the same track overwrite each other. IoU is computed on float binary masks. — [mask_dictionary_model.py](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/utils/mask_dictionary_model.py)
- **"Plus" variant.**
  - Newly appearing objects are reverse-propagated with `reverse=True` and `max_frame_num_to_track=step*2` (40 frames).
  - Reverse results **overwrite** the forward results for that ID, with no IoU check: pixels of that ID are cleared, then the reverse mask is written.
  - Forward association still uses IoU 0.8.
  - — [..._continuous_id_plus.py](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_tracking_demo_with_continuous_id_plus.py)
- **Maintainers' own caveat.** The feature is "still under development and it's not that stable now", and the basic demos "only prompt Grounded SAM 2 in specific frames, which may not be friendly to finding new objects during the whole video". — [Grounded-SAM-2 README](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/README.md)
- **User reports.**
  - A March 2025 issue asks whether continuous-ID consistency has been improved. — [Issue #85](https://github.com/IDEA-Research/Grounded-SAM-2/issues/85)
  - An April 2025 issue reports poor results on driving scenes; with prompt `'car. suv. bus.'` "the model ignores the black suv". No maintainer reply was visible. — [Issue #95](https://github.com/IDEA-Research/Grounded-SAM-2/issues/95)
- **SAM 2 constraint behind the reset pattern.** Only since the 11 December 2024 update does `SAM2VideoPredictor` "support independent per-object inference … adding new objects after tracking starts". It also added `vos_optimized=True` for full-model `torch.compile`. SAM 2/2.1 is Apache 2.0. — [facebookresearch/sam2 README](https://github.com/facebookresearch/sam2)
- **License.** Grounded-SAM-2 is Apache-2.0 (primary), with BSD-3-Clause parts and separate licenses for its Grounding DINO and SAM 2 components. — [Grounded-SAM-2 README](https://github.com/IDEA-Research/Grounded-SAM-2)

### Inferences
- **The reset-and-reseed design is the root of several failure modes:**
  1. A track that the detector misses at a single key frame (1 in 20 frames) dies, and its re-detection 20 frames later gets a new ID, causing fragmentation.
  2. Mask IoU > 0.8 between a fresh SAM-from-box mask and a 20-frame-propagated mask is a very strict test. Deforming, partially occluded or small objects (drone footage) will often fail it and get new IDs.
  3. New objects are discovered up to 19 frames late. The "plus" script's 40-frame reverse pass patches this, but overwrites without checking conflicts.
  4. The memory reset at every key frame discards long-term memory, which helps bound memory but hurts recovery from occlusion.
- Since SAM 2 now supports adding objects mid-propagation (December 2024), a modern variant should keep tracker state, add only unmatched detections, and use a loose overlap test (like SAM 3's IoU ≥ 0.1 or SAM2MOT's coverage test) instead of re-seeding everything.
- The pipeline is detector-agnostic in practice: any box detector can feed the SAM 2 image predictor. For VIAT, YOLO or YOLO-World could replace Grounding DINO directly.

### Gaps
- There are no published quantitative evaluations of the continuous-ID scripts on any MOT/VIS benchmark.
- I did not enumerate all Grounded-SAM-2 issues. Beyond #85 and #95, specific ID-switch or duplicate reports were not verified.

---

## Q3. DEVA (decoupled video segmentation) and Cutie

### Takeaway
DEVA separates a task-specific image-level segmenter from a class-agnostic temporal propagator (XMem-derived):
- Every 5 frames it forms an "in-clip consensus" over 3 frames of image proposals, selected by a small integer program.
- It merges the consensus with the propagated masks by bipartite IoU > 0.5.
- Unmatched consensus segments become new objects; objects not re-associated for a number of consecutive detection rounds are deleted.

It is fully detector-agnostic, which makes it the closest published precedent for "any detector + propagation". Its drawbacks: it is 2023-era (XMem, 480 px), slow (about 4 FPS), amplifies detector false positives through propagation, and is only semi-online (limited look-ahead, no full backward pass). Cutie is a better propagator (CVPR 2024) but has no detection loop of its own; its README points automatic use to DEVA.

### Cited Findings
- **Venue and design.** ICCV 2023 (UIUC and Adobe). Task-specific image-level segmentation plus "class/task-agnostic bi-directional temporal propagation", fused semi-online. — [DEVA repo](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)
- **In-clip consensus:**
  - Segments from the n future frames of a clip are warped back to frame t with the propagation model, forming a proposal set P.
  - An integer program selects v* ∈ {0,1}^|P|, maximizing Σ(Supp_i + Penal_i). Supp_i = v_i·Σ_j IoU_ij over the other proposals with IoU > 0.5. The penalty is α = 0.5 per selected segment. Constraint: no two selected proposals may have IoU > 0.5.
- **Merge and deletion:**
  - Propagated segments and consensus segments are associated by maximum bipartite matching with minimum IoU 0.5. Matched pairs are merged by union; unmatched propagated segments are kept; unmatched consensus segments become new objects.
  - Objects are deleted after L consecutive failed associations (L = 5 in the paper).
  - Overlaps are rendered with smaller segments on top.
  - — [DEVA paper (ar5iv)](https://ar5iv.labs.arxiv.org/html/2309.03903)
- **Modes.** Online uses clip size n = 1. Semi-online uses n = 3 and merges every 5 frames, so the image model runs on about 60% of frames. — [DEVA paper](https://ar5iv.labs.arxiv.org/html/2309.03903)
- **Demo defaults:**
  - Text mode (Grounded-SAM): `detection_every=5`, `num_voting_frames=3`, `temporal_setting='semionline'` (or `online`), `max_missed_detection_count=10`, `max_num_objects=-1`.
  - Automatic (SAM grid) mode: `max_missed_detection_count=5`, `max_num_objects=200`, optional `suppress_small_objects`.
  - Thresholds: `DINO_THRESHOLD=0.35`, `DINO_NMS_THRESHOLD=0.8`, `SAM_NUM_POINTS_PER_SIDE=64`, `SAM_PRED_IOU_THRESHOLD=0.88`, `SAM_OVERLAP_THRESHOLD=0.8`.
  - — [ext_eval_args.py](https://github.com/hkchengrex/Tracking-Anything-with-DEVA/blob/main/deva/ext/ext_eval_args.py)
  - Propagation defaults: `size=480` (shorter side), `chunk_size=-1` (objects processed in parallel), and XMem memory settings `max_mid_term_frames=10`, `min_mid_term_frames=5`, `max_long_term_elements=10000` ("increase if objects disappear for a long time"), `mem_every=5`. — [eval_args.py](https://github.com/hkchengrex/Tracking-Anything-with-DEVA/blob/main/deva/inference/eval_args.py)
- **Documented semantics.** `detection_every`: "a higher number means faster inference but slower responses to new objects". `max_missed_detection_count`: "maximum number of consecutive detections that can be missed before an object is deleted from memory". `max_num_objects`: "new objects are ignored if this is exceeded". — [DEVA DEMO.md](https://github.com/hkchengrex/Tracking-Anything-with-DEVA/blob/main/docs/DEMO.md)
- **Image models demonstrated:** Mask2Former, EntitySeg, ReferFormer, DIS saliency, SAM and Grounding-DINO. The propagator is a modified XMem trained class-agnostically. — [DEVA paper](https://ar5iv.labs.arxiv.org/html/2309.03903)
- **Results:**
  - VIPSeg VPQ 52.2 (Mask2Former Swin-B, semi-online).
  - BURST test OWTA_all 70.5, with the uncommon-class OWTA improved to 44.1; the EntitySeg variant reaches OWTA_uncommon 53.3.
  - DAVIS-2017 J&F 87.6 for the propagation module alone; Ref-YouTubeVOS J&F 66.0.
  - Speed: 4.0 FPS on VIPSeg on an A6000 (propagation 83 ms/frame, Mask2Former-R50 about 200 ms/frame, consensus 226 ms per merge).
  - These numbers were extracted by fetch from ar5iv and should be spot-checked against the PDF. — [DEVA paper](https://ar5iv.labs.arxiv.org/html/2309.03903)
- **Stated limitations.** "Positive detections are amplified temporally due to propagation", and DEVA underperforms end-to-end methods on closed-set benchmarks. — [DEVA repo](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)
- **On aerial data.** DEVA scores HOTA 34.98 on AeroVIS, against 55.39 for the best AeroTrack variant and 25.61 for native SAM 3. — [AeroTrack](https://arxiv.org/html/2607.08075)
- **Cutie.**
  - CVPR 2024 Highlight, MIT license. Object-level memory reading with a query-based "object transformer".
  - +8.7 J&F over XMem on MOSE at comparable running time; +4.2 J&F over DeAOT while being 3× faster. — [Cutie arXiv abstract](https://arxiv.org/abs/2310.12982)
  - Supports adding and removing objects during inference (`scripting_demo_add_del_objects.py`). The README says "for automatic video segmentation/integration with external detectors, see DEVA". — [Cutie repo](https://github.com/hkchengrex/Cutie)
- **Roboflow `trackers` package** (Apache-2.0) includes "McByte (mask-conditioned tracking with SAM/Cutie masks)" alongside SORT, ByteTrack, OC-SORT, BoT-SORT and C-BIoU. — [roboflow/trackers](https://github.com/roboflow/trackers)

### Inferences
- **DEVA's consensus is a principled duplicate and false-positive filter.** A proposal is kept only if it is supported, via IoU > 0.5 after warping, by proposals in other frames of the clip. This is the multi-frame analogue of SAM 3's confirmation rule and is reusable with any detector.
- **Plugging in a modern detector.** SAM 3's image detector, YOLO-World + SAM box→mask, or a fine-tuned YOLO-seg can be DEVA's image model, since the design only needs per-frame instance masks. I found no report of anyone doing this with SAM 3.
- **Deletion horizon.** `max_missed_detection_count` counts detection rounds, so at `detection_every=5` the text-mode default of 10 means about 50 frames before deletion. A re-detection after deletion gets a new ID.
- **Recommendation.** DEVA is best treated as a design reference (consensus plus merge rules) rather than a component to adopt. Its XMem propagator at 480 px is likely weak for small drone objects, and AeroVIS confirms this.

### Gaps
- I could not verify DEVA's repository license (the LICENSE fetch returned 404).
- No published DEVA configuration uses SAM 3 or YOLO-World as the image model.
- There is no full offline or bidirectional DEVA mode in the demo arguments (only `semionline`/`online`); the paper's "bi-directional" refers to past propagation plus near-future image segmentations.

---

## Q4. MASA, SAM2MOT and other SAM 2/SAM 3-based MOT; open-vocabulary MOT — which are practical for offline auto-labeling with arbitrary detectors?

### Takeaway
Two families are relevant:
- **Appearance-association add-ons for any detector.** MASA (CVPR 2024, Apache-2.0) is the practical, released option. It is purely appearance-based and does not propagate masks.
- **"Tracking-by-segmentation" systems** that let a detector prompt SAM 2 and manage the lifecycle: SAM2MOT (AAAI 2026), Seg2Track-SAM2, Seg2Track++, LiAM-SAM (September 2026), AeroTrack (July 2026), and older Det-SAM2 and SAM-Track.

The SAM 2-based systems show large identity gains over ByteTrack with the same detector (DanceTrack HOTA 75.8 vs 53.3) and strong UAVDT results. However, several have no released code (SAM2MOT, AeroTrack "upon acceptance") or unclear code. End-to-end open-vocabulary trackers (OVTrack, OVTR, GLEE, SLAck) are not detector-agnostic and trail SAM-based pipelines on SA-Co and AeroVIS, so they are poor fits for a pluggable auto-labeler.

### Cited Findings
**MASA (CVPR 2024 Highlight)**
- A "universal instance appearance model" learned from SAM-segmented SA-1B images with exhaustive data transformations. It sits as an adapter on frozen detector backbone features and uses bi-softmax nearest-neighbour matching of instance embeddings, with no motion cues. Its track head is 4 conv layers + 1 FC on RoI-Align features. — [MASA paper](https://arxiv.org/html/2406.04221)
- Plug-in detectors demonstrated: Grounding DINO, Detic, YOLOX, CO-DETR, and SAM ("Segment and Track Anything"). Apache-2.0; code released June 2024; weights on Hugging Face. — [MASA repo](https://github.com/siyuanliii/masa)
- **Results:**
  - TAO TETA 34.9 / AssocA 37.6 (Grounding-DINO); SAM-H TETA 34.5, versus fully supervised TETer 34.6.
  - BDD100K MOT with the same detections as ByteTrack: mIDF1 55.6 / IDF1 71.7, versus ByteTrack 54.8 / 70.4.
  - BDD MOTS mIDF1 49.7, versus UNINEXT-H 48.5.
  - About 10× faster than SAM's everything-mode for equal recall on UVO.
  - — [MASA paper](https://arxiv.org/html/2406.04221)
- **Conflicting numbers:** the README reports open-vocabulary MOT at "47.3 TETA (base) and 44.0 AssocA (novel)" — [MASA repo](https://github.com/siyuanliii/masa). OVTR's comparison lists MASA at novel TETA 30.0 and base 36.9 on TAO val — [OVTR paper](https://arxiv.org/html/2503.10616). The protocols and detectors evidently differ.

**SAM2MOT ("Tracking by Segmentation"; arXiv 2504.04519, April 2025; AAAI 2026)**
- **Addition:** high-confidence detections are Hungarian-matched to existing tracks. For each unmatched detection, compute its overlap with the untracked region (the image minus the union of tracked masks). If more than 50% of the box area is untracked, it becomes a new object and its box prompts SAM 2.
- **Removal:** SAM 2 logits are binned by τ_r = 8.0, τ_p = 6.0 and τ_s = 2.0 into reliable, pending, suspicious and lost. A lost object is removed after 25 frames of tolerance.
- **Quality reconstruction:** when a track is "pending" and matched a high-confidence detection, the detection box re-prompts SAM 2.
- **Cross-object interaction:** when two objects' mask IoU exceeds 0.8, the variance of their logits over N = 10 frames identifies the wrongly tracked one, whose memory entries are removed.
- **Components:** Co-DINO-l or Grounding-DINO-l (COCO, no fine-tuning) with SAM 2.1-L.
- **Results:**
  - DanceTrack HOTA 75.8 / IDF1 83.9 (vs MOTIP 73.7/79.4; vs ByteTrack with the same detector 53.3/53.7).
  - UAVDT MOTA 66.1 / IDF1 79.4 at IoU 0.4 (vs DroneMOT 50.1/69.6).
  - BDD100K MOTA 63.0 (below UNINEXT's 67.1) and IDF1 73.7 (vs MASA 71.7).
- **Limitations:** "limitations in FPS" (no number given), and boxes "only cover the visible parts of objects", which mismatches amodal ground truth.
- — [SAM2MOT paper](https://arxiv.org/html/2504.04519v2)
- **Repo:** Apache-2.0, but code is "Incoming: We will release our code soon"; accepted November 2025. — [SAM2MOT repo](https://github.com/TripleJoy/SAM2MOT)

**Seg2Track-SAM2 (arXiv 2509.11772, September 2025; v2 March 2026)**
- **Detector:** detector-agnostic; default YOLOv11 at confidence ≥ 0.5, also tested with RT-DETR and YOLOv26.
- **Birth:** a proposal whose overlap with existing masks is below τ_v (0.6 for cars, 0.85 for pedestrians) starts a new track.
- **Track states:** a SAM 2 IoU score ≤ 0.10 is low; 0.10–0.70 is "Uncertain".
- **Death:** after n_tries = 5 consecutive low-confidence frames.
- **Reinforcement:** only Uncertain tracks are re-prompted, using a Hungarian match on the Euclidean distance between box and mask centres.
- **Memory:** a sliding window of T_w = 16 frames reduces memory by up to 75% with negligible loss.
- **Results:** KITTI MOTS test Car HOTA 74.11 / AssA 78.15; Pedestrian 59.93 / 65.86; MOTS Challenge HOTA 61.14. FPS not given.
- — [Seg2Track-SAM2](https://arxiv.org/html/2509.11772v2); code at [hcmr-lab/Seg2Track-SAM2](https://github.com/hcmr-lab/Seg2Track-SAM2)

**Seg2Track++ (arXiv 2606.03875, 2 June 2026)**
- **Track existence:** a Bernoulli filter keeps a track-existence probability, reinforced for matched tracks by L = exp(α(w_s·s + w_IoU·IoU)). Unmatched tracks decay as exp(−λΔt)(1−o)+o, which is occlusion-aware.
- **Birth:** low overlap plus high confidence.
- **Association:** mask-centroid-distance cost with confidence-aware cost modulation.
- **Components and results:** YOLO11-seg with SAM 2.1-L; KITTI MOTS Car HOTA 74.56, Pedestrian 60.20. No speed reported; code availability unclear.
- — [Seg2Track++](https://arxiv.org/html/2606.03875)

**LiAM-SAM (arXiv 2609.28078, 23 September 2026)**
- A detector-agnostic SAM 2 MOT system that treats MOT as "lifecycle memory integrity".
- Contrastive track initiation, conditioned on nearby existing tracks, prevents faulty or duplicate births. Motion- and geometry-grounded memory correction and adaptive context memory support re-ID after occlusion or re-entry.
- Reports +10.5 HOTA, +17.4 AssA and −96% ID switches over its baseline in association-heavy settings. No code or speed stated.
- — [LiAM-SAM](https://arxiv.org/abs/2609.28078)

**AeroTrack / UAV-OVVIS (arXiv 2607.08075, July 2026) — drone-specific, training-free**
- **Pipeline:** open-vocabulary detection on key frames every Δ = 25 frames (YOLO-World, Grounding DINO or the SAM 3 detector). SAM 2 or SAM 3 propagates masks within each segment [k, k+Δ), and the segmenter's state "is reset at segment boundaries to constrain the effective memory length".
- **Lifecycle-aware ID Association (LIA):** keeps active and lost banks. The score is 0.45·IoU + 0.30·centre-similarity + 0.25·area-consistency, with hard gates on area ratio [ρ_min, ρ_max] and centre distance ≤ γ·√(max area). Lost tracks expire after T_lost.
- **AeroVIS HOTA:** YOLO-World+SAM3 55.82; GDINO+SAM3 55.39; SAM3+SAM3 54.43; GDINO+SAM2 48.42; YOLO-World+SAM2 47.55; DEVA 34.98; native SAM3 25.61; GLEE 24.71; OVFormer 5.94.
- Code and data "will be open-sourced upon acceptance". — [AeroTrack](https://arxiv.org/html/2607.08075)

**Older detection-driven SAM pipelines**
- **Det-SAM2 (arXiv 2411.18977, November 2024).**
  - YOLO boxes serve as SAM 2 prompts; detection runs every 30 frames by default at confidence 0.85, and new categories can be added mid-inference.
  - `max_inference_state_frames=60` and `max_frame_num_to_track=60` give "constant GPU and memory usage" for unlimited-length video. No VRAM or speed numbers.
  - Apache-2.0 + BSD-3-Clause. — [Det-SAM2](https://github.com/motern88/Det-SAM2)
- **SAM-Track / Segment-and-Track-Anything (April 2023; AGPL-3.0).**
  - Key-frame segment-everything (SAM) or Grounding-DINO runs every `sam_gap` frames, with DeAOT propagation.
  - `max_obj_num` covers about 255 objects in about 16 GB. — [SAM-Track repo](https://github.com/z-x-yang/Segment-and-Track-Anything)
  - New-object test (code-verified): `new_obj_mask = (track_mask==0) * seg_mask`. An object is accepted only if `new_obj_area/obj_area` ≥ `min_new_obj_iou` and its area ≥ `min_area`. — [SegTracker.py](https://github.com/z-x-yang/Segment-and-Track-Anything/blob/main/SegTracker.py)
- **SAMOFT (arXiv 2605.09417, May 2026).** SAM masks plus dense optical flow refine Kalman predictions, add mask-centroid matching for low-confidence or occluded detections, apply flow-statistics correction, and use cluster-aware ReID. Tested on DanceTrack and MOTChallenge; numbers and code not verified. — [SAMOFT](https://arxiv.org/abs/2605.09417)
- **SAM2Auto (arXiv 2506.07850, June 2025).** Claims "the first fully automated annotation pipeline for video datasets requiring no human intervention", using SMART-OD detection plus FLASH video instance segmentation and "statistical approaches" to bridge detection gaps. Quantitative results not verified. — [SAM2Auto](https://arxiv.org/abs/2506.07850)

**End-to-end and learned open-vocabulary MOT**
- **OVTR** (arXiv 2503.10616, March 2025; code at github.com/jinyanglii/OVTR):
  - An end-to-end transformer where queries are born when confidence exceeds τ_det and end after T_miss missed frames.
  - TAO val novel TETA 31.4 and base 36.6, versus MASA 30.0/36.9 and OVTrack 27.8/35.5.
  - Speed: OVTR 3.4 FPS, OVTR-Lite 12.4 FPS, OVTrack 3.1 FPS. — [OVTR](https://arxiv.org/html/2503.10616)
- **OVTrack** (CVPR 2023). — [OVTrack paper](https://arxiv.org/pdf/2304.08408)
- **SLAck** (ECCV 2024). Fuses semantic, location and appearance cues early through a spatio-temporal object graph and claims state-of-the-art novel-class AssocA on TAO TETA. Code at github.com/siyuanliii/SLAck. — [SLAck paper](https://arxiv.org/pdf/2409.11235)
- **GLEE** (CVPR 2024 Highlight, MIT license). — [GLEE repo](https://github.com/FoundationVision/GLEE)
  - It scores 0.1 cgF1 / 11.8 pHOTA on SA-Co SA-V — [SAM 3 paper](https://arxiv.org/html/2511.16719) — and 24.71 HOTA on AeroVIS — [AeroTrack](https://arxiv.org/html/2607.08075).
  - TROY-VIS (WACV 2025) is reported as 20× faster than GLEE-Lite with comparable accuracy on BURST and LV-VIS. — [TROY-VIS paper](https://openaccess.thecvf.com/content/WACV2025/papers/Yan_Towards_Real-Time_Open-Vocabulary_Video_Instance_Segmentation_WACV_2025_paper.pdf)

### Inferences
- **Practical today for an offline, arbitrary-detector labeler:**
  1. A custom SAM 2.1/SAM 3-tracker loop that adopts SAM2MOT, Seg2Track or AeroTrack lifecycle rules. The algorithms are fully described even where code is missing.
  2. MASA as an optional appearance-embedding source for a lost-track bank and re-ID.
  3. Det-SAM2 as a code reference for bounded-memory SAM 2 streaming.
- End-to-end open-vocabulary trackers need their own training, cannot use VIAT's fine-tuned YOLO models, and perform poorly on SA-Co and aerial data, so they are not recommended.
- **Common structure across the SAM 2-based systems.** They independently converge on:
  - a birth gate combining high detector confidence with low overlap with existing masks (SAM 3 IoU < 0.1; SAM2MOT > 50% untracked; Seg2Track < 0.6/0.85; SAM-Track coverage ratio);
  - track states from the tracker's own confidence (logits or predicted IoU);
  - re-prompting from matched high-confidence detections when tracker confidence sags;
  - deletion after roughly 5–30 frames of low confidence;
  - bounded memory (sliding window of 16, segment resets every 25, or 60-frame windows).

  That consensus makes a reliable skeleton for VIAT (see Q7).
- **Visible-part boxes.** Boxes derived from propagated masks cover only the visible part (a SAM2MOT limitation). If VIAT's box convention is amodal (as in MOT17-style ground truth), mask-derived boxes will systematically disagree during occlusion. VisDrone-style conventions should be checked, and the detector box could be kept when available.

### Gaps
- **Code status:** SAM2MOT and AeroTrack code are unreleased; Seg2Track++ and LiAM-SAM code status is unknown; the Seg2Track-SAM2 code license was not verified.
- Per-frame detection frequency for SAM2MOT and Seg2Track-SAM2 is not explicitly stated in what I read, though both appear to run the detector every frame.
- Exact TAO numbers for SLAck and GLEE were not retrieved.
- Track-Anything (an interactive SAM + XMem tool) was not researched in this pass; it is click-prompted and not detection-driven.
- AeroTrack's ρ_min, ρ_max, γ, T_lost and match threshold values are not given in the main text.

---

## Q5. Classic tracking-by-detection as an auto-labeler (open-vocab detector + ByteTrack / BoT-SORT / OC-SORT / Deep OC-SORT)

### Takeaway
Box-only tracking-by-detection is the cheapest birth/death/ID machinery. Its overhead is negligible next to the detector. Birth, death and re-ID are controlled by a handful of thresholds (Ultralytics defaults: high 0.25, low 0.1, new track 0.25, buffer 30 frames, match 0.8, optional BoT-SORT ReID). For small, dense drone objects its identity quality is low: YOLO + ByteTrack reaches only about HOTA 38.5 on VisDrone-MOT val, and SAM 2-propagation trackers beat it substantially on identity (DanceTrack +22.5 HOTA with the same detector; UAVDT +16 MOTA over DroneMOT). The pragmatic role for tracking-by-detection in VIAT is a fast first pass or an association layer, with mask propagation used where identity or mask quality matters.

### Cited Findings
- **Ultralytics `botsort.yaml` defaults (verbatim; file header "AGPL-3.0 License"):** `track_high_thresh: 0.25`, `track_low_thresh: 0.1`, `new_track_thresh: 0.25`, `track_buffer: 30`, `match_thresh: 0.8`, `fuse_score: True`, `gmc_method: sparseOptFlow`, `proximity_thresh: 0.5`, `appearance_thresh: 0.8`, `with_reid: False`, `model: auto`. — [botsort.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/botsort.yaml)
- **Ultralytics `bytetrack.yaml` defaults** (same six shared parameters, with comments):
  - `new_track_thresh`: "Start a new track if no match ≥ this; higher reduces false tracks".
  - `track_buffer`: "Frames to keep lost tracks alive; higher handles occlusion, increases ID switches risk".
  - `track_low_thresh`: the second-stage low-score matching, which "balances recovery vs drift". — [bytetrack.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml)
- **Ultralytics tracking docs.** BoT-SORT's native ReID (`with_reid: True`) uses YOLO features or `.pt`/`.onnx`/`.engine` ReID models. `proximity_thresh` is the minimum IoU before appearance is considered; ByteTrack has "no appearance model". `persist=True` must be used when passing consecutive frames of the same stream. — [Ultralytics Track mode](https://docs.ultralytics.com/modes/track/)
- **supervision `ByteTrack`:** `track_activation_threshold=0.25`, `lost_track_buffer=30`, `minimum_matching_threshold=0.8`, `frame_rate=30`, and `minimum_consecutive_frames=1` ("Number of consecutive frames that an object must be tracked before considered valid"). It is deprecated in supervision 0.28.0 and slated for removal in 0.31.0, replaced by `ByteTrackTracker` in the `trackers` package. — [supervision trackers docs](https://supervision.roboflow.com/latest/trackers/); supervision is MIT-licensed — [supervision repo](https://github.com/roboflow/supervision)
- **Roboflow `trackers`** (Apache-2.0): SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU and McByte (mask-conditioned, using SAM/Cutie masks), with `trackers tune` for hyperparameter search. — [roboflow/trackers](https://github.com/roboflow/trackers)
- **BoxMOT** (AGPL-3.0):
  - Trackers: ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, StrongSORT, HybridSORT, BoostTrack, OccluBoost and SFSort. It accepts NumPy detections from "any model" plus BGR frames via `tracker.update()`, with configurable ReID (e.g. `lmbn_n_duke`).
  - Benchmarks: MOT17-ablation best OccluBoost HOTA 71.10 / MOTA 78.50 / IDF1 85.28; SportsMOT val HybridSORT HOTA 81.14. — [BoxMOT](https://github.com/mikel-brostrom/boxmot)
- **BoT-SORT (2022)** on the MOT17 test set: 80.5 MOTA, 80.2 IDF1, 65.0 HOTA (pedestrian classic). — [BoT-SORT paper](https://arxiv.org/pdf/2206.14651)
- **Drone and small-object numbers** (abstract-level via search snippet; the full text was paywalled or blocked by a captcha):
  - OATrack (Sensors 2026) reaches HOTA 38.58 vs 38.48 for official Ultralytics ByteTrack on VisDrone-MOT val, while cutting ID switches by 69.2%, fragmentation by 29.7% and false positives by 68.8%. — [OATrack (DOI)](https://doi.org/10.3390/s26165222)
  - Quick-UAV OCSORT (SIVP 2025) improves MOTA by 0.88, IDF1 by 5.51 and HOTA by 3.01 on VisDrone2019 at 19.58 FPS. — [Quick-UAV OCSORT](https://link.springer.com/article/10.1007/s11760-025-04149-w)
- **Box tracking-by-detection versus mask-propagation tracking, same detector:**
  - SAM2MOT vs ByteTrack on DanceTrack: HOTA 75.8 vs 53.3, IDF1 83.9 vs 53.7. — [SAM2MOT](https://arxiv.org/html/2504.04519v2)
  - SAM 3 vs "SAM 3 Detector + T-by-D" on SA-V: cgF1 30.3 vs 25.7, pHOTA 58.0 vs 55.7. — [SAM 3 paper](https://arxiv.org/html/2511.16719)
  - On UAVDT, SAM2MOT reaches MOTA 66.1 / IDF1 79.4 vs DroneMOT 50.1/69.6. — [SAM2MOT](https://arxiv.org/html/2504.04519v2)

### Inferences
- **How the parameters map to the lifecycle:**
  - Birth = `new_track_thresh`, plus a confirmation count (`minimum_consecutive_frames` in supervision/trackers; ByteTrack's tentative-to-activated rule).
  - Death = `track_buffer` / `lost_track_buffer` in frames. It should scale with the frame rate; supervision does this via `frame_rate`.
  - Re-ID = BoT-SORT `with_reid` (appearance ≥ 0.8 and IoU ≥ 0.5 proximity), or StrongSORT / Deep OC-SORT in BoxMOT, recoverable only within the buffer.
- **Tuning for offline labeling.** Latency is irrelevant, so settings can be more conservative than real-time defaults: a higher `new_track_thresh` with 2–3 frame confirmation, a longer `track_buffer` (60–150 frames) since offline stitching can repair errors later, and ReID enabled for re-entries.
- **Why drone footage is hard.** Tiny boxes make IoU association brittle: a few pixels of motion or a camera jitter takes IoU to zero, hence camera-motion compensation in BoT-SORT and B-IoU or centre-distance gating in UAV variants. Absolute VisDrone-MOT HOTA around 38 with YOLO + ByteTrack implies heavy manual correction. Mask-propagation methods help identity, but only if detection recall is fixed first (see AeroTrack's DetA).
- **Licensing.** Ultralytics and BoxMOT are AGPL-3.0, which matters if VIAT is distributed; supervision (MIT) and trackers (Apache-2.0) are permissive alternatives.

### Gaps
- I found no single, recent, verified table of ByteTrack, BoT-SORT, OC-SORT and Deep OC-SORT on VisDrone-MOT or UAVDT under a common detector. The numbers above are abstract-level and should be verified before quoting.
- No head-to-head labeling-quality study (time-to-correct) comparing box tracking-by-detection with mask propagation for annotation was found.

---

## Q6. Compute trade-off: detector every frame vs every K frames, choosing K, avoiding duplicate tracks, and handling lost-then-re-detected objects

### Takeaway
Systems split into:
- **Every-frame detection** (SAM 3/3.1, most tracking-by-detection, apparently SAM2MOT and Seg2Track), which lets confirmation and keep-alive logic run on dense evidence.
- **Key-frame detection** (DEVA K = 5, Grounded-SAM-2 K = 20, AeroTrack Δ = 25, Det-SAM2 K = 30), which trades latency of new-object discovery for cost.

No surveyed system chooses K adaptively; K is a fixed hyperparameter. The classic alternative is confidence-triggered detection. Duplicate prevention always comes down to an overlap gate between unmatched detections and existing masks, plus a temporal duplicate-removal rule. Lost-then-re-detected objects get a new ID in almost every system unless there is an explicit lost-track bank (AeroTrack's LIA, ByteTrack/BoT-SORT's buffer, SAM2MOT's 25-frame tolerance) or memory-based re-acquisition (SAM 2/SAM 3 memory, LiAM-SAM).

### Cited Findings
- **Every-frame detection.** SAM 3's detector "runs on every frame" and costs about 30 ms per image on H200 with 100+ objects; tracking cost scales per object. — [SAM 3 paper](https://arxiv.org/html/2511.16719)
  - In SAM 3.1 the detector and vision encoder are batched, and objects are multiplexed 16 per bucket. — [RELEASE_SAM3p1.md](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md), [Meta blog](https://ai.meta.com/blog/segment-anything-model-3/)
- **Key-frame K values and their rationale:**
  - DEVA: `detection_every=5` ("higher … faster inference but slower responses to new objects"), with the image model on about 60% of frames in semi-online mode. — [DEVA DEMO.md](https://github.com/hkchengrex/Tracking-Anything-with-DEVA/blob/main/docs/DEMO.md), [DEVA paper](https://ar5iv.labs.arxiv.org/html/2309.03903)
  - Grounded-SAM-2: `step=20`. — [continuous_id script](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_tracking_demo_with_continuous_id.py)
  - AeroTrack: Δ = 25. — [AeroTrack](https://arxiv.org/html/2607.08075)
  - Det-SAM2: every 30 frames at confidence 0.85. — [Det-SAM2](https://github.com/motern88/Det-SAM2)
  - SAM-Track: `sam_gap` ("increasing … improves inference speed"). — [SAM-Track](https://github.com/z-x-yang/Segment-and-Track-Anything)
  - FoundationMotion: refined SAM 2 tracking with detections on every 5th frame. — [FoundationMotion](https://arxiv.org/pdf/2512.10927)
- **Adaptive-K classic.** "Confidence Trigger Detection" (2019) runs the detector at a fixed frequency **or** when a low-confidence condition triggers. — [CTD paper](https://arxiv.org/pdf/1902.00615)
- **Re-prompting on demand.**
  - SAM2MOT re-prompts only when a track is "pending" and matched a high-confidence detection. — [SAM2MOT](https://arxiv.org/html/2504.04519v5)
  - SAM 3 re-conditions every 16 frames. Optionally it re-conditions when the box IoU between the detection and the tracker falls below a threshold with a detection score ≥ 0.8; this trigger is disabled (−1) in the shipped 3.1 config. — [sam3_video_base.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_video_base.py), [model_builder.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py)
- **Duplicate-avoidance gates:**
  - SAM 3: detection NMS at IoU 0.1 (IoM in 3.1). New only if mask IoU < 0.1 with every track. Hotstart removes the later-born masklet after 8 frames of sharing a detection with an earlier one. Per-frame suppression of the most recently occluded masklet when two masks reach IoU ≥ 0.7. — [sam3_video_base.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_video_base.py)
  - SAM2MOT: more than 50% of the box must lie in the untracked region. — [SAM2MOT](https://arxiv.org/html/2504.04519v2)
  - Seg2Track-SAM2: overlap < 0.6 for cars or < 0.85 for pedestrians. — [Seg2Track-SAM2](https://arxiv.org/html/2509.11772v2)
  - SAM-Track: untracked-area fraction ≥ `min_new_obj_iou`. — [SegTracker.py](https://github.com/z-x-yang/Segment-and-Track-Anything/blob/main/SegTracker.py)
  - DEVA: bipartite IoU > 0.5 merge plus a no-overlap constraint (IoU > 0.5) in consensus. — [DEVA paper](https://ar5iv.labs.arxiv.org/html/2309.03903)
  - Grounded-SAM-2: mask IoU > 0.8 to reuse an ID. — [mask_dictionary_model.py](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/utils/mask_dictionary_model.py)
  - LiAM-SAM: contrastive initiation against nearby tracks. — [LiAM-SAM](https://arxiv.org/abs/2609.28078)
- **Lost and re-detected handling:**
  - ByteTrack/BoT-SORT keep lost tracks for `track_buffer=30` frames and can match them via IoU (plus ReID in BoT-SORT). — [Ultralytics botsort.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/botsort.yaml)
  - SAM2MOT has a 25-frame lost tolerance. — [SAM2MOT](https://arxiv.org/html/2504.04519v2)
  - Seg2Track deletes after 5 low-confidence frames. — [Seg2Track-SAM2](https://arxiv.org/html/2509.11772v2)
  - DEVA deletes after 10 (text mode) or 5 (auto mode) missed detection rounds and relies on XMem long-term memory otherwise. — [ext_eval_args.py](https://github.com/hkchengrex/Tracking-Anything-with-DEVA/blob/main/deva/ext/ext_eval_args.py)
  - AeroTrack keeps a lost bank with T_lost expiry and geometric scoring. — [AeroTrack](https://arxiv.org/html/2607.08075)
  - SAM 3 never deletes after hotstart but has no explicit re-ID. — [sam3_video_base.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_video_base.py)
  - Grounded-SAM-2 drops anything not re-detected at the next key frame. — [continuous_id script](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_tracking_demo_with_continuous_id.py)
- **Memory bounding as part of the trade-off:**
  - Seg2Track's 16-frame sliding window gives up to 75% memory reduction with "negligible" loss; "smaller windows provide only modest speedups". — [Seg2Track-SAM2](https://arxiv.org/html/2509.11772v2)
  - AeroTrack's segment reset keeps memory stable and supports more than 10× the target budget of native SAM 3. — [AeroTrack](https://arxiv.org/html/2607.08075)

### Inferences
- **Choosing K for VIAT:**
  - For drone or small-object footage, use K = 1 (every frame) or small K (≤ 5). Detection recall is the bottleneck (native SAM 3 DetA 12.3 on AeroVIS), small objects move out of stale masks quickly, and per-frame evidence is what makes keep-alive, confirmation and hotstart statistics meaningful.
  - For general video with few, large objects, K = 5–25 with propagation in between saves most of the detector cost.
  - An adaptive rule is straightforward: detect on key frames, and additionally when any track's tracker score falls into an "uncertain" band (Seg2Track 0.10–0.70; SAM2MOT "pending"), when a scene cut is detected, or when the frame border region changes (entries). This follows Confidence Trigger Detection and SAM2MOT's on-demand reconstruction.
- **Where the cost actually goes.** Per-frame cost is detector + N_active × tracker. SAM 3.1's multiplexing and track retirement (not present in SAM 3) are the two main levers. Detector-every-frame is cheap relative to mask tracking when N is large: SAM 3 detection takes about 30 ms with 100+ objects, while SAM 3 tracking drops to roughly 1.6 FPS at 128 objects on an H100 (secondary source).
- **Duplicate gating.** The most robust approach combines a loose overlap test (IoU or IoM ≥ 0.1 means "explained") with a coverage test (more than 50% of the detection outside existing masks means new). IoM matters for small objects partly overlapping large ones. Add temporal duplicate removal, since duplicates usually persist for many frames: SAM 3's 8-of-15-frames rule is a good default.
- **Re-detection policy.** A lost-bank with a geometric gate (AeroTrack) plus an appearance embedding (MASA or BoT-SORT-style ReID) should be tried **before** spawning a new ID. Long gaps are better left to the offline stitching stage, which is out of scope here.

### Gaps
- No surveyed paper reports a sweep of K versus accuracy for SAM 2/SAM 3-based pipelines. AeroTrack provides no Δ ablation in its main text.
- No published adaptive-K scheme specifically for SAM 2/SAM 3 video auto-labeling was found.

---

## Q7. Synthesis: comparison table and a recommended detect → associate → propagate → merge algorithm

### Takeaway
No released system satisfies all of VIAT's requirements at once: pluggable detector, many small objects, offline back-fill, bounded memory, explicit re-ID, and a permissive license. Several are clearly out:
- SAM 3/3.1 native is the most refined lifecycle but is SAM-License-bound, cannot swap detectors, and scales poorly to dense, long drone video.
- The best-performing drone design (AeroTrack) has no code yet.
- SAM2MOT's code is unreleased.
- DEVA is dated.

The recommended path is a custom VIAT orchestrator that reimplements SAM 3's code-verified lifecycle heuristics on top of a pluggable detector and a pluggable propagator (SAM 2.1/SAM2-TRT, the SAM 3 tracker, or box tracking-by-detection). It would add three things SAM 3 lacks: segment-based memory bounding (AeroTrack / Seg2Track), a lost-track re-ID bank (AeroTrack LIA, optionally MASA embeddings), and a backward back-fill pass (Grounded-SAM-2 "plus"). It should also emit per-track evidence counters that seed the review queue.

### Cited Findings
- Every row of the table below is sourced in Q1–Q6. Key sources are repeated in the table's last column.
- SAM 2 allows adding new objects after tracking starts, since December 2024. — [sam2 README](https://github.com/facebookresearch/sam2)
- SAM 3's per-frame outputs include `obj_id_to_score` (the detection score of the masklet) and `obj_id_to_tracker_score`, plus the `removed_obj_ids` and `suppressed_obj_ids` sets. These are usable QA signals. — [HF SAM3 Video docs](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_video), [sam3_video_inference.py](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_video_inference.py)

### Inferences

#### Comparison table

| Approach (date, venue) | Birth | Death | Re-ID / duplicates | Detector-agnostic? | Offline / bidirectional | Speed (reported) | License | Code maturity | Key source |
|---|---|---|---|---|---|---|---|---|---|
| **SAM 3 native video** (Nov 2025) | Per-frame SAM 3 detector; spawn if score ≥ 0.7 and mask IoU < 0.1 with all tracks | Only hotstart removal (≥ 8 of first 15 frames unmatched); otherwise never deleted. Keep-alive hiding is disabled in the shipped config, so unsupported (drifted) masklets keep being output | No explicit re-ID (memory only); NMS 0.1; hotstart dup removal (8 frames); occlusion suppression at IoU ≥ 0.7; recondition every 16 | No (SAM 3 concept detector; text/exemplar) | Offline 15-frame look-ahead; forward then backward from prompt frame; no per-track back-fill | ~5 objects near-real-time on H200; H100 ~26.5 FPS (1 obj) → 1.6 FPS (128 obj) [secondary] | SAM License (military/ITAR exclusion) | Official, active; OOM issues on long videos | [code](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_video_base.py), [paper](https://arxiv.org/html/2511.16719) |
| **SAM 3.1 Object Multiplex** (Mar 2026) | Score ≥ 0.65; IoM NMS; no spawn within 2.5% of border; confirm after 3 consecutive dets | Hotstart removal; keep-alive (0/8/−4) hides unmatched tracks anywhere | As SAM 3; recondition IoM ≥ 0.5 | No | As SAM 3 | ~7× SAM 3 at 128 objects on H100; 16→32 FPS medium counts | SAM License | New; OOM reports on 24 GB; builder cap `max_num_objects=16`; Ultralytics video unsupported | [release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md), [blog](https://ai.meta.com/blog/segment-anything-model-3/) |
| **Grounded-SAM-2 continuous ID** (Aug 2024) | Every 20 frames; GDINO 0.25/0.25 → SAM 2 image masks; new if IoU ≤ 0.8 with all propagated masks | Implicit: not re-detected at key frame → dropped (state reset) | None; greedy first match | Yes in practice (GDINO/Florence-2 local; GD1.5/1.6/DINO-X **API-only**) | "Plus": reverse 2×step for new objects (overwrite) | Not reported | Apache-2.0 (+ component licenses) | Demo scripts; maintainers call it unstable | [script](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_tracking_demo_with_continuous_id.py) |
| **DEVA** (ICCV 2023) | In-clip consensus (3 frames, IP with IoU > 0.5 support) every 5 frames; unmatched consensus → new | After 10 (text) / 5 (auto) missed detection rounds | XMem long-term memory; bipartite IoU > 0.5 merge | **Yes** (any image segmenter) | Semi-online look-ahead only | 4.0 FPS (A6000, VIPSeg) | Not verified | Stable 2023 research code | [paper](https://ar5iv.labs.arxiv.org/html/2309.03903), [args](https://github.com/hkchengrex/Tracking-Anything-with-DEVA/blob/main/deva/ext/ext_eval_args.py) |
| **SAM-Track** (Apr 2023) | Key frame every `sam_gap`; untracked-area ratio ≥ `min_new_obj_iou`, area ≥ `min_area` | Not documented | None | SAM-everything / GDINO | No | ~255 objects in ~16 GB | **AGPL-3.0** | Older | [repo](https://github.com/z-x-yang/Segment-and-Track-Anything) |
| **Det-SAM2** (Nov 2024) | YOLO every 30 frames, conf 0.85 | Windowed state (60 frames) | Not documented | YOLO (adaptable) | No | "Constant" memory; no numbers | Apache-2.0 + BSD-3 | Tech-report code | [repo](https://github.com/motern88/Det-SAM2) |
| **SAM2MOT** (AAAI 2026) | Hungarian + > 50% of box in untracked region | Logit states (8/6/2); lost removed after 25 frames | Cross-object interaction (mIoU > 0.8, logit variance over 10 frames); re-prompt when pending | Yes (Co-DINO, GDINO) | Online | "Limited FPS" (undisclosed) | Apache-2.0 | **Code not released** | [paper](https://arxiv.org/html/2504.04519v2) |
| **Seg2Track-SAM2** (Sep 2025) | Overlap < 0.6 / 0.85, conf ≥ 0.5 | 5 consecutive frames with score ≤ 0.10 | Re-prompt "Uncertain" tracks (Hungarian on centre distance); 16-frame window | Yes (YOLOv11, RT-DETR, …) | Online | Not reported | Paper CC BY 4.0; code license unverified | Public repo | [paper](https://arxiv.org/html/2509.11772v2) |
| **Seg2Track++** (Jun 2026) | Low overlap + high confidence | Bernoulli existence probability below threshold | Occlusion-aware decay; MCD + CCM association | Yes (YOLO11-seg) | Online | Not reported | Unknown | Code unclear | [paper](https://arxiv.org/html/2606.03875) |
| **LiAM-SAM** (Sep 2026) | Contrastive initiation (anti-duplicate) | Lifecycle memory | Adaptive context memory for re-entry; −96% ID switches vs baseline | Yes | Online | Not reported | Unknown | Code not stated | [abs](https://arxiv.org/abs/2609.28078) |
| **AeroTrack** (Jul 2026, UAV) | Key-frame OV detection every 25 frames (YOLO-World / GDINO / SAM 3) | Lost bank with T_lost expiry; tracker reset per segment | LIA: 0.45 IoU + 0.30 centre + 0.25 area, gated | **Yes** (and tracker-agnostic: SAM 2/SAM 3) | Segment-wise; no back-fill reported | FPS ≈ native SAM 3 with > 10× target budget (RTX 3090) | Unknown | **Code "upon acceptance"** | [paper](https://arxiv.org/html/2607.08075) |
| **MASA** (CVPR 2024) | Per-frame detector | Tracker-defined | Appearance bi-softmax (no motion) | **Yes** (GDINO, Detic, YOLOX, CO-DETR, SAM) | Online | ~10× faster than SAM-everything (UVO) | Apache-2.0 | Released, weights on HF | [paper](https://arxiv.org/html/2406.04221), [repo](https://github.com/siyuanliii/masa) |
| **OVTR / OVTrack / GLEE / SLAck** (2023–2025) | Learned queries or detections (OVTR: conf > τ_det) | OVTR: T_miss frames | Learned association | No (own detector) | Online | OVTR 3.4 FPS, Lite 12.4; OVTrack 3.1 | GLEE MIT; others not verified | Research code | [OVTR](https://arxiv.org/html/2503.10616) |
| **ByteTrack / BoT-SORT** (Ultralytics, BoxMOT, trackers) | `new_track_thresh` 0.25 (+ optional N-frame confirmation) | `track_buffer` 30 frames | IoU/Kalman; BoT-SORT ReID (appearance 0.8, proximity 0.5) + camera-motion compensation | **Yes** | Online | Negligible vs detector | Ultralytics & BoxMOT **AGPL-3.0**; supervision MIT; trackers Apache-2.0 | Very mature | [botsort.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/botsort.yaml), [BoxMOT](https://github.com/mikel-brostrom/boxmot) |

#### Recommended algorithm skeleton (offline, pluggable detector and propagator)

Default values in brackets are taken from the cited systems; they are starting points to tune on VIAT data, not validated optima.

```text
INPUT  video frames F[0..T), class prompts P, detector D (SAM3-detector | YOLO(-World) | GDINO),
       propagator S (SAM2.1 video predictor | SAM3 tracker | SAM2-TRT | box-only ByteTrack),
       optional appearance embedder E (MASA adapter | BoT-SORT ReID | CLIP crop embedding)
PARAMS K (detect stride) [1 for drone/small objects; 5–25 general]         # SAM3=1, DEVA=5, AeroTrack=25
       τ_keep [0.4–0.5]  τ_new [0.65–0.7]                                  # SAM3.1 / SAM3
       τ_explained (IoU or IoM) [0.1]  τ_confirm_match [0.5]                # SAM3 assoc / trk_assoc
       coverage_new [0.5 of det area outside all track masks]              # SAM2MOT
       N_confirm [3 consecutive matched detection rounds]                  # SAM3.1 confirmation
       H [15 frames hotstart], U [8 unmatched], Dup [8 duplicate frames]   # SAM3 hotstart
       keep_alive init/max/min [0 / 8 / -4]                                # SAM3.1
       recond every R [16] frames if det score ≥ 0.8 and IoU ≥ 0.8 (IoM ≥ 0.5), 1-to-1 only  # SAM3
       overlap_suppress IoU [0.7] → hide the most recently occluded track  # SAM3
       T_lost [25–30 frames at 30 fps; ≈ DEVA 10 missed rounds]            # SAM2MOT / ByteTrack
       segment length L [16–25] for memory reset                           # Seg2Track / AeroTrack
       backfill B [2K–40 frames]                                           # Grounded-SAM-2 plus
       max_active [raise above 16 for drone; retire dead tracks]           # SAM3.1 default is 16

STATE per track: id, class, first_frame, status ∈ {tentative, confirmed, lost, retired},
      keep_alive, consecutive_det, unmatched_frames[], dup_frames{other_id: []},
      masks/boxes/tracker_score/det_score per frame, embedding bank, QA counters

PASS 1 (forward):
for t in 0..T-1:
  M̂ = S.propagate(active_tracks, t)                           # batched / multiplexed
  O = D(F[t], P) if (t % K == 0 or trigger(t)) else ∅          # trigger: any track tracker_score in
                                                               #   uncertain band, scene cut, border change
  O = nms(O, iom=True); O = {o ∈ O : score ≥ τ_keep}; masks(O) via D or SAM-from-box
  A = overlap(O, M̂)                                            # mask IoU and IoM matrices (N×M)
  matched_trk = {j : max_i IoU[i,j] ≥ τ_confirm_match}
  explained   = {i : max_j max(IoU, IoM)[i,j] ≥ τ_explained}
  candidates  = {i ∉ explained : score_i ≥ τ_new and uncovered_frac(i, ∪M̂) ≥ coverage_new}
  for i in candidates:
      j* = best_lost_track(i)  # gate: AeroTrack geometry (IoU/centre/area) + E-embedding sim, within T_lost
      if j*: revive j* (S.add_prompt(mask_i, id=j*)); else spawn tentative id (S.add_prompt(mask_i))
  update keep_alive (+1 matched, −1 unmatched), consecutive_det, unmatched/dup counters
  promote tentative→confirmed when consecutive_det ≥ N_confirm
  hotstart: tracks younger than H with unmatched ≥ U, or sharing a detection with an older
            track for ≥ Dup frames → DELETE retroactively (never emitted)
  if t % R == 0: for unambiguous (det i ↔ track j) high-confidence pairs: S.add_mask(j, mask_i)
  overlap suppression: for track pairs with mask IoU ≥ 0.7, hide the most recently occluded one
  lost/retire: keep_alive ≤ min or tracker mask empty → status=lost (hidden; still in memory briefly);
               lost for > T_lost → retire (free S state; keep embedding/box history in lost bank)
  every L frames: reset S memory, re-seed active tracks from current masks (bounded VRAM)
  emit frame t − H (delayed output buffer), hiding tentative/deleted/suppressed tracks

PASS 2 (backward back-fill):
for each confirmed track with first_frame f0 > 0:
  run S backward from f0 seeded with its first confirmed mask, for ≤ B frames,
  stopping when tracker_score < τ or mask area → 0 or overlap with another track ≥ 0.7
  (optionally require at least one supporting detection within the back-filled span)

PASS 3 (handoff to offline repair and triage — covered by other researchers):
  drop tracks shorter than N_min or with low detection-support ratio; stitch tracklets;
  interpolate gaps; boxes = tight box of mask (or detector box if the amodal convention is required);
  per-track risk score from QA counters: hotstart near-misses, unmatched streaks,
  duplicate-overlap frames, recondition corrections with low IoU (drift), class flips,
  low mean det/tracker score → ranked review queue
```

#### Design notes for VIAT
- **Two execution modes behind one orchestrator:**
  - **Mode A, "SAM 3 native":** few objects, general video, SAM License acceptable. Call `build_sam3_predictor`, run text-prompted propagation offline, raise `max_num_objects`, then run pass 2 and pass 3 externally.
  - **Mode B, "pluggable":** drone video, fine-tuned YOLO detectors, or permissive licensing needed. The pseudocode above with SAM 2.1 (Apache-2.0) or SAM2-TRT as the propagator.
- **Box-only fast path.** When only boxes are needed and objects are tiny, pass 1 with `S` = ByteTrack/BoT-SORT (plus a lost-bank embedding) is the cheapest. SAM 2 box→mask can be run afterwards only for frames or tracks flagged for review. This is supported by the "masks help identity but detection recall dominates" evidence (SAM 3 vs SAM 3 Detector + T-by-D; AeroTrack DetA).
- **Two GPUs.** Put the detector on GPU 0 and the propagator on GPU 1, or use SAM 3's built-in multi-GPU sharding in Mode A, since its masklets are distributed across GPUs by design.

### Gaps
- None of the recommended thresholds has been validated on VisDrone-style footage in a single end-to-end study. AeroTrack is the closest, and its code and full hyperparameters are unreleased.
- No published study measures human correction time (the actual goal) for these pipelines. The benchmark metrics (HOTA, cgF1, pHOTA) are proxies.
