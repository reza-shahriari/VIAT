# Quality control, statistical guarantees and review triage for auto-generated detection and tracking labels

Scope: measuring and guaranteeing auto-label quality, finding label errors automatically, deciding what a human must look at (ideally one decision per track), and auditing when to stop. Researched as of 2026-09-30. "Classic" marks pre-2024 work. Licenses were checked against the GitHub API on 2026-09-30 where a repo exists. Unless stated otherwise, every tool listed runs locally (Python, GPU optional). API-only or paid options are marked **[PAID/API]**.

---

## 1. Threshold-based auto-labeling (TBAL): how many human-validated samples certify an error rate, how thresholds are chosen, and how it extends to detection/tracking

### Takeaway
TBAL trains or uses a model, labels a random human "validation" set, and picks the lowest confidence threshold whose estimated error on validation points above it, plus a safety margin, is ≤ ε. Everything above that threshold is auto-accepted. The error guarantee holds, but it costs validation labels. Theory needs about O(1/ε²) validated points per round, and per class when thresholds are class-wise. In experiments a 200-class problem needed 10k validation labels to auto-label only about 32% of the data at ≤10% error. Better confidence functions (Colander, NeurIPS 2024) and 2025 conformal/PAC follow-ups (PAC Labeling, Conformal Labeling) raise coverage a lot at the same guarantee. No published TBAL variant targets boxes or tracks directly, so the detection and tracking mapping below is my own inference.

### Cited Findings
**Vishwakarma, Lin, Sala, Vinayak, "Promises and Pitfalls of Threshold-based Auto-labeling", NeurIPS 2023 (spotlight), arXiv 2211.12620 (classic, 2023).** Code: https://github.com/harit7/TBAL. The GitHub API shows no license declared.
- The workflow: "validation data obtained from humans is used to find a confidence threshold above which the data is machine-labeled". A "hidden downside of TBAL systems is potentially prohibitive validation data usage" — [arXiv abs](https://arxiv.org/abs/2211.12620)
- The algorithm is iterative (Alg. 1). Each round it trains ĥ_i on the queried labels, estimates a threshold t̂_i, auto-labels unlabeled points with g(ĥ_i,x) ≥ t̂_i, removes those points and the matching validation points, then actively queries a new batch — [paper PDF](https://arxiv.org/pdf/2211.12620)
- Threshold rule (Alg. 2): candidate thresholds are the validation scores. A candidate is discarded if fewer than n₀ validation points lie above it. The chosen threshold is **t̂ = min{t : Ê_a(ĥ,t | X_v) + C₁·σ̂ ≤ ε_a}**, i.e. empirical error above t plus C₁ standard deviations stays within the error tolerance ε_a. Experiments use C₁ = 0.25 — [paper PDF](https://arxiv.org/pdf/2211.12620)
- Theorem 3.2 bounds auto-labeling error by the per-round empirical error + Rademacher-complexity terms + O(√(1/n_v)) terms. "validation data of size O(1/ε_a²) in each round is sufficient to get a O(ε_a) bound on the excess auto-labeling error". A Feller-type lower bound (Lemma 3.3) shows this is **necessary up to log factors**: with too few validation points, "there is a constant probability [≥1/4] of erroneously deciding on a threshold" — [paper PDF](https://arxiv.org/pdf/2211.12620)
- Unit-Ball synthetic set (ε_a = 1%):
  - With no UCB margin (C₁=0) and N_v=100, realized error was 3.10%, violating the 1% target.
  - With C₁=0.25 and N_v=2000, error was 0.54% at 97.2% coverage — [paper PDF, Table 1](https://arxiv.org/pdf/2211.12620)
- IMDB (ε_a = 5%):
  - C₁=0 gave 4.3–4.8% error at 83–93% coverage.
  - C₁=0.25 gave 1.3–2.3% error but only 64–70% coverage (N_v 200–1000) — [paper PDF, Table 2](https://arxiv.org/pdf/2211.12620)
- **Tiny-ImageNet (200 classes, CLIP embeddings, ε_a = 10%):**
  - N_v = 2,000 → 0% coverage.
  - N_v = 10,000 with C₁=0 → 13.4% error (target violated) at 43.8% coverage.
  - N_v = 10,000 with C₁=0.25 → 9.0% error at 32.2% coverage — [paper PDF, Table 3](https://arxiv.org/pdf/2211.12620)
- "not using UCB leads to high auto-labeling error (i.e. not meeting the guarantees) even when there is a sufficient amount of validation data". The UCB keeps error under target "but suffers in coverage" — [paper PDF](https://arxiv.org/pdf/2211.12620)
- On a synthetic example, TBAL labels "almost all of the data accurately (close to 100% coverage) with less than 1% auto-labeling error". Active learning + selective classification labels ≈25%, and using the AL model directly gives ≈50% error. In other words, even a weak model can safely auto-label a confident region — [paper PDF](https://arxiv.org/pdf/2211.12620)

**Vishwakarma et al., "Pearls from Pebbles: Improved Confidence Functions for Auto-labeling" (Colander), NeurIPS 2024, arXiv 2404.16188.**
- Venue: [NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/1d051fb631f104cb2a621451f37676b9-Paper-Conference.pdf).
- Code is reported to live under the first author's GitHub [github.com/harit7](https://github.com/harit7) ("TBAL-Colander-NeurIPS-24" per a search index). I could not resolve that exact repo name via the GitHub API. The license is unknown.
- Objective: maximize coverage subject to auto-labeling error ≤ ε_a, **jointly over the confidence function g and class-wise thresholds t ∈ Tᵏ**. The optimization uses a sigmoid-relaxed surrogate — [arXiv HTML](https://arxiv.org/html/2404.16188v1)
- g(x) = softmax(W₂·tanh(W₁·z(x))), where z concatenates the model's logits and pre-logit features. g is trained on a calibration split D_cal (ν = 25–50% of the validation data). Thresholds are then estimated on the held-out split D_th with Algorithm 2 (C₁ = 0.25) — [arXiv HTML](https://arxiv.org/html/2404.16188v1)
- Results at ε_a = 5% (Squentropy training + Colander): MNIST 97.2% coverage / 4.1% error; CIFAR-10 79.0% / 2.3%; 20 Newsgroups 82.9% / 3.3%; Tiny-ImageNet 66.5% / 0.6%. Temperature scaling on CIFAR-10 reached only 23.2% coverage with 7.3% error. Headline: "up to 60% improvements on coverage over the baselines while maintaining auto-labeling error below 5%" — [arXiv HTML](https://arxiv.org/html/2404.16188v1)

**2025–2026 follow-ups with guarantees**
- **PAC Labeling — Candès, Ilyas, Zrnic, "Probably Approximately Correct Labels", arXiv 2506.10908 (v3 Oct 2025); listed as an ICML 2026 poster.**
  - Code: [github.com/tijana-zrnic/pac-labels](https://github.com/tijana-zrnic/pac-labels/). The GitHub API reports the license as NOASSERTION, i.e. non-standard or unclear — [arXiv](https://arxiv.org/abs/2506.10908), [ICML 2026](https://icml.cc/virtual/2026/poster/63255)
  - Guarantee: "with high probability, the overall labeling error is small" (average loss ≤ ε with probability ≥ 1−α), nonasymptotic, with minimal assumptions on data or model — [arXiv](https://arxiv.org/abs/2506.10908)
  - Method: sample m points with probabilities π_i and build upper confidence bounds on the loss (betting-based or CLT). Set û = min{U_i : L̂^{U_i}(α) > ε}. Items with uncertainty ≥ û go to the expert and the rest are AI-labeled. A "PAC router" learns which of several models to trust per item — [arXiv HTML](https://arxiv.org/html/2506.10908)
  - Results at ε = 0.05 (share of expert labels saved):
    - ImageNet (ResNet-152): **59.6%** saved, realized error 4.73%
    - ImageNet-v2: 39.1%
    - text tasks: 13.8–28.1%
    - PAC router (GPT vs Claude): 41.6%
    - Fixed uncertainty thresholds "often fail or are conservative". No object detection experiments — [arXiv HTML](https://arxiv.org/html/2506.10908)
- **Conformal Labeling — Huang, Liao, Xi, Zeng, Zhao, Wei, "Model-agnostic Selective Labeling with Provable Statistical Guarantees", arXiv 2510.14581 (Oct 2025).** The OpenReview version is titled "Selective Labeling with False Discovery Rate Control" ([OpenReview](https://openreview.net/forum?id=7ExZkCQU5Z)); acceptance status not verified.
  - Guarantee: it controls the **FDR of AI-assigned labels**, i.e. "a predefined fraction of AI-assigned labels is correct on average" — [arXiv](https://arxiv.org/abs/2510.14581)
  - p-value: p_j = (Σ𝟙{S_i < S_{n+j}} + U_j(1 + Σ𝟙{S_i = S_{n+j}}))/(n₀+1). The sums run over the n₀ calibration items the model **mislabeled**, S is an uncertainty score such as 1 − max-softmax, and U_j ~ Uniform(0,1) breaks ties.
  - Selection: Benjamini–Hochberg at an adjusted level.
  - Calibration: a 10% calibration ratio is recommended; 5% still controlled FDR but with more variance — [arXiv HTML](https://arxiv.org/html/2510.14581)
  - Results at target FDR 10%:
    - ImageNet ResNet-34: realized FDR 9.97%, 65.0% of items auto-labeled
    - ImageNet DenseNet-161: 72.7% auto-labeled
    - ImageNet-V2 ResNet-34: 41.9% auto-labeled
    - Without selection, full AI labeling error was 7–47% depending on task. Code was released via an anonymous repo link — [arXiv HTML](https://arxiv.org/html/2510.14581)
- Background: conformal selection "selects candidates whose unobserved outcomes exceed user-specified values while ensuring finite-sample FDR control" and has been used "to filter trust-worthy outcomes from a foundation model (Gui et al., 2024)… while routing others to human experts" — [Optimized Conformal Selection, arXiv 2411.17983](https://arxiv.org/html/2411.17983)
- Aerial detection, pseudo-labels: PECL (Lu et al., NeurIPS 2024) uses a "conformal pseudo-label explorer" and a multi-clue evaluator to pick pseudo-labels for sparsely annotated aerial detection. It is motivated by "imbalanced probabilities/confidences associated with predicted aerial objects" that break fixed thresholds. This is a learned selection policy, not an FDR guarantee — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/47a287298e7887d1c25d4aabb918bd54-Abstract-Conference.html)

### Inferences
- **Choosing the unit of the guarantee is the key design decision for detection.**
  - *Per-box acceptance:* a box is an "error" if it is a false positive, has the wrong class, or IoU < τ with the true object. This maps directly onto TBAL, Conformal Labeling or PAC Labeling, with the calibration set being human-verified auto-boxes sampled at random.
  - *Missed objects (recall)* cannot be certified by per-box acceptance, because a box that doesn't exist can't be accepted. They need a per-frame or per-image "exhaustiveness" check: see SeqCRC's box-count-recall loss in §2 and SAM 3's exhaustivity verification in §5.
  - *Tracks:* the natural unit is the whole track. An error is any ID switch, drift beyond an IoU threshold, a wrong class, or a fragment. Calibrate a track-level score, e.g. min/mean SAM2 predicted IoU, the fraction of frames where the object is present, detector–propagation IoU agreement, or re-ID consistency, against human accept/reject decisions on randomly sampled tracks.
- **Calibration sets must contain enough model errors.** In Conformal Labeling the smallest possible p-value is 1/(n₀+1), where n₀ is the number of calibration items the model got wrong. With a 5% base error rate, 400 calibration items contain only about 20 errors. So small FDR targets need hundreds to thousands of verified items pooled across videos, not per video.
- **Class-wise thresholds multiply the validation cost** (Colander uses thresholds per class). VisDrone-style data has about 10 classes with heavy imbalance, so rare classes will either stay unaccepted (sent to humans) or need pooled thresholds.
- **Exchangeability matters in video.** Calibration items must be drawn at random from the same pool that will be auto-accepted: random frames or tracks across the whole job, not the first minute of a video. Adjacent frames are highly correlated, so treating 30 fps frames as independent overstates the effective sample size.
- **UCB and conservative variants are not optional.** TBAL's C₁=0 runs violated the target even with plenty of data.

### Gaps
- I found no paper applying TBAL, Colander or PAC Labeling to object detection boxes or MOT tracks with measured coverage and error. The detection/track mapping above is inference.
- The Colander repo URL and license, and the licenses of TBAL and pac-labels code, are unverified or non-standard; check before reuse.
- The formal venue of Conformal Labeling (ICLR 2026?) is unverified.

---

## 2. Conformal prediction, conformal risk control (CRC) and Learn-then-Test (LTT) for auto-labeling and object detection: guarantees on box coverage and missed objects; recipes and calibration sizes

### Takeaway
There are now working, post-hoc, model-agnostic tools that give finite-sample guarantees for detectors. SeqCRC (2025, `cods` toolkit) jointly calibrates:
- a confidence threshold that bounds the expected missed-object rate,
- box margins that bound expected localization non-coverage,
- class prediction sets.

CRC gives expectation guarantees with a simple threshold rule. LTT and PAC-style binomial bounds give "with probability 1−δ" guarantees. PAC MOT (2022) extends this to association edges between frames. All of these need exchangeable, randomly sampled human-verified calibration data. Hundreds to a few thousand calibration images or objects are typical.

### Cited Findings
- **Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei, Schuster; arXiv 2208.02814, v4 June 2025).**
  - Guarantees E[L(λ̂)] ≤ α for any monotone bounded loss (bound B), using **λ̂ = inf{λ : (n/(n+1))·R̂_n(λ) + B/(n+1) ≤ α}**. Tight to within O(1/n).
  - Examples include FNR control in multilabel classification and tumor segmentation.
  - Code: [github.com/aangelopoulos/conformal-risk](https://github.com/aangelopoulos/conformal-risk) (MIT per GitHub API) — [arXiv](https://arxiv.org/abs/2208.02814)
- **Learn then Test (Angelopoulos, Bates, Candès, Jordan, Lei; arXiv 2110.01052, 2021–22, classic).**
  - Frames risk control as multiple hypothesis testing: one p-value per candidate λ from the calibration set, then FWER control (Bonferroni, fixed-sequence).
  - Gives **P(R(λ̂) ≤ α) ≥ 1−δ**, including for non-monotone and multiple risks.
  - Examples include multi-label FDR control and instance-segmentation IoU control.
  - Code: [github.com/aangelopoulos/ltt](https://github.com/aangelopoulos/ltt) (MIT) — [arXiv](https://arxiv.org/abs/2110.01052)
- **SeqCRC — Andéol, Mossina, Mazoyer, Gerchinovitz, "Conformal Object Detection by Sequential Risk Control", arXiv 2505.24038 (v2 31 Oct 2025).**
  - Defines Conformal Object Detection with sequential parameters:
    - λ^cnf, a confidence filter, controlled with a **"box-count-recall" loss (ratio of missed objects)**
    - λ^loc, additive or multiplicative box margins, with a **pixel-wise loss (average fraction of the true box area covered)**
    - λ^cls, class prediction sets
  - Guarantee: E[L] ≤ α per task. It can be image-level or object-level ("each annotated object counts as an instance"), and is marginal (expectation) and not per-instance.
  - Requires exchangeability. Non-monotone losses, caused by matching changes, are handled with a "monotonization trick" that is conservative.
  - Multiplicative margins are preferred because with additive ones "predictions for small objects may suffer very large corrections".
  - Toolkit: **[github.com/leoandeol/cods](https://github.com/leoandeol/cods)**. The GitHub API shows no license declared, so default copyright applies; ask the authors. Experiments on COCO — [arXiv HTML](https://arxiv.org/html/2505.24038), [arXiv abs](https://arxiv.org/abs/2505.24038)
  - A companion applied paper covers railway-signal detection: "Sequential Conformal Risk Control for Safe Railway Signaling Detection", PMLR v266 (2025) — [PMLR](https://proceedings.mlr.press/v266/andeol25a.html)
- **Timans, Straehle, Sakmann, Nalisnick, "Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction", ECCV 2024 (oral).**
  - First builds conformal class-label sets, then box intervals conditional on them. This "broadens the validity of the conformal coverage guarantees to include incorrectly classified objects".
  - Ensemble and quantile-regression variants make intervals adapt to object size ("more equitable coverage across different scales").
  - "Desired coverage levels are satisfied with practically tight predictive uncertainty intervals" — [arXiv](https://arxiv.org/abs/2403.07263), [ECCV](https://eccv.ecva.net/virtual/2024/oral/139). Code URL and datasets were not verified in my fetch.
- **PAC multi-object detection and tracking — Li, Park, Ji, Lee, Bastani, arXiv 2204.07482 (2022, classic).**
  - Builds (ε,δ)-correct prediction sets around each detection and around the **set of candidate edges (frame-to-frame transitions)**.
  - Calibration picks the largest τ such that calibration errors ≤ k*, where F(k*; n, ε) ≤ δ (binomial CDF).
  - COCO person/car: ε_det=0.2, δ=10⁻⁵, with about **6,000 calibration examples per detection component**. Composed set error was 0.097 (target 0.2).
  - MOT17: ε_edge=0.005, δ=10⁻²; the edge-set false-negative rate was 0.004.
  - Code is on a Google Drive link — [ar5iv](https://ar5iv.labs.arxiv.org/html/2204.07482)
- **Conformal Prediction Sets for Instance Segmentation (Lu, Kluger, Bates, Wang; arXiv 2602.10045, Feb 2026, rev. Jul 2026).**
  - Guarantees the probability that at least one mask in the returned set has high IoU with the true instance at a queried pixel. Sets are adaptive in size.
  - Evaluated on agricultural fields, cells and vehicles; it outperforms naive parameter selection and morphological dilation. Code URL not found — [arXiv](https://arxiv.org/abs/2602.10045)
- Conformal ideas also appear in active learning for video detection. FOCAL (IEEE BigData 2023) proposes "conformal active learning" samplers that exploit the sequential structure of video — [arXiv](https://arxiv.org/abs/2311.10591) (details in §5).
- For auto-labeling specifically, see Conformal Labeling (FDR-controlled acceptance, 10% calibration) and PAC Labeling in §1.

### Inferences
Practical recipe for VIAT. This is inference from the cited methods; formulas are from CRC/LTT/PAC.
1. **Auto-accept boxes with FDR ≤ α.** Score each auto-box, e.g. 1 − confidence, or a learned score mixing confidence, SAM mask IoU, and agreement between two detectors. Draw a *random* calibration sample of auto-boxes and have a human mark each correct or incorrect. Compute conformal p-values against the incorrect ones and select with BH (Conformal Labeling).
2. **Bound the missed-object rate per frame.** Calibrate the detector's confidence threshold with the SeqCRC box-count-recall loss on fully verified frames (every object checked). With CRC's B/(n+1) term and B = 1, the achievable α is bounded below by about 1/(n+1). So 100 calibration frames cannot certify a miss rate below about 1%, and 500–1,000 frames are a realistic range.
3. **Tight "for this batch" guarantees.** If the requirement is a high-probability statement ("with 95% confidence the accepted set has ≤2% bad boxes"), use LTT or PAC binomial bounds. Minimum zero-failure calibration sizes are then about 149 (2%), 299 (1%) and 598 (0.5%) at δ = 0.05 (see §6).
4. **Tracks.** Treat association like PAC MOT's edge sets. A frame where the calibrated edge set contains more than one plausible candidate is an association ambiguity that belongs in the review queue.
5. **Small objects (VisDrone).** Use multiplicative, size-adaptive margins (SeqCRC, Timans). Calibrate per size stratum, because miss rates for tiny objects differ sharply.

### Gaps
- I could not extract SeqCRC's numeric results (realized risks, margin sizes, COCO split sizes) or Timans' datasets and code URL from the fetched pages.
- I found no conformal method validated on aerial/VisDrone data or on SAM-propagated tracks.
- The `cods` license is unspecified.

---

## 3. Voxel51 (2025) "Auto-Labeling Data for Object Detection" / "Verified Auto Labeling"

### Takeaway
Voxel51 trained detectors on zero-shot foundation-model labels and compared them with human labels:
- For VOC and COCO, auto-labels reached about 93–95% of human-label mAP50.
- For LVIS and BDD (long-tail, small, driving scenes) they reached only about 62–68%.
- Labeling time was roughly 5,300× lower and cost roughly 105,000× lower.

The key operational finding is that **a low confidence threshold (≈0.2, favoring recall) produces better downstream models than a high-precision threshold**. Their product workflow ("Verified Auto Labeling") adds human verification prioritized by confidence, but that feature is **[PAID]** FiftyOne Enterprise beta only.

### Cited Findings
- **Paper:** Griffin, Gangwar, Sela, Corso (Voxel51), "Auto-Labeling Data for Object Detection", arXiv 2506.02359, 3 June 2025 — [arXiv](https://arxiv.org/abs/2506.02359)
- **Setup:**
  - Datasets: VOC (20 classes, 16,551 train images), COCO (80, 118,287), LVIS (1,203, 100,170), BDD (10, 70,000).
  - Auto-labelers: YOLO-World (72.9M params), YOLOE (35.2M) and Grounding DINO (172.2M). Labeling VOC took 197 s, 205 s and 2,290 s respectively.
  - Downstream detectors trained on the labels: YOLO11 n–x and RT-DETR — [arXiv HTML](https://arxiv.org/html/2506.02359)
- **Downstream mAP50, YOLO11n trained on human vs auto labels:**
  - VOC 0.756 vs 0.715
  - COCO 0.496 vs 0.460
  - LVIS 0.087 vs 0.059
  - BDD 0.434 vs 0.271 — [arXiv HTML](https://arxiv.org/html/2506.02359)
- **Thresholds:** swept from 0.05 to 0.9. The best downstream mAP came with YOLO-World at α = 0.1–0.2, and **α = 0.2 was the single reliable configuration**. High-confidence thresholds (0.8) gave the best precision but the **worst downstream mAP in all experiments**. "High auto-label recall is the best single predictor of downstream model performance followed by high auto-label F1 score" — [arXiv HTML](https://arxiv.org/html/2506.02359)
- **Auto-label F1 vs human labels (best setting):** VOC 0.785, COCO 0.640, LVIS 0.215, BDD 0.499 — [arXiv HTML](https://arxiv.org/html/2506.02359)
- **Rare classes:** LVIS rare classes drop more than 97% in mAP relative to frequent classes. Infrequent BDD classes drop 65–71% with auto-labels — [arXiv HTML](https://arxiv.org/html/2506.02359)
- **Cost (all 4 datasets, 3.45M objects):**
  - Human: 6,703 hours and $124,092.54.
  - Auto (YOLO-World at 0.2): 1.27 hours and $1.18 on a rented L40S at $0.93/h — [arXiv HTML](https://arxiv.org/html/2506.02359)
  - The blog headline says "100,000× cheaper and 5,000× faster" — [Voxel51 blog, 4 June 2025](https://voxel51.com/blog/zero-shot-auto-labeling-rivals-human-performance)
- **Verified Auto Labeling (VAL) workflow (blog):**
  1. Auto-label with a foundation model.
  2. Use moderate thresholds (0.2–0.5, not 0.8–0.9).
  3. Humans verify, "prioritized by confidence scoring" so that effort goes to "labels most needing review".
  - Rare or specialized classes (LVIS, BDD) need a hybrid of auto and targeted human labeling.
  - VAL is "in beta for FiftyOne Enterprise customers only — not open-source" **[PAID]**.
  - The blog claims "up to 95%" of human-label model performance — [Voxel51 blog](https://voxel51.com/blog/zero-shot-auto-labeling-rivals-human-performance)

### Inferences
- Computed from the paper's numbers:
  - Auto-label mAP50 as a share of human-label mAP50: VOC 94.6%, COCO 92.7%, LVIS 67.8%, BDD 62.4%.
  - Time ratio 6,703 h / 1.27 h ≈ 5,300×; cost ratio ≈ 105,000×.
- BDD (driving, many small or occluded objects, 10 classes) is the closest proxy to VisDrone-style drone footage, and it was the **worst case at 62% of human-label mAP**. VIAT should not assume zero-shot labels are "good enough" for aerial small objects without review.
- Two operating points are needed:
  - **Low threshold (≈0.2) to generate candidates**, protecting recall, which drives training value.
  - A separate **acceptance decision** (§1/§2), so precision errors are routed to review rather than removed by a high threshold.
- The open-source parts (FiftyOne App, Brain: Apache-2.0, see §4/§5) can reproduce most of VAL's triage locally. Only the enterprise "VAL" packaging is paid.

### Gaps
- The paper does not report human verification time or how much verification closes the gap to human-label mAP.
- "RER-backed confidence scoring" in the blog is not defined in the sources I found.
- No code or data release URL was found for the paper.

---

## 4. Label-error detection for detection datasets and video/MOT annotations; running it on auto-labels; measured precision/recall

### Takeaway
The mature open-source tools score each image or box by comparing the given labels with **out-of-sample** model predictions (ObjectLab/cleanlab), or by the detector's training loss on each given box (loss inspection, WACV 2024). Measured precision among the top-ranked suspects is 42–72% on real COCO/VOC errors, and 97% on a proprietary set. But the best methods still **miss up to 66% of errors** (2025 KITTI benchmark), and missing boxes are the hardest error type. For video there is little academic work: Uber's 2026 production system scores each frame for ID swaps and position jumps with XGBoost over motion and appearance features, but reports no metrics.

### Cited Findings
**cleanlab ObjectLab** — Tkachenko, Thyagarajan, Mueller, ICML 2023 DMLR workshop, arXiv 2309.00832 (classic, 2023). Implemented in `cleanlab.object_detection`: [github.com/cleanlab/cleanlab](https://github.com/cleanlab/cleanlab) (**Apache-2.0** per GitHub API, 2026; runs locally, Python 3.10+) — [cleanlab repo](https://github.com/cleanlab/cleanlab), [tutorial](https://docs.cleanlab.ai/stable/tutorials/object_detection.html)
- **Scoring:**
  - Box similarity is sim(B₁,B₂) = α·k(B₁,B₂) + (1−α)·IoU(B₁,B₂), where k is a Gaussian kernel on box distance.
  - Per annotated box:
    - *badloc* = max sim to a predicted box of the same class (1 if there is none).
    - *swap* = 1 − max sim to a predicted box of a **different** class with confidence > τ = 0.95.
  - Per predicted box with confidence > τ:
    - *overlooked* = sim to the nearest same-class annotated box, or sim*·(1−p̂) if there is none.
  - Each subtype is softmin-pooled over boxes, q̄ = ⟨q, softmax(1−q)⟩. The image score is the geometric mean of the three — [ObjectLab paper PDF](https://dmlr.ai/assets/accepted-papers/112/CameraReady/objectlab_icml2023_lowerres.pdf)
- **Out-of-sample predictions:** quality scores "are computed using out-of-sample predictions… via straightforward 5-fold cross-validation" — [paper PDF](https://dmlr.ai/assets/accepted-papers/112/CameraReady/objectlab_icml2023_lowerres.pdf)
- **Results:**
  - COCO-full, precision@100 (manually verified): ObjectLab 0.58 vs 0.22 for a per-image mAP score (Detectron2-X101), and 0.42 vs 0.17 (Faster R-CNN).
  - ObjectLab also had the best AP and precision@K on COCO-bench (2,171 images, 5 classes, 251 truly mislabeled images per triple annotation) and on SYNTHIA-AL (5,000 images, 22% with injected errors).
  - Estimated COCO 2017 error prevalence: about 5% of images have an overlooked box, 3% a badly located box, 0.7% a swapped class — [paper PDF](https://dmlr.ai/assets/accepted-papers/112/CameraReady/objectlab_icml2023_lowerres.pdf)
  - Benchmark code: github.com/cleanlab/object-detection-benchmarks (cited in the paper; the GitHub API returned 404 in 2026) — [paper PDF](https://dmlr.ai/assets/accepted-papers/112/CameraReady/objectlab_icml2023_lowerres.pdf)
- **Behavior:** ObjectLab "does not falsely flag images with imperfect model predictions that are not actually mislabeled", because low-confidence predictions are ignored — [cleanlab blog](https://cleanlab.ai/blog/learn/object-detection/)

**Loss inspection — Schubert, Riedlinger, Kahl, Kröll, Schoenen, Šegvić, Rottmann, "Identifying Label Errors in Object Detection Datasets by Loss Inspection", WACV 2024.** Code: [github.com/mschubert95/identifying_label_errors_in_od](https://github.com/mschubert95/identifying_label_errors_in_od) (MMDetection-based; **Apache-2.0** per GitHub API) — [WACV](https://openaccess.thecvf.com/content/WACV2024/html/Schubert_Identifying_Label_Errors_in_Object_Detection_Datasets_by_Loss_Inspection_WACV_2024_paper.html)
- **Method:** score each given box by "the sum of both stages' classification and regression losses" of a two-stage detector (Swin-T or ResNeSt101 with a Cascade RoI head). Simulated error types are drops (missing boxes), flips (wrong class), shifts (misplaced boxes) and spawns (extra boxes) — [arXiv](https://arxiv.org/abs/2303.06999)
- **Simulated errors, AUROC:** EMNIST-Det 99.46 vs detection score 73.24 and entropy 71.49. BDD 96.30 vs 76.82. Max F1 on BDD 56.59 vs 31.14 — [arXiv HTML](https://arxiv.org/html/2303.06999)
- **Real errors:** precision among the top-200 proposals was **71.5% on VOC** and **61.0% on COCO**, and **97%** on a proprietary dataset — [arXiv HTML](https://arxiv.org/html/2303.06999)

**2025–2026 benchmark and correction work**
- **REC✓D — Penquitt, Klees, Cakaj, Kondermann, Rottmann, Schmarje, "From Label Error Detection to Correction", arXiv 2508.06556 (v2 30 Jan 2026).** Code: [github.com/JonathanKlees/rechecked](https://github.com/JonathanKlees/rechecked) (**MIT** per GitHub API).
  - Pipeline: detectors (YOLOX, Cascade R-CNN) + error detectors (MetaDetect, instance-wise loss, ObjectLab) → crowdsourced microtask verification (11 annotators per task, yielding soft labels) → correction.
  - On the KITTI pedestrian class, at least **18% of original annotations were missing or inaccurate**: 228 overlooked pedestrians, 156 misaligned boxes and 85 false-positive labels.
  - "up to 66% of errors remain undetected" by current methods.
  - Measured microtask cost per box: drawing a box 44.11 s; semantic validation ("is it a pedestrian?") 37.87 s; high-quality validation with activity classification 124.75 s — [arXiv HTML](https://arxiv.org/html/2508.06556)
- **Learning to detect label errors by injecting them — Penquitt, Riedlinger, Heller, Reischl, Rottmann, arXiv 2508.17930 (Aug 2025).** Injects synthetic errors and recasts error detection as instance segmentation. It works for object detection, semantic segmentation and instance segmentation, and releases a benchmark of **459 real label errors in Cityscapes**. Paper is CC BY 4.0; code URL not found — [arXiv](https://arxiv.org/abs/2508.17930)
- **MJ-COCO — Kim et al., arXiv 2506.00997 (June 2025).** Loss/gradient-based error detection plus pseudo-label refinement fixes missing labels, wrong classes, inaccurate boxes, duplicates and group-labeling inconsistencies. It adds more than 200,000 small-object annotations vs COCO, and models trained on MJ-COCO beat COCO-trained models in AP and AP_S — [arXiv](https://arxiv.org/abs/2506.00997)
- **CLOD — Chachuła et al., arXiv 2211.13993 (2022); journal version in Machine Learning (Springer, 2025).** Confident learning extended to detection, clustering boxes by IoU. It flagged "nearly 80% of artificially disturbed bounding boxes with a false positive rate below 0.1", and cleaning with its most confident suggestions "improved mAP scores by 16% to 46%" — [arXiv](https://arxiv.org/abs/2211.13993), [Springer](https://link.springer.com/article/10.1007/s10994-025-06976-x)

**FiftyOne Brain mistakenness** ([fiftyone-brain](https://github.com/voxel51/fiftyone-brain) is **Apache-2.0** per GitHub API; runs locally)
- `compute_mistakenness()` compares a prediction field with a "ground truth" field. It ranks detections where the model "is highly confident but wrong (according to the human annotation label)" as likely mistakes.
- It sets `possible_missing` on predicted objects with no match that are "deemed to be likely correct annotations", and `possible_spurious` counts for likely spurious annotations — [FiftyOne Brain docs](https://docs.voxel51.com/user_guide/brain.html), [detection mistakes tutorial](https://docs.voxel51.com/tutorials/detection_mistakes.html)
- I found no published precision/recall for it.

**Video / MOT label errors**
- **Uber** (blog, 23 Apr 2026) runs an ML validator inside its uLabel tool.
  - Model: XGBoost over an **11-frame sliding window with 283 features per frame**. Features are visual (template matching of box content across frames), motion (velocity, acceleration, direction consistency, aspect-ratio change, edge proximity) and coordinate (frame-to-frame position and size change).
  - Output: a per-frame error probability. It targets "ID swaps" and "position jumps"; freeze and scale/aspect errors are planned.
  - Operators accept, fix or dismiss flags. It is "deployed… across all our bounding box annotation projects", but **no precision/recall or time savings were published**. Internal, not public — [Uber blog](https://www.uber.com/us/en/blog/validating-bounding-box/)
- Academic work on noisy MOT labels exists (e.g., MOTNL, "Multi-object Tracking with Noisy Labels", which diagnoses noisy labels during training) — [ResearchGate](https://www.researchgate.net/publication/364159361_Multi-object_Tracking_with_Noisy_Labels). I found no benchmarked MOT label-error *detector* with precision/recall figures.

### Inferences
- **Running these on auto-labels.** Treat the auto-labels as the "given labels" and supply an *independent* prediction source. Options:
  1. Train a VIAT YOLO on the auto-labels with K-fold cross-fitting (ObjectLab's 5-fold recipe) and use each fold's out-of-sample predictions.
  2. Use a different model family, e.g. YOLO-World labels vs SAM 3 text-prompt predictions, or vice versa.
  3. Use SAM-refined boxes as the localization reference for the *badloc* score.
- If predictions come from the same model that produced the labels, disagreement is near zero on exactly the systematic errors. Independence of the checker is the crux.
- **Loss inspection** is the most natural fit when VIAT already trains a detector on the auto-labels: the per-box loss at the end of training is nearly free. **ObjectLab** is cheaper to bolt on because it needs only predictions, and it is already Apache-2.0 packaged.
- **Expect recall-type errors to dominate on drone data**: missed tiny objects, as with COCO's ~5% overlooked-box prevalence and KITTI's 228 overlooked pedestrians. Error detectors miss many of these, so recall must also be audited statistically (§6), not only via error ranking.
- **For tracks,** an Uber-style per-frame classifier (motion, appearance and size-change features over a sliding window) is simple to replicate locally with XGBoost or LightGBM on VIAT's own track data. Frame scores can be aggregated to a track score, e.g. max or top-k mean, to sort tracks for review.

### Gaps
- No published precision/recall for FiftyOne mistakenness or for the Uber system.
- No standard benchmark for label-error detection in MOT/video annotations was found.
- REC✓D's per-method precision/recall numbers were not extracted beyond the aggregates above.
- The 37.87 s "semantic validation" figure includes crowd microtask overhead; per-decision expert verification times (see §5) are far lower, so the numbers are not directly comparable.

---

## 5. Human-review prioritization: active learning, exception-based and track-level review, batch correction by clustering, near-duplicate removal — and reported human-time savings

### Takeaway
The biggest measured savings come from **changing the human's task from drawing to verifying**, and from **deferring only uncertain decisions**:
- Yes/no verification of a box takes about 1.6 s vs 26–42 s to draw one, a 6–9× saving (classic).
- SAM 2's propagate-then-correct loop cut per-frame mask annotation from 37.8 s to 4.5 s (8.4×).
- SAM 3's learned AI verifiers roughly doubled data-engine throughput.
- SPAM (ECCV 2024) annotated MOT datasets with 3–20% of full human effort by asking humans only about uncertain graph nodes and edges at tracklet level. Trackers trained on those labels matched GT-trained trackers.

Evidence for clustering-based batch correction is thin (patents, small user studies). Dedup tools exist (FiftyOne Brain Apache-2.0; fastdup non-commercial license), and video AL papers select about 10–12% of frames.

### Cited Findings
**Verification vs drawing (classic)**
- Papadopoulos et al., "We don't need no bounding-boxes", CVPR 2016:
  - Yes/no box verification took **1.6 s**, and a 5-way verification 2.4 s.
  - ILSVRC reports **26 s** to draw a box without quality control and **42 s** with it.
  - Total effort fell **6–9×**. Detectors reached about 88% of fully supervised mAP on VOC 2007 (AlexNet 45 vs 51; VGG16 58 vs 66), with about 2.5 checks per image — [ar5iv](https://ar5iv.labs.arxiv.org/html/1602.08405), [CVPR PDF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Papadopoulos_We_Dont_Need_CVPR_2016_paper.pdf)

**Propagation and verification loops in video**
- **SAM 2 data engine** (Ravi et al., 2024/ICLR 2025):
  - Phase 1, SAM per frame: 37.8 s/frame.
  - Phase 2, SAM + SAM 2 mask propagation: 7.4 s (5.1×).
  - Phase 3, full SAM 2 with refinement clicks on intermediate frames: **4.5 s/frame (8.4×)** — [SAM 2 paper](https://arxiv.org/pdf/2408.00714)
  - Code is Apache-2.0 per GitHub API ([facebookresearch/sam2](https://github.com/facebookresearch/sam2)).
- **SAM 3 data engine** (arXiv 2511.16719, Nov 2025):
  - Human tasks are split into **Mask Verification** (accept or reject each mask) and **Exhaustivity Verification** ("if all instances of the NP have been masked"). Failing items go to manual correction.
  - Fine-tuned Llama 3.2 "AI verifiers" trained on Phase-1 human accept/reject labels "roughly double[] the data engine's throughput", with "near-human accuracy", letting humans focus "on the most challenging cases".
  - Using the exhaustivity verifier instead of SAM 3's own confidence scores improved cgF1 by +7.2, and mask verification added +1.1 — [arXiv HTML](https://arxiv.org/html/2511.16719)
- **Kuznetsova et al. (Google), "Efficient video annotation with visual interpolation and frame selection guidance", WACV 2021 (classic).** A learned interpolation/extrapolation model plus a "what frame to annotate next" guide cut manually drawn boxes by **60% vs linear interpolation and 35% vs an off-the-shelf tracker** — [arXiv](https://arxiv.org/abs/2012.12554)

**Track-level, uncertainty-deferred annotation**
- **SPAM (Cetintas, Meinhardt, Brasó, Leal-Taixé), "SPAMming Labels: Efficient Annotations for the Trackers of Tomorrow", ECCV 2024.**
  - A pre-trained model produces pseudo-labels. A hierarchical graph (10 GNN levels, from tracklets up to tracks) covers both detections and associations.
  - Node uncertainty is uncert(v) = max over neighbor edges of the binary entropy H(ŷ_(v,u)). Only uncertain nodes are sent to the annotator.
  - Actions are costed in clicks: accept/reject a detection = 1, refine a box = 2, associate across time = 1.
  - Human effort was **3.3% of objects on MOT17, 10% on MOT20, 20% on DanceTrack** ("3–20% of the human labeling effort").
  - Trackers trained on SPAM labels, MOT17 test:
    - ByteTrack 51.6 HOTA / 64.0 MOTA / 63.0 IDF1 vs 50.6 / 60.4 / 61.1 with GT labels
    - GHOST 51.3 / 61.9 / 62.1 vs 49.5 / 58.0 / 59.0
  - DanceTrack test: ByteTrack 41.3 vs 41.0 HOTA; GHOST 43.9 vs 45.8.
  - Paper is CC BY 4.0. The authors say they release code. The repo github.com/ocetintas/SPAM (first author's handle, no license declared) exists but I did not confirm it is the official release — [arXiv](https://arxiv.org/abs/2404.11426), [arXiv HTML](https://arxiv.org/html/2404.11426)

**Active learning with real costs, including video**
- **FOCAL (Kokilepersaud et al., IEEE BigData 2023):** the first cost-aware video AL dataset, with 126 sequences over 69 scenes and real annotation and QA time per sequence. Annotation time depends on the assistive tool, occlusion, weather and motion. The best "conformal active learning" sampler cost **113 hours less** than the best traditional AL at comparable performance, with ≥77.67% less compute overhead. Paper is CC BY 4.0 — [arXiv](https://arxiv.org/abs/2311.10591)
- **Temporal-coherence AL for video detection (Bengar et al., ICCVW 2019, classic).** Detections are linked forward and backward in a graph to *estimate FP and FN without labels*, and frames are ranked by estimated errors. Experiments used 12% of SYNTHIA-AL and 10% of ImageNet-VID frames — [arXiv](https://arxiv.org/abs/1908.11757)
- **Foundation-Assisted Active Learning for Object Detection Annotation (Zhang et al., arXiv 2607.16671, July 2026; remote sensing: DIOR, HRSC2016, DOTAv2, FAIR1M).**
  - UPN proposals are refined by SAM2, and DINOv2 features drive diversity sampling. Humans "verify rather than draw". Uncertainty is the harmonic mean of SAM-vs-detector IoU agreement and classification confidence.
  - UPN+SAM2 top-200 proposal recall on HRSC2016 was 98.3% at IoU 0.5.
  - Cold start with 20/30 labeled images: 57%/61% mAP vs 22%/45% without box switching. **No time measurements.** Paper is CC BY 4.0; no code URL — [arXiv HTML](https://arxiv.org/html/2607.16671)

**Near-duplicate and uniqueness filtering**
- **FiftyOne Brain** (Apache-2.0) offers similarity search, "finding unique and representative samples", and detection of media-quality problems and annotation mistakes — [fiftyone-brain](https://github.com/voxel51/fiftyone-brain)
- **fastdup** (Visual Layer) finds duplicates and near-duplicates, outliers and mislabels, but is licensed **CC BY-NC-ND 4.0 (non-commercial, no derivatives)**. This matters if VIAT is distributed commercially — [fastdup LICENSE](https://github.com/visual-layer/fastdup/blob/main/LICENSE), [repo](https://github.com/visual-layer/fastdup)

**Batch correction via embedding clustering**
- A 2025 PCB-defect pipeline embeds images, clusters them (k-means/GMM/HDBSCAN), verifies at cluster level, and "concentrates human effort on ambiguous clusters" — [Sensors 2025](https://doi.org/10.3390/s25206470)
- A user study with 24 participants found that SOM-based cluster visualizations "speed up labeling" (classic) — [Springer](https://link.springer.com/chapter/10.1007/978-3-642-21602-2_67)
- Neither source gives detection-specific time savings in the retrieved text.

**Confidence-prioritized review as the vendor pattern:** Voxel51 VAL prioritizes human verification by confidence **[PAID]** — [Voxel51 blog](https://voxel51.com/blog/zero-shot-auto-labeling-rivals-human-performance)

### Inferences
- **"One decision per track" is supported by the evidence.** A track-level accept/reject should cost on the order of a yes/no verification (≈1.6–2.4 s per decision in Papadopoulos) plus the time to scan a thumbnail strip. That is orders of magnitude less than per-frame review. SPAM shows most tracklet decisions can be auto-resolved (97% on MOT17). This time estimate is extrapolated; no study measured strip review directly.
- Suggested triage queue order for VIAT, combining cited mechanisms:
  1. Tracks or frames with highest *estimated error*: SPAM-style edge entropy, Uber-style motion anomalies, low SAM2 predicted IoU or object score (§8).
  2. ObjectLab/loss-inspection outliers (§4).
  3. Exhaustivity checks on sampled frames for missed objects (SAM 3 EV-style).
  4. A random audit sample (§6) that is *never* chosen by the model, to keep the error estimate unbiased.
- **Batch correction.** Clustering crops of *flagged* false positives with DINOv2 or CLIP embeddings and applying one decision per cluster is plausible and cheap. Every cluster decision should still be spot-checked, because published speedups for detection are lacking.
- **Dedup.** For review volume, sample frames at about 1–2 fps or use FiftyOne near-duplicate grouping. For VIAT distribution, prefer FiftyOne Brain (Apache-2.0) or a custom embedding-cosine dedup over fastdup, because of its NC-ND license.

### Gaps
- No controlled study measured time per *track* verification with thumbnail strips.
- No detection-specific measurement of cluster-level bulk correction speedups was found.
- Bengar et al.'s performance at 10–12% of frames relative to full data was not verified in detail.
- Recent unreviewed items (titles only):
  - "Generative Verification: Rethinking the Uncertainty Signal for Active Learning of Object Detection" (arXiv 2609.20262, Sep 2026) — [arXiv](https://arxiv.org/pdf/2609.20262)
  - "Human-efficient annotation system via active semi-supervised learning and selective review" (2026) — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S266730532600075X)

---

## 6. Statistical auditing: how many random samples to review to estimate precision/recall with a confidence interval; acceptance sampling; stratified sampling; worked sample sizes

### Takeaway
Use exact (Clopper–Pearson) or Wilson intervals on a **random** audit sample, separate from the triage queue:
- Certifying "≤1% bad items at 95% confidence" needs **299 reviewed items with zero errors**, or 473 if one error is found.
- Certifying ≤2% needs 149 items (zero errors) or 236 (one error).
- Estimating a ~90% precision to ±3% needs about 385 items.

Stratifying by model confidence and over-sampling low-confidence strata shrinks the interval (about 1.85× effective sample size in the worked example below). Prediction-powered and active inference use model scores to cut labels by a further 25–60% vs uniform sampling, while keeping valid CIs.

### Cited Findings
- **Interval formulas:**
  - Wilson score interval: p ∈ [1/(1+z²/n)]·(p̂ + z²/(2n) ± (z/(2n))·√(4n·p̂(1−p̂) + z²)).
  - Clopper–Pearson: B(α/2; x, n−x+1) < p < B(1−α/2; x+1, n−x) (beta quantiles). Its coverage is "never less than the nominal 1−α", so it is conservative.
  - Wilson is "the most accurate and the most robust" of the approximations.
  - **Rule of three:** 0 events in n gives an approximate 95% upper bound of 3/n.
  - Wald intervals behave badly near 0 or 1, which is exactly the regime of error rates — [Wikipedia: Binomial proportion confidence interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)
- **Prediction-powered inference (PPI)** (Angelopoulos, Bates, Fannjiang, Jordan, Zrnic, arXiv 2301.09633, 2023) gives "provably valid confidence intervals" for means and other quantities by combining a small labeled set with model predictions on a large unlabeled set, "without making any assumptions on the machine-learning algorithm". More accurate predictions give narrower intervals. Code: [ppi_py](https://github.com/aangelopoulos/ppi_py) (**MIT** per GitHub API) — [arXiv](https://arxiv.org/abs/2301.09633)
- **Active statistical inference** (Zrnic & Candès, ICML 2024):
  - Estimator: θ̂ = (1/n)Σ[f(Xᵢ) + (Yᵢ − f(Xᵢ))·ξᵢ/π(Xᵢ)], where ξ marks labeled items.
  - Sampling rule: **π(x) ∝ √E[(Y−f(X))² | X]**, i.e. label where the model is uncertain, estimated with a learned error predictor mixed with uniform sampling.
  - Savings:
    - Post-election survey: 85%+ vs classical, about 25% vs uniform (PPI).
    - Census regression: about 80% vs classical, **>60% vs uniform**.
  - Code: [github.com/tijana-zrnic/active-inference](https://github.com/tijana-zrnic/active-inference) (no license declared) — [arXiv HTML](https://arxiv.org/html/2403.03208), [ICML](https://icml.cc/virtual/2024/poster/34504)
- **Stratified PPI** (Fisch et al., NeurIPS 2024) shows that simple stratification with appropriate sample allocation gives "substantially tighter confidence intervals", especially when the autorater's accuracy varies across strata — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c9fcd02e6445c7dfbad6986abee53d0d-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2406.04291)
- **Guarantee-style bounds in auto-labeling:** PAC Labeling uses betting-based or CLT upper confidence bounds to decide what the expert must label — [arXiv HTML](https://arxiv.org/html/2506.10908). PAC MOT uses the binomial CDF condition F(k; n, ε) ≤ δ — [ar5iv](https://ar5iv.labs.arxiv.org/html/2204.07482)

### Inferences
**Worked examples.** I computed these with exact binomial (Clopper–Pearson) math from the formulas above; they are not from a paper.

*Items to review for a 95% one-sided guarantee that the error rate is ≤ p_U, given c errors found in the sample:*

| target p_U | c = 0 | c = 1 | c = 2 | c = 5 |
|---|---|---|---|---|
| 5% | 59 | 93 | 124 | 208 |
| 2% | 149 | 236 | 313 | 523 |
| 1% | 299 | 473 | 628 | 1,049 |

- With zero errors, n = ⌈ln(1−conf)/ln(1−p_U)⌉:
  - At 90% confidence: 45 (5%), 114 (2%), 230 (1%), 460 (0.5%).
  - At 99% confidence: 90 (5%), 228 (2%), 459 (1%), 919 (0.5%).
- One-sided 95% Clopper–Pearson upper bounds for observed errors (with two-sided Wilson 95% intervals where computed):

| errors / sample | 95% upper bound | Wilson 95% interval |
|---|---|---|
| 0/100 | 2.95% | 0–3.7% |
| 1/100 | 4.66% | — |
| 3/100 | 7.57% | 1.0–8.5% |
| 0/300 | 0.99% | — |
| 3/300 | 2.56% | 0.34–2.90% |
| 5/500 | 2.09% | — |
| 10/1000 | 1.69% | — |

- Estimation (not certification) sample sizes, 95%, normal approximation n = z²p(1−p)/E²:
  - p ≈ 0.90: ±5% needs 139; ±3% needs 385; ±2% needs 865.
  - p ≈ 0.95: ±2% needs 457; ±1% needs 1,825.
  - Worst case p = 0.5 with ±5%: 385.
- **Stratified example.** Accepted boxes are split by confidence:
  - high: 80% of items, 1% error
  - mid: 15% of items, 10% error
  - low: 5% of items, 40% error
  - Overall error is 4.3%. With n = 400:
    - simple random sampling SE = 1.01 pp
    - proportional allocation SE = 0.91 pp
    - **Neyman allocation** (n_h ∝ N_h·√(p_h(1−p_h)), i.e. 214 / 121 / 66) SE = **0.75 pp**, about 1.85× the effective sample size of simple random sampling.

**Audit design for VIAT**
- **Define separate audit units:**
  - (a) *precision of accepted boxes*: sample accepted boxes and ask a yes/no question, which is cheap;
  - (b) *recall / missed objects*: sample frames and have a human exhaustively check them for missing objects, which is expensive and clustered;
  - (c) *track correctness*: sample tracks and review the strip.
  - Report each with its own CI.
- **Clustering.** Boxes in the same frame, and frames in the same video, are correlated (design effect > 1). Sample frames and tracks across many videos, and compute CIs at the cluster level (e.g., a bootstrap over videos) rather than pretending boxes are independent.
- **When to stop reviewing.** After the triage queue has been worked, draw a *fresh random sample from the unreviewed remainder*. Stop when the upper confidence bound of its error rate is ≤ target. Otherwise send the next tranche of top-ranked items to review and repeat. Anytime-valid (betting) bounds, as used in PAC Labeling, permit peeking during sequential review without inflating error.
- **Using model scores for estimation.** Use the ObjectLab or track-error score as f(X) in PPI or active inference so fewer audited items are needed for the same CI width. Label with π ∝ estimated error so reviewers mostly see likely errors while the estimate stays unbiased.

### Gaps
- I did not retrieve ISO 2859-1 / ANSI Z1.4 acceptance-sampling tables. The (n, c) plans above are computed directly from the binomial.
- No paper was found that audits auto-generated *detection or tracking* datasets with PPI or active inference. The application is inference.
- Recent related unreviewed items (titles only):
  - "Which Metrics Save the Most Human Annotation? Prediction-Powered Evaluation and Meta-Evaluation" (arXiv 2608.26638) — [arXiv](https://arxiv.org/pdf/2608.26638)
  - "Active Multiple-Prediction-Powered Inference" (arXiv 2605.08429) — [arXiv](https://arxiv.org/html/2605.08429v1)
  - "Robust Conformalized Selection with Noisy Responses", for when reviewers themselves err (arXiv 2607.22985) — [arXiv](https://arxiv.org/html/2607.22985v1)

---

## 7. How good do labels need to be? Detector (and tracker) training robustness to label noise — setting acceptance targets

### Takeaway
For training, detectors are fairly tolerant of **missing** and **extra** boxes:
- Dropping 30% of boxes cost only about 5% mAP on VOC (classic).
- Tiny aerial detectors lost less than 2 mAP even at high missing/extra rates.

They are much less tolerant of **wrong classes** and, for **tiny objects**, of **inaccurate boxes**: mAP roughly halved under heavy box noise on AI-TOD. Voxel51 found **recall matters more than precision** for training on auto-labels. Tracker training is tolerant as well: SPAM labels built from 3–20% human effort matched GT-trained trackers. Evaluation and test sets need much higher quality and consistent guidelines than training sets.

### Cited Findings
- **Tiny aerial objects** (Zhu, Xu, Yang, Zhang, Zhang, Xia, "Robust Tiny Object Detection in Aerial Images amidst Label Noise", arXiv 2401.08056, Jan 2024).
  - Setup: AI-TOD-v2.0 (28,036 images, **average object size 12.7 px**) with FCOS+RFLA.
  - Missing and extra labels cost less than 2 mAP even at 60% noise.
  - Class shifts cause significant degradation.
  - Inaccurate boxes reduce mAP by "nearly half". The fetched summaries disagree on whether this was at 30% or 60% box noise.
  - The proposed DN-TOD gains +4.9 mAP at 40% mixed noise. Code was promised; URL not verified — [arXiv HTML](https://arxiv.org/html/2401.08056), [arXiv abs](https://arxiv.org/abs/2401.08056)
- **CLOD study** (Chachuła et al.): "every type of training dataset noise significantly decreases model quality", and more strongly with more noisy boxes. Ranking by harm: **uniform class-label noise > spurious boxes > missing boxes > location/scale noise** (location/scale magnitude was 25% of box dimensions) — [arXiv PDF](https://arxiv.org/pdf/2211.13993)
- **Missing annotations (classic):** "After dropping 30% of the annotations… Faster-RCNN only drops by 5% on the PASCAL VOC dataset". Soft Sampling beats baselines by more than 3% on OpenImagesV3 — [Wu et al., Soft Sampling, arXiv 1806.06986](https://arxiv.org/abs/1806.06986v1)
- **Auto-label training** (Voxel51 2025): "High auto-label recall is the best single predictor of downstream model performance". High-precision thresholds (0.8) produced the worst downstream mAP. On COCO, where even the best auto-label setting reached only 0.64 F1 against human labels, detectors trained on auto-labels still reached about 93% of human-label mAP50 (0.460 vs 0.496, YOLO11n) — [arXiv HTML](https://arxiv.org/html/2506.02359)
- **Verification-only supervision (classic):** detectors reached about 88% of fully supervised mAP on VOC 2007 without any drawn boxes — [ar5iv](https://ar5iv.labs.arxiv.org/html/1602.08405)
- **Re-annotation is not automatically better** (Ma et al., CVPRW 2022). Professional re-annotation of 80k COCO and 5k Open Images images *lowered* COCO mAP, because re-annotators added small and ambiguous objects the original guideline skipped, and re-annotating the test set "decreased the score largely". Re-annotation *raised* Open Images mAP. Guideline consistency between training and evaluation labels drives measured quality — [CVPRW paper](https://openaccess.thecvf.com/content/CVPR2022W/VDU/papers/Ma_The_Effect_of_Improving_Annotation_Quality_on_Object_Detection_Datasets_CVPRW_2022_paper.pdf), [summary](https://medium.com/sinicx/the-effect-of-improving-annotation-quality-on-object-detection-datasets-cvpr2022-workshop-64ef51fe5261)
- **Cleaning helps when errors are real:** CLOD cleaning gave +16–46% mAP, and MJ-COCO's more than 200k added small-object labels improved AP and AP_S — [CLOD arXiv](https://arxiv.org/abs/2211.13993), [MJ-COCO arXiv](https://arxiv.org/abs/2506.00997)
- **UNA benchmark** (Ryoo et al., arXiv 2312.13822, Dec 2023) covers categorization, localization, missing and bogus-box noise jointly. Code: [github.com/Ryoo72/UNA](https://github.com/Ryoo72/UNA) (**MIT**). I did not extract numbers — [arXiv](https://arxiv.org/abs/2312.13822)
- **Trackers:** trackers trained on SPAM labels (3.3% human effort on MOT17) matched or exceeded GT-trained trackers: ByteTrack 51.6 vs 50.6 HOTA — [arXiv HTML](https://arxiv.org/html/2404.11426)

### Inferences
- **Suggested acceptance targets for *training* data** (my synthesis, not a published standard):
  - Prioritize recall: keep the candidate threshold low and catch missed objects in audits.
  - Keep class-flip and FP rates low, e.g. target ≤2–5% of accepted boxes, certified via §6.
  - For tiny drone objects, treat localization quality as critical: SAM refinement and a box-tightness check are needed, because box noise is what halves tiny-object mAP.
- **For evaluation/test splits**, hold labels to a much stricter, guideline-consistent standard (≤1% certified error, exhaustive frames), because label noise there distorts measured mAP/HOTA directly (Ma et al.).
- **Error types need different verification routes.** Missing-box errors are cheap to tolerate in training but expensive to find, while class and localization errors are expensive to tolerate. VIAT's verifier spend should go to class and localization verification per box; recall should be handled statistically with sampled exhaustive frames.

### Gaps
- No controlled study was found quantifying how *tracker* training (re-ID/association) degrades with ID-switch or fragment noise at different rates. MOTNL exists, but numbers were not retrieved.
- The exact noise level at which AI-TOD box noise halves mAP (30% vs 60%) is unresolved between sources.
- No VisDrone-specific label-noise sensitivity study was found.

---

## 8. Estimating track quality: HOTA/IDF1/MOTA of auto-generated tracks against a small verified subset, and track quality without ground truth

### Takeaway
With a verified subset, compute HOTA (= √(DetA·AssA)) and its sub-metrics on a few exhaustively annotated clips or sparse frames, using TrackEval (MIT) or TETA (Apache-2.0, designed for incomplete or sparse annotations as in TAO's 1-fps labels). Add random track-level accept/reject audits with binomial CIs (§6). Without ground truth, usable per-frame signals exist:
- SAM 2's predicted-IoU and signed occlusion/object scores,
- mask overlap between different tracks,
- motion and appearance anomaly classifiers (Uber),
- graph-based FP/FN estimation (temporal coherence),
- association uncertainty (SPAM edge entropy, PAC edge sets).

None of these has a published calibrated mapping to HOTA, so they are ranking signals for triage, not quality estimates.

### Cited Findings
- **HOTA** (Luiten et al., IJCV 2021): combines detection accuracy (DetA) and association accuracy (AssA) over IoU thresholds 0.05–0.95, with HOTA = √(DetA × AssA). It decomposes into detection recall/precision, association recall/precision and localization accuracy — [HOTA, IJCV](https://dl.acm.org/doi/10.1007/s11263-020-01375-2)
- TrackEval implements HOTA and other MOT metrics: [github.com/JonathonLuiten/TrackEval](https://github.com/JonathonLuiten/TrackEval) (**MIT** per GitHub API). VIAT already vendors TrackEval under `viat/evaluation/trackEval/` (per the repo's CLAUDE.md).
- **Sparse or incomplete ground truth:**
  - TAO videos are 30 fps but annotated at **1 fps**, about 40 annotated frames per video, and the training annotations are incomplete — [TAO paper](https://arxiv.org/pdf/2005.10356), [OVT-B NeurIPS 2024 summary](https://proceedings.neurips.cc/paper_files/paper/2024/file/1adeeac24ce6168e20bcee85645720e9-Paper-Datasets_and_Benchmarks_Track.pdf)
  - **TETA** (Li et al., ECCV 2022) splits tracking into localization, association and classification accuracy and "supports evaluation with incomplete annotations". Code: [github.com/SysCV/tet](https://github.com/SysCV/tet) (**Apache-2.0**) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2207.12978)
- **Measured pseudo-track quality through downstream training:** SPAM's MOT17 validation table (trackers trained on the generated labels, evaluated on the validation split) gives 50.5 HOTA / 59.5 MOTA / 64.5 IDF1 with 0% manual annotation + pseudo-labels, and 52.5 / 61.8 / 66.2 with a 3.3% annotation budget. I did not retrieve the direct quality of the labels themselves against GT. SPAM also self-trains on pseudo-labels for a 4–7 HOTA gain with zero annotation — [arXiv HTML](https://arxiv.org/html/2404.11426)
- **SAM 2 per-frame confidence signals:** each prediction has an IoU score (estimated alignment with ground truth) and an occlusion score, where "positive values signify object presence, negative values indicate absence, and the magnitude reflects confidence" — [SAM2MOT, arXiv 2504.04519](https://arxiv.org/html/2504.04519v5)
  - The same search surfaced a failure check where two tracked instances' masks overlapping with mask IoU > 0.8 signals both tracking the same object. I could not pin this to a specific paper in the fetched text; it appeared in results alongside [SAM2MOT](https://arxiv.org/html/2504.04519v5) and [Seg2Track-SAM2](https://arxiv.org/html/2509.11772v2).
- **Frame-level error classifier without GT:** Uber's XGBoost scores each frame for ID swaps and position jumps from motion, visual and coordinate features — [Uber blog](https://www.uber.com/us/en/blog/validating-bounding-box/)
- **Graph-based FP/FN estimation without labels:** temporal-coherence AL links detections forward and backward and minimizes an energy to estimate false positives and false negatives per frame (classic) — [arXiv](https://arxiv.org/abs/1908.11757)
- **Association uncertainty:** SPAM uses the max binary entropy over graph edges per node — [arXiv HTML](https://arxiv.org/html/2404.11426). PAC MOT calibrates edge prediction sets with a false-negative rate of 0.004 at ε = 0.005 on MOT17 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2204.07482)
- **Older GT-free MOT quality metrics (classic):**
  - SQE, a "Self Quality Evaluation" metric for tuning MOT parameters without GT — [arXiv](https://arxiv.org/pdf/2004.07472)
  - A naive-Bayes combination of shape, appearance and motion features to score each tracking decision — [ResearchGate](https://www.researchgate.net/publication/261387356_Evaluation_of_tracking_algorithm_performance_without_ground-truth_data)

### Inferences
- **Verified-subset protocol for VIAT:**
  1. Randomly sample short clips across videos, stratified by predicted track difficulty. Annotate them exhaustively at a sparse rate (1–2 fps, TAO-style) and compute HOTA / DetA / AssA / IDF1 with the vendored TrackEval, or TETA if annotation is incomplete. Bootstrap over clips for CIs.
  2. Separately, randomly sample whole tracks for yes/no track audits (correct / ID switch / drift / wrong class / fragment) and report the track-level error rate with Clopper–Pearson bounds (§6).
- **GT-free track score for triage**, built from cited signals:
  - min and lower-quantile SAM2 predicted IoU over the track
  - fraction of frames with a negative object score followed by re-appearance (a re-acquisition risk)
  - maximum mask overlap with any other track (identity-merge risk)
  - per-frame motion/scale jump anomalies (Uber-style features)
  - disagreement between the propagated box and an independent detector's box
  - association entropy at track births and deaths
  - Calibrate this score against the random track audits (§1/§2) so it becomes an acceptance rule with a known FDR, not just a ranking.
- **Detection-side metrics** (DetPr/DetRe) can be estimated from sampled exhaustive frames. **Association metrics** (AssA, IDF1) require whole-track verification, so budget human time on full tracks rather than scattered frames for the association part.

### Gaps
- I found no 2024–2026 method that predicts HOTA/IDF1 of auto-generated tracks without ground truth with a validated error bar. GT-free signals remain uncalibrated heuristics.
- No study measured how well SAM2 predicted-IoU or object scores predict tracking failures (AUROC) on drone/VisDrone video.
- The mask-IoU > 0.8 merge-detection heuristic could not be attributed to a specific paper with certainty.
