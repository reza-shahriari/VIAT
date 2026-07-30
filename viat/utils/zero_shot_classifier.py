import os
import cv2
import json
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None


def check_transformers():
    try:
        import transformers
        return True
    except ImportError:
        return False


class ZeroShotClassifierManager:
    """
    Manager for zero-shot image classification using CLIP / SigLIP.

    Supports two annotation-refinement modes that can be used independently
    or together:

    Mode A – "correctness"
        For each annotation, look at only the *allowed* candidate labels
        defined for that class (e.g. ["car", "vehicle", "sedan"]).  If the
        top CLIP score is too low or ambiguous the annotation is flagged
        uncertain.  The annotation label is NOT changed to a sub-label; it
        is only flagged when CLIP disagrees with the existing label.

    Mode B – "mislabel"
        Compare every annotation against the full list of top-level class
        names.  If CLIP's top pick is a *different* class than the current
        annotation label (and the margin is decisive enough) the annotation
        is flagged as a potential mislabel.

    Both modes produce the same output: annotations whose ``uncertain``
    attribute is set to True.  No label is silently changed without the
    user's knowledge.
    """

    def __init__(self, checkpoints_dir=None):
        import sys as _sys
        # Resolve absolute checkpoints directory (same logic as ZeroShotManager)
        if checkpoints_dir is not None:
            self.checkpoints_dir = checkpoints_dir
        else:
            if getattr(_sys, 'frozen', False):
                _project_root = os.path.dirname(_sys.executable)
            else:
                # __file__ is viat/utils/zero_shot_classifier.py → go up 2 levels
                _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.checkpoints_dir = os.path.join(_project_root, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.model = None
        self.processor = None
        self.current_model_type = None
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Availability / loading
    # ------------------------------------------------------------------
    def is_available(self):
        return check_transformers()

    def load_model(self, model_type="openai/clip-vit-base-patch32"):
        """Load a HuggingFace CLIP or SigLIP model."""
        if self.current_model_type == model_type and self.model is not None:
            return True, "Model already loaded"

        if not self.is_available():
            return False, (
                "Transformers is not installed.  "
                "Please run: pip install transformers"
            )

        try:
            from transformers import (
                AutoProcessor,
                AutoModelForZeroShotImageClassification,
                CLIPModel,
                CLIPProcessor,
            )

            safe_name = model_type.replace("/", "_")
            local_path = os.path.join(
                self.checkpoints_dir, "zero_shot", safe_name
            )
            is_local = os.path.exists(os.path.join(local_path, "config.json"))

            if is_local:
                try:
                    if "clip" in model_type.lower() and "siglip" not in model_type.lower():
                        self.model = CLIPModel.from_pretrained(
                            local_path, local_files_only=True
                        )
                        self.processor = AutoProcessor.from_pretrained(
                            local_path, local_files_only=True
                        )
                    else:
                        self.model = AutoModelForZeroShotImageClassification.from_pretrained(
                            local_path, local_files_only=True
                        )
                        self.processor = AutoProcessor.from_pretrained(
                            local_path, local_files_only=True
                        )
                    self.current_model_type = model_type
                    if self.device == "cuda":
                        self.model = self.model.to(self.device)
                    return True, "Loaded local model successfully"
                except Exception as e:
                    print(
                        f"Failed to load local model from {local_path}, "
                        f"downloading…  Details: {e}"
                    )

            # Download to temp cache inside checkpoints
            temp_cache = os.path.join(
                self.checkpoints_dir, "zero_shot", "temp_cache"
            )
            os.makedirs(temp_cache, exist_ok=True)

            try:
                if "clip" in model_type.lower() and "siglip" not in model_type.lower():
                    self.model = CLIPModel.from_pretrained(
                        model_type, cache_dir=temp_cache
                    )
                    self.processor = AutoProcessor.from_pretrained(
                        model_type, cache_dir=temp_cache
                    )
                else:
                    self.model = AutoModelForZeroShotImageClassification.from_pretrained(
                        model_type, cache_dir=temp_cache
                    )
                    self.processor = AutoProcessor.from_pretrained(
                        model_type, cache_dir=temp_cache
                    )
            except Exception:
                self.processor = CLIPProcessor.from_pretrained(
                    model_type, cache_dir=temp_cache
                )
                self.model = CLIPModel.from_pretrained(
                    model_type, cache_dir=temp_cache
                )

            # Persist clean copy for offline use
            os.makedirs(local_path, exist_ok=True)
            self.processor.save_pretrained(local_path)
            self.model.save_pretrained(local_path)

            import shutil
            try:
                shutil.rmtree(temp_cache)
            except Exception as e:
                print(f"Warning: could not remove temp HF cache: {e}")

            if self.device == "cuda":
                self.model = self.model.to(self.device)

            self.current_model_type = model_type
            return True, "Classification model downloaded and loaded successfully"
        except Exception as e:
            return False, f"Failed to load classification model: {str(e)}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _crop_with_padding(self, image_array, box, padding_percent=0.15):
        """Crop a bounding box from the image with a contextual margin."""
        h, w = image_array.shape[:2]
        rect = box.rect
        x, y, bw, bh = rect.x(), rect.y(), rect.width(), rect.height()

        pad_x = int(bw * padding_percent)
        pad_y = int(bh * padding_percent)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        cropped = image_array[y1:y2, x1:x2]
        if cropped.size == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return cropped

    def _classify_image(self, cropped_image, candidate_classes):
        """
        Run zero-shot classification on a cropped image.

        Returns a sorted list of (class_name, confidence_score) tuples,
        highest score first.
        """
        if not self.model or not self.processor or not candidate_classes:
            return []

        try:
            image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            inputs = self.processor(
                text=candidate_classes,
                images=pil_image,
                return_tensors="pt",
                padding=True,
            )

            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = (
                outputs.logits_per_image
                if hasattr(outputs, "logits_per_image")
                else outputs.logits
            )
            probs = logits.softmax(dim=1).cpu().numpy()[0]

            results = [
                (candidate_classes[i], float(probs[i]))
                for i in range(len(candidate_classes))
            ]
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        except Exception as e:
            print(f"Classification inference error: {e}")
            return []

    # ------------------------------------------------------------------
    # JSON preset (optional)
    # ------------------------------------------------------------------
    def load_rules_from_json(self, json_path):
        """Load a rules preset from a JSON file.  Returns None on failure."""
        if not json_path or not os.path.exists(json_path):
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Public: build a rules_config dict directly from UI values
    # ------------------------------------------------------------------
    @staticmethod
    def build_config(
        mode="correctness",
        rules=None,
        classes=None,
        overlap_groups=None,
        global_fallback=None,
    ):
        """
        Build a rules_config dict from explicit parameters.

        Args:
            mode:            "correctness", "mislabel", or "both".
            rules:           dict  {class_name: [allowed_label, …]}  (Mode A)
            classes:         list  [top_level_class, …]              (Mode B)
            overlap_groups:  list of lists – classes that are near-synonyms
                             and should not trigger ambiguity warnings.
            global_fallback: extra labels tried when no rule matches.
        """
        return {
            "mode": mode,
            "rules": rules or {},
            "classes": classes or [],
            "overlap_groups": overlap_groups or [],
            "global_fallback": global_fallback or [],
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def refine_annotations(
        self,
        image_array,
        annotations,
        rules_config,
        min_confidence=0.3,
        overlap_margin=0.05,
    ):
        """
        Refine a list of BoundingBox objects using zero-shot classification.

        Updates the ``uncertain`` attribute in-place.  The ``class_name`` is
        intentionally NOT changed — that would hide mislabels from the user.
        Instead, the annotation is flagged ``uncertain=True`` so it shows up
        in the uncertain-frames panel for manual review.

        Args:
            image_array:    BGR numpy array of the current frame.
            annotations:    List of BoundingBox objects.
            rules_config:   Dict returned by ``build_config`` or loaded from JSON.
            min_confidence: Minimum softmax score for the top class.
            overlap_margin: Minimum gap between #1 and #2 scores.
        """
        if not self.model or not rules_config:
            return

        mode = rules_config.get("mode", "correctness")
        rules = rules_config.get("rules", {})
        top_level_classes = rules_config.get("classes", [])
        overlap_groups = rules_config.get("overlap_groups", [])
        global_fallback = rules_config.get("global_fallback", [])

        for bbox in annotations:
            original_class = bbox.class_name
            is_uncertain = False

            # ----------------------------------------------------------
            # Mode A: Correctness check
            # ----------------------------------------------------------
            if mode in ("correctness", "both"):
                candidates = list(rules.get(original_class, []))
                if not candidates:
                    candidates = list(global_fallback)

                if candidates:
                    cropped = self._crop_with_padding(image_array, bbox)
                    results = self._classify_image(cropped, candidates)

                    if results:
                        top1_class, top1_score = results[0]

                        if top1_score < min_confidence:
                            # Try global fallback if we haven't already
                            if candidates != global_fallback and global_fallback:
                                fb_results = self._classify_image(
                                    cropped, global_fallback
                                )
                                if fb_results:
                                    fb_top, fb_score = fb_results[0]
                                    if fb_score >= min_confidence:
                                        results = fb_results
                                        top1_class, top1_score = results[0]
                                    else:
                                        is_uncertain = True
                                else:
                                    is_uncertain = True
                            else:
                                is_uncertain = True

                        if not is_uncertain and len(results) > 1:
                            top2_class, top2_score = results[1]
                            margin = top1_score - top2_score
                            if margin < overlap_margin:
                                # Are these known near-synonyms?
                                are_overlapping = any(
                                    top1_class in grp and top2_class in grp
                                    for grp in overlap_groups
                                )
                                if not are_overlapping:
                                    is_uncertain = True

            # ----------------------------------------------------------
            # Mode B: Mislabel check
            # ----------------------------------------------------------
            if mode in ("mislabel", "both") and not is_uncertain:
                if top_level_classes:
                    cropped = self._crop_with_padding(image_array, bbox)
                    results = self._classify_image(cropped, top_level_classes)

                    if results:
                        top1_class, top1_score = results[0]

                        # Flag if CLIP picks a *different* class with enough margin
                        if (
                            top1_class.lower() != original_class.lower()
                            and top1_score >= min_confidence
                        ):
                            margin = 0.0
                            if len(results) > 1:
                                margin = top1_score - results[1][1]
                            if margin >= overlap_margin:
                                # CLIP is confident it belongs to a different class
                                is_uncertain = True

            bbox.uncertain = is_uncertain
