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
    Manager for zero-shot object classification (e.g., using CLIP or SigLIP).
    Refines bounding boxes by cropping and re-evaluating their classes.
    """
    def __init__(self, checkpoints_dir="checkpoints"):
        self.checkpoints_dir = checkpoints_dir
        self.model = None
        self.processor = None
        self.current_model_type = None
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    def is_available(self):
        return check_transformers()

    def load_model(self, model_type="openai/clip-vit-base-patch32"):
        """
        Load a HuggingFace CLIP or SigLIP model.
        """
        if self.current_model_type == model_type and self.model is not None:
            return True, "Model already loaded"
            
        if not self.is_available():
            return False, "Transformers is not installed. Please run: pip install transformers"
            
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotImageClassification, CLIPModel, CLIPProcessor
            
            # Define local path in checkpoints folder
            safe_name = model_type.replace('/', '_')
            local_path = os.path.join(self.checkpoints_dir, "zero_shot", safe_name)
            
            # Check if local model files exist
            is_local = os.path.exists(os.path.join(local_path, "config.json"))
            
            if is_local:
                try:
                    # Attempt to load locally first without checking HuggingFace Hub
                    if "clip" in model_type.lower() and "siglip" not in model_type.lower():
                        self.model = CLIPModel.from_pretrained(local_path, local_files_only=True)
                        self.processor = AutoProcessor.from_pretrained(local_path, local_files_only=True)
                    else:
                        self.model = AutoModelForZeroShotImageClassification.from_pretrained(local_path, local_files_only=True)
                        self.processor = AutoProcessor.from_pretrained(local_path, local_files_only=True)
                    self.current_model_type = model_type
                    if self.device == "cuda":
                        self.model = self.model.to(self.device)
                    return True, "Loaded local model successfully"
                except Exception as e:
                    # Local load failed, proceed to download
                    print(f"Failed to load local model from {local_path}, downloading... Details: {e}")
            
            # If not local, download to a temporary cache inside the checkpoints folder
            temp_cache = os.path.join(self.checkpoints_dir, "zero_shot", "temp_cache")
            os.makedirs(temp_cache, exist_ok=True)
            
            try:
                # Use AutoModel if it's a standard HF model
                if "clip" in model_type.lower() and "siglip" not in model_type.lower():
                    self.model = CLIPModel.from_pretrained(model_type, cache_dir=temp_cache)
                    self.processor = AutoProcessor.from_pretrained(model_type, cache_dir=temp_cache)
                else:
                    self.model = AutoModelForZeroShotImageClassification.from_pretrained(model_type, cache_dir=temp_cache)
                    self.processor = AutoProcessor.from_pretrained(model_type, cache_dir=temp_cache)
            except Exception:
                # Fallback for standard CLIP
                self.processor = CLIPProcessor.from_pretrained(model_type, cache_dir=temp_cache)
                self.model = CLIPModel.from_pretrained(model_type, cache_dir=temp_cache)
                
            # Save a clean copy to local_path so it can be loaded locally next time
            os.makedirs(local_path, exist_ok=True)
            self.processor.save_pretrained(local_path)
            self.model.save_pretrained(local_path)
            
            # Clean up temporary HF cache to save disk space
            import shutil
            try:
                shutil.rmtree(temp_cache)
            except Exception as e:
                print(f"Warning: Could not remove temp HF cache: {e}")
                
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                
            self.current_model_type = model_type
            return True, "Classification model downloaded and loaded successfully"
        except Exception as e:
            return False, f"Failed to load classification model: {str(e)}"

    def _crop_with_padding(self, image_array, box, padding_percent=0.15):
        """
        Crop a bounding box from the image with a contextual margin.
        """
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
        # if crop is empty (invalid box), return a 1x1 black pixel
        if cropped.size == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return cropped

    def _classify_image(self, cropped_image, candidate_classes):
        """
        Run zero-shot classification on a cropped image against a list of text candidate classes.
        Returns a sorted list of (class_name, confidence_score) tuples.
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
                padding=True
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # CLIP outputs image-text similarity logits
            logits_per_image = outputs.logits_per_image if hasattr(outputs, 'logits_per_image') else outputs.logits
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
            results = [(candidate_classes[i], float(probs[i])) for i in range(len(candidate_classes))]
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        except Exception as e:
            print(f"Classification inference error: {e}")
            return []

    def load_rules_from_json(self, json_path):
        """
        Load hierarchy and overlap rules from a JSON file.
        """
        if not os.path.exists(json_path):
            return None
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def refine_annotations(self, image_array, annotations, rules_config, min_confidence=0.3, overlap_margin=0.05):
        """
        Refine a list of BoundingBox objects using zero-shot classification.
        Updates the class_name and uncertain attributes in-place.
        """
        if not self.model or not rules_config:
            return
            
        rules = rules_config.get("rules", {})
        overlap_groups = rules_config.get("overlap_groups", [])
        global_fallback = rules_config.get("global_fallback", [])
        
        for bbox in annotations:
            original_class = bbox.class_name
            
            candidates = []
            if original_class in rules:
                candidates = rules[original_class]
            
            if not candidates:
                # No specific rule for this class, maybe use global fallback?
                candidates = global_fallback
                
            if not candidates:
                continue
                
            cropped = self._crop_with_padding(image_array, bbox)
            results = self._classify_image(cropped, candidates)
            
            if not results:
                continue
                
            top1_class, top1_score = results[0]
            
            # Handle Uncertainty
            is_uncertain = False
            
            if top1_score < min_confidence:
                # Top confidence is too low. Try global fallback if we haven't already.
                if candidates != global_fallback and global_fallback:
                    fallback_results = self._classify_image(cropped, global_fallback)
                    if fallback_results:
                        f_top1_class, f_top1_score = fallback_results[0]
                        if f_top1_score >= min_confidence:
                            results = fallback_results
                            top1_class, top1_score = results[0]
                        else:
                            is_uncertain = True
                else:
                    is_uncertain = True
            
            if not is_uncertain and len(results) > 1:
                top2_class, top2_score = results[1]
                margin = top1_score - top2_score
                
                # If margin is very small, they might be overlapping classes (e.g. Truck vs Vehicle)
                if margin < overlap_margin:
                    # Check if they belong to the same overlap group
                    are_overlapping = False
                    for group in overlap_groups:
                        if top1_class in group and top2_class in group:
                            are_overlapping = True
                            break
                    if not are_overlapping:
                        is_uncertain = True
            
            if is_uncertain:
                bbox.uncertain = True
                # Optional: Revert to original class if we are completely uncertain
                # bbox.class_name = original_class
            else:
                bbox.class_name = top1_class
                bbox.uncertain = False
