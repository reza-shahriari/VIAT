import os
import cv2
import numpy as np

try:
    import torch
    from ultralytics import YOLOWorld
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ZeroShotDetector:
    def load_model(self, model_type, checkpoints_dir):
        raise NotImplementedError
        
    def set_classes(self, classes_list):
        raise NotImplementedError
        
    def predict(self, image_array):
        """Returns [{'box': [x1,y1,x2,y2], 'class_name': str, 'score': float}, ...]"""
        raise NotImplementedError


class YoloWorldDetector(ZeroShotDetector):
    def __init__(self):
        self.model = None
        self.classes = []

    def load_model(self, model_type, checkpoints_dir):
        if not ULTRALYTICS_AVAILABLE:
            return False, "Ultralytics is not installed. Please run: pip install ultralytics"
            
        try:
            model_path = os.path.join(checkpoints_dir, model_type)
            if not os.path.exists(model_path):
                old_cwd = os.getcwd()
                os.chdir(checkpoints_dir)
                try:
                    self.model = YOLOWorld(model_type)
                finally:
                    os.chdir(old_cwd)
            else:
                self.model = YOLOWorld(model_path)
            return True, "Model loaded successfully"
        except Exception as e:
            return False, f"Failed to load YOLO-World: {str(e)}"

    def set_classes(self, classes_list):
        if self.model:
            self.classes = classes_list
            self.model.set_classes(classes_list)

    def predict(self, image_array):
        if not self.model:
            return []
            
        results = self.model(image_array, verbose=False)
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                score = float(box.conf[0].cpu().numpy())
                
                class_name = self.classes[cls_id] if cls_id < len(self.classes) else f"class_{cls_id}"
                
                detections.append({
                    'box': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    'class_name': class_name,
                    'score': score
                })
        return detections

class GroundingDinoDetector(ZeroShotDetector):
    def __init__(self):
        self.model = None
        self.processor = None
        self.classes = []
        self.text_prompt = ""

    def load_model(self, model_type, checkpoints_dir):
        if not TRANSFORMERS_AVAILABLE:
            return False, "Transformers is not installed. Please run: pip install -r dino_req.txt"
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_type)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_type)
            
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            
            return True, "Model loaded successfully"
        except Exception as e:
            return False, f"Failed to load Grounding DINO: {str(e)}"

    def set_classes(self, classes_list):
        self.classes = classes_list
        self.text_prompt = " . ".join(classes_list) + " ."

    def predict(self, image_array):
        if not self.model or not self.processor or not self.classes:
            return []
            
        try:
            from PIL import Image
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            inputs = self.processor(images=pil_image, text=self.text_prompt, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.3,
                text_threshold=0.25,
                target_sizes=[pil_image.size[::-1]]
            )
            
            detections = []
            if len(results) > 0:
                res = results[0]
                boxes = res["boxes"].cpu().numpy()
                scores = res["scores"].cpu().numpy()
                labels = res["labels"]
                
                for box, score, label in zip(boxes, scores, labels):
                    detections.append({
                        'box': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        'class_name': label,
                        'score': float(score)
                    })
            return detections
        except Exception as e:
            print(f"Grounding DINO inference error: {e}")
            return []

class Florence2Detector(ZeroShotDetector):
    def __init__(self):
        self.model = None
        self.processor = None
        self.classes = []
        self.text_prompt = ""

    def load_model(self, model_type, checkpoints_dir):
        if not TRANSFORMERS_AVAILABLE:
            return False, "Transformers is not installed. Please run: pip install -r florence_req.txt"
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_type, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_type, trust_remote_code=True)
            
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                
            return True, "Model loaded successfully"
        except Exception as e:
            return False, f"Failed to load Florence-2: {str(e)}"

    def set_classes(self, classes_list):
        self.classes = classes_list
        class_text = ", ".join(classes_list)
        self.text_prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {class_text}"

    def predict(self, image_array):
        if not self.model or not self.processor or not self.classes:
            return []
            
        try:
            from PIL import Image
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            inputs = self.processor(text=self.text_prompt, images=pil_image, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
                
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task="<CAPTION_TO_PHRASE_GROUNDING>", 
                image_size=(pil_image.width, pil_image.height)
            )
            
            detections = []
            if "<CAPTION_TO_PHRASE_GROUNDING>" in parsed_answer:
                res = parsed_answer["<CAPTION_TO_PHRASE_GROUNDING>"]
                bboxes = res.get("bboxes", [])
                labels = res.get("labels", [])
                
                for box, label in zip(bboxes, labels):
                    detections.append({
                        'box': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        'class_name': label,
                        'score': 1.0
                    })
            return detections
        except Exception as e:
            print(f"Florence-2 inference error: {e}")
            return []

class ZeroShotManager:
    """
    Manager for zero-shot object detection, capable of supporting multiple models (YOLO-World, DINO, etc.)
    """
    def __init__(self, checkpoints_dir="checkpoints"):
        self.checkpoints_dir = checkpoints_dir
        
        import sys
        if getattr(sys, 'frozen', False):
            project_root = os.path.dirname(sys.executable)
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
        self.chkp_dir = os.path.join(project_root, self.checkpoints_dir)
        if not os.path.exists(self.chkp_dir):
            os.makedirs(self.chkp_dir, exist_ok=True)
            
        self.detector = None
        self.current_model_type = None

    def is_available(self):
        return ULTRALYTICS_AVAILABLE

    def load_model(self, model_type="yolov8s-world.pt"):
        if self.current_model_type == model_type and self.detector is not None:
            return True, "Model already loaded"
            
        if "yolo" in model_type.lower():
            self.detector = YoloWorldDetector()
        elif "dino" in model_type.lower():
            self.detector = GroundingDinoDetector()
        elif "florence" in model_type.lower():
            self.detector = Florence2Detector()
        else:
            return False, f"Unknown zero-shot model type: {model_type}"
            
        success, msg = self.detector.load_model(model_type, self.chkp_dir)
        if success:
            self.current_model_type = model_type
        return success, msg

    def set_classes(self, classes_list):
        if self.detector:
            self.detector.set_classes(classes_list)

    def predict(self, image_array):
        if not self.detector:
            return []
        return self.detector.predict(image_array)
