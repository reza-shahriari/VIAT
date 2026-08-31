"""
Automated Model Inference Runner for VIAT Evaluation Engine

Loads model weights (.pt, .engine, .onnx) using Ultralytics YOLO or OpenCV/ONNX,
runs inference on input video files or image datasets, and formats predictions
into text files for immediate evaluation against Ground Truth.
"""

import os
import cv2
import glob

class ModelRunner:
    """
    Executes model inference on media sources and formats detections for evaluation.
    """

    @staticmethod
    def _process_video(model, video_path, conf_thr, img_size, output_dir):
        media_name = os.path.splitext(os.path.basename(video_path))[0]
        txt_path = os.path.join(output_dir, f"{media_name}.txt")

        cap = cv2.VideoCapture(video_path)
        lines = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=conf_thr, imgsz=img_size, verbose=False)
            frame_preds = []

            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0].cpu().numpy())
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        # Format: [class_id, x1, y1, x2, y2, conf]
                        frame_preds.append(f"[{cls_id}, {int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}, {conf:.4f}]")

            lines.append(";".join(frame_preds) + ";\n")

        cap.release()
        with open(txt_path, "w") as f:
            f.writelines(lines)

    @staticmethod
    def run_inference(weights_path, media_path, conf_thr=0.25, img_size=640, output_dir="/tmp/viat_eval_auto_dt"):
        """
        Run inference on video or image folder using model weights.
        Returns path to directory containing formatted prediction txt files.
        """
        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights file not found: {weights_path}")
        if not media_path or not os.path.exists(media_path):
            raise FileNotFoundError(f"Media path not found: {media_path}")

        os.makedirs(output_dir, exist_ok=True)

        try:
            from ultralytics import YOLO
            model = YOLO(weights_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model weights '{weights_path}': {str(e)}")

        if os.path.isfile(media_path):
            ModelRunner._process_video(model, media_path, conf_thr, img_size, output_dir)

        elif os.path.isdir(media_path):
            # Check if directory contains videos
            video_extensions = ('*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm', '*.mpg', '*.MOV', '*.m4v')
            video_files = []
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(media_path, ext)))
                
            if video_files:
                # Process all videos in the directory
                for v_file in sorted(video_files):
                    ModelRunner._process_video(model, v_file, conf_thr, img_size, output_dir)
            else:
                # Run inference on image directory
                image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(media_path, ext)))
                image_files.extend(glob.glob(os.path.join(media_path, "**", ext), recursive=True))

            image_files = sorted(image_files)
            folder_name = os.path.basename(os.path.normpath(media_path))
            txt_path = os.path.join(output_dir, f"{folder_name}.txt")
            lines = []

            for img_p in image_files:
                img = cv2.imread(img_p)
                if img is None:
                    continue
                results = model.predict(source=img, conf=conf_thr, imgsz=img_size, verbose=False)
                frame_preds = []

                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            cls_id = int(box.cls[0].cpu().numpy())
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            frame_preds.append(f"[{cls_id}, {int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}, {conf:.4f}]")

                lines.append(";".join(frame_preds) + ";\n")

            with open(txt_path, "w") as f:
                f.writelines(lines)

        return output_dir
