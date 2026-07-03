import cv2
import os
import argparse
import sys
import json

def parse_raya_header(lines):
    header = []
    classes = []
    in_header = False
    header_end_idx = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        header.append(line)
        if line == "###":
            if not in_header:
                in_header = True
            else:
                header_end_idx = i
                break
        elif in_header:
            if line.startswith("- ") and not line.startswith("-nc:"):
                classes.append(line[2:])
                
    return header, classes, header_end_idx + 1

def process(video_path, txt_path, out_dir, batch_frames, scale=1.0):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    print(f"Reading annotations from {txt_path}...")
    with open(txt_path, "r") as f:
        lines = f.readlines()
        
    header, classes, data_start_idx = parse_raya_header(lines)
    data_lines = [line.strip() for line in lines[data_start_idx:]]
    print(f"Found {len(classes)} classes. Loaded {len(data_lines)} frames of annotations.")
    
    print(f"Opening video {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    print(f"Original video: {orig_w}x{orig_h} @ {fps:.2f}fps, ~{total_frames} frames.")
    if scale != 1.0:
        print(f"Resizing output frames to: {new_w}x{new_h}.")
        
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    batch_idx = 0
    frame_idx = 0
    
    out_txt_lines = []
    current_batch_dir = None
    out_video = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Check if we need to start a new batch
        if frame_idx % batch_frames == 0:
            if current_batch_dir is not None:
                if out_video is not None:
                    out_video.release()
                    
                # Save previous batch txt and json
                out_txt_path = os.path.join(current_batch_dir, f"dataset_video.txt")
                with open(out_txt_path, "w") as f:
                    f.write("\n".join(header) + "\n")
                    f.write("\n".join(out_txt_lines) + "\n")
                
                size_path = os.path.join(current_batch_dir, "dataset_video_metadata.json")
                sizes_dict = {str(i): [new_h, new_w] for i in range(len(out_txt_lines))}
                with open(size_path, "w") as f:
                    json.dump({"original_sizes": sizes_dict}, f, indent=2)
                    
                print(f"Saved batch {batch_idx}: {current_batch_dir}")
                batch_idx += 1
                
            current_batch_dir = os.path.join(out_dir, f"{base_name}_batch_{batch_idx:03d}")
            os.makedirs(current_batch_dir, exist_ok=True)
            out_txt_lines = []
            print(f"Writing batch {batch_idx} to {current_batch_dir}...")
            
            video_path = os.path.join(current_batch_dir, "dataset_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(video_path, fourcc, fps, (new_w, new_h))
            
        if scale != 1.0:
            frame = cv2.resize(frame, (new_w, new_h))
            
        # Save frame to video
        out_video.write(frame)
        
        # Process annotation line if exists
        if frame_idx < len(data_lines):
            line = data_lines[frame_idx]
            if line and line != "[]":
                # scale bounding boxes
                if scale != 1.0:
                    new_annotations = []
                    # Format: [class,x,y,w,h,...];...
                    ann_strs = line.split(";")
                    for ann_str in ann_strs:
                        ann_str = ann_str.strip()
                        if not ann_str or ann_str == "[]": continue
                        
                        inner = ann_str.strip("[]")
                        parts = inner.split(",")
                        if len(parts) >= 5:
                            cls_id = parts[0]
                            # x, y, w, h
                            x = int(float(parts[1]) * scale)
                            y = int(float(parts[2]) * scale)
                            w = int(float(parts[3]) * scale)
                            h = int(float(parts[4]) * scale)
                            
                            new_parts = [cls_id, str(x), str(y), str(w), str(h)] + parts[5:]
                            new_annotations.append("[" + ",".join(new_parts) + "]")
                    line = ";".join(new_annotations) + ";" if new_annotations else "[]"
            out_txt_lines.append(line)
        else:
            out_txt_lines.append("[]")
            
        frame_idx += 1
        
    if current_batch_dir is not None:
        if out_video is not None:
            out_video.release()
            
        out_txt_path = os.path.join(current_batch_dir, f"annotations.txt")
        with open(out_txt_path, "w") as f:
            f.write("\n".join(header) + "\n")
            f.write("\n".join(out_txt_lines) + "\n")
            
        size_path = os.path.join(current_batch_dir, "dataset_video_metadata.json")
        sizes_dict = {str(i): [new_h, new_w] for i in range(len(out_txt_lines))}
        with open(size_path, "w") as f:
            json.dump({"original_sizes": sizes_dict}, f, indent=2)
            
        print(f"Saved batch {batch_idx}: {current_batch_dir}")
        
    cap.release()
    print("All batches processed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Raya annotated video into batches of frames")
    parser.add_argument("--video", type=str, default=r"c:\Users\Player One\Pictures\R-M-S-U.v4-rfdetr-medium.yolov11\dataset_video.mp4", help="Path to input video")
    parser.add_argument("--txt", type=str, default=r"c:\Users\Player One\Pictures\R-M-S-U.v4-rfdetr-medium.yolov11\dataset_video.txt", help="Path to Raya annotation text file")
    parser.add_argument("--out_dir", type=str, default=r"c:\Users\Player One\Pictures\R-M-S-U.v4-rfdetr-medium.yolov11\batches", help="Directory to save output batches")
    parser.add_argument("--batch_frames", type=int, default=500, help="Number of frames per batch (default: 1500)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor to resize frames and annotations")
    
    args = parser.parse_args()
    process(args.video, args.txt, args.out_dir, args.batch_frames, args.scale)
