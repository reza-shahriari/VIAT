try:
    from viat.evaluation.conf import configs
except ImportError:
    try:
        from ..conf import configs
    except ImportError:
        from conf import configs
import os
import csv
import glob

def create_table(Config: configs, track_check=False, detection_check=False, center_check=False, speed_check=False):
    try:
        if center_check + track_check + detection_check >= 2:
            res_nums = []
            res_names = []
            det_dir = Config.det_path if hasattr(Config, 'det_path') else ""
            if isinstance(det_dir, list) and len(det_dir) > 0:
                det_dir = det_dir[0]
            folder_path = os.path.join(det_dir, 'evaluation_result')
            video_count = len(glob.glob(det_dir + '/*.txt'))

            if detection_check:
                det_metrics = ['Precision', 'Recall', 'F1', 'AP', 'AP50']
                det_eval_path = os.path.join(folder_path, 'eval_detection.csv')
                speed_metrics = ['F1', 'AP50']
                if os.path.exists(det_eval_path):
                    with open(det_eval_path, 'r', encoding='utf-8') as f:
                        det_lines = [line.strip() for line in f if line.strip()]
                    if len(det_lines) >= 2:
                        det_names = [n.strip() for n in det_lines[0].split(',')]
                        det_all_nums = [v.strip() for v in (det_lines[-1] if video_count > 1 else det_lines[1]).split(',')]
                        vid_name = det_all_nums[0] if len(det_all_nums) > 0 else ""

                        for metric in det_metrics:
                            for i in range(len(det_names)):
                                if det_names[i] == metric and i < len(det_all_nums):
                                    res_nums.append(det_all_nums[i])
                                    res_names.append(metric)
                                    break

                        if speed_check:
                            slow_nums, medium_nums, fast_nums = None, None, None
                            for line in det_lines:
                                parts = [p.strip() for p in line.split(',')]
                                if video_count > 1:
                                    if parts[0] == 'slow_all_video': slow_nums = parts
                                    elif parts[0] == 'medium_all_video': medium_nums = parts
                                    elif parts[0] == 'fast_all_video': fast_nums = parts
                                else:
                                    if parts[0] == vid_name + '_slow': slow_nums = parts
                                    elif parts[0] == vid_name + '_medium': medium_nums = parts
                                    elif parts[0] == vid_name + '_fast': fast_nums = parts

                            for metric in speed_metrics:
                                for i in range(len(det_names)):
                                    if det_names[i] == metric:
                                        if fast_nums is not None and i < len(fast_nums):
                                            res_nums.append(fast_nums[i])
                                            res_names.append(metric + '_fast')
                                        if medium_nums is not None and i < len(medium_nums):
                                            res_nums.append(medium_nums[i])
                                            res_names.append(metric + '_medium')
                                        if slow_nums is not None and i < len(slow_nums):
                                            res_nums.append(slow_nums[i])
                                            res_names.append(metric + '_slow')

            if track_check:
                track_eval_path = os.path.join(folder_path, 'eval_tracker_summary.csv')
                track_metrics = ['HOTA', 'MOTA', 'MOTP']
                if os.path.exists(track_eval_path):
                    with open(track_eval_path, 'r', encoding='utf-8') as f:
                        track_lines = [line.strip() for line in f if line.strip()]
                    if len(track_lines) >= 2:
                        track_names = [n.strip() for n in track_lines[0].split(',')]
                        track_nums = [v.strip() for v in track_lines[1].split(',')]
                        for metric in track_metrics:
                            for i in range(len(track_names)):
                                if track_names[i] == metric and i < len(track_nums):
                                    try:
                                        res_nums.append(float(track_nums[i]) / 100)
                                    except Exception:
                                        res_nums.append(track_nums[i])
                                    res_names.append(metric)
                                    break

            if center_check:
                center_metrics = ['F1', 'accuracy']
                pattern = "fall_in_center_evaluate_tracker_margin_*.csv"
                center_files = glob.glob(os.path.join(folder_path, pattern))
                if center_files:
                    with open(center_files[0], 'r', encoding='utf-8') as f:
                        center_lines = [line.strip() for line in f if line.strip()]
                    if len(center_lines) >= 2:
                        center_headers = [h.strip() for h in center_lines[0].split(',')]
                        all_vid = [v.strip() for v in center_lines[-1].split(',')]
                        for metric in center_metrics:
                            for i in range(len(center_headers)):
                                if center_headers[i] == metric and i < len(all_vid):
                                    res_nums.append(all_vid[i])
                                    res_names.append('center_check_' + metric)
                                    break

            if res_names:
                os.makedirs(folder_path, exist_ok=True)
                with open(os.path.join(folder_path, 'combined.csv'), 'w', encoding='utf-8') as f:
                    f.write(','.join(res_names) + '\n')
                    formatted_nums = []
                    for val in res_nums:
                        try:
                            formatted_nums.append("%.2f" % round(float(val), 3))
                        except Exception:
                            formatted_nums.append(str(val))
                    f.write(','.join(formatted_nums) + '\n')
    except Exception as e:
        pass