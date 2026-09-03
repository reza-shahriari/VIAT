#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import csv
import json
import shutil
import pathlib
import time
import re
import copy
from collections import defaultdict
import cv2
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    from rich.console import Console
    from rich.progress import Progress
    cons = Console()
except ImportError:
    class DummyStatus:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class DummyConsole:
        def print(self, *args, **kwargs):
            clean_args = [re.sub(r'\[/?bold.*?\]', '', str(a)) for a in args]
            print(*clean_args)
        def status(self, *args, **kwargs):
            return DummyStatus()
    cons = DummyConsole()
    Progress = None

try:
    from viat.evaluation.utils.converter import coco2bb
    from viat.evaluation.evaluators import coco_evaluator
    from viat.evaluation.utils.enumerators import BBType
    from viat.evaluation.utils.convert_ourformat2json import convert_to_json
    from viat.evaluation.utils.convert_ourformat2mot import convert_to_mot
    from viat.evaluation.utils.write_combined_results import create_table
    from viat.evaluation.conf.configs import config
    from viat.evaluation.evaluators.coco_evaluator import jaccard
    from viat.evaluation.utils.plotter import plot_map_by_class, plot_map_by_size
except ImportError:
    from .utils.converter import coco2bb
    from .evaluators import coco_evaluator
    from .utils.enumerators import BBType
    from .utils.convert_ourformat2json import convert_to_json
    from .utils.convert_ourformat2mot import convert_to_mot
    from .utils.write_combined_results import create_table
    from .conf.configs import config
    from .evaluators.coco_evaluator import jaccard
    from .utils.plotter import plot_map_by_class, plot_map_by_size

try:
    from viat.evaluation.trackEval.scripts.our_run_mot_challenge import initialize as run_tracker
except ImportError:
    run_tracker = None


class Evaluate():
    def __init__(self,gt_path,det_path,category_names,quality_thr,
                 size_thr,margin_check,margin_value,center_check,
                 detection_check,tracking_check,speed_check,visualize_check,
                 combine = True,visualize_iou = 0.5,ignore_all = True,center_conf_thr = None,
                 class_mapping=None, ignored_videos=None, video_class_mappings=None, video_category_mappings=None, target_classes=None) :
        self.exts = ['*.mp4','*.avi','*.mkv','*.mov','*.webm','*.mpg', '*.MOV','*.m4v']
        self.gt_path = gt_path if isinstance(gt_path,list) else [gt_path]
        self.det_path = det_path if isinstance(det_path, list) else [det_path]
        self.quality_thr = quality_thr
        self.size_thr = size_thr
        self.margin_check = margin_check
        self.margin_value = margin_value
        self.center_check = center_check
        self.detection_check = detection_check
        self.write_detection_check_res = True
        self.tracking_check = tracking_check
        self.speed_check = speed_check
        self.combine_check = combine
        self.visualize_iou = visualize_iou
        self.visualize_check = visualize_check
        self.ignore_all = ignore_all
        self.center_conf_thr = center_conf_thr
        self.category_names = category_names
        self.target_classes = target_classes
        self.class_mapping = class_mapping
        self.ignored_videos = ignored_videos
        self.video_class_mappings = video_class_mappings
        self.video_category_mappings = video_category_mappings
        if self.center_conf_thr is None and self.detection_check is False:
            self.detection_test = True
            self.write_detection_check_res = False
    def concatenate_images(self,img_list, axis):
        """
        Concatenates a list of images along a given axis.
        
        Parameters:
            img_list (list): List of images (as numpy arrays).
            axis (int): Axis along which the images are to be concatenated. 
                        Use 0 for vertical and 1 for horizontal concatenation.
        
        Returns:
            result: The concatenated image.
        """
        # Find maximum dimensions
        max_height = max(img.shape[0] for img in img_list)
        max_width = max(img.shape[1] for img in img_list)

        # Pad images to maximum dimensions
        padded_imgs = [cv2.copyMakeBorder(img, 
                                        top=0, 
                                        bottom=max_height - img.shape[0] if axis == 1 else 0, 
                                        left=0, 
                                        right=max_width - img.shape[1] if axis == 0 else 0, 
                                        borderType=cv2.BORDER_CONSTANT, 
                                        value=[0, 0, 0]) for img in img_list]

        # Concatenate images
        result = np.concatenate(padded_imgs, axis=axis)

        return result
    
    def visualize_detect(self):
        opt_score = 0.5
        try:
            if hasattr(self, 'df') and self.df is not None:
                if hasattr(self.df, 'iloc') and len(self.df) > 0 and 'Score' in self.df.columns:
                    val = self.df.iloc[-1]['Score']
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        opt_score = float(val)
                elif isinstance(self.df, list) and len(self.df) > 0 and isinstance(self.df[-1], dict) and 'Score' in self.df[-1]:
                    val = self.df[-1]['Score']
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        opt_score = float(val)
        except Exception:
            opt_score = 0.5
        with cons.status('[bold purple] Visualizing') as status:
            for txt_path in self.gt_path:
                for gt_file in glob.glob(txt_path+'/*.txt'):
                    name = os.path.splitext(os.path.split(gt_file)[1])[0]
                    with open(gt_file, 'r') as gfile:
                        gt_lines = gfile.readlines()
                    for dets_txt_path in self.det_path:
                        if os.path.exists(dets_txt_path+'/'+name+'.txt'):
                            with open(dets_txt_path+'/'+name+'.txt', 'r') as file:
                                dt_lines = file.readlines()
                            os.makedirs(os.path.join(dets_txt_path,'visualize'), exist_ok=True)
                            for ext in self.exts:
                                if os.path.exists(os.path.join(txt_path,name+ext.strip('*'))):
                                    video = cv2.VideoCapture(os.path.join(txt_path,name+ext.strip('*')))
                                    fps:int = int(video.get(5))
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                    factor:int = 1
                                    out_FP = cv2.VideoWriter(os.path.join(dets_txt_path,'visualize',name+'_FP.mp4'), fourcc, fps, (int(video.get(3)),int(video.get(4))))
                                    out_FN = cv2.VideoWriter(os.path.join(dets_txt_path,'visualize',name+'_FN.mp4'), fourcc, fps, (int(video.get(3)),int(video.get(4))))
                                    img_list_fp = []
                                    img_list_fn = []
                                    for i, gt_line in enumerate(gt_lines):
                                        dt_raw = dt_lines[i].strip().rstrip(';').strip() if i < len(dt_lines) else ""
                                        try:
                                            dt_line = eval(dt_raw) if dt_raw else []
                                        except Exception:
                                            dt_line = []
                                        if not isinstance(dt_line, list):
                                            dt_line = []
                                        if len(dt_line) > 0 and not isinstance(dt_line[0], list):
                                            dt_line = [dt_line]

                                        s, frame = video.read()
                                        if not s:
                                            break
                                        frame_FN = copy.deepcopy(frame)
                                        frame_FP = copy.deepcopy(frame)
                                        frame_show = copy.deepcopy(frame)
                                        for sub_dt_line in dt_line:
                                            flg = False
                                            score = sub_dt_line[5] if len(sub_dt_line) > 5 else 1.0
                                            if score >= opt_score: # consider all_data['Score'] as optimum score.
                                                gt_list = [p.strip() for p in gt_line.split(';') if p.strip()]
                                                new_gt_line = []
                                                for sub_gt_line_base in list(gt_list):
                                                    try:
                                                        sub_gt_line = eval(sub_gt_line_base)
                                                        sub_gt_line = [sub_gt_line[1], sub_gt_line[2], sub_gt_line[1]+sub_gt_line[3], sub_gt_line[2]+sub_gt_line[4]]
                                                        iou = jaccard(sub_gt_line, sub_dt_line[1:5])
                                                        if iou > self.visualize_iou:
                                                            flg = True
                                                            gt_list = [item for item in gt_list if item != sub_gt_line_base]
                                                    except Exception:
                                                        continue
                                                    else:
                                                        new_gt_line.append([sub_gt_line_base, iou])
                                                if flg:
                                                    sub_dt_line = [int(i) for i in sub_dt_line[1:5]]
                                                    cv2.rectangle(frame_show, tuple(sub_dt_line[0:2]), tuple(sub_dt_line[2:]), (0,255,0), 1, 1)
                                                else:
                                                    for sub_new_gt_line in new_gt_line:
                                                        iou = sub_new_gt_line[1]
                                                        sub_new_gt_line = eval(sub_new_gt_line[0])
                                                        sub_new_gt_line = [sub_new_gt_line[1], sub_new_gt_line[2], sub_new_gt_line[1]+sub_new_gt_line[3], sub_new_gt_line[2]+sub_new_gt_line[4], sub_new_gt_line[5], sub_new_gt_line[6]]
                                                        bb = [int(i) for i in sub_new_gt_line]
                                                        cv2.rectangle(frame_FP, (bb[0], bb[1]),(bb[2], bb[3]), (3,186,255), 1, 1)
                                                        cv2.putText(frame_FP, f'IOU: {round(iou,3)}, sz: {sub_new_gt_line[4]}, Q: {sub_new_gt_line[5]}', (bb[0], bb[3]+5), 0, 0.3, (3,186,255), 1)
                                                    sub_dt_line = [int(i) for i in sub_dt_line[1:5]]
                                                    cv2.rectangle(frame_FP, tuple(sub_dt_line[0:2]), tuple(sub_dt_line[2:]), (0,0,255), 1, 1)
                                                    cv2.putText(frame_FP, f'Sc: {score}', (sub_dt_line[0], sub_dt_line[1]-5), 0, 1, (0, 0, 255), 2)
                                                    cv2.putText(frame_FP, f'Frame Num: {i+1}', (1, 20), 0, 0.5, (0, 0, 255), 1)
                                                    fp_cropped = frame_FP[sub_dt_line[1]:sub_dt_line[3],sub_dt_line[0]:sub_dt_line[2]]
                                                    cv2.putText(fp_cropped, f'Frame Num: {i+1}', (1, 10), 0, 0.1, (0, 0, 255), 1)
                                                    img_list_fp.append(fp_cropped)
                                                    out_FP.write(frame_FP)
                                                for sub_gt_line in gt_list:
                                                    sub_gt_line = eval(sub_gt_line)
                                                    sub_gt_line = [sub_gt_line[1], sub_gt_line[2], sub_gt_line[1]+sub_gt_line[3], sub_gt_line[2]+sub_gt_line[4]]
                                                    fn_cropped = frame_FN[max(0, int(sub_gt_line[1])):int(min(sub_gt_line[3], frame_FN.shape[0])),max(0, int(sub_gt_line[0])):int(min(sub_gt_line[2], frame_FN.shape[1]))]
                                                    cv2.putText(fn_cropped, f'Frame Num: {i+1}', (1, 10), 0, 0.1, (255, 0, 0), 1)
                                                    if fn_cropped.shape[1] == 0 or fn_cropped.shape[0] == 0:
                                                        print(fn_cropped.shape, sub_gt_line, frame_FN.shape, gt_list, i, name)
                                                    else:
                                                        img_list_fn.append(fn_cropped)
                                                    cv2.rectangle(frame_FN, (int(sub_gt_line[0]), int(sub_gt_line[1])), (int(sub_gt_line[2]),int(sub_gt_line[3])), (255,0,0), 1, 1)
                                                    cv2.putText(frame_FN, 'FN', (int(sub_gt_line[0]), int(sub_gt_line[1]-5)), 0, 1, (255, 0, 0), 2)
                                                    cv2.putText(frame_FN, f'Frame Num: {i+1}', (1, 20), 0, 0.5, (255, 0, 0), 1)
                                                    out_FN.write(frame_FN)
                                        cv2.imshow('show', frame_show)
                                        cv2.waitKey(1)
                                    # if len(img_list_fp):
                                        # img_fp = self.concatenate_images(img_list_fp, 1)
                                        # cv2.imwrite(os.path.join(dets_txt_path,'visualize',name+'_FP.png'), img_fp)
                                    # if len(img_list_fn):
                                        # img_fn = self.concatenate_images(img_list_fn, 1)
                                        # cv2.imwrite(os.path.join(dets_txt_path,'visualize',name+'_FN.png'), img_fn)
                                    out_FN.release()
                                    out_FP.release()
                                    file.close()
                                    gfile.close()
        
    def eval_track(self, Config):
        if run_tracker is None:
            cons.print("[yellow]Warning: Tracking evaluation module (TrackEval) is not available, skipping tracking evaluation.[/yellow]")
            return

        gt_paths = self.gt_path if isinstance(self.gt_path, list) else [self.gt_path]
        det_paths = self.det_path if isinstance(self.det_path, list) else [self.det_path]

        try:
            if len(gt_paths) > 1 and len(det_paths) > 1:
                Config.stadium_downloads = ['_all', '_download', '_stadium']
                for i, j in zip(gt_paths, det_paths):
                    convert_to_mot([i], [j], Config, self.size_thr, self.quality_thr)
                    run_tracker(Config)
                convert_to_mot(gt_paths, det_paths, Config, self.size_thr, self.quality_thr)
                run_tracker(Config)
            else:
                Config.stadium_downloads = []
                with cons.status('[bold blue] writing mot format files ') as status:
                    convert_to_mot(gt_paths, det_paths, Config, self.size_thr, self.quality_thr)
                cons.print("[bold cyan]mot format files created")
                with cons.status('[bold green] evaluating tracker') as status:
                    run_tracker(Config)

                track_tmp = os.path.join(gt_paths[0], 'Track')
                if os.path.exists(track_tmp):
                    shutil.rmtree(track_tmp, ignore_errors=True)
                cons.print('[bold cyan] tracker evaluated successfully :smiley:')
        except Exception as e:
            cons.print(f"[bold red]Tracking evaluation error:[/bold red] {e}")
            logger.exception(f"Error during tracking evaluation: {e}")
    
    def _get_category_id_to_name(self):
        """Build the real class-id -> class-name mapping from the COCO jsons
        written by convert_to_json (falls back to target_classes order)."""
        id_to_name = {}
        for folder in list(self.gt_path) + list(self.det_path):
            if not folder or not os.path.isdir(folder):
                continue
            for j_path in glob.glob(os.path.join(folder, '*.json')):
                try:
                    with open(j_path, 'r') as f:
                        cats = json.load(f).get('categories', [])
                except Exception:
                    continue
                for c in cats:
                    try:
                        cid = int(c.get('id'))
                    except (TypeError, ValueError):
                        continue
                    name = c.get('name')
                    if name is None:
                        continue
                    name = str(name)
                    # Prefer a real (non-numeric) name over a str(id) fallback
                    if cid not in id_to_name or id_to_name[cid].isdigit():
                        id_to_name[cid] = name
        for cid, name in list(id_to_name.items()):
            if name.isdigit():
                if isinstance(self.target_classes, list):
                    if 0 <= cid < len(self.target_classes):
                        id_to_name[cid] = str(self.target_classes[cid])
                    elif 0 <= cid - 1 < len(self.target_classes):
                        id_to_name[cid] = str(self.target_classes[cid - 1])
                elif isinstance(self.category_names, list):
                    if 0 <= cid < len(self.category_names):
                        id_to_name[cid] = str(self.category_names[cid])
                    elif 0 <= cid - 1 < len(self.category_names):
                        id_to_name[cid] = str(self.category_names[cid - 1])
        if not id_to_name:
            if isinstance(self.target_classes, list):
                id_to_name = {i: str(n) for i, n in enumerate(self.target_classes)}
            elif isinstance(self.category_names, list):
                id_to_name = {i: str(n) for i, n in enumerate(self.category_names)}
        return id_to_name

    def _compute_detection_diagnostics(self, out_folder, iou_thr=0.5):
        """Match detections against GT over the per-video jsons and persist
        per-detection records (confusion matrix, IoUs, calibration sweep,
        aspect-ratio bias, spatial error coords) for the diagnostic plots."""
        gt_paths = [self.gt_path] if isinstance(self.gt_path, str) else self.gt_path
        det_paths = [self.det_path] if isinstance(self.det_path, str) else self.det_path
        id_to_name = self._get_category_id_to_name()

        # Only per-video jsons: aggregates would double-count detections
        skip_names = {'all_video'}
        if self.category_names:
            for cat in self.category_names:
                if cat != "default":
                    skip_names.add(cat)

        def _is_aggregate(name):
            return (name in skip_names or name.endswith('_all_video')
                    or name.endswith(('_slow', '_medium', '_fast')))

        matched_records = []
        fp_records = []
        fn_records = []
        computed_video_metrics = []
        per_video_records = {}
        canvas = [0, 0]

        for det_dir in det_paths:
            for gt_dir in gt_paths:
                for det_json_path in glob.glob(os.path.join(det_dir, '*.json')):
                    name = os.path.splitext(os.path.split(det_json_path)[1])[0]
                    if _is_aggregate(name):
                        continue
                    gt_json_path = os.path.join(gt_dir, name + '.json')
                    if not os.path.exists(gt_json_path):
                        continue
                    try:
                        with open(gt_json_path, 'r') as f:
                            gt_data = json.load(f)
                        with open(det_json_path, 'r') as f:
                            det_data = json.load(f)
                    except Exception:
                        continue

                    for img in det_data.get('images', []) + gt_data.get('images', []):
                        canvas[0] = max(canvas[0], int(img.get('width') or 0))
                        canvas[1] = max(canvas[1], int(img.get('height') or 0))

                    gts_by_img = defaultdict(list)
                    for ann in gt_data.get('annotations', []):
                        if ann.get('ignore'):
                            continue
                        gts_by_img[ann.get('image_id')].append(ann)

                    dets_by_img = defaultdict(list)
                    for ann in det_data.get('annotations', []):
                        dets_by_img[ann.get('image_id')].append(ann)

                    v_matched = []
                    v_fp = []
                    v_fn = []

                    for img_id, dets in dets_by_img.items():
                        gts = gts_by_img.get(img_id, [])
                        gt_boxes = [a['bbox'] for a in gts]
                        gt_xy = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in gt_boxes]
                        gt_used = [False] * len(gts)

                        for d in sorted(dets, key=lambda a: a.get('score', 0), reverse=True):
                            db = d['bbox']
                            dxy = [db[0], db[1], db[0] + db[2], db[1] + db[3]]
                            d_cls = id_to_name.get(d.get('category_id'), str(d.get('category_id')))
                            cx = db[0] + db[2] / 2.0
                            cy = db[1] + db[3] / 2.0
                            best_iou, best_g = 0.0, -1
                            for gi, gxy in enumerate(gt_xy):
                                if gt_used[gi]:
                                    continue
                                iou = jaccard(dxy, gxy)
                                if iou > best_iou:
                                    best_iou, best_g = iou, gi
                            if best_g >= 0 and best_iou >= iou_thr:
                                gt_used[best_g] = True
                                g_cls = id_to_name.get(gts[best_g].get('category_id'),
                                                       str(gts[best_g].get('category_id')))
                                rec = {'gt': g_cls, 'det': d_cls,
                                       'iou': float(best_iou),
                                       'score': float(d.get('score', 0)),
                                       'cx': cx, 'cy': cy, 'w': db[2], 'h': db[3]}
                                matched_records.append(rec)
                                v_matched.append(rec)
                            else:
                                rec = {'det': d_cls, 'score': float(d.get('score', 0)),
                                       'cx': cx, 'cy': cy, 'w': db[2], 'h': db[3],
                                       'max_iou': float(best_iou)}
                                fp_records.append(rec)
                                v_fp.append(rec)
                        for gi, used in enumerate(gt_used):
                            if not used:
                                gb = gt_boxes[gi]
                                g_cls = id_to_name.get(gts[gi].get('category_id'),
                                                       str(gts[gi].get('category_id')))
                                rec = {'gt': g_cls, 'cx': gb[0] + gb[2] / 2.0,
                                       'cy': gb[1] + gb[3] / 2.0, 'w': gb[2], 'h': gb[3]}
                                fn_records.append(rec)
                                v_fn.append(rec)

                    # Compute real detection metrics for this video sequence
                    v_tp_count = len(v_matched)
                    v_fp_count = len(v_fp)
                    v_fn_count = len(v_fn)
                    v_total_gt = v_tp_count + v_fn_count
                    v_p = v_tp_count / (v_tp_count + v_fp_count) if (v_tp_count + v_fp_count) > 0 else 0.0
                    v_r = v_tp_count / v_total_gt if v_total_gt > 0 else 0.0
                    v_f1 = 2 * v_p * v_r / (v_p + v_r) if (v_p + v_r) > 0 else 0.0

                    v_dets = v_matched + v_fp
                    v_ap50 = 0.0
                    if v_dets and v_total_gt > 0:
                        v_confs = np.array([r['score'] for r in v_dets])
                        v_is_tp = np.array([True] * len(v_matched) + [False] * len(v_fp))
                        v_precs, v_recs = [], []
                        for t in np.arange(0.05, 1.0, 0.05):
                            sel = v_confs >= t
                            tp_s = int(v_is_tp[sel].sum())
                            n_s = int(sel.sum())
                            v_precs.append(tp_s / n_s if n_s else 0.0)
                            v_recs.append(tp_s / v_total_gt)
                        s_idx = np.argsort(v_recs)
                        s_r = np.array(v_recs)[s_idx]
                        s_p = np.array(v_precs)[s_idx]
                        v_ap50 = float(np.trapz(s_p, s_r)) if len(s_r) > 1 else v_p

                    v_met_dict = {
                        'name': name,
                        'video': name,
                        'ap50': v_ap50,
                        'f1': v_f1,
                        'precision': v_p,
                        'recall': v_r,
                        'metrics': {
                            'AP50': v_ap50,
                            'F1': v_f1,
                            'Precision': v_p,
                            'Recall': v_r,
                            'TP': v_tp_count,
                            'FP': v_fp_count,
                            'FN': v_fn_count
                        }
                    }
                    computed_video_metrics.append(v_met_dict)
                    per_video_records[name] = {
                        'matched': v_matched,
                        'fp': v_fp,
                        'fn': v_fn,
                        'metric': v_met_dict
                    }

        if not (matched_records or fp_records or fn_records):
            return

        def _build_single_diagnostic_dict(m_records, f_records, n_records, cvs, v_metrics=None):
            if not (m_records or f_records or n_records):
                return {}
            
            cls_set = sorted({r['gt'] for r in m_records} | {r['gt'] for r in n_records}
                             | {r['det'] for r in m_records} | {r['det'] for r in f_records})
            cls_idx = {c: i for i, c in enumerate(cls_set)}
            n_cls = len(cls_set)
            cm = np.zeros((n_cls + 1, n_cls + 1))
            for r in m_records:
                cm[cls_idx[r['gt']], cls_idx[r['det']]] += 1
            for r in f_records:
                cm[n_cls, cls_idx[r['det']]] += 1
            for r in n_records:
                cm[cls_idx[r['gt']], n_cls] += 1

            all_dets = m_records + f_records
            total_gt = len(m_records) + len(n_records)
            confs = np.array([r['score'] for r in all_dets]) if all_dets else np.array([])
            is_tp = np.array([True] * len(m_records) + [False] * len(f_records))
            thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
            precs, recs, f1_scores = [], [], []
            for t in thresholds:
                sel = confs >= t
                tp_sel = int(is_tp[sel].sum())
                n_sel = int(sel.sum())
                p = tp_sel / n_sel if n_sel else 0.0
                rc = tp_sel / total_gt if total_gt else 0.0
                precs.append(p)
                recs.append(rc)
                f1_scores.append(2 * p * rc / (p + rc) if (p + rc) else 0.0)

            opt_thr = float(thresholds[int(np.argmax(f1_scores))]) if f1_scores else 0.5
            ece_val = 0.0
            if len(confs):
                for lo in np.arange(0.0, 1.0, 0.1):
                    hi = lo + 0.1
                    in_bin = (confs >= lo) & (confs < hi) if hi < 1.0 else (confs >= lo) & (confs <= 1.0)
                    if in_bin.any():
                        ece_val += abs(float(is_tp[in_bin].mean()) - float(confs[in_bin].mean())) * (int(in_bin.sum()) / len(confs))

            conf_tp_list = [float(r['score']) for r in m_records if r['gt'] == r['det']]
            conf_fp_list = [float(r['score']) for r in f_records] + [float(r['score']) for r in m_records if r['gt'] != r['det']]

            per_class_crvs = {}
            for c in cls_set:
                c_matches = [r for r in m_records if r['gt'] == c and r['det'] == c]
                c_fps = [r for r in f_records if r['det'] == c] + [r for r in m_records if r['det'] == c and r['gt'] != c]
                c_gts_total = len([r for r in m_records if r['gt'] == c]) + len([r for r in n_records if r['gt'] == c])
                c_dets = c_matches + c_fps
                if not c_dets or c_gts_total == 0:
                    continue

                c_confs = np.array([r['score'] for r in c_dets])
                c_is_tp = np.array([True] * len(c_matches) + [False] * len(c_fps))
                c_precs, c_recs, c_f1s = [], [], []
                for t in thresholds:
                    sel = c_confs >= t
                    tp_sel = int(c_is_tp[sel].sum())
                    n_sel = int(sel.sum())
                    p = tp_sel / n_sel if n_sel else 0.0
                    rc = tp_sel / c_gts_total if c_gts_total else 0.0
                    c_precs.append(p)
                    c_recs.append(rc)
                    c_f1s.append(2 * p * rc / (p + rc) if (p + rc) else 0.0)

                c_opt_idx = int(np.argmax(c_f1s)) if c_f1s else 0
                sorted_rc_indices = np.argsort(c_recs)
                sorted_rc = np.array(c_recs)[sorted_rc_indices]
                sorted_pr = np.array(c_precs)[sorted_rc_indices]
                c_ap = float(np.trapz(sorted_pr, sorted_rc)) if len(sorted_rc) > 1 else 0.0

                per_class_crvs[c] = {
                    'confidences': thresholds.tolist(),
                    'precisions': c_precs,
                    'recalls': c_recs,
                    'f1s': c_f1s,
                    'optimal_thr': float(thresholds[c_opt_idx]) if c_f1s else 0.5,
                    'peak_f1': float(c_f1s[c_opt_idx]) if c_f1s else 0.0,
                    'ap': max(0.0, min(1.0, c_ap))
                }

            classification_err = len([r for r in m_records if r['gt'] != r['det']])
            localization_err = len([r for r in f_records if r.get('max_iou', 0) >= 0.1])
            background_fp = len([r for r in f_records if r.get('max_iou', 0) < 0.1])
            missed_fn = len(n_records)
            err_breakdown = {
                'classification': classification_err,
                'localization': localization_err,
                'background_fp': background_fp,
                'missed_fn': missed_fn
            }

            ratio_edges = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, np.inf]
            ratio_mids = [0.125, 0.375, 0.625, 0.875, 1.25, 1.75, 2.5, 3.5, 6.0]
            ar_ratios, ar_errors = [], []
            if all_dets:
                ratios = np.array([r['w'] / r['h'] if r['h'] else 0.0 for r in all_dets])
                for bi in range(len(ratio_edges) - 1):
                    in_bin = (ratios >= ratio_edges[bi]) & (ratios < ratio_edges[bi + 1])
                    if in_bin.any():
                        ar_ratios.append(ratio_mids[bi])
                        ar_errors.append(float((is_tp[in_bin] == 0).mean() * 100))

            return {
                'classes': cls_set,
                'confusion': cm.tolist(),
                'ious': [r['iou'] for r in m_records],
                'calibration': {
                    'confidences': thresholds.tolist(),
                    'precisions': precs,
                    'recalls': recs,
                    'ece_score': float(ece_val),
                    'optimal_thr': opt_thr,
                },
                'conf_tp': conf_tp_list,
                'conf_fp': conf_fp_list,
                'per_class_curves': per_class_crvs,
                'error_breakdown': err_breakdown,
                'aspect_ratio': {'ratios': ar_ratios, 'error_rates': ar_errors},
                'fp_coords': [[r['cx'], r['cy']] for r in f_records],
                'fn_coords': [[r['cx'], r['cy']] for r in n_records],
                'canvas_size': [cvs[0] or 1920, cvs[1] or 1080],
                'video_metrics': v_metrics or [],
            }

        # Build Per-Video Diagnostics
        by_video = {}
        for v_name, v_data in per_video_records.items():
            by_video[v_name] = _build_single_diagnostic_dict(
                v_data['matched'], v_data['fp'], v_data['fn'], canvas, [v_data['metric']]
            )

        # Build Per-Category Diagnostics
        by_category = {}
        cat_to_vids = defaultdict(list)
        if hasattr(self, 'video_category_mappings') and self.video_category_mappings:
            for k, v in self.video_category_mappings.items():
                if isinstance(v, list):
                    for item in v:
                        if item in per_video_records:
                            cat_to_vids[k].append(item)
                        else:
                            cat_to_vids[item].append(k)
                elif isinstance(v, str):
                    cat_to_vids[v].append(k)

        for cat_name, v_names in cat_to_vids.items():
            c_matched = []
            c_fp = []
            c_fn = []
            c_v_mets = []
            for vn in set(v_names):
                if vn in per_video_records:
                    c_matched.extend(per_video_records[vn]['matched'])
                    c_fp.extend(per_video_records[vn]['fp'])
                    c_fn.extend(per_video_records[vn]['fn'])
                    c_v_mets.append(per_video_records[vn]['metric'])
            if c_matched or c_fp or c_fn:
                by_category[cat_name] = _build_single_diagnostic_dict(
                    c_matched, c_fp, c_fn, canvas, c_v_mets
                )

        # Build Overall Dataset Diagnostics
        diag = _build_single_diagnostic_dict(
            matched_records, fp_records, fn_records, canvas, computed_video_metrics
        )
        diag['by_video'] = by_video
        diag['by_category'] = by_category

        with open(os.path.join(out_folder, 'diagnostics.json'), 'w') as f:
            json.dump(diag, f)

    def eval_detect(self):
        def evaluate_detector(dets_path, gt_path):
            if hasattr(coco_evaluator, 'evaluate_coco_files'):
                return coco_evaluator.evaluate_coco_files(gt_path, dets_path)
            dets = coco2bb(dets_path, BBType.DETECTED)
            gts = coco2bb(gt_path)
            coc = coco_evaluator.get_coco_summary(gts, dets)
            if 'Score' not in coc:
                coc['Score'] = 0.5
            return coc
        # Creating json file
        gt_path = [self.gt_path] if isinstance(self.gt_path,str) else self.gt_path
        det_path = [self.det_path] if isinstance(self.det_path,str) else self.det_path
        with cons.status('[bold blue] writing json files ') as status:
            convert_to_json(gt_path, det_path,self.size_thr,
                            self.quality_thr,self.speed_check,self.ignore_all,self.category_names,
                            class_mapping=self.class_mapping,
                            ignored_videos=self.ignored_videos,
                            video_class_mappings=self.video_class_mappings,
                            video_category_mappings=self.video_category_mappings,
                            target_classes=self.target_classes)
        data_list = []
        cons.print("[bold cyan]json files created")
        # evaluating created json files
        with cons.status('[bold green] evaluating detector') as status:
            for dets_txts_path in (self.det_path):
                for gts_txts_path in self.gt_path:
                    for i, det_json in enumerate(glob.glob(dets_txts_path+'/*.json')):
                        name = os.path.split(det_json)[1]
                        f_name = os.path.splitext(name)[0]
                        if os.path.exists(gts_txts_path+f'/{name}'):
                            gts_json = gts_txts_path+f'/{name}'
                            data_list.append({**{'Video_name':f_name},**evaluate_detector(det_json, gts_json)})
        data_list = sorted(data_list, key = lambda x: x['Video_name'])
        # Writing Data into file
        slow = []
        medium = []
        fast = []
        for i in range(len(data_list)):
            if data_list[i]['Video_name'] == 'medium_all_video':
                medium = data_list[i]
                data_list = data_list[:i] + data_list[i+1:]
                break
        for i in range(len(data_list)):    
            if data_list[i]['Video_name'] == 'fast_all_video':
                fast = data_list[i]
                data_list = data_list[:i] + data_list[i+1:]
                break
        for i in range(len(data_list)):    
            if data_list[i]['Video_name'] == 'slow_all_video':
                slow = data_list[i]
                data_list = data_list[:i] + data_list[i+1:]
                break
        new_list =[]
        all_data = []
        categories_data = []
        for data in data_list:
            if data['Video_name'] == 'all_video':
                all_data = data
                continue
            if data['Video_name'] in self.category_names :
                categories_data.append(data)
                continue
            new_list.append(data)
        if len(categories_data):
            for data in categories_data:
                new_list.append(data)
        if slow != []:
            new_list.append(slow)
        if medium != []:
            new_list.append(medium)
        if fast != []:                
            new_list.append(fast)
        if len(all_data):
            new_list.append(all_data)

        out_folder = self.det_path[0]+'/evaluation_result'
        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

        # Persist real per-detection diagnostics for the advanced diagnostic plots
        try:
            self._compute_detection_diagnostics(out_folder)
        except Exception as diag_err:
            cons.print(f"[yellow]Warning: detection diagnostics failed: {diag_err}[/yellow]")

        # Generate per-class and per-size mAP plots from summary data
        target_summary_data = all_data if len(all_data) > 0 else (new_list[0] if len(new_list) > 0 else {})
        if target_summary_data:
            per_class_m = target_summary_data.get('per_class_metrics', {})
            per_size_m = target_summary_data.get('per_size_metrics', {})

            # Map numeric category IDs to the real class names from the COCO categories
            class_id_to_name = self._get_category_id_to_name()
            named_class_m = {}
            for c_id, metrics in per_class_m.items():
                try:
                    c_key = int(c_id)
                except (TypeError, ValueError):
                    c_key = c_id
                target_name = class_id_to_name.get(c_key) or str(c_id)
                named_class_m[target_name] = metrics

            if named_class_m:
                try:
                    plot_map_by_class(named_class_m, os.path.join(out_folder, 'map_by_class.png'))
                except Exception:
                    pass
            if per_size_m:
                try:
                    plot_map_by_size(per_size_m, os.path.join(out_folder, 'map_by_size.png'))
                except Exception:
                    pass

            # Persist named per-class & per-size metrics for the UI analytics view
            def _clean(v):
                if v is None:
                    return None
                if isinstance(v, np.generic):
                    v = v.item()
                try:
                    if np.isnan(v):
                        return None
                except (TypeError, ValueError):
                    pass
                return v

            named_per_class = {
                c_name: {
                    'AP50': _clean(m.get('AP50')),
                    'AP': _clean(m.get('AP')),
                    'TP': _clean(m.get('TP')),
                    'FP': _clean(m.get('FP')),
                    'FN': _clean(m.get('FN')),
                }
                for c_name, m in named_class_m.items()
            }
            named_sizes = {s_name: _clean(s_val) for s_name, s_val in per_size_m.items()}
            try:
                with open(os.path.join(out_folder, 'per_class_metrics.json'), 'w') as f:
                    json.dump({'per_class_metrics': named_per_class,
                               'per_size_metrics': named_sizes}, f, indent=2)
            except Exception as e:
                cons.print(f"[yellow]Warning: could not save per-class metrics: {e}[/yellow]")

        if HAS_PANDAS and pd is not None:
            df = pd.DataFrame(new_list)
            # Drop nested dict columns before saving to CSV
            for col in ['per_class_metrics', 'per_size_metrics']:
                if col in df.columns:
                    df = df.drop(columns=[col])
            for i in df:
                for j in range(len(df[i])):
                    if type(df[i][j]) in [np.ndarray, np.array]:
                        df[i][j] = 0
            if self.write_detection_check_res:
                df.to_csv(out_folder+f"/eval_detection.csv", index=False, header=True)
            self.df = df
        else:
            if self.write_detection_check_res and new_list:
                headers = []
                skip_keys = {'per_class_metrics', 'per_size_metrics'}
                for item in new_list:
                    for k in item.keys():
                        if k not in headers and k not in skip_keys:
                            headers.append(k)
                with open(out_folder+"/eval_detection.csv", "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for row in new_list:
                        cleaned_row = {}
                        for k, v in row.items():
                            if k in skip_keys:
                                continue
                            if isinstance(v, (np.ndarray, list)):
                                cleaned_row[k] = 0
                            else:
                                cleaned_row[k] = v
                        writer.writerow(cleaned_row)
            self.df = new_list

        # JSON files are intentionally kept for debugging; remove manually when no longer needed.
        cons.print('[bold cyan]detector evaluated successfully & mAP plots saved! :smiley:')
    
    def eval_center(self):
        with cons.status('[bold green] evaluating center bbox') as status:
            #defining variables as needed
            metric_list = []
            margin = (self.margin_value/100) if self.margin_check else 0.0
            metric_list.append(['video name', 'precision', 'recall', 'F1', 'accuracy', 'TP', 'FP', 'FN', 'TN'])
            pred_path = self.det_path[0]
            gt_path = self.gt_path[0] 
            TPs_all, FPs_all, FNs_all, TNs_all = 0, 0, 0, 0
            TPs_download, FPs_download, FNs_download, TNs_download = 0, 0, 0, 0
            TPs_stadium, FPs_stadium, FNs_stadium, TNs_stadium = 0, 0, 0, 0
            video_counter = 0
            #fining TP,FP ,... for all videos
            for txt in glob.glob(pred_path+'/*.txt'):
                video_counter +=1
                download_check = False
                stadium_check = False
                TP, FP, FN, TN = 0, 0, 0, 0                
                with open(txt,'r') as pred:
                    pred_lines = pred.readlines()
                name = os.path.split(txt)[1]
                with open(gt_path+'/'+name, 'r') as gt:
                    gt_lines = gt.readlines()
                if ('download' in txt or 'Download' in txt) and ('download' in gt_path or 'Download' in gt_path):
                    download_check = True
                elif ('stadium' in txt or 'Stadium' in txt) and ('stadium' in gt_path or 'Stadium' in gt_path):
                    stadium_check = True
                if self.center_conf_thr is not None:
                    conf_threshold = self.center_conf_thr
                else:
                    conf_threshold = 0.5
                    try:
                        if hasattr(self, 'df') and self.df is not None:
                            if hasattr(self.df, 'iloc') and len(self.df) > 0 and 'Score' in self.df.columns:
                                val = self.df.iloc[-1]['Score']
                                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                                    conf_threshold = float(val)
                            elif isinstance(self.df, list) and len(self.df) > 0 and isinstance(self.df[-1], dict) and 'Score' in self.df[-1]:
                                val = self.df[-1]['Score']
                                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                                    conf_threshold = float(val)
                    except Exception:
                        conf_threshold = 0.5
                for i, pred_line_str in enumerate(pred_lines):
                    pred_clean = pred_line_str.strip().rstrip(';').strip()
                    try:
                        pred_line = eval(pred_clean) if pred_clean else []
                    except Exception:
                        pred_line = []
                    if not isinstance(pred_line, list):
                        pred_line = []
                    if len(pred_line) > 0 and not isinstance(pred_line[0], list):
                        pred_line = [pred_line]

                    gt_raw = gt_lines[i] if i < len(gt_lines) else ""
                    gt_parts = [p.strip() for p in gt_raw.split(';') if p.strip()]

                    if len(pred_line):
                        for sub_pred_line in pred_line:
                            if len(sub_pred_line) > 5 and sub_pred_line[5] < conf_threshold:
                                continue
                            InGTBox = False
                            if len(gt_parts):
                                for sub_gt_str in gt_parts:
                                    qualified = False
                                    try:
                                        sub_gt_line = eval(sub_gt_str)
                                    except Exception:
                                        continue
                                    if len(sub_gt_line) < 5 or len(sub_pred_line) < 5:
                                        continue
                                    xc_pred = (sub_pred_line[1]+sub_pred_line[3])/2
                                    yc_pred = (sub_pred_line[2]+sub_pred_line[4])/2
                                    if xc_pred >= sub_gt_line[1]-margin*sub_gt_line[3] and xc_pred <= (1+margin)*(sub_gt_line[3])+sub_gt_line[1] and  yc_pred >= sub_gt_line[2]-margin*sub_gt_line[4] and yc_pred <= (1+margin)*(sub_gt_line[4])+sub_gt_line[2]:
                                        InGTBox = True
                                        size_val = sub_gt_line[5] if len(sub_gt_line) > 5 else 100
                                        qual_val = sub_gt_line[6] if len(sub_gt_line) > 6 else 100
                                        if (size_val <= 0 or size_val >= self.size_thr) and (qual_val <= 0 or qual_val >= self.quality_thr):
                                            qualified = True
                                            TP+=1
                                            break
                                if InGTBox and (not qualified):
                                    pass
                                if (not InGTBox):
                                    FP+=1
                            else:
                                FP+=len(pred_line)
                                break
                    else:
                        if len(gt_parts):
                            for sub_gt_str in gt_parts:
                                try:
                                    sub_gt_line = eval(sub_gt_str)
                                except Exception:
                                    continue
                                size_val = sub_gt_line[5] if len(sub_gt_line) > 5 else 100
                                qual_val = sub_gt_line[6] if len(sub_gt_line) > 6 else 100
                                if (size_val > 0 and size_val < self.size_thr) or (qual_val > 0 and qual_val < self.quality_thr):
                                    continue
                                FN+=1
                        else:
                            TN+=1
                
                if download_check:
                    TPs_download += TP
                    FPs_download += FP
                    TNs_download += TN
                    FNs_download += FN
                
                if stadium_check:
                    TPs_stadium += TP
                    FPs_stadium += FP
                    TNs_stadium += TN
                    FNs_stadium += FN
                    
                precision = TP/(TP+FP)
                recall = TP/(TP+FN)
                TPs_all += TP
                FPs_all += FP
                TNs_all += TN
                FNs_all += FN
                metric_list.append([name, precision, recall, 2*precision*recall/(precision+recall), (TP+TN)/(TP+TN+FN+FP), TP, FP, FN, TN])
            if TPs_download:
                precision = TPs_download/(TPs_download+FPs_download)
                recall = TPs_download/(TPs_download+FNs_download)
                metric_list.append(['download_video',precision, recall, 2*precision*recall/(precision+recall), (TPs_download+TNs_download)/(TPs_download+TNs_download+FNs_download+FPs_download), TPs_download, FPs_download, FNs_download, TNs_download])
            if TPs_stadium:
                precision = TPs_stadium/(TPs_stadium+FPs_stadium)
                recall = TPs_stadium/(TPs_stadium+FNs_stadium)
                metric_list.append(['stadium_video',precision, recall, 2*precision*recall/(precision+recall), (TPs_stadium+TNs_stadium)/(TPs_stadium+TNs_stadium+FNs_stadium+FPs_stadium), TPs_stadium, FPs_stadium, FNs_stadium, TNs_stadium])
            precision = TPs_all/(TPs_all+FPs_all)
            recall = TPs_all/(TPs_all+FNs_all)
            if video_counter>1:
                metric_list.append(['all_video',precision, recall, 2*precision*recall/(precision+recall), (TPs_all+TNs_all)/(TPs_all+TNs_all+FNs_all+FPs_all), TPs_all, FPs_all, FNs_all, TNs_all])
            # writing data into file
            os.makedirs(self.det_path[0] + '/evaluation_result/', exist_ok=True)
            with open(self.det_path[0]+'/evaluation_result/'+'fall_in_center_evaluate_tracker_margin_'+str(margin)+'.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(metric_list) # Use writerow for single list
            cons.print('[bold cyan] center bbox evaluated successfully :smiley:')
    
    def evaluate_all(self):
        t0 = time.time()
        gt_path = self.gt_path[0] +'/Track/gt'
        dets_path_tracker = self.gt_path[0] +'/Track/dets'
        Config = config(gt_path,dets_path_tracker,self.det_path[0])
        
        if self.tracking_check:
            self.eval_track(Config)
        if self.detection_check:
            self.eval_detect()            
        if self.center_check:
            self.eval_center()
        if self.combine_check:
            with cons.status('[bold red] writing combined file') as status:
                create_table(Config, self.tracking_check, self.detection_check, self.center_check, self.speed_check)    
            cons.print('[bold cyan] combined file written')
        if self.visualize_check:
            self.visualize_detect()
        t1 = time.time()
        
        cons.print('[bold green] evaluation Done in %i seconds'%int(t1-t0))

if __name__ == '__main__':
    gt_path = input('enter gt_path')
    det_path = input('enter det_path')
    quality_thr = eval(input('enter quality thr'))
    size_thr= eval (input('enter size thr'))
    
    detection_check = input('do you want to check detection? (y/n)')
    detection_check = True if detection_check.lower()=='y' or detection_check.lower() =='yes' else False
    if detection_check:
        speed_check = input('do you want to check detection in different speeds? (y/n)')
        speed_check = True if speed_check.lower()=='y' or speed_check.lower() =='yes' else False
    track_check = input('do you want to check track? (y/n)')
    track_check = True if track_check.lower()=='y' or track_check.lower() =='yes' else False
    center_check = input('do you want check center? (y/n)')
    center_check = True if center_check.lower()=='y' or center_check.lower() =='yes'  else False
    margin_check = input('do you want use margin? (y/n)')
    margin_check = True if margin_check.lower()=='y' or margin_check.lower() =='yes'  else False
    margin_val = 0
    if margin_check:
        margin_val = eval(input('enter margin value'))
    visualize_check = input('do you want to visualize detection problems? (y/n)')
    visualize_check = True if visualize_check.lower()=='y' or visualize_check.lower() =='yes' else False
    visualize_iou = 0
    if visualize_check:
        visualize_iou = eval(input('enter visualize IOU'))
        
    
    
    evaluate = Evaluate(gt_path,det_path,quality_thr,size_thr,margin_check,
                    margin_val,center_check,detection_check,track_check,speed_check,
                    visualize_check,True,visualize_iou) 
    evaluate.evaluate_all()