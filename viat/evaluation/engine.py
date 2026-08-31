#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import csv
import shutil
import pathlib
import time
import re
import copy
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
        
    def eval_track(self,Config):
        if run_tracker is None:
            cons.print("[yellow]Warning: Tracking evaluation module (TrackEval) is not available, skipping tracking evaluation.[/yellow]")
            return
        if (self.gt_path,list) and len(self.gt_path)>1:
            if isinstance(self.gt_path,list) and isinstance(self.det_path,list):
                Config.stadium_downloads =['_all','_download','_stadium']
                for i,j in zip(self.gt_path,self.det_path):
                    gt_path = [i]
                    det_path = [j]
                    convert_to_mot(gt_path,det_path,Config,self.size_thr,self.quality_thr)
                    run_tracker(Config)
                convert_to_mot(self.gt_path,self.det_path,Config,self.size_thr,self.quality_thr)
                run_tracker(Config)
            elif isinstance(self.gt_path,list) or isinstance(self.det,list):
                ValueError(f"gt_path and det_path should be in same type\ntype gt_path is {type(gt_path)} and type det_path is {type(det_path)}")   
        else:
            Config.stadium_downloads = []
            gt_path = [self.gt_path] if type(self.gt_path) == str else self.gt_path
            det_path = [self.det_path] if type(self.det_path) == str else self.det_path
            with cons.status('[bold blue] writing mot format files ') as status:
                convert_to_mot(gt_path,det_path,Config,self.size_thr,self.quality_thr)
            cons.print("[bold cyan]mot format files created")
            with cons.status('[bold green] evaluating tracker') as status:
                run_tracker(Config)
            if os.path.exists(self.gt_path[0] +'/Track'):
                shutil.rmtree(self.gt_path[0] +'/Track')
            cons.print('[bold cyan] tracker evaluated successfully :smiley:')
    
    def eval_detect(self):
        def evaluate_detector(dets_path, gt_path):
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

        # Generate per-class and per-size mAP plots from summary data
        target_summary_data = all_data if len(all_data) > 0 else (new_list[0] if len(new_list) > 0 else {})
        if target_summary_data:
            per_class_m = target_summary_data.get('per_class_metrics', {})
            per_size_m = target_summary_data.get('per_size_metrics', {})

            # Map numeric category IDs to category names if available
            named_class_m = {}
            for c_id, metrics in per_class_m.items():
                if isinstance(c_id, int) and 0 <= c_id - 1 < len(self.category_names):
                    c_name = self.category_names[c_id - 1]
                else:
                    c_name = str(c_id)
                named_class_m[c_name] = metrics

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