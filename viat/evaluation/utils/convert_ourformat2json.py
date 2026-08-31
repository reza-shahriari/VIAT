#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 23:45:31 2023

@author: hds
"""
import os
import glob
import json as original_json
import cv2
import numpy as np

class NumpyEncoder(original_json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class _json_wrapper:
    @staticmethod
    def dumps(obj, **kwargs):
        kwargs['cls'] = NumpyEncoder
        return original_json.dumps(obj, **kwargs)
    @staticmethod
    def dump(obj, fp, **kwargs):
        kwargs['cls'] = NumpyEncoder
        return original_json.dump(obj, fp, **kwargs)
json = _json_wrapper()
try:
    import tqdm
except ImportError:
    class _DummyPbar:
        def __init__(self, iterable=None, total=None, desc=None, *args, **kwargs):
            self.iterable = iterable
            self.total = total
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            if self.iterable is not None:
                return iter(self.iterable)
            return iter([])

        def update(self, n=1):
            self.n += n

        def set_description(self, *args, **kwargs):
            pass

        def set_postfix(self, *args, **kwargs):
            pass

    class _DummyTqdmModule:
        def __call__(self, iterable=None, *args, **kwargs):
            return _DummyPbar(iterable, *args, **kwargs)

        def tqdm(self, iterable=None, *args, **kwargs):
            return _DummyPbar(iterable, *args, **kwargs)

    tqdm = _DummyTqdmModule()

try:
    from viat.evaluation.evaluators.coco_evaluator import jaccard
except ImportError:
    from ..evaluators.coco_evaluator import jaccard

import time

def strip_header_lines(lines):
    """Strip Raya header block (delimited by ###) or leading comment lines."""
    in_header = False
    data_lines = []
    for line in lines:
        sline = line.strip()
        if sline == "###":
            in_header = not in_header
            continue
        if in_header or sline.startswith("#"):
            continue
        data_lines.append(line)
    return data_lines


def build_categories(category_names, observed_cat_ids):
    """Build a clean, non-duplicate COCO categories list.

    Annotations use 0-indexed class IDs (VIAT/YOLO convention).
    If category_names are provided we generate one entry per name
    at id == index (0, 1, 2, ...), which matches the annotation IDs.
    Any observed category_id not covered by category_names gets a
    fallback numeric name.
    """
    seen_ids = set()
    categories = []

    if category_names:
        for idx, name in enumerate(category_names):
            if idx not in seen_ids:
                categories.append({'id': idx, 'name': name, 'supercategory': 'none'})
                seen_ids.add(idx)

    # Cover any annotation IDs not already in the list
    for cid in sorted(observed_cat_ids):
        if cid not in seen_ids:
            # Try to resolve a friendly name from category_names
            name = category_names[cid] if (category_names and 0 <= cid < len(category_names)) else str(cid)
            categories.append({'id': cid, 'name': name, 'supercategory': 'none'})
            seen_ids.add(cid)

    if not categories:
        categories = [{'id': 0, 'name': 'Object', 'supercategory': 'none'}]

    return categories


def convert_to_json(groundtruth_txt_paths,dets_txt_paths,size_thr=None,quality_thr=None,check_for_speeds=False,ignore_all = True,category_names=[],class_mapping=None, ignored_videos=None, video_class_mappings=None, video_category_mappings=None):
    # groundtruth_txt_paths = []
    # groundtruth_txt_paths.append(input('Enter ground truth txts path or d for default: '))
    # if groundtruth_txt_paths[0] == 'd':
    #     gt_type_folder_name = 'crop_640_640'
    #     print('\033[1m'+f'Using Ground Truths From {gt_type_folder_name} Folder'+'\033[0m')
    #     groundtruth_txt_paths = []
    #     groundtruth_txt_paths.append('/media/hds/88AA0CE2AA0CCE9E/Users/HDS/Desktop/Works/UAVs-Works/dataset/use/lablelling/Download/Labeled/test/current_useful/'+gt_type_folder_name)
    #     groundtruth_txt_paths.append('/media/hds/88AA0CE2AA0CCE9E/Users/HDS/Desktop/Works/UAVs-Works/dataset/use/lablelling/Stadium/Labeled/test/'+gt_type_folder_name)
    # dets_txt_paths = []
    # dets_txt_paths.append(input('Enter stadium detection txts path: '))
    # dets_txt_paths.append(input('Enter download detection txts path: '))
    if size_thr == None:	    
        size_thr = eval(input('Enter Thr for Size: '))	    
    if quality_thr == None:	
        quality_thr = eval(input('Enter Thr for Quality: '))
    ignore_all = ignore_all # ignore all detection that placed on hard gtruth area (hard area is area that it's size and quality of bbox is below threshold) 
    
    cnt = 0
    det_cnt = 0
    img_id = 0
    all_img = []
    sub_all_img = []
    all_annotations = []
    # nospeed_all_annotations = []
    # nospeed_all_det_annotations = []
    slow_all_annotations = []
    slow_all_det_annotations = []
    medium_all_annotations = []
    medium_all_det_annotations = []
    fast_all_annotations = []
    fast_all_det_annotations = []
    all_det_annotations = []
    sub_all_det_annotations = []
    sub_all_annotations = []
    SLOW_THR = 3.2
    MEDIUM_THR = 15.5
    FAST_THR = 15.5
    exts=['*.mp4','*.avi','*.mkv','*.mov','*.webm','*.mpg', '*.MOV','*.m4v']
    folder_len = len(groundtruth_txt_paths)*len(dets_txt_paths)
    sub_all=None
    video_count = 0
    with tqdm.tqdm(total=folder_len, desc="Writing Json", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for gt_category_counter,txt_path in enumerate(groundtruth_txt_paths):
            for  dt_category_counter,dets_txt_path in enumerate(dets_txt_paths):
                if len(category_names) ==0:
                    sub_all = None
                    if ('stadium' in txt_path or 'Stadium' in txt_path) and ('stadium' in dets_txt_path or 'Stadium' in dets_txt_path):
                        sub_all = 'stadium'
                        sub_all_det_annotations = []
                        sub_all_annotations = []
                        sub_all_img = []
                    if ('download' in txt_path or 'Download' in txt_path) and ('download' in dets_txt_path or 'Download' in dets_txt_path):
                        sub_all = 'download'
                        sub_all_det_annotations = []
                        sub_all_annotations = []
                        sub_all_img = []
                else:
                    if dt_category_counter == gt_category_counter :
                        sub_all = category_names[gt_category_counter]
                        sub_all_det_annotations = []
                        sub_all_annotations = []
                        sub_all_img = []
                for ext in exts:
                    for video in glob.glob(os.path.join(txt_path,ext)):
                        Video = cv2.VideoCapture(video)
                        width = Video.get(3)
                        height = Video.get(4)
                        name = os.path.splitext(os.path.split(video)[1])[0]
                        if ignored_videos and name in ignored_videos:
                            continue
                        # if name != 'out277':
                        #     continue
                        if not os.path.exists(os.path.join(dets_txt_path,name+'.txt')):
                            # print(os.path.join(dets_txt_path,name+'.txt')," doesn't exist")
                            continue
                        with open(os.path.join(dets_txt_path,name+'.txt'),'r') as det_txt:
                            det_lines = strip_header_lines(det_txt.readlines())
                            txt = os.path.splitext(video)[0]+'.txt'
                            images = []
                            annotations = []
                            det_annotations = []
                            # annotation for different speed
                            fast_annotations =[]
                            fast_det_annotations =[]
                            medium_annotations = []
                            medium_det_annotations = []
                            slow_annotations =[]
                            slow_det_annotations =[]
                            # nospeed_annotations =[]
                            # nospeed_det_annotations =[]
                            score_dict = {}
                            local_img_id = 0
                            
                            with open(txt,'r') as t:
                                lines = strip_header_lines(t.readlines())
                                last_frame_middle_bbox = [-2,-2,-2]
                                for i,line in enumerate(lines):
                                    img_id+=1
                                    local_img_id+=1
                                    distance = -1
                                    
                                    if "DELETED" in line:
                                        continue
                                        
                                    line = line.split(';')[0:-1]
                                    det_line = []
                                    try:
                                        raw = det_lines[i].strip().rstrip(';').strip() if i < len(det_lines) else ""
                                        det_line = eval(raw) if raw else []
                                        if not isinstance(det_line, list):
                                            det_line = []
                                        # normalise: if single detection (not nested), wrap it
                                        if len(det_line) > 0 and not isinstance(det_line[0], list):
                                            det_line = [det_line]
                                    except Exception as e:
                                        print(e, f'in {name} in line {i}')
                                    
                                    ignore_gt_bbox = []
                                    #save middle of the box for last frame and its frame number to calculate speed

                                    if len(det_line)==1:
                                            if len(det_line[0]) > 5:
                                                score = det_line[0][5]
                                                if score > 1:
                                                    continue
                                    
                                    if len(line):
                                        for sub_line in line:
                                            sline = eval(sub_line)
                                            bbox = sline[1:5]
                                            cx = bbox[0]+bbox[2]/2
                                            cy = bbox[1]+bbox[3]/2
                                            this_frame_middle_bbox = [img_id,cx,cy]
                                            distance = -1
                                            if last_frame_middle_bbox[0] == img_id -1:
                                                delta_x = this_frame_middle_bbox[1] - last_frame_middle_bbox[1]
                                                delta_y = this_frame_middle_bbox[2] - last_frame_middle_bbox[2]
                                                distance = (delta_x**2 + delta_y**2)**0.5
                                            last_frame_middle_bbox = this_frame_middle_bbox
                                            if len(sline)>5:
                                                #difficult attribute apply. difficult for seperate object shadow from true object.
                                                if len(sline) == 8:
                                                    difficult = sline[7]
                                                else:
                                                    difficult = 0 #difficult False (label is not shadow)
                                                size_thr_now = sline[5]
                                                quality_thr_now = sline[6]
                                            else:
                                                size_thr_now = 100
                                                quality_thr_now = 100
                                                difficult = 0 #difficult False (label is not shadow)
                                            # ignore gtruth if shadow labeled.
                                            if difficult:
                                                continue
                                            if (size_thr_now <= 0 or size_thr_now >= size_thr) and (quality_thr_now <= 0 or quality_thr_now >= quality_thr):
                                                pass
                                            else:
                                                ignore_gt_bbox.append(sline)
                                                bbox=[]
                                            if len(bbox):
                                                cnt+=1
                                                gt_cat_id = sline[0]
                                                if class_mapping is not None or video_class_mappings is not None:
                                                    mapped_target = None
                                                    if video_class_mappings and name in video_class_mappings:
                                                        mapped_target = video_class_mappings[name].get(gt_cat_id, video_class_mappings[name].get(str(gt_cat_id)))
                                                    if mapped_target is None and class_mapping is not None:
                                                        mapped_target = class_mapping.get(gt_cat_id, class_mapping.get(str(gt_cat_id)))
                                                    
                                                    if mapped_target is not None:
                                                        if mapped_target == "__IGNORE__":
                                                            continue  # Ignore this gt
                                                        gt_cat_id = mapped_target if isinstance(mapped_target, int) else 1

                                                res = {'area':int(bbox[2]*bbox[3]),
                                                'bbox':bbox,
                                                'category_id':gt_cat_id,
                                                'id':cnt,
                                                'ignore':0,
                                                'image_id':img_id,
                                                'iscrowd':0,
                                                'segmentation':[[]]}
                                                
                                                annotations.append(res)
                                                all_annotations.append(res)
                                                sub_all_annotations.append(res)

                                                if distance ==-1:
                                                    pass
                                                elif distance < SLOW_THR:  
                                                    slow_annotations.append(res)
                                                    slow_all_annotations.append(res)
                                                elif distance <= MEDIUM_THR:                                                   
                                                    medium_annotations.append(res)
                                                    medium_all_annotations.append(res)
                                                elif distance >FAST_THR:
                                                    fast_annotations.append(res)
                                                    fast_all_annotations.append(res)

                                     
                                    if len(det_line):
                                        for sub_det_line in det_line:
                                            #check detection is not in very hard(ignore gt) area. if place in very hard place ignore it too.
                                            ignore_flg = False
                                            for sub_ignore_gt_bbox in ignore_gt_bbox:
                                                bbox = sub_ignore_gt_bbox[1:5]
                                                sub_ignore_gt_bbox = [sub_ignore_gt_bbox[1], sub_ignore_gt_bbox[2], sub_ignore_gt_bbox[1]+sub_ignore_gt_bbox[3], sub_ignore_gt_bbox[2]+sub_ignore_gt_bbox[4]]
                                                if jaccard(sub_ignore_gt_bbox, sub_det_line[1:5]) > 0.4: 
                                                    cnt+=1
                                                    if ignore_all:
                                                        ignore_flg = True
                                                    else:
                                                        res = {'area':int(bbox[2]*bbox[3]),
                                                        'bbox':bbox,
                                                        'category_id':sline[0],
                                                        'id':cnt,
                                                        'ignore':0,
                                                        'image_id':img_id,
                                                        'iscrowd':0,
                                                        'segmentation':[[]]}
                                                        annotations.append(res)
                                                        # if cat_count==0:
                                                        all_annotations.append(res)
                                                        sub_all_annotations.append(res)
                                                        # else:
                                                            # sub_alls_det_annotations[category_counter].append(res)
                                                            # sub_alls_annotations[category_counter].append(res)
                                                        
                                            if not ignore_flg:
                                                w = sub_det_line[3]-sub_det_line[1]
                                                h = sub_det_line[4]-sub_det_line[2]
                                                det_bbox = [sub_det_line[1],sub_det_line[2],w,h]
                                            else:
                                                det_bbox = []
                                            if len(det_bbox):
                                                det_cat_id = sub_det_line[0] if len(sub_det_line) > 0 else 1
                                                if class_mapping is not None or video_class_mappings is not None:
                                                    mapped_target = None
                                                    if video_class_mappings and name in video_class_mappings:
                                                        mapped_target = video_class_mappings[name].get(det_cat_id, video_class_mappings[name].get(str(det_cat_id)))
                                                    if mapped_target is None and class_mapping is not None:
                                                        mapped_target = class_mapping.get(det_cat_id, class_mapping.get(str(det_cat_id)))
                                                    
                                                    if mapped_target is not None:
                                                        if mapped_target == "__IGNORE__":
                                                            continue  # Ignore this prediction
                                                        det_cat_id = mapped_target if isinstance(mapped_target, int) else 1

                                                res = {'area':int(det_bbox[2]*det_bbox[3]),
                                                'bbox':det_bbox,
                                                'category_id': det_cat_id,
                                                'id':det_cnt,
                                                'ignore':0,
                                                'image_id':img_id,
                                                'iscrowd':0,
                                                'score': sub_det_line[5],
                                                'segmentation':[[]]}
                                                det_annotations.append(res)
                                                all_det_annotations.append(res)
                                                sub_all_det_annotations.append(res)
                                                det_cnt+=1
                                                if distance ==-1:
                                                    pass
                                                elif distance < SLOW_THR:
                                                    slow_det_annotations.append(res)
                                                    slow_all_det_annotations.append(res)
                                                elif distance <= MEDIUM_THR:
                                                    medium_det_annotations.append(res)
                                                    medium_all_det_annotations.append(res)
                                                elif distance > FAST_THR:
                                                    fast_det_annotations.append(res)
                                                    fast_all_det_annotations.append(res)
                                    images.append({'file_name':name+f'_{local_img_id}.jpg','height':height, 'id':img_id,'width':width})
                                    all_img.append({'file_name':name+f'_{local_img_id}.jpg','height':height, 'id':img_id,'width':width})
                                    sub_all_img.append({'file_name':name+f'_{local_img_id}.jpg','height':height, 'id':img_id,'width':width})

                                observed_cats = set(a['category_id'] for a in annotations + det_annotations)
                                categories = build_categories(category_names, observed_cats)

                                gts = {'annotations':annotations ,'categories':categories , 'images':images,'type':'instances'}
                                # nospeed_gts = {'annotations':nospeed_annotations ,'categories':categories , 'images':images,'type':'instances'}
                                slow_gts = {'annotations':slow_annotations ,'categories':categories , 'images':images,'type':'instances'}
                                medium_gts = {'annotations':medium_annotations ,'categories':categories , 'images':images,'type':'instances'}
                                fast_gts = {'annotations':fast_annotations ,'categories':categories , 'images':images,'type':'instances'}
                                dets = {'annotations':det_annotations ,'categories':categories , 'images':images}
                                # nospeed_dets = {'annotations':nospeed_det_annotations ,'categories':categories , 'images':images}
                                slow_dets = {'annotations':slow_det_annotations ,'categories':categories , 'images':images}
                                medium_dets = {'annotations':medium_det_annotations ,'categories':categories , 'images':images}
                                fast_dets = {'annotations':fast_det_annotations ,'categories':categories , 'images':images}
                                
                                jsonString = json.dumps(gts)
                                # nospeed_jsonString = json.dumps(nospeed_gts)
                                slow_jsonString = json.dumps(slow_gts)
                                medium_jsonString = json.dumps(medium_gts)
                                fast_jsonString = json.dumps(fast_gts)
                                det_jsonString = json.dumps(dets)
                                # nospeed_det_jsonString = json.dumps(nospeed_dets)
                                slow_det_jsonString = json.dumps(slow_dets)
                                medium_det_jsonString = json.dumps(medium_dets)
                                fast_det_jsonString = json.dumps(fast_dets)
                                with open(f'{txt_path}/{name}.json','w') as js:
                                    js.write(jsonString)
                                # with open(f'{txt_path}/{name}_unknown.json','w') as js:
                                    # js.write(nospeed_jsonString)
                                if check_for_speeds:
                                    if len(slow_annotations):
                                        with open(f'{txt_path}/{name}_slow.json','w') as js:
                                            js.write(slow_jsonString)
                                    if len(medium_annotations):
                                        with open(f'{txt_path}/{name}_medium.json','w') as js:
                                            js.write(medium_jsonString)
                                    if len(fast_annotations):
                                        with open(f'{txt_path}/{name}_fast.json','w') as js:
                                            js.write(fast_jsonString)    
                                with open(f'{dets_txt_path}/{name}.json','w') as jsd:
                                    jsd.write(det_jsonString)
                                # with open(f'{dets_txt_path}/{name}_unknown.json','w') as jsd:
                                    # jsd.write(nospeed_det_jsonString)   
                                if check_for_speeds:
                                    if len(slow_det_annotations):
                                        with open(f'{dets_txt_path}/{name}_slow.json','w') as jsd:
                                            jsd.write(slow_det_jsonString)
                                    if len(medium_det_annotations):
                                        with open(f'{dets_txt_path}/{name}_medium.json','w') as jsd:
                                            jsd.write(medium_det_jsonString)
                                    if len(fast_det_annotations):
                                        with open(f'{dets_txt_path}/{name}_fast.json','w') as jsd:
                                            jsd.write(fast_det_jsonString)
                                js.close()
                                jsd.close()
                                video_count+=1
                # if cat_count==0:
                if sub_all is not None and video_count>=1:
                    if len(category_names)!= 1 and (len(category_names)==0 or dt_category_counter==gt_category_counter):
                        sub_all_gts = {'annotations':sub_all_annotations ,'categories':categories , 'images':sub_all_img,'type':'instances'}
                        sub_all_dets = {'annotations':sub_all_det_annotations ,'categories':categories , 'images':sub_all_img}
                        sub_all_jsonString = json.dumps(sub_all_gts)
                        sub_all_det_jsonString = json.dumps(sub_all_dets)
                        with open(f'{txt_path}/{sub_all}.json','w') as jssa:
                            jssa.write(sub_all_jsonString)
                        with open(f'{dets_txt_path}/{sub_all}.json','w') as jssda:
                            jssda.write(sub_all_det_jsonString)
                        jssa.close()
                        jssda.close()
                pbar.update()

        # Generate virtual category bins based on UI video mappings
        if video_category_mappings:
            from collections import defaultdict
            cat_bins_img = defaultdict(list)
            cat_bins_gt = defaultdict(list)
            cat_bins_dt = defaultdict(list)
            img_to_vid = {img['id']: img['file_name'].replace('.jpg', '') for img in all_img}

            for img in all_img:
                vid = img_to_vid.get(img['id'], '')
                cats = video_category_mappings.get(vid, [])
                for cat in cats:
                    cat_bins_img[cat].append(img)

            for ann in all_annotations:
                vid = img_to_vid.get(ann['image_id'], '')
                for cat in video_category_mappings.get(vid, []):
                    cat_bins_gt[cat].append(ann)

            for ann in all_det_annotations:
                vid = img_to_vid.get(ann['image_id'], '')
                for cat in video_category_mappings.get(vid, []):
                    cat_bins_dt[cat].append(ann)

            all_observed_cats = set(a['category_id'] for a in all_annotations + all_det_annotations)
            categories = build_categories(category_names, all_observed_cats)

            for cat in cat_bins_img.keys():
                cat_gts = {'annotations': cat_bins_gt[cat], 'categories': categories, 'images': cat_bins_img[cat], 'type': 'instances'}
                cat_dets = {'annotations': cat_bins_dt[cat], 'categories': categories, 'images': cat_bins_img[cat]}
                
                with open(os.path.join(groundtruth_txt_paths[0], f"{cat}.json"), 'w') as f:
                    json.dump(cat_gts, f)
                with open(os.path.join(dets_txt_paths[0], f"{cat}.json"), 'w') as f:
                    json.dump(cat_dets, f)
        if video_count > 1:
            all_observed_cats = set(a['category_id'] for a in all_annotations + all_det_annotations)
            categories = build_categories(category_names, all_observed_cats)


            all_gts = {'annotations':all_annotations ,'categories':categories , 'images':all_img,'type':'instances'}
            slow_all_gts = {'annotations':slow_all_annotations ,'categories':categories , 'images':all_img,'type':'instances'}
            medium_all_gts = {'annotations':medium_all_annotations ,'categories':categories , 'images':all_img,'type':'instances'}
            fast_all_gts = {'annotations':fast_all_annotations ,'categories':categories , 'images':all_img,'type':'instances'}
            # nospeed_all_gts = {'annotations':nospeed_all_annotations ,'categories':categories , 'images':all_img,'type':'instances'}
            all_dets = {'annotations':all_det_annotations ,'categories':categories , 'images':all_img}
            slow_all_dets = {'annotations':slow_all_det_annotations ,'categories':categories , 'images':all_img}
            medium_all_dets = {'annotations':medium_all_det_annotations ,'categories':categories , 'images':all_img}
            fast_all_dets = {'annotations':fast_all_det_annotations ,'categories':categories , 'images':all_img}
            # nospeed_all_dets = {'annotations':nospeed_all_det_annotations ,'categories':categories , 'images':all_img}
            
            all_jsonString = json.dumps(all_gts)
            slow_all_jsonString = json.dumps(slow_all_gts)
            medium_all_jsonString = json.dumps(medium_all_gts)
            fast_all_jsonString = json.dumps(fast_all_gts)
            # nospeed_all_jsonString = json.dumps(nospeed_all_gts)
            all_det_jsonString = json.dumps(all_dets)
            slow_all_det_jsonString = json.dumps(slow_all_dets)
            medium_all_det_jsonString = json.dumps(medium_all_dets)
            fast_all_det_jsonString = json.dumps(fast_all_dets)
            # nospeed_all_det_jsonString = json.dumps(nospeed_all_dets)
            if txt_path != '':
                with open(f'{txt_path}/all_video.json','w') as jsa:
                    jsa.write(all_jsonString)
                if check_for_speeds:
                    if len(slow_all_annotations):
                        with open(f'{txt_path}/slow_all_video.json','w') as jsa:
                            jsa.write(slow_all_jsonString)
                    if len(medium_all_annotations):
                        with open(f'{txt_path}/medium_all_video.json','w') as jsa:
                            jsa.write(medium_all_jsonString)
                    if len(fast_all_annotations):
                        with open(f'{txt_path}/fast_all_video.json','w') as jsa:
                            jsa.write(fast_all_jsonString)
                # with open(f'{txt_path}/unknown_all_video.json','w') as jsa:
                    # jsa.write(nospeed_all_jsonString)
            else:
                for txt_path in groundtruth_txt_paths:
                    if txt_path != '':
                        with open(f'{txt_path}/all_video.json','w') as jsa:
                            jsa.write(all_jsonString)
                    if check_for_speeds:
                        
                        if len(slow_all_annotations):
                            with open(f'{txt_path}/slow_all_video.json','w') as jsa:
                                jsa.write(slow_all_jsonString)
                        if len(medium_all_annotations):
                            with open(f'{txt_path}/medium_all_video.json','w') as jsa:
                                jsa.write(medium_all_jsonString)
                        if len(fast_all_annotations):
                            with open(f'{txt_path}/fast_all_video.json','w') as jsa:
                                jsa.write(fast_all_jsonString)
                        # with open(f'{txt_path}/unknown_all_video.json','w') as jsa:
                            # jsa.write(nospeed_all_jsonString)
            if dets_txt_path != '':
                with open(f'{dets_txt_path}/all_video.json','w') as jsda:
                    jsda.write(all_det_jsonString)
                if check_for_speeds:
                    if len(slow_all_det_annotations):
                        with open(f'{dets_txt_path}/slow_all_video.json','w') as jsda:
                            jsda.write(slow_all_det_jsonString) 
                    if len(medium_all_det_annotations):
                        with open(f'{dets_txt_path}/medium_all_video.json','w') as jsda:
                            jsda.write(medium_all_det_jsonString)     
                    if len(fast_all_det_annotations):
                        with open(f'{dets_txt_path}/fast_all_video.json','w') as jsda:
                            jsda.write(fast_all_det_jsonString)   
                # with open(f'{dets_txt_path}/unknown_all_video.json','w') as jsda:
                    # jsda.write(nospeed_all_det_jsonString)   
            else:
                for dets_txt_path in dets_txt_paths:
                    if dets_txt_path != '':
                        with open(f'{dets_txt_path}/all_video.json','w') as jsda:
                            jsda.write(all_det_jsonString)
                    if check_for_speeds:
                        if len(slow_all_det_annotations):
                            with open(f'{dets_txt_path}/slow_all_video.json','w') as jsda:
                                jsda.write(slow_all_det_jsonString) 
                        if len(medium_all_det_annotations):
                            with open(f'{dets_txt_path}/medium_all_video.json','w') as jsda:
                                jsda.write(medium_all_det_jsonString)     
                        if len(fast_all_det_annotations):
                            with open(f'{dets_txt_path}/fast_all_video.json','w') as jsda:
                                jsda.write(fast_all_det_jsonString)     
                        # with open(f'{dets_txt_path}/unknown_all_video.json','w') as jsda:
                            # jsda.write(nospeed_all_det_jsonString)    
            jsa.close()
            jsda.close()


import os
import cv2
try:
    import imagesize
except ImportError:
    imagesize = None

def get_image_info(file_path):
    if imagesize is not None:
        try:
            width, height = imagesize.get(file_path)
            return {"file_name": file_path, "height": height, "width": width}
        except Exception:
            pass
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            width, height = img.size
            return {"file_name": file_path, "height": height, "width": width}
    except Exception:
        pass
    try:
        img = cv2.imread(file_path)
        if img is not None:
            height, width = img.shape[:2]
            return {"file_name": file_path, "height": height, "width": width}
    except Exception:
        pass
    return {"file_name": file_path, "height": 0, "width": 0}

def get_bbox_info(yolo_line, img_width, img_height,is_obb=False):
    if is_obb:
        parts = yolo_line.split()
        class_id = int(parts[0]) 
        parts = list(map(float,parts[1:]))
        if len(parts) ==9:
            score = parts[-1]
            parts = parts[:-1]
        else:
            score=1
        shap = (img_width,img_height)
        bbox = [parts[i] * shap[i%2] for i in range(len(parts))]
        return class_id, bbox, score
    else:
        pass
def claculate_area(bbox):
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
    
    # Apply the Shoelace formula
    area = 0.5 * abs(x1*y2 + x2*y3 + x3*y4 + x4*y1 - (y1*x2 + y2*x3 + y3*x4 + y4*x1))
    return area
def create_coco_json(gt_dir, dt_dir):
    images = []
    gt_annotations = []
    dt_annotations = []
    categories = []

    category_set = set()
    gt_annotation_id = 1
    dt_annotation_id = 1

    for image_filename in os.listdir(gt_dir):
        if not image_filename.lower().endswith(('jpg', 'jpeg', 'png')):
            continue
        image_path = os.path.join(gt_dir, image_filename)
        image_info = get_image_info(image_path)
        image_id = len(images) + 1
        image_info["id"] = image_id
        images.append(image_info)
        gt_annotation_file = os.path.join(gt_dir, os.path.splitext(image_filename)[0] + ".txt")
        dt_annotation_file = os.path.join(dt_dir, os.path.splitext(image_filename)[0] + ".txt")
        if os.path.exists(gt_annotation_file):
            with open(gt_annotation_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id, bbox,_ =  get_bbox_info(line, image_info['width'], image_info['height'],is_obb=True)
                    category_set.add(class_id)
                    gt_annotation = {
                        "id": gt_annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": bbox,
                        "area":claculate_area(bbox),
                        "iscrowd": 0
                    }
                    gt_annotations.append(gt_annotation)
                    gt_annotation_id += 1
        if os.path.exists(dt_annotation_file):
            with open(dt_annotation_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id, bbox,score =  get_bbox_info(line, image_info['width'], image_info['height'],is_obb=True)
                    category_set.add(class_id)
                    dt_annotation = {
                        "id": dt_annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": bbox,
                        "area":claculate_area(bbox),
                        "iscrowd": 0,
                        "score":score
                    }
                    dt_annotations.append(dt_annotation)
                    dt_annotation_id += 1

    for category_id in sorted(category_set):
        categories.append({"id": category_id, "name": str(category_id)})

    gt_coco_format = {
        "images": images,
        "annotations": gt_annotations,
        "categories": categories
    }
    dt_coco_format = {
        "images": images,
        "annotations": dt_annotations,
        "categories": categories
    }
    with open(gt_dir +'/coco_ann.json', 'w') as f:
        json.dump(gt_coco_format, f, indent=4)
    with open(dt_dir +'/coco_ann.json', 'w') as f:
        json.dump(dt_coco_format, f, indent=4)



if __name__ == '__main__':
    # gt_path = input("gt path: ")
    gt_path = "/home/iust/UAV_Vision/datasetv.3.*/Download/test"
    dt_path = "/home/iust/UAV_Vision/ultralytics/runs/obb/predict/labels"    

    create_coco_json(gt_path, dt_path)