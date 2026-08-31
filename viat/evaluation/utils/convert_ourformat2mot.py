#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import cv2
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

import pathlib
try:
    from viat.evaluation.conf.configs import config
except ImportError:
    try:
        from ..conf.configs import config
    except ImportError:
        from conf.configs import config
import time

def convert_to_mot(input_gt_path,input_dt_path,Config :config,size_thr,quality_thr):
    # reading input file
    # videos should be in groundtruth_txt_paths to! 
    gt_txt_paths = glob.glob(input_gt_path[0]+'/*.txt')
    if len(input_gt_path) >1:
        gt_txt_paths.extend(glob.glob(input_gt_path[1]+'/*.txt'))
    dts_txt_paths = glob.glob(input_dt_path[0]+'/*.txt')
    if len(input_dt_path) >1:
        dts_txt_paths.extend(glob.glob(input_dt_path[1]+'/*.txt'))
    size_thr = size_thr
    quality_thr = quality_thr
    names = []
    name2path_dict = {}
    gt_lines_dict = {}
    dt_lines_dict = {}
    for txt_path in gt_txt_paths:
        lines = []
        assert os.path.isfile(txt_path),"No file found! :"+txt_path
        with open (txt_path,'r') as f:
            lines = f.readlines()
        assert len(lines),"No line in it :"+txt_path
        name = os.path.splitext(os.path.split(txt_path)[1])[0]
        name2path_dict[name] = txt_path
        names.append(name)
        gt_lines_dict[name] = lines
    
    for txt_path in dts_txt_paths:
        lines = []
        assert os.path.isfile(txt_path),"No file found! :"+txt_path
        with open (txt_path,'r') as f:
            lines = f.readlines()
        assert len(lines),"No line in it :"+txt_path
        name = os.path.splitext(os.path.split(txt_path)[1])[0]
        dt_lines_dict[name] = lines
        
    gt_out_path = Config.gt_path +'/MOT16-train/'
    pathlib.Path(gt_out_path).mkdir(parents=True, exist_ok=True)
    dt_out_path = Config.tracker_path + '/MOT16-train/botsort/data' 
    pathlib.Path(dt_out_path).mkdir(parents=True, exist_ok=True)
    for k,v in gt_lines_dict.items():
        assert dt_lines_dict.get(k,False),"name: "+ k +" in gt but not in dets"
        dt_lines = dt_lines_dict[k]
        gt_lines = gt_lines_dict[k]
        assert len(gt_lines) == len(dt_lines),"lines count are not equal for video:"+k
        gt_output_text = []
        dt_output_text = []
        seq_len = 0 
        for i in range(len(gt_lines)):
            gt_write = True
            dt_write = True
            if len(gt_lines[i].strip()) <= 5:
                gt_output_text.append("")
                gt_write = False
            else:
                gt_list =gt_lines[i].strip().split(';')[0]
                gt_list = gt_list.split(',')
                if len(gt_list)==8:
                    gt_list = gt_list[1:-1]
                else:
                    gt_list = gt_list[1:]
                gt_list[-1] = gt_list[-1][:-3]
                if float(gt_list[-2]) < size_thr or float(gt_list[-1])<quality_thr:
                    gt_output_text.append("")
                    dt_output_text.append("")
                    dt_write = False
                    gt_write = False
                    continue
            if len(dt_lines[i].strip())<=5:
                dt_output_text.append("")
                dt_write = False
            else:
                dt_list = dt_lines[i].split(",")
                dt_list[-1] = dt_list[-1][:-3]
                dt_list[0] = dt_list[0][2:]
                if float(dt_list[-1]) > 1.0:
                    gt_output_text.append("")
                    dt_output_text.append("")
                    continue
            if gt_write:
                #   frame          id           bb_left           bb_top            bb_width            bb_height           conf
                s = str(i+1) + ' ' + '1' + ' ' + gt_list[0] + ' ' + gt_list[1] + ' ' + gt_list[2] + ' ' + gt_list[3]  + ' ' + '1' +' 1 '+'\n'
                gt_output_text.append(s)
            if dt_write:
                rec = dt_list[1:-1]
                rec = [float(x) for x in rec]
                #   frame           id               bb_left            bb_top                 bb_width                    bb_height            conf
                s = str(i+1) + ' ' + '1' + ' ' +   str(rec[0]) + ' ' + str(rec[1]) + ' ' + str(rec[2]-rec[0]) + ' ' + str(rec[3]-rec[1])  + ' ' + str(dt_list[-1]) + ' 0 0 -1' +'\n'
                dt_output_text.append(s)
        
        with open(dt_out_path+'/'+k+'.txt','w')as f:
            for i in dt_output_text:
                f.write(i)
        pathlib.Path(gt_out_path+k+'/gt').mkdir(parents=True, exist_ok=True)        
        
        with open(gt_out_path+k+'/gt'+'/gt.txt','w')as f:
            for i in gt_output_text:
                f.write(i)
        
        exts=['.mp4','.avi','.mkv','.mov','.webm','.mpg', '.MOV','.m4v']
        vid = None
        for i in exts:
            video_path = name2path_dict[k][:-4] + i
            if os.path.isfile(video_path):
                vid = cv2.VideoCapture(video_path)
        assert vid,"no video found "+video_path
        
        fps = vid.get(cv2.CAP_PROP_FPS)
        _,im = vid.read()
        height = im.shape[0]
        width = im.shape[1]
        pathlib.Path(gt_out_path+k).mkdir(parents=True, exist_ok=True)        
        with open(gt_out_path + k+'/seqinfo.ini','w') as f:
             f.write('[Sequence]\n')
             f.write('name='+k+'\n')
             f.write('imDir=img1'+'\n')
             f.write('frameRate='+str(int(fps))+'\n')
             f.write('seqLength='+str(len(gt_lines_dict[k]))+'\n')
             f.write('imWidth='+str(width)+'\n')
             f.write('imHeight='+str(height)+'\n')
             f.write('imExt=.jpg')
        pathlib.Path(gt_out_path+k+'/det').mkdir(parents=True, exist_ok=True)        
        with open(gt_out_path+k+'/det'+'/det.txt','w')as f:
            for i in dt_output_text:
                f.write(i)
    seqmaps_out_path = os.path.splitext(os.path.split(gt_txt_paths[0])[0])[0] +'/Track/gt/seqmaps'
    pathlib.Path(seqmaps_out_path).mkdir(parents=True, exist_ok=True)        
    with open(seqmaps_out_path+'/MOT16-train.txt','w') as f:
        f.write("MOT16\n")
        for i in names:
            f.write(i+'\n')

    



