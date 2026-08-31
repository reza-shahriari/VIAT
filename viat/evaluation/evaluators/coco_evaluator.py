""" version ported from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    Notes:
        1) The default area thresholds here follows the values defined in COCO, that is,
        small:           area <= 32**2
        medium: 32**2 <= area <= 96**2
        large:  96**2 <= area.
        If area is not specified, all areas are considered.

        2) COCO's ground truths contain an 'area' attribute that is associated with the segmented area if
        segmentation-level information exists. While coco uses this 'area' attribute to distinguish between
        'small', 'medium', and 'large' objects, this implementation simply uses the associated bounding box
        area to filter the ground truths.

        3) COCO uses floating point bounding boxes, thus, the calculation of the box area
        for IoU purposes is the simple open-ended delta (x2 - x1) * (y2 - y1).
        PASCALVOC uses integer-based bounding boxes, and the area includes the outer edge,
        that is, (x2 - x1 + 1) * (y2 - y1 + 1). This implementation assumes the open-ended (former)
        convention for area calculation.
"""

try:
    import faster_coco_eval
    faster_coco_eval.init_as_pycocotools()
    HAS_FASTER_COCO_EVAL = True
except (ImportError, Exception):
    HAS_FASTER_COCO_EVAL = False

from collections import defaultdict
import pickle
import numpy as np
try:
    from viat.evaluation.utils.enumerators import BBFormat
except ImportError:
    try:
        from ..utils.enumerators import BBFormat
    except ImportError:
        from viat.evaluation.bounding_box import BBFormat
import sys
try:
    from shapely.geometry import Polygon
except ImportError:
    Polygon = None
import cv2
def get_coco_summary(groundtruth_bbs, detected_bbs):
    """Calculate the 12 standard metrics used in COCOEval,
        AP, AP50, AP75,
        AR1, AR10, AR100,
        APsmall, APmedium, APlarge,
        ARsmall, ARmedium, ARlarge.

        When no ground-truth can be associated with a particular class (NPOS == 0),
        that class is removed from the average calculation.
        If for a given calculation, no metrics whatsoever are available, returns NaN.

    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
    Returns:
            A dictionary with one entry for each metric.
    """
    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    def _evaluate(iou_threshold, max_dets, area_range):
        global acc,ev
        # accumulate evaluations on a per-class basis
        _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})
        for img_id, class_id in _bbs:
            # print(img_id)
            ev = _evaluate_image(
                _bbs[img_id, class_id]["dt"],
                _bbs[img_id, class_id]["gt"],
                _ious[img_id, class_id],
                iou_threshold,
                max_dets,
                area_range,
            )
            # if iou_threshold == 0.5:
            #     for i, d in enumerate(_bbs[img_id, class_id]["dt"]):
            #         d = d
            #         print(ev,d, img_id,d.get_absolute_bounding_box(format=BBFormat.XYX2Y2), d.get_confidence(), _bbs[img_id, class_id]["gt"])
            #     sys.exit()
            acc = _evals[class_id]
            acc["scores"].append(ev["scores"])
            acc["matched"].append(ev["matched"])
            acc["NP"].append(ev["NP"])

        # now reduce accumulations
        for class_id in _evals:
            acc = _evals[class_id]
            acc["scores"] = np.concatenate(acc["scores"])
            acc["matched"] = np.concatenate(acc["matched"]).astype(bool)
            acc["NP"] = np.sum(acc["NP"])

        res = []
        # run ap calculation per-class
        for class_id in _evals:
            ev = _evals[class_id]
            res.append({
                "class": class_id,
                **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"]),
            })
        return res

    iou_thresholds = np.linspace(0.4, 0.95, int(np.round((0.95 - 0.4) / 0.05)) + 1, endpoint=True)

    # compute simple AP with all thresholds, using up to 100 dets, and all areas
    full = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(0, np.inf))
        for i in iou_thresholds
    }

    AP50 = np.mean([x['AP'] for x in full[0.50] if x['AP'] is not None])
    AP40 = np.mean([x['AP'] for x in full[0.40] if x['AP'] is not None])
    # AP20 = np.mean([x['AP'] for x in full[0.20] if x['AP'] is not None])
    # AP10 = np.mean([x['AP'] for x in full[0.10] if x['AP'] is not None])
    # AP05 = np.mean([x['AP'] for x in full[0.05] if x['AP'] is not None])
    # argmax = np.argmax(full[0.5][0]["recall"]*full[0.5][0]["precision"])
    # print([x['precision'] for x in full[0.50] if x['precision'] is not None])
    TP = [x['TP'] for x in full[0.50] if x['TP'] is not None]
    if len(TP):
        TP = TP[0]
    else:
        TP = 0
    FP = [x['FP'] for x in full[0.50] if x['FP'] is not None]
    if len(FP):
        FP = FP[0]
    else:
        FP = 0
    FN = [x['total positives'] for x in full[0.50] if x['FP'] is not None]
    if len(FN):
        FN = FN[0]-TP
    else:
        FN =0
    TPs = [x['TPs'] for x in full[0.50]][0]
    FPs = [x['FPs'] for x in full[0.50]][0]
    FNs = [x['FNs'] for x in full[0.50]][0]
    F1 = [x['F1'] for x in full[0.50]][0]
    score = [x['score'] for x in full[0.50]][0]
    argmax = [x['argmax'] for x in full[0.50]][0]
    # score = [x['Scores'] for x in full[0.50] if x['Scores'] is not None][0][argmax]
    AP75 = np.mean([x['AP'] for x in full[0.75] if x['AP'] is not None])
    AP = np.mean([x['AP'] for k in full for x in full[k] if x['AP'] is not None])

    # max recall for 100 dets can also be calculated here
    AR100 = np.mean(
        [x['TP'] / x['total positives'] for k in full for x in full[k] if x['TP'] is not None])

    small16 = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(0, 16**2))
        for i in iou_thresholds
    }
    APsmall16 = [x['AP'] for k in small16 for x in small16[k] if x['AP'] is not None]
    APsmall16 = np.nan if APsmall16 == [] else np.mean(APsmall16)
    
    small32 = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(16**2, 32**2))
        for i in iou_thresholds
    }
    small = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(0, 32**2))
        for i in iou_thresholds
    }
    APsmall32 = [x['AP'] for k in small32 for x in small32[k] if x['AP'] is not None]
    APsmall32 = np.nan if APsmall32 == [] else np.mean(APsmall32)
    
    ARsmall = [
        x['TP'] / x['total positives'] for k in small for x in small[k] if x['TP'] is not None
    ]
    ARsmall = np.nan if ARsmall == [] else np.mean(ARsmall)

    medium64 = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(32**2, 64**2))
        for i in iou_thresholds
    }
    medium = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(32**2, 96**2))
        for i in iou_thresholds
    }
    APmedium64 = [x['AP'] for k in medium64 for x in medium64[k] if x['AP'] is not None]
    APmedium64 = np.nan if APmedium64 == [] else np.mean(APmedium64)
    
    medium96 = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(64**2, 96**2))
        for i in iou_thresholds
    }
    APmedium96 = [x['AP'] for k in medium96 for x in medium96[k] if x['AP'] is not None]
    APmedium96 = np.nan if APmedium96 == [] else np.mean(APmedium96)
    
    ARmedium = [
        x['TP'] / x['total positives'] for k in medium for x in medium[k] if x['TP'] is not None
    ]
    ARmedium = np.nan if ARmedium == [] else np.mean(ARmedium)

    large128 = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(96**2, 128**2))
        for i in iou_thresholds
    }
    large = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(96**2, np.inf))
        for i in iou_thresholds
    }
    APlarge128 = [x['AP'] for k in large128 for x in large128[k] if x['AP'] is not None]
    APlarge128 = np.nan if APlarge128 == [] else np.mean(APlarge128)
    
    large160 = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(128**2, 160**2))
        for i in iou_thresholds
    }
    APlarge160 = [x['AP'] for k in large160 for x in large160[k] if x['AP'] is not None]
    APlarge160 = np.nan if APlarge160 == [] else np.mean(APlarge160)
    
    large192 = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(160**2, 192**2))
        for i in iou_thresholds
    }
    APlarge192 = [x['AP'] for k in large192 for x in large192[k] if x['AP'] is not None]
    APlarge192 = np.nan if APlarge192 == [] else np.mean(APlarge192)
    
    large224 = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(192**2, 224**2))
        for i in iou_thresholds
    }
    APlarge224 = [x['AP'] for k in large224 for x in large224[k] if x['AP'] is not None]
    APlarge224 = np.nan if APlarge224 == [] else np.mean(APlarge224)
    
    large256 = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(224**2, 256**2))
        for i in iou_thresholds
    }
    APlarge256 = [x['AP'] for k in large256 for x in large256[k] if x['AP'] is not None]
    APlarge256 = np.nan if APlarge256 == [] else np.mean(APlarge256)
    
    large288 = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(256**2, 288**2))
        for i in iou_thresholds
    }
    APlarge288 = [x['AP'] for k in large288 for x in large288[k] if x['AP'] is not None]
    APlarge288 = np.nan if APlarge288 == [] else np.mean(APlarge288)
    
    large320 = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(288**2, 320**2))
        for i in iou_thresholds
    }
    APlarge320 = [x['AP'] for k in large320 for x in large320[k] if x['AP'] is not None]
    APlarge320 = np.nan if APlarge320 == [] else np.mean(APlarge320)
    
    ARlarge = [
        x['TP'] / x['total positives'] for k in large for x in large[k] if x['TP'] is not None
    ]
    ARlarge = np.nan if ARlarge == [] else np.mean(ARlarge)

    max_det1 = {
        i: _evaluate(iou_threshold=i, max_dets=1, area_range=(0, np.inf))
        for i in iou_thresholds
    }
    AR1 = np.mean([
        x['TP'] / x['total positives'] for k in max_det1 for x in max_det1[k] if x['TP'] is not None
    ])

    max_det10 = {
        i: _evaluate(iou_threshold=i, max_dets=10, area_range=(0, np.inf))
        for i in iou_thresholds
    }
    AR10 = np.mean([
        x['TP'] / x['total positives'] for k in max_det10 for x in max_det10[k]
        if x['TP'] is not None
    ])
    APsmall = [x['AP'] for k in small for x in small[k] if x['AP'] is not None]
    APsmall = np.nan if APsmall == [] else np.mean(APsmall)
    APmedium = [x['AP'] for k in medium for x in medium[k] if x['AP'] is not None]
    APmedium = np.nan if APmedium == [] else np.mean(APmedium)
    APlarge = [x['AP'] for k in large for x in large[k] if x['AP'] is not None]
    APlarge = np.nan if APlarge == [] else np.mean(APlarge)

    # Per-class AP breakdown
    per_class_metrics = {}
    for x in full[0.50]:
        c_id = x['class']
        c_ap50 = x['AP']
        c_ap_list = [entry['AP'] for k in full for entry in full[k] if entry['class'] == c_id and entry['AP'] is not None]
        c_ap = np.mean(c_ap_list) if len(c_ap_list) > 0 else np.nan
        per_class_metrics[c_id] = {'AP50': c_ap50, 'AP': c_ap}

    per_size_metrics = {
        'Small (<32²)': APsmall,
        'Medium (32²-96²)': APmedium,
        'Large (>96²)': APlarge
    }

    return {
        "Precision": full[0.5][0]["precision"][argmax] if argmax is not None and len(full[0.5]) > 0 else 0.0,
        "Recall": full[0.5][0]["recall"][argmax] if argmax is not None and len(full[0.5]) > 0 else 0.0,
        "F1": F1,
        "AP": AP,
        "AP40": AP40,
        "AP50": AP50,
        "AP75": AP75,
        "APsmall": APsmall,
        "APmedium": APmedium,
        "APlarge": APlarge,
        "APsmall16": APsmall16,
        "APsmall32": APsmall32,
        "APmedium64": APmedium64,
        "APmedium96": APmedium96,
        "APlarge128": APlarge128,
        "APlarge160": APlarge160,
        "APlarge192": APlarge192,
        "APlarge224": APlarge224,
        "APlarge256": APlarge256,
        "APlarge288": APlarge288,
        "APlarge320": APlarge320,
        "AR1": AR1,
        "AR10": AR10,
        "AR100": AR100,
        "ARsmall": ARsmall,
        "ARmedium": ARmedium,
        "ARlarge": ARlarge,
        "TP(zero_score)": TP,
        "FP(zero_score)": FP,
        "FN(zero_score)": FN,
        "Score": score,
        "TP": TPs,
        "FP": FPs,
        "FN": FNs,
        "per_class_metrics": per_class_metrics,
        "per_size_metrics": per_size_metrics
    }


def get_coco_metrics(
        groundtruth_bbs,
        detected_bbs,
        iou_threshold=0.5,
        area_range=(0, np.inf),
        max_dets=100,
):
    """ Calculate the Average Precision and Recall metrics as in COCO's official implementation
        given an IOU threshold, area range and maximum number of detections.
    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
            iou_threshold : float
                Intersection Over Union (IOU) value used to consider a TP detection.
            area_range : (numerical x numerical)
                Lower and upper bounds on annotation areas that should be considered.
            max_dets : int
                Upper bound on the number of detections to be considered for each class in an image.

    Returns:
            A list of dictionaries. One dictionary for each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['TP']: total number of True Positive detections;
            dict['FP']: total number of False Positive detections;

            if there was no valid ground truth for a specific class (total positives == 0),
            all the associated keys default to None
    """

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    # accumulate evaluations on a per-class basis
    _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})

    for img_id, class_id in _bbs:
        ev = _evaluate_image(
            _bbs[img_id, class_id]["dt"],
            _bbs[img_id, class_id]["gt"],
            _ious[img_id, class_id],
            iou_threshold,
            max_dets,
            area_range,
        )
        acc = _evals[class_id]
        acc["scores"].append(ev["scores"])
        acc["matched"].append(ev["matched"])
        acc["NP"].append(ev["NP"])

    # now reduce accumulations
    for class_id in _evals:
        acc = _evals[class_id]
        acc["scores"] = np.concatenate(acc["scores"])
        acc["matched"] = np.concatenate(acc["matched"]).astype(bool)
        acc["NP"] = np.sum(acc["NP"])

    res = {}
    # run ap calculation per-class
    for class_id in _evals:
        ev = _evals[class_id]
        res[class_id] = {
            "class": class_id,
            **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"])
        }
    return res


def _group_detections(dt, gt):
    """ simply group gts and dts on a imageXclass basis """
    bb_info = defaultdict(lambda: {"dt": [], "gt": []})
    for d in dt:
        i_id = d.get_image_name()
        c_id = d.get_class_id()
        bb_info[i_id, c_id]["dt"].append(d)
    for g in gt:
        i_id = g.get_image_name()
        c_id = g.get_class_id()
        bb_info[i_id, c_id]["gt"].append(g)
    return bb_info


def _get_area(a):
    """ COCO does not consider the outer edge as included in the bbox """
    x, y, x2, y2 = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    return (x2 - x) * (y2 - y)


def _jaccard(a, b):
    if a._format != BBFormat.OBB and b._format != BBFormat.OBB:
        xa, ya, x2a, y2a = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        xb, yb, x2b, y2b = b.get_absolute_bounding_box(format=BBFormat.XYX2Y2)

        # innermost left x
        xi = max(xa, xb)
        # innermost right x
        x2i = min(x2a, x2b)
        # same for y
        yi = max(ya, yb)
        y2i = min(y2a, y2b)

        # calculate areas
        Aa = max(x2a - xa, 0) * max(y2a - ya, 0)
        Ab = max(x2b - xb, 0) * max(y2b - yb, 0)
        Ai = max(x2i - xi, 0) * max(y2i - yi, 0)
        return Ai / (Aa + Ab - Ai)
    else:
        a_bb = a.get_absolute_bounding_box(format=BBFormat.OBB)
        b_bb = b.get_absolute_bounding_box(format=BBFormat.OBB)
        pa = [(a_bb[2*i],a_bb[2*i+1]) for i in range(len(a_bb)//2)]
        pb = [(b_bb[2*i],b_bb[2*i+1]) for i in range(len(b_bb)//2)]
        if Polygon is None:
            raise ImportError("The 'shapely' package is required for Oriented Bounding Box (OBB) IoU calculation. Please install it: pip install shapely")
        poly1 = Polygon(pa)
        poly2 = Polygon(pb)
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0

        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        
        iou = intersection_area / union_area if union_area > 0 else 0.0
        return iou

def jaccard(a, b):
    xa, ya, x2a, y2a = a
    xb, yb, x2b, y2b = b

    # innermost left x
    xi = max(xa, xb)
    # innermost right x
    x2i = min(x2a, x2b)
    # same for y
    yi = max(ya, yb)
    y2i = min(y2a, y2b)

    # calculate areas
    Aa = max(x2a - xa, 0) * max(y2a - ya, 0)
    Ab = max(x2b - xb, 0) * max(y2b - yb, 0)
    Ai = max(x2i - xi, 0) * max(y2i - yi, 0)
    return Ai / (Aa + Ab - Ai)


def _compute_ious(dt, gt):
    """ compute pairwise ious """
    if len(gt):
        img = gt[0]._image_name +'.jpg'

    ious = np.zeros((len(dt), len(gt)))
    for g_idx, g in enumerate(gt):
        for d_idx, d in enumerate(dt):
            ious[d_idx, g_idx] = _jaccard(d, g)
            # im = cv2.imread('/home/iust/UAV_Vision/datasetv.3.*/Download/test/'+img)
            # a_bb = d.get_absolute_bounding_box(format=BBFormat.OBB)
            # b_bb = g.get_absolute_bounding_box(format=BBFormat.OBB)
            # pa = np.array([[int(a_bb[2*i]),int(a_bb[2*i+1])] for i in range(len(a_bb)//2)],
            #    np.int32)
            # pb = np.array([[int(b_bb[2*i]),int(b_bb[2*i+1])] for i in range(len(b_bb)//2)],
            #    np.int32)
            # im = cv2.polylines(im, [pa], 
            #           True, (0,0,255), 2)
            # im = cv2.polylines(im, [pb], 
            #           True, (255,0,0), 2)
            # im = cv2.putText(im, str(ious[d_idx,g_idx]), (50, 50) , cv2.FONT_HERSHEY_SIMPLEX ,  
            #        1, (0,255,0), 1, cv2.LINE_AA)
            # cv2.imshow('im: ',cv2.resize(im,(640,640)))
            # cv2.waitKey(0)
    return ious


def _evaluate_image(dt, gt, ious, iou_threshold, max_dets=None, area_range=None):
    """ use COCO's method to associate detections to ground truths """
    # sort dts by increasing confidence
    dt_sort = np.argsort([-d.get_confidence() for d in dt], kind="stable")
    # sort list of dts and chop by max dets
    dt = [dt[idx] for idx in dt_sort[:max_dets]]
    ious = ious[dt_sort[:max_dets]]
    
    # generate ignored gt list by area_range
    def _is_ignore(bb):
        if area_range is None:
            return False
        return not (area_range[0] <= _get_area(bb) <= area_range[1])

    gt_ignore = [_is_ignore(g) for g in gt]

    # sort gts by ignore last
    gt_sort = np.argsort(gt_ignore, kind="stable")
    gt = [gt[idx] for idx in gt_sort]
    gt_ignore = [gt_ignore[idx] for idx in gt_sort]
    ious = ious[:, gt_sort]

    gtm = {}
    dtm = {}
    
    for d_idx, d in enumerate(dt):
        # print(d)
        # information about best match so far (m=-1 -> unmatched)
        iou = min(iou_threshold, 1 - 1e-10)
        m = -1
        for g_idx, g in enumerate(gt):
            # if this gt already matched, and not a crowd, continue
            if g_idx in gtm:
                continue
            # if dt matched to reg gt, and on ignore gt, stop
            if m > -1 and gt_ignore[m] == False and gt_ignore[g_idx] == True:
                break
            # continue to next gt unless better match made
            if ious[d_idx, g_idx] < iou:
                continue
            # if match successful and best so far, store appropriately
            iou = ious[d_idx, g_idx]

            m = g_idx
        # if match made store id of match for both dt and gt
        if m == -1:
            continue
        dtm[d_idx] = m
        gtm[m] = d_idx
    # generate ignore list for dts
    dt_ignore = [
        gt_ignore[dtm[d_idx]] if d_idx in dtm else _is_ignore(d) for d_idx, d in enumerate(dt)
    ]

    # get score for non-ignored dts
    scores = [dt[d_idx].get_confidence() for d_idx in range(len(dt)) if not dt_ignore[d_idx]]
    matched = [d_idx in dtm for d_idx in range(len(dt)) if not dt_ignore[d_idx]]

    n_gts = len([g_idx for g_idx in range(len(gt)) if not gt_ignore[g_idx]])
    return {"scores": scores, "matched": matched, "NP": n_gts}


def _compute_ap_recall(scores, matched, NP, recall_thresholds=None):
    """ This curve tracing method has some quirks that do not appear when only unique confidence thresholds
    are used (i.e. Scikit-learn's implementation), however, in order to be consistent, the COCO's method is reproduced. """
    if NP == 0:
        return {
            "precision": None,
            "recall": None,
            "AP": None,
            "interpolated precision": None,
            "interpolated recall": None,
            "total positives": None,
            "TP": None,
            "FP": None
        }

    # by default evaluate on 101 recall levels
    if recall_thresholds is None:
        recall_thresholds = np.linspace(0.0,
                                        1.00,
                                        int(np.round((1.00 - 0.0) / 0.01)) + 1,
                                        endpoint=True)

    # sort in descending score order
    inds = np.argsort(-scores, kind="stable")
    scores = scores[inds]
    matched = matched[inds]
    # print('\n', matched, '\n')
    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)

    rc = tp / NP
    pr = tp / (tp + fp)
    f1 = 2*pr*rc/(pr+rc)
    try:
        F1 = np.nanmax(f1)
    except:
        F1 = None
    if np.isnan(f1).all() or F1 == None:
        argmax, TP, FP, FN, score = None, None, None, None, None
    else:
        argmax = np.where(2*pr*rc/(pr+rc) == np.nanmax(2*pr*rc/(pr+rc)))[0][0]
        TP = tp[argmax]
        FP = fp[argmax]
        FN = NP-TP
        score = scores[argmax]
    # make precision monotonically decreasing
    i_pr = np.maximum.accumulate(pr[::-1])[::-1]

    rec_idx = np.searchsorted(rc, recall_thresholds, side="left")
    n_recalls = len(recall_thresholds)

    # get interpolated precision values at the evaluation thresholds
    i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

    return {
        "precision": pr, #precision in max F1
        "recall": rc, #recall in max F1
        "F1": F1, #max F1
        "AP": np.mean(i_pr),
        "interpolated precision": i_pr,
        "interpolated recall": recall_thresholds,
        "total positives": NP,
        "TP": tp[-1] if len(tp) != 0 else 0,
        "FP": fp[-1] if len(fp) != 0 else 0,
        "Matched":matched,
        "Scores": scores,
        "TPs": TP, #True Positive num in max F1 score
        "FPs": FP, #False Positive num in max F1 score
        "FNs": FN, #False Negative num in max F1 score
        "score":score, #Score in max F1
        "argmax": argmax #argmax in max F1
    }
