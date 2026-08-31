try:
    import faster_coco_eval
    faster_coco_eval.init_as_pycocotools()
except (ImportError, Exception):
    pass
