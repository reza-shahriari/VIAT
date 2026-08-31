try:
    import faster_coco_eval
    faster_coco_eval.init_as_pycocotools()
except (ImportError, Exception):
    pass

from .eval import Evaluator
from . import datasets
from . import metrics
from . import plotting
from . import utils
