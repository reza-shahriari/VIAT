from .file_operations import (
    save_project,
    load_project,
    export_annotations,
    get_recent_projects,
    get_last_project,
    save_last_state,
    load_last_state,
    get_config_directory,
    update_recent_projects,
    export_image_dataset_pascal_voc,
    export_image_dataset_yolo,
    export_image_dataset_coco,
    export_standard_annotations,
    import_annotations,
    backup_before_save,
    load_project_with_backup
)
from .im_tools import (
    calculate_frame_hash,
    mse_similarity,
    create_thumbnail,
)
# ``ui_creator`` imports widgets, while some widgets import utility modules
# (for example ``icon_provider``).  Importing UICreator eagerly here therefore
# creates a package-initialization cycle when the widgets package is imported.
# Keep the historical ``from viat.utils import UICreator`` API, but resolve it
# only when a caller actually requests the class.
def __getattr__(name):
    if name == "UICreator":
        from .ui_creator import UICreator

        return UICreator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
try:
    from .performance import PerfomanceManger, PerformanceManager
except ImportError:
    pass
from .dataset_manager import (
    import_dataset_dialog,
    load_dataset,
    export_dataset_dialog,
    export_dataset,
)
