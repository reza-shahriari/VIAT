
class config():
    
    def __init__(self,gt_path,tracker_path,det_path):
        self.gt_path = gt_path
        self.tracker_path = tracker_path
        self.det_path = det_path
        self.stadium_downloads = ['_all','_download','_stadium']