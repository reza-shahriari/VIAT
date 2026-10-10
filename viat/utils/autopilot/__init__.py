"""Autopilot: unattended video labeling (detect -> life-cycle track -> offline repair ->
risk-ranked review -> random-frame audit). See reports/Fully automated video annotation.md."""
from .pipeline import (AutopilotConfig, FrameSource, Cancelled, run_autopilot, save_result,
                       load_result, result_path, source_dir, default_workdir)
from .audit import upper_bound, samples_needed, sample_frames

__all__ = ['AutopilotConfig', 'FrameSource', 'Cancelled', 'run_autopilot', 'save_result', 'load_result',
           'result_path', 'source_dir', 'default_workdir', 'upper_bound', 'samples_needed', 'sample_frames']
