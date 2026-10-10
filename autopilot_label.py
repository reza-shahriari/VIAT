"""Headless Autopilot run: label a whole video without the GUI (e.g. overnight).

    python autopilot_label.py VIDEO --class car=car,vehicle --class person \
        --model sam3.1_l.pt --model yolov8x-worldv2.pt --quality max

The result is saved where VIAT looks for it; open the video in VIAT and use
Tools > Autopilot Review Panel to add the boxes to the project and review them.
"""
import argparse
import json
import sys

from viat.utils.autopilot import AutopilotConfig, FrameSource, run_autopilot, save_result, result_path

QUALITY = {'max': (1, 2, True), 'high': (1, 2, False), 'balanced': (1, 0, False), 'fast': (3, 0, False)}


def parse_class(text):
    name, _, prompts = text.partition('=')
    return {'name': name.strip(), 'prompts': [p.strip() for p in (prompts or name).split(',') if p.strip()]}


def main():
    ap = argparse.ArgumentParser(description='Unattended Autopilot labeling for one video.')
    ap.add_argument('video')
    ap.add_argument('--class', dest='classes', action='append', required=True,
                    help='NAME or NAME=prompt1,prompt2 (repeatable)')
    ap.add_argument('--model', dest='models', action='append', required=True,
                    help='detector key as in the Auto Label dialog, e.g. sam3.1_l.pt (repeatable)')
    ap.add_argument('--start', type=int, default=1, help='first frame (1-based)')
    ap.add_argument('--end', type=int, default=-1, help='last frame (1-based, default: end of video)')
    ap.add_argument('--quality', choices=sorted(QUALITY), default='high')
    ap.add_argument('--judge', dest='judges', action='append', default=None,
                    help='clip | sam_zoom | vlm | learned (repeatable; default clip+sam_zoom+learned)')
    ap.add_argument('--config', help='JSON file with extra AutopilotConfig fields')
    ap.add_argument('--out', help='also write the result JSON here')
    args = ap.parse_args()

    stride, tiles, flip = QUALITY[args.quality]
    extra = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as fh:
            extra = json.load(fh)
    cfg = AutopilotConfig.from_dict({**extra, 'classes': [parse_class(c) for c in args.classes],
                                     'models': args.models, 'start_frame': args.start - 1,
                                     'end_frame': args.end - 1 if args.end > 0 else -1,
                                     'stride': stride, 'tiles': tiles, 'flip_tta': flip,
                                     **({'judges': args.judges} if args.judges is not None else {})})
    source = FrameSource(video_path=args.video)

    def progress(stage, done, total):
        print(f'\r{stage}: {done}/{total}   ', end='', flush=True)

    result = run_autopilot(source, cfg, progress=progress)
    print()
    path = result_path(source)
    save_result(path, result)
    if args.out:
        save_result(args.out, result)
    s = result['stats']
    print(f"{s['tracks']} tracks ({s['review']} flagged), {s['possible_misses']} possible misses, "
          f"{s['boxes']} boxes in {s['seconds']} s\nresult: {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
