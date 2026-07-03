import os
import json

def generate_analytics_report(project_json, output_md, base_raya_file=None):
    """
    Parses a VIAT Project JSON file and generates a Markdown report 
    detailing labeler analytics. If a base raya text is stored in the project,
    or if a base_raya_file is provided, appends a comparison report.
    """
    try:
        with open(project_json, 'r') as f:
            data = json.load(f)
            
        analytics = data.get('labeler_analytics', {})
        if not analytics:
            # Maybe it's an old project that wasn't saved with analytics
            return False, "No labeler analytics found in this project file."
            
        base_annotations = analytics.get('base_annotations', {})
            
        prompts = analytics.get('prompts', [])
        tool_usage = analytics.get('tool_usage', {})
        deleted_frames = set(data.get('deleted_frames', []))
        
        # We can also calculate annotation sources from the annotations themselves
        annotations = data.get('annotations', [])
        source_counts = {}
        for ann in annotations:
            src = ann.get('original_source', ann.get('source', 'unknown'))
            source_counts[src] = source_counts.get(src, 0) + 1
            
        # Build the Markdown report
        report = f"# Labeler Analytics Report\n\n"
        report += f"**Project File:** `{os.path.basename(project_json)}`\n\n"
        
        report += "## High-Level Frame Metrics\n\n"
        report += "| Metric | Count |\n"
        report += "|--------|-------|\n"
        report += f"| Frames Removed (`Ctrl+X`) | {len(deleted_frames)} |\n\n"
        
        report += "## Tool Usage Summary\n\n"
        report += "| Tool | Times Activated |\n"
        report += "|------|-----------------|\n"
        report += f"| Zero-Shot Detection | {tool_usage.get('zero_shot', 0)} |\n"
        report += f"| Object Tracking | {tool_usage.get('tracking', 0)} |\n"
        report += f"| Frame Interpolation | {tool_usage.get('interpolation', 0)} |\n"
        report += f"| Magic Wand (SAM) | {tool_usage.get('magic_wand', 0)} |\n\n"
        
        report += "## Zero-Shot Prompts Used\n\n"
        if prompts:
            report += "| Prompt | Model | Frames Count |\n"
            report += "|--------|-------|--------------|\n"
            for p in prompts:
                report += f"| `{p.get('prompt', '')}` | {p.get('model', '')} | {p.get('frames_count', 0)} |\n"
        else:
            report += "No Zero-Shot prompts were used.\n"
            
        report += "\n## Current Annotations Source Breakdown\n\n"
        report += "This shows how the currently existing bounding boxes were originally created:\n\n"
        if source_counts:
            report += "| Source Tool | Number of Labels |\n"
            report += "|-------------|------------------|\n"
            for src, count in sorted(source_counts.items()):
                report += f"| {src} | {count} |\n"
        else:
            report += "No annotations found.\n"
            
        if not base_annotations and base_raya_file and os.path.exists(base_raya_file):
            try:
                from viat.utils.file_operations import import_annotations
                from viat.annotation import BoundingBox
                
                _, _, base_frames_parsed = import_annotations(base_raya_file, BoundingBox, 640, 480, data.get('class_colors', {}))
                base_annotations = {}
                for f_num, objs in base_frames_parsed.items():
                    base_annotations[str(f_num)] = []
                    for obj in objs:
                        base_annotations[str(f_num)].append(obj.to_dict())
            except Exception as e:
                report += f"\n\n*Error reading base annotation file: {e}*\n"
                
        if base_annotations:
            try:
                from viat.utils.compare_raya import generate_comparison_report_string
                
                def extract_frames(source_dict):
                    frames_dict = {}
                    for frame_str, anns in source_dict.items():
                        frame_num = int(frame_str)
                        objs = []
                        for ann in anns:
                            rect = ann.get('rect', {})
                            # The comparison expects a list [x, y, w, h]
                            box = [rect.get('x', 0), rect.get('y', 0), rect.get('width', 0), rect.get('height', 0)]
                            objs.append({
                                "class_name": ann.get('class_name', 'unknown'),
                                "box": box,
                                "attributes": ann.get('attributes', [])
                            })
                        frames_dict[frame_num] = objs
                    return frames_dict
                
                base_classes = list(data.get('class_colors', {}).keys()) # Base classes don't matter much for the report
                base_deleted = set() # We don't store base deleted frames in json right now
                base_frames = extract_frames(base_annotations)
                
                mod_classes = list(data.get('class_colors', {}).keys())
                mod_deleted = deleted_frames
                mod_frames = extract_frames(data.get('frame_annotations', {}))
                
                comp_report = generate_comparison_report_string(
                    base_classes, base_deleted, base_frames,
                    mod_classes, mod_deleted, mod_frames,
                    "Embedded Pre-labels", os.path.basename(project_json)
                )
                
                report += "\n---\n\n"
                report += f"## Comparison against Base Pre-labels\n\n"
                report += f"*(These pre-labels were automatically stored inside the project file when first imported)*\n\n"
                report += comp_report
            except Exception as ce:
                import traceback
                report += f"\n\n*Error generating comparison report: {ce}*\n\n```\n{traceback.format_exc()}\n```\n"

        with open(output_md, 'w') as f:
            f.write(report)
            
        return True, "Analytics report generated successfully."
        
    except Exception as e:
        return False, str(e)
