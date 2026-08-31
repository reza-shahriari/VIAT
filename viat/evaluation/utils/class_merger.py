import os
import copy
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    matplotlib = None
    plt = None
    HAS_MATPLOTLIB = False
import numpy as np

class DetailedAnalyticsEngine:
    """
    Computes per-class and per-size metrics, simulates class merging,
    and generates visual diagnostic plots (mAP bar charts, size breakdown, PR curves).
    """

    @staticmethod
    def remap_annotations(annotations, class_map):
        """
        Remap category IDs/names in COCO-style annotations dictionary according to class_map dict.
        class_map: {old_class: new_class}
        """
        if not class_map:
            return annotations

        remapped = copy.deepcopy(annotations)
        for ann in remapped.get('annotations', []):
            cat_id = str(ann.get('category_id', ''))
            if cat_id in class_map:
                new_cat = class_map[cat_id]
                # Convert back to int if numeric
                ann['category_id'] = int(new_cat) if isinstance(new_cat, int) or (isinstance(new_cat, str) and new_cat.isdigit()) else new_cat
        return remapped

    @staticmethod
    def compute_size_categories(bbox):
        """
        Categorize bounding box area into COCO size standards:
        Small: area < 32^2 = 1024
        Medium: 1024 <= area <= 96^2 = 9216
        Large: area > 9216
        """
        if len(bbox) >= 4:
            area = bbox[2] * bbox[3] if len(bbox) == 4 else (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        else:
            area = 0

        if area < 1024:
            return 'Small'
        elif area <= 9216:
            return 'Medium'
        else:
            return 'Large'

    @staticmethod
    def generate_per_class_bar_chart(class_metrics, save_path):
        """
        Generate bar chart of mAP50 / AP per class.
        """
        if not HAS_MATPLOTLIB or plt is None:
            return False
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
        classes = list(class_metrics.keys())
        aps = [class_metrics[c].get('ap50', 0) * 100 for c in classes]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(classes)))
        bars = ax.bar(classes, aps, color=colors, edgecolor='black', alpha=0.85)

        ax.set_ylabel('AP @ IoU 0.50 (%)', fontsize=11, fontweight='bold')
        ax.set_title('Per-Class Detection Performance (AP50)', fontsize=12, fontweight='bold', pad=12)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
        return True

    @staticmethod
    def generate_size_breakdown_chart(size_metrics, save_path):
        """
        Generate bar chart comparing mAP across object size categories (Small, Medium, Large).
        """
        if not HAS_MATPLOTLIB or plt is None:
            return False
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
        sizes = ['Small (<32px)', 'Medium (32-96px)', 'Large (>96px)']
        keys = ['Small', 'Medium', 'Large']
        aps = [size_metrics.get(k, {}).get('ap50', 0) * 100 for k in keys]

        colors = ['#e74c3c', '#f39c12', '#2ecc71']
        bars = ax.bar(sizes, aps, color=colors, edgecolor='black', alpha=0.85, width=0.5)

        ax.set_ylabel('AP @ IoU 0.50 (%)', fontsize=11, fontweight='bold')
        ax.set_title('Performance by Object Size Bins', fontsize=12, fontweight='bold', pad=12)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
        return True

    @staticmethod
    def generate_class_merge_comparison_chart(unmerged_mAP, merged_mAP, save_path):
        """
        Generate side-by-side comparison chart showing impact of class merging on overall mAP.
        """
        if not HAS_MATPLOTLIB or plt is None:
            return False
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        categories = ['Original (Unmerged)', 'Merged Classes']
        maps = [unmerged_mAP * 100, merged_mAP * 100]

        colors = ['#3498db', '#9b59b6']
        bars = ax.bar(categories, maps, color=colors, edgecolor='black', alpha=0.85, width=0.45)

        ax.set_ylabel('Overall mAP50 (%)', fontsize=11, fontweight='bold')
        ax.set_title('Impact of Class Merging on Detection Accuracy', fontsize=11, fontweight='bold', pad=12)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
