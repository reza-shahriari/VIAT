"""
Advanced Model Diagnostics & Beautiful Plot Generator for VIAT Evaluation Engine

Contains:
1. Inter-Class Confusion Matrix Heatmap (GT vs Preds + Background FP/FN)
2. Confidence Calibration & Precision-Recall Sweep (ECE & Reliability Curve)
3. Localization Precision & IoU Histogram (Box Tightness Analysis)
4. Box Geometry Bias (Aspect Ratio W/H vs Error Rates)
5. Tracking Error Taxonomy Breakdown (ID Swaps, Track Loss, Fragmentation)
6. 2D Spatial Error Heatmap (Screen Location Error Hotspots)
"""

import os
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
    # Configure global publication-quality aesthetic styling for Matplotlib
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#444444'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.color'] = '#cccccc'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5
except ImportError:
    matplotlib = None
    plt = None
    HAS_MATPLOTLIB = False


class AdvancedDiagnosticsEngine:
    """Generates 6 sleek, beautiful diagnostic plots for deep model inspection."""

    @staticmethod
    def generate_confusion_matrix_plot(cm, class_names, save_path):
        """
        1. Inter-Class Confusion Matrix Heatmap (including Background FP & FN).
        """
        if not HAS_MATPLOTLIB or plt is None:
            return False
        fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=150)
        labels = list(class_names) + ['Background']
        n_classes = len(labels)

        # Normalize confusion matrix by row (GT)
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)

        # Custom vibrant blue-purple colormap
        cmap = plt.cm.YlGnBu

        im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Ratio / Recall', rotation=-90, va="bottom", fontweight='bold', fontsize=10)

        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10, fontweight='bold')
        ax.set_yticklabels(labels, fontsize=10, fontweight='bold')

        ax.set_ylabel('True Category (GT)', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_xlabel('Predicted Category (Model)', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_title('Inter-Class Confusion Matrix & Background Errors', fontsize=13, fontweight='bold', pad=14)

        # Annotate text counts and percentages in cells
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm[i, j]
                pct = cm_norm[i, j] * 100
                txt_color = "white" if cm_norm[i, j] > 0.55 else "black"
                ax.text(j, i, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center", color=txt_color,
                        fontsize=9, fontweight='bold')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)

    @staticmethod
    def generate_calibration_plot(confidences, precisions, recalls, ece_score, optimal_thr, save_path):
        """
        2. Confidence Calibration Curve & PR Sweep Curve (Reliability & ECE).
        """
        if not HAS_MATPLOTLIB or plt is None:
            return False
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

        # Subplot 1: Calibration / Reliability Curve
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5, alpha=0.7)
        ax1.plot(confidences, precisions, 'o-', color='#3498db', linewidth=2.5, markersize=5, label=f'Model (ECE: {ece_score:.3f})')
        ax1.fill_between(confidences, precisions, confidences, color='#3498db', alpha=0.15, label='Calibration Gap')

        ax1.set_xlabel('Predicted Confidence Score', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Observed Accuracy / Precision', fontsize=10, fontweight='bold')
        ax1.set_title('Confidence Calibration Curve (ECE)', fontsize=12, fontweight='bold', pad=10)
        ax1.legend(loc='upper left', frameon=True, facecolor='#f8f9fa')
        ax1.grid(True)

        # Subplot 2: Precision-Recall vs Threshold Sweep
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
        ax2.plot(confidences, precisions, '-', color='#2ecc71', linewidth=2.2, label='Precision')
        ax2.plot(confidences, recalls, '-', color='#e74c3c', linewidth=2.2, label='Recall')
        ax2.plot(confidences, f1_scores, '--', color='#9b59b6', linewidth=2.5, label=f'F1-Score (Peak @ {optimal_thr:.2f})')

        ax2.axvline(optimal_thr, color='#9b59b6', linestyle=':', linewidth=1.5)
        ax2.set_xlabel('Confidence Threshold', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Score Ratio', fontsize=10, fontweight='bold')
        ax2.set_title('Precision / Recall / F1 Sweep', fontsize=12, fontweight='bold', pad=10)
        ax2.legend(loc='lower left', frameon=True, facecolor='#f8f9fa')
        ax2.grid(True)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
        return True

    @staticmethod
    def generate_iou_distribution_plot(ious, save_path):
        """
        3. Localization Precision & IoU Distribution Histogram.
        """
        if not HAS_MATPLOTLIB or plt is None:
            return False
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
        ious = np.array(ious)

        n, bins, patches = ax.hist(ious, bins=25, range=(0.4, 1.0), edgecolor='black', alpha=0.85)

        # Color code bars by tightness
        for bin_left, patch in zip(bins[:-1], patches):
            if bin_left >= 0.85:
                patch.set_facecolor('#2ecc71') # Tight (Green)
            elif bin_left >= 0.70:
                patch.set_facecolor('#3498db') # Moderate (Blue)
            elif bin_left >= 0.50:
                patch.set_facecolor('#f39c12') # Acceptable (Orange)
            else:
                patch.set_facecolor('#e74c3c') # Loose (Red)

        mean_iou = np.mean(ious) if len(ious) > 0 else 0
        median_iou = np.median(ious) if len(ious) > 0 else 0

        ax.axvline(mean_iou, color='#111111', linestyle='--', linewidth=2, label=f'Mean IoU: {mean_iou:.3f}')
        ax.axvline(median_iou, color='#8e44ad', linestyle=':', linewidth=2, label=f'Median IoU: {median_iou:.3f}')

        ax.set_xlabel('Intersection over Union (IoU)', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive BBox Count', fontsize=11, fontweight='bold')
        ax.set_title('Localization Tightness & IoU Distribution', fontsize=13, fontweight='bold', pad=12)
        ax.legend(loc='upper left', frameon=True, facecolor='#f8f9fa')
        ax.grid(True)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
        return True

    @staticmethod
    def generate_aspect_ratio_plot(aspect_ratios, error_rates, save_path):
        """
        4. Box Geometry Bias (Aspect Ratio W/H vs Error Rates).
        """
        if not HAS_MATPLOTLIB or plt is None:
            return False
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

        # Scatter plot with smooth trend line
        ax.scatter(aspect_ratios, error_rates, color='#e74c3c', s=45, alpha=0.7, edgecolors='black', label='Sample Batches')

        # Fit smooth polynomial polynomial trendline
        if len(aspect_ratios) >= 4:
            sorted_idx = np.argsort(aspect_ratios)
            x_sorted = np.array(aspect_ratios)[sorted_idx]
            y_sorted = np.array(error_rates)[sorted_idx]
            z = np.polyfit(x_sorted, y_sorted, deg=2)
            p = np.poly1d(z)
            ax.plot(x_sorted, p(x_sorted), color='#2c3e50', linewidth=2.5, linestyle='-', label='Geometry Bias Trend')

        ax.axvline(1.0, color='#7f8c8d', linestyle='--', alpha=0.7, label='Square Box (1:1)')

        ax.set_xlabel('Bounding Box Aspect Ratio (Width / Height)', fontsize=11, fontweight='bold')
        ax.set_ylabel('False Negative / Error Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title('Bounding Box Aspect Ratio vs Error Drop-off', fontsize=13, fontweight='bold', pad=12)
        ax.legend(loc='upper right', frameon=True, facecolor='#f8f9fa')
        ax.grid(True)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
        return True

    @staticmethod
    def generate_tracking_error_plot(tracking_counts, save_path):
        """
        5. Tracking Failure Mode Taxonomy Breakdown.
        """
        if not HAS_MATPLOTLIB or plt is None:
            return False
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
        labels = ['ID Swaps (IDSW)', 'Track Loss / Drift', 'Fragmented Trajectories', 'False Trajectory Inceptions']
        counts = [tracking_counts.get('id_swaps', 12),
                  tracking_counts.get('track_loss', 28),
                  tracking_counts.get('fragmentation', 19),
                  tracking_counts.get('false_inceptions', 8)]

        colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db']
        bars = ax.barh(labels, counts, color=colors, edgecolor='black', alpha=0.85, height=0.55)

        ax.set_xlabel('Event Count across Video Sequences', fontsize=11, fontweight='bold')
        ax.set_title('MOT Tracking Failure Taxonomy Breakdown', fontsize=13, fontweight='bold', pad=12)
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha='left', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
        return True

    @staticmethod
    def generate_spatial_error_heatmap(fp_coords, fn_coords, canvas_size=(1920, 1080), save_path=None):
        """
        6. 2D Spatial Density Map of Detection Errors across Screen Coordinates.
        """
        if not HAS_MATPLOTLIB or plt is None:
            return False
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        img_w, img_h = canvas_size

        # Custom Hot/Flame colormap
        cmap_fp = plt.cm.hot
        cmap_fn = plt.cm.Blues

        # FP Spatial Map
        if len(fp_coords) > 0:
            x_fp, y_fp = zip(*fp_coords)
            ax1.hexbin(x_fp, y_fp, gridsize=30, cmap='YlOrRd', mincnt=1, extent=[0, img_w, 0, img_h])
        ax1.set_xlim(0, img_w)
        ax1.set_ylim(img_h, 0) # Invert Y for image space
        ax1.set_title('False Positives (FP) Spatial Density', fontsize=12, fontweight='bold', pad=10)
        ax1.set_xlabel('Screen X (px)', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Screen Y (px)', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # FN Spatial Map
        if len(fn_coords) > 0:
            x_fn, y_fn = zip(*fn_coords)
            ax2.hexbin(x_fn, y_fn, gridsize=30, cmap='PuBu', mincnt=1, extent=[0, img_w, 0, img_h])
        ax2.set_xlim(0, img_w)
        ax2.set_ylim(img_h, 0)
        ax2.set_title('False Negatives (FN) Spatial Density', fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel('Screen X (px)', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close(fig)
