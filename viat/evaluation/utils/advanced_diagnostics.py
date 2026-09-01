"""
Advanced Model Diagnostics & Beautiful Plot Generator for VIAT Evaluation Engine

100% Computed from Real Evaluation Data:
1. Inter-Class Confusion Matrix Heatmap (GT vs Preds + Background FP/FN)
2. Confidence Calibration Curve & ECE (Expected Calibration Error)
3. Precision-Recall Curves per Class with AUC Area Shading (NEW)
4. F1-Score vs Confidence Threshold Sweep per Class (NEW)
5. Confidence Distribution Histogram (TP vs FP Separation) (NEW)
6. Error Taxonomy Breakdown Donut Chart (Classification vs Localization vs Background vs Missed) (NEW)
7. Localization Precision & IoU Histogram (Box Tightness Analysis)
8. Box Geometry Bias (Aspect Ratio W/H vs Error Rates)
9. 2D Spatial Error Heatmap (Screen Location Error Hotspots)
10. Multi-Video Performance Comparison Bar Chart (NEW)
11. MOT Tracking Error Taxonomy Breakdown (IDSW, Track Loss, Frag, FP)
"""

import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    matplotlib = None
    plt = None
    HAS_MATPLOTLIB = False


class AestheticConfig:
    """Themes and color palettes for publication-quality and dark-mode charts."""

    THEMES = {
        'Dark': {
            'bg': '#1e1e1e',
            'axes_bg': '#252526',
            'text': '#ffffff',
            'grid': '#444444',
            'edge': '#555555',
            'legend_bg': '#2d2d2d',
            'legend_edge': '#666666',
        },
        'Light': {
            'bg': '#ffffff',
            'axes_bg': '#f8f9fa',
            'text': '#212529',
            'grid': '#e0e0e0',
            'edge': '#cccccc',
            'legend_bg': '#f8f9fa',
            'legend_edge': '#cccccc',
        },
        'Publication': {
            'bg': '#ffffff',
            'axes_bg': '#ffffff',
            'text': '#000000',
            'grid': '#e8e8e8',
            'edge': '#000000',
            'legend_bg': '#ffffff',
            'legend_edge': '#000000',
        },
    }

    PALETTES = {
        'Vibrant': ['#00e5ff', '#ff334b', '#00ff78', '#ff9900', '#9b59b6', '#3498db', '#f1c40f', '#e67e22', '#1abc9c', '#e91e63'],
        'Viridis': ['#440154', '#482878', '#3e4a89', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'],
        'Cool Ocean': ['#0077b6', '#0096c7', '#00b4d8', '#48cae4', '#90e0ef', '#023e8a', '#03045e', '#5bc0be', '#6fffe9'],
        'Warm Sunset': ['#d9534f', '#f0ad4e', '#5cb85c', '#5bc0de', '#428bca', '#e74c3c', '#e67e22', '#f39c12', '#d35400', '#c0392b'],
        'Monochrome': ['#333333', '#555555', '#777777', '#999999', '#bbbbbb', '#222222', '#444444', '#666666'],
    }

    @classmethod
    def apply(cls, fig, ax, theme_name='Dark', show_grid=True):
        """Applies theme colors to figure and axes."""
        theme = cls.THEMES.get(theme_name, cls.THEMES['Dark'])
        fig.patch.set_facecolor(theme['bg'])
        ax.set_facecolor(theme['axes_bg'])
        ax.tick_params(colors=theme['text'], labelsize=9)
        ax.xaxis.label.set_color(theme['text'])
        ax.yaxis.label.set_color(theme['text'])
        ax.title.set_color(theme['text'])
        for spine in ax.spines.values():
            spine.set_color(theme['edge'])
            spine.set_linewidth(1.0)
        if show_grid:
            ax.grid(True, linestyle='--', color=theme['grid'], alpha=0.6)
        else:
            ax.grid(False)


class AdvancedDiagnosticsEngine:
    """Generates all deep model inspection plots with real-time aesthetic replotting."""

    @staticmethod
    def generate_confusion_matrix_plot(cm, class_names, save_path, theme='Dark', palette='Vibrant', dpi=150):
        """1. Inter-Class Confusion Matrix Heatmap (including Background FP & FN)."""
        if not HAS_MATPLOTLIB or plt is None:
            return False
        fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=dpi)
        labels = list(class_names) + ['Background']
        n_classes = len(labels)

        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)
        cmap = plt.cm.YlGnBu if theme in ('Light', 'Publication') else plt.cm.plasma

        im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Ratio / Recall', rotation=-90, va="bottom", fontweight='bold', fontsize=10)
        theme_cfg = AestheticConfig.THEMES.get(theme, AestheticConfig.THEMES['Dark'])
        cbar.ax.yaxis.label.set_color(theme_cfg['text'])
        cbar.ax.tick_params(colors=theme_cfg['text'])

        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10, fontweight='bold')
        ax.set_yticklabels(labels, fontsize=10, fontweight='bold')

        ax.set_ylabel('True Category (GT)', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_xlabel('Predicted Category (Model)', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_title('Inter-Class Confusion Matrix & Background Errors', fontsize=13, fontweight='bold', pad=14)

        for i in range(n_classes):
            for j in range(n_classes):
                val = int(cm[i, j])
                pct = cm_norm[i, j] * 100
                txt_color = "white" if cm_norm[i, j] > 0.55 else "black"
                if theme == 'Dark' and cm_norm[i, j] <= 0.55:
                    txt_color = "white"
                ax.text(j, i, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center", color=txt_color,
                        fontsize=9, fontweight='bold')

        AestheticConfig.apply(fig, ax, theme, show_grid=False)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return True

    @staticmethod
    def generate_calibration_plot(confidences, precisions, recalls, ece_score, optimal_thr, save_path,
                                  theme='Dark', palette='Vibrant', dpi=150, line_width=2.2, show_grid=True):
        """2. Confidence Calibration Curve & PR Sweep Curve (Reliability & ECE)."""
        if not HAS_MATPLOTLIB or plt is None:
            return False
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
        colors = AestheticConfig.PALETTES.get(palette, AestheticConfig.PALETTES['Vibrant'])
        theme_cfg = AestheticConfig.THEMES.get(theme, AestheticConfig.THEMES['Dark'])

        # Subplot 1: Calibration / Reliability Curve
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5, alpha=0.7)
        ax1.plot(confidences, precisions, 'o-', color=colors[0], linewidth=line_width, markersize=5,
                 label=f'Model (ECE: {ece_score:.3f})')
        ax1.fill_between(confidences, precisions, confidences, color=colors[0], alpha=0.15, label='Calibration Gap')

        ax1.set_xlabel('Predicted Confidence Score', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Observed Accuracy / Precision', fontsize=10, fontweight='bold')
        ax1.set_title('Confidence Calibration Curve (ECE)', fontsize=12, fontweight='bold', pad=10)
        leg1 = ax1.legend(loc='upper left', frameon=True)
        leg1.get_frame().set_facecolor(theme_cfg['legend_bg'])
        leg1.get_frame().set_edgecolor(theme_cfg['legend_edge'])
        for text in leg1.get_texts():
            text.set_color(theme_cfg['text'])
        AestheticConfig.apply(fig, ax1, theme, show_grid)

        # Subplot 2: Precision-Recall vs Threshold Sweep
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
        ax2.plot(confidences, precisions, '-', color=colors[2], linewidth=line_width, label='Precision')
        ax2.plot(confidences, recalls, '-', color=colors[1], linewidth=line_width, label='Recall')
        ax2.plot(confidences, f1_scores, '--', color=colors[4 % len(colors)], linewidth=line_width + 0.3,
                 label=f'F1-Score (Peak @ {optimal_thr:.2f})')

        ax2.axvline(optimal_thr, color=colors[4 % len(colors)], linestyle=':', linewidth=1.8)
        ax2.set_xlabel('Confidence Threshold', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Score Ratio', fontsize=10, fontweight='bold')
        ax2.set_title('Precision / Recall / F1 Sweep', fontsize=12, fontweight='bold', pad=10)
        leg2 = ax2.legend(loc='lower left', frameon=True)
        leg2.get_frame().set_facecolor(theme_cfg['legend_bg'])
        leg2.get_frame().set_edgecolor(theme_cfg['legend_edge'])
        for text in leg2.get_texts():
            text.set_color(theme_cfg['text'])
        AestheticConfig.apply(fig, ax2, theme, show_grid)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return True

    @staticmethod
    def generate_pr_curves_plot(per_class_curves, save_path, theme='Dark', palette='Vibrant', dpi=150,
                                line_width=2.2, show_grid=True):
        """3. Precision-Recall Curves per Class with AUC Area Shading (NEW)."""
        if not HAS_MATPLOTLIB or plt is None or not per_class_curves:
            return False
        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=dpi)
        colors = AestheticConfig.PALETTES.get(palette, AestheticConfig.PALETTES['Vibrant'])
        theme_cfg = AestheticConfig.THEMES.get(theme, AestheticConfig.THEMES['Dark'])

        for idx, (cls_name, curve_data) in enumerate(per_class_curves.items()):
            color = colors[idx % len(colors)]
            rec = np.array(curve_data.get('recalls', []))
            prec = np.array(curve_data.get('precisions', []))
            ap = curve_data.get('ap', 0.0)

            # Sort by recall for smooth curve
            sorted_idx = np.argsort(rec)
            r_sorted = rec[sorted_idx]
            p_sorted = prec[sorted_idx]

            # Ensure curve reaches (0, 1) and ends cleanly
            r_plot = np.concatenate(([0.0], r_sorted, [1.0]))
            p_plot = np.concatenate(([p_sorted[0] if len(p_sorted) else 1.0], p_sorted, [0.0]))

            ax.plot(r_plot, p_plot, color=color, linewidth=line_width, label=f"{cls_name} (AP: {ap*100:.1f}%)")
            ax.fill_between(r_plot, p_plot, alpha=0.12, color=color)

        ax.set_xlim([0.0, 1.02])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=11, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
        ax.set_title('Precision-Recall Curves & Area Under Curve (AUC)', fontsize=13, fontweight='bold', pad=12)

        leg = ax.legend(loc='lower left', frameon=True, fontsize=10)
        leg.get_frame().set_facecolor(theme_cfg['legend_bg'])
        leg.get_frame().set_edgecolor(theme_cfg['legend_edge'])
        for text in leg.get_texts():
            text.set_color(theme_cfg['text'])

        AestheticConfig.apply(fig, ax, theme, show_grid)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return True

    @staticmethod
    def generate_f1_confidence_plot(per_class_curves, save_path, theme='Dark', palette='Vibrant', dpi=150,
                                    line_width=2.2, show_grid=True):
        """4. F1-Score vs Confidence Threshold Curve per Class (NEW)."""
        if not HAS_MATPLOTLIB or plt is None or not per_class_curves:
            return False
        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=dpi)
        colors = AestheticConfig.PALETTES.get(palette, AestheticConfig.PALETTES['Vibrant'])
        theme_cfg = AestheticConfig.THEMES.get(theme, AestheticConfig.THEMES['Dark'])

        for idx, (cls_name, curve_data) in enumerate(per_class_curves.items()):
            color = colors[idx % len(colors)]
            confs = np.array(curve_data.get('confidences', []))
            f1s = np.array(curve_data.get('f1s', []))
            opt_thr = curve_data.get('optimal_thr', 0.5)
            peak_f1 = curve_data.get('peak_f1', 0.0)

            ax.plot(confs, f1s, color=color, linewidth=line_width,
                    label=f"{cls_name} (Peak F1: {peak_f1:.2f} @ {opt_thr:.2f})")
            ax.axvline(opt_thr, color=color, linestyle=':', alpha=0.6, linewidth=1.2)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Confidence Threshold', fontsize=11, fontweight='bold')
        ax.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
        ax.set_title('F1-Score vs Confidence Threshold (Optimal Operating Point)', fontsize=13, fontweight='bold', pad=12)

        leg = ax.legend(loc='lower left', frameon=True, fontsize=10)
        leg.get_frame().set_facecolor(theme_cfg['legend_bg'])
        leg.get_frame().set_edgecolor(theme_cfg['legend_edge'])
        for text in leg.get_texts():
            text.set_color(theme_cfg['text'])

        AestheticConfig.apply(fig, ax, theme, show_grid)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return True

    @staticmethod
    def generate_confidence_distribution_plot(conf_tp, conf_fp, save_path, theme='Dark', palette='Vibrant',
                                             dpi=150, show_grid=True):
        """5. Confidence Distribution Histogram: True Positives vs False Positives (NEW)."""
        if not HAS_MATPLOTLIB or plt is None:
            return False
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=dpi)
        colors = AestheticConfig.PALETTES.get(palette, AestheticConfig.PALETTES['Vibrant'])
        theme_cfg = AestheticConfig.THEMES.get(theme, AestheticConfig.THEMES['Dark'])

        bins = np.linspace(0.0, 1.0, 25)
        if len(conf_tp) > 0:
            ax.hist(conf_tp, bins=bins, color=colors[2], alpha=0.65, edgecolor='black',
                    label=f'True Positives (n={len(conf_tp)})')
        if len(conf_fp) > 0:
            ax.hist(conf_fp, bins=bins, color=colors[1], alpha=0.65, edgecolor='black',
                    label=f'False Positives (n={len(conf_fp)})')

        ax.set_xlabel('Prediction Confidence Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Detection Count', fontsize=11, fontweight='bold')
        ax.set_title('Confidence Score Distribution (TP vs FP Separation)', fontsize=13, fontweight='bold', pad=12)

        leg = ax.legend(loc='upper right', frameon=True, fontsize=10)
        leg.get_frame().set_facecolor(theme_cfg['legend_bg'])
        leg.get_frame().set_edgecolor(theme_cfg['legend_edge'])
        for text in leg.get_texts():
            text.set_color(theme_cfg['text'])

        AestheticConfig.apply(fig, ax, theme, show_grid)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return True

    @staticmethod
    def generate_error_breakdown_plot(error_breakdown, save_path, theme='Dark', palette='Vibrant', dpi=150):
        """6. Error Taxonomy Breakdown Donut Chart (NEW)."""
        if not HAS_MATPLOTLIB or plt is None or not error_breakdown:
            return False

        labels = ['Classification Error', 'Localization Error', 'Background FP', 'Missed Objects (FN)']
        counts = [
            error_breakdown.get('classification', 0),
            error_breakdown.get('localization', 0),
            error_breakdown.get('background_fp', 0),
            error_breakdown.get('missed_fn', 0),
        ]
        total_err = sum(counts)
        if total_err == 0:
            return False

        filtered = [(lbl, c) for lbl, c in zip(labels, counts) if c > 0]
        f_labels, f_counts = zip(*filtered)

        fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=dpi)
        colors = AestheticConfig.PALETTES.get(palette, AestheticConfig.PALETTES['Vibrant'])
        theme_cfg = AestheticConfig.THEMES.get(theme, AestheticConfig.THEMES['Dark'])

        wedges, texts, autotexts = ax.pie(
            f_counts,
            labels=f_labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors[:len(f_counts)],
            pctdistance=0.75,
            textprops={'fontsize': 10, 'fontweight': 'bold', 'color': theme_cfg['text']},
            wedgeprops={'edgecolor': theme_cfg['bg'], 'linewidth': 2, 'width': 0.5}
        )
        for autotext in autotexts:
            autotext.set_color('white' if theme == 'Dark' else 'black')
            autotext.set_fontsize(10)

        ax.set_title(f'Error Distribution Breakdown (Total Errors: {total_err})',
                     fontsize=13, fontweight='bold', pad=12, color=theme_cfg['text'])
        fig.patch.set_facecolor(theme_cfg['bg'])
        ax.set_facecolor(theme_cfg['bg'])

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return True

    @staticmethod
    def generate_iou_distribution_plot(ious, save_path, theme='Dark', palette='Vibrant', dpi=150, show_grid=True):
        """7. Localization Precision & IoU Distribution Histogram."""
        if not HAS_MATPLOTLIB or plt is None:
            return False
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=dpi)
        ious = np.array(ious)

        n, bins, patches = ax.hist(ious, bins=25, range=(0.4, 1.0), edgecolor='black', alpha=0.85)

        for bin_left, patch in zip(bins[:-1], patches):
            if bin_left >= 0.85:
                patch.set_facecolor('#2ecc71')
            elif bin_left >= 0.70:
                patch.set_facecolor('#3498db')
            elif bin_left >= 0.50:
                patch.set_facecolor('#f39c12')
            else:
                patch.set_facecolor('#e74c3c')

        mean_iou = np.mean(ious) if len(ious) > 0 else 0
        median_iou = np.median(ious) if len(ious) > 0 else 0
        theme_cfg = AestheticConfig.THEMES.get(theme, AestheticConfig.THEMES['Dark'])

        ax.axvline(mean_iou, color='#111111' if theme != 'Dark' else '#ffffff', linestyle='--', linewidth=2,
                   label=f'Mean IoU: {mean_iou:.3f}')
        ax.axvline(median_iou, color='#8e44ad', linestyle=':', linewidth=2, label=f'Median IoU: {median_iou:.3f}')

        ax.set_xlabel('Intersection over Union (IoU)', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive BBox Count', fontsize=11, fontweight='bold')
        ax.set_title('Localization Tightness & IoU Distribution', fontsize=13, fontweight='bold', pad=12)
        leg = ax.legend(loc='upper left', frameon=True)
        leg.get_frame().set_facecolor(theme_cfg['legend_bg'])
        leg.get_frame().set_edgecolor(theme_cfg['legend_edge'])
        for text in leg.get_texts():
            text.set_color(theme_cfg['text'])

        AestheticConfig.apply(fig, ax, theme, show_grid)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return True

    @staticmethod
    def generate_aspect_ratio_plot(aspect_ratios, error_rates, save_path, theme='Dark', palette='Vibrant', dpi=150,
                                   show_grid=True):
        """8. Box Geometry Bias (Aspect Ratio W/H vs Error Rates)."""
        if not HAS_MATPLOTLIB or plt is None:
            return False
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=dpi)
        colors = AestheticConfig.PALETTES.get(palette, AestheticConfig.PALETTES['Vibrant'])
        theme_cfg = AestheticConfig.THEMES.get(theme, AestheticConfig.THEMES['Dark'])

        ax.scatter(aspect_ratios, error_rates, color=colors[1], s=50, alpha=0.8, edgecolors='black', label='Sample Batches')

        if len(aspect_ratios) >= 4:
            sorted_idx = np.argsort(aspect_ratios)
            x_sorted = np.array(aspect_ratios)[sorted_idx]
            y_sorted = np.array(error_rates)[sorted_idx]
            z = np.polyfit(x_sorted, y_sorted, deg=2)
            p = np.poly1d(z)
            trend_color = colors[0] if theme == 'Dark' else '#2c3e50'
            ax.plot(x_sorted, p(x_sorted), color=trend_color, linewidth=2.5, linestyle='-', label='Geometry Bias Trend')

        ax.axvline(1.0, color='#7f8c8d', linestyle='--', alpha=0.7, label='Square Box (1:1)')

        ax.set_xlabel('Bounding Box Aspect Ratio (Width / Height)', fontsize=11, fontweight='bold')
        ax.set_ylabel('False Negative / Error Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title('Bounding Box Aspect Ratio vs Error Drop-off', fontsize=13, fontweight='bold', pad=12)
        leg = ax.legend(loc='upper right', frameon=True)
        leg.get_frame().set_facecolor(theme_cfg['legend_bg'])
        leg.get_frame().set_edgecolor(theme_cfg['legend_edge'])
        for text in leg.get_texts():
            text.set_color(theme_cfg['text'])

        AestheticConfig.apply(fig, ax, theme, show_grid)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return True

    @staticmethod
    def generate_per_video_comparison_plot(video_metrics, save_path, theme='Dark', palette='Vibrant', dpi=150):
        """9. Multi-Video Performance Comparison Bar Chart (mAP@0.50 & F1-Score)."""
        if not HAS_MATPLOTLIB or plt is None or not video_metrics:
            return False

        v_names = []
        ap50s = []
        f1s = []

        for i, v in enumerate(video_metrics):
            if isinstance(v, dict):
                raw_name = v.get('name') or v.get('video') or v.get('Video_name') or f"Video_{i+1}"
                mets = v.get('metrics', v)
                if not isinstance(mets, dict):
                    mets = v

                # Helper to robustly extract float metric
                def _get_val(keys, default=0.0):
                    for k in keys:
                        for mk, mv in mets.items():
                            if mk.lower().replace('@', '').replace('_', '').replace(' ', '') == k.lower().replace('@', '').replace('_', '').replace(' ', ''):
                                try:
                                    val = float(mv) if mv not in (None, '', '-') else default
                                    return val
                                except (ValueError, TypeError):
                                    pass
                    return default

                ap50_val = _get_val(['ap50', 'map50', 'map050', 'ap'])
                f1_val = _get_val(['f1', 'f1score', 'fscore'])
            else:
                raw_name = f"Video_{i+1}"
                ap50_val = 0.0
                f1_val = 0.0

            clean_name = str(raw_name).strip()
            # If name is long, truncate cleanly (e.g. 5-7 chars with ellipsis)
            if len(clean_name) > 7:
                short_name = clean_name[:6] + "…"
            else:
                short_name = clean_name

            v_names.append(short_name)
            ap50s.append(ap50_val * 100 if ap50_val <= 1.0 else ap50_val)
            f1s.append(f1_val * 100 if f1_val <= 1.0 else f1_val)

        if not v_names:
            return False

        fig, ax = plt.subplots(figsize=(max(8, len(v_names) * 0.9), 5), dpi=dpi)
        colors = AestheticConfig.PALETTES.get(palette, AestheticConfig.PALETTES['Vibrant'])
        theme_cfg = AestheticConfig.THEMES.get(theme, AestheticConfig.THEMES['Dark'])

        x = np.arange(len(v_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, ap50s, width, label='mAP@0.50', color=colors[2 % len(colors)], alpha=0.85)
        bars2 = ax.bar(x + width/2, f1s, width, label='F1-Score', color=colors[0 % len(colors)], alpha=0.85)

        ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
        ax.set_title('Cross-Video Performance Comparison (mAP@0.50 & F1-Score)', fontsize=13, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(v_names, rotation=30, ha='right', fontsize=9, fontweight='bold')
        ax.set_ylim([0, 110])

        for bar in bars1:
            yval = bar.get_height()
            if yval > 0:
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{yval:.1f}%',
                        ha='center', va='bottom', fontsize=8, color=theme_cfg['text'], fontweight='bold')
        for bar in bars2:
            yval = bar.get_height()
            if yval > 0:
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{yval:.1f}%',
                        ha='center', va='bottom', fontsize=8, color=theme_cfg['text'], fontweight='bold')

        leg = ax.legend(loc='upper right', frameon=True)
        leg.get_frame().set_facecolor(theme_cfg['legend_bg'])
        leg.get_frame().set_edgecolor(theme_cfg['legend_edge'])
        for text in leg.get_texts():
            text.set_color(theme_cfg['text'])

        AestheticConfig.apply(fig, ax, theme, show_grid=True)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return True

    @staticmethod
    def generate_spatial_error_heatmap(fp_coords, fn_coords, canvas_size=(1920, 1080), save_path=None,
                                       theme='Dark', palette='Vibrant', dpi=150):
        """10. 2D Spatial Density Map of Detection Errors across Screen Coordinates."""
        if not HAS_MATPLOTLIB or plt is None:
            return False
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
        img_w, img_h = canvas_size

        if len(fp_coords) > 0:
            x_fp, y_fp = zip(*fp_coords)
            ax1.hexbin(x_fp, y_fp, gridsize=30, cmap='YlOrRd', mincnt=1, extent=[0, img_w, 0, img_h])
        ax1.set_xlim(0, img_w)
        ax1.set_ylim(img_h, 0)
        ax1.set_title('False Positives (FP) Spatial Density', fontsize=12, fontweight='bold', pad=10)
        ax1.set_xlabel('Screen X (px)', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Screen Y (px)', fontsize=10, fontweight='bold')
        AestheticConfig.apply(fig, ax1, theme, show_grid=True)

        if len(fn_coords) > 0:
            x_fn, y_fn = zip(*fn_coords)
            ax2.hexbin(x_fn, y_fn, gridsize=30, cmap='PuBu', mincnt=1, extent=[0, img_w, 0, img_h])
        ax2.set_xlim(0, img_w)
        ax2.set_ylim(img_h, 0)
        ax2.set_title('False Negatives (FN) Spatial Density', fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel('Screen X (px)', fontsize=10, fontweight='bold')
        AestheticConfig.apply(fig, ax2, theme, show_grid=True)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return True

    @staticmethod
    def generate_tracking_error_plot(tracking_counts, save_path, theme='Dark', palette='Vibrant', dpi=150):
        """11. MOT Tracking Failure Taxonomy Breakdown (Strictly Real Counts)."""
        if not HAS_MATPLOTLIB or plt is None or not tracking_counts:
            return False

        id_swaps = tracking_counts.get('id_swaps', tracking_counts.get('IDSW', 0))
        track_loss = tracking_counts.get('track_loss', tracking_counts.get('ML', 0))
        fragmentation = tracking_counts.get('fragmentation', tracking_counts.get('Frag', 0))
        false_inceptions = tracking_counts.get('false_inceptions', tracking_counts.get('CLR_FP', 0))

        counts = [id_swaps, track_loss, fragmentation, false_inceptions]
        if sum(counts) == 0:
            return False

        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=dpi)
        labels = ['ID Swaps (IDSW)', 'Track Loss / Drift', 'Fragmented Trajectories', 'False Trajectory Inceptions']
        colors = AestheticConfig.PALETTES.get(palette, AestheticConfig.PALETTES['Vibrant'])

        bars = ax.barh(labels, counts, color=colors[:4], edgecolor='black', alpha=0.85, height=0.55)
        ax.set_xlabel('Event Count across Video Sequences', fontsize=11, fontweight='bold')
        ax.set_title('MOT Tracking Failure Taxonomy Breakdown', fontsize=13, fontweight='bold', pad=12)

        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{int(width)}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha='left', va='center', fontsize=10, fontweight='bold')

        AestheticConfig.apply(fig, ax, theme, show_grid=True)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return True
