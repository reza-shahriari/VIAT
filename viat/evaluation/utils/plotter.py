#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting utilities for visualization of evaluation metrics (mAP by class, mAP by size).
Ported from OLDEVAL/src/utils/plotter.py
"""

import os

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_map_by_class(class_metrics_dict, output_path, theme='Dark', palette='Vibrant', dpi=150):
    """
    Plots mAP by Class as a bar chart.
    class_metrics_dict: dict like {'Soldier': {'AP50': 0.85, 'AP': 0.62}, 'Human': {'AP50': 0.78, 'AP': 0.55}}
    """
    if not HAS_MATPLOTLIB or not class_metrics_dict:
        return

    from viat.evaluation.utils.advanced_diagnostics import AestheticConfig

    classes = list(class_metrics_dict.keys())
    ap50_vals = [class_metrics_dict[c].get('AP50', 0.0) or 0.0 for c in classes]
    ap_vals = [class_metrics_dict[c].get('AP', 0.0) or 0.0 for c in classes]

    # Convert to percentages if stored as decimals
    if max(ap50_vals or [0]) <= 1.0:
        ap50_vals = [v * 100 for v in ap50_vals]
    if max(ap_vals or [0]) <= 1.0:
        ap_vals = [v * 100 for v in ap_vals]

    x = np.arange(len(classes))
    width = 0.35

    colors = AestheticConfig.PALETTES.get(palette, AestheticConfig.PALETTES['Vibrant'])
    theme_cfg = AestheticConfig.THEMES.get(theme, AestheticConfig.THEMES['Dark'])

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.5), 5.5), dpi=dpi)

    bars1 = ax.bar(x - width/2, ap50_vals, width, label='mAP@0.50', color=colors[2], alpha=0.85)
    bars2 = ax.bar(x + width/2, ap_vals, width, label='mAP@[0.40:0.95]', color=colors[0], alpha=0.85)

    ax.set_ylabel('mAP (%)', fontsize=12, fontweight='bold')
    ax.set_title('mAP Performance by Class', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105)

    leg = ax.legend(fontsize=11, frameon=True)
    leg.get_frame().set_facecolor(theme_cfg['legend_bg'])
    leg.get_frame().set_edgecolor(theme_cfg['legend_edge'])
    for text in leg.get_texts():
        text.set_color(theme_cfg['text'])

    for bar in bars1:
        yval = bar.get_height()
        if yval > 0:
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}%',
                    ha='center', va='bottom', fontsize=9, color=theme_cfg['text'], fontweight='bold')
    for bar in bars2:
        yval = bar.get_height()
        if yval > 0:
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}%',
                    ha='center', va='bottom', fontsize=9, color=theme_cfg['text'], fontweight='bold')

    AestheticConfig.apply(fig, ax, theme, show_grid=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


def plot_map_by_size(size_metrics_dict, output_path, theme='Dark', palette='Vibrant', dpi=150):
    """
    Plots mAP by Object Size (Small, Medium, Large) as a bar chart.
    size_metrics_dict: dict like {'Small (<32²)': 0.45, 'Medium (32²-96²)': 0.72, 'Large (>96²)': 0.88}
    """
    if not HAS_MATPLOTLIB or not size_metrics_dict:
        return

    from viat.evaluation.utils.advanced_diagnostics import AestheticConfig

    sizes = list(size_metrics_dict.keys())
    vals = [
        size_metrics_dict[s]
        if size_metrics_dict[s] is not None and not np.isnan(size_metrics_dict[s])
        else 0.0
        for s in sizes
    ]

    if max(vals or [0]) <= 1.0:
        vals = [v * 100 for v in vals]

    colors = AestheticConfig.PALETTES.get(palette, AestheticConfig.PALETTES['Vibrant'])
    theme_cfg = AestheticConfig.THEMES.get(theme, AestheticConfig.THEMES['Dark'])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

    bars = ax.bar(sizes, vals, color=colors[:len(sizes)], width=0.45, alpha=0.85)

    ax.set_ylabel('mAP (%)', fontsize=12, fontweight='bold')
    ax.set_title('mAP Performance by Object Size', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 105)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=theme_cfg['text'])

    AestheticConfig.apply(fig, ax, theme, show_grid=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path := output_path), exist_ok=True)
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
