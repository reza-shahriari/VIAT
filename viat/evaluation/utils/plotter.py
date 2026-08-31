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


def plot_map_by_class(class_metrics_dict, output_path):
    """
    Plots mAP by Class as a bar chart.
    class_metrics_dict: dict like {'Soldier': {'AP50': 0.85, 'AP': 0.62}, 'Human': {'AP50': 0.78, 'AP': 0.55}}
    """
    if not HAS_MATPLOTLIB or not class_metrics_dict:
        return

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

    plt.figure(figsize=(max(8, len(classes) * 1.5), 6), dpi=150)
    plt.style.use('dark_background')

    bars1 = plt.bar(x - width/2, ap50_vals, width, label='mAP@0.50', color='#4CAF50')
    bars2 = plt.bar(x + width/2, ap_vals, width, label='mAP@[0.4:0.95]', color='#2196F3')

    plt.ylabel('mAP (%)', fontsize=12, fontweight='bold')
    plt.title('mAP Performance by Class', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x, classes, fontsize=11, fontweight='bold')
    plt.ylim(0, 105)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    for bar in bars1:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}%',
                     ha='center', va='bottom', fontsize=9, color='#81C784')
    for bar in bars2:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}%',
                     ha='center', va='bottom', fontsize=9, color='#64B5F6')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Class mAP plot to {output_path}")


def plot_map_by_size(size_metrics_dict, output_path):
    """
    Plots mAP by Object Size (Small, Medium, Large) as a bar chart.
    size_metrics_dict: dict like {'Small (<32²)': 0.45, 'Medium (32²-96²)': 0.72, 'Large (>96²)': 0.88}
    """
    if not HAS_MATPLOTLIB or not size_metrics_dict:
        return

    sizes = list(size_metrics_dict.keys())
    vals = [
        size_metrics_dict[s]
        if size_metrics_dict[s] is not None and not np.isnan(size_metrics_dict[s])
        else 0.0
        for s in sizes
    ]

    # Convert to percentage if decimal
    if max(vals or [0]) <= 1.0:
        vals = [v * 100 for v in vals]

    plt.figure(figsize=(8, 5), dpi=150)
    plt.style.use('dark_background')

    colors = ['#FF9800', '#00BCD4', '#E91E63']
    bars = plt.bar(sizes, vals, color=colors[:len(sizes)], width=0.45)

    plt.ylabel('mAP (%)', fontsize=12, fontweight='bold')
    plt.title('mAP Performance by Object Size', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}%',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Size mAP plot to {output_path}")
