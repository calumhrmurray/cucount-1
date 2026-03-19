#!/usr/bin/env python3
"""
Remake displacement-shape `x,y` plots using the notebook-style 2D contour look.

This mirrors the visual style used by
`intrinsic_alignment_analysis.modules.pipeline.results_plotting.plot_2d_correlations`,
but applies it to the displacement-aligned `(transverse, phi)` maps saved by
`displacement_shape_phi_map.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from displacement_shape_phi_map import get_xy_axis_labels, make_shear_segments


def load_displacement_shape_xy_results(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], str]:
    with np.load(path) as data:
        if 'rp_edges' in data:
            transverse_edges = np.asarray(data['rp_edges'], dtype=np.float64)
        elif 'theta_edges' in data:
            transverse_edges = np.asarray(data['theta_edges'], dtype=np.float64)
        else:
            raise KeyError('Expected either rp_edges or theta_edges in the input NPZ')

        transverse_bin = str(data['transverse_bin']) if 'transverse_bin' in data else 'rp'
        phi_edges = np.asarray(data['phi_edges'], dtype=np.float64)
        components = {
            'gamma_plus': np.asarray(data['gamma_plus'], dtype=np.float64),
            'gamma_cross': np.asarray(data['gamma_cross'], dtype=np.float64),
            'gamma_plus_smoothed': np.asarray(data['gamma_plus_smoothed'], dtype=np.float64),
            'gamma_cross_smoothed': np.asarray(data['gamma_cross_smoothed'], dtype=np.float64),
        }
    return transverse_edges, phi_edges, components, transverse_bin


def wrapped_xy_centers(
    transverse_edges: np.ndarray,
    phi_edges: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transverse_centers = 0.5 * (transverse_edges[:-1] + transverse_edges[1:])
    phi_centers_deg = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    phi_centers_deg = np.concatenate([phi_centers_deg, [phi_centers_deg[0] + 360.0]])
    wrapped_values = np.concatenate([values, values[:, :1]], axis=1)

    transverse_grid, phi_grid = np.meshgrid(transverse_centers, np.deg2rad(phi_centers_deg), indexing='ij')
    x = transverse_grid * np.cos(phi_grid)
    y = transverse_grid * np.sin(phi_grid)
    return x, y, wrapped_values


def contour_limits(values: np.ndarray, percentile: float = 95.0) -> tuple[float, np.ndarray]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.percentile(np.abs(finite), percentile))
        if (not np.isfinite(vmax)) or (vmax == 0.0):
            vmax = 1.0
    levels = np.linspace(-vmax, vmax, 10)
    return vmax, levels


def plot_single_xy_map(
    transverse_edges: np.ndarray,
    phi_edges: np.ndarray,
    values: np.ndarray,
    title: str,
    output_path: Path,
    transverse_bin: str,
) -> None:
    x, y, wrapped_values = wrapped_xy_centers(transverse_edges, phi_edges, values)
    xlabel, ylabel = get_xy_axis_labels(transverse_bin)
    vmax, levels = contour_limits(wrapped_values)

    fig, ax = plt.subplots(figsize=(6.6, 5.8), constrained_layout=True)
    image = ax.contourf(
        x,
        y,
        wrapped_values,
        levels=levels,
        cmap='RdBu_r',
        vmin=-vmax,
        vmax=vmax,
        extend='both',
    )
    ax.contour(
        x,
        y,
        wrapped_values,
        levels=levels,
        colors='black',
        linewidths=0.5,
        alpha=0.3,
    )
    ax.axhline(0.0, color='0.7', linewidth=1.0)
    ax.axvline(0.0, color='0.7', linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    fig.colorbar(image, ax=ax)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_smoothed_overlay(
    transverse_edges: np.ndarray,
    phi_edges: np.ndarray,
    gamma_plus_smoothed: np.ndarray,
    gamma_cross_smoothed: np.ndarray,
    output_path: Path,
    transverse_bin: str,
) -> None:
    xlabel, ylabel = get_xy_axis_labels(transverse_bin)
    segments = make_shear_segments(
        transverse_edges,
        phi_edges,
        gamma_plus_smoothed,
        gamma_cross_smoothed,
        normalize_lengths=True,
    )

    figures = [
        (gamma_plus_smoothed, r'Smoothed $\gamma_+(x, y)$'),
        (gamma_cross_smoothed, r'Smoothed $\gamma_\times(x, y)$'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=True)
    for axis, (values, title) in zip(axes, figures, strict=False):
        x, y, wrapped_values = wrapped_xy_centers(transverse_edges, phi_edges, values)
        vmax, levels = contour_limits(wrapped_values)
        image = axis.contourf(
            x,
            y,
            wrapped_values,
            levels=levels,
            cmap='RdBu_r',
            vmin=-vmax,
            vmax=vmax,
            extend='both',
        )
        axis.contour(
            x,
            y,
            wrapped_values,
            levels=levels,
            colors='black',
            linewidths=0.5,
            alpha=0.3,
        )
        if segments:
            axis.add_collection(LineCollection(segments, colors='white', linewidths=2.0, alpha=0.7, zorder=3))
            axis.add_collection(LineCollection(segments, colors='black', linewidths=1.0, alpha=0.9, zorder=4))
        axis.axhline(0.0, color='0.7', linewidth=1.0)
        axis.axvline(0.0, color='0.7', linewidth=1.0)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_aspect('equal')
        axis.grid(True, alpha=0.3, color='white', linewidth=0.5)
        fig.colorbar(image, ax=axis)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Remake displacement-shape x/y plots with notebook-style contours.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', required=True, help='Path to a displacement-shape phi-map results NPZ file.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory for the remade plots. Defaults to the input file directory.',
    )
    parser.add_argument(
        '--output-prefix',
        default=None,
        help='Prefix for output files. Defaults to the input file stem without _results.',
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')

    output_dir = args.output_dir.resolve() if args.output_dir is not None else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix or input_path.stem.removesuffix('_results')

    transverse_edges, phi_edges, components, transverse_bin = load_displacement_shape_xy_results(input_path)

    plot_single_xy_map(
        transverse_edges,
        phi_edges,
        components['gamma_plus'],
        r'$\gamma_+(x, y)$',
        output_dir / f'{output_prefix}_gamma_plus_xy_notebook_style.png',
        transverse_bin,
    )
    plot_single_xy_map(
        transverse_edges,
        phi_edges,
        components['gamma_cross'],
        r'$\gamma_\times(x, y)$',
        output_dir / f'{output_prefix}_gamma_cross_xy_notebook_style.png',
        transverse_bin,
    )
    plot_single_xy_map(
        transverse_edges,
        phi_edges,
        components['gamma_plus_smoothed'],
        r'$\gamma_+(x, y)$ Gaussian-smoothed',
        output_dir / f'{output_prefix}_gamma_plus_xy_smoothed_notebook_style.png',
        transverse_bin,
    )
    plot_single_xy_map(
        transverse_edges,
        phi_edges,
        components['gamma_cross_smoothed'],
        r'$\gamma_\times(x, y)$ Gaussian-smoothed',
        output_dir / f'{output_prefix}_gamma_cross_xy_smoothed_notebook_style.png',
        transverse_bin,
    )
    plot_smoothed_overlay(
        transverse_edges,
        phi_edges,
        components['gamma_plus_smoothed'],
        components['gamma_cross_smoothed'],
        output_dir / f'{output_prefix}_smoothed_shear_field_overlay_notebook_style.png',
        transverse_bin,
    )

    print(f'Remade notebook-style x/y plots in {output_dir}')


if __name__ == '__main__':
    main()
