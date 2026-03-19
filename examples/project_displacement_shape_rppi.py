#!/usr/bin/env python3
"""
Project displacement-shape `(s, mu)` outputs onto `(r_p, pi)` quick-look maps.

This script reuses the `compute_rp_pi()` helper from an external
`results_plotting.py` module when provided, matching the workflow used in the
intrinsic-alignment notebooks on cc-lyon.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


DEFAULT_RESULTS_PLOTTING = Path(
    '/sps/euclid/Users/cmurray/cc_lyon_ia/intrinsic_alignment_analysis/modules/pipeline/results_plotting.py'
)


def load_results_plotting_module(path: Path):
    spec = importlib.util.spec_from_file_location('results_plotting_external', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not import results_plotting module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_displacement_shape_results(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(path) as data:
        s_edges = np.asarray(data['s_edges'], dtype=np.float64)
        mu_edges = np.asarray(data['mu_edges'], dtype=np.float64)
        components = {
            'gamma_plus_d_parallel': np.asarray(data['gamma_plus_d_parallel'], dtype=np.float64),
            'gamma_plus_d_perp': np.asarray(data['gamma_plus_d_perp'], dtype=np.float64),
            'gamma_cross_d_parallel': np.asarray(data['gamma_cross_d_parallel'], dtype=np.float64),
            'gamma_cross_d_perp': np.asarray(data['gamma_cross_d_perp'], dtype=np.float64),
        }
    return s_edges, mu_edges, components


def plot_single_rppi_map(
    rp_edges: np.ndarray,
    pi_edges: np.ndarray,
    values: np.ndarray,
    title: str,
    output_path: Path,
    n_levels: int = 10,
) -> None:
    rp_centers = 0.5 * (rp_edges[:-1] + rp_edges[1:])
    pi_centers = 0.5 * (pi_edges[:-1] + pi_edges[1:])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.percentile(np.abs(finite), 95))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
    vmin = -vmax
    levels = np.linspace(vmin, vmax, n_levels)

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    image = ax.contourf(
        rp_centers,
        pi_centers,
        values.T,
        levels=levels,
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        extend='both',
    )
    ax.contour(
        rp_centers,
        pi_centers,
        values.T,
        levels=levels,
        colors='black',
        linewidths=0.5,
        alpha=0.3,
    )
    ax.set_xlabel(r'$r_p$ [Mpc/h]')
    ax.set_ylabel(r'$\pi$ [Mpc/h]')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    fig.colorbar(image, ax=ax, label='correlation [arcsec]')
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary_rppi(
    rp_edges: np.ndarray,
    pi_edges: np.ndarray,
    components: dict[str, np.ndarray],
    output_path: Path,
    n_levels: int = 10,
) -> None:
    titles = {
        'gamma_plus_d_parallel': r'$\langle \gamma_+ d_{\parallel} \rangle(r_p,\pi)$',
        'gamma_plus_d_perp': r'$\langle \gamma_+ d_{\perp} \rangle(r_p,\pi)$',
        'gamma_cross_d_parallel': r'$\langle \gamma_\times d_{\parallel} \rangle(r_p,\pi)$',
        'gamma_cross_d_perp': r'$\langle \gamma_\times d_{\perp} \rangle(r_p,\pi)$',
    }
    rp_centers = 0.5 * (rp_edges[:-1] + rp_edges[1:])
    pi_centers = 0.5 * (pi_edges[:-1] + pi_edges[1:])

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0), constrained_layout=True, sharex=True, sharey=True)
    for axis, key in zip(axes.ravel(), titles, strict=False):
        values = components[key]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            vmax = 1.0
        else:
            vmax = float(np.percentile(np.abs(finite), 95))
            if not np.isfinite(vmax) or vmax == 0.0:
                vmax = 1.0
        vmin = -vmax
        levels = np.linspace(vmin, vmax, n_levels)

        image = axis.contourf(
            rp_centers,
            pi_centers,
            values.T,
            levels=levels,
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
            extend='both',
        )
        axis.contour(
            rp_centers,
            pi_centers,
            values.T,
            levels=levels,
            colors='black',
            linewidths=0.5,
            alpha=0.3,
        )
        axis.set_xlabel(r'$r_p$ [Mpc/h]')
        axis.set_ylabel(r'$\pi$ [Mpc/h]')
        axis.set_title(titles[key])
        axis.set_aspect('equal')
        axis.grid(True, alpha=0.3, color='white', linewidth=0.5)
        fig.colorbar(image, ax=axis, label='correlation [arcsec]')

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Project displacement-shape (s, mu) outputs onto (rp, pi) maps.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to the displacement-shape results NPZ file.',
    )
    parser.add_argument(
        '--results-plotting',
        type=Path,
        default=DEFAULT_RESULTS_PLOTTING,
        help='Path to the external results_plotting.py module providing compute_rp_pi().',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory for the projected outputs. Defaults to the input file directory.',
    )
    parser.add_argument(
        '--output-prefix',
        default=None,
        help='Prefix for output files. Defaults to the input file stem without _results.',
    )
    parser.add_argument(
        '--smooth-sigma',
        type=float,
        default=2.0,
        help='Gaussian smoothing sigma applied after the (rp, pi) interpolation.',
    )
    parser.add_argument(
        '--n-levels',
        type=int,
        default=10,
        help='Number of contour levels in the quick-look plots.',
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')

    output_dir = args.output_dir.resolve() if args.output_dir is not None else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = input_path.stem.removesuffix('_results')

    results_plotting = load_results_plotting_module(args.results_plotting)
    s_edges, mu_edges, components = load_displacement_shape_results(input_path)

    projected_components: dict[str, np.ndarray] = {}
    rp_edges = None
    pi_edges = None
    for key, values in components.items():
        projected, rp_edges, pi_edges = results_plotting.compute_rp_pi(values, s_edges, mu_edges)
        if args.smooth_sigma > 0.0:
            projected = gaussian_filter(projected, sigma=args.smooth_sigma)
        projected_components[key] = projected

    if rp_edges is None or pi_edges is None:
        raise RuntimeError('Projection failed to return rp/pi edges')

    projected_path = output_dir / f'{output_prefix}_rp_pi_projected.npz'
    np.savez(
        projected_path,
        rp_edges=rp_edges,
        pi_edges=pi_edges,
        smooth_sigma=args.smooth_sigma,
        **projected_components,
    )
    print(f'Saved projected results to {projected_path}')

    summary_path = output_dir / f'{output_prefix}_rp_pi_summary.png'
    plot_summary_rppi(rp_edges, pi_edges, projected_components, summary_path, n_levels=args.n_levels)
    print(f'Saved projected summary figure to {summary_path}')

    titles = {
        'gamma_plus_d_parallel': r'$\langle \gamma_+ d_{\parallel} \rangle(r_p,\pi)$',
        'gamma_plus_d_perp': r'$\langle \gamma_+ d_{\perp} \rangle(r_p,\pi)$',
        'gamma_cross_d_parallel': r'$\langle \gamma_\times d_{\parallel} \rangle(r_p,\pi)$',
        'gamma_cross_d_perp': r'$\langle \gamma_\times d_{\perp} \rangle(r_p,\pi)$',
    }
    for key, values in projected_components.items():
        output_path = output_dir / f'{output_prefix}_{key}_rp_pi.png'
        plot_single_rppi_map(rp_edges, pi_edges, values, titles[key], output_path, n_levels=args.n_levels)
        print(f'Saved {key} projected map to {output_path}')


if __name__ == '__main__':
    main()
