#!/usr/bin/env python3
"""
Measure density around the local displacement direction on the sky.

The displacement reference direction is taken from the same first catalog spin-1
field used for phi binning:
    spin_values = (dDec, dRA * cos(dec))

where the stored components are ordered as (north, east) in the local tangent
plane, so positive and negative displacements remain distinct.

Outputs:
- a polar `(theta, phi)` correlation saved to NPZ
- a Cartesian `(x, y)` projection with the positive displacement direction on
  the +x axis (to the right)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cucount.numpy import BinAttrs, WeightAttrs, count2, setup_logging

from observed_catalog_tools import (
    DEFAULT_DESI_RANDOMS_GLOB,
    DEFAULT_DISPLACEMENT_DATA,
    create_particles,
    load_displacement_catalog,
    load_desi_catalog,
    resolve_random_catalogs,
    safe_divide,
)


def normalized_density_correlation(
    spin_particles,
    density_particles,
    random_particles_list,
    battrs: BinAttrs,
    nthreads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wattrs = WeightAttrs(spin=(1, 0))
    dd = count2(spin_particles, density_particles, battrs=battrs, wattrs=wattrs, nthreads=nthreads)

    dd_weight = np.asarray(dd['weight'], dtype=np.float64)
    dd_plus = np.asarray(dd['weight_plus'], dtype=np.float64)
    dd_cross = np.asarray(dd['weight_cross'], dtype=np.float64)

    dr_weight_sum = np.zeros_like(dd_weight)
    dr_plus_sum = np.zeros_like(dd_plus)
    dr_cross_sum = np.zeros_like(dd_cross)
    for random_particles in random_particles_list:
        dr = count2(spin_particles, random_particles, battrs=battrs, wattrs=wattrs, nthreads=nthreads)
        dr_weight_sum += np.asarray(dr['weight'], dtype=np.float64)
        dr_plus_sum += np.asarray(dr['weight_plus'], dtype=np.float64)
        dr_cross_sum += np.asarray(dr['weight_cross'], dtype=np.float64)

    dr_weight = dr_weight_sum / len(random_particles_list)
    dr_plus = dr_plus_sum / len(random_particles_list)
    dr_cross = dr_cross_sum / len(random_particles_list)

    norm_dd = safe_divide(dd_weight, np.sum(density_particles.get('individual_weight')[0]))
    norm_dr = safe_divide(dr_weight, np.mean([np.sum(p.get('individual_weight')[0]) for p in random_particles_list]))
    xi_density = safe_divide(norm_dd, norm_dr) - 1.0

    mean_plus = safe_divide(dd_plus, dd_weight)
    mean_cross = safe_divide(dd_cross, dd_weight)
    return xi_density, mean_plus, mean_cross


def plot_xy_map(
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    xi_density: np.ndarray,
    mean_plus: np.ndarray,
    mean_cross: np.ndarray,
    output_path: Path,
) -> None:
    # phi is defined relative to the positive displacement direction, so +x is
    # "along +displacement" and points to the right in the rendered image.
    theta_grid, phi_grid = np.meshgrid(theta_edges, np.deg2rad(phi_edges), indexing='ij')
    x = theta_grid * np.cos(phi_grid)
    y = theta_grid * np.sin(phi_grid)

    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_centers = np.deg2rad(0.5 * (phi_edges[:-1] + phi_edges[1:]))
    rr, pp = np.meshgrid(theta_centers, phi_centers, indexing='ij')
    xc = rr * np.cos(pp)
    yc = rr * np.sin(pp)
    ux = -(mean_plus * np.cos(pp) - mean_cross * np.sin(pp))
    uy = -(mean_plus * np.sin(pp) + mean_cross * np.cos(pp))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    pcm = axes[0].pcolormesh(x, y, xi_density, shading='auto', cmap='coolwarm')
    axes[0].axhline(0.0, color='0.7', linewidth=1.0)
    axes[0].axvline(0.0, color='0.7', linewidth=1.0)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_title('Density Correlation in Velocity-Aligned Frame')
    axes[0].set_xlabel(r'$x_{+\mathrm{vel}} = \theta \cos \phi$ [deg]')
    axes[0].set_ylabel(r'$y_{\perp} = \theta \sin \phi$ [deg]')
    fig.colorbar(pcm, ax=axes[0], label=r'$\xi(x, y)$')

    axes[1].pcolormesh(x, y, xi_density, shading='auto', cmap='Greys', alpha=0.35)
    axes[1].quiver(xc, yc, ux, uy, color='C0', angles='xy', scale_units='xy', scale=None, width=0.003)
    axes[1].axhline(0.0, color='0.7', linewidth=1.0)
    axes[1].axvline(0.0, color='0.7', linewidth=1.0)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_title('Mean Projected Velocity')
    axes[1].set_xlabel(r'$x_{+\mathrm{vel}} = \theta \cos \phi$ [deg]')
    axes[1].set_ylabel(r'$y_{\perp} = \theta \sin \phi$ [deg]')

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Measure density around the local displacement direction and project the result to x/y.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data',
        default=DEFAULT_DISPLACEMENT_DATA,
        required=DEFAULT_DISPLACEMENT_DATA is None,
        help='Path to the displacement catalog. Defaults to $CUCOUNT_DISPLACEMENT_DATA when set.',
    )
    parser.add_argument(
        '--randoms-glob',
        default=DEFAULT_DESI_RANDOMS_GLOB,
        required=DEFAULT_DESI_RANDOMS_GLOB is None,
        help='Glob pattern for DESI random catalogs. Defaults to $CUCOUNT_DESI_RANDOMS_GLOB when set.',
    )
    parser.add_argument('--output-dir', default='examples/output/displacement_density_phi', help='Directory for outputs.')
    parser.add_argument('--output-prefix', default='desi_lrg_displacement_density', help='Output file prefix.')
    parser.add_argument('--min-theta', type=float, default=0.05, help='Minimum theta in degrees.')
    parser.add_argument('--max-theta', type=float, default=2.0, help='Maximum theta in degrees.')
    parser.add_argument('--theta-bins', type=int, default=32, help='Number of theta bins.')
    parser.add_argument('--phi-bins', type=int, default=72, help='Number of phi bins over [0, 360) degrees.')
    parser.add_argument('--max-data-rows', type=int, default=None, help='Optional cap on displacement-catalog rows.')
    parser.add_argument('--max-random-files', type=int, default=4, help='Optional cap on random files.')
    parser.add_argument('--max-random-rows', type=int, default=None, help='Optional cap on rows per random file.')
    parser.add_argument('--seed', type=int, default=1234, help='Seed for reproducible sub-sampling.')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of GPUs for cucount to use.')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level)

    theta_edges = np.linspace(args.min_theta, args.max_theta, args.theta_bins + 1)
    phi_edges = np.linspace(0.0, 360.0, args.phi_bins + 1)
    battrs = BinAttrs(theta=theta_edges, phi=phi_edges)

    print('=' * 72)
    print('Displacement-density correlation in displacement-aligned coordinates')
    print('=' * 72)
    print(f'Data catalog     : {args.data}')
    print(f'Random catalogs  : {args.randoms_glob}')
    print(f'Theta range      : [{args.min_theta:.3f}, {args.max_theta:.3f}] deg')
    print(f'Theta bins       : {args.theta_bins}')
    print(f'Phi bins         : {args.phi_bins}')
    print('=' * 72)

    data, dalpha_cosdec_arcsec, ddec_arcsec = load_displacement_catalog(
        args.data,
        max_rows=args.max_data_rows,
        seed=args.seed,
    )
    print(f'Loaded {data.size:,} displacement tracers')

    # For spin-1 vectors the kernel expects (north, east) components with no
    # extra shear-convention sign flip.
    spin_particles = create_particles(
        data.ra,
        data.dec,
        data.weights,
        spin_values=(ddec_arcsec, dalpha_cosdec_arcsec),
    )
    density_particles = create_particles(data.ra, data.dec, data.weights)

    random_paths = resolve_random_catalogs(args.randoms_glob, max_files=args.max_random_files)
    random_particles_list = []
    for index, random_path in enumerate(random_paths, start=1):
        randoms = load_desi_catalog(random_path, max_rows=args.max_random_rows, seed=args.seed + index)
        random_particles_list.append(create_particles(randoms.ra, randoms.dec, randoms.weights))
    print(f'Loaded {len(random_particles_list)} random catalogs')

    xi_density, mean_plus, mean_cross = normalized_density_correlation(
        spin_particles,
        density_particles,
        random_particles_list,
        battrs=battrs,
        nthreads=args.nthreads,
    )

    results_path = output_dir / f'{args.output_prefix}_results.npz'
    np.savez(
        results_path,
        theta_edges=theta_edges,
        phi_edges=phi_edges,
        xi_density=xi_density,
        mean_plus=mean_plus,
        mean_cross=mean_cross,
    )
    print(f'Saved results to {results_path}')

    figure_path = output_dir / f'{args.output_prefix}_xy.png'
    plot_xy_map(theta_edges, phi_edges, xi_density, mean_plus, mean_cross, figure_path)
    print(f'Saved figure to {figure_path}')


if __name__ == '__main__':
    main()
