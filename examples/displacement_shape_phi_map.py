#!/usr/bin/env python3
"""
Measure shape around the local displacement direction on the sky.

Catalog 1 provides the displacement reference direction through a spin-1 unit
vector:
    spin_values = (dDec, dRA * cos(dec)) / |d|

Catalog 2 provides the spin-2 shape field:
    spin_values = -(e1, e2)

The output maps are binned in (theta, phi), where phi is measured relative to
the positive displacement direction. The rendered x axis therefore points along
+displacement, to the right.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cucount.numpy import BinAttrs, WeightAttrs, count2, setup_logging

from observed_catalog_tools import (
    DEFAULT_DESI_LRG_SHAPES,
    DEFAULT_DESI_RANDOMS_GLOB,
    DEFAULT_DISPLACEMENT_DATA,
    CatalogSample,
    create_particles,
    load_desi_catalog,
    load_displacement_catalog,
    load_unions_catalog,
    resolve_random_catalogs,
    safe_divide,
)


def mask_nonzero_displacements(
    sample: CatalogSample,
    dalpha_cosdec_arcsec: np.ndarray,
    ddec_arcsec: np.ndarray,
) -> tuple[CatalogSample, np.ndarray, np.ndarray]:
    amplitude = np.hypot(dalpha_cosdec_arcsec, ddec_arcsec)
    mask = np.isfinite(amplitude) & (amplitude > 0.0)
    return (
        CatalogSample(
            ra=sample.ra[mask],
            dec=sample.dec[mask],
            weights=sample.weights[mask],
            z=None if sample.z is None else sample.z[mask],
        ),
        dalpha_cosdec_arcsec[mask],
        ddec_arcsec[mask],
    )


def make_unit_reference_spin(
    ddec_arcsec: np.ndarray,
    dalpha_cosdec_arcsec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = np.hypot(ddec_arcsec, dalpha_cosdec_arcsec)
    return ddec_arcsec / amplitude, dalpha_cosdec_arcsec / amplitude


def sample_reference_spin(
    unit_north: np.ndarray,
    unit_east: np.ndarray,
    size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, unit_north.size, size=size, dtype=np.int64)
    return unit_north[indices], unit_east[indices]


def compute_shape_maps(
    reference_particles,
    random_reference_particles_list,
    shape_particles,
    shape_scalar_particles,
    battrs: BinAttrs,
    nthreads: int,
) -> tuple[np.ndarray, np.ndarray]:
    spin_wattrs = WeightAttrs(spin=(1, 2), reference_only=(True, False))
    pair_wattrs = WeightAttrs(spin=(1, 0), reference_only=(True, False))

    ds_spin = count2(reference_particles, shape_particles, battrs=battrs, wattrs=spin_wattrs, nthreads=nthreads)
    ds_pairs = count2(reference_particles, shape_scalar_particles, battrs=battrs, wattrs=pair_wattrs, nthreads=nthreads)['weight']
    ds_plus = safe_divide(ds_spin['weight_plus'], ds_pairs)
    ds_cross = safe_divide(ds_spin['weight_cross'], ds_pairs)

    rs_plus_sum = np.zeros_like(ds_plus, dtype=np.float64)
    rs_cross_sum = np.zeros_like(ds_cross, dtype=np.float64)
    for random_reference_particles in random_reference_particles_list:
        rs_spin = count2(random_reference_particles, shape_particles, battrs=battrs, wattrs=spin_wattrs, nthreads=nthreads)
        rs_pairs = count2(random_reference_particles, shape_scalar_particles, battrs=battrs, wattrs=pair_wattrs, nthreads=nthreads)['weight']
        rs_plus_sum += safe_divide(rs_spin['weight_plus'], rs_pairs)
        rs_cross_sum += safe_divide(rs_spin['weight_cross'], rs_pairs)

    rs_plus = rs_plus_sum / len(random_reference_particles_list)
    rs_cross = rs_cross_sum / len(random_reference_particles_list)
    return ds_plus - rs_plus, ds_cross - rs_cross


def plot_single_map(
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    output_path: Path,
) -> None:
    theta_grid, phi_grid = np.meshgrid(theta_edges, np.deg2rad(phi_edges), indexing='ij')
    x = theta_grid * np.cos(phi_grid)
    y = theta_grid * np.sin(phi_grid)

    vmax = float(np.nanmax(np.abs(values)))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    pcm = ax.pcolormesh(x, y, values, shading='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.axhline(0.0, color='0.7', linewidth=1.0)
    ax.axvline(0.0, color='0.7', linewidth=1.0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.set_xlabel(r'$x_{+\mathrm{disp}} = \theta \cos \phi$ [deg]')
    ax.set_ylabel(r'$y_{\perp} = \theta \sin \phi$ [deg]')
    fig.colorbar(pcm, ax=ax, label=colorbar_label)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Measure gamma_+ and gamma_x around the local displacement direction and project them to x/y.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data',
        default=DEFAULT_DISPLACEMENT_DATA,
        required=DEFAULT_DISPLACEMENT_DATA is None,
        help='Path to the displacement catalog. Defaults to $CUCOUNT_DISPLACEMENT_DATA when set.',
    )
    parser.add_argument(
        '--shapes',
        default=DEFAULT_DESI_LRG_SHAPES,
        required=DEFAULT_DESI_LRG_SHAPES is None,
        help='Path to the shape catalog. Defaults to $CUCOUNT_DESI_LRG_SHAPES when set.',
    )
    parser.add_argument(
        '--randoms-glob',
        default=DEFAULT_DESI_RANDOMS_GLOB,
        required=DEFAULT_DESI_RANDOMS_GLOB is None,
        help='Glob pattern for DESI random catalogs. Defaults to $CUCOUNT_DESI_RANDOMS_GLOB when set.',
    )
    parser.add_argument('--shape-weight-col', default='auto', help='Shape-catalog weight column to use, or auto.')
    parser.add_argument('--shape-e1-col', default='e1', help='Shape-catalog e1 column.')
    parser.add_argument('--shape-e2-col', default='e2', help='Shape-catalog e2 column.')
    parser.add_argument('--output-dir', default='examples/output/displacement_shape_phi', help='Directory for outputs.')
    parser.add_argument('--output-prefix', default='desi_lrg_displacement_shape', help='Output file prefix.')
    parser.add_argument('--min-theta', type=float, default=0.05, help='Minimum theta in degrees.')
    parser.add_argument('--max-theta', type=float, default=2.0, help='Maximum theta in degrees.')
    parser.add_argument('--theta-bins', type=int, default=32, help='Number of theta bins.')
    parser.add_argument('--phi-bins', type=int, default=72, help='Number of phi bins over [0, 360) degrees.')
    parser.add_argument('--max-data-rows', type=int, default=None, help='Optional cap on displacement-catalog rows.')
    parser.add_argument('--max-shape-rows', type=int, default=None, help='Optional cap on shape-catalog rows.')
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
    print('Displacement-shape correlation in displacement-aligned coordinates')
    print('=' * 72)
    print(f'Displacement data : {args.data}')
    print(f'Shape catalog     : {args.shapes}')
    print(f'Random catalogs   : {args.randoms_glob}')
    print(f'Theta range       : [{args.min_theta:.3f}, {args.max_theta:.3f}] deg')
    print(f'Theta bins        : {args.theta_bins}')
    print(f'Phi bins          : {args.phi_bins}')
    print('=' * 72)

    data, dalpha_cosdec_arcsec, ddec_arcsec = load_displacement_catalog(
        args.data,
        max_rows=args.max_data_rows,
        seed=args.seed,
    )
    data, dalpha_cosdec_arcsec, ddec_arcsec = mask_nonzero_displacements(data, dalpha_cosdec_arcsec, ddec_arcsec)
    unit_north, unit_east = make_unit_reference_spin(ddec_arcsec, dalpha_cosdec_arcsec)
    print(f'Loaded {data.size:,} displacement tracers with non-zero reference direction')

    shapes = load_unions_catalog(
        args.shapes,
        max_rows=args.max_shape_rows,
        seed=args.seed + 100,
        weight_col=args.shape_weight_col,
        e1_col=args.shape_e1_col,
        e2_col=args.shape_e2_col,
    )
    print(f'Loaded {shapes.size:,} shape tracers')

    reference_particles = create_particles(
        data.ra,
        data.dec,
        data.weights,
        spin_values=(unit_north, unit_east),
    )
    shape_particles = create_particles(shapes.ra, shapes.dec, shapes.weights, shear=(shapes.e1, shapes.e2))
    shape_scalar_particles = create_particles(shapes.ra, shapes.dec, shapes.weights)

    random_paths = resolve_random_catalogs(args.randoms_glob, max_files=args.max_random_files)
    random_reference_particles_list = []
    for index, random_path in enumerate(random_paths, start=1):
        randoms = load_desi_catalog(random_path, max_rows=args.max_random_rows, seed=args.seed + index)
        rand_north, rand_east = sample_reference_spin(unit_north, unit_east, randoms.size, seed=args.seed + 10_000 + index)
        random_reference_particles_list.append(
            create_particles(randoms.ra, randoms.dec, randoms.weights, spin_values=(rand_north, rand_east))
        )
    print(f'Loaded {len(random_reference_particles_list)} random catalogs with sampled displacement directions')

    gamma_plus, gamma_cross = compute_shape_maps(
        reference_particles,
        random_reference_particles_list,
        shape_particles,
        shape_scalar_particles,
        battrs=battrs,
        nthreads=args.nthreads,
    )

    results_path = output_dir / f'{args.output_prefix}_results.npz'
    np.savez(
        results_path,
        theta_edges=theta_edges,
        phi_edges=phi_edges,
        gamma_plus=gamma_plus,
        gamma_cross=gamma_cross,
    )
    print(f'Saved results to {results_path}')

    plus_path = output_dir / f'{args.output_prefix}_gamma_plus_xy.png'
    cross_path = output_dir / f'{args.output_prefix}_gamma_cross_xy.png'
    plot_single_map(theta_edges, phi_edges, gamma_plus, r'$\gamma_+(x, y)$', r'$\gamma_+$', plus_path)
    plot_single_map(theta_edges, phi_edges, gamma_cross, r'$\gamma_\times(x, y)$', r'$\gamma_\times$', cross_path)
    print(f'Saved gamma_+ map to {plus_path}')
    print(f'Saved gamma_x map to {cross_path}')


if __name__ == '__main__':
    main()
