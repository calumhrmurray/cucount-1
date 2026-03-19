#!/usr/bin/env python3
"""
Measure displacement-shape correlations in bins of (s, mu).

Catalog 1 is a spin-2 shape field:
    spin_values = -(e1, e2)

Catalog 2 is the sky-plane displacement field from the DESI displacement
catalogues:
    spin_values = (dDec, dRA * cos(dec))

The projected pair-frame outputs are the weighted pair averages of:
    <gamma_+ d_parallel>(s, mu)
    <gamma_+ d_perp>(s, mu)
    <gamma_x d_parallel>(s, mu)
    <gamma_x d_perp>(s, mu)

By default the displacement components are left in their catalogue units
(typically arcsec), so the correlations have units of displacement. Pass
`--normalize-displacement` to project onto unit sky-plane displacement vectors
while preserving direction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cucount.numpy import BinAttrs, MeshAttrs, WeightAttrs, count2, setup_logging

from observed_catalog_tools import (
    DEFAULT_DESI_LRG_SHAPES,
    DEFAULT_DISPLACEMENT_DATA,
    CatalogSample,
    build_distance_to_comoving,
    create_particles,
    load_displacement_catalog,
    load_unions_catalog,
    safe_divide,
)

DEFAULT_LOS = 'midpoint'


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


def normalize_displacements(
    dalpha_cosdec_arcsec: np.ndarray,
    ddec_arcsec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = np.hypot(dalpha_cosdec_arcsec, ddec_arcsec)
    return dalpha_cosdec_arcsec / amplitude, ddec_arcsec / amplitude


def apply_east_component_sign(
    dalpha_cosdec_arcsec: np.ndarray,
    east_component_sign: float,
) -> np.ndarray:
    return east_component_sign * dalpha_cosdec_arcsec


def compute_shape_displacement_correlations(
    shape_particles,
    displacement_particles,
    shape_scalar_particles,
    displacement_scalar_particles,
    battrs: BinAttrs,
    mesh_refine: float,
    nthreads: int,
) -> dict[str, np.ndarray]:
    spin_wattrs = WeightAttrs(spin=(2, 1))
    sd_mattrs = MeshAttrs(shape_particles, displacement_particles, battrs=battrs, refine=mesh_refine)

    sd_spin = count2(
        shape_particles,
        displacement_particles,
        battrs=battrs,
        wattrs=spin_wattrs,
        mattrs=sd_mattrs,
        nthreads=nthreads,
    )
    sd_pairs = count2(
        shape_scalar_particles,
        displacement_scalar_particles,
        battrs=battrs,
        mattrs=sd_mattrs,
        nthreads=nthreads,
    )['weight']

    return {
        'gamma_plus_d_parallel': safe_divide(sd_spin['weight_plus_plus'], sd_pairs),
        'gamma_plus_d_perp': safe_divide(sd_spin['weight_first_plus_second_cross'], sd_pairs),
        'gamma_cross_d_parallel': safe_divide(sd_spin['weight_first_cross_second_plus'], sd_pairs),
        'gamma_cross_d_perp': safe_divide(sd_spin['weight_cross_cross'], sd_pairs),
    }


def plot_single_smu_map(
    s_edges: np.ndarray,
    mu_edges: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    output_path: Path,
) -> None:
    s_grid, mu_grid = np.meshgrid(s_edges, mu_edges, indexing='ij')
    vmax = float(np.nanmax(np.abs(values)))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    pcm = ax.pcolormesh(s_grid, mu_grid, values, shading='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.axhline(0.0, color='0.7', linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(r'$s$ [$h^{-1}$ Mpc]')
    ax.set_ylabel(r'$\mu$')
    fig.colorbar(pcm, ax=ax, label=colorbar_label)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_smu_summary(
    s_edges: np.ndarray,
    mu_edges: np.ndarray,
    correlations: dict[str, np.ndarray],
    colorbar_label: str,
    output_path: Path,
) -> None:
    s_grid, mu_grid = np.meshgrid(s_edges, mu_edges, indexing='ij')
    figures = [
        ('gamma_plus_d_parallel', r'$\langle \gamma_+ d_{\parallel} \rangle(s,\mu)$'),
        ('gamma_plus_d_perp', r'$\langle \gamma_+ d_{\perp} \rangle(s,\mu)$'),
        ('gamma_cross_d_parallel', r'$\langle \gamma_\times d_{\parallel} \rangle(s,\mu)$'),
        ('gamma_cross_d_perp', r'$\langle \gamma_\times d_{\perp} \rangle(s,\mu)$'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), constrained_layout=True, sharex=True, sharey=True)
    for axis, (key, title) in zip(axes.ravel(), figures, strict=False):
        values = np.asarray(correlations[key], dtype=np.float64)
        vmax = float(np.nanmax(np.abs(values)))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
        pcm = axis.pcolormesh(s_grid, mu_grid, values, shading='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
        axis.axhline(0.0, color='0.7', linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel(r'$s$ [$h^{-1}$ Mpc]')
        axis.set_ylabel(r'$\mu$')
        fig.colorbar(pcm, ax=axis, label=colorbar_label)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Measure displacement-shape correlations in (s, mu) bins.',
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
    parser.add_argument('--shape-weight-col', default='auto', help='Shape-catalog weight column to use, or auto.')
    parser.add_argument('--shape-e1-col', default='e1', help='Shape-catalog e1 column.')
    parser.add_argument('--shape-e2-col', default='e2', help='Shape-catalog e2 column.')
    parser.add_argument('--shape-z-col', default='auto', help='Shape-catalog redshift column to use, or auto.')
    parser.add_argument('--output-dir', default='examples/output/displacement_shape_smu', help='Directory for outputs.')
    parser.add_argument('--output-prefix', default='desi_displacement_shape_smu', help='Output file prefix.')
    parser.add_argument('--min-s', type=float, default=1.0, help='Minimum 3D separation in h^-1 Mpc.')
    parser.add_argument('--max-s', type=float, default=120.0, help='Maximum 3D separation in h^-1 Mpc.')
    parser.add_argument('--s-bins', type=int, default=48, help='Number of separation bins.')
    parser.add_argument('--mu-bins', type=int, default=40, help='Number of mu bins over [-1, 1].')
    parser.add_argument('--los', default=DEFAULT_LOS, choices=['midpoint', 'firstpoint', 'endpoint', 'x', 'y', 'z'], help='Line-of-sight definition for mu.')
    parser.add_argument(
        '--east-component-sign',
        type=float,
        default=1.0,
        choices=(-1.0, 1.0),
        help='Multiply the eastward displacement component dRA*cos(dec) by this sign before building the spin-1 field.',
    )
    parser.add_argument(
        '--normalize-displacement',
        action='store_true',
        help='Normalize each non-zero sky-plane displacement vector to unit length before building the spin-1 field.',
    )
    parser.add_argument('--max-data-rows', type=int, default=None, help='Optional cap on displacement-catalog rows.')
    parser.add_argument('--max-shape-rows', type=int, default=None, help='Optional cap on shape-catalog rows.')
    parser.add_argument('--seed', type=int, default=1234, help='Seed for reproducible sub-sampling.')
    parser.add_argument('--mesh-refine', type=float, default=5.0, help='Mesh refinement factor passed to cucount MeshAttrs.')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of GPUs for cucount to use.')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level)

    s_edges = np.linspace(args.min_s, args.max_s, args.s_bins + 1)
    mu_edges = np.linspace(-1.0, 1.0, args.mu_bins + 1)
    battrs = BinAttrs(s=s_edges, mu=(mu_edges, args.los))
    distance_to_comoving, distance_label = build_distance_to_comoving()

    print('=' * 72)
    print('Displacement-shape correlation in (s, mu) bins')
    print('=' * 72)
    print(f'Displacement data : {args.data}')
    print(f'Shape catalog     : {args.shapes}')
    print(f's range           : [{args.min_s:.3f}, {args.max_s:.3f}] h^-1 Mpc')
    print(f's bins            : {args.s_bins}')
    print(f'mu bins           : {args.mu_bins}')
    print(f'LOS               : {args.los}')
    print(f'East comp. sign   : {args.east_component_sign:+.0f}')
    print(f'Displacement mode : {"unit direction vectors" if args.normalize_displacement else "catalog displacement amplitudes"}')
    print(f'Distance model    : {distance_label}')
    print(f'Mesh refine       : {args.mesh_refine:.2f}')
    print('=' * 72)

    data, dalpha_cosdec_arcsec, ddec_arcsec = load_displacement_catalog(
        args.data,
        max_rows=args.max_data_rows,
        seed=args.seed,
    )
    data, dalpha_cosdec_arcsec, ddec_arcsec = mask_nonzero_displacements(data, dalpha_cosdec_arcsec, ddec_arcsec)
    dalpha_cosdec_arcsec = apply_east_component_sign(dalpha_cosdec_arcsec, args.east_component_sign)
    if args.normalize_displacement:
        dalpha_cosdec_arcsec, ddec_arcsec = normalize_displacements(dalpha_cosdec_arcsec, ddec_arcsec)
    print(f'Loaded {data.size:,} displacement tracers with non-zero displacement')

    shapes = load_unions_catalog(
        args.shapes,
        max_rows=args.max_shape_rows,
        seed=args.seed + 100,
        weight_col=args.shape_weight_col,
        e1_col=args.shape_e1_col,
        e2_col=args.shape_e2_col,
        z_col=args.shape_z_col,
    )
    print(f'Loaded {shapes.size:,} shape tracers')

    if (data.z is None) or (shapes.z is None):
        raise ValueError('This (s, mu) workflow requires redshift columns for both the displacement and shape catalogs')

    data_distance = distance_to_comoving(data.z)
    shape_distance = distance_to_comoving(shapes.z)

    shape_particles = create_particles(
        shapes.ra,
        shapes.dec,
        shapes.weights,
        distance=shape_distance,
        shear=(shapes.e1, shapes.e2),
    )
    displacement_particles = create_particles(
        data.ra,
        data.dec,
        data.weights,
        distance=data_distance,
        spin_values=(ddec_arcsec, dalpha_cosdec_arcsec),
    )
    shape_scalar_particles = create_particles(shapes.ra, shapes.dec, shapes.weights, distance=shape_distance)
    displacement_scalar_particles = create_particles(data.ra, data.dec, data.weights, distance=data_distance)

    correlations = compute_shape_displacement_correlations(
        shape_particles,
        displacement_particles,
        shape_scalar_particles,
        displacement_scalar_particles,
        battrs=battrs,
        mesh_refine=args.mesh_refine,
        nthreads=args.nthreads,
    )

    results_path = output_dir / f'{args.output_prefix}_results.npz'
    colorbar_label = 'correlation [dimensionless]' if args.normalize_displacement else 'correlation [arcsec]'
    np.savez(
        results_path,
        s_edges=s_edges,
        mu_edges=mu_edges,
        los=args.los,
        east_component_sign=args.east_component_sign,
        normalize_displacement=args.normalize_displacement,
        displacement_units='unit_vector' if args.normalize_displacement else 'arcsec',
        **correlations,
    )
    print(f'Saved results to {results_path}')

    summary_path = output_dir / f'{args.output_prefix}_smu_summary.png'
    plot_smu_summary(s_edges, mu_edges, correlations, colorbar_label, summary_path)
    print(f'Saved summary figure to {summary_path}')

    single_figures = {
        'gamma_plus_d_parallel': output_dir / f'{args.output_prefix}_gamma_plus_d_parallel.png',
        'gamma_plus_d_perp': output_dir / f'{args.output_prefix}_gamma_plus_d_perp.png',
        'gamma_cross_d_parallel': output_dir / f'{args.output_prefix}_gamma_cross_d_parallel.png',
        'gamma_cross_d_perp': output_dir / f'{args.output_prefix}_gamma_cross_d_perp.png',
    }
    titles = {
        'gamma_plus_d_parallel': r'$\langle \gamma_+ d_{\parallel} \rangle(s,\mu)$',
        'gamma_plus_d_perp': r'$\langle \gamma_+ d_{\perp} \rangle(s,\mu)$',
        'gamma_cross_d_parallel': r'$\langle \gamma_\times d_{\parallel} \rangle(s,\mu)$',
        'gamma_cross_d_perp': r'$\langle \gamma_\times d_{\perp} \rangle(s,\mu)$',
    }
    for key, output_path in single_figures.items():
        plot_single_smu_map(
            s_edges,
            mu_edges,
            correlations[key],
            titles[key],
            colorbar_label,
            output_path,
        )
        print(f'Saved {key} map to {output_path}')


if __name__ == '__main__':
    main()
