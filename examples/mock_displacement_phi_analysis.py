#!/usr/bin/env python3
"""
Run displacement x density and displacement x shape phi-correlations on TFC mock catalogues.

This script is designed for the mock catalogues used by:
    /sps/euclid/Users/cmurray/cc_lyon_ia/intrinsic_alignment_analysis/notebooks/plot_tfc_mock_results.ipynb

Stage assumptions:
- ``initial`` uses the total stored displacement ``Position - POSITION_INITIAL``
  projected at the initial sky position.
- ``formation`` uses ``POSITION_FORMATION - POSITION_INITIAL`` projected at the
  formation sky position.
- ``final`` uses the total stored displacement ``Position - POSITION_INITIAL``
  projected at the final sky position.

When ``--pi-max`` is supplied, the stage-specific real-space redshift columns
are converted to comoving distance so a line-of-sight cut can be applied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

from cucount.numpy import BinAttrs, setup_logging

from displacement_density_phi_map import normalized_density_correlation, plot_xy_map
from displacement_shape_phi_map import (
    DEFAULT_PI_LOS,
    collapse_single_pi_bin,
    compute_shape_maps,
    gaussian_smooth_map,
    plot_single_map,
    plot_smoothed_maps_with_shear_field,
    sample_reference_spin,
)
from observed_catalog_tools import CatalogSample, build_distance_to_comoving, choose_rows, create_particles

STAGE_CHOICES = ('initial', 'formation', 'final')
STAGE_REDSHIFT_COLUMNS = {
    'initial': 'Z_REAL_INITIAL',
    'formation': 'Z_REAL_FORMATION',
    'final': 'Z_REAL_FINAL',
}


def tangent_basis_from_radec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))

    east = np.column_stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)])
    north = np.column_stack([
        -np.sin(dec) * np.cos(ra),
        -np.sin(dec) * np.sin(ra),
        np.cos(dec),
    ])
    return east, north


def project_displacement_to_sky(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    displacement_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    east, north = tangent_basis_from_radec(ra_deg, dec_deg)
    displacement_xyz = np.asarray(displacement_xyz, dtype=np.float64)
    d_east = np.einsum('ij,ij->i', displacement_xyz, east)
    d_north = np.einsum('ij,ij->i', displacement_xyz, north)
    return d_east, d_north


def get_stage_displacement(
    data,
    stage: str,
    row,
) -> np.ndarray:
    if stage == 'formation':
        position_stage = np.asarray(data['POSITION_FORMATION'][row], dtype=np.float64)
        position_initial = np.asarray(data['POSITION_INITIAL'][row], dtype=np.float64)
        return position_stage - position_initial

    return np.column_stack([
        np.asarray(data['DISPLACEMENT_X'][row], dtype=np.float64),
        np.asarray(data['DISPLACEMENT_Y'][row], dtype=np.float64),
        np.asarray(data['DISPLACEMENT_Z'][row], dtype=np.float64),
    ])


def load_mock_stage(
    path: str,
    stage: str,
    max_rows: int | None = None,
    seed: int = 0,
) -> tuple[CatalogSample, np.ndarray, np.ndarray]:
    stage_upper = stage.upper()
    redshift_col = STAGE_REDSHIFT_COLUMNS[stage]
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        indices = choose_rows(len(data), max_rows=max_rows, seed=seed)
        row = slice(None) if indices is None else indices

        ra = np.asarray(data[f'RA_{stage_upper}'][row], dtype=np.float64)
        dec = np.asarray(data[f'DEC_{stage_upper}'][row], dtype=np.float64)
        redshift = np.asarray(data[redshift_col][row], dtype=np.float64)
        weights = np.asarray(data['WEIGHT'][row], dtype=np.float64)
        e1 = np.asarray(data[f'S1_{stage_upper}'][row], dtype=np.float64)
        e2 = np.asarray(data[f'S2_{stage_upper}'][row], dtype=np.float64)
        displacement_xyz = get_stage_displacement(data, stage=stage, row=row)

    d_east, d_north = project_displacement_to_sky(ra, dec, displacement_xyz)
    amplitude = np.hypot(d_north, d_east)
    mask = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(redshift)
        & np.isfinite(weights)
        & np.isfinite(e1)
        & np.isfinite(e2)
        & np.isfinite(d_north)
        & np.isfinite(d_east)
        & (redshift > 0.0)
        & (weights > 0.0)
        & (amplitude > 0.0)
    )
    sample = CatalogSample(
        ra=ra[mask],
        dec=dec[mask],
        z=redshift[mask],
        weights=weights[mask],
        e1=e1[mask],
        e2=e2[mask],
    )
    return sample, d_east[mask], d_north[mask]


def load_mock_randoms(
    path: str,
    max_rows: int | None = None,
    seed: int = 0,
) -> CatalogSample:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        indices = choose_rows(len(data), max_rows=max_rows, seed=seed)
        row = slice(None) if indices is None else indices
        ra = np.asarray(data['RA'][row], dtype=np.float64)
        dec = np.asarray(data['DEC'][row], dtype=np.float64)
        redshift = np.asarray(data['Z'][row], dtype=np.float64)
        weights = np.asarray(data['WEIGHT'][row], dtype=np.float64)

    mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(redshift) & np.isfinite(weights) & (redshift > 0.0) & (weights > 0.0)
    return CatalogSample(ra=ra[mask], dec=dec[mask], z=redshift[mask], weights=weights[mask])


def run_stage(
    stage: str,
    data_path: str,
    randoms_path: str,
    transverse_edges: np.ndarray,
    phi_edges: np.ndarray,
    output_dir: Path,
    max_data_rows: int | None,
    max_random_rows: int | None,
    seed: int,
    mesh_refine: float,
    nthreads: int,
    smooth_sigma_theta_bins: float,
    smooth_sigma_phi_bins: float,
    transverse_bin: str,
    pi_max: float | None,
    distance_to_comoving,
) -> None:
    pi_edges = None
    if pi_max is not None:
        pi_edges = np.array([-pi_max, pi_max], dtype=np.float64)
    if transverse_bin == 'rp':
        if pi_edges is None:
            battrs = BinAttrs(rp=(transverse_edges, DEFAULT_PI_LOS), phi=phi_edges)
        else:
            battrs = BinAttrs(rp=(transverse_edges, DEFAULT_PI_LOS), phi=phi_edges, pi=(pi_edges, DEFAULT_PI_LOS))
    else:
        if pi_edges is None:
            battrs = BinAttrs(theta=transverse_edges, phi=phi_edges)
        else:
            battrs = BinAttrs(theta=transverse_edges, phi=phi_edges, pi=(pi_edges, DEFAULT_PI_LOS))
    sample, d_east, d_north = load_mock_stage(data_path, stage=stage, max_rows=max_data_rows, seed=seed)
    unit_north = d_north / np.hypot(d_north, d_east)
    unit_east = d_east / np.hypot(d_north, d_east)
    randoms = load_mock_randoms(randoms_path, max_rows=max_random_rows, seed=seed + 1000)
    random_north, random_east = sample_reference_spin(unit_north, unit_east, randoms.size, seed=seed + 2000)
    sample_distance = None if distance_to_comoving is None else distance_to_comoving(sample.z)
    random_distance = None if distance_to_comoving is None else distance_to_comoving(randoms.z)

    spin_particles = create_particles(sample.ra, sample.dec, sample.weights, distance=sample_distance, spin_values=(d_north, d_east))
    density_particles = create_particles(sample.ra, sample.dec, sample.weights, distance=sample_distance)
    random_particles_list = [create_particles(randoms.ra, randoms.dec, randoms.weights, distance=random_distance)]

    reference_particles = create_particles(
        sample.ra,
        sample.dec,
        sample.weights,
        distance=sample_distance,
        spin_values=(unit_north, unit_east),
    )
    random_reference_particles_list = [
        create_particles(
            randoms.ra,
            randoms.dec,
            randoms.weights,
            distance=random_distance,
            spin_values=(random_north, random_east),
        )
    ]
    shape_particles = create_particles(
        sample.ra,
        sample.dec,
        sample.weights,
        distance=sample_distance,
        shear=(sample.e1, sample.e2),
    )

    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    print('-' * 72)
    print(f'Mock stage: {stage}')
    print(f'  Data rows kept   : {sample.size:,}')
    print(f'  Random rows kept : {randoms.size:,}')
    if pi_edges is not None:
        print(f'  Pi cut           : [{pi_edges[0]:.1f}, {pi_edges[1]:.1f}] h^-1 Mpc ({DEFAULT_PI_LOS} LOS)')

    xi_density, mean_plus, mean_cross = normalized_density_correlation(
        spin_particles,
        density_particles,
        random_particles_list,
        battrs=battrs,
        nthreads=nthreads,
    )
    xi_density = collapse_single_pi_bin(xi_density)
    mean_plus = collapse_single_pi_bin(mean_plus)
    mean_cross = collapse_single_pi_bin(mean_cross)
    density_results_path = stage_dir / f'tfc_mock_{stage}_displacement_density_results.npz'
    density_payload = {
        'phi_edges': phi_edges,
        'xi_density': xi_density,
        'mean_plus': mean_plus,
        'mean_cross': mean_cross,
        'transverse_bin': transverse_bin,
    }
    if transverse_bin == 'rp':
        density_payload['rp_edges'] = transverse_edges
    else:
        density_payload['theta_edges'] = transverse_edges
    if pi_edges is not None:
        density_payload['pi_edges'] = pi_edges
        density_payload['pi_los'] = DEFAULT_PI_LOS
    np.savez(density_results_path, **density_payload)
    density_figure_path = stage_dir / f'tfc_mock_{stage}_displacement_density_xy.png'
    plot_xy_map(
        transverse_edges,
        phi_edges,
        xi_density,
        mean_plus,
        mean_cross,
        density_figure_path,
        transverse_bin=transverse_bin,
    )
    print(f'  Saved density results to {density_results_path}')

    gamma_plus, gamma_cross = compute_shape_maps(
        reference_particles,
        random_reference_particles_list,
        shape_particles,
        battrs=battrs,
        mesh_refine=mesh_refine,
        nthreads=nthreads,
    )
    gamma_plus = collapse_single_pi_bin(gamma_plus)
    gamma_cross = collapse_single_pi_bin(gamma_cross)
    gamma_plus_smoothed = gaussian_smooth_map(gamma_plus, smooth_sigma_theta_bins, smooth_sigma_phi_bins)
    gamma_cross_smoothed = gaussian_smooth_map(gamma_cross, smooth_sigma_theta_bins, smooth_sigma_phi_bins)

    shape_results_path = stage_dir / f'tfc_mock_{stage}_displacement_shape_results.npz'
    shape_payload = {
        'phi_edges': phi_edges,
        'gamma_plus': gamma_plus,
        'gamma_cross': gamma_cross,
        'gamma_plus_smoothed': gamma_plus_smoothed,
        'gamma_cross_smoothed': gamma_cross_smoothed,
        'smooth_sigma_theta_bins': smooth_sigma_theta_bins,
        'smooth_sigma_phi_bins': smooth_sigma_phi_bins,
        'transverse_bin': transverse_bin,
    }
    if transverse_bin == 'rp':
        shape_payload['rp_edges'] = transverse_edges
    else:
        shape_payload['theta_edges'] = transverse_edges
    if pi_edges is not None:
        shape_payload['pi_edges'] = pi_edges
        shape_payload['pi_los'] = DEFAULT_PI_LOS
    np.savez(shape_results_path, **shape_payload)
    plus_path = stage_dir / f'tfc_mock_{stage}_displacement_shape_gamma_plus_xy.png'
    cross_path = stage_dir / f'tfc_mock_{stage}_displacement_shape_gamma_cross_xy.png'
    plus_smoothed_path = stage_dir / f'tfc_mock_{stage}_displacement_shape_gamma_plus_xy_smoothed.png'
    cross_smoothed_path = stage_dir / f'tfc_mock_{stage}_displacement_shape_gamma_cross_xy_smoothed.png'
    overlay_path = stage_dir / f'tfc_mock_{stage}_displacement_shape_smoothed_shear_field_overlay.png'

    plot_single_map(transverse_edges, phi_edges, gamma_plus, r'$\gamma_+(x, y)$', r'$\gamma_+$', plus_path, transverse_bin=transverse_bin)
    plot_single_map(transverse_edges, phi_edges, gamma_cross, r'$\gamma_\times(x, y)$', r'$\gamma_\times$', cross_path, transverse_bin=transverse_bin)
    plot_single_map(
        transverse_edges,
        phi_edges,
        gamma_plus_smoothed,
        r'$\gamma_+(x, y)$ Gaussian-smoothed',
        r'$\gamma_+$',
        plus_smoothed_path,
        transverse_bin=transverse_bin,
    )
    plot_single_map(
        transverse_edges,
        phi_edges,
        gamma_cross_smoothed,
        r'$\gamma_\times(x, y)$ Gaussian-smoothed',
        r'$\gamma_\times$',
        cross_smoothed_path,
        transverse_bin=transverse_bin,
    )
    plot_smoothed_maps_with_shear_field(
        transverse_edges,
        phi_edges,
        gamma_plus_smoothed,
        gamma_cross_smoothed,
        overlay_path,
        transverse_bin=transverse_bin,
    )
    print(f'  Saved shape results to {shape_results_path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run displacement x density and displacement x shape phi-correlations on TFC mock catalogues.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data', required=True, help='Path to the mock galaxy catalogue FITS file.')
    parser.add_argument('--randoms', required=True, help='Path to the mock random catalogue FITS file.')
    parser.add_argument(
        '--stages',
        nargs='+',
        default=list(STAGE_CHOICES),
        choices=list(STAGE_CHOICES),
        help='Mock stages to analyse.',
    )
    parser.add_argument('--output-dir', default='examples/output/mock_displacement_phi', help='Directory for outputs.')
    parser.add_argument('--transverse-bin', default='theta', choices=['theta', 'rp'], help='Use angular theta bins or projected-separation rp bins.')
    parser.add_argument('--min-theta', type=float, default=0.05, help='Minimum theta in degrees.')
    parser.add_argument('--max-theta', type=float, default=2.0, help='Maximum theta in degrees.')
    parser.add_argument('--theta-bins', type=int, default=32, help='Number of theta bins.')
    parser.add_argument('--min-rp', type=float, default=0.2, help='Minimum projected separation rp in h^-1 Mpc.')
    parser.add_argument('--max-rp', type=float, default=40.0, help='Maximum projected separation rp in h^-1 Mpc.')
    parser.add_argument('--rp-bins', type=int, default=32, help='Number of rp bins.')
    parser.add_argument('--phi-bins', type=int, default=72, help='Number of phi bins over [0, 360) degrees.')
    parser.add_argument('--pi-max', type=float, default=None, help='If set, keep only pairs with |pi| <= this value in h^-1 Mpc using the first-point LOS.')
    parser.add_argument('--max-data-rows', type=int, default=None, help='Optional cap on mock-galaxy rows.')
    parser.add_argument('--max-random-rows', type=int, default=None, help='Optional cap on random rows.')
    parser.add_argument('--smooth-sigma-theta-bins', type=float, default=1.0)
    parser.add_argument('--smooth-sigma-phi-bins', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--mesh-refine', type=float, default=5.0, help='Mesh refinement factor passed to cucount MeshAttrs.')
    parser.add_argument('--nthreads', type=int, default=1)
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'])
    args = parser.parse_args()

    setup_logging(args.log_level)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    theta_edges = np.linspace(args.min_theta, args.max_theta, args.theta_bins + 1)
    rp_edges = np.linspace(args.min_rp, args.max_rp, args.rp_bins + 1)
    phi_edges = np.linspace(0.0, 360.0, args.phi_bins + 1)
    distance_to_comoving = None
    distance_label = None
    if (args.transverse_bin == 'rp') or (args.pi_max is not None):
        distance_to_comoving, distance_label = build_distance_to_comoving()
    if args.transverse_bin == 'rp':
        transverse_edges = rp_edges
    else:
        transverse_edges = theta_edges

    print('=' * 72)
    print('Mock displacement x density / shape phi-analysis')
    print('=' * 72)
    print(f'Mock data        : {args.data}')
    print(f'Mock randoms     : {args.randoms}')
    print(f'Stages           : {", ".join(args.stages)}')
    if args.transverse_bin == 'rp':
        print(f'rp range         : [{args.min_rp:.3f}, {args.max_rp:.3f}] h^-1 Mpc')
        print(f'rp bins          : {args.rp_bins}')
    else:
        print(f'Theta range      : [{args.min_theta:.3f}, {args.max_theta:.3f}] deg')
        print(f'Theta bins       : {args.theta_bins}')
    if args.pi_max is not None:
        print(f'Pi cut           : [-{args.pi_max:.1f}, {args.pi_max:.1f}] h^-1 Mpc ({DEFAULT_PI_LOS} LOS)')
    if distance_label is not None:
        print(f'Distance model   : {distance_label}')
    print(f'Mesh refine      : {args.mesh_refine:.2f}')
    print('Displacement use : initial/final use total displacement; formation uses formation-initial displacement')
    print('=' * 72)

    for istage, stage in enumerate(args.stages):
        run_stage(
            stage=stage,
            data_path=args.data,
            randoms_path=args.randoms,
            transverse_edges=transverse_edges,
            phi_edges=phi_edges,
            output_dir=output_dir,
            max_data_rows=args.max_data_rows,
            max_random_rows=args.max_random_rows,
            seed=args.seed + 10 * istage,
            mesh_refine=args.mesh_refine,
            nthreads=args.nthreads,
            smooth_sigma_theta_bins=args.smooth_sigma_theta_bins,
            smooth_sigma_phi_bins=args.smooth_sigma_phi_bins,
            transverse_bin=args.transverse_bin,
            pi_max=args.pi_max,
            distance_to_comoving=distance_to_comoving,
        )


if __name__ == '__main__':
    main()
