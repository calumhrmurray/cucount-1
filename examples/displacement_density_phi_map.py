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

from cucount.numpy import BinAttrs, MeshAttrs, WeightAttrs, count2, setup_logging

from observed_catalog_tools import (
    DEFAULT_DESI_RANDOMS_GLOB,
    DEFAULT_DISPLACEMENT_DATA,
    build_distance_to_comoving,
    create_particles,
    load_displacement_catalog,
    load_desi_catalog,
    resolve_random_catalogs,
    safe_divide,
)

DEFAULT_PI_LOS = 'firstpoint'


def normalized_density_correlation(
    spin_particles,
    density_particles,
    random_particles_list,
    battrs: BinAttrs,
    mesh_refine: float,
    nthreads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wattrs = WeightAttrs(spin=(1, 0))
    dd_mattrs = MeshAttrs(spin_particles, density_particles, battrs=battrs, refine=mesh_refine)
    dd = count2(spin_particles, density_particles, battrs=battrs, wattrs=wattrs, mattrs=dd_mattrs, nthreads=nthreads)

    dd_weight = np.asarray(dd['weight'], dtype=np.float64)
    dd_plus = np.asarray(dd['weight_plus'], dtype=np.float64)
    dd_cross = np.asarray(dd['weight_cross'], dtype=np.float64)

    dr_weight_sum = np.zeros_like(dd_weight)
    dr_plus_sum = np.zeros_like(dd_plus)
    dr_cross_sum = np.zeros_like(dd_cross)
    for random_particles in random_particles_list:
        dr_mattrs = MeshAttrs(spin_particles, random_particles, battrs=battrs, refine=mesh_refine)
        dr = count2(spin_particles, random_particles, battrs=battrs, wattrs=wattrs, mattrs=dr_mattrs, nthreads=nthreads)
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


def collapse_single_pi_bin(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2:
        return values
    if (values.ndim == 3) and (values.shape[-1] == 1):
        return values[..., 0]
    raise ValueError(f'Expected a 2D map or a single pi bin, got shape {values.shape}')


def get_xy_axis_labels(transverse_bin: str) -> tuple[str, str]:
    if transverse_bin == 'rp':
        return (
            r'$x_{+\mathrm{disp}} = r_p \cos \phi$ [$h^{-1}$ Mpc]',
            r'$y_{\perp} = r_p \sin \phi$ [$h^{-1}$ Mpc]',
        )
    return (
        r'$x_{+\mathrm{disp}} = \theta \cos \phi$ [deg]',
        r'$y_{\perp} = \theta \sin \phi$ [deg]',
    )


def plot_xy_map(
    transverse_edges: np.ndarray,
    phi_edges: np.ndarray,
    xi_density: np.ndarray,
    mean_plus: np.ndarray,
    mean_cross: np.ndarray,
    output_path: Path,
    transverse_bin: str = 'theta',
) -> None:
    # phi is defined relative to the positive displacement direction, so +x is
    # "along +displacement" and points to the right in the rendered image.
    transverse_grid, phi_grid = np.meshgrid(transverse_edges, np.deg2rad(phi_edges), indexing='ij')
    x = transverse_grid * np.cos(phi_grid)
    y = transverse_grid * np.sin(phi_grid)

    transverse_centers = 0.5 * (transverse_edges[:-1] + transverse_edges[1:])
    phi_centers = np.deg2rad(0.5 * (phi_edges[:-1] + phi_edges[1:]))
    rr, pp = np.meshgrid(transverse_centers, phi_centers, indexing='ij')
    xc = rr * np.cos(pp)
    yc = rr * np.sin(pp)
    ux = -(mean_plus * np.cos(pp) - mean_cross * np.sin(pp))
    uy = -(mean_plus * np.sin(pp) + mean_cross * np.cos(pp))
    xlabel, ylabel = get_xy_axis_labels(transverse_bin)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    pcm = axes[0].pcolormesh(x, y, xi_density, shading='auto', cmap='coolwarm')
    axes[0].axhline(0.0, color='0.7', linewidth=1.0)
    axes[0].axvline(0.0, color='0.7', linewidth=1.0)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_title('Density Correlation in Displacement-Aligned Frame')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    fig.colorbar(pcm, ax=axes[0], label=r'$\xi(x, y)$')

    axes[1].pcolormesh(x, y, xi_density, shading='auto', cmap='Greys', alpha=0.35)
    axes[1].quiver(xc, yc, ux, uy, color='C0', angles='xy', scale_units='xy', scale=None, width=0.003)
    axes[1].axhline(0.0, color='0.7', linewidth=1.0)
    axes[1].axvline(0.0, color='0.7', linewidth=1.0)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_title('Mean Projected Displacement')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)

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
    parser.add_argument('--transverse-bin', default='theta', choices=['theta', 'rp'], help='Use angular theta bins or projected-separation rp bins.')
    parser.add_argument('--min-theta', type=float, default=0.05, help='Minimum theta in degrees.')
    parser.add_argument('--max-theta', type=float, default=2.0, help='Maximum theta in degrees.')
    parser.add_argument('--theta-bins', type=int, default=32, help='Number of theta bins.')
    parser.add_argument('--min-rp', type=float, default=0.2, help='Minimum projected separation rp in h^-1 Mpc.')
    parser.add_argument('--max-rp', type=float, default=40.0, help='Maximum projected separation rp in h^-1 Mpc.')
    parser.add_argument('--rp-bins', type=int, default=32, help='Number of rp bins.')
    parser.add_argument('--phi-bins', type=int, default=72, help='Number of phi bins over [0, 360) degrees.')
    parser.add_argument('--pi-max', type=float, default=None, help='If set, keep only pairs with |pi| <= this value in h^-1 Mpc using the first-point LOS.')
    parser.add_argument('--max-data-rows', type=int, default=None, help='Optional cap on displacement-catalog rows.')
    parser.add_argument('--max-random-files', type=int, default=4, help='Optional cap on random files.')
    parser.add_argument('--max-random-rows', type=int, default=None, help='Optional cap on rows per random file.')
    parser.add_argument('--seed', type=int, default=1234, help='Seed for reproducible sub-sampling.')
    parser.add_argument('--mesh-refine', type=float, default=5.0, help='Mesh refinement factor passed to cucount MeshAttrs.')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of GPUs for cucount to use.')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level)

    theta_edges = np.linspace(args.min_theta, args.max_theta, args.theta_bins + 1)
    rp_edges = np.linspace(args.min_rp, args.max_rp, args.rp_bins + 1)
    phi_edges = np.linspace(0.0, 360.0, args.phi_bins + 1)
    pi_edges = None
    need_distance = (args.transverse_bin == 'rp') or (args.pi_max is not None)
    distance_to_comoving = None
    distance_label = None
    if need_distance:
        distance_to_comoving, distance_label = build_distance_to_comoving()
    if args.pi_max is not None:
        pi_edges = np.array([-args.pi_max, args.pi_max], dtype=np.float64)
    if args.transverse_bin == 'rp':
        if pi_edges is None:
            battrs = BinAttrs(rp=(rp_edges, DEFAULT_PI_LOS), phi=phi_edges)
        else:
            battrs = BinAttrs(rp=(rp_edges, DEFAULT_PI_LOS), phi=phi_edges, pi=(pi_edges, DEFAULT_PI_LOS))
        transverse_edges = rp_edges
    else:
        if pi_edges is None:
            battrs = BinAttrs(theta=theta_edges, phi=phi_edges)
        else:
            battrs = BinAttrs(theta=theta_edges, phi=phi_edges, pi=(pi_edges, DEFAULT_PI_LOS))
        transverse_edges = theta_edges

    print('=' * 72)
    print('Displacement-density correlation in displacement-aligned coordinates')
    print('=' * 72)
    print(f'Data catalog     : {args.data}')
    print(f'Random catalogs  : {args.randoms_glob}')
    if args.transverse_bin == 'rp':
        print(f'rp range         : [{args.min_rp:.3f}, {args.max_rp:.3f}] h^-1 Mpc')
        print(f'rp bins          : {args.rp_bins}')
    else:
        print(f'Theta range      : [{args.min_theta:.3f}, {args.max_theta:.3f}] deg')
        print(f'Theta bins       : {args.theta_bins}')
    print(f'Phi bins         : {args.phi_bins}')
    if pi_edges is not None:
        print(f'Pi cut           : [{pi_edges[0]:.1f}, {pi_edges[1]:.1f}] h^-1 Mpc ({DEFAULT_PI_LOS} LOS)')
    if need_distance:
        print(f'Distance model   : {distance_label}')
    print(f'Mesh refine      : {args.mesh_refine:.2f}')
    print('=' * 72)

    data, dalpha_cosdec_arcsec, ddec_arcsec = load_displacement_catalog(
        args.data,
        max_rows=args.max_data_rows,
        seed=args.seed,
    )
    print(f'Loaded {data.size:,} displacement tracers')
    data_distance = None if not need_distance else distance_to_comoving(data.z)

    # For spin-1 vectors the kernel expects (north, east) components with no
    # extra shear-convention sign flip.
    spin_particles = create_particles(
        data.ra,
        data.dec,
        data.weights,
        distance=data_distance,
        spin_values=(ddec_arcsec, dalpha_cosdec_arcsec),
    )
    density_particles = create_particles(data.ra, data.dec, data.weights, distance=data_distance)

    random_paths = resolve_random_catalogs(args.randoms_glob, max_files=args.max_random_files)
    random_particles_list = []
    for index, random_path in enumerate(random_paths, start=1):
        randoms = load_desi_catalog(random_path, max_rows=args.max_random_rows, seed=args.seed + index)
        random_distance = None if not need_distance else distance_to_comoving(randoms.z)
        random_particles_list.append(create_particles(randoms.ra, randoms.dec, randoms.weights, distance=random_distance))
    print(f'Loaded {len(random_particles_list)} random catalogs')

    xi_density, mean_plus, mean_cross = normalized_density_correlation(
        spin_particles,
        density_particles,
        random_particles_list,
        battrs=battrs,
        mesh_refine=args.mesh_refine,
        nthreads=args.nthreads,
    )
    if pi_edges is not None:
        xi_density = collapse_single_pi_bin(xi_density)
        mean_plus = collapse_single_pi_bin(mean_plus)
        mean_cross = collapse_single_pi_bin(mean_cross)

    results_path = output_dir / f'{args.output_prefix}_results.npz'
    payload = {
        'phi_edges': phi_edges,
        'xi_density': xi_density,
        'mean_plus': mean_plus,
        'mean_cross': mean_cross,
        'transverse_bin': args.transverse_bin,
    }
    if args.transverse_bin == 'rp':
        payload['rp_edges'] = rp_edges
    else:
        payload['theta_edges'] = theta_edges
    if pi_edges is not None:
        payload['pi_edges'] = pi_edges
        payload['pi_los'] = DEFAULT_PI_LOS
    np.savez(results_path, **payload)
    print(f'Saved results to {results_path}')

    figure_path = output_dir / f'{args.output_prefix}_xy.png'
    plot_xy_map(transverse_edges, phi_edges, xi_density, mean_plus, mean_cross, figure_path, transverse_bin=args.transverse_bin)
    print(f'Saved figure to {figure_path}')


if __name__ == '__main__':
    main()
