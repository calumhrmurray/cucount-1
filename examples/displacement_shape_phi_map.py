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
+displacement, to the right. Each run saves both the raw map and a Gaussian-
smoothed view of the same signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from cucount.numpy import BinAttrs, MeshAttrs, WeightAttrs, count2, setup_logging

from observed_catalog_tools import (
    DEFAULT_DESI_LRG_SHAPES,
    DEFAULT_DESI_RANDOMS_GLOB,
    DEFAULT_DISPLACEMENT_DATA,
    CatalogSample,
    build_distance_to_comoving,
    create_particles,
    load_desi_catalog,
    load_displacement_catalog,
    load_unions_catalog,
    resolve_random_catalogs,
    safe_divide,
)

DEFAULT_PI_LOS = 'firstpoint'


def gaussian_kernel1d(sigma_bins: float, truncate: float = 3.0) -> np.ndarray:
    if sigma_bins <= 0.0:
        return np.array([1.0], dtype=np.float64)
    radius = max(int(np.ceil(truncate * sigma_bins)), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def convolve_axis(values: np.ndarray, kernel: np.ndarray, axis: int, mode: str) -> np.ndarray:
    if kernel.size == 1:
        return np.asarray(values, dtype=np.float64).copy()

    moved = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    pad = kernel.size // 2
    pad_width = [(pad, pad)] + [(0, 0)] * (moved.ndim - 1)
    if mode == 'wrap':
        padded = np.pad(moved, pad_width, mode='wrap')
    elif mode == 'reflect':
        padded = np.pad(moved, pad_width, mode='reflect')
    else:
        raise ValueError(f'Unsupported convolution mode: {mode}')

    finite = np.isfinite(padded)
    filled = np.where(finite, padded, 0.0)
    weights = finite.astype(np.float64)

    convolved = np.empty_like(moved, dtype=np.float64)
    for index in range(moved.shape[0]):
        window = filled[index : index + kernel.size]
        weight_window = weights[index : index + kernel.size]
        numerator = np.tensordot(kernel, window, axes=(0, 0))
        denominator = np.tensordot(kernel, weight_window, axes=(0, 0))
        convolved[index] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator > 0.0,
        )
    return np.moveaxis(convolved, 0, axis)


def gaussian_smooth_map(values: np.ndarray, sigma_theta_bins: float, sigma_phi_bins: float) -> np.ndarray:
    smoothed = convolve_axis(values, gaussian_kernel1d(sigma_theta_bins), axis=0, mode='reflect')
    return convolve_axis(smoothed, gaussian_kernel1d(sigma_phi_bins), axis=1, mode='wrap')


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
    battrs: BinAttrs,
    mesh_refine: float,
    nthreads: int,
) -> tuple[np.ndarray, np.ndarray]:
    spin_wattrs = WeightAttrs(spin=(1, 2), reference_only=(True, False))
    ds_mattrs = MeshAttrs(reference_particles, shape_particles, battrs=battrs, refine=mesh_refine)

    ds_spin = count2(reference_particles, shape_particles, battrs=battrs, wattrs=spin_wattrs, mattrs=ds_mattrs, nthreads=nthreads)
    ds_pairs = ds_spin['weight']
    ds_plus = safe_divide(ds_spin['weight_plus'], ds_pairs)
    ds_cross = safe_divide(ds_spin['weight_cross'], ds_pairs)

    rs_plus_sum = np.zeros_like(ds_plus, dtype=np.float64)
    rs_cross_sum = np.zeros_like(ds_cross, dtype=np.float64)
    for random_reference_particles in random_reference_particles_list:
        rs_mattrs = MeshAttrs(random_reference_particles, shape_particles, battrs=battrs, refine=mesh_refine)
        rs_spin = count2(random_reference_particles, shape_particles, battrs=battrs, wattrs=spin_wattrs, mattrs=rs_mattrs, nthreads=nthreads)
        rs_pairs = rs_spin['weight']
        rs_plus_sum += safe_divide(rs_spin['weight_plus'], rs_pairs)
        rs_cross_sum += safe_divide(rs_spin['weight_cross'], rs_pairs)

    rs_plus = rs_plus_sum / len(random_reference_particles_list)
    rs_cross = rs_cross_sum / len(random_reference_particles_list)
    return ds_plus - rs_plus, ds_cross - rs_cross


def collapse_single_pi_bin(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2:
        return values
    if (values.ndim == 3) and (values.shape[-1] == 1):
        return values[..., 0]
    raise ValueError(f'Expected a 2D map or a single pi bin, got shape {values.shape}')


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


def shear_components_to_xy(
    gamma_plus: np.ndarray,
    gamma_cross: np.ndarray,
    phi_centers_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    phi = np.deg2rad(np.asarray(phi_centers_deg, dtype=np.float64))[None, :]
    cos2phi = np.cos(2.0 * phi)
    sin2phi = np.sin(2.0 * phi)
    gamma1 = -gamma_plus * cos2phi + gamma_cross * sin2phi
    gamma2 = -gamma_plus * sin2phi - gamma_cross * cos2phi
    return gamma1, gamma2


def make_shear_segments(
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    gamma_plus: np.ndarray,
    gamma_cross: np.ndarray,
    normalize_lengths: bool = False,
) -> list[np.ndarray]:
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    theta_step = max(gamma_plus.shape[0] // 16, 1)
    phi_step = max(gamma_plus.shape[1] // 18, 1)

    theta_sample = theta_centers[::theta_step]
    phi_sample = phi_centers[::phi_step]
    gamma_plus_sample = np.asarray(gamma_plus[::theta_step, ::phi_step], dtype=np.float64)
    gamma_cross_sample = np.asarray(gamma_cross[::theta_step, ::phi_step], dtype=np.float64)
    gamma1, gamma2 = shear_components_to_xy(gamma_plus_sample, gamma_cross_sample, phi_sample)

    theta_grid, phi_grid = np.meshgrid(theta_sample, np.deg2rad(phi_sample), indexing='ij')
    x = theta_grid * np.cos(phi_grid)
    y = theta_grid * np.sin(phi_grid)

    amplitude = np.hypot(gamma1, gamma2)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(amplitude) & (amplitude > 0.0)
    if not np.any(finite):
        return []

    angle = 0.5 * np.arctan2(gamma2, gamma1)
    if normalize_lengths:
        # Show orientation only, with a fixed stick length across the field.
        half_length = 0.03 * float(theta_edges[-1])
    else:
        max_amplitude = float(np.nanmax(amplitude[finite]))
        if max_amplitude <= 0.0:
            return []
        half_length = 0.03 * float(theta_edges[-1]) * amplitude / max_amplitude

    x0 = x - half_length * np.cos(angle)
    x1 = x + half_length * np.cos(angle)
    y0 = y - half_length * np.sin(angle)
    y1 = y + half_length * np.sin(angle)

    segments = [
        np.array([[x_start, y_start], [x_stop, y_stop]], dtype=np.float64)
        for x_start, y_start, x_stop, y_stop, keep in zip(
            x0.ravel(),
            y0.ravel(),
            x1.ravel(),
            y1.ravel(),
            finite.ravel(),
            strict=False,
        )
        if keep
    ]
    return segments


def plot_smoothed_maps_with_shear_field(
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    gamma_plus_smoothed: np.ndarray,
    gamma_cross_smoothed: np.ndarray,
    output_path: Path,
) -> None:
    theta_grid, phi_grid = np.meshgrid(theta_edges, np.deg2rad(phi_edges), indexing='ij')
    x = theta_grid * np.cos(phi_grid)
    y = theta_grid * np.sin(phi_grid)
    segments = make_shear_segments(
        theta_edges,
        phi_edges,
        gamma_plus_smoothed,
        gamma_cross_smoothed,
        normalize_lengths=True,
    )

    figures = [
        (gamma_plus_smoothed, r'Smoothed $\gamma_+(x, y)$', r'$\gamma_+$'),
        (gamma_cross_smoothed, r'Smoothed $\gamma_\times(x, y)$', r'$\gamma_\times$'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), constrained_layout=True)
    for axis, (values, title, colorbar_label) in zip(axes, figures, strict=False):
        vmax = float(np.nanmax(np.abs(values)))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
        pcm = axis.pcolormesh(x, y, values, shading='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
        if segments:
            axis.add_collection(LineCollection(segments, colors='white', linewidths=2.0, alpha=0.7, zorder=3))
            axis.add_collection(LineCollection(segments, colors='black', linewidths=1.0, alpha=0.9, zorder=4))
        axis.axhline(0.0, color='0.7', linewidth=1.0)
        axis.axvline(0.0, color='0.7', linewidth=1.0)
        axis.set_aspect('equal', adjustable='box')
        axis.set_title(title)
        axis.set_xlabel(r'$x_{+\mathrm{disp}} = \theta \cos \phi$ [deg]')
        axis.set_ylabel(r'$y_{\perp} = \theta \sin \phi$ [deg]')
        fig.colorbar(pcm, ax=axis, label=colorbar_label)

    fig.suptitle('Gaussian-smoothed maps with shear field overlay')
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
    parser.add_argument('--shape-z-col', default='auto', help='Shape-catalog redshift column to use when --pi-max is set, or auto.')
    parser.add_argument('--output-dir', default='examples/output/displacement_shape_phi', help='Directory for outputs.')
    parser.add_argument('--output-prefix', default='desi_lrg_displacement_shape', help='Output file prefix.')
    parser.add_argument('--min-theta', type=float, default=0.05, help='Minimum theta in degrees.')
    parser.add_argument('--max-theta', type=float, default=2.0, help='Maximum theta in degrees.')
    parser.add_argument('--theta-bins', type=int, default=32, help='Number of theta bins.')
    parser.add_argument('--phi-bins', type=int, default=72, help='Number of phi bins over [0, 360) degrees.')
    parser.add_argument('--pi-max', type=float, default=None, help='If set, keep only pairs with |pi| <= this value in h^-1 Mpc using the first-point LOS.')
    parser.add_argument('--max-data-rows', type=int, default=None, help='Optional cap on displacement-catalog rows.')
    parser.add_argument('--max-shape-rows', type=int, default=None, help='Optional cap on shape-catalog rows.')
    parser.add_argument('--max-random-files', type=int, default=4, help='Optional cap on random files.')
    parser.add_argument('--max-random-rows', type=int, default=None, help='Optional cap on rows per random file.')
    parser.add_argument(
        '--smooth-sigma-theta-bins',
        type=float,
        default=1.0,
        help='Gaussian smoothing sigma along theta, in units of theta bins.',
    )
    parser.add_argument(
        '--smooth-sigma-phi-bins',
        type=float,
        default=1.0,
        help='Gaussian smoothing sigma along phi, in units of phi bins.',
    )
    parser.add_argument('--seed', type=int, default=1234, help='Seed for reproducible sub-sampling.')
    parser.add_argument('--mesh-refine', type=float, default=5.0, help='Mesh refinement factor passed to cucount MeshAttrs.')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of GPUs for cucount to use.')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level)

    theta_edges = np.linspace(args.min_theta, args.max_theta, args.theta_bins + 1)
    phi_edges = np.linspace(0.0, 360.0, args.phi_bins + 1)
    pi_edges = None
    distance_to_comoving = None
    distance_label = None
    if args.pi_max is not None:
        pi_edges = np.array([-args.pi_max, args.pi_max], dtype=np.float64)
        battrs = BinAttrs(theta=theta_edges, phi=phi_edges, pi=(pi_edges, DEFAULT_PI_LOS))
        distance_to_comoving, distance_label = build_distance_to_comoving()
    else:
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
    if pi_edges is not None:
        print(f'Pi cut            : [{pi_edges[0]:.1f}, {pi_edges[1]:.1f}] h^-1 Mpc ({DEFAULT_PI_LOS} LOS)')
        print(f'Distance model    : {distance_label}')
    print(f'Mesh refine       : {args.mesh_refine:.2f}')
    print(f'Smoothing sigma   : theta={args.smooth_sigma_theta_bins:.2f} bins, phi={args.smooth_sigma_phi_bins:.2f} bins')
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
        z_col=args.shape_z_col if args.pi_max is not None else None,
    )
    print(f'Loaded {shapes.size:,} shape tracers')

    if args.pi_max is not None:
        if (data.z is None) or (shapes.z is None):
            raise ValueError('The pi cut requires redshift columns for both the displacement and shape catalogs')
        data_distance = distance_to_comoving(data.z)
        shape_distance = distance_to_comoving(shapes.z)
    else:
        data_distance = None
        shape_distance = None

    reference_particles = create_particles(
        data.ra,
        data.dec,
        data.weights,
        distance=data_distance,
        spin_values=(unit_north, unit_east),
    )
    shape_particles = create_particles(
        shapes.ra,
        shapes.dec,
        shapes.weights,
        distance=shape_distance,
        shear=(shapes.e1, shapes.e2),
    )
    random_paths = resolve_random_catalogs(args.randoms_glob, max_files=args.max_random_files)
    random_reference_particles_list = []
    for index, random_path in enumerate(random_paths, start=1):
        randoms = load_desi_catalog(random_path, max_rows=args.max_random_rows, seed=args.seed + index)
        random_distance = None if args.pi_max is None else distance_to_comoving(randoms.z)
        rand_north, rand_east = sample_reference_spin(unit_north, unit_east, randoms.size, seed=args.seed + 10_000 + index)
        random_reference_particles_list.append(
            create_particles(
                randoms.ra,
                randoms.dec,
                randoms.weights,
                distance=random_distance,
                spin_values=(rand_north, rand_east),
            )
        )
    print(f'Loaded {len(random_reference_particles_list)} random catalogs with sampled displacement directions')

    gamma_plus, gamma_cross = compute_shape_maps(
        reference_particles,
        random_reference_particles_list,
        shape_particles,
        battrs=battrs,
        mesh_refine=args.mesh_refine,
        nthreads=args.nthreads,
    )
    gamma_plus = collapse_single_pi_bin(gamma_plus)
    gamma_cross = collapse_single_pi_bin(gamma_cross)
    gamma_plus_smoothed = gaussian_smooth_map(gamma_plus, args.smooth_sigma_theta_bins, args.smooth_sigma_phi_bins)
    gamma_cross_smoothed = gaussian_smooth_map(gamma_cross, args.smooth_sigma_theta_bins, args.smooth_sigma_phi_bins)

    results_path = output_dir / f'{args.output_prefix}_results.npz'
    results_payload = {
        'theta_edges': theta_edges,
        'phi_edges': phi_edges,
        'gamma_plus': gamma_plus,
        'gamma_cross': gamma_cross,
        'gamma_plus_smoothed': gamma_plus_smoothed,
        'gamma_cross_smoothed': gamma_cross_smoothed,
        'smooth_sigma_theta_bins': args.smooth_sigma_theta_bins,
        'smooth_sigma_phi_bins': args.smooth_sigma_phi_bins,
    }
    if pi_edges is not None:
        results_payload['pi_edges'] = pi_edges
        results_payload['pi_los'] = DEFAULT_PI_LOS
    np.savez(results_path, **results_payload)
    print(f'Saved results to {results_path}')

    plus_path = output_dir / f'{args.output_prefix}_gamma_plus_xy.png'
    cross_path = output_dir / f'{args.output_prefix}_gamma_cross_xy.png'
    plus_smoothed_path = output_dir / f'{args.output_prefix}_gamma_plus_xy_smoothed.png'
    cross_smoothed_path = output_dir / f'{args.output_prefix}_gamma_cross_xy_smoothed.png'
    overlay_path = output_dir / f'{args.output_prefix}_smoothed_shear_field_overlay.png'
    plot_single_map(theta_edges, phi_edges, gamma_plus, r'$\gamma_+(x, y)$', r'$\gamma_+$', plus_path)
    plot_single_map(theta_edges, phi_edges, gamma_cross, r'$\gamma_\times(x, y)$', r'$\gamma_\times$', cross_path)
    plot_single_map(
        theta_edges,
        phi_edges,
        gamma_plus_smoothed,
        r'$\gamma_+(x, y)$ Gaussian-smoothed',
        r'$\gamma_+$',
        plus_smoothed_path,
    )
    plot_single_map(
        theta_edges,
        phi_edges,
        gamma_cross_smoothed,
        r'$\gamma_\times(x, y)$ Gaussian-smoothed',
        r'$\gamma_\times$',
        cross_smoothed_path,
    )
    plot_smoothed_maps_with_shear_field(
        theta_edges,
        phi_edges,
        gamma_plus_smoothed,
        gamma_cross_smoothed,
        overlay_path,
    )
    print(f'Saved gamma_+ map to {plus_path}')
    print(f'Saved gamma_x map to {cross_path}')
    print(f'Saved smoothed gamma_+ map to {plus_smoothed_path}')
    print(f'Saved smoothed gamma_x map to {cross_smoothed_path}')
    print(f'Saved smoothed shear-field overlay to {overlay_path}')


if __name__ == '__main__':
    main()
