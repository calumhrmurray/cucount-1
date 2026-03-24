#!/usr/bin/env python3
"""
Measure observed-sky angular correlations for DESI LRG lenses and UNIONS shapes.

Outputs:
- xi_gg(theta) for DESI LRG using DESI random catalogs
- xi_g+(theta) and xi_gx(theta) for DESI x UNIONS
- xi_++(theta), xi_+x(theta), and xi_xx(theta) for UNIONS shape auto-correlations
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from cucount.numpy import BinAttrs, MeshAttrs, WeightAttrs, count2, setup_logging

try:
    import treecorr

    TREECORR_AVAILABLE = True
except ImportError:
    treecorr = None
    TREECORR_AVAILABLE = False

from observed_catalog_tools import (
    CatalogSample,
    DEFAULT_DESI_DATA,
    DEFAULT_DESI_RANDOMS_GLOB,
    DEFAULT_UNIONS_SOURCES,
    create_particles,
    load_desi_catalog,
    load_unions_catalog,
    resolve_random_catalogs,
    safe_divide,
    weighted_auto_norm,
)


def load_random_catalog_samples(
    random_paths: list[str],
    max_random_rows: int | None,
    seed: int,
) -> list[CatalogSample]:
    samples = []
    for index, random_path in enumerate(random_paths, start=1):
        samples.append(load_desi_catalog(random_path, max_rows=max_random_rows, seed=seed + index))
    return samples


def concatenate_catalog_samples(samples: list[CatalogSample]) -> CatalogSample:
    if not samples:
        raise ValueError('At least one catalog sample is required')
    return CatalogSample(
        ra=np.concatenate([sample.ra for sample in samples]),
        dec=np.concatenate([sample.dec for sample in samples]),
        weights=np.concatenate([sample.weights for sample in samples]),
    )


def compute_wgg(
    lenses,
    lens_particles,
    random_samples: list[CatalogSample],
    random_particles_list,
    battrs: BinAttrs,
    mesh_refine: float,
    nthreads: int,
) -> np.ndarray:
    dd_mattrs = MeshAttrs(lens_particles, lens_particles, battrs=battrs, refine=mesh_refine)
    dd_counts = count2(lens_particles, lens_particles, battrs=battrs, mattrs=dd_mattrs, nthreads=nthreads)['weight']
    dd_normed = dd_counts / weighted_auto_norm(lenses.weights)

    dr_normed_sum = np.zeros_like(dd_normed, dtype=np.float64)
    rr_normed_sum = np.zeros_like(dd_normed, dtype=np.float64)
    for randoms, random_particles in zip(random_samples, random_particles_list, strict=False):
        dr_mattrs = MeshAttrs(lens_particles, random_particles, battrs=battrs, refine=mesh_refine)
        rr_mattrs = MeshAttrs(random_particles, random_particles, battrs=battrs, refine=mesh_refine)
        dr_counts = count2(lens_particles, random_particles, battrs=battrs, mattrs=dr_mattrs, nthreads=nthreads)['weight']
        rr_counts = count2(random_particles, random_particles, battrs=battrs, mattrs=rr_mattrs, nthreads=nthreads)['weight']
        dr_normed_sum += dr_counts / (np.sum(lenses.weights) * np.sum(randoms.weights))
        rr_normed_sum += rr_counts / weighted_auto_norm(randoms.weights)

    dr_normed = dr_normed_sum / len(random_samples)
    rr_normed = rr_normed_sum / len(random_samples)
    return safe_divide(dd_normed - 2.0 * dr_normed + rr_normed, rr_normed)


def compute_gplus(
    lens_particles,
    source_particles,
    source_scalar_particles,
    random_particles_list,
    battrs: BinAttrs,
    mesh_refine: float,
    nthreads: int,
) -> tuple[np.ndarray, np.ndarray]:
    spin_weights = WeightAttrs(spin=(0, 2))
    ps_mattrs = MeshAttrs(lens_particles, source_particles, battrs=battrs, refine=mesh_refine)
    ps_spin = count2(lens_particles, source_particles, battrs=battrs, wattrs=spin_weights, mattrs=ps_mattrs, nthreads=nthreads)
    ps_pairs = count2(lens_particles, source_scalar_particles, battrs=battrs, mattrs=ps_mattrs, nthreads=nthreads)['weight']

    ps_t = safe_divide(ps_spin['weight_plus'], ps_pairs)
    ps_x = safe_divide(ps_spin['weight_cross'], ps_pairs)

    rs_t_sum = np.zeros_like(ps_t, dtype=np.float64)
    rs_x_sum = np.zeros_like(ps_x, dtype=np.float64)
    for random_particles in random_particles_list:
        rs_mattrs = MeshAttrs(random_particles, source_particles, battrs=battrs, refine=mesh_refine)
        rs_spin = count2(random_particles, source_particles, battrs=battrs, wattrs=spin_weights, mattrs=rs_mattrs, nthreads=nthreads)
        rs_pairs = count2(random_particles, source_scalar_particles, battrs=battrs, mattrs=rs_mattrs, nthreads=nthreads)['weight']
        rs_t_sum += safe_divide(rs_spin['weight_plus'], rs_pairs)
        rs_x_sum += safe_divide(rs_spin['weight_cross'], rs_pairs)

    rs_t = rs_t_sum / len(random_particles_list)
    rs_x = rs_x_sum / len(random_particles_list)
    return ps_t - rs_t, ps_x - rs_x


def compute_xi_spin_spin(
    source_particles,
    source_scalar_particles,
    battrs: BinAttrs,
    mesh_refine: float,
    nthreads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spin_weights = WeightAttrs(spin=(2, 2))
    ss_mattrs = MeshAttrs(source_particles, source_particles, battrs=battrs, refine=mesh_refine)
    ss_spin = count2(source_particles, source_particles, battrs=battrs, wattrs=spin_weights, mattrs=ss_mattrs, nthreads=nthreads)
    ss_pairs = count2(source_scalar_particles, source_scalar_particles, battrs=battrs, mattrs=ss_mattrs, nthreads=nthreads)['weight']

    xi_plus_plus = safe_divide(ss_spin['weight_plus_plus'], ss_pairs)
    xi_plus_cross = safe_divide(ss_spin['weight_plus_cross'], ss_pairs)
    xi_cross_cross = safe_divide(ss_spin['weight_cross_cross'], ss_pairs)
    return xi_plus_plus, xi_plus_cross, xi_cross_cross


def compute_treecorr_correlations(
    lenses: CatalogSample,
    sources: CatalogSample | None,
    randoms: CatalogSample,
    theta_edges: np.ndarray,
    correlations: list[str],
    bin_slop: float,
    metric: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if not TREECORR_AVAILABLE:
        raise ImportError('TreeCorr is not available')

    min_sep = float(theta_edges[0])
    max_sep = float(theta_edges[-1])
    nbins = theta_edges.size - 1

    results: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    if 'gg' in correlations:
        t0 = time.time()
        lens_cat = treecorr.Catalog(ra=lenses.ra, dec=lenses.dec, w=lenses.weights, ra_units='deg', dec_units='deg')
        rand_cat = treecorr.Catalog(ra=randoms.ra, dec=randoms.dec, w=randoms.weights, ra_units='deg', dec_units='deg')
        dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg', bin_type='Log', bin_slop=bin_slop, metric=metric)
        rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg', bin_type='Log', bin_slop=bin_slop, metric=metric)
        dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg', bin_type='Log', bin_slop=bin_slop, metric=metric)
        dd.process(lens_cat)
        rr.process(rand_cat)
        dr.process(lens_cat, rand_cat)
        xi_gg, _ = dd.calculateXi(rr=rr, dr=dr)
        results['treecorr_xi_gg'] = np.asarray(xi_gg, dtype=np.float64)
        timings['gg'] = time.time() - t0

    if ('gs' in correlations) and (sources is not None):
        t0 = time.time()
        lens_cat = treecorr.Catalog(ra=lenses.ra, dec=lenses.dec, w=lenses.weights, ra_units='deg', dec_units='deg')
        rand_cat = treecorr.Catalog(ra=randoms.ra, dec=randoms.dec, w=randoms.weights, ra_units='deg', dec_units='deg')
        source_cat = treecorr.Catalog(
            ra=sources.ra,
            dec=sources.dec,
            w=sources.weights,
            g1=sources.e1,
            g2=sources.e2,
            ra_units='deg',
            dec_units='deg',
        )
        ng = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg', bin_type='Log', bin_slop=bin_slop, metric=metric)
        rg = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg', bin_type='Log', bin_slop=bin_slop, metric=metric)
        ng.process(lens_cat, source_cat)
        rg.process(rand_cat, source_cat)
        xi_g_plus, xi_g_cross, _ = ng.calculateXi(rg=rg)
        results['treecorr_xi_g_plus'] = np.asarray(xi_g_plus, dtype=np.float64)
        results['treecorr_xi_g_cross'] = np.asarray(xi_g_cross, dtype=np.float64)
        timings['gs'] = time.time() - t0

    if ('ss' in correlations) and (sources is not None):
        t0 = time.time()
        source_cat = treecorr.Catalog(
            ra=sources.ra,
            dec=sources.dec,
            w=sources.weights,
            g1=sources.e1,
            g2=sources.e2,
            ra_units='deg',
            dec_units='deg',
        )
        gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='deg', bin_type='Log', bin_slop=bin_slop, metric=metric)
        gg.process(source_cat)
        results['treecorr_xi_plus_plus'] = np.asarray((gg.xip + gg.xim) / 2.0, dtype=np.float64)
        results['treecorr_xi_cross_cross'] = np.asarray((gg.xip - gg.xim) / 2.0, dtype=np.float64)
        timings['ss'] = time.time() - t0

    return results, timings


def print_timing_table(cucount_timings: dict[str, float], treecorr_timings: dict[str, float] | None = None) -> None:
    print('\n' + '=' * 60)
    print('Performance Comparison')
    print('=' * 60)
    if treecorr_timings:
        print(f"{'Correlation':<15} {'cucount (s)':<15} {'TreeCorr (s)':<15} {'Speedup':<10}")
        print('-' * 60)
    else:
        print(f"{'Correlation':<15} {'cucount (s)':<15}")
        print('-' * 32)

    labels = {
        'gg': r'xi_gg',
        'gs': r'xi_g+/x',
        'ss': r'xi_++/+x/xx',
    }
    for correlation in ['gg', 'gs', 'ss']:
        if correlation not in cucount_timings:
            continue
        cucount_time = cucount_timings[correlation]
        label = labels[correlation]
        if treecorr_timings and (correlation in treecorr_timings):
            treecorr_time = treecorr_timings[correlation]
            speedup = treecorr_time / cucount_time if cucount_time > 0.0 else np.nan
            print(f'{label:<15} {cucount_time:>10.3f}     {treecorr_time:>10.3f}     {speedup:>6.1f}x')
        else:
            print(f'{label:<15} {cucount_time:>10.3f}')
    print('=' * 60)


def plot_results(
    theta_centers: np.ndarray,
    results: dict[str, np.ndarray],
    correlations: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(correlations), figsize=(6 * len(correlations), 5), sharex=True)
    if len(correlations) == 1:
        axes = [axes]

    for axis, correlation in zip(axes, correlations):
        if correlation == 'gg':
            axis.plot(theta_centers, results['xi_gg'], color='C3', marker='o', linewidth=1.8, markersize=4, label=r'$\xi_{gg}$')
            if 'treecorr_xi_gg' in results:
                axis.plot(theta_centers, results['treecorr_xi_gg'], color='C0', marker='s', linewidth=1.4, markersize=4, linestyle='--', label=r'TreeCorr $\xi_{gg}$')
            axis.axhline(0.0, color='0.7', linewidth=1.0, linestyle='--')
            axis.set_xscale('log')
            axis.set_title('DESI LRG Angular Clustering')
            axis.set_xlabel(r'$\theta$ [deg]')
            axis.set_ylabel(r'$\xi_{gg}(\theta)$')
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False)
        elif correlation == 'gs':
            axis.plot(theta_centers, results['xi_g_plus'], color='C0', marker='o', linewidth=1.8, markersize=4, label=r'$\xi_{g+}$')
            axis.plot(theta_centers, results['xi_g_cross'], color='C1', marker='s', linewidth=1.5, markersize=4, label=r'$\xi_{g\times}$')
            if 'treecorr_xi_g_plus' in results:
                axis.plot(theta_centers, results['treecorr_xi_g_plus'], color='C0', marker='^', linewidth=1.2, markersize=4, linestyle='--', alpha=0.8, label=r'TreeCorr $\xi_{g+}$')
            if 'treecorr_xi_g_cross' in results:
                axis.plot(theta_centers, results['treecorr_xi_g_cross'], color='C1', marker='v', linewidth=1.2, markersize=4, linestyle=':', alpha=0.8, label=r'TreeCorr $\xi_{g\times}$')
            axis.axhline(0.0, color='0.7', linewidth=1.0, linestyle='--')
            axis.set_xscale('log')
            axis.set_title('DESI x UNIONS Galaxy-Shear')
            axis.set_xlabel(r'$\theta$ [deg]')
            axis.set_ylabel(r'$\xi_{g\pm}(\theta)$')
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False)
        elif correlation == 'ss':
            axis.plot(theta_centers, results['xi_plus_plus'], color='C2', marker='o', linewidth=1.8, markersize=4, label=r'$\xi_{++}$')
            axis.plot(theta_centers, results['xi_plus_cross'], color='C4', marker='s', linewidth=1.5, markersize=4, label=r'$\xi_{+\times}$')
            axis.plot(theta_centers, results['xi_cross_cross'], color='C5', marker='^', linewidth=1.5, markersize=4, label=r'$\xi_{\times\times}$')
            if 'treecorr_xi_plus_plus' in results:
                axis.plot(theta_centers, results['treecorr_xi_plus_plus'], color='C2', marker='d', linewidth=1.2, markersize=4, linestyle='--', alpha=0.8, label=r'TreeCorr $\xi_{++}$')
            if 'treecorr_xi_cross_cross' in results:
                axis.plot(theta_centers, results['treecorr_xi_cross_cross'], color='C5', marker='>', linewidth=1.2, markersize=4, linestyle=':', alpha=0.8, label=r'TreeCorr $\xi_{\times\times}$')
            axis.axhline(0.0, color='0.7', linewidth=1.0, linestyle='--')
            axis.set_xscale('log')
            axis.set_title('UNIONS Shape Auto-Correlations')
            axis.set_xlabel(r'$\theta$ [deg]')
            axis.set_ylabel(r'$\xi(\theta)$')
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False)

    fig.suptitle('Observed-Sky First-Step Correlations', fontsize=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Measure DESI/UNIONS angular correlations with the current cucount API.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data',
        default=DEFAULT_DESI_DATA,
        required=DEFAULT_DESI_DATA is None,
        help='Path to the DESI observed data catalog. Defaults to $CUCOUNT_DESI_DATA when set.',
    )
    parser.add_argument(
        '--randoms-glob',
        default=DEFAULT_DESI_RANDOMS_GLOB,
        required=DEFAULT_DESI_RANDOMS_GLOB is None,
        help='Glob pattern for DESI random catalogs. Defaults to $CUCOUNT_DESI_RANDOMS_GLOB when set.',
    )
    parser.add_argument(
        '--sources',
        default=DEFAULT_UNIONS_SOURCES,
        required=DEFAULT_UNIONS_SOURCES is None,
        help='Path to the UNIONS source catalog. Defaults to $CUCOUNT_UNIONS_SOURCES when set.',
    )
    parser.add_argument('--source-weight-col', default='auto', help='UNIONS weight column to use, or auto.')
    parser.add_argument('--source-e1-col', default='e1', help='UNIONS e1 column.')
    parser.add_argument('--source-e2-col', default='e2', help='UNIONS e2 column.')
    parser.add_argument('--output-dir', default='examples/output/first_steps', help='Directory where outputs are written.')
    parser.add_argument('--output-prefix', default='desi_unions_observed', help='Prefix for output files.')
    parser.add_argument(
        '--correlations',
        nargs='+',
        default=['gg', 'gs', 'ss'],
        choices=['gg', 'gs', 'ss'],
        help='Subset of correlations to run.',
    )
    parser.add_argument('--min-theta', type=float, default=0.1, help='Minimum theta in degrees.')
    parser.add_argument('--max-theta', type=float, default=1.0, help='Maximum theta in degrees.')
    parser.add_argument('--nbins', type=int, default=12, help='Number of angular bins.')
    parser.add_argument('--max-lenses', type=int, default=None, help='Optional cap on DESI lens rows.')
    parser.add_argument('--max-sources', type=int, default=None, help='Optional cap on UNIONS source rows.')
    parser.add_argument('--max-random-files', type=int, default=4, help='Optional cap on random files for a first-step run.')
    parser.add_argument('--max-random-rows', type=int, default=None, help='Optional cap on rows loaded from each random file.')
    parser.add_argument('--seed', type=int, default=1234, help='Seed used for reproducible sub-sampling.')
    parser.add_argument('--mesh-refine', type=float, default=5.0, help='Mesh refinement factor used for cucount pair counting.')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of GPUs for cucount to use on the node.')
    parser.add_argument('--no-treecorr', action='store_true', help='Skip the optional TreeCorr comparison even if TreeCorr is installed.')
    parser.add_argument('--treecorr-bin-slop', type=float, default=0.01, help='TreeCorr bin_slop parameter.')
    parser.add_argument('--treecorr-metric', default='Euclidean', choices=['Euclidean', 'Arc'], help='TreeCorr metric to use for distance calculations.')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level)

    random_paths = resolve_random_catalogs(args.randoms_glob, max_files=args.max_random_files)
    random_samples = load_random_catalog_samples(random_paths, max_random_rows=args.max_random_rows, seed=args.seed + 200)
    random_particles_list = [create_particles(randoms.ra, randoms.dec, randoms.weights) for randoms in random_samples]
    theta_edges = np.geomspace(args.min_theta, args.max_theta, args.nbins + 1)
    theta_centers = np.sqrt(theta_edges[:-1] * theta_edges[1:])
    battrs = BinAttrs(theta=theta_edges)
    run_treecorr = (not args.no_treecorr) and TREECORR_AVAILABLE

    print('=' * 72)
    print('Observed-sky DESI/UNIONS correlations')
    print('=' * 72)
    print(f'Lenses          : {args.data}')
    print(f'Random catalogs : {len(random_paths)} files')
    print(f'Mesh refine     : {args.mesh_refine:.2f}')
    print(f'TreeCorr        : {"enabled" if run_treecorr else "disabled"}')
    if any(correlation in args.correlations for correlation in ['gs', 'ss']):
        print(f'Sources         : {args.sources}')
    print(f'Correlations    : {", ".join(args.correlations)}')
    print(f'Theta range     : [{args.min_theta:.4f}, {args.max_theta:.2f}] deg')
    print('=' * 72)

    print('\nLoading DESI lenses...')
    lenses = load_desi_catalog(args.data, max_rows=args.max_lenses, seed=args.seed)
    lens_particles = create_particles(lenses.ra, lenses.dec, lenses.weights)
    print(f'  Kept {lenses.size:,} lenses')
    print(f'  Loaded {sum(randoms.size for randoms in random_samples):,} random points across {len(random_samples)} files')

    source_particles = None
    source_scalar_particles = None
    sources = None
    if any(correlation in args.correlations for correlation in ['gs', 'ss']):
        print('\nLoading UNIONS sources...')
        sources = load_unions_catalog(
            args.sources,
            max_rows=args.max_sources,
            seed=args.seed + 100,
            weight_col=args.source_weight_col,
            e1_col=args.source_e1_col,
            e2_col=args.source_e2_col,
        )
        source_particles = create_particles(sources.ra, sources.dec, sources.weights, shear=(sources.e1, sources.e2))
        source_scalar_particles = create_particles(sources.ra, sources.dec, sources.weights)
        print(f'  Kept {sources.size:,} sources')

    results: dict[str, np.ndarray] = {}
    cucount_timings: dict[str, float] = {}

    if 'gg' in args.correlations:
        print('\nComputing xi_gg(theta)...')
        t0 = time.time()
        results['xi_gg'] = compute_wgg(
            lenses,
            lens_particles,
            random_samples,
            random_particles_list,
            battrs,
            mesh_refine=args.mesh_refine,
            nthreads=args.nthreads,
        )
        cucount_timings['gg'] = time.time() - t0
        print(f'  Completed in {cucount_timings["gg"]:.2f} s')

    if 'gs' in args.correlations:
        print('\nComputing xi_g+(theta) and xi_gx(theta)...')
        t0 = time.time()
        results['xi_g_plus'], results['xi_g_cross'] = compute_gplus(
            lens_particles,
            source_particles,
            source_scalar_particles,
            random_particles_list,
            battrs,
            mesh_refine=args.mesh_refine,
            nthreads=args.nthreads,
        )
        cucount_timings['gs'] = time.time() - t0
        print(f'  Completed in {cucount_timings["gs"]:.2f} s')

    if 'ss' in args.correlations:
        print('\nComputing xi_++(theta), xi_+x(theta), and xi_xx(theta)...')
        t0 = time.time()
        results['xi_plus_plus'], results['xi_plus_cross'], results['xi_cross_cross'] = compute_xi_spin_spin(
            source_particles,
            source_scalar_particles,
            battrs,
            mesh_refine=args.mesh_refine,
            nthreads=args.nthreads,
        )
        cucount_timings['ss'] = time.time() - t0
        print(f'  Completed in {cucount_timings["ss"]:.2f} s')

    treecorr_timings = None
    if run_treecorr:
        print('\nComputing TreeCorr comparison...')
        treecorr_randoms = concatenate_catalog_samples(random_samples)
        treecorr_results, treecorr_timings = compute_treecorr_correlations(
            lenses,
            sources,
            treecorr_randoms,
            theta_edges,
            args.correlations,
            bin_slop=args.treecorr_bin_slop,
            metric=args.treecorr_metric,
        )
        results.update(treecorr_results)
    elif (not args.no_treecorr) and (not TREECORR_AVAILABLE):
        print('\nTreeCorr is not installed in this environment; skipping comparison.')

    print_timing_table(cucount_timings, treecorr_timings)

    results_path = output_dir / f'{args.output_prefix}_results.npz'
    np.savez(results_path, theta_edges=theta_edges, theta_centers=theta_centers, **results)
    print(f'\nSaved arrays to {results_path}')

    figure_path = output_dir / f'{args.output_prefix}_summary.png'
    plot_results(theta_centers, results, args.correlations, figure_path)
    print(f'Saved figure to {figure_path}')


if __name__ == '__main__':
    main()
