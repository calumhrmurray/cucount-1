#!/usr/bin/env python3
"""
Measure angular galaxy-shear correlations for several lens samples against UNIONS.

This script computes the observed-sky angular correlations

    xi_g+(theta), xi_gx(theta)

for the default BGS, LRG, ELG, and CMASS position catalogues using a shared
UNIONS shape catalogue. It also estimates delete-one jackknife errors with a
fast analytic hole-subtraction scheme adapted from the IA pipeline and writes a
combined two-panel summary figure styled after the Moriond GS monopole plot.

The output is angular, so the figure uses theta [deg] on the x-axis rather than
the 3D separation s used in the Moriond figure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import os
from pathlib import Path
import re
import time
from typing import Sequence

os.environ.setdefault('MPLCONFIGDIR', str(Path(os.environ.get('TMPDIR', '/tmp')) / 'cucount-mpl'))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.cluster.vq import kmeans2

from cucount.numpy import BinAttrs, Particles, WeightAttrs, count2, setup_logging

from observed_catalog_tools import (
    DEFAULT_UNIONS_SOURCES,
    CatalogSample,
    choose_rows,
    load_unions_catalog,
    safe_divide,
    sky_to_cartesian,
)

DESI_CATALOG_ROOT = Path('/sps/euclid/Users/cmurray/DESI/catalogs')
DEFAULT_UNIONS_SOURCE_PATH = DEFAULT_UNIONS_SOURCES or '/sps/euclid/Users/cmurray/UNIONS/unions_shapepipe_cut_struc_2024_v1.5.3.fits'


@dataclass(frozen=True)
class ColumnFilter:
    column: str
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class SampleSpec:
    label: str
    color: str
    position_paths: tuple[str, ...]
    random_globs: tuple[str, ...]
    position_weight_cols: tuple[str, ...] | None = None
    position_weight_expression: str | None = None
    random_weight_cols: tuple[str, ...] | None = None
    random_weight_expression: str | None = None
    position_filters: tuple[ColumnFilter, ...] = ()
    random_filters: tuple[ColumnFilter, ...] = ()


@dataclass
class AngularDataset:
    ra: np.ndarray
    dec: np.ndarray
    weights: np.ndarray
    cartesian: np.ndarray
    spin_values: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.ra.size)

    def subset(self, mask: np.ndarray) -> 'AngularDataset':
        return AngularDataset(
            ra=self.ra[mask],
            dec=self.dec[mask],
            weights=self.weights[mask],
            cartesian=self.cartesian[mask],
            spin_values=None if self.spin_values is None else self.spin_values[mask],
        )

    def to_particles(self, include_spin: bool = True) -> Particles:
        kwargs = {}
        if include_spin and (self.spin_values is not None):
            kwargs['spin_values'] = self.spin_values
        return Particles(self.cartesian, weights=self.weights, **kwargs)


DEFAULT_SAMPLE_SPECS: dict[str, SampleSpec] = {
    'BGS': SampleSpec(
        label='BGS',
        color='#1f77b4',
        position_paths=(
            str(DESI_CATALOG_ROOT / 'BGS_ANY_NGC_clustering.dat.fits'),
            str(DESI_CATALOG_ROOT / 'BGS_ANY_SGC_clustering.dat.fits'),
        ),
        random_globs=(
            str(DESI_CATALOG_ROOT / 'BGS_ANY_NGC_*_clustering.ran.fits'),
            str(DESI_CATALOG_ROOT / 'BGS_ANY_SGC_*_clustering.ran.fits'),
        ),
        position_weight_cols=('WEIGHT', 'WEIGHT_FKP'),
        random_weight_cols=('WEIGHT', 'WEIGHT_FKP'),
    ),
    'LRG': SampleSpec(
        label='LRG',
        color='#2ca02c',
        position_paths=(
            str(DESI_CATALOG_ROOT / 'LRG_NGC_clustering.dat.fits'),
            str(DESI_CATALOG_ROOT / 'LRG_SGC_clustering.dat.fits'),
        ),
        random_globs=(
            str(DESI_CATALOG_ROOT / 'LRG_NGC_*_clustering.ran.fits'),
            str(DESI_CATALOG_ROOT / 'LRG_SGC_*_clustering.ran.fits'),
        ),
        position_weight_cols=('WEIGHT', 'WEIGHT_FKP'),
        random_weight_cols=('WEIGHT', 'WEIGHT_FKP'),
    ),
    'CMASS': SampleSpec(
        label='CMASS',
        color='#d62728',
        position_paths=(
            str(DESI_CATALOG_ROOT / 'galaxy_DR12v5_CMASS_North.fits'),
            str(DESI_CATALOG_ROOT / 'galaxy_DR12v5_CMASS_South.fits'),
        ),
        random_globs=(
            str(DESI_CATALOG_ROOT / 'random*_DR12v5_CMASS_North.fits'),
            str(DESI_CATALOG_ROOT / 'random*_DR12v5_CMASS_South.fits'),
        ),
        position_weight_expression='WEIGHT_FKP * WEIGHT_SYSTOT * (WEIGHT_NOZ + WEIGHT_CP - 1.0)',
        random_weight_expression='WEIGHT_FKP',
        position_filters=(ColumnFilter('Z', 0.43, 0.70),),
        random_filters=(ColumnFilter('Z', 0.43, 0.70),),
    ),
    'ELG': SampleSpec(
        label='ELG',
        color='#9467bd',
        position_paths=(
            str(DESI_CATALOG_ROOT / 'ELG_LOPnotqso_NGC_clustering.dat.fits'),
            str(DESI_CATALOG_ROOT / 'ELG_LOPnotqso_SGC_clustering.dat.fits'),
        ),
        random_globs=(
            str(DESI_CATALOG_ROOT / 'ELG_LOPnotqso_NGC_*_clustering.ran.fits'),
            str(DESI_CATALOG_ROOT / 'ELG_LOPnotqso_SGC_*_clustering.ran.fits'),
        ),
        position_weight_cols=('WEIGHT',),
        random_weight_cols=('WEIGHT',),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--sources',
        default=DEFAULT_UNIONS_SOURCE_PATH,
        required=DEFAULT_UNIONS_SOURCE_PATH is None,
        help='Path to the shared UNIONS source catalogue.',
    )
    parser.add_argument('--source-weight-col', default='auto', help='Weight column for the UNIONS catalogue.')
    parser.add_argument('--source-e1-col', default='e1', help='UNIONS e1 column.')
    parser.add_argument('--source-e2-col', default='e2', help='UNIONS e2 column.')
    parser.add_argument(
        '--source-z-col',
        default=None,
        help='Optional UNIONS source-redshift column. Use auto to auto-detect when applying source-z cuts.',
    )
    parser.add_argument('--source-z-min', type=float, default=None, help='Optional minimum source redshift.')
    parser.add_argument('--source-z-max', type=float, default=None, help='Optional maximum source redshift.')
    parser.add_argument(
        '--samples',
        nargs='+',
        default=['CMASS', 'ELG'],
        choices=list(DEFAULT_SAMPLE_SPECS),
        help='Lens samples to measure.',
    )
    parser.add_argument('--output-dir', default='examples/output/unions_galaxy_shear_jackknife', help='Directory for outputs.')
    parser.add_argument('--output-prefix', default='unions_galaxy_shear_theta', help='Prefix for combined output figures.')
    parser.add_argument('--min-theta', type=float, default=0.01, help='Minimum angular separation in degrees.')
    parser.add_argument('--max-theta', type=float, default=2.0, help='Maximum angular separation in degrees.')
    parser.add_argument('--nbins', type=int, default=18, help='Number of logarithmic theta bins.')
    parser.add_argument('--n-regions', type=int, default=16, help='Number of jackknife regions.')
    parser.add_argument('--kmeans-max-rows', type=int, default=500_000, help='Maximum source rows used to fit the k-means centres.')
    parser.add_argument('--kmeans-n-init', type=int, default=5, help='Number of k-means restarts.')
    parser.add_argument('--assignment-chunk-size', type=int, default=1_000_000, help='Chunk size used when assigning objects to centres.')
    parser.add_argument(
        '--random-index',
        type=int,
        default=0,
        help='Random tile index to select from the default random pools. Use a negative value to load all matching randoms.',
    )
    parser.add_argument('--max-sources', type=int, default=None, help='Optional cap on UNIONS rows.')
    parser.add_argument('--max-lens-rows-per-file', type=int, default=None, help='Optional cap on rows loaded from each lens file.')
    parser.add_argument('--max-random-rows-per-file', type=int, default=None, help='Optional cap on rows loaded from each random file.')
    parser.add_argument('--seed', type=int, default=1234, help='Seed used for catalogue sub-sampling and k-means.')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of GPUs for cucount to use on the node.')
    parser.add_argument('--usetex', action='store_true', help='Enable TeX rendering in the final figure.')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'])
    return parser.parse_args()


def apply_plot_style(usetex: bool) -> None:
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 20,
        'axes.titlesize': 22,
        'legend.fontsize': 14,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'text.usetex': usetex,
        'font.family': 'serif',
        'axes.grid': True,
        'grid.alpha': 0.2,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'figure.dpi': 150,
    })


def build_scalar_dataset(ra: np.ndarray, dec: np.ndarray, weights: np.ndarray) -> AngularDataset:
    return AngularDataset(
        ra=np.asarray(ra, dtype=np.float64),
        dec=np.asarray(dec, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        cartesian=np.asarray(sky_to_cartesian(ra, dec), dtype=np.float64),
    )


def build_shape_dataset(sample: CatalogSample) -> AngularDataset:
    spin_values = -np.column_stack([
        np.asarray(sample.e1, dtype=np.float64),
        np.asarray(sample.e2, dtype=np.float64),
    ])
    return AngularDataset(
        ra=np.asarray(sample.ra, dtype=np.float64),
        dec=np.asarray(sample.dec, dtype=np.float64),
        weights=np.asarray(sample.weights, dtype=np.float64),
        cartesian=np.asarray(sky_to_cartesian(sample.ra, sample.dec), dtype=np.float64),
        spin_values=spin_values,
    )


def _resolve_column_name(columns: Sequence[str], primary: str, *fallbacks: str) -> str:
    for name in (primary, *fallbacks):
        if name and (name in columns):
            return name
    raise KeyError(f'Could not find any of {primary!r}, {fallbacks!r} in the FITS columns.')


_IDENTIFIER_RE = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]*\b')


def _evaluate_weight_expression(
    expression: str,
    data,
    row,
    columns: Sequence[str],
) -> np.ndarray:
    namespace: dict[str, object] = {'np': np}
    for token in sorted(set(_IDENTIFIER_RE.findall(expression))):
        if token == 'np':
            continue
        if token not in columns:
            continue
        namespace[token] = np.asarray(data[token][row], dtype=np.float64)
    try:
        weights = eval(expression, {'__builtins__': {}}, namespace)
    except Exception as exc:
        raise ValueError(f'Failed to evaluate weight expression {expression!r}') from exc
    return np.asarray(weights, dtype=np.float64)


def _load_scalar_catalog_group(
    paths: Sequence[str],
    *,
    weight_cols: Sequence[str] | None,
    weight_expression: str | None,
    filters: Sequence[ColumnFilter],
    max_rows_per_file: int | None,
    seed: int,
    label: str,
) -> AngularDataset:
    ra_parts: list[np.ndarray] = []
    dec_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []

    for file_index, path in enumerate(paths):
        with fits.open(path, memmap=True) as hdul:
            data = hdul[1].data
            columns = hdul[1].columns.names
            row_indices = choose_rows(len(data), max_rows=max_rows_per_file, seed=seed + file_index)
            row = slice(None) if row_indices is None else row_indices

            ra_col = _resolve_column_name(columns, 'RA')
            dec_col = _resolve_column_name(columns, 'DEC', 'Dec')
            ra = np.asarray(data[ra_col][row], dtype=np.float64)
            dec = np.asarray(data[dec_col][row], dtype=np.float64)

            if weight_expression is not None:
                weights = _evaluate_weight_expression(weight_expression, data, row, columns)
            elif weight_cols:
                weights = np.ones_like(ra, dtype=np.float64)
                for name in weight_cols:
                    if name not in columns:
                        raise KeyError(f'Column {name!r} was not found in {path}')
                    weights *= np.asarray(data[name][row], dtype=np.float64)
            else:
                weights = np.ones_like(ra, dtype=np.float64)

            mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(weights) & (weights > 0.0)
            for filter_spec in filters:
                if filter_spec.column not in columns:
                    raise KeyError(f'Filter column {filter_spec.column!r} was not found in {path}')
                values = np.asarray(data[filter_spec.column][row], dtype=np.float64)
                mask &= np.isfinite(values)
                if filter_spec.min_value is not None:
                    mask &= values >= filter_spec.min_value
                if filter_spec.max_value is not None:
                    mask &= values <= filter_spec.max_value

            ra_parts.append(ra[mask])
            dec_parts.append(dec[mask])
            weight_parts.append(weights[mask])

    if not ra_parts:
        raise ValueError(f'No rows were loaded for {label}')

    return build_scalar_dataset(
        np.concatenate(ra_parts),
        np.concatenate(dec_parts),
        np.concatenate(weight_parts),
    )


def expand_random_paths(patterns: Sequence[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise FileNotFoundError(f'No random catalogues matched {pattern!r}')
        paths.extend(matched)
    return paths


def select_random_subset(paths: Sequence[str], target_index: int, sample_name: str) -> list[str]:
    selected: list[str] = []
    for path in paths:
        name = Path(path).name
        patterns = (
            r'_(\d+)_clustering',
            r'^random(\d+)(?:[_\.]|$)',
        )
        for pattern in patterns:
            match = re.search(pattern, name)
            if match and (int(match.group(1)) == target_index):
                selected.append(path)
                break
    if not selected:
        token = f'_{target_index}_'
        selected = [path for path in paths if token in Path(path).name]
    if not selected:
        prefix = f'random{target_index}_'
        selected = [path for path in paths if Path(path).name.startswith(prefix)]
    if not selected:
        raise ValueError(f'Could not find random files matching index {target_index} for sample {sample_name!r}.')
    return selected


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0.0):
        raise ValueError('Encountered zero-length vectors while normalizing k-means centres.')
    return values / norms[:, None]


def build_jackknife_centres(
    cartesian: np.ndarray,
    n_regions: int,
    *,
    seed: int,
    max_rows: int | None,
    n_init: int,
) -> np.ndarray:
    if cartesian.shape[0] < n_regions:
        raise ValueError(f'Requested {n_regions} regions, but only {cartesian.shape[0]} source objects are available.')

    if max_rows is not None and cartesian.shape[0] > max_rows:
        row_indices = choose_rows(cartesian.shape[0], max_rows=max_rows, seed=seed)
        fit_points = cartesian[row_indices]
    else:
        fit_points = cartesian

    best_centres: np.ndarray | None = None
    best_inertia = np.inf

    for init_index in range(n_init):
        rng = np.random.default_rng(seed + init_index)
        centres, _ = kmeans2(
            fit_points,
            n_regions,
            iter=50,
            minit='points',
            missing='raise',
            seed=rng,
        )
        centres = normalize_rows(np.asarray(centres, dtype=np.float64))
        labels = np.argmax(fit_points @ centres.T, axis=1)
        inertia = float(np.sum(1.0 - np.sum(fit_points * centres[labels], axis=1)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centres = centres

    if best_centres is None:
        raise RuntimeError('Failed to fit jackknife centres.')
    return best_centres


def assign_to_centres(cartesian: np.ndarray, centres: np.ndarray, chunk_size: int) -> np.ndarray:
    assignments = np.empty(cartesian.shape[0], dtype=np.int32)
    for start in range(0, cartesian.shape[0], chunk_size):
        stop = min(start + chunk_size, cartesian.shape[0])
        assignments[start:stop] = np.argmax(cartesian[start:stop] @ centres.T, axis=1)
    return assignments


def region_counts(assignments: np.ndarray, n_regions: int) -> np.ndarray:
    return np.bincount(assignments.astype(np.int64), minlength=n_regions)


def print_region_summary(label: str, assignments: np.ndarray, n_regions: int) -> None:
    counts = region_counts(assignments, n_regions)
    print(
        f'  {label:<8}: min={int(np.min(counts)):,}  '
        f'median={int(np.median(counts)):,}  max={int(np.max(counts)):,}'
    )


def extract_scalar_counts(counts, n_bins: int) -> np.ndarray:
    if isinstance(counts, dict):
        values = counts['weight'] if 'weight' in counts else next(iter(counts.values()))
    else:
        values = counts
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size != n_bins:
        raise ValueError(f'Expected {n_bins} scalar bins, got {values.size}')
    return values


def extract_plus_cross_counts(counts, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    plus = cross = None
    if isinstance(counts, dict):
        if 'weight_plus' in counts:
            plus = counts['weight_plus']
        elif 'plus' in counts:
            plus = counts['plus']
        else:
            plus = next(iter(counts.values()))

        if 'weight_cross' in counts:
            cross = counts['weight_cross']
        elif 'cross' in counts:
            cross = counts['cross']
        else:
            cross = np.zeros_like(np.asarray(plus, dtype=np.float64))
    else:
        values = np.asarray(counts, dtype=np.float64)
        if values.ndim == 2 and values.shape[0] == 2:
            plus, cross = values[0], values[1]
        elif values.ndim == 2 and values.shape[1] == 2:
            plus, cross = values[:, 0], values[:, 1]
        else:
            plus = values
            cross = np.zeros_like(values)

    plus = np.asarray(plus, dtype=np.float64).reshape(-1)
    cross = np.asarray(cross, dtype=np.float64).reshape(-1)
    if plus.size != n_bins:
        raise ValueError(f'Expected {n_bins} plus bins, got {plus.size}')
    if cross.size != n_bins:
        raise ValueError(f'Expected {n_bins} cross bins, got {cross.size}')
    return plus, cross


def count_scalar_pairs(dataset_a: AngularDataset, dataset_b: AngularDataset, battrs: BinAttrs, n_bins: int, nthreads: int) -> np.ndarray:
    if len(dataset_a) == 0 or len(dataset_b) == 0:
        return np.zeros(n_bins, dtype=np.float64)
    counts = count2(
        dataset_a.to_particles(include_spin=False),
        dataset_b.to_particles(include_spin=False),
        battrs=battrs,
        nthreads=nthreads,
    )
    return extract_scalar_counts(counts, n_bins)


def count_spin02_pairs(dataset_a: AngularDataset, dataset_b: AngularDataset, battrs: BinAttrs, n_bins: int, nthreads: int) -> tuple[np.ndarray, np.ndarray]:
    if len(dataset_a) == 0 or len(dataset_b) == 0:
        zeros = np.zeros(n_bins, dtype=np.float64)
        return zeros, zeros
    counts = count2(
        dataset_a.to_particles(include_spin=False),
        dataset_b.to_particles(include_spin=True),
        battrs=battrs,
        wattrs=WeightAttrs(spin=(0, 2)),
        nthreads=nthreads,
    )
    return extract_plus_cross_counts(counts, n_bins)


def compute_jackknife_covariance(samples: np.ndarray) -> np.ndarray:
    mean = np.mean(samples, axis=0)
    diff = samples - mean
    n_samples = samples.shape[0]
    return (n_samples - 1) / n_samples * (diff.T @ diff)


def jackknife_stats(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(samples, axis=0)
    covariance = compute_jackknife_covariance(samples)
    return mean, np.sqrt(np.diag(covariance))


def measure_gs_fast_jackknife(
    lenses: AngularDataset,
    sources: AngularDataset,
    randoms: AngularDataset,
    lens_assign: np.ndarray,
    source_assign: np.ndarray,
    random_assign: np.ndarray,
    *,
    battrs: BinAttrs,
    n_bins: int,
    n_regions: int,
    nthreads: int,
) -> dict[str, np.ndarray]:
    ah_bo_plus = np.zeros((n_regions, n_bins), dtype=np.float64)
    ah_bo_cross = np.zeros((n_regions, n_bins), dtype=np.float64)
    ah_bo_count = np.zeros((n_regions, n_bins), dtype=np.float64)

    ao_bh_plus = np.zeros((n_regions, n_bins), dtype=np.float64)
    ao_bh_cross = np.zeros((n_regions, n_bins), dtype=np.float64)
    ao_bh_count = np.zeros((n_regions, n_bins), dtype=np.float64)

    ah_bh_plus = np.zeros((n_regions, n_bins), dtype=np.float64)
    ah_bh_cross = np.zeros((n_regions, n_bins), dtype=np.float64)
    ah_bh_count = np.zeros((n_regions, n_bins), dtype=np.float64)

    rh_bo_plus = np.zeros((n_regions, n_bins), dtype=np.float64)
    rh_bo_cross = np.zeros((n_regions, n_bins), dtype=np.float64)
    rh_bo_count = np.zeros((n_regions, n_bins), dtype=np.float64)

    ro_bh_plus = np.zeros((n_regions, n_bins), dtype=np.float64)
    ro_bh_cross = np.zeros((n_regions, n_bins), dtype=np.float64)
    ro_bh_count = np.zeros((n_regions, n_bins), dtype=np.float64)

    rh_bh_plus = np.zeros((n_regions, n_bins), dtype=np.float64)
    rh_bh_cross = np.zeros((n_regions, n_bins), dtype=np.float64)
    rh_bh_count = np.zeros((n_regions, n_bins), dtype=np.float64)

    for region in range(n_regions):
        lens_hole = lenses.subset(lens_assign == region)
        lens_keep = lenses.subset(lens_assign != region)
        source_hole = sources.subset(source_assign == region)
        source_keep = sources.subset(source_assign != region)
        random_hole = randoms.subset(random_assign == region)
        random_keep = randoms.subset(random_assign != region)

        ah_bo_plus[region], ah_bo_cross[region] = count_spin02_pairs(lens_hole, source_keep, battrs, n_bins, nthreads)
        ao_bh_plus[region], ao_bh_cross[region] = count_spin02_pairs(lens_keep, source_hole, battrs, n_bins, nthreads)
        ah_bh_plus[region], ah_bh_cross[region] = count_spin02_pairs(lens_hole, source_hole, battrs, n_bins, nthreads)

        ah_bo_count[region] = count_scalar_pairs(lens_hole, source_keep, battrs, n_bins, nthreads)
        ao_bh_count[region] = count_scalar_pairs(lens_keep, source_hole, battrs, n_bins, nthreads)
        ah_bh_count[region] = count_scalar_pairs(lens_hole, source_hole, battrs, n_bins, nthreads)

        rh_bo_plus[region], rh_bo_cross[region] = count_spin02_pairs(random_hole, source_keep, battrs, n_bins, nthreads)
        ro_bh_plus[region], ro_bh_cross[region] = count_spin02_pairs(random_keep, source_hole, battrs, n_bins, nthreads)
        rh_bh_plus[region], rh_bh_cross[region] = count_spin02_pairs(random_hole, source_hole, battrs, n_bins, nthreads)

        rh_bo_count[region] = count_scalar_pairs(random_hole, source_keep, battrs, n_bins, nthreads)
        ro_bh_count[region] = count_scalar_pairs(random_keep, source_hole, battrs, n_bins, nthreads)
        rh_bh_count[region] = count_scalar_pairs(random_hole, source_hole, battrs, n_bins, nthreads)

    ps_plus_total, ps_cross_total = count_spin02_pairs(lenses, sources, battrs, n_bins, nthreads)
    ps_count_total = count_scalar_pairs(lenses, sources, battrs, n_bins, nthreads)

    rs_plus_total, rs_cross_total = count_spin02_pairs(randoms, sources, battrs, n_bins, nthreads)
    rs_count_total = count_scalar_pairs(randoms, sources, battrs, n_bins, nthreads)

    xi_plus = safe_divide(ps_plus_total, ps_count_total) - safe_divide(rs_plus_total, rs_count_total)
    xi_cross = safe_divide(ps_cross_total, ps_count_total) - safe_divide(rs_cross_total, rs_count_total)

    jk_plus = np.zeros((n_regions, n_bins), dtype=np.float64)
    jk_cross = np.zeros((n_regions, n_bins), dtype=np.float64)

    for region in range(n_regions):
        ps_count_jk = ps_count_total - ah_bo_count[region] - ao_bh_count[region] - ah_bh_count[region]
        rs_count_jk = rs_count_total - rh_bo_count[region] - ro_bh_count[region] - rh_bh_count[region]

        ps_plus_jk = safe_divide(
            ps_plus_total - ah_bo_plus[region] - ao_bh_plus[region] - ah_bh_plus[region],
            ps_count_jk,
        )
        ps_cross_jk = safe_divide(
            ps_cross_total - ah_bo_cross[region] - ao_bh_cross[region] - ah_bh_cross[region],
            ps_count_jk,
        )
        rs_plus_jk = safe_divide(
            rs_plus_total - rh_bo_plus[region] - ro_bh_plus[region] - rh_bh_plus[region],
            rs_count_jk,
        )
        rs_cross_jk = safe_divide(
            rs_cross_total - rh_bo_cross[region] - ro_bh_cross[region] - rh_bh_cross[region],
            rs_count_jk,
        )

        jk_plus[region] = ps_plus_jk - rs_plus_jk
        jk_cross[region] = ps_cross_jk - rs_cross_jk

    jk_mean_plus, jk_std_plus = jackknife_stats(jk_plus)
    jk_mean_cross, jk_std_cross = jackknife_stats(jk_cross)

    return {
        'xi_g_plus': xi_plus,
        'xi_g_cross': xi_cross,
        'jk_xi_g_plus': jk_plus,
        'jk_xi_g_cross': jk_cross,
        'jk_mean_g_plus': jk_mean_plus,
        'jk_mean_g_cross': jk_mean_cross,
        'jk_std_g_plus': jk_std_plus,
        'jk_std_g_cross': jk_std_cross,
        'jk_cov_g_plus': compute_jackknife_covariance(jk_plus),
        'jk_cov_g_cross': compute_jackknife_covariance(jk_cross),
    }


def load_sources(args: argparse.Namespace) -> AngularDataset:
    z_col = args.source_z_col
    if (args.source_z_min is not None or args.source_z_max is not None) and (z_col is None):
        z_col = 'auto'

    sources = load_unions_catalog(
        args.sources,
        max_rows=args.max_sources,
        seed=args.seed + 100,
        weight_col=args.source_weight_col,
        e1_col=args.source_e1_col,
        e2_col=args.source_e2_col,
        z_col=z_col,
    )

    if args.source_z_min is not None or args.source_z_max is not None:
        if sources.z is None:
            raise ValueError('Source-z cuts were requested, but no source redshift column could be loaded.')
        mask = np.ones(sources.size, dtype=bool)
        if args.source_z_min is not None:
            mask &= sources.z >= args.source_z_min
        if args.source_z_max is not None:
            mask &= sources.z <= args.source_z_max
        sources = CatalogSample(
            ra=sources.ra[mask],
            dec=sources.dec[mask],
            z=sources.z[mask],
            e1=sources.e1[mask],
            e2=sources.e2[mask],
            weights=sources.weights[mask],
        )

    return build_shape_dataset(sources)


def load_sample_datasets(args: argparse.Namespace, sample_name: str) -> tuple[AngularDataset, AngularDataset]:
    spec = DEFAULT_SAMPLE_SPECS[sample_name]
    random_paths = expand_random_paths(spec.random_globs)
    if args.random_index >= 0:
        random_paths = select_random_subset(random_paths, args.random_index, sample_name)

    lenses = _load_scalar_catalog_group(
        spec.position_paths,
        weight_cols=spec.position_weight_cols,
        weight_expression=spec.position_weight_expression,
        filters=spec.position_filters,
        max_rows_per_file=args.max_lens_rows_per_file,
        seed=args.seed + 1_000,
        label=f'{sample_name} lenses',
    )
    randoms = _load_scalar_catalog_group(
        random_paths,
        weight_cols=spec.random_weight_cols,
        weight_expression=spec.random_weight_expression,
        filters=spec.random_filters,
        max_rows_per_file=args.max_random_rows_per_file,
        seed=args.seed + 2_000,
        label=f'{sample_name} randoms',
    )
    return lenses, randoms


def save_sample_results(
    output_dir: Path,
    sample_name: str,
    theta_edges: np.ndarray,
    theta_centers: np.ndarray,
    centres: np.ndarray,
    source_counts: np.ndarray,
    lens_counts: np.ndarray,
    random_counts: np.ndarray,
    results: dict[str, np.ndarray],
) -> Path:
    path = output_dir / f'{sample_name.lower()}_galaxy_shear_theta.npz'
    np.savez(
        path,
        theta_edges=theta_edges,
        theta_centers=theta_centers,
        jackknife_centres=centres,
        source_region_counts=source_counts,
        lens_region_counts=lens_counts,
        random_region_counts=random_counts,
        **results,
    )
    return path


def _panel_limits(curves: Sequence[np.ndarray], symmetric: bool = False) -> tuple[float, float] | None:
    if not curves:
        return None
    values = np.concatenate([curve[np.isfinite(curve)] for curve in curves if np.any(np.isfinite(curve))])
    if values.size == 0:
        return None
    if symmetric:
        limit = 1.15 * np.max(np.abs(values))
        if limit == 0.0:
            limit = 1.0
        return -limit, limit
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin == vmax:
        delta = 0.25 * (abs(vmin) if vmin != 0.0 else 1.0)
        return vmin - delta, vmax + delta
    pad = 0.12 * (vmax - vmin)
    return vmin - pad, vmax + pad


def plot_combined_results(
    theta_centers: np.ndarray,
    sample_order: Sequence[str],
    sample_results: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    output_prefix: str,
    *,
    usetex: bool,
) -> tuple[Path, Path]:
    apply_plot_style(usetex)

    fig, (ax_plus, ax_cross) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05},
    )

    plus_curves: list[np.ndarray] = []
    cross_curves: list[np.ndarray] = []

    for sample_name in sample_order:
        spec = DEFAULT_SAMPLE_SPECS[sample_name]
        payload = sample_results[sample_name]

        plus_mean = payload['jk_mean_g_plus']
        plus_std = payload['jk_std_g_plus']
        cross_mean = payload['jk_mean_g_cross']
        cross_std = payload['jk_std_g_cross']

        plus_curves.extend([plus_mean - plus_std, plus_mean + plus_std])
        cross_curves.extend([cross_mean - cross_std, cross_mean + cross_std])

        ax_plus.plot(theta_centers, plus_mean, color=spec.color, linewidth=2.0, label=spec.label)
        ax_plus.fill_between(theta_centers, plus_mean - plus_std, plus_mean + plus_std, color=spec.color, alpha=0.2)

        ax_cross.plot(theta_centers, cross_mean, color=spec.color, linewidth=2.0)
        ax_cross.fill_between(theta_centers, cross_mean - cross_std, cross_mean + cross_std, color=spec.color, alpha=0.2)

    ax_plus.set_xscale('log')
    ax_plus.set_xlim(theta_centers[0], theta_centers[-1])
    plus_limits = _panel_limits(plus_curves, symmetric=False)
    if plus_limits is not None:
        ax_plus.set_ylim(*plus_limits)
    ax_plus.set_ylabel(r'$\xi_{g+}(\theta)$')
    ax_plus.legend(loc='best', frameon=True)
    ax_plus.axhline(0.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

    ax_cross.set_xscale('log')
    ax_cross.set_xlim(theta_centers[0], theta_centers[-1])
    cross_limits = _panel_limits(cross_curves, symmetric=True)
    if cross_limits is not None:
        ax_cross.set_ylim(*cross_limits)
    ax_cross.set_xlabel(r'$\theta\ [\mathrm{deg}]$')
    ax_cross.set_ylabel(r'$\xi_{g\times}(\theta)$')
    ax_cross.axhline(0.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

    fig.suptitle(r'Galaxy-shear angular correlations ($\xi_{g+}$ and $\xi_{g\times}$)', y=1.01)
    fig.tight_layout()

    pdf_path = output_dir / f'{output_prefix}.pdf'
    png_path = output_dir / f'{output_prefix}.png'
    fig.savefig(pdf_path, bbox_inches='tight', dpi=150)
    fig.savefig(png_path, bbox_inches='tight', dpi=180)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level)

    theta_edges = np.geomspace(args.min_theta, args.max_theta, args.nbins + 1)
    theta_centers = np.sqrt(theta_edges[:-1] * theta_edges[1:])
    n_bins = theta_edges.size - 1
    battrs = BinAttrs(theta=theta_edges)

    print('=' * 72)
    print('UNIONS angular galaxy-shear measurements')
    print('=' * 72)
    print(f'Sources         : {args.sources}')
    print(f'Samples         : {", ".join(args.samples)}')
    print(f'Theta range     : [{args.min_theta:.4f}, {args.max_theta:.4f}] deg')
    print(f'Theta bins      : {args.nbins}')
    print(f'JK regions      : {args.n_regions}')
    print(f'Random index    : {"all" if args.random_index < 0 else args.random_index}')
    print('=' * 72)

    print('\nLoading UNIONS sources...')
    t0 = time.time()
    sources = load_sources(args)
    print(f'  Kept {len(sources):,} source objects in {time.time() - t0:.2f} s')

    print('\nBuilding jackknife centres on the source catalogue...')
    t0 = time.time()
    centres = build_jackknife_centres(
        sources.cartesian,
        args.n_regions,
        seed=args.seed,
        max_rows=args.kmeans_max_rows,
        n_init=args.kmeans_n_init,
    )
    source_assign = assign_to_centres(sources.cartesian, centres, args.assignment_chunk_size)
    print(f'  Fitted {args.n_regions} centres in {time.time() - t0:.2f} s')
    print_region_summary('sources', source_assign, args.n_regions)
    source_counts = region_counts(source_assign, args.n_regions)

    sample_results: dict[str, dict[str, np.ndarray]] = {}

    for sample_name in args.samples:
        spec = DEFAULT_SAMPLE_SPECS[sample_name]
        print('\n' + '-' * 72)
        print(f'[{sample_name}] Loading lenses and randoms')
        print('-' * 72)

        t0 = time.time()
        lenses, randoms = load_sample_datasets(args, sample_name)
        print(f'  Lenses         : {len(lenses):,}')
        print(f'  Randoms        : {len(randoms):,}')
        print(f'  Catalogue load : {time.time() - t0:.2f} s')

        t0 = time.time()
        lens_assign = assign_to_centres(lenses.cartesian, centres, args.assignment_chunk_size)
        random_assign = assign_to_centres(randoms.cartesian, centres, args.assignment_chunk_size)
        print(f'  Assigned to centres in {time.time() - t0:.2f} s')
        print_region_summary('lenses', lens_assign, args.n_regions)
        print_region_summary('randoms', random_assign, args.n_regions)

        print(f'\n[{sample_name}] Measuring xi_g+(theta) and xi_gx(theta)...')
        t0 = time.time()
        results = measure_gs_fast_jackknife(
            lenses,
            sources,
            randoms,
            lens_assign,
            source_assign,
            random_assign,
            battrs=battrs,
            n_bins=n_bins,
            n_regions=args.n_regions,
            nthreads=args.nthreads,
        )
        print(f'  Completed in {time.time() - t0:.2f} s')

        output_path = save_sample_results(
            output_dir,
            sample_name,
            theta_edges,
            theta_centers,
            centres,
            source_counts,
            region_counts(lens_assign, args.n_regions),
            region_counts(random_assign, args.n_regions),
            results,
        )
        print(f'  Saved {output_path}')

        sample_results[sample_name] = results

    pdf_path, png_path = plot_combined_results(
        theta_centers,
        args.samples,
        sample_results,
        output_dir,
        args.output_prefix,
        usetex=args.usetex,
    )
    print('\nSaved combined figure outputs:')
    print(f'  {pdf_path}')
    print(f'  {png_path}')


if __name__ == '__main__':
    main()
