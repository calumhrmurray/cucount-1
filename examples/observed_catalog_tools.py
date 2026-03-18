from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
from astropy.io import fits

from cucount.numpy import Particles


DEFAULT_DESI_DATA = os.environ.get('CUCOUNT_DESI_DATA')
DEFAULT_DESI_RANDOMS_GLOB = os.environ.get('CUCOUNT_DESI_RANDOMS_GLOB')
DEFAULT_UNIONS_SOURCES = os.environ.get('CUCOUNT_UNIONS_SOURCES')

DEFAULT_DESI_H = 0.6766
DEFAULT_DESI_OMEGA_M = 0.3111


@dataclass
class CatalogSample:
    ra: np.ndarray
    dec: np.ndarray
    weights: np.ndarray
    z: np.ndarray | None = None
    e1: np.ndarray | None = None
    e2: np.ndarray | None = None

    @property
    def size(self) -> int:
        return self.ra.size


def resolve_random_catalogs(pattern: str, max_files: int | None = None) -> list[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No random catalogs matched pattern: {pattern}')
    if max_files is not None:
        paths = paths[:max_files]
    return paths


def choose_rows(size: int, max_rows: int | None, seed: int) -> np.ndarray | None:
    if max_rows is None or max_rows >= size:
        return None

    rng = np.random.default_rng(seed)
    if size <= 20_000_000:
        return np.sort(rng.choice(size, size=max_rows, replace=False))

    # Large FITS tables can make fully random indexing expensive.
    stride = max(size // max_rows, 1)
    offset = int(rng.integers(0, stride))
    indices = offset + stride * np.arange(max_rows, dtype=np.int64)
    return np.clip(indices, 0, size - 1)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator != 0,
        )


def weighted_auto_norm(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(weights) ** 2 - np.sum(weights**2))


def weighted_cross_norm(weights_one: np.ndarray, weights_two: np.ndarray) -> float:
    return float(np.sum(weights_one) * np.sum(weights_two))


def legendre_values(ell: int, mu: np.ndarray) -> np.ndarray:
    coeffs = np.zeros(ell + 1, dtype=np.float64)
    coeffs[ell] = 1.0
    return np.polynomial.legendre.legval(mu, coeffs)


def build_distance_to_comoving(h: float = DEFAULT_DESI_H, omega_m: float = DEFAULT_DESI_OMEGA_M):
    try:
        from cosmoprimo.fiducial import DESI

        cosmo = DESI(engine='eisenstein_hu')

        def distance_to_comoving(redshift: np.ndarray) -> np.ndarray:
            return np.asarray(cosmo.comoving_radial_distance(redshift), dtype=np.float64) * float(cosmo.h)

        return distance_to_comoving, 'cosmoprimo.DESI'
    except Exception:
        from astropy.cosmology import FlatLambdaCDM

        cosmo = FlatLambdaCDM(H0=100.0 * h, Om0=omega_m)

        def distance_to_comoving(redshift: np.ndarray) -> np.ndarray:
            return np.asarray(cosmo.comoving_distance(redshift).value, dtype=np.float64) * h

        return distance_to_comoving, f'FlatLambdaCDM(h={h:.4f}, Om0={omega_m:.4f})'


def sky_to_cartesian(ra_deg: np.ndarray, dec_deg: np.ndarray, distance: np.ndarray | float | None = None) -> np.ndarray:
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    if distance is None:
        distance = 1.0
    radius = np.asarray(distance, dtype=np.float64)
    conv = np.pi / 180.0
    cos_dec = np.cos(dec * conv)
    x = radius * cos_dec * np.cos(ra * conv)
    y = radius * cos_dec * np.sin(ra * conv)
    z = radius * np.sin(dec * conv)
    return np.column_stack([x, y, z])


def create_particles(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    weights: np.ndarray,
    distance: np.ndarray | None = None,
    shear: tuple[np.ndarray, np.ndarray] | None = None,
) -> Particles:
    positions = sky_to_cartesian(ra_deg, dec_deg, distance=distance)
    kwargs = {}
    if shear is not None:
        kwargs['spin_values'] = -np.column_stack([np.asarray(shear[0], dtype=np.float64), np.asarray(shear[1], dtype=np.float64)])
    return Particles(positions, weights=np.asarray(weights, dtype=np.float64), **kwargs)


def load_desi_catalog(
    path: str,
    max_rows: int | None = None,
    seed: int = 0,
    use_fkp: bool = True,
) -> CatalogSample:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        indices = choose_rows(len(data), max_rows=max_rows, seed=seed)
        row = slice(None) if indices is None else indices
        ra = np.asarray(data['RA'][row], dtype=np.float64)
        dec = np.asarray(data['DEC'][row], dtype=np.float64)
        redshift = np.asarray(data['Z'][row], dtype=np.float64)
        weights = np.asarray(data['WEIGHT'][row], dtype=np.float64)
        if use_fkp and 'WEIGHT_FKP' in hdul[1].columns.names:
            weights *= np.asarray(data['WEIGHT_FKP'][row], dtype=np.float64)

    mask = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(redshift)
        & np.isfinite(weights)
        & (redshift > 0.0)
        & (weights > 0.0)
    )
    return CatalogSample(ra=ra[mask], dec=dec[mask], z=redshift[mask], weights=weights[mask])


def load_unions_catalog(
    path: str,
    max_rows: int | None = None,
    seed: int = 0,
    weight_col: str = 'auto',
    e1_col: str = 'e1',
    e2_col: str = 'e2',
) -> CatalogSample:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        columns = hdul[1].columns.names
        dec_col = 'Dec' if 'Dec' in columns else 'DEC'
        if weight_col == 'auto':
            weight_col = 'w_des' if 'w_des' in columns else 'w'
        indices = choose_rows(len(data), max_rows=max_rows, seed=seed)
        row = slice(None) if indices is None else indices
        ra = np.asarray(data['RA'][row], dtype=np.float64)
        dec = np.asarray(data[dec_col][row], dtype=np.float64)
        e1 = np.asarray(data[e1_col][row], dtype=np.float64)
        e2 = np.asarray(data[e2_col][row], dtype=np.float64)
        weights = np.asarray(data[weight_col][row], dtype=np.float64)

    mask = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(e1)
        & np.isfinite(e2)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    return CatalogSample(ra=ra[mask], dec=dec[mask], e1=e1[mask], e2=e2[mask], weights=weights[mask])
