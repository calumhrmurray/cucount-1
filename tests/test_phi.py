import numpy as np
import pytest


def get_cartesian(ra, dec):
    conv = np.pi / 180.
    theta, phi = dec * conv, ra * conv
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return np.column_stack([x, y, z])


def make_spin_values(angles_deg, spin, amplitude=1.0):
    angles_deg = np.asarray(angles_deg, dtype=np.float64)
    amplitude = np.asarray(amplitude, dtype=np.float64)
    phase = np.deg2rad(spin * angles_deg)
    return np.column_stack([amplitude * np.cos(phase), amplitude * np.sin(phase)])


def make_particles(ra, dec, weights, spin_angles_deg=None, spin=None):
    from cucount.numpy import Particles

    positions = get_cartesian(np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64))
    kwargs = {}
    if spin is not None:
        kwargs['spin_values'] = make_spin_values(spin_angles_deg, spin)
    return Particles(positions, np.asarray(weights, dtype=np.float64), **kwargs)


def test_phi_binning_requires_spin_on_first_catalog():
    from cucount.numpy import BinAttrs, count2

    particles1 = make_particles([0.0], [0.0], [1.0])
    particles2 = make_particles([1.0], [0.0], [1.0])
    battrs = BinAttrs(theta=np.array([0.5, 2.0]), phi=np.array([0.0, 180.0, 360.0]))

    with pytest.raises(ValueError, match='Binning in phi requires WeightAttrs.spin'):
        count2(particles1, particles2, battrs=battrs)


def test_phi_binning_spin1_scalar_exact():
    from cucount.numpy import BinAttrs, WeightAttrs, count2

    particles1 = make_particles([0.0], [0.0], [1.0], spin_angles_deg=[0.0], spin=1)
    particles2 = make_particles(
        [0.0, 1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0, 0.0],
        [1.0, 2.0, 3.0, 4.0],
    )
    theta_edges = np.array([0.5, 2.0])
    phi_edges = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
    wattrs = WeightAttrs(spin=(1, 0))

    counts_theta_phi = count2(
        particles1,
        particles2,
        battrs=BinAttrs(theta=theta_edges, phi=phi_edges),
        wattrs=wattrs,
    )
    counts_theta = count2(
        particles1,
        particles2,
        battrs=BinAttrs(theta=theta_edges),
        wattrs=wattrs,
    )

    expected_plus = np.array([[-1.0, 0.0, 3.0, 0.0]])
    expected_cross = np.array([[0.0, 2.0, 0.0, -4.0]])

    assert counts_theta_phi['weight_plus'].shape == (1, 4)
    assert counts_theta_phi['weight_cross'].shape == (1, 4)
    assert np.allclose(counts_theta_phi['weight_plus'], expected_plus, atol=1e-12)
    assert np.allclose(counts_theta_phi['weight_cross'], expected_cross, atol=1e-12)
    assert np.allclose(counts_theta_phi['weight_plus'].sum(axis=1), counts_theta['weight_plus'])
    assert np.allclose(counts_theta_phi['weight_cross'].sum(axis=1), counts_theta['weight_cross'])


def test_phi_binning_spin2_periodicity():
    from cucount.numpy import BinAttrs, WeightAttrs, count2

    particles1 = make_particles([0.0], [0.0], [1.0], spin_angles_deg=[0.0], spin=2)
    particles2 = make_particles(
        [0.0, 1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    )
    theta_edges = np.array([0.5, 2.0])
    phi_edges = np.array([0.0, 90.0, 180.0])
    wattrs = WeightAttrs(spin=(2, 0))

    counts_theta_phi = count2(
        particles1,
        particles2,
        battrs=BinAttrs(theta=theta_edges, phi=phi_edges),
        wattrs=wattrs,
    )
    counts_theta = count2(
        particles1,
        particles2,
        battrs=BinAttrs(theta=theta_edges),
        wattrs=wattrs,
    )

    expected_plus = np.array([[-2.0, 2.0]])
    expected_cross = np.array([[0.0, 0.0]])

    assert counts_theta_phi['weight_plus'].shape == (1, 2)
    assert counts_theta_phi['weight_cross'].shape == (1, 2)
    assert np.allclose(counts_theta_phi['weight_plus'], expected_plus, atol=1e-12)
    assert np.allclose(counts_theta_phi['weight_cross'], expected_cross, atol=1e-12)
    assert np.allclose(counts_theta_phi['weight_plus'].sum(axis=1), counts_theta['weight_plus'])
    assert np.allclose(counts_theta_phi['weight_cross'].sum(axis=1), counts_theta['weight_cross'])


def test_phi_binning_spin1_opposite_directions_are_distinct():
    from cucount.numpy import BinAttrs, WeightAttrs, count2

    particles2 = make_particles([0.0], [1.0], [1.0])
    theta_edges = np.array([0.5, 2.0])
    phi_edges = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
    wattrs = WeightAttrs(spin=(1, 0))

    counts_pos = count2(
        make_particles([0.0], [0.0], [1.0], spin_angles_deg=[0.0], spin=1),
        particles2,
        battrs=BinAttrs(theta=theta_edges, phi=phi_edges),
        wattrs=wattrs,
    )
    counts_neg = count2(
        make_particles([0.0], [0.0], [1.0], spin_angles_deg=[180.0], spin=1),
        particles2,
        battrs=BinAttrs(theta=theta_edges, phi=phi_edges),
        wattrs=wattrs,
    )

    assert np.allclose(counts_pos['weight'], np.array([[1.0, 0.0, 0.0, 0.0]]), atol=1e-12)
    assert np.allclose(counts_neg['weight'], np.array([[0.0, 0.0, 1.0, 0.0]]), atol=1e-12)
    assert np.allclose(counts_pos['weight_plus'], np.array([[-1.0, 0.0, 0.0, 0.0]]), atol=1e-12)
    assert np.allclose(counts_neg['weight_plus'], np.array([[0.0, 0.0, 1.0, 0.0]]), atol=1e-12)
    assert np.allclose(counts_pos['weight_cross'], 0.0, atol=1e-12)
    assert np.allclose(counts_neg['weight_cross'], 0.0, atol=1e-12)


def test_phi_binning_reference_only_first_spin_with_second_spin_output():
    from cucount.numpy import BinAttrs, WeightAttrs, count2

    particles1 = make_particles([0.0], [0.0], [1.0], spin_angles_deg=[0.0], spin=1)
    particles2 = make_particles([0.0], [1.0], [2.0], spin_angles_deg=[0.0], spin=2)
    theta_edges = np.array([0.5, 2.0])
    phi_edges = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
    wattrs = WeightAttrs(spin=(1, 2), reference_only=(True, False))

    counts = count2(
        particles1,
        particles2,
        battrs=BinAttrs(theta=theta_edges, phi=phi_edges),
        wattrs=wattrs,
    )

    assert set(counts) == {'weight', 'weight_plus', 'weight_cross'}
    assert np.allclose(counts['weight'], np.array([[2.0, 0.0, 0.0, 0.0]]), atol=1e-12)
    assert np.allclose(counts['weight_plus'], np.array([[-2.0, 0.0, 0.0, 0.0]]), atol=1e-12)
    assert np.allclose(counts['weight_cross'], 0.0, atol=1e-12)
