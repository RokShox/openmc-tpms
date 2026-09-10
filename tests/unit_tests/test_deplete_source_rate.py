"""Tests for source-rate depletion normalization, including subcritical scaling."""

from math import pi
from pathlib import Path
from unittest.mock import patch

import pytest

import openmc
import openmc.deplete
from openmc.deplete.helpers import SourceRateHelper

CHAIN_PATH = Path(__file__).parents[1] / "chain_simple.xml"


def test_lib_run_mode_mapping():
    """Python ctypes run_mode table matches the C++ RunMode enum."""
    from openmc.lib.settings import _RUN_MODES
    assert _RUN_MODES[1] == 'fixed source'
    assert _RUN_MODES[2] == 'eigenvalue'
    assert _RUN_MODES[3] == 'subcritical multiplication'
    assert _RUN_MODES[4] == 'plot'
    assert _RUN_MODES[5] == 'particle restart'
    assert _RUN_MODES[6] == 'volume'


def test_source_rate_helper_factor_identity():
    helper = SourceRateHelper()
    with patch('openmc.lib.keff', return_value=(0.5, 0.01)):
        assert helper.factor(10.0) == 10.0


def test_source_rate_helper_subcritical_scale():
    helper = SourceRateHelper(subcritical_multiplication=True)
    with patch('openmc.lib.keff', return_value=(0.0, 0.0)):
        assert helper.factor(10.0) == 10.0
    with patch('openmc.lib.keff', return_value=(0.5, 0.01)):
        assert helper.factor(10.0) == pytest.approx(20.0)
    with patch('openmc.lib.keff', return_value=(1.0, 0.0)):
        with pytest.raises(RuntimeError, match="k < 1"):
            helper.factor(10.0)
    with patch('openmc.lib.keff', return_value=(1.05, 0.0)):
        with pytest.raises(RuntimeError, match="k < 1"):
            helper.factor(10.0)


def _sphere_model(run_mode):
    """Subcritical U235 sphere with an isotropic point source."""
    openmc.reset_auto_ids()
    model = openmc.Model()

    fuel = openmc.Material()
    fuel.add_nuclide('U235', 1.0)
    fuel.set_density('g/cm3', 10.0)
    fuel.depletable = True
    radius = 7.0
    fuel.volume = 4.0 / 3.0 * pi * radius**3

    sph = openmc.Sphere(r=radius, boundary_type='vacuum')
    cell = openmc.Cell(fill=fuel, region=-sph)
    model.geometry = openmc.Geometry([cell])

    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(),
        energy=openmc.stats.Watt(),
    )
    model.settings.particles = 3000
    model.settings.verbosity = 1
    model.settings.run_mode = run_mode
    if run_mode == 'subcritical multiplication':
        model.settings.batches = 16
        model.settings.inactive = 6
    else:
        model.settings.batches = 12
        model.settings.inactive = 0
    return model


def test_independent_operator_unscaled():
    """IndependentOperator must not apply 1/(1-k); fluxes are caller-normalized."""
    micro_xs = openmc.deplete.MicroXS.from_csv(
        Path(__file__).parents[1] / "micro_xs_simple.csv")
    op = openmc.deplete.IndependentOperator.from_nuclides(
        1.0, {'U235': 1.0e20}, 1.0, micro_xs, CHAIN_PATH,
        nuc_units='atom/cm3', normalization_mode='source-rate',
    )
    assert op._normalization_helper.subcritical_multiplication is False
    assert op._normalization_helper.factor(10.0) == 10.0


def test_coupled_operator_sets_subcritical_flag():
    op = openmc.deplete.CoupledOperator(
        _sphere_model('fixed source'), CHAIN_PATH,
        normalization_mode='source-rate',
    )
    assert op._normalization_helper.subcritical_multiplication is False

    op = openmc.deplete.CoupledOperator(
        _sphere_model('subcritical multiplication'), CHAIN_PATH,
        normalization_mode='source-rate',
    )
    assert op._normalization_helper.subcritical_multiplication is True

    op = openmc.deplete.CoupledOperator(
        _sphere_model('eigenvalue'), CHAIN_PATH,
        normalization_mode='fission-q',
    )
    assert not isinstance(op._normalization_helper, SourceRateHelper)


def _operator_result(model, source_rate):
    op = openmc.deplete.CoupledOperator(
        model, CHAIN_PATH, normalization_mode='source-rate',
    )
    vec = op.initial_condition()
    try:
        return op(vec, source_rate)
    finally:
        op.finalize()


@pytest.mark.flaky(reruns=1)
def test_subcritical_matches_analog_fixed_source(run_in_tmpdir):
    """Source-rate rates in subcritical mode should match analog fixed source."""
    source_rate = 1.0

    result_fs = _operator_result(_sphere_model('fixed source'), source_rate)
    result_scm = _operator_result(
        _sphere_model('subcritical multiplication'), source_rate)

    k = result_scm.k.n
    assert 0.35 < k < 0.95

    rates_fs = result_fs.rates
    rates_scm = result_scm.rates
    i_nuc = rates_fs.index_nuc['U235']
    i_rx = rates_fs.index_rx['fission']
    fission_fs = rates_fs[0, i_nuc, i_rx]
    fission_scm = rates_scm[0, i_nuc, i_rx]

    assert fission_fs > 0.0
    ratio = fission_scm / fission_fs
    assert ratio == pytest.approx(1.0, rel=0.15)
    # Without 1/(1-k), subcritical rates would be lower by about (1-k)
    assert abs(ratio - (1.0 - k)) > 0.2
