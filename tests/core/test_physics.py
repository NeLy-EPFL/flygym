"""Unit tests for flygym.compose.physics (ContactParams)."""

import pytest

from flygym.compose.physics import ContactParams


class TestContactParamsDefaults:
    def test_default_construction(self):
        params = ContactParams()
        assert params.sliding_friction == 1.0
        assert params.torsional_friction == 2e-2
        assert params.rolling_friction == 1e-4
        assert params.solver_refaccl_timeconst == 2e-4
        assert params.solver_refaccl_dampratio == 1.0
        assert params.solver_impedance_min == 0.98
        assert params.solver_impedance_max == 0.99

    def test_is_valid_defaults(self):
        assert ContactParams().is_valid() is True


class TestContactParamsTuples:
    def test_get_friction_tuple_length(self):
        params = ContactParams()
        friction = params.get_friction_tuple()
        assert len(friction) == 5

    def test_get_friction_tuple_values(self):
        params = ContactParams(sliding_friction=2.0, torsional_friction=0.1, rolling_friction=0.01)
        friction = params.get_friction_tuple()
        assert friction[0] == 2.0  # sliding_1
        assert friction[1] == 2.0  # sliding_2
        assert friction[2] == 0.1  # torsional
        assert friction[3] == 0.01  # rolling_1
        assert friction[4] == 0.01  # rolling_2

    def test_get_solref_tuple(self):
        params = ContactParams(
            solver_refaccl_timeconst=1e-3, solver_refaccl_dampratio=0.5
        )
        solref = params.get_solref_tuple()
        assert solref == (1e-3, 0.5)

    def test_get_solimp_tuple_length(self):
        params = ContactParams()
        solimp = params.get_solimp_tuple()
        assert len(solimp) == 4

    def test_get_solimp_tuple_values(self):
        params = ContactParams(
            solver_impedance_min=0.9,
            solver_impedance_max=0.95,
            solver_impedance_transitionmidpoint=0.5,
            solver_impedance_transitionsharpness=2.0,
        )
        solimp = params.get_solimp_tuple()
        assert solimp[0] == 0.9
        assert solimp[1] == 0.95
        assert solimp[2] == 0.5
        assert solimp[3] == 2.0


class TestContactParamsValidation:
    def test_negative_sliding_friction_invalid(self):
        params = ContactParams(sliding_friction=-1.0)
        assert params.is_valid(raise_on_invalid=False) is False
        with pytest.raises(ValueError):
            params.is_valid(raise_on_invalid=True)

    def test_negative_torsional_friction_invalid(self):
        params = ContactParams(torsional_friction=-0.01)
        assert params.is_valid(raise_on_invalid=False) is False

    def test_negative_rolling_friction_invalid(self):
        params = ContactParams(rolling_friction=-1e-5)
        assert params.is_valid(raise_on_invalid=False) is False

    def test_zero_friction_is_valid(self):
        params = ContactParams(sliding_friction=0.0, torsional_friction=0.0, rolling_friction=0.0)
        assert params.is_valid() is True

    def test_zero_refaccl_timeconst_invalid(self):
        params = ContactParams(solver_refaccl_timeconst=0.0)
        assert params.is_valid(raise_on_invalid=False) is False

    def test_negative_refaccl_timeconst_invalid(self):
        params = ContactParams(solver_refaccl_timeconst=-1e-4)
        assert params.is_valid(raise_on_invalid=False) is False

    def test_zero_dampratio_invalid(self):
        params = ContactParams(solver_refaccl_dampratio=0.0)
        assert params.is_valid(raise_on_invalid=False) is False

    def test_impedance_min_out_of_range_invalid(self):
        params = ContactParams(solver_impedance_min=0.0)  # must be > 0
        assert params.is_valid(raise_on_invalid=False) is False

        params = ContactParams(solver_impedance_min=1.0)  # must be < 1
        assert params.is_valid(raise_on_invalid=False) is False

    def test_impedance_max_out_of_range_invalid(self):
        params = ContactParams(solver_impedance_max=1.0)
        assert params.is_valid(raise_on_invalid=False) is False

    def test_impedance_max_less_than_min_invalid(self):
        params = ContactParams(solver_impedance_min=0.95, solver_impedance_max=0.90)
        assert params.is_valid(raise_on_invalid=False) is False

    def test_impedance_midpoint_out_of_range_invalid(self):
        params = ContactParams(solver_impedance_transitionmidpoint=0.0)
        assert params.is_valid(raise_on_invalid=False) is False
        params = ContactParams(solver_impedance_transitionmidpoint=1.0)
        assert params.is_valid(raise_on_invalid=False) is False

    def test_impedance_sharpness_below_one_invalid(self):
        params = ContactParams(solver_impedance_transitionsharpness=0.5)
        assert params.is_valid(raise_on_invalid=False) is False

    def test_impedance_sharpness_exactly_one_valid(self):
        params = ContactParams(solver_impedance_transitionsharpness=1.0)
        assert params.is_valid() is True

    def test_get_friction_tuple_raises_on_invalid(self):
        params = ContactParams(sliding_friction=-1.0)
        with pytest.raises(ValueError):
            params.get_friction_tuple()

    def test_get_solref_tuple_raises_on_invalid(self):
        params = ContactParams(solver_refaccl_timeconst=0.0)
        with pytest.raises(ValueError):
            params.get_solref_tuple()

    def test_get_solimp_tuple_raises_on_invalid(self):
        params = ContactParams(solver_impedance_min=0.0)
        with pytest.raises(ValueError):
            params.get_solimp_tuple()
