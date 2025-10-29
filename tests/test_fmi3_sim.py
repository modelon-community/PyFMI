#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2025 Modelon AB
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from pathlib import Path

import re
import pytest
import numpy as np
from scipy.interpolate import interp1d

from pyfmi import load_fmu
from pyfmi.fmi_algorithm_drivers import AssimuloFMIAlg
from pyfmi.fmi3 import FMUModelME3
from pyfmi.exceptions import FMUException

this_dir = Path(__file__).parent
FMI3_REF_FMU_PATH = Path(this_dir) / 'files' / 'reference_fmus' / '3.0'

class TestSimulation:
    """Tests involving simulation of FMUs for FMI 3."""

    def test_simulate(self):
        """Test simulate VDP model and verify the integrity of the results. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        results = fmu.simulate()

        assert results['x0'][0] == 2.0
        assert results['x1'][0] == 0.0
        assert results['x0'][-1] == pytest.approx( 2.008130983012657)
        assert results['x1'][-1] == pytest.approx(-0.042960828207896706)
        np.testing.assert_equal(results['mu'], np.ones(len(results['x0'])))

    def test_simulate_check_result_members(self):
        """Test simulate VDP model and check accessible data. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        results = fmu.simulate(options = {"ncp": 1})

        expected_variables = ['time', 'x0', 'der(x0)', 'x1', 'der(x1)', 'mu']

        assert results.keys() == expected_variables

    def test_simulate_change_ncp(self):
        """Test simulate VDP model and change ncp. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        opts = fmu.simulate_options()

        opts['ncp'] = 600
        results = fmu.simulate(options = opts)

        expected_variables = ['time', 'x0', 'der(x0)', 'x1', 'der(x1)', 'mu']

        assert results.keys() == expected_variables

        assert all(len(results[v]) == 601 for v in expected_variables)

    @pytest.mark.parametrize("rh", ["file", "memory", "csv"])
    def test_simulate_unsupported_result_handler(self, rh):
        """Verify unsupported result handlers raises an exception"""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        opts = fmu.simulate_options()

        opts['result_handling'] = rh
        opts["ncp"] = 1
        msg = f"For FMI3: 'result_handling' set to '{rh}' is not supported. " + \
                "Consider setting this option to 'binary', 'custom' or None to continue."
        with pytest.raises(NotImplementedError, match = msg):
            fmu.simulate(options = opts)

    def test_result_variable_types(self):
        """Test which FMI variable types are supported in default result handler."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "Feedthrough.fmu")
        res = fmu.simulate(options = {"ncp": 1})

        expected = set([
            'time',
            'Float64_fixed_parameter',
            'Float64_tunable_parameter',
            'Float64_continuous_input',
            'Float64_continuous_output',
            'Float64_discrete_input',
            'Float64_discrete_output',
            'Float32_continuous_input',
            'Float32_continuous_output',
            'Float32_discrete_input',
            'Float32_discrete_output',
            'Int64_input',
            'Int64_output',
            'Int32_input',
            'Int32_output',
            'Int16_input',
            'Int16_output',
            'Int8_input',
            'Int8_output',
            'UInt64_input',
            'UInt64_output',
            'UInt32_input',
            'UInt32_output',
            'UInt16_input',
            'UInt16_output',
            'UInt8_input',
            'UInt8_output',
            'Boolean_input',
            'Boolean_output',
            'Enumeration_input',
            'Enumeration_output'
        ])
        assert set(res.keys()) == expected

    @pytest.mark.parametrize("variable_base_name, value",
        [
            ("Float64_continuous", 3.14),
            ("Float32_continuous", np.float32(3.14)),
            ("Int64", 10),
            ("Int32", 10),
            ("Int16", 10),
            ("Int8", 10),
            ("UInt64", 10),
            ("UInt32", 10),
            ("UInt16", 10),
            ("UInt8", 10),
            ("Boolean", True),
            ("Enumeration", 2),
        ]
    )
    def test_result_handling_sanity_check(self, variable_base_name, value):
        """Sanity check for result handling of the various supported variable types."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "Feedthrough.fmu")
        fmu.set(f"{variable_base_name}_input", value)
        res = fmu.simulate(options = {"ncp": 0})
        assert all(v for v in res[f"{variable_base_name}_output"] == value)

    def test_result_handler_int64_limitations(self):
        """Test precision limitations for (u)int64 variables in result handling. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "Feedthrough.fmu")
        val = 9223372036854775807
        assert float(val) != val # cannot be exactly represented
        fmu.set("Int64_input", val)
        fmu.set("UInt64_input", val)
        res = fmu.simulate(options = {"ncp": 0})

        assert res["Int64_output"][-1] == float(val)
        assert res["UInt64_output"][-1] == float(val)

    def test_result_handling_with_alias(self):
        """Test that result handling works with aliases."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")
        res = fmu.simulate(0, 0.001, options = {"ncp": 1})
        assert "h_ft" in res.keys()
        np.testing.assert_equal(res["h"], res["h_ft"])

    def test_time_events(self):
        """Test simulation with time events."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "Stair.fmu")
        res = fmu.simulate(0, 10, options = {"ncp": 1})
        assert res["counter"][-1] == 9
        assert res.solver.get_statistics()["ntimeevents"] == 9

    @pytest.mark.parametrize("var_name", [
            "Float32_continuous",
            "Float64_continuous"
        ]
    )
    def test_continuous_input(self, var_name):
        """Test setting continuous inputs to float values."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "Feedthrough.fmu")
        input_var = f"{var_name}_input"
        output_var = f"{var_name}_output"

        # Generate input
        t = np.linspace(0., 10., 100)
        real_y = np.cos(t)
        real_input_traj = np.transpose(np.vstack((t, real_y)))
        input_object = (input_var, real_input_traj)

        ncp = 500
        fmu.set(input_var, real_y[0])
        res = fmu.simulate(final_time = 10, input = input_object, options = {"ncp": ncp})

        np.testing.assert_array_equal(res[input_var], res[output_var])
        output_interp = interp1d(res["time"], res[output_var])(t)
        np.testing.assert_array_almost_equal(output_interp, real_y, decimal = 3)

    def test_state_events(self):
        """Test model with state events: bouncingBall."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")
        assert fmu.get_ode_sizes()[1] > 0

        res = fmu.simulate(options = {"ncp": 0})
        assert res.solver.get_statistics()["nstateevents"] > 0

    def test_automatic_jacobian_via_directional_derivatives(self):
        """Test that the Jacobian is automatically constructed via directional derivatives (available)."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        assert fmu.get_capability_flags()['providesDirectionalDerivatives'] is True
        alg = AssimuloFMIAlg(
            start_time = 0,
            final_time = 1,
            input = None,
            model = fmu,
            options = {}
        )
        assert alg.with_jacobian is True

    def test_automatic_jacobian_via_directional_derivatives_no_dir_ders(self):
        """Test that the Jacobian is automatically constructed via directional derivatives (not available)."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "Dahlquist.fmu")
        assert fmu.get_capability_flags()['providesDirectionalDerivatives'] is False
        alg = AssimuloFMIAlg(
            start_time = 0,
            final_time = 1,
            input = None,
            model = fmu,
            options = {}
        )
        assert alg.with_jacobian is False

    def test_with_jacobian_and_directional_derivatives(self):
        """Test 'with_jacobian' options, when directional_derivatives are available."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        assert fmu.get_capability_flags()['providesDirectionalDerivatives'] is True
        fmu.simulate(options = {"with_jacobian": True, "ncp": 0})

    def test_with_jacobian_without_directional_derivatives(self):
        """Test 'with_jacobian' options, when directional_derivatives are NOT available."""
        # This relies on FMUModelME3._estimate_directional_derivative
        fmu = load_fmu(FMI3_REF_FMU_PATH / "Dahlquist.fmu")
        assert fmu.get_capability_flags()['providesDirectionalDerivatives'] is False
        opts = fmu.simulate_options()
        opts["with_jacobian"] = True
        opts["ncp"] = 0
        fmu.simulate(options = opts)
        assert opts["with_jacobian"] is True

    def test_force_finite_differences(self):
        """Test 'force_finite_differences' option."""
        fmu = FMUModelME3(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        fmu.force_finite_differences = True
        fmu.initialize()
        fmu._get_A()

    @pytest.mark.parametrize("finite_differences_method", [1, 2])
    def test_finite_differences_method(self, finite_differences_method):
        """Test valid 'finite_difference_method' options."""
        fmu = FMUModelME3(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        fmu.force_finite_differences = True
        fmu.finite_differences_method = finite_differences_method
        fmu.initialize()
        fmu._get_A()

    def test_invalid_finite_differences_method(self):
        """Test invalid 'finite_differences_method'."""
        fmu = FMUModelME3(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        fmu.force_finite_differences = True
        fmu.finite_differences_method = 3
        fmu.initialize()
        msg = "Invalid 'finite_differences_method' for FMUModelME3, must be FORWARD_DIFFERENCE (1) or CENTRAL_DIFFERENCE (2)."
        with pytest.raises(FMUException, match = re.escape(msg)):
            fmu._get_A()

class TestDynamicDiagnostics:
    """Tests involving simulation of FMI3 FMUs using 'dynamic_diagnostics' == True."""
    # TODO: These should likely be parameterized and bundled with FMI2 tests

    @pytest.mark.parametrize("fmu_path",
        [
            FMI3_REF_FMU_PATH / "VanDerPol.fmu",
            FMI3_REF_FMU_PATH / "Stair.fmu", # time events
            FMI3_REF_FMU_PATH / "BouncingBall.fmu", # state events
        ]
    )
    @pytest.mark.parametrize("solver", ["CVode", "Radau5ODE", "ExplicitEuler"])
    def test_simulate(self, fmu_path, solver):
        """Test basic simulation and verify the expected result variables exists."""
        fmu = load_fmu(fmu_path)
        opts = {
            "solver": solver,
            "ncp": 1,
            "dynamic_diagnostics": True
        }
        res = fmu.simulate(options = opts)
        assert any(res_name.startswith("@Diagnostics.") for res_name in res.keys())

    @pytest.mark.parametrize("atol", [1e-4, [1e-4], np.array([1e-4]), np.array(1e-4), (1e-4)])
    def test_dynamic_diagnostics_scalar_atol(self, atol):
        """Test scalar atol + dynamic_diagnostics."""
        model = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")

        opts = model.simulate_options()
        solver = "CVode"
        opts[f"{solver}_options"]["atol"] = atol
        opts["dynamic_diagnostics"] = True
        opts["ncp"] = 1

        model.simulate(options = opts)
