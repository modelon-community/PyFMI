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

import pytest
import numpy as np
from scipy.interpolate import interp1d

from pyfmi import load_fmu

this_dir = Path(__file__).parent.absolute()
FMI3_REF_FMU_PATH = Path(this_dir) / 'files' / 'reference_fmus' / '3.0'

class TestSimulation:
    """Tests involving simulation of FMUs for FMI 3."""

    def test_simulate(self):
        """Test simulate VDP model and verify the integrity of the results. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        results = fmu.simulate()

        assert results['x0'][0] == 2.0
        assert results['x1'][0] == 0.0
        assert results['x0'][-1] == pytest.approx( 2.00814337)
        assert results['x1'][-1] == pytest.approx(-0.04277047)
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

    @pytest.mark.xfail(strict = True, reason = "Requires support for state-events.")
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
