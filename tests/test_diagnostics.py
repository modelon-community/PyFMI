#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2025 Modelon AB
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import pytest
import numpy as np
from pathlib import Path

from pyfmi import load_fmu
from pyfmi.common.io import ResultHandler
from pyfmi.common.diagnostics import (
    DIAGNOSTICS_PREFIX,
    DynamicDiagnosticsUtils
)

this_dir = Path(__file__).parent.absolute()
FMI2_REF_FMU_PATH = Path(this_dir) / 'files' / 'reference_fmus' / '2.0'

class ResultStoreCalcDiagnostics(ResultHandler):
    """Result handler for testing explicit storage of calculated diagnostics."""
    def __init__(self, model = None):
        super().__init__(model)
        self._diags_aux = DynamicDiagnosticsUtils()

        self.supports['dynamic_diagnostics'] = True
        self.diag_params : Union[dict, None] = None
        self.diag_vars : Union[dict, None] = None
        self.diags_calc : Union[dict, None] = None

    def simulation_start(self, diagnostics_params = {}, diagnostics_vars = {}):
        self.diag_params = {k: [v[0]] for k, v in diagnostics_params.items()}
        self.diag_vars = {k: [v[0]] for k, v in diagnostics_vars.items()}
        self.diags_calc = {k: [v[0]] for k, v in self._diags_aux.setup_calculated_diagnostics_variables(diagnostics_vars).items()}

    def diagnostics_point(self, diag_data):
        # store ordinary diagnostics data
        for k, diag_val in zip(self.diag_vars.keys(), diag_data):
            self.diag_vars[k].append(diag_val)
        # calculated_diagnostics_data
        calculated_diags = self._diags_aux.get_calculated_diagnostics_point(diag_data)
        # store calculated diagnostics data
        for k, diag_val in zip(self.diags_calc.keys(), calculated_diags):
            self.diags_calc[k].append(diag_val)

    def get_result(self) -> dict:
        return {**self.diag_vars, **self.diags_calc}
    
    def get_all_diag_var_names(self) -> set:
        return set(list(self.diag_params.keys()) + list(self.diag_vars.keys()) + list(self.diags_calc.keys()))
    
class TestDynamicDiagnosticsUtils:
    """Tests relating to the DynamicDiagnosticsUtils class."""
    @pytest.mark.parametrize("cpu_time_per_step, expected_output", 
        [
            (np.array([0, 1, 2, 3, 4]), np.array([0, 1, 3, 6, 10])),
            (np.array([0, 1, 0, 1, 0]), np.array([0, 1, 1, 2, 2])),
        ]
    )
    def test_get_cpu_time(self, cpu_time_per_step, expected_output):
        """Test correctness of DynamicDiagnosticsUtils.get_cpu_time."""
        np.testing.assert_array_equal(
            DynamicDiagnosticsUtils.get_cpu_time(cpu_time_per_step), 
            expected_output
        )
    
    def test_get_events_and_steps(self):
        """Test correctness of DynamicDiagnosticsUtils.get_events_and_steps"""
        # contains all subsequences of length 2, consisting of [-1, 0, 1] 
        event_type_data = np.array([-1, -1, 0, -1, 1, 1, 0, 0, 1, -1])
        output = DynamicDiagnosticsUtils.get_events_and_steps(event_type_data)
        np.testing.assert_array_equal(
            output[f"{DIAGNOSTICS_PREFIX}nbr_events"],
            np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 6])
        )
        np.testing.assert_array_equal(
            output[f"{DIAGNOSTICS_PREFIX}nbr_time_events"],
            np.array([0, 0, 0, 0, 1, 2, 2, 2, 3, 3])
        )
        np.testing.assert_array_equal(
            output[f"{DIAGNOSTICS_PREFIX}nbr_state_events"],
            np.array([0, 0, 1, 1, 1, 1, 2, 3, 3, 3])
        )
        np.testing.assert_array_equal(
            output[f"{DIAGNOSTICS_PREFIX}nbr_steps"],
            np.array([1, 2, 2, 3, 3, 3, 3, 3, 3, 4])
        )

    def test_get_nbr_state_limits(self):
        """Test correctness of DynamicDiagnosticsUtils.get_nbr_state_limits"""
        event_type_data = np.array([-1, -1, 0, 0, 1, 1, -1, -1])
        state_error = np.array([0.9, 1.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.9])
        np.testing.assert_array_equal(
            DynamicDiagnosticsUtils.get_nbr_state_limits(event_type_data, state_error),
            np.array([0, 1, 1, 1, 1, 1, 2, 2])
        )

    def test_calculated_diagnostics_consistency(self):
        """Test that the results of computing calculated diagnostics on trajectory and 
        per-point basis are consistent."""

        diagnostics_vars = {
            f"{DIAGNOSTICS_PREFIX}cpu_time_per_step"               : np.array([0, 1, 0, 1, 1, 1, 0, 0, 1, 1]),
            f"{DIAGNOSTICS_PREFIX}event_data.event_info.event_type": np.array([-1, -1, 0, -1, 1, 1, 0, 0, 1, -1]),
            f"{DIAGNOSTICS_PREFIX}state_errors.state_x"            : np.array([0.9, 1., 0., 2., 0., 2., 2., 2., 2., 1.]),
        }

        expected = {
            f"{DIAGNOSTICS_PREFIX}cpu_time":                      np.array([0, 1, 1, 2, 3, 4, 4, 4, 5, 6]),
            f"{DIAGNOSTICS_PREFIX}nbr_events":                    np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 6]),
            f"{DIAGNOSTICS_PREFIX}nbr_time_events":               np.array([0, 0, 0, 0, 1, 2, 2, 2, 3, 3]),
            f"{DIAGNOSTICS_PREFIX}nbr_state_events":              np.array([0, 0, 1, 1, 1, 1, 2, 3, 3, 3]),
            f"{DIAGNOSTICS_PREFIX}nbr_steps":                     np.array([1, 2, 2, 3, 3, 3, 3, 3, 3, 4]),
            f"{DIAGNOSTICS_PREFIX}nbr_state_limits_step.state_x": np.array([0, 1, 1, 2, 2, 2, 2, 2, 2, 3]),
        }

        # verify via DynamicDiagnosticsUtils class functions on trajectories
        res_trajectories = {}
        res_trajectories[f"{DIAGNOSTICS_PREFIX}cpu_time"] = DynamicDiagnosticsUtils.get_cpu_time(
            diagnostics_vars[f"{DIAGNOSTICS_PREFIX}cpu_time_per_step"])
        steps = DynamicDiagnosticsUtils.get_events_and_steps(
            diagnostics_vars[f"{DIAGNOSTICS_PREFIX}event_data.event_info.event_type"],
        )
        res_trajectories.update(steps)
        res_trajectories[f"{DIAGNOSTICS_PREFIX}nbr_state_limits_step.state_x"] = DynamicDiagnosticsUtils.get_nbr_state_limits(
            diagnostics_vars[f"{DIAGNOSTICS_PREFIX}event_data.event_info.event_type"],
            diagnostics_vars[f"{DIAGNOSTICS_PREFIX}state_errors.state_x"],
        )

        assert expected.keys() == res_trajectories.keys()
        for k in res_trajectories.keys():
            np.testing.assert_array_equal(
                expected[k],
                res_trajectories[k],
                err_msg = f"{k} differs"
            )
        
        # verify via DynamicDiagnosticsUtils.get_calculated_diagnostics_point
        dyn_diags_util = DynamicDiagnosticsUtils()
        # set start values
        calc_diags_points = {k: [v[0]] for k, v in dyn_diags_util.setup_calculated_diagnostics_variables(diagnostics_vars).items()}

        n_points = len(diagnostics_vars[f"{DIAGNOSTICS_PREFIX}cpu_time_per_step"]) - 1 # -1 to adjust for start
        # loop over points and build trajectories
        for i in range(1, n_points + 1):
            diags_input = np.array([v[i] for v in diagnostics_vars.values()])
            calc_diags_point = dyn_diags_util.get_calculated_diagnostics_point(diags_input)
            for p, calc_diag_name in zip(calc_diags_point, calc_diags_points):
                calc_diags_points[calc_diag_name].append(p)

        assert expected.keys() == calc_diags_points.keys()
        for k in calc_diags_points.keys():
            np.testing.assert_array_equal(
                expected[k],
                calc_diags_points[k],
                err_msg = f"{k} differs"
            )

    @pytest.mark.parametrize("solver", ["CVode", "Radau5ODE", "ExplicitEuler"])
    @pytest.mark.parametrize("fmu", 
        [
            FMI2_REF_FMU_PATH / "BouncingBall.fmu", # state events
            FMI2_REF_FMU_PATH / "Stair.fmu", # time events
        ]
    )
    def test_simulation_trajectory_correctness(self, solver, fmu):
        """Test correctness of explicitly storing calculated diagnostic trajectories."""
        model = load_fmu(fmu)

        # 1. Simulate model using test result handler
        res_handler_test = ResultStoreCalcDiagnostics()
        opts = model.simulate_options()
        opts["solver"] = solver
        opts["dynamic_diagnostics"] = True
        opts["result_handling"] = "custom"
        opts["result_handler"] = res_handler_test
        model.simulate(options = opts)
        res_test = res_handler_test.get_result()

        model.reset()
        
        # 2. Simulate model using default result handler & retrieve result
        opts = model.simulate_options()
        opts["solver"] = solver
        opts["dynamic_diagnostics"] = True
        opts["result_handling"] = "binary"
        res_binary = model.simulate(options = opts)

        # 3. Check both results have the same diagnostics variables
        diag_vars_test = res_handler_test.get_all_diag_var_names()
        diag_vars_default = set([v for v in res_binary.keys() if v.startswith(DIAGNOSTICS_PREFIX)])

        assert diag_vars_test == diag_vars_default

        # 4. Ensure there are events
        assert res_binary[f"{DIAGNOSTICS_PREFIX}nbr_events"][-1] > 0

        # 5. Check (continuous) diagnostics trajectories are identical,
        # minus cpu time related ones
        for traj_name in res_test.keys(): # only continuous; no parameters
            if "cpu_time" in traj_name:
                continue
            np.testing.assert_array_equal(
                res_test[traj_name], 
                res_binary[traj_name],
                err_msg = f"{traj_name} not equal")
