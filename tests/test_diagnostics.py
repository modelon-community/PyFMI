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
import os
import numpy as np

from pyfmi.common.io import (
    ResultHandler,
)
from pyfmi.common.diagnostics import (
    DIAGNOSTICS_PREFIX,
    DiagnosticsHelper
)

from pyfmi.test_util import Dummy_FMUModelME2

file_path = os.path.dirname(os.path.abspath(__file__))

class ResultStoreCalcDiagnostics(ResultHandler):
    """Result handler for testing explicit storage of calculated diagnostics."""
    def __init__(self, model = None):
        super().__init__(model)
        self._diags_aux = DiagnosticsHelper()

        self.supports['dynamic_diagnostics'] = True
        self.diag_params : Union[dict, None] = None
        self.diag_vars : Union[dict, None] = None
        self.diags_calc : Union[dict, None] = None

    def simulation_start(self, diagnostics_params = {}, diagnostics_vars = {}):
        self.diag_params = {k: [v[0]] for k, v in diagnostics_params.items()}
        self.diag_vars = {k: [v[0]] for k, v in diagnostics_vars.items()}
        self.diags_calc = {k: [v[0]] for k, v in self._diags_aux.setup_calculated_diagnostics_variables(diagnostics_params, diagnostics_vars).items()}

    def diagnostics_point(self, diag_data):
        # store ordinary diagnostics data
        for k, diag_val in zip(self.diag_vars.keys(), diag_data):
            self.diag_vars[k].append(diag_val)
        # calculated_diagnostics_data
        calculated_diags = self._diags_aux.get_calculated_diagnostics_point(diag_data)
        # store calculated diagnostics data
        for k, diag_val in zip(self.diags_calc.keys(), calculated_diags):
            self.diags_calc[k].append(diag_val)

    def get_result(self):
        return {**self.diag_vars, **self.diags_calc}
    
    def get_all_diag_var_names(self) -> set:
        return set(list(self.diag_params.keys()) + list(self.diag_vars.keys()) + list(self.diags_calc.keys()))
    
class TestStoreCalculatedDiagnostics:
    """Tests relating to the DiagnosticsHelper class.""" # TODO
    @pytest.mark.parametrize("solver", ["CVode", "Radau5ODE", "ExplicitEuler"])
    def test_X(self, solver):
        """Test correctness of explicitly storing calculated diagnostic trajectories."""
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll = False)

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

        # # 4. Ensure there are events
        # assert res_binary[f"{DIAGNOSTICS_PREFIX}nbr_events"][-1] > 0

        # 5. Check (continuous) diagnostics trajectories are identical,
        # minus cpu time related ones
        for traj_name in res_test.keys(): # only continuous; no parameters
            if "cpu_time" in traj_name:
                continue
            np.testing.assert_array_equal(
                res_test[traj_name], 
                res_binary[traj_name],
                err_msg = f"{traj_name} not equal")
