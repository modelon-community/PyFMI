#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2014-2025 Modelon AB
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

"""
This file contains code for mapping FMUs to the fmu_probem specifications
required by Assimulo.
"""

# Imports for convenience/backwards compatibility
from pyfmi.simulation.assimulo_interface_fmi1 import (
    FMIODE,
    FMIODESENS
)
from pyfmi.simulation.assimulo_interface_fmi2 import (
    FMIODE2,
    FMIODESENS2
)
from pyfmi.exceptions import FMIModel_Exception, FMUException

from pyfmi.fmi1 import FMUModelME1
from pyfmi.fmi2 import FMUModelME2
from pyfmi.fmi_coupled import CoupledFMUModelME2

def get_fmi_ode_problem(
    model,
    result_file_name,
    with_jacobian: bool,
    start_time: double,
    logging: bool,
    result_handler,
    input_traj = None,
    number_of_diagnostics_variables: int = 0,
    sensitivities = None,
    extra_equations = None,
    synchronize_simulation = False
):
    """Convenience function for getting the correct FMIODEX class instance."""

    if isinstance(model, (FMUModelME2, CoupledFMUModelME2)):
        if sensitivities:
            fmu_prob = FMIODESENS2(
                model = model,
                input = input_traj,
                result_file_name = result_file_name,
                with_jacobian = with_jacobian,
                start_time = start_time,
                parameters = sensitivities,
                logging = logging,
                result_handler = result_handler,
                number_of_diagnostics_variables = number_of_diagnostics_variables
            )
        else:
            fmu_prob = FMIODE2(
                model = model,
                input = input_traj,
                result_file_name = result_file_name,
                with_jacobian = with_jacobian,
                start_time = start_time,
                logging = logging,
                result_handler = result_handler,
                extra_equations = extra_equations,
                synchronize_simulation = synchronize_simulation,
                number_of_diagnostics_variables = number_of_diagnostics_variables
            )
    elif isinstance(model, FMUModelME1):
        if sensitivities:
            fmu_prob = FMIODESENS(
                model = model,
                input = input_traj,
                result_file_name = result_file_name,
                with_jacobian = with_jacobian,
                start_time = start_time,
                parameters = sensitivities,
                logging = logging,
                result_handler = result_handler
            )
        else:
            fmu_prob = FMIODE(
                model = model,
                input = input_traj,
                result_file_name = result_file_name,
                with_jacobian = with_jacobian,
                start_time = start_time,
                logging = logging,
                result_handler = result_handler
            )
    else:
        raise FMUException("Unknown model.")
    return fmu_prob
