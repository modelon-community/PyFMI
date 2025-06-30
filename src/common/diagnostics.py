#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Modelon AB
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


## This file contains various components for the 'dynamics_diagnostics' options

from pyfmi.fmi2 import FMUModelME2
import numpy as np
import numbers

DIAGNOSTICS_PREFIX = '@Diagnostics.'

class DiagnosticsBase:
    """ Class serves as a template
        to keep track of diagnostics variables not part of the generated result file.
    """
    calculated_diagnostics = {
        'nbr_events': {'name': f'{DIAGNOSTICS_PREFIX}nbr_events', 'description': 'Cumulative number of events'},
        'nbr_time_events': {'name': f'{DIAGNOSTICS_PREFIX}nbr_time_events', 'description': 'Cumulative number of time events'},
        'nbr_state_events': {'name': f'{DIAGNOSTICS_PREFIX}nbr_state_events', 'description': 'Cumulative number of state events'},
        'nbr_steps': {'name': f'{DIAGNOSTICS_PREFIX}nbr_steps', 'description': 'Cumulative number of steps'},
        'nbr_state_limits_step': {'name': f'{DIAGNOSTICS_PREFIX}nbr_state_limits_step', 'description': 'Cumulative number of times states limit the step'},
    }

class DiagnosticsHelper: # TODO: Better name
    """ TODO, add docstring."""
    def get_calculated_diagnostics_names(self, diagnostics_vars: dict) -> dict:
        """ Given dictionary {diagnostics_var_name: description}, return a corresponding dictionary for the calculated diagnostics."""
        res = {
            f'{DIAGNOSTICS_PREFIX}cpu_time'              : 'Cumulative CPU time',
            # f'{DIAGNOSTICS_PREFIX}nbr_events'            : 'Cumulative number of events',
            # f'{DIAGNOSTICS_PREFIX}nbr_time_events'       : 'Cumulative number of time events',
            # f'{DIAGNOSTICS_PREFIX}nbr_state_events'      : 'Cumulative number of state events',
            # f'{DIAGNOSTICS_PREFIX}nbr_steps'             : 'Cumulative number of steps',
            DiagnosticsBase.calculated_diagnostics["nbr_events"]["name"]: DiagnosticsBase.calculated_diagnostics["nbr_events"]["description"],
            DiagnosticsBase.calculated_diagnostics["nbr_time_events"]["name"]: DiagnosticsBase.calculated_diagnostics["nbr_time_events"]["description"],
            DiagnosticsBase.calculated_diagnostics["nbr_state_events"]["name"]: DiagnosticsBase.calculated_diagnostics["nbr_state_events"]["description"],
            DiagnosticsBase.calculated_diagnostics["nbr_steps"]["name"]: DiagnosticsBase.calculated_diagnostics["nbr_steps"]["description"],
            # f'{DIAGNOSTICS_PREFIX}nbr_state_limits_step' : 'Cumulative number of times states limit the step',
        }

        extra_vars = {}
        for v, descr in diagnostics_vars.items():
            if v.startswith(f'{DIAGNOSTICS_PREFIX}solver.absolute_tolerance.'):
                _, state_name = v.split(".absolute_tolerance.")
                extra_vars[f'{DIAGNOSTICS_PREFIX}nbr_state_limits_step.' + state_name] = "TODO: Description"
        self._number_states = len(extra_vars)
        return {**res, **extra_vars}
    
    def _update_cache(self, a):
        """Update internal cache of latest diagnostics point."""
        self._calc_diags_vars_cache = a.copy()
    
    def init_calculated_variables(self, diag_vars: dict, diags_calc: dict) -> None:
        """Internal setup.""" # TODO: merge with another function?
        self._diags_vars_names = list(diag_vars.keys())
        self._calc_diags_vars_names = list(diags_calc.keys())

        self._calc_diags_vars_cache = [v[0] for v in diags_calc.values()]

        idx_state_errors = None
        for idx, key in enumerate(diag_vars):
            if key.startswith(f"{DIAGNOSTICS_PREFIX}state_errors."):
                idx_state_errors = idx
                break
        
        idx_nbr_state_limits_step = None
        for idx, key in enumerate(diags_calc):
            if key.startswith(f'{DIAGNOSTICS_PREFIX}nbr_state_limits_step.'):
                idx_nbr_state_limits_step = idx
                break

        # shortcut for relevant indices
        self._index_map = {
            "cpu_time": self._calc_diags_vars_names.index(f'{DIAGNOSTICS_PREFIX}cpu_time'),
            "cpu_time_per_step": self._diags_vars_names.index(f'{DIAGNOSTICS_PREFIX}cpu_time_per_step'),
            "nbr_steps": self._calc_diags_vars_names.index(f'{DIAGNOSTICS_PREFIX}nbr_steps'),
            "event_type": self._diags_vars_names.index(f'{DIAGNOSTICS_PREFIX}event_data.event_info.event_type'),
            "nbr_events": self._calc_diags_vars_names.index(f'{DIAGNOSTICS_PREFIX}nbr_events'),
            "nbr_time_events": self._calc_diags_vars_names.index(f'{DIAGNOSTICS_PREFIX}nbr_time_events'),
            "nbr_state_events": self._calc_diags_vars_names.index(f'{DIAGNOSTICS_PREFIX}nbr_state_events'),
            "nbr_state_limits": idx_nbr_state_limits_step, # index of first variable diags_calc to start with "{DIAGNOSTICS_PREFIX}nbr_state_limits_step."
            "state_errors": idx_state_errors, # index of first variable in diag_vars to start with "{DIAGNOSTICS_PREFIX}nbr_state_limits_step."
        }

    def get_calculated_diagnostics_point(self, diag_data: dict) -> dict:
        """Given a diagnostics data point, return the calculated diagnostics."""
        ret = self._calc_diags_vars_cache.copy()
        # cpu_time
        ret[self._index_map["cpu_time"]] += diag_data[self._index_map["cpu_time_per_step"]]

        # event point
        etype = diag_data[self._index_map["event_type"]]
        # TODO: make these constants?
        if etype == 1:
            ret[self._index_map["nbr_events"]] += 1
            ret[self._index_map["nbr_time_events"]] += 1
        elif etype == 0:
            ret[self._index_map["nbr_events"]] += 1
            ret[self._index_map["nbr_state_events"]] += 1
        else:
            ret[self._index_map["nbr_steps"]] += 1

        # state_errors
        if etype == -1: # no event
            index_diag_data = self._index_map["state_errors"]
            index_calc = self._index_map["nbr_state_limits"]
            for i in range(self._number_states):
                if diag_data[index_diag_data + i] >= 1.0:
                    ret[index_calc + i] = ret[index_calc + i] + 1

        self._update_cache(ret)
        return ret
    
def setup_diagnostics_variables(model, start_time, options, solver_options):
    """ Sets up initial diagnostics data. This function is called before a simulation is initiated. """
    _diagnostics_params = {}
    _diagnostics_vars = {}

    if options.get("logging", False):
        solver_name = options["solver"]

        _diagnostics_params[f"{DIAGNOSTICS_PREFIX}solver.solver_name.{solver_name}"] = (1.0, "Chosen solver.")

        support_state_errors = (solver_name=="CVode" or solver_name=="Radau5ODE")
        support_solver_order = solver_name=="CVode"
        support_elapsed_time = solver_options.get("clock_step", False)

        support_event_indicators = (solver_name=="CVode" or
                                    solver_name=="Radau5ODE" or
                                    solver_name=="LSODAR" or
                                    solver_name=="ImplicitEuler" or
                                    solver_name=="ExplicitEuler"
                                    )

        states_list = model.get_states_list() if isinstance(model, FMUModelME2) else []

        if solver_name != "ExplicitEuler":
            rtol = solver_options.get('rtol', None)
            atol = solver_options.get('atol', None)
            if (rtol is None) or (atol is None):
                rtol, atol = model.get_tolerances()
            
            # is atol is scalar, convert to list
            if isinstance(atol, numbers.Number): 
                atol = [atol]*len(states_list)
            # atol is "pseudoscalar", array/list with single entry; 
            if np.size(atol) == 1:
                # distinction here is need since e.g., np.array(1.) does not support indexing
                if isinstance(atol, np.ndarray):
                    atol = [atol.item()]*len(states_list)
                else: # general iterable, e.g., list
                    atol = [atol[0]]*len(states_list)
                
            _diagnostics_params[f"{DIAGNOSTICS_PREFIX}solver.relative_tolerance"] = (rtol, "Relative solver tolerance.")

            for idx, state in enumerate(states_list):
                _diagnostics_params[f"{DIAGNOSTICS_PREFIX}solver.absolute_tolerance."+state] = (atol[idx], "Absolute solver tolerance for "+state+".")

        _diagnostics_vars[f"{DIAGNOSTICS_PREFIX}step_time"] = (start_time, "Step time")
        if support_elapsed_time:
            _diagnostics_vars[f"{DIAGNOSTICS_PREFIX}cpu_time_per_step"] = (0, "CPU time per step.")
        if support_solver_order:
            _diagnostics_vars[f"{DIAGNOSTICS_PREFIX}solver.solver_order"] = (0.0, "Solver order for CVode used in each time step")
        if support_state_errors:
            for state in states_list:
                _diagnostics_vars[f"{DIAGNOSTICS_PREFIX}state_errors."+state] = (0.0, "State error for "+state+".")
        if support_event_indicators:
            _, nof_ei = model.get_ode_sizes()
            ei_values = model.get_event_indicators() if nof_ei > 0 else []
            for i in range(nof_ei):
                _diagnostics_vars[f"{DIAGNOSTICS_PREFIX}event_data.event_info.indicator_"+str(i+1)] = (ei_values[i], "Value for event indicator {}.".format(i+1))
            for i in range(nof_ei):
                _diagnostics_vars[f"{DIAGNOSTICS_PREFIX}event_data.event_info.state_event_info.index_"+str(i+1)] = (0.0, "Zero crossing indicator for event indicator {}".format(i+1))
            _diagnostics_vars[f"{DIAGNOSTICS_PREFIX}event_data.event_info.event_type"] = (-1, "No event=-1, state event=0, time event=1")

    return _diagnostics_params, _diagnostics_vars
