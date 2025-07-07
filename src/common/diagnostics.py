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

class DiagnosticsBase: # TODO: Future, possible deprecate & remove this?
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

class DynamicDiagnosticsUtils:
    """Utility functionality for additional diagnostics variables calculated from 
    diagnostics data produced from using 'dynamic_diagnostics'.
    
    This class contains functionality to explicitly compute and store these on a 
    per-point basis and class-methods to perform the same calculations for trajectories.
    
    Minimal example for result handler using this class:

    class ResultStoreCalcDiagnostics(ResultHandler):
        def __init__(self, model = None):
            super().__init__(model)
            self.supports['dynamic_diagnostics'] = True
            self._diags_util = DynamicDiagnosticsUtils()
        def simulation_start(self, diagnostics_params = {}, diagnostics_vars = {}):
            calc_diag_vars = self._diags_util.prepare_calculated_diagnostics(diagnostics_vars)
            # ... setup appropriate data containers
        def diagnostics_point(self, diag_data):
            calculated_diags = self._diags_util.get_calculated_diagnostics_point(diag_data)
            # ... store diagnostics data
        # ...
    """
    def __init__(self):
        self._calc_diags_vars_cache: np.ndarray = np.array([])
    
    def prepare_calculated_diagnostics(self, diagnostics_vars: dict) -> dict:
        """
        Get calculated diagnostics variable names, start_values and descriptions.
        Also sets up some internal mappings used by get_calculated_diagnostics_point.

        Parameters::

            diagnostics_vars --
                dict, 'diagnostics_vars' input given to ResultHandler.simulation_start(). 
                The order of keys here determined the output order in 'get_calculated_diagnostics_point'.

        Returns::

            dict: {calculated_diagnostics_name: (start_value, description)}
        """
        # Fixed variables
        calc_diags = {
            f"{DIAGNOSTICS_PREFIX}cpu_time"        : (0.0, "Cumulative CPU time"),
            f"{DIAGNOSTICS_PREFIX}nbr_events"      : (0, "Cumulative number of events"),
            f"{DIAGNOSTICS_PREFIX}nbr_time_events" : (0, "Cumulative number of time events"),
            f"{DIAGNOSTICS_PREFIX}nbr_state_events": (0, "Cumulative number of state events"),
            f"{DIAGNOSTICS_PREFIX}nbr_steps"       : (0, "Cumulative number of steps"),
        }

        diagnostics_vars_names = list(diagnostics_vars.keys())

        # based on states
        description = "Cumulative number of times '{}' limited the step-size."
        start_value = 0
        self._number_states = 0
        prefix = f'{DIAGNOSTICS_PREFIX}state_errors.'
        for v in diagnostics_vars_names:
            if v.startswith(prefix):
                state_name = v[len(prefix):]
                calc_diags[f"{DIAGNOSTICS_PREFIX}nbr_state_limits_step.{state_name}"] = (start_value, description.format(state_name))
                self._number_states += 1

        # index maps for calculating diagnostics variables
        calc_diags_names = list(calc_diags.keys())
        self._idx_map_calc_diags = {
            "cpu_time":         calc_diags_names.index(f'{DIAGNOSTICS_PREFIX}cpu_time'),
            "nbr_events":       calc_diags_names.index(f'{DIAGNOSTICS_PREFIX}nbr_events'),
            "nbr_time_events":  calc_diags_names.index(f'{DIAGNOSTICS_PREFIX}nbr_time_events'),
            "nbr_state_events": calc_diags_names.index(f'{DIAGNOSTICS_PREFIX}nbr_state_events'),
            "nbr_steps":        calc_diags_names.index(f'{DIAGNOSTICS_PREFIX}nbr_steps'),
            "nbr_state_limits": len(calc_diags_names) - self._number_states,
        }

        idx_state_errors = None
        for idx, key in enumerate(diagnostics_vars):
            if key.startswith(f"{DIAGNOSTICS_PREFIX}state_errors."):
                idx_state_errors = idx
                break

        idx_cpu_time_per_step = None
        if f'{DIAGNOSTICS_PREFIX}cpu_time_per_step' in diagnostics_vars_names:
            idx_cpu_time_per_step = diagnostics_vars_names.index(f'{DIAGNOSTICS_PREFIX}cpu_time_per_step')

        idx_event_type = None
        if f'{DIAGNOSTICS_PREFIX}event_data.event_info.event_type' in diagnostics_vars_names:
            idx_event_type = diagnostics_vars_names.index(f'{DIAGNOSTICS_PREFIX}event_data.event_info.event_type')

        self._idx_map_diags = {
            "cpu_time_per_step": idx_cpu_time_per_step,
            "event_type": idx_event_type,
            "state_errors": idx_state_errors, # index of first 'state_errors' variable in diag_vars
        }

        self.calc_diags_vars_cache = [v[0] for v in calc_diags.values()]
        return calc_diags
    
    def _get_calc_diags_vars_cache(self) -> np.ndarray:
        return self._calc_diags_vars_cache.copy()

    def _set_calc_diags_vars_cache(self, val: np.ndarray) -> None:
        self._calc_diags_vars_cache = val.copy()
    
    calc_diags_vars_cache = property(
        fget = _get_calc_diags_vars_cache, 
        fset = _set_calc_diags_vars_cache
    )

    def get_calculated_diagnostics_point(self, diag_data: np.ndarray) -> np.ndarray:
        """
        Given a diagnostics_point data for a single time-point, return the calculated diagnostics.
        Automatically caches the previous result for computation of calculated diagnostics 
        that are cumulative. 

        Parameters::

            diag_data --
                numpy.ndarray of input received via ResultHandler.diagnostics_point()

        Returns::

            numpy.ndarray of calculated diagnostic variables.
        """
        ret = self.calc_diags_vars_cache
        # cpu_time
        if self._idx_map_diags["cpu_time_per_step"] is not None:
            ret[self._idx_map_calc_diags["cpu_time"]] += diag_data[self._idx_map_diags["cpu_time_per_step"]]

        if self._idx_map_diags["event_type"] is not None:
            # event point
            etype = diag_data[self._idx_map_diags["event_type"]]
            if etype == 1:
                ret[self._idx_map_calc_diags["nbr_events"]] += 1
                ret[self._idx_map_calc_diags["nbr_time_events"]] += 1
            elif etype == 0:
                ret[self._idx_map_calc_diags["nbr_events"]] += 1
                ret[self._idx_map_calc_diags["nbr_state_events"]] += 1
            else:
                ret[self._idx_map_calc_diags["nbr_steps"]] += 1

            # state_errors
            if etype == -1: # no event
                index_diag_data = self._idx_map_diags["state_errors"]
                index_calc = self._idx_map_calc_diags["nbr_state_limits"]
                for i in range(self._number_states):
                    if diag_data[index_diag_data + i] >= 1.0:
                        ret[index_calc + i] = ret[index_calc + i] + 1

        self.calc_diags_vars_cache = ret
        return ret
    
    @classmethod
    def get_cpu_time(cls, cpu_time_per_step: np.ndarray) -> np.ndarray:
        """Given cpu_time_per_step, return cumulative CPU time."""
        return np.cumsum(cpu_time_per_step)
    
    @classmethod
    def get_events_and_steps(cls, event_type_data: np.ndarray) -> dict:
        f"""Given event_type_data trajectory, return dictionary of cumulative:
            '{DIAGNOSTICS_PREFIX}nbr_events',
            '{DIAGNOSTICS_PREFIX}nbr_time_events',
            '{DIAGNOSTICS_PREFIX}nbr_state_events',
            '{DIAGNOSTICS_PREFIX}nbr_steps'
        """
        return {
            f"{DIAGNOSTICS_PREFIX}nbr_events":       np.cumsum(event_type_data != -1),
            f"{DIAGNOSTICS_PREFIX}nbr_time_events":  np.cumsum(event_type_data == 1),
            f"{DIAGNOSTICS_PREFIX}nbr_state_events": np.cumsum(event_type_data == 0),
            f"{DIAGNOSTICS_PREFIX}nbr_steps":        np.concatenate(([0], np.cumsum(event_type_data[1:] == -1))), # do not count the first point
        }
    
    @classmethod
    def get_nbr_state_limits(cls, event_type_data: np.ndarray, state_error: np.ndarray) -> np.ndarray:
        """Given event_type_data trajectory, return the cumulative number of times
        the (normalized) state_error exceeded 1 (= limited step-size)."""
        return np.cumsum((event_type_data == -1) * (state_error >= 1.0))
    
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
