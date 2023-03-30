#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019-2021 Modelon AB
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

import nose
import os
import numpy as np

from pyfmi.fmi import FMUException, FMUModelME1, __ForTestingFMUModelME1, FMUModelCS1, load_fmu, \
                      FMUModelCS2, FMUModelME2, __ForTestingFMUModelME2

def get_examples_folder():
    return os.path.join(os.path.dirname(__file__), '..', 'examples')

class Dummy_FMUModelME1(__ForTestingFMUModelME1):
    # If true, makes use of the real __ForTesting implementation for nominal_continuous_states,
    # else just returns 1.0 for each.
    override_nominal_continuous_states = True

    #Override properties
    time = None
    continuous_states = None
    _nominal_continuous_states = None

    def __init__(self, states_vref, *args,**kwargs):
        FMUModelME1.__init__(self, *args, **kwargs)

        self.continuous_states = np.zeros(self.get_ode_sizes()[0])
        self._nominal_continuous_states = np.ones(self.get_ode_sizes()[0])
        self.variables = self.get_model_variables(include_alias=False)
        self.states_vref = states_vref

        self.values = {}
        for var in self.variables:
            try:
                start = self.get_variable_start(var)
            except FMUException:
                start = 0.0
            self.values[self.variables[var].value_reference] = start

        for i,vref in enumerate(self.states_vref):
            self.continuous_states[i] = self.values[vref]

    def initialize(self, *args, **kwargs):
        self.time = 0.0
        self.set_allocated_fmu(1)

    def completed_integrator_step(self, *args, **kwargs):
        for i,vref in enumerate(self.states_vref):
            self.values[vref] = self.continuous_states[i]

    def get_derivatives(self):
        return -self.continuous_states

    def get_real(self, vref):
        vals = []
        for v in vref:
            vals.append(self.values[v])
        return np.array(vals)

    def get_integer(self, vref):
        return self.get_real(vref)

    def get_boolean(self, vref):
        return self.get_real(vref)

    def get_state_value_references(self):
        return self.states_vref if self.states_vref else []

    def get_event_indicators(self, *args, **kwargs):
        return np.ones(self.get_ode_sizes()[1])

    def get_nominal_continuous_states_testimpl(self):
        if self.override_nominal_continuous_states:
            return self._nominal_continuous_states
        else:
            return super().nominal_continuous_states

    nominal_continuous_states = property(get_nominal_continuous_states_testimpl)

class Dummy_FMUModelCS1(FMUModelCS1):
    #Override properties
    time = None

    def __init__(self, states_vref, *args,**kwargs):
        FMUModelCS1.__init__(self, *args, **kwargs)

        self.variables = self.get_model_variables(include_alias=False)
        self.states_vref = states_vref

        self.values = {}
        for var in self.variables:
            try:
                start = self.get_variable_start(var)
            except FMUException:
                start = 0.0
            self.values[self.variables[var].value_reference] = start

    def initialize(self, *args, **kwargs):
        self.time = 0.0

    def do_step(self, t, h, new_step=True):
        self.time = t+h
        return 0

    def get_real(self, vref):
        vals = []
        for v in vref:
            vals.append(self.values[v])
        return np.array(vals)

    def get_integer(self, vref):
        return self.get_real(vref)

    def get_boolean(self, vref):
        return self.get_real(vref)

class Dummy_FMUModelCS2(FMUModelCS2):
    #Override properties
    time = None
    continuous_states = None

    def __init__(self, negated_aliases, *args,**kwargs):
        FMUModelCS2.__init__(self, *args, **kwargs)


        self.continuous_states = np.zeros(self.get_ode_sizes()[0])
        self.variables = self.get_model_variables(include_alias=False)
        self.negated_aliases = negated_aliases

        self.values = {}
        for var in self.variables:
            try:
                start = self.get_variable_start(var)
            except FMUException:
                start = 0.0
            self.values[self.variables[var].value_reference] = start
        for alias in self.negated_aliases:
            self.values[self.variables[alias[1]].value_reference] = -self.values[self.variables[alias[0]].value_reference]

        states = self.get_states_list()
        for i,state in enumerate(states):
            self.continuous_states[i] = self.values[states[state].value_reference]

    def setup_experiment(self, *args, **kwargs):
        self.time = 0.0

    def enter_initialization_mode(self, *args, **kwargs):
        self._has_entered_init_mode = True
        return 0

    def exit_initialization_mode(self, *args, **kwargs):
        self._has_entered_init_mode = True
        return 0

    def initialize(self, *args, **kwargs):
        self.enter_initialization_mode()

    def event_update(self, *args, **kwargs):
        pass

    def enter_continuous_time_mode(self, *args, **kwargs):
        pass

    def completed_integrator_step(self, *args, **kwargs):
        states = self.get_states_list()
        for i,state in enumerate(states):
            self.values[states[state].value_reference] = self.continuous_states[i]
        for alias in self.negated_aliases:
            self.values[self.variables[alias[1]].value_reference] = -self.values[self.variables[alias[0]].value_reference]
        return [False, False]

    def do_step(self, current_t, step_size, new_step=True):
        self.continuous_states = np.exp(-self.continuous_states*(current_t+step_size))
        self.completed_integrator_step()
        return 0

    def get_real(self, vref):
        vals = []
        for v in vref:
            vals.append(self.values[v])
        return np.array(vals).astype(float)

    def get_integer(self, vref):
        return self.get_real(vref)

    def get_boolean(self, vref):
        return self.get_real(vref)

    def set_real(self, vref, values):
        for i,v in enumerate(vref):
            self.values[v] = values[i]

    def set_integer(self, vref, values):
        for i,v in enumerate(vref):
            self.values[v] = values[i]

class Dummy_FMUModelME2(__ForTestingFMUModelME2):

    # -- Test options --
    
    # If true, makes use of the real __ForTesting implementation for nominal_continuous_states,
    # else just returns 1.0 for each.
    override_nominal_continuous_states = True

    # Values for nominal_continuous_states. 'test_sparse_option()' and more tests will break
    # if the values are not calculated from get_ode_sizes() as defined at __init__.
    _nominal_continuous_states = None

    #Override properties
    time = None
    continuous_states = None
    

    def __init__(self, negated_aliases, *args, **kwargs):
        FMUModelME2.__init__(self, *args, **kwargs)

        self.continuous_states = np.zeros(self.get_ode_sizes()[0])
        self.variables = self.get_model_variables(include_alias=False)
        self.negated_aliases = negated_aliases
        self.states = self.get_states_list()
        self._nominal_continuous_states = np.ones(self.get_ode_sizes()[0])

        self.reset()

    def reset(self, *args, **kwargs):
        self.values = {}
        for var in self.variables:
            try:
                start = self.get_variable_start(var)
            except FMUException:
                start = 0.0
            self.values[self.variables[var].value_reference] = start
        for alias in self.negated_aliases:
            self.values[self.variables[alias[1]].value_reference] = -self.values[self.variables[alias[0]].value_reference]

        states = self.states
        for i,state in enumerate(states):
            self.continuous_states[i] = self.values[states[state].value_reference]

        self.set_initialized_fmu(0)

    def setup_experiment(self, *args, **kwargs):
        self.time = 0.0

    def enter_initialization_mode(self, *args, **kwargs):
        self._has_entered_init_mode = True
        return 0

    def exit_initialization_mode(self, *args, **kwargs):
        self._has_entered_init_mode = True
        self.set_initialized_fmu(1)
        return 0

    def initialize(self, *args, **kwargs):
        self.enter_initialization_mode()
        self.exit_initialization_mode()

    def event_update(self, *args, **kwargs):
        pass

    def enter_continuous_time_mode(self, *args, **kwargs):
        pass

    def completed_integrator_step(self, *args, **kwargs):
        states = self.states
        for i,state in enumerate(states):
            self.values[states[state].value_reference] = self.continuous_states[i]
        for alias in self.negated_aliases:
            self.values[self.variables[alias[1]].value_reference] = -self.values[self.variables[alias[0]].value_reference]
        return [False, False]

    def get_derivatives(self):
        return -self.continuous_states

    def get_real(self, vref):
        self.get_derivatives()
        vals = []
        if type(vref) == int:
            vref = [vref]
        for v in vref:
            vals.append(self.values[v])
        return np.array(vals).astype(float)

    def set_real(self, vref, values):
        for i,v in enumerate(vref):
            self.values[v] = values[i]

    def get_integer(self, vref):
        return self.get_real(vref).astype(np.int32)

    def get_boolean(self, vref):
        return self.get_real(vref)

    def get_event_indicators(self, *args, **kwargs):
        return np.ones(self.get_ode_sizes()[1])

    def get_nominal_continuous_states_testimpl(self):
        if self.override_nominal_continuous_states:
            return self._nominal_continuous_states
        else:
            return super().nominal_continuous_states

    nominal_continuous_states = property(get_nominal_continuous_states_testimpl)