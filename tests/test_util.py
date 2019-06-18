#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Modelon AB
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

from pyfmi.fmi import FMUModel, FMUException, FMUModelME1, FMUModelCS1, load_fmu, FMUModelCS2, FMUModelME2

class Dummy_FMUModelME1(FMUModelME1):
    #Override properties
    time = None
    continuous_states = None
    nominal_continuous_states = None
    
    def __init__(self, states_vref, *args,**kwargs):
        FMUModelME1.__init__(self, *args, **kwargs)
    
        self.time = 0.0
        self.continuous_states = np.zeros(self.get_ode_sizes()[0])
        self.nominal_continuous_states = np.ones(self.get_ode_sizes()[0])
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
        pass
    
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

class Dummy_FMUModelCS1(FMUModelCS1):
    #Override properties
    time = None
    
    def __init__(self, states_vref, *args,**kwargs):
        FMUModelCS1.__init__(self, *args, **kwargs)
    
        self.time = 0.0
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
        pass
    
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
    
        self.time = 0.0
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
        pass
    
    def initialize(self, *args, **kwargs):
        pass
    
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
        return np.array(vals)
    
    def get_integer(self, vref):
        return self.get_real(vref)
    
    def get_boolean(self, vref):
        return self.get_real(vref)

class Dummy_FMUModelME2(FMUModelME2):
    #Override properties
    time = None
    continuous_states = None
    nominal_continuous_states = None
    
    def __init__(self, negated_aliases, *args,**kwargs):
        FMUModelME2.__init__(self, *args, **kwargs)
    
        self.continuous_states = np.zeros(self.get_ode_sizes()[0])
        self.nominal_continuous_states = np.ones(self.get_ode_sizes()[0])
        self.variables = self.get_model_variables(include_alias=False)
        self.negated_aliases = negated_aliases
        
        self.reset()
    
    def reset(self, *args, **kwargs):
        self.time = 0.0
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
        pass
    
    def initialize(self, *args, **kwargs):
        self._has_entered_init_mode = True
    
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
    
    def get_derivatives(self):
        return -self.continuous_states
        
    def get_real(self, vref):
        self.get_derivatives()
        vals = []
        for v in vref:
            vals.append(self.values[v])
        return np.array(vals)
    
    def set_real(self, vref, values):
        for i,v in enumerate(vref):
            self.values[v] = values[i]
    
    def get_integer(self, vref):
        return self.get_real(vref)
    
    def get_boolean(self, vref):
        return self.get_real(vref)
    
    def get_event_indicators(self, *args, **kwargs):
        return np.ones(self.get_ode_sizes()[1])
