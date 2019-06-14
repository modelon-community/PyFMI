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


from pyfmi import testattr
from pyfmi.fmi import FMUModel, FMUException, FMUModelME1, FMUModelCS1, load_fmu, FMUModelCS2, FMUModelME2
from pyfmi.common.io import ResultDymolaTextual, ResultDymolaBinary, ResultWriterDymola, JIOError, ResultHandlerCSV, ResultCSVTextual, ResultHandlerBinaryFile, ResultHandlerFile
import pyfmi.fmi_util as fmi_util
import pyfmi.fmi as fmi

file_path = os.path.dirname(os.path.abspath(__file__))

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

class Dummy_FMUModelME2(FMUModelME2):
    #Override properties
    time = None
    continuous_states = None
    nominal_continuous_states = None
    
    def __init__(self, negated_aliases, *args,**kwargs):
        FMUModelME2.__init__(self, *args, **kwargs)
    
        self.time = 0.0
        self.continuous_states = np.zeros(self.get_ode_sizes()[0])
        self.nominal_continuous_states = np.ones(self.get_ode_sizes()[0])
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
    
    def get_event_indicators(self, *args, **kwargs):
        return np.ones(self.get_ode_sizes()[1])

class TestResultFileText:
    
    @testattr(stddist = True)
    def test_get_description(self):
        model = Dummy_FMUModelME1([], "CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        
        result_writer = ResultHandlerFile(model)
        result_writer.set_options(model.simulate_options())
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.simulation_end()
        
        res = ResultDymolaTextual('CoupledClutches_result.txt')
        
        assert res.description[res.get_variable_index("J1.phi")] == "Absolute rotation angle of component"
    
    @testattr(stddist = True)
    def test_read_alias_derivative(self):
        simple_alias = Dummy_FMUModelME2([], "Alias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        opts = simple_alias.simulate_options()
        opts["result_handling"] = "file"

        res = simple_alias.simulate(options=opts)
        
        derx = res["der(x)"]
        dery = res["der(y)"]
        
        for i in range(len(derx)):
            nose.tools.assert_equal(derx[i], dery[i])
    
    @testattr(stddist = True)
    def test_read_all_variables_using_model_variables(self):
        simple_alias = Dummy_FMUModelME2([("x", "y")], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        opts = simple_alias.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerFile(simple_alias)
        
        res = simple_alias.simulate(options=opts)
        
        for var in simple_alias.get_model_variables():
            res[var]
    
    @testattr(stddist = True)
    def test_correct_file_after_simulation_failure(self):
        simple_alias = Dummy_FMUModelME2([("x", "y")], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        def f(*args, **kwargs):
            if simple_alias.time > 0.5:
                raise Exception
            return -simple_alias.continuous_states
        
        simple_alias.get_derivatives = f
        
        opts = simple_alias.simulate_options()
        opts["result_handling"] = "file"
        opts["solver"] = "ExplicitEuler"
        
        successful_simulation = False
        try:
            res = simple_alias.simulate(options=opts)
            successful_simulation = True #The above simulation should fail...
        except:
            pass
        
        if successful_simulation:
            raise Exception
            
        result = ResultDymolaTextual("NegatedAlias_result.txt")
        
        x = result.get_variable_data("x").x
        y = result.get_variable_data("y").x
        
        assert len(x) > 2
        
        for i in range(len(x)):
            nose.tools.assert_equal(x[i], -y[i])
    
    @testattr(stddist = True)
    def test_work_flow_me1(self):
        """Tests the work flow of write_header, write_point, write_finalize."""
        model = Dummy_FMUModelME1([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        
        bouncingBall = ResultHandlerFile(model)
        
        bouncingBall.set_options(model.simulate_options())
        bouncingBall.simulation_start()
        bouncingBall.initialize_complete()
        bouncingBall.integration_point()
        bouncingBall.simulation_end()
        
        res = ResultDymolaTextual('bouncingBall_result.txt')
        
        h = res.get_variable_data('h')
        derh = res.get_variable_data('der(h)')
        g = res.get_variable_data('g')

        nose.tools.assert_almost_equal(h.x, 1.000000, 5)
        nose.tools.assert_almost_equal(derh.x, 0.000000, 5)
    
    @testattr(stddist = True)
    def test_enumeration_file(self):
        
        model = Dummy_FMUModelME2([], "Friction2.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        data_type = model.get_variable_data_type("mode")
        
        assert data_type == fmi.FMI2_ENUMERATION
        
        opts = model.simulate_options()
        opts["result_handling"] = "file"
        
        res = model.simulate(options=opts)
        res["mode"] #Check that the enumeration variable is in the dict, otherwise exception
    
    @testattr(stddist = True)
    def test_work_flow_me2(self):
        """Tests the work flow of write_header, write_point, write_finalize."""
        model = Dummy_FMUModelME2([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        bouncingBall = ResultHandlerFile(model)
        
        bouncingBall.set_options(model.simulate_options())
        bouncingBall.simulation_start()
        bouncingBall.initialize_complete()
        bouncingBall.integration_point()
        bouncingBall.simulation_end()
        
        res = ResultDymolaTextual('bouncingBall_result.txt')
        
        h = res.get_variable_data('h')
        derh = res.get_variable_data('der(h)')
        g = res.get_variable_data('g')

        nose.tools.assert_almost_equal(h.x, 1.000000, 5)
        nose.tools.assert_almost_equal(derh.x, 0.000000, 5)

class TestResultMemory:
    def _run_negated_alias(self, model):
        opts = model.simulate_options()
        opts["result_handling"] = "memory"
        
        res = model.simulate(options=opts)
        
        # test that res['y'] returns a vector of the same length as the time
        # vector
        nose.tools.assert_equal(len(res['y']),len(res['time']), 
            "Wrong size of result vector.")
            
        x = res["x"]
        y = res["y"]
        
        for i in range(len(x)):
            nose.tools.assert_equal(x[i], -y[i])

    @testattr(stddist = True)
    def test_memory_options_me1(self):
        simple_alias = Dummy_FMUModelME1([40], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        self._run_negated_alias(simple_alias)
    
    @testattr(stddist = True)
    def test_memory_options_me2(self):
        simple_alias = Dummy_FMUModelME2([("x", "y")], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        self._run_negated_alias(simple_alias)
    
    @testattr(stddist = True)
    def test_only_parameters(self):
        model = Dummy_FMUModelME2([], "ParameterAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        opts = model.simulate_options()
        opts["result_handling"] = "memory"
        opts["filter"] = "p2"
        
        res = model.simulate(options=opts)
        
        nose.tools.assert_almost_equal(3.0, res["p2"][0])
        assert not isinstance(res.initial("p2"), np.ndarray)
        assert not isinstance(res.final("p2"), np.ndarray)
    
    @testattr(stddist = True)
    def test_enumeration_memory(self):
        
        model = Dummy_FMUModelME2([], "Friction2.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        data_type = model.get_variable_data_type("mode")
        
        assert data_type == fmi.FMI2_ENUMERATION
        
        opts = model.simulate_options()
        opts["result_handling"] = "memory"
        
        res = model.simulate(options=opts)
        res["mode"] #Check that the enumeration variable is in the dict, otherwise exception
    
class TestResultFileBinary:
    
    @testattr(stddist = True)
    def test_integer_start_time(self):
        model = Dummy_FMUModelME2([], "Alias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        opts = model.simulate_options()
        opts["result_handling"] = "binary"

        #Assert that there is no exception when reloading the file
        res = model.simulate(start_time=0, options=opts)
    
    @testattr(stddist = True)
    def test_get_description(self):
        model = Dummy_FMUModelME1([], "CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        
        result_writer = ResultHandlerBinaryFile(model)
        result_writer.set_options(model.simulate_options())
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.simulation_end()
        
        res = ResultDymolaBinary('CoupledClutches_result.mat')
        
        assert res.description[res.get_variable_index("J1.phi")] == "Absolute rotation angle of component"
    
    @testattr(stddist = True)
    def test_read_alias_derivative(self):
        simple_alias = Dummy_FMUModelME2([], "Alias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        opts = simple_alias.simulate_options()
        opts["result_handling"] = "binary"

        res = simple_alias.simulate(options=opts)
        
        derx = res["der(x)"]
        dery = res["der(y)"]
        
        for i in range(len(derx)):
            nose.tools.assert_equal(derx[i], dery[i])
    
    @testattr(stddist = True)
    def test_read_all_variables(self):
        res = ResultDymolaBinary(os.path.join(file_path, "files", "Results", "DoublePendulum.mat"))
        
        for var in res.name:
            res.get_variable_data(var)
    
    @testattr(stddist = True)
    def test_read_all_variables_using_model_variables(self):
        simple_alias = Dummy_FMUModelME2([("x", "y")], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        opts = simple_alias.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerBinaryFile(simple_alias)
        
        res = simple_alias.simulate(options=opts)
        
        for var in simple_alias.get_model_variables():
            res[var]
    
    @testattr(stddist = True)
    def test_correct_file_after_simulation_failure(self):
        simple_alias = Dummy_FMUModelME2([("x", "y")], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        def f(*args, **kwargs):
            if simple_alias.time > 0.5:
                raise Exception
            return -simple_alias.continuous_states
        
        simple_alias.get_derivatives = f
        
        opts = simple_alias.simulate_options()
        opts["result_handling"] = "binary"
        opts["solver"] = "ExplicitEuler"
        
        successful_simulation = False
        try:
            res = simple_alias.simulate(options=opts)
            successful_simulation = True #The above simulation should fail...
        except:
            pass
        
        if successful_simulation:
            raise Exception
            
        result = ResultDymolaBinary("NegatedAlias_result.mat")
        
        x = result.get_variable_data("x").x
        y = result.get_variable_data("y").x
        
        assert len(x) > 2
        
        for i in range(len(x)):
            nose.tools.assert_equal(x[i], -y[i])

    @testattr(stddist = True)
    def test_variable_alias_custom_handler(self):
        simple_alias = Dummy_FMUModelME2([("x", "y")], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        opts = simple_alias.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerBinaryFile(simple_alias)
        
        res = simple_alias.simulate(options=opts)
        
        # test that res['y'] returns a vector of the same length as the time
        # vector
        nose.tools.assert_equal(len(res['y']),len(res['time']), 
            "Wrong size of result vector.")
            
        x = res["x"]
        y = res["y"]
        
        for i in range(len(x)):
            nose.tools.assert_equal(x[i], -y[i])
    
    @testattr(stddist = True)
    def test_work_flow_me1(self):
        """Tests the work flow of write_header, write_point, write_finalize."""
        model = Dummy_FMUModelME1([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        
        bouncingBall = ResultHandlerBinaryFile(model)
        
        bouncingBall.set_options(model.simulate_options())
        bouncingBall.simulation_start()
        bouncingBall.initialize_complete()
        bouncingBall.integration_point()
        bouncingBall.simulation_end()
        
        res = ResultDymolaBinary('bouncingBall_result.mat')
        
        h = res.get_variable_data('h')
        derh = res.get_variable_data('der(h)')
        g = res.get_variable_data('g')

        nose.tools.assert_almost_equal(h.x, 1.000000, 5)
        nose.tools.assert_almost_equal(derh.x, 0.000000, 5)
    
    
    
    @testattr(stddist = True)
    def test_work_flow_me2(self):
        """Tests the work flow of write_header, write_point, write_finalize."""
        model = Dummy_FMUModelME2([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        bouncingBall = ResultHandlerBinaryFile(model)
        
        bouncingBall.set_options(model.simulate_options())
        bouncingBall.simulation_start()
        bouncingBall.initialize_complete()
        bouncingBall.integration_point()
        bouncingBall.simulation_end()
        
        res = ResultDymolaBinary('bouncingBall_result.mat')
        
        h = res.get_variable_data('h')
        derh = res.get_variable_data('der(h)')
        g = res.get_variable_data('g')

        nose.tools.assert_almost_equal(h.x, 1.000000, 5)
        nose.tools.assert_almost_equal(derh.x, 0.000000, 5)
    
    def _run_negated_alias(self, model):
        opts = model.simulate_options()
        opts["result_handling"] = "binary"
        
        res = model.simulate(options=opts)
        
        # test that res['y'] returns a vector of the same length as the time
        # vector
        nose.tools.assert_equal(len(res['y']),len(res['time']), 
            "Wrong size of result vector.")
            
        x = res["x"]
        y = res["y"]
        
        for i in range(len(x)):
            nose.tools.assert_equal(x[i], -y[i])

    @testattr(stddist = True)
    def test_binary_options_me1(self):
        simple_alias = Dummy_FMUModelME1([40], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        self._run_negated_alias(simple_alias)
    
    @testattr(stddist = True)
    def test_binary_options_me2(self):
        simple_alias = Dummy_FMUModelME2([("x", "y")], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        self._run_negated_alias(simple_alias)
    
    @testattr(stddist = True)
    def test_only_parameters(self):
        model = Dummy_FMUModelME2([], "ParameterAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        opts = model.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerBinaryFile(model)
        opts["filter"] = "p2"
        
        res = model.simulate(options=opts)
        
        nose.tools.assert_almost_equal(3.0, res["p2"][0])
    
    @testattr(stddist = True)
    def test_enumeration_binary(self):
        
        model = Dummy_FMUModelME2([], "Friction2.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        data_type = model.get_variable_data_type("mode")
        
        assert data_type == fmi.FMI2_ENUMERATION
        
        opts = model.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerBinaryFile(model)
        
        res = model.simulate(options=opts)
        res["mode"] #Check that the enumeration variable is in the dict, otherwise exception
    
class TestResultCSVTextual:
    
    @testattr(stddist = True)
    def test_enumeration_csv(self):
        
        model = Dummy_FMUModelME2([], "Friction2.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        data_type = model.get_variable_data_type("mode")
        
        assert data_type == fmi.FMI2_ENUMERATION
        
        opts = model.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerCSV(model)
        
        res = model.simulate(options=opts)
        res["mode"] #Check that the enumeration variable is in the dict, otherwise exception

    @testattr(stddist = True)
    def test_only_parameters(self):
        model = Dummy_FMUModelME2([], "ParameterAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        opts = model.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerCSV(model)
        opts["filter"] = "p2"
        
        res = model.simulate(options=opts)
        
        nose.tools.assert_almost_equal(3.0, res["p2"][0])
    
    @testattr(stddist = True)
    def test_delimiter(self):
        
        res = ResultCSVTextual(os.path.join(file_path, 'files', 'Results', 'TestCSV.csv'), delimiter=",")
        
        x = res.get_variable_data("fd.y")
        
        assert x.x[-1] == 1
    
    @testattr(stddist = True)
    def test_work_flow_me1(self):
        """Tests the work flow of write_header, write_point, write_finalize."""
        model = Dummy_FMUModelME1([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        
        bouncingBall = ResultHandlerCSV(model)
        
        bouncingBall.set_options(model.simulate_options())
        bouncingBall.simulation_start()
        bouncingBall.initialize_complete()
        bouncingBall.integration_point()
        bouncingBall.simulation_end()
        
        res = ResultCSVTextual('bouncingBall_result.csv')
        
        h = res.get_variable_data('h')
        derh = res.get_variable_data('der(h)')
        g = res.get_variable_data('g')

        nose.tools.assert_almost_equal(h.x, 1.000000, 5)
        nose.tools.assert_almost_equal(derh.x, 0.000000, 5)
    
    @testattr(stddist = True)
    def test_work_flow_me2(self):
        """Tests the work flow of write_header, write_point, write_finalize."""
        model = Dummy_FMUModelME2([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        bouncingBall = ResultHandlerCSV(model)
        
        bouncingBall.set_options(model.simulate_options())
        bouncingBall.simulation_start()
        bouncingBall.initialize_complete()
        bouncingBall.integration_point()
        bouncingBall.simulation_end()
        
        res = ResultCSVTextual('bouncingBall_result.csv')
        
        h = res.get_variable_data('h')
        derh = res.get_variable_data('der(h)')
        g = res.get_variable_data('g')

        nose.tools.assert_almost_equal(h.x, 1.000000, 5)
        nose.tools.assert_almost_equal(derh.x, 0.000000, 5)
    
    @testattr(stddist = True)
    def test_variable_alias_custom_handler(self):

        simple_alias = Dummy_FMUModelME1([40], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        
        opts = simple_alias.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerCSV(simple_alias)
        
        res = simple_alias.simulate(options=opts)
        
        # test that res['y'] returns a vector of the same length as the time
        # vector
        nose.tools.assert_equal(len(res['y']),len(res['time']), 
            "Wrong size of result vector.")
            
        x = res["x"]
        y = res["y"]
        
        for i in range(len(x)):
            nose.tools.assert_equal(x[i], -y[i])
    
    def _run_negated_alias(self, model):
        opts = model.simulate_options()
        opts["result_handling"] = "csv"
        
        res = model.simulate(options=opts)
        
        # test that res['y'] returns a vector of the same length as the time
        # vector
        nose.tools.assert_equal(len(res['y']),len(res['time']), 
            "Wrong size of result vector.")
            
        x = res["x"]
        y = res["y"]
        
        for i in range(len(x)):
            nose.tools.assert_equal(x[i], -y[i])
        
    
    @testattr(stddist = True)
    def test_csv_options_me1(self):
        simple_alias = Dummy_FMUModelME1([40], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        self._run_negated_alias(simple_alias)
    
    @testattr(stddist = True)
    def test_csv_options_me2(self):
        simple_alias = Dummy_FMUModelME2([("x", "y")], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        self._run_negated_alias(simple_alias)
    """
    @testattr(stddist = True)
    def test_csv_options_cs1(self):
        simple_alias = Dummy_FMUModelCS1([40], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS1.0"), _connect_dll=False)
        self._run_negated_alias(self, simple_alias)
    
    @testattr(stddist = True)
    def test_csv_options_cs2(self):
        simple_alias = Dummy_FMUModelCS2([("x", "y")], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS2.0"), _connect_dll=False)
        self._run_negated_alias(self, simple_alias)
    """
