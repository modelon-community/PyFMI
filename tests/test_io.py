#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Modelon AB
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

import nose
import os
import numpy as np


from pyfmi import testattr
from pyfmi.fmi import FMUModel, FMUException, FMUModelME1, FMUModelCS1, load_fmu, FMUModelCS2, FMUModelME2
from pyfmi.common.io import ResultDymolaTextual, ResultDymolaBinary, ResultWriterDymola, JIOError, ResultHandlerCSV, ResultCSVTextual, ResultHandlerBinaryFile, ResultHandlerFile
import pyfmi.fmi_util as fmi_util
import pyfmi.fmi as fmi
from pyfmi.tests.test_util import Dummy_FMUModelCS1, Dummy_FMUModelME1, Dummy_FMUModelME2, Dummy_FMUModelCS2

file_path = os.path.dirname(os.path.abspath(__file__))

assimulo_installed = True
try:
    import assimulo
except ImportError:
    assimulo_installed = False

def _run_negated_alias(model, result_type):
    opts = model.simulate_options()
    opts["result_handling"] = result_type
    
    res = model.simulate(options=opts)
    
    # test that res['y'] returns a vector of the same length as the time
    # vector
    nose.tools.assert_equal(len(res['y']),len(res['time']), 
        "Wrong size of result vector.")
        
    x = res["x"]
    y = res["y"]
    
    for i in range(len(x)):
        nose.tools.assert_equal(x[i], -y[i])

if assimulo_installed:
    class TestResultFileText_Simulation:
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
        def test_read_all_variables_using_model_variables(self):
            simple_alias = Dummy_FMUModelME2([("x", "y")], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            
            opts = simple_alias.simulate_options()
            opts["result_handling"] = "custom"
            opts["result_handler"] = ResultHandlerFile(simple_alias)
            
            res = simple_alias.simulate(options=opts)
            
            for var in simple_alias.get_model_variables():
                res[var]
                
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
        def test_no_variables(self):
            model = Dummy_FMUModelME2([], "ParameterAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            
            opts = model.simulate_options()
            opts["result_handling"] = "file"
            opts["result_file_name"] = "NoMatchingTest.txt"
            opts["filter"] = "NoMatchingVariables"
            
            res = model.simulate(options=opts)
            
            nose.tools.assert_almost_equal(1.0, res["time"][-1])
                
        @testattr(stddist = True)
        def test_enumeration_file(self):
            
            model = Dummy_FMUModelME2([], "Friction2.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            data_type = model.get_variable_data_type("mode")
            
            assert data_type == fmi.FMI2_ENUMERATION
            
            opts = model.simulate_options()
            opts["result_handling"] = "file"
            
            res = model.simulate(options=opts)
            res["mode"] #Check that the enumeration variable is in the dict, otherwise exception

class TestResultFileText:
    
    @testattr(stddist = True)
    def test_get_description(self):
        model = Dummy_FMUModelME1([], "CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        model.initialize()
        
        result_writer = ResultHandlerFile(model)
        result_writer.set_options(model.simulate_options())
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.simulation_end()
        
        res = ResultDymolaTextual('CoupledClutches_result.txt')
        
        assert res.description[res.get_variable_index("J1.phi")] == "Absolute rotation angle of component"
    
    @testattr(stddist = True)
    def test_description_not_stored(self):
        model = Dummy_FMUModelME1([], "CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        model.initialize()
        
        opts = model.simulate_options()
        opts["result_store_variable_description"] = False
        
        result_writer = ResultHandlerFile(model)
        result_writer.set_options(opts)
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.simulation_end()
        
        res = ResultDymolaTextual('CoupledClutches_result.txt')
        
        assert res.description[res.get_variable_index("J1.phi")] == "", "Description is not empty, " + res.description[res.get_variable_index("J1.phi")]
    
    @testattr(stddist = True)
    def test_get_description_unicode(self):
        model = Dummy_FMUModelME1([], "Description.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        model.initialize()
        
        result_writer = ResultHandlerFile(model)
        result_writer.set_options(model.simulate_options())
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.simulation_end()
        
        res = ResultDymolaTextual('Description_result.txt')
        desc = res.description[res.get_variable_index("x")] 

        assert desc == u"Test symbols '' ‘’"
    
    @testattr(stddist = True)
    def test_work_flow_me1(self):
        model = Dummy_FMUModelME1([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        model.initialize()
        
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
    def test_work_flow_me2(self):
        """Tests the work flow of write_header, write_point, write_finalize."""
        model = Dummy_FMUModelME2([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()
        
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

if assimulo_installed:
    class TestResultMemory_Simulation:
        @testattr(stddist = True)
        def test_memory_options_me1(self):
            simple_alias = Dummy_FMUModelME1([40], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
            _run_negated_alias(simple_alias, "memory")
        
        @testattr(stddist = True)
        def test_memory_options_me2(self):
            simple_alias = Dummy_FMUModelME2([("x", "y")], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            _run_negated_alias(simple_alias, "memory")
        
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
        def test_no_variables(self):
            model = Dummy_FMUModelME2([], "ParameterAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            
            opts = model.simulate_options()
            opts["result_handling"] = "memory"
            opts["filter"] = "NoMatchingVariables"
            
            res = model.simulate(options=opts)
            
            nose.tools.assert_almost_equal(1.0, res["time"][-1])
        
        @testattr(stddist = True)
        def test_enumeration_memory(self):
            
            model = Dummy_FMUModelME2([], "Friction2.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            data_type = model.get_variable_data_type("mode")
            
            assert data_type == fmi.FMI2_ENUMERATION
            
            opts = model.simulate_options()
            opts["result_handling"] = "memory"
            
            res = model.simulate(options=opts)
            res["mode"] #Check that the enumeration variable is in the dict, otherwise exception

class TestResultMemory:
    pass

if assimulo_installed:
    class TestResultFileBinary_Simulation:
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
        def test_only_parameters(self):
            model = Dummy_FMUModelME2([], "ParameterAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            
            opts = model.simulate_options()
            opts["result_handling"] = "custom"
            opts["result_handler"] = ResultHandlerBinaryFile(model)
            opts["filter"] = "p2"
            
            res = model.simulate(options=opts)
            
            nose.tools.assert_almost_equal(3.0, res["p2"][0])
        
        @testattr(stddist = True)
        def test_no_variables(self):
            model = Dummy_FMUModelME2([], "ParameterAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            
            opts = model.simulate_options()
            opts["result_handling"] = "custom"
            opts["result_handler"] = ResultHandlerBinaryFile(model)
            opts["filter"] = "NoMatchingVariables"
            
            res = model.simulate(options=opts)
            
            nose.tools.assert_almost_equal(1.0, res["time"][-1])
            
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
        def test_enumeration_binary(self):
            
            model = Dummy_FMUModelME2([], "Friction2.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            data_type = model.get_variable_data_type("mode")
            
            assert data_type == fmi.FMI2_ENUMERATION
            
            opts = model.simulate_options()
            opts["result_handling"] = "custom"
            opts["result_handler"] = ResultHandlerBinaryFile(model)
            
            res = model.simulate(options=opts)
            res["mode"] #Check that the enumeration variable is in the dict, otherwise exception
            
        @testattr(stddist = True)
        def test_integer_start_time(self):
            model = Dummy_FMUModelME2([], "Alias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            
            opts = model.simulate_options()
            opts["result_handling"] = "binary"

            #Assert that there is no exception when reloading the file
            res = model.simulate(start_time=0, options=opts)
            
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
        def test_binary_options_me1(self):
            simple_alias = Dummy_FMUModelME1([40], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
            _run_negated_alias(simple_alias, "binary")
        
        @testattr(stddist = True)
        def test_binary_options_me2(self):
            simple_alias = Dummy_FMUModelME2([("x", "y")], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            _run_negated_alias(simple_alias, "binary")

class TestResultFileBinary:
    @testattr(stddist = True)
    def test_get_description_unicode(self):
        model = Dummy_FMUModelME1([], "Description.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        model.initialize()
        
        result_writer = ResultHandlerBinaryFile(model)
        result_writer.set_options(model.simulate_options())
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.simulation_end()
        
        res = ResultDymolaBinary('Description_result.mat')
        
        desc = res.description[res.get_variable_index("x")]
        #This handling should in the future be nativly handled by the IO module        
        desc = desc.encode("latin_1", "replace").decode("utf-8", "replace")
        
        assert desc == u"Test symbols '' ‘’"
        
    @testattr(stddist = True)
    def test_get_description(self):
        model = Dummy_FMUModelME1([], "CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        model.initialize()
        
        result_writer = ResultHandlerBinaryFile(model)
        result_writer.set_options(model.simulate_options())
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.simulation_end()
        
        res = ResultDymolaBinary('CoupledClutches_result.mat')
        
        assert res.description[res.get_variable_index("J1.phi")] == "Absolute rotation angle of component"
    
    @testattr(stddist = True)
    def test_description_not_stored(self):
        model = Dummy_FMUModelME1([], "CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        model.initialize()
        
        opts = model.simulate_options()
        opts["result_store_variable_description"] = False
        
        result_writer = ResultHandlerBinaryFile(model)
        result_writer.set_options(opts)
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.simulation_end()
        
        res = ResultDymolaBinary('CoupledClutches_result.mat')
        
        assert res.description[res.get_variable_index("J1.phi")] == "", "Description is not empty, " + res.description[res.get_variable_index("J1.phi")]
    
    @testattr(stddist = True)
    def test_read_all_variables(self):
        res = ResultDymolaBinary(os.path.join(file_path, "files", "Results", "DoublePendulum.mat"))
        
        assert len(res.name) == 1097, "Incorrect number of variables found, should be 1097"
        
        for var in res.name:
            res.get_variable_data(var)
    
    @testattr(stddist = True)
    def test_data_matrix_delayed_loading(self):
        res = ResultDymolaBinary(os.path.join(file_path, "files", "Results", "DoublePendulum.mat"), delayed_trajectory_loading=True)
        
        data_matrix = res.get_data_matrix()
        
        [nbr_continuous_variables, nbr_points] = data_matrix.shape
        
        assert nbr_continuous_variables == 68, "Number of variables is incorrect, should be 68"
        assert nbr_points == 502, "Number of points is incorrect, should be 502"
    
    @testattr(stddist = True)
    def test_data_matrix_loading(self):
        res = ResultDymolaBinary(os.path.join(file_path, "files", "Results", "DoublePendulum.mat"), delayed_trajectory_loading=False)
        
        data_matrix = res.get_data_matrix()
        
        [nbr_continuous_variables, nbr_points] = data_matrix.shape
        
        assert nbr_continuous_variables == 68, "Number of variables is incorrect, should be 68"
        assert nbr_points == 502, "Number of points is incorrect, should be 502"
    
    @testattr(stddist = True)
    def test_read_all_variables_from_stream(self):
        
        with open(os.path.join(file_path, "files", "Results", "DoublePendulum.mat"), "rb") as f:
            res = ResultDymolaBinary(f)
            
            assert len(res.name) == 1097, "Incorrect number of variables found, should be 1097"
        
            for var in res.name:
                res.get_variable_data(var)

    @testattr(stddist = True)
    def test_compare_all_variables_from_stream(self):
        res_file = ResultDymolaBinary(os.path.join(file_path, "files", "Results", "DoublePendulum.mat"))
        
        assert len(res_file.name) == 1097, "Incorrect number of variables found, should be 1097"

        with open(os.path.join(file_path, "files", "Results", "DoublePendulum.mat"), "rb") as f:
            res_stream = ResultDymolaBinary(f)
            
            for var in res_file.name:
                x_file   = res_file.get_variable_data(var)
                x_stream = res_stream.get_variable_data(var)
                
                np.testing.assert_array_equal(x_file.x, x_stream.x, err_msg="Mismatch in array values for var=%s"%var)
                
    @testattr(stddist = True)
    def test_work_flow_me1(self):
        model = Dummy_FMUModelME1([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        model.initialize()
        
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
        model = Dummy_FMUModelME2([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()
        
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

        nose.tools.assert_almost_equal(h.x[0], 1.000000, 5)
        nose.tools.assert_almost_equal(derh.x[0], 0.000000, 5)
    
    @testattr(stddist = True)
    def test_work_flow_me2_aborted(self):
        model = Dummy_FMUModelME2([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()
        
        bouncingBall = ResultHandlerBinaryFile(model)
        
        bouncingBall.set_options(model.simulate_options())
        bouncingBall.simulation_start()
        bouncingBall.initialize_complete()
        bouncingBall.integration_point()
        bouncingBall.integration_point()
        bouncingBall.integration_point()
        #No call to simulation end to mimic an aborted simulation
        bouncingBall._file.close()
        
        res = ResultDymolaBinary('bouncingBall_result.mat')
        
        h = res.get_variable_data('h')
        derh = res.get_variable_data('der(h)')

        nose.tools.assert_almost_equal(h.x[0], 1.000000, 5, msg="Incorrect initial value for 'h', should be 1.0")
        nose.tools.assert_almost_equal(derh.x[0], 0.000000, 5, msg="Incorrect  value for 'derh', should be 0.0")
        nose.tools.assert_almost_equal(h.x[1], 1.000000, 5, msg="Incorrect value for 'h', should be 1.0")
        nose.tools.assert_almost_equal(derh.x[1], 0.000000, 5, msg="Incorrect value for 'derh', should be 0.0")
        nose.tools.assert_almost_equal(h.x[2], 1.000000, 5, msg="Incorrect value for 'h', should be 1.0")
        nose.tools.assert_almost_equal(derh.x[2], 0.000000, 5, msg="Incorrect value for 'derh', should be 0.0")
    
    @testattr(stddist = True)
    def test_filter_no_variables(self):
        model = Dummy_FMUModelME2([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()
        model.time = 1.0
        opts = model.simulate_options()
        opts["filter"] = "NoMatchingVariables"
        
        
        bouncingBall = ResultHandlerBinaryFile(model)
        
        bouncingBall.set_options(opts)
        bouncingBall.simulation_start()
        bouncingBall.initialize_complete()
        bouncingBall.integration_point()
        bouncingBall.simulation_end()
        
        res = ResultDymolaBinary('bouncingBall_result.mat')
        
        t = res.get_variable_data('time')
        nose.tools.assert_almost_equal(t.x[-1], 1.000000, 5)
    
    @testattr(stddist = True)
    def test_binary_options_cs2(self):
        simple_alias = Dummy_FMUModelCS2([("x", "y")], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS2.0"), _connect_dll=False)
        _run_negated_alias(simple_alias, "binary")

if assimulo_installed:
    class TestResultCSVTextual_Simulation:
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
        def test_no_variables(self):
            model = Dummy_FMUModelME2([], "ParameterAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            
            opts = model.simulate_options()
            opts["result_handling"] = "custom"
            opts["result_handler"] = ResultHandlerCSV(model)
            opts["filter"] = "NoMatchingVariables"
            opts["result_file_name"] = "NoMatchingTest.csv"
            
            res = model.simulate(options=opts)
            
            nose.tools.assert_almost_equal(1.0, res["time"][-1])
            
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
                
        @testattr(stddist = True)
        def test_csv_options_me1(self):
            simple_alias = Dummy_FMUModelME1([40], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
            _run_negated_alias(simple_alias, "csv")
        
        @testattr(stddist = True)
        def test_csv_options_me2(self):
            simple_alias = Dummy_FMUModelME2([("x", "y")], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            _run_negated_alias(simple_alias, "csv")
        
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

class TestResultCSVTextual:
    @testattr(stddist = True)
    def test_delimiter(self):
        
        res = ResultCSVTextual(os.path.join(file_path, 'files', 'Results', 'TestCSV.csv'), delimiter=",")
        
        x = res.get_variable_data("fd.y")
        
        assert x.x[-1] == 1
    
    @testattr(stddist = True)
    def test_work_flow_me1(self):
        model = Dummy_FMUModelME1([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        model.initialize()
        
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
        model = Dummy_FMUModelME2([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()
        
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
