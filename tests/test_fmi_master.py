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

import pytest
import os
import numpy as np
import warnings
import re
import scipy.sparse as sps
from pathlib import Path

from pyfmi import Master
from pyfmi.fmi import FMUException, FMUModelCS2, FMUModelME2
from pyfmi.test_util import Dummy_FMUModelCS2
from pyfmi.common.io import ResultHandler, ResultSizeError
from pyfmi.common.algorithm_drivers import UnrecognizedOptionError

file_path = os.path.dirname(os.path.abspath(__file__))
cs2_xml_path = os.path.join(file_path, "files", "FMUs", "XML", "CS2.0")
me2_xml_path = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0")

this_dir = Path(__file__).parent
FMI2_REF_FMU_PATH = Path(this_dir) / 'files' / 'reference_fmus' / '2.0'

import warnings
warnings.filterwarnings("ignore")

class FMUModelCS2CapabilityOverwrite(FMUModelCS2):
    """Dummy testing class with capability flags overwritten, for testing 
    (in)valid option combinations that require certain capabilities."""
    def get_capability_flags(self):
        res = super().get_capability_flags()
        res["canInterpolateInputs"] = True
        res["providesDirectionalDerivatives"] = True
        return res
    
    def _provides_directional_derivatives(self) -> bool:
        return True
    
class Test_Master:
    def test_loading_models(self):
        model_sub1 = FMUModelCS2(os.path.join(cs2_xml_path, "LinearStability.SubSystem1.fmu"), _connect_dll=False)
        model_sub2 = FMUModelCS2(os.path.join(cs2_xml_path, "LinearStability.SubSystem2.fmu"), _connect_dll=False)
       
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
        
        #Assert that loading is successful
        sim = Master(models, connections)
    
    def test_loading_wrong_model(self):
        model_sub1 = FMUModelCS2(os.path.join(cs2_xml_path, "LinearStability.SubSystem1.fmu"), _connect_dll=False)
        model_sub2 = FMUModelME2(os.path.join(me2_xml_path, "LinearStability.SubSystem2.fmu"), _connect_dll=False)
       
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
        
        with pytest.raises(FMUException):
            Master(models, connections)
    
    def test_connection_variables(self):
        model_sub1 = FMUModelCS2(os.path.join(cs2_xml_path, "LinearStability.SubSystem1.fmu"), _connect_dll=False)
        model_sub2 = FMUModelCS2(os.path.join(cs2_xml_path, "LinearStability.SubSystem2.fmu"), _connect_dll=False)
       
        models = [model_sub1, model_sub2]
       
        #Test all connections are inputs or outputs
        connections = [(model_sub1,"y1",model_sub2,"x2"),
                   (model_sub2,"y2",model_sub1,"u1")]
                   
        with pytest.raises(FMUException):
            Master(models, connections)
       
        #Test wrong input / output order
        connections = [(model_sub2,"u2", model_sub1,"y1"),
                   (model_sub2,"y2",model_sub1,"u1")]
                   
        with pytest.raises(FMUException):
            Master(models, connections)
    
    def test_basic_algebraic_loop(self):
        model_sub1 = FMUModelCS2(os.path.join(cs2_xml_path, "LinearStability.SubSystem1.fmu"), _connect_dll=False)
        model_sub2 = FMUModelCS2(os.path.join(cs2_xml_path, "LinearStability.SubSystem2.fmu"), _connect_dll=False)
        
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
                   
        sim = Master(models, connections)
        assert sim.algebraic_loops
       
        model_sub1 = FMUModelCS2(os.path.join(cs2_xml_path, "LinearStability_LinearSubSystemNoFeed1.fmu"), _connect_dll=False)
        model_sub2 = FMUModelCS2(os.path.join(cs2_xml_path, "LinearStability_LinearSubSystemNoFeed2.fmu"), _connect_dll=False)
       
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
                   
        sim = Master(models, connections)
        assert not sim.algebraic_loops
    
    def _load_basic_simulation(self):
        model_sub1 = Dummy_FMUModelCS2([], os.path.join(cs2_xml_path, "LinearCoSimulation_LinearSubSystem1.fmu"), _connect_dll=False)
        model_sub2 = Dummy_FMUModelCS2([], os.path.join(cs2_xml_path, "LinearCoSimulation_LinearSubSystem2.fmu"), _connect_dll=False)
        
        a1 = model_sub1.values[model_sub1.get_variable_valueref("a1")]
        b1 = model_sub1.values[model_sub1.get_variable_valueref("b1")]
        c1 = model_sub1.values[model_sub1.get_variable_valueref("c1")]
        d1 = model_sub1.values[model_sub1.get_variable_valueref("d1")]
        
        a2 = model_sub2.values[model_sub2.get_variable_valueref("a2")]
        b2 = model_sub2.values[model_sub2.get_variable_valueref("b2")]
        c2 = model_sub2.values[model_sub2.get_variable_valueref("c2")]
        d2 = model_sub2.values[model_sub2.get_variable_valueref("d2")]

        def do_step1(current_t, step_size, new_step=True):
            u1 = model_sub1.values[model_sub1.get_variable_valueref("u1")]
            
            model_sub1.continuous_states = 1.0/a1*(np.exp(a1*step_size)-1.0)*b1*u1+np.exp(a1*step_size)*model_sub1.continuous_states
            model_sub1.set_real([model_sub1.get_variable_valueref("y1")], c1*model_sub1.continuous_states+d1*u1)
            model_sub1.completed_integrator_step()
            return 0
        
        def do_step2(current_t, step_size, new_step=True):
            u2 = model_sub2.values[model_sub2.get_variable_valueref("u2")]
            
            model_sub2.continuous_states = 1.0/a2*(np.exp(a2*step_size)-1.0)*b2*u2+np.exp(a2*step_size)*model_sub2.continuous_states
            model_sub2.set_real([model_sub2.get_variable_valueref("y2")], c2*model_sub2.continuous_states+d2*u2)
            model_sub2.completed_integrator_step()
            return 0
            
        model_sub1.do_step = do_step1
        model_sub2.do_step = do_step2
        
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
                
        return models, connections
    
    def _sim_basic_simulation(self, models, connections, opts_update):
        master = Master(models, connections)
       
        opts = master.simulate_options()
        opts["step_size"] = 0.0005
        opts.update(opts_update)
        res = master.simulate(options=opts)
        
        assert res[models[0]].final("x1") == pytest.approx(0.0859764038708439, abs = 1e-3)
        assert res[models[1]].final("x2") == pytest.approx(0.008392664839635064, abs = 1e-4)
        
        return res
    
    def _basic_simulation(self, opts_update):
        models, connections = self._load_basic_simulation()
        self._sim_basic_simulation(models, connections, opts_update)
    
    def test_basic_simulation_txt_file(self):
        opts = {"result_handling":"file"}
        self._basic_simulation(opts)
    
    def test_basic_simulation_mat_file(self):
        opts = {"result_handling":"binary"}
        self._basic_simulation(opts)
    
    def test_basic_simulation_memory(self):
        opts = {"result_handling":"memory"}
        self._basic_simulation(opts)

    def test_basic_simulation_max_result_size(self):
        opts = {"result_max_size":10000}

        with pytest.raises(ResultSizeError):
            self._basic_simulation(opts)
    
    def test_basic_simulation_mat_file_naming(self):
        opts = {"result_handling":"binary", "result_file_name": "Should fail..."}
        
        with pytest.raises(UnrecognizedOptionError):
            self._basic_simulation(opts)
        
        opts = {"result_handling":"binary", "result_file_name": {"Should fail..."}}
        
        with pytest.raises(UnrecognizedOptionError):
            self._basic_simulation(opts)
    
    def test_basic_simulation_mat_file_naming_exists(self):
        models, connections = self._load_basic_simulation()
        
        opts = {"result_handling":"binary", "result_file_name": {models[0]: "Test1.mat", models[1]: "Test2.mat"}}
        
        res = self._sim_basic_simulation(models, connections, opts)
        
        assert os.path.isfile("Test1.mat"), "Test1.mat does not exists"
        assert os.path.isfile("Test2.mat"), "Test2.mat does not exists"
    
    def test_basic_simulation_txt_file_naming_exists(self):
        models, connections = self._load_basic_simulation()
        
        opts = {"result_handling":"file", "result_file_name": {models[0]: "Test1.txt", models[1]: "Test2.txt"}}
        
        res = self._sim_basic_simulation(models, connections, opts)
        
        assert os.path.isfile("Test1.txt"), "Test1.txt does not exists"
        assert os.path.isfile("Test2.txt"), "Test2.txt does not exists"
    
    def test_basic_simulation_csv_file_naming_exists(self):
        models, connections = self._load_basic_simulation()
        
        opts = {"result_handling":"csv", "result_file_name": {models[0]: "Test1.csv", models[1]: "Test2.csv"}}
        
        res = self._sim_basic_simulation(models, connections, opts)
        
        assert os.path.isfile("Test1.csv"), "Test1.csv does not exists"
        assert os.path.isfile("Test2.csv"), "Test2.csv does not exists"
    
    def test_basic_simulation_none_result(self):
        models, connections = self._load_basic_simulation()
        
        opts = {"result_handling": None, "step_size": 0.0005}
        
        master = Master(models, connections)
        res = master.simulate(options=opts)
        
        assert res[models[0]]._result_data is None
        assert res[models[1]]._result_data is None
    
    def test_custom_result_handler_invalid(self):
        models, connections = self._load_basic_simulation()
        
        class A:
            pass
                
        opts = {}
        opts["result_handling"] = "hejhej"
        with pytest.raises(Exception):
            self._sim_basic_simulation(models, connections, opts)

        opts["result_handling"] = "custom"
        opts["result_handler"] = A()
        err = "'result_handler' option must be a dictionary for 'result_handling' = 'custom'."
        with pytest.raises(Exception, match = re.escape(err)):
            self._sim_basic_simulation(models, connections, opts)

        opts["result_handling"] = "custom"
        opts["result_handler"] = {m: A() for m in models[1:]}
        err = "'result_handler' option does not contain result handler for model '{}'".format(models[0].get_identifier())
        with pytest.raises(Exception, match = re.escape(err)):
            self._sim_basic_simulation(models, connections, opts)

        opts["result_handling"] = "custom"
        opts["result_handler"] = {m: A() for m in models}
        err = "The result handler needs to be an instance of ResultHandler."
        with pytest.raises(Exception, match = re.escape(err)):
            self._sim_basic_simulation(models, connections, opts) 
        
    def test_custom_result_handler_valid(self):
        models, connections = self._load_basic_simulation()
        
        class B(ResultHandler):
            def __init__(self, i, *args, **kwargs):
                super().__init__(self, *args, **kwargs)
                self._i = i
            def get_result(self):
                return self._i
                
        opts = {}
        opts["result_handling"] = "custom"
        opts["result_handler"] = {m: B(i) for i, m in enumerate(models)}
        opts["step_size"] = 0.0005
        
        master = Master(models, connections)
       
        res = master.simulate(options=opts)

        assert res[models[0]]._result_data == 0, "Result is not 0"
        assert res[models[1]]._result_data == 1, "Result is not 1"
        
    def test_basic_simulation_with_block_initialization(self):
        opts = {"block_initialization": True}
        self._basic_simulation(opts)
    
    def test_integer_connections(self):
        model_sub1 = Dummy_FMUModelCS2([], os.path.join(cs2_xml_path, "IntegerStep.fmu"), _connect_dll=False)
        model_sub2 = Dummy_FMUModelCS2([], os.path.join(cs2_xml_path, "GainTestInteger.fmu"), _connect_dll=False)
        
        model_sub1.set("y", 1)
        def do_step1(current_t, step_size, new_step=True):
            model_sub1.set_integer([model_sub1.get_variable_valueref("y")], [1] if current_t+step_size < 0.5 else [3])
            model_sub1.completed_integrator_step()
            return 0
        
        def do_step2(current_t, step_size, new_step=True):
            u = model_sub2.get_integer([model_sub2.get_variable_valueref("u")])
            model_sub2.set_integer([model_sub2.get_variable_valueref("y")], 10*u)
            model_sub2.completed_integrator_step()
            return 0
            
        model_sub1.do_step = do_step1
        model_sub2.do_step = do_step2
        
        models = [model_sub1, model_sub2]
        connections = [(model_sub1, 'y', model_sub2, 'u')]

        master = Master(models, connections)

        opts = master.simulate_options()
        opts["block_initialization"] = True

        res = master.simulate(start_time=0.0, final_time=2.0, options=opts)
        
        assert res[model_sub2]["u"][0] == 1
        assert res[model_sub2]["u"][-1] == 3
    
    def test_integer_to_real_connections(self):
        model_sub1 = Dummy_FMUModelCS2([], os.path.join(cs2_xml_path, "IntegerStep.fmu"), _connect_dll=False)
        model_sub2 = Dummy_FMUModelCS2([], os.path.join(cs2_xml_path, "GainTestReal.fmu"), _connect_dll=False)
        
        model_sub1.set("y", 1)
        def do_step1(current_t, step_size, new_step=True):
            model_sub1.set_integer([model_sub1.get_variable_valueref("y")], [1] if current_t+step_size < 0.5 else [3])
            model_sub1.completed_integrator_step()
            return 0
        
        def do_step2(current_t, step_size, new_step=True):
            u = model_sub2.get_real([model_sub2.get_variable_valueref("u")])
            model_sub2.set_real([model_sub2.get_variable_valueref("y")], 10*u)
            model_sub2.completed_integrator_step()
            return 0
            
        model_sub1.do_step = do_step1
        model_sub2.do_step = do_step2
        
        models = [model_sub1, model_sub2]
        connections= [(model_sub1, 'y', model_sub2, 'u')]

        master = Master(models, connections)

        opts = master.simulate_options()
        opts["block_initialization"] = True

        res = master.simulate(start_time=0.0,final_time=2.0, options=opts)
        
        assert res[model_sub2]["u"][0] == 1.0
        assert res[model_sub2]["u"][-1] == 3.0
    
    def test_unstable_simulation(self):
        model_sub1 = Dummy_FMUModelCS2([], os.path.join(cs2_xml_path, "LinearCoSimulation_LinearSubSystem1.fmu"), _connect_dll=False)
        model_sub2 = Dummy_FMUModelCS2([], os.path.join(cs2_xml_path, "LinearCoSimulation_LinearSubSystem2.fmu"), _connect_dll=False)
        
        model_sub2.set("d2", 1.1) #Coupled system becomes unstable
        
        a1 = model_sub1.values[model_sub1.get_variable_valueref("a1")]
        b1 = model_sub1.values[model_sub1.get_variable_valueref("b1")]
        c1 = model_sub1.values[model_sub1.get_variable_valueref("c1")]
        d1 = model_sub1.values[model_sub1.get_variable_valueref("d1")]
        
        a2 = model_sub2.values[model_sub2.get_variable_valueref("a2")]
        b2 = model_sub2.values[model_sub2.get_variable_valueref("b2")]
        c2 = model_sub2.values[model_sub2.get_variable_valueref("c2")]
        d2 = model_sub2.values[model_sub2.get_variable_valueref("d2")]
        
        def do_step1(current_t, step_size, new_step=True):
            u1 = model_sub1.values[model_sub1.get_variable_valueref("u1")]
            
            model_sub1.continuous_states = 1.0/a1*(np.exp(a1*step_size)-1.0)*b1*u1+np.exp(a1*step_size)*model_sub1.continuous_states
            model_sub1.set_real([model_sub1.get_variable_valueref("y1")], c1*model_sub1.continuous_states+d1*u1)
            model_sub1.completed_integrator_step()
            return 0
        
        def do_step2(current_t, step_size, new_step=True):
            u2 = model_sub2.values[model_sub2.get_variable_valueref("u2")]
            
            model_sub2.continuous_states = 1.0/a2*(np.exp(a2*step_size)-1.0)*b2*u2+np.exp(a2*step_size)*model_sub2.continuous_states
            model_sub2.set_real([model_sub2.get_variable_valueref("y2")], c2*model_sub2.continuous_states+d2*u2)
            model_sub2.completed_integrator_step()
            return 0
            
        model_sub1.do_step = do_step1
        model_sub2.do_step = do_step2
        
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
                   
        master = Master(models, connections)
       
        opts = master.simulate_options()
        opts["step_size"] = 0.0005
       
        res = master.simulate(final_time=0.3, options=opts)
       
        assert abs(res[model_sub1].final("x1")) > 100
        assert abs(res[model_sub2].final("x2")) > 100

    @pytest.mark.parametrize("input_traj",
        [
            np.transpose( # array variant
                np.vstack((
                    np.linspace(0, 5, 10), # time
                    np.linspace(0, 5, 10), # input 1
                    np.linspace(0, 5, 10)*2, # input 2
                ))
            ),
            lambda t: [t, 2*t] # function variant
        ]
    )
    def test_external_inputs(self, input_traj):
        """Test of using external inputs for the Master algorithm."""
        fmu1 = FMUModelCS2(os.path.join(FMI2_REF_FMU_PATH, "Feedthrough.fmu"))
        fmu2 = FMUModelCS2(os.path.join(FMI2_REF_FMU_PATH, "Feedthrough.fmu"))

        models = [fmu1, fmu2]
        connections = [(fmu1, "Float64_continuous_output", fmu2, "Float64_continuous_input")]
        master = Master(models, connections)

        t_start, t_final = 0, 5
        opts = master.simulate_options()
        opts["step_size"] = 1

        # Generate input
        input_vars = [
            (fmu1, "Float64_continuous_input"),
            (fmu2, "Float64_discrete_input")
        ]
        input_object = [input_vars, input_traj]

        res = master.simulate(t_start, t_final, options = opts, input = input_object)

        np.testing.assert_array_equal(res[0]["Float64_continuous_output"], [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(res[1]["Float64_continuous_output"], [0, 1, 2, 3, 4, 5])

        np.testing.assert_array_equal(res[0]["Float64_discrete_output"], [0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(res[1]["Float64_discrete_output"], [0, 2, 4, 6, 8, 10])

class Test_Master_Result_Downsampling:
    """Tests related to the 'result_downsampling' option of the Master algorithm."""
    @classmethod
    def setup_class(cls):
        """Called with setup of class, once."""
        cls.fmu1 = FMUModelCS2(os.path.join(FMI2_REF_FMU_PATH, "Feedthrough.fmu"))
        cls.fmu2 = FMUModelCS2(os.path.join(FMI2_REF_FMU_PATH, "Feedthrough.fmu"))

        cls.models = [cls.fmu1, cls.fmu2]
        connections = [
            (cls.fmu1, "Float64_continuous_output", cls.fmu2, "Float64_continuous_input"),
            (cls.fmu1, "Float64_discrete_output", cls.fmu2, "Float64_discrete_input"),
        ]
        cls.master = Master(cls.models, connections)

    def _sim_basic_simulation(self, opts_update, final_time = 10) -> dict:
        opts = self.master.simulate_options().copy()
        opts["step_size"] = 1
        opts.update(opts_update)
        self.master.reset()
        # use an input to ensure non-trivial result
        input_vars = [(self.fmu1, "Float64_continuous_input")]
        input_object = [input_vars, lambda t: [t]]
        return self.master.simulate(0, final_time, input = input_object, options=opts)

    @pytest.mark.parametrize("factor, expected_res",
        [
            (1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), # sanity check
            (2, [0, 2, 4, 6, 8, 10]), # aligned with steps
            (3, [0, 3, 6, 9, 10]), # not aligned
            (7, [0, 7, 10]),
            (10, [0, 10]), # equal to number of steps
            (20, [0, 10]), # larger
        ]
    )
    def test_downsample_result(self, factor, expected_res):
        """ Test multiple result_downsampling_factor value and verify the result. """
        factors = {self.fmu1: factor, self.fmu2: factor}
        res = self._sim_basic_simulation({'result_downsampling_factor': factors}, final_time = 10)
        np.testing.assert_array_equal(res[self.fmu1]["Float64_continuous_output"], expected_res)
        np.testing.assert_array_equal(res[self.fmu2]["Float64_continuous_output"], expected_res)

    def test_master_alg_downsample_result_simple_value(self):
        """Test setting a simple value to the 'result_downsampling_factor' option for the Master algorithm."""
        res = self._sim_basic_simulation({'result_downsampling_factor': 2}, final_time = 10)
        expected_res = [0, 2, 4, 6, 8, 10]
        np.testing.assert_array_equal(res[self.fmu1]["Float64_continuous_output"], expected_res)
        np.testing.assert_array_equal(res[self.fmu2]["Float64_continuous_output"], expected_res)

    @pytest.mark.parametrize("factor, expected_res",
        [
            (1, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10]), # sanity
            (2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), # aligned with steps
            (3, [0, 2, 3, 5, 6, 8, 9, 10]), # not aligned
            (7, [0, 6, 7, 10]),
            (10, [0, 9, 10]), # equal to number of steps
            (20, [0, 10]), # larger
        ]
    )
    def test_with_store_step_before_update(self, factor, expected_res):
        """ Test result_downsampling_factor with store_step_before_update. """
        opts_update = {'result_downsampling_factor': {self.fmu1: factor, self.fmu2: factor},
                       'store_step_before_update': True}
        res = self._sim_basic_simulation(opts_update, final_time = 10)
        np.testing.assert_array_equal(res[self.fmu1]["Float64_continuous_output"], expected_res)
        np.testing.assert_array_equal(res[self.fmu2]["Float64_continuous_output"], expected_res)

    @pytest.mark.parametrize("value", [-10, 0])
    def test_invalid_value_non_positive_int(self, value):
        """ Verify we get an exception if the option is set to anything less than 1. """
        expected_substr = f"Valid values in option 'result_downsampling_factor' are only positive integers, got: '{value}'."
        with pytest.raises(FMUException, match = re.escape(expected_substr)):
            self._sim_basic_simulation({'result_downsampling_factor': {self.fmu1: value}})

    @pytest.mark.parametrize("value", [1/2, "0.5", False])
    def test_invalid_type(self, value):
        """ Verify we get an exception if the option is set to anything that is not an integer. """
        expected_substr = f"Values to 'result_downsampling_factor' must be an integer or dictionary with integer values, got: '{type(value)}'."
        with pytest.raises(FMUException, match = re.escape(expected_substr)):
            self._sim_basic_simulation({'result_downsampling_factor': {self.fmu1: value}})
    
    def test_error_controlled(self):
        uptate_options = {'result_downsampling_factor': {self.fmu1: 2, self.fmu2: 2},
                          'error_controlled': True,
                          'rtol': 10}
        msg = "Result downsampling not supported for error controlled simulation, no downsampling will be performed."
        with pytest.warns(UserWarning, match = msg):
            self._sim_basic_simulation(uptate_options, final_time = 1)

    def test_different_value_per_model(self):
        """Test setting different values per model."""
        factors = {self.fmu1: 2, self.fmu2: 3}
        res = self._sim_basic_simulation({'result_downsampling_factor': factors})
        np.testing.assert_array_equal(
            res[self.fmu1]["Float64_continuous_output"], 
            [0, 2, 4, 6, 8, 10]
        )
        np.testing.assert_array_equal(
            res[self.fmu2]["Float64_continuous_output"], 
            [0, 3, 6, 9, 10]
        )

    def test_partial(self):
        """Test setting values only for some models."""
        factors = {self.fmu2: 3}
        res = self._sim_basic_simulation({'result_downsampling_factor': factors})
        np.testing.assert_array_equal(
            res[self.fmu1]["Float64_continuous_output"], 
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        np.testing.assert_array_equal(
            res[self.fmu2]["Float64_continuous_output"], 
            [0, 3, 6, 9, 10]
        )

    def test_dict_invalid_key(self):
        """Test using a dictionary as input with invalid key."""
        factors = {"a": 2}
        expected_substr = "Invalid key 'a' in 'result_downsampling_factor' option dictionary, not a model."
        with pytest.raises(FMUException, match = re.escape(expected_substr)):
            self._sim_basic_simulation({'result_downsampling_factor': factors})

class Test_Master_Step_Size_Downsampling:
    """Tests on the 'step_size_downsampling_factor' option of the Master algorithm."""
    @classmethod
    def setup_class(cls):
        """Called with setup of class, once."""
        cls.fmu1 = FMUModelCS2(os.path.join(FMI2_REF_FMU_PATH, "Feedthrough.fmu"))
        cls.fmu2 = FMUModelCS2(os.path.join(FMI2_REF_FMU_PATH, "Feedthrough.fmu"))

        models = [cls.fmu1, cls.fmu2]
        connections = [
            (cls.fmu1, "Float64_continuous_output", cls.fmu2, "Float64_continuous_input"),
            (cls.fmu1, "Float64_discrete_output", cls.fmu2, "Float64_discrete_input"),
        ]
        cls.master = Master(models, connections)

    def setup_method(self, method):
        """Called with every test method."""
        self.master.reset()

    def test_with_error_control(self):
        """Test 'step_size_downsampling_factor' + 'error_controlled'."""
        opts = self.master.simulate_options()
        opts["error_controlled"] = True
        opts["step_size_downsampling_factor"] = {self.fmu1: 3, self.fmu2: 2}
        opts["step_size"] = 0.5

        msg = "Step-size downsampling not supported for error controlled simulation, no downsampling will be performed."
        with pytest.warns(UserWarning, match = re.escape(msg)):
            self.master.simulate(options = opts)

    def test_extrapolation_order(self):
        """Test combination with the 'extrapolation_order' option."""
        fmu1 = FMUModelCS2CapabilityOverwrite(os.path.join(FMI2_REF_FMU_PATH, "Feedthrough.fmu"))
        fmu2 = FMUModelCS2CapabilityOverwrite(os.path.join(FMI2_REF_FMU_PATH, "Feedthrough.fmu"))

        models = [fmu1, fmu2]
        connections = [(fmu1, "Float64_continuous_output", fmu2, "Float64_continuous_input")]
        master = Master(models, connections)
        opts = master.simulate_options()
        opts["step_size_downsampling_factor"] = {fmu1: 3, fmu2: 2}
        opts["extrapolation_order"] = 1
        opts["step_size"] = 0.5

        err_msg = "Use of 'step_size_downsampling_factor' with 'extrapolation_order' > 0 not supported."
        with pytest.raises(FMUException, match = re.escape(err_msg)):
            master.simulate(options = opts)

    def test_linear_correction(self):
        """Test combination with the 'linear_correction' option."""
        fmu1 = FMUModelCS2CapabilityOverwrite(os.path.join(FMI2_REF_FMU_PATH, "Feedthrough.fmu"))
        fmu2 = FMUModelCS2CapabilityOverwrite(os.path.join(FMI2_REF_FMU_PATH, "Feedthrough.fmu"))

        models = [fmu1, fmu2]
        connections = [(fmu1, "Float64_continuous_output", fmu2, "Float64_continuous_input")]
        master = Master(models, connections)
        opts = master.simulate_options()
        opts["step_size_downsampling_factor"] = {fmu1: 3, fmu2: 2}
        opts["linear_correction"] = True
        opts["step_size"] = 0.5

        err_msg = "Use of 'step_size_downsampling_factor' with 'linear_correction' not supported."
        with pytest.raises(FMUException, match = re.escape(err_msg)):
            master.simulate(options = opts)

    def test_check_invalid_option_input_non_dict_via_simulate_options(self):
        """Test invalid inputs to the option, not a dictionary."""
        opts = self.master.simulate_options()
        err_msg = "The options in 'step_size_downsampling_factor' needs to be provided as a dictionary."
        with pytest.raises(UnrecognizedOptionError, match = re.escape(err_msg)):
            opts["step_size_downsampling_factor"] = 2

    def test_check_invalid_option_input_non_dict_via_dictionary(self):
        """Test invalid inputs to the option, not a dictionary."""
        err_msg = "The options in 'step_size_downsampling_factor' needs to be provided as a dictionary."
        with pytest.raises(UnrecognizedOptionError, match = re.escape(err_msg)):
            self.master.simulate(options = {"step_size_downsampling_factor": 1})

    @pytest.mark.parametrize("values, err_msg",
        [
            ([0.5, 1], "Values in 'step_size_downsampling_factor' option must be of type integer, got: '<class 'float'>'."),
            (["aaa", 1], "Values in 'step_size_downsampling_factor' option must be of type integer, got: '<class 'str'>'."),
            ([-1, 1], "Valid values for option 'step_size_downsampling_factor' are positive integers, got: '-1'."),
        ]
    )
    def test_check_invalid_option_input_dict_values(self, values, err_msg):
        """Test invalid inputs to the option, not a dictionary."""
        opts = self.master.simulate_options()
        opts["step_size_downsampling_factor"] = {self.fmu1: values[0], self.fmu2: values[1]}
        with pytest.raises(FMUException, match = re.escape(err_msg)):
            self.master.simulate(options = opts)

    def test_check_invalid_option_invalid_key(self):
        """Test invalid inputs to the option, not a dictionary."""
        opts = self.master.simulate_options()
        opts["step_size_downsampling_factor"] = {'a': 1, 'b': 1}
        err_msg = "Invalid key 'a' in 'step_size_downsampling_factor' option dictionary, not a model."
        with pytest.raises(FMUException, match = re.escape(err_msg)):
            self.master.simulate(options = opts)

    def test_partial_input(self):
        """Test that partial input, i.e., not providing a value for every models is supported."""
        opts = self.master.simulate_options()
        opts["step_size_downsampling_factor"] = {self.fmu1: 2}
        opts["step_size"] = 0.25
        self.master.simulate(options = opts)

    @pytest.mark.parametrize("input_var_name, output_var_name", 
        [
            ("Float64_continuous_input", "Float64_continuous_output"),
            ("Float64_discrete_input", "Float64_discrete_output"),
        ]
    )
    @pytest.mark.parametrize("rate1, rate2, expected_res1, expected_res2",
        [
            # Sanity check
            (1, 1,
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            ),
            # external input only sampled every other step
            (2, 1,
             [1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11],
             [1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11]
            ),
            # connection only updated every other step
            (1, 2,
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
             [1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11]
            ),
            # non-aligned test 1
            (2, 3,
             [1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11],
             [1, 1, 1, 3, 3, 3, 7, 7, 7, 9, 9]
            ),
            # non-aligned test 2
            (3, 2,
             [1, 1, 1, 4, 4, 4, 7, 7, 7, 10, 10],
             [1, 1, 1, 1, 4, 4, 7, 7, 7, 7, 10]
            ),
        ]
    )
    def test_two_feedthrough_system(self, input_var_name, output_var_name, 
                                    rate1, rate2, expected_res1, expected_res2):
        """Test 'step_size_downsampling_factor' for 2 Feedthrough models + 1 external input."""
        t_start, t_final = 0, 10
        opts = self.master.simulate_options()
        opts["step_size"] = 1
        opts["step_size_downsampling_factor"] = {self.fmu1: rate1, self.fmu2: rate2}

        # Generate input
        input_object = [
            [(self.fmu1, input_var_name)],
            lambda t: [t + 1] # not starting at zero to test correct values taken with initialization
        ]

        res = self.master.simulate(t_start, t_final, options = opts, input = input_object)
        np.testing.assert_array_equal(res[0][output_var_name], expected_res1)
        np.testing.assert_array_equal(res[1][output_var_name], expected_res2)

    @pytest.mark.parametrize("input_var_name, output_var_name", 
        [
            ("Float64_continuous_input", "Float64_continuous_output"),
            ("Float64_discrete_input", "Float64_discrete_output"),
        ]
    )
    def test_three_feedthrough_system(self, input_var_name, output_var_name):
        """Test 'step_size_downsampling_factor' for 3 Feedthrough models + 1 external input,
        where 2 models connect to the same output."""
        fmu1 = FMUModelCS2(os.path.join(FMI2_REF_FMU_PATH, "Feedthrough.fmu"))
        fmu2 = FMUModelCS2(os.path.join(FMI2_REF_FMU_PATH, "Feedthrough.fmu"))
        fmu3 = FMUModelCS2(os.path.join(FMI2_REF_FMU_PATH, "Feedthrough.fmu"))

        models = [fmu1, fmu2, fmu3]
        connections = [
            (fmu1, output_var_name, fmu2, input_var_name),
            (fmu1, output_var_name, fmu3, input_var_name)]
        master = Master(models, connections)

        t_start, t_final = 0, 10
        opts = master.simulate_options()
        opts["step_size"] = 1
        opts["step_size_downsampling_factor"] = {fmu1: 2, fmu2: 3, fmu3: 4}

        # Generate input
        input_object = [
            [(fmu1, input_var_name)],
            lambda t: [t + 1] # not starting at zero to test correct values taken with initialization
        ]

        res = master.simulate(t_start, t_final, options = opts, input = input_object)
        expected_res1 = np.array([1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11])
        expected_res2 = np.array([1, 1, 1, 3, 3, 3, 7, 7, 7, 9, 9])
        expected_res3 = np.array([1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9])
        np.testing.assert_array_equal(res[0][output_var_name], expected_res1)
        np.testing.assert_array_equal(res[1][output_var_name], expected_res2)
        np.testing.assert_array_equal(res[2][output_var_name], expected_res3)

    def test_with_store_step_before_update(self):
        """Test interaction with 'store_step_before_update' option."""
        t_start, t_final = 0, 10
        opts = self.master.simulate_options()
        opts["step_size"] = 1
        opts["store_step_before_update"] = True
        opts["step_size_downsampling_factor"] = {self.fmu1: 3, self.fmu2: 2}

        # Generate input
        input_object = [
            [(self.fmu1, "Float64_continuous_input")],
            lambda t: [t + 1] # not starting at zero to test correct values taken with initialization
        ]

        res = self.master.simulate(t_start, t_final, options = opts, input = input_object)
        np.testing.assert_array_equal(
            res[0]["Float64_continuous_output"], 
        #   [1, 1, 1, X, 4, 4, 4, X, 7, 7, 7, X, 10, last = 10], # X = before update value
            [1, 1, 1, 1, 4, 4, 4, 4, 7, 7, 7, 7, 10, 10]
        )
        np.testing.assert_array_equal(
            res[1]["Float64_continuous_output"], 
        #   [1, 1, X, 1, 1, X, 4, 4, X, 7, 7, X, 7, 7, X, last = 10], # X = before update value
            [1, 1, 1, 1, 1, 1, 4, 4, 4, 7, 7, 7, 7, 7, 7, 10]
        )
    
    def test_with_logging(self):
        """Test the warning one gets when using 'logging' + 'step_size_downsampling_factor'."""
        fmu1 = FMUModelCS2CapabilityOverwrite(os.path.join(FMI2_REF_FMU_PATH, "Feedthrough.fmu"))
        fmu2 = FMUModelCS2CapabilityOverwrite(os.path.join(FMI2_REF_FMU_PATH, "Feedthrough.fmu"))

        class MasterX(Master):
            """Dummy Master class for testing that bypasses FMU calls for evaluating
            global A, B, C, D matrices."""
            def compute_global_A(self):
                return sps.csr_matrix(np.array([1.]))
            def compute_global_B(self):
                return sps.csr_matrix(np.array([2.]))
            def compute_global_C(self):
                return sps.csr_matrix(np.array([3.]))
            def compute_global_D(self):
                return sps.csr_matrix(np.array([4.]))

        models = [fmu1, fmu2]
        connections = [(fmu1, "Float64_continuous_output", fmu2, "Float64_continuous_input")]
        master = MasterX(models, connections)
        opts = master.simulate_options()
        t_start, t_final = 0, 1
        opts["step_size"] = 0.5
        opts["logging"] = True
        opts["step_size_downsampling_factor"] = {fmu1: 2}

        msg = "Both 'step_size_downsampling_factor' and 'logging' are used. " \
              "Logging of A, B, C, and D matrices will be done on the global step-size." \
              "Actual values may no longer be sensible."
        with pytest.warns(UserWarning, match = re.escape(msg)):
            master.simulate(t_start, t_final, options = opts)
