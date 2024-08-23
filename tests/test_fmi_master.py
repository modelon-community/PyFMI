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
import warnings

from pyfmi import testattr, Master
from pyfmi.fmi import FMUException, FMUModelCS2, FMUModelME2
from pyfmi.tests.test_util import Dummy_FMUModelCS2
from pyfmi.common.io import ResultHandler
from pyfmi.common.algorithm_drivers import UnrecognizedOptionError

file_path = os.path.dirname(os.path.abspath(__file__))

cs2_xml_path = os.path.join(file_path, "files", "FMUs", "XML", "CS2.0")
me2_xml_path = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0")

import warnings
warnings.filterwarnings("ignore")

class Test_Master:
    
    @testattr(stddist = True)
    def test_loading_models(self):
        model_sub1 = FMUModelCS2(os.path.join(cs2_xml_path, "LinearStability.SubSystem1.fmu"), _connect_dll=False)
        model_sub2 = FMUModelCS2(os.path.join(cs2_xml_path, "LinearStability.SubSystem2.fmu"), _connect_dll=False)
       
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
        
        #Assert that loading is successful
        sim = Master(models, connections)
    
    @testattr(stddist = True)
    def test_loading_wrong_model(self):
        model_sub1 = FMUModelCS2(os.path.join(cs2_xml_path, "LinearStability.SubSystem1.fmu"), _connect_dll=False)
        model_sub2 = FMUModelME2(os.path.join(me2_xml_path, "LinearStability.SubSystem2.fmu"), _connect_dll=False)
       
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
        
        nose.tools.assert_raises(FMUException, Master, models, connections)
    
    @testattr(stddist = True)
    def test_connection_variables(self):
        model_sub1 = FMUModelCS2(os.path.join(cs2_xml_path, "LinearStability.SubSystem1.fmu"), _connect_dll=False)
        model_sub2 = FMUModelCS2(os.path.join(cs2_xml_path, "LinearStability.SubSystem2.fmu"), _connect_dll=False)
       
        models = [model_sub1, model_sub2]
       
        #Test all connections are inputs or outputs
        connections = [(model_sub1,"y1",model_sub2,"x2"),
                   (model_sub2,"y2",model_sub1,"u1")]
                   
        nose.tools.assert_raises(FMUException, Master, models, connections)
       
        #Test wrong input / output order
        connections = [(model_sub2,"u2", model_sub1,"y1"),
                   (model_sub2,"y2",model_sub1,"u1")]
                   
        nose.tools.assert_raises(FMUException, Master, models, connections)
    
    @testattr(stddist = True)
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
        
        nose.tools.assert_almost_equal(res[models[0]].final("x1"), 0.0859764038708439, 3)
        nose.tools.assert_almost_equal(res[models[1]].final("x2"), 0.008392664839635064, 4)
        
        return res
    
    def _basic_simulation(self, opts_update):
        models, connections = self._load_basic_simulation()
        self._sim_basic_simulation(models, connections, opts_update)
        
    
    @testattr(stddist = True)
    def test_basic_simulation_txt_file(self):
        opts = {"result_handling":"file"}
        self._basic_simulation(opts)
    
    @testattr(stddist = True)
    def test_basic_simulation_mat_file(self):
        opts = {"result_handling":"binary"}
        self._basic_simulation(opts)
    
    @testattr(stddist = True)
    def test_basic_simulation_memory(self):
        opts = {"result_handling":"memory"}
        self._basic_simulation(opts)
    
    @testattr(stddist = True)
    def test_basic_simulation_mat_file_naming(self):
        opts = {"result_handling":"binary", "result_file_name": "Should fail..."}
        
        try:
            self._basic_simulation(opts)
        except UnrecognizedOptionError:
            pass
        
        opts = {"result_handling":"binary", "result_file_name": {"Should fail..."}}
        
        try:
            self._basic_simulation(opts)
        except UnrecognizedOptionError:
            pass
    
    @testattr(stddist = True)
    def test_basic_simulation_mat_file_naming_exists(self):
        models, connections = self._load_basic_simulation()
        
        opts = {"result_handling":"binary", "result_file_name": {models[0]: "Test1.mat", models[1]: "Test2.mat"}}
        
        res = self._sim_basic_simulation(models, connections, opts)
        
        assert os.path.isfile("Test1.mat"), "Test1.mat does not exists"
        assert os.path.isfile("Test2.mat"), "Test2.mat does not exists"
    
    @testattr(stddist = True)
    def test_basic_simulation_txt_file_naming_exists(self):
        models, connections = self._load_basic_simulation()
        
        opts = {"result_handling":"file", "result_file_name": {models[0]: "Test1.txt", models[1]: "Test2.txt"}}
        
        res = self._sim_basic_simulation(models, connections, opts)
        
        assert os.path.isfile("Test1.txt"), "Test1.txt does not exists"
        assert os.path.isfile("Test2.txt"), "Test2.txt does not exists"
    
    @testattr(stddist = True)
    def test_basic_simulation_csv_file_naming_exists(self):
        models, connections = self._load_basic_simulation()
        
        opts = {"result_handling":"csv", "result_file_name": {models[0]: "Test1.csv", models[1]: "Test2.csv"}}
        
        res = self._sim_basic_simulation(models, connections, opts)
        
        assert os.path.isfile("Test1.csv"), "Test1.csv does not exists"
        assert os.path.isfile("Test2.csv"), "Test2.csv does not exists"
    
    @testattr(stddist = True)
    def test_basic_simulation_none_result(self):
        models, connections = self._load_basic_simulation()
        
        opts = {"result_handling":None}
        
        master = Master(models, connections)
       
        opts["step_size"] = 0.0005
        res = master.simulate(options=opts)
        
        assert res[models[0]]._result_data == None, "Result is not none"
        assert res[models[1]]._result_data == None, "Result is not none"
    
    @testattr(stddist = True)
    def test_custom_result_handler_invalid(self):
        models, connections = self._load_basic_simulation()
        
        class A:
            pass
                
        opts = {}
        opts["result_handling"] = "hejhej"
        nose.tools.assert_raises(Exception, self._sim_basic_simulation, models, connections, opts)

        opts["result_handling"] = "custom"
        opts["result_handler"] = A()
        err = "'result_handler' option must be a dictionary for 'result_handling' = 'custom'."
        with nose.tools.assert_raises_regex(FMUException, err):
            self._sim_basic_simulation(models, connections, opts)

        opts["result_handling"] = "custom"
        opts["result_handler"] = {m: A() for m in models[1:]}
        err = "'result_handler' option does not contain result handler for model '{}'".format(models[0].get_identifier())
        with nose.tools.assert_raises_regex(FMUException, err):
            self._sim_basic_simulation(models, connections, opts)

        opts["result_handling"] = "custom"
        opts["result_handler"] = {m: A() for m in models}
        err = "The result handler needs to be an instance of ResultHandler."
        with nose.tools.assert_raises_regex(FMUException, err):
            self._sim_basic_simulation(models, connections, opts)       
        
    @testattr(stddist = True)
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
        
    @testattr(stddist = True)
    def test_basic_simulation_with_block_initialization(self):
        opts = {"block_initialization": True}
        self._basic_simulation(opts)
    
    @testattr(stddist = True)
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

        master = Master(models,connections)

        opts = master.simulate_options()
        opts["block_initialization"] = True

        res = master.simulate(start_time=0.0, final_time=2.0, options=opts)
        
        assert res[model_sub2]["u"][0] == 1
        assert res[model_sub2]["u"][-1] == 3
    
    @testattr(stddist = True)
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
    
    @testattr(stddist = True)
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

    def _verify_downsample_result(self, ref_trajs, test_trajs, steps, factor):
        """Auxiliary function for result_downsampling_factor testing. 
        Verify correct length and values of downsampled trajectories."""
        # all steps, except last one are checked
        exptected_result_size = (steps - 1)//factor + 2
        assert len(test_trajs[0]) == exptected_result_size, f"expected result size: {exptected_result_size}, actual : {len(ref_trajs[0])}"
        assert len(test_trajs[1]) == exptected_result_size, f"expected result size: {exptected_result_size}, actual : {len(ref_trajs[1])}"

        # selection mask for reference result
        downsample_indices = np.array([i%factor == 0 for i in range(steps+1)])
        downsample_indices[0] = True
        downsample_indices[-1] = True

        np.testing.assert_equal(ref_trajs[0][downsample_indices], test_trajs[0])
        np.testing.assert_equal(ref_trajs[1][downsample_indices], test_trajs[1])

    def test_downsample_result(self):
        """ Test multiple result_downsampling_factor value and verify the result. """
        opts_update = {'result_downsampling_factor': 1}
        models, connections = self._load_basic_simulation()
        ref_res = self._sim_basic_simulation(models, connections, opts_update)

        ref_traj_length = 2001
        assert len(ref_res[models[0]]['time']) == ref_traj_length
        assert len(ref_res[models[1]]['time']) == ref_traj_length

        ref_res_trajs = (ref_res[models[0]]['x1'].copy(), ref_res[models[1]]['x2'].copy())

        for f in [2, 3, 4, 5, 10, 100, 250, 600, 3000]:
            opts_update = {'result_downsampling_factor': f}
            models, connections = self._load_basic_simulation()
            test_res = self._sim_basic_simulation(models, connections, opts_update)

            test_res_trajs = (test_res[models[0]]['x1'], test_res[models[1]]['x2'])
            self._verify_downsample_result(ref_res_trajs, test_res_trajs, 2000, f)

    def test_downsample_error_check_invalid_value(self):
        """ Verify we get an exception if the option is set to anything less than 1. """
        models, connections = self._load_basic_simulation()
        test_values = [-10, -20, -1, 0]

        # TODO: tidy up with pytest
        expected_substr = "Valid values for option 'result_downsampling_factor' are only positive integers"
        for value in test_values:
            try:
                self._sim_basic_simulation(models, connections, {'result_downsampling_factor': value})
                error_raised = False
            except FMUException as e:
                error_raised = True
                assert expected_substr in str(e), f"Error was {str(e)}, expected substring {expected_substr}"
            assert error_raised

    def test_error_check_invalid_value(self):
        """ Verify we get an exception if the option is set to anything that is not an integer. """
        models, connections = self._load_basic_simulation()
        test_values = [1/2, 1/3, "0.5", False]

        # TODO: tidy up with pytest
        expected_substr = "Option 'result_downsampling_factor' must be an integer,"
        for value in test_values:
            try:
                self._sim_basic_simulation(models, connections, {'result_downsampling_factor': value})
                error_raised = False
            except FMUException as e:
                error_raised = True
                assert expected_substr in str(e), f"Error was {str(e)}, expected substring {expected_substr}"
            assert error_raised
        
    # TODO: Test case that supports storing FMU states required
    @nose.tools.nottest
    def test_error_controlled_with_downsampling(self):
        models, connections = self._load_basic_simulation()
        uptate_options = {'result_downsampling_factor': 2,
                          'error_controlled': True}
        # TODO: Tidy up with pytest
        msg = "Result downsampling not supported for error controlled simulation, no downsampling will be performed."
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("default")
            self._sim_basic_simulation(models, connections, uptate_options)
            # there will be some other warnings from FMU loading
            assert f"UserWarning('{msg}')" in [i.message for i in w]

    def test_downsample_result_with_store_step_before_update(self):
        """ Test result_downsampling_factor with store_step_before_update. """
        opts_update = {'result_downsampling_factor': 1,
                       'store_step_before_update': True}
        models, connections = self._load_basic_simulation()
        ref_res = self._sim_basic_simulation(models, connections, opts_update)
        # default setting = 2000 steps

        assert len(ref_res[models[0]]['time']) == 4001
        assert len(ref_res[models[1]]['time']) == 4001

        ref_res_traj_1 = ref_res[models[0]]['x1'].copy()[1:]
        ref_res_traj_2 = ref_res[models[1]]['x2'].copy()[1:]

        # store_step_before_update points
        ref_res_traj_1_update = ref_res_traj_1[::2]
        ref_res_traj_2_update = ref_res_traj_2[::2]

        # ordinary solution points (minus start point)
        ref_res_traj_1_base = ref_res_traj_1[1::2]
        ref_res_traj_2_base = ref_res_traj_2[1::2]

        # down-sampled variant
        opts_update = {'result_downsampling_factor': 2,
                       'store_step_before_update': True}
        models, connections = self._load_basic_simulation()
        test_res = self._sim_basic_simulation(models, connections, opts_update)

        assert len(test_res[models[0]]['time']) == 2001
        assert len(test_res[models[1]]['time']) == 2001

        test_res_traj_1 = test_res[models[0]]['x1'].copy()[1:]
        test_res_traj_2 = test_res[models[1]]['x2'].copy()[1:]

        # store_step_before_update points
        test_res_traj_1_update = test_res_traj_1[::2]
        test_res_traj_2_update = test_res_traj_2[::2]

        # ordinary solution points (minus start point)
        test_res_traj_1_base = test_res_traj_1[1::2]
        test_res_traj_2_base = test_res_traj_2[1::2]

        # compare; [1::2] is the downsampling mask, since start-point is not included
        # and lengths align on final point
        np.testing.assert_equal(ref_res_traj_1_update[1::2], test_res_traj_1_update)
        np.testing.assert_equal(ref_res_traj_2_update[1::2], test_res_traj_2_update)

        np.testing.assert_equal(ref_res_traj_1_base[1::2], test_res_traj_1_base)
        np.testing.assert_equal(ref_res_traj_2_base[1::2], test_res_traj_2_base)
