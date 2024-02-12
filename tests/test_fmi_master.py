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

from pyfmi import Master
from pyfmi.fmi import FMUException, FMUModelCS2, FMUModelME2
from pyfmi.tests.test_util import Dummy_FMUModelCS2
from pyfmi.common.algorithm_drivers import UnrecognizedOptionError

file_path = os.path.dirname(os.path.abspath(__file__))

cs2_xml_path = os.path.join(file_path, "files", "FMUs", "XML", "CS2.0")
me2_xml_path = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0")

import warnings
warnings.filterwarnings("ignore")

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

        master = Master(models,connections)

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
