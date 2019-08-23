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
import pyfmi.fmi_util as fmi_util
import pyfmi.fmi as fmi
from pyfmi import Master
from pyfmi.tests.test_util import Dummy_FMUModelME2, Dummy_FMUModelCS2

file_path = os.path.dirname(os.path.abspath(__file__))

cs2_xml_path = os.path.join(file_path, "files", "FMUs", "XML", "CS2.0")
me2_xml_path = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0")

import warnings
warnings.filterwarnings("ignore")

class Test_Master:
    
    @testattr(stddist = True)
    def test_loading_models(self):
        model_sub1 = FMUModelCS2("LinearStability.SubSystem1.fmu", cs2_xml_path, _connect_dll=False)
        model_sub2 = FMUModelCS2("LinearStability.SubSystem2.fmu", cs2_xml_path, _connect_dll=False)
       
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
        
        #Assert that loading is successful
        sim = Master(models, connections)
    
    @testattr(stddist = True)
    def test_loading_wrong_model(self):
        model_sub1 = FMUModelCS2("LinearStability.SubSystem1.fmu", cs2_xml_path, _connect_dll=False)
        model_sub2 = FMUModelME2("LinearStability.SubSystem2.fmu", me2_xml_path, _connect_dll=False)
       
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
        
        nose.tools.assert_raises(FMUException, Master, models, connections)
    
    @testattr(stddist = True)
    def test_connection_variables(self):
        model_sub1 = FMUModelCS2("LinearStability.SubSystem1.fmu", cs2_xml_path, _connect_dll=False)
        model_sub2 = FMUModelCS2("LinearStability.SubSystem2.fmu", cs2_xml_path, _connect_dll=False)
       
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
        model_sub1 = FMUModelCS2("LinearStability.SubSystem1.fmu", cs2_xml_path, _connect_dll=False)
        model_sub2 = FMUModelCS2("LinearStability.SubSystem2.fmu", cs2_xml_path, _connect_dll=False)
        
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
                   
        sim = Master(models, connections)
        assert sim.algebraic_loops
       
        model_sub1 = FMUModelCS2("LinearStability_LinearSubSystemNoFeed1.fmu", cs2_xml_path, _connect_dll=False)
        model_sub2 = FMUModelCS2("LinearStability_LinearSubSystemNoFeed2.fmu", cs2_xml_path, _connect_dll=False)
       
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
                   
        sim = Master(models, connections)
        assert not sim.algebraic_loops
    
    def _basic_simulation(self, result_handling):
        model_sub1 = Dummy_FMUModelCS2([], "LinearCoSimulation_LinearSubSystem1.fmu", cs2_xml_path, _connect_dll=False)
        model_sub2 = Dummy_FMUModelCS2([], "LinearCoSimulation_LinearSubSystem2.fmu", cs2_xml_path, _connect_dll=False)
        
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
            model_sub1.values[model_sub1.get_variable_valueref("y1")] = c1*model_sub1.continuous_states+d1*u1
            model_sub1.completed_integrator_step()
            return 0
        
        def do_step2(current_t, step_size, new_step=True):
            u2 = model_sub2.values[model_sub2.get_variable_valueref("u2")]
            
            model_sub2.continuous_states = 1.0/a2*(np.exp(a2*step_size)-1.0)*b2*u2+np.exp(a2*step_size)*model_sub2.continuous_states
            model_sub2.values[model_sub2.get_variable_valueref("y2")] = c2*model_sub2.continuous_states+d2*u2
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
        opts["result_handling"] = result_handling
        
        res = master.simulate(options=opts)
        
        nose.tools.assert_almost_equal(res[model_sub1].final("x1"), 0.0859764038708439, 3)
        nose.tools.assert_almost_equal(res[model_sub2].final("x2"), 0.008392664839635064, 4)
    
    @testattr(stddist = True)
    def test_basic_simulation_txt_file(self):
        self._basic_simulation(result_handling="file")
    
    @testattr(stddist = True)
    def test_basic_simulation_mat_file(self):
        self._basic_simulation(result_handling="binary")
    
    @testattr(stddist = True)
    def test_integer_connections(self):
        model_sub1 = Dummy_FMUModelCS2([], "IntegerStep.fmu", cs2_xml_path, _connect_dll=False)
        model_sub2 = Dummy_FMUModelCS2([], "GainTestInteger.fmu", cs2_xml_path, _connect_dll=False)
        
        model_sub1.set("y", 1)
        def do_step1(current_t, step_size, new_step=True):
            model_sub1.values[model_sub1.get_variable_valueref("y")] = 1 if current_t+step_size < 0.5 else 3
            model_sub1.completed_integrator_step()
            return 0
        
        def do_step2(current_t, step_size, new_step=True):
            u = model_sub2.values[model_sub2.get_variable_valueref("u")]
            model_sub2.values[model_sub2.get_variable_valueref("y")] = 10*u
            model_sub2.completed_integrator_step()
            return 0
            
        model_sub1.do_step = do_step1
        model_sub2.do_step = do_step2
        
        models = [model_sub1, model_sub2]
        connections=[(model_sub1,'y',model_sub2,'u')]

        master = Master(models,connections)

        opts = master.simulate_options()
        opts["block_initialization"] = True

        res = master.simulate(start_time=0.0,final_time=2.0, options=opts)
        
        assert res[model_sub2]["u"][0] == 1
        assert res[model_sub2]["u"][-1] == 3
    
    @testattr(stddist = True)
    def test_integer_to_real_connections(self):
        model_sub1 = Dummy_FMUModelCS2([], "IntegerStep.fmu", cs2_xml_path, _connect_dll=False)
        model_sub2 = Dummy_FMUModelCS2([], "GainTestReal.fmu", cs2_xml_path, _connect_dll=False)
        
        model_sub1.set("y", 1)
        def do_step1(current_t, step_size, new_step=True):
            model_sub1.values[model_sub1.get_variable_valueref("y")] = 1 if current_t+step_size < 0.5 else 3
            model_sub1.completed_integrator_step()
            return 0
        
        def do_step2(current_t, step_size, new_step=True):
            u = model_sub2.values[model_sub2.get_variable_valueref("u")]
            model_sub2.values[model_sub2.get_variable_valueref("y")] = 10*u
            model_sub2.completed_integrator_step()
            return 0
            
        model_sub1.do_step = do_step1
        model_sub2.do_step = do_step2
        
        models = [model_sub1, model_sub2]
        connections=[(model_sub1,'y',model_sub2,'u')]

        master = Master(models,connections)

        opts = master.simulate_options()
        opts["block_initialization"] = True

        res = master.simulate(start_time=0.0,final_time=2.0, options=opts)
        
        assert res[model_sub2]["u"][0] == 1.0
        assert res[model_sub2]["u"][-1] == 3.0
    
    @testattr(stddist = True)
    def test_unstable_simulation(self):
        model_sub1 = Dummy_FMUModelCS2([], "LinearCoSimulation_LinearSubSystem1.fmu", cs2_xml_path, _connect_dll=False)
        model_sub2 = Dummy_FMUModelCS2([], "LinearCoSimulation_LinearSubSystem2.fmu", cs2_xml_path, _connect_dll=False)
        
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
            model_sub1.values[model_sub1.get_variable_valueref("y1")] = c1*model_sub1.continuous_states+d1*u1
            model_sub1.completed_integrator_step()
            return 0
        
        def do_step2(current_t, step_size, new_step=True):
            u2 = model_sub2.values[model_sub2.get_variable_valueref("u2")]
            
            model_sub2.continuous_states = 1.0/a2*(np.exp(a2*step_size)-1.0)*b2*u2+np.exp(a2*step_size)*model_sub2.continuous_states
            model_sub2.values[model_sub2.get_variable_valueref("y2")] = c2*model_sub2.continuous_states+d2*u2
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


"""
To be migrated:

       
class Test_Master_Simulation:
    @classmethod
    def setUpClass(cls):
        file_name = os.path.join(get_files_path(), 'Modelica', 'LinearCoSimulation.mo')
       
        cls.linear_full = compile_fmu("LinearCoSimulation.LinearFullSystem", file_name, target="cs", version="2.0")
        cls.linear_sub1 = compile_fmu("LinearCoSimulation.LinearSubSystem1", file_name, target="cs", version="2.0")
        cls.linear_sub2 = compile_fmu("LinearCoSimulation.LinearSubSystem2", file_name, target="cs", version="2.0")
       
        compiler_options={"generate_ode_jacobian": True}
       
        cls.linear_sub1_dir = compile_fmu("LinearCoSimulation.LinearSubSystem1", file_name, target="cs", version="2.0",
                                    compiler_options={"generate_ode_jacobian": True}, compile_to="LinearSub1Dir.fmu")
        cls.linear_sub2_dir = compile_fmu("LinearCoSimulation.LinearSubSystem2", file_name, target="cs", version="2.0",
                                    compiler_options={"generate_ode_jacobian": True}, compile_to="LinearSub2Dir.fmu")

    @testattr(stddist = True)
    def test_linear_correction_simulation(self):
        model = load_fmu(self.linear_full)
       
        model_sub1 = load_fmu(self.linear_sub1_dir)
        model_sub2 = load_fmu(self.linear_sub2_dir)
       
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
                   
        master = Master(models, connections)
       
        opts = master.simulate_options()
        opts["step_size"] = 0.0005
        opts["extrapolation_order"] = 1
        opts["linear_correction"] = True
       
        res = master.simulate(options=opts)
       
        res_full = model.simulate()
       
        nose.tools.assert_almost_equal(res[model_sub1].final("x1"), res_full.final("p1.x1"), 2)
        nose.tools.assert_almost_equal(res[model_sub2].final("x2"), res_full.final("p2.x2"), 2)
       
    @testattr(stddist = True)
    def test_basic_simulation_extrapolation(self):
        model = load_fmu(self.linear_full)
       
        model.set("p1.d1", 0.1)
       
        model_sub1 = load_fmu(self.linear_sub1)
        model_sub2 = load_fmu(self.linear_sub2)
       
        model_sub1.set("d1", 0.1)
       
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
                   
        master = Master(models, connections)
       
        opts = master.simulate_options()
        opts["step_size"] = 0.0005
        opts["extrapolation_order"] = 1
       
        res = master.simulate(options=opts)
       
        res_full = model.simulate()
       
        nose.tools.assert_almost_equal(res[model_sub1].final("x1"), res_full.final("p1.x1"), 2)
        nose.tools.assert_almost_equal(res[model_sub2].final("x2"), res_full.final("p2.x2"), 2)
       
    @testattr(stddist = True)
    def test_unstable_simulation_extrapolation(self):
        model = load_fmu(self.linear_full)
       
        model_sub1 = load_fmu(self.linear_sub1)
        model_sub2 = load_fmu(self.linear_sub2)
       
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
                   
        master = Master(models, connections)
       
        opts = master.simulate_options()
        opts["step_size"] = 0.0005
        opts["extrapolation_order"] = 1
       
        res = master.simulate(final_time=0.2, options=opts)
       
        res_full = model.simulate()
       
        #The coupled system is unstable with extrapolation 1
        assert abs(res[model_sub1].final("x1")) > 100
        assert abs(res[model_sub2].final("x2")) > 100
       
        #import pylab as plt
        #plt.plot(res[model_sub1]["time"], res[model_sub1]["x1"])
        #plt.plot(res_full["time"], res_full["p1.x1"])
        #plt.show()

    @testattr(stddist = True)
    def test_initialize(self):
        model = load_fmu(self.linear_full)
        model.setup_experiment()
        model.initialize()
       
        model_sub1 = load_fmu(self.linear_sub1)
        model_sub2 = load_fmu(self.linear_sub2)
       
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
                   
        master = Master(models, connections)
       
        opts = master.simulate_options()
        opts["step_size"] = 0.0005
        opts["extrapolation_order"] = 1
       
        master.initialize(0, 1, opts)
       
        nose.tools.assert_almost_equal(model_sub1.get("y1"), model.get("p1.y1"), 2)
        nose.tools.assert_almost_equal(model_sub2.get("y2"), model.get("p2.y2"), 2)
       
        model_sub1 = load_fmu(self.linear_sub1)
        model_sub2 = load_fmu(self.linear_sub2)
       
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
                   
        master = Master(models, connections)
       
        opts = master.simulate_options()
        opts["step_size"] = 0.0005
        opts["extrapolation_order"] = 1
        opts["block_initialization"] = True
       
        master.initialize(0, 1, opts)
       
        nose.tools.assert_almost_equal(model_sub1.get("y1"), model.get("p1.y1"), 2)
        nose.tools.assert_almost_equal(model_sub2.get("y2"), model.get("p2.y2"), 2)
"""
