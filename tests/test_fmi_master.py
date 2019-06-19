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
from pyfmi.fmi import FMUModel, FMUException, FMUModelME1, FMUModelCS1, load_fmu, FMUModelCS2, FMUModelME2, PyEventInfo
from pyfmi.fmi_coupled import CoupledFMUModelME2
from pyfmi.fmi_extended import FMUModelME1Extended
import pyfmi.fmi_util as fmi_util
import pyfmi.fmi as fmi
import pyfmi.fmi_algorithm_drivers as fmi_algorithm_drivers
from pyfmi import Master
from pyfmi.tests.test_util import Dummy_FMUModelME2
from pyfmi.common.io import ResultHandler

file_path = os.path.dirname(os.path.abspath(__file__))

cs2_xml_path = os.path.join(file_path, "files", "FMUs", "XML", "CS2.0")
me2_xml_path = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0")

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
    def test_basic_simulation(self):
        model = load_fmu(self.linear_full)
       
        model_sub1 = load_fmu(self.linear_sub1)
        model_sub2 = load_fmu(self.linear_sub2)
       
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
                   
        master = Master(models, connections)
       
        opts = master.simulate_options()
        opts["step_size"] = 0.0005
       
        res = master.simulate(options=opts)
       
        res_full = model.simulate()
       
        nose.tools.assert_almost_equal(res[model_sub1].final("x1"), res_full.final("p1.x1"), 2)
        nose.tools.assert_almost_equal(res[model_sub2].final("x2"), res_full.final("p2.x2"), 2)
       
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
    def test_unstable_simulation(self):
        model = load_fmu(self.linear_full)
       
        model_sub1 = load_fmu(self.linear_sub1)
        model_sub2 = load_fmu(self.linear_sub2)
        model_sub2.set("d2", 1.1) #Coupled system becomes unstable
       
        models = [model_sub1, model_sub2]
        connections = [(model_sub1,"y1",model_sub2,"u2"),
                   (model_sub2,"y2",model_sub1,"u1")]
                   
        master = Master(models, connections)
       
        opts = master.simulate_options()
        opts["step_size"] = 0.0005
       
        res = master.simulate(final_time=0.3, options=opts)
       
        assert abs(res[model_sub1].final("x1")) > 100
        assert abs(res[model_sub2].final("x2")) > 100

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
