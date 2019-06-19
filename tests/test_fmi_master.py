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
