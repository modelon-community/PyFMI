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
import pyfmi.fmi_util as fmi_util
import pyfmi.fmi as fmi
import pyfmi.fmi_algorithm_drivers as fmi_algorithm_drivers
from pyfmi.tests.test_util import Dummy_FMUModelME2
from pyfmi.common.io import ResultHandler

file_path = os.path.dirname(os.path.abspath(__file__))

me2_xml_path = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0")

class Test_CoupledFMUModelME2:
    
    @testattr(stddist = True)
    def test_reversed_connections(self):
        model_sub_1 = FMUModelME2("LinearStability.SubSystem1.fmu", me2_xml_path, _connect_dll=False)
        model_sub_2 = FMUModelME2("LinearStability.SubSystem2.fmu", me2_xml_path, _connect_dll=False)
        model_full  = FMUModelME2("LinearStability.FullSystem.fmu", me2_xml_path, _connect_dll=False)
        
        models = [("First", model_sub_1), ("Second", model_sub_2)]
        connections = [(model_sub_2,"y1",model_sub_1,"u2"),
                       (model_sub_1,"y2",model_sub_2,"u1")]
        
        nose.tools.assert_raises(fmi.FMUException,  CoupledFMUModelME2, models, connections)
        
        connections = [(model_sub_2,"u2",model_sub_1,"y1"),
                       (model_sub_1,"u1",model_sub_2,"y2")]
                       
        nose.tools.assert_raises(fmi.FMUException,  CoupledFMUModelME2, models, connections)
    
    @testattr(stddist = True)
    def test_inputs_list(self):
        
        model_sub_1 = FMUModelME2("LinearStability.SubSystem1.fmu", me2_xml_path, _connect_dll=False)
        model_sub_2 = FMUModelME2("LinearStability.SubSystem2.fmu", me2_xml_path, _connect_dll=False)
        model_full  = FMUModelME2("LinearStability.FullSystem.fmu", me2_xml_path, _connect_dll=False)
        
        models = [("First", model_sub_1), ("Second", model_sub_2)]
        connections = [(model_sub_1,"y1",model_sub_2,"u2"),
                       (model_sub_2,"y2",model_sub_1,"u1")]
        
        coupled = CoupledFMUModelME2(models, connections=connections)

        #Inputs should not be listed if they are internally connected
        vars = coupled.get_input_list().keys()
        assert len(vars) == 0
        
        coupled = CoupledFMUModelME2(models, connections=[])
        vars = coupled.get_input_list().keys()
        assert "First.u1" in vars
        assert "Second.u2" in vars
    
    @testattr(stddist = True)
    def test_alias(self):
        model_cc_1 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        model_cc_2 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        
        models = [("First", model_cc_1), ("Second", model_cc_2)]
        connections = []
        
        coupled = CoupledFMUModelME2(models, connections)
        
        aliases = coupled.get_variable_alias("First.J4.phi")
        assert "First.J4.phi" in aliases.keys()
        assert coupled.get_variable_alias_base("First.J4.phi") == "First.J4.flange_a.phi"
    
    @testattr(stddist = True)
    def test_loading(self):
        model_cc_1 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        model_cc_2 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        
        models = [model_cc_1, model_cc_2]
        connections = []
        
        nose.tools.assert_raises(fmi.FMUException, CoupledFMUModelME2, models, connections)
        
        models = [("First", model_cc_1), model_cc_2]
        nose.tools.assert_raises(fmi.FMUException, CoupledFMUModelME2, models, connections)
        
        models = [("First", model_cc_1), ("First", model_cc_2)]
        nose.tools.assert_raises(fmi.FMUException, CoupledFMUModelME2, models, connections)
        
        models = [("First", model_cc_1), ("Second", model_cc_2)]
        coupled = CoupledFMUModelME2(models, connections)
        
        connections = [("k")]
        nose.tools.assert_raises(fmi.FMUException, CoupledFMUModelME2, models, connections)
        
        connections = [(model_cc_1, "J1.phi", model_cc_2, "J2.phi")]
        nose.tools.assert_raises(fmi.FMUException, CoupledFMUModelME2, models, connections)
    
    @testattr(stddist = True)
    def test_get_variable_valueref(self):
        model_cc_1 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        model_cc_2 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        
        models = [("First", model_cc_1), ("Second", model_cc_2)]
        connections = []
        
        coupled = CoupledFMUModelME2(models, connections)
        
        nose.tools.assert_raises(fmi.FMUException,  coupled.get_variable_valueref, "J1.w")
        
        vr_1 = coupled.get_variable_valueref("First.J1.w")
        vr_2 = coupled.get_variable_valueref("Second.J1.w")

        assert vr_1 != vr_2
        
        var_name_1 = coupled.get_variable_by_valueref(vr_1)
        var_name_2 = coupled.get_variable_by_valueref(vr_2)
        
        assert var_name_1 == "First.J1.w"
        assert var_name_2 == "Second.J1.w"
    
    @testattr(stddist = True)
    def test_ode_sizes(self):
        model_cc_1 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        model_cc_2 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        
        models = [("First", model_cc_1), ("Second", model_cc_2)]
        connections = []
        
        coupled = CoupledFMUModelME2(models, connections)
        
        [nbr_states, nbr_event_ind] = coupled.get_ode_sizes()
        
        assert nbr_states == 16
        assert nbr_event_ind == 66
    
    @testattr(stddist = True)
    def test_variable_variability(self):
        model_cc_1 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        model_cc_2 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        
        models = [("First", model_cc_1), ("Second", model_cc_2)]
        connections = []
        
        coupled = CoupledFMUModelME2(models, connections)
        
        nose.tools.assert_raises(fmi.FMUException,  coupled.get_variable_variability, "J1.w")
        
        variability = coupled.get_variable_variability("First.J1.w")
        
        assert variability == model_cc_1.get_variable_variability("J1.w")
    
    @testattr(stddist = True)
    def test_variable_causality(self):
        model_cc_1 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        model_cc_2 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        
        models = [("First", model_cc_1), ("Second", model_cc_2)]
        connections = []
        
        coupled = CoupledFMUModelME2(models, connections)
        
        nose.tools.assert_raises(fmi.FMUException,  coupled.get_variable_causality, "J1.w")
        
        causality = coupled.get_variable_causality("First.J1.w")
        
        assert causality == model_cc_1.get_variable_causality("J1.w")
    
    @testattr(stddist = True)
    def test_derivatives_list(self):
        model_cc_1 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        model_cc_2 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        
        models = [("First", model_cc_1), ("Second", model_cc_2)]
        connections = []
        
        coupled = CoupledFMUModelME2(models, connections)
        
        states = coupled.get_derivatives_list()
        
        for state in states:
            assert state.startswith("First.") or state.startswith("Second.")
            var = coupled.get_variable_by_valueref(states[state].value_reference)
            alias_vars = coupled.get_variable_alias(var).keys()
            assert state in alias_vars
    
    @testattr(stddist = True)
    def test_states_list(self):
        model_cc_1 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        model_cc_2 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        
        models = [("First", model_cc_1), ("Second", model_cc_2)]
        connections = []
        
        coupled = CoupledFMUModelME2(models, connections)
        
        states = coupled.get_states_list()
        
        for state in states:
            assert state.startswith("First.") or state.startswith("Second.")
            var = coupled.get_variable_by_valueref(states[state].value_reference)
            alias_vars = coupled.get_variable_alias(var).keys()
            assert state in alias_vars
    
    @testattr(stddist = True)
    def test_model_variables(self):
        model_cc_1 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        model_cc_2 = FMUModelME2("CoupledClutches.fmu", me2_xml_path, _connect_dll=False)
        
        models = [("First", model_cc_1), ("Second", model_cc_2)]
        connections = []
        
        coupled = CoupledFMUModelME2(models, connections)
        
        vars = coupled.get_model_variables()
        vars_1 = model_cc_1.get_model_variables()
        vars_2 = model_cc_2.get_model_variables()
        
        assert len(vars) == len(vars_1) + len(vars_2)
        
        vars = coupled.get_model_variables(include_alias=False)
        vars_1 = model_cc_1.get_model_variables(include_alias=False)
        vars_2 = model_cc_2.get_model_variables(include_alias=False)
        
        assert len(vars) == len(vars_1) + len(vars_2)
        
        vars = coupled.get_model_variables(include_alias=False, type=fmi.FMI2_INTEGER)
        vars_1 = model_cc_1.get_model_variables(include_alias=False, type=fmi.FMI2_INTEGER)
        vars_2 = model_cc_2.get_model_variables(include_alias=False, type=fmi.FMI2_INTEGER)
        
        assert len(vars) == len(vars_1) + len(vars_2)
    
    @testattr(stddist = True)
    def test_linear_example(self):
        model_sub_1 = Dummy_FMUModelME2([], "LinearStability.SubSystem1.fmu", me2_xml_path, _connect_dll=False)
        model_sub_2 = Dummy_FMUModelME2([], "LinearStability.SubSystem2.fmu", me2_xml_path, _connect_dll=False)

        def sub1(*args, **kwargs):
            u1 = model_sub_1.values[model_sub_1.get_variable_valueref("u1")]
            a1 = model_sub_1.values[model_sub_1.get_variable_valueref("a1")]
            b1 = model_sub_1.values[model_sub_1.get_variable_valueref("b1")]
            c1 = model_sub_1.values[model_sub_1.get_variable_valueref("c1")]
            d1 = model_sub_1.values[model_sub_1.get_variable_valueref("d1")]
            x1 = model_sub_1.continuous_states[0]
            model_sub_1.values[model_sub_1.get_variable_valueref("y1")] = c1*x1+d1*u1
            model_sub_1.values[model_sub_1.get_variable_valueref("x1")] = x1
            return np.array([a1*x1+b1*u1])
        
        def sub2(*args, **kwargs):
            u2 = model_sub_2.values[model_sub_2.get_variable_valueref("u2")]
            a2 = model_sub_2.values[model_sub_2.get_variable_valueref("a2")]
            b2 = model_sub_2.values[model_sub_2.get_variable_valueref("b2")]
            c2 = model_sub_2.values[model_sub_2.get_variable_valueref("c2")]
            d2 = model_sub_2.values[model_sub_2.get_variable_valueref("d2")]
            x2 = model_sub_2.continuous_states[0]
            model_sub_2.values[model_sub_2.get_variable_valueref("y2")] = c2*x2+d2*u2
            model_sub_2.values[model_sub_2.get_variable_valueref("x2")] = x2
            return np.array([a2*x2+b2*u2])
        
        model_sub_1.get_derivatives = sub1
        model_sub_2.get_derivatives = sub2
        
        models = [("First", model_sub_1), ("Second", model_sub_2)]
        connections = [(model_sub_1,"y1",model_sub_2,"u2"),
                       (model_sub_2,"y2",model_sub_1,"u1")]
        
        coupled = CoupledFMUModelME2(models, connections=connections)

        opts = {"CVode_options": {"rtol":1e-6, "atol":1e-6}}

        res = coupled.simulate(options=opts)

        nose.tools.assert_almost_equal(res.final("First.x1"),0.08597302307099872)
        nose.tools.assert_almost_equal(res.final("Second.x2"),0.0083923348082567)
        nose.tools.assert_almost_equal(res.initial("First.x1"),1.0)
        nose.tools.assert_almost_equal(res.initial("Second.x2"),1.0)
        
        nose.tools.assert_almost_equal(res.final("First.u1"),-0.25909975860402856)
        nose.tools.assert_almost_equal(res.final("Second.u2"),-0.0011806893910324295)
        nose.tools.assert_almost_equal(res.initial("First.u1"),-17.736842105263158)
        nose.tools.assert_almost_equal(res.initial("Second.u2"),-14.73684210526316)
