#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2020-2021 Modelon AB
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

import os

from pyfmi import load_fmu, testattr
from pyfmi.fmi import FMUException
from pyfmi.common.log import extract_xml_log, parse_xml_log
from pyfmi.common.diagnostics import DIAGNOSTICS_PREFIX
from pyfmi.tests.test_util import Dummy_FMUModelME2
from pyfmi.fmi_util import decode

import numpy as np
file_path = os.path.dirname(os.path.abspath(__file__))
logs = os.path.join(file_path, "files", "Logs")

class Test_Log:
    
    @testattr(stddist = True)
    def test_decode_bytes(self):
        """
        Verifies that malformed strings are still accepted and don't cause exceptions
        """
        b_string = b'[WARNING][FMU status:Warning]           <ModelicaError category="warning"><value name="msg">"\xc0\x15"</value></ModelicaError>'
        
        s_string = decode(b_string)
        
        assert s_string == '[WARNING][FMU status:Warning]           <ModelicaError category="warning"><value name="msg">"�\x15"</value></ModelicaError>', s_string
        

    @testattr(stddist = True)
    def test_extract_log(self):
        extract_xml_log("Tmp1.xml", os.path.join(logs, "CoupledClutches_log.txt"), modulename = 'Model')

        assert os.path.exists("Tmp1.xml")

        log = parse_xml_log("Tmp1.xml")

        assert "<JMIRuntime node with 3 subnodes, and named subnodes ['build_date', 'build_time']>" == str(log.nodes[1]), "Got: " + str(log.nodes[1])

    @testattr(stddist = True)
    def test_extract_log_exception(self):
        try:
            extract_xml_log("Tmp2", os.path.join(logs, "CoupledClutches_log_.txt"), modulename = 'Model')
        except FileNotFoundError:
            pass

    @testattr(stddist = True)
    def test_extract_log_cs(self):
        extract_xml_log("Tmp3.xml", os.path.join(logs, "CoupledClutches_CS_log.txt"), modulename = 'Slave')

        assert os.path.exists("Tmp3.xml")

        log = parse_xml_log("Tmp3.xml")

        assert "<JMIRuntime node with 3 subnodes, and named subnodes ['build_date', 'build_time']>" == str(log.nodes[1]), "Got: " + str(log.nodes[1])

    @testattr(stddist = True)
    def test_extract_log_wrong_modulename(self):
        extract_xml_log("Tmp4.xml", os.path.join(logs, "CoupledClutches_CS_log.txt"), modulename = 'Test')

        assert os.path.exists("Tmp4.xml")

        log = parse_xml_log("Tmp4.xml")

        try:
            log.nodes[1]
        except IndexError: #Test that the log is empty
            pass

    def _test_logging_different_solver(self, solver_name):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "Bouncing_Ball.fmu"), _connect_dll=False)
        opts=model.simulate_options()
        opts["dynamic_diagnostics"] = True
        opts["solver"] = solver_name
        res = model.simulate(options=opts)
        res_vars = res.keys()
        full_solver_name = f"{DIAGNOSTICS_PREFIX}solver.solver_name.{solver_name}"
        assert full_solver_name in res_vars, f"Missing {full_solver_name} in results!"
        assert f"{DIAGNOSTICS_PREFIX}step_time" in res_vars, f"Missing {DIAGNOSTICS_PREFIX}step_time in results!"
        np.testing.assert_equal(res[f'{DIAGNOSTICS_PREFIX}step_time'], res['time'], f"Expected {DIAGNOSTICS_PREFIX}step_time and time to be equal but they weren't!")
        np.testing.assert_equal(len(res[f'{DIAGNOSTICS_PREFIX}cpu_time']), len(res['time']),
                        f"Expected {DIAGNOSTICS_PREFIX}cpu_time and time to be of equal length but they weren't!")
        assert np.all(np.diff(res[f'{DIAGNOSTICS_PREFIX}nbr_steps'])>= 0), "Expected cumulative number of steps to increase, but wasn't!"
        np.testing.assert_equal(len(res[f'{DIAGNOSTICS_PREFIX}nbr_steps']), len(res['time']),
                        f"Expected {DIAGNOSTICS_PREFIX}cpu_time and time to be of equal length but they weren't!")
        np.testing.assert_equal(len(res['time']), len(res['h']), "Expected time and h to be of equal length but they weren't!")
        return res

    @testattr(stddist = True)
    def test_logging_option_CVode(self):
        res = self._test_logging_different_solver("CVode")
        t = res['time']
        np.testing.assert_equal(len(t), len(res[f'{DIAGNOSTICS_PREFIX}solver.solver_order']), "Unequal length for time and solver_order!")
        event_type = list(res[f'{DIAGNOSTICS_PREFIX}event_data.event_info.event_type'])
        assert event_type.count(-1) == len(event_type), "Expected no events to have happened!"
        assert (f'{DIAGNOSTICS_PREFIX}state_errors.h' in res.keys()), f"'{DIAGNOSTICS_PREFIX}state_errors.h' should be part of result variables!"


    @testattr(stddist = True)
    def test_logging_option_Radau5ODE(self):
        res = self._test_logging_different_solver("Radau5ODE")
        event_type = list(res[f'{DIAGNOSTICS_PREFIX}event_data.event_info.event_type'])
        assert event_type.count(-1) == len(event_type), "Expected no events to have happened!"
        assert (f'{DIAGNOSTICS_PREFIX}state_errors.h' in res.keys()), f"'{DIAGNOSTICS_PREFIX}state_errors.h' should be part of result variables!"

    @testattr(stddist = True)
    def test_logging_option_ImplicitEuler(self):
        res = self._test_logging_different_solver("ImplicitEuler")
        assert not (f'{DIAGNOSTICS_PREFIX}state_errors.h' in res.keys()), f"'{DIAGNOSTICS_PREFIX}state_errors.h' should not be part of result variables!"

    @testattr(stddist = True)
    def test_logging_option_ExplicitEuler(self):
        res = self._test_logging_different_solver("ExplicitEuler")
        assert not (f'{DIAGNOSTICS_PREFIX}state_errors.h' in res.keys()), f"'{DIAGNOSTICS_PREFIX}state_errors.h' should not be part of result variables!"

    @testattr(stddist = True)
    def test_logging_option_LSODAR(self):
        res = self._test_logging_different_solver("LSODAR")
        event_type = list(res[f'{DIAGNOSTICS_PREFIX}event_data.event_info.event_type'])
        assert event_type.count(-1) == len(event_type), "Expected no events to have happened, but event_type contains: {}!".format(event_type)

    @testattr(stddist = True)
    def test_calculated_diagnostic(self):
         res = self._test_logging_different_solver("CVode")
         np.testing.assert_equal(len(res['time']), len(res[f'{DIAGNOSTICS_PREFIX}nbr_steps']),
            "Expected time and Diagnostics.nbr_steps to be of equal length but they weren't!")
         np.testing.assert_equal(len(res['time']), len(res[f'{DIAGNOSTICS_PREFIX}nbr_time_events']),
            "Expected time and Diagnostics.nbr_time_events to be of equal length but they weren't!")
         np.testing.assert_equal(len(res['time']), len(res[f'{DIAGNOSTICS_PREFIX}nbr_state_events']),
            "Expected time and Diagnostics.nbr_state_events to be of equal length but they weren't!")
         np.testing.assert_equal(len(res['time']), len(res[f'{DIAGNOSTICS_PREFIX}nbr_events']),
            "Expected time and Diagnostics.nbr_events to be of equal length but they weren't!")
         np.testing.assert_equal(len(res['time']), len(res[f'{DIAGNOSTICS_PREFIX}nbr_state_limits_step.h']),
            "Expected time and Diagnostics.nbr_state_limits_step.h to be of equal length but they weren't!")


    @testattr(stddist = True)
    def test_extract_boolean_value(self):
        log = parse_xml_log(os.path.join(logs, "boolean_log.xml"))
        eis = log.find("EventInfo")
        for ei in eis:
            assert isinstance(ei.time_event_info, bool), "Expected ei.time_event_info to be bool"

    @testattr(stddist = True)
    def test_hasattr_works(self):
        """
        Tests that 'hasattr' works on the log nodes.
        """
        log = parse_xml_log(os.path.join(logs, "boolean_log.xml"))
        event_node = log.find("EventInfo")[0]
        
        assert hasattr(event_node, "t")
        assert not hasattr(event_node, "not_in_node")
        
        event_node.t #Should not cause exception
        
        try:
            event_node.not_in_node
            raise Exception("An exception was not raised for 'event_node.not_in_node'")
        except AttributeError:
            pass
