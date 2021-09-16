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

from numpy import testing
from numpy.testing._private.utils import assert_equal

from pyfmi import testattr
from pyfmi.common.log import extract_xml_log, parse_xml_log
from pyfmi.tests.test_util import Dummy_FMUModelME2

import numpy as np
file_path = os.path.dirname(os.path.abspath(__file__))
logs = os.path.join(file_path, "files", "Logs")

class Test_Log:

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
        model = Dummy_FMUModelME2([], "Bouncing_Ball.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        opts=model.simulate_options()
        opts["dynamic_diagnostics"] = True
        opts["solver"] = solver_name
        res = model.simulate(options=opts)
        res_vars = res.keys()
        assert "Diagnostics.solver.{}".format(solver_name) in res_vars, "Missing Diagnostics.solver.{} in results!".format(solver_name)
        assert "Diagnostics.step_time" in res_vars, "Missing Diagnostics.step_time in results!"
        np.testing.assert_equal(res['Diagnostics.step_time'], res['time'], "Expected Diagnostics.step_time and time to be equal but they weren't!")
        np.testing.assert_equal(len(res['Diagnostics.cum_cpu_time']), len(res['time']),
                        "Expected Diagnostics.cum_cpu_time and time to be of equal length but they weren't!")
        assert np.all(np.diff(res['Diagnostics.solver.cum_nbr_steps'])>= 0), "Expected cumulative number of steps to increase, but wasn't!"
        np.testing.assert_equal(len(res['Diagnostics.solver.cum_nbr_steps']), len(res['time']),
                        "Expected Diagnostics.cum_cpu_time and time to be of equal length but they weren't!")
        np.testing.assert_equal(len(res['time']), len(res['h']), "Expected time and h to be of equal length but they weren't!")
        return res

    @testattr(stddist = True)
    def test_logging_option_CVode(self):
        res = self._test_logging_different_solver("CVode")
        t = res['time']
        np.testing.assert_equal(len(t), len(res['Diagnostics.solver.solver_order']), "Unequal length for time and solver_order!")
        event_type = list(res['Diagnostics.event_data.event_info.event_type'])
        assert event_type.count(-1) == len(event_type), "Expected no events to have happened!"
        assert ('Diagnostics.state_errors.h' in res.keys()), "'Diagnostics.state_errors.h' should be part of result variables!"


    @testattr(stddist = True)
    def test_logging_option_Radau5ODE(self):
        res = self._test_logging_different_solver("Radau5ODE")
        event_type = list(res['Diagnostics.event_data.event_info.event_type'])
        assert event_type.count(-1) == len(event_type), "Expected no events to have happened!"
        assert ('Diagnostics.state_errors.h' in res.keys()), "'Diagnostics.state_errors.h' should be part of result variables!"

    @testattr(stddist = True)
    def test_logging_option_ImplicitEuler(self):
        res = self._test_logging_different_solver("ImplicitEuler")
        assert not ('Diagnostics.state_errors.h' in res.keys()), "'Diagnostics.state_errors.h' should not be part of result variables!"

    @testattr(stddist = True)
    def test_logging_option_ExplicitEuler(self):
        res = self._test_logging_different_solver("ExplicitEuler")
        assert not ('Diagnostics.state_errors.h' in res.keys()), "'Diagnostics.state_errors.h' should not be part of result variables!"

    @testattr(stddist = True)
    def test_logging_option_LSODAR(self):
        res = self._test_logging_different_solver("LSODAR")
        event_type = list(res['Diagnostics.event_data.event_info.event_type'])
        assert event_type.count(-1) == len(event_type), "Expected no events to have happened, but event_type contains: {}!".format(event_type)

    @testattr(stddist = True)
    def test_calculated_diagnostic(self):
         res = self._test_logging_different_solver("CVode")
         np.testing.assert_equal(len(res['time']), len(res['Diagnostics.solver.cum_nbr_steps']),
            "Expected time and Diagnostics.solver.cum_nbr_steps to be of equal length but they weren't!")
         np.testing.assert_equal(len(res['time']), len(res['Diagnostics.event_data.cum_nbr_time_events']),
            "Expected time and Diagnostics.event_data.cum_nbr_time_events to be of equal length but they weren't!")
         np.testing.assert_equal(len(res['time']), len(res['Diagnostics.event_data.cum_nbr_state_events']),
            "Expected time and Diagnostics.event_data.cum_nbr_state_events to be of equal length but they weren't!")
         np.testing.assert_equal(len(res['time']), len(res['Diagnostics.event_data.cum_nbr_events']),
            "Expected time and Diagnostics.event_data.cum_nbr_events to be of equal length but they weren't!")
         np.testing.assert_equal(len(res['time']), len(res['Diagnostics.cum_nbr_state_limits_step.h']),
            "Expected time and Diagnostics.cum_nbr_state_limits_step.h to be of equal length but they weren't!")


    @testattr(stddist = True)
    def test_extract_boolean_value(self):
        log = parse_xml_log(os.path.join(logs, "boolean_log.xml"))
        eis = log.find("EventInfo")
        for ei in eis:
            assert isinstance(ei.time_event_info, bool), "Expected ei.time_event_info to be bool"
