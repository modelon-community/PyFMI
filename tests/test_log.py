#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Copyright (C) 2020 Modelon AB
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

from pyfmi import testattr
from pyfmi.common.log import extract_xml_log, parse_xml_log
from pyfmi.tests.test_util import Dummy_FMUModelME2

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
        opts["logging"] = True
        opts["solver"] = solver_name
        model.simulate(options=opts)
        log_file = model.extract_xml_log()
        assert os.path.exists(log_file), "Missing log file for {}".format(solver_name)

    @testattr(stddist = True)
    def test_logging_option_CVode(self):
        self._test_logging_different_solver("CVode")
        
    @testattr(stddist = True)
    def test_logging_option_Radau5ODE(self):
        self._test_logging_different_solver("Radau5ODE")

    @testattr(stddist = True)
    def test_logging_option_ImplicitEuler(self):
        self._test_logging_different_solver("ImplicitEuler")

    @testattr(stddist = True)
    def test_logging_option_ExplicitEuler(self):
        self._test_logging_different_solver("ExplicitEuler")

    @testattr(stddist = True)
    def test_logging_option_LSODAR(self):
        self._test_logging_different_solver("LSODAR")
    @testattr(stddist = True)
    def test_extract_boolean_value(self):
        log = parse_xml_log(os.path.join(logs, "boolean_log.xml"))
        eis = log.find("EventInfo")
        for ei in eis:
            assert isinstance(ei.time_event_info, bool), "Expected ei.time_event_info to be bool"
