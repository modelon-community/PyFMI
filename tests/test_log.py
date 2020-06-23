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

from pyfmi import testattr, load_fmu
from pyfmi.common.log import extract_xml_log, parse_xml_log

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

    @testattr(stddist = True)
    def test_logfile_contains_specific_attributes(self):
        path_to_fmu = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "TestModel.fmu")
        log_level = 6
        fmu = load_fmu('TestModel.fmu', log_level = log_level)
        fmu.set('_log_level', log_level+1)
        fmu.initialize()
        opts = fmu.simulate_options()
        opts["initialize"] = False
        opts["logging"] = True
        opts["solver"] = 'CVode'

        res = fmu.simulate(options=opts)

        with open('TestModel_log.txt', 'r') as f:
            log_contents = f.readlines()

        found_state_names = False
        found_solver_order = False
        found_rtol = False
        found_atol = False
        found_state_errors = False
        found_solver_name = False

        # loop through the log once and mark each found entry
        for row in log_contents:
            if 'state_names' in row:
                found_state_names = True
            if 'solver_order' in row:
                found_solver_order = True
            if 'relative_tolerance' in row:
                found_rtol = True
            if 'absolute_tolerance' in row:
                found_atol = True
            if 'state_error' in row:
                found_state_errors = True
            if 'solver_name' in row:
                found_solver_name = True

        err_msg = "Could not find all expected entries in logfile"
        assert found_state_names and found_solver_order and found_rtol and found_atol and found_state_errors and found_solver_name, err_msg
