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
from tempfile import TemporaryDirectory

import nose

from pyfmi import testattr
from pyfmi.common.log import extract_xml_log, parse_xml_log
from pyfmi.tests.test_util import Dummy_FMUModelME2, get_examples_folder
from pyfmi.fmi import load_fmu, FMUException

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
        """ Verify exception is raised when file does not exist when invoking extract_xml_log. """
        filename = 'CoupledClutches_log_.txt'
        msg = "Unable to extract XML log. Cannot extract log from '{}' since it does not exist.".format(filename)
        with nose.tools.assert_raises_regex(FMUException, msg):
            extract_xml_log("Tmp2", os.path.join(logs, filename), modulename = 'Model')

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

class TestNoLogFile:
    """
        Test invoking functions on the FMU if we have no logfile.
        We do this from temporary directories to make sure that we have empty directories for the test.
    """
    @classmethod
    def setup_class(cls):
        cls.fmu_path = os.path.join(get_examples_folder(), 'files', 'FMUs', 'ME2.0', 'bouncingBall.fmu')

    @testattr(stddist = True)
    def test_no_logfile(self):
        """ Verify no logfile generated if log_level is 0. """
        with TemporaryDirectory() as tempdir:
            log_file_name = os.path.join(tempdir, 'test.txt')
            fmu = load_fmu(self.fmu_path, log_file_name = log_file_name, log_level = 0)
            res = fmu.simulate()
            err_msg = "Test failed because logfile {} was generated even though it should not".format(log_file_name)
            nose.tools.assert_false(os.path.isfile(log_file_name), err_msg)

    @testattr(stddist = True)
    def test_get_log_empty(self):
        """ Verify that get_log returns an empty list if log_level is 0. """
        with TemporaryDirectory() as tempdir:
            log_file_name = os.path.join(tempdir, 'test.txt')
            fmu = load_fmu(self.fmu_path, log_file_name = log_file_name, log_level = 0)
            res = fmu.simulate()
            nose.tools.assert_equal(fmu.get_log(), [])

    @testattr(stddist = True)
    def test_extract_xml_log_no_logfile(self):
        """ Verify that fmu.extract_xml_log() returns None if no logfile exists.
            Note that it is expected that the behavior is different from extract_xml_log which throws exception.
        """
        with TemporaryDirectory() as tempdir:
            log_file_name = os.path.join(tempdir, 'test.txt')
            fmu = load_fmu(self.fmu_path, log_file_name = log_file_name, log_level = 0)
            res = fmu.simulate()
            nose.tools.assert_is_none(fmu.extract_xml_log())

    @testattr(stddist = True)
    def test_get_number_of_lines_log_no_logfile(self):
        """ Verify that get_number_of_lines_log returns 0 if log_level is 0. """
        with TemporaryDirectory() as tempdir:
            log_file_name = os.path.join(tempdir, 'test.txt')
            fmu = load_fmu(self.fmu_path, log_file_name = log_file_name, log_level = 0)
            res = fmu.simulate()
            nlines = fmu.get_number_of_lines_log()
            err_msg = "Number of lines in log is not 0, it is {}".format(nlines)
            nose.tools.assert_equal(nlines, 0, err_msg)

    @testattr(stddist = True)
    def test_no_logfile_after_load_fmu(self):
        """ Verify no logfile created when loading of FMU is done since we didnt write any log message. """
        with TemporaryDirectory() as tempdir:
            log_file_name = os.path.join(tempdir, 'test.txt')
            fmu = load_fmu(self.fmu_path, log_file_name = log_file_name, log_level = 0)
            err_msg = "Test failed because logfile {} was generated even though it should not".format(log_file_name)
            nose.tools.assert_false(os.path.isfile(log_file_name), err_msg)