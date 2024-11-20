#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2021 Modelon AB
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
from io import StringIO
import tempfile
from shutil import rmtree
from filecmp import cmp as compare_files

from pyfmi.fmi import FMUException, load_fmu, FMUModelCS2, FMUModelME2
from pyfmi.test_util import get_examples_folder

file_path = os.path.dirname(os.path.abspath(__file__))

class TestIO(StringIO):
    """ Test class used to verify that a custom class can be used as a logger
        if it inherits specific properties.
    """
    def __init__(self, arg):
        StringIO.__init__(self, arg)


def write_stream_to_file(stream, output_file):
    """ Writes contents of 'stream' to 'output_file'.
        The stream needs to support the two functions 'seek' and 'readlines'.
    """
    stream.seek(0)
    with open(output_file, 'w') as f:
        f.writelines(stream.readlines())

def simulate_and_verify_stream_contents(compiled_fmu, fmu_loader, stream, open_to_read = False):
    """
        Loads a compiled fmu with specified class 'fmu_loader', logs into stream and
        check the contents of the stream and verifies it with a reference result.

        The boolean parameter open_to_read is set in order to reopen the stream in
        mode 'r' in order to read from it, since it is assumed the specified stream
        is given in mode 'w'. This is solely used for testing to reduce duplicated code.

    """
    fmu = fmu_loader(compiled_fmu, log_file_name = stream, log_level = 3)
    results = fmu.simulate()

    contents = []
    if open_to_read:
        stream.close()
        with open(stream.name, 'r') as f:
            contents = f.readlines()
    else:
        stream.seek(0)
        contents = stream.readlines()

    # Is enough to check substrings
    expected = [
        'FMIL: module = FMI2XML, log level = 3: fmi2_xml_get_default_experiment_start:',
        'FMIL: module = FMI2XML, log level = 3: fmi2_xml_get_default_experiment_stop:',
        'FMIL: module = FMI2XML, log level = 3: fmi2_xml_get_default_experiment_tolerance:'
    ]
    for i, line in enumerate(expected):
        err_msg = "Unable to find substring {} in list {}".format(line, "".join(contents))
        assert line in contents[i], err_msg

class Test_FMUModelME2:
    """ Test stream functionality for FMI class FMUModelME2. """
    @pytest.fixture(autouse=True)
    @classmethod
    def setup_class(cls):
        cls.example_fmu = os.path.join(get_examples_folder(), 'files', 'FMUs', 'ME2.0', 'bouncingBall.fmu')
        cls.test_class = FMUModelME2

        # Verify the installation is not corrupt while setting up the class.
        assert os.path.isfile(cls.example_fmu)

    def test_testio(self):
        """ FMUModelME2 and custom IO class. """
        stream = TestIO("")
        simulate_and_verify_stream_contents(self.example_fmu, self.test_class, stream)

    def test_stringio(self):
        """ FMUModelME2 and StringIO. """
        stream = StringIO()
        simulate_and_verify_stream_contents(self.example_fmu, self.test_class, stream)

    def test_textiowrapper(self):
        """ FMUModelME2 and TextIOWrapper. """
        p = tempfile.mkdtemp()
        output_file = os.path.join(p, 'test.txt')
        stream = open(output_file, 'w')
        simulate_and_verify_stream_contents(self.example_fmu, self.test_class, stream, True)
        if not stream.closed:
            stream.close()
        rmtree(p)

class Test_FMUModelCS2:
    """ Test stream functionality for FMI class FMUModelCS2. """
    @pytest.fixture(autouse=True)
    @classmethod
    def setup_class(cls):
        cls.example_fmu = os.path.join(get_examples_folder(), 'files', 'FMUs', 'CS2.0', 'bouncingBall.fmu')
        cls.test_class = FMUModelCS2

        # Verify the installation is not corrupt while setting up the class.
        assert os.path.isfile(cls.example_fmu)

    def test_testio(self):
        """ FMUModelCS2 and custom IO class. """
        stream = TestIO("")
        simulate_and_verify_stream_contents(self.example_fmu, self.test_class, stream)

    def test_stringio(self):
        """ FMUModelCS2 and StringIO. """
        stream = StringIO()
        simulate_and_verify_stream_contents(self.example_fmu, self.test_class, stream)

    def test_textiowrapper(self):
        """ FMUModelCS2 and TextIOWrapper. """
        p = tempfile.mkdtemp()
        output_file = os.path.join(p, 'test.txt')
        stream = open(output_file, 'w')
        simulate_and_verify_stream_contents(self.example_fmu, self.test_class, stream, True)
        if not stream.closed:
            stream.close()
        rmtree(p)

class Test_LoadFMU:
    """ Test stream functionality with load_fmu. """
    @pytest.fixture(autouse=True)
    @classmethod
    def setup_class(cls):
        cls.example_fmu = os.path.join(get_examples_folder(), 'files', 'FMUs', 'ME2.0', 'bouncingBall.fmu')
        cls.test_class = load_fmu

        # Verify the installation is not corrupt while setting up the class.
        assert os.path.isfile(cls.example_fmu)

    def test_testio(self):
        """ load_fmu and custom IO class. """
        stream = TestIO("")
        simulate_and_verify_stream_contents(Test_LoadFMU.example_fmu, Test_LoadFMU.test_class, stream)

    def test_stringio(self):
        """ load_fmu and StringIO. """
        stream = StringIO()
        simulate_and_verify_stream_contents(Test_LoadFMU.example_fmu, Test_LoadFMU.test_class, stream)

    def test_textiowrapper(self):
        """ load_fmu and TextIOWrapper. """
        p = tempfile.mkdtemp()
        output_file = os.path.join(p, 'test.txt')
        stream = open(output_file, 'w')
        simulate_and_verify_stream_contents(Test_LoadFMU.example_fmu, Test_LoadFMU.test_class, stream, True)
        if not stream.closed:
            stream.close()
        rmtree(p)

class TestXML:
    """ Test other log related functions together with streams. """
    @pytest.fixture(autouse=True)
    @classmethod
    def setup_class(cls):
        cls.example_fmu = os.path.join(get_examples_folder(), 'files', 'FMUs', 'ME2.0', 'bouncingBall.fmu')

        # Verify the installation is not corrupt while setting up the class.
        assert os.path.isfile(cls.example_fmu)

    def test_extract_xml_log(self):
        """ Compare contents of XML log when using stream and normal logfile. """
        stream = TestIO("")
        fmu_s = load_fmu(self.example_fmu, log_file_name = stream, log_level = 4)
        xml_file1 = fmu_s.get_log_filename() +'.xml'
        if os.path.isfile(xml_file1):
            os.remove(xml_file1)
        res_s = fmu_s.simulate()
        xml_log_s = fmu_s.extract_xml_log()

        log_file_name = 'test_cmp_xml_files.txt'
        if os.path.isfile(log_file_name):
            os.remove(log_file_name)
        fmu = load_fmu(self.example_fmu, log_file_name = log_file_name, log_level = 4)
        xml_file2 = 'test_cmp_xml_files.xml'
        if os.path.isfile(xml_file2):
            os.remove(xml_file2)
        res = fmu.simulate()
        xml_log = fmu.extract_xml_log()

        err_msg = "Unequal xml files, please compare the contents of:\n{}\nand\n{}".format(xml_log_s, xml_log)
        assert compare_files(xml_log_s, xml_log), err_msg

    def test_get_log(self):
        """ Test get_log throws exception if stream doesnt support getvalue. """
        stream = StringIO("")
        fmu_s = load_fmu(self.example_fmu, log_file_name = stream, log_level = 3)
        res_s = fmu_s.simulate()
        log = fmu_s.get_log()
        expected_substr = [
            'FMIL: module = FMI2XML, log level = 3: fmi2_xml_get_default_experiment_start',
            'FMIL: module = FMI2XML, log level = 3: fmi2_xml_get_default_experiment_stop',
            'FMIL: module = FMI2XML, log level = 3: fmi2_xml_get_default_experiment_tolerance'
        ]
        for i, line in enumerate(expected_substr):
            assert line in log[i]


    def test_get_log_exception1(self):
        """ Test get_log throws exception if stream doesnt allow reading (it is set for writing). """
        try:
            p = tempfile.mkdtemp()
            output_file = os.path.join(p, 'test.txt')
            stream = open(output_file, 'w')
            fmu_s = load_fmu(self.example_fmu, log_file_name = stream, log_level = 3)
            res_s = fmu_s.simulate()
            err_msg = "Unable to read from given stream, make sure the stream is readable."
            with pytest.raises(FMUException, match = err_msg):
                log = fmu_s.get_log()
        finally:
            if not stream.closed:
                stream.close()
            rmtree(p)


    def test_get_nbr_of_lines_in_log(self):
        """ Test get_number_of_lines_log when using a stream. """
        stream = StringIO("")
        fmu = load_fmu(self.example_fmu, log_file_name = stream, log_level = 3)
        assert fmu.get_number_of_lines_log() == 0
        res = fmu.simulate()
        assert fmu.get_number_of_lines_log() == 0

    def test_extract_xml_log_into_stream(self):
        """ Compare contents of XML log when extract XML into a stream. """
        stream = TestIO("")
        extracted_xml_stream = StringIO("")
        fmu_s = load_fmu(self.example_fmu, log_file_name = stream, log_level = 4)
        res_s = fmu_s.simulate()
        fmu_s.extract_xml_log(extracted_xml_stream)

        # write the contents of extract_xml_stream to a file for test
        xml_file1 = "my_new_file.xml"
        if os.path.isfile(xml_file1):
            os.remove(xml_file1)
        write_stream_to_file(extracted_xml_stream, xml_file1)

        log_file_name = 'test_cmp_xml_files.txt'
        if os.path.isfile(log_file_name):
            os.remove(log_file_name)
        fmu = load_fmu(self.example_fmu, log_file_name = log_file_name, log_level = 4)
        xml_file2 = 'test_cmp_xml_files.xml'
        if os.path.isfile(xml_file2):
            os.remove(xml_file2)
        res = fmu.simulate()
        xml_log = fmu.extract_xml_log()

        err_msg = "Unequal xml files, please compare the contents of:\n{}\nand\n{}".format(xml_file1, xml_log)
        assert compare_files(xml_file1, xml_log), err_msg
