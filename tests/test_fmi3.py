#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2025 Modelon AB
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
import re
import logging
import numpy as np
from io import StringIO
from pyfmi import load_fmu
from pathlib import Path
from pyfmi.fmi import (
    FMUModelME3,
)
from pyfmi.fmi3 import (
    FMI3_FLOAT64,
    FMI3_FLOAT32,
    FMI3_INT64,
    FMI3_INT32,
    FMI3_INT16,
    FMI3_INT8,
    FMI3_UINT64,
    FMI3_UINT32,
    FMI3_UINT16,
    FMI3_UINT8,
    FMI3_BOOL,
    FMI3_BINARY,
    FMI3_CLOCK,
    FMI3_STRING,
    FMI3_ENUM,
)
from pyfmi.exceptions import (
    FMUException,
    InvalidFMUException,
    InvalidVersionException
)

# TODO: A lot of the tests here could be parameterized with the tests in test_fmi.py
# This would however require on of the following:
# a) Changing the tests in test_fmi.py to use the FMI1/2 reference FMUs
# b) Mocking the FMUs in some capacity

this_dir = Path(__file__).parent.absolute()
FMI3_REF_FMU_PATH = Path(this_dir) / 'files' / 'reference_fmus' / '3.0'


import contextlib
from pathlib import Path

@contextlib.contextmanager
def temp_dir_context(tmpdir):
    """Provides a temporary directory as a context."""
    yield Path(tmpdir)

class TestFMI3LoadFMU:
    """Basic unit tests for FMI3 loading via 'load_fmu'."""

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Dahlquist.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "StateSpace.fmu",
        FMI3_REF_FMU_PATH / "Feedthrough.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_load_kind_auto(self, caplog, ref_fmu):
        """Test loading a ME FMU via kind 'auto'"""
        caplog.set_level(logging.WARNING)
        fmu = load_fmu(ref_fmu, kind = "auto")
        assert isinstance(fmu, FMUModelME3)
        experimental_msg = "FMI3 support is experimental."
        assert any(experimental_msg in msg for msg in caplog.messages)

    @pytest.mark.parametrize("ref_fmu", [FMI3_REF_FMU_PATH / "Clocks.fmu"])
    def test_load_kind_auto_SE(self, ref_fmu):
        """Test loading a SE only FMU via kind 'auto'"""
        msg = "Import of FMI3 Scheduled Execution FMUs is not supported"
        with pytest.raises(InvalidFMUException, match = re.escape(msg)):
            load_fmu(ref_fmu, kind = "auto")

    @pytest.mark.parametrize("ref_fmu", [FMI3_REF_FMU_PATH / "VanDerPol.fmu"])
    def test_load_kind_ME(self, ref_fmu):
        """Test loading an FMU with kind 'ME'"""
        fmu = load_fmu(ref_fmu, kind = "ME")
        assert isinstance(fmu, FMUModelME3)

    @pytest.mark.parametrize("ref_fmu", [FMI3_REF_FMU_PATH / "VanDerPol.fmu"])
    def test_load_kind_CS(self, ref_fmu):
        """Test loading an FMU with kind 'CS'"""
        msg = "Import of FMI3 Co-Simulation FMUs is not yet supported."
        with pytest.raises(InvalidFMUException, match = msg):
            load_fmu(ref_fmu, kind = "CS")

    @pytest.mark.parametrize("ref_fmu", [FMI3_REF_FMU_PATH / "Clocks.fmu"])
    def test_load_kind_SE(self, ref_fmu):
        """Test loading an FMU with kind 'SE'"""
        msg = "Import of FMI3 Scheduled Execution FMUs is not supported."
        with pytest.raises(FMUException, match = re.escape(msg)):
            load_fmu(ref_fmu, kind = "SE")

    def test_get_model_identifier(self):
        """Test that model identifier is retrieved as expected."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        assert fmu.get_identifier() == 'VanDerPol'

    def test_get_get_version(self):
        """Test that FMI version is retrieved as expected."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        assert fmu.get_version() == '3.0'

    def test_instantiation(self, tmpdir):
        """ Test that instantiation works by verifying the output in the log."""
        with temp_dir_context(tmpdir) as temp_path:
             # log_level set to 5 required by test
            fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu", log_level=5)

        substring_to_find = 'Successfully loaded all the interface functions'
        assert any(substring_to_find in line for line in fmu.get_log())

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Dahlquist.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "StateSpace.fmu",
        FMI3_REF_FMU_PATH / "Feedthrough.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_initialize_reset_terminate(self, ref_fmu):
        """Test initialize, reset and terminate of all the ME reference FMUs. """
        fmu = load_fmu(ref_fmu)
        # Should simply pass without any exceptions
        fmu.initialize()
        fmu.reset()

        # Test initialize again after resetting followed by terminate,
        # since terminating does not require reset.
        fmu.initialize()
        fmu.terminate()

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Dahlquist.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "StateSpace.fmu",
        FMI3_REF_FMU_PATH / "Feedthrough.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_enter_continuous_time_mode(self, ref_fmu):
        """Test entering continuous time mode. """
        fmu = load_fmu(ref_fmu)
        # Should simply pass without any exceptions
        fmu.initialize()
        fmu.enter_continuous_time_mode()
        fmu.terminate()

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Dahlquist.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "StateSpace.fmu",
        FMI3_REF_FMU_PATH / "Feedthrough.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_enter_event_mode(self, ref_fmu):
        """Test enter event mode. """
        fmu = load_fmu(ref_fmu)
        # Should simply pass without any exceptions
        fmu.initialize()
        fmu.enter_continuous_time_mode()
        fmu.enter_event_mode()
        fmu.terminate()

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Dahlquist.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "StateSpace.fmu",
        FMI3_REF_FMU_PATH / "Feedthrough.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_initialize_manually(self, ref_fmu):
        """Test initialize all the ME reference FMUs by entering/exiting initialization mode manually. """
        fmu = load_fmu(ref_fmu)
        # Should simply pass without any exceptions
        fmu.enter_initialization_mode()
        fmu.exit_initialization_mode()

    def test_get_double_terminate(self):
        """Test invalid call sequence raises an error. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        fmu.initialize()
        fmu.terminate()
        msg = "Termination of FMU failed, see log for possible more information."
        with pytest.raises(FMUException, match = msg):
            fmu.terminate()

    def test_get_default_experiment_start_time(self):
        """Test retrieving default experiment start time. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        assert fmu.get_default_experiment_start_time() == 0.0

    def test_free_instance_after_load(self):
        """Test invoke free instance after loading. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        fmu.free_instance()

    def test_free_instance_after_initialization(self):
        """Test invoke free instance after initialization. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        fmu.initialize()
        fmu.free_instance()

    def test_get_default_experiment_stop_time(self):
        """Test retrieving default experiment stop time. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        assert fmu.get_default_experiment_stop_time() == 20.0


    def test_get_default_experiment_tolerance(self):
        """Test retrieving default experiment tolerance. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        assert fmu.get_default_experiment_tolerance() == 0.0001

class Test_FMI3ME:
    """Basic unit tests for FMI3 import directly via the FMUModelME3 class."""
    @pytest.mark.parametrize("ref_fmu", [FMI3_REF_FMU_PATH / "VanDerPol.fmu"])
    def test_basic(self, ref_fmu):
        """Basic construction of FMUModelME3."""
        fmu = FMUModelME3(ref_fmu, _connect_dll = False)
        assert isinstance(fmu, FMUModelME3)

    @pytest.mark.parametrize("ref_fmu", [FMI3_REF_FMU_PATH / "Clocks.fmu"])
    def test_basic_wrong_fmu_type(self, ref_fmu):
        """Test using a non-ME FMU."""
        msg = "The FMU could not be loaded. This class only supports FMI 3.0 for Model Exchange."
        with pytest.raises(InvalidVersionException, match = msg):
            FMUModelME3(ref_fmu, _connect_dll = False)

    def test_logfile_content(self):
        """Test that we get the log content from FMIL parsing the modelDescription.xml."""
        log_filename = "test_fmi3_log.txt"
        FMUModelME3(FMI3_REF_FMU_PATH / "VanDerPol.fmu", log_file_name = log_filename,
                    _connect_dll = False, log_level = 5)

        with open(log_filename, "r") as file:
            data = file.read()

        assert "FMIL: module = FMILIB, log level = 4: XML specifies FMI standard version 3.0" in data
        assert "FMIL: module = FMILIB, log level = 5: Parsing finished successfully" in data

    # TODO: FUTURE: Move to test_stream.py
    def test_logging_stream(self):
        """Test logging content from FMIL using a stream."""
        log_filename = StringIO("")
        fmu = FMUModelME3(FMI3_REF_FMU_PATH / "VanDerPol.fmu", log_file_name = log_filename,
                          _connect_dll = False, log_level = 5)
        log = fmu.get_log()

        assert "FMIL: module = FMILIB, log level = 4: XML specifies FMI standard version 3.0" in log
        assert "FMIL: module = FMILIB, log level = 5: Parsing finished successfully" in log

    @pytest.mark.parametrize("log_level", [1, 2, 3, 4, 5, 6, 7])
    def test_valid_log_levels(self, log_level):
        """Test valid log levels."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = FMUModelME3(fmu_path, log_level = log_level, _connect_dll = False)
        assert log_level == fmu.get_fmil_log_level()

    def test_valid_log_level_off(self):
        """Test logging nothing."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = FMUModelME3(fmu_path, log_level = 0, _connect_dll = False)
        msg = "Logging is not enabled"
        with pytest.raises(FMUException, match = msg):
            fmu.get_fmil_log_level()

    @pytest.mark.parametrize("log_level", [-1, 8, 1.0, "DEBUG"])
    def test_invalid_log_level(self, log_level):
        """Test invalid log levels."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        msg = "The log level must be an integer between 0 and 7"
        with pytest.raises(FMUException, match = msg):
            FMUModelME3(fmu_path, log_level = log_level, _connect_dll = False)

    @pytest.mark.parametrize("variable_name, value, expected_dtype",
        [
            ("Float64_continuous_input", 3.14, np.double),
            ("Float32_continuous_input", np.float32(3.14), np.float32),
        ]
    )
    def test_set_get(self, variable_name, value, expected_dtype):
        """Test getting and setting variables of various types."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = FMUModelME3(fmu_path)
        fmu.set(variable_name, value)
        res = fmu.get(variable_name)
        assert res.dtype == expected_dtype
        assert res[0] == value

    def test_set_missing_variable(self):
        """Test setting a variable that does not exists."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = FMUModelME3(fmu_path, _connect_dll = False)
        var_name = "x0"
        err_msg = f"The variable {var_name} could not be found."
        with pytest.raises(FMUException, match = err_msg):
            fmu.set(var_name, 0.)

    def test_get_missing_variable(self):
        """Test getting a variable that does not exists."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = FMUModelME3(fmu_path, _connect_dll = False)
        var_name = "x0"
        err_msg = f"The variable {var_name} could not be found."
        with pytest.raises(FMUException, match = err_msg):
            fmu.get(var_name)

    def test_get_variable_valueref(self):
        """Test getting variable value references."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = FMUModelME3(fmu_path, _connect_dll = False)
        assert fmu.get_variable_valueref("time") == 0
        assert fmu.get_variable_valueref("Enumeration_input") == 32

    def test_get_variable_valueref_missing(self):
        """Test getting variable value references for variable that does not exist."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = FMUModelME3(fmu_path, _connect_dll = False)
        var_name = "x0"
        err_msg = f"The variable {var_name} could not be found."
        with pytest.raises(FMUException, match = err_msg):
            fmu.get_variable_valueref(var_name)

    @pytest.mark.parametrize("variable_name, expected_datatype",
        [
            ("Float64_continuous_input", FMI3_FLOAT64),
            ("Float32_continuous_input", FMI3_FLOAT32),
            ("Int64_input", FMI3_INT64),
            ("Int32_input", FMI3_INT32),
            ("Int16_input", FMI3_INT16),
            ("Int8_input" , FMI3_INT8),
            ("UInt64_input", FMI3_UINT64),
            ("UInt32_input", FMI3_UINT32),
            ("UInt16_input", FMI3_UINT16),
            ("UInt8_input",  FMI3_UINT8),
            ("Boolean_input", FMI3_BOOL),
            ("String_parameter", FMI3_STRING),
            ("Binary_input", FMI3_BINARY),
            ("Enumeration_input", FMI3_ENUM),
        ]
    )
    def test_get_variable_data_type(self, variable_name, expected_datatype):
        """Test getting variable data types."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = FMUModelME3(fmu_path, _connect_dll = False)
        assert fmu.get_variable_data_type(variable_name) == expected_datatype

    def test_get_variable_data_type_missing(self):
        """Test getting variable data type for missing variable."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = FMUModelME3(fmu_path, _connect_dll = False)
        var_name = "x0"
        err_msg = f"The variable {var_name} could not be found."
        with pytest.raises(FMUException, match = err_msg):
            fmu.get_variable_data_type(var_name)

    def test_set_array_variable(self):
        """Test setting an array variable (not yet supported). """
        fmu_path = FMI3_REF_FMU_PATH / "StateSpace.fmu"
        fmu = FMUModelME3(fmu_path)
        err_msg = "The length of valueref and values are inconsistent. Note: Array variables are not yet supported"
        with pytest.raises(FMUException, match = err_msg):
            fmu.set("x", np.array([1, 2, 3]))


class TestFMI3CS:
    # TODO: Unsupported for now
    pass

class TestFMI3SE:
    # TODO: Unsupported for now
    pass
