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
from pyfmi import load_fmu
from pathlib import Path
from pyfmi.fmi import (
    FMUModelME3,
)
from pyfmi.exceptions import (
    FMUException,
    InvalidFMUException,
    InvalidVersionException
)

this_dir = Path(__file__).parent.absolute()
REFERENCE_FMU_PATH = Path(this_dir) / 'files' / 'reference_fmus' / '3.0'

REFERENCE_FMU_NAMES = [
    "BouncingBall.fmu",
    "Dahlquist.fmu",
    "Resource.fmu",
    "StateSpace.fmu",
    "Clocks.fmu",
    "Feedthrough.fmu",
    "Stair.fmu",
    "VanDerPol.fmu",
]
REFERENCE_FMUS = [str(REFERENCE_FMU_PATH / fmu_name) 
                    for fmu_name in REFERENCE_FMU_NAMES]

def test_foo():
    expected_fmu = REFERENCE_FMU_PATH / 'VanDerPol.fmu'
    assert expected_fmu.exists()

class TestFMI3LoadFMU:
    """Basic unit tests for FMI3 loading via 'load_fmu'."""
    @pytest.mark.parametrize("ref_fmu", [
        REFERENCE_FMU_PATH / "BouncingBall.fmu",
        REFERENCE_FMU_PATH / "Dahlquist.fmu",
        REFERENCE_FMU_PATH / "Resource.fmu",
        REFERENCE_FMU_PATH / "StateSpace.fmu",
        REFERENCE_FMU_PATH / "Feedthrough.fmu",
        REFERENCE_FMU_PATH / "Stair.fmu",
        REFERENCE_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_load_kind_auto(self, ref_fmu):
        """Test loading a ME FMU via kind 'auto'"""
        fmu = load_fmu(ref_fmu, kind = "auto")
        assert isinstance(fmu, FMUModelME3)

    @pytest.mark.parametrize("ref_fmu", [REFERENCE_FMU_PATH / "Clocks.fmu"])
    def test_load_kind_auto_SE(self, ref_fmu):
        """Test loading a SE only FMU via kind 'auto'"""
        msg = "Import of FMI3 ScheduledExecution FMUs is not supported"
        with pytest.raises(InvalidFMUException, match = re.escape(msg)):
            load_fmu(ref_fmu, kind = "auto")  

    @pytest.mark.parametrize("ref_fmu", [REFERENCE_FMU_PATH / "VanDerPol.fmu"])
    def test_load_kind_ME(self, ref_fmu):
        """Test loading an FMU with kind 'ME'"""
        fmu = load_fmu(ref_fmu, kind = "ME")
        assert isinstance(fmu, FMUModelME3)

    @pytest.mark.parametrize("ref_fmu", [REFERENCE_FMU_PATH / "VanDerPol.fmu"])
    def test_load_kind_CS(self, ref_fmu):
        """Test loading an FMU with kind 'CS'"""
        msg = "Import of FMI3 CoSimulation FMUs is not yet supported."
        with pytest.raises(InvalidFMUException, match = msg):
            load_fmu(ref_fmu, kind = "CS")

    @pytest.mark.parametrize("ref_fmu", [REFERENCE_FMU_PATH / "Clocks.fmu"])
    def test_load_kind_SE(self, ref_fmu):
        """Test loading an FMU with kind 'SE'"""
        msg = 'Input-argument "kind" can only be "ME", "CS" or "auto" (default) and not: SE'
        with pytest.raises(FMUException, match = re.escape(msg)):
            load_fmu(ref_fmu, kind = "SE")


class TestFMI3ME:
    """Basic unit tests for FMI3 import directly via the FMUModelME3 class."""
    @pytest.mark.parametrize("ref_fmu", [REFERENCE_FMU_PATH / "VanDerPol.fmu"])
    def test_basic(self, ref_fmu):
        """Basic construction of FMUModelME3."""
        fmu = FMUModelME3(ref_fmu, _connect_dll = False)
        assert isinstance(fmu, FMUModelME3)

    @pytest.mark.parametrize("ref_fmu", [REFERENCE_FMU_PATH / "Clocks.fmu"])
    def test_basic_wrong_fmu_type(self, ref_fmu):
        """Test using a non-ME FMU."""
        msg = "The FMU could not be loaded. This class only supports FMI 3.0 for Model Exchange."
        with pytest.raises(InvalidVersionException, match = msg):
            FMUModelME3(ref_fmu, _connect_dll = False)

    def test_incorrect_version(self):
        """Test using an FMU with the incorrect version."""
        fmu_path = this_dir / "files" / "FMUs" / "XML" / "ME2.0" / "bouncingBall.fmu"
        msg = "The FMU could not be loaded. The FMU version is not supported by this class"
        with pytest.raises(InvalidVersionException, match = msg):
            FMUModelME3(fmu_path, _connect_dll = False)

    def test_logfile_content(self):
        """Test that we get the log content from FMIL parsing the modelDescription.xml."""
        log_filename = "test_fmi3_log.txt"
        FMUModelME3(REFERENCE_FMU_PATH / "VanDerPol.fmu", log_file_name = log_filename, 
                    _connect_dll = False, log_level = 5)
        
        with open(log_filename, "r") as file:
            data = file.read()
        
        assert "FMIL: module = FMILIB, log level = 4: XML specifies FMI standard version 3.0" in data
        assert "FMIL: module = FMILIB, log level = 5: Parsing finished successfully" in data

    # TODO: Test the same with stream

    @pytest.mark.parametrize("log_level", [1, 2, 3, 4, 5, 6, 7])
    def test_valid_log_levels(self, log_level):
        """Test valid log levels."""
        fmu_path = REFERENCE_FMU_PATH / "VanDerPol.fmu"
        fmu = FMUModelME3(fmu_path, log_level = log_level, _connect_dll = False)
        assert log_level == fmu.get_fmil_log_level()

    def test_valid_log_level_off(self):
        """Test logging nothing."""
        fmu_path = REFERENCE_FMU_PATH / "VanDerPol.fmu"
        fmu = FMUModelME3(fmu_path, log_level = 0, _connect_dll = False)
        msg = "Logging is not enabled"
        with pytest.raises(FMUException, match = msg):
            fmu.get_fmil_log_level()

    @pytest.mark.parametrize("log_level", [-1, 8, 1.0, "DEBUG"])
    def test_invalid_log_level(self, log_level):
        """Test invalid log levels."""
        fmu_path = REFERENCE_FMU_PATH / "VanDerPol.fmu"
        msg = "The log level must be an integer between 0 and 7"
        with pytest.raises(FMUException, match = msg):
            FMUModelME3(fmu_path, log_level = log_level, _connect_dll = False)

    def test_invalid_path(self):
        """Test loading an FMU on a path that does not exist."""
        msg = "Could not locate the FMU in the specified directory."
        with pytest.raises(FMUException, match = msg):
            FMUModelME3("path_that_does_not_exist.fmu", _connect_dll = False)


class TestFMI3CS:
    # TODO: Unsupported for now
    pass

class TestFMI3SE:
    # TODO: Unsupported for now
    pass
