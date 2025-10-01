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

# This file covers tests for getters on default experiment in the modelDescription.xml

import pytest
from pathlib import Path

from pyfmi.fmi1 import FMUModelME1
from pyfmi.fmi2 import FMUModelME2
from pyfmi.fmi3 import FMUModelME3

class Test_DefaultExperiment:
    """Tests on default experiment of the FMU, as defined in the modelDescription.xml."""
    @classmethod
    def setup_class(cls):
        # load once
        this_dir = Path(__file__).parent.absolute()
        xml_dir = this_dir / "files" / "FMUs" / "XML"
        cls.fmus = {
            1: FMUModelME1(str(xml_dir / "ME1.0" / "modelDescriptionAttributes"), 
                           allow_unzipped_fmu = True, _connect_dll = False),
            2: FMUModelME2(str(xml_dir / "ME2.0" / "modelDescriptionAttributes"), 
                           allow_unzipped_fmu = True, _connect_dll = False),
            3: FMUModelME3(str(xml_dir / "ME3.0" / "modelDescriptionAttributes"), 
                           allow_unzipped_fmu = True, _connect_dll = False),
        }

    @pytest.mark.parametrize("fmi_version", [1, 2, 3])
    def test_get_default_experiment_start_time(self, fmi_version):
        """Test get_default_experiment_start_time function."""
        fmu = self.fmus[fmi_version]
        assert fmu.get_default_experiment_start_time() == 1.23

    @pytest.mark.parametrize("fmi_version", [1, 2, 3])
    def test_get_default_experiment_stop_time(self, fmi_version):
        """Test get_default_experiment_stop_time function."""
        fmu = self.fmus[fmi_version]
        assert fmu.get_default_experiment_stop_time() == 4.56

    @pytest.mark.parametrize("fmi_version", [1, 2, 3])
    def test_get_default_experiment_tolerance(self, fmi_version):
        """Test get_default_experiment_start_time function."""
        fmu = self.fmus[fmi_version]
        assert fmu.get_default_experiment_tolerance() == 1e-6

    @pytest.mark.parametrize("fmi_version", [2, 3]) # FMI2+
    def test_get_default_experiment_step(self, fmi_version):
        """Test get_default_experiment_step function."""
        fmu = self.fmus[fmi_version]
        assert fmu.get_default_experiment_step() == 2e-3
