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

# This file covers tests for getters on basic model properties in the modelDescription.xml

import pytest
from pathlib import Path

from pyfmi.fmi1 import FMUModelME1
from pyfmi.fmi2 import FMUModelME2
from pyfmi.fmi3 import FMUModelME3

class Test_ModelDescriptionAttributeGetters:
    """Tests on basic attributes of the FMU, as part of the "fmiModelDescription"
    element in the modelDescription.xml."""
    # Using dummy modelDescription.xml, reference FMUs lack various attributes
    @classmethod
    def setup_class(cls):
        # load once
        this_dir = Path(__file__).parent
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
    def test_get_name(self, fmi_version):
        """Test get_name function."""
        fmu = self.fmus[fmi_version]
        assert fmu.get_name() == "myModelName"

    @pytest.mark.parametrize("fmi_version", [2, 3])
    def test_get_model_version(self, fmi_version):
        """Test get_model_version function."""
        fmu = self.fmus[fmi_version]
        assert fmu.get_model_version() == "myModelVersion"

    @pytest.mark.parametrize("fmi_version", [2, 3]) # FMI2+
    def test_get_copyright(self, fmi_version):
        """Test get_copyright function."""
        fmu = self.fmus[fmi_version]
        assert fmu.get_copyright() == "myCopyright"

    @pytest.mark.parametrize("fmi_version", [1, 2, 3])
    def test_get_author(self, fmi_version):
        """Test get_author function."""
        fmu = self.fmus[fmi_version]
        assert fmu.get_author() == "myAuthor"

    @pytest.mark.parametrize("fmi_version", [1, 2, 3])
    def test_get_description(self, fmi_version):
        """Test get_description function."""
        fmu = self.fmus[fmi_version]
        assert fmu.get_description() == "myDescription"
    
    @pytest.mark.parametrize("fmi_version", [1, 2]) # replaced by instantiationToken in FMI3
    def test_get_guid(self, fmi_version):
        """Test get_guid function."""
        fmu = self.fmus[fmi_version]
        assert fmu.get_guid() == "myGuid"

    @pytest.mark.parametrize("fmi_version", [3]) # FMI3+
    def test_get_instantiation_token(self, fmi_version):
        """Test get_instantiation_token function."""
        fmu = self.fmus[fmi_version]
        assert fmu.get_instantiation_token() == "myInstantiationToken"

    @pytest.mark.parametrize("fmi_version", [2, 3]) # FMI2+
    def test_get_license(self, fmi_version):
        """Test get_license function."""
        fmu = self.fmus[fmi_version]
        assert fmu.get_license() == "myLicense"

    @pytest.mark.parametrize("fmi_version", [1, 2, 3])
    def test_get_generation_tool(self, fmi_version):
        """Test get_generation_tool function."""
        fmu = self.fmus[fmi_version]
        assert fmu.get_generation_tool() == "myGenerationTool"

    @pytest.mark.parametrize("fmi_version", [2, 3]) # FMI2+
    def test_get_generation_date_and_time(self, fmi_version):
        """Test get_generation_date_and_time function."""
        fmu = self.fmus[fmi_version]
        assert fmu.get_generation_date_and_time() == "2025-10-01T12:34:56"

    @pytest.mark.parametrize("fmi_version", [2, 3]) # FMI2+
    def test_get_variable_naming_convention(self, fmi_version):
        """Test get_variable_naming_convention function."""
        fmu = self.fmus[fmi_version]
        assert fmu.get_variable_naming_convention() == "structured"
