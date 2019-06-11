#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Modelon AB
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

import nose
import os


from pyfmi import testattr
from pyfmi.fmi import FMUModel, FMUException, FMUModelME1, FMUModelCS1, load_fmu, FMUModelCS2, FMUModelME2, PyEventInfo
import pyfmi.fmi_util as fmi_util
import pyfmi.fmi as fmi

file_path = os.path.dirname(os.path.abspath(__file__))

class Test_load_fmu_only_XML:
    
    @testattr(stddist = True)
    def test_loading_xml_me1(self):
        
        model = FMUModelME1("CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        
        assert model.get_name() == "CoupledClutches"

    @testattr(stddist = True)
    def test_loading_xml_cs1(self):
        
        model = FMUModelCS1("CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS1.0"), _connect_dll=False)
        
        assert model.get_name() == "CoupledClutches"
        
    @testattr(stddist = True)
    def test_loading_xml_me2(self):
        
        model = FMUModelME2("CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        assert model.get_name() == "CoupledClutches"
        
    @testattr(stddist = True)
    def test_loading_xml_cs2(self):
        
        model = FMUModelCS2("CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS2.0"), _connect_dll=False)
        
        assert model.get_name() == "CoupledClutches"
