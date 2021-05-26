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
import numpy as np

from pyfmi import testattr
from pyfmi.fmi import FMUModel, FMUException, FMUModelME1, FMUModelCS1, load_fmu, FMUModelCS2, FMUModelME2, PyEventInfo
from pyfmi.fmi_coupled import CoupledFMUModelME2
from pyfmi.fmi_extended import FMUModelME1Extended
import pyfmi.fmi_util as fmi_util
import pyfmi.fmi as fmi
import pyfmi.fmi_algorithm_drivers as fmi_algorithm_drivers
from pyfmi.tests.test_util import Dummy_FMUModelME2
from pyfmi.common.io import ResultHandler

file_path = os.path.dirname(os.path.abspath(__file__))

me1_xml_path = os.path.join(file_path, "files", "FMUs", "XML", "ME1.0")

class Test_FMUModelME1Extended:
    
    @testattr(stddist = True)
    def test_log_file_name(self):
        model = FMUModelME1Extended("bouncingBall.fmu",me1_xml_path, _connect_dll=False)
        assert os.path.exists("bouncingBall_log.txt")
        model = FMUModelME1Extended("bouncingBall.fmu",me1_xml_path,log_file_name="Test_log.txt", _connect_dll=False)
        assert os.path.exists("Test_log.txt")
    
    @testattr(stddist = True)
    def test_default_experiment(self):
        model = FMUModelME1Extended("CoupledClutches.fmu", me1_xml_path, _connect_dll=False)

        assert np.abs(model.get_default_experiment_start_time()) < 1e-4
        assert np.abs(model.get_default_experiment_stop_time()-1.5) < 1e-4
        assert np.abs(model.get_default_experiment_tolerance()-0.0001) < 1e-4
