#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2010 Modelon AB
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

"""
Module containing the tests for the FMI interface.
"""

import nose
import os
import scipy.sparse.csc
from collections import OrderedDict

from pyfmi import testattr
from pyfmi.fmi import FMUModel, FMUException, FMUModelME1, FMUModelCS1, load_fmu, FMUModelCS2, FMUModelME2, PyEventInfo
import pyfmi.fmi_util as fmi_util
import pyfmi.fmi as fmi

class Test_FMIUtil:
    
    @testattr(stddist = True)
    def test_cpr_seed(self):
        structure = OrderedDict([('der(inertia3.phi)', ['inertia3.w']),
             ('der(inertia3.w)', ['damper.phi_rel', 'inertia3.phi']),
             ('der(damper.phi_rel)', ['damper.w_rel']),
             ('der(damper.w_rel)',
              ['damper.phi_rel', 'damper.w_rel', 'inertia3.phi'])])
        
        states = ['inertia3.phi', 'inertia3.w', 'damper.phi_rel', 'damper.w_rel']
        
        groups = fmi_util.cpr_seed(structure, states)
        
        assert groups[0][5] == [1,2,3]
        assert groups[1][5] == [5,7]
        assert groups[2][5] == [8,9]
        assert groups[0][4] == [0,1,2]
        assert groups[1][4] == [3,4]
        assert groups[2][4] == [5,6]
