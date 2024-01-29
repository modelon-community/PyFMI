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

import numpy as np
from collections import OrderedDict

from pyfmi import testattr
import pyfmi.fmi_util as fmi_util

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
        
        assert np.array(groups[0][5] == [1,2,3]).all()
        assert np.array(groups[1][5] == [5,7]).all()
        assert np.array(groups[2][5] == [8,9]).all()
        assert np.array(groups[0][4] == [0,1,2]).all()
        assert np.array(groups[1][4] == [3,4]).all()
        assert np.array(groups[2][4] == [5,6]).all()
    
    @testattr(stddist = True)
    def test_cpr_seed_interested_columns(self):
        structure = OrderedDict([('der(inertia3.phi)', ['inertia3.w']),
             ('der(inertia3.w)', ['damper.phi_rel', 'inertia3.phi']),
             ('der(damper.phi_rel)', ['damper.w_rel']),
             ('der(damper.w_rel)',
              ['damper.phi_rel', 'damper.w_rel', 'inertia3.phi'])])
        
        states = ['inertia3.phi', 'inertia3.w', 'damper.phi_rel', 'damper.w_rel']
        
        interested_columns = {0:1, 1:0, 2:0}
        groups = fmi_util.cpr_seed(structure, states, interested_columns)
        
        assert np.array(groups[0][5] == [1,2,3]).all()
        assert np.array(groups[1][5] == [5,7]).all()
        assert np.array(groups[0][4] == [0,1,2]).all()
        assert np.array(groups[1][4] == [3,4]).all()
        assert len(groups) == 5
        
        interested_columns = {0:1, 1:0, 3:0}
        groups = fmi_util.cpr_seed(structure, states, interested_columns)
        
        assert np.array(groups[0][5] == [1,2,3]).all()
        assert np.array(groups[1][5] == [8,9]).all()
        assert np.array(groups[0][4] == [0,1,2]).all()
        assert np.array(groups[1][4] == [5,6]).all()
        assert len(groups) == 5
        
        interested_columns = {1:1, 2:0, 3:0}
        groups = fmi_util.cpr_seed(structure, states, interested_columns)
        
        assert np.array(groups[0][5] == [3,5,7]).all()
        assert np.array(groups[1][5] == [8,9]).all()
        assert np.array(groups[0][4] == [2,3,4]).all()
        assert np.array(groups[1][4] == [5,6]).all()
        assert len(groups) == 5
