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

# Testing any functionality currently marked as deprecated
# Also shortlist for any candidates to be removed

import pytest
from pathlib import Path

from pyfmi import load_fmu
from pyfmi.common.io import ResultStorage

this_dir = Path(__file__).parent
FMI2_REF_FMU_PATH = Path(this_dir) / 'files' / 'reference_fmus' / '2.0'

def test_result_storage():
    """Test deprecation of pyfmi.common.io.ResultStorage."""
    msg = "Use pyfmi.common.io.ResultReader as base class instead."
    with pytest.warns(DeprecationWarning, match = msg):
        ResultStorage()

@pytest.mark.parametrize("result_handling", 
    [
        "file",
        "binary",
        "memory",
    ])
def test_result_handler_name_attribute(result_handling):
    """Test deprecation of `name` attribute to get variable names for result handlers."""
    fmu = load_fmu(FMI2_REF_FMU_PATH / "Dahlquist.fmu")
    res = fmu.simulate(options = {"ncp": 0, "result_handling": result_handling})

    msg = "Use the `get_variable_names` function instead."
    with pytest.warns(DeprecationWarning, match = msg):
        res.result_data.name
