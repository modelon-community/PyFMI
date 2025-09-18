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
import re
from pathlib import Path

from pyfmi import load_fmu
from pyfmi.common.io import ResultStorage
from pyfmi.master import Master
from pyfmi.fmi2 import FMUModelCS2

this_dir = Path(__file__).parent.absolute()
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

def test_master_alg_downsample_result_simple_value():
    """Test setting a simple value to the 'result_downsampling_factor' option for the Master algorithm."""
    fmu1 = FMUModelCS2(FMI2_REF_FMU_PATH / "Feedthrough.fmu")
    fmu2 = FMUModelCS2(FMI2_REF_FMU_PATH / "Feedthrough.fmu")

    models = [fmu1, fmu2]
    connections = [(fmu1, "Float64_continuous_output", fmu2, "Float64_continuous_input")]
    master = Master(models, connections)
    msg = "Use of simple value inputs for 'result_downsampling_factor' is deprecated, use a dictionary with models as keys instead."
    with pytest.warns(DeprecationWarning, match = re.escape(msg)):
        master.simulate(0, 1, options = {"step_size": 0.5, "result_downsampling_factor": 2})
