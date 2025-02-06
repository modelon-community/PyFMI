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
from pyfmi import load_fmu
from pathlib import Path
from pyfmi.fmi import FMUModelME3

this_dir = Path(__file__).parent.absolute()

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
REFERENCE_FMUS = [str(Path(this_dir) / 'files' / 'reference_fmus' / '3.0' / fmu_name) 
                    for fmu_name in REFERENCE_FMU_NAMES]

def test_foo():
    expected_fmu = Path(this_dir) / 'files' / 'reference_fmus' / '3.0' / 'VanDerPol.fmu'
    assert expected_fmu.exists()

# TODO: some hard-coded test on testing invalid FMI versions?
class TestFMI3LoadFMU:
    """Basic unit tests for FMI3 loading via 'load_fmu'."""
    @pytest.mark.parametrize("ref_fmu", REFERENCE_FMUS)
    def test_load_default(self, ref_fmu):
        fmu = load_fmu(ref_fmu)
        assert isinstance(fmu, FMUModelME3)


class TestFMI3ME:
    """Basic unit tests for FMI3 import using the reference FMUs."""
    @pytest.mark.parametrize("ref_fmu", REFERENCE_FMUS)
    def test_import_default(self, ref_fmu):
        fmu = load_fmu(ref_fmu)
        assert isinstance(fmu, FMUModelME3)


class TestFMI3CS:
    # TODO: Unsupported
    pass

class TestFMI3SE:
    # TODO: Unsupported
    pass
