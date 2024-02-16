#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2024 Modelon AB
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
import os
import pylab as pl

from pyfmi.examples import (
    fmi_bouncing_ball_native,
    fmi20_bouncing_ball_native,
    fmi_bouncing_ball,
    fmi_bouncing_ball_cs,
    fmu_with_input,
    fmu_with_input_function
)

@pytest.mark.parametrize("example", [
    fmi_bouncing_ball_native,
    fmi20_bouncing_ball_native
])
def test_run_example_bouncing_ball_native(example):
    """Test pyfmi bouncing ball native examples."""
    example.run_demo()
    pl.close("all") # close all generated figures

@pytest.mark.parametrize("fmi_version", ["1.0", "2.0"])
@pytest.mark.parametrize("example", [
    fmi_bouncing_ball,
    fmi_bouncing_ball_cs
])
def test_run_example_bouncing_ball(example, fmi_version):
    """Test pyfmi bouncing ball example in various versions."""
    example.run_demo(version = fmi_version)
    pl.close("all") # close all generated figures

@pytest.mark.skipif("nt" not in os.name, 
                    reason = "FMU is example only contains windows binaries")
@pytest.mark.parametrize("example", [
    fmu_with_input,
    fmu_with_input_function
])
def test_run_example_with_input(example):
    """Test pyfmi examples with input."""
    example.run_demo()
    pl.close("all") # close all generated figures
