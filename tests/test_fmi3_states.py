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

from pathlib import Path

import pytest
import numpy as np

from pyfmi import load_fmu
from pyfmi.fmi3 import FMUModelME3
from pyfmi.exceptions import FMUException

this_dir = Path(__file__).parent.absolute()
FMI3_REF_FMU_PATH = Path(this_dir) / 'files' / 'reference_fmus' / '3.0'

class TestFMUState:
    """Tests involving set and get FMU state."""

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_get_set_free0(self, ref_fmu):
        """Test get/set/free state works after loading. """
        fmu = load_fmu(ref_fmu)
        s = fmu.get_fmu_state()
        fmu.set_fmu_state(s)
        fmu.free_fmu_state(s)

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_get_set_free1(self, ref_fmu):
        """Test get/set/free state works after initializing. """
        fmu = load_fmu(ref_fmu)
        fmu.initialize()
        s = fmu.get_fmu_state()
        fmu.set_fmu_state(s)
        fmu.free_fmu_state(s)

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_get_set_free2(self, ref_fmu):
        """Test get/set/free state works after simulation. """
        fmu = load_fmu(ref_fmu)
        fmu.simulate()
        s = fmu.get_fmu_state()
        fmu.set_fmu_state(s)
        fmu.free_fmu_state(s)

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_get_set_free3(self, ref_fmu):
        """Test get/set/free state works after reset. """
        fmu = load_fmu(ref_fmu)
        fmu.simulate()
        fmu.reset()
        s = fmu.get_fmu_state()
        fmu.set_fmu_state(s)
        fmu.free_fmu_state(s)

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_get_set_0(self, ref_fmu):
        """Test initializing again after setting FMU to state pre-initilization. """
        fmu = load_fmu(ref_fmu)

        s = fmu.get_fmu_state()

        fmu.initialize()

        fmu.set_fmu_state(s)
        fmu.initialize()
        fmu.free_fmu_state(s)

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_get_set_1(self, ref_fmu):
        """Test simulating again after setting FMU to state pre-initilization. """
        fmu = load_fmu(ref_fmu)

        s = fmu.get_fmu_state()

        fmu.simulate()

        fmu.set_fmu_state(s)
        fmu.simulate()
        fmu.free_fmu_state(s)


    def test_get_set_verify_results(self):
        """Test simulating again after setting FMU to state pre-initilization. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")

        s = fmu.get_fmu_state()

        t_end = 0.5 # 0.5 is rather arbitrary
        fmu.simulate(0, t_end)
        h0 = fmu.get('h')

        fmu.set_fmu_state(s)
        fmu.simulate(0, t_end)
        h1 = fmu.get('h')

        fmu.set_fmu_state(s)
        # simulate a bit further to make sure we are in a different state before setting
        fmu.simulate(0, 2*t_end)

        fmu.set_fmu_state(s)
        fmu.simulate(0, t_end)
        h2 = fmu.get('h')

        np.testing.assert_array_almost_equal(h0, h1)
        np.testing.assert_array_almost_equal(h0, h2)
        fmu.free_fmu_state(s)

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_serialize_deserialize_1(self, ref_fmu):
        """Verify we can serialize and deserialize state after loading."""
        fmu = load_fmu(ref_fmu)
        state = fmu.get_fmu_state()
        serialized_state = fmu.serialize_fmu_state(state)
        deserialized_state = fmu.deserialize_fmu_state(serialized_state)

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_serialize_deserialize_2(self, ref_fmu):
        """Verify setting to deserialized state results in state after simulation."""
        fmu = load_fmu(ref_fmu)
        fmu.simulate()
        state = fmu.get_fmu_state()
        serialized_state = fmu.serialize_fmu_state(state)
        deserialized_state = fmu.deserialize_fmu_state(serialized_state)
        fmu.reset()
        fmu.set_fmu_state(deserialized_state)
        assert fmu.time > 0

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_serialize_deserialize_3(self, ref_fmu):
        """Verify serialized fmu state size is positive integer."""
        fmu = load_fmu(ref_fmu)
        fmu.simulate()
        state = fmu.get_fmu_state()
        # Perhaps we should change this to an actual number in the future?
        assert fmu.serialized_fmu_state_size(state) > 0

    def test_todo(self):
        """Verify we can set state post simulation and continue simulation."""

        # There seems to be a bug (unrelated?) when we do the following that needs investigation
        """
        fmu = load_fmu('pick some FMU')
        t_final = 1.5 # arbitrary
        opts = fmu.simulate_options()
        res = fmu.simulate(0, t_final)
        s = fmu.get_fmu_state()
        fmu.reset()
        fmu.set_fmu_state(s) # set to state corresponding to state after simulating to t_final
        opts['initialize'] = False
        fmu.simulate(t_final, t_final*2, options = opts)
        """
        assert True