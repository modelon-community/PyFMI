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

import numpy as np

from pyfmi import load_fmu
from pyfmi.fmi3 import FMUState3, FMUModelBase3

this_dir = Path(__file__).parent
FMI3_REF_FMU_PATH = Path(this_dir) / 'files' / 'reference_fmus' / '3.0'

class TestFMUState:
    """Tests involving set and get FMU state."""
    def _get_set_free_state(self, fmu: FMUModelBase3):
        s = fmu.get_fmu_state()
        assert isinstance(s, FMUState3)
        fmu.set_fmu_state(s)
        fmu.free_fmu_state(s)

    def test_get_set_state(self):
        """Test get/set/free state works after loading. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")
        self._get_set_free_state(fmu)

    def test_double_free(self):
        """Test double free_fmu_state calls. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")
        s = fmu.get_fmu_state()
        fmu.free_fmu_state(s)
        fmu.free_fmu_state(s)

    def test_get_set_state_after_initialization(self):
        """Test get/set/free state works after initializing. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")
        fmu.initialize()
        self._get_set_free_state(fmu)

    def test_get_set_state_after_simulation(self):
        """Test get/set/free state works after simulation. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")
        fmu.simulate(options = {"ncp": 0})
        self._get_set_free_state(fmu)

    def test_get_set_state_after_reset(self,):
        """Test get/set/free state works after reset. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")
        fmu.simulate(options = {"ncp": 0})
        fmu.reset()
        self._get_set_free_state(fmu)

    def test_get_set_state_pre_initialization(self):
        """Test initializing again after setting FMU to state pre-initialization. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")

        s = fmu.get_fmu_state()

        fmu.initialize()

        fmu.set_fmu_state(s)
        fmu.initialize()
        fmu.free_fmu_state(s)

    def test_get_set_state_pre_initialization_after_sim(self):
        """Test simulating again after setting FMU to state pre-initialization. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")

        s = fmu.get_fmu_state()

        fmu.simulate(options = {"ncp": 0})

        fmu.set_fmu_state(s)
        fmu.simulate(options = {"ncp": 0})
        fmu.free_fmu_state(s)

    def test_get_set_state_reload(self):
        """Test setting the FMU on a different FMU instance of the same FMU. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")
        fmu.simulate(options = {"ncp" : 0})
        s = fmu.get_fmu_state()

        fmu2 = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")
        fmu2.set_fmu_state(s)
        assert fmu.get("h")[0] != 0 # not a dummy
        assert fmu.get("h")[0] == fmu2.get("h")[0]

        fmu.free_fmu_state(s)

    def test_get_set_verify_results_1(self):
        """Test simulating again after setting FMU to state pre-initialization. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")
        s = fmu.get_fmu_state()
        t_end = 0.5

        fmu.simulate(0, t_end, options = {"ncp": 0})
        h0 = fmu.get('h')[0]
        assert h0 != 0 # Assure this is not a dummy variable

        fmu.set_fmu_state(s)
        fmu.simulate(0, t_end, options = {"ncp": 0})
        assert h0 == fmu.get('h')[0]
        fmu.free_fmu_state(s)

    def test_get_set_verify_results_2(self):
        """Test simulating again after setting FMU to state pre-initialization. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")
        s = fmu.get_fmu_state()
        t_end = 0.5

        fmu.simulate(0, t_end, options = {"ncp": 0})
        h0 = fmu.get('h')[0]
        assert h0 != 0 # Assure this is not a dummy variable

        # simulate a bit further to make sure we are in a different state before setting
        fmu.simulate(t_end, 2*t_end, options = {"ncp": 0, "initialize": False})

        fmu.set_fmu_state(s)
        fmu.simulate(0, t_end, options = {"ncp": 0})
        assert h0 == fmu.get('h')[0]
        fmu.free_fmu_state(s)

    def test_serialize_deserialize_after_load(self):
        """Verify we can serialize and deserialize state after loading."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")
        state = fmu.get_fmu_state()
        serialized_state = fmu.serialize_fmu_state(state)
        assert isinstance(serialized_state, list)
        assert len(serialized_state) == 2
        assert isinstance(serialized_state[0], np.ndarray)
        assert serialized_state[0].dtype == np.uint8 # bytes
        assert isinstance(serialized_state[1], list)

        deserialized_state = fmu.deserialize_fmu_state(serialized_state)
        assert isinstance(deserialized_state, FMUState3)
        fmu.free_fmu_state(state)

    def test_serialize_deserialize_deterministic(self):
        """Verify serialized FMU state is deterministic."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")
        fmu.simulate(options = {"ncp": 0})
        s1 = fmu.get_fmu_state()
        serialized_state_1 = fmu.serialize_fmu_state(s1)
        fmu.reset()
        fmu.simulate(options = {"ncp": 0})
        s2 = fmu.get_fmu_state()
        serialized_state_2 = fmu.serialize_fmu_state(s2)

        assert len(serialized_state_1[0]) > 0
        assert len(serialized_state_1[1]) > 0
        # XXX: This part is on the FMU, but still a pre-requisite for sensible testing overall
        np.testing.assert_array_equal(serialized_state_1[0], serialized_state_2[0])
        np.testing.assert_array_equal(serialized_state_1[1], serialized_state_2[1])
        fmu.free_fmu_state(s1)
        fmu.free_fmu_state(s2)

    def test_continue_simulation_from_state(self):
        """Verify we can get&set state post simulation and continue simulation."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")
        t_final = 1.5
        fmu.simulate(0, t_final, options = {"ncp": 0})

        s = fmu.get_fmu_state()
        fmu.reset()
        fmu.set_fmu_state(s) # set to state corresponding to state after simulating to t_final
        assert fmu.time == t_final

        fmu.simulate(t_final, t_final*2, options = {"ncp": 0, "initialize": False})
        fmu.free_fmu_state(s)

    def test_time_events(self):
        """Test get/set state for FMU with time events."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "Stair.fmu") # time events on the full seconds
        t_final_1, t_final_2 = 1.5, 2.5
        fmu.initialize()
        fmu.event_update()
        fmu.enter_continuous_time_mode()
        assert fmu.get("counter")[0] == 1

        res = fmu.simulate(0, t_final_1, options = {"ncp": 0, "initialize": False})
        assert res.solver.get_statistics()["ntimeevents"] == 1
        assert fmu.get("counter")[0] == 2

        assert fmu.get_event_info().nextEventTimeDefined
        assert fmu.get_event_info().nextEventTime == 2.

        state = fmu.get_fmu_state()

        res = fmu.simulate(t_final_1, t_final_2, options = {"ncp": 0, "initialize": False})
        assert res.solver.get_statistics()["ntimeevents"] == 1
        assert fmu.get("counter")[0] == 3

        assert fmu.get_event_info().nextEventTimeDefined
        assert fmu.get_event_info().nextEventTime == 3.

        fmu.set_fmu_state(state)

        assert fmu.get_event_info().nextEventTimeDefined
        assert fmu.get_event_info().nextEventTime == 2.

        res = fmu.simulate(t_final_1, t_final_2, options = {"ncp": 0, "initialize": False})
        assert res.solver.get_statistics()["ntimeevents"] == 1
        assert fmu.get("counter")[0] == 3

        assert fmu.get_event_info().nextEventTimeDefined
        assert fmu.get_event_info().nextEventTime == 3.

        fmu.free_fmu_state(state)
