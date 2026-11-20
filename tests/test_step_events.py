#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2026 Modelon AB
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

from pyfmi.fmi2 import FMUModelME2
from pyfmi.fmi3 import FMUModelME3
from pyfmi.common.diagnostics import EventType
from pyfmi.common.log import parse_xml_log

REFERENCE_FMU_PATH = Path(__file__).parent / 'files' / 'reference_fmus'
REFERENCE_FMU_PATHS = {FMUModelME2: REFERENCE_FMU_PATH / '2.0', FMUModelME3: REFERENCE_FMU_PATH / '3.0'}

STEP_EVENTS_SIGNALLED = 3

class StepEventModel:
    """Signals a step event on each of its first STEP_EVENTS_SIGNALLED completed integrator steps."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_events_signalled = 0
        self.event_iterations = 0

    def completed_integrator_step(self, *args, **kwargs):
        enter_event_mode, terminate_simulation = super().completed_integrator_step(*args, **kwargs)
        if self.step_events_signalled < STEP_EVENTS_SIGNALLED:
            self.step_events_signalled += 1
            enter_event_mode = True
        return enter_event_mode, terminate_simulation

    def enter_event_mode(self):
        self.event_iterations += 1
        super().enter_event_mode()

class EveryStepEventModel:
    """Signals a step event on every completed integrator step."""
    def completed_integrator_step(self, *args, **kwargs):
        _, terminate_simulation = super().completed_integrator_step(*args, **kwargs)
        return True, terminate_simulation

def load_model(model_class, fmu, *step_event_models):
    """Loads a reference FMU as model_class, signalling step events as the given step event models do."""
    model_class_with_step_events = type(model_class.__name__, (*step_event_models, model_class), {})
    return model_class_with_step_events(REFERENCE_FMU_PATHS[model_class] / fmu)

@pytest.fixture(params = [FMUModelME2, FMUModelME3], ids = ["fmi2", "fmi3"])
def model_class(request):
    return request.param

@pytest.mark.assimulo
class TestStepEvents:
    @pytest.mark.parametrize("solver", ["CVode", "Radau5ODE", "RungeKutta34", "Dopri5", "RodasODE", "LSODAR", "ExplicitEuler", "ImplicitEuler"])
    def test_each_step_event_runs_one_event_iteration(self, model_class, solver):
        """Test that each step event runs exactly one event iteration."""
        # Given
        model = load_model(model_class, "VanDerPol.fmu", StepEventModel)

        # When
        model.simulate(options = {"solver": solver})

        # Then
        assert model.event_iterations == STEP_EVENTS_SIGNALLED

    @pytest.mark.parametrize("fmu", ["Dahlquist.fmu", "BouncingBall.fmu"])
    @pytest.mark.parametrize("solver", ["CVode", "Radau5ODE", "ExplicitEuler"])
    def test_step_events_recorded_in_dynamic_diagnostics(self, model_class, solver, fmu):
        """Test that dynamic diagnostics record each step event."""
        # Given
        model = load_model(model_class, fmu, StepEventModel)

        # When
        res = model.simulate(options = {"solver": solver, "dynamic_diagnostics": True})

        # Then
        event_types = res["@Diagnostics.event_data.event_info.event_type"]
        assert np.count_nonzero(event_types == EventType.STEP_EVENT) == STEP_EVENTS_SIGNALLED
        assert res["@Diagnostics.nbr_step_events"][-1] == STEP_EVENTS_SIGNALLED

    @pytest.mark.parametrize("solver", ["CVode", "Radau5ODE", "LSODAR"])
    def test_state_event_in_step_with_step_event_recorded_as_state_event(self, model_class, solver):
        """Test that a state event is recorded as such when its step also signals a step event."""
        # Given
        opts = {"solver": solver, "dynamic_diagnostics": True, "CVode_options": {"inith": 1e-1}}
        state_events = load_model(model_class, "BouncingBall.fmu").simulate(options = opts)["@Diagnostics.nbr_state_events"][-1]
        model = load_model(model_class, "BouncingBall.fmu", EveryStepEventModel)

        # When
        res = model.simulate(options = opts)

        # Then
        assert res["@Diagnostics.nbr_state_events"][-1] == state_events

    @pytest.mark.parametrize("solver", ["CVode", "Radau5ODE", "LSODAR"])
    def test_time_event_in_step_with_step_event_recorded_as_time_event(self, model_class, solver):
        """Test that a time event is recorded as such when its step also signals a step event."""
        # Given
        opts = {"solver": solver, "dynamic_diagnostics": True, "CVode_options": {"inith": 1e-1}}
        time_events = load_model(model_class, "Stair.fmu").simulate(options = opts)["@Diagnostics.nbr_time_events"][-1]
        model = load_model(model_class, "Stair.fmu", EveryStepEventModel)

        # When
        res = model.simulate(options = opts)

        # Then
        assert res["@Diagnostics.nbr_time_events"][-1] == time_events

    def test_step_events_marked_in_debug_file(self, tmp_path, monkeypatch):
        """Test that the debug log marks each step event as a step event."""
        # Given
        monkeypatch.chdir(tmp_path)
        model = load_model(FMUModelME2, "Dahlquist.fmu", StepEventModel)
        opts = {"solver": "Radau5ODE", "logging": True, "result_handling": "csv"}

        # When
        model.simulate(options = opts)

        # Then
        debug_log = (tmp_path / "Dahlquist_debug.txt").read_text()
        assert debug_log.count("Step  event info:  True") == STEP_EVENTS_SIGNALLED

    def test_step_events_marked_in_xml_log(self, tmp_path, monkeypatch):
        """Test that the XML log marks each step event as a step event."""
        # Given
        monkeypatch.chdir(tmp_path)
        model = load_model(FMUModelME2, "Dahlquist.fmu", StepEventModel)
        opts = {"solver": "Radau5ODE", "logging": True, "result_handling": "csv"}

        # When
        model.simulate(options = opts)

        # Then
        events = parse_xml_log(model.extract_xml_log()).find("EventInfo")
        assert [event.step_event_info for event in events] == [True] * STEP_EVENTS_SIGNALLED
