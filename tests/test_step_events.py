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

from pyfmi.fmi2 import FMUModelME2
from pyfmi.fmi3 import FMUModelME3

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
