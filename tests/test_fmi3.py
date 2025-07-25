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

import re
import logging
from io import StringIO
from pathlib import Path
import contextlib

import pytest
import numpy as np
import scipy.sparse as sps

from pyfmi import load_fmu
from pyfmi.fmi import (
    FMUModelME3,
)
from pyfmi.fmi3 import (
    FMI3_Type,
    FMI3_Causality,
    FMI3_Variability,
    FMI3_Initial,
    FMI3_DependencyKind,
    FMI3EventInfo,
)
from pyfmi.exceptions import (
    FMUException,
    InvalidFMUException,
    InvalidVersionException
)

# TODO: A lot of the tests here could be parameterized with the tests in test_fmi.py
# This would however require one of the following:
# a) Changing the tests in test_fmi.py to use the FMI1/2 reference FMUs
# b) Mocking the FMUs in some capacity

this_dir = Path(__file__).parent.absolute()
FMI3_REF_FMU_PATH = Path(this_dir) / 'files' / 'reference_fmus' / '3.0'

NUMPY_MAJOR_VERSION = int(np.__version__[0])
OVERFLOW_TEST_SET = [ # parameters for overflow testing
        ("Int32_input", 2_147_483_650),
        ("Int32_input", -2_147_483_650),
        ("Int16_input", 32_770),
        ("Int16_input", -32_770),
        ("Int8_input", 200),
        ("Int8_input", -200),
        ("UInt64_input", -1),
        ("UInt32_input", 4_294_967_300),
        ("UInt32_input", -1),
        ("UInt16_input", 65_540),
        ("UInt16_input", -1),
        ("UInt8_input", 260),
        ("UInt8_input", -1),
    ]


@contextlib.contextmanager
def temp_dir_context(tmpdir):
    """Provides a temporary directory as a context."""
    yield Path(tmpdir)

class TestFMI3LoadFMU:
    """Basic unit tests for FMI3 loading via 'load_fmu'."""

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Dahlquist.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "StateSpace.fmu",
        FMI3_REF_FMU_PATH / "Feedthrough.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_load_kind_auto(self, caplog, ref_fmu):
        """Test loading a ME FMU via kind 'auto'"""
        caplog.set_level(logging.WARNING)
        fmu = load_fmu(ref_fmu, kind = "auto")
        assert isinstance(fmu, FMUModelME3)
        experimental_msg = "FMI3 support is experimental."
        assert any(experimental_msg in msg for msg in caplog.messages)

    @pytest.mark.parametrize("ref_fmu", [FMI3_REF_FMU_PATH / "Clocks.fmu"])
    def test_load_kind_auto_SE(self, ref_fmu):
        """Test loading a SE only FMU via kind 'auto'"""
        msg = "Import of FMI3 Scheduled Execution FMUs is not supported"
        with pytest.raises(InvalidFMUException, match = re.escape(msg)):
            load_fmu(ref_fmu, kind = "auto")

    @pytest.mark.parametrize("ref_fmu", [FMI3_REF_FMU_PATH / "VanDerPol.fmu"])
    def test_load_kind_ME(self, ref_fmu):
        """Test loading an FMU with kind 'ME'"""
        fmu = load_fmu(ref_fmu, kind = "ME")
        assert isinstance(fmu, FMUModelME3)

    def test_get_event_info_1(self,):
        """Test get_event_info() works as expected; no event."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu", kind = "ME")
        fmu.initialize()
        fmu.event_update()

        event_info = fmu.get_event_info()
        assert isinstance(event_info, FMI3EventInfo)
        assert not event_info.newDiscreteDtatesNeeded
        assert not event_info.terminateSimulation
        assert not event_info.nominalsOfContinuousStatesChanged
        assert not event_info.valuesOfContinuousStatesChanged
        assert not event_info.nextEventTimeDefined
        assert event_info.nextEventTime == pytest.approx(0.0) # Could be anything really though

    def test_get_event_info_2(self):
        """Test get_event_info() works as expected; time events."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "Stair.fmu", kind = "ME")
        fmu.initialize()
        fmu.event_update()

        event_info = fmu.get_event_info()
        assert isinstance(event_info, FMI3EventInfo)
        assert not event_info.newDiscreteDtatesNeeded
        assert not event_info.terminateSimulation
        assert not event_info.nominalsOfContinuousStatesChanged
        assert not event_info.valuesOfContinuousStatesChanged
        assert event_info.nextEventTimeDefined
        assert event_info.nextEventTime == pytest.approx(1.0)

    @pytest.mark.parametrize("ref_fmu", [FMI3_REF_FMU_PATH / "VanDerPol.fmu"])
    def test_load_kind_CS(self, ref_fmu):
        """Test loading an FMU with kind 'CS'"""
        msg = "Import of FMI3 Co-Simulation FMUs is not yet supported."
        with pytest.raises(InvalidFMUException, match = msg):
            load_fmu(ref_fmu, kind = "CS")

    @pytest.mark.parametrize("ref_fmu", [FMI3_REF_FMU_PATH / "Clocks.fmu"])
    def test_load_kind_SE(self, ref_fmu):
        """Test loading an FMU with kind 'SE'"""
        msg = "Import of FMI3 Scheduled Execution FMUs is not supported."
        with pytest.raises(FMUException, match = re.escape(msg)):
            load_fmu(ref_fmu, kind = "SE")

    def test_get_model_identifier(self):
        """Test that model identifier is retrieved as expected."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        assert fmu.get_identifier() == 'VanDerPol'

    def test_get_get_version(self):
        """Test that FMI version is retrieved as expected."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        assert fmu.get_version() == '3.0'

    def test_get_name(self):
        """Test that FMI name is retrieved as expected."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        assert fmu.get_name() == 'van der Pol oscillator'

    def test_get_model_version(self):
        """Test that model version is retrieved as expected."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        # TODO: Update test with some FMU that has this field specified.
        #       For now it at least verifies the call doesn't raise an exception
        #       and all of the reference FMUs have omitted this field.
        assert fmu.get_model_version() == ''

    def test_instantiation(self, tmpdir):
        """ Test that instantiation works by verifying the output in the log."""
        with temp_dir_context(tmpdir) as temp_path:
             # log_level set to 5 required by test
            fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu", log_level=5)

        substring_to_find = 'Successfully loaded all the interface functions'
        assert any(substring_to_find in line for line in fmu.get_log())

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Dahlquist.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "StateSpace.fmu",
        FMI3_REF_FMU_PATH / "Feedthrough.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_initialize_reset_terminate(self, ref_fmu):
        """Test initialize, reset and terminate of all the ME reference FMUs. """
        fmu = load_fmu(ref_fmu)
        # Should simply pass without any exceptions
        fmu.initialize()
        fmu.reset()

        # Test initialize again after resetting followed by terminate,
        # since terminating does not require reset.
        fmu.initialize()
        fmu.terminate()

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Dahlquist.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "StateSpace.fmu",
        FMI3_REF_FMU_PATH / "Feedthrough.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_enter_continuous_time_mode(self, ref_fmu):
        """Test entering continuous time mode. """
        fmu = load_fmu(ref_fmu)
        # Should simply pass without any exceptions
        fmu.initialize()
        fmu.enter_continuous_time_mode()
        fmu.terminate()

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Dahlquist.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "StateSpace.fmu",
        FMI3_REF_FMU_PATH / "Feedthrough.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_enter_event_mode(self, ref_fmu):
        """Test enter event mode. """
        fmu = load_fmu(ref_fmu)
        # Should simply pass without any exceptions
        fmu.initialize()
        fmu.enter_continuous_time_mode()
        fmu.enter_event_mode()
        fmu.terminate()

    @pytest.mark.parametrize("ref_fmu", [
        FMI3_REF_FMU_PATH / "BouncingBall.fmu",
        FMI3_REF_FMU_PATH / "Dahlquist.fmu",
        FMI3_REF_FMU_PATH / "Resource.fmu",
        FMI3_REF_FMU_PATH / "StateSpace.fmu",
        FMI3_REF_FMU_PATH / "Feedthrough.fmu",
        FMI3_REF_FMU_PATH / "Stair.fmu",
        FMI3_REF_FMU_PATH / "VanDerPol.fmu",
    ])
    def test_initialize_manually(self, ref_fmu):
        """Test initialize all the ME reference FMUs by entering/exiting initialization mode manually. """
        fmu = load_fmu(ref_fmu)
        assert fmu.time is None
        # Should simply pass without any exceptions
        fmu.enter_initialization_mode()
        fmu.exit_initialization_mode()
        assert fmu.time == 0.0

    def test_get_double_terminate(self):
        """Test invalid call sequence raises an error. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        fmu.initialize()
        fmu.terminate()
        msg = "Termination of FMU failed, see log for possible more information."
        with pytest.raises(FMUException, match = msg):
            fmu.terminate()

    def test_get_default_experiment_start_time(self):
        """Test retrieving default experiment start time. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        assert fmu.get_default_experiment_start_time() == 0.0

    def test_free_instance_after_load(self):
        """Test invoke free instance after loading. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        fmu.free_instance()

    def test_free_instance_after_initialization(self):
        """Test invoke free instance after initialization. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        fmu.initialize()
        fmu.free_instance()

    def test_get_default_experiment_stop_time(self):
        """Test retrieving default experiment stop time. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        assert fmu.get_default_experiment_stop_time() == 20.0

    def test_get_default_experiment_tolerance(self):
        """Test retrieving default experiment tolerance. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        assert fmu.get_default_experiment_tolerance() == 0.0001

    def test_get_states_list(self):
        """Test retrieving states list and check its attributes. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        states = fmu.get_states_list()

        assert len(states) == 2
        x0 = states['x0']
        x1 = states['x1']

        assert x0.description == 'the first state'
        assert x1.description == 'the second state'

        assert x0.type == FMI3_Type.FLOAT64
        assert x1.type == FMI3_Type.FLOAT64

        assert x0.causality is FMI3_Causality.OUTPUT
        assert x1.causality is FMI3_Causality.OUTPUT

        assert x0.variability is FMI3_Variability.CONTINUOUS
        assert x1.variability is FMI3_Variability.CONTINUOUS

        assert x0.value_reference == 1
        assert x1.value_reference == 3

        assert x0.initial is FMI3_Initial.EXACT
        assert x1.initial is FMI3_Initial.EXACT

    def test_get_states_list_no_states(self):
        """Test retrieving states list for model without states. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "Stair.fmu")
        assert len(fmu.get_states_list()) == 0

    def test_get_derivatives_list(self):
        """Test retrieving derivatives list and check its attributes. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        derivatives = fmu.get_derivatives_list()

        assert len(derivatives) == 2
        derx0 = derivatives['der(x0)']
        derx1 = derivatives['der(x1)']

        assert derx0.description == ''
        assert derx1.description == ''

        assert derx0.type == FMI3_Type.FLOAT64
        assert derx1.type == FMI3_Type.FLOAT64

        assert derx0.causality is FMI3_Causality.LOCAL
        assert derx1.causality is FMI3_Causality.LOCAL

        assert derx0.variability is FMI3_Variability.CONTINUOUS
        assert derx1.variability is FMI3_Variability.CONTINUOUS

        assert derx0.value_reference == 2
        assert derx1.value_reference == 4

        assert derx0.initial is FMI3_Initial.CALCULATED
        assert derx1.initial is FMI3_Initial.CALCULATED

    def test_get_derivatives_list_no_states(self):
        """Test retrieving derivatives list for model without derivatives. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "Stair.fmu")
        assert len(fmu.get_derivatives_list()) == 0

    def test_get_relative_tolerance(self):
        """Test get_relative_tolerance(). """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        assert fmu.get_relative_tolerance() == 0.0001

    def test_get_absolute_tolerances(self):
        """Test get_absolute_tolerances(). """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        fmu.initialize()
        np.testing.assert_array_almost_equal(fmu.get_absolute_tolerances(), np.array([1e-6, 1e-6]))

    def test_get_tolerances(self):
        """Test get_tolerances(). """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        fmu.initialize()
        tolerances = fmu.get_tolerances()
        assert tolerances[0] == 0.0001
        assert tolerances[0] == fmu.get_relative_tolerance()
        np.testing.assert_array_almost_equal(tolerances[1], np.array([1e-6, 1e-6]))

    def test_get_tolerances_exception(self):
        """Test that FMUException is raised if FMU is not initialized before get_tolerances()."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = load_fmu(fmu_path)
        msg = "Unable to retrieve the absolute tolerance, FMU needs to be initialized."
        with pytest.raises(FMUException, match = msg):
            fmu.get_tolerances()

    def test_get_and_set_states(self):
        """Test get and set of states."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = load_fmu(fmu_path)

        assert fmu.get('x0') == np.array([2.])
        fmu.set('x0', 3.0)
        assert fmu.get('x0') == np.array([3.])

        assert fmu.get('x1') == np.array([0.])
        fmu.set('x1', 1.12)
        assert fmu.get('x1') == np.array([1.12])

    def test_get_derivatives_with_states_set(self):
        """Test retrieve derivatives, verify values combined with setting of states."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = load_fmu(fmu_path)
        fmu.initialize()

        # Note:
        #   M(x0) = 2;
        #   M(x1) = 0;
        #   M(mu) = 1;

        #   M(der_x0) = M(x1);
        #   M(der_x1) = M(mu) * ((1.0 - M(x0) * M(x0)) * M(x1)) - M(x0);
        assert all(fmu.get_derivatives() == np.array([0., -2.]))

        fmu.set('x0', 5)
        assert all(fmu.get_derivatives() == np.array([0., -5.]))

        fmu.set('x1', 1)
        assert all(fmu.get_derivatives() == np.array([1, -29]))

    def test_get_description(self):
        """Test get descriptions."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = load_fmu(fmu_path)
        assert fmu.get_variable_description('x0') == 'the first state'
        assert fmu.get_variable_description('x1') == 'the second state'
        assert fmu.get_variable_description('mu') == ''

    def test_get_description_variable_not_found(self):
        """Test get description on a variable that does not exist."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = load_fmu(fmu_path)
        with pytest.raises(FMUException, match = "The variable idontexist could not be found."):
            fmu.get_variable_description('idontexist')

    def test_get_model_variables(self):
        """ Test get_model_variables with default arguments. """
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = load_fmu(fmu_path)
        variables = fmu.get_model_variables()

        assert len(variables) == 6

        v = variables['time']
        assert v.description     == 'Simulation time'
        assert v.name            == 'time'
        assert v.value_reference == 0
        assert v.causality       is FMI3_Causality.INDEPENDENT
        assert v.initial         is FMI3_Initial.UNKNOWN
        assert v.type            is FMI3_Type.FLOAT64
        assert v.variability     is FMI3_Variability.CONTINUOUS


        v = variables['x0']
        assert v.description     == 'the first state'
        assert v.name            == 'x0'
        assert v.value_reference == 1
        assert v.causality       is FMI3_Causality.OUTPUT
        assert v.initial         is FMI3_Initial.EXACT
        assert v.type            is FMI3_Type.FLOAT64
        assert v.variability     is FMI3_Variability.CONTINUOUS


        v = variables['der(x0)']
        assert v.description     == ''
        assert v.name            == 'der(x0)'
        assert v.value_reference == 2
        assert v.causality       is FMI3_Causality.LOCAL
        assert v.initial         is FMI3_Initial.CALCULATED
        assert v.type            is FMI3_Type.FLOAT64
        assert v.variability     is FMI3_Variability.CONTINUOUS


        v = variables['x1']
        assert v.description     == 'the second state'
        assert v.name            == 'x1'
        assert v.value_reference == 3
        assert v.causality       is FMI3_Causality.OUTPUT
        assert v.initial         is FMI3_Initial.EXACT
        assert v.type            is FMI3_Type.FLOAT64
        assert v.variability     is FMI3_Variability.CONTINUOUS


        v = variables['der(x1)']
        assert v.description     == ''
        assert v.name            == 'der(x1)'
        assert v.value_reference == 4
        assert v.causality       is FMI3_Causality.LOCAL
        assert v.initial         is FMI3_Initial.CALCULATED
        assert v.type            is FMI3_Type.FLOAT64
        assert v.variability     is FMI3_Variability.CONTINUOUS

        v = variables['mu']
        assert v.description     == ''
        assert v.name            == 'mu'
        assert v.value_reference == 5
        assert v.causality       is FMI3_Causality.PARAMETER
        assert v.initial         is FMI3_Initial.EXACT
        assert v.type            is FMI3_Type.FLOAT64
        assert v.variability     is FMI3_Variability.FIXED

    def test_get_model_variables_causality(self):
        """ Test get_model_variables by specifying causality. """
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = load_fmu(fmu_path)
        variables = fmu.get_model_variables(causality=FMI3_Causality.PARAMETER)

        assert len(variables) == 1
        assert 'mu' in variables

    def test_get_model_variables_type(self):
        """ Test get_model_variables by specifying type. """
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        variables = fmu.get_model_variables(type=FMI3_Type.FLOAT64)

        assert len(variables) == 7
        expected = [
            'time',
            'Float64_fixed_parameter',
            'Float64_tunable_parameter',
            'Float64_continuous_input',
            'Float64_continuous_output',
            'Float64_discrete_input',
            'Float64_discrete_output',
        ]

        assert expected == list(variables.keys())

    def test_get_model_variables_variability(self):
        """ Test get_model_variables by specifying variability. """
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = load_fmu(fmu_path)
        variables = fmu.get_model_variables(variability=FMI3_Variability.CONTINUOUS)

        assert len(variables) == 5
        assert 'mu' not in variables

    def test_get_model_variables_multiple(self):
        """ Test get_model_variables by specifying multiple arguments. """
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = load_fmu(fmu_path)
        variables = fmu.get_model_variables(type=FMI3_Type.FLOAT64, variability=FMI3_Variability.FIXED)

        assert len(variables) == 1
        assert 'mu' in variables

    def test_get_model_variables_several(self):
        """ Test get_model_variables by specifying several arguments. """
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        variables = fmu.get_model_variables(
            type=FMI3_Type.FLOAT64,
            causality=FMI3_Causality.INPUT,
            variability=FMI3_Variability.DISCRETE)

        assert len(variables) == 1
        assert 'Float64_discrete_input' in variables

    def test_get_model_variables_only_start(self):
        """ Test get_model_variables by specifying 'only_start'. """
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        variables = fmu.get_model_variables(only_start = True)

        expected = [
            'Float32_continuous_input',
            'Float32_discrete_input',
            'Float64_fixed_parameter',
            'Float64_tunable_parameter',
            'Float64_continuous_input',
            'Float64_discrete_input',
            'Int8_input',
            'UInt8_input',
            'Int16_input',
            'UInt16_input',
            'Int32_input',
            'UInt32_input',
            'Int64_input',
            'UInt64_input',
            'Boolean_input',
            'String_input',
            'Binary_input',
            'Enumeration_input']

        assert expected == list(variables.keys())

    def test_get_model_variables_only_fixed(self):
        """ Test get_model_variables by specifying 'only_fixed'. """
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        variables = fmu.get_model_variables(only_fixed = True)

        expected = ['Float64_fixed_parameter']

        assert expected == list(variables.keys())

    def test_get_model_variables_filter(self):
        """ Test get_model_variables by specifying filter. """
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = load_fmu(fmu_path)
        variables = fmu.get_model_variables(filter="der*")

        assert len(variables) == 2
        assert 'der(x0)' in variables
        assert 'der(x1)' in variables

    def test_get_model_variables_multiple_filters(self):
        """ Test get_model_variables by specifying multiple filters. """
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        variables = fmu.get_model_variables(filter=['*parameter', 'UInt16*'])

        expected = [
            'Float64_fixed_parameter',
            'Float64_tunable_parameter',
            'UInt16_input',
            'UInt16_output'
            ]

        assert expected == list(variables.keys())

    def test_get_model_variables_many_args(self):
        """ Test get_model_variables by specifying almost all inputs. """
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        variables = fmu.get_model_variables(
            type=FMI3_Type.FLOAT64,
            causality=FMI3_Causality.PARAMETER,
            variability=FMI3_Variability.FIXED)

        expected = [
            'Float64_fixed_parameter']

        assert expected == list(variables.keys())

        variables = fmu.get_model_variables(
            type=FMI3_Type.FLOAT64,
            causality=FMI3_Causality.PARAMETER,
            variability=FMI3_Variability.FIXED,
            filter="idontexist")

        expected = []

        assert expected == list(variables.keys())

    def test_get_input_list(self):
        """ Test get_input_list. """
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        inputs = fmu.get_input_list()

        expected = ['Float64_continuous_input']

        assert expected == list(inputs.keys())

    def test_get_output_list(self):
        """ Test get_output_list. """
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        outputs = fmu.get_output_list()

        expected = ['Float64_continuous_output']
        assert expected == list(outputs.keys())

    def test_get_output_dependencies(self):
        """ Test get_output_dependencies, Feedthrough."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        num_outputs = len(fmu.get_output_list())
        state_deps, input_deps = fmu.get_output_dependencies()

        assert num_outputs == 1
        assert len(state_deps) == num_outputs
        assert state_deps["Float64_continuous_output"] == []
        assert len(input_deps) == num_outputs
        assert input_deps["Float64_continuous_output"] == ["Float64_continuous_input"]

    def test_get_output_dependencies_2(self):
        """ Test get_output_dependencies, VanDerPol."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = load_fmu(fmu_path)
        num_outputs = len(fmu.get_output_list())
        state_deps, input_deps = fmu.get_output_dependencies()

        assert num_outputs == 2
        assert len(state_deps) == num_outputs
        assert state_deps["x0"] == ["x0"]
        assert state_deps["x1"] == ["x1"]
        assert len(input_deps) == num_outputs
        assert input_deps["x0"] == []
        assert input_deps["x1"] == []

    def test_get_output_dependencies_kind(self):
        """ Test get_output_dependencies_kind."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        num_outputs = len(fmu.get_output_list())
        state_deps_kinds, input_deps_kinds = fmu.get_output_dependencies_kind()

        assert num_outputs == 1
        assert len(state_deps_kinds) == num_outputs
        assert state_deps_kinds["Float64_continuous_output"] == []
        assert len(input_deps_kinds) == num_outputs
        assert input_deps_kinds["Float64_continuous_output"] == [FMI3_DependencyKind.CONSTANT]

    def test_get_derivatives_dependencies(self):
        """ Test get_derivatives_dependencies."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = load_fmu(fmu_path)
        num_ders = len(fmu.get_derivatives_list())
        state_deps, input_deps = fmu.get_derivatives_dependencies()

        assert num_ders == 2
        assert len(state_deps) == num_ders
        assert state_deps["der(x0)"] == ["x1"]
        assert state_deps["der(x1)"] == ["x0", "x1"]
        assert len(input_deps) == num_ders
        assert input_deps["der(x0)"] == []
        assert input_deps["der(x1)"] == []

    def test_get_derivatives_dependencies_kind(self):
        """ Test get_derivatives_dependencies_kind."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = load_fmu(fmu_path)
        num_ders = len(fmu.get_derivatives_list())
        state_deps_kinds, input_deps_kinds = fmu.get_derivatives_dependencies_kind()

        assert num_ders == 2
        assert len(state_deps_kinds) == num_ders
        assert state_deps_kinds["der(x0)"] == [FMI3_DependencyKind.CONSTANT]
        assert state_deps_kinds["der(x1)"] == [FMI3_DependencyKind.DEPENDENT, FMI3_DependencyKind.DEPENDENT]
        assert len(input_deps_kinds) == num_ders
        assert input_deps_kinds["der(x0)"] == []
        assert input_deps_kinds["der(x1)"] == []

    @pytest.mark.parametrize("function_name, valuerefs, expected_result, expected_dtype",
        [
            ("get_float64", [5, 6], np.array([0, 0], dtype=np.float64), np.float64),
            ("get_float32", [1, 2], np.array([0, 0], dtype=np.float32), np.float32),
            ("get_int64", [23, 24], np.array([0, 0], dtype=np.int64), np.int64),
            ("get_int32", [19, 20], np.array([0, 0], dtype=np.int32), np.int32),
            ("get_int16", [15, 16], np.array([0, 0], dtype=np.int16), np.int16),
            ("get_int8", [11, 12], np.array([0, 0], dtype=np.int8), np.int8),
            ("get_uint64", [25, 26], np.array([0, 0], dtype=np.uint64), np.uint64),
            ("get_uint32", [21, 22], np.array([0, 0], dtype=np.uint32), np.uint32),
            ("get_uint16", [17, 18], np.array([0, 0], dtype=np.uint16), np.uint16),
            ("get_uint8", [13, 14], np.array([0, 0], dtype=np.uint8), np.uint8),
            ("get_boolean", [27, 28], np.array([False, False], dtype=np.bool_), np.bool_),
            ("get_enum", [33, 34], np.array([1, 1], dtype=np.int64), np.int64),
        ]
    )
    def test_getX(self, function_name, valuerefs, expected_result, expected_dtype):
        """Test the various get_TYPE([<valueref>]) functions."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        res = getattr(fmu, function_name)(valuerefs) # fmu.<function_name>(valuerefs)

        np.testing.assert_equal(res, expected_result)
        assert res.dtype == expected_dtype

    def test_get_string(self):
        """Test the get_string function."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        res = fmu.get_string([29])

        np.testing.assert_equal(res, ["Set me!"])
        assert type(res) == list

    @pytest.mark.parametrize("variable_name, value, expected_dtype",
        [
            ("Float64_continuous_input", 3.14, np.double),
            ("Float32_continuous_input", np.float32(3.14), np.float32),
            ("Int64_input", 9_223_372_036_854_775_806, np.int64),
            ("Int32_input", 2_147_483_647, np.int32),
            ("Int16_input", 32_766, np.int16),
            ("Int8_input", 126, np.int8),
            ("UInt64_input", 18_446_744_073_709_551_615, np.uint64),
            ("UInt32_input", 4_294_967_294, np.uint32),
            ("UInt16_input", 65_534, np.uint16),
            ("UInt8_input", 254, np.uint8),
            ("Boolean_input", True, np.bool_),
            ("Enumeration_input", 2, np.int64),
        ]
    )
    def test_set_get(self, variable_name, value, expected_dtype):
        """Test getting and setting variables of various types."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        fmu.set(variable_name, value)
        res = fmu.get(variable_name)
        assert res.dtype == expected_dtype
        assert res[0] == value

    @pytest.mark.skipif(NUMPY_MAJOR_VERSION > 1, reason = "Error for numpy>=2")
    @pytest.mark.parametrize("variable_name, value", OVERFLOW_TEST_SET)
    # XXX: Redundant in the future
    def test_set_get_out_of_bounds_overflow_old_numpy(self, variable_name, value):
        """Test setting too large/small value for various integer types."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        with pytest.warns(DeprecationWarning, match = "overflow"):
            fmu.set(variable_name, value)

    @pytest.mark.skipif(NUMPY_MAJOR_VERSION < 2, reason = "Only deprecated for numpy<2")
    @pytest.mark.parametrize("variable_name, value", OVERFLOW_TEST_SET)
    def test_set_get_out_of_bounds_overflow_new_numpy(self, variable_name, value):
        """Test setting too large/small value for various integer types."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        with pytest.raises(OverflowError):
            fmu.set(variable_name, value)

    @pytest.mark.parametrize("variable_name, value",
        [
            ("Int64_input", 9_223_372_036_854_775_810),
            ("Int64_input", -9_223_372_036_854_775_810),
            ("UInt64_input", 18_446_744_073_709_551_620),
        ]
    )
    def test_set_large_int_overflow(self, variable_name, value):
        """Test setting too large/small value for various integer types."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        with pytest.raises(OverflowError):
            fmu.set(variable_name, value)

    def test_set_get_string(self):
        """Test getting and setting of string variables."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = load_fmu(fmu_path)
        variable_name = "String_input"
        value = "hello string"
        fmu.set(variable_name, value)
        res = fmu.get(variable_name)
        assert type(res) == list
        assert res[0] == value

    def test_directional_derivatives(self):
        """Test directional derivatives."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = load_fmu(fmu_path)
        fmu.set("mu", 2)
        fmu.initialize()
        fmu.enter_continuous_time_mode()
        fmu.continuous_states = np.array([1., 1.])

        # sequence from FMUModelME3.get_directional_derivative docstring
        states = fmu.get_states_list()
        states_references = [s.value_reference for s in states.values()]
        derivatives = fmu.get_derivatives_list()
        derivatives_references = [d.value_reference for d in derivatives.values()]
        v = np.array([1., 1.])
        dv = fmu.get_directional_derivative(states_references, derivatives_references, v)
        assert dv[0] == 1
        assert dv[1] == -5

class Test_FMI3ME:
    """Basic unit tests for FMI3 import directly via the FMUModelME3 class."""
    @pytest.mark.parametrize("ref_fmu", [FMI3_REF_FMU_PATH / "VanDerPol.fmu"])
    def test_basic(self, ref_fmu):
        """Basic construction of FMUModelME3."""
        fmu = FMUModelME3(ref_fmu, _connect_dll = False)
        assert isinstance(fmu, FMUModelME3)

    @pytest.mark.parametrize("ref_fmu", [FMI3_REF_FMU_PATH / "Clocks.fmu"])
    def test_basic_wrong_fmu_type(self, ref_fmu):
        """Test using a non-ME FMU."""
        msg = "The FMU could not be loaded. This class only supports FMI 3.0 for Model Exchange."
        with pytest.raises(InvalidVersionException, match = msg):
            FMUModelME3(ref_fmu, _connect_dll = False)

    def test_get_nominals_of_continuous_states(self):
        """Test retrieve the nominals of the continuous states. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        fmu.initialize()
        # TODO: Remove this test in the future when we can simulate the FMU fully
        nominals = fmu._get_nominal_continuous_states()
        assert all(nominals == np.array([1., 1.]))

    def test_get_nominals_of_continuous_states_pre_init(self):
        """Test that Exception is raised if FMU is not initialized before retrieving nominals of continuous states."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = load_fmu(fmu_path)
        msg = "Unable to retrieve nominals of continuous states, FMU must first be initialized."
        with pytest.raises(FMUException, match = msg):
            fmu._get_nominal_continuous_states()

    def test_logfile_content(self):
        """Test that we get the log content from FMIL parsing the modelDescription.xml."""
        log_filename = "test_fmi3_log.txt"
        FMUModelME3(FMI3_REF_FMU_PATH / "VanDerPol.fmu", log_file_name = log_filename,
                    _connect_dll = False, log_level = 5)

        with open(log_filename, "r") as file:
            data = file.read()

        assert "FMIL: module = FMILIB, log level = 4: XML specifies FMI standard version 3.0" in data
        assert "FMIL: module = FMILIB, log level = 5: Parsing finished successfully" in data

    # TODO: FUTURE: Move to test_stream.py
    def test_logging_stream(self):
        """Test logging content from FMIL using a stream."""
        log_filename = StringIO("")
        fmu = FMUModelME3(FMI3_REF_FMU_PATH / "VanDerPol.fmu", log_file_name = log_filename,
                          _connect_dll = False, log_level = 5)
        log = fmu.get_log()

        assert "FMIL: module = FMILIB, log level = 4: XML specifies FMI standard version 3.0" in log
        assert "FMIL: module = FMILIB, log level = 5: Parsing finished successfully" in log

    @pytest.mark.parametrize("log_level", [1, 2, 3, 4, 5, 6, 7])
    def test_valid_log_levels(self, log_level):
        """Test valid log levels."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = FMUModelME3(fmu_path, log_level = log_level, _connect_dll = False)
        assert log_level == fmu.get_fmil_log_level()

    def test_valid_log_level_off(self):
        """Test logging nothing."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = FMUModelME3(fmu_path, log_level = 0, _connect_dll = False)
        msg = "Logging is not enabled"
        with pytest.raises(FMUException, match = msg):
            fmu.get_fmil_log_level()

    @pytest.mark.parametrize("log_level", [-1, 8, 1.0, "DEBUG"])
    def test_invalid_log_level(self, log_level):
        """Test invalid log levels."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        msg = "The log level must be an integer between 0 and 7"
        with pytest.raises(FMUException, match = msg):
            FMUModelME3(fmu_path, log_level = log_level, _connect_dll = False)

    def test_set_missing_variable(self):
        """Test setting a variable that does not exists."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = FMUModelME3(fmu_path, _connect_dll = False)
        var_name = "x0"
        err_msg = f"The variable {var_name} could not be found."
        with pytest.raises(FMUException, match = err_msg):
            fmu.set(var_name, 0.)

    def test_get_missing_variable(self):
        """Test getting a variable that does not exists."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = FMUModelME3(fmu_path, _connect_dll = False)
        var_name = "x0"
        err_msg = f"The variable {var_name} could not be found."
        with pytest.raises(FMUException, match = err_msg):
            fmu.get(var_name)

    def test_get_variable_valueref(self):
        """Test getting variable value references."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = FMUModelME3(fmu_path, _connect_dll = False)
        assert fmu.get_variable_valueref("time") == 0
        assert fmu.get_variable_valueref("Enumeration_input") == 33

    def test_get_variable_valueref_missing(self):
        """Test getting variable value references for variable that does not exist."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = FMUModelME3(fmu_path, _connect_dll = False)
        var_name = "x0"
        err_msg = f"The variable {var_name} could not be found."
        with pytest.raises(FMUException, match = err_msg):
            fmu.get_variable_valueref(var_name)

    @pytest.mark.parametrize("ref_fmu, expected_ode_size",
        [
            (FMI3_REF_FMU_PATH / "BouncingBall.fmu", (2, 1)),
            (FMI3_REF_FMU_PATH / "Dahlquist.fmu",    (1, 0)),
            (FMI3_REF_FMU_PATH / "Feedthrough.fmu",  (0, 0)),
            (FMI3_REF_FMU_PATH / "Resource.fmu",     (0, 0)),
            (FMI3_REF_FMU_PATH / "Stair.fmu" ,       (0, 0)),
            (FMI3_REF_FMU_PATH / "StateSpace.fmu",   (1, 0)),
            (FMI3_REF_FMU_PATH / "VanDerPol.fmu",    (2, 0)),
        ]
    )
    def test_get_ode_sizes(self, ref_fmu, expected_ode_size):
        """Test get ode sizes."""
        fmu = load_fmu(ref_fmu)
        assert fmu.get_ode_sizes() == expected_ode_size

    def test_get_variable_data_type_missing(self):
        """Test getting variable data type for missing variable."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = FMUModelME3(fmu_path, _connect_dll = False)
        var_name = "x0"
        err_msg = f"The variable {var_name} could not be found."
        with pytest.raises(FMUException, match = err_msg):
            fmu.get_variable_data_type(var_name)

    def test_set_array_variable(self):
        """Test setting an array variable (not yet supported). """
        fmu_path = FMI3_REF_FMU_PATH / "StateSpace.fmu"
        fmu = FMUModelME3(fmu_path)
        err_msg = "The length of valueref and values are inconsistent. Note: Array variables are not yet supported"
        with pytest.raises(FMUException, match = err_msg):
            fmu.set("x", np.array([1, 2, 3]))

    def test_get_continuous_states(self):
        """Test retrieve the continuous states."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = load_fmu(fmu_path)
        fmu.initialize()
        assert all(fmu.continuous_states == np.array([2.0, 0.0]))

    @pytest.mark.parametrize("variable_name, expected_datatype",
        [
            ("Float64_continuous_input", FMI3_Type.FLOAT64),
            ("Float32_continuous_input", FMI3_Type.FLOAT32),
            ("Int64_input", FMI3_Type.INT64),
            ("Int32_input", FMI3_Type.INT32),
            ("Int16_input", FMI3_Type.INT16),
            ("Int8_input" , FMI3_Type.INT8),
            ("UInt64_input", FMI3_Type.UINT64),
            ("UInt32_input", FMI3_Type.UINT32),
            ("UInt16_input", FMI3_Type.UINT16),
            ("UInt8_input",  FMI3_Type.UINT8),
            ("Boolean_input", FMI3_Type.BOOL),
            ("String_input", FMI3_Type.STRING),
            ("String_output", FMI3_Type.STRING),
            ("Binary_input", FMI3_Type.BINARY),
            ("Enumeration_input", FMI3_Type.ENUM),
        ]
    )
    def test_get_variable_data_type(self, variable_name, expected_datatype):
        """Test getting variable data types."""
        fmu_path = FMI3_REF_FMU_PATH / "Feedthrough.fmu"
        fmu = FMUModelME3(fmu_path, _connect_dll = False)
        assert fmu.get_variable_data_type(variable_name) is expected_datatype

    @pytest.mark.parametrize("fmu, variable_name, expected_causality",
        [
            ("Feedthrough.fmu", "time", FMI3_Causality.INDEPENDENT),
            ("Feedthrough.fmu", "Float64_fixed_parameter", FMI3_Causality.PARAMETER),
            ("Feedthrough.fmu", "Float64_continuous_input", FMI3_Causality.INPUT),
            ("Feedthrough.fmu", "Float64_continuous_output", FMI3_Causality.OUTPUT),
            ("StateSpace.fmu", "m", FMI3_Causality.STRUCTURAL_PARAMETER),
            ("StateSpace.fmu", "x", FMI3_Causality.LOCAL),
        ]
    )
    def test_get_variable_causality(self, fmu, variable_name, expected_causality):
        """Test getting variable data causalities."""
        fmu_path = FMI3_REF_FMU_PATH / fmu
        fmu = FMUModelME3(fmu_path, _connect_dll = False)
        assert fmu.get_variable_causality(variable_name) is expected_causality

    def test_simulate(self):
        """Test basic simulation of an FMU, no result handling."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = FMUModelME3(fmu_path)
        options = fmu.simulate_options()
        options["result_handling"] = None
        fmu.simulate(0, 20, options = options)

        # reference values taken from FMI2 VDP simulation, same settings
        assert fmu.get("x0")[0] == pytest.approx( 2.008130983012657)
        assert fmu.get("x1")[0] == pytest.approx(-0.042960828207896706)

    def test_generation_tool(self):
        """Test getting generation tool."""
        fmu_path = FMI3_REF_FMU_PATH / "VanDerPol.fmu"
        fmu = FMUModelME3(fmu_path)
        assert "Reference FMUs" in fmu.get_generation_tool()

    def test_get_event_indicators(self):
        """Test get_event_indicators function."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "BouncingBall.fmu")
        assert fmu.get_ode_sizes()[1] > 0

        fmu.simulate(options = {"ncp": 0})
        event_ind = fmu.get_event_indicators()
        assert len(event_ind) == 1
        assert event_ind[0] > 0
        assert event_ind[0] == fmu.get("h")[0] # same variable

    def test_get_event_indicators_empty(self):
        """Test get_event_indicators function for a model without state events"""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        assert fmu.get_ode_sizes()[1] == 0

        event_ind = fmu.get_event_indicators()
        assert len(event_ind) == 0

    def test_get_capability_flags(self):
        """Test getting capability flags."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        capabilities = fmu.get_capability_flags()

        assert capabilities["needsExecutionTool"] is False
        assert capabilities["canBeInstantiatedOnlyOncePerProcess"] is False
        assert capabilities["canGetAndSetFMUstate"] is True
        assert capabilities["canSerializeFMUstate"] is True
        assert capabilities["providesDirectionalDerivatives"] is True
        assert capabilities["providesAdjointDerivatives"] is True
        assert capabilities["providesPerElementDependencies"] is False
        assert capabilities["providesEvaluateDiscreteStates"] is False
        assert capabilities["needsCompletedIntegratorStep"] is False

    def test_get_state_space_representation(self):
        """Test get_state_space_representation function, VanDerPol."""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        fmu.initialize()

        A, B, C, D = fmu.get_state_space_representation(A = True, B = True, C = True, D = True, use_structure_info = False)
        
        np.testing.assert_array_almost_equal(A, np.array([[0., 1.], [-1., -3.]]))
        np.testing.assert_array_almost_equal(B, np.array([[], []]))
        # np.testing.assert_array_almost_equal(C, np.array([[1., 0.], [0., 1.]]))
        np.testing.assert_array_almost_equal(C, np.array([[0., 0.], [0., 0.]])) # XXX: Reference FMU actually provides wrong output with directional_derivatives here
        np.testing.assert_array_almost_equal(D, np.array([[], []]))

        # check the "use_structure_info" yield same result, but sparse
        As, Bs, Cs, Ds = fmu.get_state_space_representation(A = True, B = True, C = True, D = True, use_structure_info = True)
        assert isinstance(As, sps.csc_matrix)
        assert isinstance(Bs, sps.csc_matrix)
        assert isinstance(Cs, sps.csc_matrix)
        assert isinstance(Ds, sps.csc_matrix)

        np.testing.assert_array_almost_equal(A, As.todense())
        np.testing.assert_array_almost_equal(B, Bs.todense())
        np.testing.assert_array_almost_equal(C, Cs.todense())
        np.testing.assert_array_almost_equal(D, Ds.todense())

class Test_FMI3Alias:
    """Various tests surrounding aliases in FMI3."""
    @classmethod
    def setup_class(cls):
        path = str(this_dir / "files" / "FMUs" / "XML" / "ME3.0" / "alias")
        cls.fmu = FMUModelME3(path, allow_unzipped_fmu = True, _connect_dll = False)
        
    def test_get_model_variables_with_alias(self):
        """Test get_model_variables including aliases."""
        model_vars = self.fmu.get_model_variables(include_alias = True)
        assert "v5" in model_vars
        assert "v5_a1" in model_vars
        assert "v5_a2" in model_vars

        assert model_vars["v5"].description == "v5_desc"
        assert not model_vars["v5"].alias
        assert model_vars["v5_a1"].description == ""
        assert model_vars["v5_a1"].alias
        assert model_vars["v5_a2"].description == "v5_a2_desc"
        assert model_vars["v5_a2"].alias
    
    def test_get_model_variables_without_alias(self):
        """Test get_model_variables not including aliases."""
        model_vars = self.fmu.get_model_variables(include_alias = False)
        assert "v5_a1" not in model_vars
        assert "v5_a2" not in model_vars

    def test_get_variable_alias_base(self):
        """Test get_variable_alias_base."""
        assert self.fmu.get_variable_alias_base("v5") == "v5"
        assert self.fmu.get_variable_alias_base("v5_a1") == "v5"
        assert self.fmu.get_variable_alias_base("v5_a2") == "v5"

    def test_get_variable_alias(self):
        """Test get_variable_alias."""
        expected_result = {"v5": False, "v5_a1": True, "v5_a2": True}
        assert self.fmu.get_variable_alias("v5") == expected_result
        assert self.fmu.get_variable_alias("v5_a1") == expected_result
        assert self.fmu.get_variable_alias("v5_a2") == expected_result
        assert self.fmu.get_variable_alias("v4") == {"v4": False}

    def test_get_description_of_alias(self):
        """Test get_variable_description on aliases."""
        assert self.fmu.get_variable_description("v5") == "v5_desc"
        assert self.fmu.get_variable_description("v5_a1") == ""
        assert self.fmu.get_variable_description("v5_a2") == "v5_a2_desc"

class TestFMI3CS:
    # TODO: Unsupported for now
    pass

class TestFMI3SE:
    # TODO: Unsupported for now
    pass
