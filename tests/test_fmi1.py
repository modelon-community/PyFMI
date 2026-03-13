#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019-2026 Modelon AB
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
import numpy as np
import types
import logging
from io import StringIO

from pyfmi.fmi import (
    FMUException,
    InvalidBinaryException,
    FMUModelME1,
    FMUModelCS1,
)
import pyfmi.fmi as fmi
from pyfmi.fmi_algorithm_drivers import (
    AssimuloFMIAlg, 
    AssimuloFMIAlgOptions
)
from pyfmi.test_util import (
    Dummy_FMUModelCS1,
    Dummy_FMUModelME1,
)
from pyfmi.common.io import ResultHandler

file_path = os.path.dirname(os.path.abspath(__file__))

FMU_PATHS     = types.SimpleNamespace()
FMU_PATHS.ME1 = types.SimpleNamespace()
FMU_PATHS.ME1.coupled_clutches = os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "CoupledClutches.fmu")
FMU_PATHS.ME1.nominal_test4    = os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NominalTest4.fmu")

class NoSolveAlg(AssimuloFMIAlg):
    """
    Algorithm that skips the solve step. Typically necessary to test DummyFMUs that
    don't have an implementation that can handle that step.
    """
    def solve(self):
        pass

@pytest.mark.assimulo
class Test_FMUModelME1_Simulation:
    def test_simulate_with_debug_option_no_state(self):
        """ Verify that an instance of CVodeDebugInformation is created """
        model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NoState.Example1.fmu"), _connect_dll=False)

        opts=model.simulate_options()
        opts["logging"] = True
        opts["result_handling"] = "csv" # set to anything except 'binary'

        #Verify that a simulation is successful
        res=model.simulate(options=opts)

        from pyfmi.debug import CVodeDebugInformation
        debug = CVodeDebugInformation("NoState_Example1_debug.txt")

    def test_no_result(self):
        model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NegatedAlias.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["result_handling"] = None
        res = model.simulate(options=opts)

        with pytest.raises(Exception):
            res._get_result_data()

        model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NegatedAlias.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["return_result"] = False
        res = model.simulate(options=opts)

        with pytest.raises(Exception):
            res._get_result_data()

    def test_custom_result_handler(self):
        model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NegatedAlias.fmu"), _connect_dll=False)

        class A:
            pass
        class B(ResultHandler):
            def get_result(self):
                return None

        opts = model.simulate_options()
        opts["result_handling"] = "hejhej"
        with pytest.raises(Exception):
            model.simulate(options=opts)
        opts["result_handling"] = "custom"
        with pytest.raises(Exception):
            model.simulate(options=opts)
        opts["result_handler"] = A()
        with pytest.raises(Exception):
            model.simulate(options=opts)
        opts["result_handler"] = B()
        res = model.simulate(options=opts)

    def setup_atol_auto_update_test_base(self):
        model = Dummy_FMUModelME1([], FMU_PATHS.ME1.nominal_test4, _connect_dll=False)
        model.override_nominal_continuous_states = False
        opts = model.simulate_options()
        opts["return_result"] = False
        opts["solver"] = "CVode"
        return model, opts

    def test_atol_auto_update1(self):
        """
        Tests that atol automatically gets updated when "atol = factor * pre_init_nominals".
        """
        model, opts = self.setup_atol_auto_update_test_base()

        opts["CVode_options"]["atol"] = 0.01 * model.nominal_continuous_states
        np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.02, 0.01])
        model.simulate(options=opts, algorithm=NoSolveAlg)
        np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.03, 0.03])

    def test_atol_auto_update2(self):
        """
        Tests that atol doesn't get auto-updated when heuristic fails.
        """
        model, opts = self.setup_atol_auto_update_test_base()

        opts["CVode_options"]["atol"] = (0.01 * model.nominal_continuous_states) + [0.01, 0.01]
        np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.03, 0.02])
        model.simulate(options=opts, algorithm=NoSolveAlg)
        np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.03, 0.02])

    def test_atol_auto_update3(self):
        """
        Tests that atol doesn't get auto-updated when nominals are never retrieved.
        """
        model, opts = self.setup_atol_auto_update_test_base()

        opts["CVode_options"]["atol"] = [0.02, 0.01]
        np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.02, 0.01])
        model.simulate(options=opts, algorithm=NoSolveAlg)
        np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.02, 0.01])

    # NOTE:
    # There are more tests for ME2 for auto update of atol, but it should be enough to test
    # one FMI version for that, because they mainly test algorithm drivers functionality.


@pytest.mark.assimulo
class Test_FMUModelME1:
    def test_invalid_binary(self):
        err_msg = "The FMU could not be loaded."
        fmu = os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "RLC_Circuit.fmu")
        with pytest.raises(InvalidBinaryException, match = err_msg):
            FMUModelME1(fmu, _connect_dll=True)

    def test_get_time_varying_variables(self):
        model = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "RLC_Circuit.fmu"), _connect_dll=False)

        [r,i,b] = model.get_model_time_varying_value_references()
        [r_f, i_f, b_f] = model.get_model_time_varying_value_references(filter="*")

        assert len(r) == len(r_f)
        assert len(i) == len(i_f)
        assert len(b) == len(b_f)

    def test_get_time_varying_variables_with_alias(self):
        model = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "Alias1.fmu"), _connect_dll=False)

        [r,i,b] = model.get_model_time_varying_value_references(filter="y*")

        assert len(r) == 1
        assert r[0] == model.get_variable_valueref("y")

    def test_get_variable_by_valueref(self):
        bounce = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        assert "der(v)" == bounce.get_variable_by_valueref(3)
        assert "v" == bounce.get_variable_by_valueref(2)

        with pytest.raises(FMUException):
            bounce.get_variable_by_valueref(7)

    def test_get_variable_nominal_valueref(self):
        bounce = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        assert bounce.get_variable_nominal("v") == bounce.get_variable_nominal(valueref=2)

    def test_log_file_name(self):
        model = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        assert os.path.exists("bouncingBall_log.txt")
        model = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False, log_file_name="Test_log.txt")
        assert os.path.exists("Test_log.txt")

    def test_ode_get_sizes(self):
        bounce = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        dq = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "dq.fmu"), _connect_dll=False)

        [nCont,nEvent] = bounce.get_ode_sizes()
        assert nCont == 2
        assert nEvent == 1

        [nCont,nEvent] = dq.get_ode_sizes()
        assert nCont == 1
        assert nEvent == 0

    def test_get_fmi_options(self):
        """
        Test that simulate_options on an FMU returns the correct options
        class instance.
        """
        bounce = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        assert isinstance(bounce.simulate_options(), AssimuloFMIAlgOptions)

    def test_get_xxx_empty(self):
        """ Test that get_xxx([]) do not calls do not trigger calls to FMU. """
        model = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        ## Tests that these do not crash and return empty arrays/lists
        assert len(model.get_real([]))    == 0, "get_real   ([]) has non-empty return"
        assert len(model.get_integer([])) == 0, "get_integer([]) has non-empty return"
        assert len(model.get_boolean([])) == 0, "get_boolean([]) has non-empty return"
        assert len(model.get_string([]))  == 0, "get_string ([]) has non-empty return"

class Test_FMUModelCS1:
    def test_invalid_binary(self):
        err_msg = "The FMU could not be loaded."
        fmu = os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "NegatedAlias.fmu")
        with pytest.raises(InvalidBinaryException, match = err_msg):
            model = FMUModelCS1(fmu, _connect_dll=True)

    def test_custom_result_handler(self):
        model = Dummy_FMUModelCS1([], os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "NegatedAlias.fmu"), _connect_dll=False)

        class A:
            pass
        class B(ResultHandler):
            def get_result(self):
                return None

        opts = model.simulate_options()
        opts["result_handling"] = "hejhej"
        with pytest.raises(Exception):
            model.simulate(options=opts)
        opts["result_handling"] = "custom"
        with pytest.raises(Exception):
            model.simulate(options=opts)
        opts["result_handler"] = A()
        with pytest.raises(Exception):
            model.simulate(options=opts)
        opts["result_handler"] = B()
        res = model.simulate(options=opts)

    def test_no_result(self):
        model = Dummy_FMUModelCS1([], os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "NegatedAlias.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["result_handling"] = None
        res = model.simulate(options=opts)

        with pytest.raises(Exception):
            res._get_result_data()

        model = Dummy_FMUModelCS1([], os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "NegatedAlias.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["return_result"] = False
        res = model.simulate(options=opts)

        with pytest.raises(Exception):
            res._get_result_data()

    def test_result_name_file(self):
        model = Dummy_FMUModelCS1([], os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "CoupledClutches.fmu"), _connect_dll=False)

        res = model.simulate(options={"result_handling":"file"})

        #Default name
        assert res.result_file == "CoupledClutches_result.txt"
        assert os.path.exists(res.result_file)

        model = Dummy_FMUModelCS1([], os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "CoupledClutches.fmu"), _connect_dll=False)
        res = model.simulate(options={"result_file_name":
                                    "CoupledClutches_result_test.txt"})

        #User defined name
        assert res.result_file == "CoupledClutches_result_test.txt"
        assert os.path.exists(res.result_file)

    def test_log_file_name(self):
        model = FMUModelCS1(os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "bouncingBall.fmu", ), _connect_dll=False)
        assert os.path.exists("bouncingBall_log.txt")
        model = FMUModelCS1(os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "bouncingBall.fmu"), _connect_dll=False, log_file_name="Test_log.txt")
        assert os.path.exists("Test_log.txt")

    def test_erreneous_ncp(self):
        model = Dummy_FMUModelCS1([], os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "NegatedAlias.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["ncp"] = 0
        with pytest.raises(FMUException):
            model.simulate(options=opts)
        opts["ncp"] = -1
        with pytest.raises(FMUException):
            model.simulate(options=opts)

@pytest.mark.assimulo
class Test_FMUModelBase:
    def test_unicode_description(self):
        full_path = os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "Description.fmu")
        model = FMUModelME1(full_path, _connect_dll=False)

        desc = model.get_variable_description("x")

        assert desc == "Test symbols '' ‘’"

    def test_get_erronous_nominals(self):
        model = FMUModelME1(FMU_PATHS.ME1.nominal_test4, _connect_dll=False)

        assert model.get_variable_nominal("x") == pytest.approx(2.0)
        assert model.get_variable_nominal("y") == pytest.approx(1.0)

    def test_caching(self):
        negated_alias = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NegatedAlias.fmu"), _connect_dll=False)

        assert len(negated_alias.cache) == 0 #No starting cache

        vars_1 = negated_alias.get_model_variables()
        vars_2 = negated_alias.get_model_variables()
        assert id(vars_1) == id(vars_2)

        vars_3 = negated_alias.get_model_variables(filter="*")
        assert id(vars_1) != id(vars_3)

        vars_4 = negated_alias.get_model_variables(type=0)
        assert id(vars_3) != id(vars_4)

        vars_5 = negated_alias.get_model_time_varying_value_references()
        vars_7 = negated_alias.get_model_time_varying_value_references()
        assert id(vars_5) != id(vars_1)
        assert id(vars_5) == id(vars_7)

        negated_alias = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NegatedAlias.fmu"), _connect_dll=False)

        assert len(negated_alias.cache) == 0 #No starting cache

        vars_6 = negated_alias.get_model_variables()
        assert id(vars_1) != id(vars_6)

    def test_get_scalar_variable(self):
        negated_alias = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NegatedAlias.fmu"), _connect_dll=False)

        sc_x = negated_alias.get_scalar_variable("x")

        assert sc_x.name == "x"
        assert sc_x.value_reference >= 0
        assert sc_x.type == fmi.FMI_REAL
        assert sc_x.variability == fmi.FMI_CONTINUOUS
        assert sc_x.causality == fmi.FMI_INTERNAL
        assert sc_x.alias == fmi.FMI_NO_ALIAS

        with pytest.raises(FMUException):
            negated_alias.get_scalar_variable("not_existing")

    def test_get_variable_description(self):
        model = FMUModelME1(FMU_PATHS.ME1.coupled_clutches, _connect_dll=False)
        assert model.get_variable_description("J1.phi") == "Absolute rotation angle of component"

    def test_simulation_without_initialization(self):
        model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        opts = model.simulate_options()
        opts["initialize"] = False

        with pytest.raises(FMUException):
            model.simulate(options=opts)

        model = Dummy_FMUModelCS1([], os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "bouncingBall.fmu"), _connect_dll=False)
        opts = model.simulate_options()
        opts["initialize"] = False

        with pytest.raises(FMUException):
            model.simulate(options=opts)

    def test_get_erroneous_nominals_capi_fmi1(self):
        """ Tests that erroneous nominals returned from getting nominals of continuous states get auto-corrected. """

        # Don't enable this except during local development. It will break all logging
        # for future test runs in the same python process.
        # If other tests also has this kind of property, only enable one at the time.
        # FIXME: Find a proper way to do it, or better, switch to a testing framework which has
        # support for it (e.g. unittest with assertLogs).
        one_off_test_logging = False

        model = Dummy_FMUModelME1([], FMU_PATHS.ME1.coupled_clutches, log_level=3, _connect_dll=False)

        if one_off_test_logging:
            log_stream = StringIO()
            logging.basicConfig(stream=log_stream, level=logging.WARNING)

        model.states_vref = [114, 115, 116, 117, 118, 119, 120, 121]
        # NOTE: Property 'nominal_continuous_states' is already overridden in Dummy_FMUModelME1, so just
        # call the underlying function immediately.
        xn = model._get_nominal_continuous_states()

        if one_off_test_logging:
            # Check warning is given:
            expected_msg1 = "The nominal value for clutch1.phi_rel is <0.0 which is illegal according to the " \
                        + "FMI specification. Setting the nominal to abs(-2.0)."
            expected_msg2 = "The nominal value for J4.w is 0.0 which is illegal according to the " \
                        + "FMI specification. Setting the nominal to 1.0."
            log = str(log_stream.getvalue())
            assert expected_msg1 in log  # First warning of 6.
            assert expected_msg2 in log  # Last warning of 6.

        # Check values are auto-corrected:
        assert xn[0] == pytest.approx(2.0)
        assert xn[1] == pytest.approx(1.0)
        assert xn[2] == pytest.approx(2.0)
        assert xn[3] == pytest.approx(2.0)
        assert xn[4] == pytest.approx(1.0)
        assert xn[5] == pytest.approx(2.0)
        assert xn[6] == pytest.approx(2.0)
        assert xn[7] == pytest.approx(1.0)
