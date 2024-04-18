#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019-2021 Modelon AB
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

import nose
import os
import numpy as np
from zipfile import ZipFile
import tempfile
import types
import logging
from io import StringIO

from pyfmi import testattr
from pyfmi.fmi import FMUException, InvalidOptionException, InvalidXMLException, InvalidBinaryException, InvalidVersionException, FMUModelME1, FMUModelCS1, load_fmu, FMUModelCS2, FMUModelME2, PyEventInfo
import pyfmi.fmi as fmi
from pyfmi.fmi_algorithm_drivers import AssimuloFMIAlg, AssimuloFMIAlgOptions, \
                                        PYFMI_JACOBIAN_LIMIT, PYFMI_JACOBIAN_SPARSE_SIZE_LIMIT
from pyfmi.tests.test_util import Dummy_FMUModelCS1, Dummy_FMUModelME1, Dummy_FMUModelME2, Dummy_FMUModelCS2, get_examples_folder
from pyfmi.common.io import ResultHandler
from pyfmi.common.algorithm_drivers import UnrecognizedOptionError
from pyfmi.common.core import create_temp_dir


class NoSolveAlg(AssimuloFMIAlg):
    """
    Algorithm that skips the solve step. Typically necessary to test DummyFMUs that
    don't have an implementation that can handle that step.
    """

    def solve(self):
        pass


assimulo_installed = True
try:
    import assimulo
except ImportError:
    assimulo_installed = False

file_path = os.path.dirname(os.path.abspath(__file__))

FMU_PATHS     = types.SimpleNamespace()
FMU_PATHS.ME1 = types.SimpleNamespace()
FMU_PATHS.ME2 = types.SimpleNamespace()
FMU_PATHS.ME1.coupled_clutches = os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "CoupledClutches.fmu")
FMU_PATHS.ME2.coupled_clutches = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "CoupledClutches.fmu")
FMU_PATHS.ME2.coupled_clutches_modified = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "CoupledClutchesModified.fmu")
FMU_PATHS.ME1.nominal_test4    = os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NominalTest4.fmu")
FMU_PATHS.ME2.nominal_test4    = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NominalTests.NominalTest4.fmu")



def _helper_unzipped_fmu_exception_invalid_dir(fmu_loader):
    """ Verify that we get an exception if unzipped FMU does not contain modelDescription.xml, which it should according to the FMI specification.
        The input argument is any of the FMU interfaces FMUModelME1, FMUModelME2, FMUModelCS1, FMUModelCS2 and load_fmu from pyfmi.fmi.
    """
    err_msg = "Specified fmu path '.*\\' needs to contain a modelDescription.xml according to the FMI specification"
    with tempfile.TemporaryDirectory() as temp_dir:
        with np.testing.assert_raises_regex(FMUException, err_msg):
            fmu = fmu_loader(temp_dir, allow_unzipped_fmu = True)

if assimulo_installed:
    class Test_FMUModelME1_Simulation:
        @testattr(stddist = True)
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

        @testattr(stddist = True)
        def test_no_result(self):
            model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NegatedAlias.fmu"), _connect_dll=False)

            opts = model.simulate_options()
            opts["result_handling"] = None
            res = model.simulate(options=opts)

            nose.tools.assert_raises(Exception,res._get_result_data)

            model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NegatedAlias.fmu"), _connect_dll=False)

            opts = model.simulate_options()
            opts["return_result"] = False
            res = model.simulate(options=opts)

            nose.tools.assert_raises(Exception,res._get_result_data)

        @testattr(stddist = True)
        def test_custom_result_handler(self):
            model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NegatedAlias.fmu"), _connect_dll=False)

            class A:
                pass
            class B(ResultHandler):
                def get_result(self):
                    return None

            opts = model.simulate_options()
            opts["result_handling"] = "hejhej"
            nose.tools.assert_raises(Exception, model.simulate, options=opts)
            opts["result_handling"] = "custom"
            nose.tools.assert_raises(Exception, model.simulate, options=opts)
            opts["result_handler"] = A()
            nose.tools.assert_raises(Exception, model.simulate, options=opts)
            opts["result_handler"] = B()
            res = model.simulate(options=opts)

        def setup_atol_auto_update_test_base(self):
            model = Dummy_FMUModelME1([], FMU_PATHS.ME1.nominal_test4, _connect_dll=False)
            model.override_nominal_continuous_states = False
            opts = model.simulate_options()
            opts["return_result"] = False
            opts["solver"] = "CVode"
            return model, opts

        @testattr(stddist = True)
        def test_atol_auto_update1(self):
            """
            Tests that atol automatically gets updated when "atol = factor * pre_init_nominals".
            """
            model, opts = self.setup_atol_auto_update_test_base()

            opts["CVode_options"]["atol"] = 0.01 * model.nominal_continuous_states
            np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.02, 0.01])
            model.simulate(options=opts, algorithm=NoSolveAlg)
            np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.03, 0.03])

        @testattr(stddist = True)
        def test_atol_auto_update2(self):
            """
            Tests that atol doesn't get auto-updated when heuristic fails.
            """
            model, opts = self.setup_atol_auto_update_test_base()

            opts["CVode_options"]["atol"] = (0.01 * model.nominal_continuous_states) + [0.01, 0.01]
            np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.03, 0.02])
            model.simulate(options=opts, algorithm=NoSolveAlg)
            np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.03, 0.02])

        @testattr(stddist = True)
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


class Test_FMUModelME1:

    @testattr(stddist = True)
    def test_unzipped_fmu_exception_invalid_dir(self):
        """ Verify that we get an exception if unzipped FMU does not contain modelDescription.xml, which it should according to the FMI specification. """
        _helper_unzipped_fmu_exception_invalid_dir(FMUModelME1)

    def _test_unzipped_bouncing_ball(self, fmu_loader):
        """ Simulates the bouncing ball FMU ME1.0 by unzipping the example FMU before loading, 'fmu_loader' is either FMUModelME1 or load_fmu. """
        tol = 1e-4
        fmu_dir = create_temp_dir()
        fmu = os.path.join(get_examples_folder(), 'files', 'FMUs', 'ME1.0', 'bouncingBall.fmu')
        with ZipFile(fmu, 'r') as fmu_zip:
            fmu_zip.extractall(path=fmu_dir)

        unzipped_fmu = fmu_loader(fmu_dir, allow_unzipped_fmu = True)
        res = unzipped_fmu.simulate(final_time = 2.0)
        value = np.abs(res.final('h') - (0.0424044))
        assert value < tol, "Assertion failed, value={} is not less than {}.".format(value, tol)

    @testattr(stddist = True)
    def test_unzipped_fmu1(self):
        """ Test load and simulate unzipped ME FMU 1.0 using FMUModelME1 """
        self._test_unzipped_bouncing_ball(FMUModelME1)

    @testattr(stddist = True)
    def test_unzipped_fmu2(self):
        """ Test load and simulate unzipped ME FMU 1.0 using load_fmu """
        self._test_unzipped_bouncing_ball(load_fmu)

    @testattr(stddist = True)
    def test_invalid_binary(self):
        err_msg = "The FMU could not be loaded."
        fmu = os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "RLC_Circuit.fmu")
        with nose.tools.assert_raises_regex(InvalidBinaryException, err_msg):
            model = FMUModelME1(fmu, _connect_dll=True)

    @testattr(stddist = True)
    def test_invalid_version(self):
        err_msg = "This class only supports FMI 1.0"
        fmu = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "LinearStability.SubSystem2.fmu")
        with nose.tools.assert_raises_regex(InvalidVersionException, err_msg):
            model = FMUModelME1(fmu, _connect_dll=True)

    @testattr(stddist = True)
    def test_get_time_varying_variables(self):
        model = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "RLC_Circuit.fmu"), _connect_dll=False)

        [r,i,b] = model.get_model_time_varying_value_references()
        [r_f, i_f, b_f] = model.get_model_time_varying_value_references(filter="*")

        assert len(r) == len(r_f)
        assert len(i) == len(i_f)
        assert len(b) == len(b_f)

    @testattr(stddist = True)
    def test_get_time_varying_variables_with_alias(self):
        model = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "Alias1.fmu"), _connect_dll=False)

        [r,i,b] = model.get_model_time_varying_value_references(filter="y*")

        assert len(r) == 1
        assert r[0] == model.get_variable_valueref("y")

    @testattr(stddist = True)
    def test_get_variable_by_valueref(self):
        bounce = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        assert "der(v)" == bounce.get_variable_by_valueref(3)
        assert "v" == bounce.get_variable_by_valueref(2)

        nose.tools.assert_raises(FMUException, bounce.get_variable_by_valueref,7)

    @testattr(stddist = True)
    def test_get_variable_nominal_valueref(self):
        bounce = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        assert bounce.get_variable_nominal("v") == bounce.get_variable_nominal(valueref=2)

    @testattr(windows_full = True)
    def test_default_experiment(self):
        model = FMUModelME1(FMU_PATHS.ME1.coupled_clutches, _connect_dll=False)

        assert np.abs(model.get_default_experiment_start_time()) < 1e-4
        assert np.abs(model.get_default_experiment_stop_time()-1.5) < 1e-4
        assert np.abs(model.get_default_experiment_tolerance()-0.0001) < 1e-4


    @testattr(stddist = True)
    def test_log_file_name(self):
        model = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        assert os.path.exists("bouncingBall_log.txt")
        model = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False, log_file_name="Test_log.txt")
        assert os.path.exists("Test_log.txt")

    @testattr(stddist = True)
    def test_ode_get_sizes(self):
        bounce = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        dq = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "dq.fmu"), _connect_dll=False)

        [nCont,nEvent] = bounce.get_ode_sizes()
        assert nCont == 2
        assert nEvent == 1

        [nCont,nEvent] = dq.get_ode_sizes()
        assert nCont == 1
        assert nEvent == 0

    @testattr(stddist = True)
    def test_get_name(self):
        bounce = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        dq = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "dq.fmu"), _connect_dll=False)

        assert bounce.get_name() == 'bouncingBall'
        assert dq.get_name() == 'dq'

    @testattr(stddist = True)
    def test_instantiate_jmu(self):
        """
        Test that FMUModelME1 can not be instantiated with a JMU file.
        """
        nose.tools.assert_raises(FMUException,FMUModelME1,'model.jmu')

    @testattr(stddist = True)
    def test_get_fmi_options(self):
        """
        Test that simulate_options on an FMU returns the correct options
        class instance.
        """
        bounce = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        assert isinstance(bounce.simulate_options(), AssimuloFMIAlgOptions)

    @testattr(stddist = True)
    def test_get_xxx_empty(self):
        """ Test that get_xxx([]) do not calls do not trigger calls to FMU. """
        model = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        ## Tests that these do not crash and return empty arrays/lists
        assert len(model.get_real([]))    == 0, "get_real   ([]) has non-empty return"
        assert len(model.get_integer([])) == 0, "get_integer([]) has non-empty return"
        assert len(model.get_boolean([])) == 0, "get_boolean([]) has non-empty return"
        assert len(model.get_string([]))  == 0, "get_string ([]) has non-empty return"

class Test_FMUModelCS1:

    @testattr(stddist = True)
    def test_unzipped_fmu_exception_invalid_dir(self):
        """ Verify that we get an exception if unzipped FMU does not contain modelDescription.xml, which it should according to the FMI specification. """
        _helper_unzipped_fmu_exception_invalid_dir(FMUModelCS1)

    def _test_unzipped_bouncing_ball(self, fmu_loader):
        """ Simulates the bouncing ball FMU CS1.0 by unzipping the example FMU before loading, 'fmu_loader' is either FMUModelCS1 or load_fmu. """
        tol = 1e-2
        fmu_dir = create_temp_dir()
        fmu = os.path.join(get_examples_folder(), 'files', 'FMUs', 'CS1.0', 'bouncingBall.fmu')
        with ZipFile(fmu, 'r') as fmu_zip:
            fmu_zip.extractall(path=fmu_dir)

        unzipped_fmu = fmu_loader(fmu_dir, allow_unzipped_fmu = True)
        res = unzipped_fmu.simulate(final_time = 2.0)
        value = np.abs(res.final('h') - (0.0424044))
        assert value < tol, "Assertion failed, value={} is not less than {}.".format(value, tol)

    @testattr(stddist = True)
    def test_unzipped_fmu1(self):
        """ Test load and simulate unzipped CS FMU 1.0 using FMUModelCS1 """
        self._test_unzipped_bouncing_ball(FMUModelCS1)

    @testattr(stddist = True)
    def test_unzipped_fmu2(self):
        """ Test load and simulate unzipped CS FMU 1.0 using load_fmu """
        self._test_unzipped_bouncing_ball(load_fmu)

    @testattr(stddist = True)
    def test_invalid_binary(self):
        err_msg = "The FMU could not be loaded."
        fmu = os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "NegatedAlias.fmu")
        with nose.tools.assert_raises_regex(InvalidBinaryException, err_msg):
            model = FMUModelCS1(fmu, _connect_dll=True)

    @testattr(stddist = True)
    def test_invalid_version(self):
        err_msg = "This class only supports FMI 1.0"
        fmu = os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "NegatedAlias.fmu")
        with nose.tools.assert_raises_regex(InvalidVersionException, err_msg):
            model = FMUModelCS1(fmu, _connect_dll=True)

    @testattr(stddist = True)
    def test_custom_result_handler(self):
        model = Dummy_FMUModelCS1([], os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "NegatedAlias.fmu"), _connect_dll=False)

        class A:
            pass
        class B(ResultHandler):
            def get_result(self):
                return None

        opts = model.simulate_options()
        opts["result_handling"] = "hejhej"
        nose.tools.assert_raises(Exception, model.simulate, options=opts)
        opts["result_handling"] = "custom"
        nose.tools.assert_raises(Exception, model.simulate, options=opts)
        opts["result_handler"] = A()
        nose.tools.assert_raises(Exception, model.simulate, options=opts)
        opts["result_handler"] = B()
        res = model.simulate(options=opts)

    @testattr(stddist = True)
    def test_no_result(self):
        model = Dummy_FMUModelCS1([], os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "NegatedAlias.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["result_handling"] = None
        res = model.simulate(options=opts)

        nose.tools.assert_raises(Exception,res._get_result_data)

        model = Dummy_FMUModelCS1([], os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "NegatedAlias.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["return_result"] = False
        res = model.simulate(options=opts)

        nose.tools.assert_raises(Exception,res._get_result_data)

    @testattr(stddist = True)
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

    @testattr(stddist = True)
    def test_default_experiment(self):
        model = FMUModelCS1(os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "CoupledClutches.fmu"), _connect_dll=False)

        assert np.abs(model.get_default_experiment_start_time()) < 1e-4
        assert np.abs(model.get_default_experiment_stop_time()-1.5) < 1e-4
        assert np.abs(model.get_default_experiment_tolerance()-0.0001) < 1e-4

    @testattr(stddist = True)
    def test_log_file_name(self):
        model = FMUModelCS1(os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "bouncingBall.fmu", ), _connect_dll=False)
        assert os.path.exists("bouncingBall_log.txt")
        model = FMUModelCS1(os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "bouncingBall.fmu"), _connect_dll=False, log_file_name="Test_log.txt")
        assert os.path.exists("Test_log.txt")

    @testattr(stddist = True)
    def test_erreneous_ncp(self):
        model = Dummy_FMUModelCS1([], os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "NegatedAlias.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["ncp"] = 0
        nose.tools.assert_raises(FMUException, model.simulate, options=opts)
        opts["ncp"] = -1
        nose.tools.assert_raises(FMUException, model.simulate, options=opts)

class Test_FMUModelBase:
    @testattr(stddist = True)
    def test_unicode_description(self):
        full_path = os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "Description.fmu")
        model = FMUModelME1(full_path, _connect_dll=False)

        desc = model.get_variable_description("x")

        assert desc == "Test symbols '' ‘’"

    @testattr(stddist = True)
    def test_get_erronous_nominals(self):
        model = FMUModelME1(FMU_PATHS.ME1.nominal_test4, _connect_dll=False)

        nose.tools.assert_almost_equal(model.get_variable_nominal("x"), 2.0)
        nose.tools.assert_almost_equal(model.get_variable_nominal("y"), 1.0)

    @testattr(stddist = True)
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

    @testattr(stddist = True)
    def test_get_scalar_variable(self):
        negated_alias = FMUModelME1(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NegatedAlias.fmu"), _connect_dll=False)

        sc_x = negated_alias.get_scalar_variable("x")

        assert sc_x.name == "x"
        assert sc_x.value_reference >= 0
        assert sc_x.type == fmi.FMI_REAL
        assert sc_x.variability == fmi.FMI_CONTINUOUS
        assert sc_x.causality == fmi.FMI_INTERNAL
        assert sc_x.alias == fmi.FMI_NO_ALIAS

        nose.tools.assert_raises(FMUException, negated_alias.get_scalar_variable, "not_existing")

    @testattr(stddist = True)
    def test_get_variable_description(self):
        model = FMUModelME1(FMU_PATHS.ME1.coupled_clutches, _connect_dll=False)
        assert model.get_variable_description("J1.phi") == "Absolute rotation angle of component"

    @testattr(stddist = True)
    def test_simulation_without_initialization(self):
        model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        opts = model.simulate_options()
        opts["initialize"] = False

        nose.tools.assert_raises(FMUException, model.simulate, options=opts)

        model = Dummy_FMUModelCS1([], os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "bouncingBall.fmu"), _connect_dll=False)
        opts = model.simulate_options()
        opts["initialize"] = False

        nose.tools.assert_raises(FMUException, model.simulate, options=opts)

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
        # NOTE: Property 'nominal_continuous_states' is already overriden in Dummy_FMUModelME1, so just
        # call the underlying function immediately.
        xn = model._get_nominal_continuous_states()

        if one_off_test_logging:
            # Check warning is given:
            expected_msg1 = "The nominal value for clutch1.phi_rel is <0.0 which is illegal according to the " \
                        + "FMI specification. Setting the nominal to abs(-2.0)."
            expected_msg2 = "The nominal value for J4.w is 0.0 which is illegal according to the " \
                        + "FMI specification. Setting the nominal to 1.0."
            log = str(log_stream.getvalue())
            nose.tools.assert_in(expected_msg1, log)  # First warning of 6.
            nose.tools.assert_in(expected_msg2, log)  # Last warning of 6.

        # Check values are auto-corrected:
        nose.tools.assert_almost_equal(xn[0], 2.0)  # -2.0
        nose.tools.assert_almost_equal(xn[1], 1.0)  #  0.0
        nose.tools.assert_almost_equal(xn[2], 2.0)  #  2.0
        nose.tools.assert_almost_equal(xn[3], 2.0)  # -2.0
        nose.tools.assert_almost_equal(xn[4], 1.0)  #  0.0
        nose.tools.assert_almost_equal(xn[5], 2.0)  #  2.0
        nose.tools.assert_almost_equal(xn[6], 2.0)  # -2.0
        nose.tools.assert_almost_equal(xn[7], 1.0)  #  0,0


class Test_LoadFMU:

    @testattr(stddist = True)
    def test_unzipped_fmu_exception_invalid_dir(self):
        """ Verify that we get an exception if unzipped FMU does not contain modelDescription.xml, which it should according to the FMI specification. """
        _helper_unzipped_fmu_exception_invalid_dir(load_fmu)

class Test_FMUModelCS2:

    @testattr(stddist = True)
    def test_unzipped_fmu_exception_invalid_dir(self):
        """ Verify that we get an exception if unzipped FMU does not contain modelDescription.xml, which it should according to the FMI specification. """
        _helper_unzipped_fmu_exception_invalid_dir(FMUModelCS2)

    def _test_unzipped_bouncing_ball(self, fmu_loader):
        """ Simulates the bouncing ball FMU CS2.0 by unzipping the example FMU before loading, 'fmu_loader' is either FMUModelCS2 or load_fmu. """
        tol = 1e-2
        fmu_dir = create_temp_dir()
        fmu = os.path.join(get_examples_folder(), 'files', 'FMUs', 'CS2.0', 'bouncingBall.fmu')
        with ZipFile(fmu, 'r') as fmu_zip:
            fmu_zip.extractall(path=fmu_dir)

        unzipped_fmu = fmu_loader(fmu_dir, allow_unzipped_fmu = True)
        res = unzipped_fmu.simulate(final_time = 2.0)
        value = np.abs(res.final('h') - (0.0424044))
        assert value < tol, "Assertion failed, value={} is not less than {}.".format(value, tol)

    @testattr(stddist = True)
    def test_unzipped_fmu1(self):
        """ Test load and simulate unzipped CS FMU 2.0 using FMUModelCS2 """
        self._test_unzipped_bouncing_ball(FMUModelCS2)

    @testattr(stddist = True)
    def test_unzipped_fmu2(self):
        """ Test load and simulate unzipped CS FMU 2.0 using load_fmu """
        self._test_unzipped_bouncing_ball(load_fmu)

    @testattr(stddist = True)
    def test_log_file_name(self):
        full_path = os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "CoupledClutches.fmu")
        model = FMUModelCS2(full_path, _connect_dll=False)

        path, file_name = os.path.split(full_path)
        assert model.get_log_filename() == file_name.replace(".","_")[:-4]+"_log.txt"

    @testattr(stddist = True)
    def test_invalid_binary(self):
        err_msg = "The FMU could not be loaded."
        fmu = os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "CoupledClutches.fmu")
        with nose.tools.assert_raises_regex(InvalidBinaryException, err_msg):
            model = FMUModelCS2(fmu, _connect_dll=True)

    @testattr(stddist = True)
    def test_invalid_version(self):
        err_msg = "The FMU version is not supported"
        fmu = os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "CoupledClutches.fmu")
        with nose.tools.assert_raises_regex(InvalidVersionException, err_msg):
            model = FMUModelCS2(fmu, _connect_dll=True)

    @testattr(stddist = True)
    def test_unzipped_fmu_exceptions(self):
        """ Verify exception is raised if 'fmu' is a file and allow_unzipped_fmu is set to True, with FMUModelCS2. """
        err_msg = "Argument named 'fmu' must be a directory if argument 'allow_unzipped_fmu' is set to True."
        with nose.tools.assert_raises_regex(FMUException, err_msg):
            model = FMUModelCS2(os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "LinearStability.SubSystem1.fmu"), _connect_dll=False, allow_unzipped_fmu=True)

    @testattr(stddist = True)
    def test_erroneous_ncp(self):
        model = FMUModelCS2(os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "CoupledClutches.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["ncp"] = 0
        nose.tools.assert_raises(FMUException, model.simulate, options=opts)
        opts["ncp"] = -1
        nose.tools.assert_raises(FMUException, model.simulate, options=opts)

    def _verify_downsample_result(self, ref_traj, test_traj, ncp, factor):
        """Auxiliary function for result_downsampling_factor testing. 
        Verify correct length and values of downsampled trajectory."""
        # all steps, except last one are checked = (ncp - 1) steps
        # ncp = 0 is illegal
        exptected_result_size = (ncp - 1)//factor + 2
        assert len(test_traj) == exptected_result_size, f"expected result size: {exptected_result_size}, actual : {len(test_traj)}"

        # selection mask for reference result
        downsample_indices = np.array([i%factor == 0 for i in range(ncp + 1)])
        downsample_indices[0] = True
        downsample_indices[-1] = True

        np.testing.assert_equal(ref_traj[downsample_indices], test_traj)

    def test_downsample_default(self):
        """ Test the default setup for result_downsampling_factor. """
        fmu = FMUModelCS2(os.path.join(get_examples_folder(), 'files', 'FMUs', 'CS2.0', 'bouncingBall.fmu'))
        opts = fmu.simulate_options()
        opts['ncp'] = 500

        assert opts['result_downsampling_factor'] == 1

        results = fmu.simulate(options = opts)

        assert len(results['time']) == 501

    def test_downsample_result(self):
        """ Test multiple result_downsampling_factor value and verify the result. """
        fmu = FMUModelCS2(os.path.join(get_examples_folder(), 'files', 'FMUs', 'CS2.0', 'bouncingBall.fmu'))
        opts = fmu.simulate_options()
        opts['ncp'] = 500
        test_var = "h" # height of bouncing ball

        # create reference result without down-sampling
        opts['result_downsampling_factor'] = 1
        ref_res = fmu.simulate(options = opts)
        assert len(ref_res['time']) == 501
        ref_res_traj = ref_res[test_var].copy()


        for f in [2, 3, 4, 5, 10, 100, 250, 499, 500, 600]:
            fmu.reset()
            opts['result_downsampling_factor'] = f
            res = fmu.simulate(options = opts)
            self._verify_downsample_result(ref_res_traj, res[test_var], opts['ncp'], f)

    def test_downsample_error_check_invalid_value(self):
        """ Verify we get an exception if the option is set to anything less than 1. """
        fmu = FMUModelCS2(os.path.join(get_examples_folder(), 'files', 'FMUs', 'CS2.0', 'bouncingBall.fmu'))
        opts = fmu.simulate_options()
        test_values = [-10, -20, -1, 0]

        # TODO: tidy up with pytest
        expected_substr = "Valid values for option 'result_downsampling_factor' are only positive integers"
        for value in test_values:
            opts['result_downsampling_factor'] = value
            try:
                fmu.simulate(options = opts)
                error_raised = False
            except FMUException as e:
                error_raised = True
                assert expected_substr in str(e), f"Error was {str(e)}, expected substring {expected_substr}"
            assert error_raised

    def test_error_check_invalid_value(self):
        """ Verify we get an exception if the option is set to anything that is not an integer. """
        fmu = FMUModelCS2(os.path.join(get_examples_folder(), 'files', 'FMUs', 'CS2.0', 'bouncingBall.fmu'))
        opts = fmu.simulate_options()
        test_values = [1/2, 1/3, "0.5", False]

        # TODO: tidy up with pytest
        expected_substr = "Option 'result_downsampling_factor' must be an integer,"
        for value in test_values:
            opts['result_downsampling_factor'] = value
            try:
                fmu.simulate(options = opts)
                error_raised = False
            except FMUException as e:
                error_raised = True
                assert expected_substr in str(e), f"Error was {str(e)}, expected substring {expected_substr}"
            assert error_raised

if assimulo_installed:
    class Test_FMUModelME2_Simulation:
        @testattr(stddist = True)
        def test_basicsens1(self):
            #Noncompliant FMI test as 'd' is parameter is not supposed to be able to be set during simulation
            model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "BasicSens1.fmu"), _connect_dll=False)

            def f(*args, **kwargs):
                d = model.values[model.variables["d"].value_reference]
                x = model.continuous_states[0]
                model.values[model.variables["der(x)"].value_reference] = d*x
                return np.array([d*x])

            model.get_derivatives = f

            opts = model.simulate_options()
            opts["sensitivities"] = ["d"]

            res = model.simulate(options=opts)
            nose.tools.assert_almost_equal(res.final('dx/dd'), 0.36789, 3)

            assert res.solver.statistics["nsensfcnfcns"] > 0

        @testattr(stddist = True)
        def test_basicsens1dir(self):
            model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "BasicSens1.fmu"), _connect_dll=False)

            caps = model.get_capability_flags()
            caps["providesDirectionalDerivatives"] = True
            model.get_capability_flags = lambda : caps

            def f(*args, **kwargs):
                d = model.values[model.variables["d"].value_reference]
                x = model.continuous_states[0]
                model.values[model.variables["der(x)"].value_reference] = d*x
                return np.array([d*x])

            def d(*args, **kwargs):
                if args[0][0] == 40:
                    return np.array([-1.0])
                else:
                    return model.continuous_states

            model.get_directional_derivative = d
            model.get_derivatives = f
            model._provides_directional_derivatives = lambda : True

            opts = model.simulate_options()
            opts["sensitivities"] = ["d"]

            res = model.simulate(options=opts)
            nose.tools.assert_almost_equal(res.final('dx/dd'), 0.36789, 3)

            assert res.solver.statistics["nsensfcnfcns"] > 0
            assert res.solver.statistics["nfcnjacs"] == 0

        @testattr(stddist = True)
        def test_basicsens2(self):
            model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "BasicSens2.fmu"), _connect_dll=False)

            caps = model.get_capability_flags()
            caps["providesDirectionalDerivatives"] = True
            model.get_capability_flags = lambda : caps

            def f(*args, **kwargs):
                d = model.values[model.variables["d"].value_reference]
                x = model.continuous_states[0]
                model.values[model.variables["der(x)"].value_reference] = d*x
                return np.array([d*x])

            def d(*args, **kwargs):
                if args[0][0] == 40:
                    return np.array([-1.0])
                else:
                    return model.continuous_states

            model.get_directional_derivative = d
            model.get_derivatives = f
            model._provides_directional_derivatives = lambda : True

            opts = model.simulate_options()
            opts["sensitivities"] = ["d"]

            res = model.simulate(options=opts)
            nose.tools.assert_almost_equal(res.final('dx/dd'), 0.36789, 3)

            assert res.solver.statistics["nsensfcnfcns"] == 0

        @testattr(stddist = True)
        def test_relative_tolerance(self):
            model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NoState.Example1.fmu"), _connect_dll=False)

            opts = model.simulate_options()
            opts["CVode_options"]["rtol"] = 1e-8

            res = model.simulate(options=opts)

            assert res.options["CVode_options"]["atol"] == 1e-10

        @testattr(stddist = True)
        def test_simulate_with_debug_option_no_state(self):
            """ Verify that an instance of CVodeDebugInformation is created """
            model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NoState.Example1.fmu"), _connect_dll=False)

            opts=model.simulate_options()
            opts["logging"] = True
            opts["result_handling"] = "csv" # set to anything except 'binary'

            #Verify that a simulation is successful
            res=model.simulate(options=opts)

            from pyfmi.debug import CVodeDebugInformation
            debug = CVodeDebugInformation("NoState_Example1_debug.txt")

        @testattr(stddist = True)
        def test_maxord_is_set(self):
            model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NoState.Example1.fmu"), _connect_dll=False)
            opts = model.simulate_options()
            opts["solver"] = "CVode"
            opts["CVode_options"]["maxord"] = 1

            res = model.simulate(final_time=1.5,options=opts)

            assert res.solver.maxord == 1

        @testattr(stddist = True)
        def test_with_jacobian_option(self):
            model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NoState.Example1.fmu"), _connect_dll=False)
            opts = model.simulate_options()
            opts["solver"] = "CVode"
            opts["result_handling"] = None

            def run_case(expected, default="Default"):
                model.reset()
                res = model.simulate(final_time=1.5,options=opts, algorithm=NoSolveAlg)
                assert res.options["with_jacobian"] == default, res.options["with_jacobian"]
                assert res.solver.problem._with_jacobian == expected, res.solver.problem._with_jacobian

            run_case(False)

            model.get_ode_sizes = lambda: (PYFMI_JACOBIAN_LIMIT+1, 0)
            run_case(True)

            opts["solver"] = "Radau5ODE"
            run_case(False)

            opts["solver"] = "CVode"
            opts["with_jacobian"] = False
            run_case(False, False)

            model.get_ode_sizes = lambda: (PYFMI_JACOBIAN_LIMIT-1, 0)
            opts["with_jacobian"] = True
            run_case(True, True)

        @testattr(stddist = True)
        def test_sparse_option(self):

            def run_case(expected_jacobian, expected_sparse, fnbr=0, nnz={}, set_sparse=False):
                class Sparse_FMUModelME2(Dummy_FMUModelME2):
                    def get_derivatives_dependencies(self):
                        return (nnz, {})

                model = Sparse_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NoState.Example1.fmu"), _connect_dll=False)
                opts = model.simulate_options()
                opts["solver"] = "CVode"
                opts["result_handling"] = None
                if set_sparse:
                    opts["CVode_options"]["linear_solver"] = "SPARSE"

                model.get_ode_sizes = lambda: (fnbr, 0)

                res = model.simulate(final_time=1.5,options=opts, algorithm=NoSolveAlg)
                assert res.solver.problem._with_jacobian == expected_jacobian, res.solver.problem._with_jacobian
                assert res.solver.linear_solver == expected_sparse, res.solver.linear_solver

            run_case(False, "DENSE")
            run_case(True, "DENSE",  PYFMI_JACOBIAN_SPARSE_SIZE_LIMIT+1, {"Dep": [1]*PYFMI_JACOBIAN_SPARSE_SIZE_LIMIT**2})
            run_case(True, "SPARSE", PYFMI_JACOBIAN_SPARSE_SIZE_LIMIT+1, {"Dep": [1]*PYFMI_JACOBIAN_SPARSE_SIZE_LIMIT})
            run_case(True, "SPARSE", PYFMI_JACOBIAN_SPARSE_SIZE_LIMIT+1, {"Dep": [1]*PYFMI_JACOBIAN_SPARSE_SIZE_LIMIT}, True)

        @testattr(stddist = True)
        def test_ncp_option(self):
            model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NoState.Example1.fmu"), _connect_dll=False)
            opts = model.simulate_options()
            assert opts["ncp"] == 500, opts["ncp"]

        @testattr(stddist = True)
        def test_solver_options(self):
            model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NoState.Example1.fmu"), _connect_dll=False)
            opts = model.simulate_options()

            try:
                opts["CVode_options"] = "ShouldFail"
                raise Exception("Setting an incorrect option should lead to exception being thrown, it wasn't")
            except UnrecognizedOptionError:
                pass

            opts["CVode_options"] = {"maxh":1.0}
            assert opts["CVode_options"]["atol"] == "Default", "Default should have been changed: " + opts["CVode_options"]["atol"]
            assert opts["CVode_options"]["maxh"] == 1.0, "Value should have been changed to 1.0: " + opts["CVode_options"]["maxh"]

        @testattr(stddist = True)
        def test_solver_options_using_defaults(self):
            model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NoState.Example1.fmu"), _connect_dll=False)
            opts = model.simulate_options()

            opts["CVode_options"] = {"maxh":1.0}
            assert opts["CVode_options"]["atol"] == "Default", "Default should have been changed: " + opts["CVode_options"]["atol"]
            assert opts["CVode_options"]["maxh"] == 1.0, "Value should have been changed to 1.0: " + opts["CVode_options"]["maxh"]

            opts["CVode_options"] = {"atol":1e-6} #Defaults should be used together with only the option atol set
            assert opts["CVode_options"]["atol"] == 1e-6, "Default should have been changed: " + opts["CVode_options"]["atol"]
            assert opts["CVode_options"]["maxh"] == "Default", "Value should have been default is: " + opts["CVode_options"]["maxh"]

        @testattr(stddist = True)
        def test_deepcopy_option(self):
            opts = AssimuloFMIAlgOptions()
            opts["CVode_options"]["maxh"] = 2.0

            import copy

            opts_copy = copy.deepcopy(opts)

            assert opts["CVode_options"]["maxh"] == opts_copy["CVode_options"]["maxh"], "Deepcopy not working..."

        @testattr(stddist = True)
        def test_maxh_option(self):
            model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NoState.Example1.fmu"), _connect_dll=False)
            opts = model.simulate_options()
            opts["result_handling"] = None

            def run_case(tstart, tstop, solver, ncp="Default"):
                model.reset()

                opts["solver"] = solver

                if ncp != "Default":
                    opts["ncp"] = ncp

                if opts["ncp"] == 0:
                    expected = 0.0
                else:
                    expected = (float(tstop)-float(tstart))/float(opts["ncp"])

                res = model.simulate(start_time=tstart, final_time=tstop,options=opts, algorithm=NoSolveAlg)
                assert res.solver.maxh == expected, res.solver.maxh
                assert res.options[solver+"_options"]["maxh"] == "Default", res.options[solver+"_options"]["maxh"]

            run_case(0,1,"CVode")
            run_case(0,1,"CVode", 0)
            run_case(0,1,"Radau5ODE")
            run_case(0,1,"Dopri5")
            run_case(0,1,"RodasODE")
            run_case(0,1,"LSODAR")
            run_case(0,1,"LSODAR")
        
        @testattr(stddist = True)
        def test_rtol_auto_update(self):
            """ Test that default rtol picks up the unbounded attribute. """
            model = Dummy_FMUModelME2([], FMU_PATHS.ME2.coupled_clutches_modified, _connect_dll=False)
            
            res = model.simulate()

            # verify appropriate rtol(s)
            for i, state in enumerate(model.get_states_list().keys()):
                if res.solver.supports.get('rtol_as_vector', False):
                    # automatic construction of rtol vector
                    if model.get_variable_unbounded(state):
                        nose.tools.assert_equal(res.solver.rtol[i], 0)
                    else:
                        nose.tools.assert_greater(res.solver.rtol[i], 0)
                else: # no support: scalar rtol
                    nose.tools.assert_true(isinstance(res.solver.rtol, float))

        @testattr(stddist = True)
        def test_rtol_vector_manual_valid(self):
            """ Tests manual valid rtol vector works; if supported. """

            model = Dummy_FMUModelME2([], FMU_PATHS.ME2.nominal_test4, _connect_dll=False)
            
            opts = model.simulate_options()
            opts["CVode_options"]["rtol"] = [1e-5, 0.]
            
            try:
                res = model.simulate(options=opts)
                # solver support
                nose.tools.assert_equal(res.solver.rtol[0], 1e-5)
                nose.tools.assert_equal(res.solver.rtol[1], 0.)
            except InvalidOptionException as e: # if no solver support
                nose.tools.assert_true(str(e).startswith("Failed to set the solver option 'rtol'"))
        
        @testattr(stddist = True)
        def test_rtol_vector_manual_size_mismatch(self):
            """ Tests invalid rtol vector: size mismatch. """
            model = Dummy_FMUModelME2([], FMU_PATHS.ME2.nominal_test4, _connect_dll=False)
            
            opts = model.simulate_options()
            opts["CVode_options"]["rtol"] = [1e-5, 0, 1e-5]
            
            err_msg = "If the relative tolerance is provided as a vector, it need to be equal to the number of states."
            with nose.tools.assert_raises_regex(InvalidOptionException, err_msg):
                model.simulate(options=opts)

        @testattr(stddist = True)
        def test_rtol_vector_manual_invalid(self):
            """ Tests invalid rtol vector: different nonzero values. """
            
            model = FMUModelME2(FMU_PATHS.ME2.coupled_clutches, _connect_dll=False)

            opts = model.simulate_options()
            opts["CVode_options"]["rtol"] = [1e-5, 0, 1e-5, 1e-5, 0, 1e-5,1e-6, 0]
            
            err_msg = "If the relative tolerance is provided as a vector, the values need to be equal except for zeros."
            with nose.tools.assert_raises_regex(InvalidOptionException, err_msg):
                model.simulate(options=opts)

        @testattr(stddist = True)
        def test_rtol_vector_manual_scalar_conversion(self):
            """ Test automatic scalar conversion of trivial rtol vector. """
            model = Dummy_FMUModelME2([], FMU_PATHS.ME2.nominal_test4, _connect_dll=False)
            
            opts = model.simulate_options()
            opts["CVode_options"]["rtol"] = [1e-5, 1e-5]
            
            #Verify no exception is raised as the rtol vector should be treated as a scalar
            res = model.simulate(options=opts)
            nose.tools.assert_equal(res.solver.rtol, 1e-5)
        
        @testattr(stddist = True)
        def test_rtol_vector_unsupported(self):
            """ Test that rtol as a vector triggers exceptions for unsupported solvers. """
            model = Dummy_FMUModelME2([], FMU_PATHS.ME2.nominal_test4, _connect_dll=False)
            opts = model.simulate_options()
            opts["result_handling"] = None

            def run_case(solver):
                model.reset()

                opts["solver"] = solver
                opts[solver+"_options"]["rtol"] = [1e-5, 0.0]
                
                try:
                    res = model.simulate(options=opts)
                    # solver support; check tolerances
                    nose.tools.assert_equal(res.solver.rtol[0], 1e-5)
                    nose.tools.assert_equal(res.solver.rtol[1], 0.0)
                except InvalidOptionException as e:
                    nose.tools.assert_true(str(e).startswith("Failed to set the solver option 'rtol'"))
                    return # OK

            run_case("CVode")
            run_case("Radau5ODE")
            run_case("Dopri5")
            run_case("RodasODE")
            run_case("LSODAR")
        
        def setup_atol_auto_update_test_base(self):
            model = Dummy_FMUModelME2([], FMU_PATHS.ME2.nominal_test4, _connect_dll=False)
            model.override_nominal_continuous_states = False
            opts = model.simulate_options()
            opts["return_result"] = False
            opts["solver"] = "CVode"
            return model, opts

        @testattr(stddist = True)
        def test_atol_auto_update1(self):
            """
            Tests that atol automatically gets updated when "atol = factor * pre_init_nominals".
            """
            model, opts = self.setup_atol_auto_update_test_base()

            opts["CVode_options"]["atol"] = 0.01 * model.nominal_continuous_states
            np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.02, 0.01])
            model.simulate(options=opts, algorithm=NoSolveAlg)
            np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.03, 0.03])

        @testattr(stddist = True)
        def test_atol_auto_update2(self):
            """
            Tests that atol doesn't get auto-updated when heuristic fails.
            """
            model, opts = self.setup_atol_auto_update_test_base()

            opts["CVode_options"]["atol"] = (0.01 * model.nominal_continuous_states) + [0.01, 0.01]
            np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.03, 0.02])
            model.simulate(options=opts, algorithm=NoSolveAlg)
            np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.03, 0.02])

        @testattr(stddist = True)
        def test_atol_auto_update3(self):
            """
            Tests that atol doesn't get auto-updated when nominals are never retrieved.
            """
            model, opts = self.setup_atol_auto_update_test_base()

            opts["CVode_options"]["atol"] = [0.02, 0.01]
            np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.02, 0.01])
            model.simulate(options=opts, algorithm=NoSolveAlg)
            np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.02, 0.01])

        @testattr(stddist = True)
        def test_atol_auto_update4(self):
            """
            Tests that atol is not auto-updated when it's set the "correct" way (post initialization).
            """
            model, opts = self.setup_atol_auto_update_test_base()
            
            model.setup_experiment()
            model.initialize()
            opts["initialize"] = False
            opts["CVode_options"]["atol"] = 0.01 * model.nominal_continuous_states
            np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.03, 0.03])
            model.simulate(options=opts, algorithm=NoSolveAlg)
            np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.03, 0.03])

        @testattr(stddist = True)
        def test_atol_auto_update5(self):
            """
            Tests that atol is automatically set and depends on rtol.
            """
            model, opts = self.setup_atol_auto_update_test_base()
            
            opts["CVode_options"]["rtol"] = 1e-6
            model.simulate(options=opts, algorithm=NoSolveAlg)
            np.testing.assert_allclose(opts["CVode_options"]["atol"], [3e-8, 3e-8])

        @testattr(stddist = True)
        def test_atol_auto_update6(self):
            """
            Tests that rtol doesn't affect explicitly set atol.
            """
            model, opts = self.setup_atol_auto_update_test_base()

            opts["CVode_options"]["rtol"] = 1e-9
            opts["CVode_options"]["atol"] = 0.01 * model.nominal_continuous_states
            np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.02, 0.01])
            model.simulate(options=opts, algorithm=NoSolveAlg)
            np.testing.assert_allclose(opts["CVode_options"]["atol"], [0.03, 0.03])


class Test_FMUModelME2:

    @testattr(stddist = True)
    def test_unzipped_fmu_exception_invalid_dir(self):
        """ Verify that we get an exception if unzipped FMU does not contain modelDescription.xml, which it should according to the FMI specification. """
        _helper_unzipped_fmu_exception_invalid_dir(FMUModelME2)

    def _test_unzipped_bouncing_ball(self, fmu_loader):
        """ Simulates the bouncing ball FMU ME2.0 by unzipping the example FMU before loading, 'fmu_loader' is either FMUModelME2 or load_fmu. """
        tol = 1e-4
        fmu_dir = create_temp_dir()
        fmu = os.path.join(get_examples_folder(), 'files', 'FMUs', 'ME2.0', 'bouncingBall.fmu')
        with ZipFile(fmu, 'r') as fmu_zip:
            fmu_zip.extractall(path=fmu_dir)

        unzipped_fmu = fmu_loader(fmu_dir, allow_unzipped_fmu = True)
        res = unzipped_fmu.simulate(final_time = 2.0)
        value = np.abs(res.final('h') - (0.0424044))
        assert value < tol, "Assertion failed, value={} is not less than {}.".format(value, tol)

    @testattr(stddist = True)
    def test_unzipped_fmu1(self):
        """ Test load and simulate unzipped ME FMU 2.0 using FMUModelME2 """
        self._test_unzipped_bouncing_ball(FMUModelME2)

    @testattr(stddist = True)
    def test_unzipped_fmu2(self):
        """ Test load and simulate unzipped ME FMU 2.0 using load_fmu """
        self._test_unzipped_bouncing_ball(load_fmu)

    @testattr(stddist = True)
    def test_unzipped_fmu_exceptions(self):
        """ Verify exception is raised if 'fmu' is a file and allow_unzipped_fmu is set to True, with FMUModelME2. """
        err_msg = "Argument named 'fmu' must be a directory if argument 'allow_unzipped_fmu' is set to True."
        with nose.tools.assert_raises_regex(FMUException, err_msg):
            model = FMUModelME2(os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "LinearStability.SubSystem2.fmu"), _connect_dll=False, allow_unzipped_fmu=True)

    @testattr(stddist = True)
    def test_invalid_binary(self):
        err_msg = "The FMU could not be loaded."
        with nose.tools.assert_raises_regex(InvalidBinaryException, err_msg):
            model = FMUModelME2(os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "LinearStability.SubSystem2.fmu"), _connect_dll=True)

    @testattr(stddist = True)
    def test_invalid_version(self):
        err_msg = "The FMU version is not supported by this class"
        with nose.tools.assert_raises_regex(InvalidVersionException, err_msg):
            model = FMUModelME2(os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "RLC_Circuit.fmu"), _connect_dll=True)

    @testattr(stddist = True)
    def test_estimate_directional_derivatives_linearstate(self):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "LinearStateSpace.fmu"), _connect_dll=False)

        def f(*args, **kwargs):
            derx1 = -1.*model.values[model.variables["x[1]"].value_reference] + model.values[model.variables["u[1]"].value_reference]
            derx2 = -1.*model.values[model.variables["x[2]"].value_reference] + model.values[model.variables["u[1]"].value_reference]

            model.values[model.variables["y[1]"].value_reference] = model.values[model.variables["x[1]"].value_reference] + model.values[model.variables["x[2]"].value_reference]

            return np.array([derx1, derx2])
        model.get_derivatives = f

        model.initialize()
        model.event_update()
        model.enter_continuous_time_mode()

        [As, Bs, Cs, Ds] = model.get_state_space_representation(use_structure_info=False)
        [A, B, C, D] = model.get_state_space_representation()

        assert As.shape == A.shape, str(As.shape)+' '+str(A.shape)
        assert Bs.shape == B.shape, str(Bs.shape)+' '+str(B.shape)
        assert Cs.shape == C.shape, str(Cs.shape)+' '+str(C.shape)
        assert Ds.shape == D.shape, str(Ds.shape)+' '+str(D.shape)

        assert np.allclose(As, A.toarray()), str(As)+' '+str(A.toarray())
        assert np.allclose(Bs, B.toarray()), str(Bs)+' '+str(B.toarray())
        assert np.allclose(Cs, C.toarray()), str(Cs)+' '+str(C.toarray())
        assert np.allclose(Ds, D.toarray()), str(Ds)+' '+str(D.toarray())

    @testattr(stddist = True)
    def test_estimate_directional_derivatives_without_structure_info(self):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "Bouncing_Ball.fmu"), _connect_dll=False)

        def f(*args, **kwargs):
            derh = model.values[model.variables["v"].value_reference]
            derv = -9.81
            model.values[model.variables["der(h)"].value_reference] = derh
            return np.array([derh, derv])
        model.get_derivatives = f

        model.initialize()
        model.event_update()
        model.enter_continuous_time_mode()

        [As, Bs, Cs, Ds] = model.get_state_space_representation(use_structure_info=False)
        [A, B, C, D] = model.get_state_space_representation()

        assert As.shape == A.shape, str(As.shape)+' '+str(A.shape)
        assert Bs.shape == B.shape, str(Bs.shape)+' '+str(B.shape)
        assert Cs.shape == C.shape, str(Cs.shape)+' '+str(C.shape)
        assert Ds.shape == D.shape, str(Ds.shape)+' '+str(D.shape)

        assert np.allclose(As, A.toarray()), str(As)+' '+str(A.toarray())
        assert np.allclose(Bs, B.toarray()), str(Bs)+' '+str(B.toarray())
        assert np.allclose(Cs, C.toarray()), str(Cs)+' '+str(C.toarray())
        assert np.allclose(Ds, D.toarray()), str(Ds)+' '+str(D.toarray())

    @testattr(stddist = True)
    def test_estimate_directional_derivatives_BCD(self):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "OutputTest2.fmu"), _connect_dll=False)

        def f(*args, **kwargs):
            x1 = model.values[model.variables["x1"].value_reference]
            x2 = model.values[model.variables["x2"].value_reference]
            u1 = model.values[model.variables["u1"].value_reference]

            model.values[model.variables["y1"].value_reference] = x1*x2 - u1
            model.values[model.variables["y2"].value_reference] = x2
            model.values[model.variables["y3"].value_reference] = u1 + x1

            model.values[model.variables["der(x1)"].value_reference] = -1.0
            model.values[model.variables["der(x2)"].value_reference] = -1.0
        model.get_derivatives = f

        model.initialize()
        model.event_update()
        model.enter_continuous_time_mode()

        for func in [model._get_B, model._get_C, model._get_D]:
            A = func(use_structure_info=True)
            B = func(use_structure_info=True, output_matrix=A)
            assert A is B #Test that the returned matrix is actually the same as the input
            assert np.allclose(A.toarray(),B.toarray())
            A = func(use_structure_info=False)
            B = func(use_structure_info=False, output_matrix=A)
            assert A is B
            assert np.allclose(A,B)
            C = func(use_structure_info=True, output_matrix=A)
            assert A is not C
            assert np.allclose(C.toarray(), A)
            D = func(use_structure_info=False, output_matrix=C)
            assert D is not C
            assert np.allclose(D, C.toarray())

        B = model._get_B(use_structure_info=True)
        C = model._get_C(use_structure_info=True)
        D = model._get_D(use_structure_info=True)

        assert np.allclose(B.toarray(), np.array([[0.0],[0.0]]))
        assert np.allclose(C.toarray(), np.array([[0.0, 0.0],[0.0, 1.0], [1.0, 0.0]]))
        assert np.allclose(D.toarray(), np.array([[-1.0],[0.0], [1.0]]))

        B = model._get_B(use_structure_info=False)
        C = model._get_C(use_structure_info=False)
        D = model._get_D(use_structure_info=False)

        assert np.allclose(B, np.array([[0.0],[0.0]]))
        assert np.allclose(C, np.array([[0.0, 0.0],[0.0, 1.0], [1.0, 0.0]]))
        assert np.allclose(D, np.array([[-1.0],[0.0], [1.0]]))

    @testattr(stddist = True)
    def test_output_dependencies(self):
        model = FMUModelME2(os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "OutputTest2.fmu"), _connect_dll=False)

        [state_dep, input_dep] = model.get_output_dependencies()

        assert state_dep["y1"][0] == "x1"
        assert state_dep["y1"][1] == "x2"
        assert state_dep["y2"][0] == "x2"
        assert state_dep["y3"][0] == "x1"
        assert input_dep["y1"][0] == "u1"
        assert input_dep["y3"][0] == "u1"
        assert len(input_dep["y2"]) == 0

    @testattr(stddist = True)
    def test_output_dependencies_2(self):
        model = FMUModelME2(FMU_PATHS.ME2.coupled_clutches, _connect_dll=False)

        [state_dep, input_dep] = model.get_output_dependencies()

        assert len(state_dep.keys()) == 0, len(state_dep.keys())
        assert len(input_dep.keys()) == 0, len(input_dep.keys())

    @testattr(stddist = True)
    def test_derivative_dependencies(self):
        model = FMUModelME2(os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NoState.Example1.fmu"), _connect_dll=False)

        [state_dep, input_dep] = model.get_derivatives_dependencies()

        assert len(state_dep.keys()) == 0, len(state_dep.keys())
        assert len(input_dep.keys()) == 0, len(input_dep.keys())

    @testattr(stddist = True)
    def test_exception_with_load_fmu(self):
        """ Verify exception is raised. """
        err_msg = "Argument named 'fmu' must be a directory if argument 'allow_unzipped_fmu' is set to True."
        test_file = 'abcdefgh1234567qwertyuiop.txt'
        rm_file = False
        if not os.path.isfile(test_file):
            with open(test_file, 'w') as fh:
                fh.write('')
            rm_file = True
        with nose.tools.assert_raises_regex(FMUException, err_msg):
            fmu = load_fmu(test_file,  allow_unzipped_fmu = True)
        if rm_file:
            os.remove(test_file)

    @testattr(stddist = True)
    def test_malformed_xml(self):
        nose.tools.assert_raises(InvalidXMLException, load_fmu, os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "MalFormed.fmu"))

    @testattr(stddist = True)
    def test_log_file_name(self):
        full_path = FMU_PATHS.ME2.coupled_clutches

        model = FMUModelME2(full_path, _connect_dll=False)

        path, file_name = os.path.split(full_path)
        assert model.get_log_filename() == file_name.replace(".","_")[:-4]+"_log.txt"

    @testattr(stddist = True)
    def test_units(self):
        model = FMUModelME2(FMU_PATHS.ME2.coupled_clutches, _connect_dll=False)

        assert model.get_variable_unit("J1.w") == "rad/s", model.get_variable_unit("J1.w")
        assert model.get_variable_unit("J1.phi") == "rad", model.get_variable_unit("J1.phi")

        nose.tools.assert_raises(FMUException, model.get_variable_unit, "clutch1.useHeatPort")
        nose.tools.assert_raises(FMUException, model.get_variable_unit, "clutch1.sss")
        nose.tools.assert_raises(FMUException, model.get_variable_unit, "clutch1.sss")

    @testattr(stddist = True)
    def test_display_units(self):
        model = FMUModelME2(FMU_PATHS.ME2.coupled_clutches, _connect_dll=False)

        assert model.get_variable_display_unit("J1.phi") == "deg", model.get_variable_display_unit("J1.phi")
        nose.tools.assert_raises(FMUException, model.get_variable_display_unit, "J1.w")

    @testattr(stddist = True)
    def test_get_xxx_empty(self):
        """ Test that get_xxx([]) do not calls do not trigger calls to FMU. """
        model = FMUModelME2(os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        ## Tests that these do not crash and return empty arrays/lists
        assert len(model.get_real([]))    == 0, "get_real   ([]) has non-empty return"
        assert len(model.get_integer([])) == 0, "get_integer([]) has non-empty return"
        assert len(model.get_boolean([])) == 0, "get_boolean([]) has non-empty return"
        assert len(model.get_string([]))  == 0, "get_string ([]) has non-empty return"

class Test_FMUModelBase2:

    @testattr(stddist = True)
    def test_relative_quantity(self):
        model = FMUModelME2(os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "test_type_definitions.fmu"), _connect_dll=False)

        rel = model.get_variable_relative_quantity("real_with_attr")
        assert rel is True, "Relative quantity should be True"
        rel = model.get_variable_relative_quantity("real_with_attr_false")
        assert rel is False, "Relative quantity should be False"

        rel = model.get_variable_relative_quantity("real_without_attr")
        assert rel is False, "Relative quantity should be (default) False"

        rel = model.get_variable_relative_quantity("real_with_typedef")
        assert rel is True, "Relative quantity should be True"

        nose.tools.assert_raises(FMUException, model.get_variable_relative_quantity, "int_with_attr")
    
    @testattr(stddist = True)
    def test_unbounded_attribute(self):
        model = FMUModelME2(os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "test_type_definitions.fmu"), _connect_dll=False)
        
        unbounded = model.get_variable_unbounded("real_with_attr")
        assert unbounded is True, "Unbounded should be True"
        unbounded = model.get_variable_unbounded("real_with_attr_false")
        assert unbounded is False, "Unbounded should be False"

        unbounded = model.get_variable_unbounded("real_without_attr")
        assert unbounded is False, "Unbounded should be (default) False"

        unbounded = model.get_variable_unbounded("real_with_typedef")
        assert unbounded is True, "Unbounded should be True"

        nose.tools.assert_raises(FMUException, model.get_variable_unbounded, "int_with_attr")

    @testattr(stddist = True)
    def test_unicode_description(self):
        model = FMUModelME2(os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "Description.fmu"), _connect_dll=False)

        desc = model.get_variable_description("x")

        assert desc == "Test symbols '' ‘’"

    @testattr(stddist = True)
    def test_declared_enumeration_type(self):
        model = FMUModelME2(os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "Enumerations.Enumeration3.fmu"), _connect_dll=False)

        enum = model.get_variable_declared_type("x")
        assert len(enum.items.keys()) == 2, len(enum.items.keys())
        enum = model.get_variable_declared_type("home")
        assert len(enum.items.keys()) == 4, len(enum.items.keys())

        assert enum.items[1][0] == "atlantis"
        assert enum.name == "Enumerations.Enumeration3.cities", "Got: " + enum.name
        assert enum.description == "", "Got: " + enum.description

        nose.tools.assert_raises(FMUException, model.get_variable_declared_type, "z")

    @testattr(stddist = True)
    def test_get_erroneous_nominals_xml(self):
        model = FMUModelME2(FMU_PATHS.ME2.nominal_test4, _connect_dll=False)

        nose.tools.assert_almost_equal(model.get_variable_nominal("x"), 2.0)
        nose.tools.assert_almost_equal(model.get_variable_nominal("y"), 1.0)

        nose.tools.assert_almost_equal(model.get_variable_nominal("x", _override_erroneous_nominal=False), -2.0)
        nose.tools.assert_almost_equal(model.get_variable_nominal("y", _override_erroneous_nominal=False), 0.0)

        x_vref = model.get_variable_valueref("x")
        y_vref = model.get_variable_valueref("y")

        nose.tools.assert_almost_equal(model.get_variable_nominal(valueref=x_vref), 2.0)
        nose.tools.assert_almost_equal(model.get_variable_nominal(valueref=y_vref), 1.0)

        nose.tools.assert_almost_equal(model.get_variable_nominal(valueref=x_vref, _override_erroneous_nominal=False), -2.0)
        nose.tools.assert_almost_equal(model.get_variable_nominal(valueref=y_vref, _override_erroneous_nominal=False), 0.0)

    def test_get_erroneous_nominals_capi(self):
        """ Tests that erroneous nominals returned from GetNominalsOfContinuousStates get auto-corrected. """

        # Don't enable this except during local development. It will break all logging
        # for future test runs in the same python process.
        # If other tests also has this kind of property, only enable one at the time.
        # FIXME: Find a proper way to do it, or better, switch to a testing framework which has
        # support for it (e.g. unittest with assertLogs).
        one_off_test_logging = False

        model = Dummy_FMUModelME2([], FMU_PATHS.ME2.coupled_clutches, log_level=3, _connect_dll=False)

        if one_off_test_logging:
            log_stream = StringIO()
            logging.basicConfig(stream=log_stream, level=logging.WARNING)

        # NOTE: Property 'nominal_continuous_states' is already overriden in Dummy_FMUModelME2, so just
        # call the underlying function immediately.
        xn = model._get_nominal_continuous_states()

        if one_off_test_logging:
            # Check warning is given:
            expected_msg1 = "The nominal value for clutch1.phi_rel is <0.0 which is illegal according to the " \
                        + "FMI specification. Setting the nominal to abs(-2.0)."
            expected_msg2 = "The nominal value for J4.w is 0.0 which is illegal according to the " \
                        + "FMI specification. Setting the nominal to 1.0."
            log = str(log_stream.getvalue())
            nose.tools.assert_in(expected_msg1, log)  # First warning of 6.
            nose.tools.assert_in(expected_msg2, log)  # Last warning of 6.

        # Check that values are auto-corrected:
        nose.tools.assert_almost_equal(xn[0], 2.0)  # -2.0
        nose.tools.assert_almost_equal(xn[1], 1.0)  #  0.0
        nose.tools.assert_almost_equal(xn[2], 2.0)  #  2.0
        nose.tools.assert_almost_equal(xn[3], 2.0)  # -2.0
        nose.tools.assert_almost_equal(xn[4], 1.0)  #  0.0
        nose.tools.assert_almost_equal(xn[5], 2.0)  #  2.0
        nose.tools.assert_almost_equal(xn[6], 2.0)  # -2.0
        nose.tools.assert_almost_equal(xn[7], 1.0)  #  0,0

    @testattr(stddist = True)
    def test_get_time_varying_variables(self):
        model = FMUModelME2(FMU_PATHS.ME2.coupled_clutches, _connect_dll=False)

        [r,i,b] = model.get_model_time_varying_value_references()
        [r_f, i_f, b_f] = model.get_model_time_varying_value_references(filter="*")

        assert len(r) == len(r_f)
        assert len(i) == len(i_f)
        assert len(b) == len(b_f)

        vars = model.get_variable_alias("J4.phi")
        for var in vars:
            [r,i,b] = model.get_model_time_varying_value_references(filter=var)
            assert len(r) == 1, len(r)

        [r,i,b] = model.get_model_time_varying_value_references(filter=list(vars.keys()))
        assert len(r) == 1, len(r)

    @testattr(stddist = True)
    def test_get_directional_derivative_capability(self):
        bounce = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        bounce.setup_experiment()
        bounce.initialize()

        # Bouncing ball don't have the capability, check that this is handled
        nose.tools.assert_raises(FMUException, bounce.get_directional_derivative, [1], [1], [1])

        bounce = Dummy_FMUModelCS2([], os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "bouncingBall.fmu"), _connect_dll=False)
        bounce.setup_experiment()
        bounce.initialize()

        # Bouncing ball don't have the capability, check that this is handled
        nose.tools.assert_raises(FMUException, bounce.get_directional_derivative, [1], [1], [1])

    @testattr(stddist = True)
    def test_simulation_without_initialization(self):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        opts = model.simulate_options()
        opts["initialize"] = False

        nose.tools.assert_raises(FMUException, model.simulate, options=opts)

        model = Dummy_FMUModelCS2([], os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "bouncingBall.fmu"), _connect_dll=False)
        opts = model.simulate_options()
        opts["initialize"] = False

        nose.tools.assert_raises(FMUException, model.simulate, options=opts)
    
    @testattr(stddist = True)
    def test_simulation_with_syncronization_exception_ME(self):
        """
        Verifies the allowed values for the option to synchronize simulations (ME)
        """
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        opts = model.simulate_options()
        opts["synchronize_simulation"] = "Hej"
        
        nose.tools.assert_raises(InvalidOptionException, model.simulate, options=opts)
        
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        opts = model.simulate_options()
        opts["synchronize_simulation"] = -1.0
        
        nose.tools.assert_raises(InvalidOptionException, model.simulate, options=opts)
    
    @testattr(stddist = True)
    def test_simulation_with_syncronization_exception_CS(self):
        """
        Verifies the allowed values for the option to synchronize simulations (CS)
        """
        model = Dummy_FMUModelCS2([], os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "bouncingBall.fmu"), _connect_dll=False)
        opts = model.simulate_options()
        opts["synchronize_simulation"] = "Hej"
        
        nose.tools.assert_raises(InvalidOptionException, model.simulate, options=opts)
        
        model = Dummy_FMUModelCS2([], os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "bouncingBall.fmu"), _connect_dll=False)
        opts = model.simulate_options()
        opts["synchronize_simulation"] = -1.0
        
        nose.tools.assert_raises(InvalidOptionException, model.simulate, options=opts)
        
    @testattr(stddist = True)
    def test_simulation_with_syncronization_ME(self):
        """
        Verifies that the option synchronize simulation works as intended in the most basic test for ME FMUs.
        """
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        opts = model.simulate_options()
        opts["synchronize_simulation"] = True
        
        res = model.simulate(final_time=0.1, options=opts)
        t = res.detailed_timings["computing_solution"]
        
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        opts = model.simulate_options()
        opts["synchronize_simulation"] = 0.1
        
        res = model.simulate(final_time=0.1, options=opts)
        tsyn = res.detailed_timings["computing_solution"]
        
        assert tsyn > t, "Syncronization does not work: %d, %d"%(t, tsyn)
        
    
    @testattr(stddist = True)
    def test_simulation_with_syncronization_CS(self):
        """
        Verifies that the option synchronize simulation works as intended in the most basic test for CS FMUs.
        """
        model = Dummy_FMUModelCS2([], os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "bouncingBall.fmu"), _connect_dll=False)
        opts = model.simulate_options()
        opts["synchronize_simulation"] = True
        
        res = model.simulate(final_time=0.1, options=opts)
        t = res.detailed_timings["computing_solution"]
        
        model = Dummy_FMUModelCS2([], os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "bouncingBall.fmu"), _connect_dll=False)
        opts = model.simulate_options()
        opts["synchronize_simulation"] = 0.1
        
        res = model.simulate(final_time=0.1, options=opts)
        tsyn = res.detailed_timings["computing_solution"]
        
        assert tsyn > t, "Syncronization does not work: %d, %d"%(t, tsyn)

    @testattr(stddist = True)
    def test_caching(self):
        negated_alias = FMUModelME2(os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NegatedAlias.fmu"), _connect_dll=False)

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

        negated_alias = FMUModelME2(os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NegatedAlias.fmu"), _connect_dll=False)

        assert len(negated_alias.cache) == 0 #No starting cache

        vars_6 = negated_alias.get_model_variables()
        assert id(vars_1) != id(vars_6)

    @testattr(stddist = True)
    def test_get_scalar_variable(self):
        negated_alias = FMUModelME2(os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NegatedAlias.fmu"), _connect_dll=False)

        sc_x = negated_alias.get_scalar_variable("x")

        assert sc_x.name == "x", sc_x.name
        assert sc_x.value_reference >= 0, sc_x.value_reference
        assert sc_x.type == fmi.FMI2_REAL, sc_x.type
        assert sc_x.variability == fmi.FMI2_CONTINUOUS, sc_x.variability
        assert sc_x.causality == fmi.FMI2_LOCAL, sc_x.causality
        assert sc_x.initial == fmi.FMI2_INITIAL_APPROX, sc_x.initial

        nose.tools.assert_raises(FMUException, negated_alias.get_scalar_variable, "not_existing")

    @testattr(stddist = True)
    def test_get_variable_description(self):
        model = FMUModelME2(FMU_PATHS.ME2.coupled_clutches, _connect_dll=False)
        assert model.get_variable_description("J1.phi") == "Absolute rotation angle of component"

class Test_load_fmu_only_XML:

    @testattr(stddist = True)
    def test_loading_xml_me1(self):

        model = FMUModelME1(FMU_PATHS.ME1.coupled_clutches, _connect_dll=False)

        assert model.get_name() == "CoupledClutches", model.get_name()

    @testattr(stddist = True)
    def test_loading_xml_cs1(self):

        model = FMUModelCS1(os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "CoupledClutches.fmu"), _connect_dll=False)

        assert model.get_name() == "CoupledClutches", model.get_name()

    @testattr(stddist = True)
    def test_loading_xml_me2(self):

        model = FMUModelME2(FMU_PATHS.ME2.coupled_clutches, _connect_dll=False)

        assert model.get_name() == "CoupledClutches", model.get_name()

    @testattr(stddist = True)
    def test_loading_xml_cs2(self):

        model = FMUModelCS2(os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "CoupledClutches.fmu"), _connect_dll=False)

        assert model.get_name() == "CoupledClutches", model.get_name()
