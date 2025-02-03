#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019-2024 Modelon AB
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import pytest
import os
import numpy as np
import time
import math
import re
from io import StringIO, BytesIO
from collections import OrderedDict

from pyfmi.fmi import (
    FMUException,
    FMUModelME2,
    FMI2_PARAMETER,
    FMI2_CONSTANT,
    InvalidOptionException
)
from pyfmi.common.io import (
    ResultHandler,
    ResultDymolaTextual,
    ResultDymolaBinary,
    JIOError,
    ResultSizeError,
    ResultHandlerCSV,
    ResultCSVTextual,
    ResultHandlerBinaryFile,
    ResultHandlerFile,
    Trajectory,
    get_result_handler
)
from pyfmi.common.diagnostics import (
    DIAGNOSTICS_PREFIX,
    setup_diagnostics_variables
)

import pyfmi.fmi as fmi
from pyfmi.test_util import Dummy_FMUModelME1, Dummy_FMUModelCS1, Dummy_FMUModelME2, Dummy_FMUModelCS2

file_path = os.path.dirname(os.path.abspath(__file__))

try:
    import assimulo
except ImportError:
    # XXX: Accept import failure due to conditional test execution
    pass

def _run_negated_alias(model, result_type, result_file_name=""):
    opts = model.simulate_options()
    opts["result_handling"] = result_type
    opts["result_file_name"] = result_file_name

    res = model.simulate(options=opts)

    # test that res['y'] returns a vector of the same length as the time
    # vector
    assert len(res['y']) ==len(res['time']), "Wrong size of result vector."

    x = res["x"]
    y = res["y"]

    for i in range(len(x)):
        assert x[i] == -y[i]

@pytest.mark.assimulo
class TestResultFileText_Simulation:

    def _correct_syntax_after_simulation_failure(self, result_file_name):
        simple_alias = Dummy_FMUModelME2([("x", "y")], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NegatedAlias.fmu"), _connect_dll=False)

        def f(*args, **kwargs):
            if simple_alias.time > 0.5:
                raise Exception
            return -simple_alias.continuous_states

        simple_alias.get_derivatives = f

        opts = simple_alias.simulate_options()
        opts["result_handling"] = "file"
        opts["solver"] = "ExplicitEuler"
        opts["result_file_name"]  = result_file_name

        successful_simulation = False
        try:
            res = simple_alias.simulate(options=opts)
            successful_simulation = True #The above simulation should fail...
        except Exception:
            pass

        if successful_simulation:
            raise Exception

        result = ResultDymolaTextual(result_file_name)

        x = result.get_variable_data("x").x
        y = result.get_variable_data("y").x

        assert len(x) > 2

        for i in range(len(x)):
            assert x[i] == -y[i]

    def test_correct_file_after_simulation_failure(self):
        self._correct_syntax_after_simulation_failure("NegatedAlias_result.txt")

    def test_correct_stream_after_simulation_failure(self):
        stream = StringIO("")
        self._correct_syntax_after_simulation_failure(stream)

    def test_read_all_variables_using_model_variables(self):
        simple_alias = Dummy_FMUModelME2([("x", "y")], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NegatedAlias.fmu"), _connect_dll=False)

        opts = simple_alias.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerFile(simple_alias)

        res = simple_alias.simulate(options=opts)

        for var in simple_alias.get_model_variables():
            res[var]

    def test_read_alias_derivative(self):
        simple_alias = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "Alias.fmu"), _connect_dll=False)

        opts = simple_alias.simulate_options()
        opts["result_handling"] = "file"

        res = simple_alias.simulate(options=opts)

        derx = res["der(x)"]
        dery = res["der(y)"]

        assert len(derx) > 0
        for i in range(len(derx)):
            assert derx[i] == dery[i]

    def test_no_variables(self):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "ParameterAlias.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["result_handling"] = "file"
        opts["result_file_name"] = "NoMatchingTest.txt"
        opts["filter"] = "NoMatchingVariables"

        res = model.simulate(options=opts)

        assert 1.0 == pytest.approx(res["time"][-1])

    def test_enumeration_file(self):

        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "Friction2.fmu"), _connect_dll=False)
        data_type = model.get_variable_data_type("mode")

        assert data_type == fmi.FMI2_ENUMERATION

        opts = model.simulate_options()
        opts["result_handling"] = "file"

        res = model.simulate(options=opts)
        res["mode"] #Check that the enumeration variable is in the dict, otherwise exception

class TestResultFileText:
    def _get_description(self, result_file_name):
        model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "CoupledClutches.fmu"), _connect_dll=False)
        model.initialize()

        result_writer = ResultHandlerFile(model)
        opts = model.simulate_options()
        opts["result_file_name"] = result_file_name
        result_writer.set_options(opts)
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.simulation_end()

        res = ResultDymolaTextual(result_file_name)

        assert res.description[res.get_variable_index("J1.phi")] == "Absolute rotation angle of component"

    def test_get_description_file(self):
        self._get_description('CoupledClutches_result.txt')

    def test_get_description_stream(self):
        stream = StringIO()
        self._get_description(stream)

    def test_description_not_stored(self):
        model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "CoupledClutches.fmu"), _connect_dll=False)
        model.initialize()

        opts = model.simulate_options()
        opts["result_store_variable_description"] = False

        result_writer = ResultHandlerFile(model)
        result_writer.set_options(opts)
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.simulation_end()

        res = ResultDymolaTextual('CoupledClutches_result.txt')

        assert res.description[res.get_variable_index("J1.phi")] == "", "Description is not empty, " + res.description[res.get_variable_index("J1.phi")]

    def _get_description_unicode(self, result_file_name):
        model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "Description.fmu"), _connect_dll=False)
        model.initialize()

        result_writer = ResultHandlerFile(model)
        opts = model.simulate_options()
        opts["result_file_name"] = result_file_name
        result_writer.set_options(opts)
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.simulation_end()

        res = ResultDymolaTextual(result_file_name)
        desc = res.description[res.get_variable_index("x")]

        assert desc == u"Test symbols '' ‘’"

    def _get_description_unicode_file(self):
        self._get_description_unicode('Description_result.txt')

    def _get_description_unicode_stream(self):
        stream = StringIO()
        self._get_description_unicode(stream)


    def _work_flow_me1(self, result_file_name):
        model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        model.initialize()

        bouncingBall = ResultHandlerFile(model)

        opts = model.simulate_options()
        opts["result_file_name"] = result_file_name
        bouncingBall.set_options(opts)
        bouncingBall.simulation_start()
        bouncingBall.initialize_complete()
        bouncingBall.integration_point()
        bouncingBall.simulation_end()

        res = ResultDymolaTextual(result_file_name)

        h = res.get_variable_data('h')
        derh = res.get_variable_data('der(h)')
        g = res.get_variable_data('g')

        assert h.x == pytest.approx(1.000000, abs = 1e-5)
        assert derh.x == pytest.approx(0.000000, abs = 1e-5)

    def test_work_flow_me1_file(self):
        self._work_flow_me1('bouncingBall_result.txt')

    def test_work_flow_me1_stream(self):
        stream = StringIO()
        self._work_flow_me1(stream)

    def _work_flow_me2(self, result_file_name):
        """Tests the work flow of write_header, write_point, write_finalize."""
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()

        bouncingBall = ResultHandlerFile(model)

        opts = model.simulate_options()
        opts["result_file_name"] = result_file_name
        bouncingBall.set_options(opts)
        bouncingBall.simulation_start()
        bouncingBall.initialize_complete()
        bouncingBall.integration_point()
        bouncingBall.simulation_end()

        res = ResultDymolaTextual(result_file_name)

        h = res.get_variable_data('h')
        derh = res.get_variable_data('der(h)')
        g = res.get_variable_data('g')

        assert h.x == pytest.approx(1.000000, abs = 1e-5)
        assert derh.x == pytest.approx(0.000000, abs = 1e-5)

    def test_work_flow_me2_file(self):
        self._work_flow_me2('bouncingBall_result.txt')

    def test_work_flow_me2_stream(self):
        stream = StringIO()
        self._work_flow_me2(stream)

    def test_work_flow_me2_stream2(self):
        """ Verify exception when using ResultHandlerFile with a stream that doesnt support 'seek'. """
        class A:
            def write(self):
                pass
        stream = A()
        msg = "Failed to write the result file. Option 'result_file_name' needs to be a filename or a class that supports 'write' and 'seek'."
        with pytest.raises(FMUException, match = msg):
            self._work_flow_me2(stream)

    def test_work_flow_me2_stream3(self):
        """ Verify exception when using ResultHandlerFile with a stream that doesnt support 'write'. """
        class A:
            def seek(self):
                pass
        stream = A()
        msg = "Failed to write the result file. Option 'result_file_name' needs to be a filename or a class that supports 'write' and 'seek'."
        with pytest.raises(FMUException, match = msg):
            self._work_flow_me2(stream)

    def test_constructor_invalid_stream1(self):
        """ Verify exception is raised for ResultDymolaTextual if fname argument is a stream not supporting 'readline'. """
        class A:
            def seek(self):
                pass
        stream = A()
        msg = "Given stream needs to support 'readline' and 'seek' in order to retrieve the results."
        with pytest.raises(JIOError, match = msg):
            res = ResultDymolaTextual(stream)

    def test_constructor_invalid_stream2(self):
        """ Verify exception is raised for ResultDymolaTextual if fname argument is a stream not supporting 'seek'. """
        class A:
            def readline(self):
                pass
        stream = A()
        msg = "Given stream needs to support 'readline' and 'seek' in order to retrieve the results."
        with pytest.raises(JIOError, match = msg):
            res = ResultDymolaTextual(stream)

@pytest.mark.assimulo
class TestResultMemory_Simulation:
    def test_memory_options_me1(self):
        simple_alias = Dummy_FMUModelME1([40], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NegatedAlias.fmu"), _connect_dll=False)
        _run_negated_alias(simple_alias, "memory")

    def test_memory_options_me2(self):
        simple_alias = Dummy_FMUModelME2([("x", "y")], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NegatedAlias.fmu"), _connect_dll=False)
        _run_negated_alias(simple_alias, "memory")

    def test_only_parameters(self):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "ParameterAlias.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["result_handling"] = "memory"
        opts["filter"] = "p2"

        res = model.simulate(options=opts)

        assert 3.0 == pytest.approx(res["p2"][0])
        assert not isinstance(res.initial("p2"), np.ndarray)
        assert not isinstance(res.final("p2"), np.ndarray)

    def test_no_variables(self):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "ParameterAlias.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["result_handling"] = "memory"
        opts["filter"] = "NoMatchingVariables"

        res = model.simulate(options=opts)

        assert 1.0 == pytest.approx(res["time"][-1])

    def test_enumeration_memory(self):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "Friction2.fmu"), _connect_dll=False)
        data_type = model.get_variable_data_type("mode")

        assert data_type == fmi.FMI2_ENUMERATION

        opts = model.simulate_options()
        opts["result_handling"] = "memory"

        res = model.simulate(options=opts)
        res["mode"] #Check that the enumeration variable is in the dict, otherwise exception

class TestResultMemory:
    pass

@pytest.mark.assimulo
class TestResultFileBinary_Simulation:
    def _correct_file_after_simulation_failure(self, result_file_name):
        simple_alias = Dummy_FMUModelME2([("x", "y")], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NegatedAlias.fmu"), _connect_dll=False)

        def f(*args, **kwargs):
            if simple_alias.time > 0.5:
                raise Exception
            return -simple_alias.continuous_states

        simple_alias.get_derivatives = f

        opts = simple_alias.simulate_options()
        opts["result_handling"] = "binary"
        opts["result_file_name"] = result_file_name
        opts["solver"] = "ExplicitEuler"

        successful_simulation = False
        try:
            res = simple_alias.simulate(options=opts)
            successful_simulation = True #The above simulation should fail...
        except Exception:
            pass

        if successful_simulation:
            raise Exception

        result = ResultDymolaBinary(result_file_name)

        x = result.get_variable_data("x").x
        y = result.get_variable_data("y").x

        assert len(x) > 2

        for i in range(len(x)):
            assert x[i] == -y[i]


    def test_work_flow_me2_file(self):
        self._correct_file_after_simulation_failure("NegatedAlias_result.mat")

    def test_work_flow_me2_stream(self):
        stream = BytesIO()
        self._correct_file_after_simulation_failure(stream)

    def _only_parameters(self, result_file_name):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "ParameterAlias.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerBinaryFile(model)
        opts["filter"] = "p2"
        opts["result_file_name"] = result_file_name

        res = model.simulate(options=opts)

        assert 3.0 == pytest.approx(res["p2"][0])

    def test_only_parameters_file(self):
        self._only_parameters("ParameterAlias_result.mat")

    def test_only_parameters_stream(self):
        stream = BytesIO()
        self._only_parameters(stream)

    def _no_variables(self, result_file_name):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "ParameterAlias.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerBinaryFile(model)
        opts["filter"] = "NoMatchingVariables"
        opts["result_file_name"] = result_file_name

        res = model.simulate(options=opts)

        assert 1.0 == pytest.approx(res["time"][-1])


    def test_no_variables_file(self):
        self._no_variables("ParameterAlias_result.mat")

    def test_no_variables_stream(self):
        stream = BytesIO()
        self._no_variables(stream)

    def test_read_alias_derivative(self):
        simple_alias = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "Alias.fmu"), _connect_dll=False)

        opts = simple_alias.simulate_options()
        opts["result_handling"] = "binary"

        res = simple_alias.simulate(options=opts)

        derx = res["der(x)"]
        dery = res["der(y)"]

        assert len(derx) > 0
        for i in range(len(derx)):
            assert derx[i] == dery[i]

    def test_enumeration_binary(self):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "Friction2.fmu"), _connect_dll=False)
        data_type = model.get_variable_data_type("mode")

        assert data_type == fmi.FMI2_ENUMERATION

        opts = model.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerBinaryFile(model)

        res = model.simulate(options=opts)
        res["mode"] #Check that the enumeration variable is in the dict, otherwise exception

    def test_integer_start_time(self):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "Alias.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["result_handling"] = "binary"

        #Assert that there is no exception when reloading the file
        res = model.simulate(start_time=0, options=opts)

    def test_read_all_variables_using_model_variables(self):
        simple_alias = Dummy_FMUModelME2([("x", "y")], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NegatedAlias.fmu"), _connect_dll=False)

        opts = simple_alias.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerBinaryFile(simple_alias)

        res = simple_alias.simulate(options=opts)

        for var in simple_alias.get_model_variables():
            res[var]

    def test_variable_alias_custom_handler(self):
        simple_alias = Dummy_FMUModelME2([("x", "y")], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NegatedAlias.fmu"), _connect_dll=False)

        opts = simple_alias.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerBinaryFile(simple_alias)

        res = simple_alias.simulate(options=opts)

        # test that res['y'] returns a vector of the same length as the time
        # vector
        assert len(res['y']) ==len(res['time']), "Wrong size of result vector."

        x = res["x"]
        y = res["y"]

        for i in range(len(x)):
            assert x[i] == -y[i]

    def test_binary_options_me1(self):
        simple_alias = Dummy_FMUModelME1([40], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NegatedAlias.fmu"), _connect_dll=False)
        _run_negated_alias(simple_alias, "binary")

    def test_binary_options_me2(self):
        simple_alias = Dummy_FMUModelME2([("x", "y")], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NegatedAlias.fmu"), _connect_dll=False)
        _run_negated_alias(simple_alias, "binary")

    def test_binary_options_me1_stream(self):
        simple_alias = Dummy_FMUModelME1([40], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NegatedAlias.fmu"), _connect_dll=False)
        stream = BytesIO()
        _run_negated_alias(simple_alias, "binary", stream)

    def test_binary_options_me2_stream(self):
        simple_alias = Dummy_FMUModelME2([("x", "y")], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NegatedAlias.fmu"), _connect_dll=False)
        stream = BytesIO()
        _run_negated_alias(simple_alias, "binary", stream)

class TestResultFileBinary:
    def _get_description_unicode(self, result_file_name):
        model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "Description.fmu"), _connect_dll=False)
        model.initialize()

        result_writer = ResultHandlerBinaryFile(model)
        opts = model.simulate_options()
        opts["result_file_name"] = result_file_name
        result_writer.set_options(opts)
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.simulation_end()

        res = ResultDymolaBinary(result_file_name)

        desc = res.description[res.get_variable_index("x")]
        #This handling should in the future be nativly handled by the IO module
        desc = desc.encode("latin_1", "replace").decode("utf-8", "replace")

        assert desc == u"Test symbols '' ‘’"

    def test_get_description_unicode_file(self):
        self._get_description_unicode('Description_result.mat')

    def test_get_description_unicode_stream(self):
        stream = BytesIO()
        self._get_description_unicode(stream)

    def test_get_description(self):
        model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "CoupledClutches.fmu"), _connect_dll=False)
        model.initialize()

        result_writer = ResultHandlerBinaryFile(model)
        result_writer.set_options(model.simulate_options())
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.simulation_end()

        res = ResultDymolaBinary('CoupledClutches_result.mat')

        assert res.description[res.get_variable_index("J1.phi")] == "Absolute rotation angle of component"

    def test_modified_result_file_data_diagnostics(self):
        """Verify that computed diagnostics can be retrieved from an updated result file"""
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "CoupledClutches.fmu"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()

        opts = model.simulate_options()
        opts["dynamic_diagnostics"] = True

        diagnostics_params = OrderedDict([('@Diagnostics.solver.solver_name.CVode', (1.0, 'Chosen solver.')), ('@Diagnostics.solver.relative_tolerance', (0.0001, 'Relative solver tolerance.')), ('@Diagnostics.solver.absolute_tolerance.clutch1.phi_rel', (1.0000000000000002e-10, 'Absolute solver tolerance for clutch1.phi_rel.')), ('@Diagnostics.solver.absolute_tolerance.clutch1.w_rel', (1.0000000000000002e-06, 'Absolute solver tolerance for clutch1.w_rel.')), ('@Diagnostics.solver.absolute_tolerance.clutch2.phi_rel', (1.0000000000000002e-10, 'Absolute solver tolerance for clutch2.phi_rel.')), ('@Diagnostics.solver.absolute_tolerance.clutch2.w_rel', (1.0000000000000002e-06, 'Absolute solver tolerance for clutch2.w_rel.')), ('@Diagnostics.solver.absolute_tolerance.clutch3.phi_rel', (1.0000000000000002e-10, 'Absolute solver tolerance for clutch3.phi_rel.')), ('@Diagnostics.solver.absolute_tolerance.clutch3.w_rel', (1.0000000000000002e-06, 'Absolute solver tolerance for clutch3.w_rel.')), ('@Diagnostics.solver.absolute_tolerance.J4.phi', (1.0000000000000002e-06, 'Absolute solver tolerance for J4.phi.')), ('@Diagnostics.solver.absolute_tolerance.J4.w', (1.0000000000000002e-06, 'Absolute solver tolerance for J4.w.'))])
        diagnostics_vars   = OrderedDict([('@Diagnostics.step_time', (0.0, 'Step time')), ('@Diagnostics.cpu_time_per_step', (0, 'CPU time per step.')), ('@Diagnostics.solver.solver_order', (0.0, 'Solver order for CVode used in each time step')), ('@Diagnostics.state_errors.clutch1.phi_rel', (0.0, 'State error for clutch1.phi_rel.')), ('@Diagnostics.state_errors.clutch1.w_rel', (0.0, 'State error for clutch1.w_rel.')), ('@Diagnostics.state_errors.clutch2.phi_rel', (0.0, 'State error for clutch2.phi_rel.')), ('@Diagnostics.state_errors.clutch2.w_rel', (0.0, 'State error for clutch2.w_rel.')), ('@Diagnostics.state_errors.clutch3.phi_rel', (0.0, 'State error for clutch3.phi_rel.')), ('@Diagnostics.state_errors.clutch3.w_rel', (0.0, 'State error for clutch3.w_rel.')), ('@Diagnostics.state_errors.J4.phi', (0.0, 'State error for J4.phi.')), ('@Diagnostics.state_errors.J4.w', (0.0, 'State error for J4.w.')), ('@Diagnostics.event_data.event_info.indicator_1', (20.0, 'Value for event indicator 1.')), ('@Diagnostics.event_data.event_info.indicator_2', (0.99999999, 'Value for event indicator 2.')), ('@Diagnostics.event_data.event_info.indicator_3', (0.99999999, 'Value for event indicator 3.')), ('@Diagnostics.event_data.event_info.indicator_4', (-10000000010.0, 'Value for event indicator 4.')), ('@Diagnostics.event_data.event_info.indicator_5', (0.99999999, 'Value for event indicator 5.')), ('@Diagnostics.event_data.event_info.indicator_6', (1.00000001, 'Value for event indicator 6.')), ('@Diagnostics.event_data.event_info.indicator_7', (1.00000001, 'Value for event indicator 7.')), ('@Diagnostics.event_data.event_info.indicator_8', (1.00000001, 'Value for event indicator 8.')), ('@Diagnostics.event_data.event_info.indicator_9', (1.0, 'Value for event indicator 9.')), ('@Diagnostics.event_data.event_info.indicator_10', (0.99999999, 'Value for event indicator 10.')), ('@Diagnostics.event_data.event_info.indicator_11', (-10.0, 'Value for event indicator 11.')), ('@Diagnostics.event_data.event_info.indicator_12', (-1e-08, 'Value for event indicator 12.')), ('@Diagnostics.event_data.event_info.indicator_13', (0.99999999, 'Value for event indicator 13.')), ('@Diagnostics.event_data.event_info.indicator_14', (0.99999999, 'Value for event indicator 14.')), ('@Diagnostics.event_data.event_info.indicator_15', (0.99999999, 'Value for event indicator 15.')), ('@Diagnostics.event_data.event_info.indicator_16', (0.99999999, 'Value for event indicator 16.')), ('@Diagnostics.event_data.event_info.indicator_17', (1.0, 'Value for event indicator 17.')), ('@Diagnostics.event_data.event_info.indicator_18', (1.0, 'Value for event indicator 18.')), ('@Diagnostics.event_data.event_info.indicator_19', (1.00000001, 'Value for event indicator 19.')), ('@Diagnostics.event_data.event_info.indicator_20', (1.00000001, 'Value for event indicator 20.')), ('@Diagnostics.event_data.event_info.indicator_21', (0.99999999, 'Value for event indicator 21.')), ('@Diagnostics.event_data.event_info.indicator_22', (1.00000001, 'Value for event indicator 22.')), ('@Diagnostics.event_data.event_info.indicator_23', (-1e-08, 'Value for event indicator 23.')), ('@Diagnostics.event_data.event_info.indicator_24', (0.99999999, 'Value for event indicator 24.')), ('@Diagnostics.event_data.event_info.indicator_25', (0.99999999, 'Value for event indicator 25.')), ('@Diagnostics.event_data.event_info.indicator_26', (0.99999999, 'Value for event indicator 26.')), ('@Diagnostics.event_data.event_info.indicator_27', (0.99999999, 'Value for event indicator 27.')), ('@Diagnostics.event_data.event_info.indicator_28', (1.00000001, 'Value for event indicator 28.')), ('@Diagnostics.event_data.event_info.indicator_29', (1.00000001, 'Value for event indicator 29.')), ('@Diagnostics.event_data.event_info.indicator_30', (1.00000001, 'Value for event indicator 30.')), ('@Diagnostics.event_data.event_info.indicator_31', (1.00000001, 'Value for event indicator 31.')), ('@Diagnostics.event_data.event_info.indicator_32', (0.99999999, 'Value for event indicator 32.')), ('@Diagnostics.event_data.event_info.indicator_33', (1.00000001, 'Value for event indicator 33.')), ('@Diagnostics.event_data.event_info.state_event_info.index_1', (0.0, 'Zero crossing indicator for event indicator 1')), ('@Diagnostics.event_data.event_info.state_event_info.index_2', (0.0, 'Zero crossing indicator for event indicator 2')), ('@Diagnostics.event_data.event_info.state_event_info.index_3', (0.0, 'Zero crossing indicator for event indicator 3')), ('@Diagnostics.event_data.event_info.state_event_info.index_4', (0.0, 'Zero crossing indicator for event indicator 4')), ('@Diagnostics.event_data.event_info.state_event_info.index_5', (0.0, 'Zero crossing indicator for event indicator 5')), ('@Diagnostics.event_data.event_info.state_event_info.index_6', (0.0, 'Zero crossing indicator for event indicator 6')), ('@Diagnostics.event_data.event_info.state_event_info.index_7', (0.0, 'Zero crossing indicator for event indicator 7')), ('@Diagnostics.event_data.event_info.state_event_info.index_8', (0.0, 'Zero crossing indicator for event indicator 8')), ('@Diagnostics.event_data.event_info.state_event_info.index_9', (0.0, 'Zero crossing indicator for event indicator 9')), ('@Diagnostics.event_data.event_info.state_event_info.index_10', (0.0, 'Zero crossing indicator for event indicator 10')), ('@Diagnostics.event_data.event_info.state_event_info.index_11', (0.0, 'Zero crossing indicator for event indicator 11')), ('@Diagnostics.event_data.event_info.state_event_info.index_12', (0.0, 'Zero crossing indicator for event indicator 12')), ('@Diagnostics.event_data.event_info.state_event_info.index_13', (0.0, 'Zero crossing indicator for event indicator 13')), ('@Diagnostics.event_data.event_info.state_event_info.index_14', (0.0, 'Zero crossing indicator for event indicator 14')), ('@Diagnostics.event_data.event_info.state_event_info.index_15', (0.0, 'Zero crossing indicator for event indicator 15')), ('@Diagnostics.event_data.event_info.state_event_info.index_16', (0.0, 'Zero crossing indicator for event indicator 16')), ('@Diagnostics.event_data.event_info.state_event_info.index_17', (0.0, 'Zero crossing indicator for event indicator 17')), ('@Diagnostics.event_data.event_info.state_event_info.index_18', (0.0, 'Zero crossing indicator for event indicator 18')), ('@Diagnostics.event_data.event_info.state_event_info.index_19', (0.0, 'Zero crossing indicator for event indicator 19')), ('@Diagnostics.event_data.event_info.state_event_info.index_20', (0.0, 'Zero crossing indicator for event indicator 20')), ('@Diagnostics.event_data.event_info.state_event_info.index_21', (0.0, 'Zero crossing indicator for event indicator 21')), ('@Diagnostics.event_data.event_info.state_event_info.index_22', (0.0, 'Zero crossing indicator for event indicator 22')), ('@Diagnostics.event_data.event_info.state_event_info.index_23', (0.0, 'Zero crossing indicator for event indicator 23')), ('@Diagnostics.event_data.event_info.state_event_info.index_24', (0.0, 'Zero crossing indicator for event indicator 24')), ('@Diagnostics.event_data.event_info.state_event_info.index_25', (0.0, 'Zero crossing indicator for event indicator 25')), ('@Diagnostics.event_data.event_info.state_event_info.index_26', (0.0, 'Zero crossing indicator for event indicator 26')), ('@Diagnostics.event_data.event_info.state_event_info.index_27', (0.0, 'Zero crossing indicator for event indicator 27')), ('@Diagnostics.event_data.event_info.state_event_info.index_28', (0.0, 'Zero crossing indicator for event indicator 28')), ('@Diagnostics.event_data.event_info.state_event_info.index_29', (0.0, 'Zero crossing indicator for event indicator 29')), ('@Diagnostics.event_data.event_info.state_event_info.index_30', (0.0, 'Zero crossing indicator for event indicator 30')), ('@Diagnostics.event_data.event_info.state_event_info.index_31', (0.0, 'Zero crossing indicator for event indicator 31')), ('@Diagnostics.event_data.event_info.state_event_info.index_32', (0.0, 'Zero crossing indicator for event indicator 32')), ('@Diagnostics.event_data.event_info.state_event_info.index_33', (0.0, 'Zero crossing indicator for event indicator 33')), ('@Diagnostics.event_data.event_info.event_type', (-1, 'No event=-1, state event=0, time event=1'))])
        diag_data = np.array([ 1.48162531e+00,  8.84345965e-04,  3.00000000e+00,  2.24464314e-04,
                              3.39935472e-02,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00,  7.67600547e-18,  2.14521388e-15, -5.73952762e+00,
                              9.99999990e-01,  9.99999990e-01,  9.99999990e-01,  9.99999990e-01,
                              1.00000000e+00,  1.00000000e+00,  1.00000001e+00,  1.00000000e+00,
                              9.99999990e-01,  1.00000000e+00,  2.00000000e+01, -1.10000000e+01,
                              9.99999990e-01,  9.99999990e-01,  9.99999990e-01,  1.10000000e+01,
                              1.00000001e+00,  1.00000001e+00,  1.00000001e+00,  9.99999990e-01,
                              1.00000001e+00,  2.00000000e+01, -1.10000000e+01,  9.99999990e-01,
                              9.99999990e-01,  9.99999990e-01,  1.10000000e+01,  1.00000001e+00,
                              1.00000001e+00,  1.00000001e+00,  9.99999990e-01,  1.00000001e+00,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00, -1.00000000e+00])

        result_writer = ResultHandlerBinaryFile(model)
        result_writer.set_options(opts)
        result_writer.simulation_start(diagnostics_params, diagnostics_vars)
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.diagnostics_point(diag_data)

        res = ResultDymolaBinary('CoupledClutches_result.mat', allow_file_updates=True)

        assert len(res.get_variable_data("@Diagnostics.state_errors.clutch1.w_rel").x) == 2, res.get_variable_data("@Diagnostics.state_errors.clutch1.w_rel").x

        time.sleep(0.1)
        result_writer.integration_point()
        result_writer.diagnostics_point(diag_data)
        result_writer.diagnostics_point(diag_data)
        result_writer.simulation_end()

        assert len(res.get_variable_data("@Diagnostics.state_errors.clutch2.w_rel").x) == 4, res.get_variable_data("@Diagnostics.state_errors.clutch2.w_rel").x

    def test_modified_result_file_data_diagnostics_steps(self):
        """Verify that diagnostics can be retrieved from an updated result file"""
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "CoupledClutches.fmu"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()

        opts = model.simulate_options()
        opts["dynamic_diagnostics"] = True

        diagnostics_params = OrderedDict([('@Diagnostics.solver.solver_name.CVode', (1.0, 'Chosen solver.')), ('@Diagnostics.solver.relative_tolerance', (0.0001, 'Relative solver tolerance.')), ('@Diagnostics.solver.absolute_tolerance.clutch1.phi_rel', (1.0000000000000002e-10, 'Absolute solver tolerance for clutch1.phi_rel.')), ('@Diagnostics.solver.absolute_tolerance.clutch1.w_rel', (1.0000000000000002e-06, 'Absolute solver tolerance for clutch1.w_rel.')), ('@Diagnostics.solver.absolute_tolerance.clutch2.phi_rel', (1.0000000000000002e-10, 'Absolute solver tolerance for clutch2.phi_rel.')), ('@Diagnostics.solver.absolute_tolerance.clutch2.w_rel', (1.0000000000000002e-06, 'Absolute solver tolerance for clutch2.w_rel.')), ('@Diagnostics.solver.absolute_tolerance.clutch3.phi_rel', (1.0000000000000002e-10, 'Absolute solver tolerance for clutch3.phi_rel.')), ('@Diagnostics.solver.absolute_tolerance.clutch3.w_rel', (1.0000000000000002e-06, 'Absolute solver tolerance for clutch3.w_rel.')), ('@Diagnostics.solver.absolute_tolerance.J4.phi', (1.0000000000000002e-06, 'Absolute solver tolerance for J4.phi.')), ('@Diagnostics.solver.absolute_tolerance.J4.w', (1.0000000000000002e-06, 'Absolute solver tolerance for J4.w.'))])
        diagnostics_vars   = OrderedDict([('@Diagnostics.step_time', (0.0, 'Step time')), ('@Diagnostics.cpu_time_per_step', (0, 'CPU time per step.')), ('@Diagnostics.solver.solver_order', (0.0, 'Solver order for CVode used in each time step')), ('@Diagnostics.state_errors.clutch1.phi_rel', (0.0, 'State error for clutch1.phi_rel.')), ('@Diagnostics.state_errors.clutch1.w_rel', (0.0, 'State error for clutch1.w_rel.')), ('@Diagnostics.state_errors.clutch2.phi_rel', (0.0, 'State error for clutch2.phi_rel.')), ('@Diagnostics.state_errors.clutch2.w_rel', (0.0, 'State error for clutch2.w_rel.')), ('@Diagnostics.state_errors.clutch3.phi_rel', (0.0, 'State error for clutch3.phi_rel.')), ('@Diagnostics.state_errors.clutch3.w_rel', (0.0, 'State error for clutch3.w_rel.')), ('@Diagnostics.state_errors.J4.phi', (0.0, 'State error for J4.phi.')), ('@Diagnostics.state_errors.J4.w', (0.0, 'State error for J4.w.')), ('@Diagnostics.event_data.event_info.indicator_1', (20.0, 'Value for event indicator 1.')), ('@Diagnostics.event_data.event_info.indicator_2', (0.99999999, 'Value for event indicator 2.')), ('@Diagnostics.event_data.event_info.indicator_3', (0.99999999, 'Value for event indicator 3.')), ('@Diagnostics.event_data.event_info.indicator_4', (-10000000010.0, 'Value for event indicator 4.')), ('@Diagnostics.event_data.event_info.indicator_5', (0.99999999, 'Value for event indicator 5.')), ('@Diagnostics.event_data.event_info.indicator_6', (1.00000001, 'Value for event indicator 6.')), ('@Diagnostics.event_data.event_info.indicator_7', (1.00000001, 'Value for event indicator 7.')), ('@Diagnostics.event_data.event_info.indicator_8', (1.00000001, 'Value for event indicator 8.')), ('@Diagnostics.event_data.event_info.indicator_9', (1.0, 'Value for event indicator 9.')), ('@Diagnostics.event_data.event_info.indicator_10', (0.99999999, 'Value for event indicator 10.')), ('@Diagnostics.event_data.event_info.indicator_11', (-10.0, 'Value for event indicator 11.')), ('@Diagnostics.event_data.event_info.indicator_12', (-1e-08, 'Value for event indicator 12.')), ('@Diagnostics.event_data.event_info.indicator_13', (0.99999999, 'Value for event indicator 13.')), ('@Diagnostics.event_data.event_info.indicator_14', (0.99999999, 'Value for event indicator 14.')), ('@Diagnostics.event_data.event_info.indicator_15', (0.99999999, 'Value for event indicator 15.')), ('@Diagnostics.event_data.event_info.indicator_16', (0.99999999, 'Value for event indicator 16.')), ('@Diagnostics.event_data.event_info.indicator_17', (1.0, 'Value for event indicator 17.')), ('@Diagnostics.event_data.event_info.indicator_18', (1.0, 'Value for event indicator 18.')), ('@Diagnostics.event_data.event_info.indicator_19', (1.00000001, 'Value for event indicator 19.')), ('@Diagnostics.event_data.event_info.indicator_20', (1.00000001, 'Value for event indicator 20.')), ('@Diagnostics.event_data.event_info.indicator_21', (0.99999999, 'Value for event indicator 21.')), ('@Diagnostics.event_data.event_info.indicator_22', (1.00000001, 'Value for event indicator 22.')), ('@Diagnostics.event_data.event_info.indicator_23', (-1e-08, 'Value for event indicator 23.')), ('@Diagnostics.event_data.event_info.indicator_24', (0.99999999, 'Value for event indicator 24.')), ('@Diagnostics.event_data.event_info.indicator_25', (0.99999999, 'Value for event indicator 25.')), ('@Diagnostics.event_data.event_info.indicator_26', (0.99999999, 'Value for event indicator 26.')), ('@Diagnostics.event_data.event_info.indicator_27', (0.99999999, 'Value for event indicator 27.')), ('@Diagnostics.event_data.event_info.indicator_28', (1.00000001, 'Value for event indicator 28.')), ('@Diagnostics.event_data.event_info.indicator_29', (1.00000001, 'Value for event indicator 29.')), ('@Diagnostics.event_data.event_info.indicator_30', (1.00000001, 'Value for event indicator 30.')), ('@Diagnostics.event_data.event_info.indicator_31', (1.00000001, 'Value for event indicator 31.')), ('@Diagnostics.event_data.event_info.indicator_32', (0.99999999, 'Value for event indicator 32.')), ('@Diagnostics.event_data.event_info.indicator_33', (1.00000001, 'Value for event indicator 33.')), ('@Diagnostics.event_data.event_info.state_event_info.index_1', (0.0, 'Zero crossing indicator for event indicator 1')), ('@Diagnostics.event_data.event_info.state_event_info.index_2', (0.0, 'Zero crossing indicator for event indicator 2')), ('@Diagnostics.event_data.event_info.state_event_info.index_3', (0.0, 'Zero crossing indicator for event indicator 3')), ('@Diagnostics.event_data.event_info.state_event_info.index_4', (0.0, 'Zero crossing indicator for event indicator 4')), ('@Diagnostics.event_data.event_info.state_event_info.index_5', (0.0, 'Zero crossing indicator for event indicator 5')), ('@Diagnostics.event_data.event_info.state_event_info.index_6', (0.0, 'Zero crossing indicator for event indicator 6')), ('@Diagnostics.event_data.event_info.state_event_info.index_7', (0.0, 'Zero crossing indicator for event indicator 7')), ('@Diagnostics.event_data.event_info.state_event_info.index_8', (0.0, 'Zero crossing indicator for event indicator 8')), ('@Diagnostics.event_data.event_info.state_event_info.index_9', (0.0, 'Zero crossing indicator for event indicator 9')), ('@Diagnostics.event_data.event_info.state_event_info.index_10', (0.0, 'Zero crossing indicator for event indicator 10')), ('@Diagnostics.event_data.event_info.state_event_info.index_11', (0.0, 'Zero crossing indicator for event indicator 11')), ('@Diagnostics.event_data.event_info.state_event_info.index_12', (0.0, 'Zero crossing indicator for event indicator 12')), ('@Diagnostics.event_data.event_info.state_event_info.index_13', (0.0, 'Zero crossing indicator for event indicator 13')), ('@Diagnostics.event_data.event_info.state_event_info.index_14', (0.0, 'Zero crossing indicator for event indicator 14')), ('@Diagnostics.event_data.event_info.state_event_info.index_15', (0.0, 'Zero crossing indicator for event indicator 15')), ('@Diagnostics.event_data.event_info.state_event_info.index_16', (0.0, 'Zero crossing indicator for event indicator 16')), ('@Diagnostics.event_data.event_info.state_event_info.index_17', (0.0, 'Zero crossing indicator for event indicator 17')), ('@Diagnostics.event_data.event_info.state_event_info.index_18', (0.0, 'Zero crossing indicator for event indicator 18')), ('@Diagnostics.event_data.event_info.state_event_info.index_19', (0.0, 'Zero crossing indicator for event indicator 19')), ('@Diagnostics.event_data.event_info.state_event_info.index_20', (0.0, 'Zero crossing indicator for event indicator 20')), ('@Diagnostics.event_data.event_info.state_event_info.index_21', (0.0, 'Zero crossing indicator for event indicator 21')), ('@Diagnostics.event_data.event_info.state_event_info.index_22', (0.0, 'Zero crossing indicator for event indicator 22')), ('@Diagnostics.event_data.event_info.state_event_info.index_23', (0.0, 'Zero crossing indicator for event indicator 23')), ('@Diagnostics.event_data.event_info.state_event_info.index_24', (0.0, 'Zero crossing indicator for event indicator 24')), ('@Diagnostics.event_data.event_info.state_event_info.index_25', (0.0, 'Zero crossing indicator for event indicator 25')), ('@Diagnostics.event_data.event_info.state_event_info.index_26', (0.0, 'Zero crossing indicator for event indicator 26')), ('@Diagnostics.event_data.event_info.state_event_info.index_27', (0.0, 'Zero crossing indicator for event indicator 27')), ('@Diagnostics.event_data.event_info.state_event_info.index_28', (0.0, 'Zero crossing indicator for event indicator 28')), ('@Diagnostics.event_data.event_info.state_event_info.index_29', (0.0, 'Zero crossing indicator for event indicator 29')), ('@Diagnostics.event_data.event_info.state_event_info.index_30', (0.0, 'Zero crossing indicator for event indicator 30')), ('@Diagnostics.event_data.event_info.state_event_info.index_31', (0.0, 'Zero crossing indicator for event indicator 31')), ('@Diagnostics.event_data.event_info.state_event_info.index_32', (0.0, 'Zero crossing indicator for event indicator 32')), ('@Diagnostics.event_data.event_info.state_event_info.index_33', (0.0, 'Zero crossing indicator for event indicator 33')), ('@Diagnostics.event_data.event_info.event_type', (-1, 'No event=-1, state event=0, time event=1'))])
        diag_data = np.array([ 1.48162531e+00,  8.84345965e-04,  3.00000000e+00,  2.24464314e-04,
                              3.39935472e-02,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00,  7.67600547e-18,  2.14521388e-15, -5.73952762e+00,
                              9.99999990e-01,  9.99999990e-01,  9.99999990e-01,  9.99999990e-01,
                              1.00000000e+00,  1.00000000e+00,  1.00000001e+00,  1.00000000e+00,
                              9.99999990e-01,  1.00000000e+00,  2.00000000e+01, -1.10000000e+01,
                              9.99999990e-01,  9.99999990e-01,  9.99999990e-01,  1.10000000e+01,
                              1.00000001e+00,  1.00000001e+00,  1.00000001e+00,  9.99999990e-01,
                              1.00000001e+00,  2.00000000e+01, -1.10000000e+01,  9.99999990e-01,
                              9.99999990e-01,  9.99999990e-01,  1.10000000e+01,  1.00000001e+00,
                              1.00000001e+00,  1.00000001e+00,  9.99999990e-01,  1.00000001e+00,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                              0.00000000e+00, -1.00000000e+00])

        result_writer = ResultHandlerBinaryFile(model)
        result_writer.set_options(opts)
        result_writer.simulation_start(diagnostics_params, diagnostics_vars)
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.diagnostics_point(diag_data)

        res = ResultDymolaBinary('CoupledClutches_result.mat', allow_file_updates=True)

        assert len(res.get_variable_data("@Diagnostics.nbr_steps").x) == 2, res.get_variable_data("@Diagnostics.nbr_steps").x

        time.sleep(0.1)
        result_writer.integration_point()
        result_writer.diagnostics_point(diag_data)
        result_writer.diagnostics_point(diag_data)
        result_writer.simulation_end()

        assert len(res.get_variable_data("@Diagnostics.nbr_steps").x) == 4, res.get_variable_data("@Diagnostics.nbr_steps").x

    def test_modified_result_file_data_2(self):
        """Verify that continuous trajectories are updated when retrieved from a result file"""
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "CoupledClutches.fmu"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()

        result_writer = ResultHandlerBinaryFile(model)
        result_writer.set_options(model.simulate_options())
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()

        res = ResultDymolaBinary('CoupledClutches_result.mat', allow_file_updates=True)

        assert len(res.get_variable_data("J1.phi").x) == 1, res.get_variable_data("J1.phi").x

        time.sleep(0.1)
        result_writer.integration_point()
        result_writer.simulation_end()

        assert len(res.get_variable_data("J1.phi").x) == 2, res.get_variable_data("J1.phi").x

    def test_modified_result_file_data_2_different(self):
        """Verify that (different) continuous trajectories are updated when retrieved from a result file"""
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "CoupledClutches.fmu"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()

        result_writer = ResultHandlerBinaryFile(model)
        result_writer.set_options(model.simulate_options())
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()

        res = ResultDymolaBinary('CoupledClutches_result.mat', allow_file_updates=True)

        assert len(res.get_variable_data("J1.phi").x) == 1, res.get_variable_data("J1.phi").x

        time.sleep(0.1)
        result_writer.integration_point()
        result_writer.simulation_end()

        assert len(res.get_variable_data("J2.phi").x) == 2, res.get_variable_data("J2.phi").x

    def test_modified_result_file_data_1(self):
        """Verify that (different) constants/parameters can be retrieved from an updated result file"""
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "CoupledClutches.fmu"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()

        result_writer = ResultHandlerBinaryFile(model)
        result_writer.set_options(model.simulate_options())
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()

        res = ResultDymolaBinary('CoupledClutches_result.mat', allow_file_updates=True)

        #Assert that no exception is raised
        res.get_variable_data("J1.J")

        time.sleep(0.1)
        result_writer.integration_point()
        result_writer.simulation_end()

        #Assert that no exception is raised
        res.get_variable_data("J2.J")

    def test_modified_result_file_data_1_delayed(self):
        """Verify that constants/parameters can be retrieved from an updated result file"""
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "CoupledClutches.fmu"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()

        result_writer = ResultHandlerBinaryFile(model)
        result_writer.set_options(model.simulate_options())
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()

        res = ResultDymolaBinary('CoupledClutches_result.mat', allow_file_updates=True)

        result_writer.integration_point()
        result_writer.simulation_end()

        #Assert that no exception is raised
        res.get_variable_data("J2.J")

    def test_modified_result_file_time(self):
        """Verify that 'time' can be retrieved from an updated result file"""
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "CoupledClutches.fmu"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()

        result_writer = ResultHandlerBinaryFile(model)
        result_writer.set_options(model.simulate_options())
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()

        res = ResultDymolaBinary('CoupledClutches_result.mat', allow_file_updates=True)

        res.get_variable_data("time")

        result_writer.integration_point()
        result_writer.simulation_end()

        res.get_variable_data("time")

    def test_description_not_stored(self):
        model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "CoupledClutches.fmu"), _connect_dll=False)
        model.initialize()

        opts = model.simulate_options()
        opts["result_store_variable_description"] = False

        result_writer = ResultHandlerBinaryFile(model)
        result_writer.set_options(opts)
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.simulation_end()

        res = ResultDymolaBinary('CoupledClutches_result.mat')

        assert res.description[res.get_variable_index("J1.phi")] == "", "Description is not empty, " + res.description[res.get_variable_index("J1.phi")]

    def test_overwriting_results(self):
        model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "CoupledClutches.fmu"), _connect_dll=False)
        model.initialize()

        opts = model.simulate_options()
        opts["result_store_variable_description"] = False

        result_writer = ResultHandlerBinaryFile(model)
        result_writer.set_options(opts)
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.simulation_end()

        res = ResultDymolaBinary('CoupledClutches_result.mat')

        time.sleep(1)

        result_writer = ResultHandlerBinaryFile(model)
        result_writer.set_options(opts)
        result_writer.simulation_start()
        result_writer.initialize_complete()
        result_writer.integration_point()
        result_writer.integration_point()
        result_writer.simulation_end()

        with pytest.raises(JIOError):
            res.get_variable_data("J1.phi")

    def test_read_all_variables(self):
        res = ResultDymolaBinary(os.path.join(file_path, "files", "Results", "DoublePendulum.mat"))

        assert len(res.name) == 1097, "Incorrect number of variables found, should be 1097"

        for var in res.name:
            res.get_variable_data(var)

    def test_data_matrix_delayed_loading(self):
        res = ResultDymolaBinary(os.path.join(file_path, "files", "Results", "DoublePendulum.mat"), delayed_trajectory_loading=True)

        data_matrix = res.get_data_matrix()

        [nbr_continuous_variables, nbr_points] = data_matrix.shape

        assert nbr_continuous_variables == 68, "Number of variables is incorrect, should be 68"
        assert nbr_points == 502, "Number of points is incorrect, should be 502"

    def test_data_matrix_loading(self):
        res = ResultDymolaBinary(os.path.join(file_path, "files", "Results", "DoublePendulum.mat"), delayed_trajectory_loading=False)

        data_matrix = res.get_data_matrix()

        [nbr_continuous_variables, nbr_points] = data_matrix.shape

        assert nbr_continuous_variables == 68, "Number of variables is incorrect, should be 68"
        assert nbr_points == 502, "Number of points is incorrect, should be 502"

    def test_read_all_variables_from_stream(self):

        with open(os.path.join(file_path, "files", "Results", "DoublePendulum.mat"), "rb") as f:
            res = ResultDymolaBinary(f)

            assert len(res.name) == 1097, "Incorrect number of variables found, should be 1097"

            for var in res.name:
                res.get_variable_data(var)

    def test_compare_all_variables_from_stream(self):
        res_file = ResultDymolaBinary(os.path.join(file_path, "files", "Results", "DoublePendulum.mat"))

        assert len(res_file.name) == 1097, "Incorrect number of variables found, should be 1097"

        with open(os.path.join(file_path, "files", "Results", "DoublePendulum.mat"), "rb") as f:
            res_stream = ResultDymolaBinary(f)
            assert len(res_stream.name) == 1097, "Incorrect number of variables found, should be 1097"

            for var in res_file.name:
                x_file   = res_file.get_variable_data(var)
                x_stream = res_stream.get_variable_data(var)

                np.testing.assert_array_equal(x_file.x, x_stream.x, err_msg="Mismatch in array values for var=%s"%var)

    def test_on_demand_loading_32_bits(self):
        res_demand = ResultDymolaBinary(os.path.join(file_path, "files", "Results", "DoublePendulum.mat"))
        res_all = ResultDymolaBinary(os.path.join(file_path, "files", "Results", "DoublePendulum.mat"))
        t_demand = res_demand.get_variable_data('time').x
        t_all = res_all.get_variable_data('time').x
        np.testing.assert_array_equal(t_demand, t_all, "On demand loaded result and all loaded does not contain equal result.")

    def test_work_flow_me1(self):
        model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        model.initialize()

        bouncingBall = ResultHandlerBinaryFile(model)

        bouncingBall.set_options(model.simulate_options())
        bouncingBall.simulation_start()
        bouncingBall.initialize_complete()
        bouncingBall.integration_point()
        bouncingBall.simulation_end()

        res = ResultDymolaBinary('bouncingBall_result.mat')

        h = res.get_variable_data('h')
        derh = res.get_variable_data('der(h)')
        g = res.get_variable_data('g')

        assert h.x == pytest.approx(1.000000, abs = 1e-5)
        assert derh.x == pytest.approx(0.000000, abs = 1e-5)

    def test_many_variables_long_descriptions(self):
        """
        Tests that large FMUs with lots of variables and huge length of descriptions gives
        a proper exception instead of a segfault. The problem occurs around 100k variables
        with 20k characters in the longest description.
        """
        model = FMUModelME2(os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "Large.fmu"), _connect_dll=False)

        res = ResultHandlerBinaryFile(model)

        res.set_options(model.simulate_options())
        with pytest.raises(FMUException):
            res.simulation_start()

    def test_work_flow_me2(self):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()

        bouncingBall = ResultHandlerBinaryFile(model)

        bouncingBall.set_options(model.simulate_options())
        bouncingBall.simulation_start()
        bouncingBall.initialize_complete()
        bouncingBall.integration_point()
        bouncingBall.simulation_end()

        res = ResultDymolaBinary('bouncingBall_result.mat')

        h = res.get_variable_data('h')
        derh = res.get_variable_data('der(h)')
        g = res.get_variable_data('g')

        assert h.x[0] == pytest.approx(1.000000, abs = 1e-5)
        assert derh.x[0] == pytest.approx(0.000000, abs = 1e-5)

    def _work_flow_me2_aborted(self, result_file_name):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()

        bouncingBall = ResultHandlerBinaryFile(model)

        opts = model.simulate_options()
        opts["result_file_name"] = result_file_name

        bouncingBall.set_options(opts)
        bouncingBall.simulation_start()
        bouncingBall.initialize_complete()
        bouncingBall.integration_point()
        bouncingBall.integration_point()
        bouncingBall.integration_point()
        #No call to simulation end to mimic an aborted simulation
        if isinstance(result_file_name, str): #avoid for streams
            bouncingBall._file.close()

        res = ResultDymolaBinary(result_file_name)

        h = res.get_variable_data('h')
        derh = res.get_variable_data('der(h)')

        assert h.x[0] == pytest.approx(1.000000, abs = 1e-5), "Incorrect initial value for 'h', should be 1.0"
        assert derh.x[0] == pytest.approx(0.000000, abs = 1e-5), "Incorrect  value for 'derh', should be 0.0"
        assert h.x[1] == pytest.approx(1.000000, abs = 1e-5), "Incorrect value for 'h', should be 1.0"
        assert derh.x[1] == pytest.approx(0.000000, abs = 1e-5), "Incorrect value for 'derh', should be 0.0"
        assert h.x[2] == pytest.approx(1.000000, abs = 1e-5), "Incorrect value for 'h', should be 1.0"
        assert derh.x[2] == pytest.approx(0.000000, abs = 1e-5), "Incorrect value for 'derh', should be 0.0"

    def test_work_flow_me2_aborted_file(self):
        self._work_flow_me2_aborted('bouncingBall_result.mat')

    def test_work_flow_me2_aborted_stream(self):
        """ Verify expected workflow for ME2 aborted simulation using byte stream. """
        stream = BytesIO()
        self._work_flow_me2_aborted(stream)

    def test_work_flow_me2_aborted_stream2(self):
        """ Verify exception when using ResultHandlerBinaryFile with a stream that doesnt support anything. """
        class A:
            pass
        stream = A()
        msg = "Failed to write the result file. Option 'result_file_name' needs to be a filename or a class that supports 'write', 'tell' and 'seek'."
        with pytest.raises(FMUException, match = msg):
            self._work_flow_me2_aborted(stream)

    def test_work_flow_me2_aborted_stream3(self):
        """ Verify exception when using ResultHandlerBinaryFile with a stream that doesnt support 'seek'. """
        class A:
            def write(self):
                pass
            def tell(self):
                pass
        stream = A()
        msg = "Failed to write the result file. Option 'result_file_name' needs to be a filename or a class that supports 'write', 'tell' and 'seek'."
        with pytest.raises(FMUException, match = msg):
            self._work_flow_me2_aborted(stream)

    def test_work_flow_me2_aborted_stream4(self):
        """ Verify exception when using ResultHandlerBinaryFile with a stream that doesnt support 'tell'. """
        class A:
            def write(self):
                pass
            def seek(self):
                pass
        stream = A()
        msg = "Failed to write the result file. Option 'result_file_name' needs to be a filename or a class that supports 'write', 'tell' and 'seek'."
        with pytest.raises(FMUException, match = msg):
            self._work_flow_me2_aborted(stream)

    def test_work_flow_me2_aborted_stream5(self):
        """ Verify exception when using ResultHandlerBinaryFile with a stream that doesnt support 'write'. """
        class A:
            def seek(self):
                pass
            def tell(self):
                pass
        stream = A()
        msg = "Failed to write the result file. Option 'result_file_name' needs to be a filename or a class that supports 'write', 'tell' and 'seek'."
        with pytest.raises(FMUException, match = msg):
            self._work_flow_me2_aborted(stream)

    def test_filter_no_variables(self):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()
        model.time = 1.0
        opts = model.simulate_options()
        opts["filter"] = "NoMatchingVariables"


        bouncingBall = ResultHandlerBinaryFile(model)

        bouncingBall.set_options(opts)
        bouncingBall.simulation_start()
        bouncingBall.initialize_complete()
        bouncingBall.integration_point()
        bouncingBall.simulation_end()

        res = ResultDymolaBinary('bouncingBall_result.mat')

        t = res.get_variable_data('time')
        assert t.x[-1] == pytest.approx(1.000000, abs = 1e-5)

    def test_binary_options_cs2(self):
        simple_alias = Dummy_FMUModelCS2([("x", "y")], os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "NegatedAlias.fmu"), _connect_dll=False)
        _run_negated_alias(simple_alias, "binary")

    def test_binary_options_cs2_stream(self):
        simple_alias = Dummy_FMUModelCS2([("x", "y")], os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "NegatedAlias.fmu"), _connect_dll=False)
        stream = BytesIO()
        _run_negated_alias(simple_alias, "binary", stream)

    def _get_bouncing_ball_dummy(self, fmu_type = 'me2'):
        """ Returns an instance of Dummy_FMUModelME2 using bouncingBall. """
        if fmu_type == 'me2':
            return Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        elif fmu_type == 'me1':
            return Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)

    def test_exception_simulation_start(self):
        """ Verify exception is raised if simulation_start is invoked without arguments. """
        model = self._get_bouncing_ball_dummy()
        opts = model.simulate_options()
        opts["dynamic_diagnostics"] = True
        model.setup_experiment()
        model.initialize()

        bouncingBall = ResultHandlerBinaryFile(model)
        bouncingBall.set_options(opts)
        msg = r"Unable to start simulation. The following keyword argument\(s\) are empty:"
        msg += r" 'diagnostics\_params' and 'diagnostics\_vars'."
        with pytest.raises(FMUException, match = msg):
            bouncingBall.simulation_start()

    def _get_diagnostics_cancelled_sim(self, result_file_name):
        """ Function used to test retrieving model variable data and diagnostics data with a cancelled sim.
            Generalized for both files and streams.
        """
        diagnostics_params = OrderedDict()
        diagnostics_vars = OrderedDict()

        model = self._get_bouncing_ball_dummy()
        opts = model.simulate_options()
        model.setup_experiment()
        model.initialize()

        # Need to mock the diagnostics variables in order to invoke simulation_start
        diagnostics_params[DIAGNOSTICS_PREFIX+"solver."+opts["solver"]] = (1.0, "Chosen solver.")
        try:
            rtol = opts['rtol']
            atol = opts['atol']
        except KeyError:
            rtol, atol = model.get_tolerances()

        diagnostics_vars[DIAGNOSTICS_PREFIX+"step_time"] = (0.0, "Step time")
        nof_states, nof_ei = model.get_ode_sizes()
        for i in range(nof_ei):
            diagnostics_vars[DIAGNOSTICS_PREFIX+"event_info.state_event_info.index_"+str(i+1)] = (0.0, "Zero crossing indicator for event indicator {}".format(i+1))

        # values used as diagnostics data at each point
        diag_data = np.array([val[0] for val in diagnostics_vars.values()], dtype=float)

        opts["dynamic_diagnostics"] = True
        opts["result_file_name"] = result_file_name

        # Generate data
        bouncingBall = ResultHandlerBinaryFile(model)
        bouncingBall.set_options(opts)
        bouncingBall.simulation_start(diagnostics_params=diagnostics_params, diagnostics_vars=diagnostics_vars)
        bouncingBall.initialize_complete() # TODO should be called before simulation_start

        model.time += 0.1
        diag_data[0] += 0.1
        bouncingBall.diagnostics_point(diag_data)
        bouncingBall.integration_point()

        model.time += 0.1
        diag_data[0] += 0.1
        bouncingBall.diagnostics_point(diag_data)
        bouncingBall.integration_point()

        model.time += 0.1
        diag_data[0] += 0.1
        bouncingBall.diagnostics_point(diag_data)
        diag_data[-1] = 1 # change one of the event indicators
        bouncingBall.diagnostics_point(diag_data)
        diag_data[-1] = 0 # change ev-indicator to original value again

        model.time += 0.1
        diag_data[0] += 0.1
        bouncingBall.diagnostics_point(diag_data)
        bouncingBall.integration_point()
        if isinstance(result_file_name, str):
            bouncingBall._file.close()

        # Extract data to be veified
        res = ResultDymolaBinary(result_file_name)
        h = res.get_variable_data('h')
        derh = res.get_variable_data('der(h)')
        ev_ind = res.get_variable_data(DIAGNOSTICS_PREFIX+'event_info.state_event_info.index_1').x

        # Verify
        assert h.x[0] == pytest.approx(1.000000, abs = 1e-5), "Incorrect initial value for 'h', should be 1.0"
        assert derh.x[0] == pytest.approx(0.000000, abs = 1e-5), "Incorrect  value for 'derh', should be 0.0"
        np.testing.assert_array_equal(ev_ind, np.array([0., 0., 0., 0., 1., 0.]))

    def test_diagnostics_data_cancelled_simulation_mat_file(self):
        """ Verify that we can retrieve data and diagnostics data after cancelled sim using matfile. """
        self._get_diagnostics_cancelled_sim("TestCancelledSim.mat")

    def test_diagnostics_data_cancelled_simulation_file_stream(self):
        """ Verify that we can retrieve data and diagnostics data after cancelled sim using filestream. """
        test_file_stream = open('myfilestream.txt', 'wb')
        self._get_diagnostics_cancelled_sim(test_file_stream)


    def test_debug_file_not_generated_when_dynamic_diagnostics_is_true(self):
        """ Verify that the debug file is not created when option dynamic_diagnostics is true. """
        model = self._get_bouncing_ball_dummy()
        opts = model.simulate_options()
        opts["dynamic_diagnostics"] = True
        potential_debug_file = "{}_debug.txt".format(model.get_identifier())
        if os.path.isfile(potential_debug_file):
            os.remove(potential_debug_file)

        model.simulate(options = opts)
        assert not os.path.isfile(potential_debug_file), "Test failed, file {} exists after simulation".format(potential_debug_file)

    def test_exception_dynamic_diagnostics_and_non_binary_result_handling(self):
        """ Verify that an exception is raised if dynamic_diagnostics is True and result_handling is not binary. """
        model = self._get_bouncing_ball_dummy()
        opts = model.simulate_options()
        opts["dynamic_diagnostics"] = True
        opts["result_handling"] = "csv" # set to anything except "binary"

        err_msg = ("The chosen result_handler does not support dynamic_diagnostics."
                   " Try using e.g., ResultHandlerBinaryFile.")
        with pytest.raises(fmi.InvalidOptionException, match = err_msg):
            model.simulate(options = opts)

    def test_exception_dynamic_diagnostics_and_non_binary_result_handling1(self):
        """ Verify that an exception is raised if dynamic diagnostics is True and result_handling is custom
        and does not support dynamic_diagnostics. """
        model = self._get_bouncing_ball_dummy()
        opts = model.simulate_options()
        opts["dynamic_diagnostics"] = True
        opts["result_handling"] = "custom" # set to anything except "binary"

        class Foo(ResultHandler):
            def get_result(self):
                return None

        foo_inst = Foo(model)
        opts["result_handler"] = foo_inst

        assert not foo_inst.supports.get('dynamic_diagnostics')

        err_msg = ("The chosen result_handler does not support dynamic_diagnostics."
                   " Try using e.g., ResultHandlerBinaryFile.")
        with pytest.raises(fmi.InvalidOptionException, match = err_msg):
            model.simulate(options = opts)

    def test_exception_dynamic_diagnostics_and_non_binary_result_handling2(self):
        """ Verify that exception is raised if dynamic diagnostics is True and result_handling is custom and valid class. """
        model = self._get_bouncing_ball_dummy()
        opts = model.simulate_options()
        opts["dynamic_diagnostics"] = True
        opts["result_handling"] = "custom" # set to anything except "binary"

        opts["result_handler"] = ResultHandlerBinaryFile(model)
        no_error = False
        exception_msg = ""
        try:
            model.simulate(options = opts)
            no_error = True
        except Exception as e:
            no_error = False
            exception_msg = str(e)
            raise e
        # In case error did not stop the test run
        assert no_error, "Error occurred: {}".format(exception_msg)

    def test_custom_result_handler_dynamic_diagnostics(self):
        """ Test dynamic diagnostics with a custom results handler that supports it. """
        model = self._get_bouncing_ball_dummy()
        opts = model.simulate_options()
        opts["dynamic_diagnostics"] = True
        opts["result_handling"] = "custom" # set to anything except "binary"

        class ResultDynDiag(ResultHandler):
            """ Dummy result handler with necessary implementations for dynamic_diagnostics."""
            def __init__(self, model = None):
                super().__init__(model)
                self.supports['dynamic_diagnostics'] = True
                self.diagnostics_point_called = False

            def diagnostics_point(self, diag_data = None):
                self.diagnostics_point_called = True

            def get_result(self):
                return None

        res_handler = ResultDynDiag()
        opts["result_handler"] = res_handler
        model.simulate(options = opts)

        assert res_handler.diagnostics_point_called, "diagnostics_point function was never called."

    def test_result_handler_supports_dynamic_diagnostics(self):
        """ Test dynamic diagnostics with a custom results handler that supports it, but lacks actual implementation. """
        model = self._get_bouncing_ball_dummy()
        opts = model.simulate_options()
        opts["dynamic_diagnostics"] = True
        opts["result_handling"] = "custom" # set to anything except "binary"

        class ResultDynDiag(ResultHandler):
            """ Dummy result handler with for dynamic_diagnostics."""
            def __init__(self, model = None):
                super().__init__(model)
                self.supports['dynamic_diagnostics'] = True

            ## lacks implementation of diagnostics_point, default will raise NotImplementedError

            def get_result(self):
                return None

        res_handler = ResultDynDiag()
        opts["result_handler"] = res_handler
        with pytest.raises(NotImplementedError):
            model.simulate(options = opts)

    def _test_no_debug_file(self, fmu_type):
        model = self._get_bouncing_ball_dummy(fmu_type=fmu_type)
        opts = model.simulate_options()
        opts["logging"] = True

        expected_debug_file = "{}_debug.txt".format(model.get_identifier())
        if os.path.isfile(expected_debug_file):
            os.remove(expected_debug_file)

        model.simulate(options = opts)

        assert not os.path.isfile(expected_debug_file), f"file {expected_debug_file} found."

    def test_debug_file_not_generated_me1(self):
        """ Verify that the debug file is not generated by enabling logging (ME1). """
        self._test_no_debug_file(fmu_type = 'me1')

    def test_debug_file_not_generated_me2(self):
        """ Verify that the debug file is not generated by enabling logging (ME2). """
        self._test_no_debug_file(fmu_type = 'me2')

    def _test_debug_file_opening(self, fmu_type):
        model = self._get_bouncing_ball_dummy(fmu_type=fmu_type)
        opts = model.simulate_options()
        opts["logging"] = True
        opts["result_handling"] = "csv" # set to anything except 'binary'

        expected_debug_file = "{}_debug.txt".format(model.get_identifier())
        if os.path.isfile(expected_debug_file):
            os.remove(expected_debug_file)

        test_str = 'thislinewillberemoved'
        # Now add test_str to file, simulate, and verify that test_str is gone
        with open(expected_debug_file, 'w') as f:
            f.write(test_str)
        model.simulate(options = opts)

        # Verify
        with open(expected_debug_file, 'r') as f:
            line = f.readline()
        assert not test_str in line, "Test failed, found '{}' in '{}'".format(test_str, line)

    def test_debug_file_opened_in_write_mode_me1(self):
        """ Verify that the debug file is opened in write mode if it already did exist (ME1). """
        self._test_debug_file_opening(fmu_type = 'me1')

    def test_debug_file_opened_in_write_mode_me2(self):
        """ Verify that the debug file is opened in write mode if it already did exist (ME2). """
        self._test_debug_file_opening(fmu_type = 'me1')

    def test_diagnostics_numerical_values(self):
        """ Verify that we get the expected values for some diagnostics. """
        model = self._get_bouncing_ball_dummy()
        opts = model.simulate_options()
        opts["dynamic_diagnostics"] = True
        opts["ncp"] = 250
        res = model.simulate(options=opts)
        length = len(res['h'])
        np.testing.assert_array_equal(res[f'{DIAGNOSTICS_PREFIX}event_data.event_info.event_type'], np.ones(length) * (-1))

        expected_solver_order = np.ones(length)
        expected_solver_order[0] = 0.0
        np.testing.assert_array_equal(res[f'{DIAGNOSTICS_PREFIX}solver.solver_order'], expected_solver_order)

    def test_get_last_result_file0(self):
        """ Verify get_last_result_file seems to point at the correct file. """
        test_model = self._get_bouncing_ball_dummy()
        file_name = "testname.mat"
        test_model._result_file = file_name
        assert test_model.get_last_result_file().split(os.sep)[-1] == file_name, "Unable to find {} in string {}".format(file_name, test_model.get_last_result_file())

    def test_get_last_result_file1(self):
        """ Verify get_last_result_file returns an absolute path. """
        test_model = self._get_bouncing_ball_dummy()
        file_name = "testname.mat"
        test_model._result_file = file_name
        assert os.path.isabs(test_model.get_last_result_file()), "Expected abspath but got {}".format(test_model.get_last_result_file())

    def test_get_last_result_file2(self):
        """ Verify get_last_result_file doesnt cause exception if the result file is not yet set. """
        test_model = self._get_bouncing_ball_dummy()
        test_model._result_file = None
        assert test_model.get_last_result_file() is None, "Expected None but got {}".format(test_model.get_last_result_file())

    def test_get_last_result_file3(self):
        """ Verify get_last_result_file doesnt cause exception if the result file is not set correctly. """
        test_model = self._get_bouncing_ball_dummy()
        test_model._result_file = 123 # arbitrary number, just verify get_last_result_file works
        assert test_model.get_last_result_file() is None, "Expected None but got {}".format(test_model.get_last_result_file())


@pytest.mark.assimulo
class TestResultCSVTextual_Simulation:
    def test_only_parameters(self):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "ParameterAlias.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerCSV(model)
        opts["filter"] = "p2"

        res = model.simulate(options=opts)

        assert 3.0 == pytest.approx(res["p2"][0])

    def test_no_variables(self):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "ParameterAlias.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerCSV(model)
        opts["filter"] = "NoMatchingVariables"
        opts["result_file_name"] = "NoMatchingTest.csv"

        res = model.simulate(options=opts)

        assert 1.0 == pytest.approx(res["time"][-1])

    def test_variable_alias_custom_handler(self):
        simple_alias = Dummy_FMUModelME1([40], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NegatedAlias.fmu"), _connect_dll=False)

        opts = simple_alias.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerCSV(simple_alias)

        res = simple_alias.simulate(options=opts)

        # test that res['y'] returns a vector of the same length as the time
        # vector
        assert len(res['y']) ==len(res['time']), "Wrong size of result vector."

        x = res["x"]
        y = res["y"]

        for i in range(len(x)):
            assert x[i] == -y[i]

    def test_csv_options_me1(self):
        simple_alias = Dummy_FMUModelME1([40], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NegatedAlias.fmu"), _connect_dll=False)
        _run_negated_alias(simple_alias, "csv")

    def test_csv_options_me2(self):
        simple_alias = Dummy_FMUModelME2([("x", "y")], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NegatedAlias.fmu"), _connect_dll=False)
        _run_negated_alias(simple_alias, "csv")

    def test_csv_options_me1_stream(self):
        simple_alias = Dummy_FMUModelME1([40], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "NegatedAlias.fmu"), _connect_dll=False)
        stream = StringIO()
        _run_negated_alias(simple_alias, "csv", stream)

    def test_csv_options_me2(self):
        simple_alias = Dummy_FMUModelME2([("x", "y")], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NegatedAlias.fmu"), _connect_dll=False)
        stream = StringIO()
        _run_negated_alias(simple_alias, "csv", stream)

    def test_enumeration_csv(self):

        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "Friction2.fmu"), _connect_dll=False)
        data_type = model.get_variable_data_type("mode")

        assert data_type == fmi.FMI2_ENUMERATION

        opts = model.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"] = ResultHandlerCSV(model)

        res = model.simulate(options=opts)
        res["mode"] #Check that the enumeration variable is in the dict, otherwise exception

class TestResultCSVTextual:

    def test_constructor_invalid_stream1(self):
        """ Verify exception is raised for ResultCSVTextual if filename argument is a stream not supporting 'readline'. """
        class A:
            def seek(self):
                pass
        stream = A()
        msg = "Given stream needs to support 'readline' and 'seek' in order to retrieve the results."
        with pytest.raises(JIOError, match = msg):
            res = ResultCSVTextual(stream)

    def test_constructor_invalid_stream2(self):
        """ Verify exception is raised for ResultCSVTextual if filename argument is a stream not supporting 'seek'. """
        class A:
            def readline(self):
                pass
        stream = A()
        msg = "Given stream needs to support 'readline' and 'seek' in order to retrieve the results."
        with pytest.raises(JIOError, match = msg):
            res = ResultCSVTextual(stream)

    def test_delimiter(self):

        res = ResultCSVTextual(os.path.join(file_path, 'files', 'Results', 'TestCSV.csv'), delimiter=",")

        x = res.get_variable_data("fd.y")

        assert x.x[-1] == 1

    def _work_flow_me1(self, result_file_name):
        model = Dummy_FMUModelME1([], os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "bouncingBall.fmu"), _connect_dll=False)
        model.initialize()

        bouncingBall = ResultHandlerCSV(model)

        opts = model.simulate_options()
        opts["result_file_name"] = result_file_name
        bouncingBall.set_options(opts)
        bouncingBall.simulation_start()
        bouncingBall.initialize_complete()
        bouncingBall.integration_point()
        bouncingBall.simulation_end()

        res = ResultCSVTextual('bouncingBall_result.csv')

        h = res.get_variable_data('h')
        derh = res.get_variable_data('der(h)')
        g = res.get_variable_data('g')

        assert h.x == pytest.approx(1.000000, abs = 1e-5)
        assert derh.x == pytest.approx(0.000000, abs = 1e-5)

    def test_work_flow_me1_file(self):
        self._work_flow_me1('bouncingBall_result.csv')

    def test_work_flow_me1_stream(self):
        stream = StringIO()
        self._work_flow_me1(stream)

    def _work_flow_me2(self, result_file_name):
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        model.setup_experiment()
        model.initialize()

        bouncingBall = ResultHandlerCSV(model)
        opts = model.simulate_options()
        opts["result_file_name"] = result_file_name
        bouncingBall.set_options(opts)
        bouncingBall.simulation_start()
        bouncingBall.initialize_complete()
        bouncingBall.integration_point()
        bouncingBall.simulation_end()

        res = ResultCSVTextual(result_file_name)

        h = res.get_variable_data('h')
        derh = res.get_variable_data('der(h)')
        g = res.get_variable_data('g')

        assert h.x == pytest.approx(1.000000, abs = 1e-5)
        assert derh.x == pytest.approx(0.000000, abs = 1e-5)


    def test_work_flow_me2_file(self):
        self._work_flow_me2('bouncingBall_result.csv')

    def test_work_flow_me2_stream(self):
        stream = StringIO()
        self._work_flow_me2(stream)

    def test_work_flow_me2_stream2(self):
        """ Verify exception when using ResultHandlerCSV with a stream that doesnt support 'write'. """
        class A:
            pass
        stream = A() # send in something that is not a string
        msg = "Failed to write the result file. Option 'result_file_name' needs to be a filename or a class that supports writing to through the 'write' method."
        with pytest.raises(FMUException, match = msg):
            self._work_flow_me2(stream)

    def test_csv_options_cs1(self):
        simple_alias = Dummy_FMUModelCS1([40], os.path.join(file_path, "files", "FMUs", "XML", "CS1.0", "NegatedAlias.fmu"), _connect_dll=False)
        _run_negated_alias(simple_alias, "csv")

    def test_csv_options_cs2(self):
        simple_alias = Dummy_FMUModelCS2([("x", "y")], os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "NegatedAlias.fmu"), _connect_dll=False)
        _run_negated_alias(simple_alias, "csv")

class TestResultDymolaBinary:

    def test_next_start_index(self):
        """
            Test that calculation of the next start index works as expected.

            This test sets up a dummy FMU and dummy trajectories since we need
            trajectories of uneven lengths.

        """
        # Begin by setting up minimal required environment in order to perform the test
        fmu = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "CoupledClutches.fmu"),
            _connect_dll=False)

        result_handler = ResultHandlerBinaryFile(fmu)

        opts = fmu.simulate_options()
        opts["result_handling"] = "binary"
        opts["result_handler"] = result_handler

        fmu.setup_experiment()
        fmu.initialize()
        opts["initialize"] = False

        result_handler.set_options(opts) # required in order to call simulation_start()
        result_handler.initialize_complete()
        result_handler.simulation_start()

        fmu.set('J4.phi', 1) # arbitrary
        result_handler.integration_point()
        rdb = ResultDymolaBinary(fmu.get_last_result_file(), allow_file_updates=True)

        # Actual test starts below
        vars_to_test = [
            'J1.J',             # this is a parameter
            'clutch1.Backward'  # this is a constant
        ]

        # if this is not True, then the rest of test does not hold
        assert vars_to_test[0] in result_handler.model.get_model_variables(causality = FMI2_PARAMETER).keys()
        assert vars_to_test[1] in result_handler.model.get_model_variables(variability = FMI2_CONSTANT).keys()
        assert 'J4.phi' in result_handler.model.states.keys()


        for v in vars_to_test:
            trajectories1 = {
                'J4.phi': Trajectory(np.array([]), np.array([])),
                v: Trajectory(np.array([0]), np.array([1]))
            }

            trajectories2 = {
                'J4.phi': Trajectory(np.array([0]), np.array([1])),
                v: Trajectory(np.array([0, 1]), np.array([1, 1]))
            }

            trajectories3 = {
                'J4.phi': Trajectory(np.array([0]), np.array([1])),
                v: Trajectory(np.array([0]), np.array([1]))
            }

            trajectories4 = {
                'J4.phi': Trajectory(np.array([0, 1]), np.array([1, 1])),
                v: Trajectory(np.array([0]), np.array([1]))
            }

            trajectories5 = {
                'J4.phi': Trajectory(np.array([0, 1, 2]), np.array([1, 1, 1])),
                v: Trajectory(np.array([0]), np.array([1]))
            }

            assert rdb._find_max_trajectory_length(trajectories1) == 0
            assert rdb._find_max_trajectory_length(trajectories2) == 1
            assert rdb._find_max_trajectory_length(trajectories3) == 1
            assert rdb._find_max_trajectory_length(trajectories4) == 2
            assert rdb._find_max_trajectory_length(trajectories5) == 3

    def _test_get_variables_data(self, dynamic_diagnostics: bool, nbr_of_calls: int, diag_data_ratio: int,
                                 vars_to_test: list, stop_index_function: callable, result_file_name: str) -> dict:
        """
            Simulates a dummy FMU and generates data for get_variables_data.

            Parameters:
            ----------
            dynamic_diagnostics : bool
                Whether to enable dynamic diagnostics during simulation.

            nbr_of_calls : int
                Number of simulation calls to perform.

            diag_data_ratio : int
                Frequency of diagnostic data logging.

            vars_to_test : list
                Variables to retrieve data from during the simulation.

            stop_index_function : callable
                Function to determine the stop index for retrieving variable data.

            result_file_name: str
                Name of the result file.

            Returns:
            -------
            dict
                A dictionary with call indices as keys and lists of trajectory data as values.

            Raises:
            ------
            AssertionError
                If no test data is generated.
        """

        fmu = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "CoupledClutches.fmu"),
            _connect_dll=False)
        f = lambda t : math.sin(5*t + math.pi/2) # function used to set values later on FMU

        result_handler = ResultHandlerBinaryFile(fmu)
        opts = fmu.simulate_options()
        opts["result_handling"] = "binary"
        opts["result_handler"] = result_handler
        opts["result_file_name"] = result_file_name
        opts["dynamic_diagnostics"] = dynamic_diagnostics
        opts["logging"] = opts["dynamic_diagnostics"]

        # tolerances are required only to invoke 'setup_diagnostics_variables'
        opts['CVode_options']['rtol'] = 1e-6
        opts['CVode_options']['atol'] = fmu.nominal_continuous_states * opts['CVode_options']['rtol']

        fmu.setup_experiment()
        fmu.initialize()
        opts["initialize"] = False

        diag_params, diag_vars = setup_diagnostics_variables(
            model = fmu,
            start_time = 0,
            options = opts,
            solver_options = opts['CVode_options'])

        result_handler.set_options(opts) # required in order to call simulation_start()
        result_handler.initialize_complete()
        if opts["dynamic_diagnostics"]:
            result_handler.simulation_start(diag_params, diag_vars)
        else:
            result_handler.simulation_start()

        fmu.set('J4.phi', f(fmu.time)) # arbitrary
        result_handler.integration_point()

        fmu.time += 0.01
        fmu.set('J4.phi', f(fmu.time)) # arbitrary

        rdb = ResultDymolaBinary(opts["result_file_name"], allow_file_updates=True)
        nbr_of_calls = nbr_of_calls
        diag_data_ratio = diag_data_ratio
        counter = 0
        start_index = 0

        nbr_of_points = 5
        h = 1/nbr_of_points * 0.1 # "arbitrary" step through time

        data_to_return = {}
        for i in range(nbr_of_calls):
            time.sleep(0.1) # without this the tests fail on Linux, should look into a proper way of solving this
            for j in range(nbr_of_points):
                result_handler.integration_point()
                if opts["dynamic_diagnostics"] and counter%diag_data_ratio==0:
                    diag_vars['@Diagnostics.step_time'] = (h, 'Step time') # arbitrary test value
                    result_handler.diagnostics_point(np.array([val[0] for val in diag_vars.values()], dtype=float))

                counter += 1
                fmu.time += h
                fmu.set('J4.phi', f(fmu.time)) # arbitrary

            trajectories, start_index = rdb.get_variables_data(vars_to_test, start_index, stop_index_function(start_index))
            data_to_return[i] = trajectories.copy()

        assert data_to_return, "Something went wrong, no test data was generated"
        return data_to_return

    def test_get_variables_data_values0(self):
        """ Verifying values from get_variables_data. """
        vars_to_test = ['J4.phi']
        test_data_sets = self._test_get_variables_data(False, 3, None, vars_to_test, lambda x: None, "TestFile00.mat")

        reference_data = {
            0: [1.00000000, 0.99875026, 0.98877108, 0.96891242, 0.93937271, 0.90044710],
            1: [0.85252452, 0.79608380, 0.73168887, 0.65998315, 0.58168309],
            2: [0.49757105, 0.40848744, 0.31532236, 0.21900669, 0.12050277]
        }

        for index, test_data in test_data_sets.items():
            np.testing.assert_array_almost_equal(test_data['J4.phi'].x, reference_data[index])

    def test_get_variables_data_values1(self):
        """ Verifying values from get_variables_data, with dynamic_diagnostics = True. """
        vars_to_test = ['time', 'J4.phi', '@Diagnostics.step_time', '@Diagnostics.nbr_steps']
        test_data_sets = self._test_get_variables_data(True, 5, 3, vars_to_test, lambda x: None, "TestFile01.mat")

        reference_data = {
            0: [ 1.00000000,  0.99875026,  0.93937271],
            1: [ 0.7960838,   0.58168309],
            2: [ 0.31532236],
            3: [ 0.02079483, -0.27559025],
            4: [-0.54735767, -0.77023125]
        }

        # Just verify results for J4.phi here, but we retrieve all four trajectories at once
        # to see that it works
        for index, test_data in test_data_sets.items():
            np.testing.assert_array_almost_equal(test_data['J4.phi'].x, reference_data[index])

    def test_get_variables_data_values2(self):
        """ Verifying values from get_variables_data, retrieving partial trajectories. """
        vars_to_test = ['time', 'J4.phi']
        test_data_sets = self._test_get_variables_data(False, 5, None, vars_to_test, lambda x: x + 1, "TestFile02.mat")

        reference_data = {
            0: [1],
            1: [0.99875026],
            2: [0.98877108],
            3: [0.96891242],
            4: [0.93937271]
        }

        for index, test_data in test_data_sets.items():
            np.testing.assert_array_almost_equal(test_data['J4.phi'].x, reference_data[index])

    def test_get_variables_data_values3(self):
        """ Verifying values from get_variables_data, and only asking for diagnostic variables. """
        vars_to_test = ['@Diagnostics.step_time', '@Diagnostics.nbr_steps']
        test_data_sets = self._test_get_variables_data(True, 5, 1, vars_to_test, lambda x: None, "TestFile03.mat")

        reference_data = {
            '@Diagnostics.step_time' : {
                0: [0.00, 0.02, 0.02, 0.02, 0.02, 0.02], # the first point 0, then 0.02 because of h=1/5 * 0.1
                1: [0.02, 0.02, 0.02, 0.02, 0.02],
                2: [0.02, 0.02, 0.02, 0.02, 0.02],
                3: [0.02, 0.02, 0.02, 0.02, 0.02],
                4: [0.02, 0.02, 0.02, 0.02, 0.02]
            },
            '@Diagnostics.nbr_steps': {
                0: [ 1.,  2.,  3.,  4.,  5.,  6.],
                1: [ 7.,  8.,  9., 10., 11.],
                2: [12., 13., 14., 15., 16.],
                3: [17., 18., 19., 20., 21.],
                4: [22., 23., 24., 25., 26.],
            }
        }

        for index, test_data in test_data_sets.items():
            np.testing.assert_array_almost_equal(test_data['@Diagnostics.step_time'].x, reference_data['@Diagnostics.step_time'][index])
            np.testing.assert_array_almost_equal(test_data['@Diagnostics.nbr_steps'].x, reference_data['@Diagnostics.nbr_steps'][index])

    def test_get_variables_data_values4(self):
        """ Verifying values from get_variables_data, partial trajectories and checking both time and diagnostic data."""
        vars_to_test = ['time', '@Diagnostics.nbr_steps']
        test_data_sets = self._test_get_variables_data(True, 5, 1, vars_to_test, lambda x: x + 2, "TestFile04.mat")

        reference_data = {
            'time' : {
                0: [0.  , 0.01],
                1: [0.03, 0.05],
                2: [0.07, 0.09],
                3: [0.11, 0.13],
                4: [0.15, 0.17]
            },
            '@Diagnostics.nbr_steps': {
                0: [1, 2],
                1: [3, 4],
                2: [5, 6],
                3: [7, 8],
                4: [9, 10],
            }
        }

        for index, test_data in test_data_sets.items():
            np.testing.assert_array_almost_equal(test_data['time'].x, reference_data['time'][index])
            np.testing.assert_array_almost_equal(test_data['@Diagnostics.nbr_steps'].x, reference_data['@Diagnostics.nbr_steps'][index])

    def test_stop_index_near_bounds(self):
        """ Verify that we get expected results near the end of the result file, including
            stop_index out of range.
        """
        fmu = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        res = fmu.simulate()
        assert len(res['h']) == 501

        rdb = ResultDymolaBinary(fmu.get_last_result_file(), allow_file_updates = True)
        np.testing.assert_array_almost_equal(
            (rdb.get_variables_data(['h'], 495, 496)[0]['h'].x),
            np.array([0.37268813]))
        np.testing.assert_array_almost_equal(
            (rdb.get_variables_data(['h'], 495, 500)[0]['h'].x),
            np.array([0.37268813, 0.37194424, 0.37120184, 0.37046092, 0.36972148]))

        np.testing.assert_array_almost_equal(
            (rdb.get_variables_data(['h'], 495, 499)[0]['h'].x),
            np.array([0.37268813, 0.37194424, 0.37120184, 0.37046092]))
        np.testing.assert_array_almost_equal(
            (rdb.get_variables_data(['h'], 495, 501)[0]['h'].x),
            np.array([0.37268813, 0.37194424, 0.37120184, 0.37046092, 0.36972148, 0.36898351]))
        np.testing.assert_array_almost_equal(
            (rdb.get_variables_data(['h'], 495, 502)[0]['h'].x),
            np.array([0.37268813, 0.37194424, 0.37120184, 0.37046092, 0.36972148, 0.36898351]))


    def test_trajectory_lengths(self):
        """ Verify lengths of trajectories are expected for a bunch of different inputs. """
        fmu = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        res = fmu.simulate()
        assert len(res['h']) == 501
        rdb = ResultDymolaBinary(fmu.get_last_result_file(), allow_file_updates = True)
        assert len(rdb.get_variables_data(['h'], 495, 496)[0]['h'].x) == 1
        assert len(rdb.get_variables_data(['h'], 495, 500)[0]['h'].x) == 5
        assert len(rdb.get_variables_data(['h'], 495, 499)[0]['h'].x) == 4
        assert len(rdb.get_variables_data(['h'], 495, 501)[0]['h'].x) == 6
        assert len(rdb.get_variables_data(['h'], 495, 502)[0]['h'].x) == 6
        # a couple of repeated values to verify the cache is not being used
        assert len(rdb.get_variables_data(['h'], 0, None)[0]['h'].x) == 501
        assert len(rdb.get_variables_data(['h'], 0, 5)[0]['h'].x) == 5
        assert len(rdb.get_variables_data(['h'], 0, None)[0]['h'].x) == 501
        assert len(rdb.get_variables_data(['h'], 0, 5)[0]['h'].x) == 5
        assert len(rdb.get_variables_data(['h'], 0, 5)[0]['h'].x) == 5

        assert len(rdb.get_variables_data(['h'], 5, 15)[0]['h'].x) == 10
        assert len(rdb.get_variables_data(['h'], 0, 550)[0]['h'].x) == 501
        assert len(rdb.get_variables_data(['h'], 0, 10000)[0]['h'].x) == 501

        # test different scenarios of start_index out of bounds
        assert len(rdb.get_variables_data(['h'], 501, 502)[0]['h'].x) == 0
        assert len(rdb.get_variables_data(['h'], 501, None)[0]['h'].x) == 0
        assert len(rdb.get_variables_data(['h'], 501)[0]['h'].x) == 0
        assert len(rdb.get_variables_data(['h'], 1234567)[0]['h'].x) == 0

        # Verify next_start_index also for no variables is equal to start_index
        assert rdb.get_variables_data([], start_index = 0)[1] == 0
        assert rdb.get_variables_data([], start_index = 1)[1] == 1
        assert rdb.get_variables_data([], start_index = 5)[1] == 5

    def test_mixes_get_variable_s_data(self):
        """Test there are no issues when mixing calls of get_variable_data and get_variables_data."""
        fmu = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        ncp = 500
        fmu.simulate(options = {"ncp": ncp})
        rdb = ResultDymolaBinary(fmu.get_last_result_file(), allow_file_updates = True)

        vars = ["time"]
        start_index, stop_index = 0, 5

        partial_1, _ = rdb.get_variables_data(vars, start_index, stop_index)
        full_traj = rdb.get_variable_data(vars[0])
        partial_2, _ = rdb.get_variables_data(vars, start_index, stop_index)

        assert len(partial_1[vars[0]].x) == (stop_index - start_index)
        assert len(full_traj.x) == (ncp + 1)
        assert len(partial_2[vars[0]].x) == (stop_index - start_index)

    def test_cpu_time(self):
        """ Verify the cumulative CPU time trajectory is never decreasing. """
        fmu = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "bouncingBall.fmu"), _connect_dll=False)
        opts = fmu.simulate_options()
        opts['dynamic_diagnostics'] = True
        opts["result_file_name"] = "TestCPUTime.mat"
        res = fmu.simulate(options = opts)

        rdb = ResultDymolaBinary(opts["result_file_name"])
        cpu_time = rdb.get_variable_data('cpu_time').x

        first_value = -1 # initialize to any negative value since the first cpu_time value is 0.0

        # Test that the data is never decreasing (since we return it using numpy cumulative sum)
        for value in cpu_time:
            assert value >= first_value
            first_value = value

@pytest.mark.assimulo
class TestFileSizeLimit:
    def _setup(self, result_type, result_file_name="", fmi_type="me"):
        if fmi_type == "me":
            model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "CoupledClutches.fmu"), _connect_dll=False)
        else:
            model = Dummy_FMUModelCS2([], os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "CoupledClutches.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["result_handling"]  = result_type
        opts["result_file_name"] = result_file_name

        return model, opts

    def _test_result(self, result_type, result_file_name="", max_size=1e6):
        model, opts = self._setup(result_type, result_file_name)

        opts["result_max_size"] = max_size

        #No exception should be raised.
        res = model.simulate(options=opts)

    def _test_result_exception(self, result_type, result_file_name="", fmi_type="me"):
        model, opts = self._setup(result_type, result_file_name, fmi_type)

        opts["result_max_size"] = 10

        with pytest.raises(ResultSizeError):
            model.simulate(options=opts)

    def _test_result_size_verification(self, result_type, result_file_name="", dynamic_diagnostics=False):
        """
        Verifies that the ResultSizeError exception is triggered (due to too large result) and also verifies
        that the resulting file is within bounds of the set maximum size.
        """
        model, opts = self._setup(result_type, result_file_name)
        model.setup_experiment()
        model.initialize()

        max_size = 1e6
        opts["result_max_size"] = max_size
        opts["dynamic_diagnostics"] = dynamic_diagnostics
        opts["logging"] = dynamic_diagnostics
        opts["ncp"] = 0 #Set to zero to circumvent the early size check
        ncp = 10000

        result_handler = get_result_handler(model, opts)

        result_handler.set_options(opts)
        result_handler.initialize_complete()

        if opts["dynamic_diagnostics"]:
            opts['CVode_options']['rtol'] = 1e-6
            opts['CVode_options']['atol'] = model.nominal_continuous_states * opts['CVode_options']['rtol']
            diag_params, diag_vars = setup_diagnostics_variables(model, 0, opts, opts['CVode_options'])
            result_handler.simulation_start(diag_params, diag_vars)
        else:
            result_handler.simulation_start()

        with pytest.raises(ResultSizeError):
            for _ in range(ncp):
                result_handler.integration_point()

                if opts["dynamic_diagnostics"]:
                    result_handler.diagnostics_point(np.array([val[0] for val in diag_vars.values()], dtype=float))

        result_file = model.get_last_result_file()
        file_size = os.path.getsize(result_file)

        assert file_size > max_size*0.9 and file_size < max_size*1.1, \
                "The file size is not within 10% of the given max size"

    def _test_result_size_early_abort(self, result_type, result_file_name=""):
        """
        Verifies that the ResultSizeError is triggered and also verifies that the cause of the error being
        triggered was due to that the ESTIMATE for the result size was too big.
        """
        model, opts = self._setup(result_type, result_file_name)

        max_size = 1e6
        opts["result_max_size"] = max_size
        opts["ncp"] = 10000000

        with pytest.raises(ResultSizeError):
            model.simulate(options=opts)

        result_file = model.get_last_result_file()
        if result_file:
            file_size = os.path.getsize(result_file)

            assert file_size > max_size*0.9 and file_size < max_size*1.1, \
                   "The file size is not within 10% of the given max size"

    def _test_result_size_early_abort(self, result_type, result_file_name=""):
        """
        Verifies that the ResultSizeError is triggered and also verifies that the cause of the error being
        triggered was due to that the ESTIMATE for the result size was too big.
        """
        model, opts = self._setup(result_type, result_file_name)

        max_size = 1e6
        opts["result_max_size"] = max_size
        opts["ncp"] = 10000000

        with pytest.raises(ResultSizeError):
            model.simulate(options=opts)

        result_file = model.get_last_result_file()
        if result_file:
            file_size = os.path.getsize(result_file)

            assert file_size < max_size*0.1, \
                    "The file size is not small, no early abort"

    # TODO: Pytest parametrization
    """
    Binary
    """
    def test_binary_file_size_verification_diagnostics(self):
        """
        Make sure that the diagnostics variables are also taken into account.
        """
        self._test_result_size_verification("binary", dynamic_diagnostics=True)

    def test_binary_file_size_verification(self):
        self._test_result_size_verification("binary")

    def test_binary_file_size_early_abort(self):
        self._test_result_size_early_abort("binary")

    def test_small_size_binary_file(self):
        self._test_result_exception("binary")

    def test_small_size_binary_file_cs(self):
        self._test_result_exception("binary", fmi_type="cs")

    def test_small_size_binary_file_stream(self):
        self._test_result_exception("binary", BytesIO())

    def test_large_size_binary_file(self):
        self._test_result("binary")

    def test_large_size_binary_file_stream(self):
        self._test_result("binary", BytesIO())

    """
    Text
    """
    def test_text_file_size_verification(self):
        self._test_result_size_verification("file")

    def test_text_file_size_early_abort(self):
        self._test_result_size_early_abort("file")

    def test_small_size_text_file(self):
        self._test_result_exception("file")

    def test_small_size_text_file_stream(self):
        self._test_result_exception("file", StringIO())

    def test_large_size_text_file(self):
        self._test_result("file")

    def test_large_size_text_file_stream(self):
        self._test_result("file", StringIO())

    """
    CSV
    """
    def test_csv_file_size_verification(self):
        self._test_result_size_verification("csv")

    def test_csv_file_size_early_abort(self):
        self._test_result_size_early_abort("csv")

    def test_small_size_csv_file(self):
        self._test_result_exception("csv")

    def test_small_size_csv_file_stream(self):
        self._test_result_exception("csv", StringIO())

    def test_large_size_csv_file(self):
        self._test_result("csv", max_size=10000000)

    def test_large_size_csv_file_stream(self):
        self._test_result("csv", StringIO(), max_size=10000000)

    """
    Memory
    """
    def test_small_size_memory(self):
        self._test_result_exception("memory")

    def test_memory_size_early_abort(self):
        self._test_result_size_early_abort("memory")

    def test_small_size_memory_stream(self):
        self._test_result_exception("memory", StringIO())

    def test_large_size_memory(self):
        self._test_result("memory")

    def test_large_size_memory_stream(self):
        self._test_result("memory", StringIO())


class ResultHandlerCustomNoSupport(ResultHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports = {} # Don't do that (the dictionary should be UPDATED, not REPLACED)

class TestCustomResultHandlerMissingSupport:
    """Test that ResultHandlers fail gracefully, even if one overwrites the supports attribute."""
    # caplog = pytest.LogCaptureFixture
    def test_limit_result_size(self, caplog):
        """Test limiting the result size when support is missing."""
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "CoupledClutches.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"]  = ResultHandlerCustomNoSupport(model)
        opts["result_file_name"]  = "res.mat"
        opts["result_max_size"] = 1000

        with pytest.raises(NotImplementedError):
            model.simulate(options = opts) # missing get_result implementation

        msg = "The chosen result handler does not support limiting the result size. Ignoring option 'result_max_size'."
        assert msg in caplog.text

    def test_dynamic_diags(self):
        """Test simulation with DynamicDiagnostics."""
        model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "CoupledClutches.fmu"), _connect_dll=False)

        opts = model.simulate_options()
        opts["result_handling"] = "custom"
        opts["result_handler"]  = ResultHandlerCustomNoSupport(model)
        opts["result_file_name"]  = "res.mat"
        opts["result_max_size"] = 0
        opts["dynamic_diagnostics"] = True

        msg = "The chosen result_handler does not support dynamic_diagnostics. Try using e.g., ResultHandlerBinaryFile."
        with pytest.raises(InvalidOptionException, match = re.escape(msg)):
            model.simulate(options = opts)
