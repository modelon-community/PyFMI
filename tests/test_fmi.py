#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Modelon AB
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

from pyfmi import testattr
from pyfmi.fmi import FMUModel, FMUException, FMUModelME1, FMUModelCS1, load_fmu, FMUModelCS2, FMUModelME2, PyEventInfo
import pyfmi.fmi_util as fmi_util
import pyfmi.fmi as fmi
import pyfmi.fmi_algorithm_drivers as fmi_algorithm_drivers
from pyfmi.tests.test_util import Dummy_FMUModelCS1, Dummy_FMUModelME1, Dummy_FMUModelME2, Dummy_FMUModelCS2
from pyfmi.common.io import ResultHandler

assimulo_installed = True
try:
    import assimulo
except ImportError:
    assimulo_installed = False

file_path = os.path.dirname(os.path.abspath(__file__))

if assimulo_installed:
    class Test_FMUModelME1_Simulation:
        @testattr(stddist = True)
        def test_simulate_with_debug_option_no_state(self):
            model = Dummy_FMUModelME1([], "NoState.Example1.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)

            opts=model.simulate_options()
            opts["logging"] = True
            
            #Verify that a simulation is successful
            res=model.simulate(options=opts)
            
            from pyfmi.debug import CVodeDebugInformation
            debug = CVodeDebugInformation("NoState_Example1_debug.txt")
        
        @testattr(stddist = True)
        def test_no_result(self):
            model = Dummy_FMUModelME1([], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)

            opts = model.simulate_options()
            opts["result_handling"] = "none"
            res = model.simulate(options=opts)

            nose.tools.assert_raises(Exception,res._get_result_data)
            
            model = Dummy_FMUModelME1([], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)

            opts = model.simulate_options()
            opts["return_result"] = False
            res = model.simulate(options=opts)

            nose.tools.assert_raises(Exception,res._get_result_data)
        
        @testattr(stddist = True)
        def test_custom_result_handler(self):
            model = Dummy_FMUModelME1([], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)

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
        

class Test_FMUModelME1:
    @testattr(stddist = True)
    def test_get_time_varying_variables(self):
        model = FMUModelME1("RLC_Circuit.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        
        [r,i,b] = model.get_model_time_varying_value_references()
        [r_f, i_f, b_f] = model.get_model_time_varying_value_references(filter="*")
        
        assert len(r) == len(r_f)
        assert len(i) == len(i_f)
        assert len(b) == len(b_f)
    
    @testattr(stddist = True)
    def test_get_time_varying_variables_with_alias(self):
        model = FMUModelME1("Alias1.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        
        [r,i,b] = model.get_model_time_varying_value_references(filter="y*")
        
        assert len(r) == 1
        assert r[0] == model.get_variable_valueref("y")
    
    @testattr(stddist = True)
    def test_get_variable_by_valueref(self):
        bounce = FMUModelME1("bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        assert "der(v)" == bounce.get_variable_by_valueref(3)
        assert "v" == bounce.get_variable_by_valueref(2)

        nose.tools.assert_raises(FMUException, bounce.get_variable_by_valueref,7)
    
    @testattr(windows_full = True)
    def test_default_experiment(self):
        model = FMUModelME1("CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        
        assert np.abs(model.get_default_experiment_start_time()) < 1e-4
        assert np.abs(model.get_default_experiment_stop_time()-1.5) < 1e-4
        assert np.abs(model.get_default_experiment_tolerance()-0.0001) < 1e-4

    
    @testattr(stddist = True)
    def test_log_file_name(self):
        model = FMUModelME1("bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        assert os.path.exists("bouncingBall_log.txt")
        model = FMUModelME1("bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False, log_file_name="Test_log.txt")
        assert os.path.exists("Test_log.txt")

    @testattr(stddist = True)
    def test_ode_get_sizes(self):
        bounce = FMUModelME1("bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        dq = FMUModelME1("dq.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        
        [nCont,nEvent] = bounce.get_ode_sizes()
        assert nCont == 2
        assert nEvent == 1

        [nCont,nEvent] = dq.get_ode_sizes()
        assert nCont == 1
        assert nEvent == 0
    
    @testattr(stddist = True)
    def test_get_name(self):
        bounce = FMUModelME1("bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        dq = FMUModelME1("dq.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        
        assert bounce.get_name() == 'bouncingBall'
        assert dq.get_name() == 'dq'
    
    @testattr(stddist = True)
    def test_instantiate_jmu(self):
        """
        Test that FMUModel can not be instantiated with a JMU file.
        """
        nose.tools.assert_raises(FMUException,FMUModelME1,'model.jmu')
    
    @testattr(stddist = True)
    def test_get_fmi_options(self):
        """
        Test that simulate_options on an FMU returns the correct options
        class instance.
        """
        bounce = FMUModelME1("bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        assert isinstance(bounce.simulate_options(), fmi_algorithm_drivers.AssimuloFMIAlgOptions)

class Test_FMUModelCS1:
    
    @testattr(stddist = True)
    def test_custom_result_handler(self):
        model = Dummy_FMUModelCS1([], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS1.0"), _connect_dll=False)
        
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
        model = Dummy_FMUModelCS1([], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS1.0"), _connect_dll=False)
        
        opts = model.simulate_options()
        opts["result_handling"] = "none"
        res = model.simulate(options=opts)

        nose.tools.assert_raises(Exception,res._get_result_data)
        
        model = Dummy_FMUModelCS1([], "NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS1.0"), _connect_dll=False)

        opts = model.simulate_options()
        opts["return_result"] = False
        res = model.simulate(options=opts)

        nose.tools.assert_raises(Exception,res._get_result_data)
    
    @testattr(stddist = True)
    def test_result_name_file(self):
        model = Dummy_FMUModelCS1([], "CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS1.0"), _connect_dll=False)
        
        res = model.simulate(options={"result_handling":"file"})

        #Default name
        assert res.result_file == "CoupledClutches_result.txt"
        assert os.path.exists(res.result_file)

        model = Dummy_FMUModelCS1([], "CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS1.0"), _connect_dll=False)
        res = model.simulate(options={"result_file_name":
                                    "CoupledClutches_result_test.txt"})

        #User defined name
        assert res.result_file == "CoupledClutches_result_test.txt"
        assert os.path.exists(res.result_file)
    
    @testattr(stddist = True)
    def test_default_experiment(self):
        model = FMUModelCS1("CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS1.0"), _connect_dll=False)
        
        assert np.abs(model.get_default_experiment_start_time()) < 1e-4
        assert np.abs(model.get_default_experiment_stop_time()-1.5) < 1e-4
        assert np.abs(model.get_default_experiment_tolerance()-0.0001) < 1e-4

    @testattr(stddist = True)
    def test_log_file_name(self):
        model = FMUModelCS1("bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS1.0"), _connect_dll=False)
        assert os.path.exists("bouncingBall_log.txt")
        model = FMUModelCS1("bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS1.0"), _connect_dll=False, log_file_name="Test_log.txt")
        assert os.path.exists("Test_log.txt")

class Test_FMUModelBase:
    @testattr(stddist = True)
    def test_unicode_description(self):
        full_path = os.path.join(file_path, "files", "FMUs", "XML", "ME1.0", "Description.fmu")
        model = FMUModelME1(full_path, _connect_dll=False)
        
        desc = model.get_variable_description("x")

        assert desc == "Test symbols '' ‘’"
    
    @testattr(stddist = True)
    def test_get_erronous_nominals(self):
        model = FMUModelME1("NominalTest4.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        
        nose.tools.assert_almost_equal(model.get_variable_nominal("x"), 2.0)
        nose.tools.assert_almost_equal(model.get_variable_nominal("y"), 1.0)
    
    @testattr(stddist = True)
    def test_caching(self):
        negated_alias = FMUModelME1("NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)

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
        
        negated_alias = FMUModelME1("NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)

        assert len(negated_alias.cache) == 0 #No starting cache
        
        vars_6 = negated_alias.get_model_variables()
        assert id(vars_1) != id(vars_6)
    
    @testattr(stddist = True)
    def test_get_scalar_variable(self):
        negated_alias = FMUModelME1("NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)

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
        model = FMUModelME1("CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        assert model.get_variable_description("J1.phi") == "Absolute rotation angle of component"
    
    @testattr(stddist = True)
    def test_simulation_without_initialization(self):
        model = Dummy_FMUModelME1([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        opts = model.simulate_options()
        opts["initialize"] = False

        nose.tools.assert_raises(FMUException, model.simulate, options=opts)
        
        model = Dummy_FMUModelCS1([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS1.0"), _connect_dll=False)
        opts = model.simulate_options()
        opts["initialize"] = False

        nose.tools.assert_raises(FMUException, model.simulate, options=opts)
        

class Test_FMUModelCS2:
    @testattr(stddist = True)
    def test_log_file_name(self):
        full_path = os.path.join(file_path, "files", "FMUs", "XML", "CS2.0", "CoupledClutches.fmu")
        model = FMUModelCS2(full_path, _connect_dll=False)
        
        path, file_name = os.path.split(full_path)
        assert model.get_log_file_name() == file_name.replace(".","_")[:-4]+"_log.txt"

if assimulo_installed:
    class Test_FMUModelME2_Simulation:
        @testattr(stddist = True)
        def test_basicsens1(self):
            #Noncompliant FMI test as 'd' is parameter is not supposed to be able to be set during simulation
            model = Dummy_FMUModelME2([], "BasicSens1.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            
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
            model = Dummy_FMUModelME2([], "BasicSens1.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            
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
            model = Dummy_FMUModelME2([], "BasicSens2.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            
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
            model = Dummy_FMUModelME2([], "NoState.Example1.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            
            opts = model.simulate_options()
            opts["CVode_options"]["rtol"] = 1e-8
            
            res = model.simulate(options=opts)
            
            assert res.options["CVode_options"]["atol"] == 1e-10
        
        @testattr(stddist = True)
        def test_simulate_with_debug_option_no_state(self):
            model = Dummy_FMUModelME2([], "NoState.Example1.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)

            opts=model.simulate_options()
            opts["logging"] = True
            
            #Verify that a simulation is successful
            res=model.simulate(options=opts)
            
            from pyfmi.debug import CVodeDebugInformation
            debug = CVodeDebugInformation("NoState_Example1_debug.txt")
        
        @testattr(stddist = True)
        def test_maxord_is_set(self):
            model = Dummy_FMUModelME2([], "NoState.Example1.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            opts = model.simulate_options()
            opts["solver"] = "CVode"
            opts["CVode_options"]["maxord"] = 1
            
            res = model.simulate(final_time=1.5,options=opts)
            
            assert res.solver.maxord == 1
        
        @testattr(stddist = True)
        def test_with_jacobian_option(self):
            model = Dummy_FMUModelME2([], "NoState.Example1.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            opts = model.simulate_options()
            opts["solver"] = "CVode"
            opts["result_handling"] = "none"
            
            from pyfmi.fmi_algorithm_drivers import AssimuloFMIAlg, PYFMI_JACOBIAN_LIMIT
            
            class TempAlg(AssimuloFMIAlg):
                def solve(self):
                    pass
            
            def run_case(expected, default="Default"):
                model.reset()
                res = model.simulate(final_time=1.5,options=opts, algorithm=TempAlg)
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
            from pyfmi.fmi_algorithm_drivers import AssimuloFMIAlg, PYFMI_JACOBIAN_SPARSE_SIZE_LIMIT, PYFMI_JACOBIAN_SPARSE_NNZ_LIMIT
            
            class TempAlg(AssimuloFMIAlg):
                def solve(self):
                    pass
            
            def run_case(expected_jacobian, expected_sparse, fnbr=0, nnz={}, set_sparse=False):
                class Sparse_FMUModelME2(Dummy_FMUModelME2):
                    def get_derivatives_dependencies(self):
                        return (nnz, {})
            
                model = Sparse_FMUModelME2([], "NoState.Example1.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
                opts = model.simulate_options()
                opts["solver"] = "CVode"
                opts["result_handling"] = "none"
                if set_sparse:
                    opts["CVode_options"]["linear_solver"] = "SPARSE"
                
                model.get_ode_sizes = lambda: (fnbr, 0)
                
                res = model.simulate(final_time=1.5,options=opts, algorithm=TempAlg)
                assert res.solver.problem._with_jacobian == expected_jacobian, res.solver.problem._with_jacobian
                assert res.solver.linear_solver == expected_sparse, res.solver.linear_solver
            
            run_case(False, "DENSE")
            run_case(True, "DENSE",  PYFMI_JACOBIAN_SPARSE_SIZE_LIMIT+1, {"Dep": [1]*PYFMI_JACOBIAN_SPARSE_SIZE_LIMIT**2})
            run_case(True, "SPARSE", PYFMI_JACOBIAN_SPARSE_SIZE_LIMIT+1, {"Dep": [1]*PYFMI_JACOBIAN_SPARSE_SIZE_LIMIT})
            run_case(True, "SPARSE", PYFMI_JACOBIAN_SPARSE_SIZE_LIMIT+1, {"Dep": [1]*PYFMI_JACOBIAN_SPARSE_SIZE_LIMIT}, True)
        
        @testattr(stddist = True)
        def test_ncp_option(self):
            model = Dummy_FMUModelME2([], "NoState.Example1.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
            opts = model.simulate_options()
            assert opts["ncp"] == 500, opts["ncp"]
            
class Test_FMUModelME2:
    
    @testattr(stddist = True)
    def test_estimate_directional_derivatives_without_structure_info(self):
        full_path = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "Bouncing_Ball.fmu")
        model = Dummy_FMUModelME2([], full_path, _connect_dll=False)
        
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
        full_path = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "OutputTest2.fmu")
        model = Dummy_FMUModelME2([], full_path, _connect_dll=False)
        
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
        full_path = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "OutputTest2.fmu")
        model = FMUModelME2(full_path, _connect_dll=False)
        
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
        full_path = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "CoupledClutches.fmu")
        
        model = FMUModelME2(full_path, _connect_dll=False)
        
        [state_dep, input_dep] = model.get_output_dependencies()
        
        assert len(state_dep.keys()) == 0, len(state_dep.keys())
        assert len(input_dep.keys()) == 0, len(input_dep.keys())
    
    @testattr(stddist = True)
    def test_derivative_dependencies(self):
        full_path = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "NoState.Example1.fmu")
        
        model = FMUModelME2(full_path, _connect_dll=False)
        
        [state_dep, input_dep] = model.get_derivatives_dependencies()
        
        assert len(state_dep.keys()) == 0, len(state_dep.keys())
        assert len(input_dep.keys()) == 0, len(input_dep.keys())
    
    @testattr(stddist = True)
    def test_malformed_xml(self):
        nose.tools.assert_raises(FMUException, load_fmu, os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "MalFormed.fmu"))
        
    @testattr(stddist = True)
    def test_log_file_name(self):
        full_path = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "CoupledClutches.fmu")
        
        model = FMUModelME2(full_path, _connect_dll=False)
        
        path, file_name = os.path.split(full_path)
        assert model.get_log_file_name() == file_name.replace(".","_")[:-4]+"_log.txt"
    
    @testattr(stddist = True)
    def test_units(self):
        model = FMUModelME2("CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        assert model.get_variable_unit("J1.w") == "rad/s", model.get_variable_unit("J1.w")
        assert model.get_variable_unit("J1.phi") == "rad", model.get_variable_unit("J1.phi")
        
        nose.tools.assert_raises(FMUException, model.get_variable_unit, "clutch1.useHeatPort")
        nose.tools.assert_raises(FMUException, model.get_variable_unit, "clutch1.sss")
        nose.tools.assert_raises(FMUException, model.get_variable_unit, "clutch1.sss")
    
    @testattr(stddist = True)
    def test_display_units(self):
        model = FMUModelME2("CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        assert model.get_variable_display_unit("J1.phi") == "deg", model.get_variable_display_unit("J1.phi")
        nose.tools.assert_raises(FMUException, model.get_variable_display_unit, "J1.w")

class Test_FMUModelBase2:
    @testattr(stddist = True)
    def test_unicode_description(self):
        full_path = os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "Description.fmu")
        model = FMUModelME2(full_path, _connect_dll=False)
        
        desc = model.get_variable_description("x")

        assert desc == "Test symbols '' ‘’"
    
    @testattr(stddist = True)
    def test_declared_enumeration_type(self):
        model = FMUModelME2("Enumerations.Enumeration3.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        enum = model.get_variable_declared_type("x")
        assert len(enum.items.keys()) == 2, len(enum.items.keys())
        enum = model.get_variable_declared_type("home")
        assert len(enum.items.keys()) == 4, len(enum.items.keys())
        
        nose.tools.assert_raises(FMUException, model.get_variable_declared_type, "z")

    @testattr(stddist = True)
    def test_get_erronous_nominals(self):
        model = FMUModelME2("NominalTests.NominalTest4.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
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
        
    
    @testattr(stddist = True)
    def test_get_time_varying_variables(self):
        model = FMUModelME2("CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
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
        bounce = Dummy_FMUModelME2([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        bounce.setup_experiment()
        bounce.initialize()

        # Bouncing ball don't have the capability, check that this is handled
        nose.tools.assert_raises(FMUException, bounce.get_directional_derivative, [1], [1], [1])
        
        bounce = Dummy_FMUModelCS2([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS2.0"), _connect_dll=False)
        bounce.setup_experiment()
        bounce.initialize()

        # Bouncing ball don't have the capability, check that this is handled
        nose.tools.assert_raises(FMUException, bounce.get_directional_derivative, [1], [1], [1])
        
    @testattr(stddist = True)
    def test_simulation_without_initialization(self):
        model = Dummy_FMUModelME2([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        opts = model.simulate_options()
        opts["initialize"] = False

        nose.tools.assert_raises(FMUException, model.simulate, options=opts)
        
        model = Dummy_FMUModelCS2([], "bouncingBall.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS2.0"), _connect_dll=False)
        opts = model.simulate_options()
        opts["initialize"] = False

        nose.tools.assert_raises(FMUException, model.simulate, options=opts)
    
    @testattr(stddist = True)
    def test_caching(self):
        negated_alias = FMUModelME2("NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)

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
        
        negated_alias = FMUModelME2("NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)

        assert len(negated_alias.cache) == 0 #No starting cache
        
        vars_6 = negated_alias.get_model_variables()
        assert id(vars_1) != id(vars_6)
    
    @testattr(stddist = True)
    def test_get_scalar_variable(self):
        negated_alias = FMUModelME2("NegatedAlias.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)

        sc_x = negated_alias.get_scalar_variable("x")
        
        assert sc_x.name == "x", sc_x.name
        assert sc_x.value_reference >= 0, sc_x.value_reference
        assert sc_x.type == fmi.FMI2_REAL, sc_x.type
        assert sc_x.variability == fmi.FMI2_CONTINUOUS, sc_x.variability
        assert sc_x.causality == fmi.FMI2_LOCAL, sc_x.causality

        nose.tools.assert_raises(FMUException, negated_alias.get_scalar_variable, "not_existing")
    
    @testattr(stddist = True)
    def test_get_variable_description(self):
        model = FMUModelME2("CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        assert model.get_variable_description("J1.phi") == "Absolute rotation angle of component"
    

class Test_load_fmu_only_XML:
    
    @testattr(stddist = True)
    def test_loading_xml_me1(self):
        
        model = FMUModelME1("CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME1.0"), _connect_dll=False)
        
        assert model.get_name() == "CoupledClutches", model.get_name()

    @testattr(stddist = True)
    def test_loading_xml_cs1(self):
        
        model = FMUModelCS1("CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS1.0"), _connect_dll=False)
        
        assert model.get_name() == "CoupledClutches", model.get_name()
        
    @testattr(stddist = True)
    def test_loading_xml_me2(self):
        
        model = FMUModelME2("CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "ME2.0"), _connect_dll=False)
        
        assert model.get_name() == "CoupledClutches", model.get_name()
        
    @testattr(stddist = True)
    def test_loading_xml_cs2(self):
        
        model = FMUModelCS2("CoupledClutches.fmu", os.path.join(file_path, "files", "FMUs", "XML", "CS2.0"), _connect_dll=False)
        
        assert model.get_name() == "CoupledClutches", model.get_name()
