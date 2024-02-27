#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2014-2021 Modelon AB
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
"""
Module for simulation algorithms to be used together with
pyfmi.fmi.FMUModel*.simulate.
"""

#from abc import ABCMeta, abstractmethod
import logging as logging_module
import time
import numpy as N

import pyfmi.fmi as fmi
import pyfmi.fmi_coupled as fmi_coupled
import pyfmi.fmi_extended as fmi_extended
from pyfmi.common.diagnostics import setup_diagnostics_variables 
from pyfmi.common.algorithm_drivers import AlgorithmBase, OptionBase, InvalidAlgorithmOptionException, InvalidSolverArgumentException, JMResultBase
from pyfmi.common.io import get_result_handler
from pyfmi.common.core import TrajectoryLinearInterpolation
from pyfmi.common.core import TrajectoryUserFunction

from timeit import default_timer as timer

default_int = int
int = N.int32
N.int = N.int32

PYFMI_JACOBIAN_LIMIT = 10
PYFMI_JACOBIAN_SPARSE_SIZE_LIMIT = 100
PYFMI_JACOBIAN_SPARSE_NNZ_LIMIT  = 0.15 #In percentage

class FMIResult(JMResultBase):
    def __init__(self, model=None, result_file_name=None, solver=None,
             result_data=None, options=None, status=0, detailed_timings=None):
        JMResultBase.__init__(self,
                model, result_file_name, solver, result_data, options)
        self.status = status
        self.detailed_timings = detailed_timings

class AssimuloFMIAlgOptions(OptionBase):
    """
    Options for the solving the FMU using the Assimulo simulation package.

    Assimulo options::

        solver --
            Specifies the simulation algorithm that is to be used.
            Default: 'CVode'

        ncp    --
            Number of communication points. If ncp is zero, the solver will
            return the internal steps taken.
            Default: '500'

        initialize --
            If set to True, the initializing algorithm defined in the FMU model
            is invoked, otherwise it is assumed the user have manually invoked
            model.initialize()
            Default is True.

        write_scaled_result --
            Set this parameter to True to write the result to file without
            taking scaling into account. If the value of scaled is False,
            then the variable scaling factors of the model are used to
            reproduced the unscaled variable values.
            Default: False

        result_file_name --
            Specifies the name of the file where the simulation result is
            written. Setting this option to an empty string results in a default
            file name that is based on the name of the model class.
            result_file_name can also be set to a stream that supports 'write',
            'tell' and 'seek'.
            Note that depending on choice of result_handling the stream needs to
            support writing to either string or bytes.
            Default: Empty string

        with_jacobian --
            Determines if the Jacobian should be computed from PyFMI (using
            either the directional derivatives, if available, or estimated using
            finite differences) or if the Jacobian should be computed by the
            chosen solver. The default is to use PyFMI if directional
            derivatives are available, otherwise computed by the chosen
            solver.
            Default: "Default"

        dynamic_diagnostics --
            If True, enables logging of diagnostics data to a result file. This requires that
            the option 'result_handler' supports 'dynamic_diagnostics', otherwise an 
            InvalidOptionException is raised.
            The default 'result_handler' ResultHandlerBinaryFile supports 'dynamic_diagnostics'.
            The diagnostics data will be available via the simulation results and/or the
            binary result file generated during simulation.
            Default: False

        logging --
            If True, creates a logfile from the solver in the current
            directory and enables logging of diagnostics data to logfile or resultfile,
            based on simulation option 'result_handling'.

            The diagnostics data is available via the simulation results similar to FMU model variables
            only if 'result_handler' supports 'dynamic_diagnostics'.
            Default: False

        result_handling --
            Specifies how the result should be handled. Either stored to
            file (txt or binary) or stored in memory. One can also use a
            custom handler.

            If 'result_handling' is 'binary', and 'logging' is also enabled,
            the diagnostics data is written to the same binary file as data of FMU model variables.
            Note that these results are interpolated such that model variable trajectory points
            are given at the same time points as diagnostics data.
            Available options: "file", "binary", "memory", "csv", "custom", None
            Default: "binary"

        result_handler --
            The handler for the result. Depending on the option in
            result_handling this either defaults to ResultHandlerFile
            or ResultHandlerMemory. If result_handling custom is chosen
            This MUST be provided.
            Default: None

        return_result --
            Determines if the simulation result should be returned or
            not. If set to False, the simulation result is not loaded
            into memory after the simulation finishes.
            Default: True

        result_store_variable_description --
            Determines if the description for the variables should be
            stored in the result file or not. Only impacts the result
            file formats that supports storing the variable description
            ("file" and "binary").
            Default: True

        filter --
            A filter for choosing which variables to actually store
            result for. The syntax can be found in
            http://en.wikipedia.org/wiki/Glob_%28programming%29 . An
            example is filter = "*der" , stor all variables ending with
            'der'. Can also be a list.
            Default: None
            
        synchronize_simulation --
            If set, the simulation will be synchronized to real-time or a
            scaled real-time, if possible. The available options are:
                True: Simulation is synchronized to real-time
                False: No synchronization
                >0 (float): Simulation is synchronized to the factored 
                            real-time. I.e. factor*real-time
                
                Example: If, set to 10: 10 simulated seconds is synchronized
                         to one real-time second.
            Default: False


    The different solvers provided by the Assimulo simulation package provides
    different options. These options are given in dictionaries with names
    consisting of the solver name concatenated by the string '_options'. The most
    common solver options are documented below, for a complete list of options
    see, http://www.jmodelica.org/assimulo

    Options for CVode::

        rtol    --
            The relative tolerance. The relative tolerance are retrieved from
            the 'default experiment' section in the XML-file and if not
            found are set to 1.0e-4
            Default: "Default" (1.0e-4)

        atol    --
            The absolute tolerance.
            Default: "Default" (rtol*0.01*(nominal values of the continuous states))

        maxh    --
            The maximum step-size allowed to be used by the solver.
            Default: "Default" (max step-size computed based on (final_time-start_time)/ncp)

        discr   --
            The discretization method. Can be either 'BDF' or 'Adams'
            Default: 'BDF'

        iter    --
            The iteration method. Can be either 'Newton' or 'FixedPoint'
            Default: 'Newton'
    """
    def __init__(self, *args, **kw):
        _defaults= {
            'solver': 'CVode',
            'ncp':500,
            'initialize':True,
            'sensitivities':None,
            'write_scaled_result':False,
            'result_file_name':'',
            'with_jacobian':"Default",
            'logging':False,
            'dynamic_diagnostics':False,
            'result_handling':"binary",
            'result_handler': None,
            'return_result': True,
            'result_store_variable_description': True,
            'filter':None,
            'synchronize_simulation':False,
            'extra_equations':None,
            'CVode_options':{'discr':'BDF','iter':'Newton',
                            'atol':"Default",'rtol':"Default","maxh":"Default",'external_event_detection':False},
            'Radau5ODE_options':{'atol':"Default",'rtol':"Default","maxh":"Default"},
            'RungeKutta34_options':{'atol':"Default",'rtol':"Default"},
            'Dopri5_options':{'atol':"Default",'rtol':"Default", "maxh":"Default"},
            'RodasODE_options':{'atol':"Default",'rtol':"Default", "maxh":"Default"},
            'LSODAR_options':{'atol':"Default",'rtol':"Default", "maxh":"Default"},
            'ExplicitEuler_options':{},
            'ImplicitEuler_options':{}
            }
        super(AssimuloFMIAlgOptions,self).__init__(_defaults)
        # for those key-value-sets where the value is a dict, don't
        # overwrite the whole dict but instead update the default dict
        # with the new values
        self._update_keep_dict_defaults(*args, **kw)

class AssimuloFMIAlg(AlgorithmBase):
    """
    Simulation algorithm for FMUs using the Assimulo package.
    """

    def __init__(self,
                 start_time,
                 final_time,
                 input,
                 model,
                 options):
        """
        Create a simulation algorithm using Assimulo.

        Parameters::

            model --
                fmi.FMUModel* object representation of the model.

            options --
                The options that should be used in the algorithm. For details on
                the options, see:

                * model.simulate_options('AssimuloFMIAlgOptions')

                or look at the docstring with help:

                * help(pyfmi.fmi_algorithm_drivers.AssimuloFMIAlgOptions)

                Valid values are:
                - A dict that overrides some or all of the default values
                  provided by AssimuloFMIAlgOptions. An empty dict will thus
                  give all options with default values.
                - AssimuloFMIAlgOptions object.
        """
        self.model = model
        self.timings = {}
        self.time_start_total = timer()

        try:
            import assimulo
        except Exception:
            raise fmi.FMUException(
                'Could not find Assimulo package. Check pyfmi.check_packages()')

        # import Assimulo dependent classes
        from pyfmi.simulation.assimulo_interface import FMIODE, FMIODESENS, FMIODE2, FMIODESENS2

        # set start time, final time and input trajectory
        self.start_time = start_time
        self.final_time = final_time
        self.input = input

        # handle options argument
        if isinstance(options, dict) and not \
            isinstance(options, AssimuloFMIAlgOptions):
            # user has passed dict with options or empty dict = default
            self.options = AssimuloFMIAlgOptions(options)
        elif isinstance(options, AssimuloFMIAlgOptions):
            # user has passed AssimuloFMIAlgOptions instance
            self.options = options
        else:
            raise InvalidAlgorithmOptionException(options)
        
        self.result_handler = get_result_handler(self.model, self.options)
        self._set_options() # set options

        #time_start = timer()

        input_traj = None
        if self.input:
            if hasattr(self.input[1],"__call__"):
                input_traj=(self.input[0],
                        TrajectoryUserFunction(self.input[1]))
            else:
                input_traj=(self.input[0],
                        TrajectoryLinearInterpolation(self.input[1][:,0],
                                                      self.input[1][:,1:]))
            #Sets the inputs, if any
            input_names  = [input_traj[0]] if isinstance(input_traj[0],str) else input_traj[0]
            input_values = input_traj[1].eval(self.start_time)[0,:]

            if len(input_names) != len(input_values):
                raise fmi.FMUException("The number of input variables is not equal to the number of input values, please verify the input object.")

            self.model.set(input_names, input_values)

        self.result_handler.set_options(self.options)

        time_end = timer()
        #self.timings["creating_result_object"] = time_end - time_start
        time_start = time_end
        time_res_init = 0.0

        # Initialize?
        if self.options['initialize']:

            if isinstance(self.model, fmi.FMUModelME1):
                self.model.time = start_time #Set start time before initialization
                self.model.initialize(tolerance=self.rtol)

            elif isinstance(self.model, ((fmi.FMUModelME2, fmi_coupled.CoupledFMUModelME2))):
                self.model.setup_experiment(tolerance=self.rtol, start_time=self.start_time, stop_time=self.final_time)
                self.model.initialize()
                self.model.event_update()
                self.model.enter_continuous_time_mode()
            else:
                raise fmi.FMUException("Unknown model.")

            time_res_init = timer()
            self.result_handler.initialize_complete()
            time_res_init = timer() - time_res_init

        elif self.model.time is None and isinstance(self.model, fmi.FMUModelME2):
            raise fmi.FMUException("Setup Experiment has not been called, this has to be called prior to the initialization call.")
        elif self.model.time is None:
            raise fmi.FMUException("The model need to be initialized prior to calling the simulate method if the option 'initialize' is set to False")

        self._set_absolute_tolerance_options()

        number_of_diagnostics_variables = 0
        if self.result_handler.supports.get('dynamic_diagnostics'):
            _diagnostics_params, _diagnostics_vars = setup_diagnostics_variables(model = self.model, 
                                                                                 start_time = self.start_time,
                                                                                 options = self.options,
                                                                                 solver_options = self.solver_options)
            number_of_diagnostics_variables = len(_diagnostics_vars)

        #See if there is an time event at start time
        if isinstance(self.model, fmi.FMUModelME1):
            event_info = self.model.get_event_info()
            if event_info.upcomingTimeEvent and event_info.nextEventTime == model.time:
                self.model.event_update()

        if abs(start_time - model.time) > 1e-14:
            logging_module.warning('The simulation start time (%f) and the current time in the model (%f) is different. Is the simulation start time correctly set?'%(start_time, model.time))

        time_end = timer()
        self.timings["initializing_fmu"] = time_end - time_start - time_res_init
        time_start = time_end

        if self.result_handler.supports.get('dynamic_diagnostics'):
            self.result_handler.simulation_start(_diagnostics_params, _diagnostics_vars)
        else:
            self.result_handler.simulation_start()

        self.timings["initializing_result"] = timer() - time_start + time_res_init

        # Sensitivities?
        if self.options["sensitivities"]:
            if self.model.get_generation_tool() != "JModelica.org" and \
               self.model.get_generation_tool() != "Optimica Compiler Toolkit":
                if isinstance(self.model, fmi.FMUModelME2):
                    for var in self.options["sensitivities"]:
                        causality = self.model.get_variable_causality(var)
                        if causality != fmi.FMI2_INPUT:
                            raise fmi.FMUException("The sensitivity parameter is not specified as an input which is required.")
                else:
                    raise fmi.FMUException("Sensitivity calculations only possible with JModelica.org generated FMUs")

            if self.options["solver"] != "CVode":
                raise fmi.FMUException("Sensitivity simulations currently only supported using the solver CVode.")

            #Checks to see if all the sensitivities are inside the model
            #else there will be an exception
            self.model.get(self.options["sensitivities"])

        if not self.input and (isinstance(self.model, ((fmi.FMUModelME2, fmi_coupled.CoupledFMUModelME2)))):
            if self.options["sensitivities"]:
                self.probl = FMIODESENS2(self.model,
                                         result_file_name = self.result_file_name,
                                         with_jacobian = self.with_jacobian,
                                         start_time = self.start_time,
                                         parameters = self.options["sensitivities"],
                                         logging = self.options["logging"],
                                         result_handler = self.result_handler,
                                         number_of_diagnostics_variables = number_of_diagnostics_variables)
            else:
                self.probl = FMIODE2(self.model,
                                     result_file_name = self.result_file_name,
                                     with_jacobian = self.with_jacobian,
                                     start_time = self.start_time,
                                     logging = self.options["logging"],
                                     result_handler = self.result_handler,
                                     extra_equations = self.options["extra_equations"],
                                     synchronize_simulation = self.options["synchronize_simulation"],
                                     number_of_diagnostics_variables = number_of_diagnostics_variables)
        elif isinstance(self.model, ((fmi.FMUModelME2, fmi_coupled.CoupledFMUModelME2))):
            if self.options["sensitivities"]:
                self.probl = FMIODESENS2(self.model,
                                         input_traj,
                                         result_file_name = self.result_file_name,
                                         with_jacobian = self.with_jacobian,
                                         start_time = self.start_time,
                                         parameters = self.options["sensitivities"],
                                         logging = self.options["logging"],
                                         result_handler = self.result_handler,
                                         number_of_diagnostics_variables = number_of_diagnostics_variables)
            else:
                self.probl = FMIODE2(self.model,
                                     input_traj,
                                     result_file_name = self.result_file_name,
                                     with_jacobian = self.with_jacobian,
                                     start_time = self.start_time,
                                     logging = self.options["logging"],
                                     result_handler = self.result_handler,
                                     extra_equations = self.options["extra_equations"],
                                     synchronize_simulation = self.options["synchronize_simulation"],
                                     number_of_diagnostics_variables = number_of_diagnostics_variables)

        elif not self.input:
            if self.options["sensitivities"]:
                self.probl = FMIODESENS(self.model,
                                        result_file_name = self.result_file_name,
                                        with_jacobian = self.with_jacobian,
                                        start_time = self.start_time,
                                        parameters = self.options["sensitivities"],
                                        logging = self.options["logging"],
                                        result_handler = self.result_handler)
            else:
                self.probl = FMIODE(self.model,
                                    result_file_name = self.result_file_name,
                                    with_jacobian = self.with_jacobian,
                                    start_time = self.start_time,
                                    logging = self.options["logging"],
                                    result_handler = self.result_handler)
        else:
            if self.options["sensitivities"]:
                self.probl = FMIODESENS(self.model,
                                        input_traj,
                                        result_file_name = self.result_file_name,
                                        with_jacobian = self.with_jacobian,
                                        start_time = self.start_time,
                                        parameters = self.options["sensitivities"],
                                        logging = self.options["logging"],
                                        result_handler = self.result_handler)
            else:
                self.probl = FMIODE(self.model,
                                    input_traj,
                                    result_file_name = self.result_file_name,
                                    with_jacobian = self.with_jacobian,
                                    start_time = self.start_time,
                                    logging = self.options["logging"],
                                    result_handler = self.result_handler)

        # instantiate solver and set options
        self.simulator = self.solver(self.probl)
        self._set_solver_options()

    def _set_options(self):
        """
        Helper function that sets options for AssimuloFMI algorithm.
        """
        # no of communication points
        self.ncp = self.options['ncp']

        self.write_scaled_result = self.options['write_scaled_result']

        # result file name
        if self.options['result_file_name'] == '':
            self.result_file_name = self.model.get_identifier()+'_result.txt'
        else:
            self.result_file_name = self.options['result_file_name']

        # solver
        import assimulo.solvers as solvers

        solver = self.options['solver']
        if hasattr(solvers, solver):
            self.solver = getattr(solvers, solver)
        else:
            raise InvalidAlgorithmOptionException(f"The solver: {solver} is unknown.")

        if self.options["dynamic_diagnostics"]:
            ## Result handler must have supports['dynamic_diagnostics'] = True
            ## e.g., result_handling = 'binary' = ResultHandlerBinaryFile 
            if not self.result_handler.supports.get('dynamic_diagnostics'):
                err_msg = ("The chosen result_handler does not support dynamic_diagnostics."
                           " Try using e.g., ResultHandlerBinaryFile.")
                raise fmi.InvalidOptionException(err_msg)
            self.options['logging'] = True
        elif self.options['logging']:
            if self.result_handler.supports.get('dynamic_diagnostics'):
                self.options["dynamic_diagnostics"] = True

        # solver options
        try:
            self.solver_options = self.options[solver+'_options']
            try:
                self.solver_options['clock_step']
            except KeyError:
                if self.options['logging']:
                    self.solver_options['clock_step'] = True
        except KeyError: #Default solver options not found
            self.solver_options = {} #Empty dict
            try:
                self.solver.atol
                self.solver_options["atol"] = "Default"
            except AttributeError:
                pass
            try:
                self.solver.rtol
                self.solver_options["rtol"] = "Default"
            except AttributeError:
                pass
            if self.options['logging']:
                self.solver_options['clock_step'] = True

        # Check relative tolerance
        # If the tolerances are not set specifically, they are set
        # according to the 'DefaultExperiment' from the XML file.

        # existence of unbounded attributes may modify rtol, but solver may not support this
        self._rtol_as_scalar_fallback = False
        try:
            #rtol was set as default
            if isinstance(self.solver_options["rtol"], str) and self.solver_options["rtol"] == "Default":
                rtol = self.model.get_relative_tolerance()
                self.solver_options['rtol'] = rtol
            
            #rtol was provided as a vector
            if isinstance(self.solver_options["rtol"], N.ndarray) or isinstance(self.solver_options["rtol"], list):
                
                #rtol all are all equal -> set it as scalar and use that
                if N.all(N.isclose(self.solver_options["rtol"], self.solver_options["rtol"][0])):
                    self.solver_options["rtol"] = self.solver_options["rtol"][0]
                    self.rtol = self.solver_options["rtol"]
                    
                else: #rtol is a vector where not all elements are equal (make sure that all are equal except zeros) (and store the rtol value)
                    fnbr, gnbr = self.model.get_ode_sizes()
                    if len(self.solver_options["rtol"]) != fnbr:
                        raise fmi.InvalidOptionException("If the relative tolerance is provided as a vector, it need to be equal to the number of states.")
                    rtol_scalar = 0.0
                    for tol in self.solver_options["rtol"]:
                        if rtol_scalar == 0.0 and tol != 0.0:
                            rtol_scalar = tol
                            continue
                        if rtol_scalar != 0.0 and tol != 0.0 and rtol_scalar != tol:
                            raise fmi.InvalidOptionException("If the relative tolerance is provided as a vector, the values need to be equal except for zeros.")
                    self.rtol = rtol_scalar
            
            else: #rtol was not provided as a vector -> modify if there are unbounded states
                self.rtol = self.solver_options["rtol"]
                
                if not isinstance(self.model, fmi.FMUModelME1):
                    unbounded_attribute = False
                    rtol_vector = []
                    for state in self.model.get_states_list():
                        if self.model.get_variable_unbounded(state):
                            unbounded_attribute = True
                            rtol_vector.append(0.0)
                        else:
                            rtol_vector.append(self.rtol)
                    
                    if unbounded_attribute:
                        self._rtol_as_scalar_fallback = True
                        self.solver_options['rtol'] = rtol_vector
            
        except KeyError:
            self.rtol = self.model.get_relative_tolerance() #No support for relative tolerance in the used solver

        self.with_jacobian = self.options['with_jacobian']
        if not (isinstance(self.model, fmi.FMUModelME2)): # or isinstance(self.model, fmi_coupled.CoupledFMUModelME2) For coupled FMUs, currently not supported
            self.with_jacobian = False #Force false flag in this case as it is not supported
        elif self.with_jacobian == "Default" and (isinstance(self.model, fmi.FMUModelME2)): #or isinstance(self.model, fmi_coupled.CoupledFMUModelME2)
            if self.model.get_capability_flags()['providesDirectionalDerivatives']:
                self.with_jacobian = True
            else:
                fnbr, gnbr = self.model.get_ode_sizes()
                if fnbr >= PYFMI_JACOBIAN_LIMIT and solver == "CVode":
                    self.with_jacobian = True
                    if fnbr >= PYFMI_JACOBIAN_SPARSE_SIZE_LIMIT:
                        try:
                            self.solver_options["linear_solver"]
                        except KeyError:
                            #Need to calculate the nnz.
                            [derv_state_dep, derv_input_dep] = self.model.get_derivatives_dependencies()
                            nnz = N.sum([len(derv_state_dep[key]) for key in derv_state_dep.keys()])+fnbr
                            if nnz/float(fnbr*fnbr) <= PYFMI_JACOBIAN_SPARSE_NNZ_LIMIT:
                                self.solver_options["linear_solver"] = "SPARSE"
                else:
                    self.with_jacobian = False

    def _set_absolute_tolerance_options(self):
        """
        Sets the absolute tolerance. Must not be called before initialization since it depends
        on state nominals.

        Assumes initial setup of default atol has been done via previous call to _set_options.

        Will try to auto-update absolute tolerances that depend on state nominals retrieved
        before initialization.
        """
        try:
            atol = self.solver_options["atol"]
            preinit_nominals = self.model._preinit_nominal_continuous_states
            if isinstance(atol, str) and atol == "Default":
                fnbr, _ = self.model.get_ode_sizes()
                rtol = self.solver_options["rtol"]
                if fnbr == 0:
                    self.solver_options["atol"] = 0.01*self.rtol
                else:
                    self.solver_options["atol"] = 0.01*self.rtol*self.model.nominal_continuous_states
            elif isinstance(preinit_nominals, N.ndarray) and (preinit_nominals.size > 0):
                # Heuristic:
                # Try to find if atol was specified as "atol = factor * model.nominal_continuous_states",
                # and if that's the case, recompute atol with nominals from after initialization.
                factors = atol / preinit_nominals
                f0 = factors[0]
                for f in factors:
                    if abs(f0 - f) > f0 * 1e-6:
                        return
                # Success.
                self.solver_options["atol"] = atol * self.model.nominal_continuous_states / preinit_nominals
                logging_module.info("Absolute tolerances have been recalculated by using values for state nominals from " +
                             "after initialization.")
        except KeyError:
            pass

    def _set_solver_options(self):
        """
        Helper function that sets options for the solver.
        """
        solver_options = self.solver_options.copy()

        #Set solver option continuous_output
        self.simulator.report_continuously = True

        #If usejac is not set, try to set it according to if directional derivatives
        #exists. Also verifies that the option "usejac" exists for the solver.
        #(Only check for FMI2)
        if self.with_jacobian and "usejac" not in solver_options:
            try:
                getattr(self.simulator, "usejac")
                solver_options["usejac"] = True
            except AttributeError:
                pass

        #Override usejac if there are no states
        fnbr, gnbr = self.model.get_ode_sizes()
        if "usejac" in solver_options and fnbr == 0:
            solver_options["usejac"] = False

        if "maxh" in solver_options and solver_options["maxh"] == "Default":
            if self.options["ncp"] == 0:
                solver_options["maxh"] = 0.0
            else:
                solver_options["maxh"] = float(self.final_time - self.start_time) / float(self.options["ncp"])
        
        if "rtol" in solver_options:
            rtol_is_vector      = (isinstance(self.solver_options["rtol"], N.ndarray) or isinstance(self.solver_options["rtol"], list))
            rtol_vector_support = self.simulator.supports.get("rtol_as_vector", False)
            
            if rtol_is_vector and not rtol_vector_support and self._rtol_as_scalar_fallback:
                logging_module.warning("The chosen solver does not support providing the relative tolerance as a vector, fallback to using a scalar instead. rtol = %g"%self.rtol)
                solver_options["rtol"] = self.rtol

        #loop solver_args and set properties of solver
        for k, v in solver_options.items():
            try:
                getattr(self.simulator,k)
            except AttributeError:
                try:
                    getattr(self.probl,k)
                except AttributeError:
                    raise InvalidSolverArgumentException(k)
                setattr(self.probl, k, v)
                continue
            try:
                setattr(self.simulator, k, v)
            except Exception as e:
                raise fmi.InvalidOptionException("Failed to set the solver option '%s' with msg: %s"%(k, str(e))) from None

        #Needs to be set as last option in order to have an impact.
        if "maxord" in solver_options:
            setattr(self.simulator, "maxord", solver_options["maxord"])

    def solve(self):
        """
        Runs the simulation.
        """
        time_start = timer()

        try:
            self.simulator.simulate(self.final_time, self.ncp)
        except Exception:
            self.result_handler.simulation_end() #Close the potentially open result files
            raise #Reraise the exception

        self.timings["storing_result"] = self.probl.timings["handle_result"]
        self.timings["computing_solution"] = timer() - time_start - self.timings["storing_result"]


    def get_result(self):
        """
        Write result to file, load result data and create an AssimuloSimResult
        object.

        Returns::

            The AssimuloSimResult object.
        """
        time_start = timer()

        if self.options["return_result"]:
            #Retrieve result
            res = self.result_handler.get_result()
        else:
            res = None

        end_time = timer()
        self.timings["returning_result"] = end_time - time_start
        self.timings["other"] = end_time - self.time_start_total- sum(self.timings.values())
        self.timings["total"] = end_time - self.time_start_total

        # create and return result object
        return FMIResult(self.model, self.result_file_name, self.simulator,
            res, self.options, detailed_timings=self.timings)

    @classmethod
    def get_default_options(cls):
        """
        Get an instance of the options class for the AssimuloFMIAlg algorithm,
        prefilled with default values. (Class method.)
        """
        return AssimuloFMIAlgOptions()


class FMICSAlgOptions(OptionBase):
    """
    Options for the solving the CS FMU.

    Options::


        ncp    --
            Number of communication points.
            Default: '500'

        initialize --
            If set to True, the initializing algorithm defined in the FMU model
            is invoked, otherwise it is assumed the user have manually invoked
            model.initialize()
            Default is True.

        stop_time_defined --
            If set to True, the model cannot be computed past the set final_time,
            even in a continuation run. This is only applicable when initialize
            is set to True. For more information, see the FMI specification.
            Default False.

        write_scaled_result --
            Set this parameter to True to write the result to file without
            taking scaling into account. If the value of scaled is False,
            then the variable scaling factors of the model are used to
            reproduced the unscaled variable values.
            Default: False

        result_file_name --
            Specifies the name of the file where the simulation result is
            written. Setting this option to an empty string results in a default
            file name that is based on the name of the model class.
            result_file_name can also be set to a stream that supports 'write',
            'tell' and 'seek'.
            Default: Empty string

        result_handling --
            Specifies how the result should be handled. Either stored to
            file (txt or binary) or stored in memory. One can also use a
            custom handler.
            Available options: "file", "binary", "memory", "csv", "custom", None
            Default: "binary"

        result_handler --
            The handler for the result. Depending on the option in
            result_handling this either defaults to ResultHandlerFile
            or ResultHandlerMemory. If result_handling custom is chosen
            This MUST be provided.
            Default: None

        return_result --
            Determines if the simulation result should be returned or
            not. If set to False, the simulation result is not loaded
            into memory after the simulation finishes.
            Default: True

        result_store_variable_description --
            Determines if the description for the variables should be
            stored in the result file or not. Only impacts the result
            file formats that supports storing the variable description
            ("file" and "binary").
            Default: True

        time_limit --
            Specifies an upper bound on the time allowed for the
            integration to be completed. The time limit is specified
            in seconds. Note that the time limit is only checked after
            a completed step. This means that if a do step takes a lot
            of time, the execution will not stop at exactly the time
            limit.
            Default: none

        filter --
            A filter for choosing which variables to actually store
            result for. The syntax can be found in
            http://en.wikipedia.org/wiki/Glob_%28programming%29 . An
            example is filter = "*der" , stor all variables ending with
            'der'. Can also be a list.
            Default: None

        silent_mode --
            Disables printouts to the console.
            Default: False
        
        synchronize_simulation --
            If set, the simulation will be synchronized to real-time or a
            scaled real-time, if possible. The available options are:
                True: Simulation is synchronized to real-time
                False: No synchronization
                >0 (float): Simulation is synchronized to the factored 
                            real-time. I.e. factor*real-time
                
                Example: If, set to 10: 10 simulated seconds is synchronized
                         to one real-time second.
            Default: False

    """
    def __init__(self, *args, **kw):
        _defaults= {
            'ncp':500,
            'initialize':True,
            'stop_time_defined': False,
            'write_scaled_result':False,
            'result_file_name':'',
            'result_handling':"binary",
            'result_handler': None,
            'result_store_variable_description': True,
            'return_result': True,
            'time_limit': None,
            'filter':None,
            'silent_mode':False,
            'synchronize_simulation':False
            }
        super(FMICSAlgOptions,self).__init__(_defaults)
        # for those key-value-sets where the value is a dict, don't
        # overwrite the whole dict but instead update the default dict
        # with the new values
        self._update_keep_dict_defaults(*args, **kw)

class FMICSAlg(AlgorithmBase):
    """
    Simulation algorithm for FMUs (Co-simulation).
    """

    def __init__(self,
                 start_time,
                 final_time,
                 input,
                 model,
                 options):
        """
        Simulation algorithm for FMUs (Co-simulation).

        Parameters::

            model --
                fmi.FMUModelCS1 object representation of the model.

            options --
                The options that should be used in the algorithm. For details on
                the options, see:

                * model.simulate_options('FMICSAlgOptions')

                or look at the docstring with help:

                * help(pyfmi.fmi_algorithm_drivers.FMICSAlgOptions)

                Valid values are:
                - A dict that overrides some or all of the default values
                  provided by FMICSAlgOptions. An empty dict will thus
                  give all options with default values.
                - FMICSAlgOptions object.
        """
        self.model = model
        self.timings = {}
        self.time_start_total = timer()

        # set start time, final time and input trajectory
        self.start_time = start_time
        self.final_time = final_time
        self.input = input

        self.status = 0

        # handle options argument
        if isinstance(options, dict) and not \
            isinstance(options, FMICSAlgOptions):
            # user has passed dict with options or empty dict = default
            self.options = FMICSAlgOptions(options)
        elif isinstance(options, FMICSAlgOptions):
            # user has passed FMICSAlgOptions instance
            self.options = options
        else:
            raise InvalidAlgorithmOptionException(options)

        # set options
        self._set_options()

        input_traj = None
        if self.input:
            if hasattr(self.input[1],"__call__"):
                input_traj=(self.input[0],
                        TrajectoryUserFunction(self.input[1]))
            else:
                input_traj=(self.input[0],
                        TrajectoryLinearInterpolation(self.input[1][:,0],
                                                      self.input[1][:,1:]))
            #Sets the inputs, if any
            self.model.set(input_traj[0], input_traj[1].eval(self.start_time)[0,:])
        self.input_traj = input_traj

        #time_start = timer()

        self.result_handler = get_result_handler(self.model, self.options)
        self.result_handler.set_options(self.options)

        time_end = timer()
        #self.timings["creating_result_object"] = time_end - time_start
        time_start = time_end
        time_res_init = 0.0

        # Initialize?
        if self.options['initialize']:
            if isinstance(self.model, ((fmi.FMUModelCS1, fmi_extended.FMUModelME1Extended))):
                self.model.initialize(start_time, final_time, stop_time_defined=self.options["stop_time_defined"])

            elif isinstance(self.model, fmi.FMUModelCS2):
                self.model.setup_experiment(start_time=start_time, stop_time_defined=self.options["stop_time_defined"], stop_time=final_time)
                self.model.initialize()

            else:
                raise fmi.FMUException("Unknown model.")

            time_res_init = timer()
            self.result_handler.initialize_complete()
            time_res_init = timer() - time_res_init

        elif self.model.time is None and isinstance(self.model, fmi.FMUModelCS2):
            raise fmi.FMUException("Setup Experiment has not been called, this has to be called prior to the initialization call.")
        elif self.model.time is None:
            raise fmi.FMUException("The model need to be initialized prior to calling the simulate method if the option 'initialize' is set to False")

        if abs(start_time - model.time) > 1e-14:
            logging_module.warning('The simulation start time (%f) and the current time in the model (%f) is different. Is the simulation start time correctly set?'%(start_time, model.time))

        time_end = timer()
        self.timings["initializing_fmu"] = time_end - time_start - time_res_init
        time_start = time_end

        self.result_handler.simulation_start()

        self.timings["initializing_result"] = timer() - time_start - time_res_init

    def _set_options(self):
        """
        Helper function that sets options for FMICS algorithm.
        """
        # no of communication points
        if self.options['ncp'] <= 0:
            raise fmi.FMUException(f"Setting {self.options['ncp']} as 'ncp' is not allowed for a CS FMU. Must be greater than 0.")
        self.ncp = self.options['ncp']

        self.write_scaled_result = self.options['write_scaled_result']

        # result file name
        if self.options['result_file_name'] == '':
            self.result_file_name = self.model.get_identifier()+'_result.txt'
        else:
            self.result_file_name = self.options['result_file_name']
            
        if self.options["synchronize_simulation"]:
            try:
                if self.options["synchronize_simulation"] is True:
                    self._synchronize_factor = 1.0
                elif self.options["synchronize_simulation"] > 0:
                    self._synchronize_factor = self.options["synchronize_simulation"]
                else:
                    raise fmi.InvalidOptionException(f"Setting {self.options['synchronize_simulation']} as 'synchronize_simulation' is not allowed. Must be True/False or greater than 0.")
            except Exception:
                raise fmi.InvalidOptionException(f"Setting {self.options['synchronize_simulation']} as 'synchronize_simulation' is not allowed. Must be True/False or greater than 0.")
        else:
            self._synchronize_factor = 0.0
        
    def _set_solver_options(self):
        """
        Helper function that sets options for the solver.
        """
        pass #No solver options

    def solve(self):
        """
        Runs the simulation.
        """
        result_handler = self.result_handler
        h = (self.final_time-self.start_time)/self.ncp
        grid = N.linspace(self.start_time,self.final_time,self.ncp+1)[:-1]

        status = 0
        final_time = self.start_time

        #For result writing
        start_time_point = timer()
        result_handler.integration_point()
        self.timings["storing_result"] = timer() - start_time_point

        #Start of simulation, start the clock
        time_start = timer()

        for t in grid:
            if self._synchronize_factor > 0:
                under_run = t/self._synchronize_factor - (timer()-time_start)
                if under_run > 0:
                    time.sleep(under_run)
            
            status = self.model.do_step(t,h)
            self.status = status

            if status != 0:

                if status == fmi.FMI_ERROR:
                    result_handler.simulation_end()
                    raise fmi.FMUException("The simulation failed. See the log for more information. Return flag %d."%status)

                elif status == fmi.FMI_DISCARD and (isinstance(self.model, fmi.FMUModelCS1) or
                                                    isinstance(self.model, fmi.FMUModelCS2)):

                    try:
                        if isinstance(self.model, fmi.FMUModelCS1):
                            last_time = self.model.get_real_status(fmi.FMI1_LAST_SUCCESSFUL_TIME)
                        else:
                            last_time = self.model.get_real_status(fmi.FMI2_LAST_SUCCESSFUL_TIME)
                        if last_time > t: #Solver succeeded in taken a step a little further than the last time
                            self.model.time = last_time
                            final_time = last_time

                            start_time_point = timer()
                            result_handler.integration_point()
                            self.timings["storing_result"] += timer() - start_time_point
                    except fmi.FMUException:
                        pass
                break
                #result_handler.simulation_end()
                #raise Exception("The simulation failed. See the log for more information. Return flag %d"%status)

            final_time = t+h

            start_time_point = timer()
            result_handler.integration_point()
            self.timings["storing_result"] += timer() - start_time_point

            if self.options["time_limit"] and (timer() - time_start) > self.options["time_limit"]:
                raise fmi.TimeLimitExceeded("The time limit was exceeded at integration time %.8E."%final_time)

            if self.input_traj is not None:
                self.model.set(self.input_traj[0], self.input_traj[1].eval(t+h)[0,:])

        #End of simulation, stop the clock
        time_stop = timer()

        result_handler.simulation_end()

        if self.status != 0:
            if not self.options["silent_mode"]:
                print('Simulation terminated prematurely. See the log for possibly more information. Return flag %d.'%status)

        #Log elapsed time
        if not self.options["silent_mode"]:
            print('Simulation interval    : ' + str(self.start_time) + ' - ' + str(final_time) + ' seconds.')
            print('Elapsed simulation time: ' + str(time_stop-time_start) + ' seconds.')

        self.timings["computing_solution"] = time_stop - time_start - self.timings["storing_result"]

    def get_result(self):
        """
        Write result to file, load result data and create an FMICSResult
        object.

        Returns::

            The FMICSResult object.
        """
        time_start = timer()

        if self.options["return_result"]:
            # Get the result
            res = self.result_handler.get_result()
        else:
            res = None

        end_time = timer()
        self.timings["returning_result"] = end_time - time_start
        self.timings["other"] = end_time - self.time_start_total- sum(self.timings.values())
        self.timings["total"] = end_time - self.time_start_total

        # create and return result object
        return FMIResult(self.model, self.result_file_name, None,
            res, self.options, status=self.status, detailed_timings=self.timings)

    @classmethod
    def get_default_options(cls):
        """
        Get an instance of the options class for the FMICSAlg algorithm,
        prefilled with default values. (Class method.)
        """
        return FMICSAlgOptions()


class SciEstAlg(AlgorithmBase):
    """
    Estimation algorithm for FMUs.
    """

    def __init__(self,
                 parameters,
                 measurements,
                 input,
                 model,
                 options):
        """
        Estimation algorithm for FMUs.

        Parameters::

            model --
                fmi.FMUModel* object representation of the model.

            options --
                The options that should be used in the algorithm. For details on
                the options, see:

                * model.simulate_options('SciEstAlgOptions')

                or look at the docstring with help:

                * help(pyfmi.fmi_algorithm_drivers.SciEstAlgAlgOptions)

                Valid values are:
                - A dict that overrides some or all of the default values
                  provided by SciEstAlgOptions. An empty dict will thus
                  give all options with default values.
                - SciEstAlgOptions object.
        """
        self.model = model

        # set start time, final time and input trajectory
        self.parameters = parameters
        self.measurements = measurements
        self.input = input

        # handle options argument
        if isinstance(options, dict) and not \
            isinstance(options, SciEstAlgOptions):
            # user has passed dict with options or empty dict = default
            self.options = SciEstAlgOptions(options)
        elif isinstance(options, SciEstAlgOptions):
            # user has passed FMICSAlgOptions instance
            self.options = options
        else:
            raise InvalidAlgorithmOptionException(options)

        # set options
        self._set_options()

        self.result_handler = get_result_handler(self.model, self.options)
        self.result_handler.set_options(self.options)
        self.result_handler.initialize_complete()

    def _set_options(self):
        """
        Helper function that sets options for FMICS algorithm.
        """
        self.options["filter"] = self.parameters

        if isinstance(self.options["scaling"], str) and self.options["scaling"] == "Default":
            scale = []
            for parameter in self.parameters:
                scale.append(self.model.get_variable_nominal(parameter))
            self.options["scaling"] = N.array(scale)

        if self.options["simulate_options"] == "Default":
            self.options["simulate_options"] = self.model.simulate_options()

        # Modify necessary options:
        self.options["simulate_options"]['ncp']    = self.measurements[1].shape[0] - 1 #Store at the same points as measurement data
        self.options["simulate_options"]['filter'] = self.measurements[0] #Only store the measurement variables (efficiency)

        if "solver" in self.options["simulate_options"]:
            solver = self.options["simulate_options"]["solver"]

            self.options["simulate_options"][solver+"_options"]["verbosity"] = 50 #Disable printout (efficiency)
            self.options["simulate_options"][solver+"_options"]["store_event_points"] = False #Disable extra store points

    def _set_solver_options(self):
        """
        Helper function that sets options for the solver.
        """
        pass

    def solve(self):
        """
        Runs the estimation.
        """
        import scipy as sci
        import scipy.optimize as sciopt
        from pyfmi.fmi_util import parameter_estimation_f

        #Define callback
        global niter
        niter = 0
        def parameter_estimation_callback(y):
            global niter
            if niter % 10 == 0:
                print("  iter    parameters ")
            #print '{:>5d} {:>15e}'.format(niter+1, parameter_estimation_f(y, self.parameters, self.measurements, self.model, self.input, self.options))
            print('{:>5d} '.format(niter+1) + str(y))
            niter += 1

        #End of simulation, stop the clock
        time_start = timer()

        p0 = []
        for i,parameter in enumerate(self.parameters):
            p0.append(self.model.get(parameter)/self.options["scaling"][i])

        print('\nRunning solver: ' + self.options["method"])
        print(' Initial parameters (scaled): ' + str(N.array(p0).flatten()))
        print(' ')

        res = sciopt.minimize(parameter_estimation_f, p0,
                                args=(self.parameters, self.measurements, self.model, self.input, self.options),
                                method=self.options["method"],
                                bounds=None,
                                constraints=(),
                                tol=self.options["tolerance"],
                                callback=parameter_estimation_callback)

        for i in range(len(self.parameters)):
            res["x"][i] = res["x"][i]*self.options["scaling"][i]

        self.res = res
        self.status = res["success"]

        #End of simulation, stop the clock
        time_stop = timer()

        if not res["success"]:
            print('Estimation failed: ' + res["message"])
        else:
            print('\nEstimation terminated successfully!')
            print(' Found parameters: ' + str(res["x"]))

        print('Elapsed estimation time: ' + str(time_stop-time_start) + ' seconds.\n')

    def get_result(self):
        """
        Write result to file, load result data and create an SciEstResult
        object.

        Returns::

            The SciEstResult object.
        """
        for i,parameter in enumerate(self.parameters):
            self.model.set(parameter, self.res["x"][i])

        self.result_handler.simulation_start()

        self.model.time = self.measurements[1][0,0]
        self.result_handler.integration_point()

        self.result_handler.simulation_end()

        self.model.reset()

        for i,parameter in enumerate(self.parameters):
            self.model.set(parameter, self.res["x"][i])

        return FMIResult(self.model, self.options["result_file_name"], None,
            self.result_handler.get_result(), self.options, status=self.status)

    @classmethod
    def get_default_options(cls):
        """
        Get an instance of the options class for the SciEstAlg algorithm,
        prefilled with default values. (Class method.)
        """
        return SciEstAlgOptions()

class SciEstAlgOptions(OptionBase):
    """
    Options for the solving an estimation problem.

    Options::

        tolerance    --
            The tolerance for the estimation algorithm
            Default: 1e-6

        method       --
            The method to use, available methods are methods from:
            scipy.optimize.minimize.
            Default: 'Nelder-Mead'

        scaling      --
            The scaling of the parameters during the estimation.
            Default: The nominal values

        simulate_options    --
            The simulation options to use when simulating the model
            in order to get the estimated data.
            Default: The default options for the underlying model.

        result_file_name --
            Specifies the name of the file where the result is written.
            Setting this option to an empty string results in a default
            file name that is based on the name of the model class.
            result_file_name can also be set to a stream that supports 'write',
            'tell' and 'seek'.
            Default: Empty string
        
        result_handling --
            Specifies how the result should be handled. Either stored to
            file (txt or binary) or stored in memory. One can also use a
            custom handler.
            Available options: "file", "binary", "memory", "csv", "custom", None
            Default: "csv"

        result_handler --
            The handler for the result. Depending on the option in
            result_handling this either defaults to ResultHandlerFile
            or ResultHandlerMemory. If result_handling custom is chosen
            This MUST be provided.
            Default: None

    """
    def __init__(self, *args, **kw):
        _defaults= {"tolerance": 1e-6,
                    'result_file_name':'',
                    'result_handling':'csv',
                    'result_handler':None,
                    'filter':None,
                    'method': 'Nelder-Mead',
                    'scaling': 'Default',
                    'simulate_options': "Default"}
        super(SciEstAlgOptions,self).__init__(_defaults)
        # for those key-value-sets where the value is a dict, don't
        # overwrite the whole dict but instead update the default dict
        # with the new values
        self._update_keep_dict_defaults(*args, **kw)
