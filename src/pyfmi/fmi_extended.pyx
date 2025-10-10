#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Copyright (C) 2014 Modelon AB
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

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""
Module containing the FMI interface Python wrappers.
"""

import logging
import numpy as np
cimport numpy as np
np.import_array()

cimport pyfmi.fmil_import as FMIL
cimport pyfmi.fmil1_import as FMIL1
cimport pyfmi.fmi1 as FMI1
from pyfmi.fmi_base import FMI_DEFAULT_LOG_LEVEL
from pyfmi.fmi1 import FMI_OK
from pyfmi.exceptions import FMUException


cdef class FMUModelME1Extended(FMI1.FMUModelME1):

    cdef public object _explicit_problem, _solver
    cdef public object _current_time, _stop_time, _input_time
    cdef public list _input_value_refs, _input_alias_type, _input_factorials
    cdef public int _input_active, _input_order
    cdef public dict _input_derivatives, _options
    cdef public np.ndarray _input_tmp

    def __init__(self, fmu, log_file_name="", log_level=FMI_DEFAULT_LOG_LEVEL, _connect_dll=True):
        #Instantiate the FMU
        FMI1.FMUModelME1.__init__(self, fmu = fmu, log_file_name = log_file_name, log_level = log_level, _connect_dll=_connect_dll)
        
        nbr_f, nbr_g = self.get_ode_sizes()

        vars = self.get_model_variables(include_alias=False, causality=0)
        input_value_refs = []
        input_alias_type = []
        for name in vars.keys():
            input_value_refs.append(self.get_variable_valueref(name))
            input_alias_type.append(-1.0 if self.get_variable_alias(name)[name] == -1 else 1.0)
        assert not np.any(input_alias_type == -1)
        
        self._input_derivatives = {}
        for val_ref in input_value_refs:
            self._input_derivatives[val_ref] = [0.0, 0.0, 0.0, 0.0]
        self._input_factorials = [1.0, 2.0, 6.0, 24.0]
        
        #Default values
        self._input_value_refs = input_value_refs
        self._input_alias_type = input_alias_type
        self._input_active = 0
        self._input_order = 0
        self._options = {"solver":"CVode","CVode_options":{}}
    
    def get_solver_options(self):
        """
        Returns the default solver options
        """
        return self._options
    
    def set_solver_options(self, options):
        """
        Specifies the underlying solver and the options to the specified
        solver.
        """
        self._options.update(options)

    def _set_solver_options(self):
        try:
            solver_options = self._options[self._options["solver"]+"_options"]
        except KeyError:
            return

        #loop solver_args and set properties of solver
        for k, v in solver_options.iteritems():
            try:
                getattr(self._solver,k)
            except AttributeError:
                getattr(self._explicit_problem,k)
                setattr(self._explicit_problem, k, v)
                continue
            setattr(self._solver, k, v)
        
    cdef void _eval_input(self, double time):
        cdef int i,j
        cdef np.ndarray input_tmp = self._input_tmp.copy()
        
        for i,ref in enumerate(self._input_value_refs):
            for j in range(self._input_order):
                input_tmp[i] += (time - self._input_time)**(j+1) * self._input_derivatives[ref][j]

        self.set_real(self._input_value_refs, input_tmp)
        
    cdef _reset_input_derivatives(self):
        for ref in self._input_value_refs:
            for i in range(self._input_order):
                self._input_derivatives[ref][i] = 0.0
    
    cpdef get_derivatives(self):
        """
        Returns the derivative of the continuous states.

        Returns::

            dx --
                The derivative as an array.

        Example::

            dx = model.get_derivatives()

        Calls the low-level FMI function: fmiGetDerivatives
        """
        cdef int status
        cdef np.ndarray[FMIL1.fmi1_real_t, ndim=1,mode='c'] values = np.empty(self._nContinuousStates,dtype=np.double)
        
        if self._input_active:
            self._eval_input(self.time)
        
        status = FMIL1.fmi1_import_get_derivatives(self._fmu, <FMIL1.fmi1_real_t*>values.data, self._nContinuousStates)

        if status != 0:
            raise FMUException('Failed to get the derivative values.')

        return values

    
    def do_step(self, FMIL1.fmi1_real_t current_time, FMIL1.fmi1_real_t step_size, new_step=True):
        """
        Performs an integrator step.

        Parameters::

            current_time --
                    The current communication point (current time) of
                    the master.
            step_size --
                    The length of the step to be taken.
            new_step --
                    True the last step was accepted by the master and
                    False if not.

        Returns::

            status --
                    The status of function which can be checked against
                    FMI_OK, FMI_WARNING. FMI_DISCARD, FMI_ERROR,
                    FMI_FATAL,FMI_PENDING...

        Calls the underlying low-level function fmiDoStep.
        """
        cdef int status
        cdef FMIL1.fmi1_boolean_t new_s
        cdef double time

        time = current_time+step_size
        
        self._input_time = current_time
        
        if self._input_active:
            self._input_tmp = self.get_real(self._input_value_refs)
        
        if step_size == 0.0:
            self.event_update()
        else:
            t,y = self._solver.simulate(time, 1)
            """
            if new_step:
                flag, t, y = self._solver.do_step(tmid=self.time,tf=self.time, initialize=True)
            else:
                try:
                    flag, t, y = self._solver.do_step(tmid=self.time,tf=self.time, initialize=False)
                except Exception:
                    print "Warning: Failed to calculate the solution at %g"%current_time
                    print "Warning: Re-initializing..."
                    flag, t, y = self._solver.do_step(tmid=self.time,tf=self.time, initialize=True)
            """
        
        if self._input_active:
            self._reset_input_derivatives()
        
        #Always deactivate inputs after a step
        self._input_active = 0
        self._input_order = 0

        return FMI_OK


    def cancel_step(self):
        """
        Cancel a current integrator step. Can only be called if the
        status from do_step returns FMI_PENDING.
        """
        raise NotImplementedError

    def get_output_derivatives(self, variables, FMIL1.fmi1_integer_t order):
        """
        Returns the output derivatives for the specified variables. The
        order specifies the nth-derivative.

        Parameters::

                variables --
                        The variables for which the output derivatives
                        should be returned.
                order --
                        The derivative order.
        """
        raise FMUException("Not Implemented.")

    def _get_types_platform(self):
        """
        Returns the set of valid compatible platforms for the Model, extracted
        from the XML.

        Returns::

            types_platform --
                The valid platforms.

        Example::

            model.types_platform
        """
        return self.model_types_platform

    types_platform = property(fget=_get_types_platform)
    
    def get_fmu_state(self):
        """
        Copies the internal state of the FMU and returns a class which
        can be used to again set the state of the FMU at a later time.
        
        
        note::
            
            fmiStatus fmiGetFMUstate (fmiComponent c, fmiFMUstate* FMUstate);
            
        """
        return self._solver.dump_solver_state()
    
    def set_fmu_state(self, fmu_state):
        """
        Sets the internal state of the FMU using a class that has been
        returned from the method get_fmu_state.
        
        note::
        
            fmiStatus fmiSetFMUstate (fmiComponent c, fmiFMUstate  FMUstate);
            
        """
        self._solver.apply_solver_state(fmu_state)

    def set_input_derivatives(self, variables, values, orders):
        """
        Sets the input derivative order for the specified variables.

        Parameters::

                variables --
                        The variables for which the input derivative
                        should be set.
                values --
                        The actual values.
                order --
                        The derivative orders to set.
        """
        cdef int status
        cdef FMIL.size_t nref
        cdef np.ndarray[FMIL1.fmi1_integer_t, ndim=1,mode='c'] np_orders = np.array(orders, dtype=np.int32, ndmin=1).flatten()
        cdef np.ndarray[FMIL1.fmi1_value_reference_t, ndim=1,mode='c'] value_refs
        cdef np.ndarray[FMIL1.fmi1_real_t, ndim=1,mode='c'] val = np.array(values, dtype=float, ndmin=1).flatten()

        nref = len(val)
        orders = np.array([0]*nref, dtype=np.int32)

        if nref != len(np_orders):
            raise FMUException("The number of variables must be the same as the number of orders.")

        for i in range(len(np_orders)):
            if np_orders[i] < 1:
                raise FMUException("The order must be greater than zero.")
            if np_orders[i] > self._input_order:
                self._input_order = np_orders[i]

        if isinstance(variables,str):
            value_refs = np.array([0], dtype=np.uint32,ndmin=1).flatten()
            value_refs[0] = self.get_variable_valueref(variables)
        elif isinstance(variables,list) and isinstance(variables[-1],str):
            value_refs = np.array([0]*nref, dtype=np.uint32,ndmin=1).flatten()
            for i in range(nref):
                value_refs[i] = self.get_variable_valueref(variables[i])
        else:
            raise FMUException("The variables must either be a string or a list of strings")
        
        for i in range(nref):
            self._input_derivatives[value_refs[i]][np_orders[i]-1] = val[i]/self._input_factorials[np_orders[i]-1]
            
        #Activate input
        self._input_active = 1

    def simulate(self,
                 start_time='Default',
                 final_time='Default',
                 input=(),
                 algorithm='FMICSAlg',
                 options={}):
        """
        Compact function for model simulation.

        The simulation method depends on which algorithm is used, this can be
        set with the function argument 'algorithm'. Options for the algorithm
        are passed as option classes or as pure dicts. See
        FMUModel.simulate_options for more details.

        The default algorithm for this function is FMICSAlg.

        Parameters::

            start_time --
                Start time for the simulation.
                Default: Start time defined in the default experiment from
                        the ModelDescription file.

            final_time --
                Final time for the simulation.
                Default: Stop time defined in the default experiment from
                        the ModelDescription file.

            input --
                Input signal for the simulation. The input should be a 2-tuple
                consisting of first the names of the input variable(s) and then
                the data matrix.
                Default: Empty tuple.

            algorithm --
                The algorithm which will be used for the simulation is specified
                by passing the algorithm class as string or class object in this
                argument. 'algorithm' can be any class which implements the
                abstract class AlgorithmBase (found in algorithm_drivers.py). In
                this way it is possible to write own algorithms and use them
                with this function.
                Default: 'FMICSAlg'

            options --
                The options that should be used in the algorithm. For details on
                the options do:

                    >> myModel = FMUModel(...)
                    >> opts = myModel.simulate_options()
                    >> opts?

                Valid values are:
                    - A dict which gives AssimuloFMIAlgOptions with
                      default values on all options except the ones
                      listed in the dict. Empty dict will thus give all
                      options with default values.
                    - An options object.
                Default: Empty dict

        Returns::

            Result object, subclass of common.algorithm_drivers.ResultBase.
        """
        if start_time == "Default":
            start_time = self.get_default_experiment_start_time()
        if final_time == "Default":
            final_time = self.get_default_experiment_stop_time()

        return self._exec_simulate_algorithm(start_time,
                                             final_time,
                                             input,
                                             'pyfmi.fmi_algorithm_drivers',
                                             algorithm,
                                             options)

    def simulate_options(self, algorithm='FMICSAlg'):
        """
        Get an instance of the simulate options class, prefilled with default
        values. If called without argument then the options class for the
        default simulation algorithm will be returned.

        Parameters::

            algorithm --
                The algorithm for which the options class should be fetched.
                Possible values are: 'FMICSAlg'.
                Default: 'FMICSAlg'

        Returns::

            Options class for the algorithm specified with default values.
        """
        return self._default_options('pyfmi.fmi_algorithm_drivers', algorithm)

    def initialize(self, start_time=0.0, stop_time=1.0, stop_time_defined=False):
        """
        Initializes the slave.

        Parameters::

            start_time --
                Start time of the simulation.
                Default: The start time defined in the model description.
            
            stop_time --
                Stop time of the simulation.
                Default: The stop time defined in the model description.
            
            stop_time_defined --
                Defines if a fixed stop time is defined or not. If this is
                set the simulation cannot go past the defined stop time.
                Default: False

        Calls the low-level FMU function: fmiInstantiateSlave
        """
        cdef char tolerance_controlled
        cdef FMIL1.fmi1_real_t tolerance

        self.time = start_time

        tolerance_controlled = 1
        tolerance = FMIL1.fmi1_import_get_default_experiment_tolerance(self._fmu)

        status = FMIL1.fmi1_import_initialize(self._fmu, tolerance_controlled, tolerance, &self._eventInfo)

        if status == 1:
            if self._enable_logging:
                logging.warning(
                    'Initialize returned with a warning.' \
                    ' Check the log for information (FMUModel.get_log).')
            else:
                logging.warning('Initialize returned with a warning.' \
                    ' Increase log_level for more information, (FMUModel(..., log_level=...)).')

        if status > 1:
            if self._enable_logging:
                raise FMUException(
                    'Initialize returned with a error.' \
                    ' Check the log for information (FMUModel.get_log).')
            else:
                raise FMUException('Initialize returned with a error.' \
                    ' Increase log_level for more information, (FMUModel(..., log_level=...)).')

        self._allocated_fmu = True
        
        #Create an assimulo problem
        from pyfmi.simulation.assimulo_interface_fmi1 import FMIODE
        self._explicit_problem = FMIODE(self, input=None, result_file_name = '',
                                        with_jacobian=False, start_time=start_time)
        
        #Import the solvers
        from assimulo import solvers
        
        #Create an assimulo solver
        self._solver = getattr(solvers, self._options["solver"])(self._explicit_problem)
        
        #Set options
        self._solver.verbosity = 50
        self._solver.report_continuously = True
        
        try:
            self._solver.rtol = tolerance
            self._solver.atol = tolerance*0.01*(self.nominal_continuous_states if self.get_ode_sizes()[0] > 0 else 1)
        except AttributeError:
            pass
        
        #Set user defined solver options
        self._set_solver_options()
        
        #Initialize CVode
        #self._solver.initialize() 
        #self._solver.initialize_options()
    
        #Store the start and stop time
        self._current_time = start_time
        self._stop_time = stop_time

    def instantiate_slave(self, name='Slave', logging=False):
        """
        Instantiate the slave.

        Parameters::

            name --
                The name of the instance.
                Default: 'Slave'

            logging --
                Defines if the logging should be turned on or off.
                Default: False, no logging.

        Calls the low-level FMI function: fmiInstantiateSlave.
        """
        self.instantiate_model(logging = logging)

    def get_capability_flags(self):
        """
        Returns a dictionary with the cability flags of the FMU.

        Capabilities::

            canHandleVariableCommunicationStepSize
            canHandleEvents
            canRejectSteps
            canInterpolateInputs
            maxOutputDerivativeOrder
            canRunAsynchronuously
            canSignalEvents
            canBeinstantiatedOnlyOncePerProcess
            canNotUseMemoryManagementFunctions
        """
        cdef dict capabilities = {}

        capabilities["canHandleVariableCommunicationStepSize"] = True
        capabilities["canHandleEvents"] = True
        capabilities["canRejectSteps"] = True
        capabilities["canInterpolateInputs"] = False
        capabilities["maxOutputDerivativeOrder"] = 0
        capabilities["canRunAsynchronuously"] = False
        capabilities["canSignalEvents"] = False
        capabilities["canBeInstantiatedOnlyOncePerProcess"] = False
        capabilities["canNotUseMemoryManagementFunctions"] = True

        return capabilities
