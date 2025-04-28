#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2025 Modelon AB
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
This file contains code for mapping FMI3 FMUs to the Problem specifications
required by Assimulo.
"""

import numpy as np
cimport numpy as np

import time
import logging as logging_module
from operator import index
import scipy.sparse as sps
from timeit import default_timer as timer

cimport pyfmi.fmil_import as FMIL
cimport pyfmi.fmil3_import as FMIL3
cimport pyfmi.fmi3 as FMI3
from pyfmi.exceptions import (
    FMUException,
    InvalidOptionException,
    FMIModel_Exception,
    FMIModelException
)

from assimulo.problem cimport cExplicit_Problem
from assimulo.exception import AssimuloRecoverableError, TerminateSimulation

cdef class FMIODE3(cExplicit_Problem):
    """ An Assimulo Explicit Model extended to FMI3 interface. """

    def __init__(self, model, input = None, result_file_name = "",
                 with_jacobian = False, start_time = 0.0, logging = False,
                 result_handler = None, extra_equations = None, synchronize_simulation = False,
                 number_of_diagnostics_variables = 0):
        # Initialize the problem.
        self._model = model
        self._adapt_input(input)
        self.input_names = []
        self.timings = {"handle_result": 0.0}

        if type(model) == FMI3.FMUModelME3: # if isinstance(model, FMUModelME3):
            self.model_me3 = model
            self.model_me3_instance = 1
        else:
            self.model_me3_instance = 0

        # Set start time to the model
        self._model.time = start_time

        self.t0 = start_time
        self.y0 = self._model.continuous_states
        self.problem_name = self._model.get_name()

        [f_nbr, g_nbr] = self._model.get_ode_sizes()

        self._f_nbr = f_nbr
        self._g_nbr = g_nbr
        self._A = None

        # Used by Assimulo
        self.state_events_use = g_nbr > 0
        self.time_events_use = True

        # If there is no state in the model, add a dummy state der(y)=0
        if f_nbr == 0:
            self.y0 = np.array([0.0])

        # Determine the result file name
        if result_file_name == '':
            self.result_file_name = model.get_name()+'_result.txt'
        else:
            self.result_file_name = result_file_name

        # Default values
        self.export = result_handler

        # Internal values
        self._logg_step_event = []
        self._write_header = True
        self._logging = logging
        self._sparse_representation = False
        # self._with_jacobian = with_jacobian # TODO
        self._with_jacobian = False

        # # If result handler support is available, logging turns into dynamic_diagnostics
        self._logging_as_dynamic_diagnostics = self._logging and result_handler.supports.get("dynamic_diagnostics", False)
        self._number_of_diagnostics_variables = number_of_diagnostics_variables

        self.jac_use = False
        # if f_nbr > 0 and with_jacobian:
        #     self.jac_use = True # Activates the jacobian

        #     # Need to calculate the nnz.
        #     [derv_state_dep, derv_input_dep] = model.get_derivatives_dependencies()
        #     self.jac_nnz = np.sum([len(derv_state_dep[key]) for key in derv_state_dep.keys()])+f_nbr

        # if extra_equations:
        #     self._extra_f_nbr = extra_equations.get_size()
        #     self._extra_y0    = extra_equations.y0
        #     self.y0 = np.append(self.y0, self._extra_y0)
        #     self._extra_equations = extra_equations

        #     if hasattr(self._extra_equations, "jac"):
        #         if hasattr(self._extra_equations, "jac_nnz"):
        #             self.jac_nnz += extra_equations.jac_nnz
        #         else:
        #             self.jac_nnz += len(self._extra_f_nbr)*len(self._extra_f_nbr)
        # else:
        #     self._extra_f_nbr = 0
        
        if synchronize_simulation:
            msg = f"Setting {synchronize_simulation} as 'synchronize_simulation' is not allowed. Must be True/False or greater than 0."
            try:
                if synchronize_simulation is True:
                    self._synchronize_factor = 1.0
                elif synchronize_simulation > 0:
                    self._synchronize_factor = synchronize_simulation
                else:
                    raise InvalidOptionException(msg)
            except Exception:
                raise InvalidOptionException(msg) 
        else:
            self._synchronize_factor = 0.0

        self._state_temp_1 = np.empty(f_nbr, dtype = np.double)
        self._event_temp_1 = np.empty(g_nbr, dtype = np.double)

    def _adapt_input(self, input):
        pass
        # TODO: Should one rename the internal class attributes from real to floatX?
        # if input is not None:
        #     input_names = input[0]
        #     self.input_len_names = len(input_names)
        #     self.input_real_value_refs = []
        #     input_real_mask = []
        #     self.input_other = []
        #     input_other_mask = []

        #     if isinstance(input_names,str):
        #         input_names = [input_names]

        #     for i,name in enumerate(input_names):
        #         if self._model.get_variable_causality(name) != FMI2_INPUT:
        #             raise FMUException("Variable '%s' is not an input. Only variables specified to be inputs are allowed."%name)

        #         if self._model.get_variable_data_type(name) == FMI2_REAL:
        #             self.input_real_value_refs.append(self._model.get_variable_valueref(name))
        #             input_real_mask.append(i)
        #         else:
        #             self.input_other.append(name)
        #             input_other_mask.append(i)

        #     self.input_real_mask  = np.array(input_real_mask)
        #     self.input_other_mask = np.array(input_other_mask)

        #     self._input_activated = 1
        # else:
        #     self._input_activated = 0

        # self.input = input

    cpdef _set_input_values(self, double t):
        pass
        # if self._input_activated:
        #     values = self.input[1].eval(t)[0,:]

        #     if self.input_real_value_refs:
        #         self._model.set_real(self.input_real_value_refs, values[self.input_real_mask])
        #     if self.input_other:
        #         self._model.set(self.input_other, values[self.input_other_mask])

    cdef _update_model(self, double t, np.ndarray[double, ndim=1, mode="c"] y):
        if self.model_me3_instance:
            # Moving data to the model
            self.model_me3._set_time(t)
            # Check if there are any states
            if self._f_nbr != 0:
                self.model_me3._set_continuous_states_fmil(y)
        else:
            # Moving data to the model
            self._model.time = t
            # Check if there are any states
            if self._f_nbr != 0:
                self._model.continuous_states = y

        # Sets the inputs, if any
        self._set_input_values(t)

    cdef int _compare(self, double t, np.ndarray[double, ndim=1, mode="c"] y):
        cdef int res

        if self.model_me3_instance:
            if t != self.model_me3._get_time():
                return 1

            if self._f_nbr == 0:
                return 0

            self.model_me3._get_continuous_states_fmil(self._state_temp_1)
            res = FMIL.memcmp(self._state_temp_1.data, y.data, self._f_nbr*sizeof(double))

            if res == 0:
                return 0

            return 1
        else:
            return t != self._model.time or (not self._f_nbr == 0 and not (self._model.continuous_states == y).all())

    def rhs(self, double t, np.ndarray[double, ndim=1, mode="c"] y, sw=None):
        """ The rhs (right-hand-side) for an ODE problem. """
        cdef int status

        if self._extra_f_nbr > 0:
            y_extra = y[-self._extra_f_nbr:]
            y       = y[:-self._extra_f_nbr]

        self._update_model(t, y)

        # Evaluating the rhs
        if self.model_me3_instance:
            status = self.model_me3._get_derivatives(self._state_temp_1)
            if status != 0:
                raise AssimuloRecoverableError

            if self._f_nbr != 0 and self._extra_f_nbr == 0:
                return self._state_temp_1
            else:
                der = self._state_temp_1.copy()
        else:
            try:
                der = self._model.get_derivatives()
            except FMUException:
                raise AssimuloRecoverableError

        # If there is no state, use the dummy
        if self._f_nbr == 0:
            der = np.array([0.0])

        if self._extra_f_nbr > 0:
            der = np.append(der, self._extra_equations.rhs(y_extra))

        return der

    def jac(self, double t, np.ndarray[double, ndim=1, mode="c"] y, sw=None):
        """ The jacobian function for an ODE problem. """
        pass
        # if self._logging:
        #     preface = "[INFO][FMU status:OK] "
        #     solver_info_tag = 'Jacobian'

        #     msg = preface + '<%s>Starting Jacobian calculation at <value name="t">        %.14E</value>.'%(solver_info_tag, t)
        #     self._model.append_log_message("Model", 4, msg)
        
        # if self._extra_f_nbr > 0:
        #     y_extra = y[-self._extra_f_nbr:]
        #     y       = y[:-self._extra_f_nbr]

        # self._update_model(t, y)

        # # Evaluating the jacobian

        # # If there are no states return a dummy jacobian.
        # if self._f_nbr == 0:
        #     if self._logging:
        #         msg = preface + '</%s>'%(solver_info_tag)
        #         self._model.append_log_message("Model", 6, msg)
            
        #     return np.array([[0.0]])

        # A = self._model._get_A(add_diag=True, output_matrix=self._A)
        # if self._A is None:
        #     self._A = A

        # if self._extra_f_nbr > 0:
        #     if hasattr(self._extra_equations, "jac"):
        #         if self._sparse_representation:

        #             Jac = A.tocoo() # Convert to COOrdinate
        #             A2 = self._extra_equations.jac(y_extra).tocoo()

        #             data = np.append(Jac.data, A2.data)
        #             row  = np.append(Jac.row, A2.row+self._f_nbr)
        #             col  = np.append(Jac.col, A2.col+self._f_nbr)

        #             # Convert to compressed sparse column
        #             Jac = sps.coo_matrix((data, (row, col)))
        #             Jac = Jac.tocsc()
        #         else:
        #             Jac = np.zeros((self._f_nbr+self._extra_f_nbr,self._f_nbr+self._extra_f_nbr))
        #             Jac[:self._f_nbr,:self._f_nbr] = A if isinstance(A, np.ndarray) else A.toarray()
        #             Jac[self._f_nbr:,self._f_nbr:] = self._extra_equations.jac(y_extra)
        #     else:
        #         raise FMUException("No Jacobian provided for the extra equations")
        # else:
        #     Jac = A
            
        # if self._logging:
        #     msg = preface + '</%s>'%(solver_info_tag)
        #     self._model.append_log_message("Model", 4, msg)

        # return Jac

    def state_events(self, double t, np.ndarray[double, ndim=1, mode="c"] y, sw=None):
        """ The event indicator function for a ODE problem. """
        return 0
        # cdef int status

        # if self._extra_f_nbr > 0:
        #     y_extra = y[-self._extra_f_nbr:]
        #     y       = y[:-self._extra_f_nbr]

        # self._update_model(t, y)

        # # Evaluating the event indicators
        # if self.model_me3_instance:
        #     status = self.model_me3._get_event_indicators(self._event_temp_1)

        #     if status != 0:
        #         raise FMUException('Failed to get the event indicators at time: %E.'%t)

        #     return self._event_temp_1
        # else:
        #     return self._model.get_event_indicators()

    def time_events(self, double t, np.ndarray[double, ndim=1, mode="c"] y, sw=None):
        """ Time event function. """
        # eInfo = self._model.get_event_info()

        # if eInfo.nextEventTimeDefined:
        #     return eInfo.nextEventTime
        return None

    def handle_result(self, solver, t, y):
        """ Post processing (stores the time points). """
        pass
        # cdef int status

        # time_start = timer()

        # if self._extra_f_nbr > 0:
        #     y_extra = y[-self._extra_f_nbr:]
        #     y       = y[:-self._extra_f_nbr]

        # # Moving data to the model
        # if self._compare(t, y):
        #     self._update_model(t, y)

        #     # Evaluating the rhs (Have to evaluate the values in the model)
        #     if self.model_me3_instance:
        #         status = self.model_me3._get_derivatives(self._state_temp_1)

        #         if status != 0:
        #             raise FMUException('Failed to get the derivatives at time: %E during handling of the result.'%t)
        #     else:
        #         rhs = self._model.get_derivatives()

        # self.export.integration_point(solver)
        # if self._extra_f_nbr > 0:
        #     self._extra_equations.handle_result(self.export, y_extra)

        # self.timings["handle_result"] += timer() - time_start

    def handle_event(self, solver, event_info):
        """ This method is called when Assimulo finds an event. """
        pass
        # cdef int status

        # if self._extra_f_nbr > 0:
        #     y_extra = solver.y[-self._extra_f_nbr:]
        #     y       = solver.y[:-self._extra_f_nbr]
        # else:
        #     y       = solver.y

        # # Moving data to the model
        # if self._compare(solver.t, y):
        #     self._update_model(solver.t, y)

        #     # Evaluating the rhs (Have to evaluate the values in the model)
        #     if self.model_me3_instance:
        #         status = self.model_me3._get_derivatives(self._state_temp_1)

        #         if status != 0:
        #             raise FMUException('Failed to get the derivatives at time: %E during handling of the event.'%solver.t,)
        #     else:
        #         rhs = self._model.get_derivatives()

        # if self._logging_as_dynamic_diagnostics:
        #     diag_data = np.ndarray(self._number_of_diagnostics_variables, dtype = float)
        #     index = 0
        #     diag_data[index] = solver.t
        #     index +=1
        #     if solver.clock_step:
        #         diag_data[index] = 0
        #         index +=1
        #     solver_name = solver.__class__.__name__
        #     if solver_name == "CVode":
        #         diag_data[index] = solver.get_last_order()
        #         index +=1
        #     if self._f_nbr > 0 and (solver_name=="CVode" or solver_name=="Radau5ODE"):
        #         for e in solver.get_weighted_local_errors():
        #             diag_data[index] = e
        #             index +=1
        #     if (solver_name=="CVode" or
        #         solver_name=="Radau5ODE" or
        #         solver_name=="LSODAR" or
        #         solver_name=="ImplicitEuler" or
        #         solver_name=="ExplicitEuler"
        #         ):
        #         if self._g_nbr > 0:
        #             for ei in self._model.get_event_indicators():
        #                 diag_data[index] = ei
        #                 index +=1
        #         if event_info[1]:
        #             while index < self._number_of_diagnostics_variables - 1:
        #                 diag_data[index] = 0
        #                 index +=1
        #             diag_data[index] = 1.0
        #             index +=1
        #         else:
        #             for ei in event_info[0]:
        #                 diag_data[index] = ei
        #                 index +=1
        #             diag_data[index] = 0.0
        #             index +=1
        #     if index != self._number_of_diagnostics_variables:
        #         raise FMUException("Failed logging diagnostics, number of data points expected to be {} but was {}".format(self._number_of_diagnostics_variables, index))
        #     self.export.diagnostics_point(diag_data)

        # # Enter event mode
        # self._model.enter_event_mode()

        # self._model.event_update()
        # eInfo = self._model.get_event_info()

        # # Check if the event affected the state values and if so sets them
        # if eInfo.valuesOfContinuousStatesChanged:
        #     if self._extra_f_nbr > 0:
        #         solver.y = self._model.continuous_states.append(solver.y[-self._extra_f_nbr:])
        #     else:
        #         solver.y = self._model.continuous_states

        # # Get new nominal values.
        # if eInfo.nominalsOfContinuousStatesChanged:
        #     solver.atol = 0.01*solver.rtol*self._model.nominal_continuous_states

        # # Check if the simulation should be terminated
        # if eInfo.terminateSimulation:
        #     raise TerminateSimulation # Exception from Assimulo

        # # Enter continuous mode again
        # self._model.enter_continuous_time_mode()

    def step_events(self, solver):
        """ Method which is called at each successful step. """
        cdef FMIL3.fmi3_boolean_t enter_event_mode = False, terminate_simulation = False

        if self._extra_f_nbr > 0:
            y_extra = solver.y[-self._extra_f_nbr:]
            y       = solver.y[:-self._extra_f_nbr]
        else:
            y       = solver.y

        # Moving data to the model
        if self._compare(solver.t, y):
            self._update_model(solver.t, y)

            # Evaluating the rhs (Have to evaluate the values in the model)
            if self.model_me3_instance:
                self.model_me3._get_derivatives(self._state_temp_1)
            else:
                rhs = self._model.get_derivatives()

        # if self._logging_as_dynamic_diagnostics:
        #     diag_data = np.ndarray(self._number_of_diagnostics_variables, dtype=float)
        #     index = 0
        #     diag_data[index] = solver.t
        #     index +=1
        #     if solver.clock_step:
        #         diag_data[index] = solver.get_elapsed_step_time()
        #         index +=1
        #     solver_name = solver.__class__.__name__
        #     if solver_name == "CVode":
        #         diag_data[index] = solver.get_last_order()
        #         index +=1
        #     support_state_errors = (solver_name=="CVode" or solver_name=="Radau5ODE")
        #     if support_state_errors and self._f_nbr > 0:
        #         for e in solver.get_weighted_local_errors():
        #             diag_data[index] = e
        #             index +=1
        #     support_event_indicators = (solver_name=="CVode" or solver_name=="Radau5ODE" or solver_name=="LSODAR" or solver_name=="ImplicitEuler")
        #     if self._g_nbr > 0 and support_event_indicators:
        #         for ei in ev_indicator_values:
        #             diag_data[index] = ei
        #             index +=1

        #         for ei in range(len(ev_indicator_values)):
        #             diag_data[index] = 0
        #             index +=1
        #     if support_event_indicators:
        #         diag_data[index] = float(-1)
        #     self.export.diagnostics_point(diag_data)


        if self.model_me3_instance:
            # TODO: Revisit "no_set_FMU_state_prior_to_current_point", first input
            self.model_me3._completed_integrator_step(True, &enter_event_mode, &terminate_simulation)
        else:
            enter_event_mode, terminate_simulation = self._model.completed_integrator_step()
        
        # ret_flag = 0
        # if enter_event_mode:
        #     self._logg_step_event += [solver.t]
        #     # Event have been detect, call event iteration.
        #     self.handle_event(solver,[0])
        #     ret_flag =  1 # Tell to reinitiate the solver.
        
        # if self._synchronize_factor > 0:
        #     under_run = solver.t/self._synchronize_factor - (timer()-self._start_time)
        #     if under_run > 0:
        #         time.sleep(under_run)
        
        # return ret_flag

    def print_step_info(self):
        """ Prints the information about step events. """
        print('\nStep-event information:\n')
        for i in range(len(self._logg_step_event)):
            print('Event at time: %e'%self._logg_step_event[i])
        print('\nNumber of events: ',len(self._logg_step_event))

    def initialize(self, solver):
        # if hasattr(solver,"linear_solver"):
        #     if solver.linear_solver == "SPARSE":
        #         self._sparse_representation = True
        
        if self._synchronize_factor > 0:
            self._start_time = timer()

    def finalize(self, solver):
        pass
        # self.export.simulation_end()

    def _set_input(self, input):
        self.__input = input

    def _get_input(self):
        return self.__input

    input = property(_get_input, _set_input, doc =
    """
    Property for accessing the input. The input must be a 2-tuple with the first
    object as a list of names of the input variables and with the other as a
    subclass of the class Trajectory.
    """)
