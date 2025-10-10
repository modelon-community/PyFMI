#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2014-2025 Modelon AB
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
This file contains code for mapping FMI2 FMUs to the Problem specifications
required by Assimulo.
"""

import numpy as np
cimport numpy as np
np.import_array()

import time
import logging as logging_module
from operator import index
import scipy.sparse as sps
from timeit import default_timer as timer

cimport pyfmi.fmil_import as FMIL
cimport pyfmi.fmi2 as FMI2
from pyfmi.fmi2 import FMI2_REAL, FMI2_INPUT
from pyfmi.exceptions import (
    FMUException,
    InvalidOptionException,
    FMIModel_Exception,
    FMIModelException
)

from assimulo.problem cimport cExplicit_Problem
from assimulo.exception import AssimuloRecoverableError, TerminateSimulation

cdef class FMIODE2(cExplicit_Problem):
    """
    An Assimulo Explicit Model extended to FMI2 interface.
    """

    def __init__(self, model, input=None, result_file_name='',
                 with_jacobian=False, start_time=0.0, logging=False,
                 result_handler=None, extra_equations=None, synchronize_simulation=False,
                 number_of_diagnostics_variables = 0):
        """
        Initialize the problem.
        """
        self._model = model
        self._adapt_input(input)
        self.input_names = []
        self.timings = {"handle_result": 0.0}

        if type(model) == FMI2.FMUModelME2: # if isinstance(model, FMUModelME2):
            self.model_me2 = model
            self.model_me2_instance = 1
        else:
            self.model_me2_instance = 0

        # Set start time to the model
        self._model.time = start_time

        self.t0 = start_time
        self.y0 = self._model.continuous_states
        self.problem_name = self._model.get_name()

        [f_nbr, g_nbr] = self._model.get_ode_sizes()

        self._f_nbr = f_nbr
        self._g_nbr = g_nbr
        self._A = None

        self.state_events_use = False
        if g_nbr > 0:
            self.state_events_use = True
        self.time_events_use = True

        # If there is no state in the model, add a dummy state der(y)=0
        if f_nbr == 0:
            self.y0 = np.array([0.0])

        # Determine the result file name
        if result_file_name == '':
            self.result_file_name = model.get_name()+'_result.txt'
        else:
            self.result_file_name = result_file_name
        self.debug_file_name = model.get_name().replace(".","_")+'_debug.txt'
        self.debug_file_object = None

        # Default values
        self.export = result_handler

        # Internal values
        self._logg_step_event = []
        self._write_header = True
        self._logging = logging
        self._sparse_representation = False
        self._with_jacobian = with_jacobian

        # # If result handler support is available, logging turns into dynamic_diagnostics
        self._logging_as_dynamic_diagnostics = self._logging and result_handler.supports.get("dynamic_diagnostics", False)
        self._number_of_diagnostics_variables = number_of_diagnostics_variables

        self.jac_use = False
        if f_nbr > 0 and with_jacobian:
            self.jac_use = True # Activates the jacobian

            # Need to calculate the nnz.
            [derv_state_dep, derv_input_dep] = model.get_derivatives_dependencies()
            self.jac_nnz = np.sum([len(derv_state_dep[key]) for key in derv_state_dep.keys()])+f_nbr

        if extra_equations:
            self._extra_f_nbr = extra_equations.get_size()
            self._extra_y0    = extra_equations.y0
            self.y0 = np.append(self.y0, self._extra_y0)
            self._extra_equations = extra_equations

            if hasattr(self._extra_equations, "jac"):
                if hasattr(self._extra_equations, "jac_nnz"):
                    self.jac_nnz += extra_equations.jac_nnz
                else:
                    self.jac_nnz += len(self._extra_f_nbr)*len(self._extra_f_nbr)
        else:
            self._extra_f_nbr = 0
        
        if synchronize_simulation:
            try:
                if synchronize_simulation is True:
                    self._synchronize_factor = 1.0
                elif synchronize_simulation > 0:
                    self._synchronize_factor = synchronize_simulation
                else:
                    raise InvalidOptionException(f"Setting {synchronize_simulation} as 'synchronize_simulation' is not allowed. Must be True/False or greater than 0.")
            except Exception:
                raise InvalidOptionException(f"Setting {synchronize_simulation} as 'synchronize_simulation' is not allowed. Must be True/False or greater than 0.") 
        else:
            self._synchronize_factor = 0.0

        self._state_temp_1 = np.empty(f_nbr, dtype = np.double)
        self._event_temp_1 = np.empty(g_nbr, dtype = np.double)

    def _adapt_input(self, input):
        if input is not None:
            input_names = input[0]
            self.input_len_names = len(input_names)
            self.input_real_value_refs = []
            input_real_mask = []
            self.input_other = []
            input_other_mask = []

            if isinstance(input_names,str):
                input_names = [input_names]

            for i,name in enumerate(input_names):
                if self._model.get_variable_causality(name) != FMI2_INPUT:
                    raise FMUException("Variable '%s' is not an input. Only variables specified to be inputs are allowed."%name)

                if self._model.get_variable_data_type(name) == FMI2_REAL:
                    self.input_real_value_refs.append(self._model.get_variable_valueref(name))
                    input_real_mask.append(i)
                else:
                    self.input_other.append(name)
                    input_other_mask.append(i)

            self.input_real_mask  = np.array(input_real_mask)
            self.input_other_mask = np.array(input_other_mask)

            self._input_activated = 1
        else:
            self._input_activated = 0

        self.input = input

    cpdef _set_input_values(self, double t):
        if self._input_activated:
            values = self.input[1].eval(t)[0,:]

            if self.input_real_value_refs:
                self._model.set_real(self.input_real_value_refs, values[self.input_real_mask])
            if self.input_other:
                self._model.set(self.input_other, values[self.input_other_mask])

    cdef _update_model(self, double t, np.ndarray[double, ndim=1, mode="c"] y):
        if self.model_me2_instance:
            # Moving data to the model
            self.model_me2._set_time(t)
            # Check if there are any states
            if self._f_nbr != 0:
                self.model_me2._set_continuous_states_fmil(y)
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

        if self.model_me2_instance:
            if t != self.model_me2._get_time():
                return 1

            if self._f_nbr == 0:
                return 0

            self.model_me2._get_continuous_states_fmil(self._state_temp_1)
            res = FMIL.memcmp(self._state_temp_1.data, y.data, self._f_nbr*sizeof(double))

            if res == 0:
                return 0

            return 1
        else:
            return t != self._model.time or (not self._f_nbr == 0 and not (self._model.continuous_states == y).all())

    def rhs(self, double t, np.ndarray[double, ndim=1, mode="c"] y, sw=None):
        """
        The rhs (right-hand-side) for an ODE problem.
        """
        cdef int status

        if self._extra_f_nbr > 0:
            y_extra = y[-self._extra_f_nbr:]
            y       = y[:-self._extra_f_nbr]

        self._update_model(t, y)

        # Evaluating the rhs
        if self.model_me2_instance:
            status = self.model_me2._get_derivatives(self._state_temp_1)
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
        """
        The jacobian function for an ODE problem.
        """
        log_requires_closing_tag = False
        
        if self._logging:
            preface = "[INFO][FMU status:OK] "
            solver_info_tag = 'Jacobian'

            msg = preface + '<%s>Starting Jacobian calculation at <value name="t">        %.14E</value>.'%(solver_info_tag, t)
            self._model.append_log_message("Model", 4, msg)
            log_requires_closing_tag = True
            log_closing_tag = preface + '</%s>'%(solver_info_tag)
        
        if self._extra_f_nbr > 0:
            y_extra = y[-self._extra_f_nbr:]
            y       = y[:-self._extra_f_nbr]

        try:
            self._update_model(t, y)

            # Evaluating the jacobian

            # If there are no states return a dummy jacobian.
            if self._f_nbr == 0:
                return np.array([[0.0]])

            A = self._model._get_A(add_diag=True, output_matrix=self._A)
            if self._A is None:
                self._A = A

            if self._extra_f_nbr > 0:
                if hasattr(self._extra_equations, "jac"):
                    if self._sparse_representation:

                        Jac = A.tocoo() # Convert to COOrdinate
                        A2 = self._extra_equations.jac(y_extra).tocoo()

                        data = np.append(Jac.data, A2.data)
                        row  = np.append(Jac.row, A2.row+self._f_nbr)
                        col  = np.append(Jac.col, A2.col+self._f_nbr)

                        # Convert to compressed sparse column
                        Jac = sps.coo_matrix((data, (row, col)))
                        Jac = Jac.tocsc()
                    else:
                        Jac = np.zeros((self._f_nbr+self._extra_f_nbr,self._f_nbr+self._extra_f_nbr))
                        Jac[:self._f_nbr,:self._f_nbr] = A if isinstance(A, np.ndarray) else A.toarray()
                        Jac[self._f_nbr:,self._f_nbr:] = self._extra_equations.jac(y_extra)
                else:
                    raise FMUException("No Jacobian provided for the extra equations")
            else:
                Jac = A
        finally: # Even includes early return
            if log_requires_closing_tag:
                self._model.append_log_message("Model", 4, log_closing_tag)

        return Jac

    def state_events(self, double t, np.ndarray[double, ndim=1, mode="c"] y, sw=None):
        """
        The event indicator function for a ODE problem.
        """
        cdef int status

        if self._extra_f_nbr > 0:
            y_extra = y[-self._extra_f_nbr:]
            y       = y[:-self._extra_f_nbr]

        self._update_model(t, y)

        # Evaluating the event indicators
        if self.model_me2_instance:
            status = self.model_me2._get_event_indicators(self._event_temp_1)

            if status != 0:
                raise FMUException('Failed to get the event indicators at time: %E.'%t)

            return self._event_temp_1
        else:
            return self._model.get_event_indicators()

    def time_events(self, double t, np.ndarray[double, ndim=1, mode="c"] y, sw=None):
        """
        Time event function.
        """
        eInfo = self._model.get_event_info()

        if eInfo.nextEventTimeDefined:
            return eInfo.nextEventTime
        return None


    def handle_result(self, solver, t, y):
        """
        Post processing (stores the time points).
        """
        cdef int status

        time_start = timer()

        if self._extra_f_nbr > 0:
            y_extra = y[-self._extra_f_nbr:]
            y       = y[:-self._extra_f_nbr]

        # Moving data to the model
        if self._compare(t, y):
            self._update_model(t, y)

            # Evaluating the rhs (Have to evaluate the values in the model)
            if self.model_me2_instance:
                status = self.model_me2._get_derivatives(self._state_temp_1)

                if status != 0:
                    raise FMUException('Failed to get the derivatives at time: %E during handling of the result.'%t)
            else:
                rhs = self._model.get_derivatives()

        self.export.integration_point(solver)
        if self._extra_f_nbr > 0:
            self._extra_equations.handle_result(self.export, y_extra)

        self.timings["handle_result"] += timer() - time_start

    def handle_event(self, solver, event_info):
        """
        This method is called when Assimulo finds an event.
        """
        cdef int status

        if self._extra_f_nbr > 0:
            y_extra = solver.y[-self._extra_f_nbr:]
            y       = solver.y[:-self._extra_f_nbr]
        else:
            y       = solver.y

        # Moving data to the model
        if self._compare(solver.t, y):
            self._update_model(solver.t, y)

            # Evaluating the rhs (Have to evaluate the values in the model)
            if self.model_me2_instance:
                status = self.model_me2._get_derivatives(self._state_temp_1)

                if status != 0:
                    raise FMUException('Failed to get the derivatives at time: %E during handling of the event.'%solver.t,)
            else:
                rhs = self._model.get_derivatives()

        if self._logging:
            str_ind = ""
            for i in self._model.get_event_indicators():
                str_ind += " %.14E"%i
            str_states = ""
            if self._f_nbr != 0:
                for i in y:
                    str_states += " %.14E"%i
            str_der = ""
            for i in self._model.get_derivatives():
                str_der += " %.14E"%i

            if not self._logging_as_dynamic_diagnostics:
                fwrite = self._get_debug_file_object()
                fwrite.write("\nDetected event at t = %.14E \n"%solver.t)
                fwrite.write(" State event info: "+" ".join(str(i) for i in event_info[0])+ "\n")
                fwrite.write(" Time  event info:  "+str(event_info[1])+ "\n")

                preface = "[INFO][FMU status:OK] "
                event_info_tag = 'EventInfo'

                msg = preface + '<%s>Detected event at <value name="t">        %.14E</value>.'%(event_info_tag, solver.t)
                self._model.append_log_message("Model", 6, msg)

                msg = preface + '  <vector name="state_event_info">' + ", ".join(str(i) for i in event_info[0]) + '</vector>'
                self._model.append_log_message("Model", 6, msg)

                msg = preface + '  <boolean name="time_event_info">' + str(event_info[1]) + '</boolean>'
                self._model.append_log_message("Model", 6, msg)

                msg = preface + '</%s>'%(event_info_tag)
                self._model.append_log_message("Model", 6, msg)
            else:
                diag_data = np.ndarray(self._number_of_diagnostics_variables, dtype=float)
                index = 0
                diag_data[index] = solver.t
                index +=1
                if solver.clock_step:
                    diag_data[index] = 0
                    index +=1
                solver_name = solver.__class__.__name__
                if solver_name == "CVode":
                    diag_data[index] = solver.get_last_order()
                    index +=1
                if self._f_nbr > 0 and (solver_name=="CVode" or solver_name=="Radau5ODE"):
                    for e in solver.get_weighted_local_errors():
                        diag_data[index] = e
                        index +=1
                if (solver_name=="CVode" or
                    solver_name=="Radau5ODE" or
                    solver_name=="LSODAR" or
                    solver_name=="ImplicitEuler" or
                    solver_name=="ExplicitEuler"
                    ):
                    if self._g_nbr > 0:
                        for ei in self._model.get_event_indicators():
                            diag_data[index] = ei
                            index +=1
                    if event_info[1]:
                        while index < self._number_of_diagnostics_variables - 1:
                            diag_data[index] = 0
                            index +=1
                        diag_data[index] = 1.0
                        index +=1
                    else:
                        for ei in event_info[0]:
                            diag_data[index] = ei
                            index +=1
                        diag_data[index] = 0.0
                        index +=1
                if index != self._number_of_diagnostics_variables:
                    raise FMUException("Failed logging diagnostics, number of data points expected to be {} but was {}".format(self._number_of_diagnostics_variables, index))
                self.export.diagnostics_point(diag_data)


        # Enter event mode
        self._model.enter_event_mode()

        self._model.event_update()
        eInfo = self._model.get_event_info()

        # Check if the event affected the state values and if so sets them
        if eInfo.valuesOfContinuousStatesChanged:
            if self._extra_f_nbr > 0:
                solver.y = self._model.continuous_states.append(solver.y[-self._extra_f_nbr:])
            else:
                solver.y = self._model.continuous_states

        # Get new nominal values.
        if eInfo.nominalsOfContinuousStatesChanged:
            solver.atol = 0.01*solver.rtol*self._model.nominal_continuous_states

        # Check if the simulation should be terminated
        if eInfo.terminateSimulation:
            raise TerminateSimulation # Exception from Assimulo

        if self._logging:
            str_ind2 = ""
            for i in self._model.get_event_indicators():
                str_ind2 += " %.14E"%i
            str_states2 = ""
            if self._f_nbr != 0:
                for i in solver.y:
                    str_states2 += " %.14E"%i
            str_der2 = ""
            for i in self._model.get_derivatives():
                str_der2 += " %.14E"%i

            if not self._logging_as_dynamic_diagnostics:
                fwrite = self._get_debug_file_object()
                fwrite.write(" Indicators (pre) : "+str_ind + "\n")
                fwrite.write(" Indicators (post): "+str_ind2+"\n")
                fwrite.write(" States (pre) : "+str_states + "\n")
                fwrite.write(" States (post): "+str_states2 + "\n")
                fwrite.write(" Derivatives (pre) : "+str_der + "\n")
                fwrite.write(" Derivatives (post): "+str_der2 + "\n\n")

                header = "Time (simulated) | Time (real) | "
                if solver.__class__.__name__=="CVode" or solver.__class__.__name__=="Radau5ODE": # Only available for CVode
                    header += "Order | Error (Weighted)"
                if self._g_nbr > 0:
                    header += "Indicators"
                fwrite.write(header+"\n")

        # Enter continuous mode again
        self._model.enter_continuous_time_mode()

    def step_events(self, solver):
        """
        Method which is called at each successful step.
        """
        cdef int enter_event_mode = 0, terminate_simulation = 0

        if self._extra_f_nbr > 0:
            y_extra = solver.y[-self._extra_f_nbr:]
            y       = solver.y[:-self._extra_f_nbr]
        else:
            y       = solver.y

        # Moving data to the model
        if self._compare(solver.t, y):
            self._update_model(solver.t, y)

            # Evaluating the rhs (Have to evaluate the values in the model)
            if self.model_me2_instance:
                self.model_me2._get_derivatives(self._state_temp_1)
            else:
                rhs = self._model.get_derivatives()

        if self._logging:
            solver_name = solver.__class__.__name__
            data_line = "%.14E"%solver.t+" | %.14E"%(solver.get_elapsed_step_time())

            if solver_name=="CVode" or solver_name=="Radau5ODE":
                err = solver.get_weighted_local_errors()
                str_err = " |"
                for i in err:
                    str_err += " %.14E"%i
                if solver_name=="CVode":
                    data_line += " | %d"%solver.get_last_order()+str_err
                else:
                    data_line += " | 5"+str_err

            if self._g_nbr > 0:
                str_ev = " |"
                ev_indicator_values = self._model.get_event_indicators()
                for i in ev_indicator_values:
                    str_ev += " %.14E"%i
                data_line += str_ev


            if not self._logging_as_dynamic_diagnostics:
                fwrite = self._get_debug_file_object()
                fwrite.write(data_line+"\n")
                preface = "[INFO][FMU status:OK] "
                solver_info_tag = 'Solver'

                msg = preface + '<%s>Successful solver step at <value name="t">        %.14E</value>.'%(solver_info_tag, solver.t)
                self._model.append_log_message("Model", 6, msg)

                if solver.clock_step:
                    msg = preface + '  <value name="elapsed_real_time">        %.14E</value>'%(solver.get_elapsed_step_time())
                    self._model.append_log_message("Model", 6, msg)

                support_solver_order = solver_name=="CVode"
                if support_solver_order:
                    msg = preface + '  <value name="solver_order">%d</value>'%(solver.get_last_order())
                    self._model.append_log_message("Model", 6, msg)

                support_state_errors = (solver_name=="CVode" or solver_name=="Radau5ODE")
                if support_state_errors:
                    state_errors = ''
                    for i in solver.get_weighted_local_errors():
                        state_errors += "        %.14E,"%i
                    msg = preface + '  <vector name="state_error">' + state_errors[:-1] + '</vector>'
                    self._model.append_log_message("Model", 6, msg)

                if self._g_nbr > 0:
                    msg = preface + '  <vector name="event_indicators">'
                    for i in ev_indicator_values:
                        msg += "        %.14E,"%i
                    msg = msg[:-1] + '</vector>'# remove last comma
                    self._model.append_log_message("Model", 6, msg)


                msg = preface + '</%s>'%(solver_info_tag)
                self._model.append_log_message("Model", 6, msg)
            else:
                diag_data = np.ndarray(self._number_of_diagnostics_variables, dtype=float)
                index = 0
                diag_data[index] = solver.t
                index +=1
                if solver.clock_step:
                    diag_data[index] = solver.get_elapsed_step_time()
                    index +=1
                solver_name = solver.__class__.__name__
                if solver_name == "CVode":
                    diag_data[index] = solver.get_last_order()
                    index +=1
                support_state_errors = (solver_name=="CVode" or solver_name=="Radau5ODE")
                if support_state_errors and self._f_nbr > 0:
                    for e in solver.get_weighted_local_errors():
                        diag_data[index] = e
                        index +=1
                support_event_indicators = (solver_name=="CVode" or solver_name=="Radau5ODE" or solver_name=="ExplicitEuler" or solver_name=="LSODAR" or solver_name=="ImplicitEuler")
                if self._g_nbr > 0 and support_event_indicators:
                    for ei in ev_indicator_values:
                        diag_data[index] = ei
                        index +=1

                    for ei in range(len(ev_indicator_values)):
                        diag_data[index] = 0
                        index +=1
                if support_event_indicators:
                    diag_data[index] = float(-1)
                self.export.diagnostics_point(diag_data)


        if self.model_me2_instance:
            self.model_me2._completed_integrator_step(&enter_event_mode, &terminate_simulation)
        else:
            enter_event_mode, terminate_simulation = self._model.completed_integrator_step()
        
        ret_flag = 0
        if enter_event_mode:
            self._logg_step_event += [solver.t]
            # Event have been detect, call event iteration.
            self.handle_event(solver,[0])
            ret_flag =  1 # Tell to reinitiate the solver.
        
        if self._synchronize_factor > 0:
            under_run = solver.t/self._synchronize_factor - (timer()-self._start_time)
            if under_run > 0:
                time.sleep(under_run)
        
        return ret_flag

    def _get_debug_file_object(self):
        if not self.debug_file_object:
            self.debug_file_object = open(self.debug_file_name, 'a')

        return self.debug_file_object

    def print_step_info(self):
        """
        Prints the information about step events.
        """
        print('\nStep-event information:\n')
        for i in range(len(self._logg_step_event)):
            print('Event at time: %e'%self._logg_step_event[i])
        print('\nNumber of events: ',len(self._logg_step_event))

    def initialize(self, solver):
        if hasattr(solver,"linear_solver"):
            if solver.linear_solver == "SPARSE":
                self._sparse_representation = True
        
        if self._synchronize_factor > 0:
            self._start_time = timer()

        if self._logging and not self._logging_as_dynamic_diagnostics:
            solver_name = solver.__class__.__name__
            self.debug_file_object = open(self.debug_file_name, 'w')
            f = self.debug_file_object

            names = ""
            for i in range(self._f_nbr):
                names += list(self._model.get_states_list().keys())[i] + ", "
            names = names[:-2] # remove trailing ', '

            preface = "[INFO][FMU status:OK] "
            init_tag = 'SolverSettings'

            msg = preface + '<%s>Solver initialized with the following attributes:'%(init_tag)
            self._model.append_log_message("Model", 6, msg)

            msg = preface + '  <vector name="state_names">' + names + '</vector>'
            self._model.append_log_message("Model", 6, msg)

            msg = preface + '  <value name="solver_name">%s</value>'%solver_name
            self._model.append_log_message("Model", 6, msg)

            support_state_errors = (solver_name=="CVode" or solver_name=="Radau5ODE")
            msg = preface + '  <boolean name="support_state_errors">%s</boolean>'%support_state_errors
            self._model.append_log_message("Model", 6, msg)

            support_event_indicators = (solver_name=="CVode" or solver_name=="Radau5ODE")
            msg = preface + '  <boolean name="support_event_indicators">%s</boolean>'%support_event_indicators
            self._model.append_log_message("Model", 6, msg)

            support_solver_order = solver_name=="CVode"
            msg = preface + '  <boolean name="support_solver_order">%s</boolean>'%support_solver_order
            self._model.append_log_message("Model", 6, msg)

            msg = preface + '  <boolean name="support_elapsed_time">%s</boolean>'%solver.clock_step
            self._model.append_log_message("Model", 6, msg)

            support_tol = solver_name != "ExplicitEuler"
            msg = preface + '  <boolean name="support_tolerances">%s</boolean>'%support_tol
            self._model.append_log_message("Model", 6, msg)

            if support_tol:
                msg = preface + '  <value name="relative_tolerance">%.14E</value>'%solver.rtol
                self._model.append_log_message("Model", 6, msg)

                atol_values = ''
                for value in solver.atol:
                    atol_values += '        %.14E,'%value
                msg = preface + '  <vector name="absolute_tolerance">' + atol_values[:-1] + '</vector>'
                self._model.append_log_message("Model", 6, msg)

            msg = preface + '</%s>'%(init_tag)
            self._model.append_log_message("Model", 6, msg)

            str_y = ""
            if self._f_nbr != 0:
                for i in solver.y:
                    str_y += " %.14E"%i

            f.write("Solver: %s \n"%solver_name)
            f.write("State variables: "+names+ "\n")

            f.write("Initial values: t = %.14E \n"%solver.t)
            f.write("Initial values: y ="+str_y+"\n\n")

            header = "Time (simulated) | Time (real) | "
            if solver_name=="CVode" or solver_name=="Radau5ODE": # Only available for CVode and Radau5ODE
                header += "Order | Error (Weighted)"
            f.write(header+"\n")

    def finalize(self, solver):
        self.export.simulation_end()

        if self.debug_file_object:
            self.debug_file_object.close()
            self.debug_file_object = None

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

class FMIODESENS2(FMIODE2):
    """
    FMIODE2 extended with sensitivity simulation capabilities
    """
    def __init__(self, model, input=None, result_file_name='',
                 with_jacobian=False, start_time=0.0, logging=False,
                 result_handler=None, extra_equations=None, parameters=None,
                 number_of_diagnostics_variables = 0):

        # Call FMIODE init method
        FMIODE2.__init__(self, model, input, result_file_name, with_jacobian,
                start_time, logging, result_handler, extra_equations,
                number_of_diagnostics_variables = number_of_diagnostics_variables)

        # Store the parameters
        if parameters is not None:
            if not isinstance(parameters,list):
                raise FMIModelException("Parameters must be a list of names.")
            self.p0 = np.array(model.get(parameters)).flatten()
            self.pbar = np.array([np.abs(x) if np.abs(x) > 0 else 1.0 for x in self.p0])
            self.param_valref = [model.get_variable_valueref(x) for x in parameters]

            for param in parameters:
                if model.get_variable_causality(param) != FMI2_INPUT and \
                   (model.get_generation_tool() != "JModelica.org" and model.get_generation_tool() != "Optimica Compiler Toolkit"):
                    raise FMIModelException("The sensitivity parameters must be specified as inputs!")

        self.parameters = parameters
        self.derivatives = [v.value_reference for i,v in model.get_derivatives_list().items()]

        if self._model.get_capability_flags()['providesDirectionalDerivatives']:
            use_rhs_sens = True
            for param in parameters:
                if model.get_variable_causality(param) != FMI2_INPUT and \
                  (model.get_generation_tool() == "JModelica.org" or model.get_generation_tool() == "Optimica Compiler Toolkit"):
                    use_rhs_sens = False
                    logging_module.warning("The sensitivity parameters must be specified as inputs in order to set up the sensitivity " \
                            "equations using directional derivatives. Disabling and using finite differences instead.")

            if use_rhs_sens:
                self.rhs_sens = self.s # Activates the jacobian

        super(FMIODESENS2, self).rhs(0.0,self.y0,None)

    def rhs(self, t, y, p=None, sw=None):
        # Sets the parameters, if any
        if self.parameters is not None:
            self._model.set(self.parameters, p)

        return FMIODE2.rhs(self,t,y,sw)

    def jac(self, t, y, p=None, sw=None):
        # Sets the parameters, if any
        if self.parameters is not None:
            self._model.set(self.parameters, p)

        return FMIODE2.jac(self,t,y,sw)

    def s(self, t, y, s, p=None, sw=None):
        # ds = df/dy s + df/dp
        J = self.jac(t,y,p,sw)
        sens_rhs = J.dot(s)

        for i,param in enumerate(self.param_valref):
            dfdpi = self._model.get_directional_derivative([param], self.derivatives, [1])
            sens_rhs[:,i] += dfdpi

        return sens_rhs
