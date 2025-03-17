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
This file contains code for mapping FMI1 FMUs to the Problem specifications
required by Assimulo.
"""
import logging as logging_module
from timeit import default_timer as timer

import numpy as np
cimport numpy as np

from pyfmi.exceptions import FMUException, FMIModel_Exception, FMIModelException

try:
    import assimulo
    assimulo_present = True
except Exception:
    logging_module.warning(
        'Could not load Assimulo module. Check pyfmi.check_packages()')
    assimulo_present = False

if assimulo_present:
    from assimulo.problem import Explicit_Problem
    from assimulo.exception import AssimuloRecoverableError, TerminateSimulation
else:
    class Explicit_Problem:
        pass

class FMIODE(Explicit_Problem):
    """
    An Assimulo Explicit Model extended to FMI1 interface.
    """
    def __init__(self, model, input=None, result_file_name='',
                 with_jacobian=False, start_time=0.0, logging=False,
                 result_handler=None):
        """
        Initialize the problem.
        """
        self._model = model
        self._adapt_input(input)
        self.timings = {"handle_result": 0.0}

        # Set start time to the model
        self._model.time = start_time

        self.t0 = start_time
        self.y0 = self._model.continuous_states
        self.name = self._model.get_name()

        [f_nbr, g_nbr] = self._model.get_ode_sizes()

        self._f_nbr = f_nbr
        self._g_nbr = g_nbr

        if g_nbr > 0:
            self.state_events = self.g
        self.time_events = self.t

        # If there is no state in the model, add a dummy
        # state der(y)=0
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
        self._sol_time = []
        self._sol_real = []
        self._sol_int  = []
        self._sol_bool = []
        self._logg_step_event = []
        self._write_header = True
        self._logging = logging

        # # If result handler support is available, logging turns into dynamic_diagnostics
        self._logging_as_dynamic_diagnostics = self._logging and result_handler.supports.get("dynamic_diagnostics", False)

        # Stores the first time point
        # [r,i,b] = self._model.save_time_point()

        # self._sol_time += [self._model.t]
        # self._sol_real += [r]
        # self._sol_int  += [i]
        # self._sol_bool += b

        self._jm_fmu = self._model.get_generation_tool() == "JModelica.org"

        if with_jacobian:
            raise FMUException("Jacobians are not supported using FMI 1.0, please use FMI 2.0")

    def _adapt_input(self, input):
        if input is not None:
            input_value_refs = []
            input_alias_type = []
            if isinstance(input[0],str):
                input_value_refs.append(self._model.get_variable_valueref(input[0]))
                input_alias_type.append(-1.0 if self._model.get_variable_alias(input[0])[input[0]] == -1 else 1.0)
            else:
                for name in input[0]:
                    input_value_refs.append(self._model.get_variable_valueref(name))
                    input_alias_type.append(-1.0 if self._model.get_variable_alias(name)[name] == -1 else 1.0)
            self.input_value_refs = input_value_refs
            self.input_alias_type = input_alias_type if np.any(input_alias_type==-1) else 1.0
        self.input = input

    def rhs(self, t, y, sw=None):
        """
        The rhs (right-hand-side) for an ODE problem.
        """
        # Moving data to the model
        self._model.time = t
        # Check if there are any states
        if self._f_nbr != 0:
            self._model.continuous_states = y

        # Sets the inputs, if any
        if self.input is not None:
            self._model.set_real(self.input_value_refs, self.input[1].eval(t)[0,:]*self.input_alias_type)

        # Evaluating the rhs
        try:
            rhs = self._model.get_derivatives()
        except FMUException:
            raise AssimuloRecoverableError

        # If there is no state, use the dummy
        if self._f_nbr == 0:
            rhs = np.array([0.0])

        return rhs

    def g(self, t, y, sw=None):
        """
        The event indicator function for a ODE problem.
        """
        # Moving data to the model
        self._model.time = t
        # Check if there are any states
        if self._f_nbr != 0:
            self._model.continuous_states = y

        # Sets the inputs, if any
        if self.input is not None:
            self._model.set_real(self.input_value_refs, self.input[1].eval(t)[0,:]*self.input_alias_type)

        # Evaluating the event indicators
        eventInd = self._model.get_event_indicators()

        return eventInd

    def t(self, t, y, sw=None):
        """
        Time event function.
        """
        eInfo = self._model.get_event_info()

        if eInfo.upcomingTimeEvent:
            return eInfo.nextEventTime
        else:
            return None


    def handle_result(self, solver, t, y):
        """
        Post processing (stores the time points).
        """
        time_start = timer()

        # Moving data to the model
        if t != self._model.time or (not self._f_nbr == 0 and not (self._model.continuous_states == y).all()):
            # Moving data to the model
            self._model.time = t
            # Check if there are any states
            if self._f_nbr != 0:
                self._model.continuous_states = y

            # Sets the inputs, if any
            if self.input is not None:
                self._model.set_real(self.input_value_refs, self.input[1].eval(t)[0,:]*self.input_alias_type)

            # Evaluating the rhs (Have to evaluate the values in the model)
            rhs = self._model.get_derivatives()

        if self.export is not None:
            self.export.integration_point()

        self.timings["handle_result"] += timer() - time_start

    def handle_event(self, solver, event_info):
        """
        This method is called when Assimulo finds an event.
        """
        # Moving data to the model
        if solver.t != self._model.time or (not self._f_nbr == 0 and not (self._model.continuous_states == solver.y).all()):
            self._model.time = solver.t
            # Check if there are any states
            if self._f_nbr != 0:
                self._model.continuous_states = solver.y

            # Sets the inputs, if any
            if self.input is not None:
                self._model.set_real(self.input_value_refs, self.input[1].eval(np.array([solver.t]))[0,:]*self.input_alias_type)

            # Evaluating the rhs (Have to evaluate the values in the model)
            rhs = self._model.get_derivatives()

        if self._logging:
            str_ind = ""
            for i in self._model.get_event_indicators():
                str_ind += " %.14E"%i
            str_states = ""
            if self._f_nbr != 0:
                for i in solver.y:
                    str_states += " %.14E"%i
            str_der = ""
            for i in self._model.get_derivatives():
                str_der += " %.14E"%i

            if not self._logging_as_dynamic_diagnostics:
                fwrite = self._get_debug_file_object()
                fwrite.write("\nDetected event at t = %.14E \n"%solver.t)
                fwrite.write(" State event info: "+" ".join(str(i) for i in event_info[0])+ "\n")
                fwrite.write(" Time  event info:  "+str(event_info[1])+ "\n")

        eInfo = self._model.get_event_info()
        eInfo.iterationConverged = False

        while not eInfo.iterationConverged:
            self._model.event_update(intermediateResult=False)

            eInfo = self._model.get_event_info()
            # Retrieve solutions (if needed)
            # if not eInfo.iterationConverged:
            #    pass

        # Check if the event affected the state values and if so sets them
        if eInfo.stateValuesChanged:
            if self._f_nbr == 0:
                solver.y[0] = 0.0
            else:
                solver.y = self._model.continuous_states

        # Get new nominal values.
        if eInfo.stateValueReferencesChanged:
            if self._f_nbr == 0:
                solver.atol = 0.01*solver.rtol*1
            else:
                solver.atol = 0.01*solver.rtol*self._model.nominal_continuous_states

        # Check if the simulation should be terminated
        if eInfo.terminateSimulation:
            raise TerminateSimulation # Exception from Assimulo

        if self._logging and not self._logging_as_dynamic_diagnostics:
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

            fwrite.write(" Indicators (pre) : "+str_ind + "\n")
            fwrite.write(" Indicators (post): "+str_ind2+"\n")
            fwrite.write(" States (pre) : "+str_states + "\n")
            fwrite.write(" States (post): "+str_states2 + "\n")
            fwrite.write(" Derivatives (pre) : "+str_der + "\n")
            fwrite.write(" Derivatives (post): "+str_der2 + "\n\n")

            header = "Time (simulated) | Time (real) | "
            if solver.__class__.__name__=="CVode": # Only available for CVode
                header += "Order | Error (Weighted)"
            if self._g_nbr > 0:
                header += "Indicators"
            fwrite.write(header+"\n")

    def step_events(self, solver):
        """
        Method which is called at each successful step.
        """
        # Moving data to the model
        if solver.t != self._model.time or (not self._f_nbr == 0 and not (self._model.continuous_states == solver.y).all()):
            self._model.time = solver.t
            # Check if there are any states
            if self._f_nbr != 0:
                self._model.continuous_states = solver.y

            # Sets the inputs, if any
            if self.input is not None:
                self._model.set_real(self.input_value_refs, self.input[1].eval(np.array([solver.t]))[0,:]*self.input_alias_type)
                # self._model.set(self.input[0],self.input[1].eval(np.array([solver.t]))[0,:])

            # Evaluating the rhs (Have to evaluate the values in the model)
            rhs = self._model.get_derivatives()

        if self._logging:

            if self._jm_fmu:
                solver_name = solver.__class__.__name__

                preface = "[INFO][FMU status:OK] "

                msg = preface + '<%s>Successfully computed a step at <value name="time">        %.14E</value>. Elapsed time for the step  <value name="t">        %.14E</value>"'%(solver_name,solver.t,solver.get_elapsed_step_time())
                self._model.append_log_message("Model", 6, msg)

                if solver_name == "CVode":
                    msg = preface + '  <vector name="weighted_error">' + " ".join(["     %.14E"%e for e in solver.get_local_errors()*solver.get_error_weights()])+"</vector>"
                    self._model.append_log_message("Model", 6, msg)

                    msg = preface + '  <vector name="order">%d</vector>'%solver.get_last_order()
                    self._model.append_log_message("Model", 6, msg)

                # End tag
                msg = preface + "</%s>"%solver_name
                self._model.append_log_message("Model", 6, msg)

            if not self._logging_as_dynamic_diagnostics:
                data_line = "%.14E"%solver.t+" | %.14E"%(solver.get_elapsed_step_time())

                if solver.__class__.__name__=="CVode": # Only available for CVode
                    ele = solver.get_local_errors()
                    eweight = solver.get_error_weights()
                    err = ele*eweight
                    str_err = " |"
                    for i in err:
                        str_err += " %.14E"%i
                    data_line += " | %d"%solver.get_last_order()+str_err

                if self._g_nbr > 0:
                    str_ev = " |"
                    for i in self._model.get_event_indicators():
                        str_ev += " %.14E"%i
                    data_line += str_ev

                fwrite = self._get_debug_file_object()
                fwrite.write(data_line+"\n")

        if self._model.completed_integrator_step():
            self._logg_step_event += [solver.t]
            # Event have been detect, call event iteration.
            # print "Step event detected at: ", solver.t
            # self.handle_event(solver,[0])
            return 1 # Tell to reinitiate the solver.
        else:
            return 0

    def print_step_info(self):
        """
        Prints the information about step events.
        """
        print('\nStep-event information:\n')
        for i in range(len(self._logg_step_event)):
            print('Event at time: %e'%self._logg_step_event[i])
        print('\nNumber of events: ',len(self._logg_step_event))

    def _get_debug_file_object(self):
        if not self.debug_file_object:
            self.debug_file_object = open(self.debug_file_name, 'a')

        return self.debug_file_object

    def initialize(self, solver):
        if self._logging and not self._logging_as_dynamic_diagnostics:
            self.debug_file_object = open(self.debug_file_name, 'w')
            f = self.debug_file_object

            model_valref = self._model.get_state_value_references()
            names = ""
            for i in model_valref:
                names += self._model.get_variable_by_valueref(i) + ", "

            f.write("Solver: %s \n"%solver.__class__.__name__)
            f.write("State variables: "+names+ "\n")

            str_y = ""
            if self._f_nbr != 0:
                for i in solver.y:
                    str_y += " %.14E"%i

            f.write("Initial values: t = %.14E \n"%solver.t)
            f.write("Initial values: y ="+str_y+"\n\n")

            header = "Time (simulated) | Time (real) | "
            if solver.__class__.__name__=="CVode": # Only available for CVode
                header += "Order | Error (Weighted)"
            f.write(header+"\n")

    def finalize(self, solver):
        if self.export is not None:
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


class FMIODESENS(FMIODE):
    """
    FMIODE extended with sensitivity simulation capabilities
    """
    def __init__(self, model, input=None, result_file_name='',
                 with_jacobian=False, start_time=0.0, parameters=None, logging=False, result_handler=None):

        # Call FMIODE init method
        FMIODE.__init__(self, model, input, result_file_name, with_jacobian,
                start_time,logging,result_handler)

        # Store the parameters
        if parameters is not None:
            if not isinstance(parameters,list):
                raise FMIModelException("Parameters must be a list of names.")
            self.p0 = np.array(model.get(parameters)).flatten()
            self.pbar = np.array([np.abs(x) if np.abs(x) > 0 else 1.0 for x in self.p0])
        self.parameters = parameters


    def rhs(self, t, y, p=None, sw=None):
        # Sets the parameters, if any
        if self.parameters is not None:
            self._model.set(self.parameters, p)

        return FMIODE.rhs(self,t,y,sw)


    def j(self, t, y, p=None, sw=None):

        # Sets the parameters, if any
        if self.parameters is not None:
            self._model.set(self.parameters, p)

        return FMIODE.j(self,t,y,sw)

    def handle_result(self, solver, t, y):
        #
        # Post processing (stores the time points).
        #
        time_start = timer()

        # Moving data to the model
        if t != self._model.time or (not self._f_nbr == 0 and not (self._model.continuous_states == y).all()):
            # Moving data to the model
            self._model.time = t
            # Check if there are any states
            if self._f_nbr != 0:
                self._model.continuous_states = y

            # Sets the inputs, if any
            if self.input is not None:
                self._model.set(self.input[0], self.input[1].eval(t)[0,:])

            # Evaluating the rhs (Have to evaluate the values in the model)
            rhs = self._model.get_derivatives()

        # Sets the parameters, if any
        if self.parameters is not None:
            p_data = np.array(solver.interpolate_sensitivity(t, 0)).flatten()

        self.export.integration_point(solver)# parameter_data=p_data)

        self.timings["handle_result"] += timer() - time_start
