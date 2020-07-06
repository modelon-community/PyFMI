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

"""
This file contains code for mapping FMUs to the Problem specifications
required by Assimulo.
"""
import logging
import logging as logging_module

import numpy as N
cimport numpy as N
import numpy.linalg as LIN
import scipy.sparse as sp
import time

from pyfmi.common.io import ResultWriterDymola
import pyfmi.fmi as fmi
from pyfmi.fmi cimport FMUModelME2
from pyfmi.common import python3_flag
from pyfmi.common.core import TrajectoryLinearInterpolation

from timeit import default_timer as timer
cimport fmil_import as FMIL

try:
    import assimulo
    assimulo_present = True
except:
    logging.warning(
        'Could not load Assimulo module. Check pyfmi.check_packages()')
    assimulo_present = False

if assimulo_present:
    from assimulo.problem import Implicit_Problem
    from assimulo.problem import Explicit_Problem
    from assimulo.exception import *
else:
    class Implicit_Problem:
        pass
    class Explicit_Problem:
        pass

class FMIModel_Exception(Exception):
    """
    A FMIModel Exception.
    """
    pass

def write_data(simulator,write_scaled_result=False, result_file_name=''):
    """
    Writes simulation data to a file. Takes as input a simulated model.
    """
    #Determine the result file name
    if result_file_name == '':
        result_file_name=simulator.problem._model.get_name()+'_result.txt'

    model = simulator.problem._model

    t = N.array(simulator.problem._sol_time)
    r = N.array(simulator.problem._sol_real)
    data = N.c_[t,r]
    if len(simulator.problem._sol_int) > 0 and len(simulator.problem._sol_int[0]) > 0:
        i = N.array(simulator.problem._sol_int)
        data = N.c_[data,i]
    if len(simulator.problem._sol_bool) > 0 and len(simulator.problem._sol_bool[0]) > 0:
        #b = N.array(simulator.problem._sol_bool).reshape(
        #    -1,len(model._save_bool_variables_val))
        b = N.array(simulator.problem._sol_bool)
        data = N.c_[data,b]

    export = ResultWriterDymola(model)
    export.write_header(file_name=result_file_name)
    map(export.write_point,(row for row in data))
    export.write_finalize()
    #fmi.export_result_dymola(model, data)

def createLogger(model, minimum_level):
    """
    Creates a logger.
    """
    filename = model.get_name()+'.log'

    log = logging.getLogger(filename)
    log.setLevel(minimum_level)

    #ch = logging.StreamHandler()
    ch = logging.FileHandler(filename, mode='w', delay=True)
    ch.setLevel(0)

    formatter = logging.Formatter("%(name)s - %(message)s")

    ch.setFormatter(formatter)

    log.addHandler(ch)

    return log

class FMIODE(Explicit_Problem):
    """
    An Assimulo Explicit Model extended to FMI interface.
    """
    def __init__(self, model, input=None, result_file_name='',
                 with_jacobian=False, start_time=0.0, logging=False, result_handler=None):
        """
        Initialize the problem.
        """
        self._model = model
        self._adapt_input(input)
        self.timings = {"handle_result": 0.0}

        #Set start time to the model
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

        #If there is no state in the model, add a dummy
        #state der(y)=0
        if f_nbr == 0:
            self.y0 = N.array([0.0])

        #Determine the result file name
        if result_file_name == '':
            self.result_file_name = model.get_name()+'_result.txt'
        else:
            self.result_file_name = result_file_name
        self.debug_file_name = model.get_name().replace(".","_")+'_debug.txt'
        self.debug_file_object = None

        #Default values
        self.export = result_handler
        
        #Internal values
        self._sol_time = []
        self._sol_real = []
        self._sol_int  = []
        self._sol_bool = []
        self._logg_step_event = []
        self._write_header = True
        self._logging = logging

        #Stores the first time point
        #[r,i,b] = self._model.save_time_point()

        #self._sol_time += [self._model.t]
        #self._sol_real += [r]
        #self._sol_int  += [i]
        #self._sol_bool += b
        
        self._jm_fmu = self._model.get_generation_tool() == "JModelica.org"

        if with_jacobian:
            raise fmi.FMUException("Jacobians are not supported using FMI 1.0, please use FMI 2.0")
    
    def _adapt_input(self, input):
        if input != None:
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
            self.input_alias_type = input_alias_type if N.any(input_alias_type==-1) else 1.0
        self.input = input
    
    def rhs(self, t, y, sw=None):
        """
        The rhs (right-hand-side) for an ODE problem.
        """
        #Moving data to the model
        self._model.time = t
        #Check if there are any states
        if self._f_nbr != 0:
            self._model.continuous_states = y

        #Sets the inputs, if any
        if self.input!=None:
            self._model.set_real(self.input_value_refs, self.input[1].eval(t)[0,:]*self.input_alias_type)

        #Evaluating the rhs
        try:
            rhs = self._model.get_derivatives()
        except fmi.FMUException:
            raise AssimuloRecoverableError

        #If there is no state, use the dummy
        if self._f_nbr == 0:
            rhs = N.array([0.0])

        return rhs

    def g(self, t, y, sw):
        """
        The event indicator function for a ODE problem.
        """
        #Moving data to the model
        self._model.time = t
        #Check if there are any states
        if self._f_nbr != 0:
            self._model.continuous_states = y

        #Sets the inputs, if any
        if self.input!=None:
            self._model.set_real(self.input_value_refs, self.input[1].eval(t)[0,:]*self.input_alias_type)

        #Evaluating the event indicators
        eventInd = self._model.get_event_indicators()

        return eventInd

    def t(self, t, y, sw):
        """
        Time event function.
        """
        eInfo = self._model.get_event_info()

        if eInfo.upcomingTimeEvent == True:
            return eInfo.nextEventTime
        else:
            return None


    def handle_result(self, solver, t, y):
        """
        Post processing (stores the time points).
        """
        time_start = timer()
        
        #Moving data to the model
        if t != self._model.time or (not self._f_nbr == 0 and not (self._model.continuous_states == y).all()):
            #Moving data to the model
            self._model.time = t
            #Check if there are any states
            if self._f_nbr != 0:
                self._model.continuous_states = y

            #Sets the inputs, if any
            if self.input!=None:
                self._model.set_real(self.input_value_refs, self.input[1].eval(t)[0,:]*self.input_alias_type)

            #Evaluating the rhs (Have to evaluate the values in the model)
            rhs = self._model.get_derivatives()
        
        if self.export != None:
            self.export.integration_point()
            
        self.timings["handle_result"] += timer() - time_start

    def handle_event(self, solver, event_info):
        """
        This method is called when Assimulo finds an event.
        """
        #Moving data to the model
        if solver.t != self._model.time or (not self._f_nbr == 0 and not (self._model.continuous_states == solver.y).all()):
            self._model.time = solver.t
            #Check if there are any states
            if self._f_nbr != 0:
                self._model.continuous_states = solver.y

            #Sets the inputs, if any
            if self.input!=None:
                self._model.set_real(self.input_value_refs, self.input[1].eval(N.array([solver.t]))[0,:]*self.input_alias_type)

            #Evaluating the rhs (Have to evaluate the values in the model)
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
                
            fwrite = self._get_debug_file_object()
            fwrite.write("\nDetected event at t = %.14E \n"%solver.t)
            fwrite.write(" State event info: "+" ".join(str(i) for i in event_info[0])+ "\n")
            fwrite.write(" Time  event info:  "+str(event_info[1])+ "\n")

        eInfo = self._model.get_event_info()
        eInfo.iterationConverged = False

        while eInfo.iterationConverged == False:
            self._model.event_update(intermediateResult=False)

            eInfo = self._model.get_event_info()
            #Retrieve solutions (if needed)
            #if eInfo.iterationConverged == False:
            #    pass

        #Check if the event affected the state values and if so sets them
        if eInfo.stateValuesChanged:
            if self._f_nbr == 0:
                solver.y[0] = 0.0
            else:
                solver.y = self._model.continuous_states

        #Get new nominal values.
        if eInfo.stateValueReferencesChanged:
            if self._f_nbr == 0:
                solver.atol = 0.01*solver.rtol*1
            else:
                solver.atol = 0.01*solver.rtol*self._model.nominal_continuous_states

        #Check if the simulation should be terminated
        if eInfo.terminateSimulation:
            raise TerminateSimulation #Exception from Assimulo

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
                
            fwrite.write(" Indicators (pre) : "+str_ind + "\n")
            fwrite.write(" Indicators (post): "+str_ind2+"\n")
            fwrite.write(" States (pre) : "+str_states + "\n")
            fwrite.write(" States (post): "+str_states2 + "\n")
            fwrite.write(" Derivatives (pre) : "+str_der + "\n")
            fwrite.write(" Derivatives (post): "+str_der2 + "\n\n")

            header = "Time (simulated) | Time (real) | "
            if solver.__class__.__name__=="CVode": #Only available for CVode
                header += "Order | Error (Weighted)"
            if self._g_nbr > 0:
                header += "Indicators"
            fwrite.write(header+"\n")
            
    def step_events(self, solver):
        """
        Method which is called at each successful step.
        """
        #Moving data to the model
        if solver.t != self._model.time or (not self._f_nbr == 0 and not (self._model.continuous_states == solver.y).all()):
            self._model.time = solver.t
            #Check if there are any states
            if self._f_nbr != 0:
                self._model.continuous_states = solver.y

            #Sets the inputs, if any
            if self.input!=None:
                self._model.set_real(self.input_value_refs, self.input[1].eval(N.array([solver.t]))[0,:]*self.input_alias_type)
                #self._model.set(self.input[0],self.input[1].eval(N.array([solver.t]))[0,:])

            #Evaluating the rhs (Have to evaluate the values in the model)
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
                
                #End tag
                msg = preface + "</%s>"%solver_name
                self._model.append_log_message("Model", 6, msg)
            
            data_line = "%.14E"%solver.t+" | %.14E"%(solver.get_elapsed_step_time())

            if solver.__class__.__name__=="CVode": #Only available for CVode
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
            #Event have been detect, call event iteration.
            #print "Step event detected at: ", solver.t
            #self.handle_event(solver,[0])
            return 1 #Tell to reinitiate the solver.
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
        if self._logging:
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
            if solver.__class__.__name__=="CVode": #Only available for CVode
                header += "Order | Error (Weighted)"
            f.write(header+"\n")

    def finalize(self, solver):
        if self.export != None:
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

        #Call FMIODE init method
        FMIODE.__init__(self, model, input, result_file_name, with_jacobian,
                start_time,logging,result_handler)

        #Store the parameters
        if parameters != None:
            if not isinstance(parameters,list):
                raise FMIModel_Exception("Parameters must be a list of names.")
            self.p0 = N.array(model.get(parameters)).flatten()
            self.pbar = N.array([N.abs(x) if N.abs(x) > 0 else 1.0 for x in self.p0])
        self.parameters = parameters


    def rhs(self, t, y, p=None, sw=None):
        #Sets the parameters, if any
        if self.parameters != None:
            self._model.set(self.parameters, p)

        return FMIODE.rhs(self,t,y,sw)


    def j(self, t, y, p=None, sw=None):

        #Sets the parameters, if any
        if self.parameters != None:
            self._model.set(self.parameters, p)

        return FMIODE.j(self,t,y,sw)

    def handle_result(self, solver, t, y):
        #
        #Post processing (stores the time points).
        #
        time_start = timer()
        
        #Moving data to the model
        if t != self._model.time or (not self._f_nbr == 0 and not (self._model.continuous_states == y).all()):
            #Moving data to the model
            self._model.time = t
            #Check if there are any states
            if self._f_nbr != 0:
                self._model.continuous_states = y

            #Sets the inputs, if any
            if self.input!=None:
                self._model.set(self.input[0], self.input[1].eval(t)[0,:])

            #Evaluating the rhs (Have to evaluate the values in the model)
            rhs = self._model.get_derivatives()

        #Sets the parameters, if any
        if self.parameters != None:
            p_data = N.array(solver.interpolate_sensitivity(t, 0)).flatten()

        self.export.integration_point(solver)#parameter_data=p_data)
        
        self.timings["handle_result"] += timer() - time_start


class FMIODE2(Explicit_Problem):
    """
    An Assimulo Explicit Model extended to FMI interface.
    """
    """
    cdef public int _f_nbr, _g_nbr, _input_activated, _extra_f_nbr, jac_nnz, input_len_names
    cdef public object _model, problem_name, result_file_name, __input, _A, debug_file_name, debug_file_object
    cdef public object export, _sparse_representation, _with_jacobian, _logging, _write_header
    cdef public dict timings
    cdef public N.ndarray y0
    cdef public list input_names, input_real_value_refs, input_real_mask, input_other, input_other_mask, _logg_step_event
    cdef public double t0
    cdef public jac_use, state_events_use, time_events_use
    cdef public FMUModelME2 model_me2
    cdef public int model_me2_instance
    cdef public N.ndarray _state_temp_1, _event_temp_1
    """
    
    def __init__(self, model, input=None, result_file_name='',
                 with_jacobian=False, start_time=0.0, logging=False, 
                 result_handler=None, extra_equations=None):
        """
        Initialize the problem.
        """
        self._model = model
        self._adapt_input(input)
        self.input_names = []
        self.timings = {"handle_result": 0.0}
        
        if type(model) == FMUModelME2: #isinstance(model, FMUModelME2):
            self.model_me2 = model
            self.model_me2_instance = 1
        else:
            self.model_me2_instance = 0

        #Set start time to the model
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

        #If there is no state in the model, add a dummy state der(y)=0
        if f_nbr == 0:
            self.y0 = N.array([0.0])

        #Determine the result file name
        if result_file_name == '':
            self.result_file_name = model.get_name()+'_result.txt'
        else:
            self.result_file_name = result_file_name
        self.debug_file_name = model.get_name().replace(".","_")+'_debug.txt'
        self.debug_file_object = None

        #Default values
        self.export = result_handler

        #Internal values
        self._logg_step_event = []
        self._write_header = True
        self._logging = logging
        self._sparse_representation = False
        self._with_jacobian = with_jacobian
        
        self.jac_use = False
        if f_nbr > 0 and with_jacobian:
            self.jac_use = True #Activates the jacobian
            
            #Need to calculate the nnz.
            [derv_state_dep, derv_input_dep] = model.get_derivatives_dependencies()
            self.jac_nnz = N.sum([len(derv_state_dep[key]) for key in derv_state_dep.keys()])+f_nbr
            
        if extra_equations:
            self._extra_f_nbr = extra_equations.get_size()
            self._extra_y0    = extra_equations.y0
            self.y0 = N.append(self.y0, self._extra_y0)
            self._extra_equations = extra_equations
            
            if hasattr(self._extra_equations, "jac"):
                if hasattr(self._extra_equations, "jac_nnz"):
                    self.jac_nnz += extra_equations.jac_nnz
                else:
                    self.jac_nnz += len(self._extra_f_nbr)*len(self._extra_f_nbr)
        else:
            self._extra_f_nbr = 0
        
        self._state_temp_1 = N.empty(f_nbr, dtype = N.double)
        self._event_temp_1 = N.empty(g_nbr, dtype = N.double)
    
    def _adapt_input(self, input):
        if input != None:
            input_names = input[0]
            self.input_len_names = len(input_names)
            self.input_real_value_refs = []
            self.input_real_mask = []
            self.input_other = []
            self.input_other_mask = []
            
            if isinstance(input_names,str):
                input_names = [input_names]
                
            for i,name in enumerate(input_names):
                if self._model.get_variable_causality(name) != fmi.FMI2_INPUT:
                    raise fmi.FMUException("Variable '%s' is not an input. Only variables specified to be inputs are allowed."%name)
                
                if self._model.get_variable_data_type(name) == fmi.FMI2_REAL:
                    self.input_real_value_refs.append(self._model.get_variable_valueref(name))
                    self.input_real_mask.append(i)
                else:
                    self.input_other.append(name)
                    self.input_other_mask.append(i)
            
            self.input_real_mask  = N.array(self.input_real_mask)
            self.input_other_mask = N.array(self.input_other_mask)
            
            self._input_activated = 1
        else:
            self._input_activated = 0

        self.input = input
        
    #cpdef _set_input_values(self, double t):
    def _set_input_values(self, t):
        if self._input_activated:
            values = self.input[1].eval(t)[0,:]
            
            if self.input_real_value_refs:
                self._model.set_real(self.input_real_value_refs, values[self.input_real_mask])
            if self.input_other:
                self._model.set(self.input_other, values[self.input_other_mask])
    
    #cdef _update_model(self, double t, N.ndarray[double, ndim=1, mode="c"] y):
    def _update_model(self, t, y):
        """
        if self.model_me2_instance:
            #Moving data to the model
            self.model_me2._set_time(t)
            #Check if there are any states
            if self._f_nbr != 0:
                self.model_me2.__set_continuous_states(y)
        else:
            #Moving data to the model
            self._model.time = t
            #Check if there are any states
            if self._f_nbr != 0:
                self._model.continuous_states = y
        """
        #Moving data to the model
        self._model.time = t
        #Check if there are any states
        if self._f_nbr != 0:
            self._model.continuous_states = y

        #Sets the inputs, if any
        self._set_input_values(t)
    
    #cdef int _compare(self, double t, N.ndarray[double, ndim=1, mode="c"] y):
    def _compare(self, t, y):
        """
        cdef int res
        
        if self.model_me2_instance:
            
            if t != self.model_me2._get_time():
                return 1
            
            if self._f_nbr == 0:
                return 0
            
            self.model_me2.__get_continuous_states(self._state_temp_1)
            res = FMIL.memcmp(self._state_temp_1.data, y.data, self._f_nbr*sizeof(double))
            
            if res == 0:
                return 0
            
            return 1
        else:
            return t != self._model.time or (not self._f_nbr == 0 and not (self._model.continuous_states == y).all())
        return 0
        """
        return t != self._model.time or (not self._f_nbr == 0 and not (self._model.continuous_states == y).all())

    #def rhs(self, double t, N.ndarray[double, ndim=1, mode="c"] y, sw=None):
    def rhs(self, t, y, sw=None):
        """
        The rhs (right-hand-side) for an ODE problem.
        """
        cdef int status

        if self._extra_f_nbr > 0:
            y_extra = y[-self._extra_f_nbr:]
            y       = y[:-self._extra_f_nbr]
        
        self._update_model(t, y)

        #Evaluating the rhs
        """
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
                self._state_temp_1 = self._model.get_derivatives()
            except fmi.FMUException:
                raise AssimuloRecoverableError
        """
        try:
            der = self._model.get_derivatives()
        except fmi.FMUException:
            raise AssimuloRecoverableError

        #If there is no state, use the dummy
        if self._f_nbr == 0:
            der = N.array([0.0])
        
        if self._extra_f_nbr > 0:
            der = N.append(der, self._extra_equations.rhs(y_extra))

        return der

    #def jac(self, double t, N.ndarray[double, ndim=1, mode="c"] y, sw=None):
    def jac(self, t, y, sw=None):
        """
        The jacobian function for an ODE problem.
        """
        if self._extra_f_nbr > 0:
            y_extra = y[-self._extra_f_nbr:]
            y       = y[:-self._extra_f_nbr]
            
        self._update_model(t, y)
        
        #Evaluating the jacobian
        
        #If there are no states return a dummy jacobian.
        if self._f_nbr == 0:
            return N.array([[0.0]])
        
        A = self._model._get_A(add_diag=True, output_matrix=self._A)
        if self._A is None:
            self._A = A

        if self._extra_f_nbr > 0:
            if hasattr(self._extra_equations, "jac"):
                if self._sparse_representation:
                    
                    Jac = A.tocoo() #Convert to COOrdinate
                    A2 = self._extra_equations.jac(y_extra).tocoo()
                    
                    data = N.append(Jac.data, A2.data)
                    row  = N.append(Jac.row, A2.row+self._f_nbr)
                    col  = N.append(Jac.col, A2.col+self._f_nbr)
                    
                    #Convert to compresssed sparse column
                    Jac = sp.coo_matrix((data, (row, col)))
                    Jac = Jac.tocsc()
                else:
                    Jac = N.zeros((self._f_nbr+self._extra_f_nbr,self._f_nbr+self._extra_f_nbr))
                    Jac[:self._f_nbr,:self._f_nbr] = A if isinstance(A, N.ndarray) else A.toarray()
                    Jac[self._f_nbr:,self._f_nbr:] = self._extra_equations.jac(y_extra)
            else:
                raise fmi.FMUException("No Jacobian provided for the extra equations")
        else:
            Jac = A

        return Jac

    #def state_events(self, double t, N.ndarray[double, ndim=1, mode="c"] y, sw):
    def state_events(self, t, y, sw):
        """
        The event indicator function for a ODE problem.
        """
        cdef int status
        
        if self._extra_f_nbr > 0:
            y_extra = y[-self._extra_f_nbr:]
            y       = y[:-self._extra_f_nbr]
            
        self._update_model(t, y)
        
        #Evaluating the event indicators
        """
        if self.model_me2_instance:
            status = self.model_me2._get_event_indicators(self._event_temp_1)
            
            if status != 0:
                raise fmi.FMUException('Failed to get the event indicators at time: %E.'%t)

            return self._event_temp_1
        else:
            return self._model.get_event_indicators()
        """
        return self._model.get_event_indicators()

    #def time_events(self, double t, N.ndarray[double, ndim=1, mode="c"] y, sw):
    def time_events(self, t, y, sw):
        """
        Time event function.
        """
        eInfo = self._model.get_event_info()

        if eInfo.nextEventTimeDefined == True:
            return eInfo.nextEventTime
        else:
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
            
        #Moving data to the model
        if self._compare(t, y):
            self._update_model(t, y)
        
            #Evaluating the rhs (Have to evaluate the values in the model)
            """
            if self.model_me2_instance:
                status = self.model_me2._get_derivatives(self._state_temp_1)
                
                if status != 0:
                    raise fmi.FMUException('Failed to get the derivatives at time: %E during handling of the result.'%t)
            else:
                rhs = self._model.get_derivatives()
            """
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

        #Moving data to the model
        if self._compare(solver.t, y):
            self._update_model(solver.t, y)

            #Evaluating the rhs (Have to evaluate the values in the model)
            """
            if self.model_me2_instance:
                status = self.model_me2._get_derivatives(self._state_temp_1)

                if status != 0:
                    raise fmi.FMUException('Failed to get the derivatives at time: %E during handling of the event.'%t)
            else:
                rhs = self._model.get_derivatives()
            """
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

        #Enter event mode
        self._model.enter_event_mode()
        
        self._model.event_update()
        eInfo = self._model.get_event_info()

        #Check if the event affected the state values and if so sets them
        if eInfo.valuesOfContinuousStatesChanged:
            if self._extra_f_nbr > 0:
                solver.y = self._model.continuous_states.append(solver.y[-self._extra_f_nbr:])
            else:
                solver.y = self._model.continuous_states

        #Get new nominal values.
        if eInfo.nominalsOfContinuousStatesChanged:
            solver.atol = 0.01*solver.rtol*self._model.nominal_continuous_states

        #Check if the simulation should be terminated
        if eInfo.terminateSimulation:
            raise TerminateSimulation #Exception from Assimulo

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
            
            fwrite = self._get_debug_file_object()
            fwrite.write(" Indicators (pre) : "+str_ind + "\n")
            fwrite.write(" Indicators (post): "+str_ind2+"\n")
            fwrite.write(" States (pre) : "+str_states + "\n")
            fwrite.write(" States (post): "+str_states2 + "\n")
            fwrite.write(" Derivatives (pre) : "+str_der + "\n")
            fwrite.write(" Derivatives (post): "+str_der2 + "\n\n")

            header = "Time (simulated) | Time (real) | "
            if solver.__class__.__name__=="CVode" or solver.__class__.__name__=="Radau5ODE": #Only available for CVode
                header += "Order | Error (Weighted)"
            if self._g_nbr > 0:
                header += "Indicators"
            fwrite.write(header+"\n")
            
        #Enter continuous mode again
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
            
        #Moving data to the model
        if self._compare(solver.t, y):
            self._update_model(solver.t, y)
        
            #Evaluating the rhs (Have to evaluate the values in the model)
            """
            if self.model_me2_instance:
                self.model_me2._get_derivatives(self._state_temp_1)
            else:
                rhs = self._model.get_derivatives()
            """
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

            fwrite = self._get_debug_file_object()
            fwrite.write(data_line+"\n")


            preface = "[INFO][FMU status:OK] "
            solver_info_tag = 'Solver'

            msg = preface + '<%s>Successful solver step at <value name="t">        %.14E</value>.'%(solver_info_tag, solver.t)
            self._model.append_log_message("Model", 6, msg)

            msg = preface + '  <value name="elapsed_real_time">        %.14E</value>'%(solver.get_elapsed_step_time())
            self._model.append_log_message("Model", 6, msg)

            solver_order = solver.get_last_order() if solver_name=="CVode" else 5 #hardcoded 5 for other solvers
            msg = preface + '  <value name="solver_order">%d</value>'%(solver_order)
            self._model.append_log_message("Model", 6, msg)

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


        if self.model_me2_instance:
            #self.model_me2._completed_integrator_step(&enter_event_mode, &terminate_simulation)
            enter_event_mode, terminate_simulation = self._model.completed_integrator_step()
        else:
            enter_event_mode, terminate_simulation = self._model.completed_integrator_step()
            
        if enter_event_mode:
            self._logg_step_event += [solver.t]
            #Event have been detect, call event iteration.
            self.handle_event(solver,[0])
            return 1 #Tell to reinitiate the solver.
        else:
            return 0
    
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
        
        if self._logging:
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
            if solver_name=="CVode" or solver_name=="Radau5ODE": #Only available for CVode
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
                 result_handler=None, extra_equations=None, parameters=None):

        #Call FMIODE init method
        FMIODE2.__init__(self, model, input, result_file_name, with_jacobian,
                start_time,logging, result_handler, extra_equations)

        #Store the parameters
        if parameters != None:
            if not isinstance(parameters,list):
                raise FMIModel_Exception("Parameters must be a list of names.")
            self.p0 = N.array(model.get(parameters)).flatten()
            self.pbar = N.array([N.abs(x) if N.abs(x) > 0 else 1.0 for x in self.p0])
            self.param_valref = [model.get_variable_valueref(x) for x in parameters]
            
            for param in parameters:
                if model.get_variable_causality(param) != fmi.FMI2_INPUT and \
                   (model.get_generation_tool() != "JModelica.org" and model.get_generation_tool() != "Optimica Compiler Toolkit"):
                    raise FMIModel_Exception("The sensitivity parameters must be specified as inputs!")
            
        self.parameters = parameters
        if python3_flag:
            self.derivatives = [v.value_reference for i,v in model.get_derivatives_list().items()]
        else:
            self.derivatives = [v.value_reference for i,v in model.get_derivatives_list().iteritems()]
        
        if self._model.get_capability_flags()['providesDirectionalDerivatives']:
            use_rhs_sens = True
            for param in parameters:
                if model.get_variable_causality(param) != fmi.FMI2_INPUT and \
                  (model.get_generation_tool() == "JModelica.org" or model.get_generation_tool() == "Optimica Compiler Toolkit"):
                    use_rhs_sens = False
                    logging_module.warning("The sensitivity parameters must be specified as inputs in order to set up the sensitivity " \
                            "equations using directional derivatives. Disabling and using finite differences instead.")
            
            if use_rhs_sens:
                self.rhs_sens = self.s #Activates the jacobian
        
        super(FMIODESENS2, self).rhs(0.0,self.y0,None)

    def rhs(self, t, y, p=None, sw=None):
        #Sets the parameters, if any
        if self.parameters != None:
            self._model.set(self.parameters, p)

        return FMIODE2.rhs(self,t,y,sw)
        
    def jac(self, t, y, p=None, sw=None):
        #Sets the parameters, if any
        if self.parameters != None:
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
        
