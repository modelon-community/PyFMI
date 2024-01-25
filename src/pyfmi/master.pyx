#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Copyright (C) 2010 Modelon AB
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

import pyfmi.fmi as fmi
from pyfmi.common.algorithm_drivers import OptionBase, InvalidAlgorithmOptionException, AssimuloSimResult
from pyfmi.common.io import get_result_handler
from pyfmi.common.core import TrajectoryLinearInterpolation
from pyfmi.common.core import TrajectoryUserFunction

from timeit import default_timer as timer

import fnmatch
import sys
import re
from collections import OrderedDict
import time
import numpy as np
import warnings
cimport numpy as np
import scipy.sparse as sp
import scipy.linalg as lin
import scipy.sparse.linalg as splin
import scipy.optimize as sopt
import scipy.version

from pyfmi.fmi cimport FMUModelCS2
from cpython cimport bool
cimport fmil_import as FMIL
from pyfmi.fmi_util import Graph
from cython.parallel import prange, parallel

IF WITH_OPENMP:
    cimport openmp

DEF SERIAL   = 0
DEF PARALLEL = 1

try:
    from numpy.lib import NumpyVersion
    USE_ROOT = NumpyVersion(scipy.version.version) >= "0.11.0"
except ImportError: #Numpy version is < 1.9.0 so assume scipy version is the same
    USE_ROOT = False

cdef reset_models(list models):
    for model in models:
        model.reset()

cdef perform_do_step(list models, dict time_spent, FMIL.fmi2_import_t** model_addresses, double cur_time, double step_size, bool new_step, int setting):
    if setting == SERIAL:
        perform_do_step_serial(models, time_spent, cur_time,  step_size,  new_step)
    else:
        perform_do_step_parallel(models, model_addresses, len(models), cur_time,  step_size,  new_step)

cdef perform_do_step_serial(list models, dict time_spent, double cur_time, double step_size, bool new_step):
    """
    Perform a do step on all the models.
    """
    cdef double time_start = 0.0
    cdef int status = 0
    
    for model in models: 
        time_start = timer()
        status = model.do_step(cur_time, step_size, new_step)
        time_spent[model] += timer() - time_start
        
        if status != 0:
            raise fmi.FMUException("The step failed for model %s at time %f. See the log for more information. Return flag %d."%(model.get_name(), cur_time, status))

cdef perform_do_step_parallel(list models, FMIL.fmi2_import_t** model_addresses, int n, double cur_time, double step_size, int new_step):
    """
    Perform a do step on all the models.
    """
    cdef int i, status = 0
    cdef int num_threads, id
    cdef double time
    
    for i in prange(n, nogil=True, schedule="dynamic", chunksize=1): 
    #for i in prange(n, nogil=True, schedule="dynamic"): 
        #num_threads = openmp.omp_get_num_threads()
        #id = openmp.omp_get_thread_num()
        #time = openmp.omp_get_wtime() 
        
        status |= FMIL.fmi2_import_do_step(model_addresses[i], cur_time, step_size, new_step)
        #time = openmp.omp_get_wtime() -time
        #printf("Time: %f (%d), %d, %d, Elapsed Time: %f \n",cur_time, i, num_threads, id,  time)

    if status != 0:
        raise fmi.FMUException("The simulation failed. See the log for more information. Return flag %d."%status)
    
    #Update local times in models
    for model in models:
        model.time = cur_time + step_size

cdef enter_initialization_mode(list models, double start_time, double final_time, object opts, dict time_spent):
    cdef int status
    for model in models:
        time_start = timer()
        model.setup_experiment(tolerance=opts["local_rtol"], start_time=start_time, stop_time=final_time)
        
        try:
            status = model.enter_initialization_mode()
            time_spent[model] += timer() - time_start
        except fmi.FMUException:
            print("The model, '" + model.get_name() + "' failed to enter initialization mode. ")
        
cdef exit_initialization_mode(list models, dict time_spent):
    for model in models:
        time_start = timer()
        model.exit_initialization_mode()
        time_spent[model] += timer() - time_start

"""
cdef perform_initialize(list models, double start_time, double final_time, object opts):
    #
    Initialize all the models.
    #
    for model in models:
        model.setup_experiment(tolerance=opts["local_rtol"], start_time=start_time, stop_time=final_time)
        model.initialize()
"""

#cdef get_fmu_states(list models):
cdef get_fmu_states(list models, dict states_dict = None):
    """
    Get the FMU states for all the models
    """
    if states_dict is None:
        return {model: model.get_fmu_state() for model in models}
    else:
        for model in states_dict.keys():
            states_dict[model] = model.get_fmu_state(states_dict[model])
        return states_dict
    
cdef set_fmu_states(dict states_dict):
    """
    Sets the FMU states for all the models
    """
    for model in states_dict.keys():
        model.set_fmu_state(states_dict[model])
        
cdef free_fmu_states(dict states_dict):
    """
    Free the FMU states for all the models
    """
    for model in states_dict.keys():
        model.free_fmu_state(states_dict[model])
        
cdef store_communication_point(object models_dict):
    for model in models_dict.keys():
        models_dict[model]["result"].integration_point()

cdef finalize_result_objects(object models_dict):
    for model in models_dict.keys():
        models_dict[model]["result"].simulation_end()
        
def init_f(y, master):
    y = y.reshape(-1, 1)
    u = master.L.dot(y)
    master.set_connection_inputs(u)
    temp_y = y - master.get_connection_outputs()

    return temp_y.flatten()
    
def init_f_block(ylocal, master, block):
    y = np.zeros((master._len_outputs))
    y[block["global_outputs_mask"]] = ylocal
    y = y.reshape(-1,1)
    
    ytmp = np.zeros((master._len_outputs))
    u = master.L.dot(y.reshape(-1,1))
    
    for model in block["inputs"].keys(): #Set the inputs
        master.set_specific_connection_inputs(model, block["inputs_mask"][model], u)
    for model in block["outputs"].keys(): #Get the outputs 
        master.get_specific_connection_outputs(model, block["outputs_mask"][model], ytmp)
    
    res = y - ytmp.reshape(-1,1)

    return res.flatten()[block["global_outputs_mask"]]
    
def init_jac(y, master):
    y = y.reshape(-1, 1)
    u = master.L.dot(y)
    master.set_connection_inputs(u)
    D = master.compute_global_D()
    DL = D.dot(master.L)
    return np.eye(*DL.shape) - DL
    
class MasterAlgOptions(OptionBase):
    """
    Options for solving coupled FMI 2 CS FMUs.

    Options::

        step_size --
            Specfies the global step-size to be used for simulating
            the coupled system.
            Default: 0.01
            
        initialize --
            If set to True, the initializing algorithm defined in the FMU models
            are invoked, otherwise it is assumed the user have manually initialized
            all models.
            Default is True.
            
        block_initialization --
            If set to True, the initialization algorithm computes the evaluation
            order of the FMUs and tries to resolve algebraic loops by this
            evaluation order.
            Default is False.
        
        extrapolation_order --
            Defines the extrapolation used in the simulation.
            Default is 0 (constant extrapolation).
        
        smooth_coupling --
            Defines if the extrapolation should be smoothen, i.e. the input
            values are adapted so that they are C^0 instead of C^(-1) in case
            extrapolation_order is > 0.
            Default is True
        
        linear_correction --
            Defines if linear correction should be used during the simulation.
            Note that this increases the simulation robustness in case of 
            algebraic loops.
            Default is False

        execution --
            Defines if the models are to be evaluated in parallel (note that it
            is not an algorithm change, just an evaluation execution within
            the same algorithm). Note that it requires that PyFMI has been
            installed with OpenMP.
            Default is serial
            
        num_threads --
            Defines the number of threads used when the execution is set
            to parallel.
            Default: Number of cores / OpenMP environment variable
            
        error_controlled --
            Defines if the algorithm should adapt the step-size during
            the simulation. Note requires that all FMUs support save/get
            state.
            Default: False
            
        atol --
            Defines the absolute tolerance used when an error controlled
            simulation is performed.
            Default: 1e-4

        rtol --
            Defines the relative tolerance used when an error controlled
            simulation is performed.
            Default: 1e-4
            
        maxh --
            Defines the maximum step-size allowed to be used together
            with an error controlled simulation.
            Default: 0.0 (i.e. inactive)

        local_rtol --
            Defines the relative tolerance that will be provided to the 
            connected FMUs during initialization of the underlying 
            models. Note, only have an effect if 'initialize' is set to
            True.
            Default: 1e-6

        result_file_name --
            Specifies the name of the file where the simulation result is
            written. Note that there should be one name for each model and
            the names of the files should be provided in a dict with the
            model as 'key' and the name of the file as 'value' in the
            dict.
            Default: Model name + "_result." + filetype

        result_handling --
            Specifies how the result should be handled. Either stored to
            file or stored in memory. One can also use a custom handler.
            Available options: "file", "binary", "memory", "csv", "custom"
            Default: "binary"

        result_handler --
            The handler for the result. Depending on the option in
            result_handling this either defaults to ResultHandlerFile
            or ResultHandlerMemory. If result_handling custom is choosen
            This MUST be provided.
            Default: None

        filter --
            A filter for choosing which variables to actually store
            result for. The syntax can be found in
            http://en.wikipedia.org/wiki/Glob_%28programming%29 . An
            example is filter = "*der" , stor all variables ending with
            'der'. Can also be a list. Note that there should be one
            filter for each model.
            Default: None
            
        logging --
            If True, store additional debug data during the simulation.
            The option is currently only useful for internal purposes.
            Default: False
        
        store_step_before_update --
            If True, store additionally the values in the underlying 
            FMUs to the result file before data has been exchange 
            between the connected models. The values will always be 
            stored after data has been exhanged between the connected
            models.
            Default: False
        
        block_initialization_type --
            Specifies which algorithm should be used to find the best
            grouping of input/outputs. This only has an effect when the
            option 'block_initialization' is set to True. The available
            values are: "greedy", "simply", "grouping". Each has their
            pros and cons, for best performance, please test.
            Default: "greedy"

        force_finite_difference_outputs --
            If set, forces the use of finite difference (first order)
            between communication points. I.e. instead of using the 
            underlying models capabilities to compute derivatives of 
            outputs, finite differences between communication points 
            will be used.
            Default: False
    """
    def __init__(self, master, *args, **kw):
        _defaults= {
        "initialize" : True,
        "local_rtol" : 1e-6,
        "rtol"       : 1e-4,
        "atol"       : 1e-4,
        "step_size"  : 0.01,
        "maxh"       : 0.0,
        "filter"     : dict((model,None) for model in master.models),
        "result_file_name"    : dict((model,None) for model in master.models),
        "result_handling"     : "binary",
        "result_handler"      : None,
        "linear_correction"   : False,
        "error_controlled"    : False if master.support_storing_fmu_states else False,
        "logging"             : False,
        "extrapolation_order" : 0, #Constant
        "store_step_before_update" : False,
        "smooth_coupling"             : True,
        "execution"                : "serial",
        "block_initialization"     : False,
        "block_initialization_type" : "greedy",
        "experimental_block_initialization_order" : None,
        "experimental_output_derivative": False,
        "experimental_finite_difference_D": False,
        "experimental_output_solve":False,
        "force_finite_difference_outputs": False,
        "num_threads":None}
        super(MasterAlgOptions,self).__init__(_defaults)
        self._update_keep_dict_defaults(*args, **kw)

cdef class Master:
    cdef public list connections, models
    cdef public dict statistics, models_id_mapping
    cdef public object opts
    cdef public object models_dict, L, L_discrete
    cdef public object _ident_matrix
    cdef public object y_prev, yd_prev, input_traj
    cdef public object DL_prev
    cdef public int algebraic_loops, storing_fmu_state
    cdef public int error_controlled, linear_correction
    cdef public int _support_directional_derivatives, _support_storing_fmu_states, _support_interpolate_inputs, _max_output_derivative_order
    cdef double rtol, atol, current_step_size
    cdef public object y_m1, yd_m1, u_m1, ud_m1, udd_m1
    cdef FMIL.fmi2_import_t** fmu_adresses
    cdef public int _len_inputs, _len_inputs_discrete, _len_outputs, _len_outputs_discrete
    cdef public int _len_derivatives
    cdef public list _storedDrow, _storedDcol
    cdef public np.ndarray _array_one
    cdef public object _D
    cdef public dict elapsed_time
    cdef public dict elapsed_time_init
    cdef public dict _error_data
    cdef public int _display_counter
    cdef public object _display_progress
    cdef public double _time_integration_start
    
    def __init__(self, models, connections):
        """
        Initializes the master algorithm.
        
        Parameters::
        
            models  
                    - A list of models that are to be simulated.
                      Needs to be a subclass of FMUModelCS.
                      
            connection  
                    - Specifices the connection between the models.
                        
                    - model_begin.variable -> model_accept.variable
                      [(model_source,"beta",model_destination,"y"),(...)]
        
        """
        if not isinstance(models, list):
            raise fmi.FMUException("The models should be provided as a list.")
        for model in models:
            if not isinstance(model, fmi.FMUModelCS2):
                raise fmi.InvalidFMUException("The Master algorithm currently only supports CS 2.0 FMUs.")
        self.fmu_adresses = <FMIL.fmi2_import_t**>FMIL.malloc(len(models)*sizeof(FMIL.fmi2_import_t*))
        
        self.connections = connections
        self.models = models
        self.models_dict = OrderedDict((model,{"model": model, "result": None, "external_input": None, 
                                               "local_input": [], "local_input_vref": [], "local_input_len": 0,
                                               "local_input_discrete": [], "local_input_discrete_vref": [], "local_input_discrete_len": 0,
                                               "local_state": [], "local_state_vref": [],
                                               "local_derivative": [], "local_derivative_vref": [],
                                               "local_output": [], "local_output_vref": [], "local_output_len": 0,
                                               "local_output_discrete": [], "local_output_discrete_vref": [], "local_output_discrete_len": 0,
                                               "local_output_range_array": None,
                                               "direct_dependence": []}) for model in models)
        self.models_id_mapping = {str(id(model)): model for model in models}
        self.elapsed_time = {model: 0.0 for model in models}
        self.elapsed_time_init = {model: 0.0 for model in models}
        self.elapsed_time["result_handling"] = 0.0
        self._display_counter = 1
        self._display_progress = True
        
        self.statistics = {}
        self.statistics["nsteps"] = 0
        self.statistics["nreject"] = 0
        
        #Initialize internal variables
        self._support_directional_derivatives = -1
        self._support_storing_fmu_states = -1
        self._support_interpolate_inputs = -1
        self._max_output_derivative_order = -1
        self._len_inputs = 0
        self._len_outputs = 0
        self._len_inputs_discrete = 0
        self._len_outputs_discrete = 0
        self._len_derivatives = 0
        self._array_one = np.array([1.0])
        self._D = None
        
        self.error_controlled = 0
        self.linear_correction = 1
        
        self.check_support_storing_fmu_state()
        
        self.connection_setup(connections)
        self.verify_connection_variables()
        self.check_algebraic_loops()
        self.set_model_order()
        self.define_connection_matrix()
        
        self.y_prev = None
        self.input_traj = None
        self._ident_matrix = sp.eye(self._len_inputs, self._len_outputs, format="csr") #y = Cx + Du , u = Ly -> DLy   DL[inputsXoutputs]
        
        self._error_data = {"time":[], "error":[], "step-size":[], "rejected":[]}
    
    def __del__(self):
        FMIL.free(self.fmu_adresses)
    
    cdef set_last_y(self, np.ndarray y):
        self.y_m1 = y.copy()
    cdef get_last_y(self):
        return self.y_m1
    cdef set_last_yd(self, np.ndarray yd):
        self.yd_m1 = yd.copy() if yd is not None else None
    cdef get_last_yd(self):
        return self.yd_m1
    cdef set_last_us(self, np.ndarray u, np.ndarray ud=None, np.ndarray udd=None):
        self.u_m1 = u.copy()
        self.ud_m1 = ud.copy() if ud is not None else None
        self.udd_m1 = udd.copy() if udd is not None else None
    cdef get_last_us(self):
        return self.u_m1, self.ud_m1, self.udd_m1
    cdef set_current_step_size(self, double step_size):
        self.current_step_size = step_size
    cdef double get_current_step_size(self):
        return self.current_step_size
        
    def report_solution(self, double cur_time):
        
        store_communication_point(self.models_dict)
        
        if self._display_progress:
            if ( timer() - self._time_integration_start) > self._display_counter*10:
                self._display_counter += 1
                
                sys.stdout.write(" Simulation time: %e" % cur_time)
                sys.stdout.write('\r')
                sys.stdout.flush()
        
    def set_model_order(self):
        i = 0
        for model in self.models_dict.keys():
            self.models_dict[model]["order"] = i
            i = i+1
        for model in self.models_dict.keys():
            self.models[self.models_dict[model]["order"]] = model
            
    def copy_fmu_addresses(self):
        for model in self.models_dict.keys():
            self.fmu_adresses[self.models_dict[model]["order"]] = (<FMUModelCS2>model)._fmu
            
    def define_connection_matrix(self):
        cdef list data = []
        cdef list row = []
        cdef list col = []
        cdef list data_discrete = []
        cdef list row_discrete = []
        cdef list col_discrete = []
        cdef int len_connections = 0
        cdef int len_connections_discrete = 0

        start_index_inputs            = 0
        start_index_outputs           = 0     
        start_index_inputs_discrete   = 0
        start_index_outputs_discrete  = 0
        start_index_states            = 0
        start_index_derivatives       = 0
        
        for model in self.models_dict.keys():
            self.models_dict[model]["global_index_inputs"]      = start_index_inputs
            self.models_dict[model]["global_index_outputs"]     = start_index_outputs
            self.models_dict[model]["global_index_inputs_discrete"]      = start_index_inputs_discrete
            self.models_dict[model]["global_index_outputs_discrete"]     = start_index_outputs_discrete
            self.models_dict[model]["global_index_states"]      = start_index_states
            self.models_dict[model]["global_index_derivatives"] = start_index_derivatives
            
            start_index_inputs      += len(self.models_dict[model]["local_input"])
            start_index_outputs     += len(self.models_dict[model]["local_output"])
            start_index_inputs_discrete      += len(self.models_dict[model]["local_input_discrete"])
            start_index_outputs_discrete     += len(self.models_dict[model]["local_output_discrete"])
            start_index_states      += len(self.models_dict[model]["local_state"])
            start_index_derivatives += len(self.models_dict[model]["local_derivative"])
        
        for connection in self.connections:
            src = connection[0]; src_var = connection[1]
            dst = connection[2]; dst_var = connection[3]
            
            if connection[0].get_variable_variability(connection[1]) == fmi.FMI2_CONTINUOUS and \
               connection[2].get_variable_variability(connection[3]) == fmi.FMI2_CONTINUOUS:
                   
                data.append(1)
                row.append(self.models_dict[dst]["global_index_inputs"]+self.models_dict[dst]["local_input"].index(dst_var))
                col.append(self.models_dict[src]["global_index_outputs"]+self.models_dict[src]["local_output"].index(src_var))
                len_connections = len_connections + 1
            else:
                data_discrete.append(1)
                row_discrete.append(self.models_dict[dst]["global_index_inputs_discrete"]+self.models_dict[dst]["local_input_discrete"].index(dst_var))
                col_discrete.append(self.models_dict[src]["global_index_outputs_discrete"]+self.models_dict[src]["local_output_discrete"].index(src_var))
                len_connections_discrete = len_connections_discrete + 1
            
        self.L = sp.csr_matrix((data, (row, col)), (len_connections,len_connections), dtype=np.float64)
        self.L_discrete = sp.csr_matrix((data_discrete, (row_discrete, col_discrete)), (len_connections_discrete,len_connections_discrete), dtype=np.float64)
        
    cpdef compute_global_D(self):
        cdef list data = []
        cdef list row = []
        cdef list col = []
        cdef int i, nlocal, status

        for model in self.models_dict.keys():
            nlocal = self.models_dict[model]["local_input_len"]
            #v = [0.0]*nlocal
            for i in range(nlocal):
                #local_D = model.get_directional_derivative([self.models_dict[model]["local_input_vref"][i]],self.models_dict[model]["local_output_vref"], [1.0])
                local_D = np.empty(self.models_dict[model]["local_output_len"])
                #status = (<FMUModelCS2>model)._get_directional_derivative(np.array([self.models_dict[model]["local_input_vref"][i]]),self.models_dict[model]["local_output_vref_array"], self._array_one, local_D)
                if self.opts["experimental_finite_difference_D"]:
                    up = (<FMUModelCS2>model).get_real(self.models_dict[model]["local_input_vref_array"][i:i+1])
                    eps = max(abs(up), 1.0)
                    yp = (<FMUModelCS2>model).get_real(self.models_dict[model]["local_output_vref_array"])
                    (<FMUModelCS2>model).set_real(self.models_dict[model]["local_input_vref_array"][i:i+1], up+eps)
                    local_D = ((<FMUModelCS2>model).get_real(self.models_dict[model]["local_output_vref_array"]) - yp)/eps
                    (<FMUModelCS2>model).set_real(self.models_dict[model]["local_input_vref_array"][i:i+1], up)
                else:
                    status = (<FMUModelCS2>model)._get_directional_derivative(self.models_dict[model]["local_input_vref_array"][i:i+1],self.models_dict[model]["local_output_vref_array"], self._array_one, local_D)
            
                    if status != 0: raise fmi.FMUException("Failed to get the directional derivatives while computing the global D matrix.")
                data.extend(local_D)
                
                if self._storedDrow is None and self._storedDcol is None:
                    col.extend([self.models_dict[model]["global_index_inputs"]+i]*len(local_D))
                    #row.extend(np.array([self.models_dict[model]["global_index_outputs"]]*self.models_dict[model]["local_output_len"])+np.array(range(self.models_dict[model]["local_output_len"])))
                    row.extend(np.array([self.models_dict[model]["global_index_outputs"]]*self.models_dict[model]["local_output_len"])+self.models_dict[model]["local_output_range_array"])
        
        if self._storedDrow is None and self._storedDcol is None:
            self._storedDrow = row
            self._storedDcol = col
        else:
            row = self._storedDrow
            col = self._storedDcol
        
        if self._D is None:
            self._D = sp.csr_matrix((data, (row, col)))#, (len(col),len(row)))
        else:
            self._D.data = np.array(data, dtype=np.float64)
            
        return self._D
            
    
    def compute_global_C(self):
        cdef list data = []
        cdef list row = []
        cdef list col = []

        for model in self.models_dict.keys():
            if model.get_generation_tool() != "JModelica.org":
                return None
            v = [0.0]*len(self.models_dict[model]["local_state_vref"])
            for i in range(len(v)):
                local_C = model.get_directional_derivative([self.models_dict[model]["local_state_vref"][i]],self.models_dict[model]["local_output_vref"], [1.0])
                data.extend(local_C)
                col.extend([self.models_dict[model]["global_index_states"]+i]*len(local_C))
                row.extend(np.array([self.models_dict[model]["global_index_outputs"]]*len(self.models_dict[model]["local_output_vref"]))+np.array(range(len(self.models_dict[model]["local_output_vref"]))))
        return sp.csr_matrix((data, (row, col)))
        
    def compute_global_A(self):
        cdef list data = []
        cdef list row = []
        cdef list col = []

        for model in self.models_dict.keys():
            if model.get_generation_tool() != "JModelica.org":
                return None
            v = [0.0]*len(self.models_dict[model]["local_state_vref"])
            for i in range(len(v)):
                local_A = model.get_directional_derivative([self.models_dict[model]["local_state_vref"][i]],self.models_dict[model]["local_derivative_vref"], [1.0])
                data.extend(local_A)
                col.extend([self.models_dict[model]["global_index_states"]+i]*len(local_A))
                row.extend(np.array([self.models_dict[model]["global_index_derivatives"]]*len(self.models_dict[model]["local_derivative_vref"]))+np.array(range(len(self.models_dict[model]["local_derivative_vref"]))))
        return sp.csr_matrix((data, (row, col)))
        
    def compute_global_B(self):
        cdef list data = []
        cdef list row = []
        cdef list col = []

        for model in self.models_dict.keys():
            if model.get_generation_tool() != "JModelica.org":
                return None
            v = [0.0]*len(self.models_dict[model]["local_input_vref"])
            for i in range(len(v)):
                local_B = model.get_directional_derivative([self.models_dict[model]["local_input_vref"][i]],self.models_dict[model]["local_derivative_vref"], [1.0])
                data.extend(local_B)
                col.extend([self.models_dict[model]["global_index_inputs"]+i]*len(local_B))
                row.extend(np.array([self.models_dict[model]["global_index_derivatives"]]*len(self.models_dict[model]["local_derivative_vref"]))+np.array(range(len(self.models_dict[model]["local_derivative_vref"]))))
        return sp.csr_matrix((data, (row, col)))
        
    def connection_setup(self, connections):
        for connection in connections:
            if connection[0].get_variable_variability(connection[1]) == fmi.FMI2_CONTINUOUS and \
               connection[2].get_variable_variability(connection[3]) == fmi.FMI2_CONTINUOUS:
                self.models_dict[connection[0]]["local_output"].append(connection[1])
                self.models_dict[connection[0]]["local_output_vref"].append(connection[0].get_variable_valueref(connection[1]))
                self.models_dict[connection[2]]["local_input"].append(connection[3])
                self.models_dict[connection[2]]["local_input_vref"].append(connection[2].get_variable_valueref(connection[3]))
            else:
                self.models_dict[connection[0]]["local_output_discrete"].append(connection[1])
                self.models_dict[connection[0]]["local_output_discrete_vref"].append(connection[0].get_variable_valueref(connection[1]))
                self.models_dict[connection[2]]["local_input_discrete"].append(connection[3])
                self.models_dict[connection[2]]["local_input_discrete_vref"].append(connection[2].get_variable_valueref(connection[3]))
                
        for model in self.models_dict.keys():
            self.models_dict[model]["local_input_len"] = len(self.models_dict[model]["local_input"])
            self.models_dict[model]["local_output_len"] = len(self.models_dict[model]["local_output"])
            self.models_dict[model]["local_input_discrete_len"] = len(self.models_dict[model]["local_input_discrete"])
            self.models_dict[model]["local_output_discrete_len"] = len(self.models_dict[model]["local_output_discrete"])
            self.models_dict[model]["local_output_range_array"] = np.array(range(self.models_dict[model]["local_output_len"]))
            self.models_dict[model]["local_output_vref_array"] = np.array(self.models_dict[model]["local_output_vref"], dtype=np.uint32)
            self.models_dict[model]["local_input_vref_array"] = np.array(self.models_dict[model]["local_input_vref"], dtype=np.uint32)
            self.models_dict[model]["local_input_vref_ones"] = np.ones(self.models_dict[model]["local_input_len"], dtype=np.int32)
            self.models_dict[model]["local_input_vref_twos"] = 2*np.ones(self.models_dict[model]["local_input_len"], dtype=np.int32)
            self.models_dict[model]["local_output_vref_ones"] = np.ones(self.models_dict[model]["local_output_len"], dtype=np.int32)
            self._len_inputs  += self.models_dict[model]["local_input_len"]
            self._len_outputs += self.models_dict[model]["local_output_len"]
            self._len_inputs_discrete  += self.models_dict[model]["local_input_discrete_len"]
            self._len_outputs_discrete += self.models_dict[model]["local_output_discrete_len"]
            
            if model.get_generation_tool() == "JModelica.org":
                self.models_dict[model]["local_state"]           = model.get_states_list().keys()
                self.models_dict[model]["local_state_vref"]      = [var.value_reference for var in model.get_states_list().values()]
                self.models_dict[model]["local_derivative"]      = model.get_derivatives_list().keys()
                self.models_dict[model]["local_derivative_vref"] = [var.value_reference for var in model.get_derivatives_list().values()]
                self.models_dict[model]["local_derivative_vref_array"] = np.array(self.models_dict[model]["local_derivative_vref"], dtype=np.uint32)
                self.models_dict[model]["local_derivative_len"] = len(self.models_dict[model]["local_derivative"])
                self._len_derivatives += self.models_dict[model]["local_derivative_len"]
                
    def verify_connection_variables(self):
        for model in self.models_dict.keys():
            for output in self.models_dict[model]["local_output"]:
                if model.get_variable_causality(output) != fmi.FMI2_OUTPUT:
                    raise fmi.FMUException("The connection variable " + output + " in model " + model.get_name() + " is not an output. ")
            for output in self.models_dict[model]["local_output_discrete"]:
                if model.get_variable_causality(output) != fmi.FMI2_OUTPUT:
                    raise fmi.FMUException("The connection variable " + output + " in model " + model.get_name() + " is not an output. ")
            for input in self.models_dict[model]["local_input"]:
                if model.get_variable_causality(input) != fmi.FMI2_INPUT:
                    raise fmi.FMUException("The connection variable " + input + " in model " + model.get_name() + " is not an input. ")
            for input in self.models_dict[model]["local_input_discrete"]:
                if model.get_variable_causality(input) != fmi.FMI2_INPUT:
                    raise fmi.FMUException("The connection variable " + input + " in model " + model.get_name() + " is not an input. ")
                    
    def check_algebraic_loops(self):
        """
        Simplified check for algebraic loops in simulation mode due to
        the limited capacity of solving the loops
        """
        self.algebraic_loops = 0
        
        for model in self.models_dict.keys():
            output_state_dep, output_input_dep = model.get_output_dependencies()
            for local_output in self.models_dict[model]["local_output"]:
                output_input_dep_dict = {key: i for i, key in enumerate(output_input_dep[local_output])}
                for local_input in self.models_dict[model]["local_input"]:
                    if local_input in output_input_dep_dict:
                        self.models_dict[model]["direct_dependence"].append((local_input, local_output))
                        self.algebraic_loops = 1
                        #break
                if self.algebraic_loops:
                    pass
                    #break
            if self.algebraic_loops:
                pass
                #break
                    
        if self.algebraic_loops:
            for model in self.models_dict.keys():
                if model.get_capability_flags()["providesDirectionalDerivatives"] is False:
                    warnings.warn("The model, " + model.get_name() + ", does not support " 
                                "directional derivatives which is necessary in-case of an algebraic loop. The simulation might become unstable...")
        
        return self.algebraic_loops
        
    def check_support_storing_fmu_state(self):
        self.storing_fmu_state = 1
        for model in self.models_dict.keys():
            if model.get_capability_flags()["canGetAndSetFMUstate"] is False:
                self.storing_fmu_state= 0
                break
        return self.storing_fmu_state
        
    cpdef np.ndarray get_connection_outputs(self):
        cdef int i, index, index_start, index_end
        cdef np.ndarray y = np.empty((self._len_outputs))

        for model in self.models:
            index_start = self.models_dict[model]["global_index_outputs"]
            index_end = index_start + self.models_dict[model]["local_output_len"]
            local_output_vref_array = (<FMUModelCS2>model).get_real(self.models_dict[model]["local_output_vref_array"])
            for i, index in enumerate(range(index_start, index_end)):
                y[index] = local_output_vref_array[i]
        return y.reshape(-1,1)
    
    cpdef np.ndarray get_connection_outputs_discrete(self):
        cdef int i, index, index_start, index_end
        cdef np.ndarray y = np.empty((self._len_outputs_discrete))

        for model in self.models:
            index_start = self.models_dict[model]["global_index_outputs_discrete"]
            index_end = index_start + self.models_dict[model]["local_output_discrete_len"]
            local_output_discrete = model.get(self.models_dict[model]["local_output_discrete"])
            for i, index in enumerate(range(index_start, index_end)):
                y[index] = local_output_discrete[i]
        return y.reshape(-1,1)
        
    cpdef np.ndarray _get_derivatives(self):
        cdef int i, index, index_start, index_end
        cdef np.ndarray xd = np.empty((self._len_derivatives))
        
        for model in self.models_dict.keys():
            if model.get_generation_tool() != "JModelica.org":
                return None

        for model in self.models:
            index_start = self.models_dict[model]["global_index_derivatives"]
            index_end = index_start + self.models_dict[model]["local_derivative_len"]
            local_derivative_vref_array = (<FMUModelCS2>model).get_real(self.models_dict[model]["local_derivative_vref_array"])
            for i, index in enumerate(range(index_start, index_end)):
                xd[index] = local_derivative_vref_array[i]

        return xd.reshape(-1,1)
    
    cpdef np.ndarray get_specific_connection_outputs_discrete(self, model, np.ndarray mask, np.ndarray yout):
        cdef int j = 0
        ytmp = model.get(np.array(self.models_dict[model]["local_output_discrete"])[mask])
        for i, flag in enumerate(mask):
            if flag:
                yout[i+self.models_dict[model]["global_index_outputs_discrete"]] = ytmp[j]
                j = j + 1
                
    cpdef np.ndarray get_specific_connection_outputs(self, model, np.ndarray mask, np.ndarray yout):
        cdef int j = 0
        cdef np.ndarray ytmp = (<FMUModelCS2>model).get_real(self.models_dict[model]["local_output_vref_array"][mask])
        for i, flag in enumerate(mask):
            if flag:
                yout[i+self.models_dict[model]["global_index_outputs"]] = ytmp[j]
                j = j + 1
        
    cpdef get_connection_derivatives(self, np.ndarray y_cur):
        #cdef list yd = []
        cdef int i = 0, inext = 0, status = 0
        cdef np.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c']  yd    = np.empty((self._len_outputs))
        cdef np.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c']  ydtmp = np.empty((self._len_outputs))
        cdef np.ndarray y_last = None
        
        if self.opts["extrapolation_order"] > 0:
            if self.max_output_derivative_order > 0 and not self.opts["force_finite_difference_outputs"]:
                for model in self.models_dict.keys():
                    #yd.extend(model.get_output_derivatives(self.models_dict[model]["local_output"], 1))
                    inext = i + self.models_dict[model]["local_output_len"]
                    status = (<FMUModelCS2>model)._get_output_derivatives(self.models_dict[model]["local_output_vref_array"], ydtmp, self.models_dict[model]["local_output_vref_ones"])
                    if status != 0: raise fmi.FMUException("Failed to get the output derivatives.")
                    yd[i:inext] = ydtmp[:inext-i]
                    i = inext
                
                return self.correct_output_derivative(yd.reshape(-1,1))
                #return yd.reshape(-1,1)
                
            else:
                        
                if self.opts["experimental_output_derivative"]:
                    JM_FMUS = True
                    for model in self.models_dict.keys():
                        if model.get_generation_tool() != "JModelica.org":
                            JM_FMUS = False
                            break
                    if JM_FMUS:
                        C = self.compute_global_C()
                        D = self.compute_global_D()
                        u,ud,udd = self.get_last_us()
                        xd = self._get_derivatives()
                        if ud is not None:
                            if udd is not None:
                                return C.dot(xd)+D.dot(ud+self.get_current_step_size()*udd)
                            else:
                                return C.dot(xd)+D.dot(ud)
                        else: #First step
                            return splin.spsolve((self._ident_matrix-D.dot(self.L)),C.dot(xd)).reshape((-1,1))

                y_last = self.get_last_y()
                if y_last is not None:
                    return (y_cur - y_last)/self.get_current_step_size()
                else:
                    return None
        else:
            return None
            
    cpdef get_connection_second_derivatives(self, np.ndarray yd_cur):

        cdef int i = 0, inext = 0, status = 0
        cdef np.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c']  ydd    = np.empty((self._len_outputs))
        cdef np.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c']  yddtmp = np.empty((self._len_outputs))
        cdef np.ndarray yd_last = None
        
        if self.opts["extrapolation_order"] > 1:
            if self.max_output_derivative_order > 1 and not self.opts["force_finite_difference_outputs"]:
                for model in self.models_dict.keys():
                    inext = i + self.models_dict[model]["local_output_len"]
                    status = (<FMUModelCS2>model)._get_output_derivatives(self.models_dict[model]["local_output_vref_array"], yddtmp, self.models_dict[model]["local_output_vref_twos"])
                    if status != 0: raise fmi.FMUException("Failed to get the output derivatives of second order.")
                    ydd[i:inext] = yddtmp[:inext-i]
                    i = inext
                
                return self.correct_output_second_derivative(ydd.reshape(-1,1))
                
            else:
                if self.opts["experimental_output_derivative"]:
                    JM_FMUS = True
                    for model in self.models_dict.keys():
                        if model.get_generation_tool() != "JModelica.org":
                            JM_FMUS = False
                            break
                    if JM_FMUS:
                        A = self.compute_global_A()
                        B = self.compute_global_B()
                        C = self.compute_global_C()
                        D = self.compute_global_D()
                        u,ud,udd = self.get_last_us()
                        xd = self._get_derivatives()
                        if ud is not None and udd is not None:
                            return C.dot(A.dot(xd))+C.dot(B.dot(ud+self.get_current_step_size()*udd))+D.dot(udd)
                        else: #First step
                            return splin.spsolve((self._ident_matrix-D.dot(self.L)),C.dot(A.dot(xd)+B.dot(self.L.dot(yd_cur)))).reshape((-1,1))
                
                yd_last = self.get_last_yd()
                if yd_last is not None:
                    return (yd_cur - yd_last)/self.get_current_step_size()
                else:
                    return None
        else:
            return None
            
    cpdef set_connection_inputs(self, np.ndarray u, np.ndarray ud=None, np.ndarray udd=None):
        cdef int i = 0, inext, status
        
        u = u.ravel()
        for model in self.models:
            i = self.models_dict[model]["global_index_inputs"] #MIGHT BE WRONG
            inext = i + self.models_dict[model]["local_input_len"]
            #model.set(self.models_dict[model]["local_input"], u[i:inext])
            (<FMUModelCS2>model).set_real(self.models_dict[model]["local_input_vref_array"], u[i:inext])
            
            if ud is not None: #Set the input derivatives
                ud = ud.ravel()
                #model.set_input_derivatives(self.models_dict[model]["local_input"], ud[i:inext], 1)
                status = (<FMUModelCS2>model)._set_input_derivatives(self.models_dict[model]["local_input_vref_array"], ud[i:inext], self.models_dict[model]["local_input_vref_ones"])
                if status != 0: raise fmi.FMUException("Failed to set the first order input derivatives.")
                
            if udd is not None: #Set the input derivatives
                udd = udd.ravel()
                status = (<FMUModelCS2>model)._set_input_derivatives(self.models_dict[model]["local_input_vref_array"], udd[i:inext], self.models_dict[model]["local_input_vref_twos"])
                if status != 0: raise fmi.FMUException("Failed to set the second order input derivatives.")
            
            i = inext
    
    cpdef set_connection_inputs_discrete(self, np.ndarray u):
        cdef int i = 0, inext, status
        
        u = u.ravel()
        for model in self.models:
            i = self.models_dict[model]["global_index_inputs_discrete"] #MIGHT BE WRONG
            inext = i + self.models_dict[model]["local_input_discrete_len"]
            model.set(self.models_dict[model]["local_input_discrete"], u[i:inext])
            
    cpdef set_specific_connection_inputs(self, model, np.ndarray mask, np.ndarray u):
        cdef int i = self.models_dict[model]["global_index_inputs"]
        cdef int inext = i + self.models_dict[model]["local_input_len"]
        cdef np.ndarray usliced = u.ravel()[i:inext]
        (<FMUModelCS2>model).set_real(self.models_dict[model]["local_input_vref_array"][mask], usliced[mask])
    
    cpdef set_specific_connection_inputs_discrete(self, model, np.ndarray mask, np.ndarray u):
        cdef int i = self.models_dict[model]["global_index_inputs_discrete"]
        cdef int inext = i + self.models_dict[model]["local_input_discrete_len"]
        cdef np.ndarray usliced = u.ravel()[i:inext]
        model.set(np.array(self.models_dict[model]["local_input_discrete"])[mask], usliced[mask])
    
    cpdef correct_output_second_derivative(self, np.ndarray ydd):
        if self.linear_correction and self.algebraic_loops and self.support_directional_derivatives:
            raise NotImplementedError
            """
            D = self.compute_global_D()
            DL = D.dot(self.L)
            
            if self.opts["extrapolation_order"] > 0:
                uold, udold = self.get_last_us()
                uhat = udold if udold is not None else np.zeros(np.array(yd).shape)
                
                z = yd - D.dot(uhat)
            
            yd = splin.spsolve((self._ident_matrix-DL),z).reshape((-1,1))
            """
        return ydd

    cpdef correct_output_derivative(self, np.ndarray yd):
        if self.linear_correction and self.algebraic_loops and self.support_directional_derivatives:
            D = self.compute_global_D()
            DL = D.dot(self.L)
            
            if self.opts["extrapolation_order"] > 0:
                uold, udold, uddold = self.get_last_us()
                uhat = udold if udold is not None else np.zeros(np.array(yd).shape)
                
                z = yd - D.dot(uhat)
            
            yd = splin.spsolve((self._ident_matrix-DL),z).reshape((-1,1))

        return yd
    
    cpdef correct_output(self, np.ndarray y, y_prev=None):
        if self.algebraic_loops and self.opts["experimental_output_solve"]:
            JM_FMUS = True
            for model in self.models_dict.keys():
                if model.get_generation_tool() != "JModelica.org":
                    JM_FMUS = False
                    break
            if JM_FMUS:
                if USE_ROOT:
                    res = sopt.root(init_f, y, args=(self))
                else:
                    res = sopt.fsolve(init_f, y, args=(self))
                if not res["success"]:
                    print(res)
                    raise fmi.FMUException("Failed to converge the output system.")
                return res["x"].reshape(-1,1)
                
        if self.linear_correction and self.algebraic_loops and y_prev is not None:# and self.support_directional_derivatives:
            D = self.compute_global_D()
            DL = D.dot(self.L)
            
            if self.opts["extrapolation_order"] > 0:
                uold, udold, uddold = self.get_last_us()
                uhat = uold + (self.get_current_step_size()*udold if udold is not None else 0.0)
                
                z = y - D.dot(uhat)
                #z = y - matvec(D,uhat.ravel())
            else:
                
                z = y - DL.dot(y_prev)
                #z = y - matvec(DL, y_prev.ravel())
            
            y = splin.spsolve((self._ident_matrix-DL),z).reshape((-1,1))
            #y = splin.lsqr((sp.eye(*DL.shape)-DL),z)[0].reshape((-1,1))

        elif self.algebraic_loops and self.support_directional_derivatives:
            pass
            
        return y
    
    cpdef modify_input(self, np.ndarray y, np.ndarray yd, np.ndarray ydd):
        cdef double h = self.get_current_step_size()
        
        u    = self.L.dot(y)
        ud   = self.L.dot(yd)  if yd  is not None else None
        udd  = self.L.dot(ydd) if ydd is not None else None
        
        if self.opts["extrapolation_order"] > 0 and self.opts["smooth_coupling"]:
            uold, udold, uddold = self.get_last_us()
            uhat = uold + (h*udold if udold is not None else 0.0)
        
            udhat = (u-uhat)/h+ud if ud is not None else ud
            
            u = uhat
            ud = udhat

        return u, ud, udd
    
    cpdef modify_input_discrete(self, np.ndarray y):
        return self.L_discrete.dot(y)
    
    cpdef exchange_connection_data(self):
        #u = Ly
        cdef np.ndarray y = self.get_connection_outputs()
        cdef np.ndarray y_discrete
        cdef np.ndarray u
        cdef np.ndarray u_discrete
        
        y   = self.correct_output(y, self.y_prev)
        yd  = self.get_connection_derivatives(y)
        ydd = self.get_connection_second_derivatives(yd)
        y_discrete = self.get_connection_outputs_discrete()
        
        self.y_prev = y.copy()
        self.yd_prev = yd.copy() if yd is not None else None
        
        u, ud, udd = self.modify_input(y, yd, ydd)
        u_discrete = self.modify_input_discrete(y_discrete)
        
        self.set_connection_inputs(u, ud=ud, udd=udd)
        self.set_connection_inputs_discrete(u_discrete)
        
        self.set_last_us(u, ud, udd)
        
        return y, yd, u
    
    def initialize(self, double start_time, double final_time, object opts):
        self.set_input(start_time)
        
        if opts["block_initialization"]:
            
            order, blocks, compressed = self.compute_evaluation_order(opts["block_initialization_type"], order=opts["experimental_block_initialization_order"])
            model_in_init_mode = {model:False for model in self.models}
            
            #Global outputs vector
            y = np.zeros((self._len_outputs))
            y_discrete = np.zeros((self._len_outputs_discrete))
            
            for block in blocks:
                if len(block["inputs"]) == 0: #No inputs in this block, only outputs
                    for model in block["outputs"].keys():
                        #If the model has not entered initialization mode, enter
                        if model_in_init_mode[model] is False:
                            enter_initialization_mode([model], start_time, final_time, opts, self.elapsed_time_init)
                            model_in_init_mode[model] = True
                        
                        #Get the outputs 
                        time_start = timer()
                        if len(y) > 0:
                            self.get_specific_connection_outputs(model, block["outputs_mask"][model], y)
                        if len(y_discrete) > 0:
                            self.get_specific_connection_outputs_discrete(model, block["outputs_discrete_mask"][model], y_discrete)
                        self.elapsed_time_init[model] += timer() - time_start
                        
                elif len(block["outputs"]) == 0: #No outputs in this block
                    
                    #Compute current global input vector
                    if len(y) > 0:
                        u = self.L.dot(y.reshape(-1,1))
                        for model in block["inputs"].keys(): #Set the inputs
                            self.set_specific_connection_inputs(model, block["inputs_mask"][model], u)
                    if len(y_discrete) > 0:
                        u_discrete = self.L_discrete.dot(y_discrete.reshape(-1,1))
                        for model in block["inputs"].keys(): #Set the inputs
                            self.set_specific_connection_inputs_discrete(model, block["inputs_discrete_mask"][model], u_discrete)
                        
                else: #Both (algebraic loop)
                    if self._len_outputs_discrete > 0:
                        raise fmi.FMUException("Block initialization is currently not supported when discrete connections (inputs or outputs) results in an algebraic loop. Please set 'block_initialization' to False.")
                    
                    #Assert models has entered initialization mode
                    for model in list(block["inputs"].keys())+list(block["outputs"].keys()): #Possible only need outputs?
                        if model_in_init_mode[model] is False:
                            enter_initialization_mode([model], start_time, final_time, opts, self.elapsed_time_init)
                            model_in_init_mode[model] = True
                            
                    #Get start values
                    for model in block["outputs"].keys():
                        self.get_specific_connection_outputs(model, block["outputs_mask"][model], y)
                    
                    if USE_ROOT:
                        res = sopt.root(init_f_block, y[block["global_outputs_mask"]], args=(self,block))
                    else:
                        res = sopt.fsolve(init_f_block, y[block["global_outputs_mask"]], args=(self,block))
                    if not res["success"]:
                        print(res)
                        raise fmi.FMUException("Failed to converge the initialization system.")
                    
                    y[block["global_outputs_mask"]] = res["x"]
                    u = self.L.dot(y.reshape(-1,1))
                    
                    for model in block["inputs"].keys():
                        #Set the inputs
                        self.set_specific_connection_inputs(model, block["inputs_mask"][model], u)
                    
            #Assert that all models has entered initialization mode        
            for model in self.models:
                if model_in_init_mode[model] is False:
                    enter_initialization_mode([model], start_time, final_time, opts, self.elapsed_time_init)
                    model_in_init_mode[model] = True
        else:
            enter_initialization_mode(self.models, start_time, final_time, opts, self.elapsed_time_init)
            
            if self.algebraic_loops: #If there is an algebraic loop, solve the resulting system
                if self.support_directional_derivatives:
                    if USE_ROOT:
                        res = sopt.root(init_f, self.get_connection_outputs(), args=(self), jac=init_jac)
                    else:
                        res = sopt.fsolve(init_f, self.get_connection_outputs(), args=(self), jac=init_jac)
                else:
                    if USE_ROOT:
                        res = sopt.root(init_f, self.get_connection_outputs(), args=(self))
                    else:
                        res = sopt.fsolve(init_f, self.get_connection_outputs(), args=(self))
                if not res["success"]:
                    print(res)
                    raise fmi.FMUException("Failed to converge the initialization system.")
                
                y_discrete = self.get_connection_outputs_discrete()
                
                u = self.L.dot(res["x"].reshape(-1,1))
                u_discrete = self.L_discrete.dot(y_discrete)
                
                self.set_connection_inputs(u)
                self.set_connection_inputs_discrete(u_discrete)
            else:
                y = self.get_connection_outputs()
                y_discrete = self.get_connection_outputs_discrete()
                u = self.L.dot(y)
                u_discrete = self.L_discrete.dot(y_discrete)
                self.set_connection_inputs(u)
                self.set_connection_inputs_discrete(u_discrete)
        
        exit_initialization_mode(self.models, self.elapsed_time_init)
        
        #Store the outputs
        self.y_prev = self.get_connection_outputs().copy()
        self.set_last_y(self.y_prev)
        
    def initialize_result_objects(self, opts):
        i = 0
        for model in self.models_dict.keys():
            result_object = get_result_handler(model, opts)
            
            if not isinstance(opts["result_file_name"], dict):
                raise fmi.FMUException("The result file names needs to be stored in a dict with the individual models as key.")
                
            from pyfmi.fmi_algorithm_drivers import FMICSAlgOptions
            local_opts = FMICSAlgOptions()
            
            if opts["result_handling"] == "file":
                prefix = "txt"
            elif opts["result_handling"] == "csv":
                prefix = "csv"
            else:
                prefix = "mat"
            
            try:
                if opts["result_file_name"][model] is None:
                    local_opts["result_file_name"] = model.get_identifier()+'_'+str(i)+'_result.'+prefix
                else:
                    local_opts["result_file_name"] = opts["result_file_name"][model]
            except KeyError:
                raise fmi.FMUException("Incorrect definition of the result file name option. No result file name found for model %s"%model.get_identifier())
                
            local_opts["filter"] = opts["filter"][model]
            
            result_object.set_options(local_opts)
            result_object.simulation_start()
            result_object.initialize_complete()
            
            i = i + 1
            
            self.models_dict[model]["result"] = result_object
            
    def jacobi_algorithm(self, double start_time, double final_time, object opts):
        cdef double step_size = opts["step_size"]
        cdef int calling_setting = SERIAL if opts["execution"] != "parallel" else PARALLEL
        cdef double tcur, step_size_old
        cdef double error
        cdef dict states = None
        cdef np.ndarray ycur, ucur
        
        if self.error_controlled:
            tcur = start_time
            
            y_old = self.get_connection_outputs()
            
            while tcur < final_time:
                #Store FMU states
                states = get_fmu_states(self.models, states)
                
                #Set input
                self.set_input(tcur)
                #Take a full step
                perform_do_step(self.models, self.elapsed_time, self.fmu_adresses, tcur, step_size, False, calling_setting)
                
                y_full = self.correct_output(self.get_connection_outputs(), y_old)
                
                #Restore FMU states
                set_fmu_states(states)
                
                #Set input
                self.set_input(tcur)
                #Take a half step
                perform_do_step(self.models, self.elapsed_time, self.fmu_adresses, tcur, step_size/2.0, False, calling_setting)
                
                #Exchange and set new inputs
                self.set_current_step_size(step_size/2.0)
                self.set_input(tcur + step_size/2.0)
                ycur, ydcur, ucur = self.exchange_connection_data()
                self.set_last_y(ycur)
                self.set_last_yd(ydcur)
                
                #Take another half step
                perform_do_step(self.models, self.elapsed_time, self.fmu_adresses, tcur+step_size/2.0, step_size/2.0, False, calling_setting)
                
                self.exchange_connection_data()
                self.set_last_y(ycur)
                self.set_last_yd(ydcur)
                y_half = self.y_prev.copy()#self.correct_output(self.get_connection_outputs(), y_old)
                
                step_size_old = step_size
                
                error = self.estimate_error(y_half, y_full)
                step_size = self.adapt_stepsize(step_size, error)
                if opts["maxh"] > 0: #Adjust for the maximum step-size (if set)
                    step_size = min(step_size, opts["maxh"])
                
                if opts["logging"]:
                    self._error_data["time"].append(tcur)
                    self._error_data["error"].append(error)
                    self._error_data["step-size"].append(step_size)
                
                if error < 1.1: #Step accepted
                    #Update the time
                    tcur += step_size_old
                    #Store data
                    time_start = timer()
                    #store_communication_point(self.models_dict)
                    self.report_solution(tcur)
                    self.elapsed_time["result_handling"] += timer() - time_start
                
                    self.statistics["nsteps"] += 1
                    y_old = y_half.copy()
                    
                    if tcur+step_size > final_time: #Make sure that we don't step over the final time
                        step_size = final_time - tcur
                else:
                    #Restore FMU states
                    set_fmu_states(states)
                    self.y_prev = y_old
                    self.set_last_y(y_old)
                    self.statistics["nreject"] += 1
                    
                    if opts["logging"]:
                        self._error_data["rejected"].append((tcur, error))
            
            if states:
                free_fmu_states(states)
        else:
            self.set_current_step_size(step_size)
            for tcur in np.arange(start_time, final_time, step_size):
                if tcur + step_size > final_time:
                    step_size = final_time - tcur
                    self.set_current_step_size(step_size)
                    
                perform_do_step(self.models, self.elapsed_time, self.fmu_adresses, tcur, step_size, True, calling_setting)
                
                if self.opts["store_step_before_update"]:
                    time_start = timer()
                    #store_communication_point(self.models_dict)
                    self.report_solution(tcur)
                    self.elapsed_time["result_handling"] += timer() - time_start
                    
                #Set external input
                self.set_input(tcur + step_size)
                ycur, ydcur, ucur = self.exchange_connection_data()
                self.set_last_y(ycur)
                self.set_last_yd(ydcur)
                
                time_start = timer()
                #store_communication_point(self.models_dict)
                self.report_solution(tcur)
                self.elapsed_time["result_handling"] += timer() - time_start
                
                self.statistics["nsteps"] += 1
                
                #Logging
                if opts["logging"]:
                    D = self.compute_global_D()
                    DL = D.dot(self.L)
                    import numpy.linalg
                    import scipy.linalg as slin
                    print("At time: %E , rho(DL)=%s"%(tcur + step_size, str(numpy.linalg.eig(DL.todense())[0])))
                    C = self.compute_global_C()
                    if C is not None:
                        C = C.todense()
                        _ident_matrix = np.eye(*DL.shape)
                        LIDLC = self.L.dot(lin.solve(_ident_matrix-DL,C))
                        print("           , rho(L(I-DL)^(-1)C)=%s"%(str(numpy.linalg.eig(LIDLC)[0])))
                    A = self.compute_global_A()
                    B = self.compute_global_B()
                    if C is not None and A is not None and B is not None:
                        A = A.todense(); B = B.todense()
                        eAH = slin.expm(A*step_size)
                        K1  = lin.solve(_ident_matrix-DL,C)
                        K2  = lin.solve(A,(eAH-np.eye(*eAH.shape)).dot(B.dot(self.L.todense())))
                        R1  = np.hstack((eAH, K1))
                        R2  = np.hstack((K2.dot(eAH), K2.dot(K1)))
                        G   = np.vstack((R1,R2))
                        G1  = K2.dot(K1)
                        print("           , rho(G)=%s"%(str(numpy.linalg.eig(G1)[0])))
                    
    
    def specify_external_input(self, input):
        input_names = input[0]
        if isinstance(input_names,tuple):
            input_names = [input_names]
            
        if hasattr(input[1],"__call__"):
            input_traj=(input_names,
                    TrajectoryUserFunction(input[1]))
        else:
            input_traj=(input_names,
                    TrajectoryLinearInterpolation(input[1][:,0],
                                                  input[1][:,1:]))
        self.input_traj = input_traj
        
    cpdef set_input(self, double time):
        if self.input_traj is not None:
            u = self.input_traj[1].eval(np.array([time]))[0,:]
            
            for i,m in enumerate(self.input_traj[0]):
                m[0].set(m[1],u[i])
    
    cdef double estimate_error(self, np.ndarray y_half, np.ndarray y_full, int order=0):
        cdef np.ndarray err = np.abs((y_half - y_full)/(1-2**(order+1)))
        
        return np.sqrt(1.0/len(y_half)*sum(err/(self.atol+self.rtol*abs(y_half))))
    
    cdef double adapt_stepsize(self, double step_size, double error, int order=0):
        """
        Adjust the stepsize depending on the error.
        """
        #Get the extrapolation order
        cdef double one_over_p = 1.0/(order+2.0)
        cdef double alpha = 0.9
        cdef double fac1 = 6.0
        cdef double fac2 = 0.2
        cdef double _tmp
        
        if error == 0.0:
            return step_size*fac1
        else:
            _tmp = alpha*(1.0/error)**one_over_p
            return step_size*min(fac1, max(fac2, _tmp))
            
    cdef reset_statistics(self):
        for key in self.elapsed_time: 
            self.elapsed_time[key] = 0.0
        for key in self.elapsed_time_init: 
            self.elapsed_time_init[key] = 0.0
        for key in self.statistics.keys():
            self.statistics[key] = 0
    
    def simulate_options(self):
        opts = MasterAlgOptions(self)

        return opts
        
    def simulate_profile(self, double start_time=0.0, double final_time=1.0, input=None, options={}):
        import pstats, cProfile
        res = None
        cProfile.runctx("res = self.simulate(start_time, final_time, input, options)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("cumulative").print_stats(30)
        
        return self.get_results(options)
    
    def simulate(self, double start_time=0.0, double final_time=1.0, input=None, options={}):
        """
        Simulates the system.
        """
        self.reset_statistics()
        
        if isinstance(options, dict) and not \
            isinstance(options, MasterAlgOptions):
            # user has passed dict with options or empty dict = default
            options = MasterAlgOptions(self, options)
        elif isinstance(options, MasterAlgOptions):
            pass
        else:
            raise InvalidAlgorithmOptionException(options)
        
        #if options == "Default":
        #   options = self.simulate_options()
        if options["linear_correction"]:
            if self.support_directional_derivatives != 1 and not options["experimental_finite_difference_D"]:
                warnings.warn("Linear correction only supported if directional derivatives are available.")
                self.linear_correction = 0
            else:
                self.linear_correction = 1
        else:
            self.linear_correction = 0
        if options["error_controlled"]:
            if self.support_storing_fmu_states != 1:
                warnings.warn("Error controlled simulation only supported if storing FMU states are available.")
                self.error_controlled = 0
            else:
                self.error_controlled = 1
                self.atol = options["atol"]
                self.rtol = options["rtol"]
        else:
            self.error_controlled = 0
        if options["extrapolation_order"] > 0:
            if self.support_interpolate_inputs != 1:
                warnings.warn("Extrapolation of inputs only supported if the individual FMUs support interpolation of inputs.")
                options["extrapolation_order"] = 0
        
        if options["num_threads"] and options["execution"] == "parallel":
            pass
            IF WITH_OPENMP: 
                openmp.omp_set_num_threads(options["num_threads"])
        if options["step_size"] <= 0.0:
            raise fmi.FMUException("The step-size must be greater than zero.")
        if options["maxh"] < 0.0:
            raise fmi.FMUException("The maximum step-size must be greater than zero (or equal to zero, i.e inactive).")
        
        self.opts = options #Store the options
            
        if input is not None:
            self.specify_external_input(input)
        
        #Initialize the models
        if options["initialize"]:
            time_start = timer()
            self.initialize(start_time, final_time, options)
            time_stop  = timer()
            print('Elapsed initialization time: ' + str(time_stop-time_start) + ' seconds.')
        
        #Store the inputs
        self.set_last_us(self.L.dot(self.get_connection_outputs()))
        
        #Copy FMU address (used when evaluating in parallel)
        self.copy_fmu_addresses()
        
        self.initialize_result_objects(options)
        store_communication_point(self.models_dict)
        #self.report_solution(tcur)
        
        #Start of simulation, start the clock
        time_start = timer()
        self._time_integration_start = time_start
        
        self.jacobi_algorithm(start_time,final_time, options)

        #End of simulation, stop the clock
        time_stop = timer()
        
        self.print_statistics(options)
        print('')
        print('Simulation interval      : ' + str(start_time) + ' - ' + str(final_time) + ' seconds.')
        print('Elapsed simulation time  : ' + str(time_stop-time_start) + ' seconds.')
        for model in self.models:
            print(' %f seconds spent in %s.'%(self.elapsed_time[model],model.get_name()))
        print(' %f seconds spent saving simulation result.'%(self.elapsed_time["result_handling"]))
        
        #Write the results to file and return
        return self.get_results(options)
    
    def get_results(self, opts):
        """
        For each model, write the result and load the result file
        """
        finalize_result_objects(self.models_dict)
        
        res = {}
        #Load data
        for i,model in enumerate(self.models):
            stored_res = self.models_dict[model]["result"].get_result()
            try:
                file_name = self.models_dict[model]["result"].file_name
            except AttributeError:
                file_name = ""
            res[i] = AssimuloSimResult(model, file_name, None, stored_res, None)
            res[model] = res[i]

        return res
    
    def print_statistics(self, opts):
        print('Master Algorithm options:')
        print(' Algorithm             : Jacobi ' + ("(variable-step)" if self.error_controlled else "(fixed-step)"))
        print('  Execution            : ' + ("Parallel" if self.opts["execution"] == "parallel" else "Serial"))
        print(' Extrapolation Order   : ' + str(opts["extrapolation_order"]) + (" (with smoothing)" if opts["smooth_coupling"] and opts["extrapolation_order"] > 0  else ""))
        if self.error_controlled:
            print(' Tolerances (relative) : ' + str(opts["rtol"]))
            print(' Tolerances (absolute) : ' + str(opts["atol"]))
        else:
            print(' Step-size             : ' + str(opts["step_size"]))
        print(' Algebraic loop        : ' + ("True" if self.algebraic_loops else "False"))
        print('  Linear Correction    : ' + ("True" if self.linear_correction else "False"))
        print('')
        print('Statistics: ')
        print(' Number of global steps        : %d'%self.statistics["nsteps"])
        if self.error_controlled:
            print(' Number of rejected steps      : %d'%self.statistics["nreject"])
    
    def _get_support_directional_derivatives(self):
        if self._support_directional_derivatives == -1:
            self._support_directional_derivatives = 1
            for model in self.models_dict.keys():
                if model.get_capability_flags()["providesDirectionalDerivatives"] is False:
                    self._support_directional_derivatives = 0
                    break 
        return self._support_directional_derivatives

    support_directional_derivatives = property(_get_support_directional_derivatives)
    
    def _get_support_storing_fmu_states(self):
        if self._support_storing_fmu_states == -1:
            self._support_storing_fmu_states = 1
            for model in self.models_dict.keys():
                if model.get_capability_flags()["canGetAndSetFMUstate"] is False:
                    self._support_storing_fmu_states = 0
                    break
        return self._support_storing_fmu_states
    
    support_storing_fmu_states = property(_get_support_storing_fmu_states)
    
    def _get_support_interpolate_inputs(self):
        if self._support_interpolate_inputs == -1:
            self._support_interpolate_inputs = 1
            for model in self.models_dict.keys():
                if model.get_capability_flags()["canInterpolateInputs"] is False:
                    self._support_interpolate_inputs = 0
                    break
        return self._support_interpolate_inputs

    support_interpolate_inputs = property(_get_support_interpolate_inputs)
    
    def _get_max_output_derivative_order(self):
        if self._max_output_derivative_order == -1:
            self._max_output_derivative_order = 10
            for model in self.models_dict.keys():
                self._max_output_derivative_order = min(self._max_output_derivative_order, model.get_capability_flags()["maxOutputDerivativeOrder"])
        return self._max_output_derivative_order
    
    max_output_derivative_order = property(_get_max_output_derivative_order)
    
    def reset(self):
        """
        Reset the coupled models.
        """
        reset_models(self.models)
    
    def visualize_connections(self, vectorized=True, delete=False):
        import subprocess
        import tempfile
        import os
        
        def strip_var(var): return var.replace(".","_").replace("[","_").replace("]","_").replace(",","_")
        def vec_vars(list_vars):
            if not isinstance(list_vars, list): list_vars = [list_vars]
            res_vars = OrderedDict()
            for var in list_vars:
                if fnmatch.fnmatch(var, "*[[]?[]]"):
                    res_vars[var[:-3]] = 1
                elif fnmatch.fnmatch(var, "*[[]?,?[]]"):
                    res_vars[var[:-5]] = 1
                else:
                    res_vars[var] = 1
            return res_vars.keys()
        
        tmp = next(tempfile._get_candidate_names())
        
        with open("%s.dot"%tmp, 'w') as dfile:
            dfile.write("""digraph "callgraph" {
    nodesep=0.5;ranksep=1;
    node [fontname=Arial, height=0, shape=box, style=filled, width=0];
    edge [fontname=Arial];\n""")
            for model in self.models_dict:
                #dfile.write('   subgraph cluster_%s { fontsize="12"; label = "%s"; \n'%(model.get_name().replace(".","_"), model.get_name()))
                dfile.write('   subgraph cluster_%s { fontsize="12"; label = "%s"; \n'%(str(id(model)), model.get_name()))
                
                if vectorized:
                    input_vars = vec_vars(self.models_dict[model]["local_input"])
                else:
                    input_vars = self.models_dict[model]["local_input"]
                
                tmp_str = "<input> Inputs|"
                for input_var in input_vars:
                    tmp_str += "<%s> %s|"%(strip_var(input_var), input_var) 
                dfile.write('       input_%s[label="{%s}" shape=Mrecord];\n'%(str(id(model)), tmp_str[:-1]))
                
                if vectorized:
                    output_vars = vec_vars(self.models_dict[model]["local_output"])
                else:
                    output_vars = self.models_dict[model]["local_output"]
                
                tmp_str = "<output> Outputs|"
                for output_var in output_vars:
                    tmp_str += "<%s> %s|"%(strip_var(output_var), output_var)
                dfile.write('       output_%s[label="{%s}" shape=Mrecord]; }\n'%(str(id(model)), tmp_str[:-1]))
                #dfile.write("   }\n")
                
                for direct_dependence in self.models_dict[model]["direct_dependence"]:
                    if vectorized:
                        dfile.write("   input_%s:%s:e -> output_%s:%s:w[color=red]; \n"%(str(id(model)),strip_var(vec_vars(direct_dependence[0])[0]),str(id(model)),strip_var(vec_vars(direct_dependence[1])[0])))
                    else:
                        dfile.write("   input_%s:%s:e -> output_%s:%s:w[color=red]; \n"%(str(id(model)),strip_var(direct_dependence[0]),str(id(model)),strip_var(direct_dependence[1])))

            for connection in self.connections:
                if vectorized:
                    dfile.write("   output_%s:%s:e -> input_%s:%s:w; \n"%(str(id(connection[0])),strip_var(vec_vars(connection[1])[0]),str(id(connection[2])),strip_var(vec_vars(connection[3])[0])))
                else:
                    dfile.write("   output_%s:%s:e -> input_%s:%s:w; \n"%(str(id(connection[0])),strip_var(connection[1]),str(id(connection[2])),strip_var(connection[3])))
            dfile.write("}\n")
        
        with open("%s.dot"%tmp, 'r') as dfile:
            lines = (line.rstrip() for line in dfile)
            unique_lines = OrderedDict.fromkeys((line for line in lines if line))
        with open("%s.dot"%tmp, 'w') as dfile:
            for line in unique_lines:
                dfile.write(line+"\n")
        
        subprocess.call(["dot -Tps %s.dot -o %s.pdf"%(tmp,tmp)], shell=True)
        #subprocess.Popen([os.path.join(os.getcwd(),"%s.pdf"%tmp)],shell=True)
        os.system("/usr/bin/xdg-open %s.pdf"%tmp)
        
    def define_graph(self):
        edges = []
        graph_info = OrderedDict()
        
        def add_info(primary_key, secondary_key, info):
            try:
                if isinstance(info, list):
                    graph_info[primary_key][secondary_key].extend(info)
                else:
                    graph_info[primary_key][secondary_key] = info
            except KeyError:
                graph_info[primary_key] = {secondary_key: info}
        
        #Define the edges:
        for model in self.models:
            for direct_dependence in self.models_dict[model]["direct_dependence"]:
                add_info(str(id(model))+"_"+direct_dependence[0], "type", 0) #Input
                add_info(str(id(model))+"_"+direct_dependence[1], "type", 1) #Output
                add_info(str(id(model))+"_"+direct_dependence[0], "model", str(id(model)))
                add_info(str(id(model))+"_"+direct_dependence[1], "model", str(id(model)))
                add_info(str(id(model)), "outputs", [str(id(model))+"_"+direct_dependence[1]])
                
                edges.append((str(id(model))+"_"+direct_dependence[0],str(id(model))+"_"+direct_dependence[1]))
        for connection in self.connections:
            add_info(str(id(connection[0]))+"_"+connection[1], "type", 1) #Output
            add_info(str(id(connection[2]))+"_"+connection[3], "type", 0) #Input
            add_info(str(id(connection[0]))+"_"+connection[1], "model", str(id(connection[0]))) #Output
            add_info(str(id(connection[2]))+"_"+connection[3], "model", str(id(connection[2]))) #Input
            add_info(str(id(connection[0])), "outputs", [str(id(connection[0]))+"_"+connection[1]])
            
            edges.append((str(id(connection[0]))+"_"+connection[1],str(id(connection[2]))+"_"+connection[3]))
        
        return Graph(edges, graph_info)
    
    def compute_evaluation_order(self, init_type="greedy", order = None):
        if order is None:
            graph = self.define_graph()
            if init_type == "grouping":
                order = graph.grouped_order(graph.strongly_connected_components())[::-1]
            elif init_type == "simple":
                order = graph.strongly_connected_components()[::-1]
            elif init_type == "greedy":
                order = graph.compute_evaluation_order()[::-1]
            else:
                raise fmi.FMUException("Unknown initialization type. Use either 'greedy', 'simple', 'grouping'")
        
        blocks = []
        for block in order:
            blocks.append({})
            blocks[-1]["inputs"] = {}
            blocks[-1]["outputs"] = {}
            blocks[-1]["has_outputs"] = False
            for variable_compound in block:
                model_id, var = variable_compound.split("_",1)
                model = self.models_id_mapping[model_id]
                if var in self.models_dict[model]["local_input"] or var in self.models_dict[model]["local_input_discrete"]:
                    try:
                        blocks[-1]["inputs"][model].append(var)
                    except KeyError:
                        blocks[-1]["inputs"][model] = [var]
                elif var in self.models_dict[model]["local_output"] or var in self.models_dict[model]["local_output_discrete"]:
                    try:
                        blocks[-1]["outputs"][model].append(var)
                    except KeyError:
                        blocks[-1]["outputs"][model] = [var]
                        blocks[-1]["has_outputs"] = True
                else:
                    raise fmi.FMUException("Something went wrong while creating the blocks.")
        
        for block in blocks:
            block["global_inputs_mask"] = np.array([False]*self._len_inputs)
            block["global_outputs_mask"] = np.array([False]*self._len_outputs)
            block["global_inputs_discrete_mask"] = np.array([False]*self._len_inputs_discrete)
            block["global_outputs_discrete_mask"] = np.array([False]*self._len_outputs_discrete)
            block["inputs_mask"] = {}
            block["outputs_mask"] = {}
            block["inputs_discrete_mask"] = {}
            block["outputs_discrete_mask"] = {}
            for model in block["inputs"].keys():
                input_vref = np.array(self.models_dict[model]["local_input"])
                input_discrete_vref = np.array(self.models_dict[model]["local_input_discrete"])
                mask = np.array([False]*len(input_vref))
                mask_discrete = np.array([False]*len(input_discrete_vref))
                for i,x in enumerate(block["inputs"][model]):
                    if len(np.where(input_vref == x)[0]) > 0:
                        pos = np.where(input_vref == x)[0][0]
                        mask[pos] = True
                        block["global_inputs_mask"][self.models_dict[model]["global_index_inputs"]+pos] = True
                    else:
                        pos = np.where(input_discrete_vref == x)[0][0]
                        mask_discrete[pos] = True
                        block["global_inputs_discrete_mask"][self.models_dict[model]["global_index_inputs_discrete"]+pos] = True
                        
                block["inputs_mask"][model] = mask
                block["inputs_discrete_mask"][model] = mask_discrete

            for model in block["outputs"].keys():
                output_vref = np.array(self.models_dict[model]["local_output"])
                output_discrete_vref = np.array(self.models_dict[model]["local_output_discrete"])
                mask = np.array([False]*len(output_vref))
                mask_discrete = np.array([False]*len(output_discrete_vref))
                for i,x in enumerate(block["outputs"][model]):

                    if len(np.where(output_vref == x)[0]) > 0:
                        pos = np.where(output_vref == x)[0][0]
                        mask[pos] = True
                        block["global_outputs_mask"][self.models_dict[model]["global_index_outputs"]+pos] = True
                    else:
                        pos = np.where(output_discrete_vref == x)[0][0]
                        mask_discrete[pos] = True
                        block["global_outputs_discrete_mask"][self.models_dict[model]["global_index_outputs_discrete"]+pos] = True
                        
                block["outputs_mask"][model] = mask
                block["outputs_discrete_mask"][model] = mask_discrete
        
        #Compress blocks
        compressed_blocks = []
        """
        compressed_blocks = [blocks[0]]
        last_has_inputs = True if len(blocks[0]["inputs"]) > 0 else False
        last_has_outputs = True if len(blocks[0]["outputs"]) > 0 else False
        for block in blocks[1:]:
            has_inputs = True if len(block["inputs"]) > 0 else False
            has_outputs = True if len(block["outputs"]) > 0 else False
            if has_inputs and has_outputs or last_has_inputs and last_has_outputs:
                compressed_blocks.append(block)
                last_has_inputs = has_inputs
                last_has_outputs = has_outputs
            elif last_has_inputs and not last_has_outputs and has_inputs and not has_outputs:
                for key in block["inputs"].keys():
                    try:
                        compressed_blocks[-1]["inputs"][key].append(block["inputs"][key])
                    except KeyError:
                        compressed_blocks[-1]["inputs"][key] = block["inputs"][key]
            elif not last_has_inputs and last_has_outputs and not has_inputs and has_outputs:
                for key in block["outputs"].keys():
                    try:
                        compressed_blocks[-1]["outputs"][key].append(block["outputs"][key])
                    except KeyError:
                        compressed_blocks[-1]["outputs"][key] = block["outputs"][key]
            else:
                compressed_blocks.append(block)
                last_has_inputs = has_inputs
                last_has_outputs = has_outputs
        """
        return order, blocks,compressed_blocks
        
        
