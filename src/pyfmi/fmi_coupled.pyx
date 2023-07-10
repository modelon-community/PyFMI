#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Copyright (C) 2017-2023 Modelon AB
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

import pyfmi.fmi as fmi
from fmi cimport FMUModelME2
cimport fmil_import as FMIL

from pyfmi.fmi_util import cpr_seed, enable_caching, Graph

from collections import OrderedDict
import time
import warnings
import numpy as np
cimport numpy as np
import scipy.optimize as sopt
import scipy

try:
    from numpy.lib import NumpyVersion
    USE_ROOT = NumpyVersion(scipy.version.version) >= "0.11.0"
except ImportError: #Numpy version is < 1.9.0 so assume scipy version is the same
    USE_ROOT = False

def init_f_block(u, coupled, block):
    
    i = 0
    for model in block["inputs"].keys(): #Set the inputs
        for var in block["inputs"][model]:
            value = coupled.models_dict[model]["couplings"][var][0].get(coupled.models_dict[model]["couplings"][var][1])
            
            model.set(var, u[i])
            i = i + 1
    
    u_new = []
    for model in block["inputs"].keys(): #Get the computed inputs
        for var in block["inputs"][model]:
            value = coupled.models_dict[model]["couplings"][var][0].get(coupled.models_dict[model]["couplings"][var][1])
            u_new.append(value)
    
    return u - np.array(u_new).ravel()
    
cdef class CoupledModelBase:
    cdef public dict cache
    cdef public object _result_file
    
    def __init__(self):
        self.cache = {}
        self._result_file = None
    
    def _default_options(self, module, algorithm):
        """
        Help method. Gets the options class for the algorithm specified in
        'algorithm'.
        """
        module = __import__(module, globals(), locals(), [algorithm], 0)
        algorithm = getattr(module, algorithm)

        return algorithm.get_default_options()
    
    def _exec_algorithm(self, module, algorithm, options):
        """
        Helper function which performs all steps of an algorithm run which are
        common to all initialize and optimize algorithms.

        Raises::

            Exception if algorithm is not a subclass of
            common.algorithm_drivers.AlgorithmBase.
        """
        base_path = 'common.algorithm_drivers'
        algdrive = __import__(base_path, globals(), locals(), [], -1)
        AlgorithmBase = getattr(getattr(algdrive,"algorithm_drivers"), 'AlgorithmBase')

        if isinstance(algorithm, basestring):
            module = __import__(module, globals(), locals(), [algorithm], -1)
            algorithm = getattr(module, algorithm)

        if not issubclass(algorithm, AlgorithmBase):
            raise Exception(str(algorithm)+
            " must be a subclass of common.algorithm_drivers.AlgorithmBase")

        # initialize algorithm
        alg = algorithm(self, options)
        # solve optimization problem/initialize
        alg.solve()
        # get and return result
        return alg.get_result()
    
    def _exec_simulate_algorithm(self,
                                 start_time,
                                 final_time,
                                 input,
                                 module,
                                 algorithm,
                                 options):
        """
        Helper function which performs all steps of an algorithm run which are
        common to all simulate algorithms.

        Raises::

            Exception if algorithm is not a subclass of
            common.algorithm_drivers.AlgorithmBase.
        """
        base_path = 'common.algorithm_drivers'
        algdrive = __import__(base_path, globals(), locals(), [], 1)
        AlgorithmBase = getattr(getattr(algdrive,"algorithm_drivers"), 'AlgorithmBase')

        if isinstance(algorithm, basestring):
            module = __import__(module, globals(), locals(), [algorithm], 0)
            algorithm = getattr(module, algorithm)

        if not issubclass(algorithm, AlgorithmBase):
            raise Exception(str(algorithm)+
            " must be a subclass of common.algorithm_drivers.AlgorithmBase")
        
        #open log file
        for model in self.models:
            model._open_log_file()
        
        try:
            # initialize algorithm
            alg = algorithm(start_time, final_time, input, self, options)
            # simulate
            alg.solve()
        except Exception:
            #close log file
            for model in self.models:
                model._close_log_file()
            raise #Reraise the exception
        
        #close log file
        for model in self.models:
            model._close_log_file()
        
        # get and return result
        return alg.get_result()
        
    def get(self, variable_name):
        """
        Returns the value(s) of the specified variable(s). The method both
        accept a single variable and a list of variables.

        Parameters::

            variable_name --
                The name of the variable(s) as string/list.

        Returns::

            The value(s).

        Example::

            # Returns the variable d
            Model.get('damper.d')
            # Returns a list of the variables
            Model.get(['damper.d','gear.a'])
        """
        if isinstance(variable_name, basestring):
            return self._get(variable_name) #Scalar case
        else:
            ret = []
            for i in xrange(len(variable_name)): #A list of variables
                ret += [self._get(variable_name[i])]
            return ret
    
    def set(self, variable_name, value):
        """
        Sets the given value(s) to the specified variable name(s) into the
        model. The method both accept a single variable and a list of variables.

        Parameters::

            variable_name --
                The name of the variable(s) as string/list.

            value --
                The value(s) to set.

        Example::

            Model.set('damper.d', 1.1)
            Model.set(['damper.d','gear.a'], [1.1, 10])
        """
        if isinstance(variable_name, basestring):
            self._set(variable_name, value) #Scalar case
        else:
            for i in xrange(len(variable_name)): #A list of variables
                self._set(variable_name[i], value[i])

cdef class CoupledFMUModelBase(CoupledModelBase):
    cdef public list connections, models
    cdef public object __t, __tolerance, _has_entered_init_mode
    cdef public int _len_inputs, _len_outputs, _len_states, _len_events
    cdef public object models_dict, models_id_mapping
    cdef public dict index,names
    
    def __init__(self, models, connections):
        #Call super
        CoupledModelBase.__init__(self)
        
        self.connections = connections
        self.names       = {model[0]:i for i,model in enumerate(models)}
        self.index       = {i:model[0] for i,model in enumerate(models)}
        self.models      = [model[1] for model in models]
        
        if len(self.names) != len(self.models):
            raise fmi.FMUException("The names of the provided models must be unique.")
        
        self.models_dict = OrderedDict((model,{"model": model,
                                               "local_input": [], "local_input_vref": [], "local_input_len": 0,
                                               "local_state": [], "local_state_vref": [], "local_state_len": 0,
                                               "local_event_len": 0,
                                               "local_derivative": [], "local_derivative_vref": [],
                                               "local_output": [], "local_output_vref": [], "local_output_len": 0,
                                               "direct_dependence": [],
                                               "couplings": {}}) for model in self.models)
        self.models_id_mapping = {str(id(model)): model for model in self.models}
        
        self._len_inputs  = 0
        self._len_outputs = 0
        self._len_states  = 0
        self._len_events  = 0
        
        #Setup and verify the connections between the models
        self.connection_setup()
        
        #Default values
        self.__t = None
        self.__tolerance = None
        
        self._has_entered_init_mode = False
    
    def connection_setup(self):
        for connection in self.connections:
            if len(connection) != 4:
                raise fmi.FMUException('The connections between the models must follow the syntax: '\
                                        '(model_source,"beta",model_destination,"y").')
            self.models_dict[connection[0]]["local_output"].append(connection[1])
            self.models_dict[connection[0]]["local_output_vref"].append(connection[0].get_variable_valueref(connection[1]))
            self.models_dict[connection[2]]["local_input"].append(connection[3])
            self.models_dict[connection[2]]["local_input_vref"].append(connection[2].get_variable_valueref(connection[3]))
            
            #Direct mapping upwards
            self.models_dict[connection[2]]["couplings"][connection[3]] = [connection[0], connection[1]]
        
        self.verify_connection_variables()
        
        for model in self.models_dict.keys():
            #self.models_dict[model]["local_input"]  = model.get_input_list().keys()
            #self.models_dict[model]["local_output"] = model.get_output_list().keys()
            self.models_dict[model]["local_state"]  = model.get_states_list().keys()
            
            self.models_dict[model]["local_input_len"]  = len(self.models_dict[model]["local_input"])
            self.models_dict[model]["local_output_len"] = len(self.models_dict[model]["local_output"])
            self.models_dict[model]["local_state_len"]  = len(self.models_dict[model]["local_state"])
            self.models_dict[model]["local_event_len"]  = model.get_ode_sizes()[1]

            self._len_inputs  += self.models_dict[model]["local_input_len"]
            self._len_outputs += self.models_dict[model]["local_output_len"]
            self._len_states  += self.models_dict[model]["local_state_len"]
            self._len_events  += self.models_dict[model]["local_event_len"]
            
        for model in self.models_dict.keys():
            output_state_dep, output_input_dep = model.get_output_dependencies()
            for local_output in self.models_dict[model]["local_output"]:
                for local_input in self.models_dict[model]["local_input"]:
                    if local_input in output_input_dep[local_output]:
                        self.models_dict[model]["direct_dependence"].append((local_input, local_output))                    
    
    def verify_connection_variables(self):
        for model in self.models_dict.keys():
            for output in self.models_dict[model]["local_output"]:
                if model.get_variable_causality(output) != fmi.FMI2_OUTPUT:
                    raise fmi.FMUException("The connection variable " + output + " in model " + model.get_name() + " is not an output. ")
            for input in self.models_dict[model]["local_input"]:
                if model.get_variable_causality(input) != fmi.FMI2_INPUT:
                    raise fmi.FMUException("The connection variable " + input + " in model " + model.get_name() + " is not an input. ")
    
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
                raise Exception("Unknown initialization type. Use either 'greedy', 'simple', 'grouping'")
        
        blocks = []
        for block in order:
            blocks.append({})
            blocks[-1]["inputs"] = {}
            blocks[-1]["outputs"] = {}
            blocks[-1]["has_outputs"] = False
            for variable_compound in block:
                model_id, var = variable_compound.split("_",1)
                model = self.models_id_mapping[model_id]
                if var in self.models_dict[model]["local_input"]:
                    try:
                        blocks[-1]["inputs"][model].append(var)
                    except KeyError:
                        blocks[-1]["inputs"][model] = [var]
                elif var in self.models_dict[model]["local_output"]:
                    try:
                        blocks[-1]["outputs"][model].append(var)
                    except KeyError:
                        blocks[-1]["outputs"][model] = [var]
                        blocks[-1]["has_outputs"] = True
                else:
                    raise fmi.FMUException("Something went wrong while creating the blocks.")
        """
        for block in blocks:
            block["inputs_mask"] = {}
            block["outputs_mask"] = {}
            
            for model in block["inputs"].keys():
                input_vref = np.array(self.models_dict[model]["local_input"])
                mask = np.array([False]*len(input_vref))
                for i,x in enumerate(block["inputs"][model]):
                    pos = np.where(input_vref == x)[0][0]
                    mask[pos] = True
                block["inputs_mask"][model] = mask
        
            for model in block["outputs"].keys():
                output_vref = np.array(self.models_dict[model]["local_output"])
                mask = np.array([False]*len(output_vref))
                for i,x in enumerate(block["outputs"][model]):
                    pos = np.where(output_vref == x)[0][0]
                    mask[pos] = True
                block["outputs_mask"][model] = mask
        """
        return order, blocks
        
    def setup_experiment(self, tolerance_defined=True, tolerance="Default", start_time="Default", stop_time_defined=False, stop_time="Default"):
        """
        Calls the underlying FMU method for creating an experiment.
        
        Parameters::
        
            tolerance_defined --
                Specifies that the model is used together with an external
                algorithm that is error controlled.
                Default: True
                
            tolerance --
                Tolerance used in the simulation.
                Default: The tolerance defined in the model description.
                
            start_time --
                Start time of the simulation.
                Default: The start time defined in the model description.
                
            stop_time_defined --
                Defines if a fixed stop time is defined or not. If this is
                set the simulation cannot go past the defined stop time.
                Default: False
                
            stop_time --
                Stop time of the simulation.
                Default: The stop time defined in the model description.
            
        """
        if tolerance == "Default":
            tolerance = self.get_default_experiment_tolerance()
        if start_time == "Default":
            start_time = self.get_default_experiment_start_time()
        if stop_time == "Default":
            stop_time = self.get_default_experiment_stop_time()
        
        self.__t = start_time
        self.__tolerance = tolerance
        
        for model in self.models:
            model.setup_experiment(tolerance_defined, tolerance, start_time, stop_time_defined, stop_time)
    
    def initialize(self, tolerance_defined=True, tolerance="Default", start_time="Default", stop_time_defined=False, stop_time="Default"):
        """
        Initializes the model and computes initial values for all variables.
        Additionally calls the setup experiment, if not already called.

        Calls the low-level FMI functions: fmi2_import_setup_experiment (optionally)
                                           fmi2EnterInitializationMode,
                                           fmi2ExitInitializationMode
        """
        if self.time == None:
            self.setup_experiment(tolerance_defined, tolerance, start_time, stop_time_defined, stop_time)
        
        self.enter_initialization_mode()
        self.exit_initialization_mode()
        
    def exit_initialization_mode(self):
        """
        Exit initialization mode by calling the low level FMI function
        fmi2ExitInitializationMode.
        
        Note that the method initialize() performs both the enter and 
        exit of initialization mode.
        """
        status = 0
        for model in self.models:
            status = max(status, model.exit_initialization_mode())
            
        return status
        
    def enter_initialization_mode(self):
        """
        Enters initialization mode by calling the low level FMI function
        fmi2EnterInitializationMode.
        
        Note that the method initialize() performs both the enter and 
        exit of initialization mode.
        """
        if self.time == None:
            raise fmi.FMUException("Setup Experiment has to be called prior to the initialization method.")
        
        self._update_coupling_equations()
        
        status = 0
        for model in self.models:
            if not model._has_entered_init_mode:
                status = max(status, model.enter_initialization_mode())
        
        self._has_entered_init_mode = True
        
        return status
    
    def reset(self):
        """
        Resets the FMU back to its original state. Note that the environment 
        has to initialize the FMU again after this function-call.
        """
        for model in self.models:
            model.reset()

        #Default values
        self.__t = None
        self._has_entered_init_mode = False

        #Internal values
        #self._log = []
        
    def terminate(self):
        """
        Calls the FMI function fmi2Terminate() on the FMU.
        After this call, any call to a function changing the state of the FMU will fail.
        """
        for model in self.models:
            model.terminate()
        
    def free_instance(self):
        """
        Calls the FMI function fmi2FreeInstance() on the FMU. Note that this is not
        needed generally.
        """
        for model in self.models:
            model.free_instance()
    
    def get_identifier(self):
        """
        Return the model identifier, name of binary model file and prefix in
        the C-function names of the model.
        """
        return "CoupledME2FMUs"
    
    def get_name(self):
        """
        Return the model name as used in the modeling environment.
        """
        return "CoupledME2FMUs"
    
    def get_model_version(self):
        """
        Returns the version of the FMU.
        """
        return "N/A"
        
    def get_author(self):
        """
        Return the name and organization of the model author.
        """
        return "N/A"

    def get_description(self):
        """
        Return the model description.
        """
        return "Coupled FMU2 ME Models"
        
    def get_copyright(self):
        """
        Return the model copyright.
        """
        return "N/A"
        
    def get_variable_naming_convention(self):
        """
        Return the variable naming convention.
        """
        naming_conv = self.models[0].get_variable_naming_convention()
        for model in self.models:
            if naming_conv != model.get_variable_naming_convention():
                naming_conv = "unknown"
                break
                
        return naming_conv
        
    def get_license(self):
        """
        Return the model license.
        """
        return "N/A"

    def get_generation_tool(self):
        """
        Return the model generation tool.
        """
        return "N/A"
        
    def get_generation_date_and_time(self):
        """
        Return the model generation date and time.
        """
        return "N/A"

    def get_guid(self):
        """
        Return the model GUID.
        """
        return "N/A"
    
    def get_version(self):
        """
        Returns the FMI version of the Model which it was generated according.

        Returns::

            version --
                The version.

        Example::

            model.get_version()
        """
        return self.models[0].get_version() #Only support FMUs from the same version

    def get_model_types_platform(self):
        """
        Returns the set of valid compatible platforms for the Model, extracted
        from the XML.
        """
        return self.models[0].get_model_types_platform()
            
    def get_ode_sizes(self):
        """
        Returns the number of continuous states and the number of event
        indicators.

        Returns::

            nbr_cont --
                The number of continuous states.

            nbr_ind --
                The number of event indicators.

        Example::

            [nbr_cont, nbr_event_ind] = model.get_ode_sizes()
        """
        return self._len_states, self._len_events
        
    def get_default_experiment_start_time(self):
        """
        Returns the default experiment start time as defined by the greatest start
        time of the coupled system.
        """
        start_time = self.models[0].get_default_experiment_start_time()
        for model in self.models:
            start_time = max(start_time, model.get_default_experiment_start_time())
            
        return start_time

    def get_default_experiment_stop_time(self):
        """
        Returns the default experiment stop time as defined by the 
        minimum stop time of the coupled system.
        """
        stop_time = self.models[0].get_default_experiment_stop_time()
        for model in self.models:
            stop_time = min(stop_time, model.get_default_experiment_stop_time())
        
        return stop_time

    def get_default_experiment_tolerance(self):
        """
        Returns the default experiment tolerance as defined by the smallest
        tolerance of all the models in the coupled system.
        """
        tol = self.models[0].get_default_experiment_tolerance()
        for model in self.models:
            tol = min(tol, model.get_default_experiment_tolerance())
        
        return tol
        
    def get_default_experiment_step(self):
        """
        Returns the default experiment step as defined by the minimum
        step of all the models in the coupled system.
        """
        step = self.models[0].get_default_experiment_step()
        for model in self.models:
            step = min(step, model.get_default_experiment_step())
        
        return step
        
    def get_variable_valueref(self, variable_name):
        """
        Extract the ValueReference given a variable name.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The ValueReference for the variable passed as argument.
        """
        cdef FMIL.fmi2_value_reference_t  vr
        
        name_parts = variable_name.split(".")
        try:
            model_name = name_parts[0]
            ind = self.names[model_name]
            model = self.models[ind]
        except Exception:
            raise fmi.FMUException("The variable %s could not be found. Was the name correctly prefixed with the model name?"%variable_name)
        
        vr = model.get_variable_valueref(variable_name[len(model_name)+1:])
        
        return self._get_global_vr(ind, vr)
        
    def get_variable_by_valueref(self, valueref, type = 0):
        """
        Get the name of a variable given a value reference. Note that it
        returns the no-aliased variable.

        Parameters::

            valueref --
                The value reference of the variable.

            type --
                The type of the variables (Real==0, Int==1, Bool==2,
                String==3, Enumeration==4).
                Default: 0 (i.e Real).

        Returns::

            The name of the variable.

        """
        local_vr = self._get_local_vr(valueref)#valueref & 0x00000000FFFFFFFF
        model_ind = self._get_model_index_from_vr(valueref)#valueref >> 32
        
        try:
            model = self.models[model_ind]
        except Exception:
            raise fmi.FMUException("Could not map the value reference to the correct model. Is the value reference correct?")
        
        local_name = model.get_variable_by_valueref(local_vr, type)
        
        return self._get_global_name(model_ind, local_name)
    
    cdef FMIL.fmi2_value_reference_t _get_local_vr(self, valueref):
        return valueref & 0x00000000FFFFFFFF
    
    cdef _get_model_index_from_vr(self, valueref):
        return long(valueref) >> 32
    
    cdef _get_global_name(self, model_ind, name):
        return self.index[model_ind] + "." + name
    
    def _get_global_vr(self, model_ind, FMIL.fmi2_value_reference_t valueref):
        return (model_ind << 32) + valueref
    
    @enable_caching
    def get_model_variables(self, type = None, include_alias = True,
                             causality = None,   variability = None,
                            only_start = False,   only_fixed = False,
                            filter = None, int _as_list = False):
        """
        Extract the names of the variables in a model.

        Parameters::

            type --
                The type of the variables (Real==0, Int==1, Bool==2,
                String==3, Enumeration==4).
                Default: None (i.e all).

            include_alias --
                If alias should be included or not.
                Default: True

            causality --
                The causality of the variables (Parameter==0, 
                Calculated Parameter==1, Input==2, Output==3, Local==4, 
                Independent==5, Unknown==6).
                Default: None (i.e all).

            variability --
                The variability of the variables (Constant==0,
                Fixed==1, Tunable==2, Discrete==3, Continuous==4, Unknown==5).
                Default: None (i.e. all)

            only_start --
                If only variables that has a start value should be
                returned.
                Default: False

            only_fixed --
                If only variables that has a start value that is fixed
                should be returned.
                Default: False

            filter --
                Filter the variables using a unix filename pattern
                matching (filter="*der*"). Can also be a list of filters
                See http://docs.python.org/2/library/fnmatch.html.
                Default None

        Returns::

            Dict with variable name as key and a ScalarVariable class as
            value.
        """
        variable_dict = OrderedDict()
        
        #WARNING FILTER NOT IMPLEMENTED PROPERLY
        
        for i,model in enumerate(self.models):
            local_dict = model.get_model_variables(type, include_alias, 
                        causality, variability, only_start,   only_fixed, filter)
            
            for key in local_dict:
                var = local_dict[key]
                global_name = self._get_global_name(i, key)
                global_vr   = self._get_global_vr(i, var.value_reference)
                
                variable_dict[self._get_global_name(i, key)] = fmi.ScalarVariable2(global_name, global_vr, 
                                            var.type, var.description, var.variability, 
                                            var.causality, var.alias, var.initial)
        
        if _as_list:
            return variable_dict.values()
        else:
            return variable_dict
        
    def _convert_local_scalar_variable(self, model_ind, scalar_variable):
        global_name = self._get_global_name(model_ind, scalar_variable.name)
        global_vr   = self._get_global_vr(model_ind, scalar_variable.value_reference)
        
        return fmi.ScalarVariable2(global_name, global_vr, 
                                            scalar_variable.type, scalar_variable.description, scalar_variable.variability, 
                                            scalar_variable.causality, scalar_variable.alias, scalar_variable.initial)

    cpdef get_variable_max(self, variable_name):
        """
        Returns the maximum value for the given variable.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The maximum value for the variable.
        """
        name_parts = variable_name.split(".")
        try:
            model_name = name_parts[0]
            ind = self.names[model_name]
            model = self.models[ind]
        except Exception:
            raise fmi.FMUException("The variable %s could not be found. Was the name correctly prefixed with the model name?"%variable_name)
        
        return model.get_variable_max(variable_name[len(model_name)+1:])
    
    cpdef np.ndarray get_real(self, valueref):
        """
        Returns the real-values from the valuereference(s).

        Parameters::

            valueref --
                A list of valuereferences.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_real([232])

        Calls the low-level FMI function: fmi2GetReal
        """
        input_valuerefs = np.array(valueref, ndmin=1).ravel()
        output_values   = []
        
        for vref in input_valuerefs:
            local_vr = self._get_local_vr(vref)
            model_ind = self._get_model_index_from_vr(vref)
        
            try:
                model = self.models[model_ind]
            except Exception:
                raise fmi.FMUException("Could not map the value reference to the correct model. Is the value reference correct?")
            
            output_values.append(model.get_real(local_vr))
            
        return np.array(output_values).ravel()
    
    def get_integer(self, valueref):
        """
        Returns the integer-values from the valuereference(s).

        Parameters::

            valueref --
                A list of valuereferences.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_integer([232])

        Calls the low-level FMI function: fmi2GetInteger
        """
        input_valuerefs = np.array(valueref, ndmin=1).ravel()
        output_values   = []
        
        for vref in input_valuerefs:
            local_vr = self._get_local_vr(vref)
            model_ind = self._get_model_index_from_vr(vref)
        
            try:
                model = self.models[model_ind]
            except Exception:
                raise fmi.FMUException("Could not map the value reference to the correct model. Is the value reference correct?")
            
            output_values.append(model.get_integer(local_vr))
            
        return np.array(output_values).ravel()
    
    
    def get_boolean(self, valueref):
        """
        Returns the boolean-values from the valuereference(s).

        Parameters::

            valueref --
                A list of valuereferences.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_boolean([232])

        Calls the low-level FMI function: fmi2GetBoolean
        """
        input_valuerefs = np.array(valueref, ndmin=1).ravel()
        output_values   = []
        
        for vref in input_valuerefs:
            local_vr = self._get_local_vr(vref)
            model_ind = self._get_model_index_from_vr(vref)
        
            try:
                model = self.models[model_ind]
            except Exception:
                raise fmi.FMUException("Could not map the value reference to the correct model. Is the value reference correct?")
            
            output_values.append(model.get_boolean(local_vr))
        
        return np.array(output_values).ravel()
    
    def get_string(self, valueref):
        """
        Returns the string-values from the valuereference(s).

        Parameters::

            valueref --
                A list of valuereferences.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_string([232])

        Calls the low-level FMI function: fmi2GetString
        """
        input_valuerefs = np.array(valueref, ndmin=1).ravel()
        output_values   = []
        
        for vref in input_valuerefs:
            local_vr = self._get_local_vr(vref)
            model_ind = self._get_model_index_from_vr(vref)
        
            try:
                model = self.models[model_ind]
            except Exception:
                raise fmi.FMUException("Could not map the value reference to the correct model. Is the value reference correct?")
            
            output_values.extend(model.get_string(local_vr))
        
        return output_values
    
    cpdef set_real(self, valueref, values):
        """
        Sets the real-values in the FMU as defined by the valuereference(s).

        Parameters::

            valueref --
                A list of valuereferences.

            values --
                Values to be set.

        Example::

            model.set_real([234,235],[2.34,10.4])

        Calls the low-level FMI function: fmi2SetReal
        """
        input_valueref = np.array(valueref, ndmin=1).ravel()
        set_value      = np.array(values, dtype=float, ndmin=1).ravel()

        if input_valueref.size != set_value.size:
            raise fmi.FMUException('The length of valueref and values are inconsistent.')
        
        for i,vref in enumerate(input_valueref):
            local_vr = self._get_local_vr(vref)
            model_ind = self._get_model_index_from_vr(vref)
        
            try:
                model = self.models[model_ind]
            except Exception:
                raise fmi.FMUException("Could not map the value reference to the correct model. Is the value reference correct?")
            
            model.set_real(local_vr, set_value[i])
    
    def set_integer(self, valueref, values):
        """
        Sets the integer-values in the FMU as defined by the valuereference(s).

        Parameters::

            valueref --
                A list of valuereferences.

            values --
                Values to be set.

        Example::

            model.set_integer([234,235],[12,-3])

        Calls the low-level FMI function: fmi2SetInteger
        """
        input_valueref = np.array(valueref, ndmin=1).ravel()
        set_value      = np.array(values, dtype=int,ndmin=1).ravel()

        if input_valueref.size != set_value.size:
            raise fmi.FMUException('The length of valueref and values are inconsistent.')
        
        for i,vref in enumerate(input_valueref):
            local_vr = self._get_local_vr(vref)
            model_ind = self._get_model_index_from_vr(vref)
        
            try:
                model = self.models[model_ind]
            except Exception:
                raise fmi.FMUException("Could not map the value reference to the correct model. Is the value reference correct?")
            
            model.set_integer(local_vr, set_value[i])
            
    def set_boolean(self, valueref, values):
        """
        Sets the boolean-values in the FMU as defined by the valuereference(s).

        Parameters::

            valueref --
                A list of valuereferences.

            values --
                Values to be set.

        Example::

            model.set_boolean([234,235],[True,False])

        Calls the low-level FMI function: fmi2SetBoolean
        """
        input_valueref = np.array(valueref, ndmin=1).ravel()
        set_value      = np.array(values, ndmin=1).ravel()

        if input_valueref.size != set_value.size:
            raise fmi.FMUException('The length of valueref and values are inconsistent.')
        
        for i,vref in enumerate(input_valueref):
            local_vr = self._get_local_vr(vref)
            model_ind = self._get_model_index_from_vr(vref)
        
            try:
                model = self.models[model_ind]
            except Exception:
                raise fmi.FMUException("Could not map the value reference to the correct model. Is the value reference correct?")
            
            model.set_boolean(local_vr, set_value[i])

    def set_string(self, valueref, values):
        """
        Sets the string-values in the FMU as defined by the valuereference(s).

        Parameters::

            valueref --
                A list of valuereferences.

            values --
                Values to be set.

        Example::

            model.set_string([234,235],['text','text'])

        Calls the low-level FMI function: fmi2SetString
        """
        input_valueref = np.array(valueref, ndmin=1).ravel()
        set_value      = np.array(values, ndmin=1).ravel()
        
        for i,vref in enumerate(input_valueref):
            local_vr = self._get_local_vr(vref)
            model_ind = self._get_model_index_from_vr(vref)
        
            try:
                model = self.models[model_ind]
            except Exception:
                raise fmi.FMUException("Could not map the value reference to the correct model. Is the value reference correct?")
            
            model.set_string(local_vr, set_value[i])        

    def _get(self, variable_name):
        """
        Helper method to get, see docstring on get.
        """
        cdef FMIL.fmi2_base_type_enu_t type

        ref  = self.get_variable_valueref(variable_name)
        type = self.get_variable_data_type(variable_name)

        if type == FMIL.fmi2_base_type_real:  #REAL
            return self.get_real([ref])
        elif type == FMIL.fmi2_base_type_int or type == FMIL.fmi2_base_type_enum: #INTEGER
            return self.get_integer([ref])
        elif type == FMIL.fmi2_base_type_str: #STRING
            return self.get_string([ref])
        elif type == FMIL.fmi2_base_type_bool: #BOOLEAN
            return self.get_boolean([ref])
        else:
            raise fmi.FMUException('Type not supported.')
    
    def _set(self, variable_name, value):
        """
        Helper method to set, see docstring on set.
        """
        cdef FMIL.fmi2_base_type_enu_t   type

        ref  = self.get_variable_valueref(variable_name)
        type = self.get_variable_data_type(variable_name)

        if type == FMIL.fmi2_base_type_real:  #REAL
            self.set_real([ref], [value])
        elif type == FMIL.fmi2_base_type_int or type == FMIL.fmi2_base_type_enum: #INTEGER
            self.set_integer([ref], [value])
        elif type == FMIL.fmi2_base_type_str: #STRING
            self.set_string([ref], [value])
        elif type == FMIL.fmi2_base_type_bool: #BOOLEAN
            self.set_boolean([ref], [value])
        else:
            raise fmi.FMUException('Type not supported.')
    
    def get_directional_derivative(self, var_ref, func_ref, v):
        """
        Returns the directional derivatives of the functions with respect
        to the given variables and in the given direction.
        In other words, it returns linear combinations of the partial derivatives
        of the given functions with respect to the selected variables.
        The point of evaluation is the current time-point.

        Parameters::

            var_ref --
                A list of variable references that the partial derivatives
                will be calculated with respect to.

            func_ref --
                A list of function references for which the partial derivatives will be calculated.

            v --
                A seed vector specifying the linear combination of the partial derivatives.

        Returns::

            value --
                A vector with the directional derivatives (linear combination of
                partial derivatives) evaluated in the current time point.


        Example::

            states = model.get_states_list()
            states_references = [s.value_reference for s in states.values()]
            derivatives = model.get_derivatives_list()
            derivatives_references = [d.value_reference for d in derivatives.values()]
            model.get_directional_derivative(states_references, derivatives_references, v)

            This returns Jv, where J is the Jacobian and v the seed vector.

            Also, only a subset of the derivatives and and states can be selected:

            model.get_directional_derivative(var_ref = [0,1], func_ref = [2,3], v = [1,2])

            This returns a vector with two values where:

            values[0] = (df2/dv0) * 1 + (df2/dv1) * 2
            values[1] = (df3/dv0) * 1 + (df3/dv1) * 2

        """
        raise NotImplementedError
        
    cpdef get_derivatives_dependencies(self):
        """
        Retrieve the list of variables that the derivatives are 
        dependent on. Returns two dictionaries, one with the states 
        and one with the inputs.
        """ 
        raise NotImplementedError
    
    cpdef get_output_dependencies(self):
        """
        Retrieve the list of variables that the outputs are 
        dependent on. Returns two dictionaries, one with the states and 
        one with the inputs.
        """
        raise NotImplementedError
        
    cpdef get_variable_min(self, variable_name):
        """
        Returns the minimum value for the given variable.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The minimum value for the variable.
        """
        name_parts = variable_name.split(".")
        try:
            model_name = name_parts[0]
            ind = self.names[model_name]
            model = self.models[ind]
        except Exception:
            raise fmi.FMUException("The variable %s could not be found. Was the name correctly prefixed with the model name?"%variable_name)
        
        return model.get_variable_min(variable_name[len(model_name)+1:])
    
    
    cpdef get_variable_start(self, variable_name):
        """
        Returns the start value for the variable or else raises
        FMUException.

        Parameters::

            variable_name --
                The name of the variable

        Returns::

            The start value.
        """
        name_parts = variable_name.split(".")
        try:
            model_name = name_parts[0]
            ind = self.names[model_name]
            model = self.models[ind]
        except Exception:
            raise fmi.FMUException("The variable %s could not be found. Was the name correctly prefixed with the model name?"%variable_name)
        
        return model.get_variable_start(variable_name[len(model_name)+1:])
        
    cpdef get_variable_unbounded(self, variable_name):
        """
        Returns the unbounded attribute for the variable or else raises
        FMUException.

        Parameters::

            variable_name --
                The name of the variable

        Returns::

            The start value.
        """
        name_parts = variable_name.split(".")
        try:
            model_name = name_parts[0]
            ind = self.names[model_name]
            model = self.models[ind]
        except Exception:
            raise fmi.FMUException("The variable %s could not be found. Was the name correctly prefixed with the model name?"%variable_name)
        
        return model.get_variable_unbounded(variable_name[len(model_name)+1:])
        
    
    cpdef FMIL.fmi2_causality_enu_t get_variable_causality(self, variable_name) except *:
        """
        Get the causality of the variable.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The variability of the variable, PARAMETER(0), CALCULATED_PARAMETER(1), INPUT(2),
            OUTPUT(3), LOCAL(4), INDEPENDENT(5), UNKNOWN(6)
        """
        name_parts = variable_name.split(".")
        try:
            model_name = name_parts[0]
            ind = self.names[model_name]
            model = self.models[ind]
        except Exception:
            raise fmi.FMUException("The variable %s could not be found. Was the name correctly prefixed with the model name?"%variable_name)
        
        return model.get_variable_causality(variable_name[len(model_name)+1:])
    
    cpdef FMIL.fmi2_initial_enu_t get_variable_initial(self, variable_name) except *:
        """
        Get initial of the variable.
        
        Parameters::
        
            variable_name --
                The name of the variable.
                
        Returns::
        
            The initial of the variable: EXACT(0), APPROX(1), 
            CALCULATED(2), UNKNOWN(3)
        """
        name_parts = variable_name.split(".")
        try:
            model_name = name_parts[0]
            ind = self.names[model_name]
            model = self.models[ind]
        except Exception:
            raise fmi.FMUException("The variable %s could not be found. Was the name correctly prefixed with the model name?"%variable_name)
        
        return model.get_variable_initial(variable_name[len(model_name)+1:])
    
    cpdef FMIL.fmi2_variability_enu_t get_variable_variability(self, variable_name) except *:
        """
        Get variability of the variable.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The variability of the variable: CONSTANT(0), FIXED(1),
            TUNABLE(2), DISCRETE(3), CONTINUOUS(4) or UNKNOWN(5)
        """
        name_parts = variable_name.split(".")
        try:
            model_name = name_parts[0]
            ind = self.names[model_name]
            model = self.models[ind]
        except Exception:
            raise fmi.FMUException("The variable %s could not be found. Was the name correctly prefixed with the model name?"%variable_name)
        
        return model.get_variable_variability(variable_name[len(model_name)+1:])

    cpdef get_variable_description(self, variable_name):
        """
        Get the description of a given variable.

        Parameter::

            variable_name --
                The name of the variable

        Returns::

            The description of the variable.
        """
        name_parts = variable_name.split(".")
        try:
            model_name = name_parts[0]
            ind = self.names[model_name]
            model = self.models[ind]
        except Exception:
            raise fmi.FMUException("The variable %s could not be found. Was the name correctly prefixed with the model name?"%variable_name)
        
        return model.get_variable_description(variable_name[len(model_name)+1:])

    
    cpdef FMIL.fmi2_base_type_enu_t get_variable_data_type(self, variable_name) except *:
        """
        Get data type of variable.

        Parameter::

            variable_name --
                The name of the variable.

        Returns::

            The type of the variable.
        """
        name_parts = variable_name.split(".")
        try:
            model_name = name_parts[0]
            ind = self.names[model_name]
            model = self.models[ind]
        except Exception:
            raise fmi.FMUException("The variable %s could not be found. Was the name correctly prefixed with the model name?"%variable_name)
        
        return model.get_variable_data_type(variable_name[len(model_name)+1:])

    
    def get_variable_alias_base(self, variable_name):
        """
        Returns the base variable for the provided variable name.

        Parameters::

            variable_name--
                Name of the variable.

        Returns:

           The base variable.
        """
        name_parts = variable_name.split(".")
        try:
            model_name = name_parts[0]
            ind = self.names[model_name]
            model = self.models[ind]
        except Exception:
            raise fmi.FMUException("The variable %s could not be found. Was the name correctly prefixed with the model name?"%variable_name)
        
        var = model.get_variable_alias_base(variable_name[len(model_name)+1:])
        
        return self._get_global_name(ind, var)
        
    def get_derivatives_list(self):
        """
        Returns a dictionary with the states derivatives.

        Returns::

            An ordered dictionary with the derivative variables.
        """
        der_list = OrderedDict()
        
        for i, model in enumerate(self.models):
            local_ders = model.get_derivatives_list()
            
            for key in local_ders:
                var = self._convert_local_scalar_variable(i, local_ders[key])
                der_list[var.name] = var
                
        return der_list
    
    def get_states_list(self):
        """
        Returns a dictionary with the states.

        Returns::

            An ordered dictionary with the state variables.
        """
        states_list = OrderedDict()
        
        for i, model in enumerate(self.models):
            local_states = model.get_states_list()
            
            for key in local_states:
                var = self._convert_local_scalar_variable(i, local_states[key])
                states_list[var.name] = var
                
        return states_list
        
    def get_input_list(self):
        """
        Returns a dictionary with input variables

        Returns::

            An ordered dictionary with the (real) (continuous) input variables.
        """
        inputs_list = OrderedDict()
        
        for i, model in enumerate(self.models):
            local_inputs = model.get_input_list()
            
            for key in local_inputs:
                var = self._convert_local_scalar_variable(i, local_inputs[key])
                inputs_list[var.name] = var
                
        #Remove the connected inputs from the inputs lists (they should not be able to be set externally)
        for i, model in enumerate(self.models):
            for local_input in self.models_dict[model]["local_input"]:
                inputs_list.pop(self._get_global_name(i, local_input))
                
        return inputs_list
        
    def get_output_list(self):
        """
        Returns a dictionary with output variables

        Returns::

            An ordered dictionary with the (real) (continuous) output variables.
        """
        outputs_list = OrderedDict()
        
        for i, model in enumerate(self.models):
            local_outputs = model.get_output_list()
            
            for key in local_outputs:
                var = self._convert_local_scalar_variable(i, local_outputs[key])
                outputs_list[var.name] = var
                
        return outputs_list
        
    def get_variable_nominal(self, variable_name=None, valueref=None):
        """
        Returns the nominal value from a real variable determined by
        either its value reference or its variable name.

        Parameters::

            variable_name --
                The name of the variable.

            valueref --
                The value reference of the variable.

        Returns::

            The nominal value of the given variable.
        """

        if valueref != None:
            local_vr = self._get_local_vr(valueref)#valueref & 0x00000000FFFFFFFF
            model_ind = self._get_model_index_from_vr(valueref)#valueref >> 32
            
            try:
                model = self.models[model_ind]
            except Exception:
                raise fmi.FMUException("Could not map the value reference to the correct model. Is the value reference correct?")
                
            return model.get_variable_nominal(valueref=local_vr)
            
        elif variable_name != None:
            name_parts = variable_name.split(".")
            try:
                model_name = name_parts[0]
                ind = self.names[model_name]
                model = self.models[ind]
            except Exception:
                raise fmi.FMUException("The variable %s could not be found. Was the name correctly prefixed with the model name?"%variable_name)
            
            return model.get_variable_nominal(variable_name=variable_name[len(model_name)+1:])
        else:
            raise fmi.FMUException('Either provide value reference or variable name.')
    
    def get_variable_alias(self, variable_name):
        """
        Return a dict of all alias variables belonging to the provided variable
        where the key are the names and the value indicating whether the variable
        should be negated or not.

        Parameters::

            variable_name--
                Name of the variable to find alias of.

        Returns::

            A dict consisting of the alias variables along with no alias variable.
            The values indicate whether or not the variable should be negated or not.

        Raises::

            FMUException if the variable is not in the model.
        """
        name_parts = variable_name.split(".")
        try:
            model_name = name_parts[0]
            ind = self.names[model_name]
            model = self.models[ind]
        except Exception:
            raise fmi.FMUException("The variable %s could not be found. Was the name correctly prefixed with the model name?"%variable_name)
        
        modified_vars = {}
        vars = model.get_variable_alias(variable_name[len(model_name)+1:])
        for var in vars:
            modified_vars[self._get_global_name(ind, var)] = vars[var]
        
        return modified_vars
    
    @enable_caching
    def get_model_time_varying_value_references(self, filter=None):
        """
        Extract the value references of the variables in a model
        that are time-varying. This method is typically used to
        retrieve the variables for which the result can be stored.

        Parameters::

            filter --
                Filter the variables using a unix filename pattern
                matching (filter="*der*"). Can also be a list of filters
                See http://docs.python.org/2/library/fnmatch.html.
                Default None

        Returns::

            Three lists with the real, integer and boolean value-references
        """
        #WARNING FILTER IS NOT PROPERLY IMPLEMENTED!
        real_ref = []
        int_ref  = []
        bool_ref = []
        
        for i,model in enumerate(self.models):
            [local_reals, local_ints, local_bools] = model.get_model_time_varying_value_references(filter)
            
            for j,vref in enumerate(local_reals):
                local_reals[j] = self._get_global_vr(i, vref)
            for j,vref in enumerate(local_ints):
                local_ints[j] = self._get_global_vr(i, vref)
            for j,vref in enumerate(local_bools):
                local_bools[j] = self._get_global_vr(i, vref)
            
            real_ref.extend(local_reals)
            int_ref.extend(local_ints)
            bool_ref.extend(local_bools)
            
        return real_ref, int_ref, bool_ref
        
    def get_fmu_state(self, state = None):
        """
        Creates a copy of the recent FMU-state and returns
        a pointer to this state which later can be used to
        set the FMU to this state.
        
        Parameters::
        
            state --
                Optionally a pointer to an already allocated FMU state
        
        Returns::

            A pointer to a copy of the recent FMU state.

        Example::

            FMU_state = model.get_fmu_state()
        """
        if not self._supports_get_set_FMU_state():
            raise fmi.FMUException('This coupled FMU does not support get and set FMU-state')

        if state is None:
            state = []
            for model in self.models:
                state.append(model.get_fmu_state())
        else:
            for i,model in enumerate(self.models):
                model.get_fmu_state(state[i])

        return state

    def set_fmu_state(self, state):
        """
        Set the FMU to a previous saved state.

        Parameter::

            state--
                A pointer to a FMU-state.

        Example::

            FMU_state = Model.get_fmu_state()
            Model.set_fmu_state(FMU_state)
        """
        if not self._supports_get_set_FMU_state():
            raise fmi.FMUException('This FMU dos not support get and set FMU-state')
            
        for i,model in enumerate(self.models):
            model.set_fmu_state(state[i])

    def free_fmu_state(self, state):
        """
        Free a previously saved FMU-state from the memory.

        Parameters::

            state--
                A pointer to the FMU-state to be set free.

        Example::

            FMU_state = Model.get_fmu_state()
            Model.free_fmu_state(FMU_state)

        """
        if not self._supports_get_set_FMU_state():
            raise fmi.FMUException('This FMU does not support get and set FMU-state')
        
        for i,model in enumerate(self.models):
            model.free_fmu_state(state[i])

    cpdef serialize_fmu_state(self, state):
        """
        Serialize the data referenced by the input argument.

        Parameters::

            state --
                A FMU-state.

        Returns::
            A vector with the serialized FMU-state.

        Example::
            FMU_state = Model.get_fmu_state()
            serialized_fmu = Model.serialize_fmu_state(FMU_state)
        """
        raise NotImplementedError

    cpdef deserialize_fmu_state(self, serialized_fmu):
        """
        De-serialize the provided byte-vector and returns the corresponding FMU-state.

        Parameters::

            serialized_fmu--
                A serialized FMU-state.

        Returns::

            A deserialized FMU-state.

        Example::

            FMU_state = Model.get_fmu_state()
            serialized_fmu = Model.serialize_fmu_state(FMU_state)
            FMU_state = Model.deserialize_fmu_state(serialized_fmu)
        """
        raise NotImplementedError

    cpdef serialized_fmu_state_size(self, state):
        """
        Returns the required size of a vector needed to serialize the specified FMU-state

        Parameters::

            state--
                A FMU-state

        Returns::

            The size of the vector.
        """
        raise NotImplementedError
        
    def set_debug_logging(self, logging_on, categories = []):
        """
        Specifies if the debugging should be turned on or off.
        Currently the only allowed value for categories is an empty list.

        Parameters::

            logging_on --
                Boolean value.

            categories --
                List of categories to log, call get_categories() for list of categories.
                Default: [] (all categories)

        Calls the low-level FMI function: fmi2SetDebuggLogging
        """
        for model in self.models:
            model.set_debug_logging(logging_on, categories)

    def get_categories(self):
        """
        Method used to retrieve the logging categories.

        Returns::
        
            A list with the categories available for logging.
        """
        categories = []
        for model in self.models:
            categories.append(model.get_categories())
            
        return categories

cdef class CoupledFMUModelME2(CoupledFMUModelBase):
    """
    A class for coupled ME models
    """
    cdef public object _order, _blocks
    cdef public object _values_continuous_states_changed
    cdef public object _nominals_continuous_states_changed
    cdef public object _preinit_nominal_continuous_states
    
    def __init__(self, models, connections):
        """
        Creates a coupled model based on ME FMUs version 2.0.
        
        Parameters::
        
            models  
                    - A list of models that are to be simulated together
                      with model names.
                      The models needs to be a subclass of FMUModelME2.
                      
            connection  
                    - Specifices the connection between the models.
                        
                    - model_begin.variable -> model_accept.variable
                      [(model_source,"beta",model_destination,"y"),(...)]
        
        """
        if not isinstance(models, list):
            raise fmi.FMUException("The models should be provided as a list.")
        for model in models:
            try:
                len(model)
            except TypeError:
                raise fmi.FMUException("The models should be provided as a list of lists with the name" \
                " of the model as the first entry and the model object as the second.")
            if len(model) != 2:
                raise fmi.FMUException("The models should be provided as a list of lists with the name" \
                " of the model as the first entry and the model object as the second.")
            if not isinstance(model[1], fmi.FMUModelME2):
                raise fmi.InvalidFMUException("The coupled model currently only supports ME 2.0 FMUs.")
                
        #Call super
        CoupledFMUModelBase.__init__(self, models, connections)
        
        self._order, self._blocks = self.compute_evaluation_order()
        
        self._values_continuous_states_changed = False
        self._nominals_continuous_states_changed = False                                     

        # State nominals retrieved before initialization
        self._preinit_nominal_continuous_states = None
    
    cpdef _get_time(self):
        """
        Returns the current time of the simulation.

        Returns::
        
            The time.
        """
        return self.__t

    cpdef _set_time(self, FMIL.fmi2_real_t t):
        """
        Sets the current time of the simulation.

        Parameters::
        
            t--
                The time to set.
        """
        for model in self.models:
            model.time = t
        self.__t = t
 
    time = property(_get_time,_set_time)
    
    def _get_continuous_states(self):
        """
        Returns a vector with the values of the continuous states.

        Returns::

            The continuous states.
        """
        states = []
        for model in self.models:
            states.append(model.continuous_states)
            
        return np.concatenate(states).ravel()

    def _set_continuous_states(self, np.ndarray[FMIL.fmi2_real_t, ndim=1, mode="c"] values):
        """
        Set the values of the continuous states.

        Parameters::

            values--
                The new values of the continuous states.
        """
        cdef int ind_start = 0, nbr_states = 0
        for model in self.models:
            nbr_states = self.models_dict[model]["local_state_len"]
            model.continuous_states = values[ind_start:ind_start+nbr_states]
            ind_start = ind_start + nbr_states

    continuous_states = property(_get_continuous_states, _set_continuous_states)
    
    def _get_nominal_continuous_states(self):
        """
        Returns the nominal values of the continuous states.

        Returns::
        
            The nominal values.
        """
        values = []
        for model in self.models:
            values.append(model.nominal_continuous_states)
            
        return np.concatenate(values).ravel()

    nominal_continuous_states = property(_get_nominal_continuous_states)
    
    def _get_connected_outputs(self):
        y = []
        
        for model in self.models:
            for var in self.models_dict[model]["local_output"]:
                y.append(model.get(var))
                
        return np.array(y).ravel()
        
    def _get_connected_inputs(self):
        u = []
        
        for model in self.models:
            for var in self.models_dict[model]["local_input"]:
                u.append(model.get(var))
                
        return np.array(u).ravel()
    
    def _update_coupling_equations(self):
        
        for block in self._blocks:
            if len(block["inputs"]) == 0: #No inputs in this block, only outputs
                for model in block["outputs"].keys():
                    #If the model has not entered initialization mode, enter
                    if model._has_entered_init_mode is False:
                        model.enter_initialization_mode()
                    
                    #Get the outputs (do we need to do this?)
                    #self.get_specific_connection_outputs(model, block["outputs_mask"][model], y)
                    
            elif len(block["outputs"]) == 0: #No outputs in this block
                
                for model in block["inputs"].keys(): #Set the inputs
                    for var in block["inputs"][model]:
                        value = self.models_dict[model]["couplings"][var][0].get(self.models_dict[model]["couplings"][var][1])

                        model.set(var, value)
                    
                    #self.set_specific_connection_inputs(model, block["inputs_mask"][model], u)
                    
            else: #Both (algebraic loop)
                
                #Assert models has entered initialization mode
                for model in list(block["inputs"].keys())+list(block["outputs"].keys()): #Possible only need outputs?
                    if model._has_entered_init_mode is False:
                        model.enter_initialization_mode()
                        
                #Get start values
                
                u = []
                for model in block["inputs"].keys():
                    u.append(model.get(block["inputs"][model]))
                
                u = np.array(u).ravel()
                
                success = False
                if USE_ROOT:
                    res = sopt.root(init_f_block, u, args=(self,block))
                    success = res["success"]
                else:
                    [res, info, ier, msg] = sopt.fsolve(init_f_block, u, args=(self,block), full_output=True)
                    success = True if ier == 1 else False
                
                if not success:
                    raise fmi.FMUException("Failed to converge the block.")
                
                """
                for model in block["outputs"].keys():
                    self.get_specific_connection_outputs(model, block["outputs_mask"][model], y)
                
                if USE_ROOT:
                    res = sopt.root(init_f_block, y[block["global_outputs_mask"]], args=(self,block))
                else:
                    res = sopt.fsolve(init_f_block, y[block["global_outputs_mask"]], args=(self,block))
                if not res["success"]:
                    print res
                    raise Exception("Failed to converge the initialization system.")
                
                y[block["global_outputs_mask"]] = res["x"]
                u = self.L.dot(y.reshape(-1,1))
                
                for model in block["inputs"].keys():
                    #Set the inputs
                    self.set_specific_connection_inputs(model, block["inputs_mask"][model], u)
                """
                #y = g(u)
                #u = Ly
                
                #-> 0 = u_n - Lg(u_n)
    
    cpdef get_derivatives(self):
        """
        Returns the derivative of the continuous states.

        Returns::

            dx --
                The derivatives as an array.

        Example::

            dx = model.get_derivatives()

        """
        self._update_coupling_equations()
        
        values = []
        for model in self.models:
            values.append(model.get_derivatives())
            
        return np.concatenate(values).ravel()

    def enter_event_mode(self):
        """
        Sets the FMU to be in event mode by calling the
        underlying FMU method.
        """
        for model in self.models:
            model.enter_event_mode()
    
    def enter_continuous_time_mode(self):
        """
        Sets the FMU to be in continuous time mode by calling the
        underlying FMU method.
        """
        for model in self.models:
            model.enter_continuous_time_mode()
            
        self._values_continuous_states_changed = False
        self._nominals_continuous_states_changed = False  
    
    def get_event_indicators(self):
        """
        Returns the event indicators at the current time-point.

        Returns::

            evInd --
                The event indicators as an array.

        Example::

            evInd = model.get_event_indicators()

        """
        self._update_coupling_equations()
        
        values = []
        for model in self.models:
            values.append(model.get_event_indicators())
            
        return np.concatenate(values).ravel()

    
    def get_tolerances(self):
        """
        Returns the relative and absolute tolerances. If the relative tolerance
        is defined in the XML-file it is used, otherwise a default of 1.e-4 is
        used. The absolute tolerance is calculated and returned according to
        the FMI specification, atol = 0.01*rtol*(nominal values of the
        continuous states).

        This method should not be called before initialization, since it depends on state nominals.

        Returns::

            rtol --
                The relative tolerance.

            atol --
                The absolute tolerance.

        Example::

            [rtol, atol] = model.get_tolerances()
        """
        rtol = self.get_relative_tolerance()
        atol = self.get_absolute_tolerances()

        return [rtol, atol]

    def get_relative_tolerance(self):
        """
        Returns the relative tolerance. If the relative tolerance
        is defined in the XML-file it is used, otherwise a default of 1.e-4 is
        used.

        Returns::

            rtol --
                The relative tolerance.
        """
        rtol = 0.0
        for model in self.models:
            rtol = max(rtol, model.get_default_experiment_tolerance())
        return rtol

    def get_absolute_tolerances(self):
        """
        Returns the absolute tolerances. They are calculated and returned according to
        the FMI specification, atol = 0.01*rtol*(nominal values of the
        continuous states)

        This method should not be called before initialization, since it depends on state nominals.

        Returns::

            atol --
                The absolute tolerances.
        """
        rtol = self.get_relative_tolerance()
        return 0.01*rtol*self.nominal_continuous_states

    def completed_integrator_step(self, no_set_FMU_state_prior_to_current_point = True):
        """
        This method must be called by the environment after every completed step
        of the integrator. If the return is True, then the environment must call
        event_update() otherwise, no action is needed.

        Returns::
            A tuple of format (a, b) where a and b indicate:
                If a is True -> Call event_update().
                        False -> Do nothing.
                If b is True -> The simulation should be terminated.
                        False -> Do nothing.

        Calls the low-level FMI function: fmi2CompletedIntegratorStep.
        """
        enter_event_mode     = False
        terminate_simulation = False 
        for model in self.models:
            [enter, terminate]   = model.completed_integrator_step()
            enter_event_mode     = enter_event_mode or enter
            terminate_simulation = terminate_simulation or terminate
        
        return enter_event_mode, terminate_simulation


    def get_event_info(self):
        """
        Returns the event information from the FMU.

        Returns::

            The event information, a struct which contains:

            newDiscreteStatesNeeded --
                Event iteration did not converge (if True).

            terminateSimulation --
                Error, terminate simulation (if True).

            nominalsOfContinuousStatesChanged --
                Values of states x have changed (if True).

            valuesOfContinuousStatesChanged --
                ValueReferences of states x changed (if True).

            nextEventTimeDefined -
                If True, nextEventTime is the next time event.

            nextEventTime --
                The next time event.

        Example::

            event_info    = model.event_info
            nextEventTime = model.event_info.nextEventTime
        """

        event_info = fmi.PyEventInfo()
        event_info.newDiscreteStatesNeeded = False
        event_info.terminateSimulation     = False
        event_info.nominalsOfContinuousStatesChanged = self._nominals_continuous_states_changed
        event_info.valuesOfContinuousStatesChanged = self._values_continuous_states_changed
        event_info.nextEventTimeDefined = False
        event_info.nextEventTime = np.inf
        
        for model in self.models:
            local_event_info  = model.get_event_info()
            event_info.newDiscreteStatesNeeded           |= local_event_info.newDiscreteStatesNeeded
            event_info.terminateSimulation               |= local_event_info.terminateSimulation
            event_info.nominalsOfContinuousStatesChanged |= local_event_info.nominalsOfContinuousStatesChanged
            event_info.valuesOfContinuousStatesChanged   |= local_event_info.valuesOfContinuousStatesChanged
            event_info.nextEventTimeDefined |= local_event_info.nextEventTimeDefined
            if local_event_info.nextEventTimeDefined:
                event_info.nextEventTime = min(event_info.nextEventTime, local_event_info.nextEventTime)
            
        return event_info


    def get_capability_flags(self):
        """
        Returns a dictionary with the capability flags of the FMU.

        Returns::
            Dictionary with keys:
                needsExecutionTool
                completedIntegratorStepNotNeeded
                canBeInstantiatedOnlyOncePerProcess
                canNotUseMemoryManagementFunctions
                canGetAndSetFMUstate
                canSerializeFMUstate
                providesDirectionalDerivatives
                completedEventIterationIsProvided
        """
        cdef dict capabilities = {}
        capabilities['needsExecutionTool']   = False
        capabilities['canGetAndSetFMUstate'] = True
        capabilities['canSerializeFMUstate'] = True
        capabilities['providesDirectionalDerivatives']      = True
        capabilities['completedIntegratorStepNotNeeded']    = True
        capabilities['canNotUseMemoryManagementFunctions']  = True
        capabilities['canBeInstantiatedOnlyOncePerProcess'] = True
        
        for model in self.models:
            local_capabilities = model.get_capability_flags()
            capabilities['needsExecutionTool']   = capabilities['needsExecutionTool'] or local_capabilities['needsExecutionTool']
            capabilities['canGetAndSetFMUstate'] = capabilities['canGetAndSetFMUstate'] and local_capabilities['canGetAndSetFMUstate']
            capabilities['canSerializeFMUstate'] = capabilities['canSerializeFMUstate'] and local_capabilities['canSerializeFMUstate']
            capabilities['completedIntegratorStepNotNeeded']    = capabilities['completedIntegratorStepNotNeeded'] and local_capabilities['completedIntegratorStepNotNeeded']
            capabilities['canNotUseMemoryManagementFunctions']  = capabilities['canNotUseMemoryManagementFunctions'] and local_capabilities['canNotUseMemoryManagementFunctions']
            capabilities['providesDirectionalDerivatives']      = capabilities['providesDirectionalDerivatives'] and local_capabilities['providesDirectionalDerivatives']
            capabilities['canBeInstantiatedOnlyOncePerProcess'] = capabilities['canBeInstantiatedOnlyOncePerProcess'] and local_capabilities['canBeInstantiatedOnlyOncePerProcess']

        return capabilities
    
    def _provides_directional_derivatives(self):
        """
        Check capability to provide directional derivatives.
        """
        capabilities = self.get_capability_flags()
        return capabilities['providesDirectionalDerivatives']
        
    def _compare_connected_outputs(self, y, y_new):
        cdef double sum = 0.0
        cdef int i, N = len(y)
        
        for i in range(N):
            prod = (y[i]-y_new[i]) * (1 / (self.__tolerance*y[i]+self.__tolerance)) #Missing nominals
            sum += prod*prod
        
        err = (sum/N)**0.5 if N > 0 else 0.0 #If there are no connections between the models
        
        if err < 1:
            return False #No new discrete states needed
        else:
            return True #Discrete states needed
    
    def event_update(self, intermediateResult=False):
        """
        Updates the event information at the current time-point. If
        intermediateResult is set to True the update_event will stop at each
        event iteration which would require to loop until
        event_info.newDiscreteStatesNeeded == fmiFalse.

        Parameters::

            intermediateResult --
                If set to True, the update_event will stop at each event
                iteration.
                Default: False.

        Example::

            model.event_update()

        """
        if intermediateResult:
            for model in self.models:
                model.event_update(True)
        else:
            tmp_values_continuous_states_changed   = False
            tmp_nominals_continuous_states_changed = False
            new_discrete_states_needed = True
            y = self._get_connected_outputs()
            
            while new_discrete_states_needed: #Missing propagation between models!
                for model in self.models:
                    model.event_update() #Should maybe be set to true
                
                #Check if any used output has changed! If so, recompute the inputs and run again
                y_new = self._get_connected_outputs()
                new_discrete_states_needed = self._compare_connected_outputs(y, y_new)
                #print "Events: ", self.time, self.continuous_states 
                #print " Data: ", y, y_new, new_discrete_states_needed
                y = y_new
                
                event_info = self.get_event_info()
                
                new_discrete_states_needed |= event_info.newDiscreteStatesNeeded
                
                if new_discrete_states_needed:
                    self._update_coupling_equations() #Do we need to call the derivatives?
                if event_info.nominalsOfContinuousStatesChanged:
                    tmp_nominals_continuous_states_changed = True
                if event_info.valuesOfContinuousStatesChanged:
                    tmp_values_continuous_states_changed = True
                
            #Set to any model if something changed (this will propagate to the coupled system)
            if tmp_values_continuous_states_changed:
                self._values_continuous_states_changed = True
            if tmp_nominals_continuous_states_changed:
                self._nominals_continuous_states_changed = True
    
    def simulate_options(self, algorithm='AssimuloFMIAlg'):
        """
        Get an instance of the simulate options class, filled with default
        values. If called without argument then the options class for the
        default simulation algorithm will be returned.

        Parameters::

            algorithm --
                The algorithm for which the options class should be fetched.
                Possible values are: 'AssimuloFMIAlg'.
                Default: 'AssimuloFMIAlg'

        Returns::

            Options class for the algorithm specified with default values.
        """
        return self._default_options('pyfmi.fmi_algorithm_drivers', algorithm)
        
    def simulate(self,
                 start_time="Default",
                 final_time="Default",
                 input=(),
                 algorithm='AssimuloFMIAlg',
                 options={}):
        """
        Compact function for model simulation.

        The simulation method depends on which algorithm is used, this can be
        set with the function argument 'algorithm'. Options for the algorithm
        are passed as option classes or as pure dicts. See
        FMUModel.simulate_options for more details.

        The default algorithm for this function is AssimuloFMIAlg.

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
                the data matrix or a function. If a data matrix, the first
                column should be a time vector and then the variable vectors as
                columns. If instead a function, the argument should correspond
                to time and the output the variable data. See the users-guide
                for examples.
                Default: Empty tuple.

            algorithm --
                The algorithm which will be used for the simulation is specified
                by passing the algorithm class as string or class object in this
                argument. 'algorithm' can be any class which implements the
                abstract class AlgorithmBase (found in algorithm_drivers.py). In
                this way it is possible to write own algorithms and use them
                with this function.
                Default: 'AssimuloFMIAlg'

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
