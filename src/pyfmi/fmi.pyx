#!/usr/bin/env python 
# -*- coding: utf-8 -*-

#    Copyright (C) 2009 Modelon AB
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
Module containing the FMI interface Python wrappers.
"""
import os
import sys
import logging

import numpy as N 
cimport numpy as N

N.import_array()

cimport fmil_import as FMIL

from pyfmi.common.core import create_temp_dir, delete_temp_dir
#from pyfmi.common.core cimport BaseModel


int = N.int32
N.int = N.int32

"""Basic flags related to FMI"""

FMI_TRUE = '\x01'
FMI_FALSE = '\x00'

# Status
FMI_OK = FMIL.fmi1_status_ok
FMI_WARNING = FMIL.fmi1_status_warning
FMI_DISCARD = FMIL.fmi1_status_discard
FMI_ERROR = FMIL.fmi1_status_error
FMI_FATAL = FMIL.fmi1_status_fatal
FMI_PENDING = FMIL.fmi1_status_pending

# Types
FMI_REAL = FMIL.fmi1_base_type_real
FMI_INTEGER  = FMIL.fmi1_base_type_int
FMI_BOOLEAN = FMIL.fmi1_base_type_bool
FMI_STRING = FMIL.fmi1_base_type_str
FMI_ENUMERATION = FMIL.fmi1_base_type_enum

# Alias data
FMI_NO_ALIAS = FMIL.fmi1_variable_is_not_alias
FMI_ALIAS = FMIL.fmi1_variable_is_alias
FMI_NEGATED_ALIAS = FMIL.fmi1_variable_is_negated_alias

# Variability
FMI_CONTINUOUS = FMIL.fmi1_variability_enu_continuous
FMI_CONSTANT = FMIL.fmi1_variability_enu_constant
FMI_PARAMETER = FMIL.fmi1_variability_enu_parameter
FMI_DISCRETE = FMIL.fmi1_variability_enu_discrete

# Causality
FMI_INPUT = FMIL.fmi1_causality_enu_input
FMI_OUTPUT = FMIL.fmi1_causality_enu_output
FMI_INTERNAL = FMIL.fmi1_causality_enu_internal
FMI_NONE = FMIL.fmi1_causality_enu_none

# FMI types
FMI_ME = FMIL.fmi1_fmu_kind_enu_me
FMI_CS_STANDALONE = FMIL.fmi1_fmu_kind_enu_cs_standalone

FMI_MIME_CS_STANDALONE = "application/x-fmu-sharedlibrary"

FMI_REGISTER_GLOBALLY = 1



"""Flags for evaluation of FMI Jacobians"""
"""Evaluate Jacobian w.r.t. states."""
FMI_STATES = 1
"""Evaluate Jacobian w.r.t. inputs."""
FMI_INPUTS = 2
"""Evaluate Jacobian of derivatives."""
FMI_DERIVATIVES = 1
"""Evaluate Jacobian of outputs."""
FMI_OUTPUTS = 2

#CALLBACKS
cdef void importlogger(FMIL.jm_callbacks* c, FMIL.jm_string module, int log_level, FMIL.jm_string message):
    #print "FMIL: module = %s, log level = %d: %s"%(module, log_level, message)
    (<FMUModelBase>c.context)._logger(module,log_level,message)

#CALLBACKS
cdef void importlogger_load_fmu(FMIL.jm_callbacks* c, FMIL.jm_string module, int log_level, FMIL.jm_string message):
    if log_level <= c.log_level:
        print "FMIL: module = %s, log level = %d: %s"%(module, log_level, message)
    #(<FMUModelBase>c.context)._logger(module,log_level,message)

cdef void fmilogger(FMIL.fmi1_component_t c, FMIL.fmi1_string_t instanceName, FMIL.fmi1_status_t status, FMIL.fmi1_string_t category, FMIL.fmi1_string_t message, ...):
    cdef char buf[1000]
    cdef FMIL.va_list args
    FMIL.va_start(args, message)
    FMIL.vsnprintf(buf, 1000, message, args)
    FMIL.va_end(args)
    print "FMU: fmiStatus = %d;  %s (%s): %s\n"%(status, instanceName, category, buf)

cdef class BaseModel:
    """ 
    Abstract Model class containing base functionality.
    """

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
        
            (FMU/JMU)Model.set('damper.d', 1.1)
            (FMU/JMU)Model.set(['damper.d','gear.a'], [1.1, 10])
        """
        if isinstance(variable_name, basestring):
            self._set(variable_name, value) #Scalar case
        else:
            for i in xrange(len(variable_name)): #A list of variables
                self._set(variable_name[i], value[i])
    
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
            (FMU/JMU)Model.get('damper.d') 
            # Returns a list of the variables
            (FMU/JMU)Model.get(['damper.d','gear.a'])
        """
        if isinstance(variable_name, basestring):
            return self._get(variable_name) #Scalar case
        else:
            ret = []
            for i in xrange(len(variable_name)): #A list of variables
                ret += [self._get(variable_name[i])]
            return ret
    
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
        algdrive = __import__(base_path, globals(), locals(), [], -1)
        AlgorithmBase = getattr(getattr(algdrive,"algorithm_drivers"), 'AlgorithmBase')
        
        if isinstance(algorithm, basestring):
            module = __import__(module, globals(), locals(), [algorithm], -1)
            algorithm = getattr(module, algorithm)
        
        if not issubclass(algorithm, AlgorithmBase):
            raise Exception(str(algorithm)+
            " must be a subclass of common.algorithm_drivers.AlgorithmBase")

        # initialize algorithm
        alg = algorithm(start_time, final_time, input, self, options)
        # simulate
        alg.solve()
        # get and return result
        return alg.get_result()

        
    def _default_options(self, module, algorithm):
        """ 
        Help method. Gets the options class for the algorithm specified in 
        'algorithm'.
        """
        module = __import__(module, globals(), locals(), [algorithm], -1)
        algorithm = getattr(module, algorithm)
        
        return algorithm.get_default_options()

class FMUException(Exception):
    """
    An FMU exception.
    """
    pass

class PyEventInfo():
    pass

cdef class ScalarVariable:
    """ 
    Class defining data structure based on the XML element ScalarVariable.
    """
    cdef object _name
    cdef FMIL.fmi1_value_reference_t _value_reference
    cdef object _description #A characater pointer but we need an own reference and this is sufficient
    cdef FMIL.fmi1_base_type_enu_t _type
    cdef FMIL.fmi1_variability_enu_t _variability
    cdef FMIL.fmi1_causality_enu_t _causality
    cdef FMIL.fmi1_variable_alias_kind_enu_t _alias
    
    def __init__(self, name, value_reference, type, description="",
                       variability=FMIL.fmi1_variability_enu_continuous,
                       causality=FMIL.fmi1_causality_enu_internal, 
                       alias=FMIL.fmi1_variable_is_not_alias):
        
        self._name            = name
        self._value_reference = value_reference
        self._type            = type
        self._description     = description
        self._variability     = variability
        self._causality       = causality
        self._alias           = alias
        
    def _get_name(self):
        """ 
        Get the value of the name attribute.
        
        Returns::
        
            The name attribute value as string.
        """
        return self._name
    name = property(_get_name)
    
    def _get_value_reference(self):
        """ 
        Get the value of the value reference attribute.
        
        Returns::
        
            The value reference as unsigned int.
        """
        return self._value_reference
    value_reference = property(_get_value_reference)
    
    def _get_type(self):
        """ 
        Get the value of the data type attribute.
        
        Returns::
        
            The data type attribute value as enumeration: FMI_REAL, 
            FMI_INTEGER, FMI_BOOLEAN, FMI_ENUMERATION or FMI_STRING.
        """
        return self._type
    type = property(_get_type)
    
    def _get_description(self):
        """ 
        Get the value of the description attribute.
        
        Returns::
        
            The description attribute value as string (empty string if 
            not set).
        """
        return self._description
    description = property(_get_description)
    
    def _get_variability(self):
        """ 
        Get the value of the variability attribute.
        
        Returns::
        
            The variability attribute value as enumeration: 
            FMI_CONTINUOUS, FMI_CONSTANT, FMI_PARAMETER or FMI_DISCRETE.
        """
        return self._variability
    variability = property(_get_variability)
    
    def _get_causality(self):
        """ 
        Get the value of the causality attribute.
        
        Returns::
        
            The causality attribute value as enumeration: FMI_INTERNAL, 
            FMI_INPUT, FMI_OUTPUT or FMI_NONE.
        """
        return self._causality
    causality = property(_get_causality)
    
    def _get_alias(self):
        """ 
        Get the value of the alias attribute.
        
        Returns::
        
            The alias attribute value as enumeration: FMI_NO_ALIAS, 
            FMI_ALIAS or FMI_NEGATED_ALIAS.
        """
        return self._alias
    alias = property(_get_alias)
    

cdef class FMUModelBase(BaseModel):
    """
    An FMI Model loaded from a DLL.
    """
    #FMIL related variables
    cdef FMIL.fmi1_callback_functions_t callBackFunctions
    cdef FMIL.jm_callbacks callbacks
    cdef FMIL.fmi_import_context_t* context 
    cdef FMIL.fmi1_import_t* _fmu
    cdef FMIL.fmi1_event_info_t _eventInfo
    cdef FMIL.fmi1_import_variable_list_t *variable_list
    cdef FMIL.fmi1_fmu_kind_enu_t fmu_kind
    cdef FMIL.jm_status_enu_t jm_status
    
    #Internal values
    cdef public object __t
    cdef public object _file_open
    cdef public object _npoints
    cdef public object _enable_logging
    cdef public object _pyEventInfo
    cdef list _log
    cdef int _version
    cdef object _allocated_dll, _allocated_context, _allocated_xml, _allocated_fmu
    cdef object _allocated_list
    cdef object _modelid
    cdef object _modelname
    cdef unsigned int _nEventIndicators
    cdef unsigned int _nContinuousStates
    cdef public list _save_real_variables_val
    cdef public list _save_int_variables_val
    cdef public list _save_bool_variables_val
    cdef public object _fmu_temp_dir
    cdef int _fmu_kind

    def __init__(self, fmu, path='.', enable_logging=True):
        """
        Constructor.
        """
        cdef int status
        cdef int version
        
        #Contains the log information
        self._log = []
        
        #Used for deallocation
        self._allocated_context = False
        self._allocated_dll = False
        self._allocated_xml = False
        self._allocated_fmu = False
        self._allocated_list = False
        self._fmu_temp_dir = None
        
        fmu_full_path = os.path.abspath(os.path.join(path,fmu))
        fmu_temp_dir  = create_temp_dir()
        self._fmu_temp_dir = fmu_temp_dir
        
        # Check that the file referenced by fmu has the correct file-ending
        if not fmu_full_path.endswith(".fmu"):
            raise FMUException("FMUModel must be instantiated with an FMU (.fmu) file.")
        
        #Specify the general callback functions
        self.callbacks.malloc  = FMIL.malloc
        self.callbacks.calloc  = FMIL.calloc
        self.callbacks.realloc = FMIL.realloc
        self.callbacks.free    = FMIL.free
        self.callbacks.logger  = importlogger
        #self.callbacks.context = NULL;
        self.callbacks.context = <void*>self #Class loggger
        self.callbacks.log_level = FMIL.jm_log_level_warning if enable_logging else FMIL.jm_log_level_nothing
        
        #Specify FMI related callbacks
        self.callBackFunctions.logger = FMIL.fmi1_log_forwarding;
        #self.callBackFunctions.logger = fmilogger;
        self.callBackFunctions.allocateMemory = FMIL.calloc;
        self.callBackFunctions.freeMemory = FMIL.free;
        self.callBackFunctions.stepFinished = NULL;
        
        self.context = FMIL.fmi_import_allocate_context(&self.callbacks)
        self._allocated_context = True
        
        #Get the FMI version of the provided model
        version = FMIL.fmi_import_get_fmi_version(self.context, fmu_full_path, fmu_temp_dir)
        self._version = version #Store version
        
        if version == FMIL.fmi_version_unknown_enu:
            last_error = FMIL.jm_get_last_error(&self.callbacks)
            raise FMUException("The FMU version could not be determined. "+last_error)
        if version != 1:
            raise FMUException("PyFMI currently only supports FMI 1.0.")
        
        #Parse the XML
        self._fmu = FMIL.fmi1_import_parse_xml(self.context, fmu_temp_dir)
        self._allocated_xml = True
        
        #Check the FMU kind
        fmu_kind = FMIL.fmi1_import_get_fmu_kind(self._fmu)
        if fmu_kind != FMI_ME and fmu_kind != FMI_CS_STANDALONE:
            raise FMUException("PyFMI currently only supports FMI 1.0 for Model Exchange.")
        self._fmu_kind = fmu_kind
        
        #Connect the DLL
        global FMI_REGISTER_GLOBALLY
        status = FMIL.fmi1_import_create_dllfmu(self._fmu, self.callBackFunctions, FMI_REGISTER_GLOBALLY);
        if status == FMIL.jm_status_error:
            last_error = FMIL.fmi1_import_get_last_error(self._fmu)
            raise FMUException(last_error)
            #raise FMUException("The DLL could not be loaded, reported error: "+ last_error)
        self._allocated_dll = True
        FMI_REGISTER_GLOBALLY += 1 #Update the global register of FMUs
        
        #Default values
        self.__t = None
        
        #Internal values
        self._file_open = False
        self._npoints = 0
        self._enable_logging = enable_logging
        self._pyEventInfo = PyEventInfo()
        
        #Load information from model
        self._modelid = FMIL.fmi1_import_get_model_identifier(self._fmu)
        self._modelname = FMIL.fmi1_import_get_model_name(self._fmu)
        self._nEventIndicators = FMIL.fmi1_import_get_number_of_event_indicators(self._fmu)
        self._nContinuousStates = FMIL.fmi1_import_get_number_of_continuous_states(self._fmu)
        
        #Instantiates the model
        #if fmu_kind == FMI_ME:
        #    self.instantiate_model(logging = enable_logging)
        #elif fmu_kind == FMI_CS_STANDALONE:
        #    self.instantiate_slave(logging = enable_logging)
        #else:
        #    raise FMUException("Unknown FMU kind.")
        
        #Store the continuous and discrete variables for result writing
        reals_continuous = self.get_model_variables(type=0, include_alias=False, variability=3)
        reals_discrete = self.get_model_variables(type=0, include_alias=False, variability=2)
        int_discrete = self.get_model_variables(type=1, include_alias=False, variability=2)
        bool_discrete = self.get_model_variables(type=2, include_alias=False, variability=2)
        
        self._save_real_variables_val = [var.value_reference for var in reals_continuous.values()]+[var.value_reference for var in reals_discrete.values()]
        self._save_int_variables_val  = [var.value_reference for var in int_discrete.values()]
        self._save_bool_variables_val = [var.value_reference for var in bool_discrete.values()]
        
        """
        #Create a JMIModel if a JModelica generated FMU is loaded
        # This is convenient for debugging purposes
        # Requires uncommenting of the alternative constructor
        # in JMUModel
        try:
            self._fmiGetJMI = self._dll.__getattr__('fmiGetJMI')
            self._fmiInstantiateModel.restype = C.c_voidp()
            self._fmiGetJMI.argtypes = [self._fmiComponent]
            self._jmi = self._fmiGetJMI(self._model)
            self._jmimodel = jmodelica.jmi.JMIModel(self._dll,self._jmi)
        except:
            print "Could not create JMIModel"
            pass
        """
    cdef _logger(self, FMIL.jm_string module, int log_level, FMIL.jm_string message):
        #print "FMIL: module = %s, log level = %d: %s"%(module, log_level, message)
        self._log.append([module,log_level,message])
    
    def get_log(self):
        """
        Returns the log information as a list. To turn on the logging use the 
        method, set_debug_logging(True) in the instantiation, 
        FMUModel(..., enable_logging=True). The log is stored as a list of lists. 
        For example log[0] are the first log message to the log and consists of, 
        in the following order, the instance name, the status, the category and 
        the message.
        
        Returns::
        
            log - A list of lists.
        """
        return self._log
        
    def print_log(self):
        """
        Prints the log information to the prompt.
        """
        cdef int N = len(self._log)
        
        for i in range(N):
            print "FMIL: module = %s, log level = %d: %s"%(self._log[i][0], self._log[i][1], self._log[i][2])
    
    #def __dealloc__(self):
    #    """
    #    Deallocate memory allocated
    #    """
    #    if self._allocated_fmu:
    #        FMIL.fmi1_import_terminate(self._fmu)
    #        FMIL.fmi1_import_free_model_instance(self._fmu)
    #    
    #    if self._allocated_dll:
    #        FMIL.fmi1_import_destroy_dllfmu(self._fmu)
    #        
    #    if self._allocated_xml:  
    #        FMIL.fmi1_import_free(self._fmu)
    #    
    #    if self._allocated_context:
    #        FMIL.fmi_import_free_context(self.context)
    #    
    #    if self._fmu_temp_dir:
    #        delete_temp_dir(self._fmu_temp_dir)

    def _get_version(self):
        """
        Returns the FMI version of the Model which it was generated according.
                
        Returns::
        
            version -- 
                The version.
                
        Example::
        
            model.version
        """
        version = FMIL.fmi1_import_get_version(self._fmu)
        return version
        
    version = property(fget=_get_version)
    
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
        
            [nCont, nEvent] = model.get_ode_sizes()
        """
        return self._nContinuousStates, self._nEventIndicators
    
    def get_real(self, valueref):
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
                
        Calls the low-level FMI function: fmiGetReal/fmiSetReal
        """
        cdef int status
        cdef FMIL.size_t nref
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()
        nref = len(val_ref)
        cdef N.ndarray[FMIL.fmi1_real_t, ndim=1,mode='c'] val = N.array([0.0]*nref,dtype=N.float, ndmin=1)
        
        
        status = FMIL.fmi1_import_get_real(self._fmu, <FMIL.fmi1_value_reference_t*>val_ref.data, nref, <FMIL.fmi1_real_t*>val.data)
        
        if status != 0:
            raise FMUException('Failed to get the Real values.')
            
        return val
        
    def set_real(self, valueref, values):
        """
        Sets the real-values in the FMU as defined by the valuereference(s).
        
        Parameters::
        
            valueref -- 
                A list of valuereferences.
                
            values -- 
                Values to be set.
        
        Example::
        
            model.set_real([234,235],[2.34,10.4])
        
        Calls the low-level FMI function: fmiGetReal/fmiSetReal
        """
        cdef int status
        cdef FMIL.size_t nref
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()
        cdef N.ndarray[FMIL.fmi1_real_t, ndim=1,mode='c'] val = N.array(values, dtype=N.float, ndmin=1).flatten()
        nref = len(val_ref)
        
        if val_ref.size != val.size:
            raise FMUException(
                'The length of valueref and values are inconsistent.')
        
        status = FMIL.fmi1_import_set_real(self._fmu, <FMIL.fmi1_value_reference_t*>val_ref.data, nref, <FMIL.fmi1_real_t*>val.data)
        
        if status != 0:
            raise FMUException('Failed to set the Real values.')
        
    def get_integer(self, valueref):
        """
        Returns the integer-values from the valuereference(s).
        
        Parameters::
        
            valueref -- 
                A list of valuereferences.
                
        Return::
            
            values -- 
                The values retrieved from the FMU.
                
        Example::
        
            val = model.get_integer([232])
                
        Calls the low-level FMI function: fmiGetInteger/fmiSetInteger
        """
        cdef int status
        cdef FMIL.size_t nref
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()

        nref = len(valueref)
        cdef N.ndarray[FMIL.fmi1_integer_t, ndim=1,mode='c'] val = N.array([0]*nref, dtype=int,ndmin=1)
        
        status = FMIL.fmi1_import_get_integer(self._fmu, <FMIL.fmi1_value_reference_t*>val_ref.data, nref, <FMIL.fmi1_integer_t*>val.data)
        
        if status != 0:
            raise FMUException('Failed to get the Integer values.')
            
        return val
        
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
        
        Calls the low-level FMI function: fmiGetInteger/fmiSetInteger
        """
        cdef int status
        cdef FMIL.size_t nref
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()
        cdef N.ndarray[FMIL.fmi1_integer_t, ndim=1,mode='c'] val = N.array(values, dtype=int,ndmin=1).flatten()
        
        nref = val_ref.size
        
        if val_ref.size != val.size:
            raise FMUException(
                'The length of valueref and values are inconsistent.')
        
        status = FMIL.fmi1_import_set_integer(self._fmu, <FMIL.fmi1_value_reference_t*>val_ref.data, nref, <FMIL.fmi1_integer_t*>val.data)
        
        if status != 0:
            raise FMUException('Failed to set the Integer values.')
        
        
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
                
        Calls the low-level FMI function: fmiGetBoolean/fmiSetBoolean
        """
        cdef int status
        cdef FMIL.size_t nref
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()

        nref = val_ref.size
        #cdef N.ndarray[FMIL.fmi1_boolean_t, ndim=1,mode='c'] val = N.array(['0']*nref, dtype=N.char.character,ndmin=1)
        cdef void *val = FMIL.malloc(sizeof(FMIL.fmi1_boolean_t)*nref)
        

        #status = FMIL.fmi1_import_get_boolean(self._fmu, <FMIL.fmi1_value_reference_t*>val_ref.data, nref, <FMIL.fmi1_boolean_t*>val.data)
        status = FMIL.fmi1_import_get_boolean(self._fmu, <FMIL.fmi1_value_reference_t*>val_ref.data, nref, <FMIL.fmi1_boolean_t*>val)
        
        return_values = []
        for i in range(nref):
            return_values.append((<FMIL.fmi1_boolean_t*>val)[i]==1)
            #print (<FMIL.fmi1_boolean_t*>val)[i], (<FMIL.fmi1_boolean_t*>val)[i]==1
        
        #print return_values
        
        #Dealloc
        FMIL.free(val)

        if status != 0:
            raise FMUException('Failed to get the Boolean values.')
        
        return N.array(return_values)
        #return val==FMI_TRUE
        
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
        
        Calls the low-level FMI function: fmiGetBoolean/fmiSetBoolean
        """
        cdef int status
        cdef FMIL.size_t nref
        
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()
        nref = val_ref.size
        
        #cdef N.ndarray[FMIL.fmi1_boolean_t, ndim=1,mode='c'] val = N.array(['0']*nref, dtype=N.char.character,ndmin=1).flatten()
        cdef void *val = FMIL.malloc(sizeof(FMIL.fmi1_boolean_t)*nref)
        
        values = N.array(values,ndmin=1).flatten()
        for i in range(nref):
            if values[i]:
                #val[i]=1
                (<FMIL.fmi1_boolean_t*>val)[i] = 1
            else:
                #val[i]=0
                (<FMIL.fmi1_boolean_t*>val)[i] = 0
        
        if val_ref.size != values.size:
            raise FMUException(
                'The length of valueref and values are inconsistent.')
        
        #status = FMIL.fmi1_import_set_boolean(self._fmu, <FMIL.fmi1_value_reference_t*>val_ref.data, nref, <FMIL.fmi1_boolean_t*>val.data)
        status = FMIL.fmi1_import_set_boolean(self._fmu, <FMIL.fmi1_value_reference_t*>val_ref.data, nref, <FMIL.fmi1_boolean_t*>val)
        
        FMIL.free(val)
        
        if status != 0:
            raise FMUException('Failed to set the Boolean values.')
        
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
                
        Calls the low-level FMI function: fmiGetString/fmiSetString
        """
        raise NotImplementedError
        cdef int status
        cdef FMIL.size_t nref
        valueref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()

        nref = len(valueref)
        values = N.ndarray([])
        
        temp = (self._fmiString*nref)()
        
        status = self._fmiGetString(self._model, valueref, nref, temp)

        if status != 0:
            raise FMUException('Failed to get the String values.')
        
        return N.array(temp)[:]
    
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
        
        Calls the low-level FMI function: fmiGetString/fmiSetString
        """
        raise NotImplementedError
        cdef int status
        cdef FMIL.size_t nref
        
        valueref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()

        nref = valueref.size
        values = N.array(values).flatten()
        
        temp = (self._fmiString*nref)()
        for i in range(nref):
            temp[i] = values[i]
        
        if valueref.size != values.size:
            raise FMUException(
                'The length of valueref and values are inconsistent.')

        status = self._fmiSetString(self._model, valueref, nref, temp)
        
        if status != 0:
            raise FMUException('Failed to set the String values.')
    
    def set_debug_logging(self,flag):
        """
        Specifies if the debugging should be turned on or off.
        
        Parameters::
        
            flag -- 
                Boolean value.
                
        Calls the low-level FMI function: fmiSetDebuggLogging
        """
        cdef FMIL.fmi1_boolean_t log
        cdef int status
        
        self.callbacks.log_level = FMIL.jm_log_level_warning if flag else FMIL.jm_log_level_nothing
        
        if flag:
            log = 1
        else:
            log = 0
        
        status = FMIL.fmi1_import_set_debug_logging(self._fmu, log)
        self._enable_logging = bool(log)
            
        if status != 0:
            raise FMUException('Failed to set the debugging option.')

    
    #def reset(self):
    #    """ 
    #    Calling this function is equivalent to reopening the model.
    #    """
    #    #Instantiate
    #    self.instantiate_model()
    #    
    #    #Default values
    #    self.__t = None
    #    
    #    #Internal values
    #    self._file_open = False
    #    self._npoints = 0
    #    self._log = []
    
    def set_fmil_log_level(self, FMIL.jm_log_level_enu_t level):
        """
        Specifices the log level for FMI Library. Note that this is
        different from the FMU logging which is specificed via
        set_debug_logging.
        
        Parameters::
		
            level --
                The log level. Available values:
                    NOTHING = 0
                    FATAL = 1
                    ERROR = 2
                    WARNING = 3
                    INFO = 4
                    VERBOSE = 5 
                    DEBUG = 6
                    ALL = 7
        """
        if level < 0 or level > 7:
            raise FMUException("Invalid log level for FMI Library (0-7).")
        self.callbacks.log_level = level

    
    def _set(self,char* variable_name, value):
        """
        Helper method to set, see docstring on set.
        """
        cdef FMIL.fmi1_value_reference_t ref
        cdef FMIL.fmi1_base_type_enu_t type
        
        ref = self.get_variable_valueref(variable_name)
        type = self.get_variable_data_type(variable_name)
        
        if type == FMIL.fmi1_base_type_real:  #REAL
            self.set_real([ref], [value])
        elif type == FMIL.fmi1_base_type_int: #INTEGER
            self.set_integer([ref], [value])
        elif type == FMIL.fmi1_base_type_str: #STRING
            self.set_string([ref], [value])
        elif type == FMIL.fmi1_base_type_bool: #BOOLEAN
            self.set_boolean([ref], [value])
        else:
            raise FMUException('Type not supported.')
        
    
    def _get(self,char* variable_name):
        """
        Helper method to get, see docstring on get.
        """
        cdef FMIL.fmi1_value_reference_t ref
        cdef FMIL.fmi1_base_type_enu_t type
        
        ref = self.get_variable_valueref(variable_name)
        type = self.get_variable_data_type(variable_name)
        
        if type == FMIL.fmi1_base_type_real:  #REAL
            return self.get_real([ref])
        elif type == FMIL.fmi1_base_type_int: #INTEGER
            return self.get_integer([ref])
        elif type == FMIL.fmi1_base_type_str: #STRING
            return self.get_string([ref])
        elif type == FMIL.fmi1_base_type_bool: #BOOLEAN
            return self.get_boolean([ref])
        else:
            raise FMUException('Type not supported.')
        
    cpdef get_variable_description(self, char* variablename):
        """
        Get the description of a given variable.
        
        Parameter::
        
            variablename --
                The name of the variable
                
        Returns::
        
            The description of the variable.
        """
        cdef FMIL.fmi1_import_variable_t* variable
        cdef char* desc
        
        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
        
        desc = FMIL.fmi1_import_get_variable_description(variable)
        
        return desc if desc != NULL else ""
        
    cpdef FMIL.fmi1_base_type_enu_t get_variable_data_type(self,char* variablename) except *:
        """ 
        Get data type of variable. 
        
        Parameter::
        
            variablename --
                The name of the variable.
                
        Returns::
        
            The type of the variable.
        """
        cdef FMIL.fmi1_import_variable_t* variable
        cdef FMIL.fmi1_base_type_enu_t type
        
        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
        
        type = FMIL.fmi1_import_get_variable_base_type(variable)
        
        return type
        
    cpdef FMIL.fmi1_value_reference_t get_variable_valueref(self, char* variablename) except *:
        """
        Extract the ValueReference given a variable name.
        
        Parameters::
        
            variablename -- 
                The name of the variable.
            
        Returns::
        
            The ValueReference for the variable passed as argument.
        """
        cdef FMIL.fmi1_import_variable_t* variable
        cdef FMIL.fmi1_value_reference_t vr
        
        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
        vr =  FMIL.fmi1_import_get_variable_vr(variable)
        
        return vr
        
    def get_variable_nominal(self, variablename=None, valueref=None):
        """
        Returns the nominal value from a real variable determined by
        either its value reference or its variable name.
        
        Parameters::
            
            valueref --
                The value reference of the variable.
            variablename --
                The name of the variable.
            
        Returns::
        
            The nominal value of the given variable.
        """
        cdef FMIL.fmi1_import_variable_t *variable
        cdef FMIL.fmi1_import_real_variable_t *real_variable

        if valueref != None:
            variable = FMIL.fmi1_import_get_variable_by_vr(self._fmu, FMIL.fmi1_base_type_real, <FMIL.fmi1_value_reference_t>valueref)
            if variable == NULL:
                raise FMUException("The variable with value reference: %s, could not be found."%str(valueref))
        elif variablename != None:
            variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
            if variable == NULL:
                raise FMUException("The variable %s could not be found."%variablename)
        else:
            raise FMUException('Either provide value reference or variable name.')
        
        real_variable = FMIL.fmi1_import_get_variable_as_real(variable)
        if real_variable == NULL:
            raise FMUException("The variable is not a real variable.")
        
        return  FMIL.fmi1_import_get_real_variable_nominal(real_variable)
    
    cpdef get_variable_fixed(self, char* variablename):
        """
        Returns if the start value is fixed (True - The value is used as
        an initial value) or not (False - The value is used as a guess
        value).
        
        Parameters::
        
            variablename --
                The name of the variable
                
        Returns::
        
            If the start value is fixed or not.
        """
        cdef FMIL.fmi1_import_variable_t *variable
        
        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
    
        status = FMIL.fmi1_import_get_variable_has_start(variable)
        
        if status == 0:
            raise FMUException("The variable %s does not have a start value."%variablename)
        
        fixed = FMIL.fmi1_import_get_variable_is_fixed(variable)
        
        return fixed==1
        
    
    cpdef get_variable_start(self,char* variablename):
        """
        Returns the start value for the variable or else raises
        FMUException.
        
        Parameters::
            
            variablename --
                The name of the variable
                
        Returns::
        
            The start value.
        """
        cdef FMIL.fmi1_import_variable_t *variable
        cdef FMIL.fmi1_import_integer_variable_t* int_variable
        cdef FMIL.fmi1_import_real_variable_t* real_variable
        cdef FMIL.fmi1_import_bool_variable_t* bool_variable
        cdef FMIL.fmi1_import_enum_variable_t* enum_variable
        cdef FMIL.fmi1_import_string_variable_t*  str_variable
        cdef FMIL.fmi1_base_type_enu_t type
        cdef int status
        cdef FMIL.fmi1_boolean_t FMITRUE = 1
        
        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
    
        status = FMIL.fmi1_import_get_variable_has_start(variable)
        
        if status == 0:
            raise FMUException("The variable %s does not have a start value."%variablename)
        
        type = FMIL.fmi1_import_get_variable_base_type(variable)
        
        if type == FMIL.fmi1_base_type_real:
            real_variable = FMIL.fmi1_import_get_variable_as_real(variable)
            return FMIL.fmi1_import_get_real_variable_start(real_variable)
            
        elif type == FMIL.fmi1_base_type_int:
            int_variable = FMIL.fmi1_import_get_variable_as_integer(variable)
            return FMIL.fmi1_import_get_integer_variable_start(int_variable)
            
        elif type == FMIL.fmi1_base_type_bool:
            bool_variable = FMIL.fmi1_import_get_variable_as_boolean(variable)
            return FMIL.fmi1_import_get_boolean_variable_start(bool_variable) == FMITRUE
            
        elif type == FMIL.fmi1_base_type_enum:
            enum_variable = FMIL.fmi1_import_get_variable_as_enum(variable)
            return FMIL.fmi1_import_get_enum_variable_start(enum_variable)
        
        elif type == FMIL.fmi1_base_type_str:
            str_variable = FMIL.fmi1_import_get_variable_as_string(variable)
            return FMIL.fmi1_import_get_string_variable_start(str_variable)
                
        else:
            raise FMUException("Unknown variable type.")
    
    cpdef get_variable_max(self,char* variablename):
        """
        Returns the maximum value for the given variable.
        
        Parameters::
        
            variablename --
                The name of the variable.
                
        Returns::
        
            The maximum value for the variable.
        """
        cdef FMIL.fmi1_import_variable_t *variable
        cdef FMIL.fmi1_import_integer_variable_t* int_variable
        cdef FMIL.fmi1_import_real_variable_t* real_variable
        cdef FMIL.fmi1_import_enum_variable_t* enum_variable
        cdef FMIL.fmi1_base_type_enu_t type
        
        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
    
        type = FMIL.fmi1_import_get_variable_base_type(variable)
        
        if type == FMIL.fmi1_base_type_real:
            real_variable = FMIL.fmi1_import_get_variable_as_real(variable)
            return FMIL.fmi1_import_get_real_variable_max(real_variable)
            
        elif type == FMIL.fmi1_base_type_int:
            int_variable = FMIL.fmi1_import_get_variable_as_integer(variable)
            return FMIL.fmi1_import_get_integer_variable_max(int_variable)
            
        elif type == FMIL.fmi1_base_type_enum:
            enum_variable = FMIL.fmi1_import_get_variable_as_enum(variable)
            return FMIL.fmi1_import_get_enum_variable_max(enum_variable)
            
        else:
            raise FMUException("The variable type does not have a maximum value.")
    
    cpdef get_variable_min(self,char* variablename):
        """
        Returns the minimum value for the given variable.
        
        Parameters::
        
            variablename --
                The name of the variable.
                
        Returns::
        
            The minimum value for the variable.
        """
        cdef FMIL.fmi1_import_variable_t *variable
        cdef FMIL.fmi1_import_integer_variable_t* int_variable
        cdef FMIL.fmi1_import_real_variable_t* real_variable
        cdef FMIL.fmi1_import_enum_variable_t* enum_variable
        cdef FMIL.fmi1_base_type_enu_t type
        
        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
    
        type = FMIL.fmi1_import_get_variable_base_type(variable)
        
        if type == FMIL.fmi1_base_type_real:
            real_variable = FMIL.fmi1_import_get_variable_as_real(variable)
            return FMIL.fmi1_import_get_real_variable_min(real_variable)
            
        elif type == FMIL.fmi1_base_type_int:
            int_variable = FMIL.fmi1_import_get_variable_as_integer(variable)
            return FMIL.fmi1_import_get_integer_variable_min(int_variable)
            
        elif type == FMIL.fmi1_base_type_enum:
            enum_variable = FMIL.fmi1_import_get_variable_as_enum(variable)
            return FMIL.fmi1_import_get_enum_variable_min(enum_variable)
            
        else:
            raise FMUException("The variable type does not have a minimum value.")
    
    
    def save_time_point(self):
        """
        Retrieves the data at the current time-point of the variables defined
        to be continuous and the variables defined to be discrete. The 
        information about the variables are retrieved from the XML-file.
                
        Returns::
        
            sol_real -- 
                The Real-valued variables.
                
            sol_int -- 
                The Integer-valued variables.
                
            sol_bool -- 
                The Boolean-valued variables.
                
        Example::
        
            [r,i,b] = model.save_time_point()
        """
        sol_real=N.array([])
        sol_int=N.array([])
        sol_bool=N.array([])
        
        if self._save_real_variables_val:
            sol_real = self.get_real(self._save_real_variables_val)
        if self._save_int_variables_val:
            sol_int  = self.get_integer(self._save_int_variables_val)
        if self._save_bool_variables_val:  
            sol_bool = self.get_boolean(self._save_bool_variables_val)
        
        return sol_real, sol_int, sol_bool
        
    
    def get_model_variables(self,type=None, include_alias=True, 
                            causality=None,   variability=None,
                            only_start=False,  only_fixed=False):
        """
        Extract the names of the variables in a model.
        
        Parameters::
        
            type --
                The type of the variables (Real==0, Int==1, Bool=2,
                String==3, Enumeration==4). Default None (i.e all).
            include_alias --
                If alias should be included or not. Default True
            causality --
                The causality of the variables (Input==0, Output==1,
                Internal==2, None==3). Default None (i.e all).
            variability --
                The variability of the variables (Constant==0, 
                Parameter==1, Discrete==2, Continuous==3). Default None 
                (i.e. all)
            only_start --
                If only variables that has a start value should be
                returned. Default False
            only_fixed --
                If only variables that has a start value that is fixed
                should be returned. Default False
        
        Returns::
        
            Dict with variable name as key and a ScalarVariable class as
            value.
        """
        cdef FMIL.fmi1_import_variable_t *variable
        cdef FMIL.fmi1_import_variable_list_t *variable_list
        cdef FMIL.size_t variable_list_size
        cdef FMIL.fmi1_value_reference_t value_ref
        cdef FMIL.fmi1_base_type_enu_t data_type,target_type
        cdef FMIL.fmi1_variability_enu_t data_variability,target_variability
        cdef FMIL.fmi1_causality_enu_t data_causality,target_causality
        cdef FMIL.fmi1_variable_alias_kind_enu_t alias_kind
        cdef char* desc
        cdef dict variable_dict = {}
        cdef int  selected_type = 0 #If a type has been selected
        cdef int  selected_variability = 0 #If a variability has been selected
        cdef int  selected_causality = 0 #If a causality has been selected
        cdef int  has_start, is_fixed
        
        variable_list = FMIL.fmi1_import_get_variable_list(self._fmu)
        variable_list_size = FMIL.fmi1_import_get_variable_list_size(variable_list)
        
        if type!=None: #A type have has been selected
            target_type = type
            selected_type = 1
        if causality!=None: #A causality has been selected
            target_causality = causality
            selected_causality = 1
        if variability!=None: #A variability has been selected
            target_variability = variability
            selected_variability = 1
        
        for i in range(variable_list_size):
        
            variable = FMIL.fmi1_import_get_variable(variable_list, i)
            
            alias_kind = FMIL.fmi1_import_get_variable_alias_kind(variable)
            name       = FMIL.fmi1_import_get_variable_name(variable)
            value_ref  = FMIL.fmi1_import_get_variable_vr(variable)
            data_type  = FMIL.fmi1_import_get_variable_base_type(variable)
            has_start  = FMIL.fmi1_import_get_variable_has_start(variable)
            data_variability = FMIL.fmi1_import_get_variability(variable)
            data_causality   = FMIL.fmi1_import_get_causality(variable)
            desc       = FMIL.fmi1_import_get_variable_description(variable)
            
            #If only variables with start are wanted, check if the variable has start
            if only_start and has_start != 1:
                continue
            
            if only_fixed:
                if has_start!=1:
                    continue
                else:
                    is_fixed = FMIL.fmi1_import_get_variable_is_fixed(variable)
                    if is_fixed !=1:
                        continue
            
            if selected_type == 1 and data_type != target_type:
                continue
            if selected_causality == 1 and data_causality != target_causality:
                continue
            if selected_variability == 1 and data_variability != target_variability:
                continue
            
            if include_alias:
                #variable_dict[name] = value_ref
                variable_dict[name] = ScalarVariable(name, 
                                       value_ref, data_type, desc if desc!=NULL else "",
                                       data_variability, data_causality,
                                       alias_kind)
            elif alias_kind ==FMIL.fmi1_variable_is_not_alias:
                #variable_dict[name] = value_ref
                variable_dict[name] = ScalarVariable(name, 
                                       value_ref, data_type, desc if desc!=NULL else "",
                                       data_variability, data_causality,
                                       alias_kind)
        
        #Free the variable list
        FMIL.fmi1_import_free_variable_list(variable_list)
        
        return variable_dict

    
    def get_variable_alias(self,char* variablename):
        """ 
        Return a dict of all alias variables belonging to the provided variable
        where the key are the names and the value indicating whether the variable
        should be negated or not.
        
        Returns::
            
            A dict consisting of the alias variables along with no alias variable.
            The values indicates wheter or not the variable should be negated or not.
            
        Raises:: 
        
            FMUException if the variable is not in the model.
        """
        cdef FMIL.fmi1_import_variable_t *variable
        cdef FMIL.fmi1_import_variable_list_t *alias_list
        cdef FMIL.size_t alias_list_size
        cdef FMIL.fmi1_variable_alias_kind_enu_t alias_kind
        cdef dict ret_values = {}
        
        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
        
        alias_list = FMIL.fmi1_import_get_variable_aliases(self._fmu, variable)
        
        alias_list_size = FMIL.fmi1_import_get_variable_list_size(alias_list)
        
        #Loop over all the alias variables
        for i in range(alias_list_size):
                
            variable = FMIL.fmi1_import_get_variable(alias_list, i)
                
            alias_kind = FMIL.fmi1_import_get_variable_alias_kind(variable)
            alias_name = FMIL.fmi1_import_get_variable_name(variable)
            
            ret_values[alias_name] = alias_kind
        
        #FREE VARIABLE LIST
        FMIL.fmi1_import_free_variable_list(alias_list)
        
        return ret_values
        
    cpdef FMIL.fmi1_variability_enu_t get_variable_variability(self,char* variablename) except *:
        """ 
        Get variability of variable. 
            
        Parameters::
        
            variablename --
            
                The name of the variable.
                    
        Returns::
        
            The variability of the variable, CONTINUOUS(3), CONSTANT(0), 
            PARAMETER(1) or DISCRETE(2)
        """
        cdef FMIL.fmi1_import_variable_t* variable
        cdef FMIL.fmi1_variability_enu_t variability
        
        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
        variability = FMIL.fmi1_import_get_variability(variable)
        
        return variability
        
    cpdef FMIL.fmi1_causality_enu_t get_variable_causality(self, char* variablename) except *:
        """
        Get the causality of the variable.
        
        Parameters::
        
            variablename --
                The name of the variable.
                
        Returns::
        
            The variability of the variable, INPUT(0), OUTPUT(1),
            INTERNAL(2), NONE(3)
        """
        cdef FMIL.fmi1_import_variable_t* variable
        cdef FMIL.fmi1_causality_enu_t causality
        
        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
    
        causality = FMIL.fmi1_import_get_causality(variable)
        
        return causality
    
    def get_name(self):
        """ 
        Return the model name as used in the modeling environment.
        """
        return self._modelname
    
    def get_identifier(self):
        """ 
        Return the model identifier, name of binary model file and prefix in 
        the C-function names of the model. 
        """
        return self._modelid
        
    def get_author(self):
        """
        Return the name and organization of the model author.
        """
        cdef char* author
        author = FMIL.fmi1_import_get_author(self._fmu)
        return author if author != NULL else ""
        
    def get_description(self):
        """
        Return the model description.
        """
        cdef char* desc
        desc = FMIL.fmi1_import_get_description(self._fmu)
        return desc if desc != NULL else ""
        
    def get_generation_tool(self):
        """
        Return the model generation tool.
        """
        cdef char* gen
        gen = FMIL.fmi1_import_get_generation_tool(self._fmu)
        return gen if gen != NULL else ""
        
    def get_guid(self):
        """
        Return the model GUID.
        """
        guid = FMIL.fmi1_import_get_GUID(self._fmu)
        return guid
        

cdef class FMUModelCS1(FMUModelBase):
    #First step only support fmi1_fmu_kind_enu_cs_standalone
    #stepFinished not supported
    #fmiResetSlave not supported
    
    def __init__(self, fmu, path='.', enable_logging=True):
        #Call super
        FMUModelBase.__init__(self,fmu,path,enable_logging)
        
        if self._fmu_kind != FMI_CS_STANDALONE:
            raise FMUException("This class only supports FMI 1.0 for Co-simulation.")
            
        self.instantiate_slave(logging = self._enable_logging)
    
    cpdef _get_time(self):
        return self.__t
    
    cpdef _set_time(self, FMIL.fmi1_real_t t):
        self.__t = t
    
    time = property(_get_time,_set_time, doc = 
    """
    Property for accessing the current time of the simulation. Calls the 
    low-level FMI function: fmiSetTime.
    """)
    
    def __dealloc__(self):
        """
        Deallocate memory allocated
        """
        if self._allocated_fmu:
            FMIL.fmi1_import_terminate_slave(self._fmu)
            FMIL.fmi1_import_free_slave_instance(self._fmu)
        
        if self._allocated_dll:
            FMIL.fmi1_import_destroy_dllfmu(self._fmu)
            
        if self._allocated_xml:  
            FMIL.fmi1_import_free(self._fmu)
        
        if self._allocated_context:
            FMIL.fmi_import_free_context(self.context)
        
        if self._fmu_temp_dir:
            import os
            import shutil
            delete_temp_dir(self._fmu_temp_dir)
    
    def do_step(self, FMIL.fmi1_real_t current_t, FMIL.fmi1_real_t step_size, new_step=True):
        """
        Performs an integrator step.
        
        Parameters::
        
            current_t --
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
        cdef FMIL.fmi1_boolean_t new_s
        
        if new_step:
            #new_s = FMI_TRUE
            new_s = 1
        else:
            #new_s = FMI_FALSE
            new_s = 0
        
        self.time = current_t+step_size
        
        status = FMIL.fmi1_import_do_step(self._fmu, current_t, step_size, new_s)
        
        return status
        
        
    def cancel_step(self):
        """
        Cancel a current integrator step. Can only be called if the
        status from do_step returns FMI_PENDING.
        """
        raise NotImplementedError
        
    def get_output_derivatives(self, variables, FMIL.fmi1_integer_t order):
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
        cdef int status
        cdef unsigned int max_output_derivative
        cdef FMIL.size_t nref
        cdef N.ndarray[FMIL.fmi1_real_t, ndim=1,mode='c'] values
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] value_refs
        cdef N.ndarray[FMIL.fmi1_integer_t, ndim=1,mode='c'] orders
        cdef FMIL.fmi1_import_capabilities_t *fmu_capabilities
        
        fmu_capabilities = FMIL.fmi1_import_get_capabilities(self._fmu)
        max_output_derivative = FMIL.fmi1_import_get_maxOutputDerivativeOrder(fmu_capabilities)
        
        if order < 1 or order > max_output_derivative:
            raise FMUException("The order must be greater than zero and below the maximum output derivative support of the FMU (%d)."%max_output_derivative)
            
        if isinstance(variables,str):
            nref = 1
            value_refs = N.array([0], dtype=N.uint32,ndmin=1).flatten()
            orders = N.array(order, dtype=N.int32)
            value_refs[0] = self.get_variable_valueref(variables)
        elif isinstance(variables,list) and isinstance(variables[-1],str):
            nref = len(variables)
            value_refs = N.array([0]*nref, dtype=N.uint32,ndmin=1).flatten()
            orders = N.array([0]*nref, dtype=N.int32)
            for i in range(nref):
                value_refs[i] = self.get_variable_valueref(variables[i])
                orders[i] = order
        else:
            raise FMUException("The variables must either be a string or a list of strings")
            
        values = N.array([0.0]*nref,dtype=N.float, ndmin=1)
            
        status = FMIL.fmi1_import_get_real_output_derivatives(self._fmu, <FMIL.fmi1_value_reference_t*>value_refs.data, nref, <FMIL.fmi1_integer_t*>orders.data, <FMIL.fmi1_real_t*>values.data)

        if status != 0:
            raise FMUException('Failed to get the Real output derivatives.')
        
        return values
    
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
        return FMIL.fmi1_import_get_types_platform(self._fmu)
        
    types_platform = property(fget=_get_types_platform)
    
    def set_input_derivatives(self, variables, values, FMIL.fmi1_integer_t order):
        """
        Sets the input derivative order for the specified variables.
        
        Parameters::
        
                variables --
                        The variables for which the input derivative 
                        should be set.
                values --
                        The actual values.
                order --
                        The derivative order to set.
        """
        cdef int status
        cdef int can_interpolate_inputs
        cdef FMIL.size_t nref
        cdef FMIL.fmi1_import_capabilities_t *fmu_capabilities
        cdef N.ndarray[FMIL.fmi1_integer_t, ndim=1,mode='c'] orders
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] value_refs
        cdef N.ndarray[FMIL.fmi1_real_t, ndim=1,mode='c'] val = N.array(values, dtype=N.float, ndmin=1).flatten()
        
        nref = len(val)
        orders = N.array([0]*nref, dtype=N.int32)
        
        fmu_capabilities = FMIL.fmi1_import_get_capabilities(self._fmu)
        can_interpolate_inputs = FMIL.fmi1_import_get_canInterpolateInputs(fmu_capabilities)
        #NOTE IS THIS THE HIGHEST ORDER OF INTERPOLATION OR SIMPLY IF IT CAN OR NOT?
        
        if order < 1:
            raise FMUException("The order must be greater than zero.")
        if not can_interpolate_inputs:
            raise FMUException("The FMU does not support input derivatives.")
        
        if isinstance(variables,str):
            value_refs = N.array([0], dtype=N.uint32,ndmin=1).flatten()
            value_refs[0] = self.get_variable_valueref(variables)
        elif isinstance(variables,list) and isinstance(variables[-1],str):
            value_refs = N.array([0]*nref, dtype=N.uint32,ndmin=1).flatten()
            for i in range(nref):
                value_refs[i] = self.get_variable_valueref(variables[i])
                orders[i] = order
        else:
            raise FMUException("The variables must either be a string or a list of strings")
        
        status = FMIL.fmi1_import_set_real_input_derivatives(self._fmu, <FMIL.fmi1_value_reference_t*>value_refs.data, nref, <FMIL.fmi1_integer_t*>orders.data, <FMIL.fmi1_real_t*>val.data)
        
        if status != 0:
            raise FMUException('Failed to set the Real input derivatives.')
    
    def simulate(self,
                 start_time=0.0,
                 final_time=1.0,
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
                Default: 0.0
                
            final_time --
                Final time for the simulation.
                Default: 1.0
                
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
    
    def initialize(self, tStart=0.0, tStop=1.0, StopTimeDefined=False):
        """
        Initializes the slave.
        
        Parameters::
        
            tStart -
            tSTop --
            StopTimeDefined --
            
        Calls the low-level FMU function: fmiInstantiateSlave
        """
        cdef int status
        cdef FMIL.fmi1_boolean_t stopDefined
        
        if StopTimeDefined:
            #stopDefined = FMI_TRUE
            stopDefined = 1
        else:
            #stopDefined = FMI_FALSE
            stopDefined = 0
        
        self.time = tStart
        
        status = FMIL.fmi1_import_initialize_slave(self._fmu, tStart, stopDefined, tStop)
        
        if status != FMIL.fmi1_status_ok:
            raise FMUException("The slave failed to initialize.")
    
        self._allocated_fmu = True
    
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
        cdef int status
        cdef FMIL.fmi1_boolean_t log
        cdef FMIL.fmi1_real_t timeout = 0.0
        #cdef FMIL.fmi1_boolean_t visible = FMI_FALSE
        cdef FMIL.fmi1_boolean_t visible = 0
        #cdef FMIL.fmi1_boolean_t interactive = FMI_FALSE
        cdef FMIL.fmi1_boolean_t interactive = 0
        cdef object location = ""
        
        if logging:
            #log = FMI_TRUE
            log = 1
        else:
            #log = FMI_FALSE
            log = 0
        
        status = FMIL.fmi1_import_instantiate_slave(self._fmu, name, location, 
                                        FMI_MIME_CS_STANDALONE, timeout, visible, 
                                        interactive)
        
        if status != FMIL.jm_status_success:
            raise FMUException('Failed to instantiate the slave.')

        #Just to be safe, some problems with Dymola (2012) FMUs not reacting
        #to logging when set to the instantiate method.
        status = FMIL.fmi1_import_set_debug_logging(self._fmu, log)
        
        if status != 0:
            raise FMUException('Failed to set the debugging option.')
    
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
        cdef FMIL.fmi1_import_capabilities_t *cap
        
        cap = FMIL.fmi1_import_get_capabilities(self._fmu)
        
        capabilities["canHandleVariableCommunicationStepSize"] = FMIL.fmi1_import_get_canHandleVariableCommunicationStepSize(cap)
        capabilities["canHandleEvents"] = FMIL.fmi1_import_get_canHandleEvents(cap)
        capabilities["canRejectSteps"] = FMIL.fmi1_import_get_canRejectSteps(cap)
        capabilities["canInterpolateInputs"] = FMIL.fmi1_import_get_canInterpolateInputs(cap)
        capabilities["maxOutputDerivativeOrder"] = FMIL.fmi1_import_get_maxOutputDerivativeOrder(cap)
        capabilities["canRunAsynchronuously"] = FMIL.fmi1_import_get_canRunAsynchronuously(cap)
        capabilities["canSignalEvents"] = FMIL.fmi1_import_get_canSignalEvents(cap)
        capabilities["canBeInstantiatedOnlyOncePerProcess"] = FMIL.fmi1_import_get_canBeInstantiatedOnlyOncePerProcess(cap)
        capabilities["canNotUseMemoryManagementFunctions"] = FMIL.fmi1_import_get_canNotUseMemoryManagementFunctions(cap)
        
        return capabilities

cdef class FMUModelME1(FMUModelBase):
    """
    An FMI Model loaded from a DLL.
    """
    
    def __init__(self, fmu, path='.', enable_logging=True):
        #Call super
        FMUModelBase.__init__(self,fmu,path,enable_logging)
        
        if self._fmu_kind != FMI_ME:
            raise FMUException("This class only supports FMI 1.0 for Model Exchange.")
            
        self.instantiate_model(logging = self._enable_logging)
    
    def _get_model_types_platform(self):
        """
        Returns the set of valid compatible platforms for the Model, extracted
        from the XML.
        
        Returns::
            
            model_types_platform -- 
                The valid platforms.
                
        Example::
        
            model.model_types_platform
        """
        return FMIL.fmi1_import_get_model_types_platform(self._fmu)
        
    model_types_platform = property(fget=_get_model_types_platform)
    
    def reset(self):
        """
        This metod resets the FMU by first calling fmiTerminate and 
        fmiFreeModelInstance and then reloades the DLL and finally
        reinstantiates using fmiInstantiateModel.
        """
        if self._allocated_fmu:
            FMIL.fmi1_import_terminate(self._fmu)
            FMIL.fmi1_import_free_model_instance(self._fmu)
        
        if self._allocated_dll:
            FMIL.fmi1_import_destroy_dllfmu(self._fmu)
        
        global FMI_REGISTER_GLOBALLY
        status = FMIL.fmi1_import_create_dllfmu(self._fmu, self.callBackFunctions, FMI_REGISTER_GLOBALLY);
        if status == FMIL.jm_status_error:
            raise FMUException("The DLL could not be reloaded, check the log for more information.")
        FMI_REGISTER_GLOBALLY += 1 #Update the global register of FMUs
        
        #Default values
        self.__t = None
        
        #Internal values
        self._file_open = False
        self._npoints = 0
        self._log = []
        
        #Instantiates the model
        self.instantiate_model(logging = self._enable_logging)

    
    def __dealloc__(self):
        """
        Deallocate memory allocated
        """
        if self._allocated_fmu:
            FMIL.fmi1_import_terminate(self._fmu)
            FMIL.fmi1_import_free_model_instance(self._fmu)
        
        if self._allocated_dll:
            FMIL.fmi1_import_destroy_dllfmu(self._fmu)
            
        if self._allocated_xml:  
            FMIL.fmi1_import_free(self._fmu)
        
        if self._allocated_context:
            FMIL.fmi_import_free_context(self.context)
        
        if self._fmu_temp_dir:
            import os
            import shutil
            delete_temp_dir(self._fmu_temp_dir)
    
    cpdef _get_time(self):
        return self.__t
    
    cpdef _set_time(self, FMIL.fmi1_real_t t):
        cdef int status
        self.__t = t
        
        status = FMIL.fmi1_import_set_time(self._fmu,t)
        
        if status != 0:
            raise FMUException('Failed to set the time.')
        
    time = property(_get_time,_set_time, doc = 
    """
    Property for accessing the current time of the simulation. Calls the 
    low-level FMI function: fmiSetTime.
    """)
    
    def _get_continuous_states(self):
        cdef int status
        cdef N.ndarray[double, ndim=1,mode='c'] ndx = N.zeros(self._nContinuousStates, dtype=N.double)
        status = FMIL.fmi1_import_get_continuous_states(self._fmu, <FMIL.fmi1_real_t*>ndx.data ,self._nContinuousStates)
        
        if status != 0:
            raise FMUException('Failed to retrieve the continuous states.')
        
        return ndx
        
    def _set_continuous_states(self, N.ndarray[FMIL.fmi1_real_t] values):
        cdef int status
        cdef N.ndarray[FMIL.fmi1_real_t, ndim=1,mode='c'] ndx = values#N.array(values,dtype=N.double,ndmin=1).flatten()
        
        if ndx.size != self._nContinuousStates:
            raise FMUException(
                'Failed to set the new continuous states. ' \
                'The number of values are not consistent with the number of '\
                'continuous states.')
        
        status = FMIL.fmi1_import_set_continuous_states(self._fmu, <FMIL.fmi1_real_t*>ndx.data ,self._nContinuousStates)
        
        if status >= 3:
            raise FMUException('Failed to set the new continuous states.')
    
    continuous_states = property(_get_continuous_states, _set_continuous_states, 
        doc=
    """
    Property for accessing the current values of the continuous states. Calls 
    the low-level FMI function: fmiSetContinuousStates/fmiGetContinuousStates.
    """)
    
    def _get_nominal_continuous_states(self):
        cdef int status
        cdef N.ndarray[FMIL.fmi1_real_t, ndim=1,mode='c'] ndx = N.zeros(self._nContinuousStates,dtype=N.double)
        
        status = FMIL.fmi1_import_get_nominal_continuous_states(
                self._fmu, <FMIL.fmi1_real_t*>ndx.data, self._nContinuousStates)
        
        if status != 0:
            raise FMUException('Failed to get the nominal values.')
            
        return ndx
    
    nominal_continuous_states = property(_get_nominal_continuous_states, doc = 
    """
    Property for accessing the nominal values of the continuous states. Calls 
    the low-level FMI function: fmiGetNominalContinuousStates.
    """)
    
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
        cdef N.ndarray[FMIL.fmi1_real_t, ndim=1,mode='c'] values = N.empty(self._nContinuousStates,dtype=N.double)

        status = FMIL.fmi1_import_get_derivatives(self._fmu, <FMIL.fmi1_real_t*>values.data, self._nContinuousStates)
        
        if status != 0:
            raise FMUException('Failed to get the derivative values.')
            
        return values
        
    def get_event_indicators(self):
        """
        Returns the event indicators at the current time-point.

        Return::
        
            evInd -- 
                The event indicators as an array.
                
        Example::
        
            evInd = model.get_event_indicators()
                
        Calls the low-level FMI function: fmiGetEventIndicators
        """
        cdef int status
        cdef N.ndarray[FMIL.fmi1_real_t, ndim=1,mode='c'] values = N.empty(self._nEventIndicators,dtype=N.double)
        
        status = FMIL.fmi1_import_get_event_indicators(self._fmu, <FMIL.fmi1_real_t*>values.data, self._nEventIndicators)
        
        if status != 0:
            raise FMUException('Failed to get the event indicators.')
            
        return values

    
    def get_tolerances(self):
        """
        Returns the relative and absolute tolerances. If the relative tolerance
        is defined in the XML-file it is used, otherwise a default of 1.e-4 is 
        used. The absolute tolerance is calculated and returned according to
        the FMI specification, atol = 0.01*rtol*(nominal values of the 
        continuous states).
                
        Returns::
        
            rtol -- 
                The relative tolerance.
                
            atol -- 
                The absolute tolerance.
                
        Example::
            
            [rtol, atol] = model.get_tolerances()
        """
        rtol = FMIL.fmi1_import_get_default_experiment_tolerance(self._fmu)
        atol = 0.01*rtol*self.nominal_continuous_states
        
        return [rtol, atol]
    
    def event_update(self, intermediateResult=False):
        """
        Updates the event information at the current time-point. If 
        intermediateResult is set to True the update_event will stop at each 
        event iteration which would require to loop until 
        event_info.iterationConverged == fmiTrue.
        
        Parameters::
        
            intermediateResult -- 
                If set to True, the update_event will stop at each event 
                iteration.
                Default: False.
                
        Example::
        
            model.event_update()
        
        Calls the low-level FMI function: fmiEventUpdate
        """
        cdef int status
        cdef FMIL.fmi1_boolean_t intermediate_result
        
        if intermediateResult:
            intermediate_result = 1
            status = FMIL.fmi1_import_eventUpdate(self._fmu, intermediate_result, &self._eventInfo)
        else:
            intermediate_result = 0
            status = FMIL.fmi1_import_eventUpdate(self._fmu, intermediate_result, &self._eventInfo)
        
        if status != 0:
            raise FMUException('Failed to update the events.')
    
    def get_event_info(self):
        """
        Returns the event information from the FMU. 
        
        Returns::
        
            The event information, a struct which contains:
        
            iterationConverged -- 
                Event iteration converged (if True).
                
            stateValueReferencesChanged -- 
                ValueReferences of states x changed (if True).
                
            stateValuesChanged -- 
                Values of states x have changed (if True).
                
            terminateSimulation -- 
                Error, terminate simulation (if True).
                
            upcomingTimeEvent - 
                If True, nextEventTime is the next time event.
                
            nextEventTime -- 
                The next time event.
                
        Example::
            
            event_info    = model.event_info
            nextEventTime = model.event_info.nextEventTime
        """
        
        self._pyEventInfo.iterationConverged          = self._eventInfo.iterationConverged == 1
        self._pyEventInfo.stateValueReferencesChanged = self._eventInfo.stateValueReferencesChanged == 1
        self._pyEventInfo.stateValuesChanged          = self._eventInfo.stateValuesChanged == 1
        self._pyEventInfo.terminateSimulation         = self._eventInfo.terminateSimulation == 1
        self._pyEventInfo.upcomingTimeEvent           = self._eventInfo.upcomingTimeEvent == 1
        self._pyEventInfo.nextEventTime               = self._eventInfo.nextEventTime

        return self._pyEventInfo
    
    def get_state_value_references(self):
        """
        Returns the continuous states valuereferences.
                
        Returns::
            
            val -- 
                The references to the continuous states.
                
        Example::
        
            val = model.get_continuous_value_reference()
            
        Calls the low-level FMI function: fmiGetStateValueReferences
        """
        cdef int status
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] values = N.zeros(self._nContinuousStates,dtype=N.uint32)

        status = FMIL.fmi1_import_get_state_value_references(
            self._fmu, <FMIL.fmi1_value_reference_t*>values.data, self._nContinuousStates)
        
        if status != 0:
            raise FMUException(
                'Failed to get the continuous state reference values.')
            
        return values
    
    def completed_integrator_step(self):
        """
        This method must be called by the environment after every completed step 
        of the integrator. If the return is True, then the environment must call 
        event_update() otherwise, no action is needed.
        
        Returns::
        
            True -> Call event_update().
            False -> Do nothing.
                
        Calls the low-level FMI function: fmiCompletedIntegratorStep.
        """
        cdef int status
        cdef FMIL.fmi1_boolean_t callEventUpdate
        
        status = FMIL.fmi1_import_completed_integrator_step(self._fmu, &callEventUpdate)
        
        if status != 0:
            raise FMUException('Failed to call FMI Completed Step.')
            
        if callEventUpdate == 1:
            return True
        else:
            return False

    
    def initialize(self, tolControlled=True, relativeTolerance=None):
        """
        Initializes the model and computes initial values for all variables, 
        including setting the start values of variables defined with a the start 
        attribute in the XML-file. 
            
        Parameters::
        
            tolControlled -- 
                If the model are going to be called by numerical solver using
                step-size control. Boolean flag.
            relativeTolerance --
                If the model are controlled by a numerical solver using
                step-size control, the same tolerance should be provided here.
                Else the default tolerance from the XML-file are used.
            
        Calls the low-level FMI function: fmiInitialize.
        """
        cdef char tolerance_controlled
        cdef FMIL.fmi1_real_t tolerance
        
        #Trying to set the initial time from the xml file, else 0.0
        if self.time == None:
            self.time = FMIL.fmi1_import_get_default_experiment_start(self._fmu)
            
        if tolControlled:
            tolerance_controlled = 1
            if relativeTolerance == None:
                tolerance = FMIL.fmi1_import_get_default_experiment_tolerance(self._fmu)
            else:
                tolerance = relativeTolerance
        else:
            tolerance_controlled = 0
            tolerance = 0.0
        
        status = FMIL.fmi1_import_initialize(self._fmu, tolerance_controlled, tolerance, &self._eventInfo)
        
        if status == 1:
            if self._enable_logging:
                logging.warning(
                    'Initialize returned with a warning.' \
                    ' Check the log for information (FMUModel.get_log).')
            else:
                logging.warning('Initialize returned with a warning.' \
                    ' Enable logging for more information, (FMUModel(..., enable_logging=True)).')
        
        if status > 1:
            if self._enable_logging:
                raise FMUException(
                    'Initialize returned with a error.' \
                    ' Check the log for information (FMUModel.get_log).')
            else:
                raise FMUException('Initialize returned with a error.' \
                    ' Enable logging for more information, (FMUModel(..., enable_logging=True)).')
        
        self._allocated_fmu = True
		
    
    def instantiate_model(self, name='Model', logging=False):
        """
        Instantiate the model.
        
        Parameters::
        
            name -- 
                The name of the instance.
                Default: 'Model'
                        
            logging -- 
                Defines if the logging should be turned on or off.
                Default: False, no logging.
                        
        Calls the low-level FMI function: fmiInstantiateModel.
        """
        #cdef FMIL.fmi1_string_t guid
        cdef FMIL.fmi1_boolean_t log
        cdef int status
        
        #guid = FMIL.fmi1_import_get_GUID(self._fmu)
        
        if logging:
            log = 1
        else:
            log = 0
        
        #status = FMIL.fmi1_import_instantiate_model(self._fmu, name, guid, log)
        status = FMIL.fmi1_import_instantiate_model(self._fmu, name)
        
        if status != 0:
            raise FMUException('Failed to instantiate the model.')
        
        #Just to be safe, some problems with Dymola (2012) FMUs not reacting
        #to logging when set to the instantiate method.
        status = FMIL.fmi1_import_set_debug_logging(self._fmu, log)
        
        if status != 0:
            raise FMUException('Failed to set the debugging option.')
    
    def simulate(self,
                 start_time=0.0,
                 final_time=1.0,
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
                Default: 0.0
                
            final_time --
                Final time for the simulation.
                Default: 1.0
                
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
        return self._exec_simulate_algorithm(start_time, 
                                             final_time, 
                                             input,
                                             'pyfmi.fmi_algorithm_drivers', 
                                             algorithm,
                                             options)
                               
    def simulate_options(self, algorithm='AssimuloFMIAlg'):
        """ 
        Get an instance of the simulate options class, prefilled with default 
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
    

#Temporary should be removed! (after a period)
cdef class FMUModel(FMUModelME1):
    def __init__(self, fmu, path='.', enable_logging=True):
        print "WARNING: This class is deprecated and has been superseded with FMUModelME1. The recommended entry-point for loading an FMU is now the function load_fmu."
        FMUModelME1.__init__(self,fmu,path,enable_logging)

def load_fmu(fmu, path='.', enable_logging=True):
    """
    Helper function for loading FMUs of different kinds.
    """
    #NOTE: This method can be made more efficient by providing
    #the unzipped part and the already read XML object to the different
    #FMU classes.
    
    #FMIL related variables
    cdef FMIL.fmi1_callback_functions_t callBackFunctions
    cdef FMIL.jm_callbacks callbacks
    cdef FMIL.fmi_import_context_t* context 
    cdef FMIL.fmi1_import_t* _fmu
    cdef FMIL.jm_string last_error
    
    #Used for deallocation
    allocated_context = False
    allocated_xml = False
    fmu_temp_dir = None
    
    fmu_full_path = os.path.abspath(os.path.join(path,fmu))
    fmu_temp_dir  = create_temp_dir()

    # Check that the file referenced by fmu has the correct file-ending
    if not fmu_full_path.endswith(".fmu"):
        raise FMUException("FMUModel must be instantiated with an FMU (.fmu) file.")
    
    #Specify the general callback functions
    callbacks.malloc  = FMIL.malloc
    callbacks.calloc  = FMIL.calloc
    callbacks.realloc = FMIL.realloc
    callbacks.free    = FMIL.free
    callbacks.logger  = importlogger_load_fmu
    callbacks.log_level = FMIL.jm_log_level_warning if enable_logging else FMIL.jm_log_level_nothing
    #callbacks.errMessageBuffer = NULL
        
    #Specify FMI related callbacks
    callBackFunctions.logger = FMIL.fmi1_log_forwarding;
    callBackFunctions.allocateMemory = FMIL.calloc;
    callBackFunctions.freeMemory = FMIL.free;
        
    context = FMIL.fmi_import_allocate_context(&callbacks)
    allocated_context = True
        
    #Get the FMI version of the provided model
    version = FMIL.fmi_import_get_fmi_version(context, fmu_full_path, fmu_temp_dir)
        
    if version == FMIL.fmi_version_unknown_enu:
        last_error = FMIL.jm_get_last_error(&callbacks)

        #Delete the context
        if allocated_context:
            FMIL.fmi_import_free_context(context)
        
        raise FMUException("The FMU version could not be determined. "+last_error)
    if version != 1:
        #Delete the context
        if allocated_context:
            FMIL.fmi_import_free_context(context)
            
        raise FMUException("PyFMI currently only supports FMI 1.0.")
        
    #Parse the XML
    _fmu = FMIL.fmi1_import_parse_xml(context, fmu_temp_dir)
    allocated_xml = True
        
    #Check the FMU kind
    fmu_kind = FMIL.fmi1_import_get_fmu_kind(_fmu)
    if fmu_kind == FMI_ME:
        model = FMUModelME1(fmu, path, enable_logging)
    elif fmu_kind == FMI_CS_STANDALONE:
        model = FMUModelCS1(fmu, path, enable_logging)
    else:
        #Delete the XML
        if allocated_xml:  
            FMIL.fmi1_import_free(_fmu)
            
        #Delete the context
        if allocated_context:
            FMIL.fmi_import_free_context(context)
        
        raise FMUException("PyFMI currently only supports FMI 1.0 for Model Exchange.")    
        
    #Delete the XML
    if allocated_xml:  
        FMIL.fmi1_import_free(_fmu)
        
    #Delete the context
    if allocated_context:
        FMIL.fmi_import_free_context(context)

    #Delete the created directory
    delete_temp_dir(fmu_temp_dir)

    return model
