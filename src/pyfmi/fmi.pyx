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
import fnmatch
import re
from collections import OrderedDict

import numpy as N
cimport numpy as N

N.import_array()

cimport fmil_import as FMIL

from pyfmi.common.core import create_temp_dir, delete_temp_dir
#from pyfmi.common.core cimport BaseModel

int   = N.int32
N.int = N.int32

"""Basic flags related to FMI"""

FMI_TRUE  = '\x01'
FMI_FALSE = '\x00'

# Status
FMI_OK      = FMIL.fmi1_status_ok
FMI_WARNING = FMIL.fmi1_status_warning
FMI_DISCARD = FMIL.fmi1_status_discard
FMI_ERROR   = FMIL.fmi1_status_error
FMI_FATAL   = FMIL.fmi1_status_fatal
FMI_PENDING = FMIL.fmi1_status_pending

# Types
FMI_REAL        = FMIL.fmi1_base_type_real
FMI_INTEGER     = FMIL.fmi1_base_type_int
FMI_BOOLEAN     = FMIL.fmi1_base_type_bool
FMI_STRING      = FMIL.fmi1_base_type_str
FMI_ENUMERATION = FMIL.fmi1_base_type_enum

FMI2_REAL        = FMIL.fmi2_base_type_real
FMI2_INTEGER     = FMIL.fmi2_base_type_int
FMI2_BOOLEAN     = FMIL.fmi2_base_type_bool
FMI2_STRING      = FMIL.fmi2_base_type_str
FMI2_ENUMERATION = FMIL.fmi2_base_type_enum

# Alias data
FMI_NO_ALIAS      = FMIL.fmi1_variable_is_not_alias
FMI_ALIAS         = FMIL.fmi1_variable_is_alias
FMI_NEGATED_ALIAS = FMIL.fmi1_variable_is_negated_alias

# Variability
FMI_CONTINUOUS = FMIL.fmi1_variability_enu_continuous
FMI_CONSTANT   = FMIL.fmi1_variability_enu_constant
FMI_PARAMETER  = FMIL.fmi1_variability_enu_parameter
FMI_DISCRETE   = FMIL.fmi1_variability_enu_discrete

FMI2_CONSTANT   = FMIL.fmi2_variability_enu_constant
FMI2_FIXED      = FMIL.fmi2_variability_enu_fixed
FMI2_TUNABLE    = FMIL.fmi2_variability_enu_tunable
FMI2_DISCRETE   = FMIL.fmi2_variability_enu_discrete
FMI2_CONTINUOUS = FMIL.fmi2_variability_enu_continuous
FMI2_UNKNOWN    = FMIL.fmi2_variability_enu_unknown

# Causality
FMI_INPUT    = FMIL.fmi1_causality_enu_input
FMI_OUTPUT   = FMIL.fmi1_causality_enu_output
FMI_INTERNAL = FMIL.fmi1_causality_enu_internal
FMI_NONE     = FMIL.fmi1_causality_enu_none

# FMI types
FMI_ME                 = FMIL.fmi1_fmu_kind_enu_me
FMI_CS_STANDALONE      = FMIL.fmi1_fmu_kind_enu_cs_standalone
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
    if c.context != NULL:
        (<FMUModelBase>c.context)._logger(module,log_level,message)

#CALLBACKS
cdef void importlogger2(FMIL.jm_callbacks* c, FMIL.jm_string module, int log_level, FMIL.jm_string message):
    #print "FMIL: module = %s, log lovel = %d: %s" %(module, log_level, message)
    if c.context != NULL:
        (<FMUModelBase2>c.context)._logger(module, log_level, message)

#CALLBACKS
cdef void importlogger_load_fmu(FMIL.jm_callbacks* c, FMIL.jm_string module, int log_level, FMIL.jm_string message):
    if log_level <= c.log_level:
        print "FMIL: module = %s, log level = %d: %s"%(module, log_level, message)
    #(<FMUModelBase>c.context)._logger(module,log_level,message)

#Old, use FMIL.fmi#_log_forwarding instead
cdef void fmilogger(FMIL.fmi1_component_t c, FMIL.fmi1_string_t instanceName, FMIL.fmi1_status_t status, FMIL.fmi1_string_t category, FMIL.fmi1_string_t message, ...):
    cdef char buf[1000]
    cdef FMIL.va_list args
    FMIL.va_start(args, message)
    FMIL.vsnprintf(buf, 1000, message, args)
    FMIL.va_end(args)
    print "FMU: fmiStatus = %d;  %s (%s): %s\n"%(status, instanceName, category, buf)

cdef class ModelBase:
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

cdef class ScalarVariable2:
    """
    Class defining data structure based on the XML element ScalarVariable.
    """

    cdef FMIL.fmi2_value_reference_t         _value_reference
    cdef FMIL.fmi2_base_type_enu_t           _type
    cdef FMIL.fmi2_variability_enu_t         _variability
    cdef FMIL.fmi2_causality_enu_t           _causality
    cdef FMIL.fmi2_variable_alias_kind_enu_t _alias
    cdef object _name
    cdef object _description #A characater pointer but we need an own reference and this is sufficient

    def __init__(self, name, value_reference, type, description = "",
                       variability = FMIL.fmi2_variability_enu_unknown,
                       causality   = FMIL.fmi2_causality_enu_unknown,
                       alias       = FMIL.fmi2_variable_is_not_alias):

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

cdef class FMUState2:
    """
    Class containing a pointer to a FMU-state.
    """

    cdef FMIL.fmi2_FMU_state_t* fmu_state

    def __init__(self):
        self.fmu_state = NULL



cdef class FMUModelBase(ModelBase):
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
    cdef int _allocated_dll, _allocated_context, _allocated_xml, _allocated_fmu
    cdef object _allocated_list
    cdef object _modelid
    cdef object _modelname
    cdef unsigned int _nEventIndicators
    cdef unsigned int _nContinuousStates
    cdef public list _save_real_variables_val
    cdef public list _save_int_variables_val
    cdef public list _save_bool_variables_val
    cdef int _fmu_kind
    cdef char* _fmu_log_name
    cdef char* _fmu_temp_dir

    def __init__(self, fmu, path='.', enable_logging=True, log_file_name=""):
        """
        Constructor.
        """
        cdef int status
        cdef int version

        #Contains the log information
        self._log = []

        #Used for deallocation
        self._allocated_context = 0
        self._allocated_dll = 0
        self._allocated_xml = 0
        self._allocated_fmu = 0
        self._allocated_list = False
        self._fmu_temp_dir = NULL
        self._fmu_log_name = NULL

        #Specify the general callback functions
        self.callbacks.malloc  = FMIL.malloc
        self.callbacks.calloc  = FMIL.calloc
        self.callbacks.realloc = FMIL.realloc
        self.callbacks.free    = FMIL.free
        self.callbacks.logger  = importlogger
        self.callbacks.context = <void*>self #Class loggger
        self.callbacks.log_level = FMIL.jm_log_level_info if enable_logging else FMIL.jm_log_level_error

        fmu_full_path = os.path.abspath(os.path.join(path,fmu))
        fmu_temp_dir  = create_temp_dir()
        self._fmu_temp_dir = <char*>FMIL.malloc((FMIL.strlen(fmu_temp_dir)+1)*sizeof(char))
        FMIL.strcpy(self._fmu_temp_dir, fmu_temp_dir)

        # Check that the file referenced by fmu has the correct file-ending
        if not fmu_full_path.endswith(".fmu"):
            raise FMUException("FMUModel must be instantiated with an FMU (.fmu) file.")

        #Specify FMI related callbacks
        self.callBackFunctions.logger = FMIL.fmi1_log_forwarding;
        self.callBackFunctions.allocateMemory = FMIL.calloc;
        self.callBackFunctions.freeMemory = FMIL.free;
        self.callBackFunctions.stepFinished = NULL;

        self.context = FMIL.fmi_import_allocate_context(&self.callbacks)
        self._allocated_context = 1

        #Get the FMI version of the provided model
        version = FMIL.fmi_import_get_fmi_version(self.context, fmu_full_path, fmu_temp_dir)
        self._version = version #Store version

        if version == FMIL.fmi_version_unknown_enu:
            last_error = FMIL.jm_get_last_error(&self.callbacks)
            if enable_logging:
                raise FMUException("The FMU version could not be determined. "+last_error)
            else:
                raise FMUException("The FMU version could not be determined. Enable logging for possibly more information.")
        if version != 1:
            raise FMUException("This class only supports FMI 1.0 (Model Exchange and Co-Simulation).")

        #Parse the XML
        self._fmu = FMIL.fmi1_import_parse_xml(self.context, fmu_temp_dir)
        if self._fmu == NULL:
            last_error = FMIL.jm_get_last_error(&self.callbacks)
            if enable_logging:
                raise FMUException("The XML file could not be parsed. "+last_error)
            else:
                raise FMUException("The XML file could not be parsed. Enable logging for possibly more information.")
        self._allocated_xml = 1

        #Check the FMU kind
        fmu_kind = FMIL.fmi1_import_get_fmu_kind(self._fmu)
        if fmu_kind != FMI_ME and fmu_kind != FMI_CS_STANDALONE:
            raise FMUException("This class only supports FMI 1.0 (Model Exchange and Co-Simulation).")
        self._fmu_kind = fmu_kind

        #Connect the DLL
        global FMI_REGISTER_GLOBALLY
        status = FMIL.fmi1_import_create_dllfmu(self._fmu, self.callBackFunctions, FMI_REGISTER_GLOBALLY);
        if status == FMIL.jm_status_error:
            last_error = FMIL.fmi1_import_get_last_error(self._fmu)
            if enable_logging:
                raise FMUException(last_error)
            else:
                raise FMUException("Error loading the binary. Enable logging for possibly more information.")
        self._allocated_dll = 1
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
        fmu_log_name = (self._modelid + "_log.txt") if log_file_name=="" else log_file_name
        self._fmu_log_name = <char*>FMIL.malloc((FMIL.strlen(fmu_log_name)+1)*sizeof(char))
        FMIL.strcpy(self._fmu_log_name, fmu_log_name)

        #Create the log file
        with open(self._fmu_log_name,'w') as file:
            for i in range(len(self._log)):
                file.write("FMIL: module = %s, log level = %d: %s\n"%(self._log[i][0], self._log[i][1], self._log[i][2]))
            self._log = []


    cdef _logger(self, FMIL.jm_string module, int log_level, FMIL.jm_string message):
        if self._fmu_log_name != NULL:
            with open(self._fmu_log_name,'a') as file:
                file.write("FMIL: module = %s, log level = %d: %s\n"%(module, log_level, message))
        else:
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
        log = []
        if self._fmu_log_name != NULL:
            with open(self._fmu_log_name,'r') as file:
                while True:
                    line = file.readline()
                    if line == "":
                        break
                    log.append(line.strip("\n"))
        return log

    cpdef _internal_set_fmu_null(self):
        """
        This methods is ONLY for testing puposes. It sets the internal
        fmu state to NULL
        """
        self._fmu = NULL

    def print_log(self):
        """
        Prints the log information to the prompt.
        """
        cdef int N
        log = self.get_log()
        N = len(log)

        for i in range(N):
            print log[i]
            #print "FMIL: module = %s, log level = %d: %s"%(log[i][0], log[i][1], log[i][2])

    def _convert_filter(self, expression):
        """
        Convert a filter based on unix filename pattern matching to a
        list of regular expressions.
        """
        regexp = []
        if isinstance(expression,str):
            regex = fnmatch.translate(expression)
            regexp = [re.compile(regex)]
        elif isinstance(expression,list):
            for i in expression:
                regex = fnmatch.translate(i)
                regexp.append(re.compile(regex))
        else:
            raise FMUException("Unknown input.")
        return regexp

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
        cdef FMIL.fmi1_import_variable_t* variable
        cdef FMIL.fmi1_variable_alias_kind_enu_t alias_kind

        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variable_name)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variable_name)

        ref =  FMIL.fmi1_import_get_variable_vr(variable)
        type = FMIL.fmi1_import_get_variable_base_type(variable)
        alias_kind = FMIL.fmi1_import_get_variable_alias_kind(variable)

        if type == FMIL.fmi1_base_type_real:  #REAL
            if alias_kind == FMI_NEGATED_ALIAS:
                value = -value
            self.set_real([ref], [value])
        elif type == FMIL.fmi1_base_type_int: #INTEGER
            if alias_kind == FMI_NEGATED_ALIAS:
                value = -value
            self.set_integer([ref], [value])
        elif type == FMIL.fmi1_base_type_str: #STRING
            self.set_string([ref], [value])
        elif type == FMIL.fmi1_base_type_bool: #BOOLEAN
            if alias_kind == FMI_NEGATED_ALIAS:
                value = not value
            self.set_boolean([ref], [value])
        else:
            raise FMUException('Type not supported.')


    def _get(self,char* variable_name):
        """
        Helper method to get, see docstring on get.
        """
        cdef FMIL.fmi1_value_reference_t ref
        cdef FMIL.fmi1_base_type_enu_t type
        cdef FMIL.fmi1_import_variable_t* variable
        cdef FMIL.fmi1_variable_alias_kind_enu_t alias_kind

        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variable_name)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variable_name)

        ref =  FMIL.fmi1_import_get_variable_vr(variable)
        type = FMIL.fmi1_import_get_variable_base_type(variable)
        alias_kind = FMIL.fmi1_import_get_variable_alias_kind(variable)

        if type == FMIL.fmi1_base_type_real:  #REAL
            value = self.get_real([ref])
            return -1*value if alias_kind == FMI_NEGATED_ALIAS else value
        elif type == FMIL.fmi1_base_type_int: #INTEGER
            value = self.get_integer([ref])
            return -1*value if alias_kind == FMI_NEGATED_ALIAS else value
        elif type == FMIL.fmi1_base_type_str: #STRING
            return self.get_string([ref])
        elif type == FMIL.fmi1_base_type_bool: #BOOLEAN
            value = self.get_boolean([ref])
            return not value if alias_kind == FMI_NEGATED_ALIAS else value
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


    def get_model_variables(self,type=None, include_alias=True,
                            causality=None,   variability=None,
                            only_start=False,  only_fixed=False,
                            filter=None):
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
            filter --
                Filter the variables using a unix filename pattern
                matching (filter="*der*"). Can also be a list of filters
                See http://docs.python.org/2/library/fnmatch.html.
                Default None

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
        cdef list filter_list = []
        cdef int  selected_type = 0 #If a type has been selected
        cdef int  selected_variability = 0 #If a variability has been selected
        cdef int  selected_causality = 0 #If a causality has been selected
        cdef int  has_start, is_fixed
        cdef int  selected_filter = 1 if filter else 0
        cdef int  length_filter = 0

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
        if selected_filter:
            filter_list = self._convert_filter(filter)
            length_filter = len(filter_list)

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

            if selected_filter:
                for j in range(length_filter):
                    if re.match(filter_list[j], name):
                        break
                else:
                    continue

            if include_alias:
                variable_dict[name] = ScalarVariable(name,
                                       value_ref, data_type, desc.decode('UTF-8') if desc!=NULL else "",
                                       data_variability, data_causality,
                                       alias_kind)

            elif alias_kind ==FMIL.fmi1_variable_is_not_alias:
                variable_dict[name] = ScalarVariable(name,
                                       value_ref, data_type, desc.decode('UTF-8') if desc!=NULL else "",
                                       data_variability, data_causality,
                                       alias_kind)

        #Free the variable list
        FMIL.fmi1_import_free_variable_list(variable_list)

        return variable_dict

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
        cdef FMIL.fmi1_import_variable_t *variable, *base_variable
        cdef FMIL.fmi1_import_variable_list_t *variable_list
        cdef FMIL.size_t variable_list_size
        cdef FMIL.fmi1_value_reference_t value_ref
        cdef FMIL.fmi1_base_type_enu_t data_type
        cdef FMIL.fmi1_variability_enu_t data_variability
        cdef FMIL.fmi1_variable_alias_kind_enu_t alias_kind
        cdef dict variable_dict = {}
        cdef list filter_list = []
        cdef dict real_var_ref = {}
        cdef dict int_var_ref = {}
        cdef dict bool_var_ref = {}
        cdef int  selected_variability = 0 #If a variability has been selected
        cdef int  selected_filter = 1 if filter else 0
        cdef int  length_filter = 0

        variable_list = FMIL.fmi1_import_get_variable_list(self._fmu)
        variable_list_size = FMIL.fmi1_import_get_variable_list_size(variable_list)

        if selected_filter:
            filter_list = self._convert_filter(filter)
            length_filter = len(filter_list)

        for i in range(variable_list_size):

            variable = FMIL.fmi1_import_get_variable(variable_list, i)

            alias_kind = FMIL.fmi1_import_get_variable_alias_kind(variable)
            name       = FMIL.fmi1_import_get_variable_name(variable)
            data_variability = FMIL.fmi1_import_get_variability(variable)
            data_type  = FMIL.fmi1_import_get_variable_base_type(variable)

            if data_type != FMI_REAL and data_type != FMI_INTEGER and data_type != FMI_BOOLEAN:
                continue

            if data_variability != FMI_CONTINUOUS and data_variability != FMI_DISCRETE:
                continue

            if selected_filter:
                for j in range(length_filter):
                    if re.match(filter_list[j], name):
                        break
                else:
                    continue
            else:
                if alias_kind != FMIL.fmi1_variable_is_not_alias:
                    continue

            if alias_kind == FMIL.fmi1_variable_is_not_alias:
                value_ref = FMIL.fmi1_import_get_variable_vr(variable)
            else:
                base_variable = FMIL.fmi1_import_get_variable_alias_base(self._fmu, variable)
                value_ref  = FMIL.fmi1_import_get_variable_vr(base_variable)

            if data_type == FMI_REAL:
                real_var_ref[value_ref] = 1
            if data_type == FMI_INTEGER:
                int_var_ref[value_ref] = 1
            if data_type == FMI_BOOLEAN:
                bool_var_ref[value_ref] = 1

        #Free the variable list
        FMIL.fmi1_import_free_variable_list(variable_list)

        return real_var_ref.keys(), int_var_ref.keys(), bool_var_ref.keys()

    def get_variable_alias_base(self, char* variablename):
        """
        Returns the base variable for the provided variable name.
        """
        cdef FMIL.fmi1_import_variable_t* variable, *base_variable
        cdef FMIL.fmi1_value_reference_t vr

        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        base_variable = FMIL.fmi1_import_get_variable_alias_base(self._fmu, variable)
        if base_variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        name = FMIL.fmi1_import_get_variable_name(base_variable)

        return name

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

    def get_variable_by_valueref(self, FMIL.fmi1_value_reference_t valueref, type=0):
        """
        Get the name of a variable given a value reference. Note that it
        returns the no-aliased variable.

        Parameters::

            valueref --
                The value reference of the variable

            type --
                The type of the variables (Real==0, Int==1, Bool=2,
                String==3, Enumeration==4). Default 0 (i.e Real).
        """
        cdef FMIL.fmi1_import_variable_t* variable

        variable = FMIL.fmi1_import_get_variable_by_vr(self._fmu, type, valueref)
        if variable==NULL:
            raise FMUException("The variable with the valuref %i could not be found."%valueref)

        name = FMIL.fmi1_import_get_variable_name(variable)

        return name

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

    def get_default_experiment_start_time(self):
        """
        Returns the default experiment start time as defined the XML
        description.
        """
        return FMIL.fmi1_import_get_default_experiment_start(self._fmu)

    def get_default_experiment_stop_time(self):
        """
        Returns the default experiment stop time as defined the XML
        description.
        """
        return FMIL.fmi1_import_get_default_experiment_stop(self._fmu)

    def get_default_experiment_tolerance(self):
        """
        Returns the default experiment tolerance as defined in the XML
        description.
        """
        return FMIL.fmi1_import_get_default_experiment_tolerance(self._fmu)

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

    def __init__(self, fmu, path='.', enable_logging=True,log_file_name=""):
        #Call super
        FMUModelBase.__init__(self,fmu,path,enable_logging,log_file_name)

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
        if self._allocated_fmu == 1:
            FMIL.fmi1_import_terminate_slave(self._fmu)
            FMIL.fmi1_import_free_slave_instance(self._fmu)

        if self._allocated_dll == 1:
            FMIL.fmi1_import_destroy_dllfmu(self._fmu)

        if self._allocated_xml == 1:
            FMIL.fmi1_import_free(self._fmu)

        if self._fmu_temp_dir != NULL:
            FMIL.fmi_import_rmdir(&self.callbacks, self._fmu_temp_dir)
            FMIL.free(self._fmu_temp_dir)
            self._fmu_temp_dir = NULL

        if self._allocated_context == 1:
            FMIL.fmi_import_free_context(self.context)

        if self._fmu_log_name != NULL:
            FMIL.free(self._fmu_log_name)
            self._fmu_log_name = NULL

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
            new_s = 1
        else:
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
        cdef int can_interpolate_inputs
        cdef FMIL.size_t nref
        cdef FMIL.fmi1_import_capabilities_t *fmu_capabilities
        cdef N.ndarray[FMIL.fmi1_integer_t, ndim=1,mode='c'] np_orders = N.array(orders, dtype=N.int32, ndmin=1).flatten()
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] value_refs
        cdef N.ndarray[FMIL.fmi1_real_t, ndim=1,mode='c'] val = N.array(values, dtype=N.float, ndmin=1).flatten()

        nref = len(val)
        orders = N.array([0]*nref, dtype=N.int32)

        if nref != len(np_orders):
            raise FMUException("The number of variables must be the same as the number of orders.")

        fmu_capabilities = FMIL.fmi1_import_get_capabilities(self._fmu)
        can_interpolate_inputs = FMIL.fmi1_import_get_canInterpolateInputs(fmu_capabilities)
        #NOTE IS THIS THE HIGHEST ORDER OF INTERPOLATION OR SIMPLY IF IT CAN OR NOT?

        for i in range(len(np_orders)):
            if np_orders[i] < 1:
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
        else:
            raise FMUException("The variables must either be a string or a list of strings")

        status = FMIL.fmi1_import_set_real_input_derivatives(self._fmu, <FMIL.fmi1_value_reference_t*>value_refs.data, nref, <FMIL.fmi1_integer_t*>np_orders.data, <FMIL.fmi1_real_t*>val.data)

        if status != 0:
            raise FMUException('Failed to set the Real input derivatives.')

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
        cdef FMIL.fmi1_boolean_t stopDefined = 1 if StopTimeDefined else 0

        self.time = tStart

        status = FMIL.fmi1_import_initialize_slave(self._fmu, tStart, stopDefined, tStop)

        if status != FMIL.fmi1_status_ok:
            raise FMUException("The slave failed to initialize.")

        self._allocated_fmu = True

    def reset(self):
        """
        This metod resets the FMU according to the reset method defined
        in the FMI1 specification.
        """

        status = FMIL.fmi1_import_reset_slave(self._fmu)
        if status != FMIL.fmi1_status_ok:
            raise FMUException("Failed to reset the FMU.")

        #Default values
        self.__t = None

        #Internal values
        self._file_open = False
        self._npoints = 0
        self._log = []


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
        cdef FMIL.fmi1_boolean_t log = 1 if logging else 0
        cdef FMIL.fmi1_real_t timeout = 0.0
        cdef FMIL.fmi1_boolean_t visible = 0
        cdef FMIL.fmi1_boolean_t interactive = 0
        cdef FMIL.fmi1_string_t location = NULL

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

    def __init__(self, fmu, path='.', enable_logging=True, log_file_name=""):
        #Call super
        FMUModelBase.__init__(self,fmu,path,enable_logging,log_file_name)

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
        if self._allocated_fmu == 1:
            FMIL.fmi1_import_terminate(self._fmu)
            FMIL.fmi1_import_free_model_instance(self._fmu)

        if self._allocated_dll == 1:
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
        if self._allocated_fmu == 1:
            FMIL.fmi1_import_terminate(self._fmu)
            FMIL.fmi1_import_free_model_instance(self._fmu)

        if self._allocated_dll == 1:
            FMIL.fmi1_import_destroy_dllfmu(self._fmu)

        if self._allocated_xml == 1:
            FMIL.fmi1_import_free(self._fmu)

        if self._fmu_temp_dir != NULL:
            FMIL.fmi_import_rmdir(&self.callbacks, self._fmu_temp_dir)
            FMIL.free(self._fmu_temp_dir)
            self._fmu_temp_dir = NULL

        if self._allocated_context == 1:
            FMIL.fmi_import_free_context(self.context)

        if self._fmu_log_name != NULL:
            FMIL.free(self._fmu_log_name)
            self._fmu_log_name = NULL

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
        #rtol = FMIL.fmi1_import_get_default_experiment_tolerance(self._fmu)
        rtol = self.get_default_experiment_tolerance()
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
        cdef FMIL.fmi1_boolean_t log
        cdef int status

        if logging:
            log = 1
        else:
            log = 0

        status = FMIL.fmi1_import_instantiate_model(self._fmu, name)

        if status != 0:
            raise FMUException('Failed to instantiate the model.')

        #Just to be safe, some problems with Dymola (2012) FMUs not reacting
        #to logging when set to the instantiate method.
        status = FMIL.fmi1_import_set_debug_logging(self._fmu, log)

        if status != 0:
            raise FMUException('Failed to set the debugging option.')

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

    def terminate(self):
        """
        Calls the FMI function fmiTerminate() on the FMU.
        After this call, any call to a function changing the state of the FMU will fail.
        """
        FMIL.fmi1_import_terminate(self._fmu)



cdef class FMUModelBase2(ModelBase):
    """
    FMI Model loaded from a dll.
    """

    #FMIL related variables
    cdef FMIL.jm_callbacks              callbacks
    cdef FMIL.fmi_import_context_t*     _context
    cdef FMIL.fmi2_callback_functions_t callBackFunctions
    cdef FMIL.fmi2_import_t*            _fmu
    cdef FMIL.fmi2_fmu_kind_enu_t       _fmu_kind
    cdef FMIL.fmi_version_enu_t         _version
    cdef FMIL.jm_string                 last_error
    cdef FMIL.size_t                    _nEventIndicators
    cdef FMIL.size_t                    _nContinuousStates
    cdef FMIL.size_t                    _nCategories
    cdef FMIL.fmi2_event_info_t         _eventInfo

    #Internal values
    cdef list           _log
    cdef object         _fmu_temp_dir
    cdef object         _fmu_full_path
    cdef public object  _enable_logging
    cdef object         _allocated_context
    cdef object         _allocated_xml
    cdef object         _allocated_dll
    cdef object         _allocated_fmu
    cdef char*          _modelId
    cdef object         _modelName
    cdef public object  _fmu_log_name
    cdef list           _categories
    cdef public list    _save_real_variables_val
    cdef public list    _save_int_variables_val
    cdef public list    _save_bool_variables_val
    cdef object         __t
    cdef public object  _pyEventInfo

    def __init__(self, fmu, path = '.', enable_logging = True, log_file_name = ""):
        """
        Constructor of the model.

        Parameters::

            fmu --
                Name of the fmu as a string.

            path --
                Path to the fmu-directory.
                Default: '.' (working directory)

            enable_logging --
                Boolean for acesss to logging-messages.
                Default: True

            log_file_name --
                Filename for file used to save logmessages.
                Default: "" (Generates automatically)

        Returns::

            A model as an object from the class FMUModelFMU2
        """

        cdef int  status
        cdef dict reals_continuous
        cdef dict reals_discrete
        cdef dict int_discrete
        cdef dict bool_discrete

        #Contains the log information
        self._log               = []
        self._enable_logging    = enable_logging
        self._fmu_log_name      = ""
        self._categories        = []

        #Used for deallocation
        self._allocated_context = False
        self._allocated_xml     = False
        self._allocated_dll     = False
        self._allocated_fmu     = False
        self._fmu_temp_dir      = ""


        #Default values
        self.__t = None

        #Internal values
        self._pyEventInfo = PyEventInfo()

        #Specify the general callback functions
        self.callbacks.malloc           = FMIL.malloc
        self.callbacks.calloc           = FMIL.calloc
        self.callbacks.realloc          = FMIL.realloc
        self.callbacks.free             = FMIL.free
        self.callbacks.logger           = importlogger2
        self.callbacks.log_level        = FMIL.jm_log_level_info if enable_logging else FMIL.jm_log_level_error
        self.callbacks.context          = <void*> self

        #Specify FMI2 related callbacks
        self.callBackFunctions.logger               = FMIL.fmi2_log_forwarding
        self.callBackFunctions.allocateMemory       = FMIL.calloc
        self.callBackFunctions.freeMemory           = FMIL.free
        #self.callBackFunctions.stepFinished        = NULL
        self.callBackFunctions.componentEnvironment = NULL


        # Check that the file referenced by fmu has the correct file-ending
        self._fmu_full_path = os.path.abspath(os.path.join(path,fmu))
        if not self._fmu_full_path.endswith('.fmu'):
            raise FMUException("FMUModel must be instantiated with an FMU (.fmu) file.")

        #Check that the file exists
        if not os.path.isfile(self._fmu_full_path):
            raise FMUException('Could not locate the FMU in the specified directory.')

        # Create a struct for allocation
        self._context           = FMIL.fmi_import_allocate_context(&self.callbacks)
        self._allocated_context = True

        #Get the FMI version of the provided model
        self._fmu_temp_dir = create_temp_dir()
        self._version      = FMIL.fmi_import_get_fmi_version(self._context, self._fmu_full_path, self._fmu_temp_dir)

        #Check the version
        if self._version == FMIL.fmi_version_unknown_enu:
            last_error = FMIL.jm_get_last_error(&self.callbacks)
            if enable_logging:
                raise FMUException("The FMU version could not be determined. "+last_error)
            else:
                raise FMUException("The FMU version could not be determined. Enable logging for possibly more information.")

        if self._version != FMIL.fmi_version_2_0_enu:
            last_error = FMIL.jm_get_last_error(&self.callbacks)
            if enable_logging:
                raise FMUException("The FMU version is not supported by this class. "+last_error)
            else:
                raise FMUException("The FMU version is not supported by this class. Enable logging for possibly more information.")

        #Parse xml and check fmu-kind
        self._fmu           = FMIL.fmi2_import_parse_xml(self._context, self._fmu_temp_dir, NULL)

        if self._fmu is NULL:
            last_error = FMIL.jm_get_last_error(&self.callbacks)
            if enable_logging:
                raise FMUException("The XML-could not be read. "+last_error)
            else:
                raise FMUException('The XML-could not be read. Enable logging for possible nore information.')

        self.callBackFunctions.componentEnvironment = <FMIL.fmi2_component_environment_t>self._fmu
        self._fmu_kind      = FMIL.fmi2_import_get_fmu_kind(self._fmu)
        self._allocated_xml = True

        #FMU kind is unknown
        if self._fmu_kind == FMIL.fmi2_fmu_kind_unknown:
            last_error = FMIL.jm_get_last_error(&self.callbacks)
            if enable_logging:
                raise FMUException("The FMU kind could not be determined. "+last_error)
            else:
                raise FMUException("The FMU kind could not be determined. Enable logging for possibly more information.")


        #Connect the DLL
        status = FMIL.fmi2_import_create_dllfmu(self._fmu, self._fmu_kind, &self.callBackFunctions)
        if status == FMIL.jm_status_error:
            last_error = FMIL.fmi2_import_get_last_error(self._fmu)
            if enable_logging:
                raise FMUException(last_error)
            else:
                raise FMUException("Error loading the binary. Enable logging for possibly more information.")
        self._allocated_dll = True

        #Load information from model
        self._modelName         = FMIL.fmi2_import_get_model_name(self._fmu)
        self._nEventIndicators  = FMIL.fmi2_import_get_number_of_event_indicators(self._fmu)
        self._nContinuousStates = FMIL.fmi2_import_get_number_of_continuous_states(self._fmu)
        self._fmu_log_name      = (self._modelName + "_log.txt") if log_file_name=="" else log_file_name
        self._nCategories       = FMIL.fmi2_import_get_log_categories_num(self._fmu)

        #Create the log file
        with open(self._fmu_log_name,'w') as file:
            for i in range(len(self._log)):
                file.write("FMIL: module = %s, log level = %d: %s\n"%(self._log[i][0], self._log[i][1], self._log[i][2]))
            self._log = []



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

        cdef int         status
        cdef FMIL.size_t nref

        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1,mode='c'] input_valueref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()
        nref = len(input_valueref)
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1,mode='c']            output_value   = N.array([0.0]*nref,dtype=N.float, ndmin=1)

        status = FMIL.fmi2_import_get_real(self._fmu, <FMIL.fmi2_value_reference_t*> input_valueref.data, nref, <FMIL.fmi2_real_t*> output_value.data)

        if status != 0:
            raise FMUException('Failed to get the Real values.')

        return output_value

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

        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1,mode='c'] input_valueref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1,mode='c']            set_value      = N.array(values, dtype=N.float, ndmin=1).flatten()

        nref = len(input_valueref)

        if len(input_valueref) != len(set_value):
            raise FMUException('The length of valueref and values are inconsistent.')

        status = FMIL.fmi2_import_set_real(self._fmu, <FMIL.fmi2_value_reference_t*> input_valueref.data, nref, <FMIL.fmi2_real_t*> set_value.data)

        if status != 0:
            raise FMUException('Failed to set the Real values.')

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

        Calls the low-level FMI function: fmiGetInteger/fmiSetInteger
        """
        cdef int         status
        cdef FMIL.size_t nref

        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1,mode='c'] input_valueref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()
        nref = len(input_valueref)
        cdef N.ndarray[FMIL.fmi2_integer_t, ndim=1,mode='c']         output_value   = N.array([0]*nref, dtype=int,ndmin=1)


        status = FMIL.fmi2_import_get_integer(self._fmu, <FMIL.fmi2_value_reference_t*> input_valueref.data, nref, <FMIL.fmi2_integer_t*> output_value.data)

        if status != 0:
            raise FMUException('Failed to get the Integer values.')

        return output_value

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

        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1,mode='c'] input_valueref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()
        cdef N.ndarray[FMIL.fmi2_integer_t, ndim=1,mode='c']         set_value      = N.array(values, dtype=int,ndmin=1).flatten()

        nref = len(input_valueref)

        if len(input_valueref) != len(set_value):
            raise FMUException('The length of valueref and values are inconsistent.')

        status = FMIL.fmi2_import_set_integer(self._fmu, <FMIL.fmi2_value_reference_t*> input_valueref.data, nref, <FMIL.fmi2_integer_t*> set_value.data)

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
        cdef int         status
        cdef FMIL.size_t nref

        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1,mode='c'] input_valueref = N.array(valueref, dtype=N.uint32, ndmin=1).flatten()
        nref = len(input_valueref)

        #cdef N.ndarray[FMIL.fmi1_boolean_t, ndim=1,mode='c'] val = N.array(['0']*nref, dtype=N.char.character,ndmin=1)
        cdef void* output_value = FMIL.malloc(sizeof(FMIL.fmi2_boolean_t)*nref)


        #status = FMIL.fmi1_import_get_boolean(self._fmu, <FMIL.fmi1_value_reference_t*>val_ref.data, nref, <FMIL.fmi1_boolean_t*>val.data)
        status = FMIL.fmi2_import_get_boolean(self._fmu, <FMIL.fmi2_value_reference_t*> input_valueref.data, nref, <FMIL.fmi2_boolean_t*> output_value)

        return_values = []
        for i in range(nref):
            return_values.append((<FMIL.fmi2_boolean_t*> output_value)[i]==1)
            #print (<FMIL.fmi1_boolean_t*>val)[i], (<FMIL.fmi1_boolean_t*>val)[i]==1

        #print return_values

        #Dealloc
        FMIL.free(output_value)

        if status != 0:
            raise FMUException('Failed to get the Boolean values.')

        #return val==FMI_TRUE
        return N.array(return_values)

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
        cdef int         status
        cdef FMIL.size_t nref

        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1,mode='c'] input_valueref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()
        nref = len(input_valueref)

        #cdef N.ndarray[FMIL.fmi1_boolean_t, ndim=1,mode='c'] val = N.array(['0']*nref, dtype=N.char.character,ndmin=1).flatten()
        cdef void* set_value = FMIL.malloc(sizeof(FMIL.fmi2_boolean_t)*nref)

        values = N.array(values,ndmin=1).flatten()
        for i in range(nref):
            if values[i]:
                #val[i]=1
                (<FMIL.fmi2_boolean_t*> set_value)[i] = 1
            else:
                #val[i]=0
                (<FMIL.fmi2_boolean_t*> set_value)[i] = 0

        if len(input_valueref) != len(values):
            raise FMUException('The length of valueref and values are inconsistent.')

        #status = FMIL.fmi1_import_set_boolean(self._fmu, <FMIL.fmi1_value_reference_t*>val_ref.data, nref, <FMIL.fmi1_boolean_t*>val.data)
        status = FMIL.fmi2_import_set_boolean(self._fmu, <FMIL.fmi2_value_reference_t*> input_valueref.data, nref, <FMIL.fmi2_boolean_t*> set_value)

        FMIL.free(set_value)

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
        cdef int         status
        cdef FMIL.size_t nref

        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1, mode='c'] input_valueref = N.array(valueref, dtype=N.uint32, ndmin=1).flatten()
        nref = len(input_valueref)
        cdef N.ndarray[FMIL.fmi2_string_t, ndim=1, mode='c']         output_value   = N.array([''] * nref, dtype=N.char, ndmin=1)


        status = FMIL.fmi2_import_get_string(self._fmu, <FMIL.fmi2_value_reference_t*> input_valueref.data, nref, <FMIL.fmi2_string_t*> output_value.data)

        if status != 0:
            raise FMUException('Failed to get the String values.')

        return output_value

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
        cdef int         status
        cdef FMIL.size_t nref

        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1,mode='c'] input_valueref = N.array(valueref, dtype=N.uint32, ndmin=1).flatten()
        cdef N.ndarray[FMIL.fmi2_string_t, ndim=1, mode='c']         set_values     = N.array(values, dtype=N.char, ndmin=1).flatten()

        nref = input_valueref.size

        if input_valueref.size != set_values.size:
            raise FMUException('The length of valueref and values are inconsistent.')

        status = FMIL.fmi2_import_set_string(self._fmu, <FMIL.fmi2_value_reference_t*> input_valueref.data, nref, <FMIL.fmi2_string_t*> set_values.data)

        if status != 0:
            raise FMUException('Failed to set the String values.')

    def _set(self,char* variable_name, value):
        """
        Helper method to set, see docstring on set.
        """
        cdef FMIL.fmi2_value_reference_t ref
        cdef FMIL.fmi2_base_type_enu_t   type

        ref  = self.get_variable_valueref(variable_name)
        type = self.get_variable_data_type(variable_name)

        if type == FMIL.fmi2_base_type_real:  #REAL
            self.set_real([ref], [value])
        elif type == FMIL.fmi2_base_type_int: #INTEGER
            self.set_integer([ref], [value])
        elif type == FMIL.fmi2_base_type_str: #STRING
            self.set_string([ref], [value])
        elif type == FMIL.fmi2_base_type_bool: #BOOLEAN
            self.set_boolean([ref], [value])
        else:
            raise FMUException('Type not supported.')

    def _get(self,char* variable_name):
        """
        Helper method to get, see docstring on get.
        """
        cdef FMIL.fmi2_value_reference_t ref
        cdef FMIL.fmi2_base_type_enu_t type

        ref  = self.get_variable_valueref(variable_name)
        type = self.get_variable_data_type(variable_name)

        if type == FMIL.fmi2_base_type_real:  #REAL
            return self.get_real([ref])
        elif type == FMIL.fmi2_base_type_int: #INTEGER
            return self.get_integer([ref])
        elif type == FMIL.fmi2_base_type_str: #STRING
            return self.get_string([ref])
        elif type == FMIL.fmi2_base_type_bool: #BOOLEAN
            return self.get_boolean([ref])
        else:
            raise FMUException('Type not supported.')





    cdef _logger(self, FMIL.jm_string module, int log_level, FMIL.jm_string message):
        #print "FMIL: module = %s, log level = %d: %s"%(module, log_level, message)
        if self._fmu_log_name != "":
            with open(self._fmu_log_name,'a') as file:
                file.write("FMIL: module = %s, log level = %d: %s\n"%(module, log_level, message))
        else:
            self._log.append([module,log_level,message])

    def get_log(self):
        """
        Returns the log information as a list. To turn on the logging use the
        method, set_debug_logging(True) in the instantiation,
        FMUModelBase2(..., enable_logging=True). The log is stored as a list of lists.
        For example log[0] are the first log message to the log and consists of,
        in the following order, the instance name, the status, the category and
        the message.

        Returns::

            log -- A list of lists.
        """
        log = []
        if self._fmu_log_name != "":
            with open(self._fmu_log_name,'r') as file:
                while True:
                    line = file.readline()
                    if line == "":
                        break
                    log.append(line.strip("\n"))
            return log
        else:
            return self._log

    def print_log(self):
        """
        Prints the log information to the prompt.
        """
        cdef int N
        log = self.get_log()
        N = len(log)

        for i in range(N):
            print log[i]

    def set_fmil_log_level(self, level):
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
        self.callbacks.log_level = <FMIL.jm_log_level_enu_t> level

    def get_fmil_log_level(self):
        """
        Returns::

            The current fmil log-level.
        """

        cdef int level

        if self._enable_logging:
            level = self.callbacks.log_level
            return level
        else:
            raise FMUException('Logging is not enabled')

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

        Calls the low-level FMI function: fmiSetDebuggLogging
        """

        cdef FMIL.fmi2_boolean_t  log
        cdef int                  status
        cdef FMIL.size_t          nCat = 0

        self.callbacks.log_level = FMIL.jm_log_level_warning if logging_on else FMIL.jm_log_level_nothing

        if logging_on:
            log = 1
        else:
            log = 0

        self._enable_logging = bool(log)

        if len(categories) > 0:
            raise FMUException('Currently the logging of categories is not availible. See the docstring for more information')

        status = FMIL.fmi2_import_set_debug_logging(self._fmu, log, nCat, NULL)

        if status != 0:
            raise FMUException('Failed to set the debugging option.')

    def get_categories(self):
        """
        Method used to retrieve the logging categories.

        Returns::
            A list with two objects. The first is the number of log categories
            and the second is a list with the categories avalible for logging.
        """

        cdef FMIL.size_t i

        if self._categories == []:
            for i in range(self._nCategories):
                (self._categories).append(FMIL.fmi2_import_get_log_category(self._fmu, i))
            output = [self._nCategories, self._categories]
        else:
            output = [self._nCategories, self._categories]

        return output




    def get_variable_nominal(self, variablename=None, valueref=None):
        """
        Returns the nominal value from a real variable determined by
        either its value reference or its variable name.

        Parameters::

            variablename --
                The name of the variable.

            valueref --
                The value reference of the variable.

        Returns::

            The nominal value of the given variable.
        """
        cdef FMIL.fmi2_import_variable_t*      variable
        cdef FMIL.fmi2_import_real_variable_t* real_variable

        if valueref != None:
            variable = FMIL.fmi2_import_get_variable_by_vr(self._fmu, FMIL.fmi2_base_type_real, <FMIL.fmi2_value_reference_t>valueref)
            if variable == NULL:
                raise FMUException("The variable with value reference: %s, could not be found."%str(valueref))
        elif variablename != None:
            variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
            if variable == NULL:
                raise FMUException("The variable %s could not be found."%variablename)
        else:
            raise FMUException('Either provide value reference or variable name.')

        real_variable = FMIL.fmi2_import_get_variable_as_real(variable)
        if real_variable == NULL:
            raise FMUException("The variable is not a real variable.")

        return  FMIL.fmi2_import_get_real_variable_nominal(real_variable)

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
        cdef FMIL.fmi2_import_variable_t* variable
        cdef char* name

        variable = FMIL.fmi2_import_get_variable_by_vr(self._fmu, <FMIL.fmi2_base_type_enu_t> type, <FMIL.fmi2_value_reference_t> valueref)
        if variable == NULL:
            raise FMUException("The variable with the valuref %i could not be found."%valueref)

        name = FMIL.fmi2_import_get_variable_name(variable)

        return name

    def get_variable_alias_base(self, char* variablename):
        """
        Returns the base variable for the provided variable name.

        Parameters::

            variablename--
                Name of the variable.

        Returns:

           The base variable.
        """
        cdef FMIL.fmi2_import_variable_t* variable, *base_variable
        cdef FMIL.fmi2_value_reference_t vr

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        base_variable = FMIL.fmi2_import_get_variable_alias_base(self._fmu, variable)
        if base_variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        name = FMIL.fmi2_import_get_variable_name(base_variable)

        return name

    def get_variable_alias(self, char* variablename):
        """
        Return a dict of all alias variables belonging to the provided variable
        where the key are the names and the value indicating whether the variable
        should be negated or not.

        Parameters::

            variablename--
                Name of the variable to find alias of.


        Returns::

            A dict consisting of the alias variables along with no alias variable.
            The values indicates wheter or not the variable should be negated or not.

        Raises::

            FMUException if the variable is not in the model.
        """
        cdef FMIL.fmi2_import_variable_t         *variable
        cdef FMIL.fmi2_import_variable_list_t    *alias_list
        cdef FMIL.size_t                         alias_list_size
        cdef FMIL.fmi2_variable_alias_kind_enu_t alias_kind
        cdef dict                                ret_values = {}
        cdef FMIL.size_t                         i
        cdef char*                               alias_name

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        alias_list = FMIL.fmi2_import_get_variable_aliases(self._fmu, variable)

        alias_list_size = FMIL.fmi2_import_get_variable_list_size(alias_list)

        #Loop over all the alias variables
        for i in range(alias_list_size):

            variable = FMIL.fmi2_import_get_variable(alias_list, i)

            alias_kind = FMIL.fmi2_import_get_variable_alias_kind(variable)
            alias_name = FMIL.fmi2_import_get_variable_name(variable)

            ret_values[alias_name] = alias_kind

        #FREE VARIABLE LIST
        FMIL.fmi2_import_free_variable_list(alias_list)

        return ret_values

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
        cdef FMIL.fmi2_import_variable_t*        variable
        cdef FMIL.fmi2_import_variable_list_t*   variable_list
        cdef FMIL.size_t                         variable_list_size
        cdef FMIL.fmi2_value_reference_t         value_ref
        cdef FMIL.fmi2_base_type_enu_t           data_type,        target_type
        cdef FMIL.fmi2_variability_enu_t         data_variability, target_variability
        cdef FMIL.fmi2_variable_alias_kind_enu_t alias_kind
        cdef int   i
        cdef int  selected_filter = 1 if filter else 0
        cdef int  length_filter = 0
        cdef dict real_var_ref = {}
        cdef dict int_var_ref = {}
        cdef dict bool_var_ref = {}

        variable_list      = FMIL.fmi2_import_get_variable_list(self._fmu, 0)
        variable_list_size = FMIL.fmi2_import_get_variable_list_size(variable_list)

        if selected_filter:
            filter_list = self._convert_filter(filter)
            length_filter = len(filter_list)

        for i in range(variable_list_size):

            variable = FMIL.fmi2_import_get_variable(variable_list, i)

            alias_kind       = FMIL.fmi2_import_get_variable_alias_kind(variable)
            name             = FMIL.fmi2_import_get_variable_name(variable)
            value_ref        = FMIL.fmi2_import_get_variable_vr(variable)
            data_type        = FMIL.fmi2_import_get_variable_base_type(variable)
            data_variability = FMIL.fmi2_import_get_variability(variable)


            if data_type != FMI2_REAL and data_type != FMI2_INTEGER and data_type != FMI2_BOOLEAN:
                continue

            if data_variability != FMI2_CONTINUOUS and data_variability != FMI2_DISCRETE and data_variability != FMI2_TUNABLE:
                continue

            if selected_filter:
                for j in range(length_filter):
                    if re.match(filter_list[j], name):
                        break
                else:
                    continue
            else:
                if alias_kind != FMIL.fmi2_variable_is_not_alias:
                    continue

            if data_type == FMI2_REAL:
                real_var_ref[value_ref] = 1
            if data_type == FMI2_INTEGER:
                int_var_ref[value_ref] = 1
            if data_type == FMI2_BOOLEAN:
                bool_var_ref[value_ref] = 1

        #Free the variable list
        FMIL.fmi2_import_free_variable_list(variable_list)

        return real_var_ref.keys(), int_var_ref.keys(), bool_var_ref.keys()


    def get_model_variables(self, type = None, include_alias = True,
                             causality = None,   variability = None,
                            only_start = False,   only_fixed = False,
                            filter = None):
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
                The causality of the variables (Parameter==0, Input==1,
                Output==2, Local==3, Unknown==4).
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
        cdef FMIL.fmi2_import_variable_t*        variable
        cdef FMIL.fmi2_import_variable_list_t*   variable_list
        cdef FMIL.size_t                         variable_list_size
        cdef FMIL.fmi2_value_reference_t         value_ref
        cdef FMIL.fmi2_base_type_enu_t           data_type,        target_type
        cdef FMIL.fmi2_variability_enu_t         data_variability, target_variability
        cdef FMIL.fmi2_causality_enu_t           data_causality,   target_causality
        cdef FMIL.fmi2_variable_alias_kind_enu_t alias_kind
        cdef char* desc
        cdef dict  variable_dict = {}
        cdef int   selected_type = 0        #If a type has been selected
        cdef int   selected_variability = 0 #If a variability has been selected
        cdef int   selected_causality = 0   #If a causality has been selected
        cdef int   has_start, is_fixed
        cdef int   i
        cdef int  selected_filter = 1 if filter else 0
        cdef int  length_filter = 0

        variable_list      = FMIL.fmi2_import_get_variable_list(self._fmu, 0)
        variable_list_size = FMIL.fmi2_import_get_variable_list_size(variable_list)

        if type != None:        #A type have has been selected
            target_type = type
            selected_type = 1
        if causality != None:   #A causality has been selected
            target_causality = causality
            selected_causality = 1
        if variability != None: #A variability has been selected
            target_variability = variability
            selected_variability = 1
        if selected_filter:
            filter_list = self._convert_filter(filter)
            length_filter = len(filter_list)

        for i in range(variable_list_size):

            variable = FMIL.fmi2_import_get_variable(variable_list, i)

            alias_kind       = FMIL.fmi2_import_get_variable_alias_kind(variable)
            name             = FMIL.fmi2_import_get_variable_name(variable)
            value_ref        = FMIL.fmi2_import_get_variable_vr(variable)
            data_type        = FMIL.fmi2_import_get_variable_base_type(variable)
            has_start        = FMIL.fmi2_import_get_variable_has_start(variable)  #fmi2_import_get_initial, may be of interest
            data_variability = FMIL.fmi2_import_get_variability(variable)
            data_causality   = FMIL.fmi2_import_get_causality(variable)
            desc             = FMIL.fmi2_import_get_variable_description(variable)

            #If only variables with start are wanted, check if the variable has start
            if only_start and has_start != 1:
                continue

            if only_fixed:
                #fixed variability requires start-value
                if has_start != 1:
                    continue
                elif (FMIL.fmi2_import_get_variability(variable) != FMIL.fmi2_variability_enu_fixed):
                    continue

            if selected_type == 1 and data_type != target_type:
                continue
            if selected_causality == 1 and data_causality != target_causality:
                continue
            if selected_variability == 1 and data_variability != target_variability:
                continue

            if selected_filter:
                for j in range(length_filter):
                    if re.match(filter_list[j], name):
                        break
                else:
                    continue

            if include_alias:
                #variable_dict[name] = value_ref
                variable_dict[name] = ScalarVariable2(name,
                                       value_ref, data_type, desc.decode('UTF-8') if desc!=NULL else "",
                                       data_variability, data_causality,
                                       alias_kind)
            elif alias_kind == FMIL.fmi2_variable_is_not_alias:
                #Exclude alias
                #variable_dict[name] = value_ref
                variable_dict[name] = ScalarVariable2(name,
                                       value_ref, data_type, desc.decode('UTF-8') if desc!=NULL else "",
                                       data_variability, data_causality,
                                       alias_kind)

        #Free the variable list
        FMIL.fmi2_import_free_variable_list(variable_list)

        return variable_dict

    def get_variable_references(self):
        """
        Retrieves the value references of all variables. The
        information about the variables are retrieved from the XML-file.

        Returns::

            vr_real --
                The Real-valued variables.

            vr_int --
                The Integer-valued variables.

            vr_bool --
                The Boolean-valued variables.

            vr_str --
                The String-valued variables.

            vr_enum --
                The Enum-valued variables

        Example::

            [r,i,b,s,e] = model.get_value_references()
        """
        vr_real = self._save_real_variables_val
        vr_int  = self._save_int_variables_val
        vr_bool = self._save_bool_variables_val

        get_str_var = self.get_model_variables(type=3)
        vr_str  = [s.value_reference for s in get_str_var.values()]

        get_enum_var = self.get_model_variables(type=4)
        vr_enum = [e.value_reference for e in get_enum_var.values()]

        return vr_real, vr_int, vr_bool, vr_str, vr_enum

    cpdef FMIL.fmi2_value_reference_t get_variable_valueref(self, char* variablename) except *:
        """
        Extract the ValueReference given a variable name.

        Parameters::

            variablename --
                The name of the variable.

        Returns::

            The ValueReference for the variable passed as argument.
        """
        cdef FMIL.fmi2_import_variable_t* variable
        cdef FMIL.fmi2_value_reference_t  vr

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
        vr =  FMIL.fmi2_import_get_variable_vr(variable)

        return vr

    cpdef FMIL.fmi2_base_type_enu_t get_variable_data_type(self, char* variablename) except *:
        """
        Get data type of variable.

        Parameter::

            variablename --
                The name of the variable.

        Returns::

            The type of the variable.
        """
        cdef FMIL.fmi2_import_variable_t* variable
        cdef FMIL.fmi2_base_type_enu_t    type

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        type = FMIL.fmi2_import_get_variable_base_type(variable)

        return type

    cpdef get_variable_description(self, char* variablename):
        """
        Get the description of a given variable.

        Parameter::

            variablename --
                The name of the variable

        Returns::

            The description of the variable.
        """
        cdef FMIL.fmi2_import_variable_t* variable
        cdef char* desc

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        desc = FMIL.fmi2_import_get_variable_description(variable)

        return desc if desc != NULL else ""

    cpdef FMIL.fmi2_variability_enu_t get_variable_variability(self,char* variablename) except *:
        """
        Get variability of the variable.

        Parameters::

            variablename --
                The name of the variable.

        Returns::

            The variability of the variable: CONSTANT(0), FIXED(1),
            TUNABLE(2), DISCRETE(3), CONTINUOUS(4) or UNKNOWN(5)
        """
        cdef FMIL.fmi2_import_variable_t* variable
        cdef FMIL.fmi2_variability_enu_t variability

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
        variability = FMIL.fmi2_import_get_variability(variable)

        return variability

    cpdef FMIL.fmi2_causality_enu_t get_variable_causality(self, char* variablename) except *:
        """
        Get the causality of the variable.

        Parameters::

            variablename --
                The name of the variable.

        Returns::

            The variability of the variable, PARAMETER(0), INPUT(1),
            OUTPUT(2), LOCAL(3), UNKNOWN(4)
        """
        cdef FMIL.fmi2_import_variable_t* variable
        cdef FMIL.fmi2_causality_enu_t causality

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        causality = FMIL.fmi2_import_get_causality(variable)

        return causality

    cpdef get_variable_start(self, char* variablename):
        """
        Returns the start value for the variable or else raises
        FMUException.

        Parameters::

            variablename --
                The name of the variable

        Returns::

            The start value.
        """
        cdef FMIL.fmi2_import_variable_t *        variable
        cdef FMIL.fmi2_base_type_enu_t            type
        cdef FMIL.fmi2_import_integer_variable_t* int_variable
        cdef FMIL.fmi2_import_real_variable_t*    real_variable
        cdef FMIL.fmi2_import_bool_variable_t*    bool_variable
        cdef FMIL.fmi2_import_enum_variable_t*    enum_variable
        cdef FMIL.fmi2_import_string_variable_t*  str_variable
        cdef int                                  status
        cdef FMIL.fmi2_boolean_t                  FMITRUE = 1

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        status = FMIL.fmi2_import_get_variable_has_start(variable)

        if status == 0:
            raise FMUException("The variable %s does not have a start value."%variablename)

        type = FMIL.fmi2_import_get_variable_base_type(variable)

        if type == FMIL.fmi2_base_type_real:
            real_variable = FMIL.fmi2_import_get_variable_as_real(variable)
            return FMIL.fmi2_import_get_real_variable_start(real_variable)

        elif type == FMIL.fmi2_base_type_int:
            int_variable = FMIL.fmi2_import_get_variable_as_integer(variable)
            return FMIL.fmi2_import_get_integer_variable_start(int_variable)

        elif type == FMIL.fmi2_base_type_bool:
            bool_variable = FMIL.fmi2_import_get_variable_as_boolean(variable)
            return FMIL.fmi2_import_get_boolean_variable_start(bool_variable) == FMITRUE

        elif type == FMIL.fmi2_base_type_enum:
            enum_variable = FMIL.fmi2_import_get_variable_as_enum(variable)
            return FMIL.fmi2_import_get_enum_variable_start(enum_variable)

        elif type == FMIL.fmi2_base_type_str:
            str_variable = FMIL.fmi2_import_get_variable_as_string(variable)
            return FMIL.fmi2_import_get_string_variable_start(str_variable)

        else:
            raise FMUException("Unknown variable type.")

    cpdef get_variable_max(self, char* variablename):
        """
        Returns the maximum value for the given variable.

        Parameters::

            variablename --
                The name of the variable.

        Returns::

            The maximum value for the variable.
        """
        cdef FMIL.fmi2_import_variable_t*         variable
        cdef FMIL.fmi2_import_integer_variable_t* int_variable
        cdef FMIL.fmi2_import_real_variable_t*    real_variable
        cdef FMIL.fmi2_import_enum_variable_t*    enum_variable
        cdef FMIL.fmi2_base_type_enu_t            type

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        type = FMIL.fmi2_import_get_variable_base_type(variable)

        if type == FMIL.fmi2_base_type_real:
            real_variable = FMIL.fmi2_import_get_variable_as_real(variable)
            return FMIL.fmi2_import_get_real_variable_max(real_variable)

        elif type == FMIL.fmi2_base_type_int:
            int_variable = FMIL.fmi2_import_get_variable_as_integer(variable)
            return FMIL.fmi2_import_get_integer_variable_max(int_variable)

        elif type == FMIL.fmi2_base_type_enum:
            enum_variable = FMIL.fmi2_import_get_variable_as_enum(variable)
            return FMIL.fmi2_import_get_enum_variable_max(enum_variable)

        else:
            raise FMUException("The variable type does not have a maximum value.")

    cpdef get_variable_min(self, char* variablename):
        """
        Returns the minimum value for the given variable.

        Parameters::

            variablename --
                The name of the variable.

        Returns::

            The minimum value for the variable.
        """
        cdef FMIL.fmi2_import_variable_t*         variable
        cdef FMIL.fmi2_import_integer_variable_t* int_variable
        cdef FMIL.fmi2_import_real_variable_t*    real_variable
        cdef FMIL.fmi2_import_enum_variable_t*    enum_variable
        cdef FMIL.fmi2_base_type_enu_t            type

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        type = FMIL.fmi2_import_get_variable_base_type(variable)

        if type == FMIL.fmi2_base_type_real:
            real_variable = FMIL.fmi2_import_get_variable_as_real(variable)
            return FMIL.fmi2_import_get_real_variable_min(real_variable)

        elif type == FMIL.fmi2_base_type_int:
            int_variable = FMIL.fmi2_import_get_variable_as_integer(variable)
            return FMIL.fmi2_import_get_integer_variable_min(int_variable)

        elif type == FMIL.fmi2_base_type_enum:
            enum_variable = FMIL.fmi2_import_get_variable_as_enum(variable)
            return FMIL.fmi2_import_get_enum_variable_min(enum_variable)

        else:
            raise FMUException("The variable type does not have a minimum value.")





    def get_fmu_state(self):
        """
        Creates a copy of the recent FMU-state and returns
        a pointer to this state which later can be used to
        set the FMU to this state.

        Returns::

            A pointer to a copy of the recent FMU state.

        Example::

            FMU_state = Model.get_fmu_state()
        """
        raise NotImplementedError

        cdef int status
        cdef object cap1, cap2
        cdef FMUState2 state = FMUState2()


        cap1 = FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_canGetAndSetFMUstate)
        cap2 = FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canGetAndSetFMUstate)
        if not cap1 and not cap2:
            raise FMUException('This FMU dos not support get and set FMU-state')

        status = FMIL.fmi2_import_get_fmu_state(self._fmu, state.fmu_state)

        if status != 0:
            raise FMUException('An error occured while trying to get the FMU-state, see the log for possible more information')

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
        raise NotImplementedError

        cdef int status
        cdef object cap1, cap2
        cdef FMUState2 internal_state = state

        cap1 = FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_canGetAndSetFMUstate)
        cap2 = FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canGetAndSetFMUstate)
        if not cap1 and not cap2:
            raise FMUException('This FMU dos not support get and set FMU-state')

        status = FMIL.fmi2_import_set_fmu_state(self._fmu, internal_state.fmu_state)

        if status != 0:
            raise FMUException('An error occured while trying to set the FMU-state, see the log for possible more information')

        return None

    def free_fmu_state(self, state):
        """
        Free a previously saved FMU-state from the memory.

        Parameters::

            state--
                A pointer to the FMU-state to be set free.

        Example::

            FMU_state = Model.get_fmu_state
            Model.free_fmu_state(FMU_state)

        """
        raise NotImplementedError

        cdef int status
        cdef FMUState2 internal_state = state

        status = FMIL.fmi2_import_free_fmu_state(self._fmu, internal_state.fmu_state)

        if status != 0:
            raise FMUException('An error occured while trying to free the FMU-state, see the log for possible more information')

        return None

    cpdef serialize_fmu_state(self, state):
        """
        Serialize the data referenced by the input argumemt.

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

        cdef int status
        cdef object cap1, cap2
        cdef FMUState2 internal_state = state

        cdef FMIL.size_t n_bytes
        cdef N.ndarray[FMIL.fmi2_byte_t, ndim=1, mode='c'] serialized_fmu

        cap1 = FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_canSerializeFMUstate)
        cap2 = FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canSerializeFMUstate)
        if not cap1 and not cap2:
            raise FMUException('This FMU dos not support serialisation of FMU-state')

        n_bytes = self.serialized_fmu_state_size(state)
        serialized_fmu = N.empty(n_bytes, dtype=N.char)

        status = FMIL.fmi2_import_serialize_fmu_state(self._fmu, internal_state.fmu_state, <FMIL.fmi2_byte_t*> serialized_fmu.data, n_bytes)

        if status != 0:
            raise FMUException('An error occured while serializing the FMU-state, see the log for possible more information')

        return serialized_fmu

    cpdef deserialize_fmu_state(self, serialized_fmu):
        """
        Deserialize the provided byte-vector and returns the corresponding FMU-state.

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

        cdef int status
        cdef N.ndarray[FMIL.fmi2_byte_t, ndim=1, mode='c'] ser_fmu = serialized_fmu
        cdef FMUState2 state = FMUState2()
        cdef FMIL.size_t n_byte = len(ser_fmu)

        status = FMIL.fmi2_import_de_serialize_fmu_state(self._fmu, <FMIL.fmi2_byte_t *> ser_fmu.data, n_byte, state.fmu_state)

        if status != 0:
            raise FMUException('An error occured while deserializing the FMU-state, see the log for possible more information')

        return state

    cpdef serialized_fmu_state_size(self, state):
        """
        Returns the required size of a vector needed to serialize the specified FMU-state

        Parameters::

            state--
                A FMU-state

        Returns::

            The size of teh vector.
        """
        raise NotImplementedError

        cdef int status
        cdef FMUState2 internal_state = state
        cdef FMIL.size_t n_bytes

        status = FMIL.fmi2_import_serialized_fmu_state_size(self._fmu, internal_state.fmu_state, &n_bytes)

        if status != 0:
            raise FMUException('An error occured while computing the FMU-state size, see the log for possible more information')

        return n_bytes




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

    def get_default_experiment_start_time(self):
        """
        Returns the default experiment start time as defined the XML
        description.
        """
        return FMIL.fmi2_import_get_default_experiment_start(self._fmu)

    def get_default_experiment_stop_time(self):
        """
        Returns the default experiment stop time as defined the XML
        description.
        """
        return FMIL.fmi2_import_get_default_experiment_stop(self._fmu)

    def get_default_experiment_tolerance(self):
        """
        Returns the default experiment tolerance as defined in the XML
        description.
        """
        return FMIL.fmi2_import_get_default_experiment_tolerance(self._fmu)

    def _convert_filter(self, expression):
        """
        Convert a filter based on unix filename pattern matching to a
        list of regular expressions.

        Parameters::

            expression--
                String or list to convert.

        Returns::

            The converted filter.
        """
        regexp = []
        if isinstance(expression,str):
            regex = fnmatch.translate(expression)
            regexp = [re.compile(regex)]
        elif isinstance(expression,list):
            for i in expression:
                regex = fnmatch.translate(i)
                regexp.append(re.compile(regex))
        else:
            raise FMUException("Unknown input.")
        return regexp

    cdef _add_scalar_variables(self, FMIL.fmi2_import_variable_list_t*   variable_list):
        """
        Helper method to create scalar variables from a variable list.
        """
        cdef FMIL.size_t             variable_list_size
        variable_dict = OrderedDict()

        variable_list_size = FMIL.fmi2_import_get_variable_list_size(variable_list)

        for i in range(variable_list_size):

            variable = FMIL.fmi2_import_get_variable(variable_list, i)

            alias_kind       = FMIL.fmi2_import_get_variable_alias_kind(variable)
            name             = FMIL.fmi2_import_get_variable_name(variable)
            value_ref        = FMIL.fmi2_import_get_variable_vr(variable)
            data_type        = FMIL.fmi2_import_get_variable_base_type(variable)
            data_variability = FMIL.fmi2_import_get_variability(variable)
            data_causality   = FMIL.fmi2_import_get_causality(variable)
            desc             = FMIL.fmi2_import_get_variable_description(variable)

            variable_dict[name] = ScalarVariable2(name,
                                    value_ref, data_type, desc.decode('UTF-8') if desc!=NULL else "",
                                    data_variability, data_causality,
                                    alias_kind)

        return variable_dict

    def get_derivatives_list(self):
        """
        Returns a dictonary with the states derivatives.

        Returns::

            An ordered dictonary with the derivative variables.
        """
        cdef FMIL.fmi2_import_variable_list_t*   variable_list

        variable_list = FMIL.fmi2_import_get_derivatives_list(self._fmu)
        if variable_list == NULL:
            raise FMUException("The returned states list is NULL.")

        variable_dict = self._add_scalar_variables(variable_list)

        #Free the variable list
        FMIL.fmi2_import_free_variable_list(variable_list)

        return variable_dict

    def get_states_list(self):
        """
        Returns a dictonary with the states.

        Returns::

            An ordered dictonary with the state variables.
        """
        cdef FMIL.fmi2_import_variable_list_t*   variable_list

        variable_list = FMIL.fmi2_import_get_states_list(self._fmu)
        if variable_list == NULL:
            raise FMUException("The returned states list is NULL.")

        variable_dict = self._add_scalar_variables(variable_list)

        #Free the variable list
        FMIL.fmi2_import_free_variable_list(variable_list)

        return variable_dict

    def get_input_list(self):
        """
        Returns a dictonary with input variables

        Returns::

            An ordered dictonary with the input variables.
        """
        cdef FMIL.fmi2_import_variable_list_t*   input_list

        input_list = FMIL.fmi2_import_get_inputs_list(self._fmu)
        if input_list == NULL:
            raise FMUException("The returned input list is NULL.")

        variable_dict = self._add_scalar_variables(input_list)

        #Free the variable list
        FMIL.fmi2_import_free_variable_list(input_list)

        return variable_dict

    def get_output_list(self):
        """
        Returns a dictonary with output variables

        Returns::

            An ordered dictonary with the output variables.
        """
        cdef FMIL.fmi2_import_variable_list_t*   output_list

        output_list = FMIL.fmi2_import_get_outputs_list(self._fmu)
        if output_list == NULL:
            raise FMUException("The returned input list is NULL.")

        variable_dict = self._add_scalar_variables(output_list)

        #Free the variable list
        FMIL.fmi2_import_free_variable_list(output_list)

        return variable_dict

    def get_capability_flags(self):
        """
        Returns a dictionary with the cability flags of the FMU.

        Returns::

            me_needsExecutionTool
            me_completedIntegratorStepNotNeeded
            me_canBeInstantiatedOnlyOncePerProcess
            me_canNotUseMemoryManagementFunctions
            me_canGetAndSetFMUstate
            me_canSerializeFMUstate
            me_providesDirectionalDerivatives
            me_completedEventIterationIsProvided
            cs_needsExecutionTool
            cs_canHandleVariableCommunicationStepSize
            cs_canHandleEvents
            cs_canInterpolateInputs
            cs_maxOutputDerivativeOrder
            cs_canRunAsynchronuously
            cs_canSignalEvents
            cs_canBeInstantiatedOnlyOncePerProcess
            cs_canNotUseMemoryManagementFunctions
            cs_canGetAndSetFMUstate
            cs_canSerializeFMUstate
            capabilities_Num
        """
        cdef dict capabilities = {}
        cdef dict cap          = {}

        capabilities['me_needsExecutionTool']                     = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_needsExecutionTool))
        capabilities['me_completedIntegratorStepNotNeeded']       = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_completedIntegratorStepNotNeeded))
        capabilities['me_canBeInstantiatedOnlyOncePerProcess']    = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_canBeInstantiatedOnlyOncePerProcess))
        capabilities['me_canNotUseMemoryManagementFunctions']     = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_canNotUseMemoryManagementFunctions))
        capabilities['me_canGetAndSetFMUstate']                   = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_canGetAndSetFMUstate))
        capabilities['me_canSerializeFMUstate']                   = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_canSerializeFMUstate))
        capabilities['me_providesDirectionalDerivatives']         = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_providesDirectionalDerivatives))
        capabilities['me_completedEventIterationIsProvided']      = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_completedEventIterationIsProvided))
        capabilities['cs_needsExecutionTool']                     = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_needsExecutionTool))
        capabilities['cs_canHandleVariableCommunicationStepSize'] = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canHandleVariableCommunicationStepSize))
        capabilities['cs_canHandleEvents']                        = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canHandleEvents))
        capabilities['cs_canInterpolateInputs']                   = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canInterpolateInputs))
        capabilities['cs_maxOutputDerivativeOrder']               = FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_maxOutputDerivativeOrder)
        capabilities['cs_canRunAsynchronuously']                  = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canRunAsynchronuously))
        capabilities['cs_canSignalEvents']                        = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canSignalEvents))
        capabilities['cs_canBeInstantiatedOnlyOncePerProcess']    = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canBeInstantiatedOnlyOncePerProcess))
        capabilities['cs_canNotUseMemoryManagementFunctions']     = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canNotUseMemoryManagementFunctions))
        capabilities['cs_canGetAndSetFMUstate']                   = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canGetAndSetFMUstate))
        capabilities['cs_canSerializeFMUstate']                   = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canSerializeFMUstate))
        capabilities['capabilities_Num']                          = FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_capabilities_Num)

        if isinstance(self,FMUModelME2):
            for key in sorted(capabilities.keys())[12:20]:
                cap[key] = capabilities[key]
            key = sorted(capabilities.keys())[0]
            cap[key] = capabilities[key]
            return cap

        elif isinstance(self,FMUModelCS2):
            for key in sorted(capabilities.keys())[0:12]:
                cap[key] = capabilities[key]
            return cap

        else:
            raise FMUException('The instance is not curent an instance of an ME-model or a CS-model. Use load_fmu for correct loading.')

    def get_directional_derivative(self, var_ref, func_ref, v):
        """
        Returns the directional derivatives of the functions with respect
        to the given variables and in the given direction.
        In other words, it returns linear combinations of the partial derivatives
        of the given functions with respect to the selected variables.
        The point of eveluation is the current time-point.

        Parameters::

            var_ref --
                A list of variable references that the partial derivatives
                will be calculated with respect to.

            func_ref --
                A list of function references for which the partial derivatives will be calculated.

            v --
                A seed vector specifing the linear combination of the partial derivatives.

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

        cdef int status
        cdef FMIL.size_t nv, nz

        #input arrays
        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1, mode='c'] v_ref = N.zeros(len(var_ref),  dtype = N.uint32)
        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1, mode='c'] z_ref = N.zeros(len(func_ref), dtype = N.uint32)
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c'] dv    = N.zeros(len(v),        dtype = N.double)
        #output array
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c'] dz    = N.zeros(len(func_ref), dtype = N.double)

        if not FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_providesDirectionalDerivatives):
            raise FMUException('This FMU does not provide directional derivatives')

        if len(var_ref) != len(v):
            raise FMUException('The length of the list with variables (var_ref) and the seed vector (V) are not equal')

        for i in range(len(var_ref)):
            v_ref[i] = var_ref[i]
            dv[i] = v[i]
        for j in range(len(func_ref)):
            z_ref[j] = func_ref[j]

        nv = len(v_ref)
        nz = len(z_ref)

        status = FMIL.fmi2_import_get_directional_derivative(self._fmu, <FMIL.fmi2_value_reference_t*> v_ref.data, nv, <FMIL.fmi2_value_reference_t*> z_ref.data, nz, <FMIL.fmi2_real_t*> dv.data, <FMIL.fmi2_real_t*> dz.data)

        if status != 0:
            raise FMUException('An error occured while getting the directional derivative, see the log for possible more information')

        return dz





    def get_version(self):
        """
        Returns the FMI version of the Model which it was generated according.

        Returns::

            version --
                The version.

        Example::

            model.get_version()
        """
        cdef char* version = FMIL.fmi2_import_get_version(self._fmu)
        return version

    def get_name(self):
        """
        Return the model name as used in the modeling environment.
        """
        return self._modelName

    def get_author(self):
        """
        Return the name and organization of the model author.
        """
        cdef char* author
        author = FMIL.fmi2_import_get_author(self._fmu)
        return author if author != NULL else ""

    def get_description(self):
        """
        Return the model description.
        """
        cdef char* desc
        desc = FMIL.fmi2_import_get_description(self._fmu)
        return desc if desc != NULL else ""

    def get_generation_tool(self):
        """
        Return the model generation tool.
        """
        cdef char* gen
        gen = FMIL.fmi2_import_get_generation_tool(self._fmu)
        return gen if gen != NULL else ""

    def get_guid(self):
        """
        Return the model GUID.
        """
        guid = FMIL.fmi2_import_get_GUID(self._fmu)
        return guid

    def get_identifier(self):
        """
        Return the model identifier, name of binary model file and prefix in
        the C-function names of the model.
        """
        return self._modelId

    def get_model_types_platform(self):
        """
        Returns the set of valid compatible platforms for the Model, extracted
        from the XML.
        """
        return FMIL.fmi2_import_get_types_platform(self._fmu)


cdef class FMUModelCS2(FMUModelBase2):
    """
    Co-simulation model loaded from a dll
    """

    def __init__(self, fmu, path = '.', enable_logging = True, log_file_name = ""):
        """
        Constructor of the model.

        Parameters::

            fmu --
                Name of the fmu as a string.

            path --
                Path to the fmu-directory.
                Default: '.' (working directory)

            enable_logging --
                Boolean for acesss to logging-messages.
                Default: True

            log_file_name --
                Filename for file used to save logmessages.
                Default: "" (Generates automatically)

        Returns::

            A model as an object from the class FMUModelCS2
        """

        #Call super
        FMUModelBase2.__init__(self, fmu, path, enable_logging, log_file_name)

        if self._fmu_kind != FMIL.fmi2_fmu_kind_cs:
            if self._fmu_kind != FMIL.fmi2_fmu_kind_me_and_cs:
                raise FMUException("This class only supports FMI 1.0 for Co-simulation.")

        if self.get_capability_flags()['cs_needsExecutionTool'] == True:
            raise FMUException('Models that need an execution tool are not supported')

        self._modelId = FMIL.fmi2_import_get_model_identifier_CS(self._fmu)
        self.instantiate_slave()

    def __dealloc__(self):
        """
        Deallocate memory allocated
        """
        if self._allocated_fmu:
            FMIL.fmi2_import_terminate_slave(self._fmu)
            FMIL.fmi2_import_free_slave_instance(self._fmu)

        if self._allocated_dll:
            FMIL.fmi2_import_destroy_dllfmu(self._fmu)

        if self._allocated_xml:
            FMIL.fmi2_import_free(self._fmu)

        if self._allocated_context:
            FMIL.fmi_import_free_context(self._context)

        if self._fmu_temp_dir:
            FMIL.fmi_import_rmdir(&self.callbacks, self._fmu_temp_dir)

    def instantiate_slave(self, name = 'Slave', visible = False):
        """
        Instantiate the slave.

        Parameters::

            name --
                The name of the instance.
                Default: 'Slave'

            visible --
                Defines if the simulator application window should be visible or not.
                Default: False, not visible.

        Calls the low-level FMI function: fmiInstantiateSlave.
        """

        cdef FMIL.fmi2_boolean_t _visible = 0
        cdef FMIL.fmi2_string_t  location = NULL
        cdef int                 status

        if visible:
            _visible = 1

        status = FMIL.fmi2_import_instantiate_slave(self._fmu, name, location, _visible)

        if status != FMIL.jm_status_success:
            raise FMUException('Failed to instantiate the slave.')

    def initialize(self, tStart=0.0, tStop=1.0, StopTimeDefined=False, relTol=None):
        """
        Initializes the slave.

        Parameters::

            tStart --
                Start time of the simulation.
                Default: 0.0

            tStop --
                Stop time of the simulation.
                Default: 1.0

            StopTimeDefined --
                Boolen indicating if tStop is defined.
                Default: False

            relTol --
                Relative tolerance used in the simulation.
                Default: None (Default experiment tolerance)

        Calls the low-level FMU function: fmiInstantiateSlave
        """
        cdef int status
        cdef FMIL.fmi2_boolean_t stopDefined
        cdef FMIL.fmi2_real_t    relativeTol

        if StopTimeDefined:
            stopDefined = 1
        else:
            stopDefined = 0

        if relTol is None:
            relativeTol = self.get_default_experiment_tolerance()
        else:
            relativeTol = relTol


        self.time = tStart
        status = FMIL.fmi2_import_initialize_slave(self._fmu, relativeTol, tStart, stopDefined, tStop)

        if status != FMIL.fmi2_status_ok:
            raise FMUException("The slave failed to initialize.")

        self._allocated_fmu = True

    def reset_slave(self):
        """
        Function called by the master after a simulation before
        starting a new one. Note that the envoronment has to initialize
        the FMU again after this function-call.
        """

        cdef int status

        status = FMIL.fmi2_import_reset_slave(self._fmu)
        if status != 0:
            raise FMUException('An error occured when reseting the slave, see the log for possible more information')

        #Default values
        self.__t = None

        #Internal values
        self._log = []


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
        self.__t = t

    time = property(_get_time,_set_time, doc =
    """
    Property for accessing the current time of the simulation. Calls the
    low-level FMI function: fmiSetTime.
    """)



    def do_step(self, FMIL.fmi2_real_t current_t, FMIL.fmi2_real_t step_size, new_step=True):
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
        cdef FMIL.fmi2_boolean_t new_s

        if new_step:
            new_s = 1
        else:
            new_s = 0

        self.time = current_t + step_size

        status = FMIL.fmi2_import_do_step(self._fmu, current_t, step_size, new_s)

        return status

    def cancel_step(self):
        """
        Cancel a current integrator step. Can only be called if the
        status from do_step returns FMI_PENDING.
        After this function has been called, only calls to the low-level
        functions:
            -fmiTerminateSlave
            -fmiResetSlave
            -fmiFreeSlaveInstance
        are allowed
        """

        cdef int status

        status = FMIL.fmi2_import_cancel_step(self._fmu)
        if status != 0:
            raise FMUException('An error occured while canceling the step')


    def set_input_derivatives(self, variables, values, FMIL.fmi2_integer_t order):
        """
        Sets the input derivative order for the specified variables.

        Parameters::

                variables --
                        The variables as a string or list of strings for
                        which the input derivative(s) should be set.

                values --
                        The actual values.

                order --
                        The derivative order to set.
        """
        cdef int          status
        cdef unsigned int can_interpolate_inputs
        cdef FMIL.size_t  nref
        cdef N.ndarray[FMIL.fmi2_integer_t, ndim=1, mode='c']         orders
        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1, mode='c'] value_refs
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c']            val = N.array(values, dtype=N.float, ndmin=1).flatten()

        nref = len(val)
        orders = N.array([0]*nref, dtype=N.int32)

        can_interpolate_inputs = FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canInterpolateInputs)
        #NOTE IS THIS THE HIGHEST ORDER OF INTERPOLATION OR SIMPLY IF IT CAN OR NOT?

        if order < 1:
            raise FMUException("The order must be greater than zero.")
        if not can_interpolate_inputs:
            raise FMUException("The FMU does not support input derivatives.")

        if isinstance(variables,str):
            value_refs = N.array([0], dtype=N.uint32, ndmin=1).flatten()
            value_refs[0] = self.get_variable_valueref(variables)
        elif isinstance(variables,list) and N.prod([int(isinstance(v,str)) for v in variables]): #prod equals 0 or 1
            value_refs = N.array([0]*nref, dtype=N.uint32,ndmin=1).flatten()
            for i in range(nref):
                value_refs[i] = self.get_variable_valueref(variables[i])
                orders[i] = order
        else:
            raise FMUException("The variables must either be a string or a list of strings")

        status = FMIL.fmi2_import_set_real_input_derivatives(self._fmu, <FMIL.fmi2_value_reference_t*> value_refs.data, nref,
                                                                <FMIL.fmi2_integer_t*> orders.data, <FMIL.fmi2_real_t*> val.data)

        if status != 0:
            raise FMUException('Failed to set the Real input derivatives.')

    def get_output_derivatives(self, variables, FMIL.fmi2_integer_t order):
        """
        Returns the output derivatives for the specified variables. The
        order specifies the nth-derivative.

        Parameters::

            variables --
                The variables for which the output derivatives
                should be returned.

            order --
                The derivative order.

        Returns::

            The derivatives of the specified order.
        """
        cdef int status
        cdef unsigned int max_output_derivative
        cdef FMIL.size_t  nref
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c']            values
        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1, mode='c'] value_refs
        cdef N.ndarray[FMIL.fmi2_integer_t, ndim=1, mode='c']         orders


        max_output_derivative = FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_maxOutputDerivativeOrder)

        if order < 1 or order > max_output_derivative:
            raise FMUException("The order must be greater than zero and below the maximum output derivative support of the FMU (%d)."%max_output_derivative)

        if isinstance(variables,str):
            nref = 1
            value_refs = N.array([0], dtype=N.uint32, ndmin=1).flatten()
            orders = N.array([order], dtype=N.int32)
            value_refs[0] = self.get_variable_valueref(variables)
        elif isinstance(variables,list) and N.prod([int(isinstance(v,str)) for v in variables]): #prod equals 0 or 1
            nref = len(variables)
            value_refs = N.array([0]*nref, dtype=N.uint32, ndmin=1).flatten()
            orders = N.array([0]*nref, dtype=N.int32)
            for i in range(nref):
                value_refs[i] = self.get_variable_valueref(variables[i])
                orders[i] = order
        else:
            raise FMUException("The variables must either be a string or a list of strings")

        values = N.array([0.0]*nref, dtype=N.float, ndmin=1)

        status = FMIL.fmi2_import_get_real_output_derivatives(self._fmu, <FMIL.fmi2_value_reference_t*> value_refs.data, nref,
                                                            <FMIL.fmi2_integer_t*> orders.data, <FMIL.fmi2_real_t*> values.data)

        if status != 0:
            raise FMUException('Failed to get the Real output derivatives.')

        return values


    def get_status(self, status_kind):
        """
        Retrieves the fmi-status for the the specified fmi-staus-kind.

        Parameters::

            status_kind --
                An integer corresponding to one of the following:
                fmiDoStepStatus       = 0
                fmiPendingStatus      = 1
                fmiLastSuccessfulTime = 2
                fmiTerminated         = 3

        Returns::

            status_ok      = 0
            status_warning = 1
            status_discard = 2
            status_error   = 3
            status_fatal   = 4
            status_pending = 5
        """

        cdef int status
        cdef int fmi_status_kind
        cdef int status_value

        if status_kind >= 0 and status_kind <= 3:
            fmi_status_kind = status_kind
        else:
            raise FMUException('Status kind has to be between 0 and 3')

        status = FMIL.fmi2_import_get_status(self._fmu, fmi_status_kind, &status_value)
        if status != 0:
            raise FMUException('An error occured while retriving the status')

        return status_value

    def get_real_status(self, status_kind):
        """
        Retrieves the status, represented as a real-value,
        for the specified fmi-status-kind.
        See docstring for function get_status() for more
        information about fmi-status-kind.

        Parameters::

            status_kind--
                integer indicating the status kind

        Returns::

            The status.
        """

        cdef int status
        cdef int fmi_status_kind
        cdef FMIL.fmi2_real_t output

        if status_kind >= 0 and status_kind <= 3:
            fmi_status_kind = status_kind
        else:
            raise FMUException('Status kind has to be between 0 and 3')


        status = FMIL.fmi2_import_get_real_status(self._fmu, fmi_status_kind, &output)
        if status != 0:
            raise FMUException('An error occured while retriving the status')

        return output

    def get_integer_status(self, status_kind):
        """
        Retrieves the status, represented as a integer-value,
        for the specified fmi-status-kind.
        See docstring for function get_status() for more
        information about fmi-status-kind.

        Parameters::

            status_kind--
                integer indicating the status kind

        Returns::

            The status.
        """

        cdef int status
        cdef int fmi_status_kind
        cdef FMIL.fmi2_integer_t output

        if status_kind >= 0 and status_kind <= 3:
            fmi_status_kind = status_kind
        else:
            raise FMUException('Status kind has to be between 0 and 3')


        status = FMIL.fmi2_import_get_integer_status(self._fmu, fmi_status_kind, &output)
        if status != 0:
            raise FMUException('An error occured while retriving the status')

        return output

    def get_boolean_status(self, status_kind):
        """
        Retrieves the status, represented as a boolean-value,
        for the specified fmi-status-kind.
        See docstring for function get_status() for more
        information about fmi-status-kind.

        Parameters::

            status_kind--
                integer indicating the status kind

        Returns::

            The status.
        """

        cdef int status
        cdef int fmi_status_kind
        cdef FMIL.fmi2_boolean_t output

        if status_kind >= 0 and status_kind <= 3:
            fmi_status_kind = status_kind
        else:
            raise FMUException('Status kind has to be between 0 and 3')


        status = FMIL.fmi2_import_get_boolean_status(self._fmu, fmi_status_kind, &output)
        if status != 0:
            raise FMUException('An error occured while retriving the status')

        return output

    def get_string_status(self, status_kind):
        """
        Retrieves the status, represented as a string-value,
        for the specified fmi-status-kind.
        See docstring for function get_status() for more
        information about fmi-status-kind.

        Parameters::

            status_kind--
                integer indicating the status kind

        Returns::

            The status.
        """

        cdef int status
        cdef int fmi_status_kind
        cdef FMIL.fmi2_string_t output

        if status_kind >= 0 and status_kind <= 3:
            fmi_status_kind = status_kind
        else:
            raise FMUException('Status kind has to be between 0 and 3')


        status = FMIL.fmi2_import_get_string_status(self._fmu, fmi_status_kind, &output)
        if status != 0:
            raise FMUException('An error occured while retriving the status')

        return output


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

cdef class FMUModelME2(FMUModelBase2):
    """
    Model-exchange model loaded from a dll
    """

    def __init__(self, fmu, path = '.', enable_logging = True, log_file_name = ""):
        """
        Constructor of the model.

        Parameters::

            fmu --
                Name of the fmu as a string.

            path --
                Path to the fmu-directory.
                Default: '.' (working directory)

            enable_logging --
                Boolean for acesss to logging-messages.
                Default: True

            log_file_name --
                Filename for file used to save logmessages.
                Default: "" (Generates automatically)

        Returns::

            A model as an object from the class FMUModelME2
        """
        #Call super
        FMUModelBase2.__init__(self, fmu, path, enable_logging, log_file_name)

        if self._fmu_kind != FMIL.fmi2_fmu_kind_me:
            if self._fmu_kind != FMIL.fmi2_fmu_kind_me_and_cs:
                raise FMUException('This class only supports FMI 2.0 for Model Exchange.')

        if self.get_capability_flags()['me_needsExecutionTool'] == True:
            raise FMUException('Models that need an execution tool are not supported')

        self._modelId = FMIL.fmi2_import_get_model_identifier_ME(self._fmu)
        self.instantiate_model()

    def __dealloc__(self):
        """
        Deallocate memory allocated
        """

        if self._allocated_fmu:
            FMIL.fmi2_import_terminate(self._fmu)
            FMIL.fmi2_import_free_model_instance(self._fmu)

        if self._allocated_dll:
            FMIL.fmi2_import_destroy_dllfmu(self._fmu)

        if self._allocated_xml:
            FMIL.fmi2_import_free(self._fmu)

        if self._allocated_context:
            FMIL.fmi_import_free_context(self._context)

        if self._fmu_temp_dir:
            FMIL.fmi_import_rmdir(&self.callbacks, self._fmu_temp_dir)


    def instantiate_model(self, name= 'Model', visible = False):
        """
        Instantiate the model.

        Parameters::

            name --
                The name of the instance.
                Default: 'Model'

            visible --
                Defines if the simulator application window should be visible or not.
                Default: False, not visible.

        Calls the low-level FMI function: fmiInstantiateModel.
        """

        cdef FMIL.fmi2_boolean_t  log
        cdef FMIL.fmi2_boolean_t  vis
        cdef int status

        if visible:
            vis = 1
        else:
            vis = 0

        status = FMIL.fmi2_import_instantiate_model(self._fmu, name, NULL, vis)

        if status != FMIL.jm_status_success:
            raise FMUException('Failed to instantiate the model.')

    def initialize(self, tolControlled = True, relativeTolerance = None):
        """
        Initializes the model and computes initial values for all variables,
        including setting the start values of variables defined with a the start
        attribute in the XML-file.

        Parameters::

            tolControlled --
                If the model are going to be called by numerical solver using
                step-size control. Boolean flag.
                Default: True

            relativeTolerance --
                If the model are controlled by a numerical solver using
                step-size control, the same tolerance should be provided here.
                Else the default tolerance from the XML-file are used.
                Default: None (default tolerance)

        Calls the low-level FMI function: fmiInitialize.
        """
        cdef FMIL.fmi2_boolean_t tolerance_controlled
        cdef FMIL.fmi2_real_t    tolerance

        #Trying to set the initial time from the xml file, else 0.0
        if self.time == None:
            self.time = FMIL.fmi2_import_get_default_experiment_start(self._fmu)

        if tolControlled:
            tolerance_controlled = 1
            if relativeTolerance == None:
                tolerance = FMIL.fmi2_import_get_default_experiment_tolerance(self._fmu)
            else:
                tolerance = relativeTolerance
        else:
            tolerance_controlled = 0
            tolerance = 0.0

        status = FMIL.fmi2_import_initialize_model(self._fmu, tolerance_controlled, tolerance, &self._eventInfo)

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

    def reset(self):
        """
        This metod resets the FMU by first calling fmiTerminate and
        fmiFreeModelInstance and then reloades the DLL and finally
        reinstantiates using fmiInstantiateModel.
        """
        if self._allocated_fmu:
            FMIL.fmi2_import_terminate(self._fmu)
            FMIL.fmi2_import_free_model_instance(self._fmu)

        if self._allocated_dll:
            FMIL.fmi2_import_destroy_dllfmu(self._fmu)

        status = FMIL.fmi2_import_create_dllfmu(self._fmu, self._fmu_kind, &self.callBackFunctions)
        if status == FMIL.jm_status_error:
            raise FMUException("The DLL could not be reloaded, check the log for more information.")

        #Default values
        self.__t = None

        #Internal values
        self._log = []

        #Instantiates the model
        self.instantiate_model()

    def terminate(self):
        """
        Calls the FMI function fmiTerminate() on the FMU.
        After this call, any call to a function changing the state of the FMU will fail.
        """
        FMIL.fmi2_import_terminate(self._fmu)


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

        cdef int status
        status = FMIL.fmi2_import_set_time(self._fmu, t)

        if status != 0:
            raise FMUException('Failed to set the time.')
        self.__t = t

    time = property(_get_time,_set_time, doc =
    """
    Property for accessing the current time of the simulation. Calls the
    low-level FMI function: fmiSetTime.
    """)


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

    def get_event_indicators(self):
        """
        Returns the event indicators at the current time-point.

        Returns::

            evInd --
                The event indicators as an array.

        Example::

            evInd = model.get_event_indicators()

        Calls the low-level FMI function: fmiGetEventIndicators
        """
        cdef int status
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c'] values = N.empty(self._nEventIndicators, dtype=N.double)

        status = FMIL.fmi2_import_get_event_indicators(self._fmu, <FMIL.fmi2_real_t*> values.data, self._nEventIndicators)

        if status != 0:
            raise FMUException('Failed to get the event indicators.')

        return values

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
        cdef FMIL.fmi2_boolean_t intermediate_result

        if intermediateResult:
            intermediate_result = 1
            status = FMIL.fmi2_import_eventUpdate(self._fmu, intermediate_result, &self._eventInfo)
        else:
            intermediate_result = 0
            status = FMIL.fmi2_import_eventUpdate(self._fmu, intermediate_result, &self._eventInfo)

        if status != 0:
            raise FMUException('Failed to update the events.')

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

        rtol = self.get_default_experiment_tolerance()
        atol = 0.01*rtol*self.nominal_continuous_states

        return [rtol, atol]

    cpdef completed_event_iteration(self):
        """
        This function has to be called by the solver when the global
        event iteration has converged.
        """

        cdef int status

        status = FMIL.fmi2_import_completed_event_iteration(self._fmu)

        if status != 0:
            raise FMUException('An error occured when updating FMU after global event iteration has converged ')

        return status

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
        cdef FMIL.fmi2_boolean_t callEventUpdate

        status = FMIL.fmi2_import_completed_integrator_step(self._fmu, &callEventUpdate)

        if status != 0:
            raise FMUException('Failed to call FMI Completed Step.')

        if callEventUpdate == 1:
            return True
        else:
            return False


    def _get_continuous_states(self):
        """
        Returns a vector with the values of the continuous states.

        Returns::

            The continuous states.
        """
        cdef int status
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c'] ndx = N.zeros(self._nContinuousStates, dtype=N.double)
        status = FMIL.fmi2_import_get_continuous_states(self._fmu, <FMIL.fmi2_real_t*> ndx.data ,self._nContinuousStates)

        if status != 0:
            raise FMUException('Failed to retrieve the continuous states.')

        return ndx

    def _set_continuous_states(self, N.ndarray[FMIL.fmi2_real_t] values):
        """
        Set the values of the continuous states.

        Parameters::

            values--
                The new values of the continuous states.
        """
        cdef int status
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1,mode='c'] ndx = values

        if ndx.size != self._nContinuousStates:
            raise FMUException(
                'Failed to set the new continuous states. ' \
                'The number of values are not consistent with the number of '\
                'continuous states.')

        status = FMIL.fmi2_import_set_continuous_states(self._fmu, <FMIL.fmi2_real_t*> ndx.data , self._nContinuousStates)

        if status >= 3:
            raise FMUException('Failed to set the new continuous states.')

    continuous_states = property(_get_continuous_states, _set_continuous_states,
        doc=
    """
    Property for accessing the current values of the continuous states. Calls
    the low-level FMI function: fmiSetContinuousStates/fmiGetContinuousStates.
    """)

    def _get_nominal_continuous_states(self):
        """
        Returns the nominal values of the continuous states.

        Returns::
            The nominal values.
        """
        cdef int status
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c'] ndx = N.zeros(self._nContinuousStates, dtype=N.double)

        status = FMIL.fmi2_import_get_nominal_continuous_states(
                self._fmu, <FMIL.fmi2_real_t*> ndx.data, self._nContinuousStates)

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
                The derivatives as an array.

        Example::

            dx = model.get_derivatives()

        Calls the low-level FMI function: fmiGetDerivatives
        """
        cdef int status
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c'] values = N.empty(self._nContinuousStates, dtype = N.double)

        status = FMIL.fmi2_import_get_derivatives(self._fmu, <FMIL.fmi2_real_t*> values.data, self._nContinuousStates)

        if status != 0:
            raise FMUException('Failed to get the derivative values.')

        return values


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

def load_fmu_deprecated(fmu, path='.', enable_logging=True, log_file_name=""):
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
    callbacks.malloc    = FMIL.malloc
    callbacks.calloc    = FMIL.calloc
    callbacks.realloc   = FMIL.realloc
    callbacks.free      = FMIL.free
    callbacks.logger    = importlogger_load_fmu
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

        if enable_logging:
            raise FMUException("The FMU version could not be determined. "+last_error)
        else:
            raise FMUException("The FMU version could not be determined. Enable logging for possibly more information.")
    if version != 1:
        #Delete the context
        if allocated_context:
            FMIL.fmi_import_free_context(context)

        raise FMUException("PyFMI currently only supports FMI 1.0.")

    #Parse the XML
    _fmu = FMIL.fmi1_import_parse_xml(context, fmu_temp_dir)
    if _fmu == NULL:
        last_error = FMIL.jm_get_last_error(&callbacks)
        if enable_logging:
            raise FMUException("The XML file could not be parsed. "+last_error)
        else:
            raise FMUException("The XML file could not be parsed. Enable logging for possibly more information.")
    allocated_xml = True

    #Check the FMU kind
    fmu_kind = FMIL.fmi1_import_get_fmu_kind(_fmu)
    if fmu_kind == FMI_ME:
        model = FMUModelME1(fmu, path, enable_logging, log_file_name)
    elif fmu_kind == FMI_CS_STANDALONE:
        model = FMUModelCS1(fmu, path, enable_logging, log_file_name)
    else:
        #Delete the XML
        if allocated_xml:
            FMIL.fmi1_import_free(_fmu)

        #Delete the context
        if allocated_context:
            FMIL.fmi_import_free_context(context)

        raise FMUException("PyFMI currently only supports FMI 1.0.")

    #Delete the XML
    if allocated_xml:
        FMIL.fmi1_import_free(_fmu)

    #Delete the context
    if allocated_context:
        FMIL.fmi_import_free_context(context)

    #Delete the created directory
    delete_temp_dir(fmu_temp_dir)

    return model

def load_fmu(fmu, path = '.', enable_logging = True, log_file_name = "", kind = 'auto'):
    """
    Helper method for creating a model instance.

    Parameters::

        fmu --
            Name of the fmu as a string.

        path --
            Path to the fmu-directory.
            Default: '.' (working directory)

        enable_logging --
            Boolean for acesss to logging-messages.
            Default: True

        log_file_name --
            Filename for file used to save logmessages.
            Default: "" (Generates automatically)

        kind --
            String indicating the kind of model to create. This is only
            needed if a FMU contains both a ME and CS model.
            Availible options:
                - 'ME'
                - 'CS'
                - 'auto'
            Default: 'auto' (Chooses ME before CS if both availible)

    Returns::

        A model instance corresponding to the loaded FMU.
    """

    #NOTE: This method can be made more efficient by providing
    #the unzipped part and the already read XML object to the different
    #FMU classes.

    #FMIL related variables
    cdef FMIL.fmi_import_context_t*     context
    cdef FMIL.jm_callbacks              callbacks
    #cdef FMIL.fmi2_xml_callbacks_t      xml_callbacks
    cdef FMIL.fmi1_callback_functions_t callBackFunctions_1
    cdef FMIL.fmi2_callback_functions_t callBackFunctions_2
    cdef FMIL.jm_string                 last_error
    cdef FMIL.fmi_version_enu_t         version
    cdef FMIL.fmi1_import_t*            fmu_1
    cdef FMIL.fmi2_import_t*            fmu_2
    cdef FMIL.fmi1_fmu_kind_enu_t       fmu_1_kind
    cdef FMIL.fmi2_fmu_kind_enu_t       fmu_2_kind

    #Variables for deallocation
    fmu_temp_dir = None
    model        = None

    # Check that the file referenced by fmu has the correct file-ending
    fmu_full_path = os.path.abspath(os.path.join(path,fmu))
    if not fmu_full_path.endswith('.fmu'):
        raise FMUException("FMUModel must be instantiated with an FMU (.fmu) file.")

    #Check that the file exists
    if not os.path.isfile(fmu_full_path):
        raise FMUException('Could not locate the FMU in the specified directory.')

    #Check that kind-argument is well-defined
    if not kind.lower() == 'auto':
        if (kind.upper() != 'ME' and kind.upper() != 'CS'):
            raise FMUException('Input-argument "kind" can only be "ME", "CS" or "auto" (default) and not: ' + kind)


    #Specify FMI related callbacks
    callbacks.malloc    = FMIL.malloc
    callbacks.calloc    = FMIL.calloc
    callbacks.realloc   = FMIL.realloc
    callbacks.free      = FMIL.free
    callbacks.logger    = importlogger_load_fmu
    callbacks.log_level = FMIL.jm_log_level_warning if enable_logging else FMIL.jm_log_level_nothing


    #Specify the xml_callbacks for FMU2
    #xml_callbacks.startHandle = None
    #xml_callbacks.dataHandle  = None
    #xml_callbacks.endHandle   = None
    #xml_callbacks.context     = None

    #Specify the general FMU1 callback functions
    callBackFunctions_1.logger         = FMIL.fmi1_log_forwarding
    callBackFunctions_1.allocateMemory = FMIL.calloc
    callBackFunctions_1.freeMemory     = FMIL.free

    #Specify the general FMU2 callback functions
    callBackFunctions_2.logger               = FMIL.fmi2_log_forwarding
    callBackFunctions_2.allocateMemory       = FMIL.calloc
    callBackFunctions_2.freeMemory           = FMIL.free


    # Create a struct for allocation
    context = FMIL.fmi_import_allocate_context(&callbacks)

    #Get the FMI version of the provided model
    fmu_temp_dir = create_temp_dir()
    version = FMIL.fmi_import_get_fmi_version(context, fmu_full_path, fmu_temp_dir)


    #Check the version
    if version == FMIL.fmi_version_unknown_enu:
        #Delete context
        last_error = FMIL.jm_get_last_error(&callbacks)
        FMIL.fmi_import_free_context(context)
        FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
        if enable_logging:
            raise FMUException("The FMU version could not be determined. "+last_error)
        else:
            raise FMUException("The FMU version could not be determined. Enable logging for possibly more information.")

    if version > 2:
        #Delete the context
        last_error = FMIL.jm_get_last_error(&callbacks)
        FMIL.fmi_import_free_context(context)
        FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
        if enable_logging:
            raise FMUException("The FMU version is unsupported. "+last_error)
        else:
            raise FMUException("The FMU version is unsupported. Enable logging for possibly more information.")


    #Parse the xml
    if version == FMIL.fmi_version_1_enu:
        #Check the fmu-kind
        fmu_1 = FMIL.fmi1_import_parse_xml(context, fmu_temp_dir)

        if fmu_1 is NULL:
            #Delete the context
            last_error = FMIL.jm_get_last_error(&callbacks)
            FMIL.fmi_import_free_context(context)
            FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
            if enable_logging:
                raise FMUException("The XML-could not be read. "+last_error)
            else:
                raise FMUException('The XML-could not be read. Enable logging for possible nore information.')

        fmu_1_kind = FMIL.fmi1_import_get_fmu_kind(fmu_1)

        #Compare fmu_kind with input-specified kind
        if fmu_1_kind == FMI_ME and kind.upper() != 'CS':
            model=FMUModelME1(fmu, path, enable_logging, log_file_name)
        elif fmu_1_kind == FMI_CS_STANDALONE and kind.upper() != 'ME':
            model=FMUModelCS1(fmu, path, enable_logging, log_file_name)
        elif fmu_1_kind == FMIL.fmi1_fmu_kind_enu_cs_tool:
            FMIL.fmi1_import_free(fmu_1)
            FMIL.fmi_import_free_context(context)
            FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
            raise FMUException("PyFMI does not support co-simulation tool")
        else:
            FMIL.fmi1_import_free(fmu_1)
            FMIL.fmi_import_free_context(context)
            FMIL.fmi_import_rmdir(&callbacks,fmu_temp_dir)
            raise FMUException('FMU is a ' + FMIL.fmi1_fmu_kind_to_string(fmu_1_kind) + ' and not a ' + kind.upper())

    elif version == FMIL.fmi_version_2_0_enu:
        #Check fmu-kind and compare with input-specified kind
        fmu_2 = FMIL.fmi2_import_parse_xml(context, fmu_temp_dir, NULL)

        if fmu_2 is NULL:
            #Delete the context
            last_error = FMIL.jm_get_last_error(&callbacks)
            FMIL.fmi_import_free_context(context)
            FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
            if enable_logging:
                raise FMUException("The XML-could not be read. "+last_error)
            else:
                raise FMUException('The XML-could not be read. Enable logging for possible nore information.')

        fmu_2_kind = FMIL.fmi2_import_get_fmu_kind(fmu_2)

        #FMU kind is unknown
        if fmu_2_kind == FMIL.fmi2_fmu_kind_unknown:
            last_error = FMIL.jm_get_last_error(&callbacks)
            FMIL.fmi2_import_free(fmu_2)
            FMIL.fmi_import_free_context(context)
            FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
            if enable_logging:
                raise FMUException("The FMU kind could not be determined. "+last_error)
            else:
                raise FMUException("The FMU kind could not be determined. Enable logging for possibly more information.")

        #FMU kind is known
        if kind.lower() == 'auto':
            if fmu_2_kind == FMIL.fmi2_fmu_kind_cs:
                model = FMUModelCS2(fmu, path, enable_logging, log_file_name)
            elif fmu_2_kind == FMIL.fmi2_fmu_kind_me or fmu_2_kind == FMIL.fmi2_fmu_kind_me_and_cs:
                model = FMUModelME2(fmu, path, enable_logging, log_file_name)
        elif kind.upper() == 'CS':
            if fmu_2_kind == FMIL.fmi2_fmu_kind_cs or fmu_2_kind == FMIL.fmi2_fmu_kind_me_and_cs:
                model = FMUModelCS2(fmu, path, enable_logging, log_file_name)
        elif kind.upper() == 'ME':
            if fmu_2_kind == FMIL.fmi2_fmu_kind_me or fmu_2_kind == FMIL.fmi2_fmu_kind_me_and_cs:
                model = FMUModelME2(fmu, path, enable_logging, log_file_name)

        #Could not match FMU kind with input-specified kind
        if model is None:
            FMIL.fmi2_import_free(fmu_2)
            FMIL.fmi_import_free_context(context)
            FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
            raise FMUException('FMU is a ' + FMIL.fmi2_fmu_kind_to_string(fmu_2_kind) + ' and not a ' + kind.upper())

    else:
        #This else-statement ensures that the variables "context" and "version" are defined before proceeding

        #Delete the context
        last_error = FMIL.jm_get_last_error(&callbacks)
        FMIL.fmi_import_free_context(context)
        FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
        if enable_logging:
            raise FMUException("The FMU version is not found. "+last_error)
        else:
            raise FMUException("The FMU version is not found. Enable logging for possibly more information.")


    #Delete
    if version == FMIL.fmi_version_1_enu:
        FMIL.fmi1_import_free(fmu_1)
        FMIL.fmi_import_free_context(context)
        FMIL.fmi_import_rmdir(&callbacks,fmu_temp_dir)

    if version == FMIL.fmi_version_2_0_enu:
        FMIL.fmi2_import_free(fmu_2)
        FMIL.fmi_import_free_context(context)
        FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)


    return model




