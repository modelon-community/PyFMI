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
Module containing the FMI interface Python wrappers.
"""
"""
For profiling:
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
import os
import sys
import logging
import fnmatch
import re
from collections import OrderedDict
cimport cython

import scipy.sparse as sp
import numpy as N
cimport numpy as N
from numpy cimport PyArray_DATA

N.import_array()

cimport fmil_import as FMIL

from pyfmi.common.core import create_temp_dir, delete_temp_dir
from pyfmi.common.core import create_temp_file, delete_temp_file
#from pyfmi.common.core cimport BaseModel

#from pyfmi.common import python3_flag, encode, decode
from pyfmi.fmi_util import cpr_seed, enable_caching, python3_flag
from pyfmi.fmi_util cimport encode, decode

int   = N.int32
N.int = N.int32

"""Basic flags related to FMI"""

FMI_TRUE  = '\x01'
FMI_FALSE = '\x00'

FMI2_TRUE = FMIL.fmi2_true
FMI2_FALSE = FMIL.fmi2_false

# Status
FMI_OK      = FMIL.fmi1_status_ok
FMI_WARNING = FMIL.fmi1_status_warning
FMI_DISCARD = FMIL.fmi1_status_discard
FMI_ERROR   = FMIL.fmi1_status_error
FMI_FATAL   = FMIL.fmi1_status_fatal
FMI_PENDING = FMIL.fmi1_status_pending

FMI1_DO_STEP_STATUS       = FMIL.fmi1_do_step_status
FMI1_PENDING_STATUS       = FMIL.fmi1_pending_status
FMI1_LAST_SUCCESSFUL_TIME = FMIL.fmi1_last_successful_time

FMI2_DO_STEP_STATUS       = FMIL.fmi2_do_step_status
FMI2_PENDING_STATUS       = FMIL.fmi2_pending_status
FMI2_LAST_SUCCESSFUL_TIME = FMIL.fmi2_last_successful_time
FMI2_TERMINATED           = FMIL.fmi2_terminated

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

FMI2_INPUT     = FMIL.fmi2_causality_enu_input
FMI2_OUTPUT    = FMIL.fmi2_causality_enu_output
FMI2_PARAMETER = FMIL.fmi2_causality_enu_parameter
FMI2_CALCULATED_PARAMETER = FMIL.fmi2_causality_enu_calculated_parameter
FMI2_LOCAL       = FMIL.fmi2_causality_enu_local
FMI2_INDEPENDENT = FMIL.fmi2_causality_enu_independent

# FMI types
FMI_ME                 = FMIL.fmi1_fmu_kind_enu_me
FMI_CS_STANDALONE      = FMIL.fmi1_fmu_kind_enu_cs_standalone
FMI_CS_TOOL            = FMIL.fmi1_fmu_kind_enu_cs_tool
FMI_MIME_CS_STANDALONE = encode("application/x-fmu-sharedlibrary")

FMI_REGISTER_GLOBALLY = 1
FMI_DEFAULT_LOG_LEVEL = FMIL.jm_log_level_error

# INITIAL
FMI2_INITIAL_EXACT      = 0
FMI2_INITIAL_APPROX     = 1
FMI2_INITIAL_CALCULATED = 2
FMI2_INITIAL_UNKNOWN    = 3

DEF FORWARD_DIFFERENCE = 1
DEF CENTRAL_DIFFERENCE = 2
FORWARD_DIFFERENCE_EPS = (N.finfo(float).eps)**0.5
CENTRAL_DIFFERENCE_EPS = (N.finfo(float).eps)**(1/3.0)

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
cdef void importlogger(FMIL.jm_callbacks* c, FMIL.jm_string module, FMIL.jm_log_level_enu_t log_level, FMIL.jm_string message):
    if c.context != NULL:
        (<FMUModelBase>c.context)._logger(module,log_level,message)
 
#CALLBACKS
cdef void importlogger2(FMIL.jm_callbacks* c, FMIL.jm_string module, FMIL.jm_log_level_enu_t log_level, FMIL.jm_string message):
    if c.context != NULL:
        (<FMUModelBase2>c.context)._logger(module, log_level, message)

#CALLBACKS
cdef void importlogger_load_fmu(FMIL.jm_callbacks* c, FMIL.jm_string module, FMIL.jm_log_level_enu_t log_level, FMIL.jm_string message):
    (<list>c.context).append("FMIL: module = %s, log level = %d: %s"%(module, log_level, message))

cdef class ModelBase:
    """
    Abstract Model class containing base functionality.
    """
    
    def __init__(self):
        self.cache = {}
        self.file_object = None
        self._additional_logger = None
        self._current_log_size = 0
        self._max_log_size = 1024**3*2 #About 2GB limit
        self._max_log_size_msg_sent = False
    
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

            model.set('damper.d', 1.1)
            model.set(['damper.d','gear.a'], [1.1, 10])
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
            model.get('damper.d')
            # Returns a list of the variables
            model.get(['damper.d','gear.a'])
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
        algdrive = __import__(base_path, globals(), locals(), [], 1)
        AlgorithmBase = getattr(getattr(algdrive,"algorithm_drivers"), 'AlgorithmBase')

        if isinstance(algorithm, basestring):
            module = __import__(module, globals(), locals(), [algorithm], 0)
            algorithm = getattr(module, algorithm)

        if not issubclass(algorithm, AlgorithmBase):
            raise Exception(str(algorithm)+
            " must be a subclass of common.algorithm_drivers.AlgorithmBase")
        
        #open log file
        self._open_log_file()
        
        try:
            # initialize algorithm
            alg = algorithm(start_time, final_time, input, self, options)
            # simulate
            alg.solve()
        except:
            #close log file
            self._close_log_file()
            raise #Reraise the exception
        
        #close log file
        self._close_log_file()
        
        # get and return result
        return alg.get_result()
        
    def _exec_estimate_algorithm(self, parameters,
                                       measurements,
                                       input,
                                       module,
                                       algorithm,
                                       options):
        """
        Helper function which performs all steps of an algorithm run which are
        common to all estimation algorithms.

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
        self._open_log_file()
        
        try:
            # initialize algorithm
            alg = algorithm(parameters, measurements, input, self, options)
            # simulate
            alg.solve()
        except:
            #close log file
            self._close_log_file()
            raise #Reraise the exception
        
        #close log file
        self._close_log_file()
        
        # get and return result
        return alg.get_result()


    def _default_options(self, module, algorithm):
        """
        Help method. Gets the options class for the algorithm specified in
        'algorithm'.
        """
        module = __import__(module, globals(), locals(), [algorithm], 0)
        algorithm = getattr(module, algorithm)

        return algorithm.get_default_options()
        
    def estimate(self,
                 parameters,
                 measurements,
                 input=(),
                 algorithm='SciEstAlg',
                 options={}):
        """
        Compact function for model estimation.

        The estimation method depends on which algorithm is used, this can be
        set with the function argument 'algorithm'. Options for the algorithm
        are passed as option classes or as pure dicts. See
        model.estimate_options for more details.

        The default algorithm for this function is SciEstAlg.

        Parameters::

            parameters --
                The tunable parameters (required)

            measurements --
                The measurements data (name, data). Note that the measurements
                needs to be distinct and equally spaced. (required)

            input --
                Input signal for the estimation. The input should be a 2-tuple
                consisting of first the names of the input variable(s) and then
                the data matrix.
                Default: Empty tuple.

            algorithm --
                The algorithm which will be used for the estimation is specified
                by passing the algorithm class as string or class object in this
                argument. 'algorithm' can be any class which implements the
                abstract class AlgorithmBase (found in algorithm_drivers.py). In
                this way it is possible to write own algorithms and use them
                with this function.
                Default: 'SciEstAlg'

            options --
                The options that should be used in the algorithm. For details on
                the options do:

                    >> myModel = load_fmu(...)
                    >> opts = myModel.estimate_options()
                    >> opts?

                Valid values are:
                    - A dict which gives SciEstAlgOptions with
                      default values on all options except the ones
                      listed in the dict. Empty dict will thus give all
                      options with default values.
                    - An options object.
                Default: Empty dict

        Returns::

            Result object, subclass of common.algorithm_drivers.ResultBase.
        """

        return self._exec_estimate_algorithm(parameters,
                                             measurements,
                                             input,
                                             'pyfmi.fmi_algorithm_drivers',
                                             algorithm,
                                             options)

    def estimate_options(self, algorithm='SciEstAlg'):
        """
        Get an instance of the estimation options class, filled with default
        values. If called without argument then the options class for the
        default estimation algorithm will be returned.

        Parameters::

            algorithm --
                The algorithm for which the options class should be fetched.
                Possible values are: 'SciEstAlg'.
                Default: 'SciEstAlg'

        Returns::

            Options class for the algorithm specified with default values.
        """
        return self._default_options('pyfmi.fmi_algorithm_drivers', algorithm)

    def get_log_filename(self):
        return decode(self._fmu_log_name)
    
    def get_log_file_name(self):
        logging.warning("The method 'get_log_file_name()' is deprecated and will be removed. Please use 'get_log_filename()' instead.")
        return self.get_log_filename()
        
    def get_number_of_lines_log(self):
        """
        Returns the number of lines in the log file.
        """
        num_lines = 0
        if self._fmu_log_name != NULL:
            with open(self._fmu_log_name,'r') as file:
                num_lines = sum(1 for line in file)
        return num_lines
        
    def print_log(self, start_lines=-1, end_lines=-1):
        """
        Prints the log information to the prompt.
        """
        cdef int N
        log = self.get_log(start_lines, end_lines)
        N = len(log)

        for i in range(N):
            print(log[i])
            
    def get_log(self, int start_lines=-1, int end_lines=-1):
        """
        Returns the log information as a list. To turn on the logging 
        use load_fmu(..., log_level=1-7) in the loading of the FMU. The 
        log is stored as a list of lists. For example log[0] are the 
        first log message to the log and consists of, in the following 
        order, the instance name, the status, the category and the 
        message.

        Returns::

            log - A list of lists.
        """
        cdef int i = 0
        cdef int num_lines = 0
        log = []
        
        if end_lines != -1:
            num_lines = self.get_number_of_lines_log()
        
        if self._fmu_log_name != NULL:
            with open(self._fmu_log_name,'r') as file:
                while True:
                    i = i + 1
                    line = file.readline()
                    if line == "":
                        break
                    if start_lines != -1 and end_lines == -1:
                        if i > start_lines:
                            break
                    elif start_lines == -1 and end_lines != -1:
                        if i < num_lines - end_lines + 1:
                            continue
                    elif start_lines != -1 and end_lines != -1:
                        if i > start_lines and i < num_lines - end_lines + 1:
                            continue
                    log.append(line.strip("\n"))
        return log
    
    def _log_open(self):
        if self.file_object:
            return True
        else:
            return False
    
    def _open_log_file(self):
        if self._fmu_log_name != NULL:
            self.file_object = open(self._fmu_log_name,'a')

    def _close_log_file(self):
        if self.file_object:
            self.file_object.close()
            self.file_object = None

    cdef _logger(self, FMIL.jm_string c_module, int log_level, FMIL.jm_string c_message) with gil:
        module  = decode(c_module)
        message = decode(c_message)
        
        if self._additional_logger:
            self._additional_logger(module, log_level, message)
            
        msg = "FMIL: module = %s, log level = %d: %s\n"%(module, log_level, message)
        
        self._current_log_size = self._current_log_size + len(msg)
        if self._current_log_size > self._max_log_size:
            if self._max_log_size_msg_sent:
                return
            
            msg = "The log file has reached its maximum size and further log messages will not be saved. To change the maximum size of the file, please use the 'set_max_log_size' method."
            self._max_log_size_msg_sent = True
            
        if self._fmu_log_name != NULL:
            if self.file_object:
                self.file_object.write(msg)
            else:
                with open(self._fmu_log_name,'a') as file:
                    file.write(msg)
        else:
            self._log.append([module,log_level,message])
            
    def append_log_message(self, module, log_level, message):
        if self._additional_logger:
            self._additional_logger(module, log_level, message)
        
        if self._fmu_log_name != NULL:
            if self.file_object:
                self.file_object.write("FMIL: module = %s, log level = %d: %s\n"%(module, log_level, message))
            else:
                with open(self._fmu_log_name,'a') as file:
                    file.write("FMIL: module = %s, log level = %d: %s\n"%(module, log_level, message))
        else:
            self._log.append([module,log_level,message])
    
    def set_max_log_size(self, number_of_characters):
        """
        Specifies the maximum number of characters to be written to the
        log file.
        
            Parameters::
            
                number_of_characters --
                    The maximum number of characters in the log.
                    Default: 1024^3 (about 1GB)
        """
        self._max_log_size = number_of_characters
        
    def get_max_log_size(self):
        """
        Returns the limit (in characters) of the log file.
        """
        return self._max_log_size
    
            
    def set_log_level(self, FMIL.jm_log_level_enu_t level):
        """
        Specifies the log level for PyFMI. Note that this is
        different from the FMU logging which is specified via
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
        if level < FMIL.jm_log_level_nothing or level > FMIL.jm_log_level_all:
            raise FMUException("Invalid log level for FMI Library (0-7).")
        self.callbacks.log_level = level

    def get_log_level(self):
        """
        Returns the log level for PyFMI, i.e., as set in set_log_level.
        Type: int
        """
        return self.callbacks.log_level

    def set_additional_logger(self, additional_logger):
        """
        Set an additional logger function that will, an addition to
        the normal FMU log file, also be fed with all model log
        messages.

        Parameter::

            additional_logger --
                The callback function that should accept three arguments:
                module(str), log_level(int), message(str)
        """
        self._additional_logger = additional_logger
        
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
            regex = ""
            for i in expression:
                regex = regex + fnmatch.translate(i) + "|"
            regexp = [re.compile(regex[:-1])]
        else:
            raise FMUException("Unknown input.")
        return regexp


class FMUException(Exception):
    """
    An FMU exception.
    """
    pass
    
class TimeLimitExceeded(FMUException):
    pass

class PyEventInfo():
    pass


cdef class ScalarVariable:
    """
    Class defining data structure based on the XML element ScalarVariable.
    """
    def __init__(self, name, value_reference, type, description="",
                       variability=FMIL.fmi1_variability_enu_continuous,
                       causality=FMIL.fmi1_causality_enu_internal,
                       alias=FMIL.fmi1_variable_is_not_alias):
        """
        Class collecting information about a scalar variable and its
        attributes. The following attributes can be retrieved::
        
            name
            value_reference
            type
            description
            variability
            causality
            alias
            
        For further information about the attributes, see the info on a
        specific attribute.
        """

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

            The data type attribute value as enumeration: FMI_REAL(0),
            FMI_INTEGER(1), FMI_BOOLEAN(2), FMI_STRING(3) or FMI_ENUMERATION(4).
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
            FMI_CONSTANT(0), FMI_PARAMETER(1), FMI_DISCRETE(2) or FMI_CONTINUOUS(3).
        """
        return self._variability
    variability = property(_get_variability)

    def _get_causality(self):
        """
        Get the value of the causality attribute.

        Returns::

            The causality attribute value as enumeration: FMI_INPUT(0),
            FMI_OUTPUT(1), FMI_INTERNAL(2) or FMI_NONE(3).
        """
        return self._causality
    causality = property(_get_causality)

    def _get_alias(self):
        """
        Get the value of the alias attribute.

        Returns::

            The alias attribute value as enumeration: FMI_NO_ALIAS(0),
            FMI_ALIAS(1) or FMI_NEGATED_ALIAS(-1).
        """
        return self._alias
    alias = property(_get_alias)

cdef class ScalarVariable2:
    """
    Class defining data structure based on the XML element ScalarVariable.
    """
    def __init__(self, name, value_reference, type, description = "",
                       variability = FMIL.fmi2_variability_enu_unknown,
                       causality   = FMIL.fmi2_causality_enu_unknown,
                       alias       = FMIL.fmi2_variable_is_not_alias,
                       initial     = FMIL.fmi2_initial_enu_unknown):
        """
        Class collecting information about a scalar variable and its
        attributes. The following attributes can be retrieved::
        
            name
            value_reference
            type
            description
            variability
            causality
            alias
            initial
            
        For further information about the attributes, see the info on a
        specific attribute.
        """

        self._name            = name
        self._value_reference = value_reference
        self._type            = type
        self._description     = description
        self._variability     = variability
        self._causality       = causality
        self._alias           = alias
        self._initial         = initial

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

            The data type attribute value as enumeration: FMI2_REAL(0),
            FMI2_INTEGER(1), FMI2_BOOLEAN(2), FMI2_STRING(3) or FMI2_ENUMERATION(4).
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

            The variability of the variable: FMI2_CONSTANT(0), FMI2_FIXED(1),
            FMI2_TUNABLE(2), FMI2_DISCRETE(3), FMI2_CONTINUOUS(4) or FMI2_UNKNOWN(5)
        """
        return self._variability
    variability = property(_get_variability)

    def _get_causality(self):
        """
        Get the value of the causality attribute.

        Returns::

            The causality of the variable, FMI2_PARAMETER(0), FMI2_CALCULATED_PARAMETER(1), FMI2_INPUT(2),
            FMI2_OUTPUT(3), FMI2_LOCAL(4), FMI2_INDEPENDENT(5), FMI2_UNKNOWN(6)
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
    
    def _get_initial(self):
        """
        Get the value of the initial attribute.

        Returns::

            The initial attribute value as enumeration: FMI2_INITIAL_EXACT, 
                              FMI2_INITIAL_APPROX, FMI2_INITIAL_CALCULATED, 
                              FMI2_INITIAL_UNKNOWN    
        """
        return self._initial
    initial = property(_get_initial)

cdef class DeclaredType2:
    """
    Class defining data structure based on the XML element Type.
    """
    def __init__(self, name, description = "", quantity = ""):
        self._name        = name
        self._description = description
        self._quantity = quantity
    
    def _get_name(self):
        """
        Get the value of the name attribute.

        Returns::

            The name attribute value as string.
        """
        return self._name
    name = property(_get_name)
    
    def _get_description(self):
        """
        Get the value of the description attribute.

        Returns::

            The description attribute value as string (empty string if
            not set).
        """
        return self._description
    description = property(_get_description)

cdef class EnumerationType2(DeclaredType2):
    """
    Class defining data structure based on the XML element Enumeration.
    """
    def __init__(self, name, description = "", quantity = "", items = None):
        DeclaredType2.__init__(self, name, description, quantity)
        
        self._items    = items
    
    def _get_quantity(self):
        """
        Get the quantity of the enumeration type.

        Returns::

            The quantity as string (empty string if
            not set).
        """
        return self._quantity
    quantity = property(_get_quantity)
    
    def _get_items(self):
        """
        Get the items of the enumeration type.
        
        Returns::
        
            The items of the enumeration type as a dict. The key is the
            enumeration value and the dict value is a tuple containing
            the name and description of the enumeration item.
        """
        return self._items
    items = property(_get_items)

cdef class IntegerType2(DeclaredType2):
    """
    Class defining data structure based on the XML element Enumeration.
    """
    def __init__(self, name, description = "", quantity = "", min = -N.inf, max = N.inf):
        DeclaredType2.__init__(self, name, description, quantity)
        
        self._min = min
        self._max = max
        
    def _get_max(self):
        """
        Get the max value for the type.
        
        Returns::
        
            The max value.
        """
        return self._max
    max = property(_get_max)
    
    def _get_min(self):
        """
        Get the min value for the type.
        
        Returns::
        
            The min value.
        """
        return self._min
    min = property(_get_min)

cdef class RealType2(DeclaredType2):
    """
    Class defining data structure based on the XML element Enumeration.
    """
    def __init__(self, name, description = "", quantity = "", min = -N.inf, max = N.inf, nominal = 1.0, unbounded = False,
                relative_quantity = False, display_unit = "", unit = ""):
        DeclaredType2.__init__(self, name, description, quantity)
        
        self._min = min
        self._max = max
        self._nominal = nominal
        self._unbounded = unbounded
        self._relative_quantity = relative_quantity
        self._display_unit = display_unit
        self._unit = unit
    
    def _get_max(self):
        """
        Get the max value for the type.
        
        Returns::
        
            The max value.
        """
        return self._max
    max = property(_get_max)
    
    def _get_min(self):
        """
        Get the min value for the type.
        
        Returns::
        
            The min value.
        """
        return self._min
    min = property(_get_min)
    
    def _get_nominal(self):
        """
        Get the nominal value for the type.
        
        Returns::
        
            The nominal value.
        """
        return self._nominal
    nominal = property(_get_nominal)
    
    def _get_unbounded(self):
        """
        Get the unbounded value for the type.
        
        Returns::
        
            The unbounded value.
        """
        return self._unbounded
    unbounded = property(_get_unbounded)
    
    def _get_relative_quantity(self):
        """
        Get the relative quantity value for the type.
        
        Returns::
        
            The relative quantity value.
        """
        return self._relative_quantity
    relative_quantity = property(_get_relative_quantity)
    
    def _get_display_unit(self):
        """
        Get the display unit value for the type.
        
        Returns::
        
            The display unit value.
        """
        return self._display_unit
    display_unit = property(_get_display_unit)
    
    def _get_unit(self):
        """
        Get the unit value for the type.
        
        Returns::
        
            The unit value.
        """
        return self._unit
    unit = property(_get_unit)
    
cdef class FMUState2:
    """
    Class containing a pointer to a FMU-state.
    """
    def __init__(self):
        self.fmu_state = NULL



cdef class FMUModelBase(ModelBase):
    """
    An FMI Model loaded from a DLL.
    """
    def __init__(self, fmu, path='.', enable_logging=None, log_file_name="", log_level=FMI_DEFAULT_LOG_LEVEL, _unzipped_dir=None, _connect_dll=True):
        """
        Constructor.
        """
        cdef int status
        cdef int version
        
        #Call super
        ModelBase.__init__(self)

        #Contains the log information
        self._log = []

        #Used for deallocation
        self._allocated_context = 0
        self._allocated_dll     = 0
        self._allocated_xml     = 0
        self._allocated_fmu     = 0
        self._instantiated_fmu  = 0
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
        
        if enable_logging==None:
            if log_level >= FMIL.jm_log_level_nothing and log_level <= FMIL.jm_log_level_all:
                if log_level == FMIL.jm_log_level_nothing:
                    enable_logging = False
                else:
                    enable_logging = True
                self.callbacks.log_level = log_level
            else:
                raise FMUException("The log level must be between %d and %d"%(FMIL.jm_log_level_nothing, FMIL.jm_log_level_all))
        else:
            logging.warning("The attribute 'enable_logging' is deprecated. Please use 'log_level' instead. Setting 'log_level' to INFO...")
            self.callbacks.log_level = FMIL.jm_log_level_info if enable_logging else FMIL.jm_log_level_error

        fmu_full_path = os.path.abspath(os.path.join(path,fmu))
        if _unzipped_dir:
            fmu_temp_dir = encode(_unzipped_dir)
        else:
            fmu_temp_dir  = encode(create_temp_dir())
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
        fmu_full_path = encode(fmu_full_path)
        
        if _unzipped_dir:
            #If the unzipped directory is provided we assume that the version
            #is correct. This is due to that the method to get the version
            #unzipps the FMU which we already have done.
            version = FMIL.fmi_version_1_enu
        else:
            version = FMIL.fmi_import_get_fmi_version(self.context, fmu_full_path, fmu_temp_dir)
        self._version = version #Store version

        if version == FMIL.fmi_version_unknown_enu:
            last_error = decode(FMIL.jm_get_last_error(&self.callbacks))
            if self.callbacks.log_level >= FMIL.jm_log_level_error:
                raise FMUException("The FMU version could not be determined. "+last_error)
            else:
                raise FMUException("The FMU version could not be determined. Enable logging for possibly more information.")
        if version != 1:
            raise FMUException("This class only supports FMI 1.0 (Model Exchange and Co-Simulation).")

        #Parse the XML
        self._fmu = FMIL.fmi1_import_parse_xml(self.context, fmu_temp_dir)
        if self._fmu == NULL:
            last_error = decode(FMIL.jm_get_last_error(&self.callbacks))
            if self.callbacks.log_level >= FMIL.jm_log_level_error:
                raise FMUException("The XML file could not be parsed. "+last_error)
            else:
                raise FMUException("The XML file could not be parsed. Enable logging for possibly more information.")
        self._allocated_xml = 1

        #Check the FMU kind
        fmu_kind = FMIL.fmi1_import_get_fmu_kind(self._fmu)
        if fmu_kind != FMI_ME and fmu_kind != FMI_CS_STANDALONE and fmu_kind != FMI_CS_TOOL:
            raise FMUException("This class only supports FMI 1.0 (Model Exchange and Co-Simulation).")
        self._fmu_kind = fmu_kind

        #Connect the DLL
        if _connect_dll:
            global FMI_REGISTER_GLOBALLY
            status = FMIL.fmi1_import_create_dllfmu(self._fmu, self.callBackFunctions, FMI_REGISTER_GLOBALLY);
            if status == FMIL.jm_status_error:
                last_error = decode(FMIL.fmi1_import_get_last_error(self._fmu))
                if self.callbacks.log_level >= FMIL.jm_log_level_error:
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
        self._modelid = decode(FMIL.fmi1_import_get_model_identifier(self._fmu))
        self._modelname = decode(FMIL.fmi1_import_get_model_name(self._fmu))
        self._nEventIndicators = FMIL.fmi1_import_get_number_of_event_indicators(self._fmu)
        self._nContinuousStates = FMIL.fmi1_import_get_number_of_continuous_states(self._fmu)
        fmu_log_name = encode((self._modelid + "_log.txt") if log_file_name=="" else log_file_name)
        self._fmu_log_name = <char*>FMIL.malloc((FMIL.strlen(fmu_log_name)+1)*sizeof(char))
        FMIL.strcpy(self._fmu_log_name, fmu_log_name)

        #Create the log file
        with open(self._fmu_log_name,'w') as file:
            for i in range(len(self._log)):
                file.write("FMIL: module = %s, log level = %d: %s\n"%(self._log[i][0], self._log[i][1], self._log[i][2]))
            self._log = []

    cpdef _internal_set_fmu_null(self):
        """
        This methods is ONLY for testing purposes. It sets the internal
        fmu state to NULL
        """
        self._fmu = NULL
        
    def get_version(self):
        """
        Returns the FMI version of the Model which it was generated according.

        Returns::

            version --
                The version.

        Example::

            model.get_version()
        """
        version = FMIL.fmi1_import_get_version(self._fmu)
        return version

    def _get_version(self):
        """
        Returns the FMI version of the Model which it was generated according.

        Returns::

            version --
                The version.

        Example::

            model.version
        """
        logging.warning("The attribute 'version' is deprecated, use 'get_version()' instead.")
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

        Calls the low-level FMI function: fmiGetReal
        """
        cdef int status
        cdef FMIL.size_t nref
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = N.array(valueref, copy=False, dtype=N.uint32).ravel()
        nref = len(val_ref)
        cdef N.ndarray[FMIL.fmi1_real_t, ndim=1,mode='c'] val = N.array([0.0]*nref, dtype=N.float, ndmin=1)


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

        Calls the low-level FMI function: fmiSetReal
        """
        cdef int status
        cdef FMIL.size_t nref
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = N.array(valueref, copy=False, dtype=N.uint32).ravel()
        cdef N.ndarray[FMIL.fmi1_real_t, ndim=1,mode='c'] val = N.array(values, copy=False, dtype=N.float).ravel()
        nref = val_ref.size

        if val_ref.size != val.size:
            raise FMUException(
                'The length of valueref and values are inconsistent.')

        status = FMIL.fmi1_import_set_real(self._fmu, <FMIL.fmi1_value_reference_t*>val_ref.data, nref, <FMIL.fmi1_real_t*>val.data)

        if status != 0:
            raise FMUException('Failed to set the Real values. See the log for possibly more information.')

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

        Calls the low-level FMI function: fmiGetInteger
        """
        cdef int status
        cdef FMIL.size_t nref
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = N.array(valueref, dtype=N.uint32,ndmin=1).ravel()

        nref = val_ref.size
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

        Calls the low-level FMI function: fmiSetInteger
        """
        cdef int status
        cdef FMIL.size_t nref
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = N.array(valueref, dtype=N.uint32,ndmin=1).ravel()
        cdef N.ndarray[FMIL.fmi1_integer_t, ndim=1,mode='c'] val = N.array(values, dtype=int,ndmin=1).ravel()

        nref = val_ref.size

        if val_ref.size != val.size:
            raise FMUException(
                'The length of valueref and values are inconsistent.')

        status = FMIL.fmi1_import_set_integer(self._fmu, <FMIL.fmi1_value_reference_t*>val_ref.data, nref, <FMIL.fmi1_integer_t*>val.data)

        if status != 0:
            raise FMUException('Failed to set the Integer values. See the log for possibly more information.')


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

        Calls the low-level FMI function: fmiGetBoolean
        """
        cdef int status
        cdef FMIL.size_t nref
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = N.array(valueref, dtype=N.uint32,ndmin=1).ravel()

        nref = val_ref.size
        #cdef N.ndarray[FMIL.fmi1_boolean_t, ndim=1,mode='c'] val = N.array(['0']*nref, dtype=N.char.character,ndmin=1)
        cdef void *val = FMIL.malloc(sizeof(FMIL.fmi1_boolean_t)*nref)


        #status = FMIL.fmi1_import_get_boolean(self._fmu, <FMIL.fmi1_value_reference_t*>val_ref.data, nref, <FMIL.fmi1_boolean_t*>val.data)
        status = FMIL.fmi1_import_get_boolean(self._fmu, <FMIL.fmi1_value_reference_t*>val_ref.data, nref, <FMIL.fmi1_boolean_t*>val)

        return_values = []
        for i in range(nref):
            return_values.append((<FMIL.fmi1_boolean_t*>val)[i]==1)

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

        Calls the low-level FMI function: fmiSetBoolean
        """
        cdef int status
        cdef FMIL.size_t nref

        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = N.array(valueref, dtype=N.uint32,ndmin=1).ravel()
        nref = val_ref.size

        #cdef N.ndarray[FMIL.fmi1_boolean_t, ndim=1,mode='c'] val = N.array(['0']*nref, dtype=N.char.character,ndmin=1).flatten()
        cdef void *val = FMIL.malloc(sizeof(FMIL.fmi1_boolean_t)*nref)

        values = N.array(values,ndmin=1).ravel()
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
            raise FMUException('Failed to set the Boolean values. See the log for possibly more information.')

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

        Calls the low-level FMI function: fmiGetString
        """
        cdef int         status
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1, mode='c'] input_valueref = N.array(valueref, dtype=N.uint32, ndmin=1).ravel()
        cdef FMIL.fmi1_string_t* output_value = <FMIL.fmi1_string_t*>FMIL.malloc(sizeof(FMIL.fmi1_string_t)*input_valueref.size)

        status = FMIL.fmi1_import_get_string(self._fmu, <FMIL.fmi1_value_reference_t*> input_valueref.data, input_valueref.size, output_value)

        if status != 0:
            raise FMUException('Failed to get the String values.')
        
        out = []
        for i in range(input_valueref.size):
            out.append(decode(output_value[i]))
            
        FMIL.free(output_value)

        return out

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

        Calls the low-level FMI function: fmiSetString
        """
        cdef int status
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = N.array(valueref, dtype=N.uint32,ndmin=1).ravel()
        cdef FMIL.fmi1_string_t* val = <FMIL.fmi1_string_t*>FMIL.malloc(sizeof(FMIL.fmi1_string_t)*val_ref.size)

        if not isinstance(values, list):
            raise FMUException(
                'The values needs to be a list of values.')
        if len(values) != val_ref.size:
            raise FMUException(
                'The length of valueref and values are inconsistent.')
        
        if python3_flag:
            values = [item.decode() if isinstance(item, bytes) else item for item in values]
        

        for i in range(val_ref.size):
            val[i] = values[i]      
        
        status = FMIL.fmi1_import_set_string(self._fmu, <FMIL.fmi1_value_reference_t*>val_ref.data, val_ref.size, val)

        FMIL.free(val)

        if status != 0:
            raise FMUException('Failed to set the String values. See the log for possibly more information.')

    def set_debug_logging(self,flag):
        """
        Specifies if the logging from the FMU should be turned on or 
        off. Note that this only determines the output from the FMU to
        PyFMI which are additionally filtered using the method 
        'set_log_level'. To specify actual given logging output, 
        please use also use that method.

        Parameters::

            flag --
                Boolean value.

        Calls the low-level FMI function: fmiSetDebuggLogging
        """
        cdef FMIL.fmi1_boolean_t log
        cdef int status

        #self.callbacks.log_level = FMIL.jm_log_level_warning if flag else FMIL.jm_log_level_nothing

        if flag:
            log = 1
        else:
            log = 0

        status = FMIL.fmi1_import_set_debug_logging(self._fmu, log)
        self._enable_logging = bool(log)

        if status != 0:
            raise FMUException('Failed to set the debugging option.')

    def set_fmil_log_level(self, FMIL.jm_log_level_enu_t level):
        logging.warning("The method 'set_fmil_log_level' is deprecated, use 'set_log_level' instead.")
        self.set_log_level(level)

    def _set(self, variable_name, value):
        """
        Helper method to set, see docstring on set.
        """
        cdef FMIL.fmi1_value_reference_t ref
        cdef FMIL.fmi1_base_type_enu_t type
        cdef FMIL.fmi1_import_variable_t* variable
        cdef FMIL.fmi1_variable_alias_kind_enu_t alias_kind
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variable_name)

        ref =  FMIL.fmi1_import_get_variable_vr(variable)
        type = FMIL.fmi1_import_get_variable_base_type(variable)
        alias_kind = FMIL.fmi1_import_get_variable_alias_kind(variable)

        if type == FMIL.fmi1_base_type_real:  #REAL
            if alias_kind == FMI_NEGATED_ALIAS:
                value = -value
            self.set_real([ref], [value])
        elif type == FMIL.fmi1_base_type_int or type == FMIL.fmi1_base_type_enum: #INTEGER
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


    def _get(self, variable_name):
        """
        Helper method to get, see docstring on get.
        """
        cdef FMIL.fmi1_value_reference_t ref
        cdef FMIL.fmi1_base_type_enu_t type
        cdef FMIL.fmi1_import_variable_t* variable
        cdef FMIL.fmi1_variable_alias_kind_enu_t alias_kind
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variable_name)

        ref =  FMIL.fmi1_import_get_variable_vr(variable)
        type = FMIL.fmi1_import_get_variable_base_type(variable)
        alias_kind = FMIL.fmi1_import_get_variable_alias_kind(variable)

        if type == FMIL.fmi1_base_type_real:  #REAL
            value = self.get_real([ref])
            return -1*value if alias_kind == FMI_NEGATED_ALIAS else value
        elif type == FMIL.fmi1_base_type_int or type == FMIL.fmi1_base_type_enum: #INTEGER
            value = self.get_integer([ref]) 
            return -1*value if alias_kind == FMI_NEGATED_ALIAS else value
        elif type == FMIL.fmi1_base_type_str: #STRING
            return self.get_string([ref])
        elif type == FMIL.fmi1_base_type_bool: #BOOLEAN
            value = self.get_boolean([ref])
            return not value if alias_kind == FMI_NEGATED_ALIAS else value
        else:
            raise FMUException('Type not supported.')
    
    cpdef get_variable_description(self, variable_name):
        """
        Get the description of a given variable.

        Parameter::

            variable_name --
                The name of the variable

        Returns::

            The description of the variable.
        """
        cdef FMIL.fmi1_import_variable_t* variable
        cdef char* desc
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        desc = FMIL.fmi1_import_get_variable_description(variable)

        return decode(desc) if desc != NULL else ""
    
    cdef _add_scalar_variable(self, FMIL.fmi1_import_variable_t* variable):
        
        if variable == NULL:
            raise FMUException("Unknown variable. Please verify the correctness of the XML file and check the log.")
        
        alias_kind = FMIL.fmi1_import_get_variable_alias_kind(variable)
        name       = decode(FMIL.fmi1_import_get_variable_name(variable))
        value_ref  = FMIL.fmi1_import_get_variable_vr(variable)
        data_type  = FMIL.fmi1_import_get_variable_base_type(variable)
        has_start  = FMIL.fmi1_import_get_variable_has_start(variable)
        data_variability = FMIL.fmi1_import_get_variability(variable)
        data_causality   = FMIL.fmi1_import_get_causality(variable)
        desc       = FMIL.fmi1_import_get_variable_description(variable)

        return ScalarVariable(name,value_ref, data_type, desc.decode('UTF-8') if desc!=NULL else "",
                            data_variability, data_causality, alias_kind)
    
    def get_scalar_variable(self, variable_name):
        """
        Get the variable as a scalar variable instance.

        Parameter::

            variable_name --
                The name of the variable.

        Returns::

            Instance of ScalarVariable.
        """
        cdef FMIL.fmi1_import_variable_t* variable
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
            
        return self._add_scalar_variable(variable)

    cpdef FMIL.fmi1_base_type_enu_t get_variable_data_type(self, variable_name) except *:
        """
        Get data type of variable.

        Parameter::

            variable_name --
                The name of the variable.

        Returns::

            The type of the variable.
        """
        cdef FMIL.fmi1_import_variable_t* variable
        cdef FMIL.fmi1_base_type_enu_t type
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        type = FMIL.fmi1_import_get_variable_base_type(variable)

        return type

    cpdef FMIL.fmi1_value_reference_t get_variable_valueref(self, variable_name) except *:
        """
        Extract the ValueReference given a variable name.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The ValueReference for the variable passed as argument.
        """
        cdef FMIL.fmi1_import_variable_t* variable
        cdef FMIL.fmi1_value_reference_t vr
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
        vr =  FMIL.fmi1_import_get_variable_vr(variable)

        return vr

    def get_variable_nominal(self, variable_name=None, valueref=None, _override_erroneous_nominal=True):
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
        cdef FMIL.fmi1_import_variable_t *variable
        cdef FMIL.fmi1_import_real_variable_t *real_variable
        cdef char* variablename = NULL
        cdef FMIL.fmi1_real_t value

        if valueref != None:
            variable = FMIL.fmi1_import_get_variable_by_vr(self._fmu, FMIL.fmi1_base_type_real, <FMIL.fmi1_value_reference_t>valueref)
            if variable == NULL:
                raise FMUException("The variable with value reference: %s, could not be found."%str(valueref))
        elif variable_name != None:
            variable_name = encode(variable_name)
            variablename  = variable_name

            variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
            if variable == NULL:
                raise FMUException("The variable %s could not be found."%variablename)
        else:
            raise FMUException('Either provide value reference or variable name.')

        real_variable = FMIL.fmi1_import_get_variable_as_real(variable)
        if real_variable == NULL:
            raise FMUException("The variable is not a real variable.")
        
        value = FMIL.fmi1_import_get_real_variable_nominal(real_variable)
        
        if _override_erroneous_nominal:
            if variable_name == None:
                variable_name = self.get_variable_by_valueref(valueref)
                variablename = variable_name
                
            if value == 0.0:
                if self.callbacks.log_level >= FMIL.jm_log_level_warning:
                    logging.warning("The nominal value for %s is 0.0 which is illegal according to the FMI specification. Setting the nominal to 1.0"%variablename)
                value = 1.0
            elif value < 0.0:
                if self.callbacks.log_level >= FMIL.jm_log_level_warning:
                    logging.warning("The nominal value for %s is <0.0 which is illegal according to the FMI specification. Setting the nominal to abs(%g)"%(variablename, value))
                value = abs(value)
            
        return value

    cpdef get_variable_fixed(self, variable_name):
        """
        Returns if the start value is fixed (True - The value is used as
        an initial value) or not (False - The value is used as a guess
        value).

        Parameters::

            variable_name --
                The name of the variable

        Returns::

            If the start value is fixed or not.
        """
        cdef FMIL.fmi1_import_variable_t *variable
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        status = FMIL.fmi1_import_get_variable_has_start(variable)

        if status == 0:
            raise FMUException("The variable %s does not have a start value."%variablename)

        fixed = FMIL.fmi1_import_get_variable_is_fixed(variable)

        return fixed==1


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
        cdef FMIL.fmi1_import_variable_t *variable
        cdef FMIL.fmi1_import_integer_variable_t* int_variable
        cdef FMIL.fmi1_import_real_variable_t* real_variable
        cdef FMIL.fmi1_import_bool_variable_t* bool_variable
        cdef FMIL.fmi1_import_enum_variable_t* enum_variable
        cdef FMIL.fmi1_import_string_variable_t*  str_variable
        cdef FMIL.fmi1_base_type_enu_t type
        cdef int status
        cdef FMIL.fmi1_boolean_t FMITRUE = 1
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

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

    cpdef get_variable_max(self, variable_name):
        """
        Returns the maximum value for the given variable.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The maximum value for the variable.
        """
        cdef FMIL.fmi1_import_variable_t *variable
        cdef FMIL.fmi1_import_integer_variable_t* int_variable
        cdef FMIL.fmi1_import_real_variable_t* real_variable
        cdef FMIL.fmi1_import_enum_variable_t* enum_variable
        cdef FMIL.fmi1_base_type_enu_t type
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

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

    cpdef get_variable_min(self, variable_name):
        """
        Returns the minimum value for the given variable.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The minimum value for the variable.
        """
        cdef FMIL.fmi1_import_variable_t *variable
        cdef FMIL.fmi1_import_integer_variable_t* int_variable
        cdef FMIL.fmi1_import_real_variable_t* real_variable
        cdef FMIL.fmi1_import_enum_variable_t* enum_variable
        cdef FMIL.fmi1_base_type_enu_t type
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

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

    @enable_caching
    def get_model_variables(self,type=None, int include_alias=True,
                            causality=None,   variability=None,
                            int only_start=False,  int only_fixed=False,
                            filter=None, int _as_list = False):
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
        cdef FMIL.fmi1_base_type_enu_t data_type,target_type = FMIL.fmi1_base_type_real
        cdef FMIL.fmi1_variability_enu_t data_variability,target_variability = FMIL.fmi1_variability_enu_constant
        cdef FMIL.fmi1_causality_enu_t data_causality,target_causality = FMIL.fmi1_causality_enu_input
        cdef FMIL.fmi1_variable_alias_kind_enu_t alias_kind
        cdef char* desc
        cdef dict variable_dict = {}
        cdef list filter_list = [], variable_return_list = []
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
            name       = decode(FMIL.fmi1_import_get_variable_name(variable))
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
                    if filter_list[j].match(name):
                        break
                else:
                    continue

            if include_alias:
                if _as_list:
                    variable_return_list.append(ScalarVariable(name,
                                       value_ref, data_type, desc.decode('UTF-8') if desc!=NULL else "",
                                       data_variability, data_causality,
                                       alias_kind))
                else:
                    variable_dict[name] = ScalarVariable(name,
                                       value_ref, data_type, desc.decode('UTF-8') if desc!=NULL else "",
                                       data_variability, data_causality,
                                       alias_kind)

            elif alias_kind ==FMIL.fmi1_variable_is_not_alias:
                if _as_list:
                    variable_return_list.append(ScalarVariable(name,
                                       value_ref, data_type, desc.decode('UTF-8') if desc!=NULL else "",
                                       data_variability, data_causality,
                                       alias_kind))
                else:
                    variable_dict[name] = ScalarVariable(name,
                                       value_ref, data_type, desc.decode('UTF-8') if desc!=NULL else "",
                                       data_variability, data_causality,
                                       alias_kind)

        #Free the variable list
        FMIL.fmi1_import_free_variable_list(variable_list)
        
        if _as_list:
            return variable_return_list
        else:
            return variable_dict
    
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
        cdef FMIL.fmi1_import_variable_t *variable
        cdef FMIL.fmi1_import_variable_t *base_variable
        cdef FMIL.fmi1_import_variable_list_t *variable_list
        cdef FMIL.size_t variable_list_size
        cdef FMIL.fmi1_value_reference_t value_ref
        cdef FMIL.fmi1_base_type_enu_t data_type
        cdef FMIL.fmi1_variability_enu_t data_variability
        cdef FMIL.fmi1_variable_alias_kind_enu_t alias_kind
        cdef list filter_list = []
        cdef list real_var_ref = []
        cdef list int_var_ref = []
        cdef list bool_var_ref = []
        cdef int  selected_variability = 0 #If a variability has been selected
        cdef int  selected_filter = 1 if filter else 0
        cdef int  length_filter = 0
        cdef dict added_vars = {}

        variable_list = FMIL.fmi1_import_get_variable_list(self._fmu)
        variable_list_size = FMIL.fmi1_import_get_variable_list_size(variable_list)

        if selected_filter:
            filter_list = self._convert_filter(filter)
            length_filter = len(filter_list)

        for i in range(variable_list_size):

            variable = FMIL.fmi1_import_get_variable(variable_list, i)

            alias_kind = FMIL.fmi1_import_get_variable_alias_kind(variable)
            name       = decode(FMIL.fmi1_import_get_variable_name(variable))
            data_variability = FMIL.fmi1_import_get_variability(variable)
            data_type  = FMIL.fmi1_import_get_variable_base_type(variable)

            if data_type != FMIL.fmi1_base_type_real and data_type != FMIL.fmi1_base_type_int and data_type != FMIL.fmi1_base_type_bool and data_type != FMIL.fmi1_base_type_enum:
                continue

            if data_variability != FMIL.fmi1_variability_enu_continuous and data_variability != FMIL.fmi1_variability_enu_discrete:
                continue
                
            if alias_kind == FMIL.fmi1_variable_is_not_alias:
                value_ref = FMIL.fmi1_import_get_variable_vr(variable)
            else:
                base_variable = FMIL.fmi1_import_get_variable_alias_base(self._fmu, variable)
                value_ref  = FMIL.fmi1_import_get_variable_vr(base_variable)
            
            if selected_filter:
                for j in range(length_filter):
                    if filter_list[j].match(name):
                        break
                else:
                    continue
                if added_vars.has_key(value_ref):
                    continue
                else:
                    added_vars[value_ref] = 1
            else:
                if alias_kind != FMIL.fmi1_variable_is_not_alias:
                    continue

            if data_type == FMIL.fmi1_base_type_real:
                real_var_ref.append(value_ref)
            if data_type == FMIL.fmi1_base_type_int or data_type == FMIL.fmi1_base_type_enum:
                int_var_ref.append(value_ref)
            if data_type == FMIL.fmi1_base_type_bool:
                bool_var_ref.append(value_ref)

        #Free the variable list
        FMIL.fmi1_import_free_variable_list(variable_list)

        return real_var_ref, int_var_ref, bool_var_ref

    def get_variable_alias_base(self, variable_name):
        """
        Returns the base variable for the provided variable name.

        Parameters::

            variable_name--
                Name of the variable.

        Returns::
            
            The base variable.
            
        """
        cdef FMIL.fmi1_import_variable_t* variable
        cdef FMIL.fmi1_import_variable_t* base_variable
        cdef FMIL.fmi1_value_reference_t vr
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        base_variable = FMIL.fmi1_import_get_variable_alias_base(self._fmu, variable)
        if base_variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        name = decode(FMIL.fmi1_import_get_variable_name(base_variable))

        return name

    def get_variable_alias(self, variable_name):
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
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        alias_list = FMIL.fmi1_import_get_variable_aliases(self._fmu, variable)

        alias_list_size = FMIL.fmi1_import_get_variable_list_size(alias_list)

        #Loop over all the alias variables
        for i in range(alias_list_size):

            variable = FMIL.fmi1_import_get_variable(alias_list, i)

            alias_kind = FMIL.fmi1_import_get_variable_alias_kind(variable)
            alias_name = decode(FMIL.fmi1_import_get_variable_name(variable))

            ret_values[alias_name] = alias_kind

        #FREE VARIABLE LIST
        FMIL.fmi1_import_free_variable_list(alias_list)

        return ret_values

    cpdef FMIL.fmi1_variability_enu_t get_variable_variability(self, variable_name) except *:
        """
        Get variability of variable.

        Parameters::

            variablename --

                The name of the variable.

        Returns::

            The variability of the variable, CONSTANT(0), PARAMETER(1), 
            DISCRETE(2) or CONTINUOUS(3)
        """
        cdef FMIL.fmi1_import_variable_t* variable
        cdef FMIL.fmi1_variability_enu_t variability
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

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

        name = decode(FMIL.fmi1_import_get_variable_name(variable))

        return name

    cpdef FMIL.fmi1_causality_enu_t get_variable_causality(self, variable_name) except *:
        """
        Get the causality of the variable.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The variability of the variable, INPUT(0), OUTPUT(1),
            INTERNAL(2), NONE(3)
        """
        cdef FMIL.fmi1_import_variable_t* variable
        cdef FMIL.fmi1_causality_enu_t causality
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

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
        return decode(desc) if desc != NULL else ""

    def get_generation_tool(self):
        """
        Return the model generation tool.
        """
        cdef char* gen
        gen = FMIL.fmi1_import_get_generation_tool(self._fmu)
        return decode(gen) if gen != NULL else ""

    def get_guid(self):
        """
        Return the model GUID.
        """
        guid = decode(FMIL.fmi1_import_get_GUID(self._fmu))
        return guid
        
cdef class FMUModelCS1(FMUModelBase):
    #First step only support fmi1_fmu_kind_enu_cs_standalone
    #stepFinished not supported

    def __init__(self, fmu, path='.', enable_logging=None, log_file_name="", log_level=FMI_DEFAULT_LOG_LEVEL, _unzipped_dir=None, _connect_dll=True):
        #Call super
        FMUModelBase.__init__(self,fmu,path,enable_logging,log_file_name, log_level, _unzipped_dir, _connect_dll)

        if self._fmu_kind != FMI_CS_STANDALONE and self._fmu_kind != FMI_CS_TOOL:
            raise FMUException("This class only supports FMI 1.0 for Co-simulation.")
        
        if _connect_dll:
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
            
        if self._instantiated_fmu == 1:
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
            value_refs = N.array([0], dtype=N.uint32,ndmin=1).ravel()
            orders = N.array(order, dtype=N.int32)
            value_refs[0] = self.get_variable_valueref(variables)
        elif isinstance(variables,list) and isinstance(variables[-1],str):
            nref = len(variables)
            value_refs = N.array([0]*nref, dtype=N.uint32,ndmin=1).ravel()
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

    def get_real_status(self, status_kind):
        """
        """
        cdef FMIL.fmi1_real_t value = 0.0
        cdef int status
        
        if status_kind == FMIL.fmi1_last_successful_time:
            status = FMIL.fmi1_import_get_real_status(self._fmu, FMIL.fmi1_last_successful_time, &value)
            
            if status != 0:
                raise FMUException('Failed to retrieve the last successful time. See the log for possibly more information.') 
                
            return value
        else:
            raise FMUException('Not supported status kind.') 
    

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
        cdef N.ndarray[FMIL.fmi1_integer_t, ndim=1,mode='c'] np_orders = N.array(orders, dtype=N.int32, ndmin=1).ravel()
        cdef N.ndarray[FMIL.fmi1_value_reference_t, ndim=1,mode='c'] value_refs
        cdef N.ndarray[FMIL.fmi1_real_t, ndim=1,mode='c'] val = N.array(values, dtype=N.float, ndmin=1).ravel()

        nref = val.size
        orders = N.array([0]*nref, dtype=N.int32)

        if nref != np_orders.size:
            raise FMUException("The number of variables must be the same as the number of orders.")

        fmu_capabilities = FMIL.fmi1_import_get_capabilities(self._fmu)
        can_interpolate_inputs = FMIL.fmi1_import_get_canInterpolateInputs(fmu_capabilities)
        #NOTE IS THIS THE HIGHEST ORDER OF INTERPOLATION OR SIMPLY IF IT CAN OR NOT?

        for i in range(np_orders.size):
            if np_orders[i] < 1:
                raise FMUException("The order must be greater than zero.")
        if not can_interpolate_inputs:
            raise FMUException("The FMU does not support input derivatives.")

        if isinstance(variables,str):
            value_refs = N.array([0], dtype=N.uint32,ndmin=1).ravel()
            value_refs[0] = self.get_variable_valueref(variables)
        elif isinstance(variables,list) and isinstance(variables[-1],str):
            value_refs = N.array([0]*nref, dtype=N.uint32,ndmin=1).ravel()
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

                    >> myModel = load_fmu(...)
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
        Get an instance of the simulate options class, filled with default
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

    def initialize(self, start_time=0.0, stop_time=1.0, stop_time_defined=False, tStart=None, tStop=None, StopTimeDefined=None):
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
        cdef int status
        cdef FMIL.fmi1_boolean_t stop_defined

        if tStart is not None:
            logging.warning("The attribute 'tStart' is deprecated and will be removed. Please use 'start_time' instead.")
            start_time = tStart #If the user has used this, use it!
        if tStop is not None:
            logging.warning("The attribute 'tStop' is deprecated and will be removed. Please use 'stop_time' instead.")
            stop_time = tStop #If the user has used this, use it!
        if StopTimeDefined is not None:
            logging.warning("The attribute 'StopTimeDefined' is deprecated and will be removed. Please use 'stop_time_defined' instead.")
            stop_time_defined = StopTimeDefined #If the user has used this, use it!

        self.time = start_time
        stop_defined = 1 if stop_time_defined else 0
        
        log_open = self._log_open()
        if not log_open and self.get_log_level() > 2:
            self._open_log_file()

        status = FMIL.fmi1_import_initialize_slave(self._fmu, start_time, stop_defined, stop_time)
        
        if not log_open and self.get_log_level() > 2:
            self._close_log_file()

        if status != FMIL.fmi1_status_ok:
            raise FMUException("The slave failed to initialize. See the log for possibly more information.")

        self._allocated_fmu = 1

    def reset(self):
        """
        This method resets the FMU according to the reset method defined
        in the FMI1 specification.
        """

        status = FMIL.fmi1_import_reset_slave(self._fmu)
        if status != FMIL.fmi1_status_ok:
            raise FMUException("Failed to reset the FMU.")
        
        #The FMU is no longer initialized
        self._allocated_fmu = 0

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
        
        name = encode(name)
        status = FMIL.fmi1_import_instantiate_slave(self._fmu, name, location,
                                        FMI_MIME_CS_STANDALONE, timeout, visible,
                                        interactive)

        if status != FMIL.jm_status_success:
            raise FMUException('Failed to instantiate the slave. See the log for possibly more information.')

        #The FMU is instantiated
        self._instantiated_fmu = 1

        #Just to be safe, some problems with Dymola (2012) FMUs not reacting
        #to logging when set to the instantiate method.
        status = FMIL.fmi1_import_set_debug_logging(self._fmu, log)

        if status != 0:
            raise FMUException('Failed to set the debugging option. See the log for possibly more information.')

    def get_capability_flags(self):
        """
        Returns a dictionary with the capability flags of the FMU.

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
        
    def terminate(self):
        """
        Calls the FMI function fmiTerminateSlave() on the FMU.
        After this call, any call to a function changing the state of the FMU will fail.
        """
        if self._allocated_fmu == 1:
            FMIL.fmi1_import_terminate_slave(self._fmu)
            self._allocated_fmu = 0 #No longer initialized
        
    def free_instance(self):
        """
        Calls the FMI function fmiFreeSlaveInstance on the FMU.
        Note that this is not needed in general as it is done automatically.
        """
        if self._instantiated_fmu == 1:
            FMIL.fmi1_import_free_slave_instance(self._fmu)
            self._instantiated_fmu = 0

cdef class FMUModelME1(FMUModelBase):
    """
    An FMI Model loaded from a DLL.
    """

    def __init__(self, fmu, path='.', enable_logging=None, log_file_name="", log_level=FMI_DEFAULT_LOG_LEVEL, _unzipped_dir=None, _connect_dll=True):
        #Call super
        FMUModelBase.__init__(self,fmu,path,enable_logging,log_file_name, log_level, _unzipped_dir, _connect_dll)

        if self._fmu_kind != FMI_ME:
            raise FMUException("This class only supports FMI 1.0 for Model Exchange.")

        if _connect_dll:
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
        fmiFreeModelInstance and then reloads the DLL and finally
        re-instantiates using fmiInstantiateModel.
        """
        if self._allocated_fmu == 1:
            FMIL.fmi1_import_terminate(self._fmu)
            self._allocated_fmu = 0
            
        if self._instantiated_fmu == 1:
            FMIL.fmi1_import_free_model_instance(self._fmu)
            self._instantiated_fmu = 0

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
            
        if self._instantiated_fmu == 1:
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
            raise FMUException('Failed to get the derivative values at time: %E.'%self.time)

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
            raise FMUException('Failed to get the event indicators at time: %E.'%self.time)

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
            raise FMUException('Failed to update the events at time: %E.'%self.time)

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
        cdef FMIL.fmi1_boolean_t call_event_update

        status = FMIL.fmi1_import_completed_integrator_step(self._fmu, &call_event_update)

        if status != 0:
            raise FMUException('Failed to call FMI completed step at time: %E.'%self.time)

        if call_event_update == 1:
            return True
        else:
            return False


    def initialize(self, tolerance_defined=True, tolerance="Default", tolControlled=None, relativeTolerance=None):
        """
        Initializes the model and computes initial values for all variables,
        including setting the start values of variables defined with a the start
        attribute in the XML-file.

        Parameters::

            tolerance_defined --
                If the model are going to be called by numerical solver using
                step-size control. Boolean flag.
                Default: True
                
            tolerance --
                If the model are controlled by a numerical solver using
                step-size control, the same tolerance should be provided here.
                Else the default tolerance from the XML-file are used.

        Calls the low-level FMI function: fmiInitialize.
        """
        cdef char tolerance_controlled
        cdef FMIL.fmi1_real_t c_tolerance

        #Trying to set the initial time from the xml file, else 0.0
        if self.time == None:
            self.time = FMIL.fmi1_import_get_default_experiment_start(self._fmu)
        
        if tolControlled is not None:
            logging.warning("The attribute 'tolControlled' is deprecated and will be removed. Please use 'tolerance_defined' instead.")
            tolerance_defined = tolControlled #If the user has used this, use it!
        if relativeTolerance is not None:
            logging.warning("The attribute 'relativeTolerance' is deprecated and will be removed. Please use 'tolerance' instead.")
            tolerance = relativeTolerance #If the user has used this, use it!
        
        if tolerance_defined:
            tolerance_controlled = 1
            if tolerance == "Default":
                c_tolerance = FMIL.fmi1_import_get_default_experiment_tolerance(self._fmu)
            else:
                c_tolerance = tolerance
        else:
            tolerance_controlled = 0
            c_tolerance = 0.0

        log_open = self._log_open()
        if not log_open and self.get_log_level() > 2:
            self._open_log_file()

        status = FMIL.fmi1_import_initialize(self._fmu, tolerance_controlled, c_tolerance, &self._eventInfo)

        if not log_open and self.get_log_level() > 2:
            self._close_log_file()

        if status == 1:
            if self._enable_logging:
                logging.warning(
                    'Initialize returned with a warning.' \
                    ' Check the log for information (model.get_log).')
            else:
                logging.warning('Initialize returned with a warning.' \
                    ' Enable logging for more information, (load_fmu(..., log_level=4)).')

        if status > 1:
            if self._enable_logging:
                raise FMUException(
                    'Initialize returned with an error.' \
                    ' Check the log for information (model.get_log).')
            else:
                raise FMUException('Initialize returned with an error.' \
                    ' Enable logging for more information, (load_fmu(..., log_level=4)).')

        self._allocated_fmu = 1


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

        name = encode(name)
        status = FMIL.fmi1_import_instantiate_model(self._fmu, name)

        if status != 0:
            raise FMUException('Failed to instantiate the model. See the log for possibly more information.')
        
        #The FMU is instantiated
        self._instantiated_fmu = 1

        #Just to be safe, some problems with Dymola (2012) FMUs not reacting
        #to logging when set to the instantiate method.
        status = FMIL.fmi1_import_set_debug_logging(self._fmu, log)

        if status != 0:
            raise FMUException('Failed to set the debugging option. See the log for possibly more information.')
            
        

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

                    >> myModel = load_fmu(...)
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

    def terminate(self):
        """
        Calls the FMI function fmiTerminate() on the FMU.
        After this call, any call to a function changing the state of the FMU will fail.
        """
        if self._allocated_fmu == 1:
            FMIL.fmi1_import_terminate(self._fmu)
            self._allocated_fmu = 0 #No longer initialized
        
    def free_instance(self):
        """
        Calls the FMI function fmiFreeModelInstance on the FMU.
        Note that this is not needed in general as it is done automatically.
        """
        if self._instantiated_fmu == 1:
            FMIL.fmi1_import_free_model_instance(self._fmu)
            self._instantiated_fmu = 0



cdef class FMUModelBase2(ModelBase):
    """
    FMI Model loaded from a dll.
    """
    def __init__(self, fmu, path='.', enable_logging=None, log_file_name="", log_level=FMI_DEFAULT_LOG_LEVEL, _unzipped_dir=None, _connect_dll=True):
        """
        Constructor of the model.

        Parameters::

            fmu --
                Name of the fmu as a string.

            path --
                Path to the fmu-directory.
                Default: '.' (working directory)

            enable_logging [DEPRECATED] --
                This option is DEPRECATED and will be removed. Please 
                use the option "log_level" instead.

            log_file_name --
                Filename for file used to save logmessages.
                Default: "" (Generates automatically)
                
            log_level --
                Determines the logging output. Can be set between 0 
                (no logging) and 7 (everything).
                Default: 2 (log error messages)

        Returns::

            A model as an object from the class FMUModelFMU2
        """

        cdef int  status
        cdef dict reals_continuous
        cdef dict reals_discrete
        cdef dict int_discrete
        cdef dict bool_discrete
        
        #Call super
        ModelBase.__init__(self)
        
        #Contains the log information
        self._log               = []
        self._enable_logging    = enable_logging

        #Used for deallocation
        self._allocated_context = 0
        self._allocated_dll = 0
        self._allocated_xml = 0
        self._allocated_fmu = 0
        self._fmu_temp_dir = NULL
        self._fmu_log_name = NULL

        #Default values
        self.__t = None
        self._A = None
        self._group_A = None
        self._mask_A = None
        self._B = None
        self._group_B = None
        self._C = None
        self._group_C = None
        self._D = None
        self._group_D = None
        self._states_references = None
        self._derivatives_references = None
        self._outputs_references = None
        self._inputs_references = None
        self._derivatives_states_dependencies = None
        self._derivatives_inputs_dependencies = None
        self._outputs_states_dependencies = None
        self._outputs_inputs_dependencies = None
        self._has_entered_init_mode = False
        self._last_accepted_time = 0.0

        #Internal values
        self._pyEventInfo   = PyEventInfo()
        self._worker_object = WorkerClass2()

        #Specify the general callback functions
        self.callbacks.malloc           = FMIL.malloc
        self.callbacks.calloc           = FMIL.calloc
        self.callbacks.realloc          = FMIL.realloc
        self.callbacks.free             = FMIL.free
        self.callbacks.logger           = importlogger2
        self.callbacks.context          = <void*>self

        #Specify FMI2 related callbacks
        self.callBackFunctions.logger               = FMIL.fmi2_log_forwarding
        self.callBackFunctions.allocateMemory       = FMIL.calloc
        self.callBackFunctions.freeMemory           = FMIL.free
        self.callBackFunctions.stepFinished         = NULL
        self.callBackFunctions.componentEnvironment = NULL        
        
        if enable_logging==None:
            if log_level >= FMIL.jm_log_level_nothing and log_level <= FMIL.jm_log_level_all:
                if log_level == FMIL.jm_log_level_nothing:
                    enable_logging = False
                else:
                    enable_logging = True
                self.callbacks.log_level = log_level
            else:
                raise FMUException("The log level must be between %d and %d"%(FMIL.jm_log_level_nothing, FMIL.jm_log_level_all))
        else:
            logging.warning("The attribute 'enable_logging' is deprecated. Please use 'log_level' instead. Setting 'log_level' to INFO...")
            self.callbacks.log_level = FMIL.jm_log_level_info if enable_logging else FMIL.jm_log_level_error        

        # Check that the file referenced by fmu has the correct file-ending
        if not fmu.endswith('.fmu'):
            raise FMUException("FMUModel must be instantiated with an FMU (.fmu) file.")

        #Check that the file exists
        self._fmu_full_path = encode(os.path.abspath(os.path.join(path,fmu)))
        if not os.path.isfile(self._fmu_full_path):
            raise FMUException('Could not locate the FMU in the specified directory.')

        # Create a struct for allocation
        self._context           = FMIL.fmi_import_allocate_context(&self.callbacks)
        self._allocated_context = 1

        #Get the FMI version of the provided model
        if _unzipped_dir:
            fmu_temp_dir  = encode(_unzipped_dir)
        else:
            fmu_temp_dir  = encode(create_temp_dir())
        self._fmu_temp_dir = <char*>FMIL.malloc((FMIL.strlen(fmu_temp_dir)+1)*sizeof(char))
        FMIL.strcpy(self._fmu_temp_dir, fmu_temp_dir)
        
        if _unzipped_dir:
            #If the unzipped directory is provided we assume that the version
            #is correct. This is due to that the method to get the version
            #unzipps the FMU which we already have done.
            self._version = FMIL.fmi_version_2_0_enu
        else:
            self._version      = FMIL.fmi_import_get_fmi_version(self._context, self._fmu_full_path, self._fmu_temp_dir)

        #Check the version
        if self._version == FMIL.fmi_version_unknown_enu:
            last_error = decode(FMIL.jm_get_last_error(&self.callbacks))
            if enable_logging:
                raise FMUException("The FMU version could not be determined. "+last_error)
            else:
                raise FMUException("The FMU version could not be determined. Enable logging for possibly more information.")

        if self._version != FMIL.fmi_version_2_0_enu:
            last_error = decode(FMIL.jm_get_last_error(&self.callbacks))
            if enable_logging:
                raise FMUException("The FMU version is not supported by this class. "+last_error)
            else:
                raise FMUException("The FMU version is not supported by this class. Enable logging for possibly more information.")

        #Parse xml and check fmu-kind
        self._fmu           = FMIL.fmi2_import_parse_xml(self._context, self._fmu_temp_dir, NULL)

        if self._fmu is NULL:
            last_error = decode(FMIL.jm_get_last_error(&self.callbacks))
            if enable_logging:
                raise FMUException("The XML-could not be read. "+last_error)
            else:
                raise FMUException('The XML-could not be read. Enable logging for possible nore information.')

        self.callBackFunctions.componentEnvironment = <FMIL.fmi2_component_environment_t>self._fmu
        self._fmu_kind      = FMIL.fmi2_import_get_fmu_kind(self._fmu)
        self._allocated_xml = 1

        #FMU kind is unknown
        if self._fmu_kind == FMIL.fmi2_fmu_kind_unknown:
            last_error = decode(FMIL.jm_get_last_error(&self.callbacks))
            if enable_logging:
                raise FMUException("The FMU kind could not be determined. "+last_error)
            else:
                raise FMUException("The FMU kind could not be determined. Enable logging for possibly more information.")
        elif self._fmu_kind == FMIL.fmi2_fmu_kind_me_and_cs:
            if isinstance(self,FMUModelME2):
                self._fmu_kind = FMIL.fmi2_fmu_kind_me
            elif isinstance(self,FMUModelCS2):
                self._fmu_kind = FMIL.fmi2_fmu_kind_cs
            else:
                raise FMUException("FMUModelBase2 cannot be used directly, use FMUModelME2 or FMUModelCS2.")

        #Connect the DLL
        if _connect_dll:
            status = FMIL.fmi2_import_create_dllfmu(self._fmu, self._fmu_kind, &self.callBackFunctions)
            if status == FMIL.jm_status_error:
                last_error = decode(FMIL.fmi2_import_get_last_error(self._fmu))
                if enable_logging:
                    raise FMUException("Error loading the binary. " + last_error)
                else:
                    raise FMUException("Error loading the binary. Enable logging for possibly more information.")
            self._allocated_dll = 1

        #Load information from model
        if isinstance(self,FMUModelME2):
            self._modelId           = decode(FMIL.fmi2_import_get_model_identifier_ME(self._fmu))
        elif isinstance(self,FMUModelCS2):
            self._modelId           = decode(FMIL.fmi2_import_get_model_identifier_CS(self._fmu))
        else:
            raise FMUException("FMUModelBase2 cannot be used directly, use FMUModelME2 or FMUModelCS2.")

        #Connect the DLL
        self._modelName         = decode(FMIL.fmi2_import_get_model_name(self._fmu))
        self._nEventIndicators  = FMIL.fmi2_import_get_number_of_event_indicators(self._fmu)
        self._nContinuousStates = FMIL.fmi2_import_get_number_of_continuous_states(self._fmu)
        fmu_log_name = encode((self._modelId + "_log.txt") if log_file_name=="" else log_file_name)
        self._fmu_log_name = <char*>FMIL.malloc((FMIL.strlen(fmu_log_name)+1)*sizeof(char))
        FMIL.strcpy(self._fmu_log_name, fmu_log_name)

        #Create the log file
        with open(self._fmu_log_name,'w') as file:
            for i in range(len(self._log)):
                file.write("FMIL: module = %s, log level = %d: %s\n"%(self._log[i][0], self._log[i][1], self._log[i][2]))
            self._log = []
            
    cpdef N.ndarray get_real(self, valueref):
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
        cdef int         status

        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1,mode='c'] input_valueref = N.array(valueref, copy=False, dtype=N.uint32).ravel()
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1,mode='c']            output_value   = N.zeros(input_valueref.size)

        status = FMIL.fmi2_import_get_real(self._fmu, <FMIL.fmi2_value_reference_t*> input_valueref.data, input_valueref.size, <FMIL.fmi2_real_t*> output_value.data)

        if status != 0:
            raise FMUException('Failed to get the Real values.')

        return output_value
    
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
        cdef int status

        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1,mode='c'] input_valueref = N.array(valueref, copy=False, dtype=N.uint32).ravel()
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1,mode='c']            set_value      = N.array(values, copy=False, dtype=N.float).ravel()

        if input_valueref.size != set_value.size:
            raise FMUException('The length of valueref and values are inconsistent.')

        status = FMIL.fmi2_import_set_real(self._fmu, <FMIL.fmi2_value_reference_t*> input_valueref.data, input_valueref.size, <FMIL.fmi2_real_t*> set_value.data)

        if status != 0:
            raise FMUException('Failed to set the Real values. See the log for possibly more information.')
    
    cdef int _get_real(self, FMIL.fmi2_value_reference_t* vrefs, size_t size, FMIL.fmi2_real_t* values):
        return FMIL.fmi2_import_get_real(self._fmu, vrefs, size, values)
    
    cdef int _set_real(self, FMIL.fmi2_value_reference_t* vrefs, FMIL.fmi2_real_t* values, size_t size):
        return FMIL.fmi2_import_set_real(self._fmu, vrefs, size, values)
    
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
        cdef int         status
        cdef FMIL.size_t nref

        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1,mode='c'] input_valueref = N.array(valueref, dtype=N.uint32,ndmin=1).ravel()
        nref = input_valueref.size
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

        Calls the low-level FMI function: fmi2SetInteger
        """
        cdef int status
        cdef FMIL.size_t nref

        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1,mode='c'] input_valueref = N.array(valueref, dtype=N.uint32,ndmin=1).ravel()
        cdef N.ndarray[FMIL.fmi2_integer_t, ndim=1,mode='c']         set_value      = N.array(values, dtype=int,ndmin=1).ravel()

        nref = input_valueref.size

        if input_valueref.size != set_value.size:
            raise FMUException('The length of valueref and values are inconsistent.')

        status = FMIL.fmi2_import_set_integer(self._fmu, <FMIL.fmi2_value_reference_t*> input_valueref.data, nref, <FMIL.fmi2_integer_t*> set_value.data)

        if status != 0:
            raise FMUException('Failed to set the Integer values. See the log for possibly more information.')

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
        cdef int         status
        cdef FMIL.size_t nref

        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1,mode='c'] input_valueref = N.array(valueref, dtype=N.uint32, ndmin=1).ravel()
        nref = input_valueref.size

        #cdef N.ndarray[FMIL.fmi1_boolean_t, ndim=1,mode='c'] val = N.array(['0']*nref, dtype=N.char.character,ndmin=1)
        cdef void* output_value = FMIL.malloc(sizeof(FMIL.fmi2_boolean_t)*nref)


        #status = FMIL.fmi1_import_get_boolean(self._fmu, <FMIL.fmi1_value_reference_t*>val_ref.data, nref, <FMIL.fmi1_boolean_t*>val.data)
        status = FMIL.fmi2_import_get_boolean(self._fmu, <FMIL.fmi2_value_reference_t*> input_valueref.data, nref, <FMIL.fmi2_boolean_t*> output_value)

        return_values = []
        for i in range(nref):
            return_values.append((<FMIL.fmi2_boolean_t*> output_value)[i]==1)

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

        Calls the low-level FMI function: fmi2SetBoolean
        """
        cdef int         status
        cdef FMIL.size_t nref

        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1,mode='c'] input_valueref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()
        nref = len(input_valueref)

        #cdef N.ndarray[FMIL.fmi1_boolean_t, ndim=1,mode='c'] val = N.array(['0']*nref, dtype=N.char.character,ndmin=1).flatten()
        cdef void* set_value = FMIL.malloc(sizeof(FMIL.fmi2_boolean_t)*nref)

        values = N.array(values,ndmin=1).ravel()
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
            raise FMUException('Failed to set the Boolean values. See the log for possibly more information.')

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
        cdef int         status
        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1, mode='c'] input_valueref = N.array(valueref, dtype=N.uint32, ndmin=1).ravel()
        cdef FMIL.fmi2_string_t* output_value = <FMIL.fmi2_string_t*>FMIL.malloc(sizeof(FMIL.fmi2_string_t)*input_valueref.size)

        status = FMIL.fmi2_import_get_string(self._fmu, <FMIL.fmi2_value_reference_t*> input_valueref.data, input_valueref.size, output_value)

        if status != 0:
            raise FMUException('Failed to get the String values.')
        
        out = []
        for i in range(input_valueref.size):
            out.append(decode(output_value[i]))
            
        FMIL.free(output_value)

        return out

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
        cdef int status
        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1,mode='c'] val_ref = N.array(valueref, dtype=N.uint32,ndmin=1).ravel()
        cdef FMIL.fmi2_string_t* val = <FMIL.fmi2_string_t*>FMIL.malloc(sizeof(FMIL.fmi2_string_t)*val_ref.size)

        if not isinstance(values, list):
            raise FMUException(
                'The values needs to be a list of values.')
        if len(values) != val_ref.size:
            raise FMUException(
                'The length of valueref and values are inconsistent.')
        
        if python3_flag:
            values = [item.decode() if isinstance(item, bytes) else item for item in values]
        
        for i in range(val_ref.size):
            val[i] = values[i]      
        
        status = FMIL.fmi2_import_set_string(self._fmu, <FMIL.fmi2_value_reference_t*>val_ref.data, val_ref.size, val)

        FMIL.free(val)

        if status != 0:
            raise FMUException('Failed to set the String values. See the log for possibly more information.')

    def _set(self, variable_name, value):
        """
        Helper method to set, see docstring on set.
        """
        cdef FMIL.fmi2_value_reference_t ref
        cdef FMIL.fmi2_base_type_enu_t   type

        ref  = self.get_variable_valueref(variable_name)
        type = self.get_variable_data_type(variable_name)

        if type == FMIL.fmi2_base_type_real:  #REAL
            self.set_real([ref], [value])
        elif type == FMIL.fmi2_base_type_int:
            self.set_integer([ref], [value])
        elif type == FMIL.fmi2_base_type_enum:
            if isinstance(value, str):
                enum_type = self.get_variable_declared_type(variable_name)
                enum_values = {v[0]: k for k, v in enum_type.items.items()}
                try:
                    self.set_integer([ref], [enum_values[value]])
                except KeyError:
                    raise FMUException("The value '%s' is not in the list of allowed enumeration items for variable '%s'. Allowed values: %s'"%(value, variable_name, ", ".join(enum_values.keys())))
            else:
                self.set_integer([ref], [value])
        elif type == FMIL.fmi2_base_type_str: #STRING
            self.set_string([ref], [value])
        elif type == FMIL.fmi2_base_type_bool: #BOOLEAN
            self.set_boolean([ref], [value])
        else:
            raise FMUException('Type not supported.')

    def _get(self, variable_name):
        """
        Helper method to get, see docstring on get.
        """
        cdef FMIL.fmi2_value_reference_t ref
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
            raise FMUException('Type not supported.')
    
    def instantiate(self, name= 'Model', visible = False):
        """
        Instantiate the model.

        Parameters::

            name --
                The name of the instance.
                Default: 'Model'

            visible --
                Defines if the simulator application window should be visible or not.
                Default: False, not visible.

        Calls the low-level FMI function: fmi2Instantiate.
        """

        cdef FMIL.fmi2_boolean_t  log
        cdef FMIL.fmi2_boolean_t  vis
        cdef int status

        if visible:
            vis = 1
        else:
            vis = 0
            
        if isinstance(self,FMUModelME2):
            fmuType = FMIL.fmi2_model_exchange
        elif isinstance(self,FMUModelCS2):
            fmuType = FMIL.fmi2_cosimulation
        else:
            raise FMUException('The instance is not curent an instance of an ME-model or a CS-model. Use load_fmu for correct loading.')

        name = encode(name)
        status =  FMIL.fmi2_import_instantiate(self._fmu, name, fmuType, NULL, vis)

        if status != FMIL.jm_status_success:
            raise FMUException('Failed to instantiate the model. See the log for possibly more information.')
    
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
        cdef int status
        
        cdef FMIL.fmi2_boolean_t stop_defined = FMI2_TRUE if stop_time_defined else FMI2_FALSE
        cdef FMIL.fmi2_boolean_t tol_defined = FMI2_TRUE if tolerance_defined else FMI2_FALSE
        
        if tolerance == "Default":
            tolerance = self.get_default_experiment_tolerance()
        if start_time == "Default":
            start_time = self.get_default_experiment_start_time()
        if stop_time == "Default":
            stop_time = self.get_default_experiment_stop_time()
        
        self.__t = start_time
        self._last_accepted_time = start_time
        self._relative_tolerance = tolerance
        
        status = FMIL.fmi2_import_setup_experiment(self._fmu,
                tol_defined, tolerance, start_time, stop_defined, stop_time)
    
        if status != 0:
            raise FMUException('Failed to setup the experiment.')
    
    def reset(self):
        """
        Resets the FMU back to its original state. Note that the environment 
        has to initialize the FMU again after this function-call.
        """
        cdef int status

        status = FMIL.fmi2_import_reset(self._fmu)
        if status != 0:
            raise FMUException('An error occured when reseting the model, see the log for possible more information')

        #Default values
        self.__t = None
        self._has_entered_init_mode = False
        
        #Reseting the allocation flags
        self._allocated_fmu = 0

        #Internal values
        self._log = []
        
    def terminate(self):
        """
        Calls the FMI function fmi2Terminate() on the FMU.
        After this call, any call to a function changing the state of the FMU will fail.
        """
        FMIL.fmi2_import_terminate(self._fmu)
        
    def free_instance(self):
        """
        Calls the FMI function fmi2FreeInstance() on the FMU. Note that this is not
        needed generally.
        """
        FMIL.fmi2_import_free_instance(self._fmu)
        
    def exit_initialization_mode(self):
        """
        Exit initialization mode by calling the low level FMI function
        fmi2ExitInitializationMode.
        
        Note that the method initialize() performs both the enter and 
        exit of initialization mode.
        """
        status = FMIL.fmi2_import_exit_initialization_mode(self._fmu)
        
        if status == 1:
            if self._enable_logging:
                logging.warning(
                    'Exit Initialize returned with a warning.' \
                    ' Check the log for information (model.get_log).')
            else:
                logging.warning('Exit Initialize returned with a warning.' \
                    ' Enable logging for more information, (load_fmu(..., log_level=4)).')

        if status > 1:
            if self._enable_logging:
                raise FMUException(
                    'Exit Initialize returned with an error.' \
                    ' Check the log for information (model.get_log).')
            else:
                raise FMUException('Exit Initialize returned with an error.' \
                    ' Enable logging for more information, (load_fmu(..., log_level=4)).')
                    

        self._allocated_fmu = 1
        
        return status
        
        
    def enter_initialization_mode(self):
        """
        Enters initialization mode by calling the low level FMI function
        fmi2EnterInitializationMode.
        
        Note that the method initialize() performs both the enter and 
        exit of initialization mode.
        """
        if self.time == None:
            raise FMUException("Setup Experiment has to be called prior to the initialization method.")
        
        status = FMIL.fmi2_import_enter_initialization_mode(self._fmu)
        
        if status == 1:
            if self._enable_logging:
                logging.warning(
                    'Enter Initialize returned with a warning.' \
                    ' Check the log for information (model.get_log).')
            else:
                logging.warning('Enter Initialize returned with a warning.' \
                    ' Enable logging for more information, (load_fmu(..., log_level=4)).')

        if status > 1:
            if self._enable_logging:
                raise FMUException(
                    'Enter Initialize returned with an error.' \
                    ' Check the log for information (model.get_log).')
            else:
                raise FMUException('Enter Initialize returned with an error.' \
                    ' Enable logging for more information, (load_fmu(..., log_level=4)).')
                    
        self._has_entered_init_mode = True
                    
        return status
    
    def initialize(self, tolerance_defined=True, tolerance="Default", start_time="Default", stop_time_defined=False, stop_time="Default"):
        """
        Initializes the model and computes initial values for all variables.
        Additionally calls the setup experiment, if not already called.
        
        Parameters::
            
            tolerance_defined --
                Specifies if the model is to be solved with an error
                controlled algorithm.
                Default: True
                
            tolerance --
                The tolerance used by the error controlled algorithm.
                Default: The tolerance defined in the model description
            
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

        Calls the low-level FMI functions: fmi2_import_setup_experiment (optionally)
                                           fmi2EnterInitializationMode,
                                           fmi2ExitInitializationMode
        """
        log_open = self._log_open()
        if not log_open and self.get_log_level() > 2:
            self._open_log_file()
        
        try:
            if self.time == None:
                self.setup_experiment(tolerance_defined, tolerance, start_time, stop_time_defined, stop_time)
            
            self.enter_initialization_mode()
            self.exit_initialization_mode()
        except:
            if not log_open and self.get_log_level() > 2:
                self._close_log_file()
                
            raise
        
        if not log_open and self.get_log_level() > 2:
            self._close_log_file()

    def set_fmil_log_level(self, FMIL.jm_log_level_enu_t level):
        logging.warning("The method 'set_fmil_log_level' is deprecated, use 'set_log_level' instead.")
        self.set_log_level(level)

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

        Calls the low-level FMI function: fmi2SetDebuggLogging
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
        
            A list with the categories available for logging.
        """
        cdef FMIL.size_t i, nbr_categories = FMIL.fmi2_import_get_log_categories_num(self._fmu)
        cdef list categories = []
        
        for i in range(nbr_categories):
            categories.append(FMIL.fmi2_import_get_log_category(self._fmu, i))

        return categories

    @enable_caching
    def get_variable_nominal(self, variable_name=None, valueref=None, _override_erroneous_nominal=True):
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
        cdef FMIL.fmi2_import_variable_t*      variable
        cdef FMIL.fmi2_import_real_variable_t* real_variable
        cdef FMIL.fmi2_real_t value
        cdef char* variablename = NULL

        if valueref != None:
            variable = FMIL.fmi2_import_get_variable_by_vr(self._fmu, FMIL.fmi2_base_type_real, <FMIL.fmi2_value_reference_t>valueref)
            if variable == NULL:
                raise FMUException("The variable with value reference: %s, could not be found."%str(valueref))
        elif variable_name != None:
            variable_name = encode(variable_name)
            variablename = variable_name
            
            variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
            if variable == NULL:
                raise FMUException("The variable %s could not be found."%variablename)
        else:
            raise FMUException('Either provide value reference or variable name.')

        real_variable = FMIL.fmi2_import_get_variable_as_real(variable)
        if real_variable == NULL:
            raise FMUException("The variable is not a real variable.")
        
        value = FMIL.fmi2_import_get_real_variable_nominal(real_variable) 
        
        if _override_erroneous_nominal:
            if variable_name == None:
                variable_name = encode(self.get_variable_by_valueref(valueref))
                variablename = variable_name
            if value == 0.0:
                if self.callbacks.log_level >= FMIL.jm_log_level_warning:
                    logging.warning("The nominal value for %s is 0.0 which is illegal according to the FMI specification. Setting the nominal to 1.0"%variablename)
                value = 1.0
            elif value < 0.0:
                if self.callbacks.log_level >= FMIL.jm_log_level_warning:
                    logging.warning("The nominal value for %s is <0.0 which is illegal according to the FMI specification. Setting the nominal to abs(%g)"%(variablename, value))
                value = abs(value)
                
        return value

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

        variable = FMIL.fmi2_import_get_variable_by_vr(self._fmu, <FMIL.fmi2_base_type_enu_t> type, <FMIL.fmi2_value_reference_t> valueref)
        if variable == NULL:
            raise FMUException("The variable with the valuref %i could not be found."%valueref)

        name = decode(FMIL.fmi2_import_get_variable_name(variable))

        return name

    def get_variable_alias_base(self, variable_name):
        """
        Returns the base variable for the provided variable name.

        Parameters::

            variable_name--
                Name of the variable.

        Returns:

           The base variable.
        """
        cdef FMIL.fmi2_import_variable_t* variable
        cdef FMIL.fmi2_import_variable_t* base_variable
        cdef FMIL.fmi2_value_reference_t vr
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        base_variable = FMIL.fmi2_import_get_variable_alias_base(self._fmu, variable)
        if base_variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        name = decode(FMIL.fmi2_import_get_variable_name(base_variable))

        return name

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
            The values indicates whether or not the variable should be negated or not.

        Raises::

            FMUException if the variable is not in the model.
        """
        cdef FMIL.fmi2_import_variable_t         *variable
        cdef FMIL.fmi2_import_variable_list_t    *alias_list
        cdef FMIL.size_t                         alias_list_size
        cdef FMIL.fmi2_variable_alias_kind_enu_t alias_kind
        cdef dict                                ret_values = {}
        cdef FMIL.size_t                         i
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        alias_list = FMIL.fmi2_import_get_variable_aliases(self._fmu, variable)

        alias_list_size = FMIL.fmi2_import_get_variable_list_size(alias_list)

        #Loop over all the alias variables
        for i in range(alias_list_size):

            variable = FMIL.fmi2_import_get_variable(alias_list, i)

            alias_kind = FMIL.fmi2_import_get_variable_alias_kind(variable)
            alias_name = decode(FMIL.fmi2_import_get_variable_name(variable))

            ret_values[alias_name] = alias_kind

        #FREE VARIABLE LIST
        FMIL.fmi2_import_free_variable_list(alias_list)

        return ret_values
    
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
        cdef FMIL.fmi2_import_variable_t*        variable
        cdef FMIL.fmi2_import_variable_list_t*   variable_list
        cdef FMIL.size_t                         variable_list_size
        cdef FMIL.fmi2_value_reference_t         value_ref
        cdef FMIL.fmi2_base_type_enu_t           data_type,        target_type
        cdef FMIL.fmi2_variability_enu_t         data_variability, target_variability
        cdef FMIL.fmi2_variable_alias_kind_enu_t alias_kind
        cdef int   i
        cdef int  selected_filter = 1 if filter else 0
        cdef list filter_list = []
        cdef int  length_filter = 0
        cdef list real_var_ref = []
        cdef list int_var_ref = []
        cdef list bool_var_ref = []
        cdef dict added_vars = {}

        variable_list      = FMIL.fmi2_import_get_variable_list(self._fmu, 0)
        variable_list_size = FMIL.fmi2_import_get_variable_list_size(variable_list)

        if selected_filter:
            filter_list = self._convert_filter(filter)
            length_filter = len(filter_list)

        for i in range(variable_list_size):

            variable = FMIL.fmi2_import_get_variable(variable_list, i)

            alias_kind       = FMIL.fmi2_import_get_variable_alias_kind(variable)
            name             = decode(FMIL.fmi2_import_get_variable_name(variable))
            value_ref        = FMIL.fmi2_import_get_variable_vr(variable)
            data_type        = FMIL.fmi2_import_get_variable_base_type(variable)
            data_variability = FMIL.fmi2_import_get_variability(variable)

            if data_type != FMIL.fmi2_base_type_real and data_type != FMIL.fmi2_base_type_int and data_type != FMIL.fmi2_base_type_bool and data_type != FMIL.fmi2_base_type_enum:
                continue

            if data_variability != FMIL.fmi2_variability_enu_continuous and data_variability != FMIL.fmi2_variability_enu_discrete and data_variability != FMIL.fmi2_variability_enu_tunable:
                continue
            
            if selected_filter:
                for j in range(length_filter):
                    if filter_list[j].match(name):
                        break
                else:
                    continue
                if added_vars.has_key(value_ref):
                    continue
                else:
                    added_vars[value_ref] = 1
            else:
                if alias_kind != FMIL.fmi2_variable_is_not_alias:
                    continue

            if data_type == FMIL.fmi2_base_type_real:
                real_var_ref.append(value_ref)
            if data_type == FMIL.fmi2_base_type_int or data_type == FMIL.fmi2_base_type_enum:
                int_var_ref.append(value_ref)
            if data_type == FMIL.fmi2_base_type_bool:
                bool_var_ref.append(value_ref)

        #Free the variable list
        FMIL.fmi2_import_free_variable_list(variable_list)

        return real_var_ref, int_var_ref, bool_var_ref

    @enable_caching
    def get_model_variables(self, type = None, int include_alias = True,
                             causality = None,   variability = None,
                            int only_start = False,  int only_fixed = False,
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
        cdef FMIL.fmi2_import_variable_t*        variable
        cdef FMIL.fmi2_import_variable_list_t*   variable_list
        cdef FMIL.size_t                         variable_list_size
        cdef FMIL.fmi2_value_reference_t         value_ref
        cdef FMIL.fmi2_base_type_enu_t           data_type,        target_type = FMIL.fmi2_base_type_real
        cdef FMIL.fmi2_variability_enu_t         data_variability, target_variability = FMIL.fmi2_variability_enu_constant
        cdef FMIL.fmi2_causality_enu_t           data_causality,   target_causality = FMIL.fmi2_causality_enu_parameter
        cdef FMIL.fmi2_variable_alias_kind_enu_t alias_kind
        cdef FMIL.fmi2_initial_enu_t             initial
        cdef char* desc
        cdef int   selected_type = 0        #If a type has been selected
        cdef int   selected_variability = 0 #If a variability has been selected
        cdef int   selected_causality = 0   #If a causality has been selected
        cdef int   has_start, is_fixed
        cdef int   i, j
        cdef int  selected_filter = 1 if filter else 0
        cdef int  length_filter = 0
        cdef list filter_list, variable_return_list = []
        variable_dict = OrderedDict()

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
            name             = decode(FMIL.fmi2_import_get_variable_name(variable))
            value_ref        = FMIL.fmi2_import_get_variable_vr(variable)
            data_type        = FMIL.fmi2_import_get_variable_base_type(variable)
            has_start        = FMIL.fmi2_import_get_variable_has_start(variable)  #fmi2_import_get_initial, may be of interest
            data_variability = FMIL.fmi2_import_get_variability(variable)
            data_causality   = FMIL.fmi2_import_get_causality(variable)
            desc             = FMIL.fmi2_import_get_variable_description(variable)
            initial          = FMIL.fmi2_import_get_initial(variable)

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
                    #if re.match(filter_list[j], name):
                    if filter_list[j].match(name):
                        break
                else:
                    continue

            if include_alias:
                if _as_list:
                    variable_return_list.append(ScalarVariable2(name,
                                       value_ref, data_type, desc.decode('UTF-8') if desc!=NULL else "",
                                       data_variability, data_causality,
                                       alias_kind, initial))
                else:
                    variable_dict[name] = ScalarVariable2(name,
                                       value_ref, data_type, desc.decode('UTF-8') if desc!=NULL else "",
                                       data_variability, data_causality,
                                       alias_kind, initial)
            elif alias_kind == FMIL.fmi2_variable_is_not_alias:
                #Exclude alias
                if _as_list:
                    variable_return_list.append(ScalarVariable2(name,
                                       value_ref, data_type, desc.decode('UTF-8') if desc!=NULL else "",
                                       data_variability, data_causality,
                                       alias_kind, initial))
                else:
                    variable_dict[name] = ScalarVariable2(name,
                                       value_ref, data_type, desc.decode('UTF-8') if desc!=NULL else "",
                                       data_variability, data_causality,
                                       alias_kind, initial)

        #Free the variable list
        FMIL.fmi2_import_free_variable_list(variable_list)
        
        if _as_list:
            return variable_return_list
        else:
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

    cpdef FMIL.fmi2_value_reference_t get_variable_valueref(self, variable_name) except *:
        """
        Extract the ValueReference given a variable name.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The ValueReference for the variable passed as argument.
        """
        cdef FMIL.fmi2_import_variable_t* variable
        cdef FMIL.fmi2_value_reference_t  vr
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
        vr =  FMIL.fmi2_import_get_variable_vr(variable)

        return vr
        
    def get_variable_declared_type(self, variable_name):
        """
        Return the given variables declared type.
        
        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The name of the declared type.
        """
        cdef FMIL.fmi2_import_variable_t* variable
        cdef FMIL.fmi2_value_reference_t  vr
        cdef FMIL.fmi2_import_variable_typedef_t* variable_type
        cdef FMIL.fmi2_base_type_enu_t    type
        cdef FMIL.fmi2_import_enumeration_typedef_t * enumeration_type
        cdef FMIL.fmi2_import_integer_typedef_t * integer_type
        cdef FMIL.fmi2_import_real_typedef_t * real_type
        cdef FMIL.fmi2_import_unit_t * type_unit
        cdef FMIL.fmi2_import_display_unit_t * type_display_unit
        cdef char * type_name
        cdef char * type_desc
        cdef object ret_type, min_val, max_val, unbounded, nominal_val
        cdef char * type_quantity
        cdef unsigned int enum_size
        cdef int item_value
        cdef char * item_desc
        cdef char * item_name
        cdef char * type_unit_name
        cdef char * type_display_unit_name
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
            
        variable_type = FMIL.fmi2_import_get_variable_declared_type(variable)
        if variable_type == NULL:
            raise FMUException("The variable %s does not have a declared type."%variablename)
        
        type_name = FMIL.fmi2_import_get_type_name(variable_type)
        type_desc = FMIL.fmi2_import_get_type_description(variable_type)
        type_quantity = FMIL.fmi2_import_get_type_quantity(variable_type)
        
        type = FMIL.fmi2_import_get_variable_base_type(variable)
        
        if type == FMIL.fmi2_base_type_enum:
            enumeration_type  = FMIL.fmi2_import_get_type_as_enum(variable_type)
            enum_size = FMIL.fmi2_import_get_enum_type_size(enumeration_type)
            items = OrderedDict()
            
            for i in range(1,enum_size+1):
                item_value = FMIL.fmi2_import_get_enum_type_item_value(enumeration_type, i)
                item_name  = FMIL.fmi2_import_get_enum_type_item_name(enumeration_type, i)
                item_desc  = FMIL.fmi2_import_get_enum_type_item_description(enumeration_type, i)
                
                items[item_value] = (item_name if item_name != NULL else "",
                                     item_desc if item_desc != NULL else "")
                                     
            ret_type = EnumerationType2(type_name if type_name != NULL else "",
                                         type_desc if type_desc != NULL else "",
                                         type_quantity if type_quantity != NULL else "", items)
                                         
                                         
        elif type == FMIL.fmi2_base_type_int:
            integer_type = FMIL.fmi2_import_get_type_as_int(variable_type)
            
            min_val = FMIL.fmi2_import_get_integer_type_min(integer_type)
            max_val = FMIL.fmi2_import_get_integer_type_max(integer_type)
            
            ret_type = IntegerType2(type_name if type_name != NULL else "",
                                         type_desc if type_desc != NULL else "",
                                         type_quantity if type_quantity != NULL else "",
                                         min_val, max_val)
        elif type == FMIL.fmi2_base_type_real:
            real_type = FMIL.fmi2_import_get_type_as_real(variable_type)
            
            min_val = FMIL.fmi2_import_get_real_type_min(real_type)
            max_val = FMIL.fmi2_import_get_real_type_max(real_type)
            nominal_val = FMIL.fmi2_import_get_real_type_nominal(real_type)
            unbounded = FMIL.fmi2_import_get_real_type_is_unbounded(real_type)
            relative_quantity = FMIL.fmi2_import_get_real_type_is_relative_quantity(real_type)
            
            type_display_unit = FMIL.fmi2_import_get_type_display_unit(real_type)
            type_unit = FMIL.fmi2_import_get_real_type_unit(real_type)
            
            type_unit_name = FMIL.fmi2_import_get_unit_name(type_unit)
            type_display_unit_name = FMIL.fmi2_import_get_display_unit_name(type_display_unit)
            
            ret_type = RealType2(type_name if type_name != NULL else "",
                                         type_desc if type_desc != NULL else "",
                                         type_quantity if type_quantity != NULL else "",
                                         min_val, max_val, nominal_val, unbounded, relative_quantity,
                                         type_display_unit_name if type_display_unit_name != NULL else "", type_unit_name if type_unit_name != NULL else "")
        else:
            raise NotImplementedError
        
        return ret_type
    
    def get_scalar_variable(self, variable_name):
        """
        The variable as a scalar variable.
        
        Parameter::
        
            variable_name --
                The name of the variable
                
        Returns::
            
            Instance of a ScalarVariable2
        """
        cdef FMIL.fmi2_import_variable_t* variable

        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
            
        return self._add_scalar_variable(variable)

    cpdef FMIL.fmi2_base_type_enu_t get_variable_data_type(self, variable_name) except *:
        """
        Get data type of variable.

        Parameter::

            variable_name --
                The name of the variable.

        Returns::

            The type of the variable.
        """
        cdef FMIL.fmi2_import_variable_t* variable
        cdef FMIL.fmi2_base_type_enu_t    type
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        type = FMIL.fmi2_import_get_variable_base_type(variable)

        return type

    cpdef get_variable_description(self, variable_name):
        """
        Get the description of a given variable.

        Parameter::

            variable_name --
                The name of the variable

        Returns::

            The description of the variable.
        """
        cdef FMIL.fmi2_import_variable_t* variable
        cdef char* desc
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        desc = FMIL.fmi2_import_get_variable_description(variable)

        return decode(desc) if desc != NULL else ""

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
        cdef FMIL.fmi2_import_variable_t* variable
        cdef FMIL.fmi2_variability_enu_t variability
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
        variability = FMIL.fmi2_import_get_variability(variable)

        return variability
    
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
        cdef FMIL.fmi2_import_variable_t* variable
        cdef FMIL.fmi2_initial_enu_t initial
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name
        
        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
        initial = FMIL.fmi2_import_get_initial(variable)
        
        return initial
    
    def get_variable_unit(self, variable_name):
        """
        Get the unit of the variable.
        
        Parameters::
        
            variable_name --
                The name of the variable.
                
        Returns::
        
            String representing the unit of the variable.
        """
        cdef FMIL.fmi2_import_variable_t* variable
        cdef FMIL.fmi2_import_real_variable_t* real_variable
        cdef FMIL.fmi2_import_unit_t* unit
        cdef FMIL.fmi2_base_type_enu_t    type
        cdef char* unit_description
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name
        
        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
            
        type = FMIL.fmi2_import_get_variable_base_type(variable)
        if type != FMIL.fmi2_base_type_real:
            raise FMUException("The variable %s is not a Real variable. Units only exists for Real variables."%variablename)
            
        real_variable = FMIL.fmi2_import_get_variable_as_real(variable)
        
        unit = FMIL.fmi2_import_get_real_variable_unit(real_variable)
        if unit == NULL:
            raise FMUException("No unit was found for the variable %s."%variablename)
        
        unit_description = FMIL.fmi2_import_get_unit_name(unit)
        
        return decode(unit_description) if unit_description != NULL else ""
        
    def get_variable_display_unit(self, variable_name):
        """
        Get the display unit of the variable.
        
        Parameters::
        
            variable_name --
                The name of the variable.
                
        Returns::
        
            String representing the display unit of the variable.
        """
        cdef FMIL.fmi2_import_variable_t* variable
        cdef FMIL.fmi2_import_real_variable_t* real_variable
        cdef FMIL.fmi2_import_display_unit_t* display_unit
        cdef FMIL.fmi2_base_type_enu_t    type
        cdef char* display_unit_description
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name
        
        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
            
        type = FMIL.fmi2_import_get_variable_base_type(variable)
        if type != FMIL.fmi2_base_type_real:
            raise FMUException("The variable %s is not a Real variable. Display units only exists for Real variables."%variablename)
            
        real_variable = FMIL.fmi2_import_get_variable_as_real(variable)
        
        display_unit = FMIL.fmi2_import_get_real_variable_display_unit(real_variable)
        if display_unit == NULL:
            raise FMUException("No display unit was found for the variable %s."%variablename)
        
        display_unit_description = FMIL.fmi2_import_get_display_unit_name(display_unit)
        
        return decode(display_unit_description) if display_unit_description != NULL else ""
        
    def get_variable_display_value(self, variable_name):
        """
        Get the display value of the variable. This value takes into account
        the display unit (i.e. converts the value in its base unit to the
        value in its display unit.)
        
        Parameters::
        
            variable_name --
                The name of the variable.
                
        Returns::
        
            Variable value in its display unit (raises exception if no display unit).
        """
        cdef FMIL.fmi2_import_variable_t* variable
        cdef FMIL.fmi2_import_display_unit_t* display_unit
        cdef FMIL.fmi2_import_variable_typedef_t* variable_typedef
        cdef FMIL.fmi2_import_real_typedef_t* variable_real_typedef
        cdef FMIL.fmi2_real_t display_value, value
        cdef FMIL.fmi2_value_reference_t  vr
        cdef int relative_quantity
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name
        
        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
            
        type = FMIL.fmi2_import_get_variable_base_type(variable)
        if type != FMIL.fmi2_base_type_real:
            raise FMUException("The variable %s is not a Real variable. Display units only exists for Real variables."%variablename)
        
        real_variable = FMIL.fmi2_import_get_variable_as_real(variable)
        display_unit  = FMIL.fmi2_import_get_real_variable_display_unit(real_variable)
        if display_unit == NULL:
            raise FMUException("No display unit was found for the variable %s."%variablename)
        
        variable_typedef = FMIL.fmi2_import_get_variable_declared_type(variable)
        if variable_typedef == NULL:
            #Use the default value for relative quantity (False)
            relative_quantity = 0
        else:
            variable_real_typedef = FMIL.fmi2_import_get_type_as_real(variable_typedef)
            if variable_real_typedef == NULL:
                #Use the default value for relative quantity (False)
                relative_quantity = 0
            else:
                relative_quantity = FMIL.fmi2_import_get_real_type_is_relative_quantity(variable_real_typedef)
        
        vr = FMIL.fmi2_import_get_variable_vr(variable)
        value = self.get_real(vr)
        
        display_value = FMIL.fmi2_import_convert_to_display_unit(value, display_unit, relative_quantity)
        
        return display_value

    cpdef FMIL.fmi2_causality_enu_t get_variable_causality(self, variable_name) except *:
        """
        Get the causality of the variable.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The causality of the variable, PARAMETER(0), CALCULATED_PARAMETER(1), INPUT(2),
            OUTPUT(3), LOCAL(4), INDEPENDENT(5), UNKNOWN(6)
        """
        cdef FMIL.fmi2_import_variable_t* variable
        cdef FMIL.fmi2_causality_enu_t causality
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL.fmi2_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        causality = FMIL.fmi2_import_get_causality(variable)

        return causality

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
        cdef FMIL.fmi2_import_variable_t *        variable
        cdef FMIL.fmi2_base_type_enu_t            type
        cdef FMIL.fmi2_import_integer_variable_t* int_variable
        cdef FMIL.fmi2_import_real_variable_t*    real_variable
        cdef FMIL.fmi2_import_bool_variable_t*    bool_variable
        cdef FMIL.fmi2_import_enum_variable_t*    enum_variable
        cdef FMIL.fmi2_import_string_variable_t*  str_variable
        cdef int                                  status
        cdef FMIL.fmi2_boolean_t                  FMITRUE = 1
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name
        
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

    cpdef get_variable_max(self, variable_name):
        """
        Returns the maximum value for the given variable.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The maximum value for the variable.
        """
        cdef FMIL.fmi2_import_variable_t*         variable
        cdef FMIL.fmi2_import_integer_variable_t* int_variable
        cdef FMIL.fmi2_import_real_variable_t*    real_variable
        cdef FMIL.fmi2_import_enum_variable_t*    enum_variable
        cdef FMIL.fmi2_base_type_enu_t            type
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

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

    cpdef get_variable_min(self, variable_name):
        """
        Returns the minimum value for the given variable.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The minimum value for the variable.
        """
        cdef FMIL.fmi2_import_variable_t*         variable
        cdef FMIL.fmi2_import_integer_variable_t* int_variable
        cdef FMIL.fmi2_import_real_variable_t*    real_variable
        cdef FMIL.fmi2_import_enum_variable_t*    enum_variable
        cdef FMIL.fmi2_base_type_enu_t            type
        
        variable_name = encode(variable_name)
        cdef char* variablename = variable_name

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

    def get_fmu_state(self, FMUState2 state = None):
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
        cdef int status
        
        if state is None:
            state = FMUState2()

        if not self._supports_get_set_FMU_state():
            raise FMUException('This FMU does not support get and set FMU-state')

        status = FMIL.fmi2_import_get_fmu_state(self._fmu, &(state.fmu_state))

        if status != 0:
            raise FMUException('An error occured while trying to get the FMU-state, see the log for possible more information')
        
        return state

    def set_fmu_state(self, FMUState2 state):
        """
        Set the FMU to a previous saved state.

        Parameter::

            state--
                A pointer to a FMU-state.

        Example::

            FMU_state = Model.get_fmu_state()
            Model.set_fmu_state(FMU_state)
        """
        cdef int status
        cdef FMIL.fmi2_FMU_state_t internal_state = state.fmu_state

        if not self._supports_get_set_FMU_state():
            raise FMUException('This FMU dos not support get and set FMU-state')

        status = FMIL.fmi2_import_set_fmu_state(self._fmu, internal_state)

        if status != 0:
            raise FMUException('An error occured while trying to set the FMU-state, see the log for possible more information')

    def free_fmu_state(self, FMUState2 state):
        """
        Free a previously saved FMU-state from the memory.

        Parameters::

            state--
                A pointer to the FMU-state to be set free.

        Example::

            FMU_state = Model.get_fmu_state()
            Model.free_fmu_state(FMU_state)

        """
        cdef int status
        cdef FMIL.fmi2_FMU_state_t internal_state = state.fmu_state
        
        if not self._supports_get_set_FMU_state():
            raise FMUException('This FMU does not support get and set FMU-state')
        if internal_state == NULL:
            print("FMU-state does not seem to be allocated.")
            return

        status = FMIL.fmi2_import_free_fmu_state(self._fmu, &internal_state)

        if status != 0:
            raise FMUException('An error occured while trying to free the FMU-state, see the log for possible more information')
        
        #Memory has been released
        state.fmu_state = NULL

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

            The size of the vector.
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
        
    def get_default_experiment_step(self):
        """
        Returns the default experiment step as defined in the XML
        description.
        """
        return FMIL.fmi2_import_get_default_experiment_step(self._fmu)

    cdef _add_scalar_variables(self, FMIL.fmi2_import_variable_list_t*   variable_list):
        """
        Helper method to create scalar variables from a variable list.
        """
        cdef FMIL.size_t             variable_list_size
        variable_dict = OrderedDict()

        variable_list_size = FMIL.fmi2_import_get_variable_list_size(variable_list)

        for i in range(variable_list_size):

            variable = FMIL.fmi2_import_get_variable(variable_list, i)
            scalar_variable = self._add_scalar_variable(variable)
            variable_dict[scalar_variable.name] = scalar_variable

        return variable_dict
    
    cdef _add_scalar_variable(self, FMIL.fmi2_import_variable_t* variable):
        
        if variable == NULL:
            raise FMUException("Unknown variable. Please verify the correctness of the XML file and check the log.")

        alias_kind       = FMIL.fmi2_import_get_variable_alias_kind(variable)
        name             = decode(FMIL.fmi2_import_get_variable_name(variable))
        value_ref        = FMIL.fmi2_import_get_variable_vr(variable)
        data_type        = FMIL.fmi2_import_get_variable_base_type(variable)
        data_variability = FMIL.fmi2_import_get_variability(variable)
        data_causality   = FMIL.fmi2_import_get_causality(variable)
        desc             = FMIL.fmi2_import_get_variable_description(variable)

        return ScalarVariable2(name, value_ref, data_type, desc.decode('UTF-8') if desc!=NULL else "",
                                    data_variability, data_causality,
                                    alias_kind)
    
    def get_derivatives_list(self):
        """
        Returns a dictionary with the states derivatives.

        Returns::

            An ordered dictionary with the derivative variables.
        """
        cdef FMIL.fmi2_import_variable_list_t*   variable_list

        variable_list = FMIL.fmi2_import_get_derivatives_list(self._fmu)
        if variable_list == NULL:
            raise FMUException("The returned states list is NULL.")

        variable_dict = self._add_scalar_variables(variable_list)

        #Free the variable list
        FMIL.fmi2_import_free_variable_list(variable_list)

        return variable_dict
        
    cpdef get_output_dependencies(self):
        """
        Retrieve the list of variables that the outputs are 
        dependent on. Returns two dictionaries, one with the states and 
        one with the inputs.
        """
        if (self._outputs_states_dependencies is not None and
            self._outputs_inputs_dependencies is not None):
               return self._outputs_states_dependencies, self._outputs_inputs_dependencies 
               
        cdef size_t *dependencyp
        cdef size_t *start_indexp
        cdef char   *factor_kindp
        cdef FMIL.fmi2_import_variable_t *variable
        cdef FMIL.fmi2_import_variable_list_t *variable_list

        if python3_flag:
            outputs = list(self.get_output_list().keys())
        else:
            outputs = self.get_output_list().keys()
        states_list = self.get_states_list()
        inputs_list = self.get_input_list()
        
        states = OrderedDict()
        inputs = OrderedDict()
        
        if len(outputs) != 0: #If there are no outputs, return empty dicts
        
            FMIL.fmi2_import_get_outputs_dependencies(self._fmu, &start_indexp, &dependencyp, &factor_kindp)
            
            if start_indexp == NULL:
                logging.warning(
                        'No dependency information for the outputs was found in the model description.' \
                        ' Assuming complete dependency with the exception if the output is a state by itself.')
                for i in range(0,len(outputs)):
                    if outputs[i] in states_list: #The output is a state in itself
                        states[outputs[i]]  = [outputs[i]]
                        inputs[outputs[i]]  = []
                    else:
                        states[outputs[i]]  = states_list.keys()
                        inputs[outputs[i]]  = inputs_list.keys()
            else:
                variable_list = FMIL.fmi2_import_get_variable_list(self._fmu, 0)
                if variable_list == NULL:
                    raise FMUException("The returned variable list is NULL.")
                
                for i in range(0,len(outputs)):
                    states[outputs[i]]  = []
                    inputs[outputs[i]] = []
                    
                    for j in range(0, start_indexp[i+1]-start_indexp[i]):
                        if dependencyp[start_indexp[i]+j] != 0:
                            variable = FMIL.fmi2_import_get_variable(variable_list, dependencyp[start_indexp[i]+j]-1)
                            name             = decode(FMIL.fmi2_import_get_variable_name(variable))
                            
                            if name in states_list:
                                states[outputs[i]].append(name)
                            elif name in inputs_list:
                                inputs[outputs[i]].append(name)                        
                
                #Free the variable list
                FMIL.fmi2_import_free_variable_list(variable_list)
                
        #Caching
        self._outputs_states_dependencies = states
        self._outputs_inputs_dependencies = inputs
               
        return states, inputs
        
    cpdef get_derivatives_dependencies(self):
        """
        Retrieve the list of variables that the derivatives are 
        dependent on. Returns two dictionaries, one with the states 
        and one with the inputs.
        """ 
        if (self._derivatives_states_dependencies is not None and
            self._derivatives_inputs_dependencies is not None):
               return self._derivatives_states_dependencies, self._derivatives_inputs_dependencies 
       
        cdef size_t *dependencyp
        cdef size_t *start_indexp
        cdef char   *factor_kindp
        cdef FMIL.fmi2_import_variable_t *variable
        cdef FMIL.fmi2_import_variable_list_t *variable_list

        if python3_flag:
            derivatives = list(self.get_derivatives_list().keys())
        else:
            derivatives = self.get_derivatives_list().keys()
        states_list = self.get_states_list()
        inputs_list = self.get_input_list()
        
        states = OrderedDict()
        inputs = OrderedDict()
        
        if len(derivatives) != 0:
            FMIL.fmi2_import_get_derivatives_dependencies(self._fmu, &start_indexp, &dependencyp, &factor_kindp)
            
            if start_indexp == NULL:
                logging.warning(
                    'No dependency information for the derivatives was found in the model description.' \
                    ' Assuming complete dependency.')
                for i in range(0,len(derivatives)):
                    states[derivatives[i]]  = states_list.keys()
                    inputs[derivatives[i]]  = inputs_list.keys()
            else:
                variable_list = FMIL.fmi2_import_get_variable_list(self._fmu, 0)
                if variable_list == NULL:
                    raise FMUException("The returned variable list is NULL.")
                
                for i in range(0,len(derivatives)):
                    states[derivatives[i]]  = []
                    inputs[derivatives[i]] = []
                    
                    for j in range(0, start_indexp[i+1]-start_indexp[i]):
                        if dependencyp[start_indexp[i]+j] != 0:
                            variable = FMIL.fmi2_import_get_variable(variable_list, dependencyp[start_indexp[i]+j]-1)
                            name             = decode(FMIL.fmi2_import_get_variable_name(variable))
                            
                            if name in states_list:
                                states[derivatives[i]].append(name)
                            elif name in inputs_list:
                                inputs[derivatives[i]].append(name)                        
                
                #Free the variable list
                FMIL.fmi2_import_free_variable_list(variable_list)
                
        #Caching
        self._derivatives_states_dependencies = states
        self._derivatives_inputs_dependencies = inputs
            
        return states, inputs
        
    def _get_directional_proxy(self, var_ref, func_ref, group=None, add_diag=False, output_matrix=None):
        cdef list data = [], row = [], col = []
        cdef list local_group
        cdef int nbr_var_ref  = len(var_ref), nbr_func_ref = len(func_ref)
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c'] v = N.zeros(nbr_var_ref, dtype = N.double)
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c'] data_local
        cdef int ind_local = 5 if add_diag else 4
        
        if not self._has_entered_init_mode:
            raise FMUException("The FMU has not entered initialization mode and thus the directional " \
                               "derivatives cannot be computed. Call enter_initialization_mode to start the initialization.")

        if group is not None:
            if output_matrix is not None:
                if not isinstance(output_matrix, sp.csc_matrix):
                    output_matrix = None
                else:
                    data_local = output_matrix.data
                
            if add_diag and output_matrix is None:
                dim = min(nbr_var_ref,nbr_func_ref)
                data.extend([0.0]*dim)
                row.extend(range(dim))
                col.extend(range(dim))
            
            for key in group["groups"]:
                local_group = group[key]
                
                v[local_group[0]] = 1.0
                
                column_data = self.get_directional_derivative(var_ref, func_ref, v)[local_group[2]]
                
                if output_matrix is None:
                    data.extend(column_data)
                    row.extend(local_group[2])
                    col.extend(local_group[3])
                else:
                    data_local[local_group[ind_local]] = column_data
                
                v[local_group[0]] = 0.0
            
            if output_matrix is not None:
                A = output_matrix 
            else:
                if len(data) == 0:
                    A = sp.csc_matrix((nbr_func_ref,nbr_var_ref))
                else:
                    A = sp.csc_matrix((data, (row, col)), (nbr_func_ref,nbr_var_ref))
                
            return A
        else:
            if output_matrix is None or \
                (not isinstance(output_matrix, N.ndarray)) or \
                (isinstance(output_matrix, N.ndarray) and (output_matrix.shape[0] != nbr_func_ref or output_matrix.shape[1] != nbr_var_ref)):
                    A = N.zeros((nbr_func_ref,nbr_var_ref))
            else:
                A = output_matrix
            
            for i in range(nbr_var_ref):
                v[i] = 1.0
                A[:, i] = self.get_directional_derivative(var_ref, func_ref, v)
                v[i] = 0.0
            return A
        
    def _get_A(self, use_structure_info=True, add_diag=True, output_matrix=None):
        if self._group_A is None and use_structure_info:
            [derv_state_dep, derv_input_dep] = self.get_derivatives_dependencies()
            if python3_flag:
                self._group_A = cpr_seed(derv_state_dep, list(self.get_states_list().keys()))
            else:
                self._group_A = cpr_seed(derv_state_dep, self.get_states_list().keys())
        if self._states_references is None:
            states                       = self.get_states_list()
            self._states_references      = [s.value_reference for s in states.values()]
        if self._derivatives_references is None:
            derivatives                  = self.get_derivatives_list()
            self._derivatives_references = [s.value_reference for s in derivatives.values()]
        
        A = self._get_directional_proxy(self._states_references, self._derivatives_references, self._group_A if use_structure_info else None, add_diag=add_diag, output_matrix=output_matrix)
        
        if self._A is None:
            self._A = A
        
        return A
        
    def _get_B(self, use_structure_info=True, add_diag=False, output_matrix=None):
        if self._group_B is None and use_structure_info:
            [derv_state_dep, derv_input_dep] = self.get_derivatives_dependencies()
            if python3_flag:
                self._group_B = cpr_seed(derv_input_dep, list(self.get_input_list().keys()))
            else:
                self._group_B = cpr_seed(derv_input_dep, self.get_input_list().keys())
        if self._inputs_references is None:
            inputs                       = self.get_input_list()
            self._inputs_references      = [s.value_reference for s in inputs.values()]
        if self._derivatives_references is None:
            derivatives                  = self.get_derivatives_list()
            self._derivatives_references = [s.value_reference for s in derivatives.values()]
        
        B = self._get_directional_proxy(self._inputs_references, self._derivatives_references, self._group_B if use_structure_info else None, add_diag=add_diag, output_matrix=output_matrix)
        
        if self._B is None:
            self._B = B
        
        return B
        
    def _get_C(self, use_structure_info=True, add_diag=False, output_matrix=None):
        if self._group_C is None and use_structure_info:
            [out_state_dep, out_input_dep] = self.get_output_dependencies()
            if python3_flag:
                self._group_C = cpr_seed(out_state_dep, list(self.get_states_list().keys()))
            else:
                self._group_C = cpr_seed(out_state_dep, self.get_states_list().keys())
        if self._states_references is None:
            states                       = self.get_states_list()
            self._states_references      = [s.value_reference for s in states.values()]
        if self._outputs_references is None:
            outputs                      = self.get_output_list()
            self._outputs_references     = [s.value_reference for s in outputs.values()]
            
        C = self._get_directional_proxy(self._states_references, self._outputs_references, self._group_C if use_structure_info else None, add_diag=add_diag, output_matrix=output_matrix)
        
        if self._C is None:
            self._C = C
        
        return C
        
    def _get_D(self, use_structure_info=True, add_diag=False, output_matrix=None):
        if self._group_D is None and use_structure_info:
            [out_state_dep, out_input_dep] = self.get_output_dependencies()
            if python3_flag:
                self._group_D = cpr_seed(out_input_dep, list(self.get_input_list().keys()))
            else:
                self._group_D = cpr_seed(out_input_dep, self.get_input_list().keys())
        if self._inputs_references is None:
            inputs                       = self.get_input_list()
            self._inputs_references      = [s.value_reference for s in inputs.values()]
        if self._outputs_references is None:
            outputs                      = self.get_output_list()
            self._outputs_references     = [s.value_reference for s in outputs.values()]
            
        D = self._get_directional_proxy(self._inputs_references, self._outputs_references, self._group_D if use_structure_info else None, add_diag=add_diag, output_matrix=output_matrix)
        
        if self._D is None:
            self._D = D
        
        return D
        
        
    def get_state_space_representation(self, A=True, B=True, C=True, D=True, use_structure_info=True):
        """
        Returns a state space representation of the model. I.e::
        
            der(x) = Ax + Bu
                y  = Cx + Du
                
        Which of the matrices to be returned can be choosen by the 
        arguments.
        
        Parameters::
        
            A --
                If the 'A' matrix should be computed or not.
                Default: True
                
            B --
                If the 'B' matrix should be computed or not.
                Default: True
                
            C --
                If the 'C' matrix should be computed or not.
                Default: True
                
            D --
                If the 'D' matrix should be computed or not.
                Default: True
                
            use_structure_info --
                Determines if the structure should be taken into account 
                or not. If so, a sparse representation is returned, 
                otherwise a dense.
                Default: True
                
        Returns::
            The A,B,C,D matrices. If not all are computed, the ones that
            are not computed will be represented by a boolean flag.

        """
        if A:
            A = self._get_A(use_structure_info)
        if B:
            B = self._get_B(use_structure_info)
        if C:
            C = self._get_C(use_structure_info)
        if D:
            D = self._get_D(use_structure_info)
            
        return A,B,C,D
        
    
    def get_states_list(self):
        """
        Returns a dictionary with the states.

        Returns::

            An ordered dictionary with the state variables.
        """
        cdef FMIL.fmi2_import_variable_list_t*   variable_list
        cdef FMIL.size_t             variable_list_size
        variable_dict = OrderedDict()
        
        variable_list = FMIL.fmi2_import_get_derivatives_list(self._fmu)
        variable_list_size = FMIL.fmi2_import_get_variable_list_size(variable_list)
        
        if variable_list == NULL:
            raise FMUException("The returned derivatives states list is NULL.")
        
        for i in range(variable_list_size):
            der_variable = FMIL.fmi2_import_get_variable(variable_list, i)
            variable     = FMIL.fmi2_import_get_real_variable_derivative_of(<FMIL.fmi2_import_real_variable_t*>der_variable)
            
            scalar_variable = self._add_scalar_variable(<FMIL.fmi2_import_variable_t*>variable)
            variable_dict[scalar_variable.name] = scalar_variable

        #Free the variable list
        FMIL.fmi2_import_free_variable_list(variable_list)

        return variable_dict

    def get_input_list(self):
        """
        Returns a dictionary with input variables

        Returns::

            An ordered dictionary with the (real) (continuous) input variables.
        """
        variable_dict = self.get_model_variables(type=FMI2_REAL, include_alias = False,
                             causality = FMI2_INPUT,   variability = FMI2_CONTINUOUS)

        return variable_dict

    def get_output_list(self):
        """
        Returns a dictionary with output variables

        Returns::

            An ordered dictionary with the (real) (continuous) output variables.
        """
        cdef FMIL.fmi2_import_variable_list_t*   variable_list
        cdef FMIL.size_t                         variable_list_size
        cdef FMIL.fmi2_import_variable_t*        variable
        variable_dict = OrderedDict()
        
        variable_list = FMIL.fmi2_import_get_outputs_list(self._fmu)
        variable_list_size = FMIL.fmi2_import_get_variable_list_size(variable_list)
        
        if variable_list == NULL:
            raise FMUException("The returned outputs list is NULL.")
        
        for i in range(variable_list_size):
            variable = FMIL.fmi2_import_get_variable(variable_list, i)
            
            scalar_variable = self._add_scalar_variable(variable)
            variable_dict[scalar_variable.name] = scalar_variable

        #Free the variable list
        FMIL.fmi2_import_free_variable_list(variable_list)

        return variable_dict

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

        cdef int status
        cdef FMIL.size_t nv, nz

        #input arrays
        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1, mode='c'] v_ref = N.zeros(len(var_ref),  dtype = N.uint32)
        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1, mode='c'] z_ref = N.zeros(len(func_ref), dtype = N.uint32)
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c'] dv    = N.zeros(len(v),        dtype = N.double)
        #output array
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c'] dz    = N.zeros(len(func_ref), dtype = N.double)

        if not self._provides_directional_derivatives():
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

        #status = FMIL.fmi2_import_get_directional_derivative(self._fmu, <FMIL.fmi2_value_reference_t*> v_ref.data, nv, <FMIL.fmi2_value_reference_t*> z_ref.data, nz, <FMIL.fmi2_real_t*> dv.data, <FMIL.fmi2_real_t*> dz.data)
        status = self._get_directional_derivative(v_ref, z_ref, dv, dz)

        if status != 0:
            raise FMUException('An error occured while getting the directional derivative, see the log for possible more information')

        return dz
        
    cdef int _get_directional_derivative(self, N.ndarray[FMIL.fmi2_value_reference_t, ndim=1, mode="c"] v_ref, 
                                               N.ndarray[FMIL.fmi2_value_reference_t, ndim=1, mode="c"] z_ref, 
                                               N.ndarray[FMIL.fmi2_real_t, ndim=1, mode="c"] dv, 
                                               N.ndarray[FMIL.fmi2_real_t, ndim=1, mode="c"] dz) except -1:
        cdef int status
        
        assert dv.size >= v_ref.size and dz.size >= z_ref.size
        
        if not self._provides_directional_derivatives():
            raise FMUException('This FMU does not provide directional derivatives')
        
        status = FMIL.fmi2_import_get_directional_derivative(self._fmu, 
                  <FMIL.fmi2_value_reference_t*> v_ref.data, v_ref.size, 
                  <FMIL.fmi2_value_reference_t*> z_ref.data, z_ref.size, 
                  <FMIL.fmi2_real_t*> dv.data, 
                  <FMIL.fmi2_real_t*> dz.data)
        
        return status 

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
        return decode(version)
        
    def get_model_version(self):
        """
        Returns the version of the FMU.
        """
        cdef char* version
        version = FMIL.fmi2_import_get_model_version(self._fmu)
        return decode(version) if version != NULL else ""

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
        return decode(author) if author != NULL else ""

    def get_description(self):
        """
        Return the model description.
        """
        cdef char* desc
        desc = FMIL.fmi2_import_get_description(self._fmu)
        return decode(desc) if desc != NULL else ""
        
    def get_copyright(self):
        """
        Return the model copyright.
        """
        cdef char* copyright
        copyright = FMIL.fmi2_import_get_copyright(self._fmu)
        return decode(copyright) if copyright != NULL else ""
        
    def get_license(self):
        """
        Return the model license.
        """
        cdef char* license
        license = FMIL.fmi2_import_get_license(self._fmu)
        return decode(license) if license != NULL else ""

    def get_generation_tool(self):
        """
        Return the model generation tool.
        """
        cdef char* gen
        gen = FMIL.fmi2_import_get_generation_tool(self._fmu)
        return decode(gen) if gen != NULL else ""
        
    def get_generation_date_and_time(self):
        """
        Return the model generation date and time.
        """
        cdef char* gen
        gen = FMIL.fmi2_import_get_generation_date_and_time(self._fmu)
        return decode(gen) if gen != NULL else ""

    def get_guid(self):
        """
        Return the model GUID.
        """
        guid = decode(FMIL.fmi2_import_get_GUID(self._fmu))
        return guid
        
    def get_variable_naming_convention(self):
        """
        Return the variable naming convention.
        """
        cdef FMIL.fmi2_variable_naming_convension_enu_t conv
        conv = FMIL.fmi2_import_get_naming_convention(self._fmu)
        if conv == FMIL.fmi2_naming_enu_flat:
            return "flat"
        elif conv == FMIL.fmi2_naming_enu_structured:
            return "structured"
        else:
            return "unknown"

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
    def __init__(self, fmu, path = '.', enable_logging = None, log_file_name = "", log_level=FMI_DEFAULT_LOG_LEVEL, _unzipped_dir=None, _connect_dll=True):
        """
        Constructor of the model.

        Parameters::

            fmu --
                Name of the fmu as a string.

            path --
                Path to the fmu-directory.
                Default: '.' (working directory)

            enable_logging [DEPRECATED] --
                This option is DEPRECATED and will be removed. Please 
                use the option "log_level" instead.

            log_file_name --
                Filename for file used to save log messages.
                Default: "" (Generates automatically)
                
            log_level --
                Determines the logging output. Can be set between 0 
                (no logging) and 7 (everything).
                Default: 2 (log error messages)

        Returns::

            A model as an object from the class FMUModelCS2
        """

        #Call super
        FMUModelBase2.__init__(self, fmu, path, enable_logging, log_file_name, log_level, _unzipped_dir, _connect_dll)

        if self._fmu_kind != FMIL.fmi2_fmu_kind_cs:
            if self._fmu_kind != FMIL.fmi2_fmu_kind_me_and_cs:
                raise FMUException("This class only supports FMI 2.0 for Co-simulation.")

        if self.get_capability_flags()['needsExecutionTool'] == True:
            raise FMUException('Models that need an execution tool are not supported')

        self._modelId = decode(FMIL.fmi2_import_get_model_identifier_CS(self._fmu))
        
        if _connect_dll:
            self.instantiate()

    def __dealloc__(self):
        """
        Deallocate memory allocated
        """
        if self._allocated_fmu == 1:
            FMIL.fmi2_import_terminate(self._fmu)
            FMIL.fmi2_import_free_instance(self._fmu)

        if self._allocated_dll == 1:
            FMIL.fmi2_import_destroy_dllfmu(self._fmu)

        if self._allocated_xml == 1:
            FMIL.fmi2_import_free(self._fmu)

        if self._fmu_temp_dir != NULL:
            FMIL.fmi_import_rmdir(&self.callbacks, self._fmu_temp_dir)
            FMIL.free(self._fmu_temp_dir)
            self._fmu_temp_dir = NULL
            
        if self._allocated_context == 1:
            FMIL.fmi_import_free_context(self._context)
            
        if self._fmu_log_name != NULL:
            FMIL.free(self._fmu_log_name)
            self._fmu_log_name = NULL
            
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
    low-level FMI function: fmiSetTime
    """)

    cpdef int do_step(self, FMIL.fmi2_real_t current_t, FMIL.fmi2_real_t step_size, new_step=True):
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

        Calls the underlying low-level function fmi2DoStep.
        """
        cdef int status
        cdef FMIL.fmi2_boolean_t new_s

        if new_step:
            new_s = 1
        else:
            new_s = 0

        self.time = current_t + step_size
        
        log_open = self._log_open()
        if not log_open and self.get_log_level() > 2:
            self._open_log_file()
        
        status = FMIL.fmi2_import_do_step(self._fmu, current_t, step_size, new_s)
        
        if not log_open and self.get_log_level() > 2:
            self._close_log_file()

        return status

    def cancel_step(self):
        """
        Cancel a current integrator step. Can only be called if the
        status from do_step returns FMI_PENDING. After this function has
        been called it is only allowed to reset the model (i.e. start
        over).
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
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c']            val = N.array(values, dtype=N.float, ndmin=1).ravel()

        nref = val.size
        orders = N.array([0]*nref, dtype=N.int32)

        can_interpolate_inputs = FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canInterpolateInputs)
        #NOTE IS THIS THE HIGHEST ORDER OF INTERPOLATION OR SIMPLY IF IT CAN OR NOT?

        if order < 1:
            raise FMUException("The order must be greater than zero.")
        if not can_interpolate_inputs:
            raise FMUException("The FMU does not support input derivatives.")

        if isinstance(variables,str):
            value_refs = N.array([0], dtype=N.uint32, ndmin=1).ravel()
            value_refs[0] = self.get_variable_valueref(variables)
        elif isinstance(variables,list) and N.prod([int(isinstance(v,str)) for v in variables]): #prod equals 0 or 1
            value_refs = N.array([0]*nref, dtype=N.uint32,ndmin=1).ravel()
            for i in range(nref):
                value_refs[i] = self.get_variable_valueref(variables[i])
                orders[i] = order
        else:
            raise FMUException("The variables must either be a string or a list of strings")

        status = self._set_input_derivatives(value_refs, val, orders)
        #status = FMIL.fmi2_import_set_real_input_derivatives(self._fmu, <FMIL.fmi2_value_reference_t*> value_refs.data, nref,
        #                                                        <FMIL.fmi2_integer_t*> orders.data, <FMIL.fmi2_real_t*> val.data)

        if status != 0:
            raise FMUException('Failed to set the Real input derivatives.')
            
    cdef int _set_input_derivatives(self, N.ndarray[FMIL.fmi2_value_reference_t, ndim=1, mode="c"] value_refs, 
                                          N.ndarray[FMIL.fmi2_real_t, ndim=1, mode="c"] values, 
                                          N.ndarray[FMIL.fmi2_integer_t, ndim=1, mode="c"] orders):
        cdef int status
        
        assert values.size >= value_refs.size and orders.size >= value_refs.size
        
        status = FMIL.fmi2_import_set_real_input_derivatives(self._fmu, 
                        <FMIL.fmi2_value_reference_t*> value_refs.data, 
                        value_refs.size, <FMIL.fmi2_integer_t*> orders.data, 
                        <FMIL.fmi2_real_t*> values.data)
        
        return status

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
            value_refs = N.array([0], dtype=N.uint32, ndmin=1).ravel()
            orders = N.array([order], dtype=N.int32)
            value_refs[0] = self.get_variable_valueref(variables)
        elif isinstance(variables,list) and N.prod([int(isinstance(v,str)) for v in variables]): #prod equals 0 or 1
            nref = len(variables)
            value_refs = N.array([0]*nref, dtype=N.uint32, ndmin=1).ravel()
            orders = N.array([0]*nref, dtype=N.int32)
            for i in range(nref):
                value_refs[i] = self.get_variable_valueref(variables[i])
                orders[i] = order
        else:
            raise FMUException("The variables must either be a string or a list of strings")

        values = N.array([0.0]*nref, dtype=N.float, ndmin=1)

        #status = FMIL.fmi2_import_get_real_output_derivatives(self._fmu, <FMIL.fmi2_value_reference_t*> value_refs.data, nref,
        #                                                    <FMIL.fmi2_integer_t*> orders.data, <FMIL.fmi2_real_t*> values.data)
        status = self._get_output_derivatives(value_refs, values, orders)

        if status != 0:
            raise FMUException('Failed to get the Real output derivatives.')

        return values
        
    cdef int _get_output_derivatives(self, N.ndarray[FMIL.fmi2_value_reference_t, ndim=1, mode="c"] value_refs, 
                                           N.ndarray[FMIL.fmi2_real_t, ndim=1, mode="c"] values,
                                           N.ndarray[FMIL.fmi2_integer_t, ndim=1, mode="c"] orders):
        cdef int status
        
        assert values.size >= value_refs.size and orders.size >= value_refs.size
        
        status = FMIL.fmi2_import_get_real_output_derivatives(self._fmu, 
                    <FMIL.fmi2_value_reference_t*> value_refs.data, value_refs.size,
                    <FMIL.fmi2_integer_t*> orders.data, <FMIL.fmi2_real_t*> values.data)
        
        return status


    def get_status(self, status_kind):
        """
        Retrieves the fmi-status for the the specified fmi-staus-kind.

        Parameters::

            status_kind --
                An integer corresponding to one of the following:
                fmi2DoStepStatus       = 0
                fmi2PendingStatus      = 1
                fmi2LastSuccessfulTime = 2
                fmi2Terminated         = 3

        Returns::

            status_ok      = 0
            status_warning = 1
            status_discard = 2
            status_error   = 3
            status_fatal   = 4
            status_pending = 5
        """

        cdef int status
        cdef FMIL.fmi2_status_kind_t fmi_status_kind
        cdef FMIL.fmi2_status_t status_value

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
                 start_time="Default",
                 final_time="Default",
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

                    >> myModel = load_fmu(...)
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
        Get an instance of the simulate options class, filled with default
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
        
    def get_capability_flags(self):
        """
        Returns a dictionary with the capability flags of the FMU.

        Returns::
            Dictionary with keys:
            needsExecutionTool
            canHandleVariableCommunicationStepSize
            canInterpolateInputs
            maxOutputDerivativeOrder
            canRunAsynchronuously
            canBeInstantiatedOnlyOncePerProcess
            canNotUseMemoryManagementFunctions
            canGetAndSetFMUstate
            providesDirectionalDerivatives
        """
        cdef dict capabilities = {}
        capabilities['needsExecutionTool']                     = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_needsExecutionTool))
        capabilities['canHandleVariableCommunicationStepSize'] = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canHandleVariableCommunicationStepSize))
        capabilities['canInterpolateInputs']                   = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canInterpolateInputs))
        capabilities['maxOutputDerivativeOrder']               = FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_maxOutputDerivativeOrder)
        capabilities['canRunAsynchronuously']                  = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canRunAsynchronuously))
        capabilities['canBeInstantiatedOnlyOncePerProcess']    = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canBeInstantiatedOnlyOncePerProcess))
        capabilities['canNotUseMemoryManagementFunctions']     = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canNotUseMemoryManagementFunctions))
        capabilities['canGetAndSetFMUstate']                   = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canGetAndSetFMUstate))
        capabilities['canSerializeFMUstate']                   = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canSerializeFMUstate))
        capabilities['providesDirectionalDerivatives']         = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_providesDirectionalDerivatives))

        return capabilities
    
    def _provides_directional_derivatives(self):
        """
        Check capability to provide directional derivatives.
        """
        return FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_providesDirectionalDerivatives)
        
    def _supports_get_set_FMU_state(self):
        """
        Check support for getting and setting the FMU state.
        """
        return FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_cs_canGetAndSetFMUstate)


cdef class FMUModelME2(FMUModelBase2):
    """
    Model-exchange model loaded from a dll
    """

    def __init__(self, fmu, path = '.', enable_logging = None, log_file_name = "", log_level=FMI_DEFAULT_LOG_LEVEL, _unzipped_dir=None, _connect_dll=True):
        """
        Constructor of the model.

        Parameters::

            fmu --
                Name of the fmu as a string.

            path --
                Path to the fmu-directory.
                Default: '.' (working directory)

            enable_logging [DEPRECATED] --
                This option is DEPRECATED and will be removed. Please 
                use the option "log_level" instead.

            log_file_name --
                Filename for file used to save logmessages.
                Default: "" (Generates automatically)
                
            log_level --
                Determines the logging output. Can be set between 0 
                (no logging) and 7 (everything).
                Default: 2 (log error messages)

        Returns::

            A model as an object from the class FMUModelME2
        """
        #Call super
        FMUModelBase2.__init__(self, fmu, path, enable_logging, log_file_name, log_level, _unzipped_dir, _connect_dll)

        if self._fmu_kind != FMIL.fmi2_fmu_kind_me:
            if self._fmu_kind != FMIL.fmi2_fmu_kind_me_and_cs:
                raise FMUException('This class only supports FMI 2.0 for Model Exchange.')

        if self.get_capability_flags()['needsExecutionTool'] == True:
            raise FMUException('Models that need an execution tool are not supported')
        
        self._eventInfo.newDiscreteStatesNeeded           = FMI2_FALSE
        self._eventInfo.terminateSimulation               = FMI2_FALSE
        self._eventInfo.nominalsOfContinuousStatesChanged = FMI2_FALSE
        self._eventInfo.valuesOfContinuousStatesChanged   = FMI2_TRUE 
        self._eventInfo.nextEventTimeDefined              = FMI2_FALSE
        self._eventInfo.nextEventTime                     = 0.0
        
        self.force_finite_differences = 0
        
        self._modelId = decode(FMIL.fmi2_import_get_model_identifier_ME(self._fmu))
        
        if _connect_dll:
            self.instantiate()

    def __dealloc__(self):
        """
        Deallocate memory allocated
        """

        if self._allocated_fmu == 1:
            FMIL.fmi2_import_terminate(self._fmu)
            FMIL.fmi2_import_free_instance(self._fmu)

        if self._allocated_dll == 1:
            FMIL.fmi2_import_destroy_dllfmu(self._fmu)

        if self._allocated_xml == 1:
            FMIL.fmi2_import_free(self._fmu)

        if self._fmu_temp_dir != NULL:
            FMIL.fmi_import_rmdir(&self.callbacks, self._fmu_temp_dir)
            FMIL.free(self._fmu_temp_dir)
            self._fmu_temp_dir = NULL
            
        if self._allocated_context == 1:
            FMIL.fmi_import_free_context(self._context)
            
        if self._fmu_log_name != NULL:
            FMIL.free(self._fmu_log_name)
            self._fmu_log_name = NULL

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
        self._pyEventInfo.newDiscreteStatesNeeded           = self._eventInfo.newDiscreteStatesNeeded == 1
        self._pyEventInfo.terminateSimulation               = self._eventInfo.terminateSimulation == 1
        self._pyEventInfo.nominalsOfContinuousStatesChanged = self._eventInfo.nominalsOfContinuousStatesChanged == 1
        self._pyEventInfo.valuesOfContinuousStatesChanged   = self._eventInfo.valuesOfContinuousStatesChanged == 1
        self._pyEventInfo.nextEventTimeDefined              = self._eventInfo.nextEventTimeDefined == 1
        self._pyEventInfo.nextEventTime                     = self._eventInfo.nextEventTime

        return self._pyEventInfo
    
    def enter_event_mode(self):
        """
        Sets the FMU to be in event mode by calling the
        underlying FMU method.
        """
        cdef int status
        status = FMIL.fmi2_import_enter_event_mode(self._fmu)
        
        if status != 0:
            raise FMUException('Failed to enter event mode.')
    
    def enter_continuous_time_mode(self):
        """
        Sets the FMU to be in continuous time mode by calling the
        underlying FMU method.
        """
        cdef int status
        status = FMIL.fmi2_import_enter_continuous_time_mode(self._fmu)
        
        if status != 0:
            raise FMUException('Failed to enter continuous time mode.')

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
            raise FMUException('Failed to get the event indicators at time: %E.'%self.time)
            

        return values

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

        Calls the low-level FMI function: fmi2NewDiscreteStates
        """
        cdef int status

        if intermediateResult:
            status = FMIL.fmi2_import_new_discrete_states(self._fmu, &self._eventInfo)
            if status != 0:
                raise FMUException('Failed to update the events at time: %E.'%self.time)
        else:
            tmp_values_continuous_states_changed   = False
            tmp_nominals_continuous_states_changed = False
            
            self._eventInfo.newDiscreteStatesNeeded = FMI2_TRUE
            while self._eventInfo.newDiscreteStatesNeeded:
                status = FMIL.fmi2_import_new_discrete_states(self._fmu, &self._eventInfo)
                
                if self._eventInfo.nominalsOfContinuousStatesChanged:
                    tmp_nominals_continuous_states_changed = True
                if self._eventInfo.valuesOfContinuousStatesChanged:
                    tmp_values_continuous_states_changed = True
                if status != 0:
                    raise FMUException('Failed to update the events at time: %E.'%self.time)
            
            # If the values in the event struct have been overwritten.
            if tmp_values_continuous_states_changed:
                self._eventInfo.valuesOfContinuousStatesChanged = True
            if tmp_nominals_continuous_states_changed:
                self._eventInfo.nominalsOfContinuousStatesChanged = True

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

    def completed_integrator_step(self, no_set_FMU_state_prior_to_current_point = True):
        """
        This method must be called by the environment after every completed step
        of the integrator. If the return is True, then the environment must call
        event_update() otherwise, no action is needed.

        Returns::

            True -> Call event_update().
            False -> Do nothing.

        Calls the low-level FMI function: fmi2CompletedIntegratorStep.
        """
        cdef int status
        cdef FMIL.fmi2_boolean_t noSetFMUStatePriorToCurrentPoint = FMI2_TRUE if no_set_FMU_state_prior_to_current_point else FMI2_FALSE
        cdef FMIL.fmi2_boolean_t enterEventMode
        cdef FMIL.fmi2_boolean_t terminateSimulation        

        status = FMIL.fmi2_import_completed_integrator_step(self._fmu, noSetFMUStatePriorToCurrentPoint, &enterEventMode, &terminateSimulation)
        
        self._last_accepted_time = self.time
        
        if status != 0:
            raise FMUException('Failed to call FMI completed step at time: %E.'%self.time)
        
        return enterEventMode==FMI2_TRUE, terminateSimulation==FMI2_TRUE

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

    def _set_continuous_states(self, N.ndarray[FMIL.fmi2_real_t, ndim=1, mode="c"] values):
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
    the low-level FMI function: fmi2SetContinuousStates/fmi2GetContinuousStates.
    """)

    def _get_nominal_continuous_states(self):
        """
        Returns the nominal values of the continuous states.

        Returns::
            The nominal values.
        """
        cdef int status
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c'] ndx = N.zeros(self._nContinuousStates, dtype=N.double)

        status = FMIL.fmi2_import_get_nominals_of_continuous_states(
                self._fmu, <FMIL.fmi2_real_t*> ndx.data, self._nContinuousStates)

        if status != 0:
            raise FMUException('Failed to get the nominal values.')

        return ndx

    nominal_continuous_states = property(_get_nominal_continuous_states, doc =
    """
    Property for accessing the nominal values of the continuous states. Calls
    the low-level FMI function: fmi2GetNominalContinuousStates.
    """)

    cpdef get_derivatives(self):
        """
        Returns the derivative of the continuous states.

        Returns::

            dx --
                The derivatives as an array.

        Example::

            dx = model.get_derivatives()

        Calls the low-level FMI function: fmi2GetDerivatives
        """
        cdef int status
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c'] values = N.empty(self._nContinuousStates, dtype = N.double)

        status = FMIL.fmi2_import_get_derivatives(self._fmu, <FMIL.fmi2_real_t*> values.data, self._nContinuousStates)

        if status != 0:
            raise FMUException('Failed to get the derivative values at time: %E.'%self.time)

        return values


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

                    >> myModel = load_fmu(...)
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
        capabilities['needsExecutionTool']                  = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_needsExecutionTool))
        capabilities['completedIntegratorStepNotNeeded']    = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_completedIntegratorStepNotNeeded))
        capabilities['canBeInstantiatedOnlyOncePerProcess'] = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_canBeInstantiatedOnlyOncePerProcess))
        capabilities['canNotUseMemoryManagementFunctions']  = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_canNotUseMemoryManagementFunctions))
        capabilities['canGetAndSetFMUstate']                = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_canGetAndSetFMUstate))
        capabilities['canSerializeFMUstate']                = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_canSerializeFMUstate))
        capabilities['providesDirectionalDerivatives']      = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_providesDirectionalDerivatives))
        capabilities['completedEventIterationIsProvided']   = bool(FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_completedEventIterationIsProvided))
        
        return capabilities
        
    def _provides_directional_derivatives(self):
        """
        Check capability to provide directional derivatives.
        """
        return FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_providesDirectionalDerivatives)
        
    def _supports_get_set_FMU_state(self):
        """
        Check support for getting and setting the FMU state.
        """
        return FMIL.fmi2_import_get_capability(self._fmu, FMIL.fmi2_me_canGetAndSetFMUstate)

    def _get_directional_proxy(self, var_ref, func_ref, group, add_diag=False, output_matrix=None):
        if not self._has_entered_init_mode:
            raise FMUException("The FMU has not entered initialization mode and thus the directional " \
                               "derivatives cannot be computed. Call enter_initialization_mode to start the initialization.")
        if self._provides_directional_derivatives() and not self.force_finite_differences:
            return FMUModelBase2._get_directional_proxy(self, var_ref, func_ref, group, add_diag, output_matrix)
        else:
            return self._estimate_directional_derivative(var_ref, func_ref, group, add_diag, output_matrix)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def _estimate_directional_derivative(self, var_ref, func_ref, dict group=None, add_diag=False, output_matrix=None):
        cdef list data = [], row = [], col = []
        cdef int sol_found = 0, dim = 0, i, j, len_v = len(var_ref), len_f = len(func_ref), local_indices_vars_nbr, status
        cdef double nominal, fac, tmp
        cdef int method = FORWARD_DIFFERENCE if self.force_finite_differences is True or self.force_finite_differences == 0 else CENTRAL_DIFFERENCE
        cdef double RUROUND = FORWARD_DIFFERENCE_EPS if method == FORWARD_DIFFERENCE else CENTRAL_DIFFERENCE_EPS
        cdef N.ndarray[FMIL.fmi2_real_t, ndim=1, mode='c'] dfpert, df, eps
        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1, mode='c'] v_ref = N.array(var_ref, copy=False, dtype = N.uint32)
        cdef N.ndarray[FMIL.fmi2_value_reference_t, ndim=1, mode='c'] z_ref = N.array(func_ref, copy=False, dtype = N.uint32)
        cdef int ind_local = 5 if add_diag else 4
        cdef list local_group
        
        cdef FMIL.fmi2_real_t *column_data_pt
        cdef FMIL.fmi2_real_t *v_pt
        cdef FMIL.fmi2_real_t *df_pt
        cdef FMIL.fmi2_real_t *eps_pt
        cdef FMIL.fmi2_real_t *tmp_val_pt
        cdef FMIL.fmi2_real_t *output_matrix_data_pt = NULL
        cdef FMIL.fmi2_value_reference_t *v_ref_pt = <FMIL.fmi2_value_reference_t*>PyArray_DATA(v_ref)
        cdef FMIL.fmi2_value_reference_t *z_ref_pt = <FMIL.fmi2_value_reference_t*>PyArray_DATA(z_ref)
        cdef FMIL.fmi2_value_reference_t *local_v_vref_pt
        cdef FMIL.fmi2_value_reference_t *local_z_vref_pt
        cdef int* local_indices_vars_pt
        cdef int* local_indices_matrix_rows_pt
        cdef int* local_indices_matrix_columns_pt
        cdef int* local_data_indices
        
        #Make sure that the work vectors has the correct lengths
        self._worker_object.verify_dimensions(max(len_v, len_f))
        
        #Get work vectors
        df_pt      = self._worker_object.get_real_vector(0)
        df         = self._worker_object.get_real_numpy_vector(0) #Should be removed in the future
        v_pt       = self._worker_object.get_real_vector(1)
        eps_pt     = self._worker_object.get_real_vector(2)
        eps        = self._worker_object.get_real_numpy_vector(2) #Should be removed in the future
        tmp_val_pt = self._worker_object.get_real_vector(3)
        
        local_v_vref_pt = self._worker_object.get_value_reference_vector(0)
        local_z_vref_pt = self._worker_object.get_value_reference_vector(1)
        
        #Get updated values for the derivatives and states
        self._get_real(z_ref_pt, len_f, df_pt)
        self._get_real(v_ref_pt, len_v, v_pt)

        for i in range(len_v):
            nominal = self.get_variable_nominal(valueref = v_ref_pt[i])
            eps_pt[i] = RUROUND*(max(abs(v_pt[i]), nominal))

        if group is not None:
            if output_matrix is not None:
                if not isinstance(output_matrix, sp.csc_matrix):
                    output_matrix = None
                else:
                    output_matrix_data_pt = <FMIL.fmi2_real_t*>PyArray_DATA(output_matrix.data)
            
            if add_diag and output_matrix is None:
                dim = min(len_v,len_f)
                data.extend([0.0]*dim)
                row.extend(range(dim))
                col.extend(range(dim))
                
            for key in group["groups"]:
                local_group = group[key]
                sol_found = 0
                local_indices_vars_pt           = <int*>PyArray_DATA(local_group[0])
                local_indices_matrix_rows_pt    = <int*>PyArray_DATA(local_group[2])
                local_indices_matrix_columns_pt = <int*>PyArray_DATA(local_group[3])
                local_data_indices              = <int*>PyArray_DATA(local_group[ind_local])
                
                local_indices_vars_nbr        = len(local_group[0])
                local_indices_matrix_rows_nbr = len(local_group[2])
                
                #Structure of a local group
                # - [0] - variable indexes
                # - [1] - variable names
                # - [2] - matrix rows
                # - [3] - matrix columns
                # - [4] - position in data vector (CSC format)
                # - [5] - position in data vector (with diag) (CSC format)
                
                #Get the local value references for the derivatives and states corresponding to the current group
                for i in range(local_indices_vars_nbr):        local_v_vref_pt[i] = v_ref_pt[local_indices_vars_pt[i]]
                for i in range(local_indices_matrix_rows_nbr): local_z_vref_pt[i] = z_ref_pt[local_indices_matrix_rows_pt[i]]
                
                for fac in [1.0, 0.1, 0.01, 0.001]: #In very special cases, the epsilon is too big, if an error, try to reduce eps
                    for i in range(local_indices_vars_nbr): tmp_val_pt[i] = v_pt[local_indices_vars_pt[i]]+fac*eps_pt[local_indices_vars_pt[i]]
                    self._set_real(local_v_vref_pt, tmp_val_pt, local_indices_vars_nbr)
                    
                    if method == FORWARD_DIFFERENCE: #Forward and Backward difference
                        column_data_pt = tmp_val_pt

                        status = self._get_real(local_z_vref_pt, local_indices_matrix_rows_nbr, tmp_val_pt)
                        if status == 0:
                            for i in range(local_indices_matrix_rows_nbr):
                                column_data_pt[i] = (tmp_val_pt[i] - df_pt[local_indices_matrix_rows_pt[i]])/(fac*eps_pt[local_indices_matrix_columns_pt[i]])
                        
                            sol_found = 1
                        else: #Backward

                            for i in range(local_indices_vars_nbr): tmp_val_pt[i] = v_pt[local_indices_vars_pt[i]]-fac*eps_pt[local_indices_vars_pt[i]]
                            self._set_real(local_v_vref_pt, tmp_val_pt, local_indices_vars_nbr)
                            
                            status = self._get_real(local_z_vref_pt, local_indices_matrix_rows_nbr, tmp_val_pt)
                            if status == 0:
                                for i in range(local_indices_matrix_rows_nbr):
                                    column_data_pt[i] = (df_pt[local_indices_matrix_rows_pt[i]] - tmp_val_pt[i])/(fac*eps_pt[local_indices_matrix_columns_pt[i]])
                                    
                                sol_found = 1
                                
                    else: #Central difference
                        dfpertp = self.get_real(z_ref[local_group[2]])
                        
                        for i in range(local_indices_vars_nbr): tmp_val_pt[i] = v_pt[local_indices_vars_pt[i]]-fac*eps_pt[local_indices_vars_pt[i]]
                        self._set_real(local_v_vref_pt, tmp_val_pt, local_indices_vars_nbr)
                        
                        dfpertm = self.get_real(z_ref[local_group[2]])
                        
                        column_data = (dfpertp - dfpertm)/(2*fac*eps[local_group[3]])
                        column_data_pt = <FMIL.fmi2_real_t*>PyArray_DATA(column_data)
                        sol_found = 1
                    
                    if sol_found:
                        if output_matrix is not None:
                            for i in range(local_indices_matrix_rows_nbr):
                                output_matrix_data_pt[local_data_indices[i]] = column_data_pt[i]
                        else:
                            for i in range(local_indices_matrix_rows_nbr):
                                data.append(column_data_pt[i])
                        break
                else:
                    raise FMUException("Failed to estimate the directional derivative at time %g."%self.time)
                
                if output_matrix is None:
                    row.extend(local_group[2])
                    col.extend(local_group[3])
                
                for i in range(local_indices_vars_nbr): tmp_val_pt[i] = v_pt[local_indices_vars_pt[i]]
                self._set_real(local_v_vref_pt, tmp_val_pt, local_indices_vars_nbr)
            
            if output_matrix is not None:
                A = output_matrix 
            else:
                if len(data) == 0:
                    A = sp.csc_matrix((len_f,len_v))
                else:
                    A = sp.csc_matrix((data, (row, col)), (len_f,len_v))
                    
            return A
        else:
            if output_matrix is None or \
                (not isinstance(output_matrix, N.ndarray)) or \
                (isinstance(output_matrix, N.ndarray) and (output_matrix.shape[0] != len_f or output_matrix.shape[1] != len_v)):
                    A = N.zeros((len_f,len_v))
            else:
                A = output_matrix
            
            if len_v == 0 or len_f == 0:
                return A
            
            dfpert = N.zeros(len_f, dtype = N.double)
            for i in range(len_v):
                tmp = v_pt[i]
                for fac in [1.0, 0.1, 0.01, 0.001]: #In very special cases, the epsilon is too big, if an error, try to reduce eps
                    v_pt[i] = tmp+fac*eps_pt[i]
                    self._set_real(v_ref_pt, v_pt, len_v)
                            
                    if method == FORWARD_DIFFERENCE: #Forward and Backward difference
                        try:
                            dfpert = self.get_real(z_ref)
                            A[:, i] = (dfpert - df)/(fac*eps_pt[i])
                            break
                        except FMUException: #Try backward difference
                            v_pt[i] = tmp - fac*eps_pt[i]
                            self._set_real(v_ref_pt, v_pt, len_v)
                            try:
                                dfpert = self.get_real(z_ref)
                                A[:, i] = (df - dfpert)/(fac*eps_pt[i])
                                break
                            except FMUException:
                                pass
                            
                    else: #Central difference
                        dfpertp = self.get_real(z_ref)
                        v_pt[i] = tmp - fac*eps_pt[i]
                        self._set_real(v_ref_pt, v_pt, len_v)
                        dfpertm = self.get_real(z_ref)
                        A[:, i] = (dfpertp - dfpertm)/(2*fac*eps_pt[i])
                        break
                else:
                    raise FMUException("Failed to estimate the directional derivative at time %g."%self.time)
                
                #Reset values
                v_pt[i] = tmp
                self._set_real(v_ref_pt, v_pt, len_v)
            
            return A
            
cdef class __ForTestingFMUModelME2(FMUModelME2):
    cdef int _get_real(self, FMIL.fmi2_value_reference_t* vrefs, size_t size, FMIL.fmi2_real_t* values):
        vr = N.zeros(size)
        for i in range(size):
            vr[i] = vrefs[i]
        
        try:
            vv = self.get_real(vr)
        except:
            return FMIL.fmi2_status_error
        
        for i in range(size):
            values[i] = vv[i]
        
        return FMIL.fmi2_status_ok
    
    cdef int _set_real(self, FMIL.fmi2_value_reference_t* vrefs, FMIL.fmi2_real_t* values, size_t size):
        vr = N.zeros(size)
        vv = N.zeros(size)
        for i in range(size):
            vr[i] = vrefs[i]
            vv[i] = values[i]
        
        try:
            self.set_real(vr, vv)
        except:
            return FMIL.fmi2_status_error
        
        return FMIL.fmi2_status_ok

#Temporary should be removed! (after a period)
cdef class FMUModel(FMUModelME1):
    def __init__(self, fmu, path='.', enable_logging=True):
        print("WARNING: This class is deprecated and has been superseded with FMUModelME1. The recommended entry-point for loading an FMU is now the function load_fmu.")
        FMUModelME1.__init__(self,fmu,path,enable_logging)
        
def _handle_load_fmu_exception(fmu, log_data):
    for log in log_data:
        print(log)

def load_fmu(fmu, path = '.', enable_logging = None, log_file_name = "", kind = 'auto', log_level=FMI_DEFAULT_LOG_LEVEL):
    """
    Helper method for creating a model instance.

    Parameters::

        fmu --
            Name of the fmu as a string.

        path --
            Path to the fmu-directory.
            Default: '.' (working directory)

        enable_logging [DEPRECATED] --
            This option is DEPRECATED and will be removed. Please use
            the option "log_level" instead.

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
            Default: 'auto' (Chooses ME before CS if both available)
        
        log_level --
            Determines the logging output. Can be set between 0 
            (no logging) and 7 (everything).
            Default: 2 (log error messages)
        
    Returns::

        A model instance corresponding to the loaded FMU.
    """

    #NOTE: This method can be made more efficient by providing
    #the unzipped part and the already read XML object to the different
    #FMU classes.

    #FMIL related variables
    cdef FMIL.fmi_import_context_t*     context
    cdef FMIL.jm_callbacks              callbacks
    cdef FMIL.jm_string                 last_error
    cdef FMIL.fmi_version_enu_t         version
    cdef FMIL.fmi1_import_t*            fmu_1 = NULL
    cdef FMIL.fmi2_import_t*            fmu_2 = NULL
    cdef FMIL.fmi1_fmu_kind_enu_t       fmu_1_kind
    cdef FMIL.fmi2_fmu_kind_enu_t       fmu_2_kind
    cdef list                           log_data = []

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
    callbacks.context   = <void*>log_data
    
    original_enable_logging = enable_logging
    
    if enable_logging==None:
        if log_level >= FMIL.jm_log_level_nothing and log_level <= FMIL.jm_log_level_all:
            if log_level == FMIL.jm_log_level_nothing:
                enable_logging = False
            else:
                enable_logging = True
            callbacks.log_level = log_level
        else:
            raise FMUException("The log level must be between %d and %d"%(FMIL.jm_log_level_nothing, FMIL.jm_log_level_all))
    else:
        logging.warning("The attribute 'enable_logging' is deprecated. Please use 'log_level' instead. Setting 'log_level' to INFO...")
        callbacks.log_level = FMIL.jm_log_level_info if enable_logging else FMIL.jm_log_level_error

    # Create a struct for allocation
    context = FMIL.fmi_import_allocate_context(&callbacks)

    #Get the FMI version of the provided model
    fmu_temp_dir = encode(create_temp_dir())
    fmu_full_path = encode(fmu_full_path)
    version = FMIL.fmi_import_get_fmi_version(context, fmu_full_path, fmu_temp_dir)

    #Check the version
    if version == FMIL.fmi_version_unknown_enu:
        #Delete context
        last_error = FMIL.jm_get_last_error(&callbacks)
        FMIL.fmi_import_free_context(context)
        FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
        if callbacks.log_level >= FMIL.jm_log_level_error:
            _handle_load_fmu_exception(fmu, log_data)
            raise FMUException("The FMU version could not be determined. "+decode(last_error))
        else:
            _handle_load_fmu_exception(fmu, log_data)
            raise FMUException("The FMU version could not be determined. Enable logging for possibly more information.")

    if version > 2:
        #Delete the context
        last_error = FMIL.jm_get_last_error(&callbacks)
        FMIL.fmi_import_free_context(context)
        FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
        if callbacks.log_level >= FMIL.jm_log_level_error:
            _handle_load_fmu_exception(fmu, log_data)
            raise FMUException("The FMU version is unsupported. "+decode(last_error))
        else:
            _handle_load_fmu_exception(fmu, log_data)
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
            if callbacks.log_level >= FMIL.jm_log_level_error:
                _handle_load_fmu_exception(fmu, log_data)
                raise FMUException("The XML-could not be read. "+decode(last_error))
            else:
                _handle_load_fmu_exception(fmu, log_data)
                raise FMUException('The XML-could not be read. Enable logging for possible nore information.')

        fmu_1_kind = FMIL.fmi1_import_get_fmu_kind(fmu_1)

        #Compare fmu_kind with input-specified kind
        if fmu_1_kind == FMI_ME and kind.upper() != 'CS':
            model=FMUModelME1(fmu, path, original_enable_logging, log_file_name,log_level, _unzipped_dir=fmu_temp_dir)
        elif (fmu_1_kind == FMI_CS_STANDALONE or fmu_1_kind == FMI_CS_TOOL) and kind.upper() != 'ME':
            model=FMUModelCS1(fmu, path, original_enable_logging, log_file_name,log_level, _unzipped_dir=fmu_temp_dir)
        else:
            FMIL.fmi1_import_free(fmu_1)
            FMIL.fmi_import_free_context(context)
            FMIL.fmi_import_rmdir(&callbacks,fmu_temp_dir)
            _handle_load_fmu_exception(fmu, log_data)
            raise FMUException('FMU is a ' + FMIL.fmi1_fmu_kind_to_string(fmu_1_kind) + ' and not a ' + kind.upper())

    elif version == FMIL.fmi_version_2_0_enu:
        #Check fmu-kind and compare with input-specified kind
        fmu_2 = FMIL.fmi2_import_parse_xml(context, fmu_temp_dir, NULL)

        if fmu_2 is NULL:
            #Delete the context
            last_error = FMIL.jm_get_last_error(&callbacks)
            FMIL.fmi_import_free_context(context)
            FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
            if callbacks.log_level >= FMIL.jm_log_level_error:
                _handle_load_fmu_exception(fmu, log_data)
                raise FMUException("The XML-could not be read. "+decode(last_error))
            else:
                _handle_load_fmu_exception(fmu, log_data)
                raise FMUException('The XML-could not be read. Enable logging for possible nore information.')

        fmu_2_kind = FMIL.fmi2_import_get_fmu_kind(fmu_2)

        #FMU kind is unknown
        if fmu_2_kind == FMIL.fmi2_fmu_kind_unknown:
            last_error = FMIL.jm_get_last_error(&callbacks)
            FMIL.fmi2_import_free(fmu_2)
            FMIL.fmi_import_free_context(context)
            FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
            if callbacks.log_level >= FMIL.jm_log_level_error:
                _handle_load_fmu_exception(fmu, log_data)
                raise FMUException("The FMU kind could not be determined. "+decode(last_error))
            else:
                _handle_load_fmu_exception(fmu, log_data)
                raise FMUException("The FMU kind could not be determined. Enable logging for possibly more information.")

        #FMU kind is known
        if kind.lower() == 'auto':
            if fmu_2_kind == FMIL.fmi2_fmu_kind_cs:
                model = FMUModelCS2(fmu, path, original_enable_logging, log_file_name,log_level, _unzipped_dir=fmu_temp_dir)
            elif fmu_2_kind == FMIL.fmi2_fmu_kind_me or fmu_2_kind == FMIL.fmi2_fmu_kind_me_and_cs:
                model = FMUModelME2(fmu, path, original_enable_logging, log_file_name,log_level, _unzipped_dir=fmu_temp_dir)
        elif kind.upper() == 'CS':
            if fmu_2_kind == FMIL.fmi2_fmu_kind_cs or fmu_2_kind == FMIL.fmi2_fmu_kind_me_and_cs:
                model = FMUModelCS2(fmu, path, original_enable_logging, log_file_name,log_level, _unzipped_dir=fmu_temp_dir)
        elif kind.upper() == 'ME':
            if fmu_2_kind == FMIL.fmi2_fmu_kind_me or fmu_2_kind == FMIL.fmi2_fmu_kind_me_and_cs:
                model = FMUModelME2(fmu, path, original_enable_logging, log_file_name,log_level, _unzipped_dir=fmu_temp_dir)

        #Could not match FMU kind with input-specified kind
        if model is None:
            FMIL.fmi2_import_free(fmu_2)
            FMIL.fmi_import_free_context(context)
            FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
            _handle_load_fmu_exception(fmu, log_data)
            raise FMUException('FMU is a ' + FMIL.fmi2_fmu_kind_to_string(fmu_2_kind) + ' and not a ' + kind.upper())

    else:
        #This else-statement ensures that the variables "context" and "version" are defined before proceeding

        #Delete the context
        last_error = FMIL.jm_get_last_error(&callbacks)
        FMIL.fmi_import_free_context(context)
        FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
        if callbacks.log_level >= FMIL.jm_log_level_error:
            _handle_load_fmu_exception(fmu, log_data)
            raise FMUException("The FMU version is not found. "+decode(last_error))
        else:
            _handle_load_fmu_exception(fmu, log_data)
            raise FMUException("The FMU version is not found. Enable logging for possibly more information.")

    #Delete
    if version == FMIL.fmi_version_1_enu:
        FMIL.fmi1_import_free(fmu_1)
        FMIL.fmi_import_free_context(context)
        #FMIL.fmi_import_rmdir(&callbacks,fmu_temp_dir)

    if version == FMIL.fmi_version_2_0_enu:
        FMIL.fmi2_import_free(fmu_2)
        FMIL.fmi_import_free_context(context)
        #FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)

    return model


cdef class WorkerClass2:
    
    def __init__(self):
        self._dim = 0
    
    def _update_work_vectors(self, dim):
        self._tmp1_val = N.zeros(dim, dtype = N.double)
        self._tmp2_val = N.zeros(dim, dtype = N.double)
        self._tmp3_val = N.zeros(dim, dtype = N.double)
        self._tmp4_val = N.zeros(dim, dtype = N.double)
        
        self._tmp1_ref = N.zeros(dim, dtype = N.uint32)
        self._tmp2_ref = N.zeros(dim, dtype = N.uint32)
        self._tmp3_ref = N.zeros(dim, dtype = N.uint32)
        self._tmp4_ref = N.zeros(dim, dtype = N.uint32)
        
    cpdef verify_dimensions(self, int dim):
        if dim > self._dim:
            self._update_work_vectors(dim)
    
    cdef N.ndarray get_real_numpy_vector(self, int index):
        cdef N.ndarray ret = None
        
        if index == 0:
            ret = self._tmp1_val
        elif index == 1:
            ret = self._tmp2_val
        elif index == 2:
            ret = self._tmp3_val
        elif index == 3:
            ret = self._tmp4_val
            
        return ret
    
    cdef FMIL.fmi2_real_t* get_real_vector(self, int index):
        cdef FMIL.fmi2_real_t* ret = NULL
        if index == 0:
            ret = <FMIL.fmi2_real_t*>PyArray_DATA(self._tmp1_val)
        elif index == 1:
            ret = <FMIL.fmi2_real_t*>PyArray_DATA(self._tmp2_val)
        elif index == 2:
            ret = <FMIL.fmi2_real_t*>PyArray_DATA(self._tmp3_val)
        elif index == 3:
            ret = <FMIL.fmi2_real_t*>PyArray_DATA(self._tmp4_val)
        
        return ret
    
    cdef N.ndarray get_value_reference_numpy_vector(self, int index):
        cdef N.ndarray ret = None
        
        if index == 0:
            ret = self._tmp1_ref
        elif index == 1:
            ret = self._tmp2_ref
        elif index == 2:
            ret = self._tmp3_ref
        elif index == 3:
            ret = self._tmp4_ref
            
        return ret
    
    cdef FMIL.fmi2_value_reference_t* get_value_reference_vector(self, int index):
        cdef FMIL.fmi2_value_reference_t* ret = NULL
        if index == 0:
            ret = <FMIL.fmi2_value_reference_t*>PyArray_DATA(self._tmp1_ref)
        elif index == 1:
            ret = <FMIL.fmi2_value_reference_t*>PyArray_DATA(self._tmp2_ref)
        elif index == 2:
            ret = <FMIL.fmi2_value_reference_t*>PyArray_DATA(self._tmp3_ref)
        elif index == 3:
            ret = <FMIL.fmi2_value_reference_t*>PyArray_DATA(self._tmp4_ref)
        
        return ret

