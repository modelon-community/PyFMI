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

import os
import logging
cimport cython

import numpy as np
cimport numpy as np
np.import_array()

int   = np.int32
np.int = np.int32

cimport pyfmi.fmil_import as FMIL
cimport pyfmi.fmil1_import as FMIL1
cimport pyfmi.fmi_base as FMI_BASE
cimport pyfmi.util as pyfmi_util
from pyfmi.util import enable_caching

from pyfmi.exceptions import (
    FMUException,
    InvalidVersionException, 
    InvalidXMLException,
    InvalidBinaryException
)

from pyfmi.common.core import create_temp_dir
from pyfmi.fmi_base import (
    PyEventInfo, 
    FMI_DEFAULT_LOG_LEVEL,
    check_fmu_args,
    _handle_load_fmu_exception
)

GLOBAL_LOG_LEVEL = 3
GLOBAL_FMU_OBJECT = None # FMI1 CS
FMI_REGISTER_GLOBALLY = 1

# Basic flags related to FMI
FMI_TRUE  = '\x01'
FMI_FALSE = '\x00'

# Status
FMI_OK      = FMIL1.fmi1_status_ok
FMI_WARNING = FMIL1.fmi1_status_warning
FMI_DISCARD = FMIL1.fmi1_status_discard
FMI_ERROR   = FMIL1.fmi1_status_error
FMI_FATAL   = FMIL1.fmi1_status_fatal
FMI_PENDING = FMIL1.fmi1_status_pending

FMI1_DO_STEP_STATUS       = FMIL1.fmi1_do_step_status
FMI1_PENDING_STATUS       = FMIL1.fmi1_pending_status
FMI1_LAST_SUCCESSFUL_TIME = FMIL1.fmi1_last_successful_time

# Types
FMI_REAL        = FMIL1.fmi1_base_type_real
FMI_INTEGER     = FMIL1.fmi1_base_type_int
FMI_BOOLEAN     = FMIL1.fmi1_base_type_bool
FMI_STRING      = FMIL1.fmi1_base_type_str
FMI_ENUMERATION = FMIL1.fmi1_base_type_enum

# Alias data
FMI_NO_ALIAS      = FMIL1.fmi1_variable_is_not_alias
FMI_ALIAS         = FMIL1.fmi1_variable_is_alias
FMI_NEGATED_ALIAS = FMIL1.fmi1_variable_is_negated_alias

# Variability
FMI_CONTINUOUS = FMIL1.fmi1_variability_enu_continuous
FMI_CONSTANT   = FMIL1.fmi1_variability_enu_constant
FMI_PARAMETER  = FMIL1.fmi1_variability_enu_parameter
FMI_DISCRETE   = FMIL1.fmi1_variability_enu_discrete

# Causality
FMI_INPUT    = FMIL1.fmi1_causality_enu_input
FMI_OUTPUT   = FMIL1.fmi1_causality_enu_output
FMI_INTERNAL = FMIL1.fmi1_causality_enu_internal
FMI_NONE     = FMIL1.fmi1_causality_enu_none


FMI_ME                 = FMIL1.fmi1_fmu_kind_enu_me
FMI_CS_STANDALONE      = FMIL1.fmi1_fmu_kind_enu_cs_standalone
FMI_CS_TOOL            = FMIL1.fmi1_fmu_kind_enu_cs_tool
cdef FMI_MIME_CS_STANDALONE = pyfmi_util.encode("application/x-fmu-sharedlibrary")

#CALLBACKS
cdef void importlogger_default(FMIL.jm_callbacks* c, FMIL.jm_string module, FMIL.jm_log_level_enu_t log_level, FMIL.jm_string message):
    msg = pyfmi_util.decode(message)
    mod = pyfmi_util.decode(module)
    if GLOBAL_FMU_OBJECT is not None:
        GLOBAL_FMU_OBJECT.append_log_message(mod,log_level,msg)
    elif log_level <= GLOBAL_LOG_LEVEL:
        print("FMIL: module = %s, log level = %d: %s\n"%(mod, log_level, msg))

#CALLBACKS
cdef void importlogger(FMIL.jm_callbacks* c, FMIL.jm_string module, FMIL.jm_log_level_enu_t log_level, FMIL.jm_string message):
    if c.context != NULL:
        (<FMUModelBase>c.context)._logger(module,log_level,message)

cdef class ScalarVariable:
    """
    Class defining data structure based on the XML element ScalarVariable.
    """
    def __init__(self, name, value_reference, type, description="",
                       variability=FMIL1.fmi1_variability_enu_continuous,
                       causality=FMIL1.fmi1_causality_enu_internal,
                       alias=FMIL1.fmi1_variable_is_not_alias):
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

cdef class FMUModelBase(FMI_BASE.ModelBase):
    """
    An FMI Model loaded from a DLL.
    """
    def __init__(self, fmu, log_file_name="", log_level=FMI_DEFAULT_LOG_LEVEL,
                 _unzipped_dir=None, _connect_dll=True, allow_unzipped_fmu = False):
        """
        Constructor of the model.

        Parameters::

            fmu --
                Name of the fmu as a string.

            log_file_name --
                Filename for file used to save logmessages.
                This argument can also be a stream if it supports 'write', for full functionality
                it must also support 'seek' and 'readlines'. If the stream requires use of other methods, such as 'drain'
                for asyncio-streams, then this needs to be implemented on the user-side, there is no additional methods invoked
                on the stream instance after 'write' has been invoked on the PyFMI side.
                The stream must also be open and writable during the entire time.
                Default: "" (Generates automatically)

            log_level --
                Determines the logging output. Can be set between 0
                (no logging) and 7 (everything).
                Default: 2 (log error messages)

            allow_unzipped_fmu --
                If set to True, the argument 'fmu' can be a path specifying a directory
                to an unzipped FMU. The structure of the unzipped FMU must conform
                to the FMI specification.
                Default: False

        Returns::

            A model as an object from the class FMUModelBase
        """
        cdef int status
        cdef int version

        #Call super
        FMI_BASE.ModelBase.__init__(self)

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

        # Used to adjust behaviour if FMU is unzipped
        self._allow_unzipped_fmu = 1 if allow_unzipped_fmu else 0

        #Specify the general callback functions
        self.callbacks.malloc  = FMIL.malloc
        self.callbacks.calloc  = FMIL.calloc
        self.callbacks.realloc = FMIL.realloc
        self.callbacks.free    = FMIL.free
        self.callbacks.logger  = importlogger
        self.callbacks.context = <void*>self #Class loggger

        if log_level >= FMIL.jm_log_level_nothing and log_level <= FMIL.jm_log_level_all:
            if log_level == FMIL.jm_log_level_nothing:
                enable_logging = False
            else:
                enable_logging = True
            self.callbacks.log_level = log_level
        else:
            raise FMUException("The log level must be between %d and %d"%(FMIL.jm_log_level_nothing, FMIL.jm_log_level_all))

        self._fmu_full_path = os.path.abspath(fmu)
        if _unzipped_dir:
            fmu_temp_dir = pyfmi_util.encode(_unzipped_dir)
        elif self._allow_unzipped_fmu:
            fmu_temp_dir = pyfmi_util.encode(fmu)
        else:
            fmu_temp_dir  = pyfmi_util.encode(create_temp_dir())
        fmu_temp_dir = os.path.abspath(fmu_temp_dir)
        self._fmu_temp_dir = <char*>FMIL.malloc((FMIL.strlen(fmu_temp_dir)+1)*sizeof(char))
        FMIL.strcpy(self._fmu_temp_dir, fmu_temp_dir)

        check_fmu_args(self._allow_unzipped_fmu, fmu, self._fmu_full_path)

        #Specify FMI related callbacks
        self.callBackFunctions.logger = FMIL1.fmi1_log_forwarding;
        self.callBackFunctions.allocateMemory = FMIL.calloc;
        self.callBackFunctions.freeMemory = FMIL.free;
        self.callBackFunctions.stepFinished = NULL;

        self.context = FMIL.fmi_import_allocate_context(&self.callbacks)
        self._allocated_context = 1

        #Get the FMI version of the provided model
        self._fmu_full_path = pyfmi_util.encode(self._fmu_full_path)

        if _unzipped_dir:
            #If the unzipped directory is provided we assume that the version
            #is correct. This is due to that the method to get the version
            #unzipps the FMU which we already have done.
            self._version = FMIL.fmi_version_1_enu
        else:
            self._version = FMI_BASE.import_and_get_version(self.context, self._fmu_full_path,
                                                   fmu_temp_dir, self._allow_unzipped_fmu)

        if self._version == FMIL.fmi_version_unknown_enu:
            last_error = pyfmi_util.decode(FMIL.jm_get_last_error(&self.callbacks))
            if self.callbacks.log_level >= FMIL.jm_log_level_error:
                raise InvalidVersionException("The FMU could not be loaded. The FMU version could not be determined. "+last_error)
            else:
                raise InvalidVersionException("The FMU could not be loaded. The FMU version could not be determined. Enable logging for possibly more information.")
        if self._version != 1:
            raise InvalidVersionException("The FMU could not be loaded. This class only supports FMI 1.0 (Model Exchange and Co-Simulation).")

        #Parse the XML
        self._fmu = FMIL1.fmi1_import_parse_xml(self.context, fmu_temp_dir)
        if self._fmu == NULL:
            last_error = pyfmi_util.decode(FMIL.jm_get_last_error(&self.callbacks))
            if self.callbacks.log_level >= FMIL.jm_log_level_error:
                raise InvalidXMLException("The FMU could not be loaded. The model data from 'modelDescription.xml' within the FMU could not be read. "+last_error)
            else:
                raise InvalidXMLException("The FMU could not be loaded. The model data from 'modelDescription.xml' within the FMU could not be read. Enable logging for possibly more information.")
        self._allocated_xml = 1

        #Check the FMU kind
        fmu_kind = FMIL1.fmi1_import_get_fmu_kind(self._fmu)
        if fmu_kind != FMI_ME and fmu_kind != FMI_CS_STANDALONE and fmu_kind != FMI_CS_TOOL:
            raise InvalidVersionException("The FMU could not be loaded. This class only supports FMI 1.0 (Model Exchange and Co-Simulation).")
        self._fmu_kind = fmu_kind

        #Connect the DLL
        if _connect_dll:
            global FMI_REGISTER_GLOBALLY
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            status = FMIL1.fmi1_import_create_dllfmu(self._fmu, self.callBackFunctions, FMI_REGISTER_GLOBALLY);
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
            if status == FMIL.jm_status_error:
                last_error = pyfmi_util.decode(FMIL1.fmi1_import_get_last_error(self._fmu))
                if self.callbacks.log_level >= FMIL.jm_log_level_error:
                    raise InvalidBinaryException("The FMU could not be loaded. "+last_error)
                else:
                    raise InvalidBinaryException("The FMU could not be loaded. Error loading the binary. Enable logging for possibly more information.")
            self._allocated_dll = 1
            FMI_REGISTER_GLOBALLY += 1 #Update the global register of FMUs

        #Default values
        self._t = None

        #Internal values
        self._file_open = False
        self._npoints = 0
        self._enable_logging = enable_logging
        self._pyEventInfo = PyEventInfo()

        #Load information from model
        self._modelId = pyfmi_util.decode(FMIL1.fmi1_import_get_model_identifier(self._fmu))
        self._modelname = pyfmi_util.decode(FMIL1.fmi1_import_get_model_name(self._fmu))
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        self._nEventIndicators = FMIL1.fmi1_import_get_number_of_event_indicators(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        self._nContinuousStates = FMIL1.fmi1_import_get_number_of_continuous_states(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if not isinstance(log_file_name, str):
            self._set_log_stream(log_file_name)
            for i in range(len(self._log)):
                try:
                    self._log_stream.write("FMIL: module = %s, log level = %d: %s\n"%(self._log[i][0], self._log[i][1], self._log[i][2]))
                except Exception:
                    if hasattr(self._log_stream, 'closed') and self._log_stream.closed:
                        logging.warning("Unable to log to closed stream.")
                    else:
                        logging.warning("Unable to log to stream.")
        else:
            fmu_log_name = pyfmi_util.encode((self._modelId + "_log.txt") if log_file_name=="" else log_file_name)
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
        cdef FMIL1.fmi1_string_t version
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        version = <FMIL1.fmi1_string_t>FMIL1.fmi1_import_get_version(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        return pyfmi_util.decode(version)

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
        cdef np.ndarray[FMIL1.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = np.asarray(valueref, dtype=np.uint32).ravel()
        cdef FMIL.size_t nref = np.size(val_ref)
        cdef np.ndarray[FMIL1.fmi1_real_t, ndim=1,mode='c'] val = np.array([0.0]*nref, dtype=float, ndmin=1)

        if nref == 0: ## get_real([])
            return val

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_get_real(self._fmu, <FMIL1.fmi1_value_reference_t*>val_ref.data, nref, <FMIL1.fmi1_real_t*>val.data)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

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
        cdef np.ndarray[FMIL1.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = np.asarray(valueref, dtype=np.uint32).ravel()
        cdef np.ndarray[FMIL1.fmi1_real_t, ndim=1,mode='c'] val = np.asarray(values, dtype=float).ravel()
        cdef FMIL.size_t nref = np.size(val_ref)

        if nref != np.size(val):
            raise FMUException(
                'The length of valueref and values are inconsistent.')

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_set_real(self._fmu, <FMIL1.fmi1_value_reference_t*>val_ref.data, nref, <FMIL1.fmi1_real_t*>val.data)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

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
        cdef np.ndarray[FMIL1.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = np.array(valueref, dtype=np.uint32,ndmin=1).ravel()
        cdef FMIL.size_t nref = np.size(val_ref)
        cdef np.ndarray[FMIL1.fmi1_integer_t, ndim=1,mode='c'] val = np.array([0]*nref, dtype=int,ndmin=1)

        if nref == 0: ## get_integer([])
            return val

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_get_integer(self._fmu, <FMIL1.fmi1_value_reference_t*>val_ref.data, nref, <FMIL1.fmi1_integer_t*>val.data)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

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
        cdef np.ndarray[FMIL1.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = np.array(valueref, dtype=np.uint32,ndmin=1).ravel()
        cdef np.ndarray[FMIL1.fmi1_integer_t, ndim=1,mode='c'] val = np.array(values, dtype=int,ndmin=1).ravel()
        cdef FMIL.size_t nref = np.size(val_ref)

        if nref != np.size(val):
            raise FMUException(
                'The length of valueref and values are inconsistent.')

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_set_integer(self._fmu, <FMIL1.fmi1_value_reference_t*>val_ref.data, nref, <FMIL1.fmi1_integer_t*>val.data)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

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
        cdef np.ndarray[FMIL1.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = np.array(valueref, dtype=np.uint32,ndmin=1).ravel()
        cdef FMIL.size_t nref = np.size(val_ref)

        if nref == 0: ## get_boolean([])
            return np.array([])

        cdef void *val = FMIL.malloc(sizeof(FMIL1.fmi1_boolean_t)*nref)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_get_boolean(self._fmu, <FMIL1.fmi1_value_reference_t*>val_ref.data, nref, <FMIL1.fmi1_boolean_t*>val)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        return_values = []
        for i in range(nref):
            return_values.append((<FMIL1.fmi1_boolean_t*>val)[i]==1)

        #Dealloc
        FMIL.free(val)

        if status != 0:
            raise FMUException('Failed to get the Boolean values.')

        return np.array(return_values)

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
        cdef np.ndarray[FMIL1.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = np.array(valueref, dtype=np.uint32,ndmin=1).ravel()
        cdef FMIL.size_t nref = np.size(val_ref)

        cdef void *val = FMIL.malloc(sizeof(FMIL1.fmi1_boolean_t)*nref)

        values = np.array(values,ndmin=1).ravel()
        for i in range(nref):
            if values[i]:
                (<FMIL1.fmi1_boolean_t*>val)[i] = 1
            else:
                (<FMIL1.fmi1_boolean_t*>val)[i] = 0

        if nref != np.size(values):
            raise FMUException(
                'The length of valueref and values are inconsistent.')

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_set_boolean(self._fmu, <FMIL1.fmi1_value_reference_t*>val_ref.data, nref, <FMIL1.fmi1_boolean_t*>val)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

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
        cdef int status
        cdef np.ndarray[FMIL1.fmi1_value_reference_t, ndim=1, mode='c'] input_valueref = np.array(valueref, dtype=np.uint32, ndmin=1).ravel()
        cdef FMIL.size_t nref = np.size(input_valueref)

        if nref == 0: ## get_string([])
            return []

        cdef FMIL1.fmi1_string_t* output_value = <FMIL1.fmi1_string_t*>FMIL.malloc(sizeof(FMIL1.fmi1_string_t)*nref)
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_get_string(self._fmu, <FMIL1.fmi1_value_reference_t*> input_valueref.data, nref, output_value)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to get the String values.')

        out = []
        for i in range(nref):
            out.append(pyfmi_util.decode(output_value[i]))

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
        cdef np.ndarray[FMIL1.fmi1_value_reference_t, ndim=1,mode='c'] val_ref = np.array(valueref, dtype=np.uint32,ndmin=1).ravel()
        cdef FMIL1.fmi1_string_t* val = <FMIL1.fmi1_string_t*>FMIL.malloc(sizeof(FMIL1.fmi1_string_t)*np.size(val_ref))

        if not isinstance(values, list):
            raise FMUException(
                'The values needs to be a list of values.')
        if len(values) != np.size(val_ref):
            raise FMUException(
                'The length of valueref and values are inconsistent.')

        values = [pyfmi_util.encode(item) for item in values]
        for i in range(np.size(val_ref)):
            val[i] = values[i]

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_set_string(self._fmu, <FMIL1.fmi1_value_reference_t*>val_ref.data, np.size(val_ref), val)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

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
        cdef FMIL1.fmi1_boolean_t log
        cdef int status

        #self.callbacks.log_level = FMIL.jm_log_level_warning if flag else FMIL.jm_log_level_nothing

        if flag:
            log = 1
        else:
            log = 0

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_set_debug_logging(self._fmu, log)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        self._enable_logging = bool(log)

        if status != 0:
            raise FMUException('Failed to set the debugging option.')

    def _set(self, variable_name, value):
        """
        Helper method to set, see docstring on set.
        """
        cdef FMIL1.fmi1_value_reference_t ref
        cdef FMIL1.fmi1_base_type_enu_t basetype
        cdef FMIL1.fmi1_import_variable_t* variable
        cdef FMIL1.fmi1_variable_alias_kind_enu_t alias_kind

        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL1.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variable_name)

        ref =  FMIL1.fmi1_import_get_variable_vr(variable)
        basetype = FMIL1.fmi1_import_get_variable_base_type(variable)
        alias_kind = FMIL1.fmi1_import_get_variable_alias_kind(variable)

        if basetype == FMIL1.fmi1_base_type_real:  #REAL
            if alias_kind == FMI_NEGATED_ALIAS:
                value = -value
            self.set_real([ref], [value])
        elif basetype == FMIL1.fmi1_base_type_int or basetype == FMIL1.fmi1_base_type_enum: #INTEGER
            if alias_kind == FMI_NEGATED_ALIAS:
                value = -value
            self.set_integer([ref], [value])
        elif basetype == FMIL1.fmi1_base_type_str: #STRING
            self.set_string([ref], [value])
        elif basetype == FMIL1.fmi1_base_type_bool: #BOOLEAN
            if alias_kind == FMI_NEGATED_ALIAS:
                value = not value
            self.set_boolean([ref], [value])
        else:
            raise FMUException('Type not supported.')


    def _get(self, variable_name):
        """
        Helper method to get, see docstring on get.
        """
        cdef FMIL1.fmi1_value_reference_t ref
        cdef FMIL1.fmi1_base_type_enu_t basetype
        cdef FMIL1.fmi1_import_variable_t* variable
        cdef FMIL1.fmi1_variable_alias_kind_enu_t alias_kind

        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL1.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variable_name)

        ref =  FMIL1.fmi1_import_get_variable_vr(variable)
        basetype = FMIL1.fmi1_import_get_variable_base_type(variable)
        alias_kind = FMIL1.fmi1_import_get_variable_alias_kind(variable)

        if basetype == FMIL1.fmi1_base_type_real:  #REAL
            value = self.get_real([ref])
            return -1*value if alias_kind == FMI_NEGATED_ALIAS else value
        elif basetype == FMIL1.fmi1_base_type_int or basetype == FMIL1.fmi1_base_type_enum: #INTEGER
            value = self.get_integer([ref])
            return -1*value if alias_kind == FMI_NEGATED_ALIAS else value
        elif basetype == FMIL1.fmi1_base_type_str: #STRING
            return self.get_string([ref])
        elif basetype == FMIL1.fmi1_base_type_bool: #BOOLEAN
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
        cdef FMIL1.fmi1_import_variable_t* variable
        cdef FMIL1.fmi1_string_t desc

        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL1.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        desc = <FMIL1.fmi1_string_t>FMIL1.fmi1_import_get_variable_description(variable)

        return pyfmi_util.decode(desc) if desc != NULL else ""

    cdef _add_scalar_variable(self, FMIL1.fmi1_import_variable_t* variable):
        cdef FMIL1.fmi1_string_t desc

        if variable == NULL:
            raise FMUException("Unknown variable. Please verify the correctness of the XML file and check the log.")

        alias_kind = FMIL1.fmi1_import_get_variable_alias_kind(variable)
        name       = pyfmi_util.decode(FMIL1.fmi1_import_get_variable_name(variable))
        value_ref  = FMIL1.fmi1_import_get_variable_vr(variable)
        data_type  = FMIL1.fmi1_import_get_variable_base_type(variable)
        has_start  = FMIL1.fmi1_import_get_variable_has_start(variable)
        data_variability = FMIL1.fmi1_import_get_variability(variable)
        data_causality   = FMIL1.fmi1_import_get_causality(variable)
        desc       = <FMIL1.fmi1_string_t>FMIL1.fmi1_import_get_variable_description(variable)

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
        cdef FMIL1.fmi1_import_variable_t* variable

        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL1.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        return self._add_scalar_variable(variable)

    cpdef FMIL1.fmi1_base_type_enu_t get_variable_data_type(self, variable_name) except *:
        """
        Get data type of variable.

        Parameter::

            variable_name --
                The name of the variable.

        Returns::

            The type of the variable.
        """
        cdef FMIL1.fmi1_import_variable_t* variable
        cdef FMIL1.fmi1_base_type_enu_t basetype

        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL1.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        basetype = FMIL1.fmi1_import_get_variable_base_type(variable)

        return basetype

    cpdef FMIL1.fmi1_value_reference_t get_variable_valueref(self, variable_name) except *:
        """
        Extract the ValueReference given a variable name.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The ValueReference for the variable passed as argument.
        """
        cdef FMIL1.fmi1_import_variable_t* variable
        cdef FMIL1.fmi1_value_reference_t vr

        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL1.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
        vr =  FMIL1.fmi1_import_get_variable_vr(variable)

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
        cdef FMIL1.fmi1_import_variable_t *variable
        cdef FMIL1.fmi1_import_real_variable_t *real_variable
        cdef char* variablename = NULL
        cdef FMIL1.fmi1_real_t value

        if valueref is not None:
            variable = FMIL1.fmi1_import_get_variable_by_vr(self._fmu, FMIL1.fmi1_base_type_real, <FMIL1.fmi1_value_reference_t>valueref)
            if variable == NULL:
                raise FMUException("The variable with value reference: %s, could not be found."%str(valueref))
        elif variable_name is not None:
            variable_name = pyfmi_util.encode(variable_name)
            variablename  = variable_name

            variable = FMIL1.fmi1_import_get_variable_by_name(self._fmu, variablename)
            if variable == NULL:
                raise FMUException("The variable %s could not be found."%variablename)
        else:
            raise FMUException('Either provide value reference or variable name.')

        real_variable = FMIL1.fmi1_import_get_variable_as_real(variable)
        if real_variable == NULL:
            raise FMUException("The variable is not a real variable.")

        value = FMIL1.fmi1_import_get_real_variable_nominal(real_variable)

        if _override_erroneous_nominal:
            if variable_name is None:
                variable_name = pyfmi_util.encode(self.get_variable_by_valueref(valueref))
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
        cdef FMIL1.fmi1_import_variable_t *variable

        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL1.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        status = FMIL1.fmi1_import_get_variable_has_start(variable)

        if status == 0:
            raise FMUException("The variable %s does not have a start value."%variablename)

        fixed = FMIL1.fmi1_import_get_variable_is_fixed(variable)

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
        cdef FMIL1.fmi1_import_variable_t *variable
        cdef FMIL1.fmi1_import_integer_variable_t* int_variable
        cdef FMIL1.fmi1_import_real_variable_t* real_variable
        cdef FMIL1.fmi1_import_bool_variable_t* bool_variable
        cdef FMIL1.fmi1_import_enum_variable_t* enum_variable
        cdef FMIL1.fmi1_import_string_variable_t*  str_variable
        cdef FMIL1.fmi1_base_type_enu_t basetype
        cdef int status
        cdef FMIL1.fmi1_boolean_t FMITRUE = 1

        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL1.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        status = FMIL1.fmi1_import_get_variable_has_start(variable)

        if status == 0:
            raise FMUException("The variable %s does not have a start value."%variablename)

        basetype = FMIL1.fmi1_import_get_variable_base_type(variable)

        if basetype == FMIL1.fmi1_base_type_real:
            real_variable = FMIL1.fmi1_import_get_variable_as_real(variable)
            return FMIL1.fmi1_import_get_real_variable_start(real_variable)

        elif basetype == FMIL1.fmi1_base_type_int:
            int_variable = FMIL1.fmi1_import_get_variable_as_integer(variable)
            return FMIL1.fmi1_import_get_integer_variable_start(int_variable)

        elif basetype == FMIL1.fmi1_base_type_bool:
            bool_variable = FMIL1.fmi1_import_get_variable_as_boolean(variable)
            return FMIL1.fmi1_import_get_boolean_variable_start(bool_variable) == FMITRUE

        elif basetype == FMIL1.fmi1_base_type_enum:
            enum_variable = FMIL1.fmi1_import_get_variable_as_enum(variable)
            return FMIL1.fmi1_import_get_enum_variable_start(enum_variable)

        elif basetype == FMIL1.fmi1_base_type_str:
            str_variable = FMIL1.fmi1_import_get_variable_as_string(variable)
            return FMIL1.fmi1_import_get_string_variable_start(str_variable)

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
        cdef FMIL1.fmi1_import_variable_t *variable
        cdef FMIL1.fmi1_import_integer_variable_t* int_variable
        cdef FMIL1.fmi1_import_real_variable_t* real_variable
        cdef FMIL1.fmi1_import_enum_variable_t* enum_variable
        cdef FMIL1.fmi1_base_type_enu_t basetype

        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL1.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        basetype = FMIL1.fmi1_import_get_variable_base_type(variable)

        if basetype == FMIL1.fmi1_base_type_real:
            real_variable = FMIL1.fmi1_import_get_variable_as_real(variable)
            return FMIL1.fmi1_import_get_real_variable_max(real_variable)

        elif basetype == FMIL1.fmi1_base_type_int:
            int_variable = FMIL1.fmi1_import_get_variable_as_integer(variable)
            return FMIL1.fmi1_import_get_integer_variable_max(int_variable)

        elif basetype == FMIL1.fmi1_base_type_enum:
            enum_variable = FMIL1.fmi1_import_get_variable_as_enum(variable)
            return FMIL1.fmi1_import_get_enum_variable_max(enum_variable)

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
        cdef FMIL1.fmi1_import_variable_t *variable
        cdef FMIL1.fmi1_import_integer_variable_t* int_variable
        cdef FMIL1.fmi1_import_real_variable_t* real_variable
        cdef FMIL1.fmi1_import_enum_variable_t* enum_variable
        cdef FMIL1.fmi1_base_type_enu_t basetype

        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL1.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        basetype = FMIL1.fmi1_import_get_variable_base_type(variable)

        if basetype == FMIL1.fmi1_base_type_real:
            real_variable = FMIL1.fmi1_import_get_variable_as_real(variable)
            return FMIL1.fmi1_import_get_real_variable_min(real_variable)

        elif basetype == FMIL1.fmi1_base_type_int:
            int_variable = FMIL1.fmi1_import_get_variable_as_integer(variable)
            return FMIL1.fmi1_import_get_integer_variable_min(int_variable)

        elif basetype == FMIL1.fmi1_base_type_enum:
            enum_variable = FMIL1.fmi1_import_get_variable_as_enum(variable)
            return FMIL1.fmi1_import_get_enum_variable_min(enum_variable)

        else:
            raise FMUException("The variable type does not have a minimum value.")

    @enable_caching
    def get_model_variables(self, type=None, int include_alias=True,
                            causality=None, variability=None,
                            int only_start=False, int only_fixed=False,
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
        cdef FMIL1.fmi1_import_variable_t *variable
        cdef FMIL1.fmi1_import_variable_list_t *variable_list
        cdef FMIL.size_t variable_list_size
        cdef FMIL1.fmi1_value_reference_t value_ref
        cdef FMIL1.fmi1_base_type_enu_t data_type,target_type = FMIL1.fmi1_base_type_real
        cdef FMIL1.fmi1_variability_enu_t data_variability,target_variability = FMIL1.fmi1_variability_enu_constant
        cdef FMIL1.fmi1_causality_enu_t data_causality,target_causality = FMIL1.fmi1_causality_enu_input
        cdef FMIL1.fmi1_variable_alias_kind_enu_t alias_kind
        cdef FMIL1.fmi1_string_t desc
        cdef dict variable_dict = {}
        cdef list filter_list = [], variable_return_list = []
        cdef int  selected_type = 0 #If a type has been selected
        cdef int  selected_variability = 0 #If a variability has been selected
        cdef int  selected_causality = 0 #If a causality has been selected
        cdef int  has_start, is_fixed
        cdef int  selected_filter = 1 if filter else 0
        cdef int  length_filter = 0

        variable_list = FMIL1.fmi1_import_get_variable_list(self._fmu)
        variable_list_size = FMIL1.fmi1_import_get_variable_list_size(variable_list)

        if type is not None: #A type have has been selected
            target_type = type
            selected_type = 1
        if causality is not None: #A causality has been selected
            target_causality = causality
            selected_causality = 1
        if variability is not None: #A variability has been selected
            target_variability = variability
            selected_variability = 1
        if selected_filter:
            filter_list = self._convert_filter(filter)
            length_filter = len(filter_list)

        for i in range(variable_list_size):

            variable = FMIL1.fmi1_import_get_variable(variable_list, i)

            alias_kind = FMIL1.fmi1_import_get_variable_alias_kind(variable)
            name       = pyfmi_util.decode(FMIL1.fmi1_import_get_variable_name(variable))
            value_ref  = FMIL1.fmi1_import_get_variable_vr(variable)
            data_type  = FMIL1.fmi1_import_get_variable_base_type(variable)
            has_start  = FMIL1.fmi1_import_get_variable_has_start(variable)
            data_variability = FMIL1.fmi1_import_get_variability(variable)
            data_causality   = FMIL1.fmi1_import_get_causality(variable)
            desc       = <FMIL1.fmi1_string_t>FMIL1.fmi1_import_get_variable_description(variable)

            #If only variables with start are wanted, check if the variable has start
            if only_start and has_start != 1:
                continue

            if only_fixed:
                if has_start!=1:
                    continue
                else:
                    is_fixed = FMIL1.fmi1_import_get_variable_is_fixed(variable)
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

            elif alias_kind ==FMIL1.fmi1_variable_is_not_alias:
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
        FMIL1.fmi1_import_free_variable_list(variable_list)

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
        cdef FMIL1.fmi1_import_variable_t *variable
        cdef FMIL1.fmi1_import_variable_t *base_variable
        cdef FMIL1.fmi1_import_variable_list_t *variable_list
        cdef FMIL.size_t variable_list_size
        cdef FMIL1.fmi1_value_reference_t value_ref
        cdef FMIL1.fmi1_base_type_enu_t data_type
        cdef FMIL1.fmi1_variability_enu_t data_variability
        cdef FMIL1.fmi1_variable_alias_kind_enu_t alias_kind
        cdef list filter_list = []
        cdef list real_var_ref = []
        cdef list int_var_ref = []
        cdef list bool_var_ref = []
        cdef int  selected_variability = 0 #If a variability has been selected
        cdef int  selected_filter = 1 if filter else 0
        cdef int  length_filter = 0
        cdef dict added_vars = {}

        variable_list = FMIL1.fmi1_import_get_variable_list(self._fmu)
        variable_list_size = FMIL1.fmi1_import_get_variable_list_size(variable_list)

        if selected_filter:
            filter_list = self._convert_filter(filter)
            length_filter = len(filter_list)

        for i in range(variable_list_size):

            variable = FMIL1.fmi1_import_get_variable(variable_list, i)

            alias_kind = FMIL1.fmi1_import_get_variable_alias_kind(variable)
            name       = pyfmi_util.decode(FMIL1.fmi1_import_get_variable_name(variable))
            data_variability = FMIL1.fmi1_import_get_variability(variable)
            data_type  = FMIL1.fmi1_import_get_variable_base_type(variable)

            if data_type != FMIL1.fmi1_base_type_real and data_type != FMIL1.fmi1_base_type_int and data_type != FMIL1.fmi1_base_type_bool and data_type != FMIL1.fmi1_base_type_enum:
                continue

            if data_variability != FMIL1.fmi1_variability_enu_continuous and data_variability != FMIL1.fmi1_variability_enu_discrete:
                continue

            if alias_kind == FMIL1.fmi1_variable_is_not_alias:
                value_ref = FMIL1.fmi1_import_get_variable_vr(variable)
            else:
                base_variable = FMIL1.fmi1_import_get_variable_alias_base(self._fmu, variable)
                value_ref  = FMIL1.fmi1_import_get_variable_vr(base_variable)

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
                if alias_kind != FMIL1.fmi1_variable_is_not_alias:
                    continue

            if data_type == FMIL1.fmi1_base_type_real:
                real_var_ref.append(value_ref)
            if data_type == FMIL1.fmi1_base_type_int or data_type == FMIL1.fmi1_base_type_enum:
                int_var_ref.append(value_ref)
            if data_type == FMIL1.fmi1_base_type_bool:
                bool_var_ref.append(value_ref)

        #Free the variable list
        FMIL1.fmi1_import_free_variable_list(variable_list)

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
        cdef FMIL1.fmi1_import_variable_t* variable
        cdef FMIL1.fmi1_import_variable_t* base_variable
        cdef FMIL1.fmi1_value_reference_t vr

        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL1.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        base_variable = FMIL1.fmi1_import_get_variable_alias_base(self._fmu, variable)
        if base_variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        name = pyfmi_util.decode(FMIL1.fmi1_import_get_variable_name(base_variable))

        return name

    def get_variable_alias(self, variable_name):
        """
        Return a dict of all alias variables belonging to the provided variable
        where the key are the names and the value indicating whether the variable
        should be negated or not.

        Returns::

            A dict consisting of the alias variables along with no alias variable.
            The values indicate whether or not the variable should be negated or not.

        Raises::

            FMUException if the variable is not in the model.
        """
        cdef FMIL1.fmi1_import_variable_t *variable
        cdef FMIL1.fmi1_import_variable_list_t *alias_list
        cdef FMIL.size_t alias_list_size
        cdef FMIL1.fmi1_variable_alias_kind_enu_t alias_kind
        cdef dict ret_values = {}

        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL1.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        alias_list = FMIL1.fmi1_import_get_variable_aliases(self._fmu, variable)

        alias_list_size = FMIL1.fmi1_import_get_variable_list_size(alias_list)

        #Loop over all the alias variables
        for i in range(alias_list_size):

            variable = FMIL1.fmi1_import_get_variable(alias_list, i)

            alias_kind = FMIL1.fmi1_import_get_variable_alias_kind(variable)
            alias_name = pyfmi_util.decode(FMIL1.fmi1_import_get_variable_name(variable))

            ret_values[alias_name] = alias_kind

        #FREE VARIABLE LIST
        FMIL1.fmi1_import_free_variable_list(alias_list)

        return ret_values

    cpdef FMIL1.fmi1_variability_enu_t get_variable_variability(self, variable_name) except *:
        """
        Get variability of variable.

        Parameters::

            variablename --

                The name of the variable.

        Returns::

            The variability of the variable, CONSTANT(0), PARAMETER(1),
            DISCRETE(2) or CONTINUOUS(3)
        """
        cdef FMIL1.fmi1_import_variable_t* variable
        cdef FMIL1.fmi1_variability_enu_t variability

        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL1.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)
        variability = FMIL1.fmi1_import_get_variability(variable)

        return variability

    def get_variable_by_valueref(self, FMIL1.fmi1_value_reference_t valueref, type=0):
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
        cdef FMIL1.fmi1_import_variable_t* variable

        variable = FMIL1.fmi1_import_get_variable_by_vr(self._fmu, type, valueref)
        if variable==NULL:
            raise FMUException("The variable with the valuref %i could not be found."%valueref)

        name = pyfmi_util.decode(FMIL1.fmi1_import_get_variable_name(variable))

        return name

    cpdef FMIL1.fmi1_causality_enu_t get_variable_causality(self, variable_name) except *:
        """
        Get the causality of the variable.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The causality of the variable, INPUT(0), OUTPUT(1),
            INTERNAL(2), NONE(3)
        """
        cdef FMIL1.fmi1_import_variable_t* variable
        cdef FMIL1.fmi1_causality_enu_t causality

        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL1.fmi1_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        causality = FMIL1.fmi1_import_get_causality(variable)

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
        return self._modelId

    def get_author(self):
        """
        Return the name and organization of the model author.
        """
        cdef FMIL1.fmi1_string_t author
        author = <FMIL1.fmi1_string_t>FMIL1.fmi1_import_get_author(self._fmu)
        return pyfmi_util.decode(author) if author != NULL else ""

    def get_default_experiment_start_time(self):
        """
        Returns the default experiment start time as defined the XML
        description.
        """
        return FMIL1.fmi1_import_get_default_experiment_start(self._fmu)

    def get_default_experiment_stop_time(self):
        """
        Returns the default experiment stop time as defined the XML
        description.
        """
        return FMIL1.fmi1_import_get_default_experiment_stop(self._fmu)

    def get_default_experiment_tolerance(self):
        """
        Returns the default experiment tolerance as defined in the XML
        description.
        """
        return FMIL1.fmi1_import_get_default_experiment_tolerance(self._fmu)

    def get_description(self):
        """
        Return the model description.
        """
        cdef FMIL1.fmi1_string_t desc
        desc = <FMIL1.fmi1_string_t>FMIL1.fmi1_import_get_description(self._fmu)
        return pyfmi_util.decode(desc) if desc != NULL else ""

    def get_generation_tool(self):
        """
        Return the model generation tool.
        """
        cdef FMIL1.fmi1_string_t gen
        gen = <FMIL1.fmi1_string_t>FMIL1.fmi1_import_get_generation_tool(self._fmu)
        return pyfmi_util.decode(gen) if gen != NULL else ""

    def get_guid(self):
        """
        Return the model GUID.
        """
        guid = pyfmi_util.decode(FMIL1.fmi1_import_get_GUID(self._fmu))
        return guid

cdef class FMUModelCS1(FMUModelBase):
    #First step only support fmi1_fmu_kind_enu_cs_standalone
    #stepFinished not supported

    def __init__(self, fmu, log_file_name="", log_level=FMI_DEFAULT_LOG_LEVEL,
                 _unzipped_dir=None, _connect_dll=True, allow_unzipped_fmu = False):
        #Call super
        FMUModelBase.__init__(self,fmu,log_file_name, log_level, _unzipped_dir, _connect_dll, allow_unzipped_fmu)

        if self._fmu_kind != FMI_CS_STANDALONE and self._fmu_kind != FMI_CS_TOOL:
            raise InvalidVersionException("The FMU could not be loaded. This class only supports FMI 1.0 for Co-simulation.")

        if _connect_dll:
            global GLOBAL_FMU_OBJECT
            GLOBAL_FMU_OBJECT = self

            self.callbacks_defaults.malloc  = FMIL.malloc
            self.callbacks_defaults.calloc  = FMIL.calloc
            self.callbacks_defaults.realloc = FMIL.realloc
            self.callbacks_defaults.free    = FMIL.free
            self.callbacks_defaults.logger  = importlogger_default
            self.callbacks_defaults.context = <void*>self
            self.callbacks_defaults.log_level = self.callbacks.log_level

            self.callbacks_standard = FMIL.jm_get_default_callbacks()
            FMIL.jm_set_default_callbacks(&self.callbacks_defaults)

            self.instantiate_slave(logging = self._enable_logging)

            FMIL.jm_set_default_callbacks(self.callbacks_standard)

            GLOBAL_FMU_OBJECT = None

    cpdef _get_time(self):
        return self._t

    cpdef _set_time(self, FMIL1.fmi1_real_t t):
        self._t = t

    def _get_module_name(self):
        return "Slave"

    time = property(_get_time,_set_time, doc =
    """
    Property for accessing the current time of the simulation. Calls the
    low-level FMI function: fmiSetTime.
    """)

    def __dealloc__(self):
        """
        Deallocate memory allocated
        """
        self._invoked_dealloc = 1

        if self._allocated_fmu == 1:
            FMIL1.fmi1_import_terminate_slave(self._fmu)

        if self._instantiated_fmu == 1:
            FMIL1.fmi1_import_free_slave_instance(self._fmu)

        if self._allocated_dll == 1:
            FMIL1.fmi1_import_destroy_dllfmu(self._fmu)

        if self._allocated_xml == 1:
            FMIL1.fmi1_import_free(self._fmu)

        if self._fmu_temp_dir != NULL:
            if not self._allow_unzipped_fmu:
                FMIL.fmi_import_rmdir(&self.callbacks, self._fmu_temp_dir)
            FMIL.free(self._fmu_temp_dir)
            self._fmu_temp_dir = NULL

        if self._allocated_context == 1:
            FMIL.fmi_import_free_context(self.context)

        if self._fmu_log_name != NULL:
            FMIL.free(self._fmu_log_name)
            self._fmu_log_name = NULL

        if self._log_stream:
            self._log_stream = None

    def do_step(self, FMIL1.fmi1_real_t current_t, FMIL1.fmi1_real_t step_size, new_step=True):
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
        cdef FMIL1.fmi1_boolean_t new_s

        if new_step:
            new_s = 1
        else:
            new_s = 0

        self.time = current_t+step_size

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_do_step(self._fmu, current_t, step_size, new_s)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        return status


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
        cdef int status
        cdef unsigned int max_output_derivative
        cdef FMIL.size_t nref
        cdef np.ndarray[FMIL1.fmi1_real_t, ndim=1,mode='c'] values
        cdef np.ndarray[FMIL1.fmi1_value_reference_t, ndim=1,mode='c'] value_refs
        cdef np.ndarray[FMIL1.fmi1_integer_t, ndim=1,mode='c'] orders
        cdef FMIL1.fmi1_import_capabilities_t *fmu_capabilities

        fmu_capabilities = FMIL1.fmi1_import_get_capabilities(self._fmu)
        max_output_derivative = FMIL1.fmi1_import_get_maxOutputDerivativeOrder(fmu_capabilities)

        if order < 1 or order > max_output_derivative:
            raise FMUException("The order must be greater than zero and below the maximum output derivative support of the FMU (%d)."%max_output_derivative)

        if isinstance(variables,str):
            nref = 1
            value_refs = np.array([0], dtype=np.uint32,ndmin=1).ravel()
            orders = np.array(order, dtype=np.int32)
            value_refs[0] = self.get_variable_valueref(variables)
        elif isinstance(variables,list) and isinstance(variables[-1],str):
            nref = len(variables)
            value_refs = np.array([0]*nref, dtype=np.uint32,ndmin=1).ravel()
            orders = np.array([0]*nref, dtype=np.int32)
            for i in range(nref):
                value_refs[i] = self.get_variable_valueref(variables[i])
                orders[i] = order
        else:
            raise FMUException("The variables must either be a string or a list of strings")

        values = np.array([0.0]*nref,dtype=float, ndmin=1)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_get_real_output_derivatives(self._fmu, <FMIL1.fmi1_value_reference_t*>value_refs.data, nref, <FMIL1.fmi1_integer_t*>orders.data, <FMIL1.fmi1_real_t*>values.data)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

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
        cdef FMIL1.fmi1_string_t types_platform
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        types_platform = <FMIL1.fmi1_string_t>FMIL1.fmi1_import_get_types_platform(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        return pyfmi_util.decode(types_platform)

    types_platform = property(fget=_get_types_platform)

    def get_real_status(self, status_kind):
        """
        """
        cdef FMIL1.fmi1_real_t value = 0.0
        cdef int status

        if status_kind == FMIL1.fmi1_last_successful_time:
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            status = FMIL1.fmi1_import_get_real_status(self._fmu, FMIL1.fmi1_last_successful_time, &value)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

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
        cdef FMIL1.fmi1_import_capabilities_t *fmu_capabilities
        cdef np.ndarray[FMIL1.fmi1_integer_t, ndim=1,mode='c'] np_orders = np.array(orders, dtype=np.int32, ndmin=1).ravel()
        cdef np.ndarray[FMIL1.fmi1_value_reference_t, ndim=1,mode='c'] value_refs
        cdef np.ndarray[FMIL1.fmi1_real_t, ndim=1,mode='c'] val = np.array(values, dtype=float, ndmin=1).ravel()
        cdef FMIL.size_t nref = np.size(val)
        orders = np.array([0]*nref, dtype=np.int32)

        if nref != np.size(np_orders):
            raise FMUException("The number of variables must be the same as the number of orders.")

        fmu_capabilities = FMIL1.fmi1_import_get_capabilities(self._fmu)
        can_interpolate_inputs = FMIL1.fmi1_import_get_canInterpolateInputs(fmu_capabilities)
        #NOTE IS THIS THE HIGHEST ORDER OF INTERPOLATION OR SIMPLY IF IT CAN OR NOT?

        for i in range(np.size(np_orders)):
            if np_orders[i] < 1:
                raise FMUException("The order must be greater than zero.")
        if not can_interpolate_inputs:
            raise FMUException("The FMU does not support input derivatives.")

        if isinstance(variables,str):
            value_refs = np.array([0], dtype=np.uint32,ndmin=1).ravel()
            value_refs[0] = self.get_variable_valueref(variables)
        elif isinstance(variables,list) and isinstance(variables[-1],str):
            value_refs = np.array([0]*nref, dtype=np.uint32,ndmin=1).ravel()
            for i in range(nref):
                value_refs[i] = self.get_variable_valueref(variables[i])
        else:
            raise FMUException("The variables must either be a string or a list of strings")

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_set_real_input_derivatives(self._fmu, <FMIL1.fmi1_value_reference_t*>value_refs.data, nref, <FMIL1.fmi1_integer_t*>np_orders.data, <FMIL1.fmi1_real_t*>val.data)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

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
        cdef int status
        cdef FMIL1.fmi1_boolean_t stop_defined

        self.time = start_time
        stop_defined = 1 if stop_time_defined else 0

        log_open = self._log_open()
        if not log_open and self.get_log_level() > 2:
            self._open_log_file()

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_initialize_slave(self._fmu, start_time, stop_defined, stop_time)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if not log_open and self.get_log_level() > 2:
            self._close_log_file()

        if status != FMIL1.fmi1_status_ok:
            raise FMUException("The slave failed to initialize. See the log for possibly more information.")

        self._allocated_fmu = 1

    def reset(self):
        """
        This method resets the FMU according to the reset method defined
        in the FMI1 specification.
        """

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_reset_slave(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        if status != FMIL1.fmi1_status_ok:
            raise FMUException("Failed to reset the FMU.")

        #The FMU is no longer initialized
        self._allocated_fmu = 0

        #Default values
        self._t = None

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
        cdef FMIL1.fmi1_boolean_t log = 1 if logging else 0
        cdef FMIL1.fmi1_real_t timeout = 0.0
        cdef FMIL1.fmi1_boolean_t visible = 0
        cdef FMIL1.fmi1_boolean_t interactive = 0
        cdef FMIL1.fmi1_string_t location = NULL

        name = pyfmi_util.encode(name)
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_instantiate_slave(self._fmu, name, location,
                                        FMI_MIME_CS_STANDALONE, timeout, visible,
                                        interactive)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != FMIL.jm_status_success:
            raise FMUException('Failed to instantiate the slave. See the log for possibly more information.')

        #The FMU is instantiated
        self._instantiated_fmu = 1

        #Just to be safe, some problems with Dymola (2012) FMUs not reacting
        #to logging when set to the instantiate method.
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_set_debug_logging(self._fmu, log)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

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
        cdef FMIL1.fmi1_import_capabilities_t *cap

        cap = FMIL1.fmi1_import_get_capabilities(self._fmu)

        capabilities["canHandleVariableCommunicationStepSize"] = FMIL1.fmi1_import_get_canHandleVariableCommunicationStepSize(cap)
        capabilities["canHandleEvents"] = FMIL1.fmi1_import_get_canHandleEvents(cap)
        capabilities["canRejectSteps"] = FMIL1.fmi1_import_get_canRejectSteps(cap)
        capabilities["canInterpolateInputs"] = FMIL1.fmi1_import_get_canInterpolateInputs(cap)
        capabilities["maxOutputDerivativeOrder"] = FMIL1.fmi1_import_get_maxOutputDerivativeOrder(cap)
        capabilities["canRunAsynchronuously"] = FMIL1.fmi1_import_get_canRunAsynchronuously(cap)
        capabilities["canSignalEvents"] = FMIL1.fmi1_import_get_canSignalEvents(cap)
        capabilities["canBeInstantiatedOnlyOncePerProcess"] = FMIL1.fmi1_import_get_canBeInstantiatedOnlyOncePerProcess(cap)
        capabilities["canNotUseMemoryManagementFunctions"] = FMIL1.fmi1_import_get_canNotUseMemoryManagementFunctions(cap)

        return capabilities

    def terminate(self):
        """
        Calls the FMI function fmiTerminateSlave() on the FMU.
        After this call, any call to a function changing the state of the FMU will fail.
        """
        if self._allocated_fmu == 1:
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            FMIL1.fmi1_import_terminate_slave(self._fmu)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
            self._allocated_fmu = 0 #No longer initialized

    def free_instance(self):
        """
        Calls the FMI function fmiFreeSlaveInstance on the FMU.
        Note that this is not needed in general as it is done automatically.
        """
        if self._instantiated_fmu == 1:
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            FMIL1.fmi1_import_free_slave_instance(self._fmu)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
            self._instantiated_fmu = 0

cdef class FMUModelME1(FMUModelBase):
    """
    An FMI Model loaded from a DLL.
    """

    def __init__(self, fmu, log_file_name="", log_level=FMI_DEFAULT_LOG_LEVEL,
                 _unzipped_dir=None, _connect_dll=True, allow_unzipped_fmu = False):
        #Call super
        FMUModelBase.__init__(self,fmu,log_file_name, log_level, _unzipped_dir, _connect_dll, allow_unzipped_fmu)

        # State nominals retrieved before initialization
        self._preinit_nominal_continuous_states = None

        if self._fmu_kind != FMI_ME:
            raise InvalidVersionException("The FMU could not be loaded. This class only supports FMI 1.0 for Model Exchange.")

        if _connect_dll:
            global GLOBAL_FMU_OBJECT
            GLOBAL_FMU_OBJECT = self

            self.callbacks_defaults.malloc  = FMIL.malloc
            self.callbacks_defaults.calloc  = FMIL.calloc
            self.callbacks_defaults.realloc = FMIL.realloc
            self.callbacks_defaults.free    = FMIL.free
            self.callbacks_defaults.logger  = importlogger_default
            self.callbacks_defaults.context = <void*>self
            self.callbacks_defaults.log_level = self.callbacks.log_level

            self.callbacks_standard = FMIL.jm_get_default_callbacks()
            FMIL.jm_set_default_callbacks(&self.callbacks_defaults)

            self.instantiate_model(logging = self._enable_logging)

            FMIL.jm_set_default_callbacks(self.callbacks_standard)

            GLOBAL_FMU_OBJECT = None

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
        cdef FMIL1.fmi1_string_t model_types_platform
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        model_types_platform = <FMIL1.fmi1_string_t>FMIL1.fmi1_import_get_model_types_platform(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        return pyfmi_util.decode(model_types_platform)

    model_types_platform = property(fget=_get_model_types_platform)

    def reset(self):
        """
        This method resets the FMU by first calling fmiTerminate and
        fmiFreeModelInstance and then reloads the DLL and finally
        re-instantiates using fmiInstantiateModel.
        """
        if self._allocated_fmu == 1:
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            FMIL1.fmi1_import_terminate(self._fmu)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
            self._allocated_fmu = 0

        if self._instantiated_fmu == 1:
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            FMIL1.fmi1_import_free_model_instance(self._fmu)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
            self._instantiated_fmu = 0

        if self._allocated_dll == 1:
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            FMIL1.fmi1_import_destroy_dllfmu(self._fmu)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        global FMI_REGISTER_GLOBALLY
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_create_dllfmu(self._fmu, self.callBackFunctions, FMI_REGISTER_GLOBALLY);
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        if status == FMIL.jm_status_error:
            raise FMUException("The DLL could not be reloaded, check the log for more information.")
        FMI_REGISTER_GLOBALLY += 1 #Update the global register of FMUs

        #Default values
        self._t = None

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
        self._invoked_dealloc = 1

        if self._allocated_fmu == 1:
            FMIL1.fmi1_import_terminate(self._fmu)

        if self._instantiated_fmu == 1:
            FMIL1.fmi1_import_free_model_instance(self._fmu)

        if self._allocated_dll == 1:
            FMIL1.fmi1_import_destroy_dllfmu(self._fmu)

        if self._allocated_xml == 1:
            FMIL1.fmi1_import_free(self._fmu)

        if self._fmu_temp_dir != NULL:
            if not self._allow_unzipped_fmu:
                FMIL.fmi_import_rmdir(&self.callbacks, self._fmu_temp_dir)
            FMIL.free(self._fmu_temp_dir)
            self._fmu_temp_dir = NULL

        if self._allocated_context == 1:
            FMIL.fmi_import_free_context(self.context)

        if self._fmu_log_name != NULL:
            FMIL.free(self._fmu_log_name)
            self._fmu_log_name = NULL

        if self._log_stream:
            self._log_stream = None

    cpdef _get_time(self):
        return self._t

    cpdef _set_time(self, FMIL1.fmi1_real_t t):
        cdef int status
        self._t = t

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_set_time(self._fmu,t)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to set the time.')

    time = property(_get_time,_set_time, doc =
    """
    Property for accessing the current time of the simulation. Calls the
    low-level FMI function: fmiSetTime.
    """)

    def _get_continuous_states(self):
        cdef int status
        cdef np.ndarray[double, ndim=1,mode='c'] ndx = np.zeros(self._nContinuousStates, dtype=np.double)
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_get_continuous_states(self._fmu, <FMIL1.fmi1_real_t*>ndx.data, self._nContinuousStates)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to retrieve the continuous states.')

        return ndx

    def _set_continuous_states(self, np.ndarray[FMIL1.fmi1_real_t] values):
        cdef int status
        cdef np.ndarray[FMIL1.fmi1_real_t, ndim=1,mode='c'] ndx = values

        if np.size(ndx) != self._nContinuousStates:
            raise FMUException(
                'Failed to set the new continuous states. ' \
                'The number of values are not consistent with the number of '\
                'continuous states.')

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_set_continuous_states(self._fmu, <FMIL1.fmi1_real_t*>ndx.data, self._nContinuousStates)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status >= 3:
            raise FMUException('Failed to set the new continuous states.')

    continuous_states = property(_get_continuous_states, _set_continuous_states,
        doc=
    """
    Property for accessing the current values of the continuous states. Calls
    the low-level FMI function: fmiSetContinuousStates/fmiGetContinuousStates.
    """)


    cdef int _get_nominal_continuous_states_fmil(self, FMIL1.fmi1_real_t* xnominal, size_t nx):
        cdef int status
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_get_nominal_continuous_states(self._fmu, xnominal, nx)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        return status

    def _get_nominal_continuous_states(self):
        """
        Returns the nominal values of the continuous states.

        Returns::
            The nominal values.
        """
        cdef int status
        cdef np.ndarray[FMIL1.fmi1_real_t, ndim=1, mode='c'] xn = np.zeros(self._nContinuousStates, dtype=np.double)

        status = self._get_nominal_continuous_states_fmil(<FMIL1.fmi1_real_t*> xn.data, self._nContinuousStates)
        if status != 0:
            raise FMUException('Failed to get the nominal values.')

        # Fallback - auto-correct the illegal nominal values:
        xvrs = self.get_state_value_references()
        for i in range(self._nContinuousStates):
            if xn[i] == 0.0:
                if self.callbacks.log_level >= FMIL.jm_log_level_warning:
                    xname = self.get_variable_by_valueref(xvrs[i])
                    logging.warning(f"The nominal value for {xname} is 0.0 which is illegal according " + \
                                     "to the FMI specification. Setting the nominal to 1.0.")
                xn[i] = 1.0
            elif xn[i] < 0.0:
                if self.callbacks.log_level >= FMIL.jm_log_level_warning:
                    xname = self.get_variable_by_valueref(xvrs[i])
                    logging.warning(f"The nominal value for {xname} is <0.0 which is illegal according " + \
                                    f"to the FMI specification. Setting the nominal to abs({xn[i]}).")
                xn[i] = abs(xn[i])

        # If called before initialization, save values in order to later perform auto-correction
        if self._allocated_fmu == 0:
            self._preinit_nominal_continuous_states = xn

        return xn

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
        cdef np.ndarray[FMIL1.fmi1_real_t, ndim=1,mode='c'] values = np.empty(self._nContinuousStates,dtype=np.double)

        if self._nContinuousStates > 0:
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            status = FMIL1.fmi1_import_get_derivatives(self._fmu, <FMIL1.fmi1_real_t*>values.data, self._nContinuousStates)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        else:
            return values

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
        cdef np.ndarray[FMIL1.fmi1_real_t, ndim=1,mode='c'] values = np.empty(self._nEventIndicators,dtype=np.double)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_get_event_indicators(self._fmu, <FMIL1.fmi1_real_t*>values.data, self._nEventIndicators)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

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

        This method should not be called before initialization, since it depends on state nominals
        which can change during initialization.

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
        return self.get_default_experiment_tolerance()

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
        cdef FMIL1.fmi1_boolean_t intermediate_result

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        if intermediateResult:
            intermediate_result = 1
            status = FMIL1.fmi1_import_eventUpdate(self._fmu, intermediate_result, &self._eventInfo)
        else:
            intermediate_result = 0
            status = FMIL1.fmi1_import_eventUpdate(self._fmu, intermediate_result, &self._eventInfo)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

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
        cdef np.ndarray[FMIL1.fmi1_value_reference_t, ndim=1,mode='c'] values = np.zeros(self._nContinuousStates,dtype=np.uint32)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_get_state_value_references(
            self._fmu, <FMIL1.fmi1_value_reference_t*>values.data, self._nContinuousStates)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

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
        cdef FMIL1.fmi1_boolean_t call_event_update

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_completed_integrator_step(self._fmu, &call_event_update)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to call FMI completed step at time: %E.'%self.time)

        if call_event_update == 1:
            return True
        else:
            return False


    def initialize(self, tolerance_defined=True, tolerance="Default"):
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
        cdef FMIL1.fmi1_real_t c_tolerance

        #Trying to set the initial time from the xml file, else 0.0
        if self.time is None:
            self.time = FMIL1.fmi1_import_get_default_experiment_start(self._fmu)

        if tolerance_defined:
            tolerance_controlled = 1
            if tolerance == "Default":
                c_tolerance = FMIL1.fmi1_import_get_default_experiment_tolerance(self._fmu)
            else:
                c_tolerance = tolerance
        else:
            tolerance_controlled = 0
            c_tolerance = 0.0

        log_open = self._log_open()
        if not log_open and self.get_log_level() > 2:
            self._open_log_file()

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_initialize(self._fmu, tolerance_controlled, c_tolerance, &self._eventInfo)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

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
        cdef FMIL1.fmi1_boolean_t log
        cdef int status

        if logging:
            log = 1
        else:
            log = 0

        name = pyfmi_util.encode(name)
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_instantiate_model(self._fmu, name)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to instantiate the model. See the log for possibly more information.')

        #The FMU is instantiated
        self._instantiated_fmu = 1

        #Just to be safe, some problems with Dymola (2012) FMUs not reacting
        #to logging when set to the instantiate method.
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL1.fmi1_import_set_debug_logging(self._fmu, log)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

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
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            FMIL1.fmi1_import_terminate(self._fmu)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
            self._allocated_fmu = 0 #No longer initialized

    def free_instance(self):
        """
        Calls the FMI function fmiFreeModelInstance on the FMU.
        Note that this is not needed in general as it is done automatically.
        """
        if self._instantiated_fmu == 1:
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            FMIL1.fmi1_import_free_model_instance(self._fmu)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
            self._instantiated_fmu = 0

cdef object _load_fmi1_fmu(
    fmu, 
    object log_file_name, 
    str kind, 
    int log_level, 
    int allow_unzipped_fmu,
    FMIL.fmi_import_context_t* context, 
    bytes fmu_temp_dir,
    FMIL.jm_callbacks callbacks,
    list log_data
):
    """
    The FMI1 part of fmi.pyx load_fmu.
    """
    # TODO: Duplicated code here for error handling
    cdef FMIL.jm_string last_error
    cdef FMIL1.fmi1_import_t* fmu_1 = NULL
    cdef FMIL1.fmi1_fmu_kind_enu_t fmu_1_kind
    model = None

    # Check the fmu-kind
    fmu_1 = FMIL1.fmi1_import_parse_xml(context, fmu_temp_dir)

    if fmu_1 is NULL:
        # Delete the context
        last_error = FMIL.jm_get_last_error(&callbacks)
        FMIL.fmi_import_free_context(context)
        if not allow_unzipped_fmu:
            FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
        if callbacks.log_level >= FMIL.jm_log_level_error:
            _handle_load_fmu_exception(log_data)
            raise InvalidXMLException("The FMU could not be loaded. The model data from 'modelDescription.xml' within the FMU could not be read. "+pyfmi_util.decode(last_error))
        else:
            _handle_load_fmu_exception(log_data)
            raise InvalidXMLException("The FMU could not be loaded. The model data from 'modelDescription.xml' within the FMU could not be read. Enable logging for possible nore information.")

    fmu_1_kind = FMIL1.fmi1_import_get_fmu_kind(fmu_1)

    # Compare fmu_kind with input-specified kind
    if fmu_1_kind == FMI_ME and kind.upper() != 'CS':
        model = FMUModelME1(fmu, log_file_name, log_level, _unzipped_dir = fmu_temp_dir,
                            allow_unzipped_fmu = allow_unzipped_fmu)
    elif (fmu_1_kind == FMI_CS_STANDALONE or fmu_1_kind == FMI_CS_TOOL) and kind.upper() != 'ME':
        model = FMUModelCS1(fmu, log_file_name, log_level, _unzipped_dir = fmu_temp_dir,
                            allow_unzipped_fmu = allow_unzipped_fmu)
    else:
        FMIL1.fmi1_import_free(fmu_1)
        FMIL.fmi_import_free_context(context)
        if not allow_unzipped_fmu:
            FMIL.fmi_import_rmdir(&callbacks,fmu_temp_dir)
        _handle_load_fmu_exception(log_data)
        raise FMUException("FMU is a {} and not a {}".format(pyfmi_util.decode(FMIL1.fmi1_fmu_kind_to_string(fmu_1_kind)), kind.upper()))

    FMIL1.fmi1_import_free(fmu_1)
    FMIL.fmi_import_free_context(context)

    return model
