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

cimport cython

import os
from enum import IntEnum
import logging
from typing import Union

import numpy as np
cimport numpy as np

cimport pyfmi.fmil_import as FMIL
cimport pyfmi.fmil3_import as FMIL3
cimport pyfmi.fmi_base as FMI_BASE
cimport pyfmi.util as pyfmi_util
from pyfmi.util import enable_caching


# TYPES
# TODO: Import into fmi.pyx for convenience imports?
class FMI3_Type(IntEnum):
    FLOAT64 = FMIL3.fmi3_base_type_float64
    FLOAT32 = FMIL3.fmi3_base_type_float32
    INT64   = FMIL3.fmi3_base_type_int64
    INT32   = FMIL3.fmi3_base_type_int32
    INT16   = FMIL3.fmi3_base_type_int16
    INT8    = FMIL3.fmi3_base_type_int8
    UINT64  = FMIL3.fmi3_base_type_uint64
    UINT32  = FMIL3.fmi3_base_type_uint32
    UINT16  = FMIL3.fmi3_base_type_uint16
    UINT8   = FMIL3.fmi3_base_type_uint8
    BOOL    = FMIL3.fmi3_base_type_bool
    BINARY  = FMIL3.fmi3_base_type_binary
    CLOCK   = FMIL3.fmi3_base_type_clock
    STRING  = FMIL3.fmi3_base_type_str
    ENUM    = FMIL3.fmi3_base_type_enum

class FMI3_Initial(IntEnum):
    EXACT       = FMIL3.fmi3_initial_enu_exact
    APPROX      = FMIL3.fmi3_initial_enu_approx
    CALCULATED  = FMIL3.fmi3_initial_enu_calculated
    UNKNOWN     = FMIL3.fmi3_initial_enu_unknown

class FMI3_Variability(IntEnum):
    CONSTANT    = FMIL3.fmi3_variability_enu_constant
    FIXED       = FMIL3.fmi3_variability_enu_fixed
    TUNABLE     = FMIL3.fmi3_variability_enu_tunable
    DISCRETE    = FMIL3.fmi3_variability_enu_discrete
    CONTINUOUS  = FMIL3.fmi3_variability_enu_continuous
    UNKNOWN     = FMIL3.fmi3_variability_enu_unknown

class FMI3_Causality(IntEnum):
    STRUCTURAL_PARAMETER    = FMIL3.fmi3_causality_enu_structural_parameter
    PARAMETER               = FMIL3.fmi3_causality_enu_parameter
    CALCULATED_PARAMETER    = FMIL3.fmi3_causality_enu_calculated_parameter
    INPUT                   = FMIL3.fmi3_causality_enu_input
    OUTPUT                  = FMIL3.fmi3_causality_enu_output
    LOCAL                   = FMIL3.fmi3_causality_enu_local
    INDEPENDENT             = FMIL3.fmi3_causality_enu_independent
    UNKNOWN                 = FMIL3.fmi3_causality_enu_unknown

from pyfmi.exceptions import (
    FMUException,
    InvalidXMLException,
    InvalidFMUException,
    InvalidBinaryException,
    InvalidVersionException,
)

from pyfmi.fmi_base import (
    FMI_DEFAULT_LOG_LEVEL,
    _handle_load_fmu_exception,
    check_fmu_args
)

from pyfmi.common.core import create_temp_dir

# CALLBACKS
cdef void importlogger3(FMIL.jm_callbacks* c, FMIL.jm_string module, FMIL.jm_log_level_enu_t log_level, FMIL.jm_string message):
    if c.context != NULL:
        (<FMUModelBase3>c.context)._logger(module, log_level, message)


cdef class FMI3ModelVariable:
    """ Class defining data structure based on the XML elements of ModelVariables. """
    def __init__(self, name, value_reference, data_type, description, variability, causality, alias, initial):
        self._name            = name
        self._value_reference = value_reference
        self._type            = data_type
        self._description     = description
        self._variability     = variability
        self._causality       = causality
        self._initial         = initial
        self._alias           = alias

    def _get_name(self):
        return self._name
    name = property(_get_name)

    def _get_alias(self):
        return self._alias
    alias = property(_get_alias)

    def _get_value_reference(self):
        return self._value_reference
    value_reference = property(_get_value_reference)

    def _get_type(self):
        return FMI3_Type(self._type)
    type = property(_get_type)

    def _get_description(self):
        return self._description
    description = property(_get_description)

    def _get_variability(self):
        return FMI3_Variability(self._variability)
    variability = property(_get_variability)

    def _get_causality(self):
        return FMI3_Causality(self._causality)
    causality = property(_get_causality)

    def _get_initial(self):
        return FMI3_Initial(self._initial)
    initial = property(_get_initial)

cdef class FMI3EventInfo:
    """ Class representing data related to event information."""
    def __init__(self):
        self.new_discrete_states_needed = FMIL3.fmi3_false
        self.terminate_simulation = FMIL3.fmi3_false
        self.nominals_of_continuous_states_changed = FMIL3.fmi3_false
        self.values_of_continuous_states_changed = FMIL3.fmi3_false
        self.next_event_time_defined = FMIL3.fmi3_false
        self.next_event_time = 0.0

cdef class FMUModelBase3(FMI_BASE.ModelBase):
    """
    FMI3 Model loaded from a dll.
    """
    def __init__(self, fmu, log_file_name = "", log_level = FMI_DEFAULT_LOG_LEVEL,
                 _unzipped_dir = None, _connect_dll = True, allow_unzipped_fmu = False):
        """
        Constructor of the model.

        Parameters::

            fmu --
                Name of the fmu as a string.

            log_file_name --
                Filename for file used to save log messages.
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

            A model as an object from the class FMUModelFMU3
        """
        logging.warning("FMI3 support is experimental.")
        # Call super
        FMI_BASE.ModelBase.__init__(self)

        # Contains the log information
        self._log = []

        # Used for deallocation
        self._allocated_context = 0
        self._allocated_xml = 0
        self._fmu_temp_dir = NULL
        self._fmu_log_name = NULL

        # Used to adjust behavior if FMU is unzipped
        self._allow_unzipped_fmu = 1 if allow_unzipped_fmu else 0

        # Default values
        self._t = None
        self._last_accepted_time = 0.0

        # Internal values
        self._enable_logging = False
        self._event_info   = FMI3EventInfo()

        # Specify the general callback functions
        self.callbacks.malloc  = FMIL.malloc
        self.callbacks.calloc  = FMIL.calloc
        self.callbacks.realloc = FMIL.realloc
        self.callbacks.free    = FMIL.free
        self.callbacks.logger  = importlogger3
        self.callbacks.context = <void*>self

        if isinstance(log_level, int) and (log_level >= FMIL.jm_log_level_nothing and log_level <= FMIL.jm_log_level_all):
            if log_level == FMIL.jm_log_level_nothing:
                self._enable_logging = False
            else:
                self._enable_logging = True
            self.callbacks.log_level = log_level
        else:
            raise FMUException(f"The log level must be an integer between {FMIL.jm_log_level_nothing} and {FMIL.jm_log_level_all}")
        self._fmu_full_path = pyfmi_util.encode(os.path.abspath(fmu))
        check_fmu_args(self._allow_unzipped_fmu, fmu, self._fmu_full_path)

        # Create a struct for allocation
        self._context = FMIL.fmi_import_allocate_context(&self.callbacks)
        self._allocated_context = 1

        # Get the FMI version of the provided model
        if _unzipped_dir:
            fmu_temp_dir = pyfmi_util.encode(_unzipped_dir)
        elif self._allow_unzipped_fmu:
            fmu_temp_dir = pyfmi_util.encode(fmu)
        else:
            fmu_temp_dir = pyfmi_util.encode(create_temp_dir())
        fmu_temp_dir = os.path.abspath(fmu_temp_dir)
        self._fmu_temp_dir = <char*>FMIL.malloc((FMIL.strlen(fmu_temp_dir)+1)*sizeof(char))
        FMIL.strcpy(self._fmu_temp_dir, fmu_temp_dir)

        if _unzipped_dir:
            # If the unzipped directory is provided we assume that the version
            # is correct. This is due to that the method to get the version
            # unzips the FMU which we already have done.
            self._version = FMIL.fmi_version_3_0_enu
        else:
            self._version = FMI_BASE.import_and_get_version(self._context, self._fmu_full_path,
                                                            fmu_temp_dir, self._allow_unzipped_fmu)
        # Check the version
        if self._version == FMIL.fmi_version_unknown_enu:
            last_error = pyfmi_util.decode(FMIL.jm_get_last_error(&self.callbacks))
            if self._enable_logging:
                raise InvalidVersionException("The FMU could not be loaded. The FMU version could not be determined. " + last_error)
            else:
                raise InvalidVersionException("The FMU could not be loaded. The FMU version could not be determined. Enable logging for possibly more information.")
        elif self._version != FMIL.fmi_version_3_0_enu:
            last_error = pyfmi_util.decode(FMIL.jm_get_last_error(&self.callbacks))
            if self._enable_logging:
                raise InvalidVersionException("The FMU could not be loaded. The FMU version is not supported by this class. " + last_error)
            else:
                raise InvalidVersionException("The FMU could not be loaded. The FMU version is not supported by this class. Enable logging for possibly more information.")

        # Parse xml and check fmu-kind
        self._fmu = FMIL3.fmi3_import_parse_xml(self._context, self._fmu_temp_dir, NULL)
        if self._fmu is NULL:
            last_error = pyfmi_util.decode(FMIL.jm_get_last_error(&self.callbacks))
            if self._enable_logging:
                raise InvalidXMLException("The FMU could not be loaded. The model data from 'modelDescription.xml' within the FMU could not be read. " + last_error)
            else:
                raise InvalidXMLException("The FMU could not be loaded. The model data from 'modelDescription.xml' within the FMU could not be read. Enable logging for possible more information.")

        self._fmu_kind = FMIL3.fmi3_import_get_fmu_kind(self._fmu)
        self._allocated_xml = 1

        # FMU kind is unknown
        if self._fmu_kind & FMIL3.fmi3_fmu_kind_unknown:
            last_error = pyfmi_util.decode(FMIL.jm_get_last_error(&self.callbacks))
            if self._enable_logging:
                raise InvalidVersionException("The FMU could not be loaded. The FMU kind could not be determined. " + last_error)
            else:
                raise InvalidVersionException("The FMU could not be loaded. The FMU kind could not be determined. Enable logging for possibly more information.")
        else:
            self._fmu_kind = self._get_fmu_kind()

        # Connect the DLL
        if _connect_dll:
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            status = FMIL3.fmi3_import_create_dllfmu(self._fmu, self._fmu_kind, NULL, NULL)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
            if status == FMIL.jm_status_error:
                last_error = pyfmi_util.decode(FMIL3.fmi3_import_get_last_error(self._fmu))
                if self._enable_logging:
                    raise InvalidBinaryException("The FMU could not be loaded. Error loading the binary. " + last_error)
                else:
                    raise InvalidBinaryException("The FMU could not be loaded. Error loading the binary. Enable logging for possibly more information.")
            self._allocated_dll = 1

        # Note that below, values are retrieved from XML (via FMIL) if .dll/.so is not connected
        self._modelId = self.get_identifier()

        self._modelName = pyfmi_util.decode(FMIL3.fmi3_import_get_model_name(self._fmu))

        # TODO: The code below is identical between FMUModelBase2 and FMUModelBase3, perhaps we can refactor this
        if not isinstance(log_file_name, str):
            self._set_log_stream(log_file_name)
            for i in range(len(self._log)):
                self._log_stream.write(
                    "FMIL: module = %s, log level = %d: %s\n" % (
                        self._log[i][0], self._log[i][1], self._log[i][2]
                    )
                )
        else:
            fmu_log_name = pyfmi_util.encode((self._modelId + "_log.txt") if log_file_name=="" else log_file_name)
            self._fmu_log_name = <char*>FMIL.malloc((FMIL.strlen(fmu_log_name)+1)*sizeof(char))
            FMIL.strcpy(self._fmu_log_name, fmu_log_name)

            # Create the log file
            with open(self._fmu_log_name,'w') as file:
                for i in range(len(self._log)):
                    file.write("FMIL: module = %s, log level = %d: %s\n" % (
                        self._log[i][0], self._log[i][1], self._log[i][2]
                    )
                )

        self._log = []

    def __dealloc__(self):
        """ Deallocate allocated memory. """
        self._invoked_dealloc = 1

        if self._initialized_fmu == 1:
            FMIL3.fmi3_import_terminate(self._fmu)

        if self._allocated_fmu == 1:
            FMIL3.fmi3_import_free_instance(self._fmu)

        if self._allocated_dll == 1:
            FMIL3.fmi3_import_destroy_dllfmu(self._fmu)

        if self._allocated_xml == 1:
            FMIL3.fmi3_import_free(self._fmu)

        if self._allocated_context == 1:
            FMIL.fmi_import_free_context(self._context)

        if self._fmu_temp_dir != NULL:
            if not self._allow_unzipped_fmu:
                FMIL.fmi_import_rmdir(&self.callbacks, self._fmu_temp_dir)
            FMIL.free(self._fmu_temp_dir)
            self._fmu_temp_dir = NULL

        if self._fmu_log_name != NULL:
            FMIL.free(self._fmu_log_name)
            self._fmu_log_name = NULL

        if self._log_stream:
            self._log_stream = None

    cpdef _get_time(self):
        """ Returns the current time of the simulation. """
        return self._t

    cpdef _set_time(self, FMIL3.fmi3_float64_t t):
        """ Sets the current time of the simulation.

            Parameters::
                t --
                    The time to set.
        """
        cdef int status
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_set_time(self._fmu, t)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to set the time.')
        self._t = t

    time = property(_get_time, _set_time,
        doc = """
            Property for accessing the current time of the simulation.
            Calls the low-level FMI function: fmi3SetTime or fmi3GetTime.
    """)


    def terminate(self):
        """ Calls the FMI function fmi3Terminate() on the FMU.
            After this call, any call to a function changing the state of the FMU will fail.
        """
        cdef FMIL3.fmi3_status_t status

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_terminate(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException("Termination of FMU failed, see log for possible more information.")

    def free_instance(self):
        """ Calls the FMI function fmi3FreeInstance() on the FMU. Note that this is not needed generally. """
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        FMIL3.fmi3_import_free_instance(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

    def reset(self):
        """ Resets the FMU back to its original state. Note that the environment
            has to initialize the FMU again after this function-call.
        """
        cdef FMIL3.fmi3_status_t status

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_reset(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        if status != 0:
            raise FMUException('An error occured when resetting the model, see the log for possible more information')

        #Default values
        self._t = None
        self._has_entered_init_mode = False

        #Reseting the allocation flags
        self._initialized_fmu = 0

        #Internal values
        self._log = []

    def _get_fmu_kind(self):
        raise FMUException("FMUModelBase3 cannot be used directly, use FMUModelME3.")

    def instantiate(self, name: str = 'Model', visible: bool = False) -> None:
        raise NotImplementedError

    def initialize(self,
        tolerance_defined=True,
        tolerance="Default",
        start_time="Default",
        stop_time_defined=False,
        stop_time="Default"
    ):
        raise NotImplementedError

    def _set(self, variable_name, value):
        """
        Helper method to set, see docstring on set.
        """
        cdef FMIL3.fmi3_value_reference_t ref

        ref = self.get_variable_valueref(variable_name)
        basetype: FMI3_Type = self.get_variable_data_type(variable_name)

        if basetype is FMI3_Type.FLOAT64:
            self.set_float64([ref], [value])
        elif basetype is FMI3_Type.FLOAT32:
            self.set_float32([ref], [value])
        # TODO: Add more types
        else:
            raise FMUException('Type not supported.')

    cpdef set_float64(self, valueref, values):
        """
        Sets the float64-values in the FMU as defined by the value reference(s).

        Parameters::

            valueref --
                A list of value references.

            values --
                Values to be set.

        Example::

            model.set_float64([234, 235],[2.34, 10.4])

        Calls the low-level FMI function: fmi3SetFloat64
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef np.ndarray[FMIL3.fmi3_float64_t, ndim=1, mode='c'] set_value = np.asarray(values, dtype = np.double).ravel()

        if np.size(input_valueref) != np.size(set_value):
            raise FMUException('The length of valueref and values are inconsistent. Note: Array variables are not yet supported')

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_set_float64(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            np.size(input_valueref),
            <FMIL3.fmi3_float64_t*> set_value.data,
            np.size(set_value)
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to set the Float64 values. See the log for possibly more information.')

    cpdef set_float32(self, valueref, values):
        """
        Sets the float32-values in the FMU as defined by the value reference(s).

        Parameters::

            valueref --
                A list of value references.

            values --
                Values to be set.

        Example::

            model.set_float32([234, 235],[2.34, 10.4])

        Calls the low-level FMI function: fmi3SetFloat32
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef np.ndarray[FMIL3.fmi3_float32_t, ndim=1, mode='c'] set_value = np.asarray(values, dtype = np.float32).ravel()

        if np.size(input_valueref) != np.size(set_value):
            raise FMUException('The length of valueref and values are inconsistent. Note: Array variables are not yet supported')

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_set_float32(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            np.size(input_valueref),
            <FMIL3.fmi3_float32_t*> set_value.data,
            np.size(set_value)
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to set the Float32 values. See the log for possibly more information.')

    def _get(self, variable_name):
        """
        Helper method to get, see docstring on get.
        """
        cdef FMIL3.fmi3_value_reference_t ref

        ref  = self.get_variable_valueref(variable_name)
        basetype: FMI3_Type = self.get_variable_data_type(variable_name)

        if basetype is FMI3_Type.FLOAT64:
            return self.get_float64([ref])
        elif basetype is FMI3_Type.FLOAT32:
            return self.get_float32([ref])
        # TODO: more types
        else:
            raise FMUException('Type not supported.')

    cpdef np.ndarray get_float64(self, valueref):
        """
        Returns the float64-values from the value reference(s).

        Parameters::

            valueref --
                A list of value references.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_float64([232])

        Calls the low-level FMI function: fmi3GetFloat64
        TODO: Currently does not support array variables
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef FMIL.size_t nref = np.size(input_valueref)
        cdef np.ndarray[FMIL3.fmi3_float64_t, ndim=1, mode='c'] output_value = np.zeros(nref, dtype = np.double)

        if nref == 0: # get_float64([]); do not invoke call to FMU
            return output_value

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        # TODO: Array variables; if any valueref points to an array, output length will be larger
        status = FMIL3.fmi3_import_get_float64(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            nref,
            <FMIL3.fmi3_float64_t*> output_value.data,
            nref
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to get the Float64 values.')

        return output_value

    cpdef np.ndarray get_float32(self, valueref):
        """
        Returns the float32-values from the value reference(s).

        Parameters::

            valueref --
                A list of value references.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_float32([232])

        Calls the low-level FMI function: fmi3GetFloat32
        TODO: Currently does not support array variables
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef FMIL.size_t nref = np.size(input_valueref)
        cdef np.ndarray[FMIL3.fmi3_float32_t, ndim=1, mode='c'] output_value = np.zeros(nref, dtype = np.float32)

        if nref == 0: # get_float32([]); do not invoke call to FMU
            return output_value

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        # TODO: Array variables; if any valueref points to an array, output length will be larger
        status = FMIL3.fmi3_import_get_float32(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            nref,
            <FMIL3.fmi3_float32_t*> output_value.data,
            nref
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to get the Float32 values.')

        return output_value


    cpdef np.ndarray get_int64(self, valueref):
        """
        Returns the int64-values from the value reference(s).

        Parameters::

            valueref --
                A list of value references.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_int64([232])

        Calls the low-level FMI function: fmi3GetInt64
        TODO: Currently does not support array variables
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef FMIL.size_t nref = np.size(input_valueref)
        cdef np.ndarray[FMIL3.fmi3_int64_t, ndim=1, mode='c'] output_value = np.zeros(nref, dtype = np.int64)

        if nref == 0: # get_int64([]); do not invoke call to FMU
            return output_value

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        # TODO: Array variables; if any valueref points to an array, output length will be larger
        status = FMIL3.fmi3_import_get_int64(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            nref,
            <FMIL3.fmi3_int64_t*> output_value.data,
            nref
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to get the Int64 values.')

        return output_value

    cpdef np.ndarray get_int32(self, valueref):
        """
        Returns the int32-values from the value reference(s).

        Parameters::

            valueref --
                A list of value references.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_int32([232])

        Calls the low-level FMI function: fmi3GetInt32
        TODO: Currently does not support array variables
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef FMIL.size_t nref = np.size(input_valueref)
        cdef np.ndarray[FMIL3.fmi3_int32_t, ndim=1, mode='c'] output_value = np.zeros(nref, dtype = np.int32)

        if nref == 0: # get_int32([]); do not invoke call to FMU
            return output_value

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        # TODO: Array variables; if any valueref points to an array, output length will be larger
        status = FMIL3.fmi3_import_get_int32(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            nref,
            <FMIL3.fmi3_int32_t*> output_value.data,
            nref
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to get the Int32 values.')

        return output_value


    cpdef np.ndarray get_boolean(self, valueref):
        """
        Returns the boolean-values from the value reference(s).

        Parameters::

            valueref --
                A list of value references.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_boolean([232])

        Calls the low-level FMI function: fmi3Getboolean
        TODO: Currently does not support array variables
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef FMIL.size_t nref = np.size(input_valueref)
        cdef np.ndarray[FMIL3.fmi3_boolean_t, ndim=1, mode='c'] output_value = np.zeros(nref, dtype = np.bool_)

        if nref == 0: # get_boolean([]); do not invoke call to FMU
            return output_value

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        # TODO: Array variables; if any valueref points to an array, output length will be larger
        status = FMIL3.fmi3_import_get_boolean(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            nref,
            <FMIL3.fmi3_boolean_t*> output_value.data,
            nref
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to get the Boolean values.')

        return output_value

    cpdef FMIL3.fmi3_value_reference_t get_variable_valueref(self, variable_name) except *:
        """
        Extract the value reference given a variable name.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The value reference for the variable passed as argument.
        """
        cdef FMIL3.fmi3_import_variable_t* variable
        cdef FMIL3.fmi3_value_reference_t vr
        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL3.fmi3_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%pyfmi_util.decode(variablename))
        vr =  FMIL3.fmi3_import_get_variable_vr(variable)

        return vr

    cdef FMIL3.fmi3_base_type_enu_t _get_variable_data_type(self, variable_name) except *:
        cdef FMIL3.fmi3_import_variable_t* variable
        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL3.fmi3_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException(f"The variable {pyfmi_util.decode(variablename)} could not be found.")
        return FMIL3.fmi3_import_get_variable_base_type(variable)

    def get_model_variables(self,
        type: Union[FMI3_Type, int, None] = None,
        include_alias: bool = True,
        causality: Union[FMI3_Causality, int, None] = None,
        variability: Union[FMI3_Variability, int, None] = None,
        only_start: bool = False,
        only_fixed: bool = False,
        filter = None) -> Dict[str, FMI3ModelVariable]:
        """
        Extract the names of the variables in a model.

        Parameters::

            type --
                The type of the variables as an instance of pyfmi.fmi3.FMI3_Type, int, or None.
                Default: None (i.e all).

            include_alias --
                Currently not supported (does nothing).
                If alias should be included or not.
                Default: True

            causality --
                The causality of the variables as an instance of pyfmi.fmi3.FMI3_Causality, int, or None.
                Default: None (i.e all).

            variability --
                The variability of the variables as an instance of pyfmi.fmi3.FMI3_Variability, int, or None.
                Default: None (i.e all).

            only_start --
                If only variables that has a start value should be returned.
                Default: False

            only_fixed --
                If only variables that has a start value that is fixed should be returned.
                Default: False

            filter --
                Filter the variables using a unix filename pattern
                matching (filter="*der*"). Can also be a list of filters
                See http://docs.python.org/2/library/fnmatch.html.
                Default: None

        Returns::

            Dict with variable name as key and a pyfmi.fmi3.FMI3ModelVariable class as value.
        """
        return self._get_model_variables(
            type = int(type) if type is not None else None,
            include_alias = include_alias,
            causality = int(causality) if causality is not None else None,
            variability = int(variability) if variability is not None else None,
            only_start = only_start,
            only_fixed = only_fixed,
            filter = filter
        )

    @enable_caching
    def _get_model_variables(self,
        type,
        include_alias,
        causality,
        variability,
        only_start,
        only_fixed,
        filter):
        cdef FMIL3.fmi3_import_variable_t*           variable
        cdef FMIL3.fmi3_import_variable_list_t*      variable_list
        cdef FMIL.size_t                             variable_list_size
        cdef FMIL3.fmi3_value_reference_t            value_ref
        cdef FMIL3.fmi3_base_type_enu_t              data_type
        cdef FMIL3.fmi3_variability_enu_t            data_variability
        cdef FMIL3.fmi3_causality_enu_t              data_causality
        cdef FMIL3.fmi3_initial_enu_t                data_initial

        cdef FMIL3.fmi3_import_alias_variable_list_t* alias_list
        cdef FMIL.size_t                              alias_list_size
        cdef FMIL3.fmi3_import_alias_variable_t*      alias_var

        cdef int has_start = 0
        # TODO: Can we rename the keyword arg 'filter' to variable_filter?
        variable_filter = filter
        # TODO: Can we rename the keyword arg 'type' to variable_type?
        variable_type = type
        cdef list filter_list, variable_return_list = []
        variable_dict = {}

        variable_list      = FMIL3.fmi3_import_get_variable_list(self._fmu, 0)
        variable_list_size = FMIL3.fmi3_import_get_variable_list_size(variable_list)

        if variable_filter:
            filter_list = self._convert_filter(variable_filter)

        user_specified_type        = isinstance(variable_type, int)
        user_specified_variability = isinstance(variability, int)
        user_specified_causality   = isinstance(causality, int)

        for index in range(variable_list_size):
            variable = FMIL3.fmi3_import_get_variable(variable_list, index)

            has_start = FMIL3.fmi3_import_get_variable_has_start(variable)
            if only_start and has_start == 0:
                continue

            data_variability = FMIL3.fmi3_import_get_variable_variability(variable)

            if only_fixed:
                # fixed variability requires start-value
                if has_start == 0:
                    continue
                elif data_variability != FMI3_Variability.FIXED:
                    continue

            name             = pyfmi_util.decode(FMIL3.fmi3_import_get_variable_name(variable))
            value_ref        = FMIL3.fmi3_import_get_variable_vr(variable)
            description      = self._get_variable_description(variable)
            data_causality   = FMIL3.fmi3_import_get_variable_causality(variable)
            data_initial     = FMIL3.fmi3_import_get_variable_initial(variable)
            data_type        = FMIL3.fmi3_import_get_variable_base_type(variable)
            # If only variables with start are wanted, check if the variable has start

            # TODO: Discuss if we want to support also regular integers as inputs
            if user_specified_type        and (data_type        != variable_type):
                continue
            if user_specified_variability and (data_variability != variability):
                continue
            if user_specified_causality   and (data_causality   != causality):
                continue

            if variable_filter:
                for pattern in filter_list:
                    if pattern.match(name):
                        break
                else:
                    continue

            variable_dict[name] = FMI3ModelVariable(
                name,
                value_ref,
                data_type,
                description,
                data_variability,
                data_causality,
                False, # alias
                data_initial
            )

            if include_alias:
                # TODO: Could add a convenience function "fmi3_import_get_variable_has_aliases"
                alias_list = FMIL3.fmi3_import_get_variable_alias_list(variable)
                alias_list_size = FMIL3.fmi3_import_get_alias_variable_list_size(alias_list)
                for idx in range(alias_list_size):
                    alias_var = FMIL3.fmi3_import_get_alias(alias_list, idx)
                    alias_name = pyfmi_util.decode(FMIL3.fmi3_import_get_alias_variable_name(alias_var))
                    alias_descr = self._get_alias_description(alias_var)
                    
                    variable_dict[alias_name] = FMI3ModelVariable(
                        alias_name,
                        value_ref,
                        data_type,
                        alias_descr,
                        data_variability,
                        data_causality,
                        True, # alias
                        data_initial
                    )

        FMIL3.fmi3_import_free_variable_list(variable_list)

        return variable_dict

    def get_input_list(self):
        """
        Returns a dictionary with input variables

        Returns::
            An dictionary with the (float64, continuous) input variables.
        """
        return self.get_model_variables(
            type = FMI3_Type.FLOAT64,
            include_alias = False,
            causality = FMI3_Causality.INPUT,
            variability = FMI3_Variability.CONTINUOUS)

    def get_variable_data_type(self, variable_name):
        """
        Get data type of variable.

        Parameter::
            variable_name --
                The name of the variable.

        Returns::
            The type of the variable, as an instance of pyfmi.fmi3.FMI3_Type.
        """
        variable_data_type = self._get_variable_data_type(variable_name)
        return FMI3_Type(int(variable_data_type))

    cdef _get_variable_description(self, FMIL3.fmi3_import_variable_t* variable):
        cdef FMIL3.fmi3_string_t desc = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_variable_description(variable)
        if desc == NULL:
            desc = ""
        desc = <FMIL3.fmi3_string_t>desc

        return pyfmi_util.decode(desc)

    cdef _get_alias_description(self, FMIL3.fmi3_import_alias_variable_t* alias_variable):
        cdef FMIL3.fmi3_string_t desc = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_alias_variable_description(alias_variable)
        if desc == NULL:
            desc = ""
        desc = <FMIL3.fmi3_string_t>desc

        return pyfmi_util.decode(desc)

    cpdef get_variable_description(self, variable_name):
        """
        Get the description of a given variable.

        Parameter::

            variable_name --
                The name of the variable

        Returns::

            The description of the variable.
        """
        cdef FMIL3.fmi3_import_variable_t* variable
        cdef FMIL3.fmi3_string_t desc
        cdef FMIL3.fmi3_import_alias_variable_list_t* alias_list
        cdef FMIL.size_t                              alias_list_size
        cdef FMIL3.fmi3_import_alias_variable_t*      alias_var

        variable_name_encoded = pyfmi_util.encode(variable_name)
        cdef char* var_name = variable_name_encoded

        variable = FMIL3.fmi3_import_get_variable_by_name(self._fmu, var_name)
        if variable == NULL:
            raise FMUException("The variable %s could not be found." % variable_name)
        
        base_name = pyfmi_util.decode(FMIL3.fmi3_import_get_variable_name(variable))

        # TODO: Could create a convenience function fmi3_import_get_alias_variable_by_name(variable_name)?

        if base_name == variable_name:
            return self._get_variable_description(variable)
        else:
            alias_list = FMIL3.fmi3_import_get_variable_alias_list(variable)
            alias_list_size = FMIL3.fmi3_import_get_alias_variable_list_size(alias_list)
            for idx in range(alias_list_size):
                alias_var = FMIL3.fmi3_import_get_alias(alias_list, idx)
                alias_name = pyfmi_util.decode(FMIL3.fmi3_import_get_alias_variable_name(alias_var))
                if alias_name == variable_name:
                    return self._get_alias_description(alias_var)
            raise FMUException("The variable %s could not be found." % variable_name)

    cdef _add_variable(self, FMIL3.fmi3_import_variable_t* variable):
        if variable == NULL:
            raise FMUException("Unknown variable. Please verify the correctness of the XML file and check the log.")

        name        = pyfmi_util.decode(FMIL3.fmi3_import_get_variable_name(variable))
        alias       = False
        value_ref   = FMIL3.fmi3_import_get_variable_vr(variable)
        data_type   = FMIL3.fmi3_import_get_variable_base_type(variable)
        variability = FMIL3.fmi3_import_get_variable_variability(variable)
        causality   = FMIL3.fmi3_import_get_variable_causality(variable)
        description = self._get_variable_description(variable)
        initial     = FMIL3.fmi3_import_get_variable_initial(variable)

        return FMI3ModelVariable(name, value_ref, data_type, description, variability, causality, alias, initial)

    def get_states_list(self):
        """ Returns a dictionary with the states.

            Returns::

                An ordered dictionary with the state variables.
        """
        cdef FMIL3.fmi3_import_variable_list_t* variable_list
        cdef FMIL.size_t                        variable_list_size
        variable_dict = {}

        variable_list = FMIL3.fmi3_import_get_continuous_state_derivatives_list(self._fmu)
        variable_list_size = FMIL3.fmi3_import_get_variable_list_size(variable_list)

        if variable_list == NULL:
            raise FMUException("The returned derivatives states list is NULL.")

        for i in range(variable_list_size):
            der_variable = FMIL3.fmi3_import_get_variable(variable_list, i)
            variable     = FMIL3.fmi3_import_get_float64_variable_derivative_of(<FMIL3.fmi3_import_float64_variable_t*>der_variable)

            scalar_variable = self._add_variable(<FMIL3.fmi3_import_variable_t*>variable)
            variable_dict[scalar_variable.name] = scalar_variable

        FMIL3.fmi3_import_free_variable_list(variable_list)

        return variable_dict

    def get_fmil_log_level(self):
        """ Returns::
                The current FMIL log-level.
        """
        cdef int level
        if self._enable_logging:
            level = self.callbacks.log_level
            return level
        else:
            raise FMUException('Logging is not enabled')

    def get_model_version(self):
        """ Returns the version of the FMU. """
        cdef FMIL3.fmi3_string_t version
        version = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_model_version(self._fmu)
        return pyfmi_util.decode(version) if version != NULL else ""

    def get_version(self):
        """ Returns the FMI version of the Model which it was generated according. """
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        cdef FMIL3.fmi3_string_t version = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_version(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        return pyfmi_util.decode(version)

    def get_name(self):
        """ Return the model name as used in the modeling environment. """
        return self._modelName

    def get_identifier(self):
        """ Return the model identifier, name of binary model file and prefix in the C-function names of the model. """
        raise NotImplementedError

    def get_ode_sizes(self):
        """ Returns the number of continuous states and the number of event indicators.

            Returns::

                Tuple (The number of continuous states, The number of event indicators)
                [n_states, n_event_ind] = model.get_ode_sizes()
        """
        return self._nContinuousStates, self._nEventIndicators

    def get_default_experiment_start_time(self):
        """ Returns the default experiment start time as defined in modelDescription.xml. """
        return FMIL3.fmi3_import_get_default_experiment_start(self._fmu)

    def get_default_experiment_stop_time(self):
        """ Returns the default experiment stop time as defined in modelDescription.xml. """
        return FMIL3.fmi3_import_get_default_experiment_stop(self._fmu)

    def get_default_experiment_tolerance(self):
        """ Returns the default experiment tolerance as defined in modelDescription.xml. """
        return FMIL3.fmi3_import_get_default_experiment_tolerance(self._fmu)


    def get_tolerances(self):
        """ Returns the relative and absolute tolerances. If the relative tolerance
            is defined in modelDescription.xml, it is used, otherwise a default of 1.e-4 is
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
        """ Returns the relative tolerance. If the relative tolerance
            is defined in modelDescription.xml, it is used, otherwise a default of 1.e-4 is
            used.

            Returns::

                rtol --
                    The relative tolerance.
        """
        return self.get_default_experiment_tolerance()

    def get_absolute_tolerances(self):
        """ Returns the absolute tolerances. They are calculated and returned according to
            the FMI specification, atol = 0.01*rtol*(nominal values of the
            continuous states)

            This method should not be called before initialization, since it depends on state nominals.

            Returns::

                atol --
                    The absolute tolerances.
        """
        if self._initialized_fmu == 0:
            raise FMUException("Unable to retrieve the absolute tolerance, FMU needs to be initialized.")

        rtol = self.get_relative_tolerance()
        return 0.01*rtol*self.nominal_continuous_states

    cpdef get_variable_unbounded(self, variable_name):
        """TODO:"""
        return False

    def get_generation_tool(self):
        """ Return the model generation tool. """
        cdef FMIL3.fmi3_string_t gen = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_generation_tool(self._fmu)
        return pyfmi_util.decode(gen) if gen != NULL else ""

    def get_variable_alias_base(self, variable_name):
        """
        Returns the base variable for the provided variable name.

        Parameters::

            variable_name--
                Name of the variable.

        Returns:

           The base variable.
        """
        cdef FMIL3.fmi3_import_variable_t* variable
        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL3.fmi3_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        name = pyfmi_util.decode(FMIL3.fmi3_import_get_variable_name(variable))

        return name

    def get_variable_alias(self, variable_name):
        """
        Return a dict of all alias variables belonging to the provided variable
        where the key are the names and the value indicates if the variable
        is an alias or not.

        Parameters::

            variable_name--
                Name of the variable to find alias of.

        Returns::

            A dict consisting of the alias variables along with no alias variable.
            The values indicate whether or not the variable is an alias or not.

        Raises::

            FMUException if the variable is not in the model.
        """
        cdef FMIL3.fmi3_import_variable_t*            variable
        cdef FMIL3.fmi3_import_alias_variable_list_t* alias_list
        cdef FMIL.size_t                              alias_list_size
        cdef FMIL3.fmi3_import_alias_variable_t*      alias_var
        cdef dict                                     ret_values = {}

        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL3.fmi3_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%variablename)

        base_name = pyfmi_util.decode(FMIL3.fmi3_import_get_variable_name(variable))
        ret_values[base_name] = False

        alias_list = FMIL3.fmi3_import_get_variable_alias_list(variable)
        alias_list_size = FMIL3.fmi3_import_get_alias_variable_list_size(alias_list)
        for idx in range(alias_list_size):
            alias_var = FMIL3.fmi3_import_get_alias(alias_list, idx)
            alias_name = pyfmi_util.decode(FMIL3.fmi3_import_get_alias_variable_name(alias_var))
            ret_values[alias_name] = True

        return ret_values

cdef class FMUModelME3(FMUModelBase3):
    """
    FMI3 ModelExchange model loaded from a dll
    """

    def __init__(self, fmu, log_file_name = "", log_level = FMI_DEFAULT_LOG_LEVEL,
                 _unzipped_dir = None, _connect_dll = True, allow_unzipped_fmu = False):
        """
        Constructor of the model.

        Parameters::

            fmu --
                Name of the fmu as a string.

            log_file_name --
                Filename for file used to save log messages.
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

            A model as an object from the class FMUModelME3
        """
        # Call super on base class
        FMUModelBase3.__init__(self, fmu, log_file_name, log_level,
                               _unzipped_dir, _connect_dll, allow_unzipped_fmu)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_get_number_of_event_indicators(self._fmu, &self._nEventIndicators)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        if status != FMIL3.fmi3_status_ok:
            raise InvalidFMUException("The FMU could not be instantiated, error retrieving number of event indicators.")

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_get_number_of_continuous_states(self._fmu, &self._nContinuousStates)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        if status != FMIL3.fmi3_status_ok:
            raise InvalidFMUException("The FMU could not be instantiated, error retrieving number of continuous states.")

        if _connect_dll:
            self.instantiate()

    def _get_fmu_kind(self):
        if self._fmu_kind & FMIL3.fmi3_fmu_kind_me:
            return FMIL3.fmi3_fmu_kind_me
        else:
            raise InvalidVersionException('The FMU could not be loaded. This class only supports FMI 3.0 for Model Exchange.')

    def get_identifier(self):
        if not self._modelId:
            self._modelId = pyfmi_util.decode(FMIL3.fmi3_import_get_model_identifier_ME(self._fmu))

        return self._modelId

    def instantiate(self, name: str = 'Model', visible: bool = False) -> None:
        """
        Instantiate the model.

        Parameters::

            name --
                The name of the instance.
                Default: 'Model'

            visible --
                Defines if the simulator application window should be visible or not.
                Default: False, not visible.

        Calls the respective low-level FMI function: fmi3InstantiateModelExchange.
        """

        cdef FMIL3.fmi3_boolean_t  log
        cdef FMIL3.fmi3_boolean_t  vis
        cdef FMIL.jm_status_enu_t status

        log = self._enable_logging
        vis = visible

        name_as_bytes = pyfmi_util.encode(name)
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_instantiate_model_exchange(
            self._fmu,
            name_as_bytes,
            NULL,
            vis,
            log
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != FMIL.jm_status_success:
            raise FMUException('Failed to instantiate the model. See the log for possibly more information.')

        self._allocated_fmu = 1


    def initialize(
        self,
        tolerance_defined=True,
        tolerance="Default",
        start_time="Default",
        stop_time_defined=False,
        stop_time="Default"
    ):
        """
        Initializes the model and computes initial values for all variables.

        Args:
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

        Calls the low-level FMI functions: fmi3EnterInitializationMode,
                                           fmi3ExitInitializationMode
        """
        log_open = self._log_open()
        if not log_open and self.get_log_level() > 2:
            self._open_log_file()

        try:
            self.enter_initialization_mode(
                tolerance_defined,
                tolerance,
                start_time,
                stop_time_defined,
                stop_time
            )
            self.exit_initialization_mode()
        except Exception:
            if not log_open and self.get_log_level() > 2:
                self._close_log_file()

            raise

        if not log_open and self.get_log_level() > 2:
            self._close_log_file()

        self._initialized_fmu = 1

    def enter_initialization_mode(
        self,
        tolerance_defined=True,
        tolerance="Default",
        start_time="Default",
        stop_time_defined=False,
        stop_time="Default"
    ):
        """ Enters initialization mode by calling the low level FMI function fmi3EnterInitializationMode.
            Note that the method initialize() performs both the enter and exit of initialization mode.

            Args:
                For a full description of the input arguments, see the docstring for method 'initialize'.
        """
        cdef FMIL3.fmi3_status_t status

        cdef FMIL3.fmi3_boolean_t stop_defined = FMIL3.fmi3_true if stop_time_defined else FMIL3.fmi3_false
        cdef FMIL3.fmi3_boolean_t tol_defined = FMIL3.fmi3_true if tolerance_defined else FMIL3.fmi3_false

        if tolerance == "Default":
            tolerance = self.get_default_experiment_tolerance()
        if start_time == "Default":
            start_time = self.get_default_experiment_start_time()
        if stop_time == "Default":
            stop_time = self.get_default_experiment_stop_time()

        self._t = start_time
        self._last_accepted_time = start_time
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_enter_initialization_mode(
            self._fmu,
            tolerance_defined,
            tolerance,
            start_time,
            stop_time_defined,
            stop_time
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != FMIL3.fmi3_status_ok:
            raise FMUException("Failed to enter initialization mode")

        self._has_entered_init_mode = True

    def exit_initialization_mode(self):
        """
            Exit initialization mode by calling the low level FMI function
            fmi3ExitInitializationMode.

            Note that the method initialize() performs both the enter and
            exit of initialization mode.
        """
        cdef FMIL3.fmi3_status_t status
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_exit_initialization_mode(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != FMIL3.fmi3_status_ok:
            raise FMUException("Failed to exit initialization mode")

    def enter_continuous_time_mode(self):
        """ Enter continuous time mode by calling the low level FMI function fmi3EnterContinuousTimeMode. """
        cdef FMIL3.fmi3_status_t status
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_enter_continuous_time_mode(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != FMIL3.fmi3_status_ok:
            raise FMUException("Failed to enter continuous time mode")

    def get_event_info(self):
        """
        Returns the event information from the FMU.

        Returns::
            The event information as an instance of pyfmi.fmi3.FMI3EventInfo
        """
        # TODO: Below is temporary for testing until we've added support for events
        self._event_info.next_event_time_defined = FMIL3.fmi3_true
        return self._event_info


    def enter_event_mode(self):
        """ Enter event mode by calling the low level FMI function fmi3EnterEventMode. """
        cdef FMIL3.fmi3_status_t status
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_enter_event_mode(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != FMIL3.fmi3_status_ok:
            raise FMUException("Failed to enter event mode")


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

        Calls the low-level FMI function: TODO
        """
        pass


    cdef FMIL3.fmi3_status_t _get_nominal_continuous_states_fmil(self, FMIL3.fmi3_float64_t* xnominal, size_t nx):
        cdef FMIL3.fmi3_status_t status
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_get_nominals_of_continuous_states(self._fmu, xnominal, nx)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        return status

    def _get_nominal_continuous_states(self):
        """
        Returns the nominal values of the continuous states.

        Returns::
            The nominal values.
        """
        cdef FMIL3.fmi3_status_t status
        cdef np.ndarray[FMIL3.fmi3_float64_t, ndim=1, mode='c'] xn = np.zeros(self._nContinuousStates, dtype=np.double)

        if self._initialized_fmu == 0:
            raise FMUException("Unable to retrieve nominals of continuous states, FMU must first be initialized.")

        status = self._get_nominal_continuous_states_fmil(<FMIL3.fmi3_float64_t*> xn.data, self._nContinuousStates)
        if status != 0:
            raise FMUException('Failed to get the nominal values.')

        # Fallback - auto-correct the illegal nominal values:
        xnames = list(self.get_states_list().keys())
        for i in range(self._nContinuousStates):
            if xn[i] == 0.0:
                if self.callbacks.log_level >= FMIL.jm_log_level_warning:
                    logging.warning(f"The nominal value for {xnames[i]} is 0.0 which is illegal according " + \
                                     "to the FMI specification. Setting the nominal to 1.0.")
                xn[i] = 1.0
            elif xn[i] < 0.0:
                if self.callbacks.log_level >= FMIL.jm_log_level_warning:
                    logging.warning(f"The nominal value for {xnames[i]} is <0.0 which is illegal according " + \
                                    f"to the FMI specification. Setting the nominal to abs({xn[i]}).")
                xn[i] = abs(xn[i])

        return xn

    nominal_continuous_states = property(_get_nominal_continuous_states, doc =
    """
    Property for accessing the nominal values of the continuous states. Calls
    the low-level FMI function: fmi3GetNominalContinuousStates.
    """)

    cdef FMIL3.fmi3_status_t _get_derivatives(self, FMIL3.fmi3_float64_t[:] values):
        cdef FMIL3.fmi3_status_t status
        if self._nContinuousStates > 0:
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            status = FMIL3.fmi3_import_get_derivatives(self._fmu, &values[0], self._nContinuousStates)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
            return status
        else:
            return FMIL3.fmi3_status_ok

    cpdef get_derivatives(self):
        """
        Returns the derivative of the continuous states.

        Returns::

            dx --
                The derivatives as an array.

        Example::

            dx = model.get_derivatives()

        Calls the low-level FMI function: fmi3GetDerivatives
        """
        cdef FMIL3.fmi3_status_t status
        cdef np.ndarray[FMIL3.fmi3_float64_t, ndim=1, mode='c'] values = np.empty(
            self._nContinuousStates,
            dtype = np.double
        )

        status = self._get_derivatives(values)

        if status != FMIL3.fmi3_status_ok:
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

    cdef FMIL3.fmi3_status_t _completed_integrator_step(self,
            FMIL3.fmi3_boolean_t no_set_FMU_state_prior_to_current_point,
            FMIL3.fmi3_boolean_t* enter_event_mode,
            FMIL3.fmi3_boolean_t* terminate_simulation
        ):
        cdef FMIL3.fmi3_status_t status

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_completed_integrator_step(
            self._fmu,
            no_set_FMU_state_prior_to_current_point,
            enter_event_mode,
            terminate_simulation
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        self._last_accepted_time = self._get_time()
        return status

    def completed_integrator_step(self, no_set_FMU_state_prior_to_current_point = True):
        """
        This method must be called by the environment after every completed step
        of the integrator. If returned value is True, then the environment must call
        event_update() otherwise, no action is needed.

        Returns::
            A tuple of format (a, b) where a and b indicate:
                If a is True -> Call event_update().
                        False -> Do nothing.
                If b is True -> The simulation should be terminated.
                        False -> Do nothing.

        Calls the low-level FMI function: fmi3CompletedIntegratorStep.
        """
        cdef FMIL3.fmi3_status_t status
        cdef FMIL3.fmi3_boolean_t noSetFMUStatePriorToCurrentPoint = FMIL3.fmi3_true if no_set_FMU_state_prior_to_current_point else FMIL3.fmi3_false
        cdef FMIL3.fmi3_boolean_t enterEventMode = FMIL3.fmi3_false
        cdef FMIL3.fmi3_boolean_t terminateSimulation = FMIL3.fmi3_false

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_completed_integrator_step(
            self._fmu,
            noSetFMUStatePriorToCurrentPoint,
            &enterEventMode,
            &terminateSimulation
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != FMIL3.fmi3_status_ok:
            raise FMUException('Failed to call FMI completed integrator step at time: %E.' % self.time)

        self._last_accepted_time = self._get_time()

        return enterEventMode == FMIL3.fmi3_true, terminateSimulation == FMIL3.fmi3_true

    cdef FMIL3.fmi3_status_t _get_continuous_states_fmil(self, FMIL3.fmi3_float64_t[:] ndx):
        cdef FMIL3.fmi3_status_t status
        if self._nContinuousStates > 0:
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            status = FMIL3.fmi3_import_get_continuous_states(self._fmu, &ndx[0] ,self._nContinuousStates)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
            return status
        else:
            return FMIL3.fmi3_status_ok

    def _get_continuous_states(self):
        """ Returns a vector with the values of the continuous states.

            Returns::
                The continuous states.
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_float64_t, ndim=1, mode='c'] ndx = np.zeros(
            self._nContinuousStates,
            dtype = np.double
        )

        status = self._get_continuous_states_fmil(ndx)

        if status != 0:
            raise FMUException('Failed to retrieve the continuous states.')

        return ndx

    cdef FMIL3.fmi3_status_t _set_continuous_states_fmil(self, FMIL3.fmi3_float64_t[:] ndx):
        cdef FMIL3.fmi3_status_t status
        if self._nContinuousStates > 0:
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            status = FMIL3.fmi3_import_set_continuous_states(self._fmu, &ndx[0], self._nContinuousStates)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
            return status
        else:
            return FMIL3.fmi3_status_ok

    def _set_continuous_states(self, np.ndarray[FMIL3.fmi3_float64_t, ndim=1, mode="c"] values):
        """ Set the values of the continuous states.

            Parameters::
                values--
                    The new values of the continuous states.
        """
        cdef FMIL3.fmi3_status_t status
        cdef np.ndarray[FMIL3.fmi3_float64_t, ndim=1,mode='c'] ndx = values

        if np.size(ndx) != self._nContinuousStates:
            raise FMUException(
                'Failed to set the new continuous states. ' \
                'The number of values are not consistent with the number of '\
                f'continuous states, which are {self._nContinuousStates}.')

        status = self._set_continuous_states_fmil(ndx)

        if status >= FMIL3.fmi3_status_error:
            raise FMUException('Failed to set the new continuous states.')

    continuous_states = property(_get_continuous_states, _set_continuous_states,
        doc = """
            Property for accessing the current values of the continuous states. Calls
            the low-level FMI function: fmi3SetContinuousStates/fmi3GetContinuousStates.
    """)

cdef void _cleanup_on_load_error(
    FMIL3.fmi3_import_t* fmu_3,
    FMIL.fmi_import_context_t* context,
    int allow_unzipped_fmu,
    FMIL.jm_callbacks callbacks,
    bytes fmu_temp_dir,
    list log_data
):
    """ To reduce some code duplication for various failures in _load_fmi3_fmu. """
    if fmu_3 is not NULL:
        FMIL3.fmi3_import_free(fmu_3)
    FMIL.fmi_import_free_context(context)
    if not allow_unzipped_fmu:
        FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
    _handle_load_fmu_exception(log_data)

cdef object _load_fmi3_fmu(
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
    The FMI3 part of fmi.pyx load_fmu.
    """
    cdef FMIL.jm_string last_error
    cdef FMIL3.fmi3_import_t* fmu_3 = NULL
    cdef FMIL3.fmi3_fmu_kind_enu_t fmu_3_kind
    model = None

    # Check fmu-kind and compare with input-specified kind
    fmu_3 = FMIL3.fmi3_import_parse_xml(context, fmu_temp_dir, NULL)

    if fmu_3 is NULL:
        # Delete the context
        _cleanup_on_load_error(fmu_3, context, allow_unzipped_fmu,
                               callbacks, fmu_temp_dir, log_data)
        if callbacks.log_level >= FMIL.jm_log_level_error:
            last_error = FMIL.jm_get_last_error(&callbacks)
            raise InvalidXMLException("The FMU could not be loaded. The model data from 'modelDescription.xml' within the FMU could not be read. " + pyfmi_util.decode(last_error))
        else:
            raise InvalidXMLException("The FMU could not be loaded. The model data from 'modelDescription.xml' within the FMU could not be read. Enable logging for possible more information.")

    fmu_3_kind = FMIL3.fmi3_import_get_fmu_kind(fmu_3)

    # FMU kind is unknown
    if fmu_3_kind & FMIL3.fmi3_fmu_kind_unknown:
        _cleanup_on_load_error(fmu_3, context, allow_unzipped_fmu,
                               callbacks, fmu_temp_dir, log_data)
        if callbacks.log_level >= FMIL.jm_log_level_error:
            last_error = FMIL.jm_get_last_error(&callbacks)
            raise FMUException("The FMU kind could not be determined. " + pyfmi_util.decode(last_error))
        else:
            raise FMUException("The FMU kind could not be determined. Enable logging for possibly more information.")

    # FMU kind is known
    if kind.lower() == "auto":
        if fmu_3_kind & FMIL3.fmi3_fmu_kind_me:
            model = FMUModelME3(fmu, log_file_name, log_level, _unzipped_dir = fmu_temp_dir,
                                allow_unzipped_fmu = allow_unzipped_fmu)
        elif fmu_3_kind & FMIL3.fmi3_fmu_kind_cs:
            raise InvalidFMUException("Import of FMI3 Co-Simulation FMUs is not yet supported.")
        elif fmu_3_kind & FMIL3.fmi3_fmu_kind_se:
            raise InvalidFMUException("Import of FMI3 Scheduled Execution FMUs is not supported.")
    elif (kind.upper() == 'ME') and (fmu_3_kind & FMIL3.fmi3_fmu_kind_me):
        model = FMUModelME3(fmu, log_file_name, log_level, _unzipped_dir = fmu_temp_dir,
                            allow_unzipped_fmu = allow_unzipped_fmu)
    elif (kind.upper() == 'CS') and (fmu_3_kind & FMIL3.fmi3_fmu_kind_cs):
        raise InvalidFMUException("Import of FMI3 Co-Simulation FMUs is not yet supported.")
    elif (kind.upper() == 'SE') and (fmu_3_kind & FMIL3.fmi3_fmu_kind_se):
        raise InvalidFMUException("Import of FMI3 Scheduled Execution FMUs is not supported.")

    # Could not match FMU kind with input-specified kind
    if model is None:
        _cleanup_on_load_error(fmu_3, context, allow_unzipped_fmu,
                               callbacks, fmu_temp_dir, log_data)

        # TODO, from FMIL we get without blank spaces, i.e. we get 'ModelExchange' and not 'Model Exchange'
        # when we invoke fmi3_fmu_kind_to_string, perhaps we should format accordingly here?
        kind_name = kind.upper()
        if kind.upper() == 'SE':
            kind_name = 'ScheduledExecution'
        elif kind.upper() == 'CS':
            kind_name = 'CoSimulation'
        elif kind.upper() == 'ME':
            kind_name = 'ModelExchange'
        raise FMUException("FMU is a {} and not a {}".format(
            pyfmi_util.decode(FMIL3.fmi3_fmu_kind_to_string(fmu_3_kind)),
            pyfmi_util.decode(kind_name)
        ))

    FMIL3.fmi3_import_free(fmu_3)
    FMIL.fmi_import_free_context(context)
    return model
