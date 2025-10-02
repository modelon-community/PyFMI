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
import functools
from typing import Union

import numpy as np
cimport numpy as np
from numpy cimport PyArray_DATA

cimport pyfmi.fmil_import as FMIL
cimport pyfmi.fmil3_import as FMIL3
cimport pyfmi.fmi_base as FMI_BASE
cimport pyfmi.util as pyfmi_util
from pyfmi.util import enable_caching

import scipy.sparse as sps


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

class FMI3_DependencyKind(IntEnum):
    DEPENDENT = FMIL3.fmi3_dependencies_kind_dependent
    CONSTANT  = FMIL3.fmi3_dependencies_kind_constant
    FIXED     = FMIL3.fmi3_dependencies_kind_fixed
    TUNABLE   = FMIL3.fmi3_dependencies_kind_tunable
    DISCRETE  = FMIL3.fmi3_dependencies_kind_discrete

# Jacobian approximation
DEF FORWARD_DIFFERENCE = 1
DEF CENTRAL_DIFFERENCE = 2
FORWARD_DIFFERENCE_EPS = (np.finfo(float).eps)**0.5
CENTRAL_DIFFERENCE_EPS = (np.finfo(float).eps)**(1/3.0)

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
        self.newDiscreteStatesNeeded           = FMIL3.fmi3_false
        self.terminateSimulation               = FMIL3.fmi3_false
        self.nominalsOfContinuousStatesChanged = FMIL3.fmi3_false
        self.valuesOfContinuousStatesChanged   = FMIL3.fmi3_false
        self.nextEventTimeDefined              = FMIL3.fmi3_false
        self.nextEventTime                     = 0.0

cdef inline void _check_input_sizes(np.ndarray input_valueref, np.ndarray set_value):
    if np.size(input_valueref) != np.size(set_value):
        raise FMUException('The length of valueref and values are inconsistent. Note: Array variables are not yet supported')

cdef inline FMIL3.fmi3_import_variable_t* _get_variable_by_name(FMIL3.fmi3_import_t* fmu, variable_name):
    cdef FMIL3.fmi3_import_variable_t* variable
    variable_name = pyfmi_util.encode(variable_name)
    cdef char* variablename = variable_name

    variable = FMIL3.fmi3_import_get_variable_by_name(fmu, variablename)
    if variable == NULL:
        raise FMUException(f"The variable {pyfmi_util.decode(variablename)} could not be found.")
    return variable


cdef class FMUState3:
    """ Class containing a pointer to a FMU-state. """
    def __init__(self):
        self.fmu_state = NULL
        self._internal_state_variables = {'initialized_fmu': None,
                                          'has_entered_init_mode': None,
                                          'time': None,
                                          'callback_log_level': None,
                                          'event_info.new_discrete_states_needed': None,
                                          'event_info.nominals_of_continuous_states_changed': None,
                                          'event_info.terminate_simulation': None,
                                          'event_info.values_of_continuous_states_changed': None,
                                          'event_info.next_event_time_defined': None,
                                          'event_info.next_event_time': None}


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

        # Caching
        self._states_references = None
        self._derivatives_references = None
        self._inputs_references = None
        self._outputs_references = None
        self._outputs_states_dependencies = None
        self._outputs_inputs_dependencies = None
        self._outputs_states_dependencies_kind = None
        self._outputs_inputs_dependencies_kind = None
        self._derivatives_states_dependencies = None
        self._derivatives_inputs_dependencies = None
        self._derivatives_states_dependencies_kind = None
        self._derivatives_inputs_dependencies_kind = None
        self._group_A = None
        self._group_B = None
        self._group_C = None
        self._group_D = None

        # Internal values
        self._enable_logging = False
        self._eventInfo   = FMI3EventInfo()
        self._worker_object = _WorkerClass3()

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
            status = FMIL3.fmi3_import_create_dllfmu(self._fmu, self._fmu_kind, <FMIL3.fmi3_instance_environment_t>self._fmu, FMIL3.fmi3_log_forwarding)
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

        # Default values
        self._t = None
        self._has_entered_init_mode = False

        # Reseting the allocation flags
        self._initialized_fmu = 0

        # Internal values
        self._eventInfo = FMI3EventInfo()
        self._log = []

    def _get_fmu_kind(self):
        raise FMUException("FMUModelBase3 cannot be used directly, use FMUModelME3.")

    def instantiate(self, name: str = 'Model', visible: bool = False) -> None:
        raise NotImplementedError # to implemented in FMUModel(ME|CS|SE)3

    def initialize(self,
        tolerance_defined=True,
        tolerance="Default",
        start_time="Default",
        stop_time_defined=False,
        stop_time="Default"
    ):
        raise NotImplementedError # to implemented in FMUModel(ME|CS|SE)3

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
        elif basetype is FMI3_Type.INT64:
            self.set_int64([ref], [value])
        elif basetype is FMI3_Type.INT32:
            self.set_int32([ref], [value])
        elif basetype is FMI3_Type.INT16:
            self.set_int16([ref], [value])
        elif basetype is FMI3_Type.INT8:
            self.set_int8([ref], [value])
        elif basetype is FMI3_Type.UINT64:
            self.set_uint64([ref], [value])
        elif basetype is FMI3_Type.UINT32:
            self.set_uint32([ref], [value])
        elif basetype is FMI3_Type.UINT16:
            self.set_uint16([ref], [value])
        elif basetype is FMI3_Type.UINT8:
            self.set_uint8([ref], [value])
        elif basetype is FMI3_Type.BOOL:
            self.set_boolean([ref], [value])
        elif basetype is FMI3_Type.STRING:
            self.set_string([ref], [value])
        elif basetype is FMI3_Type.ENUM:
            self.set_enum([ref], [value])
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
        _check_input_sizes(input_valueref, set_value)

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

    cdef FMIL3.fmi3_status_t _set_float64(self, FMIL3.fmi3_value_reference_t* vrefs, FMIL3.fmi3_float64_t* values, size_t _size):
        """Internal method for efficient setting of float64 variables."""
        cdef FMIL3.fmi3_status_t status
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        # XXX: arrays?
        status = FMIL3.fmi3_import_set_float64(self._fmu, vrefs, _size, values, _size)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        return status

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
        _check_input_sizes(input_valueref, set_value)

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

    cpdef set_int64(self, valueref, values):
        """
        Sets the int64-values in the FMU as defined by the value reference(s).

        Parameters::

            valueref --
                A list of value references.

            values --
                Values to be set.

        Example::

            model.set_int64([234, 235],[2, 10])

        Calls the low-level FMI function: fmi3SetInt64
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef np.ndarray[FMIL3.fmi3_int64_t, ndim=1, mode='c'] set_value = np.asarray(values, dtype = np.int64).ravel()
        _check_input_sizes(input_valueref, set_value)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_set_int64(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            np.size(input_valueref),
            <FMIL3.fmi3_int64_t*> set_value.data,
            np.size(set_value)
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to set the Int64 values. See the log for possibly more information.')

    cpdef set_int32(self, valueref, values):
        """
        Sets the int32-values in the FMU as defined by the value reference(s).

        Parameters::

            valueref --
                A list of value references.

            values --
                Values to be set.

        Example::

            model.set_int32([234, 235],[2, 10])

        Calls the low-level FMI function: fmi3SetInt32
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef np.ndarray[FMIL3.fmi3_int32_t, ndim=1, mode='c'] set_value = np.asarray(values, dtype = np.int32).ravel()
        _check_input_sizes(input_valueref, set_value)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_set_int32(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            np.size(input_valueref),
            <FMIL3.fmi3_int32_t*> set_value.data,
            np.size(set_value)
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to set the Int32 values. See the log for possibly more information.')

    cpdef set_int16(self, valueref, values):
        """
        Sets the int16-values in the FMU as defined by the value reference(s).

        Parameters::

            valueref --
                A list of value references.

            values --
                Values to be set.

        Example::

            model.set_int16([234, 235],[2, 10])

        Calls the low-level FMI function: fmi3SetInt16
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef np.ndarray[FMIL3.fmi3_int16_t, ndim=1, mode='c'] set_value = np.asarray(values, dtype = np.int16).ravel()
        _check_input_sizes(input_valueref, set_value)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_set_int16(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            np.size(input_valueref),
            <FMIL3.fmi3_int16_t*> set_value.data,
            np.size(set_value)
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to set the Int16 values. See the log for possibly more information.')

    cpdef set_int8(self, valueref, values):
        """
        Sets the int8-values in the FMU as defined by the value reference(s).

        Parameters::

            valueref --
                A list of value references.

            values --
                Values to be set.

        Example::

            model.set_int8([234, 235],[2, 10])

        Calls the low-level FMI function: fmi3SetInt8
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef np.ndarray[FMIL3.fmi3_int8_t, ndim=1, mode='c'] set_value = np.asarray(values, dtype = np.int8).ravel()
        _check_input_sizes(input_valueref, set_value)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_set_int8(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            np.size(input_valueref),
            <FMIL3.fmi3_int8_t*> set_value.data,
            np.size(set_value)
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to set the Int8 values. See the log for possibly more information.')

    cpdef set_uint64(self, valueref, values):
        """
        Sets the uint64-values in the FMU as defined by the value reference(s).

        Parameters::

            valueref --
                A list of value references.

            values --
                Values to be set.

        Example::

            model.set_uint64([234, 235],[2, 10])

        Calls the low-level FMI function: fmi3SetUInt64
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef np.ndarray[FMIL3.fmi3_uint64_t, ndim=1, mode='c'] set_value = np.asarray(values, dtype = np.uint64).ravel()
        _check_input_sizes(input_valueref, set_value)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_set_uint64(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            np.size(input_valueref),
            <FMIL3.fmi3_uint64_t*> set_value.data,
            np.size(set_value)
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to set the UInt64 values. See the log for possibly more information.')

    cpdef set_uint32(self, valueref, values):
        """
        Sets the uint32-values in the FMU as defined by the value reference(s).

        Parameters::

            valueref --
                A list of value references.

            values --
                Values to be set.

        Example::

            model.set_uint32([234, 235],[2, 10])

        Calls the low-level FMI function: fmi3SetUInt32
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef np.ndarray[FMIL3.fmi3_uint32_t, ndim=1, mode='c'] set_value = np.asarray(values, dtype = np.uint32).ravel()
        _check_input_sizes(input_valueref, set_value)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_set_uint32(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            np.size(input_valueref),
            <FMIL3.fmi3_uint32_t*> set_value.data,
            np.size(set_value)
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to set the UInt32 values. See the log for possibly more information.')

    cpdef set_uint16(self, valueref, values):
        """
        Sets the uint16-values in the FMU as defined by the value reference(s).

        Parameters::

            valueref --
                A list of value references.

            values --
                Values to be set.

        Example::

            model.set_uint16([234, 235],[2, 10])

        Calls the low-level FMI function: fmi3SetUInt16
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef np.ndarray[FMIL3.fmi3_uint16_t, ndim=1, mode='c'] set_value = np.asarray(values, dtype = np.uint16).ravel()
        _check_input_sizes(input_valueref, set_value)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_set_uint16(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            np.size(input_valueref),
            <FMIL3.fmi3_uint16_t*> set_value.data,
            np.size(set_value)
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to set the UInt16 values. See the log for possibly more information.')

    cpdef set_uint8(self, valueref, values):
        """
        Sets the uint8-values in the FMU as defined by the value reference(s).

        Parameters::

            valueref --
                A list of value references.

            values --
                Values to be set.

        Example::

            model.set_uint8([234, 235],[2, 10])

        Calls the low-level FMI function: fmi3SetUInt8
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef np.ndarray[FMIL3.fmi3_uint8_t, ndim=1, mode='c'] set_value = np.asarray(values, dtype = np.uint8).ravel()
        _check_input_sizes(input_valueref, set_value)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_set_uint8(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            np.size(input_valueref),
            <FMIL3.fmi3_uint8_t*> set_value.data,
            np.size(set_value)
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to set the UInt8 values. See the log for possibly more information.')

    cpdef set_boolean(self, valueref, values):
        """
        Sets the boolean-values in the FMU as defined by the value reference(s).

        Parameters::

            valueref --
                A list of value references.

            values --
                Values to be set.

        Example::

            model.set_boolean([234, 235],[True, False])

        Calls the low-level FMI function: fmi3SetBoolean
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef np.ndarray[FMIL3.fmi3_boolean_t, ndim=1, mode='c'] set_value = np.asarray(values, dtype = np.bool_).ravel()
        _check_input_sizes(input_valueref, set_value)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_set_boolean(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            np.size(input_valueref),
            <FMIL3.fmi3_boolean_t*> set_value.data,
            np.size(set_value)
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to set the Boolean values. See the log for possibly more information.')

    cpdef set_string(self, valueref, values):
        """
        Sets the string-values in the FMU as defined by the value reference(s).

        Parameters::

            valueref --
                A list of value references.

            values --
                Values to be set.

        Example::

            model.set_string([234, 235],["hello", "fmu"])

        Calls the low-level FMI function: fmi3SetString
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef FMIL3.fmi3_string_t* val = <FMIL3.fmi3_string_t*>FMIL.malloc(sizeof(FMIL3.fmi3_string_t)*np.size(values))

        if np.size(input_valueref) != np.size(values):
            raise FMUException('The length of valueref and values are inconsistent. Note: Array variables are not yet supported')

        values = [pyfmi_util.encode(item) for item in values]
        for i in range(np.size(input_valueref)):
            val[i] = values[i]

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_set_string(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            np.size(input_valueref),
            val,
            np.size(values)
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        FMIL.free(val)

        if status != 0:
            raise FMUException('Failed to set the String values. See the log for possibly more information.')

    cpdef set_enum(self, valueref, values):
        """
        Sets the enum-values in the FMU as defined by the value reference(s).

        Parameters::

            valueref --
                A list of value references.

            values --
                Values to be set.

        Example::

            model.set_enum([234, 235],[2, 10])

        Calls the low-level FMI function: fmi3SetInt64
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef np.ndarray[FMIL3.fmi3_int64_t, ndim=1, mode='c'] set_value = np.asarray(values, dtype = np.int64).ravel()
        _check_input_sizes(input_valueref, set_value)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_set_int64(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            np.size(input_valueref),
            <FMIL3.fmi3_int64_t*> set_value.data,
            np.size(set_value)
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to set the Enum values. See the log for possibly more information.')

    def _get(self, variable_name: str):
        """
        Helper method to get, see docstring on get.
        """
        cdef FMIL3.fmi3_value_reference_t ref = self.get_variable_valueref(variable_name)
        basetype: FMI3_Type = self.get_variable_data_type(variable_name)

        if basetype is FMI3_Type.FLOAT64:
            return self.get_float64([ref])
        elif basetype is FMI3_Type.FLOAT32:
            return self.get_float32([ref])
        elif basetype is FMI3_Type.INT64:
            return self.get_int64([ref])
        elif basetype is FMI3_Type.INT32:
            return self.get_int32([ref])
        elif basetype is FMI3_Type.INT16:
            return self.get_int16([ref])
        elif basetype is FMI3_Type.INT8:
            return self.get_int8([ref])
        elif basetype is FMI3_Type.UINT64:
            return self.get_uint64([ref])
        elif basetype is FMI3_Type.UINT32:
            return self.get_uint32([ref])
        elif basetype is FMI3_Type.UINT16:
            return self.get_uint16([ref])
        elif basetype is FMI3_Type.UINT8:
            return self.get_uint8([ref])
        elif basetype is FMI3_Type.BOOL:
            return self.get_boolean([ref])
        elif basetype is FMI3_Type.STRING:
            return self.get_string([ref])
        elif basetype is FMI3_Type.ENUM:
            return self.get_enum([ref])
        else:
            raise FMUException(f"Type '{basetype}' is not supported.")

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

    cdef FMIL3.fmi3_status_t _get_float64(self, FMIL3.fmi3_value_reference_t* vrefs, size_t _size, FMIL3.fmi3_float64_t* values):
        """Internal method for efficient getting of float64 variables."""
        cdef FMIL3.fmi3_status_t status
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        # XXX: arrays
        status = FMIL3.fmi3_import_get_float64(self._fmu, vrefs, _size, values, _size)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        return status

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

    cpdef np.ndarray get_int16(self, valueref):
        """
        Returns the int16-values from the value reference(s).

        Parameters::

            valueref --
                A list of value references.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_int16([232])

        Calls the low-level FMI function: fmi3GetInt16
        TODO: Currently does not support array variables
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef FMIL.size_t nref = np.size(input_valueref)
        cdef np.ndarray[FMIL3.fmi3_int16_t, ndim=1, mode='c'] output_value = np.zeros(nref, dtype = np.int16)

        if nref == 0: # get_int16([]); do not invoke call to FMU
            return output_value

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        # TODO: Array variables; if any valueref points to an array, output length will be larger
        status = FMIL3.fmi3_import_get_int16(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            nref,
            <FMIL3.fmi3_int16_t*> output_value.data,
            nref
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to get the Int16 values.')

        return output_value

    cpdef np.ndarray get_int8(self, valueref):
        """
        Returns the int8-values from the value reference(s).

        Parameters::

            valueref --
                A list of value references.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_int8([232])

        Calls the low-level FMI function: fmi3GetInt8
        TODO: Currently does not support array variables
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef FMIL.size_t nref = np.size(input_valueref)
        cdef np.ndarray[FMIL3.fmi3_int8_t, ndim=1, mode='c'] output_value = np.zeros(nref, dtype = np.int8)

        if nref == 0: # get_int8([]); do not invoke call to FMU
            return output_value

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        # TODO: Array variables; if any valueref points to an array, output length will be larger
        status = FMIL3.fmi3_import_get_int8(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            nref,
            <FMIL3.fmi3_int8_t*> output_value.data,
            nref
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to get the Int8 values.')

        return output_value

    cpdef np.ndarray get_uint64(self, valueref):
        """
        Returns the uint64-values from the value reference(s).

        Parameters::

            valueref --
                A list of value references.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_uint64([232])

        Calls the low-level FMI function: fmi3GetUInt64
        TODO: Currently does not support array variables
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef FMIL.size_t nref = np.size(input_valueref)
        cdef np.ndarray[FMIL3.fmi3_uint64_t, ndim=1, mode='c'] output_value = np.zeros(nref, dtype = np.uint64)

        if nref == 0: # get_uint64([]); do not invoke call to FMU
            return output_value

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        # TODO: Array variables; if any valueref points to an array, output length will be larger
        status = FMIL3.fmi3_import_get_uint64(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            nref,
            <FMIL3.fmi3_uint64_t*> output_value.data,
            nref
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to get the UInt64 values.')

        return output_value

    cpdef np.ndarray get_uint32(self, valueref):
        """
        Returns the uint32-values from the value reference(s).

        Parameters::

            valueref --
                A list of value references.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_uint32([232])

        Calls the low-level FMI function: fmi3GetUInt32
        TODO: Currently does not support array variables
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef FMIL.size_t nref = np.size(input_valueref)
        cdef np.ndarray[FMIL3.fmi3_uint32_t, ndim=1, mode='c'] output_value = np.zeros(nref, dtype = np.uint32)

        if nref == 0: # get_uint32([]); do not invoke call to FMU
            return output_value

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        # TODO: Array variables; if any valueref points to an array, output length will be larger
        status = FMIL3.fmi3_import_get_uint32(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            nref,
            <FMIL3.fmi3_uint32_t*> output_value.data,
            nref
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to get the UInt32 values.')

        return output_value

    cpdef np.ndarray get_uint16(self, valueref):
        """
        Returns the uint16-values from the value reference(s).

        Parameters::

            valueref --
                A list of value references.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_uint16([232])

        Calls the low-level FMI function: fmi3GetUInt16
        TODO: Currently does not support array variables
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef FMIL.size_t nref = np.size(input_valueref)
        cdef np.ndarray[FMIL3.fmi3_uint16_t, ndim=1, mode='c'] output_value = np.zeros(nref, dtype = np.uint16)

        if nref == 0: # get_uint16([]); do not invoke call to FMU
            return output_value

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        # TODO: Array variables; if any valueref points to an array, output length will be larger
        status = FMIL3.fmi3_import_get_uint16(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            nref,
            <FMIL3.fmi3_uint16_t*> output_value.data,
            nref
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to get the UInt16 values.')

        return output_value

    cpdef np.ndarray get_uint8(self, valueref):
        """
        Returns the uint8-values from the value reference(s).

        Parameters::

            valueref --
                A list of value references.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_uint8([232])

        Calls the low-level FMI function: fmi3GetUInt8
        TODO: Currently does not support array variables
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef FMIL.size_t nref = np.size(input_valueref)
        cdef np.ndarray[FMIL3.fmi3_uint8_t, ndim=1, mode='c'] output_value = np.zeros(nref, dtype = np.uint8)

        if nref == 0: # get_uint8([]); do not invoke call to FMU
            return output_value

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        # TODO: Array variables; if any valueref points to an array, output length will be larger
        status = FMIL3.fmi3_import_get_uint8(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            nref,
            <FMIL3.fmi3_uint8_t*> output_value.data,
            nref
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to get the UInt8 values.')

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

    cpdef list get_string(self, valueref):
        """
        Returns the string-values from the value reference(s).

        Parameters::

            valueref --
                A list of value references.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_string([232])

        Calls the low-level FMI function: fmi3GetString
        TODO: Currently does not support array variables
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.array(valueref, dtype = np.uint32).ravel()
        cdef FMIL.size_t nref = np.size(input_valueref)

        if nref == 0: # get_string([]); do not invoke call to FMU
            return []

        cdef FMIL3.fmi3_string_t* output_value = <FMIL3.fmi3_string_t*>FMIL.malloc(sizeof(FMIL3.fmi3_string_t)*nref)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        # TODO: Array variables; if any valueref points to an array, output length will be larger
        status = FMIL3.fmi3_import_get_string(
            self._fmu,
            <FMIL3.fmi3_value_reference_t*> input_valueref.data,
            nref,
            output_value,
            nref
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException('Failed to get the String values.')

        out = []
        for i in range(nref):
            out.append(pyfmi_util.decode(output_value[i]))

        FMIL.free(output_value)

        return out

    cpdef np.ndarray get_enum(self, valueref):
        """
        Returns the enum-values from the value reference(s).

        Parameters::

            valueref --
                A list of value references.

        Returns::

            values --
                The values retrieved from the FMU.

        Example::

            val = model.get_enum([232])

        Calls the low-level FMI function: fmi3GetInt64
        TODO: Currently does not support array variables
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] input_valueref = np.asarray(valueref, dtype = np.uint32).ravel()
        cdef FMIL.size_t nref = np.size(input_valueref)
        cdef np.ndarray[FMIL3.fmi3_int64_t, ndim=1, mode='c'] output_value = np.zeros(nref, dtype = np.int64)

        if nref == 0: # get_enum([]); do not invoke call to FMU
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
            raise FMUException('Failed to get the Enum values.')

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
        cdef FMIL3.fmi3_import_variable_t* variable = _get_variable_by_name(self._fmu, variable_name)
        return FMIL3.fmi3_import_get_variable_vr(variable)

    cdef FMIL3.fmi3_base_type_enu_t _get_variable_data_type(self, variable_name) except *:
        cdef FMIL3.fmi3_import_variable_t* variable = _get_variable_by_name(self._fmu, variable_name)
        return FMIL3.fmi3_import_get_variable_base_type(variable)

    cdef FMIL3.fmi3_causality_enu_t _get_variable_causality(self, variable_name) except *:
        cdef FMIL3.fmi3_import_variable_t* variable = _get_variable_by_name(self._fmu, variable_name)
        return FMIL3.fmi3_import_get_variable_causality(variable)

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

            if include_alias and FMIL3.fmi3_import_get_variable_has_alias(variable):
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
        # TODO: We may want to revisit the format here; e.g., optional inputs to filter different types?
        return self.get_model_variables(
            type = FMI3_Type.FLOAT64,
            include_alias = False,
            causality = FMI3_Causality.INPUT,
            variability = FMI3_Variability.CONTINUOUS)

    def get_output_list(self):
        """
        Returns a dictionary with output variables

        Returns::
            An dictionary with the (float64, continuous) output variables.
        """
        # TODO: We may want to revisit the format here; e.g., optional inputs to filter different types?
        return self.get_model_variables(
            type = FMI3_Type.FLOAT64,
            include_alias = False,
            causality = FMI3_Causality.OUTPUT,
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

    def get_variable_causality(self, variable_name):
        """
        Get causality of variable.

        Parameter::
            variable_name --
                The name of the variable.

        Returns::
            The causality of the variable, as an instance of pyfmi.fmi3.FMI3_Causality.
        """
        variable_causality = self._get_variable_causality(variable_name)
        return FMI3_Causality(int(variable_causality))

    cdef _get_variable_description(self, FMIL3.fmi3_import_variable_t* variable):
        cdef FMIL3.fmi3_string_t desc = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_variable_description(variable)
        return pyfmi_util.decode(desc) if desc != NULL else ""

    cdef _get_alias_description(self, FMIL3.fmi3_import_alias_variable_t* alias_variable):
        cdef FMIL3.fmi3_string_t desc = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_alias_variable_description(alias_variable)
        return pyfmi_util.decode(desc) if desc != NULL else ""

    cpdef get_variable_description(self, variable_name):
        """
        Get the description of a given variable or alias variable.

        Parameter::

            variable_name --
                The name of the variable

        Returns::

            The description of the variable.
        """
        variable_name_encoded = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name_encoded
        cdef FMIL3.fmi3_string_t desc = FMIL3.fmi3_import_get_variable_description_by_name(self._fmu, variablename)
        if desc == NULL:
            raise FMUException("The variable %s could not be found." % variable_name)
        return pyfmi_util.decode(desc)

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

                A dictionary with the state variables.
        """
        cdef FMIL3.fmi3_import_variable_list_t* variable_list
        cdef FMIL.size_t                        variable_list_size
        cdef FMIL3.fmi3_import_variable_t*         der_variable
        cdef FMIL3.fmi3_import_float64_variable_t* variable
        variable_dict = {}

        variable_list = FMIL3.fmi3_import_get_continuous_state_derivatives_list(self._fmu)
        if variable_list == NULL:
            raise FMUException("Unexpected failure in retrieving the continuous state derivatives list.")
        variable_list_size = FMIL3.fmi3_import_get_variable_list_size(variable_list)

        for i in range(variable_list_size):
            der_variable = FMIL3.fmi3_import_get_variable(variable_list, i)
            variable = FMIL3.fmi3_import_get_float64_variable_derivative_of(<FMIL3.fmi3_import_float64_variable_t*>der_variable)

            scalar_variable = self._add_variable(<FMIL3.fmi3_import_variable_t*>variable)
            variable_dict[scalar_variable.name] = scalar_variable

        FMIL3.fmi3_import_free_variable_list(variable_list)

        return variable_dict

    def get_derivatives_list(self):
        """
        Returns a dictionary with the states derivatives.

        Returns::

            A dictionary with the derivative variables.
        """
        cdef FMIL3.fmi3_import_variable_list_t* variable_list
        cdef FMIL.size_t                        variable_list_size
        cdef FMIL3.fmi3_import_variable_t*      variable
        variable_dict = {}

        variable_list = FMIL3.fmi3_import_get_continuous_state_derivatives_list(self._fmu)
        if variable_list == NULL:
            raise FMUException("Unexpected failure in retrieving the continuous state derivatives list.")
        variable_list_size = FMIL3.fmi3_import_get_variable_list_size(variable_list)

        for i in range(variable_list_size):
            variable = FMIL3.fmi3_import_get_variable(variable_list, i)
            scalar_variable = self._add_variable(variable)
            variable_dict[scalar_variable.name] = scalar_variable

        FMIL3.fmi3_import_free_variable_list(variable_list)

        return variable_dict

    cpdef get_output_dependencies(self):
        """
        Retrieve the variables that the (continuous, float64) outputs are
        dependent on. Returns two dictionaries, one with the states and
        one with the (continuous, float64) inputs.
        """
        # TODO: More than just float64 outputs?
        # caching
        if (self._outputs_states_dependencies is not None and
            self._outputs_inputs_dependencies is not None):
               return self._outputs_states_dependencies, self._outputs_inputs_dependencies

        cdef FMIL3.fmi3_import_variable_t      *variable
        cdef int ret
        cdef int dependsOnAll
        cdef size_t numDependencies
        cdef size_t* dependencies
        cdef char* dependenciesKind

        cdef dict output_vars = self.get_output_list()
        cdef dict states_dict = self.get_states_list()
        cdef list states_list = list(states_dict.keys())
        cdef dict inputs_dict = self.get_input_list()
        cdef list inputs_list = list(inputs_dict.keys())

        cdef dict map_vr_to_state = {state_var.value_reference: state_var_name for state_var_name, state_var in states_dict.items()}
        cdef dict map_vr_to_input = {input_var.value_reference: input_var_name for input_var_name, input_var in inputs_dict.items()}

        states = {}
        states_kind = {}
        inputs = {}
        inputs_kind = {}

        if len(output_vars) != 0: # If there are no outputs, return empty dicts
            for output_var_name, output_var in output_vars.items():
                variable = FMIL3.fmi3_import_get_variable_by_vr(self._fmu, <FMIL3.fmi3_value_reference_t>output_var.value_reference)
                if variable == NULL:
                    raise FMUException(f"Unexpected failure retreiving model variable {output_var_name}")

                ret = FMIL3.fmi3_import_get_output_dependencies(self._fmu, variable, &numDependencies, &dependsOnAll, &dependencies, &dependenciesKind)
                if ret != 0:
                    raise FMUException(f"Unexpected failure retreiving dependencies of variable {output_var_name}")

                if (numDependencies == 0) and (dependsOnAll == 0):
                    states[output_var_name] = []
                    states_kind[output_var_name] = []

                    inputs[output_var_name] = []
                    inputs_kind[output_var_name] = []
                if (numDependencies == 0) and (dependsOnAll == 1):
                    states[output_var_name] = states_list
                    states_kind[output_var_name] = [FMI3_DependencyKind.DEPENDENT]*len(states_list)

                    inputs[output_var_name] = inputs_list
                    inputs_kind[output_var_name] = [FMI3_DependencyKind.DEPENDENT]*len(inputs_list)
                else:
                    states[output_var_name] = []
                    states_kind[output_var_name] = []

                    inputs[output_var_name] = []
                    inputs_kind[output_var_name] = []
                    for i in range(numDependencies):
                        dependency_value_ref = dependencies[i]
                        if dependency_value_ref in map_vr_to_state:
                            states[output_var_name].append(map_vr_to_state[dependency_value_ref])
                            states_kind[output_var_name].append(FMI3_DependencyKind(dependenciesKind[i]))
                        elif dependency_value_ref in map_vr_to_input:
                            inputs[output_var_name].append(map_vr_to_input[dependency_value_ref])
                            inputs_kind[output_var_name].append(FMI3_DependencyKind(dependenciesKind[i]))
                        else:
                            pass # XXX: Not float64 or continuous
                # XXX: Edge case; does a state that is an output need to list itself as dependency?
                if (output_var.value_reference in map_vr_to_state) and \
                   (map_vr_to_state[output_var.value_reference] not in states[output_var_name]):
                    states[output_var_name].append(map_vr_to_state[output_var.value_reference])
                    states_kind[output_var_name].append(FMI3_DependencyKind.CONSTANT)

        # Caching
        self._outputs_states_dependencies = states
        self._outputs_states_dependencies_kind = states_kind
        self._outputs_inputs_dependencies = inputs
        self._outputs_inputs_dependencies_kind = inputs_kind

        return states, inputs

    cpdef get_output_dependencies_kind(self):
        """
        Retrieve the 'kinds' that the (continuous, float64) outputs are
        dependent on. Returns two dictionaries, one with the states
        and one with the (continuous, float64) inputs. The list of 'kinds'::

            FMI3_DependencyKind.DEPENDENT
            FMI3_DependencyKind.CONSTANT
            FMI3_DependencyKind.FIXED
            FMI3_DependencyKind.TUNABLE
            FMI3_DependencyKind.DISCRETE

        """
        # TODO: More than just float64 outputs?
        self.get_output_dependencies()
        return self._outputs_states_dependencies_kind, self._outputs_inputs_dependencies_kind

    cpdef get_derivatives_dependencies(self):
        """
        Retrieve the variables that the derivatives are
        dependent on. Returns two dictionaries, one with the states
        and one with the (continuous float64) inputs.
        """
        # TODO: More than just float64?
        # Caching
        if (self._derivatives_states_dependencies is not None and
            self._derivatives_inputs_dependencies is not None):
               return self._derivatives_states_dependencies, self._derivatives_inputs_dependencies

        cdef FMIL3.fmi3_import_variable_t      *variable
        cdef int ret
        cdef int dependsOnAll
        cdef size_t numDependencies
        cdef size_t* dependencies
        cdef char* dependenciesKind

        cdef dict derivatives = self.get_derivatives_list()
        cdef dict states_dict = self.get_states_list()
        cdef list states_list = list(states_dict.keys())
        cdef dict inputs_dict = self.get_input_list()
        cdef list inputs_list = list(inputs_dict.keys())

        cdef dict map_vr_to_state = {state_var.value_reference: state_var_name for state_var_name, state_var in states_dict.items()}
        cdef dict map_vr_to_input = {input_var.value_reference: input_var_name for input_var_name, input_var in inputs_dict.items()}

        states = {}
        states_kind = {}
        inputs = {}
        inputs_kind = {}

        if len(derivatives) != 0: # If there are no derivatives, return empty dicts
            for der_name, der_var in derivatives.items():
                variable = FMIL3.fmi3_import_get_variable_by_vr(self._fmu, <FMIL3.fmi3_value_reference_t>der_var.value_reference)
                if variable == NULL:
                    raise FMUException(f"Unexpected failure retreiving model variable {der_name}")

                ret = FMIL3.fmi3_import_get_continuous_state_derivative_dependencies(self._fmu, variable, &numDependencies, &dependsOnAll, &dependencies, &dependenciesKind)
                if ret != 0:
                    raise FMUException(f"Unexpected failure retreiving dependencies of variable {der_name}")

                if (numDependencies == 0) and (dependsOnAll == 0):
                    states[der_name] = []
                    states_kind[der_name] = []

                    inputs[der_name] = []
                    inputs_kind[der_name] = []
                if (numDependencies == 0) and (dependsOnAll == 1):
                    states[der_name] = states_list
                    states_kind[der_name] = [FMI3_DependencyKind.DEPENDENT]*len(states_list)

                    inputs[der_name] = inputs_list
                    inputs_kind[der_name] = [FMI3_DependencyKind.DEPENDENT]*len(inputs_list)
                else:
                    states[der_name] = []
                    states_kind[der_name] = []

                    inputs[der_name] = []
                    inputs_kind[der_name] = []
                    for i in range(numDependencies):
                        dependency_value_ref = dependencies[i]
                        if dependency_value_ref in map_vr_to_state:
                            states[der_name].append(map_vr_to_state[dependency_value_ref])
                            states_kind[der_name].append(FMI3_DependencyKind(dependenciesKind[i]))
                        elif dependency_value_ref in map_vr_to_input:
                            inputs[der_name].append(map_vr_to_input[dependency_value_ref])
                            inputs_kind[der_name].append(FMI3_DependencyKind(dependenciesKind[i]))
                        else:
                            pass # XXX: Not float64 or continuous

        # Caching
        self._derivatives_states_dependencies = states
        self._derivatives_states_dependencies_kind = states_kind
        self._derivatives_inputs_dependencies = inputs
        self._derivatives_inputs_dependencies_kind = inputs_kind

        return states, inputs

    cpdef get_derivatives_dependencies_kind(self):
        """
        Retrieve the 'kinds' that the derivatives are
        dependent on. Returns two dictionaries, one with the states
        and one with the (continuous float64) inputs. The list of 'kinds'::

            FMI3_DependencyKind.DEPENDENT
            FMI3_DependencyKind.CONSTANT
            FMI3_DependencyKind.FIXED
            FMI3_DependencyKind.TUNABLE
            FMI3_DependencyKind.DISCRETE

        """
        # TODO: More than just float64?
        self.get_derivatives_dependencies()

        return self._derivatives_states_dependencies_kind, self._derivatives_inputs_dependencies_kind

    def _get_directional_proxy(self, var_ref, func_ref, group = None, add_diag = False, output_matrix = None):
        cdef list data = [], row = [], col = []
        cdef list local_group
        cdef int nbr_var_ref  = len(var_ref), nbr_func_ref = len(func_ref)
        cdef np.ndarray[FMIL3.fmi3_float64_t, ndim=1, mode='c'] v = np.zeros(nbr_var_ref, dtype = np.double)
        cdef np.ndarray[FMIL3.fmi3_float64_t, ndim=1, mode='c'] data_local
        cdef int ind_local = 5 if add_diag else 4

        if not self._has_entered_init_mode:
            raise FMUException("The FMU has not entered initialization mode and thus the directional " \
                               "derivatives cannot be computed. Call enter_initialization_mode to start the initialization.")

        if group is not None:
            if output_matrix is not None:
                if not isinstance(output_matrix, sps.csc_matrix):
                    output_matrix = None
                else:
                    data_local = output_matrix.data

            if add_diag and output_matrix is None:
                dim = min(nbr_var_ref, nbr_func_ref)
                data.extend([0.0] * dim)
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
                    A = sps.csc_matrix((nbr_func_ref,nbr_var_ref))
                else:
                    A = sps.csc_matrix((data, (row, col)), (nbr_func_ref, nbr_var_ref))

            return A
        else:
            if output_matrix is None or \
                (not isinstance(output_matrix, np.ndarray)) or \
                (isinstance(output_matrix, np.ndarray) and (output_matrix.shape[0] != nbr_func_ref or output_matrix.shape[1] != nbr_var_ref)):
                    A = np.zeros((nbr_func_ref, nbr_var_ref))
            else:
                A = output_matrix

            for i in range(nbr_var_ref):
                v[i] = 1.0
                A[:, i] = self.get_directional_derivative(var_ref, func_ref, v)
                v[i] = 0.0
            return A

    def _get_A(self, use_structure_info = True, add_diag = True, output_matrix = None):
        if self._group_A is None and use_structure_info:
            [derv_state_dep, derv_input_dep] = self.get_derivatives_dependencies()
            self._group_A = pyfmi_util.cpr_seed(derv_state_dep, list(self.get_states_list().keys()))
        if self._states_references is None:
            states                       = self.get_states_list()
            self._states_references      = [s.value_reference for s in states.values()]
        if self._derivatives_references is None:
            derivatives                  = self.get_derivatives_list()
            self._derivatives_references = [s.value_reference for s in derivatives.values()]

        return self._get_directional_proxy(
            var_ref = self._states_references,
            func_ref = self._derivatives_references,
            group = self._group_A if use_structure_info else None,
            add_diag = add_diag,
            output_matrix = output_matrix
        )

    def _get_B(self, use_structure_info = True, add_diag = False, output_matrix = None):
        if self._group_B is None and use_structure_info:
            [derv_state_dep, derv_input_dep] = self.get_derivatives_dependencies()
            self._group_B = pyfmi_util.cpr_seed(derv_input_dep, list(self.get_input_list().keys()))
        if self._inputs_references is None:
            inputs                       = self.get_input_list()
            self._inputs_references      = [s.value_reference for s in inputs.values()]
        if self._derivatives_references is None:
            derivatives                  = self.get_derivatives_list()
            self._derivatives_references = [s.value_reference for s in derivatives.values()]

        return self._get_directional_proxy(
            var_ref = self._inputs_references,
            func_ref = self._derivatives_references,
            group = self._group_B if use_structure_info else None,
            add_diag = add_diag,
            output_matrix = output_matrix
        )

    def _get_C(self, use_structure_info = True, add_diag = False, output_matrix = None):
        if self._group_C is None and use_structure_info:
            [out_state_dep, out_input_dep] = self.get_output_dependencies()
            self._group_C = pyfmi_util.cpr_seed(out_state_dep, list(self.get_states_list().keys()))
        if self._states_references is None:
            states                       = self.get_states_list()
            self._states_references      = [s.value_reference for s in states.values()]
        if self._outputs_references is None:
            outputs                      = self.get_output_list()
            self._outputs_references     = [s.value_reference for s in outputs.values()]

        return self._get_directional_proxy(
            var_ref = self._states_references,
            func_ref = self._outputs_references,
            group = self._group_C if use_structure_info else None,
            add_diag = add_diag,
            output_matrix = output_matrix
        )

    def _get_D(self, use_structure_info = True, add_diag = False, output_matrix = None):
        if self._group_D is None and use_structure_info:
            [out_state_dep, out_input_dep] = self.get_output_dependencies()
            self._group_D = pyfmi_util.cpr_seed(out_input_dep, list(self.get_input_list().keys()))
        if self._inputs_references is None:
            inputs                       = self.get_input_list()
            self._inputs_references      = [s.value_reference for s in inputs.values()]
        if self._outputs_references is None:
            outputs                      = self.get_output_list()
            self._outputs_references     = [s.value_reference for s in outputs.values()]

        return self._get_directional_proxy(
            var_ref = self._inputs_references,
            func_ref = self._outputs_references,
            group = self._group_D if use_structure_info else None,
            add_diag = add_diag,
            output_matrix = output_matrix
        )

    def get_state_space_representation(self, A = True, B = True, C = True, D = True, use_structure_info = True):
        """
        Returns a state space representation of the model. I.e::

            der(x) = Ax + Bu
                y  = Cx + Du

        (float64 & continuous inputs/outputs only)

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
            The A, B, C, D matrices. If not all are computed, the ones that
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

        return A, B, C, D

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

            Also, only a subset of the derivatives and states can be selected:

            model.get_directional_derivative(var_ref = [0,1], func_ref = [2,3], v = [1,2])

            This returns a vector with two values where:

            values[0] = (df2/dv0) * 1 + (df2/dv1) * 2
            values[1] = (df3/dv0) * 1 + (df3/dv1) * 2

        """

        cdef FMIL3.fmi3_status_t status
        cdef FMIL.size_t nv, nz

        # input arrays
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] v_ref = np.zeros(len(var_ref), dtype = np.uint32)
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] z_ref = np.zeros(len(func_ref), dtype = np.uint32)
        cdef np.ndarray[FMIL3.fmi3_float64_t, ndim=1, mode='c'] dv = np.zeros(len(v), dtype = np.double)
        # output array
        cdef np.ndarray[FMIL3.fmi3_float64_t, ndim=1, mode='c'] dz = np.zeros(len(func_ref), dtype = np.double)

        if not self._provides_directional_derivatives():
            raise FMUException('This FMU does not provide directional derivatives')

        if len(var_ref) != len(v):
            raise FMUException(f'The length of the list with variables (var_ref) is {len(var_ref)} and the seed vector (v) is {len(v)}, these must be of equal length')

        for i in range(len(var_ref)):
            v_ref[i] = var_ref[i]
            dv[i] = v[i]
        for j in range(len(func_ref)):
            z_ref[j] = func_ref[j]

        nv = len(v_ref)
        nz = len(z_ref)
        # TODO: Array variables

        status = self._get_directional_derivative(v_ref, z_ref, dv, dz)

        if status != FMIL3.fmi3_status_ok:
            raise FMUException('An error occured while getting the directional derivative, see the log for possible more information')

        return dz

    cdef FMIL3.fmi3_status_t _get_directional_derivative(self, np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode="c"] v_ref,
                                               np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode="c"] z_ref,
                                               np.ndarray[FMIL3.fmi3_float64_t, ndim=1, mode="c"] dv,
                                               np.ndarray[FMIL3.fmi3_float64_t, ndim=1, mode="c"] dz) except *:
        cdef FMIL3.fmi3_status_t status

        assert np.size(dv) >= np.size(v_ref) and np.size(dz) >= np.size(z_ref)

        if not self._provides_directional_derivatives():
            raise FMUException('This FMU does not provide directional derivatives')

        # TODO: Array variables; we'll likely be calculating the output size here as well
        # XXX: Might even want to "general purpose" directional derivatives + one specifically all scalar
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_get_directional_derivative(self._fmu,
                  <FMIL3.fmi3_value_reference_t*> z_ref.data, np.size(z_ref),
                  <FMIL3.fmi3_value_reference_t*> v_ref.data, np.size(v_ref),
                  <FMIL3.fmi3_float64_t*> dv.data, np.size(dv),
                  <FMIL3.fmi3_float64_t*> dz.data, np.size(dz))
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        return status

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

    def _deactivate_logging(self):
        self.set_log_level(0)
        self.set_debug_logging(False, [])

    def set_debug_logging(self, logging_on, categories = []):
        """
        Specifies if the debugging should be turned on or off and calls fmi3SetDebugLogging
        for the specified categories, after checking they are valid.

        Note: An appropriate log_level of the FMU is required for logging.
        Typically, this is INFO - set via fmu.set_log_level(4).

        Parameters::

            logging_on --
                Boolean value.

            categories --
                List of categories to log, use get_log_categories() to query categories.
                Default: [] (all categories)

        Calls the low-level FMI function: fmi3SetDebugLogging
        """

        cdef FMIL3.fmi3_boolean_t log = 1 if logging_on else 0
        cdef FMIL3.fmi3_status_t  status
        cdef FMIL.size_t          n_cat = np.size(categories)
        cdef FMIL3.fmi3_string_t* val

        self._enable_logging = bool(log)

        val = <FMIL3.fmi3_string_t*>FMIL.malloc(sizeof(FMIL3.fmi3_string_t)*n_cat)
        values = [pyfmi_util.encode(cat) for cat in categories]
        for i in range(n_cat):
            val[i] = values[i]

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_set_debug_logging(self._fmu, log, n_cat, val)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        FMIL.free(val)

        if status != FMIL3.fmi3_status_ok:
            raise FMUException('Failed to set the debugging option.')

    def get_log_categories(self) -> dict[str, str]:
        """
        Method used to retrieve the logging categories. 

        Returns::

            dict(category_name, description string)
        """
        cdef FMIL.size_t i, nbr_categories = FMIL3.fmi3_import_get_log_categories_num(self._fmu)
        cdef dict ret = {}
        cdef str cat, descr

        for i in range(nbr_categories):
            cat = str(FMIL3.fmi3_import_get_log_category(self._fmu, i).decode())
            descr = str(FMIL3.fmi3_import_get_log_category_description(self._fmu, i).decode())
            ret[cat] = descr

        return ret

    def get_variable_by_valueref(self, valueref: int) -> str:
        """
        Get the name of a variable given a value reference.

        Parameters::

            valueref --
                The value reference of the variable.

        Returns::

            The name of the variable.

        """
        # Could have a better name?
        cdef FMIL3.fmi3_import_variable_t* variable
        variable = FMIL3.fmi3_import_get_variable_by_vr(self._fmu, <FMIL3.fmi3_value_reference_t> valueref)
        if variable == NULL:
            raise FMUException("The variable with the valuref %i could not be found."%valueref)

        return pyfmi_util.decode(FMIL3.fmi3_import_get_variable_name(variable))

    def get_variable_variability(self, variable_name: str) -> FMI3_Variability:
        """
        Get variability of the variable.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The variability of the variable as FMI3_Variability enum
        """
        cdef FMIL3.fmi3_import_variable_t* variable = _get_variable_by_name(self._fmu, variable_name)
        cdef FMIL3.fmi3_variability_enu_t variability = FMIL3.fmi3_import_get_variable_variability(variable)
        return FMI3_Variability(variability)

    def get_variable_initial(self, variable_name) -> FMI3_Initial:
        """
        Get initial of the variable.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The initial of the variable as FMI3_Initial enum
        """
        cdef FMIL3.fmi3_import_variable_t* variable = _get_variable_by_name(self._fmu, variable_name)
        cdef FMIL3.fmi3_initial_enu_t initial = FMIL3.fmi3_import_get_variable_initial(variable)
        return FMI3_Initial(initial)

    cpdef get_variable_min(self, str variable_name):
        """
        Returns the minimum value for the given variable.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The minimum value for the variable.
        """
        cdef FMIL3.fmi3_import_variable_t* variable = _get_variable_by_name(self._fmu, variable_name)
        cdef FMIL3.fmi3_base_type_enu_t base_type = FMIL3.fmi3_import_get_variable_base_type(variable)

        if base_type == FMIL3.fmi3_base_type_float64:
            return FMIL3.fmi3_import_get_float64_variable_min(FMIL3.fmi3_import_get_variable_as_float64(variable))
        elif base_type == FMIL3.fmi3_base_type_float32:
            return FMIL3.fmi3_import_get_float32_variable_min(FMIL3.fmi3_import_get_variable_as_float32(variable))
        elif base_type == FMIL3.fmi3_base_type_int64:
            return FMIL3.fmi3_import_get_int64_variable_min(FMIL3.fmi3_import_get_variable_as_int64(variable))
        elif base_type == FMIL3.fmi3_base_type_int32:
            return FMIL3.fmi3_import_get_int32_variable_min(FMIL3.fmi3_import_get_variable_as_int32(variable))
        elif base_type == FMIL3.fmi3_base_type_int16:
            return FMIL3.fmi3_import_get_int16_variable_min(FMIL3.fmi3_import_get_variable_as_int16(variable))
        elif base_type == FMIL3.fmi3_base_type_int8:
            return FMIL3.fmi3_import_get_int8_variable_min(FMIL3.fmi3_import_get_variable_as_int8(variable))
        elif base_type == FMIL3.fmi3_base_type_uint64:
            return FMIL3.fmi3_import_get_uint64_variable_min(FMIL3.fmi3_import_get_variable_as_uint64(variable))
        elif base_type == FMIL3.fmi3_base_type_uint32:
            return FMIL3.fmi3_import_get_uint32_variable_min(FMIL3.fmi3_import_get_variable_as_uint32(variable))
        elif base_type == FMIL3.fmi3_base_type_uint16:
            return FMIL3.fmi3_import_get_uint16_variable_min(FMIL3.fmi3_import_get_variable_as_uint16(variable))
        elif base_type == FMIL3.fmi3_base_type_uint8:
            return FMIL3.fmi3_import_get_uint8_variable_min(FMIL3.fmi3_import_get_variable_as_uint8(variable))
        elif base_type == FMIL3.fmi3_base_type_enum:
            return FMIL3.fmi3_import_get_enum_variable_min(FMIL3.fmi3_import_get_variable_as_enum(variable))
        else:
            raise FMUException("Given variable type does not have a minimum.")

    cpdef get_variable_max(self, str variable_name):
        """
        Returns the maximum value for the given variable.

        Parameters::

            variable_name --
                The name of the variable.

        Returns::

            The maximum value for the variable.
        """
        cdef FMIL3.fmi3_import_variable_t* variable = _get_variable_by_name(self._fmu, variable_name)
        cdef FMIL3.fmi3_base_type_enu_t base_type = FMIL3.fmi3_import_get_variable_base_type(variable)

        if base_type == FMIL3.fmi3_base_type_float64:
            return FMIL3.fmi3_import_get_float64_variable_max(FMIL3.fmi3_import_get_variable_as_float64(variable))
        elif base_type == FMIL3.fmi3_base_type_float32:
            return FMIL3.fmi3_import_get_float32_variable_max(FMIL3.fmi3_import_get_variable_as_float32(variable))
        elif base_type == FMIL3.fmi3_base_type_int64:
            return FMIL3.fmi3_import_get_int64_variable_max(FMIL3.fmi3_import_get_variable_as_int64(variable))
        elif base_type == FMIL3.fmi3_base_type_int32:
            return FMIL3.fmi3_import_get_int32_variable_max(FMIL3.fmi3_import_get_variable_as_int32(variable))
        elif base_type == FMIL3.fmi3_base_type_int16:
            return FMIL3.fmi3_import_get_int16_variable_max(FMIL3.fmi3_import_get_variable_as_int16(variable))
        elif base_type == FMIL3.fmi3_base_type_int8:
            return FMIL3.fmi3_import_get_int8_variable_max(FMIL3.fmi3_import_get_variable_as_int8(variable))
        elif base_type == FMIL3.fmi3_base_type_uint64:
            return FMIL3.fmi3_import_get_uint64_variable_max(FMIL3.fmi3_import_get_variable_as_uint64(variable))
        elif base_type == FMIL3.fmi3_base_type_uint32:
            return FMIL3.fmi3_import_get_uint32_variable_max(FMIL3.fmi3_import_get_variable_as_uint32(variable))
        elif base_type == FMIL3.fmi3_base_type_uint16:
            return FMIL3.fmi3_import_get_uint16_variable_max(FMIL3.fmi3_import_get_variable_as_uint16(variable))
        elif base_type == FMIL3.fmi3_base_type_uint8:
            return FMIL3.fmi3_import_get_uint8_variable_max(FMIL3.fmi3_import_get_variable_as_uint8(variable))
        elif base_type == FMIL3.fmi3_base_type_enum:
            return FMIL3.fmi3_import_get_enum_variable_max(FMIL3.fmi3_import_get_variable_as_enum(variable))
        else:
            raise FMUException("Given variable type does not have a maximum.")

    def get_model_version(self) -> str:
        """ Returns the version of the FMU. """
        cdef FMIL3.fmi3_string_t version = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_model_version(self._fmu)
        return pyfmi_util.decode(version) if version != NULL else ""

    def get_version(self) -> str:
        """ Returns the FMI version of the Model which it was generated according. """
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        cdef FMIL3.fmi3_string_t version = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_version(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        return pyfmi_util.decode(version)

    def get_name(self) -> str:
        """ Return the model name as used in the modeling environment. """
        return self._modelName

    def get_author(self) -> str:
        """
        Return the name and organization of the model author.
        """
        cdef FMIL3.fmi3_string_t author = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_author(self._fmu)
        return pyfmi_util.decode(author) if author != NULL else ""

    def get_copyright(self) -> str:
        """
        Return the model copyright.
        """
        cdef FMIL3.fmi3_string_t copyright = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_copyright(self._fmu)
        return pyfmi_util.decode(copyright) if copyright != NULL else ""

    def get_description(self) -> str:
        """
        Return the model description.
        """
        cdef FMIL3.fmi3_string_t desc = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_description(self._fmu)
        return pyfmi_util.decode(desc) if desc != NULL else ""

    def get_model_version(self) -> str:
        """
        Returns the version of the FMU.
        """
        cdef FMIL3.fmi3_string_t version = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_model_version(self._fmu)
        return pyfmi_util.decode(version) if version != NULL else ""

    def get_instantiation_token(self) -> str:
        """
        Returns the instatiation token of the FMU.
        """
        cdef FMIL3.fmi3_string_t inst_token = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_instantiation_token(self._fmu)
        return pyfmi_util.decode(inst_token) if inst_token != NULL else ""

    def get_license(self) -> str:
        """
        Return the model license.
        """
        cdef FMIL3.fmi3_string_t license = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_license(self._fmu)
        return pyfmi_util.decode(license) if license != NULL else ""

    def get_generation_date_and_time(self) -> str:
        """
        Return the model generation date and time.
        """
        cdef FMIL3.fmi3_string_t gen = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_generation_date_and_time(self._fmu)
        return pyfmi_util.decode(gen) if gen != NULL else ""

    def get_variable_naming_convention(self) -> str:
        """
        Return the variable naming convention.
        """
        cdef FMIL3.fmi3_variable_naming_convension_enu_t conv_enum = FMIL3.fmi3_import_get_naming_convention(self._fmu)
        cdef FMIL3.fmi3_string_t conv_name = <FMIL3.fmi3_string_t>FMIL3.fmi3_naming_convention_to_string(conv_enum)
        return pyfmi_util.decode(conv_name) if conv_name != NULL else ""

    def get_identifier(self):
        """ Return the model identifier, name of binary model file and prefix in the C-function names of the model. """
        raise NotImplementedError # to implemented in FMUModel(ME|CS|SE)3

    def get_ode_sizes(self):
        """ Returns the number of continuous states and the number of event indicators.

            Returns::

                Tuple (The number of continuous states, The number of event indicators)
                [n_states, n_event_ind] = model.get_ode_sizes()
        """
        return self._nContinuousStates, self._nEventIndicators

    def get_default_experiment_start_time(self) -> float:
        """ Returns the default experiment start time as defined in modelDescription.xml. """
        return FMIL3.fmi3_import_get_default_experiment_start(self._fmu)

    def get_default_experiment_stop_time(self) -> float:
        """ Returns the default experiment stop time as defined in modelDescription.xml. """
        return FMIL3.fmi3_import_get_default_experiment_stop(self._fmu)

    def get_default_experiment_tolerance(self) -> float:
        """ Returns the default experiment tolerance as defined in modelDescription.xml. """
        return FMIL3.fmi3_import_get_default_experiment_tolerance(self._fmu)

    def get_default_experiment_step(self) -> float:
        """Returns the default experiment step-size as defined in modelDescription.xml."""
        return FMIL3.fmi3_import_get_default_experiment_step_size(self._fmu)

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
        cdef FMIL3.fmi3_import_variable_t* variable = _get_variable_by_name(self._fmu, variable_name)
        return pyfmi_util.decode(FMIL3.fmi3_import_get_variable_name(variable))

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

        variable = _get_variable_by_name(self._fmu, variable_name)

        base_name = pyfmi_util.decode(FMIL3.fmi3_import_get_variable_name(variable))
        ret_values[base_name] = False

        alias_list = FMIL3.fmi3_import_get_variable_alias_list(variable)
        alias_list_size = FMIL3.fmi3_import_get_alias_variable_list_size(alias_list)
        for idx in range(alias_list_size):
            alias_var = FMIL3.fmi3_import_get_alias(alias_list, idx)
            alias_name = pyfmi_util.decode(FMIL3.fmi3_import_get_alias_variable_name(alias_var))
            ret_values[alias_name] = True

        return ret_values


    def get_fmu_state(self, FMUState3 state = None):
        """
        Creates a copy of the recent FMU-state and returns a pointer to this state which later can be used to
        set the FMU to this state.

        Parameters::

            state --
                Optionally a pointer to an already allocated FMU state

        Returns::

            A pointer to a copy of the recent FMU state.

        Example::

            state = fmu.get_fmu_state()
        """
        cdef FMIL3.fmi3_status_t status

        if not self._supports_get_set_FMU_state():
            raise FMUException('This FMU does not support get and set FMU-state')

        if state is None:
            state = FMUState3()

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_get_fmu_state(self._fmu, &(state.fmu_state))
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != FMIL3.fmi3_status_ok:
            raise FMUException(
                'An error occured while trying to get the FMU-state, see the log for possible more information'
            )

        state._internal_state_variables['time'] = self.time
        state._internal_state_variables['initialized_fmu'] = self._initialized_fmu
        state._internal_state_variables['has_entered_init_mode'] = self._has_entered_init_mode
        state._internal_state_variables['callback_log_level'] = self.callbacks.log_level

        state._internal_state_variables["event_info.new_discrete_states_needed"]            = self._event_info_new_discrete_states_needed
        state._internal_state_variables["event_info.terminate_simulation"]                  = self._event_info_terminate_simulation
        state._internal_state_variables["event_info.nominals_of_continuous_states_changed"] = self._event_info_nominals_of_continuous_states_changed
        state._internal_state_variables["event_info.values_of_continuous_states_changed"]   = self._event_info_values_of_continuous_states_changed
        state._internal_state_variables["event_info.next_event_time_defined"]               = self._event_info_next_event_time_defined
        state._internal_state_variables["event_info.next_event_time"]                       = self._event_info_next_event_time

        return state


    def set_fmu_state(self, FMUState3 state):
        """ Set the FMU to a previous saved state.

        Parameter::

            state--
                A pointer to a FMU-state.

        Example::

            state = fmu.get_fmu_state()
            fmu.set_fmu_state(state)
        """
        cdef FMIL3.fmi3_status_t status
        cdef FMIL3.fmi3_FMU_state_t internal_state = state.fmu_state

        if not self._supports_get_set_FMU_state():
            raise FMUException('This FMU does not support get and set FMU-state')

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_set_fmu_state(self._fmu, internal_state)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != FMIL3.fmi3_status_ok:
            raise FMUException(
                'An error occured while trying to set the FMU-state, see the log for possible more information'
            )

        if state._internal_state_variables['time'] is not None:
            self.time = state._internal_state_variables['time']
        if state._internal_state_variables['has_entered_init_mode'] is not None:
            self._has_entered_init_mode = state._internal_state_variables['has_entered_init_mode']
        if state._internal_state_variables['initialized_fmu'] is not None:
            self._initialized_fmu = state._internal_state_variables['initialized_fmu']
        if state._internal_state_variables['callback_log_level'] is not None:
            self.callbacks.log_level = state._internal_state_variables['callback_log_level']

        if state._internal_state_variables["event_info.new_discrete_states_needed"] is not None:
            self._event_info_new_discrete_states_needed = state._internal_state_variables["event_info.new_discrete_states_needed"]
        if state._internal_state_variables["event_info.nominals_of_continuous_states_changed"] is not None:
            self._event_info_nominals_of_continuous_states_changed = state._internal_state_variables["event_info.nominals_of_continuous_states_changed"]
        if state._internal_state_variables["event_info.terminate_simulation"] is not None:
            self._event_info_terminate_simulation = state._internal_state_variables["event_info.terminate_simulation"]
        if state._internal_state_variables["event_info.values_of_continuous_states_changed"] is not None:
            self._event_info_values_of_continuous_states_changed = state._internal_state_variables["event_info.values_of_continuous_states_changed"]
        if state._internal_state_variables["event_info.next_event_time_defined"] is not None:
            self._event_info_next_event_time_defined = state._internal_state_variables["event_info.next_event_time_defined"]
        if state._internal_state_variables["event_info.next_event_time"] is not None:
            self._event_info_next_event_time = state._internal_state_variables["event_info.next_event_time"]


    def free_fmu_state(self, FMUState3 state):
        """ Free a previously saved FMU-state from the memory.

        Parameters::

            state--
                A pointer to the FMU-state to be set free.

        Example::

            state = fmu.get_fmu_state()
            fmu.free_fmu_state(state)

        """
        cdef FMIL3.fmi3_status_t status
        cdef FMIL3.fmi3_FMU_state_t internal_state = state.fmu_state

        if not self._supports_get_set_FMU_state():
            raise FMUException('This FMU does not support get and set FMU-state')

        if internal_state == NULL:
            logging.warning("FMU-state not allocated, skipping 'fmi3FreeFMUState' call.")
            return

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_free_fmu_state(self._fmu, &internal_state)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != FMIL3.fmi3_status_ok:
            raise FMUException(
                'An error occured while trying to free the FMU-state, see the log for possible more information'
            )

        # Memory has been released
        state.fmu_state = NULL
        state._internal_state_variables = {}


    cpdef serialize_fmu_state(self, FMUState3 state):
        """
        Serialize the data referenced by the input argument.

        Parameters::

            state --
                A FMU-state.

        Returns::
            A list with a vector with the serialized FMU-state and internal state values.

        Example::
            state = fmu.get_fmu_state()
            serialized_state = fmu.serialize_fmu_state(state)
        """

        cdef FMIL3.fmi3_status_t status
        cdef FMIL3.bool cap_me, cap_cs

        cdef FMIL.size_t n_bytes
        cdef np.ndarray[FMIL3.fmi3_byte_t, ndim = 1, mode = 'c'] serialized_state

        cap_me = bool(FMIL3.fmi3_import_get_capability(self._fmu, FMIL3.fmi3_me_canSerializeFMUState))
        cap_cs = bool(FMIL3.fmi3_import_get_capability(self._fmu, FMIL3.fmi3_cs_canSerializeFMUState))
        if not cap_me and not cap_cs:
            raise FMUException('This FMU does not support serialization of FMU-state')

        n_bytes = self.serialized_fmu_state_size(state)
        serialized_state = np.empty(n_bytes, dtype = np.uint8)

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_serialize_fmu_state(self._fmu, state.fmu_state, <FMIL3.fmi3_byte_t*> serialized_state.data, n_bytes)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != FMIL3.fmi3_status_ok:
            raise FMUException('An error occured while serializing the FMU-state, see the log for possible more information')

        # We temporarily return a list with wrapper values in the second entry.
        # What we need to do is add serialization/deserialization for the wrapper values
        return [serialized_state, list(state._internal_state_variables.values())]


    cpdef deserialize_fmu_state(self, list serialized_state):
        """
        De-serialize the provided byte-vector and returns the corresponding FMU-state.

        Parameters::

            serialized_state--
                A serialized FMU-state.

        Returns::

            A deserialized FMU-state.

        Example::

            state = fmu.get_fmu_state()
            serialized_state = fmu.serialize_fmu_state(state)
            deserialized_state = fmu.deserialize_fmu_state(serialized_state)
        """

        cdef FMIL3.fmi3_status_t status
        cdef np.ndarray[FMIL3.fmi3_byte_t, ndim=1, mode='c'] ser_fmu = serialized_state[0]
        cdef FMUState3 state = FMUState3()
        cdef FMIL.size_t n_byte = len(ser_fmu)
        cdef list internal_state_variables = serialized_state[1]

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_de_serialize_fmu_state(self._fmu, <FMIL3.fmi3_byte_t *> ser_fmu.data, n_byte, &(state.fmu_state))
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != FMIL3.fmi3_status_ok:
            raise FMUException('An error occured while deserializing the FMU-state, see the log for possible more information')

        state._internal_state_variables = {'initialized_fmu': internal_state_variables[0],
                                           'has_entered_init_mode': internal_state_variables[1],
                                           'time': internal_state_variables[2],
                                           'callback_log_level': internal_state_variables[3],
                                           'event_info.new_discrete_states_needed': internal_state_variables[4],
                                           'event_info.nominals_of_continuous_states_changed': internal_state_variables[5],
                                           'event_info.terminate_simulation': internal_state_variables[6],
                                           'event_info.values_of_continuous_states_changed': internal_state_variables[7],
                                           'event_info.next_event_time_defined': internal_state_variables[8],
                                           'event_info.next_event_time': internal_state_variables[9]}

        return state

    cpdef serialized_fmu_state_size(self, FMUState3 state):
        """
        Returns the required size of a vector needed to serialize the specified FMU-state

        Parameters::

            state--
                A FMU-state

        Returns::

            The size of the vector.
        """

        cdef FMIL3.fmi3_status_t status
        cdef FMIL.size_t n_bytes

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_serialized_fmu_state_size(self._fmu, state.fmu_state, &n_bytes)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != FMIL3.fmi3_status_ok:
            raise FMUException(
                'An error occured while computing the FMU-state size, see the log for possible more information'
            )

        return n_bytes

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

        if self.get_capability_flags()['needsExecutionTool']:
            raise FMUException("The FMU specifies 'needsExecutionTool=true' which implies that it requires an external execution tool to simulate, this is not supported.")

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

        self.force_finite_differences = False
        self.finite_differences_method = FORWARD_DIFFERENCE

    def _get_fmu_kind(self):
        if self._fmu_kind & FMIL3.fmi3_fmu_kind_me:
            return FMIL3.fmi3_fmu_kind_me
        else:
            raise InvalidVersionException('The FMU could not be loaded. This class only supports FMI 3.0 for Model Exchange.')

    def get_identifier(self):
        if self._modelId is None:
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

    cdef FMIL3.fmi3_status_t _get_event_indicators(self, FMIL3.fmi3_float64_t[:] values):
        cdef FMIL3.fmi3_status_t status = FMIL3.fmi3_status_ok
        if self._nEventIndicators > 0: # do not call if there are no event indicators
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            status = FMIL3.fmi3_import_get_event_indicators(self._fmu, &values[0], self._nEventIndicators)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        return status

    def get_event_indicators(self):
        """
        Returns the event indicators at the current time-point.

        Returns::

            evInd --
                The event indicators as an array.

        Example::

            evInd = model.get_event_indicators()

        Calls the low-level FMI function: fmi3GetEventIndicators
        """
        cdef FMIL3.fmi3_status_t status
        cdef np.ndarray[FMIL3.fmi3_float64_t, ndim=1, mode='c'] values = np.empty(self._nEventIndicators, dtype=np.double)

        status = self._get_event_indicators(values)

        if status != FMIL3.fmi3_status_ok:
            raise FMUException('Failed to get the event indicators at time: %E.'%self.time)

        return values

    def get_event_info(self):
        """
        Returns the event information from the FMU.

        Returns::

            The event information, a class which contains:

            discreteStatesNeedUpdate --
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

            event_info    = model.get_event_info()
            nextEventTime = event_info.nextEventTime
        """
        self._eventInfo.newDiscreteStatesNeeded           = self._event_info_new_discrete_states_needed
        self._eventInfo.terminateSimulation               = self._event_info_terminate_simulation
        self._eventInfo.nominalsOfContinuousStatesChanged = self._event_info_nominals_of_continuous_states_changed
        self._eventInfo.valuesOfContinuousStatesChanged   = self._event_info_values_of_continuous_states_changed
        self._eventInfo.nextEventTimeDefined              = self._event_info_next_event_time_defined
        self._eventInfo.nextEventTime                     = self._event_info_next_event_time
        return self._eventInfo

    def enter_event_mode(self):
        """ Enter event mode by calling the low level FMI function fmi3EnterEventMode. """
        cdef FMIL3.fmi3_status_t status
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_enter_event_mode(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != FMIL3.fmi3_status_ok:
            raise FMUException("Failed to enter event mode")

    def event_update(self, intermediateResult = False):
        """
        Updates the event information at the current time-point. If
        intermediateResult is set to True the update_event will stop at each
        event iteration which would require to loop until
        event_info.discreteStatesNeedUpdate is False.

        Parameters::

            intermediateResult --
                If set to True, the update_event will stop at each event
                iteration.
                Default: False.

        Example::

            model.event_update()

        Calls the low-level FMI function: fmi3UpdateDiscreteStates
        """
        cdef FMIL3.fmi3_status_t status
        cdef FMIL3.fmi3_boolean_t tmp_values_continuous_states_changed   = False
        cdef FMIL3.fmi3_boolean_t tmp_nominals_continuous_states_changed = False

        if intermediateResult:
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            status = FMIL3.fmi3_import_update_discrete_states(
                self._fmu,
                &self._event_info_new_discrete_states_needed,
                &self._event_info_terminate_simulation,
                &self._event_info_nominals_of_continuous_states_changed,
                &self._event_info_values_of_continuous_states_changed,
                &self._event_info_next_event_time_defined,
                &self._event_info_next_event_time)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
            if status != FMIL3.fmi3_status_ok:
                raise FMUException('Failed to update the events at time: %E.'%self.time)
        else:
            self._event_info_new_discrete_states_needed = FMIL3.fmi3_true
            while self._event_info_new_discrete_states_needed:
                self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
                status = FMIL3.fmi3_import_update_discrete_states(
                    self._fmu,
                    &self._event_info_new_discrete_states_needed,
                    &self._event_info_terminate_simulation,
                    &self._event_info_nominals_of_continuous_states_changed,
                    &self._event_info_values_of_continuous_states_changed,
                    &self._event_info_next_event_time_defined,
                    &self._event_info_next_event_time)
                self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

                tmp_values_continuous_states_changed |= self._event_info_nominals_of_continuous_states_changed
                tmp_values_continuous_states_changed |= self._event_info_values_of_continuous_states_changed
                if status != FMIL3.fmi3_status_ok:
                    raise FMUException('Failed to update the events at time: %E.'%self.time)

            # Values changed at least once during event iteration
            self._event_info_nominals_of_continuous_states_changed |= tmp_values_continuous_states_changed
            self._event_info_values_of_continuous_states_changed   |= tmp_nominals_continuous_states_changed

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

    def get_capability_flags(self) -> dict:
        """
        Returns a dictionary with the capability flags of the FMU.

        Returns::
            Dictionary with keys:
            needsExecutionTool
            canBeInstantiatedOnlyOncePerProcess
            canGetAndSetFMUState
            canSerializeFMUState
            providesDirectionalDerivatives
            providesAdjointDerivatives
            providesPerElementDependencies
            providesEvaluateDiscreteStates
            needsCompletedIntegratorStep
        """
        cdef dict capabilities = {}
        capabilities['needsExecutionTool']                  = bool(FMIL3.fmi3_import_get_capability(self._fmu, FMIL3.fmi3_me_needsExecutionTool))
        capabilities['canBeInstantiatedOnlyOncePerProcess'] = bool(FMIL3.fmi3_import_get_capability(self._fmu, FMIL3.fmi3_me_canBeInstantiatedOnlyOncePerProcess))
        capabilities['canGetAndSetFMUState']                = bool(FMIL3.fmi3_import_get_capability(self._fmu, FMIL3.fmi3_me_canGetAndSetFMUState))
        capabilities['canSerializeFMUState']                = bool(FMIL3.fmi3_import_get_capability(self._fmu, FMIL3.fmi3_me_canSerializeFMUState))
        capabilities['providesDirectionalDerivatives']      = bool(FMIL3.fmi3_import_get_capability(self._fmu, FMIL3.fmi3_me_providesDirectionalDerivatives))
        capabilities['providesAdjointDerivatives']          = bool(FMIL3.fmi3_import_get_capability(self._fmu, FMIL3.fmi3_me_providesAdjointDerivatives))
        capabilities['providesPerElementDependencies']      = bool(FMIL3.fmi3_import_get_capability(self._fmu, FMIL3.fmi3_me_providesPerElementDependencies))
        capabilities['providesEvaluateDiscreteStates']      = bool(FMIL3.fmi3_import_get_capability(self._fmu, FMIL3.fmi3_me_providesEvaluateDiscreteStates))
        capabilities['needsCompletedIntegratorStep']        = bool(FMIL3.fmi3_import_get_capability(self._fmu, FMIL3.fmi3_me_needsCompletedIntegratorStep))

        return capabilities

    @functools.cache
    def _provides_directional_derivatives(self) -> bool:
        """Check capability to provide directional derivatives."""
        return bool(FMIL3.fmi3_import_get_capability(self._fmu, FMIL3.fmi3_me_providesDirectionalDerivatives))

    @functools.cache
    def _supports_get_set_FMU_state(self) -> bool:
        """Returns True if the FMU supports get and set FMU-state, otherwise False."""
        return bool(FMIL3.fmi3_import_get_capability(self._fmu, FMIL3.fmi3_me_canGetAndSetFMUState))

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

    def _get_directional_proxy(self, var_ref, func_ref, group, add_diag = False, output_matrix = None):
        if not self._has_entered_init_mode:
            raise FMUException("The FMU has not entered initialization mode and thus the directional " \
                               "derivatives cannot be computed. Call enter_initialization_mode to start the initialization.")
        if self._provides_directional_derivatives() and not self.force_finite_differences:
            return FMUModelBase3._get_directional_proxy(self, var_ref, func_ref, group, add_diag, output_matrix)
        else:
            return self._estimate_directional_derivative(var_ref, func_ref, group, add_diag, output_matrix)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def _estimate_directional_derivative(self, var_ref, func_ref, dict group = None, add_diag = False, output_matrix = None):
        cdef list data = [], row = [], col = []
        cdef int sol_found = 0, dim = 0, i, j, len_v = len(var_ref), len_f = len(func_ref), local_indices_vars_nbr, status
        cdef double tmp_nominal, fac, tmp
        cdef int method = self.finite_differences_method
        cdef double RUROUND = FORWARD_DIFFERENCE_EPS if method == FORWARD_DIFFERENCE else CENTRAL_DIFFERENCE_EPS
        cdef np.ndarray[FMIL3.fmi3_float64_t, ndim=1, mode='c'] dfpert, df, eps, nominals
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] v_ref = np.asarray(var_ref, dtype = np.uint32)
        cdef np.ndarray[FMIL3.fmi3_value_reference_t, ndim=1, mode='c'] z_ref = np.asarray(func_ref, dtype = np.uint32)
        cdef int ind_local = 5 if add_diag else 4 # index in group data vectors
        cdef list local_group

        cdef FMIL3.fmi3_float64_t *column_data_pt
        cdef FMIL3.fmi3_float64_t *v_pt
        cdef FMIL3.fmi3_float64_t *df_pt
        cdef FMIL3.fmi3_float64_t *eps_pt
        cdef FMIL3.fmi3_float64_t *tmp_val_pt
        cdef FMIL3.fmi3_float64_t *output_matrix_data_pt = NULL
        cdef FMIL3.fmi3_value_reference_t *v_ref_pt = <FMIL3.fmi3_value_reference_t*>PyArray_DATA(v_ref)
        cdef FMIL3.fmi3_value_reference_t *z_ref_pt = <FMIL3.fmi3_value_reference_t*>PyArray_DATA(z_ref)
        cdef FMIL3.fmi3_value_reference_t *local_v_vref_pt
        cdef FMIL3.fmi3_value_reference_t *local_z_vref_pt
        cdef int* local_indices_vars_pt
        cdef int* local_indices_matrix_rows_pt
        cdef int* local_indices_matrix_columns_pt
        cdef int* local_data_indices
        cdef FMIL3.fmi3_float64_t* nominals_pt

        if (method != FORWARD_DIFFERENCE) and (method != CENTRAL_DIFFERENCE):
            raise FMUException("Invalid 'finite_differences_method' for FMUModelME3, must be FORWARD_DIFFERENCE (1) or CENTRAL_DIFFERENCE (2).")

        # Make sure that the work vectors has the correct lengths
        self._worker_object.verify_dimensions(max(len_v, len_f))

        # Get work vectors
        df_pt      = self._worker_object.get_real_vector(0)
        df         = self._worker_object.get_real_numpy_vector(0) # TODO: Should be removed in the future
        v_pt       = self._worker_object.get_real_vector(1)
        eps_pt     = self._worker_object.get_real_vector(2)
        eps        = self._worker_object.get_real_numpy_vector(2) # TODO: Should be removed in the future
        tmp_val_pt = self._worker_object.get_real_vector(3)

        local_v_vref_pt = self._worker_object.get_value_reference_vector(0)
        local_z_vref_pt = self._worker_object.get_value_reference_vector(1)

        # Get updated values for the derivatives and states
        self._get_float64(z_ref_pt, len_f, df_pt)
        self._get_float64(v_ref_pt, len_v, v_pt)

        if group is not None:
            if "nominals" in group: # Re-use extracted nominals
                nominals = group["nominals"]
                nominals_pt = <FMIL3.fmi3_float64_t*>PyArray_DATA(nominals)
            else: # First time extraction of nominals
                # TODO: If we are using the states, then the nominals should instead be picked up from the C callback function for nominals
                if self._states_references and len_v == len(self._states_references) and (self._states_references[i] == var_ref[i] for i in range(len_v)):
                    group["nominals"] = np.array(self.nominal_continuous_states, dtype = float)
                    nominals = group["nominals"]
                    nominals_pt = <FMIL3.fmi3_float64_t*>PyArray_DATA(nominals)
                else:
                    group["nominals"] = np.empty(len_v, dtype = float)
                    nominals = group["nominals"]
                    nominals_pt = <FMIL3.fmi3_float64_t*>PyArray_DATA(nominals)
                    for i in range(len_v):
                        nominals_pt[i] = self.get_variable_nominal(valueref = v_ref_pt[i])

            for i in range(len_v):
                eps_pt[i] = RUROUND*(max(abs(v_pt[i]), nominals_pt[i]))
        else:
            for i in range(len_v):
                tmp_nominal = self.get_variable_nominal(valueref = v_ref_pt[i])
                eps_pt[i] = RUROUND*(max(abs(v_pt[i]), tmp_nominal))

        if group is not None:
            if output_matrix is not None:
                if not isinstance(output_matrix, sps.csc_matrix):
                    output_matrix = None
                else:
                    output_matrix_data_pt = <FMIL3.fmi3_float64_t*>PyArray_DATA(output_matrix.data)

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

                # Structure of a local group
                # - [0] - variable indexes
                # - [1] - variable names
                # - [2] - matrix rows
                # - [3] - matrix columns
                # - [4] - position in data vector (CSC format)
                # - [5] - position in data vector (with diag) (CSC format)

                # Get the local value references for the derivatives and states corresponding to the current group
                for i in range(local_indices_vars_nbr):        local_v_vref_pt[i] = v_ref_pt[local_indices_vars_pt[i]]
                for i in range(local_indices_matrix_rows_nbr): local_z_vref_pt[i] = z_ref_pt[local_indices_matrix_rows_pt[i]]

                for fac in [1.0, 0.1, 0.01, 0.001]: # In very special cases, the epsilon is too big, if an error, try to reduce eps
                    for i in range(local_indices_vars_nbr): tmp_val_pt[i] = v_pt[local_indices_vars_pt[i]]+fac*eps_pt[local_indices_vars_pt[i]]
                    self._set_float64(local_v_vref_pt, tmp_val_pt, local_indices_vars_nbr)

                    if method == FORWARD_DIFFERENCE: # Forward and Backward difference
                        column_data_pt = tmp_val_pt

                        status = self._get_float64(local_z_vref_pt, local_indices_matrix_rows_nbr, tmp_val_pt)
                        if status == 0:
                            for i in range(local_indices_matrix_rows_nbr):
                                column_data_pt[i] = (tmp_val_pt[i] - df_pt[local_indices_matrix_rows_pt[i]])/(fac*eps_pt[local_indices_matrix_columns_pt[i]])

                            sol_found = 1
                        else: # Backward

                            for i in range(local_indices_vars_nbr): tmp_val_pt[i] = v_pt[local_indices_vars_pt[i]]-fac*eps_pt[local_indices_vars_pt[i]]
                            self._set_float64(local_v_vref_pt, tmp_val_pt, local_indices_vars_nbr)

                            status = self._get_float64(local_z_vref_pt, local_indices_matrix_rows_nbr, tmp_val_pt)
                            if status == 0:
                                for i in range(local_indices_matrix_rows_nbr):
                                    column_data_pt[i] = (df_pt[local_indices_matrix_rows_pt[i]] - tmp_val_pt[i])/(fac*eps_pt[local_indices_matrix_columns_pt[i]])

                                sol_found = 1

                    else: # Central difference
                        dfpertp = self.get_float64(z_ref[local_group[2]])

                        for i in range(local_indices_vars_nbr): tmp_val_pt[i] = v_pt[local_indices_vars_pt[i]]-fac*eps_pt[local_indices_vars_pt[i]]
                        self._set_float64(local_v_vref_pt, tmp_val_pt, local_indices_vars_nbr)

                        dfpertm = self.get_float64(z_ref[local_group[2]])

                        column_data = (dfpertp - dfpertm)/(2*fac*eps[local_group[3]])
                        column_data_pt = <FMIL3.fmi3_float64_t*>PyArray_DATA(column_data)
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
                self._set_float64(local_v_vref_pt, tmp_val_pt, local_indices_vars_nbr)

            if output_matrix is not None:
                A = output_matrix
            else:
                if len(data) == 0:
                    A = sps.csc_matrix((len_f,len_v))
                else:
                    A = sps.csc_matrix((data, (row, col)), (len_f,len_v))

            return A
        else:
            if output_matrix is None or \
                (not isinstance(output_matrix, np.ndarray)) or \
                (isinstance(output_matrix, np.ndarray) and (output_matrix.shape[0] != len_f or output_matrix.shape[1] != len_v)):
                    A = np.zeros((len_f,len_v))
            else:
                A = output_matrix

            if len_v == 0 or len_f == 0:
                return A

            dfpert = np.zeros(len_f, dtype = np.double)
            df = df[:len_f] # TODO:Should be removed in the future
            for i in range(len_v):
                tmp = v_pt[i]
                for fac in [1.0, 0.1, 0.01, 0.001]: # In very special cases, the epsilon is too big, if an error, try to reduce eps
                    v_pt[i] = tmp+fac*eps_pt[i]
                    self._set_float64(v_ref_pt, v_pt, len_v)

                    if method == FORWARD_DIFFERENCE: # Forward and Backward difference
                        try:
                            dfpert = self.get_float64(z_ref)
                            A[:, i] = (dfpert - df)/(fac*eps_pt[i])
                            break
                        except FMUException: # Try backward difference
                            v_pt[i] = tmp - fac*eps_pt[i]
                            self._set_float64(v_ref_pt, v_pt, len_v)
                            try:
                                dfpert = self.get_float64(z_ref)
                                A[:, i] = (df - dfpert)/(fac*eps_pt[i])
                                break
                            except FMUException:
                                pass

                    else: # Central difference
                        dfpertp = self.get_float64(z_ref)
                        v_pt[i] = tmp - fac*eps_pt[i]
                        self._set_float64(v_ref_pt, v_pt, len_v)
                        dfpertm = self.get_float64(z_ref)
                        A[:, i] = (dfpertp - dfpertm)/(2*fac*eps_pt[i])
                        break
                else:
                    raise FMUException("Failed to estimate the directional derivative at time %g."%self.time)

                # Reset values
                v_pt[i] = tmp
                self._set_float64(v_ref_pt, v_pt, len_v)

            return A


cdef class _WorkerClass3:
    """Internal helper class used in estimating directional derivatives."""

    def __init__(self):
        self._dim = 0

    def _update_work_vectors(self, dim):
        self._tmp1_val = np.zeros(dim, dtype = np.double)
        self._tmp2_val = np.zeros(dim, dtype = np.double)
        self._tmp3_val = np.zeros(dim, dtype = np.double)
        self._tmp4_val = np.zeros(dim, dtype = np.double)

        self._tmp1_ref = np.zeros(dim, dtype = np.uint32)
        self._tmp2_ref = np.zeros(dim, dtype = np.uint32)
        self._tmp3_ref = np.zeros(dim, dtype = np.uint32)
        self._tmp4_ref = np.zeros(dim, dtype = np.uint32)

    cpdef verify_dimensions(self, int dim):
        if dim > self._dim:
            self._update_work_vectors(dim)

    cdef np.ndarray get_real_numpy_vector(self, int index):
        cdef np.ndarray ret = None

        if index == 0:
            ret = self._tmp1_val
        elif index == 1:
            ret = self._tmp2_val
        elif index == 2:
            ret = self._tmp3_val
        elif index == 3:
            ret = self._tmp4_val

        return ret

    cdef FMIL3.fmi3_float64_t* get_real_vector(self, int index):
        cdef FMIL3.fmi3_float64_t* ret = NULL
        if index == 0:
            ret = <FMIL3.fmi3_float64_t*>PyArray_DATA(self._tmp1_val)
        elif index == 1:
            ret = <FMIL3.fmi3_float64_t*>PyArray_DATA(self._tmp2_val)
        elif index == 2:
            ret = <FMIL3.fmi3_float64_t*>PyArray_DATA(self._tmp3_val)
        elif index == 3:
            ret = <FMIL3.fmi3_float64_t*>PyArray_DATA(self._tmp4_val)

        return ret

    cdef np.ndarray get_value_reference_numpy_vector(self, int index):
        cdef np.ndarray ret = None

        if index == 0:
            ret = self._tmp1_ref
        elif index == 1:
            ret = self._tmp2_ref
        elif index == 2:
            ret = self._tmp3_ref
        elif index == 3:
            ret = self._tmp4_ref

        return ret

    cdef FMIL3.fmi3_value_reference_t* get_value_reference_vector(self, int index):
        cdef FMIL3.fmi3_value_reference_t* ret = NULL
        if index == 0:
            ret = <FMIL3.fmi3_value_reference_t*>PyArray_DATA(self._tmp1_ref)
        elif index == 1:
            ret = <FMIL3.fmi3_value_reference_t*>PyArray_DATA(self._tmp2_ref)
        elif index == 2:
            ret = <FMIL3.fmi3_value_reference_t*>PyArray_DATA(self._tmp3_ref)
        elif index == 3:
            ret = <FMIL3.fmi3_value_reference_t*>PyArray_DATA(self._tmp4_ref)

        return ret

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
