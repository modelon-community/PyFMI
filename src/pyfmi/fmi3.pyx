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
import logging

import numpy as np
cimport numpy as np

cimport pyfmi.fmil_import as FMIL
cimport pyfmi.fmil3_import as FMIL3
cimport pyfmi.fmi_base as FMI_BASE
cimport pyfmi.util as pyfmi_util

# TYPES
# TODO: Import into fmi.pyx for convenience imports?
FMI3_FLOAT64 = FMIL3.fmi3_base_type_float64
FMI3_FLOAT32 = FMIL3.fmi3_base_type_float32
FMI3_INT64   = FMIL3.fmi3_base_type_int64
FMI3_INT32   = FMIL3.fmi3_base_type_int32
FMI3_INT16   = FMIL3.fmi3_base_type_int16
FMI3_INT8    = FMIL3.fmi3_base_type_int8
FMI3_UINT64  = FMIL3.fmi3_base_type_uint64
FMI3_UINT32  = FMIL3.fmi3_base_type_uint32
FMI3_UINT16  = FMIL3.fmi3_base_type_uint16
FMI3_UINT8   = FMIL3.fmi3_base_type_uint8
FMI3_BOOL    = FMIL3.fmi3_base_type_bool
FMI3_BINARY  = FMIL3.fmi3_base_type_binary
FMI3_CLOCK   = FMIL3.fmi3_base_type_clock
FMI3_STRING  = FMIL3.fmi3_base_type_str
FMI3_ENUM    = FMIL3.fmi3_base_type_enum

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
            status = FMIL3.fmi3_import_create_dllfmu(self._fmu, self._fmu_kind, NULL, FMIL3.fmi3_log_forwarding)
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

    def terminate(self):
        """
        Calls the FMI function fmi3Terminate() on the FMU.
        After this call, any call to a function changing the state of the FMU will fail.
        """
        cdef FMIL3.fmi3_status_t status

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_terminate(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != 0:
            raise FMUException("Termination of FMU failed, see log for possible more information.")

    def free_instance(self):
        """
        Calls the FMI function fmi3FreeInstance() on the FMU. Note that this is not
        needed generally.
        """
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
        cdef FMIL3.fmi3_base_type_enu_t basetype

        ref = self.get_variable_valueref(variable_name)
        basetype = self.get_variable_data_type(variable_name)

        if basetype == FMIL3.fmi3_base_type_float64:
            self.set_float64([ref], [value])
        elif basetype == FMIL3.fmi3_base_type_float32:
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
        cdef FMIL3.fmi3_base_type_enu_t basetype

        ref  = self.get_variable_valueref(variable_name)
        basetype = self.get_variable_data_type(variable_name)

        if basetype == FMIL3.fmi3_base_type_float64:
            return self.get_float64([ref])
        elif basetype == FMIL3.fmi3_base_type_float32:
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

    cpdef FMIL3.fmi3_base_type_enu_t get_variable_data_type(self, variable_name) except *:
        """
        Get data type of variable.

        Parameter::

            variable_name --
                The name of the variable.

        Returns::

            The type of the variable.
        """
        cdef FMIL3.fmi3_import_variable_t* variable
        cdef FMIL3.fmi3_base_type_enu_t basetype
        variable_name = pyfmi_util.encode(variable_name)
        cdef char* variablename = variable_name

        variable = FMIL3.fmi3_import_get_variable_by_name(self._fmu, variablename)
        if variable == NULL:
            raise FMUException("The variable %s could not be found."%pyfmi_util.decode(variablename))

        basetype = FMIL3.fmi3_import_get_variable_base_type(variable)

        return basetype

    def get_fmil_log_level(self):
        """
        Returns::
            The current FMIL log-level.
        """
        cdef int level
        if self._enable_logging:
            level = self.callbacks.log_level
            return level
        else:
            raise FMUException('Logging is not enabled')

    def get_version(self):
        """ Returns the FMI version of the Model which it was generated according. """
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        cdef FMIL3.fmi3_string_t version = <FMIL3.fmi3_string_t>FMIL3.fmi3_import_get_version(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        return pyfmi_util.decode(version)

    def get_identifier(self):
        """ Return the model identifier, name of binary model file and prefix in
            the C-function names of the model.
        """
        return NotImplementedError

    def get_default_experiment_start_time(self):
        """ Returns the default experiment start time as defined the XML description. """
        return FMIL3.fmi3_import_get_default_experiment_start(self._fmu)

    def get_default_experiment_stop_time(self):
        """ Returns the default experiment stop time as defined the XML description. """
        return FMIL3.fmi3_import_get_default_experiment_stop(self._fmu)

    def get_default_experiment_tolerance(self):
        """ Returns the default experiment tolerance as defined in the XML description. """
        return FMIL3.fmi3_import_get_default_experiment_tolerance(self._fmu)

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
        """
        Enters initialization mode by calling the low level FMI function
        fmi3EnterInitializationMode.

        Note that the method initialize() performs both the enter and
        exit of initialization mode.

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

    def enter_event_mode(self):
        """ Enter event mode by calling the low level FMI function fmi3EnterEventMode. """
        cdef FMIL3.fmi3_status_t status
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_enter_event_mode(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != FMIL3.fmi3_status_ok:
            raise FMUException("Failed to enter event mode")

    cdef int _get_continuous_states_fmil(self, FMIL3.fmi3_float64_t[:] ndx):
        cdef int status
        if self._nContinuousStates > 0:
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            status = FMIL3.fmi3_import_get_continuous_states(self._fmu, &ndx[0] ,self._nContinuousStates)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
            return status
        else:
            return FMIL3.fmi3_status_ok

    def _get_continuous_states(self):
        """
        Returns a vector with the values of the continuous states.

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

    cdef int _set_continuous_states_fmil(self, FMIL3.fmi3_float64_t[:] ndx):
        cdef int status
        if self._nContinuousStates > 0:
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            status = FMIL3.fmi3_import_set_continuous_states(self._fmu, &ndx[0], self._nContinuousStates)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
            return status
        else:
            return FMIL3.fmi3_status_ok

    def _set_continuous_states(self, np.ndarray[FMIL3.fmi3_float64_t, ndim=1, mode="c"] values):
        """
        Set the values of the continuous states.

        Parameters::

            values--
                The new values of the continuous states.
        """
        cdef int status
        cdef np.ndarray[FMIL3.fmi3_float64_t, ndim=1,mode='c'] ndx = values

        if np.size(ndx) != self._nContinuousStates:
            raise FMUException(
                'Failed to set the new continuous states. ' \
                'The number of values are not consistent with the number of '\
                f'continuous states, which are {self._nContinuousStates}.')

        status = self._set_continuous_states_fmil(ndx)

        if status >= 3:
            raise FMUException('Failed to set the new continuous states.')

    continuous_states = property(_get_continuous_states, _set_continuous_states,
        doc=
    """
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
    """
    To reduce some code duplication for various failures in _load_fmi3_fmu.
    """
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
        raise FMUException("FMU is a {} and not a {}".format(pyfmi_util.decode(FMIL3.fmi3_fmu_kind_to_string(fmu_3_kind)),  pyfmi_util.decode(kind.upper())))

    FMIL3.fmi3_import_free(fmu_3)
    FMIL.fmi_import_free_context(context)
    return model
