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

cimport pyfmi.fmil_import as FMIL
cimport pyfmi.fmil3_import as FMIL3
cimport pyfmi.fmi_base as FMI_BASE
cimport pyfmi.util as pyfmi_util

from pyfmi.exceptions import (
    FMUException,
    InvalidXMLException,
    InvalidFMUException,
    InvalidVersionException
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

        # Internal values
        self._enable_logging = False

        # Specify the general callback functions
        self.callbacks.malloc  = FMIL.malloc
        self.callbacks.calloc  = FMIL.calloc
        self.callbacks.realloc = FMIL.realloc
        self.callbacks.free    = FMIL.free
        self.callbacks.logger  = importlogger3
        self.callbacks.context = <void*>self
        # Specify FMI3 related callbacks
        # TODO

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

        #Get the FMI version of the provided model
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

        self._modelId = "TODO" # TODO

        if not isinstance(log_file_name, str):
            self._set_log_stream(log_file_name)
            for i in range(len(self._log)):
                self._log_stream.write("FMIL: module = %s, log level = %d: %s\n"%(self._log[i][0], self._log[i][1], self._log[i][2]))
        else:
            fmu_log_name = pyfmi_util.encode((self._modelId + "_log.txt") if log_file_name=="" else log_file_name)
            self._fmu_log_name = <char*>FMIL.malloc((FMIL.strlen(fmu_log_name)+1)*sizeof(char))
            FMIL.strcpy(self._fmu_log_name, fmu_log_name)

            #Create the log file
            with open(self._fmu_log_name,'w') as file:
                for i in range(len(self._log)):
                    file.write("FMIL: module = %s, log level = %d: %s\n"%(self._log[i][0], self._log[i][1], self._log[i][2]))

        self._log = []

    def __dealloc__(self):
        """
        Deallocate memory
        """
        self._invoked_dealloc = 1

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

    def _get_fmu_kind(self):
        raise FMUException("FMUModelBase3 cannot be used directly, use FMUModelME3.")

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

    def _get_fmu_kind(self):
        if self._fmu_kind & FMIL3.fmi3_fmu_kind_me:
            return FMIL3.fmi3_fmu_kind_me
        else:
            raise InvalidVersionException('The FMU could not be loaded. This class only supports FMI 3.0 for Model Exchange.')


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
