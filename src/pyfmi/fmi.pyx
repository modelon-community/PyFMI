#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2014-2023 Modelon AB
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
cimport cython

cimport pyfmi.fmil_import as FMIL
cimport pyfmi.fmil1_import as FMIL1
cimport pyfmi.fmil2_import as FMIL2

cimport pyfmi.fmi_base as FMI_BASE
cimport pyfmi.fmi1 as FMI1
cimport pyfmi.fmi2 as FMI2

from pyfmi.common.core import create_temp_dir

from pyfmi.exceptions import (
    FMUException, IOException, InvalidOptionException, TimeLimitExceeded,
    InvalidFMUException, InvalidXMLException, InvalidBinaryException,
    InvalidVersionException
)

from pyfmi.fmi_base import (
    ModelBase, LogHandler, LogHandlerDefault, PyEventInfo,
    FMI_DEFAULT_LOG_LEVEL, enable_caching, check_fmu_args
)

from pyfmi.fmi1 import (
    # Classes
    ScalarVariable, FMUModelBase, FMUModelCS1, FMUModelME1,
    # Basic flags related to FMI1
    FMI_TRUE, FMI_FALSE,
    # Status
    FMI_OK, FMI_WARNING, FMI_DISCARD,
    FMI_ERROR, FMI_FATAL, FMI_PENDING,
    FMI1_DO_STEP_STATUS, FMI1_PENDING_STATUS, FMI1_LAST_SUCCESSFUL_TIME,
    # Types
    FMI_REAL, FMI_INTEGER, FMI_BOOLEAN, FMI_STRING, FMI_ENUMERATION,
    # Alias data
    FMI_NO_ALIAS, FMI_ALIAS, FMI_NEGATED_ALIAS,
    # Variability
    FMI_CONTINUOUS, FMI_CONSTANT, FMI_PARAMETER, FMI_DISCRETE,
    # Causality
    FMI_INPUT, FMI_OUTPUT, FMI_INTERNAL, FMI_NONE,
    # Misc 
    FMI_ME, FMI_CS_STANDALONE, FMI_CS_TOOL, 
    GLOBAL_FMU_OBJECT, FMI_REGISTER_GLOBALLY, GLOBAL_LOG_LEVEL
)

from pyfmi.fmi2 import (
    # Classes
    ScalarVariable2, DeclaredType2, EnumerationType2,
    IntegerType2, RealType2, FMUState2, WorkerClass2,
    FMUModelBase2, FMUModelME2, FMUModelCS2,
    # Basic flags related to FMI
    FMI2_TRUE, FMI2_FALSE,
    # Status
    FMI2_DO_STEP_STATUS, FMI2_PENDING_STATUS, FMI2_LAST_SUCCESSFUL_TIME, FMI2_TERMINATED,
    # Types
    FMI2_REAL,FMI2_INTEGER, FMI2_BOOLEAN,
    FMI2_STRING, FMI2_ENUMERATION,
    # Variability
    FMI2_CONSTANT, FMI2_FIXED, FMI2_TUNABLE,
    FMI2_DISCRETE, FMI2_CONTINUOUS, FMI2_UNKNOWN,
    # Causality
    FMI2_INPUT, FMI2_OUTPUT, FMI2_PARAMETER, 
    FMI2_CALCULATED_PARAMETER, FMI2_LOCAL, FMI2_INDEPENDENT,
    # Dependency
    FMI2_KIND_DEPENDENT, FMI2_KIND_CONSTANT, FMI2_KIND_FIXED, 
    FMI2_KIND_TUNABLE, FMI2_KIND_DISCRETE,
    # Initial
    FMI2_INITIAL_EXACT, FMI2_INITIAL_APPROX, FMI2_INITIAL_CALCULATED, FMI2_INITIAL_UNKNOWN,
    # Jacobian approximation
    FORWARD_DIFFERENCE_EPS, CENTRAL_DIFFERENCE_EPS,
    # Flags for evaluation of FMI Jacobians
    FMI_STATES, FMI_INPUTS, FMI_DERIVATIVES, FMI_OUTPUTS,
)

# Callbacks
cdef void importlogger_load_fmu(FMIL.jm_callbacks* c, FMIL.jm_string module, FMIL.jm_log_level_enu_t log_level, FMIL.jm_string message):
    (<list>c.context).append("FMIL: module = %s, log level = %d: %s"%(module, log_level, message))

def _handle_load_fmu_exception(fmu, log_data):
    for log in log_data:
        print(log)

cpdef load_fmu(fmu, log_file_name = "", kind = 'auto',
             log_level=FMI_DEFAULT_LOG_LEVEL, allow_unzipped_fmu = False):
    """
    Helper method for creating a model instance.

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

        kind --
            String indicating the kind of model to create. This is only
            needed if a FMU contains both a ME and CS model.
            Available options:
                - 'ME'
                - 'CS'
                - 'auto'
            Default: 'auto' (Chooses ME before CS if both available)

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

        A model instance corresponding to the loaded FMU.
    """
    # TODO: This method can be made more efficient by providing
    # the unzipped part and the already read XML object to the different
    # FMU classes.
    # XXX: Does this require API changes to the respective classes?

    # FMIL related variables
    cdef FMIL.fmi_import_context_t*     context
    cdef FMIL.jm_callbacks              callbacks
    cdef FMIL.jm_string                 last_error
    cdef FMIL.fmi_version_enu_t         version
    cdef FMIL1.fmi1_import_t*           fmu_1 = NULL
    cdef FMIL2.fmi2_import_t*           fmu_2 = NULL
    cdef FMIL1.fmi1_fmu_kind_enu_t      fmu_1_kind
    cdef FMIL2.fmi2_fmu_kind_enu_t      fmu_2_kind
    cdef list                           log_data = []

    #Variables for deallocation
    fmu_temp_dir = None
    model        = None

    fmu_full_path = os.path.abspath(fmu)
    check_fmu_args(allow_unzipped_fmu, fmu, fmu_full_path)

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

    if log_level >= FMIL.jm_log_level_nothing and log_level <= FMIL.jm_log_level_all:
        if log_level == FMIL.jm_log_level_nothing:
            enable_logging = False
        else:
            enable_logging = True
        callbacks.log_level = log_level
    else:
        raise FMUException("The log level must be between %d and %d"%(FMIL.jm_log_level_nothing, FMIL.jm_log_level_all))

    # Create a struct for allocation
    context = FMIL.fmi_import_allocate_context(&callbacks)

    #Get the FMI version of the provided model
    fmu_temp_dir = FMI_BASE.encode(fmu) if allow_unzipped_fmu else FMI_BASE.encode(create_temp_dir())
    fmu_full_path = FMI_BASE.encode(fmu_full_path)
    version = FMI_BASE.import_and_get_version(context, fmu_full_path, fmu_temp_dir, allow_unzipped_fmu)

    #Check the version
    if version == FMIL.fmi_version_unknown_enu:
        #Delete context
        last_error = FMIL.jm_get_last_error(&callbacks)
        FMIL.fmi_import_free_context(context)
        if not allow_unzipped_fmu:
            FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
        if callbacks.log_level >= FMIL.jm_log_level_error:
            _handle_load_fmu_exception(fmu, log_data)
            raise InvalidVersionException("The FMU could not be loaded. The FMU version could not be determined. "+FMI_BASE.decode(last_error))
        else:
            _handle_load_fmu_exception(fmu, log_data)
            raise InvalidVersionException("The FMU could not be loaded. The FMU version could not be determined. Enable logging for possibly more information.")

    if version > 2:
        #Delete the context
        last_error = FMIL.jm_get_last_error(&callbacks)
        FMIL.fmi_import_free_context(context)
        if not allow_unzipped_fmu:
            FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
        if callbacks.log_level >= FMIL.jm_log_level_error:
            _handle_load_fmu_exception(fmu, log_data)
            raise InvalidVersionException("The FMU could not be loaded. The FMU version is unsupported. "+FMI_BASE.decode(last_error))
        else:
            _handle_load_fmu_exception(fmu, log_data)
            raise InvalidVersionException("The FMU could not be loaded. The FMU version is unsupported. Enable logging for possibly more information.")

    #Parse the xml
    if version == FMIL.fmi_version_1_enu:
        #Check the fmu-kind
        fmu_1 = FMIL1.fmi1_import_parse_xml(context, fmu_temp_dir)

        if fmu_1 is NULL:
            #Delete the context
            last_error = FMIL.jm_get_last_error(&callbacks)
            FMIL.fmi_import_free_context(context)
            if not allow_unzipped_fmu:
                FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
            if callbacks.log_level >= FMIL.jm_log_level_error:
                _handle_load_fmu_exception(fmu, log_data)
                raise InvalidXMLException("The FMU could not be loaded. The model data from 'modelDescription.xml' within the FMU could not be read. "+FMI_BASE.decode(last_error))
            else:
                _handle_load_fmu_exception(fmu, log_data)
                raise InvalidXMLException("The FMU could not be loaded. The model data from 'modelDescription.xml' within the FMU could not be read. Enable logging for possible nore information.")

        fmu_1_kind = FMIL1.fmi1_import_get_fmu_kind(fmu_1)

        #Compare fmu_kind with input-specified kind
        if fmu_1_kind == FMI_ME and kind.upper() != 'CS':
            model=FMI1.FMUModelME1(fmu, log_file_name,log_level, _unzipped_dir=fmu_temp_dir,
                              allow_unzipped_fmu = allow_unzipped_fmu)
        elif (fmu_1_kind == FMI_CS_STANDALONE or fmu_1_kind == FMI_CS_TOOL) and kind.upper() != 'ME':
            model=FMI1.FMUModelCS1(fmu, log_file_name,log_level, _unzipped_dir=fmu_temp_dir,
                              allow_unzipped_fmu = allow_unzipped_fmu)
        else:
            FMIL1.fmi1_import_free(fmu_1)
            FMIL.fmi_import_free_context(context)
            if not allow_unzipped_fmu:
                FMIL.fmi_import_rmdir(&callbacks,fmu_temp_dir)
            _handle_load_fmu_exception(fmu, log_data)
            raise FMUException("FMU is a {} and not a {}".format(FMI_BASE.decode(FMIL1.fmi1_fmu_kind_to_string(fmu_1_kind)), kind.upper()))

    elif version == FMIL.fmi_version_2_0_enu:
        #Check fmu-kind and compare with input-specified kind
        fmu_2 = FMIL2.fmi2_import_parse_xml(context, fmu_temp_dir, NULL)

        if fmu_2 is NULL:
            #Delete the context
            last_error = FMIL.jm_get_last_error(&callbacks)
            FMIL.fmi_import_free_context(context)
            if not allow_unzipped_fmu:
                FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
            if callbacks.log_level >= FMIL.jm_log_level_error:
                _handle_load_fmu_exception(fmu, log_data)
                raise InvalidXMLException("The FMU could not be loaded. The model data from 'modelDescription.xml' within the FMU could not be read. "+FMI_BASE.decode(last_error))
            else:
                _handle_load_fmu_exception(fmu, log_data)
                raise InvalidXMLException("The FMU could not be loaded. The model data from 'modelDescription.xml' within the FMU could not be read. Enable logging for possible more information.")

        fmu_2_kind = FMIL2.fmi2_import_get_fmu_kind(fmu_2)

        #FMU kind is unknown
        if fmu_2_kind == FMIL2.fmi2_fmu_kind_unknown:
            last_error = FMIL.jm_get_last_error(&callbacks)
            FMIL2.fmi2_import_free(fmu_2)
            FMIL.fmi_import_free_context(context)
            if not allow_unzipped_fmu:
                FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
            if callbacks.log_level >= FMIL.jm_log_level_error:
                _handle_load_fmu_exception(fmu, log_data)
                raise FMUException("The FMU kind could not be determined. "+FMI_BASE.decode(last_error))
            else:
                _handle_load_fmu_exception(fmu, log_data)
                raise FMUException("The FMU kind could not be determined. Enable logging for possibly more information.")

        #FMU kind is known
        if kind.lower() == 'auto':
            if fmu_2_kind == FMIL2.fmi2_fmu_kind_cs:
                model = FMI2.FMUModelCS2(fmu, log_file_name,log_level, _unzipped_dir=fmu_temp_dir,
                                    allow_unzipped_fmu = allow_unzipped_fmu)
            elif fmu_2_kind == FMIL2.fmi2_fmu_kind_me or fmu_2_kind == FMIL2.fmi2_fmu_kind_me_and_cs:
                model = FMI2.FMUModelME2(fmu, log_file_name,log_level, _unzipped_dir=fmu_temp_dir,
                                    allow_unzipped_fmu = allow_unzipped_fmu)
        elif kind.upper() == 'CS':
            if fmu_2_kind == FMIL2.fmi2_fmu_kind_cs or fmu_2_kind == FMIL2.fmi2_fmu_kind_me_and_cs:
                model = FMI2.FMUModelCS2(fmu, log_file_name,log_level, _unzipped_dir=fmu_temp_dir,
                                    allow_unzipped_fmu = allow_unzipped_fmu)
        elif kind.upper() == 'ME':
            if fmu_2_kind == FMIL2.fmi2_fmu_kind_me or fmu_2_kind == FMIL2.fmi2_fmu_kind_me_and_cs:
                model = FMI2.FMUModelME2(fmu, log_file_name,log_level, _unzipped_dir=fmu_temp_dir,
                                    allow_unzipped_fmu = allow_unzipped_fmu)

        #Could not match FMU kind with input-specified kind
        if model is None:
            FMIL2.fmi2_import_free(fmu_2)
            FMIL.fmi_import_free_context(context)
            if not allow_unzipped_fmu:
                FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
            _handle_load_fmu_exception(fmu, log_data)
            raise FMUException("FMU is a {} and not a {}".format(FMI_BASE.decode(FMIL2.fmi2_fmu_kind_to_string(fmu_2_kind)),  FMI_BASE.decode(kind.upper())))

    else:
        #This else-statement ensures that the variables "context" and "version" are defined before proceeding

        #Delete the context
        last_error = FMIL.jm_get_last_error(&callbacks)
        FMIL.fmi_import_free_context(context)
        if not allow_unzipped_fmu:
            FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
        if callbacks.log_level >= FMIL.jm_log_level_error:
            _handle_load_fmu_exception(fmu, log_data)
            raise InvalidVersionException("The FMU could not be loaded. The FMU version is not found. "+FMI_BASE.decode(last_error))
        else:
            _handle_load_fmu_exception(fmu, log_data)
            raise InvalidVersionException("The FMU could not be loaded. The FMU version is not found. Enable logging for possibly more information.")

    #Delete
    if version == FMIL.fmi_version_1_enu:
        FMIL1.fmi1_import_free(fmu_1)
        FMIL.fmi_import_free_context(context)
        #FMIL.fmi_import_rmdir(&callbacks,fmu_temp_dir)

    if version == FMIL.fmi_version_2_0_enu:
        FMIL2.fmi2_import_free(fmu_2)
        FMIL.fmi_import_free_context(context)
        #FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)

    return model
