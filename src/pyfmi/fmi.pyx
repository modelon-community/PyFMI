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

# Module containing the FMI interface Python wrappers.
"""
For profiling:
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
import os
cimport cython

cimport pyfmi.fmil_import as FMIL

cimport pyfmi.fmi_base as FMI_BASE
cimport pyfmi.fmi1 as FMI1
cimport pyfmi.fmi2 as FMI2
cimport pyfmi.fmi3 as FMI3
cimport pyfmi.util as pyfmi_util

from pyfmi.common.core import create_temp_dir

from pyfmi.exceptions import (
    FMUException,
    IOException,
    InvalidOptionException,
    TimeLimitExceeded,
    InvalidFMUException,
    InvalidXMLException,
    InvalidBinaryException,
    InvalidVersionException
)

from pyfmi.fmi_base import (
    ModelBase,
    LogHandler,
    LogHandlerDefault,
    PyEventInfo,
    FMI_DEFAULT_LOG_LEVEL,
    check_fmu_args,
    _handle_load_fmu_exception
)

from pyfmi.fmi1 import (
    # Classes
    ScalarVariable,
    FMUModelBase,
    FMUModelCS1,
    FMUModelME1,
    # Basic flags related to FMI1
    FMI_TRUE,
    FMI_FALSE,
    # Status
    FMI_OK,
    FMI_WARNING,
    FMI_DISCARD,
    FMI_ERROR,
    FMI_FATAL,
    FMI_PENDING,
    FMI1_DO_STEP_STATUS,
    FMI1_PENDING_STATUS,
    FMI1_LAST_SUCCESSFUL_TIME,
    # Types
    FMI_REAL,
    FMI_INTEGER,
    FMI_BOOLEAN,
    FMI_STRING,
    FMI_ENUMERATION,
    # Alias data
    FMI_NO_ALIAS,
    FMI_ALIAS,
    FMI_NEGATED_ALIAS,
    # Variability
    FMI_CONTINUOUS,
    FMI_CONSTANT,
    FMI_PARAMETER,
    FMI_DISCRETE,
    # Causality
    FMI_INPUT,
    FMI_OUTPUT,
    FMI_INTERNAL,
    FMI_NONE,
    # Misc 
    FMI_ME,
    FMI_CS_STANDALONE,
    FMI_CS_TOOL, 
    GLOBAL_FMU_OBJECT,
    FMI_REGISTER_GLOBALLY,
    GLOBAL_LOG_LEVEL,
)

from pyfmi.fmi2 import (
    # Classes
    ScalarVariable2,
    DeclaredType2,
    EnumerationType2,
    IntegerType2,
    RealType2,
    FMUState2,
    WorkerClass2,
    FMUModelBase2,
    FMUModelME2,
    FMUModelCS2,
    # Basic flags related to FMI
    FMI2_TRUE,
    FMI2_FALSE,
    # Status
    FMI2_DO_STEP_STATUS,
    FMI2_PENDING_STATUS,
    FMI2_LAST_SUCCESSFUL_TIME,
    FMI2_TERMINATED,
    # Types
    FMI2_REAL,
    FMI2_INTEGER,
    FMI2_BOOLEAN,
    FMI2_STRING,
    FMI2_ENUMERATION,
    # Variability
    FMI2_CONSTANT,
    FMI2_FIXED,
    FMI2_TUNABLE,
    FMI2_DISCRETE,
    FMI2_CONTINUOUS,
    FMI2_UNKNOWN,
    # Causality
    FMI2_INPUT,
    FMI2_OUTPUT,
    FMI2_PARAMETER, 
    FMI2_CALCULATED_PARAMETER,
    FMI2_LOCAL,
    FMI2_INDEPENDENT,
    # Dependency
    FMI2_KIND_DEPENDENT,
    FMI2_KIND_CONSTANT,
    FMI2_KIND_FIXED, 
    FMI2_KIND_TUNABLE,
    FMI2_KIND_DISCRETE,
    # Initial
    FMI2_INITIAL_EXACT,
    FMI2_INITIAL_APPROX,
    FMI2_INITIAL_CALCULATED,
    FMI2_INITIAL_UNKNOWN,
    # Jacobian approximation
    FORWARD_DIFFERENCE_EPS,
    CENTRAL_DIFFERENCE_EPS,
    # Flags for evaluation of FMI Jacobians
    FMI_STATES,
    FMI_INPUTS,
    FMI_DERIVATIVES,
    FMI_OUTPUTS,
)

from pyfmi.fmi3 import (
    # Classes
    FMUModelBase3,
    FMUModelME3,
)

# Callbacks
cdef void importlogger_load_fmu(FMIL.jm_callbacks* c, FMIL.jm_string module, FMIL.jm_log_level_enu_t log_level, FMIL.jm_string message):
    (<list>c.context).append("FMIL: module = %s, log level = %d: %s"%(module, log_level, message))

cpdef load_fmu(fmu, log_file_name = "", kind = 'auto',
               log_level = FMI_DEFAULT_LOG_LEVEL, allow_unzipped_fmu = False):
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
            needed if a FMU contains multiple models.
            Available options:
                - 'ME'
                - 'CS'
                - 'SE'
                - 'auto'
            Default: 'auto' (Chooses ME > CS > SE, if multiple are available)

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
    # TODO: Tons of duplicated code here for error handling

    # FMIL related variables
    cdef FMIL.fmi_import_context_t* context
    cdef FMIL.jm_callbacks          callbacks
    cdef FMIL.jm_string             last_error
    cdef FMIL.fmi_version_enu_t     version
    cdef list                       log_data = []

    # Variables for deallocation
    fmu_temp_dir = None
    model        = None

    fmu_full_path = os.path.abspath(fmu)
    check_fmu_args(allow_unzipped_fmu, fmu, fmu_full_path)

    # Check that kind-argument is well-defined
    _allowed_kinds = ["ME", "CS", "SE"]
    if (not kind.lower() == "auto") and (kind.upper() not in _allowed_kinds):
        raise FMUException('Input-argument "kind" can only be "ME", "CS", "SE" or "auto" (default) and not: ' + kind)

    # Specify FMI related callbacks
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

    # Get the FMI version of the provided model
    fmu_temp_dir = pyfmi_util.encode(fmu) if allow_unzipped_fmu else pyfmi_util.encode(create_temp_dir())
    fmu_full_path = pyfmi_util.encode(fmu_full_path)
    version = FMI_BASE.import_and_get_version(context, fmu_full_path, fmu_temp_dir, allow_unzipped_fmu)

    # Check the version & parse XML
    if version == FMIL.fmi_version_1_enu:
        model = FMI1._load_fmi1_fmu(
            fmu, log_file_name, kind, log_level, allow_unzipped_fmu,
            context, fmu_temp_dir, callbacks, log_data
        )
    elif version == FMIL.fmi_version_2_0_enu:
        model = FMI2._load_fmi2_fmu(
            fmu, log_file_name, kind, log_level, allow_unzipped_fmu,
            context, fmu_temp_dir, callbacks, log_data
        )
    elif version == FMIL.fmi_version_3_0_enu:
        model = FMI3._load_fmi3_fmu(
            fmu, log_file_name, kind, log_level, allow_unzipped_fmu,
            context, fmu_temp_dir, callbacks, log_data
        )
    else:
        last_error = FMIL.jm_get_last_error(&callbacks)
        FMIL.fmi_import_free_context(context)
        if not allow_unzipped_fmu:
            FMIL.fmi_import_rmdir(&callbacks, fmu_temp_dir)
        if callbacks.log_level >= FMIL.jm_log_level_error:
            _handle_load_fmu_exception(log_data)
            raise InvalidVersionException("The FMU could not be loaded. The FMU version is unsupported. " + pyfmi_util.decode(last_error))
        else:
            _handle_load_fmu_exception(log_data)
            raise InvalidVersionException("The FMU could not be loaded. The FMU version is unsupported. Enable logging for possibly more information.")

    return model
