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

# Module containing the FMI3 interface Python wrappers.

import numpy as np
cimport numpy as np

cimport pyfmi.fmil_import as FMIL
cimport pyfmi.fmil3_import as FMIL3
cimport pyfmi.fmi_base as FMI_BASE

cdef class FMUModelBase3(FMI_BASE.ModelBase):
    # FMIL related variables
    cdef FMIL.fmi_import_context_t* _context
    cdef FMIL3.fmi3_import_t*       _fmu
    cdef FMIL3.fmi3_fmu_kind_enu_t  _fmu_kind
    cdef FMIL.fmi_version_enu_t     _version


    # Internal values
    cdef public float  _last_accepted_time
    cdef public object _enable_logging
    cdef object     _fmu_full_path
    cdef object     _modelName
    cdef object     _t
    cdef int _allow_unzipped_fmu
    cdef int _allocated_context, _allocated_dll, _allocated_fmu, _allocated_xml

    cdef int _initialized_fmu
    cdef object  _has_entered_init_mode # this is public in FMI2 but I don't see why

    cpdef set_float64(self, valueref, values)
    cpdef set_float32(self, valueref, values)

    cpdef np.ndarray get_float64(self, valueref)
    cpdef np.ndarray get_float32(self, valueref)

    cpdef FMIL3.fmi3_value_reference_t get_variable_valueref(self, variablename) except *
    cpdef FMIL3.fmi3_base_type_enu_t get_variable_data_type(self, variable_name) except *

cdef class FMUModelME3(FMUModelBase3):
    cdef FMIL.size_t _nEventIndicators
    cdef FMIL.size_t _nContinuousStates
    cdef int _get_continuous_states_fmil(self, FMIL3.fmi3_float64_t[:] ndx)
    cdef int _set_continuous_states_fmil(self, FMIL3.fmi3_float64_t[:] ndx)

cdef void _cleanup_on_load_error(
    FMIL3.fmi3_import_t* fmu_3,
    FMIL.fmi_import_context_t* context,
    int allow_unzipped_fmu,
    FMIL.jm_callbacks callbacks,
    bytes fmu_temp_dir,
    list log_data
)

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
)
