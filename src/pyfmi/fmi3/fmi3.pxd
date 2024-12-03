#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2024 Modelon AB
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
Module containing the FMI3 interface Python wrappers.
"""
import numpy as np
cimport numpy as np

from pyfmi.fmi cimport ModelBase, WorkerClass2
cimport pyfmi.fmil_import as FMIL
cimport pyfmi.fmi3.fmil3_import as FMIL3

cdef class FMUModelBase3(ModelBase):
    """FMI3 Model loaded from a dll."""
    # FMIL related variables
    cdef FMIL.fmi_import_context_t*      _context
    # cdef FMIL3.fmi3_callback_functions_t callBackFunctions # TODO: const?
    cdef FMIL3.fmi3_import_t*            _fmu
    cdef FMIL3.fmi3_fmu_kind_enu_t       _fmu_kind
    cdef FMIL.fmi_version_enu_t          _version # TODO: To BASE?
    cdef FMIL3.fmi3_instance_environment_t _instance_environment # TODO: What is this meant to do? # TODO: const?
    cdef FMIL3.fmi3_log_message_callback_ft _log_message # TODO
    # cdef FMIL.jm_string                  last_error
    # cdef FMIL.size_t                     _nEventIndicators
    # cdef FMIL.size_t                     _nContinuousStates
    # cdef FMIL.fmi2_event_info_t          _eventInfo

    # Internal values
    # cdef public float  _last_accepted_time, _relative_tolerance
    cdef double time
    cdef object         _fmu_full_path
    cdef public object  _enable_logging
    cdef int _allow_unzipped_fmu # TODO: Move to FMUBase?
    cdef int _allocated_dll, _allocated_context, _allocated_xml, _allocated_fmu, _initialized_fmu
    cdef object         _modelName
    # cdef public list    _save_real_variables_val
    # cdef public list    _save_int_variables_val
    # cdef public list    _save_bool_variables_val
    cdef object         _t
    # cdef public object  _pyEventInfo
    cdef char* _fmu_temp_dir
    # cdef object         _states_references
    # cdef object         _inputs_references
    # cdef object         _outputs_references
    # cdef object         _derivatives_references
    # cdef object         _derivatives_states_dependencies
    # cdef object         _derivatives_inputs_dependencies
    # cdef object         _derivatives_states_dependencies_kind
    # cdef object         _derivatives_inputs_dependencies_kind
    # cdef object         _outputs_states_dependencies
    # cdef object         _outputs_inputs_dependencies
    # cdef object         _outputs_states_dependencies_kind
    # cdef object         _outputs_inputs_dependencies_kind
    # cdef object         _A, _B, _C, _D
    # cdef public object  _group_A, _group_B, _group_C, _group_D
    # cdef object         _mask_A
    # cdef object         _A_row_ind, _A_col_ind
    cdef public object  _has_entered_init_mode
    # cdef WorkerClass2 _worker_object

cdef class FMUModelCS3(FMUModelBase3):
    """FMI3 CoSimulation Model loaded from a dll."""
    pass

cdef class FMUModelME3(FMUModelBase3):
    """FMI3 ModelExchange Model loaded from a dll."""
    pass
