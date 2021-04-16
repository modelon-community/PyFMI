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
import os
import sys
import logging
import fnmatch
import re
from collections import OrderedDict

import numpy as N
cimport numpy as N

N.import_array()

cimport fmil_import as FMIL

cdef class ModelBase:
    """
    Abstract Model class containing base functionality.
    """
    cdef list _log
    cdef char* _fmu_log_name
    cdef FMIL.jm_callbacks callbacks
    cdef public dict cache
    cdef public object file_object
    cdef public object _additional_logger
    cdef public object _max_log_size_msg_sent
    cdef public unsigned long long int _current_log_size, _max_log_size

    cdef _logger(self, FMIL.jm_string module, int log_level, FMIL.jm_string message) with gil


cdef class ScalarVariable:
    """
    Class defining data structure based on the XML element ScalarVariable.
    """
    cdef object _name
    cdef FMIL.fmi1_value_reference_t _value_reference
    cdef object _description #A characater pointer but we need an own reference and this is sufficient
    cdef FMIL.fmi1_base_type_enu_t _type
    cdef FMIL.fmi1_variability_enu_t _variability
    cdef FMIL.fmi1_causality_enu_t _causality
    cdef FMIL.fmi1_variable_alias_kind_enu_t _alias

cdef class ScalarVariable2:
    """
    Class defining data structure based on the XML element ScalarVariable.
    """
    cdef object         _value_reference
    cdef FMIL.fmi2_base_type_enu_t           _type
    cdef FMIL.fmi2_variability_enu_t         _variability
    cdef FMIL.fmi2_causality_enu_t           _causality
    cdef FMIL.fmi2_variable_alias_kind_enu_t _alias
    cdef FMIL.fmi2_initial_enu_t             _initial
    cdef object _name
    cdef object _description #A characater pointer but we need an own reference and this is sufficient

cdef class DeclaredType2:
    cdef object _name
    cdef object _description
    cdef object _quantity

cdef class EnumerationType2(DeclaredType2):
    cdef object _items

cdef class IntegerType2(DeclaredType2):
    cdef int _min, _max

cdef class RealType2(DeclaredType2):
    cdef float _min, _max, _nominal
    cdef object _unbounded, _relativeQuantity, _unit, _display_unit

cdef class FMUState2:
    """
    Class containing a pointer to a FMU-state.
    """
    cdef FMIL.fmi2_FMU_state_t fmu_state
    cdef dict _internal_state_variables

cdef class FMUModelBase(ModelBase):
    """
    An FMI Model loaded from a DLL.
    """
    #FMIL related variables
    cdef FMIL.fmi1_callback_functions_t callBackFunctions
    cdef FMIL.fmi_import_context_t* context
    cdef FMIL.fmi1_import_t* _fmu
    cdef FMIL.fmi1_event_info_t _eventInfo
    cdef FMIL.fmi1_import_variable_list_t *variable_list
    cdef FMIL.fmi1_fmu_kind_enu_t fmu_kind
    cdef FMIL.jm_status_enu_t jm_status
    cdef FMIL.jm_callbacks callbacks_defaults
    cdef FMIL.jm_callbacks* callbacks_standard

    #Internal values
    cdef public object __t
    cdef public object _file_open
    cdef public object _npoints
    cdef public object _enable_logging
    cdef public object _pyEventInfo
    cdef int _version
    cdef int _instantiated_fmu
    cdef int _allocated_dll, _allocated_context, _allocated_xml, _allocated_fmu
    cdef object _allocated_list
    cdef object _modelid
    cdef object _modelname
    cdef unsigned int _nEventIndicators
    cdef unsigned int _nContinuousStates
    cdef public list _save_real_variables_val
    cdef public list _save_int_variables_val
    cdef public list _save_bool_variables_val
    cdef int _fmu_kind
    cdef char* _fmu_temp_dir

    cpdef _internal_set_fmu_null(self)
    cpdef get_variable_description(self, variablename)
    cpdef FMIL.fmi1_base_type_enu_t get_variable_data_type(self, variablename) except *
    cpdef FMIL.fmi1_value_reference_t get_variable_valueref(self, variablename) except *
    cpdef get_variable_fixed(self, variablename)
    cpdef get_variable_start(self, variablename)
    cpdef get_variable_max(self, variablename)
    cpdef get_variable_min(self, variablename)
    cpdef FMIL.fmi1_variability_enu_t get_variable_variability(self, variablename) except *
    cpdef FMIL.fmi1_causality_enu_t get_variable_causality(self, variablename) except *
    cdef _add_scalar_variable(self, FMIL.fmi1_import_variable_t* variable)

cdef class FMUModelCS1(FMUModelBase):

    cpdef _get_time(self)
    cpdef _set_time(self, FMIL.fmi1_real_t t)

cdef class FMUModelME1(FMUModelBase):

    cpdef _get_time(self)
    cpdef _set_time(self, FMIL.fmi1_real_t t)
    cpdef get_derivatives(self)

cdef class FMUModelBase2(ModelBase):
    """
    FMI Model loaded from a dll.
    """

    #FMIL related variables
    cdef FMIL.fmi_import_context_t*     _context
    cdef FMIL.fmi2_callback_functions_t callBackFunctions
    cdef FMIL.fmi2_import_t*            _fmu
    cdef FMIL.fmi2_fmu_kind_enu_t       _fmu_kind
    cdef FMIL.fmi_version_enu_t         _version
    cdef FMIL.jm_string                 last_error
    cdef FMIL.size_t                    _nEventIndicators
    cdef FMIL.size_t                    _nContinuousStates
    cdef FMIL.fmi2_event_info_t         _eventInfo

    #Internal values
    cdef public float  _last_accepted_time, _relative_tolerance
    cdef object         _fmu_full_path
    cdef public object  _enable_logging
    cdef int _allocated_dll, _allocated_context, _allocated_xml, _allocated_fmu, _initialized_fmu
    cdef object         _modelId
    cdef object         _modelName
    cdef public list    _save_real_variables_val
    cdef public list    _save_int_variables_val
    cdef public list    _save_bool_variables_val
    cdef object         __t
    cdef public object  _pyEventInfo
    cdef char* _fmu_temp_dir
    cdef object         _states_references
    cdef object         _inputs_references
    cdef object         _outputs_references
    cdef object         _derivatives_references
    cdef object         _derivatives_states_dependencies
    cdef object         _derivatives_inputs_dependencies
    cdef object         _derivatives_states_dependencies_kind
    cdef object         _derivatives_inputs_dependencies_kind
    cdef object         _outputs_states_dependencies
    cdef object         _outputs_inputs_dependencies
    cdef object         _outputs_states_dependencies_kind
    cdef object         _outputs_inputs_dependencies_kind
    cdef object         _A, _B, _C, _D
    cdef public object         _group_A, _group_B, _group_C, _group_D
    cdef object         _mask_A
    cdef object         _A_row_ind, _A_col_ind
    cdef public object  _has_entered_init_mode
    cdef WorkerClass2 _worker_object

    cpdef FMIL.fmi2_value_reference_t get_variable_valueref(self, variablename) except *
    cpdef FMIL.fmi2_base_type_enu_t get_variable_data_type(self, variablename) except *
    cpdef get_variable_description(self, variablename)
    cpdef FMIL.fmi2_variability_enu_t get_variable_variability(self, variablename) except *
    cpdef FMIL.fmi2_causality_enu_t get_variable_causality(self, variablename) except *
    cpdef get_output_dependencies(self)
    cpdef get_output_dependencies_kind(self)
    cpdef get_derivatives_dependencies(self)
    cpdef get_derivatives_dependencies_kind(self)
    cpdef get_variable_start(self, variablename)
    cpdef get_variable_max(self, variablename)
    cpdef get_variable_min(self, variablename)
    cpdef FMIL.fmi2_initial_enu_t get_variable_initial(self, variable_name) except *
    cpdef serialize_fmu_state(self, state)
    cpdef deserialize_fmu_state(self, serialized_fmu)
    cpdef serialized_fmu_state_size(self, state)
    cdef _add_scalar_variables(self, FMIL.fmi2_import_variable_list_t*   variable_list)
    cdef _add_scalar_variable(self, FMIL.fmi2_import_variable_t* variable)
    cdef int _get_directional_derivative(self, N.ndarray v_ref, N.ndarray z_ref, N.ndarray dv, N.ndarray dz) except -1
    cpdef set_real(self, valueref, values)
    cpdef N.ndarray get_real(self, valueref)
    cdef int __set_real(self, FMIL.fmi2_value_reference_t* vrefs, FMIL.fmi2_real_t* values, size_t size)
    cdef int __get_real(self, FMIL.fmi2_value_reference_t* vrefs, size_t size, FMIL.fmi2_real_t* values)
    cdef int _get_real(self, FMIL.fmi2_value_reference_t[:] valueref, size_t size, FMIL.fmi2_real_t[:] values)
    cdef int _get_integer(self, FMIL.fmi2_value_reference_t[:] valueref, size_t size, FMIL.fmi2_integer_t[:] values)
    cdef int _get_boolean(self, FMIL.fmi2_value_reference_t[:] valueref, size_t size, FMIL.fmi2_real_t[:] values)

cdef class FMUModelCS2(FMUModelBase2):

    cpdef _get_time(self)
    cpdef _set_time(self, FMIL.fmi2_real_t t)
    cpdef int do_step(self, FMIL.fmi2_real_t current_t, FMIL.fmi2_real_t step_size, new_step=*)
    cdef int _set_input_derivatives(self, N.ndarray value_refs, N.ndarray values, N.ndarray orders)
    cdef int _get_output_derivatives(self, N.ndarray value_refs, N.ndarray values, N.ndarray orders)

cdef class FMUModelME2(FMUModelBase2):

    cpdef _get_time(self)
    cpdef _set_time(self, FMIL.fmi2_real_t t)
    cpdef get_derivatives(self)
    cdef public object force_finite_differences
    cdef int _get_derivatives(self, FMIL.fmi2_real_t[:] values)
    cdef int __get_continuous_states(self, FMIL.fmi2_real_t[:] ndx)
    cdef int __set_continuous_states(self, FMIL.fmi2_real_t[:] ndx)
    cdef int _get_event_indicators(self, FMIL.fmi2_real_t[:] values)
    cdef int _completed_integrator_step(self, int* enter_event_mode, int* terminate_simulation)

cdef class WorkerClass2:
    cdef int _dim

    cdef N.ndarray _tmp1_val, _tmp2_val, _tmp3_val, _tmp4_val
    cdef N.ndarray _tmp1_ref, _tmp2_ref, _tmp3_ref, _tmp4_ref

    cdef FMIL.fmi2_real_t* get_real_vector(self, int index)
    cdef FMIL.fmi2_value_reference_t* get_value_reference_vector(self, int index)
    cdef N.ndarray get_value_reference_numpy_vector(self, int index)
    cdef N.ndarray get_real_numpy_vector(self, int index)
    cpdef verify_dimensions(self, int dim)
