#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2014-2021 Modelon AB
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

# TODO: Description
# Module containing the FMI interface Python wrappers.

import numpy as np
cimport numpy as np

cimport pyfmi.fmil_import as FMIL
cimport pyfmi.fmil1_import as FMIL1
cimport pyfmi.fmil2_import as FMIL2

cdef FMIL.fmi_version_enu_t import_and_get_version(FMIL.fmi_import_context_t*, char*, char*, int)

cimport pyfmi.fmi_base as FMI_BASE

cdef class FMUModelBase(FMI_BASE.ModelBase):
    """
    An FMI Model loaded from a DLL.
    """
    #FMIL related variables
    cdef FMIL1.fmi1_callback_functions_t callBackFunctions
    cdef FMIL.fmi_import_context_t* context
    cdef FMIL1.fmi1_import_t* _fmu
    cdef FMIL1.fmi1_event_info_t _eventInfo
    cdef FMIL1.fmi1_import_variable_list_t *variable_list
    cdef FMIL1.fmi1_fmu_kind_enu_t fmu_kind
    cdef FMIL.jm_status_enu_t jm_status
    cdef FMIL.jm_callbacks callbacks_defaults
    cdef FMIL.jm_callbacks* callbacks_standard

    #Internal values
    cdef public object _t
    cdef public object _file_open
    cdef public object _npoints
    cdef public object _enable_logging
    cdef public object _pyEventInfo
    cdef object        _fmu_full_path
    cdef int _version
    cdef int _instantiated_fmu
    cdef int _allow_unzipped_fmu
    cdef int _allocated_dll, _allocated_context, _allocated_xml, _allocated_fmu
    cdef object _allocated_list
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
    cpdef FMIL1.fmi1_base_type_enu_t get_variable_data_type(self, variablename) except *
    cpdef FMIL1.fmi1_value_reference_t get_variable_valueref(self, variablename) except *
    cpdef get_variable_fixed(self, variablename)
    cpdef get_variable_start(self, variablename)
    cpdef get_variable_max(self, variablename)
    cpdef get_variable_min(self, variablename)
    cpdef FMIL1.fmi1_variability_enu_t get_variable_variability(self, variablename) except *
    cpdef FMIL1.fmi1_causality_enu_t get_variable_causality(self, variablename) except *
    cdef _add_scalar_variable(self, FMIL1.fmi1_import_variable_t* variable)

cdef class FMUModelCS1(FMUModelBase):

    cpdef _get_time(self)
    cpdef _set_time(self, FMIL1.fmi1_real_t t)

cdef class FMUModelME1(FMUModelBase):

    cpdef _get_time(self)
    cpdef _set_time(self, FMIL1.fmi1_real_t t)
    cpdef get_derivatives(self)
    cdef int _get_nominal_continuous_states_fmil(self, FMIL1.fmi1_real_t* xnominal, size_t nx)
    cdef public object _preinit_nominal_continuous_states

cdef class FMUModelBase2(FMI_BASE.ModelBase):
    """
    FMI Model loaded from a dll.
    """

    #FMIL related variables
    cdef FMIL.fmi_import_context_t*     _context
    cdef FMIL2.fmi2_callback_functions_t callBackFunctions
    cdef FMIL2.fmi2_import_t*            _fmu
    cdef FMIL2.fmi2_fmu_kind_enu_t       _fmu_kind
    cdef FMIL.fmi_version_enu_t         _version
    cdef FMIL.jm_string                 last_error
    cdef FMIL.size_t                    _nEventIndicators
    cdef FMIL.size_t                    _nContinuousStates
    cdef FMIL2.fmi2_event_info_t         _eventInfo

    #Internal values
    cdef public float  _last_accepted_time, _relative_tolerance
    cdef object         _fmu_full_path
    cdef public object  _enable_logging
    cdef int _allow_unzipped_fmu
    cdef int _allocated_dll, _allocated_context, _allocated_xml, _allocated_fmu, _initialized_fmu
    cdef object         _modelName
    cdef public list    _save_real_variables_val
    cdef public list    _save_int_variables_val
    cdef public list    _save_bool_variables_val
    cdef object         _t
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
    cdef public object  _group_A, _group_B, _group_C, _group_D
    cdef object         _mask_A
    cdef object         _A_row_ind, _A_col_ind
    cdef public object  _has_entered_init_mode
    cdef WorkerClass2 _worker_object

    cpdef FMIL2.fmi2_value_reference_t get_variable_valueref(self, variablename) except *
    cpdef FMIL2.fmi2_base_type_enu_t get_variable_data_type(self, variablename) except *
    cpdef get_variable_description(self, variablename)
    cpdef FMIL2.fmi2_variability_enu_t get_variable_variability(self, variablename) except *
    cpdef FMIL2.fmi2_causality_enu_t get_variable_causality(self, variablename) except *
    cpdef get_output_dependencies(self)
    cpdef get_output_dependencies_kind(self)
    cpdef get_derivatives_dependencies(self)
    cpdef get_derivatives_dependencies_kind(self)
    cpdef get_variable_start(self, variablename)
    cpdef get_variable_max(self, variablename)
    cpdef get_variable_min(self, variablename)
    cpdef get_variable_unbounded(self, variablename)
    cpdef FMIL2.fmi2_initial_enu_t get_variable_initial(self, variable_name) except *
    cpdef serialize_fmu_state(self, state)
    cpdef deserialize_fmu_state(self, serialized_fmu)
    cpdef serialized_fmu_state_size(self, state)
    cdef _add_scalar_variables(self, FMIL2.fmi2_import_variable_list_t*   variable_list)
    cdef _add_scalar_variable(self, FMIL2.fmi2_import_variable_t* variable)
    cdef int _get_directional_derivative(self, np.ndarray v_ref, np.ndarray z_ref, np.ndarray dv, np.ndarray dz) except -1
    cpdef set_real(self, valueref, values)
    cpdef np.ndarray get_real(self, valueref)
    cdef int _set_real(self, FMIL2.fmi2_value_reference_t* vrefs, FMIL2.fmi2_real_t* values, size_t _size)
    cdef int _get_real_by_ptr(self, FMIL2.fmi2_value_reference_t* vrefs, size_t _size, FMIL2.fmi2_real_t* values)
    cdef int _get_real_by_list(self, FMIL2.fmi2_value_reference_t[:] valueref, size_t _size, FMIL2.fmi2_real_t[:] values)
    cdef int _get_integer(self, FMIL2.fmi2_value_reference_t[:] valueref, size_t _size, FMIL2.fmi2_integer_t[:] values)
    cdef int _get_boolean(self, FMIL2.fmi2_value_reference_t[:] valueref, size_t _size, FMIL2.fmi2_real_t[:] values)

cdef class FMUModelCS2(FMUModelBase2):

    cpdef _get_time(self)
    cpdef _set_time(self, FMIL2.fmi2_real_t t)
    cpdef int do_step(self, FMIL2.fmi2_real_t current_t, FMIL2.fmi2_real_t step_size, new_step=*)
    cdef int _set_input_derivatives(self, np.ndarray value_refs, np.ndarray values, np.ndarray orders)
    cdef int _get_output_derivatives(self, np.ndarray value_refs, np.ndarray values, np.ndarray orders)

cdef class FMUModelME2(FMUModelBase2):

    cpdef _get_time(self)
    cpdef _set_time(self, FMIL2.fmi2_real_t t)
    cpdef get_derivatives(self)
    cdef public object force_finite_differences
    cdef int _get_derivatives(self, FMIL2.fmi2_real_t[:] values)
    cdef int _get_continuous_states_fmil(self, FMIL2.fmi2_real_t[:] ndx)
    cdef int _set_continuous_states_fmil(self, FMIL2.fmi2_real_t[:] ndx)
    cdef int _get_event_indicators(self, FMIL2.fmi2_real_t[:] values)
    cdef int _completed_integrator_step(self, int* enter_event_mode, int* terminate_simulation)
    cdef int _get_nominal_continuous_states_fmil(self, FMIL2.fmi2_real_t* xnominal, size_t nx)
    cdef public object _preinit_nominal_continuous_states

cdef class WorkerClass2:
    cdef int _dim

    cdef np.ndarray _tmp1_val, _tmp2_val, _tmp3_val, _tmp4_val
    cdef np.ndarray _tmp1_ref, _tmp2_ref, _tmp3_ref, _tmp4_ref

    cdef FMIL2.fmi2_real_t* get_real_vector(self, int index)
    cdef FMIL2.fmi2_value_reference_t* get_value_reference_vector(self, int index)
    cdef np.ndarray get_value_reference_numpy_vector(self, int index)
    cdef np.ndarray get_real_numpy_vector(self, int index)
    cpdef verify_dimensions(self, int dim)
