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

# Module containing the FMI1 interface Python wrappers.

cimport pyfmi.fmil_import as FMIL
cimport pyfmi.fmil1_import as FMIL1

cimport pyfmi.fmi_base as FMI_BASE

cdef class ScalarVariable:
    """
    Class defining data structure based on the XML element ScalarVariable.
    """
    cdef object _name
    cdef FMIL1.fmi1_value_reference_t _value_reference
    cdef object _description #A character pointer but we need an own reference and this is sufficient
    cdef FMIL1.fmi1_base_type_enu_t _type
    cdef FMIL1.fmi1_variability_enu_t _variability
    cdef FMIL1.fmi1_causality_enu_t _causality
    cdef FMIL1.fmi1_variable_alias_kind_enu_t _alias

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

cdef object _load_fmi1_fmu(
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
