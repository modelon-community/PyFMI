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
    pass

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
    cdef FMIL.fmi2_value_reference_t         _value_reference
    cdef FMIL.fmi2_base_type_enu_t           _type
    cdef FMIL.fmi2_variability_enu_t         _variability
    cdef FMIL.fmi2_causality_enu_t           _causality
    cdef FMIL.fmi2_variable_alias_kind_enu_t _alias
    cdef FMIL.fmi2_initial_enu_t             _initial
    cdef object _name
    cdef object _description #A characater pointer but we need an own reference and this is sufficient

cdef class FMUState2:
    """
    Class containing a pointer to a FMU-state.
    """
    cdef FMIL.fmi2_FMU_state_t* fmu_state



cdef class FMUModelBase(ModelBase):
    """
    An FMI Model loaded from a DLL.
    """
    #FMIL related variables
    cdef FMIL.fmi1_callback_functions_t callBackFunctions
    cdef FMIL.jm_callbacks callbacks
    cdef FMIL.fmi_import_context_t* context
    cdef FMIL.fmi1_import_t* _fmu
    cdef FMIL.fmi1_event_info_t _eventInfo
    cdef FMIL.fmi1_import_variable_list_t *variable_list
    cdef FMIL.fmi1_fmu_kind_enu_t fmu_kind
    cdef FMIL.jm_status_enu_t jm_status

    #Internal values
    cdef public object __t
    cdef public object _file_open
    cdef public object _npoints
    cdef public object _enable_logging
    cdef public object _pyEventInfo
    cdef list _log
    cdef int _version
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
    cdef char* _fmu_log_name
    cdef char* _fmu_temp_dir
    
    cdef _logger(self, FMIL.jm_string module, int log_level, FMIL.jm_string message)
    cpdef _internal_set_fmu_null(self)
    cpdef get_variable_description(self, char* variablename)
    cpdef FMIL.fmi1_base_type_enu_t get_variable_data_type(self,char* variablename) except *
    cpdef FMIL.fmi1_value_reference_t get_variable_valueref(self, char* variablename) except *
    cpdef get_variable_fixed(self, char* variablename)
    cpdef get_variable_start(self,char* variablename)
    cpdef get_variable_max(self,char* variablename)
    cpdef get_variable_min(self,char* variablename)
    cpdef FMIL.fmi1_variability_enu_t get_variable_variability(self,char* variablename) except *
    cpdef FMIL.fmi1_causality_enu_t get_variable_causality(self, char* variablename) except *

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
    cdef FMIL.jm_callbacks              callbacks
    cdef FMIL.fmi_import_context_t*     _context
    cdef FMIL.fmi2_callback_functions_t callBackFunctions
    cdef FMIL.fmi2_import_t*            _fmu
    cdef FMIL.fmi2_fmu_kind_enu_t       _fmu_kind
    cdef FMIL.fmi_version_enu_t         _version
    cdef FMIL.jm_string                 last_error
    cdef FMIL.size_t                    _nEventIndicators
    cdef FMIL.size_t                    _nContinuousStates
    cdef FMIL.size_t                    _nCategories
    cdef FMIL.fmi2_event_info_t         _eventInfo

    #Internal values
    cdef list           _log
    cdef object         _fmu_full_path
    cdef public object  _enable_logging
    cdef int _allocated_dll, _allocated_context, _allocated_xml, _allocated_fmu
    cdef char*          _modelId
    cdef object         _modelName
    cdef list           _categories
    cdef public list    _save_real_variables_val
    cdef public list    _save_int_variables_val
    cdef public list    _save_bool_variables_val
    cdef object         __t
    cdef public object  _pyEventInfo
    cdef char* _fmu_log_name
    cdef char* _fmu_temp_dir
    
    cdef _logger(self, FMIL.jm_string module, int log_level, FMIL.jm_string message)
    cpdef FMIL.fmi2_value_reference_t get_variable_valueref(self, char* variablename) except *
    cpdef FMIL.fmi2_base_type_enu_t get_variable_data_type(self, char* variablename) except *
    cpdef get_variable_description(self, char* variablename)
    cpdef FMIL.fmi2_variability_enu_t get_variable_variability(self,char* variablename) except *
    cpdef FMIL.fmi2_causality_enu_t get_variable_causality(self, char* variablename) except *
    cpdef get_variable_start(self, char* variablename)
    cpdef get_variable_max(self, char* variablename)
    cpdef get_variable_min(self, char* variablename)
    cpdef FMIL.fmi2_initial_enu_t get_variable_initial(self, char* variablename)
    cpdef serialize_fmu_state(self, state)
    cpdef deserialize_fmu_state(self, serialized_fmu)
    cpdef serialized_fmu_state_size(self, state)
    cdef _add_scalar_variables(self, FMIL.fmi2_import_variable_list_t*   variable_list)
    cdef _add_scalar_variable(self, FMIL.fmi2_import_variable_t* variable)

cdef class FMUModelCS2(FMUModelBase2):

    cpdef _get_time(self)
    cpdef _set_time(self, FMIL.fmi2_real_t t)
    
cdef class FMUModelME2(FMUModelBase2):

    cpdef _get_time(self)
    cpdef _set_time(self, FMIL.fmi2_real_t t)
    cpdef get_derivatives(self)
    
