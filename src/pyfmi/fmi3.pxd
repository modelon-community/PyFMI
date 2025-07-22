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

cdef class FMI3ModelVariable:
    """ Class defining data structure based on the XML elements for ModelVariables. """
    cdef FMIL3.fmi3_value_reference_t _value_reference
    cdef FMIL3.fmi3_base_type_enu_t           _type
    cdef FMIL3.fmi3_variability_enu_t _variability
    cdef FMIL3.fmi3_causality_enu_t _causality
    cdef FMIL3.fmi3_initial_enu_t _initial
    cdef object _name
    cdef object _description
    cdef FMIL3.fmi3_boolean_t _alias

cdef class FMI3EventInfo:
    cdef public FMIL3.fmi3_boolean_t newDiscreteDtatesNeeded
    cdef public FMIL3.fmi3_boolean_t terminateSimulation
    cdef public FMIL3.fmi3_boolean_t nominalsOfContinuousStatesChanged
    cdef public FMIL3.fmi3_boolean_t valuesOfContinuousStatesChanged
    cdef public FMIL3.fmi3_boolean_t nextEventTimeDefined
    cdef public FMIL3.fmi3_float64_t nextEventTime
    # This will be populated further once we add support for CS and Clocks in particular.

cdef class FMUModelBase3(FMI_BASE.ModelBase):
    # FMIL related variables
    cdef FMIL.fmi_import_context_t* _context
    cdef FMIL3.fmi3_import_t*       _fmu
    cdef FMIL3.fmi3_fmu_kind_enu_t  _fmu_kind
    cdef FMIL.fmi_version_enu_t     _version
    cdef FMIL.size_t _nEventIndicators  # format with snake case?
    cdef FMIL.size_t _nContinuousStates # format with snake case?
    cdef FMIL3.fmi3_boolean_t _event_info_new_discrete_states_needed
    cdef FMIL3.fmi3_boolean_t _event_info_terminate_simulation
    cdef FMIL3.fmi3_boolean_t _event_info_nominals_of_continuous_states_changed
    cdef FMIL3.fmi3_boolean_t _event_info_values_of_continuous_states_changed
    cdef FMIL3.fmi3_boolean_t _event_info_next_event_time_defined
    cdef FMIL3.fmi3_float64_t _event_info_next_event_time
    cdef FMI3EventInfo _eventInfo

    # Internal values
    cdef public float  _last_accepted_time
    cdef public object _enable_logging
    cdef public object _event_info
    cdef object     _fmu_full_path
    cdef object     _modelName
    cdef object     _t
    cdef int _allow_unzipped_fmu
    cdef int _allocated_context, _allocated_dll, _allocated_fmu, _allocated_xml

    # Caching
    cdef dict _outputs_states_dependencies
    cdef dict _outputs_inputs_dependencies
    cdef dict _outputs_states_dependencies_kind
    cdef dict _outputs_inputs_dependencies_kind

    cdef int _initialized_fmu
    cdef object _has_entered_init_mode # this is public in FMI2 but I don't see why

    cpdef set_float64(self, valueref, values)
    cpdef set_float32(self, valueref, values)
    cpdef set_int64  (self, valueref, values)
    cpdef set_int32  (self, valueref, values)
    cpdef set_int16  (self, valueref, values)
    cpdef set_int8   (self, valueref, values)
    cpdef set_uint64 (self, valueref, values)
    cpdef set_uint32 (self, valueref, values)
    cpdef set_uint16 (self, valueref, values)
    cpdef set_uint8  (self, valueref, values)
    cpdef set_boolean(self, valueref, values)
    cpdef set_string (self, valueref, values)
    cpdef set_enum   (self, valueref, values)

    cpdef np.ndarray get_float64(self, valueref)
    cpdef np.ndarray get_float32(self, valueref)
    cpdef np.ndarray get_int64  (self, valueref)
    cpdef np.ndarray get_int32  (self, valueref)
    cpdef np.ndarray get_int16  (self, valueref)
    cpdef np.ndarray get_int8   (self, valueref)
    cpdef np.ndarray get_uint64 (self, valueref)
    cpdef np.ndarray get_uint32 (self, valueref)
    cpdef np.ndarray get_uint16 (self, valueref)
    cpdef np.ndarray get_uint8  (self, valueref)
    cpdef np.ndarray get_boolean(self, valueref)
    cpdef list       get_string (self, valueref)
    cpdef np.ndarray get_enum   (self, valueref)

    cpdef _get_time(self)
    cpdef _set_time(self, FMIL3.fmi3_float64_t t)

    cpdef FMIL3.fmi3_value_reference_t get_variable_valueref(self, variable_name) except *
    cdef FMIL3.fmi3_base_type_enu_t _get_variable_data_type(self, variable_name) except *
    cdef FMIL3.fmi3_causality_enu_t _get_variable_causality(self, variable_name) except *
    cpdef get_variable_description(self, variable_name)
    cdef _add_variable(self, FMIL3.fmi3_import_variable_t* variable)
    cdef FMIL3.fmi3_status_t _get_directional_derivative(self, np.ndarray v_ref, np.ndarray z_ref, np.ndarray dv, np.ndarray dz) except *
    cpdef get_variable_unbounded(self, variablename)
    cdef _get_variable_description(self, FMIL3.fmi3_import_variable_t*)
    cdef _get_alias_description(self, FMIL3.fmi3_import_alias_variable_t*)

    cpdef get_output_dependencies(self)
    cpdef get_output_dependencies_kind(self)

cdef class FMUModelME3(FMUModelBase3):
    cpdef get_derivatives(self)
    cdef FMIL3.fmi3_status_t _get_derivatives(self, FMIL3.fmi3_float64_t[:] values)
    cdef FMIL3.fmi3_status_t _get_continuous_states_fmil(self, FMIL3.fmi3_float64_t[:] ndx)
    cdef FMIL3.fmi3_status_t _set_continuous_states_fmil(self, FMIL3.fmi3_float64_t[:] ndx)
    cdef FMIL3.fmi3_status_t _get_event_indicators(self, FMIL3.fmi3_float64_t[:] ndx)
    cdef FMIL3.fmi3_status_t _completed_integrator_step(self,
        FMIL3.fmi3_boolean_t no_set_FMU_state_prior_to_current_point,
        FMIL3.fmi3_boolean_t* enter_event_mode,
        FMIL3.fmi3_boolean_t* terminate_simulation
    )
    cdef FMIL3.fmi3_status_t _get_nominal_continuous_states_fmil(self, FMIL3.fmi3_float64_t* xnominal, size_t nx)

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
