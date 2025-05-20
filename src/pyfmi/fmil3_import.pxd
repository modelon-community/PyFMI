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

#==============================================
# C headers
#==============================================

# This file contains FMIL header content specific to FMI3
cimport pyfmi.fmil_import as FMIL
from libcpp cimport bool # TODO: Possible issue due to https://github.com/cython/cython/issues/5730 ??
from libc.stdint cimport (
    int64_t,
    int32_t,
    int16_t,
    int8_t,
    uint64_t,
    uint32_t,
    uint16_t,
    uint8_t,
)

cdef extern from 'fmilib.h':
    # FMI VARIABLE TYPE DEFINITIONS
    ctypedef void*    fmi3_instance_environment_t
    ctypedef char*    fmi3_string_t
    ctypedef bool     fmi3_boolean_t
    ctypedef double   fmi3_float64_t
    ctypedef float    fmi3_float32_t
    ctypedef int64_t  fmi3_int64_t
    ctypedef int32_t  fmi3_int32_t
    ctypedef int16_t  fmi3_int16_t
    ctypedef int8_t   fmi3_int8_t
    ctypedef uint64_t fmi3_uint64_t
    ctypedef uint32_t fmi3_uint32_t
    ctypedef uint16_t fmi3_uint16_t
    ctypedef uint8_t  fmi3_uint8_t
    ctypedef uint32_t fmi3_value_reference_t

    # STRUCTS
    ctypedef enum fmi3_boolean_enu_t:
        fmi3_true = 1
        fmi3_false = 0

    cdef enum fmi3_fmu_kind_enu_t:
        fmi3_fmu_kind_unknown   = 1 << 0
        fmi3_fmu_kind_me        = 1 << 1
        fmi3_fmu_kind_cs        = 1 << 2
        fmi3_fmu_kind_se        = 1 << 3

    cdef enum fmi3_base_type_enu_t:
        fmi3_base_type_float64 = 1,
        fmi3_base_type_float32 = 2,
        fmi3_base_type_int64   = 3,
        fmi3_base_type_int32   = 4,
        fmi3_base_type_int16   = 5,
        fmi3_base_type_int8    = 6,
        fmi3_base_type_uint64  = 7,
        fmi3_base_type_uint32  = 8,
        fmi3_base_type_uint16  = 9,
        fmi3_base_type_uint8   = 10,
        fmi3_base_type_bool    = 11,
        fmi3_base_type_binary  = 12,
        fmi3_base_type_clock   = 13,
        fmi3_base_type_str     = 14,
        fmi3_base_type_enum    = 15

    cdef enum fmi3_causality_enu_t:
        fmi3_causality_enu_structural_parameter = 0,
        fmi3_causality_enu_parameter            = 1,
        fmi3_causality_enu_calculated_parameter = 2,
        fmi3_causality_enu_input                = 3,
        fmi3_causality_enu_output               = 4,
        fmi3_causality_enu_local                = 5,
        fmi3_causality_enu_independent          = 6,
        fmi3_causality_enu_unknown              = 7

    cdef enum fmi3_variability_enu_t:
        fmi3_variability_enu_constant    = 0,
        fmi3_variability_enu_fixed       = 1,
        fmi3_variability_enu_tunable     = 2,
        fmi3_variability_enu_discrete    = 3,
        fmi3_variability_enu_continuous  = 4,
        fmi3_variability_enu_unknown     = 5

    cdef enum fmi3_initial_enu_t:
        fmi3_initial_enu_exact      = 1,
        fmi3_initial_enu_approx     = 2,
        fmi3_initial_enu_calculated = 3,
        fmi3_initial_enu_unknown    = 4

    cdef struct fmi3_xml_variable_t:
        pass

    cdef struct fmi3_import_variable_list_t:
        pass
    ctypedef fmi3_xml_variable_t fmi3_import_variable_t

    cdef struct fmi3_xml_float64_variable_t:
        pass
    ctypedef fmi3_xml_float64_variable_t fmi3_import_float64_variable_t

    cdef struct fmi3_xml_unit_t:
        pass
    ctypedef fmi3_xml_unit_t fmi3_import_unit_t

    cdef struct fmi3_xml_display_unit_t:
        pass
    ctypedef fmi3_xml_display_unit_t fmi3_import_display_unit_t

    cdef struct fmi3_xml_unit_definition_list_t:
        pass
    ctypedef fmi3_xml_unit_definition_list_t fmi3_import_unit_definitions_t

    cdef struct fmi3_xml_variable_typedef_t:
        pass
    ctypedef fmi3_xml_variable_typedef_t fmi3_import_variable_typedef_t

    # Alias
    cdef struct fmi3_xml_alias_variable_list_t:
        pass
    ctypedef fmi3_xml_alias_variable_list_t fmi3_import_alias_variable_list_t

    cdef struct fmi3_xml_alias_variable_t:
        pass
    ctypedef fmi3_xml_alias_variable_t fmi3_import_alias_variable_t

    # STATUS
    ctypedef enum fmi3_status_t:
        fmi3_status_ok = 0
        fmi3_status_warning = 1
        fmi3_status_discard = 2
        fmi3_status_error = 3
        fmi3_status_fatal = 4

    # LOGGING
    ctypedef int(*fmi3_xml_element_start_handle_ft)(void*, char*, void*, char*, char**)
    ctypedef int(*fmi3_xml_element_data_handle_ft)(void*, char*, int)
    ctypedef int(*fmi3_xml_element_end_handle_ft)(void*, char*)
    cdef struct fmi3_xml_callbacks_t:
        fmi3_xml_element_start_handle_ft startHandle
        fmi3_xml_element_data_handle_ft  dataHandle
        fmi3_xml_element_end_handle_ft   endHandle
        void* context

    ctypedef void(*fmi3_log_message_callback_ft)(
        fmi3_instance_environment_t instance_environment,
        fmi3_status_t status,
        fmi3_string_t category,
        fmi3_string_t message,
        )

    cdef struct fmi3_import_t:
        pass


    # FMI SPECIFICATION METHODS (3.0)
    # BASIC
    int fmi3_import_create_dllfmu(fmi3_import_t*, fmi3_fmu_kind_enu_t, fmi3_instance_environment_t, fmi3_log_message_callback_ft )
    FMIL.jm_status_enu_t fmi3_import_instantiate_model_exchange(
        fmi3_import_t* fmu,
        fmi3_string_t instanceName,
        fmi3_string_t resourcePath,
        fmi3_boolean_t visible,
        fmi3_boolean_t loggingOn
    )
    fmi3_status_t fmi3_import_completed_integrator_step(fmi3_import_t*, fmi3_boolean_t, fmi3_boolean_t*, fmi3_boolean_t*)

    # modes
    fmi3_status_t fmi3_import_enter_initialization_mode(
        fmi3_import_t* fmu,
        fmi3_boolean_t toleranceDefined,
        fmi3_float64_t tolerance,
        fmi3_float64_t startTime,
        fmi3_boolean_t stopTimeDefined,
        fmi3_float64_t stopTime)
    fmi3_status_t fmi3_import_exit_initialization_mode(fmi3_import_t* fmu)
    fmi3_status_t fmi3_import_enter_event_mode(fmi3_import_t* fmu)
    fmi3_status_t fmi3_import_enter_continuous_time_mode(fmi3_import_t* fmu)
    # misc
    char* fmi3_import_get_version(fmi3_import_t*)
    fmi3_status_t fmi3_import_reset(fmi3_import_t* fmu)
    fmi3_status_t fmi3_import_terminate(fmi3_import_t* fmu)
    void fmi3_import_free_instance(fmi3_import_t* fmu)
    void fmi3_import_destroy_dllfmu(fmi3_import_t* fmu)

    # setting
    fmi3_status_t fmi3_import_set_time(fmi3_import_t*, fmi3_float64_t)

    fmi3_status_t fmi3_import_set_float64(fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_float64_t*, size_t)
    fmi3_status_t fmi3_import_set_float32(fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_float32_t*, size_t)
    fmi3_status_t fmi3_import_set_int64  (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_int64_t*, size_t)
    fmi3_status_t fmi3_import_set_int32  (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_int32_t*, size_t)
    fmi3_status_t fmi3_import_set_int16  (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_int16_t*, size_t)
    fmi3_status_t fmi3_import_set_int8   (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_int8_t*, size_t)
    fmi3_status_t fmi3_import_set_uint64 (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_uint64_t*, size_t)
    fmi3_status_t fmi3_import_set_uint32 (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_uint32_t*, size_t)
    fmi3_status_t fmi3_import_set_uint16 (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_uint16_t*, size_t)
    fmi3_status_t fmi3_import_set_uint8  (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_uint8_t*, size_t)
    fmi3_status_t fmi3_import_set_string (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_string_t*, size_t)
    fmi3_status_t fmi3_import_set_boolean(fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_boolean_t*, size_t)
    fmi3_status_t fmi3_import_set_continuous_states(fmi3_import_t*, fmi3_float64_t*, size_t);

    # getting
    fmi3_status_t fmi3_import_get_derivatives(fmi3_import_t *, fmi3_float64_t *, size_t)
    fmi3_status_t fmi3_import_get_float64(fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_float64_t*, size_t);
    fmi3_status_t fmi3_import_get_float32(fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_float32_t*, size_t);
    fmi3_status_t fmi3_import_get_int64  (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_int64_t*, size_t);
    fmi3_status_t fmi3_import_get_int32  (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_int32_t*, size_t);
    fmi3_status_t fmi3_import_get_int16  (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_int16_t*, size_t);
    fmi3_status_t fmi3_import_get_int8   (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_int8_t*, size_t);
    fmi3_status_t fmi3_import_get_uint64 (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_uint64_t*, size_t);
    fmi3_status_t fmi3_import_get_uint32 (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_uint32_t*, size_t);
    fmi3_status_t fmi3_import_get_uint16 (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_uint16_t*, size_t);
    fmi3_status_t fmi3_import_get_uint8  (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_uint8_t*, size_t);
    fmi3_status_t fmi3_import_get_string (fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_string_t*, size_t);
    fmi3_status_t fmi3_import_get_boolean(fmi3_import_t*, fmi3_value_reference_t*, size_t, fmi3_boolean_t*, size_t);
    fmi3_status_t fmi3_import_get_continuous_states(fmi3_import_t*, fmi3_float64_t*, size_t);
    fmi3_status_t fmi3_import_get_nominals_of_continuous_states(fmi3_import_t*, fmi3_float64_t*, size_t nx)
    fmi3_import_variable_t* fmi3_import_get_variable(fmi3_import_variable_list_t *, size_t)
    fmi3_import_variable_list_t* fmi3_import_get_variable_list(fmi3_import_t*, int)
    size_t fmi3_import_get_variable_list_size(fmi3_import_variable_list_t*)
    fmi3_import_variable_list_t* fmi3_import_get_continuous_state_derivatives_list(fmi3_import_t* fmu)
    fmi3_import_float64_variable_t* fmi3_import_get_float64_variable_derivative_of(fmi3_import_float64_variable_t* v);

    double fmi3_import_get_default_experiment_start(fmi3_import_t*);
    double fmi3_import_get_default_experiment_stop(fmi3_import_t*);
    double fmi3_import_get_default_experiment_tolerance(fmi3_import_t*);
    # save states

    # FMI HELPER METHODS (3.0)
    fmi3_fmu_kind_enu_t fmi3_import_get_fmu_kind(fmi3_import_t*)
    char* fmi3_fmu_kind_to_string(fmi3_fmu_kind_enu_t)
    char* fmi3_import_get_model_name(fmi3_import_t*)
    const char* fmi3_import_get_generation_tool(fmi3_import_t *)

    # FMI XML METHODS
    # Parsing/logging basics
    fmi3_import_t* fmi3_import_parse_xml(FMIL.fmi_import_context_t*, char*, fmi3_xml_callbacks_t*)
    void fmi3_import_free(fmi3_import_t*)

    ### Model information

    # CONVERTER METHODS

    # INTEGER

    # OTHER HELPER METHODS

    # Does NOT invoke CAPI calls

    fmi3_status_t fmi3_import_get_number_of_event_indicators(fmi3_import_t*, size_t*)
    fmi3_status_t fmi3_import_get_number_of_continuous_states(fmi3_import_t*, size_t*)
    char* fmi3_import_get_last_error(fmi3_import_t*)
    char* fmi3_import_get_model_identifier_ME(fmi3_import_t*)
    void fmi3_import_free_variable_list(fmi3_import_variable_list_t*)

    # Getting variables attributes/types
    const char* fmi3_import_get_variable_name(fmi3_import_variable_t*)
    fmi3_variability_enu_t fmi3_import_get_variable_variability(fmi3_import_variable_t*)
    fmi3_causality_enu_t fmi3_import_get_variable_causality(fmi3_import_variable_t*)
    fmi3_initial_enu_t fmi3_import_get_variable_initial(fmi3_import_variable_t*)
    fmi3_string_t fmi3_import_get_variable_description(fmi3_import_variable_t*)
    int fmi3_import_get_variable_has_start(fmi3_import_variable_t*)
    fmi3_import_variable_t* fmi3_import_get_variable(fmi3_import_variable_list_t* vl, size_t index);
    fmi3_import_variable_t* fmi3_import_get_variable_by_name(fmi3_import_t*, char*)
    fmi3_value_reference_t fmi3_import_get_variable_vr(fmi3_import_variable_t*)
    fmi3_base_type_enu_t fmi3_import_get_variable_base_type(fmi3_import_variable_t*)
    char* fmi3_import_get_model_version(fmi3_import_t*)

    # Alias
    fmi3_import_alias_variable_list_t* fmi3_import_get_variable_alias_list(fmi3_import_variable_t* v)
    size_t fmi3_import_get_alias_variable_list_size(fmi3_import_alias_variable_list_t* aliases);
    fmi3_import_alias_variable_t* fmi3_import_get_alias(fmi3_import_alias_variable_list_t* aliases, size_t index);
    const char* fmi3_import_get_alias_variable_name(fmi3_import_alias_variable_t* alias);
    const char* fmi3_import_get_alias_variable_description(fmi3_import_alias_variable_t* alias);
