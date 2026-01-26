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

#==============================================
# C headers
#==============================================

# This file contains FMIL header content specific to FMI2

from fmil_import cimport (
    fmi_import_context_t,
    jm_status_enu_t,
    jm_callbacks,
     __va_list_tag
)

cdef extern from 'fmilib.h':
    # FMI VARIABLE TYPE DEFINITIONS
    ctypedef double fmi2_real_t
    ctypedef int    fmi2_boolean_t
    ctypedef void*  fmi2_component_t
    ctypedef char*  fmi2_string_t
    ctypedef int    fmi2_integer_t
    ctypedef char   fmi2_byte_t
    ctypedef void*  fmi2_FMU_state_t
    ctypedef void*  fmi2_component_environment_t
    ctypedef size_t fmi2_value_reference_t

    # STRUCTS
    ctypedef enum fmi2_boolean_enu_t:
        fmi2_true = 1
        fmi2_false = 0

    ctypedef enum fmi2_type_t:
        fmi2_model_exchange,
        fmi2_cosimulation

    ctypedef enum fmi2_status_t:
        fmi2_status_ok = 0
        fmi2_status_warning = 1
        fmi2_status_discard = 2
        fmi2_status_error = 3
        fmi2_status_fatal = 4
        fmi2_status_pending = 5

    cdef enum fmi2_variable_alias_kind_enu_t:
        fmi2_variable_is_not_alias = 0
        fmi2_variable_is_alias = 1

    cdef enum fmi2_base_type_enu_t:
        fmi2_base_type_real = 0
        fmi2_base_type_int = 1
        fmi2_base_type_bool = 2
        fmi2_base_type_str = 3
        fmi2_base_type_enum = 4

    cdef enum fmi2_causality_enu_t:
        fmi2_causality_enu_parameter = 0
        fmi2_causality_enu_calculated_parameter = 1
        fmi2_causality_enu_input  = 2
        fmi2_causality_enu_output  = 3
        fmi2_causality_enu_local  = 4
        fmi2_causality_enu_independent = 5
        fmi2_causality_enu_unknown = 6

    cdef enum fmi2_fmu_kind_enu_t:
        fmi2_fmu_kind_unknown = 0
        fmi2_fmu_kind_me = 1
        fmi2_fmu_kind_cs = 2
        fmi2_fmu_kind_me_and_cs = 3

    cdef enum fmi2_variability_enu_t:
        fmi2_variability_enu_constant = 0
        fmi2_variability_enu_fixed = 1
        fmi2_variability_enu_tunable = 2
        fmi2_variability_enu_discrete = 3
        fmi2_variability_enu_continuous = 4
        fmi2_variability_enu_unknown = 5

    cdef enum fmi2_variable_naming_convension_enu_t:
        fmi2_naming_enu_flat = 0
        fmi2_naming_enu_structured = 1
        fmi2_naming_enu_unknown = 2

    ctypedef enum fmi2_status_kind_t:
        fmi2_do_step_status = 0
        fmi2_pending_status = 1
        fmi2_last_successful_time = 2
        fmi2_terminated = 3

    ctypedef struct fmi2_event_info_t:
        fmi2_boolean_t newDiscreteStatesNeeded
        fmi2_boolean_t terminateSimulation
        fmi2_boolean_t nominalsOfContinuousStatesChanged
        fmi2_boolean_t valuesOfContinuousStatesChanged
        fmi2_boolean_t nextEventTimeDefined
        fmi2_real_t    nextEventTime

    cdef enum fmi2_dependency_factor_kind_enu_t:
        fmi2_dependency_factor_kind_dependent = 0
        fmi2_dependency_factor_kind_constant = 1
        fmi2_dependency_factor_kind_fixed = 2
        fmi2_dependency_factor_kind_tunable = 3
        fmi2_dependency_factor_kind_discrete = 4

    cdef enum fmi2_initial_enu_t:
        fmi2_initial_enu_exact = 0
        fmi2_initial_enu_approx = 1
        fmi2_initial_enu_calculated = 2
        fmi2_initial_enu_unknown = 3

    cdef enum fmi2_capabilities_enu_t:
        fmi2_me_needsExecutionTool = 0
        fmi2_me_completedIntegratorStepNotNeeded = 1
        fmi2_me_canBeInstantiatedOnlyOncePerProcess = 2
        fmi2_me_canNotUseMemoryManagementFunctions = 3
        fmi2_me_canGetAndSetFMUstate = 4
        fmi2_me_canSerializeFMUstate = 5
        fmi2_me_providesDirectionalDerivatives = 6
        fmi2_me_completedEventIterationIsProvided = 7
        fmi2_cs_needsExecutionTool = 8
        fmi2_cs_canHandleVariableCommunicationStepSize = 9
        fmi2_cs_canInterpolateInputs = 10
        fmi2_cs_maxOutputDerivativeOrder = 11
        fmi2_cs_canRunAsynchronuously = 12
        fmi2_cs_canBeInstantiatedOnlyOncePerProcess = 13
        fmi2_cs_canNotUseMemoryManagementFunctions = 14
        fmi2_cs_canGetAndSetFMUstate = 15
        fmi2_cs_canSerializeFMUstate = 16
        fmi2_cs_providesDirectionalDerivatives = 17
        fmi2_capabilities_Num = 18

    cdef enum fmi2_SI_base_units_enu_t:
        fmi2_SI_base_unit_kg = 0
        fmi2_SI_base_unit_m = 1
        fmi2_SI_base_unit_s = 2
        fmi2_SI_base_unit_A = 3
        fmi2_SI_base_unit_K = 4
        fmi2_SI_base_unit_mol = 5
        fmi2_SI_base_unit_cd = 6
        fmi2_SI_base_unit_rad = 7
        fmi2_SI_base_units_Num = 8

    cdef struct fmi2_import_model_counts_t:
        unsigned int num_constants
        unsigned int num_tunable
        unsigned int num_fixed
        unsigned int num_discrete
        unsigned int num_continuous
        unsigned int num_inputs
        unsigned int num_outputs
        unsigned int num_local
        unsigned int num_parameters
        unsigned int num_real_vars
        unsigned int num_integer_vars
        unsigned int num_enum_vars
        unsigned int num_bool_vars
        unsigned int num_string_vars

    cdef struct fmi2_xml_variable_t:
        pass
    ctypedef fmi2_xml_variable_t fmi2_import_variable_t

    ctypedef void(*fmi2_callback_logger_ft)(fmi2_component_environment_t c,fmi2_string_t instanceName, fmi2_status_t status, fmi2_string_t category,fmi2_string_t message,...)
    ctypedef void(*fmi2_step_finished_ft)(fmi2_component_environment_t env, fmi2_status_t status)
    ctypedef void *(*fmi2_callback_allocate_memory_ft)(size_t, size_t)
    ctypedef void(*fmi2_callback_free_memory_ft)(void *)
    ctypedef int(*fmi2_xml_element_start_handle_ft)(void *, char*, void *, char*, char* *)
    ctypedef int(*fmi2_xml_element_data_handle_ft)(void *, char*, int)
    ctypedef int(*fmi2_xml_element_end_handle_ft)(void *, char*)
    ctypedef int(*fmi2_import_variable_filter_function_ft)(fmi2_import_variable_t *, void *)

    cdef struct fmi2_xml_callbacks_t:
        fmi2_xml_element_start_handle_ft startHandle
        fmi2_xml_element_data_handle_ft dataHandle
        fmi2_xml_element_end_handle_ft endHandle
        void * context

    ctypedef struct fmi2_callback_functions_t:
        fmi2_callback_logger_ft logger
        fmi2_callback_allocate_memory_ft allocateMemory
        fmi2_callback_free_memory_ft freeMemory
        fmi2_step_finished_ft stepFinished
        fmi2_component_environment_t componentEnvironment

    cdef struct fmi2_import_t:
        pass

    cdef struct fmi2_xml_real_variable_t:
        pass
    ctypedef fmi2_xml_real_variable_t fmi2_import_real_variable_t

    cdef struct fmi2_xml_display_unit_t:
        pass
    ctypedef fmi2_xml_display_unit_t fmi2_import_display_unit_t

    cdef struct fmi2_xml_unit_definitions_t:
        pass
    ctypedef fmi2_xml_unit_definitions_t fmi2_import_unit_definitions_t

    cdef struct fmi2_import_variable_list_t:
        pass

    cdef struct fmi2_xml_variable_typedef_t:
        pass
    ctypedef fmi2_xml_variable_typedef_t fmi2_import_variable_typedef_t

    cdef struct fmi2_xml_integer_variable_t:
        pass
    ctypedef fmi2_xml_integer_variable_t fmi2_import_integer_variable_t

    cdef struct fmi2_xml_real_typedef_t:
        pass
    ctypedef fmi2_xml_real_typedef_t fmi2_import_real_typedef_t

    cdef struct fmi2_xml_enum_variable_t:
        pass
    ctypedef fmi2_xml_enum_variable_t fmi2_import_enum_variable_t

    cdef struct fmi2_xml_type_definitions_t:
        pass
    ctypedef fmi2_xml_type_definitions_t fmi2_import_type_definitions_t

    cdef struct fmi2_xml_enumeration_typedef_t:
        pass
    ctypedef fmi2_xml_enumeration_typedef_t fmi2_import_enumeration_typedef_t

    cdef struct fmi2_xml_integer_typedef_t:
        pass
    ctypedef fmi2_xml_integer_typedef_t fmi2_import_integer_typedef_t

    cdef struct fmi2_xml_unit_t:
        pass
    ctypedef fmi2_xml_unit_t fmi2_import_unit_t

    cdef struct fmi2_xml_bool_variable_t:
        pass
    ctypedef fmi2_xml_bool_variable_t fmi2_import_bool_variable_t

    cdef struct fmi2_xml_string_variable_t:
        pass
    ctypedef fmi2_xml_string_variable_t fmi2_import_string_variable_t

    #FMI SPECIFICATION METHODS (2.0)
    # basic
    int fmi2_import_create_dllfmu(fmi2_import_t*, fmi2_fmu_kind_enu_t, fmi2_callback_functions_t *)
    jm_status_enu_t fmi2_import_instantiate(fmi2_import_t* fmu, fmi2_string_t instanceName, fmi2_type_t fmuType, fmi2_string_t fmuResourceLocation, fmi2_boolean_t visible)
    void fmi2_import_free_instance(fmi2_import_t* fmu)
    char* fmi2_import_get_types_platform(fmi2_import_t*)
    int fmi2_import_setup_experiment(fmi2_import_t* fmu, fmi2_boolean_t toleranceDefined, fmi2_real_t tolerance,fmi2_real_t startTime, fmi2_boolean_t stopTimeDefined,fmi2_real_t stopTime)
    int fmi2_import_do_step(fmi2_import_t*, fmi2_real_t, fmi2_real_t, fmi2_boolean_t) nogil
    int fmi2_import_completed_integrator_step(fmi2_component_t, fmi2_boolean_t, fmi2_boolean_t*, fmi2_boolean_t*)
    int fmi2_import_terminate(fmi2_import_t*)

    # modes
    int fmi2_import_enter_initialization_mode(fmi2_import_t* fmu)
    int fmi2_import_exit_initialization_mode(fmi2_import_t* fmu)
    int fmi2_import_enter_event_mode(fmi2_import_t* fmu)
    int fmi2_import_enter_continuous_time_mode(fmi2_import_t* fmu)

    # misc
    int fmi2_import_set_debug_logging(fmi2_import_t*, fmi2_boolean_t, size_t, fmi2_string_t*)
    int fmi2_import_reset(fmi2_import_t* fmu)
    int fmi2_import_cancel_step(fmi2_import_t*)
    int fmi2_import_new_discrete_states(fmi2_import_t* fmu, fmi2_event_info_t* eventInfo)
    char* fmi2_import_get_version(fmi2_import_t*)

    # setting
    int fmi2_import_set_time(fmi2_import_t*, fmi2_real_t)
    int fmi2_import_set_integer(fmi2_import_t*, fmi2_value_reference_t *, size_t, fmi2_integer_t *)
    int fmi2_import_set_real(fmi2_import_t*, fmi2_value_reference_t *, size_t, fmi2_real_t *)
    int fmi2_import_set_boolean(fmi2_import_t*, fmi2_value_reference_t *, size_t, fmi2_boolean_t *)
    int fmi2_import_set_string(fmi2_import_t*, fmi2_value_reference_t *, size_t, fmi2_string_t *)

    int fmi2_import_set_continuous_states(fmi2_import_t*, fmi2_real_t *, size_t)
    int fmi2_import_set_real_input_derivatives(fmi2_import_t*, fmi2_value_reference_t *, size_t, fmi2_integer_t *, fmi2_real_t *)

    # getting
    int fmi2_import_get_integer(fmi2_import_t*, fmi2_value_reference_t *, size_t, fmi2_integer_t *)
    int fmi2_import_get_real(fmi2_import_t*, fmi2_value_reference_t *, size_t, fmi2_real_t *)
    int fmi2_import_get_string(fmi2_import_t*, fmi2_value_reference_t *, size_t, fmi2_string_t *)
    int fmi2_import_get_boolean(fmi2_import_t*, fmi2_value_reference_t *, size_t, fmi2_boolean_t *)

    int fmi2_import_get_derivatives(fmi2_import_t*, fmi2_real_t *, size_t)
    int fmi2_import_get_event_indicators(fmi2_import_t*, fmi2_real_t *, size_t)
    int fmi2_import_get_continuous_states(fmi2_import_t*, fmi2_real_t *, size_t)
    int fmi2_import_get_nominals_of_continuous_states(fmi2_import_t* fmu, fmi2_real_t *, size_t nx)
    int fmi2_import_get_directional_derivative(fmi2_import_t*, fmi2_value_reference_t*, size_t, fmi2_value_reference_t*, size_t, fmi2_real_t*, fmi2_real_t*)
    int fmi2_import_get_real_output_derivatives(fmi2_import_t*, fmi2_value_reference_t *, size_t, fmi2_integer_t *, fmi2_real_t *)

    int fmi2_import_get_status(fmi2_import_t* fmu, const fmi2_status_kind_t s, fmi2_status_t* value)
    int fmi2_import_get_integer_status(fmi2_import_t*, int, fmi2_integer_t *)
    int fmi2_import_get_real_status(fmi2_import_t*, int, fmi2_real_t *)
    int fmi2_import_get_boolean_status(fmi2_import_t*, int, fmi2_boolean_t *)
    int fmi2_import_get_string_status(fmi2_import_t*, int, fmi2_string_t *)

    # save states
    int fmi2_import_get_fmu_state(fmi2_import_t*, fmi2_FMU_state_t *)
    int fmi2_import_set_fmu_state(fmi2_import_t*, fmi2_FMU_state_t)
    int fmi2_import_free_fmu_state(fmi2_import_t*, fmi2_FMU_state_t *)
    int fmi2_import_serialized_fmu_state_size(fmi2_import_t*, fmi2_FMU_state_t, size_t *)
    int fmi2_import_serialize_fmu_state(fmi2_import_t*, fmi2_FMU_state_t, fmi2_byte_t *, size_t)
    int fmi2_import_de_serialize_fmu_state(fmi2_import_t*, fmi2_byte_t *, size_t, fmi2_FMU_state_t *)

    #FMI HELPER METHODS (2.0)
    char* fmi2_import_get_GUID(fmi2_import_t*)
    char* fmi2_import_get_description(fmi2_import_t*)
    char* fmi2_import_get_author(fmi2_import_t*)
    char* fmi2_import_get_license(fmi2_import_t*)
    char* fmi2_import_get_generation_tool(fmi2_import_t*)
    char* fmi2_import_get_generation_date_and_time(fmi2_import_t*)
    fmi2_variable_naming_convension_enu_t fmi2_import_get_naming_convention(fmi2_import_t*)
    char* fmi2_import_get_model_name(fmi2_import_t*)

    fmi2_fmu_kind_enu_t fmi2_import_get_fmu_kind(fmi2_import_t*)

    unsigned int fmi2_import_get_capability(fmi2_import_t*, fmi2_capabilities_enu_t)
    double fmi2_import_get_default_experiment_stop(fmi2_import_t*)
    double fmi2_import_get_default_experiment_start(fmi2_import_t*)
    double fmi2_import_get_default_experiment_step(fmi2_import_t*)

    #FMI XML METHODS
    #CONVERTER METHODS
    fmi2_import_integer_variable_t * fmi2_import_get_variable_as_integer(fmi2_import_variable_t *)
    fmi2_import_real_variable_t * fmi2_import_get_variable_as_real(fmi2_import_variable_t *)
    fmi2_import_bool_variable_t * fmi2_import_get_variable_as_boolean(fmi2_import_variable_t *)
    fmi2_import_enum_variable_t * fmi2_import_get_variable_as_enum(fmi2_import_variable_t *)
    fmi2_import_string_variable_t * fmi2_import_get_variable_as_string(fmi2_import_variable_t *)

    #INTEGER
    int fmi2_import_get_integer_type_min(fmi2_import_integer_typedef_t *)
    int fmi2_import_get_integer_type_max(fmi2_import_integer_typedef_t *)

    int fmi2_import_get_integer_variable_max(fmi2_import_integer_variable_t *)
    int fmi2_import_get_integer_variable_min(fmi2_import_integer_variable_t *)
    int fmi2_import_get_integer_variable_start(fmi2_import_integer_variable_t *)

    # unsorted, but does NOT invoke CAPI calls
    size_t fmi2_import_get_number_of_event_indicators(fmi2_import_t*)
    size_t fmi2_import_get_number_of_continuous_states(fmi2_import_t*)
    fmi2_import_variable_list_t* fmi2_import_get_outputs_list(fmi2_import_t*)
    void fmi2_import_get_outputs_dependencies(fmi2_import_t* fmu, size_t** startIndex, size_t** dependency, char** factorKind)
    void fmi2_import_get_derivatives_dependencies(fmi2_import_t* fmu, size_t** startIndex, size_t** dependency, char** factorKind)
    int fmi2_import_get_enum_type_item_value(fmi2_import_enumeration_typedef_t *, unsigned int)
    char* fmi2_import_get_variable_name(fmi2_import_variable_t *)
    fmi2_real_t fmi2_import_get_real_variable_nominal(fmi2_import_real_variable_t *)
    int fmi2_import_get_variable_has_start(fmi2_import_variable_t *)
    char* fmi2_fmu_kind_to_string(fmi2_fmu_kind_enu_t)
    char* fmi2_import_get_string_variable_start(fmi2_import_string_variable_t *)
    double fmi2_import_get_default_experiment_tolerance(fmi2_import_t*)
    int fmi2_import_get_real_type_is_relative_quantity(fmi2_import_real_typedef_t *)
    unsigned int fmi2_import_get_enum_type_size(fmi2_import_enumeration_typedef_t *)
    fmi2_import_variable_list_t* fmi2_import_get_derivatives_list(fmi2_import_t* fmu)
    fmi2_import_real_variable_t* fmi2_import_get_real_variable_derivative_of(fmi2_import_real_variable_t* v)
    fmi2_causality_enu_t fmi2_import_get_causality(fmi2_import_variable_t *)
    fmi2_import_variable_list_t * fmi2_import_get_variable_aliases(fmi2_import_t*, fmi2_import_variable_t *)
    fmi2_import_variable_list_t * fmi2_import_get_variable_list(fmi2_import_t*, int)
    fmi2_import_variable_typedef_t * fmi2_import_get_variable_declared_type(fmi2_import_variable_t *)
    fmi2_boolean_t fmi2_import_get_boolean_variable_start(fmi2_import_bool_variable_t *)
    fmi2_import_enumeration_typedef_t * fmi2_import_get_type_as_enum(fmi2_import_variable_typedef_t *)
    fmi2_import_unit_t * fmi2_import_get_real_type_unit(fmi2_import_real_typedef_t *)
    int fmi2_import_get_enum_variable_min(fmi2_import_enum_variable_t *)
    char* fmi2_import_get_display_unit_name(fmi2_import_display_unit_t *)
    fmi2_real_t fmi2_import_get_real_variable_min(fmi2_import_real_variable_t *)
    fmi2_initial_enu_t fmi2_import_get_initial(fmi2_import_variable_t *)
    int fmi2_import_get_real_type_is_unbounded(fmi2_import_real_typedef_t *)
    fmi2_import_variable_t * fmi2_import_get_variable(fmi2_import_variable_list_t *, size_t)
    char* fmi2_import_get_variable_description(fmi2_import_variable_t *)
    char* fmi2_import_get_model_identifier_CS(fmi2_import_t*)
    char* fmi2_import_get_log_category(fmi2_import_t*, size_t)
    char* fmi2_import_get_log_category_description(fmi2_import_t*, size_t)
    char* fmi2_import_get_last_error(fmi2_import_t*)
    char* fmi2_import_get_enum_type_item_description(fmi2_import_enumeration_typedef_t *, unsigned int)
    fmi2_value_reference_t fmi2_import_get_variable_vr(fmi2_import_variable_t *)
    fmi2_import_display_unit_t * fmi2_import_get_type_display_unit(fmi2_import_real_typedef_t *)
    fmi2_import_variable_t * fmi2_import_get_variable_by_name(fmi2_import_t*, char*)
    double fmi2_import_get_real_type_min(fmi2_import_real_typedef_t *)
    fmi2_import_variable_t * fmi2_import_get_variable_by_vr(fmi2_import_t*, fmi2_base_type_enu_t, fmi2_value_reference_t)
    fmi2_real_t fmi2_import_get_real_variable_max(fmi2_import_real_variable_t *)
    fmi2_import_variable_list_t * fmi2_import_get_derivatives_list(fmi2_import_t*)
    void fmi2_import_free_variable_list(fmi2_import_variable_list_t *)
    void fmi2_log_forwarding(fmi2_component_t, fmi2_string_t, fmi2_status_t, fmi2_string_t, fmi2_string_t,...)
    fmi2_import_t* fmi2_import_parse_xml(fmi_import_context_t *, char*, fmi2_xml_callbacks_t *)
    fmi2_real_t fmi2_import_get_real_variable_start(fmi2_import_real_variable_t *)
    void fmi2_import_free(fmi2_import_t*)
    size_t fmi2_import_get_log_categories_num(fmi2_import_t*)
    char* fmi2_import_get_type_quantity(fmi2_import_variable_typedef_t *)
    char* fmi2_import_get_unit_name(fmi2_import_unit_t *)
    fmi2_import_variable_t * fmi2_import_get_variable_alias_base(fmi2_import_t*, fmi2_import_variable_t *)
    char* fmi2_import_get_model_identifier_ME(fmi2_import_t*)
    void fmi2_import_destroy_dllfmu(fmi2_import_t*)
    double fmi2_import_get_real_type_nominal(fmi2_import_real_typedef_t *)
    fmi2_import_real_typedef_t * fmi2_import_get_type_as_real(fmi2_import_variable_typedef_t *)
    char* fmi2_import_get_copyright(fmi2_import_t*)
    fmi2_real_t fmi2_import_convert_to_display_unit(fmi2_real_t, fmi2_import_display_unit_t *, int)
    fmi2_import_unit_t * fmi2_import_get_real_variable_unit(fmi2_import_real_variable_t *)
    fmi2_import_display_unit_t * fmi2_import_get_real_variable_display_unit(fmi2_import_real_variable_t *)
    int fmi2_import_get_enum_variable_max(fmi2_import_enum_variable_t *)
    char* fmi2_import_get_type_name(fmi2_import_variable_typedef_t *)
    double fmi2_import_get_real_type_max(fmi2_import_real_typedef_t *)
    size_t fmi2_import_get_variable_list_size(fmi2_import_variable_list_t *)
    fmi2_variable_alias_kind_enu_t fmi2_import_get_variable_alias_kind(fmi2_import_variable_t *)
    int fmi2_import_get_enum_variable_start(fmi2_import_enum_variable_t *)
    fmi2_base_type_enu_t fmi2_import_get_variable_base_type(fmi2_import_variable_t *)
    fmi2_variability_enu_t fmi2_import_get_variability(fmi2_import_variable_t *)
    char* fmi2_import_get_type_description(fmi2_import_variable_typedef_t *)
    fmi2_import_integer_typedef_t * fmi2_import_get_type_as_int(fmi2_import_variable_typedef_t *)
    char* fmi2_import_get_enum_type_item_name(fmi2_import_enumeration_typedef_t *, unsigned int)
    char* fmi2_import_get_model_version(fmi2_import_t*)
    fmi2_boolean_t fmi2_import_get_real_variable_relative_quantity(fmi2_import_real_variable_t* v)
    fmi2_boolean_t fmi2_import_get_real_variable_unbounded(fmi2_import_real_variable_t* v)

    # unsorted & unused!!!
    size_t fmi2_import_get_derivative_index(fmi2_import_variable_t *)
    void  fmi2_default_callback_logger(fmi2_component_t c, fmi2_string_t instanceName, fmi2_status_t status, fmi2_string_t category, fmi2_string_t message, ...)
    void fmi2_import_init_logger(jm_callbacks *, fmi2_callback_functions_t *)
    fmi2_import_unit_t * fmi2_import_get_unit(fmi2_import_unit_definitions_t *, unsigned int)
    fmi2_import_variable_typedef_t * fmi2_import_get_typedef(fmi2_import_type_definitions_t *, unsigned int)
    size_t fmi2_import_get_input_index(fmi2_import_variable_t *)
    size_t fmi2_SI_base_unit_exp_to_string(int *, size_t, char*)
    double fmi2_import_get_SI_unit_factor(fmi2_import_unit_t *)
    unsigned int fmi2_import_get_unit_definitions_number(fmi2_import_unit_definitions_t *)
    fmi2_import_variable_list_t * fmi2_import_join_var_list(fmi2_import_variable_list_t *, fmi2_import_variable_list_t *)
    size_t fmi2_import_get_state_index(fmi2_import_variable_t *)
    int fmi2_import_var_list_push_back(fmi2_import_variable_list_t *, fmi2_import_variable_t *)
    fmi2_import_variable_list_t * fmi2_import_append_to_var_list(fmi2_import_variable_list_t *, fmi2_import_variable_t *)
    void fmi2_log_forwarding_v(fmi2_component_t, fmi2_string_t, int, fmi2_string_t, fmi2_string_t, __va_list_tag *)
    fmi2_base_type_enu_t fmi2_import_get_base_type(fmi2_import_variable_typedef_t *)
    unsigned int fmi2_import_get_unit_display_unit_number(fmi2_import_unit_t *)
    fmi2_value_reference_t * fmi2_import_get_value_referece_list(fmi2_import_variable_list_t *)
    int fmi2_import_clear_last_error(fmi2_import_t*)
    unsigned int fmi2_import_get_type_definition_number(fmi2_import_type_definitions_t *)
    char* fmi2_SI_base_unit_to_string(fmi2_SI_base_units_enu_t)
    unsigned int fmi2_import_get_enum_type_max(fmi2_import_enumeration_typedef_t *)
    fmi2_import_unit_definitions_t * fmi2_import_get_unit_definitions(fmi2_import_t*)
    void fmi2_import_get_dependencies_derivatives_on_inputs(fmi2_import_t*, size_t * *, size_t * *, char* *)
    fmi2_import_variable_list_t * fmi2_import_prepend_to_var_list(fmi2_import_variable_list_t *, fmi2_import_variable_t *)
    void fmi2_import_set_debug_mode(fmi2_import_t*, int)
    char* fmi2_import_get_vendor_name(fmi2_import_t*, size_t)
    double fmi2_import_get_SI_unit_offset(fmi2_import_unit_t *)
    void fmi2_import_expand_variable_references(fmi2_import_t*, char*, char*, size_t)
    size_t fmi2_import_get_variable_original_order(fmi2_import_variable_t *)
    char* fmi2_variability_to_string(fmi2_variability_enu_t)
    fmi2_import_variable_list_t * fmi2_import_create_var_list(fmi2_import_t*, fmi2_import_variable_t *)
    fmi2_real_t fmi2_import_get_display_unit_offset(fmi2_import_display_unit_t *)
    fmi2_import_variable_list_t * fmi2_import_get_sublist(fmi2_import_variable_list_t *, size_t, size_t)
    fmi2_import_variable_list_t * fmi2_import_clone_variable_list(fmi2_import_variable_list_t *)
    unsigned int fmi2_import_get_enum_type_min(fmi2_import_enumeration_typedef_t *)
    char* fmi2_causality_to_string(fmi2_causality_enu_t)
    fmi2_real_t fmi2_import_convert_from_display_unit(fmi2_real_t, fmi2_import_display_unit_t *, int)
    double fmi2_import_convert_from_SI_base_unit(double, fmi2_import_unit_t *)
    int * fmi2_import_get_SI_unit_exponents(fmi2_import_unit_t *)
    char* fmi2_import_get_enum_type_value_name(fmi2_import_enumeration_typedef_t *, int)
    fmi2_import_display_unit_t * fmi2_import_get_unit_display_unit(fmi2_import_unit_t *, size_t)
    fmi2_initial_enu_t fmi2_get_valid_initial(fmi2_variability_enu_t, fmi2_causality_enu_t, fmi2_initial_enu_t)
    fmi2_import_variable_list_t * fmi2_import_filter_variables(fmi2_import_variable_list_t *, fmi2_import_variable_filter_function_ft, void *)
    void fmi2_import_get_dependencies_outputs_on_states(fmi2_import_t*, size_t * *, size_t * *, char* *)
    char* fmi2_base_type_to_string(fmi2_base_type_enu_t)
    size_t fmi2_import_get_vendors_num(fmi2_import_t*)
    void fmi2_import_collect_model_counts(fmi2_import_t*, fmi2_import_model_counts_t *)
    char* fmi2_dependency_factor_kind_to_string(fmi2_dependency_factor_kind_enu_t)
    char* fmi2_status_to_string(int)
    char* fmi2_import_get_model_standard_version(fmi2_import_t*)
    size_t fmi2_import_get_output_index(fmi2_import_variable_t *)
    fmi2_import_unit_t * fmi2_import_get_base_unit(fmi2_import_display_unit_t *)
    fmi2_initial_enu_t fmi2_get_default_initial(fmi2_variability_enu_t, fmi2_causality_enu_t)
    double fmi2_import_convert_to_SI_base_unit(double, fmi2_import_unit_t *)
    void fmi2_import_get_dependencies_derivatives_on_states(fmi2_import_t*, size_t * *, size_t * *, char* *)
    char* fmi2_naming_convention_to_string(fmi2_variable_naming_convension_enu_t)
    fmi2_import_type_definitions_t * fmi2_import_get_type_definitions(fmi2_import_t*)
    char* fmi2_initial_to_string(fmi2_initial_enu_t)
    char* fmi2_capability_to_string(fmi2_capabilities_enu_t)
    fmi2_import_variable_list_t * fmi2_import_get_states_list(fmi2_import_t*)
    fmi2_real_t fmi2_import_get_display_unit_factor(fmi2_import_display_unit_t *)

    fmi2_import_variable_list_t* fmi2_import_get_initial_unknowns_list(fmi2_import_t* fmu)
