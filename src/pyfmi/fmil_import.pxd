#!/usr/bin/env python
# -*- coding: utf-8 -*-

#    Copyright (C) 2009 Modelon AB
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
cdef extern from "stdlib.h":
    ctypedef long unsigned int size_t

    void *malloc(size_t)
    void free(void *ptr)
    void *calloc(size_t, size_t)
    void *realloc(void *, size_t)

#SEE http://wiki.cython.org/FAQ#HowdoIusevariableargs.
cdef extern from "stdarg.h":
    ctypedef struct va_list:
        pass
    ctypedef struct fake_type:
        pass
    void va_start(va_list, void* arg)
    #void* va_arg(va_list, fake_type)
    void va_end(va_list)
    int vsnprintf(char *str, size_t size, char *format, va_list ap)

#cdef extern from 'FMI1/fmi1_import.h':
cdef extern from 'fmilib.h':

    #FMI VARIABLE TYPE DEFINITIONS
    ctypedef double fmi1_real_t
    ctypedef double fmi2_real_t
    ctypedef unsigned int fmi1_value_reference_t
    ctypedef char   fmi1_boolean_t
    ctypedef int    fmi2_boolean_t
    ctypedef void * fmi1_component_t
    ctypedef void * fmi2_component_t
    ctypedef char * fmi1_string_t
    ctypedef char * fmi2_string_t
    ctypedef int    fmi1_integer_t
    ctypedef int    fmi2_integer_t
    ctypedef long unsigned int size_t
    ctypedef void * jm_voidp
    ctypedef char * jm_string
    ctypedef char   fmi2_byte_t
    ctypedef void * fmi2_FMU_state_t
    ctypedef void * fmi2_component_environment_t
    ctypedef size_t fmi2_value_reference_t

    #STRUCTS
    ctypedef enum jm_log_level_enu_t:
        jm_log_level_nothing = 0
        jm_log_level_fatal = 1
        jm_log_level_error = 2
        jm_log_level_warning = 3
        jm_log_level_info = 4
        jm_log_level_verbose = 5
        jm_log_level_debug = 6
        jm_log_level_all = 7

    ctypedef enum fmi1_status_t:
        fmi1_status_ok = 0
        fmi1_status_warning = 1
        fmi1_status_discard = 2
        fmi1_status_error = 3
        fmi1_status_fatal = 4
        fmi1_status_pending = 5

    ctypedef enum fmi2_status_t:
        fmi2_status_ok = 0
        fmi2_status_warning = 1
        fmi2_status_discard = 2
        fmi2_status_error = 3
        fmi2_status_fatal = 4
        fmi2_status_pending = 5

    cdef enum fmi1_variable_alias_kind_enu_t:
        fmi1_variable_is_negated_alias = -1
        fmi1_variable_is_not_alias = 0
        fmi1_variable_is_alias = 1

    cdef enum fmi2_variable_alias_kind_enu_t:
        fmi2_variable_is_not_alias = 0
        fmi2_variable_is_alias = 1
    ctypedef fmi2_variable_alias_kind_enu_t fmi2_variable_alias_kind_enu_t

    cdef enum fmi1_base_type_enu_t:
        fmi1_base_type_real = 0
        fmi1_base_type_int = 1
        fmi1_base_type_bool = 2
        fmi1_base_type_str = 3
        fmi1_base_type_enum = 4

    cdef enum fmi2_base_type_enu_t:
        fmi2_base_type_real = 0
        fmi2_base_type_int = 1
        fmi2_base_type_bool = 2
        fmi2_base_type_str = 3
        fmi2_base_type_enum = 4
    ctypedef fmi2_base_type_enu_t fmi2_base_type_enu_t

    ctypedef enum jm_status_enu_t:
        jm_status_error = -1
        jm_status_success = 0
        jm_status_warning = 1

    ctypedef enum fmi_version_enu_t:
        fmi_version_unknown_enu = 0
        fmi_version_1_enu = 1
        fmi_version_2_0_enu = 2
        fmi_version_unsupported_enu = 3

    cdef enum fmi1_causality_enu_t:
        fmi1_causality_enu_input = 0
        fmi1_causality_enu_output = 1
        fmi1_causality_enu_internal = 2
        fmi1_causality_enu_none = 3

    cdef enum fmi2_causality_enu_t:
        fmi2_causality_enu_parameter = 0
        fmi2_causality_enu_input = 1
        fmi2_causality_enu_output = 2
        fmi2_causality_enu_local = 3
        fmi2_causality_enu_unknown = 4
    ctypedef fmi2_causality_enu_t fmi2_causality_enu_t

    cdef enum fmi1_fmu_kind_enu_t:
        fmi1_fmu_kind_enu_me = 0
        fmi1_fmu_kind_enu_cs_standalone = 1
        fmi1_fmu_kind_enu_cs_tool = 2

    cdef enum fmi2_fmu_kind_enu_t:
        fmi2_fmu_kind_unknown = 0
        fmi2_fmu_kind_me = 1
        fmi2_fmu_kind_cs = 2
        fmi2_fmu_kind_me_and_cs = 3
    ctypedef fmi2_fmu_kind_enu_t fmi2_fmu_kind_enu_t

    cdef enum fmi1_variability_enu_t:
        fmi1_variability_enu_constant = 0
        fmi1_variability_enu_parameter = 1
        fmi1_variability_enu_discrete = 2
        fmi1_variability_enu_continuous = 3

    cdef enum fmi2_variability_enu_t:
        fmi2_variability_enu_constant = 0
        fmi2_variability_enu_fixed = 1
        fmi2_variability_enu_tunable = 2
        fmi2_variability_enu_discrete = 3
        fmi2_variability_enu_continuous = 4
        fmi2_variability_enu_unknown = 5
    ctypedef fmi2_variability_enu_t fmi2_variability_enu_t

    cdef enum fmi1_variable_naming_convension_enu_t:
        fmi1_naming_enu_flat = 0
        fmi1_naming_enu_structured = 1

    cdef enum fmi2_variable_naming_convension_enu_t:
        fmi2_naming_enu_flat = 0
        fmi2_naming_enu_structured = 1
        fmi2_naming_enu_unknown = 2
    ctypedef fmi2_variable_naming_convension_enu_t fmi2_variable_naming_convension_enu_t

    cdef enum fmi1_status_kind_t:
        fmi1_do_step_status = 0
        fmi1_pending_status = 1
        fmi1_last_successful_time = 2

    cdef enum fmi2_status_kind_t:
        fmi2_do_step_status = 0
        fmi2_pending_status = 1
        fmi2_last_successful_time = 2
        fmi2_terminated = 3

    ctypedef struct fmi1_event_info_t:
        fmi1_boolean_t iterationConverged
        fmi1_boolean_t stateValueReferencesChanged
        fmi1_boolean_t stateValuesChanged
        fmi1_boolean_t terminateSimulation
        fmi1_boolean_t upcomingTimeEvent
        fmi1_real_t nextEventTime

    ctypedef struct fmi2_event_info_t:
        fmi2_boolean_t iterationConverged
        fmi2_boolean_t stateValueReferencesChanged
        fmi2_boolean_t stateValuesChanged
        fmi2_boolean_t terminateSimulation
        fmi2_boolean_t upcomingTimeEvent
        fmi2_real_t nextEventTime

    cdef enum fmi2_dependency_factor_kind_enu_t:
        fmi2_dependency_factor_kind_nonlinear = 0
        fmi2_dependency_factor_kind_fixed = 1
        fmi2_dependency_factor_kind_discrete = 2
        fmi2_dependency_factor_kind_num = 3
    ctypedef fmi2_dependency_factor_kind_enu_t fmi2_dependency_factor_kind_enu_t

    cdef enum fmi2_initial_enu_t:
        fmi2_initial_enu_exact = 0
        fmi2_initial_enu_approx = 1
        fmi2_initial_enu_calculated = 2
        fmi2_initial_enu_unknown = 3
    ctypedef fmi2_initial_enu_t fmi2_initial_enu_t

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
        fmi2_cs_canHandleEvents = 10
        fmi2_cs_canInterpolateInputs = 11
        fmi2_cs_maxOutputDerivativeOrder = 12
        fmi2_cs_canRunAsynchronuously = 13
        fmi2_cs_canSignalEvents = 14
        fmi2_cs_canBeInstantiatedOnlyOncePerProcess = 15
        fmi2_cs_canNotUseMemoryManagementFunctions = 16
        fmi2_cs_canGetAndSetFMUstate = 17
        fmi2_cs_canSerializeFMUstate = 18
        fmi2_capabilities_Num = 19
    ctypedef fmi2_capabilities_enu_t fmi2_capabilities_enu_t

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
    ctypedef fmi2_SI_base_units_enu_t fmi2_SI_base_units_enu_t

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

    ctypedef int(*jm_compare_ft)(void *, void *)
    ctypedef jm_voidp(*jm_malloc_f)(size_t)
    ctypedef jm_voidp(*jm_calloc_f)(size_t, size_t)
    ctypedef jm_voidp(*jm_realloc_f)(void *, size_t)
    ctypedef void(*jm_free_f)(jm_voidp)
    ctypedef void *(*fmi1_callback_allocate_memory_ft)(size_t, size_t)
    ctypedef void(*fmi1_callback_free_memory_ft)(void *)
    ctypedef void(*fmi1_callback_logger_ft)(fmi1_component_t c, fmi1_string_t instanceName, fmi1_status_t status, fmi1_string_t category, fmi1_string_t message, ...)
    ctypedef void(*fmi2_callback_logger_ft)(fmi2_component_t c,fmi2_string_t instanceName, fmi2_status_t status, fmi2_string_t category,fmi2_string_t message,...)
    ctypedef void(*fmi1_step_finished_ft)(fmi1_component_t c, fmi1_status_t status)
    ctypedef void(*fmi2_step_finished_ft)(fmi2_component_environment_t env, fmi2_status_t status)
    #ctypedef void (*jm_logger_f)(jm_callbacks* c, jm_string module, jm_log_level_enu_t log_level, jm_string message)
    ctypedef void (*jm_logger_f)(jm_callbacks* c, jm_string module, int log_level, jm_string message)
    ctypedef void *(*fmi2_callback_allocate_memory_ft)(size_t, size_t)
    ctypedef void(*fmi2_callback_free_memory_ft)(void *)
    ctypedef int(*fmi2_xml_element_start_handle_ft)(void *, char *, void *, char *, char * *)
    ctypedef int(*fmi2_xml_element_data_handle_ft)(void *, char *, int)
    ctypedef int(*fmi2_xml_element_end_handle_ft)(void *, char *)
    ctypedef int(*fmi2_import_variable_filter_function_ft)(fmi2_import_variable_t *, void *)



    cdef struct fmi2_xml_callbacks_t:
        fmi2_xml_element_start_handle_ft startHandle
        fmi2_xml_element_data_handle_ft dataHandle
        fmi2_xml_element_end_handle_ft endHandle
        void * context
    ctypedef fmi2_xml_callbacks_t fmi2_xml_callbacks_t

    cdef struct jm_callbacks:
        jm_malloc_f malloc
        jm_calloc_f calloc
        jm_realloc_f realloc
        jm_free_f free
        jm_logger_f logger
        jm_log_level_enu_t log_level
        jm_voidp context
        char * errMessageBuffer

    ctypedef struct fmi1_callback_functions_t:
        fmi1_callback_logger_ft logger
        fmi1_callback_allocate_memory_ft allocateMemory
        fmi1_callback_free_memory_ft freeMemory
        fmi1_step_finished_ft stepFinished

    ctypedef struct fmi2_callback_functions_t:
        fmi2_callback_logger_ft logger
        fmi2_callback_allocate_memory_ft allocateMemory
        fmi2_callback_free_memory_ft freeMemory
        fmi2_step_finished_ft stepFinished
        fmi2_component_environment_t componentEnvironment

    cdef struct jm_named_ptr:
        jm_voidp ptr
        jm_string name

    cdef struct fmi1_xml_model_description_t:
        pass

    cdef struct fmi1_import_t:
        pass
    ctypedef fmi1_import_t fmi1_import_t

    cdef struct fmi2_import_t:
        pass
    ctypedef fmi2_import_t fmi2_import_t

    cdef struct fmi1_xml_vendor_t:
        pass
    ctypedef fmi1_xml_vendor_t fmi1_import_vendor_t

    cdef struct fmi1_xml_capabilities_t:
        pass
    ctypedef fmi1_xml_capabilities_t fmi1_import_capabilities_t

    cdef struct fmi_xml_context_t:
        pass
    ctypedef fmi_xml_context_t fmi_xml_context_t

    ctypedef fmi_xml_context_t fmi_import_context_t

    cdef struct fmi1_xml_variable_t:
        pass
    ctypedef fmi1_xml_variable_t fmi1_import_variable_t


    cdef struct fmi1_xml_real_variable_t:
        pass
    ctypedef fmi1_xml_real_variable_t fmi1_import_real_variable_t
    cdef struct fmi2_xml_real_variable_t:
        pass
    ctypedef fmi2_xml_real_variable_t fmi2_import_real_variable_t

    cdef struct fmi1_xml_display_unit_t:
        pass
    ctypedef fmi1_xml_display_unit_t fmi1_import_display_unit_t
    cdef struct fmi2_xml_display_unit_t:
        pass
    ctypedef fmi2_xml_display_unit_t fmi2_import_display_unit_t

    cdef struct fmi1_xml_unit_definitions_t:
        pass
    ctypedef fmi1_xml_unit_definitions_t fmi1_import_unit_definitions_t
    cdef struct fmi2_xml_unit_definitions_t:
        pass
    ctypedef fmi2_xml_unit_definitions_t fmi2_import_unit_definitions_t

    cdef struct fmi1_xml_vendor_list_t:
        pass
    ctypedef fmi1_xml_vendor_list_t fmi1_import_vendor_list_t

    cdef struct fmi1_import_variable_list_t:
        pass
    cdef struct fmi2_import_variable_list_t:
        pass
    ctypedef fmi2_import_variable_list_t fmi2_import_variable_list_t

    cdef struct fmi1_xml_variable_typedef_t:
        pass
    ctypedef fmi1_xml_variable_typedef_t fmi1_import_variable_typedef_t
    cdef struct fmi2_xml_variable_typedef_t:
        pass
    ctypedef fmi2_xml_variable_typedef_t fmi2_import_variable_typedef_t

    cdef struct fmi1_xml_integer_variable_t:
        pass
    ctypedef fmi1_xml_integer_variable_t fmi1_import_integer_variable_t
    cdef struct fmi2_xml_integer_variable_t:
        pass
    ctypedef fmi2_xml_integer_variable_t fmi2_import_integer_variable_t

    cdef struct fmi1_xml_real_typedef_t:
        pass
    ctypedef fmi1_xml_real_typedef_t fmi1_import_real_typedef_t
    cdef struct fmi2_xml_real_typedef_t:
        pass
    ctypedef fmi2_xml_real_typedef_t fmi2_import_real_typedef_t

    cdef struct fmi1_xml_enum_variable_t:
        pass
    ctypedef fmi1_xml_enum_variable_t fmi1_import_enum_variable_t
    cdef struct fmi2_xml_enum_variable_t:
        pass
    ctypedef fmi2_xml_enum_variable_t fmi2_import_enum_variable_t

    cdef struct fmi1_xml_type_definitions_t:
        pass
    ctypedef fmi1_xml_type_definitions_t fmi1_import_type_definitions_t
    cdef struct fmi2_xml_type_definitions_t:
        pass
    ctypedef fmi2_xml_type_definitions_t fmi2_import_type_definitions_t

    cdef struct fmi1_xml_enumeration_typedef_t:
        pass
    ctypedef fmi1_xml_enumeration_typedef_t fmi1_import_enumeration_typedef_t
    cdef struct fmi2_xml_enumeration_typedef_t:
        pass
    ctypedef fmi2_xml_enumeration_typedef_t fmi2_import_enumeration_typedef_t

    cdef struct fmi1_xml_integer_typedef_t:
        pass
    ctypedef fmi1_xml_integer_typedef_t fmi1_import_integer_typedef_t
    cdef struct fmi2_xml_integer_typedef_t:
        pass
    ctypedef fmi2_xml_integer_typedef_t fmi2_import_integer_typedef_t

    cdef struct fmi1_xml_annotation_t:
        pass
    ctypedef fmi1_xml_annotation_t fmi1_import_annotation_t

    cdef struct fmi1_xml_unit_t:
        pass
    ctypedef fmi1_xml_unit_t fmi1_import_unit_t
    cdef struct fmi2_xml_unit_t:
        pass
    ctypedef fmi2_xml_unit_t fmi2_import_unit_t

    cdef struct fmi1_xml_bool_variable_t:
        pass
    ctypedef fmi1_xml_bool_variable_t fmi1_import_bool_variable_t
    cdef struct fmi2_xml_bool_variable_t:
        pass
    ctypedef fmi2_xml_bool_variable_t fmi2_import_bool_variable_t

    cdef struct fmi1_xml_string_variable_t:
        pass
    ctypedef fmi1_xml_string_variable_t fmi1_import_string_variable_t
    cdef struct fmi2_xml_string_variable_t:
        pass
    ctypedef fmi2_xml_string_variable_t fmi2_import_string_variable_t

    cdef struct __va_list_tag:
        pass

    #FMI SPECIFICATION METHODS (2.0)
    int fmi2_import_do_step(fmi2_import_t *, fmi2_real_t, fmi2_real_t, fmi2_boolean_t)
    int fmi2_import_get_event_indicators(fmi2_import_t *, fmi2_real_t *, size_t)
    int fmi2_import_completed_integrator_step(fmi2_import_t *, fmi2_boolean_t *)
    int fmi2_import_initialize_slave(fmi2_import_t *, fmi2_real_t, fmi2_real_t, fmi2_boolean_t, fmi2_real_t)
    int fmi2_import_get_derivatives(fmi2_import_t *, fmi2_real_t *, size_t)
    int fmi2_import_reset_slave(fmi2_import_t *)
    int fmi2_import_serialize_fmu_state(fmi2_import_t *, fmi2_FMU_state_t, fmi2_byte_t *, size_t)
    int fmi2_import_set_real(fmi2_import_t *, fmi2_value_reference_t *, size_t, fmi2_real_t *)
    int fmi2_import_get_boolean(fmi2_import_t *, fmi2_value_reference_t *, size_t, fmi2_boolean_t *)
    int fmi2_import_get_state_value_references(fmi2_import_t *, fmi2_value_reference_t *, size_t)
    int fmi2_import_set_debug_logging(fmi2_import_t *, fmi2_boolean_t, size_t, fmi2_string_t*)
    int fmi2_import_eventUpdate(fmi2_import_t *, fmi2_boolean_t, fmi2_event_info_t *)
    int fmi2_import_set_time(fmi2_import_t *, fmi2_real_t)
    int fmi2_import_cancel_step(fmi2_import_t *)
    int fmi2_import_set_boolean(fmi2_import_t *, fmi2_value_reference_t *, size_t, fmi2_boolean_t *)
    int fmi2_import_set_continuous_states(fmi2_import_t *, fmi2_real_t *, size_t)
    int fmi2_import_set_string(fmi2_import_t *, fmi2_value_reference_t *, size_t, fmi2_string_t *)
    int fmi2_import_terminate(fmi2_import_t *)
    int fmi2_import_get_real_status(fmi2_import_t *, int, fmi2_real_t *)
    int fmi2_import_serialized_fmu_state_size(fmi2_import_t *, fmi2_FMU_state_t, size_t *)
    int fmi2_import_get_nominal_continuous_states(fmi2_import_t *, fmi2_real_t *, size_t)
    int fmi2_import_get_real(fmi2_import_t *, fmi2_value_reference_t *, size_t, fmi2_real_t *)
    int fmi2_import_get_continuous_states(fmi2_import_t *, fmi2_real_t *, size_t)
    int fmi2_import_free_fmu_state(fmi2_import_t *, fmi2_FMU_state_t *)
    int fmi2_import_completed_event_iteration(fmi2_import_t *)
    void fmi2_import_get_dependencies_outputs_on_inputs(fmi2_import_t *, size_t * *, size_t * *, char * *)
    void fmi2_import_free_model_instance(fmi2_import_t *)
    size_t fmi2_import_get_number_of_event_indicators(fmi2_import_t *)
    size_t fmi2_import_get_number_of_continuous_states(fmi2_import_t *)


    #FMI SPECIFICATION METHODS (1.0)
    unsigned int fmi1_import_get_number_of_event_indicators(fmi1_import_t *)
    unsigned int fmi1_import_get_number_of_continuous_states(fmi1_import_t *)
    int fmi1_import_get_state_value_references(fmi1_import_t *, fmi1_value_reference_t *, size_t)
    int fmi1_import_get_nominal_continuous_states(fmi1_import_t *, fmi1_real_t *, size_t)
    int fmi1_import_get_event_indicators(fmi1_import_t *, fmi1_real_t *, size_t)
    int fmi1_import_set_time(fmi1_import_t *, fmi1_real_t)
    int fmi1_import_set_debug_logging(fmi1_import_t *, fmi1_boolean_t)
    int fmi1_import_get_continuous_states(fmi1_import_t *, fmi1_real_t *, size_t)
    #int fmi1_import_instantiate_model(fmi1_import_t *, fmi1_string_t, fmi1_string_t, fmi1_boolean_t)
    int fmi1_import_instantiate_model(fmi1_import_t *, fmi1_string_t)
    int fmi1_import_set_continuous_states(fmi1_import_t *, fmi1_real_t *, size_t)
    int fmi1_import_get_status(fmi1_import_t *, int, int *)
    int fmi1_import_completed_integrator_step(fmi1_import_t *, fmi1_boolean_t *)
    int fmi1_import_reset_slave(fmi1_import_t *)
    int fmi1_import_eventUpdate(fmi1_import_t *, fmi1_boolean_t, fmi1_event_info_t *)
    int fmi1_import_get_canSignalEvents(fmi1_import_capabilities_t *)
    int fmi1_import_initialize(fmi1_import_t *, fmi1_boolean_t, fmi1_real_t, fmi1_event_info_t *)
    int fmi1_import_terminate_slave(fmi1_import_t *)
    int fmi1_import_terminate(fmi1_import_t *)
    int fmi1_import_cancel_step(fmi1_import_t *)
    #int fmi1_import_instantiate_slave(fmi1_import_t *, fmi1_string_t, fmi1_string_t, fmi1_string_t, fmi1_string_t, fmi1_real_t, fmi1_boolean_t, fmi1_boolean_t, fmi1_boolean_t)
    int fmi1_import_instantiate_slave(fmi1_import_t *, fmi1_string_t, fmi1_string_t, fmi1_string_t, fmi1_real_t, fmi1_boolean_t, fmi1_boolean_t)
    int fmi1_import_initialize_slave(fmi1_import_t *, fmi1_real_t, fmi1_boolean_t, fmi1_real_t)
    int fmi1_import_get_derivatives(fmi1_import_t *, fmi1_real_t *, size_t)
    int fmi1_import_do_step(fmi1_import_t *, fmi1_real_t, fmi1_real_t, fmi1_boolean_t)

    int fmi1_import_set_integer(fmi1_import_t *, fmi1_value_reference_t *, size_t, fmi1_integer_t *)
    int fmi1_import_get_integer(fmi1_import_t *, fmi1_value_reference_t *, size_t, fmi1_integer_t *)
    int fmi1_import_get_string(fmi1_import_t *, fmi1_value_reference_t *, size_t, fmi1_string_t *)
    int fmi1_import_set_string(fmi1_import_t *, fmi1_value_reference_t *, size_t, fmi1_string_t *)
    int fmi1_import_set_real(fmi1_import_t *, fmi1_value_reference_t *, size_t, fmi1_real_t *)
    int fmi1_import_get_real(fmi1_import_t *, fmi1_value_reference_t *, size_t, fmi1_real_t *)
    int fmi1_import_set_boolean(fmi1_import_t *, fmi1_value_reference_t *, size_t, fmi1_boolean_t *)
    int fmi1_import_get_boolean(fmi1_import_t *, fmi1_value_reference_t *, size_t, fmi1_boolean_t *)

    char * fmi1_import_get_entry_point(fmi1_import_t *)
    char * fmi1_import_get_generation_date_and_time(fmi1_import_t *)
    char * fmi1_import_get_GUID(fmi1_import_t *)

    #FMI HELPER METHODS (2.0)
    char * fmi2_get_types_platform()
    char * fmi2_import_get_GUID(fmi2_import_t *)
    char * fmi2_import_get_author(fmi2_import_t *)
    char * fmi2_import_get_generation_date_and_time(fmi2_import_t *)
    char * fmi2_import_get_model_name(fmi2_import_t *)
    fmi2_fmu_kind_enu_t fmi2_import_get_fmu_kind(fmi2_import_t *)
    char * fmi2_import_get_description(fmi2_import_t *)
    int fmi2_import_create_dllfmu(fmi2_import_t *, fmi2_fmu_kind_enu_t, fmi2_callback_functions_t *)
    char * fmi2_import_get_license(fmi2_import_t *)
    char * fmi2_import_get_version(fmi2_import_t *)
    unsigned int fmi2_import_get_capability(fmi2_import_t *, fmi2_capabilities_enu_t)
    double fmi2_import_get_default_experiment_stop(fmi2_import_t *)
    double fmi2_import_get_default_experiment_start(fmi2_import_t *)
    char * fmi2_import_get_generation_tool(fmi2_import_t *)

    #FMI HELPER METHODS (1.0)
    char * fmi1_import_get_version(fmi1_import_t *)
    int fmi1_import_create_dllfmu(fmi1_import_t *, fmi1_callback_functions_t, int)
    void fmi1_log_forwarding(fmi1_component_t c, fmi1_string_t instanceName, fmi1_status_t status, fmi1_string_t category, fmi1_string_t message,...)
    char * fmi_import_get_dll_path(char *, char *, jm_callbacks *)
    char * fmi_import_get_model_description_path(char *, jm_callbacks *)
    void fmi1_import_destroy_dllfmu(fmi1_import_t *)
    void fmi1_import_free_slave_instance(fmi1_import_t *)
    void fmi1_import_free_model_instance(fmi1_import_t *)
    void fmi1_import_free(fmi1_import_t *)
    fmi_import_context_t * fmi_import_allocate_context(jm_callbacks *)
    char * fmi1_import_get_mime_type(fmi1_import_t *)
    char * fmi1_import_get_last_error(fmi1_import_t *)
    void fmi_import_free_context(fmi_import_context_t *)
    char * fmi1_base_type_to_string(fmi1_base_type_enu_t)
    fmi1_import_t * fmi1_import_parse_xml(fmi_import_context_t *, char *)
    char * fmi1_variability_to_string(fmi1_variability_enu_t)
    char * fmi1_fmu_kind_to_string(fmi1_fmu_kind_enu_t)
    char * fmi1_get_platform()
    char * fmi1_status_to_string(int)
    fmi_version_enu_t fmi_import_get_fmi_version(fmi_import_context_t*, char*, char*)
    int fmi_import_rmdir(jm_callbacks*, char *)


    #FMI XML METHODS

    ###Model information
    double fmi1_import_get_default_experiment_tolerance(fmi1_import_t *)
    double fmi1_import_get_default_experiment_stop(fmi1_import_t *)
    double fmi1_import_get_default_experiment_start(fmi1_import_t *)
    char * fmi1_import_get_author(fmi1_import_t *)
    char * fmi1_import_get_description(fmi1_import_t *)
    char * fmi1_import_get_types_platform(fmi1_import_t *)
    char * fmi1_import_get_generation_tool(fmi1_import_t *)
    char * fmi1_import_get_model_version(fmi1_import_t *)
    char * fmi1_import_get_model_types_platform(fmi1_import_t *)
    char * fmi1_import_get_model_name(fmi1_import_t *)
    char * fmi1_import_get_model_identifier(fmi1_import_t *)
    char * fmi1_import_get_vendor_name(fmi1_import_vendor_t *)
    fmi1_import_vendor_list_t * fmi1_import_get_vendor_list(fmi1_import_t *)
    char * fmi1_naming_convention_to_string(fmi1_variable_naming_convension_enu_t)


    fmi1_import_variable_t* fmi1_import_get_variable_by_name(fmi1_import_t* fmu, char* name)
    fmi1_import_variable_t* fmi1_import_get_variable_by_vr(fmi1_import_t* fmu, fmi1_base_type_enu_t baseType, fmi1_value_reference_t vr)
    fmi1_value_reference_t  fmi1_import_get_variable_vr(fmi1_import_variable_t *)
    char *                  fmi1_import_get_variable_description(fmi1_import_variable_t *)
    char *                  fmi1_import_get_variable_name(fmi1_import_variable_t *)
    int                     fmi1_import_get_variable_has_start(fmi1_import_variable_t *)
    int                     fmi1_import_get_variable_is_fixed(fmi1_import_variable_t *)

    #CONVERTER METHODS
    fmi1_import_integer_variable_t * fmi1_import_get_variable_as_integer(fmi1_import_variable_t *)
    fmi1_import_real_variable_t    * fmi1_import_get_variable_as_real(fmi1_import_variable_t *)
    fmi1_import_bool_variable_t    * fmi1_import_get_variable_as_boolean(fmi1_import_variable_t *)
    fmi1_import_enum_variable_t    * fmi1_import_get_variable_as_enum(fmi1_import_variable_t *)
    fmi1_import_string_variable_t  * fmi1_import_get_variable_as_string(fmi1_import_variable_t *)

    fmi2_import_integer_variable_t * fmi2_import_get_variable_as_integer(fmi2_import_variable_t *)
    fmi2_import_real_variable_t * fmi2_import_get_variable_as_real(fmi2_import_variable_t *)
    fmi2_import_bool_variable_t * fmi2_import_get_variable_as_boolean(fmi2_import_variable_t *)
    fmi2_import_enum_variable_t * fmi2_import_get_variable_as_enum(fmi2_import_variable_t *)
    fmi2_import_string_variable_t * fmi2_import_get_variable_as_string(fmi2_import_variable_t *)

    #INTEGER
    int fmi1_import_get_integer_type_min(fmi1_import_integer_typedef_t *)
    int fmi1_import_get_integer_type_max(fmi1_import_integer_typedef_t *)
    int fmi1_import_get_integer_status(fmi1_import_t *, int, fmi1_integer_t *)

    int fmi2_import_get_integer_type_min(fmi2_import_integer_typedef_t *)
    int fmi2_import_get_integer_type_max(fmi2_import_integer_typedef_t *)
    int fmi2_import_get_integer_status(fmi2_import_t *, int, fmi2_integer_t *)

    int fmi1_import_get_integer_variable_max(fmi1_import_integer_variable_t *)
    int fmi1_import_get_integer_variable_min(fmi1_import_integer_variable_t *)
    int fmi1_import_get_integer_variable_start(fmi1_import_integer_variable_t *)

    int fmi2_import_get_integer_variable_max(fmi2_import_integer_variable_t *)
    int fmi2_import_get_integer_variable_min(fmi2_import_integer_variable_t *)
    int fmi2_import_get_integer_variable_start(fmi2_import_integer_variable_t *)
    
    fmi1_import_integer_typedef_t * fmi1_import_get_type_as_int(fmi1_import_variable_typedef_t *)

    #ENUMERATIONS
    unsigned int fmi1_import_get_enum_type_min(fmi1_import_enumeration_typedef_t *)
    unsigned int fmi1_import_get_enum_type_size(fmi1_import_enumeration_typedef_t *)
    unsigned int fmi1_import_get_enum_type_max(fmi1_import_enumeration_typedef_t *)

    int fmi1_import_get_enum_variable_min(fmi1_import_enum_variable_t *)
    int fmi1_import_get_enum_variable_max(fmi1_import_enum_variable_t *)
    int fmi1_import_get_enum_variable_start(fmi1_import_enum_variable_t *)

    #REAL
    double fmi1_import_get_real_type_min(fmi1_import_real_typedef_t *)
    double fmi1_import_get_real_type_nominal(fmi1_import_real_typedef_t *)
    double fmi1_import_get_real_type_max(fmi1_import_real_typedef_t *)

    fmi1_real_t fmi1_import_get_real_variable_max(fmi1_import_real_variable_t *)
    fmi1_real_t fmi1_import_get_real_variable_start(fmi1_import_real_variable_t *)
    fmi1_real_t fmi1_import_get_real_variable_nominal(fmi1_import_real_variable_t *)
    fmi1_real_t fmi1_import_get_real_variable_min(fmi1_import_real_variable_t *)
    
    fmi1_import_unit_t * fmi1_import_get_real_variable_unit(fmi1_import_real_variable_t *)
    fmi1_import_display_unit_t * fmi1_import_get_real_variable_display_unit(fmi1_import_real_variable_t *)
    fmi1_import_real_typedef_t * fmi1_import_get_type_as_real(fmi1_import_variable_typedef_t *)

    #BOOLEAN
    fmi1_boolean_t fmi1_import_get_boolean_variable_start(fmi1_import_bool_variable_t *)
    char *         fmi1_import_get_string_variable_start(fmi1_import_string_variable_t *)



    #TYPES
    fmi1_variability_enu_t fmi1_import_get_variability(fmi1_import_variable_t *)
    fmi1_causality_enu_t   fmi1_import_get_causality(fmi1_import_variable_t *)
    fmi1_base_type_enu_t   fmi1_import_get_base_type(fmi1_import_variable_typedef_t *)


    fmi1_import_capabilities_t * fmi1_import_get_capabilities(fmi1_import_t *)


    #VARIABLE LIST METHODS
    fmi1_import_variable_list_t * fmi1_import_get_variable_list(fmi1_import_t *)
    fmi1_import_variable_list_t * fmi1_import_get_variable_aliases(fmi1_import_t *, fmi1_import_variable_t *)
    void fmi1_import_free_variable_list(fmi1_import_variable_list_t *)
    size_t fmi1_import_get_variable_list_size(fmi1_import_variable_list_t *)
    fmi1_import_variable_t * fmi1_import_get_variable(fmi1_import_variable_list_t *, unsigned int)


    fmi1_variable_alias_kind_enu_t fmi1_import_get_variable_alias_kind(fmi1_import_variable_t *)



    unsigned int fmi1_import_get_unit_definitions_number(fmi1_import_unit_definitions_t *)
    unsigned int fmi1_import_get_maxOutputDerivativeOrder(fmi1_import_capabilities_t *)
    fmi1_import_unit_definitions_t * fmi1_import_get_unit_definitions(fmi1_import_t *)
    char * fmi1_import_get_type_description(fmi1_import_variable_typedef_t *)

    int fmi1_import_get_canHandleEvents(fmi1_import_capabilities_t *)
    fmi1_import_variable_list_t * fmi1_import_create_var_list(fmi1_import_t *, fmi1_import_variable_t *)


    
    fmi1_import_display_unit_t * fmi1_import_get_unit_display_unit(fmi1_import_unit_t *, size_t)
    char * fmi1_import_get_type_quantity(fmi1_import_variable_typedef_t *)
    unsigned int fmi1_import_get_unit_display_unit_number(fmi1_import_unit_t *)

    size_t fmi1_import_get_type_definition_number(fmi1_import_type_definitions_t *)
    int fmi1_import_get_canRunAsynchronuously(fmi1_import_capabilities_t *)

    unsigned int fmi1_import_get_number_of_vendors(fmi1_import_vendor_list_t *)
    char * fmi1_import_get_enum_type_item_description(fmi1_import_enumeration_typedef_t *, unsigned int)
    int fmi1_import_get_real_type_is_relative_quantity(fmi1_import_real_typedef_t *)
    int fmi1_import_clear_last_error(fmi1_import_t *)
    fmi1_value_reference_t * fmi1_import_get_value_referece_list(fmi1_import_variable_list_t *)
    fmi1_import_annotation_t * fmi1_import_get_vendor_annotation(fmi1_import_vendor_t *, unsigned int)
    int fmi1_import_get_real_status(fmi1_import_t *, int, fmi1_real_t *)
    fmi1_real_t fmi1_import_convert_from_display_unit(fmi1_real_t, fmi1_import_display_unit_t *, int)
    unsigned int fmi1_import_get_number_of_vendor_annotations(fmi1_import_vendor_t *)
    int fmi1_import_get_canRejectSteps(fmi1_import_capabilities_t *)
    fmi1_import_vendor_t * fmi1_import_get_vendor(fmi1_import_vendor_list_t *, unsigned int)
    int fmi1_import_get_canNotUseMemoryManagementFunctions(fmi1_import_capabilities_t *)
    int fmi1_import_get_real_output_derivatives(fmi1_import_t *, fmi1_value_reference_t *, size_t, fmi1_integer_t *, fmi1_real_t *)
    int fmi1_import_get_string_status(fmi1_import_t *, int, fmi1_string_t *)

    ctypedef int(*fmi1_import_variable_filter_function_ft)(fmi1_import_variable_t *, void *)
    fmi1_import_variable_list_t * fmi1_import_filter_variables(fmi1_import_variable_list_t *, fmi1_import_variable_filter_function_ft, void *)
    fmi1_import_variable_list_t * fmi1_import_get_sublist(fmi1_import_variable_list_t *, unsigned int, unsigned int)
    int fmi1_import_set_real_input_derivatives(fmi1_import_t *, fmi1_value_reference_t *, size_t, fmi1_integer_t *, fmi1_real_t *)
    char * fmi1_import_get_unit_name(fmi1_import_unit_t *)
    char * fmi1_import_get_display_unit_name(fmi1_import_display_unit_t *)
    fmi1_import_type_definitions_t * fmi1_import_get_type_definitions(fmi1_import_t *)
    fmi1_import_variable_typedef_t * fmi1_import_get_typedef(fmi1_import_type_definitions_t *, unsigned int)
    fmi1_import_enumeration_typedef_t * fmi1_import_get_type_as_enum(fmi1_import_variable_typedef_t *)
    fmi1_import_variable_list_t * fmi1_import_prepend_to_var_list(fmi1_import_variable_list_t *, fmi1_import_variable_list_t *)
    int fmi1_import_get_canBeInstantiatedOnlyOncePerProcess(fmi1_import_capabilities_t *)
    size_t fmi1_import_get_number_of_additional_models(fmi1_import_t *)
    int fmi1_import_get_boolean_status(fmi1_import_t *, int, fmi1_boolean_t *)
    fmi1_import_variable_t * fmi1_import_get_variable_alias_base(fmi1_import_t *, fmi1_import_variable_t *)
    int fmi1_import_get_canHandleVariableCommunicationStepSize(fmi1_import_capabilities_t *)
    
    char * fmi1_import_get_enum_type_item_name(fmi1_import_enumeration_typedef_t *, unsigned int)
    fmi1_variable_naming_convension_enu_t fmi1_import_get_naming_convention(fmi1_import_t *)
    fmi1_import_unit_t * fmi1_import_get_unit(fmi1_import_unit_definitions_t *, unsigned int)
    fmi1_real_t fmi1_import_get_display_unit_gain(fmi1_import_display_unit_t *)
    fmi1_real_t fmi1_import_get_display_unit_offset(fmi1_import_display_unit_t *)
    fmi1_import_variable_typedef_t * fmi1_import_get_variable_declared_type(fmi1_import_variable_t *)
    fmi1_import_variable_list_t * fmi1_import_join_var_list(fmi1_import_variable_list_t *, fmi1_import_variable_list_t *)
    
    fmi1_import_variable_list_t * fmi1_import_clone_variable_list(fmi1_import_variable_list_t *)

    fmi1_import_display_unit_t * fmi1_import_get_type_display_unit(fmi1_import_real_typedef_t *)
    int fmi1_import_get_manual_start(fmi1_import_t *)
    fmi1_import_unit_t * fmi1_import_get_real_type_unit(fmi1_import_real_typedef_t *)

    fmi1_base_type_enu_t fmi1_import_get_variable_base_type(fmi1_import_variable_t *)
    char * fmi1_import_get_additional_model_name(fmi1_import_t *, size_t)
    fmi1_import_variable_list_t * fmi1_import_get_direct_dependency(fmi1_import_t *, fmi1_import_variable_t *)
    int fmi1_import_get_canInterpolateInputs(fmi1_import_capabilities_t *)
    char * fmi1_import_get_model_standard_version(fmi1_import_t *)
    
    fmi1_fmu_kind_enu_t fmi1_import_get_fmu_kind(fmi1_import_t *)
    fmi1_import_unit_t * fmi1_import_get_base_unit(fmi1_import_display_unit_t *)
    char * fmi1_import_get_annotation_value(fmi1_import_annotation_t *)
    char * fmi1_import_get_type_name(fmi1_import_variable_typedef_t *)
    fmi1_real_t fmi1_import_convert_to_display_unit(fmi1_real_t, fmi1_import_display_unit_t *, int)
    fmi1_import_variable_list_t * fmi1_import_append_to_var_list(fmi1_import_variable_list_t *, fmi1_import_variable_t *)
    char * fmi1_causality_to_string(fmi1_causality_enu_t)
    char * fmi1_import_get_annotation_name(fmi1_import_annotation_t *)


    #OTHER HELPER METHODS
    void jm_set_default_callbacks(jm_callbacks *)
    jm_string jm_get_last_error(jm_callbacks *)
    void jm_clear_last_error(jm_callbacks *)
    void jm_log(jm_callbacks *, char *, int, char *)
    void * mempcpy(void *, void *, size_t)
    void * memset(void *, int, size_t)
    char * strcat(char *, char *)
    size_t strlen(char *)
    jm_callbacks * jm_get_default_callbacks()
    void jm_log_v(jm_callbacks *, char *, int, char *, __va_list_tag *)





    #UNSORTED!!!
    size_t fmi2_import_get_derivative_index(fmi2_import_variable_t *)
    void fmi2_default_callback_logger(fmi2_component_t, fmi2_string_t, int, fmi2_string_t, fmi2_string_t)


    void fmi2_import_init_logger(jm_callbacks *, fmi2_callback_functions_t *)
    int fmi2_import_get_enum_type_item_value(fmi2_import_enumeration_typedef_t *, unsigned int)


    char * fmi2_import_get_variable_name(fmi2_import_variable_t *)
    fmi2_import_unit_t * fmi2_import_get_unit(fmi2_import_unit_definitions_t *, unsigned int)
    fmi2_import_variable_typedef_t * fmi2_import_get_typedef(fmi2_import_type_definitions_t *, unsigned int)
    size_t fmi2_import_get_input_index(fmi2_import_variable_t *)


    size_t fmi2_SI_base_unit_exp_to_string(int *, size_t, char *)
    double fmi2_import_get_SI_unit_factor(fmi2_import_unit_t *)


    unsigned int fmi2_import_get_unit_definitions_number(fmi2_import_unit_definitions_t *)

    fmi2_real_t fmi2_import_get_real_variable_nominal(fmi2_import_real_variable_t *)


    int fmi2_import_get_variable_has_start(fmi2_import_variable_t *)

    fmi2_import_variable_list_t * fmi2_import_join_var_list(fmi2_import_variable_list_t *, fmi2_import_variable_list_t *)

    size_t fmi2_import_get_state_index(fmi2_import_variable_t *)


    int fmi2_import_var_list_push_back(fmi2_import_variable_list_t *, fmi2_import_variable_t *)
    fmi2_import_variable_list_t * fmi2_import_append_to_var_list(fmi2_import_variable_list_t *, fmi2_import_variable_t *)


    char * fmi2_fmu_kind_to_string(fmi2_fmu_kind_enu_t)


    void fmi2_log_forwarding_v(fmi2_component_t, fmi2_string_t, int, fmi2_string_t, fmi2_string_t, __va_list_tag *)


    fmi2_base_type_enu_t fmi2_import_get_base_type(fmi2_import_variable_typedef_t *)

    int fmi2_import_set_integer(fmi2_import_t *, fmi2_value_reference_t *, size_t, fmi2_integer_t *)
    unsigned int fmi2_import_get_unit_display_unit_number(fmi2_import_unit_t *)
    int fmi2_import_get_string_status(fmi2_import_t *, int, fmi2_string_t *)


    fmi2_value_reference_t * fmi2_import_get_value_referece_list(fmi2_import_variable_list_t *)

    fmi2_import_variable_list_t * fmi2_import_get_outputs_list(fmi2_import_t *)
    int fmi2_import_set_real_input_derivatives(fmi2_import_t *, fmi2_value_reference_t *, size_t, fmi2_integer_t *, fmi2_real_t *)

    int fmi2_import_clear_last_error(fmi2_import_t *)

    char * fmi2_import_get_string_variable_start(fmi2_import_string_variable_t *)
    unsigned int fmi2_import_get_type_definition_number(fmi2_import_type_definitions_t *)

    char * fmi2_SI_base_unit_to_string(fmi2_SI_base_units_enu_t)
    double fmi2_import_get_default_experiment_tolerance(fmi2_import_t *)
    unsigned int fmi2_import_get_enum_type_max(fmi2_import_enumeration_typedef_t *)

    int fmi2_import_get_real_type_is_relative_quantity(fmi2_import_real_typedef_t *)
    unsigned int fmi2_import_get_enum_type_size(fmi2_import_enumeration_typedef_t *)




    fmi2_causality_enu_t fmi2_import_get_causality(fmi2_import_variable_t *)
    fmi2_import_unit_definitions_t * fmi2_import_get_unit_definitions(fmi2_import_t *)
    void fmi2_import_get_dependencies_derivatives_on_inputs(fmi2_import_t *, size_t * *, size_t * *, char * *)
    fmi2_import_variable_list_t * fmi2_import_prepend_to_var_list(fmi2_import_variable_list_t *, fmi2_import_variable_t *)
    fmi2_import_variable_list_t * fmi2_import_get_variable_aliases(fmi2_import_t *, fmi2_import_variable_t *)
    fmi2_import_variable_list_t * fmi2_import_get_variable_list(fmi2_import_t *, int)
    fmi2_import_variable_typedef_t * fmi2_import_get_variable_declared_type(fmi2_import_variable_t *)
    void fmi2_import_set_debug_mode(fmi2_import_t *, int)
    char * fmi2_import_get_vendor_name(fmi2_import_t *, size_t)

    fmi2_boolean_t fmi2_import_get_boolean_variable_start(fmi2_import_bool_variable_t *)
    fmi2_import_enumeration_typedef_t * fmi2_import_get_type_as_enum(fmi2_import_variable_typedef_t *)
    double fmi2_import_get_SI_unit_offset(fmi2_import_unit_t *)

    void fmi2_import_expand_variable_references(fmi2_import_t *, char *, char *, size_t)
    int fmi2_import_terminate_slave(fmi2_import_t *)
    fmi2_import_unit_t * fmi2_import_get_real_type_unit(fmi2_import_real_typedef_t *)
    int fmi2_import_instantiate_slave(fmi2_import_t *, fmi2_string_t, fmi2_string_t, fmi2_boolean_t)

    int fmi2_import_get_enum_variable_min(fmi2_import_enum_variable_t *)

    char * fmi2_import_get_display_unit_name(fmi2_import_display_unit_t *)
    fmi2_real_t fmi2_import_get_real_variable_min(fmi2_import_real_variable_t *)







    fmi2_initial_enu_t fmi2_import_get_initial(fmi2_import_variable_t *)
    size_t fmi2_import_get_variable_original_order(fmi2_import_variable_t *)

    char * fmi2_variability_to_string(fmi2_variability_enu_t)
    int fmi2_import_get_real_type_is_unbounded(fmi2_import_real_typedef_t *)
    fmi2_import_variable_t * fmi2_import_get_variable(fmi2_import_variable_list_t *, size_t)

    char * fmi2_import_get_variable_description(fmi2_import_variable_t *)
    fmi2_import_variable_list_t * fmi2_import_create_var_list(fmi2_import_t *, fmi2_import_variable_t *)
    char * fmi2_import_get_model_identifier_CS(fmi2_import_t *)
    char * fmi2_import_get_log_category(fmi2_import_t *, size_t)
    fmi2_real_t fmi2_import_get_display_unit_offset(fmi2_import_display_unit_t *)

    fmi2_import_variable_list_t * fmi2_import_get_inputs_list(fmi2_import_t *)


    fmi2_variable_naming_convension_enu_t fmi2_import_get_naming_convention(fmi2_import_t *)

    fmi2_import_variable_list_t * fmi2_import_get_sublist(fmi2_import_variable_list_t *, size_t, size_t)
    fmi2_import_variable_list_t * fmi2_import_clone_variable_list(fmi2_import_variable_list_t *)


    unsigned int fmi2_import_get_enum_type_min(fmi2_import_enumeration_typedef_t *)


    int fmi2_import_get_directional_derivative(fmi2_import_t *, fmi2_value_reference_t*, size_t, fmi2_value_reference_t*, size_t, fmi2_real_t*, fmi2_real_t*)
    char * fmi2_import_get_last_error(fmi2_import_t *)
    char * fmi2_import_get_enum_type_item_description(fmi2_import_enumeration_typedef_t *, unsigned int)
    char * fmi2_causality_to_string(fmi2_causality_enu_t)
    fmi2_real_t fmi2_import_convert_from_display_unit(fmi2_real_t, fmi2_import_display_unit_t *, int)
    char * fmi2_import_get_types_platform(fmi2_import_t *)
    double fmi2_import_convert_from_SI_base_unit(double, fmi2_import_unit_t *)
    int fmi2_import_set_fmu_state(fmi2_import_t *, fmi2_FMU_state_t)
    void fmi2_import_free_slave_instance(fmi2_import_t *)
    int * fmi2_import_get_SI_unit_exponents(fmi2_import_unit_t *)
    int fmi2_import_get_real_output_derivatives(fmi2_import_t *, fmi2_value_reference_t *, size_t, fmi2_integer_t *, fmi2_real_t *)
    char * fmi2_import_get_enum_type_value_name(fmi2_import_enumeration_typedef_t *, int)
    fmi2_value_reference_t fmi2_import_get_variable_vr(fmi2_import_variable_t *)
    fmi2_import_display_unit_t * fmi2_import_get_unit_display_unit(fmi2_import_unit_t *, size_t)
    int fmi2_import_get_fmu_state(fmi2_import_t *, fmi2_FMU_state_t *)
    fmi2_import_display_unit_t * fmi2_import_get_type_display_unit(fmi2_import_real_typedef_t *)
    fmi2_import_variable_t * fmi2_import_get_variable_by_name(fmi2_import_t *, char *)
    fmi2_initial_enu_t fmi2_get_valid_initial(fmi2_variability_enu_t, fmi2_causality_enu_t, fmi2_initial_enu_t)
    double fmi2_import_get_real_type_min(fmi2_import_real_typedef_t *)
    fmi2_import_variable_t * fmi2_import_get_variable_by_vr(fmi2_import_t *, fmi2_base_type_enu_t, fmi2_value_reference_t)
    fmi2_real_t fmi2_import_get_real_variable_max(fmi2_import_real_variable_t *)
    fmi2_import_variable_list_t * fmi2_import_get_derivatives_list(fmi2_import_t *)

    int fmi2_import_de_serialize_fmu_state(fmi2_import_t *, fmi2_byte_t *, size_t, fmi2_FMU_state_t *)

    fmi2_import_variable_list_t * fmi2_import_filter_variables(fmi2_import_variable_list_t *, fmi2_import_variable_filter_function_ft, void *)

    int fmi2_import_initialize_model(fmi2_import_t *, fmi2_boolean_t, fmi2_real_t, fmi2_event_info_t *)
    void fmi2_import_get_dependencies_outputs_on_states(fmi2_import_t *, size_t * *, size_t * *, char * *)
    void fmi2_import_free_variable_list(fmi2_import_variable_list_t *)
    char * fmi2_base_type_to_string(fmi2_base_type_enu_t)
    void fmi2_log_forwarding(fmi2_component_t, fmi2_string_t, fmi2_status_t, fmi2_string_t, fmi2_string_t,...)


    fmi2_import_t * fmi2_import_parse_xml(fmi_import_context_t *, char *, fmi2_xml_callbacks_t *)
    size_t fmi2_import_get_vendors_num(fmi2_import_t *)
    fmi2_real_t fmi2_import_get_real_variable_start(fmi2_import_real_variable_t *)




    void fmi2_import_collect_model_counts(fmi2_import_t *, fmi2_import_model_counts_t *)

    int fmi2_import_get_status(fmi2_import_t *, int, int *)

    void fmi2_import_free(fmi2_import_t *)
    size_t fmi2_import_get_log_categories_num(fmi2_import_t *)
    char * fmi2_import_get_type_quantity(fmi2_import_variable_typedef_t *)
    char * fmi2_import_get_unit_name(fmi2_import_unit_t *)
    fmi2_import_variable_t * fmi2_import_get_variable_alias_base(fmi2_import_t *, fmi2_import_variable_t *)


    char * fmi2_dependency_factor_kind_to_string(fmi2_dependency_factor_kind_enu_t)
    int fmi2_import_instantiate_model(fmi2_import_t *, fmi2_string_t, fmi2_string_t, fmi2_boolean_t)
    char * fmi2_status_to_string(int)
    char * fmi2_import_get_model_identifier_ME(fmi2_import_t *)
    char * fmi2_import_get_model_standard_version(fmi2_import_t *)
    int fmi2_import_get_boolean_status(fmi2_import_t *, int, fmi2_boolean_t *)
    void fmi2_import_destroy_dllfmu(fmi2_import_t *)
    int fmi2_import_get_integer(fmi2_import_t *, fmi2_value_reference_t *, size_t, fmi2_integer_t *)
    int fmi2_import_get_string(fmi2_import_t *, fmi2_value_reference_t *, size_t, fmi2_string_t *)
    size_t fmi2_import_get_output_index(fmi2_import_variable_t *)
    double fmi2_import_get_real_type_nominal(fmi2_import_real_typedef_t *)


    fmi2_import_real_typedef_t * fmi2_import_get_type_as_real(fmi2_import_variable_typedef_t *)
    fmi2_import_unit_t * fmi2_import_get_base_unit(fmi2_import_display_unit_t *)
    char * fmi2_import_get_copyright(fmi2_import_t *)
    fmi2_real_t fmi2_import_convert_to_display_unit(fmi2_real_t, fmi2_import_display_unit_t *, int)
    fmi2_import_unit_t * fmi2_import_get_real_variable_unit(fmi2_import_real_variable_t *)
    fmi2_initial_enu_t fmi2_get_default_initial(fmi2_variability_enu_t, fmi2_causality_enu_t)
    fmi2_import_display_unit_t * fmi2_import_get_real_variable_display_unit(fmi2_import_real_variable_t *)
    int fmi2_import_get_enum_variable_max(fmi2_import_enum_variable_t *)
    char * fmi2_import_get_type_name(fmi2_import_variable_typedef_t *)
    double fmi2_import_get_real_type_max(fmi2_import_real_typedef_t *)
    size_t fmi2_import_get_variable_list_size(fmi2_import_variable_list_t *)
    double fmi2_import_convert_to_SI_base_unit(double, fmi2_import_unit_t *)
    void fmi2_import_get_dependencies_derivatives_on_states(fmi2_import_t *, size_t * *, size_t * *, char * *)





    fmi2_variable_alias_kind_enu_t fmi2_import_get_variable_alias_kind(fmi2_import_variable_t *)
    char * fmi2_naming_convention_to_string(fmi2_variable_naming_convension_enu_t)
    int fmi2_import_get_enum_variable_start(fmi2_import_enum_variable_t *)

    fmi2_import_type_definitions_t * fmi2_import_get_type_definitions(fmi2_import_t *)
    char * fmi2_initial_to_string(fmi2_initial_enu_t)
    fmi2_base_type_enu_t fmi2_import_get_variable_base_type(fmi2_import_variable_t *)

    fmi2_variability_enu_t fmi2_import_get_variability(fmi2_import_variable_t *)


    char * fmi2_import_get_type_description(fmi2_import_variable_typedef_t *)

    char * fmi2_capability_to_string(fmi2_capabilities_enu_t)
    fmi2_import_integer_typedef_t * fmi2_import_get_type_as_int(fmi2_import_variable_typedef_t *)
    char * fmi2_import_get_enum_type_item_name(fmi2_import_enumeration_typedef_t *, unsigned int)
    char * fmi2_import_get_model_version(fmi2_import_t *)
    fmi2_import_variable_list_t * fmi2_import_get_states_list(fmi2_import_t *)

    fmi2_real_t fmi2_import_get_display_unit_factor(fmi2_import_display_unit_t *)


