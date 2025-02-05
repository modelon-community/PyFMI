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

# This file contains FMIL header content specific to FMI1

from fmil_import cimport (
    fmi_import_context_t,
)

cdef extern from 'fmilib.h':
    #FMI VARIABLE TYPE DEFINITIONS
    ctypedef double fmi1_real_t
    ctypedef unsigned int fmi1_value_reference_t
    ctypedef char   fmi1_boolean_t
    ctypedef void*  fmi1_component_t
    ctypedef char*  fmi1_string_t
    ctypedef int    fmi1_integer_t

    #STRUCTS
    ctypedef enum fmi1_status_t:
        fmi1_status_ok = 0
        fmi1_status_warning = 1
        fmi1_status_discard = 2
        fmi1_status_error = 3
        fmi1_status_fatal = 4
        fmi1_status_pending = 5

    cdef enum fmi1_variable_alias_kind_enu_t:
        fmi1_variable_is_negated_alias = -1
        fmi1_variable_is_not_alias = 0
        fmi1_variable_is_alias = 1

    cdef enum fmi1_base_type_enu_t:
        fmi1_base_type_real = 0
        fmi1_base_type_int = 1
        fmi1_base_type_bool = 2
        fmi1_base_type_str = 3
        fmi1_base_type_enum = 4

    cdef enum fmi1_causality_enu_t:
        fmi1_causality_enu_input = 0
        fmi1_causality_enu_output = 1
        fmi1_causality_enu_internal = 2
        fmi1_causality_enu_none = 3

    cdef enum fmi1_fmu_kind_enu_t:
        fmi1_fmu_kind_enu_me = 0
        fmi1_fmu_kind_enu_cs_standalone = 1
        fmi1_fmu_kind_enu_cs_tool = 2

    cdef enum fmi1_variability_enu_t:
        fmi1_variability_enu_constant = 0
        fmi1_variability_enu_parameter = 1
        fmi1_variability_enu_discrete = 2
        fmi1_variability_enu_continuous = 3

    cdef enum fmi1_variable_naming_convension_enu_t:
        fmi1_naming_enu_flat = 0
        fmi1_naming_enu_structured = 1

    ctypedef enum fmi1_status_kind_t:
        fmi1_do_step_status = 0
        fmi1_pending_status = 1
        fmi1_last_successful_time = 2

    ctypedef struct fmi1_event_info_t:
        fmi1_boolean_t iterationConverged
        fmi1_boolean_t stateValueReferencesChanged
        fmi1_boolean_t stateValuesChanged
        fmi1_boolean_t terminateSimulation
        fmi1_boolean_t upcomingTimeEvent
        fmi1_real_t nextEventTime

    ctypedef void *(*fmi1_callback_allocate_memory_ft)(size_t, size_t)
    ctypedef void(*fmi1_callback_free_memory_ft)(void *)
    ctypedef void(*fmi1_callback_logger_ft)(fmi1_component_t c, fmi1_string_t instanceName, fmi1_status_t status, fmi1_string_t category, fmi1_string_t message, ...)
    ctypedef void(*fmi1_step_finished_ft)(fmi1_component_t c, fmi1_status_t status)

    ctypedef struct fmi1_callback_functions_t:
        fmi1_callback_logger_ft logger
        fmi1_callback_allocate_memory_ft allocateMemory
        fmi1_callback_free_memory_ft freeMemory
        fmi1_step_finished_ft stepFinished

    cdef struct fmi1_xml_model_description_t:
        pass

    cdef struct fmi1_import_t:
        pass

    cdef struct fmi1_xml_vendor_t:
        pass
    ctypedef fmi1_xml_vendor_t fmi1_import_vendor_t

    cdef struct fmi1_xml_capabilities_t:
        pass
    ctypedef fmi1_xml_capabilities_t fmi1_import_capabilities_t

    cdef struct fmi1_xml_variable_t:
        pass
    ctypedef fmi1_xml_variable_t fmi1_import_variable_t

    cdef struct fmi1_xml_real_variable_t:
        pass
    ctypedef fmi1_xml_real_variable_t fmi1_import_real_variable_t

    cdef struct fmi1_xml_display_unit_t:
        pass
    ctypedef fmi1_xml_display_unit_t fmi1_import_display_unit_t

    cdef struct fmi1_xml_unit_definitions_t:
        pass
    ctypedef fmi1_xml_unit_definitions_t fmi1_import_unit_definitions_t

    cdef struct fmi1_xml_vendor_list_t:
        pass
    ctypedef fmi1_xml_vendor_list_t fmi1_import_vendor_list_t

    cdef struct fmi1_import_variable_list_t:
        pass

    cdef struct fmi1_xml_variable_typedef_t:
        pass
    ctypedef fmi1_xml_variable_typedef_t fmi1_import_variable_typedef_t

    cdef struct fmi1_xml_integer_variable_t:
        pass
    ctypedef fmi1_xml_integer_variable_t fmi1_import_integer_variable_t

    cdef struct fmi1_xml_real_typedef_t:
        pass
    ctypedef fmi1_xml_real_typedef_t fmi1_import_real_typedef_t

    cdef struct fmi1_xml_enum_variable_t:
        pass
    ctypedef fmi1_xml_enum_variable_t fmi1_import_enum_variable_t

    cdef struct fmi1_xml_type_definitions_t:
        pass
    ctypedef fmi1_xml_type_definitions_t fmi1_import_type_definitions_t

    cdef struct fmi1_xml_enumeration_typedef_t:
        pass
    ctypedef fmi1_xml_enumeration_typedef_t fmi1_import_enumeration_typedef_t

    cdef struct fmi1_xml_integer_typedef_t:
        pass
    ctypedef fmi1_xml_integer_typedef_t fmi1_import_integer_typedef_t

    cdef struct fmi1_xml_annotation_t:
        pass
    ctypedef fmi1_xml_annotation_t fmi1_import_annotation_t

    cdef struct fmi1_xml_unit_t:
        pass
    ctypedef fmi1_xml_unit_t fmi1_import_unit_t

    cdef struct fmi1_xml_bool_variable_t:
        pass
    ctypedef fmi1_xml_bool_variable_t fmi1_import_bool_variable_t

    cdef struct fmi1_xml_string_variable_t:
        pass
    ctypedef fmi1_xml_string_variable_t fmi1_import_string_variable_t

    #FMI SPECIFICATION METHODS (1.0)
    int fmi1_import_get_state_value_references(fmi1_import_t *, fmi1_value_reference_t *, size_t)
    int fmi1_import_get_nominal_continuous_states(fmi1_import_t *, fmi1_real_t *, size_t)
    int fmi1_import_get_event_indicators(fmi1_import_t *, fmi1_real_t *, size_t)
    int fmi1_import_set_time(fmi1_import_t *, fmi1_real_t)
    int fmi1_import_set_debug_logging(fmi1_import_t *, fmi1_boolean_t)
    int fmi1_import_get_continuous_states(fmi1_import_t *, fmi1_real_t *, size_t)
    int fmi1_import_instantiate_model(fmi1_import_t *, fmi1_string_t)
    int fmi1_import_set_continuous_states(fmi1_import_t *, fmi1_real_t *, size_t)
    int fmi1_import_get_status(fmi1_import_t *, int, int *)
    int fmi1_import_completed_integrator_step(fmi1_import_t *, fmi1_boolean_t *)
    int fmi1_import_reset_slave(fmi1_import_t *)
    int fmi1_import_eventUpdate(fmi1_import_t *, fmi1_boolean_t, fmi1_event_info_t *)
    int fmi1_import_initialize(fmi1_import_t *, fmi1_boolean_t, fmi1_real_t, fmi1_event_info_t *)
    int fmi1_import_terminate_slave(fmi1_import_t *)
    int fmi1_import_terminate(fmi1_import_t *)
    int fmi1_import_cancel_step(fmi1_import_t *)
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

    char * fmi1_import_get_types_platform(fmi1_import_t *)
    char * fmi1_import_get_model_types_platform(fmi1_import_t *)
    int fmi1_import_get_real_output_derivatives(fmi1_import_t *, fmi1_value_reference_t *, size_t, fmi1_integer_t *, fmi1_real_t *)
    int fmi1_import_get_string_status(fmi1_import_t *, int, fmi1_string_t *)
    int fmi1_import_set_real_input_derivatives(fmi1_import_t *, fmi1_value_reference_t *, size_t, fmi1_integer_t *, fmi1_real_t *)
    int fmi1_import_get_boolean_status(fmi1_import_t *, int, fmi1_boolean_t *)
    int fmi1_import_get_integer_status(fmi1_import_t *, int, fmi1_integer_t *)
    int fmi1_import_get_real_status(fmi1_import_t *, int, fmi1_real_t *)
    char * fmi1_import_get_version(fmi1_import_t *)
    int fmi1_import_create_dllfmu(fmi1_import_t *, fmi1_callback_functions_t, int)
    void fmi1_import_destroy_dllfmu(fmi1_import_t *)
    void fmi1_import_free_slave_instance(fmi1_import_t *)
    void fmi1_import_free_model_instance(fmi1_import_t *)

    #FMI HELPER METHODS (1.0)
    void fmi1_log_forwarding(fmi1_component_t c, fmi1_string_t instanceName, fmi1_status_t status, fmi1_string_t category, fmi1_string_t message,...)
    void fmi1_import_free(fmi1_import_t *)
    char * fmi1_import_get_mime_type(fmi1_import_t *)
    char * fmi1_import_get_last_error(fmi1_import_t *)
    void fmi_import_free_context(fmi_import_context_t *)
    char * fmi1_base_type_to_string(fmi1_base_type_enu_t)
    fmi1_import_t * fmi1_import_parse_xml(fmi_import_context_t *, char *)
    char * fmi1_variability_to_string(fmi1_variability_enu_t)
    char * fmi1_fmu_kind_to_string(fmi1_fmu_kind_enu_t)
    char * fmi1_status_to_string(int)
    unsigned int fmi1_import_get_number_of_event_indicators(fmi1_import_t *)
    unsigned int fmi1_import_get_number_of_continuous_states(fmi1_import_t *)
    int fmi1_import_get_canSignalEvents(fmi1_import_capabilities_t *)
    char * fmi1_import_get_entry_point(fmi1_import_t *)
    char * fmi1_import_get_generation_date_and_time(fmi1_import_t *)
    char * fmi1_import_get_GUID(fmi1_import_t *)

    #FMI XML METHODS

    ###Model information
    double fmi1_import_get_default_experiment_tolerance(fmi1_import_t *)
    double fmi1_import_get_default_experiment_stop(fmi1_import_t *)
    double fmi1_import_get_default_experiment_start(fmi1_import_t *)
    char * fmi1_import_get_author(fmi1_import_t *)
    char * fmi1_import_get_description(fmi1_import_t *)
    char * fmi1_import_get_generation_tool(fmi1_import_t *)
    char * fmi1_import_get_model_version(fmi1_import_t *)
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

    #INTEGER
    int fmi1_import_get_integer_type_min(fmi1_import_integer_typedef_t *)
    int fmi1_import_get_integer_type_max(fmi1_import_integer_typedef_t *)

    int fmi1_import_get_integer_variable_max(fmi1_import_integer_variable_t *)
    int fmi1_import_get_integer_variable_min(fmi1_import_integer_variable_t *)
    int fmi1_import_get_integer_variable_start(fmi1_import_integer_variable_t *)

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
    fmi1_real_t fmi1_import_convert_from_display_unit(fmi1_real_t, fmi1_import_display_unit_t *, int)
    unsigned int fmi1_import_get_number_of_vendor_annotations(fmi1_import_vendor_t *)
    int fmi1_import_get_canRejectSteps(fmi1_import_capabilities_t *)
    fmi1_import_vendor_t * fmi1_import_get_vendor(fmi1_import_vendor_list_t *, unsigned int)
    int fmi1_import_get_canNotUseMemoryManagementFunctions(fmi1_import_capabilities_t *)

    ctypedef int(*fmi1_import_variable_filter_function_ft)(fmi1_import_variable_t *, void *)
    fmi1_import_variable_list_t * fmi1_import_filter_variables(fmi1_import_variable_list_t *, fmi1_import_variable_filter_function_ft, void *)
    fmi1_import_variable_list_t * fmi1_import_get_sublist(fmi1_import_variable_list_t *, unsigned int, unsigned int)
    char * fmi1_import_get_unit_name(fmi1_import_unit_t *)
    char * fmi1_import_get_display_unit_name(fmi1_import_display_unit_t *)
    fmi1_import_type_definitions_t * fmi1_import_get_type_definitions(fmi1_import_t *)
    fmi1_import_variable_typedef_t * fmi1_import_get_typedef(fmi1_import_type_definitions_t *, unsigned int)
    fmi1_import_enumeration_typedef_t * fmi1_import_get_type_as_enum(fmi1_import_variable_typedef_t *)
    fmi1_import_variable_list_t * fmi1_import_prepend_to_var_list(fmi1_import_variable_list_t *, fmi1_import_variable_list_t *)
    int fmi1_import_get_canBeInstantiatedOnlyOncePerProcess(fmi1_import_capabilities_t *)
    size_t fmi1_import_get_number_of_additional_models(fmi1_import_t *)
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
