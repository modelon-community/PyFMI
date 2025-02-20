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



cdef extern from 'fmilib.h':
    # FMI VARIABLE TYPE DEFINITIONS
    ctypedef void*  fmi3_instance_environment_t
    ctypedef char*  fmi3_string_t
    ctypedef bool   fmi3_boolean_t
    ctypedef double fmi3_float64_t

    # STRUCTS
    ctypedef enum fmi3_boolean_enu_t:
        fmi3_true = 1
        fmi3_false = 0

    # STATUS
    cdef enum fmi3_fmu_kind_enu_t:
        fmi3_fmu_kind_unknown = 1 << 0
        fmi3_fmu_kind_me = 1 << 1
        fmi3_fmu_kind_cs = 1 << 2
        fmi3_fmu_kind_se = 1 << 3

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
    fmi3_status_t fmi3_import_set_time(fmi3_import_t *, fmi3_float64_t)

    # getting
    double fmi3_import_get_default_experiment_start(fmi3_import_t*);
    double fmi3_import_get_default_experiment_stop(fmi3_import_t*);
    double fmi3_import_get_default_experiment_tolerance(fmi3_import_t*);
    # save states

    # FMI HELPER METHODS (3.0)
    fmi3_fmu_kind_enu_t fmi3_import_get_fmu_kind(fmi3_import_t*)
    char* fmi3_fmu_kind_to_string(fmi3_fmu_kind_enu_t)
    char* fmi3_import_get_model_name(fmi3_import_t*)

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
    void fmi3_log_forwarding(fmi3_instance_environment_t, fmi3_status_t, fmi3_string_t, fmi3_string_t)
    char* fmi3_import_get_last_error(fmi3_import_t*)
    char* fmi3_import_get_model_identifier_ME(fmi3_import_t*)