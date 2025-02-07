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


# XXX: Copy more header content from FMI3 as it becomes necessary

cdef extern from 'fmilib.h':
    # FMI VARIABLE TYPE DEFINITIONS

    # STATUS
    cdef enum fmi3_fmu_kind_enu_t:
        fmi3_fmu_kind_unknown = 1
        fmi3_fmu_kind_me = 2
        fmi3_fmu_kind_cs = 4
        fmi3_fmu_kind_se = 8

    # LOGGING
    ctypedef int(*fmi3_xml_element_start_handle_ft)(void*, char*, void*, char*, char**)
    ctypedef int(*fmi3_xml_element_data_handle_ft)(void*, char*, int)
    ctypedef int(*fmi3_xml_element_end_handle_ft)(void*, char*)
    cdef struct fmi3_xml_callbacks_t:
        fmi3_xml_element_start_handle_ft startHandle
        fmi3_xml_element_data_handle_ft  dataHandle
        fmi3_xml_element_end_handle_ft   endHandle
        void* context

    cdef struct fmi3_import_t:
        pass
    # FMI SPECIFICATION METHODS (3.0)
    # BASIC
    # modes

    # misc

    # setting

    # getting

    # save states

    # FMI HELPER METHODS (3.0)
    fmi3_fmu_kind_enu_t fmi3_import_get_fmu_kind(fmi3_import_t*)
    char* fmi3_fmu_kind_to_string(fmi3_fmu_kind_enu_t)

    # FMI XML METHODS
    # Parsing/logging basics
    fmi3_import_t* fmi3_import_parse_xml(FMIL.fmi_import_context_t*, char*, fmi3_xml_callbacks_t*)
    void fmi3_import_free(fmi3_import_t*)

    ### Model information

    # CONVERTER METHODS

    # INTEGER

    # OTHER HELPER METHODS

    # Does NOT invoke CAPI calls
