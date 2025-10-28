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

# Module containing the abstract base class for FMI interface Python wrappers.
# Plus some auxiliary functions

cimport pyfmi.fmil_import as FMIL

cdef FMIL.fmi_version_enu_t import_and_get_version(FMIL.fmi_import_context_t*, char*, char*, int)

cdef class ModelBase:
    """
    Abstract Model class containing base functionality.
    """
    cdef list _log
    cdef char* _fmu_log_name
    cdef FMIL.jm_callbacks callbacks
    cdef public dict cache
    cdef public object _log_stream
    cdef public object file_object
    cdef public object _additional_logger
    cdef public object _max_log_size_msg_sent
    cdef public object _result_file
    cdef public object _log_handler
    cdef object _modelId
    cdef public int _log_is_stream, _invoked_dealloc
    cdef public unsigned long long int _current_log_size, _max_log_size
    cdef char* _fmu_temp_dir

    cdef str _increment_log_size_and_check_max_size(self, str msg)
    cdef _logger(self, FMIL.jm_string module, int log_level, FMIL.jm_string message) with gil

cdef class LogHandler:
    cdef unsigned long _max_log_size

    cpdef void set_max_log_size(self, unsigned long val)
    cpdef void capi_start_callback(self, int limit_reached, unsigned long current_log_size)
    cpdef void capi_end_callback  (self, int limit_reached, unsigned long current_log_size)

cdef class LogHandlerDefault(LogHandler):
    cdef unsigned long _log_checkpoint

    cdef void _update_checkpoint (self, int limit_reached, unsigned long current_log_size)
    cpdef unsigned long get_log_checkpoint(self)
    cpdef void capi_start_callback(self, int limit_reached, unsigned long current_log_size)
    cpdef void capi_end_callback  (self, int limit_reached, unsigned long current_log_size)
