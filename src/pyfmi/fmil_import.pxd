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

# This file contains FMIL header content not specific to a given FMI version

cdef extern from "stdlib.h":
    ctypedef long unsigned int size_t

    void *malloc(size_t)
    void free(void *ptr)
    void *calloc(size_t, size_t)
    void *realloc(void *, size_t)

cdef extern from "string.h":
    int memcmp(void *s1, void *s2, size_t n);

cdef extern from "stdio.h":
    ctypedef struct FILE:
        pass
    int fprintf(FILE *restrict, const char *restrict, ...)
    FILE *fopen(const char *path, const char *mode)
    int fclose(FILE *file_pointer)

#SEE http://wiki.cython.org/FAQ#HowdoIusevariableargs.
cdef extern from "stdarg.h":
    ctypedef struct va_list:
        pass
    ctypedef struct fake_type:
        pass
    void va_start(va_list, void* arg)
    void va_end(va_list)
    int vsnprintf(char *str, size_t size, char *format, va_list ap)

cdef extern from 'fmilib.h':
    # ctypedef long unsigned int size_t # same as via stdlib.h
    ctypedef void*  jm_voidp
    ctypedef char*  jm_string

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

    ctypedef enum jm_status_enu_t:
        jm_status_error = -1
        jm_status_success = 0
        jm_status_warning = 1

    ctypedef enum fmi_version_enu_t:
        fmi_version_unknown_enu = 0
        fmi_version_1_enu = 1
        fmi_version_2_0_enu = 2
        fmi_version_unsupported_enu = 3

    ctypedef int(*jm_compare_ft)(void *, void *)
    ctypedef jm_voidp(*jm_malloc_f)(size_t)
    ctypedef jm_voidp(*jm_calloc_f)(size_t, size_t)
    ctypedef jm_voidp(*jm_realloc_f)(void *, size_t)
    ctypedef void(*jm_free_f)(jm_voidp)
    ctypedef void (*jm_logger_f)(jm_callbacks* c, jm_string module, jm_log_level_enu_t log_level, jm_string message) except *

    cdef struct jm_callbacks:
        jm_malloc_f malloc
        jm_calloc_f calloc
        jm_realloc_f realloc
        jm_free_f free
        jm_logger_f logger
        jm_log_level_enu_t log_level
        jm_voidp context
        char * errMessageBuffer

    cdef struct jm_named_ptr:
        jm_voidp ptr
        jm_string name

    cdef struct fmi_xml_context_t:
        pass

    ctypedef fmi_xml_context_t fmi_import_context_t

    cdef struct __va_list_tag:
        pass

    #FMI HELPER METHODS
    char * fmi_import_get_dll_path(char *, char *, jm_callbacks *)
    char * fmi_import_get_model_description_path(char *, jm_callbacks *)
    fmi_import_context_t * fmi_import_allocate_context(jm_callbacks *)
    void fmi_import_free_context(fmi_import_context_t *)
    fmi_version_enu_t fmi_import_get_fmi_version(fmi_import_context_t*, char*, char*)
    int fmi_import_rmdir(jm_callbacks*, char *)

    #OTHER HELPER METHODS
    void jm_set_default_callbacks(jm_callbacks *)
    jm_string jm_get_last_error(jm_callbacks *)
    # void jm_clear_last_error(jm_callbacks *)
    void jm_log(jm_callbacks *, char *, int, char *)
    void * mempcpy(void *, void *, size_t)
    void * memcpy(void *, void *, size_t)
    void * memset(void *, int, size_t)
    char * strcat(char *, char *)
    char * strcpy(char *, char *)
    size_t strlen(char *)
    jm_callbacks * jm_get_default_callbacks()
    void jm_log_v(jm_callbacks *, char *, int, char *, __va_list_tag *)
