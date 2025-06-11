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

from libc.stdio cimport FILE

import numpy as np
cimport numpy as np

IF UNAME_SYSNAME == "Windows":
    cdef extern from "stdio.h" nogil:
        ctypedef struct FILE:
            pass
        long long _ftelli64(FILE *stream)
        int _fseeki64(FILE *stream, long long offset, int whence)
    cdef inline int os_specific_fseek(FILE *stream, long long offset, int whence):
        return _fseeki64(stream, offset, whence)
    cdef inline long long os_specific_ftell(FILE *stream):
        return _ftelli64(stream)
ELSE:
    cdef extern from "stdio.h" nogil:
        ctypedef struct FILE:
            pass
        long long ftello(FILE *stream)
        int fseeko(FILE *stream, long long offset, int whence)
    cdef inline int os_specific_fseek(FILE *stream, long long offset, int whence):
        return fseeko(stream, offset, whence)
    cdef inline long long os_specific_ftell(FILE *stream):
        return ftello(stream)

cdef class DumpDataFMI3:
    cdef np.ndarray time_tmp
    # TODO: Investigate if there is a difference in performance by declaring 'model'
    #       as an object instead of FMI3.FMUModelME3
    cdef public object model
    cdef public dict value_references, type_getters
    cdef public object _file
    cdef int _with_diagnostics
    cdef dump_data(self, np.ndarray data)
