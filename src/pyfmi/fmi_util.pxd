#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019-2022 Modelon AB
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
from pyfmi.fmi cimport FMUModelME2

cpdef decode(x)
cpdef encode(x)


"""
    Below we define a 'modification' to fseek that is OS specific in order to handle very large files.
    This is because fseek/ftell is not sufficient as soon as the number of bytes in a result file
    exceed the maximum value for long int.
"""
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

cdef class DumpData:
    cdef np.ndarray real_var_ref, int_var_ref, bool_var_ref
    cdef np.ndarray real_var_tmp, int_var_tmp, bool_var_tmp
    cdef np.ndarray time_tmp
    cdef public FMUModelME2 model_me2
    cdef public int model_me2_instance
    cdef public object _file, model
    cdef size_t real_size, int_size, bool_size
    cdef int _with_diagnostics
    cdef dump_data(self, np.ndarray data)
