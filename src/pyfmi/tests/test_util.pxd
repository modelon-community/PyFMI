#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2014-2024 Modelon AB
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

""" Collection of classes used for testing purposes."""
cimport pyfmi.fmil_import as FMIL
from pyfmi.fmi cimport FMUModelME1, FMUModelME2

cdef class _ForTestingFMUModelME1(FMUModelME1):
    cdef int _get_nominal_continuous_states_fmil(self, FMIL.fmi1_real_t* xnominal, size_t nx)
    cpdef set_allocated_fmu(self, int value)

cdef class _ForTestingFMUModelME2(FMUModelME2):
    cdef int _get_real_by_ptr(self, FMIL.fmi2_value_reference_t* vrefs, size_t _size, FMIL.fmi2_real_t* values)
    cdef int _set_real(self, FMIL.fmi2_value_reference_t* vrefs, FMIL.fmi2_real_t* values, size_t _size)
    cdef int _get_real_by_list(self, FMIL.fmi2_value_reference_t[:] valueref, size_t _size, FMIL.fmi2_real_t[:] values)
    cdef int _get_integer(self, FMIL.fmi2_value_reference_t[:] valueref, size_t _size, FMIL.fmi2_integer_t[:] values)
    cdef int _get_boolean(self, FMIL.fmi2_value_reference_t[:] valueref, size_t _size, FMIL.fmi2_real_t[:] values)
    cdef int _get_nominal_continuous_states_fmil(self, FMIL.fmi2_real_t* xnominal, size_t nx)
    cpdef set_initialized_fmu(self, int value)