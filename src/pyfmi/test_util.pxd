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
cimport pyfmi.fmil1_import as FMIL1
cimport pyfmi.fmil2_import as FMIL2
cimport pyfmi.fmi1 as FMI1
cimport pyfmi.fmi2 as FMI2

cdef class _ForTestingFMUModelME1(FMI1.FMUModelME1):
    cdef int _get_nominal_continuous_states_fmil(self, FMIL1.fmi1_real_t* xnominal, size_t nx)
    cpdef set_allocated_fmu(self, int value)

cdef class _ForTestingFMUModelME2(FMI2.FMUModelME2):
    cdef int _get_real_by_ptr(self, FMIL2.fmi2_value_reference_t* vrefs, size_t _size, FMIL2.fmi2_real_t* values)
    cdef int _set_real(self, FMIL2.fmi2_value_reference_t* vrefs, FMIL2.fmi2_real_t* values, size_t _size)
    cdef int _get_real_by_list(self, FMIL2.fmi2_value_reference_t[:] valueref, size_t _size, FMIL2.fmi2_real_t[:] values)
    cdef int _get_integer(self, FMIL2.fmi2_value_reference_t[:] valueref, size_t _size, FMIL2.fmi2_integer_t[:] values)
    cdef int _get_boolean(self, FMIL2.fmi2_value_reference_t[:] valueref, size_t _size, FMIL2.fmi2_real_t[:] values)
    cdef int _get_nominal_continuous_states_fmil(self, FMIL2.fmi2_real_t* xnominal, size_t nx)
    cpdef set_initialized_fmu(self, int value)
