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


import numpy as np
cimport numpy as np
np.import_array()
import logging
cimport pyfmi.fmi3 as FMI3

from assimulo.problem cimport cExplicit_Problem

cdef class FMIODE3(cExplicit_Problem):
    """ An Assimulo Explicit Model extended to FMI3 interface. """
    cdef public int _f_nbr, _g_nbr, _input_activated, _extra_f_nbr, jac_nnz, input_len_names
    cdef public object _model, problem_name, result_file_name, __input, _A
    cdef public object export, _sparse_representation, _with_jacobian, _logging, _write_header, _start_time
    cdef public dict timings
    cdef public np.ndarray y0, input_float64_mask, input_other_mask
    cdef public list input_names, input_float64_value_refs, input_other, _logg_step_event
    cdef public double t0, _synchronize_factor
    cdef public jac_use, state_events_use, time_events_use # Flags for Assimulo
    cdef public FMI3.FMUModelME3 model_me3
    cdef public int model_me3_instance
    cdef public np.ndarray _state_temp_1, _event_temp_1

    cdef int _logging_as_dynamic_diagnostics
    cdef int _number_of_diagnostics_variables
    cpdef _set_input_values(self, double t)
    cdef _update_model(self, double t, np.ndarray[double, ndim=1, mode="c"] y)
    cdef int _compare(self, double t, np.ndarray[double, ndim=1, mode="c"] y)
    cdef list _vrefs32_nostate_eval, _vrefs64_nostate_eval
