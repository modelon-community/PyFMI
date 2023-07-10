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


import numpy as N
cimport numpy as N

from pyfmi.fmi cimport FMUModelME2

try:
    import assimulo
    assimulo_present = True
except Exception:
    logging.warning(
        'Could not load Assimulo module. Check pyfmi.check_packages()')
    assimulo_present = False

if assimulo_present:
    from assimulo.problem import Implicit_Problem
    from assimulo.problem import Explicit_Problem
    from assimulo.problem cimport cExplicit_Problem
    from assimulo.exception import *
else:
    class Implicit_Problem:
        pass
    class Explicit_Problem:
        pass


cdef class FMIODE2(cExplicit_Problem):
    """
    An Assimulo Explicit Model extended to FMI interface.
    """
    cdef public int _f_nbr, _g_nbr, _input_activated, _extra_f_nbr, jac_nnz, input_len_names
    cdef public object _model, problem_name, result_file_name, __input, _A, debug_file_name, debug_file_object
    cdef public object export, _sparse_representation, _with_jacobian, _logging, _write_header, _start_time
    cdef public dict timings
    cdef public N.ndarray y0, input_real_mask, input_other_mask
    cdef public list input_names, input_real_value_refs, input_other, _logg_step_event
    cdef public double t0, _synchronize_factor
    cdef public jac_use, state_events_use, time_events_use
    cdef public FMUModelME2 model_me2
    cdef public int model_me2_instance
    cdef public N.ndarray _state_temp_1, _event_temp_1

    cdef int _logging_to_mat
    cpdef _set_input_values(self, double t)
    cdef _update_model(self, double t, N.ndarray[double, ndim=1, mode="c"] y)
    cdef int _compare(self, double t, N.ndarray[double, ndim=1, mode="c"] y)
