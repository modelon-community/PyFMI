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

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np

cimport pyfmi.fmi3 as FMI3
from pyfmi.fmi3 import FMI3_Type, FMI3_Causality, FMI3_Initial, FMI3_Variability
from pyfmi.exceptions import FMUException
from pyfmi.fmi_util cimport (
    _read_trajectory32,
    _read_trajectory64,
)

cdef extern from "stdio.h":
    FILE *fdopen(int, const char *)
    FILE *fopen(const char *, const char *)
    size_t fread(void*, size_t, size_t, FILE *)
    int fseek(FILE *, long, int)
    int fclose(FILE *)

cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef prepare_data_info_fmi3(np.ndarray[int, ndim=2] data_info, list sorted_vars, list diagnostics_param_values, int nof_diag_vars, model):
    cdef int index_fixed    = 1
    cdef int index_variable = 1
    cdef int nof_sorted_vars = len(sorted_vars)
    cdef int nof_diag_params = len(diagnostics_param_values)
    cdef int i
    cdef int last_data_matrix = -1, last_index = -1
    cdef dict params = {
        FMI3_Type.FLOAT64: [],
        FMI3_Type.FLOAT32: [],
        FMI3_Type.INT64: [],
        FMI3_Type.INT32: [],
        FMI3_Type.INT16: [],
        FMI3_Type.INT8: [],
        FMI3_Type.UINT64: [],
        FMI3_Type.UINT32: [],
        FMI3_Type.UINT16: [],
        FMI3_Type.UINT8: [],
        FMI3_Type.BOOL: [],
        FMI3_Type.ENUM: [],
    }
    cdef dict variables = {
        FMI3_Type.FLOAT64: [],
        FMI3_Type.FLOAT32: [],
        FMI3_Type.INT64: [],
        FMI3_Type.INT32: [],
        FMI3_Type.INT16: [],
        FMI3_Type.INT8: [],
        FMI3_Type.UINT64: [],
        FMI3_Type.UINT32: [],
        FMI3_Type.UINT16: [],
        FMI3_Type.UINT8: [],
        FMI3_Type.BOOL: [],
        FMI3_Type.ENUM: [],
    }

    last_vref = -1

    for i in range(1, nof_sorted_vars + 1):
        var = sorted_vars[i - 1]
        data_info[2, i] = 0
        data_info[3, i] = -1

        if last_vref == var.value_reference: # alias
            data_info[0, i] = last_data_matrix
            data_info[1, i] = last_index
        else:
            last_vref = var.value_reference
            is_fixed_or_const = var.variability in (FMI3_Variability.FIXED, FMI3_Variability.CONSTANT)

            if is_fixed_or_const:
                last_data_matrix = 1
                index_fixed = index_fixed + 1
                last_index = index_fixed
            else:
                last_data_matrix = 2
                index_variable = index_variable + 1
                last_index = index_variable

            try:
                if is_fixed_or_const:
                    params[var.type].append(last_vref)
                else:
                    variables[var.type].append(last_vref)

            except KeyError:
                raise FMUException(
                    f"Unknown type {var.type} detected for variable {var.name} when writing the results."
                )

            data_info[1, i] = last_index
            data_info[0, i] = last_data_matrix

    data_info[0, 0] = 0
    data_info[1, 0] = 1
    data_info[2, 0] = 0
    data_info[3, 0] = -1

    for i in range(nof_sorted_vars + 1, nof_sorted_vars + 1 + nof_diag_params):
        data_info[0, i] = 1
        data_info[2, i] = 0
        data_info[3, i] = -1
        index_fixed = index_fixed + 1
        data_info[1, i] = index_fixed

    last_index = 0
    for i in range(nof_sorted_vars + 1 + nof_diag_params, nof_sorted_vars + 1 + nof_diag_params + nof_diag_vars):
        data_info[0, i] = 3
        data_info[2, i] = 0
        data_info[3, i] = -1
        last_index = last_index + 1
        data_info[1, i] = last_index

    data = np.append(
        model.time,
        np.concatenate(
            (model.get_float64(params[FMI3_Type.FLOAT64]),
             model.get_float32(params[FMI3_Type.FLOAT32]),
             model.get_int64(params[FMI3_Type.INT64]).astype(float),
             model.get_int32(params[FMI3_Type.INT32]).astype(float),
             model.get_int16(params[FMI3_Type.INT16]).astype(float),
             model.get_int8(params[FMI3_Type.INT8]).astype(float),
             model.get_uint64(params[FMI3_Type.UINT64]).astype(float),
             model.get_uint32(params[FMI3_Type.UINT32]).astype(float),
             model.get_uint16(params[FMI3_Type.UINT16]).astype(float),
             model.get_uint8(params[FMI3_Type.UINT8]).astype(float),
             model.get_boolean(params[FMI3_Type.BOOL]).astype(float),
             model.get_int64(params[FMI3_Type.ENUM]).astype(float),
             np.array(diagnostics_param_values).astype(float)
            ),
            axis = 0
        )
    )

    return data, variables

cdef class DumpDataFMI3:
    def __init__(self, model, filep, value_references, with_diagnostics):
        self.value_references = {
            k: np.array(v, dtype=np.uint32, ndmin=1).ravel() for k,v in value_references.items()
        }

        self.time_tmp = np.zeros(1)

        self._file = filep
        self.model = model

        self._with_diagnostics = with_diagnostics

        # For quick access when writing data
        self.type_getters = {
            FMI3_Type.FLOAT64: self.model.get_float64,
            FMI3_Type.FLOAT32: self.model.get_float32,
            FMI3_Type.INT64:   self.model.get_int64,
            FMI3_Type.INT32:   self.model.get_int32,
            FMI3_Type.INT16:   self.model.get_int16,
            FMI3_Type.INT8:    self.model.get_int8,
            FMI3_Type.UINT64:  self.model.get_uint64,
            FMI3_Type.UINT32:  self.model.get_uint32,
            FMI3_Type.UINT16:  self.model.get_uint16,
            FMI3_Type.UINT8:   self.model.get_uint8,
            FMI3_Type.BOOL:    self.model.get_boolean,
            FMI3_Type.ENUM:    self.model.get_int64
        }

    cdef dump_data(self, np.ndarray data):
        self._file.write(data.tobytes(order="F"))

    def save_point(self):
        """ Saves a point of simulation data to the result. """
        if self._with_diagnostics:
            self.dump_data(np.array(float(1.0)))

        self.time_tmp[0] = self.model.time
        self.dump_data(self.time_tmp)

        for data_type, value_references in self.value_references.items():
            if np.size(value_references) > 0:
                if data_type == FMI3_Type.FLOAT64:
                    self.dump_data(
                        self.type_getters[data_type](value_references)
                    )
                else:
                    self.dump_data(
                        self.type_getters[data_type](value_references).astype(float)
                    )

    def save_diagnostics_point(self, diag_data):
        """ Saves a point of diagnostics data to the result. """
        self.dump_data(np.array(float(2.0)))
        self.time_tmp[0] = self.model.time
        self.dump_data(self.time_tmp)
        self.dump_data(diag_data)


@cython.boundscheck(False)
@cython.wraparound(False)
def read_trajectory(file_name, long long data_index, long long file_position, long long sizeof_type, long long nbr_points, long long nbr_variables):
    """
    Reads a trajectory from a binary file.

    Parameters::

        file_name --
            File to read from.

        data_index --
            Which position has the variable for which the trajectory is
            to be read.

        file_position --
            Where in the file does the matrix of a trajectories start.

        sizeof_type --
            Size of the data type that the result is stored in

        nbr_points --
            Number of points in the result

        nbr_variables --
            Number of variables in the result

    Returns::

        A numpy array with the trajectory
    """
    cdef long long start_point = data_index * sizeof_type
    cdef long long end_point   = sizeof_type * (nbr_points * nbr_variables)
    cdef long long interval    = sizeof_type * nbr_variables
    if sizeof_type == 4:
        return _read_trajectory32(file_name, start_point, end_point, interval, file_position, nbr_points)
    elif sizeof_type == 8:
        return _read_trajectory64(file_name, start_point, end_point, interval, file_position, nbr_points)
    else:
        raise FMUException("Failed to read the result. The result is on an unsupported format. Can only read data that is either a 32 or 64 bit double.")
