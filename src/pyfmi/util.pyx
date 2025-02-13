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

# Module containing general utility functions independent of FMI
# This is split from fmi_util.pyx to avoid circular dependencies

import numpy as np
cimport numpy as np

import functools
import marshal

cpdef decode(x):
    if isinstance(x, bytes):
        return x.decode(errors="replace")
    else:
        return x

cpdef encode(x):
    if isinstance(x, str):
        return x.encode()
    else:
        return x

def enable_caching(obj):
    @functools.wraps(obj, ('__name__', '__doc__'))
    def memoizer(*args, **kwargs):
        cache = args[0].cache #First argument is the self object
        key = (obj, marshal.dumps(args[1:]), marshal.dumps(kwargs))

        if len(cache) > 1000: #Remove items from cache in case it grows large
            cache.popitem()

        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]

    return memoizer

cpdef cpr_seed(dependencies, list column_keys, dict interested_columns = None):
    cdef int i=0,j=0,k=0
    cdef int n_col = len(column_keys)#len(dependencies.keys())
    cdef dict columns_taken# = {key: 1 for key in dependencies.keys() if len(dependencies[key]) == 0}
    cdef dict groups = {}
    cdef dict column_dict = {}
    cdef dict column_keys_dict = {}
    cdef dict data_index = {}

    row_keys_dict    = {s:i for i,s in enumerate(dependencies.keys())}
    column_keys_dict = {s:i for i,s in enumerate(column_keys)}
    column_dict      = {i:[] for i,s in enumerate(column_keys)}
    for i,dx in enumerate(dependencies.keys()):
        for x in dependencies[dx]:
            column_dict[column_keys_dict[x]].append(dx)
    columns_taken = {key: 1 for key in column_dict.keys() if len(column_dict[key]) == 0}

    k = 0
    kd = 0
    data_index = {}
    data_index_with_diag = {}
    for i in range(n_col):
        data_index[i] = list(range(k, k + len(column_dict[i])))
        k = k + len(column_dict[i])

        data_index_with_diag[i] = []
        diag_added = False
        for x in column_dict[i]:
            ind = row_keys_dict[x]
            if ind < i:
                data_index_with_diag[i].append(kd)
                kd = kd + 1
            else:
                if ind == i:
                    diag_added = True
                if not diag_added:
                    kd = kd + 1
                    diag_added = True
                data_index_with_diag[i].append(kd)
                kd = kd + 1
        if not diag_added:
            kd = kd + 1

    nnz = k
    nnz_with_diag = kd

    k = 0
    for i in range(n_col):
        if (i in columns_taken) or (interested_columns is not None and not (i in interested_columns)):
            continue

        # New group
        groups[k] = [[i], column_dict[i][:], [row_keys_dict[x] for x in column_dict[i]], [i]*len(column_dict[i]), data_index[i], data_index_with_diag[i]]

        for j in range(i+1, n_col):
            if (j in columns_taken) or (interested_columns is not None and not (j in interested_columns)):
                continue

            intersect = frozenset(groups[k][1]).intersection(column_dict[j])
            if not intersect:

                #structure
                # - [0] - variable indexes
                # - [1] - variable names
                # - [2] - matrix rows
                # - [3] - matrix columns
                # - [4] - position in data vector (CSC format)
                # - [5] - position in data vector (with diag) (CSC format)

                groups[k][0].append(j)
                groups[k][1].extend(column_dict[j])
                groups[k][2].extend([row_keys_dict[x] for x in column_dict[j]])
                groups[k][3].extend([j]*len(column_dict[j]))
                groups[k][4].extend(data_index[j])
                groups[k][5].extend(data_index_with_diag[j])
                columns_taken[j] = 1

        groups[k][0] = np.array(groups[k][0],dtype=np.int32)
        groups[k][2] = np.array(groups[k][2],dtype=np.int32)
        groups[k][3] = np.array(groups[k][3],dtype=np.int32)
        groups[k][4] = np.array(groups[k][4],dtype=np.int32)
        groups[k][5] = np.array(groups[k][5],dtype=np.int32)
        k = k + 1

    groups["groups"] = list(groups.keys())
    groups["nnz"] = nnz
    groups["nnz_with_diag"] = nnz_with_diag

    return groups
