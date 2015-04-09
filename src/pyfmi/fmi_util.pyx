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
"""
Module containing the FMI interface Python wrappers.
"""
from collections import OrderedDict

cpdef cpr_seed(dependencies, list column_keys):
    cdef int i=0,j=0,k=0
    cdef int n_col = len(column_keys)#len(dependencies.keys())
    cdef dict columns_taken# = {key: 1 for key in dependencies.keys() if len(dependencies[key]) == 0}
    cdef dict groups = {}
    cdef dict column_dict = {}
    cdef dict column_keys_dict = {}

    row_keys_dict    = {s:i for i,s in enumerate(dependencies.keys())}
    column_keys_dict = {s:i for i,s in enumerate(column_keys)}
    column_dict      = {i:[] for i,s in enumerate(column_keys)}
    for i,dx in enumerate(dependencies.keys()):
        for x in dependencies[dx]:
            column_dict[column_keys_dict[x]].append(dx)
    columns_taken = {key: 1 for key in column_dict.keys() if len(column_dict[key]) == 0}

    for i in range(n_col):
        if columns_taken.has_key(i):
            continue
            
        # New group
        groups[k] = ([i], column_dict[i][:], [row_keys_dict[x] for x in column_dict[i]], [i]*len(column_dict[i]))
        
        for j in range(i+1, n_col):
            if columns_taken.has_key(j):
                continue
            
            intersect = frozenset(groups[k][1]).intersection(column_dict[j])
            if not intersect:
                groups[k][0].append(j)
                groups[k][1].extend(column_dict[j])
                groups[k][2].extend([row_keys_dict[x] for x in column_dict[j]])
                groups[k][3].extend([j]*len(column_dict[j]))
                columns_taken[j] = 1
            
        k = k + 1
        
    return groups
