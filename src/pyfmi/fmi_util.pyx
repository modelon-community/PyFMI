#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2014-2023 Modelon AB
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

"""
Module containing the FMI interface Python wrappers.
"""
import collections
import itertools

import numpy as np
cimport numpy as np

cimport pyfmi.fmil_import as FMIL
cimport pyfmi.fmi2 as FMI2 # TODO
cimport pyfmi.fmi3 as FMI3 # TODO
from pyfmi.fmi3 import FMUModelBase3, FMI3_Type, FMI3_Causality, FMI3_Initial, FMI3_Variability
from pyfmi.fmi1 import ( # TODO
    FMI_NEGATED_ALIAS, FMI_PARAMETER, FMI_CONSTANT,
    FMI_REAL, FMI_INTEGER, FMI_ENUMERATION, FMI_BOOLEAN
)

cimport pyfmi.util as pyfmi_util
from pyfmi.util import (
    encode,
    decode
)

from pyfmi.exceptions import FMUException, IOException

cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double quad_err(np.ndarray[double, ndim=1] sim, np.ndarray[double, ndim=1] est, int n):
#def quad_err(sim, est, int n):
    cdef double s = 0
    for i in range(n):
        s += (sim[i]-est[i])**2

    return s

cpdef parameter_estimation_f(y, parameters, measurments, model, input, options):
    cdef double err = 0
    cdef int n

    model.reset()

    for i,parameter in enumerate(parameters):
        model.set(parameter, y[i]*options["scaling"][i])

    # Simulate model response with new parameter values
    res = model.simulate(measurments[1][0,0], final_time=measurments[1][-1,0], input=input, options=options["simulate_options"])

    n = measurments[1].shape[0]
    for i,parameter in enumerate(measurments[0]):
        err += quad_err(res[parameter], measurments[1][:,i+1], n)

    return 1.0/n*err**(0.5)

cpdef list convert_array_names_list_names(np.ndarray names):
    cdef int max_length = names.shape[0]
    cdef int nbr_items  = len(names[0])
    cdef int i, j = 0
    cdef char *tmp = <char*>FMIL.calloc(max_length,sizeof(char))
    cdef list output = []
    cdef bytes py_str

    for i in range(nbr_items):
        for j in range(max_length):
            try:
                tmp[j] = ord(names[j,i])
            except ValueError:
                break

        py_str = tmp[:j]
        output.append(py_str)

    FMIL.free(tmp)

    return output

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list convert_array_names_list_names_int(np.ndarray[int, ndim=2] names):
    cdef int max_length = names.shape[0]
    cdef int nbr_items  = names.shape[1]
    cdef int i, j = 0, ch
    cdef char *tmp = <char*>FMIL.calloc(max_length,sizeof(char))
    cdef list output = []
    cdef bytes py_str

    for i in range(nbr_items):
        for j in range(max_length):
            ch = names[j,i]
            if ch==0:
                break
            else:
                tmp[j] = ch

        py_str = tmp[:j]
        if j == max_length - 1:
            py_str = py_str.replace(b" ", b"")
        output.append(py_str)

    FMIL.free(tmp)

    return output

def _is_real_or_float(variable, model):
    if isinstance(model, FMUModelBase3):
        return variable.type == FMI3_Type.FLOAT64
    else:
        return variable.type == FMI_REAL

def _is_int_or_enum(variable, model):
    if isinstance(model, FMUModelBase3):
        return (variable.type == FMI3_Type.INT64) or (variable.type == FMI3_Type.ENUM)
    else:
        return (variable.type == FMI_INTEGER) or (variable.type == FMI_ENUMERATION)

def _is_bool(variable, model):
    if isinstance(model, FMUModelBase3):
        return variable.type == FMI3_Type.BOOL
    else:
        return variable.type == FMI_BOOLEAN


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef prepare_data_info(np.ndarray[int, ndim=2] data_info, list sorted_vars, list diagnostics_param_values, int nof_diag_vars, model):
    cdef int index_fixed    = 1
    cdef int index_variable = 1
    cdef int nof_sorted_vars = len(sorted_vars)
    cdef int nof_diag_params = len(diagnostics_param_values)
    cdef int i, alias, data_type, variability
    cdef int last_data_matrix = -1, last_index = -1
    cdef int _FMI_NEGATED_ALIAS = FMI_NEGATED_ALIAS # TODO
    cdef int _FMI_PARAMETER = FMI_PARAMETER, _FMI_CONSTANT = FMI_CONSTANT # TODO
    cdef int _FMI_REAL = FMI_REAL, _FMI_INTEGER = FMI_INTEGER # TODO
    cdef int _FMI_ENUMERATION = FMI_ENUMERATION, _FMI_BOOLEAN = FMI_BOOLEAN # TODO
    cdef list param_real = [], param_int = [], param_bool = []
    cdef list param_float64 = [], param_float32 = [], param_int64 = [], param_int32 = []
    # Note that for FMI3 we can put all the value references in the same lists
    # regardless of bitness since they are written to the simulation results as floats.
    cdef list varia_real = [], varia_int = [], varia_bool = []
    last_vref = -1

    is_fmi3 = isinstance(model, FMUModelBase3)

    for i in range(1, nof_sorted_vars + 1):
        var = sorted_vars[i-1]
        data_info[2,i] = 0
        data_info[3,i] = -1

        if var.alias == _FMI_NEGATED_ALIAS:
            alias = -1
        else:
            alias = 1

        if last_vref == var.value_reference:
            data_info[0,i] = last_data_matrix
            data_info[1,i] = alias*last_index
        else:
            # TODO cleanup and improve
            if is_fmi3:
                variability = var.variability.value
                data_type = var.type.value
            else:
                variability = var.variability
                data_type   = var.type
            last_vref   = var.value_reference

            if is_fmi3:
                cond = var.variability is FMI3_Variability.FIXED or var.variability is FMI3_Variability.CONSTANT
            else:
                cond = variability == _FMI_PARAMETER or variability == _FMI_CONSTANT

            if cond:
                last_data_matrix = 1
                index_fixed = index_fixed + 1
                last_index = index_fixed

                # Make make sure we add the parameters to the correct list of value references
                if _is_real_or_float(var, model):
                    if is_fmi3:
                        if var.type == FMI3_Type.FLOAT64:
                            param_float64.append(last_vref)
                        elif var.type == FMI3_Type.FLOAT32:
                            param_float32.append(last_vref)
                        else:
                            raise NotImplementedError
                    else:
                        param_real.append(last_vref)
                elif _is_int_or_enum(var, model):
                    if is_fmi3:
                        if var.type == FMI3_Type.INT64:
                            param_int64.append(last_vref)
                        elif var.type == FMI3_Type.INT32:
                            param_int32.append(last_vref)
                        else:
                            raise NotImplementedError
                    else:
                        param_int.append(last_vref)
                elif _is_bool(var, model):
                    param_bool.append(last_vref)
                else:
                    raise FMUException("Unknown type detected for variable %s when writing the results."%var.name)
            else:
                last_data_matrix = 2
                index_variable = index_variable + 1
                last_index = index_variable

                # No need to consider bitness of the variable here
                if _is_real_or_float(var, model):
                    varia_real.append(last_vref)
                elif _is_int_or_enum(var, model):
                    varia_int.append(last_vref)
                elif _is_bool(var, model):
                    varia_bool.append(last_vref)
                else:
                    raise FMUException("Unknown type detected for variable %s when writing the results."%var.name)

            data_info[1, i] = alias*last_index
            data_info[0, i] = last_data_matrix

    data_info[0, 0] = 0
    data_info[1, 0] = 1
    data_info[2, 0] = 0
    data_info[3, 0] = -1

    for i in range(nof_sorted_vars+1, nof_sorted_vars + 1 + nof_diag_params):
        data_info[0,i] = 1
        data_info[2,i] = 0
        data_info[3,i] = -1
        index_fixed = index_fixed + 1
        data_info[1,i] = index_fixed

    last_index = 0
    for i in range(nof_sorted_vars + 1 + nof_diag_params, nof_sorted_vars + 1 + nof_diag_params + nof_diag_vars):
        data_info[0,i] = 3
        data_info[2,i] = 0
        data_info[3,i] = -1
        last_index = last_index + 1
        data_info[1,i] = last_index

    if is_fmi3:
        data = np.append(
            model.time,
            np.concatenate(
                (model.get_float64(param_float64),
                 model.get_float64(param_float32),
                 model.get_int64(param_int64).astype(float),
                 model.get_int32(param_int32).astype(float),
                 model.get_boolean(param_bool).astype(float),
                 np.array(diagnostics_param_values).astype(float)
                ),
                axis = 0
            )
        )
    else:
        data = np.append(
            model.time,
            np.concatenate(
                (model.get_real(param_real),
                 model.get_integer(param_int).astype(float),
                 model.get_boolean(param_bool).astype(float),
                 np.array(diagnostics_param_values).astype(float)
                ),
                axis = 0
            )
        )

    return data, varia_real, varia_int, varia_bool

cpdef convert_str_list(list data):
    cdef int length = 0
    cdef int items = len(data)
    cdef int i,j, tmp_length, k
    cdef char *output
    cdef char *tmp
    cdef bytes py_string

    for i in range(items):
        data[i] = pyfmi_util.encode(data[i])
        j = len(data[i])
        if j+1 > length:
            length = j+1

    output = <char*>FMIL.calloc(items*length,sizeof(char))

    for i in range(items):
        tmp = data[i]
        tmp_length = len(tmp)
        k = i*length

        FMIL.memcpy(&output[k], tmp, tmp_length)
        #FMIL.memset(&output[k+tmp_length], ' ', length-tmp_length) #Adding padding, seems to be necessary :(

    py_string = output[:items*length]

    FMIL.free(output)

    return length, py_string

cpdef convert_sorted_vars_name_desc(list sorted_vars, list diag_params, list diag_vars):
    cdef int items = len(sorted_vars)
    cdef int nof_diag_params = len(diag_params)
    cdef int nof_diag_vars = len(diag_vars)
    cdef int i, name_length_trial, desc_length_trial, kd, kn
    cdef list desc = [pyfmi_util.encode("Time in [s]")]
    cdef list name = [pyfmi_util.encode("time")]
    cdef int name_length = len(name[0])+1
    cdef int desc_length = len(desc[0])+1
    cdef char *desc_output
    cdef char *name_output
    cdef char *ctmp_name
    cdef char *ctmp_desc
    cdef int tot_nof_vars = items+nof_diag_params+nof_diag_vars

    for tmp_name, tmp_desc in itertools.chain([(var.name, var.description) for var in sorted_vars],
                                             diag_params, diag_vars):
        tmp_name = pyfmi_util.encode(tmp_name)
        tmp_desc = pyfmi_util.encode(tmp_desc)

        name.append(tmp_name)
        desc.append(tmp_desc)

        name_length_trial = len(tmp_name)
        desc_length_trial = len(tmp_desc)

        if name_length_trial+1 > name_length:
             name_length = name_length_trial + 1
        if desc_length_trial+1 > desc_length:
             desc_length = desc_length_trial + 1


    name_output = <char*>FMIL.calloc((tot_nof_vars+1)*name_length,sizeof(char))
    if name_output == NULL:
        raise FMUException("Failed to allocate memory for storing the names of the variables. " \
                               "Please reduce the number of stored variables by using filters.")
    desc_output = <char*>FMIL.calloc((tot_nof_vars+1)*desc_length,sizeof(char))
    if desc_output == NULL:
        raise FMUException("Failed to allocate memory for storing the description of the variables. " \
                               "Please reduce the number of stored variables or disable storing of the description.")

    for i in range(tot_nof_vars+1):
        ctmp_name = name[i]
        ctmp_desc = desc[i]

        name_length_trial = len(ctmp_name)
        desc_length_trial = len(ctmp_desc)
        kn = i*name_length
        kd = i*desc_length

        FMIL.memcpy(&name_output[kn], ctmp_name, name_length_trial)
        FMIL.memcpy(&desc_output[kd], ctmp_desc, desc_length_trial)

    py_desc_string = desc_output[:(tot_nof_vars+1)*desc_length]
    py_name_string = name_output[:(tot_nof_vars+1)*name_length]

    FMIL.free(name_output)
    FMIL.free(desc_output)

    return name_length, py_name_string, desc_length, py_desc_string

cpdef convert_sorted_vars_name(list sorted_vars, list diag_param_names, list diag_vars):
    cdef int items = len(sorted_vars)
    cdef int nof_diag_params = len(diag_param_names)
    cdef int nof_diag_vars = len(diag_vars)
    cdef int i, name_length_trial, kn
    cdef list name = [pyfmi_util.encode("time")]
    cdef int name_length = len(name[0])+1
    cdef char *name_output
    cdef char *ctmp_name
    cdef int tot_nof_vars = items+nof_diag_params+nof_diag_vars

    for tmp_name in itertools.chain( [var.name for var in sorted_vars], diag_param_names, diag_vars):
        tmp_name = pyfmi_util.encode(tmp_name)
        name.append(tmp_name)

        name_length_trial = len(tmp_name)

        if name_length_trial+1 > name_length:
             name_length = name_length_trial + 1

    name_output = <char*>FMIL.calloc((tot_nof_vars+1)*name_length,sizeof(char))
    if name_output == NULL:
        raise FMUException("Failed to allocate memory for storing the names of the variables. " \
                               "Please reduce the number of stored variables by using filters.")

    for i in range(tot_nof_vars+1):
        ctmp_name = name[i]

        name_length_trial = len(ctmp_name)
        kn = i*name_length

        FMIL.memcpy(&name_output[kn], ctmp_name, name_length_trial)

    py_name_string = name_output[:(tot_nof_vars+1)*name_length]

    FMIL.free(name_output)

    return name_length, py_name_string


cpdef convert_scalarvariable_name_to_str(list data):
    cdef int length = 0
    cdef int items = len(data)
    cdef int i,j, tmp_length, k
    cdef char *output
    cdef char *tmp
    cdef bytes py_string

    for i in range(items):
        j = len(data[i].name)
        if j+1 > length:
            length = j+1

    output = <char*>FMIL.calloc(items*length,sizeof(char))

    for i in range(items):
        py_byte_string = data[i].name
        tmp = py_byte_string
        tmp_length = len(tmp)
        k = i*length

        FMIL.memcpy(&output[k], tmp, tmp_length)
        #FMIL.memset(&output[k+tmp_length], ' ', length-tmp_length) #Adding padding, seems to be necessary :(

    py_string = output[:items*length]

    FMIL.free(output)

    return length, py_string

"""
class Graph:

    def __init__(self, edges):
        self.edges   = edges
        self.nodes   = set(node for node in itertools.chain(*edges))
        self.lowlink = dict([node, -1] for node in self.nodes)
        self.number  = dict([node, -1] for node in self.nodes)
        self.index = 0
        self.stack = []
        self.connected_components = []

    def _strongly_connected_components(self, node):
        self.lowlink[node] = self.index
        self.number[node]  = self.index

        self.index = self.index + 1
        self.stack.append(node)

        for v,w in (edge for edge in self.edges if edge[0] == node):
            if self.number[w] < 0: #Not numbered
                self._strongly_connected_components(w)
                self.lowlink[node] = min(self.lowlink[node], self.lowlink[w])
            elif self.number[w] < self.number[v]:
                if w in self.stack:
                    self.lowlink[node] = min(self.lowlink[node], self.number[w])

        if self.lowlink[node] == self.number[node]:
            #node is the root of a component
            #Start new strong component
            self.connected_components.append([])
            while self.stack and self.number[self.stack[-1]] >= self.number[node]:
                self.connected_components[-1].append(self.stack.pop())

    def strongly_connected_components(self):
        for node in self.nodes:
            if self.number[node] < 0:
                self._strongly_connected_components(node)
        return self.connected_components
"""
class OrderedSet(collections.abc.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

GRAPH_INPUT  = 0
GRAPH_OUTPUT = 1
GRAPH_SCC    = 2

class Graph:

    def __init__(self, edges, graph_info):
        self.edges   = edges
        self.nodes   = OrderedSet(node for node in itertools.chain(*edges))
        self.lowlink = dict([node, -1] for node in self.nodes)
        self.number  = dict([node, -1] for node in self.nodes)
        self.index = 0
        self.stack = []
        self.connected_components = []
        self.graph_info = graph_info
        self._unknown_index = 31415926

        self.edges_0 = {}
        self.edges_1 = {}

        for edge in self.edges:
            try:
                self.edges_0[edge[0]].append(edge[1])
            except KeyError:
                self.edges_0[edge[0]] = [edge[1]]
            try:
                self.edges_1[edge[1]].append(edge[0])
            except KeyError:
                self.edges_1[edge[1]] = [edge[0]]

    def _dfs(self, start_node):
        self.visited_nodes[start_node] = None

        for v, w in (edge for edge in self.edges if edge[0] == start_node):
            if not (w in self.visited_nodes):
                self._dfs(w)

    def dfs(self, start_node):
        self.visited_nodes = {}
        self._dfs(start_node)

        return self.visited_nodes

    def join_output_trees(self, connected_components):
        #Reverse order
        connected_components = connected_components[::-1]

        trees = {}
        joined_nodes = {}

        for node in connected_components:
            if len(node) > 1:
                continue
            node = node[0]
            if self.graph_info[node]["type"] == GRAPH_OUTPUT:
                model = self.graph_info[node]["model"]
                if not (model in trees):
                    trees[model] = collections.OrderedDict()
                    trees[model][node] = self.dfs(node)
                    joined_nodes[node] = [node]
                else:
                    included = False
                    #spanning_tree = self.dfs(node)
                    for out in trees[model].keys():
                        if node in trees[model][out]: #Node is in a previouos spanning tree (cannot join them)
                            pass
                        else:
                            print("Joining: ", out, node)
                            joined_nodes[out].append(node)
                            trees[model][out].update(self.dfs(node)) #Can be needed if they are not in the same tree
                            included = True
                            break
                    if not included:
                        joined_nodes[node] = [node]
                        trees[model][node] = self.dfs(node)# spanning_tree
        return trees, joined_nodes

    def simple_loop(self, start_node): #Must be output
        visited_nodes = {}
        loop = False

        stack = [w for v,w in (edge for edge in self.edges if edge[0] == start_node and self.graph_info[edge[1]]["model"] != self.graph_info[edge[0]]["model"])]
        while stack and not loop:
            e = stack.pop()
            if not (e in visited_nodes):
                visited_nodes[v] = None
                for v,w in (edge for edge in self.edges if edge[0] == e):
                    if w == start_node:
                        loop = True
                        break
                    stack.append(w)

        return loop

    def _strongly_connected_components(self, node):
        self.lowlink[node] = self.index
        self.number[node]  = self.index

        self.index = self.index + 1
        self.stack.append(node)

        if node in self.edges_0_edge:
            for v,w in self.edges_0_edge[node]:
                if self.number[w] < 0: #Not numbered
                    self._strongly_connected_components(w)
                    self.lowlink[node] = min(self.lowlink[node], self.lowlink[w])
                elif self.number[w] < self.number[v]:
                    if w in self.stack:
                        self.lowlink[node] = min(self.lowlink[node], self.number[w])

        if self.lowlink[node] == self.number[node]:
            #node is the root of a component
            #Start new strong component
            self.connected_components.append([])
            while self.stack and self.number[self.stack[-1]] >= self.number[node]:
                self.connected_components[-1].append(self.stack.pop())

    def strongly_connected_components(self):
        self.lowlink = dict([node, -1] for node in self.nodes)
        self.number  = dict([node, -1] for node in self.nodes)
        self.index = 0
        self.stack = []
        self.connected_components = []

        self.edges_0_edge = {}
        for edge in self.edges:
            try:
                self.edges_0_edge[edge[0]].append(edge)
            except KeyError:
                self.edges_0_edge[edge[0]] = [edge]

        for node in self.nodes:
            if self.number[node] < 0:
                self._strongly_connected_components(node)
        return self.connected_components

    def group_node(self, list connected_component):
        nodes = self.nodes
        edges = self.edges
        connected_component_dict = {k: v for v, k in enumerate(connected_component)}

        output = True
        for node in connected_component:
            if self.graph_info[node]["type"] != GRAPH_OUTPUT:
                output = False
                break
        model = True
        for node in connected_component[1:]:
            if self.graph_info[node]["model"] != self.graph_info[connected_component[0]]["model"]:
                model = False
                self._unknown_index = self._unknown_index + 1
                break
        new_node = "+".join(connected_component)
        nodes.add(new_node)
        self.graph_info[new_node] = {"type": GRAPH_OUTPUT if output else GRAPH_SCC, "model": self.graph_info[connected_component[0]]["model"] if model else self._unknown_index}
        for j,edge in enumerate(edges):
            if edge[0] in connected_component_dict and edge[1] in connected_component_dict:
                edges[j] = (None, None) #Necessary to remove the current edges
            elif edge[0] in connected_component_dict:
                edges[j] = (new_node, edge[1])
            elif edge[1] in connected_component_dict:
                edges[j] = (edge[0], new_node)
        for node in connected_component:
            nodes.discard(node)
            self.graph_info.pop(node)

        #Get unique list
        self.edges = list(OrderedSet([x for x in edges if x != (None,None)]))

        return new_node

    def tear_node(self, node):
        #Remove edges that belong to node (is output) and that are connected to outputs in the same model
        for j, edge in enumerate(self.edges):
            if (edge[0] == node or edge[1] == node) and self.graph_info[edge[0]]["model"] == self.graph_info[edge[1]]["model"]:
                self.edges[j] = (None, None)

        #Get unique list
        self.edges = list(OrderedSet([x for x in self.edges if x != (None,None)]))

    def group_connected_components(self, connected_components):
        #Update edges and nodes
        for i,conn in enumerate(connected_components):
            if len(conn) > 1:
                self.group_node(conn)
                connected_components[i] = ["+".join(conn)]

    def add_edges_between_outputs(self):
        for node in self.nodes:
            if self.graph_info[node]["type"] == GRAPH_OUTPUT:
                model = self.graph_info[node]["model"]
                for companion_output in self.graph_info[model]["outputs"]:
                    try:
                        self.graph_info[companion_output] #Node is still available? i.e. not in a SCC
                        if node != companion_output: #No edge to itself
                            self.edges.append((node, companion_output))
                    except Exception:
                        pass
        self.edges = list(OrderedSet(self.edges))

    def tear_graph(self, connected_components):
        torn_graph = False
        #Update edges and nodes
        for i,conn in enumerate(connected_components):
            if len(conn) > 1: #More than one node
                same_model = True
                model = self.graph_info[conn[0]]["model"]
                for node in conn:
                    if model != self.graph_info[node]["model"]:
                        same_model = False
                        break
                if same_model: #Component complete, only outputs from the same model
                    self.group_node(conn)
                else:
                    torn_graph = True
                    #Needs tearing
                    choices = {}
                    for node in conn:
                        if self.graph_info[node]["type"] == GRAPH_OUTPUT: #Possible choice
                            try:
                                choices[self.graph_info[node]["model"]].append(node) #Weight (external)
                            except KeyError:
                                choices[self.graph_info[node]["model"]] = [node] #Weight (external)
                    valid_choices = {}
                    for model in choices.keys():
                        if len(choices[model]) > 1: #There are nodes that are possible choices for tearing here
                            for node in choices[model]:
                                if self.simple_loop(node): #Produces a loop? At least one node does
                                    valid_choices[node] = 0 #Zero weight
                    for edge in self.edges:
                        try:
                            if self.graph_info[edge[1]]["model"] != self.graph_info[edge[0]]["model"]:
                                valid_choices[edge[0]] += 1
                        except KeyError:
                            pass
                    torn_node = valid_choices.keys()[0]
                    for node in valid_choices.keys():
                        if valid_choices[node] > valid_choices[torn_node]: #If the weight is greater
                            torn_node = node

                    print("Variable to tear: ", torn_node)
                    self.tear_node(torn_node)

        return torn_graph

    def _check_feed_through(self, nodes):
        feed_through = False

        for node in nodes:
            if node in self.edges_0:
                feed_through = True
                break
        return feed_through

    def prepare_graph(self):
        connected_components_first = {}
        connected_components_second = {}
        for node in self.nodes:
            potential = True
            potential_second = False
            if self.graph_info[node]["type"] == GRAPH_OUTPUT:
                model = self.graph_info[node]["model"]
                list_of_connections = []

                if node in self.edges_1: #The node is in a direct feed-through
                    potential = False

                if potential:
                    if model in connected_components_first:
                        connected_components_first[model].append(node)
                    else:
                        connected_components_first[model] = [node]
                else:
                    list_of_connections = self.edges_0[node] #The node is connected somewhere

                    if len(list_of_connections) > 0:
                        potential_second = not self._check_feed_through(list_of_connections)

                    if potential_second:
                        if model in connected_components_second:
                            connected_components_second[model].append(node)
                        else:
                            connected_components_second[model] = [node]
        for model in connected_components_first.keys():
            if len(connected_components_first[model]) > 1:
                self.group_node(connected_components_first[model])
        for model in connected_components_second.keys():
            if len(connected_components_second[model]) > 1:
                self.group_node(connected_components_second[model])

    def split_components(self, connected_components):
        blocks = []
        for scc in connected_components:
            if isinstance(scc, list): #The scc is a list of components
                blocks.append([])
                for node in scc:
                    blocks[-1].extend(node.split("+"))
            else:
                blocks.append([scc.split("+")])

        return blocks

    def compute_evaluation_order_old(self):
        SCCs = self.strongly_connected_components()
        self.group_connected_components(SCCs) #Group the SCCs
        self.add_edges_between_outputs() #Add edges between outputs

        while True:
            SCCs = self.strongly_connected_components()
            torn = self.tear_graph(SCCs)
            if not torn:
                break

        return self.split_components(SCCs)

    def compute_evaluation_order(self):
        self.prepare_graph()
        SCCs = self.strongly_connected_components()[::-1]

        i = 0
        while i < len(SCCs):
            f = SCCs[i]
            b = 0
            if len(f) == 1 and self.graph_info[f[0]]["type"] == GRAPH_OUTPUT:
                for j in range(i-1):
                    e = SCCs[j]
                    if len(e) == 1 and self.graph_info[e[0]]["type"] == GRAPH_OUTPUT and \
                        self.graph_info[e[0]]["model"] == self.graph_info[f[0]]["model"] and \
                        f[0] not in self.dfs(e[0]).keys():
                            SCCs[j] = [self.group_node(f+e)]
                            SCCs.pop(i)
                            b = 1
                            break
            if b == 0:
                i = i +1
            if len(f) > 1:
                self.group_node(f)

        return self.split_components(self.strongly_connected_components())

    def grouped_order(self, connected_components):
        #Update edges and nodes
        self.group_connected_components(connected_components)

        roots = []
        for node in self.nodes:
            for edge in self.edges:
                if edge[1] == node: #Not root node
                    break
            else:
                roots.append(node)

        graph = {node:[] for node in self.nodes}
        for edge in self.edges:
            graph[edge[0]].append(edge[1])

        def set_levels(queue, level):
            new_queue = []
            for ite in queue:
                levels[ite] = level
                new_queue.extend(graph[ite])
            return new_queue

        queue = []
        level = 0
        levels = {}
        for root in roots:
            levels[root] = 0
            queue.extend(graph[root])
        while len(queue) > 0:
            level = level + 1
            queue = set_levels(queue, level)

        grouped_connected_components = [[] for i in range(level+1)]
        for node in levels:
            grouped_connected_components[levels[node]].extend(node.split("+"))
        grouped_connected_components.reverse()

        return grouped_connected_components

    def dump_graph_dot(self, filename, custom_syntax=False):
        """
            digraph {
            node [texmode = "math"];
            u12 -> y12 -> u23 -> y13 -> u21 -> y11 -> u12 -> y22;
            u11 -> y21 -> u22 -> y12;
            y22 -> u13 -> y23;
            u11 [label="u_1^{[1]}", pos="-0.9,4!"];
            u12;// [label="u_1^{[2]}"];
            u22 [label="u_2^{[2]}"];
            u21 [label="u_2^{[1]}"];
            u13;// [label="u_1^{[3]}"];
            u23 [label="u_2^{[3]}"];
            y11 [label="y_1^{[1]}"];
            y21 [label="y_2^{[1]}"];
            y12 [label="y_1^{[2]}"];
            y22;// [label="y_2^{[2]}"];
            y13 [label="y_1^{[3]}"];
            y23;// [label="y_2^{[3]}"];
        }
        """
        with open(filename, 'w') as f:
            f.write('digraph { \n node [texmode = "math"]; \n splines="curved"; \n')
            for edge in self.edges:
                f.write(' "%s" -> "%s" \n'%(edge[0], edge[1]))
            for node in self.nodes:
                if custom_syntax:
                    label=""
                    for n in node.split("+"):
                        spl = n.split("|")
                        label = label+"%s_%s^{[%s]}, "%(spl[0],spl[1],spl[2])
                    label = label[:-2]
                    f.write(' "%s" [color=none, label="%s"] \n'%(node, label))
                else:
                    f.write(' "%s" [color=none, label="%s"] \n'%(node, node))
            f.write('}')

cdef class DumpData:
    def __init__(self, model, filep, real_var_ref, int_var_ref, bool_var_ref, with_diagnostics):
        if type(model) == FMI2.FMUModelME2:
            self.real_var_ref = np.array(real_var_ref, dtype=np.uint32, ndmin=1).ravel()
            self.int_var_ref  = np.array(int_var_ref,  dtype=np.uint32, ndmin=1).ravel()
            self.bool_var_ref = np.array(bool_var_ref, dtype=np.uint32, ndmin=1).ravel()
        else:
            self.real_var_ref = np.array(real_var_ref, ndmin=1).ravel()
            self.int_var_ref  = np.array(int_var_ref, ndmin=1).ravel()
            self.bool_var_ref = np.array(bool_var_ref, ndmin=1).ravel()

        self.real_size = np.size(self.real_var_ref)
        self.int_size  = np.size(self.int_var_ref)
        self.bool_size = np.size(bool_var_ref)

        self.real_var_tmp = np.zeros(self.real_size)
        self.int_var_tmp  = np.zeros(self.int_size, dtype=np.int32)
        self.bool_var_tmp = np.zeros(self.bool_size)
        self.time_tmp     = np.zeros(1)

        self._file = filep

        if type(model) == FMI2.FMUModelME2: #isinstance(model, FMUModelME2):
            self.model_me2 = model
            self.model_me2_instance = 1
        elif type(model) == FMI3.FMUModelME3:
            self.model_me3 = model
            self.model_me3_instance = 1
        else:
            self.model_me2_instance = 0
            self.model_me3_instance = 0
            self.model = model

        self._with_diagnostics = with_diagnostics

    cdef dump_data(self, np.ndarray data):
        self._file.write(data.tobytes(order="F"))

    def save_point(self):
        if self._with_diagnostics:
            self.dump_data(np.array(float(1.0)))
        if self.model_me2_instance:
            self.time_tmp[0] = self.model_me2.time
            self.dump_data(self.time_tmp)

            if self.real_size > 0:
                self.model_me2._get_real_by_list(self.real_var_ref, self.real_size, self.real_var_tmp)
                self.dump_data(self.real_var_tmp)

            if self.int_size > 0:
                self.model_me2._get_integer(self.int_var_ref, self.int_size, self.int_var_tmp)
                self.dump_data(self.int_var_tmp.astype(float))

            if self.bool_size > 0:
                self.model_me2._get_boolean(self.bool_var_ref, self.bool_size, self.bool_var_tmp)
                self.dump_data(self.bool_var_tmp)
        elif self.model_me3_instance:

            self.time_tmp[0] = self.model_me3.time
            self.dump_data(self.time_tmp)

            if self.real_size > 0:
                # TODO: Do we want a get_float64_by_list also for performance reasons?
                r = self.model_me3.get_float64(self.real_var_ref)
                self.dump_data(r)

            if self.int_size > 0:
                i = self.model_me3.get_int64(self.int_var_ref).astype(float)
                self.dump_data(i)

            if self.bool_size > 0:
                b = self.model_me3.get_boolean(self.bool_var_ref).astype(float)
                self.dump_data(b)
        else:
            self.dump_data(np.array(float(self.model.time)))

            if self.real_size > 0:
                r = self.model.get_real(self.real_var_ref)
                self.dump_data(r)

            if self.int_size > 0:
                i = self.model.get_integer(self.int_var_ref).astype(float)
                self.dump_data(i)

            if self.bool_size > 0:
                b = self.model.get_boolean(self.bool_var_ref).astype(float)
                self.dump_data(b)


    def save_diagnostics_point(self, diag_data):
        """ Saves a point of diagnostics data to the result. """
        self.dump_data(np.array(float(2.0)))
        if self.model_me2_instance:
            self.time_tmp[0] = self.model_me2.time
            self.dump_data(self.time_tmp)
        elif self.model_me3_instance:
            self.time_tmp[0] = self.model_me3.time
            self.dump_data(self.time_tmp)
        else:
            self.dump_data(np.array(float(self.model.time)))
        self.dump_data(diag_data)

cdef extern from "stdio.h":
    FILE *fdopen(int, const char *)
    FILE *fopen(const char *, const char *)
    size_t fread(void*, size_t, size_t, FILE *)
    int fseek(FILE *, long, int)
    int fclose(FILE *)

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

DTYPE32 = np.float32
ctypedef np.float32_t DTYPE32_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _read_trajectory32(
        file_name,
        long long start_point,
        long long end_point,
        long long interval,
        long long file_position,
        long long nbr_points
    ):
    cdef long long i = 0
    cdef long long offset = 0
    cdef FILE* cfile
    cdef np.ndarray[DTYPE32_t, ndim=1] data
    cdef DTYPE32_t* data_ptr
    cdef size_t sizeof_dtype = sizeof(DTYPE32_t)

    cfile = fopen(file_name, 'rb')

    data = np.empty(nbr_points, dtype=DTYPE32)
    data_ptr = <DTYPE32_t*>data.data

    os_specific_fseek(cfile, file_position, 0)
    #for offset in range(start_point, end_point, interval):
    for offset from start_point <= offset < end_point by interval:
        os_specific_fseek(cfile, file_position + offset, 0)
        fread(<void*>(data_ptr + i), sizeof_dtype, 1, cfile)
        i = i + 1

    fclose(cfile)

    return data

DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _read_trajectory64(
        file_name,
        long long start_point,
        long long end_point,
        long long interval,
        long long file_position,
        long long nbr_points
    ):

    cdef long long i = 0
    cdef long long offset = 0
    cdef FILE* cfile
    cdef np.ndarray[DTYPE_t, ndim=1] data
    cdef DTYPE_t* data_ptr
    cdef size_t sizeof_dtype = sizeof(DTYPE_t)

    data = np.empty(nbr_points, dtype=DTYPE)
    data_ptr = <DTYPE_t*>data.data

    cfile = fopen(file_name, 'rb')
    os_specific_fseek(cfile, file_position, 0)
    #for offset in range(start_point, end_point, interval):
    for offset from start_point <= offset < end_point by interval:
        os_specific_fseek(cfile, file_position + offset, 0)
        fread(<void*>(data_ptr + i), sizeof_dtype, 1, cfile)
        i = i + 1

    fclose(cfile)

    return data

@cython.boundscheck(False)
@cython.wraparound(False)
def read_diagnostics_trajectory(
        file_name,
        int read_diag_data,
        int has_position_data,
        np.ndarray[long long, ndim=1] file_pos_model_var,
        np.ndarray[long long, ndim=1] file_pos_diag_var,
        long long data_index,
        long long file_position,
        long long int sizeof_type,
        long long nbr_model_points,
        long long nbr_diag_points,
        long long nbr_model_variables,
        long long nbr_diag_variables
    ):
    """ Reads a diagnostic trajectory from the result file. """
    cdef long long file_pos
    cdef long long iter_point        = 0
    cdef long long model_var_counter = 0
    cdef long long diag_var_counter  = 0
    cdef long long i                 = 0
    cdef long long end_point   = sizeof_type * (nbr_diag_points * (nbr_diag_variables + 1) + \
                                                          nbr_model_points * (nbr_model_variables + 1))
    cdef long long model_var_interval = sizeof_type * nbr_model_variables
    cdef long long diag_var_interval  = sizeof_type * nbr_diag_variables
    cdef FILE* cfile
    cdef np.ndarray[DTYPE_t, ndim=1] data
    cdef DTYPE_t* data_ptr
    cdef size_t sizeof_dtype = sizeof(DTYPE_t)
    cdef np.ndarray[DTYPE_t, ndim=1] flag
    cdef DTYPE_t* flag_ptr

    cfile = fopen(file_name, 'rb')

    if read_diag_data == 1:
        data = np.empty(nbr_diag_points, dtype=DTYPE)
    else:
        data = np.empty(nbr_model_points, dtype=DTYPE)
    data_ptr = <DTYPE_t*>data.data
    flag = np.empty(1, dtype=DTYPE)
    flag_ptr = <DTYPE_t*>flag.data

    if has_position_data == 1:
        if read_diag_data == 1:
            file_pos_list = file_pos_diag_var
        else:
            file_pos_list = file_pos_model_var
        for file_pos in file_pos_list:
            os_specific_fseek(cfile, file_pos+data_index*sizeof_type, 0)
            fread(<void*>(data_ptr + i), sizeof_dtype, 1, cfile)
            i += 1
    else:
        while iter_point < end_point:
            os_specific_fseek(cfile, file_position+iter_point,0)
            fread(<void*>(flag_ptr), sizeof_dtype, 1, cfile)
            iter_point += sizeof_type;
            file_pos = os_specific_ftell(cfile)
            if flag[0] == 1.0:
                file_pos_model_var[model_var_counter] = file_pos
                model_var_counter +=1
                if not read_diag_data:
                    os_specific_fseek(cfile, file_position+iter_point+data_index*sizeof_type, 0)
                    fread(<void*>(data_ptr + i), sizeof_dtype, 1, cfile)
                    i += 1
                iter_point += model_var_interval
            elif flag[0] == 2.0:
                file_pos_diag_var[diag_var_counter] = file_pos
                diag_var_counter +=1
                if read_diag_data:
                    os_specific_fseek(cfile, file_position+iter_point+data_index*sizeof_type, 0)
                    fread(<void*>(data_ptr + i), sizeof_dtype, 1, cfile)
                    i += 1
                iter_point += diag_var_interval
            else:
                fclose(cfile)
                raise IOException("Result file is corrupt, cannot read results.")
    fclose(cfile)
    return data, file_pos_model_var, file_pos_diag_var


@cython.boundscheck(False)
@cython.wraparound(False)
def read_name_list(file_name, int file_position, int nbr_variables, int max_length):
    """
    Reads a list of names from a binary file.

    Parameters::

        file_name --
            File to read from.

        file_position --
            Where in the file does the list of names start

        nbr_variables --
            Number of variables to read.

        max_length --
            Maximum length of a variable

    Returns::

        A dict with the names as key and an index as value
    """
    cdef int i = 0, j = 0, need_replace = 0
    cdef FILE* cfile
    cdef char *tmp = <char*>FMIL.calloc(max_length,sizeof(char))
    cdef bytes py_str
    cdef dict data = {}

    if tmp == NULL:
        raise IOException("Couldn't allocate memory to read name list.")

    cfile = fopen(file_name, 'rb')
    fseek(cfile, file_position, 0)
    for i in range(nbr_variables):
        fread(<void*>(tmp), max_length, 1, cfile)
        py_str = tmp

        if i == 0:
            if len(py_str) == max_length:
                need_replace = 1

        if need_replace:
            py_str = py_str.replace(b" ", b"")
        data[py_str] = i

    fclose(cfile)
    FMIL.free(tmp)

    return data
