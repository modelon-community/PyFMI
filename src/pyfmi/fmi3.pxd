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

# Module containing the FMI3 interface Python wrappers.

cimport pyfmi.fmil_import as FMIL
cimport pyfmi.fmil3_import as FMIL3
cimport pyfmi.fmi_base as FMI_BASE

cdef class FMUModelBase3(FMI_BASE.ModelBase):
    pass

cdef class FMUModelME3(FMUModelBase3):
    pass

cdef object _load_fmi3_fmu(
    fmu, 
    object log_file_name, 
    str kind, 
    int log_level, 
    int allow_unzipped_fmu,
    FMIL.fmi_import_context_t* context, 
    bytes fmu_temp_dir,
    FMIL.jm_callbacks callbacks,
    list log_data
)
