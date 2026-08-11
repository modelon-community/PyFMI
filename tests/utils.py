#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2026 Modelon AB
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import functools
from pathlib import Path

from pyfmi.fmi3 import FMUModelME3

this_dir = Path(__file__).parent
FMI3_REF_FMU_PATH = Path(this_dir) / 'files' / 'reference_fmus' / '3.0'

# possibly move to some util function and use more widely for all PyFMI testing
@functools.cache
def _fmu_cached(fmu_path, model_class = FMUModelME3, allow_unzipped_fmu = False, _connect_dll = True, **kwargs):
    return model_class(
        fmu = fmu_path,
        allow_unzipped_fmu = allow_unzipped_fmu,
        _connect_dll = _connect_dll,
        **kwargs
    )

def _get_fmu(fmu_path, model_class = FMUModelME3, allow_unzipped_fmu = False, _connect_dll = True, **kwargs):
    fmu = _fmu_cached(
        fmu_path = fmu_path,
        model_class = model_class,
        allow_unzipped_fmu = allow_unzipped_fmu,
        _connect_dll = _connect_dll,
        **kwargs
    )
    if _connect_dll:
        fmu.free_instance()
        fmu.instantiate()
        fmu.reset()
    return fmu

def get_fmi3_reference_fmu(name, model_class = FMUModelME3, allow_unzipped_fmu = False, _connect_dll = True, **kwargs):
    fmu_path = FMI3_REF_FMU_PATH / (name + ".fmu")
    return _get_fmu(
        fmu_path = fmu_path,
        model_class = model_class,
        allow_unzipped_fmu = allow_unzipped_fmu,
        _connect_dll = _connect_dll,
        **kwargs
    )
