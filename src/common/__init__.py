#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2010 Modelon AB
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
The JModelica.org <http:/www.jmodelica.org/> Python package for common classes
and functions.
"""

__all__ = ['algorithm_drivers', 'core', 'diagnostics', 'io', 'xmlparser', 'plotting']

import sys
python3_flag = True if sys.hexversion > 0x03000000 else False
## for backwards compatibility, to not break 'from pyfmi.common import diagnostics_prefix'
## TODO: Future: remove this
from .diagnostics import DIAGNOSTICS_PREFIX as diagnostics_prefix

if python3_flag:
    def encode(x):
        if isinstance(x, str):
            return x.encode()
        else:
            return x
    def decode(x):
        if isinstance(x, bytes):
            return x.decode()
        else:
            return x
else:
    def encode(x):
        return x
    def decode(x):
        return x
