#!/usr/bin/env python
# -*- coding: utf-8 -*-

#    Copyright (C) 2020 Modelon AB
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, version 3 of the License.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Prettyprinter for in memory log trees (as parsed by parser.py)
"""

from numpy import ndarray
from .tree import *

def prettyprint(write, node):
    """Prettyprint a log node to the write callback write."""
    if isinstance(node, Node):
        write('<' + node.type + '>')
        for (key, child) in zip(node.keys, node.nodes):
            if key is not None:
                write('<' + key + ' = ')
                prettyprint(write, child)
                write('>')
            else:
                prettyprint(write, child)            
        write('</' + node.type + '>')
    elif isinstance(node, Comment):
        write(node.text)
    elif isinstance(node, str):
        write(repr(node))
    elif isinstance(node, ndarray) and node.ndim > 1:
        # Multiline array literals won't be indented properly; since we're
        # not keeping track of indentation, just put them on their own lines.
        write('\n')
        write(str(node))
    else:
        write(str(node))

def prettyprint_to_file(destfilename, node):
    with open(destfilename, 'w') as f:
        prettyprint(f.write, node)
