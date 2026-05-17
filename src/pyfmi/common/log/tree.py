#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Copyright (C) 2020 Modelon AB
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
Tree representation for an FMU XML log format

Each node is represented as a Node, Comment, or leaf (other types)
"""

class Comment(object):
    """Log comment node.

    Attributes:
    text -- the comment text without enclosing braces {}
    """
    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return '<Comment: ' + self.text + '>'

class Node(object):
    """Log node.

    Attributes:
    type  -- a string
    nodes -- a list of child nodes, in order
    """

    def __init__(self, node_type):
        self.type  = node_type
        self.nodes = []
        self.keys  = []
        self.dict  = {}

    def add(self, node, key=None):
        self.keys.append(key)
        self.nodes.append(node)
        if key is not None:
            if key in self.dict:
                # Duplicate attribute value ==> record no value. (should not happen)
                # consider: Is None the best to use for this?
                self.dict[key] = None
            else:
                self.dict[key] = node

    def __repr__(self):
        return ('<' + self.type + ' node with ' + repr(len(self.nodes))
                + ' subnodes, and named subnodes ' + repr(list(self.dict.keys())) + '>')


    def __iter__(self):
        return iter(self.nodes)

    def __contains__(self, key):
        return key in self.dict

    def __getitem__(self, key):
        return self.dict[key]

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            classname = type(self).__name__
            msg = "%s object has no attribute %s"%(classname, name)
            raise AttributeError(msg)

    def __setitem__(self, key, value):
         self.dict[key] = value
    
    def find(self, types):
        """
        Return a list of children with the given type(s), in order.

        types may be a string or list of strings.
        """
        if isinstance(types, str):
            types = [types]
        
        nodes = []        
        for node in self.nodes:
            if isinstance(node, Node):
                if node.type in types:
                    nodes.append(node)
                else:
                    nodes.extend(node.find(types))
        return nodes
