#!/usr/bin/env python
# -*- coding: utf-8 -*-

#    Copyright (C) 2014 Modelon AB
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
Parser for a XML based FMU log format
"""

from xml import sax
import re
import numpy as np
from .tree import *
from pyfmi.fmi_util import python3_flag


## Leaf parser ##

floatingpoint_re = "^[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?$"
integer_re       = "^[0-9]+$"
quoted_string_re = '^"(?:[^"]|"")*"$'

integer_pattern       = re.compile(integer_re)
floatingpoint_pattern = re.compile(floatingpoint_re)
quoted_string_pattern = re.compile(quoted_string_re)

comma_re     = "((?:[^,']|(?:'[^']*'))*)(?:,|\Z)"
semicolon_re = "((?:[^;']|(?:'[^']*'))*)(?:;|\Z)"

comma_pattern     = re.compile(comma_re)
semicolon_pattern = re.compile(semicolon_re)

def parse_value(text):
    """Parse the string text and return a string, float, or int."""
    text = text.strip()
    if integer_pattern.match(text):
        return int(text)
    elif floatingpoint_pattern.match(text):
        return float(text)
    else:
        if quoted_string_pattern.match(text):
            text = text[1:-1].replace('""','"')
        else:
            assert '"' not in text            
        # for python 2 we need to avoid printing all strings as u'...'
        return text if python3_flag else text.encode('ascii', 'xmlcharrefreplace')

def parse_vector(text):
    
    text = text.strip()
    if text == "":
        return np.zeros(0)
    parts = comma_pattern.split(text)
    parts = filter(None, map(str, parts))
    return np.asarray([parse_value(part) for part in parts])

def parse_matrix(text):
    text = text.strip()
    if text == "":
        return np.zeros((0,0))
    parts = semicolon_pattern.split(text)
    parts = filter(None, map(str, parts))
    return np.asarray([parse_vector(part) for part in parts])
    

## SAX based parser ##

attribute_ns = "http://www.modelon.com/log/attribute"
node_ns      = "http://www.modelon.com/log/node"

class ContentHandler(sax.ContentHandler):
    def __init__(self):
        sax.ContentHandler.__init__(self)
        self.nodes = [Node("Log")]
        self.leafparser = None
        self.leafkey    = None
        self.chars      = []

    def get_root(self):
        return self.nodes[0].nodes[0]

    def take_chars(self):
        chars = "".join(self.chars)
        self.chars = []
        return chars

    def create_comment(self):
        if len(self.chars) > 0:
            comment = self.take_chars()
            if comment != '':
                self.nodes[-1].add(Comment(comment))

# sax.ContentHandler callbacks:

    def characters(self, content):
        self.chars.append(content)

    def startElement(self, type, attrs):
        self.create_comment()
        
        key = attrs.get('name')
        # convert to string if key is not None
        # because Python 2.x returns unicode while Python 3.x does not
        key = key if not key else str(key)

        self.chars = []
        self.leafkey = self.leafparser = None
        
        if   type == 'value':  self.leafparser = parse_value
        elif type == 'vector': self.leafparser = parse_vector
        elif type == 'matrix': self.leafparser = parse_matrix

        if self.leafparser is not None:
            self.leafkey = key
        else:
            node = Node(type)            
            #if len(self.nodes) > 0:
            self.nodes[-1].add(node, key)
            self.nodes.append(node)
            
    def endElement(self, type):        
        # todo: verify name matching?
        if self.leafparser is not None:
            node = self.leafparser(self.take_chars())
            self.nodes[-1].add(node, self.leafkey)
            self.leafparser = self.leafkey = None
        else:
            self.create_comment()
            self.nodes.pop()

def create_parser():
    # note: hope that we get an IncrementalParser,
    # or FMU log parsing won't work
    parser = sax.make_parser()
    handler = ContentHandler()
    parser.setContentHandler(handler)
    return parser, handler

def parse_xml_log(filename, accept_errors=False):
    """
    Parse a pure XML log as created by extract_xml_log, return the root node.

    If accept_errors is True and a parse error occurs, the results of parsing
    up to that point will be returned.
    """
    parser, handler = create_parser()
    try:
        parser.parse(filename)
    except sax.SAXException as e:
        if accept_errors:
            print('Warning: Failure during parsing of XML FMU log:\n', e)
            print('Parsed log will be incomplete.')
        else:
            raise Exception('Failed to parse XML FMU log:\n' + repr(e))
        
    return handler.get_root()


def parse_fmu_xml_log(filename, modulename = 'Model', accept_errors=False):
    """
    Parse the XML contents of a FMU log and return the root node.

    modulename selects the module as recorded in the beginning of each line by
    FMI Library. If accept_errors is True and a parse error occurs, the
    results of parsing up to that point will be returned.
    """
    parser, handler = create_parser()
    try:
        with open(filename, 'r') as f:
            filter_xml_log(parser.feed, f, modulename)

        parser.close()
    except sax.SAXException as e:
        if accept_errors:
            print('Warning: Failure during parsing of XML FMU log:\n', e)
            print('Parsed log will be incomplete')
        else:
            raise Exception('Failed to parse XML FMU log:\n' + repr(e))
    
    return handler.get_root()

def extract_xml_log(destfilename, filename, modulename = 'Model'):
    """
    Extract the XML contents of a FMU log and write as a new file destfilename.

    modulename selects the module as recorded in the beginning of each line by
    FMI Library.
    """
    with open(filename, 'r') as sourcefile:
        with open(destfilename, 'w') as destfile:
            filter_xml_log(destfile.write, sourcefile, modulename)

def filter_xml_log(write, sourcefile, modulename = 'Model'):
    write('<?xml version="1.0" encoding="UTF-8"?>\n<JMILog category="info">\n')

    pre_re = r'FMIL: module = ' + modulename + r', log level = ([0-9]+): \[([^]]+)\]\[FMU status:([^]]+)\] '
    pre_pattern = re.compile(pre_re)

    for line in sourcefile:
        m = pre_pattern.match(line)
        if m is not None:
            # log_level, category, fmu_status = m.groups()
            write(line[m.end():])

    write('</JMILog>\n')
