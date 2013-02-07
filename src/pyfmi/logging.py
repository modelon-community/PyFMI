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
Utility functions for extracting and filtering FMU logs
"""

def FMU_write_log_to_file(model, tags=[], file_name='fmu_log.txt'):
    f = open(file_name,'w')

    if len(tags)==0:
        for msg in model.get_log():
            f.write("FMIL: module = " + msg[0] + " log level = " + str(msg[1]) + ": " + msg[2] + "\n")
    else:
        for msg in model.get_log():
            for tag in tags:
            	if msg[2].find(tag)>=0:
                    f.write(msg[2].split(tag)[1] + "\n")

    f.close()
    
