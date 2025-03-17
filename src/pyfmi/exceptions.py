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

# This file contains the various exceptions classes used in PyFMI

class FMUException(Exception):
    """
    An FMU exception.
    """
    pass

class FMIModel_Exception(Exception):
    """
    A FMIModel Exception.
    """
    # TODO Future; remove
    pass

class FMIModelException(FMIModel_Exception):
    """
    A FMIModel Exception.
    """
    pass

class IOException(FMUException):
    """
        Exception covering issues related to writing/reading data.
    """
    pass

class InvalidOptionException(FMUException):
    """
        Exception covering issues related to invalid choices of options.
    """
    pass

class TimeLimitExceeded(FMUException):
    pass

class InvalidFMUException(FMUException):
    """
    Exception covering problems with the imported FMU.
    """
    pass

class InvalidXMLException(InvalidFMUException):
    """
    Exception covering problem with the XML-file in the imported FMU.
    """
    pass

class InvalidBinaryException(InvalidFMUException):
    """
    Exception covering problem with the binary in the imported FMU.
    """
    pass

class InvalidVersionException(InvalidFMUException):
    """
    Exception covering problem with the version of the imported FMU.
    """
    pass
