#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2025Modelon AB
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

cimport pyfmi.fmil1_import as FMIL1

cdef class ScalarVariable:
    """
    Class defining data structure based on the XML element ScalarVariable.
    """
    def __init__(self, name, value_reference, type, description="",
                       variability=FMIL1.fmi1_variability_enu_continuous,
                       causality=FMIL1.fmi1_causality_enu_internal,
                       alias=FMIL1.fmi1_variable_is_not_alias):
        """
        Class collecting information about a scalar variable and its
        attributes. The following attributes can be retrieved::

            name
            value_reference
            type
            description
            variability
            causality
            alias

        For further information about the attributes, see the info on a
        specific attribute.
        """

        self._name            = name
        self._value_reference = value_reference
        self._type            = type
        self._description     = description
        self._variability     = variability
        self._causality       = causality
        self._alias           = alias

    def _get_name(self):
        """
        Get the value of the name attribute.

        Returns::

            The name attribute value as string.
        """
        return self._name
    name = property(_get_name)

    def _get_value_reference(self):
        """
        Get the value of the value reference attribute.

        Returns::

            The value reference as unsigned int.
        """
        return self._value_reference
    value_reference = property(_get_value_reference)

    def _get_type(self):
        """
        Get the value of the data type attribute.

        Returns::

            The data type attribute value as enumeration: FMI_REAL(0),
            FMI_INTEGER(1), FMI_BOOLEAN(2), FMI_STRING(3) or FMI_ENUMERATION(4).
        """
        return self._type
    type = property(_get_type)

    def _get_description(self):
        """
        Get the value of the description attribute.

        Returns::

            The description attribute value as string (empty string if
            not set).
        """
        return self._description
    description = property(_get_description)

    def _get_variability(self):
        """
        Get the value of the variability attribute.

        Returns::

            The variability attribute value as enumeration:
            FMI_CONSTANT(0), FMI_PARAMETER(1), FMI_DISCRETE(2) or FMI_CONTINUOUS(3).
        """
        return self._variability
    variability = property(_get_variability)

    def _get_causality(self):
        """
        Get the value of the causality attribute.

        Returns::

            The causality attribute value as enumeration: FMI_INPUT(0),
            FMI_OUTPUT(1), FMI_INTERNAL(2) or FMI_NONE(3).
        """
        return self._causality
    causality = property(_get_causality)

    def _get_alias(self):
        """
        Get the value of the alias attribute.

        Returns::

            The alias attribute value as enumeration: FMI_NO_ALIAS(0),
            FMI_ALIAS(1) or FMI_NEGATED_ALIAS(-1).
        """
        return self._alias
    alias = property(_get_alias)

