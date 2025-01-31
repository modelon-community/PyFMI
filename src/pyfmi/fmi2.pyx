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

import numpy as np
cimport numpy as np
cimport pyfmi.fmil2_import as FMIL2

cdef class ScalarVariable2:
    """
    Class defining data structure based on the XML element ScalarVariable.
    """
    def __init__(self, name, value_reference, type, description = "",
                       variability = FMIL2.fmi2_variability_enu_unknown,
                       causality   = FMIL2.fmi2_causality_enu_unknown,
                       alias       = FMIL2.fmi2_variable_is_not_alias,
                       initial     = FMIL2.fmi2_initial_enu_unknown):
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
            initial

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
        self._initial         = initial

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

            The data type attribute value as enumeration: FMI2_REAL(0),
            FMI2_INTEGER(1), FMI2_BOOLEAN(2), FMI2_STRING(3) or FMI2_ENUMERATION(4).
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

            The variability of the variable: FMI2_CONSTANT(0), FMI2_FIXED(1),
            FMI2_TUNABLE(2), FMI2_DISCRETE(3), FMI2_CONTINUOUS(4) or FMI2_UNKNOWN(5)
        """
        return self._variability
    variability = property(_get_variability)

    def _get_causality(self):
        """
        Get the value of the causality attribute.

        Returns::

            The causality of the variable, FMI2_PARAMETER(0), FMI2_CALCULATED_PARAMETER(1), FMI2_INPUT(2),
            FMI2_OUTPUT(3), FMI2_LOCAL(4), FMI2_INDEPENDENT(5), FMI2_UNKNOWN(6)
        """
        return self._causality
    causality = property(_get_causality)

    def _get_alias(self):
        """
        Get the value of the alias attribute.

        Returns::

            The alias attribute value as enumeration: FMI_NO_ALIAS,
            FMI_ALIAS or FMI_NEGATED_ALIAS.
        """
        return self._alias
    alias = property(_get_alias)

    def _get_initial(self):
        """
        Get the value of the initial attribute.

        Returns::

            The initial attribute value as enumeration: FMI2_INITIAL_EXACT,
                              FMI2_INITIAL_APPROX, FMI2_INITIAL_CALCULATED,
                              FMI2_INITIAL_UNKNOWN
        """
        return self._initial
    initial = property(_get_initial)

cdef class DeclaredType2:
    """
    Class defining data structure based on the XML element Type.
    """
    def __init__(self, name, description = "", quantity = ""):
        self._name        = name
        self._description = description
        self._quantity = quantity

    def _get_name(self):
        """
        Get the value of the name attribute.

        Returns::

            The name attribute value as string.
        """
        return self._name
    name = property(_get_name)

    def _get_description(self):
        """
        Get the value of the description attribute.

        Returns::

            The description attribute value as string (empty string if
            not set).
        """
        return self._description
    description = property(_get_description)

cdef class EnumerationType2(DeclaredType2):
    """
    Class defining data structure based on the XML element Enumeration.
    """
    def __init__(self, name, description = "", quantity = "", items = None):
        DeclaredType2.__init__(self, name, description, quantity)

        self._items    = items

    def _get_quantity(self):
        """
        Get the quantity of the enumeration type.

        Returns::

            The quantity as string (empty string if
            not set).
        """
        return self._quantity
    quantity = property(_get_quantity)

    def _get_items(self):
        """
        Get the items of the enumeration type.

        Returns::

            The items of the enumeration type as a dict. The key is the
            enumeration value and the dict value is a tuple containing
            the name and description of the enumeration item.
        """
        return self._items
    items = property(_get_items)

cdef class IntegerType2(DeclaredType2):
    """
    Class defining data structure based on the XML element Enumeration.
    """
    def __init__(self, name, description = "", quantity = "", min = -np.inf, max = np.inf):
        DeclaredType2.__init__(self, name, description, quantity)

        self._min = min
        self._max = max

    def _get_max(self):
        """
        Get the max value for the type.

        Returns::

            The max value.
        """
        return self._max
    max = property(_get_max)

    def _get_min(self):
        """
        Get the min value for the type.

        Returns::

            The min value.
        """
        return self._min
    min = property(_get_min)

cdef class RealType2(DeclaredType2):
    """
    Class defining data structure based on the XML element Enumeration.
    """
    def __init__(self, name, description = "", quantity = "", min = -np.inf, max = np.inf, nominal = 1.0, unbounded = False,
                relative_quantity = False, display_unit = "", unit = ""):
        DeclaredType2.__init__(self, name, description, quantity)

        self._min = min
        self._max = max
        self._nominal = nominal
        self._unbounded = unbounded
        self._relative_quantity = relative_quantity
        self._display_unit = display_unit
        self._unit = unit

    def _get_max(self):
        """
        Get the max value for the type.

        Returns::

            The max value.
        """
        return self._max
    max = property(_get_max)

    def _get_min(self):
        """
        Get the min value for the type.

        Returns::

            The min value.
        """
        return self._min
    min = property(_get_min)

    def _get_nominal(self):
        """
        Get the nominal value for the type.

        Returns::

            The nominal value.
        """
        return self._nominal
    nominal = property(_get_nominal)

    def _get_unbounded(self):
        """
        Get the unbounded value for the type.

        Returns::

            The unbounded value.
        """
        return self._unbounded
    unbounded = property(_get_unbounded)

    def _get_relative_quantity(self):
        """
        Get the relative quantity value for the type.

        Returns::

            The relative quantity value.
        """
        return self._relative_quantity
    relative_quantity = property(_get_relative_quantity)

    def _get_display_unit(self):
        """
        Get the display unit value for the type.

        Returns::

            The display unit value.
        """
        return self._display_unit
    display_unit = property(_get_display_unit)

    def _get_unit(self):
        """
        Get the unit value for the type.

        Returns::

            The unit value.
        """
        return self._unit
    unit = property(_get_unit)

cdef class FMUState2:
    """
    Class containing a pointer to a FMU-state.
    """
    def __init__(self):
        self.fmu_state = NULL
        self._internal_state_variables = {'initialized_fmu': None,
                                          'has_entered_init_mode': None,
                                          'time': None,
                                          'callback_log_level': None,
                                          'event_info.new_discrete_states_needed': None,
                                          'event_info.nominals_of_continuous_states_changed': None,
                                          'event_info.terminate_simulation': None,
                                          'event_info.values_of_continuous_states_changed': None,
                                          'event_info.next_event_time_defined': None,
                                          'event_info.next_event_time': None}

