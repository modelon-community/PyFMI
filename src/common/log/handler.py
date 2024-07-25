#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2024 Modelon AB
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
XXX: There are some practical limitations around how this works 
if one updated the maximum log size after previously exceeding it
"""

class LogHandler:
    """Base class for a log handling class."""
    def __init__(self, max_log_size):
        self._max_log_size = max_log_size

    def _set_max_log_size(self, val):
        self._max_log_size = val
    max_log_size = property(
        fget = lambda self: self._max_log_size,
        fset = _set_max_log_size,
        doc = "Maximal size (number of characters) of raw text log."
    )

    def capi_start_callback(self, current_log_size):
        """Callback invoked directly before an FMI CAPI call."""
        pass

    def capi_end_callback(self, current_log_size):
        """Callback invoked directly after an FMI CAPI call."""
        pass

class LogHandlerDefault(LogHandler):
    """Default LogHandler that uses checkpoints around FMI CAPI calls to 
    ensure logs are truncated at checkpoints. For FMUs generating XML during 
    CAPI calls, this ensures valid XML. """
    def __init__(self, max_log_size):
        super().__init__(max_log_size)
        self._log_checkpoint = 0

    def _set_log_checkpoint(self, val):
        self._log_checkpoint = val
    log_checkpoint = property(
        fget = lambda self: self._log_checkpoint,
        fset = _set_log_checkpoint,
        doc = "Latest log size before/after a FMU CAPI call that does not exceed the maximum log size."
    )

    def _update_checkpoint(self, current_log_size):
        if current_log_size <= self.max_log_size:
            self.log_checkpoint = current_log_size

    def capi_start_callback(self, current_log_size):
        self._update_checkpoint(current_log_size)

    def capi_end_callback(self, current_log_size):
        self._update_checkpoint(current_log_size)
