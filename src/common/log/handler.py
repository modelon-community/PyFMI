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
    def __init__(self, max_log_size: int):
        self._max_log_size = max_log_size

    def _set_max_log_size(self, val: int):
        self._max_log_size = val
    max_log_size = property(
        fget = lambda self: self._max_log_size,
        fset = _set_max_log_size,
        doc = "Maximal size (number of characters) of raw text log."
    )

    def capi_start_callback(self, limit_reached: bool, current_log_size: int):
        """Callback invoked directly before an FMI CAPI call."""
        pass

    def capi_end_callback(self, limit_reached: bool, current_log_size: int):
        """Callback invoked directly after an FMI CAPI call."""
        pass

class LogHandlerDefault(LogHandler):
    """Default LogHandler that uses checkpoints around FMI CAPI calls to 
    ensure logs are truncated at checkpoints. For FMUs generating XML during 
    CAPI calls, this ensures valid XML. """
    def __init__(self, max_log_size: int):
        super().__init__(max_log_size)
        self._log_checkpoint = 0

    log_checkpoint = property(
        fget = lambda self: self._log_checkpoint,
        doc = "Latest log size before/after a FMU CAPI call that does not exceed the maximum log size."
    )

    def _update_checkpoint(self, limit_reached: bool, current_log_size: int):
        if not limit_reached and (current_log_size <= self.max_log_size):
            self._log_checkpoint = current_log_size

    def capi_start_callback(self, limit_reached: bool, current_log_size: int):
        self._update_checkpoint(limit_reached, current_log_size)

    def capi_end_callback(self, limit_reached: bool, current_log_size: int):
        self._update_checkpoint(limit_reached, current_log_size)
