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

"""
Module containing the FMI interface Python wrappers.
"""

"""
For profiling:
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""

import os
import logging
import fnmatch
import re
from io import UnsupportedOperation
cimport cython

import functools
import marshal

import numpy as np
cimport numpy as np

cimport pyfmi.fmil_import as FMIL
cimport pyfmi.util as pyfmi_util 
from pyfmi.exceptions import FMUException

int   = np.int32
np.int = np.int32

# FMI types
FMI_DEFAULT_LOG_LEVEL = FMIL.jm_log_level_error

cdef FMIL.fmi_version_enu_t import_and_get_version(FMIL.fmi_import_context_t* context, char* fmu_full_path, char* fmu_temp_dir, int allow_unzipped_fmu):
    """ Invokes the necessary FMIL functions to retrieve FMI version while accounting
        for the conditional 'allow_unzipped_fmu'.
    """
    if allow_unzipped_fmu:
        return FMIL.fmi_import_get_fmi_version(context, NULL, fmu_temp_dir)
    else:
        return FMIL.fmi_import_get_fmi_version(context, fmu_full_path, fmu_temp_dir)

cdef class ModelBase:
    """
    Abstract Model class containing base functionality.
    """

    def __init__(self):
        self.cache = {}
        self.file_object = None
        self._log_is_stream = 0
        self._additional_logger = None
        self._current_log_size = 0
        self._max_log_size = 2*10**9 # 2GB limit
        self._max_log_size_msg_sent = False
        self._log_stream = None
        self._modelId = None
        self._invoked_dealloc = 0 # Set to 1 when __dealloc__ is called
        self._result_file = None
        self._log_handler = LogHandlerDefault(self._max_log_size)

    def _set_log_stream(self, stream):
        """ Function that sets the class property 'log_stream' and does error handling. """
        if not hasattr(stream, 'write'):
            raise FMUException("Only streams that support 'write' are supported.")

        self._log_stream = stream
        self._log_is_stream = 1

    def set(self, variable_name, value):
        """
        Sets the given value(s) to the specified variable name(s) into the
        model. The method both accept a single variable and a list of variables.

        Parameters::

            variable_name --
                The name of the variable(s) as string/list.

            value --
                The value(s) to set.

        Example::

            model.set('damper.d', 1.1)
            model.set(['damper.d','gear.a'], [1.1, 10])
        """
        if isinstance(variable_name, basestring):
            self._set(variable_name, value) #Scalar case
        else:
            for i in xrange(len(variable_name)): #A list of variables
                self._set(variable_name[i], value[i])

    def get(self, variable_name):
        """
        Returns the value(s) of the specified variable(s). The method both
        accept a single variable and a list of variables.

        Parameters::

            variable_name --
                The name of the variable(s) as string/list.

        Returns::

            The value(s).

        Example::

            # Returns the variable d
            model.get('damper.d')
            # Returns a list of the variables
            model.get(['damper.d','gear.a'])
        """
        if isinstance(variable_name, basestring):
            return self._get(variable_name) #Scalar case
        else:
            return [self._get(variable_name[i]) for i in xrange(len(variable_name))]

    def _exec_algorithm(self, module, algorithm, options):
        """
        Helper function which performs all steps of an algorithm run which are
        common to all initialize and optimize algorithms.

        Raises::

            Exception if algorithm is not a subclass of
            common.algorithm_drivers.AlgorithmBase.
        """
        base_path = 'common.algorithm_drivers'
        algdrive = __import__(base_path, globals(), locals(), [], -1)
        AlgorithmBase = getattr(getattr(algdrive,"algorithm_drivers"), 'AlgorithmBase')

        if isinstance(algorithm, basestring):
            module = __import__(module, globals(), locals(), [algorithm], -1)
            algorithm = getattr(module, algorithm)

        if not issubclass(algorithm, AlgorithmBase):
            raise Exception(str(algorithm)+
            " must be a subclass of common.algorithm_drivers.AlgorithmBase")

        # initialize algorithm
        alg = algorithm(self, options)
        # solve optimization problem/initialize
        alg.solve()
        # get and return result
        return alg.get_result()

    def _exec_simulate_algorithm(self,
                                 start_time,
                                 final_time,
                                 input,
                                 module,
                                 algorithm,
                                 options):
        """
        Helper function which performs all steps of an algorithm run which are
        common to all simulate algorithms.

        Raises::

            Exception if algorithm is not a subclass of
            common.algorithm_drivers.AlgorithmBase.
        """
        base_path = 'common.algorithm_drivers'
        algdrive = __import__(base_path, globals(), locals(), [], 1)
        AlgorithmBase = getattr(getattr(algdrive,"algorithm_drivers"), 'AlgorithmBase')

        if isinstance(algorithm, basestring):
            module = __import__(module, globals(), locals(), [algorithm], 0)
            algorithm = getattr(module, algorithm)

        if not issubclass(algorithm, AlgorithmBase):
            raise Exception(str(algorithm)+
            " must be a subclass of common.algorithm_drivers.AlgorithmBase")

        #open log file
        self._open_log_file()

        try:
            # initialize algorithm
            alg = algorithm(start_time, final_time, input, self, options)
            # simulate
            alg.solve()
        except Exception:
            #close log file
            self._close_log_file()
            raise #Reraise the exception

        #close log file
        self._close_log_file()

        # get and return result
        return alg.get_result()

    def _exec_estimate_algorithm(self, parameters,
                                       measurements,
                                       input,
                                       module,
                                       algorithm,
                                       options):
        """
        Helper function which performs all steps of an algorithm run which are
        common to all estimation algorithms.

        Raises::

            Exception if algorithm is not a subclass of
            common.algorithm_drivers.AlgorithmBase.
        """
        base_path = 'common.algorithm_drivers'
        algdrive = __import__(base_path, globals(), locals(), [], 1)
        AlgorithmBase = getattr(getattr(algdrive,"algorithm_drivers"), 'AlgorithmBase')

        if isinstance(algorithm, basestring):
            module = __import__(module, globals(), locals(), [algorithm], 0)
            algorithm = getattr(module, algorithm)

        if not issubclass(algorithm, AlgorithmBase):
            raise Exception(str(algorithm)+
            " must be a subclass of common.algorithm_drivers.AlgorithmBase")

        #open log file
        self._open_log_file()

        try:
            # initialize algorithm
            alg = algorithm(parameters, measurements, input, self, options)
            # simulate
            alg.solve()
        except Exception:
            #close log file
            self._close_log_file()
            raise #Reraise the exception

        #close log file
        self._close_log_file()

        # get and return result
        return alg.get_result()


    def _default_options(self, module, algorithm):
        """
        Help method. Gets the options class for the algorithm specified in
        'algorithm'.
        """
        module = __import__(module, globals(), locals(), [algorithm], 0)
        algorithm = getattr(module, algorithm)

        return algorithm.get_default_options()

    def estimate(self,
                 parameters,
                 measurements,
                 input=(),
                 algorithm='SciEstAlg',
                 options={}):
        """
        Compact function for model estimation.

        The estimation method depends on which algorithm is used, this can be
        set with the function argument 'algorithm'. Options for the algorithm
        are passed as option classes or as pure dicts. See
        model.estimate_options for more details.

        The default algorithm for this function is SciEstAlg.

        Parameters::

            parameters --
                The tunable parameters (required)

            measurements --
                The measurements data (name, data). Note that the measurements
                needs to be distinct and equally spaced. (required)

            input --
                Input signal for the estimation. The input should be a 2-tuple
                consisting of first the names of the input variable(s) and then
                the data matrix.
                Default: Empty tuple.

            algorithm --
                The algorithm which will be used for the estimation is specified
                by passing the algorithm class as string or class object in this
                argument. 'algorithm' can be any class which implements the
                abstract class AlgorithmBase (found in algorithm_drivers.py). In
                this way it is possible to write own algorithms and use them
                with this function.
                Default: 'SciEstAlg'

            options --
                The options that should be used in the algorithm. For details on
                the options do:

                    >> myModel = load_fmu(...)
                    >> opts = myModel.estimate_options()
                    >> opts?

                Valid values are:
                    - A dict which gives SciEstAlgOptions with
                      default values on all options except the ones
                      listed in the dict. Empty dict will thus give all
                      options with default values.
                    - An options object.
                Default: Empty dict

        Returns::

            Result object, subclass of common.algorithm_drivers.ResultBase.
        """

        return self._exec_estimate_algorithm(parameters,
                                             measurements,
                                             input,
                                             'pyfmi.fmi_algorithm_drivers',
                                             algorithm,
                                             options)

    def estimate_options(self, algorithm='SciEstAlg'):
        """
        Get an instance of the estimation options class, filled with default
        values. If called without argument then the options class for the
        default estimation algorithm will be returned.

        Parameters::

            algorithm --
                The algorithm for which the options class should be fetched.
                Possible values are: 'SciEstAlg'.
                Default: 'SciEstAlg'

        Returns::

            Options class for the algorithm specified with default values.
        """
        return self._default_options('pyfmi.fmi_algorithm_drivers', algorithm)

    def get_last_result_file(self):
        """ Returns absolute path to the simulation result file for the last simulation if
            the fmu has been simulated. Otherwise returns None."""
        return os.path.abspath(self._result_file) if isinstance(self._result_file, str) else None

    def get_log_filename(self):
        """ Returns a name of the logfile, if logging is done to a stream it returns
            a string formatted base on the model identifier.
        """
        return '{}_log'.format(self._modelId) if self._log_is_stream else pyfmi_util.decode(self._fmu_log_name)

    def get_number_of_lines_log(self):
        """
            Returns the number of lines in the log file.
            If logging was done to a stream, this function returns 0.
        """
        num_lines = 0
        if self._fmu_log_name != NULL:
            with open(self._fmu_log_name,'r') as file:
                num_lines = sum(1 for line in file)
        return num_lines

    def print_log(self, start_lines=-1, end_lines=-1):
        """
        Prints the log information to the prompt.
        """
        cdef int N
        log = self.get_log(start_lines, end_lines)
        N = len(log)

        for i in range(N):
            print(log[i])
    
    def _get_module_name(self):
        return "Model"

    def extract_xml_log(self, file_name=None):
        """
        Extract the XML contents of a FMU log and write as a new file.
        If logging was done to a stream, it needs to support 'seek', otherwise an FMUException is raised
        when invoking this function.

        Parameters::

            file_name --
                Name of the file which holds the extracted log, or a stream to write to
                that supports the function 'write'. Default behaviour is to write to a file.
                Default: get_log_filename() + xml

        Returns::
            If extract to a file:
                file_path -- path to extracted XML file
            otherwise function returns nothing
        """
        # TODO: Needs to be here for now, causes issues with the imports in via __init__ files
        from pyfmi.common.log import extract_xml_log

        if file_name is None:
            file_name = "{}.{}".format(os.path.splitext(self.get_log_filename())[0], 'xml')

        module_name = self._get_module_name()

        is_stream = self._log_is_stream and self._log_stream
        if is_stream:
            try:
                self._log_stream.seek(0)
            except AttributeError:
                raise FMUException("In order to extract the XML-log from a stream, it needs to support 'seek'")

        if isinstance(self._log_handler, LogHandlerDefault):
            max_size = self._log_handler.get_log_checkpoint() if self._max_log_size_msg_sent else None
            extract_xml_log(file_name, self._log_stream if is_stream else self.get_log_filename(), module_name,
                            max_size = max_size)
        else:
            extract_xml_log(file_name, self._log_stream if is_stream else self.get_log_filename(), module_name)

        if isinstance(file_name, str):
            return os.path.abspath(file_name)
        else:
            # If we extract the log into a stream, return None
            return None

    def get_log(self, int start_lines=-1, int end_lines=-1):
        """
        Returns the log information as a list. To turn on the logging
        use load_fmu(..., log_level=1-7) in the loading of the FMU. The
        log is stored as a list of lists. For example log[0] are the
        first log message to the log and consists of, in the following
        order, the instance name, the status, the category and the
        message.

        This function also works if logging was done to a stream if and only if
        the stream is readable and supports 'seek' and 'readlines'.

        Returns::

            log - A list of lists.
        """
        cdef int i = 0
        cdef int num_lines = 0
        log = []

        if end_lines != -1:
            num_lines = self.get_number_of_lines_log()

        if self._fmu_log_name != NULL:
            with open(self._fmu_log_name,'r') as file:
                while True:
                    i = i + 1
                    line = file.readline()
                    if line == "":
                        break
                    if start_lines != -1 and end_lines == -1:
                        if i > start_lines:
                            break
                    elif start_lines == -1 and end_lines != -1:
                        if i < num_lines - end_lines + 1:
                            continue
                    elif start_lines != -1 and end_lines != -1:
                        if i > start_lines and i < num_lines - end_lines + 1:
                            continue
                    log.append(line.strip("\n"))
        elif self._log_stream_is_open():
            try:
                self._log_stream.seek(0)
                return [line.strip("\n") for line in self._log_stream.readlines() if line]
            except AttributeError as e:
                err_msg = "Unable to get log from stream if it does not support 'seek' and 'readlines'."
                raise FMUException(err_msg) from e
            except UnsupportedOperation as e:
                # This happens if we are not allowed to read from given stream
                err_msg = "Unable to read from given stream, make sure the stream is readable."
                raise FMUException(err_msg) from e
        elif self._log_stream is not None:
            if hasattr(self._log_stream, 'closed') and self._log_stream.closed:
                raise FMUException("Unable to get log from closed stream.")
            else:
                raise FMUException("Unable to get log from stream, please verify that it is open.")

        return log

    def _log_open(self):
        if self.file_object:
            return True
        elif self._log_stream_is_open():
            return True
        else:
            return False

    def _log_stream_is_open(self):
        """ Returns True or False based on if logging is done to a stream and if it is open or closed. """
        return self._log_stream is not None and not self._log_stream.closed

    def _open_log_file(self):
        """ Opens the log file if we are not logging into a given stream. """
        if not self._log_is_stream and self._fmu_log_name != NULL:
            self.file_object = open(self._fmu_log_name,'a')

    def _close_log_file(self):
        if self.file_object:
            self.file_object.close()
            self.file_object = None

    cdef _logger(self, FMIL.jm_string c_module, int log_level, FMIL.jm_string c_message) with gil:
        cdef FMIL.FILE *f
        module  = pyfmi_util.decode(c_module)
        message = pyfmi_util.decode(c_message)

        if self._additional_logger:
            self._additional_logger(module, log_level, message)

        if self._max_log_size_msg_sent:
            return

        msg = "FMIL: module = %s, log level = %d: %s\n"%(module, log_level, message)

        if self._current_log_size + len(msg) > self._max_log_size:
            msg = "The log file has reached its maximum size and further log messages will not be saved. To change the maximum size of the file, please use the 'set_max_log_size' method.\n"
            self._max_log_size_msg_sent = True
        self._current_log_size = self._current_log_size + len(msg)

        if self._fmu_log_name != NULL:
            if self.file_object:
                self.file_object.write(msg)
            else:
                try:
                    with open(self._fmu_log_name,'a') as file:
                        file.write(msg)
                except Exception: #In some cases (especially when Python closes the above will fail when messages are provided from the FMI terminate/free methods)
                    f = FMIL.fopen(self._fmu_log_name, "a");
                    if (f != NULL):
                        FMIL.fprintf(f, "FMIL: module = %s, log level = %d: %s\n",c_module, log_level, c_message)
                        FMIL.fclose(f)
        elif self._log_stream:
            try:
                self._log_stream.write(msg)
            except Exception:
                # Try to catch exception if stream is closed or not writable
                # which could be due to the stream is given in 'read-mode.
                if not self._invoked_dealloc:
                    if hasattr(self._log_stream, 'closed') and self._log_stream.closed:
                        logging.warning("Unable to log to closed stream.")
                    else:
                        logging.warning("Unable to log to stream.")
                self._log_stream = None
        else:
            if isinstance(self._log, list):
                self._log.append([module,log_level,message])

    def append_log_message(self, module, log_level, message):
        if self._additional_logger:
            self._additional_logger(module, log_level, message)

        if self._max_log_size_msg_sent:
            return

        msg = "FMIL: module = %s, log level = %d: %s\n"%(module, log_level, message)

        if self._current_log_size + len(msg) > self._max_log_size:
            msg = "The log file has reached its maximum size and further log messages will not be saved. To change the maximum size of the file, please use the 'set_max_log_size' method.\n"
            self._max_log_size_msg_sent = True
        self._current_log_size = self._current_log_size + len(msg)

        if self._fmu_log_name != NULL:
            if self.file_object:
                self.file_object.write(msg)
            else:
                with open(self._fmu_log_name,'a') as file:
                    file.write(msg)
        elif self._log_stream is not None:
            self._log_stream.write(msg)
        else:
            if isinstance(self._log, list):
                self._log.append([module,log_level,message])

    def set_max_log_size(self, number_of_characters):
        """
        Specifies the maximum number of characters to be written to the
        log file.

            Parameters::

                number_of_characters --
                    The maximum number of characters in the log.
                    Default: 2e9 (2GB)
        """
        self._max_log_size = number_of_characters
        self._log_handler.set_max_log_size(number_of_characters)

        if self._max_log_size > self._current_log_size: # re-enable logging
            self._max_log_size_msg_sent = False

    def get_max_log_size(self):
        """
        Returns the limit (in characters) of the log file.
        """
        return self._max_log_size

    def has_reached_max_log_size(self):
        """
        Returns True if the log has reached the maximum allowed limit on
        the size of the log file, otherwise, returns False.
        """
        return self._max_log_size_msg_sent

    def set_log_level(self, FMIL.jm_log_level_enu_t level):
        """
        Specifies the log level for PyFMI. Note that this is
        different from the FMU logging which is specified via
        set_debug_logging.

        Parameters::

            level --
                The log level. Available values:
                    NOTHING = 0
                    FATAL = 1
                    ERROR = 2
                    WARNING = 3
                    INFO = 4
                    VERBOSE = 5
                    DEBUG = 6
                    ALL = 7
        """
        if level < FMIL.jm_log_level_nothing or level > FMIL.jm_log_level_all:
            raise FMUException("Invalid log level for FMI Library (0-7).")
        self.callbacks.log_level = level

    def get_log_level(self):
        """
        Returns the log level for PyFMI, i.e., as set in set_log_level.
        Type: int
        """
        return self.callbacks.log_level

    def set_additional_logger(self, additional_logger):
        """
        Set an additional logger function that will, an addition to
        the normal FMU log file, also be fed with all model log
        messages.

        Parameter::

            additional_logger --
                The callback function that should accept three arguments:
                module(str), log_level(int), message(str)
        """
        self._additional_logger = additional_logger

    def _convert_filter(self, expression):
        """
        Convert a filter based on unix filename pattern matching to a
        list of regular expressions.

        Parameters::

            expression--
                String or list to convert.

        Returns::

            The converted filter.
        """
        regexp = []
        if isinstance(expression,str):
            regex = fnmatch.translate(expression)
            regexp = [re.compile(regex)]
        elif isinstance(expression,list):
            regex = ""
            for i in expression:
                regex = regex + fnmatch.translate(i) + "|"
            regexp = [re.compile(regex[:-1])]
        else:
            raise FMUException("Unknown input.")
        return regexp

class PyEventInfo(): # TODO: Should this be a cpdef + FMIX variants?
    pass

# TODO: Move to common/log/ ? (new .pyx file there)
cdef class LogHandler:
    """Base class for a log handling class."""
    def __init__(self, max_log_size):
        self._max_log_size = max_log_size

    cpdef void set_max_log_size(self, unsigned long val):
        self._max_log_size = val

    cpdef void capi_start_callback(self, int limit_reached, unsigned long current_log_size):
        """Callback invoked directly before an FMI CAPI call."""
        pass

    cpdef void capi_end_callback(self, int limit_reached, unsigned long current_log_size):
        """Callback invoked directly after an FMI CAPI call."""
        pass

cdef class LogHandlerDefault(LogHandler):
    """Default LogHandler that uses checkpoints around FMI CAPI calls to
    ensure logs are truncated at checkpoints. For FMUs generating XML during
    CAPI calls, this ensures valid XML. """
    def __init__(self, max_log_size):
        super().__init__(max_log_size)
        self._log_checkpoint = 0

    cdef void _update_checkpoint(self, int limit_reached, unsigned long current_log_size):
        if not limit_reached and (current_log_size <= self._max_log_size):
            self._log_checkpoint = current_log_size

    cpdef unsigned long get_log_checkpoint(self):
        return self._log_checkpoint

    cpdef void capi_start_callback(self, int limit_reached, unsigned long current_log_size):
        self._update_checkpoint(limit_reached, current_log_size)

    cpdef void capi_end_callback(self, int limit_reached, unsigned long current_log_size):
        self._update_checkpoint(limit_reached, current_log_size)

def check_fmu_args(allow_unzipped_fmu, fmu, fmu_full_path):
    """ Function utilized by two base classes and load_fmu that does the
        error checking for the three input arguments named 'allow_unzipped_fmu', 'fmu' and
        the constructed variable 'fmu_full_path'.
    """
    if allow_unzipped_fmu:
        if not os.path.isdir(fmu):
            msg = "Argument named 'fmu' must be a directory if argument 'allow_unzipped_fmu' is set to True."
            raise FMUException(msg)
        if not os.path.isfile(os.path.join(fmu, 'modelDescription.xml')):
            err_msg = "Specified fmu path '{}' needs".format(fmu)
            err_msg += " to contain a modelDescription.xml according to the FMI specification."
            raise FMUException(err_msg)
    else:
        # Check that the file referenced by fmu is appropriate
        if not fmu_full_path.endswith('.fmu' if isinstance(fmu_full_path, str) else pyfmi_util.encode('.fmu')):
            raise FMUException("Instantiating an FMU requires an FMU (.fmu) file, specified file has extension {}".format(os.path.splitext(fmu_full_path)[1]))

        if not os.path.isfile(fmu_full_path):
            raise FMUException('Could not locate the FMU in the specified directory.')

def _handle_load_fmu_exception(log_data):
    for log in log_data:
        print(log)
