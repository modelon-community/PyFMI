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

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import os
import logging

# TODO: Status comparisons should actually use the FMIL.jm_status_enum

from pyfmi.fmi cimport (
    ModelBase, 
    # WorkerClass2,
    import_and_get_version, 
    # FMUException,
    # check_fmu_args,
    # create_temp_dir,
    # InvalidVersionException
)
# TODO: cyclic dependencies right here
# from pyfmi.fmi import (
#     # PyEventInfo, 
#     FMUException, 
#     check_fmu_args, 
#     create_temp_dir, 
#     InvalidVersionException, 
#     # InvalidBinaryException, 
#     # InvalidXMLException, 
# )
cimport pyfmi.fmil_import as FMIL
cimport pyfmi.fmi3.fmil3_import as FMIL3

from pyfmi.common.core import create_temp_dir

from pyfmi.fmi_util cimport encode, decode

# Should use the same variable as in fmi.pyx
FMI_DEFAULT_LOG_LEVEL = FMIL.jm_log_level_error

#CALLBACKS
cdef void importlogger3(FMIL.jm_callbacks* c, FMIL.jm_string module, FMIL.jm_log_level_enu_t log_level, FMIL.jm_string message):
    if c.context != NULL:
        (<FMUModelBase3>c.context)._logger(module, log_level, message)

# XXX: These shouldn't be here, but otherwise we get cyclic dependencies of fmi.pyx & fmi3.pyx
class FMUException(Exception):
    pass
class InvalidFMUException(FMUException):
    pass
class InvalidXMLException(InvalidFMUException):
    pass
class InvalidBinaryException(InvalidFMUException):
    pass
class InvalidVersionException(InvalidFMUException):
    pass

# XXX: These shouldn't be here, but otherwise we get cyclic dependencies of fmi.pyx & fmi3.pyx
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
        if not fmu_full_path.endswith('.fmu' if isinstance(fmu_full_path, str) else encode('.fmu')):
            raise FMUException("Instantiating an FMU requires an FMU (.fmu) file, specified file has extension {}".format(os.path.splitext(fmu_full_path)[1]))

        if not os.path.isfile(fmu_full_path):
            raise FMUException('Could not locate the FMU in the specified directory.')


cdef class FMUModelBase3(ModelBase):
    """
    FMI Model loaded from a dll.
    """
    def __init__(self, fmu, log_file_name = "", log_level = FMI_DEFAULT_LOG_LEVEL,
                 _unzipped_dir = None, _connect_dll = True, allow_unzipped_fmu = False):
        """
        Constructor of the model.

        Parameters::

            fmu --
                Name of the fmu as a string.

            log_file_name --
                Filename for file used to save log messages.
                This argument can also be a stream if it supports 'write', for full functionality
                it must also support 'seek' and 'readlines'. If the stream requires use of other methods, such as 'drain'
                for asyncio-streams, then this needs to be implemented on the user-side, there is no additional methods invoked
                on the stream instance after 'write' has been invoked on the PyFMI side.
                The stream must also be open and writable during the entire time.
                Default: "" (Generates automatically)

            log_level --
                Determines the logging output. Can be set between 0
                (no logging) and 7 (everything).
                Default: 2 (log error messages)
            allow_unzipped_fmu --
                If set to True, the argument 'fmu' can be a path specifying a directory
                to an unzipped FMU. The structure of the unzipped FMU must conform
                to the FMI specification.
                Default: False

        Returns::

            A model as an object from the class FMUModelFMU3
        """

        cdef int  status
        cdef dict reals_continuous
        cdef dict reals_discrete
        cdef dict int_discrete
        cdef dict bool_discrete

        #Call super
        ModelBase.__init__(self)

        #Contains the log information
        self._log = []

        #Used for deallocation
        self._allocated_context = 0
        self._allocated_dll = 0
        self._allocated_xml = 0
        self._allocated_fmu = 0
        self._initialized_fmu = 0
        self._fmu_temp_dir = NULL
        self._fmu_log_name = NULL

        # Used to adjust behavior if FMU is unzipped
        self._allow_unzipped_fmu = 1 if allow_unzipped_fmu else 0

        #Default values
        # self._t = None
        # self._A = None
        # self._group_A = None
        # self._mask_A = None
        # self._B = None
        # self._group_B = None
        # self._C = None
        # self._group_C = None
        # self._D = None
        # self._group_D = None
        # self._states_references = None
        # self._derivatives_references = None
        # self._outputs_references = None
        # self._inputs_references = None
        # self._derivatives_states_dependencies = None
        # self._derivatives_inputs_dependencies = None
        # self._outputs_states_dependencies = None
        # self._outputs_inputs_dependencies = None
        # self._has_entered_init_mode = False
        # self._last_accepted_time = 0.0

        #Internal values
        # self._pyEventInfo   = PyEventInfo()
        # self._worker_object = WorkerClass2()

        #Specify the general callback functions
        self.callbacks.malloc  = FMIL.malloc
        self.callbacks.calloc  = FMIL.calloc
        self.callbacks.realloc = FMIL.realloc
        self.callbacks.free    = FMIL.free
        self.callbacks.logger  = importlogger3
        self.callbacks.context = <void*>self

        #Specify FMI3 related callbacks
        # self.callBackFunctions.logger               = FMIL3.fmi3_log_forwarding
        # self.callBackFunctions.allocateMemory       = FMIL.calloc
        # self.callBackFunctions.freeMemory           = FMIL.free
        # self.callBackFunctions.stepFinished         = NULL
        # self.callBackFunctions.componentEnvironment = NULL

        self._instance_environment = NULL # TODO
        self._log_message = NULL # TODO

        if log_level >= FMIL.jm_log_level_nothing and log_level <= FMIL.jm_log_level_all:
            if log_level == FMIL.jm_log_level_nothing:
                enable_logging = False
            else:
                enable_logging = True
            self.callbacks.log_level = log_level
        else:
            raise FMUException("The log level must be between %d and %d"%(FMIL.jm_log_level_nothing, FMIL.jm_log_level_all))
        self._enable_logging = enable_logging

        self._fmu_full_path = encode(os.path.abspath(fmu))
        check_fmu_args(self._allow_unzipped_fmu, fmu, self._fmu_full_path)

        # Create a struct for allocation
        self._context           = FMIL.fmi_import_allocate_context(&self.callbacks)
        self._allocated_context = 1

        #Get the FMI version of the provided model
        if _unzipped_dir:
            fmu_temp_dir = encode(_unzipped_dir)
        elif self._allow_unzipped_fmu:
            fmu_temp_dir = encode(fmu)
        else:
            fmu_temp_dir  = encode(create_temp_dir())
        fmu_temp_dir = os.path.abspath(fmu_temp_dir)
        self._fmu_temp_dir = <char*>FMIL.malloc((FMIL.strlen(fmu_temp_dir)+1)*sizeof(char))
        FMIL.strcpy(self._fmu_temp_dir, fmu_temp_dir)

        if _unzipped_dir:
            # If the unzipped directory is provided we assume that the version
            # is correct. This is due to that the method to get the version
            # unzips the FMU which we already have done.
            self._version = FMIL.fmi_version_3_0_enu
        else:
            self._version = import_and_get_version(self._context, self._fmu_full_path,
                                                   fmu_temp_dir, self._allow_unzipped_fmu)

        # Check the version
        if self._version == FMIL.fmi_version_unknown_enu:
            last_error = decode(FMIL.jm_get_last_error(&self.callbacks))
            if enable_logging:
                raise InvalidVersionException("The FMU could not be loaded. The FMU version could not be determined. " + last_error)
            else:
                raise InvalidVersionException("The FMU could not be loaded. The FMU version could not be determined. Enable logging for possibly more information.")

        if self._version != FMIL.fmi_version_3_0_enu:
            last_error = decode(FMIL.jm_get_last_error(&self.callbacks))
            if enable_logging:
                raise InvalidVersionException("The FMU could not be loaded. The FMU version is not supported by this class. " + last_error)
            else:
                raise InvalidVersionException("The FMU could not be loaded. The FMU version is not supported by this class. Enable logging for possibly more information.")

        # Parse xml and check fmu-kind
        self._fmu = FMIL3.fmi3_import_parse_xml(self._context, self._fmu_temp_dir, NULL)

        if self._fmu is NULL:
            last_error = decode(FMIL.jm_get_last_error(&self.callbacks))
            if enable_logging:
                raise InvalidXMLException("The FMU could not be loaded. The model data from 'modelDescription.xml' within the FMU could not be read. " + last_error)
            else:
                raise InvalidXMLException("The FMU could not be loaded. The model data from 'modelDescription.xml' within the FMU could not be read. Enable logging for possible more information.")

        # self.callBackFunctions.componentEnvironment = <FMIL.fmi2_component_environment_t>self._fmu
        self._fmu_kind = FMIL3.fmi3_import_get_fmu_kind(self._fmu)
        print("\t PyFMI, fmu_kind: ", self._fmu_kind)
        self._allocated_xml = 1

        #FMU kind is unknown
        if self._fmu_kind == FMIL3.fmi3_fmu_kind_unknown:
            last_error = decode(FMIL.jm_get_last_error(&self.callbacks))
            if enable_logging:
                raise InvalidVersionException("The FMU could not be loaded. The FMU kind could not be determined. " + last_error)
            else:
                raise InvalidVersionException("The FMU could not be loaded. The FMU kind could not be determined. Enable logging for possibly more information.")
        else:
            if isinstance(self, FMUModelME3):
                self._fmu_kind = FMIL3.fmi3_fmu_kind_me
            elif isinstance(self, FMUModelCS3):
                self._fmu_kind = FMIL3.fmi3_fmu_kind_cs
            else:
                # TODO: This was simplified from FMUModel2; is there a use in supporting this as stand-alone, isn't this meant to be an abstract base class?
                # TODO: Scheduled execution
                raise FMUException("FMUModelBase3 cannot be used directly, use FMUModelME3 or FMUModelCS3.")

        # Connect the DLL
        if _connect_dll:
            self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
            # status = FMIL3.fmi3_import_create_dllfmu(self._fmu, self._fmu_kind, &self.callBackFunctions)
            status = FMIL3.fmi3_import_create_dllfmu(self._fmu, self._fmu_kind, self._instance_environment, self._log_message)
            self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
            if status == FMIL.jm_status_error:
                last_error = decode(FMIL3.fmi3_import_get_last_error(self._fmu))
                if enable_logging:
                    raise InvalidBinaryException("The FMU could not be loaded. Error loading the binary. " + last_error)
                else:
                    raise InvalidBinaryException("The FMU could not be loaded. Error loading the binary. Enable logging for possibly more information.")
            self._allocated_dll = 1

        # Load information from model
        if isinstance(self, FMUModelME3):
            self._modelId = decode(FMIL3.fmi3_import_get_model_identifier_ME(self._fmu))
        elif isinstance(self, FMUModelCS3):
            self._modelId = decode(FMIL3.fmi3_import_get_model_identifier_CS(self._fmu))
        else:
            # TODO: SE
            raise FMUException("FMUModelBase3 cannot be used directly, use FMUModelME3 or FMUModelCS3.")

        # Connect the DLL
        self._modelName = decode(FMIL3.fmi3_import_get_model_name(self._fmu))
        print("\tfmi3.pyx model name :", self._modelName)
        # TODO
        # self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        # self._nEventIndicators = FMIL.fmi2_import_get_number_of_event_indicators(self._fmu) # TODO: needs wrapper now
        # self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        # self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        # self._nContinuousStates = FMIL.fmi2_import_get_number_of_continuous_states(self._fmu) # TODO: needs wrapper now
        # self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if not isinstance(log_file_name, str):
            self._set_log_stream(log_file_name)
            for i in range(len(self._log)):
                self._log_stream.write("FMIL: module = %s, log level = %d: %s\n"%(self._log[i][0], self._log[i][1], self._log[i][2]))
        else:
            fmu_log_name = encode((self._modelId + "_log.txt") if log_file_name == "" else log_file_name)
            self._fmu_log_name = <char*>FMIL.malloc((FMIL.strlen(fmu_log_name)+1)*sizeof(char))
            FMIL.strcpy(self._fmu_log_name, fmu_log_name)

            # Create the log file
            with open(self._fmu_log_name, 'w') as file:
                for i in range(len(self._log)):
                    file.write("FMIL: module = %s, log level = %d: %s\n"%(self._log[i][0], self._log[i][1], self._log[i][2]))

        self._log = []

    def instantiate(self, name = "Model", visible = False):
        """
        Instantiate the model.

        Parameters::

            name --
                The name of the instance.
                Default: 'Model'

            visible --
                Defines if the simulator application window should be visible or not.
                Default: False, not visible.

        Calls the low-level FMI function: fmi3Instantiate.
        """

        cdef FMIL3.fmi3_boolean_t log = False # TODO; should be taken from function parameter
        cdef FMIL3.fmi3_boolean_t vis = visible
        cdef FMIL.jm_status_enu_t status
        instanceName = encode(name)

        # TODO: Should likely have separate functions for the different FMU types

        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        if isinstance(self, FMUModelME3):
            # TODO: Call is outdated on current FMIL master
            print("Instantiating Model Exchange FMU")
            status = FMIL3.fmi3_import_instantiate_model_exchange(self._fmu, instanceName, NULL, vis, log, NULL, NULL)
        elif isinstance(self, FMUModelCS3):
            # TODO
            print("Instantiating Co Simulation FMU")
            # TODO
            # status = FMIL3.fmi3_import_instantiate_co_simulation(self._fmu, instanceName, NULL, vis, log, NULL, NULL)
            pass
        else:
            raise FMUException('The instance is an instance of an ME-model or a CS-model. Use load_fmu for correct loading.')
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status != FMIL.jm_status_success:
            raise FMUException('Failed to instantiate the model. See the log for possibly more information.')

        self._allocated_fmu = 1

    def initialize(self, tolerance_defined=True, tolerance="Default", start_time="Default", stop_time_defined=False, stop_time="Default"):
        """
        Initializes the model and computes initial values for all variables.
        Additionally calls the setup experiment, if not already called.

        Parameters::

            tolerance_defined --
                Specifies if the model is to be solved with an error
                controlled algorithm.
                Default: True

            tolerance --
                The tolerance used by the error controlled algorithm.
                Default: The tolerance defined in the model description

            start_time --
                Start time of the simulation.
                Default: The start time defined in the model description.

            stop_time_defined --
                Defines if a fixed stop time is defined or not. If this is
                set the simulation cannot go past the defined stop time.
                Default: False

            stop_time --
                Stop time of the simulation.
                Default: The stop time defined in the model description.

        Calls the low-level FMI functions: fmi2_import_setup_experiment (optionally)
                                           fmi2EnterInitializationMode,
                                           fmi2ExitInitializationMode
        """
        log_open = self._log_open()
        if not log_open and self.get_log_level() > 2:
            self._open_log_file()


        try:
            self.time = 0. # TODO
            # if self.time is None:
            #     self.setup_experiment(tolerance_defined, tolerance, start_time, stop_time_defined, stop_time)
            # TODO: What setup_experiment previously did now needs to be done here
            # TODO: Forward input to enter_initialization_mode
            self.enter_initialization_mode()
            self.exit_initialization_mode()
        except Exception:
            if not log_open and self.get_log_level() > 2:
                self._close_log_file()

            raise

        if not log_open and self.get_log_level() > 2:
            self._close_log_file()

    def enter_initialization_mode(self):
        """
        Enters initialization mode by calling the low level FMI function
        fmi3EnterInitializationMode.

        Note that the method initialize() performs both the enter and
        exit of initialization mode.
        """
        # TODO: Fill with sensible values
        cdef FMIL3.fmi3_boolean_t tolerance_defined = True
        cdef FMIL3.fmi3_boolean_t stop_time_defined = True
        cdef FMIL3.fmi3_float64_t tolerance = 1e-6
        cdef FMIL3.fmi3_float64_t start_time = 0
        cdef FMIL3.fmi3_float64_t stop_time = 1

        if self.time is None:
            raise FMUException("Setup Experiment has to be called prior to the initialization method.")


        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_enter_initialization_mode(
            self._fmu,
            tolerance_defined,
            tolerance,
            start_time,
            stop_time_defined,
            stop_time
        )
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status == 1:
            if self._enable_logging:
                logging.warning(
                    'Enter Initialize returned with a warning.' \
                    ' Check the log for information (model.get_log).')
            else:
                logging.warning('Enter Initialize returned with a warning.' \
                    ' Enable logging for more information, (load_fmu(..., log_level=4)).')

        if status > 1:
            if self._enable_logging:
                raise FMUException(
                    'Enter Initialize returned with an error.' \
                    ' Check the log for information (model.get_log).')
            else:
                raise FMUException('Enter Initialize returned with an error.' \
                    ' Enable logging for more information, (load_fmu(..., log_level=4)).')

        self._has_entered_init_mode = True

        # TODO: Workaround to be able to legally call terminate directly afterwards
        # initialize otherwise leads to event mode
        return status

    def exit_initialization_mode(self):
        """
        Exit initialization mode by calling the low level FMI function
        fmi2ExitInitializationMode.

        Note that the method initialize() performs both the enter and
        exit of initialization mode.
        """
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        status = FMIL3.fmi3_import_exit_initialization_mode(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)

        if status == 1:
            if self._enable_logging:
                logging.warning(
                    'Exit Initialize returned with a warning.' \
                    ' Check the log for information (model.get_log).')
            else:
                logging.warning('Exit Initialize returned with a warning.' \
                    ' Enable logging for more information, (load_fmu(..., log_level=4)).')

        if status > 1:
            if self._enable_logging:
                raise FMUException(
                    'Exit Initialize returned with an error.' \
                    ' Check the log for information (model.get_log).')
            else:
                raise FMUException('Exit Initialize returned with an error.' \
                    ' Enable logging for more information, (load_fmu(..., log_level=4)).')


        self._initialized_fmu = 1
        status = FMIL3.fmi3_import_enter_continuous_time_mode(self._fmu)

        return status

    def terminate(self):
        """
        Calls the FMI function fmi3Terminate() on the FMU.
        After this call, any call to a function changing the state of the FMU will fail.
        """
        self._log_handler.capi_start_callback(self._max_log_size_msg_sent, self._current_log_size)
        FMIL3.fmi3_import_terminate(self._fmu)
        self._log_handler.capi_end_callback(self._max_log_size_msg_sent, self._current_log_size)
        self._initialized_fmu = 0 # Do not call terminate again in dealloc

cdef class FMUModelCS3(FMUModelBase3):
    """
    Model-exchange model loaded from a dll
    """
    pass

cdef class FMUModelME3(FMUModelBase3):
    """
    Model-exchange model loaded from a dll
    """
    def __init__(self, fmu, log_file_name = "", log_level = FMI_DEFAULT_LOG_LEVEL,
                 _unzipped_dir = None, _connect_dll = True, allow_unzipped_fmu = False):
        """
        Constructor of the model.

        Parameters::

            fmu --
                Name of the fmu as a string.

            log_file_name --
                Filename for file used to save log messages.
                This argument can also be a stream if it supports 'write', for full functionality
                it must also support 'seek' and 'readlines'. If the stream requires use of other methods, such as 'drain'
                for asyncio-streams, then this needs to be implemented on the user-side, there is no additional methods invoked
                on the stream instance after 'write' has been invoked on the PyFMI side.
                The stream must also be open and writable during the entire time.
                Default: "" (Generates automatically)

            log_level --
                Determines the logging output. Can be set between 0
                (no logging) and 7 (everything).
                Default: 2 (log error messages)
            allow_unzipped_fmu --
                If set to True, the argument 'fmu' can be a path specifying a directory
                to an unzipped FMU. The structure of the unzipped FMU must conform
                to the FMI specification.
                Default: False

        Returns::

            A model as an object from the class FMUModelME3
        """
        #Call super
        FMUModelBase3.__init__(self, fmu, log_file_name, log_level,
                               _unzipped_dir, _connect_dll, allow_unzipped_fmu)

        if not (self._fmu_kind & FMIL3.fmi3_fmu_kind_me):
            raise InvalidVersionException('The FMU could not be loaded. This class only supports FMI 3.0 for Model Exchange.')

        # if self.get_capability_flags()['needsExecutionTool']:
        #     raise FMUException("The FMU specifies 'needsExecutionTool=true' which implies that it requires an external execution tool to simulate, this is not supported.")

        # self._eventInfo.newDiscreteStatesNeeded           = FMI2_FALSE
        # self._eventInfo.terminateSimulation               = FMI2_FALSE
        # self._eventInfo.nominalsOfContinuousStatesChanged = FMI2_FALSE
        # self._eventInfo.valuesOfContinuousStatesChanged   = FMI2_TRUE
        # self._eventInfo.nextEventTimeDefined              = FMI2_FALSE
        # self._eventInfo.nextEventTime                     = 0.0

        # self.force_finite_differences = 0

        # # State nominals retrieved before initialization
        # self._preinit_nominal_continuous_states = None

        # TODO: duplicate?
        # self._modelId = decode(FMIL3.fmi3_import_get_model_identifier_ME(self._fmu))

        if _connect_dll:
            self.instantiate()

    def __dealloc__(self):
        """
        Deallocate memory allocated
        """
        self._invoked_dealloc = 1

        if self._initialized_fmu == 1:
            FMIL3.fmi3_import_terminate(self._fmu)

        if self._allocated_fmu == 1:
            FMIL3.fmi3_import_free_instance(self._fmu)

        if self._allocated_dll == 1:
            FMIL3.fmi3_import_destroy_dllfmu(self._fmu)

        if self._allocated_xml == 1:
            FMIL3.fmi3_import_free(self._fmu)

        if self._fmu_temp_dir != NULL:
            if not self._allow_unzipped_fmu:
                FMIL.fmi_import_rmdir(&self.callbacks, self._fmu_temp_dir)
            FMIL.free(self._fmu_temp_dir)
            self._fmu_temp_dir = NULL

        if self._allocated_context == 1:
            FMIL.fmi_import_free_context(self._context)

        if self._fmu_log_name != NULL:
            FMIL.free(self._fmu_log_name)
            self._fmu_log_name = NULL

        if self._log_stream:
            self._log_stream = None
