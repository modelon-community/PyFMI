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
Module for simulation algorithms to be used together with
pyfmi.fmi.FMUModel.simulate.
"""

#from abc import ABCMeta, abstractmethod
import logging
import time
import numpy as N

import pyfmi
import pyfmi.fmi as fmi
from pyfmi.common.algorithm_drivers import AlgorithmBase, AssimuloSimResult, OptionBase, InvalidAlgorithmOptionException, InvalidSolverArgumentException, JMResultBase
from pyfmi.common.io import ResultDymolaTextual, ResultHandlerFile, ResultHandlerMemory, ResultHandler
from pyfmi.common.core import TrajectoryLinearInterpolation
from pyfmi.common.core import TrajectoryUserFunction

try:
    import assimulo
    assimulo_present = True
except:
    logging.warning(
        'Could not load Assimulo module. Check pyfmi.check_packages()')
    assimulo_present = False

if assimulo_present:
    from pyfmi.simulation.assimulo_interface import FMIODE, FMIODESENS,FMIODE_deprecated, FMIODE2, FMIODESENS2
    from pyfmi.simulation.assimulo_interface import write_data
    import assimulo.solvers as solvers

default_int = int
int = N.int32
N.int = N.int32


class FMIResult(JMResultBase):

    def __getitem__(self, key):
        val = self.result_data.get_variable_data(key)

        if self.is_variable(key):
            return val.x
        else:
            #When there is a sensitivity variable (dx/dp) in the result
            #the variable does not exists in the XML file, so cache the
            #error and set variability to 0. If the variable does not
            #exists in the result file, an error is raised prior to this.
            try:
                variability = self.model.get_variable_variability(key)
            except FMUException:
                variability = fmi.FMI_CONTINUOUS

            if variability == fmi.FMI_CONSTANT or variability == fmi.FMI_PARAMETER:
            #Variable is a parameter or constant
                return val.x[0]
            else:
                time = self.result_data.get_variable_data('time')
                return N.array([val.x[0]]*N.size(time.t))

        #return self.result_data.get_variable_data(key)

class AssimuloFMIAlgOptions(OptionBase):
    """
    Options for the solving the FMU using the Assimulo simulation package.
    Currently, the only solver in the Assimulo package that fully supports
    simulation of FMUs is the solver CVode.

    Assimulo options::

        solver --
            Specifies the simulation algorithm that is to be used. Currently the
            only supported solver is 'CVode'.
            Default: 'CVode'

        ncp    --
            Number of communication points. If ncp is zero, the solver will
            return the internal steps taken.
            Default: '0'

        initialize --
            If set to True, the initializing algorithm defined in the FMU model
            is invoked, otherwise it is assumed the user have manually invoked
            model.initialize()
            Default is True.

        write_scaled_result --
            Set this parameter to True to write the result to file without
            taking scaling into account. If the value of scaled is False,
            then the variable scaling factors of the model are used to
            reproduced the unscaled variable values.
            Default: False

        result_file_name --
            Specifies the name of the file where the simulation result is
            written. Setting this option to an empty string results in a default
            file name that is based on the name of the model class.
            Default: Empty string

        with_jacobian --
            Set to True if an FMU Jacobian for the ODE is available or
            False otherwise.
            Default: False

        logging --
            If True, creates a logfile from the solver in the current
            directory.
            Default: False

        result_handling --
            Specifies how the result should be handled. Either stored to
            file or stored in memory. One can also use a custom handler.
            Available options: "file", "memory", "custom"
            Default: "file"

        result_handler --
            The handler for the result. Depending on the option in
            result_handling this either defaults to ResultHandlerFile
            or ResultHandlerMemory. If result_handling custom is choosen
            This MUST be provided.
            Default: None

        filter --
            A filter for choosing which variables to actually store
            result for. The syntax can be found in
            http://en.wikipedia.org/wiki/Glob_%28programming%29 . An
            example is filter = "*der" , stor all variables ending with
            'der'. Can also be a list.
            Default: None


    The different solvers provided by the Assimulo simulation package provides
    different options. These options are given in dictionaries with names
    consisting of the solver name concatenated by the string '_option'. The most
    common solver options are documented below, for a complete list of options
    see, http://www.jmodelica.org/assimulo

    Options for CVode::

        rtol    --
            The relative tolerance. The relative tolerance are retrieved from
            the 'default experiment' section in the XML-file and if not
            found are set to 1.0e-4
            Default: "Default" (1.0e-4)

        atol    --
            The absolute tolerance.
            Default: "Default" (rtol*0.01*(nominal values of the continuous states))

        discr   --
            The discretization method. Can be either 'BDF' or 'Adams'
            Default: 'BDF'

        iter    --
            The iteration method. Can be either 'Newton' or 'FixedPoint'
            Default: 'Newton'
    """
    def __init__(self, *args, **kw):
        _defaults= {
            'solver': 'CVode',
            'ncp':0,
            'initialize':True,
            'sensitivities':None,
            'write_scaled_result':False,
            'result_file_name':'',
            'with_jacobian':False,
            'logging':False,
            'result_handling':"file",
            'result_handler': None,
            'filter':None,
            'CVode_options':{'discr':'BDF','iter':'Newton',
                             'atol':"Default",'rtol':"Default",},
            'Radau5_options':{'atol':"Default",'rtol':"Default"}
            }
        super(AssimuloFMIAlgOptions,self).__init__(_defaults)
        # for those key-value-sets where the value is a dict, don't
        # overwrite the whole dict but instead update the default dict
        # with the new values
        self._update_keep_dict_defaults(*args, **kw)



class AssimuloFMIAlg_deprecated(AlgorithmBase):
    """
    Simulation algortihm for FMUs using the Assimulo package.
    """

    def __init__(self,
                 start_time,
                 final_time,
                 input,
                 model,
                 options):
        """
        Create a simulation algorithm using Assimulo.

        Parameters::

            model --
                fmi.FMUModel object representation of the model.

            options --
                The options that should be used in the algorithm. For details on
                the options, see:

                * model.simulate_options('AssimuloFMIAlgOptions')

                or look at the docstring with help:

                * help(pyfmi.fmi_algorithm_drivers.AssimuloFMIAlgOptions)

                Valid values are:
                - A dict that overrides some or all of the default values
                  provided by AssimuloFMIAlgOptions. An empty dict will thus
                  give all options with default values.
                - AssimuloFMIAlgOptions object.
        """
        self.model = model

        if not assimulo_present:
            raise Exception(
                'Could not find Assimulo package. Check pyfmi.check_packages()')

        # set start time, final time and input trajectory
        self.start_time = start_time
        self.final_time = final_time
        self.input = input
        self.model.time = start_time #Also set start time into the model

        # handle options argument
        if isinstance(options, dict) and not \
            isinstance(options, AssimuloFMIAlgOptions):
            # user has passed dict with options or empty dict = default
            self.options = AssimuloFMIAlgOptions(options)
        elif isinstance(options, AssimuloFMIAlgOptions):
            # user has passed AssimuloFMIAlgOptions instance
            self.options = options
        else:
            raise InvalidAlgorithmOptionException(options)

        # set options
        self._set_options()

        input_traj = None
        if self.input:
            if hasattr(self.input[1],"__call__"):
                input_traj=(self.input[0],
                        TrajectoryUserFunction(self.input[1]))
            else:
                input_traj=(self.input[0],
                        TrajectoryLinearInterpolation(self.input[1][:,0],
                                                      self.input[1][:,1:]))
            #Sets the inputs, if any
            self.model.set(input_traj[0], input_traj[1].eval(self.start_time)[0,:])

        # Initialize?
        if self.options['initialize']:
            try:
                self.model.initialize(relativeTolerance=self.solver_options['rtol'])
            except KeyError:
                rtol, atol = self.model.get_tolerances()
                self.model.initialize(relativeTolerance=rtol)

        # Sensitivities?
        if self.options["sensitivities"]:
            if self.options["solver"] != "CVode":
                raise Exception("Sensitivity simulations currently only supported using the solver CVode.")

                #Checks to see if all the sensitivities are inside the model
                #else there will be an exception
                self.model.get(self.options["sensitivities"])


        if not self.input:
            if self.options["sensitivities"]:
                self.probl = FMIODESENS(self.model, result_file_name=self.result_file_name,with_jacobian=self.with_jacobian,start_time=self.start_time,parameters=self.options["sensitivities"],logging=self.options["logging"])
            else:
                self.probl = FMIODE_deprecated(self.model, result_file_name=self.result_file_name,with_jacobian=self.with_jacobian,start_time=self.start_time,logging=self.options["logging"])
        else:
            if self.options["sensitivities"]:
                self.probl = FMIODESENS(
                self.model, input_traj, result_file_name=self.result_file_name,with_jacobian=self.with_jacobian,start_time=self.start_time,parameters=self.options["sensitivities"],logging=self.options["logging"])
            else:
                self.probl = FMIODE_deprecated(
                self.model, input_traj, result_file_name=self.result_file_name,with_jacobian=self.with_jacobian,start_time=self.start_time,logging=self.options["logging"])

        # instantiate solver and set options
        self.simulator = self.solver(self.probl)
        self._set_solver_options()

    def _set_options(self):
        """
        Helper function that sets options for AssimuloFMI algorithm.
        """
        # no of communication points
        self.ncp = self.options['ncp']

        self.write_scaled_result = self.options['write_scaled_result']

        self.with_jacobian = self.options['with_jacobian']

        # result file name
        if self.options['result_file_name'] == '':
            self.result_file_name = self.model.get_identifier()+'_result.txt'
        else:
            self.result_file_name = self.options['result_file_name']

        # solver
        solver = self.options['solver']
        if hasattr(solvers, solver):
            self.solver = getattr(solvers, solver)
        else:
            raise InvalidAlgorithmOptionException(
                "The solver: "+solver+ " is unknown.")

        # solver options
        try:
            self.solver_options = self.options[solver+'_options']
        except KeyError: #Default solver options not found
            self.solver_options = {} #Empty dict
            try:
                self.solver.atol
                self.solver_options["atol"] = "Default"
            except AttributeError:
                pass
            try:
                self.solver.rtol
                self.solver_options["rtol"] = "Default"
            except AttributeError:
                pass

        #Check relative tolerance
        #If the tolerances are not set specifically, they are set
        #according to the 'DefaultExperiment' from the XML file.
        try:
            if self.solver_options["rtol"] == "Default":
                rtol, atol = self.model.get_tolerances()
                self.solver_options['rtol'] = rtol
        except KeyError:
            pass

        #Check absolute tolerance
        try:
            if self.solver_options["atol"] == "Default":
                rtol, atol = self.model.get_tolerances()
                fnbr, gnbr = self.model.get_ode_sizes()
                if fnbr == 0:
                    self.solver_options['atol'] = 0.01*rtol
                else:
                    self.solver_options['atol'] = atol
        except KeyError:
            pass

    def _set_solver_options(self):
        """
        Helper function that sets options for the solver.
        """
        solver_options = self.solver_options.copy()

        #Set solver option continuous_output
        self.simulator.continuous_output = True

        #loop solver_args and set properties of solver
        for k, v in solver_options.iteritems():
            try:
                getattr(self.simulator,k)
            except AttributeError:
                try:
                    getattr(self.probl,k)
                except AttributeError:
                    raise InvalidSolverArgumentException(k)
                setattr(self.probl, k, v)
                continue
            setattr(self.simulator, k, v)

    def solve(self):
        """
        Runs the simulation.
        """
        self.simulator.simulate(self.final_time, self.ncp)

    def get_result(self):
        """
        Write result to file, load result data and create an AssimuloSimResult
        object.

        Returns::

            The AssimuloSimResult object.
        """
        if not self.simulator.continuous_output:
            write_data(self.simulator,self.write_scaled_result, self.result_file_name)
        # load result file
        res = ResultDymolaTextual(self.result_file_name)
        # create and return result object
        return FMIResult(self.model, self.result_file_name, self.simulator,
            res, self.options)

    @classmethod
    def get_default_options(cls):
        """
        Get an instance of the options class for the AssimuloFMIAlg algorithm,
        prefilled with default values. (Class method.)
        """
        return AssimuloFMIAlgOptions()



class AssimuloFMIAlg(AlgorithmBase):
    """
    Simulation algortihm for FMUs using the Assimulo package.
    """

    def __init__(self,
                 start_time,
                 final_time,
                 input,
                 model,
                 options):
        """
        Create a simulation algorithm using Assimulo.

        Parameters::

            model --
                fmi.FMUModel object representation of the model.

            options --
                The options that should be used in the algorithm. For details on
                the options, see:

                * model.simulate_options('AssimuloFMIAlgOptions')

                or look at the docstring with help:

                * help(pyfmi.fmi_algorithm_drivers.AssimuloFMIAlgOptions)

                Valid values are:
                - A dict that overrides some or all of the default values
                  provided by AssimuloFMIAlgOptions. An empty dict will thus
                  give all options with default values.
                - AssimuloFMIAlgOptions object.
        """
        self.model = model

        if not assimulo_present:
            raise Exception(
                'Could not find Assimulo package. Check pyfmi.check_packages()')

        # set start time, final time and input trajectory
        self.start_time = start_time
        self.final_time = final_time
        self.input = input
        self.model.time = start_time #Also set start time into the model

        # handle options argument
        if isinstance(options, dict) and not \
            isinstance(options, AssimuloFMIAlgOptions):
            # user has passed dict with options or empty dict = default
            self.options = AssimuloFMIAlgOptions(options)
        elif isinstance(options, AssimuloFMIAlgOptions):
            # user has passed AssimuloFMIAlgOptions instance
            self.options = options
        else:
            raise InvalidAlgorithmOptionException(options)

        # set options
        self._set_options()

        input_traj = None
        if self.input:
            if hasattr(self.input[1],"__call__"):
                input_traj=(self.input[0],
                        TrajectoryUserFunction(self.input[1]))
            else:
                input_traj=(self.input[0],
                        TrajectoryLinearInterpolation(self.input[1][:,0],
                                                      self.input[1][:,1:]))
            #Sets the inputs, if any
            self.model.set(input_traj[0], input_traj[1].eval(self.start_time)[0,:])

        if self.options["result_handling"] == "file":
            self.result_handler = ResultHandlerFile(self.model)
        elif self.options["result_handling"] == "memory":
            self.result_handler = ResultHandlerMemory(self.model)
        elif self.options["result_handling"] == "custom":
            self.result_handler = self.options["result_handler"]
            if self.result_handler == None:
                raise Exception("The result handler needs to be specified when using a custom result handling.")
            if not isinstance(self.result_handler, ResultHandler):
                raise Exception("The result handler needs to be a subclass of ResultHandler.")
        else:
            raise Exception("Unknown option to result_handling.")

        self.result_handler.set_options(self.options)
        self.result_handler.simulation_start()

        # Initialize?
        if self.options['initialize']:
            try:
                self.model.initialize(relativeTolerance=self.solver_options['rtol'])
            except KeyError:
                rtol, atol = self.model.get_tolerances()
                self.model.initialize(relativeTolerance=rtol)

        self.result_handler.initialize_complete()

        # Sensitivities?
        if self.options["sensitivities"]:
            if self.model.get_generation_tool() != "JModelica.org":
                raise Exception("Sensitivity calculations only possible with JModelica.org generated FMUs")
                
            if self.options["solver"] != "CVode":
                raise Exception("Sensitivity simulations currently only supported using the solver CVode.")

                #Checks to see if all the sensitivities are inside the model
                #else there will be an exception
                self.model.get(self.options["sensitivities"])

        if not self.input and isinstance(self.model, fmi.FMUModelME2):
            if self.options["sensitivities"]:
                self.probl = FMIODESENS2(self.model, result_file_name=self.result_file_name, start_time=self.start_time, parameters=self.options["sensitivities"],logging=self.options["logging"], result_handler=self.result_handler)
            else:
                self.probl = FMIODE2(self.model, result_file_name=self.result_file_name, start_time=self.start_time,logging=self.options["logging"], result_handler=self.result_handler)
        elif isinstance(self.model, fmi.FMUModelME2):
            if self.options["sensitivities"]:
                self.probl = FMIODESENS2(
                self.model, input_traj, result_file_name=self.result_file_name, start_time=self.start_time,parameters=self.options["sensitivities"],logging=self.options["logging"], result_handler=self.result_handler)
            else:
                self.probl = FMIODE2(
                self.model, input_traj, result_file_name=self.result_file_name, start_time=self.start_time,logging=self.options["logging"], result_handler=self.result_handler)

        elif not self.input:
            if self.options["sensitivities"]:
                self.probl = FMIODESENS(self.model, result_file_name=self.result_file_name,with_jacobian=self.with_jacobian,start_time=self.start_time,parameters=self.options["sensitivities"],logging=self.options["logging"], result_handler=self.result_handler)
            else:
                self.probl = FMIODE(self.model, result_file_name=self.result_file_name,with_jacobian=self.with_jacobian,start_time=self.start_time,logging=self.options["logging"], result_handler=self.result_handler)
        else:
            if self.options["sensitivities"]:
                self.probl = FMIODESENS(
                self.model, input_traj, result_file_name=self.result_file_name,with_jacobian=self.with_jacobian,start_time=self.start_time,parameters=self.options["sensitivities"],logging=self.options["logging"], result_handler=self.result_handler)
            else:
                self.probl = FMIODE(
                self.model, input_traj, result_file_name=self.result_file_name,with_jacobian=self.with_jacobian,start_time=self.start_time,logging=self.options["logging"], result_handler=self.result_handler)

        # instantiate solver and set options
        self.simulator = self.solver(self.probl)
        self._set_solver_options()

    def _set_options(self):
        """
        Helper function that sets options for AssimuloFMI algorithm.
        """
        # no of communication points
        self.ncp = self.options['ncp']

        self.write_scaled_result = self.options['write_scaled_result']

        self.with_jacobian = self.options['with_jacobian']

        # result file name
        if self.options['result_file_name'] == '':
            self.result_file_name = self.model.get_identifier()+'_result.txt'
        else:
            self.result_file_name = self.options['result_file_name']

        # solver
        solver = self.options['solver']
        if hasattr(solvers, solver):
            self.solver = getattr(solvers, solver)
        else:
            raise InvalidAlgorithmOptionException(
                "The solver: "+solver+ " is unknown.")

        # solver options
        try:
            self.solver_options = self.options[solver+'_options']
        except KeyError: #Default solver options not found
            self.solver_options = {} #Empty dict
            try:
                self.solver.atol
                self.solver_options["atol"] = "Default"
            except AttributeError:
                pass
            try:
                self.solver.rtol
                self.solver_options["rtol"] = "Default"
            except AttributeError:
                pass

        #Check relative tolerance
        #If the tolerances are not set specifically, they are set
        #according to the 'DefaultExperiment' from the XML file.
        try:
            if self.solver_options["rtol"] == "Default":
                rtol, atol = self.model.get_tolerances()
                self.solver_options['rtol'] = rtol
        except KeyError:
            pass

        #Check absolute tolerance
        try:
            if self.solver_options["atol"] == "Default":
                rtol, atol = self.model.get_tolerances()
                fnbr, gnbr = self.model.get_ode_sizes()
                if fnbr == 0:
                    self.solver_options['atol'] = 0.01*rtol
                else:
                    self.solver_options['atol'] = atol
        except KeyError:
            pass

    def _set_solver_options(self):
        """
        Helper function that sets options for the solver.
        """
        solver_options = self.solver_options.copy()

        #Set solver option continuous_output
        self.simulator.continuous_output = True

        #loop solver_args and set properties of solver
        for k, v in solver_options.iteritems():
            try:
                getattr(self.simulator,k)
            except AttributeError:
                try:
                    getattr(self.probl,k)
                except AttributeError:
                    raise InvalidSolverArgumentException(k)
                setattr(self.probl, k, v)
                continue
            setattr(self.simulator, k, v)

    def solve(self):
        """
        Runs the simulation.
        """
        self.simulator.simulate(self.final_time, self.ncp)

    def get_result(self):
        """
        Write result to file, load result data and create an AssimuloSimResult
        object.

        Returns::

            The AssimuloSimResult object.
        """
        # load result file
        res = self.result_handler.get_result()
        # create and return result object
        return FMIResult(self.model, self.result_file_name, self.simulator,
            res, self.options)

    @classmethod
    def get_default_options(cls):
        """
        Get an instance of the options class for the AssimuloFMIAlg algorithm,
        prefilled with default values. (Class method.)
        """
        return AssimuloFMIAlgOptions()


class FMICSAlgOptions(OptionBase):
    """
    Options for the solving the CS FMU.

    Options::


        ncp    --
            Number of communication points.
            Default: '500'

        initialize --
            If set to True, the initializing algorithm defined in the FMU model
            is invoked, otherwise it is assumed the user have manually invoked
            model.initialize()
            Default is True.

        write_scaled_result --
            Set this parameter to True to write the result to file without
            taking scaling into account. If the value of scaled is False,
            then the variable scaling factors of the model are used to
            reproduced the unscaled variable values.
            Default: False

        result_file_name --
            Specifies the name of the file where the simulation result is
            written. Setting this option to an empty string results in a default
            file name that is based on the name of the model class.
            Default: Empty string

        result_handling --
            Specifies how the result should be handled. Either stored to
            file or stored in memory. One can also use a custom handler.
            Available options: "file", "memory", "custom"
            Default: "file"

        result_handler --
            The handler for the result. Depending on the option in
            result_handling this either defaults to ResultHandlerFile
            or ResultHandlerMemory. If result_handling custom is choosen
            This MUST be provided.
            Default: None

        filter --
            A filter for choosing which variables to actually store
            result for. The syntax can be found in
            http://en.wikipedia.org/wiki/Glob_%28programming%29 . An
            example is filter = "*der" , stor all variables ending with
            'der'. Can also be a list.
            Default: None


    """
    def __init__(self, *args, **kw):
        _defaults= {
            'ncp':500,
            'initialize':True,
            'write_scaled_result':False,
            'result_file_name':'',
            'result_handling':"file",
            'result_handler': None,
            'filter':None,
            }
        super(FMICSAlgOptions,self).__init__(_defaults)
        # for those key-value-sets where the value is a dict, don't
        # overwrite the whole dict but instead update the default dict
        # with the new values
        self._update_keep_dict_defaults(*args, **kw)

class FMICSAlg(AlgorithmBase):
    """
    Simulation algortihm for FMUs (Co-simulation).
    """

    def __init__(self,
                 start_time,
                 final_time,
                 input,
                 model,
                 options):
        """
        Simulation algortihm for FMUs (Co-simulation).

        Parameters::

            model --
                fmi.FMUModelCS1 object representation of the model.

            options --
                The options that should be used in the algorithm. For details on
                the options, see:

                * model.simulate_options('FMICSAlgOptions')

                or look at the docstring with help:

                * help(pyfmi.fmi_algorithm_drivers.FMICSAlgOptions)

                Valid values are:
                - A dict that overrides some or all of the default values
                  provided by FMICSAlgOptions. An empty dict will thus
                  give all options with default values.
                - FMICSAlgOptions object.
        """
        self.model = model

        # set start time, final time and input trajectory
        self.start_time = start_time
        self.final_time = final_time
        self.input = input

        # handle options argument
        if isinstance(options, dict) and not \
            isinstance(options, FMICSAlgOptions):
            # user has passed dict with options or empty dict = default
            self.options = FMICSAlgOptions(options)
        elif isinstance(options, FMICSAlgOptions):
            # user has passed FMICSAlgOptions instance
            self.options = options
        else:
            raise InvalidAlgorithmOptionException(options)

        # set options
        self._set_options()

        input_traj = None
        if self.input:
            if hasattr(self.input[1],"__call__"):
                input_traj=(self.input[0],
                        TrajectoryUserFunction(self.input[1]))
            else:
                input_traj=(self.input[0],
                        TrajectoryLinearInterpolation(self.input[1][:,0],
                                                      self.input[1][:,1:]))
            #Sets the inputs, if any
            self.model.set(input_traj[0], input_traj[1].eval(self.start_time)[0,:])
        self.input_traj = input_traj

        if self.options["result_handling"] == "file":
            self.result_handler = ResultHandlerFile(self.model)
        elif self.options["result_handling"] == "memory":
            self.result_handler = ResultHandlerMemory(self.model)
        elif self.options["result_handling"] == "custom":
            self.result_handler = self.options["result_handler"]
            if self.result_handler == None:
                raise Exception("The result handler needs to be specified when using a custom result handling.")
            if not isinstance(self.result_handler, ResultHandler):
                raise Exception("The result handler needs to be a subclass of ResultHandler.")
        else:
            raise Exception("Unknown option to result_handling.")

        self.result_handler.set_options(self.options)
        self.result_handler.simulation_start()

        # Initialize?
        if self.options['initialize']:
            self.model.initialize(start_time, final_time, StopTimeDefined=True)

        self.result_handler.initialize_complete()

    def _set_options(self):
        """
        Helper function that sets options for FMICS algorithm.
        """
        # no of communication points
        self.ncp = self.options['ncp']

        self.write_scaled_result = self.options['write_scaled_result']

        # result file name
        if self.options['result_file_name'] == '':
            self.result_file_name = self.model.get_identifier()+'_result.txt'
        else:
            self.result_file_name = self.options['result_file_name']

    def _set_solver_options(self):
        """
        Helper function that sets options for the solver.
        """
        pass #No solver options

    def solve(self):
        """
        Runs the simulation.
        """
        result_handler = self.result_handler
        h = (self.final_time-self.start_time)/self.ncp
        grid = N.linspace(self.start_time,self.final_time,self.ncp+1)[:-1]

        status = 0

        #For result writing
        result_handler.integration_point()

        #Start of simulation, start the clock
        time_start = time.clock()

        for t in grid:
            status = self.model.do_step(t,h)

            if status != 0:
                result_handler.simulation_end()
                raise Exception("The simulation failed. See the log for more information. Return flag %d"%status)

            result_handler.integration_point()

            if self.input_traj != None:
                self.model.set(self.input_traj[0], self.input_traj[1].eval(t+h)[0,:])

        #End of simulation, stop the clock
        time_stop = time.clock()

        result_handler.simulation_end()

        #Log elapsed time
        print 'Simulation interval    : ' + str(self.start_time) + ' - ' + str(self.final_time) + ' seconds.'
        print 'Elapsed simulation time: ' + str(time_stop-time_start) + ' seconds.'

    def get_result(self):
        """
        Write result to file, load result data and create an FMICSResult
        object.

        Returns::

            The FMICSResult object.
        """
        # Get the result
        res = self.result_handler.get_result()

        # create and return result object
        return FMIResult(self.model, self.result_file_name, None,
            res, self.options)

    @classmethod
    def get_default_options(cls):
        """
        Get an instance of the options class for the FMICSAlg algorithm,
        prefilled with default values. (Class method.)
        """
        return FMICSAlgOptions()


