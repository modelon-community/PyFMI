#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2010-2024 Modelon AB
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
Module for writing optimization and simulation results to file.
"""
from operator import itemgetter, attrgetter
import codecs
import re
import sys
import os
import logging as logging_module
from functools import reduce
from typing import Union
from shutil import disk_usage

import numpy as np
import scipy
scipy_minmaj = tuple(map(int, scipy.__version__.split('.')[:2]))
if scipy_minmaj >= (1, 8):
    # due to a DeprecationWarning, we do below, this will likely need another change in a future update of scipy.
    from scipy.io.matlab._mio4 import MatFile4Reader, VarReader4, convert_dtypes, mdtypes_template, mxSPARSE_CLASS
else:
    from scipy.io.matlab.mio4 import MatFile4Reader, VarReader4, convert_dtypes, mdtypes_template, mxSPARSE_CLASS

import pyfmi.fmi as fmi # TODO
import pyfmi.fmi_util as fmi_util
from pyfmi.fmi3 import FMUModelBase3, FMI3_Type
from pyfmi.common import encode, decode
from pyfmi.common.diagnostics import DIAGNOSTICS_PREFIX, DiagnosticsBase
from pyfmi.exceptions import FMUException

SYS_LITTLE_ENDIAN = sys.byteorder == 'little'
NCP_LARGE = 5000
MB_FACTOR = 1e6 # 1024**2 = MiB
GB_FACTOR = 1e9 # 1024**3 = GiB
DISK_PROTECTION = 50*MB_FACTOR # 50MB

def get_available_disk_space(local_path):
    """
    Get the available disk space at a given local path. If the local path does not exists, it returns a very large
    value (max of size_t). Furthermore, 50mb is subtracted from the available free space and the reason for doing
    so is that running out of disk space can lead to any number of bad things and by making sure that there is at
    least some available disk left will lead to a controlled environment where the user can take appropriate
    actions.
    """
    try:
        free_space = disk_usage(local_path).free - DISK_PROTECTION

    #If the user is saving to a stream or something else than a file - catch it here and return a large value.
    except (FileNotFoundError, TypeError):
        free_space = sys.maxsize
    return free_space

class Trajectory:
    """
    Class for storing a time series.
    """

    def __init__(self,t,x):
        """
        Constructor for the Trajectory class.

        Parameters::

            t --
                Abscissa of the trajectory.
            x --
                The ordinate of the trajectory.
        """
        self.t = t
        self.x = x

class ResultStorage:
    pass

    def get_variable_data(self, key):
        raise NotImplementedError

    def is_variable(self, var):
        raise NotImplementedError

    def is_negated(self, var):
        raise NotImplementedError

class ResultHandler:
    def __init__(self, model = None):
        self.model = model
        # Dictionary of support capabilities
        # should be UPDATED, not REPLACED, for stating supported capabilities
        self.supports = {"dynamic_diagnostics": False,
                         "result_max_size": False}
        self.is_fmi3 = isinstance(model, FMUModelBase3)

    def simulation_start(self, diagnostics_params = {}, diagnostics_vars = {}):
        """
        This method is called before the simulation has started and
        after the initialization call for the FMU.
        This function also takes two keyword arguments 'diagnostics_params'
        and 'diagnostics_vars' which are dicts containing information about what
        diagnostic parameters and variables to generate results for.
        """
        pass

    def initialize_complete(self):
        """
        This method is called after the initialization method of the FMU
        has been been called.
        """
        pass

    def integration_point(self, solver = None):
        """
        This method is called for each time-point for which result are
        to be stored as indicated by the "number of communication points"
        provided to the simulation method.
        """
        pass

    def diagnostics_point(self, diag_data = None):
        """
        Generates a data point for diagnostics data.
        """
        if self.supports.get('dynamic_diagnostics', False):
            raise NotImplementedError

    def simulation_end(self):
        """
        This method is called at the end of a simulation.
        """
        pass

    def set_options(self, options = None):
        """
        Options are the options dictionary provided to the simulation
        method.
        """
        self.options = options

    def get_result(self):
        """
        Method for retrieving the result. This method should return a
        result of an instance of ResultStorage or of an instance of a
        subclass of ResultStorage.
        """
        raise NotImplementedError

class ResultDymola:
    """
    Base class for representation of a result file.
    """

    def _get_name(self):
        return [decode(n) for n in self.name_lookup.keys()]

    name = property(fget = _get_name)

    def get_variable_index(self,name):
        """
        Retrieve the index in the name vector of a given variable.

        Parameters::

            name --
                Name of variable.

        Returns::

            In integer index.
        """
        #Strip name of spaces, for instance a[2, 1] to a[2,1]
        name = name.replace(" ", "")

        try:
            if isinstance(self, ResultDymolaBinary):
                return self.name_lookup[encode(name)]
            else:
                return self.name_lookup[name]
        except KeyError:
            #Variable was not found so check if it was a derivative variable
            #and check if there exists a variable with another naming
            #convention
            if self._check_if_derivative_variable(name):
                try:
                    #First do a simple search for the other naming convention
                    if isinstance(self, ResultDymolaBinary):
                        return self.name_lookup[encode(self._convert_dx_name(name))]
                    else:
                        return self.name_lookup[self._convert_dx_name(name)]
                except KeyError:
                    return self._exhaustive_search_for_derivatives(name)
            else:
                raise VariableNotFoundError("Cannot find variable " +
                                        name + " in data file.")

    def _correct_index_for_alias(self, data_index: int) -> tuple[int, int]:
        """ This function returns a correction for 'data_index' for alias variables.
            Negative values for data_index implies that it corresponds to an alias variable.
            The variable which it is aliased for is located in position |data_index| shifted by one.
        """
        factor = -1 if data_index < 0 else 1
        return factor*data_index - 1, factor

    def _check_if_derivative_variable(self, name: str) -> bool:
        """
        Check if a variable is a derivative variable or not.
        """
        return name.startswith("der(") or name.split(".")[-1].startswith("der(")

    def _exhaustive_search_for_derivatives(self, name):
        """
        Perform an exhaustive search for a derivative variable by
        first retrieving the underlying state and for each its alias
        check if there exists a derivative variable.
        """
        #Find alias for name
        state = self._find_underlying_state(name)
        index = self.get_variable_index(state)

        alias_index = np.where(self.dataInfo[:,1]==self.dataInfo[index,1])[0]

        #Loop through all alias
        for ind in alias_index:
            #Get the trial name
            trial_name = list(self.name_lookup.keys())[ind]

            #Create the derivative name
            if isinstance(self, ResultDymolaBinary):
                der_trial_name = self._create_derivative_from_state(decode(trial_name))
            else:
                der_trial_name = self._create_derivative_from_state(trial_name)

            try:
                if isinstance(self, ResultDymolaBinary):
                    return self.name_lookup[encode(der_trial_name)]
                else:
                    return self.name_lookup[der_trial_name]
            except KeyError:
                try:
                    if isinstance(self, ResultDymolaBinary):
                        return self.name_lookup[encode(self._convert_dx_name(der_trial_name))]
                    else:
                        return self.name_lookup[self._convert_dx_name(der_trial_name)]
                except KeyError:
                    pass
        else:
            raise VariableNotFoundError("Cannot find variable " +
                                        name + " in data file.")

    def _find_underlying_state(self, name):
        """
        Finds the underlying state of a derivative variable. der(PI.x)
        -> PI.x.
        """
        spl = name.split(".")

        if spl[0].startswith("der("):
            spl[0] = spl[0][4:] #Remove der(
            spl[-1] = spl[-1][:-1] #Remove )
            return ".".join(spl)
        elif spl[-1].startswith("der("):
            spl[-1] = spl[-1][4:] #Remove der(
            spl[-1] = spl[-1][:-1] #Remove )
            return ".".join(spl)
        else:
            return name

    def _create_derivative_from_state(self, name):
        """
        Create a derivative variable from a state by adding for instance
        to PI.x -> der(PI.x).
        """
        return "der("+name+")"

    def _convert_dx_name(self, name):
        """
        Internal method for converting the derivative variable into the
        "other" convention. A derivative variable can either be on the
        form PI.der(x) and der(PI.x).

        Returns the original name if the name was not a derivative name.
        """
        spl = name.split(".") #Split name

        if spl[0].startswith("der("): #der(PI.x)
            spl[0] = spl[0][4:] #Remove der
            spl[-1] = "der("+spl[-1] #Add der

            return ".".join(spl) #PI.der(x)

        elif spl[-1].startswith("der("): #PI.der(x)
            spl[0] = "der("+spl[0] #Add der
            spl[-1] = spl[-1][4:] #Remove der

            return ".".join(spl)

        else: #Variable was not a derivative variable
            return name

class ResultCSVTextual:
    """ Class representing a simulation or optimization result loaded from a CSV-file. """
    def __init__(self, filename, delimiter=";"):
        """
        Load a result file written on CSV format.

        Parameters::

            filename --
                Name of file or stream object which the result is written to.
                If filename is a stream object, it needs to support 'readline' and 'seek'.

            delimiter --
                The delimiter the data values is separated by.
                Default: ";""
        """

        if isinstance(filename, str):
            fid = codecs.open(filename,'r','utf-8')
        else: #assume stream
            if not(hasattr(filename, 'readline') and hasattr(filename, 'seek')):
                raise JIOError("Given stream needs to support 'readline' and 'seek' in order to retrieve the results.")
            fid = filename
            fid.seek(0, 0) # need to start reading from beginning

        if delimiter == ";":
            name = fid.readline().strip().split(delimiter)
        elif delimiter == ",":
            name = [s[1:-1] for s in re.findall('".+?"', fid.readline().strip())]
        else:
            raise JIOError('Unsupported separator.')
        self._name = name

        self.data_matrix = {}
        for i,n in enumerate(name):
            self.data_matrix[n] = i

        data = []
        while True:
            row = fid.readline().strip().split(delimiter)

            if row[-1] == "" or row[-1] == "\n":
                break

            data.append([float(d) for d in row])

        self.data = np.array(data)

    def get_variable_data(self,name):
        """
        Retrieve the data sequence for a variable with a given name.

        Parameters::

            name --
                Name of the variable.

        Returns::

            A Trajectory object containing the time vector and the data vector
            of the variable.
        """
        if name == 'time':
            return Trajectory(self.data[:,0],self.data[:,0])
        else:
            return Trajectory(self.data[:,0],self.data[:,self.data_matrix[name]])

    def is_variable(self, name):
        return True

    def is_negated(self, name):
        return False

    def get_data_matrix(self):
        """
        Returns the result matrix.

        Returns::

            The result data matrix.
        """
        return self.data

class ResultWriter():
    """
    Base class for writing results to file.
    """

    def write_header(self):
        """
        The header is intended to be used for writing general information about
        the model. This is intended to be called once.
        """
        pass

    def write_point(self):
        """
        This method does the writing of the actual result.
        """
        pass

    def write_finalize(self):
        """
        The finalize method can be used to for instance close the file.
        """
        pass

class ResultWriterDymola(ResultWriter):
    """
    Export an optimization or simulation result to file in Dymola's result file
    format.
    """
    def __init__(self, model, format='txt'):
        """
        Export an optimization or simulation result to file in Dymolas result
        file format.

        Parameters::

            model --
                A FMIModel object.

            format --
                A text string equal either to 'txt' for textual format or 'mat'
                for binary Matlab format.
                Default: 'txt'

        Limitations::

            Currently only textual format is supported.
        """
        self.model = model

        if format!='txt':
            raise JIOError('The format is currently not supported.')

        #Internal values
        self._file_open = False
        self._npoints = 0


    def write_header(self, file_name='', parameters=None):
        """
        Opens the file and writes the header. This includes the information
        about the variables and a table determining the link between variables
        and data.

        Parameters::

            file_name --
                If no file name is given, the name of the model (as defined by
                FMUModel*.get_identifier()) concatenated with the string '_result' is
                used. A file suffix equal to the format argument is then
                appended to the file name.
                Default: Empty string.
        """
        if file_name=='':
            file_name=self.model.get_identifier() + '_result.txt'

        # Open file
        f = codecs.open(file_name,'w','utf-8')
        self._file_open = True

        # Write header
        f.write('#1\n')
        f.write('char Aclass(3,11)\n')
        f.write('Atrajectory\n')
        f.write('1.1\n')
        f.write('\n')

        # all lists that we need for later
        vrefs_alias = []
        vrefs_noalias = []
        vrefs = []
        names_alias = []
        names_noalias = []
        names = []
        aliases_alias = []
        aliases = []
        descriptions_alias = []
        descriptions = []
        variabilities_alias = []
        variabilities_noalias = []
        variabilities = []
        types_alias = []
        types_noalias = []
        types = []

        for var in self.model.get_model_variables().values():
            if not var.type == fmi.FMI_STRING and not var.type == fmi.FMI_ENUMERATION:
                    if var.alias == fmi.FMI_NO_ALIAS:
                        vrefs_noalias.append(var.value_reference)
                        names_noalias.append(var.name)
                        aliases.append(var.alias)
                        descriptions.append(var.description)
                        variabilities_noalias.append(var.variability)
                        types_noalias.append(var.type)
                    else:
                        vrefs_alias.append(var.value_reference)
                        names_alias.append(var.name)
                        aliases_alias.append(var.alias)
                        descriptions_alias.append(var.description)
                        variabilities_alias.append(var.variability)
                        types_alias.append(var.type)

        # need to save these no alias lists for later
        vrefs = vrefs_noalias[:]
        names = names_noalias[:]
        types = types_noalias[:]
        variabilities = variabilities_noalias[:]

        # merge lists
        vrefs.extend(vrefs_alias)
        names.extend(names_alias)
        aliases.extend(aliases_alias)
        descriptions.extend(descriptions_alias)
        variabilities.extend(variabilities_alias)
        types.extend(types_alias)

        # zip to list of tuples and sort - non alias variables are now
        # guaranteed to be first in list
        names_noalias = sorted(zip(
            tuple(vrefs_noalias),
            tuple(names_noalias)),
            key=itemgetter(0))
        variabilities_noalias = sorted(zip(
            tuple(vrefs_noalias),
            tuple(variabilities_noalias)),
            key=itemgetter(0))
        types_noalias = sorted(zip(
            tuple(vrefs_noalias),
            tuple(types_noalias)),
            key=itemgetter(0))
        names = sorted(zip(
            tuple(vrefs),
            tuple(names)),
            key=itemgetter(0))
        aliases = sorted(zip(
            tuple(vrefs),
            tuple(aliases)),
            key=itemgetter(0))
        descriptions = sorted(zip(
            tuple(vrefs),
            tuple(descriptions)),
            key=itemgetter(0))
        variabilities = sorted(zip(
            tuple(vrefs),
            tuple(variabilities)),
            key=itemgetter(0))
        types = sorted(zip(
            tuple(vrefs),
            tuple(types)),
            key=itemgetter(0))

        num_vars = len(names)

        names_sens = []
        descs_sens = []
        cont_vars = []

        if parameters is not None:
            vars = self.model.get_model_variables(type=0,include_alias=False,variability=3)
            for i in self.model.get_state_value_references():
                for j in vars.keys():
                    if i == vars[j].value_reference:
                        cont_vars.append(vars[j].name)

        if parameters is not None:
            for j in range(len(parameters)):
                for i in range(len(self.model.continuous_states)):
                    names_sens += ['d'+cont_vars[i]+'/d'+parameters[j]]
                    descs_sens  += ['Sensitivity of '+cont_vars[i]+' with respect to '+parameters[j]+'.']

        # Find the maximum name and description length
        max_name_length = len('Time')
        max_desc_length = len('Time in [s]')

        for i in range(len(names)):
            name = names[i][1]
            desc = descriptions[i][1]

            if (len(name)>max_name_length):
                max_name_length = len(name)

            if (len(desc)>max_desc_length):
                max_desc_length = len(desc)

        for i in range(len(names_sens)):
            name = names_sens[i]
            desc = descs_sens[i]

            if (len(name)>max_name_length):
                max_name_length = len(name)

            if (len(desc)>max_desc_length):
                max_desc_length = len(desc)

        f.write('char name(%d,%d)\n' % (num_vars+len(names_sens)+1, max_name_length))
        f.write('time\n')

        for name in names:
            f.write(name[1] +'\n')
        for name in names_sens:
            f.write(name + '\n')

        f.write('\n')

        # Write descriptions
        f.write('char description(%d,%d)\n' % (num_vars+len(names_sens) + 1, max_desc_length))
        f.write('Time in [s]\n')

        # Loop over all variables, not only those with a description
        for desc in descriptions:
            f.write(desc[1] +'\n')
        for desc in descs_sens:
            f.write(desc + '\n')

        f.write('\n')

        # Write data meta information

        f.write('int dataInfo(%d,%d)\n' % (num_vars+len(names_sens) + 1, 4))
        f.write('0 1 0 -1 # time\n')

        list_of_continuous_states = np.append(self.model._save_real_variables_val,
            self.model._save_int_variables_val)
        list_of_continuous_states = np.append(list_of_continuous_states,
            self.model._save_bool_variables_val).tolist()
        list_of_continuous_states = dict(zip(list_of_continuous_states,
            range(len(list_of_continuous_states))))
        valueref_of_continuous_states = []

        cnt_1 = 1
        cnt_2 = 1
        n_parameters = 0
        datatable1 = False
        for i, name in enumerate(names):
            if aliases[i][1] == 0: # no alias
                if variabilities[i][1] == fmi.FMI_PARAMETER or \
                    variabilities[i][1] == fmi.FMI_CONSTANT:
                    cnt_1 += 1
                    n_parameters += 1
                    f.write('1 %d 0 -1 # ' % cnt_1 + name[1]+'\n')
                    datatable1 = True
                else:
                    cnt_2 += 1
                    valueref_of_continuous_states.append(
                        list_of_continuous_states[name[0]])
                    f.write('2 %d 0 -1 # ' % cnt_2 + name[1] +'\n')
                    datatable1 = False

            elif aliases[i][1] == 1: # alias
                if datatable1:
                    f.write('1 %d 0 -1 # ' % cnt_1 + name[1]+'\n')
                else:
                    f.write('2 %d 0 -1 # ' % cnt_2 + name[1] +'\n')
            else:
                if datatable1:
                    f.write('1 -%d 0 -1 # ' % cnt_1 + name[1]+'\n')
                else:
                    f.write('2 -%d 0 -1 # ' % cnt_2 + name[1] +'\n')
        for i, name in enumerate(names_sens):
            cnt_2 += 1
            f.write('2 %d 0 -1 # ' % cnt_2 + name +'\n')


        f.write('\n')

        # Write data
        # Write data set 1
        f.write('float data_1(%d,%d)\n' % (2, n_parameters + 1))
        f.write("%.14E" % self.model.time)
        str_text = ''

        # write constants and parameters
        for i, name in enumerate(names_noalias):
            if variabilities_noalias[i][1] == fmi.FMI_CONSTANT or \
                variabilities_noalias[i][1] == fmi.FMI_PARAMETER:
                    if types_noalias[i][1] == fmi.FMI_REAL:
                        str_text = str_text + (
                            " %.14E" % (self.model.get_real([name[0]])))
                    elif types_noalias[i][1] == fmi.FMI_INTEGER:
                        str_text = str_text + (
                            " %.14E" % (self.model.get_integer([name[0]])))
                    elif types_noalias[i][1] == fmi.FMI_BOOLEAN:
                        str_text = str_text + (
                            " %.14E" % (float(
                                self.model.get_boolean([name[0]])[0])))

        f.write(str_text)
        f.write('\n')
        self._point_last_t = f.tell()
        f.write("%s" % ' '*28)
        f.write(str_text)

        f.write('\n\n')

        self._nvariables = len(valueref_of_continuous_states)+1
        self._nvariables_sens = len(names_sens)


        f.write('float data_2(')
        self._point_npoints = f.tell()
        f.write(' '*(14+4+14))
        f.write('\n')

        #f.write('%s,%d)\n' % (' '*14, self._nvariables))

        self._file = f
        self._data_order = valueref_of_continuous_states

    def write_point(self, data=None, parameter_data=[]):
        """
        Writes the current status of the model to file. If the header has not
        been written previously it is written now. If data is specified it is
        written instead of the current status.

        Parameters::

                data --
                    A one dimensional array of variable trajectory data. data
                    should consist of information about the status in the order
                    specified by FMUModel*.save_time_point()
                    Default: None
        """
        f = self._file
        data_order = self._data_order

        #If data is none, store the current point from the model
        if data is None:
            #Retrieves the time-point
            [r,i,b] = self.model.save_time_point()
            data = np.append(np.append(np.append(self.model.time,r),i),b)

        #Write the point
        str_text = (" %.14E" % data[0])
        for j in range(self._nvariables-1):
            str_text = str_text + (" %.14E" % (data[1+data_order[j]]))
        for j in range(len(parameter_data)):
            str_text = str_text + (" %.14E" % (parameter_data[j]))
        f.write(str_text+'\n')

        #Update number of points
        self._npoints+=1

    def write_finalize(self):
        """
        Finalize the writing by filling in the blanks in the created file. The
        blanks consists of the number of points and the final time (in data set
        1). Also closes the file.
        """
        #If open, finalize and close
        if self._file_open:

            f = self._file

            f.seek(self._point_last_t)

            f.write('%.14E'%self.model.time)

            f.seek(self._point_npoints)
            f.write('%d,%d)' % (self._npoints, self._nvariables+self._nvariables_sens))
            #f.write('%d'%self._npoints)
            f.seek(-1,2)
            #Close the file
            f.write('\n')
            f.close()
            self._file_open = False


class ResultStorageMemory(ResultDymola):
    """
    Class representing a simulation result that is kept in MEMORY.
    """
    def __init__(self, model, data, vars_ref, vars):
        """
        Load result from the ResultHandlerMemory

        Parameters::

            model --
                Instance of the FMUModel*.
            data --
                The simulation data.
        """
        self.model = model
        self.vars = vars
        self._name = [var.name for var in vars.values()]
        self.data = {}
        self.data_matrix = data

        #time real integer boolean
        real_val_ref    = vars_ref[0]
        integer_val_ref = vars_ref[1]
        boolean_val_ref = vars_ref[2]

        self.time = data[:,0]
        for i,ref in enumerate(real_val_ref+integer_val_ref+boolean_val_ref):
            self.data[ref] = data[:,i+1]

    def get_variable_data(self,name):
        """
        Retrieve the data sequence for a variable with a given name.

        Parameters::

            name --
                Name of the variable.

        Returns::

            A Trajectory object containing the time vector and the data vector
            of the variable.
        """
        if name == 'time':
            return Trajectory(self.time,self.time)
        else:
            try:
                var = self.vars[name]
            except KeyError:
                raise VariableNotFoundError("Cannot find variable " +
                                        name + " in data file.")

            factor = -1 if var.alias == fmi.FMI_NEGATED_ALIAS else 1

            if var.variability == fmi.FMI_CONSTANT or var.variability == fmi.FMI_PARAMETER:
                return Trajectory([self.time[0],self.time[-1]],np.array([self.model.get(name),self.model.get(name)]).ravel())
            else:
                return Trajectory(self.time,factor*self.data[var.value_reference])


    def is_variable(self, name):
        """
        Returns True if the given name corresponds to a time-varying variable.

        Parameters::

            name --
                Name of the variable/parameter/constant.

        Returns::

            True if the variable is time-varying.
        """
        if name == 'time':
            return True
        variability = self.vars[name].variability

        if variability ==  fmi.FMI_CONSTANT or variability == fmi.FMI_PARAMETER:
            return False
        else:
            return True

    def is_negated(self, name):
        """
        Returns True if the given name corresponds to a negated result vector.

        Parameters::

            name --
                Name of the variable/parameter/constant.

        Returns::

            True if the result should be negated
        """
        alias = self.vars[name].alias

        if alias == fmi.FMI_NEGATED_ALIAS:
            return True
        else:
            return False

    def get_column(self, name):
        """
        Returns the column number in the data matrix where the values of the
        variable are stored.

        Parameters::

            name --
                Name of the variable/parameter/constant.

        Returns::

            The column number.
        """
        raise NotImplementedError

    def get_data_matrix(self):
        """
        Returns the result matrix.

        Returns::

            The result data matrix.
        """
        return self.data_matrix

class ResultDymolaTextual(ResultDymola):
    """
    Class representing a simulation or optimization result loaded from a Dymola
    binary file.
    """
    def __init__(self,fname):
        """
        Load a result file written on Dymola textual format.

        Parameters::

            fname --
                Name of file or stream object which the result is written to.
                If fname is a stream, it needs to support 'readline' and 'seek'.
        """
        if isinstance(fname, str):
            fid = codecs.open(fname,'r','utf-8')
        else:
            if not (hasattr(fname, 'readline') and hasattr(fname, 'seek')):
                raise JIOError("Given stream needs to support 'readline' and 'seek' in order to retrieve the results.")
            fid = fname
            fid.seek(0,0) # Needs to start from beginning of file

        # Read Aclass section
        nLines = self._find_phrase(fid, 'char Aclass')

        nLines = int(nLines[0])
        self.Aclass = [fid.readline().strip() for i in range(nLines)]

        # Read name section
        nLines = self._find_phrase(fid, 'char name')

        nLines = int(nLines[0])
        self._name = [fid.readline().strip().replace(" ","") for i in range(nLines)]
        self.name_lookup = {key:ind for ind,key in enumerate(self._name)}

        # Read description section
        nLines = self._find_phrase(fid, 'char description')

        nLines = int(nLines[0])
        self.description = [fid.readline().strip() for i in range(nLines)]

        # Read dataInfo section
        nLines = self._find_phrase(fid, 'int dataInfo')

        nCols = nLines[2].partition(')')
        nLines = int(nLines[0])
        nCols = int(nCols[0])

        self.dataInfo = np.array([list(map(int,fid.readline().split()[0:nCols])) for i in range(nLines)])

        # Find out how many data matrices there are
        if len(self._name) == 1: #Only time
            nData = 2
        else:
            nData = max(self.dataInfo[:,0])

        self.data = []
        for i in range(0,nData):
            line = fid.readline()
            tmp = line.partition(' ')
            while tmp[0]!='float' and tmp[0]!='double' and line!='':
                line = fid.readline()
                tmp = line.partition(' ')
            if line=='':
                raise JIOError('The result does not seem to be of a supported format.')
            tmp = tmp[2].partition('(')
            nLines = tmp[2].partition(',')
            nCols = nLines[2].partition(')')
            nLines = int(nLines[0])
            nCols = int(nCols[0])
            data = []
            for i in range(0,nLines):
                info = []
                while len(info) < nCols and line != '':
                    line = fid.readline()
                    info.extend(line.split())
                try:
                    data.append(list(map(float,info[0:nCols])))
                except ValueError: #Handle 1.#INF's and such
                    data.append(list(map(robust_float,info[0:nCols])))
                if len(info) == 0 and i < nLines-1:
                    raise JIOError("Inconsistent number of lines in the result data.")
                del(info)
            self.data.append(np.array(data))

        if len(self.data) == 0:
            raise JIOError('Could not find any variable data in the result file.')

    def _find_phrase(self,fid, phrase):
        line = fid.readline()
        tmp = line.partition('(')
        while tmp[0]!=phrase and line!='':
            line = fid.readline()
            tmp = line.partition('(')
        if line=='':
            raise JIOError("The result does not seem to be of a supported format.")
        return tmp[2].partition(',')

    def get_variable_data(self,name):
        """
        Retrieve the data sequence for a variable with a given name.

        Parameters::

            name --
                Name of the variable.

        Returns::

            A Trajectory object containing the time vector and the data vector
            of the variable.
        """
        if name == 'time' or name== 'Time':
            variable_index = 0
        else:
            variable_index  = self.get_variable_index(name)

        data_index = self.dataInfo[variable_index][1]
        data_index, factor = self._correct_index_for_alias(data_index)
        data_mat = self.dataInfo[variable_index][0]-1

        if data_mat < 0:
            # Take into account that the 'Time' variable can be named either 'Time' or 'time'.

            data_mat = 1 if len(self.data) > 1 else 0

        return Trajectory(
            self.data[data_mat][:, 0],factor*self.data[data_mat][:, data_index])

    def is_variable(self, name):
        """
        Returns True if the given name corresponds to a time-varying variable.

        Parameters::

            name --
                Name of the variable/parameter/constant

        Returns::

            True if the variable is time-varying.
        """
        if name == 'time' or name== 'Time':
            return True
        variable_index  = self.get_variable_index(name)
        data_mat = self.dataInfo[variable_index][0]-1
        if data_mat<0:
            data_mat = 0

        if data_mat == 0:
            return False
        else:
            return True

    def is_negated(self, name):
        """
        Returns True if the given name corresponds to a negated result vector.

        Parameters::

            name --
                Name of the variable/parameter/constant.

        Returns::

            True if the result should be negated
        """
        variable_index  = self.get_variable_index(name)
        data_index = self.dataInfo[variable_index][1]
        if data_index<0:
            return True
        else:
            return False

    def get_column(self, name):
        """
        Returns the column number in the data matrix where the values of the
        variable are stored.

        Parameters::

            name --
                Name of the variable/parameter/constant.

        Returns::

            The column number.
        """
        if name == 'time' or name== 'Time':
            return 0

        if not self.is_variable(name):
            raise VariableNotTimeVarying("Variable " + name + " is not time-varying.")
        variable_index  = self.get_variable_index(name)
        data_index = self.dataInfo[variable_index][1]
        data_index, _ = self._correct_index_for_alias(data_index)

        return data_index

    def get_data_matrix(self):
        """
        Returns the result matrix.

        Returns::

            The result data matrix.
        """
        return self.data[1]

    def shift_time(self,time_shift):
        """
        Shift the time vector using a fixed offset.

        Parameters::
            time_shift --
                The time shift offset.
        """
        for i in range(len(self.data)):
            for j in range(np.shape(self.data[i])[0]):
                self.data[i][j,0] = self.data[i][j,0] + time_shift

    def append(self, res):
        """
        Append another simulation result. The time vector of the appended
        trajectories is shifted so that the appended trajectories appears
        after the original result trajectories.

        Parameters::
            res --
                A simulation result object of type DymolaResultTextual.
        """
        n_points = np.size(res.data[1],0)
        time_shift = self.data[1][-1,0]
        self.data[1] = np.vstack((self.data[1],res.data[1]))
        self.data[1][n_points:,0] = self.data[1][n_points:,0] + time_shift

#Overriding SCIPYs default reader for MATLAB v4 format
class DelayedVarReader4(VarReader4):
    def read_sub_array(self, hdr, copy=True):
        if hdr.name == b"data_2":
            ret = {"section": "data_2",
                   "file_position": self.mat_stream.tell(),
                   "sizeof_type": hdr.dtype.itemsize,
                   "nbr_points": hdr.dims[1],
                   "nbr_variables": hdr.dims[0]}
            return ret
        elif hdr.name == b"name":
            ret = {"section": "name",
                   "file_position": self.mat_stream.tell(),
                   "sizeof_type": hdr.dtype.itemsize,
                   "max_length": hdr.dims[0],
                   "nbr_variables": hdr.dims[1]}
            return ret
        else:
            arr = super(DelayedVarReader4, self).read_sub_array(hdr, copy)
            return arr

    def read_char_array(self, hdr):
        return self.read_sub_array(hdr)

#Need to hook in the variable reader above
class DelayedVariableLoad(MatFile4Reader):
    def initialize_read(self):
        self.dtypes = convert_dtypes(mdtypes_template, self.byte_order)
        self._matrix_reader = DelayedVarReader4(self)

    def read_var_header(self):
        """ This function overrides method 'read_var_header' from the scipy class scipy.io.matlab.mio4.MatFile4Reader.
            The reason is that we need to make sure the type of the elements
            in the tuple 'hdr.dims' are sufficient in order to carry out the multiplication
            to assign 'remaining_bytes' its value. For large files it can otherwise be an issue.
        """
        hdr = self._matrix_reader.read_header()
        hdr.dims = tuple(int(i) for i in hdr.dims)
        n = reduce(lambda x, y: x*y, hdr.dims, 1)  # fast product
        remaining_bytes = hdr.dtype.itemsize * n
        if hdr.is_complex and not hdr.mclass == mxSPARSE_CLASS:
            remaining_bytes *= 2
        next_position = self.mat_stream.tell() + remaining_bytes
        return hdr, next_position

class ResultDymolaBinary(ResultDymola):
    """
    Class representing a simulation or optimization result loaded from a Dymola
    binary file.
    """
    def __init__(self, fname, delayed_trajectory_loading = True, allow_file_updates=False):
        """
        Load a result file written on Dymola binary format.

        Parameters::

            fname --
                Name of file or a stream object supported by scipy.io.loadmat,
                which the result is written to.

            delayed_trajectory_loading --
                Determines if the trajectories are loaded on demand or
                all at the same time. Only works for files or streams that
                are based on a file and has attribute name.
                Default: True (for files)

            allow_file_updates --
                If this is True, file updates (in terms of more
                data points being added to the result file) is allowed.
                The number of variables stored in the file needs to be
                exactly the same and only the number of data points for
                the continuous variables are allowed to change.
                Default: False
        """

        if isinstance(fname, str):
            self._fname = fname
            self._is_stream = False
        elif hasattr(fname, "name") and os.path.isfile(fname.name):
            self._fname = fname.name
            self._is_stream = False
        else:
            self._fname = fname
            self._is_stream = True
            delayed_trajectory_loading = False
        self._allow_file_updates = allow_file_updates
        self._last_set_of_indices = (None, None) # used for dealing with cached data and partial trajectories

        data_sections = ["name", "dataInfo", "data_2", "data_3", "data_4"]
        if not self._is_stream:
            with open(self._fname, "rb") as f:
                delayed = DelayedVariableLoad(f, chars_as_strings=False)
                try:
                    self.raw = delayed.get_variables(variable_names = data_sections)
                    self._contains_diagnostic_data = True
                    self._data_3_info = self.raw["data_3"]
                    self._data_3 = {}
                    self._data_4_info = self.raw["data_4"]

                    self._has_file_pos_data = False

                except Exception:
                    self._contains_diagnostic_data = False
                    data_sections = ["name", "dataInfo", "data_2"]
        else:
            data_sections = ["name", "dataInfo", "data_2"]
            self._contains_diagnostic_data = False

        if delayed_trajectory_loading or self._contains_diagnostic_data:
            if not self._contains_diagnostic_data:
                with open(self._fname, "rb") as f:
                    delayed = DelayedVariableLoad(f, chars_as_strings=False)
                    self.raw = delayed.get_variables(variable_names = data_sections)

            self._data_2_info = self.raw["data_2"]
            self._data_2 = {}
            if self._contains_diagnostic_data:
                self._file_pos_model_var = np.empty(self._data_2_info["nbr_points"], dtype=np.longlong)
                self._file_pos_diag_var = np.empty(self._data_3_info.shape[0], dtype=np.longlong)
            self._name_info   = self.raw["name"]

            self.name_lookup = self._get_name_dict()
        else:
            self.raw = scipy.io.loadmat(self._fname,chars_as_strings=False, variable_names = data_sections)
            self._data_2 = self.raw["data_2"]

            name = self.raw['name']

            self._name = fmi_util.convert_array_names_list_names_int(name.view(np.int32))
            self.name_lookup = {key:ind for ind,key in enumerate(self._name)}

        self.dataInfo  = self.raw['dataInfo'].transpose()
        self._dataInfo = self.raw['dataInfo']

        self._delayed_loading = delayed_trajectory_loading
        self._description = None
        self._data_1      = None
        self._mtime       = None if self._is_stream else os.path.getmtime(self._fname)
        self._data_sections = data_sections

    def _update_data_info(self):
        """
        Updates the data related to the data sections in case of that the file has changed.
        """
        data_sections = [d for d in self._data_sections if d.startswith("data_")]
        with open(self._fname, "rb") as f:
            delayed  = DelayedVariableLoad(f, chars_as_strings=False)
            self.raw = delayed.get_variables(variable_names = data_sections)
        self._mtime = os.path.getmtime(self._fname)
        self._data_2_info = self.raw["data_2"]
        self._data_2 = {}

        if self._contains_diagnostic_data:
            self._data_3_info = self.raw["data_3"]
            self._data_3 = {}
            self._data_4_info = self.raw["data_4"]

            self._file_pos_model_var = np.empty(self._data_2_info["nbr_points"], dtype=np.longlong)
            self._file_pos_diag_var = np.empty(self._data_3_info.shape[0], dtype=np.longlong)

            self._has_file_pos_data = False

    def _verify_file_data(self):
        if self._mtime != os.path.getmtime(self._fname):
            if self._allow_file_updates:
                self._update_data_info()
            else:
                raise JIOError("The result file has been modified since the result object was created. Please make sure that different filenames are used for different simulations.")

    def _get_data_1(self):
        if self._data_1 is None:
            if not self._is_stream and self._mtime != os.path.getmtime(self._fname) and not self._allow_file_updates:
                raise JIOError("The result file has been modified since the result object was created. Please make sure that different filenames are used for different simulations.")

            self._data_1 = scipy.io.loadmat(self._fname,chars_as_strings=False, variable_names=["data_1"])["data_1"]

        return self._data_1

    data_1 = property(_get_data_1, doc =
    """
    Property for accessing the constant/parameter vector.
    """)

    def _get_name_dict(self):
        file_position  = self._name_info["file_position"]
        sizeof_type    = self._name_info["sizeof_type"]
        max_length     = self._name_info["max_length"]
        nbr_variables  = self._name_info["nbr_variables"]

        name_dict = fmi_util.read_name_list(encode(self._fname), file_position, int(nbr_variables), int(max_length))

        if self._contains_diagnostic_data:  # Populate name dict with diagnostics variables calculated on the fly
            dict_names = list(name_dict.keys())
            name_dict[DiagnosticsBase.calculated_diagnostics['nbr_steps']['name']] = None
            for name in dict_names:
                if isinstance(name, bytes):
                    name = decode(name)
                if name == f'{DIAGNOSTICS_PREFIX}event_data.event_info.event_type':
                    name_dict[DiagnosticsBase.calculated_diagnostics['nbr_time_events']['name']] = None
                    name_dict[DiagnosticsBase.calculated_diagnostics['nbr_state_events']['name']] = None
                    name_dict[DiagnosticsBase.calculated_diagnostics['nbr_events']['name']] = None
                    continue
                if f'{DIAGNOSTICS_PREFIX}state_errors.' in name:
                    state_name = name.replace(f'{DIAGNOSTICS_PREFIX}state_errors.', '')
                    name_dict["{}.{}".format(DiagnosticsBase.calculated_diagnostics['nbr_state_limits_step']['name'], state_name)] = None
                if name == f'{DIAGNOSTICS_PREFIX}cpu_time_per_step':
                    name_dict[f'{DIAGNOSTICS_PREFIX}cpu_time'] = None

        return name_dict

    def _can_use_partial_cache(self, start_index: int, stop_index: Union[int, None]):
        """ Checks if start_index and stop_index are equal to the last cached indices. """
        return self._allow_file_updates and (self._last_set_of_indices == (start_index, stop_index))

    def _get_trajectory(self, data_index, start_index = 0, stop_index = None):
        if isinstance(self._data_2, dict):
            self._verify_file_data()

            index_in_cache = data_index in self._data_2
            partial_cache_ok = self._can_use_partial_cache(start_index, stop_index)
            if (index_in_cache and not self._allow_file_updates) or (index_in_cache and partial_cache_ok):
                return self._data_2[data_index]

            file_position  = self._data_2_info["file_position"]
            sizeof_type    = self._data_2_info["sizeof_type"]
            nbr_points     = self._data_2_info["nbr_points"]
            nbr_variables  = self._data_2_info["nbr_variables"]

            # Account for sub-sets of data
            start_index = max(0, start_index)
            stop_index = max(0, nbr_points if stop_index is None else min(nbr_points, stop_index))
            new_file_position = file_position + start_index*sizeof_type*nbr_variables
            # Finally when stop_index = None, we can end up with start > stop,
            # therefore we need to use min(start, stop)
            start_index = min(start_index, stop_index)
            new_nbr_points = stop_index - start_index

            self._data_2[data_index] = fmi_util.read_trajectory(
                encode(self._fname),
                data_index,
                new_file_position,
                sizeof_type,
                int(new_nbr_points),
                int(nbr_variables)
            )

            return self._data_2[data_index]
        else:
            return self._data_2[data_index,:]

    def _get_diagnostics_trajectory(self, data_index, start_index = 0, stop_index = None):
        """ Returns trajectory for the diagnostics variable that corresponds to index 'data_index'. """
        self._verify_file_data()

        index_in_cache = data_index in self._data_3
        partial_cache_ok = self._can_use_partial_cache(start_index, stop_index)
        if (index_in_cache and not self._allow_file_updates) or (index_in_cache and partial_cache_ok):
            return self._data_3[data_index]
        self._data_3[data_index] = self._read_trajectory_data(data_index, True, start_index, stop_index)[start_index:stop_index]
        return self._data_3[data_index]

    def _read_trajectory_data(self, data_index, read_diag_data, start_index = 0, stop_index = None):
        """ Reads corresponding trajectory data for variable with index 'data_index',
            note that 'read_diag_data' is a boolean used when this function is invoked for
            diagnostic variables.
        """
        self._verify_file_data()

        file_position   = int(self._data_2_info["file_position"])
        sizeof_type     = int(self._data_2_info["sizeof_type"])
        nbr_points      = int(self._data_2_info["nbr_points"])
        nbr_variables   = int(self._data_2_info["nbr_variables"])

        nbr_diag_points    = int(self._data_3_info.shape[0])
        nbr_diag_variables = int(self._data_4_info.shape[0])

        # TODO account for start_index and stop_index

        data, self._file_pos_model_var, self._file_pos_diag_var = fmi_util.read_diagnostics_trajectory(
            encode(self._fname),
            int(read_diag_data),
            int(self._has_file_pos_data),
            self._file_pos_model_var,
            self._file_pos_diag_var,
            data_index,
            file_position,
            sizeof_type,
            nbr_points,
            nbr_diag_points,
            nbr_variables,
            nbr_diag_variables
        )
        self._has_file_pos_data = True

        return data

    def _get_interpolated_trajectory(self, data_index: int, start_index: int = 0, stop_index: int = None) -> Trajectory:
        """ Returns an interpolated trajectory for variable of corresponding index 'data_index'. """
        self._verify_file_data()

        index_in_cache = data_index in self._data_2
        partial_cache_ok = self._can_use_partial_cache(start_index, stop_index)
        if (index_in_cache and not self._allow_file_updates) or (index_in_cache and partial_cache_ok):
            return self._data_2[data_index]

        diag_time_vector = self._get_diagnostics_trajectory(0, start_index, stop_index)
        time_vector      = self._read_trajectory_data(0, False, start_index, stop_index)
        data             = self._read_trajectory_data(data_index, False, start_index, stop_index)

        f = scipy.interpolate.interp1d(time_vector, data, fill_value="extrapolate")

        # note that we dont need to slice here because diag_time_vector is already sliced accordingly
        self._data_2[data_index] = f(diag_time_vector)
        return self._data_2[data_index]

    def _get_description(self):
        if not self._description:
            description = scipy.io.loadmat(
                self._fname,
                chars_as_strings = False,
                variable_names = ["description"])["description"]
            self._description = ["".join(description[:,i]).rstrip() for i in range(np.size(description[0,:]))]

        return self._description

    description = property(_get_description, doc =
    """
    Property for accessing the description vector.
    """)

    def _map_index_to_data_properties(self, variable_name: str) -> tuple[int, int, int]:
        """ Returns the data matrix ID, index and factor for given variable name. """

        variable_index = self.get_variable_index(variable_name)
        data_mat = self._dataInfo[0][variable_index]

        data_index = self._dataInfo[1][variable_index]
        data_index, factor = self._correct_index_for_alias(data_index)

        if data_mat == 0:
            # Take into account that the 'Time' variable can be named either 'Time' or 'time'.
            data_mat = 2 if len(self.raw['data_2'])> 0 else 1

        return factor, data_index, data_mat

    def _get_variable_data_as_trajectory(self,
                                         name: str,
                                         time: Union[Trajectory, None] = None,
                                         start_index: Union[int, None] = 0,
                                         stop_index : Union[int, None]= None) -> Trajectory:
        """
        Retrieve an instance of Trajectory for a variable with a given name.

        Parameters::
            name --
                Name of variable.
            time --
                [Optional] An array with time data. If specified, the number of data points
                          must align with the length of the data for requested variable.
            start_index --
                [Optional] An integer to enable retrieving subsets of the full trajectories, starting
                          from corresponding data at start_index.
            stop_index --
                [Optional] An integer to enable retrieving subsets of the full trajectories, ending
                          at corresponding data at stop_index.

        Returns::

            A Trajectory object containing the time vector and the data vector
            of the variable.
        """
        if isinstance(name, bytes):
            name = decode(name)

        if time is None:
            # Get the time trajectory
            if not self._contains_diagnostic_data:
                time = self._get_trajectory(0, start_index, stop_index)
            else:
                # Since we interpolate data if diagnostics is enabled
                time = self._get_diagnostics_trajectory(0, start_index, stop_index)

        if name == 'time' or name == 'Time':
            return Trajectory(time, time)
        elif self._contains_diagnostic_data and (
                name in [
                    DiagnosticsBase.calculated_diagnostics['nbr_events']['name'],
                    DiagnosticsBase.calculated_diagnostics['nbr_time_events']['name'],
                    DiagnosticsBase.calculated_diagnostics['nbr_state_events']['name'],
                    DiagnosticsBase.calculated_diagnostics['nbr_steps']['name']]
            ):
            return Trajectory(
                time, self._calculate_events_and_steps(name)[start_index:stop_index])
        elif self._contains_diagnostic_data and (
                f"{DiagnosticsBase.calculated_diagnostics['nbr_state_limits_step']['name']}." in name
            ):
            return Trajectory(
                time, self._calculate_nbr_state_limits_step(name)[start_index:stop_index])
        elif self._contains_diagnostic_data and (
                name == f'{DIAGNOSTICS_PREFIX}cpu_time'
            ):
            return Trajectory(
                time, np.cumsum(self.get_variable_data(f'{DIAGNOSTICS_PREFIX}cpu_time_per_step').x)[start_index:stop_index])

        factor, data_index, data_mat = self._map_index_to_data_properties(name)

        if data_mat == 1:
            return Trajectory(
                self.data_1[0, start_index:stop_index], factor*self.data_1[data_index, start_index:stop_index])
        elif data_mat == 2 and not self._contains_diagnostic_data:
            return Trajectory(
                time, factor*self._get_trajectory(data_index, start_index, stop_index))
        elif data_mat == 3:
            return Trajectory(
                time, self._get_diagnostics_trajectory(data_index+1, start_index, stop_index))
        else:
            return Trajectory(
                time, factor*self._get_interpolated_trajectory(data_index, start_index, stop_index))


    def get_variable_data(self, name: str) -> Trajectory:
        """
        Retrieve the data sequence for a variable with a given name.

        Parameters::

            name --
                Name of the variable.

        Returns::

            A Trajectory object containing the time vector and the data vector
            of the variable.
        """
        # prevent caching interference with 'get_variables_data'
        self._last_set_of_indices = (None, None)
        return self._get_variable_data_as_trajectory(name)

    def get_variables_data(self,
                           names: list[str],
                           start_index: int = 0,
                           stop_index: Union[int, None] = None
    ) -> tuple[dict[str, Trajectory], Union[int, None]]:
        """"
            Returns trajectories for each variable in 'names' with lengths adjusted for the
            interval [start_index, stop_index], i.e. partial trajectories.
            Improper values for start_index and stop_index that are out of bounds are automatically corrected,
            such that:
                Negative values are always adjusted to 0 or larger.
                Out of bounds for stop_index is adjusted for the number of available data points, example:
                If start_index = 0, stop_index = 5 and there are only 3 data points available,
                then returned trajectories are of length 3.
                If start_index is larger than or equal to the number of available data points, empty trajectories
                are returned, i.e. trajectories of length 0.
            Note that trajectories for parameters are always of length 2 if indices 0 and 1 are
            part of the requested trajectory since they reflect the values of before and after initialization.
            Therefore if you request a trajectory for a parameter with start_index>=2, returned trajectory is empty.

            By default, start_index = 0 and stop_index = None, which implies that the full trajectory is returned.

            Parameters::

                names --
                    List of variables names for which to fetch trajectories.

                start_index --
                    The index from where the trajectory data starts from.

                stop_index --
                    The index from where the trajectory data ends. If stop_index is set to None,
                    it implies that all data in the slice [start_index:] is returned.

            Raises::
                ValueError                        -- If stop_index < start_index.

            Returns::
                Tuple: (dict of trajectories with keys corresponding to variable names, next start index (non-negative))
        """

        """
            TODO:
            For [start_index, stop_index] we can do performance improvements
            since methods such as _get_trajectory, get_diagnostics_trajectory
            and _get_interpolated_trajectory always retrieve the full
            trajectory which is then sliced, here. Instead we can account for
            start_index and stop_index and reduce the time spent reading unused data points.
        """
        if isinstance(start_index, int) and isinstance(stop_index, int) and stop_index < start_index:
            raise ValueError(f"Invalid values for {start_index=} and {stop_index=}, " + \
                              "'start_index' needs to be less than or equal to 'stop_index'.")

        trajectories = {}

        # Get the corresponding time trajectory
        if not self._contains_diagnostic_data:
            time = self._get_trajectory(0, start_index, stop_index)
        else:
            # Since we interpolate data if diagnostics is enabled
            time = self._get_diagnostics_trajectory(0, start_index, stop_index)

        # If stop_index > number of data points, and data gets added while we are iterating
        # then we might get trajectories of unequal lengths. Therefore ensure we set stop_index here accordingly.
        if stop_index is None:
            stop_index = len(time) + start_index
        else:
            stop_index = min(len(time) + start_index, stop_index)

        for name in names:
            trajectories[name] = self._get_variable_data_as_trajectory(name, time, start_index, stop_index)

        largest_trajectory_length = self._find_max_trajectory_length(trajectories)
        new_start_index = (start_index + largest_trajectory_length) if trajectories else start_index

        self._last_set_of_indices = (start_index, stop_index) # update them before we exit
        return trajectories, new_start_index

    def _find_max_trajectory_length(self, trajectories):
        """
            Given a dict of trajectories, find the length of the largest trajectory
            among the set of continuous variables. We disregard parameters/constants since they are not stored
            with the same amount of data points as trajectories for continuous variables.
        """
        return max([0] + [len(t.x) for v, t in trajectories.items() if self.is_variable(v)])

    def _calculate_events_and_steps(self, name):
        if name in self._data_3:
            return self._data_3[name]
        all_events_name = DiagnosticsBase.calculated_diagnostics['nbr_events']['name']
        time_events_name = DiagnosticsBase.calculated_diagnostics['nbr_time_events']['name']
        state_events_name = DiagnosticsBase.calculated_diagnostics['nbr_state_events']['name']
        steps_name = DiagnosticsBase.calculated_diagnostics['nbr_steps']['name']
        try:
            event_type_data = self.get_variable_data(f'{DIAGNOSTICS_PREFIX}event_data.event_info.event_type')
        except Exception:
            if name == steps_name:
                self._data_3[steps_name] = np.array(range(len(self._get_diagnostics_trajectory(0))))
                return self._data_3[name]
        self._data_3[all_events_name] = np.zeros(len(event_type_data.x))
        self._data_3[time_events_name] = np.zeros(len(event_type_data.x))
        self._data_3[state_events_name] = np.zeros(len(event_type_data.x))
        self._data_3[steps_name] = np.zeros(len(event_type_data.x))
        nof_events = 0
        nof_time_events = 0
        nof_state_events = 0
        nof_steps = 0
        for ind, etype in enumerate(event_type_data.x):
            if etype == 1:
                nof_events += 1
                nof_time_events += 1
            elif etype == 0:
                nof_state_events += 1
                nof_events += 1
            else:
                nof_steps += 1
            self._data_3[all_events_name][ind] = nof_events
            self._data_3[time_events_name][ind] = nof_time_events
            self._data_3[state_events_name][ind] = nof_state_events
            self._data_3[steps_name][ind] = nof_steps
        return self._data_3[name]

    def _calculate_nbr_state_limits_step(self, name):
        if name in self._data_3:
            return self._data_3[name]
        step_limitation_name = '{}.'.format(DiagnosticsBase.calculated_diagnostics['nbr_state_limits_step']['name'])
        state_name = name.replace(step_limitation_name, '')
        state_error_data = self.get_variable_data(f'{DIAGNOSTICS_PREFIX}state_errors.'+state_name)
        event_type_data = self.get_variable_data(f'{DIAGNOSTICS_PREFIX}event_data.event_info.event_type')
        self._data_3[name] = np.zeros(len(event_type_data.x))
        nof_times_state_limits_step = 0
        for ind, state_error in enumerate(state_error_data.x):
            if event_type_data.x[ind] == -1 and state_error >= 1.0:
                nof_times_state_limits_step += 1
            self._data_3[name][ind] = nof_times_state_limits_step
        return self._data_3[name]

    def is_variable(self, name):
        """
        Returns True if the given name corresponds to a time-varying variable.

        Parameters::

            name --
                Name of the variable/parameter/constant.

        Returns::

            True if the variable is time-varying.
        """
        if name == 'time' or name== 'Time':
            return True
        elif name in [DiagnosticsBase.calculated_diagnostics['nbr_events']['name'],
                      DiagnosticsBase.calculated_diagnostics['nbr_time_events']['name'],
                      DiagnosticsBase.calculated_diagnostics['nbr_state_events']['name'],
                      DiagnosticsBase.calculated_diagnostics['nbr_steps']['name'],
                      f'{DIAGNOSTICS_PREFIX}cpu_time']:
            return True
        elif '{}.'.format(DiagnosticsBase.calculated_diagnostics['nbr_state_limits_step']['name']) in name:
            return True

        variable_index  = self.get_variable_index(name)
        data_mat = self._dataInfo[0][variable_index]
        if data_mat < 1:
            data_mat = 1

        return data_mat != 1

    def is_negated(self, name):
        """
        Returns True if the given name corresponds to a negated result vector.

        Parameters::

            name --
                Name of the variable/parameter/constant.

        Returns::

            True if the result should be negated
        """
        variable_index  = self.get_variable_index(name)
        data_index = self._dataInfo [1][variable_index]
        if data_index<0:
            return True
        else:
            return False

    def get_column(self, name):
        """
        Returns the column number in the data matrix where the values of the
        variable are stored.

        Parameters::

            name --
                Name of the variable/parameter/constant.

        Returns::

            The column number.
        """
        if name == 'time' or name== 'Time':
            return 0

        if not self.is_variable(name):
            raise VariableNotTimeVarying("Variable " + name + " is not time-varying.")
        variable_index  = self.get_variable_index(name)
        data_index = self._dataInfo[1][variable_index]
        data_index, _ = self._correct_index_for_alias(data_index)

        return data_index

    def get_data_matrix(self):
        """
        Returns the result matrix. If delayed loading is used, this will
        force loading of the full result matrix.

        Returns::

            The result data matrix as a numpy array
        """
        if isinstance(self._data_2, dict):
            return scipy.io.loadmat(self._fname,chars_as_strings=False, variable_names=["data_2"])["data_2"]
        return self._data_2

class ResultHandlerMemory(ResultHandler):
    def __init__(self, model, delimiter=";"):
        super().__init__(model)
        self.supports['result_max_size'] = True
        self._first_point = True

    def simulation_start(self):
        """
        This method is called before the simulation has started and before
        the initialization of the model.
        """
        model = self.model
        opts = self.options

        self.vars = model.get_model_variables(filter=opts["filter"])

        #Store the continuous and discrete variables for result writing
        self.real_var_ref, self.int_var_ref, self.bool_var_ref = model.get_model_time_varying_value_references(filter=opts["filter"])

        self.real_sol = []
        self.int_sol  = []
        self.bool_sol = []
        self.time_sol = []
        self.param_sol= []

        self.model = model

    def initialize_complete(self):
        """
        This method is called after the initialization method of the FMU
        has been been called.
        """
        pass

    def integration_point(self, solver = None):
        """
        This method is called for each time-point for which result are
        to be stored as indicated by the "number of communcation points"
        provided to the simulation method.
        """
        model = self.model

        previous_size = 0
        if self._first_point:
            previous_size = sys.getsizeof(self.time_sol) + sys.getsizeof(self.real_sol) + \
                           sys.getsizeof(self.int_sol)  + sys.getsizeof(self.bool_sol) + \
                           sys.getsizeof(self.param_sol)

        #Retrieves the time-point
        self.time_sol += [model.time]
        self.real_sol += [model.get_real(self.real_var_ref)]
        self.int_sol  += [model.get_integer(self.int_var_ref)]
        self.bool_sol += [model.get_boolean(self.bool_var_ref)]

        #Sets the parameters, if any
        if solver and self.options["sensitivities"]:
            self.param_sol += [np.array(solver.interpolate_sensitivity(model.time, 0)).flatten()]

        max_size = self.options.get("result_max_size", None)
        if max_size is not None:
            current_size = sys.getsizeof(self.time_sol) + sys.getsizeof(self.real_sol) + \
                           sys.getsizeof(self.int_sol)  + sys.getsizeof(self.bool_sol) + \
                           sys.getsizeof(self.param_sol)

            verify_result_size("", self._first_point, current_size, previous_size, max_size, self.options["ncp"], self.model.time)
            self._first_point = False

    def get_result(self):
        """
        Method for retrieving the result. This method should return a
        result of an instance of ResultBase or of an instance of a
        subclass of ResultBase.
        """
        t = np.array(self.time_sol)
        r = np.array(self.real_sol)
        data = np.c_[t,r]

        if len(self.int_sol) > 0 and len(self.int_sol[0]) > 0:
            i = np.array(self.int_sol)
            data = np.c_[data,i]
        if len(self.bool_sol) > 0 and len(self.bool_sol[0]) > 0:
            b = np.array(self.bool_sol)
            data = np.c_[data,b]

        return ResultStorageMemory(self.model, data, [self.real_var_ref,self.int_var_ref,self.bool_var_ref], self.vars)

class ResultHandlerCSV(ResultHandler):
    def __init__(self, model, delimiter=";"):
        super().__init__(model)
        self.supports['result_max_size'] = True
        self.delimiter = delimiter
        self._current_file_size = 0
        self._first_point = True

    def _write(self, msg):
        self._current_file_size = self._current_file_size+len(msg)
        self._file.write(msg)

    def initialize_complete(self):
        pass

    def simulation_start(self):
        """
        Opens the file and writes the header. This includes the information
        about the variables and a table determining the link between variables
        and data.
        """
        opts = self.options
        model = self.model

        #Internal values
        self.file_open = False
        self.nbr_points = 0
        delimiter = self.delimiter

        self.file_name = opts["result_file_name"]
        try:
            self.parameters = opts["sensitivities"]
        except KeyError:
            self.parameters = None

        if self.file_name == "":
            self.file_name=self.model.get_identifier() + '_result.csv'
        self.model._result_file = self.file_name

        vars = model.get_model_variables(filter=opts["filter"])

        const_valref_real = []
        const_name_real = []
        const_alias_real = []
        const_valref_int = []
        const_name_int = []
        const_alias_int = []
        const_valref_bool = []
        const_name_bool = []
        const_alias_bool = []
        cont_valref_real = []
        cont_name_real = []
        cont_alias_real = []
        cont_valref_int = []
        cont_name_int = []
        cont_alias_int = []
        cont_valref_bool = []
        cont_name_bool = []
        cont_alias_bool = []

        for name in vars.keys():
            var = vars[name]
            if var.type == fmi.FMI_REAL:
                if var.variability == fmi.FMI_CONSTANT or var.variability == fmi.FMI_PARAMETER:
                    const_valref_real.append(var.value_reference)
                    const_name_real.append(var.name)
                    const_alias_real.append(-1 if var.alias == fmi.FMI_NEGATED_ALIAS else 1)
                else:
                    cont_valref_real.append(var.value_reference)
                    cont_name_real.append(var.name)
                    cont_alias_real.append(-1 if var.alias == fmi.FMI_NEGATED_ALIAS else 1)
            elif var.type == fmi.FMI_INTEGER or var.type == fmi.FMI_ENUMERATION:
                if var.variability == fmi.FMI_CONSTANT or var.variability == fmi.FMI_PARAMETER:
                    const_valref_int.append(var.value_reference)
                    const_name_int.append(var.name)
                    const_alias_int.append(-1 if var.alias == fmi.FMI_NEGATED_ALIAS else 1)
                else:
                    cont_valref_int.append(var.value_reference)
                    cont_name_int.append(var.name)
                    cont_alias_int.append(-1 if var.alias == fmi.FMI_NEGATED_ALIAS else 1)
            elif var.type == fmi.FMI_BOOLEAN:
                if var.variability == fmi.FMI_CONSTANT or var.variability == fmi.FMI_PARAMETER:
                    const_valref_bool.append(var.value_reference)
                    const_name_bool.append(var.name)
                    const_alias_bool.append(-1 if var.alias == fmi.FMI_NEGATED_ALIAS else 1)
                else:
                    cont_valref_bool.append(var.value_reference)
                    cont_name_bool.append(var.name)
                    cont_alias_bool.append(-1 if var.alias == fmi.FMI_NEGATED_ALIAS else 1)

        # Open file
        if isinstance(self.file_name, str):
            f = codecs.open(self.file_name,'w','utf-8')
            self.file_open = True
        else:
            if not hasattr(self.file_name, 'write'):
                raise FMUException("Failed to write the result file. Option 'result_file_name' needs to be a filename or a class that supports writing to through the 'write' method.")
            f = self.file_name #assume it is a stream
            self.file_open = False
        self._file = f

        if delimiter == ",":
            name_str = '"time"'
            for name in const_name_real+const_name_int+const_name_bool+cont_name_real+cont_name_int+cont_name_bool:
                name_str += delimiter+'"'+name+'"'
        else:
            name_str = "time"
            for name in const_name_real+const_name_int+const_name_bool+cont_name_real+cont_name_int+cont_name_bool:
                name_str += delimiter+name

        self._write(name_str+"\n")

        const_val_real    = model.get_real(const_valref_real)
        const_val_int     = model.get_integer(const_valref_int)
        const_val_bool    = model.get_boolean(const_valref_bool)

        const_str = ""
        for i,val in enumerate(const_val_real):
            const_str += "%.14E"%(const_alias_real[i]*val)+delimiter
        for i,val in enumerate(const_val_int):
            const_str += "%.14E"%(const_alias_int[i]*val)+delimiter
        for i,val in enumerate(const_val_bool):
            const_str += "%.14E"%(const_alias_bool[i]*val)+delimiter
        self.const_str = const_str

        self.cont_valref_real = cont_valref_real
        self.cont_alias_real  = np.array(cont_alias_real)
        self.cont_valref_int  = cont_valref_int
        self.cont_alias_int  = np.array(cont_alias_int)
        self.cont_valref_bool = cont_valref_bool
        self.cont_alias_bool  = np.array(cont_alias_bool)

    def integration_point(self, solver = None):
        """
        Writes the current status of the model to file.
        """
        model = self.model
        delimiter = self.delimiter
        previous_size = self._current_file_size

        #Retrieves the time-point
        t = model.time
        r = model.get_real(self.cont_valref_real)*self.cont_alias_real
        i = model.get_integer(self.cont_valref_int)*self.cont_alias_int
        b = model.get_boolean(self.cont_valref_bool)*self.cont_alias_bool

        data = np.append(np.append(r,i),b)

        cont_str = ""
        for val in data:
            cont_str += "%.14E%s"%(val,delimiter)

        if len(cont_str) == 0 and len(self.const_str) == 0:
            self._write("%.14E"%(t))
        else:
            self._write("%.14E%s"%(t,delimiter))

        if len(cont_str) == 0:
            self._write(self.const_str[:-1]+"\n")
        else:
            self._write(self.const_str)
            self._write(cont_str[:-1]+"\n")

        max_size = self.options.get("result_max_size", None)
        if max_size is not None:
            verify_result_size(self.file_name, self._first_point, self._current_file_size, previous_size, max_size, self.options["ncp"], self.model.time)
            self._first_point = False

    def simulation_end(self):
        """
        Finalize the writing by closing the file.
        """
        #If open, finalize and close
        if self.file_open:
            self._file.close()
            self.file_open = False

    def get_result(self):
        """
        Method for retrieving the result. This method should return a
        result of an instance of ResultBase or of an instance of a
        subclass of ResultBase.
        """
        return ResultCSVTextual(self.file_name, self.delimiter)

class ResultHandlerFile(ResultHandler):
    """
    Export an optimization or simulation result to file in Dymola's result file
    format.
    """
    def __init__(self, model):
        super().__init__(model)
        self.supports['result_max_size'] = True
        self._current_file_size = 0
        self._first_point = True

    def initialize_complete(self):
        pass

    def simulation_start(self):
        """
        Opens the file and writes the header. This includes the information
        about the variables and a table determining the link between variables
        and data.
        """
        opts = self.options
        model = self.model

        #Internal values
        self.file_open = False
        self._is_stream = False
        self.nbr_points = 0

        self.file_name = opts["result_file_name"]
        try:
            self.parameters = opts["sensitivities"]
        except KeyError:
            self.parameters = None

        if self.file_name == "":
            self.file_name=self.model.get_identifier() + '_result.txt'
        self.model._result_file = self.file_name

        #Store the continuous and discrete variables for result writing
        self.real_var_ref, self.int_var_ref, self.bool_var_ref = model.get_model_time_varying_value_references(filter=opts["filter"])

        parameters = self.parameters

        # Open file
        if isinstance(self.file_name, str):
            f = codecs.open(self.file_name,'w','utf-8')
        else:
            if not (hasattr(self.file_name, 'write') and hasattr(self.file_name, 'seek')):
                raise FMUException("Failed to write the result file. Option 'result_file_name' needs to be a filename or a class that supports 'write' and 'seek'.")
            f = self.file_name #assume it is a stream
            self._is_stream = True
        self.file_open = True
        self._file = f

        # Write header
        self._write('#1\n')
        self._write('char Aclass(3,11)\n')
        self._write('Atrajectory\n')
        self._write('1.1\n')
        self._write('\n')

        # all lists that we need for later
        vrefs_alias = []
        vrefs_noalias = []
        vrefs = []
        names_alias = []
        names_noalias = []
        names = []
        aliases_alias = []
        aliases = []
        descriptions_alias = []
        descriptions = []
        variabilities_alias = []
        variabilities_noalias = []
        variabilities = []
        types_alias = []
        types_noalias = []
        types = []

        for var in self.model.get_model_variables(filter=self.options["filter"]).values():
            if not var.type == fmi.FMI_STRING:
                    if var.alias == fmi.FMI_NO_ALIAS:
                        vrefs_noalias.append(var.value_reference)
                        names_noalias.append(var.name)
                        aliases.append(var.alias)
                        descriptions.append(var.description)
                        variabilities_noalias.append(var.variability)
                        types_noalias.append(var.type)
                    else:
                        vrefs_alias.append(var.value_reference)
                        names_alias.append(var.name)
                        aliases_alias.append(var.alias)
                        descriptions_alias.append(var.description)
                        variabilities_alias.append(var.variability)
                        types_alias.append(var.type)

        # need to save these no alias lists for later
        vrefs = vrefs_noalias[:]
        names = names_noalias[:]
        types = types_noalias[:]
        variabilities = variabilities_noalias[:]

        # merge lists
        vrefs.extend(vrefs_alias)
        names.extend(names_alias)
        aliases.extend(aliases_alias)
        descriptions.extend(descriptions_alias)
        variabilities.extend(variabilities_alias)
        types.extend(types_alias)

        # zip to list of tuples and sort - non alias variables are now
        # guaranteed to be first in list
        names_noalias = sorted(zip(
            tuple(vrefs_noalias),
            tuple(names_noalias)),
            key=itemgetter(0))
        variabilities_noalias = sorted(zip(
            tuple(vrefs_noalias),
            tuple(variabilities_noalias)),
            key=itemgetter(0))
        types_noalias = sorted(zip(
            tuple(vrefs_noalias),
            tuple(types_noalias)),
            key=itemgetter(0))
        names = sorted(zip(
            tuple(vrefs),
            tuple(names)),
            key=itemgetter(0))
        aliases = sorted(zip(
            tuple(vrefs),
            tuple(aliases)),
            key=itemgetter(0))
        descriptions = sorted(zip(
            tuple(vrefs),
            tuple(descriptions)),
            key=itemgetter(0))
        variabilities = sorted(zip(
            tuple(vrefs),
            tuple(variabilities)),
            key=itemgetter(0))
        types = sorted(zip(
            tuple(vrefs),
            tuple(types)),
            key=itemgetter(0))

        num_vars = len(names)

        names_sens = []
        descs_sens = []
        cont_vars = []

        if parameters is not None:

            if isinstance(self.model, fmi.FMUModelME2):
                vars = self.model.get_model_variables(type=fmi.FMI2_REAL,include_alias=False,variability=fmi.FMI2_CONTINUOUS,filter=self.options["filter"])
                state_vars = [v.value_reference for i,v in self.model.get_states_list().items()]
            else:
                vars = self.model.get_model_variables(type=fmi.FMI_REAL,include_alias=False,variability=fmi.FMI_CONTINUOUS,filter=self.options["filter"])
                state_vars = self.model.get_state_value_references()
            for i in state_vars:
                for j in vars.keys():
                    if i == vars[j].value_reference:
                        cont_vars.append(vars[j].name)

            for j in range(len(parameters)):
                for i in range(len(self.model.continuous_states)):
                    names_sens += ['d'+cont_vars[i]+'/d'+parameters[j]]
                    descs_sens  += ['Sensitivity of '+cont_vars[i]+' with respect to '+parameters[j]+'.']

        # Find the maximum name and description length
        max_name_length = len('Time')
        max_desc_length = len('Time in [s]')

        for i in range(len(names)):
            name = names[i][1]
            desc = descriptions[i][1]

            if (len(name)>max_name_length):
                max_name_length = len(name)

            if (len(desc)>max_desc_length):
                max_desc_length = len(desc)

        for i in range(len(names_sens)):
            name = names_sens[i]
            desc = descs_sens[i]

            if (len(name)>max_name_length):
                max_name_length = len(name)

            if (len(desc)>max_desc_length):
                max_desc_length = len(desc)

        self._write('char name(%d,%d)\n' % (num_vars+len(names_sens)+1, max_name_length))
        self._write('time\n')

        for name in names:
            self._write(name[1] +'\n')
        for name in names_sens:
            self._write(name + '\n')

        self._write('\n')

        if not opts["result_store_variable_description"]:
            max_desc_length = 0
            descriptions    = [[0,""] for d in descriptions]
            descs_sens      = ["" for d in descs_sens]

        # Write descriptions
        self._write('char description(%d,%d)\n' % (num_vars+len(names_sens) + 1, max_desc_length))
        self._write('Time in [s]\n')

        # Loop over all variables, not only those with a description
        for desc in descriptions:
            self._write(desc[1] +'\n')
        for desc in descs_sens:
            self._write(desc + '\n')

        self._write('\n')

        # Write data meta information

        self._write('int dataInfo(%d,%d)\n' % (num_vars+len(names_sens) + 1, 4))
        self._write('0 1 0 -1 # time\n')

        lst_real_cont = dict(zip(self.real_var_ref,range(len(self.real_var_ref))))
        lst_int_cont  = dict(zip(self.int_var_ref,[len(self.real_var_ref)+x for x in range(len(self.int_var_ref))]))
        lst_bool_cont = dict(zip(self.bool_var_ref,[len(self.real_var_ref)+len(self.int_var_ref)+x for x in range(len(self.bool_var_ref))]))

        valueref_of_continuous_states = []
        list_of_parameters = []

        cnt_1 = 1
        cnt_2 = 1
        n_parameters = 0
        datatable1 = False
        last_real_vref = -1; last_int_vref = -1; last_bool_vref = -1
        for i, name in enumerate(names):
            update = False
            if (types[i][1] == fmi.FMI_REAL and last_real_vref != name[0]):
                last_real_vref = name[0]
                update = True
            if ((types[i][1] == fmi.FMI_INTEGER or types[i][1] == fmi.FMI_ENUMERATION) and last_int_vref != name[0]):
                last_int_vref = name[0]
                update = True
            if (types[i][1] == fmi.FMI_BOOLEAN and last_bool_vref != name[0]):
                last_bool_vref = name[0]
                update = True
            if update:
                if aliases[i][1] == 0:
                    if variabilities[i][1] == fmi.FMI_PARAMETER or \
                        variabilities[i][1] == fmi.FMI_CONSTANT:
                        cnt_1 += 1
                        n_parameters += 1
                        datatable1 = True
                        list_of_parameters.append((types[i][0],types[i][1]))
                    else:
                        cnt_2 += 1
                        #valueref_of_continuous_states.append(
                        #    list_of_continuous_states[name[0]])
                        if types[i][1] == fmi.FMI_REAL:
                            valueref_of_continuous_states.append(lst_real_cont[name[0]])
                        elif types[i][1] == fmi.FMI_INTEGER or types[i][1] == fmi.FMI_ENUMERATION:
                            valueref_of_continuous_states.append(lst_int_cont[name[0]])
                        else:
                            valueref_of_continuous_states.append(lst_bool_cont[name[0]])
                        datatable1 = False
                else:
                    base_var = self.model.get_variable_alias_base(name[1])
                    variability = self.model.get_variable_variability(base_var)
                    data_type = self.model.get_variable_data_type(base_var)
                    if data_type != types[i][1]:
                        raise Exception
                    if variability == fmi.FMI_PARAMETER or \
                        variability == fmi.FMI_CONSTANT:
                        cnt_1 += 1
                        n_parameters += 1
                        datatable1 = True
                        list_of_parameters.append((types[i][0],types[i][1]))
                    else:
                        cnt_2 += 1
                        #valueref_of_continuous_states.append(
                        #    list_of_continuous_states[name[0]])
                        if types[i][1] == fmi.FMI_REAL:
                            valueref_of_continuous_states.append(lst_real_cont[name[0]])
                        elif types[i][1] == fmi.FMI_INTEGER or types[i][1] == fmi.FMI_ENUMERATION:
                            valueref_of_continuous_states.append(lst_int_cont[name[0]])
                        else:
                            valueref_of_continuous_states.append(lst_bool_cont[name[0]])
                        datatable1 = False

            if aliases[i][1] == 0: # no alias
                #if variabilities[i][1] == fmi.FMI_PARAMETER or \
                #    variabilities[i][1] == fmi.FMI_CONSTANT:
                if datatable1:
                    #cnt_1 += 1
                    #n_parameters += 1
                    self._write('1 %d 0 -1 # ' % cnt_1 + name[1]+'\n')
                    #datatable1 = True
                else:
                    #cnt_2 += 1
                    #valueref_of_continuous_states.append(
                    #    list_of_continuous_states[name[0]])
                    self._write('2 %d 0 -1 # ' % cnt_2 + name[1] +'\n')
                    #datatable1 = False

            elif aliases[i][1] == 1: # alias
                if datatable1:
                    self._write('1 %d 0 -1 # ' % cnt_1 + name[1]+'\n')
                else:
                    self._write('2 %d 0 -1 # ' % cnt_2 + name[1] +'\n')
            else:
                if datatable1:
                    self._write('1 -%d 0 -1 # ' % cnt_1 + name[1]+'\n')
                else:
                    self._write('2 -%d 0 -1 # ' % cnt_2 + name[1] +'\n')
        for i, name in enumerate(names_sens):
            cnt_2 += 1
            self._write('2 %d 0 -1 # ' % cnt_2 + name +'\n')


        self._write('\n')

        # Write data
        # Write data set 1
        self._write('float data_1(%d,%d)\n' % (2, n_parameters + 1))
        self._write("%.14E" % self.model.time)
        str_text = ''

        # write constants and parameters
        for i, dtype in enumerate(list_of_parameters):
            vref = dtype[0]
            if dtype[1] == fmi.FMI_REAL:
                str_text = str_text + (
                    " %.14E" % (self.model.get_real([vref])[0]))
            elif dtype[1] == fmi.FMI_INTEGER or dtype[1] == fmi.FMI_ENUMERATION:
                str_text = str_text + (
                    " %.14E" % (self.model.get_integer([vref])[0]))
            elif dtype[1] == fmi.FMI_BOOLEAN:
                str_text = str_text + (
                    " %.14E" % (float(
                        self.model.get_boolean([vref])[0])))

        self._write(str_text)
        self._write('\n')
        self._point_last_t = f.tell()
        self._write("%s" % ' '*28)
        self._write(str_text)

        self._write('\n\n')

        self._nvariables = len(valueref_of_continuous_states)+1
        self._nvariables_sens = len(names_sens)


        self._write('float data_2(')
        self._point_npoints = f.tell()
        self._write(' '*(14+4+14))
        self._write('\n')

        self._data_order  = np.array(valueref_of_continuous_states)
        self.real_var_ref = np.array(self.real_var_ref)
        self.int_var_ref  = np.array(self.int_var_ref)
        self.bool_var_ref = np.array(self.bool_var_ref)

    def _write(self, msg):
        self._current_file_size = self._current_file_size+len(msg)
        self._file.write(msg)

    def integration_point(self, solver = None):#parameter_data=[]):
        """
        Writes the current status of the model to file. If the header has not
        been written previously it is written now. If data is specified it is
        written instead of the current status.

        Parameters::

                data --
                    A one dimensional array of variable trajectory data. data
                    should consist of information about the status in the order
                    specified by FMUModel*.save_time_point()
                    Default: None
        """
        data_order = self._data_order
        model = self.model
        previous_size = self._current_file_size

        #Retrieves the time-point
        r = model.get_real(self.real_var_ref)
        i = model.get_integer(self.int_var_ref)
        b = model.get_boolean(self.bool_var_ref)

        data = np.append(np.append(r,i),b)

        #Write the point
        str_text = (" %.14E" % self.model.time) + ''.join([" %.14E" % (data[data_order[j]]) for j in range(self._nvariables-1)])

        #Sets the parameters, if any
        if solver and self.options["sensitivities"]:
            parameter_data = np.array(solver.interpolate_sensitivity(model.time, 0)).flatten()
            for j in range(len(parameter_data)):
                str_text = str_text + (" %.14E" % (parameter_data[j]))

        self._write(str_text+'\n')

        #Update number of points
        self.nbr_points+=1

        max_size = self.options.get("result_max_size", None)
        if max_size is not None:
            verify_result_size(self.file_name, self._first_point, self._current_file_size, previous_size, max_size, self.options["ncp"], self.model.time)
            self._first_point = False

    def simulation_end(self):
        """
        Finalize the writing by filling in the blanks in the created file. The
        blanks consists of the number of points and the final time (in data set
        1). Also closes the file.
        """
        #If open, finalize and close
        if self.file_open:

            f = self._file

            f.seek(self._point_last_t)

            f.write('%.14E'%self.model.time)

            f.seek(self._point_npoints)
            f.write('%d,%d)' % (self.nbr_points, self._nvariables+self._nvariables_sens))

            if self._is_stream: #Seek relative to file end to allowed for string streams
                f.seek(0, os.SEEK_END)
                f.seek(f.tell()-1, os.SEEK_SET)
            else:
                f.seek(-1,2)
                #Close the file
                f.write('\n')
                f.close()
            self.file_open = False

    def get_result(self):
        """
        Method for retrieving the result. This method should return a
        result of an instance of ResultBase or of an instance of a
        subclass of ResultBase.
        """
        return ResultDymolaTextual(self.file_name if not self._is_stream else self._file)

class ResultHandlerDummy(ResultHandler):
    def get_result(self):
        return None

class JIOError(Exception):
    """
    Base class for exceptions specific to this module.
    """

    def __init__(self, message):
        """
        Create new error with a specific message.

        Parameters::

            message --
                The error message.
        """
        self.message = message

    def __str__(self):
        """
        Print error message when class instance is printed.

        Overrides the general-purpose special method such that a string
        representation of an instance of this class will be the error message.
        """
        return self.message


class VariableNotFoundError(JIOError):
    """
    Exception that is thrown when a variable is not found in a data file.
    """
    pass

class VariableNotTimeVarying(JIOError):
    """
    Exception that is thrown when a column is asked for a parameter/constant.
    """
    pass

class ResultSizeError(JIOError):
    """
    Exception that is raised when a set maximum result size is exceeded.
    """

def robust_float(value):
    """
    Function for robust handling of float values such as INF and NAN.
    """
    try:
        return float(value)
    except ValueError:
        if value.startswith("1.#INF"):
            return float(np.inf)
        elif value.startswith("-1.#INF"):
            return float(-np.inf)
        elif value.startswith("1.#QNAN") or value.startswith("-1.#IND"):
            return float(np.nan)
        else:
            raise ValueError

mat4_template = {'header': [('type', 'i4'),
                            ('mrows', 'i4'),
                            ('ncols', 'i4'),
                            ('imagf', 'i4'),
                            ('namlen', 'i4')]}

class ResultHandlerBinaryFile(ResultHandler):
    """
    Export an optimization or simulation result to file in Dymola's binary result file
    format (MATLAB v4 format).

    All solution points are stored as float64s, this may lead to precision loss 
    for very large values of (U)Int64 variables in FMI3.
    Result storage of strings not supported.
    """
    def __init__(self, model):
        super().__init__(model)
        self.supports['dynamic_diagnostics'] = True
        self.supports['result_max_size']     = True
        self.data_2_header_end_position = 0
        self._size_point = -1
        self._first_point = True

    def _data_header(self, name, nbr_rows, nbr_cols, data_type):
        if data_type == "int":
            t = 10*2
        elif data_type == "double":
            t = 10*0
        elif data_type == "char":
            t = 10*5 + 1
        header = np.empty((), mat4_template["header"])
        header["type"] = (not SYS_LITTLE_ENDIAN) * 1000 + t
        header["mrows"] = nbr_rows
        header["ncols"] = nbr_cols
        header["imagf"] = 0
        header["namlen"] = len(name) + 1

        return header

    def __write_header(self, header, name):
        """
        Dumps the header and name to file.
        """
        self._file.write(header.tobytes(order="F"))
        self._file.write(name.encode() + b"\0")

    def _write_header(self, name, nbr_rows, nbr_cols, data_type):
        """
        Computes the header as well as dumps the header to file.
        """
        header = self._data_header(name, nbr_rows, nbr_cols, data_type)

        self.__write_header(header, name)

    def convert_char_array(self, data):
        data = np.array(data)
        dtype = data.dtype
        dims = [data.shape[0], int(data.dtype.str[2:])]

        data = np.ndarray(shape=(dims[0], int(dtype.str[2:])), dtype=dtype.str[:2]+"1", buffer=data)
        data[data == ""] = " "

        if dtype.kind == "U":
            tmp = np.ndarray(shape=(), dtype=(dtype.str[:2] + str(np.prod(dims))), buffer=data)
            buf = tmp.item().encode('latin-1')
            data = np.ndarray(shape=dims, dtype="S1", buffer=buf)

        return data

    def dump_data(self, data):
        self._file.write(data.tobytes(order="F"))

    def dump_native_data(self, data):
        self._file.write(data)

    def initialize_complete(self):
        pass

    def simulation_start(self, diagnostics_params={}, diagnostics_vars={}):
        """
        Opens the file and writes the header. This includes the information
        about the variables and a table determining the link between variables
        and data.
        This function also takes two keyword arguments 'diagnostics_params'
        and 'diagnostics_vars' which are dicts containing information about what
        diagnostic parameters and variables to generate results for.
        """
        opts = self.options

        #Internal values
        self.file_open = False
        self._is_stream = False
        self.nbr_points = 0
        self.nbr_diag_points = 0
        self.nof_diag_vars = len(diagnostics_vars)
        try:
            # If we check for both we enable testing of mock-ups that utilize the option dynamic_diagnostics
            # since logging will always be set to True if dynamic_diagnostics is True when
            # invoked via normal simulation 'sequencing'
            self._with_diagnostics = opts["logging"] or opts["dynamic_diagnostics"]
        except Exception:
            self._with_diagnostics = False

        if self._with_diagnostics and (len(diagnostics_params) < 1 or self.nof_diag_vars < 1):
            msg = "Unable to start simulation. The following keyword argument(s) are empty:"
            if len(diagnostics_params) < 1:
                msg += " 'diagnostics_params'"
                if self.nof_diag_vars < 1:
                    msg += " and"
            if self.nof_diag_vars < 1:
                msg += " 'diagnostics_vars'."
            raise FMUException(msg)

        self.file_name = opts["result_file_name"]
        try:
            self.parameters = opts["sensitivities"]
        except KeyError:
            self.parameters = False

        if self.parameters:
            raise FMUException("Storing sensitivity results are not supported using this format. Use the file format instead.")

        if self.file_name == "":
            self.file_name=self.model.get_identifier() + '_result.mat'
        self.model._result_file = self.file_name

        # Open file
        file_name = self.file_name
        if isinstance(self.file_name, str):
            self._file = open(file_name,'wb')
        else:
            if not (hasattr(self.file_name, 'write') and hasattr(self.file_name, 'seek') and (hasattr(self.file_name, 'tell'))):
                raise FMUException("Failed to write the result file. Option 'result_file_name' needs to be a filename or a class that supports 'write', 'tell' and 'seek'.")
            self._file = self.file_name #assume it is a stream
            self._is_stream = True
        self.file_open = True


        aclass_data = ["Atrajectory", "1.1", " ", "binTrans"]
        aclass_data = self.convert_char_array(aclass_data)
        self._write_header("Aclass", aclass_data.shape[0], aclass_data.shape[1], "char")
        self.dump_data(aclass_data)

        sorted_vars = self._get_sorted_vars()
        len_name_items = len(sorted_vars)+len(diagnostics_params)+len(diagnostics_vars)+1
        len_desc_items = len_name_items

        if opts["result_store_variable_description"]:
            var_desc = [(name, diagnostics_vars[name][1]) for name in diagnostics_vars]
            param_desc =  [(name, diagnostics_params[name][1]) for name in diagnostics_params]
            len_name_data, name_data, len_desc_data, desc_data = fmi_util.convert_sorted_vars_name_desc(
                sorted_vars,
                param_desc,
                var_desc
            )
        else:
            len_name_data, name_data = fmi_util.convert_sorted_vars_name(
                sorted_vars,
                list(diagnostics_params.keys()),
                list(diagnostics_vars.keys())
            )
            len_desc_data = 1
            desc_data = encode(" "*len_desc_items)

        self._write_header("name", len_name_data, len_name_items, "char")
        self.dump_native_data(name_data)

        self._write_header("description", len_desc_data, len_desc_items, "char")
        self.dump_native_data(desc_data)

        #Create the data info structure (and return parameters)
        data_info = np.zeros((4, len_name_items), dtype=np.int32)

        if self.is_fmi3:
            parameter_data, value_references = fmi_util.prepare_data_info_fmi3(
                data_info,
                sorted_vars,
                [val[0] for val in diagnostics_params.values()],
                self.nof_diag_vars,
                self.model
            )
        else:
            [parameter_data, sorted_vars_real_vref, sorted_vars_int_vref, sorted_vars_bool_vref]  = fmi_util.prepare_data_info(
                data_info,
                sorted_vars,
                [val[0] for val in diagnostics_params.values()],
                self.nof_diag_vars,
                self.model
            )

        self._write_header("dataInfo", data_info.shape[0], data_info.shape[1], "int")
        self.dump_data(data_info)

        #Dump parameters to file
        self._write_header("data_1", len(parameter_data), 2, "double")
        self.dump_data(parameter_data)

        #Record the position so that we can later modify the end time
        self.data_1_header_position = self._file.tell()
        self.dump_data(parameter_data)

        self.data_3_header_position = self._file.tell()
        if self._with_diagnostics:
            self._nof_diag_vars = self.nof_diag_vars + 1
            self._data_3_header = self._data_header("data_3", self.nbr_diag_points, 0, "double")
            self.__write_header(self._data_3_header, "data_3")
            self._data_4_header = self._data_header("data_4", self._nof_diag_vars, 0, "double")
            self.__write_header(self._data_4_header, "data_4")

        self.nbr_points = 0

        #Record the position so that we can later modify the number of result points stored
        self.data_2_header_position = self._file.tell()

        if self.is_fmi3:
            self._len_vars_ref = sum(len(v) for v in value_references.values()) + 1
        else:
            self._len_vars_ref =  len(sorted_vars_real_vref)+len(sorted_vars_int_vref)+len(sorted_vars_bool_vref) + 1

        self._data_2_header = self._data_header("data_2", self._len_vars_ref, 1, "double")
        self._data_2_header["ncols"] = self.nbr_points

        self.__write_header(self._data_2_header, "data_2")
        self.data_2_header_end_position = self._file.tell()

        if self.is_fmi3:
            self.dump_data_internal = fmi_util.DumpDataFMI3(self.model, self._file, value_references, self._with_diagnostics)
        else:
            real_var_ref = np.array(sorted_vars_real_vref)
            int_var_ref  = np.array(sorted_vars_int_vref)
            bool_var_ref = np.array(sorted_vars_bool_vref)
            self.dump_data_internal = fmi_util.DumpData(self.model, self._file, real_var_ref, int_var_ref, bool_var_ref, self._with_diagnostics)

        if self._with_diagnostics:
            diag_data = np.array([val[0] for val in diagnostics_vars.values()], dtype=float)
            self.diagnostics_point(diag_data)

    def _get_sorted_vars(self):
        """ Returns a list of all model variable value references for those data types that are
            written to the results, sorted in increasing order, and ordered by data type.
        """
        sorted_vars = []
        if self.is_fmi3:
            data_types = [
                FMI3_Type.FLOAT64,
                FMI3_Type.FLOAT32,
                FMI3_Type.INT64,
                FMI3_Type.INT32,
                FMI3_Type.INT16,
                FMI3_Type.INT8,
                FMI3_Type.UINT64,
                FMI3_Type.UINT32,
                FMI3_Type.UINT16,
                FMI3_Type.UINT8,
                FMI3_Type.BOOL,
                FMI3_Type.ENUM
            ]
        else:
            data_types = [
                fmi.FMI_REAL,
                fmi.FMI_INTEGER,
                fmi.FMI_BOOLEAN,
                fmi.FMI_ENUMERATION
            ]

        for data_type in data_types:
            sorted_vars += sorted(
                list(self.model.get_model_variables(type=data_type, filter=self.options["filter"]).values()),
                key=attrgetter("value_reference")
            )

        return sorted_vars

    def integration_point(self, solver = None):
        """
        Writes the current status of the model to file.
        """
        if self._size_point == -1:
            pos = self._file.tell()
        self.dump_data_internal.save_point()
        if self._size_point == -1:
            self._size_point = self._file.tell() - pos

        #Increment number of points
        self.nbr_points += 1

        #Make sure that file is always consistent
        self._make_consistent()

    def diagnostics_point(self, diag_data):
        """ Generates a data point for diagnostics data by invoking the util function save_diagnostics_point. """
        self.dump_data_internal.save_diagnostics_point(diag_data)
        self.nbr_diag_points += 1
        self._make_consistent(diag=True)

    def _make_consistent(self, diag=False):
        """
        This method makes sure that the result file is always consistent, meaning that it is
        always possible to load the result file in the result class. The method makes the
        result file consistent by going back in the result file and updates the final time
        as well as the number of result points in the file in specific locations of the
        result file. In the end, it puts the file pointer back to the end of the file (which
        allows further writing of new result points)
        """
        f = self._file

        #Get current position
        file_pos = f.tell()

        f.seek(self.data_1_header_position)
        t = np.array([float(self.model.time)])
        self.dump_data(t)


        if not diag:
            f.seek(self.data_2_header_position)
            self._data_2_header["ncols"] = self.nbr_points
            self.__write_header(self._data_2_header, "data_2")
        else:
            f.seek(self.data_3_header_position)
            self._data_3_header["mrows"] = self.nbr_diag_points
            self.__write_header(self._data_3_header, "data_3")

        #Reset file pointer
        f.seek(file_pos)

        max_size = self.options.get("result_max_size", None)
        if max_size is not None:
            verify_result_size(self.file_name, self._first_point, file_pos, file_pos-self._size_point, max_size, self.options["ncp"], self.model.time)
            #We can go in here before we've stored a full result point (due to storing diagnostic points). So check that a point has been fully stored
            if self._first_point and self._size_point > 0:
                self._first_point = False

    def simulation_end(self):
        """
        Finalize the writing by filling in the blanks in the created file. The
        blanks consists of the number of points and the final time (in data set
        1). Also closes the file.
        """
        #If open, finalize and close
        f = self._file

        if f:
            if not self._is_stream:
                f.close()
            self._file = None

    def get_result(self):
        """
        Method for retrieving the result. This method should return a
        result of an instance of ResultBase or of an instance of a
        subclass of ResultBase.
        """
        return ResultDymolaBinary(self.file_name)

def verify_result_size(file_name, first_point, current_size, previous_size, max_size, ncp, time):
    free_space = get_available_disk_space(file_name)

    if first_point:
        point_size = current_size - previous_size
        estimate = ncp*point_size + previous_size

        msg = ""
        if estimate > max_size:
            msg = msg + "The result is estimated to exceed the allowed maximum size (limit: %g GB, estimate: %g GB). "%(max_size/GB_FACTOR, estimate/GB_FACTOR)

        if estimate > free_space:
            msg = msg + "The result is estimated to exceed the available disk space (available: %g GB, estimate: %g GB). "%(free_space/GB_FACTOR, estimate/GB_FACTOR)

        if msg != "":
            if ncp > NCP_LARGE:
                msg = msg + "The number of result points is large (%d), consider reducing the number of points. "%ncp
            raise ResultSizeError(msg + "To change the maximum allowed result file size, please use the option 'result_max_size'")

    if current_size > max_size:
        raise ResultSizeError("Maximum size of the result reached (limit: %g GB) at time t=%g. "
                            "To change the maximum allowed result size, please use the option "
                            "'result_max_size' or consider reducing the number of communication "
                            "points alternatively the number of variables to store result for."%(max_size/GB_FACTOR, time))

    if free_space <= 0:
        raise ResultSizeError("Not enough disk space to continue to save the result at time t=%g."%time)

def get_result_handler(model, opts):
    result_handler = None

    if isinstance(model, FMUModelBase3) and (opts['result_handling'] not in ["binary", "custom", None]):
        raise NotImplementedError(
            f"For FMI3: 'result_handling' set to '{opts['result_handling']}' is not supported. " + \
            "Consider setting this option to 'binary', 'custom' or None to continue.")

    if opts["result_handling"] == "file":
        result_handler = ResultHandlerFile(model)
    elif opts["result_handling"] == "binary":
        if "sensitivities" in opts and opts["sensitivities"]:
            logging_module.warning('The binary result file do not currently support storing of sensitivity results. Switching to textual result format.')
            result_handler = ResultHandlerFile(model)
        else:
            result_handler = ResultHandlerBinaryFile(model)
    elif opts["result_handling"] == "memory":
        result_handler = ResultHandlerMemory(model)
    elif opts["result_handling"] == "csv":
        result_handler = ResultHandlerCSV(model, delimiter=",")
    elif opts["result_handling"] == "custom":
        result_handler = opts["result_handler"]
        if result_handler is None:
            raise FMUException("The result handler needs to be specified when using a custom result handling.")
        if not isinstance(result_handler, ResultHandler):
            raise FMUException("The result handler needs to be a subclass of ResultHandler.")
    elif opts["result_handling"] is None:
        result_handler = ResultHandlerDummy(model)
    else:
        raise FMUException("Unknown option to result_handling.")

    if (opts.get("result_max_size", 0) > 0) and not result_handler.supports.get("result_max_size", False):
        logging_module.warning("The chosen result handler does not support limiting the result size. Ignoring option 'result_max_size'.")

    return result_handler
