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
Module for writing optimization and simulation results to file.
"""
from operator import itemgetter
import array
import codecs

import numpy as N
import scipy.io

import xmlparser
import pyfmi.fmi as fmi

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

    def simulation_start(self):
        """
        This method is called before the simulation has started and 
        after the initialization call for the FMU.
        """
        pass
        
    def initialize_complete(self):
        """ 
        This method is called after the initialization method of the FMU
        has been been called.
        """
        pass
        
    def integration_point(self, solver=None):
        """ 
        This method is called for each time-point for which result are
        to be stored as indicated by the "number of communcation points"
        provided to the simulation method.
        """
        pass
        
    def simulation_end(self):
        """
        This method is called at the end of a simulation.
        """
        pass
    
    def set_options(self, options):
        """
        Options are the options dictionary provided to the simulation
        method.
        """
        pass
        
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
    def get_variable_index(self,name): 
        """
        Retrieve the index in the name vector of a given variable.
        
        Parameters::
        
            name --
                Name of variable.

        Returns::
        
            In integer index.
        """
        #Strip name of spaces, for instace a[2, 1] to a[2,1]
        name = name.replace(" ", "")
        
        try:
            #return self.name.index(name)
            return self.name_lookup[name]
        except KeyError as ex:
        #except ValueError as ex:
            #Variable was not found so check if it was a derivative variable
            #and check if there exists a variable with another naming
            #convention
            if self._check_if_derivative_variable(name):
                try:
                    #First do a simple search for the other naming convention
                    #return self.name.index(self._convert_dx_name(name))
                    return self.name_lookup[self._convert_dx_name(name)]
                #except ValueError as ex:
                except KeyError as ex:
                    return self._exhaustive_search_for_derivatives(name)
            else:
                raise VariableNotFoundError("Cannot find variable " +
                                        name + " in data file.")
            
    
    def _check_if_derivative_variable(self, name):
        """
        Check if a variable is a derivative variable or not.
        """
        if name.startswith("der(") or name.split(".")[-1].startswith("der("):
            return True
        else:
            return False
        
    
    def _exhaustive_search_for_derivatives(self, name):
        """
        Perform an exhaustive search for a derivative variable by 
        first retrieving the underlying state and for each its alias
        check if there exists a derivative variable.
        """
        #Find alias for name
        state = self._find_underlying_state(name)
        index = self.get_variable_index(state)
        
        alias_index = N.where(self.dataInfo[:,1]==self.dataInfo[index,1])[0]
        
        #Loop through all alias
        for ind in alias_index:
            #Get the trial name
            trial_name = self.name[ind]
            
            #Create the derivative name
            der_trial_name = self._create_derivative_from_state(trial_name)
            
            try:
                #return self.name.index(der_trial_name)
                return self.name_lookup[der_trial_name]
            #except ValueError as ex:
            except KeyError as ex:
                try:
                    #return self.name.index(self._convert_dx_name(der_trial_name))
                    return self.name_lookup[self._convert_dx_name(der_trial_name)]
                except KeyError as ex:
                #except ValueError as ex:
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
    def __init__(self, filename):
        
        fid = codecs.open(filename,'r','utf-8')
        
        name = fid.readline().strip().split(";")
        self.name = name
        
        self.data_matrix = {}
        for i,n in enumerate(name):
            self.data_matrix[n] = i
        
        data = []
        while True:
            row = fid.readline().strip().split(";")
            
            if row[-1] == "" or row[-1] == "\n":
                break
            
            data.append([float(d) for d in row])
            
        self.data = N.array(data)
        
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
    
    def write_header():
        """ 
        The header is intended to be used for writing general information about 
        the model. This is intended to be called once.
        """
        pass
        
    def write_point():
        """ 
        This method does the writing of the actual result. 
        """
        pass
        
    def write_finalize():
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
                FMUModel.get_identifier()) concatenated with the string '_result' is 
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
        
        if parameters != None:
            vars = self.model.get_model_variables(type=0,include_alias=False,variability=3)
            for i in self.model.get_state_value_references():
                for j in vars.keys():
                    if i == vars[j].value_reference:
                        cont_vars.append(vars[j].name)
        
        if parameters != None:
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
        
        list_of_continuous_states = N.append(self.model._save_real_variables_val, 
            self.model._save_int_variables_val)
        list_of_continuous_states = N.append(list_of_continuous_states, 
            self.model._save_bool_variables_val).tolist()
        list_of_continuous_states = dict(zip(list_of_continuous_states, 
            xrange(len(list_of_continuous_states))))
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
                    specified by FMUModel.save_time_point()
                    Default: None
        """
        f = self._file
        data_order = self._data_order

        #If data is none, store the current point from the model
        if data==None:
            #Retrieves the time-point
            [r,i,b] = self.model.save_time_point()
            data = N.append(N.append(N.append(self.model.time,r),i),b)

        #Write the point
        str_text = (" %.14E" % data[0])
        for j in xrange(self._nvariables-1):
            str_text = str_text + (" %.14E" % (data[1+data_order[j]]))
        for j in xrange(len(parameter_data)):
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
                Instance of the FMUModel. 
            data -- 
                The simulation data. 
        """             
        self.model = model 
        self.vars = vars 
        self.name = [var.name for var in vars.values()] 
        self.data = {} 
            
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
            except KeyError as ex: 
                raise VariableNotFoundError("Cannot find variable " + 
                                        name + " in data file.") 
                                            
            factor = -1 if var.alias == fmi.FMI_NEGATED_ALIAS else 1 
                
            if var.variability == fmi.FMI_CONSTANT or var.variability == fmi.FMI_PARAMETER: 
                return Trajectory([self.time[0],self.time[-1]],[self.model.get(name),self.model.get(name)]) 
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
        return self.data 

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
                Name of file.
        """
        fid = codecs.open(fname,'r','utf-8')
        
        result  = [];
     
        # Read Aclass section
        nLines = self._find_phrase(fid, 'char Aclass')

        nLines = int(nLines[0])
        Aclass = [fid.readline().strip() for i in range(nLines)]
        #Aclass = []
        #for i in range(0,nLines):
        #    Aclass.append(fid.readline().strip())
        self.Aclass = Aclass

        # Read name section
        nLines = self._find_phrase(fid, 'char name')

        nLines = int(nLines[0])
        name = [fid.readline().strip().replace(" ","") for i in range(nLines)]
        #name = []
        #for i in range(0,nLines):
        #    name.append(fid.readline().strip().replace(" ",""))
        self.name = name
        self.name_lookup = {key:ind for ind,key in enumerate(self.name)}
     
        # Read description section  
        nLines = self._find_phrase(fid, 'char description') 

        nLines = int(nLines[0])
        description = [fid.readline().strip() for i in range(nLines)]
        #description = []
        #for i in range(0,nLines):
        #    description.append(fid.readline().strip())
        self.description = description

        # Read dataInfo section
        nLines = self._find_phrase(fid, 'int dataInfo')

        nCols = nLines[2].partition(')')
        nLines = int(nLines[0])
        nCols = int(nCols[0])
        dataInfo = [map(int,fid.readline().split()[0:nCols]) for i in range(nLines)]
        #dataInfo = []
        #for i in range(0,nLines):
        #    info = fid.readline().split()
        #    dataInfo.append(map(int,info[0:nCols]))
        self.dataInfo = N.array(dataInfo)

        # Find out how many data matrices there are
        nData = max(self.dataInfo[:,0])
        #nData = 0
        #for i in range(0,nLines):
        #    if dataInfo[i][0] > nData:
        #        nData = dataInfo[i][0]
                
        self.data = []
        for i in range(0,nData): 
            l = fid.readline()
            tmp = l.partition(' ')
            while tmp[0]!='float' and tmp[0]!='double' and l!='':
                l = fid.readline()
                tmp = l. partition(' ')
            if l=='':
                raise JIOError('The result does not seem to be of a supported format.')
            tmp = tmp[2].partition('(')
            nLines = tmp[2].partition(',')
            nCols = nLines[2].partition(')')
            nLines = int(nLines[0])
            nCols = int(nCols[0])
            data = []
            for i in range(0,nLines):
                info = []
                while len(info) < nCols and l != '':
                    l = fid.readline()
                    info.extend(l.split())
                try:
                    data.append(map(float,info[0:nCols]))
                except ValueError: #Handle 1.#INF's and such
                    data.append(map(robust_float,info[0:nCols]))
                if len(info) == 0 and i < nLines-1:
                    raise JIOError("Inconsistent number of lines in the result data.")
                del(info)
            self.data.append(N.array(data))
            
        if len(self.data) == 0:
            raise JIOError('Could not find any variable data in the result file.')
            
    def _find_phrase(self,fid, phrase):
        l = fid.readline()
        tmp = l.partition('(')
        while tmp[0]!=phrase and l!='':
            l = fid.readline()
            tmp = l. partition('(')
        if l=='':
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
        if name == 'time':
            varInd = 0
        else:
            varInd  = self.get_variable_index(name)
            
        dataInd = self.dataInfo[varInd][1]
        factor = 1
        if dataInd < 0:
            factor = -1
            dataInd = -dataInd -1
        else:
            dataInd = dataInd - 1
        dataMat = self.dataInfo[varInd][0]-1
        
        if dataMat < 0:
            # Take into account that the 'Time' variable has data matrix index 0
            # and that 'time' is called 'Time' in Dymola results
             dataMat = 1 if len(self.data) > 1 else 0
             
        return Trajectory(
            self.data[dataMat][:,0],factor*self.data[dataMat][:,dataInd])
        
    def is_variable(self, name):
        """
        Returns True if the given name corresponds to a time-varying variable.
        
        Parameters::
        
            name -- 
                Name of the variable/parameter/constant
                
        Returns::
        
            True if the variable is time-varying.
        """
        if name == 'time':
            return True
        varInd  = self.get_variable_index(name)
        dataMat = self.dataInfo[varInd][0]-1
        if dataMat<0:
            dataMat = 0
        
        if dataMat == 0:
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
        varInd  = self.get_variable_index(name)
        dataInd = self.dataInfo[varInd][1]
        if dataInd<0:
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
        if name == 'time':
            return 0

        if not self.is_variable(name):
            raise VariableNotTimeVarying("Variable " +
                                        name + " is not time-varying.")
        varInd  = self.get_variable_index(name)
        dataInd = self.dataInfo[varInd][1]
        factor = 1
        if dataInd<0:
            factor = -1
            dataInd = -dataInd -1
        else:
            dataInd = dataInd - 1
            
        return dataInd
        
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
            for j in range(N.shape(self.data[i])[0]):
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
        n_points = N.size(res.data[1],0)
        time_shift = self.data[1][-1,0]
        self.data[1] = N.vstack((self.data[1],res.data[1]))
        self.data[1][n_points:,0] = self.data[1][n_points:,0] + time_shift 

class ResultDymolaBinary(ResultDymola):
    """ 
    Class representing a simulation or optimization result loaded from a Dymola 
    binary file.
    """

    def __init__(self,fname):
        """
        Load a result file written on Dymola binary format.

        Parameters::
        
            fname --
                Name of file.
        """
        self.raw = scipy.io.loadmat(fname,chars_as_strings=False)
        name = self.raw['name']
        self.name = [
            array.array(
                'u',
                name[:,i].tolist()).tounicode().rstrip().replace(" ","") \
                for i in range(0,name[0,:].size)]
        self.name_lookup = {key:ind for ind,key in enumerate(self.name)}

        self._loaded_description = False
                
    def _get_description(self):
        if self._loaded_description == False:
            description = self.raw['description']
            self._description = [
            array.array(
                'u',
                description[:,i].tolist()).tounicode().rstrip() \
                for i in range(0,description[0,:].size)]
            self._loaded_description = True
        return self._description

    description = property(_get_description, doc = 
    """
    Property for accessing the description vector.
    """)
       
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
            varInd = 0;
        else:
            varInd  = self.get_variable_index(name)
            
        dataInd = self.raw['dataInfo'][1][varInd]
        dataMat = self.raw['dataInfo'][0][varInd]
        factor = 1
        if dataInd<0:
            factor = -1
            dataInd = -dataInd -1
        else:
            dataInd = dataInd - 1
        
        
            
        if dataMat == 0:
            # Take into account that the 'Time' variable has data matrix index 0
            # and that 'time' is called 'Time' in Dymola results
            dataMat = 2 if len(self.raw['data_2'])> 0 else 1
                
        return Trajectory(self.raw['data_%d'%dataMat][0,:],factor*self.raw['data_%d'%dataMat][dataInd,:])
                
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
        varInd  = self.get_variable_index(name)
        dataMat = self.raw['dataInfo'][0][varInd]
        if dataMat<1:
            dataMat = 1
        
        if dataMat == 1:
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
        varInd  = self.get_variable_index(name)
        dataInd = self.raw['dataInfo'][1][varInd]
        if dataInd<0:
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
        if name == 'time':
            return 0
        
        if not self.is_variable(name):
            raise VariableNotTimeVarying("Variable " +
                                        name + " is not time-varying.")
        varInd  = self.get_variable_index(name)
        dataInd = self.raw['dataInfo'][1][varInd]
        factor = 1
        if dataInd<0:
            factor = -1
            dataInd = -dataInd -1
        else:
            dataInd = dataInd - 1
            
        return dataInd
    
    def get_data_matrix(self):
        """
        Returns the result matrix.
                
        Returns::
        
            The result data matrix.
        """
        return self.raw['data_%d'%2]
        


class ResultHandlerMemory(ResultHandler):
    def __init__(self, model):
        self.model = model
        
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
        
        #Retrieves the time-point
        self.time_sol += [model.time]
        self.real_sol += [model.get_real(self.real_var_ref)]
        self.int_sol  += [model.get_integer(self.int_var_ref)]
        self.bool_sol += [model.get_boolean(self.bool_var_ref)]
        
        #Sets the parameters, if any
        if solver and self.options["sensitivities"]:
            self.param_sol += [N.array(solver.interpolate_sensitivity(model.time, 0)).flatten()]
        
    def simulation_end(self):
        """ 
        The finalize method can be used to for instance close the file.
        ANd this method is called after the simulation has completed.
        """
        pass
        
    def get_result(self):
        """
        Method for retrieving the result. This method should return a 
        result of an instance of ResultBase or of an instance of a 
        subclass of ResultBase.
        """
        t = N.array(self.time_sol) 
        r = N.array(self.real_sol) 
        data = N.c_[t,r] 
        
        if len(self.int_sol) > 0 and len(self.int_sol[0]) > 0: 
            i = N.array(self.int_sol) 
            data = N.c_[data,i] 
        if len(self.bool_sol) > 0 and len(self.bool_sol[0]) > 0: 
            b = N.array(self.bool_sol) 
            data = N.c_[data,b] 

        return ResultStorageMemory(self.model, data, [self.real_var_ref,self.int_var_ref,self.bool_var_ref], self.vars)
        
    def set_options(self, options):
        """
        Options are the options dictionary provided to the simulation
        method.
        """
        self.options = options

class ResultHandlerCSV(ResultHandler):
    def __init__(self, model, delimiter=";"):
        self.model = model
        self.delimiter = delimiter
    
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
        f = codecs.open(self.file_name,'w','utf-8')
        self.file_open = True
        
        name_str = "time"
        for name in const_name_real+const_name_int+const_name_bool+cont_name_real+cont_name_int+cont_name_bool:
            name_str += delimiter+name
            
        f.write(name_str+"\n")
        
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
            
        #for val in N.append(const_val_real,N.append(const_val_int,const_val_boolean)):
        #    const_str += "%.14E"%val+delimiter
        self.const_str = const_str
        
        self._file = f
        
        self.cont_valref_real = cont_valref_real
        self.cont_alias_real  = N.array(cont_alias_real)
        self.cont_valref_int  = cont_valref_int
        self.cont_alias_int  = N.array(cont_alias_int)
        self.cont_valref_bool = cont_valref_bool
        self.cont_alias_bool  = N.array(cont_alias_bool)
        
    def integration_point(self, solver = None):
        """ 
        Writes the current status of the model to file. If the header has not 
        been written previously it is written now. If data is specified it is 
        written instead of the current status.
        
        Parameters::
            
                data --
                    A one dimensional array of variable trajectory data. data 
                    should consist of information about the status in the order 
                    specified by FMUModel.save_time_point()
                    Default: None
        """
        f = self._file
        model = self.model

        #Retrieves the time-point
        t = model.time
        r = model.get_real(self.cont_valref_real)*self.cont_alias_real
        i = model.get_integer(self.cont_valref_int)*self.cont_alias_int
        b = model.get_boolean(self.cont_valref_bool)*self.cont_alias_bool
        
        data = N.append(N.append(r,i),b)
        
        cont_str = ""
        for val in data:
            cont_str += "%.14E;"%val
            
        f.write("%.14E;"%t)
        f.write(self.const_str)
        f.write(cont_str[:-1]+"\n")
        

    def simulation_end(self):
        """ 
        Finalize the writing by filling in the blanks in the created file. The 
        blanks consists of the number of points and the final time (in data set 
        1). Also closes the file.
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
        return ResultCSVTextual(self.file_name)
        
    def set_options(self, options):
        """
        Options are the options dictionary provided to the simulation
        method.
        """
        self.options = options

class ResultHandlerFile(ResultHandler):
    """ 
    Export an optimization or simulation result to file in Dymola's result file 
    format.
    """
    def __init__(self, model):
        self.model = model
    
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
        
        self.file_name = opts["result_file_name"]
        try:
            self.parameters = opts["sensitivities"]
        except KeyError:
            self.parameters = None
        
        if self.file_name == "":
            self.file_name=self.model.get_identifier() + '_result.txt'
            
        #Store the continuous and discrete variables for result writing
        self.real_var_ref, self.int_var_ref, self.bool_var_ref = model.get_model_time_varying_value_references(filter=opts["filter"])
        
        file_name = self.file_name
        parameters = self.parameters
        
        # Open file
        f = codecs.open(file_name,'w','utf-8')
        self.file_open = True
        
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
        
        if parameters != None:
            vars = self.model.get_model_variables(type=0,include_alias=False,variability=3,filter=self.options["filter"])
            for i in self.model.get_state_value_references():
                for j in vars.keys():
                    if i == vars[j].value_reference:
                        cont_vars.append(vars[j].name)
        
        if parameters != None:
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
                    f.write('1 %d 0 -1 # ' % cnt_1 + name[1]+'\n')
                    #datatable1 = True
                else:
                    #cnt_2 += 1
                    #valueref_of_continuous_states.append(
                    #    list_of_continuous_states[name[0]])
                    f.write('2 %d 0 -1 # ' % cnt_2 + name[1] +'\n')
                    #datatable1 = False
                
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
        for i, dtype in enumerate(list_of_parameters):
            vref = dtype[0]
            if dtype[1] == fmi.FMI_REAL:
                str_text = str_text + (
                    " %.14E" % (self.model.get_real([vref])))
            elif dtype[1] == fmi.FMI_INTEGER or dtype[1] == fmi.FMI_ENUMERATION:
                str_text = str_text + (
                    " %.14E" % (self.model.get_integer([vref])))
            elif dtype[1] == fmi.FMI_BOOLEAN:
                str_text = str_text + (
                    " %.14E" % (float(
                        self.model.get_boolean([vref])[0])))
                        
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

    def integration_point(self, solver = None):#parameter_data=[]):
        """ 
        Writes the current status of the model to file. If the header has not 
        been written previously it is written now. If data is specified it is 
        written instead of the current status.
        
        Parameters::
            
                data --
                    A one dimensional array of variable trajectory data. data 
                    should consist of information about the status in the order 
                    specified by FMUModel.save_time_point()
                    Default: None
        """
        f = self._file
        data_order = self._data_order
        model = self.model

        #Retrieves the time-point
        r = model.get_real(self.real_var_ref)
        i = model.get_integer(self.int_var_ref)
        b = model.get_boolean(self.bool_var_ref)
        
        data = N.append(N.append(r,i),b)

        #Write the point
        str_text = (" %.14E" % self.model.time) + ''.join([" %.14E" % (data[data_order[j]]) for j in range(self._nvariables-1)])
        
        #Sets the parameters, if any
        if solver and self.options["sensitivities"]:
            parameter_data = N.array(solver.interpolate_sensitivity(model.time, 0)).flatten()
            for j in xrange(len(parameter_data)):
                str_text = str_text + (" %.14E" % (parameter_data[j]))
                    
        f.write(str_text+'\n')
        
        #Update number of points
        self.nbr_points+=1

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
            #f.write('%d'%self._npoints)
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
        return ResultDymolaTextual(self.file_name)
        
    def set_options(self, options):
        """
        Options are the options dictionary provided to the simulation
        method.
        """
        self.options = options
        
class ResultHandlerDummy(ResultHandler):
    def __init__(self, model):
        self.model = model
    
    def get_result(self):
        return None

class ResultWriterDymola_deprecated(ResultWriter):
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
        
    
    def write_header(self, file_name=''):
        """
        Opens the file and writes the header. This includes the information 
        about the variables and a table determining the link between variables 
        and data.
        
        Parameters::
        
            file_name --
                If no file name is given, the name of the model (as defined by 
                FMUModel.get_identifier()) concatenated with the string '_result' is 
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
        
        for var in self.model._md.get_model_variables():
            ftype = var.get_fundamental_type()
            if not isinstance(ftype,xmlparser.String) and \
                not isinstance(ftype,xmlparser.Enumeration):
                    if var.get_alias() == xmlparser.NO_ALIAS:
                        vrefs_noalias.append(var.get_value_reference())
                        names_noalias.append(var.get_name())
                        aliases.append(var.get_alias())
                        descriptions.append(var.get_description())
                        variabilities_noalias.append(var.get_variability())
                        types_noalias.append(
                            xmlparser._translate_fundamental_type(ftype))
                    else:
                        vrefs_alias.append(var.get_value_reference())
                        names_alias.append(var.get_name())
                        aliases_alias.append(var.get_alias())
                        descriptions_alias.append(var.get_description())
                        variabilities_alias.append(var.get_variability())
                        types_alias.append(
                            xmlparser._translate_fundamental_type(ftype))
                        
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

        f.write('char name(%d,%d)\n' % (num_vars+1, max_name_length))
        f.write('time\n')

        for name in names:
            f.write(name[1] +'\n')

        f.write('\n')

        # Write descriptions       
        f.write('char description(%d,%d)\n' % (num_vars + 1, max_desc_length))
        f.write('Time in [s]\n')

        # Loop over all variables, not only those with a description
        for desc in descriptions:
            f.write(desc[1] +'\n')
                
        f.write('\n')

        # Write data meta information
        
        f.write('int dataInfo(%d,%d)\n' % (num_vars + 1, 4))
        f.write('0 1 0 -1 # time\n')
        
        list_of_continuous_states = N.append(self.model._save_cont_valueref[0], 
            self.model._save_cont_valueref[1])
        list_of_continuous_states = N.append(list_of_continuous_states, 
            self.model._save_cont_valueref[2]).tolist()
        list_of_continuous_states = dict(zip(list_of_continuous_states, 
            xrange(len(list_of_continuous_states))))
        valueref_of_continuous_states = []
        
        cnt_1 = 1
        cnt_2 = 1
        n_parameters = 0
        datatable1 = False
        for i, name in enumerate(names):
            if aliases[i][1] == 0: # no alias
                if variabilities[i][1] == xmlparser.PARAMETER or \
                    variabilities[i][1] == xmlparser.CONSTANT:
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

        f.write('\n')

        # Write data
        # Write data set 1
        f.write('float data_1(%d,%d)\n' % (2, n_parameters + 1))
        f.write("%.14E" % self.model.time)
        str_text = ''
        
        # write constants and parameters
        for i, name in enumerate(names_noalias):
            if variabilities_noalias[i][1] == xmlparser.CONSTANT or \
                variabilities_noalias[i][1] == xmlparser.PARAMETER:
                    if types_noalias[i][1] == xmlparser.REAL:
                        str_text = str_text + (
                            " %.14E" % (self.model.get_real([name[0]])))
                    elif types_noalias[i][1] == xmlparser.INTEGER:
                        str_text = str_text + (
                            " %.14E" % (self.model.get_integer([name[0]])))
                    elif types_noalias[i][1] == xmlparser.BOOLEAN:
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
        
        
        f.write('float data_2(')
        self._point_npoints = f.tell()
        f.write(' '*(14+4+14))
        f.write('\n')
        
        #f.write('%s,%d)\n' % (' '*14, self._nvariables))
        
        self._file = f
        self._data_order = valueref_of_continuous_states
        
    def write_point(self, data=None):
        """ 
        Writes the current status of the model to file. If the header has not 
        been written previously it is written now. If data is specified it is 
        written instead of the current status.
        
        Parameters::
            
                data --
                    A one dimensional array of variable trajectory data. data 
                    should consist of information about the status in the order 
                    specified by FMUModel.save_time_point()
                    Default: None
        """
        f = self._file
        data_order = self._data_order

        #If data is none, store the current point from the model
        if data==None:
            #Retrieves the time-point
            [r,i,b] = self.model.save_time_point()
            data = N.append(N.append(N.append(self.model.time,r),i),b)

        #Write the point
        str_text = (" %.14E" % data[0])
        for j in xrange(self._nvariables-1):
            str_text = str_text + (" %.14E" % (data[1+data_order[j]]))
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
            f.write('%d,%d)' % (self._npoints, self._nvariables))
            #f.write('%d'%self._npoints)
            f.seek(-1,2)
            #Close the file
            f.write('\n')
            f.close()
            self._file_open = False
    
    
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
    
def robust_float(value):
    """
    Function for robust handling of float values such as INF and NAN.
    """
    try:
        return float(value)
    except ValueError:
        if value.startswith("1.#INF"):
            return float(N.inf)
        elif value.startswith("-1.#INF"):
            return float(-N.inf)
        elif value.startswith("1.#QNAN") or value.startswith("-1.#IND"):
            return float(N.nan)
        else:
            raise ValueError
