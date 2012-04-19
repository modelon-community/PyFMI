#!/usr/bin/env python 
# -*- coding: utf-8 -*-

#    Copyright (C) 2009 Modelon AB
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, version 3 of the License.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Module containing base classes and functionality.
"""

import zipfile
import tempfile
import platform as PL
import os
import sys
import shutil

import numpy as N
import numpy.ctypeslib as Nct

# location for temporary JModelica files
def get_temp_location():
    if sys.platform == 'win32':
        return os.path.join(tempfile._get_default_tempdir(),'JModelica.org')
    elif sys.platform == 'darwin':
        return os.path.join(tempfile._get_default_tempdir(),'JModelica.org')
    else:
        return os.path.join(tempfile._get_default_tempdir(),os.environ['USER'],'JModelica.org')

tmp_location = get_temp_location()

class BaseModel(object):
    """ 
    Abstract Model class containing base functionality.
    """
    
    def __init__(self):
        raise Exception("This is an abstract class it can not be instantiated.")
    
    def optimize(self):
        raise NotImplementedError('This method is not available in BaseModel.')

    def optimize_options(self, algorithm):
        raise NotImplementedError('This method is not available in BaseModel.')
        
    def simulate(self):
        raise NotImplementedError('This method is not available in BaseModel.')

    def simulate_options(self, algorithm):
        raise NotImplementedError('This method is not available in BaseModel.')
    
    def initialize(self):
        raise NotImplementedError('This method is not available in BaseModel.')
        
    def initialize_options(self, algorithm):
        raise NotImplementedError('This method is not available in BaseModel.')
    
    def set_real(self, valueref, value):
        raise NotImplementedError('This method is currently not supported.')
    
    def get_real(self, valueref):
        raise NotImplementedError('This method is currently not supported.')
    
    def set_integer(self, valueref, value):
        raise NotImplementedError('This method is currently not supported.')
    
    def get_integer(self, valueref):
        raise NotImplementedError('This method is currently not supported.')
    
    def set_boolean(self, valueref, value):
        raise NotImplementedError('This method is currently not supported.')
    
    def get_boolean(self, valueref):
        raise NotImplementedError('This method is currently not supported.')
    
    def set_string(self, valueref, value):
        raise NotImplementedError('This method is currently not supported.')
    
    def get_string(self, valueref):
        raise NotImplementedError('This method is currently not supported.')
    
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
        
            (FMU/JMU)Model.set('damper.d', 1.1)
            (FMU/JMU)Model.set(['damper.d','gear.a'], [1.1, 10])
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
            (FMU/JMU)Model.get('damper.d') 
            # Returns a list of the variables
            (FMU/JMU)Model.get(['damper.d','gear.a'])
        """
        if isinstance(variable_name, basestring):
            return self._get(variable_name) #Scalar case
        else:
            ret = []
            for i in xrange(len(variable_name)): #A list of variables
                ret += [self._get(variable_name[i])]
            return ret
    
    def _exec_algorithm(self, module, algorithm, options):
        """ 
        Helper function which performs all steps of an algorithm run which are 
        common to all initialize and optimize algorithms.
        
        Raises:: 
        
            Exception if algorithm is not a subclass of 
            common.algorithm_drivers.AlgorithmBase.
        """
        base_path = 'algorithm_drivers'
        algdrive = __import__(base_path, globals(), locals(), [], -1)
        AlgorithmBase = getattr(algdrive, 'AlgorithmBase')
        
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
        base_path = 'algorithm_drivers'
        algdrive = __import__(base_path, globals(), locals(), [], -1)
        AlgorithmBase = getattr(algdrive, 'AlgorithmBase')
        
        if isinstance(algorithm, basestring):
            module = __import__(module, globals(), locals(), [algorithm], -1)
            algorithm = getattr(module, algorithm)
        
        if not issubclass(algorithm, AlgorithmBase):
            raise Exception(str(algorithm)+
            " must be a subclass of common.algorithm_drivers.AlgorithmBase")

        # initialize algorithm
        alg = algorithm(start_time, final_time, input, self, options)
        # simulate
        alg.solve()
        # get and return result
        return alg.get_result()

        
    def _default_options(self, module, algorithm):
        """ 
        Help method. Gets the options class for the algorithm specified in 
        'algorithm'.
        """
        module = __import__(module, globals(), locals(), [algorithm], -1)
        algorithm = getattr(module, algorithm)
        
        return algorithm.get_default_options()
        

def get_platform_suffix(type = "dynamic_lib"):
    """
    Get the platform dependent suffix based on the file type.
    
    Parameters::
    
        type --
            The file type. Currently only dynamic_lib is possible.
            Default: 'dynamic_lib'
            
    Returns::
    
        The platform specific file suffix depending on type or empty string if 
        no possible match was found.
    """
    #Detect file suffix depending on type
    platform = '' 
    if sys.platform == 'win32':
        if type == 'dynamic_lib':
            return '.dll'
    elif sys.platform == 'darwin':
        if type == 'dynamic_lib':
            return '.dylib'
    else:
        if type == 'dynamic_lib':
            return '.so'
    return ''

def get_platform_dir():
    """
    Get the platform specific name of binaries directory.
    
    Returns::
    
        The name of the binaries directory. Possible values are:
            - win32
            - win64
            - darwin32
            - darwin64
            - linux32
            - linux64
    """
    #Detect platform
    if sys.platform == 'win32':
        platform = 'win'
    elif sys.platform == 'darwin':
        platform = 'darwin'
    else:
        platform = 'linux'
    
    if PL.architecture()[0].startswith('32'):
        platform += '32'
    else:
        platform += '64'
        
    return platform

def rename_to_tmp(filename, path ='.', filetype = 'dynamic_lib'):
    """
    Take a file and give it a random temporary name.
    
    Parameters::
    
        filename --
            Name of file to rename.
            
        path --
            Path to the file to rename. This is also where the renamed file will 
            end up.
            Default: Current directory.
            
        filetype --
            Type of file to rename, used so that platform specific suffixes can 
            be taken into consideration. Currently only dynamic libary is possible.
            Default: 'dynamic_lib'
    """
    tempfilename = tempfile.mktemp(suffix=get_platform_suffix(filetype), dir=path)
    shutil.move(os.path.join(path, filename), tempfilename)
    return tempfilename

def get_files_in_archive(path):
    """
    Get paths to all unit files and directories in archive.
    
    Parameters::
    
        path --
            The path to the archive directory.
            
    Returns::
        
        Dict with path to the file or directory as value or None if not found. 
        Keys are used to access the unit specific files or directories, possible 
        values are:
            - root : Root of archive (same as path)
            - model_desc : XML description of model (required)
            - image : Image file of model icon (optional)
            - documentation_dir : Directory containing the model documentation (optional)
            - sources_dir : Directory containing source files (optional)
            - binaries_dir : Directory containing the binaries (required)
            - resources_dir : Directory containing resources needed by the model (optional)
    """
    
    files =  {'root':path, 'model_desc':None, 'image': None, 
              'documentation_dir': None, 'sources_dir':None, 
              'binaries_dir': None, 'resources_dir':None}
    
    # model description XML file
    filepath = os.path.join(path, 'modelDescription.xml')
    if os.path.exists(filepath):
        files['model_desc'] = filepath
        
    # model image file
    filepath = os.path.join(path, 'model.png')
    if os.path.exists(filepath):
        files['image'] = filepath
    
    # documentation directory
    filepath = os.path.join(path, 'documentation')
    if os.path.exists(filepath):
        files['documentation_dir'] = filepath
        
    # source directory
    filepath = os.path.join(path, 'sources')
    if os.path.exists(filepath):
        files['sources_dir'] = filepath
        
    # binaries directory
    filepath = os.path.join(path, 'binaries', get_platform_dir())
    if os.path.exists(filepath):
        files['binaries_dir'] = filepath
    
    # resource directory
    filepath = os.path.join(path, 'resources')
    if os.path.exists(filepath):
        files['resources_dir'] = filepath
        
    return files
            

def unzip_unit(archive, path='.'):
    """
    Unzip a unit file.
    
    Extracts all files in archive in temporary location as returned by 
    get_temp_location().
    
    Parameters::
    
        archive --
            The archive file name.
            
        path --
            The path to the archive file.
            Default: Current directory.
            
    Returns::
    
        Path to the root of the extracted archive.
    """
    # return arg
    #ret_val = {'model_desc':None, 'model_values':None, 'binary':None}
    
    # unzip whole archive
    try:
        archive = zipfile.ZipFile(os.path.join(path,archive))
    except IOError:
        raise IOError('Could not locate the file: ' + str(archive))

    # create temporary directory
    tmpdir = create_temp_dir()
    
    # extract all into temp_dir
    archive.extractall(path=tmpdir)
    
    return tmpdir


def create_temp_dir():
    """
    Create a temporary directory for extracting an FMU in or similar
    """
    # create JModelica directory for temporary files (if not already created)
    if not os.path.exists(tmp_location):
        os.makedirs(tmp_location)

    # create temporary directory
    tmpdir = tempfile.mkdtemp(prefix='jm_tmp', dir=tmp_location)
    
    return tmpdir

        
def get_unit_name(class_name, unit_type='JMU'):
    """
    Computes the unit name from a class name.
    
    Parameters::
        
        class_name -- 
            The name of the model.
            
        unit_type --
            The unit type. Possible values: JMU, FMU, FMUX.
            Default: 'JMU'
        
    Returns::
    
        The unit name (replaced dots with underscores).
    """
    if unit_type == 'JMU':
        return class_name.replace('.','_')+'.jmu' 
    elif unit_type == 'FMU':
        return class_name.replace('.','_')+'.fmu' 
    elif unit_type == 'FMUX':
        return class_name.replace('.','_')+'.fmux'
    else:
        raise Exception("The unit type %s is unknown" %unit_type)
        
def get_temp_location():
    """
    Get the directory where the temporary files are placed.
    
    Returns::
    
        The location of temporary files.
    """
    return tmp_location

def list_to_string(item_list):
    """
    Helper function that takes a list of items, which are typed to str and 
    returned as a string with the list items separated by platform dependent 
    path separator. For example: 
        (platform = win)
        item_list = [1, 2, 3]
        return value: '1;2;3'
    """
    ret_str = ''
    for l in item_list:
        ret_str =ret_str+str(l)+os.pathsep
    return ret_str

## This is an api comment.
# @param libname Name of library.
# @param path Path to library.
def load_DLL(libname, path):
    """ 
    Loads a model from a DLL file and returns it.
    
    The filepath can be be both with or without file suffixes (as long as 
    standard file suffixes are used, that is).
    
    Example inputs that should work:
      >> lib = loadDLL('model')
      >> lib = loadDLL('model.dll')
      >> lib = loadDLL('model.so')
    . All of the above should work on the JModelica supported platforms.
    However, the first one is recommended as it is the most platform independent 
    syntax.
    
    Parameters::
    
        libname -- 
            Name of the library without prefix.
            
        path -- 
            The relative or absolute path to the library.
    
    See also http://docs.python.org/library/ct.html
    """
    if sys.platform == 'win32':
        # Temporarily add the value of 'path' to system library path in case the dll 
        # is dependent on other dlls. In that case they should be located in 'path'.
        libpath = 'PATH'
        if os.environ.has_key(libpath):
            oldpath = os.environ[libpath]
        else:
            oldpath = None
        
        if oldpath is not None:
            newpath = path + os.pathsep + oldpath
        else:
            newpath = path
        os.environ[libpath] = newpath
    
    # Don't catch this exception since it hides the actual source
    # of the error.
    dll = Nct.load_library(libname, path)
    
    if sys.platform == 'win32':
        # Set back to the old path
        if oldpath is not None:
            os.environ[libpath] = oldpath
        else:
            del os.environ[libpath]
            
    return dll

class Trajectory:
    """
    Base class for representation of trajectories.
    """
    
    def __init__(self, abscissa, ordinate, tol=1e-8):
        """
        Default constructor for creating a tracjectory object.

        Parameters::
        
            abscissa -- 
                One dimensional numpy array containing the n abscissa 
                (independent) values.
            
            ordinate -- 
                Two dimensional n x m numpy matrix containing the ordiate 
                values. The matrix has the same number of rows as the abscissa 
                has elements. The number of columns is equal to the number of
                output variables.
            
            tol --
                Minimum distance between abcissae. If two abscissae are closer
                than the given tolerance, the largest one is moved.
        """
        self._abscissa = abscissa.astype('float')
        self._ordinate = ordinate
        self._n = N.size(abscissa)
        self._x0 = abscissa[0]
        self._xf = abscissa[-1]
        
        if not N.all(N.diff(self.abscissa) >= 0):
            raise Exception("The abscissae must be increasing.")
        
        [double_point_indices] = N.nonzero(N.abs(N.diff(self.abscissa)) <= tol)
        while (len(double_point_indices) > 0):
            for i in double_point_indices:
                 self.abscissa[i+1] = self.abscissa[i+1] + tol
            [double_point_indices] = N.nonzero(
                    N.abs(N.diff(self.abscissa)) <= tol)
    
    def eval(self,x):
        """
        Evaluate the trajectory at a specifed abscissa.

        Parameters::
        
            x -- 
                One dimensional numpy array, or scalar number, containing n 
                abscissa value(s).

        Returns::
        
            Two dimensional n x m matrix containing the ordinate values 
            corresponding to the argument x.
        """
        pass

    def _set_abscissa(self, absscissa):
        self._abscissa[:] = abscissa

    def _get_abscissa(self):
        return self._abscissa

    abscissa = property(_get_abscissa, _set_abscissa, doc=
    """
    Property for accessing the abscissa of the trajectory.
    """)

    def _set_ordinate(self, absscissa):
        self._ordinate[:] = ordinate

    def _get_ordinate(self):
        return self._ordinate

    ordinate = property(_get_ordinate, _set_ordinate, doc=
    """
    Property for accessing the ordinate of the trajectory.
    """)

class TrajectoryLinearInterpolation(Trajectory):

    def eval(self,x):
        """
        Evaluate the trajectory at a specifed abscissa.

        Parameters::
        
            x -- 
                One dimensional numpy array, or scalar number, containing n 
                abscissa value(s).

        Returns::
        
            Two dimensional n x m matrix containing the ordinate values 
            corresponding to the argument x.
        """        
        y = N.zeros([N.size(x),N.size(self.ordinate,1)])
        for i in range(N.size(y,1)):
            y[:,i] = N.interp(x,self.abscissa,self.ordinate[:,i])
        return y
        
class TrajectoryUserFunction(Trajectory):
    
    def __init__(self, func):
        """
        Constructor for creating a user defined trajectory function.
        
        Parameters::
        
            func -- 
                A function which calculates the ordinate values.
        """
        
        self.traj = func
        
    def eval(self, x):
        """
        Evaluate the trajectory at a specifed abscissa.

        Parameters::
        
            x -- 
                One dimensional numpy array, or scalar number, containing a 
                abscissa value.

        Returns::
        
            Two dimensional n x m matrix containing the ordinate values 
            corresponding to the argument x.
        """
        try:
            y = N.array(N.matrix(self.traj(float(x))))
        except TypeError:
            y = N.array(N.matrix(self.traj(x)).transpose())
                                       #In order to guarantee that the
                                       #return values are on the correct
                                       #form. May need to be evaluated
                                       #for speed improvements.
        return y
